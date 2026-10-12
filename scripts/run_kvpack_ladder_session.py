#!/usr/bin/env python3
"""Orchestrate the kvpack ladder GPU session, one stage at a time.

This script is a pure sequencer + receipt aggregator. It performs NO direct
accelerator work itself: every GPU-touching sub-step is shelled out through
``scripts/accelerator_safe.py --execute``, which is the sole component that
takes the accelerator lease, checks for other active GPU users, and holds
the quiet windows. This orchestrator can therefore be exercised without
accelerator work by default (omit ``--execute``); pass ``--execute`` to run
for real. The planning mode still performs the producer readiness preflight
over SSH unless ``--skip-producer-health-check`` is set, so it is not fully
offline.

Stages (see the campaign design doc for full rationale):
  1. pacing-fix smoke: 8192-depth disaggregated p4 performance-only run.
  2. 130815-depth EEE A/B at 256 output tokens: arm A is an EEE-active
     attribution packet (expected to reproduce the quantized wire
     blackouts; not gated), arm B is the EEE-off packet under the operator
     ruling recorded 2026-08-20 (merit-gated: CV and per-rep link floor;
     cross-rep exactness is enforced inside the qualifier). Paused by default
     after the stage -- the whole
     downstream ladder's headline claim depends on this A/B, so an
     operator must explicitly confirm before the session continues.
  3. E2 quality scoring at 65536 and 131008 depths, native vs kquant. Includes
     re-tokenizing the three frozen E2 corpora to the new depths (CPU-only,
     sha256-verified against the frozen receipt); corpus-too-short rows remain
     explicit skips. The aggregate verdict preserves E2's preregistered
     replicated-and-persistent cross-document rule.
  4. RUNG 1: naive-transfer baseline -- swap the resident producer to the
     pre-streaming image, run the p4 ladder against it, swap back, and
     verify the fixed image is healthy again before ending the stage.
  5. RUNG 3: deep warm-hit probe at 65536 and ~130815 depths.
  6. RUNG 4: one deep delta cell at 65536 (prefix cut 32768).

The default execution order is 1, 2, 3, 5, 6, 4. The producer-swapping
stage 4 is always last among the selected ladder stages and requires the
explicit ``--allow-producer-swap`` operator flag.

Fail-closed discipline: any stage failure (nonzero exit, missing receipt,
gate mismatch) stops the whole session immediately. There are no retries
and no skip-ahead. ``--from-stage``/``--to-stage`` let an operator resume a
session that was stopped, without re-running stages whose completion marker
(``STAGE_DONE`` file) is already present under the stage's own out-dir.

This script reads (but never writes) the receiver's replay ledger before
each remote arm so reruns always start above its durable generation
watermark. It never takes the accelerator lock itself (only
accelerator_safe.py does that). Remote lifecycle changes are limited to the
explicit stage-2 fresh restart, the stage-3 stop/restart scoring windows,
and the receipted stage-4 producer swap.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
ACCELERATOR_SAFE = REPO_ROOT / "scripts" / "accelerator_safe.py"
LOCK_PATH = Path("/tmp/ferrite.gpu.lock")
REMOTE_LOCK_PATH = "/tmp/ferrite.gpu.lock"
PRODUCER_CONTAINER_USER = "1000:1000"
PRODUCER_STOP_GRACE_SECONDS = 10.0
MAX_GENERATION = (1 << 64) - 1

# Operator ruling authorizing --eee off, recorded 2026-08-20 (option 1 of
# pending ruling (d)); qualify_nvfp4_fast.py refuses --eee off without a
# citation of the ledger entry recording it.
EEE_OFF_RULING_CITATION = (
    "goal-parity-ledger-2026-08: EEE link ruling — operator decision (2026-08-20)"
)

# Per that ruling, EEE-off is the enrolled link invariant for the
# disaggregated lane: every qualify cell in this session runs with it,
# EXCEPT stage 2's arm A, which stays EEE-active on purpose (attribution).
EEE_OFF_ARGS = ["--eee", "off", "--eee-off-ruling", EEE_OFF_RULING_CITATION]

# Full-length identifiers confirmed verbatim in the campaign design; every
# other model/hash/path input is required explicitly from the operator
# rather than defaulted, because the recon only recovered TRUNCATED sha256
# prefixes for the model artifacts (memory's "dc9865ef...", "7e9b74b7...",
# "ca6518d0...", "c49c171f...") -- hardcoding a completion of a truncated
# hash would be guessing, which this harness must never do silently.
DEFAULT_CHECKPOINT_REVISION = "d5109a1d187c27bd1734e81844e71aa4d964e66a"
STAGE3_CHECKPOINT_MAX_MODEL_LEN = 131_072
STAGE3_ENGINE_HEADROOM = 1_024
STAGE3_TEACHER_FORCED_GENERATED_TOKENS = 1
STAGE3_QUALITY_POLICY = (
    "docs/nvfp4-fast-lane-evidence-20260817.md#e-series-preregistration-"
    "yardstick-before-routing"
)

STAGE_NAMES = {
    1: "stage1-smoke-8192",
    2: "stage2-130815-rerun",
    3: "stage3-e2-quality",
    4: "stage4-naive-baseline",
    5: "stage5-warm-hit",
    6: "stage6-delta-65536",
}
DEFAULT_STAGE_ORDER = (1, 2, 3, 5, 6, 4)


class SessionAbort(RuntimeError):
    """Raised to stop the whole session; the message is the operator-facing reason."""


def selected_stage_order(from_stage: int, to_stage: int) -> tuple[int, ...]:
    """Return the selected numeric interval in blast-radius-safe order."""
    return tuple(
        stage for stage in DEFAULT_STAGE_ORDER if from_stage <= stage <= to_stage
    )


def require_stage_authorized(args: argparse.Namespace, stage: int) -> None:
    """Keep the producer swap behind an explicit operator authorization."""
    if stage == 4 and not args.allow_producer_swap:
        raise SessionAbort(
            "stage 4 requires --allow-producer-swap because it stops and replaces "
            "the resident producer"
        )


# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------


def check_lock_not_held() -> None:
    """Refuse to start if another accelerator_safe.py invocation currently
    holds the exclusive flock -- this orchestrator never takes the lock
    itself, it only peeks and releases immediately."""
    if not LOCK_PATH.exists():
        return
    handle = LOCK_PATH.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(handle, fcntl.LOCK_UN)
    except BlockingIOError as error:
        raise SessionAbort(
            f"refusing to start: {LOCK_PATH} is held by another accelerator user"
        ) from error
    finally:
        handle.close()


def replay_high_water(cluster_config: Path) -> int:
    """Read the receiver's durable replay watermark without weakening it.

    The Rust receiver accepts only a private regular non-symlink ledger and
    deserializes an exact ``{"highest_generation": {...}}`` shape. Mirror
    those checks here so generation planning cannot use a different or
    attacker-substituted state file.
    """
    try:
        config = json.loads(cluster_config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SessionAbort(
            f"cannot read replay ledger path from cluster config {cluster_config}: {error}"
        ) from error
    if not isinstance(config, dict):
        raise SessionAbort(f"cluster config {cluster_config} is not a JSON object")
    configured_path = config.get("replay_ledger")
    if not isinstance(configured_path, str) or not configured_path:
        raise SessionAbort(
            f"cluster config {cluster_config} has no non-empty replay_ledger path"
        )
    ledger_path = Path(configured_path)
    if not ledger_path.is_absolute():
        ledger_path = cluster_config.parent / ledger_path

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(ledger_path, flags)
    except OSError as error:
        raise SessionAbort(f"cannot open replay ledger {ledger_path}: {error}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise SessionAbort(
                f"replay ledger must be a regular non-symlink file: {ledger_path}"
            )
        if metadata.st_mode & 0o077:
            raise SessionAbort(
                f"replay ledger must have mode 0600 or stricter: {ledger_path}"
            )
        with os.fdopen(descriptor, encoding="utf-8") as handle:
            descriptor = -1
            state = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SessionAbort(f"cannot parse replay ledger {ledger_path}: {error}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    if not isinstance(state, dict) or set(state) != {"highest_generation"}:
        raise SessionAbort(
            f"replay ledger {ledger_path} must contain only highest_generation"
        )
    generations = state["highest_generation"]
    if not isinstance(generations, dict):
        raise SessionAbort(
            f"replay ledger {ledger_path} highest_generation must be an object"
        )
    for key, value in generations.items():
        if (
            not isinstance(key, str)
            or isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= MAX_GENERATION
        ):
            raise SessionAbort(
                f"replay ledger {ledger_path} has an invalid generation entry"
            )
    return max(generations.values(), default=0)


def resolve_first_generation(
    args: argparse.Namespace,
    configured: int,
    *,
    handoffs: int,
    cell: str,
) -> int:
    """Choose a positive generation range above the live replay watermark."""
    if configured < 1:
        raise SessionAbort(f"{cell}: configured first generation must be positive")
    if handoffs < 1:
        raise SessionAbort(f"{cell}: handoff count must be positive")
    if not args.execute:
        first = configured
        source = "dry-run configured value"
    else:
        high_water = replay_high_water(args.cluster_config)
        if high_water == MAX_GENERATION:
            raise SessionAbort(
                f"{cell}: replay generation space is exhausted at {high_water}"
            )
        first = max(configured, high_water + 1)
        source = f"replay high-water {high_water}"
    if first > MAX_GENERATION - (handoffs - 1):
        raise SessionAbort(
            f"{cell}: {handoffs} handoffs starting at {first} overflow u64"
        )
    print(
        f"{cell}: generations {first}..{first + handoffs - 1} "
        f"({source})",
        flush=True,
    )
    return first


def check_qualifier_has_metal(args: argparse.Namespace) -> None:
    """Refuse to start if muser-remote-qualify was built without `metal`.

    Without it the binary refuses every remote cell with "requires macOS and
    the metal feature", the receiver never binds, and each producer handoff
    is refused -- which exits the producer fail-closed and cascades into the
    next stage. That failure is silent at the stage level, so it is checked
    once, up front.
    """
    binary = REPO_ROOT / "target" / "release" / "muser-remote-qualify"
    if not binary.exists():
        raise SessionAbort(f"muser-remote-qualify missing at {binary}")
    blob = binary.read_bytes()
    if b"requires macOS and the metal feature" in blob:
        raise SessionAbort(
            f"{binary} was built WITHOUT the metal feature; every remote cell "
            "would be refused. Rebuild: cargo build --release --locked "
            "-p muser-bench --bin muser-remote-qualify --features metal"
        )


def check_producer_healthy(args: argparse.Namespace) -> None:
    """Refuse to start if the resident producer is not healthy.

    Default check: the resident container on the Spark node reports
    docker state "running". Operators can substitute a fuller check (e.g.
    a /healthz probe) via --producer-health-cmd; pass
    --skip-producer-health-check only for a --dry-run-only planning pass.
    """
    if args.skip_producer_health_check:
        print("WARNING: producer health check skipped by operator flag", file=sys.stderr)
        return
    if args.producer_health_cmd:
        command = args.producer_health_cmd
    else:
        if not args.resident_container:
            raise SessionAbort(
                "cannot check producer health: pass --resident-container or "
                "--producer-health-cmd (or --skip-producer-health-check to bypass)"
            )
        # "running" is not "ready": after a fail-closed exit the supervisor
        # restarts the container and it spends ~2-3 minutes loading weights
        # with no socket. A stage that starts in that window dies, exits
        # without a receiver, and takes the producer down again -- the
        # cascade that ate most of 2026-08-21. Wait for the socket.
        command = [
            "ssh",
            args.spark_host,
            f"docker inspect --format '{{{{.State.Status}}}}' {args.resident_container} "
            f"| grep -qx running && test -S {args.node_work_dir}/producer.sock "
            f"&& echo ready || echo not-ready",
        ]
    deadline = time.monotonic() + args.producer_ready_timeout
    while True:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
        status = completed.stdout.strip()
        if completed.returncode == 0 and status.endswith("ready") and status != "not-ready":
            return
        if time.monotonic() > deadline:
            raise SessionAbort(
                f"refusing to start: producer not ready within "
                f"{args.producer_ready_timeout}s (last={status!r}, "
                f"stderr={completed.stderr.strip()!r})"
            )
        print(f"producer not ready ({status!r}); waiting", flush=True)
        time.sleep(15)


def node_state_commands(spark_host: str, node_work_dir: str) -> list[tuple[str, list[str]]]:
    """Build the read-only node diagnostics required after a stage abort."""
    lease_code = (
        "import fcntl,sys\n"
        f'h=open("{REMOTE_LOCK_PATH}","a+")\n'
        "try:\n"
        " fcntl.flock(h.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
        "except BlockingIOError:\n"
        ' print("LEASE HELD")\n'
        " sys.exit(1)\n"
        'print("LEASE FREE")\n'
    )
    return [
        (
            "docker_ps",
            [
                "ssh",
                spark_host,
                "docker ps -a --format '{{.Names}} {{.Status}}'",
            ],
        ),
        (
            "sockets",
            ["ssh", spark_host, f"ls -la {shlex.quote(node_work_dir)}/*.sock"],
        ),
        (
            "lease_probe",
            ["ssh", spark_host, f"python3 -c {shlex.quote(lease_code)}"],
        ),
    ]


def append_node_state_receipt(
    destination: Path,
    *,
    spark_host: str,
    node_work_dir: str,
    context: str,
) -> Path:
    """Append a best-effort node snapshot without masking the original abort."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).isoformat()
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(f"\n=== NODE STATE RECEIPT {stamp} context={context} ===\n")
        for label, command in node_state_commands(spark_host, node_work_dir):
            handle.write(f"[{label}] $ {shlex.join(command)}\n")
            try:
                completed = subprocess.run(
                    command, capture_output=True, text=True, timeout=60
                )
            except (OSError, subprocess.SubprocessError) as error:
                handle.write(f"probe_error={type(error).__name__}: {error}\n")
                continue
            handle.write(f"exit={completed.returncode}\n")
            if completed.stdout:
                handle.write(completed.stdout)
                if not completed.stdout.endswith("\n"):
                    handle.write("\n")
            if completed.stderr:
                handle.write(completed.stderr)
                if not completed.stderr.endswith("\n"):
                    handle.write("\n")
        handle.write("=== END NODE STATE RECEIPT ===\n")
    return destination


# --------------------------------------------------------------------------
# accelerator_safe.py invocation helper
# --------------------------------------------------------------------------


def fresh_qualify_dir(base: Path) -> Path:
    """Return an unused `<base>/qualify` path.

    qualify_nvfp4_fast.py creates its out-dir with exist_ok=False to protect
    evidence, so any rerun of a stage dies on the previous attempt's
    directory. Each attempt gets its own suffix and prior debris stays,
    per the append-only policy.
    """
    candidate = base / "qualify"
    attempt = 1
    while candidate.exists():
        attempt += 1
        candidate = base / f"qualify-r{attempt}"
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def fresh_attempt_file(base: Path) -> Path:
    """Return an unused attempt-numbered sibling without creating it."""
    base.parent.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while True:
        candidate = base.with_name(f"{base.stem}-a{attempt}{base.suffix}")
        if not candidate.exists():
            return candidate
        attempt += 1


def run_leased(
    *,
    identity: str,
    cell: str,
    out_dir: Path,
    command: list[str],
    execute: bool,
    quiet_seconds: int = 10,
) -> dict[str, Any]:
    """Run one command through accelerator_safe.py and return its receipt.

    In dry-run mode (execute=False) this only prints accelerator_safe.py's
    own plan (which never touches the GPU) and returns a synthetic
    "planned" record -- no receipt file is produced, matching
    accelerator_safe.py's own dry-run behavior.
    """
    wrapped = [
        "python3",
        str(ACCELERATOR_SAFE),
        "--identity",
        identity,
        "--cell",
        cell,
        "--out-dir",
        str(out_dir),
        "--quiet-seconds",
        str(quiet_seconds),
    ]
    if execute:
        wrapped.append("--execute")
    wrapped.append("--")
    wrapped.extend(command)

    print(f"\n--- running leased stage: identity={identity} cell={cell} ---")
    print(" ".join(wrapped))
    completed = subprocess.run(wrapped)
    if not execute:
        if completed.returncode != 0:
            raise SessionAbort(
                f"accelerator_safe.py dry-run itself failed for cell {cell!r} "
                f"(exit={completed.returncode}); fix the command before --execute"
            )
        return {"mode": "dry-run", "cell": cell, "identity": identity}

    receipt = latest_receipt(out_dir)
    if receipt is None:
        raise SessionAbort(
            f"cell {cell!r}: accelerator_safe.py produced no result receipt "
            f"under {out_dir}; treating as a failure"
        )
    if receipt.get("exit_status") != 0:
        raise SessionAbort(
            f"cell {cell!r} failed: exit_status={receipt.get('exit_status')} "
            f"(see {receipt.get('command_log')})"
        )
    return receipt


def latest_receipt(out_dir: Path) -> dict[str, Any] | None:
    candidates = sorted(out_dir.glob("*.result.json"))
    if not candidates:
        return None
    latest = candidates[-1]
    return json.loads(latest.read_text(encoding="utf-8"))


def _producer_restart_command(
    args: argparse.Namespace, container: str, extra: list[str] | None = None
) -> list[str]:
    """Producer restarts run ON the node (its docstring: the docker socket
    is local there); the driver always invokes it over ssh."""
    return [
        "ssh",
        args.spark_host,
        "python3",
        args.node_restart_script,
        "--container",
        container,
        "--timeout",
        "600",
    ] + (extra or [])


def _quiesce_remote_producer(
    args: argparse.Namespace,
    container: str,
    *,
    grace_seconds: float = PRODUCER_STOP_GRACE_SECONDS,
) -> dict[str, Any]:
    """Stop a producer and prove its node-side accelerator lease is free."""
    stop_command = ["ssh", args.spark_host, "docker", "stop", container]
    inspect_command = [
        "ssh",
        args.spark_host,
        "docker",
        "inspect",
        "--format",
        "{{.State.Status}}",
        container,
    ]
    lease_code = (
        "import fcntl;"
        f'h=open("{REMOTE_LOCK_PATH}","a+");'
        "fcntl.flock(h.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);"
        'print("LEASE FREE")'
    )
    lease_probe_command = ["ssh", args.spark_host, f"python3 -c '{lease_code}'"]
    holder_hint_command = ["ssh", args.spark_host, "fuser", "-v", REMOTE_LOCK_PATH]
    commands = {
        "stop_command": stop_command,
        "inspect_command": inspect_command,
        "lease_probe_command": lease_probe_command,
        "holder_hint_command": holder_hint_command,
    }

    print("stop producer: " + " ".join(stop_command))
    print("wait for exited: " + " ".join(inspect_command))
    print("prove lease free: " + " ".join(lease_probe_command))
    print("if held, identify holder: " + " ".join(holder_hint_command))
    if not args.execute:
        return {"mode": "dry-run", **commands}

    stopped = subprocess.run(stop_command, capture_output=True, text=True, timeout=60)
    if stopped.returncode != 0:
        raise SessionAbort(
            f"stage 4: failed to stop producer {container}: "
            f"{(stopped.stderr or stopped.stdout).strip()}"
        )

    status_deadline = time.monotonic() + grace_seconds
    last_status = ""
    while True:
        inspected = subprocess.run(
            inspect_command, capture_output=True, text=True, timeout=60
        )
        last_status = inspected.stdout.strip()
        if inspected.returncode == 0 and last_status == "exited":
            break
        if time.monotonic() >= status_deadline:
            raise SessionAbort(
                f"stage 4: producer {container} did not reach exited state within "
                f"{grace_seconds:g}s (last={last_status!r}, "
                f"stderr={inspected.stderr.strip()!r})"
            )
        time.sleep(1)

    # Container shutdown and flock release are separate state transitions;
    # give each the full advertised grace rather than sharing one deadline.
    lease_deadline = time.monotonic() + grace_seconds
    while True:
        probe = subprocess.run(
            lease_probe_command, capture_output=True, text=True, timeout=60
        )
        if probe.returncode == 0 and probe.stdout.strip() == "LEASE FREE":
            return {"mode": "execute", "status": "exited", "lease": "free", **commands}
        if time.monotonic() >= lease_deadline:
            break
        time.sleep(1)

    holder = subprocess.run(
        holder_hint_command, capture_output=True, text=True, timeout=60
    )
    holder_hint = "\n".join(
        part.strip() for part in (holder.stdout, holder.stderr) if part.strip()
    )
    if not holder_hint:
        holder_hint = f"fuser exited {holder.returncode} without output"
    raise SessionAbort(
        f"stage 4: accelerator lease {REMOTE_LOCK_PATH} remained held after stopping "
        f"{container}; stop the holding producer before swapping. Holder hint: {holder_hint}"
    )


def _naive_container_binds(args: argparse.Namespace) -> list[str]:
    return [
        f"{REMOTE_LOCK_PATH}:{REMOTE_LOCK_PATH}",
        f"{args.node_checkpoint_dir}:/models/checkpoint:ro",
        f"{args.node_engine_config}:/run/muser/config.json:ro",
        f"{args.node_pki_dir}:/run/muser/pki:ro",
        f"{args.node_work_dir}:/run/muser/work",
        f"{args.node_receipts_dir}:/receipts",
    ]


def _naive_container_argv(args: argparse.Namespace) -> list[str]:
    return [
        "/opt/muser/scripts/gx10/vllm/resident_producer.py",
        "--model",
        "/models/checkpoint",
        "--config",
        "/run/muser/config.json",
        "--sock",
        args.naive_remote_sock,
        "--startup-receipt",
        f"/receipts/{args.naive_startup_receipt}",
        "--lease-file",
        REMOTE_LOCK_PATH,
        "--rope-cache-output",
        f"/run/muser/work/{args.naive_rope_cache}",
        "--max-model-len",
        "131072",
        "--max-num-batched-tokens",
        "131072",
        "--kv-cache-memory-bytes",
        "8589934592",
        "--gpu-memory-utilization",
        "0.82",
    ]


def _naive_container_create_shell(args: argparse.Namespace) -> str:
    """Return the node-side docker create command for the isolated naive arm."""
    command = [
        "docker",
        "create",
        "--name",
        args.naive_container,
        "--gpus",
        "all",
        "--network",
        "host",
        "--ipc",
        "private",
        "--restart",
        "unless-stopped",
        "--user",
        PRODUCER_CONTAINER_USER,
    ]
    for bind in _naive_container_binds(args):
        command.extend(["-v", bind])
    command.extend(["--entrypoint", "python3", args.naive_image])
    command.extend(_naive_container_argv(args))
    return shlex.join(command)


def _ensure_naive_container_command(args: argparse.Namespace) -> list[str]:
    """Create the naive container or replace an inactive one with stale config."""
    container = shlex.quote(args.naive_container)
    socket_path = shlex.quote(args.naive_remote_sock)
    container_user = shlex.quote(PRODUCER_CONTAINER_USER)
    image = shlex.quote(args.naive_image)
    entrypoint = shlex.quote('["python3"]')
    argv = shlex.quote(
        json.dumps(_naive_container_argv(args), separators=(",", ":"))
    )
    binds = shlex.quote(
        json.dumps(_naive_container_binds(args), separators=(",", ":"))
    )
    create = _naive_container_create_shell(args)
    remote_shell = (
        f"if docker inspect {container} >/dev/null 2>&1; then "
        "if docker inspect --format '{{range .Config.Cmd}}{{println .}}{{end}}' "
        f"{container} | grep -Fxq -- {socket_path} && "
        f"docker inspect --format '{{{{.Config.User}}}}' {container} "
        f"| grep -Fxq -- {container_user} && "
        f"docker inspect --format '{{{{.Config.Image}}}}' {container} "
        f"| grep -Fxq -- {image} && "
        f"docker inspect --format '{{{{json .Config.Entrypoint}}}}' {container} "
        f"| grep -Fxq -- {entrypoint} && "
        f"docker inspect --format '{{{{json .Config.Cmd}}}}' {container} "
        f"| grep -Fxq -- {argv} && "
        f"docker inspect --format '{{{{json .HostConfig.Binds}}}}' {container} "
        f"| grep -Fxq -- {binds}; then "
        "echo 'naive container present with exact image, mounts, argv, socket, and user'; "
        f"elif docker inspect --format '{{{{.State.Status}}}}' {container} "
        "| grep -Eq '^(created|exited)$'; then "
        f"docker rm {container} && {create}; "
        "else echo 'refusing to replace an active naive container with stale config' >&2; "
        "exit 1; fi; "
        f"else {create}; fi"
    )
    return ["ssh", args.spark_host, remote_shell]


def _stage4_remote_sock(args: argparse.Namespace, container: str) -> str:
    """Select the socket belonging to the producer used by a stage-4 arm."""
    if container == args.naive_container:
        return args.naive_remote_sock
    if container == args.resident_container:
        return args.remote_sock
    raise SessionAbort(f"stage 4 received an unknown producer container: {container}")


def _stage4_producer_profile_args(
    args: argparse.Namespace, container: str
) -> list[str]:
    """Select the receipt contract for the producer arm being measured."""
    if container == args.naive_container:
        return ["--pre-streaming-control"]
    if container == args.resident_container:
        return []
    raise SessionAbort(f"stage 4 received an unknown producer container: {container}")


def _packet_summary(receipt: dict[str, Any]) -> dict[str, Any] | None:
    """Return the qualifier binary's packet-summary JSON from the command log.

    The qualifier prints one JSON object line carrying the packet verdict
    fields (remote_ttft_cv, installed_payload_gbps, stable, ...); the last
    such line in the log wins. The wrapper-level control.json does NOT carry
    these fields, so the gate must read them here.
    """
    log_path = receipt.get("command_log")
    if not log_path:
        return None
    path = Path(log_path)
    if not path.exists():
        return None
    summary: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not (line.startswith("{") and '"remote_ttft_cv"' in line):
                continue
            try:
                summary = json.loads(line)
            except json.JSONDecodeError:
                continue
    return summary


# --------------------------------------------------------------------------
# Stage completion markers (resume support)
# --------------------------------------------------------------------------


def stage_dir(out_root: Path, stage: int) -> Path:
    return out_root / STAGE_NAMES[stage]


def stage_done_marker(out_root: Path, stage: int) -> Path:
    return stage_dir(out_root, stage) / "STAGE_DONE"


def stage_already_done(out_root: Path, stage: int) -> bool:
    return stage_done_marker(out_root, stage).exists()


def mark_stage_done(out_root: Path, stage: int, summary: dict[str, Any]) -> None:
    directory = stage_dir(out_root, stage)
    directory.mkdir(parents=True, exist_ok=True)
    marker = stage_done_marker(out_root, stage)
    marker.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Stage 1: pacing-fix smoke (8192-depth p4 performance-only)
# --------------------------------------------------------------------------


def stage1(args: argparse.Namespace, out_root: Path) -> dict[str, Any]:
    out_dir = stage_dir(out_root, 1)
    identity = f"{args.identity_prefix}-stage1"
    first_generation = resolve_first_generation(
        args,
        args.stage1_first_generation,
        handoffs=6,
        cell="stage 1 smoke",
    )
    command = [
        "python3",
        str(REPO_ROOT / "scripts" / "qualify_nvfp4_fast.py"),
        "--model",
        str(args.model),
        "--prompt-token-fixture",
        str(args.fixture_8192),
        "--cluster-config",
        str(args.cluster_config),
        "--rope-cache",
        str(args.rope_cache),
        "--out-dir",
        str(fresh_qualify_dir(out_dir)),
        "--identity",
        identity,
        "--resident",
        args.resident_container,
        "--remote-token-fixture",
        args.remote_fixture_8192,
        "--first-generation",
        str(first_generation),
        "--mode",
        "p4",
        "--performance-only",
        "--variant",
        "text",
        "--output-tokens",
        "256",
        "--verify-length",
        "7",
        "--spark-host",
        args.spark_host,
        "--receiver-host",
        args.receiver_host,
        "--receiver-port",
        str(args.receiver_port),
        "--remote-sock",
        args.remote_sock,
    ]
    # Stage 1 deliberately stays EEE-active: it is the harness smoke, and
    # stage 2's arm A must still find the link in the enrolled-active state
    # for the attribution A/B. The first EEE mutation of the session is
    # stage 2 arm B; stages 4/6 then run under the enrolled EEE-off state.
    receipt = run_leased(
        identity=identity,
        cell="smoke-8192-p4",
        out_dir=out_dir,
        command=command,
        execute=args.execute,
    )
    return receipt


# --------------------------------------------------------------------------
# Stage 2: 130815-depth rerun at 256 output tokens
# --------------------------------------------------------------------------


def _stage2_qualify_command(
    args: argparse.Namespace,
    qualify_out: Path,
    identity: str,
    first_generation: int,
    *,
    eee_off: bool,
) -> list[str]:
    command = [
        "python3",
        str(REPO_ROOT / "scripts" / "qualify_nvfp4_fast.py"),
        "--model",
        str(args.model),
        "--prompt-token-fixture",
        str(args.fixture_130815),
        "--cluster-config",
        str(args.cluster_config),
        "--rope-cache",
        str(args.rope_cache),
        "--out-dir",
        str(qualify_out),
        "--identity",
        identity,
        "--resident",
        args.resident_container,
        "--remote-token-fixture",
        args.remote_fixture_130815,
        "--first-generation",
        str(first_generation),
        "--mode",
        "p4",
        "--performance-only",
        "--variant",
        "text",
        "--output-tokens",
        "256",
        "--verify-length",
        "7",
        "--spark-host",
        args.spark_host,
        "--receiver-host",
        args.receiver_host,
        "--receiver-port",
        str(args.receiver_port),
        "--remote-sock",
        args.remote_sock,
    ]
    if eee_off:
        command.extend(["--eee", "off", "--eee-off-ruling", EEE_OFF_RULING_CITATION])
    return command


def stage2_merit_failures(
    control: dict[str, Any], summary: dict[str, Any]
) -> list[str]:
    """Evaluate only the load-bearing deep-packet merit evidence."""
    cv = summary.get("remote_ttft_cv")
    link_min = summary.get("installed_payload_gbps_min")
    rates = summary.get("installed_payload_gbps") or []
    below_floor = [
        rate for rate in rates if isinstance(rate, (int, float)) and rate < 3.0
    ]
    failures: list[str] = []
    if control.get("exit_status") != 0:
        failures.append(f"wrapper exit_status={control.get('exit_status')}")
    if not isinstance(cv, (int, float)) or cv > 0.02:
        failures.append(f"remote_ttft_cv={cv} (gate <= 0.02)")
    if not isinstance(link_min, (int, float)) or link_min < 3.0:
        failures.append(
            f"installed_payload_gbps_min={link_min} "
            f"(gate >= 3.0 on every counted rep; sub-floor reps={below_floor})"
        )
    # remote.rs:1186-1188 returns Err before a summary can be emitted when
    # any repetition changes tokens or logits. Its summary field at line
    # ~1301 is consequently hardcoded true and is not independent evidence.
    return failures


def stage2(args: argparse.Namespace, out_root: Path) -> dict[str, Any]:
    """EEE A/B at the deepest cell, per the operator ruling of 2026-08-20.

    Arm A (attribution): EEE-active packet, expected to reproduce the
    quantized wire blackouts; its packet outcome is recorded but NOT
    fatal -- the stall is the evidence this arm exists to capture.
    Arm B (fix): EEE-off packet under the recorded ledger ruling; the
    session's fail-closed gate applies to its merit legs (CV, per-rep
    link floor, wrapper exit). The binary rejects cross-repetition
    nondeterminism before it can emit a packet summary.
    """
    out_dir = stage_dir(out_root, 2)
    results: dict[str, Any] = {}

    a_dir = out_dir / "eee-active-trace"
    a_identity = f"{args.identity_prefix}-stage2-eee-active"
    existing_a = latest_receipt(a_dir) if a_dir.exists() else None
    if args.execute and existing_a and existing_a.get("exit_status") == 0:
        # Per-arm resume: a completed attribution packet is evidence, not
        # something to burn 50 minutes re-measuring after an arm-B abort.
        print(
            "stage 2: reusing the completed EEE-active attribution arm "
            f"({existing_a.get('command_log')})"
        )
        a_receipt: dict[str, Any] = existing_a
    else:
        a_first_generation = resolve_first_generation(
            args,
            args.stage2_first_generation,
            handoffs=6,
            cell="stage 2 EEE-active attribution",
        )
        a_qualify_dir = fresh_qualify_dir(a_dir)
        if a_qualify_dir.name != "qualify":
            print(
                "stage 2: prior EEE-active attempt debris retained; "
                f"using {a_qualify_dir}"
            )
        a_command = _stage2_qualify_command(
            args,
            a_qualify_dir,
            a_identity,
            a_first_generation,
            eee_off=False,
        )
        try:
            a_receipt = run_leased(
                identity=a_identity,
                cell="130815-eee-active-attribution-p4",
                out_dir=a_dir,
                command=a_command,
                execute=args.execute,
            )
        except SessionAbort as abort:
            if not args.execute:
                raise
            # The attribution arm reproducing the stall (or dying on it) is an
            # expected outcome, not a session failure; record it and continue
            # to the EEE-off arm, which carries the actual gate.
            a_receipt = {"attribution_arm_failed": True, "detail": str(abort)}
    results["eee_active"] = a_receipt
    if args.execute and not a_receipt.get("attribution_arm_failed"):
        results["eee_active_summary"] = _packet_summary(a_receipt)

    if args.execute and args.fresh_producer_before_eee_off:
        # 2026-08-20 21:10Z: the producer exited mid-prefill on the 9th
        # consecutive deep handoff of the evening (arm B rep 3); the
        # recreated container lost the crash logs. Until that stability
        # finding is root-caused, each gated packet starts from a fresh
        # producer so cumulative engine state cannot confound the A/B.
        restart = _producer_restart_command(args, args.resident_container)
        print("\n--- stage 2: fresh producer before the EEE-off packet ---")
        print(" ".join(restart))
        completed = subprocess.run(restart)
        if completed.returncode != 0:
            raise SessionAbort(
                "stage 2: producer restart before the EEE-off arm failed; "
                "node state unknown -- refusing to fire the packet"
            )

    # The qualify wrapper fail-closes on an existing out-dir (evidence
    # protection), so each attempt gets its own dir; prior debris stays.
    b_dir = out_dir / "eee-off"
    attempt = 1
    while (b_dir / "qualify").exists():
        attempt += 1
        b_dir = out_dir / f"eee-off-r{attempt}"
    if attempt > 1:
        print(f"stage 2: prior EEE-off attempt debris retained; running attempt {attempt} in {b_dir}")
    b_identity = f"{args.identity_prefix}-stage2-eee-off"
    b_first_generation = resolve_first_generation(
        args,
        args.stage2_eee_off_first_generation,
        handoffs=6,
        cell="stage 2 EEE-off merit arm",
    )
    b_command = _stage2_qualify_command(
        args,
        b_dir / "qualify",
        b_identity,
        b_first_generation,
        eee_off=True,
    )
    b_receipt = run_leased(
        identity=b_identity,
        cell="130815-eee-off-p4",
        out_dir=b_dir,
        command=b_command,
        execute=args.execute,
    )
    results["eee_off"] = b_receipt

    if not args.execute:
        return results

    control_path = b_dir / "qualify" / "control.json"
    if not control_path.exists():
        raise SessionAbort(
            f"stage 2 (eee-off): control receipt missing at {control_path}; "
            "cannot evaluate the merit gates"
        )
    control = json.loads(control_path.read_text(encoding="utf-8"))
    summary = _packet_summary(b_receipt)
    if summary is None:
        raise SessionAbort(
            "stage 2 (eee-off): qualifier packet summary not found in the "
            "command log; cannot evaluate the merit gates"
        )
    results["eee_off_summary"] = summary

    failures = stage2_merit_failures(control, summary)
    if failures:
        raise SessionAbort(
            "stage 2 (eee-off): the 130815 packet fails its merit gates even "
            "with EEE off: " + "; ".join(failures) + ". The EEE attribution "
            "did not hold, or a new failure mode appeared. Refusing to "
            "continue to stage 3 -- surface this for an operator decision."
        )
    # The packet's composite `stable` flag also contains the <= 4 s
    # TTFT-median leg, structurally unattainable at this depth (pending
    # owner ruling (a) in docs/pending-owner-rulings-20260820.md); the
    # merit gates above are the legs a deep cell can honestly earn.
    results["stable_flag_as_reported"] = summary.get("stable")
    return results


# --------------------------------------------------------------------------
# Stage 3: E2 quality scoring at 65536 and 131008
# --------------------------------------------------------------------------

DOCUMENTS = ("rust", "python", "docs")


def stage3_node_out_dir(
    args: argparse.Namespace, fixture_tokens: int, attempt: int
) -> str:
    """Namespace O_EXCL scorer output by session identity and local attempt."""
    if attempt < 1:
        raise ValueError("stage-3 attempt must be positive")
    readable = "".join(
        character if character.isalnum() or character in "-_." else "-"
        for character in args.identity_prefix
    ).strip("-.")
    readable = readable[:48] or "session"
    digest = hashlib.sha256(args.identity_prefix.encode("utf-8")).hexdigest()[:10]
    namespace = f"{readable}-{digest}-a{attempt}"
    return f"{args.node_results_root}/kvpack-ladder-e2-{fixture_tokens}-{namespace}"


def stage3_native_max_model_len(fixture_tokens: int) -> int:
    """Return a valid vLLM context limit for one stage-3 fixture.

    The scorer generates one bookkeeping token in teacher-forced-only mode.
    Preserve the usual engine headroom at shallower depths, but never request
    more than the pinned checkpoint's declared context limit.
    """
    required_tokens = fixture_tokens + STAGE3_TEACHER_FORCED_GENERATED_TOKENS
    if fixture_tokens < 2 or required_tokens > STAGE3_CHECKPOINT_MAX_MODEL_LEN:
        raise SessionAbort(
            f"stage 3 fixture length {fixture_tokens} plus "
            f"{STAGE3_TEACHER_FORCED_GENERATED_TOKENS} bookkeeping token exceeds "
            f"the checkpoint context limit {STAGE3_CHECKPOINT_MAX_MODEL_LEN}"
        )
    return min(
        fixture_tokens + STAGE3_ENGINE_HEADROOM,
        STAGE3_CHECKPOINT_MAX_MODEL_LEN,
    )


def validate_stage3_yardstick(
    path: Path, length: int, documents: tuple[str, ...]
) -> dict[str, Any]:
    """Require a complete, internally consistent per-cell yardstick receipt.

    A false cell is retained for the preregistered cross-document decision;
    it is not by itself a stage failure.  The E2 policy classifies a
    one-document exceedance as content-local and requires replication at two
    adjacent measured lengths before it can establish a quality blocker.
    """
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SessionAbort(f"stage 3 yardstick receipt is unreadable: {path}") from error
    if (
        not isinstance(report, dict)
        or report.get("schema") != "muser.nvfp4-quant-yardstick.v1"
        or report.get("status") != "measured"
        or not isinstance(report.get("comparisons"), list)
    ):
        raise SessionAbort(f"stage 3 yardstick receipt has an invalid shape: {path}")

    expected_ids = {f"e2-{document}-{length}" for document in documents}
    comparisons: dict[str, dict[str, Any]] = {}
    for comparison in report["comparisons"]:
        if not isinstance(comparison, dict) or not isinstance(comparison.get("id"), str):
            raise SessionAbort(f"stage 3 yardstick comparison is malformed: {path}")
        fixture_id = comparison["id"]
        if fixture_id in comparisons:
            raise SessionAbort(
                f"stage 3 yardstick repeats comparison {fixture_id!r}: {path}"
            )
        comparisons[fixture_id] = comparison
    if set(comparisons) != expected_ids:
        raise SessionAbort(
            f"stage 3 yardstick comparison set differs at depth {length}: "
            f"expected {sorted(expected_ids)}, got {sorted(comparisons)}"
        )

    for fixture_id, comparison in comparisons.items():
        native = comparison.get("native_vs_kquant")
        if (
            comparison.get("token_count") != length
            or comparison.get("regime") != "long-context"
            or not isinstance(native, dict)
            or not isinstance(native.get("top_token_passed"), bool)
            or not isinstance(native.get("perplexity_passed"), bool)
            or not isinstance(native.get("passed"), bool)
        ):
            raise SessionAbort(
                f"stage 3 yardstick comparison {fixture_id!r} has an invalid "
                f"verdict shape: {path}"
            )
        expected_passed = native["top_token_passed"] and native["perplexity_passed"]
        if native["passed"] is not expected_passed:
            raise SessionAbort(
                f"stage 3 yardstick comparison {fixture_id!r} has an inconsistent "
                f"combined verdict: {path}"
            )
    return report


def stage3_quality_verdict(
    reports: dict[int, dict[str, Any]],
    documents: tuple[str, ...],
) -> dict[str, Any]:
    """Apply the frozen E2 replicated-and-persistent content-control rule."""
    lengths = sorted(reports)
    if len(lengths) < 2:
        raise SessionAbort(
            "stage 3 quality verdict requires at least two measured depths"
        )

    summaries: list[dict[str, Any]] = []
    expected_documents = set(documents)
    for length in lengths:
        comparisons = reports[length].get("comparisons")
        if not isinstance(comparisons, list):
            raise SessionAbort(
                f"stage 3 quality report at depth {length} lacks comparisons"
            )
        measured: dict[str, dict[str, Any]] = {}
        for comparison in comparisons:
            fixture_id = comparison["id"]
            prefix = "e2-"
            suffix = f"-{length}"
            if not fixture_id.startswith(prefix) or not fixture_id.endswith(suffix):
                raise SessionAbort(
                    f"stage 3 quality comparison id is invalid at depth {length}: "
                    f"{fixture_id!r}"
                )
            document = fixture_id[len(prefix) : -len(suffix)]
            if document not in expected_documents or document in measured:
                raise SessionAbort(
                    f"stage 3 quality comparison document is invalid at depth "
                    f"{length}: {document!r}"
                )
            measured[document] = comparison
        if len(measured) < 2:
            raise SessionAbort(
                f"stage 3 quality verdict has fewer than two documents at depth {length}"
            )

        failed = sorted(
            document
            for document, comparison in measured.items()
            if comparison["native_vs_kquant"]["passed"] is False
        )
        summaries.append(
            {
                "token_count": length,
                "measured_documents": sorted(measured),
                "missing_documents": sorted(expected_documents - measured.keys()),
                "failed_documents": failed,
                "failed_document_count": len(failed),
                "content_local": len(failed) == 1,
                "replicated_exceedance": len(failed) >= 2,
            }
        )

    first_persistent_index: int | None = None
    for index, summary in enumerate(summaries):
        persistent = (
            index + 1 < len(summaries)
            and summary["replicated_exceedance"]
            and summaries[index + 1]["replicated_exceedance"]
        )
        summary["persistent_at_next_measured_length"] = persistent
        if persistent and first_persistent_index is None:
            first_persistent_index = index

    any_failures = any(row["failed_document_count"] for row in summaries)
    any_replicated = any(row["replicated_exceedance"] for row in summaries)
    if first_persistent_index is not None:
        status = "quality-blocker"
        branch = "replicated-persistent-length-effect"
    elif any_replicated:
        status = "pass"
        branch = "nonpersistent-replicated-exceedance"
    elif any_failures:
        status = "pass"
        branch = "content-sensitive-envelope"
    else:
        status = "pass"
        branch = "inside-yardstick-band"

    full_coverage = [
        row["token_count"]
        for row in summaries
        if not row["missing_documents"]
    ]
    return {
        "schema": "muser.kvpack-ladder-stage3-quality-verdict.v1",
        "status": status,
        "branch": branch,
        "preregistered_method": {
            "policy": STAGE3_QUALITY_POLICY,
            "documents_required": len(documents),
            "length_effect": (
                "at least two documents exceed their unchanged per-cell gates "
                "at one measured length and the next"
            ),
            "single_document_exceedance": "content-local-published-sensitivity",
        },
        "expected_documents": list(documents),
        "deepest_measured_tokens": lengths[-1],
        "deepest_full_coverage_tokens": max(full_coverage) if full_coverage else None,
        "first_persistent_exceedance_tokens": (
            summaries[first_persistent_index]["token_count"]
            if first_persistent_index is not None
            else None
        ),
        "lengths": summaries,
        "seal_eligible": False,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_stage3_quality_verdict(
    path: Path,
    verdict: dict[str, Any],
    yardstick_paths: dict[int, Path],
    fixture_receipt: Path,
) -> None:
    """Commit the aggregate quality decision without overwriting evidence."""
    verdict["inputs"] = {
        "extended_fixture_receipt": {
            "path": str(fixture_receipt),
            "sha256": _sha256(fixture_receipt),
        },
        "yardsticks": [
            {
                "token_count": length,
                "path": str(yardstick_paths[length]),
                "sha256": _sha256(yardstick_paths[length]),
            }
            for length in sorted(yardstick_paths)
        ],
    }
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(verdict, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def stage3(args: argparse.Namespace, out_root: Path) -> dict[str, Any]:
    out_dir = stage_dir(out_root, 3)
    lengths = [65536, 131008]

    # 3a. Extend the frozen E2 fixtures (CPU-only; not wrapped in
    # accelerator_safe.py -- see the module docstring for the rationale,
    # and the recon note that this must be re-verified with --dry-run if
    # `muser-bench tokenize` ever turns out to touch the accelerator).
    # The extender fail-closes on an existing output dir; each attempt gets
    # a fresh one and prior debris stays (same pattern as stage 2's arms).
    extended_dir = out_dir / "e2-extended-fixtures"
    attempt = 1
    while extended_dir.exists():
        attempt += 1
        extended_dir = out_dir / f"e2-extended-fixtures-r{attempt}"
    extend_command = [
        "python3",
        str(REPO_ROOT / "scripts" / "extend_nvfp4_e2_fixtures.py"),
        "--source-dir",
        str(args.e2_source_dir),
        "--receipt",
        str(args.e2_receipt),
        "--bench",
        str(args.bench),
        "--model",
        str(args.model),
        "--lengths",
        *[str(length) for length in lengths],
        "--documents",
        *DOCUMENTS,
        "--output-dir",
        str(extended_dir),
    ]
    print("\n--- stage3a: extend E2 fixtures (CPU-only, unwrapped) ---")
    print(" ".join(extend_command))
    if args.execute:
        completed = subprocess.run(extend_command)
        if completed.returncode != 0:
            raise SessionAbort(
                f"stage 3a failed: extend_nvfp4_e2_fixtures.py exited "
                f"{completed.returncode}"
            )

    receipts: dict[str, Any] = {"extend_command": extend_command}

    quality_reports: dict[int, dict[str, Any]] = {}
    yardstick_paths: dict[int, Path] = {}
    for length in lengths:
        # Depth rows use only the documents whose extended fixture exists --
        # stage 3a records corpus-too-short documents as skipped (the docs
        # corpus is ~109k tokens, so 131008 runs rust+python only).
        if args.execute:
            docs_for_length = tuple(
                document
                for document in DOCUMENTS
                if (extended_dir / f"e2-{document}-{length}.tokens").exists()
            )
            if len(docs_for_length) < 2:
                raise SessionAbort(
                    f"stage 3: fewer than two extended fixtures exist at depth "
                    f"{length} ({docs_for_length}); refusing a one-document row"
                )
            if docs_for_length != DOCUMENTS:
                print(
                    f"stage 3 depth {length}: running documents {docs_for_length} "
                    "(others recorded corpus-too-short by stage 3a)"
                )
        else:
            docs_for_length = DOCUMENTS

        # 3b. Native lane (GPU, on the Spark inside the producer image --
        # the E2 precedent; vLLM does not run on the Mac). The driver
        # builds the per-depth fixture manifest itself (mirroring the
        # frozen e2-manifest.json row shape), stages fixtures plus the
        # scoring script to the node, runs the scorer in a one-shot
        # container, and copies the result back.
        native_out_dir = out_dir / "stage3-native" / str(length)
        native_out_dir.mkdir(parents=True, exist_ok=True)
        # The manifest lives BESIDE the fixtures: token_file entries are
        # relative, and both consumers resolve them against the manifest's
        # own directory (the Mac assembler directly; the node scorer via the
        # staged copy in node_fixture_dir).
        manifest_path = extended_dir / f"e2-manifest-{length}.json"
        max_len = stage3_native_max_model_len(length)
        kv_bytes = 8 << 30
        timeout_seconds = 2400 if length > 100_000 else 1800

        # Schema and row shape are the scorer's contract
        # (score_nvfp4_drift.py load_manifest), not ours to choose.
        manifest = {
            "schema": "muser.nvfp4-drift-fixtures.v1",
            "fixtures": [
                {
                    "context_length": length,
                    "document": document,
                    "id": f"e2-{document}-{length}",
                    "output_tokens": 0,
                    "regime": "long-context",
                    "token_file": f"e2-{document}-{length}.tokens",
                }
                for document in docs_for_length
            ],
        }
        print(f"\n--- stage3b-pre: build depth manifest {manifest_path} ---")
        if args.execute:
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

        node_fixture_dir = f"{args.node_work_root}/kvpack-ladder-e2-{length}"
        node_out_dir = stage3_node_out_dir(args, length, attempt)
        node_qual_dir = f"{args.node_work_root}/kvpack-ladder-qualification"
        staging_commands = [
            ["ssh", args.spark_host, "mkdir", "-p", node_fixture_dir, node_out_dir, node_qual_dir],
            [
                "scp",
                "-q",
                str(REPO_ROOT / "scripts" / "gx10" / "vllm" / "score_nvfp4_drift.py"),
                f"{args.spark_host}:{node_qual_dir}/",
            ],
            ["scp", "-q", str(manifest_path)]
            + [str(extended_dir / f"e2-{document}-{length}.tokens") for document in docs_for_length]
            + [f"{args.spark_host}:{node_fixture_dir}/"],
        ]
        print(f"\n--- stage3b staging (node file placement, unwrapped) ---")
        for staging in staging_commands:
            print(" ".join(staging))
            if args.execute:
                completed = subprocess.run(staging)
                if completed.returncode != 0:
                    raise SessionAbort(
                        f"stage 3b staging failed at depth {length}: {' '.join(staging)}"
                    )

        native_command = [
            "ssh",
            args.spark_host,
            "timeout",
            "--signal=INT",
            "--kill-after=30s",
            f"{timeout_seconds}s",
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "--ipc=host",
            "--user",
            "1000:1000",
            "--entrypoint",
            "python3",
            "-e", "MUSER_NVFP4_EXACT=0",
            "-e", "PYTHONHASHSEED=0",
            "-e", "VLLM_ATTENTION_BACKEND=FLASH_ATTN",
            "-e", "VLLM_ENABLE_V1_MULTIPROCESSING=0",
            "-e", "VLLM_USE_FLASHINFER_SAMPLER=0",
            "-e", "CUBLAS_WORKSPACE_CONFIG=:4096:8",
            "-e", "PYTHONPATH=/qualification:/opt/muser/scripts/gx10/vllm",
            "-v", "/tmp/ferrite.gpu.lock:/tmp/ferrite.gpu.lock",
            "-v", f"{args.node_checkpoint_dir}:/models/checkpoint:ro",
            "-v", f"{args.node_engine_config}:/run/muser/config.json:ro",
            "-v", f"{node_qual_dir}:/qualification:ro",
            "-v", f"{node_fixture_dir}:/fixtures:ro",
            "-v", f"{node_out_dir}:/out",
            args.native_image,
            "/qualification/score_nvfp4_drift.py",
            "--model", "/models/checkpoint",
            "--tokenizer", "/models/checkpoint",
            "--config", "/run/muser/config.json",
            "--checkpoint-revision", args.checkpoint_revision,
            "--checkpoint-artifact-sha256", args.checkpoint_artifact_sha256,
            "--fixture-manifest", f"/fixtures/e2-manifest-{length}.json",
            "--output", f"/out/native-{length}.json",
            # Fresh per attempt: the scorer creates this with mkdir(mode)
            # and a leftover from an aborted run makes it fail closed.
            "--progress-dir", f"/out/progress-{length}-a{attempt}",
            "--lease-file", "/tmp/ferrite.gpu.lock",
            "--max-model-len", str(max_len),
            "--max-num-batched-tokens", str(max_len),
            "--gpu-memory-utilization", "0.82",
            "--kv-cache-memory-bytes", str(kv_bytes),
            "--teacher-forced-only",
        ]
        # The node's accelerator lease is held by the resident producer for
        # its whole lifetime, and the one-shot scorer needs it, so the two
        # cannot coexist. Stop the producer for the scoring window and bring
        # it back through the full restart ritual afterwards.
        # The supervisor watches the container and restarts it within
        # seconds, taking the lease straight back, so it must be paused for
        # the scoring window. SIGSTOP/SIGCONT rather than kill: the
        # supervisor is long-lived node infrastructure we do not own.
        pause = [
            "ssh",
            args.spark_host,
            "pkill -STOP -f '[s]upervise_resident_producer[.]py' || true",
        ]
        stop = ["ssh", args.spark_host, "docker", "stop", args.resident_container]
        print("\n--- stage3b: pause the supervisor and stop the producer ---")
        print(" ".join(pause))
        print(" ".join(stop))
        if args.execute:
            subprocess.run(pause)
            if subprocess.run(stop).returncode != 0:
                subprocess.run(
                    ["ssh", args.spark_host,
                     "pkill -CONT -f '[s]upervise_resident_producer[.]py' || true"]
                )
                raise SessionAbort(
                    f"stage 3: could not stop {args.resident_container} to free "
                    "the node accelerator lease"
                )

        identity = f"{args.identity_prefix}-stage3-native-{length}"
        try:
            native_receipt = build_or_leased(
                args,
                identity=identity,
                cell=f"e2-{length}-native",
                out_dir=native_out_dir,
                command=native_command,
            )
        finally:
            # Always restore the producer, including when scoring failed:
            # leaving the node without one would break every later stage and
            # any other lane sharing the box.
            restart = _producer_restart_command(args, args.resident_container)
            resume = [
                "ssh",
                args.spark_host,
                "pkill -CONT -f '[s]upervise_resident_producer[.]py' || true",
            ]
            print("\n--- stage3b: restore the producer and resume the supervisor ---")
            print(" ".join(restart))
            print(" ".join(resume))
            if args.execute:
                restarted = subprocess.run(restart).returncode
                # Resume the supervisor even if the restart failed: it is the
                # thing that recovers the node when we cannot.
                subprocess.run(resume)
                if restarted != 0:
                    raise SessionAbort(
                        "stage 3: the resident producer did not come back after "
                        "node scoring; fix the node before resuming"
                    )

        fetch = [
            "scp",
            "-q",
            f"{args.spark_host}:{node_out_dir}/native-{length}.json",
            str(native_out_dir / f"native-{length}.json"),
        ]
        print(" ".join(fetch))
        if args.execute:
            completed = subprocess.run(fetch)
            if completed.returncode != 0 or not (native_out_dir / f"native-{length}.json").exists():
                raise SessionAbort(
                    f"stage 3b: native result missing after scoring at depth {length}"
                )
        receipts[f"native-{length}"] = native_receipt

        # 3c/3c.5. Mac reference lanes: the kquant production reference and
        # the Q6 alternate that calibrates a DEPTH-LOCAL yardstick band
        # (E1 design computed at this depth; the frozen <=32k content-
        # control band does not transfer to 65k/131k).
        lane_paths: dict[str, Path] = {}
        for lane_name, lane_model, lane_sha in (
            ("kquant", args.kquant_model, args.kquant_model_sha256),
            ("alternate", args.alternate_model, args.alternate_model_sha256),
        ):
            reference_path, lane_receipts = _stage3_reference_lane(
                args,
                out_dir,
                extended_dir,
                manifest_path,
                length,
                docs_for_length,
                lane_name,
                lane_model,
                lane_sha,
            )
            lane_paths[lane_name] = reference_path
            receipts[f"{lane_name}-{length}"] = lane_receipts

        # 3d. Depth-local quant-vs-quant yardstick verdict (CPU, unwrapped):
        # the band is computed from kquant-vs-alternate AT this depth and
        # native is checked against it. The frozen content-control
        # comparator requires the full 3-doc nested ladder and rejects
        # per-depth rows, so it is not used here.
        compare_command = [
            "python3",
            str(REPO_ROOT / "scripts" / "evaluate_nvfp4_quant_yardstick.py"),
            "--kquant",
            str(lane_paths["kquant"]),
            "--alternate",
            str(lane_paths["alternate"]),
            "--native",
            str(native_out_dir / f"native-{length}.json"),
            "--output",
            str(out_dir / f"stage3-yardstick-{length}.json"),
        ]
        print(f"\n--- stage3d: depth-local yardstick at {length} (CPU-only, unwrapped) ---")
        print(" ".join(compare_command))
        if args.execute:
            completed = subprocess.run(compare_command)
            if completed.returncode != 0:
                raise SessionAbort(
                    f"stage 3d failed at depth {length}: "
                    f"evaluate_nvfp4_quant_yardstick.py exited {completed.returncode}"
                )
            yardstick_path = out_dir / f"stage3-yardstick-{length}.json"
            quality_reports[length] = validate_stage3_yardstick(
                yardstick_path,
                length,
                docs_for_length,
            )
            yardstick_paths[length] = yardstick_path
        receipts[f"yardstick-{length}"] = compare_command

    quality_verdict_path = out_dir / "stage3-quality-verdict.json"
    print(
        "\n--- stage3e: apply frozen replicated-and-persistent quality policy "
        f"({quality_verdict_path}) ---"
    )
    if args.execute:
        verdict = stage3_quality_verdict(quality_reports, DOCUMENTS)
        write_stage3_quality_verdict(
            quality_verdict_path,
            verdict,
            yardstick_paths,
            extended_dir / "extended-receipt.json",
        )
        receipts["quality_verdict"] = {
            "path": str(quality_verdict_path),
            "status": verdict["status"],
            "branch": verdict["branch"],
        }
        print(json.dumps(receipts["quality_verdict"], sort_keys=True))
        if verdict["status"] != "pass":
            raise SessionAbort(
                "stage 3 quality blocker: replicated yardstick exceedance persists "
                f"at the next measured depth (see {quality_verdict_path})"
            )
    else:
        receipts["quality_verdict"] = {"path": str(quality_verdict_path)}

    return receipts


def _stage3_reference_lane(
    args: argparse.Namespace,
    out_dir: Path,
    extended_dir: Path,
    manifest_path: Path,
    length: int,
    docs_for_length: tuple[str, ...],
    lane_name: str,
    lane_model: Path,
    lane_sha: str,
) -> tuple[Path, dict[str, Any]]:
    """Capture + assemble one Mac reference lane (kquant or Q6 alternate)."""
    lane_receipts: dict[str, Any] = {}
    scratch_namespace = hashlib.sha256(
        args.identity_prefix.encode("utf-8")
    ).hexdigest()[:16]
    for document in docs_for_length:
        doc_out = out_dir / f"stage3-{lane_name}" / f"{document}-{length}"
        doc_out.mkdir(parents=True, exist_ok=True)
        # Full-vocabulary uint16 rows are 12 GiB at 65k and 25 GiB at 131k.
        # They are an intermediate cross-binding carrier, not a comparator
        # input after validation. Keep them on the internal disk and retain
        # only capture_llama_perplexity.py's compact authenticated reduction
        # on the append-only evidence volume.
        scratch_logits = (
            args.stage3_scratch_root
            / scratch_namespace
            / lane_name
            / f"{document}-{length}"
            / "logits.bin"
        )
        # The assembler requires the pinned comparator receipt as a
        # llama-receipt.json sibling of each capture.
        if args.execute:
            shutil.copyfile(args.llama_perplexity_receipt, doc_out / "llama-receipt.json")
        identity = f"{args.identity_prefix}-stage3-{lane_name}-{document}-{length}"
        capture_command = [
            "python3",
            str(REPO_ROOT / "scripts" / "capture_llama_perplexity.py"),
            "--model",
            str(lane_model),
            "--expected-model-sha256",
            lane_sha,
            "--token-fixture",
            str(extended_dir / f"e2-{document}-{length}.tokens"),
            "--corpus",
            str(args.e2_source_dir / f"e2-{document}.txt"),
            # Exactly the fixture length: the assembler enforces
            # runtime context == len(tokens) (E2 precedent).
            "--context-length",
            str(length),
            "--batch-size",
            "2048",
            "--ubatch-size",
            "512",
            "--llama-perplexity",
            str(args.llama_perplexity_binary),
            "--llama-receipt",
            str(args.llama_perplexity_receipt),
            "--logits-out",
            str(scratch_logits),
            "--compact-teacher-output",
            str(doc_out / "teacher-evidence.json"),
            "--scratch-root",
            str(args.stage3_scratch_root),
            "--discard-raw-after-compact",
            "--command-log",
            str(doc_out / "capture.command.log"),
            "--output",
            str(doc_out / "perplexity-capture.json"),
            "--identity",
            identity,
        ]
        lane_receipts[document] = build_or_leased(
            args,
            identity=identity,
            cell=f"e2-{document}-{length}-{lane_name}-capture",
            out_dir=doc_out,
            command=capture_command,
        )

    reference_path = out_dir / f"{lane_name}-reference-{length}.json"
    assemble_command = [
        "python3",
        str(REPO_ROOT / "scripts" / "assemble_kquant_content_control_reference.py"),
        "--manifest",
        str(manifest_path),
    ]
    for document in docs_for_length:
        capture_path = (
            out_dir / f"stage3-{lane_name}" / f"{document}-{length}" / "perplexity-capture.json"
        )
        assemble_command.append(f"--capture=e2-{document}-{length}={capture_path}")
    assemble_command += ["--output", str(reference_path)]
    print(
        f"\n--- stage3c.5 [{lane_name}]: assemble reference at depth {length} "
        "(CPU-only, unwrapped) ---"
    )
    print(" ".join(assemble_command))
    if args.execute:
        completed = subprocess.run(assemble_command)
        if completed.returncode != 0:
            raise SessionAbort(
                f"stage 3c.5 [{lane_name}] failed at depth {length}: "
                f"assembler exited {completed.returncode}"
            )
    lane_receipts["assemble_command"] = assemble_command
    return reference_path, lane_receipts


def build_or_leased(
    args: argparse.Namespace,
    *,
    identity: str,
    cell: str,
    out_dir: Path,
    command: list[str],
    note: str | None = None,
) -> dict[str, Any]:
    if note:
        print(note)
    return run_leased(
        identity=identity,
        cell=cell,
        out_dir=out_dir,
        command=command,
        execute=args.execute,
    )


# --------------------------------------------------------------------------
# Stage 4: RUNG 1 naive-transfer baseline
# --------------------------------------------------------------------------


def _stage4_supervisor_command(args: argparse.Namespace, signal: str) -> list[str]:
    """Signal only the named resident-producer supervisor on the node."""
    if signal not in {"STOP", "CONT"}:
        raise ValueError(f"unsupported supervisor signal: {signal}")
    return [
        "ssh",
        args.spark_host,
        f"pkill -{signal} -f '[s]upervise_resident_producer[.]py' || true",
    ]


def _remote_container_status(args: argparse.Namespace, container: str) -> str | None:
    inspected = subprocess.run(
        [
            "ssh",
            args.spark_host,
            "docker",
            "inspect",
            "--format",
            "{{.State.Status}}",
            container,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if inspected.returncode != 0:
        return None
    return inspected.stdout.strip()


def _recover_stage4_resident(args: argparse.Namespace) -> dict[str, Any]:
    """Best-effort fail-closed recovery for every producer-swap exit path."""
    restart_command = _producer_restart_command(args, args.resident_container)
    plan: dict[str, Any] = {
        "inspect_naive": args.naive_container,
        "stop_naive_if_active": True,
        "restart_resident_if_inactive": restart_command,
    }
    print(
        "stage 4 recovery: inspect the naive producer, stop it if active, "
        "and restore the resident if inactive"
    )
    if not args.execute:
        return {"mode": "dry-run", **plan}

    errors: list[str] = []
    naive_status = _remote_container_status(args, args.naive_container)
    if naive_status in {"running", "restarting", "paused"}:
        try:
            plan["quiesce_naive"] = _quiesce_remote_producer(
                args, args.naive_container
            )
        except SessionAbort as error:
            errors.append(str(error))

    resident_before = _remote_container_status(args, args.resident_container)
    restart_exit: int | None = None
    if resident_before != "running":
        print("stage 4 recovery restart: " + " ".join(restart_command))
        restarted = subprocess.run(restart_command)
        restart_exit = restarted.returncode
        if restarted.returncode != 0:
            errors.append(
                f"resident restart ritual exited {restarted.returncode} during recovery"
            )

    resident_after = _remote_container_status(args, args.resident_container)
    return {
        "mode": "execute",
        **plan,
        "naive_status_before_recovery": naive_status,
        "resident_status_before_recovery": resident_before,
        "resident_restart_exit": restart_exit,
        "resident_status_after_recovery": resident_after,
        "resident_running": resident_after == "running",
        "errors": errors,
    }


def _stage4_swap_window(
    args: argparse.Namespace, out_dir: Path
) -> dict[str, Any]:
    """Run the destructive producer-swap portion while the supervisor is paused."""
    receipts: dict[str, Any] = {}
    print("\n--- stage4a-ter: stop resident and prove the accelerator lease is free ---")
    receipts["quiesce_resident"] = _quiesce_remote_producer(
        args, args.resident_container
    )

    # 4b. Swap to the naive (pre-streaming) container. Dry-run first is
    # mandatory per the design; the real swap only fires with --execute.
    dry_run_swap = _producer_restart_command(args, args.naive_container, ["--dry-run"])
    print("\n--- stage4b: dry-run swap to naive container (unwrapped, remote-node op) ---")
    print(" ".join(dry_run_swap))
    if args.execute:
        completed = subprocess.run(dry_run_swap)
        if completed.returncode != 0:
            raise SessionAbort("stage 4b: dry-run of naive-container swap failed")

    swap_to_naive = _producer_restart_command(args, args.naive_container)
    print(" ".join(swap_to_naive))
    if args.execute:
        completed = subprocess.run(swap_to_naive)
        if completed.returncode != 0:
            raise SessionAbort(
                "stage 4b: naive-container swap failed to become ready; "
                "abort-safe recovery will restore the resident"
            )
    receipts["swap_to_naive_command"] = swap_to_naive

    # 4c. Run the p4 ladder against the naive container at two depths.
    for depth_name, fixture, remote_fixture, first_gen in (
        ("65536", args.fixture_65536, args.remote_fixture_65536, args.stage4_first_generation),
        (
            "130815",
            args.fixture_130815,
            args.remote_fixture_130815,
            args.stage4_first_generation + 100,
        ),
    ):
        cell_out = out_dir / f"naive-{depth_name}"
        identity = f"{args.identity_prefix}-stage4-naive-{depth_name}"
        first_gen = resolve_first_generation(
            args,
            first_gen,
            handoffs=6,
            cell=f"stage 4 naive {depth_name}",
        )
        command = [
            "python3",
            str(REPO_ROOT / "scripts" / "qualify_nvfp4_fast.py"),
            "--model",
            str(args.model),
            "--prompt-token-fixture",
            str(fixture),
            "--cluster-config",
            str(args.cluster_config),
            "--rope-cache",
            str(args.rope_cache),
            "--out-dir",
            str(fresh_qualify_dir(cell_out)),
            "--identity",
            identity,
            "--resident",
            args.naive_container,
            "--remote-token-fixture",
            remote_fixture,
            "--first-generation",
            str(first_gen),
            "--mode",
            "p4",
            "--performance-only",
            "--variant",
            "text",
            "--output-tokens",
            "256",
            "--verify-length",
            "7",
            "--spark-host",
            args.spark_host,
            "--receiver-host",
            args.receiver_host,
            "--receiver-port",
            str(args.receiver_port),
            "--remote-sock",
            _stage4_remote_sock(args, args.naive_container),
        ] + _stage4_producer_profile_args(
            args, args.naive_container
        ) + EEE_OFF_ARGS
        receipts[f"naive-{depth_name}"] = run_leased(
            identity=identity,
            cell=f"naive-{depth_name}-p4",
            out_dir=cell_out,
            command=command,
            execute=args.execute,
        )

    # 4d. Redeploy the fixed image after the naive measurement.
    print("\n--- stage4c-bis: stop naive and prove the accelerator lease is free ---")
    receipts["quiesce_naive"] = _quiesce_remote_producer(args, args.naive_container)

    redeploy = _producer_restart_command(args, args.resident_container)
    print("\n--- stage4d: redeploy fixed image (unwrapped, remote-node op) ---")
    print(" ".join(redeploy))
    if args.execute:
        completed = subprocess.run(redeploy)
        if completed.returncode != 0:
            raise SessionAbort(
                "NODE LEFT ON BASELINE IMAGE: redeploying the fixed producer image "
                f"({args.resident_container}) failed after the naive-baseline stage. "
                "Abort-safe recovery will retry before the supervisor resumes."
            )
    receipts["redeploy_command"] = redeploy
    return receipts


def _stage4_guarded_swap(
    args: argparse.Namespace, out_dir: Path
) -> dict[str, Any]:
    """Pause supervision, run the swap, restore the resident, then resume."""
    receipts: dict[str, Any] = {}
    pause = _stage4_supervisor_command(args, "STOP")
    resume = _stage4_supervisor_command(args, "CONT")
    receipts["supervisor_pause_command"] = pause
    receipts["supervisor_resume_command"] = resume
    print("\n--- stage4a-ter-pre: pause the resident supervisor for the swap window ---")
    print(" ".join(pause))
    try:
        if args.execute and subprocess.run(pause).returncode != 0:
            raise SessionAbort(
                "stage 4: failed to pause the resident supervisor; refusing to stop "
                "the resident while it can be automatically restarted"
            )
        receipts.update(_stage4_swap_window(args, out_dir))
    finally:
        try:
            recovery = _recover_stage4_resident(args)
        except Exception as error:  # noqa: BLE001 - recovery must still resume supervision
            recovery = {
                "mode": "execute" if args.execute else "dry-run",
                "resident_running": False,
                "errors": [f"recovery raised {error!r}"],
            }
        receipts["abort_safe_recovery"] = recovery
        print("\n--- stage4d-bis: resume the resident supervisor ---")
        print(" ".join(resume))
        try:
            resume_exit = (
                subprocess.run(resume, timeout=60).returncode if args.execute else None
            )
        except (OSError, subprocess.SubprocessError) as error:
            print(f"stage 4: resuming the supervisor raised {error!r}", file=sys.stderr)
            resume_exit = -1
        receipts["supervisor_resume_exit"] = resume_exit
        if args.execute and (
            not receipts["abort_safe_recovery"].get("resident_running")
            or resume_exit != 0
        ):
            raise SessionAbort(
                "stage 4 abort-safe recovery did not leave both the fixed resident "
                "running and its supervisor resumed; operator recovery is required"
            )
    return receipts


def stage4(args: argparse.Namespace, out_root: Path) -> dict[str, Any]:
    out_dir = stage_dir(out_root, 4)
    out_dir.mkdir(parents=True, exist_ok=True)
    receipts: dict[str, Any] = {}

    # 4a. Capture current (fixed-image) state before any swap.
    before_command = ["ssh", args.spark_host, "docker", "inspect", args.resident_container]
    receipts["before_swap_inspect_command"] = before_command
    if args.execute:
        completed = subprocess.run(before_command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise SessionAbort(
                f"stage 4a: docker inspect of {args.resident_container} failed before swap"
            )
        (out_dir / "before-swap-inspect.json").write_text(completed.stdout, encoding="utf-8")

    # 4a-bis. The naive container is a historical artifact and may have been
    # pruned; its image is what we actually pin. Recreate it from the image
    # with the resident container's own mounts and argv so the only variable
    # between the two arms is the image itself.
    ensure_naive = _ensure_naive_container_command(args)
    print("\n--- stage4a-bis: ensure the naive container exists ---")
    print(" ".join(ensure_naive))
    if args.execute and subprocess.run(ensure_naive).returncode != 0:
        raise SessionAbort(
            f"stage 4: could not create {args.naive_container} from "
            f"{args.naive_image}; the rung-1 baseline cannot run"
        )
    receipts["ensure_naive_command"] = ensure_naive
    receipts.update(_stage4_guarded_swap(args, out_dir))

    # Post-redeploy sanity check: rerun the stage-1-shape smoke at a
    # shallow depth to confirm the node is actually healthy again.
    sanity_out = out_dir / "post-redeploy-smoke"
    sanity_identity = f"{args.identity_prefix}-stage4-post-redeploy-smoke"
    sanity_first_generation = resolve_first_generation(
        args,
        args.stage4_first_generation + 200,
        handoffs=6,
        cell="stage 4 post-redeploy smoke",
    )
    sanity_command = [
        "python3",
        str(REPO_ROOT / "scripts" / "qualify_nvfp4_fast.py"),
        "--model",
        str(args.model),
        "--prompt-token-fixture",
        str(args.fixture_8192),
        "--cluster-config",
        str(args.cluster_config),
        "--rope-cache",
        str(args.rope_cache),
        "--out-dir",
        str(fresh_qualify_dir(sanity_out)),
        "--identity",
        sanity_identity,
        "--resident",
        args.resident_container,
        "--remote-token-fixture",
        args.remote_fixture_8192,
        "--first-generation",
        str(sanity_first_generation),
        "--mode",
        "p4",
        "--performance-only",
        "--variant",
        "text",
        "--output-tokens",
        "256",
        "--verify-length",
        "7",
        "--spark-host",
        args.spark_host,
        "--receiver-host",
        args.receiver_host,
        "--receiver-port",
        str(args.receiver_port),
        "--remote-sock",
        _stage4_remote_sock(args, args.resident_container),
    ] + _stage4_producer_profile_args(
        args, args.resident_container
    ) + EEE_OFF_ARGS
    receipts["post_redeploy_smoke"] = run_leased(
        identity=sanity_identity,
        cell="post-redeploy-smoke-8192",
        out_dir=sanity_out,
        command=sanity_command,
        execute=args.execute,
    )
    return receipts


# --------------------------------------------------------------------------
# Stage 5: RUNG 3 deep warm-hit
# --------------------------------------------------------------------------


def _run_probe(
    *, identity: str, cell: str, out_dir: Path, command: list[str], execute: bool
) -> dict[str, Any]:
    """Unleased client step (no GPU): plain subprocess, fail-closed."""
    if not execute:
        return {"mode": "dry-run", "cell": cell, "identity": identity}
    completed = subprocess.run(command)
    if completed.returncode != 0:
        raise SessionAbort(f"stage 5 probe {cell!r} failed (exit {completed.returncode})")
    return {"cell": cell, "identity": identity, "exit": 0}


def _wait_port_ready(host: str, port: int, timeout_seconds: int, proc: subprocess.Popen) -> None:
    import socket

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SessionAbort(
                f"stage 5: the leased serve exited (code {proc.returncode}) "
                "before becoming ready"
            )
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError:
            time.sleep(3)
    raise SessionAbort(f"stage 5: serve on {host}:{port} not ready within {timeout_seconds}s")


def stage5(args: argparse.Namespace, out_root: Path) -> dict[str, Any]:
    """RUNG 3, warmhit-1 architecture (receipted precedent: nvfp4-pacing8g
    warmhit-1): each leased accelerator_safe cell IS a fresh `muser serve`;
    its one unleased HTTP/ssh probe runs beside it. A fresh process per depth
    is load-bearing because the enrolled fixtures are nested prefixes: sharing
    a radix would make the second depth's purported cold leg a partial hit.
    """
    out_dir = stage_dir(out_root, 5)
    out_dir.mkdir(parents=True, exist_ok=True)
    receipts: dict[str, Any] = {}
    for index, (depth_name, fixture) in enumerate(
        (("65536", args.fixture_65536), ("130815", args.fixture_130815))
    ):
        receipts[depth_name] = _stage5_depth(
            args, out_dir, depth_name, fixture, index
        )
    return receipts


def _stage5_depth(
    args: argparse.Namespace,
    out_dir: Path,
    depth_name: str,
    fixture: Path,
    index: int,
) -> dict[str, Any]:
    """Run one warm-hit depth against a new, independently leased server."""
    import secrets
    import urllib.request

    cell_out = out_dir / depth_name
    cell_out.mkdir(parents=True, exist_ok=True)
    serve_dir = cell_out / "serve"
    serve_dir.mkdir(parents=True, exist_ok=True)
    port = int(args.warmhit_base_url.rsplit(":", 1)[1].strip("/"))
    shutdown_token = secrets.token_hex(16)
    serve_identity = f"{args.identity_prefix}-stage5-serve-{depth_name}"
    serve_cmd = [
        "python3",
        str(ACCELERATOR_SAFE),
        "--identity",
        serve_identity,
        "--cell",
        f"warmhit-serve-{depth_name}",
        "--out-dir",
        str(serve_dir),
        "--quiet-seconds",
        "10",
    ]
    if args.execute:
        serve_cmd.append("--execute")
    serve_cmd += [
        "--",
        "env",
        "MUSER_CROSS_VENDOR_QK=1",
        str(REPO_ROOT / "target" / "release" / "muser"),
        "serve",
        "--model",
        str(args.model),
        "--prefill",
        "remote",
        "--cluster-config",
        str(args.cluster_config),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--api-key-file",
        str(args.warmhit_bearer_token_file),
        "--benchmark-deadline-seconds",
        "5400",
        "--benchmark-shutdown-token",
        shutdown_token,
    ]
    print(f"\n--- stage5 {depth_name}: fresh leased serve ---")
    print(" ".join(serve_cmd))

    serve_proc: subprocess.Popen | None = None
    result: dict[str, Any] = {
        "serve_identity": serve_identity,
        "serve_command": serve_cmd,
    }
    if args.execute:
        serve_proc = subprocess.Popen(serve_cmd)
    else:
        completed = subprocess.run(serve_cmd)
        if completed.returncode != 0:
            raise SessionAbort(
                f"stage 5 ({depth_name}): accelerator_safe dry-run of the serve cell failed"
            )
        result["serve"] = {"mode": "dry-run"}

    try:
        if serve_proc is not None:
            _wait_port_ready("127.0.0.1", port, 900, serve_proc)
        result["probe"] = _stage5_probe(
            args, cell_out, depth_name, fixture, index
        )
    finally:
        if serve_proc is not None:
            request = urllib.request.Request(
                f"{args.warmhit_base_url.rstrip('/')}/__muser/benchmark/shutdown",
                data=shutdown_token.encode(),
                method="POST",
            )
            try:
                urllib.request.urlopen(request, timeout=10).read()
            except OSError as error:
                print(
                    f"stage 5 ({depth_name}): shutdown POST failed ({error}); "
                    "waiting on deadline/terminate"
                )
            try:
                serve_proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                print(
                    f"stage 5 ({depth_name}): serve did not exit after shutdown; "
                    "terminating the wrapper"
                )
                serve_proc.terminate()
                serve_proc.wait(timeout=30)
            result["serve_exit"] = serve_proc.returncode
            result["serve_receipt"] = latest_receipt(serve_dir)

    if serve_proc is not None:
        if serve_proc.returncode != 0:
            raise SessionAbort(
                f"stage 5 ({depth_name}): leased serve exited "
                f"{serve_proc.returncode} after the probe"
            )
        if (
            not isinstance(result["serve_receipt"], dict)
            or result["serve_receipt"].get("exit_status") != 0
        ):
            raise SessionAbort(
                f"stage 5 ({depth_name}): successful serve receipt missing under "
                f"{serve_dir}"
            )
    return result


def _stage5_probe(
    args: argparse.Namespace,
    cell_out: Path,
    depth_name: str,
    fixture: Path,
    index: int,
) -> dict[str, Any]:
    identity = f"{args.identity_prefix}-stage5-warmhit-{depth_name}"
    evidence_path = fresh_attempt_file(cell_out / f"warmhit-{depth_name}.json")
    first_generation = resolve_first_generation(
        args,
        args.stage5_first_generation + index * 100,
        handoffs=2,
        cell=f"stage 5 warm-hit {depth_name}",
    )
    command = [
        "python3",
        str(REPO_ROOT / "scripts" / "gx10" / "vllm" / "warmhit_probe.py"),
        "--base-url",
        args.warmhit_base_url,
        "--bearer-token-file",
        str(args.warmhit_bearer_token_file),
        "--token-fixture",
        str(fixture),
        "--miss-token-fixture",
        str(args.warmhit_miss_fixture),
        "--node",
        args.spark_host,
        "--container",
        args.resident_container,
        "--sock",
        args.remote_sock,
        "--receiver-host",
        args.receiver_host,
        "--receiver-port",
        str(args.receiver_port),
        "--host-work",
        args.warmhit_host_work,
        "--first-generation",
        str(first_generation),
        "--request-prefix",
        identity,
        # The probe's 240 s default cannot cover a deep remote prefill:
        # at 65536 the producer was killed mid-handoff and the empty result
        # was misread as a correctness failure.
        "--producer-timeout-seconds",
        str(900 if int(depth_name) >= 65536 else 240),
        "--out",
        str(evidence_path),
    ]
    # The probe is an unleased HTTP/ssh client (no GPU): this depth's fresh
    # serve process holds the local lease for the whole probe.
    print(f"\n--- stage5 probe {depth_name} (unleased client) ---")
    print(" ".join(command))
    receipt = _run_probe(
        identity=identity,
        cell=f"warmhit-{depth_name}",
        out_dir=cell_out,
        command=command,
        execute=args.execute,
    )
    receipt = dict(receipt)
    receipt["warmhit_evidence_path"] = str(evidence_path)
    receipt["warm_ttft_below_cold"] = None
    receipt["headline_contradiction"] = None

    if args.execute:
        if not evidence_path.exists():
            raise SessionAbort(
                f"stage 5 ({depth_name}): warmhit_probe.py produced no evidence file "
                f"at {evidence_path}"
            )
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if evidence.get("legs_valid") is not True:
            raise SessionAbort(
                f"stage 5 ({depth_name}): a cold/warm leg did not complete "
                f"({evidence.get('leg_errors')}); this is an infrastructure "
                "failure, NOT a warm-hit correctness result"
            )
        if not evidence.get("outputs_match"):
            raise SessionAbort(
                f"stage 5 ({depth_name}): cold and warm response text did not match "
                "bit-exactly -- this is the core correctness claim for warm-hit reuse"
            )
        if evidence.get("miss_control_valid") is not True:
            raise SessionAbort(
                f"stage 5 ({depth_name}): the miss control did not complete "
                f"({evidence.get('miss_control_error')}); this is an infrastructure "
                "failure, NOT a warm-hit correctness result"
            )
        warm_below_cold = evidence.get("warm_ttft_below_cold") is True
        receipt["warm_ttft_below_cold"] = warm_below_cold
        receipt["headline_contradiction"] = not warm_below_cold
        if not warm_below_cold:
            print(
                f"WARNING: stage 5 ({depth_name}): warm-leg TTFT was not below the "
                "cold leg -- headline-contradicting result; flag for the owner ruling "
                "before writing it into the ledger (not a hard abort).",
                file=sys.stderr,
            )
    return receipt


# --------------------------------------------------------------------------
# Stage 6: RUNG 4 one deep delta cell at 65536 (cut 32768)
# --------------------------------------------------------------------------


def stage6(args: argparse.Namespace, out_root: Path) -> dict[str, Any]:
    out_dir = stage_dir(out_root, 6)
    out_dir.mkdir(parents=True, exist_ok=True)
    cut = 32768

    prefix_path = out_dir / f"e2-prefix-{cut}.tokens"
    remote_prefix_path = args.stage6_remote_prefix_fixture

    print(
        "\n--- stage6 setup: slice local prefix fixture, copy cut+1 witness to node ---"
    )
    slice_note = (
        f"NOTE: this driver does not fabricate the prefix slice contents -- it expects "
        f"--fixture-65536 to already exist locally as a whitespace token-id file, and "
        f"slices its first {cut} ids into {prefix_path} (local, this cut length) while "
        f"uploading the first {cut + 1} ids to {remote_prefix_path}, the matching "
        f"witness required by run_delta_probe's receiver-arming protocol."
    )
    print(slice_note)
    if args.execute:
        tokens = [int(value) for value in args.fixture_65536.read_bytes().split()]
        local_prefix, node_witness = delta_prefix_slices(tokens, cut)
        prefix_path.write_text(
            "\n".join(str(token) for token in local_prefix) + "\n", encoding="utf-8"
        )

        # remote_prefix_path is the path INSIDE the producer container; ssh
        # writes on the host, where the same file lives under the work dir
        # that is bind-mounted to /run/muser/work.
        host_prefix_path = f"{args.node_work_dir}/{Path(remote_prefix_path).name}"
        copy_command = [
            "ssh",
            args.spark_host,
            f"cat > {host_prefix_path}",
        ]
        print(
            " ".join(copy_command)
            + f" < {len(node_witness)}-token witness sliced from {args.fixture_65536}"
        )
        completed = subprocess.run(
            copy_command,
            input="\n".join(str(token) for token in node_witness) + "\n",
            text=True,
        )
        if completed.returncode != 0:
            raise SessionAbort("stage 6: failed to copy the prefix witness to the node")
        readback = subprocess.run(
            ["ssh", args.spark_host, f"cat {host_prefix_path}"],
            capture_output=True,
            text=True,
        )
        if readback.returncode != 0:
            raise SessionAbort("stage 6: failed to read back the node prefix witness")
        try:
            node_witness_readback = [int(value) for value in readback.stdout.split()]
        except ValueError as error:
            raise SessionAbort("stage 6: node prefix witness is not a token-id file") from error
        validate_delta_witness(local_prefix, node_witness_readback, cut)

    identity = f"{args.identity_prefix}-stage6-delta-65536"
    first_generation = resolve_first_generation(
        args,
        args.stage6_first_generation,
        handoffs=3,
        cell="stage 6 delta 65536",
    )
    command = [
        "python3",
        str(REPO_ROOT / "scripts" / "qualify_nvfp4_fast.py"),
        "--model",
        str(args.model),
        "--prompt-token-fixture",
        str(args.fixture_65536),
        "--cluster-config",
        str(args.cluster_config),
        "--rope-cache",
        str(args.rope_cache),
        "--out-dir",
        str(fresh_qualify_dir(out_dir)),
        "--identity",
        identity,
        "--resident",
        args.resident_container,
        "--remote-token-fixture",
        args.remote_fixture_65536,
        "--first-generation",
        str(first_generation),
        "--mode",
        "diagnostic",
        "--variant",
        "text",
        "--delta-prefix-cut",
        str(cut),
        "--prefix-token-fixture",
        str(prefix_path),
        "--remote-prefix-token-fixture",
        remote_prefix_path,
        "--output-tokens",
        "32",
        "--spark-host",
        args.spark_host,
        "--receiver-host",
        args.receiver_host,
        "--receiver-port",
        str(args.receiver_port),
        "--remote-sock",
        args.remote_sock,
    ] + EEE_OFF_ARGS
    receipt = run_leased(
        identity=identity,
        cell=f"delta-65536-cut{cut}",
        out_dir=out_dir,
        command=command,
        execute=args.execute,
    )
    return receipt


def delta_prefix_slices(tokens: list[int], cut: int) -> tuple[list[int], list[int]]:
    """Return the receiver's cut-token prefix and producer's cut+1 witness."""
    if cut < 1:
        raise SessionAbort(f"stage 6: delta prefix cut must be positive, got {cut}")
    if len(tokens) < cut + 1:
        raise SessionAbort(
            f"stage 6: 65536-depth fixture has only {len(tokens)} tokens, "
            f"cannot build a {cut + 1}-token witness"
        )
    local_prefix = tokens[:cut]
    node_witness = tokens[: cut + 1]
    validate_delta_witness(local_prefix, node_witness, cut)
    return local_prefix, node_witness


def validate_delta_witness(
    local_prefix: list[int], node_witness: list[int], cut: int
) -> None:
    """Fail before arming handoff unless the delta identity convention holds."""
    if len(local_prefix) != cut:
        raise SessionAbort(
            f"stage 6: local delta prefix has {len(local_prefix)} tokens, expected {cut}"
        )
    if len(node_witness) != cut + 1:
        raise SessionAbort(
            f"stage 6: node delta witness has {len(node_witness)} tokens, "
            f"expected {cut + 1}"
        )
    if node_witness[:cut] != local_prefix:
        raise SessionAbort(
            "stage 6: node delta witness prefix differs from the local cut-token slice"
        )


STAGE_FUNCS: dict[int, Callable[[argparse.Namespace, Path], dict[str, Any]]] = {
    1: stage1,
    2: stage2,
    3: stage3,
    4: stage4,
    5: stage5,
    6: stage6,
}


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    today = dt.date.today().isoformat().replace("-", "")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(tempfile.gettempdir()) / f"muser-results/kvpack-ladder-{today}",
    )
    parser.add_argument("--identity-prefix", default=f"kvpack-ladder-{today}")
    parser.add_argument("--from-stage", type=int, default=1, choices=range(1, 7))
    parser.add_argument("--to-stage", type=int, default=6, choices=range(1, 7))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="actually run stages (through accelerator_safe.py --execute); "
        "default is accelerator-free planning, but producer readiness is "
        "still checked unless --skip-producer-health-check is set",
    )
    parser.add_argument(
        "--pause-after-stage",
        type=int,
        action="append",
        default=None,
        help="stage number(s) after which the session stops and requires "
        "--confirm-stage to resume (default: [2], since stage 2's result "
        "gates the whole downstream ladder's headline claim)",
    )
    parser.add_argument(
        "--confirm-stage",
        type=int,
        action="append",
        default=[],
        help="acknowledge that a paused stage's result has been reviewed and the "
        "session may proceed past it",
    )
    parser.add_argument(
        "--allow-producer-swap",
        action="store_true",
        help="authorize stage 4 to stop the resident and run the naive producer arm",
    )
    parser.add_argument("--skip-producer-health-check", action="store_true")
    parser.add_argument(
        "--producer-ready-timeout",
        type=float,
        default=600.0,
        help="seconds to wait for the producer socket before refusing to start",
    )
    parser.add_argument(
        "--producer-health-cmd",
        nargs="+",
        default=None,
        help="override the default docker-inspect health check with a custom command",
    )

    # Shared infrastructure
    parser.add_argument("--spark-host", required=True)
    parser.add_argument("--receiver-host", required=True)
    parser.add_argument("--receiver-port", type=int, default=29590)
    parser.add_argument("--remote-sock", default="/run/muser/work/producer.sock")
    parser.add_argument(
        "--naive-remote-sock", default="/run/muser/work/producer-naive.sock"
    )
    parser.add_argument("--resident-container", required=True, help="current fixed-image (593b96a-class) container name")
    parser.add_argument("--naive-container", default="muser-redhat-native-f1-2112ceb", help="pre-streaming (2112ceb-class) container name")
    parser.add_argument("--cluster-config", type=Path, required=True)
    parser.add_argument("--rope-cache", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True, help="NVFP4 producer gguf")
    parser.add_argument("--bench", type=Path, required=True, help="muser-bench binary, for CPU tokenization")

    # Stage 1/2/4/6 token fixtures
    parser.add_argument("--fixture-8192", type=Path, required=True)
    parser.add_argument("--remote-fixture-8192", required=True)
    parser.add_argument("--fixture-65536", type=Path, required=True)
    parser.add_argument("--remote-fixture-65536", required=True)
    parser.add_argument("--fixture-130815", type=Path, required=True)
    parser.add_argument("--remote-fixture-130815", required=True)
    parser.add_argument("--stage1-first-generation", type=int, default=950000)
    parser.add_argument("--stage2-first-generation", type=int, default=950100)
    parser.add_argument(
        "--fresh-producer-before-eee-off",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="restart the resident producer before stage 2's gated EEE-off "
        "packet (cumulative-state control after the 2026-08-20 mid-prefill "
        "producer exit)",
    )
    parser.add_argument(
        "--stage2-eee-off-first-generation",
        type=int,
        default=950200,
        help="first generation for stage 2's EEE-off arm (the EEE-active "
        "attribution arm uses --stage2-first-generation)",
    )
    parser.add_argument("--stage4-first-generation", type=int, default=950900)
    parser.add_argument("--stage5-first-generation", type=int, default=950500)
    parser.add_argument("--stage6-first-generation", type=int, default=950700)
    parser.add_argument(
        "--stage6-remote-prefix-fixture",
        default="/run/muser/work/e2-prefix-32769.tokens",
        help="node-side path for the cut+1 prefix witness file",
    )

    # Stage 3
    parser.add_argument(
        "--native-image",
        default="muser/gx10-vllm-native:593b96a",
        help="producer image used for the one-shot stage-3 scoring container",
    )
    parser.add_argument(
        "--node-checkpoint-dir",
        required=True,
        help="node-side HF checkpoint dir mounted at /models/checkpoint",
    )
    parser.add_argument(
        "--node-engine-config",
        required=True,
        help="node-side engine config mounted at /run/muser/config.json",
    )
    parser.add_argument(
        "--node-work-root",
        required=True,
        help="node-side work root for staged fixtures and scripts",
    )
    parser.add_argument(
        "--node-results-root",
        required=True,
        help="node-side results root; stage-3 outputs are fetched back from here",
    )
    parser.add_argument(
        "--naive-image",
        default="muser/gx10-vllm-native:2112ceb",
        help="pre-streaming image backing the rung-1 naive baseline",
    )
    parser.add_argument(
        "--node-pki-dir",
        required=True,
    )
    parser.add_argument(
        "--node-work-dir",
        required=True,
    )
    parser.add_argument(
        "--node-receipts-dir",
        required=True,
    )
    parser.add_argument(
        "--naive-startup-receipt",
        default="runtime-naive-2112ceb-ctx131072.json",
    )
    parser.add_argument(
        "--naive-rope-cache",
        default="rope-f32le-naive-2112ceb-ctx131072.bin",
    )
    parser.add_argument(
        "--node-restart-script",
        required=True,
        help="node-side copy of restart_resident_producer.py (docker socket is local there)",
    )
    parser.add_argument("--e2-source-dir", type=Path, required=True, help="frozen nvfp4-e-series-fixtures-20260817 dir")
    parser.add_argument("--e2-receipt", type=Path, required=True)
    parser.add_argument(
        "--vllm-engine-config",
        type=Path,
        required=True,
        help="(legacy, unused since stage 3b runs node-side via "
        "--node-engine-config; kept for launcher compatibility)",
    )
    parser.add_argument("--checkpoint-revision", default=DEFAULT_CHECKPOINT_REVISION)
    parser.add_argument("--checkpoint-artifact-sha256", required=True)
    parser.add_argument("--kquant-model", type=Path, required=True)
    parser.add_argument("--kquant-model-sha256", required=True)
    parser.add_argument(
        "--alternate-model",
        type=Path,
        required=True,
        help="Q6 alternate lane model for the depth-local yardstick (E1 design)",
    )
    parser.add_argument(
        "--alternate-model-sha256",
        default="fb5f80d110c4fa932cc652e70873c0bd12c0954009038aa675e65086104c2739",
    )
    parser.add_argument(
        "--stage3-scratch-root",
        type=Path,
        default=REPO_ROOT / "target" / "kvpack-stage3-scratch",
        help="internal-disk root for transient full-vocabulary stage-3 logits",
    )
    parser.add_argument("--llama-perplexity-binary", type=Path, required=True)
    parser.add_argument(
        "--llama-perplexity-receipt",
        type=Path,
        required=True,
        help="pinned llama-perplexity binary receipt, passed as "
        "capture_llama_perplexity.py --llama-receipt for every document",
    )

    # Stage 5
    parser.add_argument("--warmhit-base-url", required=True)
    parser.add_argument("--warmhit-bearer-token-file", type=Path, required=True)
    parser.add_argument("--warmhit-miss-fixture", type=Path, required=True)
    parser.add_argument("--warmhit-host-work", required=True, help="node-local host path backing /run/muser/work")

    return parser.parse_args(argv)


def new_session_manifest(
    args: argparse.Namespace, session_log_path: Path
) -> dict[str, Any]:
    """Start a manifest while retaining every prior run at the same root."""
    manifest: dict[str, Any] = {
        "schema": "muser.kvpack-ladder-session.v1",
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "execute": args.execute,
        "identity_prefix": args.identity_prefix,
        "stages": {},
    }
    if not session_log_path.exists():
        return manifest
    try:
        previous = json.loads(session_log_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SessionAbort(
            f"cannot preserve prior session manifest {session_log_path}: {error}"
        ) from error
    if not isinstance(previous, dict):
        raise SessionAbort(
            f"cannot preserve prior session manifest {session_log_path}: not an object"
        )
    prior_runs = previous.get("prior_runs", [])
    if not isinstance(prior_runs, list):
        raise SessionAbort(
            f"cannot preserve prior session manifest {session_log_path}: "
            "prior_runs is not a list"
        )
    previous_run = {
        key: value for key, value in previous.items() if key != "prior_runs"
    }
    manifest["prior_runs"] = [*prior_runs, previous_run]
    return manifest


def write_session_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def record_session_abort(
    *,
    args: argparse.Namespace,
    out_root: Path,
    session_log_path: Path,
    manifest: dict[str, Any],
    stage: int | None,
    error: Exception,
) -> Path:
    """Record an expected or unexpected abort before control leaves main."""
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    context = "preflight-abort" if stage is None else f"stage-{stage}-abort"
    receipt_root = out_root if stage is None else stage_dir(out_root, stage)
    node_state_path = receipt_root / f"node-state-on-abort-{stamp}.log"
    append_node_state_receipt(
        node_state_path,
        spark_host=args.spark_host,
        node_work_dir=args.node_work_dir,
        context=context,
    )
    abort_record = {
        "aborted": True,
        "reason": str(error),
        "error_type": type(error).__name__,
        "unexpected_error": not isinstance(error, SessionAbort),
        "node_state_receipt": str(node_state_path),
    }
    if stage is None:
        manifest["preflight_abort"] = abort_record
    else:
        manifest["stages"][str(stage)] = abort_record
        manifest["aborted_at_stage"] = stage
    manifest["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_session_manifest(session_log_path, manifest)
    return node_state_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pause_after = (
        set(args.pause_after_stage) if args.pause_after_stage is not None else {2}
    )
    confirmed = set(args.confirm_stage)

    if args.from_stage > args.to_stage:
        print("--from-stage must be <= --to-stage", file=sys.stderr)
        return 2

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    session_log_path = out_root / "session-manifest.json"
    manifest = new_session_manifest(args, session_log_path)
    current_stage: int | None = None

    try:
        check_lock_not_held()
        check_qualifier_has_metal(args)
        check_producer_healthy(args)

        for stage in selected_stage_order(args.from_stage, args.to_stage):
            current_stage = stage
            if stage_already_done(out_root, stage):
                print(
                    f"stage {stage} ({STAGE_NAMES[stage]}) already complete; skipping"
                )
                manifest["stages"][str(stage)] = {"skipped": True}
                continue

            print(f"\n=========== STAGE {stage}: {STAGE_NAMES[stage]} ===========")
            require_stage_authorized(args, stage)
            result = STAGE_FUNCS[stage](args, out_root)
            manifest["stages"][str(stage)] = {"result": result}
            if args.execute:
                mark_stage_done(
                    out_root,
                    stage,
                    {"completed_at": dt.datetime.now(dt.timezone.utc).isoformat()},
                )

            if stage in pause_after and stage not in confirmed:
                print(
                    f"\nPAUSED after stage {stage} ({STAGE_NAMES[stage]}): review "
                    f"its result under {stage_dir(out_root, stage)} and rerun with "
                    f"--from-stage {stage + 1} --confirm-stage {stage} to continue.",
                    file=sys.stderr,
                )
                manifest["paused_after_stage"] = stage
                write_session_manifest(session_log_path, manifest)
                return 0

        manifest["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_session_manifest(session_log_path, manifest)
        print(
            f"\nSession complete through stage {args.to_stage}. "
            f"Manifest: {session_log_path}"
        )
        return 0
    except Exception as error:
        record_session_abort(
            args=args,
            out_root=out_root,
            session_log_path=session_log_path,
            manifest=manifest,
            stage=current_stage,
            error=error,
        )
        if isinstance(error, SessionAbort):
            location = (
                "preflight"
                if current_stage is None
                else f"stage {current_stage} ({STAGE_NAMES[current_stage]})"
            )
            print(f"\nABORT at {location}: {error}", file=sys.stderr)
            return 1
        location = "preflight" if current_stage is None else f"stage {current_stage}"
        print(
            f"\nUNEXPECTED ABORT at {location}: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        raise


if __name__ == "__main__":
    sys.exit(main())
