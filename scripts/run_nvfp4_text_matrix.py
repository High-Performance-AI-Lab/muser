#!/usr/bin/env python3
"""Run and evaluate the production native-NVFP4 remote text packet."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Any

from qualify_nvfp4_fast import (
    QUALIFIER_BINARY,
    require_metal_qualifier,
    show_eee,
    validate_fast_cluster_config,
)
from release_lock import force_unsealed
from release_readiness import atomic_json
from run_kvpack_ladder_session import replay_high_water


ROOT = Path(__file__).resolve().parents[1]
DEPTHS = (8192, 32768, 65536, 131008)
COUNTED_REPETITIONS = 3
LINK_GBPS_MINIMUM = 3.0
TTFT_CV_MAXIMUM = 0.02
MINIMUM_FREE_BYTES = 50 * 1024**3
HEX64 = re.compile(r"[0-9a-f]{64}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cluster-config", type=Path, required=True)
    parser.add_argument("--rope-cache", type=Path, required=True)
    parser.add_argument("--fixture", action="append", default=[], metavar="DEPTH=PATH")
    parser.add_argument(
        "--remote-fixture", action="append", default=[], metavar="DEPTH=CONTAINER_PATH"
    )
    parser.add_argument("--identity", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--resident", required=True)
    parser.add_argument("--spark-host", required=True)
    parser.add_argument("--receiver-host", required=True)
    parser.add_argument("--receiver-port", type=int, default=29590)
    parser.add_argument("--remote-sock", default="/run/muser/work/producer.sock")
    parser.add_argument(
        "--node-work-dir",
        required=True,
    )
    parser.add_argument("--eee-off-ruling", required=True)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def output_tokens(depth: int) -> int:
    return 48 if depth == 131008 else 256


def parse_depth_paths(values: list[str], *, remote: bool) -> dict[int, Path | str]:
    parsed: dict[int, Path | str] = {}
    for value in values:
        raw_depth, separator, raw_path = value.partition("=")
        if not separator or not raw_depth.isdigit() or not raw_path:
            raise RuntimeError(f"invalid depth mapping: {value!r}")
        depth = int(raw_depth)
        if depth not in DEPTHS or depth in parsed:
            raise RuntimeError(f"unexpected or duplicate depth mapping: {value!r}")
        if remote:
            if not raw_path.startswith("/run/muser/work/") or not re.fullmatch(
                r"/[A-Za-z0-9._/-]+", raw_path
            ):
                raise RuntimeError(f"unsafe remote fixture path: {raw_path!r}")
            parsed[depth] = raw_path
        else:
            parsed[depth] = Path(raw_path).expanduser().resolve()
    if set(parsed) != set(DEPTHS):
        raise RuntimeError(
            f"fixture depths differ: missing={sorted(set(DEPTHS) - set(parsed))}"
        )
    return parsed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def coefficient_of_variation(values: list[int]) -> float:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return 0.0 if mean == 0 else math.sqrt(variance) / mean


def json_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def validate_fast_records(
    records: list[dict[str, Any]], *, identity: str, depth: int
) -> dict[str, Any]:
    expected_outputs = output_tokens(depth)
    samples = [
        record
        for record in records
        if record.get("schema") == "muser.remote-qualify.v1"
        and record.get("kind") == "fast-performance-sample"
    ]
    summaries = [
        record
        for record in records
        if record.get("schema") == "muser.remote-qualify.v1"
        and record.get("kind") == "fast-performance-summary"
    ]
    if len(samples) != COUNTED_REPETITIONS + 1 or len(summaries) != 1:
        raise RuntimeError(
            f"depth {depth} lacks one warmup plus {COUNTED_REPETITIONS} counted samples"
        )
    if [sample.get("repetition") for sample in samples] != list(
        range(COUNTED_REPETITIONS + 1)
    ) or [sample.get("warmup") for sample in samples] != [True, False, False, False]:
        raise RuntimeError(f"depth {depth} warmup/counting order differs")
    for sample in samples:
        if (
            sample.get("identity") != identity
            or sample.get("prompt_positions") != depth
            or sample.get("output_tokens") != expected_outputs
            or sample.get("producer_receipt_profile") != "enrolled"
            or sample.get("deterministic_against_first") is not True
            or not isinstance(sample.get("remote_ttft_ns"), int)
            or sample["remote_ttft_ns"] <= 0
            or not isinstance(sample.get("installed_payload_gbps"), (int, float))
            or sample["installed_payload_gbps"] < LINK_GBPS_MINIMUM
            or HEX64.fullmatch(str(sample.get("generated_tokens_sha256"))) is None
            or HEX64.fullmatch(str(sample.get("full_logit_digest"))) is None
        ):
            raise RuntimeError(f"depth {depth} contains an invalid performance sample")
    if len({sample["generated_tokens_sha256"] for sample in samples}) != 1 or len(
        {sample["full_logit_digest"] for sample in samples}
    ) != 1:
        raise RuntimeError(f"depth {depth} output changed across repetitions")

    counted = samples[1:]
    ttfts = [sample["remote_ttft_ns"] for sample in counted]
    links = [float(sample["installed_payload_gbps"]) for sample in counted]
    cv = coefficient_of_variation(ttfts)
    summary = summaries[0]
    expected_median = sorted(ttfts)[len(ttfts) // 2]
    if (
        summary.get("identity") != identity
        or summary.get("performance_only") is not True
        or summary.get("reference_comparison") is not None
        or summary.get("prompt_positions") != depth
        or summary.get("output_tokens") != expected_outputs
        or summary.get("remote_ttft_raw_ns") != ttfts
        or summary.get("warmup_repetitions") != 1
        or summary.get("remote_ttft_warmup_ns") != samples[0]["remote_ttft_ns"]
        or summary.get("remote_ttft_median_ns") != expected_median
        or not math.isclose(float(summary.get("remote_ttft_cv", -1)), cv, abs_tol=1e-12)
        or summary.get("remote_ttft_target_applicable") is not False
        or summary.get("installed_payload_gbps") != links
        or summary.get("installed_payload_gbps_minimum") != LINK_GBPS_MINIMUM
        or summary.get("installed_payload_gbps_min") != min(links)
        or summary.get("producer_receipt_profile") != "enrolled"
        or summary.get("fast_generated_tokens_sha256")
        != samples[0]["generated_tokens_sha256"]
        or summary.get("fast_full_logit_digest") != samples[0]["full_logit_digest"]
        or summary.get("deterministic") is not True
        or summary.get("stable") is not True
        or summary.get("seal_eligible") is not False
    ):
        raise RuntimeError(f"depth {depth} summary does not bind its counted samples")
    if cv > TTFT_CV_MAXIMUM:
        raise RuntimeError(f"depth {depth} TTFT CV {cv:.6%} exceeds 2%")
    return {
        "prompt_tokens": depth,
        "output_tokens": expected_outputs,
        "warmup_ttft_ns": samples[0]["remote_ttft_ns"],
        "remote_ttft_raw_ns": ttfts,
        "remote_ttft_median_ns": expected_median,
        "remote_ttft_cv": cv,
        "installed_payload_gbps": links,
        "installed_payload_gbps_median": sorted(links)[len(links) // 2],
        "installed_payload_gbps_min": min(links),
        "generated_tokens_sha256": samples[0]["generated_tokens_sha256"],
        "full_logit_digest": samples[0]["full_logit_digest"],
        "producer_receipt_profile": "enrolled-v2-streaming",
    }


def run_capture(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60
        )
        return {"command": command, "exit_code": completed.returncode, "output": completed.stdout}
    except (OSError, subprocess.SubprocessError) as error:
        return {"command": command, "error": f"{type(error).__name__}: {error}"}


def ssh_command(host: str, *remote_args: str) -> list[str]:
    """Preserve remote argv boundaries when OpenSSH rebuilds a shell command."""
    return ["ssh", host, " ".join(shlex.quote(value) for value in remote_args)]


def capture_node_state(options: argparse.Namespace, path: Path, *, context: str) -> dict[str, Any]:
    lease_code = (
        "import fcntl,sys;h=open('/tmp/ferrite.gpu.lock','a+');"
        "\ntry: fcntl.flock(h.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)"
        "\nexcept BlockingIOError: print('LEASE HELD');sys.exit(1)"
        "\nprint('LEASE FREE')"
    )
    commands = {
        "resident_state": ssh_command(
            options.spark_host,
            "docker",
            "inspect",
            "--format",
            "{{json .State}}",
            options.resident,
        ),
        "resident_identity": ssh_command(
            options.spark_host,
            "docker",
            "inspect",
            "--format",
            "{{.Image}} {{.RestartCount}}",
            options.resident,
        ),
        "socket": ssh_command(
            options.spark_host,
            "docker",
            "exec",
            options.resident,
            "test",
            "-S",
            options.remote_sock,
        ),
        "lease": ssh_command(options.spark_host, "python3", "-c", lease_code),
        "supervisor": ssh_command(
            options.spark_host, "pgrep", "-af", "supervise_resident_producer.py"
        ),
        "containers": ssh_command(
            options.spark_host,
            "docker",
            "ps",
            "--format",
            "{{.Names}} {{.Status}}",
        ),
        "accelerators": ssh_command(
            options.spark_host,
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ),
        "storage": ssh_command(
            options.spark_host, "df", "-Pk", options.node_work_dir
        ),
    }
    checks = {name: run_capture(command) for name, command in commands.items()}
    state_check = checks["resident_state"]
    try:
        state = json.loads(state_check.get("output", ""))
    except json.JSONDecodeError as error:
        raise RuntimeError("cannot parse resident state") from error
    failures = []
    if state_check.get("exit_code") != 0 or state.get("Status") != "running":
        failures.append("resident is not running")
    if checks["socket"].get("exit_code") != 0:
        failures.append("producer socket is absent")
    if checks["lease"].get("exit_code") != 1 or "LEASE HELD" not in checks["lease"].get(
        "output", ""
    ):
        failures.append("resident does not prove the accelerator lease held")
    if checks["supervisor"].get("exit_code") != 0:
        failures.append("resident supervisor is inactive")
    storage_lines = checks["storage"].get("output", "").splitlines()
    try:
        available = int(storage_lines[-1].split()[3]) * 1024
    except (IndexError, ValueError) as error:
        raise RuntimeError("cannot parse node storage receipt") from error
    if available < MINIMUM_FREE_BYTES:
        failures.append(f"node free storage {available} is below 50 GiB")
    receipt = {
        "schema": "muser.gx10-node-state.v1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "context": context,
        "checks": checks,
        "node_free_bytes": available,
        "status": "passed" if not failures else "failed",
        "failures": failures,
    }
    atomic_json(path, receipt)
    if failures:
        raise RuntimeError("; ".join(failures))
    return receipt


def verify_fixtures(
    options: argparse.Namespace,
    local: dict[int, Path | str],
    remote: dict[int, Path | str],
) -> dict[str, Any]:
    records = {}
    for depth in DEPTHS:
        local_path = local[depth]
        remote_path = remote[depth]
        assert isinstance(local_path, Path)
        assert isinstance(remote_path, str)
        if not local_path.is_file() or local_path.is_symlink():
            raise RuntimeError(f"local fixture is missing or unsafe: {local_path}")
        token_count = len(local_path.read_bytes().split())
        if token_count != depth:
            raise RuntimeError(f"local fixture {local_path} has {token_count} tokens, expected {depth}")
        remote_sha = run_capture(
            ["ssh", options.spark_host, "docker", "exec", options.resident, "sha256sum", remote_path]
        )
        remote_count = run_capture(
            ["ssh", options.spark_host, "docker", "exec", options.resident, "wc", "-w", remote_path]
        )
        local_sha = sha256(local_path)
        if remote_sha.get("exit_code") != 0 or remote_sha.get("output", "").split()[:1] != [
            local_sha
        ]:
            raise RuntimeError(f"remote fixture differs at depth {depth}")
        try:
            observed_count = int(remote_count.get("output", "").split()[0])
        except (IndexError, ValueError) as error:
            raise RuntimeError(f"cannot read remote fixture count at depth {depth}") from error
        if remote_count.get("exit_code") != 0 or observed_count != depth:
            raise RuntimeError(f"remote fixture has wrong token count at depth {depth}")
        records[str(depth)] = {
            "local_path": str(local_path),
            "remote_path": remote_path,
            "sha256": local_sha,
            "tokens": depth,
        }
    return records


def cell_command(
    options: argparse.Namespace,
    *,
    depth: int,
    fixture: Path,
    remote_fixture: str,
    first_generation: int,
    cell_dir: Path,
) -> list[str]:
    inner = [
        sys.executable,
        str(ROOT / "scripts" / "qualify_nvfp4_fast.py"),
        "--model", str(options.model),
        "--prompt-token-fixture", str(fixture),
        "--cluster-config", str(options.cluster_config),
        "--rope-cache", str(options.rope_cache),
        "--out-dir", str(cell_dir / "qualify"),
        "--identity", options.identity,
        "--resident", options.resident,
        "--remote-token-fixture", remote_fixture,
        "--first-generation", str(first_generation),
        "--mode", "p4",
        "--performance-only",
        "--repetitions", str(COUNTED_REPETITIONS),
        "--variant", "text",
        "--output-tokens", str(output_tokens(depth)),
        "--verify-length", "7",
        "--spark-host", options.spark_host,
        "--receiver-host", options.receiver_host,
        "--receiver-port", str(options.receiver_port),
        "--remote-sock", options.remote_sock,
        "--eee", "off",
        "--eee-off-ruling", options.eee_off_ruling,
    ]
    command = [
        sys.executable,
        str(ROOT / "scripts" / "accelerator_safe.py"),
        "--identity", options.identity,
        "--cell", f"remote-text-{depth}",
        "--out-dir", str(cell_dir),
        "--result-receipt", str(cell_dir / "accelerator-result.json"),
        "--quiet-seconds", "10",
    ]
    if options.execute:
        command.append("--execute")
    return [*command, "--", *inner]


def validate_execution_receipts(
    *,
    options: argparse.Namespace,
    depth: int,
    command: list[str],
    cell_dir: Path,
    process_exit: int,
) -> tuple[dict[str, Any], Path, Path]:
    receipt_path = cell_dir / "accelerator-result.json"
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise RuntimeError(f"depth {depth} has no accelerator result receipt")
    accelerator = json.loads(receipt_path.read_text(encoding="utf-8"))
    separator = command.index("--")
    if (
        accelerator.get("schema") != "muser.accelerator-result.v1"
        or accelerator.get("identity") != options.identity
        or accelerator.get("cell") != f"remote-text-{depth}"
        or accelerator.get("command") != command[separator + 1 :]
        or accelerator.get("exit_status") != process_exit
        or process_exit != 0
    ):
        raise RuntimeError(f"depth {depth} accelerator receipt differs from execution")
    log_path = Path(str(accelerator.get("command_log", "")))
    if (
        not log_path.is_absolute()
        or not log_path.is_file()
        or log_path.is_symlink()
        or log_path.parent.resolve() != cell_dir.resolve()
    ):
        raise RuntimeError(f"depth {depth} command log is missing or unsafe")
    control_path = cell_dir / "qualify" / "control.json"
    if not control_path.is_file() or control_path.is_symlink():
        raise RuntimeError(f"depth {depth} control receipt is missing or unsafe")
    control = json.loads(control_path.read_text(encoding="utf-8"))
    if (
        control.get("schema") != "muser.nvfp4-fast-control.v1"
        or control.get("identity") != options.identity
        or control.get("mode") != "p4"
        or control.get("variant") != "text"
        or control.get("repetitions") != COUNTED_REPETITIONS
        or control.get("output_tokens") != output_tokens(depth)
        or control.get("performance_only") is not True
        or control.get("producer_receipt_profile") != "enrolled"
        or control.get("exit_status") != 0
        or control.get("eee_check_error") is not None
        or "EEE status: disabled" not in str(control.get("eee_before", ""))
        or "EEE status: disabled" not in str(control.get("eee_after", ""))
        or len(control.get("producer_receipts", [])) != COUNTED_REPETITIONS + 1
    ):
        raise RuntimeError(f"depth {depth} control receipt is incomplete or mismatched")
    return accelerator, log_path, control_path


def main() -> int:
    options = parse_args()
    evidence = options.evidence_dir or options.out.with_name("remote-evidence")
    completed: dict[str, Any] = {}
    try:
        local = parse_depth_paths(options.fixture, remote=False)
        remote = parse_depth_paths(options.remote_fixture, remote=True)
        if not options.execute:
            commands = []
            generation = 1
            for depth in DEPTHS:
                cell_dir = evidence / f"remote-text-{depth}"
                commands.append(
                    cell_command(
                        options,
                        depth=depth,
                        fixture=local[depth],
                        remote_fixture=remote[depth],
                        first_generation=generation,
                        cell_dir=cell_dir,
                    )
                )
                generation += COUNTED_REPETITIONS + 1
            print(
                json.dumps(
                    {
                        "schema": "muser.remote-text-matrix-plan.v1",
                        "mode": "plan",
                        "identity": options.identity,
                        "depths": list(DEPTHS),
                        "variant": "text",
                        "counted_repetitions": COUNTED_REPETITIONS,
                        "warmup_repetitions_per_depth": 1,
                        "generation_policy": "live replay high-water plus one per depth",
                        "commands": commands,
                        "accelerator_touched": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if options.out.exists() or options.out.is_symlink():
            raise RuntimeError(f"refusing to replace output: {options.out}")
        if evidence.exists() or evidence.is_symlink():
            raise RuntimeError(f"refusing to reuse evidence directory: {evidence}")
        for path in (options.model, options.cluster_config, options.rope_cache):
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(f"required input is missing or unsafe: {path}")
        if shutil.disk_usage(evidence.parent).free < MINIMUM_FREE_BYTES:
            raise RuntimeError("evidence volume has less than 50 GiB free")
        require_metal_qualifier(QUALIFIER_BINARY)
        validate_fast_cluster_config(options.cluster_config)
        evidence.mkdir(parents=True)
        fixture_receipt = verify_fixtures(options, local, remote)
        atomic_json(evidence / "fixtures.json", {"fixtures": fixture_receipt})
        show_eee(options.spark_host, expect="disabled")
        capture_node_state(options, evidence / "node-state-before.json", context="before")
        initial_high_water = replay_high_water(options.cluster_config)
        atomic_json(
            evidence / "generation-preflight.json",
            {"live_high_water": initial_high_water, "handoffs_per_depth": 4},
        )
        for depth in DEPTHS:
            high_water = replay_high_water(options.cluster_config)
            first_generation = high_water + 1
            cell_dir = evidence / f"remote-text-{depth}"
            command = cell_command(
                options,
                depth=depth,
                fixture=local[depth],
                remote_fixture=remote[depth],
                first_generation=first_generation,
                cell_dir=cell_dir,
            )
            result = subprocess.run(command, cwd=ROOT)
            _, log_path, control_path = validate_execution_receipts(
                options=options,
                depth=depth,
                command=command,
                cell_dir=cell_dir,
                process_exit=result.returncode,
            )
            receipt_path = cell_dir / "accelerator-result.json"
            cell = validate_fast_records(json_records(log_path), identity=options.identity, depth=depth)
            final_high_water = replay_high_water(options.cluster_config)
            expected_high_water = first_generation + COUNTED_REPETITIONS
            if final_high_water != expected_high_water:
                raise RuntimeError(
                    f"depth {depth} replay high-water {final_high_water} differs from expected "
                    f"{expected_high_water}"
                )
            completed[str(depth)] = {
                **cell,
                "first_generation": first_generation,
                "last_generation": expected_high_water,
                "command_log": str(log_path.resolve()),
                "command_log_sha256": sha256(log_path),
                "accelerator_result": str(receipt_path.resolve()),
                "accelerator_result_sha256": sha256(receipt_path),
                "control": str(control_path.resolve()),
                "control_sha256": sha256(control_path),
            }
        capture_node_state(options, evidence / "node-state-after.json", context="after")
        receipt: dict[str, Any] = {
            "schema": "muser.remote-prefill.native-text.v1",
            "status": "passed",
            "identity": options.identity,
            "variant": "text",
            "depths": list(DEPTHS),
            "counted_repetitions": COUNTED_REPETITIONS,
            "warmup_repetitions_per_depth": 1,
            "ttft_cv_maximum": TTFT_CV_MAXIMUM,
            "installed_payload_gbps_minimum": LINK_GBPS_MINIMUM,
            "cells": completed,
            "initial_replay_high_water": initial_high_water,
            "final_replay_high_water": replay_high_water(options.cluster_config),
            "evidence_dir": str(evidence.resolve()),
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "seal_eligible": True,
        }
        force_unsealed(receipt, lane="remote")
        atomic_json(options.out, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    except BaseException as error:
        if evidence.is_dir():
            try:
                capture_node_state(options, evidence / "node-state-abort.json", context="abort")
            except BaseException as state_error:
                state_failure = f"{type(state_error).__name__}: {state_error}"
            else:
                state_failure = None
            try:
                high_water = replay_high_water(options.cluster_config)
            except BaseException:
                high_water = None
            atomic_json(
                evidence / "ABORT.json",
                {
                    "schema": "muser.remote-text-matrix-abort.v1",
                    "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "error": f"{type(error).__name__}: {error}",
                    "completed_cells": completed,
                    "live_replay_high_water": high_water,
                    "node_state_error": state_failure,
                    "automatic_retry": False,
                },
            )
        print(f"native remote text matrix failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
