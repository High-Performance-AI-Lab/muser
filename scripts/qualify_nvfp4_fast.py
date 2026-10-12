#!/usr/bin/env python3
"""Qualify the native NVFP4 lane without mutating the link or exact anchor."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


MAC_EEE_INTERFACE = "en0"
REPO_ROOT = Path(__file__).resolve().parents[1]
QUALIFIER_BINARY = REPO_ROOT / "target" / "release" / "muser-remote-qualify"
# Turning EEE off forces a link retrain on its own; give the Mac PHY a full
# settle before trusting the re-read (design floor: >= 30 s).
EEE_OFF_SETTLE_SECONDS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt-token-fixture", type=Path, required=True)
    parser.add_argument("--cluster-config", type=Path, required=True)
    parser.add_argument("--rope-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--resident", required=True)
    parser.add_argument("--remote-token-fixture", required=True)
    parser.add_argument("--first-generation", type=int, required=True)
    parser.add_argument("--mode", choices=("diagnostic", "p4"))
    parser.add_argument("--performance-only", action="store_true")
    parser.add_argument(
        "--pre-streaming-control",
        action="store_true",
        help="require the historical v2/native producer profile that defers "
        "its first segment until D2H is complete; valid only for a text P4 "
        "performance-only control",
    )
    parser.add_argument("--variant", choices=("text", "target-plus-dflash"), default="text")
    parser.add_argument("--output-tokens", type=int, default=None)
    parser.add_argument("--dflash", type=Path)
    parser.add_argument("--remote-dflash-session")
    parser.add_argument("--dflash-identity-sha256")
    parser.add_argument("--verify-length", type=int, choices=(3, 7, 15), default=7)
    parser.add_argument(
        "--repetitions",
        type=int,
        help="counted P4 performance repetitions; defaults to 5. Release "
        "matrix packets pin 3 and still receive one additional warmup.",
    )
    parser.add_argument(
        "--eee",
        choices=("require-disabled", "off"),
        default="require-disabled",
        help="Mac Ethernet EEE mode: require-disabled (default) asserts the "
        "enrolled post-migration invariant and never mutates the link; off "
        "snapshots the pre-mutation link state, disables EEE on Mac en0 before "
        "the run (control.json records eee_mutated=true), and fails closed "
        "unless the re-read reports 'EEE status: disabled'. Off refuses to "
        "mutate unless --eee-off-ruling cites the 2026-08-20 owner ruling.",
    )
    parser.add_argument(
        "--eee-off-ruling",
        help="ledger reference (e.g. 'goal-parity-ledger-2026-08.md#<anchor>') "
        "recording the explicit owner ruling that authorizes --eee off; "
        "required by --eee off, recorded in control.json as eee_off_ruling",
    )
    parser.add_argument("--spark-host", required=True)
    parser.add_argument("--receiver-host", required=True)
    parser.add_argument("--receiver-port", type=int, default=29590)
    parser.add_argument(
        "--remote-request-script",
        default="/opt/muser/scripts/gx10/vllm/request_producer.py",
    )
    parser.add_argument("--remote-sock", default="/run/muser/work/producer.sock")
    parser.add_argument(
        "--delta-prefix-cut",
        type=int,
        default=0,
        help="run the delta probe: prefix install, prefix-cut handoff, full reference",
    )
    parser.add_argument(
        "--prefix-token-fixture",
        type=Path,
        help="local token fixture holding exactly --delta-prefix-cut tokens",
    )
    parser.add_argument(
        "--remote-prefix-token-fixture",
        help="node-side token fixture holding --delta-prefix-cut + 1 tokens "
        "(the producer holds back the last token, so the extra token makes "
        "the prefix handoff cover the whole prefix)",
    )
    return parser.parse_args()


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_fast_cluster_config(path: Path, expected_receiver: str) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("producer_mode") != "native":
        raise ValueError(
            "fast-lane cluster config must declare producer_mode=native"
        )
    if value.get("advertised_receiver_host") != expected_receiver:
        raise ValueError(
            "fast-lane cluster config must advertise the enrolled receiver "
            f"{expected_receiver}"
        )


def validate_pre_streaming_control(options: argparse.Namespace) -> None:
    """Keep the historical deferred-transfer exception out of enrolled cells."""
    if options.pre_streaming_control and (
        options.mode != "p4"
        or not options.performance_only
        or options.variant != "text"
        or options.delta_prefix_cut
    ):
        raise ValueError(
            "--pre-streaming-control requires --mode p4 --performance-only "
            "--variant text and cannot modify a delta probe"
        )


def validate_performance_mode(options: argparse.Namespace) -> None:
    """Admit the Rust qualifier's P4 packet and one-handoff diagnostic path."""
    if options.performance_only and options.mode not in ("p4", "diagnostic"):
        raise ValueError("performance-only requires p4 or diagnostic mode")


def performance_repetitions(options: argparse.Namespace) -> int:
    """Resolve the counted repetitions without changing diagnostic geometry."""
    requested = options.repetitions
    if requested is None:
        return 1 if options.mode == "diagnostic" or options.delta_prefix_cut else 5
    if (
        options.mode != "p4"
        or not options.performance_only
        or options.delta_prefix_cut
        or requested < 3
    ):
        raise ValueError(
            "--repetitions requires a P4 performance-only cell and must be at least 3"
        )
    return requested


def producer_receipt_arguments(options: argparse.Namespace) -> list[str]:
    """Select the qualifier receipt contract for the chosen execution path."""
    if options.performance_only or options.mode != "diagnostic":
        return ["--external-producer-receipt-dir", str(options.out_dir)]
    first_transfer = (
        f"f-{options.mode}-{options.variant}-g{options.first_generation}"
    )
    return [
        "--external-producer-receipt",
        str(options.out_dir / f"{first_transfer}-client.json"),
    ]


def show_eee(_host: str, *, expect: str | None = "disabled") -> str:
    """Read Mac Ethernet EEE state and assert the selected invariant.

    The post-migration path enters the MikroTik fabric through Mac ``en0``;
    the producer's 200 GbE interface does not implement the ethtool EEE API.
    macOS reports EEE as a selected media option. Requiring the option to be
    supported makes its absence from the current media line positive evidence
    that EEE is disabled, rather than treating an unknown driver as a pass.
    """
    if expect not in ("active", "disabled", None):
        raise ValueError(f"unknown EEE expectation {expect!r}")
    result = run(["ifconfig", "-m", MAC_EEE_INTERFACE], capture=True)
    assert result.stdout is not None
    media = next(
        (
            line.strip()
            for line in result.stdout.splitlines()
            if line.lstrip().startswith("media:")
        ),
        None,
    )
    if media is None or "status: active" not in result.stdout:
        raise RuntimeError(f"Mac {MAC_EEE_INTERFACE} is not an active Ethernet link")
    if "mediaopt energy-efficient-ethernet" not in result.stdout:
        raise RuntimeError(
            f"Mac {MAC_EEE_INTERFACE} does not expose an EEE media capability; "
            "cannot prove the enrolled state"
        )
    enabled = "energy-efficient-ethernet" in media
    summary = f"EEE status: {'enabled - active' if enabled else 'disabled'}\n"
    print(result.stdout, end="" if result.stdout.endswith("\n") else "\n", flush=True)
    print(summary, end="", flush=True)
    if expect == "active" and not enabled:
        raise RuntimeError(f"Mac {MAC_EEE_INTERFACE} EEE must be enabled-active")
    if expect == "disabled" and enabled:
        raise RuntimeError(
            f"Mac {MAC_EEE_INTERFACE} EEE is still armed; refusing to run"
        )
    return result.stdout + summary


def disable_eee(host: str) -> None:
    """Disable Mac Ethernet EEE under the ruling, then settle the retrain."""
    if "EEE status: disabled" in show_eee(host, expect=None):
        print(f"eee-off: Mac {MAC_EEE_INTERFACE} already disabled", flush=True)
        return
    run(
        [
            "sudo",
            "-n",
            "ifconfig",
            MAC_EEE_INTERFACE,
            "-mediaopt",
            "energy-efficient-ethernet",
        ]
    )
    print(
        f"eee-off: settling {EEE_OFF_SETTLE_SECONDS}s through the link retrain",
        flush=True,
    )
    time.sleep(EEE_OFF_SETTLE_SECONDS)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.rename(path)


def copy_receipt(host: str, resident: str, remote: str, local: Path) -> None:
    result = subprocess.run(
        ["ssh", host, "docker", "exec", resident, "cat", remote],
        check=True,
        stdout=subprocess.PIPE,
    )
    temporary = local.with_name(f".{local.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(result.stdout)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.rename(local)


METAL_GUARD = b"remote qualification requires macOS and the metal feature"


def require_metal_qualifier(path: Path) -> None:
    """Fail closed if muser-remote-qualify lacks the `metal` feature.

    Built without it the binary refuses every remote cell, so the receiver
    never binds, every producer handoff is refused, and each refusal exits
    the producer fail-closed. The visible symptom is always downstream --
    lease-unavailable, empty responses, timeouts -- which cost a full day on
    2026-08-21. Checked here because every remote cell goes through this
    wrapper.
    """
    if not path.is_file():
        raise RuntimeError(f"qualifier binary missing: {path}")
    if METAL_GUARD in path.read_bytes():
        raise RuntimeError(
            f"{path} was built WITHOUT the metal feature; every remote cell "
            "would be refused. Rebuild with: cargo build --release --locked "
            "-p muser-bench --bin muser-remote-qualify --features metal"
        )


def main() -> int:
    options = parse_args()
    if options.delta_prefix_cut:
        if options.delta_prefix_cut % 256 != 0:
            raise ValueError("--delta-prefix-cut must be 256-aligned")
        if not options.prefix_token_fixture or not options.remote_prefix_token_fixture:
            raise ValueError(
                "the delta probe needs --prefix-token-fixture and --remote-prefix-token-fixture"
            )
    elif options.mode is None:
        raise ValueError("either --mode or --delta-prefix-cut is required")
    if os.environ.get("MUSER_ACCELERATOR_LEASE") != "1":
        raise RuntimeError("must run below scripts/accelerator_safe.py")
    if options.first_generation < 1:
        raise ValueError("first generation must be positive")
    validate_performance_mode(options)
    validate_pre_streaming_control(options)
    if options.eee == "off" and not options.eee_off_ruling:
        raise ValueError(
            "--eee off can mutate Mac Ethernet state and needs "
            "--eee-off-ruling citing the ledger entry that records the "
            "explicit owner ruling"
        )
    if options.eee != "off" and options.eee_off_ruling:
        raise ValueError("--eee-off-ruling is only meaningful with --eee off")
    # Refuse a featureless qualifier before observing or mutating the remote
    # link. In particular, --eee off must never retrain the link for a binary
    # that cannot run the Metal receiver.
    require_metal_qualifier(QUALIFIER_BINARY)
    validate_fast_cluster_config(options.cluster_config, options.receiver_host)
    dflash_values = (
        options.dflash,
        options.remote_dflash_session,
        options.dflash_identity_sha256,
    )
    if (options.variant == "target-plus-dflash") != all(dflash_values):
        raise ValueError("target-plus-dflash requires all local/remote DFlash artifacts")
    if options.variant == "target-plus-dflash":
        raise ValueError(
            "native NVFP4 speculative decode is unqualified; qualify the text/plain-decode lane or use the kquant speculative lane"
        )
    if options.variant == "text" and any(dflash_values):
        raise ValueError("text qualification cannot carry DFlash artifacts")
    repetitions = performance_repetitions(options)
    if options.mode == "diagnostic" or options.delta_prefix_cut:
        output_tokens = options.output_tokens if options.output_tokens is not None else 32
    else:
        # p4 defaults to 256 outputs; an explicit flag overrides (the 131072-context
        # ceiling needs 48 at the 131008 depth to leave prompt+output inside the cap).
        output_tokens = options.output_tokens if options.output_tokens is not None else 256
    options.out_dir.mkdir(parents=True, exist_ok=False)
    control: dict[str, Any] = {
        "schema": "muser.nvfp4-fast-control.v1",
        "identity": options.identity,
        "mode": options.mode,
        "variant": options.variant,
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_sha256": sha256(options.model),
        "prompt_sha256": sha256(options.prompt_token_fixture),
        "cluster_config_sha256": sha256(options.cluster_config),
        "repetitions": repetitions,
        "output_tokens": output_tokens,
        "performance_only": options.performance_only,
        "producer_receipt_profile": (
            "historical-pre-streaming-control"
            if options.pre_streaming_control
            else "enrolled"
        ),
        "eee_mutated": False,
        "producer_receipts": [],
    }
    qualifier: subprocess.Popen[str] | None = None
    status = 1
    expected_eee_state = "disabled"
    try:
        if options.eee == "off":
            control["eee_off_ruling"] = options.eee_off_ruling
            # Unasserted snapshot BEFORE the mutation records whether the
            # migrated Mac Ethernet path was already in its enrolled state.
            control["eee_pre_mutation"] = show_eee(options.spark_host, expect=None)
            # Record the mutation before issuing it: even a failed verify
            # below leaves the link in an owner-visible mutated state.
            control["eee_mutated"] = True
            disable_eee(options.spark_host)
        control["eee_before"] = show_eee(
            options.spark_host, expect=expected_eee_state
        )
        qualifier_command = [
            str(QUALIFIER_BINARY),
            "--model",
            str(options.model),
            "--prompt-token-fixture",
            str(options.prompt_token_fixture),
            "--cluster-config",
            str(options.cluster_config),
            "--variant",
            options.variant,
            "--repetitions",
            str(repetitions),
            "--output-tokens",
            str(output_tokens),
            "--verify-length",
            str(options.verify_length),
            "--identity",
            options.identity,
        ]
        if options.delta_prefix_cut:
            qualifier_command.extend(
                [
                    "--delta-prefix-cut",
                    str(options.delta_prefix_cut),
                    "--prefix-prompt-fixture",
                    str(options.prefix_token_fixture),
                ]
            )
        else:
            qualifier_command.append(f"--{options.mode}")
            if options.performance_only:
                qualifier_command.append("--performance-only")
            else:
                qualifier_command.append("--drift-graded")
            if options.pre_streaming_control:
                qualifier_command.append("--pre-streaming-control")
            if options.dflash is not None:
                qualifier_command.extend(["--dflash", str(options.dflash)])
            elif options.mode == "p4" and not options.performance_only:
                qualifier_command.append("--reference-once")
            qualifier_command.extend(producer_receipt_arguments(options))
        environment = os.environ.copy()
        environment["MUSER_CROSS_VENDOR_QK"] = "1"
        environment["MUSER_CROSS_VENDOR_ROPE_CACHE"] = str(options.rope_cache)
        if options.mode == "p4":
            environment["MUSER_REMOTE_QUALIFY_SERIAL"] = "1"
        if options.mode == "diagnostic":
            environment["MUSER_REMOTE_CACHE_DIFF"] = "1"
        print("+ " + " ".join(qualifier_command), flush=True)
        qualifier = subprocess.Popen(
            qualifier_command,
            text=True,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        assert qualifier.stdout is not None
        generation = options.first_generation
        phase_count = 0

        def fire(tokens_remote: str, cut: int) -> None:
            nonlocal generation, phase_count
            tag = "delta" if options.delta_prefix_cut else f"{options.mode}-{options.variant}"
            transfer = f"f-{tag}-g{generation}"
            remote_receipt = f"/run/muser/work/{transfer}-client.json"
            producer = [
                "ssh",
                options.spark_host,
                "docker",
                "exec",
                options.resident,
                "python3",
                options.remote_request_script,
                "--sock",
                options.remote_sock,
                "--tokens",
                tokens_remote,
                "--request-id",
                transfer,
                "--generation",
                str(generation),
                "--transfer-id",
                transfer,
                "--receiver-host",
                options.receiver_host,
                "--receiver-port",
                str(options.receiver_port),
                "--output",
                remote_receipt,
                "--timeout-seconds",
                "1800",
            ]
            if cut:
                producer.extend(["--prefix-cut", str(cut)])
            if options.remote_dflash_session is not None:
                producer.extend(
                    [
                        "--dflash-session",
                        options.remote_dflash_session,
                        "--dflash-identity-sha256",
                        options.dflash_identity_sha256,
                    ]
                )
            try:
                result = run(producer, capture=True)
            except subprocess.CalledProcessError:
                # Drain the qualifier's own output before raising: the real
                # failure (e.g., a dead Metal session after the phase line)
                # is usually sitting unread in its pipe.
                try:
                    qualifier.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                else:
                    for tail_line in qualifier.stdout:
                        print(tail_line, end="", flush=True)
                raise
            if result.stdout:
                print(result.stdout.splitlines()[-1], flush=True)
            local_receipt = options.out_dir / f"{transfer}-client.json"
            copy_receipt(
                options.spark_host,
                options.resident,
                remote_receipt,
                local_receipt,
            )
            control["producer_receipts"].append(
                {"path": str(local_receipt), "sha256": sha256(local_receipt)}
            )
            generation += 1
            phase_count += 1

        for line in qualifier.stdout:
            print(line, end="", flush=True)
            if options.delta_prefix_cut:
                if "remote-delta-prefix-receive:start" in line:
                    fire(options.remote_prefix_token_fixture, 0)
                elif "remote-delta-reference-receive:start" in line:
                    fire(options.remote_token_fixture, 0)
                elif "remote-delta-receive:start" in line:
                    fire(options.remote_token_fixture, options.delta_prefix_cut)
                continue
            if "prefill-export-receive:start" not in line:
                continue
            fire(options.remote_token_fixture, 0)
        status = qualifier.wait()
        if status != 0:
            raise RuntimeError(f"qualifier failed with exit status {status}")
        expected_phases = (
            3
            if options.delta_prefix_cut
            # Only the --performance-only p4 lane (run_fast_performance_only,
            # crates/muser-bench/src/remote.rs) runs a preregistered warmup
            # handoff before the counted repetitions (owner ruling
            # 2026-08-19, plan §7.3): that path loops `0..=repetitions`,
            # firing one extra remote-target phase. The drift-graded p4 lane
            # (--drift-graded --reference-once, same file's generic
            # repetition loop) computes its single local reference locally
            # and loops `0..repetitions`, so it fires exactly `repetitions`
            # remote phases and must NOT get the +1.
            else (
                repetitions + 1
                if options.mode == "p4"
                and options.dflash is None
                and options.performance_only
                else repetitions * (2 if options.dflash is not None else 1)
            )
        )
        if phase_count != expected_phases:
            raise RuntimeError(
                f"qualifier requested {phase_count} producer phases, expected {expected_phases}"
            )
        status = 0
    finally:
        if qualifier is not None and qualifier.poll() is None:
            qualifier.terminate()
            qualifier.wait()
        try:
            # No silent mid-session restore: record the final link state in
            # the mode the run was selected to hold.
            control["eee_after"] = show_eee(
                options.spark_host, expect=expected_eee_state
            )
        except BaseException as error:
            control["eee_check_error"] = repr(error)
            status = status or 1
        control["exit_status"] = status
        control["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        atomic_json(options.out_dir / "control.json", control)
    # Reached only when the try body completed without raising: status is 0
    # unless the closing link-state verification failed — propagate that to
    # the exit code so exit-code-gated session automation cannot count a
    # cell with a violated link invariant as clean.
    return status


if __name__ == "__main__":
    sys.exit(main())
