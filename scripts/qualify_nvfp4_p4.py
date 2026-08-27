#!/usr/bin/env python3
"""Run the P4 five-repetition target-only cell against a resident Spark producer.

The outer invocation must be serialized by ``accelerator_safe.py``.  This
driver deliberately has no wall-clock deadline for model work: it advances on
process exit, socket readiness, and producer acknowledgements. The migrated
wired fabric requires Mac Ethernet EEE to remain disabled throughout.
"""

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

from qualify_nvfp4_fast import show_eee



def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-model-sha256")
    parser.add_argument("--prompt-token-fixture", type=Path, required=True)
    parser.add_argument("--cluster-config", type=Path, required=True)
    parser.add_argument("--rope-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--resident", default="muser-dudeman-exact-resident-v99")
    parser.add_argument("--remote-token-fixture", default="/service/prompt-2048.tokens")
    parser.add_argument("--first-generation", type=int, default=122)
    parser.add_argument("--receiver-host", required=True)
    parser.add_argument("--receiver-port", type=int, default=29590)
    parser.add_argument("--spark-host", required=True)
    return parser.parse_args()


def run(command: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(
        command,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.rename(path)


def copy_container_receipt(
    spark_host: str, resident: str, remote_path: str, local_path: Path
) -> None:
    command = ["ssh", spark_host, "docker", "exec", resident, "cat", remote_path]
    print("+ " + " ".join(command) + f" > {local_path}", flush=True)
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    temporary = local_path.with_name(f".{local_path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(result.stdout)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.rename(local_path)


def main() -> int:
    options = args()
    if os.environ.get("MUSER_ACCELERATOR_LEASE") != "1":
        raise SystemExit("must run below scripts/accelerator_safe.py")
    options.out_dir.mkdir(parents=True, exist_ok=False)

    receipt_path = options.out_dir / "p4-control.json"
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    control: dict[str, Any] = {
        "schema": "muser.nvfp4-p4-control.v1",
        "identity": options.identity,
        "started_at": started,
        "model_sha256": options.expected_model_sha256 or sha256(options.model),
        "model_sha256_source": (
            "pinned-artifact-receipt"
            if options.expected_model_sha256
            else "computed-for-cell"
        ),
        "prompt_sha256": sha256(options.prompt_token_fixture),
        "cluster_config_sha256": sha256(options.cluster_config),
        "first_generation": options.first_generation,
        "repetitions": 5,
        "output_tokens": 256,
        "producer_receipts": [],
    }
    qualifier: subprocess.Popen[str] | None = None
    status = 1
    try:
        control["eee_before"] = show_eee(options.spark_host, expect="disabled")

        qualifier_command = [
            "target/release/muser-remote-qualify",
            "--model",
            str(options.model),
            "--prompt-token-fixture",
            str(options.prompt_token_fixture),
            "--cluster-config",
            str(options.cluster_config),
            "--variant",
            "text",
            "--repetitions",
            "5",
            "--output-tokens",
            "256",
            "--verify-length",
            "3",
            "--identity",
            options.identity,
            "--p4",
            "--external-producer-receipt-dir",
            str(options.out_dir),
        ]
        environment = os.environ.copy()
        environment["MUSER_CROSS_VENDOR_QK"] = "1"
        environment["MUSER_CROSS_VENDOR_ROPE_CACHE"] = str(options.rope_cache)
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
        for line in qualifier.stdout:
            print(line, end="", flush=True)
            if "phase=remote-target-prefill-export-receive:start" not in line:
                continue
            if generation >= options.first_generation + 5:
                raise RuntimeError("qualifier requested more than five remote repetitions")
            transfer = f"p4-dudeman-2048-g{generation}"
            remote_receipt = f"/service/{transfer}-client.json"
            producer = [
                "ssh",
                options.spark_host,
                "docker",
                "exec",
                options.resident,
                "python3",
                "/workspace/scripts/gx10/vllm/request_producer.py",
                "--sock",
                "/service/producer.sock",
                "--tokens",
                options.remote_token_fixture,
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
            run(producer)
            local_receipt = options.out_dir / f"{transfer}-client.json"
            copy_container_receipt(
                options.spark_host, options.resident, remote_receipt, local_receipt
            )
            control["producer_receipts"].append(
                {"path": str(local_receipt), "sha256": sha256(local_receipt)}
            )
            print(f"producer-complete generation={generation}", flush=True)
            generation += 1

        status = qualifier.wait()
        if status != 0:
            raise RuntimeError(f"qualifier failed with exit status {status}")
        if generation != options.first_generation + 5:
            raise RuntimeError(
                f"qualifier completed after {generation - options.first_generation} remote repetitions"
            )
        return 0
    except BaseException:
        if qualifier is not None and qualifier.poll() is None:
            qualifier.terminate()
            qualifier.wait()
        raise
    finally:
        try:
            control["eee_after"] = show_eee(options.spark_host, expect="disabled")
        except BaseException as error:
            control["eee_check_error"] = repr(error)
            status = status or 1
        control["exit_status"] = status
        control["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        atomic_json(receipt_path, control)


if __name__ == "__main__":
    sys.exit(main())
