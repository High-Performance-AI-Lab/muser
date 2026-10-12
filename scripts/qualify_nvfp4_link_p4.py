#!/usr/bin/env python3
"""Run the focused five-repetition P4 installed-payload qualification."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from qualify_nvfp4_p4 import show_eee


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cluster-config", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resident", required=True)
    parser.add_argument("--generation-start", type=int, required=True)
    parser.add_argument("--request-prefix", required=True)
    parser.add_argument("--remote-tokens", default="/service/prompt-2048.tokens")
    parser.add_argument("--remote-output-prefix", required=True)
    parser.add_argument("--port-start", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--spark-host", required=True)
    parser.add_argument("--receiver-host", required=True)
    return parser.parse_args()


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def coefficient_of_variation(values: list[float]) -> float:
    return statistics.pstdev(values) / statistics.mean(values)


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing existing aggregate output: {args.output}")
    if args.generation_start <= 0:
        raise RuntimeError("generation start must be positive")

    eee_before = show_eee(args.spark_host)
    repetitions: list[dict[str, Any]] = []
    payload_sha256: str | None = None
    content_sha256: str | None = None
    for repetition in range(5):
        generation = args.generation_start + repetition
        request_id = f"{args.request_prefix}-g{generation}"
        result_path = args.output.with_name(f"{args.output.stem}-g{generation}.json")
        log_path = args.output.with_name(f"{args.output.stem}-g{generation}.log")
        if result_path.exists() or log_path.exists():
            raise RuntimeError(f"refusing existing repetition evidence for {request_id}")
        command = [
            sys.executable,
            "scripts/qualify_nvfp4_streaming.py",
            "--server-binary",
            str(args.server_binary),
            "--model",
            str(args.model),
            "--cluster-config",
            str(args.cluster_config),
            "--tokens",
            str(args.tokens),
            "--output",
            str(result_path),
            "--port",
            str(args.port_start + repetition),
            "--resident",
            args.resident,
            "--generation",
            str(generation),
            "--request-id",
            request_id,
            "--remote-tokens",
            args.remote_tokens,
            "--remote-output",
            f"{args.remote_output_prefix}-g{generation}-client.json",
            "--cache-hit-repetitions",
            "1",
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--spark-host",
            args.spark_host,
            "--receiver-host",
            args.receiver_host,
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        log_path.write_text(completed.stdout + completed.stderr)
        if completed.returncode != 0:
            raise RuntimeError(
                f"{request_id} failed with {completed.returncode}: "
                f"{(completed.stdout + completed.stderr)[-8192:]}"
            )
        result = json.loads(result_path.read_text())
        handoff = result["producer"]["response"]["producer_receipt"]["handoff"]
        transfer = result["transfer"][-1]
        if (
            not handoff.get("ack")
            or handoff.get("payload_wire_source") != "linux-tcp-info-busy-time-v1"
            or handoff.get("payload_pacing_bps") != 4_000_000_000
            or handoff.get("segments") != 16
            or handoff.get("payload_bytes") != transfer.get("bytes_total")
        ):
            raise RuntimeError(f"{request_id} lacks the pinned link receipt contract")
        current_payload = handoff["payload_sha256"]
        current_content = result["first_stream"]["content_sha256"]
        payload_sha256 = payload_sha256 or current_payload
        content_sha256 = content_sha256 or current_content
        if current_payload != payload_sha256 or current_content != content_sha256:
            raise RuntimeError(f"{request_id} changed payload or streamed content")
        wire_ns = int(handoff["payload_wire_ns"])
        payload_bytes = int(handoff["payload_bytes"])
        repetitions.append(
            {
                "repetition": repetition,
                "generation": generation,
                "request_id": request_id,
                "payload_bytes": payload_bytes,
                "payload_wire_ns": wire_ns,
                "installed_payload_gbps": payload_bytes * 8.0 / wire_ns,
                "ttft_ns": int(result["first_stream"]["ttft_ns"]),
                "result": str(result_path),
                "log": str(log_path),
            }
        )

    rates = [row["installed_payload_gbps"] for row in repetitions]
    ttfts = [row["ttft_ns"] for row in repetitions]
    rate_cv = coefficient_of_variation(rates)
    verdict = min(rates) >= 3.0 and rate_cv <= 0.02
    report = {
        "schema": "muser.nvfp4-p4-link-qualification.v1",
        "repetitions": repetitions,
        "installed_payload_gbps": {
            "raw": rates,
            "median": statistics.median(rates),
            "cv": rate_cv,
            "minimum_required": 3.0,
            "maximum_cv": 0.02,
        },
        "remote_serving_ttft_ns": {
            "raw": ttfts,
            "median": statistics.median(ttfts),
            "cv": coefficient_of_variation([float(value) for value in ttfts]),
        },
        "payload_sha256": payload_sha256,
        "content_sha256": content_sha256,
        "payload_wire_source": "linux-tcp-info-busy-time-v1",
        "payload_pacing_bps": 4_000_000_000,
        "eee_before": eee_before,
        "eee_after": show_eee(args.spark_host),
        "green": verdict,
    }
    write_exclusive(args.output, report)
    print(json.dumps({"output": str(args.output), "green": verdict, **report["installed_payload_gbps"]}, sort_keys=True))
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
