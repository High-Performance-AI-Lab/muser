#!/usr/bin/env python3
"""Seal the frozen eight-prompt Metal DFlash qualification packet."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re

from packet_integrity import collect_unique_packet, publish_new
from release_lock import force_unsealed

ROOT = Path(__file__).resolve().parents[1]
TRACKED_TUNING_FREEZE = ROOT / "release/dflash-tuning-v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--tuning-freeze", type=Path, required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def cv(values: list[int]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)) / mean


def token_digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None


def file_digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def main() -> int:
    args = parse_args()
    expected_tuning = TRACKED_TUNING_FREEZE.resolve()
    tuning_path = args.tuning_freeze.resolve()
    if (
        tuning_path != expected_tuning
        or not tuning_path.is_file()
        or tuning_path.is_symlink()
    ):
        raise SystemExit(
            "--tuning-freeze must be the tracked release/dflash-tuning-v1.json"
        )
    tuning = json.loads(tuning_path.read_text())
    failures: list[str] = []
    verify_length = tuning.get("selected_verify_length")
    if (
        tuning.get("schema") != "muser.dflash-tuning-freeze.v1"
        or tuning.get("status") != "frozen"
        or verify_length not in (3, 7, 15)
    ):
        failures.append("tracked DFlash tuning selection is absent or not frozen")
    records = [
        json.loads(line) for line in args.ledger.read_text().splitlines() if line.strip()
    ]
    expected = {
        f"dflash-{depth}-p{variant}"
        for depth in (512, 2048, 8192, 32768)
        for variant in (1, 2)
    }
    relevant, packet_failures = collect_unique_packet(
        records,
        expected,
        identity=args.identity,
        key=lambda record: (
            record.get("cell") if record.get("engine") == "dflash" else None
        ),
        label="DFlash",
    )
    failures.extend(packet_failures)
    llama_relevant, packet_failures = collect_unique_packet(
        records,
        expected,
        identity=args.identity,
        key=lambda record: (
            record.get("cell") if record.get("engine") == "llama-dflash" else None
        ),
        label="llama-DFlash",
    )
    failures.extend(packet_failures)
    cells: dict[str, object] = {}
    speedups: list[float] = []
    token_digests: set[str] = set()
    for name in sorted(expected):
        record = relevant.get(name)
        llama_record = llama_relevant.get(name)
        if record is None or llama_record is None:
            continue
        fingerprint = record.get("fingerprint", {})
        if (
            record.get("status") != "passed"
            or fingerprint.get("prompt_tokens") != int(name.split("-")[1])
            or fingerprint.get("output_tokens") != 256
            or fingerprint.get("verify_length") != verify_length
            or fingerprint.get("target_backend") != "metal"
            or fingerprint.get("assistant_backend") != "metal"
            or not file_digest(fingerprint.get("prompt_file_sha256"))
            or not token_digest(fingerprint.get("generated_tokens_sha256"))
            or fingerprint.get("sampled_scalar_oracle")
            != "muser-engine-scalar-full-distribution-v1"
            or fingerprint.get("sampled_tokens") != 32
            or not isinstance(fingerprint.get("sampled_seed"), int)
            or fingerprint.get("sampled_temperature_milli") != 800
            or fingerprint.get("sampled_top_p_milli") != 950
            or fingerprint.get("sampled_top_k") != 50
            or not token_digest(fingerprint.get("sampled_generated_tokens_sha256"))
            or not isinstance(fingerprint.get("sampled_drafted_tokens"), int)
            or fingerprint.get("sampled_drafted_tokens", 0) <= 0
        ):
            failures.append(f"{name} has a mixed route or correctness identity")
            continue
        target = record.get("target_only_raw_ns")
        draft = record.get("raw_ns")
        if (
            not isinstance(target, list)
            or len(target) != 5
            or not isinstance(draft, list)
            or len(draft) != 5
            or not all(isinstance(value, int) and value > 0 for value in target + draft)
            or record.get("measurement_order") != [
                ["target-only", "dflash"],
                ["dflash", "target-only"],
                ["dflash", "target-only"],
                ["target-only", "dflash"],
                ["target-only", "dflash"],
            ]
        ):
            failures.append(f"{name} lacks five complete ABBA-ordered paired samples")
            continue
        target_cv = cv(target)
        draft_cv = cv(draft)
        if target_cv > 0.03 or draft_cv > 0.03:
            failures.append(f"{name} is unstable")
            continue
        llama_raw = llama_record.get("raw_ns")
        llama_cv = cv(llama_raw) if isinstance(llama_raw, list) and llama_raw else 1.0
        llama_fingerprint = llama_record.get("fingerprint", {})
        if (
            llama_record.get("status") != "passed"
            or not isinstance(llama_raw, list)
            or len(llama_raw) != 5
            or not all(isinstance(value, int) and value > 0 for value in llama_raw)
            or llama_cv > 0.03
            or llama_fingerprint.get("output_tokens") != 256
            or llama_fingerprint.get("verify_length") != verify_length
            or llama_fingerprint.get("prompt_tokens") != int(name.split("-")[1])
            or llama_fingerprint.get("route") != "llama-draft-dflash"
            or llama_fingerprint.get("prompt_file_sha256")
            != fingerprint.get("prompt_file_sha256")
            or llama_fingerprint.get("generated_tokens_sha256")
            != fingerprint.get("generated_tokens_sha256")
        ):
            failures.append(f"{name} has invalid or mismatched llama-DFlash evidence")
            continue
        paired = [left / right for left, right in zip(target, draft)]
        cell_speedup = geometric_mean(paired)
        if cell_speedup < 1.0:
            failures.append(f"{name} regresses target-only generation")
        muser_median = sorted(draft)[len(draft) // 2]
        llama_median = sorted(llama_raw)[len(llama_raw) // 2]
        versus_llama = llama_median / muser_median
        if versus_llama < 1.0:
            failures.append(f"{name} is slower than llama-DFlash")
        speedups.extend(paired)
        token_digests.add(fingerprint["generated_tokens_sha256"])
        cells[name] = {
            "target_cv": target_cv,
            "dflash_cv": draft_cv,
            "geometric_mean_speedup": cell_speedup,
            "generated_tokens_sha256": fingerprint["generated_tokens_sha256"],
            "sampled_generated_tokens_sha256": fingerprint[
                "sampled_generated_tokens_sha256"
            ],
            "llama_dflash_cv": llama_cv,
            "versus_llama_dflash": versus_llama,
        }
    overall = geometric_mean(speedups) if speedups else None
    if overall is None or overall < 1.10:
        failures.append("Metal DFlash geometric-mean speedup is below 1.10x")
    eligible = not failures and len(cells) == 8
    receipt = {
        "schema": "muser.dflash.seal.v1",
        "status": "passed" if eligible else "failed",
        "identity": args.identity,
        "verify_length": verify_length,
        "cells": cells,
        "geometric_mean_speedup": overall,
        "failures": failures,
        "ledger_sha256": hashlib.sha256(args.ledger.read_bytes()).hexdigest(),
        "tuning_freeze_sha256": hashlib.sha256(tuning_path.read_bytes()).hexdigest(),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "seal_eligible": eligible,
    }
    force_unsealed(receipt, lane="dflash")
    encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    publish_new(args.out, encoded)
    print(encoded, end="")
    return 0 if eligible else 1


if __name__ == "__main__":
    raise SystemExit(main())
