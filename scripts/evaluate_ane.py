#!/usr/bin/env python3
"""Seal the complete public-CoreML ANE versus Metal-DFlash packet."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

from packet_integrity import collect_unique_packet, publish_new
from release_lock import force_unsealed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def main() -> int:
    args = parse_args()
    records = [
        json.loads(line) for line in args.ledger.read_text().splitlines() if line.strip()
    ]
    expected = {
        f"ane-{depth}-p{variant}" for depth in (512, 2048, 8192, 32768) for variant in (1, 2)
    }
    relevant, failures = collect_unique_packet(
        records,
        expected,
        identity=args.identity,
        key=lambda record: record.get("cell") if record.get("engine") == "ane" else None,
        label="ANE",
    )
    speedups: list[float] = []
    all_metal_verify = 0
    all_ane_verify = 0
    cells: dict[str, object] = {}
    fingerprints: list[dict] = []
    for name in sorted(expected):
        record = relevant.get(name)
        if record is None:
            continue
        if record.get("status") != "passed":
            failures.append(f"{name} did not pass")
            continue
        if (
            record.get("cv", 1.0) > 0.03
            or record.get("metal_dflash_cv", 1.0) > 0.03
            or record.get("target_only_cv", 1.0) > 0.03
        ):
            failures.append(f"{name} is unstable")
            continue
        values = record.get("speedups")
        taxes = record.get("verification_taxes")
        metal_raw = record.get("metal_dflash_raw_ns")
        ane_raw = record.get("raw_ns")
        metal_verify = record.get("metal_target_verify_ns")
        ane_verify = record.get("ane_target_verify_ns")
        if (
            not isinstance(values, list) or len(values) != 3
            or not all(isinstance(value, (int, float)) and value > 0 for value in values)
            or not isinstance(taxes, list) or len(taxes) != 3
            or not all(isinstance(value, (int, float)) for value in taxes)
            or not isinstance(metal_raw, list) or len(metal_raw) != 3
            or not isinstance(ane_raw, list) or len(ane_raw) != 3
            or not isinstance(metal_verify, list) or len(metal_verify) != 3
            or not isinstance(ane_verify, list) or len(ane_verify) != 3
            or not all(
                isinstance(value, int) and value > 0 for value in metal_verify + ane_verify
            )
        ):
            failures.append(f"{name} has invalid paired evidence")
            continue
        cell_speedup = geometric_mean([float(value) for value in values])
        if cell_speedup < 1.10:
            failures.append(f"{name} ANE/Metal speedup is only {cell_speedup:.6f}x")
        speedups.extend(float(value) for value in values)
        verification_ratio = sum(ane_verify) / sum(metal_verify)
        if verification_ratio > 1.02:
            failures.append(
                f"{name} target-verification tax is {(verification_ratio - 1.0):.6%}"
            )
        all_metal_verify += sum(metal_verify)
        all_ane_verify += sum(ane_verify)
        fingerprint = record.get("fingerprint")
        if not isinstance(fingerprint, dict) or fingerprint.get("compute_units") != "CPU_AND_NE":
            failures.append(f"{name} has no CPU_AND_NE identity")
            continue
        fingerprints.append(fingerprint)
        cells[name] = {
            "speedup": cell_speedup,
            "verification_tax": verification_ratio - 1.0,
            "ane_cv": record.get("cv"),
            "metal_cv": record.get("metal_dflash_cv"),
            "target_cv": record.get("target_only_cv"),
        }
    for key in (
        "target_identity", "dflash_identity", "manifest_sha256",
        "compute_plan_receipt_sha256", "compute_units", "verify_length",
    ):
        if len({fingerprint.get(key) for fingerprint in fingerprints}) > 1:
            failures.append(f"ANE packet mixed {key}")
    overall = geometric_mean(speedups) if speedups else None
    if overall is not None and overall < 1.10:
        failures.append(f"ANE packet geometric-mean speedup is only {overall:.6f}x")
    receipt = {
        "schema": "muser.ane.seal.v1",
        "status": "passed" if not failures and len(cells) == 8 else "failed",
        "identity": args.identity,
        "cells": cells,
        "geometric_mean_speedup": overall,
        "mean_target_verification_tax": (
            all_ane_verify / all_metal_verify - 1.0 if all_metal_verify else None
        ),
        "failures": failures,
        "ledger_sha256": hashlib.sha256(args.ledger.read_bytes()).hexdigest(),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "seal_eligible": not failures and len(cells) == 8,
    }
    force_unsealed(receipt, lane="ane")
    encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    publish_new(args.out, encoded)
    print(encoded, end="")
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
