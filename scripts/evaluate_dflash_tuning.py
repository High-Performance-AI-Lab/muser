#!/usr/bin/env python3
"""Freeze the best DFlash verification length from the disjoint tuning packet."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    records = [
        json.loads(line)
        for line in args.ledger.read_text().splitlines()
        if line.strip()
    ]
    expected = {
        f"dflash-tune-{depth}-p{variant}-v{verify}"
        for depth in (256, 4096)
        for variant in (1, 2)
        for verify in (3, 7, 15)
    }
    relevant, failures = collect_unique_packet(
        records,
        expected,
        identity=args.identity,
        key=lambda record: (
            record.get("cell") if record.get("engine") == "dflash" else None
        ),
        label="DFlash tuning",
    )
    scores: dict[int, float] = {}
    for verify in (3, 7, 15):
        cells = [
            relevant[cell]
            for cell in sorted(expected)
            if cell.endswith(f"-v{verify}") and cell in relevant
        ]
        speedups: list[float] = []
        for record in cells:
            if record.get("status") != "passed":
                failures.append(f"{record.get('cell')} did not pass")
                continue
            fingerprint = record.get("fingerprint", {})
            depth = int(str(record.get("cell")).split("-")[2])
            if (
                fingerprint.get("prompt_tokens") != depth
                or fingerprint.get("output_tokens") != 256
                or fingerprint.get("verify_length") != verify
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
                failures.append(f"{record.get('cell')} has a mixed or incomplete route")
                continue
            draft = record.get("raw_ns")
            target = record.get("target_only_raw_ns")
            if (
                not isinstance(draft, list)
                or not isinstance(target, list)
                or len(draft) != 3
                or len(target) != 3
                or not all(
                    isinstance(value, int) and value > 0 for value in draft + target
                )
            ):
                failures.append(f"{record.get('cell')} lacks three paired raw samples")
                continue
            if cv(draft) > 0.03 or cv(target) > 0.03:
                failures.append(f"{record.get('cell')} is unstable")
                continue
            speedups.extend(left / right for left, right in zip(target, draft))
        if len(speedups) == 12:
            scores[verify] = geometric_mean(speedups)

    selected = max(scores, key=lambda verify: (scores[verify], -verify)) if scores else None
    receipt = {
        "schema": "muser.dflash-tuning.receipt.v1",
        "status": "passed" if not failures and selected is not None else "failed",
        "identity": args.identity,
        "selected_verify_length": selected,
        "geometric_mean_speedup_by_verify_length": {
            str(key): value for key, value in sorted(scores.items())
        },
        "cells": len(relevant),
        "failures": failures,
        "ledger_sha256": sha256(args.ledger),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "seal_eligible": not failures and selected is not None,
    }
    force_unsealed(receipt, lane="dflash-tuning")
    encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    publish_new(args.out, encoded)
    print(encoded, end="")
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
