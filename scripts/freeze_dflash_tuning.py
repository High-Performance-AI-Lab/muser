#!/usr/bin/env python3
"""Freeze disjoint unsealed DFlash tuning evidence into release identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "release/dflash-tuning-v1.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=TARGET)
    args = parser.parse_args()
    try:
        receipt_path = args.receipt.resolve()
        output = args.out.resolve()
        if output != TARGET.resolve():
            raise RuntimeError("tuning freeze output must be release/dflash-tuning-v1.json")
        if not receipt_path.is_file() or receipt_path.is_symlink():
            raise RuntimeError("tuning receipt is missing or unsafe")
        receipt_bytes = receipt_path.read_bytes()
        receipt = json.loads(receipt_bytes)
        selected = receipt.get("selected_verify_length")
        if (
            receipt.get("schema") != "muser.unsealed-qualification.v1"
            or receipt.get("lane") != "dflash-tuning"
            or receipt.get("status") != "passed"
            or receipt.get("seal_eligible") is not False
            or receipt.get("would_be_seal_eligible") is not True
            or selected not in (3, 7, 15)
            or not isinstance(receipt.get("identity"), str)
            or not receipt["identity"]
        ):
            raise RuntimeError(
                "receipt is not passing disjoint unsealed DFlash tuning evidence"
            )
        scores = receipt.get("geometric_mean_speedup_by_verify_length")
        if not isinstance(scores, dict) or set(scores) != {"3", "7", "15"}:
            raise RuntimeError("tuning receipt does not contain every candidate score")
        if not all(
            isinstance(value, (int, float)) and value > 0 for value in scores.values()
        ):
            raise RuntimeError("tuning receipt contains an invalid candidate score")
        expected = max((3, 7, 15), key=lambda value: (scores[str(value)], -value))
        if selected != expected:
            raise RuntimeError("tuning selection differs from the deterministic policy")
        frozen = {
            "schema": "muser.dflash-tuning-freeze.v1",
            "status": "frozen",
            "selected_verify_length": selected,
            "allowed_verify_lengths": [3, 7, 15],
            "selection_method": (
                "highest geometric mean speedup over the disjoint 256/4096 by p1/p2 "
                "tuning packet; ties prefer the shorter verify length"
            ),
            "source_evidence": {
                "apparatus_identity": receipt["identity"],
                "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
                "ledger_sha256": receipt.get("ledger_sha256"),
                "scores": scores,
            },
        }
        temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(frozen, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
        descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        print(json.dumps(frozen, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as error:
        print(f"DFlash tuning freeze failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
