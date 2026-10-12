#!/usr/bin/env python3
"""Select the native NVFP4 context cap from the frozen routing ladder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aligned(row: dict[str, Any], positions: list[int], key: str) -> list[Any]:
    values = row[key]
    return [values[position - 1] for position in positions]


def ppl(logprobs: list[float]) -> float:
    return math.exp(-math.fsum(float(value) for value in logprobs) / len(logprobs))


def compare(reference: dict[str, Any], native: dict[str, Any]) -> dict[str, Any]:
    for key in ("id", "regime", "token_count", "token_ids_sha256"):
        if reference.get(key) != native.get(key):
            raise ValueError(f"routing fixture differs at {key}")
    positions = [int(value) for value in reference["scored_positions"]]
    native_logs = [float(value) for value in aligned(native, positions, "target_logprobs")]
    native_top = [int(value) for value in aligned(native, positions, "teacher_forced_top_token_ids")]
    reference_logs = [float(value) for value in reference["target_logprobs"]]
    reference_top = [int(value) for value in reference["teacher_forced_top_token_ids"]]
    if not positions or len(positions) != len(reference_logs) or len(reference_logs) != len(reference_top):
        raise ValueError("routing reference rows differ")
    reference_ppl = ppl(reference_logs)
    native_ppl = ppl(native_logs)
    disagreements = sum(left != right for left, right in zip(reference_top, native_top))
    rate = disagreements / len(positions)
    relative_ppl = native_ppl / reference_ppl - 1.0
    passed = relative_ppl <= 0.05 and rate <= 0.15
    return {
        "id": reference["id"],
        "token_count": reference["token_count"],
        "scored_rows": len(positions),
        "perplexity": {
            "kquant": reference_ppl,
            "native": native_ppl,
            "relative_delta": relative_ppl,
        },
        "top_token_disagreement": {
            "mismatches": disagreements,
            "rate": rate,
        },
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kquant", required=True, type=Path)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--decomposition", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reference = json.loads(args.kquant.read_text(encoding="utf-8"))
    native = json.loads(args.native.read_text(encoding="utf-8"))
    decomposition = json.loads(args.decomposition.read_text(encoding="utf-8"))
    reference_rows = {row["id"]: row for row in reference["fixtures"]}
    native_rows = {row["id"]: row for row in native["fixtures"]}
    required = {"long-8192", "long-16384", "long-32768"}
    if reference_rows.keys() != required or native_rows.keys() != required:
        parser.error("routing ladder must contain exactly 8k, 16k, and 32k")
    comparisons = sorted(
        (compare(reference_rows[key], native_rows[key]) for key in required),
        key=lambda row: row["token_count"],
    )
    material = decomposition.get("long_context_fast_path_material") is True
    contiguous_passing = []
    for row in comparisons:
        if not row["passed"]:
            break
        contiguous_passing.append(row["token_count"])
    context_cap = max(contiguous_passing) if contiguous_passing and material else None
    status = "routing-required" if context_cap is not None else "passed-no-cap"
    if material and not contiguous_passing:
        status = "quality-blocker"
    report = {
        "schema": "muser.nvfp4-context-routing.v1",
        "status": status,
        "thresholds_preregistered": {
            "relative_ppl_regression": 0.05,
            "per_step_top_token_disagreement": 0.15,
        },
        "inputs": {
            "kquant_sha256": sha256(args.kquant),
            "native_sha256": sha256(args.native),
            "decomposition_sha256": sha256(args.decomposition),
        },
        "long_context_fast_path_material": material,
        "comparisons": comparisons,
        "native_context_cap_tokens": context_cap,
        "above_cap_route": "kquant" if context_cap is not None else None,
        "cap_requires_all_lower_tested_rungs_to_pass": True,
        "seal_eligible": False,
    }
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": status, "context_cap": context_cap, "output": str(args.output)}))
    if status == "quality-blocker":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
