#!/usr/bin/env python3
"""Compare exact and native Spark drift-score reports without waiving drift."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def divergence(left: list[int], right: list[int]) -> dict[str, object]:
    size = max(len(left), len(right))
    mismatches = sum(
        index >= len(left) or index >= len(right) or left[index] != right[index]
        for index in range(size)
    )
    first = next(
        (
            index
            for index in range(size)
            if index >= len(left) or index >= len(right) or left[index] != right[index]
        ),
        None,
    )
    return {"rows": size, "mismatches": mismatches, "rate": mismatches / size, "first": first}


def compare_fixture(reference: dict, native: dict) -> dict[str, object]:
    for key in ("id", "regime", "token_count", "token_ids_sha256", "output_tokens"):
        if reference.get(key) != native.get(key):
            raise ValueError(f"fixture {reference.get('id')} differs at {key}")
    reference_logs = reference["target_logprobs"]
    positions = reference.get("scored_positions")
    if positions is None:
        positions = list(range(1, len(reference_logs) + 1))
    if len(positions) != len(reference_logs):
        raise ValueError(f"fixture {reference['id']} reference positions differ")
    native_all_logs = native["target_logprobs"]
    native_all_top = native["teacher_forced_top_token_ids"]
    if any(not isinstance(position, int) or not 1 <= position <= len(native_all_logs) for position in positions):
        raise ValueError(f"fixture {reference['id']} reference position is out of range")
    native_logs = [native_all_logs[position - 1] for position in positions]
    native_top = [native_all_top[position - 1] for position in positions]
    deltas = [float(n) - float(e) for e, n in zip(reference_logs, native_logs)]
    boundaries = reference["boundary_positions"]
    position_indexes = {position: index for index, position in enumerate(positions)}
    boundary_rows = []
    for position in boundaries:
        if position not in position_indexes:
            raise ValueError(f"fixture {reference['id']} boundary is not scored")
        index = position_indexes[position]
        boundary_rows.append(
            {
                "position": position,
                "reference": reference_logs[index],
                "native": native_logs[index],
                "delta": deltas[index],
                "abs_delta": abs(deltas[index]),
            }
        )
    reference_ppl = float(reference["perplexity"])
    native_ppl = math.exp(-sum(float(value) for value in native_logs) / len(native_logs))
    teacher = divergence(
        reference["teacher_forced_top_token_ids"], native_top
    )
    greedy = divergence(reference["generated_tokens"], native["generated_tokens"])
    catastrophic_reasons = []
    relative_ppl = (native_ppl - reference_ppl) / reference_ppl
    if not all(math.isfinite(value) for value in (reference_ppl, native_ppl, relative_ppl)):
        catastrophic_reasons.append("non-finite perplexity")
    if relative_ppl > 0.25:
        catastrophic_reasons.append("relative perplexity regression exceeds 25%")
    if greedy["rate"] > 0.5:
        catastrophic_reasons.append("greedy divergence exceeds 50%")
    return {
        "id": reference["id"],
        "regime": reference["regime"],
        "token_count": reference["token_count"],
        "scored_rows": len(positions),
        "scored_position_first": positions[0],
        "scored_position_last": positions[-1],
        "perplexity": {
            "reference": reference_ppl,
            "native": native_ppl,
            "absolute_delta": native_ppl - reference_ppl,
            "relative_delta": relative_ppl,
        },
        "teacher_forced_greedy": teacher,
        "greedy_stream": greedy,
        "target_logprob_delta": {
            "mean_signed": sum(deltas) / len(deltas),
            "mean_abs": sum(abs(value) for value in deltas) / len(deltas),
            "max_abs": max(abs(value) for value in deltas),
        },
        "boundary_logprobs": boundary_rows,
        "generated_tokens_sha256": {
            "reference": reference["generated_tokens_sha256"],
            "native": native["generated_tokens_sha256"],
        },
        "catastrophic": bool(catastrophic_reasons),
        "catastrophic_reasons": catastrophic_reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    native = json.loads(args.native.read_text())
    if reference.get("producer_mode") not in {"exact", "reference"}:
        parser.error("reference report identity is invalid")
    if native.get("schema") != "muser.spark-nvfp4-drift-score.v1" or native.get("producer_mode") != "native":
        parser.error("native report identity is invalid")
    reference_rows = {row["id"]: row for row in reference["fixtures"]}
    native_rows = {row["id"]: row for row in native["fixtures"]}
    if reference_rows.keys() != native_rows.keys():
        parser.error("reference/native fixture sets differ")
    comparisons = [compare_fixture(reference_rows[key], native_rows[key]) for key in reference_rows]
    regimes = {row["regime"] for row in comparisons}
    required = {"original", "code", "agentic", "long-context"}
    if not required.issubset(regimes):
        parser.error(f"drift packet lacks regimes {sorted(required - regimes)}")
    report = {
        "schema": "muser.nvfp4-fast-drift-envelope.v1",
        "status": "catastrophic" if any(row["catastrophic"] for row in comparisons) else "measured",
        "thresholds_declared_before_run": {
            "relative_perplexity_regression": 0.25,
            "greedy_divergence_rate": 0.5,
            "non_finite": "catastrophic",
        },
        "baseline_mode": reference["producer_mode"],
        "reference_report_sha256": sha256(args.reference),
        "native_report_sha256": sha256(args.native),
        "comparisons": comparisons,
        "seal_eligible": False,
    }
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}, sort_keys=True))
    if report["status"] == "catastrophic":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
