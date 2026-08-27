#!/usr/bin/env python3
"""Attribute native NVFP4 drift across artifact and fast-path effects."""

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


def disagreement(left: list[int], right: list[int]) -> dict[str, Any]:
    if len(left) != len(right) or not left:
        raise ValueError("top-token rows differ")
    mismatches = sum(a != b for a, b in zip(left, right))
    return {"rows": len(left), "mismatches": mismatches, "rate": mismatches / len(left)}


def aligned(row: dict[str, Any], positions: list[int], key: str) -> list[Any]:
    values = row[key]
    if any(position < 1 or position > len(values) for position in positions):
        raise ValueError(f"fixture {row['id']} lacks aligned {key} rows")
    return [values[position - 1] for position in positions]


def ppl(logprobs: list[float]) -> float:
    return math.exp(-math.fsum(float(value) for value in logprobs) / len(logprobs))


def attribution_classes(kquant: list[int], exact: list[int], native: list[int]) -> dict[str, Any]:
    counts = {
        "all_equal": 0,
        "artifact_only": 0,
        "fast_path_only": 0,
        "compensated": 0,
        "compounded": 0,
    }
    for k_value, e_value, n_value in zip(kquant, exact, native):
        if k_value == e_value == n_value:
            key = "all_equal"
        elif k_value != e_value and e_value == n_value:
            key = "artifact_only"
        elif k_value == e_value and e_value != n_value:
            key = "fast_path_only"
        elif k_value == n_value and k_value != e_value:
            key = "compensated"
        else:
            key = "compounded"
        counts[key] += 1
    rows = len(kquant)
    return {key: {"count": value, "rate": value / rows} for key, value in counts.items()}


def compare_fixture(
    reference: dict[str, Any],
    exact: dict[str, Any],
    native: dict[str, Any],
    weight_only: dict[str, Any] | None,
) -> dict[str, Any]:
    for key in ("id", "regime", "token_count", "token_ids_sha256"):
        if not reference.get(key) == exact.get(key) == native.get(key):
            raise ValueError(f"fixture identity differs at {key}")
    positions = [int(value) for value in reference["scored_positions"]]
    k_logs = [float(value) for value in reference["target_logprobs"]]
    k_top = [int(value) for value in reference["teacher_forced_top_token_ids"]]
    if len(positions) != len(k_logs) or len(k_logs) != len(k_top):
        raise ValueError("reference scored rows differ")
    e_logs = [float(value) for value in aligned(exact, positions, "target_logprobs")]
    n_logs = [float(value) for value in aligned(native, positions, "target_logprobs")]
    e_top = [int(value) for value in aligned(exact, positions, "teacher_forced_top_token_ids")]
    n_top = [int(value) for value in aligned(native, positions, "teacher_forced_top_token_ids")]
    k_ppl, e_ppl, n_ppl = ppl(k_logs), ppl(e_logs), ppl(n_logs)
    artifact_log = math.log(e_ppl / k_ppl)
    fast_log = math.log(n_ppl / e_ppl)
    total_log = math.log(n_ppl / k_ppl)
    pairwise = {
        "kquant_vs_exact": disagreement(k_top, e_top),
        "exact_vs_native": disagreement(e_top, n_top),
        "kquant_vs_native": disagreement(k_top, n_top),
    }
    top_gate = 0.15 if reference["regime"] == "long-context" else 0.05
    total_relative = n_ppl / k_ppl - 1.0
    gates = {
        "relative_ppl_regression": {
            "limit": 0.05,
            "value": total_relative,
            "passed": total_relative <= 0.05,
        },
        "per_step_top_token_disagreement": {
            "limit": top_gate,
            "value": pairwise["kquant_vs_native"]["rate"],
            "passed": pairwise["kquant_vs_native"]["rate"] <= top_gate,
        },
    }
    result: dict[str, Any] = {
        "id": reference["id"],
        "regime": reference["regime"],
        "token_count": reference["token_count"],
        "scored_rows": len(positions),
        "scored_position_first": positions[0],
        "scored_position_last": positions[-1],
        "perplexity": {
            "kquant": k_ppl,
            "exact_redhat": e_ppl,
            "native_w4a4": n_ppl,
            "artifact_relative": e_ppl / k_ppl - 1.0,
            "fast_path_relative": n_ppl / e_ppl - 1.0,
            "total_relative": total_relative,
            "artifact_log_ratio": artifact_log,
            "fast_path_log_ratio": fast_log,
            "total_log_ratio": total_log,
            "log_additivity_error": total_log - artifact_log - fast_log,
        },
        "top_token_disagreement": pairwise,
        "top_token_attribution": attribution_classes(k_top, e_top, n_top),
        "gates": gates,
        "passed": all(value["passed"] for value in gates.values()),
    }
    if weight_only is not None:
        if weight_only["id"] != reference["id"]:
            raise ValueError("weight-only fixture identity differs")
        w_logs = [float(value) for value in aligned(weight_only, positions, "target_logprobs")]
        w_top = [int(value) for value in aligned(weight_only, positions, "teacher_forced_top_token_ids")]
        w_ppl = ppl(w_logs)
        result["weight_only_supporting_comparison"] = {
            "perplexity": w_ppl,
            "relative_to_w4a4": w_ppl / n_ppl - 1.0,
            "top_token_disagreement_vs_w4a4": disagreement(w_top, n_top),
            "artifact_confounded": True,
        }
    return result


def rows(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in report["fixtures"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kquant", required=True, type=Path)
    parser.add_argument("--exact", required=True, type=Path)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--weight-only", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in (args.kquant, args.exact, args.native)]
    reference_rows, exact_rows, native_rows = map(rows, reports)
    if not reference_rows.keys() == exact_rows.keys() == native_rows.keys():
        parser.error("fixture sets differ")
    weight_report = json.loads(args.weight_only.read_text(encoding="utf-8")) if args.weight_only else None
    weight_rows = rows(weight_report) if weight_report else {}
    if weight_report and weight_rows.keys() != reference_rows.keys():
        parser.error("weight-only fixture set differs")
    comparisons = [
        compare_fixture(
            reference_rows[fixture_id],
            exact_rows[fixture_id],
            native_rows[fixture_id],
            weight_rows.get(fixture_id),
        )
        for fixture_id in reference_rows
    ]
    long_rows = [row for row in comparisons if row["regime"] == "long-context"]
    if len(long_rows) != 1:
        parser.error("exactly one reduced long-context fixture is required")
    long_row = long_rows[0]
    residual_material = (
        abs(long_row["perplexity"]["fast_path_relative"]) > 0.02
        or long_row["top_token_disagreement"]["exact_vs_native"]["rate"] > 0.05
    )
    report = {
        "schema": "muser.nvfp4-drift-decomposition.v1",
        "status": "passed" if all(row["passed"] for row in comparisons) else "quality-blocker",
        "thresholds_preregistered": {
            "relative_ppl_regression": 0.05,
            "per_step_disagreement_default": 0.05,
            "per_step_disagreement_long_context": 0.15,
            "long_context_fast_path_material_ppl": 0.02,
            "long_context_fast_path_material_disagreement": 0.05,
        },
        "inputs": {
            "kquant_sha256": sha256(args.kquant),
            "exact_sha256": sha256(args.exact),
            "native_sha256": sha256(args.native),
            "weight_only_sha256": sha256(args.weight_only) if args.weight_only else None,
        },
        "comparisons": comparisons,
        "long_context_fast_path_material": residual_material,
        "seal_eligible": False,
    }
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": report["status"], "output": str(args.output)}, sort_keys=True))
    if report["status"] == "quality-blocker":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
