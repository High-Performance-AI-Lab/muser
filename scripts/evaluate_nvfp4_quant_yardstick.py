#!/usr/bin/env python3
"""Calibrate native NVFP4 drift against an aligned quant-vs-quant yardstick."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any


BOOTSTRAP_REPLICATES = 2_000
DISAGREEMENT_MARGIN = 0.02
PPL_MARGIN = 0.01
PPL_FLOOR = 0.05
Z_95 = 1.959963984540054


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def report_rows(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in report["fixtures"]}


def wilson_interval(mismatches: int, rows: int) -> tuple[float, float]:
    if rows <= 0 or not 0 <= mismatches <= rows:
        raise ValueError("invalid disagreement counts")
    rate = mismatches / rows
    denominator = 1.0 + Z_95 * Z_95 / rows
    center = (rate + Z_95 * Z_95 / (2.0 * rows)) / denominator
    spread = (
        Z_95
        * math.sqrt(rate * (1.0 - rate) / rows + Z_95 * Z_95 / (4.0 * rows * rows))
        / denominator
    )
    return max(0.0, center - spread), min(1.0, center + spread)


def paired_ppl_bootstrap(
    baseline_logs: list[float], alternate_logs: list[float], fixture_id: str
) -> tuple[float, float]:
    if len(baseline_logs) != len(alternate_logs) or not baseline_logs:
        raise ValueError("paired logprob rows differ")
    # PPL_alt / PPL_base = exp(mean(logp_base - logp_alt)).
    deltas = [left - right for left, right in zip(baseline_logs, alternate_logs)]
    seed = int.from_bytes(hashlib.sha256(fixture_id.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    rows = len(deltas)
    samples = []
    for _ in range(BOOTSTRAP_REPLICATES):
        mean_delta = math.fsum(deltas[rng.randrange(rows)] for _ in range(rows)) / rows
        samples.append(math.expm1(mean_delta))
    samples.sort()
    low = samples[math.floor(0.025 * (BOOTSTRAP_REPLICATES - 1))]
    high = samples[math.ceil(0.975 * (BOOTSTRAP_REPLICATES - 1))]
    return low, high


def aligned(row: dict[str, Any], positions: list[int], key: str) -> list[Any]:
    values = row[key]
    if any(position < 1 or position > len(values) for position in positions):
        raise ValueError(f"fixture {row['id']} lacks aligned {key} rows")
    return [values[position - 1] for position in positions]


def perplexity(logprobs: list[float]) -> float:
    return math.exp(-math.fsum(logprobs) / len(logprobs))


def compare(
    baseline: dict[str, Any], alternate: dict[str, Any], native: dict[str, Any] | None
) -> dict[str, Any]:
    for key in ("id", "regime", "token_count", "token_ids_sha256"):
        if baseline.get(key) != alternate.get(key):
            raise ValueError(f"fixture {baseline.get('id')} differs at {key}")
    positions = [int(value) for value in baseline["scored_positions"]]
    if positions != [int(value) for value in alternate["scored_positions"]]:
        raise ValueError(f"fixture {baseline['id']} scored positions differ")
    baseline_logs = [float(value) for value in baseline["target_logprobs"]]
    alternate_logs = [float(value) for value in alternate["target_logprobs"]]
    baseline_top = [int(value) for value in baseline["teacher_forced_top_token_ids"]]
    alternate_top = [int(value) for value in alternate["teacher_forced_top_token_ids"]]
    if not positions or not (
        len(positions)
        == len(baseline_logs)
        == len(alternate_logs)
        == len(baseline_top)
        == len(alternate_top)
    ):
        raise ValueError(f"fixture {baseline['id']} aligned rows differ")
    mismatches = sum(left != right for left, right in zip(baseline_top, alternate_top))
    rate = mismatches / len(positions)
    wilson_low, wilson_high = wilson_interval(mismatches, len(positions))
    baseline_ppl = perplexity(baseline_logs)
    alternate_ppl = perplexity(alternate_logs)
    relative_ppl = alternate_ppl / baseline_ppl - 1.0
    bootstrap_low, bootstrap_high = paired_ppl_bootstrap(
        baseline_logs, alternate_logs, baseline["id"]
    )
    top_gate = min(1.0, wilson_high + DISAGREEMENT_MARGIN)
    ppl_gate = max(
        PPL_FLOOR,
        max(abs(bootstrap_low), abs(bootstrap_high)) + PPL_MARGIN,
    )
    result: dict[str, Any] = {
        "id": baseline["id"],
        "regime": baseline["regime"],
        "token_count": baseline["token_count"],
        "scored_rows": len(positions),
        "scored_position_first": positions[0],
        "scored_position_last": positions[-1],
        "yardstick": {
            "top_token_disagreement": {
                "mismatches": mismatches,
                "rate": rate,
                "wilson_95": [wilson_low, wilson_high],
            },
            "relative_perplexity_delta": relative_ppl,
            "paired_row_bootstrap_95": [bootstrap_low, bootstrap_high],
        },
        "calibrated_gates": {
            "top_token_disagreement": top_gate,
            "positive_relative_perplexity_regression": ppl_gate,
        },
        "provisional_15_percent_is_no_more_permissive": 0.15 <= top_gate,
    }
    if native is not None:
        for key in ("id", "regime", "token_count", "token_ids_sha256"):
            if baseline.get(key) != native.get(key):
                raise ValueError(f"native fixture {baseline['id']} differs at {key}")
        native_logs = [float(value) for value in aligned(native, positions, "target_logprobs")]
        native_top = [
            int(value) for value in aligned(native, positions, "teacher_forced_top_token_ids")
        ]
        native_mismatches = sum(
            left != right for left, right in zip(baseline_top, native_top)
        )
        native_rate = native_mismatches / len(positions)
        native_relative_ppl = perplexity(native_logs) / baseline_ppl - 1.0
        result["native_vs_kquant"] = {
            "top_token_disagreement": {
                "mismatches": native_mismatches,
                "rate": native_rate,
            },
            "relative_perplexity_delta": native_relative_ppl,
            "top_token_passed": native_rate <= top_gate,
            "perplexity_passed": native_relative_ppl <= ppl_gate,
            "passed": native_rate <= top_gate and native_relative_ppl <= ppl_gate,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kquant", required=True, type=Path)
    parser.add_argument("--alternate", required=True, type=Path)
    parser.add_argument("--native", action="append", type=Path, default=[])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    baseline_report = json.loads(args.kquant.read_text(encoding="utf-8"))
    alternate_report = json.loads(args.alternate.read_text(encoding="utf-8"))
    baseline = report_rows(baseline_report)
    alternate = report_rows(alternate_report)
    if baseline.keys() != alternate.keys():
        parser.error("kquant and alternate fixture sets differ")
    native: dict[str, dict[str, Any]] = {}
    for path in args.native:
        for fixture_id, row in report_rows(json.loads(path.read_text(encoding="utf-8"))).items():
            if fixture_id not in baseline:
                continue
            if fixture_id in native:
                parser.error(f"duplicate native fixture: {fixture_id}")
            native[fixture_id] = row
    if args.native and native.keys() != baseline.keys():
        parser.error(f"native reports lack fixtures: {sorted(baseline.keys() - native.keys())}")
    comparisons = [
        compare(baseline[fixture_id], alternate[fixture_id], native.get(fixture_id))
        for fixture_id in baseline
    ]
    report = {
        "schema": "muser.nvfp4-quant-yardstick.v1",
        "status": "measured",
        "calibration_preregistered": {
            "confidence": "two-sided-95-percent",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_unit": "paired-scored-row",
            "disagreement_margin": DISAGREEMENT_MARGIN,
            "positive_relative_ppl_floor": PPL_FLOOR,
            "relative_ppl_margin": PPL_MARGIN,
        },
        "inputs": {
            "kquant_sha256": sha256(args.kquant),
            "alternate_sha256": sha256(args.alternate),
            "native_sha256": [sha256(path) for path in args.native],
        },
        "model_identity": {
            "kquant": baseline_report.get("model_sha256"),
            "alternate": alternate_report.get("model_sha256"),
            "alternate_lane": alternate_report.get("reference_lane"),
        },
        "comparisons": comparisons,
        "seal_eligible": False,
    }
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"output": str(args.output), "fixtures": len(comparisons)}))


if __name__ == "__main__":
    main()
