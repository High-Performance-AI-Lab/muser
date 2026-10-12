#!/usr/bin/env python3
"""Evaluate nested content controls and select the E-series routing branch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


POSITION_BIN_TOKENS = 512


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in report["fixtures"]}


def aligned(row: dict[str, Any], positions: list[int], key: str) -> list[Any]:
    values = row[key]
    if any(position < 1 or position > len(values) for position in positions):
        raise ValueError(f"fixture {row['id']} lacks aligned {key} rows")
    return [values[position - 1] for position in positions]


def perplexity(logprobs: list[float]) -> float:
    return math.exp(-math.fsum(logprobs) / len(logprobs))


def fixture_identity(fixture_id: str) -> tuple[str, int]:
    prefix, separator, raw_length = fixture_id.rpartition("-")
    if not separator or not prefix.startswith("e2-") or not raw_length.isdigit():
        raise ValueError(f"invalid E2 fixture id: {fixture_id}")
    return prefix.removeprefix("e2-"), int(raw_length)


def yardstick_by_length(report: dict[str, Any]) -> dict[int, dict[str, float]]:
    bands = {}
    for row in report["comparisons"]:
        if row["regime"] != "long-context":
            continue
        token_count = int(row["token_count"])
        if token_count not in {8192, 16384, 32768}:
            continue
        gates = row["calibrated_gates"]
        bands[token_count] = {
            "top": float(gates["top_token_disagreement"]),
            "ppl": float(gates["positive_relative_perplexity_regression"]),
        }
    if bands.keys() != {8192, 16384, 32768}:
        raise ValueError("yardstick lacks the 8k/16k/32k bands")
    return bands


def select_band(bands: dict[int, dict[str, float]], length: int) -> tuple[int, dict[str, float]]:
    selected = next((value for value in sorted(bands) if value >= length), max(bands))
    return selected, bands[selected]


def compare(
    baseline: dict[str, Any], native: dict[str, Any], bands: dict[int, dict[str, float]]
) -> dict[str, Any]:
    for key in ("id", "regime", "token_count", "token_ids_sha256"):
        if baseline.get(key) != native.get(key):
            raise ValueError(f"fixture {baseline.get('id')} differs at {key}")
    document, length = fixture_identity(baseline["id"])
    if length != baseline["token_count"]:
        raise ValueError(f"fixture {baseline['id']} length differs")
    positions = [int(value) for value in baseline["scored_positions"]]
    baseline_logs = [float(value) for value in baseline["target_logprobs"]]
    baseline_top = [int(value) for value in baseline["teacher_forced_top_token_ids"]]
    native_logs = [float(value) for value in aligned(native, positions, "target_logprobs")]
    native_top = [int(value) for value in aligned(native, positions, "teacher_forced_top_token_ids")]
    if not positions or len(positions) != len(baseline_logs) or len(positions) != len(baseline_top):
        raise ValueError(f"fixture {baseline['id']} scored rows differ")
    band_length, gate = select_band(bands, length)
    mismatch_flags = [left != right for left, right in zip(baseline_top, native_top)]
    mismatches = sum(mismatch_flags)
    top_rate = mismatches / len(positions)
    relative_ppl = perplexity(native_logs) / perplexity(baseline_logs) - 1.0
    bins: dict[int, dict[str, int]] = {}
    for position, mismatch in zip(positions, mismatch_flags):
        start = ((position - 1) // POSITION_BIN_TOKENS) * POSITION_BIN_TOKENS + 1
        value = bins.setdefault(start, {"rows": 0, "mismatches": 0})
        value["rows"] += 1
        value["mismatches"] += int(mismatch)
    bin_rows = [
        {
            "position_first": start,
            "position_last": min(start + POSITION_BIN_TOKENS - 1, length - 1),
            "rows": value["rows"],
            "mismatches": value["mismatches"],
            "rate": value["mismatches"] / value["rows"],
            "exceeds_cell_gate": value["mismatches"] / value["rows"] > gate["top"],
        }
        for start, value in sorted(bins.items())
    ]
    top_passed = top_rate <= gate["top"]
    ppl_passed = relative_ppl <= gate["ppl"]
    return {
        "id": baseline["id"],
        "document": document,
        "token_count": length,
        "scored_rows": len(positions),
        "scored_position_first": positions[0],
        "scored_position_last": positions[-1],
        "yardstick_band_tokens": band_length,
        "calibrated_gates": {
            "top_token_disagreement": gate["top"],
            "positive_relative_perplexity_regression": gate["ppl"],
        },
        "top_token_disagreement": {"mismatches": mismatches, "rate": top_rate},
        "relative_perplexity_delta": relative_ppl,
        "position_bins": bin_rows,
        "sensitive_position_bin_starts": [
            row["position_first"] for row in bin_rows if row["exceeds_cell_gate"]
        ],
        "top_token_passed": top_passed,
        "perplexity_passed": ppl_passed,
        "passed": top_passed and ppl_passed,
    }


def route(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    documents = sorted({row["document"] for row in comparisons})
    lengths = sorted({row["token_count"] for row in comparisons})
    if len(documents) < 3 or lengths != [2048, 4096, 8192, 16384, 32768]:
        raise ValueError("content control requires three documents and the full nested ladder")
    summaries = []
    for length in lengths:
        selected = [row for row in comparisons if row["token_count"] == length]
        if {row["document"] for row in selected} != set(documents):
            raise ValueError(f"content control length {length} lacks documents")
        failures = [row for row in selected if not row["passed"]]
        replicated = len(failures) >= 2
        summaries.append(
            {
                "token_count": length,
                "failed_documents": [row["document"] for row in failures],
                "failed_document_count": len(failures),
                "replicated_exceedance": replicated,
                "content_local": len(failures) == 1,
            }
        )
    first_effect_index = next(
        (
            index
            for index in range(len(summaries) - 1)
            if summaries[index]["replicated_exceedance"]
            and summaries[index + 1]["replicated_exceedance"]
        ),
        None,
    )
    for index, summary in enumerate(summaries):
        summary["persistent_at_next_length"] = (
            index + 1 < len(summaries)
            and summary["replicated_exceedance"]
            and summaries[index + 1]["replicated_exceedance"]
        )
    any_failures = any(summary["failed_document_count"] for summary in summaries)
    if first_effect_index is None:
        return {
            "status": "no-cap",
            "branch": "content-sensitive-envelope" if any_failures else "inside-yardstick-band",
            "native_context_cap_tokens": None,
            "above_cap_route": None,
            "lengths": summaries,
        }
    if first_effect_index == 0:
        return {
            "status": "quality-blocker",
            "branch": "replicated-length-effect-below-first-shipping-rung",
            "native_context_cap_tokens": None,
            "above_cap_route": None,
            "lengths": summaries,
        }
    return {
        "status": "routing-required",
        "branch": "replicated-persistent-length-effect",
        "native_context_cap_tokens": lengths[first_effect_index - 1],
        "above_cap_route": "kquant",
        "first_effect_tokens": lengths[first_effect_index],
        "lengths": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kquant", required=True, type=Path)
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--yardstick", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    kquant_report = json.loads(args.kquant.read_text(encoding="utf-8"))
    native_report = json.loads(args.native.read_text(encoding="utf-8"))
    yardstick_report = json.loads(args.yardstick.read_text(encoding="utf-8"))
    kquant = rows(kquant_report)
    native = rows(native_report)
    if kquant.keys() != native.keys():
        parser.error("kquant and native fixture sets differ")
    bands = yardstick_by_length(yardstick_report)
    comparisons = [compare(kquant[key], native[key], bands) for key in kquant]
    decision = route(comparisons)
    report = {
        "schema": "muser.nvfp4-content-control-routing.v1",
        "status": decision["status"],
        "preregistered_method": {
            "documents_required": 3,
            "nested_lengths": [2048, 4096, 8192, 16384, 32768],
            "position_bin_tokens": POSITION_BIN_TOKENS,
            "length_effect": "two-of-three documents exceed at one length and the next",
            "short_context_band": "2k and 4k use the conservative 8k yardstick band",
        },
        "inputs": {
            "kquant_sha256": sha256(args.kquant),
            "native_sha256": sha256(args.native),
            "yardstick_sha256": sha256(args.yardstick),
        },
        "comparisons": comparisons,
        "routing_decision": decision,
        "seal_eligible": False,
    }
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"output": str(args.output), "status": decision["status"]}))
    if decision["status"] == "quality-blocker":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
