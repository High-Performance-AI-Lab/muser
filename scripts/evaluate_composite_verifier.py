#!/usr/bin/env python3
"""Evaluate composite GX verifier economics against the qualified lane."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any


SCHEMA = "muser.composite-verifier-economics.v1"


def percentile(values: list[float], probability: float) -> float:
    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("invalid percentile input")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def expected_commits_iid(alpha: float, candidates: int) -> float:
    if alpha == 1.0:
        return float(candidates)
    return (1.0 - alpha**candidates) / (1.0 - alpha)


def required_alpha(required_commits: float, candidates: int) -> float | None:
    if required_commits <= 1.0:
        return 0.0
    if required_commits > candidates:
        return None
    low = 0.0
    high = 1.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if expected_commits_iid(middle, candidates) < required_commits:
            low = middle
        else:
            high = middle
    return high


def read_benchmark(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if value.get("schema") != "muser.spark-native-nvfp4-verifier-benchmark.v1":
        raise ValueError("input is not a native verifier benchmark")
    if value.get("genesis", {}).get("kind") != "redhat_portable_kv_import":
        raise ValueError("benchmark does not prove a RedHat portable-KV genesis")
    return value


def write_exclusive(path: Path, value: object) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draft-ms", type=float, default=26.9)
    parser.add_argument("--rtt-ms", type=float, default=0.78)
    parser.add_argument("--capture-wire-ms", type=float, default=4.37)
    parser.add_argument("--sparse-wire-ms", type=float, default=0.01)
    parser.add_argument("--baseline-tps", type=float, default=107.9)
    parser.add_argument(
        "--accepted-prefix-counts",
        type=Path,
        help="optional JSON array of actual accepted draft counts per round",
    )
    args = parser.parse_args()
    for name in ("draft_ms", "rtt_ms", "capture_wire_ms", "sparse_wire_ms"):
        if getattr(args, name) < 0.0:
            parser.error(f"{name} must be nonnegative")
    if args.baseline_tps <= 0.0:
        parser.error("baseline TPS must be positive")

    benchmark = read_benchmark(args.benchmark)
    if len(benchmark["cells"]) != 1:
        raise ValueError("economics gate requires one verifier shape")
    cell = benchmark["cells"][0]
    candidates = int(cell["candidate_count"])
    target_ms = [float(sample["wall_ns"]) / 1e6 for sample in cell["samples"]]
    fixed_ms = args.draft_ms + args.rtt_ms + args.capture_wire_ms + args.sparse_wire_ms
    total_ms = [wall + fixed_ms for wall in target_ms]
    required_commits = [args.baseline_tps * wall / 1000.0 for wall in total_ms]
    median_required = statistics.median(required_commits)
    p95_required = percentile(required_commits, 0.95)
    median_alpha = required_alpha(median_required, candidates)
    p95_alpha = required_alpha(p95_required, candidates)
    full_accept_tps = [candidates * 1000.0 / wall for wall in total_ms]

    actual = None
    if args.accepted_prefix_counts is not None:
        counts = json.loads(args.accepted_prefix_counts.read_text())
        if not isinstance(counts, list) or not counts:
            raise ValueError("accepted-prefix file must be a nonempty JSON array")
        if any(not isinstance(value, int) or not 0 <= value <= candidates - 1 for value in counts):
            raise ValueError("accepted-prefix count is outside the verifier window")
        committed = [1 + value for value in counts]
        mean_committed = statistics.fmean(committed)
        actual = {
            "rounds": len(counts),
            "mean_committed_tokens": mean_committed,
            "accepted_draft_rate": sum(counts) / ((candidates - 1) * len(counts)),
            "median_target_effective_tps": mean_committed * 1000.0 / statistics.median(total_ms),
            "p95_wall_effective_tps": mean_committed * 1000.0 / percentile(total_ms, 0.95),
        }
        actual["beats_baseline_median"] = (
            actual["median_target_effective_tps"] > args.baseline_tps
        )
        actual["beats_baseline_p95_wall"] = (
            actual["p95_wall_effective_tps"] > args.baseline_tps
        )

    result = {
        "schema": SCHEMA,
        "benchmark": str(args.benchmark),
        "candidate_count": candidates,
        "baseline_tps": args.baseline_tps,
        "fixed_overhead_ms": {
            "draft": args.draft_ms,
            "rtt": args.rtt_ms,
            "capture_wire": args.capture_wire_ms,
            "sparse_q_wire": args.sparse_wire_ms,
            "total": fixed_ms,
        },
        "target_wall_ms": {
            "median": statistics.median(target_ms),
            "p95": percentile(target_ms, 0.95),
            "range": [min(target_ms), max(target_ms)],
        },
        "full_accept_tps": {
            "median": statistics.median(full_accept_tps),
            "p05": percentile(full_accept_tps, 0.05),
        },
        "required_mean_committed_tokens": {
            "median_wall": median_required,
            "p95_wall": p95_required,
        },
        "iid_acceptance_threshold": {
            "median_wall": median_alpha,
            "p95_wall": p95_alpha,
        },
        "full_accept_beats_baseline_median": statistics.median(full_accept_tps)
        > args.baseline_tps,
        "full_accept_beats_baseline_p95_wall": percentile(full_accept_tps, 0.05)
        > args.baseline_tps,
        "actual_acceptance": actual,
    }
    write_exclusive(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
