#!/usr/bin/env python3
"""Reduce MUSER_DFLASH_CYCLE_TRACE output into the K0 economics table.

Reads the muser-dflash-qualify JSONL stdout (cycle_trace per sample) and the
stderr log of `dflash-cycle-verify ...` lines (Metal verify-path breakdown),
then emits a min/median/max component table over all recorded cycles.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys

VERIFY_RE = re.compile(
    r"dflash-cycle-verify candidates=(?P<candidates>\d+)"
    r" checkpoint_ns=(?P<checkpoint_ns>\d+)"
    r" forward_ns=(?P<forward_ns>\d+)"
    r" decision_ns=(?P<decision_ns>\d+)"
    r" commit_ns=(?P<commit_ns>\d+)"
)


def stats(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-jsonl", required=True)
    parser.add_argument("--stderr-log", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cycles: list[dict] = []
    samples = []
    for line in open(args.samples_jsonl):
        line = line.strip()
        if not line.startswith("{"):
            continue
        record = json.loads(line)
        if record.get("kind") != "sample":
            continue
        samples.append(record)
        cycles.extend(record.get("cycle_trace") or [])
    if len(cycles) < 64:
        raise SystemExit(f"need at least 64 traced cycles, got {len(cycles)}")

    verify = []
    for line in open(args.stderr_log):
        match = VERIFY_RE.search(line)
        if match:
            verify.append({k: int(v) for k, v in match.groupdict().items()})

    table: dict[str, object] = {}
    table["cycles"] = len(cycles)
    table["verify_breakdown_rows"] = len(verify)
    for field in ("draft_ns", "verify_ns", "cycle_ns"):
        table[field] = stats([c[field] / 1e6 for c in cycles])
    residual = [(c["cycle_ns"] - c["draft_ns"] - c["verify_ns"]) / 1e6 for c in cycles]
    table["host_residual_ns"] = stats(residual)
    table["accepted_per_cycle"] = stats([float(c["accepted"]) for c in cycles])
    table["drafted_per_cycle"] = stats([float(c["drafted"]) for c in cycles])
    acceptance = sum(c["accepted"] for c in cycles) / max(1, sum(c["drafted"] for c in cycles))
    table["cycle_acceptance_rate"] = acceptance
    hist: dict[int, int] = {}
    for c in cycles:
        hist[c["accepted"]] = hist.get(c["accepted"], 0) + 1
    table["accepted_prefix_histogram"] = {str(k): hist[k] for k in sorted(hist)}
    if verify:
        for field in ("checkpoint_ns", "forward_ns", "decision_ns", "commit_ns"):
            table[f"verify_{field}"] = stats([v[field] / 1e6 for v in verify])
        accounted = [
            (v["checkpoint_ns"] + v["forward_ns"] + v["decision_ns"] + v["commit_ns"]) / 1e6
            for v in verify
        ]
        table["verify_accounted_ns"] = stats(accounted)
    table["per_sample"] = [
        {
            "repetition": s["repetition"],
            "rounds": s["rounds"],
            "acceptance_rate": s["acceptance_rate"],
            "draft_ms": s["draft_ns"] / 1e6,
            "target_verify_ms": s["target_verify_ns"] / 1e6,
            "dflash_ms": s["dflash_ns"] / 1e6,
            "target_only_ms": s["target_only_ns"] / 1e6,
            "speedup": s["speedup"],
        }
        for s in samples
    ]
    table["schema"] = "muser.dflash-cycle-table.v1"
    with open(args.output, "w") as stream:
        json.dump(table, stream, indent=2, sort_keys=True)
        stream.write("\n")
    json.dump(table, sys.stdout, indent=2, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
