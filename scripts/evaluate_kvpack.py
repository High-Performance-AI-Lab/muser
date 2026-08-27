#!/usr/bin/env python3
"""Fail-closed evaluator for resident, durable, and remote prefix packets."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

from packet_integrity import collect_unique_packet, publish_new
from release_lock import force_unsealed


EXACT_DEPTHS = (8192, 16384, 32768, 65536, 131008)
ANCESTOR_CUTS = (8192, 16384, 32768, 65536, 128768)
SUFFIXES = (1, 255, 256, 257, 2047)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def coefficient_of_variation(values: list[int]) -> float:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return variance**0.5 / mean


def close(left: float, right: float) -> bool:
    return abs(left - right) <= max(1e-12, abs(right) * 1e-12)


def expected_cells() -> dict[str, tuple[str, str, int, int, int]]:
    output: dict[str, tuple[str, str, int, int, int]] = {}
    for source in ("resident", "durable", "remote"):
        for depth in EXACT_DEPTHS:
            output[f"kvpack-{source}-exact-{depth}"] = (
                source, "exact-final", depth, depth, 0,
            )
        for cut in ANCESTOR_CUTS:
            for suffix in SUFFIXES:
                output[f"kvpack-{source}-ancestor-{cut}-s{suffix}"] = (
                    source, "deepest-ancestor", cut + suffix, cut, suffix,
                )
    return output


def main() -> int:
    args = parse_args()
    records = [
        json.loads(line) for line in args.ledger.read_text().splitlines() if line.strip()
    ]
    expected = expected_cells()
    relevant, failures = collect_unique_packet(
        records,
        set(expected),
        identity=args.identity,
        key=lambda record: (
            record.get("cell") if record.get("engine") == "kvpack" else None
        ),
        label="kvpack",
    )
    cells: dict[str, object] = {}
    speedups: dict[str, list[float]] = {
        "resident": [], "durable": [], "remote": [],
    }
    for name, (source, lookup, prompt, cut, suffix) in expected.items():
        record = relevant.get(name)
        if record is None:
            continue
        fingerprint = record.get("fingerprint", {})
        if (
            record.get("status") != "passed"
            or fingerprint.get("source") != source
            or fingerprint.get("lookup") != lookup
            or fingerprint.get("prompt_tokens") != prompt
            or fingerprint.get("published_cut") != cut
            or fingerprint.get("suffix_tokens") != suffix
            or not isinstance(fingerprint.get("generated_tokens_sha256"), str)
            or len(fingerprint["generated_tokens_sha256"]) != 64
            or not isinstance(fingerprint.get("full_logit_digest"), str)
            or len(fingerprint["full_logit_digest"]) != 64
        ):
            failures.append(f"{name} failed its exact route/correctness identity")
            continue
        raw = record.get("raw_ns")
        full = record.get("full_recompute_ns")
        if (
            not isinstance(raw, list)
            or len(raw) != 3
            or not all(isinstance(value, int) and value > 0 for value in raw)
            or not isinstance(full, int)
            or full <= 0
        ):
            failures.append(f"{name} has invalid raw timing evidence")
            continue
        cv = record.get("cv", 1.0)
        actual_cv = coefficient_of_variation(raw)
        if (
            not isinstance(cv, (int, float))
            or isinstance(cv, bool)
            or not math.isfinite(cv)
            or cv < 0
            or not close(cv, actual_cv)
            or cv > 0.02
        ):
            failures.append(f"{name} is unstable")
            continue
        source_prefill = record.get("source_prefill_ns")
        publication_ns = record.get("publication_ns")
        miss_lookup_ns = record.get("miss_lookup_ns")
        publication = record.get("publication_overhead_ratio")
        miss = record.get("miss_overhead_ratio")
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in (source_prefill, publication_ns, miss_lookup_ns)
        ):
            failures.append(f"{name} has invalid overhead timing components")
            continue
        if (
            not isinstance(publication, (int, float))
            or isinstance(publication, bool)
            or not math.isfinite(publication)
            or publication < 0
            or not close(publication, publication_ns / source_prefill)
            or publication > 0.05
        ):
            failures.append(f"{name} publication overhead exceeds 5%")
        if (
            not isinstance(miss, (int, float))
            or isinstance(miss, bool)
            or not math.isfinite(miss)
            or miss < 0
            or not close(miss, miss_lookup_ns / full)
            or miss > 0.02
        ):
            failures.append(f"{name} miss overhead exceeds 2%")
        reported_speedup = record.get("speedup_geomean_cell")
        arithmetic_speedup = full / (sum(raw) / len(raw))
        if (
            not isinstance(reported_speedup, (int, float))
            or isinstance(reported_speedup, bool)
            or not math.isfinite(reported_speedup)
            or not close(reported_speedup, arithmetic_speedup)
        ):
            failures.append(f"{name} speedup summary differs from raw timings")
        paired = [full / value for value in raw]
        cell_speedup = geometric_mean(paired)
        if cell_speedup <= 1.0:
            failures.append(f"{name} does not beat full local recomputation")
        speedups[source].extend(paired)
        cells[name] = {
            "cv": cv,
            "geometric_mean_speedup": cell_speedup,
            "publication_overhead_ratio": publication,
            "miss_overhead_ratio": miss,
            "generated_tokens_sha256": fingerprint["generated_tokens_sha256"],
            "full_logit_digest": fingerprint["full_logit_digest"],
        }
    source_speedups = {
        source: geometric_mean(values) if values else None for source, values in speedups.items()
    }
    for source, value in source_speedups.items():
        if value is None or value < 1.10:
            failures.append(f"{source} geometric-mean speedup is below 1.10x")
    eligible = not failures and len(cells) == len(expected)
    receipt = {
        "schema": "muser.kvpack.seal.v1",
        "status": "passed" if eligible else "failed",
        "identity": args.identity,
        "cells": cells,
        "geometric_mean_speedup_by_source": source_speedups,
        "failures": failures,
        "ledger_sha256": hashlib.sha256(args.ledger.read_bytes()).hexdigest(),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "seal_eligible": eligible,
    }
    force_unsealed(receipt, lane="kvpack")
    encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    publish_new(args.out, encoded)
    print(encoded, end="")
    return 0 if eligible else 1


if __name__ == "__main__":
    raise SystemExit(main())
