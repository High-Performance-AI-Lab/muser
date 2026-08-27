#!/usr/bin/env python3
"""Compare retained teacher-forced top-1 traces from two E2 checkpoint runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA = "muser.checkpoint-top1-agreement.v1"
INPUT_SCHEMA = "muser.spark-nvfp4-drift-score.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_receipt(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("schema") != INPUT_SCHEMA:
        raise ValueError(f"{path} is not an E2 drift receipt")
    fixtures = value.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        raise ValueError(f"{path} has no fixtures")
    return value


def expected_emitted(match_probability: float, draft_length: int) -> float:
    candidate_count = draft_length + 1
    if match_probability == 1.0:
        return float(candidate_count)
    return (1.0 - match_probability**candidate_count) / (1.0 - match_probability)


def summarize(
    compared: int,
    mismatched: int,
    draft_length: int,
    target_wall_ms: float,
    fixed_overhead_ms: float,
    goal_tps: float,
) -> dict[str, Any]:
    mismatch_rate = mismatched / compared
    match_probability = 1.0 - mismatch_rate
    emitted = expected_emitted(match_probability, draft_length)
    projected_wall_ms = target_wall_ms + fixed_overhead_ms
    projected_tps = emitted * 1000.0 / projected_wall_ms
    target_budget_ms = emitted * 1000.0 / goal_tps - fixed_overhead_ms
    return {
        "compared_rows": compared,
        "mismatched_rows": mismatched,
        "mismatch_rate": mismatch_rate,
        "match_probability": match_probability,
        "iid_expected_emitted_per_round": emitted,
        "iid_projected_tps": projected_tps,
        "target_wall_budget_ms_at_goal": target_budget_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--draft-length", type=int, default=15)
    parser.add_argument("--target-wall-ms", type=float, required=True)
    parser.add_argument("--fixed-overhead-ms", type=float, required=True)
    parser.add_argument("--goal-tps", type=float, default=107.9)
    args = parser.parse_args()
    if args.draft_length < 1:
        parser.error("draft length must be positive")
    if not all(
        math.isfinite(value) and value > 0.0
        for value in (args.target_wall_ms, args.fixed_overhead_ms, args.goal_tps)
    ):
        parser.error("timing and throughput arguments must be finite and positive")
    if args.output.exists() or args.output.is_symlink():
        parser.error("refusing to replace an existing output receipt")

    left = read_receipt(args.left)
    right = read_receipt(args.right)
    if left["fixture_manifest_sha256"] != right["fixture_manifest_sha256"]:
        raise ValueError("checkpoint receipts use different fixture manifests")
    left_fixtures = {fixture["id"]: fixture for fixture in left["fixtures"]}
    right_fixtures = {fixture["id"]: fixture for fixture in right["fixtures"]}
    if left_fixtures.keys() != right_fixtures.keys():
        raise ValueError("checkpoint receipts have different fixture IDs")

    rows: list[dict[str, Any]] = []
    category_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    total_compared = 0
    total_mismatched = 0
    for fixture_id in sorted(left_fixtures):
        left_fixture = left_fixtures[fixture_id]
        right_fixture = right_fixtures[fixture_id]
        for field in ("token_file_sha256", "token_ids_sha256", "token_count"):
            if left_fixture[field] != right_fixture[field]:
                raise ValueError(f"fixture {fixture_id} differs at {field}")
        left_tokens = left_fixture["teacher_forced_top_token_ids"]
        right_tokens = right_fixture["teacher_forced_top_token_ids"]
        if len(left_tokens) != len(right_tokens) or not left_tokens:
            raise ValueError(f"fixture {fixture_id} has invalid top-token geometry")
        compared = len(left_tokens)
        mismatched = sum(a != b for a, b in zip(left_tokens, right_tokens, strict=True))
        category = fixture_id.split("-", 2)[1] if "-" in fixture_id else "unknown"
        category_counts[category][0] += compared
        category_counts[category][1] += mismatched
        total_compared += compared
        total_mismatched += mismatched
        rows.append(
            {
                "fixture_id": fixture_id,
                "category": category,
                **summarize(
                    compared,
                    mismatched,
                    args.draft_length,
                    args.target_wall_ms,
                    args.fixed_overhead_ms,
                    args.goal_tps,
                ),
            }
        )

    receipt = {
        "schema": SCHEMA,
        "left": {
            "path": str(args.left),
            "sha256": sha256_file(args.left),
            "checkpoint_revision": left["checkpoint_revision"],
            "checkpoint_artifact_sha256": left["checkpoint_artifact_sha256"],
        },
        "right": {
            "path": str(args.right),
            "sha256": sha256_file(args.right),
            "checkpoint_revision": right["checkpoint_revision"],
            "checkpoint_artifact_sha256": right["checkpoint_artifact_sha256"],
        },
        "fixture_manifest_sha256": left["fixture_manifest_sha256"],
        "projection": {
            "draft_length": args.draft_length,
            "candidate_count": args.draft_length + 1,
            "target_wall_ms": args.target_wall_ms,
            "fixed_overhead_ms": args.fixed_overhead_ms,
            "goal_tps": args.goal_tps,
            "assumption": (
                "teacher-forced checkpoint top-1 agreement is treated as an iid "
                "per-row proposal match; this is a risk screen, not measured "
                "DFlash acceptance or a serving throughput claim"
            ),
        },
        "fixtures": rows,
        "categories": {
            category: summarize(
                counts[0],
                counts[1],
                args.draft_length,
                args.target_wall_ms,
                args.fixed_overhead_ms,
                args.goal_tps,
            )
            for category, counts in sorted(category_counts.items())
        },
        "aggregate": summarize(
            total_compared,
            total_mismatched,
            args.draft_length,
            args.target_wall_ms,
            args.fixed_overhead_ms,
            args.goal_tps,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps(receipt["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
