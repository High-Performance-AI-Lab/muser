#!/usr/bin/env python3
"""Project a longer native drift score onto frozen prefix fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


def token_digest(tokens: list[int]) -> str:
    return hashlib.sha256(
        b"".join(token.to_bytes(4, "little") for token in tokens)
    ).hexdigest()


def read_tokens(path: Path) -> list[int]:
    return [int(value) for value in path.read_bytes().split()]


def load_manifest(path: Path) -> dict[str, tuple[dict[str, Any], Path, list[int]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "muser.nvfp4-drift-fixtures.v1":
        raise ValueError(f"invalid manifest: {path}")
    rows = {}
    for entry in manifest["fixtures"]:
        token_path = Path(entry["token_file"])
        if not token_path.is_absolute():
            token_path = path.parent / token_path
        rows[entry["id"]] = (entry, token_path, read_tokens(token_path))
    return rows


def parse_mapping(value: str) -> tuple[str, str]:
    target, separator, source = value.partition("=")
    if not separator or not target or not source:
        raise argparse.ArgumentTypeError("mapping must be TARGET=SOURCE")
    return target, source


def project_row(
    source_row: dict[str, Any],
    source_tokens: list[int],
    target_entry: dict[str, Any],
    target_path: Path,
    target_tokens: list[int],
) -> dict[str, Any]:
    if source_row["token_count"] != len(source_tokens):
        raise ValueError("source report token count differs")
    if source_row["token_ids_sha256"] != token_digest(source_tokens):
        raise ValueError("source report token digest differs")
    if source_tokens[: len(target_tokens)] != target_tokens:
        raise ValueError("target fixture is not a source prefix")
    row_count = len(target_tokens) - 1
    logs = [float(value) for value in source_row["target_logprobs"][:row_count]]
    top = [int(value) for value in source_row["teacher_forced_top_token_ids"][:row_count]]
    if len(logs) != row_count or len(top) != row_count:
        raise ValueError("source report lacks target rows")
    boundaries = sorted(
        {1, len(target_tokens) // 4, len(target_tokens) // 2, 3 * len(target_tokens) // 4, len(target_tokens) - 1}
    )
    mean_nll = -math.fsum(logs) / len(logs)
    return {
        "id": target_entry["id"],
        "regime": target_entry["regime"],
        "token_file": str(target_path),
        "token_count": len(target_tokens),
        "token_ids_sha256": token_digest(target_tokens),
        "output_tokens": 0,
        "target_logprobs": logs,
        "teacher_forced_top_token_ids": top,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "boundary_positions": boundaries,
        "projected_from": source_row["id"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--target-manifest", required=True, type=Path)
    parser.add_argument("--map", required=True, action="append", type=parse_mapping)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    if source.get("schema") != "muser.spark-nvfp4-drift-score.v1":
        parser.error("source score identity is invalid")
    source_rows = {row["id"]: row for row in source["fixtures"]}
    source_fixtures = load_manifest(args.source_manifest)
    target_fixtures = load_manifest(args.target_manifest)
    mapping = dict(args.map)
    if mapping.keys() != target_fixtures.keys():
        parser.error("mapping and target fixture sets differ")
    results = []
    for target_id, (target_entry, target_path, target_tokens) in target_fixtures.items():
        source_id = mapping[target_id]
        if source_id not in source_rows or source_id not in source_fixtures:
            parser.error(f"unknown source fixture: {source_id}")
        _, _, source_tokens = source_fixtures[source_id]
        results.append(
            project_row(
                source_rows[source_id],
                source_tokens,
                target_entry,
                target_path,
                target_tokens,
            )
        )
    report = {
        "schema": "muser.spark-nvfp4-drift-score-projection.v1",
        "producer_mode": source["producer_mode"],
        "checkpoint_artifact_sha256": source["checkpoint_artifact_sha256"],
        "checkpoint_revision": source["checkpoint_revision"],
        "vllm_commit": source["vllm_commit"],
        "source_report_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(args.source_manifest.read_bytes()).hexdigest(),
        "fixture_manifest_sha256": hashlib.sha256(args.target_manifest.read_bytes()).hexdigest(),
        "route": source["route"],
        "fixtures": results,
        "seal_eligible": False,
    }
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"output": str(args.output), "fixtures": len(results)}))


if __name__ == "__main__":
    main()
