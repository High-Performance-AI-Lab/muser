#!/usr/bin/env python3
"""Assemble teacher-forced kquant rows for NVFP4 drift decomposition."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import llama_perplexity_evidence


DEFAULT_MODEL_SHA256 = "7e9b74b7c8875e9e265695df9613bf6290f2392e479ce740495a129019c488d8"
DEFAULT_REFERENCE_LANE = "tier-1-kquant-llama.cpp"
DEFAULT_SCHEMA = "muser.kquant-drift-decomposition-reference.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_digest(tokens: list[int]) -> str:
    return hashlib.sha256(
        b"".join(token.to_bytes(4, "little") for token in tokens)
    ).hexdigest()


def parse_cell(value: str) -> tuple[str, Path]:
    fixture_id, separator, path = value.partition("=")
    if not separator or not fixture_id.replace("-", "").isalnum() or not path:
        raise argparse.ArgumentTypeError("reference cell must be ID=PERPLEXITY_CAPTURE")
    return fixture_id, Path(path)


def reference_row(
    fixture: dict[str, Any],
    tokens: list[int],
    evidence: dict[str, Any],
    source: Path,
) -> dict[str, Any]:
    rows = evidence["rows"]
    positions = [int(row["position"]) + 1 for row in rows]
    if not positions or positions != sorted(positions) or positions[-1] >= len(tokens):
        raise ValueError(f"fixture {fixture['id']} has invalid teacher positions")
    logs = [-float(row["target_nll"]) for row in rows]
    top = [int(row["candidates"][0]["token_id"]) for row in rows]
    boundaries = sorted(
        {
            positions[0],
            positions[len(positions) // 4],
            positions[len(positions) // 2],
            positions[3 * len(positions) // 4],
            positions[-1],
        }
    )
    mean_nll = -math.fsum(logs) / len(logs)
    return {
        "id": fixture["id"],
        "regime": fixture["regime"],
        "token_count": len(tokens),
        "token_ids_sha256": token_digest(tokens),
        "scored_positions": positions,
        "target_logprobs": logs,
        "teacher_forced_top_token_ids": top,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "boundary_positions": boundaries,
        "source_capture": str(source),
        "source_capture_sha256": sha256(source),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--llama-receipt", required=True, type=Path)
    parser.add_argument("--cell", required=True, action="append", type=parse_cell)
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--reference-lane", default=DEFAULT_REFERENCE_LANE)
    parser.add_argument(
        "--report-schema",
        choices=(DEFAULT_SCHEMA, "muser.llama-drift-reference.v1"),
        default=DEFAULT_SCHEMA,
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(args.model_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in args.model_sha256
    ):
        parser.error("model SHA-256 must be lowercase hexadecimal")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    receipt = json.loads(args.llama_receipt.read_text(encoding="utf-8"))
    if manifest.get("schema") != "muser.nvfp4-drift-fixtures.v1":
        parser.error("fixture manifest identity is invalid")
    if receipt.get("schema") != "muser.llama_comparator.source_receipt.v3":
        parser.error("llama receipt identity is invalid")
    cells = dict(args.cell)
    if len(cells) != len(args.cell):
        parser.error("reference cell IDs are duplicated")
    results = []
    for fixture in manifest["fixtures"]:
        fixture_id = fixture["id"]
        if fixture_id not in cells:
            parser.error(f"missing reference cell: {fixture_id}")
        capture_path = cells[fixture_id]
        capture = json.loads(capture_path.read_text(encoding="utf-8"))
        if capture.get("schema") != "muser.llama-perplexity-capture.v1" or capture.get("status") != "passed":
            parser.error(f"perplexity capture is not passed: {fixture_id}")
        if capture.get("artifacts", {}).get("model", {}).get("sha256") != args.model_sha256:
            parser.error(f"perplexity capture model differs: {fixture_id}")
        token_path = Path(fixture["token_file"])
        if not token_path.is_absolute():
            token_path = args.manifest.parent / token_path
        tokens = [int(value) for value in token_path.read_bytes().split()]
        captured_token_path = Path(capture["artifacts"]["token_fixture"]["path"])
        captured_tokens = [int(value) for value in captured_token_path.read_bytes().split()]
        if captured_tokens != tokens:
            parser.error(f"perplexity capture token IDs differ: {fixture_id}")
        runtime = capture["runtime"]
        if runtime["context_length"] != len(tokens):
            parser.error(f"perplexity context differs from manifest: {fixture_id}")
        evidence = llama_perplexity_evidence.validate_teacher_evidence(
            capture_path.parent / "logits.u16",
            expected_upstream_commit=receipt["source_commit"],
            expected_patch_sha256=receipt["patch_sha256"],
            expected_context_length=runtime["context_length"],
            expected_chunks=runtime["chunks"],
            expected_batch_size=runtime["batch_size"],
            expected_ubatch_size=runtime["ubatch_size"],
            expected_threads=runtime["threads"],
            expected_kv_cache=runtime["kv_cache"],
            expected_model_transformer_layers=52,
            runtime_route="full-gpu",
        )
        results.append(reference_row(fixture, tokens, evidence, capture_path))
    if cells.keys() != {fixture["id"] for fixture in manifest["fixtures"]}:
        parser.error("reference cells include unknown fixture IDs")
    report = {
        "schema": args.report_schema,
        "producer_mode": "reference",
        "reference_lane": args.reference_lane,
        "model_sha256": args.model_sha256,
        "fixture_manifest_sha256": sha256(args.manifest),
        "llama_source_commit": receipt["source_commit"],
        "llama_patch_sha256": receipt["patch_sha256"],
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
