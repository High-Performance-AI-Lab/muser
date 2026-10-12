#!/usr/bin/env python3
"""Assemble retained and new kquant cells into one drift reference report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import llama_perplexity_evidence


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


def read_tokens(path: Path) -> list[int]:
    tokens = [int(value) for value in path.read_bytes().split()]
    if not tokens or any(not 0 <= token < 202_048 for token in tokens):
        raise ValueError(f"invalid token fixture: {path}")
    return tokens


def parse_cell(value: str) -> tuple[str, tuple[Path, Path, Path]]:
    fixture_id, separator, paths = value.partition("=")
    parts = paths.split(",")
    if not separator or not fixture_id.replace("-", "").isalnum() or len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "reference cell must be ID=PERPLEXITY_CAPTURE,GREEDY_REPORT,GENERATED_TOKENS"
        )
    return fixture_id, (Path(parts[0]), Path(parts[1]), Path(parts[2]))


def reference_row(
    fixture: dict[str, Any],
    tokens: list[int],
    evidence: dict[str, Any],
    generated: list[int],
    sources: dict[str, Any],
) -> dict[str, Any]:
    rows = evidence["rows"]
    positions = [int(row["position"]) + 1 for row in rows]
    if not positions or positions != sorted(positions) or positions[-1] >= len(tokens):
        raise ValueError(f"fixture {fixture['id']} has invalid teacher positions")
    target_logprobs = [-float(row["target_nll"]) for row in rows]
    top_tokens = [int(row["candidates"][0]["token_id"]) for row in rows]
    boundaries = sorted(
        {
            positions[0],
            positions[len(positions) // 4],
            positions[len(positions) // 2],
            positions[3 * len(positions) // 4],
            positions[-1],
        }
    )
    mean_nll = -math.fsum(target_logprobs) / len(target_logprobs)
    return {
        "id": fixture["id"],
        "regime": fixture["regime"],
        "token_count": len(tokens),
        "token_ids_sha256": token_digest(tokens),
        "output_tokens": len(generated),
        "generated_tokens": generated,
        "generated_tokens_sha256": token_digest(generated),
        "scored_positions": positions,
        "target_logprobs": target_logprobs,
        "teacher_forced_top_token_ids": top_tokens,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "boundary_positions": boundaries,
        "sources": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--llama-receipt", required=True, type=Path)
    parser.add_argument("--cell", required=True, action="append", type=parse_cell)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
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
        capture_path, greedy_path, generated_path = cells[fixture_id]
        for path in (capture_path, greedy_path, generated_path):
            if not path.is_file() or path.is_symlink():
                parser.error(f"reference artifact is not a regular file: {path}")
        capture = json.loads(capture_path.read_text(encoding="utf-8"))
        greedy = json.loads(greedy_path.read_text(encoding="utf-8"))
        if capture.get("schema") != "muser.llama-perplexity-capture.v1" or capture.get("status") != "passed":
            parser.error(f"perplexity capture is not passed: {fixture_id}")
        if greedy.get("schema") != "muser.llama-quality-capture.v1" or greedy.get("status") != "passed":
            parser.error(f"greedy capture is not passed: {fixture_id}")
        source = Path(fixture["token_file"])
        if not source.is_absolute():
            source = args.manifest.parent / source
        tokens = read_tokens(source)
        generated = read_tokens(generated_path)
        if len(generated) != fixture["output_tokens"]:
            parser.error(f"greedy output count differs: {fixture_id}")
        if greedy.get("generated_tokens_sha256") != "sha256:" + token_digest(generated):
            parser.error(f"greedy output digest differs: {fixture_id}")
        runtime = capture["runtime"]
        logits_path = capture_path.parent / "logits.u16"
        evidence = llama_perplexity_evidence.validate_teacher_evidence(
            logits_path,
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
        if runtime["context_length"] != len(tokens):
            parser.error(f"perplexity context differs from manifest: {fixture_id}")
        results.append(
            reference_row(
                fixture,
                tokens,
                evidence,
                generated,
                {
                    "perplexity_capture": str(capture_path),
                    "perplexity_capture_sha256": sha256(capture_path),
                    "greedy_capture": str(greedy_path),
                    "greedy_capture_sha256": sha256(greedy_path),
                    "generated_tokens": str(generated_path),
                    "generated_tokens_file_sha256": sha256(generated_path),
                },
            )
        )
    if cells.keys() != {fixture["id"] for fixture in manifest["fixtures"]}:
        parser.error("reference cells include unknown fixture IDs")
    report = {
        "schema": "muser.kquant-drift-reference.v1",
        "producer_mode": "reference",
        "reference_lane": "tier-1-kquant-llama.cpp",
        "model_sha256": "7e9b74b7c8875e9e265695df9613bf6290f2392e479ce740495a129019c488d8",
        "llama_source_commit": receipt["source_commit"],
        "llama_patch_sha256": receipt["patch_sha256"],
        "fixture_manifest_sha256": sha256(args.manifest),
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
