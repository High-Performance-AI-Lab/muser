#!/usr/bin/env python3
"""Freeze the E1 yardstick and E2 nested content-control fixture manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from build_nvfp4_drift_fixtures import BOS_TOKEN, SCHEMA, sha256, tokenize, write_exclusive


E1_SOURCE_IDS = ("code", "agentic", "diverse-p1", "standard-2048", "long-context")
E1_ROUTING_IDS = ("long-8192", "long-16384")
E2_LENGTHS = (2048, 4096, 8192, 16384, 32768)


def token_digest(tokens: list[int]) -> str:
    return hashlib.sha256(
        b"".join(token.to_bytes(4, "little") for token in tokens)
    ).hexdigest()


def read_tokens(path: Path) -> list[int]:
    tokens = [int(value) for value in path.read_bytes().split()]
    if len(tokens) < 2 or any(not 0 <= value < 202_048 for value in tokens):
        raise ValueError(f"invalid token fixture: {path}")
    return tokens


def manifest_rows(path: Path) -> dict[str, tuple[dict[str, Any], Path]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"invalid fixture manifest: {path}")
    rows = {}
    for entry in manifest["fixtures"]:
        token_path = Path(entry["token_file"])
        if not token_path.is_absolute():
            token_path = path.parent / token_path
        rows[entry["id"]] = (entry, token_path.resolve())
    return rows


def e1_manifest(source: Path, routing: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_rows = manifest_rows(source)
    routing_rows = manifest_rows(routing)
    entries = []
    receipts = []
    for fixture_id in (*E1_SOURCE_IDS, *E1_ROUTING_IDS):
        table = source_rows if fixture_id in E1_SOURCE_IDS else routing_rows
        if fixture_id not in table:
            raise ValueError(f"E1 source fixture is missing: {fixture_id}")
        original, token_path = table[fixture_id]
        tokens = read_tokens(token_path)
        entries.append(
            {
                "id": fixture_id,
                "regime": original["regime"],
                "token_file": str(token_path),
                "output_tokens": 0,
            }
        )
        receipts.append(
            {
                "id": fixture_id,
                "source_manifest": str(source if fixture_id in E1_SOURCE_IDS else routing),
                "token_file": str(token_path),
                "token_file_sha256": sha256(token_path),
                "token_count": len(tokens),
                "token_ids_sha256": token_digest(tokens),
            }
        )
    return {"schema": SCHEMA, "fixtures": entries}, receipts


def source_document(paths: list[Path]) -> tuple[str, list[dict[str, Any]]]:
    chunks = []
    receipts = []
    for path in paths:
        content = path.read_text(encoding="utf-8", errors="strict")
        chunks.append(f"\n\n===== {path.as_posix()} =====\n{content}")
        receipts.append(
            {
                "path": path.as_posix(),
                "sha256": sha256(path),
                "bytes": len(content.encode("utf-8")),
            }
        )
    if not chunks:
        raise ValueError("content-control document has no source files")
    return "".join(chunks), receipts


def document_groups(repo: Path) -> dict[str, list[Path]]:
    return {
        "rust": sorted((repo / "crates").rglob("*.rs"))
        + sorted((repo / "crates").rglob("*.toml")),
        "python": sorted((repo / "scripts").rglob("*.py"))
        + sorted((repo / "scripts").rglob("*.sh")),
        "docs": sorted((repo / "docs").rglob("*.md"))
        + sorted((repo / "datasets").rglob("*.jsonl"))
        + sorted((repo / "datasets").rglob("*.json"))
        + sorted((repo / "datasets").rglob("*.txt")),
    }


def nested_prefixes(tokens: list[int]) -> dict[int, list[int]]:
    if len(tokens) < max(E2_LENGTHS) - 1:
        raise ValueError("content-control document tokenizes below 32k")
    full = [BOS_TOKEN, *tokens[: max(E2_LENGTHS) - 1]]
    return {length: full[:length] for length in E2_LENGTHS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--routing-manifest", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--bench", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    for path in (
        args.source_manifest,
        args.routing_manifest,
        args.repo,
        args.model,
        args.bench,
    ):
        if not path.exists():
            parser.error(f"required input does not exist: {path}")
    if args.output_dir.exists():
        parser.error("output directory already exists")
    args.output_dir.mkdir(parents=True)

    e1, e1_receipts = e1_manifest(args.source_manifest, args.routing_manifest)
    e1_path = args.output_dir / "e1-manifest.json"
    write_exclusive(e1_path, json.dumps(e1, indent=2, sort_keys=True) + "\n")

    e2_entries = []
    e2_documents = []
    for document_id, paths in document_groups(args.repo.resolve()).items():
        text, sources = source_document(paths)
        text_path = args.output_dir / f"e2-{document_id}.txt"
        write_exclusive(text_path, text)
        raw_tokens = tokenize(args.bench, args.model, text_path)
        prefixes = nested_prefixes(raw_tokens)
        fixture_rows = []
        for length, tokens in prefixes.items():
            fixture_id = f"e2-{document_id}-{length}"
            token_path = args.output_dir / f"{fixture_id}.tokens"
            write_exclusive(token_path, " ".join(map(str, tokens)) + "\n")
            e2_entries.append(
                {
                    "id": fixture_id,
                    "regime": "long-context",
                    "document": document_id,
                    "context_length": length,
                    "token_file": token_path.name,
                    "output_tokens": 0,
                }
            )
            fixture_rows.append(
                {
                    "id": fixture_id,
                    "token_file": token_path.name,
                    "token_file_sha256": sha256(token_path),
                    "token_count": len(tokens),
                    "token_ids_sha256": token_digest(tokens),
                }
            )
        e2_documents.append(
            {
                "id": document_id,
                "text_file": text_path.name,
                "text_file_sha256": sha256(text_path),
                "source_files": sources,
                "raw_token_count": len(raw_tokens),
                "fixtures": fixture_rows,
            }
        )
    e2 = {"schema": SCHEMA, "fixtures": e2_entries}
    e2_path = args.output_dir / "e2-manifest.json"
    write_exclusive(e2_path, json.dumps(e2, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema": "muser.nvfp4-e-series-fixtures.v1",
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": sha256(args.source_manifest),
        "routing_manifest": str(args.routing_manifest.resolve()),
        "routing_manifest_sha256": sha256(args.routing_manifest),
        "tokenizer_model": str(args.model.resolve()),
        "tokenizer_model_sha256": sha256(args.model),
        "tokenizer_bench": str(args.bench.resolve()),
        "tokenizer_bench_sha256": sha256(args.bench),
        "e1_manifest": e1_path.name,
        "e1_manifest_sha256": sha256(e1_path),
        "e1_fixtures": e1_receipts,
        "e2_manifest": e2_path.name,
        "e2_manifest_sha256": sha256(e2_path),
        "e2_documents": e2_documents,
        "nested_prefix_lengths": list(E2_LENGTHS),
        "position_bin_tokens": 512,
        "seal_eligible": False,
    }
    receipt_path = args.output_dir / "receipt.json"
    write_exclusive(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "e1_manifest": str(e1_path),
                "e2_manifest": str(e2_path),
                "receipt": str(receipt_path),
            }
        )
    )


if __name__ == "__main__":
    main()
