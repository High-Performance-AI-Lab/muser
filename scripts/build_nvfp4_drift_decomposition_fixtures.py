#!/usr/bin/env python3
"""Build frozen reduced-exact and long-routing NVFP4 decomposition fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


BOS_TOKEN = 200_000


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
    if len(tokens) < 2 or any(not 0 <= token < 202_048 for token in tokens):
        raise ValueError(f"invalid token fixture: {path}")
    return tokens


def write_exclusive(path: Path, content: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def emit_fixture(
    output_dir: Path,
    fixture_id: str,
    regime: str,
    tokens: list[int],
    source: Path,
    slice_description: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    destination = output_dir / f"{fixture_id}.tokens"
    write_exclusive(destination, "\n".join(map(str, tokens)) + "\n")
    entry = {
        "id": fixture_id,
        "regime": regime,
        "token_file": destination.name,
        "output_tokens": 256,
    }
    receipt = {
        "id": fixture_id,
        "regime": regime,
        "source": str(source),
        "source_sha256": sha256(source),
        "slice": slice_description,
        "token_count": len(tokens),
        "token_ids_sha256": token_digest(tokens),
        "token_file": destination.name,
        "token_file_sha256": sha256(destination),
    }
    return entry, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != "muser.nvfp4-drift-fixtures.v1":
        parser.error("source manifest identity is invalid")
    if args.output_dir.exists():
        parser.error("output directory already exists")
    args.output_dir.mkdir(parents=True)
    sources: dict[str, tuple[Path, list[int]]] = {}
    for fixture in manifest["fixtures"]:
        path = Path(fixture["token_file"])
        if not path.is_absolute():
            path = args.source_manifest.parent / path
        sources[fixture["id"]] = (path, read_tokens(path))
    required = {"code", "agentic", "long-context", "diverse-p1", "standard-2048"}
    if not required.issubset(sources):
        parser.error(f"source manifest lacks {sorted(required - sources.keys())}")

    reduced_specs = [
        ("code-r4096", "code", "code", sources["code"][1][:4096], {"start": 0, "end": 4096}),
        (
            "agentic-r4096",
            "agentic",
            "agentic",
            sources["agentic"][1][:4096],
            {"start": 0, "end": 4096},
        ),
        (
            "long-tail-r4096",
            "long-context",
            "long-context",
            [BOS_TOKEN, *sources["long-context"][1][-4095:]],
            {
                "source_start": len(sources["long-context"][1]) - 4095,
                "source_end": len(sources["long-context"][1]),
                "bos_prepended": True,
            },
        ),
        (
            "diverse-p1",
            "original",
            "diverse-p1",
            sources["diverse-p1"][1],
            {"start": 0, "end": len(sources["diverse-p1"][1])},
        ),
        (
            "standard-2048",
            "original",
            "standard-2048",
            sources["standard-2048"][1],
            {"start": 0, "end": len(sources["standard-2048"][1])},
        ),
    ]
    reduced_entries = []
    receipt_rows = []
    for fixture_id, regime, source_id, tokens, slice_description in reduced_specs:
        entry, receipt = emit_fixture(
            args.output_dir,
            fixture_id,
            regime,
            tokens,
            sources[source_id][0],
            slice_description,
        )
        reduced_entries.append(entry)
        receipt_rows.append(receipt)

    exact_specs = [
        ("code-r2048", "code", "code", sources["code"][1][:2048]),
        ("agentic-r2048", "agentic", "agentic", sources["agentic"][1][:2048]),
        (
            "long-tail-r2048",
            "long-context",
            "long-tail-r4096",
            [BOS_TOKEN, *sources["long-context"][1][-4095:]][:2048],
        ),
        ("diverse-p1-r192", "original", "diverse-p1", sources["diverse-p1"][1]),
        (
            "standard-r2048",
            "original",
            "standard-2048",
            sources["standard-2048"][1][:2048],
        ),
    ]
    exact_entries = []
    for fixture_id, regime, source_id, tokens in exact_specs:
        source_path = (
            args.output_dir / "long-tail-r4096.tokens"
            if source_id == "long-tail-r4096"
            else sources[source_id][0]
        )
        entry, receipt = emit_fixture(
            args.output_dir,
            fixture_id,
            regime,
            tokens,
            source_path,
            {"start": 0, "end": len(tokens), "purpose": "exact-attribution"},
        )
        exact_entries.append(entry)
        receipt_rows.append(receipt)

    routing_entries = []
    for token_count in (8192, 16384, 32768):
        fixture_id = f"long-{token_count}"
        tokens = sources["long-context"][1][:token_count]
        entry, receipt = emit_fixture(
            args.output_dir,
            fixture_id,
            "long-context",
            tokens,
            sources["long-context"][0],
            {"start": 0, "end": token_count},
        )
        routing_entries.append(entry)
        receipt_rows.append(receipt)

    reduced_manifest = {
        "schema": "muser.nvfp4-drift-fixtures.v1",
        "fixtures": reduced_entries,
    }
    exact_manifest = {
        "schema": "muser.nvfp4-drift-fixtures.v1",
        "fixtures": exact_entries,
    }
    routing_manifest = {
        "schema": "muser.nvfp4-drift-fixtures.v1",
        "fixtures": routing_entries,
    }
    reduced_path = args.output_dir / "reduced-manifest.json"
    exact_path = args.output_dir / "exact-manifest.json"
    routing_path = args.output_dir / "routing-manifest.json"
    write_exclusive(reduced_path, json.dumps(reduced_manifest, indent=2, sort_keys=True) + "\n")
    write_exclusive(exact_path, json.dumps(exact_manifest, indent=2, sort_keys=True) + "\n")
    write_exclusive(routing_path, json.dumps(routing_manifest, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema": "muser.nvfp4-drift-decomposition-fixtures.v1",
        "source_manifest": str(args.source_manifest),
        "source_manifest_sha256": sha256(args.source_manifest),
        "reduced_manifest": reduced_path.name,
        "reduced_manifest_sha256": sha256(reduced_path),
        "exact_manifest": exact_path.name,
        "exact_manifest_sha256": sha256(exact_path),
        "routing_manifest": routing_path.name,
        "routing_manifest_sha256": sha256(routing_path),
        "fixtures": receipt_rows,
        "seal_eligible": False,
    }
    receipt_path = args.output_dir / "receipt.json"
    write_exclusive(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"receipt": str(receipt_path), "fixtures": len(receipt_rows)}))


if __name__ == "__main__":
    main()
