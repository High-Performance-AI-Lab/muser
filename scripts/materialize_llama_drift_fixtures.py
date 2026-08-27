#!/usr/bin/env python3
"""Materialize a line-oriented llama comparator view of a drift manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "muser.nvfp4-drift-fixtures.v1"
        or not isinstance(manifest.get("fixtures"), list)
    ):
        parser.error("input is not a drift fixture manifest")
    if args.output_dir.exists():
        parser.error("output directory already exists")
    args.output_dir.mkdir(parents=True)
    rows = []
    for fixture in manifest["fixtures"]:
        fixture_id = fixture.get("id")
        raw_path = fixture.get("token_file")
        if not isinstance(fixture_id, str) or not isinstance(raw_path, str):
            parser.error("fixture entry is malformed")
        source = Path(raw_path)
        if not source.is_absolute():
            source = args.manifest.parent / source
        tokens = read_tokens(source)
        destination = args.output_dir / f"{fixture_id}.tokens"
        write_exclusive(destination, "\n".join(map(str, tokens)) + "\n")
        if read_tokens(destination) != tokens:
            raise RuntimeError("line-oriented token materialization changed token IDs")
        rows.append(
            {
                "id": fixture_id,
                "source": str(source),
                "source_sha256": sha256(source),
                "line_fixture": destination.name,
                "line_fixture_sha256": sha256(destination),
                "token_count": len(tokens),
                "token_ids_sha256": token_digest(tokens),
            }
        )
    receipt = {
        "schema": "muser.llama-drift-fixture-view.v1",
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "fixtures": rows,
        "seal_eligible": False,
    }
    receipt_path = args.output_dir / "receipt.json"
    write_exclusive(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"receipt": str(receipt_path), "fixtures": len(rows)}))


if __name__ == "__main__":
    main()
