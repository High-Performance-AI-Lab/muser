#!/usr/bin/env python3
"""Re-tokenize the frozen 2026-08-17 E2 corpora to deeper nested lengths.

The original E2 fixtures (``scripts/build_nvfp4_e_series_fixtures.py``) were
frozen against a repo snapshot on 2026-08-17 and only go to 32768 tokens.
This helper does NOT re-glob the live repository (which has drifted since
then) -- it only re-tokenizes the already-frozen ``e2-{doc}.txt`` files that
live under the frozen evidence directory, after verifying each one's sha256
against the frozen ``receipt.json`` written at build time. This is the only
way to extend the E2 ladder to new depths without silently changing the
underlying corpus.

Pure CPU tokenization: shells out to ``<bench> tokenize``, the same
CPU-only subcommand used by ``build_nvfp4_drift_fixtures.tokenize()``. This
script performs no accelerator work and is intentionally NOT wrapped in
accelerator_safe.py by its caller -- verify that assumption with --dry-run
before trusting it in a new environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from build_nvfp4_drift_fixtures import BOS_TOKEN, tokenize, write_exclusive

SCHEMA = "muser.nvfp4-e2-extended-fixtures.v1"
DOCUMENT_IDS = ("rust", "python", "docs")


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


def load_receipt(receipt_path: Path) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "muser.nvfp4-e-series-fixtures.v1":
        raise ValueError(f"unexpected schema in frozen receipt: {receipt_path}")
    return receipt


def source_text_sha256(receipt: dict[str, Any], document_id: str) -> tuple[str, str]:
    """Return (text_file_name, sha256) for a document row in the frozen receipt."""
    for row in receipt.get("e2_documents", []):
        if row.get("id") == document_id:
            return row["text_file"], row["text_file_sha256"]
    raise ValueError(f"frozen receipt has no e2 document row for {document_id!r}")


def nested_prefix(tokens: list[int], length: int) -> list[int]:
    if len(tokens) < length - 1:
        raise ValueError(
            f"frozen corpus tokenizes to {len(tokens)} tokens, "
            f"too short to extend to length {length}"
        )
    return [BOS_TOKEN, *tokens[: length - 1]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        required=True,
        type=Path,
        help="frozen nvfp4-e-series-fixtures-20260817 evidence directory (read-only)",
    )
    parser.add_argument(
        "--receipt",
        required=True,
        type=Path,
        help="receipt.json inside --source-dir (verified before any tokenization)",
    )
    parser.add_argument("--bench", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--lengths",
        required=True,
        nargs="+",
        type=int,
        help="new nested prefix lengths to produce, e.g. 65536 131008",
    )
    parser.add_argument(
        "--documents",
        nargs="+",
        default=list(DOCUMENT_IDS),
        choices=DOCUMENT_IDS,
        help="which frozen e2 documents to extend (default: all three)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="new, must-not-already-exist output directory for extended fixtures",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.source_dir, args.receipt, args.bench, args.model):
        if not path.exists():
            raise SystemExit(f"required input does not exist: {path}")
    if args.output_dir.exists():
        raise SystemExit(f"output directory already exists: {args.output_dir}")
    for length in args.lengths:
        if length < 2:
            raise SystemExit(f"invalid nested prefix length: {length}")

    receipt = load_receipt(args.receipt)
    args.output_dir.mkdir(parents=True)

    document_rows: list[dict[str, Any]] = []
    for document_id in args.documents:
        text_name, expected_sha256 = source_text_sha256(receipt, document_id)
        text_path = args.source_dir / text_name
        if not text_path.exists():
            raise SystemExit(f"frozen source text missing on disk: {text_path}")
        actual_sha256 = sha256(text_path)
        if actual_sha256 != expected_sha256:
            raise SystemExit(
                f"frozen corpus drift detected for {document_id!r}: "
                f"{text_path} sha256 is {actual_sha256}, "
                f"receipt expects {expected_sha256} -- refusing to tokenize"
            )

        raw_tokens = tokenize(args.bench, args.model, text_path)
        fixture_rows = []
        for length in args.lengths:
            if len(raw_tokens) < length - 1:
                # A corpus that cannot honestly fill the depth is recorded,
                # not repeated: repetition would rebuild the synthetic-
                # periodicity problem this lane exists to avoid.
                fixture_rows.append(
                    {
                        "id": f"e2-{document_id}-{length}",
                        "document": document_id,
                        "context_length": length,
                        "skipped": "corpus-too-short",
                        "raw_token_count": len(raw_tokens),
                    }
                )
                print(
                    f"extend: {document_id!r} corpus has {len(raw_tokens)} tokens; "
                    f"cannot extend to {length}; row recorded as skipped",
                    flush=True,
                )
                continue
            tokens = nested_prefix(raw_tokens, length)
            fixture_id = f"e2-{document_id}-{length}"
            token_path = args.output_dir / f"{fixture_id}.tokens"
            # One token id per line -- the format every frozen fixture and
            # downstream reader (capture, scorer manifest) uses.
            write_exclusive(token_path, "\n".join(map(str, tokens)) + "\n")
            fixture_rows.append(
                {
                    "id": fixture_id,
                    "document": document_id,
                    "context_length": length,
                    "token_file": token_path.name,
                    "token_file_sha256": sha256(token_path),
                    "token_count": len(tokens),
                    "token_ids_sha256": token_digest(tokens),
                }
            )
        document_rows.append(
            {
                "id": document_id,
                "source_text_file": text_name,
                "source_text_file_sha256": actual_sha256,
                "source_text_file_sha256_verified_against_frozen_receipt": True,
                "raw_token_count": len(raw_tokens),
                "fixtures": fixture_rows,
            }
        )

    extended_receipt = {
        "schema": SCHEMA,
        "source_dir": str(args.source_dir.resolve()),
        "frozen_receipt": str(args.receipt.resolve()),
        "frozen_receipt_sha256": sha256(args.receipt),
        "tokenizer_model": str(args.model.resolve()),
        "tokenizer_bench": str(args.bench.resolve()),
        "nested_prefix_lengths": list(args.lengths),
        "e2_documents": document_rows,
    }
    receipt_path = args.output_dir / "extended-receipt.json"
    write_exclusive(
        receipt_path, json.dumps(extended_receipt, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {"output_dir": str(args.output_dir), "receipt": str(receipt_path)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
