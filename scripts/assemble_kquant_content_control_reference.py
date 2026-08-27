#!/usr/bin/env python3
"""Assemble per-document kquant perplexity captures into one E2 content-control
reference report, in the ``{"fixtures": [...]}`` shape that
``evaluate_nvfp4_content_control.py`` requires (``rows()`` reads
``report["fixtures"]``; each row needs ``target_logprobs`` and
``teacher_forced_top_token_ids``).

This is the CPU-only glue between stage 3c's per-document
``capture_llama_perplexity.py`` receipts and stage 3d's comparator: it
re-derives the exact teacher-forced target logprobs and top-1 token ids from
each capture's quantized-logits sibling files (via
``llama_perplexity_evidence.validate_teacher_evidence``), the same way
``assemble_kquant_drift_reference.py`` does for the drift-decomposition
pipeline -- but without that pipeline's greedy/generated-token requirement,
which content-control comparison does not need.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import llama_perplexity_evidence


COMPACT_TEACHER_SCHEMA = "muser.llama-perplexity-compact-teacher.v1"


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
    if not tokens:
        raise ValueError(f"empty token fixture: {path}")
    return tokens


def parse_capture(value: str) -> tuple[str, Path]:
    fixture_id, separator, path = value.partition("=")
    if not separator or not fixture_id:
        raise argparse.ArgumentTypeError("capture must be ID=PERPLEXITY_CAPTURE_JSON")
    return fixture_id, Path(path)


def reference_row(
    fixture: dict[str, Any], tokens: list[int], evidence: dict[str, Any]
) -> dict[str, Any]:
    rows = evidence["rows"]
    positions = [int(row["position"]) + 1 for row in rows]
    if not positions or positions != sorted(positions) or positions[-1] >= len(tokens):
        raise ValueError(f"fixture {fixture['id']} has invalid teacher positions")
    return {
        "id": fixture["id"],
        "regime": fixture["regime"],
        "token_count": len(tokens),
        "token_ids_sha256": token_digest(tokens),
        "scored_positions": positions,
        "target_logprobs": [-float(row["target_nll"]) for row in rows],
        "teacher_forced_top_token_ids": [
            int(row["candidates"][0]["token_id"]) for row in rows
        ],
    }


def exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} keys differ: expected {sorted(expected)}, got {sorted(value)}"
        )


def compact_reference_row(
    fixture: dict[str, Any],
    tokens: list[int],
    capture: dict[str, Any],
    receipt: dict[str, Any],
    capture_path: Path,
) -> dict[str, Any]:
    """Validate and reduce a compacted, previously cross-bound teacher report."""
    teacher = capture.get("artifacts", {}).get("teacher_evidence")
    if not isinstance(teacher, dict):
        raise ValueError("capture lacks teacher evidence")
    source_keys = {"quantized_logits", "exact_top10", "runtime"}
    if not source_keys <= teacher.keys():
        raise ValueError("compact teacher evidence lacks source artifact identities")
    if not isinstance(teacher.get("source_artifacts_retained"), bool):
        raise ValueError("compact teacher evidence lacks a retention verdict")
    compact_artifact = teacher.get("compact")
    if not isinstance(compact_artifact, dict):
        raise ValueError("capture lacks compact teacher artifact")
    exact_keys(compact_artifact, {"path", "bytes", "sha256"}, "compact artifact")
    compact_path = Path(compact_artifact["path"])
    if (
        compact_path.is_symlink()
        or not compact_path.is_file()
        or compact_path.parent.resolve() != capture_path.parent.resolve()
    ):
        raise ValueError(f"compact teacher report is not a local regular file: {compact_path}")
    if (
        compact_path.stat().st_size != compact_artifact["bytes"]
        or sha256(compact_path) != compact_artifact["sha256"]
    ):
        raise ValueError("compact teacher artifact identity differs from capture")

    compact = json.loads(compact_path.read_text(encoding="utf-8"))
    exact_keys(
        compact,
        {"schema", "status", "validation", "geometry", "metrics", "rows"},
        "compact teacher report",
    )
    if (
        compact["schema"] != COMPACT_TEACHER_SCHEMA
        or compact["status"] != "validated"
    ):
        raise ValueError("compact teacher report is not validated")

    validation = compact["validation"]
    exact_keys(
        validation,
        {
            "validator",
            "quantized_cross_binding",
            "upstream_commit",
            "patch_sha256",
            "evidence_id",
            "source_artifacts",
        },
        "compact validation",
    )
    if (
        validation["validator"]
        != "scripts/llama_perplexity_evidence.py::validate_teacher_evidence"
        or validation["quantized_cross_binding"] != "validated-before-compaction"
        or validation["upstream_commit"] != receipt["source_commit"]
        or validation["patch_sha256"] != receipt["patch_sha256"]
        or validation["source_artifacts"]
        != {key: teacher[key] for key in sorted(source_keys)}
    ):
        raise ValueError("compact validation identity differs from capture")
    evidence_id = validation["evidence_id"]
    if not isinstance(evidence_id, str) or len(evidence_id) != 64:
        raise ValueError("compact evidence ID is invalid")

    geometry = compact["geometry"]
    exact_keys(
        geometry,
        {"context_length", "vocab_size", "chunks", "scored_rows"},
        "compact geometry",
    )
    expected_positions = list(range(len(tokens) // 2, len(tokens) - 1))
    if (
        geometry["context_length"] != len(tokens)
        or geometry["chunks"] != 1
        or geometry["scored_rows"] != len(expected_positions)
        or not isinstance(geometry["vocab_size"], int)
        or isinstance(geometry["vocab_size"], bool)
        or geometry["vocab_size"] < 2
    ):
        raise ValueError("compact teacher geometry differs from fixture")

    compact_rows = compact["rows"]
    if not isinstance(compact_rows, list) or len(compact_rows) != len(expected_positions):
        raise ValueError("compact teacher rows differ from fixture geometry")
    target_logprobs: list[float] = []
    top_tokens: list[int] = []
    for index, (row, position) in enumerate(zip(compact_rows, expected_positions)):
        if not isinstance(row, dict):
            raise ValueError(f"compact teacher row {index} is not an object")
        exact_keys(
            row,
            {
                "chunk",
                "position",
                "input_token_id",
                "target_token_id",
                "target_nll",
                "teacher_forced_top_token_id",
            },
            f"compact teacher row {index}",
        )
        top_token = row["teacher_forced_top_token_id"]
        target_nll = row["target_nll"]
        if (
            row["chunk"] != 0
            or row["position"] != position
            or row["input_token_id"] != tokens[position]
            or row["target_token_id"] != tokens[position + 1]
            or isinstance(top_token, bool)
            or not isinstance(top_token, int)
            or not 0 <= top_token < geometry["vocab_size"]
            or isinstance(target_nll, bool)
            or not isinstance(target_nll, (int, float))
            or not math.isfinite(float(target_nll))
            or float(target_nll) < 0.0
        ):
            raise ValueError(f"compact teacher row {index} differs from fixture")
        target_logprobs.append(-float(target_nll))
        top_tokens.append(top_token)

    metrics = compact["metrics"]
    exact_keys(metrics, {"exact_target_nll_sum", "exact_perplexity"}, "compact metrics")
    nll_sum = math.fsum(-value for value in target_logprobs)
    exact_perplexity = math.exp(nll_sum / len(target_logprobs))
    if (
        capture.get("metrics") != metrics
        or not math.isclose(nll_sum, float(metrics["exact_target_nll_sum"]), rel_tol=1e-12)
        or not math.isclose(exact_perplexity, float(metrics["exact_perplexity"]), rel_tol=5e-6)
    ):
        raise ValueError("compact teacher metrics do not reproduce the capture")

    return {
        "id": fixture["id"],
        "regime": fixture["regime"],
        "token_count": len(tokens),
        "token_ids_sha256": token_digest(tokens),
        "scored_positions": [position + 1 for position in expected_positions],
        "target_logprobs": target_logprobs,
        "teacher_forced_top_token_ids": top_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--capture", required=True, action="append", type=parse_capture)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != "muser.nvfp4-drift-fixtures.v1":
        parser.error("fixture manifest identity is invalid")
    manifest_fixtures = {fixture["id"]: fixture for fixture in manifest["fixtures"]}

    captures = dict(args.capture)
    if len(captures) != len(args.capture):
        parser.error("capture fixture IDs are duplicated")
    if not captures.keys() <= manifest_fixtures.keys():
        parser.error("capture references a fixture ID absent from the manifest")

    results = []
    for fixture_id, capture_path in captures.items():
        if not capture_path.is_file() or capture_path.is_symlink():
            parser.error(f"capture receipt is not a regular file: {capture_path}")
        capture = json.loads(capture_path.read_text(encoding="utf-8"))
        if (
            capture.get("schema") != "muser.llama-perplexity-capture.v1"
            or capture.get("status") != "passed"
        ):
            parser.error(f"perplexity capture is not passed: {fixture_id}")

        llama_receipt_path = capture_path.parent / "llama-receipt.json"
        if not llama_receipt_path.is_file() or llama_receipt_path.is_symlink():
            parser.error(f"llama receipt is not a regular file: {llama_receipt_path}")
        receipt = json.loads(llama_receipt_path.read_text(encoding="utf-8"))
        if receipt.get("schema") != "muser.llama_comparator.source_receipt.v3":
            parser.error(f"llama receipt identity is invalid: {llama_receipt_path}")

        fixture = manifest_fixtures[fixture_id]
        token_file = Path(fixture["token_file"])
        if not token_file.is_absolute():
            token_file = args.manifest.parent / token_file
        tokens = read_tokens(token_file)

        runtime = capture["runtime"]
        if runtime["context_length"] != len(tokens):
            parser.error(f"perplexity context differs from token fixture: {fixture_id}")
        teacher = capture.get("artifacts", {}).get("teacher_evidence", {})
        if isinstance(teacher, dict) and "compact" in teacher:
            try:
                results.append(
                    compact_reference_row(
                        fixture, tokens, capture, receipt, capture_path
                    )
                )
            except (OSError, UnicodeError, ValueError, KeyError) as error:
                parser.error(f"compact teacher evidence is invalid for {fixture_id}: {error}")
        else:
            logits_path = capture_path.parent / "logits.bin"
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
            results.append(reference_row(fixture, tokens, evidence))

    report = {
        "schema": "muser.kquant-content-control-reference.v1",
        "producer_mode": "reference",
        "reference_lane": "tier-1-kquant-llama.cpp",
        "fixture_manifest": str(args.manifest),
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
