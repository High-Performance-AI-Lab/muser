#!/usr/bin/env python3
"""Capture one receipt-bound llama-perplexity teacher-evidence cell."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import llama_perplexity_evidence
from nvfp4_to_f16_gguf import write_receipt
import representative_target_smoke as base


PINNED_LLAMA_COMMIT = "89e0aa6fd362617d9073e0dafc18e41241521572"
COMPACT_TEACHER_SCHEMA = "muser.llama-perplexity-compact-teacher.v1"


def token_file(path: Path) -> list[int]:
    try:
        values = [int(value) for value in path.read_bytes().split()]
    except (OSError, ValueError) as error:
        raise RuntimeError(f"invalid token fixture {path}: {error}") from error
    if not values or any(not 0 <= value <= 0x7FFFFFFF for value in values):
        raise RuntimeError(f"empty or out-of-range token fixture: {path}")
    return values


def compact_teacher_report(evidence: dict) -> dict:
    """Reduce a fully validated raw-logits bundle to the rows consumers use.

    ``validate_teacher_evidence`` has already cross-bound every exact row to
    the full-vocabulary uint16 artifact.  The compact report retains that
    validation's source hashes plus the exact target NLL and top-1 token used
    by the content-control comparator, so the multi-gigabyte raw file can stay
    operational scratch rather than append-only campaign evidence.
    """
    header = evidence["header"]
    rows = evidence["rows"]
    metrics = evidence["metrics"]
    if not rows or not math.isfinite(float(metrics["exact_perplexity"])):
        raise RuntimeError("validated teacher evidence is empty or non-finite")
    return {
        "schema": COMPACT_TEACHER_SCHEMA,
        "status": "validated",
        "validation": {
            "validator": (
                "scripts/llama_perplexity_evidence.py::validate_teacher_evidence"
            ),
            "quantized_cross_binding": "validated-before-compaction",
            "upstream_commit": header["upstream_commit"],
            "patch_sha256": header["patch_sha256"],
            "evidence_id": header["evidence_id"],
            "source_artifacts": evidence["artifacts"],
        },
        "geometry": {
            "context_length": header["context_length"],
            "vocab_size": header["vocab_size"],
            "chunks": header["chunks"],
            "scored_rows": header["scored_rows"],
        },
        "metrics": metrics,
        "rows": [
            {
                "chunk": row["chunk"],
                "position": row["position"],
                "input_token_id": row["input_token_id"],
                "target_token_id": row["target_token_id"],
                "target_nll": row["target_nll"],
                "teacher_forced_top_token_id": row["candidates"][0]["token_id"],
            }
            for row in rows
        ],
    }


def require_scratch_path(path: Path, scratch_root: Path) -> None:
    """Constrain later cleanup to exact files below an explicit scratch root."""
    root = scratch_root.resolve()
    candidate = path.resolve()
    if scratch_root.is_symlink() or not scratch_root.is_dir():
        raise RuntimeError(f"scratch root is not a real directory: {scratch_root}")
    if candidate == root or not candidate.is_relative_to(root):
        raise RuntimeError(f"scratch artifact escapes --scratch-root: {path}")


def remove_raw_scratch(logits_path: Path, scratch_root: Path) -> None:
    """Remove only the three validated raw siblings from internal scratch."""
    paths = (
        logits_path,
        llama_perplexity_evidence.top10_path_for(logits_path),
        llama_perplexity_evidence.runtime_path_for(logits_path),
    )
    for path in paths:
        require_scratch_path(path, scratch_root)
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"validated scratch artifact is not a regular file: {path}")
    for path in paths:
        path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--model-receipt", type=Path)
    parser.add_argument("--token-fixture", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--ubatch-size", type=int, required=True)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--llama-perplexity", type=Path, required=True)
    parser.add_argument("--llama-receipt", type=Path, required=True)
    parser.add_argument("--logits-out", type=Path, required=True)
    parser.add_argument(
        "--compact-teacher-output",
        type=Path,
        help="retain a compact, cross-bound teacher report after raw validation",
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        help="explicit root containing --logits-out when raw scratch is discarded",
    )
    parser.add_argument(
        "--discard-raw-after-compact",
        action="store_true",
        help="delete only the validated raw-logits siblings below --scratch-root",
    )
    parser.add_argument("--command-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.discard_raw_after_compact and (
        args.compact_teacher_output is None or args.scratch_root is None
    ):
        raise SystemExit(
            "--discard-raw-after-compact requires --compact-teacher-output and "
            "--scratch-root"
        )
    if args.scratch_root is not None:
        args.scratch_root.mkdir(parents=True, exist_ok=True)
        try:
            require_scratch_path(args.logits_out, args.scratch_root)
        except RuntimeError as error:
            raise SystemExit(str(error)) from error
    if min(args.context_length, args.batch_size, args.ubatch_size, args.threads) <= 0:
        raise SystemExit("runtime geometry must be positive")
    if args.batch_size > args.context_length or args.ubatch_size > args.batch_size:
        raise SystemExit("runtime batch geometry is inconsistent")
    tokens = token_file(args.token_fixture)
    if len(tokens) != args.context_length:
        raise SystemExit("token fixture length differs from --context-length")

    model = base.checked_file(args.model, "target model")
    if model["sha256"] != args.expected_model_sha256:
        raise SystemExit("target model differs from --expected-model-sha256")
    model_receipt = None
    if args.model_receipt is not None:
        model_receipt = base.checked_file(args.model_receipt, "model receipt")
        receipt = json.loads(args.model_receipt.read_text())
        if (
            receipt.get("schema") != "muser.nvfp4-f16-gguf-repack.v4"
            or Path(receipt.get("out", "")).resolve() != args.model.resolve()
            or receipt.get("sha256") != model["sha256"]
        ):
            raise SystemExit("model receipt does not bind the target model")

    binary = base.checked_file(args.llama_perplexity, "llama-perplexity")
    llama_receipt_file = base.checked_file(
        args.llama_receipt, "llama comparator receipt"
    )
    llama_receipt = json.loads(args.llama_receipt.read_text())
    expected_binary = llama_receipt.get("artifacts", {}).get("llama-perplexity", {})
    if (
        llama_receipt.get("schema") != "muser.llama_comparator.source_receipt.v3"
        or llama_receipt.get("source_commit") != PINNED_LLAMA_COMMIT
        or llama_receipt.get("executed") is not False
        or llama_receipt.get("build", {}).get("metal") is not True
        or expected_binary.get("bytes") != binary["bytes"]
        or expected_binary.get("sha256") != binary["sha256"]
    ):
        raise SystemExit("llama-perplexity differs from the pinned comparator")

    fixture = base.checked_file(args.token_fixture, "token fixture")
    corpus = base.checked_file(args.corpus, "corpus placeholder")
    siblings = (
        args.logits_out,
        llama_perplexity_evidence.top10_path_for(args.logits_out),
        llama_perplexity_evidence.runtime_path_for(args.logits_out),
        args.command_log,
        args.output,
    ) + ((args.compact_teacher_output,) if args.compact_teacher_output else ())
    for path in siblings:
        path.parent.mkdir(parents=True, exist_ok=True)
        base.validate_output_path(path)

    command = [
        str(args.llama_perplexity),
        "-m", str(args.model),
        "-f", str(args.corpus),
        "-c", str(args.context_length),
        "-b", str(args.batch_size),
        "-ub", str(args.ubatch_size),
        "-t", str(args.threads),
        "-ngl", "99",
        "-fa", "1",
        "-ctk", "f16",
        "-ctv", "f16",
        "--chunks", "1",
        "--save-all-logits", str(args.logits_out),
    ]
    environment = os.environ.copy()
    environment["MUSER_COMPARATOR_TOKEN_FIXTURE"] = str(args.token_fixture.resolve())
    with args.command_log.open("xb") as log:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            timeout=args.timeout_seconds,
            check=False,
        )
        log.flush()
        os.fsync(log.fileno())
    if completed.returncode != 0:
        raise SystemExit(f"llama-perplexity exited {completed.returncode}")

    evidence = llama_perplexity_evidence.validate_teacher_evidence(
        args.logits_out,
        expected_upstream_commit=PINNED_LLAMA_COMMIT,
        expected_patch_sha256=llama_receipt["patch_sha256"],
        expected_context_length=args.context_length,
        expected_chunks=1,
        expected_batch_size=args.batch_size,
        expected_ubatch_size=args.ubatch_size,
        expected_threads=args.threads,
        expected_kv_cache="f16",
        expected_model_transformer_layers=52,
        runtime_route="full-gpu",
    )
    compact_artifact = None
    source_artifacts_retained = True
    if args.compact_teacher_output is not None:
        compact = compact_teacher_report(evidence)
        write_receipt(args.compact_teacher_output, compact)
        compact_artifact = base.checked_file(
            args.compact_teacher_output, "compact teacher evidence"
        )
        if args.discard_raw_after_compact:
            try:
                remove_raw_scratch(args.logits_out, args.scratch_root)
            except RuntimeError as error:
                raise SystemExit(str(error)) from error
            source_artifacts_retained = False

    teacher_evidence = dict(evidence["artifacts"])
    if compact_artifact is not None:
        teacher_evidence["compact"] = compact_artifact
        teacher_evidence["source_artifacts_retained"] = source_artifacts_retained
    report = {
        "schema": "muser.llama-perplexity-capture.v1",
        "status": "passed",
        "seal_eligible": False,
        "accelerator_touched": True,
        "identity": args.identity,
        "runtime": {
            "context_length": args.context_length,
            "batch_size": args.batch_size,
            "ubatch_size": args.ubatch_size,
            "threads": args.threads,
            "chunks": 1,
            "gpu_layers": 99,
            "flash_attention": True,
            "kv_cache": "f16",
        },
        "artifacts": {
            "model": model,
            "model_receipt": model_receipt,
            "token_fixture": fixture,
            "corpus_placeholder": corpus,
            "llama_perplexity": binary,
            "llama_receipt": llama_receipt_file,
            "command_log": base.checked_file(args.command_log, "command log"),
            "teacher_evidence": teacher_evidence,
        },
        "metrics": evidence["metrics"],
    }
    write_receipt(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
