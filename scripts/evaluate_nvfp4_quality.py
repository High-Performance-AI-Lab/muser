#!/usr/bin/env python3
"""Measure the P0 NVFP4 drift envelope against the kquant Tier-1 anchor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import llama_perplexity_evidence


PINNED_LLAMA_COMMIT = "89e0aa6fd362617d9073e0dafc18e41241521572"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def token_file(path: Path) -> list[int]:
    try:
        result = [int(value) for value in path.read_bytes().split()]
    except (OSError, ValueError) as error:
        raise RuntimeError(f"invalid token fixture {path}: {error}") from error
    if not result or any(token < 0 or token > 0x7FFFFFFF for token in result):
        raise RuntimeError(f"empty or out-of-range token fixture: {path}")
    return result


def logits_tokens(path: Path) -> list[int]:
    with path.open("rb") as stream:
        if stream.read(8) != b"_logits_":
            raise RuntimeError(f"wrong llama logits magic: {path}")
        context, _vocab, chunks = struct.unpack("<Iii", stream.read(12))
        if chunks != 1:
            raise RuntimeError("P0 quality cells require exactly one chunk")
        return list(struct.unpack(f"<{context}i", stream.read(context * 4)))


def approximate_logprob(row: dict, candidate: dict) -> float:
    return float(row["minimum_log_probability"]) + float(row["row_scale"]) * int(
        candidate["quantized_u16"]
    )


def compare_rows(nvfp4_rows: list[dict], kquant_rows: list[dict]) -> dict[str, object]:
    if len(nvfp4_rows) != len(kquant_rows) or not nvfp4_rows:
        raise RuntimeError("teacher evidence row counts differ or are empty")
    top1_mismatches = 0
    top10_overlap = 0
    nll_deltas = []
    for index, (nvfp4, kquant) in enumerate(zip(nvfp4_rows, kquant_rows, strict=True)):
        identity = ("chunk", "position", "input_token_id", "target_token_id")
        if any(nvfp4[key] != kquant[key] for key in identity):
            raise RuntimeError(f"teacher row identity mismatch at {index}")
        nv_ids = [int(value["token_id"]) for value in nvfp4["candidates"]]
        kq_ids = [int(value["token_id"]) for value in kquant["candidates"]]
        top1_mismatches += nv_ids[0] != kq_ids[0]
        top10_overlap += len(set(nv_ids) & set(kq_ids))
        nll_deltas.append(float(nvfp4["target_nll"]) - float(kquant["target_nll"]))

    boundary_nvfp4 = nvfp4_rows[-1]
    boundary_kquant = kquant_rows[-1]
    nv_by_id = {
        int(value["token_id"]): value for value in boundary_nvfp4["candidates"]
    }
    kq_by_id = {
        int(value["token_id"]): value for value in boundary_kquant["candidates"]
    }
    intersection = sorted(set(nv_by_id) & set(kq_by_id))
    logprob_deltas = [
        approximate_logprob(boundary_nvfp4, nv_by_id[token])
        - approximate_logprob(boundary_kquant, kq_by_id[token])
        for token in intersection
    ]
    nv_origin = float(boundary_nvfp4["candidates"][0]["logit"])
    kq_origin = float(boundary_kquant["candidates"][0]["logit"])
    centered_logit_deltas = [
        (float(nv_by_id[token]["logit"]) - nv_origin)
        - (float(kq_by_id[token]["logit"]) - kq_origin)
        for token in intersection
    ]

    def boundary_view(row: dict) -> list[dict[str, object]]:
        return [
            {
                "token_id": int(value["token_id"]),
                "logit": float(value["logit"]),
                "approx_logprob": approximate_logprob(row, value),
            }
            for value in row["candidates"]
        ]

    count = len(nvfp4_rows)
    return {
        "scored_rows": count,
        "teacher_forced_greedy_divergences": top1_mismatches,
        "teacher_forced_greedy_divergence_rate": top1_mismatches / count,
        "mean_top10_overlap": top10_overlap / (10 * count),
        "target_nll_drift": {
            "mean_signed": math.fsum(nll_deltas) / count,
            "mean_absolute": math.fsum(abs(value) for value in nll_deltas) / count,
            "maximum_absolute": max(abs(value) for value in nll_deltas),
        },
        "boundary": {
            "position": int(boundary_nvfp4["position"]),
            "target_token_id": int(boundary_nvfp4["target_token_id"]),
            "argmax_equal": (
                boundary_nvfp4["candidates"][0]["token_id"]
                == boundary_kquant["candidates"][0]["token_id"]
            ),
            "top10_intersection": len(intersection),
            "intersection_token_ids": intersection,
            "maximum_approx_logprob_drift": (
                max(abs(value) for value in logprob_deltas) if logprob_deltas else None
            ),
            "mean_absolute_approx_logprob_drift": (
                math.fsum(abs(value) for value in logprob_deltas) / len(logprob_deltas)
                if logprob_deltas
                else None
            ),
            "maximum_centered_logit_drift": (
                max(abs(value) for value in centered_logit_deltas)
                if centered_logit_deltas
                else None
            ),
            "nvfp4_top10": boundary_view(boundary_nvfp4),
            "kquant_top10": boundary_view(boundary_kquant),
        },
    }


def greedy_metrics(nvfp4_path: Path, kquant_path: Path) -> dict[str, object]:
    nvfp4 = token_file(nvfp4_path)
    kquant = token_file(kquant_path)
    width = max(len(nvfp4), len(kquant))
    mismatches = sum(
        index >= len(nvfp4)
        or index >= len(kquant)
        or nvfp4[index] != kquant[index]
        for index in range(width)
    )
    first = next(
        (
            index
            for index in range(width)
            if index >= len(nvfp4)
            or index >= len(kquant)
            or nvfp4[index] != kquant[index]
        ),
        None,
    )
    return {
        "nvfp4_tokens": len(nvfp4),
        "kquant_tokens": len(kquant),
        "mismatches": mismatches,
        "divergence_rate": mismatches / width if width else None,
        "first_mismatch": first,
        "nvfp4_sha256": sha256(nvfp4_path),
        "kquant_sha256": sha256(kquant_path),
    }


def capture_report(
    path: Path,
    *,
    identity: str,
    mode: str,
    require_model_receipt: bool,
) -> dict[str, object]:
    value = json.loads(path.read_text())
    if (
        value.get("schema") != "muser.llama-quality-capture.v1"
        or value.get("status") != "passed"
        or value.get("identity") != identity
        or value.get("mode") != mode
        or value.get("accelerator_touched") is not True
        or value.get("output_tokens_generated") != 256
    ):
        raise RuntimeError(f"capture report contract mismatch: {path}")
    artifacts = value.get("artifacts")
    outputs = value.get("outputs")
    if not isinstance(artifacts, dict) or not isinstance(outputs, dict):
        raise RuntimeError(f"capture report omitted artifacts or outputs: {path}")
    if require_model_receipt and not isinstance(artifacts.get("model_receipt"), dict):
        raise RuntimeError(f"NVFP4 capture omitted its model receipt: {path}")
    return value


def perplexity_capture(
    path: Path,
    *,
    logits_path: Path,
    identity: str,
    context_length: int,
    batch_size: int,
    ubatch_size: int,
    threads: int,
    require_model_receipt: bool,
) -> dict[str, object]:
    value = json.loads(path.read_text())
    runtime = value.get("runtime")
    artifacts = value.get("artifacts")
    expected_runtime = {
        "context_length": context_length,
        "batch_size": batch_size,
        "ubatch_size": ubatch_size,
        "threads": threads,
        "chunks": 1,
        "gpu_layers": 99,
        "flash_attention": True,
        "kv_cache": "f16",
    }
    if (
        value.get("schema") != "muser.llama-perplexity-capture.v1"
        or value.get("status") != "passed"
        or value.get("identity") != identity
        or value.get("accelerator_touched") is not True
        or runtime != expected_runtime
        or not isinstance(artifacts, dict)
    ):
        raise RuntimeError(f"perplexity capture contract mismatch: {path}")
    if require_model_receipt and not isinstance(artifacts.get("model_receipt"), dict):
        raise RuntimeError(f"NVFP4 perplexity capture omitted model receipt: {path}")
    teacher = artifacts.get("teacher_evidence")
    logits = teacher.get("quantized_logits") if isinstance(teacher, dict) else None
    if not isinstance(logits, dict) or logits != {
        "sha256": sha256(logits_path),
        "size_bytes": logits_path.stat().st_size,
    }:
        raise RuntimeError(f"perplexity capture does not bind logits: {path}")
    return value


def dflash_metrics(
    nvfp4_path: Path,
    kquant_path: Path,
    *,
    identity: str,
) -> dict[str, object]:
    def read(path: Path, *, require_model_receipt: bool) -> dict[str, object]:
        value = capture_report(
            path,
            identity=identity,
            mode="dflash",
            require_model_receipt=require_model_receipt,
        )
        if (
            value.get("expected_tokens_match") is not True
            or value.get("dflash_route_active") is not True
            or not isinstance(value["artifacts"].get("expected_tokens"), dict)
        ):
            raise RuntimeError(f"DFlash report is not bound to a target stream: {path}")
        timings = value.get("llama", value).get("timings")
        if not isinstance(timings, dict):
            raise RuntimeError(f"DFlash evidence omitted llama timings: {path}")
        drafted = timings.get("draft_n")
        accepted = timings.get("draft_n_accepted")
        if (
            not isinstance(drafted, int)
            or not isinstance(accepted, int)
            or drafted <= 0
            or not 0 <= accepted <= drafted
        ):
            raise RuntimeError(f"invalid DFlash counters: {path}")
        return {
            "drafted": drafted,
            "accepted": accepted,
            "acceptance": accepted / drafted,
            "report": str(path.resolve()),
            "report_sha256": sha256(path),
            "expected_tokens": value["artifacts"]["expected_tokens"],
        }

    nvfp4 = read(nvfp4_path, require_model_receipt=True)
    kquant = read(kquant_path, require_model_receipt=False)
    return {
        "nvfp4": nvfp4,
        "kquant": kquant,
        "acceptance_delta": float(nvfp4["acceptance"]) - float(kquant["acceptance"]),
    }


def publish(path: Path, value: dict[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to replace report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvfp4-logits", type=Path, required=True)
    parser.add_argument("--kquant-logits", type=Path, required=True)
    parser.add_argument("--nvfp4-capture", type=Path, required=True)
    parser.add_argument("--kquant-capture", type=Path, required=True)
    parser.add_argument("--llama-receipt", type=Path, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--nvfp4-greedy-tokens", type=Path)
    parser.add_argument("--kquant-greedy-tokens", type=Path)
    parser.add_argument("--nvfp4-dflash", type=Path)
    parser.add_argument("--kquant-dflash", type=Path)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = json.loads(args.llama_receipt.read_text())
    if (
        receipt.get("schema") != "muser.llama_comparator.source_receipt.v3"
        or receipt.get("source_commit") != PINNED_LLAMA_COMMIT
    ):
        raise SystemExit("llama receipt is not the pinned comparator")
    patch_sha256 = receipt["patch_sha256"]
    nvfp4_capture = perplexity_capture(
        args.nvfp4_capture,
        logits_path=args.nvfp4_logits,
        identity=args.identity,
        context_length=args.context_length,
        batch_size=min(args.batch_size, args.context_length),
        ubatch_size=min(args.ubatch_size, args.context_length),
        threads=args.threads,
        require_model_receipt=True,
    )
    kquant_capture = perplexity_capture(
        args.kquant_capture,
        logits_path=args.kquant_logits,
        identity=args.identity,
        context_length=args.context_length,
        batch_size=min(args.batch_size, args.context_length),
        ubatch_size=min(args.ubatch_size, args.context_length),
        threads=args.threads,
        require_model_receipt=False,
    )
    common = {
        "expected_upstream_commit": PINNED_LLAMA_COMMIT,
        "expected_patch_sha256": patch_sha256,
        "expected_context_length": args.context_length,
        "expected_chunks": 1,
        "expected_batch_size": min(args.batch_size, args.context_length),
        "expected_ubatch_size": min(args.ubatch_size, args.context_length),
        "expected_threads": args.threads,
        "expected_kv_cache": "f16",
        "expected_model_transformer_layers": 52,
        "runtime_route": "full-gpu",
    }
    nvfp4 = llama_perplexity_evidence.validate_teacher_evidence(
        args.nvfp4_logits, **common
    )
    kquant = llama_perplexity_evidence.validate_teacher_evidence(
        args.kquant_logits, **common
    )
    if logits_tokens(args.nvfp4_logits) != logits_tokens(args.kquant_logits):
        raise SystemExit("NVFP4 and kquant evidence use different token streams")
    row_metrics = compare_rows(nvfp4["rows"], kquant["rows"])
    nv_perplexity = float(nvfp4["metrics"]["exact_perplexity"])
    kq_perplexity = float(kquant["metrics"]["exact_perplexity"])
    report: dict[str, object] = {
        "schema": "muser.nvfp4-quality-drift.v1",
        "status": "measured",
        "seal_eligible": False,
        "identity": args.identity,
        "context_length": args.context_length,
        "perplexity": {
            "nvfp4": nv_perplexity,
            "kquant": kq_perplexity,
            "absolute_delta": nv_perplexity - kq_perplexity,
            "relative_delta": (nv_perplexity - kq_perplexity) / kq_perplexity,
        },
        "teacher_forced": row_metrics,
        "artifacts": {
            "nvfp4": nvfp4["artifacts"],
            "kquant": kquant["artifacts"],
            "llama_receipt_sha256": sha256(args.llama_receipt),
            "nvfp4_capture": {
                "path": str(args.nvfp4_capture.resolve()),
                "sha256": sha256(args.nvfp4_capture),
                "model": nvfp4_capture["artifacts"]["model"],
                "model_receipt": nvfp4_capture["artifacts"]["model_receipt"],
            },
            "kquant_capture": {
                "path": str(args.kquant_capture.resolve()),
                "sha256": sha256(args.kquant_capture),
                "model": kquant_capture["artifacts"]["model"],
            },
        },
    }
    if (args.nvfp4_greedy_tokens is None) != (args.kquant_greedy_tokens is None):
        raise SystemExit("both greedy token fixtures must be supplied together")
    if args.nvfp4_greedy_tokens is not None:
        report["greedy_stream"] = greedy_metrics(
            args.nvfp4_greedy_tokens, args.kquant_greedy_tokens
        )
    if (args.nvfp4_dflash is None) != (args.kquant_dflash is None):
        raise SystemExit("both DFlash results must be supplied together")
    if args.nvfp4_dflash is not None:
        report["dflash"] = dflash_metrics(
            args.nvfp4_dflash,
            args.kquant_dflash,
            identity=args.identity,
        )
    publish(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
