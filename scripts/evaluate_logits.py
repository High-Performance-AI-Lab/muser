#!/usr/bin/env python3
"""Evaluate Muse CPU/Metal and llama.cpp teacher-forced logit evidence.

The llama saved-logit parser and floor-aware probability reconstruction are a
Muse-only reduction of Ferrite's accepted `scripts/qwen25_logit_parity.py`.
Raw CPU/Metal f32 rows provide the absolute-logit gate; llama's compact u16
rows provide target-NLL and rank-overlap gates on the exact same token stream.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
from typing import BinaryIO

import llama_perplexity_evidence


LLAMA_MAGIC = b"_logits_"
MUSER_MAGIC = b"MUSLOG1\0"
TOP_K = 10


class EvidenceError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-rows", type=Path, required=True)
    parser.add_argument("--candidate-logits", type=Path, required=True)
    parser.add_argument("--cpu-logits", type=Path, required=True)
    parser.add_argument("--llama-logits", type=Path, required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--llama-upstream-commit", required=True)
    parser.add_argument("--llama-patch-sha256", required=True)
    parser.add_argument("--model-transformer-layers", type=int, default=52)
    parser.add_argument(
        "--llama-route", choices=("full-gpu", "cpu-only"), default="full-gpu"
    )
    parser.add_argument("--report", type=Path)
    parser.add_argument("--maximum-cpu-logit-error", type=float, default=0.5)
    parser.add_argument("--maximum-relative-target-nll-error", type=float, default=0.005)
    parser.add_argument("--minimum-top1-agreement", type=float, default=0.985)
    parser.add_argument("--minimum-mean-top10-overlap", type=float, default=0.90)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_exact(handle: BinaryIO, count: int, label: str) -> bytes:
    value = handle.read(count)
    if len(value) != count:
        raise EvidenceError(f"truncated {label}: expected {count} bytes, got {len(value)}")
    return value


def read_muser_raw(path: Path) -> dict:
    with path.open("rb") as handle:
        if read_exact(handle, 8, "Muser magic") != MUSER_MAGIC:
            raise EvidenceError(f"wrong Muser raw-logit magic: {path}")
        version, context, vocab, rows = struct.unpack(
            "<4I", read_exact(handle, 16, "Muser header")
        )
        if version != 1 or context < 4 or vocab < TOP_K or rows != context - 1 - context // 2:
            raise EvidenceError(f"invalid Muser raw-logit geometry: {path}")
        tokens = list(struct.unpack(f"<{context}I", read_exact(handle, context * 4, "tokens")))
        payload = read_exact(handle, rows * vocab * 4, "Muser logit rows")
        if handle.read(1):
            raise EvidenceError(f"trailing bytes in Muser raw logits: {path}")
    return {
        "context": context,
        "vocab": vocab,
        "rows": rows,
        "tokens": tokens,
        "payload": payload,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def raw_cpu_metrics(cpu: dict, candidate: dict) -> dict:
    for field in ("context", "vocab", "rows", "tokens"):
        if cpu[field] != candidate[field]:
            raise EvidenceError(f"CPU/candidate raw-logit {field} differs")
    left = array.array("f")
    right = array.array("f")
    left.frombytes(cpu["payload"])
    right.frombytes(candidate["payload"])
    if sys.byteorder != "little":
        left.byteswap()
        right.byteswap()
    nonfinite = sum(not math.isfinite(value) for value in left) + sum(
        not math.isfinite(value) for value in right
    )
    if nonfinite:
        return {"maximum_absolute_error": math.inf, "nonfinite_values": nonfinite}
    maximum = max((abs(a - b) for a, b in zip(left, right, strict=True)), default=0.0)
    return {"maximum_absolute_error": maximum, "nonfinite_values": 0}


def read_rows(path: Path, identity: str) -> list[dict]:
    documents = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise EvidenceError(f"invalid candidate JSON at line {line_number}: {error}") from error
        if value.get("kind") == "row":
            if (
                value.get("schema") != "muser.forward-evidence.v1"
                or value.get("identity") != identity
                or not isinstance(value.get("top_k"), list)
                or len(value["top_k"]) != TOP_K
                or not math.isfinite(float(value.get("target_nll", math.nan)))
            ):
                raise EvidenceError(f"invalid candidate row at line {line_number}")
            ids = [item.get("token") for item in value["top_k"]]
            logits = [item.get("logit") for item in value["top_k"]]
            if (
                any(not isinstance(token, int) or token < 0 for token in ids)
                or len(set(ids)) != TOP_K
                or any(not isinstance(logit, (int, float)) or not math.isfinite(logit) for logit in logits)
                or any(left < right for left, right in zip(logits, logits[1:]))
            ):
                raise EvidenceError(f"invalid candidate top-10 at line {line_number}")
            documents.append(value)
    if not documents:
        raise EvidenceError("candidate row evidence is empty")
    return documents


def llama_metrics(
    path: Path, candidate: dict, rows: list[dict], exact_evidence: dict
) -> dict:
    exact_rows = exact_evidence["rows"]
    with path.open("rb") as handle:
        if read_exact(handle, 8, "llama magic") != LLAMA_MAGIC:
            raise EvidenceError("wrong llama saved-logit magic")
        context, vocab, chunks = struct.unpack("<Iii", read_exact(handle, 12, "llama header"))
        if chunks != 1 or context != candidate["context"] or vocab != candidate["vocab"]:
            raise EvidenceError("llama and Muser logit geometry differs")
        tokens = list(struct.unpack(f"<{context}i", read_exact(handle, context * 4, "llama tokens")))
        if tokens != candidate["tokens"]:
            raise EvidenceError("llama and Muser teacher token streams differ")
        expected_rows = context - 1 - context // 2
        if len(rows) != expected_rows:
            raise EvidenceError("candidate row count differs from llama score window")
        if len(exact_rows) != expected_rows:
            raise EvidenceError("exact llama row count differs from score window")
        padded_vocab = 2 * ((vocab + 1) // 2)
        candidate_nll = 0.0
        llama_nll = 0.0
        top1_matches = 0
        overlaps = []
        centered_errors = []
        for row_index, position in enumerate(range(context // 2, context - 1)):
            row = rows[row_index]
            exact = exact_rows[row_index]
            if (
                row.get("window") != 0
                or row.get("pos") != position
                or row.get("input_token") != tokens[position]
                or row.get("target_token") != tokens[position + 1]
            ):
                raise EvidenceError(f"candidate position/token witness differs at {position}")
            scale, floor = struct.unpack("<ff", read_exact(handle, 8, "llama row scale"))
            if not math.isfinite(scale) or scale < 0 or not math.isfinite(floor):
                raise EvidenceError(f"nonfinite llama row scale at {position}")
            read_exact(handle, padded_vocab * 2, "llama row")
            candidate_top = [item["token"] for item in row["top_k"]]
            llama_top = [item["token_id"] for item in exact["candidates"]]
            top1_matches += candidate_top[0] == llama_top[0]
            overlaps.append(len(set(llama_top).intersection(candidate_top)) / TOP_K)
            candidate_logits = {item["token"]: float(item["logit"]) for item in row["top_k"]}
            llama_logits = {
                item["token_id"]: float(item["logit"])
                for item in exact["candidates"]
            }
            common = [token for token in candidate_top if token in llama_logits]
            if common:
                anchor = common[0]
                for token in common:
                    centered_errors.append(
                        abs(
                            (candidate_logits[token] - candidate_logits[anchor])
                            - (llama_logits[token] - llama_logits[anchor])
                        )
                    )
            candidate_nll += float(row["target_nll"])
            llama_nll += float(exact["target_nll"])
        if handle.read(1):
            raise EvidenceError("trailing bytes in llama saved logits")
    if (
        len(overlaps) != expected_rows
        or not centered_errors
        or llama_nll <= 0
        or not math.isfinite(llama_nll)
    ):
        raise EvidenceError("llama evidence has insufficient recoverable rank/NLL rows")
    return {
        "scored_rows": expected_rows,
        "candidate_target_nll_sum": candidate_nll,
        "llama_target_nll_sum": llama_nll,
        "relative_target_nll_error": abs(candidate_nll - llama_nll) / llama_nll,
        "top1_agreement": top1_matches / expected_rows,
        "mean_top10_overlap": sum(overlaps) / len(overlaps),
        "top10_rows_recoverable": len(overlaps),
        "maximum_centered_logit_error": max(centered_errors),
        "exact_evidence_artifacts": exact_evidence["artifacts"],
    }


def publish(path: Path, value: dict) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    args = parse_args()
    gates = {
        "maximum_cpu_logit_error": args.maximum_cpu_logit_error,
        "maximum_relative_target_nll_error": args.maximum_relative_target_nll_error,
        "minimum_top1_agreement": args.minimum_top1_agreement,
        "minimum_mean_top10_overlap": args.minimum_mean_top10_overlap,
        "nonfinite_values": 0,
    }
    plan = {
        "schema": "muser.logit-parity.evaluation.v1",
        "kind": "dry-run",
        "accelerator_touched": False,
        "identity": args.identity,
        "inputs": {
            "candidate_rows": str(args.candidate_rows),
            "candidate_logits": str(args.candidate_logits),
            "cpu_logits": str(args.cpu_logits),
            "llama_logits": str(args.llama_logits),
        },
        "gates": gates,
        "seal_eligible": False,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    try:
        cpu = read_muser_raw(args.cpu_logits)
        candidate = read_muser_raw(args.candidate_logits)
        raw_metrics = raw_cpu_metrics(cpu, candidate)
        rows = read_rows(args.candidate_rows, args.identity)
        exact_evidence = llama_perplexity_evidence.validate_teacher_evidence(
            args.llama_logits,
            expected_upstream_commit=args.llama_upstream_commit,
            expected_patch_sha256=args.llama_patch_sha256,
            expected_context_length=candidate["context"],
            expected_chunks=1,
            expected_batch_size=66,
            expected_ubatch_size=66,
            expected_threads=20,
            expected_kv_cache="f16",
            expected_model_transformer_layers=args.model_transformer_layers,
            runtime_route=args.llama_route,
        )
        reference_metrics = llama_metrics(
            args.llama_logits, candidate, rows, exact_evidence
        )
        failures = []
        if raw_metrics["nonfinite_values"] != 0:
            failures.append("nonfinite CPU/Metal logits")
        if raw_metrics["maximum_absolute_error"] > args.maximum_cpu_logit_error:
            failures.append("CPU/Metal maximum absolute logit error")
        if reference_metrics["relative_target_nll_error"] > args.maximum_relative_target_nll_error:
            failures.append("relative target-NLL error")
        if reference_metrics["top1_agreement"] < args.minimum_top1_agreement:
            failures.append("top-1 agreement")
        if reference_metrics["mean_top10_overlap"] < args.minimum_mean_top10_overlap:
            failures.append("mean top-10 overlap")
        report = {
            **plan,
            "kind": "evaluation",
            "raw_cpu_candidate": raw_metrics,
            "llama_reference": reference_metrics,
            "failures": failures,
            "status": "passed" if not failures else "failed",
            "seal_eligible": not failures,
        }
    except (
        OSError,
        ValueError,
        struct.error,
        json.JSONDecodeError,
        EvidenceError,
        llama_perplexity_evidence.LlamaPerplexityEvidenceError,
    ) as error:
        report = {**plan, "kind": "evaluation", "status": "failed", "failures": [str(error)]}
    if args.report is not None:
        publish(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
