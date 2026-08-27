#!/usr/bin/env python3
"""Cross-bind long-context Muser greedy decisions to exact llama rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct

import llama_perplexity_evidence


class GreedyEvidenceError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--muser-evidence", type=Path, required=True)
    parser.add_argument("--prompt-fixture", type=Path, required=True)
    parser.add_argument("--generated-fixture", type=Path, required=True)
    parser.add_argument("--llama-logits", type=Path, required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--llama-upstream-commit", required=True)
    parser.add_argument("--llama-patch-sha256", required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--model-transformer-layers", type=int, default=52)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--require-snapshot-replay", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def token_file(path: Path) -> list[int]:
    try:
        tokens = [int(value) for value in path.read_bytes().split()]
    except (OSError, ValueError) as error:
        raise GreedyEvidenceError(f"invalid token fixture {path}: {error}") from error
    if not tokens or any(token < 0 or token > 0x7FFFFFFF for token in tokens):
        raise GreedyEvidenceError(f"empty or out-of-range token fixture: {path}")
    return tokens


def llama_tokens(path: Path) -> list[int]:
    with path.open("rb") as handle:
        if handle.read(8) != b"_logits_":
            raise GreedyEvidenceError("wrong llama logit magic")
        context, _vocab, chunks = struct.unpack("<Iii", handle.read(12))
        if chunks != 1:
            raise GreedyEvidenceError("greedy oracle requires exactly one llama chunk")
        return list(struct.unpack(f"<{context}i", handle.read(context * 4)))


def muser_rows(
    path: Path, identity: str, require_snapshot_replay: bool
) -> tuple[list[dict], dict]:
    rows = []
    summaries = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise GreedyEvidenceError(f"invalid Muser JSON line {line_number}: {error}") from error
        if value.get("schema") != "muser.greedy-evidence.v1":
            continue
        if value.get("identity") != identity:
            raise GreedyEvidenceError("Muser evidence identity differs")
        if value.get("kind") == "decision":
            rows.append(value)
        elif value.get("kind") == "summary":
            summaries.append(value)
    if len(summaries) != 1 or len(rows) != 64:
        raise GreedyEvidenceError("Muser evidence requires 64 decisions and one summary")
    if summaries[0].get("seal_eligible") is not True:
        raise GreedyEvidenceError("Muser greedy producer did not mark complete evidence")
    summary = summaries[0]
    if require_snapshot_replay and (
        summary.get("snapshot_replay_requested") is not True
        or summary.get("snapshot_position") != summary.get("prompt_tokens")
        or summary.get("snapshot_replay_exact") is not True
        or summary.get("snapshot_replay_generated_tokens_sha256")
        != summary.get("generated_tokens_sha256")
        or summary.get("snapshot_replay_all_logits_sha256")
        != summary.get("all_logits_sha256")
    ):
        raise GreedyEvidenceError(
            "Muser evidence lacks exact detached snapshot replay"
        )
    return rows, summary


def publish(path: Path, value: dict) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    args = parse_args()
    plan = {
        "schema": "muser.greedy-parity.v1",
        "kind": "dry-run" if args.dry_run else "evaluation",
        "accelerator_touched": False,
        "identity": args.identity,
        "context_length": args.context_length,
        "required_output_tokens": 64,
        "policy": "exact-token-and-exact-llama-top1",
        "require_snapshot_replay": args.require_snapshot_replay,
        "seal_eligible": False,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    try:
        prompt = token_file(args.prompt_fixture)
        generated = token_file(args.generated_fixture)
        if len(generated) != 64 or len(prompt) + len(generated) != args.context_length:
            raise GreedyEvidenceError("prompt/generated geometry differs from the cell")
        combined = prompt + generated
        if llama_tokens(args.llama_logits) != combined:
            raise GreedyEvidenceError("llama token stream differs from the exact combined fixture")
        validated = llama_perplexity_evidence.validate_teacher_evidence(
            args.llama_logits,
            expected_upstream_commit=args.llama_upstream_commit,
            expected_patch_sha256=args.llama_patch_sha256,
            expected_context_length=args.context_length,
            expected_chunks=1,
            expected_batch_size=min(args.batch_size, args.context_length),
            expected_ubatch_size=min(args.ubatch_size, args.context_length),
            expected_threads=args.threads,
            expected_kv_cache="f16",
            expected_model_transformer_layers=args.model_transformer_layers,
        )
        rows, summary = muser_rows(
            args.muser_evidence, args.identity, args.require_snapshot_replay
        )
        exact_by_position = {row["position"]: row for row in validated["rows"]}
        mismatches = []
        normalized = []
        for index, (candidate, expected_token) in enumerate(zip(rows, generated, strict=True)):
            position = len(prompt) - 1 + index
            candidates = candidate.get("candidates")
            if (
                candidate.get("index") != index
                or candidate.get("position") != position
                or candidate.get("target_position") != len(prompt) + index
                or candidate.get("selected_token_id") != expected_token
                or not isinstance(candidates, list)
                or len(candidates) != 2
                or candidates[0].get("token_id") != expected_token
                or candidates[0].get("token_id") == candidates[1].get("token_id")
            ):
                raise GreedyEvidenceError(f"invalid Muser decision row {index}")
            llama = exact_by_position.get(position)
            if llama is None or llama.get("target_token_id") != expected_token:
                raise GreedyEvidenceError(f"llama evidence omitted target position {position}")
            llama_top1 = llama["candidates"][0]["token_id"]
            if llama_top1 != expected_token:
                mismatches.append({
                    "index": index,
                    "position": position,
                    "muser": expected_token,
                    "llama": llama_top1,
                })
            normalized.append({
                "index": index,
                "position": position,
                "token_id": expected_token,
                "muser_margin": candidates[0]["logit"] - candidates[1]["logit"],
                "llama_margin": (
                    llama["candidates"][0]["logit"]
                    - llama["candidates"][1]["logit"]
                ),
            })
        if summary.get("generated_token_ids") != generated:
            raise GreedyEvidenceError("Muser summary token vector differs from row evidence")
        report = {
            **plan,
            "mismatches": mismatches,
            "decisions": normalized,
            "muser_all_logits_sha256": summary.get("all_logits_sha256"),
            "snapshot_position": summary.get("snapshot_position"),
            "snapshot_replay_all_logits_sha256": summary.get(
                "snapshot_replay_all_logits_sha256"
            ),
            "snapshot_replay_exact": summary.get("snapshot_replay_exact"),
            "llama_artifacts": validated["artifacts"],
            "combined_tokens_sha256": hashlib.sha256(
                b"".join(token.to_bytes(4, "little") for token in combined)
            ).hexdigest(),
            "status": "passed" if not mismatches else "failed",
            "seal_eligible": not mismatches,
        }
    except (
        OSError,
        ValueError,
        KeyError,
        struct.error,
        GreedyEvidenceError,
        llama_perplexity_evidence.LlamaPerplexityEvidenceError,
    ) as error:
        report = {**plan, "status": "failed", "failures": [str(error)]}
    publish(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
