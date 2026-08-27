#!/usr/bin/env python3
"""Canonical, GPU-free comparison of retained GX speculative traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


SUMMARY_SCHEMA = "muser.gx10-strict-diagnostic-summary.v1"
RECEIPT_SCHEMA = "muser.gx10-container.receipt.v1"
RESULT_SCHEMA = "muser.gx-trace-diff.v1"
TRACE_FIELDS = {
    "accepted_local_sha256",
    "accepted_remote_sha256",
    "draft_local_sha256",
    "draft_remote_sha256",
    "first_count_divergence_round",
    "first_proposal_token_divergence",
    "local_acceptance",
    "local_accepted",
    "local_accepted_prefix_counts",
    "local_drafted",
    "remote_acceptance",
    "remote_accepted",
    "remote_accepted_prefix_counts",
    "remote_drafted",
}


class TraceError(RuntimeError):
    pass


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TraceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise TraceError(f"input must be a regular non-symlink file: {path}")
    try:
        value = json.loads(path.read_text(), object_pairs_hook=reject_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TraceError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TraceError(f"top-level JSON must be an object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_digest(tokens: list[int]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        if isinstance(token, bool) or not isinstance(token, int) or not 0 <= token <= 0xFFFFFFFF:
            raise TraceError("trace tokens must be u32 integers")
        digest.update(token.to_bytes(4, "little"))
    return digest.hexdigest()


def counts(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise TraceError(f"{label} must be a non-empty integer array")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value):
        raise TraceError(f"{label} must contain non-negative integers")
    return value


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TraceError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise TraceError(f"{label} must be finite")
    return result


def first_mismatch(left: list[int], right: list[int]) -> int | None:
    for index, (left_item, right_item) in enumerate(zip(left, right)):
        if left_item != right_item:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def validate_side(trace: dict[str, Any], side: str) -> dict[str, Any]:
    drafted = trace[f"{side}_drafted"]
    accepted = trace[f"{side}_accepted"]
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in (drafted, accepted)):
        raise TraceError(f"{side} drafted/accepted counters must be non-negative integers")
    if accepted > drafted:
        raise TraceError(f"{side} accepted counter exceeds drafted counter")
    prefix_counts = counts(trace[f"{side}_accepted_prefix_counts"], f"{side} counts")
    if sum(prefix_counts) != accepted:
        raise TraceError(f"{side} accepted-prefix sum does not equal accepted counter")
    acceptance = finite_number(trace[f"{side}_acceptance"], f"{side} acceptance")
    expected = accepted / drafted if drafted else 0.0
    if not math.isclose(acceptance, expected, rel_tol=0.0, abs_tol=1e-15):
        raise TraceError(f"{side} acceptance is not canonical accepted/drafted")
    for field in ("draft", "accepted"):
        digest = trace[f"{field}_{side}_sha256"]
        if not isinstance(digest, str) or len(digest) != 64:
            raise TraceError(f"{field}_{side}_sha256 must be a lowercase SHA-256")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise TraceError(f"{field}_{side}_sha256 is not hexadecimal") from exc
        if digest != digest.lower():
            raise TraceError(f"{field}_{side}_sha256 must be lowercase")
    return {
        "drafted": drafted,
        "accepted": accepted,
        "acceptance": expected,
        "accepted_prefix_counts": prefix_counts,
        "draft_sha256": trace[f"draft_{side}_sha256"],
        "accepted_sha256": trace[f"accepted_{side}_sha256"],
    }


def compare(summary: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    if summary.get("schema") != SUMMARY_SCHEMA:
        raise TraceError("unexpected diagnostic summary schema")
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise TraceError("unexpected GX container receipt schema")
    if summary.get("image_id") != receipt.get("image_id"):
        raise TraceError("summary and container receipt image identities differ")
    if summary.get("failure_gate") != "dflash-assistant-trace-parity":
        raise TraceError("summary did not fail at the assistant trace parity gate")
    gate = summary.get("gate_order_evidence")
    if not isinstance(gate, dict) or not all(
        gate.get(field) is True
        for field in ("exact_target_tokens", "exact_target_full_logits", "exact_final_dflash_tokens")
    ):
        raise TraceError("prerequisite exactness gates are absent or false")
    trace = summary.get("trace")
    if not isinstance(trace, dict) or set(trace) != TRACE_FIELDS:
        raise TraceError("trace object has missing or unknown fields")
    local = validate_side(trace, "local")
    remote = validate_side(trace, "remote")
    count_index = first_mismatch(
        local["accepted_prefix_counts"], remote["accepted_prefix_counts"]
    )
    draft_equal = local["draft_sha256"] == remote["draft_sha256"]
    accepted_equal = local["accepted_sha256"] == remote["accepted_sha256"]
    counts_equal = count_index is None
    semantic = not (draft_equal and accepted_equal and counts_equal)
    if not counts_equal:
        classification = "semantic-accepted-prefix-counts"
    elif not draft_equal:
        classification = "semantic-proposal-tokens"
    elif not accepted_equal:
        classification = "semantic-accepted-prefix-tokens"
    else:
        classification = "equal-after-canonicalization"
    return {
        "schema": RESULT_SCHEMA,
        "status": "passed",
        "kind": "gpu-free-offline-diagnostic",
        "identity": summary.get("identity"),
        "image_id": summary.get("image_id"),
        "canonicalization": {
            "token_framing": "u32 little-endian concatenation then lowercase SHA-256",
            "accepted_prefix_counts": "ordered non-negative integer array",
            "acceptance": "accepted/drafted; input float must agree within 1e-15",
            "json": "object order and float spelling ignored; duplicate and unknown trace keys rejected",
        },
        "comparison": {
            "classification": classification,
            "semantic_divergence": semantic,
            "representation_only": not semantic,
            "draft_digest_equal": draft_equal,
            "accepted_digest_equal": accepted_equal,
            "accepted_prefix_counts_equal": counts_equal,
            "first_count_divergence_round_zero_based": count_index,
            "local": local,
            "remote": remote,
        },
        "scope": {
            "target_path_bit_exact": True,
            "assistant_path_bit_exact": not semantic,
            "conclusion": (
                "target full-logit parity does not prove assistant proposal/cache parity; "
                "the ordered accepted-prefix counts differ semantically"
            ),
            "unresolved": (
                "the retained run has digests but no raw proposal-token arrays, so the first "
                "proposal token and pre-proposal assistant context cannot be reconstructed offline"
            ),
        },
        "accelerator_touched": False,
        "gx_touched": False,
        "qualification_eligible": False,
        "readiness_eligible": False,
        "seal_eligible": False,
    }


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path = path.resolve()
    if path.exists():
        raise TraceError(f"refusing to replace output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--container-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = load_json(args.summary)
        receipt = load_json(args.container_receipt)
        result = compare(summary, receipt)
        result["inputs"] = {
            "summary": str(args.summary.resolve()),
            "summary_sha256": sha256(args.summary),
            "container_receipt": str(args.container_receipt.resolve()),
            "container_receipt_sha256": sha256(args.container_receipt),
        }
        atomic_write(args.output, result)
    except TraceError as exc:
        print(f"gx-trace-diff: {exc}", file=os.sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
