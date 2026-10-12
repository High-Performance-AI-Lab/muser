#!/usr/bin/env python3
"""Strict reader for the patched llama-perplexity teacher-evidence siblings.

The reviewed comparator writes two siblings whenever ``--save-all-logits PATH``
is active.  This module validates their wire format, cross-binds every exact
pre-quantization top-ten row to PATH's existing uint16 row, and validates the
effective runtime contract without executing a llama binary.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any, BinaryIO


TOP10_SUFFIX = ".muser-top10.jsonl"
RUNTIME_SUFFIX = ".muser-runtime.json"
TOP10_SCHEMA = "muser.llama_perplexity.exact_top10.v3"
RUNTIME_SCHEMA = "muser.llama_perplexity.runtime.v1"
LLAMA_MAGIC = b"_logits_"
MAX_TOP10_BYTES = 64 << 20
MAX_RUNTIME_BYTES = 1 << 20

TOP10_HEADER_KEYS = {
    "schema",
    "evidence_id",
    "upstream_commit",
    "patch_sha256",
    "context_length",
    "vocab_size",
    "chunks",
    "scored_rows",
    "logit_stage",
    "target_nll_stage",
    "float_encoding",
    "rank_order",
    "quantized_cross_binding",
}
TOP10_ROW_KEYS = {
    "chunk",
    "position",
    "input_token_id",
    "target_token_id",
    "target_nll",
    "row_scale",
    "minimum_log_probability",
    "candidates",
}
CANDIDATE_KEYS = {"token_id", "logit", "quantized_u16"}
RUNTIME_KEYS = {
    "schema",
    "evidence_id",
    "upstream_commit",
    "patch_sha256",
    "artifacts",
    "requested",
    "effective",
    "result",
}
REQUESTED_KEYS = {
    "context_length",
    "batch_size",
    "ubatch_size",
    "chunks",
    "threads",
    "threads_batch",
    "gpu_layer_limit",
    "flash_attention_type",
    "cache_type_k",
    "cache_type_v",
}
EFFECTIVE_KEYS = {
    "context_length",
    "context_length_per_sequence",
    "batch_size",
    "ubatch_size",
    "sequence_capacity",
    "threads",
    "threads_batch",
    "flash_attention",
    "flash_attention_auto",
    "offload_kqv",
    "kv_cache_kind",
    "cache_type_k",
    "cache_type_v",
    "model_transformer_layers",
    "gpu_layer_limit",
    "gpu_transformer_layers",
    "cpu_transformer_layers",
    "other_transformer_layers",
    "all_transformer_layers_gpu",
    "output_layer_device_type",
    "output_layer_gpu",
    "full_model_gpu_offload",
}
RESULT_KEYS = {"completed", "chunks", "scored_rows", "perplexity"}


class LlamaPerplexityEvidenceError(RuntimeError):
    """The llama teacher artifact set is incomplete or internally inconsistent."""


def top10_path_for(logits_path: Path) -> Path:
    return Path(str(logits_path) + TOP10_SUFFIX)


def runtime_path_for(logits_path: Path) -> Path:
    return Path(str(logits_path) + RUNTIME_SUFFIX)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _binary32(value: float) -> float:
    """Round one arithmetic result to the producer's binary32 precision."""

    return struct.unpack("<f", struct.pack("<f", value))[0]


def _reject_constant(value: str) -> None:
    raise LlamaPerplexityEvidenceError(f"non-finite JSON number {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LlamaPerplexityEvidenceError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _parse_json(raw: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise LlamaPerplexityEvidenceError(f"invalid JSON in {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise LlamaPerplexityEvidenceError(f"{label} must be a JSON object")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise LlamaPerplexityEvidenceError(
            f"{label} keys differ: expected {sorted(expected)}, got {sorted(value)}"
        )


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LlamaPerplexityEvidenceError(
            f"{label} must be an integer >= {minimum}"
        )
    return value


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LlamaPerplexityEvidenceError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        condition = "finite and positive" if positive else "finite"
        raise LlamaPerplexityEvidenceError(f"{label} must be {condition}")
    return result


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise LlamaPerplexityEvidenceError(f"{label} must be boolean")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise LlamaPerplexityEvidenceError(f"{label} must be a non-empty string")
    return value


def _read_exact(handle: BinaryIO, size: int, label: str) -> bytes:
    value = handle.read(size)
    if len(value) != size:
        raise LlamaPerplexityEvidenceError(
            f"truncated quantized logits while reading {label}: "
            f"expected {size} bytes, got {len(value)}"
        )
    return value


def _load_runtime(
    path: Path,
    *,
    expected_upstream_commit: str,
    expected_patch_sha256: str,
    context_length: int,
    chunks: int,
    scored_rows: int,
    batch_size: int,
    ubatch_size: int,
    threads: int,
    kv_cache: str,
    model_transformer_layers: int,
    evidence_id: str,
    runtime_route: str,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise LlamaPerplexityEvidenceError(f"runtime evidence is not a regular file: {path}")
    if path.stat().st_size > MAX_RUNTIME_BYTES:
        raise LlamaPerplexityEvidenceError(f"runtime evidence is too large: {path}")
    try:
        runtime = _parse_json(path.read_text(encoding="utf-8"), str(path))
    except (OSError, UnicodeError) as exc:
        raise LlamaPerplexityEvidenceError(f"cannot read runtime evidence {path}: {exc}") from exc
    _exact_keys(runtime, RUNTIME_KEYS, "runtime")
    if runtime["schema"] != RUNTIME_SCHEMA:
        raise LlamaPerplexityEvidenceError("unexpected runtime evidence schema")
    if runtime["evidence_id"] != evidence_id:
        raise LlamaPerplexityEvidenceError("runtime/top-10 evidence ID mismatch")
    if runtime["upstream_commit"] != expected_upstream_commit:
        raise LlamaPerplexityEvidenceError("runtime upstream commit mismatch")
    if runtime["patch_sha256"] != expected_patch_sha256:
        raise LlamaPerplexityEvidenceError("runtime patch SHA-256 mismatch")

    artifacts = runtime["artifacts"]
    if artifacts != {
        "quantized_logits": "caller_path",
        "exact_top10_suffix": TOP10_SUFFIX,
    }:
        raise LlamaPerplexityEvidenceError("runtime artifact naming contract mismatch")

    requested = runtime["requested"]
    if not isinstance(requested, dict):
        raise LlamaPerplexityEvidenceError("runtime requested contract must be an object")
    _exact_keys(requested, REQUESTED_KEYS, "runtime requested")
    for key, minimum in (
        ("context_length", 1),
        ("batch_size", 1),
        ("ubatch_size", 1),
        ("chunks", 1),
        ("threads", 1),
        ("gpu_layer_limit", 0),
    ):
        _integer(requested[key], f"requested runtime {key}", minimum=minimum)
    for key in ("flash_attention_type", "cache_type_k", "cache_type_v"):
        _string(requested[key], f"requested runtime {key}")
    if runtime_route not in {"full-gpu", "cpu-only"}:
        raise LlamaPerplexityEvidenceError(f"unsupported runtime route: {runtime_route}")
    gpu_layer_limit = 99 if runtime_route == "full-gpu" else 0
    expected_requested = {
        "context_length": context_length,
        "batch_size": batch_size,
        "ubatch_size": ubatch_size,
        "chunks": chunks,
        "threads": threads,
        "gpu_layer_limit": gpu_layer_limit,
        "flash_attention_type": "enabled",
        "cache_type_k": kv_cache,
        "cache_type_v": kv_cache,
    }
    for key, expected in expected_requested.items():
        if requested[key] != expected:
            raise LlamaPerplexityEvidenceError(
                f"requested runtime {key} mismatch: expected {expected!r}, got {requested[key]!r}"
            )
    requested_threads_batch = requested["threads_batch"]
    if (
        isinstance(requested_threads_batch, bool)
        or not isinstance(requested_threads_batch, int)
        or requested_threads_batch not in (-1, threads)
    ):
        raise LlamaPerplexityEvidenceError(
            "requested runtime threads_batch must inherit from or equal --threads"
        )

    effective = runtime["effective"]
    if not isinstance(effective, dict):
        raise LlamaPerplexityEvidenceError("runtime effective contract must be an object")
    _exact_keys(effective, EFFECTIVE_KEYS, "runtime effective")
    for key, minimum in (
        ("context_length", 1),
        ("context_length_per_sequence", 1),
        ("batch_size", 1),
        ("ubatch_size", 1),
        ("sequence_capacity", 1),
        ("threads", 1),
        ("threads_batch", 1),
        ("model_transformer_layers", 1),
        ("gpu_layer_limit", 0),
        ("gpu_transformer_layers", 0),
        ("cpu_transformer_layers", 0),
        ("other_transformer_layers", 0),
    ):
        _integer(effective[key], f"effective runtime {key}", minimum=minimum)
    for key in (
        "flash_attention",
        "flash_attention_auto",
        "offload_kqv",
        "all_transformer_layers_gpu",
        "output_layer_gpu",
        "full_model_gpu_offload",
    ):
        _boolean(effective[key], f"effective runtime {key}")
    for key in (
        "kv_cache_kind",
        "cache_type_k",
        "cache_type_v",
        "output_layer_device_type",
    ):
        _string(effective[key], f"effective runtime {key}")
    effective_context = ((context_length + 255) // 256) * 256
    expected_effective = {
        "context_length": effective_context,
        "context_length_per_sequence": effective_context,
        "batch_size": batch_size,
        "ubatch_size": ubatch_size,
        "sequence_capacity": 1,
        "threads": threads,
        "threads_batch": threads,
        "flash_attention": True,
        "flash_attention_auto": False,
        "offload_kqv": True,
        # Muse uses llama.cpp's hybrid SWA/NoPE memory implementation.  The
        # requested f16 types are authenticated above; this backend currently
        # reports no single classic-KV dtype because its planes are mixed.
        "kv_cache_kind": "not-classic-kv-cache",
        "cache_type_k": "unknown",
        "cache_type_v": "unknown",
        "all_transformer_layers_gpu": runtime_route == "full-gpu",
        "output_layer_gpu": runtime_route == "full-gpu",
        "full_model_gpu_offload": runtime_route == "full-gpu",
    }
    for key, expected in expected_effective.items():
        if effective[key] != expected:
            raise LlamaPerplexityEvidenceError(
                f"effective runtime {key} mismatch: expected {expected!r}, got {effective[key]!r}"
            )
    transformer_layers = _integer(
        effective["model_transformer_layers"],
        "effective model_transformer_layers",
        minimum=1,
    )
    if transformer_layers != model_transformer_layers:
        raise LlamaPerplexityEvidenceError(
            "effective model_transformer_layers does not match the sealed model contract"
        )
    gpu_layers = _integer(
        effective["gpu_transformer_layers"],
        "effective gpu_transformer_layers",
    )
    cpu_layers = _integer(
        effective["cpu_transformer_layers"],
        "effective cpu_transformer_layers",
    )
    other_layers = _integer(
        effective["other_transformer_layers"],
        "effective other_transformer_layers",
    )
    if gpu_layers + cpu_layers + other_layers != transformer_layers:
        raise LlamaPerplexityEvidenceError("effective transformer-layer counts do not sum")
    effective_gpu_limit = _integer(
        effective["gpu_layer_limit"], "effective gpu layer limit"
    )
    if runtime_route == "full-gpu":
        if gpu_layers != transformer_layers or cpu_layers or other_layers:
            raise LlamaPerplexityEvidenceError(
                "effective runtime did not place every transformer layer on GPU"
            )
        if effective_gpu_limit < transformer_layers:
            raise LlamaPerplexityEvidenceError(
                "effective GPU layer limit is below model layer count"
            )
        if effective["output_layer_device_type"] not in {"gpu", "igpu"}:
            raise LlamaPerplexityEvidenceError(
                "effective output layer is not on a GPU device"
            )
    else:
        if gpu_layers or cpu_layers != transformer_layers or other_layers:
            raise LlamaPerplexityEvidenceError(
                "effective CPU reference did not place every transformer layer on CPU"
            )
        if effective_gpu_limit != 0:
            raise LlamaPerplexityEvidenceError(
                "effective CPU reference retained a GPU layer limit"
            )
        if effective["output_layer_device_type"] != "cpu":
            raise LlamaPerplexityEvidenceError(
                "effective CPU reference output layer is not on CPU"
            )

    result = runtime["result"]
    if not isinstance(result, dict):
        raise LlamaPerplexityEvidenceError("runtime result must be an object")
    _exact_keys(result, RESULT_KEYS, "runtime result")
    if _boolean(result["completed"], "runtime result completed") is not True:
        raise LlamaPerplexityEvidenceError("runtime result is incomplete")
    result_chunks = _integer(result["chunks"], "runtime result chunks", minimum=1)
    result_rows = _integer(
        result["scored_rows"], "runtime result scored_rows", minimum=1
    )
    if result_chunks != chunks or result_rows != scored_rows:
        raise LlamaPerplexityEvidenceError("runtime result geometry mismatch")
    _finite(result["perplexity"], "runtime result perplexity", positive=True)
    return runtime


def validate_teacher_evidence(
    logits_path: Path,
    *,
    expected_upstream_commit: str,
    expected_patch_sha256: str,
    expected_context_length: int,
    expected_chunks: int,
    expected_batch_size: int,
    expected_ubatch_size: int,
    expected_threads: int,
    expected_kv_cache: str,
    expected_model_transformer_layers: int,
    runtime_route: str = "full-gpu",
) -> dict[str, Any]:
    """Validate all three siblings and return exact top-ten rows plus identities."""

    if logits_path.is_symlink() or not logits_path.is_file():
        raise LlamaPerplexityEvidenceError(
            f"quantized logits are not a regular file: {logits_path}"
        )
    top10_path = top10_path_for(logits_path)
    runtime_path = runtime_path_for(logits_path)
    if top10_path.is_symlink() or not top10_path.is_file():
        raise LlamaPerplexityEvidenceError(f"top-10 evidence is not a regular file: {top10_path}")
    if top10_path.stat().st_size > MAX_TOP10_BYTES:
        raise LlamaPerplexityEvidenceError(f"top-10 evidence is too large: {top10_path}")
    if len(expected_upstream_commit) != 40 or any(
        char not in "0123456789abcdef" for char in expected_upstream_commit
    ):
        raise LlamaPerplexityEvidenceError("expected upstream commit must be lowercase hex")
    if len(expected_patch_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in expected_patch_sha256
    ):
        raise LlamaPerplexityEvidenceError("expected patch SHA-256 must be lowercase hex")
    if expected_kv_cache not in {"f16", "q8_0"}:
        raise LlamaPerplexityEvidenceError("expected KV cache must be f16 or q8_0")
    for value, label, minimum in (
        (expected_context_length, "expected context length", 2),
        (expected_chunks, "expected chunks", 1),
        (expected_batch_size, "expected batch size", 1),
        (expected_ubatch_size, "expected ubatch size", 1),
        (expected_threads, "expected threads", 1),
    ):
        _integer(value, label, minimum=minimum)
    if (
        isinstance(expected_model_transformer_layers, bool)
        or not isinstance(expected_model_transformer_layers, int)
        or expected_model_transformer_layers < 1
    ):
        raise LlamaPerplexityEvidenceError(
            "expected model transformer layer count must be a positive integer"
        )

    try:
        top10_handle = top10_path.open("r", encoding="utf-8")
        logits_handle = logits_path.open("rb")
    except (OSError, UnicodeError) as exc:
        raise LlamaPerplexityEvidenceError(f"cannot open llama teacher evidence: {exc}") from exc

    rows: list[dict[str, Any]] = []
    with top10_handle, logits_handle:
        header_line = top10_handle.readline()
        if not header_line:
            raise LlamaPerplexityEvidenceError("top-10 evidence is empty")
        header = _parse_json(header_line, f"{top10_path}:1")
        _exact_keys(header, TOP10_HEADER_KEYS, "top-10 header")
        if header["schema"] != TOP10_SCHEMA:
            raise LlamaPerplexityEvidenceError("unexpected top-10 schema")
        evidence_id = header["evidence_id"]
        if not isinstance(evidence_id, str) or len(evidence_id) != 64 or any(
            char not in "0123456789abcdef" for char in evidence_id
        ):
            raise LlamaPerplexityEvidenceError(
                "top-10 evidence ID must be 64 lowercase hex characters"
            )
        if header["upstream_commit"] != expected_upstream_commit:
            raise LlamaPerplexityEvidenceError("top-10 upstream commit mismatch")
        if header["patch_sha256"] != expected_patch_sha256:
            raise LlamaPerplexityEvidenceError("top-10 patch SHA-256 mismatch")
        for key, minimum in (
            ("context_length", 2),
            ("vocab_size", 2),
            ("chunks", 1),
            ("scored_rows", 1),
        ):
            _integer(header[key], f"top-10 header {key}", minimum=minimum)
        expected_header_literals = {
            "logit_stage": "llama_get_logits_before_log_softmax_and_u16_quantization",
            "target_nll_stage": "raw_float32_logits_before_floor_and_u16_quantization",
            "float_encoding": "decimal_max_digits10_roundtrip_binary32",
            "rank_order": "logit_desc_token_id_asc",
            "quantized_cross_binding": "row_scale_minimum_log_probability_and_candidate_u16",
        }
        for key, expected in expected_header_literals.items():
            if header[key] != expected:
                raise LlamaPerplexityEvidenceError(f"unexpected top-10 header {key}")

        if _read_exact(logits_handle, 8, "magic") != LLAMA_MAGIC:
            raise LlamaPerplexityEvidenceError("invalid quantized logits magic")
        context_length, vocab_size, chunks = struct.unpack(
            "<Iii", _read_exact(logits_handle, 12, "header")
        )
        if (
            context_length != expected_context_length
            or chunks != expected_chunks
            or vocab_size < 2
        ):
            raise LlamaPerplexityEvidenceError(
                "quantized logits geometry does not match the expected contract"
            )
        scored_per_chunk = context_length - 1 - context_length // 2
        scored_rows = chunks * scored_per_chunk
        if {
            "context_length": header["context_length"],
            "vocab_size": header["vocab_size"],
            "chunks": header["chunks"],
            "scored_rows": header["scored_rows"],
        } != {
            "context_length": context_length,
            "vocab_size": vocab_size,
            "chunks": chunks,
            "scored_rows": scored_rows,
        }:
            raise LlamaPerplexityEvidenceError("top-10 header geometry mismatch")
        token_count = context_length * chunks
        tokens = struct.unpack(
            f"<{token_count}i",
            _read_exact(logits_handle, token_count * 4, "token IDs"),
        )
        padded_vocab_u16 = 2 * ((vocab_size + 1) // 2)

        line_number = 1
        for chunk in range(chunks):
            for position in range(context_length // 2, context_length - 1):
                line_number += 1
                raw = top10_handle.readline()
                if not raw:
                    raise LlamaPerplexityEvidenceError(
                        f"top-10 evidence ended before row {len(rows)}"
                    )
                row = _parse_json(raw, f"{top10_path}:{line_number}")
                _exact_keys(row, TOP10_ROW_KEYS, f"top-10 row {len(rows)}")
                offset = chunk * context_length + position
                expected_row = {
                    "chunk": chunk,
                    "position": position,
                    "input_token_id": tokens[offset],
                    "target_token_id": tokens[offset + 1],
                }
                for key, expected in expected_row.items():
                    actual = _integer(row[key], f"top-10 row {len(rows)} {key}")
                    if actual != expected:
                        raise LlamaPerplexityEvidenceError(
                            f"top-10 row {len(rows)} {key} mismatch"
                        )
                target_nll = _finite(
                    row["target_nll"], f"top-10 row {len(rows)} target_nll"
                )
                if target_nll < 0.0:
                    raise LlamaPerplexityEvidenceError(
                        f"top-10 row {len(rows)} target_nll must be nonnegative"
                    )

                scale_bytes = _read_exact(logits_handle, 4, "row scale")
                minimum_bytes = _read_exact(
                    logits_handle, 4, "row minimum log probability"
                )
                scale = struct.unpack("<f", scale_bytes)[0]
                minimum = struct.unpack("<f", minimum_bytes)[0]
                row_scale = _finite(row["row_scale"], f"top-10 row {len(rows)} scale")
                if row_scale < 0.0:
                    raise LlamaPerplexityEvidenceError(
                        f"top-10 row {len(rows)} scale must be nonnegative"
                    )
                row_minimum = _finite(
                    row["minimum_log_probability"],
                    f"top-10 row {len(rows)} minimum log probability",
                )
                if struct.pack("<f", row_scale) != scale_bytes:
                    raise LlamaPerplexityEvidenceError(
                        f"top-10 row {len(rows)} scale is not bit-exact"
                    )
                if struct.pack("<f", row_minimum) != minimum_bytes:
                    raise LlamaPerplexityEvidenceError(
                        f"top-10 row {len(rows)} minimum is not bit-exact"
                    )
                quantized = _read_exact(
                    logits_handle, padded_vocab_u16 * 2, "row probabilities"
                )
                target_u16 = struct.unpack_from(
                    "<H", quantized, expected_row["target_token_id"] * 2
                )[0]
                if target_u16 > 0:
                    # The llama producer derives this row from float operands.
                    # Keep the reconstruction in binary32 too: evaluating it as
                    # Python binary64 can move a valid half-bin boundary just
                    # beyond the unchanged quantization tolerance.
                    quantized_target_nll = -_binary32(
                        _binary32(scale * target_u16) + minimum
                    )
                    # nearest_int bounds quantization to half a bin, but the
                    # exact NLL and the saved minimum reach that bin through
                    # different binary32 expression trees. Allow one percent
                    # of a bin for those final roundings; this is still far
                    # below one encoded probability step and cannot bind an
                    # adjacent quantized value.
                    target_tolerance = max(scale * 0.51, 1e-6)
                    if abs(target_nll - quantized_target_nll) > target_tolerance:
                        raise LlamaPerplexityEvidenceError(
                            f"top-10 row {len(rows)} exact target_nll does not match quantized cross-binding"
                        )
                candidates = row["candidates"]
                if not isinstance(candidates, list) or len(candidates) != 10:
                    raise LlamaPerplexityEvidenceError(
                        f"top-10 row {len(rows)} must contain exactly ten candidates"
                    )
                parsed: list[tuple[int, float, int]] = []
                for rank, candidate in enumerate(candidates):
                    if not isinstance(candidate, dict):
                        raise LlamaPerplexityEvidenceError(
                            f"top-10 row {len(rows)} candidate {rank} is not an object"
                        )
                    _exact_keys(
                        candidate,
                        CANDIDATE_KEYS,
                        f"top-10 row {len(rows)} candidate {rank}",
                    )
                    token_id = _integer(
                        candidate["token_id"],
                        f"top-10 row {len(rows)} candidate {rank} token_id",
                    )
                    if token_id >= vocab_size:
                        raise LlamaPerplexityEvidenceError(
                            f"top-10 row {len(rows)} candidate {rank} token is outside vocabulary"
                        )
                    logit = _finite(
                        candidate["logit"],
                        f"top-10 row {len(rows)} candidate {rank} logit",
                    )
                    quantized_u16 = _integer(
                        candidate["quantized_u16"],
                        f"top-10 row {len(rows)} candidate {rank} quantized_u16",
                    )
                    if quantized_u16 > 65535:
                        raise LlamaPerplexityEvidenceError(
                            f"top-10 row {len(rows)} candidate {rank} uint16 is out of range"
                        )
                    actual_u16 = struct.unpack_from("<H", quantized, token_id * 2)[0]
                    if quantized_u16 != actual_u16:
                        raise LlamaPerplexityEvidenceError(
                            f"top-10 row {len(rows)} candidate {rank} is not cross-bound to quantized row"
                        )
                    parsed.append((token_id, logit, quantized_u16))
                if len({candidate[0] for candidate in parsed}) != 10:
                    raise LlamaPerplexityEvidenceError(
                        f"top-10 row {len(rows)} candidate IDs are not distinct"
                    )
                for left, right in zip(parsed, parsed[1:]):
                    if left[1] < right[1] or (
                        left[1] == right[1] and left[0] > right[0]
                    ):
                        raise LlamaPerplexityEvidenceError(
                            f"top-10 row {len(rows)} violates exact rank order"
                        )
                rows.append(row)

        for extra in top10_handle:
            if extra.strip():
                raise LlamaPerplexityEvidenceError("unexpected extra top-10 row")
        if logits_handle.read(1):
            raise LlamaPerplexityEvidenceError("unexpected trailing quantized logits bytes")

    runtime = _load_runtime(
        runtime_path,
        expected_upstream_commit=expected_upstream_commit,
        expected_patch_sha256=expected_patch_sha256,
        context_length=expected_context_length,
        chunks=expected_chunks,
        scored_rows=len(rows),
        batch_size=expected_batch_size,
        ubatch_size=expected_ubatch_size,
        threads=expected_threads,
        kv_cache=expected_kv_cache,
        model_transformer_layers=expected_model_transformer_layers,
        evidence_id=evidence_id,
        runtime_route=runtime_route,
    )
    exact_target_nll_sum = math.fsum(float(row["target_nll"]) for row in rows)
    exact_perplexity = math.exp(exact_target_nll_sum / len(rows))
    runtime_perplexity = float(runtime["result"]["perplexity"])
    if not math.isclose(
        exact_perplexity, runtime_perplexity, rel_tol=5e-6, abs_tol=1e-7
    ):
        raise LlamaPerplexityEvidenceError(
            "exact target-NLL rows do not reproduce runtime perplexity"
        )
    return {
        "header": header,
        "rows": rows,
        "runtime": runtime,
        "metrics": {
            "exact_target_nll_sum": exact_target_nll_sum,
            "exact_perplexity": exact_perplexity,
        },
        "artifacts": {
            "quantized_logits": {
                "sha256": _sha256(logits_path),
                "size_bytes": logits_path.stat().st_size,
            },
            "exact_top10": {
                "sha256": _sha256(top10_path),
                "size_bytes": top10_path.stat().st_size,
            },
            "runtime": {
                "sha256": _sha256(runtime_path),
                "size_bytes": runtime_path.stat().st_size,
            },
        },
    }
