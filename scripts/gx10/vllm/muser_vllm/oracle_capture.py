"""Read-only full target oracle capture across block or sequential forwards."""

from __future__ import annotations

import hashlib
import os
import re
import threading
from pathlib import Path
from typing import Any


SCHEMA = "muser.spark-composite-target-oracle.v1"
TARGET_LAYERS = (1, 13, 25, 37, 49)
HIDDEN_SIZE = 6656
VOCAB_SIZE = 202_048
_LOCK = threading.Lock()
_ACTIVE: dict[str, Any] | None = None
_ORIGINAL_DECODER_INIT: Any = None
_ORIGINAL_DECODER_FORWARD: Any = None
_ORIGINAL_COMPUTE_LOGITS: Any = None


def _write_exclusive(path: Path, payload: memoryview) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def begin_capture(request_id: str, expected_rows: int, output_prefix: Path) -> None:
    global _ACTIVE
    if expected_rows < 1:
        raise RuntimeError("target oracle capture requires positive rows")
    hidden_path = output_prefix.with_suffix(".hidden.f32")
    logits_path = output_prefix.with_suffix(".logits.f32")
    extra_logits_path = output_prefix.with_suffix(".extra-logits.f32")
    for path in (hidden_path, logits_path, extra_logits_path):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"refusing to replace target oracle capture: {path}")
    with _LOCK:
        if _ACTIVE is not None:
            raise RuntimeError("a target oracle capture is already active")
        _ACTIVE = {
            "expected_rows": expected_rows,
            "hidden": {layer: [] for layer in TARGET_LAYERS},
            "logits": [],
            "output_prefix": output_prefix,
            "request_id": request_id,
        }


def _capture_hidden(layer: int, hidden_states: Any) -> None:
    if layer not in TARGET_LAYERS:
        return
    with _LOCK:
        active = _ACTIVE
        if active is None:
            return
        if hidden_states.ndim != 2 or hidden_states.shape[1] != HIDDEN_SIZE:
            raise RuntimeError(f"target oracle hidden shape differs: {tuple(hidden_states.shape)}")
        active["hidden"][layer].append(hidden_states.detach().float().cpu())


def _capture_logits(logits: Any) -> None:
    with _LOCK:
        active = _ACTIVE
        if active is None or logits is None:
            return
        if logits.ndim != 2 or logits.shape[1] != VOCAB_SIZE:
            raise RuntimeError(f"target oracle logit shape differs: {tuple(logits.shape)}")
        active["logits"].append(logits.detach().float().cpu())


def decoder_layer_init(module: Any, *args: Any, **kwargs: Any) -> None:
    if _ORIGINAL_DECODER_INIT is None:
        raise RuntimeError("target oracle init invoked before installation")
    prefix = kwargs.get("prefix")
    if prefix is None and len(args) >= 4:
        prefix = args[3]
    if not isinstance(prefix, str):
        raise RuntimeError("target oracle decoder layer has no stable prefix")
    match = re.search(r"(?:^|\.)layers\.(\d+)$", prefix)
    if match is None:
        raise RuntimeError(f"cannot derive target oracle layer from {prefix!r}")
    _ORIGINAL_DECODER_INIT(module, *args, **kwargs)
    module._muser_oracle_layer_index = int(match.group(1))


def decoder_layer_forward(
    module: Any,
    positions: Any,
    hidden_states: Any,
    residual: Any,
) -> tuple[Any, Any]:
    if _ORIGINAL_DECODER_FORWARD is None:
        raise RuntimeError("target oracle forward invoked before installation")
    result = _ORIGINAL_DECODER_FORWARD(module, positions, hidden_states, residual)
    _capture_hidden(module._muser_oracle_layer_index, result[0])
    return result


def compute_logits(module: Any, hidden_states: Any) -> Any:
    if _ORIGINAL_COMPUTE_LOGITS is None:
        raise RuntimeError("target oracle logits invoked before installation")
    logits = _ORIGINAL_COMPUTE_LOGITS(module, hidden_states)
    _capture_logits(logits)
    return logits


def install_oracle_capture() -> dict[str, Any]:
    global _ORIGINAL_DECODER_INIT, _ORIGINAL_DECODER_FORWARD, _ORIGINAL_COMPUTE_LOGITS
    from vllm.model_executor.models import muse_glimmer

    if _ORIGINAL_DECODER_INIT is not None:
        raise RuntimeError("target oracle capture was installed twice")
    _ORIGINAL_DECODER_INIT = muse_glimmer.MuseGlimmerDecoderLayer.__init__
    _ORIGINAL_DECODER_FORWARD = muse_glimmer.MuseGlimmerDecoderLayer.forward
    _ORIGINAL_COMPUTE_LOGITS = muse_glimmer.MuseGlimmerForCausalLM.compute_logits
    muse_glimmer.MuseGlimmerDecoderLayer.__init__ = decoder_layer_init
    muse_glimmer.MuseGlimmerDecoderLayer.forward = decoder_layer_forward
    muse_glimmer.MuseGlimmerForCausalLM.compute_logits = compute_logits
    return {
        "active": True,
        "numeric_effect": "none-return-original-hidden-and-logits",
        "target_layers": list(TARGET_LAYERS),
    }


def finish_capture() -> dict[str, Any]:
    global _ACTIVE
    import numpy as np
    import torch

    with _LOCK:
        active = _ACTIVE
        if active is None:
            raise RuntimeError("target oracle capture is not active")
        try:
            expected = active["expected_rows"]
            per_layer = []
            for layer in TARGET_LAYERS:
                chunks = active["hidden"][layer]
                if not chunks:
                    raise RuntimeError(f"target oracle missed hidden layer {layer}")
                matrix = torch.cat(chunks, dim=0)
                if matrix.shape != (expected, HIDDEN_SIZE):
                    raise RuntimeError(
                        f"target oracle layer {layer} has {tuple(matrix.shape)}, "
                        f"expected {(expected, HIDDEN_SIZE)}"
                    )
                per_layer.append(matrix)
            logit_call_rows = [int(chunk.shape[0]) for chunk in active["logits"]]
            logits = torch.cat(active["logits"], dim=0) if active["logits"] else None
            duplicate_logit_row = None
            extra_logits = None
            logit_selection = "all_rows"
            # With prompt_logprobs enabled, pinned vLLM invokes compute_logits
            # once for the fresh prompt rows and once for sampling. At the
            # prompt/decode boundary this repeats the same final hidden row.
            # Remove exactly one bit-identical adjacent duplicate; any other
            # shape remains ambiguous and fails closed.
            if logits is not None and logits.shape == (expected + 1, VOCAB_SIZE):
                duplicates = [
                    row
                    for row in range(expected)
                    if torch.equal(logits[row], logits[row + 1])
                ]
                if len(duplicates) == 1:
                    duplicate_logit_row = duplicates[0] + 1
                    logits = torch.cat(
                        (logits[:duplicate_logit_row], logits[duplicate_logit_row + 1 :]),
                        dim=0,
                    )
                    logit_selection = "removed_one_adjacent_bit_exact_duplicate"
                elif logit_call_rows == [1, expected]:
                    # Pinned vLLM computes the generated-token sampling row as
                    # a singleton, then materializes all fresh prompt-logprob
                    # rows as one block. Preserve both: the block is the
                    # verification matrix and the singleton is the actual
                    # sampling witness. They may differ because LM-head shape
                    # changes floating-point reduction order.
                    extra_logits = logits[:1]
                    logits = logits[1:]
                    logit_selection = "block_rows_after_leading_sampling_witness"
                elif logit_call_rows == [expected, 1]:
                    extra_logits = logits[-1:]
                    logits = logits[:-1]
                    logit_selection = "block_rows_before_trailing_sampling_witness"
            if logits is None or logits.shape != (expected, VOCAB_SIZE):
                actual = None if logits is None else tuple(logits.shape)
                raise RuntimeError(
                    f"target oracle logits have {actual}, expected {(expected, VOCAB_SIZE)}; "
                    f"compute_logits call rows were {logit_call_rows}"
                )
            hidden = torch.stack(per_layer, dim=1).contiguous()
            hidden_array = hidden.numpy().astype(np.dtype("<f4"), copy=False)
            logits_array = logits.contiguous().numpy().astype(np.dtype("<f4"), copy=False)
            hidden_payload = memoryview(hidden_array).cast("B")
            logits_payload = memoryview(logits_array).cast("B")
            prefix: Path = active["output_prefix"]
            prefix.parent.mkdir(parents=True, exist_ok=True)
            hidden_path = prefix.with_suffix(".hidden.f32")
            logits_path = prefix.with_suffix(".logits.f32")
            _write_exclusive(hidden_path, hidden_payload)
            _write_exclusive(logits_path, logits_payload)
            extra_receipt = None
            if extra_logits is not None:
                extra_array = (
                    extra_logits.contiguous().numpy().astype(np.dtype("<f4"), copy=False)
                )
                extra_payload = memoryview(extra_array).cast("B")
                extra_path = prefix.with_suffix(".extra-logits.f32")
                _write_exclusive(extra_path, extra_payload)
                comparison = torch.abs(extra_logits[0] - logits[-1])
                extra_receipt = {
                    "path": str(extra_path),
                    "bytes": len(extra_payload),
                    "sha256": hashlib.sha256(extra_payload).hexdigest(),
                    "relationship": "sampling_witness_vs_final_block_row",
                    "bit_exact": torch.equal(extra_logits[0], logits[-1]),
                    "max_abs_error": float(comparison.max().item()),
                    "mean_abs_error": float(comparison.mean().item()),
                    "sampling_argmax": int(torch.argmax(extra_logits[0]).item()),
                    "block_argmax": int(torch.argmax(logits[-1]).item()),
                }
            row_digests = [
                hashlib.sha256(logits_array[row].tobytes()).hexdigest()
                for row in range(expected)
            ]
            return {
                "schema": SCHEMA,
                "request_id": active["request_id"],
                "rows": expected,
                "compute_logits_call_rows": logit_call_rows,
                "removed_duplicate_logit_row": duplicate_logit_row,
                "logit_selection": logit_selection,
                "extra_logits": extra_receipt,
                "target_layers": list(TARGET_LAYERS),
                "hidden": {
                    "path": str(hidden_path),
                    "bytes": len(hidden_payload),
                    "sha256": hashlib.sha256(hidden_payload).hexdigest(),
                    "layout": "row-target-layer-hidden",
                    "dtype": "f32_le",
                },
                "logits": {
                    "path": str(logits_path),
                    "bytes": len(logits_payload),
                    "sha256": hashlib.sha256(logits_payload).hexdigest(),
                    "row_sha256": row_digests,
                    "layout": "row-vocabulary",
                    "dtype": "f32_le",
                },
            }
        finally:
            _ACTIVE = None


def abort_capture() -> None:
    global _ACTIVE
    with _LOCK:
        _ACTIVE = None
