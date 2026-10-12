"""Exact token-major target hidden capture for the Spark DFlash producer."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import threading
import time
from typing import Any, Callable


SCHEMA = "muser.spark-dflash-target-features.v1"
TARGET_LAYERS = (1, 13, 25, 37, 49)
HIDDEN_SIZE = 6656
CAPTURE_DTYPES = {
    "f16_le": ("float16", "<f2", 2),
    "f32_le": ("float32", "<f4", 4),
}
_LOCK = threading.Lock()
_ACTIVE: dict[str, Any] | None = None
_COMPLETED: dict[str, Any] | None = None


def begin_capture(
    request_id: str,
    cached_tokens: int,
    output: Path,
    *,
    token_ids: list[int] | None = None,
    session_output: Path | None = None,
    device: str | None = None,
    dtype: str = "f32_le",
    host_ready_callback: Callable[[dict[str, Any]], None] | None = None,
    row_selection: str = "prefix",
) -> None:
    global _ACTIVE
    if cached_tokens < 1:
        raise RuntimeError("DFlash capture requires a non-empty cached prefix")
    if dtype not in CAPTURE_DTYPES:
        raise RuntimeError(f"unsupported DFlash capture dtype: {dtype}")
    if row_selection not in {"prefix", "suffix"}:
        raise RuntimeError(f"unsupported DFlash capture row selection: {row_selection}")
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"refusing to replace DFlash feature capture: {output}")
    if (token_ids is None) != (session_output is None):
        raise RuntimeError("DFlash session build requires both tokens and an output")
    if token_ids is not None and len(token_ids) != cached_tokens + 1:
        raise RuntimeError("DFlash session build tokens differ from the captured prefix")
    if token_ids is not None and row_selection != "prefix":
        raise RuntimeError("DFlash session build requires prefix row selection")
    if session_output is not None and (session_output.exists() or session_output.is_symlink()):
        raise RuntimeError(f"refusing to replace DFlash session: {session_output}")
    matrix = None
    copy_stream = None
    if device is not None:
        import torch

        torch_dtype = getattr(torch, CAPTURE_DTYPES[dtype][0])
        if torch.device(device).type == "cuda":
            # A dedicated stream copies each completed layer into one pinned,
            # layer-major host allocation while later decoder layers run.
            matrix = torch.empty(
                (len(TARGET_LAYERS), cached_tokens, HIDDEN_SIZE),
                dtype=torch_dtype,
                device="cpu",
                pin_memory=True,
            )
            copy_stream = torch.cuda.Stream(device=device)
        else:
            matrix = torch.empty(
                (len(TARGET_LAYERS), cached_tokens, HIDDEN_SIZE),
                dtype=torch_dtype,
                device=device,
            )
    with _LOCK:
        if _ACTIVE is not None:
            raise RuntimeError("a DFlash target-feature capture is already active")
        _ACTIVE = {
            "request_id": request_id,
            "cached_tokens": cached_tokens,
            "capture_started_ns": time.perf_counter_ns(),
            "output": output,
            "matrix": matrix,
            "copy_stream": copy_stream,
            "device": device,
            "dtype": dtype,
            "seen": set(),
            "layer_timings": {},
            "host_ready_callback": host_ready_callback,
            "host_ready_callback_completed_offset_ns": None,
            "host_ready_callback_started_offset_ns": None,
            "host_ready_offset_ns": None,
            "materialized_payload": None,
            "materialized_payload_sha256": None,
            "payload_ready_offset_ns": None,
            "row_selection": row_selection,
            "token_ids": list(token_ids) if token_ids is not None else None,
            "session_output": session_output,
        }


def capture_layer(layer: int, hidden_states: Any) -> None:
    callback_dispatch: tuple[
        Callable[[dict[str, Any]], None], dict[str, Any], dict[str, Any]
    ] | None = None
    with _LOCK:
        active = _ACTIVE
        if active is None or layer not in TARGET_LAYERS:
            return
        if layer in active["seen"]:
            raise RuntimeError(f"DFlash target layer {layer} was captured twice")
        arrival_ns = time.perf_counter_ns()
        cached_tokens = active["cached_tokens"]
        if hidden_states.ndim != 2 or tuple(hidden_states.shape)[1] != HIDDEN_SIZE:
            raise RuntimeError(
                f"DFlash target layer {layer} has shape {tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[0] < cached_tokens:
            raise RuntimeError(
                f"DFlash target layer {layer} has only {hidden_states.shape[0]} rows"
            )
        import torch

        matrix = active["matrix"]
        if matrix is None:
            torch_dtype = getattr(torch, CAPTURE_DTYPES[active["dtype"]][0])
            matrix = torch.empty(
                (len(TARGET_LAYERS), cached_tokens, HIDDEN_SIZE),
                dtype=torch_dtype,
                device=hidden_states.device,
            )
            active["matrix"] = matrix
            active["device"] = str(hidden_states.device)
        copy_stream = active["copy_stream"]
        selected_rows = (
            hidden_states[:cached_tokens]
            if active["row_selection"] == "prefix"
            else hidden_states[-cached_tokens:]
        )
        if copy_stream is not None:
            current_stream = torch.cuda.current_stream(hidden_states.device)
            copy_stream.wait_stream(current_stream)
            index = TARGET_LAYERS.index(layer)
            with torch.cuda.stream(copy_stream):
                matrix[index].copy_(
                    selected_rows.detach(), non_blocking=True
                )
            hidden_states.record_stream(copy_stream)
        elif matrix.device != hidden_states.device:
            raise RuntimeError(
                f"DFlash target layer {layer} moved from {matrix.device} "
                f"to {hidden_states.device}"
            )
        else:
            # CPU fixtures and the exact producer retain the synchronous
            # fallback without requiring a CUDA runtime.
            index = TARGET_LAYERS.index(layer)
            matrix[index].copy_(selected_rows.detach())
        copy_enqueued_ns = time.perf_counter_ns()
        active["layer_timings"][layer] = {
            "arrival_offset_ns": arrival_ns - active["capture_started_ns"],
            "copy_enqueued_offset_ns": copy_enqueued_ns
            - active["capture_started_ns"],
            "copy_is_async": copy_stream is not None,
            "layer": layer,
        }
        active["seen"].add(layer)
        callback = active["host_ready_callback"]
        if callback is not None and len(active["seen"]) == len(TARGET_LAYERS):
            # All layer copies share one ordered stream. Synchronizing it after
            # layer 49 makes every selected row host-ready before the callback.
            if copy_stream is not None:
                copy_stream.synchronize()
            active["host_ready_offset_ns"] = (
                time.perf_counter_ns() - active["capture_started_ns"]
            )
            matrix = active["matrix"].permute(1, 0, 2).contiguous()
            _, numpy_dtype, bytes_per_element = CAPTURE_DTYPES[active["dtype"]]
            payload = bytes(
                memoryview(matrix.numpy().astype(numpy_dtype, copy=False)).cast("B")
            )
            payload_sha256 = hashlib.sha256(payload).hexdigest()
            active["materialized_payload"] = payload
            active["materialized_payload_sha256"] = payload_sha256
            active["payload_ready_offset_ns"] = (
                time.perf_counter_ns() - active["capture_started_ns"]
            )
            active["host_ready_callback"] = None
            callback_dispatch = (
                callback,
                {
                    "schema": SCHEMA,
                    "request_id": active["request_id"],
                    "payload": payload,
                    "sha256": payload_sha256,
                    "bytes": len(payload),
                    "cached_tokens": active["cached_tokens"],
                    "capture_started_ns": active["capture_started_ns"],
                    "host_ready_offset_ns": active["host_ready_offset_ns"],
                    "payload_ready_offset_ns": active["payload_ready_offset_ns"],
                    "layer_timings": [
                        active["layer_timings"][target_layer]
                        for target_layer in TARGET_LAYERS
                    ],
                    "target_layers": list(TARGET_LAYERS),
                    "hidden_size": HIDDEN_SIZE,
                    "dtype": active["dtype"],
                    "row_bytes": len(TARGET_LAYERS)
                    * HIDDEN_SIZE
                    * bytes_per_element,
                    "layout": "token-major-selected-layer-major-hidden",
                },
                active,
            )

    if callback_dispatch is not None:
        callback, callback_receipt, active = callback_dispatch
        callback_started_ns = time.perf_counter_ns()
        with _LOCK:
            if _ACTIVE is active:
                active["host_ready_callback_started_offset_ns"] = (
                    callback_started_ns - active["capture_started_ns"]
                )
        try:
            # Never hold the capture lock across transport or user code.
            callback(callback_receipt)
        finally:
            callback_completed_ns = time.perf_counter_ns()
            with _LOCK:
                if _ACTIVE is active:
                    active["host_ready_callback_completed_offset_ns"] = (
                        callback_completed_ns - active["capture_started_ns"]
                    )


def finish_capture(
    *, materialize: bool = True, include_payload: bool = False
) -> dict[str, Any] | None:
    global _ACTIVE
    with _LOCK:
        active = _ACTIVE
        if active is None:
            return None
        try:
            finish_started_ns = time.perf_counter_ns()
            missing = [layer for layer in TARGET_LAYERS if layer not in active["seen"]]
            if missing:
                raise RuntimeError(f"DFlash target capture missed layers {missing}")
            if (
                active["copy_stream"] is not None
                and active["materialized_payload"] is None
            ):
                active["copy_stream"].synchronize()
            # The host staging buffer is layer-major for contiguous async
            # copies. Transpose once into DFlash's token-major ABI. Keep it as
            # a buffer view so hashing and writing do not allocate a second
            # hundreds-of-megabytes bytes object.
            _, numpy_dtype, bytes_per_element = CAPTURE_DTYPES[active["dtype"]]
            payload = active["materialized_payload"]
            payload_sha256 = active["materialized_payload_sha256"]
            if payload is None:
                matrix = active["matrix"].permute(1, 0, 2).contiguous()
                payload = memoryview(
                    matrix.numpy().astype(numpy_dtype, copy=False)
                ).cast("B")
                payload_sha256 = hashlib.sha256(payload).hexdigest()
            output: Path = active["output"]
            if materialize:
                output.parent.mkdir(parents=True, exist_ok=True)
                descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                try:
                    written = 0
                    while written < len(payload):
                        written += os.write(descriptor, payload[written:])
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            receipt = {
                "schema": SCHEMA,
                "request_id": active["request_id"],
                "output": str(output) if materialize else None,
                "materialized": materialize,
                "sha256": payload_sha256,
                "bytes": len(payload),
                "cached_tokens": active["cached_tokens"],
                "capture_started_ns": active["capture_started_ns"],
                "finish_started_offset_ns": finish_started_ns
                - active["capture_started_ns"],
                "host_ready_callback_completed_offset_ns": active[
                    "host_ready_callback_completed_offset_ns"
                ],
                "host_ready_callback_started_offset_ns": active[
                    "host_ready_callback_started_offset_ns"
                ],
                "host_ready_offset_ns": active["host_ready_offset_ns"],
                "layer_timings": [
                    active["layer_timings"][layer] for layer in TARGET_LAYERS
                ],
                "target_layers": list(TARGET_LAYERS),
                "hidden_size": HIDDEN_SIZE,
                "dtype": active["dtype"],
                "row_bytes": len(TARGET_LAYERS) * HIDDEN_SIZE * bytes_per_element,
                "layout": "token-major-selected-layer-major-hidden",
                "payload_ready_offset_ns": active["payload_ready_offset_ns"],
                "seal_eligible": False,
            }
            if include_payload:
                # A serving caller needs the same authenticated bytes without
                # a disposable fsync/read round trip. Copy once out of the
                # pinned staging allocation so its lifetime is independent of
                # the capture hook's global state.
                receipt["payload"] = (
                    payload if isinstance(payload, bytes) else bytes(payload)
                )
            receipt["finish_completed_offset_ns"] = (
                time.perf_counter_ns() - active["capture_started_ns"]
            )
            return receipt
        finally:
            _ACTIVE = None


def abort_capture() -> None:
    global _ACTIVE, _COMPLETED
    with _LOCK:
        _ACTIVE = None
        _COMPLETED = None


def _exclusive_text(path: Path, text: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="ascii") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def finish_capture_for_connector() -> dict[str, Any] | None:
    """Finish and build a draft session only for native deferred requests."""
    global _COMPLETED
    with _LOCK:
        active = _ACTIVE
        if active is None or active["session_output"] is None:
            return None
        token_ids = list(active["token_ids"])
        session_output = Path(active["session_output"])
    receipt = finish_capture()
    if receipt is None:
        raise RuntimeError("native DFlash capture disappeared before connector seal")
    fifo_raw = os.environ.get("MUSER_DFLASH_JOBS_FIFO")
    if not fifo_raw:
        raise RuntimeError("native DFlash capture requires MUSER_DFLASH_JOBS_FIFO")
    fifo = Path(fifo_raw)
    if not fifo.exists() or not stat.S_ISFIFO(fifo.stat().st_mode):
        raise RuntimeError(f"DFlash jobs FIFO is unavailable: {fifo}")
    prefix = session_output.with_suffix("")
    tokens_path = prefix.with_suffix(".tokens")
    job_path = prefix.with_suffix(".dflash.job")
    status_path = prefix.with_suffix(".dflash.status")
    stdout_path = prefix.with_suffix(".dflash.stdout")
    _exclusive_text(tokens_path, "".join(f"{token}\n" for token in token_ids))
    lines = [
        f"n_ctx {max(2048, len(token_ids))}",
        "n_batch 2048",
        "n_ubatch 512",
        "flash_attn 1",
        "skip_tail 1",
        f"tokens {tokens_path.resolve()}",
        f"draft_out {session_output.resolve()}",
        f"dflash_features {Path(receipt['output']).resolve()}",
        f"stdout {stdout_path.resolve()}",
        f"status {status_path.resolve()}",
    ]
    _exclusive_text(job_path, "\n".join(lines) + "\n")
    started = time.perf_counter_ns()
    try:
        descriptor = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
    except OSError as error:
        raise RuntimeError("DFlash exporter is not reading its jobs FIFO") from error
    with os.fdopen(descriptor, "w", encoding="ascii") as jobs:
        jobs.write(f"{job_path.resolve()}\n")
        jobs.flush()
    deadline = time.monotonic() + float(
        os.environ.get("MUSER_DFLASH_TIMEOUT_SECONDS", "180")
    )
    while time.monotonic() < deadline:
        try:
            status = status_path.read_text(encoding="ascii").strip()
        except FileNotFoundError:
            time.sleep(0.025)
            continue
        if status != "ok":
            raise RuntimeError(f"DFlash exporter returned status {status!r}")
        break
    else:
        raise RuntimeError("DFlash exporter timed out")
    if not session_output.is_file():
        raise RuntimeError("DFlash exporter did not publish the requested session")
    session_payload = session_output.read_bytes()
    receipt = dict(receipt)
    receipt["session"] = {
        "path": str(session_output),
        "sha256": hashlib.sha256(session_payload).hexdigest(),
        "bytes": len(session_payload),
        "build_ns": time.perf_counter_ns() - started,
        "job": str(job_path),
        "stdout": str(stdout_path),
    }
    with _LOCK:
        _COMPLETED = receipt
    return receipt


def consume_completed_capture() -> dict[str, Any] | None:
    global _COMPLETED
    with _LOCK:
        receipt = _COMPLETED
        _COMPLETED = None
        return receipt
