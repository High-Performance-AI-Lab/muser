"""Exact producer RoPE-cache serialization shared by startup and tests."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any


SCHEMA = "muser.vllm-rope-cache.v2"


def interleave_f16_cache(cache: Any) -> Any:
    """Return ``[cos0, sin0, cos1, sin1, ...]`` as little-endian FP32."""
    import numpy as np
    import torch

    if cache.device.type != "cpu":
        raise ValueError("RoPE cache must be on CPU before serialization")
    if cache.dtype != torch.float16 or cache.ndim != 2:
        raise ValueError("RoPE cache must be a rank-2 torch.float16 tensor")
    if cache.shape[1] <= 0 or cache.shape[1] % 2:
        raise ValueError("RoPE cache width must be positive and even")
    half = cache.shape[1] // 2
    interleaved = torch.stack((cache[:, :half], cache[:, half:]), dim=-1)
    return interleaved.reshape(cache.shape).numpy().astype(np.dtype("<f4"), copy=False)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
