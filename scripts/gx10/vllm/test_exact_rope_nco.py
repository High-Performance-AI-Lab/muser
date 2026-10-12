#!/usr/bin/env python3
"""Compare the canonical Triton RoPE microfixture with retained Mac bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from muser_vllm.exact_rope import exact_rope


TOKENS = 4
HEAD_DIM = 128


def source(heads: int, salt: int) -> np.ndarray:
    count = TOKENS * heads * HEAD_DIM
    values = ((np.arange(count) * 37 + salt) % 1009).astype(np.float32)
    return (values * np.float32(0.003_906_25) - np.float32(1.75)).astype(
        np.float16
    ).astype(np.float32).reshape(TOKENS, heads * HEAD_DIM)


def interleaved_to_neox(values: torch.Tensor, heads: int) -> torch.Tensor:
    return (
        values.reshape(TOKENS, heads, HEAD_DIM // 2, 2)
        .transpose(-2, -1)
        .reshape(TOKENS, heads * HEAD_DIM)
    )


def neox_to_interleaved(values: torch.Tensor, heads: int) -> torch.Tensor:
    return (
        values.reshape(TOKENS, heads, 2, HEAD_DIM // 2)
        .transpose(-2, -1)
        .reshape(TOKENS, heads * HEAD_DIM)
    )


def compare(name: str, actual: np.ndarray, expected_path: Path) -> dict[str, object]:
    expected = np.fromfile(expected_path, dtype="<f4").reshape(actual.shape)
    mismatch = actual.view(np.uint32) != expected.view(np.uint32)
    return {
        "name": name,
        "elements": int(actual.size),
        "mismatches": int(mismatch.sum()),
        "max_abs": float(np.max(np.abs(actual - expected))),
        "actual_sha256": hashlib.sha256(actual.astype("<f4").tobytes()).hexdigest(),
        "expected_sha256": hashlib.sha256(expected.astype("<f4").tobytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mac-output", required=True, type=Path)
    args = parser.parse_args()
    query = interleaved_to_neox(torch.from_numpy(source(2, 11)).half().cuda(), 2)
    key = interleaved_to_neox(torch.from_numpy(source(1, 29)).half().cuda(), 1)
    positions = torch.arange(TOKENS, device="cuda", dtype=torch.long)
    loaded_cache_shape_only = torch.empty((TOKENS, HEAD_DIM), device="cuda", dtype=torch.float16)
    query, key = exact_rope(query, key, loaded_cache_shape_only, positions, HEAD_DIM)
    query = neox_to_interleaved(query, 2).float().cpu().numpy()
    key = neox_to_interleaved(key, 1).float().cpu().numpy()
    reports = [
        compare("q", query, args.mac_output / "q.f32le"),
        compare("k", key, args.mac_output / "k.f32le"),
    ]
    result = {
        "schema": "muser.cross-vendor-q30-rope-micro.v1",
        "reports": reports,
        "bit_exact": all(report["mismatches"] == 0 for report in reports),
        "seal_eligible": False,
    }
    print(json.dumps(result, sort_keys=True))
    if not result["bit_exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
