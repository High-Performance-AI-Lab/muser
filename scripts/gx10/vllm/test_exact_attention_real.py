#!/usr/bin/env python3
"""Bit-gate the CUDA attention kernel against a captured Metal layer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from muser_vllm.exact_attention import exact_attention_gate


TOKENS = 32
HEADS = 32
KV_HEADS = 2
HEAD_DIM = 128


def read_f32(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    expected = int(np.prod(shape))
    if values.size != expected:
        raise RuntimeError(f"{path} has {values.size} values, expected {expected}")
    return values.reshape(shape)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "query": args.capture_dir / "Qcur_rope-0.f32",
        "key": args.capture_dir / "Kcur_rope-0.f32",
        "value": args.capture_dir / "Vcur-0.f32",
        "expected": args.capture_dir / "attn_out-0.f32",
    }
    query = read_f32(paths["query"], (TOKENS, HEADS * HEAD_DIM))
    key = read_f32(paths["key"], (TOKENS, KV_HEADS * HEAD_DIM))
    value = read_f32(paths["value"], (TOKENS, KV_HEADS * HEAD_DIM))
    expected = read_f32(paths["expected"], (TOKENS, HEADS * HEAD_DIM)).astype(
        np.float16
    )

    query_cuda = torch.from_numpy(query).to("cuda", dtype=torch.float16)
    key_cuda = torch.from_numpy(key).to("cuda", dtype=torch.float16)
    value_cuda = torch.from_numpy(value).to("cuda", dtype=torch.float16)
    # The pinned exp contract returns exactly zero below -80, making the
    # sigmoid gate exactly one and exposing the raw attention boundary.
    gate_cuda = torch.full_like(query_cuda, 100.0)
    actual = (
        exact_attention_gate(
            query_cuda,
            key_cuda,
            value_cuda,
            gate_cuda,
            HEAD_DIM**-0.5,
            TOKENS,
        )
        .cpu()
        .numpy()
    )
    expected_bits = expected.view(np.uint16)
    actual_bits = actual.view(np.uint16)
    mismatched_by_token = np.count_nonzero(
        expected_bits != actual_bits, axis=1
    ).tolist()
    mismatched = int(sum(mismatched_by_token))
    max_abs = float(
        np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32)))
    )
    report = {
        "schema": "muser.spark-exact-attention-real-fixture.v1",
        "source_sha256": {name: sha256(path) for name, path in paths.items()},
        "shape": [TOKENS, HEADS * HEAD_DIM],
        "mismatched": mismatched,
        "mismatched_by_token": mismatched_by_token,
        "max_abs": max_abs,
        "seal_eligible": False,
    }
    print(json.dumps(report, sort_keys=True))
    if mismatched:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
