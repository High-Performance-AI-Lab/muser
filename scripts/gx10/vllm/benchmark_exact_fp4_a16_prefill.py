#!/usr/bin/env python3
"""Bounded real-projection benchmark for the exact A16 NVFP4 prefill kernel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from safetensors.torch import load_file

from muser_vllm.exact_fp4_mm import exact_fp4_a16_mm


PREFIX = "model.language_model.layers.1.self_attn.q_proj"
HIDDEN = 6656
OUTPUT = 4096


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-cache", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--tokens", required=True, type=int)
    parser.add_argument("--reps", type=int, default=2)
    args = parser.parse_args()
    if not 16 <= args.tokens <= 2048 or not 1 <= args.reps <= 5:
        parser.error("tokens must be 16..2048 and reps must be 1..5")

    tensors = load_file(args.tensor_cache, device="cpu")
    raw = np.fromfile(args.input, dtype="<f4").reshape(-1, HIDDEN)
    repeats = (args.tokens + raw.shape[0] - 1) // raw.shape[0]
    activation = torch.from_numpy(
        np.tile(raw, (repeats, 1))[: args.tokens].astype(np.float16, copy=True)
    ).cuda()
    weight = tensors[PREFIX + ".weight_packed"].contiguous().cuda()
    weight_scale = tensors[PREFIX + ".weight_scale"].contiguous().cuda()
    weight_scale2 = torch.tensor(
        1.0 / float(tensors[PREFIX + ".weight_global_scale"]),
        dtype=torch.float32,
        device="cuda",
    )

    started = time.perf_counter()
    output = exact_fp4_a16_mm(
        activation, weight, weight_scale, weight_scale2
    )
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - started
    if output.shape != (args.tokens, OUTPUT):
        raise RuntimeError(f"unexpected output shape: {tuple(output.shape)}")

    samples = []
    for _ in range(args.reps):
        started = time.perf_counter()
        output = exact_fp4_a16_mm(
            activation, weight, weight_scale, weight_scale2
        )
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - started)
    median = float(np.median(np.asarray(samples, dtype=np.float64)))
    integer_ops = 2 * args.tokens * HIDDEN * OUTPUT
    print(
        json.dumps(
            {
                "schema": "muser.spark-exact-fp4-a16-prefill-benchmark.v1",
                "shape": [args.tokens, OUTPUT, HIDDEN],
                "warmup_seconds": warmup_seconds,
                "samples_seconds": samples,
                "median_seconds": median,
                "effective_tops": integer_ops / median / 1.0e12,
                "output_checksum": float(output.float().sum().cpu()),
                "status": "pass",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
