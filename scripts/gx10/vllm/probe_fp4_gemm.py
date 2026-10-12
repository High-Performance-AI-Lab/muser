#!/usr/bin/env python3
"""Diagnose Blackwell NVFP4 GEMM accumulation order on a closed fixture."""

from __future__ import annotations

import json

import numpy as np
import torch

from muser_vllm.exact_fp4_quant import exact_scaled_fp4_quant
from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm


E2M1 = np.asarray(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def logical_scales(swizzled: torch.Tensor, rows: int, groups: int) -> np.ndarray:
    rounded_groups = (groups + 3) // 4 * 4
    raw = swizzled.view(torch.uint8).cpu().numpy().reshape(-1)
    codes = np.empty((rows, groups), dtype=np.uint8)
    for row in range(rows):
        for group in range(groups):
            offset = (
                (row // 128) * rounded_groups * 128
                + (group // 4) * 512
                + (row % 32) * 16
                + ((row % 128) // 32) * 4
                + group % 4
            )
            codes[row, group] = raw[offset]
    return torch.from_numpy(codes).view(torch.float8_e4m3fn).float().numpy()


def unpack(values: torch.Tensor) -> np.ndarray:
    raw = values.cpu().numpy()
    result = np.empty((raw.shape[0], raw.shape[1] * 2), dtype=np.float32)
    result[:, 0::2] = E2M1[raw & 0x0F]
    result[:, 1::2] = E2M1[raw >> 4]
    return result


def add32(left: np.float32, right: np.float32) -> np.float32:
    return np.float32(left + right)


def sequential_axis(values: np.ndarray) -> np.ndarray:
    total = np.zeros(values.shape[:-1], dtype=np.float32)
    for index in range(values.shape[-1]):
        total = np.add(total, values[..., index], dtype=np.float32)
    return total


def pairwise_axis(values: np.ndarray) -> np.ndarray:
    work = values
    while work.shape[-1] > 1:
        paired = np.add(work[..., 0:-1:2], work[..., 1::2], dtype=np.float32)
        if work.shape[-1] % 2:
            work = np.concatenate((paired, work[..., -1:]), axis=-1)
        else:
            work = paired
    return work[..., 0]


def reduce_chunks(values: np.ndarray, width: int, inner_pairwise: bool) -> np.ndarray:
    reducer = pairwise_axis if inner_pairwise else sequential_axis
    partial = np.stack(
        [reducer(values[..., index : index + width])
         for index in range(0, values.shape[-1], width)],
        axis=-1,
    )
    return sequential_axis(partial)


def reduce_strided(values: np.ndarray, lanes: int, tree: bool) -> np.ndarray:
    partial = np.stack(
        [sequential_axis(values[..., lane::lanes]) for lane in range(lanes)], axis=-1
    )
    return pairwise_axis(partial) if tree else sequential_axis(partial)


def main() -> None:
    torch.manual_seed(170018)
    rows, outputs, width = 33, 256, 6656
    a = torch.randn((rows, width), dtype=torch.float32).mul_(0.75).to(torch.float16)
    b = torch.randn((outputs, width), dtype=torch.float32).mul_(0.5).to(torch.float16)
    a_global = torch.tensor(43.75, dtype=torch.float32, device="cuda")
    b_global = torch.tensor(37.25, dtype=torch.float32, device="cuda")
    a_packed, a_scales = exact_scaled_fp4_quant(a.cuda(), a_global)
    b_packed, b_scales = exact_scaled_fp4_quant(b.cuda(), b_global)
    alpha = torch.tensor(
        1.0 / (float(a_global) * float(b_global)), dtype=torch.float32, device="cuda"
    )
    actual = flashinfer_scaled_fp4_mm(
        a_packed,
        b_packed,
        a_scales,
        b_scales,
        alpha,
        torch.float16,
        backend="cutlass",
    )
    torch.cuda.synchronize()

    a_codes = unpack(a_packed)
    b_codes = unpack(b_packed)
    groups = width // 16
    a_scale = logical_scales(a_scales, rows, groups)
    b_scale = logical_scales(b_scales, outputs, groups)
    actual_bits = actual.view(torch.uint16).cpu().numpy()
    candidates: dict[str, np.ndarray] = {}
    reducers = {
        "sequential": sequential_axis,
        "pairwise": pairwise_axis,
    }
    for chunk in (2, 4, 8, 16, 32, 64, 128, 256):
        reducers[f"chunk{chunk}-sequential"] = (
            lambda values, chunk=chunk: reduce_chunks(values, chunk, False)
        )
        reducers[f"chunk{chunk}-pairwise"] = (
            lambda values, chunk=chunk: reduce_chunks(values, chunk, True)
        )
    for lanes in (2, 4, 8, 16, 32):
        reducers[f"stride{lanes}-sequential"] = (
            lambda values, lanes=lanes: reduce_strided(values, lanes, False)
        )
        reducers[f"stride{lanes}-pairwise"] = (
            lambda values, lanes=lanes: reduce_strided(values, lanes, True)
        )
    contributions = np.empty((rows, outputs, groups), dtype=np.float32)
    for group in range(groups):
        start = group * 16
        block = np.zeros((rows, outputs), dtype=np.float32)
        for element in range(16):
            product = np.multiply(
                a_codes[:, start + element, None],
                b_codes[None, :, start + element],
                dtype=np.float32,
            )
            block = np.add(block, product, dtype=np.float32)
        scale = np.multiply(
            a_scale[:, group, None], b_scale[None, :, group], dtype=np.float32
        )
        contributions[..., group] = np.multiply(block, scale, dtype=np.float32)
    alpha_cpu = np.float32(alpha.cpu())
    for name, reducer in reducers.items():
        value = np.multiply(reducer(contributions), alpha_cpu, dtype=np.float32)
        candidates[name] = value.astype(np.float16).view(np.uint16)

    matches = {
        name: int(np.count_nonzero(bits == actual_bits))
        for name, bits in candidates.items()
    }
    best = sorted(matches.items(), key=lambda item: (-item[1], item[0]))[:12]
    print(
        json.dumps(
            {
                "schema": "muser.spark-fp4-gemm-accumulation-probe.v1",
                "shape": [rows, outputs, width],
                "cells": rows * outputs,
                "best": best,
                "actual_sha256": __import__("hashlib").sha256(
                    actual_bits.tobytes()
                ).hexdigest(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
