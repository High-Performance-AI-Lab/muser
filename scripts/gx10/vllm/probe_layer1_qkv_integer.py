#!/usr/bin/env python3
"""Compare isolated Spark and Mac integer NVFP4 layer-1 QKV projections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from safetensors.torch import load_file

from muser_vllm.exact_fp4_mm import (
    exact_fp4_mm,
    integer_fp4_a16_from_q8,
    quantize_nvfp4_q8k,
)
from muser_vllm.exact_fp4_quant import exact_scaled_fp4_quant
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    pad_nvfp4_weight_for_cutlass,
    swizzle_blockscale,
)


PREFIX = "model.language_model.layers.1.self_attn"
HIDDEN = 6656
WIDTHS = {"q": 4096, "k": 256, "v": 256}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-cache", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--mac-output", required=True, type=Path)
    parser.add_argument("--token-indices", default="7,12,22")
    args = parser.parse_args()
    token_indices = [int(value) for value in args.token_indices.split(",")]
    if not token_indices:
        raise RuntimeError("retained token selection is empty")

    started = time.perf_counter()
    tensors = load_file(args.tensor_cache, device="cpu")
    raw_input = np.fromfile(args.input, dtype="<f4").reshape(-1, HIDDEN)
    if max(token_indices) >= raw_input.shape[0]:
        raise RuntimeError("retained token selection is outside the input fixture")
    values = torch.from_numpy(
        raw_input[token_indices].astype(np.float16, copy=True)
    ).cuda()
    print(
        json.dumps(
            {
                "event": "fixture_load",
                "seconds": time.perf_counter() - started,
                "tokens": token_indices,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    packed_parts = []
    scale_parts = []
    input_globals = []
    weight_globals = []
    for projection in WIDTHS:
        base = f"{PREFIX}.{projection}_proj"
        packed_parts.append(tensors[base + ".weight_packed"])
        scale_parts.append(tensors[base + ".weight_scale"])
        input_scale_key = base + ".input_global_scale"
        if input_scale_key in tensors:
            input_globals.append(float(tensors[input_scale_key]))
        weight_globals.append(float(tensors[base + ".weight_global_scale"]))
    if input_globals and len(input_globals) != len(WIDTHS):
        raise RuntimeError("mixed QKV activation precision in fixture cache")
    if input_globals and len(set(input_globals)) != 1:
        raise RuntimeError("layer-1 QKV projections do not share activation scales")

    prepared = time.perf_counter()
    packed = torch.cat(packed_parts, dim=0).cuda()
    if input_globals:
        if len(set(weight_globals)) != 1:
            raise RuntimeError("W4A4 QKV fixture does not share global weight scales")
        weight_global = torch.tensor(weight_globals[0], dtype=torch.float32, device="cuda")
        scales = swizzle_blockscale(torch.cat(scale_parts, dim=0).cuda())
        packed, padding_bytes = pad_nvfp4_weight_for_cutlass(packed)
        if padding_bytes:
            raise RuntimeError(f"unexpected layer-1 QKV padding: {padding_bytes}")
        input_global = torch.tensor(input_globals[0], dtype=torch.float32, device="cuda")
        activation, activation_scale = exact_scaled_fp4_quant(values, input_global)
        qkv = exact_fp4_mm(
            activation,
            packed,
            activation_scale,
            scales,
            torch.reciprocal(input_global) * torch.reciprocal(weight_global),
            torch.reciprocal(weight_global),
            torch.reciprocal(input_global),
        )
        activation_precision = "nvfp4"
    else:
        quant, activation_scales = quantize_nvfp4_q8k(values)
        projected = []
        for projection_index, projection in enumerate(WIDTHS):
            projected.append(
                integer_fp4_a16_from_q8(
                    quant,
                    activation_scales,
                    packed_parts[projection_index].contiguous().cuda(),
                    scale_parts[projection_index].contiguous().cuda(),
                    torch.tensor(
                        1.0 / weight_globals[projection_index],
                        dtype=torch.float32,
                        device="cuda",
                    ),
                )
            )
        qkv = torch.cat(projected, dim=-1)
        activation_precision = "f16-weight-only-q8k-exact"
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "event": "integer_qkv",
                "seconds": time.perf_counter() - prepared,
                "shape": list(qkv.shape),
                "activation_precision": activation_precision,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    reports = []
    offset = 0
    for projection, width in WIDTHS.items():
        actual = qkv[:, offset : offset + width]
        if projection in ("q", "k"):
            heads = width // 128
            order = torch.tensor(
                [index for pair in zip(range(64), range(64, 128)) for index in pair],
                dtype=torch.int64,
                device="cuda",
            )
            actual = actual.reshape(-1, heads, 128)[:, :, order].reshape(-1, width)
        actual_array = actual.cpu().numpy()
        expected = (
            np.fromfile(args.mac_output / f"{projection}.f32le", dtype="<f4")
            .reshape(len(token_indices), width)
            .astype(np.float16)
        )
        mismatch = actual_array.view(np.uint16) != expected.view(np.uint16)
        locations = np.argwhere(mismatch)
        first = None
        if locations.size:
            token, element = (int(value) for value in locations[0])
            first = {
                "fixture_token": token,
                "source_token": token_indices[token],
                "element": element,
                "actual": float(actual_array[token, element]),
                "actual_bits": int(actual_array.view(np.uint16)[token, element]),
                "expected": float(expected[token, element]),
                "expected_bits": int(expected.view(np.uint16)[token, element]),
            }
        reports.append(
            {
                "projection": projection,
                "elements": int(actual_array.size),
                "mismatches": int(mismatch.sum()),
                "mismatches_by_token": [int(value) for value in mismatch.sum(axis=1)],
                "max_abs": float(
                    np.max(
                        np.abs(
                            actual_array.astype(np.float32)
                            - expected.astype(np.float32)
                        )
                    )
                ),
                "first_mismatch": first,
            }
        )
        offset += width

    result = {
        "schema": "muser.cross-vendor-nvfp4-layer1-qkv-integer.v2",
        "token_indices": token_indices,
        "activation_precision": activation_precision,
        "reports": reports,
        "mismatches": sum(report["mismatches"] for report in reports),
        "seal_eligible": False,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    if result["mismatches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
