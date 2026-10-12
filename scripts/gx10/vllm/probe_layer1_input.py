#!/usr/bin/env python3
"""Isolate live producer layer-1 input norm and merged QKV boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from muser_vllm.exact_fp4_quant import exact_scaled_fp4_quant
from muser_vllm.exact_rms_norm import exact_split_rms_norm
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    pad_nvfp4_weight_for_cutlass,
    swizzle_blockscale,
)
from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm


PREFIX = "model.language_model"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--live-layer0", type=Path)
    inputs.add_argument("--mac-layer0", type=Path)
    parser.add_argument("--override-token", type=int)
    parser.add_argument("--override-element", type=int)
    parser.add_argument("--override-value", type=float)
    parser.add_argument("--mac-layer1", required=True, type=Path)
    args = parser.parse_args()
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]

    def tensor(key: str) -> torch.Tensor:
        with safe_open(
            args.checkpoint / weight_map[key], framework="pt", device="cpu"
        ) as handle:
            return handle.get_tensor(key)

    hidden = 6656
    widths = {"q": 4096, "k": 256, "v": 256}
    if args.live_layer0 is not None:
        live = np.fromfile(args.live_layer0 / "layer_out.f16", dtype="<f2").reshape(
            -1, hidden
        )
    else:
        live = (
            np.fromfile(args.mac_layer0 / "l_out-0.f32", dtype="<f4")
            .reshape(-1, hidden)
            .astype("<f2")
        )
    overrides = (args.override_token, args.override_element, args.override_value)
    if any(value is not None for value in overrides):
        if any(value is None for value in overrides):
            raise RuntimeError("layer-0 override requires token, element, and value")
        live[int(args.override_token), int(args.override_element)] = np.float16(
            args.override_value
        )
    values = torch.from_numpy(live).cuda()
    layer = f"{PREFIX}.layers.1"
    weight = tensor(f"{layer}.input_layernorm.weight").to(
        dtype=torch.float16, device="cuda"
    )
    normed = exact_split_rms_norm(values, weight, 1.0e-5, 1)

    reports: list[dict[str, object]] = []

    def expected(name: str, width: int) -> np.ndarray:
        return (
            np.fromfile(args.mac_layer1 / f"{name}.f32", dtype="<f4")
            .astype("<f2")
            .reshape(-1, width)
        )

    def compare(name: str, actual: torch.Tensor, reference: np.ndarray) -> None:
        torch.cuda.synchronize()
        actual_array = actual.detach().to(torch.float16).cpu().numpy()
        mismatch = actual_array.view(np.uint16) != reference.view(np.uint16)
        locations = np.argwhere(mismatch)
        first = None
        if locations.size:
            row, column = (int(value) for value in locations[0])
            first = {
                "token": row,
                "element": column,
                "actual": float(actual_array[row, column]),
                "actual_bits": int(actual_array.view(np.uint16)[row, column]),
                "expected": float(reference[row, column]),
                "expected_bits": int(reference.view(np.uint16)[row, column]),
            }
        reports.append(
            {
                "name": name,
                "elements": actual_array.size,
                "mismatches": int(mismatch.sum()),
                "token0_mismatches": int(mismatch[0].sum()),
                "mismatches_by_token": [int(value) for value in mismatch.sum(axis=1)],
                "max_abs": float(
                    np.max(
                        np.abs(
                            actual_array.astype(np.float32)
                            - reference.astype(np.float32)
                        )
                    )
                ),
                "first": first,
            }
        )

    compare("attn_norm", normed, expected("attn_norm-0", hidden))

    packed_parts = []
    scale_parts = []
    input_globals = []
    weight_globals = []
    for projection in ("q", "k", "v"):
        base = f"{layer}.self_attn.{projection}_proj"
        packed_parts.append(tensor(base + ".weight_packed"))
        scale_parts.append(tensor(base + ".weight_scale"))
        input_globals.append(float(tensor(base + ".input_global_scale")))
        weight_globals.append(float(tensor(base + ".weight_global_scale")))
    if len(set(input_globals)) != 1 or len(set(weight_globals)) != 1:
        raise RuntimeError("layer-1 QKV projections do not share global scales")
    packed = torch.cat(packed_parts, dim=0).cuda()
    scales = swizzle_blockscale(torch.cat(scale_parts, dim=0).cuda())
    packed, padding_bytes = pad_nvfp4_weight_for_cutlass(packed)
    if padding_bytes:
        raise RuntimeError(f"unexpected layer-1 QKV padding: {padding_bytes}")
    activation, activation_scale = exact_scaled_fp4_quant(
        normed,
        torch.tensor(input_globals[0], dtype=torch.float32, device="cuda"),
    )
    alpha = torch.tensor(
        1.0 / (input_globals[0] * weight_globals[0]),
        dtype=torch.float32,
        device="cuda",
    )
    qkv = flashinfer_scaled_fp4_mm(
        activation,
        packed,
        activation_scale,
        scales,
        alpha,
        torch.float16,
        backend="cutlass",
    )
    offset = 0
    for projection in ("q", "k", "v"):
        width = widths[projection]
        actual = qkv[:, offset : offset + width]
        if projection in ("q", "k"):
            heads = width // 128
            order = torch.tensor(
                [index for pair in zip(range(64), range(64, 128)) for index in pair],
                dtype=torch.int64,
                device="cuda",
            )
            actual = actual.reshape(-1, heads, 128)[:, :, order].reshape(-1, width)
        compare(projection, actual, expected(f"{projection.upper()}cur-0", width))
        offset += width
    print(
        json.dumps(
            {
                "schema": "muser.spark-layer1-input-probe.v1",
                "reports": reports,
                "seal_eligible": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
