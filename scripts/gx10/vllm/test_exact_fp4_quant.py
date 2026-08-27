#!/usr/bin/env python3
"""One-shot CUDA fixture for the pinned exact NVFP4 activation quantizer."""

from __future__ import annotations

import json

import torch

from muser_vllm.exact_fp4_quant import exact_scaled_fp4_quant


def magnitude_code(magnitude: float, scale: float) -> int:
    if magnitude <= scale * 0.25:
        return 0
    if magnitude < scale * 0.75:
        return 1
    if magnitude <= scale * 1.25:
        return 2
    if magnitude < scale * 1.75:
        return 3
    if magnitude <= scale * 2.5:
        return 4
    if magnitude < scale * 3.5:
        return 5
    if magnitude <= scale * 5.0:
        return 6
    return 7


def reference(values: torch.Tensor, global_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    rows, columns = values.shape
    groups = columns // 16
    rounded_rows = (rows + 127) // 128 * 128
    rounded_groups = (groups + 3) // 4 * 4
    packed = torch.empty((rows, columns // 2), dtype=torch.uint8)
    scales = torch.zeros((rounded_rows, rounded_groups), dtype=torch.uint8)
    for row in range(rows):
        for group in range(groups):
            block = values[row, group * 16 : (group + 1) * 16]
            abs_max = block.abs().max().to(torch.float32)
            normalized = abs_max * torch.tensor(1.0 / 6.0, dtype=torch.float32)
            scale_value = (normalized * global_scale).clamp(max=448.0)
            fp8 = scale_value.to(torch.float8_e4m3fn)
            scale_code = int(fp8.view(torch.uint8))
            decoded_scale = float(fp8.to(torch.float32))
            for pair in range(8):
                codes = []
                for value in block[pair * 2 : pair * 2 + 2]:
                    raw = int(value.view(torch.uint16))
                    sign = ((raw >> 15) & 1) << 3
                    magnitude = float(value.abs().to(torch.float32) * global_scale)
                    codes.append(sign | magnitude_code(magnitude, decoded_scale))
                packed[row, group * 8 + pair] = codes[0] | (codes[1] << 4)
            m_tile = row // 128
            k_tile = group // 4
            outer_m = row % 32
            inner_m = (row % 128) // 32
            inner_k = group % 4
            offset = (
                m_tile * (rounded_groups // 4) * 32 * 4 * 4
                + k_tile * 32 * 4 * 4
                + outer_m * 4 * 4
                + inner_m * 4
                + inner_k
            )
            scales.view(-1)[offset] = scale_code
    return packed, scales


def main() -> None:
    torch.manual_seed(0)
    fixture = torch.randn((5, 64), dtype=torch.float32).mul_(2.5).to(torch.float16)
    fixture[0, :16] = torch.tensor(
        [
            -0.0,
            0.0,
            0.25,
            -0.25,
            0.75,
            -0.75,
            1.25,
            -1.25,
            1.75,
            -1.75,
            2.5,
            -2.5,
            3.5,
            -3.5,
            5.0,
            -6.0,
        ],
        dtype=torch.float16,
    )
    global_scale = 43.75
    expected_packed, expected_scales = reference(fixture, global_scale)
    actual_packed, actual_scales = exact_scaled_fp4_quant(
        fixture.cuda(), torch.tensor(global_scale, dtype=torch.float32, device="cuda")
    )
    torch.cuda.synchronize()
    actual_packed = actual_packed.cpu()
    actual_scale_bytes = actual_scales.view(torch.uint8).cpu()
    packed_mismatches = int((actual_packed != expected_packed).sum())
    scale_mismatches = int((actual_scale_bytes != expected_scales).sum())
    if packed_mismatches or scale_mismatches:
        raise RuntimeError(
            f"exact quantizer mismatch: packed={packed_mismatches}, scales={scale_mismatches}"
        )
    print(
        json.dumps(
            {
                "schema": "muser.spark-exact-fp4-quant-fixture.v1",
                "device": torch.cuda.get_device_name(),
                "packed_bytes": actual_packed.numel(),
                "packed_mismatches": packed_mismatches,
                "scale_bytes": actual_scale_bytes.numel(),
                "scale_mismatches": scale_mismatches,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
