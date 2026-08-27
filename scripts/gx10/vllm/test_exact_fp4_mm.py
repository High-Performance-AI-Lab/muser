#!/usr/bin/env python3
"""Small GB10 oracle fixture for the integer NVFP4 contraction."""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import torch

from muser_vllm.exact_fp4_mm import CONTRACTION_SCALE_INV, exact_fp4_mm
from muser_vllm.exact_fp4_quant import exact_scaled_fp4_quant


E2M1_Q1 = np.asarray(
    [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
    dtype=np.int64,
)


def logical_scale_codes(swizzled: torch.Tensor, rows: int, groups: int) -> np.ndarray:
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
    return codes


def e4m3_q9(codes: np.ndarray) -> np.ndarray:
    magnitude_code = codes & 0x7F
    exponent = (magnitude_code >> 3) & 0x0F
    mantissa = magnitude_code & 7
    magnitude = np.where(
        exponent == 0,
        mantissa,
        (8 + mantissa).astype(np.int64) << np.maximum(exponent.astype(np.int64) - 1, 0),
    ).astype(np.int64)
    return np.where((codes & 0x80) != 0, -magnitude, magnitude)


def unpack_q1(values: torch.Tensor) -> np.ndarray:
    raw = values.cpu().numpy()
    result = np.empty((raw.shape[0], raw.shape[1] * 2), dtype=np.int64)
    result[:, 0::2] = E2M1_Q1[raw & 15]
    result[:, 1::2] = E2M1_Q1[raw >> 4]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    torch.manual_seed(170072)
    # FlashInfer's production FP4 kernel requires an output tile width of 128;
    # keep the smoke fixture at its smallest serving-valid geometry.
    rows, outputs, width = 32, 128, 128
    activation_global = torch.tensor(43.75, dtype=torch.float32, device="cuda")
    weight_global = torch.tensor(37.25, dtype=torch.float32, device="cuda")
    a = torch.randn((rows, width), dtype=torch.float16, device="cuda")
    b = torch.randn((outputs, width), dtype=torch.float16, device="cuda")
    a_packed, a_scale = exact_scaled_fp4_quant(a, activation_global)
    b_packed, b_scale = exact_scaled_fp4_quant(b, weight_global)
    weight_scale2 = torch.reciprocal(weight_global)
    input_scale = torch.reciprocal(activation_global)
    alpha = weight_scale2 * input_scale
    actual = exact_fp4_mm(
        a_packed, b_packed, a_scale, b_scale, alpha, weight_scale2, input_scale
    )
    torch.cuda.synchronize()

    a_codes = unpack_q1(a_packed)
    b_codes = unpack_q1(b_packed)
    groups = width // 16
    a_scales = e4m3_q9(logical_scale_codes(a_scale, rows, groups))
    b_scales = e4m3_q9(logical_scale_codes(b_scale, outputs, groups))
    expected = np.empty((rows, outputs), dtype=np.float16)
    for row in range(rows):
        for output in range(outputs):
            total = np.int64(0)
            for group in range(groups):
                block = np.dot(
                    a_codes[row, group * 16 : (group + 1) * 16],
                    b_codes[output, group * 16 : (group + 1) * 16],
                )
                total += block * a_scales[row, group] * b_scales[output, group]
            scaled = np.multiply(
                np.float32(total), np.float32(CONTRACTION_SCALE_INV), dtype=np.float32
            )
            scaled = np.multiply(
                scaled, np.float32(weight_scale2.cpu()), dtype=np.float32
            )
            scaled = np.multiply(
                scaled, np.float32(input_scale.cpu()), dtype=np.float32
            )
            expected[row, output] = np.float16(scaled)
    actual_array = actual.cpu().numpy()
    mismatch = actual_array.view(np.uint16) != expected.view(np.uint16)
    locations = np.argwhere(mismatch)
    mismatches = int(locations.shape[0])
    first = None
    if locations.size:
        row, column = (int(value) for value in locations[0])
        first = {
            "row": row,
            "column": column,
            "actual": float(actual_array[row, column]),
            "actual_bits": int(actual_array.view(np.uint16)[row, column]),
            "expected": float(expected[row, column]),
            "expected_bits": int(expected.view(np.uint16)[row, column]),
        }
    print(
        json.dumps(
            {
                "schema": "muser.spark-exact-fp4-mm-fixture.v1",
                "shape": [rows, outputs, width],
                "mismatches": mismatches,
                "first_mismatch": first,
                "max_abs": float(
                    np.max(
                        np.abs(
                            actual_array.astype(np.float32)
                            - expected.astype(np.float32)
                        )
                    )
                ),
                "mode": "full-integer-q1-q9-i64",
                "sha256": hashlib.sha256(actual_array.tobytes()).hexdigest(),
            },
            sort_keys=True,
        )
    )
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
