#!/usr/bin/env python3
"""GB10 microfixture for weight-only NVFP4 × dynamic Q8_K."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from muser_vllm.exact_fp4_mm import (
    A16_Q8_CONTRACTION_SCALE_INV,
    exact_fp4_a16_mm,
)


E2M1_Q1 = np.asarray(
    [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
    dtype=np.int64,
)


def e4m3_q9(codes: np.ndarray) -> np.ndarray:
    magnitude_code = codes & 0x7F
    exponent = (magnitude_code >> 3) & 0x0F
    mantissa = magnitude_code & 7
    magnitude = np.where(
        exponent == 0,
        mantissa,
        (8 + mantissa).astype(np.int64)
        << np.maximum(exponent.astype(np.int64) - 1, 0),
    ).astype(np.int64)
    return np.where((codes & 0x80) != 0, -magnitude, magnitude)


def unpack_q1(values: np.ndarray) -> np.ndarray:
    result = np.empty((values.shape[0], values.shape[1] * 2), dtype=np.int64)
    result[:, 0::2] = E2M1_Q1[values & 15]
    result[:, 1::2] = E2M1_Q1[values >> 4]
    return result


def f32_mul(left: np.float32, right: np.float32) -> np.float32:
    return np.float32(np.float64(left) * np.float64(right))


def f32_add(left: np.float32, right: np.float32) -> np.float32:
    return np.float32(np.float64(left) + np.float64(right))


def quantize_q8k(values: np.ndarray) -> tuple[np.ndarray, np.float32]:
    values = values.astype(np.float16).astype(np.float32)
    magnitudes = np.abs(values)
    best = int(np.argmax(magnitudes))
    if magnitudes[best] == 0.0:
        return np.zeros(256, dtype=np.int8), np.float32(0.0)
    inverse_scale = np.float32(np.float32(-127.0) / values[best])
    # Exact fma.rn for f32 operands: their product plus the magic integer fits
    # within binary64's exact significand before the single f32 rounding.
    rounded_f32 = np.asarray(
        [
            np.float32(
                np.float64(inverse_scale) * np.float64(value)
                + np.float64(np.float32(12582912.0))
            )
            for value in values
        ],
        dtype=np.float32,
    )
    rounded_bits = rounded_f32.view(np.int32)
    rounded = (rounded_bits & 0x007FFFFF) - 0x00400000
    quant = np.minimum(127, rounded).astype(np.int8)
    scale = np.float32(np.float32(1.0) / inverse_scale)
    return quant, scale


def main() -> None:
    tokens, outputs, width = 7, 11, 256
    scale2 = np.float32(0.25)
    finite_scales = np.asarray(
        [code for code in range(256) if code not in (0x7F, 0xFF)], dtype=np.uint8
    )
    one_group = np.asarray(
        [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE], dtype=np.uint8
    )
    packed = np.empty((outputs, width // 2), dtype=np.uint8)
    scale_codes = np.empty((outputs, width // 16), dtype=np.uint8)
    for row in range(outputs):
        packed[row] = np.tile(one_group, width // 16)
        for group in range(width // 16):
            scale_codes[row, group] = finite_scales[
                (row * 29 + group * 17) % finite_scales.size
            ]
    activation = np.empty((tokens, width), dtype=np.float32)
    for token in range(tokens):
        for element in range(width):
            activation[token, element] = np.float32(
                ((token * 19 + element * 7 + 3) - 901.0) * 0.003125 + 0.00001
            )
        activation[token, 0] = 2.0
        activation[token, 1] = -2.0
    activation_f16 = activation.astype(np.float16)

    actual = exact_fp4_a16_mm(
        torch.from_numpy(activation_f16).cuda(),
        torch.from_numpy(packed).cuda(),
        torch.from_numpy(scale_codes).cuda().view(torch.float8_e4m3fn),
        torch.tensor(scale2, dtype=torch.float32, device="cuda"),
    )
    torch.cuda.synchronize()

    weights_q1 = unpack_q1(packed)
    scales_q9 = e4m3_q9(scale_codes)
    expected = np.empty((tokens, outputs), dtype=np.float16)
    q8_fixture = np.empty((tokens, width), dtype=np.int8)
    q8_scales = np.empty((tokens,), dtype=np.float32)
    for token in range(tokens):
        q8, q8_scale = quantize_q8k(activation_f16[token])
        q8_fixture[token] = q8
        q8_scales[token] = q8_scale
        for row in range(outputs):
            integer_total = np.int64(0)
            for group in range(width // 16):
                start = group * 16
                dot = np.dot(
                    weights_q1[row, start : start + 16],
                    q8[start : start + 16].astype(np.int64),
                )
                integer_total += dot * scales_q9[row, group]
            contribution = f32_mul(
                np.float32(integer_total), np.float32(A16_Q8_CONTRACTION_SCALE_INV)
            )
            contribution = f32_mul(contribution, q8_scale)
            contribution = f32_mul(contribution, scale2)
            expected[token, row] = np.float16(f32_add(np.float32(0.0), contribution))

    actual_array = actual.cpu().numpy()
    mismatch = actual_array.view(np.uint16) != expected.view(np.uint16)
    locations = np.argwhere(mismatch)
    first = None
    if locations.size:
        token, row = (int(value) for value in locations[0])
        first = {
            "token": token,
            "row": row,
            "actual": float(actual_array[token, row]),
            "actual_bits": int(actual_array.view(np.uint16)[token, row]),
            "expected": float(expected[token, row]),
            "expected_bits": int(expected.view(np.uint16)[token, row]),
        }
    report = {
        "schema": "muser.spark-exact-fp4-a16-q8-fixture.v1",
        "shape": [tokens, outputs, width],
        "mismatches": int(locations.shape[0]),
        "first_mismatch": first,
        "mode": "q8k-e2m1-q1-e4m3-q9-i64",
        "q8_sha256": hashlib.sha256(q8_fixture.tobytes()).hexdigest(),
        "q8_scale_sha256": hashlib.sha256(q8_scales.tobytes()).hexdigest(),
        "output_sha256": hashlib.sha256(actual_array.tobytes()).hexdigest(),
    }
    print(json.dumps(report, sort_keys=True))
    if locations.size:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
