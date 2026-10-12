#!/usr/bin/env python3
"""GB10 fixture for the pinned cross-vendor Muse attention kernel."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import ctypes

import numpy as np
import torch

from muser_vllm.exact_attention import (
    _pack_exact_attention_inputs,
    exact_attention_gate,
)


_LIBC = ctypes.CDLL(None)
_LIBC.fmaf.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
_LIBC.fmaf.restype = ctypes.c_float


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def fma(left: float, right: float, addend: float) -> float:
    return float(_LIBC.fmaf(f32(left), f32(right), f32(addend)))


def add(left: float, right: float) -> float:
    return f32(f32(left) + f32(right))


def mul(left: float, right: float) -> float:
    return f32(f32(left) * f32(right))


def div(numerator: float, denominator: float) -> float:
    return f32(f32(numerator) / f32(denominator))


def exact_exp(value: float) -> float:
    value = f32(value)
    if value < -80.0:
        return 0.0
    rounding = f32(12582912.0)
    z = fma(value, 1.4426950216293335, rounding)
    n = add(z, -rounding)
    reduced = fma(-n, 0.693145751953125, value)
    reduced = fma(-n, 1.428606765330187e-6, reduced)
    exponent = (struct.unpack("<I", struct.pack("<f", z))[0] << 23) & 0xFFFFFFFF
    scale_bits = (exponent + 0x3F800000) & 0xFFFFFFFF
    scale = struct.unpack("<f", struct.pack("<I", scale_bits))[0]
    squared = mul(reduced, reduced)
    p1 = fma(0.008247390389442444, reduced, 0.04189976677298546)
    p2 = fma(0.16668395698070526, reduced, 0.4999912679195404)
    p2 = fma(p1, squared, p2)
    correction = fma(p2, squared, mul(0.9999994039535522, reduced))
    return fma(correction, scale, scale)


def reference(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    gate: np.ndarray,
    scale: float,
    sliding_window: int,
) -> np.ndarray:
    tokens, query_width = query.shape
    heads = query_width // 128
    kv_heads = key.shape[1] // 128
    output = np.empty_like(query, dtype=np.float16)
    for row in range(tokens):
        first = max(0, row + 1 - sliding_window)
        for head in range(heads):
            kv_head = head // (heads // kv_heads)
            running_max = f32(-3.4028234663852886e38)
            denominator = f32(0.0)
            accumulators = [[f32(0.0)] * 4 for _ in range(32)]
            for key_row in range(first, row + 1):
                partial = [f32(0.0)] * 32
                for lane in range(32):
                    for chunk in range(4):
                        dim = chunk * 32 + lane
                        partial[lane] = fma(
                            float(query[row, head * 128 + dim]),
                            float(key[key_row, kv_head * 128 + dim]),
                            partial[lane],
                        )
                offset = 16
                while offset:
                    for lane in range(offset):
                        partial[lane] = add(partial[lane], partial[lane + offset])
                    offset //= 2
                score = mul(partial[0], scale)
                next_max = max(running_max, score)
                old_factor = exact_exp(add(running_max, -next_max))
                new_factor = exact_exp(add(score, -next_max))
                denominator = fma(denominator, old_factor, new_factor)
                for lane in range(32):
                    for chunk in range(4):
                        dim = chunk * 32 + lane
                        accumulators[lane][chunk] = fma(
                            float(value[key_row, kv_head * 128 + dim]),
                            new_factor,
                            mul(accumulators[lane][chunk], old_factor),
                        )
                running_max = next_max
            for lane in range(32):
                for chunk in range(4):
                    dim = chunk * 32 + lane
                    attention = div(accumulators[lane][chunk], denominator)
                    gate_value = float(gate[row, head * 128 + dim])
                    sigmoid = div(1.0, add(1.0, exact_exp(-gate_value)))
                    output[row, head * 128 + dim] = np.float16(
                        mul(attention, sigmoid)
                    )
    return output


def fixture_values(rows: int, width: int, multiplier: int, scale: float) -> np.ndarray:
    return np.asarray(
        [((index * multiplier % 1009) - 503.0) * scale for index in range(rows * width)],
        dtype=np.float32,
    ).astype(np.float16).reshape(rows, width)


def main() -> None:
    tokens = 5
    heads = 32
    kv_heads = 2
    query = fixture_values(tokens, heads * 128, 37, 0.0009765625)
    key = fixture_values(tokens, kv_heads * 128, 41, 0.00146484375)
    value = fixture_values(tokens, kv_heads * 128, 43, 0.001953125)
    gate = fixture_values(tokens, heads * 128, 47, 0.00390625)
    scale = f32(1.0 / math.sqrt(128.0))
    expected = reference(query, key, value, gate, scale, tokens)
    actual = exact_attention_gate(
        torch.from_numpy(query).cuda(),
        torch.from_numpy(key).cuda(),
        torch.from_numpy(value).cuda(),
        torch.from_numpy(gate).cuda(),
        scale,
        tokens,
    )
    torch.cuda.synchronize()
    actual = actual.cpu().numpy()
    mismatches = int(np.count_nonzero(actual.view(np.uint16) != expected.view(np.uint16)))
    if mismatches:
        raise RuntimeError(f"exact attention mismatch: {mismatches}")

    # Recreate the live vLLM QKV split: each logical V row is 256 values but
    # retains the 4608-value projection stride.  The pinned kernel accepts only
    # packed rows, so the forward adapter must materialize this view first.
    projection = torch.empty((tokens, 4096 + 256 + 256), dtype=torch.float16, device="cuda")
    split_query, split_key, split_value = projection.split([4096, 256, 256], dim=-1)
    split_query.copy_(torch.from_numpy(query).cuda())
    split_key.copy_(torch.from_numpy(key).cuda())
    split_value.copy_(torch.from_numpy(value).cuda())
    if split_value.is_contiguous() or split_value.stride() != (4608, 1):
        raise RuntimeError(f"fixture did not reproduce the QKV split stride: {split_value.stride()}")
    packed_query, packed_key, packed_value, packed_gate = _pack_exact_attention_inputs(
        split_query,
        split_key,
        split_value,
        torch.from_numpy(gate).cuda(),
    )
    split_actual = exact_attention_gate(
        packed_query,
        packed_key,
        packed_value,
        packed_gate,
        scale,
        tokens,
    )
    torch.cuda.synchronize()
    split_actual = split_actual.cpu().numpy()
    split_mismatches = int(
        np.count_nonzero(split_actual.view(np.uint16) != expected.view(np.uint16))
    )
    if split_mismatches:
        raise RuntimeError(f"packed split attention mismatch: {split_mismatches}")
    print(
        json.dumps(
            {
                "schema": "muser.spark-exact-attention-fixture.v1",
                "device": torch.cuda.get_device_name(),
                "elements": actual.size,
                "mismatches": mismatches,
                "packed_split_stride": [4608, 1],
                "packed_split_mismatches": split_mismatches,
                "deterministic_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
