#!/usr/bin/env python3
"""GB10 fixture for the pinned cross-vendor Muse SwiGLU kernel."""

from __future__ import annotations

import ctypes
import hashlib
import json
import struct

import numpy as np
import torch

from muser_vllm.exact_swiglu import exact_swiglu


_LIBC = ctypes.CDLL(None)
_LIBC.fmaf.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
_LIBC.fmaf.restype = ctypes.c_float


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f32(value)))[0]


def from_bits(value: int) -> float:
    return struct.unpack("<f", struct.pack("<I", value & 0xFFFFFFFF))[0]


def fma(left: float, right: float, addend: float) -> float:
    return float(_LIBC.fmaf(f32(left), f32(right), f32(addend)))


def add(left: float, right: float) -> float:
    return f32(f32(left) + f32(right))


def mul(left: float, right: float) -> float:
    return f32(f32(left) * f32(right))


def div(numerator: float, denominator: float) -> float:
    return f32(f32(numerator) / f32(denominator))


def neon_exp(value: float) -> float:
    value = f32(value)
    rounding = f32(12582912.0)
    z = fma(value, 1.4426950216293335, rounding)
    n = add(z, -rounding)
    reduced = fma(-n, 0.693145751953125, value)
    reduced = fma(-n, 1.428606765330187e-6, reduced)
    exponent = (bits(z) << 23) & 0xFFFFFFFF
    scale = from_bits(exponent + 0x3F800000)
    squared = mul(reduced, reduced)
    t1 = fma(0.008247390389442444, reduced, 0.04189976677298546)
    t2 = fma(0.16668395698070526, reduced, 0.4999912679195404)
    t2 = fma(t1, squared, t2)
    polynomial = fma(t2, squared, mul(0.9999994039535522, reduced))
    if abs(n) <= 126.0:
        return fma(polynomial, scale, scale)
    delta = 0x82000000 if n <= 0.0 else 0
    scale1 = from_bits(delta + 0x7F000000)
    scale2 = from_bits(exponent - delta)
    if abs(n) > 192.0:
        return mul(scale1, scale1)
    return mul(fma(scale2, polynomial, scale2), scale1)


def reference(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    output = np.empty_like(gate, dtype=np.float16)
    for index in range(gate.size):
        gate_value = float(gate.flat[index])
        silu = div(gate_value, add(1.0, neon_exp(-gate_value)))
        output.flat[index] = np.float16(mul(silu, float(up.flat[index])))
    return output


def main() -> None:
    rows, width = 3, 19968
    gate = np.asarray(
        [((index * 47 % 1009) - 503.0) * 0.00390625 for index in range(rows * width)],
        dtype=np.float32,
    ).astype(np.float16).reshape(rows, width)
    up = np.asarray(
        [((index * 53 % 1013) - 506.0) * 0.001953125 for index in range(rows * width)],
        dtype=np.float32,
    ).astype(np.float16).reshape(rows, width)
    expected = reference(gate, up)
    packed = np.concatenate((gate, up), axis=-1)
    actual = exact_swiglu(torch.from_numpy(packed).cuda())
    torch.cuda.synchronize()
    actual = actual.cpu().numpy()
    mismatches = int(np.count_nonzero(actual.view(np.uint16) != expected.view(np.uint16)))
    if mismatches:
        raise RuntimeError(f"exact SwiGLU mismatch: {mismatches}")
    print(
        json.dumps(
            {
                "schema": "muser.spark-exact-swiglu-fixture.v1",
                "device": torch.cuda.get_device_name(),
                "elements": actual.size,
                "mismatches": mismatches,
                "deterministic_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
