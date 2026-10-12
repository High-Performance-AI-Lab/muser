"""Reference NVFP4 decoding shared by P-series artifact tools."""

from __future__ import annotations

import numpy as np


E2M1 = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def e4m3fn_table() -> np.ndarray:
    """Return the raw-byte to IEEE-f32 E4M3FN lookup table."""
    values = np.zeros(256, dtype=np.float32)
    for byte in range(256):
        sign = -1.0 if byte & 0x80 else 1.0
        exponent = (byte >> 3) & 0x0F
        mantissa = byte & 0x07
        if exponent == 0:
            value = (mantissa / 8.0) * 2.0**-6
        elif exponent == 0x0F and mantissa == 0x07:
            value = np.nan
        else:
            value = (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
        values[byte] = sign * value
    return values


E4M3FN = e4m3fn_table()


def compressed_tensors_scale2(weight_global_scale: float) -> np.float32:
    """Convert compressed-tensors' inverse global scale to ModelOpt scale2."""
    value = np.float32(weight_global_scale)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("NVFP4 weight_global_scale must be finite and positive")
    return np.float32(1.0) / value


def dequantize(
    packed: np.ndarray, block_scale: np.ndarray, tensor_scale: float
) -> np.ndarray:
    """Decode E2M1 pairs with per-16 E4M3FN and a multiplicative scale2."""
    if packed.ndim != 2 or block_scale.ndim != 2:
        raise ValueError("NVFP4 packed weights and block scales must be matrices")
    rows, packed_columns = packed.shape
    columns = packed_columns * 2
    if columns % 16 or block_scale.shape != (rows, columns // 16):
        raise ValueError("NVFP4 block-scale geometry does not match group size 16")
    low = E2M1[np.asarray(packed, dtype=np.uint8) & 0x0F]
    high = E2M1[np.asarray(packed, dtype=np.uint8) >> 4]
    values = np.empty((rows, columns), dtype=np.float32)
    values[:, 0::2] = low
    values[:, 1::2] = high
    scales = E4M3FN[np.asarray(block_scale, dtype=np.uint8)]
    if not np.isfinite(scales).all():
        raise ValueError("NVFP4 block scales contain E4M3FN NaN encodings")
    grouped = values.reshape(rows, columns // 16, 16)
    grouped *= scales[:, :, None]
    values = grouped.reshape(rows, columns)
    values *= np.float32(tensor_scale)
    return values
