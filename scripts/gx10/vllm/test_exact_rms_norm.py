#!/usr/bin/env python3
"""GB10 fixture for the pinned cross-vendor Muse RMSNorm kernel."""

from __future__ import annotations

import argparse
import json
import math
import struct
import hashlib
import time

import numpy as np
import torch

from muser_vllm.exact_rms_norm import exact_rms_norm, exact_split_rms_norm


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def reference(
    values: np.ndarray,
    weight: np.ndarray | None,
    eps: float,
    weight_offset: int,
    *,
    split_scale: bool = False,
) -> np.ndarray:
    rows, width = values.shape
    output = np.empty_like(values, dtype=np.float16)
    for row in range(rows):
        partial = [0.0] * 32
        for lane in range(32):
            total = 0.0
            for index in range(lane, width, 32):
                value = f32(float(values[row, index]))
                total = f32(f32(value * value) + total)
            partial[lane] = total
        offset = 16
        while offset:
            for lane in range(offset):
                partial[lane] = f32(partial[lane] + partial[lane + offset])
            offset //= 2
        mean = f32(f32(partial[0] * f32(1.0 / width)) + f32(eps))
        inverse = f32(1.0 / f32(math.sqrt(mean)))
        for index in range(width):
            normalized = f32(f32(float(values[row, index])) * inverse)
            if weight is not None:
                if split_scale:
                    normalized = f32(float(np.float16(normalized)))
                multiplier = f32(f32(float(weight[index])) + f32(weight_offset))
                normalized = f32(normalized * multiplier)
            output[row, index] = np.float16(normalized)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--widths", default="128,6656")
    args = parser.parse_args()
    widths = [int(value) for value in args.widths.split(",")]
    if not widths or any(width not in (128, 6656) for width in widths):
        raise RuntimeError("RMS fixture widths must be a non-empty subset of 128,6656")
    deterministic_hashes = {"fused": {}, "split": {}}
    for width in widths:
        started = time.perf_counter()
        print(json.dumps({"event": "width_start", "width": width}), flush=True)
        deterministic_values = np.asarray(
            [((index * 37 % 1009) - 503.0) * 0.00390625 for index in range(5 * width)],
            dtype=np.float32,
        ).astype(np.float16).reshape(5, width)
        deterministic_weight = np.asarray(
            [((index * 19 % 257) - 128.0) * 0.00024414063 for index in range(width)],
            dtype=np.float32,
        ).astype(np.float16)
        expected = reference(
            deterministic_values, deterministic_weight, 1.0e-5, 1
        )
        actual = exact_rms_norm(
            torch.from_numpy(deterministic_values).cuda(),
            torch.from_numpy(deterministic_weight).cuda(),
            1.0e-5,
            True,
            1,
        )
        torch.cuda.synchronize()
        actual = actual.cpu().numpy()
        if np.any(actual.view(np.uint16) != expected.view(np.uint16)):
            raise RuntimeError(f"deterministic exact RMSNorm mismatch at width {width}")
        deterministic_hashes["fused"][str(width)] = hashlib.sha256(
            actual.tobytes()
        ).hexdigest()
        split_expected = reference(
            deterministic_values,
            deterministic_weight,
            1.0e-5,
            1,
            split_scale=True,
        )
        split_actual = exact_split_rms_norm(
            torch.from_numpy(deterministic_values).cuda(),
            torch.from_numpy(deterministic_weight).cuda(),
            1.0e-5,
            1,
        )
        torch.cuda.synchronize()
        split_actual = split_actual.cpu().numpy()
        split_mismatches = int(
            np.count_nonzero(
                split_actual.view(np.uint16) != split_expected.view(np.uint16)
            )
        )
        if split_mismatches:
            raise RuntimeError(
                f"deterministic split RMSNorm mismatch at width {width}: "
                f"{split_mismatches}"
            )
        deterministic_hashes["split"][str(width)] = hashlib.sha256(
            split_actual.tobytes()
        ).hexdigest()
        print(
            json.dumps(
                {
                    "event": "width_complete",
                    "width": width,
                    "seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
    print(
        json.dumps(
            {
                "schema": "muser.spark-exact-rms-norm-fixture.v2",
                "device": torch.cuda.get_device_name(),
                "widths": widths,
                "mismatches": 0,
                "deterministic_sha256": deterministic_hashes,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
