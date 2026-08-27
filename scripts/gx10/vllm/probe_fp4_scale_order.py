#!/usr/bin/env python3
"""Diagnose the final scaling/reduction order of sparse real NVFP4 GEMM ties."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


PREFIX = "model.language_model.layers.0"
E2M1 = np.asarray(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)


def f32(value: float) -> np.float32:
    return np.float32(value)


def add(left: float, right: float) -> np.float32:
    return np.add(f32(left), f32(right), dtype=np.float32)


def mul(left: float, right: float) -> np.float32:
    return np.multiply(f32(left), f32(right), dtype=np.float32)


def div(numerator: float, denominator: float) -> np.float32:
    return np.divide(f32(numerator), f32(denominator), dtype=np.float32)


def e4m3fn(code: int) -> np.float32:
    sign = -1.0 if code & 0x80 else 1.0
    exponent = (code >> 3) & 0x0F
    mantissa = code & 0x07
    if exponent == 0:
        magnitude = f32(mantissa) * f32(1.0 / 512.0)
    elif exponent == 0x0F and mantissa == 0x07:
        return f32(np.nan)
    else:
        magnitude = f32(1.0 + mantissa * 0.125) * f32(2.0 ** (exponent - 7))
    return mul(sign, magnitude)


POSITIVE_E4 = np.asarray([e4m3fn(code) for code in range(0x7F)], dtype=np.float32)


def round_e4_positive(value: np.float32) -> int:
    value = min(f32(value), f32(448.0))
    distances = np.abs(POSITIVE_E4 - value, dtype=np.float32)
    minimum = distances.min()
    choices = np.flatnonzero(distances == minimum)
    for choice in choices:
        if int(choice) % 2 == 0:
            return int(choice)
    return int(choices[0])


def activation_group(values: np.ndarray, global_scale: np.float32) -> tuple[np.ndarray, int]:
    values = values.astype(np.float16).astype(np.float32)
    abs_max = np.max(np.abs(values, dtype=np.float32))
    scale_code = round_e4_positive(mul(mul(abs_max, 1.0 / 6.0), global_scale))
    scale = e4m3fn(scale_code)
    codes = np.empty(16, dtype=np.uint8)
    for index, value in enumerate(values):
        sign = 8 if np.signbit(value) else 0
        magnitude = mul(abs(value), global_scale)
        code = (
            0 if magnitude <= mul(scale, 0.25) else
            1 if magnitude < mul(scale, 0.75) else
            2 if magnitude <= mul(scale, 1.25) else
            3 if magnitude < mul(scale, 1.75) else
            4 if magnitude <= mul(scale, 2.5) else
            5 if magnitude < mul(scale, 3.5) else
            6 if magnitude <= mul(scale, 5.0) else 7
        )
        codes[index] = sign | code
    return codes, scale_code


def sequential(values: np.ndarray) -> np.float32:
    total = f32(0.0)
    for value in values:
        total = add(total, value)
    return total


def pairwise(values: np.ndarray) -> np.float32:
    work = [f32(value) for value in values]
    while len(work) > 1:
        paired = [add(work[index], work[index + 1]) for index in range(0, len(work) - 1, 2)]
        if len(work) % 2:
            paired.append(work[-1])
        work = paired
    return work[0]


def stride32(values: np.ndarray) -> np.float32:
    lanes = [sequential(values[lane::32]) for lane in range(32)]
    return pairwise(np.asarray(lanes, dtype=np.float32))


def reduce_chunks(values: np.ndarray, width: int, inner_pairwise: bool) -> np.float32:
    reducer = pairwise if inner_pairwise else sequential
    partials = [reducer(values[index : index + width]) for index in range(0, len(values), width)]
    return sequential(np.asarray(partials, dtype=np.float32))


def reduce_strided(values: np.ndarray, lanes: int, tree: bool) -> np.float32:
    partials = np.asarray(
        [sequential(values[lane::lanes]) for lane in range(lanes)], dtype=np.float32
    )
    return pairwise(partials) if tree else sequential(partials)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--mac-capture", required=True, type=Path)
    parser.add_argument("--mac-layer1", required=True, type=Path)
    args = parser.parse_args()
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]

    cells = [
        {
            "name": "attn_o_proj",
            "base": f"{PREFIX}.self_attn.o_proj",
            "input": "attn_gated-0.f32",
            "expected": "attn_o_proj-0.f32",
            "token": 12,
            "row": 301,
            "producer": -0.308349609375,
            "capture": args.mac_capture,
        },
        {
            "name": "ffn_out",
            "base": f"{PREFIX}.mlp.down_proj",
            "input": "ffn_swiglu-0.f32",
            "expected": "ffn_out-0.f32",
            "token": 19,
            "row": 326,
            "producer": 0.796875,
            "capture": args.mac_capture,
        },
        {
            "name": "layer1_k_proj",
            "base": "model.language_model.layers.1.self_attn.k_proj",
            "input": "attn_norm-0.f32",
            "expected": "Kcur-0.f32",
            "token": 7,
            "row": 66,
            "expected_element": 132,
            "producer": -0.19873046875,
            "capture": args.mac_layer1,
        },
    ]
    reports = []
    for cell in cells:
        base = str(cell["base"])

        def selected(key: str, row: int | None = None) -> torch.Tensor:
            path = args.checkpoint / weight_map[key]
            with safe_open(path, framework="pt", device="cpu") as handle:
                if row is None:
                    return handle.get_tensor(key)
                return handle.get_slice(key)[row : row + 1]

        packed = selected(base + ".weight_packed", int(cell["row"])).numpy().reshape(-1)
        scales_tensor = selected(base + ".weight_scale", int(cell["row"]))
        scales = scales_tensor.view(torch.uint8).numpy().reshape(-1)
        input_global = f32(float(selected(base + ".input_global_scale").reshape(-1)[0]))
        weight_global = f32(float(selected(base + ".weight_global_scale").reshape(-1)[0]))
        source_values = np.fromfile(
            Path(cell["capture"]) / str(cell["input"]), dtype="<f4"
        ).reshape(32, -1)[int(cell["token"])]
        expected_element = int(cell.get("expected_element", cell["row"]))
        expected = np.fromfile(
            Path(cell["capture"]) / str(cell["expected"]), dtype="<f4"
        ).reshape(32, -1)[int(cell["token"]), expected_element]
        if source_values.size // 16 != scales.size or source_values.size // 2 != packed.size:
            raise RuntimeError(f"{cell['name']} geometry is inconsistent")
        contributions = np.empty(scales.size, dtype=np.float32)
        for group in range(scales.size):
            activation_codes, activation_scale = activation_group(
                source_values[group * 16 : (group + 1) * 16], input_global
            )
            block_sum = f32(0.0)
            for element in range(16):
                byte = int(packed[group * 8 + element // 2])
                weight_code = byte & 0x0F if element % 2 == 0 else byte >> 4
                block_sum = add(
                    block_sum,
                    mul(E2M1[weight_code], E2M1[int(activation_codes[element])]),
                )
            contributions[group] = mul(
                block_sum, mul(e4m3fn(int(scales[group])), e4m3fn(activation_scale))
            )
        reductions = {
            "sequential": sequential(contributions),
            "pairwise": pairwise(contributions),
            "stride32-pairwise": stride32(contributions),
        }
        for width in (2, 4, 8, 16, 32, 64, 128, 256):
            reductions[f"chunk{width}-sequential"] = reduce_chunks(
                contributions, width, False
            )
            reductions[f"chunk{width}-pairwise"] = reduce_chunks(
                contributions, width, True
            )
        for lanes in (2, 4, 8, 16, 32, 64):
            reductions[f"stride{lanes}-sequential"] = reduce_strided(
                contributions, lanes, False
            )
            reductions[f"stride{lanes}-pairwise"] = reduce_strided(
                contributions, lanes, True
            )
        candidates = {}
        for reduction_name, total in reductions.items():
            scale2 = div(1.0, weight_global)
            input_inverse = div(1.0, input_global)
            alpha_product_f32 = div(1.0, mul(input_global, weight_global))
            alpha_python_f64 = f32(1.0 / (float(input_global) * float(weight_global)))
            variants = {
                "two-step": mul(mul(total, scale2), input_inverse),
                "combined-f32": mul(total, alpha_product_f32),
                "combined-f64": mul(total, alpha_python_f64),
                "scale2-times-input-inverse": mul(total, mul(scale2, input_inverse)),
            }
            for scale_name, value in variants.items():
                rounded = np.float16(value)
                candidates[f"{reduction_name}/{scale_name}"] = {
                    "f32": float(value),
                    "f16": float(rounded),
                    "bits": int(rounded.view(np.uint16)),
                }
        expected_bits = int(np.float16(expected).view(np.uint16))
        producer_bits = int(np.float16(cell["producer"]).view(np.uint16))
        reports.append(
            {
                "name": cell["name"],
                "token": cell["token"],
                "row": cell["row"],
                "expected_element": expected_element,
                "input_global": float(input_global),
                "weight_global": float(weight_global),
                "expected": float(np.float16(expected)),
                "expected_bits": expected_bits,
                "producer": cell["producer"],
                "producer_bits": producer_bits,
                "producer_matches": sorted(
                    name for name, value in candidates.items() if value["bits"] == producer_bits
                ),
                "expected_matches": sorted(
                    name for name, value in candidates.items() if value["bits"] == expected_bits
                ),
                "candidate_bit_histogram": {
                    str(bits): sum(1 for value in candidates.values() if value["bits"] == bits)
                    for bits in sorted({value["bits"] for value in candidates.values()})
                },
            }
        )
    print(json.dumps({"schema": "muser.spark-fp4-scale-order-probe.v1", "reports": reports}, sort_keys=True))


if __name__ == "__main__":
    main()
