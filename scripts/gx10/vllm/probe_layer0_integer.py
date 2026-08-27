#!/usr/bin/env python3
"""Recompute one complete Muse layer with structural exact kernels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from safetensors.torch import load_file

from muser_vllm.exact_attention import exact_attention_gate
from muser_vllm.exact_fp4_mm import exact_fp4_a16_mm, exact_fp4_mm
from muser_vllm.exact_fp4_quant import exact_scaled_fp4_quant
from muser_vllm.exact_rms_norm import exact_rms_norm, exact_split_rms_norm
from muser_vllm.exact_rope import exact_rope
from muser_vllm.exact_swiglu import exact_swiglu
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    pad_nvfp4_weight_for_cutlass,
    swizzle_blockscale,
)


PREFIX = "model.language_model"
HIDDEN = 6656
ATTN = 4096
KV = 256
HEAD_DIM = 128
INTERMEDIATE = 19968


class StageMismatch(RuntimeError):
    pass


def neox_to_interleaved(values: torch.Tensor, heads: int) -> torch.Tensor:
    return (
        values.reshape(-1, heads, 2, HEAD_DIM // 2)
        .transpose(-2, -1)
        .reshape(values.shape)
    )


def interleaved_to_neox(values: torch.Tensor, heads: int) -> torch.Tensor:
    return (
        values.reshape(-1, heads, HEAD_DIM // 2, 2)
        .transpose(-2, -1)
        .reshape(values.shape)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-cache", required=True, type=Path)
    parser.add_argument("--tokens", required=True, type=Path)
    parser.add_argument("--mac-capture", required=True, type=Path)
    parser.add_argument("--query-scale", type=float, default=3.87)
    parser.add_argument("--eps", type=float, default=1.0e-5)
    args = parser.parse_args()

    started = time.perf_counter()
    tensors = load_file(args.tensor_cache, device="cpu")
    token_ids = [int(value) for value in args.tokens.read_text().split()]
    rows = np.fromfile(args.mac_capture / "attn_norm-0.f32", dtype="<f4").size // HIDDEN
    if rows <= 0 or len(token_ids) < rows:
        raise RuntimeError("token fixture does not cover the retained Mac rows")
    embedding = tensors["__fixture__.embedding"][:rows].to(
        dtype=torch.float16, device="cuda"
    )
    print(
        json.dumps(
            {"event": "fixture_load", "rows": rows, "seconds": time.perf_counter() - started}
        ),
        flush=True,
    )
    layer = f"{PREFIX}.layers.0"
    reports: list[dict[str, object]] = []

    def compare(name: str, actual: torch.Tensor, width: int) -> None:
        torch.cuda.synchronize()
        actual_array = actual.detach().to(torch.float16).cpu().numpy().reshape(rows, width)
        expected = (
            np.fromfile(args.mac_capture / f"{name}.f32", dtype="<f4")
            .reshape(rows, width)
            .astype(np.float16)
        )
        mismatch = actual_array.view(np.uint16) != expected.view(np.uint16)
        locations = np.argwhere(mismatch)
        report: dict[str, object] = {
            "name": name,
            "elements": int(actual_array.size),
            "mismatches": int(mismatch.sum()),
            "max_abs": float(
                np.max(np.abs(actual_array.astype(np.float32) - expected.astype(np.float32)))
            ),
            "first_mismatch": None,
        }
        if locations.size:
            token, element = (int(value) for value in locations[0])
            report["first_mismatch"] = {
                "token": token,
                "element": element,
                "actual_bits": int(actual_array.view(np.uint16)[token, element]),
                "expected_bits": int(expected.view(np.uint16)[token, element]),
            }
        reports.append(report)
        print(json.dumps({"event": "stage", **report}, sort_keys=True), flush=True)
        if report["mismatches"]:
            raise StageMismatch(name)

    def linear(values: torch.Tensor, suffix: str) -> torch.Tensor:
        base = f"{layer}.{suffix}"
        begin = time.perf_counter()
        packed = tensors[base + ".weight_packed"].contiguous().cuda()
        weight_global = tensors[base + ".weight_global_scale"].float().cuda()
        input_scale_key = base + ".input_global_scale"
        if input_scale_key in tensors:
            scales = swizzle_blockscale(tensors[base + ".weight_scale"].cuda())
            packed, padding = pad_nvfp4_weight_for_cutlass(packed)
            if padding:
                raise RuntimeError(f"unexpected exact-linear padding for {suffix}: {padding}")
            input_global = tensors[input_scale_key].float().cuda()
            activation, activation_scale = exact_scaled_fp4_quant(values, input_global)
            result = exact_fp4_mm(
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
            scales = tensors[base + ".weight_scale"].contiguous().cuda()
            result = exact_fp4_a16_mm(
                values,
                packed,
                scales,
                torch.reciprocal(weight_global),
            )
            activation_precision = "f16-weight-only-q8k-exact"
        torch.cuda.synchronize()
        print(
            json.dumps(
                {
                    "event": "integer_linear",
                    "linear": suffix,
                    "seconds": time.perf_counter() - begin,
                    "activation_precision": activation_precision,
                }
            ),
            flush=True,
        )
        return result

    try:
        empty = torch.empty(0, dtype=torch.float16, device="cuda")
        residual = exact_rms_norm(embedding, empty, args.eps, False, 0)
        input_weight = tensors[f"{layer}.input_layernorm.weight"].half().cuda()
        attn_norm = exact_rms_norm(residual, input_weight, args.eps, True, 1)
        compare("attn_norm-0", attn_norm, HIDDEN)

        # The native GGUF converter un-permutes HF's half-split Q/K rows into
        # adjacent RoPE pairs. Perform that fixed permutation before QK norm:
        # its floating reduction tree observes element order even though the
        # mathematical sum of squares is permutation-invariant.
        q = neox_to_interleaved(linear(attn_norm, "self_attn.q_proj"), 32)
        k = neox_to_interleaved(linear(attn_norm, "self_attn.k_proj"), 2)
        v = linear(attn_norm, "self_attn.v_proj")
        compare("Qcur-0", q, ATTN)
        compare("Kcur-0", k, KV)
        compare("Vcur-0", v, KV)

        q_scale = torch.full((HEAD_DIM,), args.query_scale, device="cuda")
        k_scale = torch.ones((HEAD_DIM,), device="cuda")
        q = exact_split_rms_norm(
            q.reshape(-1, HEAD_DIM), q_scale, args.eps, 0
        ).reshape(rows, ATTN)
        k = exact_split_rms_norm(
            k.reshape(-1, HEAD_DIM), k_scale, args.eps, 0
        ).reshape(rows, KV)
        compare("Qcur_normed-0", q, ATTN)
        compare("Kcur_normed-0", k, KV)

        positions = torch.arange(rows, device="cuda", dtype=torch.long)
        source_cache_shape_only = torch.empty(
            (rows, HEAD_DIM), device="cuda", dtype=torch.float16
        )
        q_neox, k_neox = exact_rope(
            interleaved_to_neox(q, 32),
            interleaved_to_neox(k, 2),
            source_cache_shape_only,
            positions,
            HEAD_DIM,
        )
        q = neox_to_interleaved(q_neox, 32)
        k = neox_to_interleaved(k_neox, 2)
        compare("Qcur_rope-0", q, ATTN)
        compare("Kcur_rope-0", k, KV)

        gate = linear(attn_norm, "self_attn.gate_proj")
        compare("attn_gate_proj-0", gate, ATTN)
        attention = exact_attention_gate(q, k, v, gate, 1.0 / 128.0**0.5, rows)
        compare("attn_gated-0", attention, ATTN)
        projected = linear(attention, "self_attn.o_proj")
        compare("attn_o_proj-0", projected, HIDDEN)

        post_weight = tensors[f"{layer}.post_attention_layernorm.weight"].half().cuda()
        post = exact_split_rms_norm(projected, post_weight, 1.0e-8, 1)
        ffn_input = (residual + post).to(torch.float16)
        compare("ffn_inp-0", ffn_input, HIDDEN)
        pre_weight = tensors[f"{layer}.pre_feedforward_layernorm.weight"].half().cuda()
        ffn_norm = exact_split_rms_norm(ffn_input, pre_weight, args.eps, 1)
        compare("ffn_norm-0", ffn_norm, HIDDEN)
        ffn_gate = linear(ffn_norm, "mlp.gate_proj")
        ffn_up = linear(ffn_norm, "mlp.up_proj")
        compare("ffn_gate-0", ffn_gate, INTERMEDIATE)
        compare("ffn_up-0", ffn_up, INTERMEDIATE)
        ffn_swiglu = exact_swiglu(torch.cat((ffn_gate, ffn_up), dim=-1))
        compare("ffn_swiglu-0", ffn_swiglu, INTERMEDIATE)
        ffn_out = linear(ffn_swiglu, "mlp.down_proj")
        compare("ffn_out-0", ffn_out, HIDDEN)
        post_ffn_weight = tensors[f"{layer}.post_feedforward_layernorm.weight"].half().cuda()
        post_ffn = exact_split_rms_norm(ffn_out, post_ffn_weight, 1.0e-8, 1)
        layer_out = (ffn_input + post_ffn).to(torch.float16)
        compare("l_out-0", layer_out, HIDDEN)
    except StageMismatch as error:
        print(
            json.dumps(
                {
                    "schema": "muser.cross-vendor-integer-layer0.v2",
                    "bit_exact": False,
                    "stopped_after": str(error),
                    "reports": reports,
                    "seal_eligible": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        raise SystemExit(1)

    print(
        json.dumps(
            {
                "schema": "muser.cross-vendor-integer-layer0.v2",
                "bit_exact": True,
                "reports": reports,
                "seal_eligible": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
