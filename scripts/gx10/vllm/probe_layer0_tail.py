#!/usr/bin/env python3
"""Isolate producer layer-0 tail operations against retained Mac boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from muser_vllm.exact_attention import exact_attention_gate
from muser_vllm.exact_fp4_quant import exact_scaled_fp4_quant
from muser_vllm.exact_fp4_mm import exact_fp4_mm
from muser_vllm.exact_rms_norm import exact_rms_norm, exact_split_rms_norm
from muser_vllm.exact_swiglu import exact_swiglu
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    pad_nvfp4_weight_for_cutlass,
    swizzle_blockscale,
)
from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm


PREFIX = "model.language_model"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tokens", required=True, type=Path)
    parser.add_argument("--mac-capture", required=True, type=Path)
    parser.add_argument("--live-capture", type=Path)
    parser.add_argument("--tensor-cache", type=Path)
    parser.add_argument(
        "--backend",
        default="cutlass",
        choices=("cutlass", "cudnn", "exact", "exact-output"),
    )
    args = parser.parse_args()
    cached = load_file(args.tensor_cache, device="cpu") if args.tensor_cache else None
    if cached is None:
        index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())
        weight_map = index["weight_map"]
    else:
        weight_map = {}

    def tensor(key: str) -> torch.Tensor:
        if cached is not None:
            return cached[key]
        with safe_open(
            args.checkpoint / weight_map[key], framework="pt", device="cpu"
        ) as handle:
            return handle.get_tensor(key)

    def linear(values: torch.Tensor, base: str) -> torch.Tensor:
        started = time.perf_counter()
        packed = tensor(base + ".weight_packed").cuda()
        scales = swizzle_blockscale(tensor(base + ".weight_scale").cuda())
        packed, _ = pad_nvfp4_weight_for_cutlass(packed)
        input_global = float(tensor(base + ".input_global_scale"))
        weight_global = float(tensor(base + ".weight_global_scale"))
        activation, activation_scale = exact_scaled_fp4_quant(
            values, torch.tensor(input_global, dtype=torch.float32, device="cuda")
        )
        alpha = torch.tensor(
            1.0 / (input_global * weight_global),
            dtype=torch.float32,
            device="cuda",
        )
        if args.backend == "exact" or (
            args.backend == "exact-output" and packed.shape[0] == 6656
        ):
            diagnostics: dict[str, object] = {}
            result = exact_fp4_mm(
                activation,
                packed,
                activation_scale,
                scales,
                alpha,
                torch.tensor(1.0 / weight_global, dtype=torch.float32, device="cuda"),
                torch.tensor(1.0 / input_global, dtype=torch.float32, device="cuda"),
                diagnostics,
            )
            selected_backend = "exact"
        else:
            selected_backend = (
                "cutlass" if args.backend == "exact-output" else args.backend
            )
            diagnostics = {}
            result = flashinfer_scaled_fp4_mm(
                activation,
                packed,
                activation_scale,
                scales,
                alpha,
                torch.float16,
                backend=selected_backend,
            )
        torch.cuda.synchronize()
        print(
            json.dumps(
                {
                    "event": "linear_timing",
                    "base": base,
                    "backend": selected_backend,
                    "seconds": time.perf_counter() - started,
                    **diagnostics,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return result

    reports: list[dict[str, object]] = []

    def reference(name: str, width: int) -> torch.Tensor:
        values = np.fromfile(args.mac_capture / f"{name}.f32", dtype="<f4")
        if values.size % width:
            raise RuntimeError(f"Mac capture {name} has invalid width")
        return torch.from_numpy(values.reshape(-1, width).astype(np.float16)).cuda()

    def compare(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
        torch.cuda.synchronize()
        actual = actual.detach().to(torch.float16).cpu().numpy()
        expected = expected.detach().to(torch.float16).cpu().numpy()
        mismatches = int(
            np.count_nonzero(actual.view(np.uint16) != expected.view(np.uint16))
        )
        mismatch = actual.view(np.uint16) != expected.view(np.uint16)
        locations = np.argwhere(mismatch)
        first = None
        if locations.size:
            token, element = (int(value) for value in locations[0])
            first = {
                "token": token,
                "element": element,
                "actual": float(actual[token, element]),
                "actual_bits": int(actual.view(np.uint16)[token, element]),
                "expected": float(expected[token, element]),
                "expected_bits": int(expected.view(np.uint16)[token, element]),
            }
        reports.append(
            {
                "name": name,
                "elements": actual.size,
                "mismatches": mismatches,
                "mismatches_by_token": [
                    int(value)
                    for value in np.count_nonzero(
                        actual.view(np.uint16) != expected.view(np.uint16), axis=1
                    )
                ],
                "max_abs": float(
                    np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32)))
                ),
                "first_mismatch": first,
            }
        )

    def live(name: str, width: int) -> torch.Tensor:
        if args.live_capture is None:
            raise RuntimeError("live capture was not configured")
        values = np.fromfile(args.live_capture / f"{name}.f16", dtype="<f2")
        if values.size % width:
            raise RuntimeError(f"live capture {name} has invalid width")
        return torch.from_numpy(values.reshape(-1, width)).cuda()

    hidden = 6656
    attn = 4096
    kv = 256
    intermediate = 19968
    layer = f"{PREFIX}.layers.0"
    attn_norm = reference("attn_norm-0", hidden)
    gate = (
        live("attn_gate", attn)
        if args.live_capture is not None
        else linear(attn_norm, f"{layer}.self_attn.gate_proj")
    )
    compare("attn_gate_proj", gate, reference("attn_gate_proj-0", attn))

    value = live("attn_v", kv) if args.live_capture is not None else reference("Vcur-0", kv)
    query = live("attn_q", attn) if args.live_capture is not None else reference("Qcur_rope-0", attn)
    key = live("attn_k", kv) if args.live_capture is not None else reference("Kcur_rope-0", kv)
    gated = exact_attention_gate(
        query, key, value, gate, 1.0 / 128.0**0.5, query.shape[0]
    )
    compare("attn_gated", gated, reference("attn_gated-0", attn))

    projected = linear(gated, f"{layer}.self_attn.o_proj")
    compare("attn_o_proj", projected, reference("attn_o_proj-0", hidden))

    token_ids = [
        int(line)
        for line in args.tokens.read_text().splitlines()
        if line.strip()
    ][: attn_norm.shape[0]]
    if len(token_ids) != attn_norm.shape[0]:
        raise RuntimeError("token fixture is shorter than the Mac capture")
    embedding_key = f"{PREFIX}.embed_tokens.weight"
    if cached is not None:
        embedding = cached["__fixture__.embedding"][: len(token_ids)]
    else:
        with safe_open(
            args.checkpoint / weight_map[embedding_key], framework="pt", device="cpu"
        ) as handle:
            embedding_slice = handle.get_slice(embedding_key)
            embedding = torch.cat(
                [embedding_slice[token_id : token_id + 1] for token_id in token_ids],
                dim=0,
            )
    embedding = embedding.to(dtype=torch.float16, device="cuda")
    residual = exact_rms_norm(embedding, embedding, 1.0e-5, False, 0)
    post_weight = tensor(f"{layer}.post_attention_layernorm.weight").to(
        dtype=torch.float16, device="cuda"
    )
    post = exact_split_rms_norm(projected, post_weight, 1.0e-8, 1)
    ffn_input = (residual + post).to(torch.float16)
    compare("ffn_inp", ffn_input, reference("ffn_inp-0", hidden))

    pre_weight = tensor(f"{layer}.pre_feedforward_layernorm.weight").to(
        dtype=torch.float16, device="cuda"
    )
    ffn_norm = exact_split_rms_norm(ffn_input, pre_weight, 1.0e-5, 1)
    compare("ffn_norm", ffn_norm, reference("ffn_norm-0", hidden))
    ffn_gate = linear(ffn_norm, f"{layer}.mlp.gate_proj")
    ffn_up = linear(ffn_norm, f"{layer}.mlp.up_proj")
    compare("ffn_gate", ffn_gate, reference("ffn_gate-0", intermediate))
    compare("ffn_up", ffn_up, reference("ffn_up-0", intermediate))
    ffn_swiglu = (torch.nn.functional.silu(ffn_gate) * ffn_up).to(torch.float16)
    compare("ffn_swiglu-stock", ffn_swiglu, reference("ffn_swiglu-0", intermediate))
    exact_ffn_swiglu = exact_swiglu(torch.cat((ffn_gate, ffn_up), dim=-1))
    compare(
        "ffn_swiglu-exact",
        exact_ffn_swiglu,
        reference("ffn_swiglu-0", intermediate),
    )
    ffn_out = linear(exact_ffn_swiglu, f"{layer}.mlp.down_proj")
    compare("ffn_out", ffn_out, reference("ffn_out-0", hidden))
    post_ffn_weight = tensor(f"{layer}.post_feedforward_layernorm.weight").to(
        dtype=torch.float16, device="cuda"
    )
    post_ffn = exact_split_rms_norm(ffn_out, post_ffn_weight, 1.0e-8, 1)
    layer_out = (ffn_input + post_ffn).to(torch.float16)
    compare("layer_out", layer_out, reference("l_out-0", hidden))
    print(
        json.dumps(
            {
                "schema": "muser.spark-layer0-tail-probe.v1",
                "backend": args.backend,
                "reports": reports,
                "seal_eligible": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
