#!/usr/bin/env python3
"""Capture the exact producer-side layer-0 QKV boundary for M0 bisection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

from muser_vllm.exact_fp4_quant import exact_scaled_fp4_quant
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    pad_nvfp4_weight_for_cutlass,
    swizzle_blockscale,
)
from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm


PREFIX = "model.language_model"


def write_exclusive(path: Path, payload: bytes) -> str:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tokens", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    tokens = [int(line) for line in args.tokens.read_text().splitlines() if line.strip()][:-1]
    if not tokens:
        parser.error("token fixture has no cached prefix")
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]

    def path_for(key: str) -> Path:
        return args.checkpoint / weight_map[key]

    embedding_key = f"{PREFIX}.embed_tokens.weight"
    with safe_open(path_for(embedding_key), framework="pt", device="cpu") as handle:
        embedding_slice = handle.get_slice(embedding_key)
        embeddings = torch.cat(
            [embedding_slice[token : token + 1] for token in tokens], dim=0
        ).to(dtype=torch.float16, device="cuda")

    norm_key = f"{PREFIX}.layers.0.input_layernorm.weight"
    with safe_open(path_for(norm_key), framework="pt", device="cpu") as handle:
        norm_weight = handle.get_tensor(norm_key).to(dtype=torch.float16, device="cuda")

    epsilon = 1.0e-5
    embedding_f32 = embeddings.float()
    embedded = (
        embedding_f32
        * torch.rsqrt(embedding_f32.pow(2).mean(-1, keepdim=True) + epsilon)
    ).to(torch.float16)
    embedded_f32 = embedded.float()
    attn_norm = (
        embedded_f32
        * torch.rsqrt(embedded_f32.pow(2).mean(-1, keepdim=True) + epsilon)
        * (norm_weight.float() + 1.0)
    ).to(torch.float16)

    packed_parts = []
    scale_parts = []
    input_scales = []
    weight_scales = []
    for projection in ("q", "k", "v"):
        base = f"{PREFIX}.layers.0.self_attn.{projection}_proj"
        with safe_open(path_for(base + ".weight_packed"), framework="pt", device="cpu") as handle:
            packed_parts.append(handle.get_tensor(base + ".weight_packed"))
            scale_parts.append(handle.get_tensor(base + ".weight_scale"))
            input_scales.append(float(handle.get_tensor(base + ".input_global_scale")))
            weight_scales.append(float(handle.get_tensor(base + ".weight_global_scale")))
    if len(set(input_scales)) != 1 or len(set(weight_scales)) != 1:
        raise RuntimeError("layer-0 fused QKV does not share global scales")
    weight_packed = torch.cat(packed_parts, dim=0).cuda()
    weight_scale = swizzle_blockscale(torch.cat(scale_parts, dim=0).cuda())
    weight_packed, padding_bytes = pad_nvfp4_weight_for_cutlass(weight_packed)
    if padding_bytes != 0:
        raise RuntimeError(f"unexpected layer-0 QKV weight padding: {padding_bytes}")
    input_scale_inv = torch.tensor(input_scales[0], dtype=torch.float32, device="cuda")
    activation_packed, activation_scale = exact_scaled_fp4_quant(
        attn_norm, input_scale_inv
    )
    alpha = torch.tensor(
        1.0 / (input_scales[0] * weight_scales[0]),
        dtype=torch.float32,
        device="cuda",
    )
    qkv = flashinfer_scaled_fp4_mm(
        activation_packed,
        weight_packed,
        activation_scale,
        weight_scale,
        alpha,
        torch.float16,
        backend="cutlass",
    )
    torch.cuda.synchronize()

    outputs = {
        "attn_norm.f16": tensor_bytes(attn_norm),
        "activation.packed": tensor_bytes(activation_packed),
        "activation.scales-swizzled": tensor_bytes(activation_scale),
        "qkv.f16": tensor_bytes(qkv),
    }
    receipt = {
        "schema": "muser.spark-layer0-qkv-capture.v1",
        "tokens": len(tokens),
        "hidden": attn_norm.shape[-1],
        "qkv": qkv.shape[-1],
        "input_global_scale": input_scales[0],
        "weight_global_scale": weight_scales[0],
        "files": {
            name: {
                "bytes": len(payload),
                "sha256": write_exclusive(args.output_dir / name, payload),
            }
            for name, payload in outputs.items()
        },
        "seal_eligible": False,
    }
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
