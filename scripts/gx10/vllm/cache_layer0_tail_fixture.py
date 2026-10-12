#!/usr/bin/env python3
"""Extract the small immutable tensor set used by the layer-0 tail probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


PREFIX = "model.language_model"
LINEARS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.gate_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tokens", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to replace fixture cache: {args.output}")

    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    layer = f"{PREFIX}.layers.0"
    required_keys: list[str] = []
    optional_keys: list[str] = []
    for linear in LINEARS:
        base = f"{layer}.{linear}"
        required_keys.extend(
            base + suffix
            for suffix in (
                ".weight_packed",
                ".weight_scale",
                ".weight_global_scale",
            )
        )
        optional_keys.append(base + ".input_global_scale")
    input_scale_count = sum(key in weight_map for key in optional_keys)
    if input_scale_count not in (0, len(optional_keys)):
        raise RuntimeError("mixed layer-0 activation precision in checkpoint")
    keys = required_keys + [key for key in optional_keys if key in weight_map]
    keys.extend(
        f"{layer}.{name}.weight"
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        )
    )

    by_shard: dict[str, list[str]] = {}
    for key in keys:
        by_shard.setdefault(weight_map[key], []).append(key)
    tensors: dict[str, torch.Tensor] = {}
    for shard, shard_keys in sorted(by_shard.items()):
        with safe_open(args.checkpoint / shard, framework="pt", device="cpu") as handle:
            for key in shard_keys:
                tensors[key] = handle.get_tensor(key).contiguous()

    token_ids = [
        int(line) for line in args.tokens.read_text().splitlines() if line.strip()
    ]
    embedding_key = f"{PREFIX}.embed_tokens.weight"
    with safe_open(
        args.checkpoint / weight_map[embedding_key], framework="pt", device="cpu"
    ) as handle:
        embedding = handle.get_slice(embedding_key)
        tensors["__fixture__.embedding"] = torch.cat(
            [embedding[token_id : token_id + 1] for token_id in token_ids], dim=0
        ).contiguous()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": "muser.spark-layer0-tail-cache.v2",
        "checkpoint": str(args.checkpoint),
        "token_sha256": hashlib.sha256(args.tokens.read_bytes()).hexdigest(),
        "token_count": str(len(token_ids)),
        "activation_precision": "nvfp4" if input_scale_count else "f16-weight-only",
    }
    save_file(tensors, args.output, metadata=metadata)
    print(
        json.dumps(
            {
                **metadata,
                "output": str(args.output),
                "bytes": args.output.stat().st_size,
                "tensor_count": len(tensors),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
