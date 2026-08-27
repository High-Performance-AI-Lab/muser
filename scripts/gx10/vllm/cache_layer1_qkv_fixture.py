#!/usr/bin/env python3
"""Extract the immutable HF tensor set used by the layer-1 QKV micro-fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


PREFIX = "model.language_model.layers.1.self_attn"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to replace fixture cache: {args.output}")

    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    required_keys = [
        f"{PREFIX}.{projection}_proj{suffix}"
        for projection in ("q", "k", "v")
        for suffix in (
            ".weight_packed",
            ".weight_scale",
            ".weight_global_scale",
        )
    ]
    optional_keys = [
        f"{PREFIX}.{projection}_proj.input_global_scale"
        for projection in ("q", "k", "v")
    ]
    keys = required_keys + [key for key in optional_keys if key in weight_map]
    input_scale_count = sum(key in weight_map for key in optional_keys)
    if input_scale_count not in (0, len(optional_keys)):
        raise RuntimeError("mixed QKV activation precision in checkpoint")
    by_shard: dict[str, list[str]] = {}
    for key in keys:
        by_shard.setdefault(weight_map[key], []).append(key)
    tensors = {}
    for shard, shard_keys in sorted(by_shard.items()):
        with safe_open(args.checkpoint / shard, framework="pt", device="cpu") as handle:
            for key in shard_keys:
                tensors[key] = handle.get_tensor(key).contiguous()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": "muser.spark-layer1-qkv-cache.v2",
        "checkpoint": str(args.checkpoint),
        "layer": "1",
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
