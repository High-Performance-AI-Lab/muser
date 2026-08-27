#!/usr/bin/env python3
"""Bit-compare the in-tree NVFP4 oracle with the read-only Ferrite B10 lineage."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct

import numpy as np

from nvfp4_codec import compressed_tensors_scale2, dequantize
from nvfp4_to_f16_gguf import (
    read_checkpoint_receipt,
    rope_row_order,
    safetensors_inventory,
    sha256,
    write_receipt,
)


def load_lineage(path: Path):
    spec = importlib.util.spec_from_file_location("muser_ferrite_b10_lineage", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import Ferrite B10 lineage: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "dequant_numpy", None)):
        raise RuntimeError("Ferrite B10 lineage omitted dequant_numpy")
    return module


def sampled_rows(count: int) -> list[int]:
    if count <= 0:
        raise RuntimeError("NVFP4 matrix has no rows")
    return sorted({0, count // 2, count - 1})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-receipt", type=Path, required=True)
    parser.add_argument("--lineage-script", type=Path, required=True)
    parser.add_argument("--expected-matrices", type=int, default=416)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_receipt = read_checkpoint_receipt(
        args.checkpoint_receipt, args.checkpoint
    )
    assert checkpoint_receipt is not None
    lineage = load_lineage(args.lineage_script)
    inventory = safetensors_inventory(args.checkpoint)
    config = json.loads((args.checkpoint / "config.json").read_text())["text_config"]
    attention_heads = int(config["num_attention_heads"])
    kv_attention_heads = int(config["num_key_value_heads"])
    suffix = ".weight_packed"
    bases = sorted(name[: -len(suffix)] for name in inventory if name.endswith(suffix))
    if len(bases) != args.expected_matrices:
        raise RuntimeError(
            f"NVFP4 matrix count {len(bases)} != {args.expected_matrices}"
        )

    decoded_digest = hashlib.sha256()
    source_digest = hashlib.sha256()
    mismatches: list[dict[str, object]] = []
    decoded_values = 0
    sampled_row_count = 0
    rope_permuted_matrices = 0
    for base in bases:
        packed = inventory[base + suffix].array()
        scales = inventory[base + ".weight_scale"].array()
        global_scale = float(
            inventory[base + ".weight_global_scale"].array().reshape(-1)[0]
        )
        scale2 = compressed_tensors_scale2(global_scale)
        heads = (
            attention_heads
            if base.endswith(".self_attn.q_proj")
            else kv_attention_heads
            if base.endswith(".self_attn.k_proj")
            else None
        )
        row_order = (
            rope_row_order(packed.shape[0], heads)
            if heads is not None
            else np.arange(packed.shape[0], dtype=np.int64)
        )
        rope_permuted_matrices += heads is not None
        for output_row in sampled_rows(packed.shape[0]):
            source_row = int(row_order[output_row])
            packed_row = np.asarray(packed[source_row : source_row + 1])
            scale_row = np.asarray(scales[source_row : source_row + 1])
            actual = dequantize(packed_row, scale_row, scale2)
            expected = lineage.dequant_numpy(packed_row, scale_row, scale2)
            encoded_name = base.encode()
            identity = struct.pack("<I", len(encoded_name)) + encoded_name + struct.pack(
                "<II", output_row, source_row
            )
            source_digest.update(identity)
            source_digest.update(packed_row.tobytes())
            source_digest.update(scale_row.tobytes())
            source_digest.update(struct.pack("<f", global_scale))
            decoded_digest.update(identity)
            decoded_digest.update(actual.astype("<f4", copy=False).tobytes())
            decoded_values += actual.size
            sampled_row_count += 1
            if not np.array_equal(actual, expected):
                differing = int(np.count_nonzero(actual != expected))
                mismatches.append(
                    {
                        "tensor": base,
                        "output_row": output_row,
                        "source_row": source_row,
                        "values": differing,
                    }
                )

    result: dict[str, object] = {
        "schema": "muser.nvfp4-oracle-lineage-check.v4",
        "status": "passed" if not mismatches else "failed",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_artifact_sha256": checkpoint_receipt["artifact_sha256"],
        "checkpoint_receipt": str(args.checkpoint_receipt.resolve()),
        "checkpoint_receipt_sha256": sha256(args.checkpoint_receipt),
        "lineage_script": str(args.lineage_script.resolve()),
        "lineage_script_sha256": sha256(args.lineage_script),
        "matrices": len(bases),
        "sampled_rows": sampled_row_count,
        "decoded_values": decoded_values,
        "sample_source_sha256": source_digest.hexdigest(),
        "decoded_f32_sha256": decoded_digest.hexdigest(),
        "bit_exact_mismatches": len(mismatches),
        "rope_permuted_qk_matrices": rope_permuted_matrices,
        "mismatches": mismatches,
        "scale_contract": {
            "checkpoint_field": "weight_global_scale",
            "decode_scale2": "float32(1.0) / float32(weight_global_scale)",
        },
        "rope_permutation_contract": {
            "q_heads": attention_heads,
            "k_heads": kv_attention_heads,
            "operation": "reshape(heads,2,rows/heads/2,...).swapaxes(1,2).reshape(original)",
            "reference_commit": "8918deaa8ea79ad859dd73ab66f4c452fa70c4ce",
            "reference_file": "conversion/muse_glimmer.py",
            "reference_file_sha256": "dd9e86a74fd3e6e90ebc74b7185b5d12b47ae7935b3464832f82f5f612ba4474",
        },
        "norm_shift_contract": {
            "shift": 1.0,
            "layer_norms": 208,
            "final_norm_shift": 0.0,
            "reference_commit": "8918deaa8ea79ad859dd73ab66f4c452fa70c4ce",
            "reference_file": "conversion/muse_glimmer.py",
            "reference_file_sha256": "dd9e86a74fd3e6e90ebc74b7185b5d12b47ae7935b3464832f82f5f612ba4474",
        },
    }
    write_receipt(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
