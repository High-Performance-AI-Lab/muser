#!/usr/bin/env python3
"""Compare pinned vLLM RoPE with retained normalized Mac Q/K rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding import get_rope

from muser_vllm.exact_rope import exact_rope


def load(path: Path, width: int) -> torch.Tensor:
    values = np.fromfile(path, dtype="<f4")
    if values.size % width:
        raise RuntimeError(f"{path} has {values.size} values, not rows of {width}")
    return torch.from_numpy(values.reshape(-1, width).astype("<f2")).cuda()


def report(name: str, actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    actual_array = actual.detach().to(torch.float16).cpu().numpy()
    expected_array = expected.detach().to(torch.float16).cpu().numpy()
    mismatch = actual_array.view(np.uint16) != expected_array.view(np.uint16)
    locations = np.argwhere(mismatch)
    return {
        "name": name,
        "elements": int(actual_array.size),
        "mismatches": int(mismatch.sum()),
        "mismatches_by_token": [int(row.sum()) for row in mismatch],
        "first_mismatch": None
        if locations.size == 0
        else [int(value) for value in locations[0]],
        "max_abs": float(
            np.max(
                np.abs(
                    actual_array.astype(np.float32)
                    - expected_array.astype(np.float32)
                )
            )
        ),
    }


def interleaved_to_neox(values: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    return (
        values.reshape(-1, heads, head_dim // 2, 2)
        .transpose(-2, -1)
        .reshape(values.shape)
    )


def neox_to_interleaved(values: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
    return (
        values.reshape(-1, heads, 2, head_dim // 2)
        .transpose(-2, -1)
        .reshape(values.shape)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mac-capture", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    text = config.get("text_config", config)
    head_dim = int(text["head_dim"])
    query = load(args.mac_capture / "Qcur_normed-0.f32", 4096)
    key = load(args.mac_capture / "Kcur_normed-0.f32", 256)
    expected_query = load(args.mac_capture / "Qcur_rope-0.f32", 4096)
    expected_key = load(args.mac_capture / "Kcur_rope-0.f32", 256)
    query = interleaved_to_neox(query, 32, head_dim)
    key = interleaved_to_neox(key, 2, head_dim)
    positions = torch.arange(query.shape[0], dtype=torch.long, device="cuda")
    with set_current_vllm_config(VllmConfig()):
        rope = get_rope(
            head_dim,
            max_position=int(text["max_position_embeddings"]),
            rope_parameters=text["rope_parameters"],
            is_neox_style=True,
        ).cuda()
        actual_query, actual_key = rope(positions, query, key)
        exact_query, exact_key = exact_rope(
            query, key, rope.cos_sin_cache, positions, head_dim
        )
    actual_query = neox_to_interleaved(actual_query, 32, head_dim)
    actual_key = neox_to_interleaved(actual_key, 2, head_dim)
    exact_query = neox_to_interleaved(exact_query, 32, head_dim)
    exact_key = neox_to_interleaved(exact_key, 2, head_dim)
    print(
        json.dumps(
            {
                "schema": "muser.spark-real-rope-probe.v1",
                "head_dim": head_dim,
                "reports": [
                    report("query-vllm-rope", actual_query, expected_query),
                    report("key-vllm-rope", actual_key, expected_key),
                    report("query-exact-rope", exact_query, expected_query),
                    report("key-exact-rope", exact_key, expected_key),
                ],
                "seal_eligible": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
