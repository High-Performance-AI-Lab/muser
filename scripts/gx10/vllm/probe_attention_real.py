#!/usr/bin/env python3
"""Compare exact CUDA attention with retained real-model Mac boundaries."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from muser_vllm.exact_attention import exact_attention_gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mac-capture", required=True, type=Path)
    args = parser.parse_args()

    def load(name: str, width: int) -> np.ndarray:
        return (
            np.fromfile(args.mac_capture / f"{name}.f32", dtype="<f4")
            .astype("<f2")
            .reshape(-1, width)
        )

    query = load("Qcur_rope-0", 4096)
    key = load("Kcur_rope-0", 256)
    value = load("Vcur-0", 256)
    gate = load("attn_gate_proj-0", 4096)
    expected = load("attn_gated-0", 4096)
    actual = exact_attention_gate(
        torch.from_numpy(query).cuda(),
        torch.from_numpy(key).cuda(),
        torch.from_numpy(value).cuda(),
        torch.from_numpy(gate).cuda(),
        1.0 / math.sqrt(128.0),
        1024,
    )
    torch.cuda.synchronize()
    actual = actual.cpu().numpy()
    mismatch = actual.view(np.uint16) != expected.view(np.uint16)
    locations = np.argwhere(mismatch)
    first = None
    if locations.size:
        token, element = (int(value) for value in locations[0])
        first = {
            "token": token,
            "element": element,
            "actual": float(actual[token, element]),
            "expected": float(expected[token, element]),
        }
    print(
        json.dumps(
            {
                "schema": "muser.spark-real-attention-probe.v1",
                "elements": actual.size,
                "mismatches": int(mismatch.sum()),
                "mismatches_by_token": [int(row.sum()) for row in mismatch],
                "max_abs": float(
                    np.max(
                        np.abs(
                            actual.astype(np.float32) - expected.astype(np.float32)
                        )
                    )
                ),
                "first": first,
                "seal_eligible": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
