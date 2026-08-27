#!/usr/bin/env python3
"""Compare the pinned producer QK-norm boundary with retained Mac rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from muser_vllm.exact_rms_norm import exact_rms_norm, exact_split_rms_norm


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
        "mismatches_by_token": [int(row.sum()) for row in mismatch.reshape(32, -1)],
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mac-capture", required=True, type=Path)
    parser.add_argument("--query-scale", required=True, type=float)
    parser.add_argument("--eps", type=float, default=1.0e-5)
    args = parser.parse_args()

    query = load(args.mac_capture / "Qcur-0.f32", 128)
    key = load(args.mac_capture / "Kcur-0.f32", 128)
    expected_query = load(args.mac_capture / "Qcur_normed-0.f32", 128)
    expected_key = load(args.mac_capture / "Kcur_normed-0.f32", 128)
    empty = torch.empty(0, dtype=torch.float16, device="cuda")
    query_scale = torch.full(
        (128,), args.query_scale, dtype=torch.float32, device="cuda"
    )
    key_scale = torch.ones((128,), dtype=torch.float32, device="cuda")

    # This is the literal pinned vLLM forward contract: weightless RMSNorm
    # returns model dtype, followed by Python scalar multiplication for Q.
    query_vllm = exact_rms_norm(query, empty, args.eps, False, 0)
    query_vllm = query_vllm * args.query_scale
    key_vllm = exact_rms_norm(key, empty, args.eps, False, 0)

    reports = [
        report("query-vllm-python-scale", query_vllm, expected_query),
        report("key-vllm-weightless", key_vllm, expected_key),
        report(
            "query-explicit-triton-scale",
            exact_split_rms_norm(query, query_scale, args.eps, 0),
            expected_query,
        ),
        report(
            "key-explicit-triton-scale",
            exact_split_rms_norm(key, key_scale, args.eps, 0),
            expected_key,
        ),
        report(
            "query-fused-scale",
            exact_rms_norm(query, query_scale, args.eps, True, 0),
            expected_query,
        ),
    ]
    print(
        json.dumps(
            {
                "schema": "muser.spark-real-qk-norm-probe.v1",
                "query_scale": args.query_scale,
                "eps": args.eps,
                "reports": reports,
                "seal_eligible": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
