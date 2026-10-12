#!/usr/bin/env python3
"""Cheap contract check for the weight-only NVFP4 loader patch."""

from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from muser_vllm.exact_fp4_mm import (
    _exact_ct_process_weights,
    _exact_integer_a16_process_weights,
    metadata,
)


class FakeKernel:
    def __init__(self) -> None:
        self.calls = 0

    def process_weights_after_loading(self, layer: object) -> None:
        self.calls += 1
        _exact_integer_a16_process_weights(self, layer)


def main() -> None:
    packed = torch.arange(24, dtype=torch.uint8).reshape(6, 4)
    artifact_scales = torch.nn.Parameter(
        torch.tensor([0.25, 0.5, 1.0], dtype=torch.float32),
        requires_grad=False,
    )
    layer = SimpleNamespace(
        weight_packed=packed,
        weight_global_scale=artifact_scales,
        logical_widths=[2, 2, 2],
    )
    kernel = FakeKernel()
    scheme = SimpleNamespace(use_a16=True, kernel=kernel)

    _exact_ct_process_weights(scheme, layer)

    assert not hasattr(layer, "weight_packed")
    assert layer.weight is packed
    assert kernel.calls == 1
    assert layer.muser_exact_a16_q8 is True
    assert layer.weight_global_scale.dtype == torch.float32
    assert layer.weight_global_scale.numel() == 3
    assert torch.equal(
        layer.weight_global_scale.detach(),
        torch.tensor([4.0, 2.0, 1.0], dtype=torch.float32),
    )
    report = metadata() | {
        "schema": "muser.spark-exact-fp4-a16-loader.v1",
        "artifact_scale_count": artifact_scales.numel(),
        "runtime_scale_count": layer.weight_global_scale.numel(),
        "logical_widths": layer.logical_widths,
        "marlin_repack": "bypassed",
        "status": "pass",
    }
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
