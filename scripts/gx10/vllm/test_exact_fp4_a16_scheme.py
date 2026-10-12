#!/usr/bin/env python3
"""Validate vLLM's real Dudeman scheme selection without loading the model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

from muser_vllm.exact_fp4_mm import install_exact_fp4_mm


PREFIX = "model.language_model.layers.1.self_attn"
PROJECTIONS = ("q_proj", "k_proj", "v_proj")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    args = parser.parse_args()

    config = json.loads((args.model / "config.json").read_text())
    quantization_config = config["quantization_config"]
    patch_receipt = install_exact_fp4_mm()

    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

    quant_config = CompressedTensorsConfig.from_config(quantization_config)
    scheme = quant_config.get_scheme(
        torch.nn.Linear(1, 1, bias=False),
        layer_name=f"{PREFIX}.q_proj",
    )
    if scheme is None or scheme.__class__.__name__ != "CompressedTensorsW4A4Fp4":
        raise RuntimeError(f"unexpected selected scheme: {type(scheme).__name__}")
    if not scheme.use_a16:
        raise RuntimeError("weight-only checkpoint did not select the A16 scheme")

    index = json.loads((args.model / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    raw_scales = []
    source_files = []
    for projection in PROJECTIONS:
        key = f"{PREFIX}.{projection}.weight_global_scale"
        source = args.model / weight_map[key]
        with safe_open(source, framework="pt", device="cpu") as handle:
            raw_scales.append(handle.get_tensor(key).reshape(()).to(torch.float32))
        source_files.append(source.name)
    artifact_scales = torch.stack(raw_scales)
    if torch.unique(artifact_scales).numel() != len(PROJECTIONS):
        raise RuntimeError("fixture requires distinct Q/K/V artifact scales")

    packed_sentinel = torch.arange(3, dtype=torch.uint8).reshape(3, 1)
    layer = SimpleNamespace(
        weight_packed=packed_sentinel,
        weight_global_scale=torch.nn.Parameter(
            artifact_scales.clone(), requires_grad=False
        ),
        logical_widths=[1, 1, 1],
    )
    scheme.process_weights_after_loading(layer)
    expected_runtime = torch.reciprocal(artifact_scales)
    if not torch.equal(layer.weight_global_scale.detach(), expected_runtime):
        raise RuntimeError("patched loader did not preserve per-projection scales")
    if layer.weight is not packed_sentinel or hasattr(layer, "weight_packed"):
        raise RuntimeError("patched loader changed the checkpoint weight address map")
    if not getattr(layer, "muser_exact_a16_q8", False):
        raise RuntimeError("patched Marlin bypass was not selected")

    print(
        json.dumps(
            {
                "schema": "muser.spark-exact-fp4-a16-scheme.v1",
                "model_type": config["model_type"],
                "checkpoint_format": quantization_config["format"],
                "scheme": type(scheme).__name__,
                "use_a16": scheme.use_a16,
                "kernel": type(scheme.kernel).__name__,
                "artifact_scale_bits": [
                    int(value.view(torch.int32)) for value in artifact_scales
                ],
                "runtime_scale_bits": [
                    int(value.view(torch.int32)) for value in expected_runtime
                ],
                "source_files": source_files,
                "patch_selection": patch_receipt["selection"],
                "marlin_repack": "bypassed",
                "status": "pass",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
