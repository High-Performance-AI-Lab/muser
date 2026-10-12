#!/usr/bin/env python3
"""Export the official five-layer DFlash graph as public-CoreML shards.

The output is intentionally narrow: INT8 1x1-convolution mlprograms over exact
input- or output-channel partitions, plus fused output/residual/norm/FFN tail
programs and a content-addressed manifest consumed by Muser. Official wide matrices use
compiler-safe channel slices so every compiled program stays within the ANE
channel and package-size contracts.
Run this from an isolated, pinned coremltools environment; it never invokes
prediction or benchmarks the Neural Engine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

MAX_SHARD_BYTES = 250 * 1024 * 1024
ANE_CHANNEL_SLICE = 6656
# The first tail program also carries the official 4,096 -> 6,656 output
# projection. A 9,216-channel FFN partition plus that matrix contains
# 211,288,064 INT8 weight bytes. The remaining 10,752-channel continuation
# contains 214,695,936 bytes.
# Both remain below the release package ceiling without splitting a dimension
# the public compiler has already accepted.
ANE_TAIL_CHANNEL_SLICE = 9216
# On the release M3 Ultra compute plan, the 6,656 -> 1,024 K/V convolutions
# fall back to CPU while the otherwise identical 2,048+-output programs stay
# resident on ANE. Pad only the internal convolution and slice its public
# output back to the exact projection width.
ANE_OUTPUT_CHANNEL_FLOOR = 2048
ROOT = Path(__file__).resolve().parents[1]
COREMLTOOLS_VERSION = "9.0"
NUMPY_VERSION = "2.1.3"
ANE_BOOK_REVISION = "3cf5969eda414832e0cb6c58e3372400fc3c6277"


def imports():
    try:
        import coremltools as ct
        import coremltools.optimize.coreml as cto
        import numpy as np
        from coremltools.converters.mil.mil import Builder as mb
        from coremltools.converters.mil.mil import types
    except ImportError as error:
        raise SystemExit(
            "export requires pinned coremltools and numpy: " + str(error)
        ) from error
    if ct.__version__ != COREMLTOOLS_VERSION or np.__version__ != NUMPY_VERSION:
        raise SystemExit(
            "export requires the pinned Core ML toolchain: "
            f"coremltools=={COREMLTOOLS_VERSION}, numpy=={NUMPY_VERSION}; got "
            f"coremltools=={ct.__version__}, numpy=={np.__version__}"
        )
    return ct, cto, np, mb, types


def tree_files(path: Path) -> list[tuple[str, Path]]:
    if path.is_file():
        return [(".", path)]
    if not path.is_dir():
        raise ValueError(f"shard is not a file or directory: {path}")
    result: list[tuple[str, Path]] = []
    for child in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
        if child.is_symlink():
            raise ValueError(f"shard contains symlink: {child}")
        if child.is_file():
            result.append((child.relative_to(path).as_posix(), child))
        elif not child.is_dir():
            raise ValueError(f"shard contains special entry: {child}")
    return result


def tree_receipt(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    for relative, file in tree_files(path):
        name = relative.encode("utf-8")
        size = file.stat().st_size
        total += size
        digest.update(struct.pack("<Q", len(name)))
        digest.update(name)
        digest.update(struct.pack("<Q", size))
        with file.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    return total, digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def development_dflash_identity(config_path: Path, weights_path: Path) -> str:
    digest = hashlib.sha256(b"muser-dflash-artifact-v1\0")
    for path in (config_path, weights_path):
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def describe_official(extractor: Path, artifact: Path) -> dict:
    result = subprocess.run(
        [str(extractor), "--artifact", str(artifact), "--describe"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return json.loads(result.stdout)


def extract_official_projection(
    extractor: Path,
    artifact: Path,
    tensor_name: str,
    output: Path,
    rows: int,
    columns: int,
) -> None:
    result = subprocess.run(
        [
            str(extractor),
            "--artifact", str(artifact),
            "--tensor", tensor_name,
            "--output", str(output),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    receipt = json.loads(result.stdout)
    if (
        receipt.get("schema") != "muser.dflash-projection-f32.v1"
        or receipt.get("tensor") != tensor_name
        or receipt.get("rows") != rows
        or receipt.get("columns") != columns
        or receipt.get("bytes") != rows * columns * 4
        or output.stat().st_size != rows * columns * 4
    ):
        raise ValueError(f"extractor returned invalid geometry for {tensor_name}")


def projection_geometry(config: dict) -> list[tuple[str, str, int, int]]:
    layers = int(config["num_hidden_layers"])
    if layers != 5:
        raise ValueError(f"release DFlash must have five layers, got {layers}")
    hidden = int(config["hidden_size"])
    q_width = int(config["num_attention_heads"]) * int(config["head_dim"])
    kv_width = int(config["num_key_value_heads"]) * int(config["head_dim"])
    intermediate = int(config["intermediate_size"])
    sampled = len(config["dflash_config"]["target_layer_ids"])
    values = [("fc", "fc.weight", sampled * hidden, hidden)]
    for layer in range(layers):
        prefix = f"layers.{layer}"
        for suffix, input_width, output_width in (
            ("self_attn.q_proj.weight", hidden, q_width),
            ("self_attn.k_proj.weight", hidden, kv_width),
            ("self_attn.v_proj.weight", hidden, kv_width),
            ("self_attn.o_proj.weight", q_width, hidden),
            ("mlp.gate_proj.weight", hidden, intermediate),
            ("mlp.up_proj.weight", hidden, intermediate),
            ("mlp.down_proj.weight", intermediate, hidden),
        ):
            projection = f"{prefix}.{suffix.removesuffix('.weight').replace('self_attn.', '') .replace('mlp.', '')}"
            values.append((projection, f"{prefix}.{suffix}", input_width, output_width))
    return values


def projection_fragments(
    projection: str, tensor: str, input_width: int, output_width: int,
) -> list[dict]:
    if input_width > ANE_CHANNEL_SLICE and output_width > ANE_CHANNEL_SLICE:
        raise ValueError(
            f"{projection} requires unsupported two-dimensional ANE partitioning"
        )
    fragments = []
    if input_width > ANE_CHANNEL_SLICE:
        for input_offset in range(0, input_width, ANE_CHANNEL_SLICE):
            fragments.append(
                {
                    "projection": projection,
                    "tensor": tensor,
                    "input_offset": input_offset,
                    "input_width": min(ANE_CHANNEL_SLICE, input_width - input_offset),
                    "output_offset": 0,
                    "output_width": output_width,
                }
            )
    elif output_width > ANE_CHANNEL_SLICE:
        for output_offset in range(0, output_width, ANE_CHANNEL_SLICE):
            fragments.append(
                {
                    "projection": projection,
                    "tensor": tensor,
                    "input_offset": 0,
                    "input_width": input_width,
                    "output_offset": output_offset,
                    "output_width": min(ANE_CHANNEL_SLICE, output_width - output_offset),
                }
            )
    else:
        fragments.append(
            {
                "projection": projection,
                "tensor": tensor,
                "input_offset": 0,
                "input_width": input_width,
                "output_offset": 0,
                "output_width": output_width,
            }
        )
    return fragments


def fused_fragments(
    projections: list[tuple[str, str, int, int]], block_size: int,
    split_capture_fc: bool = False,
) -> list[dict]:
    """Build compiler-safe physical graphs while retaining logical outputs.

    Q/K/V share one input and fit comfortably below the artifact ceiling.
    The output projection and FFN are emitted in the fused layer-tail graphs.
    """
    logical = {
        projection: {
            "projection": projection,
            "tensor": tensor,
            "input_width": input_width,
            "output_width": output_width,
        }
        for projection, tensor, input_width, output_width in projections
    }
    result: list[dict] = []

    def add_single(projection: str) -> None:
        value = logical[projection]
        for fragment in projection_fragments(
            projection, value["tensor"], value["input_width"], value["output_width"]
        ):
            result.append(
                {
                    "projection": projection,
                    "batch": block_size,
                    "input_offset": fragment["input_offset"],
                    "input_width": fragment["input_width"],
                    "output_width": fragment["output_width"],
                    "components": [
                        {
                            "projection": projection,
                            "tensor": value["tensor"],
                            "tensor_output_offset": fragment["output_offset"],
                            "output_offset": 0,
                            "projection_offset": fragment["output_offset"],
                            "output_width": fragment["output_width"],
                        }
                    ],
                }
            )

    fc = logical["fc"]
    # The retained MLComputePlan packet proves the 33,280 -> 6,656 FC graph
    # ANE-resident, and its INT8 package is 221.5 MB. The stable v7 route keeps
    # it whole. The default-off v8 experiment emits one input slice per exact
    # target capture so those five predictions can be hidden under later Metal
    # layers instead of paying their dispatches serially.
    fc_offsets = (
        range(0, fc["input_width"], fc["output_width"])
        if split_capture_fc else [0]
    )
    for input_offset in fc_offsets:
        input_width = (
            min(fc["output_width"], fc["input_width"] - input_offset)
            if split_capture_fc else fc["input_width"]
        )
        result.append(
            {
                "projection": "fc",
                "batch": block_size,
                "input_offset": input_offset,
                "input_width": input_width,
                "output_width": fc["output_width"],
                "components": [
                    {
                        "projection": "fc",
                        "tensor": fc["tensor"],
                        "tensor_output_offset": 0,
                        "output_offset": 0,
                        "projection_offset": 0,
                        "output_width": fc["output_width"],
                    }
                ],
            }
        )
    layers = len([name for name in logical if name.endswith(".q_proj")])
    for layer in range(layers):
        qkv = [
            f"layers.{layer}.q_proj",
            f"layers.{layer}.k_proj",
            f"layers.{layer}.v_proj",
        ]
        input_widths = {logical[name]["input_width"] for name in qkv}
        if len(input_widths) != 1 or next(iter(input_widths)) > ANE_CHANNEL_SLICE:
            raise ValueError(f"layer {layer} Q/K/V cannot use the fused ANE graph")
        physical_offset = 0
        components = []
        for name in qkv:
            width = logical[name]["output_width"]
            components.append(
                {
                    "projection": name,
                    "tensor": logical[name]["tensor"],
                    "tensor_output_offset": 0,
                    "output_offset": physical_offset,
                    "projection_offset": 0,
                    "output_width": width,
                }
            )
            physical_offset += width
        result.append(
            {
                "projection": f"layers.{layer}.qkv_fused",
                "batch": 2 * block_size,
                "input_offset": 0,
                "input_width": next(iter(input_widths)),
                "output_width": physical_offset,
                "components": components,
            }
        )

    return result


def fused_tail_fragments(
    projections: list[tuple[str, str, int, int]],
) -> list[dict]:
    logical = {
        projection: {
            "tensor": tensor,
            "input_width": input_width,
            "output_width": output_width,
        }
        for projection, tensor, input_width, output_width in projections
    }
    layers = len([name for name in logical if name.endswith(".gate_proj")])
    result = []
    for layer in range(layers):
        gate = logical[f"layers.{layer}.gate_proj"]
        up = logical[f"layers.{layer}.up_proj"]
        down = logical[f"layers.{layer}.down_proj"]
        if (
            gate["input_width"] != up["input_width"]
            or gate["output_width"] != up["output_width"]
            or down["input_width"] != gate["output_width"]
            or down["output_width"] != gate["input_width"]
        ):
            raise ValueError(f"layer {layer} fused FFN geometry differs")
        o = logical[f"layers.{layer}.o_proj"]
        if o["output_width"] != gate["input_width"]:
            raise ValueError(f"layer {layer} output projection has the wrong hidden width")
        head_width = min(ANE_TAIL_CHANNEL_SLICE, gate["output_width"])
        widths = [head_width]
        if head_width < gate["output_width"]:
            widths.append(gate["output_width"] - head_width)
        offset = 0
        for order, width in enumerate(widths):
            head = order == 0
            int8_weight_bytes = 3 * width * gate["input_width"]
            if head:
                int8_weight_bytes += o["input_width"] * o["output_width"]
            if int8_weight_bytes > MAX_SHARD_BYTES:
                raise ValueError(
                    f"layer {layer} fused FFN partition requires "
                    f"{int8_weight_bytes} INT8 weight bytes; release limit is "
                    f"{MAX_SHARD_BYTES}"
                )
            result.append(
                {
                    "layer": layer,
                    "order": order,
                    "head": head,
                    "intermediate_offset": offset,
                    "intermediate_width": width,
                    "hidden_width": gate["input_width"],
                    "attention_width": o["input_width"],
                    "int8_weight_bytes": int8_weight_bytes,
                    "gate_tensor": gate["tensor"],
                    "up_tensor": up["tensor"],
                    "down_tensor": down["tensor"],
                    "o_tensor": o["tensor"] if head else None,
                    "norm_tensor": (
                        f"layers.{layer}.post_attention_layernorm.weight" if head else None
                    ),
                }
            )
            offset += width
    return result


def export_projection(
    *, ct, cto, np, mb, types, weight, batch: int, input_width: int,
    output_width: int, output: Path,
) -> None:
    if tuple(weight.shape) != (output_width, input_width):
        raise ValueError(
            f"{output.name}: weight shape {weight.shape}, expected {(output_width, input_width)}"
        )
    convolution_width = max(output_width, ANE_OUTPUT_CHANNEL_FLOOR)
    conv_weight = np.zeros(
        (convolution_width, input_width, 1, 1), dtype=np.float32
    )
    conv_weight[:output_width, :, 0, 0] = np.asarray(weight, dtype=np.float32)

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, input_width, batch, 1), dtype=types.fp32)],
        opset_version=ct.target.iOS18,
    )
    def program(input):
        projected = mb.conv(
            x=input,
            weight=conv_weight,
            pad_type="valid",
            strides=[1, 1],
            dilations=[1, 1],
            groups=1,
            name="padded_output" if convolution_width != output_width else "output",
        )
        if convolution_width == output_width:
            return projected
        return mb.slice_by_size(
            x=projected,
            begin=[0, 0, 0, 0],
            size=[1, output_width, batch, 1],
            name="output",
        )

    model = ct.convert(
        program,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
    )
    quantizer = cto.OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int8", granularity="per_tensor"
    )
    model = cto.linear_quantize_weights(
        model, cto.OptimizationConfig(global_config=quantizer)
    )
    if output.exists():
        shutil.rmtree(output)
    model.save(str(output))


def export_fused_ffn(
    *, ct, cto, np, mb, types, gate_weight, up_weight, down_weight,
    batch: int, hidden_width: int, intermediate_width: int, output: Path,
) -> None:
    if tuple(gate_weight.shape) != (intermediate_width, hidden_width):
        raise ValueError(f"{output.name}: invalid fused gate shape {gate_weight.shape}")
    if tuple(up_weight.shape) != (intermediate_width, hidden_width):
        raise ValueError(f"{output.name}: invalid fused up shape {up_weight.shape}")
    if tuple(down_weight.shape) != (hidden_width, intermediate_width):
        raise ValueError(f"{output.name}: invalid fused down shape {down_weight.shape}")
    gate_conv = np.asarray(gate_weight, dtype=np.float32)[:, :, None, None]
    up_conv = np.asarray(up_weight, dtype=np.float32)[:, :, None, None]
    down_conv = np.asarray(down_weight, dtype=np.float32)[:, :, None, None]

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, hidden_width, batch, 1), dtype=types.fp32)],
        opset_version=ct.target.iOS18,
    )
    def program(input):
        gate = mb.conv(
            x=input, weight=gate_conv, pad_type="valid", strides=[1, 1],
            dilations=[1, 1], groups=1, name="gate",
        )
        up = mb.conv(
            x=input, weight=up_conv, pad_type="valid", strides=[1, 1],
            dilations=[1, 1], groups=1, name="up",
        )
        activated = mb.mul(
            x=mb.mul(x=gate, y=mb.sigmoid(x=gate)), y=up, name="activated",
        )
        return mb.conv(
            x=activated, weight=down_conv, pad_type="valid", strides=[1, 1],
            dilations=[1, 1], groups=1, name="output",
        )

    model = ct.convert(
        program,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
    )
    quantizer = cto.OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int8", granularity="per_tensor"
    )
    model = cto.linear_quantize_weights(
        model, cto.OptimizationConfig(global_config=quantizer)
    )
    if output.exists():
        shutil.rmtree(output)
    model.save(str(output))


def export_fused_tail_head(
    *, ct, cto, np, mb, types, o_weight, norm_weight, gate_weight, up_weight,
    down_weight, rms_norm_eps: float, batch: int, hidden_width: int,
    attention_width: int, intermediate_width: int, output: Path,
) -> None:
    if tuple(o_weight.shape) != (hidden_width, attention_width):
        raise ValueError(f"{output.name}: invalid output projection shape {o_weight.shape}")
    if tuple(norm_weight.shape) != (hidden_width,):
        raise ValueError(f"{output.name}: invalid post-attention norm shape {norm_weight.shape}")
    if tuple(gate_weight.shape) != (intermediate_width, hidden_width):
        raise ValueError(f"{output.name}: invalid fused gate shape {gate_weight.shape}")
    if tuple(up_weight.shape) != (intermediate_width, hidden_width):
        raise ValueError(f"{output.name}: invalid fused up shape {up_weight.shape}")
    if tuple(down_weight.shape) != (hidden_width, intermediate_width):
        raise ValueError(f"{output.name}: invalid fused down shape {down_weight.shape}")
    o_conv = np.asarray(o_weight, dtype=np.float32)[:, :, None, None]
    norm_scale = np.asarray(norm_weight, dtype=np.float32)[None, :, None, None]
    gate_conv = np.asarray(gate_weight, dtype=np.float32)[:, :, None, None]
    up_conv = np.asarray(up_weight, dtype=np.float32)[:, :, None, None]
    down_conv = np.asarray(down_weight, dtype=np.float32)[:, :, None, None]

    @mb.program(
        input_specs=[mb.TensorSpec(
            shape=(1, attention_width + hidden_width, batch, 1), dtype=types.fp32,
        )],
        opset_version=ct.target.iOS18,
    )
    def program(input):
        attention = mb.slice_by_size(
            x=input, begin=[0, 0, 0, 0], size=[1, attention_width, batch, 1],
            name="attention",
        )
        residual = mb.slice_by_size(
            x=input, begin=[0, attention_width, 0, 0],
            size=[1, hidden_width, batch, 1], name="residual",
        )
        projected = mb.conv(
            x=attention, weight=o_conv, pad_type="valid", strides=[1, 1],
            dilations=[1, 1], groups=1, name="output_projection",
        )
        post_attention = mb.add(x=residual, y=projected, name="post_attention")
        variance = mb.reduce_mean(
            x=mb.square(x=post_attention), axes=[1], keep_dims=True, name="variance",
        )
        inverse_rms = mb.rsqrt(
            x=mb.add(x=variance, y=np.float32(rms_norm_eps)), name="inverse_rms",
        )
        normed = mb.mul(
            x=mb.mul(x=post_attention, y=inverse_rms), y=norm_scale, name="normed",
        )
        gate = mb.conv(
            x=normed, weight=gate_conv, pad_type="valid", strides=[1, 1],
            dilations=[1, 1], groups=1, name="gate",
        )
        up = mb.conv(
            x=normed, weight=up_conv, pad_type="valid", strides=[1, 1],
            dilations=[1, 1], groups=1, name="up",
        )
        activated = mb.mul(
            x=mb.mul(x=gate, y=mb.sigmoid(x=gate)), y=up, name="activated",
        )
        partial = mb.conv(
            x=activated, weight=down_conv, pad_type="valid", strides=[1, 1],
            dilations=[1, 1], groups=1, name="partial",
        )
        base = mb.add(x=post_attention, y=partial, name="base")
        return mb.concat(values=[base, normed], axis=1, name="output")

    model = ct.convert(
        program,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
    )
    quantizer = cto.OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int8", granularity="per_tensor"
    )
    model = cto.linear_quantize_weights(
        model, cto.OptimizationConfig(global_config=quantizer)
    )
    if output.exists():
        shutil.rmtree(output)
    model.save(str(output))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dflash", required=True, type=Path,
        help="Pinned official DFlash GGUF, or a development artifact directory",
    )
    parser.add_argument(
        "--extractor", type=Path,
        default=ROOT / "target" / "release" / "muser-dflash-extract",
        help="CPU GGUF projection extractor built from this source revision",
    )
    parser.add_argument("--output", required=True, type=Path, help="New output directory")
    parser.add_argument("--target-identity", required=True, help="Exact pinned target identity")
    parser.add_argument("--dflash-identity", help="Expected identity; defaults to artifact digest")
    parser.add_argument(
        "--stateful-attention",
        type=Path,
        help="Exported attention-only artifact to embed for all five persistent layer states",
    )
    parser.add_argument(
        "--split-capture-fc", action="store_true",
        help=(
            "Emit five exact accumulating FC input slices for the default-off "
            "layer-capture pipeline (requires --stateful-attention)"
        ),
    )
    parser.add_argument(
        "--fused-attention-layers",
        type=Path,
        help=(
            "Directory containing layer-0..layer-4 runtime-bundle exports; "
            "enables the fail-closed v9 QKV+stateful-attention route"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate the source and print the exact shard plan without CoreML or output writes",
    )
    return parser.parse_args()


def validate_stateful_attention(
    root: Path | None, expected_dflash: str
) -> tuple[dict, Path] | None:
    if root is None:
        return None
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(f"stateful attention manifest is missing or unsafe: {manifest_path}")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        value.get("schema") != "muser.dflash-stateful-attention-only-export.v1"
        or value.get("mode") != "exported"
        or value.get("block_size") != 16
        or value.get("query_size") != 16
        or value.get("chunk") != 16
        or value.get("kv_write_chunk") != 16
        or value.get("attention_op") != "manual"
        or value.get("kv_join") != "split"
        or value.get("state_group_kv_heads") != 8
        or value.get("max_context") != 1088
        or value.get("dflash_sha256") != expected_dflash
        or value.get("inputs") != [
            "query", "noise_key", "noise_value", "target_key", "target_value",
            "attention_mask", "kv_write_mask",
        ]
        or value.get("output") != "attention"
        or value.get("compute_units") != "CPU_AND_NE"
    ):
        raise ValueError("stateful attention artifact is outside the release contract")
    package = root / value.get("package", "")
    if not package.is_dir() or package.is_symlink():
        raise ValueError("stateful attention package is missing or unsafe")
    size, digest = tree_receipt(package)
    if size != value.get("package_bytes") or digest != value.get("package_sha256"):
        raise ValueError("stateful attention package differs from its manifest")
    if not 0 < size <= MAX_SHARD_BYTES:
        raise ValueError("stateful attention package exceeds the release size contract")
    return value, package


def validate_fused_attention_layers(
    root: Path | None, expected_dflash: str
) -> list[tuple[dict, Path]]:
    if root is None:
        return []
    expected_inputs = [
        "noise_hidden", "query_selector", "target_projected", "target_mask",
        "target_rope_cos", "target_rope_sin", "noise_rope_cos",
        "noise_rope_sin", "replay_target_key", "replay_target_value",
        "replay_mode", "attention_mask", "kv_write_mask",
    ]
    result = []
    for layer in range(5):
        layer_root = root / f"layer-{layer}"
        manifest_path = layer_root / "manifest.json"
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError(f"fused attention layer {layer} manifest is missing or unsafe")
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            value.get("schema") != "muser.dflash-stateful-attention-export.v1"
            or value.get("mode") != "exported"
            or value.get("runtime_bundle") is not True
            or value.get("layer") != layer
            or value.get("block_size") != 16
            or value.get("query_size") != 16
            or value.get("max_context") != 1088
            or value.get("kv_write_chunk") != 16
            or value.get("kv_join") != "split"
            or value.get("state_group_kv_heads") != 4
            or value.get("inputs") != expected_inputs
            or value.get("output") != "attention_target_kv_bundle"
            or value.get("compute_units") != "CPU_AND_NE"
            or value.get("dflash_sha256") != expected_dflash
        ):
            raise ValueError(f"fused attention layer {layer} is outside the v9 contract")
        package = layer_root / value.get("package", "")
        if not package.is_dir() or package.is_symlink():
            raise ValueError(f"fused attention layer {layer} package is missing or unsafe")
        size, digest = tree_receipt(package)
        if (
            not 0 < size <= MAX_SHARD_BYTES
            or size != value.get("package_bytes")
            or digest != value.get("package_sha256")
        ):
            raise ValueError(f"fused attention layer {layer} package differs")
        result.append((value, package))
    return result


def main() -> int:
    args = parse_args()
    official = args.dflash.is_file()
    if official:
        if not args.extractor.is_file() or args.extractor.is_symlink():
            raise ValueError(f"missing or unsafe GGUF extractor: {args.extractor}")
        config = describe_official(args.extractor, args.dflash)
        dflash_identity = file_sha256(args.dflash)
        source_format = "official-gguf"
        extractor_identity = file_sha256(args.extractor)
        weights = None
    else:
        config_path = args.dflash / "config.json"
        weights_path = args.dflash / "model.safetensors"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        dflash_identity = development_dflash_identity(config_path, weights_path)
        source_format = "development-safetensors"
        extractor_identity = None
    projections = projection_geometry(config)
    batch = int(config.get("block_size", 16))
    if args.split_capture_fc and args.stateful_attention is None:
        raise ValueError("--split-capture-fc requires --stateful-attention")
    fragments = fused_fragments(projections, batch, args.split_capture_fc)
    tail_fragments = fused_tail_fragments(projections)
    attention_source = validate_stateful_attention(args.stateful_attention, dflash_identity)
    fused_attention_sources = validate_fused_attention_layers(
        args.fused_attention_layers, dflash_identity
    )
    if fused_attention_sources and (args.split_capture_fc or attention_source is None):
        raise ValueError(
            "v9 fused attention requires one whole FC and the v8 attention fallback"
        )
    manifest_version = (
        9 if fused_attention_sources
        else 8 if args.split_capture_fc
        else 7 if attention_source is not None
        else 6
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "schema": "muser.dflash-coreml-export-plan.v1",
                    "mode": "dry-run",
                    "source_format": source_format,
                    "dflash_identity": args.dflash_identity or dflash_identity,
                    "target_identity": args.target_identity,
                    "manifest_version": manifest_version,
                    "block_size": batch,
                    "tensor_layout": "[1,C,T,1]",
                    "toolchain": {
                        "coremltools": COREMLTOOLS_VERSION,
                        "numpy": NUMPY_VERSION,
                        "ane_book_revision": ANE_BOOK_REVISION,
                    },
                    "projection_count": len(projections),
                    "shard_count": len(fragments) + len(tail_fragments),
                    "decode_predictions_per_round": (
                        sum(fragment["projection"] == "fc" for fragment in fragments)
                        + len(tail_fragments)
                        + len(fused_attention_sources)
                        if fused_attention_sources
                        else len(fragments) + len(tail_fragments)
                        + (
                            int(config["num_hidden_layers"])
                            * (batch // int(attention_source[0]["query_size"]))
                            if attention_source
                            else 0
                        )
                    ),
                    "shards": [dict(order=order, **fragment)
                               for order, fragment in enumerate(fragments)],
                    "ffn_shards": [],
                    "tail_shards": tail_fragments,
                    "attention_shards": (
                        [
                            {
                                "order": layer,
                                "layer": layer,
                                "max_context": 1088,
                                "query_size": 16,
                                "chunk": 16,
                                "kv_write_chunk": 16,
                                "kv_join": "split",
                                "state_group_kv_heads": 8,
                            }
                            for layer in range(int(config["num_hidden_layers"]))
                        ]
                        if attention_source else []
                    ),
                    "fused_attention_shards": [
                        {
                            "order": layer,
                            "layer": layer,
                            "max_context": value["max_context"],
                            "query_size": value["query_size"],
                            "state_group_kv_heads": 4,
                            "bundle_channels": 6144,
                        }
                        for layer, (value, _) in enumerate(fused_attention_sources)
                    ],
                    "output_created": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    ct, cto, np, mb, types = imports()
    if official:
        weights = None
    else:
        try:
            from safetensors.numpy import load_file
        except ImportError as error:
            raise SystemExit("development export requires safetensors: " + str(error)) from error
        weights = load_file(str(weights_path))
    if args.output.exists() or args.output.is_symlink():
        raise ValueError(f"output directory must be absent: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".muser-coreml-", dir=args.output.parent.resolve()))
    extraction = Path(tempfile.mkdtemp(prefix="muser-dflash-f32-"))

    shards = []
    tail_shards = []
    attention_shards = []
    fused_attention_shards = []
    try:
        geometry = {
            projection: (tensor, input_width, output_width)
            for projection, tensor, input_width, output_width in projections
        }
        physical_groups = list(dict.fromkeys(fragment["projection"] for fragment in fragments))
        for physical_group in physical_groups:
            group_fragments = [
                fragment for fragment in fragments
                if fragment["projection"] == physical_group
            ]
            logical_names = list(dict.fromkeys(
                component["projection"]
                for fragment in group_fragments
                for component in fragment["components"]
            ))
            group_weights = {}
            raw_paths = []
            for logical_name in logical_names:
                tensor_name, input_width, output_width = geometry[logical_name]
                if official:
                    raw = extraction / f"projection-{len(raw_paths):02d}-{len(shards):03d}.f32"
                    extract_official_projection(
                        args.extractor, args.dflash, tensor_name, raw,
                        output_width, input_width,
                    )
                    raw_paths.append(raw)
                    group_weights[logical_name] = np.memmap(
                        raw, mode="r", dtype="<f4", shape=(output_width, input_width)
                    )
                else:
                    if tensor_name not in weights:
                        raise ValueError(f"missing DFlash tensor {tensor_name}")
                    group_weights[logical_name] = weights[tensor_name]
            for fragment in group_fragments:
                order = len(shards)
                input_offset = fragment["input_offset"]
                fragment_input = fragment["input_width"]
                fragment_output = fragment["output_width"]
                fragment_batch = fragment["batch"]
                package = stage / f"projection-{order:03d}.mlpackage"
                weight_parts = []
                physical_cursor = 0
                for component in fragment["components"]:
                    if component["output_offset"] != physical_cursor:
                        raise ValueError(f"{physical_group} has a physical output gap")
                    source_offset = component["tensor_output_offset"]
                    width = component["output_width"]
                    source = group_weights[component["projection"]]
                    weight_parts.append(source[
                        source_offset:source_offset + width,
                        input_offset:input_offset + fragment_input,
                    ])
                    physical_cursor += width
                if physical_cursor != fragment_output:
                    raise ValueError(f"{physical_group} physical width mismatch")
                fragment_weight = np.concatenate(weight_parts, axis=0)
                export_projection(
                    ct=ct, cto=cto, np=np, mb=mb, types=types,
                    weight=fragment_weight, batch=fragment_batch,
                    input_width=fragment_input, output_width=fragment_output,
                    output=package,
                )
                size, digest = tree_receipt(package)
                if not 0 < size <= MAX_SHARD_BYTES:
                    raise ValueError(
                        f"{package.name} is {size} bytes; release limit is {MAX_SHARD_BYTES}"
                    )
                shards.append(
                    {
                        "order": order,
                        "path": package.name,
                        "input_name": "input",
                        "output_name": "output",
                        "projection": physical_group,
                        "input_offset": input_offset,
                        "input_width": fragment_input,
                        "output_offset": 0,
                        "output_width": fragment_output,
                        "input_shape": [1, fragment_input, fragment_batch, 1],
                        "input_elements": fragment_batch * fragment_input,
                        "output_elements": fragment_batch * fragment_output,
                        "bytes": size,
                        "sha256": digest,
                        "components": [
                            {
                                "projection": component["projection"],
                                "output_offset": component["output_offset"],
                                "projection_offset": component["projection_offset"],
                                "output_width": component["output_width"],
                            }
                            for component in fragment["components"]
                        ],
                    }
                )
                del fragment_weight
            if official:
                group_weights.clear()
                for raw in raw_paths:
                    raw.unlink()

        by_layer = {}
        for fragment in tail_fragments:
            by_layer.setdefault(fragment["layer"], []).append(fragment)
        for layer, layer_fragments in sorted(by_layer.items()):
            tensor_specs = {
                "gate": (
                    layer_fragments[0]["gate_tensor"],
                    layer_fragments[0]["hidden_width"],
                    int(config["intermediate_size"]),
                ),
                "up": (
                    layer_fragments[0]["up_tensor"],
                    layer_fragments[0]["hidden_width"],
                    int(config["intermediate_size"]),
                ),
                "down": (
                    layer_fragments[0]["down_tensor"],
                    int(config["intermediate_size"]),
                    layer_fragments[0]["hidden_width"],
                ),
            }
            if layer_fragments[0]["head"]:
                tensor_specs["o"] = (
                    layer_fragments[0]["o_tensor"],
                    layer_fragments[0]["attention_width"],
                    layer_fragments[0]["hidden_width"],
                )
                tensor_specs["norm"] = (
                    layer_fragments[0]["norm_tensor"],
                    layer_fragments[0]["hidden_width"],
                    1,
                )
            layer_weights = {}
            raw_paths = []
            for name, (tensor_name, input_width, output_width) in tensor_specs.items():
                if official:
                    raw = extraction / f"tail-{layer:02d}-{name}.f32"
                    extract_official_projection(
                        args.extractor, args.dflash, tensor_name, raw,
                        output_width, input_width,
                    )
                    raw_paths.append(raw)
                    layer_weights[name] = np.memmap(
                        raw, mode="r", dtype="<f4", shape=(output_width, input_width)
                    )
                else:
                    if tensor_name not in weights:
                        raise ValueError(f"missing DFlash tensor {tensor_name}")
                    layer_weights[name] = weights[tensor_name]
            for fragment in layer_fragments:
                offset = fragment["intermediate_offset"]
                width = fragment["intermediate_width"]
                package_order = len(shards) + len(tail_shards)
                package = stage / f"projection-{package_order:03d}.mlpackage"
                export_args = dict(
                    ct=ct, cto=cto, np=np, mb=mb, types=types,
                    gate_weight=layer_weights["gate"][offset:offset + width, :],
                    up_weight=layer_weights["up"][offset:offset + width, :],
                    down_weight=layer_weights["down"][:, offset:offset + width],
                    batch=batch, hidden_width=fragment["hidden_width"],
                    intermediate_width=width, output=package,
                )
                if fragment["head"]:
                    export_fused_tail_head(
                        **export_args,
                        o_weight=layer_weights["o"],
                        norm_weight=np.asarray(layer_weights["norm"]).reshape(-1),
                        rms_norm_eps=float(config["rms_norm_eps"]),
                        attention_width=fragment["attention_width"],
                    )
                else:
                    export_fused_ffn(**export_args)
                size, digest = tree_receipt(package)
                if not 0 < size <= MAX_SHARD_BYTES:
                    raise ValueError(
                        f"{package.name} is {size} bytes; release limit is {MAX_SHARD_BYTES}"
                    )
                input_channels = (
                    fragment["attention_width"] + fragment["hidden_width"]
                    if fragment["head"] else fragment["hidden_width"]
                )
                output_channels = (
                    2 * fragment["hidden_width"]
                    if fragment["head"] else fragment["hidden_width"]
                )
                tail_shards.append(
                    {
                        "order": fragment["order"],
                        "layer": layer,
                        "head": fragment["head"],
                        "path": package.name,
                        "input_name": "input",
                        "output_name": "output",
                        "intermediate_offset": offset,
                        "intermediate_width": width,
                        "input_shape": [1, input_channels, batch, 1],
                        "input_elements": batch * input_channels,
                        "output_elements": batch * output_channels,
                        "bytes": size,
                        "sha256": digest,
                    }
                )
            layer_weights.clear()
            if official:
                for raw in raw_paths:
                    raw.unlink()

        if attention_source is not None:
            attention_manifest, attention_package = attention_source
            destination = stage / "stateful-attention.mlpackage"
            shutil.copytree(attention_package, destination)
            size, digest = tree_receipt(destination)
            if (
                size != attention_manifest["package_bytes"]
                or digest != attention_manifest["package_sha256"]
            ):
                raise ValueError("copied stateful attention package changed identity")
            attention_shards = [
                {
                    "order": layer,
                    "layer": layer,
                    "path": destination.name,
                    "max_context": 1088,
                    "query_size": 16,
                    "chunk": 16,
                    "kv_write_chunk": 16,
                    "kv_join": "split",
                    "state_group_kv_heads": 8,
                    "bytes": size,
                    "sha256": digest,
                }
                for layer in range(int(config["num_hidden_layers"]))
            ]

        for layer, (source_manifest, source_package) in enumerate(
            fused_attention_sources
        ):
            destination = stage / f"fused-attention-{layer}.mlpackage"
            shutil.copytree(source_package, destination)
            size, digest = tree_receipt(destination)
            if (
                size != source_manifest["package_bytes"]
                or digest != source_manifest["package_sha256"]
            ):
                raise ValueError(f"copied fused attention layer {layer} changed identity")
            fused_attention_shards.append(
                {
                    "order": layer,
                    "layer": layer,
                    "path": destination.name,
                    "max_context": 1088,
                    "block_size": 16,
                    "query_size": 16,
                    "kv_write_chunk": 16,
                    "kv_join": "split",
                    "state_group_kv_heads": 4,
                    "bundle_channels": 6144,
                    "bytes": size,
                    "sha256": digest,
                }
            )

        manifest = {
            "version": manifest_version,
            "backend": "public_coreml",
            "compute_units": "CPU_AND_NE",
            "weight_dtype": "int8",
            "projection_operator": "conv1x1",
            "target_identity": args.target_identity,
            "dflash_identity": args.dflash_identity or dflash_identity,
            "dflash_source_format": source_format,
            "extractor_sha256": extractor_identity,
            "assistant_layers": 5,
            "block_size": batch,
            "shards": shards,
            "ffn_shards": [],
            "tail_shards": tail_shards,
            "attention_shards": attention_shards,
            "fused_attention_shards": fused_attention_shards,
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.rename(stage, args.output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(extraction, ignore_errors=True)
    print(args.output / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
