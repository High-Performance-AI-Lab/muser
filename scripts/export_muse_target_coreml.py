#!/usr/bin/env python3
"""Export one real Muse target-layer tail as public-CoreML INT8 shards.

This is the target-decode ANE POC, not the DFlash exporter.  Metal retains
Q/K/V, RoPE, KV mutation, attention, and the sigmoid gate.  Core ML receives
the gated attention result plus the layer residual and executes the large
output/FFN projections with Muse's exact sandwich-norm ordering.  The final
post-FFN norm/residual stays on Metal because it must see the sum of every
down-projection partition.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from export_dflash_coreml import (
    ANE_BOOK_REVISION,
    COREMLTOOLS_VERSION,
    MAX_SHARD_BYTES,
    NUMPY_VERSION,
    file_sha256,
    imports,
    tree_receipt,
)

HIDDEN = 6656
ATTENTION = 4096
INTERMEDIATE = 19968
HEAD_INTERMEDIATE = 8192
POST_NORM_EPS = 1.0e-8
RMS_NORM_EPS = 1.0e-5
ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--extractor",
        type=Path,
        default=ROOT / "target" / "release" / "muser-dflash-extract",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def extract(
    extractor: Path,
    model: Path,
    tensor: str,
    output: Path,
    rows: int,
    columns: int,
) -> None:
    result = subprocess.run(
        [
            str(extractor), "--artifact", str(model), "--raw-tensor",
            "--tensor", tensor, "--output", str(output),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    receipt = json.loads(result.stdout)
    if (
        receipt.get("rows") != rows
        or receipt.get("columns") != columns
        or receipt.get("bytes") != rows * columns * 4
        or output.stat().st_size != rows * columns * 4
    ):
        raise ValueError(f"invalid extraction receipt for {tensor}")


def rms_norm(mb, np, value, scale, epsilon: float, prefix: str):
    variance = mb.reduce_mean(
        x=mb.square(x=value), axes=[1], keep_dims=True, name=f"{prefix}_variance"
    )
    inverse = mb.rsqrt(
        x=mb.add(x=variance, y=np.float32(epsilon)), name=f"{prefix}_inverse_rms"
    )
    return mb.mul(
        x=mb.mul(x=value, y=inverse), y=scale, name=f"{prefix}_normed"
    )


def quantized_model(ct, cto, model):
    quantizer = cto.OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int8", granularity="per_tensor"
    )
    return cto.linear_quantize_weights(
        model, cto.OptimizationConfig(global_config=quantizer)
    )


def export_head(
    *, ct, cto, np, mb, types, output: Path, batch: int,
    o_weight, post_attn_weight, ffn_norm_weight, gate_weight, up_weight, down_weight,
) -> None:
    o_conv = np.asarray(o_weight, dtype=np.float32)[:, :, None, None]
    gate_conv = np.asarray(gate_weight, dtype=np.float32)[:, :, None, None]
    up_conv = np.asarray(up_weight, dtype=np.float32)[:, :, None, None]
    down_conv = np.asarray(down_weight, dtype=np.float32)[:, :, None, None]
    post_attn_scale = np.asarray(post_attn_weight, dtype=np.float32)[None, :, None, None]
    ffn_scale = np.asarray(ffn_norm_weight, dtype=np.float32)[None, :, None, None]

    @mb.program(
        input_specs=[mb.TensorSpec(
            shape=(1, ATTENTION + HIDDEN, batch, 1), dtype=types.fp32
        )],
        opset_version=ct.target.iOS18,
    )
    def program(input):
        attention = mb.slice_by_size(
            x=input, begin=[0, 0, 0, 0], size=[1, ATTENTION, batch, 1],
            name="attention",
        )
        residual = mb.slice_by_size(
            x=input, begin=[0, ATTENTION, 0, 0], size=[1, HIDDEN, batch, 1],
            name="residual",
        )
        projected = mb.conv(
            x=attention, weight=o_conv, pad_type="valid", strides=[1, 1],
            dilations=[1, 1], groups=1, name="output_projection",
        )
        projected = rms_norm(
            mb, np, projected, post_attn_scale, POST_NORM_EPS, "post_attention"
        )
        ffn_input = mb.add(x=residual, y=projected, name="ffn_input")
        normed = rms_norm(mb, np, ffn_input, ffn_scale, RMS_NORM_EPS, "ffn")
        gate = mb.conv(
            x=normed, weight=gate_conv, pad_type="valid", strides=[1, 1],
            dilations=[1, 1], groups=1, name="gate",
        )
        up = mb.conv(
            x=normed, weight=up_conv, pad_type="valid", strides=[1, 1],
            dilations=[1, 1], groups=1, name="up",
        )
        activated = mb.mul(
            x=mb.mul(x=gate, y=mb.sigmoid(x=gate)), y=up, name="activated"
        )
        partial = mb.conv(
            x=activated, weight=down_conv, pad_type="valid", strides=[1, 1],
            dilations=[1, 1], groups=1, name="down_partial",
        )
        return mb.concat(values=[ffn_input, normed, partial], axis=1, name="output")

    model = ct.convert(
        program, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.CPU_AND_NE, compute_precision=ct.precision.FLOAT16,
    )
    model = quantized_model(ct, cto, model)
    model.save(str(output))


def export_tail(
    *, ct, cto, np, mb, types, output: Path, batch: int,
    gate_weight, up_weight, down_weight,
) -> None:
    gate_conv = np.asarray(gate_weight, dtype=np.float32)[:, :, None, None]
    up_conv = np.asarray(up_weight, dtype=np.float32)[:, :, None, None]
    down_conv = np.asarray(down_weight, dtype=np.float32)[:, :, None, None]

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, HIDDEN, batch, 1), dtype=types.fp32)],
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
            x=mb.mul(x=gate, y=mb.sigmoid(x=gate)), y=up, name="activated"
        )
        return mb.conv(
            x=activated, weight=down_conv, pad_type="valid", strides=[1, 1],
            dilations=[1, 1], groups=1, name="output",
        )

    model = ct.convert(
        program, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.CPU_AND_NE, compute_precision=ct.precision.FLOAT16,
    )
    model = quantized_model(ct, cto, model)
    model.save(str(output))


def main() -> int:
    args = parse_args()
    if not 0 <= args.layer < 52 or args.layer % 4 == 3:
        raise ValueError("--layer must select one of Muse's 39 SWA layers")
    if args.batch != 16:
        raise ValueError("the first target-ANE POC is frozen at batch 16")
    plan = {
        "schema": "muser.muse-target-coreml-export-plan.v1",
        "model_sha256": file_sha256(args.model),
        "layer": args.layer,
        "kind": "swa_rope_2048",
        "batch": args.batch,
        "split": [HEAD_INTERMEDIATE, INTERMEDIATE - HEAD_INTERMEDIATE],
        "metal_ops": ["qkvg", "qk_norm", "rope", "kv", "attention", "sigmoid_gate", "post_ffn_norm_residual"],
        "ane_ops": ["output_projection", "post_attention_norm", "ffn_norm", "gate_up", "silu", "down_projection"],
        "toolchain": {
            "coremltools": COREMLTOOLS_VERSION,
            "numpy": NUMPY_VERSION,
            "ane_book_revision": ANE_BOOK_REVISION,
        },
    }
    if args.dry_run:
        print(json.dumps(dict(plan, mode="dry-run", output_created=False), indent=2, sort_keys=True))
        return 0
    if args.output.exists() or args.output.is_symlink():
        raise ValueError(f"output must be absent: {args.output}")
    if not args.extractor.is_file() or args.extractor.is_symlink():
        raise ValueError(f"extractor is absent or unsafe: {args.extractor}")
    ct, cto, np, mb, types = imports()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".muser-target-ane-", dir=args.output.parent.resolve()))
    extraction = Path(tempfile.mkdtemp(prefix="muser-target-f32-"))
    try:
        prefix = f"blk.{args.layer}"
        specs = {
            "o": (f"{prefix}.attn_output.weight", HIDDEN, ATTENTION),
            "post_attn": (f"{prefix}.post_attention_norm.weight", 1, HIDDEN),
            "ffn_norm": (f"{prefix}.ffn_norm.weight", 1, HIDDEN),
            "gate": (f"{prefix}.ffn_gate.weight", INTERMEDIATE, HIDDEN),
            "up": (f"{prefix}.ffn_up.weight", INTERMEDIATE, HIDDEN),
            "down": (f"{prefix}.ffn_down.weight", HIDDEN, INTERMEDIATE),
        }
        weights = {}
        for name, (tensor, rows, columns) in specs.items():
            raw = extraction / f"{name}.f32"
            extract(args.extractor, args.model, tensor, raw, rows, columns)
            weights[name] = np.memmap(raw, mode="r", dtype="<f4", shape=(rows, columns))
        head = stage / "target-tail-head.mlpackage"
        tail = stage / "target-tail-continuation.mlpackage"
        export_head(
            ct=ct, cto=cto, np=np, mb=mb, types=types, output=head, batch=args.batch,
            o_weight=weights["o"],
            post_attn_weight=np.asarray(weights["post_attn"]).reshape(-1),
            ffn_norm_weight=np.asarray(weights["ffn_norm"]).reshape(-1),
            gate_weight=weights["gate"][:HEAD_INTERMEDIATE],
            up_weight=weights["up"][:HEAD_INTERMEDIATE],
            down_weight=weights["down"][:, :HEAD_INTERMEDIATE],
        )
        export_tail(
            ct=ct, cto=cto, np=np, mb=mb, types=types, output=tail, batch=args.batch,
            gate_weight=weights["gate"][HEAD_INTERMEDIATE:],
            up_weight=weights["up"][HEAD_INTERMEDIATE:],
            down_weight=weights["down"][:, HEAD_INTERMEDIATE:],
        )
        packages = []
        for order, package in enumerate((head, tail)):
            size, digest = tree_receipt(package)
            if not 0 < size <= MAX_SHARD_BYTES:
                raise ValueError(f"{package.name} size {size} exceeds {MAX_SHARD_BYTES}")
            packages.append({
                "order": order,
                "path": package.name,
                "bytes": size,
                "sha256": digest,
                "input_shape": [1, ATTENTION + HIDDEN if order == 0 else HIDDEN, args.batch, 1],
                "output_shape": [1, 3 * HIDDEN if order == 0 else HIDDEN, args.batch, 1],
            })
        manifest = dict(
            plan,
            version=1,
            backend="public_coreml",
            compute_units="CPU_AND_NE",
            weight_dtype="int8",
            projection_operator="conv1x1",
            packages=packages,
        )
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
