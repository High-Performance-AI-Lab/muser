#!/usr/bin/env python3
"""Repack Muse compressed-tensors NVFP4 into muser's native 4.5-bit GGUF.

The primary tensor remains packed E2M1 (two weights/byte). Each matrix has a
bound raw-E4M3FN companion (one scale/16 weights) and one f32 weight-scale
companion. W4A4 checkpoints additionally carry the exact compressed-tensors
input-global-scale divisor for every matrix. No large matrix is dequantized or
requantized. Q/K rows receive the same Muse RoPE permutation proven in P0; all
non-quantized tensors follow the proven P0 F16/F32 conversion and +1 layer-norm
contract.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import struct
import uuid

import numpy as np

import nvfp4_to_f16_gguf as p0
from nvfp4_codec import compressed_tensors_scale2


TYPE_NVFP4_E2M1 = 1000
TYPE_F8_E4M3FN = 1001
SCALE_SUFFIX = ".nvfp4_scale"
SCALE2_SUFFIX = ".nvfp4_scale2"
INPUT_SCALE_INV_SUFFIX = ".nvfp4_input_scale_inv"
SCHEMA = "muser.nvfp4-native-gguf.v2"
RMS_EPS_KEY = "muse-glimmer.attention.layer_norm_rms_epsilon"


@dataclass(frozen=True)
class NativeTensor:
    name: str
    dims: tuple[int, ...]
    tensor_type: int
    offset: int
    size: int
    source: p0.GgufTensor
    component: str


def encoded_string_metadata(key: str, value: str) -> bytes:
    key_bytes = key.encode()
    value_bytes = value.encode()
    return (
        struct.pack("<Q", len(key_bytes))
        + key_bytes
        + struct.pack("<I", 8)
        + struct.pack("<Q", len(value_bytes))
        + value_bytes
    )


def reference_f32_metadata(reference: p0.GgufReference, key: str) -> np.float32:
    """Decode one required scalar-f32 value from retained GGUF metadata."""
    for candidate, raw in reference.metadata:
        if candidate != key:
            continue
        offset = 8 + len(key.encode())
        value_type = struct.unpack_from("<I", raw, offset)[0]
        if value_type != 6:
            raise RuntimeError(f"reference metadata {key} is not scalar f32")
        return np.float32(struct.unpack_from("<f", raw, offset + 4)[0])
    raise RuntimeError(f"reference GGUF is missing required metadata {key}")


def validate_rms_epsilon(
    reference: p0.GgufReference, text_config: dict[str, object]
) -> dict[str, object]:
    if "rms_norm_eps" not in text_config:
        raise RuntimeError("checkpoint text_config is missing rms_norm_eps")
    checkpoint = np.float32(text_config["rms_norm_eps"])
    artifact = reference_f32_metadata(reference, RMS_EPS_KEY)
    checkpoint_bits = int(checkpoint.view(np.uint32))
    artifact_bits = int(artifact.view(np.uint32))
    if not np.isfinite(checkpoint) or checkpoint <= 0:
        raise RuntimeError("checkpoint rms_norm_eps must be finite and positive")
    if checkpoint_bits != artifact_bits:
        raise RuntimeError(
            "checkpoint rms_norm_eps differs from reference GGUF metadata: "
            f"0x{checkpoint_bits:08x} != 0x{artifact_bits:08x}"
        )
    return {
        "checkpoint_field": "text_config.rms_norm_eps",
        "gguf_field": RMS_EPS_KEY,
        "value_f32": float(checkpoint),
        "bits": f"0x{checkpoint_bits:08x}",
    }


def native_metadata(
    reference: p0.GgufReference,
    source_artifact_sha256: str,
    activation_precision: str = "f16",
) -> list[bytes]:
    if activation_precision not in ("f16", "nvfp4"):
        raise ValueError(f"unsupported activation precision {activation_precision}")
    metadata = p0.output_metadata(reference.metadata)
    metadata.extend(
        [
            encoded_string_metadata("muser.weight_precision", "nvfp4"),
            encoded_string_metadata(
                "muser.activation_precision", activation_precision
            ),
            encoded_string_metadata("muser.nvfp4.schema", SCHEMA),
            encoded_string_metadata(
                "muser.nvfp4.source_artifact_sha256", source_artifact_sha256
            ),
        ]
    )
    return metadata


def native_rows(
    reference: p0.GgufReference, activation_precision: str = "f16"
) -> list[NativeTensor]:
    if activation_precision not in ("f16", "nvfp4"):
        raise ValueError(f"unsupported activation precision {activation_precision}")
    rows: list[NativeTensor] = []
    offset = 0
    for tensor in reference.tensors:
        shape = p0.expected_shape(tensor)
        is_matrix = (
            tensor.name.startswith("blk.")
            and tensor.name.split(".", 2)[2] in p0.MATRIX_NAMES
        )
        if is_matrix:
            if len(tensor.dims) != 2 or tensor.dims[0] % 16:
                raise RuntimeError(f"invalid NVFP4 GGUF geometry for {tensor.name}")
            logical = int(np.prod(tensor.dims))
            components = [
                (tensor.name, tensor.dims, TYPE_NVFP4_E2M1, logical // 2, "packed"),
                (
                    tensor.name + SCALE_SUFFIX,
                    (tensor.dims[0] // 16, tensor.dims[1]),
                    TYPE_F8_E4M3FN,
                    logical // 16,
                    "block_scale",
                ),
                (tensor.name + SCALE2_SUFFIX, (1,), p0.TYPE_F32, 4, "scale2"),
            ]
            if activation_precision == "nvfp4":
                components.append(
                    (
                        tensor.name + INPUT_SCALE_INV_SUFFIX,
                        (1,),
                        p0.TYPE_F32,
                        4,
                        "input_scale_inv",
                    )
                )
        else:
            tensor_type = p0.output_type(tensor)
            components = [
                (
                    tensor.name,
                    tensor.dims,
                    tensor_type,
                    int(np.prod(tensor.dims)) * p0.TYPE_SIZE[tensor_type],
                    "plain",
                )
            ]
        for name, dims, tensor_type, size, component in components:
            rows.append(
                NativeTensor(name, tuple(dims), tensor_type, offset, size, tensor, component)
            )
            offset += p0.align(size)
    return rows


def write_header(
    stream,
    reference: p0.GgufReference,
    source_artifact_sha256: str,
    activation_precision: str,
) -> list[NativeTensor]:
    rows = native_rows(reference, activation_precision)
    metadata = native_metadata(reference, source_artifact_sha256, activation_precision)
    stream.write(
        struct.pack("<IIQQ", p0.GGUF_MAGIC, p0.GGUF_VERSION, len(rows), len(metadata))
    )
    for raw in metadata:
        stream.write(raw)
    for row in rows:
        name = row.name.encode()
        stream.write(struct.pack("<Q", len(name)) + name)
        stream.write(struct.pack("<I", len(row.dims)))
        stream.write(struct.pack(f"<{len(row.dims)}Q", *row.dims))
        stream.write(struct.pack("<IQ", row.tensor_type, row.offset))
    stream.write(b"\0" * (-stream.tell() % p0.ALIGNMENT))
    return rows


def matrix_sources(
    inventory: dict[str, p0.SafeTensor], source_name: str, shape: tuple[int, ...]
) -> tuple[p0.SafeTensor, p0.SafeTensor, p0.SafeTensor, p0.SafeTensor | None]:
    packed = inventory[source_name + ".weight_packed"]
    scales = inventory[source_name + ".weight_scale"]
    global_scale = inventory[source_name + ".weight_global_scale"]
    input_scale_inv = inventory.get(source_name + ".input_global_scale")
    if (
        packed.dtype != "U8"
        or scales.dtype != "F8_E4M3"
        or global_scale.dtype != "F32"
        or packed.shape != (shape[0], shape[1] // 2)
        or scales.shape != (shape[0], shape[1] // 16)
        or global_scale.shape not in ((), (1,))
    ):
        raise RuntimeError(f"NVFP4 geometry mismatch for {source_name}")
    if np.isin(np.asarray(scales.array()), np.array([0x7F, 0xFF], dtype=np.uint8)).any():
        raise RuntimeError(f"NVFP4 block scales contain E4M3FN NaN for {source_name}")
    compressed_tensors_scale2(float(global_scale.array().reshape(-1)[0]))
    if input_scale_inv is not None:
        if input_scale_inv.dtype != "F32" or input_scale_inv.shape not in ((), (1,)):
            raise RuntimeError(f"NVFP4 input scale geometry mismatch for {source_name}")
        value = float(input_scale_inv.array().reshape(-1)[0])
        if not np.isfinite(value) or value <= 0.0:
            raise RuntimeError(
                f"NVFP4 input scale for {source_name} must be finite and positive"
            )
    return packed, scales, global_scale, input_scale_inv


def activation_precision(
    reference: p0.GgufReference, inventory: dict[str, p0.SafeTensor]
) -> str:
    present: list[str] = []
    absent: list[str] = []
    for tensor in reference.tensors:
        if not (
            tensor.name.startswith("blk.")
            and tensor.name.split(".", 2)[2] in p0.MATRIX_NAMES
        ):
            continue
        source_name = p0.hf_name(tensor.name)
        assert source_name is not None
        target = (
            present
            if source_name + ".input_global_scale" in inventory
            else absent
        )
        target.append(source_name)
    if present and absent:
        raise RuntimeError(
            "mixed NVFP4 activation precision: input_global_scale is absent for "
            + ", ".join(absent[:8])
        )
    return "nvfp4" if present else "f16"


def row_order(tensor: p0.GgufTensor, shape: tuple[int, ...], q_heads: int, k_heads: int):
    suffix = tensor.name.split(".", 2)[2]
    heads = q_heads if suffix == "attn_q.weight" else k_heads if suffix == "attn_k.weight" else None
    return p0.rope_row_order(shape[0], heads) if heads is not None else np.arange(shape[0])


def write_matrix_component(
    stream,
    row: NativeTensor,
    inventory: dict[str, p0.SafeTensor],
    q_heads: int,
    k_heads: int,
) -> int:
    shape = p0.expected_shape(row.source)
    source_name = p0.hf_name(row.source.name)
    assert source_name is not None
    packed, scales, global_scale, input_scale_inv = matrix_sources(
        inventory, source_name, shape
    )
    order = row_order(row.source, shape, q_heads, k_heads)
    if row.component == "packed":
        encoded = np.asarray(packed.array()[order], dtype=np.uint8).tobytes()
    elif row.component == "block_scale":
        encoded = np.asarray(scales.array()[order], dtype=np.uint8).tobytes()
    elif row.component == "scale2":
        scale2 = compressed_tensors_scale2(float(global_scale.array().reshape(-1)[0]))
        encoded = struct.pack("<f", float(scale2))
    elif row.component == "input_scale_inv":
        if input_scale_inv is None:
            raise RuntimeError(f"missing NVFP4 input scale for {source_name}")
        encoded = struct.pack(
            "<f", float(input_scale_inv.array().reshape(-1)[0])
        )
    else:
        raise RuntimeError(f"unknown matrix component {row.component}")
    stream.write(encoded)
    return len(encoded)


def write_plain_component(
    stream,
    row: NativeTensor,
    reference_path: Path,
    reference: p0.GgufReference,
    inventory: dict[str, p0.SafeTensor],
    qk_scale_factor: float,
) -> int:
    tensor = row.source
    source_name = p0.hf_name(tensor.name)
    if source_name is None:
        return p0.copy_reference_identity(
            stream, reference_path, reference, tensor, qk_scale_factor
        )
    shift = (
        np.float32(1.0)
        if tensor.name.startswith("blk.")
        and tensor.name.split(".", 2)[2] in p0.NORM_NAMES
        else np.float32(0.0)
    )
    return p0.write_bf16(stream, inventory[source_name], row.tensor_type, shift)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-gguf", type=Path, required=True)
    parser.add_argument("--checkpoint-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.out.exists() or args.out.is_symlink():
        raise SystemExit(f"refusing to replace output: {args.out}")
    if args.receipt.exists() or args.receipt.is_symlink():
        raise SystemExit(f"refusing to replace receipt: {args.receipt}")
    checkpoint_receipt = p0.read_checkpoint_receipt(args.checkpoint_receipt, args.checkpoint)
    assert checkpoint_receipt is not None
    artifact_sha = checkpoint_receipt["artifact_sha256"]
    reference = p0.read_reference(args.reference_gguf)
    inventory = p0.safetensors_inventory(args.checkpoint)
    activation = activation_precision(reference, inventory)
    config = json.loads((args.checkpoint / "config.json").read_text())
    text = config["text_config"]
    rms_norm_contract = validate_rms_epsilon(reference, text)
    qk_scale_factor = float(text["qk_scale_factor"])
    q_heads = int(text["num_attention_heads"])
    k_heads = int(text["num_key_value_heads"])
    base_validation = p0.validate_sources(
        args.reference_gguf,
        reference,
        inventory,
        qk_scale_factor,
        q_heads,
        k_heads,
    )
    for tensor in reference.tensors:
        if tensor.name.startswith("blk.") and tensor.name.split(".", 2)[2] in p0.MATRIX_NAMES:
            source_name = p0.hf_name(tensor.name)
            assert source_name is not None
            matrix_sources(inventory, source_name, p0.expected_shape(tensor))
    rows = native_rows(reference, activation)
    validation = {
        **base_validation,
        "native_tensors": len(rows),
        "native_payload_bytes": sum(p0.align(row.size) for row in rows),
        "nvfp4_bits_per_weight": 4.5,
        "activation_precision": activation,
        "rms_norm_contract": rms_norm_contract,
    }
    if args.check:
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.parent / f".{args.out.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            rows = write_header(stream, reference, artifact_sha, activation)
            data_start = stream.tell()
            for index, row in enumerate(rows):
                if stream.tell() != data_start + row.offset:
                    raise RuntimeError(f"output offset mismatch before {row.name}")
                written = (
                    write_matrix_component(stream, row, inventory, q_heads, k_heads)
                    if row.component != "plain"
                    else write_plain_component(
                        stream,
                        row,
                        args.reference_gguf,
                        reference,
                        inventory,
                        qk_scale_factor,
                    )
                )
                if written != row.size:
                    raise RuntimeError(
                        f"encoded size mismatch for {row.name}: {written} != {row.size}"
                    )
                p0.write_padding(stream, written)
                print(f"[{index + 1}/{len(rows)}] {row.name} -> {row.component}", flush=True)
            stream.flush()
            os.fsync(stream.fileno())
        p0.publish_temporary_exclusive(temporary, args.out)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    receipt = {
        "schema": SCHEMA,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_artifact_sha256": artifact_sha,
        "checkpoint_receipt": str(args.checkpoint_receipt.resolve()),
        "checkpoint_receipt_sha256": p0.sha256(args.checkpoint_receipt),
        "reference_gguf": str(args.reference_gguf.resolve()),
        "reference_gguf_sha256": p0.sha256(args.reference_gguf),
        "out": str(args.out.resolve()),
        "bytes": args.out.stat().st_size,
        "sha256": p0.sha256(args.out),
        "validation": validation,
        "format": {
            "packed_type_id": TYPE_NVFP4_E2M1,
            "block_scale_type_id": TYPE_F8_E4M3FN,
            "block_size": 16,
            "scale2": "float32(1.0) / float32(weight_global_scale)",
            "matrix_companions": [SCALE_SUFFIX, SCALE2_SUFFIX],
            "activation_precision": activation,
            "input_scale_inv": (
                "checkpoint input_global_scale; dynamic group-16 E2M1 divisor"
                if activation == "nvfp4"
                else None
            ),
            "activation_companion": (
                INPUT_SCALE_INV_SUFFIX if activation == "nvfp4" else None
            ),
        },
        "rope_permutation": "reshape(heads,2,rows/heads/2,...).swapaxes(1,2).reshape(original)",
        "layer_norm_shift": 1.0,
    }
    p0.write_receipt(args.receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
