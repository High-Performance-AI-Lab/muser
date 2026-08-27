#!/usr/bin/env python3
"""Deterministically repack Muse compressed-tensors NVFP4 as F16 GGUF.

This is the P0 validation artifact, not the P1 native serving format. It
clones metadata and tensor ordering from the pinned Muse kquant GGUF while
replacing every language-model matrix with the exact F16 cast of the NVFP4
checkpoint's decoded values. Work is streamed by tensor and row chunk.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import uuid

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nvfp4_codec import compressed_tensors_scale2, dequantize  # noqa: E402
import receipt_hf_checkpoint  # noqa: E402


GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
ALIGNMENT = 32
TYPE_F32 = 0
TYPE_F16 = 1
TYPE_SIZE = {TYPE_F32: 4, TYPE_F16: 2}
SIMPLE_KV = {
    0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2),
    4: ("I", 4), 5: ("i", 4), 6: ("f", 4), 7: ("?", 1),
    10: ("Q", 8), 11: ("q", 8), 12: ("d", 8),
}
NORMALIZED_F32 = {
    "attn_norm.weight",
    "post_attention_norm.weight",
    "post_ffw_norm.weight",
    "ffn_norm.weight",
    "attn_q_norm.weight",
    "attn_k_norm.weight",
}
MATRIX_NAMES = {
    "ffn_down.weight": "mlp.down_proj",
    "ffn_gate.weight": "mlp.gate_proj",
    "ffn_up.weight": "mlp.up_proj",
    "attn_gate.weight": "self_attn.gate_proj",
    "attn_k.weight": "self_attn.k_proj",
    "attn_output.weight": "self_attn.o_proj",
    "attn_q.weight": "self_attn.q_proj",
    "attn_v.weight": "self_attn.v_proj",
}
NORM_NAMES = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "post_ffw_norm.weight": "post_feedforward_layernorm.weight",
    "ffn_norm.weight": "pre_feedforward_layernorm.weight",
}


@dataclass(frozen=True)
class SafeTensor:
    path: Path
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int

    def array(self) -> np.memmap:
        dtypes = {"U8": "u1", "F8_E4M3": "u1", "F32": "<f4", "BF16": "<u2"}
        if self.dtype not in dtypes:
            raise RuntimeError(f"unsupported safetensors dtype {self.dtype}")
        return np.memmap(
            self.path,
            dtype=dtypes[self.dtype],
            mode="r",
            offset=self.offset,
            shape=self.shape or (1,),
            order="C",
        )


@dataclass(frozen=True)
class GgufTensor:
    name: str
    dims: tuple[int, ...]
    source_type: int
    source_offset: int


@dataclass(frozen=True)
class GgufReference:
    metadata: list[tuple[str, bytes]]
    tensors: list[GgufTensor]
    data_offset: int


def align(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


def read_string(stream) -> str:
    length = struct.unpack("<Q", stream.read(8))[0]
    return stream.read(length).decode("utf-8")


def read_reference(path: Path) -> GgufReference:
    with path.open("rb") as stream:
        magic, version, tensor_count, metadata_count = struct.unpack(
            "<IIQQ", stream.read(24)
        )
        if magic != GGUF_MAGIC or version != GGUF_VERSION:
            raise RuntimeError("reference is not a GGUF v3 artifact")
        metadata: list[tuple[str, bytes]] = []
        for _ in range(metadata_count):
            start = stream.tell()
            key = read_string(stream)
            value_type = struct.unpack("<I", stream.read(4))[0]
            if value_type == 8:
                read_string(stream)
            elif value_type == 9:
                array_type = struct.unpack("<I", stream.read(4))[0]
                length = struct.unpack("<Q", stream.read(8))[0]
                if array_type == 8:
                    for _ in range(length):
                        read_string(stream)
                elif array_type in SIMPLE_KV:
                    stream.seek(length * SIMPLE_KV[array_type][1], os.SEEK_CUR)
                else:
                    raise RuntimeError(f"unsupported GGUF array type {array_type}")
            elif value_type in SIMPLE_KV:
                stream.seek(SIMPLE_KV[value_type][1], os.SEEK_CUR)
            else:
                raise RuntimeError(f"unsupported GGUF metadata type {value_type}")
            end = stream.tell()
            stream.seek(start)
            metadata.append((key, stream.read(end - start)))
        tensors = []
        for _ in range(tensor_count):
            name = read_string(stream)
            dimensions = struct.unpack("<I", stream.read(4))[0]
            dims = struct.unpack(f"<{dimensions}Q", stream.read(8 * dimensions))
            tensor_type, offset = struct.unpack("<IQ", stream.read(12))
            tensors.append(GgufTensor(name, dims, tensor_type, offset))
        data_offset = align(stream.tell())
    return GgufReference(metadata, tensors, data_offset)


def safetensors_inventory(checkpoint: Path) -> dict[str, SafeTensor]:
    inventory: dict[str, SafeTensor] = {}
    for shard in sorted(checkpoint.glob("*.safetensors")):
        with shard.open("rb") as stream:
            header_length = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(header_length))
        data_offset = 8 + header_length
        for name, info in header.items():
            if not isinstance(info, dict) or "dtype" not in info:
                continue
            begin, end = info["data_offsets"]
            if name in inventory:
                raise RuntimeError(f"duplicate safetensors key {name}")
            inventory[name] = SafeTensor(
                shard,
                info["dtype"],
                tuple(info["shape"]),
                data_offset + begin,
                end - begin,
            )
    if not inventory:
        raise RuntimeError("checkpoint has no safetensors inventory")
    return inventory


def hf_name(name: str) -> str | None:
    if name == "token_embd.weight":
        return "model.language_model.embed_tokens.weight"
    if name == "output.weight":
        return "lm_head.weight"
    if name == "output_norm.weight":
        return "model.language_model.norm.weight"
    parts = name.split(".", 2)
    if len(parts) != 3 or parts[0] != "blk":
        raise RuntimeError(f"unknown Muse GGUF tensor {name}")
    layer = int(parts[1])
    suffix = parts[2]
    if suffix in ("attn_q_norm.weight", "attn_k_norm.weight"):
        return None
    tail = MATRIX_NAMES.get(suffix) or NORM_NAMES.get(suffix)
    if tail is None:
        raise RuntimeError(f"unknown Muse GGUF tensor suffix {suffix}")
    return f"model.language_model.layers.{layer}.{tail}"


def output_type(tensor: GgufTensor) -> int:
    if tensor.name == "output_norm.weight":
        return TYPE_F32
    if tensor.name.startswith("blk."):
        suffix = tensor.name.split(".", 2)[2]
        if suffix in NORMALIZED_F32:
            return TYPE_F32
    return TYPE_F16


def expected_shape(tensor: GgufTensor) -> tuple[int, ...]:
    return tuple(reversed(tensor.dims))


def rope_row_order(rows: int, heads: int) -> np.ndarray:
    """Return GGUF output-row -> HF source-row order for RoPE Q/K weights."""
    if heads <= 0 or rows <= 0 or rows % (2 * heads):
        raise ValueError("RoPE row geometry must divide into heads and complex pairs")
    return (
        np.arange(rows, dtype=np.int64)
        .reshape(heads, 2, rows // heads // 2)
        .swapaxes(1, 2)
        .reshape(rows)
    )


def encoded_file_type() -> bytes:
    key = "general.file_type"
    return struct.pack("<Q", len(key)) + key.encode() + struct.pack("<II", 4, 1)


def output_metadata(metadata: list[tuple[str, bytes]]) -> list[bytes]:
    result = []
    for key, raw in metadata:
        if key == "general.quantization_version" or key.startswith("quantize."):
            continue
        result.append(encoded_file_type() if key == "general.file_type" else raw)
    return result


def write_header(stream, reference: GgufReference) -> list[tuple[GgufTensor, int, int]]:
    metadata = output_metadata(reference.metadata)
    rows: list[tuple[GgufTensor, int, int]] = []
    offset = 0
    for tensor in reference.tensors:
        tensor_type = output_type(tensor)
        size = int(np.prod(tensor.dims)) * TYPE_SIZE[tensor_type]
        rows.append((tensor, tensor_type, offset))
        offset += align(size)
    stream.write(
        struct.pack(
            "<IIQQ", GGUF_MAGIC, GGUF_VERSION, len(rows), len(metadata)
        )
    )
    for raw in metadata:
        stream.write(raw)
    for tensor, tensor_type, offset in rows:
        encoded = tensor.name.encode()
        stream.write(struct.pack("<Q", len(encoded)) + encoded)
        stream.write(struct.pack("<I", len(tensor.dims)))
        stream.write(struct.pack(f"<{len(tensor.dims)}Q", *tensor.dims))
        stream.write(struct.pack("<IQ", tensor_type, offset))
    stream.write(b"\0" * (-stream.tell() % ALIGNMENT))
    return rows


def write_padding(stream, size: int) -> None:
    stream.write(b"\0" * (-size % ALIGNMENT))


def write_bf16(
    stream, tensor: SafeTensor, target_type: int, shift: np.float32 = np.float32(0.0)
) -> int:
    if tensor.dtype != "BF16":
        raise RuntimeError(f"expected BF16 source, got {tensor.dtype}")
    source = tensor.array().reshape(-1)
    chunk_elements = 8 * 1024 * 1024
    written = 0
    for start in range(0, source.size, chunk_elements):
        bits = np.asarray(source[start : start + chunk_elements], dtype=np.uint32)
        values = (bits << 16).view(np.float32)
        if shift != np.float32(0.0):
            values = values + shift
        encoded = values.astype("<f4" if target_type == TYPE_F32 else "<f2").tobytes()
        stream.write(encoded)
        written += len(encoded)
    return written


def write_quantized(
    stream,
    inventory: dict[str, SafeTensor],
    base: str,
    shape: tuple[int, ...],
    rope_heads: int | None = None,
) -> int:
    packed = inventory[base + ".weight_packed"]
    scales = inventory[base + ".weight_scale"]
    global_scale = inventory[base + ".weight_global_scale"]
    if (
        packed.dtype != "U8"
        or scales.dtype != "F8_E4M3"
        or global_scale.dtype != "F32"
        or packed.shape != (shape[0], shape[1] // 2)
        or scales.shape != (shape[0], shape[1] // 16)
        or global_scale.shape not in ((), (1,))
    ):
        raise RuntimeError(f"NVFP4 geometry mismatch for {base}")
    packed_array = packed.array()
    scale_array = scales.array()
    row_order = (
        rope_row_order(shape[0], rope_heads)
        if rope_heads is not None
        else np.arange(shape[0], dtype=np.int64)
    )
    global_scale_value = float(global_scale.array().reshape(-1)[0])
    scale2 = compressed_tensors_scale2(global_scale_value)
    written = 0
    for start in range(0, shape[0], 128):
        end = min(start + 128, shape[0])
        source_rows = row_order[start:end]
        values = dequantize(
            np.asarray(packed_array[source_rows]),
            np.asarray(scale_array[source_rows]),
            scale2,
        )
        encoded = values.astype("<f2").tobytes()
        stream.write(encoded)
        written += len(encoded)
    return written


def copy_reference_identity(
    stream,
    reference_path: Path,
    reference: GgufReference,
    tensor: GgufTensor,
    qk_scale_factor: float,
) -> int:
    encoded = reference_architecture_constant(
        reference_path, reference, tensor, qk_scale_factor
    )
    stream.write(encoded)
    return len(encoded)


def reference_architecture_constant(
    reference_path: Path,
    reference: GgufReference,
    tensor: GgufTensor,
    qk_scale_factor: float,
) -> bytes:
    if tensor.source_type != TYPE_F32:
        raise RuntimeError(f"identity Q/K norm is not F32: {tensor.name}")
    size = int(np.prod(tensor.dims)) * TYPE_SIZE[TYPE_F32]
    with reference_path.open("rb") as source:
        source.seek(reference.data_offset + tensor.source_offset)
        encoded = source.read(size)
    values = np.frombuffer(encoded, dtype="<f4")
    expected_value = qk_scale_factor if tensor.name.endswith("attn_q_norm.weight") else 1.0
    expected = np.full(values.shape, expected_value, dtype=np.float32)
    if not np.array_equal(values, expected):
        raise RuntimeError(
            f"materialized Q/K norm disagrees with config for {tensor.name}"
        )
    return encoded


def reference_f32_values(
    reference_path: Path, reference: GgufReference, tensor: GgufTensor
) -> np.ndarray:
    if tensor.source_type != TYPE_F32:
        raise RuntimeError(f"reference tensor is not F32: {tensor.name}")
    size = int(np.prod(tensor.dims)) * TYPE_SIZE[TYPE_F32]
    with reference_path.open("rb") as source:
        source.seek(reference.data_offset + tensor.source_offset)
        encoded = source.read(size)
    if len(encoded) != size:
        raise RuntimeError(f"truncated reference tensor: {tensor.name}")
    return np.frombuffer(encoded, dtype="<f4")


def validate_sources(
    reference_path: Path,
    reference: GgufReference,
    inventory: dict[str, SafeTensor],
    qk_scale_factor: float,
    attention_heads: int,
    kv_attention_heads: int,
) -> dict[str, int]:
    quantized = 0
    bf16 = 0
    identities = 0
    rope_permuted = 0
    shifted_norms = 0
    output_bytes = 0
    for tensor in reference.tensors:
        shape = expected_shape(tensor)
        tensor_type = output_type(tensor)
        output_bytes += align(int(np.prod(tensor.dims)) * TYPE_SIZE[tensor_type])
        source_name = hf_name(tensor.name)
        if source_name is None:
            reference_architecture_constant(
                reference_path, reference, tensor, qk_scale_factor
            )
            identities += 1
            continue
        if tensor.name.startswith("blk.") and tensor.name.split(".", 2)[2] in MATRIX_NAMES:
            packed = inventory.get(source_name + ".weight_packed")
            scales = inventory.get(source_name + ".weight_scale")
            global_scale = inventory.get(source_name + ".weight_global_scale")
            if packed is None or scales is None or global_scale is None:
                raise RuntimeError(f"missing NVFP4 components for {source_name}")
            if (
                packed.dtype != "U8"
                or scales.dtype != "F8_E4M3"
                or global_scale.dtype != "F32"
                or packed.shape != (shape[0], shape[1] // 2)
                or scales.shape != (shape[0], shape[1] // 16)
                or global_scale.shape not in ((), (1,))
            ):
                raise RuntimeError(f"NVFP4 geometry mismatch for {source_name}")
            suffix = tensor.name.split(".", 2)[2]
            if suffix == "attn_q.weight":
                rope_row_order(shape[0], attention_heads)
                rope_permuted += 1
            elif suffix == "attn_k.weight":
                rope_row_order(shape[0], kv_attention_heads)
                rope_permuted += 1
            quantized += 1
            continue
        source = inventory.get(source_name)
        if source is None:
            raise RuntimeError(f"missing BF16 tensor {source_name}")
        if source.dtype != "BF16" or source.shape != shape:
            raise RuntimeError(
                f"BF16 tensor mismatch for {source_name}: {source.dtype} {source.shape} != {shape}"
            )
        if tensor.name.startswith("blk.") and tensor.name.split(".", 2)[2] in NORM_NAMES:
            raw = np.asarray(source.array().reshape(-1), dtype=np.uint32)
            shifted = (raw << 16).view(np.float32) + np.float32(1.0)
            if not np.array_equal(
                shifted,
                reference_f32_values(reference_path, reference, tensor),
            ):
                raise RuntimeError(
                    f"Muse unit-offset norm disagrees with reference: {tensor.name}"
                )
            shifted_norms += 1
        bf16 += 1
    return {
        "tensors": len(reference.tensors),
        "quantized_matrices": quantized,
        "bf16_tensors": bf16,
        "identity_qk_norms": identities,
        "rope_permuted_qk_matrices": rope_permuted,
        "unit_offset_layer_norms": shifted_norms,
        "aligned_tensor_bytes": output_bytes,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_receipt(path: Path, payload: dict[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to replace receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        publish_temporary_exclusive(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def publish_temporary_exclusive(temporary: Path, destination: Path) -> None:
    """Atomically expose a complete same-filesystem file without clobbering."""
    os.link(temporary, destination)
    temporary.unlink()
    directory = os.open(
        destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def read_checkpoint_receipt(path: Path | None, checkpoint: Path) -> dict | None:
    if path is None:
        return None
    receipt = json.loads(path.read_text())
    if receipt.get("schema") != "muser.hf-checkpoint.receipt.v1":
        raise RuntimeError("checkpoint receipt has the wrong schema")
    if Path(receipt["checkpoint"]).resolve() != checkpoint.resolve():
        raise RuntimeError("checkpoint receipt path does not match --checkpoint")
    expected_rows = receipt.get("files")
    if not isinstance(expected_rows, list) or not expected_rows:
        raise RuntimeError("checkpoint receipt omitted its file inventory")
    expected: dict[str, dict[str, object]] = {}
    for row in expected_rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise RuntimeError("checkpoint receipt has an invalid file row")
        if row["path"] in expected:
            raise RuntimeError("checkpoint receipt has a duplicate file path")
        expected[row["path"]] = row
    actual = receipt_hf_checkpoint.checkpoint_files(checkpoint.resolve())
    if set(actual) != set(expected):
        raise RuntimeError("checkpoint contents differ from the receipt file set")
    verified_rows: list[dict[str, object]] = []
    for name in sorted(expected):
        wanted = expected[name]
        file = actual[name]
        size = file.stat().st_size
        digest = sha256(file)
        if size != wanted.get("size") or digest != wanted.get("sha256"):
            raise RuntimeError(f"checkpoint file differs from receipt: {name}")
        verified_rows.append({"path": name, "size": size, "sha256": digest})
    if receipt_hf_checkpoint.artifact_digest(verified_rows) != receipt.get(
        "artifact_sha256"
    ):
        raise RuntimeError("checkpoint artifact digest differs from receipt")
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-gguf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint-receipt", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.receipt is not None and args.checkpoint_receipt is None:
        raise SystemExit("--receipt requires --checkpoint-receipt")
    if args.check and args.receipt is not None:
        raise SystemExit("--receipt cannot be combined with --check")
    if args.out.exists() or args.out.is_symlink():
        raise SystemExit(f"refusing to replace output: {args.out}")
    if args.receipt is not None and (args.receipt.exists() or args.receipt.is_symlink()):
        raise SystemExit(f"refusing to replace receipt: {args.receipt}")
    checkpoint_receipt = read_checkpoint_receipt(
        args.checkpoint_receipt, args.checkpoint
    )
    reference = read_reference(args.reference_gguf)
    inventory = safetensors_inventory(args.checkpoint)
    config = json.loads((args.checkpoint / "config.json").read_text())
    text_config = config["text_config"]
    qk_scale_factor = float(text_config["qk_scale_factor"])
    attention_heads = int(text_config["num_attention_heads"])
    kv_attention_heads = int(text_config["num_key_value_heads"])
    if not np.isfinite(qk_scale_factor) or qk_scale_factor <= 0:
        raise RuntimeError("checkpoint qk_scale_factor must be finite and positive")
    validation = validate_sources(
        args.reference_gguf,
        reference,
        inventory,
        qk_scale_factor,
        attention_heads,
        kv_attention_heads,
    )
    if args.check:
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.parent / f".{args.out.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            rows = write_header(stream, reference)
            for index, (tensor, target_type, offset) in enumerate(rows):
                expected_position = rows[0][2] + offset
                data_start = stream.tell() if index == 0 else data_start
                if stream.tell() != data_start + expected_position:
                    raise RuntimeError(f"output offset mismatch before {tensor.name}")
                shape = expected_shape(tensor)
                source_name = hf_name(tensor.name)
                if source_name is None:
                    written = copy_reference_identity(
                        stream,
                        args.reference_gguf,
                        reference,
                        tensor,
                        qk_scale_factor,
                    )
                elif tensor.name.startswith("blk.") and tensor.name.split(".", 2)[2] in MATRIX_NAMES:
                    suffix = tensor.name.split(".", 2)[2]
                    rope_heads = (
                        attention_heads
                        if suffix == "attn_q.weight"
                        else kv_attention_heads
                        if suffix == "attn_k.weight"
                        else None
                    )
                    written = write_quantized(
                        stream, inventory, source_name, shape, rope_heads
                    )
                else:
                    source = inventory[source_name]
                    if source.shape != shape:
                        raise RuntimeError(
                            f"shape mismatch for {source_name}: {source.shape} != {shape}"
                        )
                    shift = (
                        np.float32(1.0)
                        if tensor.name.startswith("blk.")
                        and tensor.name.split(".", 2)[2] in NORM_NAMES
                        else np.float32(0.0)
                    )
                    written = write_bf16(stream, source, target_type, shift)
                wanted = int(np.prod(tensor.dims)) * TYPE_SIZE[target_type]
                if written != wanted:
                    raise RuntimeError(
                        f"encoded size mismatch for {tensor.name}: {written} != {wanted}"
                    )
                write_padding(stream, written)
                print(
                    f"[{index + 1}/{len(rows)}] {tensor.name} -> "
                    f"{'F32' if target_type == TYPE_F32 else 'F16'}",
                    flush=True,
                )
            stream.flush()
            os.fsync(stream.fileno())
        publish_temporary_exclusive(temporary, args.out)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    result: dict[str, object] = {
        "schema": "muser.nvfp4-f16-gguf-repack.v4",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_artifact_sha256": (
            checkpoint_receipt.get("artifact_sha256") if checkpoint_receipt else None
        ),
        "checkpoint_receipt": (
            str(args.checkpoint_receipt.resolve()) if args.checkpoint_receipt else None
        ),
        "checkpoint_receipt_sha256": (
            sha256(args.checkpoint_receipt) if args.checkpoint_receipt else None
        ),
        "reference_gguf": str(args.reference_gguf.resolve()),
        "reference_gguf_bytes": args.reference_gguf.stat().st_size,
        "reference_gguf_sha256": sha256(args.reference_gguf),
        "out": str(args.out.resolve()),
        "bytes": args.out.stat().st_size,
        "sha256": sha256(args.out),
        "tensors": len(reference.tensors),
        "metadata_entries": len(output_metadata(reference.metadata)),
        "validation": validation,
        "scale_contract": {
            "checkpoint_field": "weight_global_scale",
            "checkpoint_semantics": "inverse multiplicative tensor scale",
            "decode_scale2": "float32(1.0) / float32(weight_global_scale)",
            "reference_commit": "7347430f4466d4f55cfb841974ee64b80fc18d93",
            "reference_file_sha256": "79fdfc9ee63106241a50989a7db70506a3e49f543bd80f96108333bcef663f3e",
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
            "targets": [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "post_feedforward_layernorm.weight",
                "pre_feedforward_layernorm.weight"
            ],
            "final_norm_shift": 0.0,
            "reference_commit": "8918deaa8ea79ad859dd73ab66f4c452fa70c4ce",
            "reference_file": "conversion/muse_glimmer.py",
            "reference_file_sha256": "dd9e86a74fd3e6e90ebc74b7185b5d12b47ae7935b3464832f82f5f612ba4474",
        },
    }
    if args.receipt is not None:
        write_receipt(args.receipt, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
