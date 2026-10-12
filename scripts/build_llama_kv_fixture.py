#!/usr/bin/env python3
"""Build the canonical F16 Muse KV prefix from pinned llama captures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path

LLAMA_REVISION = "89e0aa6fd362617d9073e0dafc18e41241521572"
MODEL_SHA256 = "7e9b74b7c8875e9e265695df9613bf6290f2392e479ce740495a129019c488d8"
METALLIB_SHA256 = "c018ab9c01c18bf79a83272d25973a981138bdeec9ac677e63afef07b21ac033"
PROMPT_FILE_SHA256 = "51f7585267ec0cae2e3be865eea2b874b8e497d5f4271aea276f1f1b90c0f370"
PROMPT_TOKENS_SHA256 = "791f77305adabfcc6ed5741c707fbdd16658a735be6412ea97ba1b014ff096f5"
POSITIONS = 2048
LAYERS = 52
ELEMENTS_PER_TOKEN = 256
CAPTURE_ROWS = 512
CAPTURE_BYTES = CAPTURE_ROWS * ELEMENTS_PER_TOKEN * 4
PLANE_BYTES = POSITIONS * ELEMENTS_PER_TOKEN * 2


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def regular_file(path: Path, expected_bytes: int | None = None) -> None:
    metadata = path.lstat()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"not a retained regular file: {path}")
    if expected_bytes is not None and metadata.st_size != expected_bytes:
        raise ValueError(
            f"{path} has {metadata.st_size} bytes, expected {expected_bytes}"
        )


def f32_to_f16(source: Path) -> bytes:
    regular_file(source, CAPTURE_BYTES)
    raw = source.read_bytes()
    output = bytearray(len(raw) // 2)
    for index, (value,) in enumerate(struct.iter_unpack("<f", raw)):
        struct.pack_into("<e", output, index * 2, value)
    return bytes(output)


def write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
    finally:
        os.close(descriptor)


def selected_sources(layer: int, kind: str) -> list[str]:
    if kind == "key":
        label = "Kcur_normed" if layer % 4 == 3 else "Kcur_rope"
        ordinals = range(4)
    else:
        label = "Vcur"
        ordinals = range(0, 8, 2)
    return [f"prompt.{ordinal}.{label}-{layer}.f32" for ordinal in ordinals]


def build(capture_dir: Path, output_dir: Path, llama_logits: Path) -> dict[str, object]:
    if capture_dir.is_symlink() or not capture_dir.is_dir():
        raise ValueError("capture directory must be a real directory")
    if output_dir.exists():
        raise ValueError("output directory must not already exist")
    regular_file(llama_logits, 202_048 * 4)
    output_dir.mkdir(mode=0o700)
    records: list[dict[str, object]] = []
    for layer in range(LAYERS):
        for kind in ("key", "value"):
            names = selected_sources(layer, kind)
            converted = bytearray()
            source_hashes: dict[str, str] = {}
            for name in names:
                source = capture_dir / name
                converted.extend(f32_to_f16(source))
                source_hashes[name] = digest(source)
            if len(converted) != PLANE_BYTES:
                raise ValueError(f"layer {layer} {kind} produced {len(converted)} bytes")
            filename = f"layer-{layer:02}.{kind}.f16"
            destination = output_dir / filename
            write_exclusive(destination, converted)
            records.append(
                {
                    "bytes": len(converted),
                    "filename": filename,
                    "kind": kind,
                    "layer": layer,
                    "sha256": digest(destination),
                    "source_sha256": source_hashes,
                    "sources": names,
                }
            )
    manifest: dict[str, object] = {
        "elements_per_token": ELEMENTS_PER_TOKEN,
        "encoding": "f16le",
        "llama_full_logits_sha256": digest(llama_logits),
        "llama_revision": LLAMA_REVISION,
        "metallib_sha256": METALLIB_SHA256,
        "model_sha256": MODEL_SHA256,
        "positions": POSITIONS,
        "prompt_file_sha256": PROMPT_FILE_SHA256,
        "prompt_tokens_sha256": PROMPT_TOKENS_SHA256,
        "records": records,
        "schema": "muser.llama-kv-prefix-fixture.v1",
    }
    encoded = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    write_exclusive(output_dir / "manifest.json", encoded)
    directory = os.open(output_dir, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llama-logits", type=Path, required=True)
    options = parser.parse_args()
    manifest = build(options.capture_dir, options.output_dir, options.llama_logits)
    print(
        json.dumps(
            {
                "manifest": str(options.output_dir / "manifest.json"),
                "records": len(manifest["records"]),
                "schema": manifest["schema"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
