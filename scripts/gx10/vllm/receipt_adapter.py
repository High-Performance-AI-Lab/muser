#!/usr/bin/env python3
"""Derive the closed NVFP4 producer-adapter identity from an image receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


SOURCE_KEYS = {
    "connector_sha256": "scripts/gx10/vllm/muser_vllm/connector.py",
    "dflash_capture_sha256": "scripts/gx10/vllm/muser_vllm/dflash_capture.py",
    "exact_attention_sha256": "scripts/gx10/vllm/muser_vllm/exact_attention.py",
    "exact_fp4_quant_sha256": "scripts/gx10/vllm/muser_vllm/exact_fp4_quant.py",
    "exact_fp4_mm_sha256": "scripts/gx10/vllm/muser_vllm/exact_fp4_mm.py",
    "exact_rms_norm_sha256": "scripts/gx10/vllm/muser_vllm/exact_rms_norm.py",
    "exact_rope_sha256": "scripts/gx10/vllm/muser_vllm/exact_rope.py",
    "exact_swiglu_sha256": "scripts/gx10/vllm/muser_vllm/exact_swiglu.py",
    "handoff_sender_sha256": "scripts/gx10/llamacpp/muser_v2_send.py",
    "native_capture_sha256": "scripts/gx10/vllm/muser_vllm/native_capture.py",
    "packing_sha256": "scripts/gx10/vllm/muser_vllm/packing.py",
    "rope_cache_sha256": "scripts/gx10/vllm/muser_vllm/rope_cache.py",
}


def canonical_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derive_identity(image_receipt: dict[str, Any]) -> dict[str, str]:
    if image_receipt.get("schema") != "muser.spark-nvfp4-image-rebuild.v1":
        raise ValueError("image receipt schema is not the pinned NVFP4 rebuild schema")
    image_id = image_receipt.get("image_id")
    commit = image_receipt.get("vllm_commit")
    sources = image_receipt.get("sources")
    if not isinstance(image_id, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ValueError("image receipt has no content-addressed image ID")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("image receipt has no exact vLLM commit")
    if not isinstance(sources, dict):
        raise ValueError("image receipt source manifest is missing")
    identity = {
        "schema": "muser.spark-nvfp4-adapter.v1",
        "vllm_commit": commit,
        "image_id": image_id,
    }
    for output_name, source_name in SOURCE_KEYS.items():
        digest = sources.get(source_name)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"image receipt source digest is missing: {source_name}")
        identity[output_name] = digest
    return identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    image_receipt = json.loads(args.image_receipt.read_text(encoding="utf-8"))
    identity = derive_identity(image_receipt)
    receipt = {
        "schema": "muser.spark-nvfp4-adapter-receipt.v1",
        "adapter_sha256": canonical_sha256(identity),
        "identity": identity,
        "image_receipt_sha256": sha256_file(args.image_receipt),
    }
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
