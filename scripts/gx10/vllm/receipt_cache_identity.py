#!/usr/bin/env python3
"""Derive a mode-separated target-cache identity for one producer adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path


def derive(model_sha256: str, adapter_sha256: str, producer_mode: str) -> dict[str, object]:
    for name, value in (("model", model_sha256), ("adapter", adapter_sha256)):
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError(f"{name} identity is not lowercase SHA-256")
    if producer_mode not in {"exact", "native"}:
        raise ValueError("producer mode must be exact or native")
    identity = {
        "schema": "muser.nvfp4-target-cache.v1",
        "weight_precision": "nvfp4",
        "model_sha256": model_sha256,
        "producer_adapter_sha256": adapter_sha256,
        "producer_mode": producer_mode,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": "muser.nvfp4-target-cache-receipt.v1",
        "target_cache_identity_sha256": hashlib.sha256(encoded).hexdigest(),
        "identity": identity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--adapter-receipt", required=True, type=Path)
    parser.add_argument("--producer-mode", choices=("exact", "native"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    adapter = json.loads(args.adapter_receipt.read_text(encoding="utf-8"))
    adapter_sha256 = adapter.get("adapter_sha256")
    if not isinstance(adapter_sha256, str):
        parser.error("adapter receipt has no adapter_sha256")
    receipt = derive(args.model_sha256, adapter_sha256, args.producer_mode)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
