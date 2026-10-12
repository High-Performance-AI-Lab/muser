#!/usr/bin/env python3
"""Atomically add proven fused-attention packages to a receipted v7 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import tempfile
from pathlib import Path


def tree_receipt(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    for child in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
        if child.is_symlink():
            raise ValueError(f"package contains symlink: {child}")
        if not child.is_file():
            continue
        relative = child.relative_to(path).as_posix().encode()
        size = child.stat().st_size
        total += size
        digest.update(struct.pack("<Q", len(relative)))
        digest.update(relative)
        digest.update(struct.pack("<Q", size))
        with child.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return total, digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--fused-layers", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError(f"output must be absent: {args.output}")
    manifest = json.loads((args.base / "manifest.json").read_text())
    if (
        manifest.get("version") != 7
        or manifest.get("backend") != "public_coreml"
        or manifest.get("compute_units") != "CPU_AND_NE"
        or len(manifest.get("attention_shards", [])) != 5
        or len(manifest.get("tail_shards", [])) != 10
        or manifest.get("fused_attention_shards")
    ):
        raise ValueError("base artifact is outside the v7 assembly contract")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".muser-ane-v9-", dir=args.output.parent))
    shutil.rmtree(stage)
    try:
        shutil.copytree(args.base, stage)
        fused = []
        for layer in range(5):
            source_root = args.fused_layers / f"layer-{layer}"
            source_manifest = json.loads((source_root / "manifest.json").read_text())
            if (
                source_manifest.get("layer") != layer
                or source_manifest.get("runtime_bundle") is not True
                or source_manifest.get("state_group_kv_heads") != 4
                or source_manifest.get("dflash_sha256") != manifest["dflash_identity"]
            ):
                raise ValueError(f"fused layer {layer} is outside the v9 contract")
            source = source_root / source_manifest["package"]
            size, digest = tree_receipt(source)
            if size != source_manifest["package_bytes"] or digest != source_manifest["package_sha256"]:
                raise ValueError(f"fused layer {layer} differs from its manifest")
            destination = stage / f"fused-attention-{layer}.mlpackage"
            shutil.copytree(source, destination)
            fused.append(
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
        manifest["version"] = 9
        manifest["fused_attention_shards"] = fused
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.rename(stage, args.output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(args.output / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
