#!/usr/bin/env python3
"""Export all five proven sg4 fused-attention layer packages atomically."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dflash", required=True, type=Path)
    parser.add_argument("--extractor", required=True, type=Path)
    parser.add_argument("--layer-zero", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError(f"output must be absent: {args.output}")
    source_manifest = json.loads((args.layer_zero / "manifest.json").read_text())
    if source_manifest.get("layer") != 0 or source_manifest.get("state_group_kv_heads") != 4:
        raise ValueError("layer-zero seed is outside the sg4 contract")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".muser-fused-layers-", dir=args.output.parent))
    exporter = Path(__file__).with_name("export_dflash_stateful_attention_coreml.py")
    try:
        shutil.copytree(args.layer_zero, stage / "layer-0")
        for layer in range(1, 5):
            subprocess.run(
                [
                    os.environ["MUSER_COREML_PYTHON"],
                    str(exporter),
                    "--dflash", str(args.dflash),
                    "--extractor", str(args.extractor),
                    "--output", str(stage / f"layer-{layer}"),
                    "--layer", str(layer),
                    "--max-context", "1088",
                    "--query-size", "16",
                    "--attention-query-chunk", "4",
                    "--kv-write-chunk", "16",
                    "--attention-op", "manual",
                    "--kv-join", "split",
                    "--state-group-kv-heads", "4",
                    "--runtime-bundle",
                ],
                check=True,
            )
        os.rename(stage, args.output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
