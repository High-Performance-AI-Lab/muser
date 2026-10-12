#!/usr/bin/env python3
"""Measure one warm public-CoreML prediction per manifest shard.

This is a diagnostic receipt, not a release performance seal. It must run
under ``accelerator_safe.py`` and never changes the artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    if os.environ.get("MUSER_ACCELERATOR_LEASE") != "1":
        raise SystemExit("execution must be a child of accelerator_safe.py")
    args = parse_args()
    import coremltools as ct
    import numpy as np

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = args.manifest.parent.resolve()
    records = []
    groups = (
        ("linear", manifest.get("shards", [])),
        ("ffn", manifest.get("ffn_shards", [])),
        ("tail", manifest.get("tail_shards", [])),
    )
    for family, specs in groups:
        for spec in specs:
            package = (root / spec["path"]).resolve()
            if not package.is_relative_to(root):
                raise ValueError(f"shard escapes artifact root: {spec['path']}")
            model = ct.models.MLModel(
                str(package), compute_units=ct.ComputeUnit.CPU_AND_NE
            )
            shape = tuple(int(value) for value in spec["input_shape"])
            values = np.zeros(shape, dtype=np.float32)
            # First use specializes/warms the public model. Retain it separately
            # and measure exactly one immediately adjacent warm prediction.
            started = time.perf_counter_ns()
            first = model.predict({spec["input_name"]: values})
            first_ns = time.perf_counter_ns() - started
            if spec["output_name"] not in first:
                raise ValueError(f"missing output for {spec['path']}")
            started = time.perf_counter_ns()
            second = model.predict({spec["input_name"]: values})
            warm_ns = time.perf_counter_ns() - started
            output = second.get(spec["output_name"])
            if output is None or int(output.size) != int(spec["output_elements"]):
                raise ValueError(f"output geometry differs for {spec['path']}")
            records.append(
                {
                    "family": family,
                    "path": spec["path"],
                    "projection": spec.get("projection"),
                    "layer": spec.get("layer"),
                    "order": spec["order"],
                    "head": spec.get("head"),
                    "bytes": spec["bytes"],
                    "first_ns": first_ns,
                    "warm_ns": warm_ns,
                }
            )
    receipt = {
        "schema": "muser.coreml-shard-latency.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "compute_units": "CPU_AND_NE",
        "coremltools_version": ct.__version__,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "shard_count": len(records),
        "warm_total_ns": sum(record["warm_ns"] for record in records),
        "records": records,
        "seal_eligible": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
