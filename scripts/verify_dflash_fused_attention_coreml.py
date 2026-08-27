#!/usr/bin/env python3
"""Run a deterministic two-round smoke of one fused DFlash CoreML layer."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from datetime import datetime, timezone
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
        digest.update(child.read_bytes())
    return total, digest.hexdigest()


def inputs(np, round_index: int) -> dict[str, object]:
    rng = np.random.default_rng(0x4D555345 + round_index)
    block, hidden, kv_width, half, maximum = 16, 6656, 1024, 64, 1088
    start = round_index * block
    position = np.arange(start, start + block, dtype=np.float64)[:, None]
    frequency = np.power(10_000.0, -2.0 * np.arange(half) / 128.0)[None, :]
    angle = position * frequency
    attention_mask = np.full((1, 1, block, maximum + block), -10_000.0, np.float16)
    attention_mask[:, :, :, : start + block] = 0.0
    attention_mask[:, :, :, maximum:] = 0.0
    write_mask = np.zeros((1, 1, maximum, block), np.float16)
    for row in range(block):
        write_mask[0, 0, start + row, row] = 1.0
    return {
        "noise_hidden": rng.standard_normal((1, hidden, block, 1)).astype(np.float16) * 0.01,
        "query_selector": np.eye(block, dtype=np.float16).reshape(1, 1, block, block),
        "target_projected": rng.standard_normal((1, hidden, block, 1)).astype(np.float16) * 0.01,
        "target_mask": np.ones((1, 1, block, 1), np.float16),
        "target_rope_cos": np.cos(angle).astype(np.float16),
        "target_rope_sin": np.sin(angle).astype(np.float16),
        "noise_rope_cos": np.cos(angle + block * frequency).astype(np.float16),
        "noise_rope_sin": np.sin(angle + block * frequency).astype(np.float16),
        "replay_target_key": np.zeros((1, kv_width, block, 1), np.float16),
        "replay_target_value": np.zeros((1, kv_width, block, 1), np.float16),
        "replay_mode": np.zeros((1, 1, 1, 1), np.float16),
        "attention_mask": attention_mask,
        "kv_write_mask": write_mask,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    import coremltools as ct
    import numpy as np

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "muser.dflash-stateful-attention-export.v1"
        or manifest.get("mode") != "exported"
        or manifest.get("runtime_bundle") is not True
        or manifest.get("state_group_kv_heads") != 4
        or manifest.get("output") != "attention_target_kv_bundle"
    ):
        raise ValueError("manifest is outside the fused runtime smoke contract")
    package = args.manifest.parent / manifest["package"]
    size, digest = tree_receipt(package)
    if size != manifest["package_bytes"] or digest != manifest["package_sha256"]:
        raise ValueError("fused package differs from its manifest")

    model = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_AND_NE)
    states = [model.make_state(), model.make_state()]
    rounds = []
    for round_index in range(2):
        values = inputs(np, round_index)
        outputs = [
            np.asarray(model.predict(values, state=state)[manifest["output"]])
            for state in states
        ]
        exact = np.array_equal(outputs[0], outputs[1])
        finite = bool(np.isfinite(outputs[0]).all())
        nonzero = bool(np.any(outputs[0] != 0))
        shape = list(outputs[0].shape)
        if not exact or not finite or not nonzero or shape != [1, 6144, 16, 1]:
            raise RuntimeError(
                f"round {round_index} failed: exact={exact}, finite={finite}, "
                f"nonzero={nonzero}, shape={shape}"
            )
        rounds.append(
            {
                "round": round_index,
                "bit_exact_across_independent_states": exact,
                "finite": finite,
                "nonzero": nonzero,
                "shape": shape,
                "output_sha256": hashlib.sha256(outputs[0].tobytes()).hexdigest(),
            }
        )
    receipt = {
        "schema": "muser.dflash-fused-attention-runtime-smoke.v1",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "compute_units": "CPU_AND_NE",
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "package_sha256": digest,
        "state_group_kv_heads": 4,
        "status": "passed",
        "rounds": rounds,
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
