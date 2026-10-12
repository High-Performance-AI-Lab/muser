#!/usr/bin/env python3
"""Fail closed unless the exact Apple release host and model artifacts are present."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MEMORY = 96 * 1024**3
SERIAL_METAL_TEST_POLICY = {
    "accelerator_wrapper": "scripts/accelerator_safe.py --execute",
    "argv": [
        "cargo",
        "test",
        "--workspace",
        "--all-features",
        "--all-targets",
        "--locked",
        "--",
        "--test-threads=1",
    ],
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def required_path(variable: str) -> Path:
    raw = os.environ.get(variable)
    if not raw:
        raise RuntimeError(f"{variable} is required")
    path = Path(raw).expanduser().resolve()
    if not path.exists() or path.is_symlink():
        raise RuntimeError(f"{variable} is missing or unsafe: {path}")
    return path


def validate_serial_metal_test_policy(matrix: dict) -> dict:
    policy = matrix.get("preflight", {}).get("serial_metal_tests")
    if policy != SERIAL_METAL_TEST_POLICY:
        raise RuntimeError(
            "release matrix must pin the all-features Metal suite to "
            "accelerator_safe --execute with --test-threads=1"
        )
    return policy


def main() -> int:
    try:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise RuntimeError("release qualification requires macOS arm64")
        memory = int(subprocess.run(
            ["sysctl", "-n", "hw.memsize"], check=True, text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip())
        if memory != EXPECTED_MEMORY:
            raise RuntimeError(f"release host has {memory} bytes RAM, expected {EXPECTED_MEMORY}")
        manifest = json.loads((ROOT / "docs/release-artifacts.json").read_text())
        paths = {
            "target": required_path("MUSER_MODEL"),
            "dflash": required_path("MUSER_DFLASH"),
            "vision": required_path("MUSER_MMPROJ"),
        }
        verified = {}
        for name, path in paths.items():
            expected = manifest["artifacts"][name]
            size = path.stat().st_size
            sha256 = digest(path)
            if size != expected["bytes"] or sha256 != expected["sha256"]:
                raise RuntimeError(f"{name} artifact differs from the immutable manifest")
            verified[name] = {"path": str(path), "bytes": size, "sha256": sha256}
        results = required_path("MUSER_RELEASE_RESULTS")
        if not results.is_dir() or results == ROOT or ROOT in results.parents:
            raise RuntimeError("MUSER_RELEASE_RESULTS must be an existing external directory")
        config = required_path("MUSER_RELEASE_MATRIX_CONFIG")
        if not config.is_file():
            raise RuntimeError("MUSER_RELEASE_MATRIX_CONFIG must be a regular file")
        matrix = json.loads(config.read_text(encoding="utf-8"))
        serial_metal_tests = validate_serial_metal_test_policy(matrix)
        print(json.dumps({
            "schema": "muser.release-host-preflight.v1",
            "status": "passed",
            "memory_bytes": memory,
            "artifacts": verified,
            "results": str(results),
            "matrix_config": str(config),
            "serial_metal_tests": serial_metal_tests,
        }, indent=2, sort_keys=True))
        return 0
    except (OSError, KeyError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"release host preflight failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
