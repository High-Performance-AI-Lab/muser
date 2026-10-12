#!/usr/bin/env python3
"""Compute the canonical Muser campaign identity v3."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

IDENTITY_FILES = (
    "Cargo.lock",
    "release/feature-contract-v1.json",
    "release/findings-v1.json",
    "release/llama-server-compat-v1.json",
    "release/dflash-route-policy-v1.json",
    "release/dflash-tuning-v1.json",
    "release/gx10-cross-vendor-math-v1.json",
    "release/nvfp4-runtime-identity-v1.json",
    "docs/release-artifacts.json",
    "docs/release-model-metadata.json",
    "third_party/kvpack/provenance.json",
    "third_party/metal/provenance.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_metadata() -> tuple[str, dict]:
    raw = subprocess.run(
        ["cargo", "metadata", "--locked", "--format-version", "1"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    value = json.loads(raw)
    # Absolute workspace paths are environmental rather than release identity.
    for package in value.get("packages", []):
        manifest = Path(package["manifest_path"])
        try:
            package["manifest_path"] = manifest.relative_to(ROOT).as_posix()
        except ValueError:
            package["manifest_path"] = manifest.name
    value.pop("target_directory", None)
    value.pop("workspace_root", None)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), value


def identity(binaries: dict[str, Path] | None = None) -> dict:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    if status:
        raise RuntimeError("campaign identity v3 requires a clean worktree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE
    ).stdout.strip()
    files = {}
    for relative in IDENTITY_FILES:
        path = ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"identity file is missing or unsafe: {relative}")
        files[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    apparatus = hashlib.sha256()
    for path in sorted((ROOT / "scripts").rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            apparatus.update(path.relative_to(ROOT).as_posix().encode())
            apparatus.update(b"\0")
            apparatus.update(bytes.fromhex(sha256(path)))
    metadata_sha256, _ = canonical_metadata()
    binary_values = {}
    for name, path in sorted((binaries or {}).items()):
        resolved = path.resolve()
        if not resolved.is_file() or resolved.is_symlink():
            raise RuntimeError(f"binary is missing or unsafe: {name}={path}")
        binary_values[name] = {"bytes": resolved.stat().st_size, "sha256": sha256(resolved)}
    payload = {
        "schema": "muser.campaign-identity.v3",
        "source": {"commit": commit, "tree": tree},
        "files": files,
        "cargo_metadata_sha256": metadata_sha256,
        "apparatus_sha256": apparatus.hexdigest(),
        "binaries": binary_values,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["digest"] = hashlib.sha256(encoded).hexdigest()
    return payload


def parse_named(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        name, separator, raw = value.partition("=")
        if not separator or not name or name in result:
            raise ValueError(f"invalid or duplicate NAME=PATH: {value!r}")
        result[name] = Path(raw)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", action="append", default=[], metavar="NAME=PATH")
    args = parser.parse_args()
    try:
        print(json.dumps(identity(parse_named(args.binary)), indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"release identity failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
