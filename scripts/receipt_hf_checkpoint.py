#!/usr/bin/env python3
"""Verify a pinned Hugging Face snapshot and publish an immutable receipt."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import tempfile
import uuid


REVISION = re.compile(r"[0-9a-f]{40}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if relative.parts[:1] == (".cache",):
            continue
        if path.is_symlink():
            raise RuntimeError(f"checkpoint contains a symlink: {relative}")
        if path.is_file():
            files[relative.as_posix()] = path
    return files


def load_expected(
    path: Path, repo_id: str, revision: str
) -> tuple[dict[str, dict[str, object]], str]:
    encoded = path.read_bytes()
    manifest = json.loads(encoded)
    if manifest.get("id") != repo_id:
        raise RuntimeError(
            f"API manifest repo mismatch: {manifest.get('id')!r} != {repo_id!r}"
        )
    if manifest.get("sha") != revision:
        raise RuntimeError(
            f"API manifest revision mismatch: {manifest.get('sha')!r} != {revision!r}"
        )
    siblings = manifest.get("siblings")
    if not isinstance(siblings, list) or not siblings:
        raise RuntimeError("API manifest has no sibling inventory")
    expected: dict[str, dict[str, object]] = {}
    for sibling in siblings:
        name = sibling.get("rfilename")
        size = sibling.get("size")
        if not isinstance(name, str) or not isinstance(size, int):
            raise RuntimeError("API manifest sibling lacks rfilename or size")
        if name in expected:
            raise RuntimeError(f"duplicate API manifest path: {name}")
        expected[name] = sibling
    return expected, hashlib.sha256(encoded).hexdigest()


def artifact_digest(rows: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            f"{row['path']}\0{row['size']}\0{row['sha256']}\n".encode()
        )
    return digest.hexdigest()


def publish(path: Path, receipt: dict[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to replace receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    encoded = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        written = 0
        while written < len(encoded):
            written += os.write(fd, encoded[written:])
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.link(temporary, path)
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--api-manifest", type=Path, required=True)
    parser.add_argument("--host-role", choices=("mac", "spark"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if REVISION.fullmatch(args.revision) is None:
        parser.error("--revision must be an exact lowercase 40-hex commit")
    return args


def main() -> int:
    args = parse_args()
    root = args.checkpoint.resolve()
    if not root.is_dir():
        raise SystemExit(f"checkpoint directory is missing: {root}")
    expected, api_manifest_sha256 = load_expected(
        args.api_manifest, args.repo_id, args.revision
    )
    actual = checkpoint_files(root)
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing or unexpected:
        raise SystemExit(
            f"checkpoint file set mismatch: missing={missing}, unexpected={unexpected}"
        )

    rows: list[dict[str, object]] = []
    for name in sorted(expected):
        path = actual[name]
        size = path.stat().st_size
        wanted_size = expected[name]["size"]
        if size != wanted_size:
            raise SystemExit(f"size mismatch for {name}: {size} != {wanted_size}")
        digest = sha256(path)
        lfs = expected[name].get("lfs")
        lfs_sha256 = lfs.get("sha256") if isinstance(lfs, dict) else None
        if lfs_sha256 is not None and digest != lfs_sha256:
            raise SystemExit(
                f"LFS SHA-256 mismatch for {name}: {digest} != {lfs_sha256}"
            )
        rows.append(
            {
                "path": name,
                "size": size,
                "sha256": digest,
                "hub_lfs_sha256": lfs_sha256,
            }
        )

    receipt: dict[str, object] = {
        "schema": "muser.hf-checkpoint.receipt.v1",
        "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repo_id": args.repo_id,
        "revision": args.revision,
        "checkpoint": str(root),
        "host_role": args.host_role,
        "hostname": socket.gethostname(),
        "api_manifest_sha256": api_manifest_sha256,
        "artifact_sha256": artifact_digest(rows),
        "file_count": len(rows),
        "total_size": sum(int(row["size"]) for row in rows),
        "files": rows,
    }
    publish(args.out, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
