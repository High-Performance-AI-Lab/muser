#!/usr/bin/env python3
"""Verify every file in the pinned minimal kvpack source snapshot.

The working tree must match the receipted per-file hashes exactly. Local
deviations from upstream are allowed only as documented patch entries with
an id, a reason, and the receipted files they touch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "third_party" / "kvpack"
PROVENANCE = VENDOR / "provenance.json"
EXPECTED_COMMIT = "70c34c7d790dbfc9c1271727dd34ea0e863404d2"
EXPECTED_TAG = "kvpack-v0.1.0-alpha.2-rc1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_files() -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(VENDOR.rglob("*")):
        if path == PROVENANCE:
            continue
        # Cargo may be invoked directly against the excluded upstream
        # workspace during its own test lane. Its build cache is not vendored
        # source and must neither enter provenance nor make a later audit
        # depend on which lane ran first.
        if path.relative_to(VENDOR).parts[0] == "target":
            continue
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise RuntimeError(f"unsafe vendored entry: {path}")
        if path.is_file():
            result[path.relative_to(VENDOR).as_posix()] = sha256(path)
    return result


def main() -> int:
    value = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    failures: list[str] = []
    if value.get("schema") != "muser.vendored-source.v1":
        failures.append("wrong provenance schema")
    if value.get("upstream_commit") != EXPECTED_COMMIT:
        failures.append("wrong upstream commit")
    if value.get("upstream_tag") != EXPECTED_TAG:
        failures.append("wrong upstream tag")
    patches = value.get("patches")
    if not isinstance(patches, list):
        failures.append("patches must be a list (empty for a pristine snapshot)")
    else:
        for patch in patches:
            if not isinstance(patch, dict) or not all(
                isinstance(patch.get(key), str) and patch.get(key)
                for key in ("id", "reason")
            ) or not isinstance(patch.get("files"), list) or not patch["files"]:
                failures.append(f"malformed patch entry: {patch!r}")
                continue
            unknown = sorted(
                path for path in patch["files"]
                if not isinstance(path, str) or path not in (value.get("files") or {})
            )
            if unknown:
                failures.append(f"patch {patch['id']} names unreceipted files: {unknown}")
    actual = source_files()
    expected = value.get("files")
    if not isinstance(expected, dict):
        failures.append("per-file hash map is absent")
    else:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(
            path for path in set(actual) & set(expected) if actual[path] != expected[path]
        )
        if missing:
            failures.append(f"missing vendored files: {missing}")
        if extra:
            failures.append(f"unreceipted vendored files: {extra}")
        if changed:
            failures.append(f"changed vendored files: {changed}")
    result = {"status": "failed" if failures else "passed", "failures": failures}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
