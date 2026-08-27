#!/usr/bin/env python3
"""Verify the three external model artifacts used by the private demo."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--vision", type=Path, required=True)
    parser.add_argument("--dflash", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if not args.manifest.is_file() or args.manifest.is_symlink():
        raise SystemExit("release artifact manifest is missing or unsafe")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected = manifest.get("artifacts", {})
    selected = {
        "target": args.target,
        "vision": args.vision,
        "dflash": args.dflash,
    }
    results = []
    failed = False
    for kind, path in selected.items():
        record = expected.get(kind, {})
        result = {
            "kind": kind,
            "filename": path.name,
            "expected_filename": record.get("filename"),
            "expected_bytes": record.get("bytes"),
            "expected_sha256": record.get("sha256"),
        }
        if not path.is_file() or path.is_symlink():
            result["status"] = "missing-or-unsafe"
            failed = True
        else:
            size = path.stat().st_size
            actual = sha256(path)
            result["actual_bytes"] = size
            result["actual_sha256"] = actual
            result["status"] = (
                "verified"
                if size == record.get("bytes") and actual == record.get("sha256")
                else "mismatch"
            )
            failed |= result["status"] != "verified"
        results.append(result)
    print(json.dumps({"status": "failed" if failed else "verified", "artifacts": results},
                     indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
