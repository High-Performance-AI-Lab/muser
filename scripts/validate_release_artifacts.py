#!/usr/bin/env python3
"""Verify release-pinned model artifacts without downloading or modifying them."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "release-artifacts.json"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="directory containing all three release-pinned filenames",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="require all artifacts (uses --artifact-dir or MUSER_MODEL_DIR)",
    )
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="KIND=PATH",
        help="verify one target, vision, or dflash artifact (repeatable)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "muser.release-artifacts.v2"
        or manifest.get("revision") != "a0532f7263ee67f1e0a5f5c5fdcd50dd62fc9aa4"
        or set(manifest.get("artifacts", {})) != {"target", "vision", "dflash"}
    ):
        raise SystemExit("release artifact manifest is outside the v0.1 identity contract")
    expected = manifest["artifacts"]
    for kind, record in expected.items():
        if (
            record.get("revision") != manifest["revision"]
            or not isinstance(record.get("bytes"), int)
            or record["bytes"] <= 0
            or not isinstance(record.get("url"), str)
            or f"/resolve/{manifest['revision']}/{record.get('filename')}?download=true"
            not in record["url"]
        ):
            raise SystemExit(f"manifest artifact {kind!r} is invalid")
    selected: dict[str, Path] = {}
    if args.require_all and args.artifact_dir is None:
        import os

        configured = os.environ.get("MUSER_MODEL_DIR")
        if not configured:
            raise SystemExit("--require-all needs --artifact-dir or MUSER_MODEL_DIR")
        args.artifact_dir = Path(configured)
    if args.artifact_dir:
        selected.update(
            (kind, args.artifact_dir / record["filename"])
            for kind, record in expected.items()
        )
    for item in args.artifact:
        if "=" not in item:
            raise SystemExit(f"invalid --artifact {item!r}; expected KIND=PATH")
        kind, raw_path = item.split("=", 1)
        if kind not in expected:
            raise SystemExit(f"unknown artifact kind {kind!r}")
        selected[kind] = Path(raw_path)
    if not selected:
        print(
            json.dumps(
                {
                    "schema": manifest["schema"],
                    "status": "not_run",
                    "reason": "no artifact paths supplied; ordinary CI validates manifest policy only",
                },
                sort_keys=True,
            )
        )
        return 0

    failed = False
    for kind, path in selected.items():
        record = expected[kind]
        result = {
            "kind": kind,
            "path": str(path),
            "expected_sha256": record["sha256"],
            "expected_bytes": record["bytes"],
        }
        if not path.is_file():
            result.update(status="missing")
            failed = True
        else:
            size = path.stat().st_size
            if size != record["bytes"]:
                result.update(actual_bytes=size, status="size-mismatch")
                failed = True
                print(json.dumps(result, sort_keys=True))
                continue
            actual = digest(path)
            result.update(
                actual_bytes=size,
                actual_sha256=actual,
                status="verified" if actual == record["sha256"] else "mismatch",
            )
            failed |= actual != record["sha256"]
        print(json.dumps(result, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
