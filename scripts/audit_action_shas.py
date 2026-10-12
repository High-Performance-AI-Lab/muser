#!/usr/bin/env python3
"""Reject mutable GitHub Action refs in tracked workflows."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
USE = re.compile(r"^\s*uses:\s*([^\s#]+)\s*$")
SHA = re.compile(r"^[^@]+@[0-9a-f]{40}$")


def main() -> int:
    failures: list[str] = []
    for path in sorted((ROOT / ".github" / "workflows").glob("*.y*ml")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = USE.match(line)
            if match and not SHA.fullmatch(match.group(1)):
                failures.append(
                    f"{path.relative_to(ROOT)}:{number}: mutable action ref {match.group(1)!r}"
                )
    result = {"status": "failed" if failures else "passed", "failures": failures}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
