#!/usr/bin/env python3
"""Re-stamp the Muser-side SHA-256 receipts in docs/extraction-manifest.md.

The manifest carries three kinds of hash records:

1. The "Stage 2 shader extraction receipt" table (single `SHA-256` column).
   This is a frozen historical/source witness recorded at initial
   extraction time (the doc explicitly preserves it even after a file is
   later adapted, e.g. `rmsnorm_batch_tail.metal`). It is NEVER rewritten
   by this script.

2. The "parity-restoration pass" table with separate `source SHA-256` and
   `Muser SHA-256` columns. Only the `Muser SHA-256` column (the last
   64-hex-char hash on each data row) is a live, re-stampable record of
   the current muser file. The `source SHA-256` column is a historical
   fact about the Ferrite source and is never rewritten.

3. Two standalone prose sentences that record a "current SHA-256" for a
   muser file outside of any table (`rmsnorm_batch_tail.metal` and
   `muse_reference.metal`). These are also live, re-stampable records.

Usage:
    python3 scripts/restamp_extraction_manifest.py --check
    python3 scripts/restamp_extraction_manifest.py --write
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "docs" / "extraction-manifest.md"
SHADER_DIR = REPO_ROOT / "crates" / "muser-engine" / "src" / "shaders" / "ferrite"
SHADER_ROOT = REPO_ROOT / "crates" / "muser-engine" / "src" / "shaders"

# The commit at which the current re-stamping waves (1-2) began. A mismatch
# on a file git shows as untouched since this point is a *pre-existing*
# manifest/file drift, not something waves 1-2 caused — restamping it here
# would silently launder an unrelated integrity problem, so --write refuses
# and --check reports it separately without failing the run.
DEFAULT_WAVE_SINCE = "e309c83"

HASH_RE = re.compile(r"[0-9a-f]{64}")


@dataclass
class Record:
    """One live, re-stampable Muser-hash record."""

    label: str  # human-readable name for reporting, e.g. the file name
    muser_path: Path  # absolute path to the file on disk
    manifest_hash: str  # the hash currently recorded in the manifest text
    # span (start, end) of the manifest_hash substring within the raw text,
    # used to splice in a replacement without touching anything else.
    span: tuple[int, int]


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_parity_table_records(text: str) -> list[Record]:
    """Parse the 'parity-restoration pass' table (Muser SHA-256 column)."""
    records: list[Record] = []
    header_idx = text.find("| Muser file | Ferrite source | source SHA-256 | Muser SHA-256 |")
    if header_idx == -1:
        return records
    # Data rows follow immediately after the header + separator line.
    rest = text[header_idx:]
    lines = rest.splitlines(keepends=True)
    offset = header_idx + len(lines[0]) + len(lines[1])  # skip header + '|---|---|...' separator
    row_lines = lines[2:]
    for line in row_lines:
        if not line.startswith("|"):
            break
        m = re.match(r"\|\s*`([^`]+)`\s*\|", line)
        if not m:
            offset += len(line)
            continue
        filename = m.group(1)
        hashes = list(HASH_RE.finditer(line))
        if not hashes:
            offset += len(line)
            continue
        last = hashes[-1]
        muser_path = SHADER_DIR / filename
        records.append(
            Record(
                label=filename,
                muser_path=muser_path,
                manifest_hash=last.group(0),
                span=(offset + last.start(), offset + last.end()),
            )
        )
        offset += len(line)
    return records


def find_prose_records(text: str) -> list[Record]:
    """Parse the two standalone 'current SHA-256' prose sentences."""
    records: list[Record] = []

    m = re.search(
        r"`rmsnorm_batch_tail\.metal` is the one subsequently adapted in Muser: its\n"
        r"current SHA-256 is\n"
        r"`([0-9a-f]{64})`\.",
        text,
    )
    if m:
        records.append(
            Record(
                label="rmsnorm_batch_tail.metal (prose)",
                muser_path=SHADER_DIR / "rmsnorm_batch_tail.metal",
                manifest_hash=m.group(1),
                span=m.span(1),
            )
        )

    m = re.search(
        r"`muse_reference\.metal` SHA-256\n`([0-9a-f]{64})`\)",
        text,
    )
    if m:
        records.append(
            Record(
                label="muse_reference.metal (prose)",
                muser_path=SHADER_ROOT / "muse_reference.metal",
                manifest_hash=m.group(1),
                span=m.span(1),
            )
        )

    return records


def collect_records(text: str) -> list[Record]:
    return find_parity_table_records(text) + find_prose_records(text)


def wave_touched_files(since: str) -> set[Path] | None:
    """Absolute paths of files changed from `since`, including the worktree.

    Returns None if the range can't be resolved (e.g. `since` isn't a known
    revision in this checkout) — callers should then fall back to treating
    scope as unknown rather than silently gating on an empty set.
    """
    try:
        out = subprocess.run(
            ["git", "diff", "--name-only", since, "--"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return {REPO_ROOT / line for line in out.stdout.splitlines() if line.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="report mismatches and exit nonzero if any exist")
    group.add_argument("--write", action="store_true", help="update stale Muser-hash records in place")
    parser.add_argument(
        "--wave-since",
        default=DEFAULT_WAVE_SINCE,
        help=f"git rev; only files changed since this rev (exclusive) are eligible for --write (default {DEFAULT_WAVE_SINCE})",
    )
    args = parser.parse_args()

    text = MANIFEST_PATH.read_text()
    records = collect_records(text)
    if not records:
        print("no re-stampable Muser-hash records found in manifest", file=sys.stderr)
        return 2

    touched = wave_touched_files(args.wave_since)

    mismatches: list[tuple[Record, str]] = []
    for rec in records:
        if not rec.muser_path.exists():
            print(f"MISSING FILE: {rec.label} -> {rec.muser_path}", file=sys.stderr)
            continue
        actual = sha256_of(rec.muser_path)
        if actual != rec.manifest_hash:
            mismatches.append((rec, actual))

    def in_scope(rec: Record) -> bool:
        return touched is None or rec.muser_path.resolve() in touched

    in_scope_mismatches = [(r, a) for r, a in mismatches if in_scope(r)]
    out_of_scope_mismatches = [(r, a) for r, a in mismatches if not in_scope(r)]

    if args.check:
        if not mismatches:
            print(f"OK: {len(records)} Muser-hash record(s) checked, 0 mismatch(es).")
            return 0
        for rec, actual in mismatches:
            print(f"MISMATCH: {rec.label}")
            print(f"  manifest: {rec.manifest_hash}")
            print(f"  actual:   {actual}")
        return 1

    # --write: only ever touches in-scope (wave-modified) records; splice
    # replacements in from the end of the file backwards so earlier spans
    # stay valid, preserving all other text untouched.
    if out_of_scope_mismatches:
        print(
            f"REFUSING to write {len(out_of_scope_mismatches)} out-of-scope mismatch(es) "
            f"(not touched since {args.wave_since}) — integrity finding, not a restamp target:"
        )
        for rec, actual in out_of_scope_mismatches:
            print(f"  SKIPPED: {rec.label}")
            print(f"    manifest: {rec.manifest_hash}")
            print(f"    actual:   {actual}")
        return 1

    if not in_scope_mismatches:
        print("nothing in-scope to re-stamp.")
        return 0

    in_scope_mismatches.sort(key=lambda pair: pair[0].span[0], reverse=True)
    new_text = text
    restamped: list[str] = []
    for rec, actual in in_scope_mismatches:
        start, end = rec.span
        found = new_text[start:end]
        if found != rec.manifest_hash:
            print(
                f"ABORT: span integrity check failed for {rec.label}: "
                f"expected {rec.manifest_hash!r} at [{start}:{end}], found {found!r}",
                file=sys.stderr,
            )
            return 3
        new_text = new_text[:start] + actual + new_text[end:]
        restamped.append(rec.label)

    MANIFEST_PATH.write_text(new_text)
    for label in reversed(restamped):
        print(f"re-stamped: {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
