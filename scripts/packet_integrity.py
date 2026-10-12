"""Shared fail-closed collection for append-only qualification ledgers."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable
import os
from pathlib import Path
from typing import Any, TypeVar


Key = TypeVar("Key", bound=Hashable)


def publish_new(path: Path, payload: str) -> None:
    """Publish a compact receipt once without replacing prior evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        encoded = payload.encode()
        written = 0
        while written < len(encoded):
            written += os.write(fd, encoded[written:])
        os.fsync(fd)
    finally:
        os.close(fd)


def collect_unique_packet(
    records: Iterable[dict[str, Any]],
    expected: set[Key],
    *,
    identity: str,
    key: Callable[[dict[str, Any]], Key | None],
    label: str,
) -> tuple[dict[Key, dict[str, Any]], list[str]]:
    """Return exactly one record per expected key or a fail-closed diagnosis.

    Qualification ledgers are append-only.  Keeping the last record in a
    dictionary would let a selected-cell rerun conceal an earlier failed or
    otherwise different attempt.  A valid packet therefore contains exactly
    one record for every expected key under the requested campaign identity.
    Other identities and other lane keys may coexist in the same ledger.
    """

    grouped: dict[Key, list[dict[str, Any]]] = {}
    for record in records:
        if record.get("identity") != identity:
            continue
        record_key = key(record)
        if record_key in expected:
            grouped.setdefault(record_key, []).append(record)

    failures: list[str] = []
    missing = sorted(expected - set(grouped), key=str)
    duplicates = sorted(
        (record_key for record_key, values in grouped.items() if len(values) != 1),
        key=str,
    )
    if missing:
        failures.append(f"incomplete {label} packet: missing {missing}")
    if duplicates:
        failures.append(
            f"{label} packet contains duplicate records (selected-cell reruns are not sealable): "
            f"{duplicates}"
        )

    return (
        {
            record_key: values[0]
            for record_key, values in grouped.items()
            if len(values) == 1
        },
        failures,
    )
