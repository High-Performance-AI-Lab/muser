#!/usr/bin/env python3
"""Single fail-closed authority for candidate/seal-producing tools."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "release" / "release-lock.json"


class ReleaseLocked(RuntimeError):
    pass


def load_release_lock() -> dict:
    if not LOCK_PATH.is_file() or LOCK_PATH.is_symlink():
        raise ReleaseLocked(f"release lock is missing or unsafe: {LOCK_PATH}")
    try:
        value = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseLocked(f"release lock cannot be read: {error}") from error
    if value.get("schema") != "muser.release-lock.v1":
        raise ReleaseLocked("release lock has an unsupported schema")
    return value


def require_sealing_enabled(operation: str) -> None:
    lock = load_release_lock()
    if lock.get("sealing_enabled") is not True:
        reason = lock.get("reason", "no reason recorded")
        raise ReleaseLocked(f"{operation} disabled by {LOCK_PATH}: {reason}")


def require_candidate_enabled(operation: str) -> None:
    lock = load_release_lock()
    if not (
        lock.get("sealing_enabled") is True
        and lock.get("tagging_enabled") in (True, False)
        and lock.get("publishing_enabled") in (True, False)
    ):
        reason = lock.get("reason", "no reason recorded")
        raise ReleaseLocked(f"{operation} disabled by {LOCK_PATH}: {reason}")


def force_unsealed(receipt: dict, *, lane: str | None = None) -> dict:
    """Turn every evaluator verdict into non-seal qualification evidence.

    Individual evaluators never regain seal authority. Once readiness exists,
    only the atomic campaign may expose a seal bundle.
    """
    lock = load_release_lock()
    original_schema = receipt.get("schema")
    original_eligible = receipt.get("seal_eligible") is True
    receipt["schema"] = "muser.unsealed-qualification.v1"
    receipt["would_be_seal_schema"] = original_schema
    receipt["would_be_seal_eligible"] = original_eligible
    receipt["seal_eligible"] = False
    receipt["release_lock_state"] = lock.get("state")
    if lane is not None:
        if not lane or receipt.get("lane") not in (None, lane):
            raise ValueError("unsealed receipt has an invalid or conflicting lane")
        receipt["lane"] = lane
    return receipt
