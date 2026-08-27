"""One-shot in-process receipt channel for the resident producer."""

from __future__ import annotations

import threading
from typing import Any

_lock = threading.Lock()
_receipt: dict[str, Any] | None = None


def ensure_slot_available() -> None:
    with _lock:
        if _receipt is not None:
            raise RuntimeError("previous Muser prefill receipt was not consumed")


def publish_receipt(receipt: dict[str, Any]) -> None:
    global _receipt
    with _lock:
        if _receipt is not None:
            raise RuntimeError("Muser prefill receipt slot is already occupied")
        _receipt = receipt


def consume_receipt() -> dict[str, Any]:
    global _receipt
    with _lock:
        if _receipt is None:
            raise RuntimeError("vLLM completed without a Muser handoff receipt")
        receipt = _receipt
        _receipt = None
        return receipt
