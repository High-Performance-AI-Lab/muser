from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "scripts" / "evaluate_nvfp4_content_control.py"
SPEC = importlib.util.spec_from_file_location("evaluate_nvfp4_content_control", PATH)
assert SPEC is not None and SPEC.loader is not None
control = importlib.util.module_from_spec(SPEC)
sys.modules["evaluate_nvfp4_content_control"] = control
SPEC.loader.exec_module(control)


def comparison(document: str, length: int, passed: bool) -> dict:
    return {
        "document": document,
        "token_count": length,
        "passed": passed,
    }


def packet(failures: dict[int, set[str]]) -> list[dict]:
    return [
        comparison(document, length, document not in failures.get(length, set()))
        for length in (2048, 4096, 8192, 16384, 32768)
        for document in ("a", "b", "c")
    ]


class ContentControlTests(unittest.TestCase):
    def test_one_document_spike_is_content_sensitive_without_cap(self) -> None:
        decision = control.route(packet({8192: {"a"}}))
        self.assertEqual(decision["status"], "no-cap")
        self.assertEqual(decision["branch"], "content-sensitive-envelope")

    def test_replicated_persistent_effect_selects_previous_rung(self) -> None:
        decision = control.route(packet({8192: {"a", "b"}, 16384: {"b", "c"}}))
        self.assertEqual(decision["status"], "routing-required")
        self.assertEqual(decision["native_context_cap_tokens"], 4096)

    def test_first_rung_persistent_effect_is_blocker(self) -> None:
        decision = control.route(packet({2048: {"a", "b"}, 4096: {"b", "c"}}))
        self.assertEqual(decision["status"], "quality-blocker")

    def test_short_lengths_use_eight_k_yardstick(self) -> None:
        bands = {8192: {"top": 0.1}, 16384: {"top": 0.2}, 32768: {"top": 0.3}}
        selected, band = control.select_band(bands, 4096)
        self.assertEqual(selected, 8192)
        self.assertEqual(band["top"], 0.1)


if __name__ == "__main__":
    unittest.main()
