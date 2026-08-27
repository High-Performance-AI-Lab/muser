from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "scripts" / "evaluate_nvfp4_fast_drift.py"
SPEC = importlib.util.spec_from_file_location("evaluate_nvfp4_fast_drift", PATH)
assert SPEC is not None and SPEC.loader is not None
drift = importlib.util.module_from_spec(SPEC)
sys.modules["evaluate_nvfp4_fast_drift"] = drift
SPEC.loader.exec_module(drift)


def fixture(mode: str) -> dict:
    shift = 0.1 if mode == "native" else 0.0
    return {
        "id": "code",
        "regime": "code",
        "token_count": 4,
        "token_ids_sha256": "a" * 64,
        "output_tokens": 3,
        "target_logprobs": [-1.0 - shift, -2.0, -3.0] if mode == "native" else [-1.0, -3.0],
        "teacher_forced_top_token_ids": [1, 2, 3] if mode == "native" else [1, 3],
        "generated_tokens": [4, 5, 6],
        "generated_tokens_sha256": ("b" if mode == "exact" else "c") * 64,
        "perplexity": 7.4,
        "boundary_positions": [1, 3],
        **({"scored_positions": [1, 3]} if mode != "native" else {}),
    }


class FastDriftTests(unittest.TestCase):
    def test_comparison_reports_measured_nonzero_drift(self) -> None:
        value = drift.compare_fixture(fixture("reference"), fixture("native"))
        self.assertFalse(value["catastrophic"])
        self.assertGreater(value["perplexity"]["relative_delta"], 0)
        self.assertEqual(value["greedy_stream"]["rate"], 0)
        self.assertGreater(value["target_logprob_delta"]["max_abs"], 0)
        self.assertEqual(value["scored_rows"], 2)

    def test_divergence_counts_length_mismatch(self) -> None:
        value = drift.divergence([1, 2], [1, 3, 4])
        self.assertEqual(value["mismatches"], 2)
        self.assertEqual(value["first"], 1)


if __name__ == "__main__":
    unittest.main()
