from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "scripts" / "evaluate_nvfp4_quant_yardstick.py"
SPEC = importlib.util.spec_from_file_location("evaluate_nvfp4_quant_yardstick", PATH)
assert SPEC is not None and SPEC.loader is not None
yardstick = importlib.util.module_from_spec(SPEC)
sys.modules["evaluate_nvfp4_quant_yardstick"] = yardstick
SPEC.loader.exec_module(yardstick)


def row(top: list[int], logs: list[float]) -> dict:
    return {
        "id": "case",
        "regime": "code",
        "token_count": 5,
        "token_ids_sha256": "a",
        "scored_positions": [1, 2, 3, 4],
        "target_logprobs": logs,
        "teacher_forced_top_token_ids": top,
    }


class QuantYardstickTests(unittest.TestCase):
    def test_wilson_interval_contains_observed_rate(self) -> None:
        low, high = yardstick.wilson_interval(10, 100)
        self.assertLess(low, 0.1)
        self.assertGreater(high, 0.1)

    def test_calibrated_gate_uses_band_plus_frozen_margin(self) -> None:
        baseline = row([1, 2, 3, 4], [-1.0, -1.0, -1.0, -1.0])
        alternate = row([1, 2, 9, 4], [-1.01, -1.01, -1.01, -1.01])
        native = row([1, 8, 9, 4], [-1.02, -1.02, -1.02, -1.02])
        measured = yardstick.compare(baseline, alternate, native)
        upper = measured["yardstick"]["top_token_disagreement"]["wilson_95"][1]
        self.assertAlmostEqual(
            measured["calibrated_gates"]["top_token_disagreement"],
            min(1.0, upper + yardstick.DISAGREEMENT_MARGIN),
        )
        self.assertTrue(measured["native_vs_kquant"]["perplexity_passed"])

    def test_bootstrap_is_deterministic(self) -> None:
        baseline = [-1.0, -1.1, -1.2, -1.3]
        alternate = [-1.1, -1.0, -1.3, -1.2]
        self.assertEqual(
            yardstick.paired_ppl_bootstrap(baseline, alternate, "fixture"),
            yardstick.paired_ppl_bootstrap(baseline, alternate, "fixture"),
        )


if __name__ == "__main__":
    unittest.main()
