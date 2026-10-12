from __future__ import annotations

import array
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


comparison = load_script(
    "compare_composite_oracles",
    "scripts/gx10/vllm/compare_composite_oracles.py",
)
economics = load_script(
    "evaluate_composite_verifier",
    "scripts/evaluate_composite_verifier.py",
)


class CompositeVerifierToolTests(unittest.TestCase):
    def test_matrix_comparison_reports_exact_rows_argmax_and_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            left = Path(directory) / "left.f32"
            right = Path(directory) / "right.f32"
            left.write_bytes(array.array("f", [1.0, 2.0, 3.0, 4.0]).tobytes())
            right.write_bytes(array.array("f", [1.0, 2.0, 3.5, 3.0]).tobytes())
            result = comparison.compare_matrix(
                left, right, rows=2, row_floats=2, include_argmax=True
            )
            self.assertFalse(result["bit_exact"])
            self.assertEqual(result["exact_rows"], 1)
            self.assertEqual(result["differing_values"], 2)
            self.assertEqual(result["max_abs_error"], 1.0)
            self.assertEqual(result["argmax_matches"], 1)

    def test_iid_threshold_inverts_carried_frontier_expectation(self) -> None:
        for alpha in (0.0, 0.5, 0.95, 1.0):
            expected = economics.expected_commits_iid(alpha, 16)
            recovered = economics.required_alpha(expected, 16)
            self.assertIsNotNone(recovered)
            self.assertTrue(math.isclose(recovered, alpha, abs_tol=1e-12))
        self.assertIsNone(economics.required_alpha(16.01, 16))

    def test_percentile_is_inclusive_and_deterministic(self) -> None:
        self.assertEqual(economics.percentile([1.0, 2.0, 3.0], 0.5), 2.0)
        self.assertEqual(economics.percentile([1.0, 3.0], 0.25), 1.5)


if __name__ == "__main__":
    unittest.main()
