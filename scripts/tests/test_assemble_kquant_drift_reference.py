from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "scripts" / "assemble_kquant_drift_reference.py"
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("assemble_kquant_drift_reference", PATH)
assert SPEC is not None and SPEC.loader is not None
assemble = importlib.util.module_from_spec(SPEC)
sys.modules["assemble_kquant_drift_reference"] = assemble
SPEC.loader.exec_module(assemble)


class KquantReferenceAssemblyTests(unittest.TestCase):
    def test_token_digest_is_bare_hex_for_native_report_compatibility(self) -> None:
        digest = assemble.token_digest([1, 2])
        self.assertEqual(len(digest), 64)
        self.assertNotIn(":", digest)

    def test_reference_positions_predict_the_following_target(self) -> None:
        evidence = {
            "rows": [
                {"position": 2, "target_nll": 0.25, "candidates": [{"token_id": 9}]},
                {"position": 3, "target_nll": 0.5, "candidates": [{"token_id": 8}]},
            ]
        }
        row = assemble.reference_row(
            {"id": "code", "regime": "code"},
            [200_000, 1, 2, 3, 4],
            evidence,
            [5, 6],
            {},
        )
        self.assertEqual(row["scored_positions"], [3, 4])
        self.assertEqual(row["target_logprobs"], [-0.25, -0.5])
        self.assertEqual(row["teacher_forced_top_token_ids"], [9, 8])


if __name__ == "__main__":
    unittest.main()
