from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
PATH = ROOT / "scripts" / "evaluate_nvfp4_drift_decomposition.py"
SPEC = importlib.util.spec_from_file_location("evaluate_nvfp4_drift_decomposition", PATH)
assert SPEC is not None and SPEC.loader is not None
evaluation = importlib.util.module_from_spec(SPEC)
sys.modules["evaluate_nvfp4_drift_decomposition"] = evaluation
SPEC.loader.exec_module(evaluation)
PROJECT_PATH = ROOT / "scripts" / "project_nvfp4_drift_score.py"
PROJECT_SPEC = importlib.util.spec_from_file_location("project_nvfp4_drift_score", PROJECT_PATH)
assert PROJECT_SPEC is not None and PROJECT_SPEC.loader is not None
projection = importlib.util.module_from_spec(PROJECT_SPEC)
sys.modules["project_nvfp4_drift_score"] = projection
PROJECT_SPEC.loader.exec_module(projection)
ASSEMBLE_PATH = ROOT / "scripts" / "assemble_kquant_drift_decomposition_reference.py"
ASSEMBLE_SPEC = importlib.util.spec_from_file_location(
    "assemble_kquant_drift_decomposition_reference", ASSEMBLE_PATH
)
assert ASSEMBLE_SPEC is not None and ASSEMBLE_SPEC.loader is not None
assembly = importlib.util.module_from_spec(ASSEMBLE_SPEC)
sys.modules["assemble_kquant_drift_decomposition_reference"] = assembly
ASSEMBLE_SPEC.loader.exec_module(assembly)
ROUTING_PATH = ROOT / "scripts" / "evaluate_nvfp4_routing_ladder.py"
ROUTING_SPEC = importlib.util.spec_from_file_location("evaluate_nvfp4_routing_ladder", ROUTING_PATH)
assert ROUTING_SPEC is not None and ROUTING_SPEC.loader is not None
routing = importlib.util.module_from_spec(ROUTING_SPEC)
sys.modules["evaluate_nvfp4_routing_ladder"] = routing
ROUTING_SPEC.loader.exec_module(routing)


class DriftDecompositionTests(unittest.TestCase):
    def test_routing_gate_aligns_reference_positions(self) -> None:
        reference = {
            "id": "long-8192",
            "regime": "long-context",
            "token_count": 3,
            "token_ids_sha256": "a",
            "scored_positions": [1, 2],
            "target_logprobs": [-1.0, -1.0],
            "teacher_forced_top_token_ids": [7, 8],
        }
        native = {
            "id": "long-8192",
            "regime": "long-context",
            "token_count": 3,
            "token_ids_sha256": "a",
            "target_logprobs": [-1.01, -1.01],
            "teacher_forced_top_token_ids": [7, 9],
        }
        measured = routing.compare(reference, native)
        self.assertFalse(measured["passed"])
        self.assertEqual(measured["top_token_disagreement"]["rate"], 0.5)

    def test_routing_source_requires_contiguous_passing_prefix(self) -> None:
        source = ROUTING_PATH.read_text(encoding="utf-8")
        self.assertIn("contiguous_passing = []", source)
        self.assertIn('if not row["passed"]:', source)
        self.assertIn('"cap_requires_all_lower_tested_rungs_to_pass": True', source)

    def test_kquant_reference_rows_keep_explicit_scored_positions(self) -> None:
        row = assembly.reference_row(
            {"id": "case", "regime": "code"},
            [1, 2, 3],
            {
                "rows": [
                    {"position": 0, "target_nll": 2.0, "candidates": [{"token_id": 8}]},
                    {"position": 1, "target_nll": 1.0, "candidates": [{"token_id": 9}]},
                ]
            },
            Path(__file__),
        )
        self.assertEqual(row["scored_positions"], [1, 2])
        self.assertEqual(row["target_logprobs"], [-2.0, -1.0])

    def test_projection_requires_a_true_prefix(self) -> None:
        source = {
            "id": "source",
            "token_count": 4,
            "token_ids_sha256": projection.token_digest([1, 2, 3, 4]),
            "target_logprobs": [-1.0, -2.0, -3.0],
            "teacher_forced_top_token_ids": [2, 3, 4],
        }
        with self.assertRaisesRegex(ValueError, "not a source prefix"):
            projection.project_row(
                source,
                [1, 2, 3, 4],
                {"id": "target", "regime": "code"},
                Path("target.tokens"),
                [1, 9],
            )

    def test_attribution_classes_are_mutually_exclusive(self) -> None:
        measured = evaluation.attribution_classes(
            [1, 1, 1, 1, 1],
            [1, 2, 1, 2, 2],
            [1, 2, 3, 1, 3],
        )
        self.assertEqual(
            {key: value["count"] for key, value in measured.items()},
            {
                "all_equal": 1,
                "artifact_only": 1,
                "fast_path_only": 1,
                "compensated": 1,
                "compounded": 1,
            },
        )

    def test_log_ppl_components_are_additive(self) -> None:
        reference = {
            "id": "case",
            "regime": "code",
            "token_count": 3,
            "token_ids_sha256": "a",
            "scored_positions": [1, 2],
            "target_logprobs": [-1.0, -2.0],
            "teacher_forced_top_token_ids": [1, 2],
        }
        exact = {
            "id": "case",
            "regime": "code",
            "token_count": 3,
            "token_ids_sha256": "a",
            "target_logprobs": [-1.1, -2.1],
            "teacher_forced_top_token_ids": [1, 3],
        }
        native = exact | {
            "target_logprobs": [-1.2, -2.2],
            "teacher_forced_top_token_ids": [1, 3],
        }
        measured = evaluation.compare_fixture(reference, exact, native, None)
        self.assertAlmostEqual(measured["perplexity"]["log_additivity_error"], 0.0)
        self.assertFalse(measured["passed"])


if __name__ == "__main__":
    unittest.main()
