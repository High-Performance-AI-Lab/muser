from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "scripts" / "assemble_kquant_content_control_reference.py"
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "assemble_kquant_content_control_reference", PATH
)
assert SPEC is not None and SPEC.loader is not None
assemble = importlib.util.module_from_spec(SPEC)
sys.modules["assemble_kquant_content_control_reference"] = assemble
SPEC.loader.exec_module(assemble)


class CompactContentControlReferenceTests(unittest.TestCase):
    def test_compact_rows_are_bound_to_fixture_and_capture(self) -> None:
        tokens = [10, 11, 12, 13, 14, 15]
        source_artifacts = {
            "quantized_logits": {"sha256": "d" * 64, "size_bytes": 123},
            "exact_top10": {"sha256": "e" * 64, "size_bytes": 456},
            "runtime": {"sha256": "f" * 64, "size_bytes": 789},
        }
        metrics = {
            "exact_target_nll_sum": 2.75,
            "exact_perplexity": math.exp(2.75 / 2),
        }
        compact = {
            "schema": assemble.COMPACT_TEACHER_SCHEMA,
            "status": "validated",
            "validation": {
                "validator": (
                    "scripts/llama_perplexity_evidence.py::validate_teacher_evidence"
                ),
                "quantized_cross_binding": "validated-before-compaction",
                "upstream_commit": "a" * 40,
                "patch_sha256": "b" * 64,
                "evidence_id": "c" * 64,
                "source_artifacts": source_artifacts,
            },
            "geometry": {
                "context_length": 6,
                "vocab_size": 100,
                "chunks": 1,
                "scored_rows": 2,
            },
            "metrics": metrics,
            "rows": [
                {
                    "chunk": 0,
                    "position": 3,
                    "input_token_id": 13,
                    "target_token_id": 14,
                    "target_nll": 1.25,
                    "teacher_forced_top_token_id": 21,
                },
                {
                    "chunk": 0,
                    "position": 4,
                    "input_token_id": 14,
                    "target_token_id": 15,
                    "target_nll": 1.5,
                    "teacher_forced_top_token_id": 22,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            compact_path = parent / "teacher-evidence.json"
            compact_path.write_text(json.dumps(compact), encoding="utf-8")
            capture_path = parent / "perplexity-capture.json"
            capture = {
                "artifacts": {
                    "teacher_evidence": {
                        **source_artifacts,
                        "compact": {
                            "path": str(compact_path),
                            "bytes": compact_path.stat().st_size,
                            "sha256": assemble.sha256(compact_path),
                        },
                        "source_artifacts_retained": False,
                    }
                },
                "metrics": metrics,
            }
            receipt = {"source_commit": "a" * 40, "patch_sha256": "b" * 64}
            fixture = {"id": "fixture", "regime": "long-context"}

            row = assemble.compact_reference_row(
                fixture, tokens, capture, receipt, capture_path
            )

        self.assertEqual(row["scored_positions"], [4, 5])
        self.assertEqual(row["target_logprobs"], [-1.25, -1.5])
        self.assertEqual(row["teacher_forced_top_token_ids"], [21, 22])
        self.assertEqual(row["token_ids_sha256"], assemble.token_digest(tokens))


if __name__ == "__main__":
    unittest.main()
