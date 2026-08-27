from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "scripts" / "capture_llama_perplexity.py"
SPEC = importlib.util.spec_from_file_location("capture_llama_perplexity", PATH)
assert SPEC is not None and SPEC.loader is not None
capture = importlib.util.module_from_spec(SPEC)
sys.modules["capture_llama_perplexity"] = capture
SPEC.loader.exec_module(capture)


class CaptureLlamaPerplexityTests(unittest.TestCase):
    def test_token_fixture_is_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens"
            path.write_text("0\n1\n2147483647\n")
            self.assertEqual(capture.token_file(path), [0, 1, 2147483647])
            path.write_text("-1\n")
            with self.assertRaisesRegex(RuntimeError, "out-of-range"):
                capture.token_file(path)

    def test_compact_report_retains_only_comparator_rows_and_source_hashes(self) -> None:
        evidence = {
            "header": {
                "upstream_commit": "a" * 40,
                "patch_sha256": "b" * 64,
                "evidence_id": "c" * 64,
                "context_length": 6,
                "vocab_size": 100,
                "chunks": 1,
                "scored_rows": 2,
            },
            "rows": [
                {
                    "chunk": 0,
                    "position": 3,
                    "input_token_id": 13,
                    "target_token_id": 14,
                    "target_nll": 1.25,
                    "candidates": [{"token_id": 21}],
                },
                {
                    "chunk": 0,
                    "position": 4,
                    "input_token_id": 14,
                    "target_token_id": 15,
                    "target_nll": 1.5,
                    "candidates": [{"token_id": 22}],
                },
            ],
            "metrics": {
                "exact_target_nll_sum": 2.75,
                "exact_perplexity": math.exp(2.75 / 2),
            },
            "artifacts": {
                "quantized_logits": {"sha256": "d" * 64, "size_bytes": 123},
                "exact_top10": {"sha256": "e" * 64, "size_bytes": 456},
                "runtime": {"sha256": "f" * 64, "size_bytes": 789},
            },
        }

        report = capture.compact_teacher_report(evidence)

        self.assertEqual(report["schema"], capture.COMPACT_TEACHER_SCHEMA)
        self.assertEqual(
            report["validation"]["source_artifacts"], evidence["artifacts"]
        )
        self.assertEqual(report["rows"][0]["teacher_forced_top_token_id"], 21)
        self.assertNotIn("candidates", report["rows"][0])

    def test_raw_cleanup_is_limited_to_three_files_below_scratch_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "scratch"
            raw = root / "cell" / "logits.bin"
            raw.parent.mkdir(parents=True)
            siblings = (
                raw,
                Path(str(raw) + ".muser-top10.jsonl"),
                Path(str(raw) + ".muser-runtime.json"),
            )
            for path in siblings:
                path.write_bytes(b"evidence")
            unrelated = raw.parent / "keep.txt"
            unrelated.write_text("keep")

            capture.remove_raw_scratch(raw, root)

            self.assertTrue(unrelated.exists())
            self.assertTrue(all(not path.exists() for path in siblings))
            with self.assertRaisesRegex(RuntimeError, "escapes"):
                capture.require_scratch_path(Path(directory) / "outside", root)


if __name__ == "__main__":
    unittest.main()
