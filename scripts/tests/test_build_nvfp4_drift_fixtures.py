from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "scripts" / "build_nvfp4_drift_fixtures.py"
SPEC = importlib.util.spec_from_file_location("build_nvfp4_drift_fixtures", PATH)
assert SPEC is not None and SPEC.loader is not None
fixtures = importlib.util.module_from_spec(SPEC)
sys.modules["build_nvfp4_drift_fixtures"] = fixtures
SPEC.loader.exec_module(fixtures)


class DriftFixtureBuildTests(unittest.TestCase):
    def test_agentic_packet_excludes_answers_stubs_and_checkers(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "tasks.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": "task-1",
                        "category": "code",
                        "difficulty": "easy",
                        "prompt": "do the task",
                        "tools": [
                            {
                                "name": "edit",
                                "description": "edit a file",
                                "parameters": {"type": "object"},
                                "stub": {"secret": "observation"},
                            }
                        ],
                        "expected": "answer",
                        "checker": {"kind": "exact"},
                        "reference_solution": "answer",
                    }
                )
                + "\n"
            )
            text, ids = fixtures.agentic_packet(path)
        self.assertEqual(ids, ["task-1"])
        self.assertIn("do the task", text)
        self.assertNotIn("answer", text)
        self.assertNotIn("observation", text)
        self.assertNotIn("checker", text)

    def test_parse_original_is_closed(self) -> None:
        fixture_id, path = fixtures.parse_original("p1=/tmp/p1.tokens")
        self.assertEqual(fixture_id, "p1")
        self.assertEqual(path, Path("/tmp/p1.tokens"))
        with self.assertRaises(Exception):
            fixtures.parse_original("bad")


if __name__ == "__main__":
    unittest.main()
