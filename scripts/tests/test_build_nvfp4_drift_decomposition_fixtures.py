from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "scripts" / "build_nvfp4_drift_decomposition_fixtures.py"
SPEC = importlib.util.spec_from_file_location("build_nvfp4_drift_decomposition_fixtures", PATH)
assert SPEC is not None and SPEC.loader is not None
fixtures = importlib.util.module_from_spec(SPEC)
sys.modules["build_nvfp4_drift_decomposition_fixtures"] = fixtures
SPEC.loader.exec_module(fixtures)


class DriftDecompositionFixtureTests(unittest.TestCase):
    def test_emitted_fixture_is_line_oriented_and_sequence_bound(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source.tokens"
            source.write_text("200000 1 2\n", encoding="utf-8")
            entry, receipt = fixtures.emit_fixture(
                root,
                "case",
                "code",
                [200_000, 1, 2],
                source,
                {"start": 0, "end": 3},
            )
            emitted = (root / entry["token_file"]).read_text(encoding="utf-8")
        self.assertEqual(emitted, "200000\n1\n2\n")
        self.assertEqual(receipt["token_count"], 3)
        self.assertEqual(len(receipt["token_ids_sha256"]), 64)

    def test_exact_attribution_manifest_is_declared(self) -> None:
        source = PATH.read_text(encoding="utf-8")
        self.assertIn('"exact_manifest": exact_path.name', source)
        self.assertIn('"code-r2048"', source)
        self.assertIn('"long-tail-r2048"', source)
        self.assertIn('"diverse-p1-r192"', source)


if __name__ == "__main__":
    unittest.main()
