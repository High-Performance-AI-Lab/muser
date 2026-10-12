from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
PATH = ROOT / "scripts" / "build_nvfp4_e_series_fixtures.py"
SPEC = importlib.util.spec_from_file_location("build_nvfp4_e_series_fixtures", PATH)
assert SPEC is not None and SPEC.loader is not None
fixtures = importlib.util.module_from_spec(SPEC)
sys.modules["build_nvfp4_e_series_fixtures"] = fixtures
SPEC.loader.exec_module(fixtures)


class ESeriesFixtureTests(unittest.TestCase):
    def test_nested_prefixes_share_one_fixed_document(self) -> None:
        tokens = list(range(40_000))
        prefixes = fixtures.nested_prefixes(tokens)
        self.assertEqual(list(prefixes), list(fixtures.E2_LENGTHS))
        self.assertTrue(all(len(prefixes[length]) == length for length in prefixes))
        self.assertEqual(prefixes[2048], prefixes[32768][:2048])
        self.assertEqual(prefixes[2048][0], fixtures.BOS_TOKEN)

    def test_e1_manifest_selects_required_rows(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            entries = {}
            for fixture_id in (*fixtures.E1_SOURCE_IDS, *fixtures.E1_ROUTING_IDS):
                token_path = root / f"{fixture_id}.tokens"
                token_path.write_text("1 2 3\n", encoding="utf-8")
                entries[fixture_id] = {
                    "id": fixture_id,
                    "regime": "long-context" if fixture_id.startswith("long") else "code",
                    "token_file": token_path.name,
                    "output_tokens": 0,
                }
            source = root / "source.json"
            routing = root / "routing.json"
            source.write_text(
                json.dumps(
                    {
                        "schema": fixtures.SCHEMA,
                        "fixtures": [entries[key] for key in fixtures.E1_SOURCE_IDS],
                    }
                ),
                encoding="utf-8",
            )
            routing.write_text(
                json.dumps(
                    {
                        "schema": fixtures.SCHEMA,
                        "fixtures": [entries[key] for key in fixtures.E1_ROUTING_IDS],
                    }
                ),
                encoding="utf-8",
            )
            manifest, receipts = fixtures.e1_manifest(source, routing)
        self.assertEqual(
            [row["id"] for row in manifest["fixtures"]],
            [*fixtures.E1_SOURCE_IDS, *fixtures.E1_ROUTING_IDS],
        )
        self.assertEqual(len(receipts), 7)


if __name__ == "__main__":
    unittest.main()
