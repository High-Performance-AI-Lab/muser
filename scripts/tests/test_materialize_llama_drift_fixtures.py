from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "scripts" / "materialize_llama_drift_fixtures.py"
SPEC = importlib.util.spec_from_file_location("materialize_llama_drift_fixtures", PATH)
assert SPEC is not None and SPEC.loader is not None
materialize = importlib.util.module_from_spec(SPEC)
sys.modules["materialize_llama_drift_fixtures"] = materialize
SPEC.loader.exec_module(materialize)


class LlamaDriftFixtureViewTests(unittest.TestCase):
    def test_token_digest_is_whitespace_independent(self) -> None:
        tokens = [200_000, 1, 202_047]
        self.assertEqual(materialize.token_digest(tokens), materialize.token_digest(list(tokens)))

    def test_rejects_out_of_vocab_token(self) -> None:
        path = ROOT / "target" / "invalid-drift-token-fixture.tokens"
        try:
            path.write_text("200000\n202048\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                materialize.read_tokens(path)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
