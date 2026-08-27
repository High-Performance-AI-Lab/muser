from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import generate_social_card


class GenerateSocialCardTests(unittest.TestCase):
    def test_card_is_deterministic_and_receipt_cited(self) -> None:
        first = generate_social_card.render()
        second = generate_social_card.render()
        self.assertEqual(first, second)
        self.assertIn('width="1200" height="630"', first)
        self.assertIn("3.75–4.26×", first)
        self.assertIn("phase4-disagg-20260820", first)
        self.assertTrue(all(generate_social_card.SOURCES))

    def test_check_accepts_only_current_generated_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "card.svg"
            output.write_text(generate_social_card.render(), encoding="utf-8")
            with mock.patch(
                "sys.argv", ["generate_social_card.py", "--check", "--output", str(output)]
            ):
                self.assertEqual(generate_social_card.main(), 0)


if __name__ == "__main__":
    unittest.main()
