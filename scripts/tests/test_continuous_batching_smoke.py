from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import continuous_batching_smoke as smoke


class ContinuousBatchingSmokeTests(unittest.TestCase):
    def test_derived_prompts_are_distinct_and_preserve_bos_and_length(self) -> None:
        source = list(range(128))
        prompts = smoke.derive_prompts(source)
        self.assertEqual(len(prompts), 4)
        self.assertEqual({len(prompt) for prompt in prompts}, {len(source)})
        self.assertEqual([prompt[0] for prompt in prompts], [source[0]] * 4)
        self.assertEqual(len({smoke.token_digest(prompt) for prompt in prompts}), 4)

    def test_response_tokens_use_target_ids_and_raw_bytes(self) -> None:
        value = {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {"id": 7, "bytes": [65], "top_logprobs": []},
                            {"id": 9, "bytes": [0xC3, 0xA9], "top_logprobs": []},
                        ]
                    }
                }
            ]
        }
        tokens, raw = smoke.response_tokens(value)
        self.assertEqual(tokens, [7, 9])
        self.assertEqual(raw, b"A\xc3\xa9")
        expected = hashlib.sha256(b"\x07\0\0\0\x09\0\0\0").hexdigest()
        self.assertEqual(smoke.token_digest(tokens), "sha256:" + expected)

    def test_cases_pin_rng_and_keep_grammar_per_request(self) -> None:
        plain = smoke.case_payload([1, 2], 0, "plain", 16)
        numeric = smoke.case_payload([1, 2], 2, "numeric", 16)
        uppercase = smoke.case_payload([1, 2], 3, "uppercase", 16)
        self.assertNotEqual(plain["seed"], numeric["seed"])
        self.assertNotIn("grammar", plain)
        self.assertEqual(numeric["grammar"], 'root ::= [0-9]+')
        self.assertEqual(uppercase["grammar"], 'root ::= [A-Z]+')
        self.assertFalse(numeric["ignore_eos"])

    def test_server_command_is_explicit_metal_without_ane_flags(self) -> None:
        command = smoke.server_command(
            Path("muser"),
            Path("model.gguf"),
            4964,
            4,
            4096,
            Path("api.key"),
            "shutdown",
            3600,
        )
        self.assertEqual(command[command.index("--backend") + 1], "metal")
        self.assertEqual(command[command.index("--parallel") + 1], "4")
        self.assertNotIn("ane", " ".join(command).lower())


if __name__ == "__main__":
    unittest.main()
