from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "scripts" / "evaluate_nvfp4_quality.py"
SPEC = importlib.util.spec_from_file_location("evaluate_nvfp4_quality", PATH)
assert SPEC is not None and SPEC.loader is not None
quality = importlib.util.module_from_spec(SPEC)
sys.modules["evaluate_nvfp4_quality"] = quality
SPEC.loader.exec_module(quality)


def row(position: int, top: int, target_nll: float, offset: int = 0) -> dict:
    candidates = [
        {"token_id": token, "logit": float(20 - rank + offset), "quantized_u16": 65000 - rank}
        for rank, token in enumerate([top, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ]
    return {
        "chunk": 0,
        "position": position,
        "input_token_id": 100 + position,
        "target_token_id": 200 + position,
        "target_nll": target_nll,
        "row_scale": 0.001,
        "minimum_log_probability": -70.0,
        "candidates": candidates,
    }


class EvaluateNvfp4QualityTests(unittest.TestCase):
    def test_perplexity_capture_binds_model_and_logits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logits = root / "logits"
            logits.write_bytes(b"fixture")
            capture = root / "capture.json"
            capture.write_text(
                json.dumps(
                    {
                        "schema": "muser.llama-perplexity-capture.v1",
                        "status": "passed",
                        "identity": "quality-id",
                        "accelerator_touched": True,
                        "runtime": {
                            "context_length": 192,
                            "batch_size": 192,
                            "ubatch_size": 192,
                            "threads": 20,
                            "chunks": 1,
                            "gpu_layers": 99,
                            "flash_attention": True,
                            "kv_cache": "f16",
                        },
                        "artifacts": {
                            "model_receipt": {"sha256": "receipt"},
                            "teacher_evidence": {
                                "quantized_logits": {
                                    "sha256": quality.sha256(logits),
                                    "size_bytes": logits.stat().st_size,
                                }
                            },
                        },
                    }
                )
            )
            value = quality.perplexity_capture(
                capture,
                logits_path=logits,
                identity="quality-id",
                context_length=192,
                batch_size=192,
                ubatch_size=192,
                threads=20,
                require_model_receipt=True,
            )
            self.assertEqual(value["status"], "passed")
            logits.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "does not bind logits"):
                quality.perplexity_capture(
                    capture,
                    logits_path=logits,
                    identity="quality-id",
                    context_length=192,
                    batch_size=192,
                    ubatch_size=192,
                    threads=20,
                    require_model_receipt=True,
                )

    def test_row_drift_is_measured_not_zero_gated(self) -> None:
        measured = quality.compare_rows(
            [row(0, 1, 2.0), row(1, 11, 3.0)],
            [row(0, 1, 2.25), row(1, 12, 2.5, 1)],
        )
        self.assertEqual(measured["teacher_forced_greedy_divergences"], 1)
        self.assertEqual(measured["teacher_forced_greedy_divergence_rate"], 0.5)
        self.assertAlmostEqual(measured["target_nll_drift"]["mean_signed"], 0.125)
        self.assertFalse(measured["boundary"]["argmax_equal"])

    def test_greedy_stream_counts_length_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "left.tokens"
            right = root / "right.tokens"
            left.write_text("1\n2\n3\n")
            right.write_text("1\n4\n")
            measured = quality.greedy_metrics(left, right)
            self.assertEqual(measured["mismatches"], 2)
            self.assertEqual(measured["first_mismatch"], 1)
            self.assertAlmostEqual(measured["divergence_rate"], 2 / 3)

    def test_dflash_reports_bind_identity_receipt_and_expected_stream(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def write(name: str, *, receipt: bool) -> Path:
                path = root / name
                path.write_text(
                    json.dumps(
                        {
                            "schema": "muser.llama-quality-capture.v1",
                            "status": "passed",
                            "identity": "quality-id",
                            "mode": "dflash",
                            "accelerator_touched": True,
                            "output_tokens_generated": 256,
                            "expected_tokens_match": True,
                            "dflash_route_active": True,
                            "artifacts": {
                                "model_receipt": {"sha256": "a"} if receipt else None,
                                "expected_tokens": {"sha256": "b"},
                            },
                            "outputs": {"generated_tokens": "tokens"},
                            "llama": {
                                "timings": {"draft_n": 10, "draft_n_accepted": 8}
                            },
                        }
                    )
                )
                return path

            nvfp4 = write("nvfp4.json", receipt=True)
            kquant = write("kquant.json", receipt=False)
            measured = quality.dflash_metrics(
                nvfp4, kquant, identity="quality-id"
            )
            self.assertEqual(measured["nvfp4"]["accepted"], 8)
            self.assertEqual(measured["acceptance_delta"], 0.0)
            self.assertEqual(measured["nvfp4"]["report_sha256"], quality.sha256(nvfp4))

            value = json.loads(nvfp4.read_text())
            value["identity"] = "other-id"
            nvfp4.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, "contract mismatch"):
                quality.dflash_metrics(nvfp4, kquant, identity="quality-id")


if __name__ == "__main__":
    unittest.main()
