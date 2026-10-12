from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import representative_target_smoke as smoke


def telemetry_snapshot(tokens: int, offset: float = 0.0) -> dict[str, object]:
    phases = {
        name: {"samples": 1, "total_ms": 1.0 + offset, "mean_ms": 1.0 + offset}
        for name in smoke.SNAPSHOT_PHASES
    }
    return {
        "schema_version": 1,
        "_queue_depth": 0,
        "_decode": {"completion_tokens": tokens},
        "_phases": {
            **phases,
            "last_request_decode_tok_s": 2.0 + offset,
        },
        "wire": {"ttft_ms": {"p50": 10.0 + offset, "p95": 20.0 + offset}},
    }


class RepresentativeTargetSmokeTests(unittest.TestCase):
    def test_token_digest_is_little_endian_u32(self) -> None:
        expected = hashlib.sha256(b"\x01\x00\x00\x00\x00\x01\x00\x00").hexdigest()
        self.assertEqual(smoke.token_digest([1, 256]), "sha256:" + expected)

    def test_comparator_receipt_binds_the_exact_binary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "llama-server"
            binary.write_bytes(b"pinned comparator")
            receipt = root / "source-receipt.json"
            receipt.write_text(
                json.dumps(
                    {
                        "schema": "muser.llama_comparator.source_receipt.v3",
                        "executed": False,
                        "source_commit": "a" * 40,
                        "build": {"metal": True},
                        "artifacts": {
                            "llama-server": {
                                "bytes": binary.stat().st_size,
                                "sha256": smoke.sha256(binary),
                            }
                        },
                    }
                )
            )
            value, _ = smoke.validate_comparator(binary, receipt)
            self.assertEqual(value["source_commit"], "a" * 40)
            binary.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "differs"):
                smoke.validate_comparator(binary, receipt)

    def test_server_origins_are_loopback_only(self) -> None:
        self.assertEqual(smoke.loopback_origin("http://127.0.0.1:4949").port, 4949)
        for value in (
            "https://127.0.0.1:4949",
            "http://example.com:4949",
            "http://127.0.0.1:4949/path",
            "http://127.0.0.1",
        ):
            with self.assertRaisesRegex(RuntimeError, "loopback"):
                smoke.loopback_origin(value)

    def test_muser_launch_binds_metallib_only_to_muser_environment(self) -> None:
        parts = smoke.loopback_origin("http://127.0.0.1:4949")
        metallib = Path("pinned-llama.metallib")
        _, muser_environment = smoke.server_command(
            "muser", Path("muser"), Path("model.gguf"), metallib,
            parts, "token", 60,
        )
        _, llama_environment = smoke.server_command(
            "llama", Path("llama-server"), Path("model.gguf"), None,
            parts, "token", 60,
        )
        self.assertEqual(
            muser_environment["MUSER_GGML_METALLIB"], str(metallib.resolve())
        )
        self.assertNotIn("MUSER_GGML_METALLIB", llama_environment)
        command, _ = smoke.server_command(
            "muser", Path("muser"), Path("model.gguf"), metallib,
            parts, "token", 60, Path("private-api-key"),
        )
        self.assertEqual(command[-2:], ["--api-key-file", "private-api-key"])
        with self.assertRaisesRegex(RuntimeError, "requires"):
            smoke.server_command(
                "muser", Path("muser"), Path("model.gguf"), None,
                parts, "token", 60,
            )

    def test_muser_build_check_binds_hash_version_and_no_ane_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / "muser"
            binary.write_bytes(
                b"#!/bin/sh\n# this binary was built without the ane-coreml feature\n"
                b"echo 'muser 0.1.0-test'\n"
            )
            binary.chmod(0o700)
            artifact, version = smoke.validate_muser_build(binary, smoke.sha256(binary))
            self.assertEqual(artifact["sha256"], smoke.sha256(binary))
            self.assertEqual(version, "muser 0.1.0-test")
            with self.assertRaisesRegex(RuntimeError, "differs"):
                smoke.validate_muser_build(binary, "0" * 64)

    def test_metallib_check_binds_sibling_source_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metallib = root / "llama.metallib"
            metallib.write_bytes(b"pinned metal library")
            (root / "source-receipt.json").write_text(
                json.dumps(
                    {
                        "schema": "muser.llama_metallib.source_receipt.v1",
                        "source_commit": smoke.PINNED_LLAMA_COMMIT,
                        "artifact_name": metallib.name,
                        "binary_size_bytes": metallib.stat().st_size,
                        "binary_sha256": smoke.sha256(metallib),
                    }
                )
            )
            artifact, receipt = smoke.validate_metallib(metallib)
            self.assertEqual(artifact["sha256"], smoke.sha256(metallib))
            self.assertEqual(receipt["path"], str((root / "source-receipt.json").resolve()))

    def test_request_telemetry_delta_attributes_all_exposed_target_phases(self) -> None:
        before = telemetry_snapshot(1)
        after = telemetry_snapshot(5)
        increments = {
            "queue": (1, 0.25),
            "prefill": (1, 12.5),
            "sampling": (4, 1.0),
            "grammar": (0, 0.0),
            "detokenization": (5, 0.75),
            "enqueue_write": (6, 0.5),
        }
        for name, (samples, total_ms) in increments.items():
            prior = before["_phases"][name]
            current = after["_phases"][name]
            current["samples"] = prior["samples"] + samples
            current["total_ms"] = prior["total_ms"] + total_ms
            current["mean_ms"] = current["total_ms"] / current["samples"]
        after["_phases"]["last_request_decode_tok_s"] = 9.5
        value = smoke.request_telemetry_delta(
            before,
            after,
            expected_tokens=4,
            client_ttft_ns=7_000_000,
            final_timings={"prompt_ms": 12.0, "predicted_ms": 40.0},
        )
        self.assertEqual(value["schema"], "muser.request-telemetry-delta.v1")
        self.assertEqual(value["completion_tokens"]["delta"], 4)
        self.assertEqual(value["phases"]["prefill"]["total_ms"], 12.5)
        self.assertEqual(value["phases"]["grammar"]["samples"], 0)
        self.assertEqual(value["decode"]["last_request_decode_tok_s"], 9.5)
        self.assertFalse(value["ttft"]["snapshot_delta_available"])

    def test_request_telemetry_delta_rejects_counter_regression_or_contamination(self) -> None:
        before = telemetry_snapshot(5)
        after = telemetry_snapshot(4)
        with self.assertRaisesRegex(RuntimeError, "completion-token delta"):
            smoke.request_telemetry_delta(
                before, after, 1, 1, {"prompt_ms": 1.0, "predicted_ms": 1.0}
            )
        after = telemetry_snapshot(6)
        after["_phases"]["prefill"]["total_ms"] = 0.5
        with self.assertRaisesRegex(RuntimeError, "counter regressed: prefill"):
            smoke.request_telemetry_delta(
                before, after, 1, 1, {"prompt_ms": 1.0, "predicted_ms": 1.0}
            )

    def test_snapshot_validation_rejects_unknown_or_nonfinite_phase_data(self) -> None:
        value = telemetry_snapshot(0)
        value["_phases"]["unknown"] = {"samples": 0, "total_ms": 0, "mean_ms": 0}
        with self.assertRaisesRegex(RuntimeError, "phase surface"):
            smoke.telemetry_snapshot_view(value)
        value = telemetry_snapshot(0)
        value["_phases"]["sampling"]["total_ms"] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "finite"):
            smoke.telemetry_snapshot_view(value)

    def test_snapshot_json_rejects_duplicate_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            json.loads('{"schema_version":1,"schema_version":1}', object_pairs_hook=smoke._reject_duplicate_keys)


if __name__ == "__main__":
    unittest.main()
