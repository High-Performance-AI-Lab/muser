"""CPU-only contracts for the native onboarding control bridge."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "gx10" / "vllm" / "muser_native_prefilld.py"
SPEC = importlib.util.spec_from_file_location("muser_native_prefilld_under_test", MODULE_PATH)
assert SPEC and SPEC.loader
native = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native)


SHA = "a" * 64


def checkpoint_digest(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["filename"].encode())
        digest.update(b"\0")
        digest.update(str(row["bytes"]).encode())
        digest.update(b"\0")
        digest.update(row["sha256"].encode())
        digest.update(b"\n")
    return digest.hexdigest()


class NativePrefilldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def fixture(self) -> tuple[Path, dict]:
        pki = self.root / "pki"
        pki.mkdir()
        for name in ("gx10.cert.pem", "gx10.key.pem", "ca.cert.pem", "hmac.key"):
            (pki / name).write_text("fixture\n", encoding="utf-8")
        docker = self.root / "docker"
        docker.write_text("fixture\n", encoding="utf-8")
        docker.chmod(0o700)
        checkpoint = self.root / "checkpoint"
        checkpoint.mkdir()
        work = self.root / "work"
        work.mkdir()
        identity = {
            "schema": native.RUNTIME_SCHEMA,
            "status": "frozen",
            "source_runtime_identity_sha256": native.SOURCE_RUNTIME_IDENTITY_SHA256,
            "product_lane": "fixture",
            "checkpoint": {
                "revision": "d" * 40,
                "artifact_sha256": SHA,
            },
            "producer_image": {"image_id": "sha256:" + SHA},
            "vllm_overlay": {"adapter_sha256": SHA, "vllm_commit": "e" * 40},
            "rope_cache": {"schema": native.ROPE_CACHE_SCHEMA, "bytes": 64, "sha256": SHA},
            "consumer": {
                "sha256": SHA,
                "tokenizer_sha256": SHA,
                "chat_template_sha256": SHA,
                "context_policy_sha256": SHA,
                "target_cache_identity_sha256": SHA,
            },
            "onboarding_qualification": {},
            "evidence": ["fixture"],
            "ledger_basis": "fixture",
        }
        identity_path = self.root / "identity.json"
        identity_path.write_text(json.dumps(identity), encoding="utf-8")
        config = {
            "schema": native.SCHEMA,
            "schema_version": 1,
            "listen_host": "127.0.0.1",
            "listen_port": 29591,
            "certificate_chain": str(pki / "gx10.cert.pem"),
            "private_key": str(pki / "gx10.key.pem"),
            "peer_ca": str(pki / "ca.cert.pem"),
            "peer_leaf_sha256": [SHA],
            "receiver_server_name": "muser-receiver",
            "receiver_leaf_sha256": SHA,
            "hmac_key_file": str(pki / "hmac.key"),
            "hmac_key_id": "fixture",
            "hmac_epoch": 1,
            "generation_ledger": str(self.root / "generation.json"),
            "work_dir": str(work),
            "container_runtime": str(docker),
            "container_image": "sha256:" + SHA,
            "container_name": "muser-native-fixture",
            "runtime_identity": str(identity_path),
            "checkpoint_dir": str(checkpoint),
            "timeout_seconds": 900,
            "max_context": 131072,
            "checkpoint_artifact_sha256": SHA,
            "checkpoint_revision": "d" * 40,
            "model_sha256": SHA,
            "model_revision": "d" * 40,
            "tokenizer_revision": "d" * 40,
            "tokenizer_sha256": SHA,
            "chat_template_sha256": SHA,
            "context_policy_sha256": SHA,
            "adapter_sha256": SHA,
            "target_cache_identity_sha256": SHA,
            "vllm_commit": "e" * 40,
            "producer_socket": str(work / "producer.sock"),
            "startup_receipt": str(work / "startup.json"),
            "rope_cache_output": str(work / "rope.bin"),
            "rope_cache_bytes": 64,
            "rope_cache_sha256": SHA,
        }
        path = self.root / "handoff.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        return path, config

    def test_load_config_binds_every_runtime_identity_root(self) -> None:
        path, _ = self.fixture()
        config = native.load_config(path)
        self.assertEqual(config["container_image"], "sha256:" + SHA)
        self.assertEqual(config["checkpoint_revision"], "d" * 40)
        value = json.loads(path.read_text())
        value["adapter_sha256"] = "b" * 64
        path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(native.NativePrefilldError, "differs from runtime identity"):
            native.load_config(path)

    def test_legacy_rope_cache_identity_is_refused_by_name(self) -> None:
        path, config = self.fixture()
        identity_path = Path(config["runtime_identity"])
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        identity["rope_cache"]["schema"] = "muser.vllm-rope-cache.v1"
        identity_path.write_text(json.dumps(identity), encoding="utf-8")
        with self.assertRaisesRegex(native.NativePrefilldError, "RoPE cache identity"):
            native.load_config(path)

    def test_checkpoint_manifest_is_hashed_file_by_file_and_as_an_aggregate(self) -> None:
        path, _ = self.fixture()
        config = native.load_config(path)
        rows = []
        for name, payload in (("a.json", b"one"), ("b.bin", b"two")):
            artifact = config["checkpoint_dir"] / name
            artifact.write_bytes(payload)
            rows.append(
                {
                    "filename": name,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        config["identity"]["checkpoint"] = {
            "repository": "fixture/repo",
            "revision": "d" * 40,
            "directory": "checkpoint",
            "total_bytes": sum(row["bytes"] for row in rows),
            "artifact_sha256": checkpoint_digest(rows),
            "files": rows,
            "runtime_receipt": "fixture",
            "runtime_receipt_sha256": SHA,
        }
        native.validate_checkpoint(config)
        (config["checkpoint_dir"] / "b.bin").write_bytes(b"bad")
        with self.assertRaisesRegex(native.NativePrefilldError, "SHA-256 differs"):
            native.validate_checkpoint(config)

    def test_producer_receipt_must_bind_prompt_generation_and_streaming_phases(self) -> None:
        request = {"prompt_token_ids": [1, 2, 3]}
        transfer = "muser-remote-1-7"
        value = {
            "schema": native.CLIENT_RECEIPT_SCHEMA,
            "response": {
                "status": "ok",
                "request_id": transfer,
                "prompt_token_count": 3,
                "producer_receipt": {
                    "schema": native.PRODUCER_RECEIPT_SCHEMA,
                    "producer_mode": "native",
                    "vllm_commit": "e" * 40,
                    "prompt_token_count": 3,
                    "prefix_cut": 0,
                    "token_ids_sha256": native.token_digest([1, 2, 3]),
                    "handoff": {
                        "transfer_id": transfer,
                        "generation": 7,
                        "ack": True,
                        "payload_bytes": 100,
                        "payload_wire_ns": 10,
                        "payload_wire_source": "linux-tcp-info-busy-time-v1",
                        "payload_pacing_bps": 8_000_000_000,
                        "segments": 52,
                        "transfer_start_unix_ns": 1_000,
                        "first_segment_sent_unix_ns": 1_010,
                        "transfer_acked_unix_ns": 2_000,
                    },
                    "phase_ns": {
                        "first_segment_sent_offset": 5,
                        "d2h_complete_offset": 20,
                        "connector_total": 30,
                    },
                },
            },
        }
        receipt = native.validate_producer_receipt(value, request, transfer, 7, "e" * 40)
        self.assertEqual(receipt["prefill_end_unix_ns"], 1_020)
        self.assertEqual(receipt["payload_wire_ns"], 10)
        value["response"]["producer_receipt"]["handoff"]["generation"] = 8
        with self.assertRaisesRegex(native.NativePrefilldError, "bind the transfer"):
            native.validate_producer_receipt(value, request, transfer, 7, "e" * 40)

    def test_producer_config_contains_paths_not_secret_bytes(self) -> None:
        path, _ = self.fixture()
        config = native.load_config(path)
        value = native.producer_config(config)
        self.assertEqual(set(value), {"schema", "checkpoint_artifact_sha256", "checkpoint_revision", "vllm_commit", "connector"})
        self.assertEqual(value["connector"]["hmac_key_file"], "/run/muser/pki/hmac.key")
        self.assertNotIn("hmac_key", value["connector"])


if __name__ == "__main__":
    unittest.main()
