from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gx10" / "vllm"))

from muser_vllm.composite_bundle import (
    CompositeBundleError,
    CompositeBundleWriter,
    EXPECTED_LAYERS,
    bundle_root_sha256,
    expected_layer_bytes,
    load_hmac_key,
    read_bundle_manifest,
    read_layer_payload,
)


class CompositeBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.key_path = self.root / "composite.key"
        self.key_path.write_text("5a" * 32 + "\n")
        os.chmod(self.key_path, 0o600)
        self.key = load_hmac_key(self.key_path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_bundle(self, name: str = "bundle") -> tuple[Path, dict[str, object]]:
        destination = self.root / name
        writer = CompositeBundleWriter(
            destination,
            token_ids=[1, 2],
            source_checkpoint_revision="redhat-test-revision",
            source_checkpoint_artifact_sha256="1" * 64,
            source_engine_mode="native",
            hmac_key_id="test-key",
            hmac_key=self.key,
        )
        size = expected_layer_bytes(1)
        for layer in range(EXPECTED_LAYERS):
            writer.write_layer(layer, bytes([layer]) * size)
        return destination, writer.commit()

    def test_bundle_publishes_only_a_complete_authenticated_closure(self) -> None:
        destination, written = self.write_bundle()
        manifest = read_bundle_manifest(
            destination,
            key=self.key,
            expected_key_id="test-key",
            expected_source_artifact_sha256="1" * 64,
        )
        self.assertEqual(manifest, written)
        self.assertEqual(len(bundle_root_sha256(manifest)), 64)
        self.assertEqual(read_layer_payload(destination, manifest, 7), bytes([7]) * 1024)
        self.assertFalse((self.root / ".bundle.publish.lock").exists())

    def test_wrong_key_payload_mutation_and_extra_objects_fail_closed(self) -> None:
        destination, manifest = self.write_bundle()
        with self.assertRaisesRegex(CompositeBundleError, "HMAC rejected"):
            read_bundle_manifest(
                destination,
                key=b"x" * 32,
                expected_key_id="test-key",
            )
        layer = destination / manifest["layer_files"][3]["path"]
        layer.write_bytes(b"x" * layer.stat().st_size)
        with self.assertRaisesRegex(CompositeBundleError, "digest/length"):
            read_layer_payload(destination, manifest, 3)
        (destination / "unexpected").write_bytes(b"")
        with self.assertRaisesRegex(CompositeBundleError, "unexpected objects"):
            read_bundle_manifest(
                destination,
                key=self.key,
                expected_key_id="test-key",
            )

    def test_unknown_manifest_field_and_duplicate_destination_are_rejected(self) -> None:
        destination, manifest = self.write_bundle()
        manifest["unknown"] = True
        (destination / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(CompositeBundleError, "keys are not closed"):
            read_bundle_manifest(
                destination,
                key=self.key,
                expected_key_id="test-key",
            )
        with self.assertRaisesRegex(CompositeBundleError, "already exists"):
            CompositeBundleWriter(
                destination,
                token_ids=[1, 2],
                source_checkpoint_revision="redhat-test-revision",
                source_checkpoint_artifact_sha256="1" * 64,
                source_engine_mode="native",
                hmac_key_id="test-key",
                hmac_key=self.key,
            )

    def test_key_permissions_and_incomplete_bundle_are_rejected(self) -> None:
        os.chmod(self.key_path, 0o644)
        with self.assertRaisesRegex(CompositeBundleError, "group/world"):
            load_hmac_key(self.key_path)
        os.chmod(self.key_path, 0o600)
        writer = CompositeBundleWriter(
            self.root / "incomplete",
            token_ids=[1, 2],
            source_checkpoint_revision="redhat-test-revision",
            source_checkpoint_artifact_sha256="1" * 64,
            source_engine_mode="native",
            hmac_key_id="test-key",
            hmac_key=self.key,
        )
        writer.write_layer(0, bytes(expected_layer_bytes(1)))
        with self.assertRaisesRegex(CompositeBundleError, "incomplete"):
            writer.commit()
        writer.abort()


if __name__ == "__main__":
    unittest.main()
