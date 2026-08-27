from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "receipt_hf_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("receipt_hf_checkpoint", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
receipt_hf_checkpoint = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(receipt_hf_checkpoint)


class ReceiptHfCheckpointTests(unittest.TestCase):
    def test_manifest_and_artifact_digest_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_bytes(b"{}\n")
            (checkpoint / "weights.safetensors").write_bytes(b"weights")
            (checkpoint / ".cache" / "huggingface").mkdir(parents=True)
            (checkpoint / ".cache" / "huggingface" / "ignored").write_text("x")

            weight_digest = hashlib.sha256(b"weights").hexdigest()
            manifest = {
                "id": "owner/model",
                "sha": "a" * 40,
                "siblings": [
                    {"rfilename": "config.json", "size": 3},
                    {
                        "rfilename": "weights.safetensors",
                        "size": 7,
                        "lfs": {"sha256": weight_digest},
                    },
                ],
            }
            manifest_path = root / "api.json"
            manifest_path.write_text(json.dumps(manifest))
            expected, manifest_digest = receipt_hf_checkpoint.load_expected(
                manifest_path, "owner/model", "a" * 40
            )
            self.assertEqual(set(expected), {"config.json", "weights.safetensors"})
            self.assertEqual(
                manifest_digest, hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            )
            files = receipt_hf_checkpoint.checkpoint_files(checkpoint)
            self.assertEqual(set(files), set(expected))

            rows = [
                {"path": name, "size": path.stat().st_size, "sha256": receipt_hf_checkpoint.sha256(path)}
                for name, path in sorted(files.items())
            ]
            self.assertEqual(
                receipt_hf_checkpoint.artifact_digest(rows),
                receipt_hf_checkpoint.artifact_digest(list(rows)),
            )

    def test_manifest_rejects_wrong_revision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "api.json"
            path.write_text(
                json.dumps(
                    {
                        "id": "owner/model",
                        "sha": "b" * 40,
                        "siblings": [{"rfilename": "x", "size": 1}],
                    }
                )
            )
            with self.assertRaisesRegex(RuntimeError, "revision mismatch"):
                receipt_hf_checkpoint.load_expected(
                    path, "owner/model", "a" * 40
                )

    def test_checkpoint_rejects_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.write_text("x")
            (root / "link").symlink_to(target)
            with self.assertRaisesRegex(RuntimeError, "symlink"):
                receipt_hf_checkpoint.checkpoint_files(root)

    def test_receipt_publish_never_clobbers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            receipt_hf_checkpoint.publish(path, {"schema": "first"})
            with self.assertRaisesRegex(RuntimeError, "refusing to replace"):
                receipt_hf_checkpoint.publish(path, {"schema": "second"})
            self.assertEqual(json.loads(path.read_text()), {"schema": "first"})


if __name__ == "__main__":
    unittest.main()
