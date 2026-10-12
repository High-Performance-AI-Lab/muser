"""Closed-source-manifest checks for the mounted NVFP4 runtime overlay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from scripts.gx10.vllm.receipt_adapter import SOURCE_KEYS, canonical_sha256
from scripts.gx10.vllm.receipt_cache_identity import derive as derive_cache_identity


ROOT = Path(__file__).resolve().parents[2]
OVERLAY_PATH = ROOT / "release" / "nvfp4-runtime-overlay-v2.json"
NATIVE_IDENTITY_PATH = (
    ROOT / "scripts" / "gx10" / "vllm" / "native_onboarding_identity_v1.json"
)


class Nvfp4RuntimeOverlayTests(unittest.TestCase):
    def test_overlay_closes_every_mounted_source_and_binds_adapter_identity(self) -> None:
        overlay = json.loads(OVERLAY_PATH.read_text(encoding="utf-8"))
        native = json.loads(NATIVE_IDENTITY_PATH.read_text(encoding="utf-8"))
        self.assertEqual(overlay["schema"], "muser.nvfp4-runtime-overlay.v1")
        self.assertIn(overlay["status"], {"candidate", "qualified"})
        self.assertEqual(overlay["base_image_id"], native["producer_image"]["image_id"])
        self.assertEqual(overlay["adapter_sha256"], native["vllm_overlay"]["adapter_sha256"])
        self.assertEqual(
            overlay["target_cache_identity_sha256"],
            native["consumer"]["target_cache_identity_sha256"],
        )

        expected = {
            "scripts/gx10/vllm/resident_producer.py",
            "scripts/gx10/vllm/request_producer.py",
            "scripts/gx10/llamacpp/muser_v2_send.py",
            "scripts/gx10/llamacpp/llamacpp_session_send.py",
            "scripts/gx10/llamacpp/protocol.py",
        }
        expected.update(
            str(path.relative_to(ROOT))
            for path in (ROOT / "scripts" / "gx10" / "vllm" / "muser_vllm").glob("*.py")
        )
        self.assertEqual(set(overlay["mounts"]), expected)
        for logical_name, wanted in overlay["mounts"].items():
            actual = hashlib.sha256((ROOT / logical_name).read_bytes()).hexdigest()
            self.assertEqual(actual, wanted, logical_name)

        adapter = overlay["adapter_identity"]
        self.assertEqual(canonical_sha256(adapter), overlay["adapter_sha256"])
        self.assertEqual(adapter["image_id"], overlay["base_image_id"])
        self.assertEqual(adapter["vllm_commit"], overlay["vllm_commit"])
        for key, source in SOURCE_KEYS.items():
            self.assertEqual(adapter[key], overlay["mounts"][source], source)

        cache = derive_cache_identity(
            native["consumer"]["sha256"], overlay["adapter_sha256"], "native"
        )
        self.assertEqual(
            cache["target_cache_identity_sha256"],
            overlay["target_cache_identity_sha256"],
        )

    def test_startup_keeps_128k_context_with_an_8k_profile_shape(self) -> None:
        overlay = json.loads(OVERLAY_PATH.read_text(encoding="utf-8"))
        startup = overlay["startup"]
        self.assertEqual(startup["max_model_len"], 131072)
        self.assertEqual(startup["max_num_batched_tokens"], 8192)
        self.assertTrue(startup["chunked_prefill"])


if __name__ == "__main__":
    unittest.main()
