#!/usr/bin/env python3
"""Focused tests for the append-only GX10 image receipt builder."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "build_gx10_container.py"
SPEC = importlib.util.spec_from_file_location("build_gx10_container_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


class AtomicReceiptTests(unittest.TestCase):
    def test_publishes_complete_receipt_and_refuses_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "receipt.json"
            expected = {"schema": "muser.gx10-container.receipt.v1", "status": "built"}
            BUILD.write_receipt_atomic(receipt, expected)
            self.assertEqual(json.loads(receipt.read_text(encoding="utf-8")), expected)

            with self.assertRaises(FileExistsError):
                BUILD.write_receipt_atomic(receipt, {"status": "replacement"})
            self.assertEqual(json.loads(receipt.read_text(encoding="utf-8")), expected)

    def test_adapter_digest_binds_exporter_and_cuda_patch(self) -> None:
        hashes = {name: "00" * 32 for name in BUILD.FILES}
        baseline = BUILD.adapter_digest(hashes)
        for changed in ("spark_kv_export.cpp", "muser_cuda_metal_compat.patch"):
            mutated = dict(hashes)
            mutated[changed] = "01" * 32
            self.assertNotEqual(BUILD.adapter_digest(mutated), baseline)


if __name__ == "__main__":
    unittest.main()
