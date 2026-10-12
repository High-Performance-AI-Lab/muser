from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def load(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


codec = load("nvfp4_codec")
repack = load("nvfp4_to_f16_gguf")
native_repack = load("nvfp4_to_native_gguf")
oracle = load("verify_nvfp4_oracle")
capture = load("capture_llama_quality")


class Nvfp4CodecTests(unittest.TestCase):
    def test_e2m1_nibble_order_and_scale_order(self) -> None:
        packed = np.array(
            [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]],
            dtype=np.uint8,
        )
        scales = np.array([[0x38]], dtype=np.uint8)  # E4M3FN 1.0
        got = codec.dequantize(packed, scales, 0.5)
        expected = codec.E2M1.reshape(1, 16) * np.float32(0.5)
        np.testing.assert_array_equal(got, expected)

    def test_compressed_tensors_global_scale_is_reciprocal(self) -> None:
        self.assertEqual(float(codec.compressed_tensors_scale2(2.0)), 0.5)
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            codec.compressed_tensors_scale2(0.0)
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            codec.compressed_tensors_scale2(float("inf"))

    def test_e4m3fn_known_values(self) -> None:
        self.assertEqual(float(codec.E4M3FN[0x00]), 0.0)
        self.assertEqual(float(codec.E4M3FN[0x38]), 1.0)
        self.assertEqual(float(codec.E4M3FN[0x40]), 2.0)
        self.assertEqual(float(codec.E4M3FN[0xB8]), -1.0)
        self.assertTrue(np.isnan(codec.E4M3FN[0x7F]))

    def test_rejects_wrong_scale_geometry(self) -> None:
        with self.assertRaisesRegex(ValueError, "geometry"):
            codec.dequantize(
                np.zeros((1, 8), dtype=np.uint8),
                np.zeros((1, 2), dtype=np.uint8),
                1.0,
            )


class Nvfp4RepackTests(unittest.TestCase):
    def test_rope_row_order_matches_pinned_llama_conversion(self) -> None:
        np.testing.assert_array_equal(
            repack.rope_row_order(8, 2),
            np.array([0, 2, 1, 3, 4, 6, 5, 7], dtype=np.int64),
        )
        with self.assertRaisesRegex(ValueError, "RoPE row geometry"):
            repack.rope_row_order(7, 2)

    def test_muse_layer_norm_shift_is_applied_in_float32(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "norm.bin"
            np.array([0x3F00, 0xBF00], dtype="<u2").tofile(path)  # +0.5, -0.5 BF16
            tensor = repack.SafeTensor(path, "BF16", (2,), 0, 4)
            out = Path(directory) / "out.bin"
            with out.open("wb") as stream:
                written = repack.write_bf16(
                    stream, tensor, repack.TYPE_F32, np.float32(1.0)
                )
            self.assertEqual(written, 8)
            np.testing.assert_array_equal(
                np.fromfile(out, dtype="<f4"), np.array([1.5, 0.5], dtype=np.float32)
            )

    def test_muse_name_mapping(self) -> None:
        self.assertEqual(
            repack.hf_name("blk.17.attn_gate.weight"),
            "model.language_model.layers.17.self_attn.gate_proj",
        )
        self.assertEqual(
            repack.hf_name("blk.0.post_ffw_norm.weight"),
            "model.language_model.layers.0.post_feedforward_layernorm.weight",
        )
        self.assertIsNone(repack.hf_name("blk.51.attn_q_norm.weight"))
        self.assertEqual(
            repack.hf_name("token_embd.weight"),
            "model.language_model.embed_tokens.weight",
        )

    def test_output_types(self) -> None:
        matrix = repack.GgufTensor("blk.0.attn_q.weight", (16, 8), 12, 0)
        norm = repack.GgufTensor("blk.0.attn_norm.weight", (8,), 0, 0)
        self.assertEqual(repack.output_type(matrix), repack.TYPE_F16)
        self.assertEqual(repack.output_type(norm), repack.TYPE_F32)

    def test_checkpoint_receipt_binds_checkpoint_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            artifact = checkpoint / "config.json"
            artifact.write_bytes(b"{}")
            digest = repack.sha256(artifact)
            rows = [{"path": "config.json", "size": 2, "sha256": digest}]
            receipt = root / "checkpoint.json"
            receipt.write_text(
                json.dumps(
                    {
                        "schema": "muser.hf-checkpoint.receipt.v1",
                        "checkpoint": str(checkpoint),
                        "artifact_sha256": repack.receipt_hf_checkpoint.artifact_digest(rows),
                        "files": rows,
                    }
                )
            )
            parsed = repack.read_checkpoint_receipt(receipt, checkpoint)
            self.assertEqual(parsed["files"], rows)
            artifact.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "differs from receipt"):
                repack.read_checkpoint_receipt(receipt, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                repack.read_checkpoint_receipt(receipt, root / "other")

    def test_receipt_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "receipt.json"
            repack.write_receipt(receipt, {"schema": "test"})
            self.assertEqual(json.loads(receipt.read_text()), {"schema": "test"})
            with self.assertRaisesRegex(RuntimeError, "refusing to replace"):
                repack.write_receipt(receipt, {"schema": "replacement"})


class NativeNvfp4RepackTests(unittest.TestCase):
    def test_native_repack_requires_checkpoint_and_gguf_rms_epsilon_bits(self) -> None:
        key = native_repack.RMS_EPS_KEY

        def scalar(value: float) -> bytes:
            encoded = key.encode()
            return (
                struct.pack("<Q", len(encoded))
                + encoded
                + struct.pack("<If", 6, value)
            )

        reference = repack.GgufReference([(key, scalar(1.0e-5))], [], 0)
        contract = native_repack.validate_rms_epsilon(
            reference, {"rms_norm_eps": 1.0e-5}
        )
        self.assertEqual(contract["bits"], "0x3727c5ac")
        with self.assertRaisesRegex(RuntimeError, "differs"):
            native_repack.validate_rms_epsilon(
                reference, {"rms_norm_eps": 1.0e-6}
            )
        with self.assertRaisesRegex(RuntimeError, "missing rms_norm_eps"):
            native_repack.validate_rms_epsilon(reference, {})

    def test_native_rows_keep_exact_four_point_five_bit_components(self) -> None:
        matrix = repack.GgufTensor("blk.0.attn_q.weight", (32, 8), 12, 0)
        norm = repack.GgufTensor("blk.0.attn_norm.weight", (8,), 0, 0)
        reference = repack.GgufReference([], [matrix, norm], 0)
        rows = native_repack.native_rows(reference)
        self.assertEqual(
            [(row.name, row.tensor_type, row.size) for row in rows[:3]],
            [
                ("blk.0.attn_q.weight", native_repack.TYPE_NVFP4_E2M1, 128),
                ("blk.0.attn_q.weight.nvfp4_scale", native_repack.TYPE_F8_E4M3FN, 16),
                ("blk.0.attn_q.weight.nvfp4_scale2", repack.TYPE_F32, 4),
            ],
        )
        self.assertEqual(rows[3].component, "plain")
        self.assertEqual(rows[3].tensor_type, repack.TYPE_F32)

    def test_native_metadata_binds_precision_schema_and_source(self) -> None:
        reference = repack.GgufReference([], [], 0)
        raw = b"".join(native_repack.native_metadata(reference, "ab" * 32))
        self.assertIn(b"muser.weight_precision", raw)
        self.assertIn(b"nvfp4", raw)
        self.assertIn(native_repack.SCHEMA.encode(), raw)
        self.assertIn(("ab" * 32).encode(), raw)

    def test_w4a4_rows_bind_one_input_scale_per_matrix(self) -> None:
        matrix = repack.GgufTensor("blk.0.attn_q.weight", (32, 8), 12, 0)
        reference = repack.GgufReference([], [matrix], 0)
        rows = native_repack.native_rows(reference, "nvfp4")
        self.assertEqual(len(rows), 4)
        self.assertEqual(
            rows[-1].name,
            "blk.0.attn_q.weight.nvfp4_input_scale_inv",
        )
        self.assertEqual(rows[-1].component, "input_scale_inv")
        raw = b"".join(
            native_repack.native_metadata(reference, "ab" * 32, "nvfp4")
        )
        self.assertIn(b"muser.activation_precision", raw)


class Nvfp4OracleTests(unittest.TestCase):
    def test_sample_rows_covers_edges_and_middle(self) -> None:
        self.assertEqual(oracle.sampled_rows(1), [0])
        self.assertEqual(oracle.sampled_rows(2), [0, 1])
        self.assertEqual(oracle.sampled_rows(7), [0, 3, 6])
        with self.assertRaisesRegex(RuntimeError, "no rows"):
            oracle.sampled_rows(0)


class CaptureLlamaQualityTests(unittest.TestCase):
    def test_token_fixture_is_exclusive_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.txt"
            capture.write_tokens(path, [0, 1, 0xFFFFFFFF])
            self.assertEqual(capture.token_file(path), [0, 1, 0xFFFFFFFF])
            with self.assertRaisesRegex(RuntimeError, "refusing to replace"):
                capture.write_tokens(path, [2])


if __name__ == "__main__":
    unittest.main()
