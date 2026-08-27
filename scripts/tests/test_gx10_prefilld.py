from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
import unittest.mock


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "scripts" / "gx10" / "llamacpp" / "muser_prefilld.py"
SPEC = importlib.util.spec_from_file_location("muser_prefilld_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
PREFILLD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFILLD)


class ContainerConfigTests(unittest.TestCase):
    def config(self, root: Path) -> Path:
        for name in (
            "producer.cert.pem",
            "producer.key.pem",
            "ca.cert.pem",
            "handoff.key",
            "sender.py",
            "docker",
        ):
            (root / name).write_bytes(b"fixture")
        adapter = "a" * 64
        image = "sha256:" + "b" * 64
        receipt = {
            "schema": "muser.gx10-container.receipt.v1",
            "status": "built",
            "architecture": "arm64",
            "image_id": image,
            "adapter_sha256": adapter,
            "cuda_matmul": "default",
            "entrypoint": ["/opt/muser/bin/spark_kv_export"],
        }
        (root / "container.json").write_text(json.dumps(receipt))
        value = {
            "schema_version": 8,
            "listen_host": "127.0.0.1",
            "listen_port": 29591,
            "certificate_chain": "producer.cert.pem",
            "private_key": "producer.key.pem",
            "peer_ca": "ca.cert.pem",
            "peer_leaf_sha256": ["c" * 64],
            "receiver_server_name": "muser-receiver",
            "receiver_leaf_sha256": "d" * 64,
            "hmac_key_file": "handoff.key",
            "hmac_key_id": "release-key",
            "hmac_epoch": 1,
            "generation_ledger": "generation.json",
            "work_dir": "work",
            "export_binary": "/opt/muser/bin/spark_kv_export",
            "container_runtime": "docker",
            "container_image": image,
            "container_receipt": "container.json",
            "sender_script": "sender.py",
            "timeout_seconds": 900,
            "max_context": 131072,
            "model_sha256": "e" * 64,
            "model_revision": "official-revision",
            "tokenizer_revision": "official-tokenizer",
            "tokenizer_sha256": "f" * 64,
            "chat_template_sha256": "0" * 64,
            "context_policy_sha256": "1" * 64,
            "adapter_sha256": adapter,
            "target_cache_identity_sha256": "2" * 64,
            "dflash_identity_sha256": "3" * 64,
            "dflash_gguf_sha256": "4" * 64,
            "dflash_kv_heads": 8,
            "dflash_head_dim": 128,
            "dflash_context_geometry": {
                "layers": 5,
                "elements_per_token": 1024,
                "sink_size": 64,
                "window_size": 2048,
            },
            "mmproj_sha256": "5" * 64,
            "preprocessing_sha256": "6" * 64,
        }
        path = root / "handoff.json"
        path.write_text(json.dumps(value))
        return path

    def test_sealed_combined_container_config_loads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            loaded = PREFILLD.load_config(self.config(root))
            self.assertTrue(PREFILLD.config_is_containerized(loaded["schema_version"]))
            self.assertTrue(PREFILLD.config_has_dflash(loaded["schema_version"]))
            self.assertTrue(PREFILLD.config_has_vision(loaded["schema_version"]))
            self.assertNotIn("llama_source_dir", loaded)
            self.assertTrue(loaded["work_dir"].is_dir())
            self.assertEqual(
                loaded["dflash_context_geometry"]["window_size"], 2048
            )

    def test_dflash_context_geometry_is_closed_and_matches_kv_width(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.config(root)
            value = json.loads(path.read_text())
            value["dflash_context_geometry"]["window_size"] = 0
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(PREFILLD.PrefilldError, "context geometry"):
                PREFILLD.load_config(path)

            value["dflash_context_geometry"]["window_size"] = 2048
            value["dflash_context_geometry"]["elements_per_token"] = 512
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(PREFILLD.PrefilldError, "context geometry"):
                PREFILLD.load_config(path)

    def test_sigterm_shutdown_bypasses_request_error_handlers(self) -> None:
        self.assertTrue(issubclass(PREFILLD.PrefilldShutdown, BaseException))
        self.assertFalse(issubclass(PREFILLD.PrefilldShutdown, Exception))
        with self.assertRaises(PREFILLD.PrefilldShutdown):
            PREFILLD.request_shutdown(15, None)

    def test_sender_receipt_requires_payload_only_wire_timing(self) -> None:
        receipt = {
            "ack": True,
            "transfer_start_unix_ns": 10,
            "first_segment_sent_unix_ns": 20,
            "transfer_acked_unix_ns": 40,
            "payload_bytes": 1_000_000,
            "payload_wire_ns": 2_000_000,
        }
        parsed = PREFILLD.parse_sender_receipt(json.dumps(receipt))
        self.assertEqual(parsed["payload_wire_ns"], 2_000_000)
        receipt["payload_wire_ns"] = 0
        with self.assertRaisesRegex(PREFILLD.PrefilldError, "valid transfer receipt"):
            PREFILLD.parse_sender_receipt(json.dumps(receipt))

    def test_write_job_file_is_closed_and_absolute(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = root / "job"
            PREFILLD.write_job_file(
                job,
                n_ctx=8192,
                tokens_path=root / "tokens",
                nope_fifo=root / "fifo",
                stdout_path=root / "stdout",
                status_path=root / "status",
            )
            body = job.read_text(encoding="ascii")
            self.assertIn("n_ctx 8192\n", body)
            self.assertIn(f"tokens {root.resolve() / 'tokens'}\n", body)
            self.assertNotIn("draft_out", body)
            self.assertNotIn("multimodal_plan", body)

    def test_target_fifo_waits_for_delayed_exporter_writer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fifo = Path(temporary) / "target.fifo"
            os.mkfifo(fifo, 0o600)
            observed: list[bytes] = []

            def consume() -> None:
                with fifo.open("rb", buffering=0) as stream:
                    observed.append(stream.read())

            reader = threading.Thread(target=consume)
            reader.start()
            time.sleep(0.05)
            self.assertTrue(reader.is_alive(), "reader must wait for the exporter")
            with fifo.open("wb", buffering=0) as stream:
                stream.write(b"target-planes")
            reader.join(timeout=1)
            self.assertFalse(reader.is_alive())
            self.assertEqual(observed, [b"target-planes"])

            source = MODULE_PATH.read_text(encoding="utf-8")
            self.assertNotIn("nope_fifo_anchor", source)

    def test_warm_docker_command_uses_serve_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = PREFILLD.load_config(self.config(root))
            command = PREFILLD.warm_docker_command(
                config,
                Path("/models/muse.gguf"),
                None,
                None,
                root / "jobs.fifo",
            )
            self.assertEqual(command[0], str(config["container_runtime"]))
            self.assertIn("-d", command)
            self.assertIn(PREFILLD.WARM_CONTAINER, command)
            self.assertNotIn("--rm", command)
            self.assertIn("--serve-jobs", command)
            self.assertIn("--cuda-metal-compatible-full", command)
            strict_index = command.index("MUSER_CROSS_VENDOR_QK=1")
            self.assertEqual(command[strict_index - 1], "-e")
            self.assertLess(strict_index, command.index(config["container_image"]))
            self.assertNotIn("--tokens", command)

    def test_full_compatibility_route_is_strict_and_pool_backed(self) -> None:
        exporter = (ROOT / "scripts/gx10/llamacpp/spark_kv_export.cpp").read_text()
        patch = (ROOT / "scripts/gx10/llamacpp/muser_cuda_metal_compat.patch").read_text()
        self.assertIn(
            'setenv("MUSER_CUDA_METAL_COMPAT_STRICT", "1", 1)', exporter
        )
        self.assertIn("MUSER_CUDA_METAL_COMPAT_STRICT", patch)
        self.assertIn("muser_sincos_fixed", patch)
        self.assertIn("muser_phase_from_angle", patch)
        self.assertIn("cache[i0] = cos_theta*attn_factor", patch)
        self.assertIn("cache[i0 + 1] = sin_theta*attn_factor", patch)
        self.assertIn("muser_cross_vendor_dflash_attention_f32", patch)
        self.assertIn(
            "muser_cross_vendor_dflash_attention_f32<<<blocks, 32", patch
        )
        self.assertIn(
            "muser_cross_vendor_attention_f32<<<blocks, 32", patch
        )
        self.assertIn(
            "dot = muser_add_rn(dot, __shfl_down_sync", patch
        )
        self.assertNotIn(
            "muser_cross_vendor_attention_f32<<<blocks, 1", patch
        )
        self.assertIn(
            'const int round_f16 = getenv("MUSER_CROSS_VENDOR_QK") &&',
            patch,
        )
        self.assertIn(
            "((op == 1 && ggml_nelements(src1) == src0->ne[0])",
            patch,
        )
        self.assertIn("__half2float(__float2half_rn(r))", patch)
        self.assertIn('strncmp(dst->name, "ffn_inp-", 8)', patch)
        self.assertIn('strncmp(dst->name, "l_out-", 6)', patch)
        self.assertIn(
            "muser_rms_norm_cross_vendor_f32<<<nrows, 32", patch
        )
        self.assertIn(
            "yr[i] = __half2float(__float2half_rn(normalized))", patch
        )
        self.assertIn("muser_rms_norm_cross_vendor_weighted_f32", patch)
        self.assertIn('strcmp(dst->name, "attn_norm-0") == 0', patch)
        self.assertIn(
            "const float weighted = muser_fma_rn(normalized, weight[i], 0.0f)",
            patch,
        )
        self.assertIn("src0->op == GGML_OP_RMS_NORM", patch)
        self.assertIn(
            "const float x0 = __half2float(__float2half_rn(x[ix + i0]))",
            patch,
        )
        self.assertIn(
            "dst[idst + i0] = __half2float(__float2half_rn(",
            patch,
        )
        self.assertNotIn("cache[i0] = cosf(angle)", patch)
        dflash_rope = (
            ROOT / "scripts/gx10/llamacpp/muser_dflash_rope_nco.patch"
        ).read_text()
        self.assertIn(
            "const float x0 = __half2float(__float2half_rn(x[ix + pair]))",
            dflash_rope,
        )
        self.assertIn(
            "dst[idst + pair] = __half2float(__float2half_rn(",
            dflash_rope,
        )
        self.assertIn("ggml_cuda_pool_alloc<float> scores_alloc", patch)
        self.assertNotIn("CUDA_CHECK(cudaMalloc(&scores", patch)
        self.assertNotIn('setenv("MUSER_CUDA_CPU_ORDER_ATTN_DEBUG"', exporter)
        self.assertRegex(
            exporter,
            r"llama_context_params dparams = cparams;\s+"
            r"// Muser's DFlash context cache is f32\.[\s\S]*?"
            r"dparams\.type_k = GGML_TYPE_F32;\s+"
            r"dparams\.type_v = GGML_TYPE_F32;",
        )
        dflash_mode = 'setenv("MUSER_DFLASH_ATTENTION_F32", "1", 1)'
        draft_decode = "llama_decode(state->draft_ctx, *state->draft_inject)"
        clear_mode = 'unsetenv("MUSER_DFLASH_ATTENTION_F32")'
        self.assertIn(dflash_mode, exporter)
        self.assertIn(draft_decode, exporter)
        self.assertIn(clear_mode, exporter)
        self.assertLess(exporter.index(dflash_mode), exporter.index(draft_decode))
        self.assertLess(exporter.index(draft_decode), exporter.index(clear_mode))

    def test_live_batch_full_is_off_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with unittest.mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(PREFILLD.LIVE_BATCH_FULL_ENV, None)
                config = PREFILLD.load_config(self.config(root))
                self.assertFalse(PREFILLD.live_batch_full_enabled(config))
                command = PREFILLD.warm_docker_command(
                    config, Path("/models/muse.gguf"), None, None, root / "jobs.fifo"
                )
            self.assertNotIn(f"{PREFILLD.LIVE_BATCH_FULL_ENV}=1", command)

    def test_live_batch_full_config_field_reaches_the_container(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.config(root)
            value = json.loads(path.read_text())
            value[PREFILLD.LIVE_BATCH_FULL_FIELD] = True
            path.write_text(json.dumps(value))
            with unittest.mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(PREFILLD.LIVE_BATCH_FULL_ENV, None)
                config = PREFILLD.load_config(path)
                self.assertTrue(PREFILLD.live_batch_full_enabled(config))
                command = PREFILLD.warm_docker_command(
                    config, Path("/models/muse.gguf"), None, None, root / "jobs.fifo"
                )
            index = command.index(f"{PREFILLD.LIVE_BATCH_FULL_ENV}=1")
            self.assertEqual(command[index - 1], "-e")
            self.assertLess(index, command.index(config["container_image"]))

    def test_live_batch_full_env_overrides_the_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.config(root)
            value = json.loads(path.read_text())
            value[PREFILLD.LIVE_BATCH_FULL_FIELD] = True
            path.write_text(json.dumps(value))
            config = PREFILLD.load_config(path)
            with unittest.mock.patch.dict(
                os.environ, {PREFILLD.LIVE_BATCH_FULL_ENV: "0"}
            ):
                self.assertFalse(PREFILLD.live_batch_full_enabled(config))
            with unittest.mock.patch.dict(
                os.environ, {PREFILLD.LIVE_BATCH_FULL_ENV: "1"}
            ):
                self.assertTrue(PREFILLD.live_batch_full_enabled({}))

    def test_live_batch_full_must_be_boolean(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.config(root)
            value = json.loads(path.read_text())
            value[PREFILLD.LIVE_BATCH_FULL_FIELD] = "1"
            path.write_text(json.dumps(value))
            with self.assertRaises(PREFILLD.PrefilldError):
                PREFILLD.load_config(path)

    def test_container_receipt_identity_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.config(root)
            receipt = json.loads((root / "container.json").read_text())
            receipt["image_id"] = "sha256:" + "7" * 64
            (root / "container.json").write_text(json.dumps(receipt))
            with self.assertRaises(PREFILLD.PrefilldError):
                PREFILLD.load_config(config)


if __name__ == "__main__":
    unittest.main()
