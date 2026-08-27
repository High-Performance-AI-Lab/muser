from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import struct
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gx10" / "llamacpp"))
sys.path.insert(0, str(ROOT / "scripts" / "gx10" / "vllm"))

import muser_v2_send as llamacpp_v2
from muser_v2_send import muse_intents, packed_planes
from muser_vllm.packing import (
    ROW_BYTES,
    interleaved_to_neox_order,
    neox_to_interleaved_order,
    pack_intent_payload,
    token_ids_sha256,
)
from muser_vllm.receipt import consume_receipt, ensure_slot_available, publish_receipt
from muser_vllm import dflash_capture, exact_rope, native_capture, rope_cache
from muser_vllm.exact_fp4_quant import _disable_fused_input_quantization
from muser_vllm.exact_attention import _pack_exact_attention_inputs

_resident_spec = importlib.util.spec_from_file_location(
    "muser_resident_producer",
    ROOT / "scripts" / "gx10" / "vllm" / "resident_producer.py",
)
assert _resident_spec is not None and _resident_spec.loader is not None
resident = importlib.util.module_from_spec(_resident_spec)
_resident_spec.loader.exec_module(resident)
_request_spec = importlib.util.spec_from_file_location(
    "muser_request_producer",
    ROOT / "scripts" / "gx10" / "vllm" / "request_producer.py",
)
assert _request_spec is not None and _request_spec.loader is not None
request_producer = importlib.util.module_from_spec(_request_spec)
_request_spec.loader.exec_module(request_producer)
_adapter_spec = importlib.util.spec_from_file_location(
    "muser_receipt_adapter",
    ROOT / "scripts" / "gx10" / "vllm" / "receipt_adapter.py",
)
assert _adapter_spec is not None and _adapter_spec.loader is not None
receipt_adapter = importlib.util.module_from_spec(_adapter_spec)
_adapter_spec.loader.exec_module(receipt_adapter)
_cache_identity_spec = importlib.util.spec_from_file_location(
    "muser_receipt_cache_identity",
    ROOT / "scripts" / "gx10" / "vllm" / "receipt_cache_identity.py",
)
assert _cache_identity_spec is not None and _cache_identity_spec.loader is not None
cache_identity = importlib.util.module_from_spec(_cache_identity_spec)
_cache_identity_spec.loader.exec_module(cache_identity)
_score_spec = importlib.util.spec_from_file_location(
    "muser_score_nvfp4_drift",
    ROOT / "scripts" / "gx10" / "vllm" / "score_nvfp4_drift.py",
)
assert _score_spec is not None and _score_spec.loader is not None
score_drift = importlib.util.module_from_spec(_score_spec)
_score_spec.loader.exec_module(score_drift)
_fast_qualifier_spec = importlib.util.spec_from_file_location(
    "muser_qualify_nvfp4_fast",
    ROOT / "scripts" / "qualify_nvfp4_fast.py",
)
assert _fast_qualifier_spec is not None and _fast_qualifier_spec.loader is not None
fast_qualifier = importlib.util.module_from_spec(_fast_qualifier_spec)
_fast_qualifier_spec.loader.exec_module(fast_qualifier)
_warmhit_spec = importlib.util.spec_from_file_location(
    "muser_warmhit_probe",
    ROOT / "scripts" / "gx10" / "vllm" / "warmhit_probe.py",
)
assert _warmhit_spec is not None and _warmhit_spec.loader is not None
warmhit_probe = importlib.util.module_from_spec(_warmhit_spec)
_warmhit_spec.loader.exec_module(warmhit_probe)


class WarmHitGenerationTests(unittest.TestCase):
    def test_explicit_generations_are_sequential_and_never_modulo_wrapped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bearer = Path(directory) / "bearer"
            bearer.write_text("token", encoding="utf-8")
            args = SimpleNamespace(
                base_url="http://127.0.0.1:8080",
                bearer_token_file=bearer,
                first_generation=1_960_206,
                host_work="/host/work",
                node_work="/run/muser/work",
                node="spark",
                container="resident",
                sock="/run/muser/work/producer.sock",
                receiver_host="192.0.2.10",
                receiver_port=29590,
                producer_timeout_seconds=240.0,
                request_prefix="attempt-9-warmhit",
            )
            responses = [
                mock.Mock(returncode=0),
                mock.Mock(returncode=0, stdout="first", stderr=""),
                mock.Mock(returncode=0),
                mock.Mock(returncode=0, stdout="second", stderr=""),
            ]
            with mock.patch.object(
                warmhit_probe.subprocess, "run", side_effect=responses
            ) as run:
                probe = warmhit_probe.Probe(args)
                cold = probe.drive_producer("cold", [1, 2])
                miss = probe.drive_producer("miss", [3, 4])

        first_command = run.call_args_list[1].args[0]
        second_command = run.call_args_list[3].args[0]
        self.assertEqual(
            first_command[first_command.index("--generation") + 1], "1960206"
        )
        self.assertEqual(
            second_command[second_command.index("--generation") + 1], "1960207"
        )
        self.assertIn("attempt-9-warmhit-cold-g1960206", first_command)
        self.assertIn("attempt-9-warmhit-miss-g1960207", second_command)
        self.assertEqual(
            run.call_args_list[0].args[0][-1],
            "cat > /host/work/attempt-9-warmhit-cold-g1960206.tokens",
        )
        self.assertEqual(
            first_command[first_command.index("--tokens") + 1],
            "/run/muser/work/attempt-9-warmhit-cold-g1960206.tokens",
        )
        self.assertEqual(
            first_command[first_command.index("--output") + 1],
            "/run/muser/work/attempt-9-warmhit-cold-g1960206.json",
        )
        self.assertEqual(
            second_command[second_command.index("--output") + 1],
            "/run/muser/work/attempt-9-warmhit-miss-g1960207.json",
        )
        self.assertEqual(cold["generation"], 1_960_206)
        self.assertEqual(miss["generation"], 1_960_207)

    def test_rejects_unsafe_request_prefix_before_any_ssh(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bearer = Path(directory) / "bearer"
            bearer.write_text("token", encoding="utf-8")
            args = SimpleNamespace(
                base_url="http://127.0.0.1:8080",
                bearer_token_file=bearer,
                first_generation=1,
                request_prefix="../../unsafe",
            )

            with self.assertRaisesRegex(ValueError, "request prefix"):
                warmhit_probe.Probe(args)


class WarmHitFailClosedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.args = SimpleNamespace(
            token_fixture=Path("long.tokens"),
            miss_token_fixture=Path("miss.tokens"),
            container="resident",
            first_generation=960_206,
        )

    @staticmethod
    def valid_leg(text: str, ttft: float, *, producer: bool) -> dict[str, object]:
        record: dict[str, object] = {
            "response": {
                "http_status": 200,
                "text": text,
                "ttft_first_token_s": ttft,
                "total_s": ttft + 1,
            }
        }
        if producer:
            record["producer"] = {"returncode": 0}
        return record

    def test_invalid_cold_leg_skips_warm_and_miss_but_retains_evidence(self) -> None:
        probe = mock.Mock()
        probe.run_leg.return_value = {
            "error": "TimeoutError('cold')",
            "producer": {"returncode": 1},
        }

        evidence = warmhit_probe.run_probe(self.args, probe, [1, 2], [3, 4])

        probe.run_leg.assert_called_once_with("cold", [1, 2], True)
        self.assertFalse(evidence["legs_valid"])
        self.assertIn("TimeoutError", evidence["leg_errors"]["cold"])
        self.assertIn("skipped", evidence["warm"])
        self.assertIn("skipped", evidence["miss"])
        self.assertFalse(evidence["miss_control_valid"])

    def test_snapshot_failures_are_recorded_and_invalidate_the_warm_leg(self) -> None:
        probe = object.__new__(warmhit_probe.Probe)
        probe.args = SimpleNamespace(producer_wait_seconds=0.0)
        probe.post = mock.Mock(
            return_value={
                "http_status": 200,
                "text": "served",
                "ttft_first_token_s": 1.0,
            }
        )
        probe.snapshot = mock.Mock(
            side_effect=(RuntimeError("before unavailable"), RuntimeError("after unavailable"))
        )

        record = probe.run_leg("warm", [1, 2], False)

        self.assertEqual(len(record["leg_warnings"]), 2)
        error = warmhit_probe.leg_error(record, producer_required=False)
        self.assertIn("snapshot_before failed", error)
        self.assertIn("snapshot_after failed", error)

    def test_main_exits_nonzero_when_legs_are_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bearer = root / "bearer"
            token_fixture = root / "long.tokens"
            miss_fixture = root / "miss.tokens"
            for path in (bearer, token_fixture, miss_fixture):
                path.write_text("1\n2\n", encoding="utf-8")
            args = SimpleNamespace(
                bearer_token_file=bearer,
                token_fixture=token_fixture,
                miss_token_fixture=miss_fixture,
                out=root / "evidence.json",
            )
            evidence = {
                "legs_valid": False,
                "leg_errors": {"warm": "snapshot_after failed"},
                "outputs_match": False,
                "warm_ttft_below_cold": False,
                "miss_control_valid": False,
                "miss_control_error": "warm leg invalid",
            }
            with (
                mock.patch.object(warmhit_probe, "parse_args", return_value=args),
                mock.patch.object(warmhit_probe, "Probe"),
                mock.patch.object(
                    warmhit_probe, "read_tokens", side_effect=([1, 2], [3, 4])
                ),
                mock.patch.object(warmhit_probe, "run_probe", return_value=evidence),
                mock.patch.object(warmhit_probe, "write_evidence") as write_evidence,
            ):
                status = warmhit_probe.main()

        self.assertEqual(status, 1)
        write_evidence.assert_called_once_with(args.out, evidence)

    def test_text_mismatch_skips_irrelevant_miss_control(self) -> None:
        probe = mock.Mock()
        probe.run_leg.side_effect = (
            self.valid_leg("cold text", 10.0, producer=True),
            self.valid_leg("warm text", 1.0, producer=False),
        )

        evidence = warmhit_probe.run_probe(self.args, probe, [1, 2], [3, 4])

        self.assertEqual(probe.run_leg.call_count, 2)
        self.assertTrue(evidence["legs_valid"])
        self.assertFalse(evidence["outputs_match"])
        self.assertIn("skipped", evidence["miss"])
        self.assertFalse(evidence["miss_control_valid"])

    def test_success_requires_a_valid_miss_control(self) -> None:
        probe = mock.Mock()
        probe.run_leg.side_effect = (
            self.valid_leg("same", 10.0, producer=True),
            self.valid_leg("same", 1.0, producer=False),
            {
                "response": {"http_status": 503, "text": ""},
                "producer": {"returncode": 1},
            },
        )

        evidence = warmhit_probe.run_probe(self.args, probe, [1, 2], [3, 4])

        self.assertEqual(probe.run_leg.call_count, 3)
        self.assertTrue(evidence["legs_valid"])
        self.assertTrue(evidence["outputs_match"])
        self.assertTrue(evidence["warm_ttft_below_cold"])
        self.assertFalse(evidence["miss_control_valid"])
        self.assertIn("HTTP status", evidence["miss_control_error"])


class PackingTests(unittest.TestCase):
    def test_native_and_exact_target_caches_are_distinct(self) -> None:
        exact = cache_identity.derive("1" * 64, "2" * 64, "exact")
        native = cache_identity.derive("1" * 64, "2" * 64, "native")
        self.assertNotEqual(
            exact["target_cache_identity_sha256"],
            native["target_cache_identity_sha256"],
        )

    def test_packed_payload_is_plane_major_and_range_exact(self) -> None:
        cached = 513
        intent = muse_intents(cached)[-1]
        order = packed_planes(intent)
        planes = {}
        for index, key in enumerate(order):
            rows = [bytes([(index + row) % 251]) * ROW_BYTES for row in range(cached)]
            planes[key] = b"".join(rows)
        payload = pack_intent_payload(intent, order, planes, cached)
        expected = b"".join(
            planes[key][
                intent.start * ROW_BYTES : (intent.start + intent.count) * ROW_BYTES
            ]
            for key in order
        )
        self.assertEqual(payload, expected)
        self.assertEqual(len(payload), intent.count * 26 * ROW_BYTES)

    def test_missing_or_short_plane_fails_closed(self) -> None:
        intent = muse_intents(1)[0]
        order = packed_planes(intent)
        planes = {key: bytes(ROW_BYTES) for key in order}
        del planes[order[-1]]
        with self.assertRaisesRegex(ValueError, "missing Muse KV plane"):
            pack_intent_payload(intent, order, planes, 1)
        planes[order[-1]] = b"short"
        with self.assertRaisesRegex(ValueError, "expected"):
            pack_intent_payload(intent, order, planes, 1)

    def test_token_digest_is_u32_little_endian(self) -> None:
        expected = hashlib.sha256(
            b"\x01\x00\x00\x00\x04\x03\x02\x01"
        ).hexdigest()
        self.assertEqual(token_ids_sha256([1, 0x01020304]), expected)
        with self.assertRaises(ValueError):
            token_ids_sha256([-1])


class LlamacppLiveFifoScheduleTests(unittest.TestCase):
    adapter = "89819a82f8088e1067af81efef38f79cd0754c18003cb1fd78c9bee07cda398c"

    @staticmethod
    def fixture(position: int = 1) -> tuple[bytes, dict[tuple[object, ...], bytes]]:
        encoded = bytearray()
        payloads: dict[tuple[object, ...], bytes] = {}
        role_codes = {
            "nope_key": 0,
            "nope_value": 1,
            "swa_key": 2,
            "swa_value": 3,
        }
        for intent_index, intent in enumerate(
            llamacpp_v2.tile_major_fifo_intents(position)
        ):
            payload = bytearray()
            for plane_index, (layer, role) in enumerate(packed_planes(intent)):
                plane = bytes([(intent_index * 29 + plane_index) % 251]) * (
                    intent.count * 512
                )
                encoded.extend(
                    llamacpp_v2.STREAM_HEADER.pack(
                        llamacpp_v2.STREAM_MAGIC,
                        layer,
                        role_codes[role],
                        intent.start,
                        intent.count,
                        256,
                        len(plane),
                    )
                )
                encoded.extend(plane)
                payload.extend(plane)
            payloads[llamacpp_v2.intent_coordinates(intent)] = bytes(payload)
        return bytes(encoded), payloads

    def test_pinned_fifo_and_wire_schedules_are_semantically_identical(self) -> None:
        for position in (1, 2048, 2049):
            source = llamacpp_v2.tile_major_fifo_intents(position)
            wire = muse_intents(position)
            self.assertEqual(
                {llamacpp_v2.intent_coordinates(intent) for intent in source},
                {llamacpp_v2.intent_coordinates(intent) for intent in wire},
            )
            self.assertEqual(len(source), len(wire))
        self.assertEqual(llamacpp_v2.tile_major_fifo_intents(2048)[0].role, "nope_tile")
        self.assertEqual(muse_intents(2048)[0].role, "swa_tile")

    def test_receipted_tile_major_fifo_is_reordered_to_layer_major_wire(self) -> None:
        encoded, payloads = self.fixture()
        source = llamacpp_v2.tile_major_fifo_intents(1)
        self.assertEqual(source[0].role, "nope_tile")

        self.assertEqual(
            llamacpp_v2.TILE_MAJOR_FIFO_ADAPTERS,
            {
                "89819a82f8088e1067af81efef38f79cd0754c18003cb1fd78c9bee07cda398c",
                "e3abb106a70de03dadc50f9427997f37ea88c17e29742b4be04f76709dd478b9",
                "9f7995a8d1dff3f04e46453e4135e46091150c5affc73a1f20f64bb4da3731fe",
                "3f86b5ed0f2c73c0c1b68f6529075fb6e21723e8da170430ac75f1c1106b7560",
                "fbb73b2393833ba5efeb7e2c726f87dc6c8652a8e5430c8b447502bd32d8a7da",
            },
        )
        for adapter in llamacpp_v2.TILE_MAJOR_FIFO_ADAPTERS:
            with self.subTest(adapter=adapter):
                actual = list(
                    llamacpp_v2.live_target_payloads(io.BytesIO(encoded), 1, adapter)
                )

                intents = [intent for intent, _payload in actual]
                self.assertEqual(intents, muse_intents(1))
                self.assertEqual(intents[0].role, "swa_tile")
                self.assertEqual(intents[-1].role, "nope_tile")
                for intent, payload in actual:
                    self.assertEqual(
                        payload,
                        payloads[llamacpp_v2.intent_coordinates(intent)],
                    )

    def test_unknown_adapter_and_frame_drift_fail_closed(self) -> None:
        encoded, _payloads = self.fixture()
        with self.assertRaisesRegex(
            llamacpp_v2.ProtocolError, "no qualified live target FIFO schedule"
        ):
            list(
                llamacpp_v2.live_target_payloads(
                    io.BytesIO(encoded), 1, "f" * 64
                )
            )

        corrupt = bytearray(encoded)
        corrupt[0] ^= 0xFF
        with self.assertRaisesRegex(
            llamacpp_v2.ProtocolError, "receipt-bound FIFO schedule"
        ):
            list(
                llamacpp_v2.live_target_payloads(
                    io.BytesIO(corrupt), 1, self.adapter
                )
            )

    def test_extra_fifo_bytes_are_refused(self) -> None:
        encoded, _payloads = self.fixture()
        with self.assertRaisesRegex(llamacpp_v2.ProtocolError, "unexpected extra"):
            list(
                llamacpp_v2.live_target_payloads(
                    io.BytesIO(encoded + struct.pack("B", 1)), 1, self.adapter
                )
            )


class PackingLayoutTests(unittest.TestCase):
    def test_neox_keys_are_canonicalized_to_interleaved_pairs(self) -> None:
        self.assertEqual(neox_to_interleaved_order(8), (0, 4, 1, 5, 2, 6, 3, 7))
        self.assertEqual(interleaved_to_neox_order(8), (0, 2, 4, 6, 1, 3, 5, 7))
        source = tuple(range(8))
        canonical = tuple(source[index] for index in neox_to_interleaved_order(8))
        restored = tuple(
            canonical[index] for index in interleaved_to_neox_order(8)
        )
        self.assertEqual(restored, source)
        with self.assertRaisesRegex(ValueError, "positive and even"):
            neox_to_interleaved_order(7)

    def test_vllm_rope_cache_is_interleaved_and_f32(self) -> None:
        import numpy as np
        import torch

        source = torch.tensor(
            [[1.0, 2.0, 10.0, 20.0], [3.0, 4.0, 30.0, 40.0]],
            dtype=torch.float16,
        )
        actual = rope_cache.interleave_f16_cache(source)
        self.assertEqual(actual.dtype, np.dtype("<f4"))
        np.testing.assert_array_equal(
            actual,
            np.asarray([[1.0, 10.0, 2.0, 20.0], [3.0, 30.0, 4.0, 40.0]]),
        )

    def test_canonical_q30_nco_matches_cpp_byte_fixture(self) -> None:
        table = exact_rope.canonical_nco_interleaved_table(4)
        value = 0xCBF29CE484222325
        for byte in table.tobytes():
            value ^= byte
            value = (value * 0x00000100000001B3) & 0xFFFF_FFFF_FFFF_FFFF
        self.assertEqual(value, 0xEC36949CC4F7B428)

    def test_resident_validates_loaded_cache_and_exports_canonical_nco(self) -> None:
        source = (
            ROOT / "scripts" / "gx10" / "vllm" / "resident_producer.py"
        ).read_text()
        self.assertIn("engine.apply_model(", source)
        self.assertIn("torch.equal(reference, cache)", source)
        self.assertIn("canonical_nco_interleaved_table", source)
        self.assertIn('"source_device": str(reference.device)', source)


class ReceiptTests(unittest.TestCase):
    def tearDown(self) -> None:
        try:
            consume_receipt()
        except RuntimeError:
            pass

    def test_receipt_slot_is_exactly_once(self) -> None:
        ensure_slot_available()
        publish_receipt({"ok": True})
        with self.assertRaises(RuntimeError):
            ensure_slot_available()
        with self.assertRaises(RuntimeError):
            publish_receipt({"ok": False})
        self.assertEqual(consume_receipt(), {"ok": True})
        with self.assertRaises(RuntimeError):
            consume_receipt()

    def test_adapter_identity_is_canonical_and_source_bound(self) -> None:
        sources = {
            source: format(index + 1, "064x")
            for index, source in enumerate(receipt_adapter.SOURCE_KEYS.values())
        }
        image = {
            "schema": "muser.spark-nvfp4-image-rebuild.v1",
            "image_id": "sha256:" + "a" * 64,
            "vllm_commit": "b" * 40,
            "sources": sources,
        }
        identity = receipt_adapter.derive_identity(image)
        self.assertEqual(len(receipt_adapter.canonical_sha256(identity)), 64)
        image["sources"] = dict(sources)
        del image["sources"][next(iter(receipt_adapter.SOURCE_KEYS.values()))]
        with self.assertRaisesRegex(ValueError, "source digest is missing"):
            receipt_adapter.derive_identity(image)


class ResidentContractTests(unittest.TestCase):
    def test_drift_scorer_accepts_closed_e2_teacher_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "tokens.txt").write_text("1\n2\n3\n", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema": "muser.nvfp4-drift-fixtures.v1",
                        "fixtures": [
                            {
                                "id": "e2-docs-3",
                                "regime": "long-context",
                                "document": "docs",
                                "context_length": 3,
                                "token_file": "tokens.txt",
                                "output_tokens": 0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            loaded = score_drift.load_manifest(manifest, 4, teacher_forced_only=True)
        self.assertEqual(loaded[0]["tokens"], [1, 2, 3])

    def test_drift_scorer_extracts_actual_and_top_prompt_logprobs(self) -> None:
        class Logprob:
            def __init__(self, value: float) -> None:
                self.logprob = value

        output = type(
            "Output",
            (),
            {
                "prompt_logprobs": [
                    None,
                    {7: Logprob(-1.0), 9: Logprob(-0.5)},
                    {8: Logprob(-0.25)},
                ]
            },
        )()
        target, top = score_drift.extract_prompt_rows(output, [1, 7, 8])
        self.assertEqual(target, [-1.0, -0.25])
        self.assertEqual(top, [9, 8])

    def test_drift_scorer_teacher_forced_mode_generates_one_token(self) -> None:
        source = (ROOT / "scripts/gx10/vllm/score_nvfp4_drift.py").read_text()
        self.assertIn("generated_token_limit = 1 if args.teacher_forced_only else 256", source)
        self.assertIn('"evaluation_mode": (', source)
        self.assertIn("muser.spark-nvfp4-drift-score-progress.v1", source)
        self.assertIn('flush=True', source)
        self.assertIn("checkpoint identity overrides must be supplied together", source)

    def test_native_is_default_and_exact_requires_closed_flag(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MUSER_NVFP4_EXACT", None)
            self.assertEqual(resident.producer_mode(), "native")
        with mock.patch.dict(os.environ, {"MUSER_NVFP4_EXACT": "1"}):
            self.assertEqual(resident.producer_mode(), "exact")

    def test_offline_drift_scorer_disables_only_the_kv_connector(self) -> None:
        producer = (ROOT / "scripts/gx10/vllm/resident_producer.py").read_text()
        scorer = (ROOT / "scripts/gx10/vllm/score_nvfp4_drift.py").read_text()
        self.assertIn('getattr(args, "disable_kv_connector", False)', producer)
        self.assertIn("args.disable_kv_connector = True", scorer)
        self.assertNotIn("disable-kv-connector", producer)
        with mock.patch.dict(os.environ, {"MUSER_NVFP4_EXACT": "yes"}):
            with self.assertRaisesRegex(RuntimeError, "exactly 0 or 1"):
                resident.producer_mode()

    def test_native_startup_warmup_is_internal_and_receipted(self) -> None:
        class Prompt:
            def __init__(self, *, prompt_token_ids: list[int]) -> None:
                self.prompt_token_ids = prompt_token_ids

        class Params:
            def __init__(self, **values: object) -> None:
                self.values = values

        class Engine:
            def __init__(self) -> None:
                self.call: tuple[Prompt, Params, bool] | None = None

            def generate(self, prompt: Prompt, params: Params, *, use_tqdm: bool):
                self.call = (prompt, params, use_tqdm)
                sample = type("Sample", (), {"token_ids": [19]})()
                request = type("Request", (), {"outputs": [sample]})()
                return [request]

        engine = Engine()
        with mock.patch.dict(os.environ, {"MUSER_NVFP4_EXACT": "0"}):
            receipt = resident.run_native_startup_warmup(
                engine,
                token_count=4,
                sampling_params_type=Params,
                tokens_prompt_type=Prompt,
            )
        self.assertTrue(receipt["performed"])
        self.assertTrue(receipt["excluded_from_performance_claims"])
        self.assertEqual(receipt["first_token_id"], 19)
        assert engine.call is not None
        prompt, params, use_tqdm = engine.call
        self.assertEqual(prompt.prompt_token_ids, [1, 2, 3, 4])
        self.assertEqual(params.values["extra_args"], {"muser_startup_warmup": True})
        self.assertNotIn("kv_transfer_params", params.values["extra_args"])
        self.assertFalse(use_tqdm)

    def test_exact_attention_packs_split_projection_views(self) -> None:
        import torch

        projection = torch.arange(3 * 4608, dtype=torch.float16).reshape(3, 4608)
        query, key, value = projection.split([4096, 256, 256], dim=-1)
        gate = torch.zeros((3, 4096), dtype=torch.float16)
        self.assertEqual(value.stride(), (4608, 1))
        packed = _pack_exact_attention_inputs(query, key, value, gate)
        self.assertTrue(all(tensor.is_contiguous() for tensor in packed))
        self.assertEqual(packed[2].stride(), (256, 1))
        torch.testing.assert_close(packed[2], value, rtol=0.0, atol=0.0)

    def test_exact_quantizer_disables_prequantized_activation_escape_hatch(self) -> None:
        self.assertIsNone(_disable_fused_input_quantization(object()))
        source = (
            ROOT
            / "scripts"
            / "gx10"
            / "vllm"
            / "muser_vllm"
            / "exact_fp4_quant.py"
        ).read_text()
        self.assertIn(
            "FlashInferCutlassNvFp4LinearKernel.input_quant_key =", source
        )
        self.assertIn('"fused_input_quantization": "disabled"', source)

    def test_exact_fp4_mm_uses_full_order_free_integer_contraction(self) -> None:
        source = (
            ROOT
            / "scripts"
            / "gx10"
            / "vllm"
            / "muser_vllm"
            / "exact_fp4_mm.py"
        ).read_text()
        producer = (
            ROOT / "scripts" / "gx10" / "vllm" / "resident_producer.py"
        ).read_text()
        self.assertIn('SCHEMA = "muser.spark-exact-fp4-mm.v3"', source)
        self.assertIn("E4M3_Q9_SCALE = 512", source)
        self.assertIn("CONTRACTION_SCALE_INV", source)
        self.assertIn("A16_Q8_CONTRACTION_SCALE_INV", source)
        self.assertIn("quantize_nvfp4_q8k", source)
        self.assertIn("integer_fp4_a16_from_q8", source)
        self.assertIn("tl.zeros((32,), tl.int64)", source)
        self.assertIn("tl.sum(partial, axis=0)", source)
        self.assertIn("for chunk in range(0, group_chunks):", source)
        self.assertNotIn("tl.static_range(0, group_chunks)", source)
        self.assertNotIn("torch.nonzero(", source)
        self.assertIn(
            '"selection": "w4a4-and-weight-only-nvfp4-linears"', source
        )
        self.assertIn(
            '"stock_cutlass_and_marlin": "disabled-on-exact-lane"', source
        )
        self.assertIn('"fused_scale_contract": "per-logical-projection-preserved"', source)
        self.assertIn("install_exact_fp4_mm()", producer)
        self.assertIn('"fp4_mm": fp4_mm', producer)
        self.assertIn('"--startup-only"', producer)
        self.assertIn("if args.startup_only:", producer)
        self.assertIn('"--startup-dummy"', producer)
        self.assertIn('load_format="dummy" if args.startup_dummy', producer)
        self.assertIn("DEFAULT_KV_CACHE_BYTES = 1 << 30", producer)
        self.assertIn("kv_cache_memory_bytes=args.kv_cache_memory_bytes", producer)
        self.assertIn('"enable_jit_warmup": False', producer)
        self.assertIn('"enable_cutedsl_warmup": False', producer)

        metal = (
            ROOT / "crates" / "muser-engine" / "src" / "shaders" / "nvfp4.metal"
        ).read_text()
        context = (
            ROOT / "crates" / "muser-engine" / "src" / "metal" / "context.rs"
        ).read_text()
        self.assertIn("muser_e2m1_q1", metal)
        self.assertIn("muser_e4m3_q9", metal)
        self.assertIn("muser_simd_sum_i64", metal)
        self.assertIn("long sums[NC]", metal)
        self.assertIn('include_str!("../shaders/nvfp4.metal")', context)
        self.assertIn("cross_vendor_options.set_fast_math_enabled(false)", context)

    def test_exact_attention_is_request_scoped_and_preserves_cache_writes(self) -> None:
        attention = (
            ROOT
            / "scripts"
            / "gx10"
            / "vllm"
            / "muser_vllm"
            / "exact_attention.py"
        ).read_text()
        producer = (
            ROOT / "scripts" / "gx10" / "vllm" / "resident_producer.py"
        ).read_text()
        self.assertIn('SCHEMA = "muser.spark-exact-attention.v6"', attention)
        self.assertIn("_pack_exact_attention_inputs(", attention)
        self.assertIn("packed contiguous rows", attention)
        self.assertIn("module.attn(q, k, v)", attention)
        self.assertLess(
            attention.index("attention = exact_attention_gate("),
            attention.index("module.attn(q, k, v)"),
        )
        self.assertIn("requires an initial contiguous prefill", attention)
        self.assertIn("shfl.sync.down.b32", attention)
        self.assertIn("shfl.sync.idx.b32", attention)
        self.assertNotIn("tl.sum(partial, axis=0)", attention)
        self.assertIn(
            "tl.range(0, token_count, loop_unroll_factor=1)", attention
        )
        self.assertNotIn("tl.static_range(0, token_count)", attention)
        self.assertIn('"attn_q_after_cache_write"', attention)
        self.assertIn("set_exact_attention_enabled(True)", producer)
        self.assertIn("if exact_mode:", producer)
        self.assertIn("set_exact_stage_capture_enabled(False)", producer)
        self.assertIn("set_exact_attention_enabled(False)", producer)

    def test_exact_sandwich_norm_and_swiglu_are_source_bound(self) -> None:
        rms = (
            ROOT
            / "scripts"
            / "gx10"
            / "vllm"
            / "muser_vllm"
            / "exact_rms_norm.py"
        ).read_text()
        swiglu = (
            ROOT
            / "scripts"
            / "gx10"
            / "vllm"
            / "muser_vllm"
            / "exact_swiglu.py"
        ).read_text()
        rope = (
            ROOT
            / "scripts"
            / "gx10"
            / "vllm"
            / "muser_vllm"
            / "exact_rope.py"
        ).read_text()
        attention = (
            ROOT
            / "scripts"
            / "gx10"
            / "vllm"
            / "muser_vllm"
            / "exact_attention.py"
        ).read_text()
        producer = (
            ROOT / "scripts" / "gx10" / "vllm" / "resident_producer.py"
        ).read_text()
        self.assertIn("exact_split_rms_norm", rms)
        self.assertIn("MuseGlimmerDecoderLayer.forward", rms)
        self.assertLess(
            rms.index('_capture_stage(capture_dir, "layer_out", hidden_states)'),
            rms.index("capture_layer(module._muser_layer_index, hidden_states)"),
        )
        self.assertIn("MuseGlimmerMLP.forward", swiglu)
        self.assertIn('triton_op("muser::exact_rope_neox"', rope)
        self.assertIn("integer-q30-horner-iterative-f32-theta", rope)
        self.assertIn("_mul_rn(x0, cosine)", rope)
        self.assertIn("module.rotary_emb.cos_sin_cache", attention)
        self.assertIn("install_exact_swiglu()", producer)
        self.assertIn('"schema": "muser.spark-nvfp4-runtime.v10"', producer)
        self.assertIn('"selection": "stock-vllm-native-tensor-core"', producer)
        self.assertIn("run_native_startup_warmup(", producer)
        self.assertIn('extra_args={"muser_startup_warmup": True}', producer)
        self.assertIn('if engine_touched and response["status"] != "ok":', producer)
        self.assertIn("os._exit(75)", producer)

    def test_connector_uses_attention_physical_slot_mapping(self) -> None:
        source = (
            ROOT / "scripts" / "gx10" / "vllm" / "muser_vllm" / "connector.py"
        ).read_text()
        self.assertIn("attn_metadata.slot_mapping", source)
        self.assertNotIn("request.block_ids[0], dtype=torch.int64", source)
        self.assertIn("device_pair: torch.Tensor", source)
        self.assertIn("canonical_pair.record_stream(self._copy_stream)", source)
        self.assertIn("key, value = selected.split(EXPECTED_HEAD_DIM, dim=-1)", source)
        self.assertIn('self._producer_mode = "exact" if exact_flag == "1" else "native"', source)
        self.assertIn('if self._producer_mode == "native":', source)
        self.assertIn('extra_args.get("muser_startup_warmup") is True', source)
        self.assertIn('"host_materialize_hash"', source)
        self.assertIn("key = key.index_select(-1, order)", source)
        self.assertIn("return torch.stack((key, value), dim=0)", source)
        self.assertNotIn("key = key.reshape(", source)

    def test_native_benchmark_refuses_exact_patches_and_warms_full_shape(self) -> None:
        source = (
            ROOT / "scripts" / "gx10" / "vllm" / "benchmark_native_prefill.py"
        ).read_text()
        dockerfile = (ROOT / "scripts" / "gx10" / "vllm" / "Dockerfile").read_text()
        receipt = (ROOT / "scripts" / "gx10" / "vllm" / "receipt_image.py").read_text()
        self.assertIn('SCHEMA = "muser.spark-native-nvfp4-prefill-benchmark.v1"', source)
        self.assertIn('os.environ["MUSER_NVFP4_EXACT"] = "0"', source)
        self.assertNotIn("install_exact_", source)
        self.assertIn("enable_chunked_prefill=False", source)
        self.assertIn("enable_prefix_caching=False", source)
        self.assertIn("# One full-shape warmup", source)
        self.assertIn("benchmark_native_prefill.py", dockerfile)
        self.assertIn("request_producer.py", dockerfile)
        self.assertIn("benchmark_native_prefill.py", receipt)
        self.assertIn("request_producer.py", receipt)
        self.assertIn("muser_vllm/dflash_capture.py", receipt)
        adapter_receipt = (
            ROOT / "scripts" / "gx10" / "vllm" / "receipt_adapter.py"
        ).read_text()
        self.assertIn('"dflash_capture_sha256"', adapter_receipt)
        self.assertIn("nvcr.io/nvidia/vllm:26.07-py3@sha256:95c498", dockerfile)
        self.assertIn("--no-deps --force-reinstall", dockerfile)
        self.assertIn('torch.version.cuda == "13.3"', dockerfile)
        self.assertNotIn("--torch-backend=cu129", dockerfile)
        self.assertNotIn("transformers==5.15.0", dockerfile)
        self.assertIn("sha256:95c498a475142c20c989c65e5d223348", receipt)

    def test_fast_qualifier_observes_eee_without_mutating_the_link(self) -> None:
        source = (ROOT / "scripts" / "qualify_nvfp4_fast.py").read_text()
        self.assertIn("enabled - active", source)
        self.assertIn('"EEE status: disabled"', source)
        self.assertIn('"eee_mutated": False', source)
        # The default mode is the enrolled invariant: observe, never mutate.
        self.assertIn('default="require-disabled"', source)
        self.assertIn('if options.eee == "off":', source)
        # Off mode is authorization-gated and must snapshot the link before
        # any mutation, even though the enrolled default is already off.
        self.assertIn('"--eee-off-ruling"', source)
        self.assertIn('control["eee_off_ruling"]', source)
        self.assertIn(
            'control["eee_pre_mutation"] = show_eee(options.spark_host, expect=None)',
            source,
        )
        self.assertNotIn("owner-ruled", source)
        # A failed closing link-state verification must reach the process
        # exit code, not just control.json: no early return before the
        # finally block, and main returns the (possibly bumped) status.
        self.assertIn("status = status or 1", source)
        self.assertIn("\n    return status\n", source)
        self.assertNotIn("return 0", source)
        self.assertIn('"--drift-graded"', source)
        self.assertIn('environment["MUSER_REMOTE_CACHE_DIFF"] = "1"', source)
        self.assertIn('qualifier_command.append("--reference-once")', source)
        self.assertIn('qualifier_command.append("--performance-only")', source)
        self.assertIn('environment["MUSER_REMOTE_QUALIFY_SERIAL"] = "1"', source)

    def test_fast_qualifier_metal_guard_matches_rust_stub(self) -> None:
        rust_source = (ROOT / "crates" / "muser-bench" / "src" / "remote.rs").read_text()
        guard = fast_qualifier.METAL_GUARD.decode("utf-8")
        self.assertEqual(
            guard,
            "remote qualification requires macOS and the metal feature",
        )
        self.assertIn(f'Err("{guard}".into())', rust_source)
        self.assertTrue(fast_qualifier.QUALIFIER_BINARY.is_absolute())

    def test_fast_qualifier_checks_metal_before_link_state(self) -> None:
        order: list[str] = []

        def reject_featureless(_path: Path) -> None:
            order.append("guard")
            raise RuntimeError("featureless")

        options = mock.Mock(
            delta_prefix_cut=0,
            mode="p4",
            first_generation=1,
            performance_only=False,
            pre_streaming_control=False,
            eee="off",
            eee_off_ruling="ledger#ruling",
            cluster_config=ROOT / "missing-cluster.json",
        )
        with (
            mock.patch.object(fast_qualifier, "parse_args", return_value=options),
            mock.patch.dict(os.environ, {"MUSER_ACCELERATOR_LEASE": "1"}),
            mock.patch.object(
                fast_qualifier,
                "require_metal_qualifier",
                side_effect=reject_featureless,
            ),
            mock.patch.object(
                fast_qualifier,
                "show_eee",
                side_effect=lambda *_args, **_kwargs: order.append("eee"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "featureless"):
                fast_qualifier.main()
        self.assertEqual(order, ["guard"])

    def test_fast_qualifier_eee_modes_fail_closed(self) -> None:
        active = """en0: flags=UP
\tmedia: autoselect (10Gbase-T <full-duplex,energy-efficient-ethernet>)
\tstatus: active
\tsupported media:
\t\tmedia 10Gbase-T mediaopt full-duplex mediaopt energy-efficient-ethernet
"""
        disabled = """en0: flags=UP
\tmedia: autoselect (10Gbase-T <full-duplex>)
\tstatus: active
\tsupported media:
\t\tmedia 10Gbase-T mediaopt full-duplex mediaopt energy-efficient-ethernet
"""
        with mock.patch.object(
            fast_qualifier, "run", return_value=mock.Mock(stdout=active)
        ):
            self.assertIn(
                "EEE status: enabled - active",
                fast_qualifier.show_eee("host", expect="active"),
            )
            with self.assertRaisesRegex(RuntimeError, "still armed"):
                fast_qualifier.show_eee("host")
        with mock.patch.object(
            fast_qualifier, "run", return_value=mock.Mock(stdout=disabled)
        ):
            self.assertIn("EEE status: disabled", fast_qualifier.show_eee("host"))
            with self.assertRaisesRegex(RuntimeError, "enabled-active"):
                fast_qualifier.show_eee("host", expect="active")
            # The pre-mutation snapshot is unasserted: any state is recorded.
            self.assertIn(
                "EEE status: disabled",
                fast_qualifier.show_eee("host", expect=None),
            )
        with mock.patch.object(
            fast_qualifier,
            "run",
            return_value=mock.Mock(
                stdout="en0: flags=UP\n\tmedia: autoselect\n\tstatus: active\n"
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "cannot prove"):
                fast_qualifier.show_eee("host")
        with self.assertRaisesRegex(ValueError, "unknown EEE expectation"):
            fast_qualifier.show_eee("host", expect="idle")

    def test_fast_qualifier_eee_off_uses_mac_media_option_and_settles(self) -> None:
        active = """en0: flags=UP
\tmedia: autoselect (10Gbase-T <full-duplex,energy-efficient-ethernet>)
\tstatus: active
\tsupported media:
\t\tmedia 10Gbase-T mediaopt full-duplex mediaopt energy-efficient-ethernet
"""
        with mock.patch.object(
            fast_qualifier,
            "run",
            side_effect=(mock.Mock(stdout=active), mock.Mock(stdout=None)),
        ) as run_call, mock.patch.object(fast_qualifier.time, "sleep") as sleep_call:
            fast_qualifier.disable_eee("host")
        command = run_call.call_args_list[1].args[0]
        self.assertEqual(
            command,
            [
                "sudo",
                "-n",
                "ifconfig",
                fast_qualifier.MAC_EEE_INTERFACE,
                "-mediaopt",
                "energy-efficient-ethernet",
            ],
        )
        sleep_call.assert_called_once_with(fast_qualifier.EEE_OFF_SETTLE_SECONDS)
        self.assertGreaterEqual(fast_qualifier.EEE_OFF_SETTLE_SECONDS, 30)

    def test_fast_qualifier_requires_native_cluster_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cluster.json"
            path.write_text(
                json.dumps(
                    {
                        "producer_mode": "native",
                        "advertised_receiver_host": "192.0.2.10",
                    }
                )
            )
            fast_qualifier.validate_fast_cluster_config(path, "192.0.2.10")
            path.write_text(
                json.dumps(
                    {
                        "producer_mode": "exact",
                        "advertised_receiver_host": "192.0.2.10",
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "producer_mode=native"):
                fast_qualifier.validate_fast_cluster_config(path, "192.0.2.10")
            path.write_text(
                json.dumps(
                    {
                        "producer_mode": "native",
                        "advertised_receiver_host": "192.0.2.99",
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "enrolled receiver"):
                fast_qualifier.validate_fast_cluster_config(path, "192.0.2.10")

    def test_pre_streaming_control_is_restricted_to_text_p4_performance(self) -> None:
        options = SimpleNamespace(
            pre_streaming_control=True,
            mode="p4",
            performance_only=True,
            variant="text",
            delta_prefix_cut=0,
        )
        fast_qualifier.validate_pre_streaming_control(options)

        for override in (
            {"mode": "diagnostic"},
            {"performance_only": False},
            {"variant": "target-plus-dflash"},
            {"delta_prefix_cut": 256},
        ):
            invalid = SimpleNamespace(**vars(options))
            for key, value in override.items():
                setattr(invalid, key, value)
            with self.assertRaisesRegex(ValueError, "requires --mode p4"):
                fast_qualifier.validate_pre_streaming_control(invalid)

        enrolled = SimpleNamespace(**vars(options))
        enrolled.pre_streaming_control = False
        enrolled.mode = "diagnostic"
        fast_qualifier.validate_pre_streaming_control(enrolled)

    def test_performance_mode_admits_p4_and_single_handoff_diagnostic(self) -> None:
        for mode in ("p4", "diagnostic"):
            fast_qualifier.validate_performance_mode(
                SimpleNamespace(performance_only=True, mode=mode)
            )
        fast_qualifier.validate_performance_mode(
            SimpleNamespace(performance_only=False, mode=None)
        )
        with self.assertRaisesRegex(ValueError, "p4 or diagnostic"):
            fast_qualifier.validate_performance_mode(
                SimpleNamespace(performance_only=True, mode=None)
            )

    def test_fast_qualifier_release_repetitions_keep_one_p4_warmup(self) -> None:
        options = SimpleNamespace(
            repetitions=3,
            mode="p4",
            performance_only=True,
            delta_prefix_cut=0,
        )
        self.assertEqual(fast_qualifier.performance_repetitions(options), 3)
        options.repetitions = None
        self.assertEqual(fast_qualifier.performance_repetitions(options), 5)
        options.mode = "diagnostic"
        self.assertEqual(fast_qualifier.performance_repetitions(options), 1)

        for override in (
            {"repetitions": 2, "mode": "p4"},
            {"repetitions": 3, "mode": "diagnostic"},
            {"repetitions": 3, "performance_only": False},
            {"repetitions": 3, "delta_prefix_cut": 256},
        ):
            invalid = SimpleNamespace(**vars(options))
            invalid.mode = "p4"
            invalid.performance_only = True
            invalid.delta_prefix_cut = 0
            for key, value in override.items():
                setattr(invalid, key, value)
            with self.assertRaisesRegex(ValueError, "P4 performance-only"):
                fast_qualifier.performance_repetitions(invalid)

    def test_performance_diagnostic_uses_receipt_directory(self) -> None:
        out_dir = Path("/evidence/qualify")
        diagnostic = SimpleNamespace(
            performance_only=True,
            mode="diagnostic",
            variant="text",
            first_generation=7,
            out_dir=out_dir,
        )
        self.assertEqual(
            fast_qualifier.producer_receipt_arguments(diagnostic),
            ["--external-producer-receipt-dir", str(out_dir)],
        )
        diagnostic.performance_only = False
        self.assertEqual(
            fast_qualifier.producer_receipt_arguments(diagnostic),
            [
                "--external-producer-receipt",
                "/evidence/qualify/f-diagnostic-text-g7-client.json",
            ],
        )
        diagnostic.mode = "p4"
        self.assertEqual(
            fast_qualifier.producer_receipt_arguments(diagnostic),
            ["--external-producer-receipt-dir", str(out_dir)],
        )

    def test_request_fixture_parser_is_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.tokens"
            path.write_text("1\n\n202047\n")
            self.assertEqual(request_producer.read_tokens(path), [1, 202047])
            path.write_text("1\n202048\n")
            with self.assertRaisesRegex(ValueError, "out-of-vocabulary"):
                request_producer.read_tokens(path)

    def test_dflash_capture_writes_token_major_selected_layers(self) -> None:
        import numpy as np
        import torch

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "features.f32"
            dflash_capture.begin_capture("capture-vector", 2, output)
            for layer in dflash_capture.TARGET_LAYERS:
                rows = torch.full(
                    (3, dflash_capture.HIDDEN_SIZE),
                    float(layer),
                    dtype=torch.float16,
                )
                dflash_capture.capture_layer(layer, rows)
            receipt = dflash_capture.finish_capture()
            self.assertIsNotNone(receipt)
            values = np.fromfile(output, dtype="<f4").reshape(
                2, len(dflash_capture.TARGET_LAYERS), dflash_capture.HIDDEN_SIZE
            )
            for index, layer in enumerate(dflash_capture.TARGET_LAYERS):
                self.assertTrue(np.all(values[:, index, :] == float(layer)))
            self.assertEqual(receipt["bytes"], values.nbytes)
            self.assertEqual(receipt["dtype"], "f32_le")
            self.assertEqual(
                receipt["row_bytes"],
                len(dflash_capture.TARGET_LAYERS) * dflash_capture.HIDDEN_SIZE * 4,
            )
            self.assertEqual(
                receipt["layout"], "token-major-selected-layer-major-hidden"
            )
            self.assertEqual(
                [timing["layer"] for timing in receipt["layer_timings"]],
                list(dflash_capture.TARGET_LAYERS),
            )
            for timing in receipt["layer_timings"]:
                self.assertGreaterEqual(timing["arrival_offset_ns"], 0)
                self.assertGreaterEqual(
                    timing["copy_enqueued_offset_ns"], timing["arrival_offset_ns"]
                )
            self.assertGreaterEqual(
                receipt["finish_completed_offset_ns"],
                receipt["finish_started_offset_ns"],
            )

    def test_dflash_capture_supports_explicit_little_endian_f16(self) -> None:
        import numpy as np
        import torch

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "features.f16"
            callbacks = []

            def host_ready(receipt: dict[str, object]) -> None:
                lock_was_free = dflash_capture._LOCK.acquire(blocking=False)
                if lock_was_free:
                    dflash_capture._LOCK.release()
                callbacks.append((receipt, lock_was_free))

            dflash_capture.begin_capture(
                "capture-f16-vector",
                2,
                output,
                device="cpu",
                dtype="f16_le",
                host_ready_callback=host_ready,
            )
            active = dflash_capture._ACTIVE
            self.assertEqual(active["matrix"].dtype, torch.float16)
            for layer in dflash_capture.TARGET_LAYERS:
                rows = torch.full(
                    (2, dflash_capture.HIDDEN_SIZE),
                    float(layer) + 0.5,
                    dtype=torch.float16,
                )
                dflash_capture.capture_layer(layer, rows)
            receipt = dflash_capture.finish_capture(include_payload=True)
            self.assertEqual(len(callbacks), 1)
            callback_receipt, lock_was_free = callbacks[0]
            self.assertTrue(lock_was_free)
            self.assertIs(receipt["payload"], callback_receipt["payload"])
            values = np.fromfile(output, dtype="<f2").reshape(
                2, len(dflash_capture.TARGET_LAYERS), dflash_capture.HIDDEN_SIZE
            )
            for index, layer in enumerate(dflash_capture.TARGET_LAYERS):
                self.assertTrue(np.all(values[:, index, :] == float(layer) + 0.5))
            self.assertEqual(receipt["dtype"], "f16_le")
            self.assertEqual(receipt["bytes"], values.nbytes)
            self.assertEqual(receipt["payload"], output.read_bytes())
            self.assertEqual(
                receipt["sha256"], hashlib.sha256(receipt["payload"]).hexdigest()
            )
            self.assertEqual(
                receipt["row_bytes"],
                len(dflash_capture.TARGET_LAYERS) * dflash_capture.HIDDEN_SIZE * 2,
            )
            self.assertGreaterEqual(
                receipt["payload_ready_offset_ns"], receipt["host_ready_offset_ns"]
            )
            self.assertGreaterEqual(
                receipt["host_ready_callback_completed_offset_ns"],
                receipt["host_ready_callback_started_offset_ns"],
            )

    def test_dflash_capture_can_select_candidate_suffix_rows(self) -> None:
        import numpy as np
        import torch

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "suffix.f32"
            dflash_capture.begin_capture(
                "capture-suffix-vector",
                2,
                output,
                row_selection="suffix",
            )
            for layer in dflash_capture.TARGET_LAYERS:
                rows = torch.arange(4, dtype=torch.float32).reshape(4, 1).expand(
                    4, dflash_capture.HIDDEN_SIZE
                )
                dflash_capture.capture_layer(layer, rows)
            receipt = dflash_capture.finish_capture()
            values = np.fromfile(output, dtype="<f4").reshape(
                2, len(dflash_capture.TARGET_LAYERS), dflash_capture.HIDDEN_SIZE
            )
            self.assertTrue(np.all(values[0] == 2.0))
            self.assertTrue(np.all(values[1] == 3.0))
            self.assertEqual(receipt["cached_tokens"], 2)

    def test_dflash_capture_rejects_unknown_dtype(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "unsupported.*dtype"):
                dflash_capture.begin_capture(
                    "capture-bad-dtype",
                    1,
                    Path(directory) / "features.bin",
                    dtype="bf16_le",
                )

    def test_dflash_capture_rejects_unknown_row_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "row selection"):
                dflash_capture.begin_capture(
                    "capture-bad-row-selection",
                    1,
                    Path(directory) / "features.bin",
                    row_selection="middle",
                )

    def test_dflash_capture_stages_all_layers_before_one_host_materialization(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "features.f32"
            dflash_capture.begin_capture("staged-vector", 2, output, device="cpu")
            active = dflash_capture._ACTIVE
            self.assertEqual(
                tuple(active["matrix"].shape),
                (len(dflash_capture.TARGET_LAYERS), 2, dflash_capture.HIDDEN_SIZE),
            )
            for layer in dflash_capture.TARGET_LAYERS:
                dflash_capture.capture_layer(
                    layer,
                    torch.full(
                        (2, dflash_capture.HIDDEN_SIZE),
                        float(layer),
                        dtype=torch.float16,
                    ),
                )
            self.assertEqual(active["matrix"].device.type, "cpu")
            self.assertIsNone(active["copy_stream"])
            receipt = dflash_capture.finish_capture()
            self.assertEqual(receipt["bytes"], 2 * 5 * dflash_capture.HIDDEN_SIZE * 4)

    def test_dflash_capture_can_stop_at_the_serving_memory_boundary(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must-not-exist.f32"
            dflash_capture.begin_capture("memory-vector", 1, output, device="cpu")
            for layer in dflash_capture.TARGET_LAYERS:
                dflash_capture.capture_layer(
                    layer,
                    torch.full(
                        (1, dflash_capture.HIDDEN_SIZE),
                        float(layer),
                        dtype=torch.float16,
                    ),
                )
            receipt = dflash_capture.finish_capture(materialize=False)
            self.assertFalse(output.exists())
            self.assertFalse(receipt["materialized"])
            self.assertIsNone(receipt["output"])
            self.assertEqual(receipt["bytes"], 5 * dflash_capture.HIDDEN_SIZE * 4)

    def test_dflash_capture_can_return_the_serving_payload(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must-not-exist.f32"
            dflash_capture.begin_capture("payload-vector", 1, output, device="cpu")
            for layer in dflash_capture.TARGET_LAYERS:
                dflash_capture.capture_layer(
                    layer,
                    torch.full(
                        (1, dflash_capture.HIDDEN_SIZE),
                        float(layer),
                        dtype=torch.float16,
                    ),
                )
            receipt = dflash_capture.finish_capture(
                materialize=False, include_payload=True
            )
            self.assertFalse(output.exists())
            self.assertEqual(len(receipt["payload"]), receipt["bytes"])
            self.assertEqual(
                hashlib.sha256(receipt["payload"]).hexdigest(), receipt["sha256"]
            )

    def test_native_capture_returns_the_unmodified_layer_result(self) -> None:
        import torch

        output = torch.ones((2, dflash_capture.HIDDEN_SIZE), dtype=torch.float16)
        residual = torch.zeros_like(output)
        expected = (output, residual)
        original = native_capture._ORIGINAL_DECODER_FORWARD
        try:
            native_capture._ORIGINAL_DECODER_FORWARD = (
                lambda _module, _positions, _hidden, _residual: expected
            )
            module = type(
                "Layer", (), {"_muser_layer_index": dflash_capture.TARGET_LAYERS[0]}
            )()
            with mock.patch.object(native_capture, "capture_layer") as capture:
                actual = native_capture.native_decoder_layer_forward(
                    module, None, output, residual
                )
                capture.assert_called_once_with(module._muser_layer_index, output)
            self.assertIs(actual, expected)
            self.assertIs(actual[0], output)
        finally:
            native_capture._ORIGINAL_DECODER_FORWARD = original

    def test_native_dflash_capture_builds_before_the_deferred_seal(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fifo = root / "jobs.fifo"
            os.mkfifo(fifo, 0o600)
            session = root / "native.session"
            features = root / "native.f32"
            ready = threading.Event()

            def exporter() -> None:
                descriptor = os.open(fifo, os.O_RDWR)
                ready.set()
                with os.fdopen(descriptor, "r", encoding="ascii") as jobs:
                    job = Path(jobs.readline().strip())
                fields = dict(
                    line.split(" ", 1)
                    for line in job.read_text(encoding="ascii").splitlines()
                )
                Path(fields["draft_out"]).write_bytes(b"draft-session")
                Path(fields["stdout"]).write_text("fixture exporter\n")
                Path(fields["status"]).write_text("ok\n")

            worker = threading.Thread(target=exporter)
            worker.start()
            self.assertTrue(ready.wait(timeout=5))
            dflash_capture.begin_capture(
                "native-build",
                2,
                features,
                token_ids=[1, 2, 3],
                session_output=session,
            )
            for layer in dflash_capture.TARGET_LAYERS:
                dflash_capture.capture_layer(
                    layer,
                    torch.full(
                        (3, dflash_capture.HIDDEN_SIZE),
                        float(layer),
                        dtype=torch.float16,
                    ),
                )
            with mock.patch.dict(
                os.environ, {"MUSER_DFLASH_JOBS_FIFO": str(fifo)}, clear=False
            ):
                receipt = dflash_capture.finish_capture_for_connector()
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive())
            self.assertEqual(receipt["session"]["sha256"], hashlib.sha256(b"draft-session").hexdigest())
            self.assertEqual(
                dflash_capture.consume_completed_capture()["request_id"],
                "native-build",
            )

    def test_dflash_exporter_accepts_exact_external_target_features(self) -> None:
        source = (
            ROOT
            / "scripts"
            / "gx10"
            / "llamacpp"
            / "spark_kv_export.cpp"
        ).read_text()
        self.assertIn("--dflash-features", source)
        self.assertIn("read_dflash_features(", source)
        self.assertIn("external DFlash feature byte count differs", source)
        self.assertIn("external DFlash feature positions are not contiguous", source)
        self.assertLess(
            source.index("if (!job.dflash_features_path.empty()) {", source.index("static int run_job")),
            source.index("else if (!job.load_path.empty())"),
        )

    def test_accelerator_lease_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ferrite.gpu.lock"
            first = resident.acquire_accelerator_lease(path)
            try:
                with self.assertRaisesRegex(RuntimeError, "lease unavailable"):
                    resident.acquire_accelerator_lease(path)
            finally:
                first.close()
            second = resident.acquire_accelerator_lease(path)
            second.close()

    def test_closed_config_and_request(self) -> None:
        config = {
            "schema": resident.SCHEMA,
            "vllm_commit": resident.PINNED_VLLM_COMMIT,
            "checkpoint_revision": "revision",
            "checkpoint_artifact_sha256": "a" * 64,
            "connector": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config))
            self.assertEqual(resident.load_config(path), config)
        request = {
            "schema": resident.REQUEST_SCHEMA,
            "request_id": "p2-seam-1",
            "token_ids": [1, 2],
            "handoff": {
                "generation": 1,
                "receiver_host": "192.0.2.10",
                "receiver_port": 29590,
                "transfer_id": "p2-seam-1-1",
            },
        }
        self.assertEqual(resident.validate_request(request, 2048), request)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request["handoff"].update(
                {
                    "dflash_session": str(root / "deferred.session"),
                    "dflash_identity_sha256": "b" * 64,
                    "dflash_kv_heads": 8,
                    "dflash_head_dim": 128,
                    "dflash_context_layers": 5,
                    "dflash_context_elements_per_token": 1024,
                    "dflash_context_sink_size": 64,
                    "dflash_context_window_size": 2048,
                }
            )
            with mock.patch.dict(
                os.environ,
                {
                    "MUSER_NVFP4_EXACT": "0",
                    "MUSER_DFLASH_JOBS_FIFO": str(root / "jobs.fifo"),
                    "MUSER_DFLASH_SESSION_DIR": str(root),
                },
                clear=False,
            ):
                self.assertEqual(resident.validate_request(request, 2048), request)
            with self.assertRaisesRegex(ValueError, "neither precomputed nor"):
                resident.validate_request(request, 2048)
        request["extra"] = True
        with self.assertRaisesRegex(ValueError, "request keys"):
            resident.validate_request(request, 2048)


if __name__ == "__main__":
    unittest.main()
