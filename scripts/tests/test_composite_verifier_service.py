from __future__ import annotations

import hashlib
import hmac
import importlib.util
import json
from pathlib import Path
import socket
import struct
import tempfile
import threading
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "composite_verifier_service",
    ROOT / "scripts" / "gx10" / "vllm" / "composite_verifier_service.py",
)
assert SPEC is not None and SPEC.loader is not None
service = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(service)


class CompositeVerifierServiceTests(unittest.TestCase):
    def test_checkpoint_identity_is_derived_from_loaded_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_bytes(b"{}")
            (root / "weights").mkdir()
            (root / "weights" / "model.safetensors").write_bytes(b"weights")
            rows = [
                ("config.json", b"{}"),
                ("weights/model.safetensors", b"weights"),
            ]
            expected = hashlib.sha256()
            for name, payload in rows:
                expected.update(
                    f"{name}\0{len(payload)}\0{hashlib.sha256(payload).hexdigest()}\n".encode()
                )
            self.assertEqual(
                service.checkpoint_artifact_sha256(root), expected.hexdigest()
            )
            (root / "weights" / "model.safetensors").write_bytes(b"changed")
            self.assertNotEqual(
                service.checkpoint_artifact_sha256(root), expected.hexdigest()
            )

    def test_domain_tag_matches_the_kvpack_wire_transcript(self) -> None:
        key = bytes(range(32))
        stream = b'{"closed":true}'
        body = (
            b"kvpack-domain-mac-v1\0"
            + struct.pack(">H", len(service.REQUEST_DOMAIN))
            + service.REQUEST_DOMAIN
            + stream
        )
        self.assertEqual(
            service.domain_tag(key, service.REQUEST_DOMAIN, stream),
            hmac.new(key, body, hashlib.sha256).hexdigest(),
        )

    def test_greedy_carried_frontier_geometry(self) -> None:
        candidates = [10, 11, 12, 13]
        self.assertEqual(service.greedy_decision(candidates, [11, 99, 13, 14]), (1, 99))
        self.assertEqual(service.greedy_decision(candidates, [11, 12, 13, 14]), (3, 14))
        with self.assertRaisesRegex(service.ServiceError, "geometry"):
            service.greedy_decision(candidates, [11])

    def test_block_granular_parent_cache_lag_is_bounded(self) -> None:
        self.assertEqual(service.authenticated_parent_cache_lag(2048, 2063, 16), 15)
        self.assertEqual(service.authenticated_parent_cache_lag(2064, 2064, 16), 0)
        with self.assertRaisesRegex(service.ServiceError, "authenticated block cut"):
            service.authenticated_parent_cache_lag(2064, 2063, 16)
        with self.assertRaisesRegex(service.ServiceError, "partial block"):
            service.authenticated_parent_cache_lag(2048, 2064, 16)
        with self.assertRaisesRegex(service.ServiceError, "authenticated block cut"):
            service.authenticated_parent_cache_lag(2049, 2063, 16)

    def test_prompt_logprob_tail_rechecks_recomputed_frontier(self) -> None:
        class Logprob:
            rank = 1
            logprob = 0.0

        rows = [None] + [{10: Logprob()}] * 14 + [
            {20: Logprob()},
            {21: Logprob()},
            {22: Logprob()},
        ]
        self.assertEqual(
            service.target_tokens_from_prompt_tail(
                rows,
                candidate_count=3,
                parent_lag=15,
                frontier=20,
                generated=23,
            ),
            [21, 22, 23],
        )
        rows[-3] = {99: Logprob()}
        with self.assertRaisesRegex(service.ServiceError, "frontier witness"):
            service.target_tokens_from_prompt_tail(
                rows,
                candidate_count=3,
                parent_lag=15,
                frontier=20,
                generated=23,
            )

    def test_request_frame_requires_canonical_authenticated_json(self) -> None:
        key = bytes(range(32))
        core = {
            "base_head_sha256": "0" * 64,
            "candidates": [],
            "command": "open",
            "request_id": "open-1",
            "schema": service.REQUEST_SCHEMA,
            "sent_unix_ms": 1,
            "session_id": "fixture",
        }
        envelope = service.signed_envelope(core, key, service.REQUEST_DOMAIN)
        raw = service.canonical_json(envelope)
        sender, receiver = socket.socketpair()
        with sender, receiver:
            sender.sendall(struct.pack(">Q", len(raw)) + raw)
            self.assertEqual(service.receive_request(receiver, key), core)

        noncanonical = json.dumps(envelope, indent=2).encode()
        sender, receiver = socket.socketpair()
        with sender, receiver:
            sender.sendall(struct.pack(">Q", len(noncanonical)) + noncanonical)
            with self.assertRaisesRegex(service.ServiceError, "canonical JSON"):
                service.receive_request(receiver, key)

    def test_f16_response_advertises_and_enforces_hidden_geometry(self) -> None:
        key = bytes(range(32))
        payload = bytes(2 * service.ROW_BYTES)
        core = service.frame_core(
            {**service.payload_geometry(), "status": "opened"},
            payload,
            service.FINAL_SCHEMA,
        )
        sender, receiver = socket.socketpair()
        with sender, receiver:
            worker = threading.Thread(
                target=service.send_final, args=(sender, core, payload, key)
            )
            worker.start()
            response, received_payload = service.receive_frame(
                receiver,
                key,
                expected_schema=service.FINAL_SCHEMA,
                domain=service.FINAL_DOMAIN,
            )
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive())
        self.assertEqual(service.FINAL_SCHEMA.rsplit(".", 1)[-1], "v2")
        self.assertEqual(response["payload_dtype"], "f16_le")
        self.assertEqual(response["payload_row_bytes"], 5 * 6656 * 2)
        self.assertEqual(response["payload_bytes"], len(payload))
        self.assertEqual(received_payload, payload)

        with socket.socket() as stream:
            with self.assertRaisesRegex(service.ServiceError, "geometry"):
                service.send_final(
                    stream, core, payload[:-1], key
                )

    def test_capture_abi_validation_rejects_f32_transport(self) -> None:
        capture = {
            "bytes": service.ROW_BYTES,
            "cached_tokens": 1,
            "dtype": "f16_le",
            "hidden_size": service.HIDDEN_SIZE,
            "layout": service.HIDDEN_LAYOUT,
            "payload": bytes(service.ROW_BYTES),
            "row_bytes": service.ROW_BYTES,
            "target_layers": list(service.TARGET_LAYER_IDS),
        }
        self.assertEqual(
            service.validate_hidden_capture(capture, 1), capture["payload"]
        )
        capture["dtype"] = "f32_le"
        with self.assertRaisesRegex(service.ServiceError, "ABI"):
            service.validate_hidden_capture(capture, 1)

    def test_capture_timing_reports_last_layer_overlap_budget(self) -> None:
        layer_timings = [
            {
                "arrival_offset_ns": index * 100,
                "copy_enqueued_offset_ns": index * 100 + 25,
                "copy_is_async": True,
                "layer": layer,
            }
            for index, layer in enumerate(service.TARGET_LAYER_IDS)
        ]
        timing = service.capture_timing(
            {
                "capture_started_ns": 10_000,
                "finish_completed_offset_ns": 900,
                "finish_started_offset_ns": 800,
                "layer_timings": layer_timings,
            },
            10_500,
        )
        self.assertEqual(timing["generate_finished_offset_ns"], 500)
        self.assertEqual(timing["last_layer_arrival_to_generate_finish_ns"], 100)
        self.assertEqual(
            timing["last_layer_copy_enqueue_to_generate_finish_ns"], 75
        )

    def test_verify_wire_orders_authenticated_provisional_before_bound_final(
        self,
    ) -> None:
        key = bytes(range(32))
        candidates = [10, 11]
        provisional_payload = bytes(2 * service.ROW_BYTES)
        provisional = service.provisional_frame(
            session_id="session-a",
            request_id="verify-1",
            base_head_sha256="a" * 64,
            candidates=candidates,
            payload=provisional_payload,
            host_ready_offset_ns=100,
            payload_ready_offset_ns=125,
        )
        final_payload = b""
        final = service.frame_core(
            {
                **service.payload_geometry(),
                "base_head_sha256": "a" * 64,
                "candidate_count": len(candidates),
                "candidate_tokens_sha256": service.token_digest(candidates),
                "committed_count": 1,
                "committed_payload_bytes": service.ROW_BYTES,
                "committed_payload_sha256": service.digest_bytes(
                    provisional_payload[: service.ROW_BYTES]
                ),
                "provisional_sha256": provisional["provisional_sha256"],
                "request_id": "verify-1",
                "session_id": "session-a",
                "status": "verified",
            },
            final_payload,
            service.FINAL_SCHEMA,
        )
        sender, receiver = socket.socketpair()

        def send_pair() -> None:
            service.send_provisional(sender, provisional, provisional_payload, key)
            service.send_final(sender, final, final_payload, key)

        with sender, receiver:
            worker = threading.Thread(target=send_pair)
            worker.start()
            received_provisional, received_hidden = service.receive_frame(
                receiver,
                key,
                expected_schema=service.PROVISIONAL_SCHEMA,
                domain=service.PROVISIONAL_DOMAIN,
            )
            received_final, received_final_payload = service.receive_frame(
                receiver,
                key,
                expected_schema=service.FINAL_SCHEMA,
                domain=service.FINAL_DOMAIN,
            )
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive())
        service.validate_provisional_final_binding(
            received_provisional, received_hidden, received_final
        )
        self.assertEqual(received_hidden, provisional_payload)
        self.assertEqual(received_final_payload, b"")
        self.assertEqual(received_provisional["candidate_count"], 2)
        wrong_prefix = dict(received_final)
        wrong_prefix["committed_payload_sha256"] = "0" * 64
        with self.assertRaisesRegex(service.ServiceError, "bind"):
            service.validate_provisional_final_binding(
                received_provisional, received_hidden, wrong_prefix
            )

    def test_provisional_payload_corruption_fails_closed(self) -> None:
        key = bytes(range(32))
        payload = bytes(service.ROW_BYTES)
        core = service.provisional_frame(
            session_id="session-a",
            request_id="verify-corrupt",
            base_head_sha256="b" * 64,
            candidates=[10],
            payload=payload,
            host_ready_offset_ns=100,
            payload_ready_offset_ns=125,
        )
        header = service.canonical_json(
            service.signed_envelope(core, key, service.PROVISIONAL_DOMAIN)
        )
        corrupted = payload[:-1] + b"\x01"
        sender, receiver = socket.socketpair()

        def send_corrupt() -> None:
            sender.sendall(struct.pack(">QQ", len(header), len(corrupted)) + header)
            sender.sendall(corrupted)

        with sender, receiver:
            worker = threading.Thread(target=send_corrupt)
            worker.start()
            with self.assertRaisesRegex(service.ServiceError, "digest"):
                service.receive_frame(
                    receiver,
                    key,
                    expected_schema=service.PROVISIONAL_SCHEMA,
                    domain=service.PROVISIONAL_DOMAIN,
                )
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive())

    def test_verifier_identity_has_a_cross_language_protocol_vector(self) -> None:
        identity = {
            "bundle_root_sha256": "a" * 64,
            "hidden_abi": {
                "dtype": "f16_le",
                "hidden_size": 6656,
                "layout": "token-major-selected-layer-major-hidden",
                "target_layers": [1, 13, 25, 37, 49],
            },
            "source_checkpoint_artifact_sha256": "b" * 64,
            "source_checkpoint_revision": "source-rev",
            "target_checkpoint_artifact_sha256": "c" * 64,
            "target_checkpoint_revision": "target-rev",
        }
        self.assertEqual(
            service.verifier_identity_sha256(identity),
            "0513acbb5e7f9594a93562a3cc5ad2ac80336f18e25be60858cb331fb8f64dd3",
        )

    def test_exact_replay_reemits_identical_provisional_then_final(self) -> None:
        key = bytes(range(32))
        request = {
            "base_head_sha256": "c" * 64,
            "candidates": [10],
            "command": "verify",
            "request_id": "verify-replay",
            "schema": service.REQUEST_SCHEMA,
            "sent_unix_ms": 1,
            "session_id": "session-a",
        }
        payload = bytes(service.ROW_BYTES)
        provisional = service.provisional_frame(
            session_id="session-a",
            request_id="verify-replay",
            base_head_sha256=request["base_head_sha256"],
            candidates=request["candidates"],
            payload=payload,
            host_ready_offset_ns=100,
            payload_ready_offset_ns=125,
        )
        final = service.frame_core(
            {
                **service.payload_geometry(),
                "base_head_sha256": request["base_head_sha256"],
                "candidate_count": 1,
                "candidate_tokens_sha256": service.token_digest([10]),
                "committed_count": 1,
                "committed_payload_bytes": service.ROW_BYTES,
                "committed_payload_sha256": service.digest_bytes(payload),
                "provisional_sha256": provisional["provisional_sha256"],
                "request_id": request["request_id"],
                "session_id": request["session_id"],
                "status": "verified",
            },
            b"",
            service.FINAL_SCHEMA,
        )
        verifier = object.__new__(service.CompositeVerifier)
        verifier._session_id = request["session_id"]
        verifier._replays = {
            request["request_id"]: (
                service.request_identity(request),
                (provisional, payload),
                (final, b""),
            )
        }

        for _ in range(2):
            sender, receiver = socket.socketpair()

            def replay() -> None:
                replay_final, replay_payload = verifier.handle(
                    request,
                    lambda core, hidden: service.send_provisional(
                        sender, core, hidden, key
                    ),
                )
                service.send_final(sender, replay_final, replay_payload, key)

            with sender, receiver:
                worker = threading.Thread(target=replay)
                worker.start()
                replayed_provisional, replayed_hidden = service.receive_frame(
                    receiver,
                    key,
                    expected_schema=service.PROVISIONAL_SCHEMA,
                    domain=service.PROVISIONAL_DOMAIN,
                )
                replayed_final, replayed_final_payload = service.receive_frame(
                    receiver,
                    key,
                    expected_schema=service.FINAL_SCHEMA,
                    domain=service.FINAL_DOMAIN,
                )
                worker.join(timeout=5)
                self.assertFalse(worker.is_alive())
            self.assertEqual(replayed_provisional, provisional)
            self.assertEqual(replayed_hidden, payload)
            self.assertEqual(replayed_final, final)
            self.assertEqual(replayed_final_payload, b"")


if __name__ == "__main__":
    unittest.main()
