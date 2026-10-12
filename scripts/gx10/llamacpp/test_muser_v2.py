#!/usr/bin/env python3
"""Offline contract checks for the Muser V2 wrapper around the GX10 parser."""

import contextlib
import hashlib
import hmac
import base64
import io
import json
import os
import socket
import struct
import tempfile
import time
import types
from pathlib import Path

import muser_v2_send as v2

from muser_v2_send import (
    DeferredHandoffV2Sender,
    GGML_TYPE_F32,
    Intent,
    MAGIC,
    ProtocolError,
    build_begin,
    build_seal,
    canonical,
    dflash_intents,
    f16_to_f32_le,
    load_mac_key,
    muse_intents,
    payload_for,
    read_frame,
)
from muser_prefilld import materialize_multimodal, validate_request


def main() -> None:
    assert v2._LinuxTcpInfo.busy_time.offset == 168
    if os.sys.platform.startswith("linux"):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        client = socket.create_connection(listener.getsockname())
        peer, _ = listener.accept()
        try:
            assert isinstance(v2.linux_tcp_busy_time_us(client), int)
            assert v2.configure_linux_pacing(client) == 1_000_000_000
        finally:
            peer.close()
            client.close()
            listener.close()
    intents = muse_intents(2049)
    assert len(intents) == 5 + 5 * 3
    first_swa = next(item for item in intents if item.role == "swa_tile")
    assert first_swa.start == 1 and first_swa.count == 511
    assert first_swa.elements_per_token == 256 * 26
    last_swa = next(item for item in reversed(intents) if item.role == "swa_tile")
    assert last_swa.start + last_swa.count == 2049
    assert all(item.sequence == index for index, item in enumerate(intents))
    # Delta (prefix-cut) schedules: cut 0 is identical to the full schedule;
    # a cut tiles only the suffix, and the SWA span is the newly in-window
    # tail. A suffix longer than the 2048 window re-sends the whole window.
    assert muse_intents(2049) == muse_intents(2049, 0)
    delta = muse_intents(2049, 1024)
    delta_nope = [item for item in delta if item.role == "nope_tile"]
    assert delta_nope[0].start == 1024 and delta_nope[0].count == 512
    assert delta_nope[-1].start + delta_nope[-1].count == 2049
    delta_swa = [item for item in delta if item.role == "swa_tile"]
    assert delta_swa[0].start == 1024 and delta_swa[-1].start + delta_swa[-1].count == 2049
    assert all(item.sequence == index for index, item in enumerate(delta))
    long_delta = muse_intents(4096, 512)
    long_swa = [item for item in long_delta if item.role == "swa_tile"]
    assert long_swa[0].start == 2048  # window slid past the cut: resend the window
    assert long_swa[-1].start + long_swa[-1].count == 4096
    for bad_cut in (-1, 2049, 4096):
        try:
            muse_intents(2049, bad_cut)
            raise AssertionError(f"cut {bad_cut} accepted")
        except ProtocolError:
            pass
    # The sealed begin hash excludes prefix_cut: the receiver seals the
    # canonical typed manifest, which drops the field on its typed parse.
    begin_args = types.SimpleNamespace(
        transfer_id="t",
        generation=1,
        timeout_seconds=900,
        adapter_sha256="a" * 64,
        chat_template_sha256="b" * 64,
        context_policy_sha256="c" * 64,
        model_revision="m",
        model_sha256="d" * 64,
        tokenizer_revision="r",
        tokenizer_sha256="e" * 64,
        multimodal_projector_sha256=None,
        multimodal_preprocessing_sha256=None,
        multimodal_image_sequence_sha256=None,
        hmac_key_id="k",
        hmac_epoch=1,
        target_cache_identity_sha256="f" * 64,
        dflash_session=None,
        dflash_identity_sha256=None,
    )
    plain = build_begin(begin_args, [1, 2, 3], [])
    delta = dict(plain)
    delta["prefix_cut"] = 256
    assert "prefix_cut" not in plain
    key = b"k" * 32
    assert build_seal(plain, [], "0" * 64, 0, key) == build_seal(delta, [], "0" * 64, 0, key)
    draft = dflash_intents(2113, 5, 256, 64, 2048, len(intents))
    assert len(draft) == 20
    assert draft[0].start == 0 and draft[0].count == 64
    assert draft[1].start == 65 and draft[1].count == 2048
    legacy_draft = dflash_intents(2049, 5, 256, 64, 1024, len(intents))
    assert legacy_draft[0].start == 0 and legacy_draft[0].count == 64
    assert legacy_draft[1].start == 1025 and legacy_draft[1].count == 1024
    converted = f16_to_f32_le(struct.pack("<2e", 1.5, -2.0))
    assert struct.unpack("<2f", converted) == (1.5, -2.0)
    native = struct.pack("<2f", 1.5, -2.0)
    native_source = types.SimpleNamespace(
        element_type=GGML_TYPE_F32,
        k_planes=[object()],
        read_plane=lambda *_args: native,
    )
    native_intent = Intent(0, "dflash", "dflash_key", 0, 0, 1, "f32_le", 2)
    assert payload_for(native_source, native_intent, native_source) == native

    descriptors = [
        {
            "sequence": 0,
            "component_id": "target",
            "role": "nope_key",
            "layer": 3,
            "logical_start": 0,
            "logical_count": 1,
            "element_type": "f16_le",
            "elements_per_token": 256,
            "byte_len": 512,
            "sha256": hashlib.sha256(b"x" * 512).hexdigest(),
        }
    ]
    args = types.SimpleNamespace(
        transfer_id="vector",
        generation=9,
        timeout_seconds=900,
        adapter_sha256="a" * 64,
        chat_template_sha256="b" * 64,
        context_policy_sha256="c" * 64,
        model_revision="m",
        model_sha256="d" * 64,
        tokenizer_revision="t",
        tokenizer_sha256="e" * 64,
        hmac_key_id="key",
        hmac_epoch=4,
        target_cache_identity_sha256="f" * 64,
        dflash_session=None,
        dflash_identity_sha256=None,
        multimodal_projector_sha256=None,
        multimodal_preprocessing_sha256=None,
        multimodal_image_sequence_sha256=None,
    )
    begin = build_begin(args, [1], descriptors)
    payload_hash = hashlib.sha256(b"x" * 512).hexdigest()
    key = bytes([7]) * 32
    seal = build_seal(begin, descriptors, payload_hash, 512, key)
    expected = hmac.new(key, canonical(seal["core"]), hashlib.sha256).hexdigest()
    assert seal["hmac_sha256"] == expected
    assert seal["core"]["begin_sha256"] == hashlib.sha256(canonical(begin)).hexdigest()

    image = b"not-an-image-but-an-authenticated-offline-fixture"
    image_sha = hashlib.sha256(image).hexdigest()
    image_sequence_sha = hashlib.sha256(bytes.fromhex(image_sha)).hexdigest()
    multimodal_request = {
        "schema_version": 2,
        "request_id": "vision-vector",
        "deadline_unix_ms": time.time_ns() // 1_000_000 + 60_000,
        "segments": [
            {"kind": "tokens", "token_ids": [1, 2]},
            {
                "kind": "image",
                "data_base64": base64.b64encode(image).decode("ascii"),
                "sha256": image_sha,
                "projected_tokens": 4,
            },
            {"kind": "tokens", "token_ids": [3, 4]},
        ],
        "multimodal": {
            "projector_sha256": "1" * 64,
            "preprocessing_sha256": "2" * 64,
            "image_sequence_sha256": image_sequence_sha,
        },
        "receiver_host": "192.0.2.1",
        "receiver_port": 29590,
    }
    validate_request(
        multimodal_request,
        {
            "schema_version": 3,
            "max_context": 131072,
            "mmproj_sha256": "1" * 64,
            "preprocessing_sha256": "2" * 64,
        },
    )
    validate_request(
        multimodal_request,
        {
            "schema_version": 4,
            "max_context": 131072,
            "mmproj_sha256": "1" * 64,
            "preprocessing_sha256": "2" * 64,
        },
    )
    with tempfile.TemporaryDirectory(prefix="muser-multimodal-") as directory:
        witness, plan, artifacts = materialize_multimodal(
            Path(directory),
            "vector",
            multimodal_request["segments"],
        )
        assert [int(line) for line in witness.read_text().splitlines()] == [
            1,
            2,
            0x7FFFFFFF,
            0x7FFFFFFF,
            0x7FFFFFFF,
            0x7FFFFFFF,
            3,
            4,
        ]
        assert len(plan.read_text().splitlines()) == 3
        assert len(artifacts) == 5

    left, right = socket.socketpair()
    try:
        # Rust's struct serializer does not promise sorted transport-header
        # keys; transport framing accepts it while the HMAC material above
        # remains strictly canonical.
        encoded = b'{"kind":"ack","transfer_id":"vector","generation":9}'
        right.sendall(MAGIC + struct.pack("<IQ", len(encoded), 0) + encoded)
        header, payload = read_frame(left)
        assert header["kind"] == "ack" and payload == b""
    finally:
        left.close()
        right.close()

    descriptor, path = tempfile.mkstemp(prefix="muser-mac-key-")
    try:
        os.write(descriptor, ("07" * 32 + "\n").encode())
        os.close(descriptor)
        os.chmod(path, 0o600)
        assert load_mac_key(path) == key
    finally:
        if os.path.exists(path):
            os.unlink(path)

    class FakeWire:
        def close(self) -> None:
            pass

    frames = []
    original_connect = v2.connect_tls
    original_write = v2.write_frame
    original_payload = v2.write_payload_frame
    original_read = v2.read_frame
    original_session_planes = v2.SessionPlanes
    descriptor, path = tempfile.mkstemp(prefix="muser-deferred-key-")
    try:
        os.write(descriptor, key)
        os.close(descriptor)
        os.chmod(path, 0o600)
        deferred_args = types.SimpleNamespace(
            **vars(args),
            hmac_key_file=path,
            receiver_host="127.0.0.1",
            receiver_port=29590,
            ca_cert="unused",
            client_cert="unused",
            client_key="unused",
            server_name="unused",
            server_leaf_sha256="0" * 64,
        )
        wire = FakeWire()
        v2.connect_tls = lambda _args: wire
        v2.write_frame = lambda _wire, header, payload=b"": frames.append(
            (header, payload)
        )
        v2.write_payload_frame = (
            lambda _wire, header, payload: (
                frames.append((header, payload)) or 1
            )
        )
        v2.read_frame = lambda _wire: (
            {
                "kind": "ack",
                "transfer_id": "vector",
                "generation": 9,
            },
            b"",
        )
        sender = DeferredHandoffV2Sender(deferred_args, [1, 2])
        for intent in muse_intents(1):
            payload = bytes(intent.count * intent.elements_per_token * 2)
            sender.send(intent, payload)
        receipt = sender.seal()
        assert receipt["ack"] and receipt["segments"] == 4
        assert receipt["payload_bytes"] == 4 * 256 * 26 * 2
        assert frames[0][0]["kind"] == "begin"
        assert frames[-1][0]["kind"] == "seal"

        # Wire trace is env-gated: unset (and "0") is off; when set, per-
        # segment JSONL lands on stderr and the receipt schema is untouched.
        assert sender._wire_trace is None
        os.environ.pop("MUSER_GX10_WIRE_TRACE", None)
        os.environ["MUSER_GX10_WIRE_TRACE"] = "0"
        try:
            assert DeferredHandoffV2Sender(deferred_args, [1, 2])._wire_trace is None
            os.environ["MUSER_GX10_WIRE_TRACE"] = "1"
            frames.clear()
            trace_stderr = io.StringIO()
            with contextlib.redirect_stderr(trace_stderr):
                traced = DeferredHandoffV2Sender(deferred_args, [1, 2])
                for intent in muse_intents(1):
                    payload = bytes(intent.count * intent.elements_per_token * 2)
                    traced.send(intent, payload)
                traced_receipt = traced.seal()
        finally:
            del os.environ["MUSER_GX10_WIRE_TRACE"]
        assert traced_receipt.keys() == receipt.keys()
        assert traced_receipt["segments"] == 4
        trace_prefix = "muser-v2-send: wire-trace "
        trace_lines = [
            line
            for line in trace_stderr.getvalue().splitlines()
            if line.startswith(trace_prefix)
        ]
        assert len(trace_lines) == 4
        entries = [json.loads(line[len(trace_prefix) :]) for line in trace_lines]
        assert [entry["seq"] for entry in entries] == [0, 1, 2, 3]
        assert all(
            set(entry) == {"seq", "sent_unix_ns", "write_ns", "snapshot"}
            for entry in entries
        )
        assert all(entry["write_ns"] == 1 for entry in entries)
        assert traced._wire_trace == []  # dumped exactly once on close

        class FakeDFlashPlanes:
            n_layer = 5
            blocks = [types.SimpleNamespace(cell_count=1)]
            element_type = GGML_TYPE_F32

            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def read_plane(self, _planes, _layer, start, end) -> bytes:
                return bytes((end - start) * 8 * 128 * 4)

            def read_value_plane(self, _layer, start, end) -> bytes:
                return bytes((end - start) * 8 * 128 * 4)

            k_planes = [object()] * 5

        frames.clear()
        v2.SessionPlanes = FakeDFlashPlanes
        dflash_values = vars(deferred_args).copy()
        dflash_values.update(
            dflash_session="synthetic.dflash.session",
            dflash_identity_sha256="9" * 64,
            dflash_kv_heads=8,
            dflash_head_dim=128,
            dflash_context_layers=5,
            dflash_context_elements_per_token=1024,
            dflash_context_sink_size=64,
            dflash_context_window_size=2048,
        )
        dflash_args = types.SimpleNamespace(**dflash_values)
        sender = DeferredHandoffV2Sender(dflash_args, [1, 2])
        for intent in muse_intents(1):
            payload = bytes(intent.count * intent.elements_per_token * 2)
            sender.send(intent, payload)
        receipt = sender.seal()
        assert receipt["ack"] and receipt["segments"] == 14
        assert frames[0][0]["manifest"]["components"][1] == {
            "id": "dflash",
            "kind": "dflash_context",
            "required": True,
            "identity_sha256": "9" * 64,
        }
        assert [frame[0]["descriptor"]["component_id"] for frame in frames[1:-1]][-10:] == [
            "dflash"
        ] * 10
    finally:
        v2.connect_tls = original_connect
        v2.write_frame = original_write
        v2.write_payload_frame = original_payload
        v2.read_frame = original_read
        v2.SessionPlanes = original_session_planes
        if os.path.exists(path):
            os.unlink(path)

    print("OK: Muser V2 GX10 adapter contract passes")


if __name__ == "__main__":
    main()
