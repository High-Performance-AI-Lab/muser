#!/usr/bin/env python3
"""Failure-injection tests for the llama.cpp lane (F-rows in docs/FAILURE_MODES.md).

Offline, no GX10 or Mac receiver needed: crafts minimal session binaries and
asserts the sender's fail-closed behavior.

  F4a  session truncated in header     -> ProtocolError (explicit, no struct.error)
  F4b  session truncated in plane data -> ProtocolError at read_plane (short read)
  F18  begin manifest precision triple -> locked to f16/f16/q4_k_m (fail-closed label)
  F20  non-fp16 KV plane header        -> ProtocolError (dtype fail-closed)
  F20b session token prefix mismatch   -> ProtocolError (wrong-cache fail-closed)

Run: python3 scripts/gx10/llamacpp/test_failure_injection.py
"""

import os
import struct
import sys
import tempfile
import types

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))  # protocol.py sits in scripts/gx10/
from llamacpp_session_send import (  # noqa: E402
    GGML_TYPE_F16,
    GGML_TYPE_F32,
    SESSION_MAGIC,
    SESSION_VERSION,
    ProtocolError,
    SessionPlanes,
    build_begin,
)
from protocol import PORTABLE_ABI, PROTOCOL  # noqa: E402


def fake_session(k_type: int, tokens: list[int]) -> str:
    """Minimal 'GGSN' v9 session: 1 stream, 1 cell, 1 layer, one K plane."""
    buf = b""
    buf += struct.pack("<III", SESSION_MAGIC, SESSION_VERSION, len(tokens))
    buf += struct.pack(f"<{len(tokens)}i", *tokens)
    arch = b"qwen2"
    buf += struct.pack("<I", len(arch)) + arch
    buf += struct.pack("<I", 1)  # n_stream
    buf += struct.pack("<I", 1)  # cell_count
    buf += struct.pack("<iI", 0, 0)  # cell meta: pos 0, n_seq_id 0
    buf += struct.pack("<II", 0, 1)  # v_trans=0, n_layer=1
    row = 4 * 128 * 2  # num_kv_heads * head_dim * sizeof(f16)
    buf += struct.pack("<iQ", k_type, row)
    buf += b"\x00" * row  # one cell of K data
    # parser stops after the expected planes are indexed; trailing V block
    # irrelevant for the dtype-injection cases
    fd, path = tempfile.mkstemp(suffix=".bin", prefix="kv-session-inject-")
    with os.fdopen(fd, "wb") as fh:
        fh.write(buf)
    return path


def expect_protocol_error(name: str, fn) -> None:
    try:
        fn()
    except ProtocolError as exc:
        print(f"PASS {name}: ProtocolError: {exc}")
        return
    print(f"FAIL {name}: no ProtocolError raised")
    sys.exit(1)


def check(name: str, ok: bool, detail: str) -> None:
    if ok:
        print(f"PASS {name}: {detail}")
        return
    print(f"FAIL {name}: {detail}")
    sys.exit(1)


def full_session(
    tokens: list[int], element_type: int = GGML_TYPE_F16
) -> str:
    """Complete 'GGSN' v9 session: 1 stream, 1 cell, 1 layer, K+V planes."""
    buf = b""
    buf += struct.pack("<III", SESSION_MAGIC, SESSION_VERSION, len(tokens))
    buf += struct.pack(f"<{len(tokens)}i", *tokens)
    arch = b"qwen2"
    buf += struct.pack("<I", len(arch)) + arch
    buf += struct.pack("<I", 1)  # n_stream
    buf += struct.pack("<I", 1)  # cell_count
    buf += struct.pack("<iI", 0, 0)  # cell meta: pos 0, n_seq_id 0
    buf += struct.pack("<II", 0, 1)  # v_trans=0, n_layer=1
    element_size = 2 if element_type == GGML_TYPE_F16 else 4
    row = 4 * 128 * element_size
    for _ in range(2):  # K block, then V block (v_trans == 0)
        buf += struct.pack("<iQ", element_type, row)
        buf += b"\x00" * row
    fd, path = tempfile.mkstemp(suffix=".bin", prefix="kv-session-inject-")
    with os.fdopen(fd, "wb") as fh:
        fh.write(buf)
    return path


def case_f4a_truncated_header() -> None:
    # ntok claims 4 tokens but the file ends after 2 -> struct.error path
    buf = struct.pack("<III", SESSION_MAGIC, SESSION_VERSION, 4)
    buf += struct.pack("<2i", 0, 1)
    fd, path = tempfile.mkstemp(suffix=".bin", prefix="kv-session-inject-")
    with os.fdopen(fd, "wb") as fh:
        fh.write(buf)
    try:
        expect_protocol_error(
            "F4a truncated session header rejected",
            lambda: SessionPlanes(path, num_kv_heads=4, head_dim=128, fixture=[0, 1]),
        )
    finally:
        os.unlink(path)


def case_f4b_truncated_plane() -> None:
    fixture = [0, 1, 2, 3]
    path = full_session(tokens=fixture[:-1])
    try:
        with open(path, "r+b") as fh:  # cut 10 bytes off the trailing V data
            fh.truncate(os.path.getsize(path) - 10)
        planes = SessionPlanes(path, num_kv_heads=4, head_dim=128, fixture=fixture)
        expect_protocol_error(
            "F4b truncated plane data rejected",
            lambda: planes.read_plane(planes.v_planes, 0),
        )
    finally:
        os.unlink(path)


def case_f18_precision_locked() -> None:
    args = types.SimpleNamespace(
        timeout_seconds=900,
        consumer_engine_abi="c",
        consumer_node="mac",
        producer_engine_abi="p",
        producer_node="gx10",
        trust_domain="lab",
        head_dim=128,
        max_context=32768,
        num_kv_heads=4,
        adapter_sha256="a",
        chat_template_sha256="t",
        context_policy_sha256="x",
        model_revision="r",
        model_sha256="m",
        tokenizer_revision="tr",
        tokenizer_sha256="ts",
    )
    planes = types.SimpleNamespace(n_layer=1)
    begin = build_begin(args, planes, cached=3, plane_bytes=1024, fixture=[0, 1, 2, 3])
    check(
        "F18 begin precision triple locked",
        begin["precision"] == {"compute": "float16", "kv": "float16", "weights": "q4_k_m"}
        and begin["portable_abi"] == PORTABLE_ABI
        and begin["protocol"] == PROTOCOL,
        f"precision={begin['precision']} abi={begin['portable_abi']}",
    )


def case_dflash_f32_is_explicit_and_exact() -> None:
    fixture = [0, 1, 2, 3]
    path = full_session(fixture[:-1], GGML_TYPE_F32)
    try:
        expect_protocol_error(
            "target parser refuses f32 KV",
            lambda: SessionPlanes(path, 4, 128, fixture),
        )
        planes = SessionPlanes(
            path, 4, 128, fixture, element_type=GGML_TYPE_F32
        )
        check(
            "DFlash parser retains native f32 KV",
            planes.element_size == 4
            and len(planes.read_plane(planes.k_planes, 0)) == 4 * 128 * 4,
            f"element_size={planes.element_size}",
        )
    finally:
        os.unlink(path)


def main() -> None:
    fixture = [0, 1, 2, 3]

    case_f4a_truncated_header()
    case_f4b_truncated_plane()
    case_f18_precision_locked()
    case_dflash_f32_is_explicit_and_exact()

    path = fake_session(k_type=0, tokens=fixture[:-1])  # 0 = GGML_TYPE_F32
    try:
        expect_protocol_error(
            "F20 non-fp16 K plane rejected",
            lambda: SessionPlanes(path, num_kv_heads=4, head_dim=128, fixture=fixture),
        )
    finally:
        os.unlink(path)

    path = fake_session(k_type=GGML_TYPE_F16, tokens=[9, 9, 9])
    try:
        expect_protocol_error(
            "F20b session token prefix mismatch rejected",
            lambda: SessionPlanes(path, num_kv_heads=4, head_dim=128, fixture=fixture),
        )
    finally:
        os.unlink(path)

    print("OK: all failure-injection cases fail closed")


if __name__ == "__main__":
    main()
