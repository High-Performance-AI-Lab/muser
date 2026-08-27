#!/usr/bin/env python3
"""Offline v2 tests for the llama.cpp lane: hybrid (iSWA) session parsing.

Crafts a synthetic two-block session (full-attention block + sliding-window
block, as llama_kv_cache_iswa::state_write serializes them) and asserts:

  - block discovery and layer->(block, plane) mapping from the layout table
  - per-block plane row validation against per-class geometry
  - SWA trailing-window slicing by token range (cell-pos aware)
  - full-context planes ship whole
  - build_begin_v2 frame/payload accounting matches layout_walk
  - fail-closed: uncovered layer, v_trans=1 slicing, wrong row, short file

Run: python3 scripts/gx10/llamacpp/test_session_v2.py
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
    SESSION_MAGIC,
    SESSION_VERSION,
    ProtocolError,
    SessionPlanes,
    build_begin_v2,
    class_layer_groups,
    class_row_expectations,
    load_layout,
)
from protocol import PORTABLE_ABI_V2, layout_walk  # noqa: E402

# Synthetic geometry: 6 layers; layers 0-4 sliding-window (2 kv heads x hd 4
# -> 16-byte rows, window 3), layer 5 full attention (2 kv heads x hd 8 ->
# 32-byte rows). Fixture of 8 tokens -> 7 cached cells, positions 0..6.
SWA_ROW = 2 * 4 * 2
FULL_ROW = 2 * 8 * 2
N_CELLS = 7

TABLE = [
    {
        "class": "gqa-windowed",
        "from": 0,
        "until": 6,
        "step": 1,
        "except": [5],
        "kv_heads": 2,
        "head_dim": 4,
        "dtype": "float16",
        "window_tokens": 3,
        "roles": ["key", "value"],
    },
    {
        "class": "gqa-full",
        "from": 5,
        "until": 6,
        "step": 1,
        "except": [],
        "kv_heads": 2,
        "head_dim": 8,
        "dtype": "float16",
        "window_tokens": 0,
        "roles": ["key", "value"],
    },
]
LAYOUT_DOC = {"name": "synthetic-hybrid", "weights_precision": "bf16",
              "layout_table": TABLE}


def _plane_bytes(n_layers: int, row: int, v_trans: int) -> bytes:
    """Per-plane data where row j is filled with byte j, so slices are
    identifiable. K planes and V planes share the pattern; v_trans=1 uses
    the transposed header form and [n_embd][cell] data."""
    buf = b""
    for _ in range(n_layers):
        if v_trans == 0:
            buf += struct.pack("<iQ", GGML_TYPE_F16, row)
        else:
            buf += struct.pack("<iII", GGML_TYPE_F16, 2, row // 2)
        for j in range(N_CELLS):
            buf += bytes([j] * row)
    return buf


def _block(n_layers: int, row: int, v_trans: int) -> bytes:
    buf = struct.pack("<I", 1)  # n_stream
    buf += struct.pack("<I", N_CELLS)
    for j in range(N_CELLS):
        buf += struct.pack("<iI", j, 0)  # pos j, no seq ids
    buf += struct.pack("<II", v_trans, n_layers)
    buf += _plane_bytes(n_layers, row, 0)  # K
    buf += _plane_bytes(n_layers, row, v_trans)  # V
    return buf


def hybrid_session(tokens: list[int], swa_v_trans: int = 0,
                   full_row: int = FULL_ROW, drop_swa_block: bool = False) -> str:
    buf = struct.pack("<III", SESSION_MAGIC, SESSION_VERSION, len(tokens))
    buf += struct.pack(f"<{len(tokens)}i", *tokens)
    arch = b"gemma4"
    buf += struct.pack("<I", len(arch)) + arch
    buf += _block(1, full_row, 0)  # full-attention cache first
    if not drop_swa_block:
        buf += _block(5, SWA_ROW, swa_v_trans)
    fd, path = tempfile.mkstemp(suffix=".bin", prefix="kv-session-v2-")
    with os.fdopen(fd, "wb") as fh:
        fh.write(buf)
    return path


def uniform_session(tokens: list[int], nonempty_blocks: int = 1) -> str:
    """DFlash-shaped state: one zero-layer encoder memory module precedes
    the sole uniform five-layer KV module."""
    buf = struct.pack("<III", SESSION_MAGIC, SESSION_VERSION, len(tokens))
    buf += struct.pack(f"<{len(tokens)}i", *tokens)
    arch = b"dflash"
    buf += struct.pack("<I", len(arch)) + arch
    buf += _block(0, SWA_ROW, 0)
    for _ in range(nonempty_blocks):
        buf += _block(5, SWA_ROW, 0)
    fd, path = tempfile.mkstemp(suffix=".bin", prefix="kv-session-uniform-")
    with os.fdopen(fd, "wb") as fh:
        fh.write(buf)
    return path


def parse(path: str, fixture: list[int]) -> SessionPlanes:
    return SessionPlanes(
        path, 0, 0, fixture,
        row_expectations=class_row_expectations(TABLE, 6),
        layer_groups=class_layer_groups(TABLE),
    )


def check(name: str, ok: bool, detail: str) -> None:
    if ok:
        print(f"PASS {name}: {detail}")
        return
    print(f"FAIL {name}: {detail}")
    sys.exit(1)


def expect_protocol_error(name: str, fn) -> None:
    try:
        fn()
    except ProtocolError as exc:
        print(f"PASS {name}: ProtocolError: {exc}")
        return
    print(f"FAIL {name}: no ProtocolError raised")
    sys.exit(1)


def main() -> None:
    fixture = list(range(8))

    path = uniform_session(fixture[:-1])
    try:
        uniform = SessionPlanes(path, 2, 4, fixture)
        check(
            "uniform session skips zero-layer memory module",
            uniform.arch == "dflash"
            and uniform.n_layer == 5
            and uniform.cell_count == N_CELLS
            and len(uniform.blocks) == 1,
            f"layers={uniform.n_layer} cells={uniform.cell_count}",
        )
    finally:
        os.unlink(path)

    path = uniform_session(fixture[:-1], nonempty_blocks=2)
    try:
        expect_protocol_error(
            "multiple uniform KV blocks rejected",
            lambda: SessionPlanes(path, 2, 4, fixture),
        )
    finally:
        os.unlink(path)

    groups = class_layer_groups(TABLE)
    check("layer groups", groups == [[5], [0, 1, 2, 3, 4]], f"groups={groups}")
    rows = class_row_expectations(TABLE, 6)
    check(
        "row expectations",
        rows == [SWA_ROW] * 5 + [FULL_ROW],
        f"rows={rows}",
    )

    path = hybrid_session(fixture[:-1])
    try:
        planes = parse(path, fixture)
        check(
            "hybrid session parses",
            planes.arch == "gemma4"
            and planes.n_layer == 6
            and planes.cell_count == N_CELLS
            and len(planes.blocks) == 2
            and planes.blocks[0].n_layer == 1
            and planes.blocks[1].n_layer == 5,
            f"blocks={[b.n_layer for b in planes.blocks]}",
        )

        # SWA window slice [cached-3, cached): rows 4, 5, 6 of the plane.
        windowed = planes.read_layer_plane(0, "key", 4, 7)
        expected = b"".join(bytes([j] * SWA_ROW) for j in (4, 5, 6))
        check("swa window slice", windowed == expected,
              f"{len(windowed)} bytes match rows 4..6")

        windowed_v = planes.read_layer_plane(2, "value", 4, 7)
        check("swa value window slice", windowed_v == expected,
              f"{len(windowed_v)} bytes match rows 4..6")

        # Full-attention plane ships whole.
        full = planes.read_layer_plane(5, "key", 0, 7)
        expected_full = b"".join(bytes([j] * FULL_ROW) for j in range(7))
        check("full plane whole", full == expected_full,
              f"{len(full)} bytes match rows 0..6")

        expect_protocol_error(
            "uncovered layer rejected",
            lambda: planes.read_layer_plane(9, "key", 0, 7),
        )
        expect_protocol_error(
            "range outside cells rejected",
            lambda: planes.read_layer_plane(0, "key", 6, 8),
        )

        # begin + walk: frames and payload accounting match the table.
        args = types.SimpleNamespace(
            timeout_seconds=900, consumer_engine_abi="c", consumer_node="mac",
            producer_engine_abi="p", producer_node="gx10", trust_domain="lab",
            head_dim=0, max_context=32768, num_kv_heads=0,
            adapter_sha256="a", chat_template_sha256="t",
            context_policy_sha256="x", model_revision="r", model_sha256="m",
            tokenizer_revision="tr", tokenizer_sha256="ts",
        )
        begin = build_begin_v2(args, planes, 7, LAYOUT_DOC, fixture)
        payload = 10 * 3 * SWA_ROW + 2 * 7 * FULL_ROW
        check(
            "begin v2 accounting",
            begin["portable_abi"] == PORTABLE_ABI_V2
            and begin["expected_layer_frames"] == 12
            and begin["expected_payload_bytes"] == payload
            and begin["geometry"]["num_layers"] == 6
            and begin["geometry"]["num_kv_heads"] == 0
            and begin["precision"]["weights"] == "bf16",
            f"frames={begin['expected_layer_frames']} "
            f"payload={begin['expected_payload_bytes']}",
        )
        total = 0
        for _cls, layer, role, start, end in layout_walk(begin):
            total += len(planes.read_layer_plane(layer, role, start, end))
        check("walk payload matches begin", total == payload,
              f"walked {total} of {payload} bytes")

        # Wire schedule: absent by default; decode-priority streams the
        # windowed class (newest cuts) first even when the table declares
        # the full class first; unknown values fail closed.
        check("schedule absent by default", "schedule" not in begin,
              "begin carries no schedule key")
        reversed_doc = {"name": "reversed", "layout_table": [TABLE[1], TABLE[0]]}
        layer_order = [entry[1] for entry in layout_walk(
            build_begin_v2(args, planes, 7, reversed_doc, fixture))]
        check(
            "layer-order keeps declared class order",
            layer_order == [5, 5, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
            f"layers={layer_order}",
        )
        priority_doc = {
            "name": "reversed-priority",
            "schedule": "decode-priority",
            "layout_table": [TABLE[1], TABLE[0]],
        }
        priority_walk = layout_walk(
            build_begin_v2(args, planes, 7, priority_doc, fixture))
        priority_layers = [entry[1] for entry in priority_walk]
        priority_roles = [entry[2] for entry in priority_walk]
        check(
            "decode-priority streams newest cuts first, K then V",
            priority_layers == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
            and priority_roles == ["key", "value"] * 6,
            f"layers={priority_layers}",
        )
        expect_protocol_error(
            "unknown schedule fails closed in walk",
            lambda: layout_walk({**begin, "schedule": "k-first"}),
        )
    finally:
        os.unlink(path)

    # Fail-closed variants.
    path = hybrid_session(fixture[:-1], swa_v_trans=1)
    try:
        planes = parse(path, fixture)
        expect_protocol_error(
            "v_trans=1 window slicing rejected",
            lambda: planes.read_layer_plane(0, "value", 4, 7),
        )
        whole = planes.read_layer_plane(0, "value", 0, 7)
        check("v_trans=1 whole plane untransposed", len(whole) == 7 * SWA_ROW,
              f"{len(whole)} bytes")
    finally:
        os.unlink(path)

    path = hybrid_session(fixture[:-1], full_row=FULL_ROW + 2)
    try:
        expect_protocol_error(
            "wrong block row rejected",
            lambda: parse(path, fixture),
        )
    finally:
        os.unlink(path)

    path = hybrid_session(fixture[:-1], drop_swa_block=True)
    try:
        expect_protocol_error(
            "missing swa block rejected",
            lambda: parse(path, fixture),
        )
    finally:
        os.unlink(path)

    # load_layout validates required fields.
    fd, doc_path = tempfile.mkstemp(suffix=".json", prefix="kv-layout-")
    import json

    with os.fdopen(fd, "w") as fh:
        json.dump({"name": "broken", "layout_table": [{"class": "x"}]}, fh)
    try:
        expect_protocol_error(
            "layout class missing fields rejected",
            lambda: load_layout(doc_path),
        )
    finally:
        os.unlink(doc_path)

    fd, doc_path = tempfile.mkstemp(suffix=".json", prefix="kv-layout-")
    with os.fdopen(fd, "w") as fh:
        json.dump({"name": "bad-schedule", "schedule": "k-first",
                   "layout_table": TABLE}, fh)
    try:
        expect_protocol_error(
            "unknown layout schedule rejected",
            lambda: load_layout(doc_path),
        )
    finally:
        os.unlink(doc_path)

    print("OK: all v2 session cases pass")


if __name__ == "__main__":
    main()
