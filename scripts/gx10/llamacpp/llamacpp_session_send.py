#!/usr/bin/env python3
"""llama.cpp producer leg: parse a llama_state_save_file session and stream
the fp16 KV planes to the ferrite receiver over kvpack-live-f16-le-v1.

Runs on the GX10 host (Python 3 stdlib only; protocol.py sits next to this
file in the GX10 work directory). Replaces the vLLM connector's extraction
path; the wire protocol, identity pinning, plane order (per layer: key then
value), and seal semantics are byte-identical to scripts/gx10/connector.py.

Session layout (llama.cpp master, LLAMA_SESSION_MAGIC 'GGSN' version 9,
whole-context save from spark_kv_export):

  u32 magic, u32 version, u32 ntok, i32 tokens[ntok]
  u32 arch_len, arch bytes
  u32 n_stream
  per stream: u32 cell_count; if 0 continue
    meta: per cell: i32 pos, u32 n_seq_id, i32 seq_id[n_seq_id]
    u32 v_trans, u32 n_layer
    K: per layer: i32 k_type, u64 k_row_bytes, cell_count * k_row_bytes data
    V (v_trans == 0): same as K
    V (v_trans == 1): per layer: i32 v_type, u32 v_el, u32 n_embd,
                      n_embd * cell_count * v_el data (transposed; needs
                      numpy to untranspose -- export with flash attention
                      enabled to get v_trans == 0 instead)

With flash attention enabled the K and V data blocks are already in the
canonical plane layout [token][kv_head][head_dim] fp16 little-endian, so
each plane is a single contiguous read -- no conversion pass at all.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from protocol import (  # noqa: E402
    PORTABLE_ABI,
    PORTABLE_ABI_V2,
    PROTOCOL,
    SCHEMA_VERSION,
    WIRE_SCHEDULE_LAYER_ORDER,
    KNOWN_WIRE_SCHEDULES,
    LiveHandoffSender,
    ProtocolError,
    TlsClientConfig,
    layout_walk,
    token_ids_sha256,
)

SESSION_MAGIC = 0x6767736E  # LLAMA_FILE_MAGIC_GGSN, read as u32 LE
SESSION_VERSION = 9
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1


def _u32(buf: bytes, off: int) -> tuple[int, int]:
    return struct.unpack_from("<I", buf, off)[0], off + 4


def _i32(buf: bytes, off: int) -> tuple[int, int]:
    return struct.unpack_from("<i", buf, off)[0], off + 4


def _u64(buf: bytes, off: int) -> tuple[int, int]:
    return struct.unpack_from("<Q", buf, off)[0], off + 8


class BlockPlanes:
    """One llama_kv_cache state block inside the session file.

    Hybrid (iSWA) models serialize two blocks: the full-attention cache
    first, then the sliding-window cache (llama_kv_cache_iswa::state_write).
    """

    def __init__(self) -> None:
        self.v_trans = 0
        self.n_layer = 0
        self.cell_count = 0
        self.cell_pos: list[int] = []
        self.k_planes: list[tuple[int, int, int]] = []  # (offset, length, row_bytes)
        self.v_planes: list[tuple[int, int, int]] = []


class SessionPlanes:
    """Offset index into the KV plane blocks of a llama.cpp session file."""

    def __init__(self, path: str, num_kv_heads: int, head_dim: int, fixture: list[int],
                 row_expectations: list[int] | None = None,
                 layer_groups: list[list[int]] | None = None,
                 element_type: int = GGML_TYPE_F16):
        self.path = path
        self.size = os.path.getsize(path)
        if element_type not in (GGML_TYPE_F16, GGML_TYPE_F32):
            raise ProtocolError("session parser element type is unsupported")
        self.element_type = element_type
        self.element_size = 2 if element_type == GGML_TYPE_F16 else 4
        # v2: per-layer expected row bytes (mixed head_dim models); None =
        # the uniform v1 check (num_kv_heads * head_dim * 2 for every layer).
        # layer_groups (v2 only): expected block layout as ordered groups of
        # global layer indices — group 0 is the full-attention block, group 1
        # (when present) the sliding-window block.
        self._row_expectations = row_expectations
        self._layer_groups = layer_groups
        self.blocks: list[BlockPlanes] = []
        self._layer_map: dict[int, tuple[int, int]] = {}
        try:
            self._parse(num_kv_heads, head_dim, fixture)
        except struct.error as error:
            raise ProtocolError(
                f"truncated or malformed session file: {error}"
            ) from error

    def _parse_block(self, fh, block: BlockPlanes, row_for_plane) -> None:
        (n_stream,) = struct.unpack("<I", fh.read(4))
        for _ in range(n_stream):
            (cell_count,) = struct.unpack("<I", fh.read(4))
            if cell_count == 0:
                continue
            if block.cell_count != 0:
                raise ProtocolError("more than one non-empty stream in session block")
            block.cell_count = cell_count
            # cell meta: pos i32, n_seq_id u32, seq ids
            for _ in range(cell_count):
                meta = fh.read(8)
                pos, n_seq_id = struct.unpack("<iI", meta)
                block.cell_pos.append(pos)
                if n_seq_id:
                    fh.read(4 * n_seq_id)
            if block.cell_pos != sorted(block.cell_pos) or any(
                block.cell_pos[i + 1] != block.cell_pos[i] + 1
                for i in range(len(block.cell_pos) - 1)
            ):
                raise ProtocolError("session cell positions are not contiguous")
            block.v_trans, block.n_layer = struct.unpack("<II", fh.read(8))
            for plane_index in range(block.n_layer):
                expected_row = row_for_plane(block, plane_index)
                k_type, k_row = struct.unpack("<iQ", fh.read(12))
                if k_type != self.element_type or k_row != expected_row:
                    raise ProtocolError(
                        f"unexpected K plane header layer={plane_index} "
                        f"type={k_type} row={k_row} (expected {expected_row})"
                    )
                off = fh.tell()
                length = cell_count * k_row
                fh.seek(length, os.SEEK_CUR)
                block.k_planes.append((off, length, k_row))
            if block.v_trans == 0:
                for plane_index in range(block.n_layer):
                    expected_row = row_for_plane(block, plane_index)
                    v_type, v_row = struct.unpack("<iQ", fh.read(12))
                    if v_type != self.element_type or v_row != expected_row:
                        raise ProtocolError(
                            f"unexpected V plane header layer={plane_index} "
                            f"type={v_type} row={v_row} (expected {expected_row})"
                        )
                    off = fh.tell()
                    length = cell_count * v_row
                    fh.seek(length, os.SEEK_CUR)
                    block.v_planes.append((off, length, v_row))
            else:
                for plane_index in range(block.n_layer):
                    expected_row = row_for_plane(block, plane_index)
                    v_type, v_el, n_embd = struct.unpack("<iII", fh.read(12))
                    if (v_type != self.element_type or v_el != self.element_size
                            or n_embd * self.element_size != expected_row):
                        raise ProtocolError(
                            f"unexpected transposed V header type={v_type} el={v_el} n={n_embd}"
                        )
                    off = fh.tell()
                    length = cell_count * n_embd * v_el
                    fh.seek(length, os.SEEK_CUR)
                    block.v_planes.append((off, length, n_embd * v_el))
        if block.cell_count == 0:
            raise ProtocolError("session block holds no KV cells")

    def _parse(self, num_kv_heads: int, head_dim: int, fixture: list[int]) -> None:
        with open(self.path, "rb") as fh:
            head = fh.read(12)
            magic, version, ntok = struct.unpack("<III", head)
            if magic != SESSION_MAGIC:
                raise ProtocolError(f"session magic {magic:#x} != GGSN")
            if version != SESSION_VERSION:
                raise ProtocolError(f"session version {version} != {SESSION_VERSION}")
            stored = fh.read(4 * ntok)
            stored_tokens = list(struct.unpack(f"<{ntok}i", stored))
            expected = fixture[: len(fixture) - 1]
            if stored_tokens != expected:
                raise ProtocolError(
                    "session token prefix does not match the fixture "
                    f"({ntok} vs {len(expected)})"
                )
            arch_len = struct.unpack("<I", fh.read(4))[0]
            self.arch = fh.read(arch_len).decode("utf-8", "replace")
            if self._row_expectations is None:
                # Uniform-layout models normally serialize one memory block.
                # DFlash serializes its encoder-position state first (zero KV
                # layers), followed by the five-layer decoder KV block. Parse
                # every complete memory module and retain the sole block that
                # actually owns K/V planes.
                kv_blocks: list[BlockPlanes] = []
                while fh.tell() < self.size:
                    block = BlockPlanes()
                    self._parse_block(
                        fh, block,
                        lambda _b, _i: num_kv_heads * head_dim * self.element_size
                    )
                    if block.n_layer:
                        kv_blocks.append(block)
                if len(kv_blocks) != 1:
                    raise ProtocolError(
                        "uniform session must contain exactly one non-empty KV block"
                    )
                self.blocks = kv_blocks
                self.n_layer = kv_blocks[0].n_layer
            else:
                # v2: one block per layer group (full-attention first, then
                # sliding-window); plane order inside a block is ascending
                # global layer index (llama_kv_cache layer filter order).
                groups = self._layer_groups
                if groups is None:
                    raise ProtocolError("v2 session parse requires layer groups")
                if len(self._row_expectations) != sum(len(g) for g in groups):
                    raise ProtocolError(
                        f"layout expects {len(self._row_expectations)} layers, "
                        f"groups cover {sum(len(g) for g in groups)}"
                    )
                rows: dict[int, int] = {}
                for group in groups:
                    for layer in group:
                        rows[layer] = self._row_expectations[layer]

                def make_row_fn(group: list[int]):
                    def row_fn(_block: BlockPlanes, plane_index: int) -> int:
                        if plane_index >= len(group):
                            raise ProtocolError(
                                f"session block has more than the {len(group)} "
                                "layers the layout group expects"
                            )
                        return rows[group[plane_index]]

                    return row_fn

                for group in groups:
                    block = BlockPlanes()
                    self._parse_block(fh, block, make_row_fn(group))
                    if block.n_layer != len(group):
                        raise ProtocolError(
                            f"session block has {block.n_layer} layers, "
                            f"layout group expects {len(group)}"
                        )
                    self.blocks.append(block)
                if fh.read(1):
                    raise ProtocolError("trailing data after the session KV blocks")
                self.n_layer = sum(block.n_layer for block in self.blocks)
                for block_index, group in enumerate(groups):
                    for plane_index, layer in enumerate(group):
                        self._layer_map[layer] = (block_index, plane_index)
        self.cell_count = self.blocks[0].cell_count
        self.v_trans = self.blocks[0].v_trans
        # Legacy v1 views (single-block callers).
        self.k_planes = self.blocks[0].k_planes
        self.v_planes = self.blocks[0].v_planes

    def read_plane(self, planes: list[tuple[int, int, int]], index: int,
                   start_token: int = 0, end_token: int | None = None,
                   block: BlockPlanes | None = None) -> bytes:
        off, _length, row_bytes = planes[index]
        block = block if block is not None else self.blocks[0]
        if end_token is None:
            end_token = block.cell_count
        if block.cell_pos:
            base = block.cell_pos[0]
            row_start = start_token - base
            row_end = end_token - base
            if row_start < 0 or row_end > block.cell_count or row_end < row_start:
                raise ProtocolError(
                    f"token range [{start_token},{end_token}) outside session "
                    f"cells [{base},{base + block.cell_count})"
                )
        else:
            row_start, row_end = start_token, end_token
        start_off = off + row_start * row_bytes
        length = (row_end - row_start) * row_bytes
        with open(self.path, "rb") as fh:
            fh.seek(start_off)
            data = fh.read(length)
        if len(data) != length:
            raise ProtocolError("short read on KV plane")
        return data

    def read_value_plane(self, index: int, start_token: int = 0,
                         end_token: int | None = None) -> bytes:
        return self._read_value_plane_from(self.blocks[0], index, start_token, end_token)

    def _read_value_plane_from(self, block: BlockPlanes, index: int,
                               start_token: int, end_token: int | None) -> bytes:
        if block.v_trans != 0 and (
            start_token != 0 or end_token not in (None, block.cell_count)
        ):
            raise ProtocolError(
                "windowed slicing needs v_trans=0; re-run spark_kv_export with "
                "--flash-attn 1"
            )
        data = self.read_plane(block.v_planes, index, start_token, end_token, block)
        if block.v_trans == 0:
            return data
        # v_trans == 1: data is [n_embd][cell] fp16; canonical is [cell][n_embd].
        try:
            import numpy as np
        except ImportError as error:
            raise ProtocolError(
                "session has v_trans=1 (flash attention disabled) and numpy is "
                "unavailable for the transpose; re-run spark_kv_export with "
                "--flash-attn 1"
            ) from error
        n_embd = len(data) // (2 * block.cell_count)
        arr = np.frombuffer(data, dtype="<f2").reshape(n_embd, block.cell_count)
        return np.ascontiguousarray(arr.T).tobytes()

    def read_layer_plane(self, layer: int, role: str, start_token: int,
                         end_token: int) -> bytes:
        """v2: read the plane for a global layer index, honoring the block
        layout and the token range (SWA trailing window)."""
        if layer not in self._layer_map:
            raise ProtocolError(f"layer {layer} is not covered by the layout table")
        block_index, plane_index = self._layer_map[layer]
        block = self.blocks[block_index]
        if role == "key":
            return self.read_plane(block.k_planes, plane_index, start_token,
                                   end_token, block)
        return self._read_value_plane_from(block, plane_index, start_token, end_token)


def read_fixture(path: str) -> list[int]:
    tokens = [int(line) for line in open(path) if line.strip()]
    if len(tokens) < 2:
        raise ProtocolError("prompt fixture must hold at least two token IDs")
    return tokens


def build_begin(
    args, planes: SessionPlanes, cached: int, plane_bytes: int, fixture: list[int]
) -> dict:
    now_ms = time.time_ns() // 1_000_000
    return {
        "cached_token_count": cached,
        "created_unix_ms": now_ms,
        "deadline_unix_ms": now_ms + args.timeout_seconds * 1000,
        "endpoints": {
            "consumer_engine_abi": args.consumer_engine_abi,
            "consumer_node": args.consumer_node,
            "producer_engine_abi": args.producer_engine_abi,
            "producer_node": args.producer_node,
            "trust_domain": args.trust_domain,
        },
        "expected_layer_frames": planes.n_layer * 2,
        "expected_payload_bytes": plane_bytes * planes.n_layer * 2,
        "geometry": {
            "head_dim": args.head_dim,
            "max_context_tokens": args.max_context,
            "num_kv_heads": args.num_kv_heads,
            "num_layers": planes.n_layer,
        },
        "identity": {
            "adapter_sha256": args.adapter_sha256,
            "chat_template_sha256": args.chat_template_sha256,
            "context_policy_sha256": args.context_policy_sha256,
            "model_revision": args.model_revision,
            "model_sha256": args.model_sha256,
            "tokenizer_revision": args.tokenizer_revision,
            "tokenizer_sha256": args.tokenizer_sha256,
        },
        "portable_abi": PORTABLE_ABI,
        "precision": {
            "compute": "float16",
            "kv": "float16",
            "weights": "q4_k_m",
        },
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "strategy": "consumer_last_prompt_token",
        "token_ids_sha256": token_ids_sha256(fixture),
        "transfer_id": secrets.token_hex(32),
    }


def load_layout(path: str) -> dict:
    """v2 layout document: {name, layout_table, schedule?}. Classes mirror
    the handoff LayoutClassV2 (from/until/step/except, kv_heads, head_dim,
    dtype, window_tokens, roles). The optional `schedule` arms a wire
    schedule variant ("layer-order" default, "decode-priority"); unknown
    values fail closed here."""
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    table = doc["layout_table"]
    for cls in table:
        for field in ("class", "from", "until", "step", "except", "kv_heads",
                      "head_dim", "dtype", "window_tokens", "roles"):
            if field not in cls:
                raise ProtocolError(f"layout class missing field {field}")
    schedule = doc.get("schedule", WIRE_SCHEDULE_LAYER_ORDER)
    if schedule not in KNOWN_WIRE_SCHEDULES:
        raise ProtocolError(f"layout document declares an unknown wire schedule: {schedule!r}")
    return doc


def class_row_expectations(table: list[dict], num_layers: int) -> list[int]:
    """Per-layer expected row bytes from the layout table; every layer
    must belong to exactly one class."""
    expectations: list[int | None] = [None] * num_layers
    for cls in table:
        for layer in range(cls["from"], cls["until"], max(1, cls["step"])):
            if layer in cls["except"]:
                continue
            if expectations[layer] is not None:
                raise ProtocolError(f"layer {layer} covered by two classes")
            expectations[layer] = cls["kv_heads"] * cls["head_dim"] * 2
    if any(row is None for row in expectations):
        missing = [str(i) for i, row in enumerate(expectations) if row is None]
        raise ProtocolError(f"layout table leaves layers uncovered: {','.join(missing)}")
    return [row for row in expectations if row is not None]


def class_layer_groups(table: list[dict]) -> list[list[int]]:
    """Expected session block layout: group 0 holds the layers of every
    full-context class (window_tokens == 0), group 1 — appended only when
    present — the layers of the sliding-window classes. This mirrors
    llama_kv_cache_iswa::state_write (full-attention cache first, then the
    SWA cache); layers ascend inside each group."""
    base: list[int] = []
    swa: list[int] = []
    for cls in table:
        target = swa if cls["window_tokens"] else base
        for layer in range(cls["from"], cls["until"], max(1, cls["step"])):
            if layer not in cls["except"]:
                target.append(layer)
    groups = []
    if base:
        groups.append(sorted(base))
    if swa:
        groups.append(sorted(swa))
    return groups


def build_begin_v2(
    args, planes: SessionPlanes, cached: int, layout_doc: dict, fixture: list[int]
) -> dict:
    begin = build_begin(args, planes, cached, 0, fixture)
    table = layout_doc["layout_table"]
    frames = 0
    payload = 0
    for cls in table:
        window = cached if cls["window_tokens"] == 0 else min(cls["window_tokens"], cached)
        planes_in_class = len([
            layer
            for layer in range(cls["from"], cls["until"], max(1, cls["step"]))
            if layer not in cls["except"]
        ]) * len(cls["roles"])
        frames += planes_in_class
        payload += planes_in_class * window * cls["kv_heads"] * cls["head_dim"] * 2
    begin["expected_layer_frames"] = frames
    begin["expected_payload_bytes"] = payload
    begin["layout_table"] = table
    begin["portable_abi"] = PORTABLE_ABI_V2
    if "schedule" in layout_doc:
        begin["schedule"] = layout_doc["schedule"]
    if len(table) == 1:
        begin["geometry"]["num_kv_heads"] = table[0]["kv_heads"]
        begin["geometry"]["head_dim"] = table[0]["head_dim"]
    else:
        begin["geometry"]["num_kv_heads"] = 0
        begin["geometry"]["head_dim"] = 0
    begin["geometry"]["num_layers"] = max(
        layer
        for cls in table
        for layer in range(cls["from"], cls["until"], max(1, cls["step"]))
        if layer not in cls["except"]
    ) + 1
    begin["precision"]["weights"] = layout_doc.get("weights_precision", "q4_k_m")
    return begin


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", required=True)
    p.add_argument("--prompt-token-fixture", required=True)
    p.add_argument("--num-layers", type=int, default=28)
    p.add_argument("--num-kv-heads", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--max-context", type=int, default=32768)
    p.add_argument("--receiver-host", required=True)
    p.add_argument("--receiver-port", type=int, default=29590)
    p.add_argument("--server-name", default="kvpack-mac")
    p.add_argument("--ca-cert")
    p.add_argument("--client-cert")
    p.add_argument("--client-key")
    p.add_argument("--model-sha256")
    p.add_argument("--model-revision")
    p.add_argument("--tokenizer-revision", default="7ae557604adf67be50417f59c2c2f167def9a775")
    p.add_argument("--tokenizer-sha256")
    p.add_argument("--chat-template-sha256")
    p.add_argument("--context-policy-sha256")
    p.add_argument("--adapter-sha256")
    p.add_argument("--producer-engine-abi")
    p.add_argument("--consumer-engine-abi")
    p.add_argument("--producer-node", required=True)
    p.add_argument("--consumer-node", default="mac-m3-ultra")
    p.add_argument("--trust-domain", default="lab-prefill-e2e")
    p.add_argument("--timeout-seconds", type=int, default=900)
    p.add_argument("--max-frame-mib", type=int, default=64)
    p.add_argument("--timing-out", default=None)
    p.add_argument("--layout-json", default=None,
                   help="v2 layout document (name + layout_table); sends the "
                        "v2 wire with per-class geometry and SWA window slicing")
    p.add_argument("--dump-only", action="store_true",
                   help="parse and verify the session, print the plane index, do not connect")
    args = p.parse_args()

    layout_doc = load_layout(args.layout_json) if args.layout_json else None

    parse_started = time.perf_counter()
    fixture = read_fixture(args.prompt_token_fixture)
    cached = len(fixture) - 1
    row_expectations = (
        class_row_expectations(layout_doc["layout_table"], args.num_layers)
        if layout_doc is not None
        else None
    )
    layer_groups = (
        class_layer_groups(layout_doc["layout_table"])
        if layout_doc is not None
        else None
    )
    planes = SessionPlanes(
        args.session, args.num_kv_heads, args.head_dim, fixture,
        row_expectations=row_expectations,
        layer_groups=layer_groups,
    )
    if planes.cell_count != cached:
        raise ProtocolError(
            f"session caches {planes.cell_count} cells, fixture expects {cached}"
        )
    if planes.n_layer != args.num_layers:
        raise ProtocolError(f"session has {planes.n_layer} layers, expected {args.num_layers}")
    parse_ns = int((time.perf_counter() - parse_started) * 1e9)

    plane_bytes = cached * args.num_kv_heads * args.head_dim * 2
    summary = {
        "arch": planes.arch,
        "cached_token_count": cached,
        "cell_count": planes.cell_count,
        "layers": planes.n_layer,
        "parse_seconds": round(parse_ns / 1e9, 3),
        "plane_bytes": plane_bytes,
        "session_bytes": planes.size,
        "v_trans": planes.v_trans,
    }
    print(f"[llamacpp-sender] session {json.dumps(summary, sort_keys=True)}", flush=True)
    if args.dump_only:
        return
    missing = [a for a in ("ca_cert", "client_cert", "client_key", "model_sha256",
                           "model_revision", "tokenizer_sha256", "chat_template_sha256",
                           "context_policy_sha256", "adapter_sha256",
                           "producer_engine_abi", "consumer_engine_abi")
               if getattr(args, a) is None]
    if missing:
        p.error("missing required arguments for sending: " +
                ", ".join("--" + a.replace("_", "-") for a in missing))

    begin = (
        build_begin_v2(args, planes, cached, layout_doc, fixture)
        if layout_doc is not None
        else build_begin(args, planes, cached, plane_bytes, fixture)
    )
    tls = TlsClientConfig(
        ca_cert=args.ca_cert,
        client_cert=args.client_cert,
        client_key=args.client_key,
        host=args.receiver_host,
        port=args.receiver_port,
        server_name=args.server_name,
        timeout_seconds=args.timeout_seconds,
    )
    sender = LiveHandoffSender(tls, begin, max_payload_bytes=args.max_frame_mib * 1024 * 1024)
    layer_timings = []
    read_ns_total = 0
    committed = False
    try:
        if layout_doc is not None:
            walk = layout_walk(begin)
        else:
            walk = [
                (None, layer, role, 0, cached)
                for layer in range(planes.n_layer)
                for role in ("key", "value")
            ]
        for _cls, layer, role, start, end in walk:
            read_started = time.perf_counter_ns()
            if layout_doc is not None:
                payload = planes.read_layer_plane(layer, role, start, end)
            elif role == "key":
                payload = planes.read_plane(planes.k_planes, layer, start, end)
            else:
                payload = planes.read_value_plane(layer, start, end)
            read_ns = time.perf_counter_ns() - read_started
            read_ns_total += read_ns
            send_started = time.perf_counter_ns()
            result = sender.send_plane(layer, role, payload)
            send_ns = time.perf_counter_ns() - send_started
            layer_timings.append({
                "layer": layer,
                "role": role,
                "read_ns": read_ns,
                "send_hash_ns": result["hash_ns"],
                "send_write_ns": result["write_ns"],
                "send_total_ns": send_ns,
                "token_start": start,
                "token_end": end,
            })
        seal_started = time.perf_counter_ns()
        artifact = sender.seal(fixture)
        seal_to_ack_ns = time.perf_counter_ns() - seal_started
        committed = True
    finally:
        if not committed:
            sender.abort()
        sender.close()

    totals = {
        "parse_ns": parse_ns,
        "plane_read_ns": read_ns_total,
        "seal_to_ack_ns": seal_to_ack_ns,
        "send_hash_ns": sum(t["send_hash_ns"] for t in layer_timings),
        "send_write_ns": sum(t["send_write_ns"] for t in layer_timings),
        "wire_payload_bytes": begin["expected_payload_bytes"],
    }
    print(f"[llamacpp-sender] totals {json.dumps(totals, sort_keys=True)}", flush=True)
    print(f"[llamacpp-sender] artifact_sha256 {artifact}", flush=True)
    if args.timing_out:
        with open(args.timing_out, "w", encoding="utf-8") as fh:
            json.dump({"layers": layer_timings, "totals_ns": totals,
                       "artifact_sha256": artifact}, fh, indent=1)


if __name__ == "__main__":
    main()
