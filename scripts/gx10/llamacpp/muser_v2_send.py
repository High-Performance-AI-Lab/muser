#!/usr/bin/env python3
"""Stream a llama.cpp Muse session into a Muser Handoff V2 receiver.

The offline path retains the qualified llama.cpp session parser. The live path
accepts 512-token NoPE tiles and the metadata-ordered logical SWA tail directly
from CUDA storage, then applies the epoch-bound HMAC seal over Muser TLS 1.3.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import hmac
import json
import os
import secrets
import socket
import ssl
import struct
import sys
import tempfile
import time
from dataclasses import dataclass

try:
    import numpy as _np
except ImportError:
    _np = None

from llamacpp_session_send import (
    GGML_TYPE_F16,
    GGML_TYPE_F32,
    ProtocolError,
    SessionPlanes,
    class_layer_groups,
    class_row_expectations,
    load_layout,
    read_fixture,
)

MAGIC = b"KVPKV2\0\0"
PROTOCOL = "kvpack-live-handoff-v2"
ALPN = "muser-kvpack-v2"
NOPE_TILE = 512
NOPE_LAYERS = tuple(range(3, 52, 4))
SWA_LAYERS = tuple(layer for layer in range(52) if layer not in NOPE_LAYERS)
SWA_GROUP = 13
PACKED_PLANES = 26
PACKED_ELEMENTS = 256 * PACKED_PLANES
MAX_HEADER = 8 * 1024 * 1024
MAX_PAYLOAD = 512 * 1024 * 1024
SOCKET_BUFFER_BYTES = 8 * 1024 * 1024
LINUX_SO_MAX_PACING_RATE = 47
HANDOFF_PACING_BYTES_PER_SECOND = 1_000_000_000  # 8.0 Gbps; the direct link measures 9.4


def handoff_pacing_bytes_per_second() -> int:
    """Configured payload pacing ceiling (default 8.0 Gbps).

    The N-series pin of 500 MB/s protected the 3 Gbps product floor on an
    unhealthy link. The direct 10GbE path measures 9.4 Gbps single-stream, so
    the default now sits ~15% under line rate. MUSER_GX10_PACING_BYTES_PER_SECOND
    overrides for experiments without a redeploy; a value the kernel refuses
    still fails closed in `configure_linux_pacing`.
    """
    raw = os.environ.get("MUSER_GX10_PACING_BYTES_PER_SECOND")
    if raw is None:
        return HANDOFF_PACING_BYTES_PER_SECOND
    try:
        value = int(raw)
    except ValueError:
        raise ProtocolError("MUSER_GX10_PACING_BYTES_PER_SECOND is not an integer")
    if value <= 0:
        raise ProtocolError("MUSER_GX10_PACING_BYTES_PER_SECOND must be positive")
    return value
STREAM_MAGIC = b"MUSENP1\0"
STREAM_HEADER = struct.Struct("<8sIBQIIQ")
TILE_MAJOR_FIFO_ADAPTERS = frozenset(
    {
        # gx10-container-dflash-rope-nco-v1-20260816.json. This adapter emits
        # each NoPE tile followed by its three SWA groups. The Handoff V2 wire
        # schedule became layer-major later, so the bridge verifies this
        # receipt-bound source order and reorders only after admission.
        "89819a82f8088e1067af81efef38f79cd0754c18003cb1fd78c9bee07cda398c",
        # gx10-container-89e0aa6-combined-abi-v2-20260824T081003Z.json.
        # This rebuild changes the cross-vendor arithmetic patches but keeps
        # the same receipted muser_streaming_kv.patch (9ddad885...) and the
        # same StreamPlaneHeader/tile-major emission order.
        "e3abb106a70de03dadc50f9427997f37ea88c17e29742b4be04f76709dd478b9",
        # gx10-container-89e0aa6-combined-rms-rope-f16-20260824T104500Z.json.
        # The fused-RMS/RoPE rebuild retains that identical streaming patch
        # and source schedule; only its cross-vendor arithmetic patch changed.
        "9f7995a8d1dff3f04e46453e4135e46091150c5affc73a1f20f64bb4da3731fe",
        # gx10-container-89e0aa6-combined-attn32-20260824T114600Z.json.
        # The attention-tree rebuild keeps the same receipted streaming patch
        # and tile-major source schedule.
        "3f86b5ed0f2c73c0c1b68f6529075fb6e21723e8da170430ac75f1c1106b7560",
        # gx10-container-89e0aa6-combined-attn32-residual-f16-20260824T115620Z.json.
        # Named residual materialization does not alter the FIFO ABI or order.
        "fbb73b2393833ba5efeb7e2c726f87dc6c8652a8e5430c8b447502bd32d8a7da",
    }
)


class _LinuxTcpInfo(ctypes.Structure):
    _fields_ = [
        ("state_bytes", ctypes.c_uint8 * 8),
        ("rto", ctypes.c_uint32),
        ("ato", ctypes.c_uint32),
        ("snd_mss", ctypes.c_uint32),
        ("rcv_mss", ctypes.c_uint32),
        ("unacked", ctypes.c_uint32),
        ("sacked", ctypes.c_uint32),
        ("lost", ctypes.c_uint32),
        ("retrans", ctypes.c_uint32),
        ("fackets", ctypes.c_uint32),
        ("last_data_sent", ctypes.c_uint32),
        ("last_ack_sent", ctypes.c_uint32),
        ("last_data_recv", ctypes.c_uint32),
        ("last_ack_recv", ctypes.c_uint32),
        ("pmtu", ctypes.c_uint32),
        ("rcv_ssthresh", ctypes.c_uint32),
        ("rtt", ctypes.c_uint32),
        ("rttvar", ctypes.c_uint32),
        ("snd_ssthresh", ctypes.c_uint32),
        ("snd_cwnd", ctypes.c_uint32),
        ("advmss", ctypes.c_uint32),
        ("reordering", ctypes.c_uint32),
        ("rcv_rtt", ctypes.c_uint32),
        ("rcv_space", ctypes.c_uint32),
        ("total_retrans", ctypes.c_uint32),
        ("pacing_rate", ctypes.c_uint64),
        ("max_pacing_rate", ctypes.c_uint64),
        ("bytes_acked", ctypes.c_uint64),
        ("bytes_received", ctypes.c_uint64),
        ("segs_out", ctypes.c_uint32),
        ("segs_in", ctypes.c_uint32),
        ("notsent_bytes", ctypes.c_uint32),
        ("min_rtt", ctypes.c_uint32),
        ("data_segs_in", ctypes.c_uint32),
        ("data_segs_out", ctypes.c_uint32),
        ("delivery_rate", ctypes.c_uint64),
        ("busy_time", ctypes.c_uint64),
        ("rwnd_limited", ctypes.c_uint64),
        ("sndbuf_limited", ctypes.c_uint64),
    ]


def _linux_tcp_info(stream: socket.socket) -> "_LinuxTcpInfo | None":
    """Return the raw Linux TCP_INFO view of ``stream``, or None off Linux."""
    if not sys.platform.startswith("linux") or not hasattr(socket, "TCP_INFO"):
        return None
    try:
        raw = stream.getsockopt(
            socket.IPPROTO_TCP, socket.TCP_INFO, ctypes.sizeof(_LinuxTcpInfo)
        )
    except (AttributeError, OSError):
        return None
    if len(raw) < ctypes.sizeof(_LinuxTcpInfo):
        return None
    return _LinuxTcpInfo.from_buffer_copy(raw)


def linux_tcp_busy_time_us(stream: socket.socket) -> int | None:
    """Return Linux TCP_INFO cumulative busy time, or None off Linux."""
    info = _linux_tcp_info(stream)
    return None if info is None else int(info.busy_time)


def linux_tcp_wire_snapshot(stream: socket.socket) -> dict | None:
    """Wire-health snapshot for the env-gated per-segment trace.

    A stall self-attributes from these counters: a total_retrans jump plus a
    snd_cwnd collapse plus rto doubling from ~207 ms means wire loss riding
    the RTO backoff ladder; an rwnd_limited jump instead means the receiver
    window was the brake. Returns None off Linux.
    """
    info = _linux_tcp_info(stream)
    if info is None:
        return None
    return {
        "total_retrans": int(info.total_retrans),
        "retrans": int(info.retrans),
        "lost": int(info.lost),
        "rto": int(info.rto),
        "snd_cwnd": int(info.snd_cwnd),
        "busy_time": int(info.busy_time),
        "rwnd_limited": int(info.rwnd_limited),
        "sndbuf_limited": int(info.sndbuf_limited),
        "notsent_bytes": int(info.notsent_bytes),
        "delivery_rate": int(info.delivery_rate),
    }


def configure_linux_pacing(stream: socket.socket) -> int | None:
    """Pin the GX10 payload ceiling just under the measured link rate."""
    if not sys.platform.startswith("linux"):
        return None
    value = struct.pack("Q", handoff_pacing_bytes_per_second())
    try:
        stream.setsockopt(socket.SOL_SOCKET, LINUX_SO_MAX_PACING_RATE, value)
        actual = stream.getsockopt(
            socket.SOL_SOCKET, LINUX_SO_MAX_PACING_RATE, len(value)
        )
    except OSError:
        return None
    return int(struct.unpack("Q", actual)[0])


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_exact(stream: ssl.SSLSocket, count: int) -> bytes:
    output = bytearray()
    while len(output) < count:
        block = stream.recv(count - len(output))
        if not block:
            raise ProtocolError("receiver closed the V2 stream")
        output.extend(block)
    return bytes(output)


def write_frame(
    stream: ssl.SSLSocket, header: dict, payload: bytes = b""
) -> None:
    encoded = canonical(header)
    if not encoded or len(encoded) > MAX_HEADER or len(payload) > MAX_PAYLOAD:
        raise ProtocolError("V2 frame exceeds the closed wire bounds")
    stream.sendall(
        MAGIC
        + struct.pack("<I", len(encoded))
        + struct.pack("<Q", len(payload))
        + encoded
        + payload
    )


def write_payload_frame(stream: ssl.SSLSocket, header: dict, payload: bytes) -> int:
    """Write one payload frame and return only time spent pushing it to TLS."""
    if not payload:
        raise ProtocolError("payload frame cannot be empty")
    started = time.perf_counter_ns()
    write_frame(stream, header, payload)
    elapsed = time.perf_counter_ns() - started
    if elapsed <= 0:
        raise ProtocolError("payload wire timer did not advance")
    return elapsed


def read_frame(stream: ssl.SSLSocket) -> tuple[dict, bytes]:
    if read_exact(stream, len(MAGIC)) != MAGIC:
        raise ProtocolError("receiver returned bad V2 frame magic")
    header_len = struct.unpack("<I", read_exact(stream, 4))[0]
    payload_len = struct.unpack("<Q", read_exact(stream, 8))[0]
    if not 0 < header_len <= MAX_HEADER or payload_len > MAX_PAYLOAD:
        raise ProtocolError("receiver V2 frame lengths are outside bounds")
    header_bytes = read_exact(stream, header_len)
    header = json.loads(header_bytes)
    if not isinstance(header, dict):
        raise ProtocolError("receiver V2 header is not a JSON object")
    return header, read_exact(stream, payload_len)


def load_mac_key(path: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        stat = os.fstat(descriptor)
        if stat.st_mode & 0o077:
            raise ProtocolError("HMAC key file must not be accessible by group/other")
        raw = os.read(descriptor, 4096)
        if os.read(descriptor, 1):
            raise ProtocolError("HMAC key file is unexpectedly large")
    finally:
        os.close(descriptor)
    stripped = raw.strip()
    if len(stripped) == 64:
        try:
            text = stripped.decode("ascii")
            key = bytes.fromhex(text)
        except (UnicodeDecodeError, ValueError) as error:
            raise ProtocolError("HMAC key is not lowercase hexadecimal") from error
        if text != text.lower():
            raise ProtocolError("HMAC key must use lowercase hexadecimal")
    elif len(raw) == 32:
        key = raw
    else:
        raise ProtocolError("HMAC key must be 32 raw bytes or 64 lowercase hex digits")
    if len(key) != 32:
        raise ProtocolError("HMAC key decoded to the wrong size")
    return key


def connect_tls(args: argparse.Namespace) -> ssl.SSLSocket:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.maximum_version = ssl.TLSVersion.TLSv1_3
    context.load_verify_locations(cafile=args.ca_cert)
    context.load_cert_chain(args.client_cert, args.client_key)
    context.set_alpn_protocols([ALPN])
    raw = socket.create_connection(
        (args.receiver_host, args.receiver_port), timeout=args.timeout_seconds
    )
    raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    # Leave the Linux send buffer under TCP autotuning. The GX10 node's
    # net.core.wmem_max is smaller than this requested value; explicitly
    # setting SO_SNDBUF would user-lock the socket at that small cap and turn
    # each layer burst into a scheduling-sensitive stop/start transfer.
    raw.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCKET_BUFFER_BYTES)
    pacing_bytes_per_second = configure_linux_pacing(raw)
    if (
        sys.platform.startswith("linux")
        and pacing_bytes_per_second != handoff_pacing_bytes_per_second()
    ):
        raw.close()
        raise ProtocolError("GX10 handoff TCP pacing could not be pinned")
    raw.settimeout(args.timeout_seconds)
    try:
        stream = context.wrap_socket(raw, server_hostname=args.server_name)
    except BaseException:
        raw.close()
        raise
    if stream.selected_alpn_protocol() != ALPN:
        stream.close()
        raise ProtocolError("receiver did not negotiate the exact Muser ALPN")
    leaf = stream.getpeercert(binary_form=True)
    actual_leaf = sha256(leaf) if leaf is not None else "absent"
    if actual_leaf != args.server_leaf_sha256:
        stream.close()
        raise ProtocolError(
            f"receiver TLS leaf pin mismatch (presented {actual_leaf})"
        )
    return stream


@dataclass(frozen=True)
class Intent:
    sequence: int
    component_id: str
    role: str
    layer: int | None
    start: int
    count: int
    element_type: str
    elements_per_token: int


def packed_planes(intent: Intent) -> list[tuple[int, str]]:
    if intent.role == "nope_tile":
        return [
            (layer, role)
            for layer in NOPE_LAYERS
            for role in ("nope_key", "nope_value")
        ]
    if intent.role == "swa_tile":
        if intent.layer is None:
            raise ProtocolError("swa_tile is missing a group layer")
        try:
            index = SWA_LAYERS.index(intent.layer)
        except ValueError as error:
            raise ProtocolError("swa_tile group is not a Muse SWA layer") from error
        if index % SWA_GROUP:
            raise ProtocolError("swa_tile group is not aligned")
        group = SWA_LAYERS[index : index + SWA_GROUP]
        if len(group) != SWA_GROUP:
            raise ProtocolError("swa_tile group is short")
        return [
            (layer, role) for layer in group for role in ("swa_key", "swa_value")
        ]
    raise ProtocolError(f"{intent.role} is not a packed Muse tile")


def muse_intents(position: int, prefix_cut: int = 0) -> list[Intent]:
    if position < 1:
        raise ProtocolError("cannot hand off an empty prefix")
    if prefix_cut < 0 or prefix_cut >= position:
        raise ProtocolError("prefix cut must leave a nonempty suffix")
    if len(SWA_LAYERS) % SWA_GROUP:
        raise ProtocolError("Muse SWA layers are not divisible into pipe-safe groups")
    # Delta transfers skip the held prefix: NoPE planes are absolute, so only
    # suffix tiles move; SWA planes are a window, so only the newly in-window
    # positions move (a suffix longer than the window re-sends the window).
    swa_start = max(prefix_cut, position - 2048)
    output: list[Intent] = []
    tiles = [
        (start, min(NOPE_TILE, position - start))
        for start in range(prefix_cut, position, NOPE_TILE)
    ]
    # Layer-major order (2026-08-19, connector streaming): every SWA group is
    # emitted for all position tiles as soon as its layers exist mid-prefill,
    # and the NoPE tiles — which need the last NoPE layer — come last, so the
    # producer can stream segments during prefill. The sink installs by
    # segment coordinates; wire order carries no install semantics. Mirrors
    # crates/muser-cluster/src/schedule.rs::muse_schedule_span_for exactly.
    for group_start in range(0, len(SWA_LAYERS), SWA_GROUP):
        for start, count in tiles:
            tile_end = start + count
            if tile_end <= swa_start:
                continue
            chunk_start = max(start, swa_start)
            chunk_count = tile_end - chunk_start
            output.append(
                Intent(
                    len(output),
                    "target",
                    "swa_tile",
                    SWA_LAYERS[group_start],
                    chunk_start,
                    chunk_count,
                    "f16_le",
                    PACKED_ELEMENTS,
                )
            )
    for start, count in tiles:
        output.append(
            Intent(
                len(output),
                "target",
                "nope_tile",
                None,
                start,
                count,
                "f16_le",
                PACKED_ELEMENTS,
            )
        )
    return output


def tile_major_fifo_intents(position: int) -> list[Intent]:
    """The exact live-FIFO order emitted by the pinned llama.cpp adapter."""
    if position < 1:
        raise ProtocolError("cannot hand off an empty prefix")
    swa_start = max(0, position - 2048)
    output: list[Intent] = []
    for start in range(0, position, NOPE_TILE):
        count = min(NOPE_TILE, position - start)
        output.append(
            Intent(
                len(output),
                "target",
                "nope_tile",
                None,
                start,
                count,
                "f16_le",
                PACKED_ELEMENTS,
            )
        )
        tile_end = start + count
        if tile_end <= swa_start:
            continue
        chunk_start = max(start, swa_start)
        chunk_count = tile_end - chunk_start
        for group_start in range(0, len(SWA_LAYERS), SWA_GROUP):
            output.append(
                Intent(
                    len(output),
                    "target",
                    "swa_tile",
                    SWA_LAYERS[group_start],
                    chunk_start,
                    chunk_count,
                    "f16_le",
                    PACKED_ELEMENTS,
                )
            )
    return output


def intent_coordinates(intent: Intent) -> tuple[object, ...]:
    """Semantic coordinates independent of a schedule's sequence number."""
    return (
        intent.component_id,
        intent.role,
        intent.layer,
        intent.start,
        intent.count,
        intent.element_type,
        intent.elements_per_token,
    )


def dflash_intents(
    position: int,
    layers: int,
    width: int,
    sink_size: int,
    window_size: int,
    sequence: int,
) -> list[Intent]:
    ranges = (
        [(0, position)]
        if position <= sink_size + window_size
        else [(0, sink_size), (position - window_size, window_size)]
    )
    output: list[Intent] = []
    for layer in range(layers):
        for role in ("dflash_key", "dflash_value"):
            for start, count in ranges:
                output.append(
                    Intent(
                        sequence + len(output),
                        "dflash",
                        role,
                        layer,
                        start,
                        count,
                        "f32_le",
                        width,
                    )
                )
    return output


_numpy_fallback_logged = False


def f16_to_f32_le(payload: bytes) -> bytes:
    if len(payload) % 2:
        raise ProtocolError("DFlash f16 payload is not word aligned")
    if _np is not None:
        return _np.frombuffer(payload, dtype="<f2").astype("<f4").tobytes()
    global _numpy_fallback_logged
    if not _numpy_fallback_logged:
        print(
            "muser-v2-send: numpy unavailable, falling back to the slow "
            "per-element DFlash f16->f32 conversion",
            file=sys.stderr,
            flush=True,
        )
        _numpy_fallback_logged = True
    output = bytearray(len(payload) * 2)
    for index in range(0, len(payload), 2):
        value = struct.unpack_from("<e", payload, index)[0]
        struct.pack_into("<f", output, index * 2, value)
    return bytes(output)


def payload_for(
    planes: SessionPlanes,
    intent: Intent,
    dflash_planes: SessionPlanes | None = None,
) -> bytes:
    source = planes if intent.component_id == "target" else dflash_planes
    if source is None:
        raise ProtocolError("DFlash segment requested without a draft session")
    if intent.role in ("nope_tile", "swa_tile"):
        blob = bytearray()
        for layer, role_name in packed_planes(intent):
            role = "key" if role_name.endswith("_key") else "value"
            blob.extend(
                source.read_layer_plane(
                    layer, role, intent.start, intent.start + intent.count
                )
            )
        return bytes(blob)
    role = "key" if intent.role.endswith("_key") else "value"
    if intent.component_id == "target":
        payload = source.read_layer_plane(
            intent.layer, role, intent.start, intent.start + intent.count
        )
    elif role == "key":
        payload = source.read_plane(
            source.k_planes,
            intent.layer,
            intent.start,
            intent.start + intent.count,
        )
    else:
        payload = source.read_value_plane(
            intent.layer, intent.start, intent.start + intent.count
        )
    if intent.element_type != "f32_le":
        return payload
    if source.element_type == GGML_TYPE_F32:
        return payload
    if source.element_type == GGML_TYPE_F16:
        return f16_to_f32_le(payload)
    raise ProtocolError("DFlash source cache has an unsupported element type")


def build_material(
    planes: SessionPlanes,
    intents: list[Intent],
    dflash_planes: SessionPlanes | None = None,
) -> tuple[list[dict], int]:
    descriptors: list[dict] = []
    total = 0
    for intent in intents:
        payload = payload_for(planes, intent, dflash_planes)
        descriptor = {
            "sequence": intent.sequence,
            "component_id": intent.component_id,
            "role": intent.role,
            "layer": intent.layer,
            "logical_start": intent.start,
            "logical_count": intent.count,
            "element_type": intent.element_type,
            "elements_per_token": intent.elements_per_token,
            "byte_len": len(payload),
            "sha256": sha256(payload),
        }
        width = 4 if intent.element_type == "f32_le" else 2
        expected = intent.count * intent.elements_per_token * width
        if len(payload) != expected:
            raise ProtocolError(
                f"layer {intent.layer} {intent.role} has {len(payload)} bytes, "
                f"expected {expected}"
            )
        descriptors.append(descriptor)
        total += len(payload)
    return descriptors, total


def build_begin(
    args: argparse.Namespace,
    cached_tokens: list[int],
    descriptors: list[dict],
    prefix_cut: int = 0,
) -> dict:
    now = time.time_ns() // 1_000_000
    begin = {
        "protocol": PROTOCOL,
        "transfer_id": args.transfer_id or secrets.token_hex(32),
        "generation": args.generation,
        "created_unix_ms": now,
        "expires_unix_ms": now + args.timeout_seconds * 1000,
        "identity": {
            "adapter_sha256": args.adapter_sha256,
            "chat_template_sha256": args.chat_template_sha256,
            "context_policy_sha256": args.context_policy_sha256,
            "model_revision": args.model_revision,
            "model_sha256": args.model_sha256,
            "tokenizer_revision": args.tokenizer_revision,
            "tokenizer_sha256": args.tokenizer_sha256,
        },
        "prompt_token_ids": cached_tokens,
        "multimodal": (
            {
                "projector_sha256": args.multimodal_projector_sha256,
                "preprocessing_sha256": args.multimodal_preprocessing_sha256,
                "image_sequence_sha256": args.multimodal_image_sequence_sha256,
            }
            if args.multimodal_projector_sha256
            else None
        ),
        "hmac": {"key_id": args.hmac_key_id, "epoch": args.hmac_epoch},
        "components": [
            {
                "id": "target",
                "kind": "target_kv",
                "required": True,
                "identity_sha256": args.target_cache_identity_sha256,
            },
            *(
                [
                    {
                        "id": "dflash",
                        "kind": "dflash_context",
                        "required": True,
                        "identity_sha256": args.dflash_identity_sha256,
                    }
                ]
                if args.dflash_session
                else []
            ),
        ],
        "segments": descriptors,
    }
    # A delta handoff names its prefix cut; full transfers omit the field so
    # their canonical manifest stays byte-identical to the pre-delta protocol.
    if prefix_cut:
        begin["prefix_cut"] = prefix_cut
    return begin


def build_seal(
    begin: dict,
    descriptors: list[dict],
    payload_hash: str,
    total_bytes: int,
    key: bytes,
) -> dict:
    descriptor_hash = hashlib.sha256()
    for descriptor in descriptors:
        descriptor_hash.update(canonical(descriptor))
    # The receiver seals the canonical typed manifest, which drops
    # prefix_cut on its typed parse; the field rides the wire but must not
    # enter the sealed begin hash or the delta seal mismatches.
    sealed_begin = {field: value for field, value in begin.items() if field != "prefix_cut"}
    core = {
        "transfer_id": begin["transfer_id"],
        "generation": begin["generation"],
        "begin_sha256": sha256(canonical(sealed_begin)),
        "descriptor_sha256": descriptor_hash.hexdigest(),
        "payload_sha256": payload_hash,
        "segment_count": len(descriptors),
        "total_bytes": total_bytes,
    }
    return {
        "core": core,
        "hmac_sha256": hmac.new(key, canonical(core), hashlib.sha256).hexdigest(),
    }


def require_sha256(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    fields = (
        "model_sha256",
        "tokenizer_sha256",
        "chat_template_sha256",
        "context_policy_sha256",
        "adapter_sha256",
        "target_cache_identity_sha256",
        "server_leaf_sha256",
    )
    for field in fields:
        value = getattr(args, field)
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            parser.error(f"--{field.replace('_', '-')} must be lowercase SHA-256")


def parse_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session")
    parser.add_argument("--nope-fifo")
    parser.add_argument("--dflash-session")
    parser.add_argument("--dflash-identity-sha256")
    parser.add_argument("--dflash-kv-heads", type=int)
    parser.add_argument("--dflash-head-dim", type=int)
    parser.add_argument("--dflash-context-layers", type=int)
    parser.add_argument("--dflash-context-elements-per-token", type=int)
    parser.add_argument("--dflash-context-sink-size", type=int)
    parser.add_argument("--dflash-context-window-size", type=int)
    parser.add_argument("--multimodal-projector-sha256")
    parser.add_argument("--multimodal-preprocessing-sha256")
    parser.add_argument("--multimodal-image-sequence-sha256")
    parser.add_argument("--prompt-token-fixture", required=True)
    parser.add_argument(
        "--layout-json",
        default=os.path.join(os.path.dirname(__file__), "muse-glimmer-30b.layout.json"),
    )
    parser.add_argument("--receiver-host", required=True)
    parser.add_argument("--receiver-port", type=int, default=29590)
    parser.add_argument("--server-name", default="muser-prefilld")
    parser.add_argument("--ca-cert", required=True)
    parser.add_argument("--client-cert", required=True)
    parser.add_argument("--client-key", required=True)
    parser.add_argument("--server-leaf-sha256", required=True)
    parser.add_argument("--hmac-key-file", required=True)
    parser.add_argument("--hmac-key-id", required=True)
    parser.add_argument("--hmac-epoch", type=int, required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--transfer-id")
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--chat-template-sha256", required=True)
    parser.add_argument("--context-policy-sha256", required=True)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--target-cache-identity-sha256", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--dump-only", action="store_true")
    return parser, parser.parse_args()


def fifo_exact(stream: object, count: int, *, allow_eof: bool = False) -> bytes:
    output = bytearray()
    while len(output) < count:
        block = stream.read(count - len(output))
        if not block:
            if allow_eof and not output:
                return b""
            raise ProtocolError("streaming target FIFO ended inside a frame")
        output.extend(block)
    return bytes(output)


def read_live_fifo_payload(stream: object, expected: Intent) -> bytes:
    role_codes = {
        "nope_key": 0,
        "nope_value": 1,
        "swa_key": 2,
        "swa_value": 3,
    }
    payload = bytearray()
    for layer, role in packed_planes(expected):
        raw = fifo_exact(stream, STREAM_HEADER.size)
        magic, got_layer, got_role, start, count, elements, byte_len = (
            STREAM_HEADER.unpack(raw)
        )
        if (
            magic != STREAM_MAGIC
            or got_role != role_codes[role]
            or got_layer != layer
            or start != expected.start
            or count != expected.count
            or elements != 256
            or byte_len != expected.count * 512
        ):
            raise ProtocolError(
                "live target frame differs from the receipt-bound FIFO schedule"
            )
        payload.extend(fifo_exact(stream, byte_len))
    return bytes(payload)


def live_target_payloads(
    stream: object,
    position: int,
    adapter_sha256: str,
):
    """Yield verified FIFO payloads in the current Handoff V2 wire order.

    The pinned llama.cpp exporter predates the layer-major wire schedule and
    emits tile-major FIFO blobs. Semantic coordinates are identical. Admit
    only its receipted adapter digest, verify every plane in source order, and
    spool out-of-order blobs on the node's internal temporary filesystem until
    their layer-major turn. No payload is relabelled or admitted by shape only.
    """
    if adapter_sha256 not in TILE_MAJOR_FIFO_ADAPTERS:
        raise ProtocolError(
            "adapter has no qualified live target FIFO schedule: "
            f"{adapter_sha256}"
        )
    source = tile_major_fifo_intents(position)
    wire = muse_intents(position)
    source_keys = [intent_coordinates(intent) for intent in source]
    wire_keys = [intent_coordinates(intent) for intent in wire]
    if (
        len(set(source_keys)) != len(source_keys)
        or len(set(wire_keys)) != len(wire_keys)
        or set(source_keys) != set(wire_keys)
    ):
        raise ProtocolError("live FIFO and Handoff V2 schedules are not equivalent")

    with tempfile.TemporaryFile(prefix="muser-live-target-reorder-") as spool:
        buffered: dict[tuple[object, ...], tuple[int, int]] = {}
        next_wire = 0
        for source_intent, source_key in zip(source, source_keys, strict=True):
            payload = read_live_fifo_payload(stream, source_intent)
            if source_key in buffered:
                raise ProtocolError("live target FIFO repeated a semantic segment")
            if source_key == wire_keys[next_wire]:
                expected = wire[next_wire]
                next_wire += 1
                yield expected, payload
            else:
                spool.seek(0, os.SEEK_END)
                offset = spool.tell()
                if spool.write(payload) != len(payload):
                    raise ProtocolError("live target reorder spool write was short")
                buffered[source_key] = (offset, len(payload))

            while next_wire < len(wire) and wire_keys[next_wire] in buffered:
                expected = wire[next_wire]
                offset, byte_len = buffered.pop(wire_keys[next_wire])
                spool.flush()
                spool.seek(offset)
                payload = spool.read(byte_len)
                if len(payload) != byte_len:
                    raise ProtocolError("live target reorder spool read was short")
                next_wire += 1
                yield expected, payload

        if fifo_exact(stream, 1, allow_eof=True):
            raise ProtocolError("live target FIFO contains an unexpected extra frame")
        if next_wire != len(wire) or buffered:
            raise ProtocolError("live target FIFO stopped before the wire schedule completed")


def descriptor_for(intent: Intent, payload: bytes) -> dict:
    width = 4 if intent.element_type == "f32_le" else 2
    expected = intent.count * intent.elements_per_token * width
    if len(payload) != expected:
        raise ProtocolError(
            f"layer {intent.layer} {intent.role} has {len(payload)} bytes, expected {expected}"
        )
    return {
        "sequence": intent.sequence,
        "component_id": intent.component_id,
        "role": intent.role,
        "layer": intent.layer,
        "logical_start": intent.start,
        "logical_count": intent.count,
        "element_type": intent.element_type,
        "elements_per_token": intent.elements_per_token,
        "byte_len": len(payload),
        "sha256": sha256(payload),
    }


class DeferredHandoffV2Sender:
    """One fail-closed, layer-streamed Handoff V2 transfer.

    Producers supply the already-packed payloads in ``muse_intents`` order.
    An optional precomputed DFlash session is appended immediately after the
    streamed target schedule and before the single atomic seal.
    This keeps TLS, identity, descriptor hashing, sealing, and ACK validation in
    the same implementation used by the qualified llama.cpp live path.
    """

    def __init__(
        self, args: argparse.Namespace, fixture: list[int], prefix_cut: int = 0
    ) -> None:
        if len(fixture) < 2:
            raise ProtocolError("deferred handoff fixture needs a cached prefix and boundary")
        if prefix_cut < 0 or prefix_cut >= len(fixture) - 1 or prefix_cut % 256 != 0:
            raise ProtocolError(
                "prefix cut must be 256-aligned and leave a nonempty suffix"
            )
        self._args = args
        self._fixture = list(fixture)
        cached_tokens = fixture[:-1]
        self._intents = muse_intents(len(cached_tokens), prefix_cut)
        self._dflash_planes: SessionPlanes | None = None
        self._dflash_session_path: str | None = getattr(args, "dflash_session", None)
        if getattr(args, "dflash_session", None):
            width = args.dflash_kv_heads * args.dflash_head_dim
            self._intents.extend(
                dflash_intents(
                    len(cached_tokens),
                    args.dflash_context_layers,
                    width,
                    args.dflash_context_sink_size,
                    args.dflash_context_window_size,
                    len(self._intents),
                )
            )
        self._begin = build_begin(args, cached_tokens, [], prefix_cut)
        self._begin["deferred_segments"] = True
        self._key = load_mac_key(args.hmac_key_file)
        self._transfer_start_unix_ns = time.time_ns()
        self._wire = connect_tls(args)
        self._descriptors: list[dict] = []
        self._payload_stream = hashlib.sha256()
        self._total = 0
        self._payload_wire_ns = 0
        self._payload_pacing_bps = (
            handoff_pacing_bytes_per_second() * 8
            if sys.platform.startswith("linux")
            else None
        )
        self._payload_busy_start_us: int | None = None
        # MUSER_GX10_WIRE_TRACE gates per-segment wire telemetry (unset or
        # "0" = off = the qualified default). Trace entries live in memory
        # and dump to stderr on close; receipts and payload bytes untouched.
        self._wire_trace: list[dict] | None = (
            [] if os.environ.get("MUSER_GX10_WIRE_TRACE", "0") not in ("", "0") else None
        )
        self._first_segment_sent_unix_ns = 0
        self._closed = False
        self._committed = False
        try:
            write_frame(self._wire, {"kind": "begin", "manifest": self._begin})
        except BaseException:
            self._wire.close()
            self._closed = True
            raise

    @property
    def next_intent(self) -> Intent | None:
        index = len(self._descriptors)
        return self._intents[index] if index < len(self._intents) else None

    def send(self, intent: Intent, payload: bytes) -> dict:
        if self._closed:
            raise ProtocolError("cannot send on a closed Handoff V2 transfer")
        expected = self.next_intent
        if expected is None:
            raise ProtocolError("producer supplied an unexpected extra Muse segment")
        if expected.component_id != "target":
            raise ProtocolError("producer supplied target bytes after the DFlash schedule began")
        if intent != expected:
            raise ProtocolError("producer segment differs from the declared Muse schedule")
        descriptor = descriptor_for(expected, payload)
        self._payload_stream.update(payload)
        self._total += len(payload)
        if self._payload_busy_start_us is None:
            self._payload_busy_start_us = linux_tcp_busy_time_us(self._wire)
        write_ns = write_payload_frame(
            self._wire,
            {
                "kind": "segment",
                "sequence": expected.sequence,
                "descriptor": descriptor,
            },
            payload,
        )
        self._payload_wire_ns += write_ns
        if self._wire_trace is not None:
            self._wire_trace.append(
                {
                    "seq": expected.sequence,
                    "sent_unix_ns": time.time_ns(),
                    "write_ns": write_ns,
                    "snapshot": linux_tcp_wire_snapshot(self._wire),
                }
            )
        if self._first_segment_sent_unix_ns == 0:
            self._first_segment_sent_unix_ns = time.time_ns()
        self._descriptors.append(descriptor)
        return descriptor

    def _send_deferred_dflash(self) -> None:
        if self._dflash_session_path is not None and self._dflash_planes is None:
            self._dflash_planes = SessionPlanes(
                self._dflash_session_path,
                self._args.dflash_kv_heads,
                self._args.dflash_head_dim,
                self._fixture,
                element_type=GGML_TYPE_F32,
            )
            if (
                self._dflash_planes.n_layer != self._args.dflash_context_layers
                or self._dflash_planes.blocks[0].cell_count != len(self._fixture) - 1
            ):
                raise ProtocolError("deferred draft session does not contain the exact prefix")
        while self.next_intent is not None:
            expected = self.next_intent
            if expected is None or expected.component_id != "dflash":
                raise ProtocolError("deferred transfer stopped before the DFlash schedule")
            if self._dflash_planes is None:
                raise ProtocolError("DFlash schedule exists without a draft session")
            payload = payload_for(self._dflash_planes, expected, self._dflash_planes)
            descriptor = descriptor_for(expected, payload)
            self._payload_stream.update(payload)
            self._total += len(payload)
            self._payload_wire_ns += write_payload_frame(
                self._wire,
                {
                    "kind": "segment",
                    "sequence": expected.sequence,
                    "descriptor": descriptor,
                },
                payload,
            )
            self._descriptors.append(descriptor)

    def seal(self) -> dict:
        if self._closed:
            raise ProtocolError("cannot seal a closed Handoff V2 transfer")
        self._send_deferred_dflash()
        if self.next_intent is not None:
            raise ProtocolError("cannot seal an incomplete Muse transfer")
        seal = build_seal(
            self._begin,
            self._descriptors,
            self._payload_stream.hexdigest(),
            self._total,
            self._key,
        )
        write_frame(self._wire, {"kind": "seal", "manifest": seal})
        header, payload = read_frame(self._wire)
        if payload or header.get("kind") != "ack":
            raise ProtocolError(f"receiver returned non-ACK frame: {header!r}")
        if (
            header.get("transfer_id") != self._begin["transfer_id"]
            or header.get("generation") != self._begin["generation"]
        ):
            raise ProtocolError("receiver ACK identity differs from the committed generation")
        payload_wire_ns = self._payload_wire_ns
        payload_wire_source = "sendall-blocked-time-v1"
        payload_busy_end_us = linux_tcp_busy_time_us(self._wire)
        if (
            self._payload_busy_start_us is not None
            and payload_busy_end_us is not None
            and payload_busy_end_us > self._payload_busy_start_us
        ):
            payload_wire_ns = (
                payload_busy_end_us - self._payload_busy_start_us
            ) * 1_000
            payload_wire_source = "linux-tcp-info-busy-time-v1"
        self._committed = True
        receipt = {
            "ack": True,
            "streaming_target": True,
            "generation": self._begin["generation"],
            "transfer_id": self._begin["transfer_id"],
            "transfer_start_unix_ns": self._transfer_start_unix_ns,
            "first_segment_sent_unix_ns": self._first_segment_sent_unix_ns,
            "transfer_acked_unix_ns": time.time_ns(),
            "payload_bytes": self._total,
            "payload_sha256": self._payload_stream.hexdigest(),
            "payload_wire_ns": payload_wire_ns,
            "payload_wire_source": payload_wire_source,
            "payload_pacing_bps": self._payload_pacing_bps,
            "segments": len(self._descriptors),
        }
        self.close()
        return receipt

    def abort(self, reason: str = "producer failure") -> None:
        if self._closed:
            return
        try:
            write_frame(self._wire, {"kind": "abort", "reason": reason})
        finally:
            self.close()

    def _dump_wire_trace(self) -> None:
        """Emit the gated per-segment trace as stderr JSONL, then drop it.

        seal() and abort() both terminate through close(), so the dump hooks
        there and fires exactly once per transfer; the receipt dict and the
        wire protocol never carry these lines.
        """
        if not self._wire_trace:
            return
        try:
            for entry in self._wire_trace:
                print(
                    "muser-v2-send: wire-trace "
                    + json.dumps(entry, sort_keys=True, separators=(",", ":")),
                    file=sys.stderr,
                )
            sys.stderr.flush()
        finally:
            self._wire_trace = []

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._committed:
            try:
                write_frame(self._wire, {"kind": "abort", "reason": "producer failure"})
            except Exception:
                pass
        self._wire.close()
        self._dump_wire_trace()

    def __enter__(self) -> "DeferredHandoffV2Sender":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def stream_live_target(
    args: argparse.Namespace,
    layout: dict,
    fixture: list[int],
    cached_tokens: list[int],
) -> None:
    """Send target planes directly from CUDA, then an optional draft session."""
    if args.dump_only:
        raise ProtocolError("--dump-only cannot consume a live target FIFO")
    begin = build_begin(args, cached_tokens, [])
    begin["deferred_segments"] = True
    key = load_mac_key(args.hmac_key_file)
    transfer_start_unix_ns = time.time_ns()
    wire = connect_tls(args)
    descriptors: list[dict] = []
    payload_stream = hashlib.sha256()
    total = 0
    payload_wire_ns = 0
    first_segment_sent_unix_ns = 0
    committed = False
    try:
        write_frame(wire, {"kind": "begin", "manifest": begin})
        with open(args.nope_fifo, "rb", buffering=SOCKET_BUFFER_BYTES) as fifo:
            for expected, payload in live_target_payloads(
                fifo, len(cached_tokens), args.adapter_sha256
            ):
                descriptor = descriptor_for(expected, payload)
                descriptors.append(descriptor)
                payload_stream.update(payload)
                total += len(payload)
                payload_wire_ns += write_payload_frame(
                    wire,
                    {
                        "kind": "segment",
                        "sequence": expected.sequence,
                        "descriptor": descriptor,
                    },
                    payload,
                )
                if first_segment_sent_unix_ns == 0:
                    first_segment_sent_unix_ns = time.time_ns()

        dflash_planes = None
        remaining: list[Intent] = []
        if args.dflash_session:
            dflash_planes = SessionPlanes(
                args.dflash_session,
                args.dflash_kv_heads,
                args.dflash_head_dim,
                fixture,
                element_type=GGML_TYPE_F32,
            )
            if (
                dflash_planes.n_layer != args.dflash_context_layers
                or dflash_planes.blocks[0].cell_count != len(cached_tokens)
            ):
                raise ProtocolError("final draft session does not contain the exact prefix")
            width = args.dflash_kv_heads * args.dflash_head_dim
            remaining.extend(
                dflash_intents(
                    len(cached_tokens),
                    args.dflash_context_layers,
                    width,
                    args.dflash_context_sink_size,
                    args.dflash_context_window_size,
                    len(descriptors),
                )
            )
        for intent in remaining:
            if dflash_planes is None:
                raise ProtocolError("DFlash segment requested without a draft session")
            payload = payload_for(dflash_planes, intent, dflash_planes)
            descriptor = descriptor_for(intent, payload)
            descriptors.append(descriptor)
            payload_stream.update(payload)
            total += len(payload)
            payload_wire_ns += write_payload_frame(
                wire,
                {
                    "kind": "segment",
                    "sequence": intent.sequence,
                    "descriptor": descriptor,
                },
                payload,
            )
        seal = build_seal(begin, descriptors, payload_stream.hexdigest(), total, key)
        write_frame(wire, {"kind": "seal", "manifest": seal})
        header, payload = read_frame(wire)
        if payload or header.get("kind") != "ack":
            raise ProtocolError(f"receiver returned non-ACK frame: {header!r}")
        if (
            header.get("transfer_id") != begin["transfer_id"]
            or header.get("generation") != begin["generation"]
        ):
            raise ProtocolError("receiver ACK identity differs from the committed generation")
        committed = True
        transfer_acked_unix_ns = time.time_ns()
        print(
            json.dumps(
                {
                    "ack": True,
                    "streaming_target": True,
                    "generation": begin["generation"],
                    "transfer_id": begin["transfer_id"],
                    "transfer_start_unix_ns": transfer_start_unix_ns,
                    "first_segment_sent_unix_ns": first_segment_sent_unix_ns,
                    "transfer_acked_unix_ns": transfer_acked_unix_ns,
                    "payload_bytes": total,
                    "payload_wire_ns": payload_wire_ns,
                    "segments": len(descriptors),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        if not committed:
            try:
                write_frame(wire, {"kind": "abort", "reason": "producer failure"})
            except Exception:
                pass
        wire.close()


def main() -> None:
    parser, args = parse_args()
    require_sha256(parser, args)
    if args.hmac_epoch < 1 or args.generation < 1:
        parser.error("HMAC epoch and generation must be positive")
    if bool(args.dflash_session) != bool(args.dflash_identity_sha256):
        parser.error("--dflash-session and --dflash-identity-sha256 are a pair")
    multimodal_values = (
        args.multimodal_projector_sha256,
        args.multimodal_preprocessing_sha256,
        args.multimodal_image_sequence_sha256,
    )
    if any(multimodal_values) and not all(multimodal_values):
        parser.error("all three --multimodal-*-sha256 values are required together")
    if args.dflash_session and any(multimodal_values):
        parser.error("this release keeps DFlash and multimodal transfers disjoint")
    dflash_geometry = (
        args.dflash_kv_heads,
        args.dflash_head_dim,
        args.dflash_context_layers,
        args.dflash_context_elements_per_token,
        args.dflash_context_sink_size,
        args.dflash_context_window_size,
    )
    if args.dflash_session:
        if any(value is None or value < 1 for value in dflash_geometry):
            parser.error("all DFlash context geometry fields must be positive")
        if (
            args.dflash_kv_heads * args.dflash_head_dim
            != args.dflash_context_elements_per_token
        ):
            parser.error(
                "DFlash elements_per_token differs from KV heads times head dimension"
            )
    elif any(value is not None for value in dflash_geometry):
        parser.error("DFlash context geometry requires --dflash-session")
    if args.dflash_identity_sha256 and (
        len(args.dflash_identity_sha256) != 64
        or any(c not in "0123456789abcdef" for c in args.dflash_identity_sha256)
    ):
        parser.error("--dflash-identity-sha256 must be lowercase SHA-256")
    for field, value in zip(
        (
            "projector",
            "preprocessing",
            "image-sequence",
        ),
        multimodal_values,
        strict=True,
    ):
        if value and (len(value) != 64 or any(c not in "0123456789abcdef" for c in value)):
            parser.error(f"--multimodal-{field}-sha256 must be lowercase SHA-256")
    layout = load_layout(args.layout_json)
    if layout.get("name") != "muse-glimmer-30b":
        parser.error("layout must be the Muse Glimmer 30B layout")
    fixture = read_fixture(args.prompt_token_fixture)
    cached_tokens = fixture[:-1]
    rows = class_row_expectations(layout["layout_table"], 52)
    groups = class_layer_groups(layout["layout_table"])
    if args.nope_fifo:
        stream_live_target(args, layout, fixture, cached_tokens)
        return
    if not args.session:
        parser.error("--session is required unless --nope-fifo is used")
    parse_started = time.perf_counter_ns()
    planes = SessionPlanes(
        args.session,
        2,
        128,
        fixture,
        row_expectations=rows,
        layer_groups=groups,
    )
    if planes.n_layer != 52 or planes.blocks[0].cell_count != len(cached_tokens):
        raise ProtocolError("llama session does not contain the exact Muse prefix")
    intents = muse_intents(len(cached_tokens))
    dflash_planes = None
    if args.dflash_session:
        dflash_planes = SessionPlanes(
            args.dflash_session,
            args.dflash_kv_heads,
            args.dflash_head_dim,
            fixture,
            element_type=GGML_TYPE_F32,
        )
        if (
            dflash_planes.n_layer != args.dflash_context_layers
            or dflash_planes.blocks[0].cell_count != len(cached_tokens)
        ):
            raise ProtocolError("draft session does not contain the exact five-layer prefix")
        dflash_width = args.dflash_kv_heads * args.dflash_head_dim
        intents.extend(
            dflash_intents(
                len(cached_tokens),
                args.dflash_context_layers,
                dflash_width,
                args.dflash_context_sink_size,
                args.dflash_context_window_size,
                len(intents),
            )
        )
    descriptors, total = build_material(planes, intents, dflash_planes)
    begin = build_begin(args, cached_tokens, descriptors)
    print(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "cached_tokens": len(cached_tokens),
                "segments": len(descriptors),
                "payload_bytes": total,
                "parse_and_hash_ns": time.perf_counter_ns() - parse_started,
                "transfer_id": begin["transfer_id"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.dump_only:
        return
    key = load_mac_key(args.hmac_key_file)
    transfer_start_unix_ns = time.time_ns()
    stream = connect_tls(args)
    payload_stream = hashlib.sha256()
    committed = False
    first_segment_sent_unix_ns = 0
    payload_wire_ns = 0
    try:
        write_frame(stream, {"kind": "begin", "manifest": begin})
        for intent, descriptor in zip(intents, descriptors, strict=True):
            payload = payload_for(planes, intent, dflash_planes)
            if sha256(payload) != descriptor["sha256"]:
                raise ProtocolError("llama session changed between descriptor and send passes")
            payload_stream.update(payload)
            payload_wire_ns += write_payload_frame(
                stream,
                {"kind": "segment", "sequence": intent.sequence},
                payload,
            )
            if first_segment_sent_unix_ns == 0:
                first_segment_sent_unix_ns = time.time_ns()
        seal = build_seal(
            begin, descriptors, payload_stream.hexdigest(), total, key
        )
        write_frame(stream, {"kind": "seal", "manifest": seal})
        header, payload = read_frame(stream)
        if payload or header.get("kind") != "ack":
            raise ProtocolError(f"receiver returned non-ACK frame: {header!r}")
        if (
            header.get("transfer_id") != begin["transfer_id"]
            or header.get("generation") != begin["generation"]
        ):
            raise ProtocolError("receiver ACK identity differs from the committed generation")
        committed = True
        transfer_acked_unix_ns = time.time_ns()
        print(
            json.dumps(
                {
                    "ack": True,
                    "generation": begin["generation"],
                    "transfer_id": begin["transfer_id"],
                    "transfer_start_unix_ns": transfer_start_unix_ns,
                    "first_segment_sent_unix_ns": first_segment_sent_unix_ns,
                    "transfer_acked_unix_ns": transfer_acked_unix_ns,
                    "payload_bytes": total,
                    "payload_wire_ns": payload_wire_ns,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        if not committed:
            try:
                write_frame(stream, {"kind": "abort", "reason": "producer failure"})
            except Exception:
                pass
        stream.close()


if __name__ == "__main__":
    main()
