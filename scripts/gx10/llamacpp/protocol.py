"""Dependency-free Python implementation of kvpack-live-f16-le-v1."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import ssl
import struct
import sys
import time
from dataclasses import dataclass
from typing import Any, BinaryIO, Callable

FRAME_MAGIC = b"KVHF"
SCHEMA_VERSION = 1
PROTOCOL = "kvpack-live-f16-le-v1"
PORTABLE_ABI = "canonical-kv-f16-le-v1"
FRAME_HEADER = struct.Struct(">4sBBHIQ")
FRAME_BEGIN = 1
FRAME_LAYER = 2
FRAME_SEAL = 3
FRAME_ABORT = 4
FRAME_ACK = 5
MAX_JSON_BYTES = 1024 * 1024
PORTABLE_ABI_V2 = "canonical-kv-v2"
WIRE_SCHEDULE_LAYER_ORDER = "layer-order"
WIRE_SCHEDULE_DECODE_PRIORITY = "decode-priority"
KNOWN_WIRE_SCHEDULES = frozenset(
    {WIRE_SCHEDULE_LAYER_ORDER, WIRE_SCHEDULE_DECODE_PRIORITY}
)

# WS9 transport squeeze knobs (all disabled-at-default-behavior-preserving).
SOCKET_BUFFER_BYTES_ENV = "KVPACK_SOCKET_BUFFER_BYTES"
DEFAULT_SOCKET_BUFFER_BYTES = 8 * 1024 * 1024
INFLIGHT_WINDOW_ENV = "KVPACK_INFLIGHT_WINDOW"
DEFAULT_INFLIGHT_WINDOW = 4
# WS9 TENT-lite multi-stream spraying (opt-in, off by default). The receiver
# must set KVPACK_HANDOFF_STREAMS to the SAME value; the two ends negotiate
# nothing, so a mismatch deadlocks the session into the receiver's timeout.
# With the env unset the producer is byte-identical to the pre-spray path:
# one connection, BEGIN + ordered layer frames + SEAL on stream 0.
SPRAY_STREAMS_ENV = "KVPACK_SPRAY_STREAMS"
# A stream is resteered away when its sliding-window throughput drops below
# this fraction of the best stream's, sustained over at least this many sends
# on that stream. Reassembly never depends on the assignment: every frame
# carries its sequence and is assigned exactly once at send time.
SPRAY_RESTEER_FRACTION_ENV = "KVPACK_SPRAY_RESTEER_FRACTION"
DEFAULT_SPRAY_RESTEER_FRACTION = 0.5
SPRAY_RESTEER_MIN_OBSERVATIONS_ENV = "KVPACK_SPRAY_RESTEER_MIN_OBSERVATIONS"
DEFAULT_SPRAY_RESTEER_MIN_OBSERVATIONS = 4
SPRAY_HISTORY_WINDOW = 8
# TLS 1.3 suites the receiver is configured to prefer, in server-preference
# order (hardware AES exists on both ends of the direct link). CPython's
# ssl module cannot order TLS 1.3 suites client-side (set_ciphers only
# covers <= TLS 1.2 and there is no set_ciphersuites binding), so the
# producer verifies the negotiated suite fail-closed after the handshake
# instead of pretending to control it.
ALLOWED_TLS13_SUITES = frozenset(
    {
        "TLS_AES_256_GCM_SHA384",
        "TLS_CHACHA20_POLY1305_SHA256",
        "TLS_AES_128_GCM_SHA256",
    }
)


def _socket_buffer_bytes() -> int:
    raw = os.environ.get(SOCKET_BUFFER_BYTES_ENV, "")
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_SOCKET_BUFFER_BYTES
    return value if value > 0 else DEFAULT_SOCKET_BUFFER_BYTES


def _inflight_window_frames() -> int:
    raw = os.environ.get(INFLIGHT_WINDOW_ENV, "")
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_INFLIGHT_WINDOW
    return value if value > 0 else DEFAULT_INFLIGHT_WINDOW


def _spray_streams() -> int:
    raw = os.environ.get(SPRAY_STREAMS_ENV, "")
    try:
        value = int(raw)
    except ValueError:
        return 1
    return value if value >= 1 else 1


def _spray_resteer_fraction() -> float:
    raw = os.environ.get(SPRAY_RESTEER_FRACTION_ENV, "")
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_SPRAY_RESTEER_FRACTION
    return value if 0 < value <= 1 else DEFAULT_SPRAY_RESTEER_FRACTION


def _spray_resteer_min_observations() -> int:
    raw = os.environ.get(SPRAY_RESTEER_MIN_OBSERVATIONS_ENV, "")
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_SPRAY_RESTEER_MIN_OBSERVATIONS
    return value if value >= 1 else DEFAULT_SPRAY_RESTEER_MIN_OBSERVATIONS


def _unsent_bytes(sock: socket.socket) -> int | None:
    """Bytes queued in the kernel send buffer of `sock`, or None when the
    platform cannot report them. Linux: TIOCOUTQ. macOS: SO_NWRITE."""
    try:
        if sys.platform == "darwin":
            # SO_NWRITE (0x1024 on Darwin) is not always exposed by the
            # socket module even though getsockopt supports it.
            return sock.getsockopt(
                socket.SOL_SOCKET, getattr(socket, "SO_NWRITE", 0x1024)
            )
        import fcntl

        # Linux TIOCOUTQ (== SIOCOUTQ).
        return struct.unpack(
            "i", fcntl.ioctl(sock.fileno(), 0x5411, struct.pack("i", 0))
        )[0]
    except (AttributeError, OSError):
        return None


def _dial_tls(
    tls: "TlsClientConfig", context: ssl.SSLContext
) -> tuple[ssl.SSLSocket, BinaryIO]:
    """Open one pinned mTLS connection to the receiver: TCP on the direct
    link with explicit buffer sizing, TLS 1.3, ALPN and cipher-suite verified
    fail-closed. Used for the primary connection and for every sprayed aux
    connection (identical policy per stream)."""
    raw = socket.create_connection(
        (tls.host, tls.port),
        timeout=tls.timeout_seconds,
        source_address=(tls.source_host, 0) if tls.source_host else None,
    )
    raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    # Explicit buffer sizing for the direct link; release qualification uses
    # measured installed-payload throughput rather than a link-class label.
    buffer_bytes = _socket_buffer_bytes()
    raw.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer_bytes)
    raw.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_bytes)
    raw.settimeout(tls.timeout_seconds)
    try:
        wrapped = context.wrap_socket(raw, server_hostname=tls.server_name)
    except BaseException:
        raw.close()
        raise
    if wrapped.selected_alpn_protocol() != "kvpack-handoff/1":
        wrapped.close()
        raise ProtocolError("receiver did not negotiate kvpack-handoff/1")
    cipher = wrapped.cipher()
    if cipher is None or cipher[0] not in ALLOWED_TLS13_SUITES:
        wrapped.close()
        raise ProtocolError(
            f"receiver negotiated unexpected TLS suite {cipher!r}; "
            "expected AES-256-GCM (preferred) or a TLS 1.3 fallback"
        )
    return wrapped, wrapped.makefile("rwb", buffering=0)


class _SprayRouter:
    """Round-robin plane assignment across sprayed streams with health-based
    resteering (WS9 TENT-lite).

    Plane `sequence` goes to `active[sequence % len(active)]`, assigned
    exactly once at send time — the receiver reassembles strictly by sequence,
    so resteering never affects correctness. Each stream keeps a sliding
    window of the last SPRAY_HISTORY_WINDOW (bytes, duration_ns) sends; when
    a stream's throughput stays below `fraction` x the best stream's for at
    least `min_observations` of its own sends it leaves the active set (the
    last active stream is never removed). Each stream also has its own
    _InflightWindow: the credit window is per-connection.
    """

    def __init__(
        self,
        connections: "list[tuple[socket.socket, BinaryIO]]",
        *,
        fraction: "float | None" = None,
        min_observations: "int | None" = None,
        window_frames: "int | None" = None,
    ) -> None:
        if not connections:
            raise ProtocolError("spray requires at least one stream")
        self._connections = list(connections)
        frames = _inflight_window_frames() if window_frames is None else window_frames
        self._windows = [
            _InflightWindow(sock, frames) for sock, _stream in self._connections
        ]
        self._active = list(range(len(self._connections)))
        self._history: "list[list[tuple[int, int]]]" = [
            [] for _ in self._connections
        ]
        self._fraction = (
            _spray_resteer_fraction() if fraction is None else fraction
        )
        self._min_observations = (
            _spray_resteer_min_observations()
            if min_observations is None
            else min_observations
        )

    def active_streams(self) -> list[int]:
        return list(self._active)

    def pick(self, sequence: int) -> "tuple[int, BinaryIO, _InflightWindow]":
        index = self._active[sequence % len(self._active)]
        return index, self._connections[index][1], self._windows[index]

    @staticmethod
    def _throughput(history: "list[tuple[int, int]]") -> float:
        return sum(item[0] for item in history) / max(
            sum(item[1] for item in history), 1
        )

    def record(self, index: int, byte_count: int, duration_ns: int) -> None:
        history = self._history[index]
        history.append((byte_count, max(duration_ns, 1)))
        del history[:-SPRAY_HISTORY_WINDOW]
        if len(history) < self._min_observations or len(self._active) <= 1:
            return
        if index not in self._active:
            return
        throughputs = [
            (stream, self._throughput(self._history[stream]))
            for stream in self._active
            if self._history[stream]
        ]
        best = max(rate for _stream, rate in throughputs)
        if self._throughput(history) < self._fraction * best:
            self._active.remove(index)
            print(
                f"[kvpack-spray] stream {index} resteered away "
                f"(throughput below {self._fraction:.2f}x best; "
                f"active streams now {self._active})"
            )

    def close_aux(self) -> None:
        """Close aux streams (index >= 1); stream 0 stays owned by the
        sender's normal close path."""
        for sock, stream in self._connections[1:]:
            close = getattr(stream, "close", None)
            if close is not None:
                try:
                    close()
                except OSError:
                    pass
            try:
                sock.close()
            except OSError:
                pass


class _InflightWindow:
    """Sender-side credit window over unacknowledged plane frames.

    kvpack-live-f16-le-v1 has no per-plane acknowledgement: frames are
    strictly ordered and the receiver speaks only the terminal ACK/ABORT.
    The honest sender-side bound therefore uses kernel send-queue occupancy
    as the receiver-consumption proxy: after `max_frames` plane frames are
    written without the queue draining below the low watermark, the
    producer blocks until the receiver has consumed enough bytes. This is
    an approximation of a permit protocol (a drained queue implies
    consumption, not verification); it deliberately changes nothing on the
    wire. When the platform cannot report queue occupancy the window
    degrades to the pre-WS9 behavior (kernel backpressure only).
    """

    def __init__(
        self,
        sock: socket.socket,
        max_frames: int,
        *,
        probe: "Callable[[socket.socket], int | None] | None" = None,
        watermark: int | None = None,
    ) -> None:
        self._sock = sock
        self._max_frames = max_frames
        self._probe = _unsent_bytes if probe is None else probe
        self._watermark = (
            max(_socket_buffer_bytes() // 4, 1) if watermark is None else watermark
        )
        self._outstanding = 0

    def before_frame(self) -> None:
        if self._outstanding < self._max_frames:
            return
        timeout = self._sock.gettimeout()
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            unsent = self._probe(self._sock)
            if unsent is None:
                # No occupancy probe on this platform: fall back to kernel
                # backpressure (documented degradation, window disabled).
                self._outstanding = 0
                return
            if unsent <= self._watermark:
                self._outstanding = 0
                return
            if deadline is not None and time.monotonic() > deadline:
                raise ProtocolError(
                    "in-flight window stalled: receiver stopped draining the link"
                )
            time.sleep(0.001)

    def after_frame(self) -> None:
        self._outstanding += 1


def _scheduled_classes(begin: dict[str, Any]) -> list[dict[str, Any]]:
    """Class traversal order for the begin's wire schedule. `layer-order`
    (or an absent schedule) is the declared table order; `decode-priority`
    streams windowed classes (the newest cuts) first, each group keeping
    its declared relative order. Mirrors BeginManifestV1's derivation in
    kvpack-handoff exactly; an unknown schedule fails closed."""
    table = begin["layout_table"]
    schedule = begin.get("schedule", WIRE_SCHEDULE_LAYER_ORDER)
    if schedule == WIRE_SCHEDULE_LAYER_ORDER:
        return list(table)
    if schedule == WIRE_SCHEDULE_DECODE_PRIORITY:
        windowed = [cls for cls in table if cls["window_tokens"] > 0]
        full = [cls for cls in table if cls["window_tokens"] == 0]
        return windowed + full
    raise ProtocolError(f"layout table declares an unknown wire schedule: {schedule!r}")


def _class_layers(cls: dict[str, Any]) -> list[int]:
    return [
        layer
        for layer in range(cls["from"], cls["until"], max(1, cls["step"]))
        if layer not in cls["except"]
    ]


def layout_walk(begin: dict[str, Any]) -> list[tuple[dict[str, Any], int, str, int, int]]:
    """Expected (class, layer, role, range_start, range_end) per sequence,
    walking a v2 begin's layout table in schedule order (declared order
    unless the begin arms `decode-priority`)."""
    cached = begin["cached_token_count"]
    walk: list[tuple[dict[str, Any], int, str, int, int]] = []
    for cls in _scheduled_classes(begin):
        for layer in _class_layers(cls):
            for role in cls["roles"]:
                window = (
                    cached
                    if cls["window_tokens"] == 0
                    else min(cls["window_tokens"], cached)
                )
                walk.append((cls, layer, role, cached - window, cached))
    return walk


class ProtocolError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise ProtocolError(f"cannot encode canonical JSON: {error}") from error


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def token_ids_sha256(token_ids: list[int]) -> str:
    digest = hashlib.sha256(b"kvpack-live-token-ids-v1\0")
    for token in token_ids:
        if not isinstance(token, int) or not 0 <= token <= 0xFFFFFFFF:
            raise ProtocolError("token IDs must be u32 values")
        digest.update(struct.pack("<I", token))
    return digest.hexdigest()


def descriptor_chain_sha256(headers: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256(b"kvpack-live-descriptor-chain-v1\0")
    for header in headers:
        digest.update(canonical_json(header))
        digest.update(b"\n")
    return digest.hexdigest()


def artifact_sha256(
    begin: dict[str, Any], headers: list[dict[str, Any]], core: dict[str, Any]
) -> str:
    digest = hashlib.sha256(b"kvpack-live-artifact-v1\0")
    digest.update(canonical_json(begin))
    digest.update(b"\n")
    for header in headers:
        digest.update(canonical_json(header))
        digest.update(b"\n")
    digest.update(canonical_json(core))
    return digest.hexdigest()


def _read_exact(stream: BinaryIO, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise ProtocolError("unexpected EOF in live handoff frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _write_all(stream: BinaryIO, data: bytes) -> None:
    # Raw SocketIO.write() is a single send() and may accept fewer bytes than
    # requested for very large buffers; loop until everything is on the wire.
    view = memoryview(data)
    while view:
        written = stream.write(view)
        if written is None or written <= 0:
            raise ProtocolError("live handoff stream write made no progress")
        view = view[written:]


def write_frame(
    stream: BinaryIO,
    kind: int,
    manifest: dict[str, Any],
    payload: bytes = b"",
    *,
    max_payload_bytes: int,
) -> None:
    encoded = canonical_json(manifest)
    if not 0 < len(encoded) <= MAX_JSON_BYTES:
        raise ProtocolError("frame JSON exceeds the protocol bound")
    if len(payload) > max_payload_bytes:
        raise ProtocolError("frame payload exceeds the configured bound")
    if kind != FRAME_LAYER and payload:
        raise ProtocolError("only layer frames may contain a payload")
    _write_all(
        stream,
        FRAME_HEADER.pack(
            FRAME_MAGIC,
            SCHEMA_VERSION,
            kind,
            0,
            len(encoded),
            len(payload),
        ),
    )
    _write_all(stream, encoded)
    _write_all(stream, payload)
    stream.flush()


def read_frame(
    stream: BinaryIO, *, max_payload_bytes: int
) -> tuple[int, dict[str, Any], bytes]:
    raw = _read_exact(stream, FRAME_HEADER.size)
    magic, version, kind, flags, json_length, payload_length = FRAME_HEADER.unpack(raw)
    if magic != FRAME_MAGIC or version != SCHEMA_VERSION or flags != 0:
        raise ProtocolError("invalid frame magic, version, or reserved flags")
    if kind not in {FRAME_BEGIN, FRAME_LAYER, FRAME_SEAL, FRAME_ABORT, FRAME_ACK}:
        raise ProtocolError(f"unknown frame kind {kind}")
    if not 0 < json_length <= MAX_JSON_BYTES or payload_length > max_payload_bytes:
        raise ProtocolError("frame lengths exceed configured bounds")
    if kind != FRAME_LAYER and payload_length:
        raise ProtocolError("non-layer frame declared a payload")
    encoded = _read_exact(stream, json_length)
    try:
        manifest = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError(f"invalid frame JSON: {error}") from error
    if canonical_json(manifest) != encoded:
        raise ProtocolError("frame JSON is not canonical")
    payload = _read_exact(stream, payload_length)
    return kind, manifest, payload


@dataclass(frozen=True)
class TlsClientConfig:
    ca_cert: str
    client_cert: str
    client_key: str
    host: str
    port: int
    server_name: str
    timeout_seconds: float
    source_host: str | None = None


class LiveHandoffSender:
    def __init__(
        self,
        tls: TlsClientConfig,
        begin: dict[str, Any],
        *,
        max_payload_bytes: int,
    ) -> None:
        self._begin = begin
        self._headers: list[dict[str, Any]] = []
        self._max_payload_bytes = max_payload_bytes
        self._payload_bytes = 0
        self._payload_hash = hashlib.sha256()
        if begin.get("layout_table"):
            if begin.get("portable_abi") != PORTABLE_ABI_V2:
                raise ProtocolError("a v2 layout table requires the canonical-kv-v2 ABI")
            self._layout_walk: list[tuple[dict[str, Any], int, str, int, int]] | None = (
                layout_walk(begin)
            )
            if len(self._layout_walk) != begin["expected_layer_frames"]:
                raise ProtocolError(
                    "layout table does not cover the declared layer-frame count"
                )
        else:
            self._layout_walk = None
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=tls.ca_cert)
        context.minimum_version = ssl.TLSVersion.TLSv1_3
        context.maximum_version = ssl.TLSVersion.TLSv1_3
        # Only affects TLS <= 1.2 (kept as defense in depth should the floor
        # ever drop); TLS 1.3 suite preference is receiver-side and is
        # verified against ALLOWED_TLS13_SUITES after the handshake below.
        context.set_ciphers("ECDHE+AESGCM:ECDHE+CHACHA20")
        context.load_cert_chain(certfile=tls.client_cert, keyfile=tls.client_key)
        context.set_alpn_protocols(["kvpack-handoff/1"])
        self._socket, self._stream = _dial_tls(tls, context)
        self._window = _InflightWindow(self._socket, _inflight_window_frames())
        try:
            write_frame(
                self._stream,
                FRAME_BEGIN,
                begin,
                max_payload_bytes=max_payload_bytes,
            )
        except BaseException:
            self.close()
            raise
        # WS9 TENT-lite spraying: after BEGIN on stream 0, open streams-1
        # identically pinned aux mTLS connections that carry only layer
        # frames. The receiver must run KVPACK_HANDOFF_STREAMS with the same
        # value. Off by default: no router attribute, single-stream behavior
        # is byte-identical to the pre-spray producer.
        streams = _spray_streams()
        if streams > 1:
            try:
                connections = [(self._socket, self._stream)] + [
                    _dial_tls(tls, context) for _ in range(streams - 1)
                ]
                self._spray_router = _SprayRouter(connections)
            except BaseException:
                self.close()
                raise
            print(f"[kvpack-spray] {streams} streams active ({SPRAY_STREAMS_ENV}={streams})")

    def send_plane(self, layer: int, role: str, payload: bytes) -> None:
        sequence = len(self._headers)
        if self._layout_walk is None:
            expected_layer = sequence // 2
            expected_role = "key" if sequence % 2 == 0 else "value"
            if layer != expected_layer or role != expected_role:
                raise ProtocolError(
                    f"out-of-order layer plane: got {layer}/{role}, "
                    f"expected {expected_layer}/{expected_role}"
                )
            start = 0
            end = self._begin["cached_token_count"]
            kvh = self._begin["geometry"]["num_kv_heads"]
            hd = self._begin["geometry"]["head_dim"]
            dtype_tag = None
            class_tag = None
        else:
            if sequence >= len(self._layout_walk):
                raise ProtocolError("layer plane exceeds the declared layout table")
            cls, expected_layer, expected_role, start, end = self._layout_walk[sequence]
            if layer != expected_layer or role != expected_role:
                raise ProtocolError(
                    f"out-of-order layer plane: got {layer}/{role}, expected "
                    f"{expected_layer}/{expected_role} (class {cls['class']!r})"
                )
            kvh = cls["kv_heads"]
            hd = cls["head_dim"]
            dtype_tag = cls["dtype"]
            class_tag = cls["class"]
        expected_length = (end - start) * kvh * hd * 2
        if len(payload) != expected_length:
            raise ProtocolError(
                f"layer payload has {len(payload)} bytes, expected {expected_length}"
            )
        hash_started_ns = time.perf_counter_ns()
        payload_sha256 = sha256_hex(payload)
        hash_ns = time.perf_counter_ns() - hash_started_ns
        dump_dir = os.environ.get("KVPACK_PLANE_OUT")
        if dump_dir:
            # Diagnostic plane capture (F24): persist the exact shipped
            # bytes for offline replay. Only active when explicitly armed.
            suffix = "k" if role == "key" else "v"
            with open(
                os.path.join(dump_dir, f"layer-{layer:02d}-{suffix}.f16"), "wb"
            ) as dump_handle:
                dump_handle.write(payload)
        header = {
            "byte_length": len(payload),
            "layer": layer,
            "logical_token_end": end,
            "logical_token_start": start,
            "role": role,
            "schema_version": SCHEMA_VERSION,
            "sequence": sequence,
            "sha256": payload_sha256,
            "shape": [end - start, kvh, hd],
            "transfer_id": self._begin["transfer_id"],
        }
        if dtype_tag is not None:
            header["dtype"] = dtype_tag
        if class_tag is not None:
            header["layout_class"] = class_tag
        router = getattr(self, "_spray_router", None)
        if router is None:
            window = getattr(self, "_window", None)
            if window is None:
                # Hand-wired test harnesses bypass __init__; keep the default
                # window policy rather than disabling it.
                window = _InflightWindow(self._socket, _inflight_window_frames())
                self._window = window
            stream_index = None
            stream = self._stream
        else:
            stream_index, stream, window = router.pick(sequence)
        window.before_frame()
        write_started_ns = time.perf_counter_ns()
        write_frame(
            stream,
            FRAME_LAYER,
            header,
            payload,
            max_payload_bytes=self._max_payload_bytes,
        )
        window.after_frame()
        write_ns = time.perf_counter_ns() - write_started_ns
        if router is not None and stream_index is not None:
            router.record(stream_index, len(payload), write_ns)
        self._headers.append(header)
        self._payload_hash.update(payload)
        self._payload_bytes += len(payload)
        return {"hash_ns": hash_ns, "write_ns": write_ns}

    def seal(self, prompt_token_ids: list[int]) -> str:
        if len(self._headers) != self._begin["expected_layer_frames"]:
            raise ProtocolError("cannot seal before every ordered layer plane is sent")
        if self._payload_bytes != self._begin["expected_payload_bytes"]:
            raise ProtocolError("payload byte count does not match the begin manifest")
        now_ms = time.time_ns() // 1_000_000
        core = {
            "completed_unix_ms": now_ms,
            "descriptor_chain_sha256": descriptor_chain_sha256(self._headers),
            "frame_count": len(self._headers),
            "payload_bytes": self._payload_bytes,
            "payload_sha256": self._payload_hash.hexdigest(),
            "prompt_token_ids": prompt_token_ids,
            "protocol": PROTOCOL,
            "schema_version": SCHEMA_VERSION,
            "strategy": "consumer_last_prompt_token",
            "token_ids_sha256": token_ids_sha256(prompt_token_ids),
            "transfer_id": self._begin["transfer_id"],
        }
        seal = dict(core)
        seal["artifact_sha256"] = artifact_sha256(self._begin, self._headers, core)
        write_frame(
            self._stream,
            FRAME_SEAL,
            seal,
            max_payload_bytes=self._max_payload_bytes,
        )
        kind, ack, payload = read_frame(
            self._stream, max_payload_bytes=self._max_payload_bytes
        )
        if kind == FRAME_ABORT:
            raise ProtocolError(f"receiver rejected handoff with code {ack.get('code')!r}")
        expected_ack = {
            "artifact_sha256": seal["artifact_sha256"],
            "protocol": PROTOCOL,
            "schema_version": SCHEMA_VERSION,
            "status": "committed",
            "transfer_id": self._begin["transfer_id"],
        }
        if kind != FRAME_ACK or payload or ack != expected_ack:
            raise ProtocolError("receiver acknowledgement did not match the terminal seal")
        return seal["artifact_sha256"]

    def abort(self, code: str = "producer_failed") -> None:
        try:
            manifest = {
                "code": code,
                "protocol": PROTOCOL,
                "schema_version": SCHEMA_VERSION,
                "transfer_id": self._begin["transfer_id"],
            }
            write_frame(
                self._stream,
                FRAME_ABORT,
                manifest,
                max_payload_bytes=self._max_payload_bytes,
            )
        except BaseException:
            pass

    def close(self) -> None:
        router = getattr(self, "_spray_router", None)
        if router is not None:
            router.close_aux()
        stream = getattr(self, "_stream", None)
        sock = getattr(self, "_socket", None)
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
