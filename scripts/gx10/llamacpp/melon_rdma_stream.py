#!/usr/bin/env python3
"""ctypes binding + socket-like wrapper over native/melon_rdma/melon_rdma_pipe.c.

Gives `muser_v2_send.py` an RDMA-backed replacement for the plain TCP socket
`connect_tls()` builds today, with the same `.recv()`/`.sendall()` surface a
Python `ssl.MemoryBIO`-terminated TLS session needs to pump ciphertext
through. Bootstraps the RC QP over an already-connected TCP socket (qpn/psn/
gid exchange), then all further I/O goes over RDMA SEND/RECV.

Standalone self-test:
  Spark (listener/server role):
    python3 melon_rdma_stream.py --listen 0.0.0.0:29123 --dev rocep1s0f1 --gid 2
  Mac (connects out/client role):
    python3 melon_rdma_stream.py --connect <spark-ip>:29123 --dev mlx5_0 --gid 0 --send-test
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import os
import socket
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_NATIVE_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "native", "melon_rdma"))


def _library_name() -> str:
    return "libmelon_rdma_pipe.dylib" if sys.platform == "darwin" else "libmelon_rdma_pipe.so"


def _load_library() -> ctypes.CDLL:
    path = os.environ.get("MELON_RDMA_PIPE_LIB", os.path.join(_NATIVE_DIR, _library_name()))
    lib = ctypes.CDLL(path)
    lib.melon_rdma_pipe_open.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.melon_rdma_pipe_open.restype = ctypes.c_void_p
    lib.melon_rdma_pipe_send.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.melon_rdma_pipe_send.restype = ctypes.c_int
    lib.melon_rdma_pipe_recv.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.melon_rdma_pipe_recv.restype = ctypes.c_ssize_t
    lib.melon_rdma_pipe_close.argtypes = [ctypes.c_void_p]
    lib.melon_rdma_pipe_last_error.argtypes = []
    lib.melon_rdma_pipe_last_error.restype = ctypes.c_char_p
    return lib


class MelonRdmaError(RuntimeError):
    pass


class MelonRdmaStream:
    """Socket-like (.recv/.sendall/.close) wrapper over the RDMA byte-pipe.

    `bootstrap_sock` must already be TCP-connected (client) or accepted
    (server); this takes ownership of its underlying fd. Everything after
    construction goes over RDMA — the bootstrap socket is only used, once,
    inside the C layer to exchange qpn/psn/gid.
    """

    def __init__(self, bootstrap_sock: socket.socket, dev: str, gid_index: int):
        self._lib = _load_library()
        # socket.create_connection(..., timeout=...) (and any prior
        # .settimeout() call) puts the underlying OS fd into non-blocking
        # mode — Python emulates the blocking-with-timeout behavior itself
        # via select(). The C layer does plain blocking recv()/send(), so
        # the fd handed to it must be put back into real blocking mode
        # first, or its first read races the peer and fails with EAGAIN.
        bootstrap_sock.setblocking(True)
        fd = os.dup(bootstrap_sock.fileno())
        bootstrap_sock.detach()
        handle = self._lib.melon_rdma_pipe_open(fd, dev.encode("ascii"), gid_index)
        if not handle:
            error = self._lib.melon_rdma_pipe_last_error().decode("utf-8", "replace")
            # melon_rdma_pipe_open() already closed `fd` itself (its failure
            # path runs melon_rdma_pipe_close()) — closing it again here
            # would double-close and mask the real error behind an EBADF.
            raise MelonRdmaError(f"melon_rdma_pipe_open({dev!r}, gid_index={gid_index}) failed: {error}")
        self._handle = handle

    def recv(self, n: int) -> bytes:
        buf = ctypes.create_string_buffer(n)
        got = self._lib.melon_rdma_pipe_recv(self._handle, buf, n)
        if got < 0:
            error = self._lib.melon_rdma_pipe_last_error().decode("utf-8", "replace")
            raise MelonRdmaError(f"melon_rdma_pipe_recv failed: {error}")
        return buf.raw[:got]

    def sendall(self, data: bytes) -> None:
        rc = self._lib.melon_rdma_pipe_send(self._handle, data, len(data))
        if rc != 0:
            error = self._lib.melon_rdma_pipe_last_error().decode("utf-8", "replace")
            raise MelonRdmaError(f"melon_rdma_pipe_send failed: {error}")

    def close(self) -> None:
        if self._handle:
            self._lib.melon_rdma_pipe_close(self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class MelonRdmaTlsStream:
    """TLS 1.3 (client role) terminated over an `ssl.MemoryBIO` pair, with
    ciphertext pumped through a `MelonRdmaStream` instead of a real socket —
    so `ssl.SSLContext.wrap_bio()` never touches an actual OS socket. Exposes
    the same `.recv()`/`.sendall()`/`.getpeercert()`/`.selected_alpn_protocol()`
    /`.close()` surface as the plain `ssl.SSLSocket` `connect_tls()` returns
    today, so call sites in `muser_v2_send.py` do not need to change.
    """

    def __init__(self, rdma_stream: MelonRdmaStream, context: "ssl.SSLContext", server_hostname: str):
        import ssl as _ssl

        self._rdma = rdma_stream
        self._incoming = _ssl.MemoryBIO()
        self._outgoing = _ssl.MemoryBIO()
        self._ssl_obj = context.wrap_bio(
            self._incoming, self._outgoing, server_hostname=server_hostname
        )
        self._SSLWantReadError = _ssl.SSLWantReadError
        self._SSLWantWriteError = _ssl.SSLWantWriteError
        self._do_handshake()

    def _pump_out(self) -> None:
        data = self._outgoing.read()
        if data:
            self._rdma.sendall(data)

    def _fill_in(self, want: int = 65536) -> None:
        data = self._rdma.recv(want)
        if not data:
            self._incoming.write_eof()
        else:
            self._incoming.write(data)

    def _do_handshake(self) -> None:
        while True:
            try:
                self._ssl_obj.do_handshake()
                self._pump_out()
                return
            except self._SSLWantReadError:
                self._pump_out()
                self._fill_in()
            except self._SSLWantWriteError:
                self._pump_out()

    def recv(self, n: int) -> bytes:
        while True:
            try:
                return self._ssl_obj.read(n)
            except self._SSLWantReadError:
                self._fill_in()
            except self._SSLWantWriteError:
                self._pump_out()

    def sendall(self, data: bytes) -> None:
        sent = 0
        while sent < len(data):
            try:
                n = self._ssl_obj.write(data[sent:])
                sent += n
                self._pump_out()
            except self._SSLWantReadError:
                self._fill_in()
            except self._SSLWantWriteError:
                self._pump_out()

    def getpeercert(self, binary_form: bool = False):
        return self._ssl_obj.getpeercert(binary_form=binary_form)

    def selected_alpn_protocol(self):
        return self._ssl_obj.selected_alpn_protocol()

    def shutdown(self, _how: int) -> None:
        # Best-effort graceful TLS close_notify. The RDMA byte-pipe has no
        # native half-close to map `how` (SHUT_RDWR etc.) onto, and callers
        # in this codebase already wrap shutdown() in try/except and ignore
        # failures — matching a real ssl.SSLSocket.shutdown()'s own
        # unreliability once the peer may already be gone.
        try:
            self._ssl_obj.unwrap()
        except Exception:
            pass

    def close(self) -> None:
        self._rdma.close()


def _parse_host_port(value: str) -> tuple[str, int]:
    host, _, port = value.rpartition(":")
    return host or "0.0.0.0", int(port)


def _recv_exact(stream: "MelonRdmaStream", n: int) -> bytes:
    out = bytearray()
    while len(out) < n:
        chunk = stream.recv(n - len(out))
        if not chunk:
            raise MelonRdmaError("peer closed before delivering the expected bytes")
        out.extend(chunk)
    return bytes(out)


# Phased, one-directional-at-a-time protocol (no simultaneous bidirectional
# traffic): client sends the full payload then its digest; only once that is
# fully drained does the server send back a 1-byte ack. Overlapping sends in
# both directions at once will starve the small, fixed-depth RX ring on
# whichever side is not yet draining it — a test-protocol hazard, not a
# byte-pipe correctness question.
_DIGEST_BYTES = hashlib.sha256().digest_size


def _self_test_server(listen: str, dev: str, gid: int) -> None:
    host, port = _parse_host_port(listen)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, port))
    listener.listen(1)
    print(f"melon_rdma_stream: listening on {host}:{port} for the bootstrap connection", flush=True)
    raw, addr = listener.accept()
    print(f"melon_rdma_stream: bootstrap accepted from {addr}", flush=True)
    stream = MelonRdmaStream(raw, dev, gid)
    print("melon_rdma_stream: RDMA pipe activated (server side)", flush=True)

    length_bytes = _recv_exact(stream, 8)
    payload_len = int.from_bytes(length_bytes, "big")
    print(f"melon_rdma_stream: server expecting {payload_len} bytes", flush=True)

    digest = hashlib.sha256()
    remaining = payload_len
    started = time.monotonic()
    while remaining > 0:
        chunk = stream.recv(min(remaining, 1 << 20))
        if not chunk:
            raise MelonRdmaError("peer closed mid-payload")
        digest.update(chunk)
        remaining -= len(chunk)
    elapsed = time.monotonic() - started
    client_digest = _recv_exact(stream, _DIGEST_BYTES)
    ok = digest.digest() == client_digest
    gbit_s = (payload_len * 8 / elapsed / 1e9) if elapsed > 0 else float("inf")
    print(
        f"melon_rdma_stream: server received {payload_len} bytes in {elapsed:.3f}s "
        f"({gbit_s:.3f} Gbit/s) digest_match={ok}",
        flush=True,
    )
    stream.sendall(b"\x01" if ok else b"\x00")
    stream.close()
    if not ok:
        raise MelonRdmaError("received payload digest did not match — byte-pipe correctness FAILED")
    print("MELON_RDMA_SELFTEST PASS", flush=True)


def _self_test_client(connect: str, dev: str, gid: int, payload_bytes: int) -> None:
    host, port = _parse_host_port(connect)
    raw = socket.create_connection((host, port), timeout=10)
    stream = MelonRdmaStream(raw, dev, gid)
    print("melon_rdma_stream: RDMA pipe activated (client side)", flush=True)
    payload = os.urandom(payload_bytes)
    digest = hashlib.sha256(payload).digest()

    started = time.monotonic()
    stream.sendall(len(payload).to_bytes(8, "big"))
    stream.sendall(payload)
    stream.sendall(digest)
    ack = _recv_exact(stream, 1)
    elapsed = time.monotonic() - started
    ok = ack == b"\x01"
    gbit_s = (len(payload) * 8 / elapsed / 1e9) if elapsed > 0 else float("inf")
    print(
        f"melon_rdma_stream: client sent {len(payload)} bytes in {elapsed:.3f}s "
        f"({gbit_s:.3f} Gbit/s) server_ack_ok={ok}",
        flush=True,
    )
    stream.close()
    if not ok:
        raise MelonRdmaError("server reported a digest mismatch — byte-pipe correctness FAILED")
    print("MELON_RDMA_SELFTEST PASS", flush=True)


def _tls_client_test(
    connect: str, dev: str, gid: int, cert_dir: str, alpn: str, server_hostname: str
) -> None:
    import ssl as _ssl

    host, port = _parse_host_port(connect)
    raw = socket.create_connection((host, port), timeout=10)
    rdma_stream = MelonRdmaStream(raw, dev, gid)
    print("melon_rdma_stream: RDMA pipe activated (TLS client)", flush=True)

    context = _ssl.SSLContext(_ssl.PROTOCOL_TLS_CLIENT)
    context.minimum_version = _ssl.TLSVersion.TLSv1_3
    context.maximum_version = _ssl.TLSVersion.TLSv1_3
    context.load_verify_locations(cafile=os.path.join(cert_dir, "ca.cert.pem"))
    context.load_cert_chain(
        os.path.join(cert_dir, "client.cert.pem"), os.path.join(cert_dir, "client.key.pem")
    )
    context.set_alpn_protocols([alpn])

    tls = MelonRdmaTlsStream(rdma_stream, context, server_hostname)
    print(
        f"melon_rdma_stream: TLS handshake complete, alpn={tls.selected_alpn_protocol()!r}",
        flush=True,
    )
    tls.sendall(b"hello over rdma-tls")
    reply = tls.recv(64)
    print(f"melon_rdma_stream: server replied {reply!r}", flush=True)
    tls.close()
    if reply != b"ack":
        raise MelonRdmaError(f"unexpected server reply: {reply!r}")
    print("MELON_RDMA_TLS_SELFTEST PASS", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", help="host:port to listen on (server role)")
    parser.add_argument("--connect", help="host:port to connect to (client role)")
    parser.add_argument("--dev", required=True, help="RDMA device name, e.g. rocep1s0f1 or mlx5_0")
    parser.add_argument("--gid", type=int, required=True, help="local GID table index")
    parser.add_argument("--payload-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument(
        "--tls-cert-dir", help="run the TLS-over-RDMA client self-test using certs from this dir"
    )
    parser.add_argument("--tls-alpn", default="melon-rdma-tls-selftest-v1")
    parser.add_argument("--tls-server-hostname", default="melon-rdma-test-server")
    args = parser.parse_args()
    if bool(args.listen) == bool(args.connect):
        parser.error("pass exactly one of --listen or --connect")
    if args.listen:
        _self_test_server(args.listen, args.dev, args.gid)
    elif args.tls_cert_dir:
        _tls_client_test(
            args.connect, args.dev, args.gid, args.tls_cert_dir, args.tls_alpn, args.tls_server_hostname
        )
    else:
        _self_test_client(args.connect, args.dev, args.gid, args.payload_bytes)


if __name__ == "__main__":
    main()
