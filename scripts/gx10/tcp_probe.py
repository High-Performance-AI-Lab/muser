#!/usr/bin/env python3
"""Raw TCP throughput probe for the GX10<->Mac wired fabric path.

Purpose
-------
Answer exactly one question, fast: is the wire healthy? The disaggregated
lane's product transport adds TLS, framing, per-segment verification, and
kernel pacing on top of raw TCP; when the product path looks slow, this probe
establishes the raw ceiling so you know whether to debug the network or the
software above it. Run it over the exact wired path your lane uses (never
Wi-Fi), and re-establish the raw reference after any topology change.

Usage
-----
Mac (receiver):     python3 scripts/gx10/tcp_probe.py server 29599
Node (sender):      python3 /tmp/tcp_probe.py client 192.0.2.10 29599 5 1
                    # args: host port seconds streams
Reverse direction:  swap the roles. Diagnose both ways; the paths differ.

Interpreting
------------
- ~9+ Gbps single stream: the link is fine; look at the product path
  (pacing pin, ledger placement, producer health) before touching the NIC.
- ~3-6 Gbps single stream: check MTU, socket buffers, and on the GB10 the
  ConnectX-7 driver (the pre-580.142 throttle caps ~13 Gbps on the 200GbE
  ports). Parallel streams (--streams 4) separating cleanly means a
  per-stream limit, not a link limit.
- Bimodal stalls of ~1 s under load are NOT a raw-link symptom; that pattern
  in the product lane points at the receiver's durable reserve (see
  durable_fsync_probe.py).

No GPU, no model, no secrets. Pure TCP with a zeroed payload; safe to run any
time. The server exits on SIGINT; clients are bounded by --seconds.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time


def serve(port: int) -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("0.0.0.0", port))
    listener.listen(64)
    print(f"tcp_probe: listening on :{port} (Ctrl-C to stop)", flush=True)

    def drain(conn: socket.socket, peer: str) -> None:
        total = 0
        start = time.perf_counter()
        try:
            while True:
                chunk = conn.recv(1 << 20)
                if not chunk:
                    break
                total += len(chunk)
        finally:
            conn.close()
        elapsed = time.perf_counter() - start
        if elapsed > 0 and total > 0:
            rate = 8 * total / elapsed / 1e9
            print(
                f"tcp_probe: {peer} sent {total / 1e9:.3f} GB in {elapsed:.2f}s "
                f"= {rate:.3f} Gbps",
                flush=True,
            )

    try:
        while True:
            conn, (host, _) = listener.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            threading.Thread(target=drain, args=(conn, host), daemon=True).start()
    except KeyboardInterrupt:
        pass


def one_stream(host: str, port: int, seconds: float, results: list[int], index: int) -> None:
    sock = socket.create_connection((host, port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    buf = b"\0" * (1 << 20)
    total = 0
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        total += sock.send(buf)
    sock.close()
    results[index] = total


def client(host: str, port: int, seconds: float, streams: int, as_json: bool) -> None:
    results = [0] * streams
    threads = [
        threading.Thread(target=one_stream, args=(host, port, seconds, results, i))
        for i in range(streams)
    ]
    start = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.perf_counter() - start
    per_stream = [round(8 * b / elapsed / 1e9, 3) for b in results]
    total_gbps = round(8 * sum(results) / elapsed / 1e9, 3)
    if as_json:
        print(
            json.dumps(
                {
                    "schema": "muser.tcp-probe.v1",
                    "host": host,
                    "port": port,
                    "seconds": seconds,
                    "streams": streams,
                    "per_stream_gbps": per_stream,
                    "total_gbps": total_gbps,
                },
                sort_keys=True,
            )
        )
    else:
        print(
            f"streams={streams} total={total_gbps:.3f} Gbps "
            f"(per-stream: {' '.join(str(v) for v in per_stream)})",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    server_parser = sub.add_parser("server", help="receive and report")
    server_parser.add_argument("port", type=int)
    client_parser = sub.add_parser("client", help="send zeros for N seconds")
    client_parser.add_argument("host")
    client_parser.add_argument("port", type=int)
    client_parser.add_argument("seconds", type=float)
    client_parser.add_argument("streams", type=int)
    client_parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()
    if args.command == "server":
        serve(args.port)
    else:
        if args.seconds <= 0 or args.streams <= 0:
            parser.error("seconds and streams must be positive")
        client(args.host, args.port, args.seconds, args.streams, args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
