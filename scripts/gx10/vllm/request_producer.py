#!/usr/bin/env python3
"""Submit one closed-schema token fixture to the resident Spark producer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import time
from pathlib import Path

SCHEMA = "muser.spark-nvfp4-prefill-request.v1"


def read_tokens(path: Path) -> list[int]:
    tokens: list[int] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        value = line.strip()
        if not value:
            continue
        try:
            token = int(value, 10)
        except ValueError as error:
            raise ValueError(f"invalid token at line {line_number}") from error
        if not 0 <= token < 202048:
            raise ValueError(f"out-of-vocabulary token at line {line_number}")
        tokens.append(token)
    if len(tokens) < 2:
        raise ValueError("fixture must contain at least two tokens")
    return tokens


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def write_exclusive(path: Path, value: object) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sock", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--generation", required=True, type=int)
    parser.add_argument("--transfer-id", required=True)
    parser.add_argument("--receiver-host", required=True)
    parser.add_argument("--receiver-port", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dflash-session")
    parser.add_argument("--dflash-identity-sha256")
    parser.add_argument("--dflash-kv-heads", type=int)
    parser.add_argument("--dflash-head-dim", type=int)
    parser.add_argument("--dflash-context-layers", type=int)
    parser.add_argument("--dflash-context-elements-per-token", type=int)
    parser.add_argument("--dflash-context-sink-size", type=int)
    parser.add_argument("--dflash-context-window-size", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument(
        "--prefix-cut",
        type=int,
        default=0,
        help="256-aligned held-prefix cut; the handoff then carries only the suffix",
    )
    args = parser.parse_args()
    tokens = read_tokens(Path(args.tokens))
    request = {
        "schema": SCHEMA,
        "request_id": args.request_id,
        "token_ids": tokens,
        "handoff": {
            "generation": args.generation,
            "receiver_host": args.receiver_host,
            "receiver_port": args.receiver_port,
            "transfer_id": args.transfer_id,
        },
    }
    if bool(args.dflash_session) != bool(args.dflash_identity_sha256):
        parser.error("--dflash-session and --dflash-identity-sha256 are a pair")
    if args.prefix_cut:
        if args.prefix_cut % 256 != 0 or args.prefix_cut >= len(tokens) - 1:
            parser.error("--prefix-cut must be 256-aligned and leave a nonempty suffix")
        request["handoff"]["prefix_cut"] = args.prefix_cut
    if args.dflash_session:
        geometry = (
            args.dflash_kv_heads,
            args.dflash_head_dim,
            args.dflash_context_layers,
            args.dflash_context_elements_per_token,
            args.dflash_context_sink_size,
            args.dflash_context_window_size,
        )
        if any(value is None or value < 1 for value in geometry):
            parser.error("all DFlash context geometry fields must be positive")
        if (
            args.dflash_kv_heads * args.dflash_head_dim
            != args.dflash_context_elements_per_token
        ):
            parser.error("DFlash context width differs from KV geometry")
        request["handoff"].update(
            {
                "dflash_session": args.dflash_session,
                "dflash_identity_sha256": args.dflash_identity_sha256,
                "dflash_kv_heads": args.dflash_kv_heads,
                "dflash_head_dim": args.dflash_head_dim,
                "dflash_context_layers": args.dflash_context_layers,
                "dflash_context_elements_per_token": args.dflash_context_elements_per_token,
                "dflash_context_sink_size": args.dflash_context_sink_size,
                "dflash_context_window_size": args.dflash_context_window_size,
            }
        )
    payload = canonical(request) + b"\n"
    started = time.perf_counter_ns()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(args.timeout_seconds)
        client.connect(args.sock)
        client.sendall(payload)
        # Keep both directions open while the resident engine works. The
        # resident watches this socket to cancel abandoned requests; a
        # write-half close is indistinguishable from a vanished client on
        # that side of a Unix stream even though this client still intends
        # to read the response.
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = client.recv(65536)
            if not chunk:
                break
            size += len(chunk)
            if size > 8 * 1024 * 1024:
                raise RuntimeError("producer response exceeded 8 MiB")
            chunks.append(chunk)
    response = json.loads(b"".join(chunks))
    if not isinstance(response, dict) or response.get("status") != "ok":
        raise RuntimeError(f"producer request failed: {response!r}")
    receipt = {
        "schema": "muser.spark-nvfp4-prefill-client.v1",
        "created_unix_ms": time.time_ns() // 1_000_000,
        "client_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "request_sha256": hashlib.sha256(canonical(request)).hexdigest(),
        "token_fixture": str(Path(args.tokens).resolve()),
        "token_fixture_sha256": hashlib.sha256(
            Path(args.tokens).read_bytes()
        ).hexdigest(),
        "token_count": len(tokens),
        "total_ns": time.perf_counter_ns() - started,
        "response": response,
    }
    write_exclusive(Path(args.output), receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
