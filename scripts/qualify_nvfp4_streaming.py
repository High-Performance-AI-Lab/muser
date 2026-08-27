#!/usr/bin/env python3
"""Qualify one real remote NVFP4 SSE request and one resident prefix hit."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
from pathlib import Path
import queue
import subprocess
import threading
import time
from typing import Any


def distribution(values: list[int]) -> dict[str, Any]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    ordered = sorted(values)
    return {
        "raw_ns": values,
        "median_ns": ordered[len(ordered) // 2],
        "cv": math.sqrt(variance) / mean if mean else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dflash", type=Path)
    parser.add_argument("--cluster-config", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=4963)
    parser.add_argument("--resident", required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--remote-tokens", default="/workspace/seam-33.tokens")
    parser.add_argument("--remote-output", required=True)
    parser.add_argument(
        "--remote-request-script",
        default="/workspace/scripts/gx10/vllm/request_producer.py",
    )
    parser.add_argument("--remote-sock", default="/service/producer.sock")
    parser.add_argument("--dflash-session", default="/dflash/dflash.session")
    parser.add_argument("--dflash-identity-sha256")
    parser.add_argument("--cache-hit-repetitions", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--spark-host", required=True)
    parser.add_argument("--receiver-host", required=True)
    return parser.parse_args()


def request(
    port: int,
    method: str,
    path: str,
    body: bytes = b"",
    timeout: float = 10,
    headers: dict[str, str] | None = None,
) -> tuple[int, bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    request_headers = dict(headers or {})
    if body:
        request_headers.setdefault("Content-Type", "application/json")
    connection.request(method, path, body=body, headers=request_headers)
    response = connection.getresponse()
    payload = response.read()
    status = response.status
    connection.close()
    return status, payload


def wait_ready(port: int, process: subprocess.Popen[str], timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"server exited during startup with {code}")
        try:
            status, _ = request(port, "GET", "/healthz", timeout=2)
            if status == 200:
                return
        except OSError:
            pass
        time.sleep(0.1)
    raise RuntimeError("server did not become ready")


def stream_chat(
    port: int, payload: dict[str, Any], timeout: int, api_key: str
) -> dict[str, Any]:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    started = time.monotonic_ns()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    connection.request(
        "POST",
        "/v1/chat/completions",
        body=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {api_key}",
        },
    )
    response = connection.getresponse()
    if response.status != 200:
        failure = response.read()
        connection.close()
        raise RuntimeError(f"stream returned HTTP {response.status}: {failure[:8192]!r}")
    pieces: list[str] = []
    event_count = 0
    usage: object = None
    first_content_ns: int | None = None
    saw_role = False
    saw_terminal = False
    saw_done = False
    while line := response.readline():
        line = line.strip()
        if not line or not line.startswith(b"data: "):
            continue
        data = line[len(b"data: ") :]
        if data == b"[DONE]":
            saw_done = True
            break
        event_count += 1
        event = json.loads(data)
        choices = event.get("choices", [])
        if not choices:
            usage = event.get("usage")
            continue
        choice = choices[0]
        delta = choice.get("delta", {})
        saw_role |= delta.get("role") == "assistant"
        content = delta.get("content")
        if isinstance(content, str) and content:
            if first_content_ns is None:
                first_content_ns = time.monotonic_ns()
            pieces.append(content)
        saw_terminal |= choice.get("finish_reason") in {"stop", "length"}
    finished = time.monotonic_ns()
    connection.close()
    content = "".join(pieces)
    if not content or not saw_role or not saw_terminal or not saw_done:
        raise RuntimeError("stream omitted content, role, terminal chunk, or [DONE]")
    if not isinstance(usage, dict):
        raise RuntimeError("stream omitted requested usage")
    return {
        "content_bytes": len(content.encode()),
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "events": event_count,
        "ttft_ns": (first_content_ns or finished) - started,
        "total_ns": finished - started,
        "usage": usage,
    }


def snapshot(port: int, api_key: str) -> dict[str, Any]:
    status, body = request(
        port,
        "GET",
        "/snapshot",
        timeout=10,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    if status != 200:
        raise RuntimeError(f"snapshot returned HTTP {status}")
    return json.loads(body)


def producer(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        "ssh",
        args.spark_host,
        "docker",
        "exec",
        args.resident,
        "python3",
        args.remote_request_script,
        "--sock",
        args.remote_sock,
        "--tokens",
        args.remote_tokens,
        "--request-id",
        args.request_id,
        "--generation",
        str(args.generation),
        "--transfer-id",
        args.request_id,
        "--receiver-host",
        args.receiver_host,
        "--receiver-port",
        "29590",
        "--output",
        args.remote_output,
    ]
    if args.dflash is not None:
        command.extend(
            [
                "--dflash-session",
                args.dflash_session,
                "--dflash-identity-sha256",
                args.dflash_identity_sha256,
            ]
        )
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.timeout_seconds,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"producer exited with {completed.returncode}: {completed.stdout[-8192:]}"
        )
    rows = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    if not rows:
        raise RuntimeError(f"producer returned no JSON: {completed.stdout[-4096:]}")
    return json.loads(rows[-1])


def main() -> int:
    args = parse_args()
    if (args.dflash is None) != (args.dflash_identity_sha256 is None):
        raise RuntimeError("--dflash and --dflash-identity-sha256 are a pair")
    if args.cache_hit_repetitions <= 0:
        raise RuntimeError("--cache-hit-repetitions must be positive")
    tokens = [int(token) for token in args.tokens.read_text().split()]
    if len(tokens) < 2:
        raise RuntimeError("remote qualification requires at least two tokens")
    shutdown_token = "muser-nvfp4-streaming-qualification-v1"
    api_key = hashlib.sha256(f"{args.request_id}:management".encode()).hexdigest()
    api_key_path = args.output.with_suffix(".api-key")
    api_key_path.parent.mkdir(parents=True, exist_ok=True)
    api_key_path.write_text(api_key + "\n")
    api_key_path.chmod(0o600)
    effective_cluster_path = args.output.with_suffix(".cluster.json")
    effective_cluster = json.loads(args.cluster_config.read_text())
    # The cluster contract caps transport timeouts at 15 minutes. Keep the
    # harness budget independent so teardown and evidence collection can use a
    # longer deadline without producing an invalid server configuration.
    transport_timeout_ms = min(args.timeout_seconds * 1_000, 900_000)
    effective_cluster["timeout_ms"] = transport_timeout_ms
    effective_cluster["wait_for_producer_ms"] = transport_timeout_ms
    effective_cluster["replay_ledger"] = str(args.output.with_suffix(".replay.json"))
    effective_cluster_path.write_text(
        json.dumps(effective_cluster, sort_keys=True, indent=2) + "\n"
    )
    effective_cluster_sha256 = hashlib.sha256(effective_cluster_path.read_bytes()).hexdigest()
    server_command = [
        str(args.server_binary),
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--model",
        str(args.model),
        "--backend",
        "metal",
        "--parallel",
        "1",
        "--max-context",
        "4096",
        "--resident-cache-bytes",
        str(128 * 1024 * 1024),
        "--prefix-cache",
        "on",
        "--prefill",
        "remote",
        "--cluster-config",
        str(effective_cluster_path),
        "--api-key-file",
        str(api_key_path),
        "--benchmark-shutdown-token",
        shutdown_token,
        "--benchmark-deadline-seconds",
        str(args.timeout_seconds + 60),
    ]
    if args.dflash is not None:
        server_command.extend(
            ["--dflash", str(args.dflash), "--dflash-backend", "metal"]
        )
    server = subprocess.Popen(
        server_command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    server_lines: list[str] = []
    assert server.stdout is not None
    drain = threading.Thread(
        target=lambda: server_lines.extend(iter(server.stdout.readline, "")), daemon=True
    )
    drain.start()
    ready = False
    try:
        wait_ready(args.port, server, args.timeout_seconds)
        ready = True
        base_payload: dict[str, Any] = {
            "model": "muse-glimmer-30b",
            "messages": [{"role": "user", "content": "qualification"}],
            "muser_prompt_token_ids": tokens,
            "max_tokens": 8,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        result_queue: queue.Queue[tuple[str, object]] = queue.Queue()

        def first_request() -> None:
            try:
                result_queue.put(
                    ("ok", stream_chat(args.port, base_payload, args.timeout_seconds, api_key))
                )
            except BaseException as error:  # relay thread failures to the main cell
                result_queue.put(("error", error))

        request_thread = threading.Thread(target=first_request, daemon=True)
        request_thread.start()
        time.sleep(0.25)
        if not request_thread.is_alive() and result_queue.empty():
            raise RuntimeError("stream request stopped before producer admission")
        producer_receipt = producer(args)
        request_thread.join(args.timeout_seconds)
        if request_thread.is_alive():
            raise RuntimeError("stream request made no progress after producer completion")
        status, first_value = result_queue.get_nowait()
        if status != "ok":
            raise first_value  # type: ignore[misc]
        first_stream = first_value
        first_snapshot = snapshot(args.port, api_key)

        partial_result = {
            "schema": "muser.nvfp4-streaming-qualification.v1",
            "request_id": args.request_id,
            "generation": args.generation,
            "effective_cluster_config_sha256": effective_cluster_sha256,
            "prompt_tokens": len(tokens),
            "first_stream": first_stream,
            "producer": producer_receipt,
            "economics_after_remote": first_snapshot["economics"],
            "transfer": first_snapshot.get("transfers", []),
            "remote_stream_green": True,
            "prefix_hit_green": False,
        }
        args.output.write_text(json.dumps(partial_result, sort_keys=True, indent=2) + "\n")

        # A constrained request deliberately bypasses DFlash, reaching the
        # target prefix-reuse lookup. No second producer request is issued.
        cached_payload = dict(base_payload)
        cached_payload["response_format"] = {"type": "text"}
        first_economics = first_snapshot["economics"]
        if first_economics["disagg_prefills"] < 1:
            raise RuntimeError("remote streamed request did not record disaggregated prefill")
        # Prime the cache-hit request path once before the five recorded P4
        # samples. The cold request establishes the KV cache; this additional
        # unmeasured hit removes one-time HTTP/serve-path initialization from
        # the warm TTFT distribution.
        cache_hit_warmup = stream_chat(
            args.port, cached_payload, args.timeout_seconds, api_key
        )
        if cache_hit_warmup["content_sha256"] != first_stream["content_sha256"]:
            raise RuntimeError("cache-hit warmup content differs from the remote stream")
        warmup_snapshot = snapshot(args.port, api_key)
        warmup_economics = warmup_snapshot["economics"]
        if warmup_economics["cache_hits"] <= first_economics["cache_hits"]:
            raise RuntimeError("cache-hit warmup did not record a prefix cache hit")

        cached_streams: list[dict[str, Any]] = []
        previous_economics = warmup_economics
        final_snapshot = first_snapshot
        for repetition in range(args.cache_hit_repetitions):
            cached_stream = stream_chat(
                args.port, cached_payload, args.timeout_seconds, api_key
            )
            if cached_stream["content_sha256"] != first_stream["content_sha256"]:
                raise RuntimeError(
                    f"cached stream {repetition} content differs from the remote stream"
                )
            final_snapshot = snapshot(args.port, api_key)
            current_economics = final_snapshot["economics"]
            if current_economics["cache_hits"] <= previous_economics["cache_hits"]:
                raise RuntimeError(
                    f"cached stream {repetition} did not record a prefix cache hit"
                )
            cached_streams.append(cached_stream)
            previous_economics = current_economics
        cache_hit_ttft = distribution(
            [int(stream["ttft_ns"]) for stream in cached_streams]
        )
        response = producer_receipt.get("response", {})
        handoff = response.get("producer_receipt", {}).get("handoff", {})
        if response.get("status") != "ok" or not handoff.get("ack"):
            raise RuntimeError("producer handoff was not acknowledged")
        result = {
            "schema": "muser.nvfp4-streaming-qualification.v1",
            "request_id": args.request_id,
            "generation": args.generation,
            "effective_cluster_config_sha256": effective_cluster_sha256,
            "prompt_tokens": len(tokens),
            "first_stream": first_stream,
            "cache_hit_warmup": cache_hit_warmup,
            "cached_stream": cached_streams[0],
            "cached_streams": cached_streams,
            "cache_hit_ttft": cache_hit_ttft,
            "producer": producer_receipt,
            "economics_after_remote": first_economics,
            "economics_after_cache_hit_warmup": warmup_economics,
            "economics_after_cache_hit": previous_economics,
            "transfer": final_snapshot.get("transfers", []),
            "remote_stream_green": True,
            "prefix_hit_green": True,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
        print(json.dumps(result, sort_keys=True))
    finally:
        if ready and server.poll() is None:
            status, body = request(
                args.port,
                "POST",
                "/__muser/benchmark/shutdown",
                shutdown_token.encode(),
                timeout=10,
            )
            if status != 200:
                raise RuntimeError(f"cooperative shutdown returned HTTP {status}: {body!r}")
        code = server.wait(timeout=30 if ready else args.timeout_seconds)
        drain.join(timeout=5)
        api_key_path.unlink(missing_ok=True)
        if code != 0:
            raise RuntimeError(
                f"server exited with {code}: {''.join(server_lines)[-8192:]}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
