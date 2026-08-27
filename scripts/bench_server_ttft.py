#!/usr/bin/env python3
"""Measure warm-server TTFT with an identical audited token-ID prompt."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import time
from urllib.parse import urlsplit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--server-binary", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--depth", required=True, type=int)
    parser.add_argument("--repetitions", required=True, type=int)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--engine", choices=("muser", "llama"), required=True)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--server-deadline-seconds", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def server_command(args: argparse.Namespace, parts, token: str) -> tuple[list[str], dict[str, str]]:
    if parts.scheme != "http" or parts.hostname not in ("127.0.0.1", "::1", "localhost"):
        raise RuntimeError("qualification server must use a loopback HTTP origin")
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        raise RuntimeError("--base-url must be an origin without a path, query, or fragment")
    port = parts.port
    if port is None:
        raise RuntimeError("qualification server origin must contain an explicit port")
    environment = os.environ.copy()
    if args.engine == "muser":
        command = [
            args.server_binary, "serve", "--host", "127.0.0.1", "--port", str(port),
            "--model", args.model_path, "--backend", "metal", "--prefix-cache", "off",
            "--benchmark-shutdown-token", token, "--benchmark-deadline-seconds",
            str(args.server_deadline_seconds),
        ]
    else:
        command = [
            args.server_binary, "-m", args.model_path, "--host", "127.0.0.1",
            "--port", str(port), "-t", "20", "-ngl", "99", "-b", "2048",
            "-ub", "512", "-ctk", "f16", "-ctv", "f16", "-fa", "1",
            "--parallel", "1",
        ]
        environment["MUSER_COMPARATOR_BENCHMARK_TOKEN"] = token
        environment["MUSER_COMPARATOR_BENCHMARK_DEADLINE_SECONDS"] = str(
            args.server_deadline_seconds
        )
    return command, environment


def wait_ready(parts, engine: str, process: subprocess.Popen[bytes], timeout: int) -> None:
    deadline = time.monotonic() + timeout
    path = "/healthz" if engine == "muser" else "/health"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"{engine} server exited during startup with {return_code}")
        try:
            connection = http.client.HTTPConnection(parts.hostname, parts.port, timeout=2)
            connection.request("GET", path)
            response = connection.getresponse()
            response.read()
            connection.close()
            if response.status == 200:
                return
        except OSError:
            pass
        time.sleep(0.1)
    raise RuntimeError(f"{engine} server did not become ready before its client deadline")


def cooperative_shutdown(parts, token: str) -> None:
    connection = http.client.HTTPConnection(parts.hostname, parts.port, timeout=10)
    connection.request(
        "POST", "/__muser/benchmark/shutdown", body=token.encode(),
        headers={"Content-Type": "application/octet-stream"},
    )
    response = connection.getresponse()
    payload = response.read(8192)
    connection.close()
    if response.status != 200:
        raise RuntimeError(
            f"qualification server refused cooperative shutdown: HTTP {response.status}: "
            f"{payload.decode(errors='replace')}"
        )


def request_spec(
    engine: str, model: str, tokens: list[int], *, reuse_prompt: bool = False
) -> tuple[str, bytes]:
    if engine == "muser":
        path = "/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "qualification-token-fixture"}],
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "muser_prompt_token_ids": tokens,
            "muser_baseline_ttft": True,
        }
    else:
        path = "/completion"
        payload = {
            "prompt": tokens,
            "n_predict": 1,
            "temperature": 0.0,
            "stream": True,
            "cache_prompt": reuse_prompt,
        }
    return path, json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def one_request(
    parts, path_suffix: str, body: bytes, timeout: int, engine: str
) -> tuple[int, str, int | None]:
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise RuntimeError("--base-url must be an HTTP(S) origin")
    cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    connection = cls(parts.hostname, parts.port, timeout=timeout)
    path = (parts.path.rstrip("/") if parts.path else "") + path_suffix
    connection.connect()
    connection.request(
        "POST",
        path,
        body=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    # http.client.request returns only after the complete body has been sent.
    sent_ns = time.perf_counter_ns()
    response = connection.getresponse()
    if response.status != 200:
        payload = response.read(8192).decode(errors="replace")
        connection.close()
        raise RuntimeError(f"server returned HTTP {response.status}: {payload}")
    first_ns = None
    first_content = ""
    prompt_count = None
    while True:
        line = response.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        event = json.loads(data)
        if engine == "muser":
            usage = event.get("usage")
            if isinstance(usage, dict) and isinstance(usage.get("prompt_tokens"), int):
                prompt_count = usage["prompt_tokens"]
            choices = event.get("choices")
            content = (
                choices[0].get("delta", {}).get("content")
                if isinstance(choices, list) and choices
                else None
            )
        else:
            if isinstance(event.get("tokens_evaluated"), int):
                prompt_count = event["tokens_evaluated"]
            content = event.get("content")
        if isinstance(content, str) and content and first_ns is None:
            first_ns = time.perf_counter_ns()
            first_content = content
    connection.close()
    if first_ns is None:
        raise RuntimeError("SSE response completed without a nonempty content token")
    return first_ns - sent_ns, first_content, prompt_count


def cv(values: list[int]) -> float:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / mean if mean else 0.0


def capture_rendezvous() -> None:
    raw_path = os.environ.get("MUSER_TTFT_CAPTURE_READY_FILE")
    raw_pause = os.environ.get("MUSER_TTFT_CAPTURE_PAUSE_MS")
    if raw_path is None:
        if raw_pause is not None:
            raise RuntimeError(
                "MUSER_TTFT_CAPTURE_PAUSE_MS requires MUSER_TTFT_CAPTURE_READY_FILE"
            )
        return
    if raw_pause is None:
        raise RuntimeError(
            "MUSER_TTFT_CAPTURE_PAUSE_MS is required for capture rendezvous"
        )
    try:
        pause_ms = int(raw_pause)
    except ValueError as error:
        raise RuntimeError("MUSER_TTFT_CAPTURE_PAUSE_MS must be an integer") from error
    if not 1_000 <= pause_ms <= 30_000:
        raise RuntimeError("MUSER_TTFT_CAPTURE_PAUSE_MS must be in 1000..=30000")
    path = Path(raw_path)
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise RuntimeError(f"capture-ready parent must be a real directory: {parent}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(
            descriptor,
            b'{"schema":"muser.server-ttft-capture-ready.v1","ready":true}\n',
        )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    time.sleep(pause_ms / 1000)


def capture_reuses_prompt(engine: str) -> bool:
    raw = os.environ.get("MUSER_TTFT_CAPTURE_REUSE_PROMPT")
    if raw is None:
        return False
    if raw != "1":
        raise RuntimeError("MUSER_TTFT_CAPTURE_REUSE_PROMPT must equal 1")
    if engine != "llama":
        raise RuntimeError("capture prompt reuse is supported only for llama")
    if os.environ.get("MUSER_TTFT_CAPTURE_READY_FILE") is None:
        raise RuntimeError("capture prompt reuse requires the capture rendezvous")
    return True


def main() -> int:
    args = parse_args()
    fixture_bytes = args.prompt_file.read_bytes()
    try:
        tokens = [int(line) for line in fixture_bytes.splitlines() if line.strip()]
    except ValueError as error:
        raise SystemExit(f"invalid decimal-u32 token fixture: {error}") from error
    if len(tokens) != args.depth or any(token < 0 or token > 0xFFFFFFFF for token in tokens):
        raise SystemExit("token fixture depth/range does not match the requested cell")
    prompt_sha = hashlib.sha256(fixture_bytes).hexdigest()
    trace_reuse = capture_reuses_prompt(args.engine)
    report = {
        "schema": "muser.server-ttft.v2",
        "kind": "dry-run" if args.dry_run else "plan",
        "accelerator_touched": False,
        "engine": args.engine,
        "identity": args.identity,
        "depth": args.depth,
        "repetitions": args.repetitions,
        "prompt_sha256": prompt_sha,
        "measurement": "http-request-send-complete-to-first-nonempty-sse-content",
        "cache_required": "trace-only-exact-prompt-reuse" if trace_reuse else "disabled",
        "prompt_schema": "decimal-u32-lines-v1",
        "server_lifecycle": "leased-start-ready-exact-requests-cooperative-exit",
        "warmup_policy": "one-uncached-request-after-ready-before-timing-v1",
        "server_binary": args.server_binary,
        "server_model_path": args.model_path,
        "server_deadline_seconds": args.server_deadline_seconds,
    }
    if args.dry_run:
        print(json.dumps(report, sort_keys=True))
        return 0
    expected = 5
    if args.repetitions != expected:
        raise SystemExit(f"{args.engine} TTFT requires exactly {expected} repetitions")
    parts = urlsplit(args.base_url)
    token = secrets.token_hex(32)
    command, environment = server_command(args, parts, token)
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, env=environment)
    path, body = request_spec(
        args.engine, args.model, tokens, reuse_prompt=trace_reuse
    )
    raw = []
    digests = []
    prompt_counts = []
    run_error: BaseException | None = None
    try:
        wait_ready(parts, args.engine, process, args.timeout_seconds)
        # Discard one uncached request after the server is ready so PSO/runtime
        # warmup is not counted in the three/five recorded samples.
        one_request(parts, path, body, args.timeout_seconds, args.engine)
        capture_rendezvous()
        for repetition in range(args.repetitions):
            elapsed, content, prompt_count = one_request(
                parts, path, body, args.timeout_seconds, args.engine
            )
            raw.append(elapsed)
            digests.append(hashlib.sha256(content.encode()).hexdigest())
            if prompt_count is not None:
                prompt_counts.append(prompt_count)
            print(json.dumps({
                "schema": "muser.server-ttft.v2", "kind": "sample",
                "engine": args.engine, "identity": args.identity, "depth": args.depth,
                "repetition": repetition, "elapsed_ns": elapsed,
                "first_content_sha256": digests[-1], "reported_prompt_tokens": prompt_count,
                "prompt_sha256": prompt_sha,
            }, sort_keys=True))
    except BaseException as error:
        run_error = error
    finally:
        if process.poll() is None:
            try:
                cooperative_shutdown(parts, token)
            except BaseException as error:
                if run_error is None:
                    run_error = error
        # The coordinator never signals or kills the server. The patched
        # child owns both this endpoint and its self-deadline.
        return_code = process.wait()
        if return_code != 0 and run_error is None:
            run_error = RuntimeError(f"{args.engine} server exited abnormally with {return_code}")
    if run_error is not None:
        raise run_error
    print(
        json.dumps(
            {
                "schema": "muser.server-ttft.v2",
                "kind": "summary",
                "engine": args.engine,
                "identity": args.identity,
                "depth": args.depth,
                "raw_ns": raw,
                "cv": cv(raw),
                "stable": cv(raw) <= 0.02,
                "prompt_sha256": prompt_sha,
                "first_content_digests": digests,
                "reported_prompt_tokens": prompt_counts,
                "server_lifecycle": "leased-start-ready-exact-requests-cooperative-exit",
                "warmup_policy": "one-uncached-request-after-ready-before-timing-v1",
                "seal_eligible": not trace_reuse
                and len(prompt_counts) == args.repetitions
                and all(count == args.depth for count in prompt_counts),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
