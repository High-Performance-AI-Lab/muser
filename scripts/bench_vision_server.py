#!/usr/bin/env python3
"""Measure warm OpenAI-compatible vision TTFT with one exact image fixture."""

from __future__ import annotations

import argparse
import base64
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
    parser.add_argument("--mmproj", required=True)
    parser.add_argument("--mtmd-bridge")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--repetitions", type=int, required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--engine", choices=("muser", "llama"), required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--server-deadline-seconds", type=int, default=3600)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def cv(values: list[int]) -> float:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / mean if mean else 0.0


def server_command(args: argparse.Namespace, parts, token: str) -> tuple[list[str], dict[str, str]]:
    if parts.scheme != "http" or parts.hostname not in ("127.0.0.1", "::1", "localhost"):
        raise RuntimeError("qualification server must use a loopback HTTP origin")
    if parts.path not in ("", "/") or parts.query or parts.fragment or parts.port is None:
        raise RuntimeError("--base-url must be a loopback origin with an explicit port")
    environment = os.environ.copy()
    if args.engine == "muser":
        if not args.mtmd_bridge:
            raise RuntimeError("Muser vision qualification requires --mtmd-bridge")
        command = [
            args.server_binary, "serve", "--host", "127.0.0.1", "--port", str(parts.port),
            "--model", args.model_path, "--backend", "metal", "--prefix-cache", "off",
            "--mmproj", args.mmproj, "--mtmd-bridge", args.mtmd_bridge,
            "--benchmark-shutdown-token", token, "--benchmark-deadline-seconds",
            str(args.server_deadline_seconds),
        ]
    else:
        command = [
            args.server_binary, "-m", args.model_path, "--mmproj", args.mmproj,
            "--host", "127.0.0.1", "--port", str(parts.port), "-t", "20", "-ngl", "99",
            "-b", "2048", "-ub", "512", "-ctk", "f16", "-ctv", "f16", "-fa", "1",
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


def one_request(parts, body: bytes, timeout: int) -> tuple[int, str, int | None]:
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise RuntimeError("--base-url must be an HTTP(S) origin")
    cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    connection = cls(parts.hostname, parts.port, timeout=timeout)
    path = (parts.path.rstrip("/") if parts.path else "") + "/v1/chat/completions"
    connection.connect()
    connection.request(
        "POST", path, body=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    sent_ns = time.perf_counter_ns()
    response = connection.getresponse()
    if response.status != 200:
        payload = response.read(8192).decode(errors="replace")
        connection.close()
        raise RuntimeError(f"server returned HTTP {response.status}: {payload}")
    first_ns = None
    first_content = ""
    prompt_tokens = None
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
        usage = event.get("usage")
        if isinstance(usage, dict) and isinstance(usage.get("prompt_tokens"), int):
            prompt_tokens = usage["prompt_tokens"]
        choices = event.get("choices")
        content = (
            choices[0].get("delta", {}).get("content")
            if isinstance(choices, list) and choices
            else None
        )
        if isinstance(content, str) and content and first_ns is None:
            first_ns = time.perf_counter_ns()
            first_content = content
    connection.close()
    if first_ns is None:
        raise RuntimeError("SSE response completed without a nonempty content token")
    return first_ns - sent_ns, first_content, prompt_tokens


def main() -> int:
    args = parse_args()
    image = args.image.read_bytes()
    image_sha = hashlib.sha256(image).hexdigest()
    data_url = "data:image/png;base64," + base64.b64encode(image).decode()
    payload = {
        "model": args.model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": "\nDescribe the image."},
            ],
        }],
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_prompt": False,
    }
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    plan = {
        "schema": "muser.vision-server-ttft.v1",
        "kind": "dry-run" if args.dry_run else "plan",
        "accelerator_touched": False,
        "engine": args.engine,
        "identity": args.identity,
        "fixture": args.fixture,
        "image_sha256": image_sha,
        "repetitions": args.repetitions,
        "measurement": "http-request-send-complete-to-first-nonempty-SSE-content",
        "cache_required": "disabled",
        "server_lifecycle": "leased-start-ready-exact-requests-cooperative-exit",
        "server_binary": args.server_binary,
        "server_model_path": args.model_path,
        "mmproj": args.mmproj,
        "mtmd_bridge": args.mtmd_bridge if args.engine == "muser" else None,
        "server_deadline_seconds": args.server_deadline_seconds,
    }
    if args.dry_run:
        print(json.dumps(plan, sort_keys=True))
        return 0
    expected = 3 if args.engine == "muser" else 5
    if args.repetitions != expected:
        raise SystemExit(f"{args.engine} vision TTFT requires exactly {expected} repetitions")
    parts = urlsplit(args.base_url)
    token = secrets.token_hex(32)
    command, environment = server_command(args, parts, token)
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, env=environment)
    raw: list[int] = []
    content_digests: list[str] = []
    prompt_counts: list[int] = []
    run_error: BaseException | None = None
    try:
        wait_ready(parts, args.engine, process, args.timeout_seconds)
        for repetition in range(args.repetitions):
            elapsed, content, prompt_tokens = one_request(parts, body, args.timeout_seconds)
            raw.append(elapsed)
            content_digests.append(hashlib.sha256(content.encode()).hexdigest())
            if prompt_tokens is not None:
                prompt_counts.append(prompt_tokens)
            print(json.dumps({
                "schema": "muser.vision-server-ttft.v1",
                "kind": "sample",
                "engine": args.engine,
                "identity": args.identity,
                "fixture": args.fixture,
                "repetition": repetition,
                "elapsed_ns": elapsed,
                "first_content_sha256": content_digests[-1],
                "reported_prompt_tokens": prompt_tokens,
                "image_sha256": image_sha,
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
        return_code = process.wait()
        if return_code != 0 and run_error is None:
            run_error = RuntimeError(f"{args.engine} server exited abnormally with {return_code}")
    if run_error is not None:
        raise run_error
    coefficient = cv(raw)
    print(json.dumps({
        "schema": "muser.vision-server-ttft.v1",
        "kind": "summary",
        "engine": args.engine,
        "identity": args.identity,
        "fixture": args.fixture,
        "raw_ns": raw,
        "cv": coefficient,
        "stable": coefficient <= 0.02,
        "image_sha256": image_sha,
        "first_content_digests": content_digests,
        "reported_prompt_tokens": prompt_counts,
        "server_lifecycle": "leased-start-ready-exact-requests-cooperative-exit",
        "seal_eligible": len(set(content_digests)) == 1
        and len(prompt_counts) == args.repetitions
        and len(set(prompt_counts)) == 1,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
