#!/usr/bin/env python3
"""Run a non-notarial real-model continuous-batching human smoke.

The driver starts a fresh target-only Muser server for each requested resident
slot count.  It compares four stateful generations run in isolation with the
same four generations released concurrently, using target logprob token IDs
as the exact oracle.  The four requests have distinct raw-token prompts,
sampling seeds, and grammar policies so KV, RNG, and grammar state leakage is
observable.  The four-slot cell also checks slow/disconnected-client isolation
and the bounded 64-waiter admission queue.

This is diagnostic evidence only.  It never emits a release seal or readiness
receipt.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import http.client
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import tempfile
import threading
import time
from urllib.parse import urlsplit

from bench_server_ttft import cooperative_shutdown, wait_ready


MODEL_ID = "muse-glimmer-30b"
QUEUE_LIMIT = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--muser-server", type=Path, required=True)
    parser.add_argument("--prompt-token-fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--parallel", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--base-port", type=int, default=4960)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--server-deadline-seconds", type=int, default=3600)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_file(path: Path, label: str) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label} is missing or unsafe: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def token_digest(tokens: list[int]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(token.to_bytes(4, "little", signed=False))
    return "sha256:" + digest.hexdigest()


def read_prompt(path: Path) -> list[int]:
    try:
        values = [int(value) for value in path.read_bytes().split()]
    except ValueError as error:
        raise RuntimeError(f"invalid decimal-u32 prompt fixture: {error}") from error
    if len(values) < 64 or any(not 0 <= value <= 0xFFFFFFFF for value in values):
        raise RuntimeError("prompt fixture must contain at least 64 valid u32 token IDs")
    return values


def derive_prompts(base: list[int]) -> list[list[int]]:
    """Keep a likely BOS fixed while rotating the remaining audited tokens."""
    body = base[1:]
    prompts = []
    for index, offset in enumerate((0, 17, 37, 73)):
        shift = offset % len(body)
        rotated = body[shift:] + body[:shift]
        prompt = [base[0], *rotated]
        if index:
            # Make the final frontier visibly different even for periodic
            # fixtures without inventing a token outside the audited set.
            prompt[-1], prompt[-1 - index] = prompt[-1 - index], prompt[-1]
        prompts.append(prompt)
    digests = {token_digest(prompt) for prompt in prompts}
    if len(digests) != 4:
        raise RuntimeError("could not derive four distinct prompts from fixture")
    return prompts


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


class Client:
    def __init__(self, origin: str, api_key: str, timeout: int) -> None:
        parts = urlsplit(origin)
        if (
            parts.scheme != "http"
            or parts.hostname not in ("127.0.0.1", "::1", "localhost")
            or parts.port is None
            or parts.path not in ("", "/")
            or parts.query
            or parts.fragment
        ):
            raise RuntimeError("server origin must be loopback HTTP with an explicit port")
        self.parts = parts
        self.api_key = api_key
        self.timeout = timeout

    def connection(self) -> http.client.HTTPConnection:
        return http.client.HTTPConnection(
            self.parts.hostname, self.parts.port, timeout=self.timeout
        )

    def request(
        self,
        route: str,
        payload: object | None = None,
        *,
        method: str = "GET",
        headers: dict[str, str] | None = None,
    ) -> tuple[int, object | None]:
        body = None if payload is None else json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode()
        request_headers = {
            "Authorization": f"Bearer {self.api_key}",
            **(headers or {}),
        }
        if body is not None:
            request_headers["Content-Type"] = "application/json"
        connection = self.connection()
        try:
            connection.request(method, route, body=body, headers=request_headers)
            response = connection.getresponse()
            raw = response.read()
            try:
                value = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                value = raw.decode(errors="replace")
            return response.status, value
        finally:
            connection.close()

    def require(
        self,
        route: str,
        payload: object | None = None,
        *,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        status: int = 200,
    ) -> dict:
        actual, value = self.request(route, payload, method=method, headers=headers)
        if actual != status or not isinstance(value, dict):
            raise RuntimeError(f"{route}: expected HTTP {status} JSON, got {actual} {value}")
        return value

    def metrics(self) -> dict[str, float]:
        status, value = self.request("/metrics")
        if status != 200 or not isinstance(value, str):
            raise RuntimeError(f"/metrics failed: HTTP {status} {value}")
        metrics: dict[str, float] = {}
        for line in value.splitlines():
            if not line or line.startswith("#") or "{" in line:
                continue
            fields = line.split()
            if len(fields) == 2:
                try:
                    metrics[fields[0]] = float(fields[1])
                except ValueError:
                    pass
        return metrics


def case_payload(prompt: list[int], index: int, session_id: str, output: int) -> dict:
    payload: dict[str, object] = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": f"batch-isolation-case-{index}"}],
        "muser_prompt_token_ids": prompt,
        "max_tokens": output,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 50,
        "seed": 0x4D55534552000000 + index,
        "ignore_eos": True,
        "cache_prompt": False,
        "logprobs": True,
        "top_logprobs": 0,
        "session_id": session_id,
        "expected_revision": 0,
    }
    if index == 2:
        payload["grammar"] = 'root ::= [0-9]+'
        payload["ignore_eos"] = False
    elif index == 3:
        payload["grammar"] = 'root ::= [A-Z]+'
        payload["ignore_eos"] = False
    return payload


def response_tokens(value: dict) -> tuple[list[int], bytes]:
    choices = value.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError("chat response did not contain exactly one choice")
    choice = choices[0]
    rows = choice.get("logprobs", {}).get("content") if isinstance(choice, dict) else None
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("chat response omitted target token logprobs")
    tokens: list[int] = []
    raw = bytearray()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), int):
            raise RuntimeError("chat logprob row omitted token ID")
        token_bytes = row.get("bytes")
        if not isinstance(token_bytes, list) or any(
            not isinstance(byte, int) or not 0 <= byte <= 255 for byte in token_bytes
        ):
            raise RuntimeError("chat logprob row omitted raw bytes")
        tokens.append(row["id"])
        raw.extend(token_bytes)
    return tokens, bytes(raw)


def create_session(client: Client, session_id: str) -> None:
    value = client.require("/v1/sessions", {"id": session_id}, method="POST", status=201)
    if value.get("revision") != 0:
        raise RuntimeError("new logical session did not begin at revision zero")


def delete_session(client: Client, session_id: str) -> None:
    status, value = client.request(f"/v1/sessions/{session_id}", method="DELETE")
    if status != 204 or value is not None:
        raise RuntimeError(f"logical session delete failed: HTTP {status} {value}")


def generate_case(
    client: Client,
    prompt: list[int],
    index: int,
    session_id: str,
    output: int,
) -> dict[str, object]:
    key = f"batch-smoke-{session_id}-{secrets.token_hex(8)}"
    started = time.perf_counter_ns()
    value = client.require(
        "/v1/chat/completions",
        case_payload(prompt, index, session_id, output),
        method="POST",
        headers={"Idempotency-Key": key},
    )
    elapsed = time.perf_counter_ns() - started
    tokens, raw = response_tokens(value)
    if value.get("muser_session_revision") != 1:
        raise RuntimeError("stateful generation did not commit revision one")
    view = client.require(f"/v1/sessions/{session_id}")
    if view.get("revision") != 1 or view.get("busy") is not False:
        raise RuntimeError("committed logical session retained wrong revision/busy state")
    if index == 2 and raw and any(byte not in b"0123456789" for byte in raw):
        raise RuntimeError("numeric grammar case emitted a non-digit byte")
    if index == 3 and raw and any(byte not in b"ABCDEFGHIJKLMNOPQRSTUVWXYZ" for byte in raw):
        raise RuntimeError("uppercase grammar case emitted an out-of-grammar byte")
    return {
        "tokens": tokens,
        "tokens_sha256": token_digest(tokens),
        "raw_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "elapsed_ns": elapsed,
        "revision": 1,
    }


def run_equivalence(client: Client, prompts: list[list[int]], output: int, label: str) -> dict:
    isolated = []
    for index, prompt in enumerate(prompts):
        session_id = f"{label}-isolated-{index}-{secrets.token_hex(4)}"
        create_session(client, session_id)
        try:
            isolated.append(generate_case(client, prompt, index, session_id, output))
        finally:
            delete_session(client, session_id)

    session_ids = [f"{label}-concurrent-{index}-{secrets.token_hex(4)}" for index in range(4)]
    for session_id in session_ids:
        create_session(client, session_id)
    try:
        barrier = threading.Barrier(4)

        def concurrent(index: int) -> dict[str, object]:
            barrier.wait(timeout=10)
            return generate_case(client, prompts[index], index, session_ids[index], output)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(concurrent, index) for index in range(4)]
            concurrent_results = [future.result() for future in futures]
    finally:
        for session_id in session_ids:
            delete_session(client, session_id)
    for index, (expected, actual) in enumerate(zip(isolated, concurrent_results, strict=True)):
        if expected["tokens"] != actual["tokens"]:
            raise RuntimeError(f"case {index} changed between isolated and concurrent execution")
    return {
        "exact_token_equality": True,
        "prompt_sha256": [token_digest(prompt) for prompt in prompts],
        "isolated": [{key: value for key, value in item.items() if key != "tokens"} for item in isolated],
        "concurrent": [
            {key: value for key, value in item.items() if key != "tokens"}
            for item in concurrent_results
        ],
    }


def busy_slots(client: Client) -> int:
    status, value = client.request("/slots")
    if status != 200 or not isinstance(value, list):
        raise RuntimeError(f"/slots failed: HTTP {status} {value}")
    return sum(bool(slot.get("is_processing")) for slot in value if isinstance(slot, dict))


def wait_for_busy(client: Client, minimum: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if busy_slots(client) >= minimum:
            return
        time.sleep(0.025)
    raise RuntimeError(f"did not observe {minimum} busy slot(s)")


def wait_for_idle(client: Client, timeout: float) -> float:
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if busy_slots(client) == 0:
            return time.monotonic() - started
        time.sleep(0.025)
    raise RuntimeError("serving slots remained busy after client disconnect")


def slow_disconnect_check(
    client: Client, prompt: list[int], baseline: dict[str, object], output: int
) -> dict:
    payload = {
        "prompt": prompt,
        "n_predict": min(1024, max(output * 8, 256)),
        "temperature": 0,
        "ignore_eos": True,
        "cache_prompt": False,
        "return_tokens": True,
        "stream": True,
        "id_slot": 0,
    }
    body = json.dumps(payload, separators=(",", ":")).encode()
    connection = client.connection()
    connection.connect()
    assert connection.sock is not None
    connection.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
    connection.request(
        "POST",
        "/completion",
        body=body,
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    if response.status != 200:
        raise RuntimeError(f"slow stream returned HTTP {response.status}")
    wait_for_busy(client, 1, 5)
    session_id = f"slow-peer-{secrets.token_hex(4)}"
    create_session(client, session_id)
    try:
        fast = generate_case(client, prompt, 0, session_id, output)
    finally:
        delete_session(client, session_id)
    if fast["tokens_sha256"] != baseline["tokens_sha256"]:
        raise RuntimeError("healthy peer changed while a slow client occupied another slot")
    connection.close()
    released = wait_for_idle(client, 6)
    return {
        "healthy_peer_exact": True,
        "disconnect_slot_release_seconds": released,
        "slow_receive_buffer_bytes": 1024,
    }


def queue_worker(client: Client, start: threading.Event, prompt: list[int]) -> tuple[int, object | None]:
    start.wait(timeout=10)
    return client.request(
        "/completion",
        {
            "prompt": prompt[:1],
            "n_predict": 256,
            "temperature": 0,
            "ignore_eos": True,
            "cache_prompt": False,
            "return_tokens": True,
            "id_slot": 0,
        },
        method="POST",
    )


def queue_limit_check(client: Client, prompt: list[int]) -> dict:
    before = client.metrics().get("muser_overload_rejections_total", 0.0)
    start = threading.Event()
    count = QUEUE_LIMIT + 2
    with ThreadPoolExecutor(max_workers=count) as executor:
        futures: list[Future[tuple[int, object | None]]] = [
            executor.submit(queue_worker, client, start, prompt) for _ in range(count)
        ]
        start.set()
        max_queue = 0
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            depth = int(client.metrics().get("muser_queue_depth", 0.0))
            max_queue = max(max_queue, depth)
            if depth >= QUEUE_LIMIT:
                break
            time.sleep(0.01)
        probe_start = threading.Event()
        probe_start.set()
        probe_status, probe = queue_worker(client, probe_start, prompt)
        results = [future.result() for future in futures]
    statuses = [status for status, _ in results]
    after = client.metrics().get("muser_overload_rejections_total", 0.0)
    if max_queue < QUEUE_LIMIT:
        raise RuntimeError(f"bounded queue never reached {QUEUE_LIMIT}; observed {max_queue}")
    if probe_status != 429 or not isinstance(probe, dict):
        raise RuntimeError(f"queue overflow probe returned HTTP {probe_status}: {probe}")
    if 200 not in statuses or 429 not in statuses or after <= before:
        raise RuntimeError(f"queue results did not prove admission and rejection: {statuses}")
    return {
        "configured_waiters": QUEUE_LIMIT,
        "observed_max_queue_depth": max_queue,
        "overflow_probe_status": probe_status,
        "worker_status_counts": {str(status): statuses.count(status) for status in sorted(set(statuses))},
        "overload_rejections_delta": int(after - before),
    }


def server_command(
    server: Path,
    model: Path,
    port: int,
    parallel: int,
    max_context: int,
    key_file: Path,
    token: str,
    deadline: int,
) -> list[str]:
    return [
        str(server), "serve", "--host", "127.0.0.1", "--port", str(port),
        "--model", str(model), "--backend", "metal", "--parallel", str(parallel),
        "--max-context", str(max_context), "--context-policy", "error",
        "--prefix-cache", "off", "--api-key-file", str(key_file),
        "--benchmark-shutdown-token", token, "--benchmark-deadline-seconds", str(deadline),
    ]


def run_parallel_cell(
    args: argparse.Namespace,
    prompts: list[list[int]],
    parallel: int,
    output_dir: Path,
) -> dict:
    port = args.base_port + parallel
    origin = f"http://127.0.0.1:{port}"
    parts = urlsplit(origin)
    api_key = secrets.token_urlsafe(32)
    shutdown_token = secrets.token_hex(32)
    with tempfile.TemporaryDirectory(prefix=f"muser-batch-p{parallel}-") as directory:
        private = Path(directory)
        key_file = private / "api.key"
        key_file.write_text(api_key + "\n", encoding="utf-8")
        key_file.chmod(0o600)
        environment = os.environ.copy()
        environment["MUSER_HOME"] = str(private / "home")
        command = server_command(
            args.muser_server, args.model, port, parallel, args.max_context,
            key_file, shutdown_token, args.server_deadline_seconds,
        )
        log_path = output_dir / f"parallel-{parallel}.server.log"
        run_error: BaseException | None = None
        result = None
        with log_path.open("xb") as log:
            process = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, env=environment,
            )
            try:
                wait_ready(parts, "muser", process, args.timeout_seconds)
                client = Client(origin, api_key, args.timeout_seconds)
                metrics_before = client.metrics()
                equivalence = run_equivalence(
                    client, prompts, args.output_tokens, f"p{parallel}"
                )
                metrics_after = client.metrics()
                packed_batches = int(
                    metrics_after.get("muser_decode_packed_batches_total", 0)
                    - metrics_before.get("muser_decode_packed_batches_total", 0)
                )
                packed_rows = int(
                    metrics_after.get("muser_decode_packed_rows_total", 0)
                    - metrics_before.get("muser_decode_packed_rows_total", 0)
                )
                if parallel > 1 and packed_rows <= packed_batches:
                    raise RuntimeError("concurrent cell did not record a multi-row decode batch")
                result = {
                    "parallel": parallel,
                    "command": command,
                    "server_log": str(log_path),
                    "equivalence": equivalence,
                    "scheduler": {
                        "packed_batches_delta": packed_batches,
                        "packed_rows_delta": packed_rows,
                    },
                }
                if parallel == 4:
                    result["slow_disconnect"] = slow_disconnect_check(
                        client, prompts[0], equivalence["isolated"][0], args.output_tokens
                    )
                    result["queue_limit"] = queue_limit_check(client, prompts[0])
            except BaseException as error:
                run_error = error
            finally:
                if process.poll() is None:
                    try:
                        cooperative_shutdown(parts, shutdown_token)
                    except BaseException as error:
                        if run_error is None:
                            run_error = error
                return_code = process.wait()
                log.flush()
                os.fsync(log.fileno())
                if return_code != 0 and run_error is None:
                    run_error = RuntimeError(f"Muser server exited with {return_code}")
        if run_error is not None:
            raise run_error
        assert result is not None
        return result


def validate_args(args: argparse.Namespace, prompt: list[int]) -> None:
    if args.parallel != [1, 2, 4]:
        raise RuntimeError("human smoke requires exactly --parallel 1 2 4")
    if args.output_tokens < 1 or args.output_tokens > 256:
        raise RuntimeError("--output-tokens must be in 1..=256")
    if args.max_context < len(prompt) + max(1024, args.output_tokens):
        raise RuntimeError("--max-context cannot fit the prompt and slow-client probe")
    if not 1024 <= args.base_port <= 65400:
        raise RuntimeError("--base-port must leave room for parallel-specific ports")
    if args.output.exists() or args.output.is_symlink():
        raise RuntimeError(f"refusing to replace output: {args.output}")


def main() -> int:
    args = parse_args()
    model = checked_file(args.model, "target model")
    server = checked_file(args.muser_server, "Muser server")
    fixture = checked_file(args.prompt_token_fixture, "prompt token fixture")
    base_prompt = read_prompt(args.prompt_token_fixture)
    validate_args(args, base_prompt)
    prompts = derive_prompts(base_prompt)
    plan = {
        "schema": "muser.continuous-batching-human-smoke.v1",
        "status": "planned" if args.dry_run else "running",
        "notarial": False,
        "seal_eligible": False,
        "accelerator_touched": False,
        "identity": args.identity,
        "artifacts": {"model": model, "muser_server": server, "prompt_fixture": fixture},
        "cells": {
            "parallel": args.parallel,
            "prompts": 4,
            "prompt_tokens": len(base_prompt),
            "output_tokens": args.output_tokens,
            "max_context": args.max_context,
        },
        "checks": [
            "isolated-vs-concurrent target-token equality",
            "four independent logical sessions and prompt KV frontiers",
            "deterministic per-request RNG isolation",
            "numeric and uppercase grammar isolation",
            "multi-row scheduler dispatch at parallelism 2 and 4",
            "slow/disconnected-client isolation at parallelism 4",
            "64-waiter queue saturation and HTTP 429 overflow",
        ],
        "prompt_sha256": [token_digest(prompt) for prompt in prompts],
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = [
        run_parallel_cell(args, prompts, parallel, args.output.parent)
        for parallel in args.parallel
    ]
    report = {
        **plan,
        "status": "passed",
        "accelerator_touched": True,
        "results": results,
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
