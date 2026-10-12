#!/usr/bin/env python3
"""Measure source-pinned llama.cpp DFlash decode on an exact token fixture."""

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

from bench_server_ttft import cooperative_shutdown, wait_ready


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dflash", required=True)
    parser.add_argument("--prompt-token-fixture", type=Path, required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--verify-length", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--server-deadline-seconds", type=int, default=1800)
    parser.add_argument(
        "--human-smoke", action="store_true",
        help="allow one non-notarial representative sample instead of the formal 5-rep lane",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def cv(values: list[int]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)) / mean


def token_digest(tokens: list[int]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(token.to_bytes(4, "little", signed=False))
    return "sha256:" + digest.hexdigest()


def one_request(parts, body: bytes, timeout: int) -> tuple[int, list[int], dict]:
    connection = http.client.HTTPConnection(parts.hostname, parts.port, timeout=timeout)
    connection.request(
        "POST", "/completion", body=body,
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    payload = response.read()
    connection.close()
    if response.status != 200:
        raise RuntimeError(
            f"llama DFlash returned HTTP {response.status}: {payload[:8192].decode(errors='replace')}"
        )
    result = json.loads(payload)
    tokens = result.get("tokens")
    timings = result.get("timings")
    if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
        raise RuntimeError("llama DFlash response omitted raw generated token IDs")
    if not isinstance(timings, dict) or not isinstance(timings.get("predicted_ms"), (int, float)):
        raise RuntimeError("llama DFlash response omitted decode-only timings")
    elapsed_ns = round(float(timings["predicted_ms"]) * 1_000_000)
    if elapsed_ns <= 0:
        raise RuntimeError("llama DFlash returned a non-positive decode timing")
    return elapsed_ns, tokens, timings


def main() -> int:
    args = parse_args()
    expected_repetitions = 1 if args.human_smoke else 5
    if args.repetitions != expected_repetitions or args.output_tokens != 256:
        raise SystemExit(
            f"this DFlash mode requires {expected_repetitions} repetitions and 256 output tokens"
        )
    if args.verify_length not in (3, 7, 15):
        raise SystemExit("verify length must be 3, 7, or 15")
    fixture = args.prompt_token_fixture.read_bytes()
    try:
        prompt = [int(line) for line in fixture.splitlines() if line.strip()]
    except ValueError as error:
        raise SystemExit(f"invalid decimal-u32 token fixture: {error}") from error
    if len(prompt) != args.depth or any(token < 0 or token > 0xFFFFFFFF for token in prompt):
        raise SystemExit("token fixture depth/range does not match the requested cell")
    fixture_sha = hashlib.sha256(fixture).hexdigest()
    plan = {
        "schema": "muser.llama-dflash.v1",
        "kind": "dry-run" if args.dry_run else "plan",
        "accelerator_touched": False,
        "identity": args.identity,
        "depth": args.depth,
        "output_tokens": args.output_tokens,
        "verify_length": args.verify_length,
        "repetitions": args.repetitions,
        "prompt_file_sha256": fixture_sha,
        "measurement": "llama-server-predicted-ms-decode-only",
        "speculative_type": "draft-dflash",
        "server_lifecycle": "leased-start-ready-exact-requests-cooperative-exit",
        "notarial": not args.human_smoke,
    }
    if args.dry_run:
        print(json.dumps(plan, sort_keys=True))
        return 0

    from urllib.parse import urlsplit
    parts = urlsplit(args.base_url)
    if parts.scheme != "http" or parts.hostname not in ("127.0.0.1", "::1", "localhost"):
        raise SystemExit("llama DFlash qualification requires a loopback HTTP origin")
    if parts.port is None or parts.path not in ("", "/") or parts.query or parts.fragment:
        raise SystemExit("--base-url must be a loopback origin with an explicit port")
    token = secrets.token_hex(32)
    environment = os.environ.copy()
    environment["MUSER_COMPARATOR_BENCHMARK_TOKEN"] = token
    environment["MUSER_COMPARATOR_BENCHMARK_DEADLINE_SECONDS"] = str(
        args.server_deadline_seconds
    )
    command = [
        args.server_binary, "-m", args.model, "-md", args.dflash,
        "--spec-type", "draft-dflash", "--spec-draft-n-max", str(args.verify_length),
        "--spec-draft-n-min", str(args.verify_length), "--spec-draft-p-min", "0",
        "--host", "127.0.0.1", "--port", str(parts.port), "-t", "20", "-ngl", "99",
        "-b", "2048", "-ub", "512", "-ctk", "f16", "-ctv", "f16", "-fa", "1",
        "--parallel", "1",
    ]
    body = json.dumps({
        "prompt": prompt,
        "n_predict": args.output_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "cache_prompt": False,
        "return_tokens": True,
        "stream": False,
    }, sort_keys=True, separators=(",", ":")).encode()
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, env=environment)
    raw: list[int] = []
    canonical_digest: str | None = None
    run_error: BaseException | None = None
    try:
        wait_ready(parts, "llama", process, args.timeout_seconds)
        for repetition in range(args.repetitions):
            elapsed_ns, tokens, timings = one_request(parts, body, args.timeout_seconds)
            if len(tokens) != args.output_tokens or timings.get("predicted_n") != args.output_tokens:
                raise RuntimeError("llama DFlash did not produce the exact requested output length")
            if timings.get("prompt_n") != args.depth:
                raise RuntimeError("llama DFlash did not evaluate the exact prompt depth")
            if not isinstance(timings.get("draft_n"), int) or timings["draft_n"] <= 0:
                raise RuntimeError("llama comparator did not activate its DFlash route")
            digest = token_digest(tokens)
            if canonical_digest is not None and digest != canonical_digest:
                raise RuntimeError("llama DFlash output changed between repetitions")
            canonical_digest = digest
            raw.append(elapsed_ns)
            print(json.dumps({
                "schema": "muser.llama-dflash.v1", "kind": "sample",
                "identity": args.identity, "depth": args.depth, "repetition": repetition,
                "elapsed_ns": elapsed_ns, "output_tokens": args.output_tokens,
                "verify_length": args.verify_length, "generated_tokens_sha256": digest,
                "prompt_file_sha256": fixture_sha, "drafted_tokens": timings["draft_n"],
                "accepted_draft_tokens": timings.get("draft_n_accepted"),
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
            run_error = RuntimeError(f"llama DFlash server exited abnormally with {return_code}")
    if run_error is not None:
        raise run_error
    stability = cv(raw)
    print(json.dumps({
        "schema": "muser.llama-dflash.v1", "kind": "summary",
        "identity": args.identity, "depth": args.depth, "raw_ns": raw,
        "cv": stability, "stable": stability <= 0.03,
        "output_tokens": args.output_tokens, "verify_length": args.verify_length,
        "prompt_file_sha256": fixture_sha,
        "generated_tokens_sha256": canonical_digest,
        "seal_eligible": canonical_digest is not None and not args.human_smoke,
        "notarial": not args.human_smoke,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
