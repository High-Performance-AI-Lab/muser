#!/usr/bin/env python3
"""Run one non-notarial target-only Muser/llama parity and timing smoke."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import secrets
import socket
import subprocess
import tempfile
import time
from urllib.parse import urlsplit

from bench_server_ttft import cooperative_shutdown, wait_ready


TARGET_BYTES = 16_756_681_056
TARGET_SHA256 = "7e9b74b7c8875e9e265695df9613bf6290f2392e479ce740495a129019c488d8"
NO_ANE_BUILD_MARKER = b"this binary was built without the ane-coreml feature"
PINNED_LLAMA_COMMIT = "89e0aa6fd362617d9073e0dafc18e41241521572"
SNAPSHOT_MAX_BYTES = 4 * 1024 * 1024
SNAPSHOT_PHASES = ("queue", "prefill", "sampling", "grammar", "detokenization",
                   "enqueue_write", "dflash_draft", "dflash_target_verify")
TARGET_PHASES = ("queue", "prefill", "sampling", "grammar", "detokenization", "enqueue_write")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt-token-fixture", type=Path, required=True)
    parser.add_argument("--muser-server", type=Path, required=True)
    parser.add_argument("--expected-muser-sha256", required=True)
    parser.add_argument("--muser-metallib", type=Path, required=True)
    parser.add_argument("--llama-server", type=Path, required=True)
    parser.add_argument("--llama-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--muser-url", default="http://127.0.0.1:4949")
    parser.add_argument("--llama-url", default="http://127.0.0.1:8080")
    parser.add_argument(
        "--llama-context",
        type=int,
        default=None,
        help="explicit llama-server -c; required for prompt+output beyond its default context",
    )
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--server-deadline-seconds", type=int, default=1800)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--dry-run", action="store_true", help="alias for --check")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_digest(tokens: list[int]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(token.to_bytes(4, "little", signed=False))
    return "sha256:" + digest.hexdigest()


def checked_file(path: Path, label: str) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label} is missing or unsafe: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def validate_comparator(binary: Path, receipt_path: Path) -> tuple[dict, dict]:
    binary_receipt = checked_file(binary, "llama-server")
    receipt_receipt = checked_file(receipt_path, "llama comparator receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = receipt.get("artifacts", {}).get("llama-server", {})
    if (
        receipt.get("schema") != "muser.llama_comparator.source_receipt.v3"
        or receipt.get("executed") is not False
        or receipt.get("build", {}).get("metal") is not True
        or not isinstance(receipt.get("source_commit"), str)
        or len(receipt["source_commit"]) != 40
        or expected.get("bytes") != binary_receipt["bytes"]
        or expected.get("sha256") != binary_receipt["sha256"]
    ):
        raise RuntimeError("llama-server differs from the pinned unexecuted v3 receipt")
    return receipt, receipt_receipt


def validate_metallib(path: Path) -> tuple[dict, dict]:
    artifact = checked_file(path, "Muser GGML Metal library")
    receipt_path = path.parent / "source-receipt.json"
    receipt_file = checked_file(receipt_path, "Muser GGML Metal source receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema") != "muser.llama_metallib.source_receipt.v1"
        or receipt.get("source_commit") != PINNED_LLAMA_COMMIT
        or receipt.get("artifact_name") != path.name
        or receipt.get("binary_size_bytes") != artifact["bytes"]
        or receipt.get("binary_sha256") != artifact["sha256"]
    ):
        raise RuntimeError("Muser GGML Metal library differs from its pinned source receipt")
    return artifact, receipt_file


def validate_muser_build(path: Path, expected_sha256: str) -> tuple[dict, str]:
    if len(expected_sha256) != 64 or any(value not in "0123456789abcdef" for value in expected_sha256):
        raise RuntimeError("--expected-muser-sha256 must be 64 lowercase hex digits")
    artifact = checked_file(path, "Muser server")
    if artifact["sha256"] != expected_sha256:
        raise RuntimeError("Muser server differs from --expected-muser-sha256")
    if NO_ANE_BUILD_MARKER not in path.read_bytes():
        raise RuntimeError("Muser server does not prove the default no-ANE build feature identity")
    completed = subprocess.run(
        [str(path.resolve()), "--version"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=10,
    )
    version = completed.stdout.decode(errors="replace").strip()
    if completed.returncode != 0 or not version.startswith("muser 0.1."):
        raise RuntimeError(f"Muser server version preflight failed: {version}")
    return artifact, version


def validate_output_path(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to replace output: {path}")
    ancestor = path.parent
    while not ancestor.exists():
        if ancestor == ancestor.parent:
            raise RuntimeError("output path has no existing ancestor")
        ancestor = ancestor.parent
    if not ancestor.is_dir() or ancestor.is_symlink() or not os.access(ancestor, os.W_OK):
        raise RuntimeError(f"output ancestor is not a writable non-symlink directory: {ancestor}")


def validate_free_ports(parts: list[object]) -> None:
    endpoints = [(value.hostname, value.port) for value in parts]
    if len(set(endpoints)) != len(endpoints):
        raise RuntimeError("Muser and llama comparator ports must differ")
    reservations = []
    try:
        for host, port in endpoints:
            family = socket.AF_INET6 if ":" in host else socket.AF_INET
            reservation = socket.socket(family, socket.SOCK_STREAM)
            try:
                reservation.bind((host, port))
            except OSError as error:
                reservation.close()
                raise RuntimeError(f"loopback port {host}:{port} is unavailable: {error}") from error
            reservations.append(reservation)
    finally:
        for reservation in reservations:
            reservation.close()


def loopback_origin(value: str) -> object:
    parts = urlsplit(value)
    if (
        parts.scheme != "http"
        or parts.hostname not in ("127.0.0.1", "::1", "localhost")
        or parts.port is None
        or parts.path not in ("", "/")
        or parts.query
        or parts.fragment
    ):
        raise RuntimeError("server URL must be a loopback HTTP origin with an explicit port")
    return parts


def server_command(
    engine: str,
    binary: Path,
    model: Path,
    muser_metallib: Path | None,
    parts: object,
    token: str,
    deadline: int,
    api_key_file: Path | None = None,
) -> tuple[list[str], dict[str, str]]:
    environment = os.environ.copy()
    # The diagnostic binds the external GGML kernels as a Muser artifact;
    # never let an ambient value leak into either launch identity.
    environment.pop("MUSER_GGML_METALLIB", None)
    if engine == "muser":
        if muser_metallib is None:
            raise RuntimeError("Muser launch requires the pinned GGML Metal library")
        environment["MUSER_GGML_METALLIB"] = str(muser_metallib.resolve())
        command = [
            str(binary), "serve", "--host", "127.0.0.1", "--port", str(parts.port),
            "--model", str(model), "--backend", "metal", "--parallel", "1",
            "--prefix-cache", "off", "--benchmark-shutdown-token", token,
            "--benchmark-deadline-seconds", str(deadline),
        ]
        if api_key_file is not None:
            command.extend(("--api-key-file", str(api_key_file)))
    else:
        command = [
            str(binary), "-m", str(model), "--host", "127.0.0.1", "--port",
            str(parts.port), "-t", "20", "-ngl", "99", "-b", "2048", "-ub",
            # Comparator-sensitivity knob: identical default; the env-prefix
            # form lands in the recorded cell command, so provenance holds.
            os.environ.get("MUSER_LLAMA_UBATCH", "512"),
            "-ctk", "f16", "-ctv", "f16", "-fa", "1", "--parallel", "1",
        ]
        environment["MUSER_COMPARATOR_BENCHMARK_TOKEN"] = token
        environment["MUSER_COMPARATOR_BENCHMARK_DEADLINE_SECONDS"] = str(deadline)
    return command, environment


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"snapshot {label} must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise RuntimeError(f"snapshot {label} must be a finite nonnegative number")
    return result


def _counter(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"snapshot {label} must be a nonnegative integer")
    return value


def telemetry_snapshot_view(snapshot: object) -> dict[str, object]:
    if not isinstance(snapshot, dict) or snapshot.get("schema_version") != 1:
        raise RuntimeError("snapshot must be a schema-version 1 JSON object")
    phases = snapshot.get("_phases")
    if not isinstance(phases, dict) or set(phases) != {*SNAPSHOT_PHASES, "last_request_decode_tok_s"}:
        raise RuntimeError("snapshot _phases does not match the schema-version 1 phase surface")
    phase_view: dict[str, object] = {}
    for name in SNAPSHOT_PHASES:
        measurement = phases[name]
        if not isinstance(measurement, dict) or set(measurement) != {"samples", "total_ms", "mean_ms"}:
            raise RuntimeError(f"snapshot _phases.{name} has an invalid shape")
        samples = _counter(measurement["samples"], f"_phases.{name}.samples")
        total_ms = _finite_number(measurement["total_ms"], f"_phases.{name}.total_ms")
        mean_ms = _finite_number(measurement["mean_ms"], f"_phases.{name}.mean_ms")
        if samples == 0 and (total_ms != 0 or mean_ms != 0):
            raise RuntimeError(f"snapshot _phases.{name} has a nonzero empty counter")
        phase_view[name] = {
            "samples": samples,
            "total_ms": total_ms,
            "mean_ms": mean_ms,
        }
    decode = snapshot.get("_decode")
    if not isinstance(decode, dict):
        raise RuntimeError("snapshot _decode must be an object")
    wire = snapshot.get("wire")
    if not isinstance(wire, dict) or not isinstance(wire.get("ttft_ms"), dict):
        raise RuntimeError("snapshot wire.ttft_ms must be an object")
    ttft = wire["ttft_ms"]
    if set(ttft) != {"p50", "p95"}:
        raise RuntimeError("snapshot wire.ttft_ms has an invalid shape")
    return {
        "completion_tokens": _counter(
            decode.get("completion_tokens"), "_decode.completion_tokens"
        ),
        "last_request_decode_tok_s": _finite_number(
            phases["last_request_decode_tok_s"],
            "_phases.last_request_decode_tok_s",
        ),
        "queue_depth": _counter(snapshot.get("_queue_depth"), "_queue_depth"),
        "ttft_window_ms": {
            "p50": _finite_number(ttft["p50"], "wire.ttft_ms.p50"),
            "p95": _finite_number(ttft["p95"], "wire.ttft_ms.p95"),
        },
        "phases": phase_view,
    }


def request_telemetry_delta(
    before: object,
    after: object,
    expected_tokens: int,
    client_ttft_ns: int,
    final_timings: object,
) -> dict[str, object]:
    left = telemetry_snapshot_view(before)
    right = telemetry_snapshot_view(after)
    if not isinstance(final_timings, dict):
        raise RuntimeError("Muser final timings must be an object")
    if isinstance(client_ttft_ns, bool) or not isinstance(client_ttft_ns, int) or client_ttft_ns <= 0:
        raise RuntimeError("Muser client TTFT must be a positive integer")
    prompt_ms = _finite_number(final_timings.get("prompt_ms"), "final timings.prompt_ms")
    predicted_ms = _finite_number(
        final_timings.get("predicted_ms"), "final timings.predicted_ms"
    )
    phase_deltas: dict[str, object] = {}
    for name in TARGET_PHASES:
        prior = left["phases"][name]
        current = right["phases"][name]
        samples = current["samples"] - prior["samples"]
        total_ms = current["total_ms"] - prior["total_ms"]
        if samples < 0 or total_ms < -1e-9:
            raise RuntimeError(f"snapshot phase counter regressed: {name}")
        total_ms = max(0.0, total_ms)
        phase_deltas[name] = {
            "samples": samples,
            "total_ms": total_ms,
            "mean_ms": total_ms / samples if samples else None,
        }
    completion_delta = right["completion_tokens"] - left["completion_tokens"]
    if completion_delta != expected_tokens:
        raise RuntimeError(
            "snapshot completion-token delta does not match the isolated request "
            f"({completion_delta} != {expected_tokens})"
        )
    return {
        "schema": "muser.request-telemetry-delta.v1",
        "isolation": "parallel=1; snapshot after warmup and immediately before/after request",
        "completion_tokens": {
            "before": left["completion_tokens"],
            "after": right["completion_tokens"],
            "delta": completion_delta,
        },
        "decode": {
            "server_final_predicted_ms": predicted_ms,
            "last_request_decode_tok_s": right["last_request_decode_tok_s"],
        },
        "phases": phase_deltas,
        "queue_depth": {
            "before": left["queue_depth"],
            "after": right["queue_depth"],
            "semantics": "endpoint gauge; queue duration is phases.queue.total_ms",
        },
        "ttft": {
            "request_local_client_ns": client_ttft_ns,
            "snapshot_window_before_ms": left["ttft_window_ms"],
            "snapshot_window_after_ms": right["ttft_window_ms"],
            "snapshot_delta_available": False,
        },
        "prefill": {
            "server_final_prompt_ms": prompt_ms,
            "phase_counter_delta": phase_deltas["prefill"],
        },
        "limitations": [
            "wire.ttft_ms is a rolling percentile window, not a subtractable request counter",
            "last_request_decode_tok_s is a gauge; isolation makes its post-request value attributable",
            "unconstrained requests legitimately produce a zero grammar-phase delta",
        ],
    }


def snapshot_request(parts: object, api_key: str, timeout: int) -> dict[str, object]:
    connection = http.client.HTTPConnection(parts.hostname, parts.port, timeout=timeout)
    try:
        connection.request(
            "GET",
            "/snapshot",
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        )
        response = connection.getresponse()
        payload = response.read(SNAPSHOT_MAX_BYTES + 1)
        if response.status != 200:
            raise RuntimeError(
                f"snapshot returned HTTP {response.status}: {payload[:8192].decode(errors='replace')}"
            )
        if len(payload) > SNAPSHOT_MAX_BYTES:
            raise RuntimeError("snapshot exceeded the diagnostic body limit")
        try:
            value = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise RuntimeError(f"snapshot returned malformed JSON: {error}") from error
        telemetry_snapshot_view(value)
        return value
    finally:
        connection.close()


def stream_request(parts: object, body: bytes, timeout: int) -> dict[str, object]:
    connection = http.client.HTTPConnection(parts.hostname, parts.port, timeout=timeout)
    connection.connect()
    connection.request(
        "POST",
        "/completion",
        body=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    sent_ns = time.perf_counter_ns()
    response = connection.getresponse()
    if response.status != 200:
        payload = response.read(8192).decode(errors="replace")
        connection.close()
        raise RuntimeError(f"completion returned HTTP {response.status}: {payload}")
    first_token_ns = None
    tokens: list[int] = []
    timings = None
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
        event_tokens = event.get("tokens")
        if isinstance(event_tokens, list):
            if not all(isinstance(token, int) and 0 <= token <= 0xFFFFFFFF for token in event_tokens):
                raise RuntimeError("stream returned an invalid raw token ID")
            if event_tokens and first_token_ns is None:
                first_token_ns = time.perf_counter_ns()
            tokens.extend(event_tokens)
        if isinstance(event.get("timings"), dict):
            timings = event["timings"]
    finished_ns = time.perf_counter_ns()
    connection.close()
    if first_token_ns is None or timings is None:
        raise RuntimeError("stream omitted first-token or final timing evidence")
    return {
        "ttft_ns": first_token_ns - sent_ns,
        "wall_ns": finished_ns - sent_ns,
        "tokens": tokens,
        "timings": timings,
    }


def run_engine(
    engine: str,
    binary: Path,
    model: Path,
    muser_metallib: Path | None,
    parts: object,
    body: bytes,
    timeout: int,
    deadline: int,
    log_path: Path,
    extra_command: tuple[str, ...] = (),
) -> dict[str, object]:
    token = secrets.token_hex(32)
    api_key = secrets.token_hex(32) if engine == "muser" else None
    api_key_path: Path | None = None
    if api_key is not None:
        descriptor, name = tempfile.mkstemp(
            prefix=".representative-target-api-key.", dir=log_path.parent
        )
        api_key_path = Path(name)
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write(api_key)
            stream.flush()
            os.fsync(stream.fileno())
    try:
        command, environment = server_command(
            engine, binary, model, muser_metallib, parts, token, deadline, api_key_path,
        )
        command.extend(extra_command)
    except BaseException:
        if api_key_path is not None:
            api_key_path.unlink(missing_ok=True)
        raise
    run_error: BaseException | None = None
    result = None
    telemetry_delta = None
    try:
        with log_path.open("xb") as log:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
            )
            try:
                wait_ready(parts, engine, process, timeout)
                # One-token warmup initializes pipelines without polluting the
                # representative 2048+256 measurement.
                warmup_payload = json.loads(body)
                warmup_payload["n_predict"] = 1
                stream_request(
                    parts,
                    json.dumps(warmup_payload, sort_keys=True, separators=(",", ":")).encode(),
                    timeout,
                )
                before = (
                    snapshot_request(parts, api_key, timeout)
                    if api_key is not None
                    else None
                )
                result = stream_request(parts, body, timeout)
                if api_key is not None:
                    after = snapshot_request(parts, api_key, timeout)
                    telemetry_delta = request_telemetry_delta(
                        before,
                        after,
                        len(result["tokens"]),
                        result["ttft_ns"],
                        result["timings"],
                    )
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
                log.flush()
                os.fsync(log.fileno())
                if return_code != 0 and run_error is None:
                    run_error = RuntimeError(f"{engine} server exited with {return_code}")
    finally:
        if api_key_path is not None:
            api_key_path.unlink(missing_ok=True)
    if run_error is not None:
        raise run_error
    assert result is not None
    tokens = result.pop("tokens")
    timings = result["timings"]
    for field in ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms"):
        if not isinstance(timings.get(field), (int, float)) or timings[field] <= 0:
            raise RuntimeError(f"{engine} final timings omitted positive {field}")
    output = {
        **result,
        "generated_tokens": tokens,
        "generated_tokens_sha256": token_digest(tokens),
        "command": command,
        "server_log": str(log_path),
    }
    if telemetry_delta is not None:
        output["telemetry_delta"] = telemetry_delta
    return output


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


def main() -> int:
    args = parse_args()
    fixture = checked_file(args.prompt_token_fixture, "prompt token fixture")
    try:
        prompt = [int(value) for value in args.prompt_token_fixture.read_bytes().split()]
    except ValueError as error:
        raise SystemExit(f"invalid decimal-u32 prompt fixture: {error}") from error
    if len(prompt) != args.prompt_tokens or any(not 0 <= token <= 0xFFFFFFFF for token in prompt):
        raise SystemExit("prompt fixture does not match --prompt-tokens")
    model = checked_file(args.model, "target model")
    if model["bytes"] != TARGET_BYTES or model["sha256"] != TARGET_SHA256:
        raise SystemExit("target model differs from the pinned v0.1 artifact")
    muser, muser_version = validate_muser_build(
        args.muser_server, args.expected_muser_sha256
    )
    muser_metallib, metallib_receipt = validate_metallib(args.muser_metallib)
    llama_receipt, receipt_file = validate_comparator(args.llama_server, args.llama_receipt)
    if llama_receipt["source_commit"] != PINNED_LLAMA_COMMIT:
        raise SystemExit("llama comparator source commit is not the pinned v0.1 commit")
    llama = checked_file(args.llama_server, "llama-server")
    muser_parts = loopback_origin(args.muser_url)
    llama_parts = loopback_origin(args.llama_url)
    if args.llama_context is not None:
        if args.llama_context < args.prompt_tokens + args.output_tokens:
            raise SystemExit("--llama-context must cover prompt tokens plus output tokens")
        llama_extra_command: tuple[str, ...] = ("-c", str(args.llama_context))
    else:
        llama_extra_command = ()
    validate_free_ports([muser_parts, llama_parts])
    validate_output_path(args.output)
    _, muser_environment = server_command(
        "muser", args.muser_server, args.model, args.muser_metallib,
        muser_parts, "preflight", args.server_deadline_seconds,
    )
    if muser_environment.get("MUSER_GGML_METALLIB") != str(args.muser_metallib.resolve()):
        raise SystemExit("Muser launch environment omitted the pinned GGML Metal library")
    plan = {
        "schema": "muser.representative-target-smoke.v1",
        "status": "checked" if args.check or args.dry_run else "running",
        "notarial": False,
        "seal_eligible": False,
        "accelerator_touched": False,
        "identity": args.identity,
        "cell": {
            "prompt_tokens": args.prompt_tokens,
            "output_tokens": args.output_tokens,
            "concurrency": 1,
            "target_only": True,
            "llama_context": args.llama_context,
        },
        "artifacts": {
            "model": model,
            "prompt_fixture": fixture,
            "muser_server": muser,
            "muser_version": muser_version,
            "muser_build_features": "default-metal-no-ane",
            "muser_metallib": muser_metallib,
            "muser_metallib_receipt": metallib_receipt,
            "llama_server": llama,
            "llama_receipt": receipt_file,
            "llama_source_commit": llama_receipt["source_commit"],
        },
        "measurement": {
            "warmup": "one 2048+1 uncached streamed request per engine",
            "sample": "one 2048+256 uncached streamed request per engine",
            "ttft": "request-send-complete to first raw token SSE event",
            "prefill_decode": "server final timing object",
            "muser_telemetry": (
                "authenticated /snapshot before/after the measured request; monotonic phase "
                "counters are differenced and completion-token isolation is exact"
            ),
        },
        "preflight": {
            "ports_available": [args.muser_url, args.llama_url],
            "output_writable": str(args.output),
            "required_environment": {
                "MUSER_GGML_METALLIB": str(args.muser_metallib.resolve()),
            },
            "accelerator_or_model_initialized": False,
        },
    }
    if args.check or args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {
            "prompt": prompt,
            "n_predict": args.output_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
            "cache_prompt": False,
            "return_tokens": True,
            "stream": True,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    muser_result = run_engine(
        "muser", args.muser_server, args.model, args.muser_metallib,
        muser_parts, payload,
        args.timeout_seconds, args.server_deadline_seconds,
        args.output.with_suffix(".muser-server.log"),
    )
    llama_result = run_engine(
        "llama", args.llama_server, args.model, None,
        llama_parts, payload,
        args.timeout_seconds, args.server_deadline_seconds,
        args.output.with_suffix(".llama-server.log"),
        llama_extra_command,
    )
    muser_tokens = muser_result.pop("generated_tokens")
    llama_tokens = llama_result.pop("generated_tokens")
    exact = muser_tokens == llama_tokens and len(muser_tokens) == args.output_tokens
    report = {
        **plan,
        "status": "passed" if exact else "failed",
        "accelerator_touched": True,
        "exact_tokens": exact,
        "first_mismatch": next(
            (
                index
                for index, (left, right) in enumerate(zip(muser_tokens, llama_tokens))
                if left != right
            ),
            None if len(muser_tokens) == len(llama_tokens) else min(len(muser_tokens), len(llama_tokens)),
        ),
        "muser": muser_result,
        "llama": llama_result,
        "ratios_llama_over_muser": {
            "ttft": llama_result["ttft_ns"] / muser_result["ttft_ns"],
            "prefill": llama_result["timings"]["prompt_ms"] / muser_result["timings"]["prompt_ms"],
            "decode": llama_result["timings"]["predicted_ms"] / muser_result["timings"]["predicted_ms"],
            "wall": llama_result["wall_ns"] / muser_result["wall_ns"],
        },
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
