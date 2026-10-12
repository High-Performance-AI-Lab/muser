#!/usr/bin/env python3
"""Run one non-notarial Muser/llama DFlash speculative parity and timing smoke.

This is the speculative sibling of representative_target_smoke.py: same fresh
servers, same one 2048+1 uncached streamed warmup, same one measured uncached
2048+256 streamed request, same loopback-only lifecycle. Both engines run
their native DFlash route at verify length 15 on the pinned draft artifact:

- llama: -md <draft> --spec-type draft-dflash --spec-draft-n-max 15
  --spec-draft-n-min 15 --spec-draft-p-min 0
- muser: --dflash <draft> --dflash-backend metal, DFLASH_VERIFY_LEN default 15

The request body deliberately omits ignore_eos: muser's speculative narrow
contract rejects stop-condition exceptions, so both engines run the plain
greedy stop-at-EOS contract with n_predict as the only bound. Token equality
against both the llama DFlash stream and (via the recorded digest) muser's
own target-only greedy stream is the losslessness gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import representative_target_smoke as base

DFLASH_BYTES = 1_631_205_312
DFLASH_SHA256 = "27d9a805fa29b943cfb6ad4843367cd4eaaaf06bd452d8cc3e00a2cd18a677bc"
VERIFY_LENGTH = 15
SPEC_PHASES = ("dflash_draft", "dflash_target_verify")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dflash", type=Path, required=True)
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
    parser.add_argument(
        "--verify-length",
        type=int,
        default=VERIFY_LENGTH,
        choices=(3, 7, 15),
        help="speculative verify length for BOTH engines; the engine accepts "
        "only 3, 7 or 15",
    )
    parser.add_argument(
        "--llama-context",
        type=int,
        default=None,
        help="explicit llama-server -c; must cover prompt tokens plus output tokens",
    )
    parser.add_argument(
        "--target-only-digest",
        default=None,
        help="expected sha256:<hex> of muser target-only greedy tokens for this cell",
    )
    parser.add_argument("--muser-url", default="http://127.0.0.1:4949")
    parser.add_argument("--llama-url", default="http://127.0.0.1:8080")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--server-deadline-seconds", type=int, default=1800)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--dry-run", action="store_true", help="alias for --check")
    return parser.parse_args()


def spec_server_command(
    engine: str,
    binary: Path,
    model: Path,
    dflash: Path,
    muser_metallib: Path | None,
    parts: object,
    token: str,
    deadline: int,
    api_key_file: Path | None = None,
    llama_context: int | None = None,
    verify_length: int = VERIFY_LENGTH,
) -> tuple[list[str], dict[str, str]]:
    command, environment = base.server_command(
        engine, binary, model, muser_metallib, parts, token, deadline, api_key_file,
    )
    if engine == "muser":
        command.extend((
            "--dflash", str(dflash),
            "--dflash-backend", "metal",
        ))
        environment["MUSER_DFLASH_VERIFY_LEN"] = str(verify_length)
    else:
        command.extend((
            "-md", str(dflash),
            "--spec-type", "draft-dflash",
            "--spec-draft-n-max", str(verify_length),
            "--spec-draft-n-min", str(verify_length),
            "--spec-draft-p-min", "0",
        ))
        if llama_context is not None:
            command.extend(("-c", str(llama_context)))
    return command, environment


def main() -> int:
    args = parse_args()
    fixture = base.checked_file(args.prompt_token_fixture, "prompt token fixture")
    try:
        prompt = [int(value) for value in args.prompt_token_fixture.read_bytes().split()]
    except ValueError as error:
        raise SystemExit(f"invalid decimal-u32 prompt fixture: {error}") from error
    if len(prompt) != args.prompt_tokens or any(not 0 <= token <= 0xFFFFFFFF for token in prompt):
        raise SystemExit("prompt fixture does not match --prompt-tokens")
    model = base.checked_file(args.model, "target model")
    if model["bytes"] != base.TARGET_BYTES or model["sha256"] != base.TARGET_SHA256:
        raise SystemExit("target model differs from the pinned v0.1 artifact")
    dflash = base.checked_file(args.dflash, "DFlash draft model")
    if dflash["bytes"] != DFLASH_BYTES or dflash["sha256"] != DFLASH_SHA256:
        raise SystemExit("DFlash draft model differs from the pinned Stage B artifact")
    muser, muser_version = base.validate_muser_build(
        args.muser_server, args.expected_muser_sha256
    )
    muser_metallib, metallib_receipt = base.validate_metallib(args.muser_metallib)
    llama_receipt, receipt_file = base.validate_comparator(args.llama_server, args.llama_receipt)
    if llama_receipt["source_commit"] != base.PINNED_LLAMA_COMMIT:
        raise SystemExit("llama comparator source commit is not the pinned v0.1 commit")
    llama = base.checked_file(args.llama_server, "llama-server")
    muser_parts = base.loopback_origin(args.muser_url)
    llama_parts = base.loopback_origin(args.llama_url)
    if args.llama_context is not None and args.llama_context < args.prompt_tokens + args.output_tokens:
        raise SystemExit("--llama-context must cover prompt tokens plus output tokens")
    base.validate_free_ports([muser_parts, llama_parts])
    base.validate_output_path(args.output)
    plan = {
        "schema": "muser.representative-dflash-smoke.v1",
        "status": "checked" if args.check or args.dry_run else "running",
        "notarial": False,
        "seal_eligible": False,
        "accelerator_touched": False,
        "identity": args.identity,
        "cell": {
            "prompt_tokens": args.prompt_tokens,
            "output_tokens": args.output_tokens,
            "concurrency": 1,
            "target_only": False,
            "speculative_type": "draft-dflash",
            "verify_length": args.verify_length,
            "llama_context": args.llama_context,
        },
        "artifacts": {
            "model": model,
            "dflash": dflash,
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
            "sample": "one 2048+256 uncached streamed request per engine, DFlash route",
            "ttft": "request-send-complete to first raw token SSE event",
            "prefill_decode": "server final timing object",
            "stop_contract": "no ignore_eos on either engine; EOS stops both lanes",
            "muser_telemetry": (
                "authenticated /snapshot before/after the measured request; monotonic phase "
                "counters are differenced including dflash_draft/dflash_target_verify"
            ),
        },
        "preflight": {
            "ports_available": [args.muser_url, args.llama_url],
            "output_writable": str(args.output),
            "required_environment": {
                "MUSER_GGML_METALLIB": str(args.muser_metallib.resolve()),
                "MUSER_DFLASH_VERIFY_LEN": str(args.verify_length),
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
            "cache_prompt": False,
            "return_tokens": True,
            "stream": True,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    def run(engine: str, binary: Path, metallib: Path | None, parts: object, log: Path):
        token = base.secrets.token_hex(32)
        api_key = base.secrets.token_hex(32) if engine == "muser" else None
        api_key_path: Path | None = None
        if api_key is not None:
            import tempfile

            descriptor, name = tempfile.mkstemp(
                prefix=".representative-dflash-api-key.", dir=log.parent
            )
            api_key_path = Path(name)
            with os.fdopen(descriptor, "w", encoding="ascii") as stream:
                stream.write(api_key)
                stream.flush()
                os.fsync(stream.fileno())
        try:
            command, environment = spec_server_command(
                engine, binary, args.model, args.dflash, metallib, parts, token,
                args.server_deadline_seconds, api_key_path, args.llama_context,
                args.verify_length,
            )
        except BaseException:
            if api_key_path is not None:
                api_key_path.unlink(missing_ok=True)
            raise
        run_error: BaseException | None = None
        result = None
        telemetry_delta = None
        try:
            with log.open("xb") as log_stream:
                process = base.subprocess.Popen(
                    command,
                    stdin=base.subprocess.DEVNULL,
                    stdout=log_stream,
                    stderr=base.subprocess.STDOUT,
                    env=environment,
                )
                try:
                    base.wait_ready(parts, engine, process, args.timeout_seconds)
                    warmup_payload = json.loads(payload)
                    warmup_payload["n_predict"] = 1
                    base.stream_request(
                        parts,
                        json.dumps(warmup_payload, sort_keys=True, separators=(",", ":")).encode(),
                        args.timeout_seconds,
                    )
                    before = (
                        base.snapshot_request(parts, api_key, args.timeout_seconds)
                        if api_key is not None
                        else None
                    )
                    result = base.stream_request(parts, payload, args.timeout_seconds)
                    if api_key is not None:
                        after = base.snapshot_request(parts, api_key, args.timeout_seconds)
                        telemetry_delta = base.request_telemetry_delta(
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
                            base.cooperative_shutdown(parts, token)
                        except BaseException as error:
                            if run_error is None:
                                run_error = error
                    return_code = process.wait()
                    log_stream.flush()
                    os.fsync(log_stream.fileno())
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
            "generated_tokens_sha256": base.token_digest(tokens),
            "command": command,
            "server_log": str(log),
        }
        if telemetry_delta is not None:
            spec_phases = {}
            for name in SPEC_PHASES:
                before_view = base.telemetry_snapshot_view(before)["phases"][name]
                after_view = base.telemetry_snapshot_view(after)["phases"][name]
                samples = after_view["samples"] - before_view["samples"]
                total_ms = after_view["total_ms"] - before_view["total_ms"]
                if samples < 0 or total_ms < -1e-9:
                    raise RuntimeError(f"snapshot spec phase counter regressed: {name}")
                spec_phases[name] = {
                    "samples": samples,
                    "total_ms": max(0.0, total_ms),
                    "mean_ms": max(0.0, total_ms) / samples if samples else None,
                }
            if engine == "muser" and spec_phases["dflash_draft"]["samples"] <= 0:
                raise RuntimeError("muser DFlash route did not activate during the measured request")
            telemetry_delta["spec_phases"] = spec_phases
            output["telemetry_delta"] = telemetry_delta
        return output

    muser_result = run(
        "muser", args.muser_server, args.muser_metallib, muser_parts,
        args.output.with_suffix(".muser-server.log"),
    )
    llama_result = run(
        "llama", args.llama_server, None, llama_parts,
        args.output.with_suffix(".llama-server.log"),
    )
    muser_tokens = muser_result.pop("generated_tokens")
    llama_tokens = llama_result.pop("generated_tokens")
    exact = muser_tokens == llama_tokens
    digest = base.token_digest(muser_tokens)
    target_only_match = (
        args.target_only_digest is None or digest == args.target_only_digest
    )
    status = "passed" if exact and target_only_match else "failed"
    report = {
        **plan,
        "status": status,
        "accelerator_touched": True,
        "exact_tokens": exact,
        "target_only_digest_expected": args.target_only_digest,
        "target_only_digest_match": target_only_match,
        "output_tokens_generated": len(muser_tokens),
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
    base.atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
