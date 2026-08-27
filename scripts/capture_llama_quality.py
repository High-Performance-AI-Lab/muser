#!/usr/bin/env python3
"""Capture one exact 2048->256 llama target or DFlash P0 quality stream."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import representative_target_smoke as base
from nvfp4_to_f16_gguf import write_receipt


PINNED_LLAMA_COMMIT = "89e0aa6fd362617d9073e0dafc18e41241521572"


def token_file(path: Path) -> list[int]:
    try:
        values = [int(value) for value in path.read_bytes().split()]
    except (OSError, ValueError) as error:
        raise RuntimeError(f"invalid token fixture {path}: {error}") from error
    if not values or any(not 0 <= value <= 0xFFFFFFFF for value in values):
        raise RuntimeError(f"empty or out-of-range token fixture: {path}")
    return values


def write_tokens(path: Path, tokens: list[int]) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to replace token fixture: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = ("\n".join(map(str, tokens)) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--model-receipt", type=Path)
    parser.add_argument("--dflash", type=Path)
    parser.add_argument("--expected-dflash-sha256")
    parser.add_argument("--expected-tokens", type=Path)
    parser.add_argument("--prompt-token-fixture", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--llama-server", type=Path, required=True)
    parser.add_argument("--llama-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens-out", type=Path, required=True)
    parser.add_argument("--combined-out", type=Path, required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--server-deadline-seconds", type=int, default=3600)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompt = token_file(args.prompt_token_fixture)
    if len(prompt) != args.prompt_tokens:
        raise SystemExit("prompt fixture does not match --prompt-tokens")
    if args.output_tokens != 256:
        raise SystemExit("P0 standard stream requires exactly 256 output tokens")
    if (args.dflash is None) != (args.expected_dflash_sha256 is None):
        raise SystemExit("--dflash and --expected-dflash-sha256 are required together")

    model = base.checked_file(args.model, "target model")
    if model["sha256"] != args.expected_model_sha256:
        raise SystemExit("target model differs from --expected-model-sha256")
    model_receipt = None
    if args.model_receipt is not None:
        model_receipt = base.checked_file(args.model_receipt, "model receipt")
        receipt = json.loads(args.model_receipt.read_text())
        if (
            receipt.get("schema") != "muser.nvfp4-f16-gguf-repack.v4"
            or Path(receipt.get("out", "")).resolve() != args.model.resolve()
            or receipt.get("sha256") != model["sha256"]
        ):
            raise SystemExit("model receipt does not bind the target model")
    dflash = None
    if args.dflash is not None:
        dflash = base.checked_file(args.dflash, "DFlash model")
        if dflash["sha256"] != args.expected_dflash_sha256:
            raise SystemExit("DFlash model differs from --expected-dflash-sha256")
    expected_tokens = None
    expected_tokens_artifact = None
    if args.expected_tokens is not None:
        expected_tokens_artifact = base.checked_file(
            args.expected_tokens, "expected token fixture"
        )
        expected_tokens = token_file(args.expected_tokens)
        if len(expected_tokens) != args.output_tokens:
            raise SystemExit("expected token fixture does not match --output-tokens")
    llama_receipt, llama_receipt_file = base.validate_comparator(
        args.llama_server, args.llama_receipt
    )
    if llama_receipt["source_commit"] != PINNED_LLAMA_COMMIT:
        raise SystemExit("llama comparator is not the pinned source commit")
    llama = base.checked_file(args.llama_server, "llama-server")
    prompt_artifact = base.checked_file(args.prompt_token_fixture, "prompt fixture")
    parts = base.loopback_origin(args.url)
    base.validate_free_ports([parts])
    base.validate_output_path(args.output)
    base.validate_output_path(args.tokens_out)
    base.validate_output_path(args.combined_out)

    extra_command: tuple[str, ...] = ()
    if args.dflash is not None:
        extra_command = (
            "-md", str(args.dflash),
            "--spec-type", "draft-dflash",
            "--spec-draft-n-max", "15",
            "--spec-draft-n-min", "15",
            "--spec-draft-p-min", "0",
        )
    plan: dict[str, object] = {
        "schema": "muser.llama-quality-capture.v1",
        "status": "checked" if args.check else "running",
        "seal_eligible": False,
        "identity": args.identity,
        "mode": "dflash" if args.dflash is not None else "target-only",
        "cell": {
            "prompt_tokens": args.prompt_tokens,
            "output_tokens": args.output_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
            "cache_prompt": False,
        },
        "artifacts": {
            "model": model,
            "model_receipt": model_receipt,
            "dflash": dflash,
            "expected_tokens": expected_tokens_artifact,
            "prompt_fixture": prompt_artifact,
            "llama_server": llama,
            "llama_receipt": llama_receipt_file,
            "llama_source_commit": llama_receipt["source_commit"],
        },
        "outputs": {
            "report": str(args.output.resolve()),
            "generated_tokens": str(args.tokens_out.resolve()),
            "combined_tokens": str(args.combined_out.resolve()),
        },
    }
    if args.check:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(
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
    result = base.run_engine(
        "llama",
        args.llama_server,
        args.model,
        None,
        parts,
        body,
        args.timeout_seconds,
        args.server_deadline_seconds,
        args.output.with_suffix(".server.log"),
        extra_command,
    )
    tokens = result.pop("generated_tokens")
    exact_expected = expected_tokens is None or tokens == expected_tokens
    timings = result["timings"]
    dflash_active = args.dflash is None or (
        isinstance(timings.get("draft_n"), int) and timings["draft_n"] > 0
    )
    complete = (
        len(tokens) == args.output_tokens
        and timings.get("prompt_n") == args.prompt_tokens
        and timings.get("predicted_n") == args.output_tokens
    )
    status = "passed" if complete and exact_expected and dflash_active else "failed"
    report = {
        **plan,
        "status": status,
        "accelerator_touched": True,
        "output_tokens_generated": len(tokens),
        "generated_tokens_sha256": base.token_digest(tokens),
        "combined_tokens_sha256": base.token_digest(prompt + tokens),
        "expected_tokens_match": exact_expected,
        "dflash_route_active": dflash_active,
        "llama": result,
    }
    write_tokens(args.tokens_out, tokens)
    write_tokens(args.combined_out, prompt + tokens)
    write_receipt(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
