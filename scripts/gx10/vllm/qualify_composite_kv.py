#!/usr/bin/env python3
"""Export or import the exact RedHat-prefix/Dudeman-target KV seam on GX10."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any


SCHEMA_EXPORT = "muser.spark-composite-kv-export.v1"
SCHEMA_IMPORT = "muser.spark-composite-kv-import.v1"
VOCAB_SIZE = 202_048


def read_tokens(path: Path, count: int) -> list[int]:
    tokens = [int(value) for value in path.read_text().split()]
    if len(tokens) < count:
        raise ValueError("composite fixture is shorter than the requested prompt")
    tokens = tokens[:count]
    if any(token < 0 or token >= VOCAB_SIZE for token in tokens):
        raise ValueError("composite fixture contains an out-of-vocabulary token")
    return tokens


def write_exclusive(path: Path, value: object) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def logprob_rows(rows: Any) -> list[Any] | None:
    if rows is None:
        return None
    result: list[Any] = []
    for row in rows:
        if row is None:
            result.append(None)
            continue
        if not isinstance(row, dict):
            raise RuntimeError(f"unexpected vLLM logprob row {type(row)!r}")
        entries = []
        for token, value in sorted(row.items()):
            logprob = float(value.logprob)
            if not math.isfinite(logprob):
                raise RuntimeError("vLLM returned a non-finite log probability")
            entries.append(
                {
                    "decoded_token": getattr(value, "decoded_token", None),
                    "logprob": logprob,
                    "rank": getattr(value, "rank", None),
                    "token": int(token),
                }
            )
        result.append(entries)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("export", "import"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--hmac-key-file", type=Path, required=True)
    parser.add_argument("--hmac-key-id", required=True)
    parser.add_argument("--source-checkpoint-revision", required=True)
    parser.add_argument("--source-checkpoint-artifact-sha256", required=True)
    parser.add_argument("--loaded-checkpoint-revision", required=True)
    parser.add_argument("--loaded-checkpoint-artifact-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-hidden", action="store_true")
    parser.add_argument("--oracle-capture", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument(
        "--append-tokens",
        type=Path,
        help="optional whitespace token IDs appended after the authenticated prompt",
    )
    parser.add_argument(
        "--append-token",
        action="append",
        type=int,
        default=[],
        help="token ID appended after the authenticated prompt; repeat for a block",
    )
    args = parser.parse_args()

    for path, kind in (
        (args.model, "model"),
        (args.fixture, "fixture"),
        (args.hmac_key_file, "HMAC key"),
    ):
        exists = path.is_dir() if kind == "model" else path.is_file()
        if not exists:
            parser.error(f"{kind} path does not exist")
    if not args.bundle.is_absolute() or not args.output.is_absolute():
        parser.error("bundle and output paths must be absolute")
    if args.prompt_tokens < 2 or args.prompt_tokens + 1 > 4096:
        parser.error("prompt token count is outside the qualifier context")
    if args.mode == "export" and args.bundle.exists():
        parser.error("export bundle already exists")
    if args.mode == "import" and not args.bundle.is_dir():
        parser.error("import bundle does not exist")
    if args.oracle_capture and (args.mode != "import" or args.capture_hidden):
        parser.error("oracle capture is import-only and mutually exclusive with hidden capture")
    if not 1 <= args.max_tokens <= 64:
        parser.error("max tokens must be inside 1..=64")
    for name in (
        "source_checkpoint_artifact_sha256",
        "loaded_checkpoint_artifact_sha256",
    ):
        value = getattr(args, name)
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            parser.error(f"{name} must be lowercase SHA-256")

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["MUSER_NVFP4_EXACT"] = "0"

    import torch
    import transformers
    import vllm
    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.config import KVTransferConfig

    from muser_vllm.composite_bundle import (
        bundle_root_sha256,
        load_hmac_key,
        read_bundle_manifest,
    )

    tokens = read_tokens(args.fixture, args.prompt_tokens)
    if args.append_tokens is not None and args.append_token:
        parser.error("append-tokens and append-token are mutually exclusive")
    appended: list[int] = list(args.append_token)
    if args.append_tokens is not None:
        if not args.append_tokens.is_file():
            parser.error("append token fixture does not exist")
        appended = [int(value) for value in args.append_tokens.read_text().split()]
    if len(appended) > 64:
        parser.error("append token list must contain at most 64 tokens")
    if any(token < 0 or token >= VOCAB_SIZE for token in appended):
        parser.error("append token list contains an out-of-vocabulary token")
    request_tokens = tokens + appended
    expected_oracle_rows = 1 + len(appended) + args.max_tokens - 1
    extra = {
        "bundle_path": str(args.bundle),
        "hmac_key_file": str(args.hmac_key_file),
        "hmac_key_id": args.hmac_key_id,
        "mode": args.mode,
        "source_checkpoint_artifact_sha256": args.source_checkpoint_artifact_sha256,
        "source_checkpoint_revision": args.source_checkpoint_revision,
        "source_engine_mode": "native",
    }
    transfer = KVTransferConfig(
        kv_connector="MuserCompositeKvConnector",
        kv_role="kv_producer" if args.mode == "export" else "kv_consumer",
        kv_connector_module_path="muser_vllm.composite_connector",
        kv_connector_extra_config=extra,
    )
    capture_install = None
    capture_path = args.output.with_suffix(".hidden.f32")
    if args.capture_hidden:
        from muser_vllm.native_capture import install_native_capture

        capture_install = install_native_capture()
    if args.oracle_capture:
        from muser_vllm.oracle_capture import install_oracle_capture

        capture_install = install_oracle_capture()

    load_started = time.perf_counter_ns()
    engine = LLM(
        model=str(args.model),
        tokenizer=str(args.model),
        load_format="safetensors",
        quantization=None,
        dtype="float16",
        kv_cache_dtype="auto",
        enforce_eager=True,
        enable_chunked_prefill=False,
        enable_prefix_caching=True,
        disable_hybrid_kv_cache_manager=True,
        enable_flashinfer_autotune=False,
        language_model_only=True,
        max_model_len=4096,
        max_num_batched_tokens=4096,
        max_num_seqs=1,
        gpu_memory_utilization=0.82,
        kv_cache_memory_bytes=1 << 30,
        kv_transfer_config=transfer,
        seed=0,
    )
    engine_load_ns = time.perf_counter_ns() - load_started

    if args.capture_hidden:
        from muser_vllm.dflash_capture import begin_capture

        capture_rows = 1 if args.mode == "import" else args.prompt_tokens
        begin_capture(
            f"composite-{args.mode}", capture_rows, capture_path, device="cuda"
        )
    if args.oracle_capture:
        from muser_vllm.oracle_capture import begin_capture as begin_oracle_capture

        begin_oracle_capture(
            "composite-target-oracle",
            expected_oracle_rows,
            args.output.with_suffix(".oracle"),
        )
    sampling = SamplingParams(
        temperature=0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
        # The pinned engine exposes at most twenty API logprobs. Full-logit
        # oracle capture is a separate hook; this qualifier first proves the
        # external KV cut and target transition itself.
        prompt_logprobs=20 if args.mode == "import" else None,
        logprobs=20 if args.mode == "import" else None,
        skip_reading_prefix_cache=False if args.mode == "import" else True,
        seed=0,
    )
    torch.cuda.synchronize()
    run_started = time.perf_counter_ns()
    try:
        outputs = engine.generate(
            TokensPrompt(prompt_token_ids=request_tokens), sampling, use_tqdm=False
        )
        capture_receipt = None
        if args.capture_hidden:
            from muser_vllm.dflash_capture import finish_capture

            capture_receipt = finish_capture()
        if args.oracle_capture:
            from muser_vllm.oracle_capture import finish_capture as finish_oracle_capture

            capture_receipt = finish_oracle_capture()
        torch.cuda.synchronize()
    except BaseException:
        if args.capture_hidden:
            from muser_vllm.dflash_capture import abort_capture

            abort_capture()
        if args.oracle_capture:
            from muser_vllm.oracle_capture import abort_capture as abort_oracle_capture

            abort_oracle_capture()
        raise
    run_ns = time.perf_counter_ns() - run_started

    output = outputs[0]
    generated = list(output.outputs[0].token_ids)
    if len(generated) != args.max_tokens:
        raise RuntimeError("composite qualifier produced the wrong number of tokens")
    key = load_hmac_key(args.hmac_key_file)
    manifest = read_bundle_manifest(
        args.bundle,
        key=key,
        expected_key_id=args.hmac_key_id,
        expected_source_artifact_sha256=args.source_checkpoint_artifact_sha256,
    )
    if manifest["token_ids"] != tokens or manifest["cached_token_count"] != len(tokens) - 1:
        raise RuntimeError("composite bundle transcript differs after engine execution")

    cached_tokens = getattr(output, "num_cached_tokens", None)
    if args.mode == "import" and cached_tokens != len(tokens) - 1:
        raise RuntimeError(
            f"composite import reused {cached_tokens!r} tokens, expected {len(tokens) - 1}"
        )
    if args.mode == "export" and cached_tokens not in (None, 0):
        raise RuntimeError("composite export unexpectedly reused a prefix")
    receipt = {
        "schema": SCHEMA_EXPORT if args.mode == "export" else SCHEMA_IMPORT,
        "created_unix_ms": time.time_ns() // 1_000_000,
        "mode": args.mode,
        "bundle": {
            "path": str(args.bundle),
            "root_sha256": bundle_root_sha256(manifest),
            "cached_token_count": manifest["cached_token_count"],
            "token_ids_sha256": manifest["token_ids_sha256"],
            "portable_kv_abi": manifest["portable_kv_abi"],
        },
        "source_checkpoint": {
            "revision": args.source_checkpoint_revision,
            "artifact_sha256": args.source_checkpoint_artifact_sha256,
        },
        "loaded_checkpoint": {
            "revision": args.loaded_checkpoint_revision,
            "artifact_sha256": args.loaded_checkpoint_artifact_sha256,
        },
        "engine": {
            "vllm": vllm.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "load_ns": engine_load_ns,
            "run_ns": run_ns,
            "num_cached_tokens": cached_tokens,
        },
        "request": {
            "append_tokens": appended,
            "expected_oracle_rows": expected_oracle_rows,
            "max_tokens": args.max_tokens,
        },
        "generated_tokens": generated,
        "prompt_logprobs": logprob_rows(output.prompt_logprobs),
        "output_logprobs": logprob_rows(output.outputs[0].logprobs),
        "hidden_capture": capture_receipt,
        "capture_install": capture_install,
    }
    write_exclusive(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
