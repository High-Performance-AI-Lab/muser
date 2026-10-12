#!/usr/bin/env python3
"""Screen warm prefix-cached native NVFP4 speculative verification shapes.

This is an upper-bound performance screen, not a serving implementation.  Each request
reuses one committed prompt prefix, teacher-forces a fresh draft suffix, asks
vLLM for prompt log probabilities (forcing an LM-head decision at every draft
position), and generates the all-accepted bonus token.  The measured wall time
therefore includes scheduler and result materialization overhead that a fused
verifier RPC would eventually remove.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any


SCHEMA = "muser.spark-native-nvfp4-verifier-benchmark.v1"
VOCAB_SIZE = 202_048


def read_tokens(path: Path) -> list[int]:
    tokens = [int(value) for value in path.read_text().split()]
    if len(tokens) < 16:
        raise ValueError("verifier fixture requires at least sixteen tokens")
    if any(token < 0 or token >= VOCAB_SIZE for token in tokens):
        raise ValueError("verifier fixture has an out-of-vocabulary token")
    return tokens


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_exclusive(path: Path, value: object) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")


def coefficient_of_variation(values: list[int]) -> float:
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else math.inf


def metric_snapshot(output: Any) -> dict[str, int | float]:
    metrics = getattr(output, "metrics", None)
    if metrics is None:
        return {}
    captured: dict[str, int | float] = {}
    for name in (
        "arrival_time",
        "first_scheduled_time",
        "first_token_time",
        "finished_time",
        "scheduler_time",
        "model_forward_time",
        "model_execute_time",
    ):
        value = getattr(metrics, name, None)
        if isinstance(value, (int, float)) and math.isfinite(value):
            captured[name] = value
    return captured


def engine_quantization_config(model: Path) -> dict[str, Any] | None:
    config_path = model / "config.json"
    if not config_path.is_file():
        return None
    config = json.loads(config_path.read_text())
    quantization = config.get("quantization_config")
    return quantization if isinstance(quantization, dict) else None


def fresh_suffix(tokens: list[int], candidate_count: int, repetition: int) -> list[int]:
    # A distinct first suffix token prevents a previous speculative branch from
    # becoming a longer cache hit.  Remaining tokens retain fixture-like IDs.
    salt = candidate_count * 10_007 + (repetition + 1) * 1_009
    return [
        (tokens[(salt + index) % len(tokens)] + salt + 97 * index) % VOCAB_SIZE
        for index in range(candidate_count)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--checkpoint-artifact-sha256", required=True)
    parser.add_argument("--draft-lengths", default="3,7,15")
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument(
        "--capture-hidden",
        action="store_true",
        help="capture and materialize the five f32 DFlash target layers",
    )
    parser.add_argument(
        "--composite-bundle",
        type=Path,
        help="authenticated RedHat portable-KV genesis imported before timing",
    )
    parser.add_argument("--composite-hmac-key-file", type=Path)
    parser.add_argument("--composite-hmac-key-id")
    parser.add_argument("--source-checkpoint-revision")
    parser.add_argument("--source-checkpoint-artifact-sha256")
    args = parser.parse_args()

    try:
        draft_lengths = [int(value) for value in args.draft_lengths.split(",")]
    except ValueError:
        parser.error("draft lengths must be comma-separated integers")
    if not draft_lengths or any(length < 1 or length > 64 for length in draft_lengths):
        parser.error("draft lengths must be inside 1..=64")
    if len(set(draft_lengths)) != len(draft_lengths):
        parser.error("draft lengths must be unique")
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    if not args.model.is_dir() or not args.fixture.is_file():
        parser.error("model and fixture must exist")
    if (
        args.prompt_tokens < 16
        or args.prompt_tokens + max(draft_lengths) + 1 > args.max_model_len
    ):
        parser.error("prompt plus draft exceeds the configured model length")
    if not 0.1 <= args.gpu_memory_utilization <= 0.95:
        parser.error("GPU memory utilization is outside the closed safe range")
    if len(args.checkpoint_artifact_sha256) != 64 or any(
        value not in "0123456789abcdef"
        for value in args.checkpoint_artifact_sha256
    ):
        parser.error("checkpoint artifact digest must be lowercase SHA-256")
    composite_values = (
        args.composite_bundle,
        args.composite_hmac_key_file,
        args.composite_hmac_key_id,
        args.source_checkpoint_revision,
        args.source_checkpoint_artifact_sha256,
    )
    if any(value is not None for value in composite_values) and not all(
        value is not None for value in composite_values
    ):
        parser.error("all composite genesis arguments must be supplied together")
    if args.composite_bundle is not None:
        if not args.composite_bundle.is_dir() or not args.composite_bundle.is_absolute():
            parser.error("composite bundle must be an existing absolute directory")
        if not args.composite_hmac_key_file.is_file():
            parser.error("composite HMAC key does not exist")
        if len(args.source_checkpoint_artifact_sha256) != 64 or any(
            value not in "0123456789abcdef"
            for value in args.source_checkpoint_artifact_sha256
        ):
            parser.error("source checkpoint artifact must be lowercase SHA-256")

    tokens = read_tokens(args.fixture)
    if len(tokens) < args.prompt_tokens:
        parser.error("fixture is shorter than the requested committed prompt")
    committed = tokens[: args.prompt_tokens]

    if os.environ.get("MUSER_NVFP4_EXACT") == "1":
        parser.error("native verifier benchmark refuses MUSER_NVFP4_EXACT=1")
    os.environ["MUSER_NVFP4_EXACT"] = "0"
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch
    import transformers
    import vllm
    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.config import KVTransferConfig

    native_capture = None
    if args.capture_hidden:
        from muser_vllm.native_capture import install_native_capture

        native_capture = install_native_capture()

    transfer = None
    composite_manifest = None
    if args.composite_bundle is not None:
        from muser_vllm.composite_bundle import (
            bundle_root_sha256,
            load_hmac_key,
            read_bundle_manifest,
        )

        composite_manifest = read_bundle_manifest(
            args.composite_bundle,
            key=load_hmac_key(args.composite_hmac_key_file),
            expected_key_id=args.composite_hmac_key_id,
            expected_source_artifact_sha256=args.source_checkpoint_artifact_sha256,
        )
        if composite_manifest["token_ids"] != committed:
            parser.error("composite bundle transcript differs from benchmark fixture")
        transfer = KVTransferConfig(
            kv_connector="MuserCompositeKvConnector",
            kv_role="kv_consumer",
            kv_connector_module_path="muser_vllm.composite_connector",
            kv_connector_extra_config={
                "bundle_path": str(args.composite_bundle),
                "hmac_key_file": str(args.composite_hmac_key_file),
                "hmac_key_id": args.composite_hmac_key_id,
                "mode": "import",
                "source_checkpoint_artifact_sha256": args.source_checkpoint_artifact_sha256,
                "source_checkpoint_revision": args.source_checkpoint_revision,
                "source_engine_mode": "native",
            },
        )

    load_started_ns = time.perf_counter_ns()
    engine = LLM(
        model=str(args.model),
        tokenizer=str(args.tokenizer or args.model),
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
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_memory_bytes=1 << 30,
        kv_transfer_config=transfer,
        seed=0,
    )
    engine_load_ns = time.perf_counter_ns() - load_started_ns

    fill = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True, seed=0)
    verifier = SamplingParams(
        temperature=0,
        max_tokens=1,
        ignore_eos=True,
        prompt_logprobs=1,
        logprobs=1,
        # vLLM defaults this to true whenever prompt_logprobs is requested.
        # The verifier only needs rows at/after the already-witnessed carried
        # frontier, so explicitly retain the committed-prefix cache hit.
        skip_reading_prefix_cache=False,
        seed=0,
    )

    # Materialize the committed prefix in the block cache.
    fill_output = engine.generate(
        TokensPrompt(prompt_token_ids=committed), fill, use_tqdm=False
    )[0]
    torch.cuda.synchronize()
    genesis_cached_tokens = getattr(fill_output, "num_cached_tokens", None)
    if composite_manifest is not None and genesis_cached_tokens != args.prompt_tokens - 1:
        raise RuntimeError(
            "composite verifier genesis did not import the exact external KV cut: "
            f"got {genesis_cached_tokens!r}, expected {args.prompt_tokens - 1}"
        )

    cells: list[dict[str, Any]] = []
    for draft_length in draft_lengths:
        candidate_count = draft_length + 1  # carried target frontier + drafts
        samples: list[dict[str, Any]] = []
        # The negative repetition is a same-shape warmup with a unique branch.
        for repetition in range(-1, args.repetitions):
            suffix = fresh_suffix(tokens, candidate_count, repetition)
            prompt = committed + suffix
            capture_path = None
            if args.capture_hidden:
                from muser_vllm.dflash_capture import begin_capture

                capture_path = args.output.parent / (
                    f".{args.output.name}.d{draft_length}.r{repetition}.f32"
                )
                begin_capture(
                    f"verifier-d{draft_length}-r{repetition}",
                    candidate_count,
                    capture_path,
                    device="cuda",
                )
            torch.cuda.synchronize()
            started_ns = time.perf_counter_ns()
            try:
                outputs = engine.generate(
                    TokensPrompt(prompt_token_ids=prompt), verifier, use_tqdm=False
                )
                capture_receipt = None
                capture_finish_ns = 0
                if args.capture_hidden:
                    from muser_vllm.dflash_capture import finish_capture

                    capture_finish_started_ns = time.perf_counter_ns()
                    # Serving sends the pinned-host payload directly. Keep the
                    # D2H copy, transpose, and digest in the measured region,
                    # but do not fsync a disposable benchmark file.
                    capture_receipt = finish_capture(materialize=False)
                    capture_finish_ns = (
                        time.perf_counter_ns() - capture_finish_started_ns
                    )
                torch.cuda.synchronize()
            except BaseException:
                if args.capture_hidden:
                    from muser_vllm.dflash_capture import abort_capture

                    abort_capture()
                raise
            elapsed_ns = time.perf_counter_ns() - started_ns
            if capture_receipt is not None:
                expected_capture_bytes = candidate_count * 5 * 6656 * 4
                if (
                    capture_receipt["bytes"] != expected_capture_bytes
                    or capture_receipt["cached_tokens"] != candidate_count
                ):
                    raise RuntimeError("DFlash hidden capture geometry differs")
                if capture_receipt.get("materialized") is not False:
                    raise RuntimeError("verifier benchmark unexpectedly materialized capture")
                if capture_path is None or capture_path.exists():
                    raise RuntimeError("verifier benchmark left a capture file behind")
            output = outputs[0]
            generated = list(output.outputs[0].token_ids)
            if len(generated) != 1:
                raise RuntimeError("verifier request did not return one bonus token")
            prompt_logprobs = output.prompt_logprobs
            if prompt_logprobs is None:
                raise RuntimeError("verifier request omitted prompt log probabilities")
            # vLLM intentionally returns fewer prompt-logprob entries after a
            # prefix-cache hit. Candidate zero is the already-witnessed carried
            # frontier; only the D subsequent draft decisions must be fresh.
            materialized_prompt_rows = sum(row is not None for row in prompt_logprobs)
            if (
                len(prompt_logprobs) != candidate_count
                or prompt_logprobs[0] is not None
                or any(row is None for row in prompt_logprobs[1:])
                or materialized_prompt_rows != draft_length
            ):
                raise RuntimeError(
                    "cached verifier prompt-logprob geometry differs: "
                    f"len={len(prompt_logprobs)}, materialized="
                    f"{materialized_prompt_rows}, expected=[None, {draft_length} rows]"
                )
            if output.outputs[0].logprobs is None:
                raise RuntimeError("verifier request omitted the bonus target row")
            if repetition < 0:
                continue
            cached = getattr(output, "num_cached_tokens", None)
            if not isinstance(cached, int) or cached != args.prompt_tokens:
                raise RuntimeError(
                    f"prefix cache reused {cached!r} tokens, expected exactly "
                    f"the {args.prompt_tokens}-token authenticated parent cut"
                )
            sample = {
                "repetition": repetition,
                "wall_ns": elapsed_ns,
                "draft_length": draft_length,
                "candidate_count": candidate_count,
                "target_sample_count": draft_length + 2,
                "full_accept_tokens_per_second": candidate_count * 1e9 / elapsed_ns,
                "bonus_token_id": generated[0],
                "num_cached_tokens": cached,
                "materialized_prompt_logprob_rows": materialized_prompt_rows,
                "hidden_capture": capture_receipt,
                "hidden_capture_finish_ns": capture_finish_ns,
                "metrics": metric_snapshot(output),
            }
            samples.append(sample)
            print(
                "[muser-native-nvfp4-verifier] "
                f"draft={draft_length} rep={repetition} "
                f"wall={elapsed_ns / 1e6:.3f}ms "
                f"full_accept_tps={sample['full_accept_tokens_per_second']:.2f} "
                f"cached={sample['num_cached_tokens']}",
                flush=True,
            )
        walls = [int(sample["wall_ns"]) for sample in samples]
        cells.append(
            {
                "draft_length": draft_length,
                "candidate_count": candidate_count,
                "target_sample_count": draft_length + 2,
                "samples": samples,
                "median_wall_ns": int(statistics.median(walls)),
                "range_wall_ns": [min(walls), max(walls)],
                "cv": coefficient_of_variation(walls),
                "median_full_accept_tokens_per_second": statistics.median(
                    float(sample["full_accept_tokens_per_second"])
                    for sample in samples
                ),
            }
        )

    receipt = {
        "schema": SCHEMA,
        "created_unix_ms": time.time_ns() // 1_000_000,
        "checkpoint": {
            "path": str(args.model),
            "revision": args.checkpoint_revision,
            "artifact_sha256": args.checkpoint_artifact_sha256,
            "quantization_config": engine_quantization_config(args.model),
        },
        "genesis": {
            "kind": "redhat_portable_kv_import" if composite_manifest is not None else "local_prefill",
            "num_cached_tokens": genesis_cached_tokens,
            "bundle_root_sha256": (
                bundle_root_sha256(composite_manifest)
                if composite_manifest is not None
                else None
            ),
            "source_checkpoint_revision": args.source_checkpoint_revision,
            "source_checkpoint_artifact_sha256": args.source_checkpoint_artifact_sha256,
        },
        "fixture": {
            "path": str(args.fixture),
            "sha256": sha256_file(args.fixture),
            "committed_prompt_tokens": args.prompt_tokens,
        },
        "engine": {
            "mode": "native",
            "selection": "stock-vllm-native-tensor-core",
            "prefix_caching": True,
            "prompt_logprobs": 1,
            "bonus_logprobs": 1,
            "skip_reading_prefix_cache": False,
            "native_hidden_capture": native_capture,
            "engine_load_ns": engine_load_ns,
        },
        "runtime": {
            "gpu": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "vllm": vllm.__version__,
        },
        "cells": cells,
        "interpretation": {
            "kind": "upper_bound_screen_not_hardware_kill_gate",
            "includes": [
                "scheduler",
                "greedy_teacher_forced_lm_head",
                "top1_prompt_logprob_materialization",
                "one_bonus_token",
            ]
            + (
                ["five-layer f32 hidden capture and materialization"]
                if args.capture_hidden
                else []
            ),
            "excludes": [
                "network",
                "round authentication",
                "KV mirror delta",
                "sampled top-k rows and maximal-coupling arithmetic",
            ]
            + ([] if args.capture_hidden else ["DFlash hidden capture"]),
            "limitation": (
                "This is a greedy/all-row LM-head timing screen. It does not "
                "qualify sampled sparse-maximal verification or output correctness."
            ),
        },
    }
    write_exclusive(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
