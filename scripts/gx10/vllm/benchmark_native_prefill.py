#!/usr/bin/env python3
"""Benchmark stock vLLM native NVFP4 prefill without installing exact patches."""

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


SCHEMA = "muser.spark-native-nvfp4-prefill-benchmark.v1"
NOMINAL_PARAMETERS = 30_000_000_000


def read_tokens(path: Path) -> list[int]:
    tokens = [int(value) for value in path.read_text().split()]
    if len(tokens) < 2:
        raise ValueError("native benchmark fixture requires at least two tokens")
    if any(token < 0 or token >= 202_048 for token in tokens):
        raise ValueError("native benchmark fixture has an out-of-vocabulary token")
    return tokens


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def coefficient_of_variation(values: list[int]) -> float:
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else math.inf


def write_exclusive(path: Path, value: object) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")


def engine_quantization_config(model: Path) -> dict[str, Any] | None:
    config_path = model / "config.json"
    if not config_path.is_file():
        return None
    config = json.loads(config_path.read_text())
    quantization = config.get("quantization_config")
    return quantization if isinstance(quantization, dict) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--checkpoint-artifact-sha256", required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument(
        "--enforce-eager", dest="enforce_eager", action="store_true", default=True
    )
    parser.add_argument(
        "--no-enforce-eager", dest="enforce_eager", action="store_false"
    )
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    if args.prompt_tokens < 2 or args.prompt_tokens > args.max_model_len:
        parser.error("prompt token count is outside the configured model length")
    if not args.model.is_dir():
        parser.error("model must be a checkpoint directory")
    if not args.fixture.is_file():
        parser.error("fixture is not a regular file")
    if not 0.1 <= args.gpu_memory_utilization <= 0.95:
        parser.error("gpu memory utilization is outside the closed safe range")
    if len(args.checkpoint_artifact_sha256) != 64 or any(
        value not in "0123456789abcdef"
        for value in args.checkpoint_artifact_sha256
    ):
        parser.error("checkpoint artifact digest must be lowercase SHA-256")
    tokens = read_tokens(args.fixture)
    if len(tokens) < args.prompt_tokens:
        parser.error("fixture is shorter than the requested prompt")
    prompt = tokens[: args.prompt_tokens]

    # The benchmark is intentionally stock: importing muser_vllm exact modules
    # or setting MUSER_NVFP4_EXACT would invalidate the native-path claim.
    if os.environ.get("MUSER_NVFP4_EXACT") == "1":
        parser.error("native benchmark refuses MUSER_NVFP4_EXACT=1")
    os.environ["MUSER_NVFP4_EXACT"] = "0"
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch
    import transformers
    import vllm
    from vllm import LLM, SamplingParams, TokensPrompt

    started_ns = time.perf_counter_ns()
    engine = LLM(
        model=str(args.model),
        tokenizer=str(args.tokenizer or args.model),
        load_format="safetensors",
        quantization=None,
        dtype="float16",
        kv_cache_dtype="auto",
        enforce_eager=args.enforce_eager,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        disable_hybrid_kv_cache_manager=True,
        enable_flashinfer_autotune=False,
        language_model_only=True,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_memory_bytes=1 << 30,
        seed=0,
    )
    engine_load_ns = time.perf_counter_ns() - started_ns
    sampling = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True, seed=0)

    # One full-shape warmup keeps compilation and allocator growth out of the
    # selection samples while exercising the same scheduler/GEMM geometry.
    engine.generate(
        TokensPrompt(prompt_token_ids=prompt), sampling, use_tqdm=False
    )
    torch.cuda.synchronize()

    samples: list[dict[str, Any]] = []
    for repetition in range(args.repetitions):
        torch.cuda.synchronize()
        sample_started_ns = time.perf_counter_ns()
        outputs = engine.generate(
            TokensPrompt(prompt_token_ids=prompt), sampling, use_tqdm=False
        )
        torch.cuda.synchronize()
        elapsed_ns = time.perf_counter_ns() - sample_started_ns
        generated = list(outputs[0].outputs[0].token_ids)
        if len(generated) != 1:
            raise RuntimeError("native benchmark did not produce exactly one token")
        effective_tflops = (
            2.0
            * NOMINAL_PARAMETERS
            * float(args.prompt_tokens - 1)
            / float(elapsed_ns)
            / 1.0e3
        )
        samples.append(
            {
                "repetition": repetition,
                "generate_wall_ns": elapsed_ns,
                "effective_tflops": effective_tflops,
                "first_token_id": generated[0],
            }
        )
        print(
            "[muser-native-nvfp4-bench] "
            f"rep={repetition} wall={elapsed_ns / 1e9:.6f}s "
            f"effective_tflops={effective_tflops:.3f} token={generated[0]}",
            flush=True,
        )

    raw_ns = [int(sample["generate_wall_ns"]) for sample in samples]
    first_tokens = [int(sample["first_token_id"]) for sample in samples]
    payload = {
        "schema": SCHEMA,
        "created_unix_ms": time.time_ns() // 1_000_000,
        "producer_mode": "native",
        "checkpoint": {
            "path": str(args.model),
            "revision": args.checkpoint_revision,
            "artifact_sha256": args.checkpoint_artifact_sha256,
            "quantization_config": engine_quantization_config(args.model),
        },
        "fixture": {
            "path": str(args.fixture),
            "sha256": sha256_file(args.fixture),
            "prompt_tokens": args.prompt_tokens,
        },
        "engine": {
            "enforce_eager": args.enforce_eager,
            "chunked_prefill": False,
            "prefix_caching": False,
            "max_num_seqs": 1,
            "kv_cache_dtype": "auto",
            "dtype": "float16",
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
        "samples": samples,
        "summary": {
            "repetitions": args.repetitions,
            "median_generate_wall_ns": int(statistics.median(raw_ns)),
            "cv": coefficient_of_variation(raw_ns),
            "median_effective_tflops": statistics.median(
                float(sample["effective_tflops"]) for sample in samples
            ),
            "deterministic_first_token": len(set(first_tokens)) == 1,
            "first_token_id": first_tokens[0],
        },
        "seal_eligible": False,
    }
    write_exclusive(args.output, payload)
    print(json.dumps(payload["summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
