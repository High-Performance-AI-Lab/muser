#!/usr/bin/env python3
"""Score a closed fixture manifest on one pinned native or exact vLLM engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import resident_producer as resident


SCHEMA = "muser.nvfp4-drift-fixtures.v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_digest(tokens: list[int]) -> str:
    return hashlib.sha256(
        b"".join(token.to_bytes(4, "little") for token in tokens)
    ).hexdigest()


def read_tokens(path: Path) -> list[int]:
    tokens = [int(value) for value in path.read_bytes().split()]
    if len(tokens) < 2 or any(not 0 <= token < 202048 for token in tokens):
        raise ValueError(f"invalid token fixture {path}")
    return tokens


def extract_prompt_rows(output: Any, tokens: list[int]) -> tuple[list[float], list[int]]:
    rows = output.prompt_logprobs
    if not isinstance(rows, list) or len(rows) != len(tokens):
        raise RuntimeError("vLLM prompt-logprob rows do not match the fixture")
    target_logprobs: list[float] = []
    top_token_ids: list[int] = []
    for position, target in enumerate(tokens[1:], 1):
        row = rows[position]
        if not isinstance(row, dict) or target not in row:
            raise RuntimeError(f"prompt-logprob row {position} omits target token {target}")
        target_value = float(row[target].logprob)
        if not math.isfinite(target_value):
            raise RuntimeError(f"prompt-logprob row {position} is non-finite")
        top_token, _ = max(row.items(), key=lambda item: float(item[1].logprob))
        target_logprobs.append(target_value)
        top_token_ids.append(int(top_token))
    return target_logprobs, top_token_ids


def load_manifest(
    path: Path, max_model_len: int, teacher_forced_only: bool = False
) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != {"fixtures", "schema"}:
        raise ValueError("fixture manifest has an unknown shape")
    if value.get("schema") != SCHEMA or not isinstance(value.get("fixtures"), list):
        raise ValueError(f"fixture manifest must use {SCHEMA}")
    seen: set[str] = set()
    loaded: list[dict[str, Any]] = []
    for entry in value["fixtures"]:
        required = {"id", "regime", "token_file", "output_tokens"}
        optional = {"document", "context_length"}
        if (
            not isinstance(entry, dict)
            or not required.issubset(entry)
            or set(entry) - required - optional
        ):
            raise ValueError("fixture entry has an unknown shape")
        fixture_id = entry["id"]
        if (
            not isinstance(fixture_id, str)
            or not fixture_id
            or fixture_id in seen
            or not fixture_id.replace("-", "").isalnum()
        ):
            raise ValueError("fixture id is invalid or duplicated")
        seen.add(fixture_id)
        output_tokens = entry["output_tokens"]
        allowed_output_tokens = {0, 256} if teacher_forced_only else {256}
        if output_tokens not in allowed_output_tokens:
            raise ValueError("drift fixture output-token contract differs from evaluation mode")
        token_path = Path(entry["token_file"])
        if not token_path.is_absolute():
            token_path = path.parent / token_path
        tokens = read_tokens(token_path)
        if "context_length" in entry and entry["context_length"] != len(tokens):
            raise ValueError(f"fixture {fixture_id} context metadata differs")
        generated_token_limit = 1 if teacher_forced_only else output_tokens
        if len(tokens) + generated_token_limit > max_model_len:
            raise ValueError(f"fixture {fixture_id} exceeds the engine context")
        loaded.append(
            entry
            | {
                "token_path": token_path,
                "tokens": tokens,
            }
        )
    if not loaded:
        raise ValueError("fixture manifest is empty")
    return loaded


def score_fixture(
    engine: Any,
    fixture: dict[str, Any],
    sampling_params_type: Any,
    tokens_prompt_type: Any,
    generated_token_limit: int,
) -> dict[str, Any]:
    from muser_vllm.exact_attention import set_exact_attention_enabled

    tokens = fixture["tokens"]
    exact = resident.producer_mode() == "exact"
    started = time.perf_counter_ns()
    if exact:
        set_exact_attention_enabled(True)
    try:
        outputs = engine.generate(
            tokens_prompt_type(prompt_token_ids=tokens),
            sampling_params_type(
                temperature=0,
                max_tokens=generated_token_limit,
                ignore_eos=True,
                seed=0,
                prompt_logprobs=1,
                extra_args={"muser_startup_warmup": True},
            ),
            use_tqdm=False,
        )
    finally:
        if exact:
            set_exact_attention_enabled(False)
    output = outputs[0]
    generated = [int(token) for token in output.outputs[0].token_ids]
    if len(generated) != generated_token_limit:
        raise RuntimeError(f"fixture {fixture['id']} generated an incomplete stream")
    target_logprobs, top_token_ids = extract_prompt_rows(output, tokens)
    mean_nll = -sum(target_logprobs) / len(target_logprobs)
    boundaries = sorted({1, len(tokens) // 4, len(tokens) // 2, 3 * len(tokens) // 4, len(tokens) - 1})
    return {
        "id": fixture["id"],
        "regime": fixture["regime"],
        "token_file": str(fixture["token_path"]),
        "token_file_sha256": sha256(fixture["token_path"]),
        "token_count": len(tokens),
        "token_ids_sha256": token_digest(tokens),
        "output_tokens": len(generated),
        "generated_tokens": generated,
        "generated_tokens_sha256": token_digest(generated),
        "target_logprobs": target_logprobs,
        "teacher_forced_top_token_ids": top_token_ids,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "boundary_positions": boundaries,
        "total_ns": time.perf_counter_ns() - started,
    }


def write_exclusive(path: Path, value: object) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-revision")
    parser.add_argument("--checkpoint-artifact-sha256")
    parser.add_argument("--fixture-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--progress-dir",
        type=Path,
        help="append-only per-fixture checkpoints for bounded exact runs",
    )
    parser.add_argument("--lease-file", required=True, type=Path)
    parser.add_argument("--max-model-len", type=int, default=33_024)
    parser.add_argument("--max-num-batched-tokens", type=int, default=33_024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=3 << 30)
    parser.add_argument(
        "--teacher-forced-only",
        action="store_true",
        help="generate one bookkeeping token; score only prompt rows",
    )
    args = parser.parse_args()
    args.startup_dummy = False
    args.startup_only = False
    args.disable_kv_connector = True
    fixtures = load_manifest(
        args.fixture_manifest, args.max_model_len, args.teacher_forced_only
    )
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config = resident.load_config(Path(args.config))
    overrides = (args.checkpoint_revision, args.checkpoint_artifact_sha256)
    if any(overrides) and not all(overrides):
        parser.error("checkpoint identity overrides must be supplied together")
    if all(overrides):
        if len(args.checkpoint_artifact_sha256) != 64 or any(
            value not in "0123456789abcdef" for value in args.checkpoint_artifact_sha256
        ):
            parser.error("checkpoint artifact override must be lowercase SHA-256")
        config = config | {
            "checkpoint_revision": args.checkpoint_revision,
            "checkpoint_artifact_sha256": args.checkpoint_artifact_sha256,
        }
    if args.progress_dir is not None:
        args.progress_dir.mkdir(mode=0o700)
    lease = resident.acquire_accelerator_lease(args.lease_file)
    from vllm import SamplingParams, TokensPrompt

    started = time.perf_counter_ns()
    engine, quantizer, fp4_mm, rms_norm, swiglu, attention, native_capture = (
        resident.build_engine(args, config)
    )
    generated_token_limit = 1 if args.teacher_forced_only else 256
    results = []
    for fixture_index, fixture in enumerate(fixtures):
        result = score_fixture(
            engine,
            fixture,
            SamplingParams,
            TokensPrompt,
            generated_token_limit,
        )
        results.append(result)
        if args.progress_dir is not None:
            checkpoint = {
                "schema": "muser.spark-nvfp4-drift-score-progress.v1",
                "producer_mode": resident.producer_mode(),
                "checkpoint_artifact_sha256": config["checkpoint_artifact_sha256"],
                "checkpoint_revision": config["checkpoint_revision"],
                "vllm_commit": resident.PINNED_VLLM_COMMIT,
                "fixture_manifest_sha256": sha256(args.fixture_manifest),
                "evaluation_mode": (
                    "teacher-forced-only"
                    if args.teacher_forced_only
                    else "teacher-forced-and-greedy"
                ),
                "fixture": result,
                "seal_eligible": False,
            }
            checkpoint_path = args.progress_dir / f"{fixture_index:02d}-{fixture['id']}.json"
            write_exclusive(checkpoint_path, checkpoint)
            print(
                json.dumps(
                    {
                        "checkpoint": str(checkpoint_path),
                        "fixture": fixture["id"],
                        "total_ns": result["total_ns"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    report = {
        "schema": "muser.spark-nvfp4-drift-score.v1",
        "created_unix_ms": time.time_ns() // 1_000_000,
        "producer_mode": resident.producer_mode(),
        "checkpoint_artifact_sha256": config["checkpoint_artifact_sha256"],
        "checkpoint_revision": config["checkpoint_revision"],
        "vllm_commit": resident.PINNED_VLLM_COMMIT,
        "fixture_manifest": str(args.fixture_manifest),
        "fixture_manifest_sha256": sha256(args.fixture_manifest),
        "engine": {
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "seed": 0,
            "temperature": 0,
            "prefix_caching": False,
            "chunked_prefill": False,
            "kv_connector": None,
            "evaluation_mode": (
                "teacher-forced-only" if args.teacher_forced_only else "teacher-forced-and-greedy"
            ),
        },
        "route": {
            "activation_quantizer": quantizer,
            "fp4_mm": fp4_mm,
            "rms_norm": rms_norm,
            "swiglu": swiglu,
            "attention": attention,
            "native_capture": native_capture,
        },
        "fixtures": results,
        "total_ns": time.perf_counter_ns() - started,
        "seal_eligible": False,
    }
    write_exclusive(args.output, report)
    print(json.dumps({
        "schema": report["schema"],
        "producer_mode": report["producer_mode"],
        "fixtures": [{"id": row["id"], "total_ns": row["total_ns"]} for row in results],
        "output": str(args.output),
    }, sort_keys=True))
    lease.close()


if __name__ == "__main__":
    main()
