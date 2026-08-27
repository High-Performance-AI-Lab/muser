#!/usr/bin/env python3
"""Resident pinned Muse Glimmer NVFP4 prefill producer for DGX Spark."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import socket
import stat
import subprocess
import threading
import time
import traceback
from functools import partial
from pathlib import Path
from typing import Any

PINNED_VLLM_COMMIT = "6adad08767583f52eb4d2122111af0bf638ed5e6"
SCHEMA = "muser.spark-nvfp4-producer-config.v1"
REQUEST_SCHEMA = "muser.spark-nvfp4-prefill-request.v1"
WATCHDOG_SECONDS = int(os.environ.get("MUSER_VLLM_WATCHDOG_SECONDS", "900"))
EXPECTED_ROPE_MODULES = 39
EXPECTED_HEAD_SIZE = 128
EXPECTED_CONTEXT_LENGTH = 131072
DEFAULT_KV_CACHE_BYTES = 1 << 30


def producer_mode() -> str:
    """Return the closed producer lane selected for this process."""
    value = os.environ.get("MUSER_NVFP4_EXACT", "0")
    if value not in {"0", "1"}:
        raise RuntimeError("MUSER_NVFP4_EXACT must be exactly 0 or 1")
    return "exact" if value == "1" else "native"


def acquire_accelerator_lease(path: Path):
    """Hold the Spark host accelerator lease for the resident lifetime."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    wait_seconds = float(os.environ.get("MUSER_ACCELERATOR_LEASE_WAIT_SECONDS", "0"))
    if not 0.0 <= wait_seconds <= 60.0:
        handle.close()
        raise RuntimeError("accelerator lease wait must be between 0 and 60 seconds")
    deadline = time.monotonic() + wait_seconds
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() >= deadline:
                handle.close()
                raise RuntimeError(f"accelerator lease unavailable: {path}") from None
            time.sleep(0.1)
    return handle


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if not isinstance(config, dict) or config.get("schema") != SCHEMA:
        raise ValueError(f"producer config must use schema {SCHEMA}")
    if config.get("vllm_commit") != PINNED_VLLM_COMMIT:
        raise ValueError("producer config does not pin the qualified vLLM commit")
    required = {
        "checkpoint_artifact_sha256",
        "checkpoint_revision",
        "connector",
        "schema",
        "vllm_commit",
    }
    if set(config) != required:
        raise ValueError(
            f"producer config keys are {sorted(config)}, expected {sorted(required)}"
        )
    if not isinstance(config["connector"], dict):
        raise ValueError("producer connector config must be an object")
    for name in ("checkpoint_artifact_sha256",):
        value = config[name]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError(f"{name} is not a lowercase SHA-256")
    return config


def validate_request(value: object, max_model_len: int) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != REQUEST_SCHEMA:
        raise ValueError(f"request must use schema {REQUEST_SCHEMA}")
    required = {"handoff", "request_id", "schema", "token_ids"}
    if set(value) != required:
        raise ValueError(f"request keys are {sorted(value)}, expected {sorted(required)}")
    request_id = value["request_id"]
    if (
        not isinstance(request_id, str)
        or not 1 <= len(request_id) <= 128
        or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in request_id)
    ):
        raise ValueError("request_id is outside the closed identifier grammar")
    tokens = value["token_ids"]
    if not isinstance(tokens, list) or not 2 <= len(tokens) <= max_model_len:
        raise ValueError("token_ids length is outside the producer context")
    if any(not isinstance(token, int) or not 0 <= token < 202048 for token in tokens):
        raise ValueError("token_ids contain an out-of-vocabulary ID")
    handoff = value["handoff"]
    required_handoff = {"generation", "receiver_host", "receiver_port", "transfer_id"}
    dflash_handoff = {
        "dflash_session",
        "dflash_identity_sha256",
        "dflash_kv_heads",
        "dflash_head_dim",
        "dflash_context_layers",
        "dflash_context_elements_per_token",
        "dflash_context_sink_size",
        "dflash_context_window_size",
    }
    if not isinstance(handoff, dict) or set(handoff) not in (
        required_handoff,
        required_handoff | dflash_handoff,
        required_handoff | {"prefix_cut"},
        required_handoff | dflash_handoff | {"prefix_cut"},
    ):
        raise ValueError("handoff context does not match the closed target/DFlash schema")
    prefix_cut = handoff.get("prefix_cut", 0)
    if not isinstance(prefix_cut, int) or prefix_cut < 0 or prefix_cut % 256 != 0:
        raise ValueError("handoff prefix_cut must be a nonnegative 256-aligned integer")
    if not isinstance(handoff["generation"], int) or handoff["generation"] < 1:
        raise ValueError("handoff generation must be positive")
    if not isinstance(handoff["receiver_host"], str) or not handoff["receiver_host"]:
        raise ValueError("receiver_host must be non-empty")
    if not isinstance(handoff["receiver_port"], int) or not 1 <= handoff["receiver_port"] <= 65535:
        raise ValueError("receiver_port is invalid")
    if not isinstance(handoff["transfer_id"], str) or not 1 <= len(handoff["transfer_id"]) <= 256:
        raise ValueError("transfer_id is invalid")
    if dflash_handoff.issubset(handoff):
        session = handoff["dflash_session"]
        session_path = Path(session) if isinstance(session, str) else None
        deferred_root = os.environ.get("MUSER_DFLASH_SESSION_DIR")
        deferred_native = (
            producer_mode() == "native"
            and os.environ.get("MUSER_DFLASH_JOBS_FIFO")
            and deferred_root
            and session_path is not None
            and not session_path.exists()
            and session_path.parent.is_dir()
            and session_path.parent.resolve() == Path(deferred_root).resolve()
            and not session_path.is_symlink()
        )
        if session_path is None or (not session_path.is_file() and not deferred_native):
            raise ValueError("dflash_session is neither precomputed nor a native deferred output")
        identity = handoff["dflash_identity_sha256"]
        if (
            not isinstance(identity, str)
            or len(identity) != 64
            or any(c not in "0123456789abcdef" for c in identity)
        ):
            raise ValueError("dflash_identity_sha256 is not lowercase SHA-256")
        if handoff["dflash_kv_heads"] != 8 or handoff["dflash_head_dim"] != 128:
            raise ValueError("DFlash geometry differs from the qualified 8x128 contract")
        context_values = [
            handoff["dflash_context_layers"],
            handoff["dflash_context_elements_per_token"],
            handoff["dflash_context_sink_size"],
            handoff["dflash_context_window_size"],
        ]
        if (
            any(not isinstance(value, int) or value < 1 for value in context_values)
            or handoff["dflash_context_elements_per_token"]
            != handoff["dflash_kv_heads"] * handoff["dflash_head_dim"]
        ):
            raise ValueError("DFlash context geometry is invalid")
    return value


def build_engine(args: argparse.Namespace, config: dict[str, Any]):
    from vllm import LLM
    from vllm.config import KVTransferConfig

    mode = producer_mode()
    native_capture = {"active": False}
    if mode == "exact":
        from muser_vllm.exact_attention import install_exact_attention
        from muser_vllm.exact_fp4_mm import install_exact_fp4_mm
        from muser_vllm.exact_fp4_quant import install_exact_fp4_quantizer
        from muser_vllm.exact_rms_norm import install_exact_rms_norm
        from muser_vllm.exact_swiglu import install_exact_swiglu

        quantizer = install_exact_fp4_quantizer()
        fp4_mm = install_exact_fp4_mm()
        rms_norm = install_exact_rms_norm()
        swiglu = install_exact_swiglu()
        attention = install_exact_attention()
    else:
        from muser_vllm.native_capture import install_native_capture

        native_capture = install_native_capture()
        inactive = {
            "active": False,
            "producer_mode": mode,
            "selection": "stock-vllm-native-tensor-core",
        }
        quantizer = inactive | {"component": "activation_quantizer"}
        fp4_mm = inactive | {"component": "fp4_mm"}
        rms_norm = inactive | {"component": "rms_norm"}
        swiglu = inactive | {"component": "swiglu"}
        attention = inactive | {"component": "attention"}
    transfer = None
    if not getattr(args, "disable_kv_connector", False):
        transfer = KVTransferConfig(
            kv_connector="MuserMuseHandoffConnector",
            kv_role="kv_producer",
            kv_connector_module_path="muser_vllm.connector",
            kv_connector_extra_config=config["connector"],
        )
    engine = LLM(
        model=args.model,
        tokenizer=args.tokenizer or args.model,
        load_format="dummy" if args.startup_dummy else "safetensors",
        quantization=None,
        kv_cache_dtype="auto",
        dtype="float16",
        enforce_eager=True,
        enable_chunked_prefill=False,
        enable_prefix_caching=bool(getattr(args, "enable_prefix_caching", False)),
        disable_hybrid_kv_cache_manager=True,
        enable_flashinfer_autotune=False,
        language_model_only=True,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        kernel_config={
            "enable_cutedsl_warmup": False,
            "enable_jit_warmup": False,
        },
        seed=0,
        kv_transfer_config=transfer,
    )
    return engine, quantizer, fp4_mm, rms_norm, swiglu, attention, native_capture


def export_loaded_rope_cache(
    model: Any,
    *,
    output: str,
    expected_modules: int,
    context_length: int,
    head_size: int,
) -> dict[str, Any]:
    """Run inside the vLLM worker and retain the model's actual cache bytes."""
    import torch

    from muser_vllm.exact_rope import canonical_nco_interleaved_table
    from muser_vllm.rope_cache import (
        SCHEMA as ROPE_SCHEMA,
        sha256_bytes,
        write_exclusive,
    )

    matches: list[tuple[str, torch.Tensor]] = []
    for name, module in model.named_modules(remove_duplicate=False):
        cache = getattr(module, "cos_sin_cache", None)
        if isinstance(cache, torch.Tensor) and tuple(cache.shape) == (
            context_length,
            head_size,
        ):
            matches.append((name, cache))
    if len(matches) != expected_modules:
        raise RuntimeError(
            f"loaded model exposes {len(matches)} text RoPE modules, "
            f"expected {expected_modules}"
        )
    reference = matches[0][1]
    if reference.dtype != torch.float16 or reference.device.type != "cuda":
        raise RuntimeError(
            f"loaded RoPE cache is {reference.dtype} on {reference.device}, "
            "expected CUDA float16"
        )
    for name, cache in matches[1:]:
        if cache.dtype != reference.dtype or cache.device != reference.device:
            raise RuntimeError(f"RoPE cache {name} has a different dtype or device")
        if not torch.equal(reference, cache):
            raise RuntimeError(f"RoPE cache {name} differs from the first text layer")
    payload = canonical_nco_interleaved_table(context_length, head_size).tobytes()
    output_path = Path(output)
    write_exclusive(output_path, payload)
    return {
        "schema": ROPE_SCHEMA,
        "output": str(output_path),
        "output_bytes": len(payload),
        "output_sha256": sha256_bytes(payload),
        "output_dtype": "f32le",
        "output_layout": "position-major-interleaved-cos-sin",
        "source_bytes": len(payload),
        "source_sha256": sha256_bytes(payload),
        "source_dtype": "canonical-q30-nco-f32le",
        "source_layout": "position-major-interleaved-cos-sin",
        "source_device": str(reference.device),
        "module_count": len(matches),
        "module_names_sha256": sha256_bytes(
            ("\n".join(name for name, _ in matches) + "\n").encode()
        ),
        "context_length": context_length,
        "head_size": head_size,
        "seal_eligible": False,
    }
def runtime_receipt(
    config: dict[str, Any],
    config_path: Path,
    lease_path: Path,
    args: argparse.Namespace,
    rope_cache: dict[str, Any],
    quantizer: dict[str, Any],
    fp4_mm: dict[str, Any],
    rms_norm: dict[str, Any],
    swiglu: dict[str, Any],
    attention: dict[str, Any],
    native_capture: dict[str, Any],
    startup_warmup: dict[str, Any],
    dflash_exporter: dict[str, Any],
) -> dict[str, Any]:
    import torch
    import transformers
    import vllm

    return {
        "schema": "muser.spark-nvfp4-runtime.v10",
        "created_unix_ms": time.time_ns() // 1_000_000,
        "config_sha256": hashlib.sha256(_canonical(config)).hexdigest(),
        "config_path": str(config_path),
        "accelerator_lease_path": str(lease_path),
        "checkpoint_artifact_sha256": config["checkpoint_artifact_sha256"],
        "checkpoint_revision": config["checkpoint_revision"],
        "producer_mode": producer_mode(),
        "rope_cache": rope_cache,
        "activation_quantizer": quantizer,
        "fp4_mm": fp4_mm,
        "rms_norm": rms_norm,
        "swiglu": swiglu,
        "attention": attention,
        "native_target_hidden_capture": native_capture,
        "startup_warmup": startup_warmup,
        "dflash_exporter": dflash_exporter,
        "engine": {
            "dtype": "float16",
            "enable_flashinfer_autotune": False,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "language_model_only": True,
            "load_format": "dummy" if args.startup_dummy else "safetensors",
            "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
            "kernel_warmup": startup_warmup["selection"],
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": 1,
            "model": args.model,
            "startup_dummy": args.startup_dummy,
            "startup_only": args.startup_only,
            "tokenizer": args.tokenizer or args.model,
        },
        "determinism": {
            "attention_backend": os.environ.get("VLLM_ATTENTION_BACKEND"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "enforce_eager": True,
            "seed": 0,
            "v1_multiprocessing": os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING"),
        },
        "runtime": {
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "vllm": vllm.__version__,
            "vllm_commit": PINNED_VLLM_COMMIT,
        },
    }


def run_native_startup_warmup(
    engine: Any,
    *,
    token_count: int,
    sampling_params_type: Any,
    tokens_prompt_type: Any,
) -> dict[str, Any]:
    """Compile the product prefill shape before advertising readiness.

    This request deliberately has no handoff destination. The connector sees
    the closed internal marker and observes no layers, so the warmup changes
    neither exported bytes nor cache identity. It is startup evidence, never a
    performance sample.
    """
    if producer_mode() != "native" or token_count == 0:
        return {
            "schema": "muser.spark-native-startup-warmup.v1",
            "performed": False,
            "selection": "skipped-exact-or-explicit-zero",
            "token_count": token_count,
            "excluded_from_performance_claims": True,
        }
    token_ids = [1 + (index % 202_047) for index in range(token_count)]
    started = time.perf_counter_ns()
    outputs = engine.generate(
        tokens_prompt_type(prompt_token_ids=token_ids),
        sampling_params_type(
            temperature=0,
            max_tokens=1,
            ignore_eos=True,
            seed=0,
            extra_args={"muser_startup_warmup": True},
        ),
        use_tqdm=False,
    )
    generated = outputs[0].outputs[0].token_ids
    if len(generated) != 1:
        raise RuntimeError("native startup warmup did not generate exactly one token")
    token_bytes = b"".join(token.to_bytes(4, "little") for token in token_ids)
    return {
        "schema": "muser.spark-native-startup-warmup.v1",
        "performed": True,
        "selection": "native-fixed-shape-before-ready",
        "token_count": token_count,
        "token_ids_sha256": hashlib.sha256(token_bytes).hexdigest(),
        "first_token_id": generated[0],
        "total_ns": time.perf_counter_ns() - started,
        "excluded_from_performance_claims": True,
    }


def write_exclusive_json(path: Path, value: object) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def start_dflash_exporter(args: argparse.Namespace) -> tuple[subprocess.Popen[str] | None, dict[str, Any]]:
    values = (args.dflash_exporter, args.dflash_target, args.dflash_model, args.dflash_jobs_fifo)
    if not any(values):
        return None, {"active": False}
    if not all(values) or producer_mode() != "native":
        raise RuntimeError("DFlash exporter arguments are an all-or-none native-mode contract")
    exporter = Path(args.dflash_exporter)
    target = Path(args.dflash_target)
    draft = Path(args.dflash_model)
    fifo = Path(args.dflash_jobs_fifo)
    for label, path in (("exporter", exporter), ("target", target), ("draft", draft)):
        if not path.is_file():
            raise RuntimeError(f"DFlash {label} is not a regular file: {path}")
    if fifo.exists() or fifo.is_symlink():
        raise RuntimeError(f"refusing to replace DFlash jobs FIFO: {fifo}")
    fifo.parent.mkdir(parents=True, exist_ok=True)
    os.mkfifo(fifo, 0o600)
    log_path = fifo.with_suffix(".exporter.log")
    log = log_path.open("x", encoding="utf-8")
    command = [
        str(exporter),
        "--model",
        str(target),
        "--serve-jobs",
        str(fifo),
        "--draft-model",
        str(draft),
    ]
    process = subprocess.Popen(
        command,
        text=True,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    os.environ["MUSER_DFLASH_JOBS_FIFO"] = str(fifo)
    os.environ["MUSER_DFLASH_SESSION_DIR"] = str(fifo.parent)
    deadline = time.monotonic() + float(
        os.environ.get("MUSER_DFLASH_STARTUP_TIMEOUT_SECONDS", "300")
    )
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"DFlash exporter exited during startup with {process.returncode}"
                )
            log.flush()
            if "CUDA warmup complete" in log_path.read_text(
                encoding="utf-8", errors="replace"
            ):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("DFlash exporter startup timed out")
    except BaseException:
        process.terminate()
        process.wait(timeout=30)
        raise
    finally:
        log.close()
    return process, {
        "active": True,
        "selection": "persistent-external-target-features",
        "command": command,
        "pid": process.pid,
        "jobs_fifo": str(fifo),
        "log": str(log_path),
        "target_model": str(target),
        "draft_model": str(draft),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sock", default="/run/muser/producer.sock")
    parser.add_argument("--startup-receipt", required=True)
    parser.add_argument("--lease-file", required=True)
    parser.add_argument("--rope-cache-output", required=True)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--enable-prefix-caching",
        action="store_true",
        help="enable vLLM prefix caching (qualification lane; the connector "
        "gathers cached-prefix KV from the block table)",
    )
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument(
        "--kv-cache-memory-bytes",
        type=int,
        default=DEFAULT_KV_CACHE_BYTES,
        help="explicit deterministic KV allocation; long-context qualification raises this",
    )
    parser.add_argument("--dflash-exporter")
    parser.add_argument("--dflash-target")
    parser.add_argument("--dflash-model")
    parser.add_argument("--dflash-jobs-fifo")
    parser.add_argument(
        "--native-warmup-token-count",
        type=int,
        default=2048,
        help="compile this native prefill shape before the socket becomes ready",
    )
    parser.add_argument(
        "--startup-only",
        action="store_true",
        help="exit successfully after the engine and startup receipt are complete",
    )
    parser.add_argument(
        "--startup-dummy",
        action="store_true",
        help="wiring-only startup with dummy weights; requires --startup-only",
    )
    args = parser.parse_args()
    if args.startup_dummy and not args.startup_only:
        parser.error("--startup-dummy is valid only with --startup-only")
    if args.startup_only and any(
        (args.dflash_exporter, args.dflash_target, args.dflash_model, args.dflash_jobs_fifo)
    ):
        parser.error("startup-only cannot leave a DFlash exporter child")
    minimum_tokens = 256 if args.startup_dummy else 2048
    if (
        args.max_model_len < minimum_tokens
        or args.max_num_batched_tokens < minimum_tokens
    ):
        parser.error(
            f"producer context and batch ceiling must cover {minimum_tokens} tokens"
        )
    if not 0.1 <= args.gpu_memory_utilization <= 0.95:
        parser.error("gpu memory utilization is outside the closed safe range")
    if not 1 << 30 <= args.kv_cache_memory_bytes <= 8 << 30:
        parser.error("KV cache allocation must be between 1 and 8 GiB")
    if not 0 <= args.native_warmup_token_count <= args.max_model_len:
        parser.error("native warmup token count is outside the producer context")

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config_path = Path(args.config)
    config = load_config(config_path)
    lease_path = Path(args.lease_file)
    accelerator_lease = acquire_accelerator_lease(lease_path)

    from vllm import SamplingParams, TokensPrompt

    (
        engine,
        quantizer,
        fp4_mm,
        rms_norm,
        swiglu,
        attention,
        native_capture,
    ) = build_engine(args, config)
    rope_results = engine.apply_model(
        partial(
            export_loaded_rope_cache,
            output=args.rope_cache_output,
            expected_modules=EXPECTED_ROPE_MODULES,
            context_length=EXPECTED_CONTEXT_LENGTH,
            head_size=EXPECTED_HEAD_SIZE,
        )
    )
    if len(rope_results) != 1 or not isinstance(rope_results[0], dict):
        raise RuntimeError("RoPE cache export did not return one worker receipt")
    startup_warmup = run_native_startup_warmup(
        engine,
        token_count=(0 if args.startup_dummy else args.native_warmup_token_count),
        sampling_params_type=SamplingParams,
        tokens_prompt_type=TokensPrompt,
    )
    dflash_process, dflash_exporter = start_dflash_exporter(args)
    write_exclusive_json(
        Path(args.startup_receipt),
        runtime_receipt(
            config,
            config_path,
            lease_path,
            args,
            rope_results[0],
            quantizer,
            fp4_mm,
            rms_norm,
            swiglu,
            attention,
            native_capture,
            startup_warmup,
            dflash_exporter,
        ),
    )
    if args.startup_only:
        print(
            f"[muser-nvfp4-producer] startup-only complete; lease={lease_path}",
            flush=True,
        )
        return
    socket_path = Path(args.sock)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        mode = socket_path.lstat().st_mode
        if not stat.S_ISSOCK(mode):
            raise RuntimeError("refusing to replace a non-socket producer path")
        socket_path.unlink()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    os.chmod(socket_path, 0o600)
    server.listen(1)
    print(
        f"[muser-nvfp4-producer] ready; lease={lease_path}; "
        f"lease_fd={accelerator_lease.fileno()}",
        flush=True,
    )
    consecutive_errors = 0
    while True:
        connection, _ = server.accept()
        with connection:
            engine_touched = False
            try:
                if dflash_process is not None and dflash_process.poll() is not None:
                    raise RuntimeError(
                        f"DFlash exporter exited with {dflash_process.returncode}"
                    )
                line = connection.makefile("r").readline()
                if not line or len(line) > 8 * 1024 * 1024:
                    raise ValueError("producer request line is empty or oversized")
                request = validate_request(json.loads(line), args.max_model_len)
                started = time.perf_counter_ns()
                result_box: dict[str, Any] = {}
                dflash_feature_dir = os.environ.get("MUSER_DFLASH_FEATURE_DIR")
                deferred_dflash = (
                    producer_mode() == "native"
                    and request["handoff"].get("dflash_session")
                    and not Path(request["handoff"]["dflash_session"]).exists()
                )
                if dflash_feature_dir or deferred_dflash:
                    from muser_vllm.dflash_capture import begin_capture

                    session_output = (
                        Path(request["handoff"]["dflash_session"])
                        if deferred_dflash
                        else None
                    )
                    feature_root = (
                        Path(dflash_feature_dir)
                        if dflash_feature_dir
                        else session_output.parent
                    )
                    begin_capture(
                        request["request_id"],
                        len(request["token_ids"]) - 1,
                        feature_root / f"{request['request_id']}.f32",
                        token_ids=request["token_ids"] if deferred_dflash else None,
                        session_output=session_output,
                        device="cuda" if deferred_dflash else None,
                    )
                exact_mode = producer_mode() == "exact"
                if exact_mode:
                    from muser_vllm.exact_attention import set_exact_attention_enabled
                    from muser_vllm.exact_rms_norm import (
                        set_exact_stage_capture_enabled,
                    )

                def generate() -> None:
                    try:
                        result_box["outputs"] = engine.generate(
                            TokensPrompt(prompt_token_ids=request["token_ids"]),
                            SamplingParams(
                                temperature=0,
                                max_tokens=1,
                                ignore_eos=True,
                                seed=0,
                                extra_args={
                                    "kv_transfer_params": {
                                        "muser_handoff": request["handoff"]
                                    }
                                },
                            ),
                            use_tqdm=False,
                        )
                    except BaseException as error:
                        result_box["error"] = error

                worker = threading.Thread(target=generate, daemon=True)
                if exact_mode:
                    set_exact_attention_enabled(True)
                    set_exact_stage_capture_enabled(True)
                try:
                    engine_touched = True
                    worker.start()
                    worker.join(WATCHDOG_SECONDS)
                finally:
                    if exact_mode:
                        set_exact_stage_capture_enabled(False)
                        set_exact_attention_enabled(False)
                if worker.is_alive():
                    print("[muser-nvfp4-producer] watchdog fired", flush=True)
                    os._exit(75)
                if "error" in result_box:
                    raise result_box["error"]
                outputs = result_box["outputs"]
                generated = outputs[0].outputs[0].token_ids
                from muser_vllm.dflash_capture import (
                    consume_completed_capture,
                    finish_capture,
                )
                from muser_vllm.receipt import consume_receipt

                receipt = consume_receipt()
                dflash_features = consume_completed_capture()
                if dflash_features is None:
                    dflash_features = finish_capture()
                response = {
                    "status": "ok",
                    "request_id": request["request_id"],
                    "prompt_token_count": len(request["token_ids"]),
                    "first_vllm_token_id": generated[0] if len(generated) == 1 else None,
                    "total_ns": time.perf_counter_ns() - started,
                    "producer_receipt": receipt,
                    "dflash_features": dflash_features,
                }
            except BaseException as error:
                from muser_vllm.dflash_capture import abort_capture

                abort_capture()
                traceback.print_exc()
                consecutive_errors += 1
                response = {
                    "status": "error",
                    "error": str(error),
                    "consecutive_errors": consecutive_errors,
                }
            else:
                consecutive_errors = 0
            try:
                connection.sendall((_canonical(response) + b"\n"))
            finally:
                # A connector/send failure can leave vLLM's synchronous V1
                # engine request registered even though generate() raised.
                # Reusing that engine produced a host-side busy loop with no
                # GPU work. Fail closed after returning the error so an
                # orchestrator can restart from the persistent compile cache.
                if engine_touched and response["status"] != "ok":
                    os._exit(75)
            if consecutive_errors >= 3:
                os._exit(75)


if __name__ == "__main__":
    main()
