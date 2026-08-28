#!/usr/bin/env python3
"""Resident single-flight GX10 llama.cpp prefill producer for Muser."""

from __future__ import annotations

import argparse
import base64
import binascii
import fcntl
import hashlib
import json
import os
import select
import signal
import socket
import ssl
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MAGIC = b"MUPCTL1\0"
ALPN = "muser-prefill-control-v1"
MAX_JSON = 64 * 1024 * 1024
GPU_LOCK = "/tmp/muser.gx10.gpu.lock"
WARM_CONTAINER = "muser-gx10-prefill-warm"
READY_MARK = "[spark-kv-export] engine ready, waiting for jobs"

# Default-off live-mode perf experiment. Optional in every config schema so an
# A/B arm needs no schema bump; the MUSER_LIVE_BATCH_FULL environment variable
# overrides the config so the orchestrator can flip arms without rewriting a
# sealed handoff file. Only "1" enables it.
LIVE_BATCH_FULL_FIELD = "live_batch_full"
LIVE_BATCH_FULL_ENV = "MUSER_LIVE_BATCH_FULL"
OPTIONAL_CONFIG_FIELDS = frozenset({LIVE_BATCH_FULL_FIELD})


def live_batch_full_enabled(config: dict) -> bool:
    override = os.environ.get(LIVE_BATCH_FULL_ENV)
    if override is not None:
        return override.strip() == "1"
    return bool(config.get(LIVE_BATCH_FULL_FIELD, False))


class PrefilldError(RuntimeError):
    pass


class PrefilldShutdown(BaseException):
    """Process shutdown that must bypass per-request ``except Exception`` blocks."""


def request_shutdown(_signum: int, _frame: object) -> None:
    # The control loop deliberately catches ordinary request and TLS errors so
    # one bad client cannot take the producer down.  A termination request is
    # different: raising a BaseException gets past those handlers and unwinds
    # through main's warm-exporter/GPU-lease cleanup before the process exits.
    raise PrefilldShutdown


def compute_clients() -> list[tuple[int, str]]:
    """Return current CUDA compute clients without mutating the GPU."""
    probe = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if probe.returncode != 0:
        raise PrefilldError(f"nvidia-smi compute preflight failed: {probe.stderr[-512:]}")
    clients: list[tuple[int, str]] = []
    for line in probe.stdout.splitlines():
        if not line.strip():
            continue
        raw_pid, separator, name = line.partition(",")
        if not separator:
            raise PrefilldError("nvidia-smi returned an unparseable compute client")
        try:
            pid = int(raw_pid.strip())
        except ValueError as error:
            raise PrefilldError("nvidia-smi returned a nonnumeric compute PID") from error
        clients.append((pid, name.strip()))
    return clients


def acquire_gpu_lease() -> object:
    descriptor = os.open(GPU_LOCK, os.O_RDWR | os.O_CREAT, 0o600)
    handle = os.fdopen(descriptor, "r+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise PrefilldError("GX10 GPU lease is already held") from error
    clients = compute_clients()
    if clients:
        handle.close()
        rendered = ", ".join(f"{pid}:{name}" for pid, name in clients)
        raise PrefilldError(
            f"GX10 has pre-existing compute clients: {rendered}; if these are a "
            "previous resident producer, stop it (docker stop <its container>) "
            "and retry"
        )
    return handle


def jobs_fifo_path(config: dict) -> Path:
    return config["work_dir"] / "jobs.fifo"


def write_job_file(
    path: Path,
    *,
    n_ctx: int,
    tokens_path: Path,
    nope_fifo: Path,
    stdout_path: Path,
    status_path: Path,
    draft_out: Path | None = None,
    multimodal_plan: Path | None = None,
) -> None:
    lines = [
        f"n_ctx {n_ctx}",
        "n_batch 2048",
        "n_ubatch 512",
        "flash_attn 1",
        "skip_tail 1",
        f"tokens {tokens_path.resolve()}",
        f"nope_fifo {nope_fifo.resolve()}",
        f"stdout {stdout_path.resolve()}",
        f"status {status_path.resolve()}",
    ]
    if draft_out is not None:
        lines.append(f"draft_out {draft_out.resolve()}")
    if multimodal_plan is not None:
        lines.append(f"multimodal_plan {multimodal_plan.resolve()}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        handle.write("\n".join(lines) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def warm_docker_command(
    config: dict,
    model: Path,
    mmproj: Path | None,
    dflash: Path | None,
    jobs_fifo: Path,
) -> list[str]:
    command = [
        str(config["container_runtime"]),
        "run",
        "-d",
        "--name",
        WARM_CONTAINER,
        "--gpus",
        "all",
    ]
    if config_is_containerized(config["schema_version"]):
        command.extend(
            [
                "--read-only",
                "--tmpfs",
                "/tmp",
                "--network",
                "none",
                "--pids-limit",
                "256",
            ]
        )
    else:
        command.extend(["-v", f"{config['llama_source_dir']}:/src", "-w", "/src"])
    command.extend(["-v", f"{config['work_dir']}:{config['work_dir']}"])
    mounts = {model.parent.resolve()}
    if mmproj is not None:
        mounts.add(mmproj.parent.resolve())
    if dflash is not None:
        mounts.add(dflash.parent.resolve())
    for mount in sorted(mounts, key=str):
        command.extend(["-v", f"{mount}:{mount}:ro"])
    if config_is_containerized(config["schema_version"]):
        command.extend(
            [
                "-e",
                "CUDA_CACHE_PATH=/tmp/compute-cache",
                "-e",
                "MUSER_CROSS_VENDOR_QK=1",
            ]
        )
    if live_batch_full_enabled(config):
        command.extend(["-e", f"{LIVE_BATCH_FULL_ENV}=1"])
    command.append(config["container_image"])
    if not config_is_containerized(config["schema_version"]):
        command.append(config["export_binary"])
    command.extend([
        "--model", str(model.resolve()),
        "--serve-jobs", str(jobs_fifo.resolve()),
        "--cuda-metal-compatible-full",
    ])
    if mmproj is not None:
        command.extend(["--mmproj", str(mmproj.resolve())])
    if dflash is not None:
        command.extend(["--draft-model", str(dflash.resolve())])
    return command


def container_running(config: dict) -> bool:
    probe = subprocess.run(
        [
            str(config["container_runtime"]),
            "inspect",
            "-f",
            "{{.State.Running}}",
            WARM_CONTAINER,
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return probe.returncode == 0 and probe.stdout.strip() == "true"


def stop_warm_exporter(config: dict) -> None:
    subprocess.run(
        [str(config["container_runtime"]), "rm", "-f", WARM_CONTAINER],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def restart_warm_exporter(config: dict) -> None:
    """The only cancel the resident exporter has. A job whose consumer is gone
    keeps the GPU busy until it finishes on its own; restarting the exporter
    reclaims it now, and the reload happens immediately so the next request
    meets a warming exporter instead of a wedged one."""
    model, mmproj, dflash = config["_warm_args"]
    print(
        "muser-prefilld: restarting the warm exporter to cancel an abandoned job",
        file=sys.stderr,
        flush=True,
    )
    start_warm_exporter(config, model, mmproj, dflash)


def exporter_logs(config: dict) -> str:
    probe = subprocess.run(
        [str(config["container_runtime"]), "logs", "--tail", "200", WARM_CONTAINER],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return probe.stdout or ""


def start_warm_exporter(
    config: dict,
    model: Path,
    mmproj: Path | None,
    dflash: Path | None,
) -> None:
    stop_warm_exporter(config)
    jobs_fifo = jobs_fifo_path(config)
    if jobs_fifo.exists():
        jobs_fifo.unlink()
    os.mkfifo(jobs_fifo, 0o600)
    if live_batch_full_enabled(config):
        print(
            f"muser-prefilld: {LIVE_BATCH_FULL_ENV}=1 -- live prefill keeps the "
            "full n_batch=2048 and drains once per decode batch. DEFAULT-OFF "
            "experiment; the emitted tile schedule is unchanged.",
            file=sys.stderr,
            flush=True,
        )
    command = warm_docker_command(config, model, mmproj, dflash, jobs_fifo)
    started = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if started.returncode != 0:
        raise PrefilldError(
            f"warm exporter failed to start: {(started.stderr or started.stdout or '')[-512:]}"
        )
    deadline = time.time() + min(config["timeout_seconds"], 180)
    while time.time() < deadline:
        logs = exporter_logs(config)
        if READY_MARK in logs:
            return
        if not container_running(config):
            raise PrefilldError(f"warm exporter exited during model load: {logs[-1024:]}")
        time.sleep(0.2)
    raise PrefilldError("warm exporter did not become ready before timeout")


def receiver_gone(control) -> bool:
    """The receiver holds its control connection open, silent, for the whole
    job; the sender may sit in a FIFO open and never touch its socket, so the
    control stream is the only liveness signal that fires while the exporter
    is still computing. Any traffic here mid-job — close, reset, or bytes —
    means the receiver is no longer waiting for this job's output."""
    try:
        readable, _, _ = select.select([control], [], [], 0)
    except (OSError, ValueError):
        return True
    if not readable:
        return False
    previous = control.gettimeout()
    try:
        control.settimeout(0.2)
        control.recv(4096)
    except (ssl.SSLWantReadError, TimeoutError):
        return False
    except (ssl.SSLError, OSError):
        return True
    finally:
        try:
            control.settimeout(previous)
        except OSError:
            pass
    return True


def wait_job_status(
    config: dict, status_path: Path, timeout_seconds: int, sender=None, control=None
) -> str:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if status_path.is_file():
            return status_path.read_text(encoding="ascii").strip()
        if not container_running(config):
            raise PrefilldError(
                f"warm exporter exited during a job: {exporter_logs(config)[-1024:]}"
            )
        sender_done = sender is not None and sender.poll() is not None
        if sender_done and sender.returncode == 0:
            # The transfer is delivered and acked: the sender exits 0 only
            # after the receiver's handoff ACK, and the receiver may drop its
            # control stream the moment it commits. Neither is abandonment —
            # only the exporter's own status file is still owed.
            pass
        elif sender_done or (control is not None and receiver_gone(control)):
            # Nobody will consume this job's output; waiting out the full
            # budget would wedge the daemon for every request behind it.
            raise PrefilldError(
                "receiver went away before the job finished; canceling the abandoned prefill"
            )
        time.sleep(0.05)
    raise PrefilldError("warm exporter job timed out")


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def read_exact(stream: ssl.SSLSocket, count: int) -> bytes:
    output = bytearray()
    while len(output) < count:
        block = stream.recv(count - len(output))
        if not block:
            raise PrefilldError("control peer closed the stream")
        output.extend(block)
    return bytes(output)


def read_frame(stream: ssl.SSLSocket) -> dict:
    if read_exact(stream, 8) != MAGIC:
        raise PrefilldError("control frame magic mismatch")
    length = struct.unpack("<I", read_exact(stream, 4))[0]
    if not 0 < length <= MAX_JSON:
        raise PrefilldError("control frame length is outside bounds")
    encoded = read_exact(stream, length)
    value = json.loads(encoded)
    if not isinstance(value, dict) or canonical(value) != encoded:
        raise PrefilldError("control frame is not canonical JSON")
    return value


def write_frame(stream: ssl.SSLSocket, value: dict) -> None:
    encoded = canonical(value)
    if not 0 < len(encoded) <= MAX_JSON:
        raise PrefilldError("control response exceeds bounds")
    stream.sendall(MAGIC + struct.pack("<I", len(encoded)) + encoded)


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def config_has_dflash(schema: int) -> bool:
    return schema in (2, 4, 6, 8)


def config_has_vision(schema: int) -> bool:
    return schema in (3, 4, 7, 8)


def config_is_containerized(schema: int) -> bool:
    return schema in (5, 6, 7, 8)


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    expected_v1 = {
        "schema_version",
        "listen_host",
        "listen_port",
        "certificate_chain",
        "private_key",
        "peer_ca",
        "peer_leaf_sha256",
        "receiver_server_name",
        "receiver_leaf_sha256",
        "hmac_key_file",
        "hmac_key_id",
        "hmac_epoch",
        "generation_ledger",
        "work_dir",
        "export_binary",
        "container_runtime",
        "container_image",
        "llama_source_dir",
        "sender_script",
        "timeout_seconds",
        "max_context",
        "model_sha256",
        "model_revision",
        "tokenizer_revision",
        "tokenizer_sha256",
        "chat_template_sha256",
        "context_policy_sha256",
        "adapter_sha256",
        "target_cache_identity_sha256",
    }
    dflash_fields = {
        "dflash_identity_sha256",
        "dflash_gguf_sha256",
        "dflash_kv_heads",
        "dflash_head_dim",
        "dflash_context_geometry",
    }
    vision_fields = {
        "mmproj_sha256",
        "preprocessing_sha256",
    }
    schema = config.get("schema_version")
    expected_v5 = (expected_v1 - {"llama_source_dir"}) | {"container_receipt"}
    expected = {
        1: expected_v1,
        2: expected_v1 | dflash_fields,
        3: expected_v1 | vision_fields,
        4: expected_v1 | dflash_fields | vision_fields,
        5: expected_v5,
        6: expected_v5 | dflash_fields,
        7: expected_v5 | vision_fields,
        8: expected_v5 | dflash_fields | vision_fields,
    }.get(schema, set())
    if set(config) - OPTIONAL_CONFIG_FIELDS != expected or schema not in range(1, 9):
        raise PrefilldError("handoff config fields or schema version differ")
    if not isinstance(config.get(LIVE_BATCH_FULL_FIELD, False), bool):
        raise PrefilldError(f"{LIVE_BATCH_FULL_FIELD} must be a boolean")
    root = path.parent
    for field in (
        "certificate_chain",
        "private_key",
        "peer_ca",
        "hmac_key_file",
        "generation_ledger",
        "work_dir",
        "sender_script",
        "container_runtime",
    ):
        config[field] = resolve(root, config[field])
    if config_is_containerized(schema):
        config["container_receipt"] = resolve(root, config["container_receipt"])
    else:
        config["llama_source_dir"] = resolve(root, config["llama_source_dir"])
    for field in (
        "model_sha256",
        "tokenizer_sha256",
        "chat_template_sha256",
        "context_policy_sha256",
        "adapter_sha256",
        "target_cache_identity_sha256",
        "receiver_leaf_sha256",
    ):
        if not is_sha256(config[field]):
            raise PrefilldError(f"{field} must be lowercase SHA-256")
    if config_has_dflash(schema):
        for field in ("dflash_identity_sha256", "dflash_gguf_sha256"):
            if not is_sha256(config[field]):
                raise PrefilldError(f"{field} must be lowercase SHA-256")
        if (
            not isinstance(config["dflash_kv_heads"], int)
            or config["dflash_kv_heads"] < 1
            or not isinstance(config["dflash_head_dim"], int)
            or config["dflash_head_dim"] < 1
        ):
            raise PrefilldError("DFlash KV geometry is invalid")
        geometry = config["dflash_context_geometry"]
        geometry_fields = {
            "layers",
            "elements_per_token",
            "sink_size",
            "window_size",
        }
        if (
            not isinstance(geometry, dict)
            or set(geometry) != geometry_fields
            or any(
                not isinstance(geometry[field], int) or geometry[field] < 1
                for field in geometry_fields
            )
            or geometry["elements_per_token"]
            != config["dflash_kv_heads"] * config["dflash_head_dim"]
        ):
            raise PrefilldError("DFlash context geometry is invalid")
    if config_has_vision(schema):
        for field in ("mmproj_sha256", "preprocessing_sha256"):
            if not is_sha256(config[field]):
                raise PrefilldError(f"{field} must be lowercase SHA-256")
    pins = config["peer_leaf_sha256"]
    if not isinstance(pins, list) or not pins or any(not is_sha256(pin) for pin in pins):
        raise PrefilldError("peer_leaf_sha256 must be a nonempty SHA-256 list")
    if (
        not isinstance(config["listen_port"], int)
        or not 0 < config["listen_port"] <= 65535
        or not isinstance(config["timeout_seconds"], int)
        or not 0 < config["timeout_seconds"] <= 900
        or not isinstance(config["max_context"], int)
        or not 2 <= config["max_context"] <= 131072
        or not isinstance(config["hmac_epoch"], int)
        or config["hmac_epoch"] < 1
    ):
        raise PrefilldError("handoff config numeric bounds are invalid")
    for field in (
        "certificate_chain",
        "private_key",
        "peer_ca",
        "hmac_key_file",
        "sender_script",
        "container_runtime",
    ):
        if not config[field].is_file():
            raise PrefilldError(f"{field} is not a file: {config[field]}")
    export_binary = config["export_binary"]
    if config_is_containerized(schema):
        if export_binary != "/opt/muser/bin/spark_kv_export":
            raise PrefilldError("container exporter must use the sealed /opt/muser entrypoint")
        receipt_path = config["container_receipt"]
        if not receipt_path.is_file() or receipt_path.is_symlink():
            raise PrefilldError(f"container_receipt is not a regular file: {receipt_path}")
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PrefilldError(f"cannot parse container receipt: {error}") from error
        if (
            receipt.get("schema") != "muser.gx10-container.receipt.v1"
            or receipt.get("status") != "built"
            or receipt.get("architecture") != "arm64"
            or receipt.get("image_id") != config["container_image"]
            or receipt.get("adapter_sha256") != config["adapter_sha256"]
            or receipt.get("cuda_matmul") not in {"default", "force-cublas", "force-mmq"}
            or receipt.get("entrypoint") != [export_binary]
        ):
            raise PrefilldError("container receipt differs from the armed exporter identity")
    else:
        if not config["llama_source_dir"].is_dir():
            raise PrefilldError(
                f"llama_source_dir is not a directory: {config['llama_source_dir']}"
            )
        if (
            not isinstance(export_binary, str)
            or not export_binary.startswith("/src/build/bin/")
            or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/._-" for character in export_binary)
            or not (config["llama_source_dir"] / "build/bin" / Path(export_binary).name).is_file()
        ):
            raise PrefilldError("export_binary is not a built binary under /src/build/bin")
    image = config["container_image"]
    if (
        not isinstance(image, str)
        or not 1 <= len(image) <= 255
        or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789/._:@-" for character in image)
    ):
        raise PrefilldError("container_image violates its closed bounds")
    if config_is_containerized(schema) and not (
        image.startswith("sha256:") and is_sha256(image.removeprefix("sha256:"))
    ):
        raise PrefilldError("sealed exporter container must be addressed by exact image ID")
    config["work_dir"].mkdir(parents=True, exist_ok=True)
    return config


def validate_common_request(value: dict) -> None:
    request_id = value.get("request_id")
    if (
        not isinstance(request_id, str)
        or not 0 < len(request_id) <= 128
        or any(not (character.isalnum() or character in "-_.") for character in request_id)
        or not isinstance(value.get("deadline_unix_ms"), int)
        or value["deadline_unix_ms"] <= time.time_ns() // 1_000_000
        or not isinstance(value.get("receiver_host"), str)
        or not 0 < len(value["receiver_host"]) <= 253
        or any(
            not (character.isalnum() or character in ".:-_")
            for character in value["receiver_host"]
        )
        or not isinstance(value.get("receiver_port"), int)
        or not 0 < value["receiver_port"] <= 65535
    ):
        raise PrefilldError("control request common fields violate their closed bounds")


def validate_request(value: dict, config: dict) -> None:
    max_context = config["max_context"]
    validate_common_request(value)
    if value.get("schema_version") == 2:
        expected = {
            "schema_version",
            "request_id",
            "deadline_unix_ms",
            "segments",
            "multimodal",
            "receiver_host",
            "receiver_port",
        }
        if set(value) != expected or not config_has_vision(config["schema_version"]):
            raise PrefilldError("multimodal control requires the vision producer identity")
        identity = value.get("multimodal")
        if (
            not isinstance(identity, dict)
            or set(identity)
            != {"projector_sha256", "preprocessing_sha256", "image_sequence_sha256"}
            or identity.get("projector_sha256") != config["mmproj_sha256"]
            or identity.get("preprocessing_sha256") != config["preprocessing_sha256"]
            or not is_sha256(identity.get("image_sequence_sha256"))
        ):
            raise PrefilldError("multimodal request identity differs from the armed producer")
        segments = value.get("segments")
        if not isinstance(segments, list) or not segments:
            raise PrefilldError("multimodal request has no segments")
        positions = 0
        image_digests = bytearray()
        image_count = 0
        for segment in segments:
            if not isinstance(segment, dict):
                raise PrefilldError("multimodal segment is not an object")
            if segment.get("kind") == "tokens":
                if set(segment) != {"kind", "token_ids"}:
                    raise PrefilldError("token segment fields differ")
                tokens = segment.get("token_ids")
                if (
                    not isinstance(tokens, list)
                    or not tokens
                    or any(
                        not isinstance(token, int) or not 0 <= token <= 0x7FFFFFFF
                        for token in tokens
                    )
                ):
                    raise PrefilldError("token segment violates its closed bounds")
                positions += len(tokens)
            elif segment.get("kind") == "image":
                if set(segment) != {"kind", "data_base64", "sha256", "projected_tokens"}:
                    raise PrefilldError("image segment fields differ")
                encoded = segment.get("data_base64")
                projected = segment.get("projected_tokens")
                if (
                    not isinstance(encoded, str)
                    or not encoded
                    or len(encoded) > 48 * 1024 * 1024
                    or not is_sha256(segment.get("sha256"))
                    or not isinstance(projected, int)
                    or not 0 < projected <= 4096
                ):
                    raise PrefilldError("image segment violates its closed bounds")
                try:
                    decoded = base64.b64decode(encoded, validate=True)
                except (binascii.Error, ValueError) as error:
                    raise PrefilldError("image segment is not canonical base64") from error
                if not decoded or len(decoded) > 32 * 1024 * 1024:
                    raise PrefilldError("decoded image violates its closed bounds")
                digest = hashlib.sha256(decoded).hexdigest()
                if digest != segment["sha256"]:
                    raise PrefilldError("decoded image SHA-256 differs from the request")
                image_digests.extend(bytes.fromhex(digest))
                image_count += 1
                positions += projected
            else:
                raise PrefilldError("unknown multimodal segment kind")
        if (
            not 1 <= image_count <= 8
            or not 2 <= positions <= max_context
            or segments[-1].get("kind") != "tokens"
            or hashlib.sha256(image_digests).hexdigest()
            != identity["image_sequence_sha256"]
        ):
            raise PrefilldError("multimodal positions or image sequence identity differ")
        return
    expected = {
        "schema_version",
        "request_id",
        "deadline_unix_ms",
        "prompt_token_ids",
        "receiver_host",
        "receiver_port",
    }
    tokens = value.get("prompt_token_ids")
    if (
        set(value) != expected
        or value.get("schema_version") != 1
        or not isinstance(tokens, list)
        or not 2 <= len(tokens) <= max_context
        or any(not isinstance(token, int) or not 0 <= token <= 0x7FFFFFFF for token in tokens)
    ):
        raise PrefilldError("control request violates its closed bounds")


def tls_context(config: dict) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.maximum_version = ssl.TLSVersion.TLSv1_3
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_cert_chain(config["certificate_chain"], config["private_key"])
    context.load_verify_locations(cafile=config["peer_ca"])
    context.set_alpn_protocols([ALPN])
    return context


def accept_control(
    listener: socket.socket, context: ssl.SSLContext, config: dict
) -> ssl.SSLSocket:
    raw, _address = listener.accept()
    raw.settimeout(config["timeout_seconds"])
    try:
        stream = context.wrap_socket(raw, server_side=True)
    except BaseException:
        raw.close()
        raise
    if stream.selected_alpn_protocol() != ALPN:
        stream.close()
        raise PrefilldError("control peer negotiated the wrong ALPN")
    leaf = stream.getpeercert(binary_form=True)
    digest = hashlib.sha256(leaf or b"").hexdigest()
    if digest not in config["peer_leaf_sha256"]:
        stream.close()
        raise PrefilldError("control peer TLS leaf pin mismatch")
    return stream


def allocate_generation(path: Path) -> int:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        generation = value["next_generation"]
    except FileNotFoundError:
        generation = 1
    if not isinstance(generation, int) or generation < 1:
        raise PrefilldError("generation ledger is corrupt")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"next_generation": generation + 1}, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()
    return generation


def write_tokens(path: Path, tokens: list[int]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        for token in tokens:
            handle.write(f"{token}\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_bytes(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def materialize_multimodal(
    work_dir: Path, prefix: str, segments: list[dict]
) -> tuple[Path, Path, list[Path]]:
    witness_path = work_dir / f"{prefix}.witnesses"
    plan_path = work_dir / f"{prefix}.multimodal-plan"
    witnesses: list[int] = []
    artifacts: list[Path] = [witness_path, plan_path]
    plan_lines: list[str] = []
    for index, segment in enumerate(segments):
        if segment["kind"] == "tokens":
            path = work_dir / f"{prefix}.{index}.tokens"
            write_tokens(path, segment["token_ids"])
            witnesses.extend(segment["token_ids"])
            plan_lines.append(f"tokens\t{path}")
        else:
            path = work_dir / f"{prefix}.{index}.image"
            payload = base64.b64decode(segment["data_base64"], validate=True)
            write_bytes(path, payload)
            witnesses.extend([0x7FFFFFFF] * segment["projected_tokens"])
            plan_lines.append(
                f"image\t{path}\t{segment['projected_tokens']}\t{segment['sha256']}"
            )
        artifacts.append(path)
    write_tokens(witness_path, witnesses)
    descriptor = os.open(plan_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write("\n".join(plan_lines) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return witness_path, plan_path, artifacts


def run_request(
    config: dict,
    model: Path,
    mmproj: Path | None,
    dflash: Path | None,
    request: dict,
    control=None,
) -> dict:
    token_path = None
    dflash_session_path = None
    nope_fifo_path = None
    job_path = None
    stdout_path = None
    status_path = None
    request_artifacts: list[Path] = []
    try:
        generation = allocate_generation(config["generation_ledger"])
        prefix = f"{request['request_id']}-{generation}"
        if request["schema_version"] == 2:
            token_path, plan_path, request_artifacts = materialize_multimodal(
                config["work_dir"], prefix, request["segments"]
            )
        else:
            token_path = config["work_dir"] / f"{prefix}.tokens"
            write_tokens(token_path, request["prompt_token_ids"])
            request_artifacts = [token_path]
            plan_path = None
        if dflash is not None:
            dflash_session_path = config["work_dir"] / f"{prefix}.dflash.session"
        n_ctx = min(config["max_context"], max(request_positions(request), 2048))
        sender_command = [
                sys.executable,
                str(config["sender_script"]),
                "--prompt-token-fixture",
                str(token_path),
                "--receiver-host",
                request["receiver_host"],
                "--receiver-port",
                str(request["receiver_port"]),
                "--server-name",
                config["receiver_server_name"],
                "--ca-cert",
                str(config["peer_ca"]),
                "--client-cert",
                str(config["certificate_chain"]),
                "--client-key",
                str(config["private_key"]),
                "--server-leaf-sha256",
                config["receiver_leaf_sha256"],
                "--hmac-key-file",
                str(config["hmac_key_file"]),
                "--hmac-key-id",
                config["hmac_key_id"],
                "--hmac-epoch",
                str(config["hmac_epoch"]),
                "--generation",
                str(generation),
                "--model-sha256",
                config["model_sha256"],
                "--model-revision",
                config["model_revision"],
                "--tokenizer-revision",
                config["tokenizer_revision"],
                "--tokenizer-sha256",
                config["tokenizer_sha256"],
                "--chat-template-sha256",
                config["chat_template_sha256"],
                "--context-policy-sha256",
                config["context_policy_sha256"],
                "--adapter-sha256",
                config["adapter_sha256"],
                "--target-cache-identity-sha256",
                config["target_cache_identity_sha256"],
            ]
        if dflash is not None:
            sender_command.extend(
                [
                    "--dflash-session",
                    str(dflash_session_path),
                    "--dflash-identity-sha256",
                    config["dflash_identity_sha256"],
                    "--dflash-kv-heads",
                    str(config["dflash_kv_heads"]),
                    "--dflash-head-dim",
                    str(config["dflash_head_dim"]),
                    "--dflash-context-layers",
                    str(config["dflash_context_geometry"]["layers"]),
                    "--dflash-context-elements-per-token",
                    str(config["dflash_context_geometry"]["elements_per_token"]),
                    "--dflash-context-sink-size",
                    str(config["dflash_context_geometry"]["sink_size"]),
                    "--dflash-context-window-size",
                    str(config["dflash_context_geometry"]["window_size"]),
                ]
            )
        if request["schema_version"] == 2:
            sender_command.extend(
                [
                    "--multimodal-projector-sha256",
                    request["multimodal"]["projector_sha256"],
                    "--multimodal-preprocessing-sha256",
                    request["multimodal"]["preprocessing_sha256"],
                    "--multimodal-image-sequence-sha256",
                    request["multimodal"]["image_sequence_sha256"],
                ]
            )
        if plan_path is not None and mmproj is None:
            raise PrefilldError("multimodal request reached a producer without --mmproj")
        nope_fifo_path = config["work_dir"] / f"{prefix}.nope-fifo"
        job_path = config["work_dir"] / f"{prefix}.job"
        stdout_path = config["work_dir"] / f"{prefix}.export.stdout"
        status_path = config["work_dir"] / f"{prefix}.status"
        os.mkfifo(nope_fifo_path, 0o600)
        sender_command.extend(["--nope-fifo", str(nope_fifo_path)])
        sender_command.extend(["--transfer-id", prefix])
        print(f"muser-prefilld: transfer_id={prefix}", file=sys.stderr, flush=True)
        write_job_file(
            job_path,
            n_ctx=n_ctx,
            tokens_path=token_path,
            nope_fifo=nope_fifo_path,
            stdout_path=stdout_path,
            status_path=status_path,
            draft_out=dflash_session_path,
            multimodal_plan=plan_path,
        )
        sender = subprocess.Popen(
            sender_command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # Do not hold a synthetic O_RDWR FIFO endpoint here. The CUDA exporter
        # opens its writer only after prefill, so closing such an endpoint
        # first makes the already-open sender observe a false EOF. Let the
        # FIFO's normal open rendezvous pair the sender reader with the delayed
        # exporter writer; either side may arrive first without fabricating an
        # end-of-stream.
        with jobs_fifo_path(config).open("w", encoding="ascii") as jobs:
            jobs.write(f"{job_path.resolve()}\n")
            jobs.flush()
        try:
            status = wait_job_status(
                config,
                status_path,
                config["timeout_seconds"],
                sender=sender,
                control=control,
            )
        except PrefilldError as cancel:
            sender.kill()
            sender_stdout, sender_stderr = sender.communicate()
            # The restart below destroys the only copy of the exporter's
            # output; capture the post-mortem first, or every canceled job
            # is undiagnosable after the fact.
            postmortem = exporter_logs(config)[-2000:]
            print(
                f"muser-prefilld: canceling job: {cancel}\n"
                f"muser-prefilld: exporter tail before restart:\n{postmortem}\n"
                f"muser-prefilld: sender tail: {(sender_stderr or sender_stdout or '')[-1500:]}",
                file=sys.stderr,
                flush=True,
            )
            restart_warm_exporter(config)
            # A canceled job is a per-request failure. Naming the container
            # runtime here would read as a device failure and fail-stop the
            # lane; the exporter restart above already reclaimed the GPU.
            raise subprocess.CalledProcessError(
                1,
                ["abandoned-prefill"],
                "",
                postmortem,
            ) from None
        if status != "ok":
            sender.kill()
        sender_stdout, sender_stderr = sender.communicate()
        exporter_stdout = (
            stdout_path.read_text(encoding="utf-8") if stdout_path.is_file() else ""
        )
        exporter_stderr = exporter_logs(config)
        exported = subprocess.CompletedProcess(
            ["spark_kv_export", "--job", str(job_path)],
            0 if status == "ok" else 1,
            exporter_stdout,
            exporter_stderr,
        )
        sent = subprocess.CompletedProcess(
            sender_command, sender.returncode, sender_stdout, sender_stderr
        )
        sys.stdout.write(exported.stdout or "")
        sys.stderr.write(exported.stderr or "")
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout.write(sent.stdout or "")
        sys.stderr.write(sent.stderr or "")
        sys.stdout.flush()
        sys.stderr.flush()
        if exported.returncode != 0:
            raise subprocess.CalledProcessError(
                exported.returncode, exported.args, exported.stdout, exported.stderr
            )
        if sent.returncode != 0:
            raise subprocess.CalledProcessError(
                sent.returncode, sender_command, sent.stdout, sent.stderr
            )
        phases = parse_export_phases(exported.stdout)
        transfer = parse_sender_receipt(sent.stdout)
        prefill_tokens = request_positions(request) - 1
        return {
            "prefill_start_unix_ns": phases["prefill_compute_start_epoch_ms"] * 1_000_000,
            "prefill_end_unix_ns": phases["prefill_compute_end_epoch_ms"] * 1_000_000,
            "state_saved_unix_ns": phases["export_complete_epoch_ms"] * 1_000_000,
            "transfer_start_unix_ns": transfer["transfer_start_unix_ns"],
            "first_segment_sent_unix_ns": transfer["first_segment_sent_unix_ns"],
            "transfer_acked_unix_ns": transfer["transfer_acked_unix_ns"],
            "prefill_tokens": prefill_tokens,
            "payload_bytes": transfer["payload_bytes"],
            "payload_wire_ns": transfer["payload_wire_ns"],
        }
    finally:
        for path in (
            nope_fifo_path,
            dflash_session_path,
            job_path,
            stdout_path,
            status_path,
            *request_artifacts,
        ):
            if path is None:
                continue
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def parse_export_phases(stdout: str) -> dict[str, int]:
    required = {
        "prefill_compute_start_epoch_ms",
        "prefill_compute_end_epoch_ms",
        "export_complete_epoch_ms",
    }
    phases: dict[str, int] = {}
    for line in stdout.splitlines():
        key, separator, raw = line.partition(" ")
        if separator and key in required:
            try:
                phases[key] = int(raw.strip())
            except ValueError as error:
                raise PrefilldError(f"invalid exporter phase marker {line!r}") from error
    missing = required.difference(phases)
    if missing:
        raise PrefilldError(f"exporter omitted phase markers: {sorted(missing)}")
    if not (
        phases["prefill_compute_start_epoch_ms"]
        <= phases["prefill_compute_end_epoch_ms"]
        <= phases["export_complete_epoch_ms"]
    ):
        raise PrefilldError("exporter phase markers are out of order")
    return phases


def parse_sender_receipt(stdout: str) -> dict[str, int]:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        required = {
            "transfer_start_unix_ns",
            "first_segment_sent_unix_ns",
            "transfer_acked_unix_ns",
            "payload_bytes",
            "payload_wire_ns",
        }
        if isinstance(value, dict) and value.get("ack") is True and required <= value.keys():
            if any(isinstance(value[key], bool) or not isinstance(value[key], int) for key in required):
                break
            if not (
                value["transfer_start_unix_ns"]
                <= value["first_segment_sent_unix_ns"]
                <= value["transfer_acked_unix_ns"]
            ) or value["payload_bytes"] <= 0 or value["payload_wire_ns"] <= 0:
                break
            return {key: value[key] for key in required}
    raise PrefilldError("sender omitted a valid transfer receipt")


def request_positions(request: dict) -> int:
    if request["schema_version"] == 1:
        return len(request["prompt_token_ids"])
    return sum(
        len(segment["token_ids"])
        if segment["kind"] == "tokens"
        else segment["projected_tokens"]
        for segment in request["segments"]
    )


def response(
    request_id: str,
    status: str,
    error: str | None = None,
    receipt: dict | None = None,
) -> dict:
    value = {
        "schema_version": 1,
        "request_id": request_id,
        "status": status,
    }
    if error is not None:
        value["error"] = error[:1024] or "producer failure"
    if receipt is not None:
        value["receipt"] = receipt
    return value


def main() -> None:
    signal.signal(signal.SIGTERM, request_shutdown)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mmproj", type=Path)
    parser.add_argument("--dflash", type=Path)
    parser.add_argument("--handoff-config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.handoff_config)
    if not args.model.is_file():
        raise PrefilldError(f"model is not a file: {args.model}")
    if file_sha256(args.model) != config["model_sha256"]:
        raise PrefilldError("model SHA-256 differs from the handoff identity")
    if args.mmproj is not None:
        if not config_has_vision(config["schema_version"]):
            raise PrefilldError("--mmproj requires a vision handoff config")
        if not args.mmproj.is_file():
            raise PrefilldError(f"mmproj is not a file: {args.mmproj}")
        if file_sha256(args.mmproj) != config["mmproj_sha256"]:
            raise PrefilldError("mmproj SHA-256 differs from the handoff identity")
    elif config_has_vision(config["schema_version"]):
        raise PrefilldError("vision handoff config requires --mmproj")
    if args.dflash is not None:
        if not config_has_dflash(config["schema_version"]):
            raise PrefilldError("--dflash requires a DFlash handoff config")
        if not args.dflash.is_file():
            raise PrefilldError(f"DFlash GGUF is not a file: {args.dflash}")
        if file_sha256(args.dflash) != config["dflash_gguf_sha256"]:
            raise PrefilldError("DFlash GGUF SHA-256 differs from the handoff identity")
    elif config_has_dflash(config["schema_version"]):
        raise PrefilldError("DFlash handoff config requires --dflash; combined transfer cannot degrade")
    stop_warm_exporter(config)
    drain_deadline = time.time() + 10
    while time.time() < drain_deadline and compute_clients():
        time.sleep(0.2)
    gpu_lease = acquire_gpu_lease()
    try:
        config["_warm_args"] = (args.model, args.mmproj, args.dflash)
        start_warm_exporter(config, args.model, args.mmproj, args.dflash)
        print("muser-prefilld: warm exporter ready", file=sys.stderr, flush=True)
        context = tls_context(config)
        listener = socket.socket(socket.AF_INET6 if ":" in config["listen_host"] else socket.AF_INET)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((config["listen_host"], config["listen_port"]))
        listener.listen(8)
        while True:
            try:
                stream = accept_control(listener, context, config)
            except Exception as error:
                print(f"muser-prefilld: rejected control handshake: {error}", file=sys.stderr)
                continue
            hardware_failure = False
            try:
                request = read_frame(stream)
                validate_request(request, config)
                request_id = request["request_id"]
                write_frame(stream, response(request_id, "accepted"))
                try:
                    receipt = run_request(
                        config, args.model, args.mmproj, args.dflash, request, control=stream
                    )
                except subprocess.CalledProcessError as error:
                    print(
                        f"muser-prefilld: job {request_id} failed: {error}",
                        file=sys.stderr, flush=True,
                    )
                    # Only "the warm exporter container itself is gone" is a
                    # genuine device failure worth fail-stopping the lane for.
                    # A per-job "fail" status (consumer/pipe drop, EPIPE from a
                    # dropped Mac connection) surfaces as spark_kv_export's own
                    # nonzero exit and must not cost a 30B reload.
                    command = error.cmd if isinstance(error.cmd, list) else []
                    hardware_failure = bool(command) and command[:1] == [
                        str(config["container_runtime"])
                    ]
                    write_frame(stream, response(request_id, "failed", str(error)))
                except OSError as error:
                    print(
                        f"muser-prefilld: job {request_id} plumbing error: {error}",
                        file=sys.stderr, flush=True,
                    )
                    # Local FIFO/plumbing errors (broken pipe, stale files) are
                    # not device failures; retry-arm the lane instead of
                    # fail-stopping it.
                    write_frame(stream, response(request_id, "failed", str(error)))
                else:
                    write_frame(stream, response(request_id, "committed", receipt=receipt))
            except Exception as error:
                print(
                    f"muser-prefilld: request failed before dispatch: {error!r}",
                    file=sys.stderr, flush=True,
                )
                try:
                    write_frame(stream, response("unknown", "failed", str(error)))
                except Exception:
                    pass
            finally:
                stream.close()
            if hardware_failure:
                raise PrefilldError(
                    "accelerator producer exited abnormally; lane stopped without retry"
                )
    finally:
        stop_warm_exporter(config)
        gpu_lease.close()


if __name__ == "__main__":
    try:
        main()
    except PrefilldShutdown:
        print("muser-prefilld: stopped", file=sys.stderr, flush=True)
