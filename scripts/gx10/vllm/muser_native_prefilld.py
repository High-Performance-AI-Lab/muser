#!/usr/bin/env python3
"""mTLS control bridge for the pinned resident native-NVFP4 producer.

The vLLM image owns the GPU and Handoff V2 data plane. This small host-side
daemon owns only lifecycle and the authenticated control protocol used by a
Mac receiver: start one exact image, translate one closed token request to
the resident Unix socket, and return producer phase evidence. No cache bytes
cross the control connection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import sys
import time
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
LLAMACPP_DIR = SCRIPT_DIR.parent / "llamacpp"
sys.path.insert(0, str(LLAMACPP_DIR))
import muser_prefilld as control  # noqa: E402


SCHEMA = "muser.native-prefilld.v1"
RUNTIME_SCHEMA = "muser.native-onboarding-identity.v1"
ROPE_CACHE_SCHEMA = "muser.vllm-rope-cache.v2"
SOURCE_RUNTIME_IDENTITY_SHA256 = (
    "e2485c468d4467edccc8385cf62545216970df7b8eef6ecd2b10be5fe0a68ee7"
)
PRODUCER_CONFIG_SCHEMA = "muser.spark-nvfp4-producer-config.v1"
PRODUCER_RECEIPT_SCHEMA = "muser.spark-nvfp4-prefill.v2"
CLIENT_RECEIPT_SCHEMA = "muser.spark-nvfp4-prefill-client.v1"
REQUEST_SCRIPT = "/opt/muser/scripts/gx10/vllm/request_producer.py"
NODE_GPU_LOCK = "/tmp/ferrite.gpu.lock"
MAX_MODEL_TOKENS = 131_072
# vLLM profiles this shape during every cold engine start. Keeping it equal to
# MAX_MODEL_TOKENS spends roughly two unnecessary minutes profiling a 128K
# batch on GB10. The connector is qualified to accumulate ordinary vLLM
# chunks and exports the complete cache only on the final chunk.
STARTUP_BATCH_TOKENS = 8_192

# The released image is the immutable CUDA/vLLM dependency root. Muser's
# much smaller Python runtime is deployed beside this daemon, integrity-bound
# by the enrollment runtime digest, and mounted read-only over the copy baked
# into the image. This lets a source clone actually run the code it deployed
# without rebuilding or downloading another ~10 GB image for every fix.
CONTAINER_OVERLAY_MOUNTS = (
    (SCRIPT_DIR / "resident_producer.py", "/opt/muser/scripts/gx10/vllm/resident_producer.py"),
    (SCRIPT_DIR / "request_producer.py", "/opt/muser/scripts/gx10/vllm/request_producer.py"),
    (SCRIPT_DIR / "muser_vllm", "/opt/muser/scripts/gx10/vllm/muser_vllm"),
    (LLAMACPP_DIR / "muser_v2_send.py", "/opt/muser/scripts/gx10/llamacpp/muser_v2_send.py"),
    (
        LLAMACPP_DIR / "llamacpp_session_send.py",
        "/opt/muser/scripts/gx10/llamacpp/llamacpp_session_send.py",
    ),
    (LLAMACPP_DIR / "protocol.py", "/opt/muser/scripts/gx10/llamacpp/protocol.py"),
    # MelonDMA RDMA transport (off by default; MUSER_TRANSPORT=rdma below
    # opts in). melon_rdma_stream.py is muser_v2_send.py's optional RDMA
    # byte-pipe; the .so must be compiled inside the exact pinned image (see
    # `docker run --rm --entrypoint gcc ...` against melon_rdma_pipe.c on the
    # GX10 node) so its glibc/libibverbs ABI always matches what runs it.
    (
        LLAMACPP_DIR / "melon_rdma_stream.py",
        "/opt/muser/scripts/gx10/llamacpp/melon_rdma_stream.py",
    ),
    (
        LLAMACPP_DIR / "libmelon_rdma_pipe.so",
        "/opt/muser/scripts/gx10/llamacpp/libmelon_rdma_pipe.so",
    ),
)
DEPLOYED_LLAMACPP_FILES = (
    "muser_prefilld.py",
    "muser-prefilld",
    "muser-prefilld.service",
    "muser_v2_send.py",
    "llamacpp_session_send.py",
    "protocol.py",
    "muser_prefill_producer.sh",
    "muse-glimmer-30b.layout.json",
)
DEPLOYED_VLLM_FILES = (
    "muser_native_prefilld.py",
    "resident_producer.py",
    "request_producer.py",
    "Dockerfile",
)


class NativePrefilldError(RuntimeError):
    pass


class NativePrefilldShutdown(BaseException):
    pass


def request_shutdown(_signum: int, _frame: object) -> None:
    raise NativePrefilldShutdown


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def safe_name(value: object, maximum: int = 128) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= maximum
        and all(character.isalnum() or character in "-_." for character in value)
    )


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_file(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise NativePrefilldError(f"{label} is unavailable: {path}: {error}") from error
    if not stat.S_ISREG(mode) or path.is_symlink():
        raise NativePrefilldError(f"{label} is not a regular file: {path}")


def runtime_overlay_mounts() -> list[tuple[Path, str]]:
    """Resolve one immutable set of host runtime paths for Docker.

    A package directory is mounted as a directory because Python imports a
    closed set of modules from it. Every shipped module is already included
    in the Mac-side deployed-runtime digest; refusing symlinks here prevents
    an activation from escaping that staged tree after enrollment.
    """

    resolved: list[tuple[Path, str]] = []
    for source, target in CONTAINER_OVERLAY_MOUNTS:
        try:
            mode = source.lstat().st_mode
        except OSError as error:
            raise NativePrefilldError(
                f"native runtime overlay is unavailable: {source}: {error}"
            ) from error
        if source.is_symlink() or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise NativePrefilldError(
                f"native runtime overlay is not a regular file or directory: {source}"
            )
        resolved.append((source.resolve(strict=True), target))
    return resolved


def deployed_runtime_sha256() -> str:
    """Reproduce the Mac deployer's content root over the staged lane.

    The registry digest is not merely bookkeeping: enrollment sends it to
    this daemon, and the daemon recomputes it before mounting any staged code
    into the immutable dependency image. A partial copy or post-enrollment
    edit therefore fails before CUDA is touched.
    """

    lane = SCRIPT_DIR.parent
    files: list[tuple[str, Path]] = [
        (f"llamacpp/{name}", LLAMACPP_DIR / name) for name in DEPLOYED_LLAMACPP_FILES
    ]
    files.append(("bootstrap_node.sh", lane / "bootstrap_node.sh"))
    files.extend((f"vllm/{name}", SCRIPT_DIR / name) for name in DEPLOYED_VLLM_FILES)
    package = SCRIPT_DIR / "muser_vllm"
    try:
        package_mode = package.lstat().st_mode
    except OSError as error:
        raise NativePrefilldError(f"native runtime package is unavailable: {error}") from error
    if package.is_symlink() or not stat.S_ISDIR(package_mode):
        raise NativePrefilldError("native runtime package is not a regular directory")
    files.extend(
        (f"vllm/muser_vllm/{path.name}", path)
        for path in package.iterdir()
        if path.suffix == ".py"
    )
    files.sort(key=lambda item: item[0])

    digest = hashlib.sha256(b"muser.deployed-runtime.v1\0")
    for logical_name, path in files:
        regular_file(path, f"deployed runtime {logical_name}")
        payload = path.read_bytes()
        logical = logical_name.encode()
        digest.update(len(logical).to_bytes(8, "big"))
        digest.update(logical)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def load_runtime_identity(path: Path) -> dict[str, Any]:
    regular_file(path, "native runtime identity")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NativePrefilldError(f"cannot parse native runtime identity: {error}") from error
    expected = {
        "schema",
        "status",
        "source_runtime_identity_sha256",
        "product_lane",
        "checkpoint",
        "producer_image",
        "vllm_overlay",
        "rope_cache",
        "consumer",
        "onboarding_qualification",
        "evidence",
        "ledger_basis",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise NativePrefilldError("native runtime identity field set differs")
    if (
        value.get("schema") != RUNTIME_SCHEMA
        or value.get("status") != "frozen"
        or value.get("source_runtime_identity_sha256")
        != SOURCE_RUNTIME_IDENTITY_SHA256
    ):
        raise NativePrefilldError("native onboarding identity is not frozen v1")
    rope_cache = value.get("rope_cache")
    if (
        not isinstance(rope_cache, dict)
        or set(rope_cache) != {"schema", "bytes", "sha256"}
        or rope_cache.get("schema") != ROPE_CACHE_SCHEMA
        or not isinstance(rope_cache.get("bytes"), int)
        or rope_cache["bytes"] <= 0
        or not is_sha256(rope_cache.get("sha256"))
    ):
        raise NativePrefilldError("native RoPE cache identity is not frozen v2")
    return value


def load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NativePrefilldError(f"cannot parse native handoff config: {error}") from error
    expected = {
        "schema",
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
        "container_runtime",
        "container_image",
        "container_name",
        "runtime_identity",
        "runtime_sha256",
        "checkpoint_dir",
        "timeout_seconds",
        "max_context",
        "checkpoint_artifact_sha256",
        "checkpoint_revision",
        "model_sha256",
        "model_revision",
        "tokenizer_revision",
        "tokenizer_sha256",
        "chat_template_sha256",
        "context_policy_sha256",
        "adapter_sha256",
        "target_cache_identity_sha256",
        "vllm_commit",
        "producer_socket",
        "startup_receipt",
        "rope_cache_output",
        "rope_cache_bytes",
        "rope_cache_sha256",
    }
    if not isinstance(config, dict) or set(config) != expected:
        raise NativePrefilldError("native handoff config field set differs")
    root = path.parent
    for field in (
        "certificate_chain",
        "private_key",
        "peer_ca",
        "hmac_key_file",
        "generation_ledger",
        "work_dir",
        "container_runtime",
        "runtime_identity",
        "checkpoint_dir",
        "producer_socket",
        "startup_receipt",
        "rope_cache_output",
    ):
        config[field] = resolve(root, config[field])
    for field in (
        "checkpoint_artifact_sha256",
        "model_sha256",
        "tokenizer_sha256",
        "chat_template_sha256",
        "context_policy_sha256",
        "adapter_sha256",
        "target_cache_identity_sha256",
        "runtime_sha256",
        "receiver_leaf_sha256",
        "rope_cache_sha256",
    ):
        if not is_sha256(config[field]):
            raise NativePrefilldError(f"{field} is not a lowercase SHA-256")
    pins = config["peer_leaf_sha256"]
    if not isinstance(pins, list) or not pins or any(not is_sha256(pin) for pin in pins):
        raise NativePrefilldError("peer_leaf_sha256 must be a nonempty digest list")
    image = config["container_image"]
    if not isinstance(image, str) or not image.startswith("sha256:") or not is_sha256(image[7:]):
        raise NativePrefilldError("native image must use an exact sha256 ID")
    if (
        config.get("schema") != SCHEMA
        or config.get("schema_version") != 1
        or config.get("listen_port") not in range(1, 65536)
        or config.get("timeout_seconds") not in range(1, 901)
        or config.get("max_context") not in range(2, 131073)
        or not isinstance(config.get("hmac_epoch"), int)
        or config["hmac_epoch"] < 1
        or not safe_name(config.get("container_name"))
        or config["checkpoint_revision"] != config["model_revision"]
        or config["checkpoint_revision"] != config["tokenizer_revision"]
        or not isinstance(config.get("rope_cache_bytes"), int)
        or config["rope_cache_bytes"] <= 0
    ):
        raise NativePrefilldError("native handoff identity or numeric bounds differ")
    for field in ("certificate_chain", "private_key", "peer_ca", "hmac_key_file"):
        regular_file(config[field], field)
    try:
        runtime = config["container_runtime"].resolve(strict=True)
    except OSError as error:
        raise NativePrefilldError(
            f"container_runtime is unavailable: {config['container_runtime']}: {error}"
        ) from error
    regular_file(runtime, "container_runtime")
    if not os.access(runtime, os.X_OK):
        raise NativePrefilldError(f"container_runtime is not executable: {runtime}")
    config["container_runtime"] = runtime
    if not config["checkpoint_dir"].is_dir() or config["checkpoint_dir"].is_symlink():
        raise NativePrefilldError("checkpoint_dir is not a regular directory")
    config["work_dir"].mkdir(parents=True, exist_ok=True)

    identity = load_runtime_identity(config["runtime_identity"])
    checkpoint = identity.get("checkpoint", {})
    producer = identity.get("producer_image", {})
    overlay = identity.get("vllm_overlay", {})
    consumer = identity.get("consumer", {})
    rope_cache = identity.get("rope_cache", {})
    comparisons = (
        (checkpoint.get("revision"), config["checkpoint_revision"]),
        (checkpoint.get("artifact_sha256"), config["checkpoint_artifact_sha256"]),
        (producer.get("image_id"), config["container_image"]),
        (overlay.get("adapter_sha256"), config["adapter_sha256"]),
        (overlay.get("vllm_commit"), config["vllm_commit"]),
        (consumer.get("sha256"), config["model_sha256"]),
        (consumer.get("tokenizer_sha256"), config["tokenizer_sha256"]),
        (consumer.get("chat_template_sha256"), config["chat_template_sha256"]),
        (consumer.get("context_policy_sha256"), config["context_policy_sha256"]),
        (
            consumer.get("target_cache_identity_sha256"),
            config["target_cache_identity_sha256"],
        ),
        (rope_cache.get("bytes"), config["rope_cache_bytes"]),
        (rope_cache.get("sha256"), config["rope_cache_sha256"]),
    )
    if any(actual != wanted for actual, wanted in comparisons):
        raise NativePrefilldError("native handoff config differs from runtime identity")
    config["identity"] = identity
    if deployed_runtime_sha256() != config["runtime_sha256"]:
        raise NativePrefilldError("staged native runtime differs from enrollment digest")
    return config


def validate_checkpoint(config: dict[str, Any]) -> None:
    checkpoint = config["identity"]["checkpoint"]
    expected = {
        "repository",
        "revision",
        "directory",
        "total_bytes",
        "artifact_sha256",
        "files",
        "runtime_receipt",
        "runtime_receipt_sha256",
    }
    if not isinstance(checkpoint, dict) or set(checkpoint) != expected:
        raise NativePrefilldError("checkpoint manifest field set differs")
    rows = checkpoint.get("files")
    if not isinstance(rows, list) or not rows:
        raise NativePrefilldError("checkpoint manifest is empty")
    aggregate = hashlib.sha256()
    total = 0
    previous = None
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"filename", "bytes", "sha256"}:
            raise NativePrefilldError("checkpoint file row field set differs")
        name = row["filename"]
        size = row["bytes"]
        digest = row["sha256"]
        if (
            not safe_name(name, 255)
            or "/" in name
            or ".." in name
            or not isinstance(size, int)
            or size <= 0
            or not is_sha256(digest)
            or (previous is not None and name <= previous)
        ):
            raise NativePrefilldError("checkpoint file row is unsafe or unsorted")
        artifact = config["checkpoint_dir"] / name
        regular_file(artifact, f"checkpoint file {name}")
        if artifact.stat().st_size != size:
            raise NativePrefilldError(f"checkpoint file {name} byte count differs")
        actual = sha256_file(artifact)
        if actual != digest:
            raise NativePrefilldError(f"checkpoint file {name} SHA-256 differs")
        aggregate.update(name.encode())
        aggregate.update(b"\0")
        aggregate.update(str(size).encode())
        aggregate.update(b"\0")
        aggregate.update(digest.encode())
        aggregate.update(b"\n")
        total += size
        previous = name
    if total != checkpoint["total_bytes"] or aggregate.hexdigest() != checkpoint["artifact_sha256"]:
        raise NativePrefilldError("checkpoint aggregate differs from its frozen identity")


def producer_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": PRODUCER_CONFIG_SCHEMA,
        "checkpoint_artifact_sha256": config["checkpoint_artifact_sha256"],
        "checkpoint_revision": config["checkpoint_revision"],
        "vllm_commit": config["vllm_commit"],
        "connector": {
            "adapter_sha256": config["adapter_sha256"],
            "ca_cert": "/run/muser/pki/ca.cert.pem",
            "chat_template_sha256": config["chat_template_sha256"],
            "client_cert": "/run/muser/pki/gx10.cert.pem",
            "client_key": "/run/muser/pki/gx10.key.pem",
            "context_policy_sha256": config["context_policy_sha256"],
            "hmac_epoch": config["hmac_epoch"],
            "hmac_key_file": "/run/muser/pki/hmac.key",
            "hmac_key_id": config["hmac_key_id"],
            "model_revision": config["model_revision"],
            "model_sha256": config["model_sha256"],
            "server_leaf_sha256": config["receiver_leaf_sha256"],
            "server_name": config["receiver_server_name"],
            "target_cache_identity_sha256": config["target_cache_identity_sha256"],
            "tokenizer_revision": config["tokenizer_revision"],
            "tokenizer_sha256": config["tokenizer_sha256"],
        },
    }


def write_atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def docker(config: dict[str, Any], *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(config["container_runtime"]), *args],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def stop_container(config: dict[str, Any]) -> None:
    docker(config, "rm", "-f", config["container_name"], check=False)


def container_running(config: dict[str, Any]) -> bool:
    result = docker(
        config,
        "inspect",
        "--format",
        "{{.State.Running}}",
        config["container_name"],
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def start_container(config: dict[str, Any]) -> Path:
    exact = docker(config, "image", "inspect", "--format", "{{.Id}}", config["container_image"])
    if exact.stdout.strip() != config["container_image"]:
        raise NativePrefilldError("docker resolved a different native image ID")
    stop_container(config)
    for path in (
        config["producer_socket"],
        Path(f"{config['producer_socket']}.idle"),
        config["startup_receipt"],
        config["rope_cache_output"],
    ):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        if path.exists() or path.is_symlink():
            raise NativePrefilldError(f"cannot clear stale native runtime path: {path}")
    producer_path = config["work_dir"] / "native-producer-config.json"
    write_atomic_json(producer_path, producer_config(config))
    Path(NODE_GPU_LOCK).touch(mode=0o600, exist_ok=True)
    # Enrollment exposes ``lane/pki`` as an atomic symlink to the active
    # epoch. Resolve it once so Docker binds one immutable credential set for
    # the container's lifetime rather than following a later enrollment.
    pki_dir = config["certificate_chain"].resolve(strict=True).parent
    command = [
        "run",
        "-d",
        "--name",
        config["container_name"],
        "--gpus",
        "all",
        "--network",
        "host",
        "--ipc",
        "host",
        "--restart",
        "no",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "MUSER_NVFP4_EXACT=0",
        # MelonDMA RDMA transport: off by default (MUSER_TRANSPORT defaults
        # to "tcp" below, unchanged behavior). Read from this host process's
        # own environment (e.g. the systemd unit's EnvironmentFile) so the
        # transport can be flipped without editing this file again. Device
        # nodes are passed through unconditionally (harmless, world-rw on
        # this host) so a later opt-in does not itself require a restart
        # for that half of it.
        "-e",
        f"MUSER_TRANSPORT={os.environ.get('MUSER_TRANSPORT', 'tcp')}",
        "-e",
        f"MUSER_RDMA_DEV={os.environ.get('MUSER_RDMA_DEV', 'rocep1s0f1')}",
        "-e",
        f"MUSER_RDMA_GID={os.environ.get('MUSER_RDMA_GID', '2')}",
        "-e",
        "MELON_RDMA_PIPE_LIB=/opt/muser/scripts/gx10/llamacpp/libmelon_rdma_pipe.so",
        "--device=/dev/infiniband/uverbs0",
        "--device=/dev/infiniband/uverbs1",
        "--device=/dev/infiniband/uverbs2",
        "--device=/dev/infiniband/uverbs3",
        "--device=/dev/infiniband/rdma_cm",
        "-v",
        f"{NODE_GPU_LOCK}:{NODE_GPU_LOCK}",
        "-v",
        f"{config['checkpoint_dir']}:/models/checkpoint:ro",
        "-v",
        f"{producer_path}:/run/muser/config.json:ro",
        "-v",
        f"{pki_dir}:/run/muser/pki:ro",
        "-v",
        f"{config['work_dir']}:/run/muser/work",
    ]
    for source, target in runtime_overlay_mounts():
        command.extend(["-v", f"{source}:{target}:ro"])
    command.extend(
        [
            config["container_image"],
            "--model",
            "/models/checkpoint",
            "--config",
            "/run/muser/config.json",
            "--sock",
            "/run/muser/work/producer.sock",
            "--startup-receipt",
            "/run/muser/work/native-startup-receipt.json",
            "--lease-file",
            NODE_GPU_LOCK,
            "--rope-cache-output",
            "/run/muser/work/native-rope-cache-f32le.bin",
            "--max-model-len",
            str(MAX_MODEL_TOKENS),
            "--max-num-batched-tokens",
            str(STARTUP_BATCH_TOKENS),
            "--kv-cache-memory-bytes",
            "8589934592",
            "--gpu-memory-utilization",
            "0.82",
        ]
    )
    started = docker(config, *command, check=False)
    if started.returncode != 0:
        raise NativePrefilldError(f"native container failed to start: {started.stderr[-1024:]}")
    deadline = time.monotonic() + config["timeout_seconds"]
    while time.monotonic() < deadline:
        if (
            config["producer_socket"].is_socket()
            and Path(f"{config['producer_socket']}.idle").is_file()
            and config["startup_receipt"].is_file()
            and config["rope_cache_output"].is_file()
        ):
            regular_file(config["startup_receipt"], "native startup receipt")
            regular_file(config["rope_cache_output"], "native RoPE cache")
            if (
                config["rope_cache_output"].stat().st_size != config["rope_cache_bytes"]
                or sha256_file(config["rope_cache_output"])
                != config["rope_cache_sha256"]
            ):
                raise NativePrefilldError("native RoPE cache differs from its frozen identity")
            return producer_path
        if not container_running(config):
            logs = docker(config, "logs", "--tail", "200", config["container_name"], check=False)
            raise NativePrefilldError(
                "native container exited during startup: " + (logs.stdout or logs.stderr)[-2048:]
            )
        time.sleep(0.5)
    raise NativePrefilldError("native container did not become ready before timeout")


def recover_container(config: dict[str, Any], reason: str) -> None:
    """Discard an in-flight engine and restore one known-ready producer.

    A failed ``docker exec`` is not sufficient cancellation: the resident
    vLLM process may still own the GPU and its single request slot. Replacing
    the container is the bounded recovery boundary for an abandoned request.
    """
    try:
        start_container(config)
    except Exception as error:
        raise NativePrefilldError(
            f"{reason}; native producer recovery failed: {error}"
        ) from error


def cancel_inner_request(config: dict[str, Any], transfer_id: str) -> bool:
    """Cancel only the request process and wait for verified warm-engine idle."""
    pattern = rf"[r]equest_producer\.py.*--request-id {transfer_id}"
    docker(
        config,
        "exec",
        config["container_name"],
        "pkill",
        "-TERM",
        "-f",
        pattern,
        check=False,
    )
    idle_path = Path(f"{config['producer_socket']}.idle")
    # With in-process vLLM the resident cannot safely preempt an active CUDA
    # forward from the watcher thread. It notices the closed client, lets the
    # current engine step reach a safe boundary, aborts on that same thread,
    # and only then publishes the idle marker. The qualified 128K prefill is
    # below this bound; failure still falls back to an exact container restart.
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if container_running(config) and idle_path.is_file():
            return True
        if not container_running(config):
            return False
        time.sleep(0.05)
    return False


def restore_request_slot_after_client_failure(
    config: dict[str, Any], transfer_id: str
) -> None:
    """Do not accept the next control request until the inner slot is usable.

    The request client has its own bounded socket timeout. If that client
    exits first, the resident may still be finishing cancellation on the
    engine thread. Merely observing a live container is therefore not a
    readiness proof; require its idle marker or replace it exactly.
    """
    idle_path = Path(f"{config['producer_socket']}.idle")
    if container_running(config) and idle_path.is_file():
        return
    if container_running(config) and cancel_inner_request(config, transfer_id):
        return
    recover_container(config, "failed native request did not return the warm slot to idle")


def run_controlled_command(
    config: dict[str, Any],
    command: list[str],
    control_stream: object,
    transfer_id: str,
) -> subprocess.CompletedProcess[str]:
    """Run one request while the receiver's control stream remains alive."""
    process = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + config["timeout_seconds"] + 30
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            process.kill()
            process.communicate()
            if cancel_inner_request(config, transfer_id):
                raise NativePrefilldError(
                    "native producer request timed out; request canceled and warm "
                    "producer retained"
                )
            recover_container(config, "warm request timeout cancellation did not reach idle")
            raise NativePrefilldError(
                "native producer request timed out; warm cancel failed and "
                "resident producer restarted"
            )
        try:
            stdout, stderr = process.communicate(timeout=min(0.05, remaining))
            return subprocess.CompletedProcess(
                command, process.returncode, stdout, stderr
            )
        except subprocess.TimeoutExpired:
            pass
        if control.receiver_gone(control_stream):
            process.kill()
            process.communicate()
            if cancel_inner_request(config, transfer_id):
                raise NativePrefilldError(
                    "receiver went away before native prefill finished; "
                    "request canceled and warm producer retained"
                )
            recover_container(config, "warm request cancellation did not reach idle")
            raise NativePrefilldError(
                "receiver went away before native prefill finished; warm cancel "
                "failed and resident producer restarted"
            )


def token_digest(tokens: list[int]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(token.to_bytes(4, "little"))
    return digest.hexdigest()


def validate_producer_receipt(
    value: object,
    request: dict[str, Any],
    transfer_id: str,
    generation: int,
    vllm_commit: str,
) -> dict[str, int]:
    if not isinstance(value, dict) or value.get("schema") != CLIENT_RECEIPT_SCHEMA:
        raise NativePrefilldError("native producer client receipt schema differs")
    response = value.get("response")
    if not isinstance(response, dict) or response.get("status") != "ok":
        raise NativePrefilldError("native producer returned a failed response")
    producer = response.get("producer_receipt")
    if (
        response.get("request_id") != transfer_id
        or response.get("prompt_token_count") != len(request["prompt_token_ids"])
        or not isinstance(producer, dict)
        or producer.get("schema") != PRODUCER_RECEIPT_SCHEMA
        or producer.get("producer_mode") != "native"
        or producer.get("vllm_commit") != vllm_commit
        or producer.get("prompt_token_count") != len(request["prompt_token_ids"])
        or producer.get("prefix_cut") != 0
        or producer.get("token_ids_sha256") != token_digest(request["prompt_token_ids"])
    ):
        raise NativePrefilldError("native producer receipt does not bind the prompt")
    handoff = producer.get("handoff")
    phase = producer.get("phase_ns")
    if (
        not isinstance(handoff, dict)
        or handoff.get("transfer_id") != transfer_id
        or handoff.get("generation") != generation
        or handoff.get("ack") is not True
        or not isinstance(phase, dict)
    ):
        raise NativePrefilldError("native producer receipt does not bind the transfer")
    required_handoff = (
        "payload_bytes",
        "payload_wire_ns",
        "transfer_start_unix_ns",
        "first_segment_sent_unix_ns",
        "transfer_acked_unix_ns",
    )
    if any(not isinstance(handoff.get(name), int) or handoff[name] <= 0 for name in required_handoff):
        raise NativePrefilldError("native handoff phase receipt is incomplete")
    d2h = phase.get("d2h_complete_offset")
    connector_total = phase.get("connector_total")
    first_offset = phase.get("first_segment_sent_offset")
    if (
        not isinstance(d2h, int)
        or not isinstance(connector_total, int)
        or not isinstance(first_offset, int)
        or not 0 < first_offset <= d2h <= connector_total
        or handoff.get("payload_wire_source") not in (
            "linux-tcp-info-busy-time-v1",
            # RDMA has no TCP_INFO-equivalent kernel counter; melon_rdma_pipe.c's
            # send() blocks synchronously per operation, so the wall-clock
            # sendall-blocked measurement it falls back to is exact wire time
            # for this transport, not an estimate — see muser_v2_send.py's
            # DeferredHandoffV2Sender.seal(). Only accepted together with
            # MUSER_TRANSPORT=rdma in this same daemon's own environment,
            # which os.environ.get() below re-checks independently rather
            # than trusting the label a compromised producer client could
            # otherwise forge.
            "melon-rdma-sendall-blocked-time-v1"
            if os.environ.get("MUSER_TRANSPORT") == "rdma"
            else "linux-tcp-info-busy-time-v1",
        )
        or not isinstance(handoff.get("segments"), int)
        or handoff["segments"] <= 0
        or not isinstance(handoff.get("payload_pacing_bps"), int)
        or handoff["payload_pacing_bps"] < 4_000_000_000
        or not (
            handoff["transfer_start_unix_ns"]
            <= handoff["first_segment_sent_unix_ns"]
            <= handoff["transfer_acked_unix_ns"]
        )
    ):
        raise NativePrefilldError("native streaming phase evidence is invalid")
    start = handoff["transfer_start_unix_ns"]
    end = start + d2h
    return {
        "prefill_start_unix_ns": start,
        "prefill_end_unix_ns": end,
        "state_saved_unix_ns": end,
        "transfer_start_unix_ns": start,
        "first_segment_sent_unix_ns": handoff["first_segment_sent_unix_ns"],
        "transfer_acked_unix_ns": handoff["transfer_acked_unix_ns"],
        "prefill_tokens": len(request["prompt_token_ids"]),
        "payload_bytes": handoff["payload_bytes"],
        "payload_wire_ns": handoff["payload_wire_ns"],
    }


def run_request(
    config: dict[str, Any], request: dict[str, Any], control_stream: object
) -> dict[str, int]:
    generation = control.allocate_generation(config["generation_ledger"])
    transfer_id = f"{request['request_id']}-{generation}"
    if len(transfer_id) > 256:
        raise NativePrefilldError("native transfer id exceeds its closed bound")
    token_path = config["work_dir"] / f"{transfer_id}.tokens"
    receipt_path = config["work_dir"] / f"{transfer_id}-client.json"
    control.write_tokens(token_path, request["prompt_token_ids"])
    try:
        command = [
            str(config["container_runtime"]),
            "exec",
            config["container_name"],
            "python3",
            REQUEST_SCRIPT,
            "--sock",
            "/run/muser/work/producer.sock",
            "--tokens",
            f"/run/muser/work/{token_path.name}",
            "--request-id",
            transfer_id,
            "--generation",
            str(generation),
            "--transfer-id",
            transfer_id,
            "--receiver-host",
            request["receiver_host"],
            "--receiver-port",
            str(request["receiver_port"]),
            "--output",
            f"/run/muser/work/{receipt_path.name}",
            "--timeout-seconds",
            str(config["timeout_seconds"]),
        ]
        result = run_controlled_command(
            config, command, control_stream, transfer_id
        )
        if result.returncode != 0:
            restore_request_slot_after_client_failure(config, transfer_id)
            raise NativePrefilldError(
                f"native producer request failed ({result.returncode}): {result.stderr[-1024:]}"
            )
        regular_file(receipt_path, "native producer request receipt")
        value = json.loads(receipt_path.read_text(encoding="utf-8"))
        return validate_producer_receipt(
            value,
            request,
            transfer_id,
            generation,
            config["vllm_commit"],
        )
    finally:
        token_path.unlink(missing_ok=True)
        receipt_path.unlink(missing_ok=True)


def serve(config: dict[str, Any]) -> None:
    tls = control.tls_context(config)
    family = socket.AF_INET6 if ":" in config["listen_host"] else socket.AF_INET
    listener = socket.socket(family)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((config["listen_host"], config["listen_port"]))
    listener.listen(8)
    print("muser-native-prefilld: resident producer ready", file=sys.stderr, flush=True)
    while True:
        # The inner container deliberately has no Docker restart policy: this
        # bridge owns its identity and readiness checks. Recover before
        # accepting another request; if recovery itself fails, exit so the
        # outer service supervisor can restart the whole bridge.
        if not container_running(config):
            print(
                "muser-native-prefilld: resident container is down; restarting",
                file=sys.stderr,
                flush=True,
            )
            start_container(config)
        try:
            stream = control.accept_control(listener, tls, config)
        except Exception as error:
            print(f"muser-native-prefilld: rejected control handshake: {error}", file=sys.stderr)
            continue
        request_id = "unknown"
        try:
            request = control.read_frame(stream)
            control.validate_request(request, config)
            if request["schema_version"] != 1:
                raise NativePrefilldError("native/text producer refuses multimodal control")
            request_id = request["request_id"]
            control.write_frame(stream, control.response(request_id, "accepted"))
            receipt = run_request(config, request, stream)
            control.write_frame(
                stream,
                control.response(request_id, "committed", receipt=receipt),
            )
        except Exception as error:
            print(f"muser-native-prefilld: request {request_id} failed: {error}", file=sys.stderr)
            try:
                control.write_frame(stream, control.response(request_id, "failed", str(error)))
            except Exception:
                pass
        finally:
            stream.close()


def main() -> None:
    signal.signal(signal.SIGTERM, request_shutdown)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.handoff_config)
    validate_checkpoint(config)
    producer_path: Path | None = None
    try:
        producer_path = start_container(config)
        serve(config)
    finally:
        stop_container(config)
        if producer_path is not None:
            producer_path.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except NativePrefilldShutdown:
        print("muser-native-prefilld: stopped", file=sys.stderr, flush=True)
