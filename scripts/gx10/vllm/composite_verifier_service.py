#!/usr/bin/env python3
"""Persistent authenticated GX Dudeman verifier for composite DFlash trials.

The service imports one RedHat portable-KV genesis, evaluates the held prompt
boundary under Dudeman, and then verifies carried-frontier candidate windows.
The committed token transcript is authoritative; vLLM's prefix cache is only
rebuildable soft state. Verify calls stream an authenticated provisional f16
hidden frame after layer 49 becomes host-ready, followed by a bound final result.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import socket
import struct
import time
from typing import Any, Callable


REQUEST_SCHEMA = "muser.composite-verifier-rpc-request.v2"
PROVISIONAL_SCHEMA = "muser.composite-verifier-rpc-provisional.v1"
FINAL_SCHEMA = "muser.composite-verifier-rpc-final.v2"
RECEIPT_SCHEMA = "muser.composite-verifier-service.v2"
MAC_PREFIX = b"kvpack-domain-mac-v1\0"
REQUEST_DOMAIN = b"muser-composite-verifier-rpc-request-v2"
PROVISIONAL_DOMAIN = b"muser-composite-verifier-rpc-provisional-v1"
FINAL_DOMAIN = b"muser-composite-verifier-rpc-final-v2"
GENESIS_SCHEMA = "muser.composite-verifier-genesis.v2"
PROVISIONAL_COMMITMENT_DOMAIN = "muser-composite-verifier-provisional-commitment-v1"
VERIFIER_IDENTITY_DOMAIN = "muser-composite-verifier-identity-v1"
VOCAB_SIZE = 202_048
TARGET_LAYER_IDS = (1, 13, 25, 37, 49)
TARGET_LAYERS = len(TARGET_LAYER_IDS)
HIDDEN_SIZE = 6_656
HIDDEN_DTYPE = "f16_le"
HIDDEN_ELEMENT_BYTES = 2
HIDDEN_LAYOUT = "token-major-selected-layer-major-hidden"
ROW_BYTES = TARGET_LAYERS * HIDDEN_SIZE * HIDDEN_ELEMENT_BYTES
MAX_DRAFTS = 64
MAX_FRAME_BYTES = 1 << 20
MAX_PAYLOAD_BYTES = (MAX_DRAFTS + 1) * ROW_BYTES


class ServiceError(RuntimeError):
    """A closed protocol or target-execution invariant failed."""


Frame = tuple[dict[str, Any], bytes]
ProvisionalSender = Callable[[dict[str, Any], bytes], None]


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest_bytes(value: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(value).hexdigest()


def checkpoint_artifact_sha256(root: Path) -> str:
    """Derive the checkpoint identity from the bytes vLLM will load."""
    files: list[tuple[str, Path]] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if relative.parts[:1] == (".cache",):
            continue
        if path.is_symlink():
            raise ServiceError(f"checkpoint contains a symlink: {relative}")
        if path.is_file():
            files.append((relative.as_posix(), path))
    if not files:
        raise ServiceError("checkpoint contains no files")

    artifact = hashlib.sha256()
    for relative, path in sorted(files):
        before = path.stat()
        file_digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                file_digest.update(chunk)
        after = path.stat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ServiceError(f"checkpoint file changed while hashing: {relative}")
        artifact.update(
            f"{relative}\0{after.st_size}\0{file_digest.hexdigest()}\n".encode()
        )
    return artifact.hexdigest()


def token_digest(tokens: list[int]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(struct.pack("<I", token))
    return digest.hexdigest()


def domain_tag(key: bytes, domain: bytes, stream: bytes) -> str:
    if not 1 <= len(domain) <= 0xFFFF:
        raise ServiceError("HMAC domain length is invalid")
    body = MAC_PREFIX + struct.pack(">H", len(domain)) + domain + stream
    return hmac.new(key, body, hashlib.sha256).hexdigest()


def signed_envelope(core: dict[str, Any], key: bytes, domain: bytes) -> dict[str, Any]:
    return {
        "core": core,
        "hmac_sha256": domain_tag(key, domain, canonical_json(core)),
    }


def verify_envelope(
    envelope: object, key: bytes, domain: bytes, expected_schema: str
) -> dict[str, Any]:
    if not isinstance(envelope, dict) or set(envelope) != {"core", "hmac_sha256"}:
        raise ServiceError("authenticated envelope keys differ")
    core = envelope["core"]
    supplied = envelope["hmac_sha256"]
    if not isinstance(core, dict) or core.get("schema") != expected_schema:
        raise ServiceError("authenticated envelope schema differs")
    if (
        not isinstance(supplied, str)
        or len(supplied) != 64
        or any(character not in "0123456789abcdef" for character in supplied)
    ):
        raise ServiceError("authenticated envelope tag is malformed")
    expected = domain_tag(key, domain, canonical_json(core))
    if not hmac.compare_digest(supplied, expected):
        raise ServiceError("authenticated envelope tag differs")
    return core


def read_exact(stream: socket.socket, length: int) -> bytes:
    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = stream.recv(remaining)
        if not chunk:
            raise EOFError("peer closed a partial frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def receive_request(stream: socket.socket, key: bytes) -> dict[str, Any] | None:
    prefix = stream.recv(8)
    if not prefix:
        return None
    if len(prefix) != 8:
        prefix += read_exact(stream, 8 - len(prefix))
    length = struct.unpack(">Q", prefix)[0]
    if not 1 <= length <= MAX_FRAME_BYTES:
        raise ServiceError("request frame length is outside the closed bound")
    raw = read_exact(stream, length)
    try:
        envelope = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ServiceError("request frame is not canonical JSON") from error
    if canonical_json(envelope) != raw:
        raise ServiceError("request frame is not canonical JSON")
    core = verify_envelope(envelope, key, REQUEST_DOMAIN, REQUEST_SCHEMA)
    return core


def frame_core(
    core: dict[str, Any], payload: bytes, schema: str
) -> dict[str, Any]:
    if len(payload) % ROW_BYTES != 0:
        raise ServiceError("frame payload does not contain whole hidden rows")
    if any(
        key in core
        for key in ("schema", "payload_bytes", "payload_rows", "payload_sha256")
    ):
        raise ServiceError("frame core already contains transport-owned fields")
    framed = dict(core)
    framed["schema"] = schema
    framed["payload_bytes"] = len(payload)
    framed["payload_rows"] = len(payload) // ROW_BYTES
    framed["payload_sha256"] = digest_bytes(payload)
    return framed


def validate_frame_core(
    core: dict[str, Any], payload: bytes, expected_schema: str
) -> None:
    if (
        core.get("schema") != expected_schema
        or core.get("payload_dtype") != HIDDEN_DTYPE
        or core.get("payload_row_bytes") != ROW_BYTES
        or len(payload) % ROW_BYTES != 0
        or len(payload) > MAX_PAYLOAD_BYTES
        or core.get("payload_bytes") != len(payload)
        or core.get("payload_rows") != len(payload) // ROW_BYTES
        or core.get("payload_sha256") != digest_bytes(payload)
    ):
        raise ServiceError("frame payload ABI, geometry, or digest differs")


def send_frame(
    stream: socket.socket,
    core: dict[str, Any],
    payload: bytes,
    key: bytes,
    *,
    expected_schema: str,
    domain: bytes,
) -> None:
    validate_frame_core(core, payload, expected_schema)
    header = canonical_json(signed_envelope(core, key, domain))
    if len(header) > MAX_FRAME_BYTES:
        raise ServiceError("frame header exceeds the closed bound")
    stream.sendall(struct.pack(">QQ", len(header), len(payload)) + header)
    if payload:
        stream.sendall(payload)


def send_provisional(
    stream: socket.socket, core: dict[str, Any], payload: bytes, key: bytes
) -> None:
    send_frame(
        stream,
        core,
        payload,
        key,
        expected_schema=PROVISIONAL_SCHEMA,
        domain=PROVISIONAL_DOMAIN,
    )


def send_final(
    stream: socket.socket, core: dict[str, Any], payload: bytes, key: bytes
) -> None:
    send_frame(
        stream,
        core,
        payload,
        key,
        expected_schema=FINAL_SCHEMA,
        domain=FINAL_DOMAIN,
    )


def receive_frame(
    stream: socket.socket,
    key: bytes,
    *,
    expected_schema: str,
    domain: bytes,
) -> tuple[dict[str, Any], bytes]:
    header_bytes, payload_bytes = struct.unpack(">QQ", read_exact(stream, 16))
    if not 1 <= header_bytes <= MAX_FRAME_BYTES or payload_bytes > MAX_PAYLOAD_BYTES:
        raise ServiceError("frame lengths are outside the closed bounds")
    raw_header = read_exact(stream, header_bytes)
    try:
        envelope = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ServiceError("frame header is not canonical JSON") from error
    if canonical_json(envelope) != raw_header:
        raise ServiceError("frame header is not canonical JSON")
    core = verify_envelope(envelope, key, domain, expected_schema)
    payload = read_exact(stream, payload_bytes)
    validate_frame_core(core, payload, expected_schema)
    if expected_schema == PROVISIONAL_SCHEMA:
        validate_provisional_commitment(core)
    return core, payload


def payload_geometry() -> dict[str, Any]:
    return {
        "payload_dtype": HIDDEN_DTYPE,
        "payload_row_bytes": ROW_BYTES,
    }


def provisional_commitment_sha256(core: dict[str, Any]) -> str:
    commitment_core = dict(core)
    commitment_core.pop("provisional_sha256", None)
    return digest_bytes(
        canonical_json(
            {
                "core": commitment_core,
                "domain": PROVISIONAL_COMMITMENT_DOMAIN,
            }
        )
    )


def verifier_identity_sha256(identity: dict[str, Any]) -> str:
    return digest_bytes(
        canonical_json(
            {
                "domain": VERIFIER_IDENTITY_DOMAIN,
                "identity": identity,
            }
        )
    )


def validate_provisional_commitment(core: dict[str, Any]) -> None:
    supplied = core.get("provisional_sha256")
    if (
        not isinstance(supplied, str)
        or len(supplied) != 64
        or not hmac.compare_digest(supplied, provisional_commitment_sha256(core))
    ):
        raise ServiceError("provisional commitment differs")


def provisional_frame(
    *,
    session_id: str,
    request_id: str,
    base_head_sha256: str,
    candidates: list[int],
    payload: bytes,
    host_ready_offset_ns: int,
    payload_ready_offset_ns: int,
) -> dict[str, Any]:
    if (
        type(host_ready_offset_ns) is not int
        or type(payload_ready_offset_ns) is not int
        or host_ready_offset_ns < 0
        or payload_ready_offset_ns < host_ready_offset_ns
    ):
        raise ServiceError("provisional hidden timing differs")
    core = frame_core(
        {
            **payload_geometry(),
            "base_head_sha256": base_head_sha256,
            "candidate_count": len(candidates),
            "candidate_tokens_sha256": token_digest(candidates),
            "frame_kind": "verify_hidden_provisional",
            "host_ready_offset_ns": host_ready_offset_ns,
            "payload_ready_offset_ns": payload_ready_offset_ns,
            "replayed": False,
            "request_id": request_id,
            "session_id": session_id,
        },
        payload,
        PROVISIONAL_SCHEMA,
    )
    core["provisional_sha256"] = provisional_commitment_sha256(core)
    return core


def validate_provisional_final_binding(
    provisional: dict[str, Any], provisional_payload: bytes, final: dict[str, Any]
) -> None:
    validate_provisional_commitment(provisional)
    validate_frame_core(provisional, provisional_payload, PROVISIONAL_SCHEMA)
    candidate_count = provisional.get("candidate_count")
    committed_count = final.get("committed_count")
    committed_payload_bytes = final.get("committed_payload_bytes")
    if (
        provisional.get("schema") != PROVISIONAL_SCHEMA
        or final.get("schema") != FINAL_SCHEMA
        or final.get("status") != "verified"
        or final.get("provisional_sha256") != provisional.get("provisional_sha256")
        or final.get("session_id") != provisional.get("session_id")
        or final.get("request_id") != provisional.get("request_id")
        or final.get("base_head_sha256") != provisional.get("base_head_sha256")
        or final.get("candidate_count") != provisional.get("candidate_count")
        or final.get("candidate_tokens_sha256")
        != provisional.get("candidate_tokens_sha256")
        or type(candidate_count) is not int
        or type(committed_count) is not int
        or not 1 <= committed_count <= candidate_count
        or provisional.get("payload_rows") != provisional.get("candidate_count")
        or provisional.get("payload_bytes")
        != provisional.get("candidate_count") * ROW_BYTES
        or committed_payload_bytes != committed_count * ROW_BYTES
        or final.get("committed_payload_sha256")
        != digest_bytes(memoryview(provisional_payload)[:committed_payload_bytes])
        or final.get("payload_rows") != 0
        or final.get("payload_bytes") != 0
        or final.get("payload_sha256") != digest_bytes(b"")
    ):
        raise ServiceError("final result does not bind its provisional frame")


def validate_hidden_capture(capture: dict[str, Any], rows: int) -> bytes:
    payload = capture.get("payload")
    if (
        capture.get("dtype") != HIDDEN_DTYPE
        or capture.get("layout") != HIDDEN_LAYOUT
        or capture.get("row_bytes") != ROW_BYTES
        or capture.get("cached_tokens") != rows
        or capture.get("hidden_size") != HIDDEN_SIZE
        or capture.get("target_layers") != list(TARGET_LAYER_IDS)
        or capture.get("bytes") != rows * ROW_BYTES
        or not isinstance(payload, bytes)
        or len(payload) != rows * ROW_BYTES
    ):
        raise ServiceError("target-hidden capture ABI or geometry differs")
    return payload


def capture_timing(
    capture: dict[str, Any], generate_finished_ns: int
) -> dict[str, Any]:
    capture_started_ns = capture.get("capture_started_ns")
    finish_started_offset_ns = capture.get("finish_started_offset_ns")
    finish_completed_offset_ns = capture.get("finish_completed_offset_ns")
    layer_timings = capture.get("layer_timings")
    if (
        type(capture_started_ns) is not int
        or type(generate_finished_ns) is not int
        or generate_finished_ns < capture_started_ns
        or type(finish_started_offset_ns) is not int
        or type(finish_completed_offset_ns) is not int
        or finish_started_offset_ns < 0
        or finish_completed_offset_ns < finish_started_offset_ns
        or not isinstance(layer_timings, list)
        or any(not isinstance(timing, dict) for timing in layer_timings)
        or [timing.get("layer") for timing in layer_timings]
        != list(TARGET_LAYER_IDS)
        or any(
            type(timing.get("arrival_offset_ns")) is not int
            or type(timing.get("copy_enqueued_offset_ns")) is not int
            or timing["arrival_offset_ns"] < 0
            or timing["copy_enqueued_offset_ns"] < timing["arrival_offset_ns"]
            for timing in layer_timings
        )
    ):
        raise ServiceError("target-hidden capture timing differs")
    generate_finished_offset_ns = generate_finished_ns - capture_started_ns
    last_layer = layer_timings[-1]
    if (
        generate_finished_offset_ns < last_layer["copy_enqueued_offset_ns"]
        or finish_started_offset_ns < generate_finished_offset_ns
    ):
        raise ServiceError("target-hidden capture finished after generate returned")
    host_ready_offset_ns = capture.get("host_ready_offset_ns")
    if host_ready_offset_ns is not None:
        payload_ready_offset_ns = capture.get("payload_ready_offset_ns")
        callback_started_offset_ns = capture.get(
            "host_ready_callback_started_offset_ns"
        )
        callback_completed_offset_ns = capture.get(
            "host_ready_callback_completed_offset_ns"
        )
        if (
            type(host_ready_offset_ns) is not int
            or type(payload_ready_offset_ns) is not int
            or type(callback_started_offset_ns) is not int
            or type(callback_completed_offset_ns) is not int
            or host_ready_offset_ns < last_layer["copy_enqueued_offset_ns"]
            or payload_ready_offset_ns < host_ready_offset_ns
            or callback_started_offset_ns < payload_ready_offset_ns
            or callback_completed_offset_ns < callback_started_offset_ns
            or generate_finished_offset_ns < callback_completed_offset_ns
        ):
            raise ServiceError("target-hidden host-ready timing differs")
    return {
        "capture_started_ns": capture_started_ns,
        "finish_completed_offset_ns": finish_completed_offset_ns,
        "finish_started_offset_ns": finish_started_offset_ns,
        "generate_finished_offset_ns": generate_finished_offset_ns,
        "last_layer_arrival_to_generate_finish_ns": generate_finished_offset_ns
        - last_layer["arrival_offset_ns"],
        "last_layer_copy_enqueue_to_generate_finish_ns": generate_finished_offset_ns
        - last_layer["copy_enqueued_offset_ns"],
        "layer_timings": layer_timings,
        **(
            {
                "host_ready_callback_completed_offset_ns": capture[
                    "host_ready_callback_completed_offset_ns"
                ],
                "host_ready_callback_started_offset_ns": capture[
                    "host_ready_callback_started_offset_ns"
                ],
                "host_ready_offset_ns": capture["host_ready_offset_ns"],
                "payload_ready_offset_ns": capture["payload_ready_offset_ns"],
            }
            if host_ready_offset_ns is not None
            else {}
        ),
    }


def rank_one_token(row: object) -> int:
    if not isinstance(row, dict) or not row:
        raise ServiceError("target prompt-logprob row is absent")
    ranked: list[tuple[int, float, int]] = []
    for raw_token, value in row.items():
        token = int(raw_token)
        rank = getattr(value, "rank", None)
        logprob = float(getattr(value, "logprob", math.nan))
        if not 0 <= token < VOCAB_SIZE or not math.isfinite(logprob):
            raise ServiceError("target prompt-logprob entry is invalid")
        ranked.append((rank if isinstance(rank, int) else 1 << 30, -logprob, token))
    ranked.sort()
    if ranked[0][0] != 1:
        raise ServiceError("target prompt-logprob row has no rank-one token")
    return ranked[0][2]


def greedy_decision(candidates: list[int], target_tokens: list[int]) -> tuple[int, int]:
    if not candidates or len(target_tokens) != len(candidates):
        raise ServiceError("carried-frontier target geometry differs")
    drafts = candidates[1:]
    for index, draft in enumerate(drafts):
        if draft != target_tokens[index]:
            return index, target_tokens[index]
    return len(drafts), target_tokens[-1]


def authenticated_parent_cache_lag(
    cached_tokens: Any, parent_tokens: int, block_size: int
) -> int:
    """Validate a derived APC cut without weakening the authenticated parent."""
    if (
        type(cached_tokens) is not int
        or type(parent_tokens) is not int
        or type(block_size) is not int
        or parent_tokens < 1
        or block_size < 1
        or cached_tokens < 0
        or cached_tokens > parent_tokens
        or cached_tokens % block_size != 0
    ):
        raise ServiceError("verify prefix-cache cut is not an authenticated block cut")
    lag = parent_tokens - cached_tokens
    if lag >= block_size:
        raise ServiceError("verify prefix-cache lag exceeds one partial block")
    return lag


def target_tokens_from_prompt_tail(
    rows: Any,
    *,
    candidate_count: int,
    parent_lag: int,
    frontier: int,
    generated: int,
) -> list[int]:
    """Recover candidate target witnesses after a block-granular APC replay."""
    expected_rows = parent_lag + candidate_count
    if (
        not isinstance(rows, list)
        or len(rows) != expected_rows
        or not rows
        or rows[0] is not None
        or any(row is None for row in rows[1:])
    ):
        raise ServiceError("verify prompt-logprob coverage differs")
    candidate_rows = rows[-candidate_count:]
    if parent_lag == 0:
        if candidate_rows[0] is not None:
            raise ServiceError("verify cached-parent frontier row differs")
    elif rank_one_token(candidate_rows[0]) != frontier:
        raise ServiceError("verify recomputed frontier witness differs")
    return [rank_one_token(row) for row in candidate_rows[1:]] + [generated]


def transition_head(
    base_head: str,
    committed_tokens: list[int],
    frontier_out: int,
    output_height: int,
    hidden_sha256: str,
) -> str:
    return digest_bytes(
        canonical_json(
            {
                "base_head_sha256": base_head,
                "committed_tokens": committed_tokens,
                "frontier_out": frontier_out,
                "hidden_sha256": hidden_sha256,
                "output_height": output_height,
                "schema": "muser.composite-verifier-head.v1",
            }
        )
    )


def request_identity(core: dict[str, Any]) -> str:
    return digest_bytes(canonical_json(core))


class CompositeVerifier:
    def __init__(self, args: argparse.Namespace, key: bytes) -> None:
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        os.environ["MUSER_NVFP4_EXACT"] = "0"

        import torch
        import transformers
        import vllm
        from vllm import LLM, SamplingParams
        from vllm.config import KVTransferConfig

        from muser_vllm.composite_bundle import (
            bundle_root_sha256,
            read_bundle_manifest,
        )
        from muser_vllm.native_capture import install_native_capture

        self._torch = torch
        self._sampling_params = SamplingParams
        self._tokens_prompt = __import__("vllm", fromlist=["TokensPrompt"]).TokensPrompt
        self._args = args
        self._key = key
        self._session_id = args.session_id
        self._prompt = read_tokens(args.fixture, args.prompt_tokens)
        loaded_checkpoint_artifact = checkpoint_artifact_sha256(args.model)
        if loaded_checkpoint_artifact != args.checkpoint_artifact_sha256:
            raise ServiceError(
                "loaded target checkpoint artifact differs from the pinned identity"
            )
        manifest = read_bundle_manifest(
            args.bundle,
            key=key,
            expected_key_id=args.hmac_key_id,
            expected_source_artifact_sha256=args.source_checkpoint_artifact_sha256,
        )
        if manifest["token_ids"] != self._prompt:
            raise ServiceError("composite bundle transcript differs from service fixture")
        self._bundle_root = bundle_root_sha256(manifest)
        self._verifier_identity = {
            "bundle_root_sha256": self._bundle_root,
            "hidden_abi": {
                "dtype": HIDDEN_DTYPE,
                "hidden_size": HIDDEN_SIZE,
                "layout": HIDDEN_LAYOUT,
                "target_layers": list(TARGET_LAYER_IDS),
            },
            "source_checkpoint_artifact_sha256": args.source_checkpoint_artifact_sha256,
            "source_checkpoint_revision": args.source_checkpoint_revision,
            "target_checkpoint_artifact_sha256": loaded_checkpoint_artifact,
            "target_checkpoint_revision": args.checkpoint_revision,
        }
        self._verifier_identity_sha256 = verifier_identity_sha256(
            self._verifier_identity
        )
        transfer = KVTransferConfig(
            kv_connector="MuserCompositeKvConnector",
            kv_role="kv_consumer",
            kv_connector_module_path="muser_vllm.composite_connector",
            kv_connector_extra_config={
                "bundle_path": str(args.bundle),
                "hmac_key_file": str(args.hmac_key_file),
                "hmac_key_id": args.hmac_key_id,
                "mode": "import",
                "source_checkpoint_artifact_sha256": args.source_checkpoint_artifact_sha256,
                "source_checkpoint_revision": args.source_checkpoint_revision,
                "source_engine_mode": "native",
            },
        )
        self._capture_install = install_native_capture()
        load_started = time.perf_counter_ns()
        self._engine = LLM(
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
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_model_len,
            max_num_seqs=1,
            gpu_memory_utilization=args.gpu_memory_utilization,
            kv_cache_memory_bytes=args.kv_cache_memory_bytes,
            kv_transfer_config=transfer,
            seed=0,
        )
        self._load_ns = time.perf_counter_ns() - load_started
        cache_block_size = self._engine.llm_engine.vllm_config.cache_config.block_size
        if type(cache_block_size) is not int or not 1 <= cache_block_size <= 256:
            raise ServiceError("vLLM cache block size is outside the qualified bound")
        self._cache_block_size = cache_block_size
        self._runtime = {
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "vllm": vllm.__version__,
            "kv_cache_block_size": cache_block_size,
        }
        self._opened = False
        self._evaluated = list(self._prompt)
        self._frontier: int | None = None
        self._output_height = 0
        self._head = "0" * 64
        self._replays: dict[str, tuple[str, Frame | None, Frame]] = {}
        self._samples: list[dict[str, Any]] = []

    def handle(
        self,
        core: dict[str, Any],
        provisional_sender: ProvisionalSender | None = None,
    ) -> Frame:
        required = {
            "base_head_sha256",
            "candidates",
            "command",
            "request_id",
            "schema",
            "sent_unix_ms",
            "session_id",
        }
        if set(core) != required:
            raise ServiceError("request core keys differ")
        if core["session_id"] != self._session_id:
            raise ServiceError("request session differs")
        request_id = core["request_id"]
        if (
            not isinstance(request_id, str)
            or not 1 <= len(request_id) <= 128
            or any(
                not (character.isascii() and (character.isalnum() or character in "._:-"))
                for character in request_id
            )
        ):
            raise ServiceError("request ID is invalid")
        identity = request_identity(core)
        if request_id in self._replays:
            previous_identity, provisional, final = self._replays[request_id]
            if identity != previous_identity:
                raise ServiceError("request ID changed intent")
            if provisional is not None:
                if provisional_sender is None:
                    raise ServiceError("verify replay requires a provisional sender")
                provisional_sender(dict(provisional[0]), provisional[1])
            # Exact replay preserves the original authenticated frame cores.
            return dict(final[0]), final[1]
        command = core["command"]
        provisional: Frame | None = None
        if command == "open":
            header, payload = self._open(core)
        elif command == "verify":
            if provisional_sender is None:
                raise ServiceError("verify request requires a provisional sender")
            header, payload, provisional = self._verify(core, provisional_sender)
        elif command == "close":
            header, payload = self._close(core)
        else:
            raise ServiceError("request command is invalid")
        self._replays[request_id] = (
            identity,
            provisional,
            (dict(header), payload),
        )
        return header, payload

    def _open(self, core: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
        if self._opened or core["base_head_sha256"] != "0" * 64 or core["candidates"]:
            raise ServiceError("open request state differs")
        from muser_vllm.dflash_capture import abort_capture, begin_capture, finish_capture

        begin_capture(
            "composite-verifier-open",
            1,
            Path("/tmp/muser-composite-verifier-open.hidden.f16"),
            device="cuda",
            dtype=HIDDEN_DTYPE,
        )
        sampling = self._sampling_params(
            temperature=0,
            max_tokens=1,
            ignore_eos=True,
            logprobs=1,
            skip_reading_prefix_cache=False,
            seed=0,
        )
        try:
            self._torch.cuda.synchronize()
            started = time.perf_counter_ns()
            output = self._engine.generate(
                self._tokens_prompt(prompt_token_ids=self._prompt),
                sampling,
                use_tqdm=False,
            )[0]
            generate_finished_ns = time.perf_counter_ns()
            capture = finish_capture(materialize=False, include_payload=True)
            self._torch.cuda.synchronize()
        except Exception:
            abort_capture()
            raise
        wall_ns = time.perf_counter_ns() - started
        if capture is None:
            raise ServiceError("open target-hidden capture disappeared")
        payload = validate_hidden_capture(capture, 1)
        timing = capture_timing(capture, generate_finished_ns)
        generated = list(output.outputs[0].token_ids)
        cached = getattr(output, "num_cached_tokens", None)
        if len(generated) != 1:
            raise ServiceError("open request did not produce one frontier")
        if cached != len(self._prompt) - 1:
            raise ServiceError("open request did not import the exact composite cut")
        self._frontier = int(generated[0])
        self._head = digest_bytes(
            canonical_json(
                {
                    "bundle_root_sha256": self._bundle_root,
                    "frontier": self._frontier,
                    "prompt_tokens_sha256": token_digest(self._prompt),
                    "schema": GENESIS_SCHEMA,
                    "verifier_identity_sha256": self._verifier_identity_sha256,
                }
            )
        )
        self._opened = True
        header = {
            **payload_geometry(),
            "capture_timing": timing,
            "accepted_drafts": 0,
            "base_head_sha256": "0" * 64,
            "committed_count": 0,
            "committed_tokens": [],
            "frontier_out": self._frontier,
            "new_head_sha256": self._head,
            "num_cached_tokens": cached,
            "output_height": 0,
            "replayed": False,
            "request_id": core["request_id"],
            "session_id": self._session_id,
            "status": "opened",
            "target_tokens": [self._frontier],
            "transcript_sha256": token_digest(self._evaluated),
            "verifier_identity": self._verifier_identity,
            "verifier_identity_sha256": self._verifier_identity_sha256,
            "wall_ns": wall_ns,
        }
        self._samples.append(dict(header))
        return frame_core(header, payload, FINAL_SCHEMA), payload

    def _verify(
        self, core: dict[str, Any], provisional_sender: ProvisionalSender
    ) -> tuple[dict[str, Any], bytes, Frame]:
        if not self._opened or self._frontier is None:
            raise ServiceError("verify request precedes open")
        if core["base_head_sha256"] != self._head:
            raise ServiceError("verify request parent head is stale")
        candidates = core["candidates"]
        if (
            not isinstance(candidates, list)
            or not 1 <= len(candidates) <= MAX_DRAFTS + 1
            or any(type(token) is not int or not 0 <= token < VOCAB_SIZE for token in candidates)
            or candidates[0] != self._frontier
        ):
            raise ServiceError("verify candidate geometry or frontier differs")

        from muser_vllm.dflash_capture import abort_capture, begin_capture, finish_capture

        candidate_count = len(candidates)
        candidate_tokens_sha256 = token_digest(candidates)
        prompt = self._evaluated + candidates
        if len(prompt) + 1 > self._args.max_model_len:
            raise ServiceError("verify request exceeds the configured context")
        provisional_frames: list[Frame] = []
        provisional_send_timing: dict[str, int] = {}

        def on_host_ready(host_ready: dict[str, Any]) -> None:
            if provisional_frames:
                raise ServiceError("verify capture produced two provisional frames")
            provisional_payload = validate_hidden_capture(host_ready, candidate_count)
            provisional_core = provisional_frame(
                session_id=self._session_id,
                request_id=core["request_id"],
                base_head_sha256=core["base_head_sha256"],
                candidates=candidates,
                payload=provisional_payload,
                host_ready_offset_ns=host_ready["host_ready_offset_ns"],
                payload_ready_offset_ns=host_ready["payload_ready_offset_ns"],
            )
            send_started_ns = time.perf_counter_ns()
            provisional_sender(provisional_core, provisional_payload)
            send_completed_ns = time.perf_counter_ns()
            capture_started_ns = host_ready["capture_started_ns"]
            provisional_send_timing.update(
                {
                    "provisional_send_completed_offset_ns": send_completed_ns
                    - capture_started_ns,
                    "provisional_send_started_offset_ns": send_started_ns
                    - capture_started_ns,
                }
            )
            provisional_frames.append((provisional_core, provisional_payload))

        begin_capture(
            f"composite-verifier-{core['request_id']}",
            candidate_count,
            Path(f"/tmp/{core['request_id']}.hidden.f16"),
            device="cuda",
            dtype=HIDDEN_DTYPE,
            host_ready_callback=on_host_ready,
            # APC may replay an authenticated parent's partial cache block.
            # Only the candidate tail belongs to this DFlash round.
            row_selection="suffix",
        )
        sampling = self._sampling_params(
            temperature=0,
            max_tokens=1,
            ignore_eos=True,
            prompt_logprobs=1,
            logprobs=1,
            skip_reading_prefix_cache=False,
            seed=0,
        )
        try:
            self._torch.cuda.synchronize()
            started = time.perf_counter_ns()
            output = self._engine.generate(
                self._tokens_prompt(prompt_token_ids=prompt), sampling, use_tqdm=False
            )[0]
            generate_finished_ns = time.perf_counter_ns()
            capture = finish_capture(materialize=False, include_payload=True)
            self._torch.cuda.synchronize()
        except Exception:
            abort_capture()
            raise
        wall_ns = time.perf_counter_ns() - started
        if capture is None:
            raise ServiceError("verify target-hidden capture disappeared")
        all_hidden = validate_hidden_capture(capture, candidate_count)
        if len(provisional_frames) != 1 or all_hidden is not provisional_frames[0][1]:
            raise ServiceError("verify did not reuse its provisional hidden payload")
        timing = capture_timing(capture, generate_finished_ns)
        send_started_offset_ns = provisional_send_timing.get(
            "provisional_send_started_offset_ns"
        )
        send_completed_offset_ns = provisional_send_timing.get(
            "provisional_send_completed_offset_ns"
        )
        if (
            type(send_started_offset_ns) is not int
            or type(send_completed_offset_ns) is not int
            or send_started_offset_ns < timing["payload_ready_offset_ns"]
            or send_completed_offset_ns < send_started_offset_ns
            or timing["host_ready_callback_completed_offset_ns"]
            < send_completed_offset_ns
        ):
            raise ServiceError("verify provisional-send timing differs")
        timing.update(provisional_send_timing)
        cached = getattr(output, "num_cached_tokens", None)
        parent_lag = authenticated_parent_cache_lag(
            cached, len(self._evaluated), self._cache_block_size
        )
        generated = list(output.outputs[0].token_ids)
        if len(generated) != 1:
            raise ServiceError("verify request did not produce one bonus token")
        rows = output.prompt_logprobs
        target_tokens = target_tokens_from_prompt_tail(
            rows,
            candidate_count=candidate_count,
            parent_lag=parent_lag,
            frontier=self._frontier,
            generated=int(generated[0]),
        )
        accepted_drafts, frontier_out = greedy_decision(candidates, target_tokens)
        committed_count = 1 + accepted_drafts
        committed_tokens = candidates[:committed_count]
        committed_payload = memoryview(all_hidden)[: committed_count * ROW_BYTES]
        base_head = self._head
        next_evaluated = self._evaluated + committed_tokens
        next_output_height = self._output_height + committed_count
        hidden_sha256 = digest_bytes(committed_payload)
        next_head = transition_head(
            base_head,
            committed_tokens,
            frontier_out,
            next_output_height,
            hidden_sha256,
        )
        header = {
            **payload_geometry(),
            "capture_timing": timing,
            "accepted_drafts": accepted_drafts,
            "base_head_sha256": base_head,
            "candidate_count": candidate_count,
            "candidate_tokens_sha256": candidate_tokens_sha256,
            "committed_count": committed_count,
            "committed_payload_bytes": len(committed_payload),
            "committed_payload_sha256": hidden_sha256,
            "committed_tokens": committed_tokens,
            "frontier_out": frontier_out,
            "new_head_sha256": next_head,
            "num_cached_tokens": cached,
            "output_height": next_output_height,
            "replayed": False,
            "request_id": core["request_id"],
            "provisional_sha256": provisional_frames[0][0]["provisional_sha256"],
            "session_id": self._session_id,
            "status": "verified",
            "target_tokens": target_tokens,
            "transcript_sha256": token_digest(next_evaluated),
            "wall_ns": wall_ns,
        }
        final_payload = b""
        final_core = frame_core(header, final_payload, FINAL_SCHEMA)
        validate_provisional_final_binding(
            provisional_frames[0][0], provisional_frames[0][1], final_core
        )
        # Commit target state only after the complete provisional/final pair
        # has been constructed and cross-validated. A failed post-provisional
        # check therefore leaves the authenticated parent reusable on retry.
        self._evaluated = next_evaluated
        self._output_height = next_output_height
        self._head = next_head
        self._frontier = frontier_out
        sample = dict(header)
        sample["apc_parent_lag"] = parent_lag
        sample["prompt_logprob_rows"] = len(rows)
        self._samples.append(sample)
        return final_core, final_payload, provisional_frames[0]

    def _close(self, core: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
        if not self._opened or core["base_head_sha256"] != self._head or core["candidates"]:
            raise ServiceError("close request state differs")
        payload = b""
        header = {
            **payload_geometry(),
            "accepted_drafts": 0,
            "base_head_sha256": self._head,
            "committed_count": 0,
            "committed_tokens": [],
            "frontier_out": self._frontier,
            "new_head_sha256": self._head,
            "num_cached_tokens": len(self._evaluated),
            "output_height": self._output_height,
            "replayed": False,
            "request_id": core["request_id"],
            "session_id": self._session_id,
            "status": "closed",
            "target_tokens": [],
            "transcript_sha256": token_digest(self._evaluated),
            "wall_ns": 0,
        }
        return frame_core(header, payload, FINAL_SCHEMA), payload

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": RECEIPT_SCHEMA,
            "created_unix_ms": time.time_ns() // 1_000_000,
            "session_id": self._session_id,
            "bundle_root_sha256": self._bundle_root,
            "checkpoint": {
                "revision": self._args.checkpoint_revision,
                "artifact_sha256": self._args.checkpoint_artifact_sha256,
            },
            "engine_load_ns": self._load_ns,
            "runtime": self._runtime,
            "capture_install": self._capture_install,
            "output_height": self._output_height,
            "evaluated_tokens": len(self._evaluated),
            "final_head_sha256": self._head,
            "final_frontier": self._frontier,
            "verifier_identity": self._verifier_identity,
            "verifier_identity_sha256": self._verifier_identity_sha256,
            "samples": self._samples,
            "wire_protocol": {
                "final_schema": FINAL_SCHEMA,
                "provisional_schema": PROVISIONAL_SCHEMA,
                "request_schema": REQUEST_SCHEMA,
            },
            "seal_eligible": False,
        }


def read_tokens(path: Path, count: int) -> list[int]:
    tokens = [int(value) for value in path.read_text().split()]
    if len(tokens) < count:
        raise ServiceError("service fixture is shorter than the requested prompt")
    tokens = tokens[:count]
    if any(not 0 <= token < VOCAB_SIZE for token in tokens):
        raise ServiceError("service fixture contains an out-of-vocabulary token")
    return tokens


def write_exclusive(path: Path, value: object) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--hmac-key-file", type=Path, required=True)
    parser.add_argument("--hmac-key-id", required=True)
    parser.add_argument("--source-checkpoint-revision", required=True)
    parser.add_argument("--source-checkpoint-artifact-sha256", required=True)
    parser.add_argument("--checkpoint-revision", required=True)
    parser.add_argument("--checkpoint-artifact-sha256", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=1 << 30)
    args = parser.parse_args()
    for path, directory in (
        (args.model, True),
        (args.fixture, False),
        (args.bundle, True),
        (args.hmac_key_file, False),
    ):
        if (path.is_dir() if directory else path.is_file()) is False:
            parser.error(f"required path does not exist: {path}")
    if not args.output.is_absolute() or args.output.exists() or args.output.is_symlink():
        parser.error("output must be a new absolute path")
    if not 1 <= args.listen_port <= 65535:
        parser.error("listen port is invalid")
    if args.prompt_tokens < 2 or args.prompt_tokens + 2 > args.max_model_len:
        parser.error("prompt/model context geometry is invalid")
    if not 0.1 <= args.gpu_memory_utilization <= 0.95:
        parser.error("GPU memory utilization is outside the closed range")
    return args


def main() -> None:
    args = parse_args()
    from muser_vllm.composite_bundle import load_hmac_key

    key = load_hmac_key(args.hmac_key_file)
    verifier = CompositeVerifier(args, key)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((args.listen_host, args.listen_port))
    listener.listen(1)
    print(
        json.dumps(
            {
                "schema": "muser.composite-verifier-ready.v2",
                "final_schema": FINAL_SCHEMA,
                "host": args.listen_host,
                "port": args.listen_port,
                "provisional_schema": PROVISIONAL_SCHEMA,
                "request_schema": REQUEST_SCHEMA,
                "session_id": args.session_id,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        closed = False
        while not closed:
            connection, _ = listener.accept()
            with connection:
                while True:
                    core: dict[str, Any] = {}
                    provisional_sent = False
                    try:
                        core = receive_request(connection, key)
                        if core is None:
                            break

                        def emit_provisional(
                            provisional_core: dict[str, Any],
                            provisional_payload: bytes,
                        ) -> None:
                            nonlocal provisional_sent
                            send_provisional(
                                connection,
                                provisional_core,
                                provisional_payload,
                                key,
                            )
                            provisional_sent = True

                        header, payload = verifier.handle(
                            core,
                            emit_provisional,
                        )
                        send_final(connection, header, payload, key)
                        if header["status"] == "closed":
                            closed = True
                            break
                    except (ConnectionError, EOFError) as error:
                        print(
                            json.dumps(
                                {
                                    "error": type(error).__name__,
                                    "request_id": core.get("request_id", "invalid"),
                                    "schema": "muser.composite-verifier-connection-abort.v1",
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                        # The authenticated request/result replay table remains
                        # live across a transport reconnect. A client may retry
                        # the identical request ID; changed intent still fails.
                        break
                    except ServiceError as error:
                        print(
                            json.dumps(
                                {
                                    "error": str(error),
                                    "provisional_sent": provisional_sent,
                                    "request_id": core.get("request_id", "invalid"),
                                    "schema": "muser.composite-verifier-request-failure.v1",
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                        if provisional_sent:
                            # Never follow a provisional frame with an unbound
                            # generic error. EOF aborts the incomplete pair;
                            # the client can reconnect and retry the identical
                            # request against the unchanged parent state.
                            break
                        try:
                            error_payload = b""
                            error_core = frame_core(
                                {
                                    **payload_geometry(),
                                    "accepted_drafts": 0,
                                    "base_head_sha256": "0" * 64,
                                    "committed_count": 0,
                                    "committed_tokens": [],
                                    "error": str(error),
                                    "frontier_out": None,
                                    "new_head_sha256": "0" * 64,
                                    "num_cached_tokens": None,
                                    "output_height": 0,
                                    "replayed": False,
                                    "request_id": (
                                        core.get("request_id", "invalid")
                                        if isinstance(core, dict)
                                        else "invalid"
                                    ),
                                    "session_id": args.session_id,
                                    "status": "error",
                                    "target_tokens": [],
                                    "transcript_sha256": "0" * 64,
                                    "wall_ns": 0,
                                },
                                error_payload,
                                FINAL_SCHEMA,
                            )
                            send_final(
                                connection,
                                error_core,
                                error_payload,
                                key,
                            )
                        except ConnectionError:
                            pass
                        break
    finally:
        listener.close()
        write_exclusive(args.output, verifier.receipt())


if __name__ == "__main__":
    main()
