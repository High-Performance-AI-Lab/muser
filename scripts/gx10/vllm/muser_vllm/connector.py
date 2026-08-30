"""Pinned vLLM producer for Muse Glimmer NVFP4 -> Muser Handoff V2.

This connector intentionally supports one exact configuration: vLLM commit
6adad08767583f52eb4d2122111af0bf638ed5e6, FlashAttention HND f16 KV, one
Muse request, and target-only Handoff V2. Chunked prefill is accumulated only
in vLLM's ordinary KV cache; the connector activates on the final chunk and
gathers the complete prefix from the physical block table. Any layout or
scheduler change fails closed before a seal can be emitted.
"""

from __future__ import annotations

import hashlib
import os
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

_LLAMACPP_DIR = Path(__file__).resolve().parents[2] / "llamacpp"
if str(_LLAMACPP_DIR) not in sys.path:
    sys.path.insert(0, str(_LLAMACPP_DIR))

from muser_v2_send import (  # noqa: E402
    NOPE_LAYERS,
    DeferredHandoffV2Sender,
    ProtocolError,
    muse_intents,
    packed_planes,
)

from .packing import neox_to_interleaved_order, pack_intent_payload, token_ids_sha256
from .receipt import ensure_slot_available, publish_receipt

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request


PINNED_VLLM_COMMIT = "6adad08767583f52eb4d2122111af0bf638ed5e6"
EXPECTED_LAYERS = 52
EXPECTED_KV_HEADS = 2
EXPECTED_HEAD_DIM = 128
_REQUEST_CANCELLATION = threading.Event()


def clear_request_cancellation() -> None:
    _REQUEST_CANCELLATION.clear()


def request_cancellation() -> None:
    _REQUEST_CANCELLATION.set()


@dataclass
class MuseRequestMeta:
    prompt_token_ids: list[int]
    handoff: dict[str, Any]
    scheduled_perf_ns: int
    prefill_chunks: int = 1


@dataclass
class _ScheduledRequest:
    meta: MuseRequestMeta | None
    prompt_token_count: int
    chunks: int = 1


@dataclass
class MuseConnectorMetadata(KVConnectorMetadata):
    requests: list[MuseRequestMeta] = field(default_factory=list)


@dataclass
class _PendingLayer:
    layer: int
    device_pair: torch.Tensor
    host_pair: torch.Tensor
    ready: torch.cuda.Event
    copy_started_ns: int


class MuserMuseHandoffConnector(KVConnectorBase_V1):
    """Single-request native Muse producer with an exact Handoff V2 seam."""

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        return True

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: "VllmConfig") -> str | None:
        return "HND"

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        if vllm_config.parallel_config.tensor_parallel_size != 1:
            raise ValueError("Muse Handoff V2 requires tensor_parallel_size=1")
        self._block_size = int(vllm_config.cache_config.block_size)
        self._prefix_caching = bool(
            getattr(vllm_config.cache_config, "enable_prefix_caching", False)
        )
        text = vllm_config.model_config.hf_text_config
        geometry = (
            int(text.num_hidden_layers),
            int(text.num_key_value_heads),
            int(text.head_dim),
        )
        if geometry != (EXPECTED_LAYERS, EXPECTED_KV_HEADS, EXPECTED_HEAD_DIM):
            raise ValueError(f"unexpected Muse text geometry {geometry!r}")
        self._extra = dict(
            vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        )
        self._require_static_config()
        exact_flag = os.environ.get("MUSER_NVFP4_EXACT", "0")
        if exact_flag not in {"0", "1"}:
            raise ValueError("MUSER_NVFP4_EXACT must be exactly 0 or 1")
        self._producer_mode = "exact" if exact_flag == "1" else "native"
        self._request: MuseRequestMeta | None = None
        self._sender: DeferredHandoffV2Sender | None = None
        self._pending: dict[int, _PendingLayer] = {}
        self._copy_stream: torch.cuda.Stream | None = None
        self._started_perf_ns = 0
        self._first_layer_perf_ns = 0
        self._last_layer_perf_ns = 0
        # Streaming state (2026-08-19): intents are layer-major, so a segment
        # can be packed and sent as soon as every layer it packs has been
        # materialized — during prefill, not after it. Sends always happen in
        # canonical intent order (`_next_intent` only advances in order).
        self._intents: list = []
        self._intent_layers: list[set[int]] = []
        self._next_intent = 0
        self._planes: dict[tuple[int, str], bytes] = {}
        self._layer_digests: dict[int, str] = {}
        self._materialized: set[int] = set()
        self._first_send_perf_ns = 0
        self._last_materialized_perf_ns = 0
        self._layers_saved = 0
        # Sends run on a dedicated thread: the forward thread only enqueues
        # intents whose layers are saved, so paced socket writes never stall
        # prefill. `None` on the queue is the stop sentinel.
        self._send_queue: queue.Queue | None = None
        self._sender_thread: threading.Thread | None = None
        self._send_error: Exception | None = None
        # Scheduler-side only. Worker instances receive the final metadata
        # object and never populate this table. A None meta is the closed
        # startup-warmup marker, which may itself span multiple chunks.
        self._scheduled_requests: dict[str, _ScheduledRequest] = {}

    def _require_static_config(self) -> None:
        required = {
            "adapter_sha256",
            "ca_cert",
            "chat_template_sha256",
            "client_cert",
            "client_key",
            "context_policy_sha256",
            "hmac_epoch",
            "hmac_key_file",
            "hmac_key_id",
            "model_revision",
            "model_sha256",
            "server_leaf_sha256",
            "server_name",
            "target_cache_identity_sha256",
            "tokenizer_revision",
            "tokenizer_sha256",
        }
        missing = sorted(required - self._extra.keys())
        if missing:
            raise ValueError(f"Muse Handoff connector missing config keys: {missing}")
        for name in (
            "adapter_sha256",
            "chat_template_sha256",
            "context_policy_sha256",
            "model_sha256",
            "server_leaf_sha256",
            "target_cache_identity_sha256",
            "tokenizer_sha256",
        ):
            value = self._extra[name]
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"{name} is not a lowercase SHA-256")

    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, MuseConnectorMetadata):
            raise ProtocolError("unexpected Muse connector metadata type")
        if not metadata.requests:
            return
        if len(metadata.requests) != 1 or self._request is not None:
            raise ProtocolError("Muse Handoff V2 requires one inactive request")
        ensure_slot_available()
        request = metadata.requests[0]
        self._request = request
        self._started_perf_ns = time.perf_counter_ns()
        self._sender = DeferredHandoffV2Sender(
            self._sender_args(request.handoff),
            request.prompt_token_ids,
            request.handoff.get("prefix_cut", 0),
        )
        prefix_cut = request.handoff.get("prefix_cut", 0)
        self._intents = muse_intents(len(request.prompt_token_ids) - 1, prefix_cut)
        self._intent_layers = [
            {layer for layer, _ in packed_planes(intent)} for intent in self._intents
        ]
        self._next_intent = 0
        self._planes = {}
        self._layer_digests = {}
        self._materialized = set()
        self._first_send_perf_ns = 0
        self._last_materialized_perf_ns = 0
        self._send_queue = queue.Queue()
        self._send_error = None
        self._sender_thread = threading.Thread(
            target=self._sender_loop, name="muser-handoff-sender", daemon=True
        )
        self._sender_thread.start()

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        if self._request is None:
            return
        if not isinstance(attn_metadata, FlashAttentionMetadata):
            raise ProtocolError(
                f"Muse seam requires FlashAttentionMetadata, got {type(attn_metadata)!r}"
            )
        layer = self._parse_layer(layer_name)
        if layer in self._materialized or layer != self._layers_saved:
            raise ProtocolError(
                f"vLLM layer order changed: got {layer}, expected {self._layers_saved}"
            )
        self._layers_saved += 1
        if self._send_error is not None:
            # Defer the transport error until wait_for_save. A disappearing
            # receiver and its control-client cancellation arrive on separate
            # sockets and can race by a few milliseconds. Raising from this
            # model hook makes vLLM dump the entire scheduled prompt; waiting
            # until the post-forward seam avoids that disclosure and lets a
            # controlled cancellation complete without poisoning the engine.
            return
        pair = self._extract_pair(
            kv_layer,
            attn_metadata.slot_mapping,
            getattr(attn_metadata, "block_table", None),
            len(self._request.prompt_token_ids) - 1,
        )
        if pair.dtype != torch.float16:
            raise ProtocolError(f"KV dtype is {pair.dtype}, expected torch.float16")
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream(device=pair.device)
        current = torch.cuda.current_stream(device=pair.device)
        self._copy_stream.wait_stream(current)
        with torch.cuda.stream(self._copy_stream):
            canonical_pair = pair.contiguous()
            host_pair = torch.empty(
                canonical_pair.shape, dtype=torch.float16, device="cpu", pin_memory=True
            )
            copied_ns = time.perf_counter_ns()
            host_pair.copy_(canonical_pair, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(self._copy_stream)
        # Keep the gathered device allocation alive until `ready` has been
        # synchronized in wait_for_save. The caching allocator tracks its
        # creation stream, not this copy stream; dropping the last reference
        # here permits reuse while the D2H DMA is still reading it.
        canonical_pair.record_stream(self._copy_stream)
        self._pending[layer] = _PendingLayer(
            layer, canonical_pair, host_pair, ready, copied_ns
        )
        now = time.perf_counter_ns()
        if self._first_layer_perf_ns == 0:
            self._first_layer_perf_ns = now
        self._last_layer_perf_ns = now
        # Streaming seam: enqueue every intent whose layers have all been
        # saved, in canonical order. Layers arrive strictly in order, so an
        # intent is ready once its highest layer number was saved.
        while self._next_intent < len(self._intents) and max(
            self._intent_layers[self._next_intent]
        ) < self._layers_saved:
            self._send_queue.put(self._intents[self._next_intent])
            self._next_intent += 1

    def _materialize_layer(self, layer: int) -> None:
        # Runs on the sender thread only.
        if layer in self._materialized:
            return
        request = self._request
        if request is None:
            raise ProtocolError("materialize without an active request")
        pending = self._pending.pop(layer)
        pending.ready.synchronize()
        pair = pending.host_pair
        expected_shape = (
            2,
            len(request.prompt_token_ids) - 1,
            EXPECTED_KV_HEADS,
            EXPECTED_HEAD_DIM,
        )
        if tuple(pair.shape) != expected_shape:
            raise ProtocolError(
                f"layer {layer} canonical KV shape is {tuple(pair.shape)}, "
                f"expected {expected_shape}"
            )
        key = pair[0].contiguous().numpy().astype("<f2", copy=False).tobytes()
        value = pair[1].contiguous().numpy().astype("<f2", copy=False).tobytes()
        nope = layer in NOPE_LAYERS
        self._planes[(layer, "nope_key" if nope else "swa_key")] = key
        self._planes[(layer, "nope_value" if nope else "swa_value")] = value
        self._layer_digests[layer] = hashlib.sha256(key + value).hexdigest()
        self._materialized.add(layer)
        self._last_materialized_perf_ns = time.perf_counter_ns()

    def _sender_loop(self) -> None:
        # Owns the socket: materializes each intent's layers, packs, and
        # sends in canonical order until the stop sentinel. On any failure the
        # error is latched for wait_for_save and the remaining queue is
        # drained so its join() cannot hang.
        try:
            while True:
                intent = self._send_queue.get()
                try:
                    if intent is None:
                        return
                    for layer in sorted(self._intent_layers[intent.sequence]):
                        self._materialize_layer(layer)
                    request = self._request
                    sender = self._sender
                    if request is None or sender is None:
                        raise ProtocolError("send without an active request")
                    payload = pack_intent_payload(
                        intent,
                        packed_planes(intent),
                        self._planes,
                        len(request.prompt_token_ids) - 1,
                    )
                    if self._first_send_perf_ns == 0:
                        self._first_send_perf_ns = time.perf_counter_ns()
                    sender.send(intent, payload)
                finally:
                    self._send_queue.task_done()
        except Exception as error:
            self._send_error = error
            while True:
                try:
                    self._send_queue.get_nowait()
                except queue.Empty:
                    return
                self._send_queue.task_done()

    def wait_for_save(self) -> None:
        if self._request is None:
            return
        sender = self._sender
        request = self._request
        if sender is None:
            raise ProtocolError("Muse sender was not initialized")
        committed = False
        try:
            if _REQUEST_CANCELLATION.is_set():
                return
            if self._layers_saved != EXPECTED_LAYERS:
                raise ProtocolError(
                    f"incomplete Muse layer set: received {self._layers_saved}"
                )
            wait_started_ns = time.perf_counter_ns()
            streamed_at_wait_entry = self._next_intent
            # Every layer's D2H was enqueued during prefill and the sender
            # thread has been streaming the whole time; enqueue the tail
            # intents (NoPE tiles, which need the last NoPE layer), wait for
            # the wire to drain, and stop the thread.
            while self._next_intent < len(self._intents):
                self._send_queue.put(self._intents[self._next_intent])
                self._next_intent += 1
            self._send_queue.put(None)
            self._sender_thread.join()
            if self._send_error is not None:
                # The host watcher closes the Unix request client after it
                # observes the receiver's control connection disappear. Give
                # that independent signal a small race window before treating
                # a data-plane loss as an ordinary producer failure.
                if _REQUEST_CANCELLATION.wait(timeout=0.25):
                    return
                raise ProtocolError(f"Muse sender thread failed: {self._send_error}")
            if len(self._materialized) != EXPECTED_LAYERS:
                raise ProtocolError(
                    f"sender thread materialized {len(self._materialized)} "
                    f"of {EXPECTED_LAYERS} layers"
                )
            sent_ns = time.perf_counter_ns()
            # The seam hash is layer-ordered regardless of send order.
            seam = hashlib.sha256()
            layer_hashes: list[dict[str, Any]] = []
            for layer in range(EXPECTED_LAYERS):
                digest = self._layer_digests[layer]
                seam.update(layer.to_bytes(2, "little"))
                seam.update(bytes.fromhex(digest))
                layer_hashes.append({"layer": layer, "kv_sha256": digest})
            from muser_vllm.dflash_capture import finish_capture_for_connector

            dflash_features = finish_capture_for_connector()
            dflash_ready_ns = time.perf_counter_ns()
            transfer = sender.seal()
            sealed_ns = time.perf_counter_ns()
            committed = True
            publish_receipt(
                {
                    "schema": (
                        "muser.spark-nvfp4-prefill.v1"
                        if self._producer_mode == "exact"
                        else "muser.spark-nvfp4-prefill.v2"
                    ),
                    "vllm_commit": PINNED_VLLM_COMMIT,
                    "producer_mode": self._producer_mode,
                    "prompt_token_count": len(request.prompt_token_ids),
                    "prefill_chunks": request.prefill_chunks,
                    "chunked_prefill": request.prefill_chunks > 1,
                    "prefix_cut": request.handoff.get("prefix_cut", 0),
                    "token_ids_sha256": token_ids_sha256(request.prompt_token_ids),
                    "layer_kv_sha256": layer_hashes,
                    "seam_sha256": seam.hexdigest(),
                    "phase_ns": {
                        "scheduled_to_connector_start": max(
                            0, self._started_perf_ns - request.scheduled_perf_ns
                        ),
                        "first_layer_offset": max(
                            0, self._first_layer_perf_ns - self._started_perf_ns
                        ),
                        "last_layer_enqueue_offset": max(
                            0, self._last_layer_perf_ns - self._started_perf_ns
                        ),
                        "wait_for_save_offset": max(
                            0, wait_started_ns - self._started_perf_ns
                        ),
                        # D2H and host materialization now happen per layer in
                        # save_kv_layer; these offsets prove the streaming
                        # overlap (first send precedes the last D2H).
                        "d2h_wait": 0,
                        "d2h_complete_offset": max(
                            0, self._last_materialized_perf_ns - self._started_perf_ns
                        ),
                        "first_segment_sent_offset": max(
                            0, self._first_send_perf_ns - self._started_perf_ns
                        ),
                        "streamed_segments_at_wait": streamed_at_wait_entry,
                        "host_materialize_hash": 0,
                        "pack_send": max(0, sent_ns - wait_started_ns),
                        "dflash_build": max(0, dflash_ready_ns - sent_ns),
                        "seal": max(0, sealed_ns - dflash_ready_ns),
                        "connector_total": sealed_ns - self._started_perf_ns,
                    },
                    "handoff": transfer,
                    "dflash_features": dflash_features,
                }
            )
        finally:
            if not committed:
                if self._sender_thread is not None and self._sender_thread.is_alive():
                    self._send_queue.put(None)
                    self._sender_thread.join(timeout=5)
                if self._sender_thread is not None and self._sender_thread.is_alive():
                    # Socket shutdown is the only safe way to interrupt a
                    # sender blocked in TLS I/O. It owns protocol writes; this
                    # thread must not race it by emitting an abort frame.
                    sender.interrupt()
                    self._sender_thread.join(timeout=5)
                else:
                    sender.abort("vllm producer failure")
            self._request = None
            self._sender = None
            self._pending = {}
            self._copy_stream = None
            self._first_layer_perf_ns = 0
            self._last_layer_perf_ns = 0
            self._intents = []
            self._intent_layers = []
            self._next_intent = 0
            self._planes = {}
            self._layer_digests = {}
            self._materialized = set()
            self._first_send_perf_ns = 0
            self._last_materialized_perf_ns = 0
            self._layers_saved = 0
            self._send_queue = None
            self._sender_thread = None
            self._send_error = None

    def _extract_pair(
        self,
        kv_layer: torch.Tensor,
        attention_slot_mapping: torch.Tensor,
        block_table: torch.Tensor | None,
        cached_token_count: int,
    ) -> torch.Tensor:
        if kv_layer.ndim != 4:
            raise ProtocolError(f"unexpected HND KV rank/shape {tuple(kv_layer.shape)}")
        if kv_layer.shape[1] != EXPECTED_KV_HEADS:
            raise ProtocolError(f"unexpected HND KV head axis {tuple(kv_layer.shape)}")
        if kv_layer.shape[2] != self._block_size:
            raise ProtocolError(f"unexpected HND block axis {tuple(kv_layer.shape)}")
        if kv_layer.shape[3] != 2 * EXPECTED_HEAD_DIM:
            raise ProtocolError(f"unexpected HND packed K/V axis {tuple(kv_layer.shape)}")
        # The attention backend's slot mapping is authoritative. Scheduler
        # block IDs are logical allocation metadata in vLLM V2 and are not a
        # stable substitute for the physical scatter slots used by the
        # FlashAttention cache write. The full unchunked step includes the
        # boundary token; Handoff intentionally exports only its prefix.
        if attention_slot_mapping.ndim != 1:
            raise ProtocolError(
                "unexpected FlashAttention slot mapping shape "
                f"{tuple(attention_slot_mapping.shape)}"
            )
        new_tokens = attention_slot_mapping.numel()
        if new_tokens > cached_token_count + 1:
            raise ProtocolError(
                "FlashAttention slot mapping covers "
                f"{new_tokens} tokens, expected at most {cached_token_count + 1}"
            )
        hit = cached_token_count + 1 - new_tokens
        if hit == 0:
            slots = attention_slot_mapping[:cached_token_count].to(
                device=kv_layer.device, non_blocking=True
            )
        else:
            # Prefix-cache hit: the slot mapping covers only the newly
            # computed tokens (the last one is the held boundary). The cached
            # prefix lives in the request's block table; gather those physical
            # slots and fail closed if the table cannot prove them.
            if block_table is None or block_table.ndim != 2 or block_table.shape[0] != 1:
                raise ProtocolError(
                    "prefix-cache hit without a single-request block table"
                )
            table = block_table[0].to(device=kv_layer.device, dtype=torch.long)
            needed_blocks = (hit + self._block_size - 1) // self._block_size
            if table.numel() < needed_blocks or bool((table[:needed_blocks] < 0).any()):
                raise ProtocolError("prefix-cache block table is short or invalid")
            positions = torch.arange(hit, device=kv_layer.device)
            prefix_slots = (
                table[positions // self._block_size] * self._block_size
                + positions % self._block_size
            )
            slots = torch.cat(
                [
                    prefix_slots,
                    attention_slot_mapping[: new_tokens - 1].to(
                        device=kv_layer.device, dtype=torch.long, non_blocking=True
                    ),
                ]
            )
        if slots.numel() != cached_token_count:
            raise ProtocolError(
                f"gathered {slots.numel()} slots, expected {cached_token_count}"
            )
        block_indices = slots // self._block_size
        offsets = slots % self._block_size
        selected = kv_layer[block_indices, :, offsets, :]
        if tuple(selected.shape[1:]) != (
            EXPECTED_KV_HEADS,
            2 * EXPECTED_HEAD_DIM,
        ):
            raise ProtocolError(f"unexpected selected HND KV shape {tuple(selected.shape)}")
        key, value = selected.split(EXPECTED_HEAD_DIM, dim=-1)
        # The exact route writes canonical interleaved keys into vLLM's cache.
        # Stock native vLLM writes post-RoPE NeoX half-split keys. Transform
        # only the gathered export view so the hook has zero effect on model
        # computation while both producer modes satisfy the Mac seam ABI.
        if self._producer_mode == "native":
            order = torch.tensor(
                neox_to_interleaved_order(EXPECTED_HEAD_DIM),
                dtype=torch.long,
                device=key.device,
            )
            key = key.index_select(-1, order)
        return torch.stack((key, value), dim=0)

    def _sender_args(self, handoff: dict[str, Any]) -> SimpleNamespace:
        base = {"generation", "receiver_host", "receiver_port", "transfer_id"}
        dflash = {
            "dflash_session",
            "dflash_identity_sha256",
            "dflash_kv_heads",
            "dflash_head_dim",
            "dflash_context_layers",
            "dflash_context_elements_per_token",
            "dflash_context_sink_size",
            "dflash_context_window_size",
        }
        delta = {"prefix_cut"}
        if set(handoff) not in (base | delta, base | dflash | delta, base | dflash, base):
            raise ProtocolError(
                "handoff context must be target-only or the complete DFlash schema"
            )
        if not isinstance(handoff["generation"], int) or handoff["generation"] < 1:
            raise ProtocolError("handoff generation must be positive")
        prefix_cut = handoff.get("prefix_cut", 0)
        if not isinstance(prefix_cut, int) or prefix_cut < 0:
            raise ProtocolError("handoff prefix_cut must be a nonnegative integer")
        merged = dict(self._extra)
        merged.update(handoff)
        merged.setdefault("timeout_seconds", 900)
        # MelonDMA RDMA transport: opt-in via this container's own
        # environment (set at `docker run` time in muser_native_prefilld.py;
        # defaults to unchanged "tcp" behavior). connect_wire() in
        # muser_v2_send.py reads these off the DeferredHandoffV2Sender args
        # namespace built from this dict.
        merged.setdefault("transport", os.environ.get("MUSER_TRANSPORT", "tcp"))
        merged.setdefault("rdma_dev", os.environ.get("MUSER_RDMA_DEV", "rocep1s0f1"))
        merged.setdefault("rdma_gid", int(os.environ.get("MUSER_RDMA_GID", "2")))
        merged.update({
                "dflash_session": handoff.get("dflash_session"),
                "dflash_identity_sha256": handoff.get("dflash_identity_sha256"),
                "dflash_kv_heads": handoff.get("dflash_kv_heads"),
                "dflash_head_dim": handoff.get("dflash_head_dim"),
                "dflash_context_layers": handoff.get("dflash_context_layers"),
                "dflash_context_elements_per_token": handoff.get(
                    "dflash_context_elements_per_token"
                ),
                "dflash_context_sink_size": handoff.get("dflash_context_sink_size"),
                "dflash_context_window_size": handoff.get(
                    "dflash_context_window_size"
                ),
                "multimodal_projector_sha256": None,
                "multimodal_preprocessing_sha256": None,
                "multimodal_image_sequence_sha256": None,
            })
        return SimpleNamespace(**merged)

    @staticmethod
    def _parse_layer(layer_name: str) -> int:
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", layer_name)
        if match is None:
            raise ProtocolError(f"cannot parse Muse layer index from {layer_name!r}")
        layer = int(match.group(1))
        if not 0 <= layer < EXPECTED_LAYERS:
            raise ProtocolError(f"Muse layer index {layer} is out of range")
        return layer

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        if num_external_tokens:
            raise ProtocolError("producer-only connector cannot load external tokens")

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        metadata = MuseConnectorMetadata()
        for req_id in scheduler_output.finished_req_ids:
            self._scheduled_requests.pop(req_id, None)
        preempted = set(scheduler_output.preempted_req_ids or ())
        if preempted.intersection(self._scheduled_requests):
            raise ProtocolError("a chunked Muse prefill was preempted before handoff")

        cached = scheduler_output.scheduled_cached_reqs
        scheduled_count = len(scheduler_output.scheduled_new_reqs) + len(cached.req_ids)
        if scheduled_count > 1:
            raise ProtocolError("Muse Handoff V2 accepts one request at a time")
        scheduled_perf_ns = time.perf_counter_ns()
        for request in scheduler_output.scheduled_new_reqs:
            token_ids = list(request.prompt_token_ids or [])
            if len(token_ids) < 2:
                raise ProtocolError("handoff requires at least two prompt tokens")
            if request.num_computed_tokens and not self._prefix_caching:
                raise ProtocolError("Muse producer requires a fresh full prefill")
            computed = int(request.num_computed_tokens or 0)
            scheduled = scheduler_output.num_scheduled_tokens[request.req_id]
            remaining = len(token_ids) - computed
            if computed < 0 or scheduled <= 0 or scheduled > remaining:
                raise ProtocolError("invalid initial Muse prefill schedule")
            if len(request.block_ids) != 1:
                raise ProtocolError("Muse producer requires one KV cache group")
            extra_args = getattr(request.sampling_params, "extra_args", None) or {}
            transfer = extra_args.get("kv_transfer_params") or {}
            handoff = transfer.get("muser_handoff")
            if extra_args.get("muser_startup_warmup") is True:
                if transfer:
                    raise ProtocolError("startup warmup cannot carry transfer state")
                request_meta = None
            else:
                if not isinstance(handoff, dict):
                    raise ProtocolError(
                        "request is missing kv_transfer_params.muser_handoff"
                    )
                request_meta = MuseRequestMeta(
                    prompt_token_ids=token_ids,
                    handoff=dict(handoff),
                    scheduled_perf_ns=scheduled_perf_ns,
                )
            pending = _ScheduledRequest(
                meta=request_meta,
                prompt_token_count=len(token_ids),
            )
            if scheduled == remaining:
                if request_meta is not None:
                    metadata.requests.append(request_meta)
            else:
                self._scheduled_requests[request.req_id] = pending

        for index, req_id in enumerate(cached.req_ids):
            if req_id in cached.resumed_req_ids:
                self._scheduled_requests.pop(req_id, None)
                raise ProtocolError("a chunked Muse prefill resumed after preemption")
            pending = self._scheduled_requests.get(req_id)
            output_tokens = int(cached.num_output_tokens[index])
            if pending is None:
                # Decode steps after a completed handoff remain ordinary
                # cached requests. A context-phase chunk without scheduler
                # state means the connector missed the first chunk.
                if output_tokens == 0:
                    raise ProtocolError("unknown cached Muse prefill request")
                continue
            if output_tokens != 0:
                self._scheduled_requests.pop(req_id, None)
                raise ProtocolError("Muse decode began before final prefill handoff")
            computed = int(cached.num_computed_tokens[index])
            scheduled = scheduler_output.num_scheduled_tokens[req_id]
            remaining = pending.prompt_token_count - computed
            if computed <= 0 or scheduled <= 0 or scheduled > remaining:
                self._scheduled_requests.pop(req_id, None)
                raise ProtocolError("invalid cached Muse prefill schedule")
            pending.chunks += 1
            if scheduled == remaining:
                self._scheduled_requests.pop(req_id, None)
                if pending.meta is not None:
                    pending.meta.scheduled_perf_ns = scheduled_perf_ns
                    pending.meta.prefill_chunks = pending.chunks
                    metadata.requests.append(pending.meta)
        return metadata

    def request_finished(
        self, request: "Request", block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None

    def shutdown(self) -> None:
        if self._sender is not None:
            self._sender.abort("producer shutdown")
