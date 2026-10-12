"""Pinned vLLM connector for RedHat-prefix/Dudeman-target composition.

This is the inverse seam missing from the original producer-only connector.
Export mode captures the exact Handoff V2 portable prefix representation from
a RedHat native prefill. Import mode authenticates that bundle, converts only
the canonical key layout back to stock vLLM's NeoX cache layout, and installs
the exact (possibly sub-block) prefix into blocks allocated for a Dudeman
request. The held boundary token is then evaluated normally by Dudeman.

The implementation is intentionally synchronous and single-request. Its first
purpose is composite-genesis correctness; a serving implementation may pipeline
the copies only after this path is a retained oracle.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

from .composite_bundle import (
    CompositeBundleError,
    CompositeBundleWriter,
    EXPECTED_HEAD_DIM,
    EXPECTED_KV_HEADS,
    EXPECTED_LAYERS,
    bundle_root_sha256,
    load_hmac_key,
    read_bundle_manifest,
    read_layer_payload,
)
from .packing import interleaved_to_neox_order, neox_to_interleaved_order

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


@dataclass
class CompositeRequestMeta:
    request_id: str
    prompt_token_ids: list[int]
    cached_token_count: int
    block_ids: list[int]
    block_size: int
    operation: str

    def slot_mapping(self) -> torch.Tensor:
        if self.cached_token_count < 1:
            raise CompositeBundleError("composite request has an empty cached cut")
        required_blocks = (self.cached_token_count + self.block_size - 1) // self.block_size
        if len(self.block_ids) < required_blocks:
            raise CompositeBundleError("composite request has too few allocated cache blocks")
        blocks = torch.tensor(self.block_ids[:required_blocks], dtype=torch.long)
        offsets = torch.arange(self.block_size, dtype=torch.long)
        slots = (blocks[:, None] * self.block_size + offsets[None, :]).reshape(-1)
        return slots[: self.cached_token_count]


@dataclass
class CompositeConnectorMetadata(KVConnectorMetadata):
    requests: list[CompositeRequestMeta] = field(default_factory=list)


class MuserCompositeKvConnector(KVConnectorBase_V1):
    """One exact portable-prefix export or import per engine process."""

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
            raise CompositeBundleError("composite KV requires tensor_parallel_size=1")
        text = vllm_config.model_config.hf_text_config
        geometry = (
            int(text.num_hidden_layers),
            int(text.num_key_value_heads),
            int(text.head_dim),
        )
        if geometry != (EXPECTED_LAYERS, EXPECTED_KV_HEADS, EXPECTED_HEAD_DIM):
            raise CompositeBundleError(f"unexpected composite Muse geometry {geometry!r}")
        self._block_size = int(vllm_config.cache_config.block_size)
        self._extra = dict(
            vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        )
        required = {
            "bundle_path",
            "hmac_key_file",
            "hmac_key_id",
            "mode",
            "source_checkpoint_artifact_sha256",
            "source_checkpoint_revision",
            "source_engine_mode",
        }
        if set(self._extra) != required:
            raise CompositeBundleError(
                f"composite connector keys are {sorted(self._extra)}, expected {sorted(required)}"
            )
        self._mode = self._extra["mode"]
        if self._mode not in {"export", "import"}:
            raise CompositeBundleError("composite connector mode must be export or import")
        self._bundle = Path(self._extra["bundle_path"])
        if not self._bundle.is_absolute():
            raise CompositeBundleError("composite bundle path must be absolute")
        self._key_id = self._extra["hmac_key_id"]
        self._key = load_hmac_key(Path(self._extra["hmac_key_file"]))
        self._source_revision = self._extra["source_checkpoint_revision"]
        self._source_artifact = self._extra["source_checkpoint_artifact_sha256"]
        self._source_engine_mode = self._extra["source_engine_mode"]
        self._manifest: dict[str, Any] | None = None
        if self._mode == "import":
            self._manifest = read_bundle_manifest(
                self._bundle,
                key=self._key,
                expected_key_id=self._key_id,
                expected_source_artifact_sha256=self._source_artifact,
            )
        self._requests_need_load: dict[str, "Request"] = {}
        self._writer: CompositeBundleWriter | None = None
        self._active_export: CompositeRequestMeta | None = None
        self._import_receipt: dict[str, Any] | None = None

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, CompositeConnectorMetadata):
            raise CompositeBundleError("unexpected composite connector metadata")
        if not metadata.requests:
            return
        if len(metadata.requests) != 1:
            raise CompositeBundleError("composite connector requires one request")
        request = metadata.requests[0]
        if request.operation == "export":
            if self._mode != "export" or self._writer is not None:
                raise CompositeBundleError("composite export state is already active")
            self._writer = CompositeBundleWriter(
                self._bundle,
                token_ids=request.prompt_token_ids,
                source_checkpoint_revision=self._source_revision,
                source_checkpoint_artifact_sha256=self._source_artifact,
                source_engine_mode=self._source_engine_mode,
                hmac_key_id=self._key_id,
                hmac_key=self._key,
            )
            self._active_export = request
            return
        if request.operation != "import" or self._mode != "import":
            raise CompositeBundleError("composite request operation differs from process mode")
        self._install_import(forward_context, request)

    def _install_import(
        self, forward_context: "ForwardContext", request: CompositeRequestMeta
    ) -> None:
        manifest = self._manifest
        if manifest is None:
            raise CompositeBundleError("composite import has no authenticated manifest")
        if request.cached_token_count != manifest["cached_token_count"]:
            raise CompositeBundleError("composite scheduler/manifest cut differs")
        slots = request.slot_mapping()
        inverse = np.asarray(interleaved_to_neox_order(EXPECTED_HEAD_DIM), dtype=np.int64)
        layer_digests: list[dict[str, Any]] = []
        started_ns = time.perf_counter_ns()
        seen: set[int] = set()
        for layer_name, layer_module in forward_context.no_compile_layers.items():
            kv_layer = getattr(layer_module, "kv_cache", None)
            if kv_layer is None:
                continue
            layer = self._parse_layer(layer_name)
            if layer in seen:
                raise CompositeBundleError("duplicate composite target KV layer")
            seen.add(layer)
            self._validate_hnd_cache(kv_layer)
            payload = read_layer_payload(self._bundle, manifest, layer)
            source = np.frombuffer(payload, dtype="<f2").reshape(
                2,
                request.cached_token_count,
                EXPECTED_KV_HEADS,
                EXPECTED_HEAD_DIM,
            )
            key_neox = np.ascontiguousarray(source[0][..., inverse])
            value = np.ascontiguousarray(source[1])
            packed = np.concatenate((key_neox, value), axis=-1)
            source_tensor = torch.from_numpy(packed).to(
                device=kv_layer.device, dtype=torch.float16, non_blocking=False
            )
            device_slots = slots.to(device=kv_layer.device, non_blocking=False)
            block_indices = device_slots // self._block_size
            offsets = device_slots % self._block_size
            kv_layer[block_indices, :, offsets, :] = source_tensor
            layer_digests.append(
                {"layer": layer, "portable_sha256": manifest["layer_files"][layer]["sha256"]}
            )
        if seen != set(range(EXPECTED_LAYERS)):
            raise CompositeBundleError(f"composite target layer set is incomplete: {sorted(seen)}")
        torch.cuda.synchronize()
        self._import_receipt = {
            "bundle_root_sha256": bundle_root_sha256(manifest),
            "cached_token_count": request.cached_token_count,
            "layer_digests": sorted(layer_digests, key=lambda value: value["layer"]),
            "request_id": request.request_id,
            "wall_ns": time.perf_counter_ns() - started_ns,
        }

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        if self._active_export is None:
            return
        if not isinstance(attn_metadata, FlashAttentionMetadata):
            raise CompositeBundleError(
                f"composite export requires FlashAttentionMetadata, got {type(attn_metadata)!r}"
            )
        writer = self._writer
        if writer is None:
            raise CompositeBundleError("composite export writer disappeared")
        layer = self._parse_layer(layer_name)
        request = self._active_export
        pair = self._extract_portable_pair(
            kv_layer, attn_metadata.slot_mapping, request.cached_token_count
        )
        host = pair.detach().cpu().numpy().astype("<f2", copy=False)
        writer.write_layer(layer, host.tobytes(order="C"))

    def wait_for_save(self) -> None:
        if self._active_export is None:
            return
        writer = self._writer
        if writer is None:
            raise CompositeBundleError("composite export writer disappeared")
        try:
            writer.commit()
        except BaseException:
            writer.abort()
            raise
        finally:
            self._writer = None
            self._active_export = None

    def _extract_portable_pair(
        self,
        kv_layer: torch.Tensor,
        attention_slot_mapping: torch.Tensor,
        cached_token_count: int,
    ) -> torch.Tensor:
        self._validate_hnd_cache(kv_layer)
        if attention_slot_mapping.ndim != 1 or attention_slot_mapping.numel() < cached_token_count:
            raise CompositeBundleError("composite attention slot mapping is too short")
        slots = attention_slot_mapping[:cached_token_count].to(
            device=kv_layer.device, non_blocking=False
        )
        block_indices = slots // self._block_size
        offsets = slots % self._block_size
        selected = kv_layer[block_indices, :, offsets, :]
        key, value = selected.split(EXPECTED_HEAD_DIM, dim=-1)
        if self._source_engine_mode == "native":
            order = torch.tensor(
                neox_to_interleaved_order(EXPECTED_HEAD_DIM),
                dtype=torch.long,
                device=key.device,
            )
            key = key.index_select(-1, order)
        return torch.stack((key, value), dim=0).contiguous()

    def _validate_hnd_cache(self, kv_layer: torch.Tensor) -> None:
        if (
            kv_layer.ndim != 4
            or kv_layer.shape[1] != EXPECTED_KV_HEADS
            or kv_layer.shape[2] != self._block_size
            or kv_layer.shape[3] != 2 * EXPECTED_HEAD_DIM
            or kv_layer.dtype != torch.float16
        ):
            raise CompositeBundleError(f"unexpected composite HND KV shape {tuple(kv_layer.shape)}")

    @staticmethod
    def _parse_layer(layer_name: str) -> int:
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", layer_name)
        if match is None:
            raise CompositeBundleError(f"cannot parse Muse layer index from {layer_name!r}")
        layer = int(match.group(1))
        if not 0 <= layer < EXPECTED_LAYERS:
            raise CompositeBundleError("composite layer index is out of range")
        return layer

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        if self._mode != "import":
            return 0, False
        manifest = self._manifest
        assert manifest is not None
        prompt = list(request.prompt_token_ids or [])
        transcript = manifest["token_ids"]
        cached = int(manifest["cached_token_count"])
        if len(prompt) < len(transcript) or prompt[: len(transcript)] != transcript:
            return 0, False
        if num_computed_tokens >= cached:
            return 0, False
        return cached - num_computed_tokens, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        if num_external_tokens < 0:
            raise CompositeBundleError("negative composite external-token count")
        if num_external_tokens:
            if self._mode != "import" or request.request_id in self._requests_need_load:
                raise CompositeBundleError("unexpected duplicate composite load allocation")
            self._requests_need_load[request.request_id] = request

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        metadata = CompositeConnectorMetadata()
        if len(scheduler_output.scheduled_new_reqs) > 1:
            raise CompositeBundleError("composite connector accepts one request at a time")
        for request in scheduler_output.scheduled_new_reqs:
            token_ids = list(request.prompt_token_ids or [])
            if len(request.block_ids) != 1:
                raise CompositeBundleError("composite connector requires one KV cache group")
            if self._mode == "export":
                if request.num_computed_tokens or len(token_ids) < 2:
                    raise CompositeBundleError("composite export requires a fresh full prompt")
                scheduled = scheduler_output.num_scheduled_tokens[request.req_id]
                if scheduled != len(token_ids):
                    raise CompositeBundleError("composite export requires one unchunked prefill")
                metadata.requests.append(
                    CompositeRequestMeta(
                        request_id=request.req_id,
                        prompt_token_ids=token_ids,
                        cached_token_count=len(token_ids) - 1,
                        block_ids=list(request.block_ids[0]),
                        block_size=self._block_size,
                        operation="export",
                    )
                )
                continue
            pending = self._requests_need_load.get(request.req_id)
            if pending is None:
                continue
            manifest = self._manifest
            assert manifest is not None
            metadata.requests.append(
                CompositeRequestMeta(
                    request_id=request.req_id,
                    prompt_token_ids=token_ids,
                    cached_token_count=int(manifest["cached_token_count"]),
                    block_ids=list(request.block_ids[0]),
                    block_size=self._block_size,
                    operation="import",
                )
            )
            del self._requests_need_load[request.req_id]
        if self._requests_need_load:
            raise CompositeBundleError("composite load allocation was not scheduled")
        return metadata

    def request_finished(
        self, request: "Request", block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        return False, None

    def shutdown(self) -> None:
        if self._writer is not None:
            self._writer.abort()
            self._writer = None
            self._active_export = None

    @property
    def import_receipt(self) -> dict[str, Any] | None:
        return self._import_receipt
