"""Cross-vendor RMSNorm arithmetic shared with muser-engine Metal."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any


SCHEMA = "muser.spark-exact-rms-norm.v3"
_FIRST_USE = True
_STAGE_CAPTURE_ENABLED = False
_STAGE_CAPTURE_DONE = False
_ORIGINAL_DECODER_INIT: Any = None


try:
    import torch
    import triton
    import triton.language as tl
except ImportError:  # Host-side receipt/unit tools do not require CUDA/Triton.
    torch = None
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _sqrt_rn(value):
        return tl.inline_asm_elementwise(
            "sqrt.rn.f32 $0, $1;",
            "=f,f",
            [value],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )


    @triton.jit
    def _div_rn(numerator, denominator):
        return tl.inline_asm_elementwise(
            "div.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [numerator, denominator],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )


    @triton.jit
    def _add_rn(left, right):
        return tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [left, right],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )


    @triton.jit
    def _fma_rn(left, right, accumulator):
        return tl.inline_asm_elementwise(
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            [left, right, accumulator],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )


    @triton.jit
    def _shuffle_down_16(value):
        return tl.inline_asm_elementwise(
            "shfl.sync.down.b32 $0, $1, 16, 0x1f, 0xffffffff;",
            "=f,f",
            [value],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )


    @triton.jit
    def _shuffle_down_8(value):
        return tl.inline_asm_elementwise(
            "shfl.sync.down.b32 $0, $1, 8, 0x1f, 0xffffffff;",
            "=f,f",
            [value],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )


    @triton.jit
    def _shuffle_down_4(value):
        return tl.inline_asm_elementwise(
            "shfl.sync.down.b32 $0, $1, 4, 0x1f, 0xffffffff;",
            "=f,f",
            [value],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )


    @triton.jit
    def _shuffle_down_2(value):
        return tl.inline_asm_elementwise(
            "shfl.sync.down.b32 $0, $1, 2, 0x1f, 0xffffffff;",
            "=f,f",
            [value],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )


    @triton.jit
    def _shuffle_down_1(value):
        return tl.inline_asm_elementwise(
            "shfl.sync.down.b32 $0, $1, 1, 0x1f, 0xffffffff;",
            "=f,f",
            [value],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )


    @triton.jit
    def _broadcast_first(value):
        return tl.inline_asm_elementwise(
            "shfl.sync.idx.b32 $0, $1, 0, 0x1f, 0xffffffff;",
            "=f,f",
            [value],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )


    @triton.jit
    def _metal_warp_sum(value):
        value = _add_rn(value, _shuffle_down_16(value))
        value = _add_rn(value, _shuffle_down_8(value))
        value = _add_rn(value, _shuffle_down_4(value))
        value = _add_rn(value, _shuffle_down_2(value))
        value = _add_rn(value, _shuffle_down_1(value))
        return _broadcast_first(value)


    @triton.jit
    def _exact_rms_norm_kernel(
        input_ptr,
        weight_ptr,
        output_ptr,
        eps: tl.constexpr,
        width: tl.constexpr,
        with_scale: tl.constexpr,
        weight_offset: tl.constexpr,
    ):
        row = tl.program_id(0)
        lane = tl.arange(0, 32)
        partial = tl.zeros((32,), tl.float32)
        for base in tl.range(0, width, 32, loop_unroll_factor=1):
            value = tl.load(input_ptr + row * width + base + lane).to(tl.float32)
            partial = tl.fma(value, value, partial)
        total = _metal_warp_sum(partial)
        mean = tl.fma(total, 1.0 / width, eps)
        inverse = _div_rn(1.0, _sqrt_rn(mean))
        for base in tl.range(0, width, 32, loop_unroll_factor=1):
            value = tl.load(input_ptr + row * width + base + lane).to(tl.float32)
            normalized = tl.fma(value, inverse, 0.0)
            if with_scale:
                weight = tl.load(weight_ptr + base + lane).to(tl.float32)
                normalized = tl.fma(normalized, weight + weight_offset, 0.0)
            tl.store(output_ptr + row * width + base + lane, normalized)


    @triton.jit
    def _exact_scale_kernel(
        input_ptr,
        weight_ptr,
        output_ptr,
        count: tl.constexpr,
        width: tl.constexpr,
        weight_offset: tl.constexpr,
    ):
        index = tl.program_id(0) * 256 + tl.arange(0, 256)
        mask = index < count
        value = tl.load(input_ptr + index, mask=mask).to(tl.float32)
        weight = tl.load(weight_ptr + index % width, mask=mask).to(tl.float32)
        multiplier = _add_rn(weight, float(weight_offset))
        scaled = _fma_rn(value, multiplier, 0.0)
        tl.store(output_ptr + index, scaled, mask=mask)


    @torch.library.triton_op("muser::exact_rms_norm", mutates_args={})
    def exact_rms_norm(
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        with_scale: bool,
        weight_offset: int,
    ) -> torch.Tensor:
        if input.ndim < 1 or input.shape[-1] % 32:
            raise ValueError("exact RMSNorm width must be positive and divisible by 32")
        if input.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("exact RMSNorm input must be FP16 or BF16")
        if with_scale and (weight.ndim != 1 or weight.numel() != input.shape[-1]):
            raise ValueError("exact RMSNorm weight geometry differs from its input")
        matrix = input.reshape(-1, input.shape[-1])
        output = torch.empty_like(matrix)
        torch.library.wrap_triton(_exact_rms_norm_kernel)[(matrix.shape[0],)](
            matrix,
            weight,
            output,
            eps=eps,
            width=matrix.shape[1],
            with_scale=with_scale,
            weight_offset=weight_offset,
            num_warps=1,
        )
        return output.view_as(input)


    @torch.library.triton_op("muser::exact_split_rms_norm", mutates_args={})
    def exact_split_rms_norm(
        input: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        weight_offset: int,
    ) -> torch.Tensor:
        """RMS, dtype boundary, then scale, matching Muse sandwich norms."""
        if weight.ndim != 1 or weight.numel() != input.shape[-1]:
            raise ValueError("split RMSNorm weight geometry differs from its input")
        normalized = exact_rms_norm(input, weight, eps, False, 0)
        output = torch.empty_like(normalized)
        count = normalized.numel()
        torch.library.wrap_triton(_exact_scale_kernel)[(triton.cdiv(count, 256),)](
            normalized,
            weight,
            output,
            count=count,
            width=input.shape[-1],
            weight_offset=weight_offset,
            num_warps=4,
        )
        return output

else:

    def exact_rms_norm(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("exact producer RMSNorm requires Torch and Triton")

    def exact_split_rms_norm(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("exact producer split RMSNorm requires Torch and Triton")


def vllm_exact_rms_norm_forward(module: Any, hidden_states: Any) -> Any:
    """Drop-in forward method for vLLM's Muse Glimmer RMSNorm."""
    global _FIRST_USE
    weight = module.weight if module.with_scale else hidden_states
    if _FIRST_USE:
        print(
            "[muser-exact-rms-norm] first-use "
            f"shape={tuple(hidden_states.shape)} with_scale={module.with_scale}",
            flush=True,
        )
        _FIRST_USE = False
    return exact_rms_norm(
        hidden_states,
        weight,
        float(module.eps),
        bool(module.with_scale),
        int(module.weight_offset),
    )


def set_exact_stage_capture_enabled(enabled: bool) -> None:
    global _STAGE_CAPTURE_ENABLED
    _STAGE_CAPTURE_ENABLED = bool(enabled)


def _capture_stage(directory: Path, name: str, values: Any) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    payload = (
        values[:32]
        .detach()
        .to(torch.float16)
        .contiguous()
        .cpu()
        .numpy()
        .tobytes()
    )
    path = directory / f"{name}.f16"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)


def vllm_exact_decoder_layer_init(module: Any, *args: Any, **kwargs: Any) -> None:
    """Retain the layer index needed by the split next-layer norm contract."""
    if _ORIGINAL_DECODER_INIT is None:
        raise RuntimeError("exact decoder init was invoked before installation")
    prefix = kwargs.get("prefix")
    if prefix is None and len(args) >= 4:
        prefix = args[3]
    if not isinstance(prefix, str):
        raise RuntimeError("Muse decoder layer has no stable module prefix")
    match = re.search(r"(?:^|\.)layers\.(\d+)$", prefix)
    if match is None:
        raise RuntimeError(f"cannot derive Muse layer index from prefix {prefix!r}")
    _ORIGINAL_DECODER_INIT(module, *args, **kwargs)
    module._muser_layer_index = int(match.group(1))


def vllm_exact_decoder_layer_forward(
    module: Any,
    positions: Any,
    hidden_states: Any,
    residual: Any,
) -> tuple[Any, Any]:
    """Preserve the observable split boundary in Muse sandwich norms."""
    global _STAGE_CAPTURE_DONE
    del residual
    capture_dir_raw = os.environ.get("MUSER_EXACT_STAGE_DIR")
    capture = bool(
        _STAGE_CAPTURE_ENABLED and not _STAGE_CAPTURE_DONE and capture_dir_raw
    )
    capture_dir = Path(capture_dir_raw) if capture_dir_raw else None
    residual = hidden_states
    if capture:
        _capture_stage(capture_dir, "layer_input", residual)
    norm = module.input_layernorm
    if module._muser_layer_index == 0:
        hidden_states = norm(hidden_states)
    else:
        hidden_states = exact_split_rms_norm(
            hidden_states, norm.weight, float(norm.eps), int(norm.weight_offset)
        )
    if capture:
        _capture_stage(capture_dir, "attn_norm", hidden_states)
    hidden_states = module.self_attn(positions=positions, hidden_states=hidden_states)
    if capture:
        _capture_stage(capture_dir, "attn_o_proj", hidden_states)
    norm = module.post_attention_layernorm
    hidden_states = exact_split_rms_norm(
        hidden_states, norm.weight, float(norm.eps), int(norm.weight_offset)
    )
    if capture:
        _capture_stage(capture_dir, "attn_post_norm", hidden_states)
    hidden_states = (residual + hidden_states).to(hidden_states.dtype)
    if capture:
        _capture_stage(capture_dir, "ffn_inp", hidden_states)

    residual = hidden_states
    norm = module.pre_feedforward_layernorm
    hidden_states = exact_split_rms_norm(
        hidden_states, norm.weight, float(norm.eps), int(norm.weight_offset)
    )
    if capture:
        _capture_stage(capture_dir, "ffn_norm", hidden_states)
    hidden_states = module.mlp(hidden_states)
    if capture:
        _capture_stage(capture_dir, "ffn_out", hidden_states)
    norm = module.post_feedforward_layernorm
    hidden_states = exact_split_rms_norm(
        hidden_states, norm.weight, float(norm.eps), int(norm.weight_offset)
    )
    if capture:
        _capture_stage(capture_dir, "ffn_post_norm", hidden_states)
    hidden_states = (residual + hidden_states).to(hidden_states.dtype)
    if capture:
        _capture_stage(capture_dir, "layer_out", hidden_states)
        _STAGE_CAPTURE_DONE = True
    # The draft artifact names raw target layers 2/14/26/38/50. Its encoder
    # consumes the input to each of those layers, which is the complete output
    # of zero-based layers 1/13/25/37/49. Capture only after both the attention
    # and feed-forward residual branches have crossed their pinned fp16
    # boundary; capturing above at ffn_inp silently records the wrong state
    # even though target KV and logits remain exact.
    from muser_vllm.dflash_capture import capture_layer

    capture_layer(module._muser_layer_index, hidden_states)
    return hidden_states, residual


def install_exact_rms_norm() -> dict[str, Any]:
    """Patch Muse Glimmer before model construction and fail closed."""
    global _ORIGINAL_DECODER_INIT
    import torch
    import triton
    from vllm.model_executor.models import muse_glimmer

    _ORIGINAL_DECODER_INIT = muse_glimmer.MuseGlimmerDecoderLayer.__init__
    muse_glimmer.MuseGlimmerDecoderLayer.__init__ = vllm_exact_decoder_layer_init
    muse_glimmer.MuseGlimmerRMSNorm.forward = vllm_exact_rms_norm_forward
    muse_glimmer.MuseGlimmerDecoderLayer.forward = vllm_exact_decoder_layer_forward
    if muse_glimmer.MuseGlimmerRMSNorm.forward is not vllm_exact_rms_norm_forward:
        raise RuntimeError("failed to install exact Muse Glimmer RMSNorm")
    if (
        muse_glimmer.MuseGlimmerDecoderLayer.forward
        is not vllm_exact_decoder_layer_forward
    ):
        raise RuntimeError("failed to install exact Muse sandwich-norm boundaries")
    if (
        muse_glimmer.MuseGlimmerDecoderLayer.__init__
        is not vllm_exact_decoder_layer_init
    ):
        raise RuntimeError("failed to install exact Muse layer-index binding")
    return {
        "schema": SCHEMA,
        "implementation": "triton-one-warp-lane-stride32",
        "patched_targets": [
            "MuseGlimmerRMSNorm.forward",
            "MuseGlimmerDecoderLayer.__init__",
            "MuseGlimmerDecoderLayer.forward",
        ],
        "reduction_contract": "lane-stride32-fma_then-binary-tree",
        "inverse_contract": "sqrt.rn.f32_then-div.rn.f32",
        "sandwich_norm_contract": "rms-f16-boundary-scale-f16-boundary",
        "input_norm_contract": "layer0-fused_layers1plus-split",
        "diagnostic_stage_capture": bool(os.environ.get("MUSER_EXACT_STAGE_DIR")),
        "output_boundary": "model-dtype",
        "torch": torch.__version__,
        "triton": triton.__version__,
    }
