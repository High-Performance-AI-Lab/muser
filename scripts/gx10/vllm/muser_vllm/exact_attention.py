"""Cross-vendor Muse attention arithmetic shared with muser-engine Metal."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .exact_rope import exact_rope, metadata as exact_rope_metadata


SCHEMA = "muser.spark-exact-attention.v6"
_ENABLED = False
_FIRST_USE = True
_ORIGINAL_FORWARD: Any = None


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
    def _fma_rn(left, right, addend):
        return tl.inline_asm_elementwise(
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            [left, right, addend],
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
    def _mul_rn(left, right):
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [left, right],
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


    # Triton's generic reduction may choose a different tree.  The Metal seam
    # contract is five ordered shuffle-down additions and a lane-zero
    # broadcast, so spell out that graph in PTX.
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
    def _exact_expf(value):
        value = value.to(tl.float32)
        rounding = 12582912.0
        z = _fma_rn(value, 1.4426950216293335, rounding)
        n = _add_rn(z, -rounding)
        reduced = _fma_rn(-n, 0.693145751953125, value)
        reduced = _fma_rn(-n, 1.428606765330187e-6, reduced)
        exponent = z.to(tl.uint32, bitcast=True) << 23
        scale = (exponent + 0x3F800000).to(tl.float32, bitcast=True)
        squared = _mul_rn(reduced, reduced)
        p1 = _fma_rn(0.008247390389442444, reduced, 0.04189976677298546)
        p2 = _fma_rn(0.16668395698070526, reduced, 0.4999912679195404)
        p2 = _fma_rn(p1, squared, p2)
        correction = _fma_rn(
            p2, squared, _mul_rn(0.9999994039535522, reduced)
        )
        result = _fma_rn(correction, scale, scale)
        return tl.where(value < -80.0, 0.0, result)


    @triton.jit
    def _exact_attention_gate_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        gate_ptr,
        output_ptr,
        token_count,
        n_heads: tl.constexpr,
        n_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        attention_scale: tl.constexpr,
        sliding_window: tl.constexpr,
    ):
        query_row = tl.program_id(0)
        head = tl.program_id(1)
        lane = tl.arange(0, 32)
        kv_head = head // (n_heads // n_kv_heads)
        query_base = (query_row * n_heads + head) * head_dim

        query0 = tl.load(query_ptr + query_base + lane).to(tl.float32)
        query1 = tl.load(query_ptr + query_base + 32 + lane).to(tl.float32)
        query2 = tl.load(query_ptr + query_base + 64 + lane).to(tl.float32)
        query3 = tl.load(query_ptr + query_base + 96 + lane).to(tl.float32)

        running_max = tl.full((32,), -3.4028234663852886e38, tl.float32)
        denominator = tl.zeros((32,), tl.float32)
        accumulator0 = tl.zeros((32,), tl.float32)
        accumulator1 = tl.zeros((32,), tl.float32)
        accumulator2 = tl.zeros((32,), tl.float32)
        accumulator3 = tl.zeros((32,), tl.float32)
        first_visible = tl.maximum(0, query_row + 1 - sliding_window)

        # Preserve the chronological online-softmax recurrence without
        # specializing it into one copy per prompt token.  static_range made
        # a 2048-token prompt emit a ~30 MiB kernel and spend minutes in LLVM;
        # this runtime loop is the same scalar order with unrolling disabled.
        for key_row in tl.range(0, token_count, loop_unroll_factor=1):
            active = (key_row >= first_visible) & (key_row <= query_row)
            key_base = (key_row * n_kv_heads + kv_head) * head_dim
            partial = tl.zeros((32,), tl.float32)
            partial = _fma_rn(
                query0,
                tl.load(key_ptr + key_base + lane).to(tl.float32),
                partial,
            )
            partial = _fma_rn(
                query1,
                tl.load(key_ptr + key_base + 32 + lane).to(tl.float32),
                partial,
            )
            partial = _fma_rn(
                query2,
                tl.load(key_ptr + key_base + 64 + lane).to(tl.float32),
                partial,
            )
            partial = _fma_rn(
                query3,
                tl.load(key_ptr + key_base + 96 + lane).to(tl.float32),
                partial,
            )
            score = _mul_rn(_metal_warp_sum(partial), attention_scale)
            next_max = tl.where(active, tl.maximum(running_max, score), running_max)
            old_factor = tl.where(
                active, _exact_expf(_add_rn(running_max, -next_max)), 1.0
            )
            new_factor = tl.where(
                active, _exact_expf(_add_rn(score, -next_max)), 0.0
            )
            denominator = _fma_rn(denominator, old_factor, new_factor)

            value_base = (key_row * n_kv_heads + kv_head) * head_dim
            accumulator0 = _fma_rn(
                tl.load(value_ptr + value_base + lane).to(tl.float32),
                new_factor,
                _mul_rn(accumulator0, old_factor),
            )
            accumulator1 = _fma_rn(
                tl.load(value_ptr + value_base + 32 + lane).to(tl.float32),
                new_factor,
                _mul_rn(accumulator1, old_factor),
            )
            accumulator2 = _fma_rn(
                tl.load(value_ptr + value_base + 64 + lane).to(tl.float32),
                new_factor,
                _mul_rn(accumulator2, old_factor),
            )
            accumulator3 = _fma_rn(
                tl.load(value_ptr + value_base + 96 + lane).to(tl.float32),
                new_factor,
                _mul_rn(accumulator3, old_factor),
            )
            running_max = next_max

        output_base = (query_row * n_heads + head) * head_dim
        attention0 = _div_rn(accumulator0, denominator)
        gate0 = tl.load(gate_ptr + output_base + lane).to(tl.float32)
        sigmoid0 = _div_rn(1.0, _add_rn(1.0, _exact_expf(-gate0)))
        tl.store(output_ptr + output_base + lane, _mul_rn(attention0, sigmoid0))
        attention1 = _div_rn(accumulator1, denominator)
        gate1 = tl.load(gate_ptr + output_base + 32 + lane).to(tl.float32)
        sigmoid1 = _div_rn(1.0, _add_rn(1.0, _exact_expf(-gate1)))
        tl.store(output_ptr + output_base + 32 + lane, _mul_rn(attention1, sigmoid1))
        attention2 = _div_rn(accumulator2, denominator)
        gate2 = tl.load(gate_ptr + output_base + 64 + lane).to(tl.float32)
        sigmoid2 = _div_rn(1.0, _add_rn(1.0, _exact_expf(-gate2)))
        tl.store(output_ptr + output_base + 64 + lane, _mul_rn(attention2, sigmoid2))
        attention3 = _div_rn(accumulator3, denominator)
        gate3 = tl.load(gate_ptr + output_base + 96 + lane).to(tl.float32)
        sigmoid3 = _div_rn(1.0, _add_rn(1.0, _exact_expf(-gate3)))
        tl.store(output_ptr + output_base + 96 + lane, _mul_rn(attention3, sigmoid3))


    @torch.library.triton_op("muser::exact_attention_gate", mutates_args={})
    def exact_attention_gate(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor,
        attention_scale: float,
        sliding_window: int,
    ) -> torch.Tensor:
        if query.ndim != 2 or key.ndim != 2 or value.ndim != 2 or gate.ndim != 2:
            raise ValueError("exact attention inputs must be rank-two packed token rows")
        if query.shape != gate.shape or key.shape != value.shape:
            raise ValueError("exact attention input shapes disagree")
        if query.shape[0] != key.shape[0] or query.shape[1] % 128 or key.shape[1] % 128:
            raise ValueError("exact attention geometry is invalid")
        if query.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("exact attention query must use the model dtype")
        if key.dtype != query.dtype or value.dtype != query.dtype or gate.dtype != query.dtype:
            raise TypeError("exact attention inputs must share one model dtype")
        if not all(tensor.is_contiguous() for tensor in (query, key, value, gate)):
            raise ValueError("exact attention inputs must use packed contiguous rows")
        token_count = query.shape[0]
        n_heads = query.shape[1] // 128
        n_kv_heads = key.shape[1] // 128
        if n_heads % n_kv_heads:
            raise ValueError("exact attention GQA head ratio is invalid")
        if sliding_window <= 0:
            sliding_window = token_count
        output = torch.empty_like(query)
        torch.library.wrap_triton(_exact_attention_gate_kernel)[(token_count, n_heads)](
            query,
            key,
            value,
            gate,
            output,
            token_count=token_count,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=128,
            attention_scale=float(attention_scale),
            sliding_window=int(sliding_window),
            num_warps=1,
        )
        return output

else:

    def exact_attention_gate(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("exact producer attention requires Torch and Triton")


def set_exact_attention_enabled(enabled: bool) -> None:
    global _ENABLED
    _ENABLED = bool(enabled)


def _pack_exact_attention_inputs(
    query: Any, key: Any, value: Any, gate: Any
) -> tuple[Any, Any, Any, Any]:
    """Materialize the packed row layout assumed by the pinned Triton kernel.

    vLLM's QKV projection is one contiguous ``[q | k | v]`` allocation.  Its
    V split therefore has the projection width as its row stride even though
    its logical row is only ``kv_size`` elements.  The exact kernel deliberately
    carries no dynamic strides, so passing that view makes token zero correct
    and every later token read from the wrong address.
    """
    return tuple(tensor.contiguous() for tensor in (query, key, value, gate))


def _neox_to_interleaved(values: Any, heads: int, head_dim: int) -> Any:
    return (
        values.reshape(-1, heads, 2, head_dim // 2)
        .transpose(-2, -1)
        .reshape(values.shape)
        .contiguous()
    )


def _interleaved_to_neox(values: Any, heads: int, head_dim: int) -> Any:
    return (
        values.reshape(-1, heads, head_dim // 2, 2)
        .transpose(-2, -1)
        .reshape(values.shape)
        .contiguous()
    )


def _capture_live_layer0(module: Any, name: str, values: Any) -> None:
    directory_raw = os.environ.get("MUSER_EXACT_STAGE_DIR")
    if module.layer_idx != 0 or not directory_raw:
        return
    directory = Path(directory_raw)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.f16"
    if path.exists():
        return
    payload = (
        values[:32]
        .detach()
        .to(torch.float16)
        .contiguous()
        .cpu()
        .numpy()
        .tobytes()
    )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)


def vllm_exact_attention_forward(
    module: Any, positions: Any, hidden_states: Any
) -> Any:
    """Drop-in Muse attention forward with an exact initial-prefill output."""
    global _FIRST_USE
    if not _ENABLED:
        return _ORIGINAL_FORWARD(module, positions, hidden_states)

    qkv, _ = module.qkv_proj(hidden_states)
    q, k, v = qkv.split([module.q_size, module.kv_size, module.kv_size], dim=-1)
    # The HF checkpoint stores Q/K rows in NEOX half-split order. The native
    # GGUF converter un-permutes those rows into adjacent RoPE pairs. Apply the
    # same fixed permutation before QK norm: its floating reduction tree sees
    # element order even though the mathematical norm is permutation-invariant.
    q = _neox_to_interleaved(q, module.num_heads, module.head_dim)
    k = _neox_to_interleaved(k, module.num_kv_heads, module.head_dim)
    if module.use_qk_norm:
        q = module.qk_norm(q.reshape(-1, module.head_dim)).reshape(-1, module.q_size)
        q = q * module.scale_query_by
        k = module.qk_norm(k.reshape(-1, module.head_dim)).reshape(-1, module.kv_size)
        q = q.to(v.dtype)
        k = k.to(v.dtype)
    if module.rotary_emb is not None:
        q_neox, k_neox = exact_rope(
            _interleaved_to_neox(q, module.num_heads, module.head_dim),
            _interleaved_to_neox(k, module.num_kv_heads, module.head_dim),
            module.rotary_emb.cos_sin_cache,
            positions,
            module.head_dim,
        )
        q = _neox_to_interleaved(q_neox, module.num_heads, module.head_dim)
        k = _neox_to_interleaved(k_neox, module.num_kv_heads, module.head_dim)

    if not module.use_output_gate:
        raise RuntimeError("exact Muse attention requires the output gate")
    gate, _ = module.output_gate_proj(hidden_states)
    _capture_live_layer0(module, "attn_q", q)
    _capture_live_layer0(module, "attn_k", k)
    _capture_live_layer0(module, "attn_v", v)
    _capture_live_layer0(module, "attn_gate", gate)
    if positions.ndim != 1 or positions.numel() != q.shape[0]:
        raise RuntimeError("exact Muse attention requires one packed sequence")
    expected = torch.arange(q.shape[0], device=positions.device, dtype=positions.dtype)
    if not torch.equal(positions, expected):
        raise RuntimeError("exact Muse attention requires an initial contiguous prefill")
    if _FIRST_USE:
        print(
            "[muser-exact-attention] first-use "
            f"shape={tuple(q.shape)} layer={module.layer_idx} rope={module.use_rope}",
            flush=True,
        )
        _FIRST_USE = False
    exact_q, exact_k, exact_v, exact_gate = _pack_exact_attention_inputs(
        q, k, v, gate
    )
    attention = exact_attention_gate(
        exact_q,
        exact_k,
        exact_v,
        exact_gate,
        float(module.scaling),
        int(module.config.sliding_window) if module.use_rope else 0,
    )
    _capture_live_layer0(module, "attn_gated", attention)
    # Preserve vLLM's cache writes and connector lifecycle only after the seam
    # result is materialized.  The backend may alias or mutate Q while
    # producing its ignored vendor-specific output; using Q afterward corrupts
    # causal scores beginning with the second prompt token.
    module.attn(q, k, v)
    _capture_live_layer0(module, "attn_q_after_cache_write", q)
    output, _ = module.o_proj(attention)
    return output


def install_exact_attention() -> dict[str, Any]:
    """Patch Muse attention before model construction and fail closed."""
    global _ORIGINAL_FORWARD
    import torch
    import triton
    from vllm.model_executor.models import muse_glimmer

    if _ORIGINAL_FORWARD is None:
        _ORIGINAL_FORWARD = muse_glimmer.MuseGlimmerAttention.forward
    muse_glimmer.MuseGlimmerAttention.forward = vllm_exact_attention_forward
    if muse_glimmer.MuseGlimmerAttention.forward is not vllm_exact_attention_forward:
        raise RuntimeError("failed to install exact Muse attention")
    return {
        "schema": SCHEMA,
        "implementation": "triton-one-warp-pinned-shuffle-down-online-softmax",
        "patched_target": "MuseGlimmerAttention.forward",
        "activation": "request-scoped",
        "dot_contract": "lane-stride32-fma_then-binary-tree",
        "qk_layout": "hf-neox-to-native-interleaved-before-qk-norm-and-cache",
        "softmax_contract": "chronological-online-pinned-expf",
        "rope": exact_rope_metadata(),
        "output_boundary": "sigmoid-gated-model-dtype",
        "torch": torch.__version__,
        "triton": triton.__version__,
    }
