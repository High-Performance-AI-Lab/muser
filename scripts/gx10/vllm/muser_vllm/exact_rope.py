"""Canonical Q30-NCO Muse RoPE shared with muser-engine Metal."""

from __future__ import annotations

from typing import Any

import numpy as np


SCHEMA = "muser.spark-exact-rope.v2"
SIN_Q30 = (0, 843_314_857, 0, -86_699_833, 0, 2_673_908, 0, -38_839)
COS_Q30 = (1_073_741_823, 0, -331_167_724, 0, 17_017_653, 0, -341_628)
FIXED_SCALE_INV = np.float32(2.0**-30)
PHASE_PER_RADIAN = 683_565_275.576_431_6
MUSE_THETA_SCALE = np.asarray(0x3F508AC1, dtype=np.uint32).view(np.float32)
MUSE_HEAD_DIM = 128
MUSE_FREQUENCY_BASE = 500_000.0


def _fixed_horner(coefficients: tuple[int, ...], x: np.ndarray) -> np.ndarray:
    """Vectorized signed-i32 Horner with the Rust/CUDA wrapping contract."""
    accumulator = np.full(x.shape, coefficients[-1], dtype=np.int64)
    x_i64 = x.astype(np.uint64).astype(np.int64)
    for coefficient in reversed(coefficients[:-1]):
        accumulator = (coefficient + ((accumulator * x_i64) >> 32)).astype(
            np.int32
        ).astype(np.int64)
    return accumulator.astype(np.int32)


def _sincos_fixed(phase: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    quadrant = (phase >> np.uint32(30)) & np.uint32(3)
    mirrored = ((phase >> np.uint32(29)) & np.uint32(1)).astype(bool)
    raw_x = (phase & np.uint32(0x1FFF_FFFF)) << np.uint32(3)
    reflected = np.where(
        raw_x == 0,
        np.uint32(0xFFFF_FFFF),
        np.uint32(0) - raw_x,
    ).astype(np.uint32)
    x = np.where(mirrored, reflected, raw_x).astype(np.uint32)
    sin_fixed = _fixed_horner(SIN_Q30, x)
    cos_fixed = _fixed_horner(COS_Q30, x)
    sin_local_fixed = np.where(mirrored, cos_fixed, sin_fixed)
    cos_local_fixed = np.where(mirrored, sin_fixed, cos_fixed)
    sin_local = sin_local_fixed.astype(np.float32) * FIXED_SCALE_INV
    cos_local = cos_local_fixed.astype(np.float32) * FIXED_SCALE_INV
    sine = np.select(
        (quadrant == 0, quadrant == 1, quadrant == 2),
        (sin_local, cos_local, -sin_local),
        default=-cos_local,
    ).astype(np.float32)
    cosine = np.select(
        (quadrant == 0, quadrant == 1, quadrant == 2),
        (cos_local, -sin_local, -cos_local),
        default=sin_local,
    ).astype(np.float32)
    return sine, cosine


def canonical_nco_interleaved_table(
    context_length: int,
    head_dim: int = MUSE_HEAD_DIM,
    frequency_base: float = MUSE_FREQUENCY_BASE,
) -> np.ndarray:
    """Return canonical little-endian f32 ``cos,sin`` pairs."""
    if context_length <= 0:
        raise ValueError("strict RoPE NCO requires positive context length")
    if head_dim != MUSE_HEAD_DIM:
        raise ValueError("strict RoPE NCO requires Muse head_dim=128")
    if np.float32(frequency_base).view(np.uint32) != np.float32(
        MUSE_FREQUENCY_BASE
    ).view(np.uint32):
        raise ValueError("strict RoPE NCO requires frequency_base=500000")

    theta = np.arange(context_length, dtype=np.float32)
    table = np.empty((context_length, head_dim), dtype=np.dtype("<f4"))
    for pair in range(head_dim // 2):
        rounded = np.floor(theta.astype(np.float64) * PHASE_PER_RADIAN + 0.5)
        phase = rounded.astype(np.uint64).astype(np.uint32)
        sine, cosine = _sincos_fixed(phase)
        table[:, pair * 2] = cosine
        table[:, pair * 2 + 1] = sine
        theta = np.multiply(theta, MUSE_THETA_SCALE, dtype=np.float32)
    return table


def canonical_nco_neox_table(context_length: int) -> np.ndarray:
    """Return canonical f32 rows as ``cos[64],sin[64]`` for Triton."""
    pairs = canonical_nco_interleaved_table(context_length).reshape(
        context_length, MUSE_HEAD_DIM // 2, 2
    )
    return np.concatenate((pairs[:, :, 0], pairs[:, :, 1]), axis=1)


try:
    import torch
    import triton
    import triton.language as tl
except ImportError:  # Host-side receipt/unit tools do not require CUDA/Triton.
    torch = None
    triton = None
    tl = None


_DEVICE_TABLES: dict[tuple[str, int | None], Any] = {}


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
    def _exact_rope_neox_kernel(
        input_ptr,
        cache_ptr,
        positions_ptr,
        output_ptr,
        pair_count: tl.constexpr,
        heads: tl.constexpr,
        head_dim: tl.constexpr,
    ):
        pair_index = tl.program_id(0) * 256 + tl.arange(0, 256)
        mask = pair_index < pair_count
        pairs_per_head: tl.constexpr = head_dim // 2
        pairs_per_token: tl.constexpr = heads * pairs_per_head
        token = pair_index // pairs_per_token
        within_token = pair_index % pairs_per_token
        head = within_token // pairs_per_head
        pair = within_token % pairs_per_head
        base = token * heads * head_dim + head * head_dim
        position = tl.load(positions_ptr + token, mask=mask, other=0).to(tl.int64)
        cache_base = position * head_dim
        cosine = tl.load(cache_ptr + cache_base + pair, mask=mask).to(tl.float32)
        sine = tl.load(
            cache_ptr + cache_base + pairs_per_head + pair, mask=mask
        ).to(tl.float32)
        x0 = tl.load(input_ptr + base + pair, mask=mask).to(tl.float32)
        x1 = tl.load(input_ptr + base + pairs_per_head + pair, mask=mask).to(
            tl.float32
        )
        tl.store(
            output_ptr + base + pair,
            _fma_rn(-x1, sine, _mul_rn(x0, cosine)),
            mask=mask,
        )
        tl.store(
            output_ptr + base + pairs_per_head + pair,
            _fma_rn(x0, sine, _mul_rn(x1, cosine)),
            mask=mask,
        )


    @torch.library.triton_op("muser::exact_rope_neox", mutates_args={})
    def exact_rope_neox(
        values: torch.Tensor,
        canonical_cache: torch.Tensor,
        positions: torch.Tensor,
        head_dim: int,
    ) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] % head_dim:
            raise ValueError("exact RoPE values must be packed token rows")
        if values.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("exact RoPE values must use model dtype")
        if head_dim != MUSE_HEAD_DIM:
            raise ValueError("exact RoPE requires Muse head_dim=128")
        if positions.ndim != 1 or positions.numel() != values.shape[0]:
            raise ValueError("exact RoPE positions differ from the packed rows")
        if (
            canonical_cache.ndim != 2
            or canonical_cache.shape[1] != head_dim
            or canonical_cache.dtype != torch.float32
        ):
            raise ValueError("canonical RoPE cache geometry or dtype is invalid")
        heads = values.shape[1] // head_dim
        pair_count = values.shape[0] * heads * (head_dim // 2)
        output = torch.empty_like(values)
        torch.library.wrap_triton(_exact_rope_neox_kernel)[
            (triton.cdiv(pair_count, 256),)
        ](
            values,
            canonical_cache,
            positions,
            output,
            pair_count=pair_count,
            heads=heads,
            head_dim=head_dim,
            num_warps=4,
        )
        return output


    def _canonical_device_table(device: torch.device, rows: int) -> torch.Tensor:
        key = (device.type, device.index)
        cached = _DEVICE_TABLES.get(key)
        if cached is None or cached.shape[0] < rows:
            cached = torch.from_numpy(canonical_nco_neox_table(rows)).to(device)
            _DEVICE_TABLES[key] = cached
        return cached


    def exact_rope(
        query: torch.Tensor,
        key: torch.Tensor,
        source_cache: torch.Tensor,
        positions: torch.Tensor,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = query.shape[0]
        if source_cache.ndim != 2 or tuple(source_cache.shape[1:]) != (head_dim,):
            raise ValueError("loaded vLLM RoPE cache geometry is invalid")
        expected = torch.arange(rows, device=positions.device, dtype=positions.dtype)
        if not torch.equal(positions, expected):
            raise ValueError("exact RoPE requires contiguous initial-prefill positions")
        canonical_cache = _canonical_device_table(query.device, rows)
        return (
            exact_rope_neox(query, canonical_cache, positions, head_dim),
            exact_rope_neox(key, canonical_cache, positions, head_dim),
        )

else:

    def exact_rope_neox(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("exact producer RoPE requires Torch and Triton")

    def exact_rope(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("exact producer RoPE requires Torch and Triton")


def metadata() -> dict[str, Any]:
    if torch is None or triton is None:
        raise RuntimeError("exact producer RoPE metadata requires Torch and Triton")
    return {
        "schema": SCHEMA,
        "implementation": "triton-neox-canonical-q30-nco",
        "coefficient_contract": "integer-q30-horner-iterative-f32-theta",
        "arithmetic_contract": "mul-rn-cos-then-fma-rn-sin",
        "output_boundary": "model-dtype",
        "torch": torch.__version__,
        "triton": triton.__version__,
    }
