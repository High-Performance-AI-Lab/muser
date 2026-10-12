"""Deterministic NVFP4 activation quantization for the Spark producer.

The stock vLLM kernel uses ``rcp.approx.ftz.f32`` while deriving dynamic
group scales.  That approximation is deliberately fast but cannot be
reproduced bit-for-bit on Metal.  This module keeps the FP4 CUTLASS GEMM and
replaces only its activation-quantization prepass with a single Triton kernel
whose arithmetic contract is shared with muser-engine.
"""

from __future__ import annotations

from typing import Any


SCHEMA = "muser.spark-exact-fp4-quant.v1"
_FIRST_USE = True


def _disable_fused_input_quantization(_self: Any) -> None:
    """Force the linear kernel to invoke this module's pinned quantizer.

    vLLM may otherwise pass a ``QuantizedActivation`` produced by a fusion
    pass.  That path calls the stock CUDA quantizer before ``apply_weights``
    and silently bypasses the cross-vendor arithmetic contract.
    """
    return None


def install_exact_fp4_quantizer() -> dict[str, Any]:
    """Install the pinned quantizer into vLLM's compressed-tensors scheme."""
    import torch

    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_w4a4_nvfp4 as scheme,
    )
    from vllm.model_executor.kernels.linear.nvfp4 import flashinfer

    scheme.scaled_fp4_quant = vllm_exact_scaled_fp4_quant
    flashinfer.scaled_fp4_quant = vllm_exact_scaled_fp4_quant
    flashinfer.FlashInferCutlassNvFp4LinearKernel.input_quant_key = (
        _disable_fused_input_quantization
    )
    if flashinfer.scaled_fp4_quant is not vllm_exact_scaled_fp4_quant:
        raise RuntimeError("failed to install exact quantizer on NVFP4 linear kernel")
    if (
        flashinfer.FlashInferCutlassNvFp4LinearKernel.input_quant_key
        is not _disable_fused_input_quantization
    ):
        raise RuntimeError("failed to disable fused NVFP4 input quantization")
    return {
        "schema": SCHEMA,
        "implementation": "triton-single-pass-group16",
        "patched_targets": [
            "compressed_tensors_w4a4_nvfp4.scaled_fp4_quant",
            "kernels.linear.nvfp4.flashinfer.scaled_fp4_quant",
            "FlashInferCutlassNvFp4LinearKernel.input_quant_key",
        ],
        "fused_input_quantization": "disabled",
        "scale_contract": "f16-max_mul-f32-one-sixth_mul-global_rne-e4m3fn",
        "code_contract": "compare_abs-f16_mul-global_against_e4m3fn-thresholds",
        "triton": __import__("triton").__version__,
        "torch": torch.__version__,
    }


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
    def _e2m1_magnitude_code(magnitude, scale):
        return tl.where(
            magnitude <= scale * 0.25,
            0,
            tl.where(
                magnitude < scale * 0.75,
                1,
                tl.where(
                    magnitude <= scale * 1.25,
                    2,
                    tl.where(
                        magnitude < scale * 1.75,
                        3,
                        tl.where(
                            magnitude <= scale * 2.5,
                            4,
                            tl.where(
                                magnitude < scale * 3.5,
                                5,
                                tl.where(magnitude <= scale * 5.0, 6, 7),
                            ),
                        ),
                    ),
                ),
            ),
        ).to(tl.uint8)


    @triton.jit
    def _exact_scaled_fp4_quant_kernel(
        input_ptr,
        global_scale_ptr,
        output_ptr,
        scale_ptr,
        groups_per_row: tl.constexpr,
        packed_per_row: tl.constexpr,
        scale_k_tiles: tl.constexpr,
    ):
        group_linear = tl.program_id(0)
        row = group_linear // groups_per_row
        group = group_linear - row * groups_per_row
        group_base = row * groups_per_row * 16 + group * 16

        values = tl.load(input_ptr + group_base + tl.arange(0, 16)).to(tl.float32)
        abs_max = tl.max(tl.abs(values), axis=0)
        global_scale = tl.load(global_scale_ptr).to(tl.float32)

        # 0x3e2aaaab: correctly-rounded f32 1/6.  Keeping each multiplication
        # as a standalone expression pins the two f32 rounding boundaries.
        normalized_max = abs_max * 0.16666667163372039794921875
        scale_value = tl.minimum(normalized_max * global_scale, 448.0)
        scale_fp8 = scale_value.to(tl.float8e4nv)
        scale_code = scale_fp8.to(tl.uint8, bitcast=True)
        decoded_scale = scale_fp8.to(tl.float32)

        pair = tl.arange(0, 8)
        even_half = tl.load(input_ptr + group_base + pair * 2)
        odd_half = tl.load(input_ptr + group_base + pair * 2 + 1)
        even = even_half.to(tl.float32)
        odd = odd_half.to(tl.float32)
        even_magnitude = tl.abs(even) * global_scale
        odd_magnitude = tl.abs(odd) * global_scale

        even_sign = (even_half.to(tl.uint16, bitcast=True) >> 15).to(tl.uint8) << 3
        odd_sign = (odd_half.to(tl.uint16, bitcast=True) >> 15).to(tl.uint8) << 3
        even_code = _e2m1_magnitude_code(even_magnitude, decoded_scale) | even_sign
        odd_code = _e2m1_magnitude_code(odd_magnitude, decoded_scale) | odd_sign
        packed = even_code | (odd_code << 4)
        tl.store(output_ptr + row * packed_per_row + group * 8 + pair, packed)

        # Tensor-core scale-factor B layout: [m/128, k/4, 32, 4, 4].
        m_tile = row // 128
        k_tile = group // 4
        outer_m = row % 32
        inner_m = (row % 128) // 32
        inner_k = group % 4
        scale_offset = (
            m_tile * scale_k_tiles * 32 * 4 * 4
            + k_tile * 32 * 4 * 4
            + outer_m * 4 * 4
            + inner_m * 4
            + inner_k
        )
        tl.store(scale_ptr + scale_offset, scale_code)


    @torch.library.triton_op("muser::exact_scaled_fp4_quant", mutates_args={})
    def exact_scaled_fp4_quant(
        input: torch.Tensor, input_global_scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input.ndim < 1:
            raise ValueError("NVFP4 activation input must have at least one dimension")
        matrix = input.reshape(-1, input.shape[-1])
        rows, columns = matrix.shape
        if columns % 16:
            raise ValueError("NVFP4 activation width must be divisible by 16")
        if matrix.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("NVFP4 activation input must be FP16 or BF16")
        groups = columns // 16
        rounded_rows = triton.cdiv(rows, 128) * 128
        rounded_groups = triton.cdiv(groups, 4) * 4
        packed = torch.empty((rows, columns // 2), dtype=torch.uint8, device=input.device)
        scale_bytes = torch.zeros(
            (rounded_rows, rounded_groups), dtype=torch.uint8, device=input.device
        )
        torch.library.wrap_triton(_exact_scaled_fp4_quant_kernel)[(rows * groups,)](
            matrix,
            input_global_scale,
            packed,
            scale_bytes,
            groups_per_row=groups,
            packed_per_row=columns // 2,
            scale_k_tiles=rounded_groups // 4,
            num_warps=1,
        )
        return packed, scale_bytes.view(torch.float8_e4m3fn)

else:

    def exact_scaled_fp4_quant(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("exact NVFP4 producer quantization requires Torch and Triton")


def vllm_exact_scaled_fp4_quant(
    input: Any,
    input_global_scale: Any,
    *,
    is_sf_swizzled_layout: bool = True,
    backend: str | None = None,
    padded_n: int | None = None,
) -> tuple[Any, Any]:
    """Drop-in replacement for vLLM's current NVFP4 quantizer API."""
    global _FIRST_USE
    if not is_sf_swizzled_layout:
        raise ValueError("muser exact NVFP4 quantizer requires swizzled scales")
    if backend not in (None, "flashinfer-cutlass", "flashinfer-cutedsl"):
        raise ValueError(f"unsupported exact NVFP4 backend: {backend}")
    width = input.shape[-1]
    if padded_n is not None:
        if padded_n < width or padded_n % 16:
            raise ValueError("padded NVFP4 activation width is invalid")
        if padded_n != width:
            input = torch.nn.functional.pad(input, (0, padded_n - width))
    if _FIRST_USE:
        print(
            "[muser-exact-fp4-quant] first-use "
            f"shape={tuple(input.shape)} backend={backend}",
            flush=True,
        )
        _FIRST_USE = False
    return exact_scaled_fp4_quant(input, input_global_scale)
