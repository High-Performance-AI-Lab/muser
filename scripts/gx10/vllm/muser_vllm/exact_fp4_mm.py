"""Order-free integer contraction for cross-vendor NVFP4 GEMMs."""

from __future__ import annotations

from typing import Any


SCHEMA = "muser.spark-exact-fp4-mm.v3"
E4M3_Q9_SCALE = 512
E2M1_Q1_SCALE = 2
CONTRACTION_SCALE_INV = 1.0 / float(E4M3_Q9_SCALE**2 * E2M1_Q1_SCALE**2)
A16_Q8_CONTRACTION_SCALE_INV = 1.0 / float(E4M3_Q9_SCALE * E2M1_Q1_SCALE)
_ORIGINAL_APPLY_WEIGHTS: Any = None
_ORIGINAL_A16_APPLY_WEIGHTS: Any = None
_ORIGINAL_A16_PROCESS_WEIGHTS: Any = None
_ORIGINAL_CT_PROCESS_WEIGHTS: Any = None


try:
    import torch
    import triton
    import triton.language as tl
except ImportError:  # Host-side receipt/unit tools do not require CUDA/Triton.
    torch = None
    triton = None
    tl = None


if triton is not None:
    from .exact_attention import _div_rn, _fma_rn, _mul_rn

    @triton.jit
    def _e2m1_q1(code):
        magnitude_code = code & 7
        magnitude = tl.where(
            magnitude_code <= 4,
            magnitude_code,
            tl.where(magnitude_code == 5, 6, tl.where(magnitude_code == 6, 8, 12)),
        ).to(tl.int32)
        return tl.where((code & 8) != 0, -magnitude, magnitude)

    @triton.jit
    def _e4m3_q9(code):
        """Decode finite E4M3FN into exact signed units of 2^-9."""
        magnitude_code = code & 0x7F
        exponent = (magnitude_code >> 3) & 0x0F
        mantissa = magnitude_code & 7
        exponent_i32 = exponent.to(tl.int32)
        mantissa_i32 = mantissa.to(tl.int32)
        magnitude = tl.where(
            exponent == 0,
            mantissa_i32,
            (8 + mantissa_i32) << tl.maximum(exponent_i32 - 1, 0),
        ).to(tl.int32)
        return tl.where((code & 0x80) != 0, -magnitude, magnitude)

    @triton.jit
    def _swizzled_scale_offset(row, group, scale_k_tiles):
        return (
            (row // 128) * scale_k_tiles * 32 * 4 * 4
            + (group // 4) * 32 * 4 * 4
            + (row % 32) * 4 * 4
            + ((row % 128) // 32) * 4
            + group % 4
        )

    @triton.jit
    def _integer_fp4_mm_kernel(
        activation_ptr,
        weight_ptr,
        activation_scale_ptr,
        weight_scale_ptr,
        output_ptr,
        weight_scale2_ptr,
        input_scale_ptr,
        output_size,
        groups,
        packed_per_row,
        group_chunks,
        activation_scale_k_tiles,
        weight_scale_k_tiles,
        contraction_scale_inv: tl.constexpr,
    ):
        cell = tl.program_id(0)
        token = cell // output_size
        row = cell - token * output_size
        lane = tl.arange(0, 32)
        partial = tl.zeros((32,), tl.int64)

        for chunk in range(0, group_chunks):
            group = chunk * 32 + lane
            active = group < groups
            block_dot = tl.zeros((32,), tl.int32)
            activation_base = token * packed_per_row + group * 8
            weight_base = row * packed_per_row + group * 8
            for pair in tl.static_range(0, 8):
                activation_byte = tl.load(
                    activation_ptr + activation_base + pair, mask=active, other=0
                ).to(tl.uint8)
                weight_byte = tl.load(
                    weight_ptr + weight_base + pair, mask=active, other=0
                ).to(tl.uint8)
                block_dot += _e2m1_q1(weight_byte & 15) * _e2m1_q1(
                    activation_byte & 15
                )
                block_dot += _e2m1_q1(weight_byte >> 4) * _e2m1_q1(
                    activation_byte >> 4
                )

            activation_scale_offset = _swizzled_scale_offset(
                token, group, activation_scale_k_tiles
            )
            weight_scale_offset = _swizzled_scale_offset(
                row, group, weight_scale_k_tiles
            )
            activation_scale_code = tl.load(
                activation_scale_ptr + activation_scale_offset,
                mask=active,
                other=0,
            ).to(tl.uint8)
            weight_scale_code = tl.load(
                weight_scale_ptr + weight_scale_offset,
                mask=active,
                other=0,
            ).to(tl.uint8)
            contribution = (
                block_dot.to(tl.int64)
                * _e4m3_q9(weight_scale_code).to(tl.int64)
                * _e4m3_q9(activation_scale_code).to(tl.int64)
            )
            partial += tl.where(active, contribution, 0)

        # Integer addition is associative while this sum stays in range. Muse's
        # widest contraction is below 2^58, so vendor reduction topology is
        # irrelevant and no overflow is possible.
        total = tl.sum(partial, axis=0)
        scaled = _mul_rn(total.to(tl.float32), contraction_scale_inv)
        scaled = _mul_rn(scaled, tl.load(weight_scale2_ptr).to(tl.float32))
        scaled = _mul_rn(scaled, tl.load(input_scale_ptr).to(tl.float32))
        tl.store(output_ptr + cell, scaled)

    @triton.jit
    def _quantize_q8k_kernel(
        input_ptr,
        quant_ptr,
        scale_ptr,
        blocks_per_token: tl.constexpr,
    ):
        cell = tl.program_id(0)
        token = cell // blocks_per_token
        block = cell - token * blocks_per_token
        offsets = tl.arange(0, 256)
        base = token * blocks_per_token * 256 + block * 256
        values = tl.load(input_ptr + base + offsets).to(tl.float32)
        magnitudes = tl.abs(values)
        abs_max = tl.max(magnitudes, axis=0)
        # The reference scans left-to-right with a strict greater-than test.
        # Select the first index explicitly so equal-magnitude opposite-sign
        # maxima cannot inherit a backend-specific reduction tie break.
        first_index = tl.min(
            tl.where(magnitudes == abs_max, offsets, 256), axis=0
        )
        signed_max = tl.load(input_ptr + base + first_index).to(tl.float32)
        nonzero = abs_max != 0.0
        inverse_scale = tl.where(nonzero, _div_rn(-127.0, signed_max), 0.0)
        rounded_bits = _fma_rn(inverse_scale, values, 12582912.0).to(
            tl.int32, bitcast=True
        )
        rounded = (rounded_bits & 0x007FFFFF) - 0x00400000
        quant = tl.where(nonzero, tl.minimum(127, rounded), 0).to(tl.int8)
        tl.store(quant_ptr + base + offsets, quant)
        scale = tl.where(nonzero, _div_rn(1.0, inverse_scale), 0.0)
        tl.store(scale_ptr + cell, scale)

    @triton.jit
    def _integer_fp4_a16_from_q8_kernel(
        quant_ptr,
        activation_scale_ptr,
        weight_ptr,
        weight_scale_ptr,
        weight_scale2_ptr,
        output_ptr,
        output_size: tl.constexpr,
        token_count: tl.constexpr,
        groups: tl.constexpr,
        packed_per_row: tl.constexpr,
        blocks_per_token: tl.constexpr,
        contraction_scale_inv: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        token = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        row = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        # Triton's INT8 tensor dot requires K >= 32. The artifact scale changes
        # every 16 values, so pad each independent K=16 group with exact zeros
        # instead of combining groups across that semantic boundary.
        element = tl.arange(0, 32)
        output_mask = (token[:, None] < token_count) & (row[None, :] < output_size)
        total = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for block in range(0, blocks_per_token):
            integer_total = tl.zeros((BLOCK_M, BLOCK_N), tl.int64)
            for group_in_block in tl.static_range(0, 16):
                group = block * 16 + group_in_block
                activation_offset = (
                    token[:, None] * blocks_per_token * 256
                    + block * 256
                    + group_in_block * 16
                    + element[None, :]
                )
                activation = tl.load(
                    quant_ptr + activation_offset,
                    mask=(token[:, None] < token_count) & (element[None, :] < 16),
                    other=0,
                ).to(tl.int8)
                weight_offset = (
                    row[:, None] * packed_per_row
                    + group * 8
                    + (element[None, :] // 2)
                )
                packed_weight = tl.load(
                    weight_ptr + weight_offset,
                    mask=(row[:, None] < output_size) & (element[None, :] < 16),
                    other=0,
                ).to(tl.uint8)
                weight_code = (
                    packed_weight >> ((element[None, :] & 1) * 4)
                ) & 15
                weight = _e2m1_q1(weight_code).to(tl.int8)
                group_dot = tl.dot(
                    activation,
                    tl.trans(weight),
                    out_dtype=tl.int32,
                )
                weight_scale_code = tl.load(
                    weight_scale_ptr + row * groups + group,
                    mask=row < output_size,
                    other=0,
                ).to(tl.uint8)
                integer_total += group_dot.to(tl.int64) * _e4m3_q9(
                    weight_scale_code
                )[None, :].to(tl.int64)
            contribution = _fma_rn(
                integer_total.to(tl.float32), contraction_scale_inv, 0.0
            )
            contribution = _fma_rn(
                contribution,
                tl.load(
                    activation_scale_ptr + token * blocks_per_token + block,
                    mask=token < token_count,
                    other=0.0,
                )[:, None],
                0.0,
            )
            contribution = _fma_rn(
                contribution, tl.load(weight_scale2_ptr).to(tl.float32), 0.0
            )
            total = _fma_rn(1.0, contribution, total)
        tl.store(
            output_ptr + token[:, None] * output_size + row[None, :],
            total,
            mask=output_mask,
        )

    @torch.library.triton_op("muser::integer_fp4_mm", mutates_args={})
    def integer_fp4_mm(
        activation: torch.Tensor,
        weight: torch.Tensor,
        activation_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_scale2: torch.Tensor,
        input_scale: torch.Tensor,
    ) -> torch.Tensor:
        if activation.ndim != 2 or weight.ndim != 2:
            raise ValueError("exact FP4 contraction requires matrix inputs")
        if activation.dtype != torch.uint8 or weight.dtype != torch.uint8:
            raise TypeError("exact FP4 contraction requires packed uint8 operands")
        if activation.shape[1] != weight.shape[1]:
            raise ValueError("exact FP4 contraction dimensions disagree")
        if not activation.is_contiguous() or not weight.is_contiguous():
            raise ValueError("exact FP4 contraction requires contiguous packed matrices")
        if activation_scale.dtype != torch.float8_e4m3fn:
            raise TypeError("activation block scales must be E4M3FN")
        if weight_scale.dtype != torch.float8_e4m3fn:
            raise TypeError("weight block scales must be E4M3FN")
        if weight_scale2.dtype != torch.float32 or weight_scale2.numel() != 1:
            raise TypeError("weight global scale must be one f32 scalar")
        if input_scale.dtype != torch.float32 or input_scale.numel() != 1:
            raise TypeError("input global scale must be one f32 scalar")
        if activation_scale.shape[1] % 4 or weight_scale.shape[1] % 4:
            raise ValueError("exact FP4 contraction scale geometry is invalid")

        groups = activation.shape[1] * 2 // 16
        output = torch.empty(
            (activation.shape[0], weight.shape[0]),
            dtype=torch.float16,
            device=activation.device,
        )
        torch.library.wrap_triton(_integer_fp4_mm_kernel)[(output.numel(),)](
            activation,
            weight,
            activation_scale.view(torch.uint8),
            weight_scale.view(torch.uint8),
            output,
            weight_scale2,
            input_scale,
            output_size=output.shape[1],
            groups=groups,
            packed_per_row=activation.shape[1],
            group_chunks=triton.cdiv(groups, 32),
            activation_scale_k_tiles=activation_scale.shape[1] // 4,
            weight_scale_k_tiles=weight_scale.shape[1] // 4,
            contraction_scale_inv=CONTRACTION_SCALE_INV,
            num_warps=1,
        )
        return output

    @torch.library.triton_op("muser::quantize_nvfp4_q8k", mutates_args={})
    def quantize_nvfp4_q8k(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if input.ndim != 2 or input.dtype != torch.float16:
            raise TypeError("weight-only exact NVFP4 requires a 2D F16 activation")
        if not input.is_contiguous() or input.shape[1] % 256:
            raise ValueError("weight-only exact NVFP4 requires contiguous Q8_K blocks")
        blocks_per_token = input.shape[1] // 256
        quant = torch.empty_like(input, dtype=torch.int8)
        scales = torch.empty(
            (input.shape[0], blocks_per_token), dtype=torch.float32, device=input.device
        )
        torch.library.wrap_triton(_quantize_q8k_kernel)[
            (input.shape[0] * blocks_per_token,)
        ](
            input,
            quant,
            scales,
            blocks_per_token=blocks_per_token,
            num_warps=8,
        )
        return quant, scales

    @torch.library.triton_op("muser::integer_fp4_a16_from_q8", mutates_args={})
    def integer_fp4_a16_from_q8(
        quant: torch.Tensor,
        activation_scales: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_scale2: torch.Tensor,
    ) -> torch.Tensor:
        if quant.ndim != 2 or quant.dtype != torch.int8 or not quant.is_contiguous():
            raise TypeError("exact A16 contraction requires contiguous 2D Q8 integers")
        if quant.shape[1] % 256:
            raise ValueError("exact A16 contraction requires complete Q8_K blocks")
        if weight.ndim != 2 or weight.dtype != torch.uint8 or not weight.is_contiguous():
            raise TypeError("exact A16 contraction requires contiguous packed uint8 weights")
        if weight.shape[1] * 2 != quant.shape[1]:
            raise ValueError("exact A16 contraction dimensions disagree")
        groups = quant.shape[1] // 16
        blocks_per_token = quant.shape[1] // 256
        if activation_scales.shape != (quant.shape[0], blocks_per_token):
            raise ValueError("Q8 activation scale geometry is invalid")
        if activation_scales.dtype != torch.float32:
            raise TypeError("Q8 activation scales must be f32")
        if weight_scale.shape != (weight.shape[0], groups):
            raise ValueError("NVFP4 weight scale geometry is invalid")
        if weight_scale.dtype != torch.float8_e4m3fn:
            raise TypeError("NVFP4 block scales must be E4M3FN")
        if weight_scale2.dtype != torch.float32 or weight_scale2.numel() != 1:
            raise TypeError("NVFP4 global weight scale must be one f32 scalar")
        output = torch.empty(
            (quant.shape[0], weight.shape[0]), dtype=torch.float16, device=quant.device
        )
        block_m = 16
        block_n = 16
        torch.library.wrap_triton(_integer_fp4_a16_from_q8_kernel)[
            (triton.cdiv(output.shape[0], block_m), triton.cdiv(output.shape[1], block_n))
        ](
            quant,
            activation_scales,
            weight,
            weight_scale.view(torch.uint8),
            weight_scale2,
            output,
            output_size=output.shape[1],
            token_count=output.shape[0],
            groups=groups,
            packed_per_row=weight.shape[1],
            blocks_per_token=blocks_per_token,
            contraction_scale_inv=A16_Q8_CONTRACTION_SCALE_INV,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=4,
        )
        return output

else:

    def integer_fp4_mm(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("exact FP4 contraction requires Torch and Triton")

    def quantize_nvfp4_q8k(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("exact Q8 activation quantization requires Torch and Triton")

    def integer_fp4_a16_from_q8(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("exact A16 contraction requires Torch and Triton")


def exact_fp4_mm(
    activation: Any,
    weight: Any,
    activation_scale: Any,
    weight_scale: Any,
    alpha: Any,
    weight_scale2: Any,
    input_scale: Any,
    diagnostics: dict[str, Any] | None = None,
) -> Any:
    """Execute the order-free integer NVFP4 contraction.

    ``alpha`` remains in the API because vLLM owns that tensor, but the exact
    path deliberately applies the two original global scales separately.
    """
    if alpha.dtype != torch.float32 or alpha.numel() != 1:
        raise TypeError("NVFP4 alpha must be one f32 scalar")
    if diagnostics is not None:
        diagnostics["backend"] = "integer-q1-q9-i64"
        diagnostics["output_cells"] = activation.shape[0] * weight.shape[0]
    return integer_fp4_mm(
        activation,
        weight,
        activation_scale,
        weight_scale,
        weight_scale2,
        input_scale,
    )


def exact_fp4_a16_mm(
    activation: Any,
    weight: Any,
    weight_scale: Any,
    weight_scale2: Any,
    diagnostics: dict[str, Any] | None = None,
) -> Any:
    """Execute the selected weight-only artifact's Q8 × NVFP4 contraction."""
    quant, activation_scales = quantize_nvfp4_q8k(activation)
    if diagnostics is not None:
        diagnostics["backend"] = "a16-q8k-e2m1-q1-e4m3-q9-i64"
        diagnostics["activation_q8_cells"] = activation.numel()
        diagnostics["output_cells"] = activation.shape[0] * weight.shape[0]
    return integer_fp4_a16_from_q8(
        quant, activation_scales, weight, weight_scale, weight_scale2
    )


def _exact_integer_apply_weights(
    self: Any,
    layer: Any,
    x: Any,
    bias: Any = None,
) -> Any:
    from vllm.model_executor.layers.fusion.quant_activation import (
        as_quantized_activation,
    )
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        pad_nvfp4_activation_for_cutlass,
        slice_nvfp4_output,
    )
    from vllm.model_executor.kernels.linear.nvfp4 import flashinfer

    output_size = layer.output_size_per_partition
    weights_padding_bytes = getattr(layer, "weights_padding_cols", 0)
    qa = as_quantized_activation(x, self.input_quant_key())
    if qa is not None:
        x_fp4, x_blockscale = qa.data, qa.scale
        x_fp4 = pad_nvfp4_activation_for_cutlass(x_fp4, weights_padding_bytes)
        output_dtype = qa.orig_dtype
        output_shape = [*qa.orig_shape[:-1], output_size]
    else:
        if not isinstance(x, torch.Tensor):
            raise TypeError("exact NVFP4 projection requires a tensor input")
        output_dtype = x.dtype
        output_shape = [*x.shape[:-1], output_size]
        x_fp4, x_blockscale = flashinfer.scaled_fp4_quant(
            x,
            layer.input_global_scale_inv,
            is_sf_swizzled_layout=True,
            backend="flashinfer-cutlass",
            padded_n=x.shape[-1] + weights_padding_bytes * 2,
        )
    if output_dtype != torch.float16:
        raise TypeError("exact NVFP4 projection requires FP16 activations")

    out = exact_fp4_mm(
        x_fp4,
        layer.weight,
        x_blockscale,
        layer.weight_scale,
        layer.alpha,
        layer.weight_global_scale,
        layer.input_global_scale,
    )
    out = slice_nvfp4_output(out, output_size)
    if bias is not None:
        out = out + bias
    return out.view(*output_shape)


def _exact_integer_a16_process_weights(self: Any, layer: Any) -> None:
    # The exact kernel consumes the checkpoint's canonical row-major E2M1 and
    # E4M3 tensors directly. Marlin repacking would destroy that address map.
    layer.muser_exact_a16_q8 = True


def _exact_integer_a16_apply_weights(
    self: Any,
    layer: Any,
    x: Any,
    bias: Any = None,
) -> Any:
    if not isinstance(x, torch.Tensor) or x.dtype != torch.float16:
        raise TypeError("exact weight-only NVFP4 projection requires F16 activations")
    output_shape = [*x.shape[:-1], layer.output_size_per_partition]
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    quant, activation_scales = quantize_nvfp4_q8k(x_2d)
    logical_widths = list(getattr(layer, "logical_widths", [layer.weight.shape[0]]))
    global_scales = layer.weight_global_scale.reshape(-1)
    if global_scales.numel() not in (1, len(logical_widths)):
        raise ValueError("weight-only NVFP4 global scale geometry is invalid")
    outputs = []
    row_offset = 0
    for index, width in enumerate(logical_widths):
        scale_index = 0 if global_scales.numel() == 1 else index
        outputs.append(
            integer_fp4_a16_from_q8(
                quant,
                activation_scales,
                layer.weight[row_offset : row_offset + width].contiguous(),
                layer.weight_scale[row_offset : row_offset + width].contiguous(),
                global_scales[scale_index : scale_index + 1].contiguous(),
            )
        )
        row_offset += width
    if row_offset != layer.weight.shape[0]:
        raise ValueError("weight-only NVFP4 logical widths do not cover the matrix")
    output = torch.cat(outputs, dim=-1) if len(outputs) > 1 else outputs[0]
    if bias is not None:
        output = output + bias
    return output.view(*output_shape)


def _exact_ct_process_weights(self: Any, layer: Any) -> None:
    if not self.use_a16:
        if _ORIGINAL_CT_PROCESS_WEIGHTS is None:
            raise RuntimeError("stock compressed-tensors NVFP4 loader was not retained")
        return _ORIGINAL_CT_PROCESS_WEIGHTS(self, layer)
    # Keep every fused projection's artifact scale. Stock vLLM takes max() and
    # warns about reduced accuracy; that changes Q/K/V bytes and is forbidden
    # on the exact seam lane.
    from torch.nn.parameter import Parameter

    layer.weight = layer.weight_packed
    del layer.weight_packed
    layer.weight_global_scale = Parameter(
        torch.reciprocal(layer.weight_global_scale.to(torch.float32)),
        requires_grad=False,
    )
    self.kernel.process_weights_after_loading(layer)


def install_exact_fp4_mm() -> dict[str, Any]:
    """Replace stock W4A4 and weight-only NVFP4 contractions with exact lanes."""
    global _ORIGINAL_APPLY_WEIGHTS, _ORIGINAL_A16_APPLY_WEIGHTS
    global _ORIGINAL_A16_PROCESS_WEIGHTS, _ORIGINAL_CT_PROCESS_WEIGHTS
    from vllm.model_executor.kernels.linear.nvfp4 import flashinfer
    from vllm.model_executor.kernels.linear.nvfp4.marlin import MarlinNvFp4LinearKernel
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4 import (
        CompressedTensorsW4A4Fp4,
    )

    kernel = flashinfer.FlashInferCutlassNvFp4LinearKernel
    if kernel.apply_weights is not _exact_integer_apply_weights:
        _ORIGINAL_APPLY_WEIGHTS = kernel.apply_weights
        kernel.apply_weights = _exact_integer_apply_weights
    if _ORIGINAL_APPLY_WEIGHTS is None:
        raise RuntimeError("failed to retain the stock NVFP4 linear implementation")
    if MarlinNvFp4LinearKernel.apply_weights is not _exact_integer_a16_apply_weights:
        _ORIGINAL_A16_APPLY_WEIGHTS = MarlinNvFp4LinearKernel.apply_weights
        _ORIGINAL_A16_PROCESS_WEIGHTS = MarlinNvFp4LinearKernel.process_weights_after_loading
        MarlinNvFp4LinearKernel.process_weights_after_loading = (
            _exact_integer_a16_process_weights
        )
        MarlinNvFp4LinearKernel.apply_weights = _exact_integer_a16_apply_weights
    if _ORIGINAL_A16_APPLY_WEIGHTS is None or _ORIGINAL_A16_PROCESS_WEIGHTS is None:
        raise RuntimeError("failed to retain the stock W4A16 Marlin implementation")
    if CompressedTensorsW4A4Fp4.process_weights_after_loading is not _exact_ct_process_weights:
        _ORIGINAL_CT_PROCESS_WEIGHTS = CompressedTensorsW4A4Fp4.process_weights_after_loading
        CompressedTensorsW4A4Fp4.process_weights_after_loading = _exact_ct_process_weights
    if _ORIGINAL_CT_PROCESS_WEIGHTS is None:
        raise RuntimeError("failed to retain the stock compressed-tensors NVFP4 loader")
    return metadata() | {
        "patched_targets": [
            "FlashInferCutlassNvFp4LinearKernel.apply_weights",
            "MarlinNvFp4LinearKernel.process_weights_after_loading",
            "MarlinNvFp4LinearKernel.apply_weights",
            "CompressedTensorsW4A4Fp4.process_weights_after_loading",
        ],
        "selection": "w4a4-and-weight-only-nvfp4-linears",
        "stock_cutlass_and_marlin": "disabled-on-exact-lane",
    }


def metadata() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "implementation": "w4a4-q1-q9-and-a16-q8k-q1-q9-i64",
        "e2m1_integer_scale": E2M1_Q1_SCALE,
        "e4m3_integer_scale": E4M3_Q9_SCALE,
        "accumulator": "signed-i64-order-free",
        "contraction_scale_inv": CONTRACTION_SCALE_INV,
        "a16_contraction_scale_inv": A16_Q8_CONTRACTION_SCALE_INV,
        "activation_contract": "q8k-256-first-signed-max-magic-nearest-even",
        "global_scaling": "w4a4-fixed-power-of-two-then-weight-then-input-f32",
        "a16_global_scaling": "integer-then-q8-scale-then-weight-scale-f32",
        "fused_scale_contract": "per-logical-projection-preserved",
        "output_boundary": "f16",
    }
