"""Cross-vendor Muse SwiGLU arithmetic shared with muser-engine Metal."""

from __future__ import annotations

from typing import Any


SCHEMA = "muser.spark-exact-swiglu.v1"


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


    @triton.jit
    def _neon_expf(value):
        """Scalarized ggml_v_expf, including its overflow path."""
        value = value.to(tl.float32)
        rounding = 12582912.0
        z = _fma_rn(value, 1.4426950216293335, rounding)
        n = _add_rn(z, -rounding)
        reduced = _fma_rn(-n, 0.693145751953125, value)
        reduced = _fma_rn(-n, 1.428606765330187e-6, reduced)
        exponent = z.to(tl.uint32, bitcast=True) << 23
        scale = (exponent + 0x3F800000).to(tl.float32, bitcast=True)
        squared = _mul_rn(reduced, reduced)
        t1 = _fma_rn(0.008247390389442444, reduced, 0.04189976677298546)
        t2 = _fma_rn(0.16668395698070526, reduced, 0.4999912679195404)
        t2 = _fma_rn(t1, squared, t2)
        polynomial = _fma_rn(
            t2, squared, _mul_rn(0.9999994039535522, reduced)
        )
        normal = _fma_rn(polynomial, scale, scale)

        delta = tl.where(n <= 0.0, 0x82000000, 0).to(tl.uint32)
        scale1 = (delta + 0x7F000000).to(tl.float32, bitcast=True)
        scale2 = (exponent - delta).to(tl.float32, bitcast=True)
        overflow = _mul_rn(_fma_rn(scale2, polynomial, scale2), scale1)
        extreme = _mul_rn(scale1, scale1)
        magnitude = tl.abs(n)
        return tl.where(
            magnitude > 192.0,
            extreme,
            tl.where(magnitude > 126.0, overflow, normal),
        )


    @triton.jit
    def _exact_swiglu_kernel(
        packed_ptr,
        output_ptr,
        count: tl.constexpr,
        width: tl.constexpr,
    ):
        index = tl.program_id(0) * 256 + tl.arange(0, 256)
        mask = index < count
        row = index // width
        column = index % width
        packed_base = row * width * 2
        gate = tl.load(packed_ptr + packed_base + column, mask=mask).to(tl.float32)
        up = tl.load(packed_ptr + packed_base + width + column, mask=mask).to(
            tl.float32
        )
        denominator = _add_rn(1.0, _neon_expf(-gate))
        silu = _div_rn(gate, denominator)
        tl.store(output_ptr + index, _mul_rn(silu, up), mask=mask)


    @torch.library.triton_op("muser::exact_swiglu", mutates_args={})
    def exact_swiglu(packed_gate_up: torch.Tensor) -> torch.Tensor:
        if packed_gate_up.ndim < 1 or packed_gate_up.shape[-1] % 2:
            raise ValueError("exact SwiGLU requires packed equal gate/up halves")
        if packed_gate_up.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("exact SwiGLU input must use the model dtype")
        count = packed_gate_up.numel() // 2
        output = torch.empty(
            (*packed_gate_up.shape[:-1], packed_gate_up.shape[-1] // 2),
            dtype=packed_gate_up.dtype,
            device=packed_gate_up.device,
        )
        torch.library.wrap_triton(_exact_swiglu_kernel)[(triton.cdiv(count, 256),)](
            packed_gate_up,
            output,
            count=count,
            width=packed_gate_up.shape[-1] // 2,
            num_warps=4,
        )
        return output

else:

    def exact_swiglu(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("exact producer SwiGLU requires Torch and Triton")


def vllm_exact_mlp_forward(module: Any, values: Any) -> Any:
    gate_up, _ = module.gate_up_proj(values)
    activated = exact_swiglu(gate_up)
    output, _ = module.down_proj(activated)
    return output


def install_exact_swiglu() -> dict[str, Any]:
    """Patch Muse MLP activation before model construction and fail closed."""
    import torch
    import triton
    from vllm.model_executor.models import muse_glimmer

    muse_glimmer.MuseGlimmerMLP.forward = vllm_exact_mlp_forward
    if muse_glimmer.MuseGlimmerMLP.forward is not vllm_exact_mlp_forward:
        raise RuntimeError("failed to install exact Muse SwiGLU")
    return {
        "schema": SCHEMA,
        "implementation": "triton-scalarized-ggml-neon-exp",
        "patched_target": "MuseGlimmerMLP.forward",
        "arithmetic_contract": "explicit-f32-rn-f16-output-boundary",
        "torch": torch.__version__,
        "triton": triton.__version__,
    }
