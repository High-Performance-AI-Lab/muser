//! Quantized dot/dequant kernels — muse dtypes only.
//!
//! Extraction source (PULL-AND-SIMPLIFY):
//! `crates/ferrite-inference/src/quant/{q4k,q5..,blocks,helpers,dispatch}.rs`.
//! Keep only Q4_K, Q5_K, Q6_K, Q8_0, Q4_0, F16, F32 — the dtypes the muse
//! GGUF actually uses. **Drop** the IQ2/IQ3/MLX/codebook/subspace quant zoo;
//! it is dead weight for a one-model, one-quant-family engine.

mod blocks;
mod dispatch;
mod helpers;
mod k_block;
mod nvfp4;

pub use blocks::*;
pub use dispatch::*;
pub use helpers::{f16_to_f32, f32_to_f16_quant, silu_fast};
pub use k_block::{
    dequant_q4_k, dequant_q5_k, dequant_q6_k, dot_q4_0_f32, dot_q4_k_f32, dot_q5_k_f32,
    dot_q6_k_f32, dot_q8_f32,
};
pub use nvfp4::{
    dequant_nvfp4_row, dot_nvfp4_a16_q8_f32, dot_nvfp4_block_fused_f32, dot_nvfp4_f32,
    dot_nvfp4_w4a4_f32, e2m1_from_f32, e2m1_to_f32, e4m3fn_from_f32, e4m3fn_to_f32,
    quantize_nvfp4_activation, quantize_nvfp4_q8_block, Nvfp4Q8Block,
};
