//! Fixed Muse primitive pipeline registry and dispatch helpers.

pub mod attn;
pub mod ffn;
pub mod gate;
pub mod lmhead;
pub mod multicol;
pub mod norm;
pub mod qkv;
pub mod rope;

use metal::{
    ComputeCommandEncoderRef, ComputePipelineState, ComputePipelineStateRef,
    FunctionConstantValues, MTLDataType, MTLSize,
};

use super::context::{MetalContext, MetalError};
use super::pso_cache::PsoCache;
use multicol::MultiColPipelines;

const PIPELINES: [&str; 66] = [
    "rms_norm_batch",
    "fused_rms_norm_residual_add_batch",
    "muser_fused_norm_residual_rms_norm_batch_dual_eps",
    "muser_fused_norm_residual_rms_norm_32sg",
    "sigmoid_gate_inplace",
    "rope_norm_batch_cached",
    "ffn_q4k_gate_up_silu_4r2s",
    "muser_silu_mul_inplace",
    "muser_scale_softcap_inplace",
    "muser_matmul_q4k",
    "muser_matmul_q5k",
    "muser_embedding_q4k",
    "muser_kv_store_f16",
    "muser_attention_decode_f32",
    "muser_attention_decode_splitk_f16",
    "muser_attention_decode_splitk_reduce_f32",
    "muser_kv_store_batch_f16",
    "muser_stage_swa_prefill_f16",
    "muser_stage_swa_llama_decode_f16",
    "muser_fa_causal_mask_f16",
    "muser_attention_prefill_f32",
    "muser_attention_prefill_flash_f16",
    "flash_attn_v2",
    "muser_flash_attn_decode_gqa_fa2",
    "muser_copy_row_f32",
    "muser_matvec_q4k_4r2s",
    "muser_matvec_q5k_4sg",
    "matmul_f32_batch",
    "matmul_f32_batch_tiled",
    "matmul_f32_batch_tiled_8x8",
    "matmul_q4k_batch_sgm_aligned",
    "m16_q4k_n32",
    "m16_q5k_n32",
    "m16_q6k_n32",
    "residual_add_batch",
    "silu_hadamard_batch",
    "rms_norm_per_head",
    "rms_norm_per_head_qkv_fused",
    "dflash_dual_attention_f32",
    "rms_norm_batch_inplace",
    "fused_residual_rms_norm_batch",
    "rope_batch_cached",
    "copy_f32_buffer",
    "pack_dflash_layer_major_f32",
    "argmax_f32_phase1",
    "argmax_f32_phase2",
    "greedy_argmax_f32_phase1",
    "greedy_argmax_f32_phase2",
    "muser_nvfp4_matvec_c1",
    "muser_nvfp4_matvec_c2",
    "muser_nvfp4_matvec_c4",
    "muser_nvfp4_matvec_c8",
    "muser_nvfp4_matvec_c16",
    "muser_nvfp4_w4a4_matvec_c1",
    "muser_nvfp4_w4a4_matvec_c2",
    "muser_nvfp4_w4a4_matvec_c4",
    "muser_nvfp4_w4a4_matvec_c8",
    "muser_nvfp4_w4a4_m16_n32",
    "muser_nvfp4_w4a4_matvec_c16",
    "muser_nvfp4_dequant_fixture",
    "muser_f16_matvec_c1",
    "muser_f16_matvec_c2",
    "muser_f16_matvec_c4",
    "muser_f16_matvec_c8",
    "muser_f16_matvec_c16",
    "muser_embedding_f16",
];

pub struct MetalKernels {
    cache: PsoCache,
    cross_vendor_q4k: ComputePipelineState,
    cross_vendor_q5k: ComputePipelineState,
    cross_vendor_q6k: ComputePipelineState,
    cross_vendor_rms_per_head: ComputePipelineState,
    cross_vendor_rms_unweighted: ComputePipelineState,
    cross_vendor_mul_weight: ComputePipelineState,
    cross_vendor_swiglu: ComputePipelineState,
    cross_vendor_scale: ComputePipelineState,
    cross_vendor_tanh: ComputePipelineState,
    cross_vendor_rope: ComputePipelineState,
    cross_vendor_rope_neox: ComputePipelineState,
    cross_vendor_attention_decode: ComputePipelineState,
    cross_vendor_attention_prefill: ComputePipelineState,
    cross_vendor_sigmoid_gate: ComputePipelineState,
    cross_vendor_dual_norm_residual: ComputePipelineState,
    cross_vendor_residual_add: ComputePipelineState,
    cross_vendor_nvfp4_a16_q8: ComputePipelineState,
    cross_vendor_nvfp4_w4a4_c1: ComputePipelineState,
    cross_vendor_nvfp4_w4a4_c2: ComputePipelineState,
    cross_vendor_nvfp4_w4a4_c4: ComputePipelineState,
    cross_vendor_nvfp4_w4a4_c8: ComputePipelineState,
    cross_vendor_nvfp4_w4a4_quantize_m16: ComputePipelineState,
    cross_vendor_nvfp4_w4a4_prequant_m16_n32: ComputePipelineState,
    cross_vendor_nvfp4_w4a4_m16_n32: ComputePipelineState,
    cross_vendor_nvfp4_w4a4_c16: ComputePipelineState,
    ggml_q4k: Option<ComputePipelineState>,
    ggml_q5k: Option<ComputePipelineState>,
    ggml_q6k: Option<ComputePipelineState>,
    ggml_q4k_mm_aligned: Option<ComputePipelineState>,
    ggml_q4k_mm_bounds: Option<ComputePipelineState>,
    ggml_q5k_mm_aligned: Option<ComputePipelineState>,
    ggml_q5k_mm_bounds: Option<ComputePipelineState>,
    ggml_q6k_mm_aligned: Option<ComputePipelineState>,
    ggml_q6k_mm_bounds: Option<ComputePipelineState>,
    ggml_rms_norm_mul: Option<ComputePipelineState>,
    ggml_rms_norm_mul_add: Option<ComputePipelineState>,
    ggml_unary_scale: Option<ComputePipelineState>,
    ggml_unary_tanh: Option<ComputePipelineState>,
    ggml_rope_norm: Option<ComputePipelineState>,
    llama_mul_mv_ext: Option<LlamaMulMvExtPipelines>,
    llama_flash: Option<LlamaFlashAttnPipelines>,
    /// Default-off verify route, compiled only when `MUSER_MULTI_COL_VERIFY` is set.
    multicol: Option<MultiColPipelines>,
    ferrite_f16_interleaved_nsg1: ComputePipelineState,
    ferrite_f16_interleaved_nsg2: ComputePipelineState,
    ferrite_f16_interleaved_nsg4: ComputePipelineState,
    ferrite_f16_reduce: ComputePipelineState,
}

pub(crate) struct DFlashPackGeometry {
    pub(crate) source_tokens: usize,
    pub(crate) source_start: usize,
    pub(crate) output_tokens: usize,
    pub(crate) layers: usize,
    pub(crate) hidden: usize,
}

/// llama.cpp Metal `flash_attn_ext_vec` + pad + reduce from the pinned metallib.
pub(super) struct LlamaFlashAttnPipelines {
    /// Masked ns10=128 head-major NoPE, [nsg=1,2,4][kvpad].
    vec_ns128: [[ComputePipelineState; 2]; 3],
    /// Masked ns10=256 token-major SWA, [nsg=1,2,4][kvpad].
    vec_ns256: [[ComputePipelineState; 2]; 3],
    /// Existing mask-free growing-cache route, retained for short contexts.
    vec_unmasked_ns128: [[ComputePipelineState; 2]; 3],
    vec_unmasked_ns256: [[ComputePipelineState; 2]; 3],
    reduce: ComputePipelineState,
    pad: ComputePipelineState,
    pad_unmasked: ComputePipelineState,
    /// Masked causal non-vec prefill kernel: `flash_attn_ext_f16_dk128_dv128`
    /// with mask=true, sinks/bias/scap/kvpad/bcm=false, ns10=ns20=128, nsg=4.
    prefill: ComputePipelineState,
    /// Mask block classifier (`flash_attn_ext_blk`, nqptg=8, ncpsg=32) that
    /// precomputes the skip/partial/dense byte per 8x32 mask block.
    prefill_blk: ComputePipelineState,
}

/// llama.cpp's source-pinned small-batch K-quant projection pipelines.
///
/// The upstream Metal backend changes from repeated `mul_mv` to these
/// `mul_mv_ext` kernels at batch size four. Keeping that dispatch boundary is
/// required for numerical API parity as well as performance parity.
pub(super) struct LlamaMulMvExtPipelines {
    /// Pipeline slots correspond to r1ptg 2, 3, 4, and 5.
    q4k: [ComputePipelineState; 4],
    q5k: [ComputePipelineState; 4],
    q6k: [ComputePipelineState; 4],
}

fn cross_vendor_pipeline(
    context: &MetalContext,
    name: &'static str,
) -> Result<ComputePipelineState, MetalError> {
    let function = context
        .cross_vendor_library
        .get_function(name, None)
        .map_err(|message| MetalError::Pipeline {
            name: name.to_string(),
            message,
        })?;
    context
        .device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|message| MetalError::Pipeline {
            name: name.to_string(),
            message,
        })
}

impl MetalKernels {
    pub fn new(context: &MetalContext) -> Result<Self, MetalError> {
        Ok(Self {
            cache: PsoCache::new(context, PIPELINES)?,
            cross_vendor_q4k: cross_vendor_pipeline(context, "muser_cross_vendor_q4k")?,
            cross_vendor_q5k: cross_vendor_pipeline(context, "muser_cross_vendor_q5k")?,
            cross_vendor_q6k: cross_vendor_pipeline(context, "muser_cross_vendor_q6k")?,
            cross_vendor_rms_per_head: cross_vendor_pipeline(
                context,
                "muser_cross_vendor_rms_per_head",
            )?,
            cross_vendor_rms_unweighted: cross_vendor_pipeline(
                context,
                "muser_cross_vendor_rms_unweighted",
            )?,
            cross_vendor_mul_weight: cross_vendor_pipeline(
                context,
                "muser_cross_vendor_mul_weight",
            )?,
            cross_vendor_swiglu: cross_vendor_pipeline(context, "muser_cross_vendor_swiglu")?,
            cross_vendor_scale: cross_vendor_pipeline(context, "muser_cross_vendor_scale")?,
            cross_vendor_tanh: cross_vendor_pipeline(context, "muser_cross_vendor_tanh")?,
            cross_vendor_rope: cross_vendor_pipeline(context, "muser_cross_vendor_rope")?,
            cross_vendor_rope_neox: cross_vendor_pipeline(context, "muser_cross_vendor_rope_neox")?,
            cross_vendor_attention_decode: cross_vendor_pipeline(
                context,
                "muser_cross_vendor_attention_decode",
            )?,
            cross_vendor_attention_prefill: cross_vendor_pipeline(
                context,
                "muser_cross_vendor_attention_prefill",
            )?,
            cross_vendor_sigmoid_gate: cross_vendor_pipeline(
                context,
                "muser_cross_vendor_sigmoid_gate",
            )?,
            cross_vendor_dual_norm_residual: cross_vendor_pipeline(
                context,
                "muser_cross_vendor_dual_norm_residual",
            )?,
            cross_vendor_residual_add: cross_vendor_pipeline(
                context,
                "muser_cross_vendor_residual_add",
            )?,
            cross_vendor_nvfp4_a16_q8: cross_vendor_pipeline(context, "muser_nvfp4_a16_q8_matvec")?,
            cross_vendor_nvfp4_w4a4_c1: cross_vendor_pipeline(
                context,
                "muser_nvfp4_w4a4_matvec_c1",
            )?,
            cross_vendor_nvfp4_w4a4_c2: cross_vendor_pipeline(
                context,
                "muser_nvfp4_w4a4_matvec_c2",
            )?,
            cross_vendor_nvfp4_w4a4_c4: cross_vendor_pipeline(
                context,
                "muser_nvfp4_w4a4_matvec_c4",
            )?,
            cross_vendor_nvfp4_w4a4_c8: cross_vendor_pipeline(
                context,
                "muser_nvfp4_w4a4_matvec_c8",
            )?,
            cross_vendor_nvfp4_w4a4_quantize_m16: cross_vendor_pipeline(
                context,
                "muser_nvfp4_w4a4_quantize_m16",
            )?,
            cross_vendor_nvfp4_w4a4_prequant_m16_n32: cross_vendor_pipeline(
                context,
                "muser_nvfp4_w4a4_prequant_m16_n32",
            )?,
            cross_vendor_nvfp4_w4a4_m16_n32: cross_vendor_pipeline(
                context,
                "muser_nvfp4_w4a4_m16_n32",
            )?,
            cross_vendor_nvfp4_w4a4_c16: cross_vendor_pipeline(
                context,
                "muser_nvfp4_w4a4_matvec_c16",
            )?,
            ggml_q4k: ggml_matvec_pipeline(context, "kernel_mul_mv_q4_K_f32")?,
            ggml_q5k: ggml_matvec_pipeline(context, "kernel_mul_mv_q5_K_f32")?,
            ggml_q6k: ggml_matvec_pipeline(context, "kernel_mul_mv_q6_K_f32")?,
            ggml_q4k_mm_aligned: ggml_matmul_pipeline(context, "kernel_mul_mm_q4_K_f32", false)?,
            ggml_q4k_mm_bounds: ggml_matmul_pipeline(context, "kernel_mul_mm_q4_K_f32", true)?,
            ggml_q5k_mm_aligned: ggml_matmul_pipeline(context, "kernel_mul_mm_q5_K_f32", false)?,
            ggml_q5k_mm_bounds: ggml_matmul_pipeline(context, "kernel_mul_mm_q5_K_f32", true)?,
            ggml_q6k_mm_aligned: ggml_matmul_pipeline(context, "kernel_mul_mm_q6_K_f32", false)?,
            ggml_q6k_mm_bounds: ggml_matmul_pipeline(context, "kernel_mul_mm_q6_K_f32", true)?,
            ggml_rms_norm_mul: ggml_plain_pipeline(context, "kernel_rms_norm_mul_f32_4")?,
            ggml_rms_norm_mul_add: ggml_plain_pipeline(context, "kernel_rms_norm_mul_add_f32_4")?,
            ggml_unary_scale: ggml_unary_pipeline(context, 10)?,
            ggml_unary_tanh: ggml_unary_pipeline(context, 100)?,
            ggml_rope_norm: ggml_rope_pipeline(context)?,
            llama_mul_mv_ext: llama_mul_mv_ext_pipelines(context)?,
            llama_flash: llama_flash_attn_pipelines(context)?,
            // Continuous decode needs the exact Q4_K/Q5_K multi-column
            // kernels independently of the experimental DFlash route flag.
            // The DFlash dispatcher below remains flag-gated.
            multicol: Some(MultiColPipelines::new(context)?),
            ferrite_f16_interleaved_nsg1: ferrite_f16_pipeline(
                context,
                "flash_attn_decode_vec_f16_gqa_interleaved",
                1,
            )?,
            ferrite_f16_interleaved_nsg2: ferrite_f16_pipeline(
                context,
                "flash_attn_decode_vec_f16_gqa_interleaved",
                2,
            )?,
            ferrite_f16_interleaved_nsg4: ferrite_f16_pipeline(
                context,
                "flash_attn_decode_vec_f16_gqa_interleaved",
                4,
            )?,
            ferrite_f16_reduce: ferrite_f16_pipeline(context, "flash_attn_decode_reduce_v2", 1)?,
        })
    }

    pub(super) fn bind(&self, encoder: &ComputeCommandEncoderRef, name: &'static str) {
        encoder.set_compute_pipeline_state(self.cache.get(name));
    }

    pub(super) fn cross_vendor_nvfp4_w4a4(&self, columns: usize) -> &ComputePipelineStateRef {
        match columns {
            16 => &self.cross_vendor_nvfp4_w4a4_c16,
            8 => &self.cross_vendor_nvfp4_w4a4_c8,
            4 => &self.cross_vendor_nvfp4_w4a4_c4,
            2 => &self.cross_vendor_nvfp4_w4a4_c2,
            1 => &self.cross_vendor_nvfp4_w4a4_c1,
            _ => unreachable!("NVFP4 column specialization is fixed"),
        }
    }

    pub(super) fn bind_cross_vendor(
        &self,
        encoder: &ComputeCommandEncoderRef,
        dtype: crate::gguf::GgmlType,
    ) {
        let pipeline = match dtype {
            crate::gguf::GgmlType::Q4_K => &self.cross_vendor_q4k,
            crate::gguf::GgmlType::Q5_K => &self.cross_vendor_q5k,
            crate::gguf::GgmlType::Q6_K => &self.cross_vendor_q6k,
            _ => unreachable!("cross-vendor projection is only Q4_K/Q5_K/Q6_K"),
        };
        encoder.set_compute_pipeline_state(pipeline);
    }

    pub(super) fn ggml_matvec(
        &self,
        dtype: crate::gguf::GgmlType,
    ) -> Option<&ComputePipelineStateRef> {
        match dtype {
            crate::gguf::GgmlType::Q4_K => self.ggml_q4k.as_ref().map(AsRef::as_ref),
            crate::gguf::GgmlType::Q5_K => self.ggml_q5k.as_ref().map(AsRef::as_ref),
            crate::gguf::GgmlType::Q6_K => self.ggml_q6k.as_ref().map(AsRef::as_ref),
            _ => None,
        }
    }

    pub(super) fn ggml_rms_norm_mul(&self) -> Option<&ComputePipelineStateRef> {
        self.ggml_rms_norm_mul.as_deref()
    }

    pub(super) fn ggml_rms_norm_mul_add(&self) -> Option<&ComputePipelineStateRef> {
        self.ggml_rms_norm_mul_add.as_deref()
    }

    pub(super) fn ggml_rope_norm(&self) -> Option<&ComputePipelineStateRef> {
        self.ggml_rope_norm.as_deref()
    }

    pub(crate) fn supports_projection(&self, dtype: crate::gguf::GgmlType) -> bool {
        match dtype {
            // Standalone fallbacks cover Q4_K and Q5_K for both decode and
            // batch prefill. Q6_K intentionally uses the pinned upstream
            // llama kernels so its math and dispatch remain comparator-exact.
            crate::gguf::GgmlType::Q4_K | crate::gguf::GgmlType::Q5_K => true,
            crate::gguf::GgmlType::Q6_K => {
                self.ggml_q6k.is_some()
                    && self.ggml_q6k_mm_aligned.is_some()
                    && self.ggml_q6k_mm_bounds.is_some()
            }
            crate::gguf::GgmlType::NVFP4_E2M1 => true,
            crate::gguf::GgmlType::F16 => true,
            _ => false,
        }
    }

    pub(super) fn ggml_matmul(
        &self,
        dtype: crate::gguf::GgmlType,
        bounds: bool,
    ) -> Option<&ComputePipelineStateRef> {
        match (dtype, bounds) {
            (crate::gguf::GgmlType::Q4_K, false) => self.ggml_q4k_mm_aligned.as_deref(),
            (crate::gguf::GgmlType::Q4_K, true) => self.ggml_q4k_mm_bounds.as_deref(),
            (crate::gguf::GgmlType::Q5_K, false) => self.ggml_q5k_mm_aligned.as_deref(),
            (crate::gguf::GgmlType::Q5_K, true) => self.ggml_q5k_mm_bounds.as_deref(),
            (crate::gguf::GgmlType::Q6_K, false) => self.ggml_q6k_mm_aligned.as_deref(),
            (crate::gguf::GgmlType::Q6_K, true) => self.ggml_q6k_mm_bounds.as_deref(),
            _ => None,
        }
    }

    pub(super) fn llama_mul_mv_ext(
        &self,
        dtype: crate::gguf::GgmlType,
        r1ptg: usize,
    ) -> Option<&ComputePipelineStateRef> {
        self.llama_mul_mv_ext
            .as_ref()
            .and_then(|pipelines| pipelines.pipeline(dtype, r1ptg))
    }

    pub(crate) fn has_llama_mul_mv_ext(&self) -> bool {
        self.llama_mul_mv_ext.is_some()
    }

    pub(super) fn multicol(&self) -> Option<&MultiColPipelines> {
        self.multicol.as_ref()
    }

    pub(crate) fn has_llama_flash_attn_vec(&self) -> bool {
        self.llama_flash.is_some()
    }

    pub(super) fn llama_flash(&self) -> Option<&LlamaFlashAttnPipelines> {
        self.llama_flash.as_ref()
    }

    pub(super) fn ferrite_f16_interleaved(&self, nsg: usize) -> &ComputePipelineStateRef {
        match nsg {
            1 => &self.ferrite_f16_interleaved_nsg1,
            2 => &self.ferrite_f16_interleaved_nsg2,
            4 => &self.ferrite_f16_interleaved_nsg4,
            _ => panic!("unsupported Ferrite interleaved NSG {nsg}"),
        }
    }

    pub(super) fn ferrite_f16_reduce(&self) -> &ComputePipelineStateRef {
        &self.ferrite_f16_reduce
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_dense_f32_batch(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weights: &super::buffer::GpuBuffer,
        input: &super::buffer::GpuBuffer,
        output: &super::buffer::GpuBuffer,
        rows: usize,
        cols: usize,
        batch: usize,
    ) {
        let cols_u32 = cols as u32;
        let rows_u32 = rows as u32;
        let batch_u32 = batch as u32;
        if batch >= 4 && rows >= 4 {
            self.bind(encoder, "matmul_f32_batch_tiled");
            encoder.set_buffer(0, Some(weights.metal()), 0);
            encoder.set_buffer(1, Some(input.metal()), 0);
            encoder.set_buffer(2, Some(output.metal()), 0);
            set_value(encoder, 3, &cols_u32);
            set_value(encoder, 4, &rows_u32);
            set_value(encoder, 5, &batch_u32);
            encoder.dispatch_thread_groups(
                MTLSize::new(rows.div_ceil(4) as u64, batch.div_ceil(4) as u64, 1),
                MTLSize::new(32, 1, 1),
            );
        } else {
            self.bind(encoder, "matmul_f32_batch");
            encoder.set_buffer(0, Some(weights.metal()), 0);
            encoder.set_buffer(1, Some(input.metal()), 0);
            encoder.set_buffer(2, Some(output.metal()), 0);
            set_value(encoder, 3, &cols_u32);
            set_value(encoder, 4, &rows_u32);
            encoder.dispatch_thread_groups(
                MTLSize::new(rows as u64, batch as u64, 1),
                MTLSize::new(32, 1, 1),
            );
        }
    }

    pub(crate) fn encode_silu_hadamard_batch(
        &self,
        encoder: &ComputeCommandEncoderRef,
        gate: &super::buffer::GpuBuffer,
        up: &super::buffer::GpuBuffer,
        total: usize,
    ) {
        self.bind(encoder, "silu_hadamard_batch");
        encoder.set_buffer(0, Some(gate.metal()), 0);
        encoder.set_buffer(1, Some(up.metal()), 0);
        set_value(encoder, 2, &(total as u32));
        let threads = total.div_ceil(4);
        encoder.dispatch_thread_groups(
            MTLSize::new(threads.div_ceil(1024) as u64, 1, 1),
            MTLSize::new(1024, 1, 1),
        );
    }

    pub(crate) fn encode_residual_add_batch(
        &self,
        encoder: &ComputeCommandEncoderRef,
        destination: &super::buffer::GpuBuffer,
        source: &super::buffer::GpuBuffer,
        total: usize,
    ) {
        if std::env::var_os("MUSER_CROSS_VENDOR_QK").is_some() {
            encoder.set_compute_pipeline_state(&self.cross_vendor_residual_add);
            encoder.set_buffer(0, Some(destination.metal()), 0);
            encoder.set_buffer(1, Some(source.metal()), 0);
            set_value(encoder, 2, &(total as u32));
            dispatch_1d(encoder, total);
            return;
        }
        self.bind(encoder, "residual_add_batch");
        encoder.set_buffer(0, Some(destination.metal()), 0);
        encoder.set_buffer(1, Some(source.metal()), 0);
        set_value(encoder, 2, &(total as u32));
        encoder.dispatch_thread_groups(
            MTLSize::new(total.div_ceil(1024) as u64, 1, 1),
            MTLSize::new(1024, 1, 1),
        );
    }

    pub(crate) fn encode_rms_norm_inplace(
        &self,
        encoder: &ComputeCommandEncoderRef,
        values: &super::buffer::GpuBuffer,
        weight: &super::buffer::GpuBuffer,
        dim: usize,
        eps: f32,
        rows: usize,
    ) {
        if std::env::var_os("MUSER_CROSS_VENDOR_QK").is_some() {
            // The DFlash target projection is normalized in place before its
            // persistent K/V projections.  Use the same fixed stride-32 reduction and
            // explicit learned-weight store boundary as the CUDA producer;
            // the ordinary SIMD reduction is vendor-dependent at the last
            // bits and otherwise poisons every imported draft-cache row.
            self.encode_cross_vendor_rms_then_weight(
                encoder, values, weight, values, rows, dim, eps,
            );
            return;
        }
        self.bind(encoder, "rms_norm_batch_inplace");
        encoder.set_buffer(0, Some(values.metal()), 0);
        encoder.set_buffer(1, Some(weight.metal()), 0);
        set_value(encoder, 2, &(dim as u32));
        set_value(encoder, 3, &eps);
        encoder.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(32, 1, 1));
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_fused_residual_norm(
        &self,
        encoder: &ComputeCommandEncoderRef,
        hidden: &super::buffer::GpuBuffer,
        delta: &super::buffer::GpuBuffer,
        weight: &super::buffer::GpuBuffer,
        normed: &super::buffer::GpuBuffer,
        dim: usize,
        eps: f32,
        rows: usize,
    ) {
        if std::env::var_os("MUSER_CROSS_VENDOR_QK").is_some() {
            encoder.set_compute_pipeline_state(&self.cross_vendor_residual_add);
            encoder.set_buffer(0, Some(hidden.metal()), 0);
            encoder.set_buffer(1, Some(delta.metal()), 0);
            set_value(encoder, 2, &((dim * rows) as u32));
            dispatch_1d(encoder, dim * rows);
            let residual_barrier: [&metal::ResourceRef; 1] = [hidden.metal()];
            encoder.memory_barrier_with_resources(&residual_barrier);
            self.encode_cross_vendor_rms_then_weight(
                encoder, hidden, weight, normed, rows, dim, eps,
            );
            return;
        }
        self.bind(encoder, "fused_residual_rms_norm_batch");
        encoder.set_buffer(0, Some(hidden.metal()), 0);
        encoder.set_buffer(1, Some(delta.metal()), 0);
        encoder.set_buffer(2, Some(weight.metal()), 0);
        encoder.set_buffer(3, Some(normed.metal()), 0);
        set_value(encoder, 4, &(dim as u32));
        set_value(encoder, 5, &eps);
        encoder.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(32, 1, 1));
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_rms_norm_per_head(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &super::buffer::GpuBuffer,
        weight: &super::buffer::GpuBuffer,
        output: &super::buffer::GpuBuffer,
        heads: usize,
        head_dim: usize,
        eps: f32,
    ) {
        if std::env::var_os("MUSER_CROSS_VENDOR_QK").is_some() {
            self.encode_cross_vendor_rms_then_weight(
                encoder, input, weight, output, heads, head_dim, eps,
            );
            return;
        }
        self.bind(encoder, "rms_norm_per_head");
        encoder.set_buffer(0, Some(input.metal()), 0);
        encoder.set_buffer(1, Some(weight.metal()), 0);
        encoder.set_buffer(2, Some(output.metal()), 0);
        set_value(encoder, 3, &(head_dim as u32));
        set_value(encoder, 4, &eps);
        encoder.dispatch_thread_groups(
            MTLSize::new(heads as u64, 1, 1),
            MTLSize::new(head_dim as u64, 1, 1),
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_cross_vendor_rms_per_head(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &super::buffer::GpuBuffer,
        weight: &super::buffer::GpuBuffer,
        output: &super::buffer::GpuBuffer,
        heads: usize,
        head_dim: usize,
        eps: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.cross_vendor_rms_per_head);
        encoder.set_buffer(0, Some(input.metal()), 0);
        encoder.set_buffer(1, Some(weight.metal()), 0);
        encoder.set_buffer(2, Some(output.metal()), 0);
        set_value(encoder, 3, &(head_dim as u32));
        set_value(encoder, 4, &eps);
        encoder.dispatch_thread_groups(MTLSize::new(heads as u64, 1, 1), MTLSize::new(32, 1, 1));
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_rope_neox_batch_cached(
        &self,
        encoder: &ComputeCommandEncoderRef,
        q: &super::buffer::GpuBuffer,
        k: &super::buffer::GpuBuffer,
        frequencies: &super::buffer::GpuBuffer,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        start_position: usize,
        batch: usize,
    ) {
        if std::env::var_os("MUSER_CROSS_VENDOR_QK").is_some() {
            // DFlash uses the NEOX half-split representation.  Keep that
            // representation unchanged, but consume the same canonical NCO
            // `(cos, sin)` bytes as the CUDA producer.  Layout conversion is
            // deliberately not part of this math parity route.
            assert!(
                frequencies.len() >= (start_position + batch) * head_dim,
                "strict DFlash RoPE table does not cover the requested positions"
            );
            encoder.set_compute_pipeline_state(&self.cross_vendor_rope_neox);
            encoder.set_buffer(0, Some(q.metal()), 0);
            encoder.set_buffer(1, Some(k.metal()), 0);
            encoder.set_buffer(2, Some(frequencies.metal()), 0);
            set_value(encoder, 3, &(n_heads as u32));
            set_value(encoder, 4, &(n_kv_heads as u32));
            set_value(encoder, 5, &(head_dim as u32));
            set_value(encoder, 6, &(start_position as u32));
            let pairs = (n_heads + n_kv_heads) * (head_dim / 2);
            encoder.dispatch_thread_groups(
                MTLSize::new(pairs.div_ceil(32) as u64, batch as u64, 1),
                MTLSize::new(32, 1, 1),
            );
            return;
        }
        self.bind(encoder, "rope_batch_cached");
        encoder.set_buffer(0, Some(q.metal()), 0);
        encoder.set_buffer(1, Some(k.metal()), 0);
        encoder.set_buffer(2, Some(frequencies.metal()), 0);
        set_value(encoder, 3, &(n_heads as u32));
        set_value(encoder, 4, &(n_kv_heads as u32));
        set_value(encoder, 5, &(head_dim as u32));
        set_value(encoder, 6, &(start_position as u32));
        let pairs = (n_heads + n_kv_heads) * (head_dim / 2);
        encoder.dispatch_thread_groups(
            MTLSize::new(pairs.div_ceil(32) as u64, batch as u64, 1),
            MTLSize::new(32, 1, 1),
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_dflash_dual_attention(
        &self,
        encoder: &ComputeCommandEncoderRef,
        q: &super::buffer::GpuBuffer,
        k_cache: &super::buffer::GpuBuffer,
        v_cache: &super::buffer::GpuBuffer,
        k_context: &super::buffer::GpuBuffer,
        v_context: &super::buffer::GpuBuffer,
        k_noise: &super::buffer::GpuBuffer,
        v_noise: &super::buffer::GpuBuffer,
        output: &super::buffer::GpuBuffer,
        head_dim: usize,
        cached_context: usize,
        fresh_context: usize,
        n_heads: usize,
        n_kv_heads: usize,
        batch: usize,
    ) {
        self.bind(encoder, "dflash_dual_attention_f32");
        for (index, buffer) in [
            q, k_cache, v_cache, k_context, v_context, k_noise, v_noise, output,
        ]
        .into_iter()
        .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer.metal()), 0);
        }
        set_value(encoder, 8, &(head_dim as u32));
        set_value(encoder, 9, &(cached_context as u32));
        set_value(encoder, 10, &(fresh_context as u32));
        set_value(encoder, 11, &(n_heads as u32));
        set_value(encoder, 12, &(n_kv_heads as u32));
        set_value(encoder, 13, &(batch as u32));
        encoder.dispatch_thread_groups(
            MTLSize::new(batch as u64, n_kv_heads as u64, 1),
            MTLSize::new((n_heads / n_kv_heads * 32) as u64, 1, 1),
        );
    }

    pub(crate) fn encode_copy_f32(
        &self,
        encoder: &ComputeCommandEncoderRef,
        source: &super::buffer::GpuBuffer,
        destination: &super::buffer::GpuBuffer,
        count: usize,
    ) {
        self.bind(encoder, "copy_f32_buffer");
        encoder.set_buffer(0, Some(source.metal()), 0);
        encoder.set_buffer(1, Some(destination.metal()), 0);
        set_value(encoder, 2, &(count as u32));
        dispatch_1d(encoder, count);
    }

    pub(crate) fn encode_copy_f32_region(
        &self,
        encoder: &ComputeCommandEncoderRef,
        source: &super::buffer::GpuBuffer,
        source_offset: usize,
        destination: &super::buffer::GpuBuffer,
        destination_offset: usize,
        count: usize,
    ) {
        debug_assert!(source_offset + count <= source.len());
        debug_assert!(destination_offset + count <= destination.len());
        self.bind(encoder, "copy_f32_buffer");
        encoder.set_buffer(
            0,
            Some(source.metal()),
            (source_offset * std::mem::size_of::<f32>()) as u64,
        );
        encoder.set_buffer(
            1,
            Some(destination.metal()),
            (destination_offset * std::mem::size_of::<f32>()) as u64,
        );
        set_value(encoder, 2, &(count as u32));
        dispatch_1d(encoder, count);
    }

    pub(crate) fn encode_pack_dflash_layer_major(
        &self,
        encoder: &ComputeCommandEncoderRef,
        source: &super::buffer::GpuBuffer,
        destination: &super::buffer::GpuBuffer,
        geometry: DFlashPackGeometry,
    ) {
        let DFlashPackGeometry {
            source_tokens,
            source_start,
            output_tokens,
            layers,
            hidden,
        } = geometry;
        debug_assert!(source_start + output_tokens <= source_tokens);
        debug_assert!(source.len() >= source_tokens * layers * hidden);
        debug_assert!(destination.len() >= output_tokens * layers * hidden);
        self.bind(encoder, "pack_dflash_layer_major_f32");
        encoder.set_buffer(0, Some(source.metal()), 0);
        encoder.set_buffer(1, Some(destination.metal()), 0);
        set_value(encoder, 2, &(source_tokens as u32));
        set_value(encoder, 3, &(source_start as u32));
        set_value(encoder, 4, &(output_tokens as u32));
        set_value(encoder, 5, &(layers as u32));
        set_value(encoder, 6, &(hidden as u32));
        dispatch_1d(encoder, output_tokens * layers * hidden);
    }
}

fn ferrite_f16_pipeline(
    context: &MetalContext,
    name: &str,
    nsg: u32,
) -> Result<ComputePipelineState, MetalError> {
    let constants = FunctionConstantValues::new();
    let head_dim = 128u32;
    let decode_params = false;
    constants.set_constant_value_at_index(
        &head_dim as *const u32 as *const std::ffi::c_void,
        MTLDataType::UInt,
        40,
    );
    constants.set_constant_value_at_index(
        &decode_params as *const bool as *const std::ffi::c_void,
        MTLDataType::Bool,
        92,
    );
    constants.set_constant_value_at_index(
        &nsg as *const u32 as *const std::ffi::c_void,
        MTLDataType::UInt,
        98,
    );
    let function = context
        .library
        .get_function(name, Some(constants))
        .map_err(|message| MetalError::Pipeline {
            name: format!("{name}[dk=128,nsg={nsg}]"),
            message,
        })?;
    context
        .device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|message| MetalError::Pipeline {
            name: format!("{name}[dk=128,nsg={nsg}]"),
            message,
        })
}

fn ggml_matvec_pipeline(
    context: &MetalContext,
    name: &str,
) -> Result<Option<ComputePipelineState>, MetalError> {
    let Some(library) = context.ggml_library.as_ref() else {
        return Ok(None);
    };
    let constants = FunctionConstantValues::new();
    for (value, index) in [(2i16, 600u64), (1, 602), (1, 603), (1, 604)] {
        constants.set_constant_value_at_index(
            &value as *const i16 as *const std::ffi::c_void,
            MTLDataType::Short,
            index,
        );
    }
    let function = library
        .get_function(name, Some(constants))
        .map_err(|message| MetalError::Pipeline {
            name: name.into(),
            message,
        })?;
    let pipeline = context
        .device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|message| MetalError::Pipeline {
            name: name.into(),
            message,
        })?;
    Ok(Some(pipeline))
}

fn ggml_plain_pipeline(
    context: &MetalContext,
    name: &str,
) -> Result<Option<ComputePipelineState>, MetalError> {
    let Some(library) = context.ggml_library.as_ref() else {
        return Ok(None);
    };
    let function = library
        .get_function(name, None)
        .map_err(|message| MetalError::Pipeline {
            name: name.into(),
            message,
        })?;
    context
        .device
        .new_compute_pipeline_state_with_function(&function)
        .map(Some)
        .map_err(|message| MetalError::Pipeline {
            name: name.into(),
            message,
        })
}

fn ggml_unary_pipeline(
    context: &MetalContext,
    operation: i16,
) -> Result<Option<ComputePipelineState>, MetalError> {
    let Some(library) = context.ggml_library.as_ref() else {
        return Ok(None);
    };
    let constants = FunctionConstantValues::new();
    let contiguous_small = false;
    constants.set_constant_value_at_index(
        &operation as *const i16 as *const std::ffi::c_void,
        MTLDataType::Short,
        1200,
    );
    constants.set_constant_value_at_index(
        &contiguous_small as *const bool as *const std::ffi::c_void,
        MTLDataType::Bool,
        1201,
    );
    let name = "kernel_unary_f32_f32_4";
    let function = library
        .get_function(name, Some(constants))
        .map_err(|message| MetalError::Pipeline {
            name: format!("{name}[op={operation},cnt=0]"),
            message,
        })?;
    context
        .device
        .new_compute_pipeline_state_with_function(&function)
        .map(Some)
        .map_err(|message| MetalError::Pipeline {
            name: format!("{name}[op={operation},cnt=0]"),
            message,
        })
}

fn ggml_rope_pipeline(context: &MetalContext) -> Result<Option<ComputePipelineState>, MetalError> {
    let Some(library) = context.ggml_library.as_ref() else {
        return Ok(None);
    };
    let constants = FunctionConstantValues::new();
    for index in [800, 801] {
        let disabled = false;
        constants.set_constant_value_at_index(
            &disabled as *const bool as *const std::ffi::c_void,
            MTLDataType::Bool,
            index,
        );
    }
    let name = "kernel_rope_norm_f32";
    let function = library
        .get_function(name, Some(constants))
        .map_err(|message| MetalError::Pipeline {
            name: format!("{name}[imrope=false,is_back=false]"),
            message,
        })?;
    context
        .device
        .new_compute_pipeline_state_with_function(&function)
        .map(Some)
        .map_err(|message| MetalError::Pipeline {
            name: format!("{name}[imrope=false,is_back=false]"),
            message,
        })
}

const FC_MUL_MV: u64 = 600;

fn llama_mul_mv_ext_pipelines(
    context: &MetalContext,
) -> Result<Option<LlamaMulMvExtPipelines>, MetalError> {
    let Some(library) = context.ggml_library.as_ref() else {
        return Ok(None);
    };
    Ok(Some(LlamaMulMvExtPipelines {
        q4k: llama_mul_mv_ext_group(context, library, "q4_K")?,
        q5k: llama_mul_mv_ext_group(context, library, "q5_K")?,
        q6k: llama_mul_mv_ext_group(context, library, "q6_K")?,
    }))
}

fn llama_mul_mv_ext_group(
    context: &MetalContext,
    library: &metal::Library,
    dtype: &str,
) -> Result<[ComputePipelineState; 4], MetalError> {
    let mut pipelines = Vec::with_capacity(4);
    for r1ptg in 2..=5 {
        let name = format!("kernel_mul_mv_ext_{dtype}_f32_r1_{r1ptg}");
        pipelines.push(ggml_pipeline_with_constants(
            context,
            library,
            &name,
            llama_mul_mv_ext_constants(),
            format!("{name}[nsg=2,nxpsg=8,ne12=1,r2=1,r3=1]"),
        )?);
    }
    Ok(pipelines
        .try_into()
        .unwrap_or_else(|_| unreachable!("four r1ptg pipelines are constructed")))
}

fn llama_mul_mv_ext_constants() -> FunctionConstantValues {
    let constants = FunctionConstantValues::new();
    for (value, offset) in [(2i16, 0u64), (8, 1), (1, 2), (1, 3), (1, 4)] {
        constants.set_constant_value_at_index(
            &value as *const i16 as *const std::ffi::c_void,
            MTLDataType::Short,
            FC_MUL_MV + offset,
        );
    }
    constants
}

impl LlamaMulMvExtPipelines {
    fn pipeline(
        &self,
        dtype: crate::gguf::GgmlType,
        r1ptg: usize,
    ) -> Option<&ComputePipelineStateRef> {
        let slot = r1ptg.checked_sub(2)?;
        let group = match dtype {
            crate::gguf::GgmlType::Q4_K => &self.q4k,
            crate::gguf::GgmlType::Q5_K => &self.q5k,
            crate::gguf::GgmlType::Q6_K => &self.q6k,
            _ => return None,
        };
        group.get(slot).map(AsRef::as_ref)
    }
}

fn ggml_matmul_pipeline(
    context: &MetalContext,
    name: &str,
    bounds: bool,
) -> Result<Option<ComputePipelineState>, MetalError> {
    let Some(library) = context.ggml_library.as_ref() else {
        return Ok(None);
    };
    let constants = FunctionConstantValues::new();
    let no_broadcast = false;
    constants.set_constant_value_at_index(
        &no_broadcast as *const bool as *const std::ffi::c_void,
        MTLDataType::Bool,
        700,
    );
    constants.set_constant_value_at_index(
        &bounds as *const bool as *const std::ffi::c_void,
        MTLDataType::Bool,
        701,
    );
    for index in 702..=705 {
        let one = 1i16;
        constants.set_constant_value_at_index(
            &one as *const i16 as *const std::ffi::c_void,
            MTLDataType::Short,
            index,
        );
    }
    let function = library
        .get_function(name, Some(constants))
        .map_err(|message| MetalError::Pipeline {
            name: format!("{name}[bounds={bounds}]"),
            message,
        })?;
    let pipeline = context
        .device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|message| MetalError::Pipeline {
            name: format!("{name}[bounds={bounds}]"),
            message,
        })?;
    Ok(Some(pipeline))
}

pub(super) const LLAMA_FA_NWG: i32 = 32;
pub(super) const LLAMA_FA_NCPSG: i32 = 32;
/// llama.cpp `OP_FLASH_ATTN_EXT_NQPSG`/`NCPSG` for the non-vec prefill kernel.
pub(crate) const LLAMA_FA_PREFILL_NQPTG: i32 = 8;
pub(crate) const LLAMA_FA_PREFILL_NCPSG: i32 = 32;
/// `ne00 >= 512 ? 8 : 4` in the pinned dispatch: DK128 runs four simdgroups.
pub(crate) const LLAMA_FA_PREFILL_NSG: i32 = 4;
const FC_FLASH_ATTN_EXT_PAD: u64 = 100;
const FC_FLASH_ATTN_EXT_BLK: u64 = 200;
const FC_FLASH_ATTN_EXT: u64 = 300;
const FC_FLASH_ATTN_EXT_VEC: u64 = 400;
const FC_FLASH_ATTN_EXT_VEC_REDUCE: u64 = 500;

fn llama_flash_attn_pipelines(
    context: &MetalContext,
) -> Result<Option<LlamaFlashAttnPipelines>, MetalError> {
    let Some(library) = context.ggml_library.as_ref() else {
        return Ok(None);
    };
    Ok(Some(LlamaFlashAttnPipelines {
        vec_ns128: llama_vec_group(context, library, 128, true)?,
        vec_ns256: llama_vec_group(context, library, 256, true)?,
        vec_unmasked_ns128: llama_vec_group(context, library, 128, false)?,
        vec_unmasked_ns256: llama_vec_group(context, library, 256, false)?,
        reduce: ggml_pipeline_with_constants(
            context,
            library,
            "kernel_flash_attn_ext_vec_reduce",
            llama_reduce_constants(),
            "kernel_flash_attn_ext_vec_reduce[dv=128,nwg=32]".into(),
        )?,
        pad: ggml_pipeline_with_constants(
            context,
            library,
            "kernel_flash_attn_ext_pad",
            llama_pad_constants(true),
            "kernel_flash_attn_ext_pad[ncpsg=32,mask=true]".into(),
        )?,
        pad_unmasked: ggml_pipeline_with_constants(
            context,
            library,
            "kernel_flash_attn_ext_pad",
            llama_pad_constants(false),
            "kernel_flash_attn_ext_pad[ncpsg=32,mask=false]".into(),
        )?,
        prefill: ggml_pipeline_with_constants(
            context,
            library,
            "kernel_flash_attn_ext_f16_dk128_dv128",
            llama_fa_prefill_constants(),
            "kernel_flash_attn_ext_f16_dk128_dv128[mask=true,kvpad=false,ns=128,nsg=4]".into(),
        )?,
        prefill_blk: ggml_pipeline_with_constants(
            context,
            library,
            "kernel_flash_attn_ext_blk",
            llama_fa_prefill_blk_constants(),
            "kernel_flash_attn_ext_blk[nqptg=8,ncpsg=32]".into(),
        )?,
    }))
}

fn llama_fa_prefill_constants() -> FunctionConstantValues {
    let constants = FunctionConstantValues::new();
    let true_flag = true;
    let false_flag = false;
    // has_mask=true; has_sinks, has_bias, has_scap, has_kvpad all false.
    constants.set_constant_value_at_index(
        &true_flag as *const bool as *const std::ffi::c_void,
        MTLDataType::Bool,
        FC_FLASH_ATTN_EXT,
    );
    for offset in [1u64, 2, 3, 4] {
        constants.set_constant_value_at_index(
            &false_flag as *const bool as *const std::ffi::c_void,
            MTLDataType::Bool,
            FC_FLASH_ATTN_EXT + offset,
        );
    }
    // bc_mask=false: the route only runs when the query count is 8-aligned.
    constants.set_constant_value_at_index(
        &false_flag as *const bool as *const std::ffi::c_void,
        MTLDataType::Bool,
        FC_FLASH_ATTN_EXT + 10,
    );
    let ns = 128i32;
    for (value, offset) in [(ns, 20u64), (ns, 21), (LLAMA_FA_PREFILL_NSG, 22)] {
        constants.set_constant_value_at_index(
            &value as *const i32 as *const std::ffi::c_void,
            MTLDataType::Int,
            FC_FLASH_ATTN_EXT + offset,
        );
    }
    constants
}

fn llama_fa_prefill_blk_constants() -> FunctionConstantValues {
    let constants = FunctionConstantValues::new();
    for (value, offset) in [
        (LLAMA_FA_PREFILL_NQPTG, 24u64),
        (LLAMA_FA_PREFILL_NCPSG, 25),
    ] {
        constants.set_constant_value_at_index(
            &value as *const i32 as *const std::ffi::c_void,
            MTLDataType::Int,
            FC_FLASH_ATTN_EXT_BLK + offset,
        );
    }
    constants
}

fn llama_vec_group(
    context: &MetalContext,
    library: &metal::Library,
    ns: i32,
    mask: bool,
) -> Result<[[ComputePipelineState; 2]; 3], MetalError> {
    let mut vec: [[Option<ComputePipelineState>; 2]; 3] = std::array::from_fn(|_| [None, None]);
    for (nsg_slot, nsg) in [1i32, 2, 4].into_iter().enumerate() {
        for (kvpad_slot, has_kvpad) in [false, true].into_iter().enumerate() {
            vec[nsg_slot][kvpad_slot] = Some(ggml_pipeline_with_constants(
                context,
                library,
                "kernel_flash_attn_ext_vec_f16_dk128_dv128",
                llama_vec_constants(nsg, has_kvpad, ns, mask),
                format!(
                    "kernel_flash_attn_ext_vec_f16_dk128_dv128[ns={ns},nsg={nsg},kvpad={has_kvpad},mask={mask}]"
                ),
            )?);
        }
    }
    Ok([
        [
            vec[0][0].take().expect("nsg1"),
            vec[0][1].take().expect("nsg1 kvpad"),
        ],
        [
            vec[1][0].take().expect("nsg2"),
            vec[1][1].take().expect("nsg2 kvpad"),
        ],
        [
            vec[2][0].take().expect("nsg4"),
            vec[2][1].take().expect("nsg4 kvpad"),
        ],
    ])
}

fn llama_vec_constants(nsg: i32, has_kvpad: bool, ns: i32, mask: bool) -> FunctionConstantValues {
    let constants = FunctionConstantValues::new();
    let false_flag = false;
    constants.set_constant_value_at_index(
        &mask as *const bool as *const std::ffi::c_void,
        MTLDataType::Bool,
        FC_FLASH_ATTN_EXT_VEC,
    );
    constants.set_constant_value_at_index(
        &false_flag as *const bool as *const std::ffi::c_void,
        MTLDataType::Bool,
        FC_FLASH_ATTN_EXT_VEC + 1,
    );
    constants.set_constant_value_at_index(
        &false_flag as *const bool as *const std::ffi::c_void,
        MTLDataType::Bool,
        FC_FLASH_ATTN_EXT_VEC + 2,
    );
    constants.set_constant_value_at_index(
        &false_flag as *const bool as *const std::ffi::c_void,
        MTLDataType::Bool,
        FC_FLASH_ATTN_EXT_VEC + 3,
    );
    constants.set_constant_value_at_index(
        &has_kvpad as *const bool as *const std::ffi::c_void,
        MTLDataType::Bool,
        FC_FLASH_ATTN_EXT_VEC + 4,
    );
    for (value, offset) in [(ns, 20u64), (ns, 21), (nsg, 22), (LLAMA_FA_NWG, 23)] {
        constants.set_constant_value_at_index(
            &value as *const i32 as *const std::ffi::c_void,
            MTLDataType::Int,
            FC_FLASH_ATTN_EXT_VEC + offset,
        );
    }
    constants
}

fn llama_reduce_constants() -> FunctionConstantValues {
    let constants = FunctionConstantValues::new();
    let dv = 128i32;
    constants.set_constant_value_at_index(
        &dv as *const i32 as *const std::ffi::c_void,
        MTLDataType::Int,
        FC_FLASH_ATTN_EXT_VEC_REDUCE,
    );
    constants.set_constant_value_at_index(
        &LLAMA_FA_NWG as *const i32 as *const std::ffi::c_void,
        MTLDataType::Int,
        FC_FLASH_ATTN_EXT_VEC_REDUCE + 1,
    );
    constants
}

fn llama_pad_constants(has_mask: bool) -> FunctionConstantValues {
    let constants = FunctionConstantValues::new();
    constants.set_constant_value_at_index(
        &has_mask as *const bool as *const std::ffi::c_void,
        MTLDataType::Bool,
        FC_FLASH_ATTN_EXT_PAD,
    );
    constants.set_constant_value_at_index(
        &LLAMA_FA_NCPSG as *const i32 as *const std::ffi::c_void,
        MTLDataType::Int,
        FC_FLASH_ATTN_EXT_PAD + 25,
    );
    constants
}

fn ggml_pipeline_with_constants(
    context: &MetalContext,
    library: &metal::Library,
    name: &str,
    constants: FunctionConstantValues,
    label: String,
) -> Result<ComputePipelineState, MetalError> {
    let function = library
        .get_function(name, Some(constants))
        .map_err(|message| MetalError::Pipeline {
            name: label.clone(),
            message,
        })?;
    context
        .device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|message| MetalError::Pipeline {
            name: label,
            message,
        })
}

impl LlamaFlashAttnPipelines {
    fn vec(
        &self,
        ns10: usize,
        nsg: usize,
        has_kvpad: bool,
        has_mask: bool,
    ) -> &ComputePipelineStateRef {
        let group = match (ns10, has_mask) {
            (128, true) => &self.vec_ns128,
            (256, true) => &self.vec_ns256,
            (128, false) => &self.vec_unmasked_ns128,
            (256, false) => &self.vec_unmasked_ns256,
            _ => panic!("unsupported llama FA ns10 {ns10}"),
        };
        let nsg_slot = match nsg {
            1 => 0,
            2 => 1,
            4 => 2,
            _ => panic!("unsupported llama FA NSG {nsg}"),
        };
        &group[nsg_slot][usize::from(has_kvpad)]
    }
}

pub(super) fn set_value<T>(encoder: &ComputeCommandEncoderRef, index: u64, value: &T) {
    encoder.set_bytes(
        index,
        std::mem::size_of::<T>() as u64,
        value as *const T as *const std::ffi::c_void,
    );
}

pub(super) fn dispatch_1d(encoder: &ComputeCommandEncoderRef, count: usize) {
    if count == 0 {
        return;
    }
    let width = count.min(256) as u64;
    encoder.dispatch_threads(MTLSize::new(count as u64, 1, 1), MTLSize::new(width, 1, 1));
}
