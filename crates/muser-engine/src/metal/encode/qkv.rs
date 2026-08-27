use metal::{ComputeCommandEncoderRef, MTLSize};

use super::{dispatch_1d, set_value, MetalKernels};
use crate::gguf::GgmlType;
use crate::metal::buffer::{GpuBuffer, GpuByteView, GpuBytes, GpuHalfBuffer};

impl MetalKernels {
    /// Exact two-pass M=16 W4A4 projection. Quantizing the activation once
    /// removes the former repetition in every N=32 output tile while keeping
    /// the same half boundary, E4M3 scale code, E2M1 code, integer dot, and
    /// scalar epilogue order as `muser_nvfp4_w4a4_m16_n32`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_nvfp4_w4a4_prequant_m16(
        &self,
        encoder: &ComputeCommandEncoderRef,
        packed: GpuByteView<'_>,
        scales: GpuByteView<'_>,
        scale2: f32,
        input_scale_inv: f32,
        input: &GpuBuffer,
        quantized: &GpuHalfBuffer,
        activation_scales: &GpuBytes,
        output: &GpuBuffer,
        n_in: usize,
        n_out: usize,
    ) {
        const COLUMNS: usize = 16;
        debug_assert!(n_in.is_multiple_of(64));
        debug_assert_eq!(packed.len(), n_out * n_in / 2);
        debug_assert_eq!(scales.len(), n_out * n_in / 16);
        debug_assert!(input.len() >= COLUMNS * n_in);
        debug_assert!(quantized.len() >= COLUMNS * n_in);
        debug_assert!(activation_scales.len() >= COLUMNS * n_in / 16 * 4);
        debug_assert!(output.len() >= COLUMNS * n_out);

        encoder.set_compute_pipeline_state(&self.cross_vendor_nvfp4_w4a4_quantize_m16);
        encoder.set_buffer(0, Some(input.metal()), 0);
        encoder.set_buffer(1, Some(quantized.metal()), 0);
        encoder.set_buffer(2, Some(activation_scales.metal()), 0);
        set_value(encoder, 3, &(n_in as u32));
        set_value(encoder, 4, &input_scale_inv);
        dispatch_1d(encoder, COLUMNS * n_in / 16);

        encoder.set_compute_pipeline_state(&self.cross_vendor_nvfp4_w4a4_prequant_m16_n32);
        encoder.set_buffer(0, Some(packed.metal()), packed.offset() as u64);
        encoder.set_buffer(1, Some(scales.metal()), scales.offset() as u64);
        encoder.set_buffer(2, Some(quantized.metal()), 0);
        encoder.set_buffer(3, Some(activation_scales.metal()), 0);
        encoder.set_buffer(4, Some(output.metal()), 0);
        set_value(
            encoder,
            5,
            &Nvfp4Args {
                n_in: n_in as u32,
                n_out: n_out as u32,
                col0: 0,
            },
        );
        set_value(encoder, 6, &scale2);
        set_value(encoder, 7, &input_scale_inv);
        encoder.dispatch_thread_groups(
            MTLSize::new(n_out.div_ceil(32) as u64, 1, 1),
            MTLSize::new(256, 1, 1),
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_f16_matmul(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weights: GpuByteView<'_>,
        input: &GpuBuffer,
        output: &GpuBuffer,
        n_in: usize,
        n_out: usize,
        columns: usize,
    ) {
        debug_assert!(n_in.is_multiple_of(8));
        debug_assert_eq!(weights.len(), n_in * n_out * std::mem::size_of::<u16>());
        debug_assert_eq!(input.len(), n_in * columns);
        debug_assert_eq!(output.len(), n_out * columns);
        let mut col0 = 0;
        while col0 < columns {
            let width = if columns - col0 >= 16 {
                16
            } else if columns - col0 >= 8 {
                8
            } else if columns - col0 >= 4 {
                4
            } else if columns - col0 >= 2 {
                2
            } else {
                1
            };
            self.bind(
                encoder,
                match width {
                    16 => "muser_f16_matvec_c16",
                    8 => "muser_f16_matvec_c8",
                    4 => "muser_f16_matvec_c4",
                    2 => "muser_f16_matvec_c2",
                    _ => "muser_f16_matvec_c1",
                },
            );
            encoder.set_buffer(0, Some(weights.metal()), weights.offset() as u64);
            encoder.set_buffer(1, Some(input.metal()), 0);
            encoder.set_buffer(2, Some(output.metal()), 0);
            set_value(
                encoder,
                3,
                &Nvfp4Args {
                    n_in: n_in as u32,
                    n_out: n_out as u32,
                    col0: col0 as u32,
                },
            );
            encoder
                .dispatch_thread_groups(MTLSize::new(n_out as u64, 1, 1), MTLSize::new(32, 1, 1));
            col0 += width;
        }
    }

    /// Native E2M1 projection. Width-1 is the decode kernel; wider calls use
    /// weight-stationary 2/4-column specializations and greedy decomposition,
    /// covering DFlash verification and bounded local prefill without ever
    /// expanding the packed weights.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_nvfp4_matmul(
        &self,
        encoder: &ComputeCommandEncoderRef,
        packed: GpuByteView<'_>,
        scales: GpuByteView<'_>,
        scale2: f32,
        input_scale_inv: Option<f32>,
        input: &GpuBuffer,
        output: &GpuBuffer,
        n_in: usize,
        n_out: usize,
        columns: usize,
    ) {
        debug_assert!(n_in.is_multiple_of(16));
        debug_assert_eq!(packed.len(), n_out * n_in / 2);
        debug_assert_eq!(scales.len(), n_out * n_in / 16);
        debug_assert_eq!(input.len(), n_in * columns);
        debug_assert_eq!(output.len(), n_out * columns);
        if input_scale_inv.is_none() {
            debug_assert!(n_in.is_multiple_of(256));
            encoder.set_compute_pipeline_state(&self.cross_vendor_nvfp4_a16_q8);
            encoder.set_buffer(0, Some(packed.metal()), packed.offset() as u64);
            encoder.set_buffer(1, Some(scales.metal()), scales.offset() as u64);
            encoder.set_buffer(2, Some(input.metal()), 0);
            encoder.set_buffer(3, Some(output.metal()), 0);
            set_value(encoder, 4, &(n_out as u32));
            set_value(encoder, 5, &(n_in as u32));
            set_value(encoder, 6, &(columns as u32));
            set_value(encoder, 7, &scale2);
            encoder.dispatch_thread_groups(
                MTLSize::new(n_out.div_ceil(8) as u64, columns as u64, 1),
                MTLSize::new(256, 1, 1),
            );
            return;
        }
        let mut col0 = 0;
        while col0 < columns {
            if columns - col0 >= 16
                && n_in.is_multiple_of(64)
                && std::env::var_os("MUSER_NO_M16_N32").is_none()
            {
                encoder.set_compute_pipeline_state(&self.cross_vendor_nvfp4_w4a4_m16_n32);
                encoder.set_buffer(0, Some(packed.metal()), packed.offset() as u64);
                encoder.set_buffer(1, Some(scales.metal()), scales.offset() as u64);
                encoder.set_buffer(2, Some(input.metal()), 0);
                encoder.set_buffer(3, Some(output.metal()), 0);
                set_value(
                    encoder,
                    4,
                    &Nvfp4Args {
                        n_in: n_in as u32,
                        n_out: n_out as u32,
                        col0: col0 as u32,
                    },
                );
                set_value(encoder, 5, &scale2);
                set_value(encoder, 6, &input_scale_inv.expect("W4A4 scale"));
                encoder.dispatch_thread_groups(
                    MTLSize::new(n_out.div_ceil(32) as u64, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
                col0 += 16;
                continue;
            }
            let width = if columns - col0 >= 16 {
                16
            } else if columns - col0 >= 8 {
                8
            } else if columns - col0 >= 4 {
                4
            } else if columns - col0 >= 2 {
                2
            } else {
                1
            };
            // The W4A4 integer contraction and its two scalar epilogue
            // multiplies are compiled in the no-fast-math library.
            encoder.set_compute_pipeline_state(self.cross_vendor_nvfp4_w4a4(width));
            encoder.set_buffer(0, Some(packed.metal()), packed.offset() as u64);
            encoder.set_buffer(1, Some(scales.metal()), scales.offset() as u64);
            encoder.set_buffer(2, Some(input.metal()), 0);
            encoder.set_buffer(3, Some(output.metal()), 0);
            set_value(
                encoder,
                4,
                &Nvfp4Args {
                    n_in: n_in as u32,
                    n_out: n_out as u32,
                    col0: col0 as u32,
                },
            );
            set_value(encoder, 5, &scale2);
            if let Some(value) = input_scale_inv {
                set_value(encoder, 6, &value);
            }
            encoder
                .dispatch_thread_groups(MTLSize::new(n_out as u64, 1, 1), MTLSize::new(32, 1, 1));
            col0 += width;
        }
    }

    /// Projection route for independent resident decode sequences. Exact
    /// multi-column kernels cover Q4_K/Q5_K; other pinned K-quants retain the
    /// one-row matvec arithmetic and are encoded as adjacent rows in the same
    /// command graph.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_quantized_decode_group(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weights: GpuByteView<'_>,
        input: &GpuBuffer,
        output: &GpuBuffer,
        dtype: GgmlType,
        n_in: usize,
        n_out: usize,
        rows: usize,
    ) {
        debug_assert!((1..=4).contains(&rows));
        if self.encode_cross_vendor_qk(encoder, weights, input, output, dtype, n_in, n_out, rows) {
            return;
        }
        if rows > 1
            && self.encode_exact_multicol_matvec(
                encoder, weights, input, output, dtype, n_in, n_out, rows,
            )
        {
            return;
        }
        let Some(pipeline) = self.ggml_matvec(dtype) else {
            self.encode_quantized_matmul(encoder, weights, input, output, dtype, n_in, n_out, rows);
            return;
        };
        let (block_bytes, rows_per_group) = match dtype {
            GgmlType::Q4_K => (144, 2),
            GgmlType::Q5_K => (176, 1),
            GgmlType::Q6_K => (210, 2),
            _ => unreachable!("ggml_matvec returned only for K-quant projections"),
        };
        let args = GgmlKargsMulMv::for_matmul(n_out, n_in, block_bytes, rows_per_group as i32);
        let input_stride = n_in * std::mem::size_of::<f32>();
        let output_stride = n_out * std::mem::size_of::<f32>();
        let simdgroups = 2usize;
        for row in 0..rows {
            encoder.set_compute_pipeline_state(pipeline);
            set_value(encoder, 0, &args);
            encoder.set_buffer(1, Some(weights.metal()), weights.offset() as u64);
            encoder.set_buffer(2, Some(input.metal()), (row * input_stride) as u64);
            encoder.set_buffer(3, Some(output.metal()), (row * output_stride) as u64);
            encoder.dispatch_thread_groups(
                MTLSize::new(n_out.div_ceil(rows_per_group * simdgroups) as u64, 1, 1),
                MTLSize::new(32, simdgroups as u64, 1),
            );
        }
    }

    pub fn encode_copy_row(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &GpuBuffer,
        output: &GpuBuffer,
        row: usize,
    ) {
        debug_assert!(input.len() >= (row + 1) * output.len());
        self.bind(encoder, "muser_copy_row_f32");
        encoder.set_buffer(0, Some(input.metal()), 0);
        encoder.set_buffer(1, Some(output.metal()), 0);
        set_value(encoder, 2, &(output.len() as u32));
        set_value(encoder, 3, &(row as u32));
        dispatch_1d(encoder, output.len());
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_cross_vendor_qk(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weights: GpuByteView<'_>,
        input: &GpuBuffer,
        output: &GpuBuffer,
        dtype: GgmlType,
        n_in: usize,
        n_out: usize,
        tokens: usize,
    ) -> bool {
        if std::env::var_os("MUSER_CROSS_VENDOR_QK").is_none() {
            return false;
        }
        let block_bytes = match dtype {
            GgmlType::Q4_K => 144,
            GgmlType::Q5_K => 176,
            GgmlType::Q6_K => 210,
            _ => return false,
        };
        debug_assert_eq!(weights.len(), n_out * (n_in / 256) * block_bytes);
        debug_assert_eq!(input.len(), n_in * tokens);
        debug_assert_eq!(output.len(), n_out * tokens);
        self.bind_cross_vendor(encoder, dtype);
        encoder.set_buffer(0, Some(weights.metal()), weights.offset() as u64);
        encoder.set_buffer(1, Some(input.metal()), 0);
        encoder.set_buffer(2, Some(output.metal()), 0);
        set_value(encoder, 3, &(n_out as u32));
        set_value(encoder, 4, &(n_in as u32));
        set_value(encoder, 5, &(tokens as u32));
        encoder.dispatch_thread_groups(
            MTLSize::new(n_out.div_ceil(8) as u64, tokens as u64, 1),
            MTLSize::new(256, 1, 1),
        );
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_embedding_q4k(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weights: GpuByteView<'_>,
        token_ids: &GpuByteView<'_>,
        output: &GpuBuffer,
        hidden_dim: usize,
        vocab_size: usize,
        tokens: usize,
    ) {
        if weights.len() == hidden_dim * vocab_size * std::mem::size_of::<u16>() {
            debug_assert_eq!(token_ids.len(), tokens * std::mem::size_of::<u32>());
            debug_assert_eq!(output.len(), hidden_dim * tokens);
            self.bind(encoder, "muser_embedding_f16");
            encoder.set_buffer(0, Some(weights.metal()), weights.offset() as u64);
            encoder.set_buffer(1, Some(token_ids.metal()), token_ids.offset() as u64);
            encoder.set_buffer(2, Some(output.metal()), 0);
            set_value(encoder, 3, &(hidden_dim as u32));
            set_value(encoder, 4, &(vocab_size as u32));
            set_value(encoder, 5, &(tokens as u32));
            dispatch_1d(encoder, hidden_dim * tokens);
            return;
        }
        let row_bytes = hidden_dim / 256 * 144;
        debug_assert_eq!(weights.len(), row_bytes * vocab_size);
        debug_assert_eq!(token_ids.len(), tokens * std::mem::size_of::<u32>());
        debug_assert_eq!(output.len(), hidden_dim * tokens);
        self.bind(encoder, "muser_embedding_q4k");
        encoder.set_buffer(0, Some(weights.metal()), weights.offset() as u64);
        encoder.set_buffer(1, Some(token_ids.metal()), token_ids.offset() as u64);
        encoder.set_buffer(2, Some(output.metal()), 0);
        set_value(encoder, 3, &(hidden_dim as u32));
        set_value(encoder, 4, &(vocab_size as u32));
        set_value(encoder, 5, &(tokens as u32));
        dispatch_1d(encoder, hidden_dim * tokens);
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_embedding_q4k_from_u32_buffer(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weights: GpuByteView<'_>,
        token_ids: &GpuBuffer,
        token_offset: usize,
        output: &GpuBuffer,
        hidden_dim: usize,
        vocab_size: usize,
    ) {
        if weights.len() == hidden_dim * vocab_size * std::mem::size_of::<u16>() {
            debug_assert!(token_offset < token_ids.len());
            debug_assert_eq!(output.len(), hidden_dim);
            self.bind(encoder, "muser_embedding_f16");
            encoder.set_buffer(0, Some(weights.metal()), weights.offset() as u64);
            encoder.set_buffer(1, Some(token_ids.metal()), (token_offset * 4) as u64);
            encoder.set_buffer(2, Some(output.metal()), 0);
            set_value(encoder, 3, &(hidden_dim as u32));
            set_value(encoder, 4, &(vocab_size as u32));
            set_value(encoder, 5, &1u32);
            dispatch_1d(encoder, hidden_dim);
            return;
        }
        let row_bytes = hidden_dim / 256 * 144;
        debug_assert_eq!(weights.len(), row_bytes * vocab_size);
        debug_assert!(token_offset < token_ids.len());
        debug_assert_eq!(output.len(), hidden_dim);
        self.bind(encoder, "muser_embedding_q4k");
        encoder.set_buffer(0, Some(weights.metal()), weights.offset() as u64);
        encoder.set_buffer(1, Some(token_ids.metal()), (token_offset * 4) as u64);
        encoder.set_buffer(2, Some(output.metal()), 0);
        set_value(encoder, 3, &(hidden_dim as u32));
        set_value(encoder, 4, &(vocab_size as u32));
        set_value(encoder, 5, &1u32);
        dispatch_1d(encoder, hidden_dim);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_quantized_matmul(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weights: GpuByteView<'_>,
        input: &GpuBuffer,
        output: &GpuBuffer,
        dtype: GgmlType,
        n_in: usize,
        n_out: usize,
        tokens: usize,
    ) {
        if self.encode_cross_vendor_qk(encoder, weights, input, output, dtype, n_in, n_out, tokens)
        {
            return;
        }
        if tokens == 1 {
            if let Some(pipeline) = self.ggml_matvec(dtype) {
                let (block_bytes, rows_per_group) = match dtype {
                    GgmlType::Q4_K => (144, 2),
                    GgmlType::Q5_K => (176, 1),
                    GgmlType::Q6_K => (210, 2),
                    _ => unreachable!("ggml_matvec returned only for K-quant projections"),
                };
                let args =
                    GgmlKargsMulMv::for_matmul(n_out, n_in, block_bytes, rows_per_group as i32);
                encoder.set_compute_pipeline_state(pipeline);
                set_value(encoder, 0, &args);
                encoder.set_buffer(1, Some(weights.metal()), weights.offset() as u64);
                encoder.set_buffer(2, Some(input.metal()), 0);
                encoder.set_buffer(3, Some(output.metal()), 0);
                let simdgroups = 2usize;
                encoder.dispatch_thread_groups(
                    MTLSize::new(n_out.div_ceil(rows_per_group * simdgroups) as u64, 1, 1),
                    MTLSize::new(32, simdgroups as u64, 1),
                );
                return;
            }
            let (name, row_bytes, groups, threads) = match dtype {
                GgmlType::Q4_K => (
                    "muser_matvec_q4k_4r2s",
                    n_in / 256 * 144,
                    n_out.div_ceil(8),
                    64,
                ),
                GgmlType::Q5_K => ("muser_matvec_q5k_4sg", n_in / 256 * 176, n_out, 128),
                other => panic!("unsupported Muse Metal projection dtype {other:?}"),
            };
            debug_assert_eq!(weights.len(), row_bytes * n_out);
            debug_assert_eq!(input.len(), n_in);
            debug_assert_eq!(output.len(), n_out);
            self.bind(encoder, name);
            encoder.set_buffer(0, Some(weights.metal()), weights.offset() as u64);
            encoder.set_buffer(1, Some(input.metal()), 0);
            encoder.set_buffer(2, Some(output.metal()), 0);
            set_value(encoder, 3, &(n_out as u32));
            set_value(encoder, 4, &(n_in as u32));
            encoder.dispatch_thread_groups(
                MTLSize::new(groups as u64, 1, 1),
                MTLSize::new(threads, 1, 1),
            );
            return;
        }
        // Match the source-pinned llama.cpp Metal dispatch boundary exactly:
        // K-quant projections with four through eight activation rows use
        // `mul_mv_ext`, with a token-count-specific number of rows per
        // threadgroup. This changes the floating-point reduction order, so
        // substituting repeated decode GEMVs here breaks embedding/logprob
        // numerical parity even when every other layer is identical.
        if (4..=8).contains(&tokens) {
            let r1ptg = match tokens {
                5 => 5,
                6 => 3,
                4 | 7 | 8 => 4,
                _ => unreachable!("small-batch llama projection range"),
            };
            if let Some(pipeline) = self.llama_mul_mv_ext(dtype, r1ptg) {
                let block_bytes = match dtype {
                    GgmlType::Q4_K => 144,
                    GgmlType::Q5_K => 176,
                    GgmlType::Q6_K => 210,
                    _ => unreachable!("mul_mv_ext is registered only for K-quants"),
                };
                let args = GgmlKargsMulMvExt::for_matmul(n_out, n_in, tokens, block_bytes);
                encoder.set_compute_pipeline_state(pipeline);
                set_value(encoder, 0, &args);
                encoder.set_buffer(1, Some(weights.metal()), weights.offset() as u64);
                encoder.set_buffer(2, Some(input.metal()), 0);
                encoder.set_buffer(3, Some(output.metal()), 0);
                encoder.dispatch_thread_groups(
                    MTLSize::new(n_out.div_ceil(8) as u64, tokens.div_ceil(r1ptg) as u64, 1),
                    MTLSize::new(32, 2, 1),
                );
                return;
            }
        }
        // Default-off multi-column verify route (`MUSER_MULTI_COL_VERIFY=1`
        // for the bitwise-exact dtypes, `=all` to include Q6_K). One weight
        // block load per row feeds every activation column, so weight traffic
        // no longer scales with the block length. Returns false when the route
        // is disabled or the shape/dtype has no specialization, which leaves
        // the per-token loop below in charge.
        if (2..=super::multicol::MAX_COLUMNS).contains(&tokens)
            && self
                .encode_multicol_matvec(encoder, weights, input, output, dtype, n_in, n_out, tokens)
        {
            return;
        }
        // llama.cpp keeps K-quant batches two and three on the decode GEMV;
        // the four-through-eight source-pinned route returned above.
        if tokens <= 8 {
            if let Some(pipeline) = self.ggml_matvec(dtype) {
                let (block_bytes, rows_per_group) = match dtype {
                    GgmlType::Q4_K => (144, 2),
                    GgmlType::Q5_K => (176, 1),
                    GgmlType::Q6_K => (210, 2),
                    _ => unreachable!("ggml_matvec returned only for K-quant projections"),
                };
                let args =
                    GgmlKargsMulMv::for_matmul(n_out, n_in, block_bytes, rows_per_group as i32);
                let input_stride = n_in * std::mem::size_of::<f32>();
                let output_stride = n_out * std::mem::size_of::<f32>();
                let simdgroups = 2usize;
                for token in 0..tokens {
                    encoder.set_compute_pipeline_state(pipeline);
                    set_value(encoder, 0, &args);
                    encoder.set_buffer(1, Some(weights.metal()), weights.offset() as u64);
                    encoder.set_buffer(2, Some(input.metal()), (token * input_stride) as u64);
                    encoder.set_buffer(3, Some(output.metal()), (token * output_stride) as u64);
                    encoder.dispatch_thread_groups(
                        MTLSize::new(n_out.div_ceil(rows_per_group * simdgroups) as u64, 1, 1),
                        MTLSize::new(32, simdgroups as u64, 1),
                    );
                }
                return;
            }
        }
        // M=16 K-quant batch tile (L series): the DFlash verify and draft
        // block forwards are exactly 16-row batches. The n32 tile is
        // weight-stationary with a 6 KiB threadgroup footprint, escaping the
        // retained SGM tile's occupancy bound at these shapes. Same accepted
        // half-staged simdgroup-matrix arithmetic family; exactness is gated
        // by lossless token equality.
        if tokens == 16
            && n_in.is_multiple_of(256)
            && n_out.is_multiple_of(32)
            && std::env::var_os("MUSER_NO_M16_N32").is_none()
        {
            let name = match dtype {
                GgmlType::Q4_K => "m16_q4k_n32",
                GgmlType::Q5_K => "m16_q5k_n32",
                GgmlType::Q6_K => "m16_q6k_n32",
                _ => unreachable!("encode_quantized_matmul dispatches K-quants"),
            };
            self.bind(encoder, name);
            encoder.set_buffer(0, Some(weights.metal()), weights.offset() as u64);
            encoder.set_buffer(1, Some(input.metal()), 0);
            encoder.set_buffer(2, Some(output.metal()), 0);
            set_value(encoder, 3, &(n_in as u32));
            set_value(encoder, 4, &(n_out as u32));
            set_value(encoder, 5, &(tokens as u32));
            encoder.set_threadgroup_memory_length(0, 6144);
            encoder.dispatch_thread_groups(
                MTLSize::new(n_out.div_ceil(32) as u64, 1, 1),
                MTLSize::new(128, 1, 1),
            );
            return;
        }
        // Ferrite's accepted high-occupancy Q4_K prefill kernel. Unlike the
        // excluded QKVG/FFN consolidation experiments, this preserves every
        // projection boundary and only replaces the aligned GEMM primitive.
        if dtype == GgmlType::Q4_K
            && n_in.is_multiple_of(32)
            && n_out.is_multiple_of(64)
            && tokens.is_multiple_of(16)
        {
            self.bind(encoder, "matmul_q4k_batch_sgm_aligned");
            encoder.set_buffer(0, Some(weights.metal()), weights.offset() as u64);
            encoder.set_buffer(1, Some(input.metal()), 0);
            encoder.set_buffer(2, Some(output.metal()), 0);
            set_value(encoder, 3, &(n_in as u32));
            set_value(encoder, 4, &(n_out as u32));
            set_value(encoder, 5, &(tokens as u32));
            encoder.set_threadgroup_memory_length(0, 6144);
            encoder.dispatch_thread_groups(
                MTLSize::new(tokens.div_ceil(32) as u64, n_out.div_ceil(64) as u64, 1),
                MTLSize::new(128, 1, 1),
            );
            return;
        }
        let bounds = !n_out.is_multiple_of(64) || !tokens.is_multiple_of(32);
        if let Some(pipeline) = self.ggml_matmul(dtype, bounds) {
            let block_bytes = match dtype {
                GgmlType::Q4_K => 144,
                GgmlType::Q5_K => 176,
                GgmlType::Q6_K => 210,
                _ => unreachable!("ggml_matmul returned only for K-quant projections"),
            };
            let args = GgmlKargsMulMm::for_matmul(n_out, n_in, tokens, block_bytes);
            encoder.set_compute_pipeline_state(pipeline);
            set_value(encoder, 0, &args);
            encoder.set_buffer(1, Some(weights.metal()), weights.offset() as u64);
            encoder.set_buffer(2, Some(input.metal()), 0);
            encoder.set_buffer(3, Some(output.metal()), 0);
            encoder.set_threadgroup_memory_length(0, if bounds { 8192 } else { 6144 });
            encoder.dispatch_thread_groups(
                MTLSize::new(tokens.div_ceil(32) as u64, n_out.div_ceil(64) as u64, 1),
                MTLSize::new(32, 4, 1),
            );
            return;
        }
        let (name, row_bytes) = match dtype {
            GgmlType::Q4_K => ("muser_matmul_q4k", n_in / 256 * 144),
            GgmlType::Q5_K => ("muser_matmul_q5k", n_in / 256 * 176),
            other => panic!("unsupported Muse Metal projection dtype {other:?}"),
        };
        debug_assert!(n_in.is_multiple_of(256));
        debug_assert_eq!(weights.len(), row_bytes * n_out);
        debug_assert_eq!(input.len(), n_in * tokens);
        debug_assert_eq!(output.len(), n_out * tokens);
        self.bind(encoder, name);
        encoder.set_buffer(0, Some(weights.metal()), weights.offset() as u64);
        encoder.set_buffer(1, Some(input.metal()), 0);
        encoder.set_buffer(2, Some(output.metal()), 0);
        set_value(encoder, 3, &(n_in as u32));
        set_value(encoder, 4, &(n_out as u32));
        set_value(encoder, 5, &(tokens as u32));
        dispatch_1d(encoder, n_out * tokens);
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Nvfp4Args {
    n_in: u32,
    n_out: u32,
    col0: u32,
}

const _: () = assert!(std::mem::size_of::<Nvfp4Args>() == 12);

#[cfg(test)]
mod nvfp4_tests {
    use super::*;
    use crate::gguf::GgufFile;
    use crate::metal::buffer::GpuBytes;
    use crate::metal::context::MetalContext;
    use crate::quant::dequant_nvfp4_row;
    use crate::weights::MuseWeights;
    use sha2::{Digest, Sha256};
    use std::path::Path;
    use std::time::Duration;

    fn sha256(bytes: &[u8]) -> String {
        Sha256::digest(bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }

    #[test]
    fn metal_nvfp4_dequant_is_bit_exact_for_every_finite_e4m3fn_scale() {
        let context = MetalContext::new().expect("Metal context");
        let kernels = MetalKernels::new(&context).expect("Muse primitive pipelines");
        let scale2 = 1.0f32 / 448.0f32;
        let finite_scales: Vec<u8> = (0u8..=u8::MAX)
            .filter(|byte| !matches!(byte, 0x7f | 0xff))
            .collect();
        let one_group = [0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe];
        let mut packed = Vec::with_capacity(finite_scales.len() * one_group.len());
        let mut expected = Vec::with_capacity(finite_scales.len() * 16);
        for &scale in &finite_scales {
            packed.extend_from_slice(&one_group);
            let mut decoded = [0.0f32; 16];
            dequant_nvfp4_row(&one_group, &[scale], scale2, &mut decoded);
            expected.extend_from_slice(&decoded);
        }
        let packed = GpuBytes::from_bytes(&context, &packed).unwrap();
        let scales = GpuBytes::from_bytes(&context, &finite_scales).unwrap();
        let output = GpuBuffer::zeros(&context, expected.len()).unwrap();
        let command = context.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        kernels.bind(encoder, "muser_nvfp4_dequant_fixture");
        encoder.set_buffer(0, Some(packed.metal()), 0);
        encoder.set_buffer(1, Some(scales.metal()), 0);
        encoder.set_buffer(2, Some(output.metal()), 0);
        set_value(encoder, 3, &(expected.len() as u32));
        set_value(encoder, 4, &scale2);
        dispatch_1d(encoder, expected.len());
        encoder.end_encoding();
        command.commit();
        context
            .wait_for_completion(command, Duration::from_secs(30))
            .expect("NVFP4 fixture completion");
        for (index, (&actual, &expected)) in output.as_slice().iter().zip(&expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "NVFP4 dequant mismatch at fixture element {index}"
            );
        }
    }

    #[test]
    fn native_nvfp4_a16_q8_single_and_batched_matvec_match_oracle_bits() {
        let context = MetalContext::new().expect("Metal context");
        let kernels = MetalKernels::new(&context).expect("Muse primitive pipelines");
        let scale2 = 0.25f32;
        let n_in = 256;
        let finite_scales: Vec<u8> = (0u8..=u8::MAX)
            .filter(|byte| !matches!(byte, 0x7f | 0xff))
            .collect();
        let n_out = 11; // crosses the eight-row shared-Q8 threadgroup boundary
        let columns = 7;
        let one_group = [0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe];
        let mut packed_bytes = Vec::with_capacity(n_out * n_in / 2);
        let mut scale_bytes = Vec::with_capacity(n_out * n_in / 16);
        for row in 0..n_out {
            for group in 0..n_in / 16 {
                packed_bytes.extend_from_slice(&one_group);
                scale_bytes.push(finite_scales[(row * 29 + group * 17) % finite_scales.len()]);
            }
        }
        let mut activations = vec![0.0f32; columns * n_in];
        for column in 0..columns {
            for element in 0..n_in {
                activations[column * n_in + element] =
                    ((column * 19 + element * 7 + 3) as f32 - 901.0) * 0.003125 + 0.000_01;
            }
            // Exercise signed-first-max tie handling after the F16 boundary.
            activations[column * n_in] = 2.0;
            activations[column * n_in + 1] = -2.0;
        }
        let mut expected = vec![0.0f32; columns * n_out];
        for row in 0..n_out {
            for column in 0..columns {
                expected[column * n_out + row] = crate::quant::dot_nvfp4_a16_q8_f32(
                    &packed_bytes[row * n_in / 2..(row + 1) * n_in / 2],
                    &scale_bytes[row * n_in / 16..(row + 1) * n_in / 16],
                    scale2,
                    &activations[column * n_in..(column + 1) * n_in],
                );
            }
        }
        let packed = GpuBytes::from_bytes(&context, &packed_bytes).unwrap();
        let scales = GpuBytes::from_bytes(&context, &scale_bytes).unwrap();
        let input = GpuBuffer::from_f32(&context, &activations).unwrap();
        let output = GpuBuffer::zeros(&context, expected.len()).unwrap();
        let command = context.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        kernels.encode_nvfp4_matmul(
            encoder,
            packed.view(0, packed.len()).unwrap(),
            scales.view(0, scales.len()).unwrap(),
            scale2,
            None,
            &input,
            &output,
            n_in,
            n_out,
            columns,
        );
        encoder.end_encoding();
        command.commit();
        context
            .wait_for_completion(command, Duration::from_secs(30))
            .expect("native NVFP4 matvec completion");
        for (index, (&actual, &expected)) in output.as_slice().iter().zip(&expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "NVFP4 A16 Q8 matvec mismatch at fixture element {index}: {actual} != {expected}"
            );
        }
        let output_f16: Vec<u8> = output
            .as_slice()
            .iter()
            .flat_map(|&value| half::f16::from_f32(value).to_bits().to_le_bytes())
            .collect();
        let mut q8_bytes = Vec::with_capacity(columns * n_in);
        let mut q8_scale_bytes = Vec::with_capacity(columns * std::mem::size_of::<f32>());
        for column in 0..columns {
            let q8 = crate::quant::quantize_nvfp4_q8_block(
                &activations[column * n_in..(column + 1) * n_in],
            );
            q8_bytes.extend(q8.qs.iter().map(|&value| value as u8));
            q8_scale_bytes.extend_from_slice(&q8.d.to_le_bytes());
        }
        println!(
            "{}",
            serde_json::json!({
                "schema": "muser.mac-exact-fp4-a16-q8-fixture.v1",
                "shape": [columns, n_out, n_in],
                "mismatches": 0,
                "q8_sha256": sha256(&q8_bytes),
                "q8_scale_sha256": sha256(&q8_scale_bytes),
                "output_sha256": sha256(&output_f16),
            })
        );
    }

    #[test]
    fn native_w4a4_single_and_batched_matvec_match_oracle_bits() {
        let context = MetalContext::new().expect("Metal context");
        let kernels = MetalKernels::new(&context).expect("Muse primitive pipelines");
        let weight_scale2 = 0.25f32;
        let input_scale_inv = 43.75f32;
        let n_in = 16;
        let finite_scales: Vec<u8> = (0u8..=u8::MAX)
            .filter(|byte| !matches!(byte, 0x7f | 0xff))
            .collect();
        let n_out = finite_scales.len();
        let columns = 7;
        let mut packed_bytes = Vec::with_capacity(n_out * 8);
        for _ in 0..n_out {
            packed_bytes.extend_from_slice(&[0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe]);
        }
        let mut activations = vec![0.0f32; columns * n_in];
        for column in 0..columns {
            for element in 0..n_in {
                activations[column * n_in + element] =
                    ((column * 23 + element * 11 + 5) as f32 - 67.0) * 0.03125 + 0.000_01;
            }
        }
        let mut expected = vec![0.0f32; columns * n_out];
        for row in 0..n_out {
            for column in 0..columns {
                expected[column * n_out + row] = crate::quant::dot_nvfp4_w4a4_f32(
                    &packed_bytes[row * 8..(row + 1) * 8],
                    &finite_scales[row..row + 1],
                    weight_scale2,
                    input_scale_inv,
                    &activations[column * n_in..(column + 1) * n_in],
                );
            }
        }
        let packed = GpuBytes::from_bytes(&context, &packed_bytes).unwrap();
        let scales = GpuBytes::from_bytes(&context, &finite_scales).unwrap();
        let input = GpuBuffer::from_f32(&context, &activations).unwrap();
        let output = GpuBuffer::zeros(&context, expected.len()).unwrap();
        let command = context.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        kernels.encode_nvfp4_matmul(
            encoder,
            packed.view(0, packed.len()).unwrap(),
            scales.view(0, scales.len()).unwrap(),
            weight_scale2,
            Some(input_scale_inv),
            &input,
            &output,
            n_in,
            n_out,
            columns,
        );
        encoder.end_encoding();
        command.commit();
        context
            .wait_for_completion(command, Duration::from_secs(30))
            .expect("native W4A4 matvec completion");
        for (index, (&actual, &expected)) in output.as_slice().iter().zip(&expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "W4A4 matvec mismatch at fixture element {index}: {actual} != {expected}"
            );
        }
    }

    #[test]
    fn native_w4a4_m16_n32_matches_oracle_bits() {
        let context = MetalContext::new().expect("Metal context");
        let kernels = MetalKernels::new(&context).expect("Muse primitive pipelines");
        let weight_scale2 = 0.25f32;
        let input_scale_inv = 43.75f32;
        let n_in = 256;
        let n_out = 37;
        let columns = 16;
        let packed_per_row = n_in / 2;
        let scales_per_row = n_in / 16;
        let mut packed_bytes = vec![0u8; n_out * packed_per_row];
        let mut scale_bytes = vec![0u8; n_out * scales_per_row];
        for row in 0..n_out {
            for (index, byte) in packed_bytes[row * packed_per_row..(row + 1) * packed_per_row]
                .iter_mut()
                .enumerate()
            {
                *byte = ((row * 29 + index * 17 + 3) & 0xff) as u8;
            }
            for (index, byte) in scale_bytes[row * scales_per_row..(row + 1) * scales_per_row]
                .iter_mut()
                .enumerate()
            {
                *byte = ((row * 13 + index * 7 + 1) % 0x7f) as u8;
            }
        }
        let mut activations = vec![0.0f32; columns * n_in];
        for column in 0..columns {
            for element in 0..n_in {
                activations[column * n_in + element] =
                    ((column * 23 + element * 11 + 5) as f32 - 1_407.0) * 0.001_953_125 + 0.000_01;
            }
        }
        let mut expected = vec![0.0f32; columns * n_out];
        for row in 0..n_out {
            for column in 0..columns {
                expected[column * n_out + row] = crate::quant::dot_nvfp4_w4a4_f32(
                    &packed_bytes[row * packed_per_row..(row + 1) * packed_per_row],
                    &scale_bytes[row * scales_per_row..(row + 1) * scales_per_row],
                    weight_scale2,
                    input_scale_inv,
                    &activations[column * n_in..(column + 1) * n_in],
                );
            }
        }
        let packed = GpuBytes::from_bytes(&context, &packed_bytes).unwrap();
        let scales = GpuBytes::from_bytes(&context, &scale_bytes).unwrap();
        let input = GpuBuffer::from_f32(&context, &activations).unwrap();
        let quantized = GpuHalfBuffer::zeros(&context, columns * n_in).unwrap();
        let activation_scales =
            GpuBytes::zeros(&context, columns * n_in / 16 * std::mem::size_of::<i32>()).unwrap();
        let output = GpuBuffer::zeros(&context, expected.len()).unwrap();
        let command = context.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        kernels.encode_nvfp4_w4a4_prequant_m16(
            encoder,
            packed.view(0, packed.len()).unwrap(),
            scales.view(0, scales.len()).unwrap(),
            weight_scale2,
            input_scale_inv,
            &input,
            &quantized,
            &activation_scales,
            &output,
            n_in,
            n_out,
        );
        encoder.end_encoding();
        command.commit();
        context
            .wait_for_completion(command, Duration::from_secs(30))
            .expect("native W4A4 M16 completion");
        for (index, (&actual, &expected)) in output.as_slice().iter().zip(&expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "W4A4 M16 mismatch at fixture element {index}: {actual} != {expected}"
            );
        }
    }

    #[test]
    #[ignore]
    fn diagnostic_native_w4a4_m16_n32_timing() {
        let context = MetalContext::new().expect("Metal context");
        let kernels = MetalKernels::new(&context).expect("Muse primitive pipelines");
        let n_in = 6_656usize;
        let n_out = 4_096usize;
        let columns = 16usize;
        let packed_len = n_out * n_in / 2;
        let scales_len = n_out * n_in / 16;
        let packed_bytes: Vec<u8> = (0..packed_len)
            .map(|index| (index.wrapping_mul(17).wrapping_add(3) & 0xff) as u8)
            .collect();
        let scale_bytes: Vec<u8> = (0..scales_len)
            .map(|index| (index.wrapping_mul(7).wrapping_add(1) % 0x7f) as u8)
            .collect();
        let activations: Vec<f32> = (0..columns * n_in)
            .map(|index| ((index.wrapping_mul(11) % 2_047) as f32 - 1_023.0) * 0.001_953_125)
            .collect();
        let packed = GpuBytes::from_bytes(&context, &packed_bytes).unwrap();
        let scales = GpuBytes::from_bytes(&context, &scale_bytes).unwrap();
        let input = GpuBuffer::from_f32(&context, &activations).unwrap();
        let legacy_output = GpuBuffer::zeros(&context, columns * n_out).unwrap();
        let tile_output = GpuBuffer::zeros(&context, columns * n_out).unwrap();
        let weight_scale2 = 0.25f32;
        let input_scale_inv = 43.75f32;
        let dispatch = |tile: bool, output: &GpuBuffer| {
            let command = context.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            if tile {
                kernels.encode_nvfp4_matmul(
                    encoder,
                    packed.view(0, packed.len()).unwrap(),
                    scales.view(0, scales.len()).unwrap(),
                    weight_scale2,
                    Some(input_scale_inv),
                    &input,
                    output,
                    n_in,
                    n_out,
                    columns,
                );
            } else {
                encoder.set_compute_pipeline_state(&kernels.cross_vendor_nvfp4_w4a4_c16);
                encoder.set_buffer(0, Some(packed.metal()), 0);
                encoder.set_buffer(1, Some(scales.metal()), 0);
                encoder.set_buffer(2, Some(input.metal()), 0);
                encoder.set_buffer(3, Some(output.metal()), 0);
                set_value(
                    encoder,
                    4,
                    &Nvfp4Args {
                        n_in: n_in as u32,
                        n_out: n_out as u32,
                        col0: 0,
                    },
                );
                set_value(encoder, 5, &weight_scale2);
                set_value(encoder, 6, &input_scale_inv);
                encoder.dispatch_thread_groups(
                    MTLSize::new(n_out as u64, 1, 1),
                    MTLSize::new(32, 1, 1),
                );
            }
            encoder.end_encoding();
            command.commit();
            context
                .wait_for_completion(command, Duration::from_secs(30))
                .expect("W4A4 diagnostic completion");
        };
        dispatch(false, &legacy_output);
        dispatch(true, &tile_output);
        assert_eq!(legacy_output.as_slice(), tile_output.as_slice());
        let repetitions = 20;
        let legacy_start = std::time::Instant::now();
        for _ in 0..repetitions {
            dispatch(false, &legacy_output);
        }
        let legacy = legacy_start.elapsed();
        let tile_start = std::time::Instant::now();
        for _ in 0..repetitions {
            dispatch(true, &tile_output);
        }
        let tile = tile_start.elapsed();
        println!(
            "{}",
            serde_json::json!({
                "schema": "muser.mac-nvfp4-w4a4-m16-n32-diagnostic.v1",
                "shape": [columns, n_out, n_in],
                "repetitions": repetitions,
                "legacy_ns": legacy.as_nanos(),
                "tile_ns": tile.as_nanos(),
                "speedup": legacy.as_secs_f64() / tile.as_secs_f64(),
            })
        );
    }

    #[test]
    fn retained_real_layer1_qkv_integer_fixture() {
        let (Ok(model), Ok(input_path), Ok(output_path)) = (
            std::env::var("MUSER_NVFP4_QKV_MODEL"),
            std::env::var("MUSER_NVFP4_QKV_INPUT"),
            std::env::var("MUSER_NVFP4_QKV_OUTPUT"),
        ) else {
            eprintln!(
                "skipped: set MUSER_NVFP4_QKV_MODEL, _INPUT, and _OUTPUT for the retained fixture"
            );
            return;
        };
        let token_indices: Vec<usize> = std::env::var("MUSER_NVFP4_QKV_TOKENS")
            .unwrap_or_else(|_| "7,12,22".into())
            .split(',')
            .map(|value| value.parse().expect("numeric retained token index"))
            .collect();
        assert!(
            !token_indices.is_empty(),
            "retained token selection is empty"
        );

        let model = Path::new(&model);
        let gguf = GgufFile::parse_path(model).expect("parse retained NVFP4 model");
        let weights = MuseWeights::open(model, &gguf).expect("map retained NVFP4 model");
        let input_bytes = std::fs::read(&input_path).expect("read retained layer-1 norm input");
        assert!(
            input_bytes.len().is_multiple_of(4),
            "retained input is not f32le"
        );
        let all_input: Vec<f32> = input_bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        const HIDDEN: usize = 6656;
        assert!(all_input.len().is_multiple_of(HIDDEN));
        let input_tokens = all_input.len() / HIDDEN;
        let mut selected = Vec::with_capacity(token_indices.len() * HIDDEN);
        for &token in &token_indices {
            assert!(
                token < input_tokens,
                "retained token {token} is out of range"
            );
            selected.extend_from_slice(&all_input[token * HIDDEN..(token + 1) * HIDDEN]);
        }

        let output_path = Path::new(&output_path);
        assert!(
            !output_path.exists(),
            "refusing to replace retained QKV fixture output"
        );
        std::fs::create_dir_all(output_path).expect("create retained QKV fixture output");
        let context = MetalContext::new().expect("Metal context");
        let kernels = MetalKernels::new(&context).expect("Muse primitive pipelines");
        let mapped =
            GpuBytes::from_mmap(&context, weights.mapped_file()).expect("map GGUF to Metal");
        let input = GpuBuffer::from_f32(&context, &selected).expect("retained input buffer");
        let mut outputs = Vec::new();
        let command = context.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        for (suffix, width) in [("q", 4096usize), ("k", 256), ("v", 256)] {
            let name = format!("blk.1.attn_{suffix}.weight");
            let layout = weights.layout(&name).expect("retained QKV tensor layout");
            assert_eq!((layout.n_in, layout.n_out), (HIDDEN, width));
            assert_eq!(layout.dtype, GgmlType::NVFP4_E2M1);
            let packed = mapped
                .view(layout.file_offset, layout.byte_len)
                .expect("retained packed QKV view");
            let scales = mapped
                .view(
                    layout.nvfp4_scale_offset.expect("retained QKV scales"),
                    layout.nvfp4_scale_len,
                )
                .expect("retained QKV scale view");
            let output = GpuBuffer::zeros(&context, token_indices.len() * width)
                .expect("retained QKV output buffer");
            kernels.encode_nvfp4_matmul(
                encoder,
                packed,
                scales,
                layout.nvfp4_scale2,
                layout.nvfp4_input_scale_inv,
                &input,
                &output,
                HIDDEN,
                width,
                token_indices.len(),
            );
            outputs.push((suffix, output));
        }
        encoder.end_encoding();
        command.commit();
        context
            .wait_for_completion(command, Duration::from_secs(90))
            .expect("retained integer QKV completion");

        let mut receipts = Vec::new();
        for (suffix, output) in outputs {
            let bytes: Vec<u8> = output
                .as_slice()
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            let path = output_path.join(format!("{suffix}.f32le"));
            std::fs::write(&path, &bytes).expect("write retained QKV output");
            receipts.push(serde_json::json!({
                "projection": suffix,
                "path": path,
                "bytes": bytes.len(),
                "sha256": sha256(&bytes),
            }));
        }
        println!(
            "{}",
            serde_json::json!({
                "schema": "muser.mac-nvfp4-layer1-qkv-integer.v2",
                "model": model,
                "input": input_path,
                "token_indices": token_indices,
                "activation_precision": if weights
                    .layout("blk.1.attn_q.weight")
                    .expect("retained Q layout")
                    .nvfp4_input_scale_inv
                    .is_some()
                { "nvfp4" } else { "f16-weight-only-q8k-exact" },
                "outputs": receipts,
            })
        );
    }

    #[test]
    fn retained_real_qk_norm_scale_bits() {
        let Ok(model) = std::env::var("MUSER_NVFP4_QKV_MODEL") else {
            eprintln!("skipped: set MUSER_NVFP4_QKV_MODEL for retained QK scale evidence");
            return;
        };
        let model = Path::new(&model);
        let gguf = GgufFile::parse_path(model).expect("parse retained NVFP4 model");
        let weights = MuseWeights::open(model, &gguf).expect("map retained NVFP4 model");
        let q = weights
            .f32_vec("blk.0.attn_q_norm.weight")
            .expect("read Q norm scale");
        let k = weights
            .f32_vec("blk.0.attn_k_norm.weight")
            .expect("read K norm scale");
        assert!(q.iter().all(|value| value.to_bits() == q[0].to_bits()));
        assert!(k.iter().all(|value| value.to_bits() == k[0].to_bits()));
        println!(
            "{}",
            serde_json::json!({
                "schema": "muser.retained-qk-norm-scale.v1",
                "q_value": q[0],
                "q_bits": format!("0x{:08x}", q[0].to_bits()),
                "k_value": k[0],
                "k_bits": format!("0x{:08x}", k[0].to_bits()),
            })
        );
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GgmlKargsMulMm {
    ne00: i32,
    ne02: i32,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne12: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne0: i32,
    ne1: i32,
    r2: i16,
    r3: i16,
}

const _: () = assert!(std::mem::size_of::<GgmlKargsMulMm>() == 88);

impl GgmlKargsMulMm {
    fn for_matmul(rows: usize, cols: usize, tokens: usize, block_bytes: usize) -> Self {
        let row_bytes = (cols / 256 * block_bytes) as u64;
        let weights_bytes = row_bytes * rows as u64;
        let input_row_bytes = (cols * std::mem::size_of::<f32>()) as u64;
        let input_bytes = input_row_bytes * tokens as u64;
        Self {
            ne00: cols as i32,
            ne02: 1,
            nb01: row_bytes,
            nb02: weights_bytes,
            nb03: weights_bytes,
            ne12: 1,
            nb10: std::mem::size_of::<f32>() as u64,
            nb11: input_row_bytes,
            nb12: input_bytes,
            nb13: input_bytes,
            ne0: rows as i32,
            ne1: tokens as i32,
            r2: 1,
            r3: 1,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GgmlKargsMulMv {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne10: i32,
    ne11: i32,
    ne12: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne0: i32,
    ne1: i32,
    nr0: i32,
    r2: i16,
    r3: i16,
}

const _: () = assert!(std::mem::size_of::<GgmlKargsMulMv>() == 112);

impl GgmlKargsMulMv {
    fn for_matmul(rows: usize, cols: usize, block_bytes: usize, nr0: i32) -> Self {
        let row_bytes = (cols / 256 * block_bytes) as u64;
        Self {
            ne00: cols as i32,
            ne01: rows as i32,
            ne02: 1,
            nb00: block_bytes as u64,
            nb01: row_bytes,
            nb02: row_bytes * rows as u64,
            nb03: row_bytes * rows as u64,
            ne10: cols as i32,
            ne11: 1,
            ne12: 1,
            nb10: 4,
            nb11: (cols * 4) as u64,
            nb12: (cols * 4) as u64,
            nb13: (cols * 4) as u64,
            ne0: rows as i32,
            ne1: 1,
            nr0,
            r2: 1,
            r3: 1,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GgmlKargsMulMvExt {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne10: i32,
    ne11: i32,
    ne12: i32,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ne0: i32,
    ne1: i32,
    r2: i16,
    r3: i16,
}

const _: () = assert!(std::mem::size_of::<GgmlKargsMulMvExt>() == 112);

impl GgmlKargsMulMvExt {
    fn for_matmul(rows: usize, cols: usize, tokens: usize, block_bytes: usize) -> Self {
        let row_bytes = (cols / 256 * block_bytes) as u64;
        let weights_bytes = row_bytes * rows as u64;
        let input_row_bytes = (cols * std::mem::size_of::<f32>()) as u64;
        let input_bytes = input_row_bytes * tokens as u64;
        Self {
            ne00: cols as i32,
            ne01: rows as i32,
            ne02: 1,
            nb00: block_bytes as u64,
            nb01: row_bytes,
            nb02: weights_bytes,
            nb03: weights_bytes,
            ne10: cols as i32,
            ne11: tokens as i32,
            ne12: 1,
            nb10: std::mem::size_of::<f32>() as u64,
            nb11: input_row_bytes,
            nb12: input_bytes,
            nb13: input_bytes,
            ne0: rows as i32,
            ne1: tokens as i32,
            r2: 1,
            r3: 1,
        }
    }
}
