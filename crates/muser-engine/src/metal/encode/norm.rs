use metal::{ComputeCommandEncoderRef, MTLSize};

use super::{set_value, MetalKernels};
use crate::metal::buffer::GpuBuffer;

#[repr(C)]
#[derive(Clone, Copy)]
struct GgmlMetalKargsNorm {
    ne00: i32,
    ne00_t: i32,
    nb1: u64,
    nb2: u64,
    nb3: u64,
    eps: f32,
    nef1: [i32; 3],
    nef2: [i32; 3],
    nef3: [i32; 3],
    nbf1: [u64; 3],
    nbf2: [u64; 3],
    nbf3: [u64; 3],
}

impl MetalKernels {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_cross_vendor_rms_then_weight(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &GpuBuffer,
        weight: &GpuBuffer,
        output: &GpuBuffer,
        rows: usize,
        dim: usize,
        eps: f32,
    ) {
        encoder.set_compute_pipeline_state(&self.cross_vendor_rms_unweighted);
        encoder.set_buffer(0, Some(input.metal()), 0);
        encoder.set_buffer(1, Some(output.metal()), 0);
        set_value(encoder, 2, &(dim as u32));
        set_value(encoder, 3, &eps);
        encoder.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(32, 1, 1));

        let normalized_barrier: [&metal::ResourceRef; 1] = [output.metal()];
        encoder.memory_barrier_with_resources(&normalized_barrier);
        encoder.set_compute_pipeline_state(&self.cross_vendor_mul_weight);
        encoder.set_buffer(0, Some(output.metal()), 0);
        encoder.set_buffer(1, Some(weight.metal()), 0);
        set_value(encoder, 2, &((dim * rows) as u32));
        set_value(encoder, 3, &(dim as u32));
        super::dispatch_1d(encoder, dim * rows);
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_ggml_rms_norm(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &metal::ComputePipelineStateRef,
        input: &GpuBuffer,
        weight: &GpuBuffer,
        add: &GpuBuffer,
        output: &GpuBuffer,
        dim: usize,
        eps: f32,
        rows: usize,
        fused_add: bool,
    ) {
        let row_bytes = (dim * std::mem::size_of::<f32>()) as u64;
        let plane_bytes = row_bytes * rows as u64;
        let args = GgmlMetalKargsNorm {
            ne00: dim as i32,
            ne00_t: (dim / 4) as i32,
            nb1: row_bytes,
            nb2: plane_bytes,
            nb3: plane_bytes,
            eps,
            nef1: [rows as i32, 1, if fused_add { rows as i32 } else { 1 }],
            nef2: [1, 1, 1],
            nef3: [1, 1, 1],
            nbf1: [row_bytes, row_bytes, row_bytes],
            nbf2: [plane_bytes, row_bytes, plane_bytes],
            nbf3: [plane_bytes, row_bytes, plane_bytes],
        };
        encoder.set_compute_pipeline_state(pipeline);
        set_value(encoder, 0, &args);
        encoder.set_buffer(1, Some(input.metal()), 0);
        encoder.set_buffer(2, Some(weight.metal()), 0);
        encoder.set_buffer(3, Some(add.metal()), 0);
        encoder.set_buffer(4, Some(output.metal()), 0);
        encoder.set_threadgroup_memory_length(0, 32 * std::mem::size_of::<f32>() as u64);
        let threads = (dim / 4).next_power_of_two().clamp(32, 1024);
        encoder.dispatch_thread_groups(
            MTLSize::new(rows as u64, 1, 1),
            MTLSize::new(threads as u64, 1, 1),
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_norm_residual_rms_norm_batch_dual_eps(
        &self,
        encoder: &ComputeCommandEncoderRef,
        hidden: &GpuBuffer,
        source: &GpuBuffer,
        output: &GpuBuffer,
        first_weight: &GpuBuffer,
        second_weight: &GpuBuffer,
        dim: usize,
        first_eps: f32,
        second_eps: f32,
        rows: usize,
    ) {
        debug_assert_eq!(hidden.len(), dim * rows);
        debug_assert_eq!(source.len(), dim * rows);
        debug_assert_eq!(output.len(), dim * rows);
        debug_assert_eq!(first_weight.len(), dim);
        debug_assert_eq!(second_weight.len(), dim);
        if std::env::var_os("MUSER_CROSS_VENDOR_QK").is_some() {
            self.encode_cross_vendor_rms_then_weight(
                encoder,
                source,
                first_weight,
                output,
                rows,
                dim,
                first_eps,
            );
            let post_norm_barrier: [&metal::ResourceRef; 1] = [output.metal()];
            encoder.memory_barrier_with_resources(&post_norm_barrier);

            encoder.set_compute_pipeline_state(&self.cross_vendor_residual_add);
            encoder.set_buffer(0, Some(hidden.metal()), 0);
            encoder.set_buffer(1, Some(output.metal()), 0);
            set_value(encoder, 2, &((dim * rows) as u32));
            super::dispatch_1d(encoder, dim * rows);
            let residual_barrier: [&metal::ResourceRef; 1] = [hidden.metal()];
            encoder.memory_barrier_with_resources(&residual_barrier);

            self.encode_cross_vendor_rms_then_weight(
                encoder,
                hidden,
                second_weight,
                output,
                rows,
                dim,
                second_eps,
            );
            return;
        }
        self.bind(encoder, "muser_fused_norm_residual_rms_norm_batch_dual_eps");
        encoder.set_buffer(0, Some(hidden.metal()), 0);
        encoder.set_buffer(1, Some(source.metal()), 0);
        encoder.set_buffer(2, Some(output.metal()), 0);
        encoder.set_buffer(3, Some(first_weight.metal()), 0);
        encoder.set_buffer(4, Some(second_weight.metal()), 0);
        set_value(encoder, 5, &(dim as u32));
        set_value(encoder, 6, &first_eps);
        set_value(encoder, 7, &second_eps);
        // Match the two pinned ggml f32x4 norm kernels this replaces: 1,024
        // threads, 32 SIMD-group partials, and the same second-stage simd sum.
        encoder.set_threadgroup_memory_length(0, 128);
        encoder.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(1024, 1, 1));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_norm_residual_rms_norm_32sg(
        &self,
        encoder: &ComputeCommandEncoderRef,
        hidden: &GpuBuffer,
        source: &GpuBuffer,
        output: &GpuBuffer,
        first_weight: &GpuBuffer,
        second_weight: &GpuBuffer,
        dim: usize,
        first_eps: f32,
        second_eps: f32,
    ) {
        self.encode_fused_norm_residual_rms_norm_32sg_batch(
            encoder,
            hidden,
            source,
            output,
            first_weight,
            second_weight,
            dim,
            first_eps,
            second_eps,
            1,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_norm_residual_rms_norm_32sg_batch(
        &self,
        encoder: &ComputeCommandEncoderRef,
        hidden: &GpuBuffer,
        source: &GpuBuffer,
        output: &GpuBuffer,
        first_weight: &GpuBuffer,
        second_weight: &GpuBuffer,
        dim: usize,
        first_eps: f32,
        second_eps: f32,
        rows: usize,
    ) {
        debug_assert_eq!(hidden.len(), dim * rows);
        debug_assert_eq!(source.len(), dim * rows);
        debug_assert_eq!(output.len(), dim * rows);
        debug_assert_eq!(first_weight.len(), dim);
        debug_assert_eq!(second_weight.len(), dim);
        if std::env::var_os("MUSER_CROSS_VENDOR_QK").is_some() {
            // The strict seam contract contains three observable model-dtype
            // boundaries: post norm, residual add, and next norm. Reuse the
            // decomposed no-fast-math route; the 32-SIMD serving fusion is
            // intentionally unreachable here.
            self.encode_fused_norm_residual_rms_norm_batch_dual_eps(
                encoder,
                hidden,
                source,
                output,
                first_weight,
                second_weight,
                dim,
                first_eps,
                second_eps,
                rows,
            );
            return;
        }
        self.bind(encoder, "muser_fused_norm_residual_rms_norm_32sg");
        encoder.set_buffer(0, Some(hidden.metal()), 0);
        encoder.set_buffer(1, Some(source.metal()), 0);
        encoder.set_buffer(2, Some(output.metal()), 0);
        encoder.set_buffer(3, Some(first_weight.metal()), 0);
        encoder.set_buffer(4, Some(second_weight.metal()), 0);
        set_value(encoder, 5, &(dim as u32));
        set_value(encoder, 6, &first_eps);
        set_value(encoder, 7, &second_eps);
        // 32 SIMD groups keep the 6,656-wide Muse tail resident and match the
        // accepted Ferrite geometry. 33 floats are padded to Metal's 16-byte
        // dynamic-threadgroup-memory alignment.
        encoder.set_threadgroup_memory_length(0, 144);
        encoder.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(1024, 1, 1));
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_rms_norm_mul(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &GpuBuffer,
        weight: &GpuBuffer,
        output: &GpuBuffer,
        dim: usize,
        eps: f32,
        rows: usize,
    ) {
        debug_assert_eq!(input.len(), dim * rows);
        debug_assert_eq!(weight.len(), dim);
        debug_assert_eq!(output.len(), dim * rows);
        if std::env::var_os("MUSER_CROSS_VENDOR_QK").is_some() {
            self.encode_cross_vendor_rms_per_head(encoder, input, weight, output, rows, dim, eps);
            return;
        }
        if dim.is_multiple_of(4) {
            if let Some(pipeline) = self.ggml_rms_norm_mul() {
                self.encode_ggml_rms_norm(
                    encoder, pipeline, input, weight, weight, output, dim, eps, rows, false,
                );
                return;
            }
        }

        self.bind(encoder, "rms_norm_batch");
        encoder.set_buffer(0, Some(input.metal()), 0);
        encoder.set_buffer(1, Some(weight.metal()), 0);
        encoder.set_buffer(2, Some(output.metal()), 0);
        set_value(encoder, 3, &(dim as u32));
        set_value(encoder, 4, &eps);
        encoder.set_threadgroup_memory_length(0, 32);
        encoder.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(128, 1, 1));
    }

    /// Normalize projected Q/K rows using the producer's explicit model-dtype
    /// boundary before applying the per-head scale.  vLLM materializes its
    /// weightless QK RMSNorm output in F16 and then applies `scale_query_by`
    /// as a second F16 operation.  The ordinary fused RMS+weight kernel loses
    /// that intermediate rounding point and is therefore not seam-exact.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_qk_norm(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &GpuBuffer,
        weight: &GpuBuffer,
        output: &GpuBuffer,
        dim: usize,
        eps: f32,
        rows: usize,
    ) {
        if std::env::var_os("MUSER_CROSS_VENDOR_QK").is_some() {
            self.encode_cross_vendor_rms_then_weight(
                encoder, input, weight, output, rows, dim, eps,
            );
            return;
        }
        self.encode_rms_norm_mul(encoder, input, weight, output, dim, eps, rows);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_fused_rms_norm_residual_add_batch(
        &self,
        encoder: &ComputeCommandEncoderRef,
        residual: &GpuBuffer,
        input: &GpuBuffer,
        weight: &GpuBuffer,
        dim: usize,
        eps: f32,
        rows: usize,
    ) {
        debug_assert_eq!(residual.len(), dim * rows);
        debug_assert_eq!(input.len(), dim * rows);
        debug_assert_eq!(weight.len(), dim);
        if std::env::var_os("MUSER_CROSS_VENDOR_QK").is_some() {
            // `input` is a dead projection temporary at every call site. Use
            // it as the materialized post-norm boundary, then add it to the
            // residual in a separate no-fast-math dispatch.
            self.encode_cross_vendor_rms_then_weight(encoder, input, weight, input, rows, dim, eps);
            let post_norm_barrier: [&metal::ResourceRef; 1] = [input.metal()];
            encoder.memory_barrier_with_resources(&post_norm_barrier);
            encoder.set_compute_pipeline_state(&self.cross_vendor_residual_add);
            encoder.set_buffer(0, Some(residual.metal()), 0);
            encoder.set_buffer(1, Some(input.metal()), 0);
            set_value(encoder, 2, &((dim * rows) as u32));
            super::dispatch_1d(encoder, dim * rows);
            return;
        }
        if dim.is_multiple_of(4) {
            if let Some(pipeline) = self.ggml_rms_norm_mul_add() {
                self.encode_ggml_rms_norm(
                    encoder, pipeline, input, weight, residual, residual, dim, eps, rows, true,
                );
                return;
            }
        }
        self.bind(encoder, "fused_rms_norm_residual_add_batch");
        encoder.set_buffer(0, Some(residual.metal()), 0);
        encoder.set_buffer(1, Some(input.metal()), 0);
        encoder.set_buffer(2, Some(weight.metal()), 0);
        set_value(encoder, 3, &(dim as u32));
        set_value(encoder, 4, &eps);
        encoder.set_threadgroup_memory_length(0, 32);
        encoder.dispatch_thread_groups(MTLSize::new(rows as u64, 1, 1), MTLSize::new(128, 1, 1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::context::MetalContext;
    use half::f16;
    use sha2::{Digest, Sha256};
    use std::time::Duration;

    fn exact_rms_reference(input: &[f32], weight: &[f32], dim: usize, eps: f32) -> Vec<f32> {
        assert!(input.len().is_multiple_of(dim));
        let mut output = vec![0.0; input.len()];
        for (row, values) in input.chunks_exact(dim).enumerate() {
            let mut partial = [0.0f32; 32];
            for (lane, lane_sum) in partial.iter_mut().enumerate() {
                for &value in values.iter().skip(lane).step_by(32) {
                    *lane_sum = value.mul_add(value, *lane_sum);
                }
            }
            for distance in [16, 8, 4, 2, 1] {
                for lane in 0..distance {
                    partial[lane] += partial[lane + distance];
                }
            }
            let inverse = 1.0 / partial[0].mul_add(1.0 / dim as f32, eps).sqrt();
            for index in 0..dim {
                output[row * dim + index] =
                    f16::from_f32((values[index] * inverse).mul_add(weight[index], 0.0)).to_f32();
            }
        }
        output
    }

    fn exact_split_rms_reference(input: &[f32], weight: &[f32], dim: usize, eps: f32) -> Vec<f32> {
        assert!(input.len().is_multiple_of(dim));
        let mut output = vec![0.0; input.len()];
        for (row, values) in input.chunks_exact(dim).enumerate() {
            let mut partial = [0.0f32; 32];
            for (lane, lane_sum) in partial.iter_mut().enumerate() {
                for &value in values.iter().skip(lane).step_by(32) {
                    *lane_sum = value.mul_add(value, *lane_sum);
                }
            }
            for distance in [16, 8, 4, 2, 1] {
                for lane in 0..distance {
                    partial[lane] += partial[lane + distance];
                }
            }
            let inverse = 1.0 / partial[0].mul_add(1.0 / dim as f32, eps).sqrt();
            for index in 0..dim {
                let normalized = f16::from_f32(values[index].mul_add(inverse, 0.0)).to_f32();
                output[row * dim + index] =
                    f16::from_f32(normalized.mul_add(weight[index], 0.0)).to_f32();
            }
        }
        output
    }

    #[test]
    fn cross_vendor_rms_matches_the_fixed_32_lane_oracle() {
        let context = MetalContext::new().expect("Metal context");
        let kernels = MetalKernels::new(&context).expect("Muse primitive pipelines");
        for dim in [128usize, 6_656] {
            let rows = 5;
            let input = (0..rows * dim)
                .map(|index| {
                    f16::from_f32(((index * 37 % 1009) as f32 - 503.0) * 0.003_906_25).to_f32()
                })
                .collect::<Vec<_>>();
            let weight = (0..dim)
                .map(|index| {
                    1.0 + f16::from_f32(((index * 19 % 257) as f32 - 128.0) * 0.000_244_140_63)
                        .to_f32()
                })
                .collect::<Vec<_>>();
            let expected = exact_rms_reference(&input, &weight, dim, 1.0e-5);
            let input = GpuBuffer::from_f32(&context, &input).unwrap();
            let weight = GpuBuffer::from_f32(&context, &weight).unwrap();
            let output = GpuBuffer::zeros(&context, rows * dim).unwrap();
            let command = context.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            kernels.encode_cross_vendor_rms_per_head(
                encoder, &input, &weight, &output, rows, dim, 1.0e-5,
            );
            encoder.end_encoding();
            command.commit();
            context
                .wait_for_completion(command, Duration::from_secs(30))
                .expect("cross-vendor RMS fixture completion");
            for (index, (&actual, &expected)) in output.as_slice().iter().zip(&expected).enumerate()
            {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "cross-vendor RMS mismatch at dim={dim} index={index}"
                );
            }
            let digest = Sha256::digest(
                output
                    .as_slice()
                    .iter()
                    .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
                    .collect::<Vec<_>>(),
            );
            let expected_digest = match dim {
                128 => "be21529716bd34293eb17720c215d6fd3e28346355eeca2d00139e890172391b",
                6_656 => "d30250ab87ad20da7b138ee7c57dc66aed20878655d991e5625013272db7f885",
                _ => unreachable!(),
            };
            assert_eq!(format!("{digest:x}"), expected_digest);
        }
    }

    #[test]
    fn cross_vendor_qk_norm_preserves_the_split_f16_boundary() {
        let context = MetalContext::new().expect("Metal context");
        let kernels = MetalKernels::new(&context).expect("Muse primitive pipelines");
        let dim = 128;
        let rows = 5;
        let input = (0..rows * dim)
            .map(|index| {
                f16::from_f32(((index * 37 % 1009) as f32 - 503.0) * 0.003_906_25).to_f32()
            })
            .collect::<Vec<_>>();
        let weight = (0..dim)
            .map(|index| {
                1.0 + f16::from_f32(((index * 19 % 257) as f32 - 128.0) * 0.000_244_140_63).to_f32()
            })
            .collect::<Vec<_>>();
        let expected = exact_split_rms_reference(&input, &weight, dim, 1.0e-5);
        let input = GpuBuffer::from_f32(&context, &input).unwrap();
        let weight = GpuBuffer::from_f32(&context, &weight).unwrap();
        let output = GpuBuffer::zeros(&context, rows * dim).unwrap();
        let command = context.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        kernels.encode_cross_vendor_rms_then_weight(
            encoder, &input, &weight, &output, rows, dim, 1.0e-5,
        );
        encoder.end_encoding();
        command.commit();
        context
            .wait_for_completion(command, Duration::from_secs(30))
            .expect("cross-vendor split QK RMS fixture completion");
        for (index, (&actual, &expected)) in output.as_slice().iter().zip(&expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "cross-vendor split QK RMS mismatch at index={index}"
            );
        }
    }

    #[test]
    fn strict_dual_norm_dispatch_preserves_all_model_dtype_boundaries() {
        let context = MetalContext::new().expect("Metal context");
        let kernels = MetalKernels::new(&context).expect("Muse primitive pipelines");
        let dim = 128;
        let rows = 3;
        let hidden_values = (0..rows * dim)
            .map(|index| f16::from_f32(((index * 31 % 761) as f32 - 380.0) * 0.003_906_25).to_f32())
            .collect::<Vec<_>>();
        let source_values = (0..rows * dim)
            .map(|index| {
                f16::from_f32(((index * 43 % 997) as f32 - 498.0) * 0.001_953_125).to_f32()
            })
            .collect::<Vec<_>>();
        let first_weight = (0..dim)
            .map(|index| {
                1.0 + f16::from_f32(((index * 17 % 251) as f32 - 125.0) * 0.000_244_140_63).to_f32()
            })
            .collect::<Vec<_>>();
        let second_weight = (0..dim)
            .map(|index| {
                1.0 + f16::from_f32(((index * 23 % 257) as f32 - 128.0) * 0.000_244_140_63).to_f32()
            })
            .collect::<Vec<_>>();
        let post = exact_split_rms_reference(&source_values, &first_weight, dim, 1.0e-8);
        let expected_hidden = hidden_values
            .iter()
            .zip(&post)
            .map(|(&hidden, &delta)| f16::from_f32(hidden + delta).to_f32())
            .collect::<Vec<_>>();
        let expected_output =
            exact_split_rms_reference(&expected_hidden, &second_weight, dim, 1.0e-5);

        let hidden = GpuBuffer::from_f32(&context, &hidden_values).unwrap();
        let source = GpuBuffer::from_f32(&context, &source_values).unwrap();
        let output = GpuBuffer::zeros(&context, rows * dim).unwrap();
        let first_weight = GpuBuffer::from_f32(&context, &first_weight).unwrap();
        let second_weight = GpuBuffer::from_f32(&context, &second_weight).unwrap();
        unsafe { std::env::set_var("MUSER_CROSS_VENDOR_QK", "1") };
        let command = context.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        kernels.encode_fused_norm_residual_rms_norm_32sg_batch(
            encoder,
            &hidden,
            &source,
            &output,
            &first_weight,
            &second_weight,
            dim,
            1.0e-8,
            1.0e-5,
            rows,
        );
        encoder.end_encoding();
        command.commit();
        context
            .wait_for_completion(command, Duration::from_secs(30))
            .expect("strict dual-norm completion");
        unsafe { std::env::remove_var("MUSER_CROSS_VENDOR_QK") };

        for (index, (&actual, &expected)) in
            hidden.as_slice().iter().zip(&expected_hidden).enumerate()
        {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "strict residual mismatch at {index}"
            );
        }
        for (index, (&actual, &expected)) in
            output.as_slice().iter().zip(&expected_output).enumerate()
        {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "strict next-norm mismatch at {index}"
            );
        }
    }
}
