use metal::{ComputeCommandEncoderRef, MTLSize};

use super::{set_value, MetalKernels};
use crate::metal::buffer::{GpuBuffer, GpuByteView};

#[repr(C)]
#[derive(Clone, Copy)]
struct GgmlMetalKargsRope {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    ne03: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne0: i32,
    ne1: i32,
    ne2: i32,
    ne3: i32,
    nb0: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
    n_past: i32,
    n_dims: i32,
    n_ctx_orig: i32,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    sect_0: i32,
    sect_1: i32,
    sect_2: i32,
    sect_3: i32,
    src2: bool,
}

impl MetalKernels {
    #[allow(clippy::too_many_arguments)]
    fn encode_cross_vendor_rope_norm_batch_cached(
        &self,
        encoder: &ComputeCommandEncoderRef,
        q: &GpuBuffer,
        k: &GpuBuffer,
        frequencies: &GpuBuffer,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        token_count: usize,
        positions: GpuByteView<'_>,
    ) {
        debug_assert_eq!(positions.len(), token_count * std::mem::size_of::<u32>());
        debug_assert!(frequencies.len() >= head_dim);
        encoder.set_compute_pipeline_state(&self.cross_vendor_rope);
        encoder.set_buffer(0, Some(q.metal()), 0);
        encoder.set_buffer(1, Some(k.metal()), 0);
        encoder.set_buffer(2, Some(frequencies.metal()), 0);
        encoder.set_buffer(3, Some(positions.metal()), positions.offset() as u64);
        set_value(encoder, 4, &(n_heads as u32));
        set_value(encoder, 5, &(n_kv_heads as u32));
        set_value(encoder, 6, &(head_dim as u32));
        set_value(encoder, 7, &(token_count as u32));
        let pairs = (n_heads + n_kv_heads) * (head_dim / 2);
        encoder.dispatch_thread_groups(
            MTLSize::new(pairs.div_ceil(32) as u64, token_count as u64, 1),
            MTLSize::new(32, 1, 1),
        );
    }

    /// Ferrite a85048a90 NORM-layout cached-frequency RoPE. Muse rotates
    /// adjacent pairs and skips this dispatch entirely on NoPE layers.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_rope_norm_batch_cached(
        &self,
        encoder: &ComputeCommandEncoderRef,
        q: &GpuBuffer,
        k: &GpuBuffer,
        frequencies: &GpuBuffer,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        start_position: usize,
        token_count: usize,
        positions: Option<GpuByteView<'_>>,
        freq_base: f32,
        n_ctx_orig: usize,
    ) {
        debug_assert_eq!(q.len(), token_count * n_heads * head_dim);
        debug_assert_eq!(k.len(), token_count * n_kv_heads * head_dim);
        if std::env::var_os("MUSER_CROSS_VENDOR_QK").is_some()
            || std::env::var_os("MUSER_CROSS_VENDOR_ROPE_CACHE").is_some()
        {
            if std::env::var_os("MUSER_CROSS_VENDOR_ROPE_BYPASS").is_some() {
                return;
            }
            let positions = positions.expect("cross-vendor RoPE requires explicit positions");
            self.encode_cross_vendor_rope_norm_batch_cached(
                encoder,
                q,
                k,
                frequencies,
                n_heads,
                n_kv_heads,
                head_dim,
                token_count,
                positions,
            );
            return;
        }
        debug_assert_eq!(frequencies.len(), head_dim / 2);
        if let (Some(pipeline), Some(positions)) = (self.ggml_rope_norm(), positions) {
            debug_assert_eq!(positions.len(), token_count * std::mem::size_of::<i32>());
            for (values, heads) in [(q, n_heads), (k, n_kv_heads)] {
                let row_bytes = (head_dim * std::mem::size_of::<f32>()) as u64;
                let token_bytes = row_bytes * heads as u64;
                let plane_bytes = token_bytes * token_count as u64;
                let args = GgmlMetalKargsRope {
                    ne00: head_dim as i32,
                    ne01: heads as i32,
                    ne02: token_count as i32,
                    ne03: 1,
                    nb00: std::mem::size_of::<f32>() as u64,
                    nb01: row_bytes,
                    nb02: token_bytes,
                    nb03: plane_bytes,
                    ne0: head_dim as i32,
                    ne1: heads as i32,
                    ne2: token_count as i32,
                    ne3: 1,
                    nb0: std::mem::size_of::<f32>() as u64,
                    nb1: row_bytes,
                    nb2: token_bytes,
                    nb3: plane_bytes,
                    n_past: 0,
                    n_dims: head_dim as i32,
                    n_ctx_orig: n_ctx_orig as i32,
                    freq_base,
                    freq_scale: 1.0,
                    ext_factor: 0.0,
                    attn_factor: 1.0,
                    beta_fast: 32.0,
                    beta_slow: 1.0,
                    sect_0: 0,
                    sect_1: 0,
                    sect_2: 0,
                    sect_3: 0,
                    src2: false,
                };
                encoder.set_compute_pipeline_state(pipeline);
                set_value(encoder, 0, &args);
                encoder.set_buffer(1, Some(values.metal()), 0);
                encoder.set_buffer(2, Some(positions.metal()), positions.offset() as u64);
                encoder.set_buffer(3, Some(values.metal()), 0);
                encoder.set_buffer(4, Some(values.metal()), 0);
                encoder.dispatch_thread_groups(
                    MTLSize::new(heads as u64, token_count as u64, 1),
                    MTLSize::new(head_dim.min(1024) as u64, 1, 1),
                );
            }
            return;
        }
        self.bind(encoder, "rope_norm_batch_cached");
        encoder.set_buffer(0, Some(q.metal()), 0);
        encoder.set_buffer(1, Some(k.metal()), 0);
        encoder.set_buffer(2, Some(frequencies.metal()), 0);
        set_value(encoder, 3, &(n_heads as u32));
        set_value(encoder, 4, &(n_kv_heads as u32));
        set_value(encoder, 5, &(head_dim as u32));
        set_value(encoder, 6, &(start_position as u32));
        let pairs = (n_heads + n_kv_heads) * (head_dim / 2);
        encoder.dispatch_thread_groups(
            MTLSize::new(pairs.div_ceil(32) as u64, token_count as u64, 1),
            MTLSize::new(32, 1, 1),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::buffer::GpuBytes;
    use crate::metal::context::MetalContext;
    use half::f16;
    use std::time::Duration;

    fn expected_pair(x0: f32, x1: f32, cosine: f32, sine: f32) -> (f32, f32) {
        let x0 = f16::from_f32(x0).to_f32();
        let x1 = f16::from_f32(x1).to_f32();
        let x0_cos = x0.mul_add(cosine, 0.0);
        let x1_cos = x1.mul_add(cosine, 0.0);
        (
            f16::from_f32((-x1).mul_add(sine, x0_cos)).to_f32(),
            f16::from_f32(x0.mul_add(sine, x1_cos)).to_f32(),
        )
    }

    #[test]
    fn canonical_q30_rope_matches_cpu_fma_order_bits() {
        const TOKENS: usize = 4;
        const HEAD_DIM: usize = 128;
        const Q_HEADS: usize = 2;
        const K_HEADS: usize = 1;
        let context = MetalContext::new().expect("Metal context");
        let kernels = MetalKernels::new(&context).expect("Muse primitive pipelines");
        let trig_values = crate::rope_nco::canonical_rope_table(TOKENS, HEAD_DIM, 500_000.0);
        let trig = GpuBuffer::from_f32(&context, &trig_values).expect("NCO table buffer");
        let positions_bytes: Vec<u8> = (0..TOKENS as u32).flat_map(u32::to_le_bytes).collect();
        let positions = GpuBytes::from_bytes(&context, &positions_bytes).expect("positions");

        let make_values = |heads: usize, salt: usize| {
            (0..TOKENS * heads * HEAD_DIM)
                .map(|index| {
                    f16::from_f32(((index * 37 + salt) % 1009) as f32 * 0.003_906_25 - 1.75)
                        .to_f32()
                })
                .collect::<Vec<_>>()
        };
        let q_source = make_values(Q_HEADS, 11);
        let k_source = make_values(K_HEADS, 29);
        let q = GpuBuffer::from_f32(&context, &q_source).expect("Q buffer");
        let k = GpuBuffer::from_f32(&context, &k_source).expect("K buffer");

        let command = context.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        kernels.encode_cross_vendor_rope_norm_batch_cached(
            encoder,
            &q,
            &k,
            &trig,
            Q_HEADS,
            K_HEADS,
            HEAD_DIM,
            TOKENS,
            positions.view(0, positions.len()).expect("positions view"),
        );
        encoder.end_encoding();
        command.commit();
        context
            .wait_for_completion(command, Duration::from_secs(30))
            .expect("canonical RoPE completion");

        if let Some(output_dir) = std::env::var_os("MUSER_ROPE_FIXTURE_OUTPUT") {
            let output_dir = std::path::PathBuf::from(output_dir);
            assert!(
                !output_dir.exists(),
                "refusing to replace RoPE fixture output"
            );
            std::fs::create_dir_all(&output_dir).expect("create RoPE fixture output");
            for (name, values) in [("q", q.as_slice()), ("k", k.as_slice())] {
                let bytes: Vec<u8> = values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect();
                std::fs::write(output_dir.join(format!("{name}.f32le")), bytes)
                    .expect("write RoPE fixture output");
            }
        }

        for (actual, source, heads, label) in [
            (q.as_slice(), q_source.as_slice(), Q_HEADS, "q"),
            (k.as_slice(), k_source.as_slice(), K_HEADS, "k"),
        ] {
            for token in 0..TOKENS {
                for head in 0..heads {
                    let base = (token * heads + head) * HEAD_DIM;
                    for pair in 0..HEAD_DIM / 2 {
                        let index = base + pair * 2;
                        let trig_index = token * HEAD_DIM + pair * 2;
                        let (expected0, expected1) = expected_pair(
                            source[index],
                            source[index + 1],
                            trig_values[trig_index],
                            trig_values[trig_index + 1],
                        );
                        assert_eq!(
                            actual[index].to_bits(),
                            expected0.to_bits(),
                            "{label} token {token} head {head} pair {pair} x0"
                        );
                        assert_eq!(
                            actual[index + 1].to_bits(),
                            expected1.to_bits(),
                            "{label} token {token} head {head} pair {pair} x1"
                        );
                    }
                }
            }
        }
    }
}
