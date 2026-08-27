//! Metal runtime substrate (device, command buffers, PSO cache, FFI) + the
//! per-op encoders. macOS-only, behind the `metal` feature.
//!
//! Extraction source (PULL-AND-SIMPLIFY):
//! `ferrite-metal-core::{context,buffer,pso_cache,fast_metal_ffi,
//! barrier_tracker}`. Keep the device/command-buffer/PSO-cache harness and
//! the runtime shader-compile path (`include_str!` the `.metal` sources in
//! `../shaders/`, `newLibraryWithSource` on first use, cache PSOs — this
//! keeps muser a pure-source checkout, no Xcode/`.metallib` build step).
//! **Drop** the route-registry/receipt/override machinery that substrate
//! carries in Ferrite — that's VM plumbing muser has no VM to serve.

pub mod buffer;
pub mod context;
pub mod dflash;
pub mod encode;
pub mod pso_cache;
pub mod residency;
pub mod vision;

#[cfg(test)]
mod tests {
    use super::buffer::{GpuBuffer, GpuBytes, GpuHalfBuffer};
    use super::context::MetalContext;
    use super::encode::MetalKernels;
    use crate::gguf::GgmlType;
    use crate::quant::{dequant_q4_k, dot_q4_k_f32, dot_q5_k_f32, dot_q6_k_f32, f16_to_f32};
    use sha2::{Digest, Sha256};
    use std::time::Duration;

    #[test]
    fn primitive_pipeline_matches_scalar_reference() {
        let context = MetalContext::new().expect("Metal context");
        let kernels = MetalKernels::new(&context).expect("Muse primitive pipelines");

        let input = GpuBuffer::from_f32(&context, &[1.0, -2.0, 3.0, -4.0]).unwrap();
        let weight = GpuBuffer::from_f32(&context, &[0.5, 1.0, 1.5, 2.0]).unwrap();
        let norm = GpuBuffer::zeros(&context, 4).unwrap();
        let gated = GpuBuffer::from_f32(&context, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let gate = GpuBuffer::from_f32(&context, &[0.0, 1.0, -1.0, 2.0]).unwrap();
        let ffn_gate = GpuBuffer::from_f32(&context, &[-2.0, -0.5, 0.5, 2.0]).unwrap();
        let ffn_up = GpuBuffer::from_f32(&context, &[3.0, 4.0, 5.0, 6.0]).unwrap();
        let rope_q = GpuBuffer::from_f32(&context, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let rope_k = GpuBuffer::from_f32(&context, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let rope_frequencies =
            GpuBuffer::from_f32(&context, &[1.0, 500_000.0f32.powf(-2.0 / 4.0)]).unwrap();
        let logits = GpuBuffer::from_f32(&context, &[-30.0, -1.0, 1.0, 30.0]).unwrap();

        let command = context.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        kernels.encode_rms_norm_mul(encoder, &input, &weight, &norm, 4, 1e-5, 1);
        kernels.encode_sigmoid_gate(encoder, &gated, &gate);
        kernels.encode_silu_mul(encoder, &ffn_gate, &ffn_up);
        kernels.encode_rope_norm_batch_cached(
            encoder,
            &rope_q,
            &rope_k,
            &rope_frequencies,
            1,
            1,
            4,
            7,
            1,
            None,
            500_000.0,
            128,
        );
        kernels.encode_scale_softcap(encoder, &logits, 1.0 / 26.0f32.sqrt(), 20.0);
        encoder.end_encoding();
        command.commit();
        context
            .wait_for_completion(command, Duration::from_secs(30))
            .expect("Metal completion");

        let sum_sq = 1.0 + 4.0 + 9.0 + 16.0;
        let inv = 1.0 / (sum_sq / 4.0 + 1e-5f32).sqrt();
        let expected_norm = [0.5 * inv, -2.0 * inv, 4.5 * inv, -8.0 * inv];
        assert_close(norm.as_slice(), &expected_norm, 2e-5);
        let expected_gate = [
            0.5,
            2.0 / (1.0 + (-1.0f32).exp()),
            3.0 / (1.0 + 1.0f32.exp()),
            4.0 / (1.0 + (-2.0f32).exp()),
        ];
        assert_close(gated.as_slice(), &expected_gate, 2e-5);

        let expected_ffn = [-2.0f32, -0.5, 0.5, 2.0]
            .into_iter()
            .zip([3.0f32, 4.0, 5.0, 6.0])
            .map(|(value, up)| value / (1.0 + (-value).exp()) * up)
            .collect::<Vec<_>>();
        assert_close(ffn_gate.as_slice(), &expected_ffn, 2e-5);

        let mut expected_rope = [1.0f32, 2.0, 3.0, 4.0];
        for pair in 0..2 {
            let theta = 7.0 * 500_000.0f32.powf(-2.0 * pair as f32 / 4.0);
            let (sine, cosine) = theta.sin_cos();
            let x0 = expected_rope[pair * 2];
            let x1 = expected_rope[pair * 2 + 1];
            expected_rope[pair * 2] = x0 * cosine - x1 * sine;
            expected_rope[pair * 2 + 1] = x0 * sine + x1 * cosine;
        }
        assert_close(rope_q.as_slice(), &expected_rope, 2e-4);
        assert_close(rope_k.as_slice(), &expected_rope, 2e-4);

        let expected_logits = [-30.0f32, -1.0, 1.0, 30.0].map(|value| {
            let scaled = value / 26.0f32.sqrt();
            20.0 * (scaled / 20.0).tanh()
        });
        assert_close(logits.as_slice(), &expected_logits, 2e-5);
    }

    #[test]
    fn greedy_argmax_is_lowest_index_exact_and_fails_closed_on_nonfinite() {
        fn run(values: &[f32], excluded: &[u32]) -> u32 {
            let context = MetalContext::new().expect("Metal context");
            let kernels = MetalKernels::new(&context).expect("Muse primitive pipelines");
            let values = GpuBuffer::from_f32(&context, values).expect("values");
            let blocks = values.len().div_ceil(1024);
            let partial_values = GpuBuffer::zeros(&context, blocks).expect("partial values");
            let partial_indices = GpuBuffer::zeros(&context, blocks).expect("partial indices");
            let result = GpuBuffer::zeros(&context, 1).expect("result");
            let excluded_bytes = if excluded.is_empty() {
                0u32.to_le_bytes().to_vec()
            } else {
                excluded
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect()
            };
            let excluded_buffer =
                GpuBytes::from_bytes(&context, &excluded_bytes).expect("excluded tokens");
            let command = context.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            kernels.encode_greedy_argmax_f32(
                encoder,
                &values,
                &partial_values,
                &partial_indices,
                &result,
                0,
                values.len(),
                &excluded_buffer,
                excluded.len(),
            );
            encoder.end_encoding();
            command.commit();
            context
                .wait_for_completion(command, Duration::from_secs(30))
                .expect("Metal completion");
            result.as_slice()[0].to_bits()
        }

        let mut values = vec![-10.0f32; 2_050];
        values[17] = 4.0;
        values[1_030] = 4.0;
        assert_eq!(run(&values, &[]), 17, "ties select the lowest token ID");
        assert_eq!(run(&values, &[17]), 1_030, "excluded IDs cannot win");
        values[2_000] = f32::NAN;
        assert_eq!(run(&values, &[]), u32::MAX, "nonfinite logits fail closed");
    }

    #[test]
    fn q4k_q5k_matmul_matches_cpu_reference() {
        let context = MetalContext::new().expect("Metal context");
        let kernels = MetalKernels::new(&context).expect("Muse primitive pipelines");
        let input_values = (0..256)
            .map(|index| ((index as f32 * 0.071).sin() + (index % 11) as f32 * 0.01) * 0.5)
            .collect::<Vec<_>>();
        let input = GpuBuffer::from_f32(&context, &input_values).unwrap();

        let q4_bytes = quant_rows(144, 2, false);
        let q5_bytes = quant_rows(176, 2, true);
        let q4 = GpuBytes::from_bytes(&context, &q4_bytes).unwrap();
        let q5 = GpuBytes::from_bytes(&context, &q5_bytes).unwrap();
        let q4_output = GpuBuffer::zeros(&context, 2).unwrap();
        let q5_output = GpuBuffer::zeros(&context, 2).unwrap();

        let command = context.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        kernels.encode_quantized_matmul(
            encoder,
            q4.view(0, q4.len()).unwrap(),
            &input,
            &q4_output,
            GgmlType::Q4_K,
            256,
            2,
            1,
        );
        kernels.encode_quantized_matmul(
            encoder,
            q5.view(0, q5.len()).unwrap(),
            &input,
            &q5_output,
            GgmlType::Q5_K,
            256,
            2,
            1,
        );
        encoder.end_encoding();
        command.commit();
        context
            .wait_for_completion(command, Duration::from_secs(30))
            .expect("Metal completion");

        let expected_q4 = (0..2)
            .map(|row| dot_q4_k_f32(&q4_bytes[row * 144..(row + 1) * 144], &input_values, 256))
            .collect::<Vec<_>>();
        let expected_q5 = (0..2)
            .map(|row| dot_q5_k_f32(&q5_bytes[row * 176..(row + 1) * 176], &input_values, 256))
            .collect::<Vec<_>>();
        assert_relative_close(q4_output.as_slice(), &expected_q4, 2e-4, 0.02);
        assert_relative_close(q5_output.as_slice(), &expected_q5, 2e-4, 0.02);
    }

    #[test]
    fn batched_q4k_q5k_matmul_preserves_token_major_output() {
        let context = MetalContext::new().expect("Metal context");
        let kernels = MetalKernels::new(&context).expect("Muse primitive pipelines");
        let input_values = (0..512)
            .map(|index| ((index as f32 * 0.037).cos() + (index % 17) as f32 * 0.004) * 0.3)
            .collect::<Vec<_>>();
        let input = GpuBuffer::from_f32(&context, &input_values).unwrap();
        let q4_bytes = quant_rows(144, 3, false);
        let q5_bytes = quant_rows(176, 3, true);
        let q4_weights = GpuBytes::from_bytes(&context, &q4_bytes).unwrap();
        let q5_weights = GpuBytes::from_bytes(&context, &q5_bytes).unwrap();
        let q4_output = GpuBuffer::zeros(&context, 6).unwrap();
        let q5_output = GpuBuffer::zeros(&context, 6).unwrap();

        let command = context.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        kernels.encode_quantized_matmul(
            encoder,
            q4_weights.view(0, q4_weights.len()).unwrap(),
            &input,
            &q4_output,
            GgmlType::Q4_K,
            256,
            3,
            2,
        );
        kernels.encode_quantized_matmul(
            encoder,
            q5_weights.view(0, q5_weights.len()).unwrap(),
            &input,
            &q5_output,
            GgmlType::Q5_K,
            256,
            3,
            2,
        );
        encoder.end_encoding();
        command.commit();
        context
            .wait_for_completion(command, Duration::from_secs(30))
            .expect("Metal completion");

        let mut expected_q4 = Vec::with_capacity(6);
        let mut expected_q5 = Vec::with_capacity(6);
        for token in 0..2 {
            let input = &input_values[token * 256..(token + 1) * 256];
            for row in 0..3 {
                expected_q4.push(dot_q4_k_f32(
                    &q4_bytes[row * 144..(row + 1) * 144],
                    input,
                    256,
                ));
                expected_q5.push(dot_q5_k_f32(
                    &q5_bytes[row * 176..(row + 1) * 176],
                    input,
                    256,
                ));
            }
        }
        assert_relative_close(q4_output.as_slice(), &expected_q4, 2e-4, 0.02);
        assert_relative_close(q5_output.as_slice(), &expected_q5, 2e-4, 0.02);
    }

    #[test]
    fn pinned_llama_small_batch_k_quant_projection_matches_cpu() {
        let context = MetalContext::new().expect("Metal context");
        let kernels = MetalKernels::new(&context).expect("Muse primitive pipelines");
        if !kernels.has_llama_mul_mv_ext() {
            eprintln!("skipped: set MUSER_GGML_METALLIB to the pinned llama.cpp metallib");
            return;
        }

        const COLS: usize = 256;
        const ROWS: usize = 11;
        for tokens in 4..=8 {
            let input_values = (0..tokens * COLS)
                .map(|index| ((index as f32 * 0.019).sin() + (index % 23) as f32 * 0.003) * 0.35)
                .collect::<Vec<_>>();
            let input = GpuBuffer::from_f32(&context, &input_values).unwrap();
            let q4_bytes = quant_rows(144, ROWS, false);
            let q5_bytes = quant_rows(176, ROWS, true);
            let q4_weights = GpuBytes::from_bytes(&context, &q4_bytes).unwrap();
            let q5_weights = GpuBytes::from_bytes(&context, &q5_bytes).unwrap();
            let q4_output = GpuBuffer::zeros(&context, tokens * ROWS).unwrap();
            let q5_output = GpuBuffer::zeros(&context, tokens * ROWS).unwrap();

            let command = context.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            kernels.encode_quantized_matmul(
                encoder,
                q4_weights.view(0, q4_weights.len()).unwrap(),
                &input,
                &q4_output,
                GgmlType::Q4_K,
                COLS,
                ROWS,
                tokens,
            );
            kernels.encode_quantized_matmul(
                encoder,
                q5_weights.view(0, q5_weights.len()).unwrap(),
                &input,
                &q5_output,
                GgmlType::Q5_K,
                COLS,
                ROWS,
                tokens,
            );
            encoder.end_encoding();
            command.commit();
            context
                .wait_for_completion(command, Duration::from_secs(30))
                .expect("Metal completion");

            let mut expected_q4 = Vec::with_capacity(tokens * ROWS);
            let mut expected_q5 = Vec::with_capacity(tokens * ROWS);
            for token in 0..tokens {
                let input = &input_values[token * COLS..(token + 1) * COLS];
                for row in 0..ROWS {
                    expected_q4.push(dot_q4_k_f32(
                        &q4_bytes[row * 144..(row + 1) * 144],
                        input,
                        COLS,
                    ));
                    expected_q5.push(dot_q5_k_f32(
                        &q5_bytes[row * 176..(row + 1) * 176],
                        input,
                        COLS,
                    ));
                }
            }
            assert_relative_close(q4_output.as_slice(), &expected_q4, 3e-4, 0.02);
            assert_relative_close(q5_output.as_slice(), &expected_q5, 3e-4, 0.02);
        }
    }

    /// L-series M=16 tile: synthetic CPU-differential check for the DFlash
    /// verify/draft batch GEMM across all three K-quant dtypes.
    #[test]
    fn m16_n32_matches_cpu_reference() {
        let context = MetalContext::new().expect("Metal context");
        let kernels = MetalKernels::new(&context).expect("Muse primitive pipelines");
        let tokens = 16;
        let rows = 64;
        let cols = 256;
        let input_values = (0..tokens * cols)
            .map(|index| ((index as f32 * 0.017).cos() + (index % 23) as f32 * 0.003) * 0.2)
            .collect::<Vec<_>>();
        let input = GpuBuffer::from_f32(&context, &input_values).unwrap();

        // Q6_K generator: d at offset 208, int8 scales at 192..208.
        let q6_rows = |rows: usize| -> Vec<u8> {
            let mut bytes = vec![0u8; 210 * rows];
            for row in 0..rows {
                let block = &mut bytes[row * 210..(row + 1) * 210];
                for (index, quant) in block[..192].iter_mut().enumerate() {
                    *quant = ((index * 31 + row * 13 + 7) & 0xff) as u8;
                }
                for (index, scale) in block[192..208].iter_mut().enumerate() {
                    *scale = (((index * 11 + row * 5 + 3) & 0x3f) as u8) ^ 0x20;
                    // int8
                }
                block[208..210].copy_from_slice(&0x2c00u16.to_le_bytes()); // d = 0.0625
            }
            bytes
        };

        let cases: [(Vec<u8>, GgmlType, usize); 3] = [
            (quant_rows(144, rows, false), GgmlType::Q4_K, 144),
            (quant_rows(176, rows, true), GgmlType::Q5_K, 176),
            (q6_rows(rows), GgmlType::Q6_K, 210),
        ];
        for (weight_bytes, dtype, block_bytes) in cases {
            let weights = GpuBytes::from_bytes(&context, &weight_bytes).unwrap();
            let output = GpuBuffer::zeros(&context, tokens * rows).unwrap();
            let command = context.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            kernels.encode_quantized_matmul(
                encoder,
                weights.view(0, weights.len()).unwrap(),
                &input,
                &output,
                dtype,
                cols,
                rows,
                tokens,
            );
            encoder.end_encoding();
            command.commit();
            context
                .wait_for_completion(command, Duration::from_secs(30))
                .expect("Metal completion");

            let blocks_per_row = cols / 256;
            let mut expected = Vec::with_capacity(tokens * rows);
            for token in 0..tokens {
                let x = &input_values[token * cols..(token + 1) * cols];
                for row in 0..rows {
                    let row_bytes = &weight_bytes[row * block_bytes * blocks_per_row..]
                        [..block_bytes * blocks_per_row];
                    let dot = match dtype {
                        GgmlType::Q4_K => dot_q4_k_f32(row_bytes, x, cols),
                        GgmlType::Q5_K => dot_q5_k_f32(row_bytes, x, cols),
                        GgmlType::Q6_K => dot_q6_k_f32(row_bytes, x, cols),
                        _ => unreachable!(),
                    };
                    expected.push(dot);
                }
            }
            // Same half-staged arithmetic family as the accepted SGM tile;
            // the pinned mul_mm route lands the identical value on this
            // fixture, so the envelope is the family's half activation
            // staging, not the tile.
            assert_relative_close(output.as_slice(), &expected, 2e-2, 1.0);
        }
    }

    #[test]
    fn aligned_q4k_sgm_matches_cpu_reference() {
        let context = MetalContext::new().expect("Metal context");
        let kernels = MetalKernels::new(&context).expect("Muse primitive pipelines");
        let tokens = 32;
        let rows = 64;
        let cols = 256;
        let input_values = (0..tokens * cols)
            .map(|index| ((index as f32 * 0.013).sin() + (index % 29) as f32 * 0.002) * 0.25)
            .collect::<Vec<_>>();
        let input = GpuBuffer::from_f32(&context, &input_values).unwrap();
        let q4_bytes = quant_rows(144, rows, false);
        let q4_weights = GpuBytes::from_bytes(&context, &q4_bytes).unwrap();
        let output = GpuBuffer::zeros(&context, tokens * rows).unwrap();

        let command = context.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        kernels.encode_quantized_matmul(
            encoder,
            q4_weights.view(0, q4_weights.len()).unwrap(),
            &input,
            &output,
            GgmlType::Q4_K,
            cols,
            rows,
            tokens,
        );
        encoder.end_encoding();
        command.commit();
        context
            .wait_for_completion(command, Duration::from_secs(30))
            .expect("Metal completion");

        let mut expected = Vec::with_capacity(tokens * rows);
        for token in 0..tokens {
            let input = &input_values[token * cols..(token + 1) * cols];
            for row in 0..rows {
                expected.push(dot_q4_k_f32(
                    &q4_bytes[row * 144..(row + 1) * 144],
                    input,
                    cols,
                ));
            }
        }
        // The production gate allows max absolute logit error 0.5 and
        // relative target-NLL error 0.005. This primitive uses Ferrite's
        // accepted half-staged SGM arithmetic, so test against those declared
        // release bounds rather than the f32 upstream-kernel tolerance.
        assert_relative_close(output.as_slice(), &expected, 5e-3, 0.5);
    }

    #[test]
    fn q4k_embedding_gather_matches_cpu_reference() {
        let context = MetalContext::new().expect("Metal context");
        let kernels = MetalKernels::new(&context).expect("Muse primitive pipelines");
        let q4_bytes = quant_rows(144, 2, false);
        let weights = GpuBytes::from_bytes(&context, &q4_bytes).unwrap();
        let token_bytes = [1u32, 0u32]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        let tokens = GpuBytes::from_bytes(&context, &token_bytes).unwrap();
        let output = GpuBuffer::zeros(&context, 512).unwrap();

        let command = context.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        kernels.encode_embedding_q4k(
            encoder,
            weights.view(0, weights.len()).unwrap(),
            &tokens.view(0, tokens.len()).unwrap(),
            &output,
            256,
            2,
            2,
        );
        encoder.end_encoding();
        command.commit();
        context
            .wait_for_completion(command, Duration::from_secs(30))
            .expect("Metal completion");

        let mut row0 = vec![0.0; 256];
        let mut row1 = vec![0.0; 256];
        dequant_q4_k(&q4_bytes[..144], &mut row0);
        dequant_q4_k(&q4_bytes[144..], &mut row1);
        let expected = row1.into_iter().chain(row0).collect::<Vec<_>>();
        assert_close(output.as_slice(), &expected, 2e-5);
    }

    #[test]
    fn attention_uses_explicit_logical_and_physical_ring_origins() {
        const HEAD_DIM: usize = 128;
        const HEADS: usize = 4;
        const KV_HEADS: usize = 2;
        const CAPACITY: usize = 3;
        const POSITION: usize = 4;
        const ORIGIN_LOGICAL: usize = 2;
        const ORIGIN_PHYSICAL: usize = 2;
        const WINDOW: usize = 3;

        let context = MetalContext::new().expect("Metal context");
        let kernels = MetalKernels::new(&context).expect("Muse primitive pipelines");
        let query_values = (0..HEADS * HEAD_DIM)
            .map(|index| ((index as f32 * 0.013).sin() + (index % 7) as f32 * 0.005) * 0.1)
            .collect::<Vec<_>>();
        let query = GpuBuffer::from_f32(&context, &query_values).unwrap();
        let current_key_values = kv_row(POSITION, KV_HEADS, HEAD_DIM, true);
        let current_value_values = kv_row(POSITION, KV_HEADS, HEAD_DIM, false);
        let current_key = GpuBuffer::from_f32(&context, &current_key_values).unwrap();
        let current_value = GpuBuffer::from_f32(&context, &current_value_values).unwrap();

        let mut key_cache_values = vec![0.0; CAPACITY * KV_HEADS * HEAD_DIM];
        let mut value_cache_values = vec![0.0; CAPACITY * KV_HEADS * HEAD_DIM];
        for logical in ORIGIN_LOGICAL..POSITION {
            let physical = (ORIGIN_PHYSICAL + logical - ORIGIN_LOGICAL) % CAPACITY;
            let start = physical * KV_HEADS * HEAD_DIM;
            key_cache_values[start..start + KV_HEADS * HEAD_DIM]
                .copy_from_slice(&kv_row(logical, KV_HEADS, HEAD_DIM, true));
            value_cache_values[start..start + KV_HEADS * HEAD_DIM]
                .copy_from_slice(&kv_row(logical, KV_HEADS, HEAD_DIM, false));
        }
        let key_cache = half_buffer(&context, &key_cache_values);
        let value_cache = half_buffer(&context, &value_cache_values);
        let output = GpuBuffer::zeros(&context, HEADS * HEAD_DIM).unwrap();
        let partials = GpuBuffer::zeros(
            &context,
            HEADS * crate::decode::MAX_DECODE_SPLIT_WORKGROUPS * (2 + HEAD_DIM),
        )
        .unwrap();

        let command = context.queue.new_command_buffer();
        let store = command.new_compute_command_encoder();
        let current_physical = (ORIGIN_PHYSICAL + POSITION - ORIGIN_LOGICAL) % CAPACITY;
        kernels.encode_kv_store_f16(
            store,
            &current_key,
            &current_value,
            &key_cache,
            &value_cache,
            current_physical,
        );
        store.end_encoding();
        let attention = command.new_compute_command_encoder();
        kernels.encode_attention_decode_splitk_f16(
            attention,
            &query,
            &key_cache,
            &value_cache,
            &partials,
            &output,
            HEADS,
            KV_HEADS,
            HEAD_DIM,
            POSITION,
            CAPACITY,
            ORIGIN_LOGICAL,
            ORIGIN_PHYSICAL,
            WINDOW,
            1.0 / (HEAD_DIM as f32).sqrt(),
        );
        attention.end_encoding();
        command.commit();
        context
            .wait_for_completion(command, Duration::from_secs(30))
            .expect("Metal completion");

        let expected = scalar_ring_attention(
            &query_values,
            HEADS,
            KV_HEADS,
            HEAD_DIM,
            ORIGIN_LOGICAL..=POSITION,
            usize::MAX,
        );
        assert_relative_close(output.as_slice(), &expected, 2e-4, 2e-4);
    }

    #[test]
    fn prefill_attention_reads_prior_ring_and_current_batch_before_install() {
        const HEAD_DIM: usize = 128;
        const HEADS: usize = 4;
        const KV_HEADS: usize = 2;
        const CAPACITY: usize = 3;
        const START: usize = 3;
        const TOKENS: usize = 2;
        const OLD_ORIGIN_LOGICAL: usize = 1;
        const OLD_ORIGIN_PHYSICAL: usize = 2;

        let context = MetalContext::new().expect("Metal context");
        let kernels = MetalKernels::new(&context).expect("Muse primitive pipelines");
        let query_values = (0..TOKENS * HEADS * HEAD_DIM)
            .map(|index| ((index as f32 * 0.009).sin() + (index % 13) as f32 * 0.003) * 0.1)
            .collect::<Vec<_>>();
        let query = GpuBuffer::from_f32(&context, &query_values).unwrap();
        let current_key_values = (START..START + TOKENS)
            .flat_map(|logical| kv_row(logical, KV_HEADS, HEAD_DIM, true))
            .collect::<Vec<_>>();
        let current_value_values = (START..START + TOKENS)
            .flat_map(|logical| kv_row(logical, KV_HEADS, HEAD_DIM, false))
            .collect::<Vec<_>>();
        let current_key = GpuBuffer::from_f32(&context, &current_key_values).unwrap();
        let current_value = GpuBuffer::from_f32(&context, &current_value_values).unwrap();
        let mut key_cache_values = vec![0.0; CAPACITY * KV_HEADS * HEAD_DIM];
        let mut value_cache_values = vec![0.0; CAPACITY * KV_HEADS * HEAD_DIM];
        for logical in OLD_ORIGIN_LOGICAL..START {
            let physical = (OLD_ORIGIN_PHYSICAL + logical - OLD_ORIGIN_LOGICAL) % CAPACITY;
            let start = physical * KV_HEADS * HEAD_DIM;
            key_cache_values[start..start + KV_HEADS * HEAD_DIM]
                .copy_from_slice(&kv_row(logical, KV_HEADS, HEAD_DIM, true));
            value_cache_values[start..start + KV_HEADS * HEAD_DIM]
                .copy_from_slice(&kv_row(logical, KV_HEADS, HEAD_DIM, false));
        }
        let key_cache = half_buffer(&context, &key_cache_values);
        let value_cache = half_buffer(&context, &value_cache_values);
        let output = GpuBuffer::zeros(&context, TOKENS * HEADS * HEAD_DIM).unwrap();

        let command = context.queue.new_command_buffer();
        let attention = command.new_compute_command_encoder();
        kernels.encode_attention_prefill_f32(
            attention,
            &query,
            &current_key,
            &current_value,
            &key_cache,
            &value_cache,
            &output,
            TOKENS,
            HEADS,
            KV_HEADS,
            HEAD_DIM,
            START,
            CAPACITY,
            OLD_ORIGIN_LOGICAL,
            OLD_ORIGIN_PHYSICAL,
            2,
            3,
            1.0 / (HEAD_DIM as f32).sqrt(),
            false,
        );
        attention.end_encoding();
        command.commit();
        context
            .wait_for_completion(command, Duration::from_secs(30))
            .expect("Metal completion");

        let expected = (0..TOKENS)
            .flat_map(|token| {
                scalar_ring_attention(
                    &query_values[token * HEADS * HEAD_DIM..(token + 1) * HEADS * HEAD_DIM],
                    HEADS,
                    KV_HEADS,
                    HEAD_DIM,
                    START + token + 1 - 3..=START + token,
                    START,
                )
            })
            .collect::<Vec<_>>();
        assert_relative_close(output.as_slice(), &expected, 2e-4, 2e-4);
    }

    #[test]
    fn flash_attention_v2_matches_causal_scalar_for_every_partial_query_group() {
        const HEAD_DIM: usize = 128;
        const HEADS: usize = 32;
        const KV_HEADS: usize = 2;

        let context = MetalContext::new().expect("Metal context");
        let kernels = MetalKernels::new(&context).expect("Muse primitive pipelines");
        for tokens in 1..=8 {
            const CAPACITY: usize = 64;
            // FA2 stages Q, K, and V through f16.  Round the inputs before
            // both executions so this test isolates indexing/masking and the
            // matrix reduction rather than conversion error.
            let query_values = (0..tokens * HEADS * HEAD_DIM)
                .map(|index| {
                    let value = ((index as f32 * 0.009).sin() + (index % 13) as f32 * 0.003) * 0.1;
                    f16_to_f32(f32_to_f16_rne(value))
                })
                .collect::<Vec<_>>();
            let key_values = (0..tokens)
                .flat_map(|logical| kv_row(logical, KV_HEADS, HEAD_DIM, true))
                .collect::<Vec<_>>();
            let value_values = (0..tokens)
                .flat_map(|logical| kv_row(logical, KV_HEADS, HEAD_DIM, false))
                .collect::<Vec<_>>();
            let query = GpuBuffer::from_f32(&context, &query_values).unwrap();
            let key = GpuBuffer::from_f32(&context, &key_values).unwrap();
            let value = GpuBuffer::from_f32(&context, &value_values).unwrap();
            let key_cache = GpuHalfBuffer::zeros(&context, CAPACITY * KV_HEADS * HEAD_DIM).unwrap();
            let value_cache =
                GpuHalfBuffer::zeros(&context, CAPACITY * KV_HEADS * HEAD_DIM).unwrap();
            let output = GpuBuffer::zeros(&context, tokens * HEADS * HEAD_DIM).unwrap();

            let command = context.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            kernels.encode_kv_store_batch_f16(
                encoder,
                &key,
                &value,
                &key_cache,
                &value_cache,
                KV_HEADS * HEAD_DIM,
                tokens,
                0,
                tokens,
                0,
                CAPACITY,
                0,
                0,
                HEAD_DIM,
                true,
            );
            let cache_barrier: [&metal::ResourceRef; 2] = [key_cache.metal(), value_cache.metal()];
            encoder.memory_barrier_with_resources(&cache_barrier);
            kernels.encode_flash_attention_v2(
                encoder,
                &query,
                &key_cache,
                &value_cache,
                &output,
                tokens,
                HEADS,
                KV_HEADS,
                HEAD_DIM,
                0,
                CAPACITY,
                0,
                0,
                1.0 / (HEAD_DIM as f32).sqrt(),
                true,
            );
            encoder.end_encoding();
            command.commit();
            context
                .wait_for_completion(command, Duration::from_secs(30))
                .expect("Metal completion");

            let expected = (0..tokens)
                .flat_map(|token| {
                    scalar_ring_attention(
                        &query_values[token * HEADS * HEAD_DIM..(token + 1) * HEADS * HEAD_DIM],
                        HEADS,
                        KV_HEADS,
                        HEAD_DIM,
                        0..=token,
                        tokens,
                    )
                })
                .collect::<Vec<_>>();
            assert_relative_close(output.as_slice(), &expected, 1e-5, 1e-5);
        }
    }

    #[test]
    fn pinned_llama_vec_attention_honors_batch_row_offsets_and_causal_prefixes() {
        const HEAD_DIM: usize = 128;
        const HEADS: usize = 32;
        const KV_HEADS: usize = 2;
        const TOKENS: usize = 8;
        const CAPACITY: usize = 64;

        let context = MetalContext::new().expect("Metal context");
        let kernels = MetalKernels::new(&context).expect("Muse primitive pipelines");
        if !kernels.has_llama_flash_attn_vec() {
            eprintln!("skipped: set MUSER_GGML_METALLIB to the pinned llama.cpp metallib");
            return;
        }
        let query_values = (0..TOKENS * HEADS * HEAD_DIM)
            .map(|index| {
                let value = ((index as f32 * 0.011).sin() + (index % 17) as f32 * 0.002) * 0.1;
                f16_to_f32(f32_to_f16_rne(value))
            })
            .collect::<Vec<_>>();
        let key_values = (0..TOKENS)
            .flat_map(|logical| kv_row(logical, KV_HEADS, HEAD_DIM, true))
            .collect::<Vec<_>>();
        let value_values = (0..TOKENS)
            .flat_map(|logical| kv_row(logical, KV_HEADS, HEAD_DIM, false))
            .collect::<Vec<_>>();
        let query = GpuBuffer::from_f32(&context, &query_values).unwrap();
        let key = GpuBuffer::from_f32(&context, &key_values).unwrap();
        let value = GpuBuffer::from_f32(&context, &value_values).unwrap();
        let key_cache = GpuHalfBuffer::zeros(&context, CAPACITY * KV_HEADS * HEAD_DIM).unwrap();
        let value_cache = GpuHalfBuffer::zeros(&context, CAPACITY * KV_HEADS * HEAD_DIM).unwrap();
        let mask = GpuBytes::zeros(&context, CAPACITY * std::mem::size_of::<u16>()).unwrap();
        let pad = GpuBytes::zeros(
            &context,
            32 * (2 * KV_HEADS * HEAD_DIM + 1) * std::mem::size_of::<u16>(),
        )
        .unwrap();
        let tmp = GpuBuffer::zeros(&context, HEADS * 32 * (HEAD_DIM + 2)).unwrap();
        let output = GpuBuffer::zeros(&context, TOKENS * HEADS * HEAD_DIM).unwrap();

        let command = context.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        kernels.encode_kv_store_batch_f16(
            encoder,
            &key,
            &value,
            &key_cache,
            &value_cache,
            KV_HEADS * HEAD_DIM,
            TOKENS,
            0,
            TOKENS,
            0,
            CAPACITY,
            0,
            0,
            HEAD_DIM,
            true,
        );
        let cache_barrier: [&metal::ResourceRef; 2] = [key_cache.metal(), value_cache.metal()];
        encoder.memory_barrier_with_resources(&cache_barrier);
        for row in 0..TOKENS {
            kernels.encode_llama_flash_attn_decode_vec_f16(
                encoder,
                &query,
                &key_cache,
                &value_cache,
                &mask,
                &pad,
                &tmp,
                &output,
                HEADS,
                KV_HEADS,
                HEAD_DIM,
                row + 1,
                CAPACITY,
                0,
                0,
                false,
                1.0 / (HEAD_DIM as f32).sqrt(),
                true,
                row,
                row,
            );
        }
        encoder.end_encoding();
        command.commit();
        context
            .wait_for_completion(command, Duration::from_secs(30))
            .expect("Metal completion");

        let expected = (0..TOKENS)
            .flat_map(|token| {
                scalar_ring_attention(
                    &query_values[token * HEADS * HEAD_DIM..(token + 1) * HEADS * HEAD_DIM],
                    HEADS,
                    KV_HEADS,
                    HEAD_DIM,
                    0..=token,
                    TOKENS,
                )
            })
            .collect::<Vec<_>>();
        assert_relative_close(output.as_slice(), &expected, 2e-4, 2e-4);
    }

    #[test]
    fn cross_vendor_attention_scans_wrapped_swa_in_logical_order() {
        if std::env::var_os("MUSER_CROSS_VENDOR_QK").is_none() {
            eprintln!("not run: set MUSER_CROSS_VENDOR_QK=1");
            return;
        }
        const HEAD_DIM: usize = 128;
        const HEADS: usize = 32;
        const KV_HEADS: usize = 2;
        const CAPACITY: usize = 64;
        const ORIGIN: usize = 13;

        let context = MetalContext::new().expect("Metal context");
        let kernels = MetalKernels::new(&context).expect("Muse primitive pipelines");
        let query_values = (0..HEADS * HEAD_DIM)
            .map(|index| ((index as f32 * 0.011).sin() + (index % 17) as f32 * 0.002) * 0.1)
            .collect::<Vec<_>>();
        let canonical_key = (0..CAPACITY)
            .flat_map(|logical| kv_row(logical, KV_HEADS, HEAD_DIM, true))
            .collect::<Vec<_>>();
        let canonical_value = (0..CAPACITY)
            .flat_map(|logical| kv_row(logical, KV_HEADS, HEAD_DIM, false))
            .collect::<Vec<_>>();
        let row_elements = KV_HEADS * HEAD_DIM;
        let mut wrapped_key = vec![0.0f32; canonical_key.len()];
        let mut wrapped_value = vec![0.0f32; canonical_value.len()];
        for logical in 0..CAPACITY {
            let physical = (ORIGIN + logical) % CAPACITY;
            wrapped_key[physical * row_elements..(physical + 1) * row_elements].copy_from_slice(
                &canonical_key[logical * row_elements..(logical + 1) * row_elements],
            );
            wrapped_value[physical * row_elements..(physical + 1) * row_elements].copy_from_slice(
                &canonical_value[logical * row_elements..(logical + 1) * row_elements],
            );
        }

        let query = GpuBuffer::from_f32(&context, &query_values).unwrap();
        let canonical_key = half_buffer(&context, &canonical_key);
        let canonical_value = half_buffer(&context, &canonical_value);
        let wrapped_key = half_buffer(&context, &wrapped_key);
        let wrapped_value = half_buffer(&context, &wrapped_value);
        let canonical_output = GpuBuffer::zeros(&context, HEADS * HEAD_DIM).unwrap();
        let wrapped_output = GpuBuffer::zeros(&context, HEADS * HEAD_DIM).unwrap();
        let unused_pad = GpuBytes::zeros(&context, 1).unwrap();
        let unused_tmp = GpuBuffer::zeros(&context, 1).unwrap();

        let command = context.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        for (key, value, output, origin) in [
            (&canonical_key, &canonical_value, &canonical_output, 0),
            (&wrapped_key, &wrapped_value, &wrapped_output, ORIGIN),
        ] {
            kernels.encode_llama_flash_attn_decode_vec_f16(
                encoder,
                &query,
                key,
                value,
                &unused_pad,
                &unused_pad,
                &unused_tmp,
                output,
                HEADS,
                KV_HEADS,
                HEAD_DIM,
                CAPACITY,
                CAPACITY,
                0,
                origin,
                false,
                1.0 / (HEAD_DIM as f32).sqrt(),
                false,
                0,
                0,
            );
        }
        encoder.end_encoding();
        command.commit();
        context
            .wait_for_completion(command, Duration::from_secs(30))
            .expect("Metal completion");
        assert_eq!(
            canonical_output
                .as_slice()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            wrapped_output
                .as_slice()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn cross_vendor_attention_matches_the_fixed_32_lane_cuda_oracle() {
        if std::env::var_os("MUSER_CROSS_VENDOR_QK").is_none() {
            eprintln!("not run: set MUSER_CROSS_VENDOR_QK=1");
            return;
        }
        const TOKENS: usize = 5;
        const HEAD_DIM: usize = 128;
        const HEADS: usize = 32;
        const KV_HEADS: usize = 2;

        fn fixture(rows: usize, width: usize, multiplier: usize, scale: f32) -> Vec<f32> {
            (0..rows * width)
                .map(|index| {
                    let value = ((index * multiplier % 1009) as f32 - 503.0) * scale;
                    f16_to_f32(f32_to_f16_rne(value))
                })
                .collect()
        }

        let context = MetalContext::new().expect("Metal context");
        let kernels = MetalKernels::new(&context).expect("Muse primitive pipelines");
        let query = GpuBuffer::from_f32(
            &context,
            &fixture(TOKENS, HEADS * HEAD_DIM, 37, 0.000_976_562_5),
        )
        .unwrap();
        let key = half_buffer(
            &context,
            &fixture(TOKENS, KV_HEADS * HEAD_DIM, 41, 0.001_464_843_8),
        );
        let value = half_buffer(
            &context,
            &fixture(TOKENS, KV_HEADS * HEAD_DIM, 43, 0.001_953_125),
        );
        let gate = GpuBuffer::from_f32(
            &context,
            &fixture(TOKENS, HEADS * HEAD_DIM, 47, 0.003_906_25),
        )
        .unwrap();
        let output = GpuBuffer::zeros(&context, TOKENS * HEADS * HEAD_DIM).unwrap();
        let command = context.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        kernels.encode_flash_attention_v2(
            encoder,
            &query,
            &key,
            &value,
            &output,
            TOKENS,
            HEADS,
            KV_HEADS,
            HEAD_DIM,
            0,
            TOKENS,
            0,
            TOKENS,
            1.0 / (HEAD_DIM as f32).sqrt(),
            false,
        );
        let attention_barrier: [&metal::ResourceRef; 1] = [output.metal()];
        encoder.memory_barrier_with_resources(&attention_barrier);
        kernels.encode_sigmoid_gate(encoder, &output, &gate);
        encoder.end_encoding();
        command.commit();
        context
            .wait_for_completion(command, Duration::from_secs(30))
            .expect("cross-vendor attention fixture completion");

        let digest = Sha256::digest(
            output
                .as_slice()
                .iter()
                .flat_map(|value| f32_to_f16_rne(*value).to_le_bytes())
                .collect::<Vec<_>>(),
        );
        assert_eq!(
            format!("{digest:x}"),
            "4deae3e368fdeeb0108ee086b1c2172698bce807ff77a787b53ab92fd0e1feb9"
        );
    }

    #[test]
    fn cross_vendor_swiglu_matches_the_cuda_oracle() {
        if std::env::var_os("MUSER_CROSS_VENDOR_QK").is_none() {
            eprintln!("not run: set MUSER_CROSS_VENDOR_QK=1");
            return;
        }
        const ROWS: usize = 3;
        const WIDTH: usize = 19_968;
        let gate = (0..ROWS * WIDTH)
            .map(|index| {
                let value = ((index * 47 % 1009) as f32 - 503.0) * 0.003_906_25;
                f16_to_f32(f32_to_f16_rne(value))
            })
            .collect::<Vec<_>>();
        let up = (0..ROWS * WIDTH)
            .map(|index| {
                let value = ((index * 53 % 1013) as f32 - 506.0) * 0.001_953_125;
                f16_to_f32(f32_to_f16_rne(value))
            })
            .collect::<Vec<_>>();
        let context = MetalContext::new().expect("Metal context");
        let kernels = MetalKernels::new(&context).expect("Muse primitive pipelines");
        let gate = GpuBuffer::from_f32(&context, &gate).unwrap();
        let up = GpuBuffer::from_f32(&context, &up).unwrap();
        let command = context.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        kernels.encode_silu_mul(encoder, &gate, &up);
        encoder.end_encoding();
        command.commit();
        context
            .wait_for_completion(command, Duration::from_secs(30))
            .expect("Metal completion");
        let bytes = gate
            .as_slice()
            .iter()
            .flat_map(|value| f32_to_f16_rne(*value).to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(
            Sha256::digest(&bytes).as_slice(),
            &[
                0x6d, 0x00, 0xaa, 0x9e, 0x5a, 0x46, 0x1a, 0xa3, 0x8c, 0x74, 0x6c, 0x78, 0x8c, 0x27,
                0xa8, 0xa0, 0x94, 0xfd, 0x88, 0xd5, 0xdf, 0x8f, 0x44, 0x9d, 0xab, 0x90, 0xef, 0xda,
                0xb7, 0x1f, 0x6b, 0xc5,
            ]
        );
    }

    fn half_buffer(context: &MetalContext, values: &[f32]) -> GpuHalfBuffer {
        let bits = values
            .iter()
            .copied()
            .map(f32_to_f16_rne)
            .collect::<Vec<_>>();
        GpuHalfBuffer::from_bits(context, &bits).unwrap()
    }

    fn kv_row(logical: usize, kv_heads: usize, head_dim: usize, key: bool) -> Vec<f32> {
        (0..kv_heads * head_dim)
            .map(|index| {
                let head = index / head_dim;
                let dim = index % head_dim;
                let base = logical as f32 * 0.031 + head as f32 * 0.017;
                if key {
                    (base + dim as f32 * 0.0023).sin() * 0.4
                } else {
                    (base + dim as f32 * 0.0017).cos() * 0.7
                }
            })
            .collect()
    }

    fn scalar_ring_attention(
        query: &[f32],
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        logical_range: std::ops::RangeInclusive<usize>,
        half_before: usize,
    ) -> Vec<f32> {
        let heads_per_kv = heads / kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut output = vec![0.0; heads * head_dim];
        for head in 0..heads {
            let kv_head = head / heads_per_kv;
            let q = &query[head * head_dim..(head + 1) * head_dim];
            let rows = logical_range
                .clone()
                .map(|logical| {
                    let mut keys = kv_row(logical, kv_heads, head_dim, true);
                    let mut values = kv_row(logical, kv_heads, head_dim, false);
                    if logical < half_before {
                        keys.iter_mut()
                            .for_each(|value| *value = f16_to_f32(f32_to_f16_rne(*value)));
                        values
                            .iter_mut()
                            .for_each(|value| *value = f16_to_f32(f32_to_f16_rne(*value)));
                    }
                    let key = &keys[kv_head * head_dim..(kv_head + 1) * head_dim];
                    let value = values[kv_head * head_dim..(kv_head + 1) * head_dim].to_vec();
                    let score = q.iter().zip(key).map(|(a, b)| a * b).sum::<f32>() * scale;
                    (score, value)
                })
                .collect::<Vec<_>>();
            let max_score = rows
                .iter()
                .map(|(score, _)| *score)
                .fold(f32::NEG_INFINITY, f32::max);
            let denominator = rows
                .iter()
                .map(|(score, _)| (*score - max_score).exp())
                .sum::<f32>();
            for (score, value) in rows {
                let probability = (score - max_score).exp() / denominator;
                for dim in 0..head_dim {
                    output[head * head_dim + dim] += probability * value[dim];
                }
            }
        }
        output
    }

    /// IEEE-754 binary32 to binary16, round-to-nearest with ties to even.
    /// Test-only because production conversion is performed by Metal's
    /// `float` to `half` cast when a KV row is installed.
    fn f32_to_f16_rne(value: f32) -> u16 {
        let bits = value.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exponent = ((bits >> 23) & 0xff) as i32;
        let mantissa = bits & 0x7f_ffff;

        if exponent == 0xff {
            return if mantissa == 0 {
                sign | 0x7c00
            } else {
                sign | 0x7e00
            };
        }

        let half_exponent = exponent - 127 + 15;
        if half_exponent >= 31 {
            return sign | 0x7c00;
        }
        if half_exponent <= 0 {
            if half_exponent < -10 {
                return sign;
            }
            let significand = mantissa | 0x80_0000;
            let shift = (14 - half_exponent) as u32;
            let mut rounded = significand >> shift;
            let remainder = significand & ((1u32 << shift) - 1);
            let halfway = 1u32 << (shift - 1);
            if remainder > halfway || (remainder == halfway && rounded & 1 != 0) {
                rounded += 1;
            }
            return sign | rounded as u16;
        }

        let mut rounded = mantissa >> 13;
        let remainder = mantissa & 0x1fff;
        if remainder > 0x1000 || (remainder == 0x1000 && rounded & 1 != 0) {
            rounded += 1;
        }
        let mut half_exponent = half_exponent as u32;
        if rounded == 0x400 {
            rounded = 0;
            half_exponent += 1;
            if half_exponent == 31 {
                return sign | 0x7c00;
            }
        }
        sign | ((half_exponent as u16) << 10) | rounded as u16
    }

    fn quant_rows(row_bytes: usize, rows: usize, q5: bool) -> Vec<u8> {
        let mut bytes = vec![0u8; row_bytes * rows];
        for row in 0..rows {
            let block = &mut bytes[row * row_bytes..(row + 1) * row_bytes];
            block[0..2].copy_from_slice(&0x3800u16.to_le_bytes()); // d = 0.5
            block[2..4].copy_from_slice(&0x3400u16.to_le_bytes()); // dmin = 0.25
            for (index, scale) in block[4..16].iter_mut().enumerate() {
                *scale = ((index * 13 + row * 7 + 3) & 0xff) as u8;
            }
            let quant_start = if q5 { 48 } else { 16 };
            if q5 {
                for (index, high) in block[16..48].iter_mut().enumerate() {
                    *high = ((index * 29 + row * 17 + 5) & 0xff) as u8;
                }
            }
            for (index, quant) in block[quant_start..].iter_mut().enumerate() {
                *quant = ((index * 37 + row * 11 + 9) & 0xff) as u8;
            }
        }
        bytes
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "index {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
            );
        }
    }

    fn assert_relative_close(actual: &[f32], expected: &[f32], relative: f32, absolute: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let tolerance = absolute.max(relative * expected.abs());
            assert!(
                (actual - expected).abs() <= tolerance,
                "index {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
            );
        }
    }
}
