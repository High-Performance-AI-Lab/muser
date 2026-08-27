//! Multi-column quantized matvec for the DFlash verify batch.
//!
//! The default verify route dispatches one full matvec per token per
//! projection, so weight traffic scales with the block length. These pipelines
//! load each Q4_K/Q5_K/Q6_K weight block once per row and apply it to several
//! activation columns held in registers. Column `c` of a multi-column dispatch
//! performs exactly the arithmetic the single-column kernel performs for token
//! `c`, in the same block order, so greedy exactness is structurally preserved:
//! Q4_K and Q5_K are bitwise identical to the pinned per-token matvec, Q6_K is
//! not (see `multi_column_matches_pinned_per_token_dispatch_bitwise`).
//!
//! Default-off: nothing here is compiled unless `MUSER_MULTI_COL_VERIFY` is set,
//! so the release campaign can A/B the route without paying its compile cost.
//! `=1` routes the bitwise-exact dtypes, `=all` adds Q6_K.
//!
//! Measured on M3 Ultra, 6144x6144 Q4_K, cold weights (ms per projection,
//! multi-column vs the default route): b=2 0.144/0.134, b=4 0.153/0.172,
//! b=8 0.288/0.289, b=16 0.543/0.300 (the default takes the mul_mm tile from
//! b=9 up). The verify batch is not weight-traffic bound at these sizes, so the
//! win is confined to b=4.

use std::sync::OnceLock;

use metal::{
    CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, ComputePipelineStateRef,
    MTLLanguageVersion, MTLSize,
};

use super::{set_value, MetalKernels};
use crate::gguf::GgmlType;
use crate::metal::buffer::{GpuBuffer, GpuByteView};
use crate::metal::context::{MetalContext, MetalError};

/// Column counts that have a compiled specialization, widest first. Any block
/// length in 1..=16 is covered by a greedy decomposition over this set. Wider
/// specializations were measured on M3 Ultra (6144x6144 Q4_K): 8 and 16 columns
/// per thread lose more to register pressure and lost parallelism than they
/// save in weight traffic, so 4 is the widest that pays.
const COLUMN_SPECIALIZATIONS: [usize; 3] = [4, 2, 1];

/// Must match `MUSER_MC_NSG` and `MUSER_MC_NR0` in the shader, which in turn
/// match the pinned single-token matvec dispatch shape.
const SIMDGROUPS: usize = 2;
const ROWS_PER_SIMDGROUP: usize = 2;

/// Widest verify block the route accepts (`MAX_DFLASH_BLOCK`).
pub(crate) const MAX_COLUMNS: usize = 16;

/// Which dtypes the verify route hands to the multi-column kernels.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum MultiColMode {
    /// `MUSER_MULTI_COL_VERIFY=1`: only the dtypes whose multi-column output is
    /// bitwise identical to the pinned per-token matvec on this device.
    Exact,
    /// `MUSER_MULTI_COL_VERIFY=all`: adds Q6_K, which agrees with the pinned
    /// per-token matvec to a few ULP but not bitwise (see the module tests).
    All,
}

impl MultiColMode {
    fn admits(self, dtype: GgmlType) -> bool {
        match dtype {
            GgmlType::Q4_K | GgmlType::Q5_K => true,
            GgmlType::Q6_K => self == Self::All,
            _ => false,
        }
    }
}

pub(crate) fn multi_col_verify_mode() -> Option<MultiColMode> {
    static MODE: OnceLock<Option<MultiColMode>> = OnceLock::new();
    *MODE.get_or_init(|| {
        match std::env::var("MUSER_MULTI_COL_VERIFY")
            .unwrap_or_default()
            .as_str()
        {
            "1" => Some(MultiColMode::Exact),
            "all" => Some(MultiColMode::All),
            _ => None,
        }
    })
}

pub(crate) struct MultiColPipelines {
    q4k: [ComputePipelineState; COLUMN_SPECIALIZATIONS.len()],
    q5k: [ComputePipelineState; COLUMN_SPECIALIZATIONS.len()],
    q6k: [ComputePipelineState; COLUMN_SPECIALIZATIONS.len()],
}

impl MultiColPipelines {
    /// Compiles the multi-column library. Kept out of the always-on shader
    /// source so the default route pays neither the compile nor the PSO cost.
    pub(crate) fn new(context: &MetalContext) -> Result<Self, MetalError> {
        let options = CompileOptions::new();
        options.set_fast_math_enabled(true);
        options.set_language_version(MTLLanguageVersion::V3_1);
        let library = context
            .device
            .new_library_with_source(
                include_str!("../../shaders/ferrite/matvec_multicol.metal"),
                &options,
            )
            .map_err(MetalError::ShaderCompile)?;
        let group = |prefix: &str| -> Result<[ComputePipelineState; 3], MetalError> {
            let mut states = Vec::with_capacity(COLUMN_SPECIALIZATIONS.len());
            for columns in COLUMN_SPECIALIZATIONS {
                let name = format!("{prefix}_c{columns}");
                let function =
                    library
                        .get_function(&name, None)
                        .map_err(|message| MetalError::Pipeline {
                            name: name.clone(),
                            message,
                        })?;
                states.push(
                    context
                        .device
                        .new_compute_pipeline_state_with_function(&function)
                        .map_err(|message| MetalError::Pipeline {
                            name: name.clone(),
                            message,
                        })?,
                );
            }
            Ok(states.try_into().expect("one PSO per specialization"))
        };
        Ok(Self {
            q4k: group("muser_matvec_multicol_q4k")?,
            q5k: group("muser_matvec_multicol_q5k")?,
            q6k: group("muser_matvec_multicol_q6k")?,
        })
    }

    fn pipeline(&self, dtype: GgmlType, slot: usize) -> Option<&ComputePipelineStateRef> {
        let group = match dtype {
            GgmlType::Q4_K => &self.q4k,
            GgmlType::Q5_K => &self.q5k,
            GgmlType::Q6_K => &self.q6k,
            _ => return None,
        };
        Some(group[slot].as_ref())
    }

    /// Encodes `columns` activation columns starting at `col0`. Returns false
    /// when the shape or dtype has no multi-column route, leaving the caller
    /// on its existing path.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_columns(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weights: GpuByteView<'_>,
        input: &GpuBuffer,
        output: &GpuBuffer,
        dtype: GgmlType,
        n_in: usize,
        n_out: usize,
        col0: usize,
        columns: usize,
    ) -> bool {
        let block_bytes = match dtype {
            GgmlType::Q4_K => 144,
            GgmlType::Q5_K => 176,
            GgmlType::Q6_K => 210,
            _ => return false,
        };
        if !n_in.is_multiple_of(256) || n_out == 0 || columns == 0 || col0 + columns > MAX_COLUMNS {
            return false;
        }
        let row_bytes = n_in / 256 * block_bytes;
        debug_assert!(weights.len() >= row_bytes * n_out);
        debug_assert!(input.len() >= (col0 + columns) * n_in);
        debug_assert!(output.len() >= (col0 + columns) * n_out);

        let mut done = 0;
        while done < columns {
            let remaining = columns - done;
            let slot = COLUMN_SPECIALIZATIONS
                .iter()
                .position(|&width| width <= remaining)
                .expect("1 column is always a specialization");
            let width = COLUMN_SPECIALIZATIONS[slot];
            let Some(pipeline) = self.pipeline(dtype, slot) else {
                return false;
            };
            let args = MultiColArgs {
                ne00: n_in as u32,
                ne01: n_out as u32,
                nb01: row_bytes as u32,
                col0: (col0 + done) as u32,
            };
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(weights.metal()), weights.offset() as u64);
            encoder.set_buffer(1, Some(input.metal()), 0);
            encoder.set_buffer(2, Some(output.metal()), 0);
            set_value(encoder, 3, &args);
            encoder.dispatch_thread_groups(
                MTLSize::new(n_out.div_ceil(ROWS_PER_SIMDGROUP * SIMDGROUPS) as u64, 1, 1),
                MTLSize::new(32, SIMDGROUPS as u64, 1),
            );
            done += width;
        }
        true
    }
}

impl MetalKernels {
    /// Always-on exact multi-sequence decode projection. Q4_K and Q5_K share
    /// each weight-block load across the ready sequence columns while keeping
    /// the pinned single-row reduction order bit-for-bit. Q6_K deliberately
    /// falls back to repeated pinned matvecs because its separately compiled
    /// multi-column body differs by a few ULP.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_exact_multicol_matvec(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weights: GpuByteView<'_>,
        input: &GpuBuffer,
        output: &GpuBuffer,
        dtype: GgmlType,
        n_in: usize,
        n_out: usize,
        columns: usize,
    ) -> bool {
        if !MultiColMode::Exact.admits(dtype) {
            return false;
        }
        self.multicol().is_some_and(|pipelines| {
            pipelines.encode_columns(
                encoder, weights, input, output, dtype, n_in, n_out, 0, columns,
            )
        })
    }

    /// Multi-column verify route. Returns false when the route is disabled or
    /// the shape has no specialization, so the caller keeps its default path.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_multicol_matvec(
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
        let Some(pipelines) = self.multicol() else {
            return false;
        };
        if !multi_col_verify_mode().is_some_and(|mode| mode.admits(dtype)) {
            return false;
        }
        pipelines.encode_columns(
            encoder, weights, input, output, dtype, n_in, n_out, 0, tokens,
        )
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MultiColArgs {
    ne00: u32,
    ne01: u32,
    nb01: u32,
    col0: u32,
}

const _: () = assert!(std::mem::size_of::<MultiColArgs>() == 16);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::buffer::GpuBytes;
    use crate::quant::{dot_q4_k_f32, dot_q5_k_f32, dot_q6_k_f32, f32_to_f16_quant};
    use std::time::Duration;

    const DTYPES: [GgmlType; 3] = [GgmlType::Q4_K, GgmlType::Q5_K, GgmlType::Q6_K];
    const BLOCK_LENGTHS: [usize; 4] = [2, 5, 8, 16];

    fn block_bytes(dtype: GgmlType) -> usize {
        match dtype {
            GgmlType::Q4_K => 144,
            GgmlType::Q5_K => 176,
            GgmlType::Q6_K => 210,
            other => panic!("unsupported multi-column dtype {other:?}"),
        }
    }

    /// Deterministic pseudo-random blocks. Scale halves are kept small and
    /// finite so the comparison exercises the quant path, not NaN handling.
    fn random_weights(dtype: GgmlType, rows: usize, blocks: usize, seed: u64) -> Vec<u8> {
        let stride = block_bytes(dtype);
        let mut state = seed | 1;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as u32
        };
        let mut bytes = vec![0u8; rows * blocks * stride];
        for block in bytes.chunks_exact_mut(stride) {
            for byte in block.iter_mut() {
                *byte = (next() & 0xff) as u8;
            }
            let d = 0.002 + (next() % 128) as f32 * 0.0005;
            let dmin = 0.001 + (next() % 128) as f32 * 0.0003;
            match dtype {
                GgmlType::Q6_K => {
                    block[208..210].copy_from_slice(&f32_to_f16_quant(d).to_le_bytes());
                }
                _ => {
                    block[0..2].copy_from_slice(&f32_to_f16_quant(d).to_le_bytes());
                    block[2..4].copy_from_slice(&f32_to_f16_quant(dmin).to_le_bytes());
                }
            }
        }
        bytes
    }

    fn random_activations(count: usize, seed: u64) -> Vec<f32> {
        let mut state = seed | 1;
        (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 33) as f32 / (1u32 << 31) as f32 - 0.5) * 2.0
            })
            .collect()
    }

    fn cpu_reference(dtype: GgmlType, row: &[u8], activations: &[f32], n_in: usize) -> f32 {
        match dtype {
            GgmlType::Q4_K => dot_q4_k_f32(row, activations, n_in),
            GgmlType::Q5_K => dot_q5_k_f32(row, activations, n_in),
            GgmlType::Q6_K => dot_q6_k_f32(row, activations, n_in),
            other => panic!("unsupported multi-column dtype {other:?}"),
        }
    }

    struct Fixture {
        context: MetalContext,
        pipelines: MultiColPipelines,
    }

    impl Fixture {
        fn new() -> Self {
            let context = MetalContext::new().expect("Metal context");
            let pipelines = MultiColPipelines::new(&context).expect("multi-column pipelines");
            Self { context, pipelines }
        }
    }

    /// `n_out` deliberately is not a multiple of the 4 rows a threadgroup
    /// covers, so the trailing-row guard is exercised too.
    const N_IN: usize = 512;
    const N_OUT: usize = 13;

    #[test]
    fn multi_column_matches_per_column_dispatch_bitwise() {
        let fixture = Fixture::new();
        for dtype in DTYPES {
            let weights_bytes =
                random_weights(dtype, N_OUT, N_IN / 256, 0x5eed_1234 + dtype as u64);
            let weights = GpuBytes::from_bytes(&fixture.context, &weights_bytes).unwrap();
            for tokens in BLOCK_LENGTHS {
                let activations = random_activations(tokens * N_IN, 0xa5a5_0001 + tokens as u64);
                let input = GpuBuffer::from_f32(&fixture.context, &activations).unwrap();
                let multi = GpuBuffer::zeros(&fixture.context, tokens * N_OUT).unwrap();
                let single = GpuBuffer::zeros(&fixture.context, tokens * N_OUT).unwrap();

                let command = fixture.context.queue.new_command_buffer();
                let encoder = command.new_compute_command_encoder();
                let view = weights.view(0, weights.len()).unwrap();
                assert!(fixture
                    .pipelines
                    .encode_columns(encoder, view, &input, &multi, dtype, N_IN, N_OUT, 0, tokens,));
                for column in 0..tokens {
                    let view = weights.view(0, weights.len()).unwrap();
                    assert!(fixture.pipelines.encode_columns(
                        encoder, view, &input, &single, dtype, N_IN, N_OUT, column, 1,
                    ));
                }
                encoder.end_encoding();
                command.commit();
                fixture
                    .context
                    .wait_for_completion(command, Duration::from_secs(30))
                    .expect("Metal completion");

                for (index, (&wide, &narrow)) in
                    multi.as_slice().iter().zip(single.as_slice()).enumerate()
                {
                    assert_eq!(
                        wide.to_bits(),
                        narrow.to_bits(),
                        "{dtype:?} tokens={tokens} index={index}: {wide} vs {narrow}"
                    );
                }
            }
        }
    }

    #[test]
    fn multi_column_matches_cpu_reference() {
        let fixture = Fixture::new();
        for dtype in DTYPES {
            let stride = block_bytes(dtype) * (N_IN / 256);
            let weights_bytes =
                random_weights(dtype, N_OUT, N_IN / 256, 0x1357_9bdf + dtype as u64);
            let weights = GpuBytes::from_bytes(&fixture.context, &weights_bytes).unwrap();
            let tokens = 5;
            let activations = random_activations(tokens * N_IN, 0x2468_ace0);
            let input = GpuBuffer::from_f32(&fixture.context, &activations).unwrap();
            let output = GpuBuffer::zeros(&fixture.context, tokens * N_OUT).unwrap();

            let command = fixture.context.queue.new_command_buffer();
            let encoder = command.new_compute_command_encoder();
            let view = weights.view(0, weights.len()).unwrap();
            assert!(fixture
                .pipelines
                .encode_columns(encoder, view, &input, &output, dtype, N_IN, N_OUT, 0, tokens,));
            encoder.end_encoding();
            command.commit();
            fixture
                .context
                .wait_for_completion(command, Duration::from_secs(30))
                .expect("Metal completion");

            let actual = output.as_slice();
            for token in 0..tokens {
                let column = &activations[token * N_IN..(token + 1) * N_IN];
                for row in 0..N_OUT {
                    let expected = cpu_reference(
                        dtype,
                        &weights_bytes[row * stride..(row + 1) * stride],
                        column,
                        N_IN,
                    );
                    let value = actual[token * N_OUT + row];
                    let tolerance = 2e-3f32.max(expected.abs() * 2e-3);
                    assert!(
                        (value - expected).abs() <= tolerance,
                        "{dtype:?} token={token} row={row}: {value} vs {expected}"
                    );
                }
            }
        }
    }

    /// Parity with the route the verify path uses today. Q4_K and Q5_K are
    /// bitwise identical; Q6_K agrees only to a few ULP because the pinned
    /// metallib's Q6_K body is contracted differently by its own compilation,
    /// which is why `MultiColMode::Exact` keeps Q6_K on the per-token loop.
    /// Requires the pinned llama.cpp metallib; skipped without it.
    #[test]
    fn multi_column_matches_pinned_per_token_dispatch_bitwise() {
        let fixture = Fixture::new();
        let kernels = MetalKernels::new(&fixture.context).expect("Muse primitive pipelines");
        if kernels.ggml_matvec(GgmlType::Q4_K).is_none() {
            eprintln!("skipping: MUSER_GGML_METALLIB is unset");
            return;
        }
        if multi_col_verify_mode().is_some() {
            // `encode_quantized_matmul` would take the multi-column route too,
            // leaving nothing to compare against.
            eprintln!("skipping: MUSER_MULTI_COL_VERIFY replaces the per-token route");
            return;
        }
        for dtype in DTYPES {
            let weights_bytes =
                random_weights(dtype, N_OUT, N_IN / 256, 0x0f0f_2222 + dtype as u64);
            let weights = GpuBytes::from_bytes(&fixture.context, &weights_bytes).unwrap();
            for tokens in BLOCK_LENGTHS {
                let activations = random_activations(tokens * N_IN, 0xbeef_0000 + tokens as u64);
                let input = GpuBuffer::from_f32(&fixture.context, &activations).unwrap();
                let multi = GpuBuffer::zeros(&fixture.context, tokens * N_OUT).unwrap();
                let pinned = GpuBuffer::zeros(&fixture.context, tokens * N_OUT).unwrap();

                let command = fixture.context.queue.new_command_buffer();
                let encoder = command.new_compute_command_encoder();
                let view = weights.view(0, weights.len()).unwrap();
                assert!(fixture
                    .pipelines
                    .encode_columns(encoder, view, &input, &multi, dtype, N_IN, N_OUT, 0, tokens,));
                // The per-token loop the verify path runs today.
                kernels.encode_quantized_matmul(
                    encoder,
                    weights.view(0, weights.len()).unwrap(),
                    &input,
                    &pinned,
                    dtype,
                    N_IN,
                    N_OUT,
                    tokens.min(8),
                );
                encoder.end_encoding();
                command.commit();
                fixture
                    .context
                    .wait_for_completion(command, Duration::from_secs(30))
                    .expect("Metal completion");

                let compared = tokens.min(8) * N_OUT;
                for index in 0..compared {
                    let wide = multi.as_slice()[index];
                    let narrow = pinned.as_slice()[index];
                    if MultiColMode::Exact.admits(dtype) {
                        assert_eq!(
                            wide.to_bits(),
                            narrow.to_bits(),
                            "{dtype:?} tokens={tokens} index={index}: {wide} vs {narrow}"
                        );
                    } else {
                        // Q6_K: agreement is numeric, not bitwise. The dot
                        // products cancel from ~1e3 down to ~1e1, so guard the
                        // residual rather than the ULP count; a real defect is
                        // orders of magnitude larger than this.
                        let gap = (wide - narrow).abs();
                        assert!(
                            gap <= 1e-3 + 1e-5 * narrow.abs(),
                            "{dtype:?} tokens={tokens} index={index}: {wide} vs {narrow} (gap {gap})"
                        );
                    }
                }
            }
        }
    }
}
