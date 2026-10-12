use metal::{ComputeCommandEncoderRef, MTLSize};

use super::{dispatch_1d, set_value, MetalKernels};
use crate::metal::buffer::{GpuBuffer, GpuByteView};

impl MetalKernels {
    /// Ferrite 897a6256b Q4_K SiLU gate+up route: four rows per threadgroup,
    /// two SIMD groups, with the input vector shared by both projections.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_ffn_q4k_gate_up_silu_4r2s(
        &self,
        encoder: &ComputeCommandEncoderRef,
        gate_weights: GpuByteView<'_>,
        up_weights: GpuByteView<'_>,
        input: &GpuBuffer,
        output: &GpuBuffer,
        intermediate_dim: usize,
        hidden_dim: usize,
    ) {
        let row_bytes = hidden_dim / 256 * 144;
        debug_assert_eq!(gate_weights.len(), intermediate_dim * row_bytes);
        debug_assert_eq!(up_weights.len(), gate_weights.len());
        debug_assert_eq!(input.len(), hidden_dim);
        debug_assert_eq!(output.len(), intermediate_dim);
        self.bind(encoder, "ffn_q4k_gate_up_silu_4r2s");
        encoder.set_buffer(0, Some(gate_weights.metal()), gate_weights.offset() as u64);
        encoder.set_buffer(1, Some(up_weights.metal()), up_weights.offset() as u64);
        encoder.set_buffer(2, Some(input.metal()), 0);
        encoder.set_buffer(3, Some(output.metal()), 0);
        set_value(encoder, 4, &(intermediate_dim as u32));
        set_value(encoder, 5, &(hidden_dim as u32));
        encoder.dispatch_thread_groups(
            MTLSize::new(intermediate_dim.div_ceil(4) as u64, 1, 1),
            MTLSize::new(64, 1, 1),
        );
    }

    pub fn encode_silu_mul(
        &self,
        encoder: &ComputeCommandEncoderRef,
        gate: &GpuBuffer,
        up: &GpuBuffer,
    ) {
        debug_assert_eq!(gate.len(), up.len());
        if std::env::var_os("MUSER_CROSS_VENDOR_QK").is_some() {
            debug_assert_eq!(gate.len() % 4, 0);
            encoder.set_compute_pipeline_state(&self.cross_vendor_swiglu);
        } else {
            self.bind(encoder, "muser_silu_mul_inplace");
        }
        encoder.set_buffer(0, Some(gate.metal()), 0);
        encoder.set_buffer(1, Some(up.metal()), 0);
        set_value(encoder, 2, &(gate.len() as u32));
        let threads = if std::env::var_os("MUSER_CROSS_VENDOR_QK").is_some() {
            gate.len() / 4
        } else {
            gate.len()
        };
        dispatch_1d(encoder, threads);
    }
}
