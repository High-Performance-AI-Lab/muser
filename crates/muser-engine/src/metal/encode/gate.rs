use metal::ComputeCommandEncoderRef;

use super::{dispatch_1d, set_value, MetalKernels};
use crate::metal::buffer::GpuBuffer;

impl MetalKernels {
    pub fn encode_sigmoid_gate(
        &self,
        encoder: &ComputeCommandEncoderRef,
        values: &GpuBuffer,
        gate: &GpuBuffer,
    ) {
        debug_assert_eq!(values.len(), gate.len());
        if std::env::var_os("MUSER_CROSS_VENDOR_QK").is_some() {
            encoder.set_compute_pipeline_state(&self.cross_vendor_sigmoid_gate);
        } else {
            self.bind(encoder, "sigmoid_gate_inplace");
        }
        encoder.set_buffer(0, Some(values.metal()), 0);
        encoder.set_buffer(1, Some(gate.metal()), 0);
        set_value(encoder, 2, &(values.len() as u32));
        dispatch_1d(encoder, values.len());
    }
}
