use metal::{ComputeCommandEncoderRef, MTLSize};

use super::{dispatch_1d, set_value, MetalKernels};
use crate::metal::buffer::{GpuBuffer, GpuBytes};

#[repr(C)]
#[derive(Clone, Copy)]
struct GgmlMetalKargsUnary {
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
    slope: f32,
    scale: f32,
    bias: f32,
    val: f32,
    min: f32,
    max: f32,
}

const _: () = assert!(std::mem::size_of::<GgmlMetalKargsUnary>() == 120);

impl MetalKernels {
    fn encode_ggml_unary_inplace(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &metal::ComputePipelineStateRef,
        values: &GpuBuffer,
        count: usize,
        scale: f32,
    ) {
        assert!(count.is_multiple_of(4));
        let bytes = (count * std::mem::size_of::<f32>()) as u64;
        let args = GgmlMetalKargsUnary {
            ne00: (count / 4) as i32,
            ne01: 1,
            ne02: 1,
            ne03: 1,
            nb00: std::mem::size_of::<f32>() as u64,
            nb01: bytes,
            nb02: bytes,
            nb03: bytes,
            ne0: (count / 4) as i32,
            ne1: 1,
            ne2: 1,
            ne3: 1,
            nb0: std::mem::size_of::<f32>() as u64,
            nb1: bytes,
            nb2: bytes,
            nb3: bytes,
            slope: 0.0,
            scale,
            bias: 0.0,
            val: 0.0,
            min: 0.0,
            max: 0.0,
        };
        encoder.set_compute_pipeline_state(pipeline);
        set_value(encoder, 0, &args);
        encoder.set_buffer(1, Some(values.metal()), 0);
        encoder.set_buffer(2, Some(values.metal()), 0);
        let columns = count / 4;
        let threads = columns.min(256);
        encoder.dispatch_thread_groups(
            MTLSize::new(columns.div_ceil(threads) as u64, 1, 1),
            MTLSize::new(threads as u64, 1, 1),
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_argmax_f32_rows(
        &self,
        encoder: &ComputeCommandEncoderRef,
        values: &GpuBuffer,
        partial_values: &GpuBuffer,
        partial_indices: &GpuBuffer,
        results: &GpuBuffer,
        rows: usize,
        columns: usize,
    ) {
        let blocks = columns.div_ceil(1024);
        assert!(values.len() >= rows * columns);
        assert!(partial_values.len() >= rows * blocks);
        assert!(partial_indices.len() >= rows * blocks);
        assert!(results.len() >= rows);
        let producer_barrier: [&metal::ResourceRef; 1] = [values.metal()];
        encoder.memory_barrier_with_resources(&producer_barrier);
        for row in 0..rows {
            self.bind(encoder, "argmax_f32_phase1");
            encoder.set_buffer(0, Some(values.metal()), (row * columns * 4) as u64);
            encoder.set_buffer(1, Some(partial_values.metal()), (row * blocks * 4) as u64);
            encoder.set_buffer(2, Some(partial_indices.metal()), (row * blocks * 4) as u64);
            set_value(encoder, 3, &(columns as u32));
            encoder.dispatch_thread_groups(
                MTLSize::new(blocks as u64, 1, 1),
                MTLSize::new(1024, 1, 1),
            );
            let partial_barrier: [&metal::ResourceRef; 2] =
                [partial_values.metal(), partial_indices.metal()];
            encoder.memory_barrier_with_resources(&partial_barrier);
            self.bind(encoder, "argmax_f32_phase2");
            encoder.set_buffer(0, Some(partial_values.metal()), (row * blocks * 4) as u64);
            encoder.set_buffer(1, Some(partial_indices.metal()), (row * blocks * 4) as u64);
            encoder.set_buffer(2, Some(results.metal()), (row * 4) as u64);
            set_value(encoder, 3, &(blocks as u32));
            encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1024, 1, 1));
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_greedy_argmax_f32(
        &self,
        encoder: &ComputeCommandEncoderRef,
        values: &GpuBuffer,
        partial_values: &GpuBuffer,
        partial_indices: &GpuBuffer,
        result: &GpuBuffer,
        result_offset: usize,
        columns: usize,
        excluded: &GpuBytes,
        excluded_count: usize,
    ) {
        let blocks = columns.div_ceil(1024);
        assert!(columns < 0x8000_0000);
        assert!(values.len() >= columns);
        assert!(partial_values.len() >= blocks);
        assert!(partial_indices.len() >= blocks);
        assert!(result_offset < result.len());
        assert!(excluded.len() >= excluded_count * std::mem::size_of::<u32>());
        let producer_barrier: [&metal::ResourceRef; 1] = [values.metal()];
        encoder.memory_barrier_with_resources(&producer_barrier);
        self.bind(encoder, "greedy_argmax_f32_phase1");
        encoder.set_buffer(0, Some(values.metal()), 0);
        encoder.set_buffer(1, Some(partial_values.metal()), 0);
        encoder.set_buffer(2, Some(partial_indices.metal()), 0);
        set_value(encoder, 3, &(columns as u32));
        encoder.set_buffer(4, Some(excluded.metal()), 0);
        set_value(encoder, 5, &(excluded_count as u32));
        encoder.dispatch_thread_groups(MTLSize::new(blocks as u64, 1, 1), MTLSize::new(1024, 1, 1));
        let partial_barrier: [&metal::ResourceRef; 2] =
            [partial_values.metal(), partial_indices.metal()];
        encoder.memory_barrier_with_resources(&partial_barrier);
        self.bind(encoder, "greedy_argmax_f32_phase2");
        encoder.set_buffer(0, Some(partial_values.metal()), 0);
        encoder.set_buffer(1, Some(partial_indices.metal()), 0);
        encoder.set_buffer(2, Some(result.metal()), (result_offset * 4) as u64);
        set_value(encoder, 3, &(blocks as u32));
        encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1024, 1, 1));
    }

    pub fn encode_scale_softcap(
        &self,
        encoder: &ComputeCommandEncoderRef,
        logits: &GpuBuffer,
        scale: f32,
        softcap: f32,
    ) {
        self.encode_scale_softcap_count(encoder, logits, logits.len(), scale, softcap);
    }

    pub(crate) fn encode_scale_softcap_legacy(
        &self,
        encoder: &ComputeCommandEncoderRef,
        logits: &GpuBuffer,
        scale: f32,
        softcap: f32,
    ) {
        self.bind(encoder, "muser_scale_softcap_inplace");
        encoder.set_buffer(0, Some(logits.metal()), 0);
        set_value(encoder, 1, &(logits.len() as u32));
        set_value(encoder, 2, &scale);
        set_value(encoder, 3, &softcap);
        dispatch_1d(encoder, logits.len());
    }

    pub(crate) fn encode_scale_softcap_count(
        &self,
        encoder: &ComputeCommandEncoderRef,
        logits: &GpuBuffer,
        count: usize,
        scale: f32,
        softcap: f32,
    ) {
        assert!(count <= logits.len());
        if std::env::var_os("MUSER_CROSS_VENDOR_QK").is_some() {
            encoder.set_compute_pipeline_state(&self.cross_vendor_scale);
            encoder.set_buffer(0, Some(logits.metal()), 0);
            set_value(encoder, 1, &(count as u32));
            set_value(encoder, 2, &scale);
            dispatch_1d(encoder, count);
            if softcap > 0.0 {
                let scaled_barrier: [&metal::ResourceRef; 1] = [logits.metal()];
                encoder.memory_barrier_with_resources(&scaled_barrier);
                let inverse_softcap = 1.0f32 / softcap;
                encoder.set_compute_pipeline_state(&self.cross_vendor_scale);
                encoder.set_buffer(0, Some(logits.metal()), 0);
                set_value(encoder, 1, &(count as u32));
                set_value(encoder, 2, &inverse_softcap);
                dispatch_1d(encoder, count);

                let normalized_barrier: [&metal::ResourceRef; 1] = [logits.metal()];
                encoder.memory_barrier_with_resources(&normalized_barrier);
                encoder.set_compute_pipeline_state(&self.cross_vendor_tanh);
                encoder.set_buffer(0, Some(logits.metal()), 0);
                set_value(encoder, 1, &(count as u32));
                dispatch_1d(encoder, count);

                let tanh_barrier: [&metal::ResourceRef; 1] = [logits.metal()];
                encoder.memory_barrier_with_resources(&tanh_barrier);
                encoder.set_compute_pipeline_state(&self.cross_vendor_scale);
                encoder.set_buffer(0, Some(logits.metal()), 0);
                set_value(encoder, 1, &(count as u32));
                set_value(encoder, 2, &softcap);
                dispatch_1d(encoder, count);
            }
            return;
        }
        if let (Some(scale_pipeline), Some(tanh_pipeline)) =
            (&self.ggml_unary_scale, &self.ggml_unary_tanh)
        {
            if !count.is_multiple_of(4) {
                self.bind(encoder, "muser_scale_softcap_inplace");
                encoder.set_buffer(0, Some(logits.metal()), 0);
                set_value(encoder, 1, &(count as u32));
                set_value(encoder, 2, &scale);
                set_value(encoder, 3, &softcap);
                dispatch_1d(encoder, count);
                return;
            }
            // Match pinned llama.cpp's graph literally: the LM head is
            // followed by four independently published unary nodes.  The
            // previous combined kernel used a different tanh implementation
            // and expression tree, so equal pre-softcap logits did not yield
            // equal public bytes.
            self.encode_ggml_unary_inplace(encoder, scale_pipeline, logits, count, scale);
            if softcap > 0.0 {
                for (pipeline, factor) in [
                    (scale_pipeline.as_ref(), 1.0f32 / softcap),
                    (tanh_pipeline.as_ref(), 0.0),
                    (scale_pipeline.as_ref(), softcap),
                ] {
                    let barrier: [&metal::ResourceRef; 1] = [logits.metal()];
                    encoder.memory_barrier_with_resources(&barrier);
                    self.encode_ggml_unary_inplace(encoder, pipeline, logits, count, factor);
                }
            }
            return;
        }
        self.bind(encoder, "muser_scale_softcap_inplace");
        encoder.set_buffer(0, Some(logits.metal()), 0);
        set_value(encoder, 1, &(count as u32));
        set_value(encoder, 2, &scale);
        set_value(encoder, 3, &softcap);
        dispatch_1d(encoder, count);
    }
}
