//! Standalone M=16 K-quant batch-GEMM microbenchmark (Stage B, L series).
//!
//! Runs the exact DFlash verify/draft projection shapes (M=16, f32
//! activations, GGUF K-quant weights) against candidate Metal kernels and the
//! retained SGM tile baseline, with a synthetic CPU-differential exactness
//! check per shape. No model is loaded; quant blocks are random but valid.
//!
//! Usage:
//!   muser-m16-bench [--kernels sgm,b16,r2] [--shapes 0,4,7] [--trials N]
//!                   [--no-check] [--tolerance-diff]

use std::time::Instant;

/// Pinned llama.cpp ggml mul_mm / mul_mv_ext argument layouts (see
/// crates/muser-engine/src/metal/encode/qkv.rs).
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

fn mm_args(n_in: usize, n_out: usize, block_bytes: usize) -> GgmlKargsMulMm {
    let row_bytes = (n_in / 256 * block_bytes) as u64;
    GgmlKargsMulMm {
        ne00: n_in as i32,
        ne02: 1,
        nb01: row_bytes,
        nb02: row_bytes * n_out as u64,
        nb03: row_bytes * n_out as u64,
        ne12: 1,
        nb10: 4,
        nb11: (n_in * 4) as u64,
        nb12: (n_in * 4 * M) as u64,
        nb13: (n_in * 4 * M) as u64,
        ne0: n_out as i32,
        ne1: M as i32,
        r2: 1,
        r3: 1,
    }
}

fn mve_args(n_in: usize, n_out: usize, block_bytes: usize) -> GgmlKargsMulMvExt {
    let row_bytes = (n_in / 256 * block_bytes) as u64;
    GgmlKargsMulMvExt {
        ne00: n_in as i32,
        ne01: n_out as i32,
        ne02: 1,
        nb00: block_bytes as u64,
        nb01: row_bytes,
        nb02: row_bytes * n_out as u64,
        nb03: row_bytes * n_out as u64,
        ne10: n_in as i32,
        ne11: M as i32,
        ne12: 1,
        nb10: 4,
        nb11: (n_in * 4) as u64,
        nb12: (n_in * 4 * M) as u64,
        nb13: (n_in * 4 * M) as u64,
        ne0: n_out as i32,
        ne1: M as i32,
        r2: 1,
        r3: 1,
    }
}

use metal::{
    CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLResourceOptions,
    MTLSize,
};
use muser_engine::quant::{dequant_q4_k, dequant_q5_k, dequant_q6_k};

const M: usize = 16;

#[derive(Clone, Copy, PartialEq)]
enum DType {
    Q4K,
    Q5K,
    Q6K,
}

impl DType {
    fn block_bytes(self) -> usize {
        match self {
            DType::Q4K => 144,
            DType::Q5K => 176,
            DType::Q6K => 210,
        }
    }
}

struct Shape {
    label: &'static str,
    dtype: DType,
    n_in: usize,
    n_out: usize,
    /// Dispatches per speculative cycle in the verify path / draft block.
    verify_mult: u32,
    draft_mult: u32,
}

const SHAPES: &[Shape] = &[
    Shape {
        label: "attn_q/gate 6656->4096 q4k",
        dtype: DType::Q4K,
        n_in: 6656,
        n_out: 4096,
        verify_mult: 104,
        draft_mult: 5,
    },
    Shape {
        label: "attn_k/v-q4k 6656->256 q4k",
        dtype: DType::Q4K,
        n_in: 6656,
        n_out: 256,
        verify_mult: 78,
        draft_mult: 0,
    },
    Shape {
        label: "attn_v-q6k 6656->256 q6k",
        dtype: DType::Q6K,
        n_in: 6656,
        n_out: 256,
        verify_mult: 26,
        draft_mult: 0,
    },
    Shape {
        label: "attn_output 4096->6656 q4k",
        dtype: DType::Q4K,
        n_in: 4096,
        n_out: 6656,
        verify_mult: 52,
        draft_mult: 5,
    },
    Shape {
        label: "ffn_gate/up 6656->19968 q4k",
        dtype: DType::Q4K,
        n_in: 6656,
        n_out: 19968,
        verify_mult: 104,
        draft_mult: 10,
    },
    Shape {
        label: "ffn_down-q4k 19968->6656 q4k",
        dtype: DType::Q4K,
        n_in: 19968,
        n_out: 6656,
        verify_mult: 26,
        draft_mult: 0,
    },
    Shape {
        label: "ffn_down-q6k 19968->6656 q6k",
        dtype: DType::Q6K,
        n_in: 19968,
        n_out: 6656,
        verify_mult: 26,
        draft_mult: 5,
    },
    Shape {
        label: "lm_head 6656->202048 q5k",
        dtype: DType::Q5K,
        n_in: 6656,
        n_out: 202048,
        verify_mult: 1,
        draft_mult: 0,
    },
    Shape {
        label: "draft.k 6656->1024 q4k",
        dtype: DType::Q4K,
        n_in: 6656,
        n_out: 1024,
        verify_mult: 0,
        draft_mult: 5,
    },
    Shape {
        label: "draft.v 6656->1024 q6k",
        dtype: DType::Q6K,
        n_in: 6656,
        n_out: 1024,
        verify_mult: 0,
        draft_mult: 5,
    },
    Shape {
        label: "draft.fc 33280->6656 q4k",
        dtype: DType::Q4K,
        n_in: 33280,
        n_out: 6656,
        verify_mult: 0,
        draft_mult: 1,
    },
];

const BASELINE_SOURCE: &str =
    include_str!("../../muser-engine/src/shaders/ferrite/batch_sgm_q4_aligned.metal");
const CANDIDATE_SOURCE: &str = include_str!("../shaders/m16_candidates.metal");

struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn byte(&mut self) -> u8 {
        (self.next() >> 32) as u8
    }
    fn f32(&mut self) -> f32 {
        // Uniform in [-1, 1).
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

fn f16_bits(value: f32) -> u16 {
    half::f16::from_f32(value).to_bits()
}

/// Random but valid K-quant superblocks: sane scales, random payloads.
fn gen_weights(dtype: DType, n_in: usize, n_out: usize, seed: u64) -> Vec<u8> {
    let n_sb = n_in / 256;
    let bb = dtype.block_bytes();
    let mut w = vec![0u8; n_out * n_sb * bb];
    let mut rng = XorShift(seed);
    for block in w.chunks_exact_mut(bb) {
        match dtype {
            DType::Q4K | DType::Q5K => {
                let d = f16_bits(0.02 + 0.01 * rng.f32().abs());
                let dmin = f16_bits(0.01 * rng.f32().abs());
                block[0..2].copy_from_slice(&d.to_le_bytes());
                block[2..4].copy_from_slice(&dmin.to_le_bytes());
                for b in block[4..].iter_mut() {
                    *b = rng.byte();
                }
            }
            DType::Q6K => {
                for b in block[..208].iter_mut() {
                    *b = rng.byte();
                }
                let d = f16_bits(0.02 + 0.01 * rng.f32().abs());
                block[208..210].copy_from_slice(&d.to_le_bytes());
            }
        }
    }
    w
}

/// CPU f32 reference: dequantize each output row, dot with each of M inputs.
/// Output layout is token-major: out[m * n_out + r].
fn cpu_reference(dtype: DType, weights: &[u8], x: &[f32], n_in: usize, n_out: usize) -> Vec<f32> {
    let n_sb = n_in / 256;
    let bb = dtype.block_bytes();
    let row_bytes = n_sb * bb;
    let dequant = match dtype {
        DType::Q4K => dequant_q4_k,
        DType::Q5K => dequant_q5_k,
        DType::Q6K => dequant_q6_k,
    };
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(16);
    let mut out = vec![0.0f32; M * n_out];
    let rows_per = n_out.div_ceil(threads);
    let out_ptr = out.as_mut_ptr() as usize;
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for t in 0..threads {
            let r_lo = t * rows_per;
            if r_lo >= n_out {
                break;
            }
            let r_hi = ((t + 1) * rows_per).min(n_out);
            let weights = &weights[r_lo * row_bytes..r_hi * row_bytes];
            let x: &[f32] = x;
            handles.push(scope.spawn(move || {
                let mut wrow = vec![0.0f32; n_in];
                for (i, block_row) in weights.chunks_exact(row_bytes).enumerate() {
                    let r = r_lo + i;
                    for (sb, block) in block_row.chunks_exact(bb).enumerate() {
                        dequant(block, &mut wrow[sb * 256..sb * 256 + 256]);
                    }
                    for m in 0..M {
                        let xr = &x[m * n_in..m * n_in + n_in];
                        let mut acc = 0.0f32;
                        for (a, b) in wrow.iter().zip(xr.iter()) {
                            acc += a * b;
                        }
                        // SAFETY: disjoint (m, r) cells per worker.
                        unsafe {
                            (*(out_ptr as *mut f32).add(m * n_out + r)) = acc;
                        }
                    }
                }
            }));
        }
        for h in handles {
            h.join().expect("reference worker");
        }
    });
    out
}

struct ErrorStats {
    max_abs: f32,
    argmax_flips: u32,
}

fn compare(got: &[f32], want: &[f32], n_out: usize) -> ErrorStats {
    let mut max_abs = 0.0f32;
    let mut sum = 0.0f64;
    let mut argmax_flips = 0u32;
    for m in 0..M {
        let gr = &got[m * n_out..(m + 1) * n_out];
        let wr = &want[m * n_out..(m + 1) * n_out];
        let gi = gr
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|p| p.0);
        let wi = wr
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|p| p.0);
        if gi != wi {
            argmax_flips += 1;
        }
        for (g, w) in gr.iter().zip(wr.iter()) {
            let d = (g - w).abs();
            if d > max_abs {
                max_abs = d;
            }
            sum += d as f64;
        }
    }
    let _ = sum;
    ErrorStats {
        max_abs,
        argmax_flips,
    }
}

type DispatchFn = fn(
    &ComputeCommandEncoderRef,
    &ComputePipelineState,
    &metal::Buffer,
    &metal::Buffer,
    &metal::Buffer,
    usize,
    usize,
);

fn set_struct<T>(encoder: &ComputeCommandEncoderRef, index: u64, value: &T) {
    encoder.set_bytes(
        index,
        std::mem::size_of::<T>() as u64,
        value as *const T as *const std::ffi::c_void,
    );
}

macro_rules! mm_dispatch {
    ($name:ident, $bytes:expr) => {
        fn $name(
            encoder: &ComputeCommandEncoderRef,
            pso: &ComputePipelineState,
            w: &metal::Buffer,
            x: &metal::Buffer,
            y: &metal::Buffer,
            n_in: usize,
            n_out: usize,
        ) {
            // Pinned llama mul_mm geometry: nr0=64, nr1=32, nsg=4 (bounds variant).
            let args = mm_args(n_in, n_out, $bytes);
            encoder.set_compute_pipeline_state(pso);
            set_struct(encoder, 0, &args);
            encoder.set_buffer(1, Some(w), 0);
            encoder.set_buffer(2, Some(x), 0);
            encoder.set_buffer(3, Some(y), 0);
            encoder.set_threadgroup_memory_length(0, 8192);
            encoder.dispatch_thread_groups(
                MTLSize::new(M.div_ceil(32) as u64, n_out.div_ceil(64) as u64, 1),
                MTLSize::new(32, 4, 1),
            );
        }
    };
}

mm_dispatch!(dispatch_mm_q4k, 144);
mm_dispatch!(dispatch_mm_q5k, 176);
mm_dispatch!(dispatch_mm_q6k, 210);

macro_rules! mve_dispatch {
    ($name:ident, $bytes:expr) => {
        fn $name(
            encoder: &ComputeCommandEncoderRef,
            pso: &ComputePipelineState,
            w: &metal::Buffer,
            x: &metal::Buffer,
            y: &metal::Buffer,
            n_in: usize,
            n_out: usize,
        ) {
            // mul_mv_ext, nsg=2, nxpsg=8 (nypsg=4, r0ptg=8), r1ptg=4 over 16 rows.
            let args = mve_args(n_in, n_out, $bytes);
            encoder.set_compute_pipeline_state(pso);
            set_struct(encoder, 0, &args);
            encoder.set_buffer(1, Some(w), 0);
            encoder.set_buffer(2, Some(x), 0);
            encoder.set_buffer(3, Some(y), 0);
            encoder.dispatch_thread_groups(
                MTLSize::new(n_out.div_ceil(8) as u64, M.div_ceil(4) as u64, 1),
                MTLSize::new(32, 2, 1),
            );
        }
    };
}

mve_dispatch!(dispatch_mve_q4k, 144);

fn dispatch_t32(
    encoder: &ComputeCommandEncoderRef,
    pso: &ComputePipelineState,
    w: &metal::Buffer,
    x: &metal::Buffer,
    y: &metal::Buffer,
    n_in: usize,
    n_out: usize,
) {
    encoder.set_compute_pipeline_state(pso);
    encoder.set_buffer(0, Some(w), 0);
    encoder.set_buffer(1, Some(x), 0);
    encoder.set_buffer(2, Some(y), 0);
    set_u32(encoder, 3, n_in as u32);
    set_u32(encoder, 4, n_out as u32);
    set_u32(encoder, 5, M as u32);
    encoder.set_threadgroup_memory_length(0, 5120);
    encoder.dispatch_thread_groups(
        MTLSize::new(n_out.div_ceil(64) as u64, 1, 1),
        MTLSize::new(128, 1, 1),
    );
}

fn dispatch_n32(
    encoder: &ComputeCommandEncoderRef,
    pso: &ComputePipelineState,
    w: &metal::Buffer,
    x: &metal::Buffer,
    y: &metal::Buffer,
    n_in: usize,
    n_out: usize,
) {
    encoder.set_compute_pipeline_state(pso);
    encoder.set_buffer(0, Some(w), 0);
    encoder.set_buffer(1, Some(x), 0);
    encoder.set_buffer(2, Some(y), 0);
    set_u32(encoder, 3, n_in as u32);
    set_u32(encoder, 4, n_out as u32);
    set_u32(encoder, 5, M as u32);
    encoder.set_threadgroup_memory_length(0, 6144);
    encoder.dispatch_thread_groups(
        MTLSize::new(n_out.div_ceil(32) as u64, 1, 1),
        MTLSize::new(128, 1, 1),
    );
}

fn dispatch_t64(
    encoder: &ComputeCommandEncoderRef,
    pso: &ComputePipelineState,
    w: &metal::Buffer,
    x: &metal::Buffer,
    y: &metal::Buffer,
    n_in: usize,
    n_out: usize,
) {
    encoder.set_compute_pipeline_state(pso);
    encoder.set_buffer(0, Some(w), 0);
    encoder.set_buffer(1, Some(x), 0);
    encoder.set_buffer(2, Some(y), 0);
    set_u32(encoder, 3, n_in as u32);
    set_u32(encoder, 4, n_out as u32);
    set_u32(encoder, 5, M as u32);
    encoder.set_threadgroup_memory_length(0, 10240);
    encoder.dispatch_thread_groups(
        MTLSize::new(n_out.div_ceil(64) as u64, 1, 1),
        MTLSize::new(128, 1, 1),
    );
}

fn dispatch_t128(
    encoder: &ComputeCommandEncoderRef,
    pso: &ComputePipelineState,
    w: &metal::Buffer,
    x: &metal::Buffer,
    y: &metal::Buffer,
    n_in: usize,
    n_out: usize,
) {
    encoder.set_compute_pipeline_state(pso);
    encoder.set_buffer(0, Some(w), 0);
    encoder.set_buffer(1, Some(x), 0);
    encoder.set_buffer(2, Some(y), 0);
    set_u32(encoder, 3, n_in as u32);
    set_u32(encoder, 4, n_out as u32);
    set_u32(encoder, 5, M as u32);
    encoder.set_threadgroup_memory_length(0, 20480);
    encoder.dispatch_thread_groups(
        MTLSize::new(n_out.div_ceil(64) as u64, 1, 1),
        MTLSize::new(128, 1, 1),
    );
}
mve_dispatch!(dispatch_mve_q5k, 176);
mve_dispatch!(dispatch_mve_q6k, 210);

fn dispatch_sgm(
    encoder: &ComputeCommandEncoderRef,
    pso: &ComputePipelineState,
    w: &metal::Buffer,
    x: &metal::Buffer,
    y: &metal::Buffer,
    n_in: usize,
    n_out: usize,
) {
    encoder.set_compute_pipeline_state(pso);
    encoder.set_buffer(0, Some(w), 0);
    encoder.set_buffer(1, Some(x), 0);
    encoder.set_buffer(2, Some(y), 0);
    set_u32(encoder, 3, n_in as u32);
    set_u32(encoder, 4, n_out as u32);
    set_u32(encoder, 5, M as u32);
    encoder.set_threadgroup_memory_length(0, 6144);
    encoder.dispatch_thread_groups(
        MTLSize::new(1, n_out.div_ceil(64) as u64, 1),
        MTLSize::new(128, 1, 1),
    );
}

fn dispatch_b16(
    encoder: &ComputeCommandEncoderRef,
    pso: &ComputePipelineState,
    w: &metal::Buffer,
    x: &metal::Buffer,
    y: &metal::Buffer,
    n_in: usize,
    n_out: usize,
) {
    encoder.set_compute_pipeline_state(pso);
    encoder.set_buffer(0, Some(w), 0);
    encoder.set_buffer(1, Some(x), 0);
    encoder.set_buffer(2, Some(y), 0);
    set_u32(encoder, 3, n_in as u32);
    set_u32(encoder, 4, n_out as u32);
    set_u32(encoder, 5, M as u32);
    encoder.set_threadgroup_memory_length(0, 6144);
    encoder.dispatch_thread_groups(
        MTLSize::new(1, n_out.div_ceil(64) as u64, 1),
        MTLSize::new(64, 1, 1),
    );
}

fn dispatch_r2(
    encoder: &ComputeCommandEncoderRef,
    pso: &ComputePipelineState,
    w: &metal::Buffer,
    x: &metal::Buffer,
    y: &metal::Buffer,
    n_in: usize,
    n_out: usize,
) {
    dispatch_r(encoder, pso, w, x, y, n_in, n_out, 2);
}

fn dispatch_r4(
    encoder: &ComputeCommandEncoderRef,
    pso: &ComputePipelineState,
    w: &metal::Buffer,
    x: &metal::Buffer,
    y: &metal::Buffer,
    n_in: usize,
    n_out: usize,
) {
    dispatch_r(encoder, pso, w, x, y, n_in, n_out, 4);
}

#[allow(clippy::too_many_arguments)]
fn dispatch_r(
    encoder: &ComputeCommandEncoderRef,
    pso: &ComputePipelineState,
    w: &metal::Buffer,
    x: &metal::Buffer,
    y: &metal::Buffer,
    n_in: usize,
    n_out: usize,
    rows_per_sg: usize,
) {
    encoder.set_compute_pipeline_state(pso);
    encoder.set_buffer(0, Some(w), 0);
    encoder.set_buffer(1, Some(x), 0);
    encoder.set_buffer(2, Some(y), 0);
    set_u32(encoder, 3, n_in as u32);
    set_u32(encoder, 4, n_out as u32);
    encoder.dispatch_thread_groups(
        MTLSize::new(n_out.div_ceil(8 * rows_per_sg) as u64, 1, 1),
        MTLSize::new(256, 1, 1),
    );
}

fn dispatch_dbgstage(
    encoder: &ComputeCommandEncoderRef,
    pso: &ComputePipelineState,
    w: &metal::Buffer,
    x: &metal::Buffer,
    y: &metal::Buffer,
    n_in: usize,
    n_out: usize,
) {
    encoder.set_compute_pipeline_state(pso);
    encoder.set_buffer(0, Some(w), 0);
    encoder.set_buffer(1, Some(x), 0);
    encoder.set_buffer(2, Some(y), 0);
    set_u32(encoder, 3, n_in as u32);
    set_u32(encoder, 4, n_out as u32);
    set_u32(encoder, 5, 0); // kt = 0 for the staging dump
    encoder.set_threadgroup_memory_length(0, 20480);
    encoder.dispatch_thread_groups(
        MTLSize::new(n_out.div_ceil(64) as u64, 1, 1),
        MTLSize::new(128, 1, 1),
    );
}

fn set_u32(encoder: &ComputeCommandEncoderRef, index: u64, value: u32) {
    encoder.set_bytes(
        index,
        std::mem::size_of::<u32>() as u64,
        &value as *const u32 as *const std::ffi::c_void,
    );
}

fn compile(device: &Device, source: &str, kernel: &str) -> ComputePipelineState {
    let options = CompileOptions::new();
    let library = device
        .new_library_with_source(source, &options)
        .unwrap_or_else(|e| panic!("compile library: {e}"));
    let function = library
        .get_function(kernel, None)
        .unwrap_or_else(|e| panic!("get {kernel}: {e}"));
    device
        .new_compute_pipeline_state_with_function(&function)
        .unwrap_or_else(|e| panic!("pso {kernel}: {e}"))
}

fn candidate_kernel_name(dtype: DType, variant: &str) -> Option<String> {
    let r = match (dtype, variant) {
        (DType::Q4K, "r2") => "m16_q4k_r2",
        (DType::Q5K, "r2") => "m16_q5k_r2",
        (DType::Q6K, "r2") => "m16_q6k_r2",
        (DType::Q4K, "r4") => "m16_q4k_r4",
        (DType::Q4K, "r2h") => "m16_q4k_r2h",
        (DType::Q4K, "r4h") => "m16_q4k_r4h",
        (DType::Q4K, "t128") => "m16_q4k_t128",
        (DType::Q4K, "dbgmac") => "m16_dbg_mac",
        (DType::Q4K, "dbgstage") => "m16_dbg_stage",
        (DType::Q4K, "dbgmark") => "m16_dbg_mark",
        (DType::Q4K, "t128p") => "m16_q4k_t128p",
        (DType::Q4K, "nomac") => "m16_q4k_t128_nomac",
        (DType::Q4K, "nodeq") => "m16_q4k_t128_nodeq",
        (DType::Q4K, "nobar") => "m16_q4k_t128_nobar",
        (DType::Q4K, "t128x") => "m16_q4k_t128x",
        (DType::Q4K, "t64") => "m16_q4k_t64",
        (DType::Q4K, "t32") => "m16_q4k_t32",
        (DType::Q4K, "n32") => "m16_q4k_n32",
        // k32n32 is deliberately unregistered: the variant has a staging bug.
        (DType::Q6K, "n32") => "m16_q6k_n32",
        (DType::Q5K, "n32") => "m16_q5k_n32",
        _ => return None,
    };
    Some(r.into())
}

#[allow(clippy::too_many_arguments)]
fn run_kernel(
    queue: &metal::CommandQueue,
    pso: &ComputePipelineState,
    dispatch: DispatchFn,
    w: &metal::Buffer,
    x: &metal::Buffer,
    y: &metal::Buffer,
    shape: &Shape,
    iters: usize,
    trials: usize,
) -> f64 {
    // Warmup.
    {
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        dispatch(enc, pso, w, x, y, shape.n_in, shape.n_out);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }
    let mut times = Vec::with_capacity(trials);
    for _ in 0..trials {
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        for _ in 0..iters {
            dispatch(enc, pso, w, x, y, shape.n_in, shape.n_out);
        }
        enc.end_encoding();
        let t0 = Instant::now();
        cb.commit();
        cb.wait_until_completed();
        times.push(t0.elapsed().as_secs_f64() / iters as f64);
    }
    times.sort_by(|a, b| a.total_cmp(b));
    times[times.len() / 2] * 1e3
}

fn print_row(
    shape: &Shape,
    kname: &str,
    ms: f64,
    max_abs: f64,
    flips: u32,
    weight_bytes: f64,
    macs: f64,
) {
    println!(
        "{:<34} {:<8} {:>10.4} {:>9.1} {:>9.2} {:>10.3e} {:>12}",
        shape.label,
        kname,
        ms,
        weight_bytes / (ms / 1e3) / 1e9,
        macs / (ms / 1e3) / 1e12,
        max_abs,
        flips
    );
}

fn main() {
    let mut kernels: Vec<String> = vec![
        "sgm".into(),
        "b16".into(),
        "r2".into(),
        "r4".into(),
        "r2h".into(),
        "r4h".into(),
    ];
    let mut shape_filter: Option<Vec<usize>> = None;
    let mut trials = 5usize;
    let mut check = true;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--kernels" => {
                i += 1;
                kernels = args[i].split(',').map(|s| s.trim().to_string()).collect();
            }
            "--shapes" => {
                i += 1;
                shape_filter = Some(
                    args[i]
                        .split(',')
                        .map(|s| s.trim().parse().unwrap())
                        .collect(),
                );
            }
            "--trials" => {
                i += 1;
                trials = args[i].parse().unwrap();
            }
            "--no-check" => check = false,
            "--diff" => {
                std::env::set_var("M16_DIFF", "1");
            }
            other => panic!("unknown argument {other}"),
        }
        i += 1;
    }

    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();

    let ggml_library = if kernels.iter().any(|k| k == "mm" || k == "mve") {
        let ggml_path = std::env::var("MUSER_GGML_METALLIB")
            .expect("MUSER_GGML_METALLIB must name the pinned llama.metallib for mm/mve kernels");
        Some(
            device
                .new_library_with_file(&ggml_path)
                .expect("ggml metallib"),
        )
    } else {
        None
    };
    let dtype_suffix = |d: DType| match d {
        DType::Q4K => "q4_K",
        DType::Q5K => "q5_K",
        DType::Q6K => "q6_K",
    };
    let ggml_pipeline =
        |library: &metal::Library, name: String, constants: metal::FunctionConstantValues| {
            let function = library
                .get_function(&name, Some(constants))
                .unwrap_or_else(|e| panic!("get {name}: {e}"));
            device
                .new_compute_pipeline_state_with_function(&function)
                .unwrap_or_else(|e| panic!("pso {name}: {e}"))
        };

    println!("muser-m16-bench device={} M={}", device.name(), M);
    println!(
        "{:<34} {:<8} {:>10} {:>9} {:>9} {:>10} {:>12}",
        "shape", "kernel", "ms", "GB/s", "TMAC/s", "max_abs", "argmax_flips"
    );

    let mut totals: std::collections::BTreeMap<String, (f64, f64)> =
        std::collections::BTreeMap::new();

    for (si, shape) in SHAPES.iter().enumerate() {
        if let Some(f) = &shape_filter {
            if !f.contains(&si) {
                continue;
            }
        }
        let weights = gen_weights(shape.dtype, shape.n_in, shape.n_out, 0x5EED + si as u64);
        let mut x = vec![0.0f32; M * shape.n_in];
        let mut rng = XorShift(0xAC1 + si as u64);
        for v in x.iter_mut() {
            *v = rng.f32();
        }
        let reference = if check {
            Some(cpu_reference(
                shape.dtype,
                &weights,
                &x,
                shape.n_in,
                shape.n_out,
            ))
        } else {
            None
        };

        let wbuf = device.new_buffer_with_data(
            weights.as_ptr() as *const std::ffi::c_void,
            weights.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let xbuf = device.new_buffer_with_data(
            x.as_ptr() as *const std::ffi::c_void,
            (x.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let xh: Vec<u16> = x
            .iter()
            .map(|v| half::f16::from_f32(*v).to_bits())
            .collect();
        let xhbuf = device.new_buffer_with_data(
            xh.as_ptr() as *const std::ffi::c_void,
            (xh.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let ybuf = device.new_buffer(
            (M * shape.n_out * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let weight_bytes = weights.len() as f64;
        let macs = 2.0 * shape.n_in as f64 * shape.n_out as f64 * M as f64;
        // Aim for ~2 GB of weight traffic per trial so small shapes still
        // amortize dispatch; cap iteration count for the big shapes.
        let iters = ((2.0e9 / weight_bytes).ceil() as usize).clamp(2, 512);

        for kname in &kernels {
            let (source, kernel_name, dispatch): (&str, String, DispatchFn) = match kname.as_str() {
                "mm" | "mve" => {
                    let library = ggml_library.as_ref().expect("ggml metallib");
                    let pso = if kname == "mm" {
                        let constants = metal::FunctionConstantValues::new();
                        let f = false;
                        constants.set_constant_value_at_index(
                            &f as *const bool as *const _,
                            metal::MTLDataType::Bool,
                            700,
                        );
                        let t = true;
                        constants.set_constant_value_at_index(
                            &t as *const bool as *const _,
                            metal::MTLDataType::Bool,
                            701,
                        );
                        for index in 702..=705u64 {
                            let one = 1i16;
                            constants.set_constant_value_at_index(
                                &one as *const i16 as *const _,
                                metal::MTLDataType::Short,
                                index,
                            );
                        }
                        ggml_pipeline(
                            library,
                            format!("kernel_mul_mm_{}_f32", dtype_suffix(shape.dtype)),
                            constants,
                        )
                    } else {
                        let constants = metal::FunctionConstantValues::new();
                        for (value, index) in
                            [(2i16, 600u64), (8, 601), (1, 602), (1, 603), (1, 604)]
                        {
                            constants.set_constant_value_at_index(
                                &value as *const i16 as *const _,
                                metal::MTLDataType::Short,
                                index,
                            );
                        }
                        ggml_pipeline(
                            library,
                            format!("kernel_mul_mv_ext_{}_f32_r1_4", dtype_suffix(shape.dtype)),
                            constants,
                        )
                    };
                    let dispatch: DispatchFn = match (kname.as_str(), shape.dtype) {
                        ("mm", DType::Q4K) => dispatch_mm_q4k,
                        ("mm", DType::Q5K) => dispatch_mm_q5k,
                        ("mm", DType::Q6K) => dispatch_mm_q6k,
                        ("mve", DType::Q4K) => dispatch_mve_q4k,
                        ("mve", DType::Q5K) => dispatch_mve_q5k,
                        ("mve", DType::Q6K) => dispatch_mve_q6k,
                        _ => unreachable!(),
                    };
                    // Run this kernel with the prebuilt pipeline, then continue.
                    let ms = run_kernel(
                        &queue, &pso, dispatch, &wbuf, &xbuf, &ybuf, shape, iters, trials,
                    );
                    let (mut max_abs, mut flips) = (f64::NAN, 0u32);
                    if let Some(reference) = &reference {
                        let cb = queue.new_command_buffer();
                        let enc = cb.new_compute_command_encoder();
                        dispatch(enc, &pso, &wbuf, &xbuf, &ybuf, shape.n_in, shape.n_out);
                        enc.end_encoding();
                        cb.commit();
                        cb.wait_until_completed();
                        let y = unsafe {
                            std::slice::from_raw_parts(
                                ybuf.contents() as *const f32,
                                M * shape.n_out,
                            )
                        };
                        let stats = compare(y, reference, shape.n_out);
                        max_abs = stats.max_abs as f64;
                        flips = stats.argmax_flips;
                    }
                    print_row(shape, kname, ms, max_abs, flips, weight_bytes, macs);
                    let entry = totals.entry(kname.clone()).or_default();
                    entry.0 += ms * shape.verify_mult as f64;
                    entry.1 += ms * shape.draft_mult as f64;
                    continue;
                }
                "sgm" => {
                    if shape.dtype != DType::Q4K {
                        continue;
                    }
                    (
                        BASELINE_SOURCE,
                        "matmul_q4k_batch_sgm_aligned".into(),
                        dispatch_sgm as _,
                    )
                }
                "b16" => {
                    if shape.dtype != DType::Q4K {
                        continue;
                    }
                    (
                        BASELINE_SOURCE,
                        "matmul_q4k_batch_sgm_b16_aligned".into(),
                        dispatch_b16 as _,
                    )
                }
                "r2" | "r4" | "r2h" | "r4h" | "t128" | "t128p" | "t128x" | "t64" | "t32"
                | "n32" | "nomac" | "nodeq" | "nobar" | "dbgmac" | "dbgstage" | "dbgmark" => {
                    let Some(name) = candidate_kernel_name(shape.dtype, kname) else {
                        continue;
                    };
                    let d: DispatchFn = match kname.as_str() {
                        "r4" | "r4h" => dispatch_r4,
                        "t64" => dispatch_t64,
                        "t32" => dispatch_t32,
                        "n32" => dispatch_n32,
                        "t128" | "t128p" | "t128x" | "nomac" | "nodeq" | "nobar" | "dbgmac" => {
                            dispatch_t128
                        }
                        "dbgstage" | "dbgmark" => dispatch_dbgstage,
                        _ => dispatch_r2,
                    };
                    (CANDIDATE_SOURCE, name, d)
                }
                other => panic!("unknown kernel {other}"),
            };
            let pso = compile(&device, source, &kernel_name);

            let xactive = if kname.ends_with('h') { &xhbuf } else { &xbuf };
            let (mut max_abs, mut flips) = (f64::NAN, 0u32);
            let ms = run_kernel(
                &queue, &pso, dispatch, &wbuf, xactive, &ybuf, shape, iters, trials,
            );
            if kname == "dbgmark" {
                let cb = queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                dispatch(enc, &pso, &wbuf, xactive, &ybuf, shape.n_in, shape.n_out);
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                let y = unsafe {
                    std::slice::from_raw_parts(ybuf.contents() as *const f32, M * shape.n_out)
                };
                let mut bad = 0u32;
                for k in 0..128usize {
                    for r in 0..64usize {
                        let want = ((k / 8) * 100 + (r % 8)) as f32;
                        if (y[k * 64 + r] - want).abs() > 0.5 {
                            bad += 1;
                        }
                    }
                }
                println!("  dbgmark bad={bad}");
            }
            if kname == "dbgstage" {
                // dump kt = 0 tile for threadgroup 0 is grid-wide; we compare
                // only tgid 0 (r0 = 0): rerun dispatch with B buffer = kt.
                let ktbuf = device.new_buffer_with_data(
                    &0u32 as *const u32 as *const std::ffi::c_void,
                    4,
                    MTLResourceOptions::StorageModeShared,
                );
                let _ = &ktbuf;
                let cb = queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                dispatch(enc, &pso, &wbuf, xactive, &ybuf, shape.n_in, shape.n_out);
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                let y = unsafe {
                    std::slice::from_raw_parts(ybuf.contents() as *const f32, M * shape.n_out)
                };
                // CPU dequant rows 0..63, k 0..127
                let n_sb = shape.n_in / 256;
                let bb = shape.dtype.block_bytes();
                let mut wrow = vec![0.0f32; shape.n_in];
                let mut bad = 0u32;
                for r in 0..64 {
                    let row = &weights[r * n_sb * bb..(r + 1) * n_sb * bb];
                    dequant_q4_k(&row[..bb], &mut wrow[..256]);
                    for k in 0..128 {
                        let got = y[k * 64 + r];
                        let want = wrow[k];
                        if (got - want).abs() > 0.01 * want.abs() + 0.001 {
                            if bad < 8 {
                                println!("  dbgstage r{r} k{k}: got={got:.4} want={want:.4}");
                            }
                            bad += 1;
                        }
                    }
                }
                println!("  dbgstage bad={bad}");
                for ktb in 0..16 {
                    let mut line = String::new();
                    for rtb in 0..8 {
                        let mut b = 0;
                        for k in ktb * 8..ktb * 8 + 8 {
                            for r in rtb * 8..rtb * 8 + 8 {
                                let got = y[k * 64 + r];
                                let want = {
                                    let row = &weights[r * n_sb * bb..(r + 1) * n_sb * bb];
                                    let mut wr = vec![0.0f32; 256];
                                    dequant_q4_k(&row[..bb], &mut wr);
                                    wr[k]
                                };
                                if (got - want).abs() > 0.01 * want.abs() + 0.001 {
                                    b += 1;
                                }
                            }
                        }
                        line.push_str(&if b == 0 {
                            "   .".to_string()
                        } else {
                            format!("{:4}", b)
                        });
                    }
                    println!("    kt{ktb:2}: {line}");
                }
            }
            if kname == "dbgmac" {
                let cb = queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                dispatch(enc, &pso, &wbuf, xactive, &ybuf, shape.n_in, shape.n_out);
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                let y = unsafe {
                    std::slice::from_raw_parts(ybuf.contents() as *const f32, M * shape.n_out)
                };
                // expect Y[m*n_out + r] = m*1000 + (r%8) for r < 64
                let mut bad = 0;
                for m in 0..M {
                    for r in 0..64 {
                        let want = (m * 64 + (r % 8)) as f32;
                        let got = y[m * shape.n_out + r];
                        if (got - want).abs() > 0.5 {
                            bad += 1;
                        }
                    }
                }
                println!(
                    "  dbgmac bad={bad} sample m1 r0..8: {:?}",
                    (0..8).map(|r| y[shape.n_out + r]).collect::<Vec<_>>()
                );
                // bad-cell map: rows m, cols r-blocks of 8
                for m in 0..M {
                    let mut line = String::new();
                    for rb in 0..8 {
                        let mut b = 0;
                        for r in rb * 8..rb * 8 + 8 {
                            let want = (m * 64 + (r % 8)) as f32;
                            if (y[m * shape.n_out + r] - want).abs() > 0.5 {
                                b += 1;
                            }
                        }
                        line.push_str(&format!("{:3}", b));
                    }
                    println!("    m{m:2}: {line}");
                }
            }
            if let Some(reference) = &reference {
                let cb = queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                dispatch(enc, &pso, &wbuf, xactive, &ybuf, shape.n_in, shape.n_out);
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();
                let y = unsafe {
                    std::slice::from_raw_parts(ybuf.contents() as *const f32, M * shape.n_out)
                };
                let stats = compare(y, reference, shape.n_out);
                max_abs = stats.max_abs as f64;
                flips = stats.argmax_flips;
                if std::env::var_os("M16_DIFF").is_some() {
                    for m in 0..1 {
                        for r in 0..16.min(shape.n_out) {
                            println!(
                                "  m{m} r{r}: got={:.4} want={:.4}",
                                y[m * shape.n_out + r],
                                reference[m * shape.n_out + r]
                            );
                        }
                    }
                }
            }
            print_row(shape, kname, ms, max_abs, flips, weight_bytes, macs);
            let entry = totals.entry(kname.clone()).or_default();
            entry.0 += ms * shape.verify_mult as f64;
            entry.1 += ms * shape.draft_mult as f64;
        }
    }

    println!();
    println!("cycle estimates (matmul-only, ms per speculative cycle):");
    for (kname, (verify, draft)) in &totals {
        println!("  {kname:<8} verify={verify:8.2} draft={draft:7.2}");
    }
}
