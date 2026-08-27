// ── rms_norm_llamacpp_f32 ──────────────────────────────────────────────────
//
// Bit-exact port of llama.cpp's `kernel_rms_norm_mul_f32` (F=2, scalar) —
// RMSNorm fused with the per-element weight multiply that Ferrite's
// `encode_rms_norm` always performs.
//
// Source: ggml-org/llama.cpp
//   ggml/src/ggml-metal/ggml-metal.metal
//   `kernel_rms_norm_fuse_impl` lines 2914-2973, specialization
//   `kernel_rms_norm_mul_f32` (F=2) at line 2978.
//   MIT License — Copyright (c) 2023-2026 The ggml authors.
//   See llama.cpp/LICENSE.
//
// Ferrite's default `rms_norm` kernel (rmsnorm.metal:12) uses a 32-lane
// single-simdgroup pattern AND `rsqrt(mean + eps)`. Llama uses multi-
// simdgroup reduction AND `1.0f / sqrt(mean + eps)`. `rsqrt` is a
// Metal fast-math path that differs from `1/sqrt` by a few ULP per call —
// with RMSNorm running 3× per transformer layer × 28 layers × N decode
// tokens, those ULPs compound into knife-edge logit flips past ~50 tokens.
//
// This kernel replicates llama's accumulation order exactly so Ferrite's
// decode path can be numerically identical to llama.cpp run-to-run.
//
// Buffer contract:
//   buffer(0) x      : f32 input  [n]
//   buffer(1) weight : f32 weight [n]
//   buffer(2) out    : f32 output [n]
//   buffer(3) n      : uint32 (= args.ne00)
//   buffer(4) eps    : f32
//   threadgroup(0): 32 floats (NW) for multi-simdgroup reduction
//
// Dispatch:
//   threadgroups: (1, 1, 1)  — one TG per row; caller handles per-row loop
//   threads/tg:   (NW * NSG, 1, 1)  = (128, 1, 1) to match llama's default
//   threadgroup memory: NW * sizeof(float) = 128 bytes
//
// Attribution: ported from ggml-org/llama.cpp (MIT) ggml-metal.metal.

#include <metal_stdlib>
using namespace metal;

// Compile-time constants. Llama specializes kernel_rms_norm_fuse_impl with
// the simdgroup count via runtime arg; the default device path uses
// NSG=4 (128 threads/TG). That value is baked here because Ferrite always
// dispatches with this shape.
constant short kRMS_NW  = 32;   // Metal simd width (fixed for Apple GPUs)
constant short kRMS_NSG = 4;    // simdgroups per threadgroup

kernel void rms_norm_llamacpp_f32(
    device const float * x       [[ buffer(0) ]],
    device const float * weight  [[ buffer(1) ]],
    device       float * out     [[ buffer(2) ]],
    constant     uint  & n       [[ buffer(3) ]],
    constant     float & eps     [[ buffer(4) ]],
    threadgroup  float * shmem   [[ threadgroup(0) ]],
    ushort3  tpitg [[ thread_position_in_threadgroup ]],
    ushort3  ntg   [[ threads_per_threadgroup ]],
    ushort   sgitg [[ simdgroup_index_in_threadgroup ]],
    ushort   tiisg [[ thread_index_in_simdgroup ]])
{
    // Step 1 (llama line 2926-2928): simdgroup 0 initializes the reduction
    // shmem. All NW=32 slots are zeroed so the final simd_sum across 32
    // lanes reads 0 for simdgroup indices >= NSG.
    if (sgitg == 0) {
        shmem[tiisg] = 0.0f;
    }

    // Step 2 (llama line 2939-2944): parallel sum of x*x.
    // Each thread accumulates strided-by-ntg.x = NW*NSG = 128.
    // This f32 accumulation order is the point of the port.
    float sumf = 0.0f;
    for (uint i = uint(tpitg.x); i < n; i += uint(ntg.x)) {
        sumf += x[i] * x[i];
    }

    // Step 3 (llama line 2945): within-simdgroup sum.
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4 (llama line 2949-2951): lane 0 writes its simdgroup's sum
    // into shmem[sgitg].
    if (tiisg == 0) {
        shmem[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 5 (llama line 2955-2956): final cross-simdgroup reduction.
    // All 32 lanes read shmem[tiisg] → simd_sum. Lanes tiisg >= NSG see
    // 0.0f from the initial zero-fill, so the sum is exactly the NSG-way
    // total. This is THE critical accumulation-order match with llama.
    sumf = shmem[tiisg];
    sumf = simd_sum(sumf);

    // Step 6 (llama line 2958-2959): note llama uses 1.0f/sqrt(...), NOT
    // rsqrt(...). On Metal, rsqrt is a hardware-accelerated fast-math path
    // that differs from the IEEE-correct 1/sqrt by a few ULP. Ferrite's
    // default rms_norm uses rsqrt — that's the drift source this port fixes.
    const float mean  = sumf / float(n);
    const float scale = 1.0f / sqrt(mean + eps);

    // Step 7 (llama line 2967): write y[i] = (x[i] * scale) * weight[i].
    // Parentheses match llama exactly — (x*scale)*w, not x*(scale*w) or
    // x*w*scale, because multiplication associativity can flip a ULP.
    for (uint i = uint(tpitg.x); i < n; i += uint(ntg.x)) {
        out[i] = (x[i] * scale) * weight[i];
    }
}
