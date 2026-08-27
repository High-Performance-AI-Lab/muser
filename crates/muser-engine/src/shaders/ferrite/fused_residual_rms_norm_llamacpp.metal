// ── fused_residual_rms_norm_llamacpp ───────────────────────────────────────
//
// Fused residual-add + RMSNorm + weight-multiply, using llama.cpp's
// precision path:
//   - Multi-simdgroup reduction (NSG=4, 128 threads)
//   - `1.0f / sqrt(mean + eps)` — NOT `rsqrt(...)` (Metal's fast-math path
//     differs from IEEE-correct 1/sqrt by a few ULP per call).
//
// Op:
//   hidden[i] += delta[i];
//   normed[i]  = hidden[i] * (1/sqrt(mean + eps)) * weight[i];
//
// Ferrite's default fused kernel (rmsnorm.metal:fused_residual_rms_norm) uses
// `rsqrt()`. When `FERRITE_RMSNORM_BITEXACT=1`, `encode_fused_residual_rms_norm`
// routes here to match llama.cpp's decode-path numerics exactly.
//
// Source: ggml-org/llama.cpp ggml/src/ggml-metal/ggml-metal.metal
//   `kernel_rms_norm_fuse_impl` (F=2 specialization) — MIT license.
//
// Dispatch:
//   threadgroups: (1, 1, 1)
//   threads/tg:   (128, 1, 1)
//   threadgroup memory: 128 bytes (NW=32 floats)

#include <metal_stdlib>
using namespace metal;

constant short kFR_NW  = 32;
constant short kFR_NSG = 4;

kernel void fused_residual_rms_norm_llamacpp(
    device       float * hidden  [[ buffer(0) ]],  // rw: residual target
    device const float * delta   [[ buffer(1) ]],  // r:  correction
    device const float * weight  [[ buffer(2) ]],  // r:  RMSNorm scale
    device       float * normed  [[ buffer(3) ]],  // w:  output
    constant     uint  & n       [[ buffer(4) ]],
    constant     float & eps     [[ buffer(5) ]],
    threadgroup  float * shmem   [[ threadgroup(0) ]],
    ushort3  tpitg [[ thread_position_in_threadgroup ]],
    ushort3  ntg   [[ threads_per_threadgroup ]],
    ushort   sgitg [[ simdgroup_index_in_threadgroup ]],
    ushort   tiisg [[ thread_index_in_simdgroup ]])
{
    // Step 1: residual add, hidden += delta, and accumulate sum(hidden^2)
    // in the SAME pass. Each thread handles strided-by-128 indices.
    // NOTE: `hidden += delta` must be visible to the normalization step —
    // the hidden buffer is written device-wide here, then read back below.
    float sumf = 0.0f;
    for (uint i = uint(tpitg.x); i < n; i += uint(ntg.x)) {
        float h = hidden[i] + delta[i];
        hidden[i] = h;
        sumf += h * h;
    }

    // Step 2: within-simdgroup reduction.
    sumf = simd_sum(sumf);

    // Step 3: zero shmem (only sg0 does it — same pattern as rms_norm_llamacpp).
    if (sgitg == 0) {
        shmem[tiisg] = 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: each simdgroup's lane 0 writes its partial into shmem[sgitg].
    if (tiisg == 0) {
        shmem[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 5: cross-simdgroup reduction — 32 lanes read shmem, simd_sum.
    // Lanes >= NSG see the 0.0f from init, so the sum is exactly correct.
    sumf = shmem[tiisg];
    sumf = simd_sum(sumf);

    // Step 6: llama's 1.0f/sqrt path (NOT rsqrt).
    const float mean  = sumf / float(n);
    const float scale = 1.0f / sqrt(mean + eps);

    // Step 7: write normed[i] = (hidden[i] * scale) * weight[i].
    // Parenthesization matches llama exactly.
    for (uint i = uint(tpitg.x); i < n; i += uint(ntg.x)) {
        normed[i] = (hidden[i] * scale) * weight[i];
    }
}
