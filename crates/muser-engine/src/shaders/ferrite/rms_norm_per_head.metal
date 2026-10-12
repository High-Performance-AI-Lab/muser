// ── Per-head RMSNorm ───────────────────────────────────────────────────────
//
// out[h,d] = x[h,d] / sqrt(mean(x[h,:]²) + eps) * w[d]
//
// Normalizes each attention head independently using RMSNorm with shared
// weights. Used for Gemma 4 QK norms (per-head normalization of Q and K
// after projection, before RoPE).
//
// Dispatch: threadgroups = n_heads, threads_per_tg = head_dim (≤ 1024)
// Operates in-place when input == output.

#include <metal_stdlib>
using namespace metal;

kernel void rms_norm_per_head(
    device const float *input    [[ buffer(0) ]],  // [n_heads * head_dim]
    device const float *norm_w   [[ buffer(1) ]],  // [head_dim] (shared across heads)
    device float       *output   [[ buffer(2) ]],  // [n_heads * head_dim]
    constant uint      &head_dim [[ buffer(3) ]],
    constant float     &eps      [[ buffer(4) ]],
    uint h   [[ threadgroup_position_in_grid ]],
    uint tid [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]]
)
{
    const uint hd = head_dim;
    if (tid >= hd) return;

    const uint off = h * hd;

    // Load value and compute local sum-of-squares
    threadgroup float shared[1024];
    float val = input[off + tid];
    shared[tid] = val * val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction for sum of squares
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < hd) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_rms = 1.0f / sqrt(shared[0] / float(hd) + eps);
    output[off + tid] = val * inv_rms * norm_w[tid];
}

// ── Fused QKV per-head RMSNorm ──────────────────────────────────────────
//
// Applies per-head RMSNorm to Q, K, and V buffers in a single dispatch.
// Saves 2 dispatches per layer (3→1) for Gemma 4 QK norms.
//
// Grid layout: threadgroups [0, n_q_heads) → Q, [n_q_heads, n_q_heads + n_kv_heads) → K,
//              [n_q_heads + n_kv_heads, total) → V
// Dispatch: threadgroups = n_q_heads + 2*n_kv_heads, threads_per_tg = head_dim (≤ 1024)
// Operates in-place (input == output for each buffer).

kernel void rms_norm_per_head_qkv_fused(
    device float       *q_buf      [[ buffer(0) ]],  // [n_q_heads * head_dim]
    device float       *k_buf      [[ buffer(1) ]],  // [n_kv_heads * head_dim]
    device float       *v_buf      [[ buffer(2) ]],  // [n_kv_heads * head_dim]
    device const float *q_norm_w   [[ buffer(3) ]],  // [head_dim]
    device const float *k_norm_w   [[ buffer(4) ]],  // [head_dim]
    device const float *v_norm_w   [[ buffer(5) ]],  // [head_dim]
    constant uint      &head_dim   [[ buffer(6) ]],
    constant float     &eps        [[ buffer(7) ]],
    constant uint      &n_q_heads  [[ buffer(8) ]],  // b * n_heads
    constant uint      &n_kv_heads [[ buffer(9) ]],  // b * n_kv_heads
    uint h   [[ threadgroup_position_in_grid ]],
    uint tid [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]]
)
{
    const uint hd = head_dim;
    if (tid >= hd) return;

    // Determine which buffer and local head index
    device float *data;
    device const float *w;
    uint local_h;

    if (h < n_q_heads) {
        data = q_buf;
        w = q_norm_w;
        local_h = h;
    } else if (h < n_q_heads + n_kv_heads) {
        data = k_buf;
        w = k_norm_w;
        local_h = h - n_q_heads;
    } else {
        data = v_buf;
        w = v_norm_w;
        local_h = h - n_q_heads - n_kv_heads;
    }

    const uint off = local_h * hd;

    // Load value and compute local sum-of-squares
    threadgroup float shared[1024];
    float val = data[off + tid];
    shared[tid] = val * val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction for sum of squares
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride && tid + stride < hd) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_rms = 1.0f / sqrt(shared[0] / float(hd) + eps);
    data[off + tid] = val * inv_rms * w[tid];
}
