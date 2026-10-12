#include <metal_stdlib>
using namespace metal;

// Exact extraction of Ferrite's two-phase GPU greedy reduction from
// ferrite-metal-shaders/shaders/matmul_misc_postprocess.metal. Equal values
// retain the lower index, matching the scalar first-maximum convention.
kernel void argmax_f32_phase1(
    device const float* x           [[ buffer(0) ]],
    device       float* partial_val [[ buffer(1) ]],
    device       uint*  partial_idx [[ buffer(2) ]],
    constant     uint&  n           [[ buffer(3) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint lid  [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]])
{
    threadgroup float tg_val[1024];
    threadgroup uint tg_idx[1024];
    uint gid = tgid * 1024u + lid;
    float best_val = -INFINITY;
    uint best_idx = 0u;
    if (gid < n) {
        best_val = x[gid];
        best_idx = gid;
    }
    tg_val[lid] = best_val;
    tg_idx[lid] = best_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 512u; stride > 0u; stride >>= 1u) {
        if (lid < stride && lid + stride < tg_size && tg_val[lid + stride] > tg_val[lid]) {
            tg_val[lid] = tg_val[lid + stride];
            tg_idx[lid] = tg_idx[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) {
        partial_val[tgid] = tg_val[0];
        partial_idx[tgid] = tg_idx[0];
    }
}

kernel void argmax_f32_phase2(
    device const float* partial_val [[ buffer(0) ]],
    device const uint*  partial_idx [[ buffer(1) ]],
    device       uint*  result      [[ buffer(2) ]],
    constant     uint&  n_blocks    [[ buffer(3) ]],
    uint lid [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]])
{
    threadgroup float tg_val[1024];
    threadgroup uint tg_idx[1024];
    float best_val = -INFINITY;
    uint best_idx = 0u;
    if (lid < n_blocks) {
        best_val = partial_val[lid];
        best_idx = partial_idx[lid];
    }
    tg_val[lid] = best_val;
    tg_idx[lid] = best_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 512u; stride > 0u; stride >>= 1u) {
        if (lid < stride && lid + stride < tg_size && tg_val[lid + stride] > tg_val[lid]) {
            tg_val[lid] = tg_val[lid + stride];
            tg_idx[lid] = tg_idx[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) {
        result[0] = tg_idx[0];
    }
}

// Greedy serving variant.  The high bit of every partial index carries a
// fail-closed nonfinite flag; vocabulary indices are required to fit in the
// remaining 31 bits.  `excluded` is the request's EOG set for ignore-eos
// generation.  Masking happens only inside the reduction, so the retained
// target logits remain byte-for-byte unchanged for logprob/session uses.
kernel void greedy_argmax_f32_phase1(
    device const float* x           [[ buffer(0) ]],
    device       float* partial_val [[ buffer(1) ]],
    device       uint*  partial_idx [[ buffer(2) ]],
    constant     uint&  n           [[ buffer(3) ]],
    device const uint*  excluded    [[ buffer(4) ]],
    constant     uint&  n_excluded  [[ buffer(5) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint lid  [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]])
{
    threadgroup float tg_val[1024];
    threadgroup uint tg_idx[1024];
    uint gid = tgid * 1024u + lid;
    float best_val = -INFINITY;
    uint best_idx = 0u;
    bool invalid = false;
    if (gid < n) {
        best_val = x[gid];
        best_idx = gid;
        invalid = !isfinite(best_val);
        for (uint i = 0u; i < n_excluded; ++i) {
            if (excluded[i] == gid) {
                best_val = -INFINITY;
                break;
            }
        }
    }
    tg_val[lid] = best_val;
    tg_idx[lid] = best_idx | (invalid ? 0x80000000u : 0u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 512u; stride > 0u; stride >>= 1u) {
        if (lid < stride && lid + stride < tg_size) {
            uint invalid_bits = (tg_idx[lid] | tg_idx[lid + stride]) & 0x80000000u;
            if (tg_val[lid + stride] > tg_val[lid]) {
                tg_val[lid] = tg_val[lid + stride];
                tg_idx[lid] = tg_idx[lid + stride] & 0x7fffffffu;
            }
            tg_idx[lid] = (tg_idx[lid] & 0x7fffffffu) | invalid_bits;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) {
        partial_val[tgid] = tg_val[0];
        partial_idx[tgid] = tg_idx[0];
    }
}

kernel void greedy_argmax_f32_phase2(
    device const float* partial_val [[ buffer(0) ]],
    device const uint*  partial_idx [[ buffer(1) ]],
    device       uint*  result      [[ buffer(2) ]],
    constant     uint&  n_blocks    [[ buffer(3) ]],
    uint lid [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]])
{
    threadgroup float tg_val[1024];
    threadgroup uint tg_idx[1024];
    float best_val = -INFINITY;
    uint best_idx = 0u;
    if (lid < n_blocks) {
        best_val = partial_val[lid];
        best_idx = partial_idx[lid];
    }
    tg_val[lid] = best_val;
    tg_idx[lid] = best_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 512u; stride > 0u; stride >>= 1u) {
        if (lid < stride && lid + stride < tg_size) {
            uint invalid_bits = (tg_idx[lid] | tg_idx[lid + stride]) & 0x80000000u;
            if (tg_val[lid + stride] > tg_val[lid]) {
                tg_val[lid] = tg_val[lid + stride];
                tg_idx[lid] = tg_idx[lid + stride] & 0x7fffffffu;
            }
            tg_idx[lid] = (tg_idx[lid] & 0x7fffffffu) | invalid_bits;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0u) {
        result[0] = (tg_idx[0] & 0x80000000u) != 0u ? 0xffffffffu : tg_idx[0];
    }
}
