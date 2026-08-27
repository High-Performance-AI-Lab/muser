// ── silu_hadamard_batch ───────────────────────────────────────────────────
//
// In-place fused SiLU + Hadamard product:
//   gate[i] = silu(gate[i]) * up[i]
// where silu(x) = x / (1 + exp(-x)).
//
// Works on flat [B × intermediate_dim] arrays.
// Vectorized: processes 4 elements per thread via float4 loads/stores.
//
// dispatch: (ceil(total/4096), 1, 1) × (1024, 1, 1)
//   total must be the ELEMENT count (B × intermediate_dim).
//   Kernel handles tail elements when total is not divisible by 4.
//
kernel void silu_hadamard_batch(
    device       float* gate  [[ buffer(0) ]],  // [total] rw: gate → output
    device const float* up    [[ buffer(1) ]],  // [total]
    constant     uint&  total [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]])
{
    const uint base = gid * 4u;
    if (base + 3u < total) {
        // Fast path: full float4 load/store (aligned, coalesced)
        device       float4* g4 = (device       float4*)(gate + base);
        device const float4* u4 = (device const float4*)(up   + base);
        float4 x = *g4;
        // Match pinned llama.cpp's kernel_swiglu_f32 arithmetic exactly.
        float4 s = x / (1.0f + exp(-x));
        *g4 = s * (*u4);
    } else {
        // Tail path: scalar fallback for last 1–3 elements
        for (uint i = base; i < min(base + 4u, total); ++i) {
            float x = gate[i];
            gate[i] = (x / (1.0f + exp(-x))) * up[i];
        }
    }
}

// ── gelu_hadamard_batch ───────────────────────────────────────────────────
