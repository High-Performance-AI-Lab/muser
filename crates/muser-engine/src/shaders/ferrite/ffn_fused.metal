#include <metal_stdlib>
using namespace metal;

constant uint FFN_Q4K_ROWS_PER_TG = 2;

// ── Function constants for compile-time model dimensions ────────────────────
// When defined via MTLFunctionConstantValues the compiler can unroll the
// main Q4K block loop and eliminate bounds checks.
constant uint FC_FFN_COLS [[function_constant(30)]];
constant uint FC_FFN_IDIM [[function_constant(31)]];
constant bool HAS_FC_FFN_COLS = is_function_constant_defined(FC_FFN_COLS);
constant bool HAS_FC_FFN_IDIM = is_function_constant_defined(FC_FFN_IDIM);

// ── Function constant for FFN activation (SiLU vs GELU) ─────────────────────
// 0 = SiLU (default), 1 = GELU
constant uint FC_FFN_ACTIVATION [[function_constant(32)]];
constant bool HAS_FC_FFN_ACTIVATION = is_function_constant_defined(FC_FFN_ACTIVATION);

// Activation helper — compiled away at pipeline creation time (zero runtime cost)
inline float apply_activation(float g, float up) {
    if (HAS_FC_FFN_ACTIVATION && FC_FFN_ACTIVATION == 1u) {
        // GELU: 0.5 * g * (1 + tanh(sqrt(2/π) * (g + 0.044715 * g³)))
        const float SQRT_2_OVER_PI = 0.7978845608f;
        float inner = SQRT_2_OVER_PI * (g + 0.044715f * g * g * g);
        return 0.5f * g * (1.0f + precise::tanh(inner)) * up;
    } else {
        // SiLU (default): g * sigmoid(g) * up
        return (g / (1.0f + exp(-g))) * up;
    }
}

// ── ffn_gate_up_silu ───────────────────────────────────────────────────────
//
// Fused SwiGLU FFN intermediate: out[j] = silu(W_gate[j] · x) * (W_up[j] · x)
//
// This is the inner-most hot path in Qwen2/LLaMA2 FFN layers.
// By fusing gate + up projections into one kernel we:
//   1. Read x only ONCE per row pair (vs twice with separate kernels).
//   2. Avoid materialising intermediate gate[] and up[] buffers.
//   3. Save one full round-trip to GPU memory for the hidden-state writes.
//
// Model dimensions (0.5B):  h_dim = 896,  i_dim = 4864
// dispatch_thread_groups( (i_dim, 1, 1), (32, 1, 1) )
//
kernel void ffn_gate_up_silu(
    device const float* W_gate [[ buffer(0) ]],   // [i_dim × h_dim]
    device const float* W_up   [[ buffer(1) ]],   // [i_dim × h_dim]
    device const float* x      [[ buffer(2) ]],   // [h_dim]
    device       float* out    [[ buffer(3) ]],   // [i_dim]
    constant     uint&  h_dim  [[ buffer(4) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint lid  [[ thread_index_in_simdgroup ]])
{
    uint j = tgid;   // output row index
    device const float* gate_row = W_gate + j * h_dim;
    device const float* up_row   = W_up   + j * h_dim;

    float gate_acc = 0.0f;
    float up_acc   = 0.0f;

    // Each of 32 lanes handles a strided chunk — both gate and up in one loop
    for (uint c = lid; c < h_dim; c += 32) {
        float xc = x[c];
        gate_acc += gate_row[c] * xc;
        up_acc   += up_row[c]   * xc;
    }

    gate_acc = simd_sum(gate_acc);
    up_acc   = simd_sum(up_acc);

    if (lid == 0) {
        // SiLU(g) = g * sigmoid(g) = g / (1 + exp(-g))
        float silu_g = gate_acc * (1.0f / (1.0f + exp(-gate_acc)));
        out[j] = silu_g * up_acc;
    }
}

// ── ffn_q8_gate_up_silu ───────────────────────────────────────────────────
//
// Fused SwiGLU FFN intermediate for Q8_0 weight matrices.
//   out[j] = silu(W_gate[j] · x) * (W_up[j] · x)
//
// Q8_0 block layout (34 bytes):  [d: f16][qs: 32 × i8]
// decode: float(qs[k]) * float(d)
//
// dispatch_thread_groups( (i_dim, 1, 1), (32, 1, 1) )
//
struct q8_block_ffn { half d; char qs[32]; };

kernel void ffn_q8_gate_up_silu(
    device const q8_block_ffn* W_gate [[ buffer(0) ]],   // [i_dim × n_blocks]
    device const q8_block_ffn* W_up   [[ buffer(1) ]],   // [i_dim × n_blocks]
    device const float*        x      [[ buffer(2) ]],   // [h_dim]
    device       float*        out    [[ buffer(3) ]],   // [i_dim]
    constant     uint&         cols   [[ buffer(4) ]],   // h_dim (must be % 32 == 0)
    uint tgid [[ threadgroup_position_in_grid ]],
    uint lid  [[ thread_index_in_simdgroup ]])
{
    const uint n_blocks = cols / 32;
    device const q8_block_ffn* gate_row = W_gate + tgid * n_blocks;
    device const q8_block_ffn* up_row   = W_up   + tgid * n_blocks;

    float gate_acc = 0.0f;
    float up_acc   = 0.0f;

    // Each of 32 lanes handles a strided subset of blocks
    for (uint b = lid; b < n_blocks; b += 32) {
        float scale_g = float(gate_row[b].d);
        float scale_u = float(up_row[b].d);
        uint x_off = b * 32;
        float local_gate = 0.0f;
        float local_up   = 0.0f;
        for (uint k = 0; k < 32; ++k) {
            float xk = x[x_off + k];
            local_gate += float(gate_row[b].qs[k]) * xk;
            local_up   += float(up_row[b].qs[k])   * xk;
        }
        gate_acc += local_gate * scale_g;
        up_acc   += local_up   * scale_u;
    }

    gate_acc = simd_sum(gate_acc);
    up_acc   = simd_sum(up_acc);

    if (lid == 0) {
        float silu_g = gate_acc * (1.0f / (1.0f + exp(-gate_acc)));
        out[tgid] = silu_g * up_acc;
    }
}

// ── ffn_q8_gate_up_silu_tiled ─────────────────────────────────────────────
//
// Tiled SwiGLU for Q8_0 weights — loads x into threadgroup memory once,
// reused by ROWS_PER_TG=8 output rows.  Reduces x-read bandwidth by 8×.
//
// 128 threads (4 simd groups) per threadgroup, each TG processes 8 rows:
//   - SIMD group g handles rows base+g*2 and base+g*2+1
//   - All 128 threads cooperate to load x → tg_x[] once
//
// dispatch_thread_groups( (ceil(i_dim/8), 1, 1), (128, 1, 1) )
//
constant uint ROWS_PER_TG = 8;

kernel void ffn_q8_gate_up_silu_tiled(
    device const q8_block_ffn* W_gate   [[ buffer(0) ]],
    device const q8_block_ffn* W_up     [[ buffer(1) ]],
    device const float*        x        [[ buffer(2) ]],
    device       float*        out      [[ buffer(3) ]],
    constant     uint&         cols     [[ buffer(4) ]],
    constant     uint&         i_dim    [[ buffer(5) ]],
    uint tgid  [[ threadgroup_position_in_grid ]],
    uint sgitg [[ simdgroup_index_in_threadgroup ]],    // 0..3
    uint siitg [[ thread_index_in_simdgroup ]],         // 0..31
    uint tid   [[ thread_index_in_threadgroup ]],       // 0..127
    threadgroup float* tg_x [[ threadgroup(0) ]])       // [cols] floats
{
    const uint n_blocks  = cols / 32;
    const uint base_row  = tgid * ROWS_PER_TG;

    // ── Phase 1: cooperatively load x into threadgroup memory ─────────────
    // 128 threads load a stride-128 pattern across `cols` floats.
    for (uint i = tid; i < cols; i += 128)
        tg_x[i] = x[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2: each simd group handles 2 rows ───────────────────────────
    // row0 = base_row + sgitg*2,  row1 = base_row + sgitg*2 + 1
    const uint row0 = base_row + sgitg * 2;
    const uint row1 = row0 + 1;

    float gate0 = 0.0f, up0 = 0.0f;
    float gate1 = 0.0f, up1 = 0.0f;

    if (row0 < i_dim) {
        device const q8_block_ffn* g0 = W_gate + row0 * n_blocks;
        device const q8_block_ffn* u0 = W_up   + row0 * n_blocks;
        // Stride by 32 lanes within this SIMD group: each lane handles
        // consecutive blocks; inner loop decodes 32 elements per block.
        for (uint b = siitg; b < n_blocks; b += 32) {
            float sg = float(g0[b].d);
            float su = float(u0[b].d);
            uint x_off = b * 32;
            float lg = 0.0f, lu = 0.0f;
            for (uint k = 0; k < 32; ++k) {
                float xk = tg_x[x_off + k];
                lg += float(g0[b].qs[k]) * xk;
                lu += float(u0[b].qs[k]) * xk;
            }
            gate0 += lg * sg;
            up0   += lu * su;
        }
    }

    if (row1 < i_dim) {
        device const q8_block_ffn* g1 = W_gate + row1 * n_blocks;
        device const q8_block_ffn* u1 = W_up   + row1 * n_blocks;
        for (uint b = siitg; b < n_blocks; b += 32) {
            float sg = float(g1[b].d);
            float su = float(u1[b].d);
            uint x_off = b * 32;
            float lg = 0.0f, lu = 0.0f;
            for (uint k = 0; k < 32; ++k) {
                float xk = tg_x[x_off + k];
                lg += float(g1[b].qs[k]) * xk;
                lu += float(u1[b].qs[k]) * xk;
            }
            gate1 += lg * sg;
            up1   += lu * su;
        }
    }

    gate0 = simd_sum(gate0);  up0 = simd_sum(up0);
    gate1 = simd_sum(gate1);  up1 = simd_sum(up1);

    if (siitg == 0) {
        if (row0 < i_dim) {
            float g = gate0;
            out[row0] = apply_activation(g, up0);
        }
        if (row1 < i_dim) {
            float g = gate1;
            out[row1] = apply_activation(g, up1);
        }
    }
}

// ── ffn_q8_gate_up_silu_normed ────────────────────────────────────────────
//
// Fused residual-norm + SwiGLU FFN for Q8_0 weights.
//
// Eliminates the buffer_barrier between fused_residual_rms_norm and the
// standard ffn_q8_gate_up_silu kernel by inlining the norm computation.
//
// Each TG handles one output row of gate+up:
//   1. Computes RMS of (hidden[i] + delta[i]) for all i (norm phase, cheap).
//   2. For every Q8_0 block in the row: dequantises and dot-products against
//      the inline-normalised input: x[i] = (hidden[i]+delta[i]) * inv_rms * w[i]
//
// The residual update (hidden += delta) is NOT done here to avoid write
// conflicts with concurrent dispatches. The caller is responsible for
// issuing a concurrent residual_add(hidden, delta) dispatch running in
// parallel with the down-projection that follows this kernel.
//
// Buffers:
//   0  W_gate  q8_block[i_dim × n_blocks]   gate weight (Q8_0)
//   1  W_up    q8_block[i_dim × n_blocks]   up   weight (Q8_0)
//   2  hidden  float[h_dim]                 residual accumulator (read-only here)
//   3  delta   float[h_dim]                 residual correction  (read-only)
//   4  norm_w  float[h_dim]                 RMSNorm scale weights
//   5  out     float[i_dim]                 ffn_mid output
//   6  cols    uint                          h_dim (must be % 32 == 0)
//   7  rms_eps float                         ε for RMS norm
//
// dispatch_thread_groups( (i_dim, 1, 1), (32, 1, 1) )
//
kernel void ffn_q8_gate_up_silu_normed(
    device const q8_block_ffn* W_gate  [[ buffer(0) ]],
    device const q8_block_ffn* W_up    [[ buffer(1) ]],
    device const float*        hidden  [[ buffer(2) ]],
    device const float*        delta   [[ buffer(3) ]],
    device const float*        norm_w  [[ buffer(4) ]],
    device       float*        out     [[ buffer(5) ]],
    constant     uint&         cols    [[ buffer(6) ]],
    constant     float&        rms_eps [[ buffer(7) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint lid  [[ thread_index_in_simdgroup ]])
{
    const uint n_blocks = cols / 32u;

    // ── Phase 1: compute RMS of (hidden + delta) ──────────────────────────
    // Each of 32 lanes accumulates over its stride slice.
    float sum_sq = 0.0f;
    for (uint i = lid; i < cols; i += 32u) {
        float v = hidden[i] + delta[i];
        sum_sq += v * v;
    }
    sum_sq  = simd_sum(sum_sq);
    float inv_rms = rsqrt(sum_sq / float(cols) + rms_eps);

    // ── Phase 2: Q8_0 gate + up dot products with inline normalised input ──
    device const q8_block_ffn* gate_row = W_gate + tgid * n_blocks;
    device const q8_block_ffn* up_row   = W_up   + tgid * n_blocks;

    float gate_acc = 0.0f;
    float up_acc   = 0.0f;

    for (uint b = lid; b < n_blocks; b += 32u) {
        float scale_g = float(gate_row[b].d);
        float scale_u = float(up_row[b].d);
        uint  x_off   = b * 32u;
        float lg = 0.0f, lu = 0.0f;
        for (uint k = 0u; k < 32u; ++k) {
            float xk = (hidden[x_off + k] + delta[x_off + k]) * inv_rms * norm_w[x_off + k];
            lg += float(gate_row[b].qs[k]) * xk;
            lu += float(up_row[b].qs[k])   * xk;
        }
        gate_acc += lg * scale_g;
        up_acc   += lu * scale_u;
    }

    gate_acc = simd_sum(gate_acc);
    up_acc   = simd_sum(up_acc);

    if (lid == 0u) {
        float g = gate_acc;
        out[tgid] = apply_activation(g, up_acc);
    }
}

// ── ffn_q5_gate_up_silu_normed ────────────────────────────────────────────
//
// Fused residual-norm + SwiGLU FFN for Q5_0 weights.
// Saves ~3.3 MB × 2 (gate+up) × 24 layers ≈ 158 MB DRAM/token vs Q8_0 path.
//
// Q5_0 block layout (22 bytes / 32 elements, raw bytes):
//   bytes 0-1 : d    — fp16 scale (broadcast to all 32 lanes)
//   bytes 2-5 : qh   — uint32, bit k = high bit of element k (broadcast)
//   bytes 6-21: qs_lo[16] — nibble-packed low 4 bits (2 elements per byte)
//
// Each lane `lid` (0..31) handles exactly ONE element per block iteration:
//   lo4 = qs_lo[lid/2] lower nibble (lid even) or upper nibble (lid odd)
//   hi1 = (qh >> lid) & 1
//   q   = (lo4 | (hi1 << 4)) - 16   → signed value in [-16..15]
//
// 100% SIMD utilization: all 32 lanes active per block, 28 blocks/row,
// vs the previous 87.5%-utilization j-loop design that serialised decode.
//
// dispatch_thread_groups( (i_dim, 1, 1), (32, 1, 1) )
//
