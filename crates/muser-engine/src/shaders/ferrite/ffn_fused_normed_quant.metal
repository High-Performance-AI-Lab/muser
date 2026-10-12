kernel void ffn_q5_gate_up_silu_normed(
    device const uchar* W_gate  [[ buffer(0) ]],  // raw Q5_0 bytes
    device const uchar* W_up    [[ buffer(1) ]],
    device const float* hidden  [[ buffer(2) ]],
    device const float* delta   [[ buffer(3) ]],
    device const float* norm_w  [[ buffer(4) ]],
    device       float* out     [[ buffer(5) ]],
    constant     uint&  cols    [[ buffer(6) ]],
    constant     float& rms_eps [[ buffer(7) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint lid  [[ thread_index_in_simdgroup ]])
{
    const uint n_blocks    = cols / 32u;   // e.g. 896/32 = 28
    const uint block_bytes = 22u;          // Q5_0: 2(d)+4(qh)+16(qs_lo) = 22

    // ── Phase 1: RMS of (hidden + delta) — 100% lane utilisation ─────────
    float sum_sq = 0.0f;
    for (uint k = lid; k < cols; k += 32u) {
        float v = hidden[k] + delta[k];
        sum_sq += v * v;
    }
    sum_sq = simd_sum(sum_sq);
    float inv_rms = rsqrt(sum_sq / float(cols) + rms_eps);

    // ── Phase 2: Q5_0 dot products with inline-normalised input ──────────
    // All 32 lanes work on the SAME block each iteration (broadcast reads),
    // each extracting its own element `lid` — no sequential j-loop.
    device const uchar* gate_row = W_gate + (ulong)tgid * n_blocks * block_bytes;
    device const uchar* up_row   = W_up   + (ulong)tgid * n_blocks * block_bytes;

    float gate_acc = 0.0f, up_acc = 0.0f;

    for (uint b = 0u; b < n_blocks; b++) {
        device const uchar* gb = gate_row + b * block_bytes;
        device const uchar* ub = up_row   + b * block_bytes;

        // fp16 scale — broadcast (all 32 lanes load same 2 bytes)
        ushort dg_bits = ushort(gb[0]) | (ushort(gb[1]) << 8);
        ushort du_bits = ushort(ub[0]) | (ushort(ub[1]) << 8);
        float dg = float(as_type<half>(dg_bits));
        float du = float(as_type<half>(du_bits));

        // qh uint32 — broadcast (all 32 lanes load same 4 bytes at offset 2)
        uint qh_g = uint(gb[2]) | (uint(gb[3]) << 8) | (uint(gb[4]) << 16) | (uint(gb[5]) << 24);
        uint qh_u = uint(ub[2]) | (uint(ub[3]) << 8) | (uint(ub[4]) << 16) | (uint(ub[5]) << 24);

        // qs_lo nibble: lo nibble for lid < 16, hi nibble for lid >= 16
        // GGUF Q5_0 stores [lo_0..lo_15, hi_0..hi_15] (split, not interleaved)
        uint byte_g, byte_u;
        uint lo4_g, lo4_u;
        uint hi1_g, hi1_u;
        if (lid < 16u) {
            byte_g = uint(gb[6u + lid]);
            byte_u = uint(ub[6u + lid]);
            lo4_g = byte_g & 0xFu;
            lo4_u = byte_u & 0xFu;
            hi1_g = (qh_g >> lid) & 1u;
            hi1_u = (qh_u >> lid) & 1u;
        } else {
            uint j = lid - 16u;
            byte_g = uint(gb[6u + j]);
            byte_u = uint(ub[6u + j]);
            lo4_g = (byte_g >> 4u) & 0xFu;
            lo4_u = (byte_u >> 4u) & 0xFu;
            hi1_g = (qh_g >> lid) & 1u;
            hi1_u = (qh_u >> lid) & 1u;
        }

        // Signed 5-bit value in [-16..15]
        float q_g = float(int(lo4_g | (hi1_g << 4u)) - 16);
        float q_u = float(int(lo4_u | (hi1_u << 4u)) - 16);

        // Normalised input element for position b*32 + lid
        uint k = b * 32u + lid;
        float xk = (hidden[k] + delta[k]) * inv_rms * norm_w[k];

        gate_acc += dg * q_g * xk;
        up_acc   += du * q_u * xk;
    }

    // Final warp reduction → single scalar per output row
    gate_acc = simd_sum(gate_acc);
    up_acc   = simd_sum(up_acc);

    if (lid == 0u) {
        float g = gate_acc;
        out[tgid] = apply_activation(g, up_acc);
    }
}

// ── ffn_q5_gate_up_silu_scalar_normed ─────────────────────────────────────
//
// Q5_0 SwiGLU FFN that consumes a precomputed scalar inv_rms.
// The preceding FfnInvRms op is responsible for applying hidden += delta when
// O-proj is not residualized. This avoids recomputing RMS once per FFN row.
//
// Buffers:
//   0  W_gate   raw Q5_0 gate bytes
//   1  W_up     raw Q5_0 up bytes
//   2  hidden   float[h_dim], already residual-updated
//   3  norm_w   float[h_dim]
//   4  inv_rms  float[1]
//   5  out      float[i_dim]
//   6  cols     uint h_dim, multiple of 32
//
kernel void ffn_q5_gate_up_silu_scalar_normed(
    device const uchar* W_gate  [[ buffer(0) ]],
    device const uchar* W_up    [[ buffer(1) ]],
    device const float* hidden  [[ buffer(2) ]],
    device const float* norm_w  [[ buffer(3) ]],
    device const float* inv_rms [[ buffer(4) ]],
    device       float* out     [[ buffer(5) ]],
    constant     uint&  cols    [[ buffer(6) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint lid  [[ thread_index_in_simdgroup ]])
{
    const uint n_blocks    = cols / 32u;
    const uint block_bytes = 22u;
    const float scalar_inv_rms = inv_rms[0];

    device const uchar* gate_row = W_gate + (ulong)tgid * n_blocks * block_bytes;
    device const uchar* up_row   = W_up   + (ulong)tgid * n_blocks * block_bytes;

    float gate_acc = 0.0f, up_acc = 0.0f;

    for (uint b = 0u; b < n_blocks; b++) {
        device const uchar* gb = gate_row + b * block_bytes;
        device const uchar* ub = up_row   + b * block_bytes;

        ushort dg_bits = ushort(gb[0]) | (ushort(gb[1]) << 8);
        ushort du_bits = ushort(ub[0]) | (ushort(ub[1]) << 8);
        float dg = float(as_type<half>(dg_bits));
        float du = float(as_type<half>(du_bits));

        uint qh_g = uint(gb[2]) | (uint(gb[3]) << 8) | (uint(gb[4]) << 16) | (uint(gb[5]) << 24);
        uint qh_u = uint(ub[2]) | (uint(ub[3]) << 8) | (uint(ub[4]) << 16) | (uint(ub[5]) << 24);

        uint byte_g, byte_u;
        uint lo4_g, lo4_u;
        uint hi1_g, hi1_u;
        if (lid < 16u) {
            byte_g = uint(gb[6u + lid]);
            byte_u = uint(ub[6u + lid]);
            lo4_g = byte_g & 0xFu;
            lo4_u = byte_u & 0xFu;
            hi1_g = (qh_g >> lid) & 1u;
            hi1_u = (qh_u >> lid) & 1u;
        } else {
            uint j = lid - 16u;
            byte_g = uint(gb[6u + j]);
            byte_u = uint(ub[6u + j]);
            lo4_g = (byte_g >> 4u) & 0xFu;
            lo4_u = (byte_u >> 4u) & 0xFu;
            hi1_g = (qh_g >> lid) & 1u;
            hi1_u = (qh_u >> lid) & 1u;
        }

        float q_g = float(int(lo4_g | (hi1_g << 4u)) - 16);
        float q_u = float(int(lo4_u | (hi1_u << 4u)) - 16);

        uint k = b * 32u + lid;
        float xk = hidden[k] * scalar_inv_rms * norm_w[k];

        gate_acc += dg * q_g * xk;
        up_acc   += du * q_u * xk;
    }

    gate_acc = simd_sum(gate_acc);
    up_acc   = simd_sum(up_acc);

    if (lid == 0u) {
        float g = gate_acc;
        out[tgid] = apply_activation(g, up_acc);
    }
}

// ── ffn_q5_1_gate_up_silu_normed ──────────────────────────────────────────
//
// Fused residual-norm + SwiGLU FFN for Q5_1 weights.
// Q5_1 block layout (24 bytes / 32 elements):
//   bytes 0-1 : d     — fp16 scale
//   bytes 2-3 : m     — fp16 min
//   bytes 4-7 : qh    — uint32, bit k = high bit of element k
//   bytes 8-23: qs_lo[16] — nibble-packed low 4 bits
//
// Unsigned dequant: value = d * quant5 + m   (quant5 ∈ [0,31])
//
// dispatch_thread_groups( (i_dim, 1, 1), (32, 1, 1) )
//
kernel void ffn_q5_1_gate_up_silu_normed(
    device const uchar* W_gate  [[ buffer(0) ]],
    device const uchar* W_up    [[ buffer(1) ]],
    device const float* hidden  [[ buffer(2) ]],
    device const float* delta   [[ buffer(3) ]],
    device const float* norm_w  [[ buffer(4) ]],
    device       float* out     [[ buffer(5) ]],
    constant     uint&  cols    [[ buffer(6) ]],
    constant     float& rms_eps [[ buffer(7) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint lid  [[ thread_index_in_simdgroup ]])
{
    const uint n_blocks    = cols / 32u;
    const uint block_bytes = 24u;          // Q5_1: 2(d)+2(m)+4(qh)+16(qs) = 24

    // ── Phase 1: RMS of (hidden + delta) ─────────────────────────────────
    float sum_sq = 0.0f;
    for (uint k = lid; k < cols; k += 32u) {
        float v = hidden[k] + delta[k];
        sum_sq += v * v;
    }
    sum_sq = simd_sum(sum_sq);
    float inv_rms = rsqrt(sum_sq / float(cols) + rms_eps);

    // ── Phase 2: Q5_1 dot products with inline-normalised input ──────────
    device const uchar* gate_row = W_gate + (ulong)tgid * n_blocks * block_bytes;
    device const uchar* up_row   = W_up   + (ulong)tgid * n_blocks * block_bytes;

    float gate_acc = 0.0f, up_acc = 0.0f;

    for (uint b = 0u; b < n_blocks; b++) {
        device const uchar* gb = gate_row + b * block_bytes;
        device const uchar* ub = up_row   + b * block_bytes;

        // fp16 scale (offset 0) and min (offset 2)
        ushort dg_bits = ushort(gb[0]) | (ushort(gb[1]) << 8);
        ushort mg_bits = ushort(gb[2]) | (ushort(gb[3]) << 8);
        ushort du_bits = ushort(ub[0]) | (ushort(ub[1]) << 8);
        ushort mu_bits = ushort(ub[2]) | (ushort(ub[3]) << 8);
        float dg = float(as_type<half>(dg_bits));
        float mg = float(as_type<half>(mg_bits));
        float du = float(as_type<half>(du_bits));
        float mu = float(as_type<half>(mu_bits));

        // qh uint32 at offset 4
        uint qh_g = uint(gb[4]) | (uint(gb[5]) << 8) | (uint(gb[6]) << 16) | (uint(gb[7]) << 24);
        uint qh_u = uint(ub[4]) | (uint(ub[5]) << 8) | (uint(ub[6]) << 16) | (uint(ub[7]) << 24);

        // Decode element `lid` (0..31) from the Q5_1 block.
        // lid 0..15  → low nibble of qs_lo[lid],    qh bit lid
        // lid 16..31 → high nibble of qs_lo[lid-16], qh bit lid
        float q_g, q_u;
        if (lid < 16u) {
            uint qlb_g = uint(gb[8u + lid]);
            uint qlb_u = uint(ub[8u + lid]);
            uint lo4_g = qlb_g & 0xFu;
            uint lo4_u = qlb_u & 0xFu;
            uint hi1_g = (qh_g >> lid) & 1u;
            uint hi1_u = (qh_u >> lid) & 1u;
            q_g = float(lo4_g | (hi1_g << 4u));
            q_u = float(lo4_u | (hi1_u << 4u));
        } else {
            uint j = lid - 16u;
            uint qlb_g = uint(gb[8u + j]);
            uint qlb_u = uint(ub[8u + j]);
            uint lo4_g = (qlb_g >> 4u) & 0xFu;
            uint lo4_u = (qlb_u >> 4u) & 0xFu;
            uint hi1_g = (qh_g >> lid) & 1u;
            uint hi1_u = (qh_u >> lid) & 1u;
            q_g = float(lo4_g | (hi1_g << 4u));
            q_u = float(lo4_u | (hi1_u << 4u));
        }

        // Normalised input element for position b*32 + lid
        uint k = b * 32u + lid;
        float xk = (hidden[k] + delta[k]) * inv_rms * norm_w[k];

        gate_acc += (dg * q_g + mg) * xk;
        up_acc   += (du * q_u + mu) * xk;
    }

    gate_acc = simd_sum(gate_acc);
    up_acc   = simd_sum(up_acc);

    if (lid == 0u) {
        float g = gate_acc;
        out[tgid] = apply_activation(g, up_acc);
    }
}

// ── ffn_q4k_gate_up_silu_normed ───────────────────────────────────────────
//
// Fused residual-norm + SwiGLU FFN for Q4_K weight matrices.
// Same semantics as ffn_q8_gate_up_silu_normed but decodes Q4_K blocks.
//
// Q4_K block layout (144 bytes / 256 elements):
//   bytes  0-1 : d    — fp16 super-scale
//   bytes  2-3 : dmin — fp16 super-min-scale
//   bytes  4-15: scales[12] — packed 6-bit scale + 6-bit min (8 sub-blocks of 32)
//   bytes 16-143: qs[128]  — nibble-packed 4-bit weights (2 elements/byte)
//
// Each lane `lid` (0..31) processes 8 elements per block (stride 32 across 256).
// 100% SIMD utilization: all 32 lanes active per block.
//
// dispatch_thread_groups( (i_dim, 1, 1), (32, 1, 1) )
//
kernel void ffn_q4k_gate_up_silu_normed(
    device const uchar* W_gate  [[ buffer(0) ]],
    device const uchar* W_up    [[ buffer(1) ]],
    device const float* hidden  [[ buffer(2) ]],
    device const float* delta   [[ buffer(3) ]],
    device const float* norm_w  [[ buffer(4) ]],
    device       float* out     [[ buffer(5) ]],
    constant     uint&  cols    [[ buffer(6) ]],
    constant     float& rms_eps [[ buffer(7) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint lid  [[ thread_index_in_simdgroup ]])
{
    const uint n_blocks    = cols / 256u;
    const uint block_bytes = 144u;

    // ── Phase 1: RMS of (hidden + delta) ─────────────────────────────────
    float sum_sq = 0.0f;
    for (uint i = lid; i < cols; i += 32u) {
        float v = hidden[i] + delta[i];
        sum_sq += v * v;
    }
    sum_sq = simd_sum(sum_sq);
    float inv_rms = rsqrt(sum_sq / float(cols) + rms_eps);

    // ── Phase 2: Q4_K gate + up dot products with inline-normalised input ─
    device const uchar* gate_row = W_gate + (ulong)tgid * n_blocks * block_bytes;
    device const uchar* up_row   = W_up   + (ulong)tgid * n_blocks * block_bytes;

    float gate_acc = 0.0f, up_acc = 0.0f;

    for (uint b = 0u; b < n_blocks; b++) {
        device const uchar* gb   = gate_row + b * block_bytes;
        device const uchar* ub   = up_row   + b * block_bytes;
        device const uchar* g_qs = gb + 16u;
        device const uchar* u_qs = ub + 16u;

        // Vectorized d/dmin loads (1 uint each vs 4 byte loads)
        uint gdw = *reinterpret_cast<device const uint*>(gb);
        float dg    = float(as_type<half>(ushort(gdw        & 0xFFFFu)));
        float dgmin = float(as_type<half>(ushort(gdw >> 16u)));

        uint udw = *reinterpret_cast<device const uint*>(ub);
        float du    = float(as_type<half>(ushort(udw        & 0xFFFFu)));
        float dumin = float(as_type<half>(ushort(udw >> 16u)));

        // Vectorized scale loads + hoisted d*sc / -(dmin*m)
        float g_d_sc[8], g_neg_dm[8], u_d_sc[8], u_neg_dm[8];
        {
            uint gsd0 = *reinterpret_cast<device const uint*>(gb + 4u);
            uint gsd1 = *reinterpret_cast<device const uint*>(gb + 8u);
            uint gsd2 = *reinterpret_cast<device const uint*>(gb + 12u);
            decode_all_q4k_scales(dg, dgmin, gsd0, gsd1, gsd2, g_d_sc, g_neg_dm);
        }
        {
            uint usd0 = *reinterpret_cast<device const uint*>(ub + 4u);
            uint usd1 = *reinterpret_cast<device const uint*>(ub + 8u);
            uint usd2 = *reinterpret_cast<device const uint*>(ub + 12u);
            decode_all_q4k_scales(du, dumin, usd0, usd1, usd2, u_d_sc, u_neg_dm);
        }

        // Process 4 groups × 2 sub-blocks: one qs byte yields both nibbles.
        // Halves qs reads (4 vs 8 per weight matrix) and eliminates branches.
        for (uint g = 0u; g < 4u; g++) {
            uint gqs_b = uint(g_qs[g * 32u + lid]);
            uint uqs_b = uint(u_qs[g * 32u + lid]);

            uint gnib_lo = gqs_b & 0xFu;
            uint gnib_hi = (gqs_b >> 4u) & 0xFu;
            uint unib_lo = uqs_b & 0xFu;
            uint unib_hi = (uqs_b >> 4u) & 0xFu;

            uint k_lo = b * 256u + g * 64u + lid;
            uint k_hi = k_lo + 32u;
            float xk_lo = (hidden[k_lo] + delta[k_lo]) * inv_rms * norm_w[k_lo];
            float xk_hi = (hidden[k_hi] + delta[k_hi]) * inv_rms * norm_w[k_hi];

            gate_acc += fma(g_d_sc[g*2u], float(gnib_lo), g_neg_dm[g*2u]) * xk_lo;
            gate_acc += fma(g_d_sc[g*2u+1u], float(gnib_hi), g_neg_dm[g*2u+1u]) * xk_hi;
            up_acc   += fma(u_d_sc[g*2u], float(unib_lo), u_neg_dm[g*2u]) * xk_lo;
            up_acc   += fma(u_d_sc[g*2u+1u], float(unib_hi), u_neg_dm[g*2u+1u]) * xk_hi;
        }
    }

    gate_acc = simd_sum(gate_acc);
    up_acc   = simd_sum(up_acc);

    if (lid == 0u) {
        float g = gate_acc;
        out[tgid] = apply_activation(g, up_acc);
    }
}

// ── ffn_q4k_gate_up_silu_normed_4row ─────────────────────────────────────
//
// 4-row tiled variant of ffn_q4k_gate_up_silu_normed.
// 128 threads (4 SIMD groups), each SIMD group processes one output row.
// Normalised x is cooperatively loaded into threadgroup memory once per
// Q4_K block and reused by all 4 rows.  Reduces x device-memory reads by
// 4× and RMS computation by 4× vs the 1-row variant.
//
// Threadgroup memory: 256 floats (1024 bytes) for x-block +
//                       4 floats (16 bytes) for cross-SIMD sum reduction +
//                       1 float (4 bytes) for shared inv_rms
//                     = 1044 bytes total.
//
// dispatch_thread_groups( (ceil(i_dim/2), 1, 1), (64, 1, 1) )
//
