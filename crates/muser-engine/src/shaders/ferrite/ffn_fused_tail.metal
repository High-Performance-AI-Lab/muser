kernel void ffn_q4k_gate_up_silu_normed_4row(
    device const uchar* W_gate  [[ buffer(0) ]],
    device const uchar* W_up    [[ buffer(1) ]],
    device const float* hidden  [[ buffer(2) ]],
    device const float* delta   [[ buffer(3) ]],
    device const float* norm_w  [[ buffer(4) ]],
    device       float* out     [[ buffer(5) ]],
    constant     uint&  cols    [[ buffer(6) ]],
    constant     float& rms_eps [[ buffer(7) ]],
    constant     uint&  i_dim   [[ buffer(8) ]],
    uint tgid  [[ threadgroup_position_in_grid ]],
    uint sgitg [[ simdgroup_index_in_threadgroup ]],
    uint siitg [[ thread_index_in_simdgroup ]],
    uint tid   [[ thread_index_in_threadgroup ]],
    threadgroup float* tg_mem [[ threadgroup(0) ]])
{
    // Layout: [256 x-values] [4 partial sums] [1 inv_rms]
    threadgroup float* tg_x       = tg_mem;
    threadgroup float& tg_inv_rms = tg_mem[260];

    const uint n_blocks    = cols / 256u;
    const uint block_bytes = 144u;
    const uint base_row    = tgid * 4u; // non-v4 path: 4 rows/TG, 128 threads

    // ── Phase 1: RMS of (hidden + delta) — SG0-only (stride-32) ─────────
    // Only SIMD group 0 computes sum-of-squares, matching the standalone
    // rms_norm kernel's accumulation order for bit-identical inv_rms.
    if (sgitg == 0u) {
        float sum_sq = 0.0f;
        for (uint i = siitg; i < cols; i += 32u) {
            float v = hidden[i] + delta[i];
            sum_sq += v * v;
        }
        sum_sq = simd_sum(sum_sq);
        if (siitg == 0u)
            tg_inv_rms = rsqrt(sum_sq / float(cols) + rms_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = tg_inv_rms;

    // ── Phase 2: block-by-block Q4_K gate+up with shared normalised x ───
    const uint row = base_row + sgitg;
    const bool active = (row < i_dim);

    device const uchar* gate_base = W_gate + (ulong)row * n_blocks * block_bytes;
    device const uchar* up_base   = W_up   + (ulong)row * n_blocks * block_bytes;

    float gate_acc = 0.0f, up_acc = 0.0f;

    for (uint b = 0u; b < n_blocks; b++) {
        // All 128 threads cooperatively load & normalise this block's x
        uint x_base = b * 256u;
        tg_x[tid]        = (hidden[x_base + tid]        + delta[x_base + tid])
                           * inv_rms * norm_w[x_base + tid];
        tg_x[tid + 128u] = (hidden[x_base + tid + 128u] + delta[x_base + tid + 128u])
                           * inv_rms * norm_w[x_base + tid + 128u];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            device const uchar* gb = gate_base + b * block_bytes;
            device const uchar* ub = up_base   + b * block_bytes;

            uint gdw = *reinterpret_cast<device const uint*>(gb);
            float dg    = float(as_type<half>(ushort(gdw        & 0xFFFFu)));
            float dgmin = float(as_type<half>(ushort(gdw >> 16u)));

            uint udw = *reinterpret_cast<device const uint*>(ub);
            float du    = float(as_type<half>(ushort(udw        & 0xFFFFu)));
            float dumin = float(as_type<half>(ushort(udw >> 16u)));

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

            for (uint g = 0u; g < 4u; g++) {
                uint gqs_b = uint((gb + 16u)[g * 32u + siitg]);
                uint uqs_b = uint((ub + 16u)[g * 32u + siitg]);

                float xk_lo = tg_x[g * 64u + siitg];
                float xk_hi = tg_x[g * 64u + 32u + siitg];

                gate_acc += fma(g_d_sc[g*2u], float(gqs_b & 0xFu), g_neg_dm[g*2u]) * xk_lo;
                gate_acc += fma(g_d_sc[g*2u+1u], float((gqs_b >> 4u) & 0xFu), g_neg_dm[g*2u+1u]) * xk_hi;
                up_acc   += fma(u_d_sc[g*2u], float(uqs_b & 0xFu), u_neg_dm[g*2u]) * xk_lo;
                up_acc   += fma(u_d_sc[g*2u+1u], float((uqs_b >> 4u) & 0xFu), u_neg_dm[g*2u+1u]) * xk_hi;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active) {
        gate_acc = simd_sum(gate_acc);
        up_acc   = simd_sum(up_acc);

        if (siitg == 0u) {
            float g = gate_acc;
            out[row] = apply_activation(g, up_acc);
        }
    }
}


kernel void ffn_q5k_gate_up_silu_normed(
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
    const uint block_bytes = 176u;

    // ── Phase 1: RMS of (hidden + delta) ─────────────────────────────────
    float sum_sq = 0.0f;
    for (uint i = lid; i < cols; i += 32u) {
        float v = hidden[i] + delta[i];
        sum_sq += v * v;
    }
    sum_sq = simd_sum(sum_sq);
    float inv_rms = rsqrt(sum_sq / float(cols) + rms_eps);

    // ── Phase 2: Q5_K gate + up dot products with inline-normalised input ─
    device const uchar* gate_row = W_gate + (ulong)tgid * n_blocks * block_bytes;
    device const uchar* up_row   = W_up   + (ulong)tgid * n_blocks * block_bytes;

    float gate_acc = 0.0f, up_acc = 0.0f;

    for (uint b = 0u; b < n_blocks; b++) {
        device const uchar* gb    = gate_row + b * block_bytes;
        device const uchar* ub    = up_row   + b * block_bytes;
        device const uchar* g_sc  = gb + 4u;
        device const uchar* u_sc  = ub + 4u;
        device const uchar* g_qh  = gb + 16u;
        device const uchar* u_qh  = ub + 16u;
        device const uchar* g_qs  = gb + 48u;
        device const uchar* u_qs  = ub + 48u;

        ushort dg_bits    = ushort(gb[0]) | (ushort(gb[1]) << 8u);
        ushort dgmin_bits = ushort(gb[2]) | (ushort(gb[3]) << 8u);
        float  dg    = float(as_type<half>(dg_bits));
        float  dgmin = float(as_type<half>(dgmin_bits));

        ushort du_bits    = ushort(ub[0]) | (ushort(ub[1]) << 8u);
        ushort dumin_bits = ushort(ub[2]) | (ushort(ub[3]) << 8u);
        float  du    = float(as_type<half>(du_bits));
        float  dumin = float(as_type<half>(dumin_bits));

        for (uint step = 0u; step < 8u; step++) {
            uint k  = step * 32u + lid;
            uint g  = k >> 6u;
            uint s  = (k >> 5u) & 1u;
            uint l  = k & 31u;
            uint sb = (g << 1u) | s;

            uint gsc_v, gm_v, usc_v, um_v;
            get_scale_min_k4(sb, g_sc, gsc_v, gm_v);
            get_scale_min_k4(sb, u_sc, usc_v, um_v);

            uint gqs_b = uint(g_qs[g * 32u + l]);
            uint uqs_b = uint(u_qs[g * 32u + l]);
            uint gnib  = (s == 0u) ? (gqs_b & 0xFu) : ((gqs_b >> 4u) & 0xFu);
            uint unib  = (s == 0u) ? (uqs_b & 0xFu) : ((uqs_b >> 4u) & 0xFu);

            // Splice 5th high bit from qh plane: qh[l] bit (g*2+s)
            uint g_hb = (uint(g_qh[l]) >> (g * 2u + s)) & 1u;
            uint u_hb = (uint(u_qh[l]) >> (g * 2u + s)) & 1u;
            uint gq5  = gnib | (g_hb << 4u);
            uint uq5  = unib | (u_hb << 4u);

            float gval = dg * float(gsc_v) * float(gq5) - dgmin * float(gm_v);
            float uval = du * float(usc_v) * float(uq5) - dumin * float(um_v);

            uint elem = b * 256u + k;
            float xk  = (hidden[elem] + delta[elem]) * inv_rms * norm_w[elem];
            gate_acc += gval * xk;
            up_acc   += uval * xk;
        }
    }

    gate_acc = simd_sum(gate_acc);
    up_acc   = simd_sum(up_acc);

    if (lid == 0u) {
        float g = gate_acc;
        out[tgid] = apply_activation(g, up_acc);
    }
}

// ── ffn_q4k_gate_up_silu ─────────────────────────────────────────────────
//
// Non-normed Q4_K fused gate+up for pre-normalised input.
// Reads from `x` directly (norm already applied by separate dispatch).
// Activation is selected at PSO build time via FC_FFN_ACTIVATION:
//   - default PSO: SiLU
//   - Gemma-specific PSO: GELU
// dispatch_thread_groups( (i_dim, 1, 1), (32, 1, 1) )
//
kernel void ffn_q4k_gate_up_silu(
    device const uchar* W_gate [[ buffer(0) ]],
    device const uchar* W_up   [[ buffer(1) ]],
    device const float* x      [[ buffer(2) ]],
    device       float* out    [[ buffer(3) ]],
    constant     uint&  cols   [[ buffer(4) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint lid  [[ thread_index_in_simdgroup ]])
{
    const uint n_blocks    = cols / 256u;
    const uint block_bytes = 144u;

    device const uchar* gate_row = W_gate + (ulong)tgid * n_blocks * block_bytes;
    device const uchar* up_row   = W_up   + (ulong)tgid * n_blocks * block_bytes;

    float gate_acc = 0.0f, up_acc = 0.0f;

    for (uint b = 0u; b < n_blocks; b++) {
        device const uchar* gb   = gate_row + b * block_bytes;
        device const uchar* ub   = up_row   + b * block_bytes;
        device const uchar* g_qs = gb + 16u;
        device const uchar* u_qs = ub + 16u;

        uint gdw = *reinterpret_cast<device const uint*>(gb);
        float dg    = float(as_type<half>(ushort(gdw        & 0xFFFFu)));
        float dgmin = float(as_type<half>(ushort(gdw >> 16u)));

        uint udw = *reinterpret_cast<device const uint*>(ub);
        float du    = float(as_type<half>(ushort(udw        & 0xFFFFu)));
        float dumin = float(as_type<half>(ushort(udw >> 16u)));

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

        for (uint g = 0u; g < 4u; g++) {
            uint gqs_b = uint(g_qs[g * 32u + lid]);
            uint uqs_b = uint(u_qs[g * 32u + lid]);

            uint gnib_lo = gqs_b & 0xFu;
            uint gnib_hi = (gqs_b >> 4u) & 0xFu;
            uint unib_lo = uqs_b & 0xFu;
            uint unib_hi = (uqs_b >> 4u) & 0xFu;

            uint k_lo = b * 256u + g * 64u + lid;
            uint k_hi = k_lo + 32u;
            float xk_lo = x[k_lo];
            float xk_hi = x[k_hi];

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

// ── ffn_q4k_gate_up_silu_4sg ─────────────────────────────────────────────
//
// 4-SIMD-group variant of the pre-normalised Q4_K fused gate+up path.
// Partitions Q4_K blocks across 4 SIMD groups per output row.
// For cols=3584 (7B): n_blocks=14 → 3-4 blocks per SG.
// dispatch_thread_groups( (i_dim, 1, 1), (128, 1, 1) )
//
kernel void ffn_q4k_gate_up_silu_4sg(
    device const uchar* W_gate [[ buffer(0) ]],
    device const uchar* W_up   [[ buffer(1) ]],
    device const float* x      [[ buffer(2) ]],
    device       float* out    [[ buffer(3) ]],
    constant     uint&  cols   [[ buffer(4) ]],
    uint tgid  [[ threadgroup_position_in_grid ]],
    uint sgitg [[ simdgroup_index_in_threadgroup ]],
    uint siitg [[ thread_index_in_simdgroup ]])
{
    const uint n_blocks    = cols / 256u;
    const uint block_bytes = 144u;

    device const uchar* gate_row = W_gate + (ulong)tgid * n_blocks * block_bytes;
    device const uchar* up_row   = W_up   + (ulong)tgid * n_blocks * block_bytes;

    float gate_acc = 0.0f, up_acc = 0.0f;

    for (uint b = sgitg; b < n_blocks; b += 4u) {
        device const uchar* gb   = gate_row + b * block_bytes;
        device const uchar* ub   = up_row   + b * block_bytes;
        device const uchar* g_qs = gb + 16u;
        device const uchar* u_qs = ub + 16u;

        uint gdw = *reinterpret_cast<device const uint*>(gb);
        float dg    = float(as_type<half>(ushort(gdw        & 0xFFFFu)));
        float dgmin = float(as_type<half>(ushort(gdw >> 16u)));

        uint udw = *reinterpret_cast<device const uint*>(ub);
        float du    = float(as_type<half>(ushort(udw        & 0xFFFFu)));
        float dumin = float(as_type<half>(ushort(udw >> 16u)));

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

        for (uint g = 0u; g < 4u; g++) {
            uint gqs_b = uint(g_qs[g * 32u + siitg]);
            uint uqs_b = uint(u_qs[g * 32u + siitg]);

            uint gnib_lo = gqs_b & 0xFu;
            uint gnib_hi = (gqs_b >> 4u) & 0xFu;
            uint unib_lo = uqs_b & 0xFu;
            uint unib_hi = (uqs_b >> 4u) & 0xFu;

            uint k_lo = b * 256u + g * 64u + siitg;
            uint k_hi = k_lo + 32u;
            float xk_lo = x[k_lo];
            float xk_hi = x[k_hi];

            gate_acc += fma(g_d_sc[g*2u], float(gnib_lo), g_neg_dm[g*2u]) * xk_lo;
            gate_acc += fma(g_d_sc[g*2u+1u], float(gnib_hi), g_neg_dm[g*2u+1u]) * xk_hi;
            up_acc   += fma(u_d_sc[g*2u], float(unib_lo), u_neg_dm[g*2u]) * xk_lo;
            up_acc   += fma(u_d_sc[g*2u+1u], float(unib_hi), u_neg_dm[g*2u+1u]) * xk_hi;
        }
    }

    gate_acc = simd_sum(gate_acc);
    up_acc   = simd_sum(up_acc);

    threadgroup float tg_gate[4];
    threadgroup float tg_up[4];
    if (siitg == 0u) {
        tg_gate[sgitg] = gate_acc;
        tg_up[sgitg]   = up_acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0u && siitg == 0u) {
        float g = tg_gate[0] + tg_gate[1] + tg_gate[2] + tg_gate[3];
        float u = tg_up[0]   + tg_up[1]   + tg_up[2]   + tg_up[3];
        out[tgid] = apply_activation(g, u);
    }
}

// ── ffn_q4k_gate_up_silu_4sg_tgcache ──────────────────────────────────────
//
// Threadgroup x-cache variant of ffn_q4k_gate_up_silu_4sg.
// All 128 threads cooperatively load the full x vector into threadgroup
// memory once, then each SIMD reads from tg_x instead of device x.
//
// At Gemma 4 decode shapes (cols=3072): x = 12 KB, fits in 32 KB TG limit.
// Converts 4× device reads to 1× device + 4× threadgroup reads per TG.
//
// Activation selected at PSO build time via FC_FFN_ACTIVATION (same as 4sg).
// dispatch_thread_groups( (i_dim, 1, 1), (128, 1, 1) )
//
kernel void ffn_q4k_gate_up_silu_4sg_tgcache(
    device const uchar* W_gate [[ buffer(0) ]],
    device const uchar* W_up   [[ buffer(1) ]],
    device const float* x      [[ buffer(2) ]],
    device       float* out    [[ buffer(3) ]],
    constant     uint&  cols   [[ buffer(4) ]],
    uint tgid  [[ threadgroup_position_in_grid ]],
    uint sgitg [[ simdgroup_index_in_threadgroup ]],
    uint siitg [[ thread_index_in_simdgroup ]],
    uint tid   [[ thread_index_in_threadgroup ]],
    threadgroup float* tg_x [[ threadgroup(0) ]])
{
    const uint n_blocks    = cols / 256u;
    const uint block_bytes = 144u;

    // ── Phase 1: cooperative x load (128 threads, stride-128 over cols floats) ──
    for (uint i = tid; i < cols; i += 128u)
        tg_x[i] = x[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2: compute gate+up dot products reading from tg_x ──
    device const uchar* gate_row = W_gate + (ulong)tgid * n_blocks * block_bytes;
    device const uchar* up_row   = W_up   + (ulong)tgid * n_blocks * block_bytes;

    float gate_acc = 0.0f, up_acc = 0.0f;

    for (uint b = sgitg; b < n_blocks; b += 4u) {
        device const uchar* gb   = gate_row + b * block_bytes;
        device const uchar* ub   = up_row   + b * block_bytes;
        device const uchar* g_qs = gb + 16u;
        device const uchar* u_qs = ub + 16u;

        uint gdw = *reinterpret_cast<device const uint*>(gb);
        float dg    = float(as_type<half>(ushort(gdw        & 0xFFFFu)));
        float dgmin = float(as_type<half>(ushort(gdw >> 16u)));

        uint udw = *reinterpret_cast<device const uint*>(ub);
        float du    = float(as_type<half>(ushort(udw        & 0xFFFFu)));
        float dumin = float(as_type<half>(ushort(udw >> 16u)));

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

        for (uint g = 0u; g < 4u; g++) {
            uint gqs_b = uint(g_qs[g * 32u + siitg]);
            uint uqs_b = uint(u_qs[g * 32u + siitg]);

            uint gnib_lo = gqs_b & 0xFu;
            uint gnib_hi = (gqs_b >> 4u) & 0xFu;
            uint unib_lo = uqs_b & 0xFu;
            uint unib_hi = (uqs_b >> 4u) & 0xFu;

            uint k_lo = b * 256u + g * 64u + siitg;
            uint k_hi = k_lo + 32u;
            float xk_lo = tg_x[k_lo];
            float xk_hi = tg_x[k_hi];

            gate_acc += fma(g_d_sc[g*2u], float(gnib_lo), g_neg_dm[g*2u]) * xk_lo;
            gate_acc += fma(g_d_sc[g*2u+1u], float(gnib_hi), g_neg_dm[g*2u+1u]) * xk_hi;
            up_acc   += fma(u_d_sc[g*2u], float(unib_lo), u_neg_dm[g*2u]) * xk_lo;
            up_acc   += fma(u_d_sc[g*2u+1u], float(unib_hi), u_neg_dm[g*2u+1u]) * xk_hi;
        }
    }

    gate_acc = simd_sum(gate_acc);
    up_acc   = simd_sum(up_acc);

    threadgroup float tg_gate[4];
    threadgroup float tg_up[4];
    if (siitg == 0u) {
        tg_gate[sgitg] = gate_acc;
        tg_up[sgitg]   = up_acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0u && siitg == 0u) {
        float g = tg_gate[0] + tg_gate[1] + tg_gate[2] + tg_gate[3];
        float u = tg_up[0]   + tg_up[1]   + tg_up[2]   + tg_up[3];
        out[tgid] = apply_activation(g, u);
    }
}

// ── ffn_q4k_gate_up_silu_4r2s ─────────────────────────────────────────────
//
// 4-row-per-TG fused gate+up variant using the V4 x-load pattern.
// 64 threads (2 SIMDs × 32), 4 output rows per TG (2 per SIMD).
// Each thread loads x into local registers yl[16]/yh[16] via stride-4
// block sub-groups (same as matvec_q4k_f32_v4), reusing x for both
// gate and up weight rows.
//
// Activation selected at PSO build time via FC_FFN_ACTIVATION.
// dispatch_thread_groups( (ceil(i_dim/4), 1, 1), (64, 1, 1) )
//
kernel void ffn_q4k_gate_up_silu_4r2s(
    device const uchar* W_gate [[ buffer(0) ]],
    device const uchar* W_up   [[ buffer(1) ]],
    device const float* x      [[ buffer(2) ]],
    device       float* out    [[ buffer(3) ]],
    constant     uint&  rows   [[ buffer(4) ]],
    constant     uint&  cols   [[ buffer(5) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint lid  [[ thread_index_in_simdgroup ]],
    uint sgid [[ simdgroup_index_in_threadgroup ]])
{
    const uint n_blocks    = cols / 256u;
    const uint block_bytes = 144u;
    const uint row_bytes   = n_blocks * block_bytes;

    // 2 rows per SIMD, 2 SIMDs = 4 rows per TG
    const uint base_row = tgid * 4u + sgid * 2u;
    if (base_row >= rows) return;

    // V4 thread partitioning: 4 sub-groups of 8 threads for block stride
    const uint ix = lid / 8u;   // 0..3 — block stride index
    const uint it = lid % 8u;   // 0..7 — position within block
    const uint iq = it / 4u;    // 0 or 1 — half-block selector
    const uint ir = it % 4u;    // 0..3 — quarter within half

    // x-vector pointer: each thread reads 8 positions per block (stride-4 blocks)
    device const float* xp = x + ix * 256u + 64u * iq + 8u * ir;

    float yl[16], yh[16];
    float gate_sumf[2] = {0.f, 0.f};
    float up_sumf[2]   = {0.f, 0.f};

    for (uint ib = ix; ib < n_blocks; ib += 4u) {
        // Load x slice into registers (8 elements × 4 positions)
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (uint i = 0u; i < 8u; i++) {
            yl[i]     = xp[i];       sumy[0] += yl[i];
            yl[i + 8] = xp[i + 32];  sumy[1] += yl[i + 8];
            yh[i]     = xp[i + 128]; sumy[2] += yh[i];
            yh[i + 8] = xp[i + 160]; sumy[3] += yh[i + 8];
        }

        // Gate weight: 2 rows starting at base_row + sgid*2
        device const uchar* gate_blk = W_gate + (ulong)base_row * (ulong)row_bytes
                                       + (ulong)ib * block_bytes;
        q4k_v4_dual_row_mac(gate_blk, row_bytes, yl, yh, sumy, iq, ir, gate_sumf);

        // Up weight: same 2 rows
        device const uchar* up_blk = W_up + (ulong)base_row * (ulong)row_bytes
                                     + (ulong)ib * block_bytes;
        q4k_v4_dual_row_mac(up_blk, row_bytes, yl, yh, sumy, iq, ir, up_sumf);

        xp += 4u * 256u;  // advance by 4 blocks (stride)
    }

    // Reduction across all 32 threads in each SIMD
    const float gr0 = simd_sum(gate_sumf[0]);
    const float gr1 = simd_sum(gate_sumf[1]);
    const float ur0 = simd_sum(up_sumf[0]);
    const float ur1 = simd_sum(up_sumf[1]);

    if (lid == 0u) {
                                    out[base_row]      = apply_activation(gr0, ur0);
        if (base_row + 1u < rows)   out[base_row + 1u] = apply_activation(gr1, ur1);
    }
}

// ── ffn_q5k_gate_up_silu ─────────────────────────────────────────────────
//
// Non-normed Q5_K SwiGLU for pre-normalised input.
// dispatch_thread_groups( (i_dim, 1, 1), (32, 1, 1) )
//
kernel void ffn_q5k_gate_up_silu(
    device const uchar* W_gate [[ buffer(0) ]],
    device const uchar* W_up   [[ buffer(1) ]],
    device const float* x      [[ buffer(2) ]],
    device       float* out    [[ buffer(3) ]],
    constant     uint&  cols   [[ buffer(4) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint lid  [[ thread_index_in_simdgroup ]])
{
    const uint n_blocks    = cols / 256u;
    const uint block_bytes = 176u;

    device const uchar* gate_row = W_gate + (ulong)tgid * n_blocks * block_bytes;
    device const uchar* up_row   = W_up   + (ulong)tgid * n_blocks * block_bytes;

    float gate_acc = 0.0f, up_acc = 0.0f;

    for (uint b = 0u; b < n_blocks; b++) {
        device const uchar* gb    = gate_row + b * block_bytes;
        device const uchar* ub    = up_row   + b * block_bytes;
        device const uchar* g_sc  = gb + 4u;
        device const uchar* u_sc  = ub + 4u;
        device const uchar* g_qh  = gb + 16u;
        device const uchar* u_qh  = ub + 16u;
        device const uchar* g_qs  = gb + 48u;
        device const uchar* u_qs  = ub + 48u;

        ushort dg_bits    = ushort(gb[0]) | (ushort(gb[1]) << 8u);
        ushort dgmin_bits = ushort(gb[2]) | (ushort(gb[3]) << 8u);
        float  dg    = float(as_type<half>(dg_bits));
        float  dgmin = float(as_type<half>(dgmin_bits));

        ushort du_bits    = ushort(ub[0]) | (ushort(ub[1]) << 8u);
        ushort dumin_bits = ushort(ub[2]) | (ushort(ub[3]) << 8u);
        float  du    = float(as_type<half>(du_bits));
        float  dumin = float(as_type<half>(dumin_bits));

        for (uint step = 0u; step < 8u; step++) {
            uint k  = step * 32u + lid;
            uint g  = k >> 6u;
            uint s  = (k >> 5u) & 1u;
            uint l  = k & 31u;
            uint sb = (g << 1u) | s;

            uint gsc_v, gm_v, usc_v, um_v;
            get_scale_min_k4(sb, g_sc, gsc_v, gm_v);
            get_scale_min_k4(sb, u_sc, usc_v, um_v);

            uint gqs_b = uint(g_qs[g * 32u + l]);
            uint uqs_b = uint(u_qs[g * 32u + l]);
            uint gnib  = (s == 0u) ? (gqs_b & 0xFu) : ((gqs_b >> 4u) & 0xFu);
            uint unib  = (s == 0u) ? (uqs_b & 0xFu) : ((uqs_b >> 4u) & 0xFu);

            uint g_hb = (uint(g_qh[l]) >> (g * 2u + s)) & 1u;
            uint u_hb = (uint(u_qh[l]) >> (g * 2u + s)) & 1u;
            uint gq5  = gnib | (g_hb << 4u);
            uint uq5  = unib | (u_hb << 4u);

            float gval = dg * float(gsc_v) * float(gq5) - dgmin * float(gm_v);
            float uval = du * float(usc_v) * float(uq5) - dumin * float(um_v);

            float xk = x[b * 256u + k];
            gate_acc += gval * xk;
            up_acc   += uval * xk;
        }
    }

    gate_acc = simd_sum(gate_acc);
    up_acc   = simd_sum(up_acc);

    if (lid == 0u) {
        float g = gate_acc;
        out[tgid] = apply_activation(g, up_acc);
    }
}
