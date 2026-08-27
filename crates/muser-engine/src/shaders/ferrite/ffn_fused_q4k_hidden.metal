kernel void ffn_q4k_gate_up_silu_normed_hidden_4row(
    device const uchar* W_gate  [[ buffer(0) ]],
    device const uchar* W_up    [[ buffer(1) ]],
    device const float* hidden  [[ buffer(2) ]],
    device const float* norm_w  [[ buffer(3) ]],
    device       float* out     [[ buffer(4) ]],
    constant     uint&  cols    [[ buffer(5) ]],
    constant     float& rms_eps [[ buffer(6) ]],
    constant     uint&  i_dim   [[ buffer(7) ]],
    uint tgid  [[ threadgroup_position_in_grid ]],
    uint sgitg [[ simdgroup_index_in_threadgroup ]],
    uint siitg [[ thread_index_in_simdgroup ]],
    uint tid   [[ thread_index_in_threadgroup ]],
    threadgroup float* tg_mem [[ threadgroup(0) ]])
{
    threadgroup float* tg_x       = tg_mem;
    threadgroup float& tg_inv_rms = tg_mem[260];

    const uint n_blocks    = cols / 256u;
    const uint block_bytes = 144u;
    const uint base_row    = tgid * 4u; // non-v4 path: 4 rows/TG, 128 threads

    // SG0-only reduction (stride-32) for bit-identical inv_rms.
    if (sgitg == 0u) {
        float sum_sq = 0.0f;
        for (uint i = siitg; i < cols; i += 32u) {
            float v = hidden[i];
            sum_sq += v * v;
        }
        sum_sq = simd_sum(sum_sq);
        if (siitg == 0u)
            tg_inv_rms = rsqrt(sum_sq / float(cols) + rms_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = tg_inv_rms;

    const uint row = base_row + sgitg;
    const bool active = (row < i_dim);

    device const uchar* gate_base = W_gate + (ulong)row * n_blocks * block_bytes;
    device const uchar* up_base   = W_up   + (ulong)row * n_blocks * block_bytes;

    float gate_acc = 0.0f, up_acc = 0.0f;

    for (uint b = 0u; b < n_blocks; b++) {
        uint x_base = b * 256u;
        tg_x[tid]        = hidden[x_base + tid] * inv_rms * norm_w[x_base + tid];
        tg_x[tid + 128u] = hidden[x_base + tid + 128u] * inv_rms * norm_w[x_base + tid + 128u];
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

// ── ffn_q4k_gate_up_silu_normed_4row_v4 ───────────────────────────────────
//
// Same fused Q4_K 4-row FFN as above, but processes hidden blocks in stride-4
// order so each 8-lane subgroup works on a different block simultaneously.
// This raises weight-side memory-level parallelism on M3 Ultra without
// changing the algebra: it is only a reassociation of the block reduction.
//
kernel void ffn_q4k_gate_up_silu_normed_4row_v4(
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
    // Register-based x-loading with 2 SIMD groups (64 threads).
    // 2× more TGs in flight per core for better memory-level parallelism.
    threadgroup float& tg_inv_rms = tg_mem[2];

    const uint n_blocks    = cols / 256u;
    const uint block_bytes = 144u;
    const uint base_row    = tgid * FFN_Q4K_ROWS_PER_TG;
    const uint row         = base_row + sgitg;
    const bool active      = (row < i_dim);

    const uint ix = siitg / 8u;
    const uint it = siitg % 8u;

    // SG0-only reduction (stride-32) for bit-identical inv_rms.
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

    device const uchar* gate_base = W_gate + (ulong)row * n_blocks * block_bytes;
    device const uchar* up_base   = W_up   + (ulong)row * n_blocks * block_bytes;

    // 8 independent accumulators break dependency chains across sub-block groups.
    float ga[8] = {0,0,0,0,0,0,0,0};
    float ua[8] = {0,0,0,0,0,0,0,0};

    // Main loop: NO shared memory, NO barriers. Each thread reads x from
    // device memory (L1/L2 cached). Redundant across SIMDs but cache-hot.
    for (uint ib = ix; ib < n_blocks; ib += 4u) {
        if (!active) continue;

        device const uchar* gb = gate_base + ib * block_bytes;
        device const uchar* ub = up_base   + ib * block_bytes;

        uint gdw = *reinterpret_cast<device const uint*>(gb);
        half dg    = as_type<half>(ushort(gdw        & 0xFFFFu));
        half dgmin = as_type<half>(ushort(gdw >> 16u));

        uint udw = *reinterpret_cast<device const uint*>(ub);
        half du    = as_type<half>(ushort(udw        & 0xFFFFu));
        half dumin = as_type<half>(ushort(udw >> 16u));

        half g_d_sc[8], g_neg_dm[8], u_d_sc[8], u_neg_dm[8];
        {
            uint gsd0 = *reinterpret_cast<device const uint*>(gb + 4u);
            uint gsd1 = *reinterpret_cast<device const uint*>(gb + 8u);
            uint gsd2 = *reinterpret_cast<device const uint*>(gb + 12u);
            decode_all_q4k_scales_h(dg, dgmin, gsd0, gsd1, gsd2, g_d_sc, g_neg_dm);
        }
        {
            uint usd0 = *reinterpret_cast<device const uint*>(ub + 4u);
            uint usd1 = *reinterpret_cast<device const uint*>(ub + 8u);
            uint usd2 = *reinterpret_cast<device const uint*>(ub + 12u);
            decode_all_q4k_scales_h(du, dumin, usd0, usd1, usd2, u_d_sc, u_neg_dm);
        }

        uint x_base = ib * 256u;
        _Pragma("clang loop unroll(full)")
        for (uint g = 0u; g < 4u; g++) {
            _Pragma("clang loop unroll(full)")
            for (uint step = 0u; step < 4u; step++) {
                uint lane = it + step * 8u;
                uint qs_idx = g * 32u + lane;
                uint gqs_b = uint((gb + 16u)[qs_idx]);
                uint uqs_b = uint((ub + 16u)[qs_idx]);

                // x loaded from device memory on the fly (L1/L2 cached).
                uint xi_lo = x_base + g * 64u + lane;
                uint xi_hi = xi_lo + 32u;
                float xk_lo = (hidden[xi_lo] + delta[xi_lo]) * inv_rms * norm_w[xi_lo];
                float xk_hi = (hidden[xi_hi] + delta[xi_hi]) * inv_rms * norm_w[xi_hi];

                ga[g*2u]    += float(fma(g_d_sc[g*2u],    half(gqs_b & 0xFu),              g_neg_dm[g*2u]))    * xk_lo;
                ga[g*2u+1u] += float(fma(g_d_sc[g*2u+1u], half((gqs_b >> 4u) & 0xFu), g_neg_dm[g*2u+1u])) * xk_hi;
                ua[g*2u]    += float(fma(u_d_sc[g*2u],    half(uqs_b & 0xFu),              u_neg_dm[g*2u]))    * xk_lo;
                ua[g*2u+1u] += float(fma(u_d_sc[g*2u+1u], half((uqs_b >> 4u) & 0xFu), u_neg_dm[g*2u+1u])) * xk_hi;
            }
        }
    }

    if (active) {
        float gate_acc = (ga[0]+ga[1]+ga[2]+ga[3]) + (ga[4]+ga[5]+ga[6]+ga[7]);
        float up_acc   = (ua[0]+ua[1]+ua[2]+ua[3]) + (ua[4]+ua[5]+ua[6]+ua[7]);
        gate_acc = simd_sum(gate_acc);
        up_acc   = simd_sum(up_acc);

        if (siitg == 0u) {
            float g = gate_acc;
            out[row] = apply_activation(g, up_acc);
        }
    }
}

// ── ffn_q4k_gate_up_silu_normed_hidden_v5 ─────────────────────────────────
//
// V5 fused Q4_K FFN: Adapts the matvec_q4k_f32_v4 approach to the fused
// gate+up+SiLU kernel. Key improvements over V4 fused FFN:
//
// 1. Stages normed input (hidden*inv_rms*norm_w) into yl[16]/yh[16] registers
//    per block, then reuses across gate AND up weight rows (4 weight rows total
//    per block: gate[r0], up[r0], gate[r1], up[r1])
// 2. Uses deferred scaling: d * (Σ ql*x) - dmin * (Σ x), fewer FMAs per element
// 3. uint16_t weight reads + bitmask nibble extraction (no shifts)
//
// Dispatch: 2 SIMDs × 2 output rows/SIMD = 4 output rows per TG.
//   dispatch_thread_groups( (ceil(i_dim/4), 1, 1), (64, 1, 1) )
//
// Attribution: inner loop structure adapted from matvec_q4k_f32_v4, itself from
//   ggml-org/llama.cpp Metal kernel_mul_mv_q4_K_f32.
//
kernel void ffn_q4k_gate_up_silu_normed_hidden_v5(
    device const uchar* W_gate  [[ buffer(0) ]],
    device const uchar* W_up    [[ buffer(1) ]],
    device const float* hidden  [[ buffer(2) ]],
    device const float* norm_w  [[ buffer(3) ]],
    device       float* out     [[ buffer(4) ]],
    constant     uint&  cols    [[ buffer(5) ]],
    constant     float& rms_eps [[ buffer(6) ]],
    constant     uint&  i_dim   [[ buffer(7) ]],
    uint tgid  [[ threadgroup_position_in_grid ]],
    uint sgitg [[ simdgroup_index_in_threadgroup ]],
    uint siitg [[ thread_index_in_simdgroup ]],
    uint tid   [[ thread_index_in_threadgroup ]],
    threadgroup float* tg_mem [[ threadgroup(0) ]])
{
    // ── RMS norm (shared across TG) ──
    threadgroup float& tg_inv_rms = tg_mem[2];

    const uint rc = HAS_FC_FFN_COLS ? FC_FFN_COLS : cols;
    const uint ri = HAS_FC_FFN_IDIM ? FC_FFN_IDIM : i_dim;

    // SG0-only reduction (stride-32) for bit-identical inv_rms.
    if (sgitg == 0u) {
        float sum_sq = 0.0f;
        for (uint i = siitg; i < rc; i += 32u) {
            float v = hidden[i];
            sum_sq += v * v;
        }
        sum_sq = simd_sum(sum_sq);
        if (siitg == 0u)
            tg_inv_rms = rsqrt(sum_sq / float(rc) + rms_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = tg_inv_rms;

    // ── Thread partitioning (V4 matvec style) ──
    const uint ix = siitg / 8u;   // 0..3 — block stride index
    const uint it = siitg % 8u;   // 0..7 — position within block
    const uint iq = it / 4u;      // 0 or 1 — half-block selector
    const uint ir = it % 4u;      // 0..3 — quarter within half

    const uint n_blocks    = rc / 256u;
    const uint block_bytes = 144u;
    const uint row_bytes   = n_blocks * block_bytes;

    // 2 output rows per SIMD, 2 SIMDs = 4 output rows per TG
    const uint base_row = tgid * 4u + sgitg * 2u;
    if (base_row >= ri) return;

    // Always process exactly 2 rows — guard the WRITE not the loop.
    // This lets the compiler fully unroll the row loop (matching llama.cpp's template<nr0=2>).
    float gate_sumf[2] = {0.f, 0.f};
    float up_sumf[2] = {0.f, 0.f};

    constexpr ushort kmask1 = 0x3F3Fu;
    constexpr ushort kmask2 = 0x0F0Fu;
    constexpr ushort kmask3 = 0xC0C0u;

    for (uint ib = ix; ib < n_blocks; ib += 4u) {
        // ── Phase 1: Stage normed input into registers ──
        uint x_base = ib * 256u + 64u * iq + 8u * ir;
        device const float* hp = hidden + x_base;
        device const float* np = norm_w + x_base;

        float yl[16], yh[16];
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (uint i = 0u; i < 8u; i++) {
            yl[i]     = hp[i]       * inv_rms * np[i];       sumy[0] += yl[i];
            yl[i + 8] = hp[i + 32]  * inv_rms * np[i + 32];  sumy[1] += yl[i + 8];
            yh[i]     = hp[i + 128] * inv_rms * np[i + 128]; sumy[2] += yh[i];
            yh[i + 8] = hp[i + 160] * inv_rms * np[i + 160]; sumy[3] += yh[i + 8];
        }

        // ── Phase 2: Process gate + up for each output row ──
        // Pointer-walk: start at row 0 block, advance by row_bytes each row.
        device const uchar* gate_blk = W_gate + (ulong)base_row * (ulong)row_bytes + (ulong)ib * block_bytes;
        device const uchar* up_blk   = W_up   + (ulong)base_row * (ulong)row_bytes + (ulong)ib * block_bytes;
        _Pragma("clang loop unroll(full)")
        for (short row = 0; row < 2; row++) {
            // --- Gate weight ---
            {
                device const half* dh = reinterpret_cast<device const half*>(gate_blk);
                device const ushort* sc = reinterpret_cast<device const ushort*>(gate_blk + 4u) + iq;
                ushort sc16[4];
                sc16[0] = sc[0] & kmask1;
                sc16[1] = sc[2] & kmask1;
                sc16[2] = ((sc[4])      & kmask2) | ((sc[0] & kmask3) >> 2);
                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
                thread const uchar* sc8 = reinterpret_cast<thread const uchar*>(sc16);

                device const ushort* q1 = reinterpret_cast<device const ushort*>(gate_blk + 16u) + 16u * iq + 4u * ir;
                device const ushort* q2 = q1 + 32u;

                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                _Pragma("clang loop unroll(full)")
                for (short i = 0; i < 4; i++) {
                    acc1[0] += yl[2*i + 0] * float(q1[i] & 0x000Fu);
                    acc1[1] += yl[2*i + 1] * float(q1[i] & 0x0F00u);
                    acc1[2] += yl[2*i + 8] * float(q1[i] & 0x00F0u);
                    acc1[3] += yl[2*i + 9] * float(q1[i] & 0xF000u);
                    acc2[0] += yh[2*i + 0] * float(q2[i] & 0x000Fu);
                    acc2[1] += yh[2*i + 1] * float(q2[i] & 0x0F00u);
                    acc2[2] += yh[2*i + 8] * float(q2[i] & 0x00F0u);
                    acc2[3] += yh[2*i + 9] * float(q2[i] & 0xF000u);
                }

                gate_sumf[row] +=
                    float(dh[0]) * ((acc1[0] + (1.f/256.f) * acc1[1]) * float(sc8[0]) +
                                    (acc1[2] + (1.f/256.f) * acc1[3]) * float(sc8[1]) * (1.f/16.f) +
                                    (acc2[0] + (1.f/256.f) * acc2[1]) * float(sc8[4]) +
                                    (acc2[2] + (1.f/256.f) * acc2[3]) * float(sc8[5]) * (1.f/16.f)) -
                    float(dh[1]) * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3]) +
                                    sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));
            }

            // --- Up weight ---
            {
                device const half* dh = reinterpret_cast<device const half*>(up_blk);
                device const ushort* sc = reinterpret_cast<device const ushort*>(up_blk + 4u) + iq;
                ushort sc16[4];
                sc16[0] = sc[0] & kmask1;
                sc16[1] = sc[2] & kmask1;
                sc16[2] = ((sc[4])      & kmask2) | ((sc[0] & kmask3) >> 2);
                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
                thread const uchar* sc8 = reinterpret_cast<thread const uchar*>(sc16);

                device const ushort* q1 = reinterpret_cast<device const ushort*>(up_blk + 16u) + 16u * iq + 4u * ir;
                device const ushort* q2 = q1 + 32u;

                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                _Pragma("clang loop unroll(full)")
                for (short i = 0; i < 4; i++) {
                    acc1[0] += yl[2*i + 0] * float(q1[i] & 0x000Fu);
                    acc1[1] += yl[2*i + 1] * float(q1[i] & 0x0F00u);
                    acc1[2] += yl[2*i + 8] * float(q1[i] & 0x00F0u);
                    acc1[3] += yl[2*i + 9] * float(q1[i] & 0xF000u);
                    acc2[0] += yh[2*i + 0] * float(q2[i] & 0x000Fu);
                    acc2[1] += yh[2*i + 1] * float(q2[i] & 0x0F00u);
                    acc2[2] += yh[2*i + 8] * float(q2[i] & 0x00F0u);
                    acc2[3] += yh[2*i + 9] * float(q2[i] & 0xF000u);
                }

                up_sumf[row] +=
                    float(dh[0]) * ((acc1[0] + (1.f/256.f) * acc1[1]) * float(sc8[0]) +
                                    (acc1[2] + (1.f/256.f) * acc1[3]) * float(sc8[1]) * (1.f/16.f) +
                                    (acc2[0] + (1.f/256.f) * acc2[1]) * float(sc8[4]) +
                                    (acc2[2] + (1.f/256.f) * acc2[3]) * float(sc8[5]) * (1.f/16.f)) -
                    float(dh[1]) * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3]) +
                                    sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));
            }

            gate_blk += row_bytes;  // pointer walk to next row
            up_blk   += row_bytes;
        }
    }

    // ── Reduction + SiLU + write ──
    const float g0 = simd_sum(gate_sumf[0]);
    const float u0 = simd_sum(up_sumf[0]);
    const float g1 = simd_sum(gate_sumf[1]);
    const float u1 = simd_sum(up_sumf[1]);

    if (siitg == 0u) {
        out[base_row] = apply_activation(g0, u0);
        if (base_row + 1u < ri) {
            out[base_row + 1u] = apply_activation(g1, u1);
        }
    }
}

// ── ffn_q4k_gate_up_silu_normed_hidden_4row_v4 ────────────────────────────
//
// Hidden-only version of the stride-4 fused Q4_K FFN kernel above.
//
kernel void ffn_q4k_gate_up_silu_normed_hidden_4row_v4(
    device const uchar* W_gate  [[ buffer(0) ]],
    device const uchar* W_up    [[ buffer(1) ]],
    device const float* hidden  [[ buffer(2) ]],
    device const float* norm_w  [[ buffer(3) ]],
    device       float* out     [[ buffer(4) ]],
    constant     uint&  cols    [[ buffer(5) ]],
    constant     float& rms_eps [[ buffer(6) ]],
    constant     uint&  i_dim   [[ buffer(7) ]],
    uint tgid  [[ threadgroup_position_in_grid ]],
    uint sgitg [[ simdgroup_index_in_threadgroup ]],
    uint siitg [[ thread_index_in_simdgroup ]],
    uint tid   [[ thread_index_in_threadgroup ]],
    threadgroup float* tg_mem [[ threadgroup(0) ]])
{
    threadgroup float& tg_inv_rms = tg_mem[2];

    const uint rc   = HAS_FC_FFN_COLS ? FC_FFN_COLS : cols;
    const uint ri   = HAS_FC_FFN_IDIM ? FC_FFN_IDIM : i_dim;
    const uint n_blocks    = rc / 256u;
    const uint block_bytes = 144u;
    const uint base_row    = tgid * FFN_Q4K_ROWS_PER_TG;
    const uint row         = base_row + sgitg;
    const bool active      = (row < ri);

    const uint ix = siitg / 8u;
    const uint it = siitg % 8u;

    // SG0-only reduction (stride-32) for bit-identical inv_rms.
    if (sgitg == 0u) {
        float sum_sq = 0.0f;
        for (uint i = siitg; i < rc; i += 32u) {
            float v = hidden[i];
            sum_sq += v * v;
        }
        sum_sq = simd_sum(sum_sq);
        if (siitg == 0u)
            tg_inv_rms = rsqrt(sum_sq / float(rc) + rms_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = tg_inv_rms;

    device const uchar* gate_base = W_gate + (ulong)row * n_blocks * block_bytes;
    device const uchar* up_base   = W_up   + (ulong)row * n_blocks * block_bytes;

    float ga[8] = {0,0,0,0,0,0,0,0};
    float ua[8] = {0,0,0,0,0,0,0,0};

    for (uint ib = ix; ib < n_blocks; ib += 4u) {
        if (!active) continue;

        device const uchar* gb = gate_base + ib * block_bytes;
        device const uchar* ub = up_base   + ib * block_bytes;

        uint gdw = *reinterpret_cast<device const uint*>(gb);
        half dg    = as_type<half>(ushort(gdw        & 0xFFFFu));
        half dgmin = as_type<half>(ushort(gdw >> 16u));

        uint udw = *reinterpret_cast<device const uint*>(ub);
        half du    = as_type<half>(ushort(udw        & 0xFFFFu));
        half dumin = as_type<half>(ushort(udw >> 16u));

        half g_d_sc[8], g_neg_dm[8], u_d_sc[8], u_neg_dm[8];
        {
            uint gsd0 = *reinterpret_cast<device const uint*>(gb + 4u);
            uint gsd1 = *reinterpret_cast<device const uint*>(gb + 8u);
            uint gsd2 = *reinterpret_cast<device const uint*>(gb + 12u);
            decode_all_q4k_scales_h(dg, dgmin, gsd0, gsd1, gsd2, g_d_sc, g_neg_dm);
        }
        {
            uint usd0 = *reinterpret_cast<device const uint*>(ub + 4u);
            uint usd1 = *reinterpret_cast<device const uint*>(ub + 8u);
            uint usd2 = *reinterpret_cast<device const uint*>(ub + 12u);
            decode_all_q4k_scales_h(du, dumin, usd0, usd1, usd2, u_d_sc, u_neg_dm);
        }

        uint x_base = ib * 256u;
        _Pragma("clang loop unroll(full)")
        for (uint g = 0u; g < 4u; g++) {
            _Pragma("clang loop unroll(full)")
            for (uint step = 0u; step < 4u; step++) {
                uint lane = it + step * 8u;
                uint qs_idx = g * 32u + lane;
                uint gqs_b = uint((gb + 16u)[qs_idx]);
                uint uqs_b = uint((ub + 16u)[qs_idx]);

                uint xi_lo = x_base + g * 64u + lane;
                uint xi_hi = xi_lo + 32u;
                float xk_lo = hidden[xi_lo] * inv_rms * norm_w[xi_lo];
                float xk_hi = hidden[xi_hi] * inv_rms * norm_w[xi_hi];

                ga[g*2u]    += float(fma(g_d_sc[g*2u],    half(gqs_b & 0xFu),              g_neg_dm[g*2u]))    * xk_lo;
                ga[g*2u+1u] += float(fma(g_d_sc[g*2u+1u], half((gqs_b >> 4u) & 0xFu), g_neg_dm[g*2u+1u])) * xk_hi;
                ua[g*2u]    += float(fma(u_d_sc[g*2u],    half(uqs_b & 0xFu),              u_neg_dm[g*2u]))    * xk_lo;
                ua[g*2u+1u] += float(fma(u_d_sc[g*2u+1u], half((uqs_b >> 4u) & 0xFu), u_neg_dm[g*2u+1u])) * xk_hi;
            }
        }
    }

    if (active) {
        float gate_acc = (ga[0]+ga[1]+ga[2]+ga[3]) + (ga[4]+ga[5]+ga[6]+ga[7]);
        float up_acc   = (ua[0]+ua[1]+ua[2]+ua[3]) + (ua[4]+ua[5]+ua[6]+ua[7]);
        gate_acc = simd_sum(gate_acc);
        up_acc   = simd_sum(up_acc);

        if (siitg == 0u) {
            float g = gate_acc;
            out[row] = apply_activation(g, up_acc);
        }
    }
}

// ── ffn_q4k_gate_up_silu_normed_hidden_4row_v4_4sg ──────────────────────
//
// 4-SIMD-group variant of the hidden-only fused Q4_K FFN kernel.
// Uses 128 threads/TG (4 SIMDs), processing 4 rows simultaneously.
// Halves threadgroup count vs 2-SIMD version, improving input vector
// amortization and reducing dispatch overhead.
//
kernel void ffn_q4k_gate_up_silu_normed_hidden_4row_v4_4sg(
    device const uchar* W_gate  [[ buffer(0) ]],
    device const uchar* W_up    [[ buffer(1) ]],
    device const float* hidden  [[ buffer(2) ]],
    device const float* norm_w  [[ buffer(3) ]],
    device       float* out     [[ buffer(4) ]],
    constant     uint&  cols    [[ buffer(5) ]],
    constant     float& rms_eps [[ buffer(6) ]],
    constant     uint&  i_dim   [[ buffer(7) ]],
    uint tgid  [[ threadgroup_position_in_grid ]],
    uint sgitg [[ simdgroup_index_in_threadgroup ]],
    uint siitg [[ thread_index_in_simdgroup ]],
    uint tid   [[ thread_index_in_threadgroup ]],
    threadgroup float* tg_mem [[ threadgroup(0) ]])
{
    threadgroup float& tg_inv_rms = tg_mem[4];

    const uint rc   = HAS_FC_FFN_COLS ? FC_FFN_COLS : cols;
    const uint ri   = HAS_FC_FFN_IDIM ? FC_FFN_IDIM : i_dim;
    const uint n_blocks    = rc / 256u;
    const uint block_bytes = 144u;
    const uint base_row    = tgid * 4u;
    const uint row         = base_row + sgitg;
    const bool active      = (row < ri);

    const uint ix = siitg / 8u;
    const uint it = siitg % 8u;

    // SG0-only reduction (stride-32) for bit-identical inv_rms.
    if (sgitg == 0u) {
        float sum_sq = 0.0f;
        for (uint i = siitg; i < rc; i += 32u) {
            float v = hidden[i];
            sum_sq += v * v;
        }
        sum_sq = simd_sum(sum_sq);
        if (siitg == 0u)
            tg_inv_rms = rsqrt(sum_sq / float(rc) + rms_eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = tg_inv_rms;

    device const uchar* gate_base = W_gate + (ulong)row * n_blocks * block_bytes;
    device const uchar* up_base   = W_up   + (ulong)row * n_blocks * block_bytes;

    float ga[8] = {0,0,0,0,0,0,0,0};
    float ua[8] = {0,0,0,0,0,0,0,0};

    for (uint ib = ix; ib < n_blocks; ib += 4u) {
        if (!active) continue;

        device const uchar* gb = gate_base + ib * block_bytes;
        device const uchar* ub = up_base   + ib * block_bytes;

        uint gdw = *reinterpret_cast<device const uint*>(gb);
        half dg    = as_type<half>(ushort(gdw        & 0xFFFFu));
        half dgmin = as_type<half>(ushort(gdw >> 16u));

        uint udw = *reinterpret_cast<device const uint*>(ub);
        half du    = as_type<half>(ushort(udw        & 0xFFFFu));
        half dumin = as_type<half>(ushort(udw >> 16u));

        half g_d_sc[8], g_neg_dm[8], u_d_sc[8], u_neg_dm[8];
        {
            uint gsd0 = *reinterpret_cast<device const uint*>(gb + 4u);
            uint gsd1 = *reinterpret_cast<device const uint*>(gb + 8u);
            uint gsd2 = *reinterpret_cast<device const uint*>(gb + 12u);
            decode_all_q4k_scales_h(dg, dgmin, gsd0, gsd1, gsd2, g_d_sc, g_neg_dm);
        }
        {
            uint usd0 = *reinterpret_cast<device const uint*>(ub + 4u);
            uint usd1 = *reinterpret_cast<device const uint*>(ub + 8u);
            uint usd2 = *reinterpret_cast<device const uint*>(ub + 12u);
            decode_all_q4k_scales_h(du, dumin, usd0, usd1, usd2, u_d_sc, u_neg_dm);
        }

        uint x_base = ib * 256u;
        _Pragma("clang loop unroll(full)")
        for (uint g = 0u; g < 4u; g++) {
            _Pragma("clang loop unroll(full)")
            for (uint step = 0u; step < 4u; step++) {
                uint lane = it + step * 8u;
                uint qs_idx = g * 32u + lane;
                uint gqs_b = uint((gb + 16u)[qs_idx]);
                uint uqs_b = uint((ub + 16u)[qs_idx]);

                uint xi_lo = x_base + g * 64u + lane;
                uint xi_hi = xi_lo + 32u;
                float xk_lo = hidden[xi_lo] * inv_rms * norm_w[xi_lo];
                float xk_hi = hidden[xi_hi] * inv_rms * norm_w[xi_hi];

                ga[g*2u]    += float(fma(g_d_sc[g*2u],    half(gqs_b & 0xFu),              g_neg_dm[g*2u]))    * xk_lo;
                ga[g*2u+1u] += float(fma(g_d_sc[g*2u+1u], half((gqs_b >> 4u) & 0xFu), g_neg_dm[g*2u+1u])) * xk_hi;
                ua[g*2u]    += float(fma(u_d_sc[g*2u],    half(uqs_b & 0xFu),              u_neg_dm[g*2u]))    * xk_lo;
                ua[g*2u+1u] += float(fma(u_d_sc[g*2u+1u], half((uqs_b >> 4u) & 0xFu), u_neg_dm[g*2u+1u])) * xk_hi;
            }
        }
    }

    if (active) {
        float gate_acc = (ga[0]+ga[1]+ga[2]+ga[3]) + (ga[4]+ga[5]+ga[6]+ga[7]);
        float up_acc   = (ua[0]+ua[1]+ua[2]+ua[3]) + (ua[4]+ua[5]+ua[6]+ua[7]);
        gate_acc = simd_sum(gate_acc);
        up_acc   = simd_sum(up_acc);

        if (siitg == 0u) {
            float g = gate_acc;
            out[row] = apply_activation(g, up_acc);
        }
    }
}

// ── ffn_q5k_gate_up_silu_normed ───────────────────────────────────────────
//
// Fused residual-norm + SwiGLU FFN for Q5_K weight matrices.
// Identical to the Q4_K variant but splices in the 5th high bit from qh.
//
// Q5_K block layout (176 bytes / 256 elements):
//   bytes  0-1 : d    — fp16 super-scale
//   bytes  2-3 : dmin — fp16 super-min-scale
//   bytes  4-15: scales[12] — packed 6-bit scale + 6-bit min
//   bytes 16-47: qh[32]    — 1 high bit per element (256 bits)
//   bytes 48-175: qs[128]  — nibble-packed 4-bit weights
//
// dispatch_thread_groups( (i_dim, 1, 1), (32, 1, 1) )
//
