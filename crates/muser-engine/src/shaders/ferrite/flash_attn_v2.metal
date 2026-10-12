// ── flash_attn_v2: FlashAttention-2 with simdgroup matrix multiply ────────
//
// Adapted from llama.cpp kernel_flash_attn_ext (ggml-org/llama.cpp,
// ggml/src/ggml-metal/ggml-metal.metal).  Uses simdgroup_multiply_accumulate
// for Q×K^T and softmax(QK^T)×V instead of scalar dot-product + simd_sum.
//
// Requirements:
//   - K/V must be pre-written to the f16 KV cache before this kernel runs
//     (use store_kv_batch_f16 + barrier).
//   - head_dim MUST be 128.
//
// Compile-time constants:
//   FA2_Q   = 8   — queries per threadgroup
//   FA2_C   = 64  — KV tokens per tile
//   FA2_NSG = 4   — SIMD groups per threadgroup
//
// Dispatch:
//   thread_groups = (ceil(batch_size / FA2_Q), n_heads, 1)
//   threads       = (FA2_NSG × 32, 1, 1) = (128, 1, 1)
//
// TG memory: ~6 KB  (sq 2 KB + so 2 KB + ss 2 KB)

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ── Compile-time parameters ──────────────────────────────────────────────
constant constexpr short DK   = 128;        // head dimension  (fixed)
constant constexpr short DK4  = DK  / 4;    // = 32
constant constexpr short DK8  = DK  / 8;    // = 16
constant constexpr short Q    = 8;           // queries per threadgroup
constant constexpr short C    = 64;          // KV tokens per tile
constant constexpr short NSG  = 4;           // SIMD groups per TG
constant constexpr short NW   = 32;          // simd width
constant constexpr short NQ   = Q / NSG;     // = 2 queries per SG
constant constexpr short NC   = (C / 8) / NSG; // = 2 KV 8-chunks per SG
constant constexpr short PV   = DK;          // = 128 (padded output dim)
constant constexpr short PV4  = PV / 4;      // = 32
constant constexpr short PV8  = PV / 8;      // = 16
constant constexpr short SH   = C;           // stride in score buffer (floats)

// Type aliases (following llama.cpp FA_TYPES)
using q_t    = half;
using q4_t   = half4;
using k_t    = half;
using v_t    = half;
using o_t    = float;
using o4_t   = float4;
using s_t    = float;

using q8x8_t  = simdgroup_half8x8;
using k8x8_t  = simdgroup_half8x8;
using v8x8_t  = simdgroup_half8x8;
using o8x8_t  = simdgroup_float8x8;
using s8x8_t  = simdgroup_float8x8;
using qk8x8_t = simdgroup_float8x8;

// ── Kernel ───────────────────────────────────────────────────────────────
kernel void flash_attn_v2(
    device const float*  Q_batch      [[ buffer(0)  ]],  // [B × n_heads × DK] f32
    device const half*   K_cache      [[ buffer(1)  ]],  // [n_kv_heads × max_seq × DK] f16
    device const half*   V_cache      [[ buffer(2)  ]],  // [n_kv_heads × max_seq × DK] f16
    device       float*  out          [[ buffer(3)  ]],  // [B × n_heads × DK] f32
    constant     uint&   batch_size   [[ buffer(4)  ]],
    constant     uint&   start_pos    [[ buffer(5)  ]],  // positions already in cache
    constant     uint&   n_heads      [[ buffer(6)  ]],
    constant     uint&   n_kv_heads   [[ buffer(7)  ]],
    constant     uint&   cache_stride [[ buffer(8)  ]],  // max_seq × DK (f16 elems)
    constant     float&  attn_scale_override [[ buffer(9) ]],  // >0: use instead of rsqrt(DK)
    constant     uint&   window_size  [[ buffer(10) ]],  // sliding window (0 = global attention)
    constant     uint&   head_major   [[ buffer(11) ]],  // 0: [pos,kv_head,dim], 1: [kv_head,pos,dim]
    constant     uint&   cache_logical_base [[ buffer(12) ]], // logical position stored at row 0
    constant     float&  attn_softcap[[ buffer(15) ]],  // logit soft-cap (0 = disabled)
    uint3  tgpig  [[ threadgroup_position_in_grid ]],
    ushort tiisg  [[ thread_index_in_simdgroup ]],
    ushort sgitg  [[ simdgroup_index_in_threadgroup ]])
{
    // ── Grid position ────────────────────────────────────────────────────
    const short iq1     = short(tgpig.x) * Q;          // first query index
    const short q_head  = short(tgpig.y);               // Q-head
    const short kv_head = q_head * short(n_kv_heads) / short(n_heads); // GQA
    const uint  q_dim   = n_heads * DK;

    // ── TG memory layout ─────────────────────────────────────────────────
    //  sq: [Q × DK]  half  = 2048 B
    //  so: [Q × PV]  float = 4096 B  (llama FA_TYPES: f32 O, not f16)
    //  ss: [Q × SH]  float = 2048 B
    //  Total: 8192 B = 8 KB
    // f16 O overflows around 32k full-attn (S·V exceeds 65504, then Inf*0
    // during the online rescale writes NaN into every hidden channel).
    threadgroup q_t  sq_buf [Q * DK];     // queries in half
    threadgroup o_t  so_buf [Q * PV];     // output accumulator in float
    threadgroup s_t  ss_buf [Q * SH];     // QK^T scores in float

    threadgroup q4_t* sq4 = (threadgroup q4_t*) sq_buf;
    threadgroup o4_t* so4 = (threadgroup o4_t*) so_buf;

    // K/V cache pointers and position stride for this KV-head. Ferrite's
    // growing planes are head-major; Muse SWA rings are token-major so their
    // existing ring-aware decode path can retain compact physical rows.
    const uint kv_stride = head_major != 0u ? DK : n_kv_heads * DK;
    device const k_t* pk_base = K_cache + (head_major != 0u
        ? uint(kv_head) * cache_stride
        : uint(kv_head) * DK);
    device const v_t* pv_base = V_cache + (head_major != 0u
        ? uint(kv_head) * cache_stride
        : uint(kv_head) * DK);

    // ── Load Q: f32 → f16 into threadgroup memory ────────────────────────
    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj * NSG + sgitg;
        device const float4* q4 = (device const float4*)
            (Q_batch + uint(iq1 + j) * q_dim + uint(q_head) * DK);
        for (short i = tiisg; i < DK4; i += NW) {
            if (iq1 + j < short(batch_size)) {
                sq4[j * DK4 + i] = (q4_t) q4[i];
            } else {
                sq4[j * DK4 + i] = q4_t(0);
            }
        }
    }

    // ── Zero output accumulator ──────────────────────────────────────────
    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj * NSG + sgitg;
        for (short i = tiisg; i < PV4; i += NW) {
            so4[j * PV4 + i] = o4_t(0);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Online softmax state (per-query, in registers) ───────────────────
    float S[NQ] = { 0.0f, 0.0f };
    float M[NQ] = { -FLT_MAX/2, -FLT_MAX/2 };

    const float scale = (attn_scale_override > 0.0f) ? attn_scale_override : rsqrt(float(DK));

    // ── Max KV position for this TG (exclusive) ─────────────────────────
    const uint max_kv = start_pos + min(uint(iq1 + Q), batch_size);

    // SWA loop-start cap uses the earliest query in this TG. Each query gets
    // its own exact mask below; using the last query's start for every row
    // would incorrectly discard up to Q-1 visible keys after the first wrap.
    const uint first_causal_pos = start_pos + uint(iq1) + 1u;
    const uint swa_start_raw = (window_size > 0u && first_causal_pos > window_size)
        ? (first_causal_pos - window_size) : 0u;
    const uint win_start = max(cache_logical_base, (swa_start_raw / C) * C);

    // ══════════════════════════════════════════════════════════════════════
    // MAIN TILE LOOP over KV dimension
    // ══════════════════════════════════════════════════════════════════════
    for (uint ic = win_start; ic < max_kv; ic += C) {

        // ── Phase 1: Q × K^T via simdgroup_multiply_accumulate ──────────
        {
            device const k_t* pk = pk_base
                + (ic - cache_logical_base + uint(sgitg) * 8) * kv_stride;
            threadgroup const q_t* pq = sq_buf;
            threadgroup s_t* ps = ss_buf + sgitg * 8;

            for (short cc = 0; cc < NC; ++cc) {
                qk8x8_t mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);

                k8x8_t mk[2];
                q8x8_t mq[2];

                #pragma unroll 4
                for (short i = 0; i < DK8/2; ++i) {
                    simdgroup_barrier(mem_flags::mem_none);

                    simdgroup_load(mq[0], pq + 0*8 + 16*i, DK);
                    simdgroup_load(mq[1], pq + 1*8 + 16*i, DK);

                    simdgroup_load(mk[0], pk + 0*8 + 16*i, kv_stride, ulong2(0,0), true);
                    simdgroup_load(mk[1], pk + 1*8 + 16*i, kv_stride, ulong2(0,0), true);

                    simdgroup_barrier(mem_flags::mem_none);

                    simdgroup_multiply_accumulate(mqk, mq[0], mk[0], mqk);
                    simdgroup_multiply_accumulate(mqk, mq[1], mk[1], mqk);
                }

                simdgroup_store(mqk, ps, SH, ulong2(0,0), false);

                pk += 8 * NSG * kv_stride;
                ps += 8 * NSG;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Phase 2: Online softmax ─────────────────────────────────────
        for (short jj = 0; jj < NQ; ++jj) {
            const short j = jj * NSG + sgitg;
            const uint causal_pos = start_pos + uint(iq1 + j) + 1u; // exclusive
            const uint query_swa_start = (window_size > 0u && causal_pos > window_size)
                ? (causal_pos - window_size) : 0u;

            const float m_old = M[jj];

            // Read 2 scores per thread (C=64 / NW=32 = 2), apply scale
            float s0 = -FLT_MAX/2;
            float s1 = -FLT_MAX/2;
            const uint p0 = ic + 2u * tiisg;
            const uint p1 = p0 + 1u;

            // p < query_swa_start: straggler from tile alignment or an older
            // row needed only by an earlier query in this same TG.
            if (p0 < max_kv && p0 < causal_pos && p0 >= query_swa_start)
                s0 = ss_buf[j * SH + 2 * tiisg] * scale;
            if (p1 < max_kv && p1 < causal_pos && p1 >= query_swa_start)
                s1 = ss_buf[j * SH + 2 * tiisg + 1] * scale;

            // New running max
            M[jj] = simd_max(max(M[jj], max(s0, s1)));

            // exp(score - M_new)
            const float ms  = precise::exp(m_old - M[jj]);
            const float e0  = precise::exp(s0 - M[jj]);
            const float e1  = precise::exp(s1 - M[jj]);

            // Update running sum
            S[jj] = S[jj] * ms + simd_sum(e0 + e1);

            // Store normalised exp-scores back for V multiply
            ss_buf[j * SH + 2 * tiisg    ] = e0;
            ss_buf[j * SH + 2 * tiisg + 1] = e1;

            // Rescale old output accumulator: so *= exp(M_old - M_new)
            for (short i = tiisg; i < PV4; i += NW) {
                so4[j * PV4 + i] *= ms;
            }
        }

        // ── Sparse V tile skip: if this tile is fully masked out for every
        //    query in the threadgroup, skip the expensive V load +
        //    simdgroup multiply entirely. Uses threadgroup reduction: each
        //    SG contributes its max exp-score, and we check the tile-wide
        //    max. Masked positions (outside the causal/SWA window) were
        //    left at the s0/s1 sentinel -FLT_MAX/2 above, so their exp-score
        //    underflows to exactly 0.0f; a tile is only skippable when it is
        //    *fully* masked, i.e. that max is exactly 0. A magnitude
        //    threshold like `> 1e-6f` would also drop tiles that carry a
        //    small but genuine softmax weight -- those stay counted in the
        //    denominator `S` but vanish from the `O` accumulation, biasing
        //    every output low. llama.cpp's kernel_flash_attn_ext only skips
        //    exactly-masked tiles, so we match that here.
        threadgroup float tile_max_score[NSG * NQ];
        {
            for (short jj = 0; jj < NQ; ++jj) {
                float local_max = max(ss_buf[(jj * NSG + sgitg) * SH + 2 * tiisg],
                                      ss_buf[(jj * NSG + sgitg) * SH + 2 * tiisg + 1]);
                local_max = simd_max(local_max);
                if (tiisg == 0) {
                    tile_max_score[sgitg * NQ + jj] = local_max;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Check if the entire tile is masked out (every query, every
        // position exactly 0 -- i.e. genuinely outside the window).
        bool skip_v = true;
        for (short i = 0; i < NSG * NQ; ++i) {
            if (tile_max_score[i] > 0.0f) { skip_v = false; break; }
        }

        if (!skip_v) {

        // ── Phase 3: O += softmax(QK^T) × V  ────────────────────────────
        {
            // Each SG handles NO chunks of 8 output columns
            constexpr short NO = PV8 / NSG;  // = 16/4 = 4

            // Load current O from TG memory into registers
            o8x8_t lo[NO];
            {
                threadgroup o_t* sot = so_buf + 8 * sgitg;
                for (short ii = 0; ii < NO; ++ii) {
                    simdgroup_load(lo[ii], sot, PV, ulong2(0,0), false);
                    sot += 8 * NSG;
                }
            }

            // Accumulate: O += S × V
            // Process C/8 = 8 KV chunks, paired into C/16 = 4 pairs
            {
                device const v_t* pv = pv_base
                    + (ic - cache_logical_base) * kv_stride + 8 * sgitg;

                constexpr short NVC = (C / 8) / 2;  // = 4 pairs

                for (short cc = 0; cc < NVC; ++cc) {
                    s8x8_t vs[2];
                    simdgroup_load(vs[0], ss_buf + 16*cc + 0, SH, ulong2(0,0), false);
                    simdgroup_load(vs[1], ss_buf + 16*cc + 8, SH, ulong2(0,0), false);

                    for (short ii = 0; ii < NO/2; ++ii) {
                        v8x8_t mv[4];

                        // V: [8 KV pos × 8 dims], stride = DK between rows
                        simdgroup_load(mv[0], pv + 0*NSG + 16*ii*NSG + 0*8*kv_stride, kv_stride, ulong2(0,0), false);
                        simdgroup_load(mv[1], pv + 8*NSG + 16*ii*NSG + 0*8*kv_stride, kv_stride, ulong2(0,0), false);
                        simdgroup_load(mv[2], pv + 0*NSG + 16*ii*NSG + 1*8*kv_stride, kv_stride, ulong2(0,0), false);
                        simdgroup_load(mv[3], pv + 8*NSG + 16*ii*NSG + 1*8*kv_stride, kv_stride, ulong2(0,0), false);

                        simdgroup_multiply_accumulate(lo[2*ii + 0], vs[0], mv[0], lo[2*ii + 0]);
                        simdgroup_multiply_accumulate(lo[2*ii + 1], vs[0], mv[1], lo[2*ii + 1]);
                        simdgroup_multiply_accumulate(lo[2*ii + 0], vs[1], mv[2], lo[2*ii + 0]);
                        simdgroup_multiply_accumulate(lo[2*ii + 1], vs[1], mv[3], lo[2*ii + 1]);
                    }

                    pv += 2 * 8 * kv_stride;
                }
            }

            // Store updated O back to TG memory
            {
                threadgroup o_t* sot = so_buf + 8 * sgitg;
                for (short ii = 0; ii < NO; ++ii) {
                    simdgroup_store(lo[ii], sot, PV, ulong2(0,0), false);
                    sot += 8 * NSG;
                }
            }
        }

        } // end if (!skip_v)

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ══════════════════════════════════════════════════════════════════════
    // OUTPUT: f32 accumulator with 1/S normalisation
    // ══════════════════════════════════════════════════════════════════════
    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj * NSG + sgitg;
        if (iq1 + j >= short(batch_size)) break;

        device float4* dst4 = (device float4*)
            (out + uint(iq1 + j) * q_dim + uint(q_head) * DK);

        const float inv_S = S[jj] == 0.0f ? 0.0f : 1.0f / S[jj];

        for (short i = tiisg; i < PV4; i += NW) {
            dst4[i] = so4[j * PV4 + i] * inv_S;
        }
    }
}
