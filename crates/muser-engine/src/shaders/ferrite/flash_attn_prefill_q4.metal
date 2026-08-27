// flash_attn_prefill_q4.metal — FlashAttention-2 batch prefill with Q4_0 KV cache.
//
// Adapted from flash_attn_v2.metal (F16 KV) and flash_attn_decode_vec_q4.metal
// (Q4 decode). Combines simdgroup_matrix_multiply attention with inline Q4_0
// dequant/quant, enabling batch prefill directly against Q4 KV cache without
// the F16→Q4 post-conversion step.
//
// Architecture:
//   - Stores incoming K/V as Q4_0 blocks inline (one threadgroup per batch of 8 tokens)
//   - Dequantizes Q4_0 K/V tiles into threadgroup memory (f16) for simdgroup_matrix ops
//   - Same tiling/online-softmax as flash_attn_v2 (Q=8, C=32, NSG=4)
//   - Causal masking applied
//
// Q4_0 block layout (20 bytes per 32 elements):
//   [2B f16 scale] [2B f16 min] [16B: 32 x 4-bit unsigned quants, 2/byte, low nibble first]
//   Dequant: val = u4_quant * scale + min
//
// Requirements:
//   - head_dim MUST be 128
//   - K/V cache must be pre-allocated with q4_cache_size per KV head
//
// Dispatch:
//   thread_groups = (ceil(batch_size / Q4P_Q), n_heads, 1)
//   threads       = (Q4P_NSG * 32, 1, 1) = (128, 1, 1)

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// ── Compile-time parameters ──────────────────────────────────────────────
constant constexpr short Q4P_DK   = 128;          // head dimension (fixed)
constant constexpr short Q4P_DK4  = Q4P_DK / 4;   // = 32
constant constexpr short Q4P_DK8  = Q4P_DK / 8;   // = 16
constant constexpr short Q4P_Q    = 8;             // queries per threadgroup
constant constexpr short Q4P_C    = 32;            // KV tokens per tile (smaller than F16 due to dequant cost)
constant constexpr short Q4P_NSG  = 4;             // SIMD groups per TG
constant constexpr short Q4P_NW   = 32;            // simd width
constant constexpr short Q4P_NQ   = Q4P_Q / Q4P_NSG;    // = 2 queries per SG
constant constexpr short Q4P_NC   = (Q4P_C / 8) / Q4P_NSG; // = 1 KV 8-chunk per SG
constant constexpr short Q4P_PV   = Q4P_DK;        // = 128 (padded output dim)
constant constexpr short Q4P_PV4  = Q4P_PV / 4;    // = 32
constant constexpr short Q4P_PV8  = Q4P_PV / 8;    // = 16
constant constexpr short Q4P_SH   = Q4P_C;         // stride in score buffer (floats)

// Q4_0 constants
constant constexpr uint Q4P_BLOCK_BYTES = 20u;     // 2 (scale) + 2 (min) + 16 (quants)
constant constexpr uint Q4P_BPR = (Q4P_DK + 31u) / 32u; // = 4 blocks per row

// Type aliases
using q4p_q_t    = half;
using q4p_q4_t   = half4;
using q4p_k_t    = half;
using q4p_v_t    = half;
using q4p_o_t    = half;
using q4p_o4_t   = half4;
using q4p_s_t    = float;

using q4p_q8x8_t  = simdgroup_half8x8;
using q4p_k8x8_t  = simdgroup_half8x8;
using q4p_v8x8_t  = simdgroup_half8x8;
using q4p_o8x8_t  = simdgroup_half8x8;
using q4p_s8x8_t  = simdgroup_float8x8;
using q4p_qk8x8_t = simdgroup_float8x8;

// ── Q4_0 dequantize: load one row of Q4_0 blocks into f16 threadgroup memory ──
//
// Each thread loads 1 element from each of the 4 blocks (head_dim=128 → 4 blocks).
// tid: thread index in TG (0..127), dst: threadgroup half buffer of size DK.
inline void dequant_q4_row_to_tg(
    device const uchar* row_base,   // pointer to first Q4 block of this row
    threadgroup q4p_k_t* dst,       // [DK] output in f16
    ushort tid)                     // thread index in TG (0..127)
{
    // tid maps to element position: block_idx = tid / 32, elem_in_block = tid % 32
    const ushort block_idx = tid / Q4P_NW;
    const ushort elem = tid % Q4P_NW;

    if (block_idx < Q4P_BPR) {
        device const uchar* blk = row_base + block_idx * Q4P_BLOCK_BYTES;
        float scale = float(*((device const half*)(blk + 0)));
        float bmin  = float(*((device const half*)(blk + 2)));
        device const uchar* qs = blk + 4u;
        uchar packed = qs[elem / 2u];
        uchar nibble = (elem & 1u) ? (packed >> 4u) : (packed & 0xFu);
        dst[tid] = q4p_k_t(float(nibble) * scale + bmin);
    }
}

// ── Q4_0 quantize: compress a batch of f32 K/V values into Q4_0 blocks ──
//
// Quantizes one row of DK elements from batch K/V (f32) into Q4 cache.
// Each thread in the TG cooperates: thread tid handles element tid.
inline void quant_f32_row_to_q4(
    device const float* src,         // [DK] source f32 row
    device uchar* dst_base,          // pointer to first Q4 block for this row
    ushort tid,                      // thread index in TG (0..127)
    threadgroup float* scratch)      // [8] scratch for block min/max reduction
{
    const ushort block_idx = tid / Q4P_NW;
    const ushort elem = tid % Q4P_NW;

    if (block_idx < Q4P_BPR) {
        const uint base = block_idx * 32u;
        float val = (base + elem < uint(Q4P_DK)) ? src[base + elem] : 0.0f;

        // SIMD reduction for min/max within this block's 32 threads
        float bmin = simd_min(val);
        float bmax = simd_max(val);

        float range = bmax - bmin;
        float scale = range > 0.0f ? range / 15.0f : 0.0f;
        float inv   = scale > 0.0f ? 1.0f / scale : 0.0f;

        // Quantize
        uchar q = (uchar)clamp((int)round((val - bmin) * inv), 0, 15);

        // Write Q4 block — only thread 0 of each block writes scale/min
        device uchar* blk = dst_base + block_idx * Q4P_BLOCK_BYTES;
        if (elem == 0u) {
            *((device half*)(blk + 0)) = half(scale);
            *((device half*)(blk + 2)) = half(bmin);
        }
        // Pack nibble pairs: even threads write low nibble, odd threads write high
        // Use simd_shuffle to pair up
        uchar partner = simd_shuffle(q, (elem | 1u));
        if ((elem & 1u) == 0u) {
            device uchar* qs = blk + 4u;
            qs[elem / 2u] = q | (partner << 4u);
        }
    }
}

// ── Kernel ───────────────────────────────────────────────────────────────
kernel void flash_attn_prefill_q4(
    device const float*  Q_batch       [[ buffer(0)  ]],  // [B x n_heads x DK] f32
    device       uchar*  K_cache_q4    [[ buffer(1)  ]],  // [n_kv_heads x q4_cache_stride] bytes
    device       uchar*  V_cache_q4    [[ buffer(2)  ]],  // [n_kv_heads x q4_cache_stride] bytes
    device       float*  out           [[ buffer(3)  ]],  // [B x n_heads x DK] f32
    device const float*  K_batch       [[ buffer(4)  ]],  // [B x n_kv_heads x DK] f32 — new K
    device const float*  V_batch       [[ buffer(5)  ]],  // [B x n_kv_heads x DK] f32 — new V
    constant     uint&   batch_size    [[ buffer(6)  ]],
    constant     uint&   start_pos     [[ buffer(7)  ]],  // positions already in cache
    constant     uint&   n_heads       [[ buffer(8)  ]],
    constant     uint&   n_kv_heads    [[ buffer(9)  ]],
    constant     uint&   q4_stride     [[ buffer(10) ]],  // bytes per KV head in Q4 cache
    uint3  tgpig  [[ threadgroup_position_in_grid ]],
    ushort tiisg  [[ thread_index_in_simdgroup ]],
    ushort sgitg  [[ simdgroup_index_in_threadgroup ]])
{
    // ── Grid position ────────────────────────────────────────────────────
    const short iq1     = short(tgpig.x) * Q4P_Q;         // first query index
    const short q_head  = short(tgpig.y);                  // Q-head
    const short kv_head = q_head * short(n_kv_heads) / short(n_heads); // GQA
    const uint  q_dim   = n_heads * Q4P_DK;
    const uint  kv_dim  = n_kv_heads * Q4P_DK;
    const ushort tid    = ushort(sgitg) * Q4P_NW + tiisg;  // 0..127

    // ── TG memory layout ─────────────────────────────────────────────────
    //  sq:    [Q x DK]  half  = 2048 B  (queries)
    //  so:    [Q x PV]  half  = 2048 B  (output accumulator)
    //  ss:    [Q x SH]  float = 1024 B  (QK^T scores)
    //  sk_tg: [C x DK]  half  = 8192 B  (dequantized K tile)
    //  sv_tg: [C x DK]  half  = 8192 B  (dequantized V tile)
    //  Total: ~21 KB
    threadgroup q4p_q_t   sq_buf [Q4P_Q * Q4P_DK];
    threadgroup q4p_o_t   so_buf [Q4P_Q * Q4P_PV];
    threadgroup q4p_s_t   ss_buf [Q4P_Q * Q4P_SH];
    threadgroup q4p_k_t   sk_tg  [Q4P_C * Q4P_DK];   // dequantized K tile
    threadgroup q4p_v_t   sv_tg  [Q4P_C * Q4P_DK];   // dequantized V tile

    threadgroup q4p_q4_t* sq4 = (threadgroup q4p_q4_t*) sq_buf;
    threadgroup q4p_o4_t* so4 = (threadgroup q4p_o4_t*) so_buf;

    // KV cache base pointers for this KV-head
    device uchar* pk_cache = K_cache_q4 + uint(kv_head) * q4_stride;
    device uchar* pv_cache = V_cache_q4 + uint(kv_head) * q4_stride;

    // ── Step 0: Quantize new K/V batch into Q4 cache ─────────────────────
    // Each TG quantizes the K/V of its own tile (iq1..iq1+Q) for this KV head
    // and then reads it back during Phase 1/3. Since every Q-head shares a
    // kv_head (GQA), we intentionally let ALL q_head TGs re-write their
    // kv_head's slice of the Q4 cache. The writes are deterministic in
    // (K_batch, V_batch), so concurrent writes from sibling q_head TGs are
    // idempotent. Restricting the write to `q_head % (n_heads/n_kv_heads) == 0`
    // introduces a cross-TG coherence bug: the non-writer Q-head TGs in the
    // same GQA group may read stale (zero) Q4 cache before the writer TG's
    // commits become visible. `threadgroup_barrier(mem_device)` only fences
    // *within* the current TG, which is exactly what we need once every TG
    // writes its own copy.
    for (short j = 0; j < Q4P_Q; ++j) {
        short batch_idx = iq1 + j;
        if (batch_idx >= short(batch_size)) break;

        uint cache_pos = start_pos + uint(batch_idx);
        device const float* k_src = K_batch + uint(batch_idx) * kv_dim + uint(kv_head) * Q4P_DK;
        device const float* v_src = V_batch + uint(batch_idx) * kv_dim + uint(kv_head) * Q4P_DK;
        device uchar* k_dst = pk_cache + cache_pos * Q4P_BPR * Q4P_BLOCK_BYTES;
        device uchar* v_dst = pv_cache + cache_pos * Q4P_BPR * Q4P_BLOCK_BYTES;

        // Each of 128 threads handles one element (4 blocks x 32 elements = 128)
        ushort block_idx = tid / Q4P_NW;
        ushort elem = tid % Q4P_NW;
        if (block_idx < Q4P_BPR) {
            uint base = block_idx * 32u;
            float kval = (base + elem < uint(Q4P_DK)) ? k_src[base + elem] : 0.0f;
            float vval = (base + elem < uint(Q4P_DK)) ? v_src[base + elem] : 0.0f;

            float kmin = simd_min(kval);
            float kmax = simd_max(kval);
            float krange = kmax - kmin;
            float kscale = krange > 0.0f ? krange / 15.0f : 0.0f;
            float kinv   = kscale > 0.0f ? 1.0f / kscale : 0.0f;

            float vmin = simd_min(vval);
            float vmax = simd_max(vval);
            float vrange = vmax - vmin;
            float vscale = vrange > 0.0f ? vrange / 15.0f : 0.0f;
            float vinv   = vscale > 0.0f ? 1.0f / vscale : 0.0f;

            uchar kq = (uchar)clamp((int)round((kval - kmin) * kinv), 0, 15);
            uchar vq = (uchar)clamp((int)round((vval - vmin) * vinv), 0, 15);

            device uchar* kblk = k_dst + block_idx * Q4P_BLOCK_BYTES;
            device uchar* vblk = v_dst + block_idx * Q4P_BLOCK_BYTES;

            if (elem == 0u) {
                *((device half*)(kblk + 0)) = half(kscale);
                *((device half*)(kblk + 2)) = half(kmin);
                *((device half*)(vblk + 0)) = half(vscale);
                *((device half*)(vblk + 2)) = half(vmin);
            }

            uchar k_partner = simd_shuffle(kq, (elem | 1u));
            uchar v_partner = simd_shuffle(vq, (elem | 1u));
            if ((elem & 1u) == 0u) {
                (kblk + 4u)[elem / 2u] = kq | (k_partner << 4u);
                (vblk + 4u)[elem / 2u] = vq | (v_partner << 4u);
            }
        }
    }

    threadgroup_barrier(mem_flags::mem_device);

    // ── Load Q: f32 -> f16 into threadgroup memory ───────────────────────
    for (short jj = 0; jj < Q4P_NQ; ++jj) {
        const short j = jj * Q4P_NSG + sgitg;
        device const float4* q4 = (device const float4*)
            (Q_batch + uint(iq1 + j) * q_dim + uint(q_head) * Q4P_DK);
        for (short i = tiisg; i < Q4P_DK4; i += Q4P_NW) {
            if (iq1 + j < short(batch_size)) {
                sq4[j * Q4P_DK4 + i] = (q4p_q4_t) q4[i];
            } else {
                sq4[j * Q4P_DK4 + i] = q4p_q4_t(0);
            }
        }
    }

    // ── Zero output accumulator ──────────────────────────────────────────
    for (short jj = 0; jj < Q4P_NQ; ++jj) {
        const short j = jj * Q4P_NSG + sgitg;
        for (short i = tiisg; i < Q4P_PV4; i += Q4P_NW) {
            so4[j * Q4P_PV4 + i] = q4p_o4_t(0);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Online softmax state (per-query, in registers) ───────────────────
    float S[Q4P_NQ] = { 0.0f, 0.0f };
    float M[Q4P_NQ] = { -FLT_MAX/2, -FLT_MAX/2 };

    const float scale = rsqrt(float(Q4P_DK));

    // Max KV position (exclusive): includes both cached + new batch tokens
    const uint max_kv = start_pos + min(uint(iq1 + Q4P_Q), batch_size);

    // ══════════════════════════════════════════════════════════════════════
    // MAIN TILE LOOP over KV dimension
    // ══════════════════════════════════════════════════════════════════════
    for (uint ic = 0; ic < max_kv; ic += Q4P_C) {

        // ── Dequantize K tile: Q4_0 cache → threadgroup f16 ─────────────
        // SOUND FIX (C-f2 prefix-reuse): read ALL kv_pos through Q4,
        // including current-batch positions. Mixing f32 for current-batch
        // with Q4 for historic positions created a precision asymmetry:
        // the first prefill saw all-f32 reads (no round-trip error), but
        // every subsequent prefix-reuse re-prefill saw all-Q4 reads on
        // the cached positions plus f32 on the current batch. That
        // asymmetry compounded across 28 layers and flipped the greedy
        // argmax (12095 "Paris" → 16 "1"). Under this fix both passes
        // read identically at Q4 precision, eliminating the asymmetry.
        //
        // Coherence: Step 0's Q4 cache writes are fenced by the
        // threadgroup_barrier(mem_device) below, and the idempotent
        // write scheme (every GQA-sibling TG rewrites its kv_head slice
        // with deterministic bytes) means cross-TG reads of the same
        // bytes produce the same value regardless of ordering.
        for (short row = 0; row < Q4P_C; ++row) {
            uint kv_pos = ic + uint(row);
            if (kv_pos < max_kv) {
                device const uchar* k_row = pk_cache + kv_pos * Q4P_BPR * Q4P_BLOCK_BYTES;
                dequant_q4_row_to_tg(k_row, sk_tg + row * Q4P_DK, tid);
            } else {
                // Zero padding for out-of-range positions
                sk_tg[row * Q4P_DK + tid] = q4p_k_t(0);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Phase 1: Q x K^T via simdgroup_multiply_accumulate ──────────
        {
            threadgroup const q4p_k_t* pk = sk_tg + uint(sgitg) * 8 * Q4P_DK;
            threadgroup const q4p_q_t* pq = sq_buf;
            threadgroup q4p_s_t* ps = ss_buf + sgitg * 8;

            for (short cc = 0; cc < Q4P_NC; ++cc) {
                q4p_qk8x8_t mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);

                q4p_k8x8_t mk[2];
                q4p_q8x8_t mq[2];

                #pragma unroll 4
                for (short i = 0; i < Q4P_DK8/2; ++i) {
                    simdgroup_barrier(mem_flags::mem_none);

                    simdgroup_load(mq[0], pq + 0*8 + 16*i, Q4P_DK);
                    simdgroup_load(mq[1], pq + 1*8 + 16*i, Q4P_DK);

                    // K is in TG memory (row-major: [C x DK]), load transposed
                    simdgroup_load(mk[0], pk + 0*8 + 16*i, Q4P_DK, ulong2(0,0), true);
                    simdgroup_load(mk[1], pk + 1*8 + 16*i, Q4P_DK, ulong2(0,0), true);

                    simdgroup_barrier(mem_flags::mem_none);

                    simdgroup_multiply_accumulate(mqk, mq[0], mk[0], mqk);
                    simdgroup_multiply_accumulate(mqk, mq[1], mk[1], mqk);
                }

                simdgroup_store(mqk, ps, Q4P_SH, ulong2(0,0), false);

                pk += 8 * Q4P_NSG * Q4P_DK;
                ps += 8 * Q4P_NSG;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Phase 2: Online softmax ─────────────────────────────────────
        for (short jj = 0; jj < Q4P_NQ; ++jj) {
            const short j = jj * Q4P_NSG + sgitg;
            const uint causal_pos = start_pos + uint(iq1 + j) + 1u; // exclusive

            const float m_old = M[jj];

            // Read scores, apply scale + causal mask
            // With C=32, each thread reads 1 score (32 / 32 = 1)
            float s0 = -FLT_MAX/2;
            const uint p0 = ic + uint(tiisg);

            if (p0 < max_kv && p0 < causal_pos)
                s0 = ss_buf[j * Q4P_SH + tiisg] * scale;

            // New running max
            M[jj] = simd_max(max(M[jj], s0));

            // exp(score - M_new)
            const float ms = precise::exp(m_old - M[jj]);
            const float e0 = precise::exp(s0 - M[jj]);

            // Update running sum
            S[jj] = S[jj] * ms + simd_sum(e0);

            // Store normalised exp-scores back for V multiply
            ss_buf[j * Q4P_SH + tiisg] = e0;

            // Rescale old output accumulator
            for (short i = tiisg; i < Q4P_PV4; i += Q4P_NW) {
                so4[j * Q4P_PV4 + i] *= (q4p_o_t) ms;
            }
        }

        // ── Dequantize V tile: Q4_0 cache → threadgroup f16 ─────────────
        // Same policy as the K tile: read ALL kv_pos through Q4 so the
        // first prefill and prefix-reuse re-prefill both see Q4 precision
        // (prevents per-layer compounding of a Q4 round-trip asymmetry
        // that flips the greedy argmax).
        for (short row = 0; row < Q4P_C; ++row) {
            uint kv_pos = ic + uint(row);
            if (kv_pos < max_kv) {
                device const uchar* v_row = pv_cache + kv_pos * Q4P_BPR * Q4P_BLOCK_BYTES;
                dequant_q4_row_to_tg(v_row, sv_tg + row * Q4P_DK, tid);
            } else {
                sv_tg[row * Q4P_DK + tid] = q4p_v_t(0);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Phase 3: O += softmax(QK^T) x V ─────────────────────────────
        {
            constexpr short NO = Q4P_PV8 / Q4P_NSG;  // = 16/4 = 4

            // Load current O from TG memory into registers
            q4p_o8x8_t lo[NO];
            {
                threadgroup q4p_o_t* sot = so_buf + 8 * sgitg;
                for (short ii = 0; ii < NO; ++ii) {
                    simdgroup_load(lo[ii], sot, Q4P_PV, ulong2(0,0), false);
                    sot += 8 * Q4P_NSG;
                }
            }

            // Accumulate: O += S x V
            // C=32 => C/8=4, paired into 2
            {
                threadgroup const q4p_v_t* pv = sv_tg + 8 * sgitg;

                constexpr short NVC = (Q4P_C / 8) / 2;  // = 2 pairs

                for (short cc = 0; cc < NVC; ++cc) {
                    q4p_s8x8_t vs[2];
                    simdgroup_load(vs[0], ss_buf + 16*cc + 0, Q4P_SH, ulong2(0,0), false);
                    simdgroup_load(vs[1], ss_buf + 16*cc + 8, Q4P_SH, ulong2(0,0), false);

                    for (short ii = 0; ii < NO/2; ++ii) {
                        q4p_v8x8_t mv[4];

                        // V from TG: [C x DK], stride = DK between rows
                        simdgroup_load(mv[0], pv + 0*Q4P_NSG + 16*ii*Q4P_NSG + 0*8*Q4P_DK, Q4P_DK, ulong2(0,0), false);
                        simdgroup_load(mv[1], pv + 8*Q4P_NSG + 16*ii*Q4P_NSG + 0*8*Q4P_DK, Q4P_DK, ulong2(0,0), false);
                        simdgroup_load(mv[2], pv + 0*Q4P_NSG + 16*ii*Q4P_NSG + 1*8*Q4P_DK, Q4P_DK, ulong2(0,0), false);
                        simdgroup_load(mv[3], pv + 8*Q4P_NSG + 16*ii*Q4P_NSG + 1*8*Q4P_DK, Q4P_DK, ulong2(0,0), false);

                        simdgroup_multiply_accumulate(lo[2*ii + 0], vs[0], mv[0], lo[2*ii + 0]);
                        simdgroup_multiply_accumulate(lo[2*ii + 1], vs[0], mv[1], lo[2*ii + 1]);
                        simdgroup_multiply_accumulate(lo[2*ii + 0], vs[1], mv[2], lo[2*ii + 0]);
                        simdgroup_multiply_accumulate(lo[2*ii + 1], vs[1], mv[3], lo[2*ii + 1]);
                    }

                    pv += 2 * 8 * Q4P_DK;
                }
            }

            // Store updated O back to TG memory
            {
                threadgroup q4p_o_t* sot = so_buf + 8 * sgitg;
                for (short ii = 0; ii < NO; ++ii) {
                    simdgroup_store(lo[ii], sot, Q4P_PV, ulong2(0,0), false);
                    sot += 8 * Q4P_NSG;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ══════════════════════════════════════════════════════════════════════
    // OUTPUT: half -> f32 with 1/S normalisation
    // ══════════════════════════════════════════════════════════════════════
    for (short jj = 0; jj < Q4P_NQ; ++jj) {
        const short j = jj * Q4P_NSG + sgitg;
        if (iq1 + j >= short(batch_size)) break;

        device float4* dst4 = (device float4*)
            (out + uint(iq1 + j) * q_dim + uint(q_head) * Q4P_DK);

        const float inv_S = S[jj] == 0.0f ? 0.0f : 1.0f / S[jj];

        for (short i = tiisg; i < Q4P_PV4; i += Q4P_NW) {
            dst4[i] = (float4) so4[j * Q4P_PV4 + i] * inv_S;
        }
    }
}
