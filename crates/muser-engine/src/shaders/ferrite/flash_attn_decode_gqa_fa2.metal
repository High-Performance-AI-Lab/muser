// One-query Muse specialization of flash_attn_v2. The general kernel spends
// all eight simdgroup-matrix rows on one query head and seven masked query
// rows. Muse has 16 Q heads per KV head, so this kernel maps eight Q heads
// sharing one KV head onto those same matrix rows. The QK, online-softmax,
// and S*V instruction order for every live row is unchanged.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr short DG_DK = 128;
constant constexpr short DG_DK4 = DG_DK / 4;
constant constexpr short DG_DK8 = DG_DK / 8;
constant constexpr short DG_Q = 8;
constant constexpr short DG_C = 64;
constant constexpr short DG_NSG = 4;
constant constexpr short DG_NW = 32;
constant constexpr short DG_NQ = DG_Q / DG_NSG;
constant constexpr short DG_NC = (DG_C / 8) / DG_NSG;
constant constexpr short DG_PV = DG_DK;
constant constexpr short DG_PV4 = DG_PV / 4;
constant constexpr short DG_PV8 = DG_PV / 8;
constant constexpr short DG_SH = DG_C;

using dg_q_t = half;
using dg_q4_t = half4;
using dg_k_t = half;
using dg_v_t = half;
using dg_o_t = float;
using dg_o4_t = float4;
using dg_s_t = float;
using dg_q8x8_t = simdgroup_half8x8;
using dg_k8x8_t = simdgroup_half8x8;
using dg_v8x8_t = simdgroup_half8x8;
using dg_o8x8_t = simdgroup_float8x8;
using dg_s8x8_t = simdgroup_float8x8;
using dg_qk8x8_t = simdgroup_float8x8;

kernel void muser_flash_attn_decode_gqa_fa2(
    device const float* query [[buffer(0)]],
    device const half* key_cache [[buffer(1)]],
    device const half* value_cache [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& start_pos [[buffer(4)]],
    constant uint& n_heads [[buffer(5)]],
    constant uint& n_kv_heads [[buffer(6)]],
    constant uint& cache_stride [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    constant uint& head_major [[buffer(9)]],
    constant uint& cache_logical_base [[buffer(10)]],
    constant uint& window_size [[buffer(11)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]])
{
    const short q_head0 = short(tgpig.x) * DG_Q;
    const short q_per_kv = short(n_heads / n_kv_heads);
    const short kv_head = q_head0 / q_per_kv;

    threadgroup dg_q_t sq_buf[DG_Q * DG_DK];
    threadgroup dg_o_t so_buf[DG_Q * DG_PV];
    threadgroup dg_s_t ss_buf[DG_Q * DG_SH];
    threadgroup dg_q4_t* sq4 = (threadgroup dg_q4_t*)sq_buf;
    threadgroup dg_o4_t* so4 = (threadgroup dg_o4_t*)so_buf;

    const uint kv_stride = head_major != 0u ? DG_DK : n_kv_heads * DG_DK;
    device const dg_k_t* pk_base = key_cache + (head_major != 0u
        ? uint(kv_head) * cache_stride
        : uint(kv_head) * DG_DK);
    device const dg_v_t* pv_base = value_cache + (head_major != 0u
        ? uint(kv_head) * cache_stride
        : uint(kv_head) * DG_DK);

    for (short jj = 0; jj < DG_NQ; ++jj) {
        const short j = jj * DG_NSG + sgitg;
        for (short i = tiisg; i < DG_DK4; i += DG_NW) {
            device const float4* q4 = (device const float4*)
                (query + uint(q_head0 + j) * DG_DK);
            sq4[j * DG_DK4 + i] = (dg_q4_t)q4[i];
        }
        for (short i = tiisg; i < DG_PV4; i += DG_NW) {
            so4[j * DG_PV4 + i] = dg_o4_t(0);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S[DG_NQ] = {0.0f, 0.0f};
    float M[DG_NQ] = {-FLT_MAX/2, -FLT_MAX/2};
    const uint causal_pos = start_pos + 1u;
    const uint query_swa_start =
        (window_size > 0u && causal_pos > window_size)
        ? causal_pos - window_size : 0u;
    const uint win_start = max(cache_logical_base,
        (query_swa_start / DG_C) * DG_C);

    for (uint ic = win_start; ic < causal_pos; ic += DG_C) {
        {
            device const dg_k_t* pk = pk_base
                + (ic - cache_logical_base + uint(sgitg) * 8) * kv_stride;
            threadgroup const dg_q_t* pq = sq_buf;
            threadgroup dg_s_t* ps = ss_buf + sgitg * 8;
            for (short cc = 0; cc < DG_NC; ++cc) {
                dg_qk8x8_t mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);
                dg_k8x8_t mk[2];
                dg_q8x8_t mq[2];
                #pragma unroll 4
                for (short i = 0; i < DG_DK8/2; ++i) {
                    simdgroup_barrier(mem_flags::mem_none);
                    simdgroup_load(mq[0], pq + 0*8 + 16*i, DG_DK);
                    simdgroup_load(mq[1], pq + 1*8 + 16*i, DG_DK);
                    simdgroup_load(mk[0], pk + 0*8 + 16*i, kv_stride,
                                   ulong2(0,0), true);
                    simdgroup_load(mk[1], pk + 1*8 + 16*i, kv_stride,
                                   ulong2(0,0), true);
                    simdgroup_barrier(mem_flags::mem_none);
                    simdgroup_multiply_accumulate(mqk, mq[0], mk[0], mqk);
                    simdgroup_multiply_accumulate(mqk, mq[1], mk[1], mqk);
                }
                simdgroup_store(mqk, ps, DG_SH, ulong2(0,0), false);
                pk += 8 * DG_NSG * kv_stride;
                ps += 8 * DG_NSG;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short jj = 0; jj < DG_NQ; ++jj) {
            const short j = jj * DG_NSG + sgitg;
            const float m_old = M[jj];
            float s0 = -FLT_MAX/2;
            float s1 = -FLT_MAX/2;
            const uint p0 = ic + 2u * tiisg;
            const uint p1 = p0 + 1u;
            if (p0 < causal_pos && p0 >= query_swa_start)
                s0 = ss_buf[j * DG_SH + 2 * tiisg] * scale;
            if (p1 < causal_pos && p1 >= query_swa_start)
                s1 = ss_buf[j * DG_SH + 2 * tiisg + 1] * scale;
            M[jj] = simd_max(max(M[jj], max(s0, s1)));
            const float ms = precise::exp(m_old - M[jj]);
            const float e0 = precise::exp(s0 - M[jj]);
            const float e1 = precise::exp(s1 - M[jj]);
            S[jj] = S[jj] * ms + simd_sum(e0 + e1);
            ss_buf[j * DG_SH + 2 * tiisg] = e0;
            ss_buf[j * DG_SH + 2 * tiisg + 1] = e1;
            for (short i = tiisg; i < DG_PV4; i += DG_NW) {
                so4[j * DG_PV4 + i] *= ms;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            constexpr short DG_NO = DG_PV8 / DG_NSG;
            dg_o8x8_t lo[DG_NO];
            threadgroup dg_o_t* sot = so_buf + 8 * sgitg;
            for (short ii = 0; ii < DG_NO; ++ii) {
                simdgroup_load(lo[ii], sot, DG_PV, ulong2(0,0), false);
                sot += 8 * DG_NSG;
            }

            device const dg_v_t* pv = pv_base
                + (ic - cache_logical_base) * kv_stride + 8 * sgitg;
            constexpr short DG_NVC = (DG_C / 8) / 2;
            for (short cc = 0; cc < DG_NVC; ++cc) {
                dg_s8x8_t vs[2];
                simdgroup_load(vs[0], ss_buf + 16*cc + 0,
                               DG_SH, ulong2(0,0), false);
                simdgroup_load(vs[1], ss_buf + 16*cc + 8,
                               DG_SH, ulong2(0,0), false);
                for (short ii = 0; ii < DG_NO/2; ++ii) {
                    dg_v8x8_t mv[4];
                    simdgroup_load(mv[0], pv + 0*DG_NSG + 16*ii*DG_NSG
                                   + 0*8*kv_stride, kv_stride, ulong2(0,0), false);
                    simdgroup_load(mv[1], pv + 8*DG_NSG + 16*ii*DG_NSG
                                   + 0*8*kv_stride, kv_stride, ulong2(0,0), false);
                    simdgroup_load(mv[2], pv + 0*DG_NSG + 16*ii*DG_NSG
                                   + 1*8*kv_stride, kv_stride, ulong2(0,0), false);
                    simdgroup_load(mv[3], pv + 8*DG_NSG + 16*ii*DG_NSG
                                   + 1*8*kv_stride, kv_stride, ulong2(0,0), false);
                    simdgroup_multiply_accumulate(lo[2*ii + 0], vs[0], mv[0], lo[2*ii + 0]);
                    simdgroup_multiply_accumulate(lo[2*ii + 1], vs[0], mv[1], lo[2*ii + 1]);
                    simdgroup_multiply_accumulate(lo[2*ii + 0], vs[1], mv[2], lo[2*ii + 0]);
                    simdgroup_multiply_accumulate(lo[2*ii + 1], vs[1], mv[3], lo[2*ii + 1]);
                }
                pv += 2 * 8 * kv_stride;
            }

            sot = so_buf + 8 * sgitg;
            for (short ii = 0; ii < DG_NO; ++ii) {
                simdgroup_store(lo[ii], sot, DG_PV, ulong2(0,0), false);
                sot += 8 * DG_NSG;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (short jj = 0; jj < DG_NQ; ++jj) {
        const short j = jj * DG_NSG + sgitg;
        device float4* dst4 = (device float4*)
            (out + uint(q_head0 + j) * DG_DK);
        const float inv_s = S[jj] == 0.0f ? 0.0f : 1.0f / S[jj];
        for (short i = tiisg; i < DG_PV4; i += DG_NW) {
            dst4[i] = so4[j * DG_PV4 + i] * inv_s;
        }
    }
}
