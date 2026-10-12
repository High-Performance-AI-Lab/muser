// ── _q4k_helpers.metal ────────────────────────────────────────────────────
//
// Cross-file static inline helpers for the Q4_K V4-style matvec kernels.
//
// These functions encapsulate the Q4K superblock dequant + MAC core that
// appears across matmul_q4k_v4, qkv_v4, qkv_fused_v4, ffn_sparse,
// matvec_qkv_rope_fused_v4, and matmul_q4k_v4_residuals.
//
// MUST appear early in the shader concat order (before all Q4K caller files).
// Defined in context.rs before matmul_q4k_v4, matmul_q4k_v4_residuals, etc.
//
// No #include directives. No #define macros. static inline only.

#include <metal_stdlib>
using namespace metal;

// ── q4k_v4_dual_row_mac ────────────────────────────────────────────────────
//
// Decodes two consecutive Q4K rows from `blk_start` and accumulates
// their dot-products (weighted by scales) into sumf[0] and sumf[1].
// `blk_start` points to the first row's block; the second row is at
// blk_start + row_bytes.
//
// Parameters:
//   blk_start  — pointer to the first row's block for this superblock index
//   row_bytes  — stride in bytes per weight row (n_blocks * 144)
//   yl         — thread-local x-vector low-half slice [16 floats]
//   yh         — thread-local x-vector high-half slice [16 floats]
//   sumy       — per-quarter sums of yl/yh (for dmin subtraction)
//   iq         — half-block selector (0 or 1)
//   ir         — quarter within half (0..3)
//   sumf       — accumulator [2]: sumf[0] for row 0, sumf[1] for row 1
//
static inline void q4k_v4_dual_row_mac(
    device const uchar* blk_start,
    uint                row_bytes,
    thread float        yl[16],
    thread float        yh[16],
    float4              sumy,
    uint                iq,
    uint                ir,
    thread float        sumf[2])
{
    constexpr ushort kmask1 = 0x3F3Fu;
    constexpr ushort kmask2 = 0x0F0Fu;
    constexpr ushort kmask3 = 0xC0C0u;

    device const uchar* blk = blk_start;
    _Pragma("clang loop unroll(full)")
    for (short row = 0; row < 2; row++) {
        device const half* dh = reinterpret_cast<device const half*>(blk);

        device const ushort* sc = reinterpret_cast<device const ushort*>(blk + 4u) + iq;
        ushort sc16[4];
        sc16[0] = sc[0] & kmask1;
        sc16[1] = sc[2] & kmask1;
        sc16[2] = ((sc[4])      & kmask2) | ((sc[0] & kmask3) >> 2);
        sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
        thread const uchar* sc8 = reinterpret_cast<thread const uchar*>(sc16);

        device const ushort* q1 = reinterpret_cast<device const ushort*>(blk + 16u) + 16u * iq + 4u * ir;
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

        sumf[row] +=
            float(dh[0]) * ((acc1[0] + (1.f/256.f) * acc1[1]) * float(sc8[0]) +
                            (acc1[2] + (1.f/256.f) * acc1[3]) * float(sc8[1]) * (1.f/16.f) +
                            (acc2[0] + (1.f/256.f) * acc2[1]) * float(sc8[4]) +
                            (acc2[2] + (1.f/256.f) * acc2[3]) * float(sc8[5]) * (1.f/16.f)) -
            float(dh[1]) * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3]) +
                            sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));

        blk += row_bytes;
    }
}

// ── q4k_v4_single_row_mac ──────────────────────────────────────────────────
//
// Decodes a single Q4K row from `blk` and accumulates its dot-product
// (weighted by scales) into `sumf`.
//
// Used by kernels that independently compute `blk` per row (e.g. paired
// Q/K scheduling in matvec_qkv_rope_fused_v4 where the two rows are not
// consecutive in memory).
//
// Parameters:
//   blk   — pointer to this row's block for this superblock index
//   yl    — thread-local x-vector low-half slice [16 floats]
//   yh    — thread-local x-vector high-half slice [16 floats]
//   sumy  — per-quarter sums of yl/yh (for dmin subtraction)
//   iq    — half-block selector (0 or 1)
//   ir    — quarter within half (0..3)
//   sumf  — accumulator (single float, incremented in place)
//
static inline void q4k_v4_single_row_mac(
    device const uchar* blk,
    thread float        yl[16],
    thread float        yh[16],
    float4              sumy,
    uint                iq,
    uint                ir,
    thread float&       sumf)
{
    constexpr ushort kmask1 = 0x3F3Fu;
    constexpr ushort kmask2 = 0x0F0Fu;
    constexpr ushort kmask3 = 0xC0C0u;

    device const half* dh = reinterpret_cast<device const half*>(blk);

    device const ushort* sc = reinterpret_cast<device const ushort*>(blk + 4u) + iq;
    ushort sc16[4];
    sc16[0] = sc[0] & kmask1;
    sc16[1] = sc[2] & kmask1;
    sc16[2] = ((sc[4])      & kmask2) | ((sc[0] & kmask3) >> 2);
    sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
    thread const uchar* sc8 = reinterpret_cast<thread const uchar*>(sc16);

    device const ushort* q1 = reinterpret_cast<device const ushort*>(blk + 16u) + 16u * iq + 4u * ir;
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

    sumf +=
        float(dh[0]) * ((acc1[0] + (1.f/256.f) * acc1[1]) * float(sc8[0]) +
                        (acc1[2] + (1.f/256.f) * acc1[3]) * float(sc8[1]) * (1.f/16.f) +
                        (acc2[0] + (1.f/256.f) * acc2[1]) * float(sc8[4]) +
                        (acc2[2] + (1.f/256.f) * acc2[3]) * float(sc8[5]) * (1.f/16.f)) -
        float(dh[1]) * (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3]) +
                        sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]));
}
