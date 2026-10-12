#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct block_q6_K {
    uchar ql[128];      // offset 0,   lower 4 bits (QK_K/2)
    uchar qh[64];       // offset 128, upper 2 bits (QK_K/4)
    char  scales[16];   // offset 192, signed 8-bit sub-block scales
    half  d;            // offset 208, super-block scale
};                       // total: 210 bytes

struct block_q4_K {
    half  d;            // offset 0,  super-block scale
    half  dmin;         // offset 2,  super-block min
    uchar scales[12];   // offset 4,  packed 6-bit sub-block scales + mins
    uchar qs[128];      // offset 16, nibble-packed 4-bit values (QK_K/2)
};                       // total: 144 bytes

// llama.cpp-style scale/min extraction for Q4_K/Q5_K
static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)),
                           uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

// ─── Shared simdgroup MAC tile helper ───────────────────────────────────────
//
// Performs the 4-iteration K-loop that loads 4×(8×8) A tiles and 2×(8×8) B
// tiles from threadgroup memory and accumulates into mc[8] (float accumulators).
// Used by every simdgroup-matrix batch GEMM in this file.
//
// sa  — threadgroup half* base pointer for A tile staging buffer
// sb  — threadgroup half* base pointer for B tile staging buffer
// sgitg — simdgroup index within the threadgroup (0..3)
// mc  — caller's 8-element float accumulator array (in/out)
static inline void simdgroup_mac_tile_4x2(
    threadgroup const half* sa,
    threadgroup const half* sb,
    uint sgitg,
    thread simdgroup_float8x8 mc[8])
{
    threadgroup const half* lsma = sa + ((sgitg & 1u) << 8);
    threadgroup const half* lsmb = sb + ((sgitg >> 1) << 7);

    #pragma clang loop unroll(full)
    for (uint ik = 0; ik < 4u; ++ik) {
        simdgroup_half8x8 ma[4];
        simdgroup_half8x8 mb_m[2];
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4u; ++i)
            simdgroup_load(ma[i], lsma + 64u * i, 8);
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 2u; ++i)
            simdgroup_load(mb_m[i], lsmb + 64u * i, 8);
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8u; ++i)
            simdgroup_multiply_accumulate(mc[i], mb_m[i >> 2], ma[i & 3], mc[i]);
        lsma += 512;
        lsmb += 256;
    }
}

// ─── Q5_0 simdgroup_matrix batch GEMM (block-tiled) ────────────────────────
//
// Same tile layout as Q5_1 SGM but for Q5_0 quant (symmetric: q5_val - 16).
// Q5_0 block: 22 bytes — half d (2B), uchar qh[4] (4B), uchar qs_lo[16] (16B).
// Each block has 32 elements, 5-bit symmetric.

kernel void matmul_q4k_batch_sgm_aligned(
    device const uchar* W    [[ buffer(0) ]],
    device const float* X    [[ buffer(1) ]],
    device       float* Y    [[ buffer(2) ]],
    constant     uint&  cols [[ buffer(3) ]],
    constant     uint&  rows [[ buffer(4) ]],
    constant     uint&  B    [[ buffer(5) ]],
    uint2 tgid  [[ threadgroup_position_in_grid ]],
    uint  sgitg [[ simdgroup_index_in_threadgroup ]],
    uint  tiitg [[ thread_index_in_threadgroup ]],
    threadgroup char* shmem [[ threadgroup(0) ]])
{
    constexpr uint NR0 = 64u;
    constexpr uint NR1 = 32u;
    constexpr uint NK  = 32u;

    // Batch-first dispatch: x=batch tiles, y=row tiles
    const uint r0 = tgid.y * NR0;
    const uint c0 = tgid.x * NR1;
    const uint n_k_steps = cols / NK;

    device const block_q4_K* w_typed = (device const block_q4_K*)W;
    const uint n_superblocks = cols / 256u;

    threadgroup half* sa = (threadgroup half*)(shmem);
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);

    // Rows are exact 64-wide tiles. The batch dimension may end on a
    // 16-position half tile (DFlash verifies 16 candidates); inactive input
    // rows are zero and their simdgroups suppress the final store.
    const short lr0 = short(tiitg >> 1);     // 0..63
    const short il0 = short(tiitg & 1u);
    const short lr1 = short(tiitg >> 2);     // 0..31
    const short iy  = short((tiitg & 3u) << 3);

    const short sa_sy = lr0 >> 3;
    const short sa_lx = lr0 & 7;

    const short sb_sx = short(tiitg & 3u);
    const short sb_sy = short(tiitg >> 2) >> 3;
    const short sb_ly = short(tiitg >> 2) & 7;
    const short sb_ib = short(4 * sb_sx + sb_sy);

    const uint r_global = r0 + uint(lr0);
    device const block_q4_K* w_row = w_typed + r_global * n_superblocks;
    const uint batch_row = uint(c0 + lr1);
    device const float* a_ptr = X + min(batch_row, B - 1u) * cols;

    simdgroup_float8x8 mc[8];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 8u; ++i) mc[i] = simdgroup_float8x8(0);

    for (uint ks = 0; ks < n_k_steps; ++ks) {
        const uint k_off     = ks * NK;
        const uint sb_idx    = ks >> 3u;
        const uint sub       = (ks & 7u) * 2u + uint(il0);
        const uint sub_local = sub & 3u;
        const uint q_off     = 32u * (sub / 4u) + 16u * (sub & 1u);
        const short is       = short((sub / 4u) * 2u);
        const uint shift     = sub_local < 2u ? 0u : 4u;

        // ── Phase 1: Load packed quants + scales (deferred dequant, fewer registers) ──
        ushort2 q[4];
        half hdl, hml;
        {
            device const block_q4_K* xb = w_row + sb_idx;
            device const ushort2* q16x2 = (device const ushort2*)(xb->qs + q_off);
            const uchar2 sc = get_scale_min_k4_just2(is, sub_local / 2, xb->scales);
            hdl = half(float(xb->d) * float(sc[0]));
            hml = half(float(xb->dmin) * float(sc[1]));
            #pragma clang loop unroll(full)
            for (short i = 0; i < 4; ++i) q[i] = q16x2[i];
        }

        // ── Phase 2: Pre-load activations → thread registers ──
        float4 av0 = float4(0.0f);
        float4 av1 = float4(0.0f);
        if (batch_row < B) {
            av0 = *(device const float4*)(a_ptr + k_off + iy);
            av1 = *(device const float4*)(a_ptr + k_off + iy + 4);
        }

        // ── Phase 3: Barrier — wait for prev MAC reads from sa/sb ──
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Phase 4: Dequant-on-the-fly + store → sa (half4 fma, matches gate+up pattern) ──
        {
            const half4 hdl4     = half4(hdl);
            const half4 neg_hml4 = half4(-hml);
            #pragma clang loop unroll(full)
            for (short i = 0; i < 4; ++i) {
                const ushort w0 = q[i][0];
                const ushort w1 = q[i][1];
                const half4 dq4 = fma(
                    half4(
                        half((uint(w0) >> shift)        & 0xFu),
                        half((uint(w0) >> (shift + 8u)) & 0xFu),
                        half((uint(w1) >> shift)        & 0xFu),
                        half((uint(w1) >> (shift + 8u)) & 0xFu)
                    ),
                    hdl4, neg_hml4
                );
                const uint sx      = (uint(il0) << 1u) + uint(i >> 1);
                const uint ly_base = uint(i & 1) << 2u;
                const uint ib      = (sx << 3u) + uint(sa_sy);
                const uint base    = 64u * ib + uint(sa_lx);
                *(sa + base + 8u * (ly_base + 0u)) = dq4[0];
                *(sa + base + 8u * (ly_base + 1u)) = dq4[1];
                *(sa + base + 8u * (ly_base + 2u)) = dq4[2];
                *(sa + base + 8u * (ly_base + 3u)) = dq4[3];
            }
        }

        // ── Phase 5: Store activations → sb ──
        *(threadgroup half4*)(sb + 64 * sb_ib + 8 * sb_ly)     = half4(av0);
        *(threadgroup half4*)(sb + 64 * sb_ib + 8 * sb_ly + 4) = half4(av1);

        // ── Phase 6: Barrier — make sa/sb writes visible ──
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Phase 7: MAC with stride-8 loads ──
        simdgroup_mac_tile_4x2(sa, sb, sgitg, mc);
    }

    const uint output_batch = c0 + 16u * (sgitg >> 1);
    if (output_batch < B) {
        device float* C = Y + (r0 + 32u * (sgitg & 1u)) + output_batch * rows;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8u; ++i)
            simdgroup_store(mc[i], C + 8u * (i & 3u) + 8u * rows * (i >> 2), rows);
    }
}

// ─── Ferrite Q4_K SGM specialized for the 16-token verifier ───────────────
//
// The accepted Ferrite tile above is 64 output rows × 32 batch rows and uses
// four simdgroups. Muse verification is exactly B=16, so its upper two
// simdgroups only multiply zero-padded activation rows. This shape-specialized
// form preserves Ferrite's dequantization and simdgroup-matrix arithmetic but
// uses two simdgroups. Each thread stages the same quant half for two output
// rows (r and r+32), keeping the complete 64-row weight tile while removing
// the unused 16 batch rows and their MACs.
//
// Preconditions enforced by the Rust dispatcher:
//   B == 16, rows % 64 == 0, cols % 32 == 0.
// dispatch: (1, rows/64, 1) × (64, 1, 1)
kernel void matmul_q4k_batch_sgm_b16_aligned(
    device const uchar* W    [[ buffer(0) ]],
    device const float* X    [[ buffer(1) ]],
    device       float* Y    [[ buffer(2) ]],
    constant     uint&  cols [[ buffer(3) ]],
    constant     uint&  rows [[ buffer(4) ]],
    constant     uint&  B    [[ buffer(5) ]],
    uint2 tgid  [[ threadgroup_position_in_grid ]],
    uint  sgitg [[ simdgroup_index_in_threadgroup ]],
    uint  tiitg [[ thread_index_in_threadgroup ]],
    threadgroup char* shmem [[ threadgroup(0) ]])
{
    constexpr uint NR0 = 64u;
    constexpr uint NK  = 32u;

    const uint r0 = tgid.y * NR0;
    const uint n_k_steps = cols / NK;

    device const block_q4_K* w_typed = (device const block_q4_K*)W;
    const uint n_superblocks = cols / 256u;

    threadgroup half* sa = (threadgroup half*)(shmem);
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);

    // Two threads stage the two 16-value quant halves for one row. The same
    // pair also stages row+32, covering the full Ferrite 64-row output tile.
    const short lr0 = short(tiitg >> 1);     // 0..31
    const short il0 = short(tiitg & 1u);

    // Four threads stage one 32-value activation row. With 64 threads this is
    // exactly the verifier's 16 rows; there is no padded upper half.
    const short lr1 = short(tiitg >> 2);     // 0..15
    const short iy  = short((tiitg & 3u) << 3);
    const short sb_sx = short(tiitg & 3u);
    const short sb_sy = short(tiitg >> 2) >> 3;
    const short sb_ly = short(tiitg >> 2) & 7;
    const short sb_ib = short(4 * sb_sx + sb_sy);

    device const block_q4_K* w_row[2] = {
        w_typed + (r0 + uint(lr0)) * n_superblocks,
        w_typed + (r0 + uint(lr0) + 32u) * n_superblocks,
    };
    device const float* a_ptr = X + uint(lr1) * cols;

    simdgroup_float8x8 mc[8];
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 8u; ++i) mc[i] = simdgroup_float8x8(0);

    for (uint ks = 0; ks < n_k_steps; ++ks) {
        const uint k_off     = ks * NK;
        const uint sb_idx    = ks >> 3u;
        const uint sub       = (ks & 7u) * 2u + uint(il0);
        const uint sub_local = sub & 3u;
        const uint q_off     = 32u * (sub / 4u) + 16u * (sub & 1u);
        const short is       = short((sub / 4u) * 2u);
        const uint shift     = sub_local < 2u ? 0u : 4u;

        ushort2 q[2][4];
        half hdl[2];
        half hml[2];
        #pragma clang loop unroll(full)
        for (short plane = 0; plane < 2; ++plane) {
            device const block_q4_K* xb = w_row[plane] + sb_idx;
            device const ushort2* q16x2 = (device const ushort2*)(xb->qs + q_off);
            const uchar2 sc = get_scale_min_k4_just2(is, sub_local / 2, xb->scales);
            hdl[plane] = half(float(xb->d) * float(sc[0]));
            hml[plane] = half(float(xb->dmin) * float(sc[1]));
            #pragma clang loop unroll(full)
            for (short i = 0; i < 4; ++i) q[plane][i] = q16x2[i];
        }

        const float4 av0 = *(device const float4*)(a_ptr + k_off + iy);
        const float4 av1 = *(device const float4*)(a_ptr + k_off + iy + 4);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (short plane = 0; plane < 2; ++plane) {
            const short logical_row = lr0 + 32 * plane;
            const short sa_sy = logical_row >> 3;
            const short sa_lx = logical_row & 7;
            const half4 hdl4 = half4(hdl[plane]);
            const half4 neg_hml4 = half4(-hml[plane]);
            #pragma clang loop unroll(full)
            for (short i = 0; i < 4; ++i) {
                const ushort w0 = q[plane][i][0];
                const ushort w1 = q[plane][i][1];
                const half4 dq4 = fma(
                    half4(
                        half((uint(w0) >> shift)        & 0xFu),
                        half((uint(w0) >> (shift + 8u)) & 0xFu),
                        half((uint(w1) >> shift)        & 0xFu),
                        half((uint(w1) >> (shift + 8u)) & 0xFu)
                    ),
                    hdl4, neg_hml4
                );
                const uint sx      = (uint(il0) << 1u) + uint(i >> 1);
                const uint ly_base = uint(i & 1) << 2u;
                const uint ib      = (sx << 3u) + uint(sa_sy);
                const uint base    = 64u * ib + uint(sa_lx);
                *(sa + base + 8u * (ly_base + 0u)) = dq4[0];
                *(sa + base + 8u * (ly_base + 1u)) = dq4[1];
                *(sa + base + 8u * (ly_base + 2u)) = dq4[2];
                *(sa + base + 8u * (ly_base + 3u)) = dq4[3];
            }
        }

        *(threadgroup half4*)(sb + 64 * sb_ib + 8 * sb_ly)     = half4(av0);
        *(threadgroup half4*)(sb + 64 * sb_ib + 8 * sb_ly + 4) = half4(av1);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_mac_tile_4x2(sa, sb, sgitg, mc);
    }

    device float* C = Y + (r0 + 32u * sgitg);
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 8u; ++i)
        simdgroup_store(mc[i], C + 8u * (i & 3u) + 8u * rows * (i >> 2), rows);
}
