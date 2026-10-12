// Stage B / L-series candidate M=16 K-quant batch GEMM kernels.
//
// Design: weight-stationary GEMV-style kernel. One simdgroup owns R output
// rows; the 32 lanes split K within each 256-value superblock. The 16
// activation rows are read directly from device memory (they are tiny and
// L2-resident) and shared across the R rows, so each activation vector load
// feeds R x 16 accumulators. No threadgroup memory, no barriers.
// *_h variants read activations pre-converted to half (halved L2 traffic).
//
// Exactness contract: lossless token equality at the model level. At the
// bench level each kernel is validated against the f32 CPU dequant reference
// and must sit inside the accepted half-math error envelope.

#include <metal_stdlib>
using namespace metal;

struct m16_block_q4_K {
    half  d;
    half  dmin;
    uchar scales[12];
    uchar qs[128];
};

struct m16_block_q5_K {
    half  d;
    half  dmin;
    uchar scales[12];
    uchar qh[32];
    uchar qs[128];
};

struct m16_block_q6_K {
    uchar ql[128];
    uchar qh[64];
    char  scales[16];
    half  d;
};

static inline void m16_q4k_scale_min(device const uchar* scales, uint j,
                                     thread float& sc, thread float& mn) {
    if (j < 4u) {
        sc = float(scales[j] & 63u);
        mn = float(scales[j + 4u] & 63u);
    } else {
        sc = float((scales[j + 4u] & 0x0Fu) | ((scales[j - 4u] & 0xC0u) >> 2u));
        mn = float((scales[j + 4u] >> 4u) | ((scales[j] & 0xC0u) >> 2u));
    }
}

// Accumulators: R output rows x 16 activation rows.
template <uint R>
struct M16Acc {
    float a[R][16];
};

template <uint R>
static inline void m16_acc_zero(thread M16Acc<R>& acc) {
#pragma clang loop unroll(full)
    for (uint r = 0; r < R; ++r)
#pragma clang loop unroll(full)
        for (uint m = 0; m < 16u; ++m) acc.a[r][m] = 0.0f;
}

template <uint R>
static inline void m16_acc_reduce_store(thread M16Acc<R>& acc, device float* Y,
                                        uint r0, uint n_out, uint lane) {
#pragma clang loop unroll(full)
    for (uint m = 0; m < 16u; ++m) {
#pragma clang loop unroll(full)
        for (uint r = 0; r < R; ++r) {
            const float s = simd_sum(acc.a[r][m]);
            if (lane == 0u) Y[m * n_out + r0 + r] = s;
        }
    }
}

// Dot the lane's 8 dequanted values against all 16 f32 activation rows.
static inline void m16_mac8(thread const float wv[8], uint k_base,
                            device const float4* X, uint x_stride4,
                            thread float acc[16]) {
#pragma clang loop unroll(full)
    for (uint m = 0; m < 16u; ++m) {
        device const float4* xr = X + m * x_stride4 + (k_base >> 2u);
        const float4 xa = xr[0];
        const float4 xb = xr[1];
        acc[m] += wv[0] * xa.x + wv[1] * xa.y + wv[2] * xa.z + wv[3] * xa.w
                + wv[4] * xb.x + wv[5] * xb.y + wv[6] * xb.z + wv[7] * xb.w;
    }
}

// Same against half activations (two half4 loads per row).
static inline void m16_mac8_h(thread const float wv[8], uint k_base,
                              device const half4* X, uint x_stride4,
                              thread float acc[16]) {
#pragma clang loop unroll(full)
    for (uint m = 0; m < 16u; ++m) {
        device const half4* xr = X + m * x_stride4 + (k_base >> 2u);
        const float4 xa = float4(xr[0]);
        const float4 xb = float4(xr[1]);
        acc[m] += wv[0] * xa.x + wv[1] * xa.y + wv[2] * xa.z + wv[3] * xa.w
                + wv[4] * xb.x + wv[5] * xb.y + wv[6] * xb.z + wv[7] * xb.w;
    }
}

// Dequant the lane's 8 Q4_K values of one superblock row.
static inline void m16_q4k_lane8(device const m16_block_q4_K* b, uint s,
                                 uint qs_off, uint shift, thread float wv[8]) {
    float sc, mn;
    m16_q4k_scale_min(b->scales, s, sc, mn);
    const float dl = float(b->d) * sc;
    const float ml = float(b->dmin) * mn;
    const uchar4 qa = *(device const uchar4*)(b->qs + qs_off);
    const uchar4 qb = *(device const uchar4*)(b->qs + qs_off + 4u);
#pragma clang loop unroll(full)
    for (uint i = 0; i < 4u; ++i) {
        wv[i] = dl * float((qa[i] >> shift) & 15u) - ml;
        wv[i + 4u] = dl * float((qb[i] >> shift) & 15u) - ml;
    }
}

template <uint R, bool HALF_X>
static inline void m16_q4k_body(device const uchar* W, device const uchar* X,
                                device float* Y, uint n_in, uint n_out,
                                uint tgid, uint sgitg, uint lane) {
    const uint r0 = (tgid * 8u + sgitg) * R;
    if (r0 >= n_out) return;
    const uint nsb = n_in >> 8u;
    const uint x_stride4 = n_in >> 2u;

    const uint s = lane >> 2u;
    const uint g = s >> 1u;
    const uint shift = (s & 1u) << 2u;
    const uint qs_off = 32u * g + (lane & 3u) * 8u;
    const uint k_lo = 64u * g + 32u * (s & 1u) + (lane & 3u) * 8u;

    device const m16_block_q4_K* wb =
        (device const m16_block_q4_K*)(W) + r0 * nsb;

    M16Acc<R> acc;
    m16_acc_zero(acc);

    for (uint sb = 0; sb < nsb; ++sb) {
        float wv[R][8];
#pragma clang loop unroll(full)
        for (uint r = 0; r < R; ++r)
            m16_q4k_lane8(wb + r * nsb + sb, s, qs_off, shift, wv[r]);
        const uint k_base = sb * 256u + k_lo;
#pragma clang loop unroll(full)
        for (uint r = 0; r < R; ++r) {
            if (HALF_X)
                m16_mac8_h(wv[r], k_base, (device const half4*)X, x_stride4,
                           acc.a[r]);
            else
                m16_mac8(wv[r], k_base, (device const float4*)X, x_stride4,
                         acc.a[r]);
        }
    }
    m16_acc_reduce_store(acc, Y, r0, n_out, lane);
}

#define M16_Q4K_KERNEL(name, R, HALF_X)                                       \
kernel void name(device const uchar* W [[buffer(0)]],                         \
                 device const uchar* X [[buffer(1)]],                         \
                 device float* Y [[buffer(2)]],                               \
                 constant uint& n_in [[buffer(3)]],                           \
                 constant uint& n_out [[buffer(4)]],                          \
                 uint tgid [[threadgroup_position_in_grid]],                  \
                 uint sgitg [[simdgroup_index_in_threadgroup]],               \
                 uint lane [[thread_index_in_simdgroup]]) {                   \
    m16_q4k_body<R, HALF_X>(W, X, Y, n_in, n_out, tgid, sgitg, lane);         \
}

M16_Q4K_KERNEL(m16_q4k_r2, 2, false)
M16_Q4K_KERNEL(m16_q4k_r4, 4, false)
M16_Q4K_KERNEL(m16_q4k_r2h, 2, true)
M16_Q4K_KERNEL(m16_q4k_r4h, 4, true)

kernel void m16_q6k_r2(device const uchar* W [[buffer(0)]],
                       device const float4* X [[buffer(1)]],
                       device float* Y [[buffer(2)]],
                       constant uint& n_in [[buffer(3)]],
                       constant uint& n_out [[buffer(4)]],
                       uint tgid [[threadgroup_position_in_grid]],
                       uint sgitg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    const uint r0 = (tgid * 8u + sgitg) * 2u;
    if (r0 >= n_out) return;
    const uint nsb = n_in >> 8u;
    const uint x_stride4 = n_in >> 2u;

    // Lane -> (group g, stream st, 8 consecutive positions p8) per superblock.
    const uint g = lane >> 4u;          // 0..1
    const uint st = (lane >> 2u) & 3u;  // 0..3 stream
    const uint p8 = (lane & 3u) * 8u;   // first position
    const uint ql_base = 64u * g + (st & 1u) * 32u + p8;
    const uint qh_base = 128u + 32u * g + p8;
    const uint sc_idx = 8u * g + (p8 >> 4u) + 2u * st;
    const uint ql_shift = (st & 2u) << 1u;   // 0 or 4
    const uint qh_shift = 2u * st;           // 0,2,4,6
    const uint k_lo = 128u * g + 32u * st + p8;

    device const m16_block_q6_K* wb0 =
        (device const m16_block_q6_K*)(W) + r0 * nsb;
    device const m16_block_q6_K* wb1 = wb0 + nsb;

    M16Acc<2> acc;
    m16_acc_zero(acc);

    for (uint sb = 0; sb < nsb; ++sb) {
        device const uchar* b0 = (device const uchar*)(wb0 + sb);
        device const uchar* b1 = (device const uchar*)(wb1 + sb);
        const float d0 = float(((device const half*)(b0 + 208))[0]);
        const float d1 = float(((device const half*)(b1 + 208))[0]);
        const float s0 = d0 * float(((device const char*)(b0 + 192))[sc_idx]);
        const float s1 = d1 * float(((device const char*)(b1 + 192))[sc_idx]);
        float wv0[8], wv1[8];
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8u; ++i) {
            const uint ql0 = uint(b0[ql_base + i]);
            const uint qh0 = uint(b0[qh_base + i]);
            const uint ql1 = uint(b1[ql_base + i]);
            const uint qh1 = uint(b1[qh_base + i]);
            wv0[i] = s0 * float(int(((ql0 >> ql_shift) & 15u) |
                                    (((qh0 >> qh_shift) & 3u) << 4u)) - 32);
            wv1[i] = s1 * float(int(((ql1 >> ql_shift) & 15u) |
                                    (((qh1 >> qh_shift) & 3u) << 4u)) - 32);
        }
        const uint k_base = sb * 256u + k_lo;
        m16_mac8(wv0, k_base, X, x_stride4, acc.a[0]);
        m16_mac8(wv1, k_base, X, x_stride4, acc.a[1]);
    }
    m16_acc_reduce_store(acc, Y, r0, n_out, lane);
}

kernel void m16_q5k_r2(device const uchar* W [[buffer(0)]],
                       device const float4* X [[buffer(1)]],
                       device float* Y [[buffer(2)]],
                       constant uint& n_in [[buffer(3)]],
                       constant uint& n_out [[buffer(4)]],
                       uint tgid [[threadgroup_position_in_grid]],
                       uint sgitg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    const uint r0 = (tgid * 8u + sgitg) * 2u;
    if (r0 >= n_out) return;
    const uint nsb = n_in >> 8u;
    const uint x_stride4 = n_in >> 2u;

    const uint s = lane >> 2u;
    const uint g = s >> 1u;
    const uint h = s & 1u;
    const uint shift = h << 2u;
    const uint bit = 2u * g + h;
    const uint qs_off = 32u * g + (lane & 3u) * 8u;
    const uint qh_off = (lane & 3u) * 8u;
    const uint k_lo = 64u * g + 32u * h + (lane & 3u) * 8u;

    device const m16_block_q5_K* wb0 =
        (device const m16_block_q5_K*)(W) + r0 * nsb;
    device const m16_block_q5_K* wb1 = wb0 + nsb;

    M16Acc<2> acc;
    m16_acc_zero(acc);

    for (uint sb = 0; sb < nsb; ++sb) {
        device const m16_block_q5_K* b0 = wb0 + sb;
        device const m16_block_q5_K* b1 = wb1 + sb;
        float sc0, mn0, sc1, mn1;
        m16_q4k_scale_min(b0->scales, s, sc0, mn0);
        m16_q4k_scale_min(b1->scales, s, sc1, mn1);
        const float dl0 = float(b0->d) * sc0;
        const float ml0 = float(b0->dmin) * mn0;
        const float dl1 = float(b1->d) * sc1;
        const float ml1 = float(b1->dmin) * mn1;
        const uchar4 q0a = *(device const uchar4*)(b0->qs + qs_off);
        const uchar4 q0b = *(device const uchar4*)(b0->qs + qs_off + 4u);
        const uchar4 q1a = *(device const uchar4*)(b1->qs + qs_off);
        const uchar4 q1b = *(device const uchar4*)(b1->qs + qs_off + 4u);
        float wv0[8], wv1[8];
#pragma clang loop unroll(full)
        for (uint i = 0; i < 4u; ++i) {
            wv0[i] = dl0 * float(((q0a[i] >> shift) & 15u) |
                                 (((b0->qh[qh_off + i] >> bit) & 1u) << 4u)) - ml0;
            wv0[i + 4u] = dl0 * float(((q0b[i] >> shift) & 15u) |
                                      (((b0->qh[qh_off + 4u + i] >> bit) & 1u) << 4u)) - ml0;
            wv1[i] = dl1 * float(((q1a[i] >> shift) & 15u) |
                                 (((b1->qh[qh_off + i] >> bit) & 1u) << 4u)) - ml1;
            wv1[i + 4u] = dl1 * float(((q1b[i] >> shift) & 15u) |
                                      (((b1->qh[qh_off + 4u + i] >> bit) & 1u) << 4u)) - ml1;
        }
        const uint k_base = sb * 256u + k_lo;
        m16_mac8(wv0, k_base, X, x_stride4, acc.a[0]);
        m16_mac8(wv1, k_base, X, x_stride4, acc.a[1]);
    }
    m16_acc_reduce_store(acc, Y, r0, n_out, lane);
}



// ─── t128: M=16 SGM GEMM with a 128-wide K-tile ────────────────────────────
//
// simdgroup-matrix GEMM in the retained tile's arithmetic family, but with
// 128-K staging chunks (4x fewer barriers than NK=32), 64 output rows per
// threadgroup, and a fully-packed 16-row B tile (no wasted MAC lanes).
// TG = 128 threads = 4 simdgroups; sg (rh, bh) covers output rows
// rh*32..+31 x batch columns bh*8..+7. Grid: (n_out/64, 1).
// Preconditions: n_in % 256 == 0, n_out % 64 == 0, B == 16.
//
// Conventions match the retained tile: mc[m][r], mb[m][k], ma[k][r], and
// multiply_accumulate(mc, mb, ma, mc). Tile memory:
//   sa (weights):     tile (rt, kt) at (16*kt + rt)*64, element (k,r) at 8*k+r
//   sb (activations): tile (bt, kt) at (2*kt + bt)*64, element (m,k) at 8*m+k

kernel void m16_q4k_t128(device const uchar* W [[buffer(0)]],
                         device const float4* X [[buffer(1)]],
                         device float* Y [[buffer(2)]],
                         constant uint& n_in [[buffer(3)]],
                         constant uint& n_out [[buffer(4)]],
                         constant uint& B [[buffer(5)]],
                         uint tgid [[threadgroup_position_in_grid]],
                         uint sgitg [[simdgroup_index_in_threadgroup]],
                         uint tiitg [[thread_index_in_threadgroup]],
                         threadgroup char* shmem [[threadgroup(0)]]) {
    const uint r0 = tgid * 64u;
    const uint n_kt = n_in >> 7u;   // 128-K tiles

    threadgroup half* sa = (threadgroup half*)(shmem);              // 16 KiB
    threadgroup half* sb = (threadgroup half*)(shmem + 16384);      //  4 KiB

    // Weight staging: thread covers 8 output rows x 8 k of one 8x8 tile.
    const uint wrt = tiitg >> 4u;         // 0..7 row tile
    const uint wkt = tiitg & 15u;         // 0..15 k tile within the 128 chunk
    device const m16_block_q4_K* wrows =
        (device const m16_block_q4_K*)(W) + (r0 + wrt * 8u) * (n_in >> 8u);

    // Activation staging: thread covers 2 batch rows x 8 k.
    const uint am = ((tiitg >> 6u) << 3u) + ((tiitg & 3u) << 1u);  // first batch row
    const uint akt = (tiitg >> 2u) & 15u;                          // k tile
    device const float4* xa = X + am * (n_in >> 2u);
    device const float4* xb = xa + (n_in >> 2u);

    simdgroup_float8x8 mc[4];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 4u; ++i) mc[i] = simdgroup_float8x8(0);

    const uint rh = sgitg & 1u;
    const uint bh = sgitg >> 1u;

    for (uint kt = 0; kt < n_kt; ++kt) {
        // ── Stage weights: dequant 8 rows x 8 k ──
        {
            // Global k window: kt*128 + wkt*8 .. +8. All eight values sit in
            // one 32-value sub-block of one superblock.
            const uint kg = kt * 128u + wkt * 8u;
            const uint sb_i = kg >> 8u;
            const uint ks = kg & 255u;
            const uint g = ks >> 6u;              // group of 64
            const uint l = ks & 63u;              // position within group
            const uint sub = 2u * g + (l >> 5u);  // sub-block 0..7
            const uint shift = (l & 32u) >> 3u;   // 0 or 4
            const uint boff = 32u * g + (l & 31u);

            half wv[8][8];
#pragma clang loop unroll(full)
            for (uint r = 0; r < 8u; ++r) {
                device const m16_block_q4_K* b = wrows + r * (n_in >> 8u) + sb_i;
                float sc, mn;
                m16_q4k_scale_min(b->scales, sub, sc, mn);
                const float dl = float(b->d) * sc;
                const float ml = float(b->dmin) * mn;
                const uchar4 qa = *(device const uchar4*)(b->qs + boff);
                const uchar4 qb = *(device const uchar4*)(b->qs + boff + 4u);
#pragma clang loop unroll(full)
                for (uint i = 0; i < 4u; ++i) {
                    wv[r][i] = half(dl * float((qa[i] >> shift) & 15u) - ml);
                    wv[r][i + 4u] = half(dl * float((qb[i] >> shift) & 15u) - ml);
                }
            }
            // Write tile (wrt, wkt): element (k=j, r=i) at 8*j + i.
            threadgroup half* tile = sa + (8u * wkt + wrt) * 64u;
#pragma clang loop unroll(full)
            for (uint j = 0; j < 8u; ++j) {
                half4 p0, p1;
#pragma clang loop unroll(full)
                for (uint i = 0; i < 4u; ++i) {
                    p0[i] = wv[i][j];
                    p1[i] = wv[i + 4u][j];
                }
                *(threadgroup half4*)(tile + 8u * j) = p0;
                *(threadgroup half4*)(tile + 8u * j + 4u) = p1;
            }
        }
        // ── Stage activations: 2 batch rows x 8 k ──
        {
            const uint kg = kt * 128u + akt * 8u;
            const float4 a0 = xa[kg >> 2u];
            const float4 a1 = xa[(kg >> 2u) + 1u];
            const float4 b0 = xb[kg >> 2u];
            const float4 b1 = xb[(kg >> 2u) + 1u];
            threadgroup half* tile = sb + (2u * akt + (am >> 3u)) * 64u + 8u * (am & 7u);
            *(threadgroup half4*)(tile) = half4(a0);
            *(threadgroup half4*)(tile + 4) = half4(a1);
            *(threadgroup half4*)(tile + 8) = half4(b0);
            *(threadgroup half4*)(tile + 12) = half4(b1);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── MAC: 16 k-steps of 8x8x8 per simdgroup ──
        threadgroup const half* lsma = sa + rh * 4u * 64u;
        threadgroup const half* lsmb = sb + bh * 64u;
#pragma clang loop unroll(full)
        for (uint kk = 0; kk < 16u; ++kk) {
            simdgroup_half8x8 ma[4];
            simdgroup_half8x8 mb;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 4u; ++i)
                simdgroup_load(ma[i], lsma + (8u * kk + i) * 64u, 8);
            simdgroup_load(mb, lsmb + 2u * kk * 64u, 8);
#pragma clang loop unroll(full)
            for (uint i = 0; i < 4u; ++i)
                simdgroup_multiply_accumulate(mc[i], mb, ma[i], mc[i]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store: mc[i][m][r] -> Y[bh*8+m][r0 + rh*32 + i*8 + r].
    device float* C = Y + (bh * 8u) * n_out + r0 + rh * 32u;
#pragma clang loop unroll(full)
    for (uint i = 0; i < 4u; ++i)
        simdgroup_store(mc[i], C + 8u * i, n_out);
}

kernel void m16_q4k_t128_nobar(device const uchar* W [[buffer(0)]],
                         device const float4* X [[buffer(1)]],
                         device float* Y [[buffer(2)]],
                         constant uint& n_in [[buffer(3)]],
                         constant uint& n_out [[buffer(4)]],
                         constant uint& B [[buffer(5)]],
                         uint tgid [[threadgroup_position_in_grid]],
                         uint sgitg [[simdgroup_index_in_threadgroup]],
                         uint tiitg [[thread_index_in_threadgroup]],
                         threadgroup char* shmem [[threadgroup(0)]]) {
    const uint r0 = tgid * 64u;
    const uint n_kt = n_in >> 7u;   // 128-K tiles

    threadgroup half* sa = (threadgroup half*)(shmem);              // 16 KiB
    threadgroup half* sb = (threadgroup half*)(shmem + 16384);      //  4 KiB

    // Weight staging: thread covers 8 output rows x 8 k of one 8x8 tile.
    const uint wrt = tiitg >> 4u;         // 0..7 row tile
    const uint wkt = tiitg & 15u;         // 0..15 k tile within the 128 chunk
    device const m16_block_q4_K* wrows =
        (device const m16_block_q4_K*)(W) + (r0 + wrt * 8u) * (n_in >> 8u);

    // Activation staging: thread covers 2 batch rows x 8 k.
    const uint am = ((tiitg >> 6u) << 3u) + ((tiitg & 3u) << 1u);  // first batch row
    const uint akt = (tiitg >> 2u) & 15u;                          // k tile
    device const float4* xa = X + am * (n_in >> 2u);
    device const float4* xb = xa + (n_in >> 2u);

    simdgroup_float8x8 mc[4];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 4u; ++i) mc[i] = simdgroup_float8x8(0);

    const uint rh = sgitg & 1u;
    const uint bh = sgitg >> 1u;

    for (uint kt = 0; kt < n_kt; ++kt) {
        // ── Stage weights: dequant 8 rows x 8 k ──
        {
            // Global k window: kt*128 + wkt*8 .. +8. All eight values sit in
            // one 32-value sub-block of one superblock.
            const uint kg = kt * 128u + wkt * 8u;
            const uint sb_i = kg >> 8u;
            const uint ks = kg & 255u;
            const uint g = ks >> 6u;              // group of 64
            const uint l = ks & 63u;              // position within group
            const uint sub = 2u * g + (l >> 5u);  // sub-block 0..7
            const uint shift = (l & 32u) >> 3u;   // 0 or 4
            const uint boff = 32u * g + (l & 31u);

            half wv[8][8];
#pragma clang loop unroll(full)
            for (uint r = 0; r < 8u; ++r) {
                device const m16_block_q4_K* b = wrows + r * (n_in >> 8u) + sb_i;
                float sc, mn;
                m16_q4k_scale_min(b->scales, sub, sc, mn);
                const float dl = float(b->d) * sc;
                const float ml = float(b->dmin) * mn;
                const uchar4 qa = *(device const uchar4*)(b->qs + boff);
                const uchar4 qb = *(device const uchar4*)(b->qs + boff + 4u);
#pragma clang loop unroll(full)
                for (uint i = 0; i < 4u; ++i) {
                    wv[r][i] = half(dl * float((qa[i] >> shift) & 15u) - ml);
                    wv[r][i + 4u] = half(dl * float((qb[i] >> shift) & 15u) - ml);
                }
            }
            // Write tile (wrt, wkt): element (k=j, r=i) at 8*j + i.
            threadgroup half* tile = sa + (8u * wkt + wrt) * 64u;
#pragma clang loop unroll(full)
            for (uint j = 0; j < 8u; ++j) {
                half4 p0, p1;
#pragma clang loop unroll(full)
                for (uint i = 0; i < 4u; ++i) {
                    p0[i] = wv[i][j];
                    p1[i] = wv[i + 4u][j];
                }
                *(threadgroup half4*)(tile + 8u * j) = p0;
                *(threadgroup half4*)(tile + 8u * j + 4u) = p1;
            }
        }
        // ── Stage activations: 2 batch rows x 8 k ──
        {
            const uint kg = kt * 128u + akt * 8u;
            const float4 a0 = xa[kg >> 2u];
            const float4 a1 = xa[(kg >> 2u) + 1u];
            const float4 b0 = xb[kg >> 2u];
            const float4 b1 = xb[(kg >> 2u) + 1u];
            threadgroup half* tile = sb + (2u * akt + (am >> 3u)) * 64u + 8u * (am & 7u);
            *(threadgroup half4*)(tile) = half4(a0);
            *(threadgroup half4*)(tile + 4) = half4(a1);
            *(threadgroup half4*)(tile + 8) = half4(b0);
            *(threadgroup half4*)(tile + 12) = half4(b1);
        }


        // ── MAC: 16 k-steps of 8x8x8 per simdgroup ──
        threadgroup const half* lsma = sa + rh * 4u * 64u;
        threadgroup const half* lsmb = sb + bh * 64u;
#pragma clang loop unroll(full)
        for (uint kk = 0; kk < 16u; ++kk) {
            simdgroup_half8x8 ma[4];
            simdgroup_half8x8 mb;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 4u; ++i)
                simdgroup_load(ma[i], lsma + (8u * kk + i) * 64u, 8);
            simdgroup_load(mb, lsmb + 2u * kk * 64u, 8);
#pragma clang loop unroll(full)
            for (uint i = 0; i < 4u; ++i)
                simdgroup_multiply_accumulate(mc[i], mb, ma[i], mc[i]);
        }

    }

    // Store: mc[i][m][r] -> Y[bh*8+m][r0 + rh*32 + i*8 + r].
    device float* C = Y + (bh * 8u) * n_out + r0 + rh * 32u;
#pragma clang loop unroll(full)
    for (uint i = 0; i < 4u; ++i)
        simdgroup_store(mc[i], C + 8u * i, n_out);
}


// ─── debug: MAC/store convention probe ─────────────────────────────────────
// sa tile element (k, r) = (k == r) ? 1 : 0  (identity A tiles)
// sb tile element (m, k) = m * 1000 + k
// Expected output: mc[m][r] = sum_k mb[m][k] * delta(k,r) = mb[m][r]
//   -> Y[m * n_out + r] = m * 1000 + (r % 8) for the 64x16 tile at r0.
kernel void m16_dbg_mac(device const uchar* W [[buffer(0)]],
                        device const float4* X [[buffer(1)]],
                        device float* Y [[buffer(2)]],
                        constant uint& n_in [[buffer(3)]],
                        constant uint& n_out [[buffer(4)]],
                        constant uint& B [[buffer(5)]],
                        uint tgid [[threadgroup_position_in_grid]],
                        uint sgitg [[simdgroup_index_in_threadgroup]],
                        uint tiitg [[thread_index_in_threadgroup]],
                        threadgroup char* shmem [[threadgroup(0)]]) {
    threadgroup half* sa = (threadgroup half*)(shmem);
    threadgroup half* sb = (threadgroup half*)(shmem + 16384);
    // Each thread writes one tile element-row: 64 sa tiles x 8 rows / 128 thr
    // sa tile (rt, kt) at (16*kt + rt)*64, element (k, r) at 8*k + r
    for (uint e = tiitg; e < 8192u; e += 128u) {
        const uint k = (e >> 3u) & 7u;
        const uint r = e & 7u;
        sa[e] = half(k == r ? 1.0f : 0.0f);
    }
    // sb tile (bt, kt) at (2*kt + bt)*64, element (m, k) at 8*m + k
    for (uint e = tiitg; e < 2048u; e += 128u) {
        const uint tile = e >> 6u;
        const uint mk = e & 63u;
        const uint m = mk >> 3u;
        const uint k = mk & 7u;
        const uint bt = tile & 1u;
        sb[e] = half(float((bt * 8u + m) * 64u + k));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 mc[4];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 4u; ++i) mc[i] = simdgroup_float8x8(0);
    const uint rh = sgitg & 1u;
    const uint bh = sgitg >> 1u;
    threadgroup const half* lsma = sa + rh * 4u * 64u;
    threadgroup const half* lsmb = sb + bh * 64u;
    // Only k-step kk = 0 (identity tiles at kt = 0 for rows; others also
    // identity but we take one step to keep the probe minimal).
    {
        simdgroup_half8x8 ma[4];
        simdgroup_half8x8 mb;
#pragma clang loop unroll(full)
        for (uint i = 0; i < 4u; ++i)
            simdgroup_load(ma[i], lsma + i * 64u, 8);
        simdgroup_load(mb, lsmb, 8);
#pragma clang loop unroll(full)
        for (uint i = 0; i < 4u; ++i)
            simdgroup_multiply_accumulate(mc[i], mb, ma[i], mc[i]);
    }
    device float* C = Y + (bh * 8u) * n_out + rh * 32u;
#pragma clang loop unroll(full)
    for (uint i = 0; i < 4u; ++i)
        simdgroup_store(mc[i], C + 8u * i, n_out);
}

// ─── debug: dump the staged weight tile for kt = B (reused as kt index) ────
// Y[k * 64 + r] = staged sa value for tile row r (global r0 + r), k.
kernel void m16_dbg_stage(device const uchar* W [[buffer(0)]],
                          device const float4* X [[buffer(1)]],
                          device float* Y [[buffer(2)]],
                          constant uint& n_in [[buffer(3)]],
                          constant uint& n_out [[buffer(4)]],
                          constant uint& B [[buffer(5)]],
                          uint tgid [[threadgroup_position_in_grid]],
                          uint tiitg [[thread_index_in_threadgroup]],
                          threadgroup char* shmem [[threadgroup(0)]]) {
    if (tgid != 0u) return;
    const uint r0 = tgid * 64u;
    threadgroup half* sa = (threadgroup half*)(shmem);
    const uint wrt = tiitg >> 4u;
    const uint wkt = tiitg & 15u;
    device const m16_block_q4_K* wrows =
        (device const m16_block_q4_K*)(W) + (r0 + wrt * 8u) * (n_in >> 8u);

    const uint kt = B;
    const uint kg = kt * 128u + wkt * 8u;
    const uint sb_i = kg >> 8u;
    const uint ks = kg & 255u;
    const uint g = ks >> 6u;
    const uint l = ks & 63u;
    const uint sub = 2u * g + (l >> 5u);
    const uint shift = (l & 32u) >> 3u;
    const uint boff = 32u * g + (l & 31u);

    half wv[8][8];
#pragma clang loop unroll(full)
    for (uint r = 0; r < 8u; ++r) {
        device const m16_block_q4_K* b = wrows + r * (n_in >> 8u) + sb_i;
        float sc, mn;
        m16_q4k_scale_min(b->scales, sub, sc, mn);
        const float dl = float(b->d) * sc;
        const float ml = float(b->dmin) * mn;
        const uchar4 qa = *(device const uchar4*)(b->qs + boff);
        const uchar4 qb = *(device const uchar4*)(b->qs + boff + 4u);
#pragma clang loop unroll(full)
        for (uint i = 0; i < 4u; ++i) {
            wv[r][i] = half(dl * float((qa[i] >> shift) & 15u) - ml);
            wv[r][i + 4u] = half(dl * float((qb[i] >> shift) & 15u) - ml);
        }
    }
    threadgroup half* tile = sa + (8u * wkt + wrt) * 64u;
#pragma clang loop unroll(full)
    for (uint j = 0; j < 8u; ++j) {
        half4 p0, p1;
#pragma clang loop unroll(full)
        for (uint i = 0; i < 4u; ++i) {
            p0[i] = wv[i][j];
            p1[i] = wv[i + 4u][j];
        }
        *(threadgroup half4*)(tile + 8u * j) = p0;
        *(threadgroup half4*)(tile + 8u * j + 4u) = p1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Dump: element (k, r) at tile (r>>3, k>>3) offset + 8*(k&7) + (r&7).
    for (uint e = tiitg; e < 8192u; e += 128u) {
        const uint k = e >> 6u;
        const uint r = e & 63u;
        Y[k * 64u + r] = float(sa[(8u * (k >> 3u) + (r >> 3u)) * 64u + 8u * (k & 7u) + (r & 7u)]);
    }
}

kernel void m16_dbg_mark(device const uchar* W [[buffer(0)]],
                          device const float4* X [[buffer(1)]],
                          device float* Y [[buffer(2)]],
                          constant uint& n_in [[buffer(3)]],
                          constant uint& n_out [[buffer(4)]],
                          constant uint& B [[buffer(5)]],
                          uint tgid [[threadgroup_position_in_grid]],
                          uint tiitg [[thread_index_in_threadgroup]],
                          threadgroup char* shmem [[threadgroup(0)]]) {
    if (tgid != 0u) return;
    const uint r0 = tgid * 64u;
    threadgroup half* sa = (threadgroup half*)(shmem);
    const uint wrt = tiitg >> 4u;
    const uint wkt = tiitg & 15u;
    device const m16_block_q4_K* wrows =
        (device const m16_block_q4_K*)(W) + (r0 + wrt * 8u) * (n_in >> 8u);

    const uint kt = B;
    const uint kg = kt * 128u + wkt * 8u;
    const uint sb_i = kg >> 8u;
    const uint ks = kg & 255u;
    const uint g = ks >> 6u;
    const uint l = ks & 63u;
    const uint sub = 2u * g + (l >> 5u);
    const uint shift = (l & 32u) >> 3u;
    const uint boff = 32u * g + (l & 31u);

    half wv[8][8];
#pragma clang loop unroll(full)
    for (uint r = 0; r < 8u; ++r)
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8u; ++i) wv[r][i] = half(float(wkt * 100u + r));
    threadgroup half* tile = sa + (8u * wkt + wrt) * 64u;
#pragma clang loop unroll(full)
    for (uint j = 0; j < 8u; ++j) {
        half4 p0, p1;
#pragma clang loop unroll(full)
        for (uint i = 0; i < 4u; ++i) {
            p0[i] = wv[i][j];
            p1[i] = wv[i + 4u][j];
        }
        *(threadgroup half4*)(tile + 8u * j) = p0;
        *(threadgroup half4*)(tile + 8u * j + 4u) = p1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Dump: element (k, r) at tile (r>>3, k>>3) offset + 8*(k&7) + (r&7).
    for (uint e = tiitg; e < 8192u; e += 128u) {
        const uint k = e >> 6u;
        const uint r = e & 63u;
        Y[k * 64u + r] = float(sa[(8u * (k >> 3u) + (r >> 3u)) * 64u + 8u * (k & 7u) + (r & 7u)]);
    }
}

kernel void m16_q4k_t128_nomac(device const uchar* W [[buffer(0)]],
                         device const float4* X [[buffer(1)]],
                         device float* Y [[buffer(2)]],
                         constant uint& n_in [[buffer(3)]],
                         constant uint& n_out [[buffer(4)]],
                         constant uint& B [[buffer(5)]],
                         uint tgid [[threadgroup_position_in_grid]],
                         uint sgitg [[simdgroup_index_in_threadgroup]],
                         uint tiitg [[thread_index_in_threadgroup]],
                         threadgroup char* shmem [[threadgroup(0)]]) {
    const uint r0 = tgid * 64u;
    const uint n_kt = n_in >> 7u;   // 128-K tiles

    threadgroup half* sa = (threadgroup half*)(shmem);              // 16 KiB
    threadgroup half* sb = (threadgroup half*)(shmem + 16384);      //  4 KiB

    // Weight staging: thread covers 8 output rows x 8 k of one 8x8 tile.
    const uint wrt = tiitg >> 4u;         // 0..7 row tile
    const uint wkt = tiitg & 15u;         // 0..15 k tile within the 128 chunk
    device const m16_block_q4_K* wrows =
        (device const m16_block_q4_K*)(W) + (r0 + wrt * 8u) * (n_in >> 8u);

    // Activation staging: thread covers 2 batch rows x 8 k.
    const uint am = ((tiitg >> 6u) << 3u) + ((tiitg & 3u) << 1u);  // first batch row
    const uint akt = (tiitg >> 2u) & 15u;                          // k tile
    device const float4* xa = X + am * (n_in >> 2u);
    device const float4* xb = xa + (n_in >> 2u);

    simdgroup_float8x8 mc[4];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 4u; ++i) mc[i] = simdgroup_float8x8(0);

    const uint rh = sgitg & 1u;
    const uint bh = sgitg >> 1u;

    for (uint kt = 0; kt < n_kt; ++kt) {
        // ── Stage weights: dequant 8 rows x 8 k ──
        {
            // Global k window: kt*128 + wkt*8 .. +8. All eight values sit in
            // one 32-value sub-block of one superblock.
            const uint kg = kt * 128u + wkt * 8u;
            const uint sb_i = kg >> 8u;
            const uint ks = kg & 255u;
            const uint g = ks >> 6u;              // group of 64
            const uint l = ks & 63u;              // position within group
            const uint sub = 2u * g + (l >> 5u);  // sub-block 0..7
            const uint shift = (l & 32u) >> 3u;   // 0 or 4
            const uint boff = 32u * g + (l & 31u);

            half wv[8][8];
#pragma clang loop unroll(full)
            for (uint r = 0; r < 8u; ++r) {
                device const m16_block_q4_K* b = wrows + r * (n_in >> 8u) + sb_i;
                float sc, mn;
                m16_q4k_scale_min(b->scales, sub, sc, mn);
                const float dl = float(b->d) * sc;
                const float ml = float(b->dmin) * mn;
                const uchar4 qa = *(device const uchar4*)(b->qs + boff);
                const uchar4 qb = *(device const uchar4*)(b->qs + boff + 4u);
#pragma clang loop unroll(full)
                for (uint i = 0; i < 4u; ++i) {
                    wv[r][i] = half(dl * float((qa[i] >> shift) & 15u) - ml);
                    wv[r][i + 4u] = half(dl * float((qb[i] >> shift) & 15u) - ml);
                }
            }
            // Write tile (wrt, wkt): element (k=j, r=i) at 8*j + i.
            threadgroup half* tile = sa + (8u * wkt + wrt) * 64u;
#pragma clang loop unroll(full)
            for (uint j = 0; j < 8u; ++j) {
                half4 p0, p1;
#pragma clang loop unroll(full)
                for (uint i = 0; i < 4u; ++i) {
                    p0[i] = wv[i][j];
                    p1[i] = wv[i + 4u][j];
                }
                *(threadgroup half4*)(tile + 8u * j) = p0;
                *(threadgroup half4*)(tile + 8u * j + 4u) = p1;
            }
        }
        // ── Stage activations: 2 batch rows x 8 k ──
        {
            const uint kg = kt * 128u + akt * 8u;
            const float4 a0 = xa[kg >> 2u];
            const float4 a1 = xa[(kg >> 2u) + 1u];
            const float4 b0 = xb[kg >> 2u];
            const float4 b1 = xb[(kg >> 2u) + 1u];
            threadgroup half* tile = sb + (2u * akt + (am >> 3u)) * 64u + 8u * (am & 7u);
            *(threadgroup half4*)(tile) = half4(a0);
            *(threadgroup half4*)(tile + 4) = half4(a1);
            *(threadgroup half4*)(tile + 8) = half4(b0);
            *(threadgroup half4*)(tile + 12) = half4(b1);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // MAC removed for the staging-only probe
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store: mc[i][m][r] -> Y[bh*8+m][r0 + rh*32 + i*8 + r].
    device float* C = Y + (bh * 8u) * n_out + r0 + rh * 32u;
#pragma clang loop unroll(full)
    for (uint i = 0; i < 4u; ++i)
        simdgroup_store(mc[i], C + 8u * i, n_out);
}

// ─── debug: MAC/store convention probe ─────────────────────────────────────
// sa tile element (k, r) = (k == r) ? 1 : 0  (identity A tiles)
// sb tile element (m, k) = m * 1000 + k
// Expected output: mc[m][r] = sum_k mb[m][k] * delta(k,r) = mb[m][r]
//   -> Y[m * n_out + r] = m * 1000 + (r % 8) for the 64x16 tile at r0.
kernel void m16_q4k_t128_nodeq(device const uchar* W [[buffer(0)]],
                         device const float4* X [[buffer(1)]],
                         device float* Y [[buffer(2)]],
                         constant uint& n_in [[buffer(3)]],
                         constant uint& n_out [[buffer(4)]],
                         constant uint& B [[buffer(5)]],
                         uint tgid [[threadgroup_position_in_grid]],
                         uint sgitg [[simdgroup_index_in_threadgroup]],
                         uint tiitg [[thread_index_in_threadgroup]],
                         threadgroup char* shmem [[threadgroup(0)]]) {
    const uint r0 = tgid * 64u;
    const uint n_kt = n_in >> 7u;   // 128-K tiles

    threadgroup half* sa = (threadgroup half*)(shmem);              // 16 KiB
    threadgroup half* sb = (threadgroup half*)(shmem + 16384);      //  4 KiB

    // Weight staging: thread covers 8 output rows x 8 k of one 8x8 tile.
    const uint wrt = tiitg >> 4u;         // 0..7 row tile
    const uint wkt = tiitg & 15u;         // 0..15 k tile within the 128 chunk
    device const m16_block_q4_K* wrows =
        (device const m16_block_q4_K*)(W) + (r0 + wrt * 8u) * (n_in >> 8u);

    // Activation staging: thread covers 2 batch rows x 8 k.
    const uint am = ((tiitg >> 6u) << 3u) + ((tiitg & 3u) << 1u);  // first batch row
    const uint akt = (tiitg >> 2u) & 15u;                          // k tile
    device const float4* xa = X + am * (n_in >> 2u);
    device const float4* xb = xa + (n_in >> 2u);

    simdgroup_float8x8 mc[4];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 4u; ++i) mc[i] = simdgroup_float8x8(0);

    const uint rh = sgitg & 1u;
    const uint bh = sgitg >> 1u;

    for (uint kt = 0; kt < n_kt; ++kt) {
        // Weights replaced by constants (staging+MAC probe)
        {
            threadgroup half* tile = sa + (8u * wkt + wrt) * 64u;
#pragma clang loop unroll(full)
            for (uint j = 0; j < 8u; ++j) {
                *(threadgroup half4*)(tile + 8u * j) = half4(half(0.01f));
                *(threadgroup half4*)(tile + 8u * j + 4u) = half4(half(0.01f));
            }
        }
        // ── Stage activations: 2 batch rows x 8 k ──
        {
            const uint kg = kt * 128u + akt * 8u;
            const float4 a0 = xa[kg >> 2u];
            const float4 a1 = xa[(kg >> 2u) + 1u];
            const float4 b0 = xb[kg >> 2u];
            const float4 b1 = xb[(kg >> 2u) + 1u];
            threadgroup half* tile = sb + (2u * akt + (am >> 3u)) * 64u + 8u * (am & 7u);
            *(threadgroup half4*)(tile) = half4(a0);
            *(threadgroup half4*)(tile + 4) = half4(a1);
            *(threadgroup half4*)(tile + 8) = half4(b0);
            *(threadgroup half4*)(tile + 12) = half4(b1);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── MAC: 16 k-steps of 8x8x8 per simdgroup ──
        threadgroup const half* lsma = sa + rh * 4u * 64u;
        threadgroup const half* lsmb = sb + bh * 64u;
#pragma clang loop unroll(full)
        for (uint kk = 0; kk < 16u; ++kk) {
            simdgroup_half8x8 ma[4];
            simdgroup_half8x8 mb;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 4u; ++i)
                simdgroup_load(ma[i], lsma + (8u * kk + i) * 64u, 8);
            simdgroup_load(mb, lsmb + 2u * kk * 64u, 8);
#pragma clang loop unroll(full)
            for (uint i = 0; i < 4u; ++i)
                simdgroup_multiply_accumulate(mc[i], mb, ma[i], mc[i]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store: mc[i][m][r] -> Y[bh*8+m][r0 + rh*32 + i*8 + r].
    device float* C = Y + (bh * 8u) * n_out + r0 + rh * 32u;
#pragma clang loop unroll(full)
    for (uint i = 0; i < 4u; ++i)
        simdgroup_store(mc[i], C + 8u * i, n_out);
}

// ─── debug: MAC/store convention probe ─────────────────────────────────────
// sa tile element (k, r) = (k == r) ? 1 : 0  (identity A tiles)
// sb tile element (m, k) = m * 1000 + k
// Expected output: mc[m][r] = sum_k mb[m][k] * delta(k,r) = mb[m][r]
//   -> Y[m * n_out + r] = m * 1000 + (r % 8) for the 64x16 tile at r0.
struct M16Q4KPrefetch {
    uchar4 qa[8][2];
    float dl[8];
    float ml[8];
};

kernel void m16_q4k_t128p(device const uchar* W [[buffer(0)]],
                          device const float4* X [[buffer(1)]],
                          device float* Y [[buffer(2)]],
                          constant uint& n_in [[buffer(3)]],
                          constant uint& n_out [[buffer(4)]],
                          constant uint& B [[buffer(5)]],
                          uint tgid [[threadgroup_position_in_grid]],
                          uint sgitg [[simdgroup_index_in_threadgroup]],
                          uint tiitg [[thread_index_in_threadgroup]],
                          threadgroup char* shmem [[threadgroup(0)]]) {
    const uint r0 = tgid * 64u;
    const uint n_kt = n_in >> 7u;
    const uint nsb = n_in >> 8u;

    threadgroup half* sa = (threadgroup half*)(shmem);
    threadgroup half* sb = (threadgroup half*)(shmem + 16384);

    const uint wrt = tiitg >> 4u;
    const uint wkt = tiitg & 15u;
    device const m16_block_q4_K* wrows =
        (device const m16_block_q4_K*)(W) + (r0 + wrt * 8u) * nsb;

    const uint am = ((tiitg >> 6u) << 3u) + ((tiitg & 3u) << 1u);
    const uint akt = (tiitg >> 2u) & 15u;
    device const float4* xa = X + am * (n_in >> 2u);
    device const float4* xb = xa + (n_in >> 2u);

    simdgroup_float8x8 mc[4];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 4u; ++i) mc[i] = simdgroup_float8x8(0);

    const uint rh = sgitg & 1u;
    const uint bh = sgitg >> 1u;

    auto prefetch = [&](uint kt, thread M16Q4KPrefetch& pf) {
        const uint kg = kt * 128u + wkt * 8u;
        const uint sb_i = kg >> 8u;
        const uint ks = kg & 255u;
        const uint g = ks >> 6u;
        const uint l = ks & 63u;
        const uint sub = 2u * g + (l >> 5u);
        const uint shift = (l & 32u) >> 3u;
        const uint boff = 32u * g + (l & 31u);
#pragma clang loop unroll(full)
        for (uint r = 0; r < 8u; ++r) {
            device const m16_block_q4_K* b = wrows + r * nsb + sb_i;
            float sc, mn;
            m16_q4k_scale_min(b->scales, sub, sc, mn);
            pf.dl[r] = float(b->d) * sc;
            pf.ml[r] = float(b->dmin) * mn;
            pf.qa[r][0] = *(device const uchar4*)(b->qs + boff);
            pf.qa[r][1] = *(device const uchar4*)(b->qs + boff + 4u);
        }
    };

    M16Q4KPrefetch pf;
    prefetch(0, pf);

    for (uint kt = 0; kt < n_kt; ++kt) {
        M16Q4KPrefetch cur = pf;
        if (kt + 1u < n_kt) prefetch(kt + 1, pf);

        // Dequant staged bytes -> sa
        {
            const uint kg = kt * 128u + wkt * 8u;
            const uint ks = kg & 255u;
            const uint l = ks & 63u;
            const uint shift = (l & 32u) >> 3u;
            half wv[8][8];
#pragma clang loop unroll(full)
            for (uint r = 0; r < 8u; ++r) {
#pragma clang loop unroll(full)
                for (uint i = 0; i < 4u; ++i) {
                    wv[r][i] = half(cur.dl[r] * float((cur.qa[r][0][i] >> shift) & 15u) - cur.ml[r]);
                    wv[r][i + 4u] = half(cur.dl[r] * float((cur.qa[r][1][i] >> shift) & 15u) - cur.ml[r]);
                }
            }
            threadgroup half* tile = sa + (8u * wkt + wrt) * 64u;
#pragma clang loop unroll(full)
            for (uint j = 0; j < 8u; ++j) {
                half4 p0, p1;
#pragma clang loop unroll(full)
                for (uint i = 0; i < 4u; ++i) {
                    p0[i] = wv[i][j];
                    p1[i] = wv[i + 4u][j];
                }
                *(threadgroup half4*)(tile + 8u * j) = p0;
                *(threadgroup half4*)(tile + 8u * j + 4u) = p1;
            }
        }
        // Stage activations
        {
            const uint kg = kt * 128u + akt * 8u;
            const float4 a0 = xa[kg >> 2u];
            const float4 a1 = xa[(kg >> 2u) + 1u];
            const float4 b0 = xb[kg >> 2u];
            const float4 b1 = xb[(kg >> 2u) + 1u];
            threadgroup half* tile = sb + (2u * akt + (am >> 3u)) * 64u + 8u * (am & 7u);
            *(threadgroup half4*)(tile) = half4(a0);
            *(threadgroup half4*)(tile + 4) = half4(a1);
            *(threadgroup half4*)(tile + 8) = half4(b0);
            *(threadgroup half4*)(tile + 12) = half4(b1);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half* lsma = sa + rh * 4u * 64u;
        threadgroup const half* lsmb = sb + bh * 64u;
#pragma clang loop unroll(full)
        for (uint kk = 0; kk < 16u; ++kk) {
            simdgroup_half8x8 ma[4];
            simdgroup_half8x8 mb;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 4u; ++i)
                simdgroup_load(ma[i], lsma + (8u * kk + i) * 64u, 8);
            simdgroup_load(mb, lsmb + 2u * kk * 64u, 8);
#pragma clang loop unroll(full)
            for (uint i = 0; i < 4u; ++i)
                simdgroup_multiply_accumulate(mc[i], mb, ma[i], mc[i]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* C = Y + (bh * 8u) * n_out + r0 + rh * 32u;
#pragma clang loop unroll(full)
    for (uint i = 0; i < 4u; ++i)
        simdgroup_store(mc[i], C + 8u * i, n_out);
}

// ─── t128x: t128 with cross-simdgroup K-tile split ─────────────────────────
// Each simdgroup owns 32 output rows x all 16 batch columns and processes
// half of each 128-K tile's k-steps, so every mb tile load feeds 4 MACs and
// every ma tile load feeds 2 MACs. Partial products are summed through
// threadgroup f32 scratch at the end of the K loop.

kernel void m16_q4k_t128x(device const uchar* W [[buffer(0)]],
                          device const float4* X [[buffer(1)]],
                          device float* Y [[buffer(2)]],
                          constant uint& n_in [[buffer(3)]],
                          constant uint& n_out [[buffer(4)]],
                          constant uint& B [[buffer(5)]],
                          uint tgid [[threadgroup_position_in_grid]],
                          uint sgitg [[simdgroup_index_in_threadgroup]],
                          uint tiitg [[thread_index_in_threadgroup]],
                          threadgroup char* shmem [[threadgroup(0)]]) {
    const uint r0 = tgid * 64u;
    const uint n_kt = n_in >> 7u;
    const uint nsb = n_in >> 8u;

    threadgroup half* sa = (threadgroup half*)(shmem);
    threadgroup half* sb = (threadgroup half*)(shmem + 16384);

    const uint wrt = tiitg >> 4u;
    const uint wkt = tiitg & 15u;
    device const m16_block_q4_K* wrows =
        (device const m16_block_q4_K*)(W) + (r0 + wrt * 8u) * nsb;

    const uint am = ((tiitg >> 6u) << 3u) + ((tiitg & 3u) << 1u);
    const uint akt = (tiitg >> 2u) & 15u;
    device const float4* xa = X + am * (n_in >> 2u);
    device const float4* xb = xa + (n_in >> 2u);

    simdgroup_float8x8 mc2[4][2];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 4u; ++i)
#pragma clang loop unroll(full)
        for (uint b = 0; b < 2u; ++b) mc2[i][b] = simdgroup_float8x8(0);

    const uint rh = sgitg & 1u;
    const uint kkh = sgitg >> 1u;

    for (uint kt = 0; kt < n_kt; ++kt) {
        {
            const uint kg = kt * 128u + wkt * 8u;
            const uint sb_i = kg >> 8u;
            const uint ks = kg & 255u;
            const uint g = ks >> 6u;
            const uint l = ks & 63u;
            const uint sub = 2u * g + (l >> 5u);
            const uint shift = (l & 32u) >> 3u;
            const uint boff = 32u * g + (l & 31u);
            half wv[8][8];
#pragma clang loop unroll(full)
            for (uint r = 0; r < 8u; ++r) {
                device const m16_block_q4_K* b = wrows + r * nsb + sb_i;
                float sc, mn;
                m16_q4k_scale_min(b->scales, sub, sc, mn);
                const float dl = float(b->d) * sc;
                const float ml = float(b->dmin) * mn;
                const uchar4 qa = *(device const uchar4*)(b->qs + boff);
                const uchar4 qb = *(device const uchar4*)(b->qs + boff + 4u);
#pragma clang loop unroll(full)
                for (uint i = 0; i < 4u; ++i) {
                    wv[r][i] = half(dl * float((qa[i] >> shift) & 15u) - ml);
                    wv[r][i + 4u] = half(dl * float((qb[i] >> shift) & 15u) - ml);
                }
            }
            threadgroup half* tile = sa + (8u * wkt + wrt) * 64u;
#pragma clang loop unroll(full)
            for (uint j = 0; j < 8u; ++j) {
                half4 p0, p1;
#pragma clang loop unroll(full)
                for (uint i = 0; i < 4u; ++i) {
                    p0[i] = wv[i][j];
                    p1[i] = wv[i + 4u][j];
                }
                *(threadgroup half4*)(tile + 8u * j) = p0;
                *(threadgroup half4*)(tile + 8u * j + 4u) = p1;
            }
        }
        {
            const uint kg = kt * 128u + akt * 8u;
            const float4 a0 = xa[kg >> 2u];
            const float4 a1 = xa[(kg >> 2u) + 1u];
            const float4 b0 = xb[kg >> 2u];
            const float4 b1 = xb[(kg >> 2u) + 1u];
            threadgroup half* tile = sb + (2u * akt + (am >> 3u)) * 64u + 8u * (am & 7u);
            *(threadgroup half4*)(tile) = half4(a0);
            *(threadgroup half4*)(tile + 4) = half4(a1);
            *(threadgroup half4*)(tile + 8) = half4(b0);
            *(threadgroup half4*)(tile + 12) = half4(b1);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half* lsma = sa + rh * 4u * 64u;
#pragma clang loop unroll(full)
        for (uint kk0 = 0; kk0 < 8u; ++kk0) {
            const uint kk = kkh * 8u + kk0;
            simdgroup_half8x8 ma[4];
            simdgroup_half8x8 mb[2];
#pragma clang loop unroll(full)
            for (uint i = 0; i < 4u; ++i)
                simdgroup_load(ma[i], lsma + (8u * kk + i) * 64u, 8);
#pragma clang loop unroll(full)
            for (uint i = 0; i < 2u; ++i)
                simdgroup_load(mb[i], sb + (2u * kk + i) * 64u, 8);
#pragma clang loop unroll(full)
            for (uint i = 0; i < 4u; ++i) {
                simdgroup_multiply_accumulate(mc2[i][0], mb[0], ma[i], mc2[i][0]);
                simdgroup_multiply_accumulate(mc2[i][1], mb[1], ma[i], mc2[i][1]);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Reduce the kkh halves through threadgroup f32 scratch, then store.
    threadgroup float* red = (threadgroup float*)shmem;  // 4 sgs x 8 tiles x 64
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
    for (uint i = 0; i < 4u; ++i)
#pragma clang loop unroll(full)
        for (uint b = 0; b < 2u; ++b) {
            const uint tile = ((kkh * 2u + rh) * 4u + i) * 2u + b;
            simdgroup_store(mc2[i][b], red + tile * 64u, 8u);
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 128 threads: each finalizes one output row r and 8 batch columns.
    const uint fr = tiitg >> 1u;          // 0..63 output row
    const uint fm0 = (tiitg & 1u) << 3u;  // first batch column (8 wide)
    const uint frh = fr >> 5u;
    const uint fi = (fr >> 3u) & 3u;
    const uint fb = fm0 >> 3u;
#pragma clang loop unroll(full)
    for (uint m = 0; m < 8u; ++m) {
        const uint e0 = (((0u * 2u + frh) * 4u + fi) * 2u + fb) * 64u + (fm0 & 7u | m) * 8u * 0u;
        (void)e0;
        const uint base0 = (((0u * 2u + frh) * 4u + fi) * 2u + fb) * 64u + m * 8u + (fr & 7u);
        const uint base1 = (((1u * 2u + frh) * 4u + fi) * 2u + fb) * 64u + m * 8u + (fr & 7u);
        const float v = red[base0] + red[base1];
        Y[(fm0 + m) * n_out + r0 + fr] = v;
    }
}

// ─── t64: same tile family, NK=64 (10 KiB tgmem -> higher occupancy) ───────
// TG = 128 threads, 4 simdgroups, sg (rh, bh) = 32 rows x 8 batch columns.
// sa: 64 rows x 64 k, tile (rt, kt) at (8*kt + rt)*64. sb: 16 x 64, tile
// (bt, kt) at (2*kt + bt)*64. Preconditions: n_in % 256 == 0, n_out % 64 == 0.

kernel void m16_q4k_t64(device const uchar* W [[buffer(0)]],
                        device const float4* X [[buffer(1)]],
                        device float* Y [[buffer(2)]],
                        constant uint& n_in [[buffer(3)]],
                        constant uint& n_out [[buffer(4)]],
                        constant uint& B [[buffer(5)]],
                        uint tgid [[threadgroup_position_in_grid]],
                        uint sgitg [[simdgroup_index_in_threadgroup]],
                        uint tiitg [[thread_index_in_threadgroup]],
                        threadgroup char* shmem [[threadgroup(0)]]) {
    const uint r0 = tgid * 64u;
    const uint n_kt = n_in >> 6u;
    const uint nsb = n_in >> 8u;

    threadgroup half* sa = (threadgroup half*)(shmem);             // 8 KiB
    threadgroup half* sb = (threadgroup half*)(shmem + 8192);      // 2 KiB

    // Weight staging: thread covers 4 rows x 8 k.
    const uint wrt = tiitg >> 3u;         // 0..15 quarter row tile
    const uint wkt = tiitg & 7u;          // 0..7 k tile within 64 chunk
    device const m16_block_q4_K* wrows =
        (device const m16_block_q4_K*)(W) + (r0 + wrt * 4u) * nsb;

    // Activation staging: thread covers 1 batch row x 8 k.
    const uint am = tiitg >> 3u;          // 0..15
    const uint akt = tiitg & 7u;
    device const float4* xa = X + am * (n_in >> 2u);

    simdgroup_float8x8 mc[4];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 4u; ++i) mc[i] = simdgroup_float8x8(0);

    const uint rh = sgitg & 1u;
    const uint bh = sgitg >> 1u;

    for (uint kt = 0; kt < n_kt; ++kt) {
        {
            const uint kg = kt * 64u + wkt * 8u;
            const uint sb_i = kg >> 8u;
            const uint ks = kg & 255u;
            const uint g = ks >> 6u;
            const uint l = ks & 63u;
            const uint sub = 2u * g + (l >> 5u);
            const uint shift = (l & 32u) >> 3u;
            const uint boff = 32u * g + (l & 31u);
            half wv[4][8];
#pragma clang loop unroll(full)
            for (uint r = 0; r < 4u; ++r) {
                device const m16_block_q4_K* b = wrows + r * nsb + sb_i;
                float sc, mn;
                m16_q4k_scale_min(b->scales, sub, sc, mn);
                const float dl = float(b->d) * sc;
                const float ml = float(b->dmin) * mn;
                const uchar4 qa = *(device const uchar4*)(b->qs + boff);
                const uchar4 qb = *(device const uchar4*)(b->qs + boff + 4u);
#pragma clang loop unroll(full)
                for (uint i = 0; i < 4u; ++i) {
                    wv[r][i] = half(dl * float((qa[i] >> shift) & 15u) - ml);
                    wv[r][i + 4u] = half(dl * float((qb[i] >> shift) & 15u) - ml);
                }
            }
            // Write tile (rt = wrt>>1, wkt), element (k=j, r) at 8*j + r where
            // this thread covers rows r = (wrt&1)*4 .. +3.
            threadgroup half* tile = sa + (8u * wkt + (wrt >> 1u)) * 64u + ((wrt & 1u) << 2u);
#pragma clang loop unroll(full)
            for (uint j = 0; j < 8u; ++j) {
                half4 p;
#pragma clang loop unroll(full)
                for (uint i = 0; i < 4u; ++i) p[i] = wv[i][j];
                *(threadgroup half4*)(tile + 8u * j) = p;
            }
        }
        {
            const uint kg = kt * 64u + akt * 8u;
            const float4 a0 = xa[kg >> 2u];
            const float4 a1 = xa[(kg >> 2u) + 1u];
            threadgroup half* tile = sb + (2u * akt + (am >> 3u)) * 64u + 8u * (am & 7u);
            *(threadgroup half4*)(tile) = half4(a0);
            *(threadgroup half4*)(tile + 4) = half4(a1);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half* lsma = sa + rh * 4u * 64u;
        threadgroup const half* lsmb = sb + bh * 64u;
#pragma clang loop unroll(full)
        for (uint kk = 0; kk < 8u; ++kk) {
            simdgroup_half8x8 ma[4];
            simdgroup_half8x8 mb;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 4u; ++i)
                simdgroup_load(ma[i], lsma + (8u * kk + i) * 64u, 8);
            simdgroup_load(mb, lsmb + 2u * kk * 64u, 8);
#pragma clang loop unroll(full)
            for (uint i = 0; i < 4u; ++i)
                simdgroup_multiply_accumulate(mc[i], mb, ma[i], mc[i]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* C = Y + (bh * 8u) * n_out + r0 + rh * 32u;
#pragma clang loop unroll(full)
    for (uint i = 0; i < 4u; ++i)
        simdgroup_store(mc[i], C + 8u * i, n_out);
}

// ─── t32: NK=32, NR0=64, 5 KiB tgmem ───────────────────────────────────────
kernel void m16_q4k_t32(device const uchar* W [[buffer(0)]],
                        device const float4* X [[buffer(1)]],
                        device float* Y [[buffer(2)]],
                        constant uint& n_in [[buffer(3)]],
                        constant uint& n_out [[buffer(4)]],
                        constant uint& B [[buffer(5)]],
                        uint tgid [[threadgroup_position_in_grid]],
                        uint sgitg [[simdgroup_index_in_threadgroup]],
                        uint tiitg [[thread_index_in_threadgroup]],
                        threadgroup char* shmem [[threadgroup(0)]]) {
    const uint r0 = tgid * 64u;
    const uint n_kt = n_in >> 5u;
    const uint nsb = n_in >> 8u;

    threadgroup half* sa = (threadgroup half*)(shmem);             // 4 KiB
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);      // 1 KiB

    // Weight staging: thread covers 2 rows x 8 k.
    const uint wrg = tiitg >> 2u;         // 0..31 row pair group
    const uint wkt = tiitg & 3u;          // 0..3 k tile
    device const m16_block_q4_K* wrows =
        (device const m16_block_q4_K*)(W) + (r0 + wrg * 2u) * nsb;

    // Activation staging: thread covers 1 batch row x 4 k.
    const uint am = tiitg >> 3u;          // 0..15
    const uint akt = tiitg & 7u;          // 4-k chunk
    device const float4* xa = X + am * (n_in >> 2u);

    simdgroup_float8x8 mc[4];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 4u; ++i) mc[i] = simdgroup_float8x8(0);

    const uint rh = sgitg & 1u;
    const uint bh = sgitg >> 1u;

    for (uint kt = 0; kt < n_kt; ++kt) {
        {
            const uint kg = kt * 32u + wkt * 8u;
            const uint sb_i = kg >> 8u;
            const uint ks = kg & 255u;
            const uint g = ks >> 6u;
            const uint l = ks & 63u;
            const uint sub = 2u * g + (l >> 5u);
            const uint shift = (l & 32u) >> 3u;
            const uint boff = 32u * g + (l & 31u);
            half wv[2][8];
#pragma clang loop unroll(full)
            for (uint r = 0; r < 2u; ++r) {
                device const m16_block_q4_K* b = wrows + r * nsb + sb_i;
                float sc, mn;
                m16_q4k_scale_min(b->scales, sub, sc, mn);
                const float dl = float(b->d) * sc;
                const float ml = float(b->dmin) * mn;
                const uchar4 qa = *(device const uchar4*)(b->qs + boff);
                const uchar4 qb = *(device const uchar4*)(b->qs + boff + 4u);
#pragma clang loop unroll(full)
                for (uint i = 0; i < 4u; ++i) {
                    wv[r][i] = half(dl * float((qa[i] >> shift) & 15u) - ml);
                    wv[r][i + 4u] = half(dl * float((qb[i] >> shift) & 15u) - ml);
                }
            }
            // rows (wrg&3)*2..+1, row tile wrg>>2
            threadgroup half* tile = sa + (8u * wkt + (wrg >> 2u)) * 64u + ((wrg & 3u) << 1u);
#pragma clang loop unroll(full)
            for (uint j = 0; j < 8u; ++j) {
                *(threadgroup half*)(tile + 8u * j) = wv[0][j];
                *(threadgroup half*)(tile + 8u * j + 1u) = wv[1][j];
            }
        }
        {
            const uint kg = kt * 32u + akt * 4u;
            const float4 a0 = xa[kg >> 2u];
            // tile (bt, kt4) where kt4 = akt>>1, k-in-tile (akt&1)*4..+4
            threadgroup half* tile = sb + (2u * (akt >> 1u) + (am >> 3u)) * 64u + 8u * (am & 7u) + ((akt & 1u) << 2u);
            *(threadgroup half4*)(tile) = half4(a0);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half* lsma = sa + rh * 4u * 64u;
        threadgroup const half* lsmb = sb + bh * 64u;
#pragma clang loop unroll(full)
        for (uint kk = 0; kk < 4u; ++kk) {
            simdgroup_half8x8 ma[4];
            simdgroup_half8x8 mb;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 4u; ++i)
                simdgroup_load(ma[i], lsma + (8u * kk + i) * 64u, 8);
            simdgroup_load(mb, lsmb + 2u * kk * 64u, 8);
#pragma clang loop unroll(full)
            for (uint i = 0; i < 4u; ++i)
                simdgroup_multiply_accumulate(mc[i], mb, ma[i], mc[i]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* C = Y + (bh * 8u) * n_out + r0 + rh * 32u;
#pragma clang loop unroll(full)
    for (uint i = 0; i < 4u; ++i)
        simdgroup_store(mc[i], C + 8u * i, n_out);
}

// ─── n32: NR0=32, NK=64, 6 KiB tgmem, 2 row tiles x 2 batch halves ─────────
kernel void m16_q4k_n32(device const uchar* W [[buffer(0)]],
                        device const float4* X [[buffer(1)]],
                        device float* Y [[buffer(2)]],
                        constant uint& n_in [[buffer(3)]],
                        constant uint& n_out [[buffer(4)]],
                        constant uint& B [[buffer(5)]],
                        uint tgid [[threadgroup_position_in_grid]],
                        uint sgitg [[simdgroup_index_in_threadgroup]],
                        uint tiitg [[thread_index_in_threadgroup]],
                        threadgroup char* shmem [[threadgroup(0)]]) {
    const uint r0 = tgid * 32u;
    const uint n_kt = n_in >> 6u;
    const uint nsb = n_in >> 8u;

    threadgroup half* sa = (threadgroup half*)(shmem);             // 4 KiB
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);      // 2 KiB

    // Weight staging: thread covers 2 rows x 8 k.
    const uint wrg = tiitg >> 3u;         // 0..15 row pair group
    const uint wkt = tiitg & 7u;          // 0..7 k tile
    device const m16_block_q4_K* wrows =
        (device const m16_block_q4_K*)(W) + (r0 + wrg * 2u) * nsb;

    const uint am = tiitg >> 3u;
    const uint akt = tiitg & 7u;
    device const float4* xa = X + am * (n_in >> 2u);

    simdgroup_float8x8 mc[2];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 2u; ++i) mc[i] = simdgroup_float8x8(0);

    const uint rh = sgitg & 1u;           // 16-row halves
    const uint bh = sgitg >> 1u;

    for (uint kt = 0; kt < n_kt; ++kt) {
        {
            const uint kg = kt * 64u + wkt * 8u;
            const uint sb_i = kg >> 8u;
            const uint ks = kg & 255u;
            const uint g = ks >> 6u;
            const uint l = ks & 63u;
            const uint sub = 2u * g + (l >> 5u);
            const uint shift = (l & 32u) >> 3u;
            const uint boff = 32u * g + (l & 31u);
            half wv[2][8];
#pragma clang loop unroll(full)
            for (uint r = 0; r < 2u; ++r) {
                device const m16_block_q4_K* b = wrows + r * nsb + sb_i;
                float sc, mn;
                m16_q4k_scale_min(b->scales, sub, sc, mn);
                const float dl = float(b->d) * sc;
                const float ml = float(b->dmin) * mn;
                const uchar4 qa = *(device const uchar4*)(b->qs + boff);
                const uchar4 qb = *(device const uchar4*)(b->qs + boff + 4u);
#pragma clang loop unroll(full)
                for (uint i = 0; i < 4u; ++i) {
                    wv[r][i] = half(dl * float((qa[i] >> shift) & 15u) - ml);
                    wv[r][i + 4u] = half(dl * float((qb[i] >> shift) & 15u) - ml);
                }
            }
            threadgroup half* tile = sa + (4u * wkt + (wrg >> 2u)) * 64u + ((wrg & 3u) << 1u);
#pragma clang loop unroll(full)
            for (uint j = 0; j < 8u; ++j) {
                *(threadgroup half*)(tile + 8u * j) = wv[0][j];
                *(threadgroup half*)(tile + 8u * j + 1u) = wv[1][j];
            }
        }
        {
            const uint kg = kt * 64u + akt * 8u;
            const float4 a0 = xa[kg >> 2u];
            const float4 a1 = xa[(kg >> 2u) + 1u];
            threadgroup half* tile = sb + (2u * akt + (am >> 3u)) * 64u + 8u * (am & 7u);
            *(threadgroup half4*)(tile) = half4(a0);
            *(threadgroup half4*)(tile + 4) = half4(a1);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half* lsma = sa + rh * 2u * 64u;
        threadgroup const half* lsmb = sb + bh * 64u;
#pragma clang loop unroll(full)
        for (uint kk = 0; kk < 8u; ++kk) {
            simdgroup_half8x8 ma[2];
            simdgroup_half8x8 mb;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 2u; ++i)
                simdgroup_load(ma[i], lsma + (4u * kk + i) * 64u, 8);
            simdgroup_load(mb, lsmb + 2u * kk * 64u, 8);
#pragma clang loop unroll(full)
            for (uint i = 0; i < 2u; ++i)
                simdgroup_multiply_accumulate(mc[i], mb, ma[i], mc[i]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* C = Y + (bh * 8u) * n_out + r0 + rh * 16u;
#pragma clang loop unroll(full)
    for (uint i = 0; i < 2u; ++i)
        simdgroup_store(mc[i], C + 8u * i, n_out);
}

// ─── n32: NR0=32, NK=64, 6 KiB tgmem, 2 row tiles x 2 batch halves ─────────
// ─── n32 q6k / q5k variants ────────────────────────────────────────────────

kernel void m16_q6k_n32(device const uchar* W [[buffer(0)]],
                        device const float4* X [[buffer(1)]],
                        device float* Y [[buffer(2)]],
                        constant uint& n_in [[buffer(3)]],
                        constant uint& n_out [[buffer(4)]],
                        constant uint& B [[buffer(5)]],
                        uint tgid [[threadgroup_position_in_grid]],
                        uint sgitg [[simdgroup_index_in_threadgroup]],
                        uint tiitg [[thread_index_in_threadgroup]],
                        threadgroup char* shmem [[threadgroup(0)]]) {
    const uint r0 = tgid * 32u;
    const uint n_kt = n_in >> 6u;
    const uint nsb = n_in >> 8u;

    threadgroup half* sa = (threadgroup half*)(shmem);
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);

    const uint wrg = tiitg >> 3u;         // row pair group
    const uint wkt = tiitg & 7u;
    device const m16_block_q6_K* wrows =
        (device const m16_block_q6_K*)(W) + (r0 + wrg * 2u) * nsb;

    const uint am = tiitg >> 3u;
    const uint akt = tiitg & 7u;
    device const float4* xa = X + am * (n_in >> 2u);

    simdgroup_float8x8 mc[2];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 2u; ++i) mc[i] = simdgroup_float8x8(0);

    const uint rh = sgitg & 1u;
    const uint bh = sgitg >> 1u;

    for (uint kt = 0; kt < n_kt; ++kt) {
        {
            const uint kg = kt * 64u + wkt * 8u;
            const uint sb_i = kg >> 8u;
            const uint ks = kg & 255u;
            const uint g = ks >> 7u;
            const uint rem = ks & 127u;
            const uint st = rem >> 5u;
            const uint p = rem & 31u;
            const uint ql_base = 64u * g + (st & 1u) * 32u + p;
            const uint qh_base = 128u + 32u * g + p;
            const uint ql_shift = (st & 2u) << 1u;
            const uint qh_shift = 2u * st;
            const uint sc_idx = 8u * g + (p >> 4u) + 2u * st;
            half wv[2][8];
#pragma clang loop unroll(full)
            for (uint r = 0; r < 2u; ++r) {
                device const uchar* b = (device const uchar*)(wrows + r * nsb + sb_i);
                const float d = float(((device const half*)(b + 208))[0]);
                const float dsc = d * float(((device const char*)(b + 192))[sc_idx]);
#pragma clang loop unroll(full)
                for (uint i = 0; i < 8u; ++i) {
                    const uint ql = uint(b[ql_base + i]);
                    const uint qh = uint(b[qh_base + i]);
                    wv[r][i] = half(dsc * float(int(((ql >> ql_shift) & 15u) |
                                                    (((qh >> qh_shift) & 3u) << 4u)) - 32));
                }
            }
            threadgroup half* tile = sa + (4u * wkt + (wrg >> 2u)) * 64u + ((wrg & 3u) << 1u);
#pragma clang loop unroll(full)
            for (uint j = 0; j < 8u; ++j) {
                *(threadgroup half*)(tile + 8u * j) = wv[0][j];
                *(threadgroup half*)(tile + 8u * j + 1u) = wv[1][j];
            }
        }
        {
            const uint kg = kt * 64u + akt * 8u;
            const float4 a0 = xa[kg >> 2u];
            const float4 a1 = xa[(kg >> 2u) + 1u];
            threadgroup half* tile = sb + (2u * akt + (am >> 3u)) * 64u + 8u * (am & 7u);
            *(threadgroup half4*)(tile) = half4(a0);
            *(threadgroup half4*)(tile + 4) = half4(a1);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half* lsma = sa + rh * 2u * 64u;
        threadgroup const half* lsmb = sb + bh * 64u;
#pragma clang loop unroll(full)
        for (uint kk = 0; kk < 8u; ++kk) {
            simdgroup_half8x8 ma[2];
            simdgroup_half8x8 mb;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 2u; ++i)
                simdgroup_load(ma[i], lsma + (4u * kk + i) * 64u, 8);
            simdgroup_load(mb, lsmb + 2u * kk * 64u, 8);
#pragma clang loop unroll(full)
            for (uint i = 0; i < 2u; ++i)
                simdgroup_multiply_accumulate(mc[i], mb, ma[i], mc[i]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* C = Y + (bh * 8u) * n_out + r0 + rh * 16u;
#pragma clang loop unroll(full)
    for (uint i = 0; i < 2u; ++i)
        simdgroup_store(mc[i], C + 8u * i, n_out);
}

kernel void m16_q5k_n32(device const uchar* W [[buffer(0)]],
                        device const float4* X [[buffer(1)]],
                        device float* Y [[buffer(2)]],
                        constant uint& n_in [[buffer(3)]],
                        constant uint& n_out [[buffer(4)]],
                        constant uint& B [[buffer(5)]],
                        uint tgid [[threadgroup_position_in_grid]],
                        uint sgitg [[simdgroup_index_in_threadgroup]],
                        uint tiitg [[thread_index_in_threadgroup]],
                        threadgroup char* shmem [[threadgroup(0)]]) {
    const uint r0 = tgid * 32u;
    const uint n_kt = n_in >> 6u;
    const uint nsb = n_in >> 8u;

    threadgroup half* sa = (threadgroup half*)(shmem);
    threadgroup half* sb = (threadgroup half*)(shmem + 4096);

    const uint wrg = tiitg >> 3u;
    const uint wkt = tiitg & 7u;
    device const m16_block_q5_K* wrows =
        (device const m16_block_q5_K*)(W) + (r0 + wrg * 2u) * nsb;

    const uint am = tiitg >> 3u;
    const uint akt = tiitg & 7u;
    device const float4* xa = X + am * (n_in >> 2u);

    simdgroup_float8x8 mc[2];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 2u; ++i) mc[i] = simdgroup_float8x8(0);

    const uint rh = sgitg & 1u;
    const uint bh = sgitg >> 1u;

    for (uint kt = 0; kt < n_kt; ++kt) {
        {
            const uint kg = kt * 64u + wkt * 8u;
            const uint sb_i = kg >> 8u;
            const uint ks = kg & 255u;
            const uint g = ks >> 6u;
            const uint l = ks & 63u;
            const uint sub = 2u * g + (l >> 5u);
            const uint shift = (l & 32u) >> 3u;
            const uint bit = 2u * g + (l >> 5u);
            const uint boff = 32u * g + (l & 31u);
            half wv[2][8];
#pragma clang loop unroll(full)
            for (uint r = 0; r < 2u; ++r) {
                device const m16_block_q5_K* b = wrows + r * nsb + sb_i;
                float sc, mn;
                m16_q4k_scale_min(b->scales, sub, sc, mn);
                const float dl = float(b->d) * sc;
                const float ml = float(b->dmin) * mn;
                const uchar4 qa = *(device const uchar4*)(b->qs + boff);
                const uchar4 qb = *(device const uchar4*)(b->qs + boff + 4u);
#pragma clang loop unroll(full)
                for (uint i = 0; i < 4u; ++i) {
                    const uint hba = uint(b->qh[boff - 32u * g + i]);
                    const uint hbb = uint(b->qh[boff - 32u * g + 4u + i]);
                    const uint la = (qa[i] >> shift) & 15u;
                    const uint lb = (qb[i] >> shift) & 15u;
                    wv[r][i] = half(dl * float(la | (((hba >> bit) & 1u) << 4u)) - ml);
                    wv[r][i + 4u] = half(dl * float(lb | (((hbb >> bit) & 1u) << 4u)) - ml);
                }
            }
            threadgroup half* tile = sa + (4u * wkt + (wrg >> 2u)) * 64u + ((wrg & 3u) << 1u);
#pragma clang loop unroll(full)
            for (uint j = 0; j < 8u; ++j) {
                *(threadgroup half*)(tile + 8u * j) = wv[0][j];
                *(threadgroup half*)(tile + 8u * j + 1u) = wv[1][j];
            }
        }
        {
            const uint kg = kt * 64u + akt * 8u;
            const float4 a0 = xa[kg >> 2u];
            const float4 a1 = xa[(kg >> 2u) + 1u];
            threadgroup half* tile = sb + (2u * akt + (am >> 3u)) * 64u + 8u * (am & 7u);
            *(threadgroup half4*)(tile) = half4(a0);
            *(threadgroup half4*)(tile + 4) = half4(a1);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half* lsma = sa + rh * 2u * 64u;
        threadgroup const half* lsmb = sb + bh * 64u;
#pragma clang loop unroll(full)
        for (uint kk = 0; kk < 8u; ++kk) {
            simdgroup_half8x8 ma[2];
            simdgroup_half8x8 mb;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 2u; ++i)
                simdgroup_load(ma[i], lsma + (4u * kk + i) * 64u, 8);
            simdgroup_load(mb, lsmb + 2u * kk * 64u, 8);
#pragma clang loop unroll(full)
            for (uint i = 0; i < 2u; ++i)
                simdgroup_multiply_accumulate(mc[i], mb, ma[i], mc[i]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float* C = Y + (bh * 8u) * n_out + r0 + rh * 16u;
#pragma clang loop unroll(full)
    for (uint i = 0; i < 2u; ++i)
        simdgroup_store(mc[i], C + 8u * i, n_out);
}
