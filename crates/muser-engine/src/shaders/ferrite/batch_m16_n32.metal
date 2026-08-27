// ─── M=16 K-quant batch GEMM (DFlash verify/draft, L series) ──────────────
//
// Purpose-built small-batch tile for the 16-row speculative verify and draft
// block forwards. One threadgroup covers 32 output rows x all 16 activation
// rows with a 64-wide K stage: weight-stationary, dequantized to half into
// 8x8 simdgroup tiles, 4 simdgroups of (row-half, batch-half) each. Threadgroup
// footprint is 6 KiB, so many threadgroups stay resident per core -- the
// retained n_out/64 K-serial SGM tile was occupancy-bound at these shapes.
//
// Arithmetic family: identical to the accepted half-staged SGM tile
// (Q4_K/Q5_K/Q6_K dequant to half, half8x8 tensor MAC into f32 accumulators).
// Gated by lossless token equality per the Stage B contract.
//
// Preconditions (enforced by the Rust dispatcher):
//   B == 16, n_in % 256 == 0, n_out % 32 == 0.

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

// Tile memory layout (shared by all three dtypes):
//   sa (weights):     tile (rt, kt) at (4*kt + rt)*64, element (k,r) at 8*k+r
//   sb (activations): tile (bt, kt) at (2*kt + bt)*64, element (m,k) at 8*m+k
// MAC convention: mc[m][r] += sum_k mb[m][k] * ma[k][r], stored token-major.

#define M16_N32_MAC_STORE()                                                       threadgroup const half* lsma = sa + rh * 2u * 64u;                            threadgroup const half* lsmb = sb + bh * 64u;                                 _Pragma("clang loop unroll(full)")                                            for (uint kk = 0; kk < 8u; ++kk) {                                                simdgroup_half8x8 ma[2];                                                      simdgroup_half8x8 mb;                                                         _Pragma("clang loop unroll(full)")                                            for (uint i = 0; i < 2u; ++i)                                                     simdgroup_load(ma[i], lsma + (4u * kk + i) * 64u, 8);                     simdgroup_load(mb, lsmb + 2u * kk * 64u, 8);                                  _Pragma("clang loop unroll(full)")                                            for (uint i = 0; i < 2u; ++i)                                                     simdgroup_multiply_accumulate(mc[i], mb, ma[i], mc[i]);               }

#define M16_N32_STAGE_X()                                                         {                                                                                 const uint kg = kt * 64u + akt * 8u;                                          const float4 a0 = xa[kg >> 2u];                                               const float4 a1 = xa[(kg >> 2u) + 1u];                                        threadgroup half* tile =                                                          sb + (2u * akt + (am >> 3u)) * 64u + 8u * (am & 7u);                      *(threadgroup half4*)(tile) = half4(a0);                                      *(threadgroup half4*)(tile + 4) = half4(a1);                              }

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
