// Multi-column quantized matvec for the DFlash verify batch.
//
// Structure is ported from llama.cpp's `kernel_mul_mv_ext_*` idea (one weight
// block load feeding several activation columns) applied to the pinned
// `kernel_mul_mv_{q4_K,q5_K,q6_K}_f32` bodies. Every scalar operation that
// contributes to an output element keeps the pinned kernel's order: the
// per-thread block walk is unchanged, the per-(column,row) accumulator is a
// single sequential f32 reduction over the same block sequence, and the
// cross-lane reduction is the same `simd_sum`. Only the weight loads move out
// of the column loop, so column `c` of an NC-wide dispatch performs exactly
// the arithmetic the single-column kernel performs for token `c`. Q4_K and
// Q5_K reproduce the pinned metallib bitwise; Q6_K lands a few ULP away
// because that body is contracted differently by the pinned compilation.
//
// Constraints: ne00 % 256 == 0, NSG == 2 and NR0 == 2 (must match the pinned
// dispatch shape), destination row stride == ne01.

#include <metal_stdlib>
using namespace metal;

#define MUSER_MC_QK_K 256
#define MUSER_MC_NSG  2
#define MUSER_MC_NR0  2

struct muser_multicol_args {
    uint ne00;  // columns of the weight matrix (n_in)
    uint ne01;  // rows of the weight matrix (n_out), also the dst row stride
    uint nb01;  // weight row stride in bytes
    uint col0;  // first activation column handled by this dispatch
};

typedef struct {
    half d;
    half dmin;
    uchar scales[12];
    uchar qs[128];
} muser_mc_block_q4_K;

typedef struct {
    half d;
    half dmin;
    uchar scales[12];
    uchar qh[32];
    uchar qs[128];
} muser_mc_block_q5_K;

typedef struct {
    uchar ql[128];
    uchar qh[64];
    char scales[16];
    half d;
} muser_mc_block_q6_K;

static_assert(sizeof(muser_mc_block_q4_K) == 144, "q4_K block layout");
static_assert(sizeof(muser_mc_block_q5_K) == 176, "q5_K block layout");
static_assert(sizeof(muser_mc_block_q6_K) == 210, "q6_K block layout");

// The trailing rows of the last threadgroup may fall outside the matrix. They
// re-read the last valid row instead of running off the weight allocation;
// their results are never stored.
inline uint muser_mc_row_clamp(int row, uint rows) {
    return min(uint(max(row, 0)), rows - 1);
}

template <short NC>
inline void muser_multicol_q4k_impl(
    device const uchar *weights,
    device const float *input,
    device float *output,
    constant muser_multicol_args &args,
    uint3 tgpig,
    ushort tiisg,
    ushort sgitg) {
    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    const int nb = int(args.ne00) / MUSER_MC_QK_K;
    const int first_row = (int(tgpig.x) * MUSER_MC_NSG + int(sgitg)) * MUSER_MC_NR0;

    float sumf[NC][MUSER_MC_NR0];
    for (short c = 0; c < NC; ++c) {
        for (short row = 0; row < MUSER_MC_NR0; ++row) {
            sumf[c][row] = 0.0f;
        }
    }

    device const muser_mc_block_q4_K *x[MUSER_MC_NR0];
    for (short row = 0; row < MUSER_MC_NR0; ++row) {
        const ulong offset =
            ulong(muser_mc_row_clamp(first_row + row, args.ne01)) * ulong(args.nb01);
        x[row] = (device const muser_mc_block_q4_K *)(weights + offset);
    }

    for (int ib = ix; ib < nb; ib += 4) {
        // One block load per row, reused by every activation column below.
        ushort4 wq1[MUSER_MC_NR0];
        ushort4 wq2[MUSER_MC_NR0];
        ushort wsc[MUSER_MC_NR0][3];
        float2 wd[MUSER_MC_NR0];
        for (short row = 0; row < MUSER_MC_NR0; ++row) {
            device const ushort *sc = (device const ushort *)x[row][ib].scales + iq;
            device const ushort *q1 = (device const ushort *)x[row][ib].qs + 16 * iq + 4 * ir;
            wq1[row] = ushort4(q1[0], q1[1], q1[2], q1[3]);
            wq2[row] = ushort4(q1[32], q1[33], q1[34], q1[35]);
            wsc[row][0] = sc[0];
            wsc[row][1] = sc[2];
            wsc[row][2] = sc[4];
            wd[row] = float2(float(x[row][ib].d), float(x[row][ib].dmin));
        }

        for (short c = 0; c < NC; ++c) {
            device const float *y4 = input + ulong(args.col0 + uint(c)) * ulong(args.ne00) +
                                     ib * MUSER_MC_QK_K + 64 * iq + 8 * ir;
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            for (short i = 0; i < 8; ++i) {
                yl[i + 0] = y4[i + 0];
                sumy[0] += yl[i + 0];
                yl[i + 8] = y4[i + 32];
                sumy[1] += yl[i + 8];
                yh[i + 0] = y4[i + 128];
                sumy[2] += yh[i + 0];
                yh[i + 8] = y4[i + 160];
                sumy[3] += yh[i + 8];
            }

            for (short row = 0; row < MUSER_MC_NR0; ++row) {
                ushort sc16[4];
                thread const uchar *sc8 = (thread const uchar *)sc16;
                sc16[0] = wsc[row][0] & kmask1;
                sc16[1] = wsc[row][1] & kmask1;
                sc16[2] = ((wsc[row][2] >> 0) & kmask2) | ((wsc[row][0] & kmask3) >> 2);
                sc16[3] = ((wsc[row][2] >> 4) & kmask2) | ((wsc[row][1] & kmask3) >> 2);

                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                for (short i = 0; i < 4; ++i) {
                    acc1[0] += yl[2 * i + 0] * (wq1[row][i] & 0x000F);
                    acc1[1] += yl[2 * i + 1] * (wq1[row][i] & 0x0F00);
                    acc1[2] += yl[2 * i + 8] * (wq1[row][i] & 0x00F0);
                    acc1[3] += yl[2 * i + 9] * (wq1[row][i] & 0xF000);
                    acc2[0] += yh[2 * i + 0] * (wq2[row][i] & 0x000F);
                    acc2[1] += yh[2 * i + 1] * (wq2[row][i] & 0x0F00);
                    acc2[2] += yh[2 * i + 8] * (wq2[row][i] & 0x00F0);
                    acc2[3] += yh[2 * i + 9] * (wq2[row][i] & 0xF000);
                }

                sumf[c][row] += wd[row][0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                              (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                              (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                              (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                                 wd[row][1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] +
                                               sumy[3] * sc8[7]);
            }
        }
    }

    for (short c = 0; c < NC; ++c) {
        device float *dst = output + ulong(args.col0 + uint(c)) * ulong(args.ne01);
        for (short row = 0; row < MUSER_MC_NR0; ++row) {
            const float total = simd_sum(sumf[c][row]);
            if (tiisg == 0 && uint(first_row + row) < args.ne01) {
                dst[first_row + row] = total;
            }
        }
    }
}

template <short NC>
inline void muser_multicol_q5k_impl(
    device const uchar *weights,
    device const float *input,
    device float *output,
    constant muser_multicol_args &args,
    uint3 tgpig,
    ushort tiisg,
    ushort sgitg) {
    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    const short tid = tiisg / 4;
    const short ix = tiisg % 4;
    const short iq = tid / 4;
    const short ir = tid % 4;

    const short l0 = 8 * ir;
    const short q_offset = 32 * iq + l0;
    const short y_offset = 64 * iq + l0;

    const uchar hm1 = 1u << (2 * iq);
    const uchar hm2 = hm1 << 1;
    const uchar hm3 = hm1 << 4;
    const uchar hm4 = hm2 << 4;

    const int nb = int(args.ne00) / MUSER_MC_QK_K;
    const int first_row = (int(tgpig.x) * MUSER_MC_NSG + int(sgitg)) * MUSER_MC_NR0;

    float sumf[NC][MUSER_MC_NR0];
    for (short c = 0; c < NC; ++c) {
        for (short row = 0; row < MUSER_MC_NR0; ++row) {
            sumf[c][row] = 0.0f;
        }
    }

    device const muser_mc_block_q5_K *x[MUSER_MC_NR0];
    for (short row = 0; row < MUSER_MC_NR0; ++row) {
        const ulong offset =
            ulong(muser_mc_row_clamp(first_row + row, args.ne01)) * ulong(args.nb01);
        x[row] = (device const muser_mc_block_q5_K *)(weights + offset);
    }

    for (int i = ix; i < nb; i += 4) {
        uchar wq1[MUSER_MC_NR0][8];
        uchar wq2[MUSER_MC_NR0][8];
        uchar wqh[MUSER_MC_NR0][8];
        ushort wa[MUSER_MC_NR0][3];
        float2 wd[MUSER_MC_NR0];
        for (short row = 0; row < MUSER_MC_NR0; ++row) {
            device const uchar *q1 = x[row][i].qs + q_offset;
            device const uchar *qh = x[row][i].qh + l0;
            device const ushort *a = (device const ushort *)x[row][i].scales + iq;
            for (short l = 0; l < 8; ++l) {
                wq1[row][l] = q1[l];
                wq2[row][l] = q1[l + 64];
                wqh[row][l] = qh[l];
            }
            wa[row][0] = a[0];
            wa[row][1] = a[2];
            wa[row][2] = a[4];
            wd[row] = float2(float(x[row][i].d), float(x[row][i].dmin));
        }

        for (short c = 0; c < NC; ++c) {
            device const float *y1 = input + ulong(args.col0 + uint(c)) * ulong(args.ne00) +
                                     i * MUSER_MC_QK_K + y_offset;
            device const float *y2 = y1 + 128;
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            for (short l = 0; l < 8; ++l) {
                yl[l + 0] = y1[l + 0];
                sumy[0] += yl[l + 0];
                yl[l + 8] = y1[l + 32];
                sumy[1] += yl[l + 8];
                yh[l + 0] = y2[l + 0];
                sumy[2] += yh[l + 0];
                yh[l + 8] = y2[l + 32];
                sumy[3] += yh[l + 8];
            }

            for (short row = 0; row < MUSER_MC_NR0; ++row) {
                ushort sc16[4];
                thread const uchar *sc8 = (thread const uchar *)sc16;
                sc16[0] = wa[row][0] & kmask1;
                sc16[1] = wa[row][1] & kmask1;
                sc16[2] = ((wa[row][2] >> 0) & kmask2) | ((wa[row][0] & kmask3) >> 2);
                sc16[3] = ((wa[row][2] >> 4) & kmask2) | ((wa[row][1] & kmask3) >> 2);

                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                for (short l = 0; l < 8; ++l) {
                    const uchar h = wqh[row][l];
                    acc1[0] += yl[l + 0] * (wq1[row][l] & 0x0F);
                    acc1[1] += yl[l + 8] * (wq1[row][l] & 0xF0);
                    acc1[2] += yh[l + 0] * (wq2[row][l] & 0x0F);
                    acc1[3] += yh[l + 8] * (wq2[row][l] & 0xF0);
                    acc2[0] += h & hm1 ? yl[l + 0] : 0.f;
                    acc2[1] += h & hm2 ? yl[l + 8] : 0.f;
                    acc2[2] += h & hm3 ? yh[l + 0] : 0.f;
                    acc2[3] += h & hm4 ? yh[l + 8] : 0.f;
                }

                sumf[c][row] += wd[row][0] * (sc8[0] * (acc1[0] + 16.f * acc2[0]) +
                                              sc8[1] * (acc1[1] / 16.f + 16.f * acc2[1]) +
                                              sc8[4] * (acc1[2] + 16.f * acc2[2]) +
                                              sc8[5] * (acc1[3] / 16.f + 16.f * acc2[3])) -
                                 wd[row][1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] +
                                               sumy[3] * sc8[7]);
            }
        }
    }

    for (short c = 0; c < NC; ++c) {
        device float *dst = output + ulong(args.col0 + uint(c)) * ulong(args.ne01);
        for (short row = 0; row < MUSER_MC_NR0; ++row) {
            const float total = simd_sum(sumf[c][row]);
            if (tiisg == 0 && uint(first_row + row) < args.ne01) {
                dst[first_row + row] = total;
            }
        }
    }
}

template <short NC>
inline void muser_multicol_q6k_impl(
    device const uchar *weights,
    device const float *input,
    device float *output,
    constant muser_multicol_args &args,
    uint3 tgpig,
    ushort tiisg,
    ushort sgitg) {
    constexpr uchar kmask1 = 0x03;
    constexpr uchar kmask2 = 0x0C;
    constexpr uchar kmask3 = 0x30;
    constexpr uchar kmask4 = 0xC0;

    const short tid = tiisg / 2;
    const short ix = tiisg % 2;
    const short ip = tid / 8;
    const short il = tid % 8;
    const short l0 = 4 * il;
    const short is = 8 * ip + l0 / 16;

    const short y_offset = 128 * ip + l0;
    const short q_offset_l = 64 * ip + l0;
    const short q_offset_h = 32 * ip + l0;

    const int nb = int(args.ne00) / MUSER_MC_QK_K;
    const int first_row = (int(tgpig.x) * MUSER_MC_NSG + int(sgitg)) * MUSER_MC_NR0;

    float sumf[NC][MUSER_MC_NR0];
    for (short c = 0; c < NC; ++c) {
        for (short row = 0; row < MUSER_MC_NR0; ++row) {
            sumf[c][row] = 0.0f;
        }
    }

    device const muser_mc_block_q6_K *x[MUSER_MC_NR0];
    for (short row = 0; row < MUSER_MC_NR0; ++row) {
        const ulong offset =
            ulong(muser_mc_row_clamp(first_row + row, args.ne01)) * ulong(args.nb01);
        x[row] = (device const muser_mc_block_q6_K *)(weights + offset);
    }

    for (int i = ix; i < nb; i += 2) {
        uchar4 wq1[MUSER_MC_NR0];
        uchar4 wq2[MUSER_MC_NR0];
        uchar4 wqh[MUSER_MC_NR0];
        char4 wsc[MUSER_MC_NR0];
        float wd[MUSER_MC_NR0];
        for (short row = 0; row < MUSER_MC_NR0; ++row) {
            device const uchar *q1 = x[row][i].ql + q_offset_l;
            device const uchar *qh = x[row][i].qh + q_offset_h;
            device const char *sc = x[row][i].scales + is;
            wq1[row] = uchar4(q1[0], q1[1], q1[2], q1[3]);
            wq2[row] = uchar4(q1[32], q1[33], q1[34], q1[35]);
            wqh[row] = uchar4(qh[0], qh[1], qh[2], qh[3]);
            wsc[row] = char4(sc[0], sc[2], sc[4], sc[6]);
            wd[row] = float(x[row][i].d);
        }

        for (short c = 0; c < NC; ++c) {
            device const float *y = input + ulong(args.col0 + uint(c)) * ulong(args.ne00) +
                                    i * MUSER_MC_QK_K + y_offset;
            float yl[16];
            for (short l = 0; l < 4; ++l) {
                yl[4 * l + 0] = y[l + 0];
                yl[4 * l + 1] = y[l + 32];
                yl[4 * l + 2] = y[l + 64];
                yl[4 * l + 3] = y[l + 96];
            }

            for (short row = 0; row < MUSER_MC_NR0; ++row) {
                float4 sums = {0.f, 0.f, 0.f, 0.f};
                for (short l = 0; l < 4; ++l) {
                    const uchar h = wqh[row][l];
                    sums[0] += yl[4 * l + 0] * (char((wq1[row][l] & 0xF) | ((h & kmask1) << 4)) - 32);
                    sums[1] += yl[4 * l + 1] * (char((wq2[row][l] & 0xF) | ((h & kmask2) << 2)) - 32);
                    sums[2] += yl[4 * l + 2] * (char((wq1[row][l] >> 4) | ((h & kmask3) << 0)) - 32);
                    sums[3] += yl[4 * l + 3] * (char((wq2[row][l] >> 4) | ((h & kmask4) >> 2)) - 32);
                }

                sumf[c][row] += wd[row] * (sums[0] * wsc[row][0] + sums[1] * wsc[row][1] +
                                           sums[2] * wsc[row][2] + sums[3] * wsc[row][3]);
            }
        }
    }

    for (short c = 0; c < NC; ++c) {
        device float *dst = output + ulong(args.col0 + uint(c)) * ulong(args.ne01);
        for (short row = 0; row < MUSER_MC_NR0; ++row) {
            const float total = simd_sum(sumf[c][row]);
            if (tiisg == 0 && uint(first_row + row) < args.ne01) {
                dst[first_row + row] = total;
            }
        }
    }
}

#define MUSER_MULTICOL_KERNEL(dtype, nc)                                       \
    kernel void muser_matvec_multicol_##dtype##_c##nc(                         \
        device const uchar *weights [[buffer(0)]],                             \
        device const float *input [[buffer(1)]],                               \
        device float *output [[buffer(2)]],                                    \
        constant muser_multicol_args &args [[buffer(3)]],                      \
        uint3 tgpig [[threadgroup_position_in_grid]],                          \
        ushort tiisg [[thread_index_in_simdgroup]],                            \
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {                     \
        muser_multicol_##dtype##_impl<nc>(                                     \
            weights, input, output, args, tgpig, tiisg, sgitg);                \
    }

MUSER_MULTICOL_KERNEL(q4k, 1)
MUSER_MULTICOL_KERNEL(q4k, 2)
MUSER_MULTICOL_KERNEL(q4k, 4)

MUSER_MULTICOL_KERNEL(q5k, 1)
MUSER_MULTICOL_KERNEL(q5k, 2)
MUSER_MULTICOL_KERNEL(q5k, 4)

MUSER_MULTICOL_KERNEL(q6k, 1)
MUSER_MULTICOL_KERNEL(q6k, 2)
MUSER_MULTICOL_KERNEL(q6k, 4)
