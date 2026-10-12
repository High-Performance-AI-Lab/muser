// Native NVFP4 projection kernels.
//
// Each matrix is split into packed E2M1 bytes (two values/byte), raw E4M3FN
// scales (one/16 values), and one f32 scale2. Arithmetic order matches the
// CPU/ModelOpt oracle: `(e2m1 * e4m3fn) * scale2` before the activation dot.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct muser_nvfp4_args {
    uint n_in;
    uint n_out;
    uint col0;
};

inline float muser_e2m1(uchar nibble) {
    constexpr float lut[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    return lut[nibble & 0x0f];
}

// Structural exact-lane encodings. Every E2M1 value is an integer in units
// of 2^-1 and every finite E4M3FN value is an integer in units of 2^-9.
// Their products therefore share one 2^-20 denominator across all blocks.
inline int muser_e2m1_q1(uchar nibble) {
    const uchar code = nibble & 7;
    const int magnitude = code <= 4 ? int(code)
        : (code == 5 ? 6 : (code == 6 ? 8 : 12));
    return (nibble & 8) == 0 ? magnitude : -magnitude;
}

inline int muser_e4m3_q9(uchar byte) {
    const uchar exponent = (byte >> 3) & 15;
    const uchar mantissa = byte & 7;
    const int magnitude = exponent == 0
        ? int(mantissa)
        : int(8 + mantissa) << (exponent - 1);
    return (byte & 0x80) == 0 ? magnitude : -magnitude;
}

inline long muser_simd_sum_i64(long value, ushort lane) {
    const ulong bits = as_type<ulong>(value);
    uint low = uint(bits);
    uint high = uint(bits >> 32);
    for (ushort distance = 16; distance != 0; distance >>= 1) {
        const uint other_low = simd_shuffle_down(low, distance);
        const uint other_high = simd_shuffle_down(high, distance);
        if (lane + distance < 32) {
            const uint new_low = low + other_low;
            const uint carry = new_low < low ? 1 : 0;
            low = new_low;
            high = high + other_high + carry;
        }
    }
    return as_type<long>((ulong(high) << 32) | ulong(low));
}

constant float MUSER_E4M3FN[256] = {
    0.0f, 0.001953125f, 0.00390625f, 0.005859375f, 0.0078125f, 0.009765625f, 0.01171875f, 0.013671875f,
    0.015625f, 0.017578125f, 0.01953125f, 0.021484375f, 0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
    0.03125f, 0.03515625f, 0.0390625f, 0.04296875f, 0.046875f, 0.05078125f, 0.0546875f, 0.05859375f,
    0.0625f, 0.0703125f, 0.078125f, 0.0859375f, 0.09375f, 0.1015625f, 0.109375f, 0.1171875f,
    0.125f, 0.140625f, 0.15625f, 0.171875f, 0.1875f, 0.203125f, 0.21875f, 0.234375f,
    0.25f, 0.28125f, 0.3125f, 0.34375f, 0.375f, 0.40625f, 0.4375f, 0.46875f,
    0.5f, 0.5625f, 0.625f, 0.6875f, 0.75f, 0.8125f, 0.875f, 0.9375f,
    1.0f, 1.125f, 1.25f, 1.375f, 1.5f, 1.625f, 1.75f, 1.875f,
    2.0f, 2.25f, 2.5f, 2.75f, 3.0f, 3.25f, 3.5f, 3.75f,
    4.0f, 4.5f, 5.0f, 5.5f, 6.0f, 6.5f, 7.0f, 7.5f,
    8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
    16.0f, 18.0f, 20.0f, 22.0f, 24.0f, 26.0f, 28.0f, 30.0f,
    32.0f, 36.0f, 40.0f, 44.0f, 48.0f, 52.0f, 56.0f, 60.0f,
    64.0f, 72.0f, 80.0f, 88.0f, 96.0f, 104.0f, 112.0f, 120.0f,
    128.0f, 144.0f, 160.0f, 176.0f, 192.0f, 208.0f, 224.0f, 240.0f,
    256.0f, 288.0f, 320.0f, 352.0f, 384.0f, 416.0f, 448.0f, 0.0f,
    -0.0f, -0.001953125f, -0.00390625f, -0.005859375f, -0.0078125f, -0.009765625f, -0.01171875f, -0.013671875f,
    -0.015625f, -0.017578125f, -0.01953125f, -0.021484375f, -0.0234375f, -0.025390625f, -0.02734375f, -0.029296875f,
    -0.03125f, -0.03515625f, -0.0390625f, -0.04296875f, -0.046875f, -0.05078125f, -0.0546875f, -0.05859375f,
    -0.0625f, -0.0703125f, -0.078125f, -0.0859375f, -0.09375f, -0.1015625f, -0.109375f, -0.1171875f,
    -0.125f, -0.140625f, -0.15625f, -0.171875f, -0.1875f, -0.203125f, -0.21875f, -0.234375f,
    -0.25f, -0.28125f, -0.3125f, -0.34375f, -0.375f, -0.40625f, -0.4375f, -0.46875f,
    -0.5f, -0.5625f, -0.625f, -0.6875f, -0.75f, -0.8125f, -0.875f, -0.9375f,
    -1.0f, -1.125f, -1.25f, -1.375f, -1.5f, -1.625f, -1.75f, -1.875f,
    -2.0f, -2.25f, -2.5f, -2.75f, -3.0f, -3.25f, -3.5f, -3.75f,
    -4.0f, -4.5f, -5.0f, -5.5f, -6.0f, -6.5f, -7.0f, -7.5f,
    -8.0f, -9.0f, -10.0f, -11.0f, -12.0f, -13.0f, -14.0f, -15.0f,
    -16.0f, -18.0f, -20.0f, -22.0f, -24.0f, -26.0f, -28.0f, -30.0f,
    -32.0f, -36.0f, -40.0f, -44.0f, -48.0f, -52.0f, -56.0f, -60.0f,
    -64.0f, -72.0f, -80.0f, -88.0f, -96.0f, -104.0f, -112.0f, -120.0f,
    -128.0f, -144.0f, -160.0f, -176.0f, -192.0f, -208.0f, -224.0f, -240.0f,
    -256.0f, -288.0f, -320.0f, -352.0f, -384.0f, -416.0f, -448.0f, 0.0f,
};

inline float muser_e4m3fn(uchar byte) {
    return MUSER_E4M3FN[byte];
}

inline uchar muser_e4m3fn_round_positive(float value) {
    value = min(value, 448.0f);
    uint low = 0;
    uint high = 126;
    while (low + 1 < high) {
        const uint middle = (low + high) >> 1;
        if (MUSER_E4M3FN[middle] <= value) {
            low = middle;
        } else {
            high = middle;
        }
    }
    const float low_distance = value - MUSER_E4M3FN[low];
    const float high_distance = MUSER_E4M3FN[high] - value;
    if (high_distance < low_distance ||
        (high_distance == low_distance && (high & 1u) == 0u && (low & 1u) != 0u)) {
        return uchar(high);
    }
    return uchar(low);
}

inline uchar muser_e2m1_round(float value) {
    const uchar sign = signbit(value) ? 8 : 0;
    const float magnitude = abs(value);
    const uchar code = magnitude <= 0.25f ? 0
        : magnitude < 0.75f ? 1
        : magnitude <= 1.25f ? 2
        : magnitude < 1.75f ? 3
        : magnitude <= 2.5f ? 4
        : magnitude < 3.5f ? 5
        : magnitude <= 5.0f ? 6
        : 7;
    return sign | code;
}

// Cross-vendor W4A4 activation code without a reciprocal. Both producer and
// consumer round the numerator multiplication once, then compare it against
// exactly representable E2M1 midpoints multiplied by the decoded E4M3 scale.
inline uchar muser_e2m1_round_ratio(
    float value,
    float numerator_scale,
    float denominator_scale) {
    const uchar sign = signbit(value) ? 8 : 0;
    const float magnitude = fma(abs(value), numerator_scale, 0.0f);
    const uchar code = magnitude <= denominator_scale * 0.25f ? 0
        : magnitude < denominator_scale * 0.75f ? 1
        : magnitude <= denominator_scale * 1.25f ? 2
        : magnitude < denominator_scale * 1.75f ? 3
        : magnitude <= denominator_scale * 2.5f ? 4
        : magnitude < denominator_scale * 3.5f ? 5
        : magnitude <= denominator_scale * 5.0f ? 6
        : 7;
    return sign | code;
}

// Decode eight packed E2M1 values with the same half-bit embedding used by
// MLX's native NVFP4 kernels. The embedded halves are E2M1 * 2^-14 exactly;
// the caller folds 2^14 into the block scale after accumulating one complete
// 16-value block.
inline void muser_nvfp4_accumulate_codes8(
    device const uchar *packed,
    device const float *x,
    thread float &sum) {
    const uint code = as_type<uint>(uchar4(
        *reinterpret_cast<device const packed_uchar4 *>(packed)));
    const uint even = code & 0x0f0f0f0fu;
    const uint even_signed = even | (even << 3);
    const uint odd = code & 0xf0f0f0f0u;
    const uint odd_signed = odd | (odd >> 3);
    float2 v04 = float2(as_type<half2>((even_signed << 9) & 0x8e008e00u));
    float2 v15 = float2(as_type<half2>((odd_signed << 8) & 0x8e008e00u));
    float2 v26 = float2(as_type<half2>((even_signed << 1) & 0x8e008e00u));
    float2 v37 = float2(as_type<half2>(odd_signed & 0x8e008e00u));
    sum += v04.x * x[0];
    sum += v15.x * x[1];
    sum += v26.x * x[2];
    sum += v37.x * x[3];
    sum += v04.y * x[4];
    sum += v15.y * x[5];
    sum += v26.y * x[6];
    sum += v37.y * x[7];
}

template <ushort NC>
inline void muser_nvfp4_matvec_impl(
    device const uchar *packed,
    device const uchar *scales,
    device const float *input,
    device float *output,
    constant muser_nvfp4_args &args,
    constant float &scale2,
    uint row,
    ushort lane) {
    if (row >= args.n_out) {
        return;
    }
    const uint packed_row = row * (args.n_in / 2);
    const uint scale_row = row * (args.n_in / 16);
    float sums[NC];
    for (ushort column = 0; column < NC; ++column) {
        sums[column] = 0.0f;
    }
    for (uint group = uint(lane); group < args.n_in / 16; group += 32) {
        const float block_scale = muser_e4m3fn(scales[scale_row + group]);
        const uint packed_base = packed_row + group * 8;
        const uint element_base = group * 16;
        for (ushort column = 0; column < NC; ++column) {
            device const float *x = input + (args.col0 + uint(column)) * args.n_in;
            float block_sum = 0.0f;
            muser_nvfp4_accumulate_codes8(
                packed + packed_base, x + element_base, block_sum);
            muser_nvfp4_accumulate_codes8(
                packed + packed_base + 4, x + element_base + 8, block_sum);
            const float scaled = block_sum * (block_scale * 16384.0f);
            sums[column] += scaled;
        }
    }
    for (ushort column = 0; column < NC; ++column) {
        const float total = simd_sum(sums[column]) * scale2;
        if (lane == 0) {
            output[(args.col0 + uint(column)) * args.n_out + row] = total;
        }
    }
}

#define MUSER_NVFP4_ENTRY(NC) \
kernel void muser_nvfp4_matvec_c##NC( \
    device const uchar *packed [[buffer(0)]], \
    device const uchar *scales [[buffer(1)]], \
    device const float *input [[buffer(2)]], \
    device float *output [[buffer(3)]], \
    constant muser_nvfp4_args &args [[buffer(4)]], \
    constant float &scale2 [[buffer(5)]], \
    uint row [[threadgroup_position_in_grid]], \
    ushort lane [[thread_index_in_simdgroup]]) { \
    muser_nvfp4_matvec_impl<NC>(packed, scales, input, output, args, scale2, row, lane); \
}

MUSER_NVFP4_ENTRY(1)
MUSER_NVFP4_ENTRY(2)
MUSER_NVFP4_ENTRY(4)
MUSER_NVFP4_ENTRY(8)
MUSER_NVFP4_ENTRY(16)

template <ushort NC>
inline void muser_nvfp4_w4a4_matvec_impl(
    device const uchar *packed,
    device const uchar *scales,
    device const float *input,
    device float *output,
    constant muser_nvfp4_args &args,
    constant float &weight_scale2,
    constant float &input_scale_inv,
    uint row,
    ushort lane) {
    if (row >= args.n_out) {
        return;
    }
    const uint packed_row = row * (args.n_in / 2);
    const uint scale_row = row * (args.n_in / 16);
    long sums[NC];
    for (ushort column = 0; column < NC; ++column) {
        sums[column] = 0;
    }
    for (uint group = uint(lane); group < args.n_in / 16; group += 32) {
        const int weight_scale = muser_e4m3_q9(scales[scale_row + group]);
        const uint packed_base = packed_row + group * 8;
        const uint element_base = group * 16;
        for (ushort column = 0; column < NC; ++column) {
            device const float *source =
                input + (args.col0 + uint(column)) * args.n_in + element_base;
            float rounded[16];
            float abs_max = 0.0f;
            for (ushort element = 0; element < 16; ++element) {
                rounded[element] = float(half(source[element]));
                abs_max = max(abs_max, abs(rounded[element]));
            }
            const float normalized_max =
                fma(abs_max, 0x1.555556p-3f, 0.0f);
            const uchar activation_scale_code = muser_e4m3fn_round_positive(
                min(fma(input_scale_inv, normalized_max, 0.0f), 448.0f));
            const float activation_scale = muser_e4m3fn(activation_scale_code);
            const bool activation_scale_zero = (activation_scale_code & 0x7f) == 0;
            int block_sum = 0;
            for (ushort element = 0; element < 16; ++element) {
                const uchar byte = packed[packed_base + uint(element / 2)];
                const uchar weight_code = (element & 1) == 0
                    ? byte & 0x0f
                    : byte >> 4;
                const uchar activation_code = activation_scale_zero
                    ? (signbit(rounded[element]) ? 8 : 0)
                    : muser_e2m1_round_ratio(
                        rounded[element], input_scale_inv, activation_scale);
                block_sum += muser_e2m1_q1(weight_code) * muser_e2m1_q1(activation_code);
            }
            sums[column] += long(block_sum) * long(weight_scale)
                * long(muser_e4m3_q9(activation_scale_code));
        }
    }
    for (ushort column = 0; column < NC; ++column) {
        const long integer_total = muser_simd_sum_i64(sums[column], lane);
        float scaled = float(integer_total) * 0x1p-20f;
        scaled = scaled * weight_scale2;
        scaled = scaled * (1.0f / input_scale_inv);
        const float total = float(half(scaled));
        if (lane == 0) {
            output[(args.col0 + uint(column)) * args.n_out + row] = total;
        }
    }
}

#define MUSER_NVFP4_W4A4_ENTRY(NC) \
kernel void muser_nvfp4_w4a4_matvec_c##NC( \
    device const uchar *packed [[buffer(0)]], \
    device const uchar *scales [[buffer(1)]], \
    device const float *input [[buffer(2)]], \
    device float *output [[buffer(3)]], \
    constant muser_nvfp4_args &args [[buffer(4)]], \
    constant float &weight_scale2 [[buffer(5)]], \
    constant float &input_scale_inv [[buffer(6)]], \
    uint row [[threadgroup_position_in_grid]], \
    ushort lane [[thread_index_in_simdgroup]]) { \
    muser_nvfp4_w4a4_matvec_impl<NC>( \
        packed, scales, input, output, args, weight_scale2, input_scale_inv, row, lane); \
}

MUSER_NVFP4_W4A4_ENTRY(1)
MUSER_NVFP4_W4A4_ENTRY(2)
MUSER_NVFP4_W4A4_ENTRY(4)
MUSER_NVFP4_W4A4_ENTRY(8)
MUSER_NVFP4_W4A4_ENTRY(16)

// Exact M=16 W4A4 tile for speculative verification. Each tensor operation
// contracts one 16-value E2M1 block: its small integer products and sum are
// exactly representable in the half-input/f32-accumulator matrix unit. The
// per-block E4M3 Q9 scales and cross-block accumulation remain i64, matching
// muser_nvfp4_w4a4_matvec_impl before the identical scalar epilogue.
kernel void muser_nvfp4_w4a4_m16_n32(
    device const uchar *packed [[buffer(0)]],
    device const uchar *scales [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant muser_nvfp4_args &args [[buffer(4)]],
    constant float &weight_scale2 [[buffer(5)]],
    constant float &input_scale_inv [[buffer(6)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    ushort simd [[simdgroup_index_in_threadgroup]]) {
    const uint row0 = group * 32u;
    const uint packed_per_row = args.n_in / 2u;
    const uint scales_per_row = args.n_in / 16u;
    threadgroup half weight_tile[64 * 32];  // K-major: [element][row]
    threadgroup half activation_tile[16 * 64]; // row-major: [column][element]
    threadgroup float block_dots[4 * 16 * 32]; // [block][column][row]
    threadgroup int weight_scales[4 * 32];
    threadgroup int activation_scales[4 * 16];
    long totals[2] = {0, 0};

    for (uint k_group = 0; k_group < args.n_in / 64u; ++k_group) {
        for (uint slot = 0; slot < 4u; ++slot) {
            const uint index = tid + slot * 256u;
            const uint column = index >> 6u;
            const uint element = index & 63u;
            const uint input_index = (args.col0 + column) * args.n_in
                + k_group * 64u + element;
            activation_tile[index] = half(input[input_index]);
        }
        for (uint slot = 0; slot < 8u; ++slot) {
            const uint index = tid + slot * 256u;
            const uint weight_element = index >> 5u;
            const uint local_row = index & 31u;
            const uint row = row0 + local_row;
            uchar code = 0;
            if (row < args.n_out) {
                const uint packed_index = row * packed_per_row
                    + k_group * 32u + (weight_element >> 1u);
                const uchar byte = packed[packed_index];
                code = (weight_element & 1u) == 0u ? byte & 15u : byte >> 4u;
            }
            weight_tile[weight_element * 32u + local_row] = half(muser_e2m1_q1(code));
        }
        if (tid < 128u) {
            const uint block = tid >> 5u;
            const uint local_row = tid & 31u;
            const uint row = row0 + local_row;
            weight_scales[tid] = row < args.n_out
                ? muser_e4m3_q9(scales[row * scales_per_row + k_group * 4u + block])
                : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < 64u) {
            const uint block = tid >> 4u;
            const uint column = tid & 15u;
            const uint base = column * 64u + block * 16u;
            float abs_max = 0.0f;
            for (ushort value_index = 0; value_index < 16; ++value_index) {
                abs_max = max(abs_max, abs(float(activation_tile[base + value_index])));
            }
            const float normalized_max = fma(abs_max, 0x1.555556p-3f, 0.0f);
            const uchar scale_code = muser_e4m3fn_round_positive(
                min(fma(input_scale_inv, normalized_max, 0.0f), 448.0f));
            const float scale = muser_e4m3fn(scale_code);
            activation_scales[block * 16u + column] = muser_e4m3_q9(scale_code);
            for (ushort value_index = 0; value_index < 16; ++value_index) {
                const float value = float(activation_tile[base + value_index]);
                const uchar code = (scale_code & 0x7f) == 0
                    ? (signbit(value) ? 8 : 0)
                    : muser_e2m1_round_ratio(value, input_scale_inv, scale);
                activation_tile[base + value_index] = half(muser_e2m1_q1(code));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint row_tile = uint(simd) & 3u;
        const uint column_tile = uint(simd) >> 2u;
        for (uint block = 0; block < 4u; ++block) {
            simdgroup_float8x8 result = simdgroup_float8x8(0);
            for (uint k_tile = 0; k_tile < 2u; ++k_tile) {
                simdgroup_half8x8 weights;
                simdgroup_half8x8 activations;
                simdgroup_load(
                    weights,
                    weight_tile + block * 16u * 32u + k_tile * 8u * 32u + row_tile * 8u,
                    32);
                simdgroup_load(
                    activations,
                    activation_tile + column_tile * 8u * 64u + block * 16u + k_tile * 8u,
                    64);
                simdgroup_multiply_accumulate(result, activations, weights, result);
            }
            simdgroup_store(
                result,
                block_dots + block * 16u * 32u + column_tile * 8u * 32u + row_tile * 8u,
                32);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint slot = 0; slot < 2u; ++slot) {
            const uint index = tid + slot * 256u;
            const uint result_column = index >> 5u;
            const uint result_row = index & 31u;
            for (uint block = 0; block < 4u; ++block) {
                totals[slot] += long(int(block_dots[block * 16u * 32u + index]))
                    * long(weight_scales[block * 32u + result_row])
                    * long(activation_scales[block * 16u + result_column]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint slot = 0; slot < 2u; ++slot) {
        const uint index = tid + slot * 256u;
        const uint column = index >> 5u;
        const uint local_row = index & 31u;
        const uint row = row0 + local_row;
        if (row < args.n_out) {
            float scaled = float(totals[slot]) * 0x1p-20f;
            scaled = scaled * weight_scale2;
            scaled = scaled * (1.0f / input_scale_inv);
            output[(args.col0 + column) * args.n_out + row] = float(half(scaled));
        }
    }
}

// M=16 activation quantization shared by every output tile in one
// projection. One thread owns one 16-value block so the max reduction and
// rounding order are identical to the fused oracle kernel above.
kernel void muser_nvfp4_w4a4_quantize_m16(
    device const float *input [[buffer(0)]],
    device half *quantized [[buffer(1)]],
    device int *activation_scales [[buffer(2)]],
    constant uint &n_in [[buffer(3)]],
    constant float &input_scale_inv [[buffer(4)]],
    uint block_index [[thread_position_in_grid]]) {
    const uint blocks_per_column = n_in / 16u;
    if (block_index >= 16u * blocks_per_column) {
        return;
    }
    const uint column = block_index / blocks_per_column;
    const uint block = block_index - column * blocks_per_column;
    const uint base = column * n_in + block * 16u;
    float rounded[16];
    float abs_max = 0.0f;
    for (ushort element = 0; element < 16; ++element) {
        rounded[element] = float(half(input[base + uint(element)]));
        abs_max = max(abs_max, abs(rounded[element]));
    }
    const float normalized_max = fma(abs_max, 0x1.555556p-3f, 0.0f);
    const uchar scale_code = muser_e4m3fn_round_positive(
        min(fma(input_scale_inv, normalized_max, 0.0f), 448.0f));
    const float scale = muser_e4m3fn(scale_code);
    activation_scales[block_index] = muser_e4m3_q9(scale_code);
    for (ushort element = 0; element < 16; ++element) {
        const float value = rounded[element];
        const uchar code = (scale_code & 0x7f) == 0
            ? (signbit(value) ? 8 : 0)
            : muser_e2m1_round_ratio(value, input_scale_inv, scale);
        quantized[base + uint(element)] = half(muser_e2m1_q1(code));
    }
}

// The contraction half of the exact M16/N32 tile, consuming activation bits
// and Q9 scales produced once by `muser_nvfp4_w4a4_quantize_m16`.
kernel void muser_nvfp4_w4a4_prequant_m16_n32(
    device const uchar *packed [[buffer(0)]],
    device const uchar *scales [[buffer(1)]],
    device const half *quantized [[buffer(2)]],
    device const int *prequant_scales [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant muser_nvfp4_args &args [[buffer(5)]],
    constant float &weight_scale2 [[buffer(6)]],
    constant float &input_scale_inv [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    ushort simd [[simdgroup_index_in_threadgroup]]) {
    const uint row0 = group * 32u;
    const uint packed_per_row = args.n_in / 2u;
    const uint scales_per_row = args.n_in / 16u;
    threadgroup half weight_tile[64 * 32];
    threadgroup half activation_tile[16 * 64];
    threadgroup float block_dots[4 * 16 * 32];
    threadgroup int weight_scales[4 * 32];
    threadgroup int activation_scales[4 * 16];
    long totals[2] = {0, 0};

    for (uint k_group = 0; k_group < args.n_in / 64u; ++k_group) {
        for (uint slot = 0; slot < 4u; ++slot) {
            const uint index = tid + slot * 256u;
            const uint column = index >> 6u;
            const uint element = index & 63u;
            activation_tile[index] = quantized[
                column * args.n_in + k_group * 64u + element];
        }
        for (uint slot = 0; slot < 8u; ++slot) {
            const uint index = tid + slot * 256u;
            const uint weight_element = index >> 5u;
            const uint local_row = index & 31u;
            const uint row = row0 + local_row;
            uchar code = 0;
            if (row < args.n_out) {
                const uint packed_index = row * packed_per_row
                    + k_group * 32u + (weight_element >> 1u);
                const uchar byte = packed[packed_index];
                code = (weight_element & 1u) == 0u ? byte & 15u : byte >> 4u;
            }
            weight_tile[weight_element * 32u + local_row] = half(muser_e2m1_q1(code));
        }
        if (tid < 128u) {
            const uint block = tid >> 5u;
            const uint local_row = tid & 31u;
            const uint row = row0 + local_row;
            weight_scales[tid] = row < args.n_out
                ? muser_e4m3_q9(scales[row * scales_per_row + k_group * 4u + block])
                : 0;
        }
        if (tid < 64u) {
            const uint block = tid >> 4u;
            const uint column = tid & 15u;
            activation_scales[tid] = prequant_scales[
                column * scales_per_row + k_group * 4u + block];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint row_tile = uint(simd) & 3u;
        const uint column_tile = uint(simd) >> 2u;
        for (uint block = 0; block < 4u; ++block) {
            simdgroup_float8x8 result = simdgroup_float8x8(0);
            for (uint k_tile = 0; k_tile < 2u; ++k_tile) {
                simdgroup_half8x8 weights;
                simdgroup_half8x8 activations;
                simdgroup_load(
                    weights,
                    weight_tile + block * 16u * 32u + k_tile * 8u * 32u + row_tile * 8u,
                    32);
                simdgroup_load(
                    activations,
                    activation_tile + column_tile * 8u * 64u + block * 16u + k_tile * 8u,
                    64);
                simdgroup_multiply_accumulate(result, activations, weights, result);
            }
            simdgroup_store(
                result,
                block_dots + block * 16u * 32u + column_tile * 8u * 32u + row_tile * 8u,
                32);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint slot = 0; slot < 2u; ++slot) {
            const uint index = tid + slot * 256u;
            const uint result_column = index >> 5u;
            const uint result_row = index & 31u;
            for (uint block = 0; block < 4u; ++block) {
                totals[slot] += long(int(block_dots[block * 16u * 32u + index]))
                    * long(weight_scales[block * 32u + result_row])
                    * long(activation_scales[block * 16u + result_column]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint slot = 0; slot < 2u; ++slot) {
        const uint index = tid + slot * 256u;
        const uint column = index >> 5u;
        const uint local_row = index & 31u;
        const uint row = row0 + local_row;
        if (row < args.n_out) {
            float scaled = float(totals[slot]) * 0x1p-20f;
            scaled = scaled * weight_scale2;
            scaled = scaled * (1.0f / input_scale_inv);
            output[column * args.n_out + row] = float(half(scaled));
        }
    }
}

inline int muser_nvfp4_q8_nearest(float scale, float value) {
    return (as_type<int>(fma(scale, value, 12582912.0f)) & 0x007fffff) - 0x00400000;
}

// Strict weight-only NVFP4 × Q8_K projection. One 256-value activation block
// is quantized once per threadgroup and shared by eight output rows. E2M1 Q1,
// Q8, and E4M3FN Q9 stay integer through the complete block contraction; only
// the fixed conversion, activation scale, tensor scale, and sequential block
// accumulation are scalar f32 operations in lane zero.
kernel void muser_nvfp4_a16_q8_matvec(
    device const uchar *packed [[buffer(0)]],
    device const uchar *scales [[buffer(1)]],
    device const float *input [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &cols [[buffer(5)]],
    constant uint &tokens [[buffer(6)]],
    constant float &weight_scale2 [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd [[simdgroup_index_in_threadgroup]]) {
    const uint row = group.x * 8u + uint(simd);
    const uint token = group.y;
    const bool active = row < rows && token < tokens;
    const uint packed_per_row = cols / 2u;
    const uint scales_per_row = cols / 16u;
    device const uchar *row_packed = active
        ? packed + ulong(row) * ulong(packed_per_row)
        : packed;
    device const uchar *row_scales = active
        ? scales + ulong(row) * ulong(scales_per_row)
        : scales;
    device const float *x = input + ulong(token) * ulong(cols);
    threadgroup float rounded[256];
    threadgroup float magnitudes[256];
    threadgroup float block_iscale;
    threadgroup char q8[256];
    float total = 0.0f;

    for (uint block_index = 0; block_index < cols / 256u; ++block_index) {
        device const float *xb = x + block_index * 256u;
        const float value = float(half(xb[tid]));
        rounded[tid] = value;
        magnitudes[tid] = fabs(value);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float best = 0.0f;
            uint best_index = 256u;
            for (uint index = 0; index < 256u; ++index) {
                if (magnitudes[index] > best) {
                    best = magnitudes[index];
                    best_index = index;
                }
            }
            block_iscale = best == 0.0f ? 0.0f : -127.0f / rounded[best_index];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        q8[tid] = block_iscale == 0.0f
            ? char(0)
            : char(min(127, muser_nvfp4_q8_nearest(block_iscale, rounded[tid])));
        threadgroup_barrier(mem_flags::mem_threadgroup);

        long weighted = 0;
        if (active && block_iscale != 0.0f && lane < 16) {
            const uint scale_group = block_index * 16u + uint(lane);
            const uint packed_base = scale_group * 8u;
            const uint quant_base = uint(lane) * 16u;
            int dot = 0;
            for (ushort pair = 0; pair < 8; ++pair) {
                const uchar byte = row_packed[packed_base + uint(pair)];
                dot += muser_e2m1_q1(byte & 15) * int(q8[quant_base + uint(pair) * 2u]);
                dot += muser_e2m1_q1(byte >> 4) * int(q8[quant_base + uint(pair) * 2u + 1u]);
            }
            weighted = long(dot) * long(muser_e4m3_q9(row_scales[scale_group]));
        }
        const long integer_total = muser_simd_sum_i64(weighted, lane);
        if (active && lane == 0 && block_iscale != 0.0f) {
            float contribution = fma(float(integer_total), 0x1p-10f, 0.0f);
            const float q8_scale = 1.0f / block_iscale;
            contribution = fma(contribution, q8_scale, 0.0f);
            contribution = fma(contribution, weight_scale2, 0.0f);
            total = fma(1.0f, contribution, total);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active && lane == 0) {
        output[ulong(token) * ulong(rows) + row] = float(half(total));
    }
}

template <ushort NC>
inline void muser_f16_matvec_impl(
    device const half *weights,
    device const float *input,
    device float *output,
    constant muser_nvfp4_args &args,
    uint row,
    ushort lane) {
    if (row >= args.n_out) {
        return;
    }
    device const half4 *weight_row =
        reinterpret_cast<device const half4 *>(weights + row * args.n_in);
    float sums[NC];
    for (ushort column = 0; column < NC; ++column) {
        sums[column] = 0.0f;
    }
    for (uint vector = lane; vector < args.n_in / 4; vector += 32) {
        const float4 weight = float4(weight_row[vector]);
        for (ushort column = 0; column < NC; ++column) {
            device const float4 *x = reinterpret_cast<device const float4 *>(
                input + (args.col0 + uint(column)) * args.n_in);
            sums[column] += dot(weight, x[vector]);
        }
    }
    for (ushort column = 0; column < NC; ++column) {
        const float total = simd_sum(sums[column]);
        if (lane == 0) {
            output[(args.col0 + uint(column)) * args.n_out + row] = total;
        }
    }
}

#define MUSER_F16_ENTRY(NC) \
kernel void muser_f16_matvec_c##NC( \
    device const half *weights [[buffer(0)]], \
    device const float *input [[buffer(1)]], \
    device float *output [[buffer(2)]], \
    constant muser_nvfp4_args &args [[buffer(3)]], \
    uint row [[threadgroup_position_in_grid]], \
    ushort lane [[thread_index_in_simdgroup]]) { \
    muser_f16_matvec_impl<NC>(weights, input, output, args, row, lane); \
}

MUSER_F16_ENTRY(1)
MUSER_F16_ENTRY(2)
MUSER_F16_ENTRY(4)
MUSER_F16_ENTRY(8)
MUSER_F16_ENTRY(16)


kernel void muser_embedding_f16(
    device const half *weights [[buffer(0)]],
    device const uint *tokens [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &hidden_dim [[buffer(3)]],
    constant uint &vocab_size [[buffer(4)]],
    constant uint &token_count [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= hidden_dim * token_count) {
        return;
    }
    const uint token_index = index / hidden_dim;
    const uint hidden = index - token_index * hidden_dim;
    const uint token = min(tokens[token_index], vocab_size - 1);
    output[index] = float(weights[token * hidden_dim + hidden]);
}

#pragma clang fp reassociate(off)
#pragma clang fp contract(off)

kernel void muser_nvfp4_dequant_fixture(
    device const uchar *packed [[buffer(0)]],
    device const uchar *scales [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &elements [[buffer(3)]],
    constant float &scale2 [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= elements) {
        return;
    }
    const uchar byte = packed[index / 2];
    const uchar nibble = (index & 1) == 0 ? byte & 0x0f : byte >> 4;
    output[index] = (muser_e2m1(nibble) * muser_e4m3fn(scales[index / 16])) * scale2;
}

#pragma clang fp reassociate(on)
#pragma clang fp contract(on)
