#include <metal_stdlib>
using namespace metal;

kernel void muser_silu_mul_inplace(
    device float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) {
        float value = gate[index];
        gate[index] = (value / (1.0f + exp(-value))) * up[index];
    }
}

kernel void muser_scale_softcap_inplace(
    device float *logits [[buffer(0)]],
    constant uint &count [[buffer(1)]],
    constant float &scale [[buffer(2)]],
    constant float &softcap [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) {
        float value = logits[index] * scale;
        logits[index] = softcap > 0.0f ? softcap * tanh(value / softcap) : value;
    }
}

inline float muser_f16(device const uchar *bytes) {
    ushort bits = ushort(bytes[0]) | (ushort(bytes[1]) << 8);
    return float(as_type<half>(bits));
}

inline uchar2 muser_scale_min(device const uchar *scales, uint index) {
    if (index < 4) {
        return uchar2(scales[index] & 0x3f, scales[index + 4] & 0x3f);
    }
    uchar scale = (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4);
    uchar minimum = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4);
    return uchar2(scale, minimum);
}

inline void muser_decode_all_q4k_scales(
    float d,
    float dmin,
    uint sd0,
    uint sd1,
    uint sd2,
    thread float d_scale[8],
    thread float neg_min[8]) {
    for (uint index = 0; index < 4; ++index) {
        uint shift = index * 8;
        d_scale[index] = d * float((sd0 >> shift) & 0x3f);
        neg_min[index] = -(dmin * float((sd1 >> shift) & 0x3f));
    }
    for (uint index = 0; index < 4; ++index) {
        uint shift = index * 8;
        uint upper = (sd2 >> shift) & 0xff;
        uint lower_scale = (sd0 >> shift) & 0xff;
        uint lower_min = (sd1 >> shift) & 0xff;
        d_scale[index + 4] = d * float((upper & 0x0f) | ((lower_scale >> 6) << 4));
        neg_min[index + 4] = -(dmin * float((upper >> 4) | ((lower_min >> 6) << 4)));
    }
}

// Cross-vendor K-quant arithmetic.  CUDA and Metal both quantize each f32
// activation super-block to the same Q8_K integers, reduce only integer dot
// products in parallel, then execute the two scalar IEEE-f32 boundaries in
// lane zero.  This avoids depending on vendor-specific floating subgroup-sum
// topology while retaining one output row per SIMD group.
inline int muser_q8_nearest(float scale, float value) {
    return (as_type<int>(fma(scale, value, 12582912.0f)) & 0x007fffff) - 0x00400000;
}

kernel void muser_cross_vendor_q4k(
    device const uchar *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &cols [[buffer(4)]],
    constant uint &tokens [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    uint row = group.x * 8u + simd;
    uint token = group.y;
    bool active = row < rows && token < tokens;
    uint blocks = cols / 256u;
    uint row_bytes = blocks * 144u;
    device const uchar *row_data = active ? weights + ulong(row) * ulong(row_bytes) : weights;
    device const float *x = input + ulong(token) * ulong(cols);
    threadgroup float magnitudes[256];
    threadgroup float block_iscale;
    threadgroup char q8[256];
    float total = 0.0f;

    for (uint block_index = 0; block_index < blocks; ++block_index) {
        device const float *xb = x + block_index * 256u;
        magnitudes[tid] = fabs(xb[tid]);
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
            magnitudes[0] = best;
            block_iscale = best == 0.0f ? 0.0f : -127.0f / xb[best_index];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (magnitudes[0] != 0.0f) {
            q8[tid] = char(min(127, muser_q8_nearest(block_iscale, xb[tid])));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active && magnitudes[0] != 0.0f) {
            device const uchar *block = row_data + block_index * 144u;
            device const uchar *scales = block + 4;
            device const uchar *quants = block + 16;
            int dot = 0;
            int sumi = 0;
            for (uint chunk = 0; chunk < 8u; ++chunk) {
                uchar2 scale_min = muser_scale_min(scales, chunk);
                uint packed = quants[(chunk / 2u) * 32u + lane];
                uint quant = (chunk & 1u) == 0u ? packed & 0x0fu : packed >> 4u;
                int activation = int(q8[chunk * 32u + lane]);
                dot += int(scale_min.x) * activation * int(quant);
                sumi += int(scale_min.y) * activation;
            }
            dot = simd_sum(dot);
            sumi = simd_sum(sumi);
            if (lane == 0) {
                float q8_scale = 1.0f / block_iscale;
                float d = q8_scale * muser_f16(block);
                float dmin = q8_scale * muser_f16(block + 2);
                total = fma(-dmin, float(sumi), total);
                total = fma(d, float(dot), total);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active && lane == 0) output[ulong(token) * ulong(rows) + row] = total;
}

kernel void muser_cross_vendor_q5k(
    device const uchar *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &cols [[buffer(4)]],
    constant uint &tokens [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    uint row = group.x * 8u + simd;
    uint token = group.y;
    bool active = row < rows && token < tokens;
    uint blocks = cols / 256u;
    uint row_bytes = blocks * 176u;
    device const uchar *row_data = active ? weights + ulong(row) * ulong(row_bytes) : weights;
    device const float *x = input + ulong(token) * ulong(cols);
    threadgroup float magnitudes[256];
    threadgroup float block_iscale;
    threadgroup char q8[256];
    float total = 0.0f;

    for (uint block_index = 0; block_index < blocks; ++block_index) {
        device const float *xb = x + block_index * 256u;
        magnitudes[tid] = fabs(xb[tid]);
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
            magnitudes[0] = best;
            block_iscale = best == 0.0f ? 0.0f : -127.0f / xb[best_index];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (magnitudes[0] != 0.0f) {
            q8[tid] = char(min(127, muser_q8_nearest(block_iscale, xb[tid])));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active && magnitudes[0] != 0.0f) {
            device const uchar *block = row_data + block_index * 176u;
            device const uchar *scales = block + 4;
            device const uchar *high_bits = block + 16;
            device const uchar *quants = block + 48;
            int dot = 0;
            int sumi = 0;
            for (uint chunk = 0; chunk < 8u; ++chunk) {
                uchar2 scale_min = muser_scale_min(scales, chunk);
                uint packed = quants[(chunk / 2u) * 32u + lane];
                uint quant = (chunk & 1u) == 0u ? packed & 0x0fu : packed >> 4u;
                if ((high_bits[lane] & uchar(1u << chunk)) != 0u) quant += 16u;
                int activation = int(q8[chunk * 32u + lane]);
                dot += int(scale_min.x) * activation * int(quant);
                sumi += int(scale_min.y) * activation;
            }
            dot = simd_sum(dot);
            sumi = simd_sum(sumi);
            if (lane == 0) {
                float q8_scale = 1.0f / block_iscale;
                float d = q8_scale * muser_f16(block);
                float dmin = q8_scale * muser_f16(block + 2);
                float inner = fma(d, float(dot), -(dmin * float(sumi)));
                total = total + inner;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active && lane == 0) output[ulong(token) * ulong(rows) + row] = total;
}

kernel void muser_cross_vendor_q6k(
    device const uchar *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &cols [[buffer(4)]],
    constant uint &tokens [[buffer(5)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    uint row = group.x * 8u + simd;
    uint token = group.y;
    bool active = row < rows && token < tokens;
    uint blocks = cols / 256u;
    uint row_bytes = blocks * 210u;
    device const uchar *row_data = active ? weights + ulong(row) * ulong(row_bytes) : weights;
    device const float *x = input + ulong(token) * ulong(cols);
    threadgroup float magnitudes[256];
    threadgroup float block_iscale;
    threadgroup char q8[256];
    float total = 0.0f;

    for (uint block_index = 0; block_index < blocks; ++block_index) {
        device const float *xb = x + block_index * 256u;
        magnitudes[tid] = fabs(xb[tid]);
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
            magnitudes[0] = best;
            block_iscale = best == 0.0f ? 0.0f : -127.0f / xb[best_index];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (magnitudes[0] != 0.0f) {
            q8[tid] = char(min(127, muser_q8_nearest(block_iscale, xb[tid])));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active && magnitudes[0] != 0.0f) {
            device const uchar *block = row_data + block_index * 210u;
            device const uchar *ql = block;
            device const uchar *qh = block + 128;
            device const char *scales = reinterpret_cast<device const char *>(block + 192);
            int dot = 0;
            for (uint half_index = 0; half_index < 2u; ++half_index) {
                uint qbase = half_index * 128u;
                uchar high = qh[half_index * 32u + lane];
                uchar low0 = ql[half_index * 64u + lane];
                uchar low1 = ql[half_index * 64u + 32u + lane];
                int q0 = int((low0 & 0x0fu) | (((high >> 0u) & 3u) << 4u)) - 32;
                int q1 = int((low1 & 0x0fu) | (((high >> 2u) & 3u) << 4u)) - 32;
                int q2 = int((low0 >> 4u) | (((high >> 4u) & 3u) << 4u)) - 32;
                int q3 = int((low1 >> 4u) | (((high >> 6u) & 3u) << 4u)) - 32;
                uint odd = lane >= 16u ? 1u : 0u;
                dot += int(scales[half_index * 8u + 0u + odd]) * int(q8[qbase + lane]) * q0;
                dot += int(scales[half_index * 8u + 2u + odd]) * int(q8[qbase + 32u + lane]) * q1;
                dot += int(scales[half_index * 8u + 4u + odd]) * int(q8[qbase + 64u + lane]) * q2;
                dot += int(scales[half_index * 8u + 6u + odd]) * int(q8[qbase + 96u + lane]) * q3;
            }
            dot = simd_sum(dot);
            if (lane == 0) {
                float q8_scale = 1.0f / block_iscale;
                float d = q8_scale * muser_f16(block + 208);
                total = fma(d, float(dot), total);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active && lane == 0) output[ulong(token) * ulong(rows) + row] = total;
}

// Strict one-SIMD-group RMSNorm paired with the Triton producer kernel. Each
// lane owns the same stride-32 sequence, followed by an explicit binary tree.
// Every store crosses the producer's model-dtype (F16) boundary.
kernel void muser_cross_vendor_rms_per_head(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &head_dim [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    uint head [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
    uint offset = head * head_dim;
    float sum = 0.0f;
    for (uint index = uint(lane); index < head_dim; index += 32u) {
        float value = input[offset + index];
        sum = fma(value, value, sum);
    }
    for (ushort distance = 16; distance != 0; distance >>= 1) {
        float other = simd_shuffle_down(sum, distance);
        if (lane < distance) sum = sum + other;
    }
    float inverse = lane == 0
        ? 1.0f / sqrt(fma(sum, 1.0f / float(head_dim), eps))
        : 0.0f;
    inverse = simd_broadcast_first(inverse);
    for (uint index = uint(lane); index < head_dim; index += 32u) {
        float normalized = fma(input[offset + index], inverse, 0.0f);
        output[offset + index] = float(half(fma(normalized, weight[index], 0.0f)));
    }
}

// Unweighted companion used where llama.cpp materializes RMS_NORM and MUL as
// distinct graph nodes.  Keeping the learned-weight multiply out of this
// dispatch preserves the exact f32 store/load boundary of the CUDA oracle.
kernel void muser_cross_vendor_rms_unweighted(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &row_dim [[buffer(2)]],
    constant float &eps [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
    uint offset = row * row_dim;
    float sum = 0.0f;
    for (uint index = uint(lane); index < row_dim; index += 32u) {
        float value = input[offset + index];
        sum = fma(value, value, sum);
    }
    for (ushort distance = 16; distance != 0; distance >>= 1) {
        float other = simd_shuffle_down(sum, distance);
        if (lane < distance) sum = sum + other;
    }
    float inverse = lane == 0
        ? 1.0f / sqrt(fma(sum, 1.0f / float(row_dim), eps))
        : 0.0f;
    inverse = simd_broadcast_first(inverse);
    for (uint index = uint(lane); index < row_dim; index += 32u) {
        output[offset + index] = float(half(fma(input[offset + index], inverse, 0.0f)));
    }
}

kernel void muser_cross_vendor_mul_weight(
    device float *values [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    constant uint &row_dim [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    values[index] = float(half(fma(values[index], weight[index % row_dim], 0.0f)));
}

inline float muser_cross_vendor_expf(float value);

// Bit-identical port of llama.cpp's ARM NEON ggml_v_expf four-lane helper.
// CUDA uses the same scalarized implementation in the strict producer path.
inline void muser_cross_vendor_neon_expf4(
    thread float *output,
    thread const float *input) {
    const float rounding = 0x1.8p23f;
    float z[4], n[4], reduced[4], scale[4], polynomial[4];
    uint exponent[4];
    uint overflow_mask = 0u;
    for (uint lane = 0; lane < 4u; ++lane) {
        z[lane] = fma(input[lane], 0x1.715476p+0f, rounding);
        n[lane] = z[lane] + -rounding;
        reduced[lane] = fma(-n[lane], 0x1.62e4p-1f, input[lane]);
        reduced[lane] = fma(-n[lane], 0x1.7f7d1cp-20f, reduced[lane]);
        exponent[lane] = as_type<uint>(z[lane]) << 23u;
        scale[lane] = as_type<float>(exponent[lane] + 0x3f800000u);
        if (fabs(n[lane]) > 126.0f) overflow_mask |= 1u << lane;
        float squared = reduced[lane] * reduced[lane];
        float t1 = fma(0x1.0e4020p-7f, reduced[lane], 0x1.573e2ep-5f);
        float t2 = fma(0x1.555e66p-3f, reduced[lane], 0x1.fffdb6p-2f);
        t2 = fma(t1, squared, t2);
        polynomial[lane] = fma(t2, squared, 0x1.ffffecp-1f * reduced[lane]);
    }
    if (overflow_mask == 0u) {
        for (uint lane = 0; lane < 4u; ++lane) {
            output[lane] = fma(polynomial[lane], scale[lane], scale[lane]);
        }
        return;
    }
    for (uint lane = 0; lane < 4u; ++lane) {
        uint delta = n[lane] <= 0.0f ? 0x82000000u : 0u;
        float scale1 = as_type<float>(delta + 0x7f000000u);
        float scale2 = as_type<float>(exponent[lane] - delta);
        float magnitude = fabs(n[lane]);
        if (magnitude > 192.0f) {
            output[lane] = scale1 * scale1;
        } else if (magnitude > 126.0f) {
            output[lane] = fma(scale2, polynomial[lane], scale2) * scale1;
        } else {
            output[lane] = fma(scale[lane], polynomial[lane], scale[lane]);
        }
    }
}

kernel void muser_cross_vendor_swiglu(
    device float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint group [[thread_position_in_grid]]) {
    uint base = group * 4u;
    if (base + 3u >= count) return;
    float values[4], negated[4], exponentials[4];
    for (uint lane = 0; lane < 4u; ++lane) {
        values[lane] = gate[base + lane];
        negated[lane] = -values[lane];
    }
    muser_cross_vendor_neon_expf4(exponentials, negated);
    for (uint lane = 0; lane < 4u; ++lane) {
        float silu = values[lane] / (1.0f + exponentials[lane]);
        gate[base + lane] = silu * up[base + lane];
    }
}

kernel void muser_cross_vendor_scale(
    device float *values [[buffer(0)]],
    constant uint &count [[buffer(1)]],
    constant float &scale [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    values[index] = values[index] * scale;
}

kernel void muser_cross_vendor_tanh(
    device float *values [[buffer(0)]],
    constant uint &count [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    float value = values[index];
    float magnitude = fabs(value);
    float exponential = muser_cross_vendor_expf(-2.0f * magnitude);
    float ratio = (1.0f - exponential) / (1.0f + exponential);
    values[index] = signbit(value) ? -ratio : ratio;
}

// RoPE companion to the pinned producer. The table is the canonical integer
// Q30 NCO expanded once to f32. Match CUDA's separately-rounded cosine
// multiply followed by a single sine FMA, then round the destination to f16.
kernel void muser_cross_vendor_rope(
    device float *q [[buffer(0)]],
    device float *k [[buffer(1)]],
    device const float *trig [[buffer(2)]],
    device const uint *positions [[buffer(3)]],
    constant uint &n_heads [[buffer(4)]],
    constant uint &n_kv_heads [[buffer(5)]],
    constant uint &head_dim [[buffer(6)]],
    constant uint &token_count [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint pair = group.x * 32u + lane;
    uint token = group.y;
    uint pairs_per_head = head_dim / 2u;
    uint q_pairs = n_heads * pairs_per_head;
    uint total_pairs = q_pairs + n_kv_heads * pairs_per_head;
    if (pair >= total_pairs || token >= token_count) return;
    bool is_q = pair < q_pairs;
    uint local_pair = is_q ? pair : pair - q_pairs;
    uint head = local_pair / pairs_per_head;
    uint head_pair = local_pair % pairs_per_head;
    uint index = token * (is_q ? n_heads : n_kv_heads) * head_dim
        + head * head_dim + head_pair * 2u;
    uint position = positions[token];
    device const float *angles = trig + ulong(position) * ulong(head_dim);
    float cosine = angles[head_pair * 2u];
    float sine = angles[head_pair * 2u + 1u];
    device float *values = is_q ? q : k;
    float x0 = float(half(values[index]));
    float x1 = float(half(values[index + 1u]));
    float x0_cos = fma(x0, cosine, 0.0f);
    float x1_cos = fma(x1, cosine, 0.0f);
    values[index] = float(half(fma(-x1, sine, x0_cos)));
    values[index + 1u] = float(half(fma(x0, sine, x1_cos)));
}

// DFlash companion using its existing NEOX half-split representation.  The
// trig table has the identical position-major `(cos, sin)` byte layout used
// by `muser_cross_vendor_rope` and the CUDA producer.  Only the element-pair
// addressing differs; the pinned FP32 arithmetic boundaries do not.
kernel void muser_cross_vendor_rope_neox(
    device float *q [[buffer(0)]],
    device float *k [[buffer(1)]],
    device const float *trig [[buffer(2)]],
    constant uint &n_heads [[buffer(3)]],
    constant uint &n_kv_heads [[buffer(4)]],
    constant uint &head_dim [[buffer(5)]],
    constant uint &start_position [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint pair = group.x * 32u + lane;
    uint token = group.y;
    uint pairs_per_head = head_dim / 2u;
    uint q_pairs = n_heads * pairs_per_head;
    uint total_pairs = q_pairs + n_kv_heads * pairs_per_head;
    if (pair >= total_pairs) return;
    bool is_q = pair < q_pairs;
    uint local_pair = is_q ? pair : pair - q_pairs;
    uint head = local_pair / pairs_per_head;
    uint head_pair = local_pair % pairs_per_head;
    uint head_base = token * (is_q ? n_heads : n_kv_heads) * head_dim
        + head * head_dim;
    uint position = start_position + token;
    device const float *angles = trig + ulong(position) * ulong(head_dim);
    float cosine = angles[head_pair * 2u];
    float sine = angles[head_pair * 2u + 1u];
    device float *values = is_q ? q : k;
    float x0 = float(half(values[head_base + head_pair]));
    float x1 = float(half(values[head_base + pairs_per_head + head_pair]));
    float x0_cos = fma(x0, cosine, 0.0f);
    float x1_cos = fma(x1, cosine, 0.0f);
    values[head_base + head_pair] = float(half(
        fma(-x1, sine, x0_cos)));
    values[head_base + pairs_per_head + head_pair] =
        float(half(fma(x0, sine, x1_cos)));
}

inline float muser_cross_vendor_expf(float value) {
    if (value < -80.0f) return 0.0f;
    float rounding = 0x1.8p23f;
    float z = fma(value, 0x1.715476p+0f, rounding);
    float n = z - rounding;
    float reduced = fma(-n, 0x1.62e4p-1f, value);
    reduced = fma(-n, 0x1.7f7d1cp-20f, reduced);
    uint exponent = as_type<uint>(z) << 23u;
    float scale = as_type<float>(exponent + 0x3f800000u);
    float squared = reduced * reduced;
    float p1 = fma(0x1.0e4020p-7f, reduced, 0x1.573e2ep-5f);
    float p2 = fma(0x1.555e66p-3f, reduced, 0x1.fffdb6p-2f);
    p2 = fma(p1, squared, p2);
    float correction = fma(p2, squared, 0x1.ffffecp-1f * reduced);
    return fma(correction, scale, scale);
}

// Deterministic scalar correctness path for an already-populated F16 cache.
// Q is rounded to F16 at the dot boundary, matching the pinned CPU decode
// graph, while every accumulation and exponential boundary is shared with
// the CUDA companion kernel.
kernel void muser_cross_vendor_attention_decode(
    device const float *query [[buffer(0)]],
    device const half *key_cache [[buffer(1)]],
    device const half *value_cache [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &visible [[buffer(4)]],
    constant uint &capacity [[buffer(5)]],
    constant uint &n_heads [[buffer(6)]],
    constant uint &n_kv_heads [[buffer(7)]],
    constant uint &head_dim [[buffer(8)]],
    constant float &attention_scale [[buffer(9)]],
    constant uint &head_major [[buffer(10)]],
    constant uint &query_row [[buffer(11)]],
    constant uint &output_row [[buffer(12)]],
    constant uint &cache_origin_physical [[buffer(13)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (head >= n_heads || head_dim != 128u) return;
    uint kv_head = head / (n_heads / n_kv_heads);
    device const float *q = query + (query_row * n_heads + head) * head_dim;
    float running_max = -3.402823466e+38f;
    float denominator = 0.0f;
    float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint index = 0; index < visible; ++index) {
        uint row = head_major != 0u
            ? index
            : (cache_origin_physical + index) % capacity;
        uint base = head_major != 0u
            ? (kv_head * capacity + row) * head_dim
            : (row * n_kv_heads + kv_head) * head_dim;
        float score = 0.0f;
        for (uint chunk = 0; chunk < 4u; ++chunk) {
            uint dim = chunk * 32u + lane;
            score = fma(float(half(q[dim])), float(key_cache[base + dim]), score);
        }
        score += simd_shuffle_down(score, 16u);
        score += simd_shuffle_down(score, 8u);
        score += simd_shuffle_down(score, 4u);
        score += simd_shuffle_down(score, 2u);
        score += simd_shuffle_down(score, 1u);
        score = simd_broadcast_first(score);
        score *= attention_scale;
        float next_max = max(running_max, score);
        float old_factor = muser_cross_vendor_expf(running_max - next_max);
        float new_factor = muser_cross_vendor_expf(score - next_max);
        denominator = fma(denominator, old_factor, new_factor);
        for (uint chunk = 0; chunk < 4u; ++chunk) {
            uint dim = chunk * 32u + lane;
            float old_value = accumulator[chunk] * old_factor;
            accumulator[chunk] = fma(float(value_cache[base + dim]), new_factor, old_value);
        }
        running_max = next_max;
    }
    device float *destination = output + (output_row * n_heads + head) * head_dim;
    for (uint chunk = 0; chunk < 4u; ++chunk) {
        destination[chunk * 32u + lane] = accumulator[chunk] / denominator;
    }
}

// The same scalar graph as the decode oracle, dispatched over all query rows
// at once. This removes command-encoding overhead without changing a single
// arithmetic boundary or the chronological cache traversal order.
kernel void muser_cross_vendor_attention_prefill(
    device const float *query [[buffer(0)]],
    device const half *key_cache [[buffer(1)]],
    device const half *value_cache [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &old_visible [[buffer(4)]],
    constant uint &capacity [[buffer(5)]],
    constant uint &n_heads [[buffer(6)]],
    constant uint &n_kv_heads [[buffer(7)]],
    constant uint &head_dim [[buffer(8)]],
    constant float &attention_scale [[buffer(9)]],
    constant uint &head_major [[buffer(10)]],
    constant uint &sliding_window [[buffer(11)]],
    constant uint &token_count [[buffer(12)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint head = group.x;
    uint query_row = group.y;
    if (head >= n_heads || query_row >= token_count || head_dim != 128u) return;
    uint end = old_visible + query_row + 1u;
    uint first_cache_row = sliding_window == 0u || end <= sliding_window
        ? 0u
        : end - sliding_window;
    uint visible = end - first_cache_row;
    uint kv_head = head / (n_heads / n_kv_heads);
    device const float *q = query + (query_row * n_heads + head) * head_dim;
    float running_max = -3.402823466e+38f;
    float denominator = 0.0f;
    float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint index = 0; index < visible; ++index) {
        uint row = first_cache_row + index;
        uint base = head_major != 0u
            ? (kv_head * capacity + row) * head_dim
            : (row * n_kv_heads + kv_head) * head_dim;
        float score = 0.0f;
        for (uint chunk = 0; chunk < 4u; ++chunk) {
            uint dim = chunk * 32u + lane;
            score = fma(float(half(q[dim])), float(key_cache[base + dim]), score);
        }
        score += simd_shuffle_down(score, 16u);
        score += simd_shuffle_down(score, 8u);
        score += simd_shuffle_down(score, 4u);
        score += simd_shuffle_down(score, 2u);
        score += simd_shuffle_down(score, 1u);
        score = simd_broadcast_first(score);
        score *= attention_scale;
        float next_max = max(running_max, score);
        float old_factor = muser_cross_vendor_expf(running_max - next_max);
        float new_factor = muser_cross_vendor_expf(score - next_max);
        denominator = fma(denominator, old_factor, new_factor);
        for (uint chunk = 0; chunk < 4u; ++chunk) {
            uint dim = chunk * 32u + lane;
            float old_value = accumulator[chunk] * old_factor;
            accumulator[chunk] = fma(float(value_cache[base + dim]), new_factor, old_value);
        }
        running_max = next_max;
    }
    device float *destination = output + (query_row * n_heads + head) * head_dim;
    for (uint chunk = 0; chunk < 4u; ++chunk) {
        destination[chunk * 32u + lane] = accumulator[chunk] / denominator;
    }
}

kernel void muser_cross_vendor_sigmoid_gate(
    device float *values [[buffer(0)]],
    device const float *gate [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    float sigmoid = 1.0f / (1.0f + muser_cross_vendor_expf(-gate[index]));
    values[index] *= sigmoid;
}

kernel void muser_cross_vendor_dual_norm_residual(
    device float *hidden [[buffer(0)]],
    device const float *projected [[buffer(1)]],
    device float *output [[buffer(2)]],
    device const float *post_weight [[buffer(3)]],
    device const float *ffn_weight [[buffer(4)]],
    constant uint &dim [[buffer(5)]],
    constant float &post_eps [[buffer(6)]],
    constant float &ffn_eps [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]]) {
    uint offset = row * dim;
    float post_sum = 0.0f;
    for (uint index = 0; index < dim; ++index) {
        float value = projected[offset + index];
        post_sum = fma(value, value, post_sum);
    }
    float post_scale = 1.0f / sqrt(post_sum / float(dim) + post_eps);
    float ffn_sum = 0.0f;
    for (uint index = 0; index < dim; ++index) {
        float normalized = (projected[offset + index] * post_scale) * post_weight[index];
        float value = hidden[offset + index] + normalized;
        hidden[offset + index] = value;
        ffn_sum = fma(value, value, ffn_sum);
    }
    float ffn_scale = 1.0f / sqrt(ffn_sum / float(dim) + ffn_eps);
    for (uint index = 0; index < dim; ++index) {
        output[offset + index] = (hidden[offset + index] * ffn_scale) * ffn_weight[index];
    }
}

// Keep the residual addition in its own dispatch.  The pinned CUDA graph
// materializes the post-attention RMS output before adding it to the hidden
// state; combining those expressions lets the Metal compiler contract across
// a boundary that is observable in the retained cross-vendor oracle.
kernel void muser_cross_vendor_residual_add(
    device float *destination [[buffer(0)]],
    device const float *source [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    destination[index] = float(half(destination[index] + source[index]));
}

// Accepted Muse decode shape: 2 SIMD groups x 4 output rows. Adapted from
// the standalone Ferrite lineage, with the same Q4_K accumulation order.
kernel void muser_matvec_q4k_4r2s(
    device const uchar *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &cols [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    uint block_count = cols / 256;
    uint row_bytes = block_count * 144;
    uint base_row = group * 8 + simd * 4;
    if (base_row >= rows) return;
    uint active_rows = min(4u, rows - base_row);
    float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    device const uchar *row[4] = {
        weights + ulong(base_row) * ulong(row_bytes),
        weights + ulong(base_row + 1) * ulong(row_bytes),
        weights + ulong(base_row + 2) * ulong(row_bytes),
        weights + ulong(base_row + 3) * ulong(row_bytes)
    };
    for (uint block_index = 0; block_index < block_count; ++block_index) {
        for (uint row_index = 0; row_index < active_rows; ++row_index) {
            device const uchar *block = row[row_index] + block_index * 144;
            uint delta = *reinterpret_cast<device const uint *>(block);
            float d = float(as_type<half>(ushort(delta & 0xffff)));
            float dmin = float(as_type<half>(ushort(delta >> 16)));
            uint sd0 = *reinterpret_cast<device const uint *>(block + 4);
            uint sd1 = *reinterpret_cast<device const uint *>(block + 8);
            uint sd2 = *reinterpret_cast<device const uint *>(block + 12);
            float d_scale[8];
            float neg_min[8];
            muser_decode_all_q4k_scales(d, dmin, sd0, sd1, sd2, d_scale, neg_min);
            uint input_base = block_index * 256;
            for (uint quant_group = 0; quant_group < 4; ++quant_group) {
                uint packed = uint(block[16 + quant_group * 32 + lane]);
                float low = input[input_base + quant_group * 64 + lane];
                float high = input[input_base + quant_group * 64 + 32 + lane];
                accumulator[row_index] +=
                    fma(d_scale[quant_group * 2], float(packed & 0x0f), neg_min[quant_group * 2]) * low;
                accumulator[row_index] +=
                    fma(d_scale[quant_group * 2 + 1], float(packed >> 4), neg_min[quant_group * 2 + 1]) * high;
            }
        }
    }
    for (uint row_index = 0; row_index < active_rows; ++row_index) {
        accumulator[row_index] = simd_sum(accumulator[row_index]);
    }
    if (lane == 0) {
        for (uint row_index = 0; row_index < active_rows; ++row_index) {
            output[base_row + row_index] = accumulator[row_index];
        }
    }
}

kernel void muser_matvec_q5k_4sg(
    device const uchar *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &cols [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (row >= rows) return;
    uint block_count = cols / 256;
    device const uchar *row_data = weights + ulong(row) * ulong(block_count * 176);
    float accumulator = 0.0f;
    for (uint block_index = simd; block_index < block_count; block_index += 4) {
        device const uchar *block = row_data + block_index * 176;
        device const uchar *scales = block + 4;
        device const uchar *high_bits = block + 16;
        device const uchar *quants = block + 48;
        float d = muser_f16(block);
        float dmin = muser_f16(block + 2);
        for (uint element = lane; element < 256; element += 32) {
            uint group = element >> 6;
            uint half_group = (element >> 5) & 1;
            uint local = element & 31;
            uint scale_index = group * 2 + half_group;
            uchar2 scale_min = muser_scale_min(scales, scale_index);
            uint packed = uint(quants[group * 32 + local]);
            uint nibble = half_group == 0 ? packed & 0x0f : packed >> 4;
            uint high = (uint(high_bits[local]) >> (group * 2 + half_group)) & 1;
            uint quant = nibble | (high << 4);
            float value = d * float(scale_min.x) * float(quant) - dmin * float(scale_min.y);
            accumulator += value * input[block_index * 256 + element];
        }
    }
    accumulator = simd_sum(accumulator);
    threadgroup float partial[4];
    if (lane == 0) partial[simd] = accumulator;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0 && lane == 0) {
        output[row] = partial[0] + partial[1] + partial[2] + partial[3];
    }
}

inline float muser_dot_q4k(
    device const uchar *row,
    device const float *input,
    uint n_in) {
    float total = 0.0f;
    uint blocks = n_in / 256;
    for (uint block_index = 0; block_index < blocks; ++block_index) {
        device const uchar *block = row + block_index * 144;
        device const uchar *scales = block + 4;
        device const uchar *quants = block + 16;
        device const float *x = input + block_index * 256;
        float d = muser_f16(block);
        float dmin = muser_f16(block + 2);
        uint quant_offset = 0;
        uint scale_index = 0;
        uint base = 0;
        while (base < 256) {
            uchar2 low = muser_scale_min(scales, scale_index);
            uchar2 high = muser_scale_min(scales, scale_index + 1);
            float low_scale = d * float(low.x);
            float high_scale = d * float(high.x);
            float low_min = dmin * float(low.y);
            float high_min = dmin * float(high.y);
            for (uint lane = 0; lane < 32; ++lane) {
                uchar packed = quants[quant_offset + lane];
                total += (low_scale * float(packed & 0x0f) - low_min) * x[base + lane];
                total += (high_scale * float(packed >> 4) - high_min) * x[base + lane + 32];
            }
            quant_offset += 32;
            scale_index += 2;
            base += 64;
        }
    }
    return total;
}

inline float muser_dot_q5k(
    device const uchar *row,
    device const float *input,
    uint n_in) {
    float total = 0.0f;
    uint blocks = n_in / 256;
    for (uint block_index = 0; block_index < blocks; ++block_index) {
        device const uchar *block = row + block_index * 176;
        device const uchar *scales = block + 4;
        device const uchar *high_bits = block + 16;
        device const uchar *quants = block + 48;
        device const float *x = input + block_index * 256;
        float d = muser_f16(block);
        float dmin = muser_f16(block + 2);
        uint quant_offset = 0;
        uint scale_index = 0;
        uint base = 0;
        uchar low_mask = 1;
        uchar high_mask = 2;
        while (base < 256) {
            uchar2 low = muser_scale_min(scales, scale_index);
            uchar2 high = muser_scale_min(scales, scale_index + 1);
            float low_scale = d * float(low.x);
            float high_scale = d * float(high.x);
            float low_min = dmin * float(low.y);
            float high_min = dmin * float(high.y);
            for (uint lane = 0; lane < 32; ++lane) {
                uchar packed = quants[quant_offset + lane];
                uint low_quant = uint(packed & 0x0f) + ((high_bits[lane] & low_mask) ? 16 : 0);
                uint high_quant = uint(packed >> 4) + ((high_bits[lane] & high_mask) ? 16 : 0);
                total += (low_scale * float(low_quant) - low_min) * x[base + lane];
                total += (high_scale * float(high_quant) - high_min) * x[base + lane + 32];
            }
            quant_offset += 32;
            scale_index += 2;
            base += 64;
            low_mask <<= 2;
            high_mask <<= 2;
        }
    }
    return total;
}

kernel void muser_matmul_q4k(
    device const uchar *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &n_in [[buffer(3)]],
    constant uint &n_out [[buffer(4)]],
    constant uint &tokens [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    uint total = n_out * tokens;
    if (index < total) {
        uint row = index / tokens;
        uint token = index % tokens;
        uint row_bytes = (n_in / 256) * 144;
        output[token * n_out + row] = muser_dot_q4k(weights + row * row_bytes, input + token * n_in, n_in);
    }
}

kernel void muser_matmul_q5k(
    device const uchar *weights [[buffer(0)]],
    device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &n_in [[buffer(3)]],
    constant uint &n_out [[buffer(4)]],
    constant uint &tokens [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    uint total = n_out * tokens;
    if (index < total) {
        uint row = index / tokens;
        uint token = index % tokens;
        uint row_bytes = (n_in / 256) * 176;
        output[token * n_out + row] = muser_dot_q5k(weights + row * row_bytes, input + token * n_in, n_in);
    }
}

inline float muser_q4k_value(device const uchar *row, uint element) {
    uint block_index = element / 256;
    uint within_block = element % 256;
    uint group = within_block / 64;
    uint within_group = within_block % 64;
    uint scale_index = group * 2 + (within_group >= 32 ? 1 : 0);
    uint lane = within_group % 32;
    device const uchar *block = row + block_index * 144;
    uchar2 scale_min = muser_scale_min(block + 4, scale_index);
    uchar packed = block[16 + group * 32 + lane];
    uint quant = within_group < 32 ? uint(packed & 0x0f) : uint(packed >> 4);
    return muser_f16(block) * float(scale_min.x) * float(quant)
        - muser_f16(block + 2) * float(scale_min.y);
}

kernel void muser_embedding_q4k(
    device const uchar *weights [[buffer(0)]],
    device const uint *token_ids [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &hidden_dim [[buffer(3)]],
    constant uint &vocab_size [[buffer(4)]],
    constant uint &tokens [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    uint total = hidden_dim * tokens;
    if (index < total) {
        uint token_slot = index / hidden_dim;
        uint element = index % hidden_dim;
        uint token_id = min(token_ids[token_slot], vocab_size - 1);
        uint row_bytes = (hidden_dim / 256) * 144;
        output[index] = muser_q4k_value(weights + token_id * row_bytes, element);
    }
}

kernel void muser_kv_store_f16(
    device const float *key [[buffer(0)]],
    device const float *value [[buffer(1)]],
    device half *key_cache [[buffer(2)]],
    device half *value_cache [[buffer(3)]],
    constant uint &kv_dim [[buffer(4)]],
    constant uint &write_index [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if (index < kv_dim) {
        uint destination = write_index * kv_dim + index;
        key_cache[destination] = key[index];
        value_cache[destination] = value[index];
    }
}

kernel void muser_attention_decode_f32(
    device const float *query [[buffer(0)]],
    device const half *key_cache [[buffer(1)]],
    device const half *value_cache [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &n_heads [[buffer(4)]],
    constant uint &n_kv_heads [[buffer(5)]],
    constant uint &head_dim [[buffer(6)]],
    constant uint &position [[buffer(7)]],
    constant uint &capacity [[buffer(8)]],
    constant uint &origin_logical [[buffer(9)]],
    constant uint &origin_physical [[buffer(10)]],
    constant uint &window [[buffer(11)]],
    constant float &attention_scale [[buffer(12)]],
    uint head [[thread_position_in_grid]]) {
    if (head >= n_heads || head_dim != 128 || n_kv_heads == 0) {
        return;
    }
    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_head = head / heads_per_kv;
    uint visible = window > 0 ? min(position + 1, window) : position + 1;
    uint logical_start = position + 1 - visible;
    float running_max = -3.402823466e+38f;
    float denominator = 0.0f;
    float accumulator[128];
    for (uint dim = 0; dim < 128; ++dim) {
        accumulator[dim] = 0.0f;
    }
    device const float *q = query + head * 128;
    for (uint logical = logical_start; logical <= position; ++logical) {
        uint delta = logical - origin_logical;
        uint physical = (origin_physical + delta) % capacity;
        uint kv_base = physical * n_kv_heads * 128 + kv_head * 128;
        float score = 0.0f;
        for (uint dim = 0; dim < 128; ++dim) {
            score += q[dim] * key_cache[kv_base + dim];
        }
        score *= attention_scale;
        float next_max = max(running_max, score);
        float old_factor = exp(running_max - next_max);
        float new_factor = exp(score - next_max);
        denominator = denominator * old_factor + new_factor;
        for (uint dim = 0; dim < 128; ++dim) {
            accumulator[dim] = accumulator[dim] * old_factor
                + value_cache[kv_base + dim] * new_factor;
        }
        running_max = next_max;
    }
    uint output_base = head * 128;
    for (uint dim = 0; dim < 128; ++dim) {
        output[output_base + dim] = accumulator[dim] / denominator;
    }
}

// Decode FlashAttention producer. Logical 32-token blocks are distributed
// across workgroups and simdgroups; cache addresses always use explicit ring
// origins rather than absolute-position modulo placement. Each producer emits
// the online-softmax state [max, sum, weighted values].
kernel void muser_attention_decode_splitk_f16(
    device const float *query [[buffer(0)]],
    device const half *key_cache [[buffer(1)]],
    device const half *value_cache [[buffer(2)]],
    device float *partials [[buffer(3)]],
    constant uint &n_heads [[buffer(4)]],
    constant uint &n_kv_heads [[buffer(5)]],
    constant uint &position [[buffer(6)]],
    constant uint &capacity [[buffer(7)]],
    constant uint &origin_logical [[buffer(8)]],
    constant uint &origin_physical [[buffer(9)]],
    constant uint &window [[buffer(10)]],
    constant uint &n_workgroups [[buffer(11)]],
    constant uint &n_simdgroups [[buffer(12)]],
    constant float &attention_scale [[buffer(13)]],
    threadgroup float *shared [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    const uint head = group.x;
    const uint workgroup = group.y;
    if (head >= n_heads || simdgroup >= n_simdgroups) {
        return;
    }
    const uint head_dim = 128;
    const uint heads_per_kv = n_heads / n_kv_heads;
    const uint kv_head = head / heads_per_kv;
    const uint visible = window > 0 ? min(position + 1, window) : position + 1;
    const uint logical_start = position + 1 - visible;
    const uint block_count = (visible + 31) / 32;
    const uint vector_offset = lane * 4;
    if (simdgroup == 0) {
        *((threadgroup float4 *)(shared + vector_offset)) =
            *((device const float4 *)(query + head * head_dim + vector_offset));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float4 q = *((threadgroup float4 *)(shared + vector_offset));

    float running_max = -3.402823466e+38f;
    float running_sum = 0.0f;
    float4 accumulator = 0.0f;
    for (uint block = workgroup * n_simdgroups + simdgroup;
         block < block_count;
         block += n_workgroups * n_simdgroups) {
        const uint first_offset = block * 32;
        const uint count = min(32u, visible - first_offset);
        float scores[32];
        for (uint item = 0; item < count; ++item) {
            const uint logical = logical_start + first_offset + item;
            const uint physical =
                (origin_physical + logical - origin_logical) % capacity;
            const uint base = (physical * n_kv_heads + kv_head) * head_dim;
            const float4 key =
                float4(*((device const half4 *)(key_cache + base + vector_offset)));
            scores[item] = simd_sum(dot(q, key)) * attention_scale;
        }
        float block_max = scores[0];
        for (uint item = 1; item < count; ++item) {
            block_max = max(block_max, scores[item]);
        }
        const float next_max = max(running_max, block_max);
        const float old_factor = exp(running_max - next_max);
        accumulator *= old_factor;
        running_sum *= old_factor;
        running_max = next_max;
        for (uint item = 0; item < count; ++item) {
            const float weight = exp(scores[item] - running_max);
            running_sum += weight;
            const uint logical = logical_start + first_offset + item;
            const uint physical =
                (origin_physical + logical - origin_logical) % capacity;
            const uint base = (physical * n_kv_heads + kv_head) * head_dim;
            const float4 value =
                float4(*((device const half4 *)(value_cache + base + vector_offset)));
            accumulator += weight * value;
        }
    }

    const uint simd_stride = 2 + head_dim;
    threadgroup float *simd_partial =
        shared + head_dim + simdgroup * simd_stride;
    if (lane == 0) {
        simd_partial[0] = running_max;
        simd_partial[1] = running_sum;
    }
    *((threadgroup float4 *)(simd_partial + 2 + vector_offset)) = accumulator;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0) {
        float merged_max = -3.402823466e+38f;
        for (uint index = 0; index < n_simdgroups; ++index) {
            threadgroup float *part = shared + head_dim + index * simd_stride;
            merged_max = max(merged_max, part[0]);
        }
        float merged_sum = 0.0f;
        float4 merged_accumulator = 0.0f;
        for (uint index = 0; index < n_simdgroups; ++index) {
            threadgroup float *part = shared + head_dim + index * simd_stride;
            if (part[1] == 0.0f) {
                continue;
            }
            const float correction = exp(part[0] - merged_max);
            merged_sum += part[1] * correction;
            merged_accumulator += correction *
                *((threadgroup float4 *)(part + 2 + vector_offset));
        }
        const uint partial_stride = 2 + head_dim;
        device float *output = partials
            + (head * n_workgroups + workgroup) * partial_stride;
        if (lane == 0) {
            output[0] = merged_max;
            output[1] = merged_sum;
        }
        *((device float4 *)(output + 2 + vector_offset)) = merged_accumulator;
    }
}

kernel void muser_attention_decode_splitk_reduce_f32(
    device const float *partials [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &n_heads [[buffer(2)]],
    constant uint &n_workgroups [[buffer(3)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (head >= n_heads) {
        return;
    }
    const uint head_dim = 128;
    const uint partial_stride = 2 + head_dim;
    device const float *base = partials + head * n_workgroups * partial_stride;
    float global_max = -3.402823466e+38f;
    for (uint workgroup = 0; workgroup < n_workgroups; ++workgroup) {
        global_max = max(global_max, base[workgroup * partial_stride]);
    }
    float global_sum = 0.0f;
    float4 accumulator = 0.0f;
    const uint vector_offset = lane * 4;
    for (uint workgroup = 0; workgroup < n_workgroups; ++workgroup) {
        device const float *part = base + workgroup * partial_stride;
        if (part[1] == 0.0f) {
            continue;
        }
        const float correction = exp(part[0] - global_max);
        global_sum += part[1] * correction;
        accumulator += correction *
            *((device const float4 *)(part + 2 + vector_offset));
    }
    *((device float4 *)(output + head * head_dim + vector_offset)) =
        accumulator / global_sum;
}

kernel void muser_kv_store_batch_f16(
    device const float *key [[buffer(0)]],
    device const float *value [[buffer(1)]],
    device half *key_cache [[buffer(2)]],
    device half *value_cache [[buffer(3)]],
    constant uint &kv_dim [[buffer(4)]],
    constant uint &source_first [[buffer(5)]],
    constant uint &source_count [[buffer(6)]],
    constant uint &start_position [[buffer(7)]],
    constant uint &capacity [[buffer(8)]],
    constant uint &origin_logical [[buffer(9)]],
    constant uint &origin_physical [[buffer(10)]],
    constant uint &head_dim [[buffer(11)]],
    constant uint &head_major [[buffer(12)]],
    uint index [[thread_position_in_grid]]) {
    uint total = source_count * kv_dim;
    if (index < total) {
        uint source_token = source_first + index / kv_dim;
        uint element = index % kv_dim;
        uint logical = start_position + source_token;
        uint physical = (origin_physical + logical - origin_logical) % capacity;
        uint destination = physical * kv_dim + element;
        if (head_major != 0u) {
            uint kv_head = element / head_dim;
            uint dim = element % head_dim;
            destination = (kv_head * capacity + physical) * head_dim + dim;
        }
        uint source = source_token * kv_dim + element;
        key_cache[destination] = key[source];
        value_cache[destination] = value[source];
    }
}

// Materialize one explicit SWA ring followed by the current prefill chunk as
// a compact logical F16 tail. This lets the accepted FA2 kernel retain its
// contiguous matrix loads after the production ring wraps; ordinary cache
// placement remains governed solely by origin_logical/origin_physical.
kernel void muser_stage_swa_prefill_f16(
    device const float *current_key [[buffer(0)]],
    device const float *current_value [[buffer(1)]],
    device half *ring_key [[buffer(2)]],
    device half *ring_value [[buffer(3)]],
    device half *staged_key [[buffer(4)]],
    device half *staged_value [[buffer(5)]],
    constant uint &kv_dim [[buffer(6)]],
    constant uint &old_len [[buffer(7)]],
    constant uint &old_origin_physical [[buffer(8)]],
    constant uint &ring_capacity [[buffer(9)]],
    constant uint &token_count [[buffer(10)]],
    uint index [[thread_position_in_grid]]) {
    const uint rows = old_len + token_count;
    const uint total = rows * kv_dim;
    if (index >= total) {
        return;
    }
    const uint row = index / kv_dim;
    const uint element = index % kv_dim;
    if (row < old_len) {
        const uint physical = (old_origin_physical + row) % ring_capacity;
        const uint source = physical * kv_dim + element;
        staged_key[index] = ring_key[source];
        staged_value[index] = ring_value[source];
    } else {
        const uint source = (row - old_len) * kv_dim + element;
        const half key_value = current_key[source];
        const half value_value = current_value[source];
        staged_key[index] = key_value;
        staged_value[index] = value_value;
        const uint physical = (old_origin_physical + row) % ring_capacity;
        const uint destination = physical * kv_dim + element;
        ring_key[destination] = key_value;
        ring_value[destination] = value_value;
    }
}

// Recreate llama.cpp's absolute, padded KV address space from Muser's compact
// SWA ring. Masked rows keep arbitrary bytes; the mask preserves the exact
// reduction-lane placement used by the pinned flash_attn_ext_vec DAG.
kernel void muser_stage_swa_llama_decode_f16(
    device const float *current_key [[buffer(0)]],
    device const float *current_value [[buffer(1)]],
    device half *ring_key [[buffer(2)]],
    device half *ring_value [[buffer(3)]],
    device half *staged_key [[buffer(4)]],
    device half *staged_value [[buffer(5)]],
    device half *staged_mask [[buffer(6)]],
    constant uint &kv_dim [[buffer(7)]],
    constant uint &old_len [[buffer(8)]],
    constant uint &old_origin_logical [[buffer(9)]],
    constant uint &old_origin_physical [[buffer(10)]],
    constant uint &ring_capacity [[buffer(11)]],
    constant uint &position [[buffer(12)]],
    uint index [[thread_position_in_grid]]) {
    const uint rows = old_len + 1u;
    const uint total = rows * kv_dim;
    if (index >= total) {
        return;
    }
    const uint row = index / kv_dim;
    const uint element = index % kv_dim;
    const uint logical = row < old_len ? old_origin_logical + row : position;
    const uint destination = logical * kv_dim + element;
    if (row < old_len) {
        const uint physical = (old_origin_physical + row) % ring_capacity;
        const uint source = physical * kv_dim + element;
        staged_key[destination] = ring_key[source];
        staged_value[destination] = ring_value[source];
    } else {
        const half key_value = current_key[element];
        const half value_value = current_value[element];
        staged_key[destination] = key_value;
        staged_value[destination] = value_value;
        const uint physical = (old_origin_physical + old_len) % ring_capacity;
        const uint ring_destination = physical * kv_dim + element;
        ring_key[ring_destination] = key_value;
        ring_value[ring_destination] = value_value;
    }
    if (element == 0u) {
        const uint active_start = position + 1u - ring_capacity;
        staged_mask[logical] = logical >= active_start ? half(0.0h) : -INFINITY;
    }
}

kernel void muser_attention_prefill_f32(
    device const float *query [[buffer(0)]],
    device const float *current_key [[buffer(1)]],
    device const float *current_value [[buffer(2)]],
    device const half *key_cache [[buffer(3)]],
    device const half *value_cache [[buffer(4)]],
    device float *output [[buffer(5)]],
    constant uint &token_count [[buffer(6)]],
    constant uint &n_heads [[buffer(7)]],
    constant uint &n_kv_heads [[buffer(8)]],
    constant uint &head_dim [[buffer(9)]],
    constant uint &start_position [[buffer(10)]],
    constant uint &capacity [[buffer(11)]],
    constant uint &old_origin_logical [[buffer(12)]],
    constant uint &old_origin_physical [[buffer(13)]],
    constant uint &old_len [[buffer(14)]],
    constant uint &window [[buffer(15)]],
    constant float &attention_scale [[buffer(16)]],
    uint index [[thread_position_in_grid]]) {
    uint total = token_count * n_heads;
    if (index >= total || head_dim != 128 || n_kv_heads == 0) {
        return;
    }
    uint token = index / n_heads;
    uint head = index % n_heads;
    uint kv_head = head / (n_heads / n_kv_heads);
    uint position = start_position + token;
    uint visible = window > 0 ? min(position + 1, window) : position + 1;
    uint logical_start = position + 1 - visible;
    float running_max = -3.402823466e+38f;
    float denominator = 0.0f;
    float accumulator[128];
    for (uint dim = 0; dim < 128; ++dim) {
        accumulator[dim] = 0.0f;
    }
    device const float *q = query + (token * n_heads + head) * 128;
    for (uint logical = logical_start; logical <= position; ++logical) {
        float score = 0.0f;
        if (logical < start_position) {
            uint delta = logical - old_origin_logical;
            uint physical = (old_origin_physical + delta) % capacity;
            uint base = physical * n_kv_heads * 128 + kv_head * 128;
            for (uint dim = 0; dim < 128; ++dim) {
                score += q[dim] * float(key_cache[base + dim]);
            }
            score *= attention_scale;
            float next_max = max(running_max, score);
            float old_factor = exp(running_max - next_max);
            float new_factor = exp(score - next_max);
            denominator = denominator * old_factor + new_factor;
            for (uint dim = 0; dim < 128; ++dim) {
                accumulator[dim] = accumulator[dim] * old_factor
                    + float(value_cache[base + dim]) * new_factor;
            }
            running_max = next_max;
        } else {
            uint current = logical - start_position;
            uint base = current * n_kv_heads * 128 + kv_head * 128;
            for (uint dim = 0; dim < 128; ++dim) {
                score += q[dim] * current_key[base + dim];
            }
            score *= attention_scale;
            float next_max = max(running_max, score);
            float old_factor = exp(running_max - next_max);
            float new_factor = exp(score - next_max);
            denominator = denominator * old_factor + new_factor;
            for (uint dim = 0; dim < 128; ++dim) {
                accumulator[dim] = accumulator[dim] * old_factor
                    + current_value[base + dim] * new_factor;
            }
            running_max = next_max;
        }
    }
    uint output_base = (token * n_heads + head) * 128;
    for (uint dim = 0; dim < 128; ++dim) {
        output[output_base + dim] = accumulator[dim] / denominator;
    }
}

// Causal GQA prefill with one 128-thread group per (query token, query head).
// Each dimension owns its output accumulator while four SIMD groups
// cooperatively reduce Q.K. Prior rows are read from the explicit F16 ring;
// rows in this batch stay F32 until the attention result is complete.
kernel void muser_attention_prefill_flash_f16(
    device const float *query [[buffer(0)]],
    device const float *current_key [[buffer(1)]],
    device const float *current_value [[buffer(2)]],
    device const half *key_cache [[buffer(3)]],
    device const half *value_cache [[buffer(4)]],
    device float *output [[buffer(5)]],
    constant uint &token_count [[buffer(6)]],
    constant uint &n_heads [[buffer(7)]],
    constant uint &n_kv_heads [[buffer(8)]],
    constant uint &head_dim [[buffer(9)]],
    constant uint &start_position [[buffer(10)]],
    constant uint &capacity [[buffer(11)]],
    constant uint &old_origin_logical [[buffer(12)]],
    constant uint &old_origin_physical [[buffer(13)]],
    constant uint &old_len [[buffer(14)]],
    constant uint &window [[buffer(15)]],
    constant float &scale [[buffer(16)]],
    constant uint &head_major [[buffer(17)]],
    threadgroup float *simd_sums [[threadgroup(0)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort dim [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simd [[simdgroup_index_in_threadgroup]]) {
    if (group.x >= n_heads || group.y >= token_count || dim >= head_dim) {
        return;
    }
    uint head = group.x;
    uint token = group.y;
    uint query_position = start_position + token;
    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_head = head / heads_per_kv;
    uint query_index = (token * n_heads + head) * head_dim + dim;
    float q = query[query_index];
    float maximum = -INFINITY;
    float denominator = 0.0f;
    float numerator = 0.0f;
    uint first = window > 0 && query_position + 1 > window
        ? query_position + 1 - window
        : 0;
    first = max(first, old_origin_logical);

    for (uint logical = first; logical <= query_position; ++logical) {
        float key_value;
        float value_value;
        if (logical < start_position) {
            uint old_offset = logical - old_origin_logical;
            uint physical = (old_origin_physical + old_offset) % capacity;
            uint cache_index = (physical * n_kv_heads + kv_head) * head_dim + dim;
            if (head_major != 0u) {
                cache_index = (kv_head * capacity + physical) * head_dim + dim;
            }
            key_value = float(key_cache[cache_index]);
            value_value = float(value_cache[cache_index]);
        } else {
            uint current = logical - start_position;
            uint current_index = (current * n_kv_heads + kv_head) * head_dim + dim;
            key_value = current_key[current_index];
            value_value = current_value[current_index];
        }

        float partial = simd_sum(q * key_value);
        if (lane == 0) {
            simd_sums[simd] = partial;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0) {
            float value = lane < 4 ? simd_sums[lane] : 0.0f;
            value = simd_sum(value);
            if (lane == 0) {
                simd_sums[0] = value;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float score = simd_sums[0] * scale;
        float next_maximum = max(maximum, score);
        float old_weight = isinf(maximum) ? 0.0f : exp(maximum - next_maximum);
        float new_weight = exp(score - next_maximum);
        numerator = numerator * old_weight + value_value * new_weight;
        denominator = denominator * old_weight + new_weight;
        maximum = next_maximum;
        // Prevent the first SIMD group from publishing the next reduction
        // before every dimension has consumed this score.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    output[query_index] = numerator / denominator;
    (void) old_len;
}

kernel void muser_copy_row_f32(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &row_width [[buffer(2)]],
    constant uint &row [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < row_width) {
        output[index] = input[row * row_width + index];
    }
}

// Causal f16 mask for the pinned llama.cpp `kernel_flash_attn_ext` prefill
// route: row q covers absolute KV columns [0, start_position + q]; columns
// past the diagonal are -INF exactly as ggml's causal mask produces. One
// thread per mask element over the 2D (column, row) grid.
kernel void muser_fa_causal_mask_f16(
    device half *mask [[buffer(0)]],
    constant uint &start_position [[buffer(1)]],
    constant uint &token_count [[buffer(2)]],
    constant uint &visible [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]) {
    const uint column = tid.x;
    const uint row = tid.y;
    if (column >= visible || row >= token_count) {
        return;
    }
    mask[ulong(row) * visible + column] =
        column <= start_position + row ? half(0.0h) : half(-INFINITY);
}
