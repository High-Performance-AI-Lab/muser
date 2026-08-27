// flash_attn_decode_vec.metal — Single-token decode attention with Split-K.
//
// Adapted from ggml-org/llama.cpp flash_attn_ext_vec approach, rewritten for
// Ferrite's Q8_0 KV cache layout (34-byte blocks: f16 scale + 32×i8 quants).
//
// Attribution: algorithm structure adapted from ggml-org/llama.cpp,
// ggml/src/ggml-metal/ggml-metal.metal (flash_attn_ext_vec).
//
// Architecture:
//   - Q=1: single query token (decode-only)
//   - Split-K: nwg workgroups split the KV sequence dimension per head
//   - Each workgroup computes partial online softmax + weighted V sum
//   - A separate reduce kernel combines nwg partial results
//   - 1 SIMD per threadgroup (32 threads), each thread handles head_dim/32 dims
//
// Grid:  (n_heads, nwg, 1)
// TG:    (32, 1, 1)
//
// Partial output buffer layout per (head, wg):
//   [f32 max | f32 exp_sum | f32 out[head_dim]]
//   Stride per workgroup = 2 + head_dim (in f32 elements)

#include <metal_stdlib>
using namespace metal;

constant uint FC_DK [[function_constant(40)]];
constant bool HAS_FC_DK = is_function_constant_defined(FC_DK);
constant uint HELIX_BAND_PROBE [[function_constant(44)]];
constant bool HAS_HELIX_BAND_PROBE = is_function_constant_defined(HELIX_BAND_PROBE);
constant bool FC_Q8_TILE_C8 [[function_constant(45)]];
constant bool HAS_FC_Q8_TILE_C8 = is_function_constant_defined(FC_Q8_TILE_C8);
constant bool USE_Q8_TILE_C8 = HAS_FC_Q8_TILE_C8 && FC_Q8_TILE_C8;
constant bool FC_Q8_OFFSET_ONCE [[function_constant(46)]];
constant bool HAS_FC_Q8_OFFSET_ONCE = is_function_constant_defined(FC_Q8_OFFSET_ONCE);
constant bool USE_Q8_OFFSET_ONCE = HAS_FC_Q8_OFFSET_ONCE && FC_Q8_OFFSET_ONCE;

// Required-Q8 specialization of the shipping paged block kernel.  The caller
// performs one standalone shared quantizer store plus a device barrier; this
// specialization compiles out the fused re-store, reads every token (including
// `pos`) from Q8, and never drops a small V numerator contribution.
constant bool FC_REQUIRED_Q8_CACHEONLY [[function_constant(94)]];
constant bool HAS_FC_REQUIRED_Q8_CACHEONLY =
    is_function_constant_defined(FC_REQUIRED_Q8_CACHEONLY);
constant bool REQUIRED_Q8_CACHEONLY =
    HAS_FC_REQUIRED_Q8_CACHEONLY && FC_REQUIRED_Q8_CACHEONLY;

// Schedule-only F16 GQA split-K experiment. Slot 98 is separate from the
// shipping paged FA-decode NSG specialization at slot 93.
constant uint FC_F16_INTERLEAVED_NSG [[function_constant(98)]];

// ── MK-1: DecodeParams buffer (slot 18) ───────────────────────────────────
//
// When USE_DECODE_PARAMS_BUF is true (function_constant 92), the kernels read
// pos, nwg, and chunk_size from decode_params [[buffer(18)]] instead of from
// per-dispatch setBytes uniforms. This eliminates two setBytes calls per layer
// per decode step in the frozen replay path.
//
// Only the three hot-path non-paged kernels support this variant:
//   flash_attn_decode_vec_q8, flash_attn_decode_vec_q8_v3,
//   flash_attn_decode_reduce_v2
//
// NwgGrid (slot 12, dispatch height patch) is still kept as a setBytes call.
struct DecodeParams {
    uint pos;
    uint seq_len;
    uint nwg;
    uint chunk_size;
};
constant bool USE_DECODE_PARAMS_BUF [[function_constant(92)]];
constant bool HAS_USE_DECODE_PARAMS_BUF = is_function_constant_defined(USE_DECODE_PARAMS_BUF);
constant bool BIND_DECODE_PARAMS = HAS_USE_DECODE_PARAMS_BUF && USE_DECODE_PARAMS_BUF;
constant uint GEOPRECISION_Q8_ENCODING_F16_DENSE = 1u;
constant uint GEOPRECISION_Q8_ENCODING_Q8_0_COMPACT = 100u;

struct GeoPrecisionRuntimeFrameHeader {
    uint abi_version;
    uint head_dim;
    uint n_q8;
    uint n_q4;
    uint q4_encoding;
    uint q4_transform;
    uint q4_scale_encoding;
    uint q8_positions_offset_bytes;
    uint q8_k_offset_bytes;
    uint q8_v_offset_bytes;
    uint q4_positions_offset_bytes;
    uint q4_k_offset_bytes;
    uint q4_v_offset_bytes;
    uint q4_k_scales_offset_bytes;
    uint q4_v_scales_offset_bytes;
    uint total_bytes;
};

static inline float geop_q4_dense_value(
    device const uchar* row,
    float scale,
    uint d,
    uint lid)
{
    const ushort pair_lane = ushort(lid & ~1u);
    uchar packed_src = 0u;
    if ((lid & 1u) == 0u) {
        packed_src = row[d >> 1u];
    }
    uchar packed = simd_shuffle(packed_src, pair_lane);
    uint nibble = ((lid & 1u) != 0u) ? (packed >> 4u) : (packed & 0xFu);
    return float(int(nibble) - 8) * scale;
}

static inline float4 geop_load_f32x4_or_zero(
    device const float* row,
    uint off,
    uint DK)
{
    if (off >= DK) {
        return float4(0.0f);
    }
    if (off + 3u < DK) {
        return *((device const float4*)(row + off));
    }
    return float4(
        row[off],
        (off + 1u < DK) ? row[off + 1u] : 0.0f,
        (off + 2u < DK) ? row[off + 2u] : 0.0f,
        (off + 3u < DK) ? row[off + 3u] : 0.0f
    );
}

static inline float4 geop_load_f16x4_or_zero(
    device const half* row,
    uint off,
    uint DK)
{
    if (off >= DK) {
        return float4(0.0f);
    }
    if (off + 3u < DK) {
        return float4(*((device const half4*)(row + off)));
    }
    return float4(
        float(row[off]),
        (off + 1u < DK) ? float(row[off + 1u]) : 0.0f,
        (off + 2u < DK) ? float(row[off + 2u]) : 0.0f,
        (off + 3u < DK) ? float(row[off + 3u]) : 0.0f
    );
}

static inline float4 geop_load_q8_0_x4_or_zero(
    device const uchar* row,
    uint off,
    uint DK)
{
    if (off >= DK) {
        return float4(0.0f);
    }
    if (off + 3u < DK && (off & 31u) <= 28u) {
        device const uchar* block = row + (off / 32u) * 34u;
        const float scale = float(*((device const half*)block));
        device const int8_t* quants = (device const int8_t*)(block + 2u);
        const uint lane = off & 31u;
        return float4(
            float(quants[lane]) * scale,
            float(quants[lane + 1u]) * scale,
            float(quants[lane + 2u]) * scale,
            float(quants[lane + 3u]) * scale
        );
    }

    float vals[4];
    for (uint i = 0u; i < 4u; i++) {
        const uint d = off + i;
        if (d < DK) {
            device const uchar* block = row + (d / 32u) * 34u;
            const float scale = float(*((device const half*)block));
            device const int8_t* quants = (device const int8_t*)(block + 2u);
            vals[i] = float(quants[d % 32u]) * scale;
        } else {
            vals[i] = 0.0f;
        }
    }
    return float4(vals[0], vals[1], vals[2], vals[3]);
}

static inline float4 geop_load_q4_dense_x4_or_zero(
    device const uchar* row,
    float scale,
    uint off,
    uint DK)
{
    if (off >= DK) {
        return float4(0.0f);
    }

    const uint byte_off = off >> 1u;
    const uchar packed0 = row[byte_off];
    const uchar packed1 = (off + 2u < DK) ? row[byte_off + 1u] : 0u;

    return float4(
        float(int(packed0 & 0xFu) - 8) * scale,
        (off + 1u < DK) ? float(int((packed0 >> 4u) & 0xFu) - 8) * scale : 0.0f,
        (off + 2u < DK) ? float(int(packed1 & 0xFu) - 8) * scale : 0.0f,
        (off + 3u < DK) ? float(int((packed1 >> 4u) & 0xFu) - 8) * scale : 0.0f
    );
}

static inline void geop_store_f32x4_clamped(
    device float* dst,
    float4 value,
    uint off,
    uint DK)
{
    if (off >= DK) {
        return;
    }
    if (off + 3u < DK) {
        *((device float4*)(dst + off)) = value;
        return;
    }
    dst[off] = value.x;
    if (off + 1u < DK) dst[off + 1u] = value.y;
    if (off + 2u < DK) dst[off + 2u] = value.z;
    if (off + 3u < DK) dst[off + 3u] = value.w;
}

// ── flash_attn_decode_vec_q8 ──────────────────────────────────────────────
//
// Single-token decode attention against Q8_0 KV cache.
// One threadgroup per (head, workgroup).
//
kernel void flash_attn_decode_vec_q8(
    device const float* Q               [[ buffer(0) ]],   // [n_heads, head_dim]
    device       uchar* K_cache         [[ buffer(1) ]],   // [n_kv_heads, q8_stride]
    device       uchar* V_cache         [[ buffer(2) ]],   // [n_kv_heads, q8_stride]
    device       float* partials        [[ buffer(3) ]],   // [n_heads, nwg, 2+head_dim]
    device const float* k_cur           [[ buffer(4) ]],   // [n_kv_heads, head_dim] current K
    device const float* v_cur           [[ buffer(5) ]],   // [n_kv_heads, head_dim] current V
    constant     uint&  head_dim_arg    [[ buffer(6) ]],
    constant     uint&  pos_arg         [[ buffer(7) ]],   // current position (ignored when BIND_DECODE_PARAMS)
    constant     uint&  max_seq         [[ buffer(8) ]],
    constant     uint&  heads_per_kv    [[ buffer(9) ]],
    constant     uint&  sliding_window  [[ buffer(10) ]],
    constant     uint&  q8_cache_stride [[ buffer(11) ]],  // bytes per KV head
    constant     uint&  nwg_arg         [[ buffer(12) ]],  // number of workgroups per head
    constant     uint&  chunk_size_arg  [[ buffer(13) ]],  // KV tokens per workgroup
    constant DecodeParams* decode_params [[ buffer(18) ]], // MK-1: params buf (active when BIND_DECODE_PARAMS)
    uint3 tgid [[ threadgroup_position_in_grid ]],
    uint  lid  [[ thread_index_in_simdgroup ]])
{
    const uint head = tgid.x;           // query head index
    const uint wg   = tgid.y;           // workgroup index within this head
    const uint kv_head = head / heads_per_kv;

    const uint DK = HAS_FC_DK ? FC_DK : head_dim_arg;
    const uint bpr = (DK + 31u) / 32u;  // Q8_0 blocks per row
    const uint pos = BIND_DECODE_PARAMS ? decode_params->pos : pos_arg;
    const uint seq = pos + 1u;           // total tokens
    const uint nwg = BIND_DECODE_PARAMS ? decode_params->nwg : nwg_arg;
    const uint chunk_size = BIND_DECODE_PARAMS ? decode_params->chunk_size : chunk_size_arg;

    // ── Step 0: Quantise k_cur / v_cur → Q8_0 and store in cache ──────
    // Only one TG per KV-head does the store (first Q-head, workgroup 0).
    // No barrier needed: attention reads pos from k_cur/v_cur, not cache.
    // Cache at pos is only read by future decode steps.
    if (wg == 0u && head % heads_per_kv == 0u) {
        device uchar* K_store = K_cache + kv_head * q8_cache_stride;
        device uchar* V_store = V_cache + kv_head * q8_cache_stride;
        device const float* k_src = k_cur + kv_head * DK;
        device const float* v_src = v_cur + kv_head * DK;

        for (uint b = lid; b < bpr; b += 32u) {
            const uint base  = b * 32u;
            const uint elems = min(32u, DK - base);

            // K block
            float amax_k = 0.0f;
            for (uint j = 0u; j < elems; ++j)
                amax_k = max(amax_k, abs(k_src[base + j]));
            float d_k     = amax_k / 127.0f;
            float inv_d_k = (d_k > 0.0f) ? (1.0f / d_k) : 0.0f;
            device uchar* kb = K_store + pos * bpr * 34u + b * 34u;
            *((device half*)kb) = half(d_k);
            device int8_t* kq = (device int8_t*)(kb + 2u);
            for (uint j = 0u; j < elems; ++j)
                kq[j] = (char)clamp((int)round(k_src[base + j] * inv_d_k), -127, 127);

            // V block
            float amax_v = 0.0f;
            for (uint j = 0u; j < elems; ++j)
                amax_v = max(amax_v, abs(v_src[base + j]));
            float d_v     = amax_v / 127.0f;
            float inv_d_v = (d_v > 0.0f) ? (1.0f / d_v) : 0.0f;
            device uchar* vb = V_store + pos * bpr * 34u + b * 34u;
            *((device half*)vb) = half(d_v);
            device int8_t* vq = (device int8_t*)(vb + 2u);
            for (uint j = 0u; j < elems; ++j)
                vq[j] = (char)clamp((int)round(v_src[base + j] * inv_d_v), -127, 127);
        }
    }

    // Determine KV range for this workgroup
    const uint kv_start = wg * chunk_size;
    const uint kv_end = min(kv_start + chunk_size, seq);

    // Partial output destination
    const uint partial_stride = 2u + DK;
    device float* my_partial = partials + head * nwg * partial_stride + wg * partial_stride;

    if (kv_start >= seq) {
        // No work for this workgroup — write sentinel
        if (lid == 0u) {
            my_partial[0] = -INFINITY;  // max
            my_partial[1] = 0.0f;       // exp_sum
        }
        for (uint d = lid; d < DK; d += 32u) {
            my_partial[2u + d] = 0.0f;
        }
        return;
    }

    // Load Q vector into registers (DK/32 elements per thread)
    device const float* q_head = Q + head * DK;
    float q_reg[4];
    for (uint b = 0u; b < bpr; b++) {
        q_reg[b] = q_head[b * 32u + lid];
    }

    const float scale = rsqrt(float(DK));

    // Sliding window: determine effective start
    uint effective_start = kv_start;
    if (sliding_window > 0u && seq > sliding_window) {
        uint win_start = seq - sliding_window;
        effective_start = max(kv_start, win_start);
    }

    // KV cache base pointers for this KV head
    device const uchar* K_base = K_cache + kv_head * q8_cache_stride;
    device const uchar* V_base = V_cache + kv_head * q8_cache_stride;

    // Online softmax state
    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float acc[4];  // partial output (DK/32 elements per thread)
    for (uint b = 0u; b < bpr; b++) acc[b] = 0.0f;

    // Process each KV token in this workgroup's chunk
    for (uint t = effective_start; t < kv_end; t++) {
        // Compute dot(Q, K[t])
        float score = 0.0f;

        if (t == pos) {
            // Current token: read from k_cur (not yet stored in cache)
            device const float* k_ptr = k_cur + kv_head * DK;
            for (uint b = 0u; b < bpr; b++) {
                score += q_reg[b] * k_ptr[b * 32u + lid];
            }
        } else {
            // Cached token: dequantize Q8_0
            for (uint b = 0u; b < bpr; b++) {
                device const uchar* kb = K_base + t * bpr * 34u + b * 34u;
                float k_scale = float(*((device const half*)kb));
                device const int8_t* k_quants = (device const int8_t*)(kb + 2u);
                score += q_reg[b] * float(k_quants[lid]) * k_scale;
            }
        }
        score = simd_sum(score) * scale;

        // Mask tokens before sliding window
        if (sliding_window > 0u && t < effective_start) {
            continue;
        }

        // Online softmax update with numerical stability correction
        float new_max = max(running_max, score);
        float exp_score = precise::exp(score - new_max);
        float correction = precise::exp(running_max - new_max);
        running_sum = running_sum * correction + exp_score;

        // Sparse V: skip V dequant+accumulation for negligible positions.
        // Softmax state is already updated above; just apply correction to
        // existing accumulator so the output scale stays correct.
        if (exp_score < 1e-6f) {
            for (uint b = 0u; b < bpr; b++) acc[b] *= correction;
            running_max = new_max;
            continue;
        }

        // Accumulate V[t] with correction
        if (t == pos) {
            device const float* v_ptr = v_cur + kv_head * DK;
            for (uint b = 0u; b < bpr; b++) {
                acc[b] = acc[b] * correction + exp_score * v_ptr[b * 32u + lid];
            }
        } else {
            for (uint b = 0u; b < bpr; b++) {
                device const uchar* vb = V_base + t * bpr * 34u + b * 34u;
                float v_scale = float(*((device const half*)vb));
                device const int8_t* v_quants = (device const int8_t*)(vb + 2u);
                acc[b] = acc[b] * correction + exp_score * float(v_quants[lid]) * v_scale;
            }
        }
        running_max = new_max;
    }

    // Write partial results
    if (lid == 0u) {
        my_partial[0] = running_max;
        my_partial[1] = running_sum;
    }
    for (uint b = 0u; b < bpr; b++) {
        my_partial[2u + b * 32u + lid] = acc[b];
    }
}

// ── flash_attn_decode_vec_geoprecision_dense ───────────────────────────────
//
// Direct GeoPrecision MVP route: historical KV rows come from the per-head
// GeoPrecision shadow-frame slab under a dense split contract:
//   q8_positions == [0 .. n_q8-1]
//   q4_positions == [n_q8 .. n_q8+n_q4-1]
//   n_q8 + n_q4 == pos
// The current token still reads from k_cur/v_cur, matching the existing decode
// ABI and avoiding an extra host roundtrip before the shadow frame is updated.
kernel void flash_attn_decode_vec_geoprecision_dense(
    device const float* Q                [[ buffer(0) ]],
    device       uchar* geop_frames      [[ buffer(1) ]],
    device       float* partials         [[ buffer(2) ]],
    device const float* k_cur            [[ buffer(3) ]],
    device const float* v_cur            [[ buffer(4) ]],
    constant     uint&  head_dim_arg     [[ buffer(5) ]],
    constant     uint&  pos              [[ buffer(6) ]],
    constant     uint&  frame_stride     [[ buffer(7) ]],
    constant     uint&  heads_per_kv     [[ buffer(8) ]],
    constant     uint&  sliding_window   [[ buffer(9) ]],
    constant     uint&  nwg_arg          [[ buffer(10) ]],
    constant     uint&  chunk_size_arg   [[ buffer(11) ]],
    device       half*  f16_k_cache      [[ buffer(12) ]],
    device       half*  f16_v_cache      [[ buffer(13) ]],
    constant     uint&  f16_cache_stride [[ buffer(14) ]],
    uint3 tgid [[ threadgroup_position_in_grid ]],
    uint  lid  [[ thread_index_in_simdgroup ]])
{
    const uint head = tgid.x;
    const uint wg   = tgid.y;
    const uint kv_head = head / heads_per_kv;
    const uint DK = HAS_FC_DK ? FC_DK : head_dim_arg;
    const uint bpr = (DK + 31u) / 32u;
    const uint seq = pos + 1u;
    const uint nwg = nwg_arg;
    const uint chunk_size = chunk_size_arg;
    const uint bytes_per_row = (DK + 1u) / 2u;

    const uint partial_stride = 2u + DK;
    device float* my_partial = partials + head * nwg * partial_stride + wg * partial_stride;

    if (wg == 0u && head % heads_per_kv == 0u) {
        device half* k_store = f16_k_cache + kv_head * f16_cache_stride + pos * DK;
        device half* v_store = f16_v_cache + kv_head * f16_cache_stride + pos * DK;
        device const float* k_src = k_cur + kv_head * DK;
        device const float* v_src = v_cur + kv_head * DK;
        for (uint b = 0u; b < bpr; b++) {
            const uint d = b * 32u + lid;
            if (d < DK) {
                k_store[d] = half(k_src[d]);
                v_store[d] = half(v_src[d]);
            }
        }
    }

    if (wg * chunk_size >= seq) {
        if (lid == 0u) {
            my_partial[0] = -INFINITY;
            my_partial[1] = 0.0f;
        }
        for (uint d = lid; d < DK; d += 32u) {
            my_partial[2u + d] = 0.0f;
        }
        return;
    }

    device const uchar* frame_base = geop_frames + kv_head * frame_stride;
    device const GeoPrecisionRuntimeFrameHeader* header =
        reinterpret_cast<device const GeoPrecisionRuntimeFrameHeader*>(frame_base);
    const bool geop_compact_q8 =
        header->q4_encoding == GEOPRECISION_Q8_ENCODING_Q8_0_COMPACT && header->n_q4 == 0u;

    if (header->n_q8 + header->n_q4 != pos
        || header->head_dim != DK
        || (header->q4_encoding != GEOPRECISION_Q8_ENCODING_F16_DENSE && !geop_compact_q8)
        || header->q4_transform != 0u
        || header->q4_scale_encoding != 1u) {
        if (lid == 0u) {
            my_partial[0] = -INFINITY;
            my_partial[1] = 0.0f;
        }
        for (uint d = lid; d < DK; d += 32u) {
            my_partial[2u + d] = 0.0f;
        }
        return;
    }

    const uint kv_start = wg * chunk_size;
    const uint kv_end = min(kv_start + chunk_size, seq);

    device const float* q_head = Q + head * DK;
    float q_reg[8];
    for (uint b = 0u; b < bpr; b++) {
        const uint d = b * 32u + lid;
        q_reg[b] = (d < DK) ? q_head[d] : 0.0f;
    }

    const float attn_scale = rsqrt(float(DK));
    uint effective_start = kv_start;
    if (sliding_window > 0u && seq > sliding_window) {
        uint win_start = seq - sliding_window;
        effective_start = max(kv_start, win_start);
    }

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float acc[8];
    for (uint b = 0u; b < bpr; b++) {
        acc[b] = 0.0f;
    }

    for (uint t = effective_start; t < kv_end; t++) {
        float score = 0.0f;
        if (t == pos) {
            device const float* k_ptr = k_cur + kv_head * DK;
            for (uint b = 0u; b < bpr; b++) {
                const uint d = b * 32u + lid;
                if (d < DK) {
                    score += q_reg[b] * k_ptr[d];
                }
            }
        } else if (t < header->n_q8) {
            if (geop_compact_q8) {
                device const uchar* q8_k =
                    frame_base + header->q8_k_offset_bytes + t * bpr * 34u;
                for (uint b = 0u; b < bpr; b++) {
                    const uint d = b * 32u + lid;
                    if (d < DK) {
                        device const uchar* kb = q8_k + b * 34u;
                        float k_scale = float(*((device const half*)kb));
                        device const int8_t* k_quants = (device const int8_t*)(kb + 2u);
                        score += q_reg[b] * float(k_quants[lid]) * k_scale;
                    }
                }
            } else {
                device const half* q8_k =
                    reinterpret_cast<device const half*>(frame_base + header->q8_k_offset_bytes)
                    + t * DK;
                for (uint b = 0u; b < bpr; b++) {
                    const uint d = b * 32u + lid;
                    if (d < DK) {
                        score += q_reg[b] * float(q8_k[d]);
                    }
                }
            }
        } else {
            const uint q4_row = t - header->n_q8;
            device const uchar* k_row =
                frame_base + header->q4_k_offset_bytes + q4_row * bytes_per_row;
            float k_scale = float(*reinterpret_cast<device const half*>(
                frame_base + header->q4_k_scales_offset_bytes + q4_row * 2u));
            for (uint b = 0u; b < bpr; b++) {
                const uint d = b * 32u + lid;
                if (d < DK) {
                    float k_val = geop_q4_dense_value(k_row, k_scale, d, lid);
                    score += q_reg[b] * k_val;
                }
            }
        }
        score = simd_sum(score) * attn_scale;

        float new_max = max(running_max, score);
        float exp_score = precise::exp(score - new_max);
        float correction = precise::exp(running_max - new_max);
        running_sum = running_sum * correction + exp_score;

        if (exp_score < 1e-6f) {
            for (uint b = 0u; b < bpr; b++) {
                acc[b] *= correction;
            }
            running_max = new_max;
            continue;
        }

        if (t == pos) {
            device const float* v_ptr = v_cur + kv_head * DK;
            for (uint b = 0u; b < bpr; b++) {
                const uint d = b * 32u + lid;
                if (d < DK) {
                    acc[b] = acc[b] * correction + exp_score * v_ptr[d];
                }
            }
        } else if (t < header->n_q8) {
            if (geop_compact_q8) {
                device const uchar* q8_v =
                    frame_base + header->q8_v_offset_bytes + t * bpr * 34u;
                for (uint b = 0u; b < bpr; b++) {
                    const uint d = b * 32u + lid;
                    if (d < DK) {
                        device const uchar* vb = q8_v + b * 34u;
                        float v_scale = float(*((device const half*)vb));
                        device const int8_t* v_quants = (device const int8_t*)(vb + 2u);
                        acc[b] = acc[b] * correction + exp_score * float(v_quants[lid]) * v_scale;
                    }
                }
            } else {
                device const half* q8_v =
                    reinterpret_cast<device const half*>(frame_base + header->q8_v_offset_bytes)
                    + t * DK;
                for (uint b = 0u; b < bpr; b++) {
                    const uint d = b * 32u + lid;
                    if (d < DK) {
                        acc[b] = acc[b] * correction + exp_score * float(q8_v[d]);
                    }
                }
            }
        } else {
            const uint q4_row = t - header->n_q8;
            device const uchar* v_row =
                frame_base + header->q4_v_offset_bytes + q4_row * bytes_per_row;
            float v_scale = float(*reinterpret_cast<device const half*>(
                frame_base + header->q4_v_scales_offset_bytes + q4_row * 2u));
            for (uint b = 0u; b < bpr; b++) {
                const uint d = b * 32u + lid;
                if (d < DK) {
                    float v_val = geop_q4_dense_value(v_row, v_scale, d, lid);
                    acc[b] = acc[b] * correction + exp_score * v_val;
                }
            }
        }
        running_max = new_max;
    }

    if (lid == 0u) {
        my_partial[0] = running_max;
        my_partial[1] = running_sum;
    }
    for (uint b = 0u; b < bpr; b++) {
        const uint d = b * 32u + lid;
        if (d < DK) {
            my_partial[2u + d] = acc[b];
        }
    }
}


// ── flash_attn_decode_vec_geoprecision_dense_gqa ───────────────────────────
//
// GQA-shaped direct GeoPrecision dense-split route. Mirrors the
// flash_attn_decode_vec_q8_gqa dispatch geometry:
//   grid = (n_kv_heads, nwg, heads_per_kv)
//   TG   = (32, 1, 1)
// while consuming the GeoPrecision dense frame ABI directly.
