kernel void flash_attn_decode_vec_f16_v2(
    device const float* Q               [[ buffer(0) ]],
    device       half*  K_cache         [[ buffer(1) ]],
    device       half*  V_cache         [[ buffer(2) ]],
    device       float* partials        [[ buffer(3) ]],
    device const float* k_cur           [[ buffer(4) ]],
    device const float* v_cur           [[ buffer(5) ]],
    constant     uint&  head_dim_arg    [[ buffer(6) ]],
    constant     uint&  pos             [[ buffer(7) ]],
    constant     uint&  max_seq         [[ buffer(8) ]],
    constant     uint&  heads_per_kv    [[ buffer(9) ]],
    constant     uint&  sliding_window  [[ buffer(10) ]],
    constant     uint&  f16_stride_arg  [[ buffer(11) ]],
    constant     uint&  nwg_arg         [[ buffer(12) ]],
    constant     uint&  chunk_size_arg  [[ buffer(13) ]],
    constant     float& attn_scale_arg  [[ buffer(14) ]],
    device const uchar* kv_mask         [[ buffer(15) ]],
    uint3 tgid [[ threadgroup_position_in_grid ]],
    uint  lid  [[ thread_index_in_simdgroup ]])
{
    const uint head = tgid.x;
    const uint wg   = tgid.y;
    const uint kv_head = head / heads_per_kv;
    const uint DK = HAS_FC_DK ? FC_DK : head_dim_arg;
    // N_VEC: number of float4 chunks per thread. 1 for DK≤128, 2 for 256, 4 for 512.
    const uint N_VEC = (DK + 127u) / 128u;
    const uint seq = pos + 1u;
    const uint nwg = nwg_arg;
    const uint chunk_size = chunk_size_arg;
    const uint f16_stride = f16_stride_arg;

    // F16 KV store: N_VEC×4 contiguous elements per thread
    if (wg == 0u && head % heads_per_kv == 0u) {
        device half* K_s = K_cache + kv_head * f16_stride + pos * DK;
        device half* V_s = V_cache + kv_head * f16_stride + pos * DK;
        device const float* ks = k_cur + kv_head * DK;
        device const float* vs = v_cur + kv_head * DK;
        for (uint v = 0u; v < N_VEC; v++) {
            uint off = v * 128u + lid * 4u;
            if (off < DK) {
                *((device half4*)(K_s + off)) = half4(*((device const float4*)(ks + off)));
                *((device half4*)(V_s + off)) = half4(*((device const float4*)(vs + off)));
            }
        }
    }

    const uint kv_start = wg * chunk_size;
    const uint kv_end = min(kv_start + chunk_size, seq);
    const uint partial_stride = 2u + DK;
    device float* my_partial = partials + head * nwg * partial_stride + wg * partial_stride;

    if (kv_start >= seq) {
        if (lid == 0u) { my_partial[0] = -INFINITY; my_partial[1] = 0.0f; }
        for (uint v = 0u; v < N_VEC; v++) {
            uint off = v * 128u + lid * 4u;
            if (off < DK) *((device float4*)(my_partial + 2u + off)) = float4(0.0f);
        }
        return;
    }

    // Q: N_VEC contiguous float4 vectors per thread
    float4 q_vec[4];  // max DK=512 → N_VEC=4
    for (uint v = 0u; v < N_VEC; v++) {
        uint off = v * 128u + lid * 4u;
        q_vec[v] = (off < DK) ? *((device const float4*)(Q + head * DK + off)) : float4(0.0f);
    }

    const float scale = (attn_scale_arg != 0.0f) ? attn_scale_arg : rsqrt(float(DK));

    uint eff_start = kv_start;
    if (sliding_window > 0u && seq > sliding_window)
        eff_start = max(kv_start, seq - sliding_window);

    device const half* K_head = K_cache + kv_head * f16_stride;
    device const half* V_head = V_cache + kv_head * f16_stride;

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float4 acc[4];
    for (uint v = 0u; v < N_VEC; v++) acc[v] = 0.0f;

    for (uint t = eff_start; t < kv_end; t++) {
        if (kv_mask != nullptr && t != pos && kv_mask[kv_head * max_seq + t] == 0u) continue;
        float score = 0.0f;
        if (t == pos) {
            for (uint v = 0u; v < N_VEC; v++) {
                uint off = v * 128u + lid * 4u;
                if (off < DK) {
                    float4 k_vec = *((device const float4*)(k_cur + kv_head * DK + off));
                    score += dot(q_vec[v], k_vec);
                }
            }
        } else {
            for (uint v = 0u; v < N_VEC; v++) {
                uint off = v * 128u + lid * 4u;
                if (off < DK) {
                    half4 k_h4 = *((device const half4*)(K_head + t * DK + off));
                    score += dot(q_vec[v], (float4)k_h4);
                }
            }
        }
        score = simd_sum(score) * scale;

        if (sliding_window > 0u && t < eff_start) continue;

        float new_max = max(running_max, score);
        float exp_score = precise::exp(score - new_max);
        float correction = precise::exp(running_max - new_max);
        running_sum = running_sum * correction + exp_score;

        // Sparse V: skip V read for negligible attention weights
        if (exp_score < 1e-6f) {
            for (uint v = 0u; v < N_VEC; v++) acc[v] *= correction;
            running_max = new_max;
            continue;
        }

        if (t == pos) {
            for (uint v = 0u; v < N_VEC; v++) {
                uint off = v * 128u + lid * 4u;
                if (off < DK) {
                    float4 v_vec = *((device const float4*)(v_cur + kv_head * DK + off));
                    acc[v] = acc[v] * correction + exp_score * v_vec;
                }
            }
        } else {
            for (uint v = 0u; v < N_VEC; v++) {
                uint off = v * 128u + lid * 4u;
                if (off < DK) {
                    half4 v_h4 = *((device const half4*)(V_head + t * DK + off));
                    acc[v] = acc[v] * correction + exp_score * (float4)v_h4;
                } else {
                    acc[v] *= correction;
                }
            }
        }
        running_max = new_max;
    }

    if (lid == 0u) { my_partial[0] = running_max; my_partial[1] = running_sum; }
    for (uint v = 0u; v < N_VEC; v++) {
        uint off = v * 128u + lid * 4u;
        if (off < DK) *((device float4*)(my_partial + 2u + off)) = acc[v];
    }
}

// ── flash_attn_decode_vec_f16_gqa ────────────────────────────────────────
//
// GQA-aware F16 Split-K decode attention with 3D grid.
//
// Grid: (n_kv_heads, nwg, heads_per_kv) — KV reads shared by Q-heads.
// Each Q-head loads its own Q vector, but sibling Q-heads (tgid.z = 0..heads_per_kv-1)
// share KV reads from the same KV head through L1 cache.
//
// Memory layout: F16 cache is contiguous half values, stride = max_seq * DK per KV head.
// Partial buffer: indexed by Q-head (kv_head * heads_per_kv + tgid.z) for reduce_v2.
// KV store: only wg==0 && tgid.z==0 writes (first Q-head in group).
//
// Performance target: reduce F16 decode O(n) coefficient from 0.34 to ~0.05-0.15 µs/token
// by eliminating redundant KV reads in GQA models (7 Q-heads per KV-head in Qwen 2.5 7B).
kernel void flash_attn_decode_vec_f16_gqa(
    device const float* Q               [[ buffer(0) ]],
    device       half*  K_cache         [[ buffer(1) ]],
    device       half*  V_cache         [[ buffer(2) ]],
    device       float* partials        [[ buffer(3) ]],
    device const float* k_cur           [[ buffer(4) ]],
    device const float* v_cur           [[ buffer(5) ]],
    constant     uint&  head_dim_arg    [[ buffer(6) ]],
    constant     uint&  pos             [[ buffer(7) ]],
    constant     uint&  max_seq         [[ buffer(8) ]],
    constant     uint&  heads_per_kv    [[ buffer(9) ]],
    constant     uint&  sliding_window  [[ buffer(10) ]],
    constant     uint&  f16_stride_arg  [[ buffer(11) ]],
    constant     uint&  nwg_arg         [[ buffer(12) ]],
    constant     uint&  chunk_size_arg  [[ buffer(13) ]],
    constant     float& attn_scale_arg  [[ buffer(14) ]],
    constant     uint&  cache_only_arg  [[ buffer(15) ]],
    device const uchar* kv_mask         [[ buffer(16) ]],
    uint3 tgid [[ threadgroup_position_in_grid ]],
    uint  lid  [[ thread_index_in_simdgroup ]])
{
    const uint kv_head = tgid.x;
    const uint wg      = tgid.y;
    const uint head    = kv_head * heads_per_kv + tgid.z;

    const uint DK = HAS_FC_DK ? FC_DK : head_dim_arg;
    const uint N_VEC = (DK + 127u) / 128u;
    const uint seq = pos + 1u;
    const uint nwg = nwg_arg;
    const uint chunk_size = chunk_size_arg;
    const uint f16_stride = f16_stride_arg;
    const bool cache_only = (cache_only_arg != 0u);

    // F16 KV store: skip when cache_only (shared-KV layers reuse source cache)
    if (!cache_only && wg == 0u && tgid.z == 0u) {
        device half* K_s = K_cache + kv_head * f16_stride + pos * DK;
        device half* V_s = V_cache + kv_head * f16_stride + pos * DK;
        device const float* ks = k_cur + kv_head * DK;
        device const float* vs = v_cur + kv_head * DK;
        for (uint v = 0u; v < N_VEC; v++) {
            uint off = v * 128u + lid * 4u;
            if (off < DK) {
                *((device half4*)(K_s + off)) = half4(*((device const float4*)(ks + off)));
                *((device half4*)(V_s + off)) = half4(*((device const float4*)(vs + off)));
            }
        }
    }

    const uint kv_start = wg * chunk_size;
    const uint kv_end = min(kv_start + chunk_size, seq);
    const uint partial_stride = 2u + DK;
    device float* my_partial = partials + head * nwg * partial_stride + wg * partial_stride;

    if (kv_start >= seq) {
        if (lid == 0u) { my_partial[0] = -INFINITY; my_partial[1] = 0.0f; }
        for (uint v = 0u; v < N_VEC; v++) {
            uint off = v * 128u + lid * 4u;
            if (off < DK) *((device float4*)(my_partial + 2u + off)) = float4(0.0f);
        }
        return;
    }

    float4 q_vec[4];
    for (uint v = 0u; v < N_VEC; v++) {
        uint off = v * 128u + lid * 4u;
        q_vec[v] = (off < DK) ? *((device const float4*)(Q + head * DK + off)) : float4(0.0f);
    }

    const float scale = (attn_scale_arg != 0.0f) ? attn_scale_arg : rsqrt(float(DK));

    uint eff_start = kv_start;
    if (sliding_window > 0u && seq > sliding_window)
        eff_start = max(kv_start, seq - sliding_window);

    device const half* K_head = K_cache + kv_head * f16_stride;
    device const half* V_head = V_cache + kv_head * f16_stride;

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float4 acc[4];
    for (uint v = 0u; v < N_VEC; v++) acc[v] = 0.0f;

    for (uint t = eff_start; t < kv_end; t++) {
        if (kv_mask != nullptr && t != pos && kv_mask[kv_head * max_seq + t] == 0u) continue;
        float score = 0.0f;
        if (!cache_only && t == pos) {
            for (uint v = 0u; v < N_VEC; v++) {
                uint off = v * 128u + lid * 4u;
                if (off < DK) {
                    float4 k_vec = *((device const float4*)(k_cur + kv_head * DK + off));
                    score += dot(q_vec[v], k_vec);
                }
            }
        } else {
            for (uint v = 0u; v < N_VEC; v++) {
                uint off = v * 128u + lid * 4u;
                if (off < DK) {
                    half4 k_h4 = *((device const half4*)(K_head + t * DK + off));
                    score += dot(q_vec[v], (float4)k_h4);
                }
            }
        }
        score = simd_sum(score) * scale;

        if (sliding_window > 0u && t < eff_start) continue;

        float new_max = max(running_max, score);
        float exp_score = precise::exp(score - new_max);
        float correction = precise::exp(running_max - new_max);
        running_sum = running_sum * correction + exp_score;

        // Sparse V: skip V read for negligible attention weights
        if (exp_score < 1e-6f) {
            for (uint v = 0u; v < N_VEC; v++) acc[v] *= correction;
            running_max = new_max;
            continue;
        }

        if (!cache_only && t == pos) {
            for (uint v = 0u; v < N_VEC; v++) {
                uint off = v * 128u + lid * 4u;
                if (off < DK) {
                    float4 v_vec = *((device const float4*)(v_cur + kv_head * DK + off));
                    acc[v] = acc[v] * correction + exp_score * v_vec;
                }
            }
        } else {
            for (uint v = 0u; v < N_VEC; v++) {
                uint off = v * 128u + lid * 4u;
                if (off < DK) {
                    half4 v_h4 = *((device const half4*)(V_head + t * DK + off));
                    acc[v] = acc[v] * correction + exp_score * (float4)v_h4;
                } else {
                    acc[v] *= correction;
                }
            }
        }
        running_max = new_max;
    }

    if (lid == 0u) { my_partial[0] = running_max; my_partial[1] = running_sum; }
    for (uint v = 0u; v < N_VEC; v++) {
        uint off = v * 128u + lid * 4u;
        if (off < DK) *((device float4*)(my_partial + 2u + off)) = acc[v];
    }
}

constant uint ATTN_CHUNK_F16 = 32u;

// ── flash_attn_decode_vec_f16_gqa_v2 ─────────────────────────────────────
//
// GQA F16 decode attention with C=32 block-wise online softmax.
// Drop-in replacement for flash_attn_decode_vec_f16_gqa: same buffer layout,
// same 3D grid (n_kv_heads, nwg, heads_per_kv), same cache_only + kv_mask.
// Reduces simd_sum barriers from O(seq/nwg) to O(seq/(nwg*32)) — one
// rescale per 32-token block instead of per token.
// Masked tokens (kv_mask==0) receive score=-INF so they contribute zero
// weight; all-masked blocks are skipped to avoid exp(-INF-(-INF)) = NaN.
//
kernel void flash_attn_decode_vec_f16_gqa_v2(
    device const float* Q               [[ buffer(0) ]],
    device       half*  K_cache         [[ buffer(1) ]],
    device       half*  V_cache         [[ buffer(2) ]],
    device       float* partials        [[ buffer(3) ]],
    device const float* k_cur           [[ buffer(4) ]],
    device const float* v_cur           [[ buffer(5) ]],
    constant     uint&  head_dim_arg    [[ buffer(6) ]],
    constant     uint&  pos             [[ buffer(7) ]],
    constant     uint&  max_seq         [[ buffer(8) ]],
    constant     uint&  heads_per_kv    [[ buffer(9) ]],
    constant     uint&  sliding_window  [[ buffer(10) ]],
    constant     uint&  f16_stride_arg  [[ buffer(11) ]],
    constant     uint&  nwg_arg         [[ buffer(12) ]],
    constant     uint&  chunk_size_arg  [[ buffer(13) ]],
    constant     float& attn_scale_arg  [[ buffer(14) ]],
    constant     uint&  cache_only_arg  [[ buffer(15) ]],
    device const uchar* kv_mask         [[ buffer(16) ]],
    uint3 tgid [[ threadgroup_position_in_grid ]],
    uint  lid  [[ thread_index_in_simdgroup ]])
{
    const uint kv_head = tgid.x;
    const uint wg      = tgid.y;
    const uint head    = kv_head * heads_per_kv + tgid.z;

    const uint DK = HAS_FC_DK ? FC_DK : head_dim_arg;
    const uint N_VEC = (DK + 127u) / 128u;
    const uint seq = pos + 1u;
    const uint nwg = nwg_arg;
    const uint chunk_size = chunk_size_arg;
    const uint f16_stride = f16_stride_arg;
    const bool cache_only = (cache_only_arg != 0u);

    // F16 KV store: skip when cache_only (shared-KV layers reuse source cache)
    if (!cache_only && wg == 0u && tgid.z == 0u) {
        device half* K_s = K_cache + kv_head * f16_stride + pos * DK;
        device half* V_s = V_cache + kv_head * f16_stride + pos * DK;
        device const float* ks = k_cur + kv_head * DK;
        device const float* vs = v_cur + kv_head * DK;
        for (uint v = 0u; v < N_VEC; v++) {
            uint off = v * 128u + lid * 4u;
            if (off < DK) {
                *((device half4*)(K_s + off)) = half4(*((device const float4*)(ks + off)));
                *((device half4*)(V_s + off)) = half4(*((device const float4*)(vs + off)));
            }
        }
    }

    const uint kv_start = wg * chunk_size;
    const uint kv_end = min(kv_start + chunk_size, seq);
    const uint partial_stride = 2u + DK;
    device float* my_partial = partials + head * nwg * partial_stride + wg * partial_stride;

    if (kv_start >= seq) {
        if (lid == 0u) { my_partial[0] = -INFINITY; my_partial[1] = 0.0f; }
        for (uint v = 0u; v < N_VEC; v++) {
            uint off = v * 128u + lid * 4u;
            if (off < DK) *((device float4*)(my_partial + 2u + off)) = float4(0.0f);
        }
        return;
    }

    float4 q_vec[4];
    for (uint v = 0u; v < N_VEC; v++) {
        uint off = v * 128u + lid * 4u;
        q_vec[v] = (off < DK) ? *((device const float4*)(Q + head * DK + off)) : float4(0.0f);
    }

    const float scale = (attn_scale_arg != 0.0f) ? attn_scale_arg : rsqrt(float(DK));

    uint eff_start = kv_start;
    if (sliding_window > 0u && seq > sliding_window)
        eff_start = max(kv_start, seq - sliding_window);

    device const half* K_head = K_cache + kv_head * f16_stride;
    device const half* V_head = V_cache + kv_head * f16_stride;

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float4 acc[4];
    for (uint v = 0u; v < N_VEC; v++) acc[v] = 0.0f;

    // ── Block-wise online softmax: process ATTN_CHUNK_F16 tokens per iter ──
    for (uint chunk_base = eff_start; chunk_base < kv_end; chunk_base += ATTN_CHUNK_F16) {
        const uint chunk_end = min(chunk_base + ATTN_CHUNK_F16, kv_end);
        const uint nc = chunk_end - chunk_base;

        // Phase 1: compute nc scores; kv_mask-evicted tokens get -INF
        // All simdgroup threads iterate the same t so simd_sum is always
        // called collectively (all threads take the same masked/valid path).
        float scores[ATTN_CHUNK_F16];
        for (uint j = 0u; j < nc; j++) {
            const uint t = chunk_base + j;
            if (kv_mask != nullptr && t != pos && kv_mask[kv_head * max_seq + t] == 0u) {
                scores[j] = -INFINITY;
                continue;
            }
            float score = 0.0f;
            if (!cache_only && t == pos) {
                for (uint v = 0u; v < N_VEC; v++) {
                    uint off = v * 128u + lid * 4u;
                    if (off < DK) {
                        float4 k_vec = *((device const float4*)(k_cur + kv_head * DK + off));
                        score += dot(q_vec[v], k_vec);
                    }
                }
            } else {
                for (uint v = 0u; v < N_VEC; v++) {
                    uint off = v * 128u + lid * 4u;
                    if (off < DK) {
                        half4 k_h4 = *((device const half4*)(K_head + t * DK + off));
                        score += dot(q_vec[v], (float4)k_h4);
                    }
                }
            }
            scores[j] = simd_sum(score) * scale;
        }

        // Phase 2: block max — skip all-masked blocks to avoid NaN from
        // exp(-INF - (-INF)) when running_max is also still -INF.
        float block_max = scores[0];
        for (uint j = 1u; j < nc; j++)
            block_max = max(block_max, scores[j]);
        if (block_max == -INFINITY) continue;

        float new_max = max(running_max, block_max);
        float correction = precise::exp(running_max - new_max);
        for (uint v = 0u; v < N_VEC; v++)
            acc[v] *= correction;
        running_sum *= correction;
        running_max = new_max;

        // Phase 3: accumulate V (sparse V skip covers -INF masked tokens)
        for (uint j = 0u; j < nc; j++) {
            float w = precise::exp(scores[j] - running_max);
            running_sum += w;
            if (w < 1e-6f) continue;

            const uint t = chunk_base + j;
            if (!cache_only && t == pos) {
                for (uint v = 0u; v < N_VEC; v++) {
                    uint off = v * 128u + lid * 4u;
                    if (off < DK) {
                        float4 v_vec = *((device const float4*)(v_cur + kv_head * DK + off));
                        acc[v] += w * v_vec;
                    }
                }
            } else {
                for (uint v = 0u; v < N_VEC; v++) {
                    uint off = v * 128u + lid * 4u;
                    if (off < DK) {
                        half4 v_h4 = *((device const half4*)(V_head + t * DK + off));
                        acc[v] += w * (float4)v_h4;
                    }
                }
            }
        }
    }

    if (lid == 0u) { my_partial[0] = running_max; my_partial[1] = running_sum; }
    for (uint v = 0u; v < N_VEC; v++) {
        uint off = v * 128u + lid * 4u;
        if (off < DK) *((device float4*)(my_partial + 2u + off)) = acc[v];
    }
}


// ── flash_attn_decode_vec_f16_gqa_interleaved ──────────────────────────
//
// Schedule-only interleaved sibling of gqa_v2. The external ABI remains
// [head][wg][M,S,O], but the grid is (n_heads,nwg,1), making sibling Q heads
// launch-adjacent. Each threadgroup has FC_F16_INTERLEAVED_NSG simdgroups that
// own 32-token blocks round-robin and merge into one legacy external partial.
// Dynamic threadgroup floats: Q[DK] | simd_partial[NSG][M,S,O[DK]].
kernel void flash_attn_decode_vec_f16_gqa_interleaved(
    device const float* Q               [[ buffer(0) ]],
    device       half*  K_cache         [[ buffer(1) ]],
    device       half*  V_cache         [[ buffer(2) ]],
    device       float* partials        [[ buffer(3) ]],
    device const float* k_cur           [[ buffer(4) ]],
    device const float* v_cur           [[ buffer(5) ]],
    constant     uint&  head_dim_arg    [[ buffer(6) ]],
    constant     uint&  pos             [[ buffer(7) ]],
    constant     uint&  max_seq         [[ buffer(8) ]],
    constant     uint&  heads_per_kv    [[ buffer(9) ]],
    constant     uint&  sliding_window  [[ buffer(10) ]],
    constant     uint&  f16_stride_arg  [[ buffer(11) ]],
    constant     uint&  nwg_arg         [[ buffer(12) ]],
    constant     uint&  chunk_size_arg  [[ buffer(13) ]],
    constant     float& attn_scale_arg  [[ buffer(14) ]],
    constant     uint&  cache_only_arg  [[ buffer(15) ]],
    device const uchar* kv_mask         [[ buffer(16) ]],
    threadgroup float* shared           [[ threadgroup(0) ]],
    uint3 tgid [[ threadgroup_position_in_grid ]],
    uint  lid  [[ thread_index_in_simdgroup ]],
    uint  sgid [[ simdgroup_index_in_threadgroup ]])
{
    const uint head = tgid.x;
    const uint wg = tgid.y;
    const uint kv_head = head / heads_per_kv;
    const uint DK = HAS_FC_DK ? FC_DK : head_dim_arg;
    const uint NSG = FC_F16_INTERLEAVED_NSG;
    const uint seq = pos + 1u;
    const uint nwg = nwg_arg;
    const uint f16_stride = f16_stride_arg;
    const bool cache_only = (cache_only_arg != 0u);
    (void)chunk_size_arg;

    const uint off = lid * 4u;
    if (sgid == 0u) {
        *((threadgroup float4*)(shared + off)) =
            *((device const float4*)(Q + head * DK + off));
    }

    // Restrict the fused store to one simdgroup, workgroup, and Q head per KV
    // group. Every workgroup still reads the current token from k_cur/v_cur.
    if (!cache_only && wg == 0u && sgid == 0u && head % heads_per_kv == 0u) {
        device half* K_s = K_cache + kv_head * f16_stride + pos * DK;
        device half* V_s = V_cache + kv_head * f16_stride + pos * DK;
        device const float* ks = k_cur + kv_head * DK;
        device const float* vs = v_cur + kv_head * DK;
        *((device half4*)(K_s + off)) = half4(*((device const float4*)(ks + off)));
        *((device half4*)(V_s + off)) = half4(*((device const float4*)(vs + off)));
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float4 q_vec = *((threadgroup float4*)(shared + off));
    const float scale =
        (attn_scale_arg != 0.0f) ? attn_scale_arg : rsqrt(float(DK));
    device const half* K_head = K_cache + kv_head * f16_stride;
    device const half* V_head = V_cache + kv_head * f16_stride;

    const uint exact_active_start =
        (sliding_window > 0u && seq > sliding_window)
            ? (seq - sliding_window)
            : 0u;
    const uint first_active_aligned_block = exact_active_start / ATTN_CHUNK_F16;
    const uint end_block = (seq + ATTN_CHUNK_F16 - 1u) / ATTN_CHUNK_F16;

    float running_max = -FLT_MAX / 2.0f;
    float running_sum = 0.0f;
    float4 acc = 0.0f;

    for (uint relative_block = wg * NSG + sgid;
         first_active_aligned_block + relative_block < end_block;
         relative_block += nwg * NSG) {
        const uint block = first_active_aligned_block + relative_block;
        const uint token_start = block * ATTN_CHUNK_F16;
        const uint token_end = min(token_start + ATTN_CHUNK_F16, seq);
        const uint nc = token_end - token_start;

        float scores[ATTN_CHUNK_F16];
        for (uint j = 0u; j < nc; j++) {
            const uint t = token_start + j;
            // Mask the prefix of SWA's aligned first block before reading K/V.
            if (t < exact_active_start ||
                (kv_mask != nullptr && t != pos &&
                 kv_mask[kv_head * max_seq + t] == 0u)) {
                scores[j] = -INFINITY;
                continue;
            }

            float score;
            if (!cache_only && t == pos) {
                score = dot(q_vec,
                            *((device const float4*)(k_cur + kv_head * DK + off)));
            } else {
                score = dot(q_vec,
                            (float4)(*((device const half4*)(K_head + t * DK + off))));
            }
            scores[j] = simd_sum(score) * scale;
        }

        float block_max = scores[0];
        for (uint j = 1u; j < nc; j++) block_max = max(block_max, scores[j]);
        if (block_max == -INFINITY) continue;

        const float new_max = max(running_max, block_max);
        const float correction = precise::exp(running_max - new_max);
        acc *= correction;
        running_sum *= correction;
        running_max = new_max;

        for (uint j = 0u; j < nc; j++) {
            const float w = precise::exp(scores[j] - running_max);
            running_sum += w;
            if (w < 1e-6f) continue;

            const uint t = token_start + j;
            if (!cache_only && t == pos) {
                acc += w * (*((device const float4*)(v_cur + kv_head * DK + off)));
            } else {
                acc += w * (float4)(*((device const half4*)(V_head + t * DK + off)));
            }
        }
    }

    const uint simd_partial_stride = 2u + DK;
    threadgroup float* simd_partial = shared + DK + sgid * simd_partial_stride;
    if (lid == 0u) {
        simd_partial[0] = running_max;
        simd_partial[1] = running_sum;
    }
    simd_partial[2u + off + 0u] = acc[0];
    simd_partial[2u + off + 1u] = acc[1];
    simd_partial[2u + off + 2u] = acc[2];
    simd_partial[2u + off + 3u] = acc[3];

    // Empty simdgroups publish the finite empty state and still participate.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgid == 0u) {
        float merged_max = -FLT_MAX / 2.0f;
        for (uint s = 0u; s < NSG; s++) {
            threadgroup float* p = shared + DK + s * simd_partial_stride;
            merged_max = max(merged_max, p[0]);
        }

        float merged_sum = 0.0f;
        float4 merged_acc = 0.0f;
        for (uint s = 0u; s < NSG; s++) {
            threadgroup float* p = shared + DK + s * simd_partial_stride;
            if (p[1] == 0.0f) continue;
            const float correction = precise::exp(p[0] - merged_max);
            merged_sum += p[1] * correction;
            merged_acc += correction * float4(
                p[2u + off + 0u], p[2u + off + 1u],
                p[2u + off + 2u], p[2u + off + 3u]);
        }

        const uint partial_stride = 2u + DK;
        device float* my_partial =
            partials + head * nwg * partial_stride + wg * partial_stride;
        if (lid == 0u) {
            my_partial[0] = merged_max;
            my_partial[1] = merged_sum;
        }
        *((device float4*)(my_partial + 2u + off)) = merged_acc;
    }
}


// ── flash_attn_decode_vec_f16_gqa_ilp4 ───────────────────────────────────
//
// DK=128-specialized ILP variant of flash_attn_decode_vec_f16_gqa_v2.
// Identical grid (n_kv_heads, nwg, heads_per_kv), buffer ABI, partial layout
// ([head][wg][(M, S, O[128])] → flash_attn_decode_reduce_v2), fused KV store,
// sliding window, kv_mask and cache_only semantics.
//
// The one change is the Phase-1 score loop. v2 computes, per position:
//   load K (one half4 per lane) → dot → full-width simd_sum → next position
// i.e. a load→reduce→load→reduce serial chain with ~1 outstanding cacheline
// per simdgroup. Measured on Muse 32K decode (muse-decode-profile,
// 2026-08-11): 251 GB/s amplified vs llama's 2241 on the same amplification —
// latency-bound, not bandwidth-bound. Here the score loop is batched 4 wide:
// 4 positions' partial dots are computed back-to-back (4 independent K loads
// in flight per lane) before the 4 reductions issue. Phase 3 (V accumulate)
// is already dependency-free across positions and stays as in v2.
//
kernel void flash_attn_decode_vec_f16_gqa_ilp4(
    device const float* Q               [[ buffer(0) ]],
    device       half*  K_cache         [[ buffer(1) ]],
    device       half*  V_cache         [[ buffer(2) ]],
    device       float* partials        [[ buffer(3) ]],
    device const float* k_cur           [[ buffer(4) ]],
    device const float* v_cur           [[ buffer(5) ]],
    constant     uint&  head_dim_arg    [[ buffer(6) ]],
    constant     uint&  pos             [[ buffer(7) ]],
    constant     uint&  max_seq         [[ buffer(8) ]],
    constant     uint&  heads_per_kv    [[ buffer(9) ]],
    constant     uint&  sliding_window  [[ buffer(10) ]],
    constant     uint&  f16_stride_arg  [[ buffer(11) ]],
    constant     uint&  nwg_arg         [[ buffer(12) ]],
    constant     uint&  chunk_size_arg  [[ buffer(13) ]],
    constant     float& attn_scale_arg  [[ buffer(14) ]],
    constant     uint&  cache_only_arg  [[ buffer(15) ]],
    device const uchar* kv_mask         [[ buffer(16) ]],
    uint3 tgid [[ threadgroup_position_in_grid ]],
    uint  lid  [[ thread_index_in_simdgroup ]])
{
    const uint kv_head = tgid.x;
    const uint wg      = tgid.y;
    const uint head    = kv_head * heads_per_kv + tgid.z;

    const uint DK = 128u;               // specialization; dispatch guards this
    const uint seq = pos + 1u;
    const uint nwg = nwg_arg;
    const uint chunk_size = chunk_size_arg;
    const uint f16_stride = f16_stride_arg;
    const bool cache_only = (cache_only_arg != 0u);

    // Fused F16 KV store (identical to v2): one TG per kv_head writes pos.
    if (!cache_only && wg == 0u && tgid.z == 0u) {
        device half* K_s = K_cache + kv_head * f16_stride + pos * DK;
        device half* V_s = V_cache + kv_head * f16_stride + pos * DK;
        device const float* ks = k_cur + kv_head * DK;
        device const float* vs = v_cur + kv_head * DK;
        uint off = lid * 4u;
        *((device half4*)(K_s + off)) = half4(*((device const float4*)(ks + off)));
        *((device half4*)(V_s + off)) = half4(*((device const float4*)(vs + off)));
    }

    const uint kv_start = wg * chunk_size;
    const uint kv_end = min(kv_start + chunk_size, seq);
    const uint partial_stride = 2u + DK;
    device float* my_partial = partials + head * nwg * partial_stride + wg * partial_stride;

    if (kv_start >= seq) {
        if (lid == 0u) { my_partial[0] = -INFINITY; my_partial[1] = 0.0f; }
        *((device float4*)(my_partial + 2u + lid * 4u)) = float4(0.0f);
        return;
    }

    const uint off = lid * 4u;
    const float4 q_vec = *((device const float4*)(Q + head * DK + off));
    const float scale = (attn_scale_arg != 0.0f) ? attn_scale_arg : rsqrt(float(DK));

    uint eff_start = kv_start;
    if (sliding_window > 0u && seq > sliding_window)
        eff_start = max(kv_start, seq - sliding_window);

    device const half* K_head = K_cache + kv_head * f16_stride;
    device const half* V_head = V_cache + kv_head * f16_stride;
    // k_cur is always bound on this route (cache_only=false); hoisting the
    // current-token K row out of the loop is safe and free.
    const float4 kcur_vec = *((device const float4*)(k_cur + kv_head * DK + off));

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    float4 acc = 0.0f;

    for (uint chunk_base = eff_start; chunk_base < kv_end; chunk_base += ATTN_CHUNK_F16) {
        const uint chunk_end = min(chunk_base + ATTN_CHUNK_F16, kv_end);
        const uint nc = chunk_end - chunk_base;

        // Phase 1: scores, 4 positions per round — 4 independent K loads
        // issue before the 4 reductions. Tail positions (j >= nc) and
        // kv_mask-evicted positions score -INF exactly as in v2.
        float scores[ATTN_CHUNK_F16];
        for (uint j4 = 0u; j4 < nc; j4 += 4u) {
            float p[4];
            for (uint k = 0u; k < 4u; k++) {
                const uint j = j4 + k;
                const uint t = chunk_base + j;
                if (j >= nc ||
                    (kv_mask != nullptr && t != pos && kv_mask[kv_head * max_seq + t] == 0u)) {
                    p[k] = -INFINITY;
                    continue;
                }
                if (!cache_only && t == pos) {
                    p[k] = dot(q_vec, kcur_vec);
                } else {
                    p[k] = dot(q_vec, (float4)(*((device const half4*)(K_head + t * DK + off))));
                }
            }
            // 4 back-to-back reductions. The -INF sentinel (masked position)
            // is lane-uniform, so skipping its simd_sum is convergent.
            for (uint k = 0u; k < 4u; k++) {
                const uint j = j4 + k;
                if (j >= nc) break;
                scores[j] = (p[k] == -INFINITY) ? -INFINITY : simd_sum(p[k]) * scale;
            }
        }

        // Phase 2: block max — identical to v2.
        float block_max = scores[0];
        for (uint j = 1u; j < nc; j++)
            block_max = max(block_max, scores[j]);
        if (block_max == -INFINITY) continue;

        float new_max = max(running_max, block_max);
        float correction = precise::exp(running_max - new_max);
        acc *= correction;
        running_sum *= correction;
        running_max = new_max;

        // Phase 3: V accumulate — identical to v2 (loads already independent).
        for (uint j = 0u; j < nc; j++) {
            float w = precise::exp(scores[j] - running_max);
            running_sum += w;
            if (w < 1e-6f) continue;

            const uint t = chunk_base + j;
            if (!cache_only && t == pos) {
                acc += w * (*((device const float4*)(v_cur + kv_head * DK + off)));
            } else {
                acc += w * (float4)(*((device const half4*)(V_head + t * DK + off)));
            }
        }
    }

    if (lid == 0u) { my_partial[0] = running_max; my_partial[1] = running_sum; }
    *((device float4*)(my_partial + 2u + off)) = acc;
}
