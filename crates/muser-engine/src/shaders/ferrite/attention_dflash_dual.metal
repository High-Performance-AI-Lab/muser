// ── dflash_dual_attention_f32 ──────────────────────────────────────────────
//
// Bidirectional GQA attention for DFlash draft forward.
// Logical KV layout: [cached_ctx | fresh_ctx | noise].
//
// Dispatch:
//   thread_groups = (batch_size, n_kv_heads, 1)
//   threads       = (heads_per_kv × 32, 1, 1)
//
// Ferrite's full attention translation unit declares this in
// attention_batch_flash.metal. The standalone DFlash extraction carries the
// same constant locally so it does not import unrelated attention routes.
constant constexpr uint FA_TILE_CAPACITY = 2048;

kernel void dflash_dual_attention_f32(
    device const float* Q_batch        [[ buffer(0) ]],
    device const float* K_cache        [[ buffer(1) ]],
    device const float* V_cache        [[ buffer(2) ]],
    device const float* K_ctx          [[ buffer(3) ]],
    device const float* V_ctx          [[ buffer(4) ]],
    device const float* K_noise        [[ buffer(5) ]],
    device const float* V_noise        [[ buffer(6) ]],
    device       float* out            [[ buffer(7) ]],
    constant     uint&  head_dim       [[ buffer(8) ]],
    constant     uint&  cached_ctx_len [[ buffer(9) ]],
    constant     uint&  fresh_ctx_len  [[ buffer(10) ]],
    constant     uint&  n_heads        [[ buffer(11) ]],
    constant     uint&  n_kv_heads     [[ buffer(12) ]],
    constant     uint&  batch_size     [[ buffer(13) ]],
    uint2 gid    [[ threadgroup_position_in_grid ]],
    uint  tid    [[ thread_index_in_threadgroup ]],
    uint  sgitg  [[ simdgroup_index_in_threadgroup ]],
    uint  siitg  [[ thread_index_in_simdgroup ]])
{
    const uint hd      = head_dim;
    const uint bi      = gid.x;
    const uint kv_head = gid.y;
    const uint n_sg    = n_heads / n_kv_heads;
    const uint q_head  = kv_head * n_sg + sgitg;
    const uint q_dim   = n_heads * hd;
    const uint kv_dim  = n_kv_heads * hd;
    const uint n_th    = n_sg * 32u;
    const uint ept     = (hd + 31u) / 32u;
    const float scale  = rsqrt(float(hd));

    if (bi >= batch_size || kv_head >= n_kv_heads) return;

    device const float* q_ptr = Q_batch + bi * q_dim + q_head * hd;
    float q_reg[16];
    float o_reg[16];
    for (uint e = 0; e < ept; ++e) {
        uint idx = siitg + e * 32u;
        q_reg[e] = (idx < hd) ? q_ptr[idx] : 0.0f;
        o_reg[e] = 0.0f;
    }

    float M = -INFINITY;
    float D = 0.0f;

    threadgroup float k_tile[FA_TILE_CAPACITY];
    threadgroup float v_tile[FA_TILE_CAPACITY];
    const uint fa_tile = max(1u, FA_TILE_CAPACITY / max(1u, hd));

    for (uint seg = 0; seg < 3u; ++seg) {
        const uint seg_len = (seg == 0u) ? cached_ctx_len : ((seg == 1u) ? fresh_ctx_len : batch_size);
        device const float* seg_k = (seg == 0u) ? K_cache : ((seg == 1u) ? K_ctx : K_noise);
        device const float* seg_v = (seg == 0u) ? V_cache : ((seg == 1u) ? V_ctx : V_noise);

        for (uint kt = 0; kt < seg_len; kt += fa_tile) {
            uint tl = min(fa_tile, seg_len - kt);
            uint ne = tl * hd;
            for (uint i = tid; i < ne; i += n_th) {
                uint t = i / hd;
                uint d = i % hd;
                uint off = (kt + t) * kv_dim + kv_head * hd + d;
                k_tile[i] = seg_k[off];
                v_tile[i] = seg_v[off];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint t = 0; t < tl; ++t) {
                float dot = 0.0f;
                for (uint e = 0; e < ept; ++e) {
                    uint idx = siitg + e * 32u;
                    if (idx < hd) dot += q_reg[e] * k_tile[t * hd + idx];
                }
                float s = simd_sum(dot) * scale;
                float new_M = max(M, s);
                float r = precise::exp(M - new_M);
                float e = precise::exp(s - new_M);
                D = D * r + e;
                for (uint el = 0; el < ept; ++el) {
                    uint idx = siitg + el * 32u;
                    if (idx < hd) o_reg[el] = o_reg[el] * r + e * v_tile[t * hd + idx];
                }
                M = new_M;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    float inv_D = 1.0f / D;
    device float* o = out + bi * q_dim + q_head * hd;
    for (uint e = 0; e < ept; ++e) {
        uint idx = siitg + e * 32u;
        if (idx < hd) o[idx] = o_reg[e] * inv_D;
    }
}
