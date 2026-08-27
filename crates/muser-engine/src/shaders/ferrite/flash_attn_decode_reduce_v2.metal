// Exact contiguous-partial LSE reducer from Ferrite
// flash_attn_decode_vec_grass.metal at a85048a90. The experimental Grass
// producer is intentionally not imported.
kernel void flash_attn_decode_reduce_v2(
    device const float* partials    [[ buffer(0) ]],
    device       float* out         [[ buffer(1) ]],
    constant     uint&  head_dim_arg[[ buffer(2) ]],
    constant     uint&  nwg_arg     [[ buffer(3) ]],
    constant DecodeParams* decode_params [[ buffer(18) ]],
    uint  head [[ threadgroup_position_in_grid ]],
    uint  lid  [[ thread_index_in_simdgroup ]])
{
    const uint DK = HAS_FC_DK ? FC_DK : head_dim_arg;
    const uint N_VEC = (DK + 127u) / 128u;
    const uint nwg = BIND_DECODE_PARAMS ? decode_params->nwg : nwg_arg;
    const uint partial_stride = 2u + DK;
    device const float* base = partials + head * nwg * partial_stride;

    float global_max = -INFINITY;
    for (uint w = 0u; w < nwg; w++)
        global_max = max(global_max, base[w * partial_stride]);

    float total_sum = 0.0f;
    float4 final_acc[4];
    for (uint v = 0u; v < N_VEC; v++) final_acc[v] = 0.0f;

    for (uint w = 0u; w < nwg; w++) {
        device const float* p = base + w * partial_stride;
        float w_sum = p[1];
        if (w_sum == 0.0f) continue;
        float correction = precise::exp(p[0] - global_max);
        total_sum += w_sum * correction;
        for (uint v = 0u; v < N_VEC; v++) {
            uint off = v * 128u + lid * 4u;
            if (off < DK) {
                float4 pw = *((device const float4*)(p + 2u + off));
                final_acc[v] += pw * correction;
            }
        }
    }

    float inv_sum = (total_sum > 0.0f) ? (1.0f / total_sum) : 0.0f;
    for (uint v = 0u; v < N_VEC; v++) {
        uint off = v * 128u + lid * 4u;
        if (off < DK)
            *((device float4*)(out + head * DK + off)) = final_acc[v] * inv_sum;
    }
}
