kernel void rms_norm_batch(
    device const float* x      [[ buffer(0) ]],  // [B × n]
    device const float* weight [[ buffer(1) ]],  // [n] (shared)
    device       float* out    [[ buffer(2) ]],  // [B × n]
    constant     uint&  n      [[ buffer(3) ]],
    constant     float& eps    [[ buffer(4) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint tid [[ thread_index_in_threadgroup ]],
    uint sgitg [[ simdgroup_index_in_threadgroup ]],
    uint lid [[ thread_index_in_simdgroup ]],
    threadgroup float* shared [[ threadgroup(0) ]])
{
    const uint batch = tgid;
    device const float* xb  = x   + batch * n;
    device       float* ob  = out + batch * n;
    device const float4* xb4 = (device const float4*)xb;
    device const float4* wb4 = (device const float4*)weight;
    device float4* ob4 = (device float4*)ob;
    const uint n4 = n >> 2u;

    float sum_sq = 0.0f;
    for (uint i = tid; i < n4; i += 128u)
        sum_sq += dot(xb4[i], xb4[i]);
    sum_sq = simd_sum(sum_sq);
    if (lid == 0u) shared[sgitg] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u)
        shared[4] = rsqrt((shared[0] + shared[1] + shared[2] + shared[3]) / float(n) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_rms = shared[4];
    for (uint i = tid; i < n4; i += 128u)
        ob4[i] = xb4[i] * inv_rms * wb4[i];
}

// ── rms_norm_batch_inplace ────────────────────────────────────────────────
//
// Batched in-place RMSNorm: x[b,i] = x[b,i] / rms(x[b]) * weight[i]
// Eliminates the need for a separate copy dispatch after norm→scratch→copy.
//
// dispatch: (B, 1, 1) × (32, 1, 1)
//
kernel void rms_norm_batch_inplace(
    device       float* x      [[ buffer(0) ]],  // [B × n] rw
    device const float* weight [[ buffer(1) ]],  // [n] (shared)
    constant     uint&  n      [[ buffer(2) ]],
    constant     float& eps    [[ buffer(3) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint lid  [[ thread_index_in_simdgroup ]])
{
    const uint batch = tgid;
    device float* xb = x + batch * n;

    float sum_sq = 0.0f;
    for (uint i = lid; i < n; i += 32u)
        sum_sq += xb[i] * xb[i];
    sum_sq = simd_sum(sum_sq);

    float inv_rms = rsqrt(sum_sq / float(n) + eps);

    for (uint i = lid; i < n; i += 32u)
        xb[i] = xb[i] * inv_rms * weight[i];
}

// ── fused_residual_rms_norm_batch ─────────────────────────────────────────
//
// Batched fused residual + RMSNorm:
//   hidden[b,i] += delta[b,i]
//   normed[b,i]  = hidden[b,i] * inv_rms * weight[i]
//
// dispatch: (B, 1, 1) × (32, 1, 1)
//
kernel void fused_residual_rms_norm_batch(
    device       float* hidden [[ buffer(0) ]],  // [B × n] rw
    device const float* delta  [[ buffer(1) ]],  // [B × n]
    device const float* weight [[ buffer(2) ]],  // [n] (shared)
    device       float* normed [[ buffer(3) ]],  // [B × n]
    constant     uint&  n      [[ buffer(4) ]],
    constant     float& eps    [[ buffer(5) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint lid  [[ thread_index_in_simdgroup ]])
{
    const uint batch = tgid;
    device       float* hb = hidden + batch * n;
    device const float* db = delta  + batch * n;
    device       float* nb = normed + batch * n;

    float sum_sq = 0.0f;
    for (uint i = lid; i < n; i += 32u) {
        hb[i] += db[i];
        sum_sq += hb[i] * hb[i];
    }
    sum_sq = simd_sum(sum_sq);

    float inv_rms = rsqrt(sum_sq / float(n) + eps);

    for (uint i = lid; i < n; i += 32u)
        nb[i] = hb[i] * inv_rms * weight[i];
}

// ── fused_rms_norm_residual_add_batch ──────────────────────────────────────
//
// Fused: normed = rms_norm(src) * weight;  dst += normed
// Eliminates the intermediate normed buffer write + read.
// Used for PLE post-norm + residual and post-FFN norm + residual.
//
// dispatch: (B, 1, 1) × (32, 1, 1)
//
kernel void fused_rms_norm_residual_add_batch(
    device       float* dst    [[ buffer(0) ]],  // [B × n] read+write (hidden)
    device const float* src    [[ buffer(1) ]],  // [B × n] read (attn_proj or ffn_out)
    device const float* weight [[ buffer(2) ]],  // [n] norm weights
    constant     uint&  n      [[ buffer(3) ]],
    constant     float& eps    [[ buffer(4) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint tid [[ thread_index_in_threadgroup ]],
    uint sgitg [[ simdgroup_index_in_threadgroup ]],
    uint lid [[ thread_index_in_simdgroup ]],
    threadgroup float* shared [[ threadgroup(0) ]])
{
    const uint batch = tgid;
    device       float* db = dst + batch * n;
    device const float* sb = src + batch * n;
    device float4* db4 = (device float4*)db;
    device const float4* sb4 = (device const float4*)sb;
    device const float4* wb4 = (device const float4*)weight;
    const uint n4 = n >> 2u;

    float sum_sq = 0.0f;
    for (uint i = tid; i < n4; i += 128u)
        sum_sq += dot(sb4[i], sb4[i]);
    sum_sq = simd_sum(sum_sq);
    if (lid == 0u) shared[sgitg] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u)
        shared[4] = rsqrt((shared[0] + shared[1] + shared[2] + shared[3]) / float(n) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_rms = shared[4];
    for (uint i = tid; i < n4; i += 128u)
        db4[i] += sb4[i] * inv_rms * wb4[i];
}

// Decode-only dual-epsilon tail fusion:
//   hidden += rms_norm(src, eps1) * weight1
//   output  = rms_norm(hidden, eps2) * weight2
// Muse uses eps1=1e-8 for sandwich post-norms and eps2=1e-5 for the
// following pre-norm, so the older single-epsilon batch kernel is not valid.
kernel void muser_fused_norm_residual_rms_norm_32sg(
    device float* hidden [[buffer(0)]],
    device const float* src [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const float* weight1 [[buffer(3)]],
    device const float* weight2 [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant float& eps1 [[buffer(6)]],
    constant float& eps2 [[buffer(7)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint lid [[thread_index_in_simdgroup]],
    threadgroup float* shared [[threadgroup(0)]]) {
    const uint n4 = n >> 2u;
    device float4* hidden4 = (device float4*)(hidden + row * n);
    device const float4* src4 = (device const float4*)(src + row * n);
    device float4* output4 = (device float4*)(output + row * n);
    device const float4* weight14 = (device const float4*)weight1;
    device const float4* weight24 = (device const float4*)weight2;

    float sum_src = 0.0f;
    for (uint i = tid; i < n4; i += 1024u)
        sum_src += dot(src4[i], src4[i]);
    sum_src = simd_sum(sum_src);
    if (lid == 0u) shared[sgitg] = sum_src;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float total = 0.0f;
        for (uint group = 0u; group < 32u; ++group) total += shared[group];
        shared[32] = rsqrt(total / float(n) + eps1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv_src = shared[32];
    float sum_hidden = 0.0f;
    for (uint i = tid; i < n4; i += 1024u) {
        const float4 value = hidden4[i] + src4[i] * inv_src * weight14[i];
        hidden4[i] = value;
        sum_hidden += dot(value, value);
    }
    sum_hidden = simd_sum(sum_hidden);
    if (lid == 0u) shared[sgitg] = sum_hidden;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float total = 0.0f;
        for (uint group = 0u; group < 32u; ++group) total += shared[group];
        shared[32] = rsqrt(total / float(n) + eps2);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float inv_hidden = shared[32];
    for (uint i = tid; i < n4; i += 1024u)
        output4[i] = hidden4[i] * inv_hidden * weight24[i];
}

// ── fused_inplace_norm_residual_add_batch ─────────────────────────────────
//
// Fused: src = rms_norm(src) * weight;  dst += src
// In-place norm on src, then add to dst.
// Used for post-FFN path: norm ffn_out in-place, add to hidden.
//
// dispatch: (B, 1, 1) × (32, 1, 1)
//
kernel void fused_inplace_norm_residual_add_batch(
    device       float* dst    [[ buffer(0) ]],  // [B × n] read+write (hidden)
    device       float* src    [[ buffer(1) ]],  // [B × n] read+write (ffn_out, normed in-place)
    device const float* weight [[ buffer(2) ]],  // [n] norm weights
    constant     uint&  n      [[ buffer(3) ]],
    constant     float& eps    [[ buffer(4) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint lid  [[ thread_index_in_simdgroup ]])
{
    const uint batch = tgid;
    device float* db = dst + batch * n;
    device float* sb = src + batch * n;

    // Pass 1: compute sum of squares
    float sum_sq = 0.0f;
    for (uint i = lid; i < n; i += 32u)
        sum_sq += sb[i] * sb[i];
    sum_sq = simd_sum(sum_sq);

    float inv_rms = rsqrt(sum_sq / float(n) + eps);

    // Pass 2: normalize in-place and add to dst
    for (uint i = lid; i < n; i += 32u) {
        float normed = sb[i] * inv_rms * weight[i];
        sb[i] = normed;  // in-place norm
        db[i] += normed;  // residual add
    }
}

// ── fused_norm_residual_rms_norm_batch ────────────────────────────────────
//
// Triple fusion: norm1(src) → hidden += normed → output = norm2(hidden)
// Eliminates the intermediate normalized src buffer entirely.
// Used for post-attn: norm(attn_proj), hidden += normed, ffn_normed = norm(hidden),
// and for the matching post-FFN to next-layer transition. Muse uses 1e-8 for
// the sandwich norm and 1e-5 for the following pre-layer norm.
//
// dispatch: (B, 1, 1) × (128, 1, 1)
//
kernel void muser_fused_norm_residual_rms_norm_batch_dual_eps(
    device       float* hidden      [[ buffer(0) ]],  // [B × n] read+write
    device const float* src         [[ buffer(1) ]],  // [B × n] read (attn_proj)
    device       float* output      [[ buffer(2) ]],  // [B × n] write (ffn_normed)
    device const float* weight1     [[ buffer(3) ]],  // [n] post-attn norm weights
    device const float* weight2     [[ buffer(4) ]],  // [n] pre-FFN norm weights
    constant     uint&  n           [[ buffer(5) ]],
    constant     float& first_eps   [[ buffer(6) ]],
    constant     float& second_eps  [[ buffer(7) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint tid [[ thread_index_in_threadgroup ]],
    uint sgitg [[ simdgroup_index_in_threadgroup ]],
    uint lid [[ thread_index_in_simdgroup ]],
    uint ntg [[ threads_per_threadgroup ]],
    threadgroup float* shared [[ threadgroup(0) ]])
{
    const uint batch = tgid;
    device       float* hb = hidden + batch * n;
    device const float* sb = src    + batch * n;
    device       float* ob = output + batch * n;
    device float4* hb4 = (device float4*)hb;
    device const float4* sb4 = (device const float4*)sb;
    device float4* ob4 = (device float4*)ob;
    device const float4* weight14 = (device const float4*)weight1;
    device const float4* weight24 = (device const float4*)weight2;
    const uint n4 = n >> 2u;

    // Reproduce pinned ggml kernel_rms_norm_mul_add_f32_4 exactly. In
    // particular, keep its 32-SIMD-group reduction, 1/sqrt scale, and
    // multiply/add expression shape; changing any of these moved public
    // logprobs beyond the contract in the rejected four-group fusion.
    if (sgitg == 0u)
        shared[lid] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum_sq_src = 0.0f;
    for (uint i = tid; i < n4; i += ntg)
        sum_sq_src += dot(sb4[i], sb4[i]);
    sum_sq_src = simd_sum(sum_sq_src);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u)
        shared[sgitg] = sum_sq_src;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum_sq_src = shared[lid];
    sum_sq_src = simd_sum(sum_sq_src);
    const float inv_rms1 = 1.0f / sqrt(sum_sq_src / float(n) + first_eps);

    for (uint i = tid; i < n4; i += ntg)
        hb4[i] = (sb4[i] * inv_rms1) * weight14[i] + hb4[i];

    // The split route ends one dispatch and starts another here. Force the
    // same f32 device-memory publication/reread boundary before reproducing
    // kernel_rms_norm_mul_f32_4 for the second norm.
    threadgroup_barrier(mem_flags::mem_device);
    if (sgitg == 0u)
        shared[lid] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum_sq_h = 0.0f;
    for (uint i = tid; i < n4; i += ntg)
        sum_sq_h += dot(hb4[i], hb4[i]);
    sum_sq_h = simd_sum(sum_sq_h);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u)
        shared[sgitg] = sum_sq_h;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sum_sq_h = shared[lid];
    sum_sq_h = simd_sum(sum_sq_h);
    const float inv_rms2 = 1.0f / sqrt(sum_sq_h / float(n) + second_eps);

    for (uint i = tid; i < n4; i += ntg)
        ob4[i] = (hb4[i] * inv_rms2) * weight24[i];
}

// ── fused_norm_residual_scale_rms_norm_batch ──────────────────────────────
//
// Quadruple fusion for PLE path:
//   norm1(src) → hidden += normed → hidden *= scale → output = norm2(hidden)
// Saves 2 dispatches + 2 barriers per layer vs split path.
//
// dispatch: (B, 1, 1) × (32, 1, 1)
//
kernel void fused_norm_residual_scale_rms_norm_batch(
    device       float* hidden      [[ buffer(0) ]],  // [B × n] read+write
    device const float* src         [[ buffer(1) ]],  // [B × n] read (attn_proj)
    device       float* output      [[ buffer(2) ]],  // [B × n] write (normed for next layer)
    device const float* weight1     [[ buffer(3) ]],  // [n] post-norm weights (PLE norm)
    device const float* weight2     [[ buffer(4) ]],  // [n] pre-norm weights (next layer)
    constant     uint&  n           [[ buffer(5) ]],
    constant     float& scale       [[ buffer(6) ]],
    constant     float& eps         [[ buffer(7) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint lid  [[ thread_index_in_simdgroup ]])
{
    const uint batch = tgid;
    device       float* hb = hidden + batch * n;
    device const float* sb = src    + batch * n;
    device       float* ob = output + batch * n;

    // Pass 1: compute sum of squares of src for norm1
    float sum_sq_src = 0.0f;
    for (uint i = lid; i < n; i += 32u)
        sum_sq_src += sb[i] * sb[i];
    sum_sq_src = simd_sum(sum_sq_src);
    float inv_rms1 = rsqrt(sum_sq_src / float(n) + eps);

    // Pass 2: apply norm1, residual add, scale, compute sum_sq for norm2
    float sum_sq_h = 0.0f;
    for (uint i = lid; i < n; i += 32u) {
        float normed = sb[i] * inv_rms1 * weight1[i];
        float h = (hb[i] + normed) * scale;
        hb[i] = h;
        sum_sq_h += h * h;
    }
    sum_sq_h = simd_sum(sum_sq_h);
    float inv_rms2 = rsqrt(sum_sq_h / float(n) + eps);

    // Pass 3: write output norm
    for (uint i = lid; i < n; i += 32u)
        ob[i] = hb[i] * inv_rms2 * weight2[i];
}

// ── fused_residual_scale_rms_norm_batch ───────────────────────────────────
//
// Triple fusion for non-PLE scaled layers:
//   hidden += norm(ffn_out) → hidden *= scale → output = norm(hidden)
// Same as fused_norm_residual_scale_rms_norm_batch but for the FFN residual.
//
// dispatch: (B, 1, 1) × (32, 1, 1)
//
kernel void fused_residual_scale_rms_norm_batch(
    device       float* hidden      [[ buffer(0) ]],  // [B × n] read+write
    device       float* src         [[ buffer(1) ]],  // [B × n] read+write (ffn_out, normed in-place)
    device       float* output      [[ buffer(2) ]],  // [B × n] write (normed for next layer)
    device const float* weight1     [[ buffer(3) ]],  // [n] post-FFN norm weights
    device const float* weight2     [[ buffer(4) ]],  // [n] pre-norm weights (next layer)
    constant     uint&  n           [[ buffer(5) ]],
    constant     float& scale       [[ buffer(6) ]],
    constant     float& eps         [[ buffer(7) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint lid  [[ thread_index_in_simdgroup ]])
{
    const uint batch = tgid;
    device float* hb = hidden + batch * n;
    device float* sb = src    + batch * n;
    device float* ob = output + batch * n;

    // Pass 1: compute sum of squares of src for norm1
    float sum_sq = 0.0f;
    for (uint i = lid; i < n; i += 32u)
        sum_sq += sb[i] * sb[i];
    sum_sq = simd_sum(sum_sq);
    float inv_rms1 = rsqrt(sum_sq / float(n) + eps);

    // Pass 2: in-place norm, residual add, scale, compute sum_sq for norm2
    float sum_sq_h = 0.0f;
    for (uint i = lid; i < n; i += 32u) {
        float normed = sb[i] * inv_rms1 * weight1[i];
        sb[i] = normed;
        float h = (hb[i] + normed) * scale;
        hb[i] = h;
        sum_sq_h += h * h;
    }
    sum_sq_h = simd_sum(sum_sq_h);
    float inv_rms2 = rsqrt(sum_sq_h / float(n) + eps);

    // Pass 3: write output norm
    for (uint i = lid; i < n; i += 32u)
        ob[i] = hb[i] * inv_rms2 * weight2[i];
}
