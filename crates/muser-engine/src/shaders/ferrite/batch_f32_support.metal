// ── matmul_f32_batch ─────────────────────────────────────────────────────
//
// Batch F32 GEMM (for lm_head or F32 weights).
//
// dispatch: (rows, B, 1) × (32, 1, 1)
//
kernel void matmul_f32_batch(
    device const float* W    [[ buffer(0) ]],
    device const float* X    [[ buffer(1) ]],   // [B × cols]
    device       float* Y    [[ buffer(2) ]],   // [B × rows]
    constant     uint&  cols [[ buffer(3) ]],
    constant     uint&  rows [[ buffer(4) ]],
    uint2 tgid [[ threadgroup_position_in_grid ]],
    uint  lid  [[ thread_index_in_simdgroup ]])
{
    const uint row   = tgid.x;
    const uint batch = tgid.y;

    device const float* row_w = W + row * cols;
    device const float* x     = X + batch * cols;

    float acc = 0.0f;
    for (uint c = lid; c < cols; c += 32u)
        acc += row_w[c] * x[c];

    acc = simd_sum(acc);
    if (lid == 0u)
        Y[batch * rows + row] = acc;
}

// ── matmul_f32_batch_tiled ──────────────────────────────────────────────
//
// Tiled F32 batch GEMM. Each threadgroup produces a 4-row × 4-batch = 16
// output tile. At batch=4096 × rows=2560 (Gemma 4 PLE proj matmul), this
// cuts threadgroup count by 16× vs `matmul_f32_batch` (2.6M → 164K TGs),
// increasing work-per-TG from 256 FMAs to 4096 FMAs — amortizes dispatch
// overhead which dominates the flat kernel at this shape.
//
// Measured wins (Gemma 4 PLE proj, F32 2560×256×4096):
//   flat kernel:    13,495 µs / layer
//   4×4 tile:        1,880 µs / layer (7.2× faster)
//
// dispatch: (ceil(rows/4), ceil(batch/4), 1) × (32, 1, 1)
//
kernel void matmul_f32_batch_tiled(
    device const float* W     [[ buffer(0) ]],
    device const float* X     [[ buffer(1) ]],   // [B × cols]
    device       float* Y     [[ buffer(2) ]],   // [B × rows]
    constant     uint&  cols  [[ buffer(3) ]],
    constant     uint&  rows  [[ buffer(4) ]],
    constant     uint&  batch [[ buffer(5) ]],
    uint2 tgid [[ threadgroup_position_in_grid ]],
    uint  lid  [[ thread_index_in_simdgroup ]])
{
    const uint r_base = tgid.x * 4u;
    const uint b_base = tgid.y * 4u;

    float acc[4][4] = {{0.0f}};

    for (uint c = lid; c < cols; c += 32u) {
        float w[4];
        for (uint ri = 0u; ri < 4u; ++ri) {
            uint r = r_base + ri;
            w[ri] = (r < rows) ? W[r * cols + c] : 0.0f;
        }
        float xv[4];
        for (uint bi = 0u; bi < 4u; ++bi) {
            uint b = b_base + bi;
            xv[bi] = (b < batch) ? X[b * cols + c] : 0.0f;
        }
        for (uint ri = 0u; ri < 4u; ++ri) {
            for (uint bi = 0u; bi < 4u; ++bi) {
                acc[ri][bi] += w[ri] * xv[bi];
            }
        }
    }

    for (uint ri = 0u; ri < 4u; ++ri) {
        for (uint bi = 0u; bi < 4u; ++bi) {
            float sum = simd_sum(acc[ri][bi]);
            uint r = r_base + ri;
            uint b = b_base + bi;
            if (lid == 0u && r < rows && b < batch) {
                Y[b * rows + r] = sum;
            }
        }
    }
}

// ── matmul_f32_batch_tiled_8x8 ──────────────────────────────────────────
//
// Larger 8×8 = 64-output tile variant. 4× fewer TGs vs 4×4 tile, 4× more
// work per TG. At Gemma 4 PLE proj (F32 2560×256×4096): 164K → 41K TGs,
// each doing 16384 FMAs. Register pressure is higher (64 accumulators per
// thread) which may spill on some GPUs — use cautiously.
//
// dispatch: (ceil(rows/8), ceil(batch/8), 1) × (32, 1, 1)
//
kernel void matmul_f32_batch_tiled_8x8(
    device const float* W     [[ buffer(0) ]],
    device const float* X     [[ buffer(1) ]],
    device       float* Y     [[ buffer(2) ]],
    constant     uint&  cols  [[ buffer(3) ]],
    constant     uint&  rows  [[ buffer(4) ]],
    constant     uint&  batch [[ buffer(5) ]],
    uint2 tgid [[ threadgroup_position_in_grid ]],
    uint  lid  [[ thread_index_in_simdgroup ]])
{
    const uint r_base = tgid.x * 8u;
    const uint b_base = tgid.y * 8u;

    float acc[8][8] = {{0.0f}};

    for (uint c = lid; c < cols; c += 32u) {
        float w[8];
        for (uint ri = 0u; ri < 8u; ++ri) {
            uint r = r_base + ri;
            w[ri] = (r < rows) ? W[r * cols + c] : 0.0f;
        }
        float xv[8];
        for (uint bi = 0u; bi < 8u; ++bi) {
            uint b = b_base + bi;
            xv[bi] = (b < batch) ? X[b * cols + c] : 0.0f;
        }
        for (uint ri = 0u; ri < 8u; ++ri) {
            for (uint bi = 0u; bi < 8u; ++bi) {
                acc[ri][bi] += w[ri] * xv[bi];
            }
        }
    }

    for (uint ri = 0u; ri < 8u; ++ri) {
        for (uint bi = 0u; bi < 8u; ++bi) {
            float sum = simd_sum(acc[ri][bi]);
            uint r = r_base + ri;
            uint b = b_base + bi;
            if (lid == 0u && r < rows && b < batch) {
                Y[b * rows + r] = sum;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Supporting batch kernels: norms, RoPE, element-wise ops
// ═══════════════════════════════════════════════════════════════════════════

// ── residual_add_batch ────────────────────────────────────────────────────
//
// Element-wise in-place: a[i] += b[i] for i in 0..total
// Works on flat [B × n] arrays — just pass total = B * n.
//
// dispatch: (ceil(total/1024), 1, 1) × (1024, 1, 1)
//
kernel void residual_add_batch(
    device       float* a     [[ buffer(0) ]],
    device const float* b     [[ buffer(1) ]],
    constant     uint&  total [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]])
{
    if (gid < total)
        a[gid] += b[gid];
}
