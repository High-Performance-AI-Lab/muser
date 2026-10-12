// Quantization format and algorithm provenance is pinned in `THIRD_PARTY.md`;
// the retained upstream notice is `LICENSES/GGML-LLAMA-MIT.txt`.

use super::f16_to_f32;
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// Q6_K dot product matching llama.cpp's exact deferred-scaling accumulation.
///
/// Attribution: accumulation order from ggml-org/llama.cpp
/// kernel_mul_mv_q6_K_f32_impl (ggml-metal.metal).
///
/// Key difference: accumulates sum(q * x) per sub-block first, then multiplies
/// by d * scale ONCE. The existing paths multiply d * scale * q * x per element.
pub fn dot_q6_k_f32_llama(q6k_blocks: &[u8], x: &[f32], n: usize) -> f32 {
    debug_assert!(n.is_multiple_of(256));
    let n_blocks = n / 256;
    let mut sumf = 0.0f32;

    for b in 0..n_blocks {
        let block = &q6k_blocks[b * 210..];
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        let ql = &block[0..128];
        let qh = &block[128..192];
        let sc = &block[192..208];
        let x_block = &x[b * 256..b * 256 + 256];

        // Process 256 elements in 16 sub-blocks of 16 elements each.
        // For each sub-block: accumulate sum(q * x) first, then multiply by d * scale.
        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        let mut sc_off = 0usize;
        let mut x_off = 0usize;
        for _group in 0..2 {
            // 4 sub-block pairs per group (is=0,1 for l=0..31)
            // Each sub-block pair produces sums[0..3]:
            //   sums[0] = sum(q1 * x[l])     for l=0..3  (ql low nibble)
            //   sums[1] = sum(q2 * x[l+32])  for l=0..3  (ql+32 low nibble)
            //   sums[2] = sum(q3 * x[l+64])  for l=0..3  (ql high nibble)
            //   sums[3] = sum(q4 * x[l+96])  for l=0..3  (ql+32 high nibble)
            for l_base in (0..32usize).step_by(16) {
                let is = l_base / 16;
                let mut sums = [0.0f32; 4];
                for l in l_base..l_base + 16 {
                    let ql0 = ql[ql_off + l] as u32;
                    let ql32 = ql[ql_off + l + 32] as u32;
                    let qhl = qh[qh_off + l] as u32;

                    let q1 = ((ql0 & 0xF) | ((qhl & 3) << 4)) as i32 - 32;
                    let q2 = ((ql32 & 0xF) | (((qhl >> 2) & 3) << 4)) as i32 - 32;
                    let q3 = ((ql0 >> 4) | (((qhl >> 4) & 3) << 4)) as i32 - 32;
                    let q4 = ((ql32 >> 4) | (((qhl >> 6) & 3) << 4)) as i32 - 32;

                    sums[0] += q1 as f32 * x_block[x_off + l];
                    sums[1] += q2 as f32 * x_block[x_off + l + 32];
                    sums[2] += q3 as f32 * x_block[x_off + l + 64];
                    sums[3] += q4 as f32 * x_block[x_off + l + 96];
                }

                // Deferred scaling: d * (sum(q*x) * scale) per sub-block.
                let s0 = sc[sc_off + is] as i8 as f32;
                let s1 = sc[sc_off + is + 2] as i8 as f32;
                let s2 = sc[sc_off + is + 4] as i8 as f32;
                let s3 = sc[sc_off + is + 6] as i8 as f32;
                sumf += d * (sums[0] * s0 + sums[1] * s1 + sums[2] * s2 + sums[3] * s3);
            }
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
            x_off += 128;
        }
    }

    sumf
}

/// Dot product of Q6_K x f32 vectors without materializing a dequantized row.
///
/// The public path preserves the optimized implementation shape while matching
/// llama.cpp's deferred-scaling accumulation contract.
pub fn dot_q6_k_f32(q6k_blocks: &[u8], x: &[f32], n: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return dot_q6_k_f32_neon(q6k_blocks, x, n);
    }
    #[allow(unreachable_code)]
    dot_q6_k_f32_scalar(q6k_blocks, x, n)
}

pub(crate) fn dot_q6_k_f32_scalar(q6k_blocks: &[u8], x: &[f32], n: usize) -> f32 {
    debug_assert!(n.is_multiple_of(256));
    let n_blocks = n / 256;
    let mut sumf = 0.0f32;

    for b in 0..n_blocks {
        let block = &q6k_blocks[b * 210..];
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        let ql = &block[0..128];
        let qh = &block[128..192];
        let sc = &block[192..208];
        let x_block = &x[b * 256..b * 256 + 256];

        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        let mut sc_off = 0usize;
        let mut x_off = 0usize;
        for _group in 0..2 {
            for l_base in (0..32usize).step_by(16) {
                let is = l_base / 16;
                let mut sums = [0.0f32; 4];
                for l in l_base..l_base + 16 {
                    let ql0 = ql[ql_off + l] as u32;
                    let ql32 = ql[ql_off + l + 32] as u32;
                    let qhl = qh[qh_off + l] as u32;

                    let q1 = ((ql0 & 0xF) | ((qhl & 3) << 4)) as i32 - 32;
                    let q2 = ((ql32 & 0xF) | (((qhl >> 2) & 3) << 4)) as i32 - 32;
                    let q3 = ((ql0 >> 4) | (((qhl >> 4) & 3) << 4)) as i32 - 32;
                    let q4 = ((ql32 >> 4) | (((qhl >> 6) & 3) << 4)) as i32 - 32;

                    sums[0] += q1 as f32 * x_block[x_off + l];
                    sums[1] += q2 as f32 * x_block[x_off + l + 32];
                    sums[2] += q3 as f32 * x_block[x_off + l + 64];
                    sums[3] += q4 as f32 * x_block[x_off + l + 96];
                }

                let s0 = sc[sc_off + is] as i8 as f32;
                let s1 = sc[sc_off + is + 2] as i8 as f32;
                let s2 = sc[sc_off + is + 4] as i8 as f32;
                let s3 = sc[sc_off + is + 6] as i8 as f32;
                sumf += d * (sums[0] * s0 + sums[1] * s1 + sums[2] * s2 + sums[3] * s3);
            }
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
            x_off += 128;
        }
    }

    sumf
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
/// Multiply four lanes and accumulate them in deterministic scalar order.
///
/// # Safety
/// `x_ptr` must point to at least four initialized, properly aligned `f32`
/// values. This function is compiled only for AArch64, where NEON is present.
unsafe fn ordered_mul_accumulate4(acc: &mut f32, vals: &[f32; 4], x_ptr: *const f32) {
    // SAFETY: the fixed array supplies four aligned input lanes and the caller
    // guarantees the same for `x_ptr`; `lanes` supplies four writable lanes.
    let mut lanes = [0.0f32; 4];
    // SAFETY: `vals` contains four initialized f32 lanes and the caller's
    // contract supplies four readable lanes at `x_ptr`; the output array has
    // four writable lanes.
    unsafe {
        let prod = vmulq_f32(vld1q_f32(vals.as_ptr()), vld1q_f32(x_ptr));
        vst1q_f32(lanes.as_mut_ptr(), prod);
    }
    *acc += lanes[0];
    *acc += lanes[1];
    *acc += lanes[2];
    *acc += lanes[3];
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn dot_q6_k_f32_neon(q6k_blocks: &[u8], x: &[f32], n: usize) -> f32 {
    if !n.is_multiple_of(256) {
        return 0.0;
    }
    let Some(block_bytes) = n
        .checked_div(256)
        .and_then(|blocks| blocks.checked_mul(210))
    else {
        return 0.0;
    };
    if q6k_blocks.len() < block_bytes || x.len() < n {
        return 0.0;
    }
    let n_blocks = n / 256;
    let mut sumf = 0.0f32;

    for b in 0..n_blocks {
        let block = &q6k_blocks[b * 210..];
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        let ql = &block[0..128];
        let qh = &block[128..192];
        let sc = &block[192..208];
        let x_block = &x[b * 256..b * 256 + 256];

        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        let mut sc_off = 0usize;
        let mut x_off = 0usize;
        for _group in 0..2 {
            for l_base in (0..32usize).step_by(16) {
                let mut sums = [0.0f32; 4];
                for l in (l_base..l_base + 16).step_by(4) {
                    let mut vals1 = [0.0f32; 4];
                    let mut vals2 = [0.0f32; 4];
                    let mut vals3 = [0.0f32; 4];
                    let mut vals4 = [0.0f32; 4];
                    for i in 0..4 {
                        let lane = l + i;
                        let ql0 = ql[ql_off + lane] as u32;
                        let ql32 = ql[ql_off + lane + 32] as u32;
                        let qhl = qh[qh_off + lane] as u32;

                        vals1[i] = (((ql0 & 0xF) | ((qhl & 3) << 4)) as i32 - 32) as f32;
                        vals2[i] = (((ql32 & 0xF) | (((qhl >> 2) & 3) << 4)) as i32 - 32) as f32;
                        vals3[i] = (((ql0 >> 4) | (((qhl >> 4) & 3) << 4)) as i32 - 32) as f32;
                        vals4[i] = (((ql32 >> 4) | (((qhl >> 6) & 3) << 4)) as i32 - 32) as f32;
                    }
                    // Baseline bug: 4-lane FMA + horizontal reduction changes the
                    // sub-block rounding enough to trip the llama parity guard on
                    // exact-fraction cases (~1/512 relative). Keep the vector
                    // multiply/load win, but reduce in scalar lane order so the
                    // hot path matches the scalar deferred-scaling contract.
                    // SAFETY: every pointer is derived from a bounds-checked
                    // 256-element activation block and each remaining slice
                    // contains at least the four lanes loaded by the helper.
                    unsafe {
                        ordered_mul_accumulate4(
                            &mut sums[0],
                            &vals1,
                            x_block[x_off + l..].as_ptr(),
                        );
                        ordered_mul_accumulate4(
                            &mut sums[1],
                            &vals2,
                            x_block[x_off + l + 32..].as_ptr(),
                        );
                        ordered_mul_accumulate4(
                            &mut sums[2],
                            &vals3,
                            x_block[x_off + l + 64..].as_ptr(),
                        );
                        ordered_mul_accumulate4(
                            &mut sums[3],
                            &vals4,
                            x_block[x_off + l + 96..].as_ptr(),
                        );
                    }
                }

                let is = l_base / 16;
                let s0 = sc[sc_off + is] as i8 as f32;
                let s1 = sc[sc_off + is + 2] as i8 as f32;
                let s2 = sc[sc_off + is + 4] as i8 as f32;
                let s3 = sc[sc_off + is + 6] as i8 as f32;
                sumf += d * (sums[0] * s0 + sums[1] * s1 + sums[2] * s2 + sums[3] * s3);
            }
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
            x_off += 128;
        }
    }

    sumf
}

/// Dequantize a Q6_K block (256 values in 210 bytes).
///
/// Block layout (llama.cpp `block_q6_K`):
/// ```text
///   bytes   0-127: ql[128]   - low 4 bits (nibble-packed)
///   bytes 128-191: qh[64]    - high 2 bits (4 values per byte)
///   bytes 192-207: sc[16]    - int8 sub-block scales (groups of 16 elements)
///   bytes 208-209: d         - f16 super-scale
/// ```
///
/// The element ordering is interleaved (two 128-element groups):
/// ```text
///   for each group g (0,1), for l in 0..32:
///     q1 = (ql[g*64+l] low nibble | qh[g*32+l] bits 0-1 as high) - 32
///     q2 = (ql[g*64+l+32] low nibble | qh bits 2-3 as high) - 32
///     q3 = (ql[g*64+l] high nibble | qh bits 4-5 as high) - 32
///     q4 = (ql[g*64+l+32] high nibble | qh bits 6-7 as high) - 32
///   outputs go to positions: g*128+l, g*128+l+32, g*128+l+64, g*128+l+96
/// ```
pub fn dequant_q6_k(block: &[u8], out: &mut [f32]) {
    debug_assert!(block.len() >= 210);
    debug_assert!(out.len() >= 256);

    let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
    let ql = &block[0..128];
    let qh = &block[128..192];
    let sc = &block[192..208]; // int8 viewed as i8

    let mut out_off: usize = 0;
    let mut ql_off: usize = 0;
    let mut qh_off: usize = 0;
    let mut sc_off: usize = 0;

    for _group in 0..2 {
        for l in 0..32usize {
            let is = l / 16; // 0 for l=0..15, 1 for l=16..31

            let ql0 = ql[ql_off + l] as u32;
            let ql32 = ql[ql_off + l + 32] as u32;
            let qhl = qh[qh_off + l] as u32;

            let q1 = ((ql0 & 0xF) | ((qhl & 3) << 4)) as i32 - 32;
            let q2 = ((ql32 & 0xF) | (((qhl >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql0 >> 4) | (((qhl >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql32 >> 4) | (((qhl >> 6) & 3) << 4)) as i32 - 32;

            let s1 = d * sc[sc_off + is] as i8 as f32;
            let s2 = d * sc[sc_off + is + 2] as i8 as f32;
            let s3 = d * sc[sc_off + is + 4] as i8 as f32;
            let s4 = d * sc[sc_off + is + 6] as i8 as f32;

            out[out_off + l] = s1 * q1 as f32;
            out[out_off + l + 32] = s2 * q2 as f32;
            out[out_off + l + 64] = s3 * q3 as f32;
            out[out_off + l + 96] = s4 * q4 as f32;
        }
        out_off += 128;
        ql_off += 64;
        qh_off += 32;
        sc_off += 8;
    }
}
