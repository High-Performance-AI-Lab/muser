// Quantization format and algorithm provenance is pinned in `THIRD_PARTY.md`;
// the retained upstream notice is `LICENSES/GGML-LLAMA-MIT.txt`.

use super::f16_to_f32;
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

mod q6;

pub use q6::{dequant_q6_k, dot_q6_k_f32};

pub fn dequant_q4_k(block: &[u8], out: &mut [f32]) {
    debug_assert!(block.len() >= 144);
    debug_assert!(out.len() >= 256);

    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let scales = &block[4..16];
    let qs = &block[16..144];

    // get_scale_min_k4: extract 6-bit sc and m for sub-block j (0..7)
    let get_scale_min = |j: usize| -> (f32, f32) {
        let (sc, m) = if j < 4 {
            (scales[j] & 0x3F, scales[j + 4] & 0x3F)
        } else {
            let sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
            let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
            (sc, m)
        };
        (d * sc as f32, dmin * m as f32)
    };

    // 4 outer groups of 64 elements, 2 sub-blocks per group, 32 qs bytes per group.
    let mut q_off = 0usize;
    let mut is = 0usize;
    let mut base = 0usize;
    while base < 256 {
        let (d1, m1) = get_scale_min(is);
        let (d2, m2) = get_scale_min(is + 1);
        for l in 0..32 {
            let q = qs[q_off + l];
            out[base + l] = d1 * (q & 0x0F) as f32 - m1;
            out[base + l + 32] = d2 * (q >> 4) as f32 - m2;
        }
        q_off += 32;
        is += 2;
        base += 64;
    }
}

/// Dequantize a Q5_K block (256 values packed into 176 bytes).
///
/// Q5_K layout (per ggml `block_q5_K`):
///   [d: f16 (2B)] [dmin: f16 (2B)] [scales: 12B] [qh: 32B] [qs: 128B]
///
/// Identical to Q4_K except each value has a 5th high bit stored in the
/// qh plane (256 bits = 32 bytes). The qs layout and scale packing are
/// shared with Q4_K.
///
/// Value formula: y = d * sc * (nibble | (qh_bit << 4)) - dmin * m
pub fn dequant_q5_k(block: &[u8], out: &mut [f32]) {
    debug_assert!(block.len() >= 176);
    debug_assert!(out.len() >= 256);

    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let scales = &block[4..16];
    let qh = &block[16..48];
    let qs = &block[48..176];

    let get_scale_min = |j: usize| -> (f32, f32) {
        let (sc, m) = if j < 4 {
            (scales[j] & 0x3F, scales[j + 4] & 0x3F)
        } else {
            let sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
            let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
            (sc, m)
        };
        (d * sc as f32, dmin * m as f32)
    };

    let mut q_off = 0usize;
    let mut is = 0usize;
    let mut base = 0usize;
    let mut u1 = 1u8;
    let mut u2 = 2u8;
    while base < 256 {
        let (d1, m1) = get_scale_min(is);
        let (d2, m2) = get_scale_min(is + 1);
        for l in 0..32 {
            let q = qs[q_off + l];
            let lo_nibble = (q & 0x0F) as u32;
            let hi_nibble = (q >> 4) as u32;
            let hb_lo = if qh[l] & u1 != 0 { 16u32 } else { 0 };
            let hb_hi = if qh[l] & u2 != 0 { 16u32 } else { 0 };
            out[base + l] = d1 * (lo_nibble + hb_lo) as f32 - m1;
            out[base + l + 32] = d2 * (hi_nibble + hb_hi) as f32 - m2;
        }
        q_off += 32;
        is += 2;
        base += 64;
        u1 <<= 2;
        u2 <<= 2;
    }
}

// ── Dot products on quantized data ─────────────────────────────────────

/// Dot product of Q8_0 × f32 vectors. Avoids full dequantization.
///
/// For each Q8 block of 32 values: `sum(q[i] * x[i]) * scale`.
/// This is the hot inner loop of attention and FFN matmul.
pub fn dot_q8_f32(q8_blocks: &[u8], x: &[f32], n: usize) -> f32 {
    debug_assert!(n.is_multiple_of(32));
    let n_blocks = n / 32;
    let mut acc = 0.0f32;

    for b in 0..n_blocks {
        let block = &q8_blocks[b * 34..];
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let quants = &block[2..34];
        let xslice = &x[b * 32..b * 32 + 32];

        // Inner product of 32 int8 × f32 values, scaled
        let mut block_sum = 0.0f32;
        for i in 0..32 {
            block_sum += quants[i] as i8 as f32 * xslice[i];
        }
        acc += block_sum * scale;
    }
    acc
}

/// Dot product of Q4_0 × f32 vectors.
pub fn dot_q4_0_f32(q4_blocks: &[u8], x: &[f32], n: usize) -> f32 {
    debug_assert!(n.is_multiple_of(32));
    let n_blocks = n / 32;
    let mut acc = 0.0f32;

    for b in 0..n_blocks {
        let block = &q4_blocks[b * 18..];
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let data = &block[2..18];
        let xslice = &x[b * 32..b * 32 + 32];

        let mut block_sum = 0.0f32;
        for i in 0..16 {
            let byte = data[i];
            let lo = (byte & 0x0F) as i8 - 8;
            let hi = (byte >> 4) as i8 - 8;
            block_sum += lo as f32 * xslice[i];
            block_sum += hi as f32 * xslice[i + 16];
        }
        acc += block_sum * scale;
    }
    acc
}

/// Dot product of Q4_K × f32 vectors without materializing a dequantized row.
pub fn dot_q4_k_f32(q4k_blocks: &[u8], x: &[f32], n: usize) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return dot_q4_k_f32_neon(q4k_blocks, x, n);
    }
    #[allow(unreachable_code)]
    dot_q4_k_f32_scalar(q4k_blocks, x, n)
}

/// Q4_K dot product matching llama.cpp's exact deferred-scaling accumulation.
///
/// Attribution: accumulation order from ggml-org/llama.cpp
/// kernel_mul_mv_q4_K_f32_impl (ggml-metal.metal).
///
/// Key difference from dot_q4_k_f32_scalar: factors d and dmin OUT of the
/// inner loop, accumulating raw q*x products first, then scaling once per
/// sub-block. This produces different f32 rounding than the per-element
/// `(d * sc * nibble - dmin * m) * x` formula used by the scalar path.
pub fn dot_q4_k_f32_llama(q4k_blocks: &[u8], x: &[f32], n: usize) -> f32 {
    debug_assert!(n.is_multiple_of(256));
    let n_blocks = n / 256;
    let mut sumf = 0.0f32;

    for ib in 0..n_blocks {
        let block = &q4k_blocks[ib * 144..];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let qs = &block[16..144];
        let x_block = &x[ib * 256..ib * 256 + 256];

        // Extract scales
        let get_scale_min = |j: usize| -> (u8, u8) {
            if j < 4 {
                (scales[j] & 0x3F, scales[j + 4] & 0x3F)
            } else {
                let sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
                let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
                (sc, m)
            }
        };

        // Deferred scaling: accumulate raw q*x per sub-block, then apply d*sc.
        // Separately accumulate x-sums per sub-block for the dmin*m term.
        // This matches llama.cpp's kernel_mul_mv_q4_K_f32_impl pattern.
        let mut q_off = 0usize;
        let mut is = 0usize;
        let mut base = 0usize;
        while base < 256 {
            let (sc_lo, m_lo) = get_scale_min(is);
            let (sc_hi, m_hi) = get_scale_min(is + 1);

            // Accumulate raw q*x products AND x-sums (no d/dmin yet)
            let mut acc_lo = 0.0f32;
            let mut acc_hi = 0.0f32;
            let mut xsum_lo = 0.0f32;
            let mut xsum_hi = 0.0f32;
            for l in 0..32 {
                let q = qs[q_off + l];
                let q_lo = (q & 0x0F) as f32;
                let q_hi = (q >> 4) as f32;
                let x_lo = x_block[base + l];
                let x_hi = x_block[base + l + 32];
                acc_lo += q_lo * x_lo;
                acc_hi += q_hi * x_hi;
                xsum_lo += x_lo;
                xsum_hi += x_hi;
            }

            // Deferred scaling: d * sc * Σ(q*x) - dmin * m * Σ(x)
            sumf += d * (sc_lo as f32 * acc_lo + sc_hi as f32 * acc_hi)
                - dmin * (m_lo as f32 * xsum_lo + m_hi as f32 * xsum_hi);

            q_off += 32;
            is += 2;
            base += 64;
        }
    }

    sumf
}

fn dot_q4_k_f32_scalar(q4k_blocks: &[u8], x: &[f32], n: usize) -> f32 {
    debug_assert!(n.is_multiple_of(256));
    let n_blocks = n / 256;
    let mut acc = 0.0f32;

    for b in 0..n_blocks {
        let block = &q4k_blocks[b * 144..];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let qs = &block[16..144];
        let x_block = &x[b * 256..b * 256 + 256];

        let get_scale_min = |j: usize| -> (f32, f32) {
            let (sc, m) = if j < 4 {
                (scales[j] & 0x3F, scales[j + 4] & 0x3F)
            } else {
                let sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
                let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
                (sc, m)
            };
            (d * sc as f32, dmin * m as f32)
        };

        let mut q_off = 0usize;
        let mut is = 0usize;
        let mut base = 0usize;
        while base < 256 {
            let (d1, m1) = get_scale_min(is);
            let (d2, m2) = get_scale_min(is + 1);
            for l in 0..32 {
                let q = qs[q_off + l];
                let q_lo = (q & 0x0F) as f32;
                let q_hi = (q >> 4) as f32;
                acc += d1.mul_add(q_lo, -m1) * x_block[base + l];
                acc += d2.mul_add(q_hi, -m2) * x_block[base + l + 32];
            }
            q_off += 32;
            is += 2;
            base += 64;
        }
    }

    acc
}

/// Q5_K fused dot product: dot(Q5_K_block, f32_vec) without dequantizing to f32.
///
/// Q5_K layout (176 bytes / 256 elements):
/// ```text
///   - bytes 0-1:   d (f16)
///   - bytes 2-3:   dmin (f16)
///   - bytes 4-15:  scales[12] (packed 6-bit scale/min, same as Q4_K)
///   - bytes 16-47: qh[32] (high bits: 256 bits = 32 bytes)
///   - bytes 48-175: qs[128] (low 4-bit nibbles, same as Q4_K)
/// ```
///
/// Value: y = d * sc * (nibble | (qh_bit << 4)) - dmin * m
pub fn dot_q5_k_f32(q5k_blocks: &[u8], x: &[f32], n: usize) -> f32 {
    debug_assert!(n.is_multiple_of(256));
    let n_blocks = n / 256;
    let mut acc = 0.0f32;

    for b in 0..n_blocks {
        let block = &q5k_blocks[b * 176..];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let qh = &block[16..48];
        let qs = &block[48..176];
        let x_block = &x[b * 256..b * 256 + 256];

        let get_scale_min = |j: usize| -> (f32, f32) {
            let (sc, m) = if j < 4 {
                (scales[j] & 0x3F, scales[j + 4] & 0x3F)
            } else {
                let sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
                let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
                (sc, m)
            };
            (d * sc as f32, dmin * m as f32)
        };

        let mut q_off = 0usize;
        let mut is = 0usize;
        let mut base = 0usize;
        let mut u1 = 1u8;
        let mut u2 = 2u8;
        while base < 256 {
            let (d1, m1) = get_scale_min(is);
            let (d2, m2) = get_scale_min(is + 1);
            for l in 0..32 {
                let q = qs[q_off + l];
                let lo = (q & 0x0F) as u32;
                let hi = (q >> 4) as u32;
                let hb_lo = if qh[l] & u1 != 0 { 16u32 } else { 0 };
                let hb_hi = if qh[l] & u2 != 0 { 16u32 } else { 0 };
                acc += (d1 * (lo + hb_lo) as f32 - m1) * x_block[base + l];
                acc += (d2 * (hi + hb_hi) as f32 - m2) * x_block[base + l + 32];
            }
            q_off += 32;
            is += 2;
            base += 64;
            u1 <<= 2;
            u2 <<= 2;
        }
    }

    acc
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
/// Reduce one AArch64 NEON vector.
///
/// # Safety
/// The caller must execute on AArch64 with NEON support (mandatory for the
/// architecture targeted by this compilation unit).
unsafe fn horizontal_sum_f32x4(v: float32x4_t) -> f32 {
    // SAFETY: this function exists only in the AArch64 build, whose baseline
    // architecture includes NEON, and `v` is an initialized vector value.
    unsafe { vaddvq_f32(v) }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
/// Decode sixteen Q4 nibbles and accumulate four vector products.
///
/// # Safety
/// `x_ptr` must point to at least sixteen initialized, properly aligned `f32`
/// values. The caller must run on AArch64 with NEON support.
unsafe fn accumulate_q4_k_nibbles_16(
    accs: &mut [float32x4_t; 4],
    nibbles: uint8x16_t,
    x_ptr: *const f32,
    scale: f32,
    min: f32,
) {
    // SAFETY: AArch64 guarantees NEON. The caller provides sixteen readable
    // activation lanes; pointer additions remain within that allocation, and
    // all other operations consume initialized vector values only.
    unsafe {
        let lo16 = vmovl_u8(vget_low_u8(nibbles));
        let hi16 = vmovl_u8(vget_high_u8(nibbles));

        let q32_0 = vmovl_u16(vget_low_u16(lo16));
        let q32_1 = vmovl_u16(vget_high_u16(lo16));
        let q32_2 = vmovl_u16(vget_low_u16(hi16));
        let q32_3 = vmovl_u16(vget_high_u16(hi16));

        let neg_min = vdupq_n_f32(-min);
        let vals0 = vfmaq_n_f32(neg_min, vcvtq_f32_u32(q32_0), scale);
        let vals1 = vfmaq_n_f32(neg_min, vcvtq_f32_u32(q32_1), scale);
        let vals2 = vfmaq_n_f32(neg_min, vcvtq_f32_u32(q32_2), scale);
        let vals3 = vfmaq_n_f32(neg_min, vcvtq_f32_u32(q32_3), scale);

        accs[0] = vfmaq_f32(accs[0], vals0, vld1q_f32(x_ptr.add(0)));
        accs[1] = vfmaq_f32(accs[1], vals1, vld1q_f32(x_ptr.add(4)));
        accs[2] = vfmaq_f32(accs[2], vals2, vld1q_f32(x_ptr.add(8)));
        accs[3] = vfmaq_f32(accs[3], vals3, vld1q_f32(x_ptr.add(12)));
    }
}

#[cfg(target_arch = "aarch64")]
fn dot_q4_k_f32_neon(q4k_blocks: &[u8], x: &[f32], n: usize) -> f32 {
    if !n.is_multiple_of(256) {
        return 0.0;
    }
    let Some(block_bytes) = n
        .checked_div(256)
        .and_then(|blocks| blocks.checked_mul(144))
    else {
        return 0.0;
    };
    if q4k_blocks.len() < block_bytes || x.len() < n {
        return 0.0;
    }
    let n_blocks = n / 256;
    let mut acc = 0.0f32;

    for b in 0..n_blocks {
        let block = &q4k_blocks[b * 144..];
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let qs = &block[16..144];
        let x_block = &x[b * 256..b * 256 + 256];

        let get_scale_min = |j: usize| -> (f32, f32) {
            let (sc, m) = if j < 4 {
                (scales[j] & 0x3F, scales[j + 4] & 0x3F)
            } else {
                let sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
                let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
                (sc, m)
            };
            (d * sc as f32, dmin * m as f32)
        };

        // SAFETY: this function is AArch64-only and NEON is part of the
        // architecture baseline; the intrinsic has no memory operand.
        let zero = unsafe { vdupq_n_f32(0.0) };
        let mut accs = [zero, zero, zero, zero];
        // SAFETY: as above, this is an AArch64-only register construction with
        // no pointer or memory preconditions.
        let nibble_mask = unsafe { vdupq_n_u8(0x0F) };
        let mut q_off = 0usize;
        let mut is = 0usize;
        let mut base = 0usize;
        while base < 256 {
            let (d1, m1) = get_scale_min(is);
            let (d2, m2) = get_scale_min(is + 1);
            for l in (0..32).step_by(16) {
                // SAFETY: `qs` and `x_block` are bounds-checked 256-value
                // blocks. Each loop offset leaves at least 16 bytes/floats for
                // the vector loads performed here and by the helper.
                unsafe {
                    let q = vld1q_u8(qs[q_off + l..].as_ptr());
                    let lo = vandq_u8(q, nibble_mask);
                    let hi = vshrq_n_u8(q, 4);
                    accumulate_q4_k_nibbles_16(&mut accs, lo, x_block[base + l..].as_ptr(), d1, m1);
                    accumulate_q4_k_nibbles_16(
                        &mut accs,
                        hi,
                        x_block[base + l + 32..].as_ptr(),
                        d2,
                        m2,
                    );
                }
            }
            q_off += 32;
            is += 2;
            base += 64;
        }
        // SAFETY: initialized NEON accumulators are reduced on AArch64, where
        // the required feature is guaranteed.
        acc += unsafe {
            horizontal_sum_f32x4(accs[0])
                + horizontal_sum_f32x4(accs[1])
                + horizontal_sum_f32x4(accs[2])
                + horizontal_sum_f32x4(accs[3])
        };
    }

    acc
}

// ── IQ3_K inline dot product ─────────────────────────────────────────────────

const IQ3NL: [i8; 16] = [
    -63, -40, -23, -10, 1, 13, 28, 47, -59, -36, -19, -6, 5, 17, 32, 51,
];

/// Fused IQ3_K dot product: Σ dequant(blocks[j]) * x[j], no temp allocation.
///
/// `blocks` must be `(n / 256) * 110` bytes. `n` must be a multiple of 256.
pub fn dot_iq3_k_f32(blocks: &[u8], x: &[f32], n: usize) -> f32 {
    let nb = n / 256;
    let mut acc = 0.0f32;

    for ib in 0..nb {
        let blk = &blocks[ib * 110..(ib + 1) * 110];
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let mut extra = u16::from_le_bytes([blk[2], blk[3]]);
        let mut sh = u16::from_le_bytes([blk[4], blk[5]]);
        let scales_l = &blk[6..14];
        let qh = &blk[78..110];
        let xi_base = ib * 256;

        for g in 0..8usize {
            let dl1 =
                d * (2.0 * (scales_l[g] & 0xf) as f32 + 1.0) * if sh & 1 != 0 { -1.0 } else { 1.0 };
            let dl2 =
                d * (2.0 * (scales_l[g] >> 4) as f32 + 1.0) * if sh & 2 != 0 { -1.0 } else { 1.0 };
            sh >>= 2;

            let v1 = if extra & 1 != 0 { 8 } else { 0 };
            let v2 = if extra & 2 != 0 { 8 } else { 0 };
            extra >>= 2;

            let shift_l = 2 * (g % 4);
            let shift_h = g;
            let qs = &blk[14 + (g / 4) * 32..];
            let yoff = g * 32;

            for j in 0..16usize {
                let idx1 = (((qs[j] >> shift_l) & 3) | (((qh[j] >> shift_h) & 1) << 2)) as usize;
                let idx2 =
                    (((qs[j + 16] >> shift_l) & 3) | (((qh[j + 16] >> shift_h) & 1) << 2)) as usize;
                acc += dl1 * IQ3NL[v1 + idx1] as f32 * x[xi_base + yoff + j];
                acc += dl2 * IQ3NL[v2 + idx2] as f32 * x[xi_base + yoff + j + 16];
            }
        }
    }

    acc
}
