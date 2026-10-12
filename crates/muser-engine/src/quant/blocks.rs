// Quantization format and algorithm provenance is pinned in `THIRD_PARTY.md`;
// the retained upstream notice is `LICENSES/GGML-LLAMA-MIT.txt`.

use super::f16_to_f32;

pub fn dequant_q4_0(block: &[u8], out: &mut [f32]) {
    debug_assert!(block.len() >= 18);
    debug_assert!(out.len() >= 32);

    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let data = &block[2..18];

    for i in 0..16 {
        let byte = data[i];
        // Low nibble → element i (0..15)
        let lo = (byte & 0x0F) as i8 - 8;
        // High nibble → element i+16 (16..31)
        let hi = (byte >> 4) as i8 - 8;
        out[i] = lo as f32 * scale;
        out[i + 16] = hi as f32 * scale;
    }
}

/// Dequantize a Q5_0 block (32 values packed into 22 bytes).
///
/// Block layout: [scale: f16 (2B)] [qh: 4B] [qs: 16B]
/// Q5_0 uses 5 bits per value: 4 bits in qs, 1 bit in qh.
/// Element order: elements 0-15 from low nibbles, 16-31 from high nibbles.
/// Matches llama.cpp / ggml canonical ordering.
pub fn dequant_q5_0(block: &[u8], out: &mut [f32]) {
    debug_assert!(block.len() >= 22);
    debug_assert!(out.len() >= 32);

    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));

    // qh: 4 bytes (32 bits) - 1 bit per value for high bit
    let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);

    // qs: 16 bytes of low 4-bit nibbles (32 nibbles = 2 per byte)
    let qs = &block[6..22];

    for j in 0..16 {
        let byte = qs[j];

        // Low nibble → element j (0..15)
        let lo_nibble = byte & 0x0F;
        let lo_high_bit = (((qh >> j) & 1) << 4) as u8;
        let lo_val = (((lo_high_bit | lo_nibble) as i8) - 16) as f32 * scale;
        out[j] = lo_val;

        // High nibble → element j+16 (16..31)
        let hi_nibble = byte >> 4;
        let hi_high_bit = (((qh >> (j + 16)) & 1) << 4) as u8;
        let hi_val = (((hi_high_bit | hi_nibble) as i8) - 16) as f32 * scale;
        out[j + 16] = hi_val;
    }

    // Sanity checks
    debug_assert!(
        !out.iter().any(|x| x.is_nan()),
        "Q5_0 dequantization produced NaN! scale: {}, qh: 0x{:08x}",
        scale,
        qh
    );
}

/// Dequantize a Q5_1 block (32 values packed into 24 bytes).
///
/// Block layout: [d: f16 (2B)] [m: f16 (2B)] [qh: 4B] [qs: 16B]
/// Like Q5_0 but unsigned with additive min: value = d * quant5 + m
pub fn dequant_q5_1(block: &[u8], out: &mut [f32]) {
    debug_assert!(block.len() >= 24);
    debug_assert!(out.len() >= 32);

    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let m = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    let qs = &block[8..24];

    for j in 0..16 {
        let byte = qs[j];
        let lo_nibble = (byte & 0x0F) as u32;
        let hi_nibble = (byte >> 4) as u32;
        let lo_hb = (qh >> j) & 1;
        let hi_hb = (qh >> (j + 16)) & 1;
        out[j] = d * (lo_nibble | (lo_hb << 4)) as f32 + m;
        out[j + 16] = d * (hi_nibble | (hi_hb << 4)) as f32 + m;
    }
}

/// Transcode a Q5_0 weight tensor to Q8_0 in-place (lossless: all Q5_0
/// integer values ∈ [-16, 15] fit exactly in int8).
///
/// Q5_0 block: 22 bytes — [d: f16 (2B)] [qh: 4B] [qs_lo: 16B]
/// Q8_0 block: 34 bytes — [d: f16 (2B)] [quants: 32 × i8]
///
/// Both blocks have block_size = 32 and use the same float16 scale `d`.
/// The 5-bit integers from Q5_0 are stored verbatim as int8 in Q8_0.
///
/// This is used at GPU weight upload time to replace the expensive GPU-side
/// Q5_0 decode loop (bit manipulation) with the trivially-cheap Q8_0 decode
/// (`float(qs[i]) * scale`), trading 30% more memory bandwidth for 3× fewer
/// GPU decode instructions.
pub fn transcode_q5_to_q8(q5_data: &[u8]) -> Vec<u8> {
    const Q5_BLOCK: usize = 22;
    const Q8_BLOCK: usize = 34;
    assert!(
        q5_data.len().is_multiple_of(Q5_BLOCK),
        "Q5_0 data not block-aligned"
    );
    let n_blocks = q5_data.len() / Q5_BLOCK;
    let mut out = vec![0u8; n_blocks * Q8_BLOCK];

    for b in 0..n_blocks {
        let src = &q5_data[b * Q5_BLOCK..];
        let dst = &mut out[b * Q8_BLOCK..];

        // Copy the fp16 scale unchanged (bytes 0-1)
        dst[0] = src[0];
        dst[1] = src[1];

        // Decode Q5_0 int5 → int8 (lossless, range [-16..15])
        let qh = u32::from_le_bytes([src[2], src[3], src[4], src[5]]);
        let qs = &src[6..22]; // 16 bytes = 32 nibbles

        for j in 0..16 {
            let packed = qs[j];
            let lo0 = (packed & 0x0F) as u32;
            let lo1 = ((packed >> 4) & 0x0F) as u32;
            let hi0 = (qh >> j) & 1;
            let hi1 = (qh >> (j + 16)) & 1;
            let v0 = (lo0 | (hi0 << 4)) as i8 - 16;
            let v1 = (lo1 | (hi1 << 4)) as i8 - 16;
            dst[2 + j] = v0 as u8;
            dst[2 + j + 16] = v1 as u8;
        }
    }
    out
}

/// Dequantize a Q8_0 block (32 values packed into 34 bytes).
///
/// Block layout: [scale: f16 (2B)] [quants: 32 × i8]
pub fn dequant_q8_0(block: &[u8], out: &mut [f32]) {
    debug_assert!(block.len() >= 34);
    debug_assert!(out.len() >= 32);

    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    for i in 0..32 {
        out[i] = block[2 + i] as i8 as f32 * scale;
    }
}
