/// IEEE 754 half-precision → single-precision.
/// Bit manipulation from Hacker's Delight §17/Jeroen van der Zijp.
#[inline(always)]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x03FF) as u32;

    let f_bits = if exp == 0 {
        if mant == 0 {
            sign << 31 // ±0
        } else {
            // Subnormal: renormalize
            let mut e = 0u32;
            let mut m = mant;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e += 1;
            }
            // fp16 subnormal value = mant × 2^(-24).
            // After e left-shifts, leading 1 is at bit 10.
            // Highest bit of original mant was at position (10 - e).
            // fp32 biased exp = (10 - e) + 103 = 113 - e = 127 - 14 - e.
            let e = 127 - 14 - e;
            let m = (m & 0x03FF) << 13;
            (sign << 31) | (e << 23) | m
        }
    } else if exp == 31 {
        // Inf/NaN
        (sign << 31) | (0xFF << 23) | (mant << 13)
    } else {
        // Normal
        let e = exp + 127 - 15;
        (sign << 31) | (e << 23) | (mant << 13)
    };

    f32::from_bits(f_bits)
}

/// IEEE 754 single-precision → half-precision (with subnormal support).
/// Matches the inverse of `f16_to_f32` above.
#[inline]
pub fn f32_to_f16_quant(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let mant = b & 0x7fffff;
    if exp == 0 {
        return sign; // f32 zero/subnormal → f16 zero
    }
    if exp >= 143 {
        return sign | 0x7c00; // overflow → inf
    }
    if exp > 112 {
        // Normal f16 range: f16 biased exponent = exp - 112
        return sign | (((exp - 112) as u16) << 10) | ((mant >> 13) as u16);
    }
    // Subnormal f16 range: f32 biased exp 103..=112
    // f16 subnormal value = 2^(-14) * (mantissa / 1024)
    // Reconstruct full mantissa with implicit leading 1, then shift down.
    let shift = 126 - exp; // 14..=23 for exp 103..=112
    if shift > 23 {
        return sign; // too small for f16 subnormal
    }
    let full = (1u32 << 23) | mant; // add implicit leading 1
    let f16_mant = (full >> shift) as u16;
    sign | f16_mant
}

/// SiLU/Swish: x · σ(x) — used in Qwen2 FFN gate.
#[inline(always)]
pub fn silu_fast(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
