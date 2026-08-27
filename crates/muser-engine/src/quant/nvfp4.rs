//! CPU oracle for NVIDIA NVFP4 weights.
//!
//! The product format keeps E2M1 values packed two per byte, one raw E4M3FN
//! scale per 16 values, and one f32 `scale2` per tensor.  The operation order
//! is pinned to the ModelOpt/MLX reference: `(e2m1 * e4m3fn) * scale2`.

const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

const FP4_MAX: f32 = 6.0;
const E4M3FN_MAX: f32 = 448.0;
const W4A4_INTEGER_SCALE_INV: f32 = 1.0 / 1_048_576.0;
const A16_Q8_INTEGER_SCALE_INV: f32 = 1.0 / 1_024.0;

#[inline]
fn e2m1_q1(nibble: u8) -> i32 {
    let magnitude = match nibble & 7 {
        value @ 0..=4 => i32::from(value),
        5 => 6,
        6 => 8,
        _ => 12,
    };
    if nibble & 8 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

/// Decode finite E4M3FN into exact signed units of 2^-9. This gives every
/// block scale a common integer denominator, including subnormals.
#[inline]
fn e4m3fn_q9(byte: u8) -> i32 {
    let exponent = (byte >> 3) & 0x0f;
    let mantissa = byte & 7;
    assert!(
        exponent != 0x0f || mantissa != 7,
        "E4M3FN NaN in exact contraction"
    );
    let magnitude = if exponent == 0 {
        i32::from(mantissa)
    } else {
        i32::from(8 + mantissa) << (exponent - 1)
    };
    if byte & 0x80 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

/// Q8_K activation block used by the strict weight-only NVFP4 contraction.
/// The scale and rounding contract intentionally matches llama.cpp's Q8_K
/// reference and the in-tree CUDA/Metal cross-vendor K-quant path.
#[derive(Clone, Debug, PartialEq)]
pub struct Nvfp4Q8Block {
    pub d: f32,
    pub qs: [i8; 256],
}

#[inline]
fn q8_nearest(scale: f32, value: f32) -> i32 {
    let rounded = scale.mul_add(value, 12_582_912.0);
    (rounded.to_bits() as i32 & 0x007f_ffff) - 0x0040_0000
}

/// Quantize one model-dtype activation super-block. vLLM's W4A16 Marlin
/// projection receives F16 activations, so the f16 boundary is part of the
/// cross-vendor contract even though Muser stores its activation arena as f32.
pub fn quantize_nvfp4_q8_block(input: &[f32]) -> Nvfp4Q8Block {
    assert_eq!(input.len(), 256);
    let mut rounded = [0.0f32; 256];
    let mut signed_max = 0.0f32;
    let mut abs_max = 0.0f32;
    for (slot, &value) in rounded.iter_mut().zip(input) {
        *slot = half::f16::from_f32(value).to_f32();
        let magnitude = slot.abs();
        if magnitude > abs_max {
            abs_max = magnitude;
            signed_max = *slot;
        }
    }
    if abs_max == 0.0 {
        return Nvfp4Q8Block {
            d: 0.0,
            qs: [0; 256],
        };
    }
    let inverse_scale = -127.0f32 / signed_max;
    let mut qs = [0i8; 256];
    for (quant, value) in qs.iter_mut().zip(rounded) {
        *quant = q8_nearest(inverse_scale, value).min(127) as i8;
    }
    Nvfp4Q8Block {
        d: 1.0f32 / inverse_scale,
        qs,
    }
}

#[inline]
pub fn e2m1_to_f32(nibble: u8) -> f32 {
    E2M1[usize::from(nibble & 0x0f)]
}

/// Decode one finite E4M3FN byte. The two NaN encodings return NaN so the
/// artifact loader can reject them before any inference dispatch.
#[inline]
pub fn e4m3fn_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (byte >> 3) & 0x0f;
    let mantissa = byte & 0x07;
    let magnitude = if exponent == 0 {
        f32::from(mantissa) * (1.0 / 512.0)
    } else if exponent == 0x0f && mantissa == 0x07 {
        return f32::NAN;
    } else {
        (1.0 + f32::from(mantissa) * 0.125) * 2.0f32.powi(i32::from(exponent) - 7)
    };
    sign * magnitude
}

/// Round an f32 to finite E4M3FN with ties-to-even and saturation. The W4A4
/// activation oracle uses this for its dynamic group scale. A small exhaustive
/// search keeps the rounding contract obvious; decode kernels use the same
/// finite-value table with a bounded binary search.
#[inline]
pub fn e4m3fn_from_f32(value: f32) -> u8 {
    if value.is_nan() {
        return 0x7f;
    }
    let sign = if value.is_sign_negative() { 0x80 } else { 0 };
    let magnitude = value.abs().min(E4M3FN_MAX);
    let mut best = 0u8;
    let mut best_distance = f32::INFINITY;
    for code in 0u8..=0x7e {
        let distance = (e4m3fn_to_f32(code) - magnitude).abs();
        if distance < best_distance || (distance == best_distance && code & 1 == 0 && best & 1 != 0)
        {
            best = code;
            best_distance = distance;
        }
    }
    sign | best
}

/// ModelOpt's pinned E2M1 boundary contract. The asymmetric inclusive bounds
/// are the ties-to-even choices at each representable midpoint.
#[inline]
pub fn e2m1_from_f32(value: f32) -> u8 {
    let sign = if value.is_sign_negative() { 8 } else { 0 };
    let magnitude = value.abs();
    let code = if magnitude <= 0.25 {
        0
    } else if magnitude < 0.75 {
        1
    } else if magnitude <= 1.25 {
        2
    } else if magnitude < 1.75 {
        3
    } else if magnitude <= 2.5 {
        4
    } else if magnitude < 3.5 {
        5
    } else if magnitude <= 5.0 {
        6
    } else {
        7
    };
    sign | code
}

#[inline]
fn activation_group_codes(input: &[f32], input_scale_inv: f32) -> ([u8; 16], u8) {
    assert_eq!(input.len(), 16);
    assert!(input_scale_inv.is_finite() && input_scale_inv > 0.0);
    let mut rounded = [0.0f32; 16];
    let mut abs_max = 0.0f32;
    for (slot, &value) in rounded.iter_mut().zip(input) {
        // vLLM's compressed-tensors linear receives F16 model activations.
        *slot = half::f16::from_f32(value).to_f32();
        abs_max = abs_max.max(slot.abs());
    }
    let normalized_max = abs_max * (1.0f32 / FP4_MAX);
    let scale_code = e4m3fn_from_f32((normalized_max * input_scale_inv).min(E4M3FN_MAX));
    let scale = e4m3fn_to_f32(scale_code);
    let mut codes = [0u8; 16];
    if scale == 0.0 {
        for (code, &value) in codes.iter_mut().zip(&rounded) {
            *code = if value.is_sign_negative() { 8 } else { 0 };
        }
    } else {
        for (code, &value) in codes.iter_mut().zip(&rounded) {
            let sign = if value.is_sign_negative() { 8 } else { 0 };
            let magnitude = value.abs() * input_scale_inv;
            let magnitude_code = if magnitude <= scale * 0.25 {
                0
            } else if magnitude < scale * 0.75 {
                1
            } else if magnitude <= scale * 1.25 {
                2
            } else if magnitude < scale * 1.75 {
                3
            } else if magnitude <= scale * 2.5 {
                4
            } else if magnitude < scale * 3.5 {
                5
            } else if magnitude <= scale * 5.0 {
                6
            } else {
                7
            };
            *code = sign | magnitude_code;
        }
    }
    (codes, scale_code)
}

/// Quantize F16-rounded activations into packed group-16 E2M1 plus E4M3FN
/// scales. This is the CPU fixture oracle for compressed-tensors W4A4.
pub fn quantize_nvfp4_activation(
    input: &[f32],
    input_scale_inv: f32,
    packed: &mut [u8],
    scales: &mut [u8],
) {
    assert_eq!(input.len() % 16, 0);
    assert_eq!(packed.len(), input.len() / 2);
    assert_eq!(scales.len(), input.len() / 16);
    for group in 0..scales.len() {
        let (codes, scale_code) =
            activation_group_codes(&input[group * 16..(group + 1) * 16], input_scale_inv);
        scales[group] = scale_code;
        for pair in 0..8 {
            packed[group * 8 + pair] = codes[pair * 2] | (codes[pair * 2 + 1] << 4);
        }
    }
}

#[inline]
pub fn dequant_nvfp4_row(packed: &[u8], scales: &[u8], scale2: f32, out: &mut [f32]) {
    assert_eq!(out.len() % 16, 0, "NVFP4 rows are grouped by 16");
    assert_eq!(packed.len(), out.len() / 2);
    assert_eq!(scales.len(), out.len() / 16);
    for (group, &scale_byte) in scales.iter().enumerate() {
        let scale = e4m3fn_to_f32(scale_byte);
        let packed_base = group * 8;
        let output_base = group * 16;
        for pair in 0..8 {
            let byte = packed[packed_base + pair];
            out[output_base + pair * 2] = e2m1_to_f32(byte) * scale * scale2;
            out[output_base + pair * 2 + 1] = e2m1_to_f32(byte >> 4) * scale * scale2;
        }
    }
}

#[inline]
pub fn dot_nvfp4_f32(packed: &[u8], scales: &[u8], scale2: f32, x: &[f32]) -> f32 {
    assert_eq!(x.len() % 16, 0, "NVFP4 rows are grouped by 16");
    assert_eq!(packed.len(), x.len() / 2);
    assert_eq!(scales.len(), x.len() / 16);
    let mut sum = 0.0f32;
    for (group, &scale_byte) in scales.iter().enumerate() {
        let scale = e4m3fn_to_f32(scale_byte);
        let packed_base = group * 8;
        let input_base = group * 16;
        for pair in 0..8 {
            let byte = packed[packed_base + pair];
            let weight0 = e2m1_to_f32(byte) * scale * scale2;
            let weight1 = e2m1_to_f32(byte >> 4) * scale * scale2;
            sum += weight0 * x[input_base + pair * 2];
            sum += weight1 * x[input_base + pair * 2 + 1];
        }
    }
    sum
}

/// Producer-native block-dot oracle. One E4M3 scale is applied after each
/// 16-value E2M1 dot, followed by the tensor scale.
#[inline]
pub fn dot_nvfp4_block_fused_f32(packed: &[u8], scales: &[u8], scale2: f32, x: &[f32]) -> f32 {
    assert_eq!(x.len() % 16, 0, "NVFP4 rows are grouped by 16");
    assert_eq!(packed.len(), x.len() / 2);
    assert_eq!(scales.len(), x.len() / 16);
    let mut sum = 0.0f32;
    for (group, &scale_byte) in scales.iter().enumerate() {
        let packed_base = group * 8;
        let input_base = group * 16;
        let mut block_sum = 0.0f32;
        for pair in 0..8 {
            let byte = packed[packed_base + pair];
            block_sum += e2m1_to_f32(byte) * x[input_base + pair * 2];
            block_sum += e2m1_to_f32(byte >> 4) * x[input_base + pair * 2 + 1];
        }
        sum += block_sum * e4m3fn_to_f32(scale_byte);
    }
    sum * scale2
}

/// Cross-vendor W4A4 projection oracle. E2M1 values use signed Q1 integers and
/// finite E4M3FN scales use signed Q9 integers, so the complete contraction is
/// one order-free i64 sum with a fixed 2^-20 denominator. Only the two tensor
/// scales remain floating point, and they are applied in the pinned order.
#[inline]
pub fn dot_nvfp4_w4a4_f32(
    packed: &[u8],
    scales: &[u8],
    weight_scale2: f32,
    input_scale_inv: f32,
    x: &[f32],
) -> f32 {
    assert_eq!(x.len() % 16, 0, "NVFP4 rows are grouped by 16");
    assert_eq!(packed.len(), x.len() / 2);
    assert_eq!(scales.len(), x.len() / 16);
    let mut sum = 0i64;
    for (group, &weight_scale_byte) in scales.iter().enumerate() {
        let input_base = group * 16;
        let packed_base = group * 8;
        let (activation_codes, activation_scale_byte) =
            activation_group_codes(&x[input_base..input_base + 16], input_scale_inv);
        let mut block_sum = 0i32;
        for pair in 0..8 {
            let byte = packed[packed_base + pair];
            block_sum += e2m1_q1(byte) * e2m1_q1(activation_codes[pair * 2]);
            block_sum += e2m1_q1(byte >> 4) * e2m1_q1(activation_codes[pair * 2 + 1]);
        }
        sum += i64::from(block_sum)
            * i64::from(e4m3fn_q9(weight_scale_byte))
            * i64::from(e4m3fn_q9(activation_scale_byte));
    }
    // FlashInfer CUTLASS writes the compressed-tensors linear result in the
    // model dtype (F16). Preserve that boundary even though Muser's activation
    // arena is f32, otherwise Q/K RMS normalization amplifies hidden low bits.
    let scaled = (sum as f32) * W4A4_INTEGER_SCALE_INV;
    let scaled = scaled * weight_scale2;
    let scaled = scaled * (1.0 / input_scale_inv);
    half::f16::from_f32(scaled).to_f32()
}

/// Cross-vendor weight-only projection oracle.
///
/// Each 256-value activation block is quantized once to Q8_K. Within that
/// block, doubled E2M1 codes (Q1), the Q8 integers, and raw E4M3FN scale codes
/// (Q9) form one order-free i64 contraction. The only non-exact operations are
/// then pinned as four scalar IEEE-f32 FMAs: fixed 2^-10 conversion, activation
/// scale, tensor weight scale, and sequential block accumulation. The result is
/// rounded to the producer's F16 model-dtype boundary.
#[inline]
pub fn dot_nvfp4_a16_q8_f32(packed: &[u8], scales: &[u8], weight_scale2: f32, x: &[f32]) -> f32 {
    assert_eq!(x.len() % 256, 0, "Q8 activation blocks contain 256 values");
    assert_eq!(packed.len(), x.len() / 2);
    assert_eq!(scales.len(), x.len() / 16);
    let mut total = 0.0f32;
    for block_index in 0..x.len() / 256 {
        let input_base = block_index * 256;
        let q8 = quantize_nvfp4_q8_block(&x[input_base..input_base + 256]);
        if q8.d == 0.0 {
            continue;
        }
        let mut integer_total = 0i64;
        for group_in_block in 0..16 {
            let group = block_index * 16 + group_in_block;
            let packed_base = group * 8;
            let quant_base = group_in_block * 16;
            let mut dot = 0i32;
            for pair in 0..8 {
                let byte = packed[packed_base + pair];
                dot += e2m1_q1(byte) * i32::from(q8.qs[quant_base + pair * 2]);
                dot += e2m1_q1(byte >> 4) * i32::from(q8.qs[quant_base + pair * 2 + 1]);
            }
            integer_total += i64::from(dot) * i64::from(e4m3fn_q9(scales[group]));
        }
        let mut contribution = (integer_total as f32).mul_add(A16_Q8_INTEGER_SCALE_INV, 0.0);
        contribution = contribution.mul_add(q8.d, 0.0);
        contribution = contribution.mul_add(weight_scale2, 0.0);
        total = 1.0f32.mul_add(contribution, total);
    }
    half::f16::from_f32(total).to_f32()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e2m1_lut_preserves_signed_zero_and_endpoints() {
        assert_eq!(e2m1_to_f32(0).to_bits(), 0.0f32.to_bits());
        assert_eq!(e2m1_to_f32(8).to_bits(), (-0.0f32).to_bits());
        assert_eq!(e2m1_to_f32(7), 6.0);
        assert_eq!(e2m1_to_f32(15), -6.0);
    }

    #[test]
    fn e4m3fn_lut_matches_pinned_reference_cases() {
        assert_eq!(e4m3fn_to_f32(0x00).to_bits(), 0.0f32.to_bits());
        assert_eq!(e4m3fn_to_f32(0x80).to_bits(), (-0.0f32).to_bits());
        assert_eq!(e4m3fn_to_f32(0x01), 1.0 / 512.0);
        assert_eq!(e4m3fn_to_f32(0x38), 1.0);
        assert_eq!(e4m3fn_to_f32(0x7e), 448.0);
        assert!(e4m3fn_to_f32(0x7f).is_nan());
        assert!(e4m3fn_to_f32(0xff).is_nan());
        for code in 0u8..=0x7e {
            assert_eq!(e4m3fn_from_f32(e4m3fn_to_f32(code)), code);
        }
        assert_eq!(e4m3fn_from_f32(1000.0), 0x7e);
        assert_eq!(e4m3fn_from_f32(-1000.0), 0xfe);
    }

    #[test]
    fn integer_encodings_are_exact_for_every_finite_code() {
        for code in 0u8..=u8::MAX {
            if matches!(code, 0x7f | 0xff) {
                continue;
            }
            assert_eq!(
                e4m3fn_q9(code) as f32,
                e4m3fn_to_f32(code) * 512.0,
                "E4M3 Q9 mismatch for code {code:#04x}"
            );
        }
        for code in 0u8..16 {
            assert_eq!(
                e2m1_q1(code) as f32,
                e2m1_to_f32(code) * 2.0,
                "E2M1 Q1 mismatch for code {code:#04x}"
            );
        }
    }

    #[test]
    fn e2m1_ties_follow_the_modelopt_contract() {
        for (value, code) in [
            (0.25, 0),
            (0.250_001, 1),
            (0.75, 2),
            (1.25, 2),
            (1.250_001, 3),
            (1.75, 4),
            (2.5, 4),
            (2.500_001, 5),
            (3.5, 6),
            (5.0, 6),
            (5.000_001, 7),
        ] {
            assert_eq!(e2m1_from_f32(value), code, "value {value}");
            assert_eq!(e2m1_from_f32(-value), code | 8, "value {}", -value);
        }
    }

    #[test]
    fn w4a4_oracle_quantizes_f16_activation_groups_before_the_dot() {
        let input: Vec<f32> = (0..16)
            .map(|index| (index as f32 - 7.5) * 0.125 + 0.000_01)
            .collect();
        let input_scale_inv = 43.75;
        let mut activation_packed = [0u8; 8];
        let mut activation_scales = [0u8; 1];
        quantize_nvfp4_activation(
            &input,
            input_scale_inv,
            &mut activation_packed,
            &mut activation_scales,
        );
        let weights = [0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe];
        let weight_scales = [0x38];
        let actual = dot_nvfp4_w4a4_f32(&weights, &weight_scales, 0.25, input_scale_inv, &input);
        let mut expected_codes = [0.0f32; 16];
        dequant_nvfp4_row(
            &activation_packed,
            &activation_scales,
            1.0 / input_scale_inv,
            &mut expected_codes,
        );
        let expected = half::f16::from_f32(dot_nvfp4_block_fused_f32(
            &weights,
            &weight_scales,
            0.25,
            &expected_codes,
        ))
        .to_f32();
        assert_eq!(actual.to_bits(), expected.to_bits());
    }

    #[test]
    fn a16_q8_oracle_uses_signed_first_max_and_pinned_integer_epilogue() {
        let mut input = vec![0.0f32; 256];
        input[0] = 2.0;
        input[1] = -2.0; // Equal magnitude must not replace the first maximum.
        for (index, value) in input.iter_mut().enumerate().skip(2) {
            *value = half::f16::from_f32((index as f32 - 128.0) * 0.003).to_f32();
        }
        let q8 = quantize_nvfp4_q8_block(&input);
        assert_eq!(q8.d.to_bits(), (-2.0f32 / 127.0).to_bits());
        assert_eq!(q8.qs[0], -127);
        assert_eq!(q8.qs[1], 127);

        let packed = vec![0x62u8; 128];
        let mut scales = vec![0x38u8; 16];
        scales[3] = 0x30;
        scales[11] = 0xb8;
        let actual = dot_nvfp4_a16_q8_f32(&packed, &scales, 0.25, &input);

        let mut integer_total = 0i64;
        for (group, &scale) in scales.iter().enumerate() {
            let mut dot = 0i32;
            for pair in 0..8 {
                dot += e2m1_q1(0x02) * i32::from(q8.qs[group * 16 + pair * 2]);
                dot += e2m1_q1(0x06) * i32::from(q8.qs[group * 16 + pair * 2 + 1]);
            }
            integer_total += i64::from(dot) * i64::from(e4m3fn_q9(scale));
        }
        let mut expected = (integer_total as f32).mul_add(A16_Q8_INTEGER_SCALE_INV, 0.0);
        expected = expected.mul_add(q8.d, 0.0);
        expected = expected.mul_add(0.25, 0.0);
        expected = half::f16::from_f32(expected).to_f32();
        assert_eq!(actual.to_bits(), expected.to_bits());
    }

    #[test]
    fn block_dequant_and_dot_use_the_pinned_order() {
        let packed = [0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe];
        let scales = [0x38];
        let scale2 = 0.25f32;
        let mut got = [0.0f32; 16];
        dequant_nvfp4_row(&packed, &scales, scale2, &mut got);
        let expected: [f32; 16] = [
            0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, -0.0, -0.125, -0.25, -0.375, -0.5, -0.75,
            -1.0, -1.5,
        ];
        for (actual, expected) in got.iter().zip(expected) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
        let x: Vec<f32> = (1..=16).map(|value| value as f32 * 0.125).collect();
        let expected_dot = expected
            .iter()
            .zip(&x)
            .fold(0.0f32, |sum, (&weight, &value)| sum + weight * value);
        assert_eq!(
            dot_nvfp4_f32(&packed, &scales, scale2, &x).to_bits(),
            expected_dot.to_bits()
        );
    }
}
