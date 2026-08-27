use super::blocks::{dequant_q4_0, dequant_q5_0, dequant_q5_1, dequant_q8_0};
use super::f16_to_f32;
use super::k_block::{dequant_q4_k, dequant_q5_k, dequant_q6_k};
use crate::gguf::GgmlType;

/// Dequantize one block of a dtype admitted by the standalone Muse loader.
pub fn dequant_block(dtype: GgmlType, block: &[u8], out: &mut [f32]) -> usize {
    match dtype {
        GgmlType::F32 => {
            out[0] = f32::from_le_bytes(block[..4].try_into().expect("F32 block"));
            1
        }
        GgmlType::F16 => {
            out[0] = f16_to_f32(u16::from_le_bytes(
                block[..2].try_into().expect("F16 block"),
            ));
            1
        }
        GgmlType::BF16 => {
            let bits = u16::from_le_bytes(block[..2].try_into().expect("BF16 block"));
            out[0] = f32::from_bits((bits as u32) << 16);
            1
        }
        GgmlType::NVFP4_E2M1 | GgmlType::F8_E4M3FN => {
            panic!("NVFP4 payloads require their bound companion tensors")
        }
        GgmlType::Q4_0 => {
            dequant_q4_0(block, out);
            32
        }
        GgmlType::Q4_1 => {
            let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let m = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
            for i in 0..16 {
                out[i] = d * (block[4 + i] & 0x0f) as f32 + m;
                out[i + 16] = d * (block[4 + i] >> 4) as f32 + m;
            }
            32
        }
        GgmlType::Q5_0 => {
            dequant_q5_0(block, out);
            32
        }
        GgmlType::Q5_1 => {
            dequant_q5_1(block, out);
            32
        }
        GgmlType::Q8_0 => {
            dequant_q8_0(block, out);
            32
        }
        GgmlType::Q4_K => {
            dequant_q4_k(block, out);
            256
        }
        GgmlType::Q5_K => {
            dequant_q5_k(block, out);
            256
        }
        GgmlType::Q6_K => {
            dequant_q6_k(block, out);
            256
        }
        other => panic!("unsupported Muse CPU weight dtype {other:?}"),
    }
}
