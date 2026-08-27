//! Internal cache interchange used only by `muser-kvpack`.
//!
//! This is deliberately not part of the public `Model`/`Session` surface.
//! Plane bytes are always in ascending logical-token order; physical Metal
//! ring placement is never inferred from an absolute position.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::config::{layer_kind, MUSE_LAYER_COUNT, MUSE_SWA_WINDOW};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlaneEncoding {
    F16Le,
    F32Le,
}

impl PlaneEncoding {
    pub const fn width_bytes(self) -> usize {
        match self {
            Self::F16Le => 2,
            Self::F32Le => 4,
        }
    }
}

/// One K or V plane pair for one Muse layer in logical-token order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CachePlaneSnapshot {
    pub layer: u32,
    pub logical_start: u64,
    pub logical_count: u64,
    pub encoding: PlaneEncoding,
    pub key: Arc<[u8]>,
    pub value: Arc<[u8]>,
}

/// A complete restorable cut. The 39 SWA layers contain the complete logical
/// tail and the 13 NoPE layers contain `[0, position)`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionCacheSnapshot {
    pub position: u64,
    pub tokens: Arc<[u32]>,
    pub elements_per_token: u32,
    pub layers: Arc<[CachePlaneSnapshot]>,
}

impl SessionCacheSnapshot {
    /// Validate against the Muse SWA window. Interchange consumers that do not
    /// own a session use this.
    pub fn validate(&self) -> Result<(), String> {
        self.validate_for_window(MUSE_SWA_WINDOW)
    }

    /// Validate against the SWA window a session actually allocated its planes
    /// from (`MuseConfig::sliding_window`). A session whose `max_context` is
    /// below the window still holds the complete `[0, position)` tail, so the
    /// expected count is the same `min(position, window)` in both cases — but
    /// it is checked against the window the planes were sized by, never a
    /// constant that a checkpoint may not share.
    pub fn validate_for_window(&self, swa_window: usize) -> Result<(), String> {
        if self.position == 0 {
            return Err("cache snapshot position must be nonzero".into());
        }
        if self.tokens.len() as u64 != self.position {
            return Err(format!(
                "cache snapshot carries {} tokens for position {}",
                self.tokens.len(),
                self.position
            ));
        }
        if self.elements_per_token == 0 {
            return Err("cache snapshot row width must be nonzero".into());
        }
        if self.layers.len() != MUSE_LAYER_COUNT {
            return Err(format!(
                "cache snapshot carries {} layers, expected {MUSE_LAYER_COUNT}",
                self.layers.len()
            ));
        }
        for (expected_layer, plane) in self.layers.iter().enumerate() {
            if plane.layer as usize != expected_layer {
                return Err(format!(
                    "cache snapshot layer order has {} at ordinal {expected_layer}",
                    plane.layer
                ));
            }
            let count = if layer_kind(expected_layer)
                .expect("validated Muse layer")
                .is_swa()
            {
                self.position.min(swa_window as u64)
            } else {
                self.position
            };
            let start = self.position - count;
            if plane.logical_start != start || plane.logical_count != count {
                return Err(format!(
                    "cache snapshot layer {expected_layer} has range {}..{}, expected {start}..{}",
                    plane.logical_start,
                    plane.logical_start.saturating_add(plane.logical_count),
                    self.position
                ));
            }
            let expected_bytes = usize::try_from(count)
                .ok()
                .and_then(|rows| rows.checked_mul(self.elements_per_token as usize))
                .and_then(|elements| elements.checked_mul(plane.encoding.width_bytes()))
                .ok_or_else(|| format!("cache snapshot layer {expected_layer} size overflow"))?;
            if plane.key.len() != expected_bytes || plane.value.len() != expected_bytes {
                return Err(format!(
                    "cache snapshot layer {expected_layer} has K/V bytes {}/{}, expected {expected_bytes}",
                    plane.key.len(),
                    plane.value.len()
                ));
            }
        }
        Ok(())
    }

    pub fn encoding(&self) -> Option<PlaneEncoding> {
        let first = self.layers.first()?.encoding;
        self.layers
            .iter()
            .all(|plane| plane.encoding == first)
            .then_some(first)
    }
}

pub(crate) fn u16s_to_le_bytes(values: &[u16]) -> Arc<[u8]> {
    debug_assert_eq!(
        u16::from_ne_bytes([0x02, 0x01]),
        0x0102u16,
        "f16_le plane bytes are the in-memory u16 representation"
    );
    // Same assumption as `u16_as_le_bytes_mut`: one bulk copy per plane
    // instead of a per-element rebuild of a multi-megabyte export.
    let bytes =
        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 2) };
    Arc::from(bytes)
}

pub(crate) fn f32s_to_le_bytes(values: &[f32]) -> Arc<[u8]> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    bytes.into()
}

pub(crate) fn le_bytes_to_u16s(bytes: &[u8]) -> Result<Vec<u16>, String> {
    if !bytes.len().is_multiple_of(2) {
        return Err("F16 plane byte length is not divisible by two".into());
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

pub(crate) fn le_bytes_to_f32s(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(4) {
        return Err("F32 plane byte length is not divisible by four".into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_bits(u32::from_le_bytes(chunk.try_into().unwrap())))
        .collect())
}

/// Write one token-major f16_le KV tile into a plane that may be stored
/// token-major (SWA) or head-major (NoPE llama FA). Physical origin is 0.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_f16_tile(
    dest: &mut [u16],
    capacity: usize,
    kv_dim: usize,
    head_dim: usize,
    head_major: bool,
    physical_start: usize,
    count: usize,
    bytes: &[u8],
) -> Result<(), String> {
    let expected = count
        .checked_mul(kv_dim)
        .and_then(|elements| elements.checked_mul(2))
        .ok_or_else(|| "KV tile size overflow".to_string())?;
    if dest.len() != capacity.saturating_mul(kv_dim)
        || kv_dim == 0
        || head_dim == 0
        || !kv_dim.is_multiple_of(head_dim)
        || bytes.len() != expected
        || physical_start
            .checked_add(count)
            .is_none_or(|end| end > capacity)
    {
        return Err(format!(
            "KV tile geometry mismatch: dest={}, capacity={capacity}, kv_dim={kv_dim}, start={physical_start}, count={count}, bytes={}",
            dest.len(),
            bytes.len()
        ));
    }
    let dest_bytes = u16_as_le_bytes_mut(dest);
    if !head_major {
        let offset = physical_start
            .checked_mul(kv_dim)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| "KV tile offset overflow".to_string())?;
        dest_bytes[offset..offset + bytes.len()].copy_from_slice(bytes);
        return Ok(());
    }
    let heads = kv_dim / head_dim;
    let row_bytes = head_dim * 2;
    for head in 0..heads {
        for token in 0..count {
            let source = (token * kv_dim + head * head_dim) * 2;
            let destination = (head * capacity + physical_start + token) * row_bytes;
            dest_bytes[destination..destination + row_bytes]
                .copy_from_slice(&bytes[source..source + row_bytes]);
        }
    }
    Ok(())
}

fn u16_as_le_bytes_mut(values: &mut [u16]) -> &mut [u8] {
    // Apple Silicon and the GX10 producer are little-endian; f16_le wire
    // bytes are the in-memory u16 representation.
    unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, values.len() * 2) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_major_tile_is_a_contiguous_memcpy() {
        let mut dest = vec![0u16; 8 * 4];
        let bytes: Vec<u8> = (10u16..18).flat_map(|value| value.to_le_bytes()).collect();
        write_f16_tile(&mut dest, 8, 4, 2, false, 2, 2, &bytes).unwrap();
        assert_eq!(&dest[8..16], &[10, 11, 12, 13, 14, 15, 16, 17]);
        assert!(dest[..8].iter().all(|&value| value == 0));
        assert!(dest[16..].iter().all(|&value| value == 0));
    }

    #[test]
    fn u16_planes_serialize_as_little_endian_wire_bytes() {
        let values = [0x0102u16, 0xF0FF, 0x0000];
        assert_eq!(
            &*u16s_to_le_bytes(&values),
            &[0x02, 0x01, 0xFF, 0xF0, 0x00, 0x00]
        );
        assert!(u16s_to_le_bytes(&[]).is_empty());
    }

    #[test]
    fn head_major_tile_scatters_each_head_row() {
        // 2 KV heads, head_dim 2, capacity 4, one 2-token tile at physical 1.
        // Token-major bytes: t0=[h0=1,2 | h1=3,4], t1=[h0=5,6 | h1=7,8]
        let mut dest = vec![0u16; 4 * 4];
        let bytes: Vec<u8> = [1u16, 2, 3, 4, 5, 6, 7, 8]
            .into_iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        write_f16_tile(&mut dest, 4, 4, 2, true, 1, 2, &bytes).unwrap();
        // head 0 occupies physical rows [0..4) at dest[0..8)
        assert_eq!(&dest[0..8], &[0, 0, 1, 2, 5, 6, 0, 0]);
        // head 1 occupies physical rows [0..4) at dest[8..16)
        assert_eq!(&dest[8..16], &[0, 0, 3, 4, 7, 8, 0, 0]);
    }
}
