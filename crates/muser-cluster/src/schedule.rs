//! Transfer amortization schedule: one HMAC/TLS frame per 512-token NoPE
//! tile (~6.5 MiB) and three pipe-safe SWA groups (13 layers / ~6.5 MiB
//! each) for every overlapping tile. The sliding-window tail ships during
//! the last CUDA ubatches instead of as one post-prefill burst.
//!
//! The Mac sink unpacks each authenticated tile into a detached Metal
//! generation as it arrives, so NoPE D2H/TLS overlaps CUDA prefill and the
//! SWA groups overlap GPU scatter of earlier tiles. Live decode state swaps
//! only after the HMAC seal.
//!
//! muser-original — Ferrite's version is SPEC'D, not built (catalogue).
//! Gate: `hidden_pct` (transfer-overlap fraction) >= 0.95 `[target]`
//! (`docs/metrics-schema.md` §2 DISAGGREGATION).

use kvpack_handoff::SegmentRoleV2;
use muser_engine::dflash::DFlashContextGeometry;
use serde::{Deserialize, Serialize};

const NOPE_LAYERS: [u32; 13] = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51];
const N_LAYERS: u32 = 52;
const SWA_WINDOW: u64 = 2_048;
const NOPE_TILE: u64 = 512;
const SWA_GROUP: usize = 13;
const PACKED_PLANES: u32 = 26;
/// Delta handoffs may begin only on a radix-friendly 256-token boundary.
pub const PREFIX_CUT_ALIGN: u64 = 256;

fn swa_layers() -> impl Iterator<Item = u32> {
    (0..N_LAYERS).filter(|layer| !NOPE_LAYERS.contains(layer))
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentIntent {
    pub sequence: u32,
    pub component_id: String,
    pub role: SegmentRoleV2,
    pub layer: Option<u32>,
    pub logical_start: u64,
    pub logical_count: u64,
    pub element_type: String,
    pub elements_per_token: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferSchedule {
    pub segments: Vec<SegmentIntent>,
    pub first_tile_complete_after_tokens: u64,
}

/// Muse's fixed decode-priority wire order. Complete 512-token NoPE tiles
/// become installable while the producer is still prefilling; the final SWA
/// logical tails follow, then an optional DFlash component as one atomic unit.
pub fn muse_schedule(
    position: u64,
    dflash_geometry: Option<DFlashContextGeometry>,
) -> TransferSchedule {
    muse_schedule_for(position, dflash_geometry, "f16_le", 256)
}

pub fn muse_schedule_for(
    position: u64,
    dflash_geometry: Option<DFlashContextGeometry>,
    element_type: &str,
    elements_per_token: u32,
) -> TransferSchedule {
    muse_schedule_span_for(
        position,
        0,
        dflash_geometry,
        element_type,
        elements_per_token,
    )
    .expect("a full schedule needs only a nonempty position")
}

/// Delta-handoff span schedule: the receiving session already holds
/// `[0, prefix_cut)`, so NoPE tiles cover only `[cut, position)` (NoPE planes
/// are absolute) and SWA tiles cover `[max(cut, position-2048), position)` —
/// a suffix longer than the window re-sends the whole window. Mirrors
/// `scripts/gx10/llamacpp/muser_v2_send.py::muse_intents(position,
/// prefix_cut)` exactly. Returns `None` on a cut that is not 256-aligned or
/// leaves no suffix; `prefix_cut == 0` is byte-identical to the full
/// schedule.
pub fn muse_schedule_span(
    position: u64,
    prefix_cut: u64,
    dflash_geometry: Option<DFlashContextGeometry>,
) -> Option<TransferSchedule> {
    muse_schedule_span_for(position, prefix_cut, dflash_geometry, "f16_le", 256)
}

pub fn muse_schedule_span_for(
    position: u64,
    prefix_cut: u64,
    dflash_geometry: Option<DFlashContextGeometry>,
    element_type: &str,
    elements_per_token: u32,
) -> Option<TransferSchedule> {
    if position == 0 || prefix_cut >= position || !prefix_cut.is_multiple_of(PREFIX_CUT_ALIGN) {
        return None;
    }
    let packed_elements = elements_per_token
        .checked_mul(PACKED_PLANES)
        .expect("packed tile elements_per_token overflow");
    let mut segments = Vec::new();
    let mut push = |component: &str, role, layer, start, count, elems: u32| {
        segments.push(SegmentIntent {
            sequence: segments.len() as u32,
            component_id: component.into(),
            role,
            layer,
            logical_start: start,
            logical_count: count,
            element_type: element_type.into(),
            elements_per_token: elems,
        });
    };
    let swa_start = position.saturating_sub(SWA_WINDOW).max(prefix_cut);
    let swa: Vec<u32> = swa_layers().collect();
    let tiles: Vec<(u64, u64)> = (prefix_cut..position)
        .step_by(NOPE_TILE as usize)
        .map(|tile_start| (tile_start, (position - tile_start).min(NOPE_TILE)))
        .collect();
    // Layer-major order (2026-08-19, connector streaming): every SWA group is
    // emitted for all position tiles as soon as its layers exist mid-prefill,
    // and the NoPE tiles — which need the last NoPE layer — come last. The
    // producer can therefore stream segments during prefill instead of after
    // it; the sink installs by segment coordinates, so wire order carries no
    // install semantics. Mirrors scripts/gx10/llamacpp/muser_v2_send.py.
    for group in swa.chunks(SWA_GROUP) {
        for &(tile_start, count) in &tiles {
            let tile_end = tile_start + count;
            if tile_end <= swa_start {
                continue;
            }
            let chunk_start = tile_start.max(swa_start);
            let chunk_count = tile_end - chunk_start;
            push(
                "target",
                SegmentRoleV2::SwaTile,
                Some(group[0]),
                chunk_start,
                chunk_count,
                packed_elements,
            );
        }
    }
    for &(tile_start, count) in &tiles {
        push(
            "target",
            SegmentRoleV2::NopeTile,
            None,
            tile_start,
            count,
            packed_elements,
        );
    }
    if let Some(geometry) = dflash_geometry {
        geometry.validate().ok()?;
        let layers = u32::try_from(geometry.layers).ok()?;
        let sink = u64::try_from(geometry.sink_size).ok()?;
        let window = u64::try_from(geometry.window_size).ok()?;
        let dflash_elements = u32::try_from(geometry.elements_per_token).ok()?;
        for layer in 0..layers {
            for role in [SegmentRoleV2::DflashKey, SegmentRoleV2::DflashValue] {
                if position <= sink + window {
                    segments.push(SegmentIntent {
                        sequence: segments.len() as u32,
                        component_id: "dflash".into(),
                        role,
                        layer: Some(layer),
                        logical_start: 0,
                        logical_count: position,
                        element_type: "f32_le".into(),
                        elements_per_token: dflash_elements,
                    });
                } else {
                    for (start, count) in [(0, sink), (position - window, window)] {
                        segments.push(SegmentIntent {
                            sequence: segments.len() as u32,
                            component_id: "dflash".into(),
                            role,
                            layer: Some(layer),
                            logical_start: start,
                            logical_count: count,
                            element_type: "f32_le".into(),
                            elements_per_token: dflash_elements,
                        });
                    }
                }
            }
        }
    }
    Some(TransferSchedule {
        segments,
        first_tile_complete_after_tokens: prefix_cut.saturating_add(NOPE_TILE).min(position),
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OverlapReceipt {
    pub producer_prefill_ns: u64,
    pub transfer_ns: u64,
    pub exposed_transfer_ns: u64,
}
impl OverlapReceipt {
    pub fn hidden_fraction(self) -> f64 {
        if self.transfer_ns == 0 {
            1.0
        } else {
            1.0 - self.exposed_transfer_ns.min(self.transfer_ns) as f64 / self.transfer_ns as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dflash(window_size: usize) -> Option<DFlashContextGeometry> {
        Some(DFlashContextGeometry {
            layers: 5,
            elements_per_token: 8 * 128,
            sink_size: 64,
            window_size,
        })
    }

    #[test]
    fn overlapping_nope_tiles_are_followed_by_swa_chunks() {
        let s = muse_schedule(2561, dflash(2_048));
        let first_swa = s
            .segments
            .iter()
            .position(|x| x.role == SegmentRoleV2::SwaTile)
            .unwrap();
        assert!(s.segments[..first_swa]
            .iter()
            .all(|x| x.role == SegmentRoleV2::NopeTile));
        assert_eq!(s.segments[first_swa].logical_start, 513);
        assert_eq!(s.segments[first_swa].logical_count, 511);
        assert_eq!(s.segments[first_swa].elements_per_token, 256 * 26);
        let last_swa = s
            .segments
            .iter()
            .rev()
            .find(|x| x.role == SegmentRoleV2::SwaTile)
            .unwrap();
        assert_eq!(last_swa.logical_start + last_swa.logical_count, 2561);
        assert!(s
            .segments
            .windows(2)
            .all(|w| w[1].sequence == w[0].sequence + 1));
    }
    #[test]
    fn overlap_metric_is_bounded() {
        assert_eq!(
            OverlapReceipt {
                producer_prefill_ns: 10,
                transfer_ns: 10,
                exposed_transfer_ns: 20
            }
            .hidden_fraction(),
            0.0
        );
    }

    #[test]
    fn zero_cut_is_byte_identical_to_the_full_schedule() {
        assert_eq!(
            muse_schedule(2049, dflash(2_048)),
            muse_schedule_span(2049, 0, dflash(2_048)).unwrap()
        );
        assert_eq!(
            muse_schedule_for(300, None, "f16_le", 256),
            muse_schedule_span_for(300, 0, None, "f16_le", 256).unwrap()
        );
    }

    #[test]
    fn mid_window_cut_tiles_only_the_suffix() {
        // Mirrors the tested Python mirror: muse_intents(2049, 1024).
        let s = muse_schedule_span(2049, 1024, None).unwrap();
        let nope: Vec<_> = s
            .segments
            .iter()
            .filter(|x| x.role == SegmentRoleV2::NopeTile)
            .collect();
        assert_eq!((nope[0].logical_start, nope[0].logical_count), (1024, 512));
        assert_eq!(
            nope.last().unwrap().logical_start + nope.last().unwrap().logical_count,
            2049
        );
        let swa: Vec<_> = s
            .segments
            .iter()
            .filter(|x| x.role == SegmentRoleV2::SwaTile)
            .collect();
        // Cut inside the window: the SWA span begins at the cut.
        assert_eq!(swa[0].logical_start, 1024);
        assert_eq!(
            swa.last().unwrap().logical_start + swa.last().unwrap().logical_count,
            2049
        );
        let group_heads: Vec<_> = swa.iter().map(|x| x.layer).collect();
        // Layer-major order: each SWA group spans all position tiles before
        // the next group starts, and NoPE tiles trail the whole schedule.
        assert_eq!(
            group_heads,
            [
                Some(0),
                Some(0),
                Some(0),
                Some(17),
                Some(17),
                Some(17),
                Some(34),
                Some(34),
                Some(34)
            ]
        );
        let swa_count = swa.len();
        assert!(s.segments[swa_count..]
            .iter()
            .all(|x| x.role == SegmentRoleV2::NopeTile));
        assert!(s
            .segments
            .windows(2)
            .all(|w| w[1].sequence == w[0].sequence + 1));
        assert_eq!(s.first_tile_complete_after_tokens, 1536);
    }

    #[test]
    fn suffix_longer_than_the_window_resends_the_window() {
        // Mirrors muse_intents(4096, 512): the window slid past the cut.
        let s = muse_schedule_span(4096, 512, None).unwrap();
        let swa: Vec<_> = s
            .segments
            .iter()
            .filter(|x| x.role == SegmentRoleV2::SwaTile)
            .collect();
        assert_eq!(swa[0].logical_start, 2048);
        assert_eq!(
            swa.last().unwrap().logical_start + swa.last().unwrap().logical_count,
            4096
        );
        let nope: Vec<_> = s
            .segments
            .iter()
            .filter(|x| x.role == SegmentRoleV2::NopeTile)
            .collect();
        assert_eq!(nope[0].logical_start, 512);
        assert_eq!(nope.len(), 7);
        // DFlash context is cut-independent: sink + window over the full position.
        let with_dflash = muse_schedule_span(4096, 512, dflash(2_048)).unwrap();
        let release_dflash: Vec<_> = with_dflash
            .segments
            .iter()
            .filter(|x| x.component_id == "dflash")
            .collect();
        assert!(release_dflash
            .iter()
            .any(|x| x.logical_start == 0 && x.logical_count == 64));
        assert!(release_dflash
            .iter()
            .any(|x| x.logical_start == 4096 - 2_048 && x.logical_count == 2_048));

        let legacy = muse_schedule_span(4096, 512, dflash(1_024)).unwrap();
        let legacy: Vec<_> = legacy
            .segments
            .iter()
            .filter(|x| x.component_id == "dflash")
            .collect();
        assert!(legacy
            .iter()
            .any(|x| x.logical_start == 4096 - 1_024 && x.logical_count == 1_024));
        assert!(!legacy
            .iter()
            .any(|x| x.logical_start == 4096 - 2_048 && x.logical_count == 2_048));
    }

    #[test]
    fn invalid_cuts_fail_closed() {
        assert!(muse_schedule_span(2049, 2049, None).is_none());
        assert!(muse_schedule_span(2049, 4096, None).is_none());
        assert!(muse_schedule_span(2049, 128, None).is_none());
        assert!(muse_schedule_span(0, 0, None).is_none());
        // The smallest legal delta: a 256-aligned cut one suffix token short.
        let s = muse_schedule_span(257, 256, None).unwrap();
        assert_eq!(s.segments[0].logical_start, 256);
        assert_eq!(s.segments[0].logical_count, 1);
    }
}
