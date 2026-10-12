use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use kvpack_handoff::{
    CommittedGeneration, ComponentKindV2, HandoffError, HandoffSinkV2, SegmentRoleV2,
    ValidatedBeginV2, VerifiedSealV2, VerifiedSegmentV2,
};
use muser_engine::dflash::{
    DFlashAssistant, DFlashContextGeometry, DFlashContextSnapshot, PreparedDFlashContext,
};
use muser_engine::{PreparedRemoteKvInstall, RemoteKvInstall, Session};

struct Piece {
    start: u64,
    count: u64,
    bytes: Vec<u8>,
}

/// Witnessed target span segment `(role, group layer, start, count)` for the
/// deferred delta schedule re-check at prepare time.
type DeltaWitness = Vec<(SegmentRoleV2, Option<u32>, u64, u64)>;

/// Component-scoped evidence retained after the transactional sink is
/// consumed by the handoff receiver. Qualification must prove that a
/// combined generation prepared and installed both components; aggregate
/// byte counts cannot establish that on their own.
#[derive(Debug, Clone, Default)]
pub struct ComponentInstallEvidence {
    pub target_segments: u32,
    pub target_bytes: u64,
    pub dflash_segments: u32,
    pub dflash_bytes: u64,
    pub target_prepared: bool,
    pub dflash_prepared: bool,
    pub target_installed: bool,
    pub dflash_installed: bool,
}

pub type SharedComponentInstallEvidence = Arc<Mutex<ComponentInstallEvidence>>;

/// V2 receiver sink that writes authenticated target tiles into a detached
/// Metal generation as they arrive. Live decode state is replaced only in
/// `commit`, after the HMAC seal has been verified.
pub struct MuseCacheShadow<'a> {
    session: &'a mut Session,
    dflash: Option<&'a mut DFlashAssistant>,
    generation: u64,
    transfer_id: String,
    tokens: Arc<[u32]>,
    kv_install: Option<RemoteKvInstall>,
    installed_bytes: u64,
    target_component_ids: std::collections::BTreeSet<String>,
    dflash_component_ids: std::collections::BTreeSet<String>,
    dflash_pieces: BTreeMap<(u32, bool), Vec<Piece>>,
    dflash_buffered_bytes: u64,
    dflash_buffered_byte_limit: u64,
    dflash_geometry: Option<DFlashContextGeometry>,
    dflash_width: Option<u32>,
    prepared_target: Option<PreparedRemoteKvInstall>,
    prepared_dflash: Option<PreparedDFlashContext>,
    received_segments: u32,
    target_segments: u32,
    target_bytes: u64,
    dflash_segments: u32,
    dflash_bytes: u64,
    evidence: Option<SharedComponentInstallEvidence>,
    phase: Option<crate::phase::SharedHandoffPhase>,
    /// Delta handoff cut wired in by the receiver: the engine copies the held
    /// prefix `[0, cut)` into the detached generation and only suffix tiles
    /// land. Zero is today's full transfer.
    prefix_cut: u64,
    /// Deferred streams declare no schedule at begin, so a delta sink records
    /// the target stream's `(role, layer, start, count)` and re-checks it
    /// against the span schedule at prepare time.
    delta_witness: Option<DeltaWitness>,
}

impl<'a> MuseCacheShadow<'a> {
    pub fn new(session: &'a mut Session) -> Self {
        Self {
            session,
            dflash: None,
            generation: 0,
            transfer_id: String::new(),
            tokens: Arc::from([]),
            kv_install: None,
            installed_bytes: 0,
            target_component_ids: Default::default(),
            dflash_component_ids: Default::default(),
            dflash_pieces: BTreeMap::new(),
            dflash_buffered_bytes: 0,
            dflash_buffered_byte_limit: 0,
            dflash_geometry: None,
            dflash_width: None,
            prepared_target: None,
            prepared_dflash: None,
            received_segments: 0,
            target_segments: 0,
            target_bytes: 0,
            dflash_segments: 0,
            dflash_bytes: 0,
            evidence: None,
            phase: None,
            prefix_cut: 0,
            delta_witness: None,
        }
    }

    pub fn new_combined(
        session: &'a mut Session,
        dflash: &'a mut DFlashAssistant,
        enrolled_geometry: DFlashContextGeometry,
    ) -> Result<Self, String> {
        enrolled_geometry.validate()?;
        validate_geometry_identity(enrolled_geometry, dflash.context_geometry())?;
        let mut value = Self::new(session);
        value.dflash = Some(dflash);
        value.dflash_buffered_byte_limit = enrolled_geometry.buffered_byte_limit()?;
        value.dflash_geometry = Some(enrolled_geometry);
        Ok(value)
    }

    /// Arm the sink for a delta handoff: at begin the engine copies the held
    /// prefix `[0, cut)` into the detached generation and only suffix tiles
    /// land afterwards.
    pub fn with_prefix_cut(mut self, prefix_cut: u64) -> Self {
        self.prefix_cut = prefix_cut;
        self
    }

    pub fn with_evidence(mut self, evidence: SharedComponentInstallEvidence) -> Self {
        self.evidence = Some(evidence);
        self
    }

    /// N-series phase accounting: the sink's own install/commit time is
    /// reported separately from socket drain and wire verification.
    pub fn with_phase_timing(mut self, phase: crate::phase::SharedHandoffPhase) -> Self {
        self.phase = Some(phase);
        self
    }

    fn record_phase(&self, install_ns: u64) {
        if let Some(phase) = &self.phase {
            if let Ok(mut phase) = phase.lock() {
                phase.sink_install_ns = phase.sink_install_ns.saturating_add(install_ns);
                phase.pending_install_ns = install_ns;
            }
        }
    }

    fn record_commit_phase(&self, commit_ns: u64) {
        if let Some(phase) = &self.phase {
            if let Ok(mut phase) = phase.lock() {
                phase.commit_ns = phase.commit_ns.saturating_add(commit_ns);
            }
        }
    }

    fn publish_evidence(&self, prepared: bool, installed: bool) -> kvpack_handoff::Result<()> {
        let Some(evidence) = &self.evidence else {
            return Ok(());
        };
        let mut evidence = evidence
            .lock()
            .map_err(|_| validation("component install evidence lock was poisoned"))?;
        evidence.target_segments = self.target_segments;
        evidence.target_bytes = self.target_bytes;
        evidence.dflash_segments = self.dflash_segments;
        evidence.dflash_bytes = self.dflash_bytes;
        evidence.target_prepared = prepared;
        evidence.dflash_prepared = prepared && !self.dflash_component_ids.is_empty();
        evidence.target_installed = installed;
        evidence.dflash_installed = installed && !self.dflash_component_ids.is_empty();
        Ok(())
    }
}

impl HandoffSinkV2 for MuseCacheShadow<'_> {
    fn begin(&mut self, begin: &ValidatedBeginV2) -> kvpack_handoff::Result<()> {
        self.abort();
        for component in &begin.manifest().components {
            match component.kind {
                ComponentKindV2::TargetKv => {
                    self.target_component_ids.insert(component.id.clone());
                }
                ComponentKindV2::DflashContext => {
                    if component.required && self.dflash.is_none() {
                        return Err(validation(
                            "required DFlash component needs the combined Muse sink",
                        ));
                    }
                    self.dflash_component_ids.insert(component.id.clone());
                }
                ComponentKindV2::VisionContext if component.required => {
                    return Err(validation(
                        "required vision context cannot be installed by this sink",
                    ));
                }
                ComponentKindV2::VisionContext => {}
            }
        }
        self.generation = begin.manifest().generation;
        self.transfer_id = begin.manifest().transfer_id.clone();
        self.tokens = begin.manifest().prompt_token_ids.clone().into();
        self.kv_install = Some(
            if self.prefix_cut == 0 {
                self.session
                    .begin_remote_kv_install(Arc::clone(&self.tokens))
            } else {
                self.session
                    .begin_remote_kv_install_delta(Arc::clone(&self.tokens), self.prefix_cut)
            }
            .map_err(|error| validation(&error.to_string()))?,
        );
        // A delta over a deferred stream declares no schedule at begin;
        // witness the target stream and re-check it against the span
        // schedule at prepare. Declared-mode deltas were checked at
        // admission already.
        self.delta_witness =
            (self.prefix_cut != 0 && begin.manifest().deferred_segments).then(Vec::new);
        Ok(())
    }
    fn segment_ready(&mut self, segment: VerifiedSegmentV2) -> kvpack_handoff::Result<()> {
        let started = std::time::Instant::now();
        let result = self.segment_ready_timed(segment);
        self.record_phase(crate::phase::nanos(started.elapsed()));
        result
    }
    fn prepare_commit(&mut self, _: &VerifiedSealV2) -> kvpack_handoff::Result<()> {
        let position = self.tokens.len() as u64;
        if position == 0 || self.kv_install.is_none() {
            return Err(validation("empty Muse generation"));
        }
        if let Some(witness) = self.delta_witness.take() {
            let expected = crate::schedule::muse_schedule_span(position, self.prefix_cut, None)
                .ok_or_else(|| validation("delta prefix cut cannot form a span schedule"))?;
            let matches = expected.segments.len() == witness.len()
                && expected.segments.iter().zip(&witness).all(
                    |(intent, &(role, layer, start, count))| {
                        intent.role == role
                            && intent.layer == layer
                            && intent.logical_start == start
                            && intent.logical_count == count
                    },
                );
            if !matches {
                return Err(validation(
                    "deferred target stream is not the span schedule for the prefix cut",
                ));
            }
        }
        self.kv_install
            .as_ref()
            .ok_or_else(|| validation("empty Muse generation"))?
            .validate_complete()
            .map_err(|error| validation(&error.to_string()))?;
        if !self.dflash_component_ids.is_empty() && self.dflash_pieces.is_empty() {
            return Err(validation("required DFlash component contains no planes"));
        }
        if !self.dflash_pieces.is_empty() {
            let width = self
                .dflash_width
                .ok_or_else(|| validation("DFlash context width is absent"))?;
            let geometry = self
                .dflash_geometry
                .ok_or_else(|| validation("DFlash context geometry is absent"))?;
            if width as usize != geometry.elements_per_token {
                return Err(validation(&format!(
                    "DFlash context geometry mismatch: elements_per_token expected {}, got {width}",
                    geometry.elements_per_token
                )));
            }
            let mut layers = Vec::with_capacity(geometry.layers);
            for layer in 0..u32::try_from(geometry.layers)
                .map_err(|_| validation("DFlash context geometry layers exceeds u32"))?
            {
                let key = join_dflash(
                    self.dflash_pieces
                        .remove(&(layer, true))
                        .ok_or_else(|| validation("missing DFlash K plane"))?,
                    position,
                    geometry.sink_size as u64,
                    geometry.window_size as u64,
                )?;
                let value = join_dflash(
                    self.dflash_pieces
                        .remove(&(layer, false))
                        .ok_or_else(|| validation("missing DFlash V plane"))?,
                    position,
                    geometry.sink_size as u64,
                    geometry.window_size as u64,
                )?;
                layers.push((f32_from_le(&key)?, f32_from_le(&value)?));
            }
            if !self.dflash_pieces.is_empty() {
                return Err(validation("extra DFlash context planes"));
            }
            let snapshot = DFlashContextSnapshot {
                position: position as usize,
                sink_size: geometry.sink_size,
                window_size: geometry.window_size,
                elements_per_token: width as usize,
                layers,
            };
            snapshot
                .validate()
                .map_err(|message| validation(&message))?;
            self.dflash
                .as_deref()
                .ok_or_else(|| validation("DFlash sink disappeared during prepare"))?
                .validate_context_snapshot(&snapshot)
                .map_err(|error| validation(&error.to_string()))?;
            self.prepared_dflash = Some(
                self.dflash
                    .as_deref()
                    .ok_or_else(|| validation("DFlash sink disappeared during prepare"))?
                    .prepare_context_snapshot(&snapshot)
                    .map_err(|error| validation(&error.to_string()))?,
            );
        }
        let target = self
            .kv_install
            .take()
            .ok_or_else(|| validation("Muse shadow was not prepared"))?;
        self.prepared_target = Some(
            self.session
                .prepare_remote_kv_install(target)
                .map_err(|error| validation(&error.to_string()))?,
        );
        self.publish_evidence(true, false)?;
        Ok(())
    }
    fn commit(&mut self) -> kvpack_handoff::Result<CommittedGeneration> {
        let started = std::time::Instant::now();
        let result = self.commit_inner();
        self.record_commit_phase(crate::phase::nanos(started.elapsed()));
        result
    }

    #[inline]
    fn abort(&mut self) {
        self.kv_install = None;
        self.prepared_target = None;
        self.prepared_dflash = None;
        self.installed_bytes = 0;
        self.target_component_ids.clear();
        self.dflash_component_ids.clear();
        self.dflash_pieces.clear();
        self.dflash_buffered_bytes = 0;
        self.dflash_width = None;
        self.delta_witness = None;
        self.received_segments = 0;
        self.target_segments = 0;
        self.target_bytes = 0;
        self.dflash_segments = 0;
        self.dflash_bytes = 0;
        if let Some(evidence) = &self.evidence {
            if let Ok(mut evidence) = evidence.lock() {
                *evidence = ComponentInstallEvidence::default();
            }
        }
    }
}

fn join_dflash(
    mut pieces: Vec<Piece>,
    position: u64,
    sink: u64,
    window: u64,
) -> kvpack_handoff::Result<Vec<u8>> {
    pieces.sort_by_key(|piece| piece.start);
    let expected = if position <= sink + window {
        vec![(0, position)]
    } else {
        vec![(0, sink), (position - window, window)]
    };
    if pieces.len() != expected.len() {
        return Err(validation(&format!(
            "DFlash context geometry mismatch: sink_size={sink}, window_size={window} require {} segments, got {}",
            expected.len(),
            pieces.len()
        )));
    }
    let mut bytes = Vec::new();
    for (piece, (start, count)) in pieces.into_iter().zip(expected) {
        if piece.start != start || piece.count != count {
            return Err(validation(&format!(
                "DFlash context geometry mismatch: sink_size={sink}, window_size={window} require range {start}+{count}, got {}+{}",
                piece.start, piece.count
            )));
        }
        bytes.extend_from_slice(&piece.bytes);
    }
    Ok(bytes)
}

fn f32_from_le(bytes: &[u8]) -> kvpack_handoff::Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(validation("DFlash f32 plane is not word aligned"));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_bits(u32::from_le_bytes(chunk.try_into().unwrap())))
        .collect())
}

const NOPE_LAYERS: [u32; 13] = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51];
const PACKED_PLANES: u32 = 26;
const SWA_GROUP: usize = 13;

fn swa_group(first: u32) -> kvpack_handoff::Result<[u32; SWA_GROUP]> {
    let layers: Vec<u32> = (0..52u32)
        .filter(|layer| !NOPE_LAYERS.contains(layer))
        .collect();
    let index = layers
        .iter()
        .position(|&layer| layer == first)
        .ok_or_else(|| validation("SWA tile group is not a Muse SWA layer"))?;
    if index % SWA_GROUP != 0 {
        return Err(validation("SWA tile group is not aligned"));
    }
    let slice = layers
        .get(index..index + SWA_GROUP)
        .ok_or_else(|| validation("SWA tile group is short"))?;
    let mut group = [0u32; SWA_GROUP];
    group.copy_from_slice(slice);
    Ok(group)
}

fn install_packed_tile(
    install: &mut RemoteKvInstall,
    layers: &[u32],
    start: u64,
    count: u64,
    elements_per_token: u32,
    payload: &[u8],
) -> kvpack_handoff::Result<()> {
    if elements_per_token != PACKED_PLANES * 256 {
        return Err(validation("packed tile row width is not 26 Muse planes"));
    }
    if layers.len() != 13 {
        return Err(validation("packed tile does not contain 13 layers"));
    }
    let plane_bytes = (count as usize)
        .checked_mul(256)
        .and_then(|elements| elements.checked_mul(2))
        .ok_or_else(|| validation("packed tile plane size overflow"))?;
    let expected = plane_bytes
        .checked_mul(layers.len() * 2)
        .ok_or_else(|| validation("packed tile payload size overflow"))?;
    if payload.len() != expected {
        return Err(validation(
            "packed tile payload disagrees with row geometry",
        ));
    }
    let mut offset = 0;
    for &layer in layers {
        for is_key in [true, false] {
            install
                .write_f16_tile(
                    layer as usize,
                    is_key,
                    start,
                    count,
                    &payload[offset..offset + plane_bytes],
                )
                .map_err(|error| validation(&error.to_string()))?;
            offset += plane_bytes;
        }
    }
    Ok(())
}

fn validation(message: &str) -> HandoffError {
    HandoffError::Validation(message.into())
}

fn validate_geometry_identity(
    expected: DFlashContextGeometry,
    actual: DFlashContextGeometry,
) -> Result<(), String> {
    for (name, expected, actual) in [
        ("layers", expected.layers, actual.layers),
        (
            "elements_per_token",
            expected.elements_per_token,
            actual.elements_per_token,
        ),
        ("sink_size", expected.sink_size, actual.sink_size),
        ("window_size", expected.window_size, actual.window_size),
    ] {
        if actual != expected {
            return Err(format!(
                "DFlash enrolled geometry mismatch: {name} expected {expected}, got {actual}"
            ));
        }
    }
    Ok(())
}

impl MuseCacheShadow<'_> {
    fn segment_ready_timed(&mut self, segment: VerifiedSegmentV2) -> kvpack_handoff::Result<()> {
        let (descriptor, payload) = segment.into_payload();
        let d = &descriptor;
        if self.dflash_component_ids.contains(&d.component_id) {
            if self.dflash.is_none() {
                return Ok(());
            }
            let is_key = match d.role {
                SegmentRoleV2::DflashKey => true,
                SegmentRoleV2::DflashValue => false,
                _ => return Err(validation("DFlash component contains invalid role")),
            };
            let layer = d
                .layer
                .ok_or_else(|| validation("DFlash segment has no layer"))?;
            let geometry = self
                .dflash_geometry
                .ok_or_else(|| validation("DFlash context geometry is absent"))?;
            // DFlash context is buffered in host memory until prepare_commit,
            // so its shape is bounded here rather than at join time: the
            // no plane can ever be longer than the explicitly enrolled sink
            // plus window.
            if layer as usize >= geometry.layers {
                return Err(validation(&format!(
                    "DFlash context geometry mismatch: layers expected {}, got segment layer {layer}",
                    geometry.layers
                )));
            }
            if d.logical_count > (geometry.sink_size + geometry.window_size) as u64 {
                return Err(validation(&format!(
                    "DFlash context geometry mismatch: logical_count exceeds sink_size {} + window_size {}",
                    geometry.sink_size, geometry.window_size
                )));
            }
            if d.element_type != "f32_le" {
                return Err(validation("DFlash context must use f32_le"));
            }
            self.dflash_buffered_bytes = self
                .dflash_buffered_bytes
                .saturating_add(payload.len() as u64);
            if self.dflash_buffered_bytes > self.dflash_buffered_byte_limit {
                return Err(validation(&format!(
                    "buffered DFlash context exceeds identity-derived bound {}",
                    self.dflash_buffered_byte_limit
                )));
            }
            if d.elements_per_token as usize != geometry.elements_per_token {
                return Err(validation(&format!(
                    "DFlash context geometry mismatch: elements_per_token expected {}, got {}",
                    geometry.elements_per_token, d.elements_per_token
                )));
            }
            self.dflash_width = Some(d.elements_per_token);
            self.dflash_pieces
                .entry((layer, is_key))
                .or_default()
                .push(Piece {
                    start: d.logical_start,
                    count: d.logical_count,
                    bytes: payload,
                });
            self.received_segments += 1;
            self.installed_bytes += d.byte_len;
            self.dflash_segments += 1;
            self.dflash_bytes += d.byte_len;
            return Ok(());
        }
        if !self.target_component_ids.contains(&d.component_id) {
            return Ok(());
        }
        if d.element_type != "f16_le" {
            return Err(validation("target KV must use f16_le"));
        }
        let install = self
            .kv_install
            .as_mut()
            .ok_or_else(|| validation("target tile arrived before begin"))?;
        match d.role {
            SegmentRoleV2::NopeTile => install_packed_tile(
                install,
                &NOPE_LAYERS,
                d.logical_start,
                d.logical_count,
                d.elements_per_token,
                &payload,
            )?,
            SegmentRoleV2::SwaTile => {
                let first = d
                    .layer
                    .ok_or_else(|| validation("SWA tile is missing a group layer"))?;
                let group = swa_group(first)?;
                install_packed_tile(
                    install,
                    &group,
                    d.logical_start,
                    d.logical_count,
                    d.elements_per_token,
                    &payload,
                )?;
            }
            SegmentRoleV2::NopeKey | SegmentRoleV2::SwaKey => {
                let layer = d
                    .layer
                    .ok_or_else(|| validation("target KV segment has no layer"))?;
                install
                    .write_f16_tile(
                        layer as usize,
                        true,
                        d.logical_start,
                        d.logical_count,
                        &payload,
                    )
                    .map_err(|error| validation(&error.to_string()))?;
            }
            SegmentRoleV2::NopeValue | SegmentRoleV2::SwaValue => {
                let layer = d
                    .layer
                    .ok_or_else(|| validation("target KV segment has no layer"))?;
                install
                    .write_f16_tile(
                        layer as usize,
                        false,
                        d.logical_start,
                        d.logical_count,
                        &payload,
                    )
                    .map_err(|error| validation(&error.to_string()))?;
            }
            _ => return Err(validation("target component contains non-KV role")),
        }
        if let Some(witness) = &mut self.delta_witness {
            witness.push((d.role, d.layer, d.logical_start, d.logical_count));
        }
        self.received_segments += 1;
        self.installed_bytes += d.byte_len;
        self.target_segments += 1;
        self.target_bytes += d.byte_len;
        Ok(())
    }

    fn commit_inner(&mut self) -> kvpack_handoff::Result<CommittedGeneration> {
        let target = self
            .prepared_target
            .take()
            .ok_or_else(|| validation("Muse shadow was not prepared"))?;
        // Publish evidence before the infallible engine swaps. A poisoned
        // observer may refuse the transaction, but can never turn a completed
        // target+DFlash swap into a fallible post-commit result.
        self.publish_evidence(true, true)?;
        if let Some(dflash) = self.prepared_dflash.take() {
            self.dflash
                .as_deref_mut()
                .expect("prepared DFlash handle retains its assistant")
                .commit_prepared_context(dflash);
        }
        self.session.commit_prepared_remote_kv_install(target);
        Ok(CommittedGeneration {
            transfer_id: self.transfer_id.clone(),
            generation: self.generation,
            installed_segments: self.received_segments,
            installed_bytes: self.installed_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry(window_size: usize) -> DFlashContextGeometry {
        DFlashContextGeometry {
            layers: 5,
            elements_per_token: 8 * 128,
            sink_size: 64,
            window_size,
        }
    }

    fn piece(start: u64, count: u64) -> Piece {
        Piece {
            start,
            count,
            bytes: vec![0; count as usize],
        }
    }

    #[test]
    fn enrolled_geometry_comparison_names_the_mismatched_field() {
        validate_geometry_identity(geometry(2_048), geometry(2_048)).unwrap();
        let error = validate_geometry_identity(geometry(2_048), geometry(1_024)).unwrap_err();
        assert!(error.contains("window_size"), "{error}");
        assert!(error.contains("expected 2048, got 1024"), "{error}");
    }

    #[test]
    fn segment_join_uses_the_declared_window_in_both_directions() {
        join_dflash(vec![piece(0, 64), piece(2_048, 2_048)], 4_096, 64, 2_048).unwrap();
        let error = join_dflash(vec![piece(0, 64), piece(3_072, 1_024)], 4_096, 64, 2_048)
            .unwrap_err()
            .to_string();
        assert!(error.contains("window_size=2048"), "{error}");

        join_dflash(vec![piece(0, 64), piece(3_072, 1_024)], 4_096, 64, 1_024).unwrap();
        let error = join_dflash(vec![piece(0, 64), piece(2_048, 2_048)], 4_096, 64, 1_024)
            .unwrap_err()
            .to_string();
        assert!(error.contains("window_size=1024"), "{error}");
    }

    #[test]
    fn released_geometry_keeps_the_exact_buffer_bound() {
        assert_eq!(geometry(2_048).buffered_byte_limit().unwrap(), 86_507_520);
    }
}
