use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_CONTEXT_CACHE_IDENTITY: AtomicU64 = AtomicU64::new(1);

fn next_context_cache_identity() -> u64 {
    let identity = NEXT_CONTEXT_CACHE_IDENTITY.fetch_add(1, Ordering::Relaxed);
    assert_ne!(identity, 0, "DFlash context-cache identity exhausted");
    identity
}

pub struct DFlashKvCache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    pub len: usize,
    width: usize,
}

impl DFlashKvCache {
    pub fn new(layers: usize, width: usize) -> Self {
        Self {
            k: vec![Vec::new(); layers],
            v: vec![Vec::new(); layers],
            len: 0,
            width,
        }
    }

    pub fn crop(&mut self, n: usize) {
        let keep = n.min(self.len);
        for layer in 0..self.k.len() {
            self.k[layer].truncate(keep * self.width);
            self.v[layer].truncate(keep * self.width);
        }
        self.len = keep;
    }
}

#[derive(Clone, Copy)]
pub(crate) struct TokenLayout(usize);

impl TokenLayout {
    pub fn new(width: usize) -> Self {
        assert!(width > 0);
        Self(width)
    }
    pub fn width(self) -> usize {
        self.0
    }
    pub fn elements(self, rows: usize) -> usize {
        rows * self.0
    }
    pub fn rows(self, start: usize, count: usize) -> Range<usize> {
        self.elements(start)..self.elements(start + count)
    }
}

pub struct DFlashContextKvCache {
    ctx_k: Vec<Vec<f32>>,
    ctx_v: Vec<Vec<f32>>,
    pub ctx_len: usize,
    pub ctx_offset: usize,
    pub sink_size: usize,
    pub window_size: usize,
    layout: TokenLayout,
    identity: u64,
    revision: u64,
    layer_revisions: Vec<u64>,
}

/// Bounded rollback record for one provisional Mirror-SD append. It retains
/// only compact tail rows which the append can evict; the 64-token sink and
/// untouched window structurally remain in the authoritative CPU cache.
pub(crate) struct DFlashContextCheckpoint {
    identity: u64,
    ctx_len: usize,
    ctx_offset: usize,
    revision: u64,
    sink_size: usize,
    window_size: usize,
    layout_width: usize,
    append_tokens: usize,
    removed_rows: usize,
    removed: Vec<(Vec<f32>, Vec<f32>)>,
    layer_revisions: Vec<u64>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DFlashContextSnapshot {
    pub position: usize,
    pub sink_size: usize,
    pub window_size: usize,
    pub elements_per_token: usize,
    pub layers: Vec<(Vec<f32>, Vec<f32>)>,
}

impl DFlashContextSnapshot {
    pub fn validate(&self) -> Result<(), String> {
        if self.position == 0
            || self.elements_per_token == 0
            || self.layers.is_empty()
            || self.sink_size == 0
            || self.window_size == 0
        {
            return Err("invalid DFlash context snapshot geometry".into());
        }
        let rows = self.position.min(self.sink_size + self.window_size);
        let elements = rows
            .checked_mul(self.elements_per_token)
            .ok_or("DFlash context snapshot size overflow")?;
        if self
            .layers
            .iter()
            .any(|(key, value)| key.len() != elements || value.len() != elements)
        {
            return Err("DFlash context snapshot plane length mismatch".into());
        }
        Ok(())
    }
}

impl DFlashContextKvCache {
    pub fn new(layers: usize, width: usize, sink_size: usize, window_size: usize) -> Self {
        Self {
            ctx_k: vec![Vec::new(); layers],
            ctx_v: vec![Vec::new(); layers],
            ctx_len: 0,
            ctx_offset: 0,
            sink_size,
            window_size,
            layout: TokenLayout::new(width),
            identity: next_context_cache_identity(),
            revision: 0,
            layer_revisions: vec![0; layers],
        }
    }

    pub(crate) fn layout(&self) -> TokenLayout {
        self.layout
    }

    /// Monotonic mutation identity used by accelerator mirrors. Position and
    /// length alone cannot distinguish two authenticated snapshots installed
    /// at the same cut.
    pub(crate) fn revision(&self) -> u64 {
        self.revision
    }

    /// Process-local identity for this cache incarnation. A fresh request may
    /// begin at revision zero after a previous request left an accelerator
    /// mirror at revision zero, so revision alone is not a safe mirror key.
    pub(crate) fn identity(&self) -> u64 {
        self.identity
    }

    pub(crate) fn physical_capacity(&self) -> usize {
        self.sink_size + self.window_size
    }

    pub fn geometry(&self) -> super::DFlashContextGeometry {
        super::DFlashContextGeometry {
            layers: self.ctx_k.len(),
            elements_per_token: self.layout.width(),
            sink_size: self.sink_size,
            window_size: self.window_size,
        }
    }

    pub(crate) fn checkpoint_append(&self, tokens: usize) -> DFlashContextCheckpoint {
        let maximum = self.sink_size + self.window_size;
        let removed_rows = (self.ctx_len + tokens).saturating_sub(maximum);
        debug_assert!(tokens <= self.window_size && removed_rows <= self.ctx_len);
        let removed = if removed_rows == 0 {
            vec![(Vec::new(), Vec::new()); self.ctx_k.len()]
        } else {
            let range = self.layout.rows(self.sink_size, removed_rows);
            self.ctx_k
                .iter()
                .zip(&self.ctx_v)
                .map(|(key, value)| (key[range.clone()].to_vec(), value[range.clone()].to_vec()))
                .collect()
        };
        DFlashContextCheckpoint {
            identity: self.identity,
            ctx_len: self.ctx_len,
            ctx_offset: self.ctx_offset,
            revision: self.revision,
            sink_size: self.sink_size,
            window_size: self.window_size,
            layout_width: self.layout.width(),
            append_tokens: tokens,
            removed_rows,
            removed,
            layer_revisions: self.layer_revisions.clone(),
        }
    }

    pub(crate) fn rollback_append(
        &mut self,
        checkpoint: DFlashContextCheckpoint,
    ) -> Result<(), String> {
        self.validate_rollback_target(&checkpoint)?;
        let appended = self.layout.elements(checkpoint.append_tokens);
        let insertion = self.layout.elements(self.sink_size);
        for (layer, ((key, value), (removed_key, removed_value))) in self
            .ctx_k
            .iter_mut()
            .zip(&mut self.ctx_v)
            .zip(checkpoint.removed)
            .enumerate()
        {
            if self.layer_revisions[layer] == checkpoint.layer_revisions[layer] {
                continue;
            }
            key.truncate(key.len() - appended);
            value.truncate(value.len() - appended);
            if checkpoint.removed_rows != 0 {
                key.splice(insertion..insertion, removed_key);
                value.splice(insertion..insertion, removed_value);
            }
        }
        self.ctx_len = checkpoint.ctx_len;
        self.ctx_offset = checkpoint.ctx_offset;
        self.revision = checkpoint.revision;
        self.layer_revisions = checkpoint.layer_revisions;
        if !self.ctx_k.iter().zip(&self.ctx_v).all(|(key, value)| {
            key.len() == self.layout.elements(self.ctx_len)
                && value.len() == self.layout.elements(self.ctx_len)
        }) {
            return Err("DFlash rollback produced inconsistent cache planes".into());
        }
        Ok(())
    }

    pub(crate) fn validate_completed_append(
        &self,
        checkpoint: &DFlashContextCheckpoint,
    ) -> Result<(), String> {
        self.validate_rollback_target(checkpoint)?;
        let after_rows = (checkpoint.ctx_len + checkpoint.append_tokens)
            .min(checkpoint.sink_size + checkpoint.window_size);
        if self.revision != checkpoint.revision.wrapping_add(1)
            || self.ctx_len != after_rows
            || self.ctx_offset
                != checkpoint
                    .ctx_offset
                    .checked_add(checkpoint.append_tokens)
                    .ok_or("DFlash completed-append offset overflow")?
            || self
                .layer_revisions
                .iter()
                .zip(&checkpoint.layer_revisions)
                .any(|(&after, &before)| after != before.wrapping_add(1))
        {
            return Err("DFlash provisional cache is not the complete checkpoint append".into());
        }
        Ok(())
    }

    fn validate_rollback_target(&self, checkpoint: &DFlashContextCheckpoint) -> Result<(), String> {
        if self.identity != checkpoint.identity
            || self.sink_size != checkpoint.sink_size
            || self.window_size != checkpoint.window_size
            || self.layout.width() != checkpoint.layout_width
            || self.ctx_k.len() != checkpoint.layer_revisions.len()
            || self.ctx_v.len() != checkpoint.layer_revisions.len()
            || self.layer_revisions.len() != checkpoint.layer_revisions.len()
            || checkpoint.removed.len() != checkpoint.layer_revisions.len()
        {
            return Err("DFlash rollback checkpoint identity or geometry differs".into());
        }
        if checkpoint.append_tokens == 0 || checkpoint.append_tokens > self.window_size {
            return Err("DFlash rollback checkpoint append count is invalid".into());
        }
        let capacity = self.physical_capacity();
        let appended_rows = checkpoint
            .ctx_len
            .checked_add(checkpoint.append_tokens)
            .ok_or("DFlash rollback checkpoint row count overflow")?;
        let expected_removed = appended_rows.saturating_sub(capacity);
        if checkpoint.ctx_len > capacity
            || checkpoint.removed_rows != expected_removed
            || checkpoint.removed_rows > checkpoint.ctx_len
        {
            return Err("DFlash rollback checkpoint retention geometry differs".into());
        }
        let removed_elements = self.layout.elements(checkpoint.removed_rows);
        if checkpoint
            .removed
            .iter()
            .any(|(key, value)| key.len() != removed_elements || value.len() != removed_elements)
        {
            return Err("DFlash rollback checkpoint tail planes differ".into());
        }

        let before_elements = self.layout.elements(checkpoint.ctx_len);
        let after_rows = appended_rows.min(capacity);
        let after_elements = self.layout.elements(after_rows);
        let next_revision = checkpoint.revision.wrapping_add(1);
        let scalar_state_is_before = self.revision == checkpoint.revision
            && self.ctx_len == checkpoint.ctx_len
            && self.ctx_offset == checkpoint.ctx_offset;
        let scalar_state_is_after = self.revision == next_revision
            && self.ctx_len == after_rows
            && self.ctx_offset
                == checkpoint
                    .ctx_offset
                    .checked_add(checkpoint.append_tokens)
                    .ok_or("DFlash rollback checkpoint offset overflow")?;
        if !scalar_state_is_before && !scalar_state_is_after {
            return Err("DFlash rollback target is not the checkpoint append".into());
        }

        let mut changed_layers = 0usize;
        for (layer, ((key, value), &before_revision)) in self
            .ctx_k
            .iter()
            .zip(&self.ctx_v)
            .zip(&checkpoint.layer_revisions)
            .enumerate()
        {
            let current_revision = self.layer_revisions[layer];
            let expected_elements = if current_revision == before_revision {
                before_elements
            } else if current_revision == before_revision.wrapping_add(1) {
                changed_layers += 1;
                after_elements
            } else {
                return Err(format!(
                    "DFlash rollback layer {layer} revision is not the checkpoint append"
                ));
            };
            if key.len() != expected_elements || value.len() != expected_elements {
                return Err(format!(
                    "DFlash rollback layer {layer} plane length is not the checkpoint append"
                ));
            }
        }
        if scalar_state_is_after && changed_layers != checkpoint.layer_revisions.len() {
            return Err("DFlash rollback advanced before every layer append completed".into());
        }
        Ok(())
    }

    /// Stable physical slot used by the stateful CoreML mirror. Sink rows
    /// never move; the retained tail is a ring. Keys are already RoPE'd, and
    /// DFlash attention is unmasked over retained context, so physical order
    /// does not alter the result.
    pub(crate) fn physical_slot(&self, absolute_position: usize) -> usize {
        if absolute_position < self.sink_size {
            absolute_position
        } else {
            self.sink_size + (absolute_position - self.sink_size) % self.window_size
        }
    }

    /// Absolute token position represented by one compact logical cache row.
    pub(crate) fn retained_absolute_position(&self, logical_row: usize) -> usize {
        assert!(logical_row < self.ctx_len);
        let sink_rows = self.ctx_len.min(self.sink_size);
        if logical_row < sink_rows {
            logical_row
        } else {
            let tail_rows = self.ctx_len - sink_rows;
            self.ctx_offset - tail_rows + (logical_row - sink_rows)
        }
    }

    pub(crate) fn append_layer(&mut self, layer: usize, k: &[f32], v: &[f32], tokens: usize) {
        let needed = self.layout.elements(tokens);
        assert!(k.len() >= needed && v.len() >= needed);
        let maximum = self.sink_size + self.window_size;
        // A first long-prompt batch may contain tens of thousands of rows,
        // while the assistant cache retains only sink+window. Avoid briefly
        // materializing the entire prompt in every one of the ten CPU shadow
        // planes before draining it again.
        if tokens > maximum {
            let retain = |current: &[f32], fresh: &[f32]| {
                let current_rows = current.len() / self.layout.width();
                let sink_from_current = current_rows.min(self.sink_size);
                let sink_from_fresh = self.sink_size - sink_from_current;
                let mut next = Vec::with_capacity(self.layout.elements(maximum));
                next.extend_from_slice(&current[..self.layout.elements(sink_from_current)]);
                next.extend_from_slice(&fresh[..self.layout.elements(sink_from_fresh)]);
                next.extend_from_slice(
                    &fresh[self
                        .layout
                        .rows(tokens - self.window_size, self.window_size)],
                );
                next
            };
            self.ctx_k[layer] = retain(&self.ctx_k[layer], k);
            self.ctx_v[layer] = retain(&self.ctx_v[layer], v);
            self.layer_revisions[layer] = self.layer_revisions[layer].wrapping_add(1);
            return;
        }
        self.ctx_k[layer].extend_from_slice(&k[..needed]);
        self.ctx_v[layer].extend_from_slice(&v[..needed]);
        let total = self.ctx_k[layer].len() / self.layout.width();
        if total > maximum {
            let drop = self.layout.rows(self.sink_size, total - maximum);
            self.ctx_k[layer].drain(drop.clone());
            self.ctx_v[layer].drain(drop);
        }
        self.layer_revisions[layer] = self.layer_revisions[layer].wrapping_add(1);
    }

    pub(crate) fn advance_round(&mut self, tokens: usize) {
        self.ctx_offset += tokens;
        self.ctx_len = self
            .ctx_k
            .first()
            .map_or(0, |k| k.len() / self.layout.width());
        let expected = self.layout.elements(self.ctx_len);
        assert!(self
            .ctx_k
            .iter()
            .zip(&self.ctx_v)
            .all(|(k, v)| k.len() == expected && v.len() == expected));
        self.revision = self.revision.wrapping_add(1);
    }

    /// Advance across prompt rows which cannot survive the configured
    /// sink+window retention geometry. Prompt K/V projection may skip these
    /// rows entirely, but the next retained tail row still needs its absolute
    /// RoPE position. This is valid only after the sink has been populated and
    /// before any retained tail row is appended.
    pub(crate) fn advance_prompt_gap(&mut self, next_offset: usize) -> Result<(), String> {
        let expected_sink = self.sink_size.min(self.ctx_offset);
        if next_offset <= self.ctx_offset
            || self.ctx_len != expected_sink
            || self.ctx_offset != expected_sink
        {
            return Err("DFlash prompt gap is not immediately after the retained sink".into());
        }
        self.ctx_offset = next_offset;
        self.revision = self.revision.wrapping_add(1);
        Ok(())
    }

    pub fn layer_k(&self, layer: usize) -> &[f32] {
        &self.ctx_k[layer]
    }
    pub fn layer_v(&self, layer: usize) -> &[f32] {
        &self.ctx_v[layer]
    }
    pub(crate) fn k_rows(&self, layer: usize, start: usize, count: usize) -> &[f32] {
        &self.ctx_k[layer][self.layout.rows(start, count)]
    }
    pub(crate) fn v_rows(&self, layer: usize, start: usize, count: usize) -> &[f32] {
        &self.ctx_v[layer][self.layout.rows(start, count)]
    }

    pub fn snapshot(&self) -> DFlashContextSnapshot {
        DFlashContextSnapshot {
            position: self.ctx_offset,
            sink_size: self.sink_size,
            window_size: self.window_size,
            elements_per_token: self.layout.width(),
            layers: self
                .ctx_k
                .iter()
                .cloned()
                .zip(self.ctx_v.iter().cloned())
                .collect(),
        }
    }

    pub fn install_snapshot(&mut self, snapshot: &DFlashContextSnapshot) -> Result<(), String> {
        self.validate_snapshot_identity(snapshot)?;
        self.ctx_offset = snapshot.position;
        self.ctx_len = snapshot
            .position
            .min(snapshot.sink_size + snapshot.window_size);
        for (layer, (key, value)) in snapshot.layers.iter().enumerate() {
            self.ctx_k[layer].clone_from(key);
            self.ctx_v[layer].clone_from(value);
            self.layer_revisions[layer] = self.layer_revisions[layer].wrapping_add(1);
        }
        self.revision = self.revision.wrapping_add(1);
        Ok(())
    }

    pub(crate) fn prepare_snapshot(
        &self,
        snapshot: &DFlashContextSnapshot,
    ) -> Result<Self, String> {
        self.validate_snapshot_identity(snapshot)?;
        let mut prepared = Self::new(
            self.ctx_k.len(),
            self.layout.width(),
            self.sink_size,
            self.window_size,
        );
        prepared.install_snapshot(snapshot)?;
        Ok(prepared)
    }

    pub fn validate_snapshot_identity(
        &self,
        snapshot: &DFlashContextSnapshot,
    ) -> Result<(), String> {
        snapshot.validate()?;
        let expected = self.geometry();
        for (name, expected, actual) in [
            ("layers", expected.layers, snapshot.layers.len()),
            (
                "elements_per_token",
                expected.elements_per_token,
                snapshot.elements_per_token,
            ),
            ("sink_size", expected.sink_size, snapshot.sink_size),
            ("window_size", expected.window_size, snapshot.window_size),
        ] {
            if actual != expected {
                return Err(format!(
                    "DFlash context snapshot geometry mismatch: {name} expected {expected}, got {actual}"
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn context_cache_preserves_sink_and_latest_window() {
        let mut cache = DFlashContextKvCache::new(1, 1, 2, 3);
        cache.append_layer(0, &[0., 1., 2., 3.], &[10., 11., 12., 13.], 4);
        cache.advance_round(4);
        cache.append_layer(0, &[4., 5., 6.], &[14., 15., 16.], 3);
        cache.advance_round(3);
        assert_eq!(cache.layer_k(0), &[0., 1., 4., 5., 6.]);
        assert_eq!(cache.ctx_offset, 7);
    }

    fn full_snapshot(window_size: usize) -> DFlashContextSnapshot {
        let sink_size = 64;
        let elements_per_token = 2;
        let position = sink_size + window_size;
        let elements = position * elements_per_token;
        DFlashContextSnapshot {
            position,
            sink_size,
            window_size,
            elements_per_token,
            layers: vec![(vec![0.0; elements], vec![0.0; elements]); 5],
        }
    }

    #[test]
    fn declared_2048_geometry_installs_2048_and_names_1024_mismatch() {
        let mut cache = DFlashContextKvCache::new(5, 2, 64, 2_048);
        cache.install_snapshot(&full_snapshot(2_048)).unwrap();
        let error = cache.install_snapshot(&full_snapshot(1_024)).unwrap_err();
        assert!(error.contains("window_size"), "{error}");
        assert!(error.contains("expected 2048, got 1024"), "{error}");
    }

    #[test]
    fn declared_legacy_1024_geometry_installs_1024_and_names_2048_mismatch() {
        let mut cache = DFlashContextKvCache::new(5, 2, 64, 1_024);
        cache.install_snapshot(&full_snapshot(1_024)).unwrap();
        let error = cache.install_snapshot(&full_snapshot(2_048)).unwrap_err();
        assert!(error.contains("window_size"), "{error}");
        assert!(error.contains("expected 1024, got 2048"), "{error}");
    }

    #[test]
    fn long_batch_never_retains_middle_rows() {
        let mut cache = DFlashContextKvCache::new(1, 1, 2, 3);
        let rows = (0..20).map(|value| value as f32).collect::<Vec<_>>();
        cache.append_layer(0, &rows, &rows, rows.len());
        cache.advance_round(rows.len());
        assert_eq!(cache.layer_k(0), &[0., 1., 17., 18., 19.]);
    }

    #[test]
    fn prompt_gap_preserves_sink_and_positions_the_retained_tail() {
        let mut cache = DFlashContextKvCache::new(1, 1, 2, 3);
        cache.append_layer(0, &[0., 1.], &[10., 11.], 2);
        cache.advance_round(2);
        cache.advance_prompt_gap(17).expect("skip discarded middle");
        cache.append_layer(0, &[17., 18., 19.], &[27., 28., 29.], 3);
        cache.advance_round(3);
        assert_eq!(cache.ctx_offset, 20);
        assert_eq!(cache.layer_k(0), &[0., 1., 17., 18., 19.]);
        assert_eq!(cache.retained_absolute_position(2), 17);
    }

    #[test]
    fn stateful_ring_maps_sink_and_wrapped_tail_without_reordering_cache() {
        let mut cache = DFlashContextKvCache::new(1, 1, 2, 3);
        let rows = (0..9).map(|value| value as f32).collect::<Vec<_>>();
        cache.append_layer(0, &rows, &rows, rows.len());
        cache.advance_round(rows.len());
        assert_eq!(cache.layer_k(0), &[0., 1., 6., 7., 8.]);
        let absolute = (0..cache.ctx_len)
            .map(|row| cache.retained_absolute_position(row))
            .collect::<Vec<_>>();
        assert_eq!(absolute, [0, 1, 6, 7, 8]);
        let physical = absolute
            .iter()
            .map(|&position| cache.physical_slot(position))
            .collect::<Vec<_>>();
        assert_eq!(physical, [0, 1, 3, 4, 2]);
        assert_eq!(cache.physical_capacity(), 5);
    }

    #[test]
    fn fresh_cache_has_a_distinct_accelerator_identity() {
        let first = DFlashContextKvCache::new(1, 1, 2, 3);
        let second = DFlashContextKvCache::new(1, 1, 2, 3);
        assert_ne!(first.identity(), second.identity());
        assert_eq!(first.revision(), second.revision());
    }

    #[test]
    fn mirror_checkpoint_restores_only_evicted_tail_rows() {
        let mut cache = DFlashContextKvCache::new(1, 1, 2, 3);
        cache.append_layer(0, &[0., 1., 2., 3., 4.], &[10., 11., 12., 13., 14.], 5);
        cache.advance_round(5);
        let identity = cache.identity();
        let revision = cache.revision();
        let checkpoint = cache.checkpoint_append(2);
        cache.append_layer(0, &[5., 6.], &[15., 16.], 2);
        cache.advance_round(2);
        assert_eq!(cache.layer_k(0), &[0., 1., 4., 5., 6.]);
        cache
            .validate_completed_append(&checkpoint)
            .expect("complete append stamp");
        cache
            .rollback_append(checkpoint)
            .expect("matching checkpoint rollback");
        assert_eq!(cache.layer_k(0), &[0., 1., 2., 3., 4.]);
        assert_eq!(cache.layer_v(0), &[10., 11., 12., 13., 14.]);
        assert_eq!(cache.ctx_offset, 5);
        assert_eq!(cache.identity(), identity);
        assert_eq!(cache.revision(), revision);
    }

    #[test]
    fn mirror_checkpoint_restores_a_partially_failed_layer_pass() {
        let mut cache = DFlashContextKvCache::new(2, 1, 2, 3);
        for layer in 0..2 {
            cache.append_layer(layer, &[0., 1., 2., 3., 4.], &[10., 11., 12., 13., 14.], 5);
        }
        cache.advance_round(5);
        let checkpoint = cache.checkpoint_append(2);
        cache.append_layer(0, &[5., 6.], &[15., 16.], 2);
        assert!(cache.validate_completed_append(&checkpoint).is_err());
        cache
            .rollback_append(checkpoint)
            .expect("partial-layer checkpoint rollback");
        for layer in 0..2 {
            assert_eq!(cache.layer_k(layer), &[0., 1., 2., 3., 4.]);
            assert_eq!(cache.layer_v(layer), &[10., 11., 12., 13., 14.]);
        }
        assert_eq!(cache.ctx_offset, 5);
    }

    #[test]
    fn mirror_checkpoint_restores_every_gamma14_prefix_across_window_eviction() {
        for tokens in 1usize..=15 {
            let mut cache = DFlashContextKvCache::new(5, 2, 2, 16);
            for layer in 0..5 {
                let rows = (0..18 * 2)
                    .map(|index| (layer * 100 + index) as f32)
                    .collect::<Vec<_>>();
                cache.append_layer(layer, &rows, &rows, 18);
            }
            cache.advance_round(18);
            let before = cache.snapshot();
            let before_identity = cache.identity();
            let before_revision = cache.revision();
            let checkpoint = cache.checkpoint_append(tokens);
            for layer in 0..5 {
                let fresh = (0..tokens * 2)
                    .map(|index| (10_000 + layer * 100 + index) as f32)
                    .collect::<Vec<_>>();
                cache.append_layer(layer, &fresh, &fresh, tokens);
            }
            cache.advance_round(tokens);
            cache
                .rollback_append(checkpoint)
                .expect("gamma14 checkpoint rollback");
            let after = cache.snapshot();
            assert_eq!(cache.identity(), before_identity);
            assert_eq!(cache.revision(), before_revision);
            assert_eq!(after.position, before.position);
            assert_eq!(after.layers, before.layers);
        }
    }

    #[test]
    fn mirror_checkpoint_rejects_a_different_cache_incarnation_without_mutation() {
        let source = DFlashContextKvCache::new(1, 1, 2, 3);
        let checkpoint = source.checkpoint_append(1);
        let mut other = DFlashContextKvCache::new(1, 1, 2, 3);
        other.append_layer(0, &[7.], &[9.], 1);
        other.advance_round(1);
        let identity = other.identity();
        let revision = other.revision();
        let before = other.snapshot();

        assert!(other.rollback_append(checkpoint).is_err());
        assert_eq!(other.identity(), identity);
        assert_eq!(other.revision(), revision);
        assert_eq!(other.snapshot().position, before.position);
        assert_eq!(other.snapshot().layers, before.layers);
    }

    #[test]
    fn mirror_checkpoint_rejects_intervening_append_without_mutation() {
        let mut cache = DFlashContextKvCache::new(1, 1, 2, 3);
        cache.append_layer(0, &[0., 1., 2.], &[10., 11., 12.], 3);
        cache.advance_round(3);
        let checkpoint = cache.checkpoint_append(2);
        cache.append_layer(0, &[3., 4.], &[13., 14.], 2);
        cache.advance_round(2);
        cache.append_layer(0, &[5.], &[15.], 1);
        cache.advance_round(1);
        let revision = cache.revision();
        let before = cache.snapshot();

        assert!(cache.rollback_append(checkpoint).is_err());
        assert_eq!(cache.revision(), revision);
        assert_eq!(cache.snapshot().position, before.position);
        assert_eq!(cache.snapshot().layers, before.layers);
    }
}
