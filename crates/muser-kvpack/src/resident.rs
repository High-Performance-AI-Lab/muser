//! Compressed resident token radix adapted from Ferrite's Sancho radix path.
//!
//! Unlike the Ferrite server index, entries own immutable, content-interned
//! Muse plane chunks rather than arena request IDs. This makes resident cuts
//! independent of request lifetimes and structurally shares equal chunks.
//!
//! Entries are scoped to the active Muse identity: every radix key is the
//! identity digest words followed by the raw token IDs, so a lookup under a
//! different identity — or under none — can never reach them. The resident
//! tier records provenance but does not authenticate it; the durable tier
//! remains the sole authority on which cuts may be served (see `reuse`).

use std::collections::HashMap;
use std::sync::{Arc, Weak};

use muser_engine::cache::{CachePlaneSnapshot, PlaneEncoding, SessionCacheSnapshot};
use sha2::{Digest, Sha256};

pub const RESIDENT_CUT_TOKENS: u64 = 256;

/// Key words every scoped radix key starts with: a presence tag plus the
/// eight little-endian words of the identity digest. The tag keeps an
/// unscoped (`None`) key disjoint from any digest, all-zero included.
const IDENTITY_KEY_WORDS: usize = 9;

fn scoped_key(identity: Option<[u8; 32]>, tokens: &[u32]) -> Vec<u32> {
    let mut key = Vec::with_capacity(IDENTITY_KEY_WORDS + tokens.len());
    match identity {
        Some(digest) => {
            key.push(1);
            key.extend(
                digest
                    .chunks_exact(4)
                    .map(|word| u32::from_le_bytes(word.try_into().expect("four-byte word"))),
            );
        }
        None => key.resize(IDENTITY_KEY_WORDS, 0),
    }
    key.extend_from_slice(tokens);
    key
}

#[derive(Debug, Clone)]
struct ChunkedPlane {
    layer: u32,
    logical_start: u64,
    logical_count: u64,
    encoding: PlaneEncoding,
    key: Vec<Arc<[u8]>>,
    value: Vec<Arc<[u8]>>,
}

#[derive(Debug, Clone)]
pub struct ResidentSnapshot {
    position: u64,
    tokens: Arc<[u32]>,
    elements_per_token: u32,
    layers: Arc<[ChunkedPlane]>,
    last_logits: Option<Arc<[f32]>>,
}

impl ResidentSnapshot {
    pub fn position(&self) -> u64 {
        self.position
    }

    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    pub fn last_logits(&self) -> Option<&[f32]> {
        self.last_logits.as_deref()
    }

    pub fn materialize(&self) -> Result<SessionCacheSnapshot, ResidentError> {
        let layers = self
            .layers
            .iter()
            .map(|plane| CachePlaneSnapshot {
                layer: plane.layer,
                logical_start: plane.logical_start,
                logical_count: plane.logical_count,
                encoding: plane.encoding,
                key: concatenate(&plane.key).into(),
                value: concatenate(&plane.value).into(),
            })
            .collect::<Vec<_>>();
        let snapshot = SessionCacheSnapshot {
            position: self.position,
            tokens: Arc::clone(&self.tokens),
            elements_per_token: self.elements_per_token,
            layers: layers.into(),
        };
        snapshot
            .validate()
            .map_err(ResidentError::InvalidSnapshot)?;
        Ok(snapshot)
    }
}

#[derive(Debug, Clone)]
struct ResidentEntry {
    id: u64,
    snapshot: Arc<ResidentSnapshot>,
    ancestor_reusable: bool,
    last_access: u64,
}

#[derive(Debug, Default)]
struct RadixNode {
    edge: Vec<u32>,
    entry: Option<ResidentEntry>,
    children: HashMap<u32, RadixNode>,
}

#[derive(Debug, Clone)]
pub struct ResidentHit {
    pub cut: usize,
    pub exact: bool,
    pub snapshot: Arc<ResidentSnapshot>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResidentStats {
    pub entries: usize,
    pub nodes: usize,
    pub unique_chunk_bytes: u64,
    pub queries: u64,
    pub hits: u64,
    pub evictions: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum ResidentError {
    #[error("invalid resident snapshot: {0}")]
    InvalidSnapshot(String),
    #[error("resident cache capacity must be nonzero")]
    ZeroCapacity,
    #[error("resident snapshot token witness does not match its radix key")]
    TokenMismatch,
    #[error("resident snapshot size overflow")]
    SizeOverflow,
    #[error("resident exact-final logits are empty or nonfinite")]
    InvalidLogits,
}

#[derive(Default)]
struct ChunkPool {
    by_digest: HashMap<[u8; 32], Vec<Weak<[u8]>>>,
}

impl ChunkPool {
    fn intern(&mut self, bytes: &[u8]) -> Arc<[u8]> {
        let digest: [u8; 32] = Sha256::digest(bytes).into();
        let candidates = self.by_digest.entry(digest).or_default();
        candidates.retain(|candidate| candidate.strong_count() != 0);
        if let Some(existing) = candidates
            .iter()
            .filter_map(Weak::upgrade)
            .find(|existing| existing.as_ref() == bytes)
        {
            return existing;
        }
        let chunk: Arc<[u8]> = Arc::from(bytes);
        candidates.push(Arc::downgrade(&chunk));
        chunk
    }

    fn live_bytes(&mut self) -> u64 {
        let mut bytes = 0u64;
        self.by_digest.retain(|_, candidates| {
            candidates.retain(|candidate| {
                if let Some(chunk) = candidate.upgrade() {
                    bytes = bytes.saturating_add(chunk.len() as u64);
                    true
                } else {
                    false
                }
            });
            !candidates.is_empty()
        });
        bytes
    }
}

pub struct ResidentRadix {
    root: RadixNode,
    chunks: ChunkPool,
    capacity_bytes: u64,
    clock: u64,
    next_id: u64,
    queries: u64,
    hits: u64,
    evictions: u64,
}

impl ResidentRadix {
    pub fn new(capacity_bytes: u64) -> Result<Self, ResidentError> {
        if capacity_bytes == 0 {
            return Err(ResidentError::ZeroCapacity);
        }
        Ok(Self {
            root: RadixNode::default(),
            chunks: ChunkPool::default(),
            capacity_bytes,
            clock: 0,
            next_id: 1,
            queries: 0,
            hits: 0,
            evictions: 0,
        })
    }

    /// Insert an exact completed cut under `identity`. Aligned cuts may
    /// satisfy a descendant query; non-aligned cuts are retained for
    /// exact-hit-only lookup.
    pub fn insert(
        &mut self,
        identity: Option<[u8; 32]>,
        snapshot: &SessionCacheSnapshot,
    ) -> Result<bool, ResidentError> {
        self.insert_with_logits(identity, snapshot, None)
    }

    pub fn insert_with_logits(
        &mut self,
        identity: Option<[u8; 32]>,
        snapshot: &SessionCacheSnapshot,
        last_logits: Option<&[f32]>,
    ) -> Result<bool, ResidentError> {
        snapshot
            .validate()
            .map_err(ResidentError::InvalidSnapshot)?;
        if snapshot.tokens.len() as u64 != snapshot.position {
            return Err(ResidentError::TokenMismatch);
        }
        if last_logits.is_some_and(|values| {
            values.is_empty() || values.iter().any(|value| !value.is_finite())
        }) {
            return Err(ResidentError::InvalidLogits);
        }
        let resident = Arc::new(self.chunk_snapshot(snapshot, last_logits)?);
        self.clock = self.clock.saturating_add(1);
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        let entry = ResidentEntry {
            id,
            snapshot: resident,
            ancestor_reusable: snapshot.position.is_multiple_of(RESIDENT_CUT_TOKENS),
            last_access: self.clock,
        };
        insert_recursive(
            &mut self.root,
            &scoped_key(identity, &snapshot.tokens),
            entry,
        );
        while self.resident_bytes() > self.capacity_bytes {
            let Some(evicted) = find_lru(&self.root) else {
                break;
            };
            remove_entry(&mut self.root, evicted);
            prune_empty(&mut self.root);
            self.evictions = self.evictions.saturating_add(1);
        }
        Ok(find_id(&self.root, id))
    }

    pub fn lookup(&mut self, identity: Option<[u8; 32]>, tokens: &[u32]) -> Option<ResidentHit> {
        self.lookup_capped(identity, tokens, usize::MAX)
    }

    /// Deepest hit under `identity` whose cut does not exceed `max_cut`. The
    /// cap is how the reuse ladder keeps this unauthenticated tier inside the
    /// durable-authenticated envelope: entries beyond the cap are invisible
    /// here even when they match the token chain exactly.
    pub fn lookup_capped(
        &mut self,
        identity: Option<[u8; 32]>,
        tokens: &[u32],
        max_cut: usize,
    ) -> Option<ResidentHit> {
        self.clock = self.clock.saturating_add(1);
        self.queries = self.queries.saturating_add(1);
        let key = scoped_key(identity, tokens);
        let max_depth = IDENTITY_KEY_WORDS.saturating_add(max_cut);
        let mut best = None;
        longest_match_recursive(&mut self.root, &key, 0, self.clock, max_depth, &mut best);
        if best.is_some() {
            self.hits = self.hits.saturating_add(1);
        }
        best
    }

    pub fn stats(&mut self) -> ResidentStats {
        let mut entries = 0;
        let mut nodes = 0;
        count_nodes(&self.root, &mut entries, &mut nodes);
        ResidentStats {
            entries,
            nodes,
            unique_chunk_bytes: self.chunks.live_bytes(),
            queries: self.queries,
            hits: self.hits,
            evictions: self.evictions,
        }
    }

    fn chunk_snapshot(
        &mut self,
        snapshot: &SessionCacheSnapshot,
        last_logits: Option<&[f32]>,
    ) -> Result<ResidentSnapshot, ResidentError> {
        let mut layers = Vec::with_capacity(snapshot.layers.len());
        for plane in snapshot.layers.iter() {
            let row_bytes = (snapshot.elements_per_token as usize)
                .checked_mul(plane.encoding.width_bytes())
                .ok_or(ResidentError::SizeOverflow)?;
            let chunk_bytes = row_bytes
                .checked_mul(RESIDENT_CUT_TOKENS as usize)
                .ok_or(ResidentError::SizeOverflow)?;
            layers.push(ChunkedPlane {
                layer: plane.layer,
                logical_start: plane.logical_start,
                logical_count: plane.logical_count,
                encoding: plane.encoding,
                key: plane
                    .key
                    .chunks(chunk_bytes)
                    .map(|bytes| self.chunks.intern(bytes))
                    .collect(),
                value: plane
                    .value
                    .chunks(chunk_bytes)
                    .map(|bytes| self.chunks.intern(bytes))
                    .collect(),
            });
        }
        Ok(ResidentSnapshot {
            position: snapshot.position,
            tokens: Arc::clone(&snapshot.tokens),
            elements_per_token: snapshot.elements_per_token,
            layers: layers.into(),
            last_logits: last_logits.map(Arc::from),
        })
    }

    fn resident_bytes(&mut self) -> u64 {
        self.chunks
            .live_bytes()
            .saturating_add(entry_logits_bytes(&self.root))
    }
}

fn entry_logits_bytes(node: &RadixNode) -> u64 {
    let own = node
        .entry
        .as_ref()
        .and_then(|entry| entry.snapshot.last_logits.as_ref())
        .map_or(0, |values| (values.len() as u64).saturating_mul(4));
    node.children.values().fold(own, |total, child| {
        total.saturating_add(entry_logits_bytes(child))
    })
}

fn concatenate(chunks: &[Arc<[u8]>]) -> Vec<u8> {
    let length = chunks.iter().map(|chunk| chunk.len()).sum();
    let mut bytes = Vec::with_capacity(length);
    for chunk in chunks {
        bytes.extend_from_slice(chunk);
    }
    bytes
}

fn shared_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter()
        .zip(b)
        .take_while(|(left, right)| left == right)
        .count()
}

fn insert_recursive(node: &mut RadixNode, tokens: &[u32], entry: ResidentEntry) {
    if tokens.is_empty() {
        node.entry = Some(entry);
        return;
    }
    let first = tokens[0];
    if let Some(child) = node.children.get_mut(&first) {
        let shared = shared_prefix_len(&child.edge, tokens);
        if shared == child.edge.len() {
            insert_recursive(child, &tokens[shared..], entry);
            return;
        }
        let old = node.children.remove(&first).expect("child existed");
        let old_suffix = old.edge[shared..].to_vec();
        let mut split = RadixNode {
            edge: old.edge[..shared].to_vec(),
            entry: None,
            children: HashMap::new(),
        };
        split.children.insert(
            old_suffix[0],
            RadixNode {
                edge: old_suffix,
                entry: old.entry,
                children: old.children,
            },
        );
        let new_suffix = &tokens[shared..];
        if new_suffix.is_empty() {
            split.entry = Some(entry);
        } else {
            split.children.insert(
                new_suffix[0],
                RadixNode {
                    edge: new_suffix.to_vec(),
                    entry: Some(entry),
                    children: HashMap::new(),
                },
            );
        }
        node.children.insert(first, split);
    } else {
        node.children.insert(
            first,
            RadixNode {
                edge: tokens.to_vec(),
                entry: Some(entry),
                children: HashMap::new(),
            },
        );
    }
}

fn longest_match_recursive(
    node: &mut RadixNode,
    remaining: &[u32],
    depth: usize,
    now: u64,
    max_depth: usize,
    best: &mut Option<ResidentHit>,
) {
    if let Some(entry) = node.entry.as_mut() {
        // Scoped keys put every entry at least IDENTITY_KEY_WORDS deep, so
        // the subtraction below cannot underflow.
        let exact = remaining.is_empty();
        if depth <= max_depth && (exact || entry.ancestor_reusable) {
            entry.last_access = now;
            *best = Some(ResidentHit {
                cut: depth - IDENTITY_KEY_WORDS,
                exact,
                snapshot: Arc::clone(&entry.snapshot),
            });
        }
    }
    // Entries below this depth sit beyond the cap; there is nothing to find.
    if depth >= max_depth {
        return;
    }
    let Some((&first, _)) = remaining.split_first() else {
        return;
    };
    if let Some(child) = node.children.get_mut(&first) {
        let shared = shared_prefix_len(&child.edge, remaining);
        if shared == child.edge.len() {
            longest_match_recursive(
                child,
                &remaining[shared..],
                depth + shared,
                now,
                max_depth,
                best,
            );
        }
    }
}

fn find_lru(root: &RadixNode) -> Option<u64> {
    fn visit(node: &RadixNode, best: &mut Option<(u64, u64)>) {
        if let Some(entry) = &node.entry {
            if best.is_none_or(|(_, age)| entry.last_access < age) {
                *best = Some((entry.id, entry.last_access));
            }
        }
        for child in node.children.values() {
            visit(child, best);
        }
    }
    let mut best = None;
    visit(root, &mut best);
    best.map(|(id, _)| id)
}

fn remove_entry(node: &mut RadixNode, id: u64) -> bool {
    if node.entry.as_ref().is_some_and(|entry| entry.id == id) {
        node.entry = None;
        return true;
    }
    node.children
        .values_mut()
        .any(|child| remove_entry(child, id))
}

fn prune_empty(node: &mut RadixNode) {
    node.children.retain(|_, child| {
        prune_empty(child);
        child.entry.is_some() || !child.children.is_empty()
    });
}

fn find_id(node: &RadixNode, id: u64) -> bool {
    node.entry.as_ref().is_some_and(|entry| entry.id == id)
        || node.children.values().any(|child| find_id(child, id))
}

fn count_nodes(node: &RadixNode, entries: &mut usize, nodes: &mut usize) {
    *nodes += 1;
    if node.entry.is_some() {
        *entries += 1;
    }
    for child in node.children.values() {
        count_nodes(child, entries, nodes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use muser_engine::config::{layer_kind, MUSE_LAYER_COUNT, MUSE_SWA_WINDOW};

    fn snapshot(position: usize) -> SessionCacheSnapshot {
        let tokens: Arc<[u32]> = (0..position as u32).collect::<Vec<_>>().into();
        let layers = (0..MUSE_LAYER_COUNT)
            .map(|layer| {
                let count = if layer_kind(layer).unwrap().is_swa() {
                    position.min(MUSE_SWA_WINDOW)
                } else {
                    position
                };
                let start = position - count;
                let plane = |salt: u16| {
                    let mut bytes = Vec::with_capacity(count * 2);
                    for logical in start..position {
                        bytes.extend_from_slice(
                            &((logical as u16)
                                .wrapping_add(layer as u16)
                                .wrapping_add(salt))
                            .to_le_bytes(),
                        );
                    }
                    Arc::<[u8]>::from(bytes)
                };
                CachePlaneSnapshot {
                    layer: layer as u32,
                    logical_start: start as u64,
                    logical_count: count as u64,
                    encoding: PlaneEncoding::F16Le,
                    key: plane(0),
                    value: plane(10_000),
                }
            })
            .collect::<Vec<_>>();
        SessionCacheSnapshot {
            position: position as u64,
            tokens,
            elements_per_token: 1,
            layers: layers.into(),
        }
    }

    #[test]
    fn compressed_radix_shares_chunks_and_enforces_exact_only_final_cuts() {
        let first = snapshot(256);
        let second = snapshot(512);
        let final_cut = snapshot(513);
        let mut radix = ResidentRadix::new(16 * 1024 * 1024).unwrap();
        assert!(radix.insert(None, &first).unwrap());
        assert!(radix.insert(None, &second).unwrap());
        let stats = radix.stats();
        let naive_bytes = first
            .layers
            .iter()
            .chain(second.layers.iter())
            .map(|plane| plane.key.len() + plane.value.len())
            .sum::<usize>() as u64;
        assert!(stats.unique_chunk_bytes < naive_bytes);

        let query: Vec<u32> = (0..600).collect();
        let hit = radix.lookup(None, &query).unwrap();
        assert_eq!(hit.cut, 512);
        assert!(!hit.exact);
        assert_eq!(hit.snapshot.materialize().unwrap(), second);

        assert!(radix.insert(None, &final_cut).unwrap());
        let exact = radix.lookup(None, &final_cut.tokens).unwrap();
        assert_eq!((exact.cut, exact.exact), (513, true));
        let extended: Vec<u32> = (0..514).collect();
        let ancestor = radix.lookup(None, &extended).unwrap();
        assert_eq!((ancestor.cut, ancestor.exact), (512, false));
    }

    #[test]
    fn exact_final_entry_retains_its_witnessed_logits() {
        let cut = snapshot(256);
        let logits = [0.25, -1.5, 3.0, 0.0];
        let mut radix = ResidentRadix::new(16 * 1024 * 1024).unwrap();
        assert!(radix.insert_with_logits(None, &cut, Some(&logits)).unwrap());
        let hit = radix.lookup(None, &cut.tokens).unwrap();
        assert!(hit.exact);
        assert_eq!(hit.snapshot.last_logits(), Some(logits.as_slice()));
    }

    #[test]
    fn identity_scopes_entries_and_a_wrong_identity_is_a_miss() {
        let cut = snapshot(256);
        let identity = [7; 32];
        let other = [9; 32];
        let mut radix = ResidentRadix::new(16 * 1024 * 1024).unwrap();
        assert!(radix.insert(Some(identity), &cut).unwrap());

        // Same token chain, different or absent identity: never a hit.
        assert!(radix.lookup(Some(other), &cut.tokens).is_none());
        assert!(radix.lookup(None, &cut.tokens).is_none());
        let hit = radix.lookup(Some(identity), &cut.tokens).unwrap();
        assert_eq!((hit.cut, hit.exact), (256, true));

        // The same chain under a second identity coexists without displacing
        // the first, and an unscoped entry stays invisible to both.
        assert!(radix.insert(Some(other), &cut).unwrap());
        assert!(radix.lookup(Some(other), &cut.tokens).is_some());
        assert!(radix.lookup(Some(identity), &cut.tokens).is_some());
        assert!(radix.insert(None, &cut).unwrap());
        assert!(radix.lookup(None, &cut.tokens).is_some());
        assert_eq!(radix.lookup(Some(identity), &cut.tokens).unwrap().cut, 256);
        assert_eq!(radix.stats().entries, 3);
    }

    #[test]
    fn capped_lookup_never_serves_beyond_the_cap() {
        let first = snapshot(256);
        let second = snapshot(512);
        let identity = [7; 32];
        let mut radix = ResidentRadix::new(16 * 1024 * 1024).unwrap();
        assert!(radix.insert(Some(identity), &first).unwrap());
        assert!(radix.insert(Some(identity), &second).unwrap());

        let query: Vec<u32> = (0..600).collect();
        assert_eq!(radix.lookup(Some(identity), &query).unwrap().cut, 512);
        assert_eq!(
            radix
                .lookup_capped(Some(identity), &query, 512)
                .unwrap()
                .cut,
            512
        );
        // Any cap below the deeper cut falls back to the deepest entry
        // inside the envelope; a cap below every entry is a miss.
        assert_eq!(
            radix
                .lookup_capped(Some(identity), &query, 511)
                .unwrap()
                .cut,
            256
        );
        assert_eq!(
            radix
                .lookup_capped(Some(identity), &query, 256)
                .unwrap()
                .cut,
            256
        );
        assert!(radix.lookup_capped(Some(identity), &query, 255).is_none());
        assert!(radix.lookup_capped(Some(identity), &query, 0).is_none());
    }

    #[test]
    fn capacity_evicts_whole_entries_without_corrupting_remaining_snapshots() {
        let first = snapshot(256);
        let second = snapshot(512);
        let one_snapshot_bytes = first
            .layers
            .iter()
            .map(|plane| plane.key.len() + plane.value.len())
            .sum::<usize>() as u64;
        let mut radix = ResidentRadix::new(one_snapshot_bytes + 1024).unwrap();
        assert!(radix.insert(None, &first).unwrap());
        let _ = radix.insert(None, &second).unwrap();
        let stats = radix.stats();
        assert!(stats.evictions >= 1);
        assert!(stats.unique_chunk_bytes <= one_snapshot_bytes + 1024);
    }

    #[test]
    fn in_flight_hit_survives_eviction_without_becoming_a_semantic_false_positive() {
        let first = snapshot(256);
        let second = snapshot(512);
        let one_snapshot_bytes = first
            .layers
            .iter()
            .map(|plane| plane.key.len() + plane.value.len())
            .sum::<usize>() as u64;
        let mut radix = ResidentRadix::new(one_snapshot_bytes + 1024).unwrap();
        assert!(radix.insert(None, &first).unwrap());
        let held = radix.lookup(None, &first.tokens).unwrap();

        // The hit owns immutable chunks. Capacity pressure may remove its
        // radix entry while a restore is materializing, but can never mutate
        // or reuse those bytes for the new generation.
        let _ = radix.insert(None, &second).unwrap();
        assert_eq!(held.snapshot.materialize().unwrap(), first);

        let mut wrong = first.tokens.to_vec();
        wrong[127] ^= 0x8000_0000;
        assert!(radix.lookup(None, &wrong).is_none());
    }
}
