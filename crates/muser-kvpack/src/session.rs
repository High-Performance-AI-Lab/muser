//! Authenticated Muse save, deepest-ancestor lookup, and detached restore.

use std::collections::BTreeMap;
use std::sync::Arc;

use kvpack::{
    LocalStore, MuseSessionArtifact, MuseSessionArtifactReceipt, MuseSessionWriter,
    RestoreCancellation, RestoreLimits, RestoreStatePlan, StoreError, VerifiedRestoreSink,
    WritePolicy, MUSE_EXACT_LOGITS_LAYER, MUSE_EXACT_LOGITS_STATE,
};
use kvpack_core::{representation_family_id, semantic_model_id, Id32, StateKey};
use muser_engine::cache::{CachePlaneSnapshot, PlaneEncoding, SessionCacheSnapshot};
use muser_engine::config::{MuseConfig, MUSE_LAYER_COUNT};
use muser_engine::{EngineError, Session};

use crate::layout::{descriptor, LayoutError, MuseIdentity};

pub const DURABLE_FULL_INTERVAL: u64 = 2_048;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublishReason {
    Interval,
    PromptBoundary,
    TurnBoundary,
    Explicit,
}

pub fn should_publish_full(cut: u64, reason: PublishReason) -> bool {
    cut != 0
        && (cut.is_multiple_of(DURABLE_FULL_INTERVAL)
            || matches!(
                reason,
                PublishReason::PromptBoundary
                    | PublishReason::TurnBoundary
                    | PublishReason::Explicit
            ))
}

#[derive(Debug, Clone)]
pub struct DurableHit {
    pub cut: usize,
    pub manifest_id: Id32,
}

#[derive(Debug, thiserror::Error)]
pub enum SessionCacheError {
    #[error(transparent)]
    Layout(#[from] LayoutError),
    #[error(transparent)]
    Kvpack(#[from] StoreError),
    #[error(transparent)]
    Engine(#[from] EngineError),
    #[error("durable Muse artifacts require production F16 cache planes")]
    ProductionF16Required,
    #[error("cache snapshot does not match its descriptor: {0}")]
    Snapshot(String),
    #[error(
        "failed to restore the previous live generation after an exact-state install error: {0}"
    )]
    Rollback(String),
}

/// The engine installs KV planes atomically, but exact-final state is KV plus
/// its authenticated final distribution. Keep the old generation available
/// until both pieces have been accepted so a mismatched consumer cannot leave
/// new KV paired with old/no logits.
struct SessionCheckpoint {
    snapshot: Option<SessionCacheSnapshot>,
    logits: Option<Vec<f32>>,
}

impl SessionCheckpoint {
    fn capture(session: &Session) -> Result<Self, EngineError> {
        Ok(Self {
            snapshot: if session.position() == 0 {
                None
            } else {
                Some(session.export_cache_snapshot()?)
            },
            logits: session.cached_logits().map(<[f32]>::to_vec),
        })
    }

    fn restore(self, session: &mut Session) -> Result<(), EngineError> {
        if let Some(snapshot) = self.snapshot {
            session.install_cache_snapshot(&snapshot)?;
        } else {
            session.reset();
        }
        if let Some(logits) = self.logits {
            session.install_restored_logits(&logits)?;
        }
        Ok(())
    }
}

pub(crate) fn install_exact_generation(
    session: &mut Session,
    snapshot: &SessionCacheSnapshot,
    logits: &[f32],
) -> Result<(), SessionCacheError> {
    let checkpoint = SessionCheckpoint::capture(session)?;
    session.install_cache_snapshot(snapshot)?;
    if let Err(install_error) = session.install_restored_logits(logits) {
        checkpoint.restore(session).map_err(|rollback| {
            SessionCacheError::Rollback(format!(
                "install error: {install_error}; rollback error: {rollback}"
            ))
        })?;
        return Err(install_error.into());
    }
    Ok(())
}

/// The authenticated `PrefixNode`/`LocalStore` path is the sole durable
/// authority. This wrapper never derives or catalogs a second prefix ID.
pub struct DurableCache {
    store: Arc<LocalStore>,
    config: MuseConfig,
    identity: MuseIdentity,
}

impl DurableCache {
    pub fn new(store: Arc<LocalStore>, config: MuseConfig, identity: MuseIdentity) -> Self {
        Self {
            store,
            config,
            identity,
        }
    }

    pub fn store(&self) -> &Arc<LocalStore> {
        &self.store
    }

    pub fn config(&self) -> &MuseConfig {
        &self.config
    }

    pub fn identity(&self) -> &MuseIdentity {
        &self.identity
    }

    pub fn save(
        &self,
        session: &Session,
        idempotency_key: Id32,
        publication_generation: u64,
    ) -> Result<MuseSessionArtifactReceipt, SessionCacheError> {
        let snapshot = session.export_cache_snapshot()?;
        let logits = session
            .cached_logits()
            .ok_or_else(|| SessionCacheError::Snapshot("durable cut has no final logits".into()))?;
        self.save_snapshot(&snapshot, logits, idempotency_key, publication_generation)
    }

    pub fn save_snapshot(
        &self,
        snapshot: &SessionCacheSnapshot,
        last_logits: &[f32],
        idempotency_key: Id32,
        publication_generation: u64,
    ) -> Result<MuseSessionArtifactReceipt, SessionCacheError> {
        if snapshot.encoding() != Some(PlaneEncoding::F16Le) {
            return Err(SessionCacheError::ProductionF16Required);
        }
        if last_logits.len() != self.config.vocab_size
            || last_logits.iter().any(|value| !value.is_finite())
        {
            return Err(SessionCacheError::Snapshot(
                "durable exact-final logits have invalid geometry or values".into(),
            ));
        }
        let cached = u32::try_from(snapshot.position)
            .map_err(|_| SessionCacheError::Snapshot("cut exceeds u32".into()))?;
        let descriptor = descriptor(&self.config, &self.identity, cached)?;
        let policy = WritePolicy::exact_qualified(
            idempotency_key,
            descriptor.semantic_model,
            &descriptor.family,
        )?
        .with_publication_generation(publication_generation)?;
        let mut writer = MuseSessionWriter::begin(
            Arc::clone(&self.store),
            descriptor.clone(),
            snapshot.tokens.to_vec(),
            policy,
        )?;
        for family in &descriptor.family.states {
            if family.key.layer == MUSE_EXACT_LOGITS_LAYER
                && family.key.state_name == MUSE_EXACT_LOGITS_STATE
            {
                let mut destination = writer.next_plane(family.key.clone())?;
                let bytes = last_logits
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>();
                destination.write_all(&bytes)?;
                destination.finish()?;
                continue;
            }
            let plane = snapshot
                .layers
                .get(family.key.layer as usize)
                .ok_or_else(|| SessionCacheError::Snapshot("descriptor layer is absent".into()))?;
            let bytes = match family.key.state_name.as_str() {
                "attn.k" => &plane.key,
                "attn.v" => &plane.value,
                _ => {
                    return Err(SessionCacheError::Snapshot(
                        "descriptor contains an unknown Muse state".into(),
                    ))
                }
            };
            let mut destination = writer.next_plane(family.key.clone())?;
            destination.write_all(bytes)?;
            destination.finish()?;
        }
        Ok(writer.commit()?)
    }

    pub fn find_deepest(&self, tokens: &[u32]) -> Result<Option<DurableHit>, SessionCacheError> {
        if tokens.is_empty() {
            return Ok(None);
        }
        let requested_count = u32::try_from(tokens.len())
            .map_err(|_| SessionCacheError::Snapshot("request exceeds u32".into()))?;
        let requested = descriptor(&self.config, &self.identity, requested_count)?;
        let (_, nodes) = self.store.derive_input_cut(
            &requested.semantic_model,
            &requested.family,
            tokens,
            &[],
        )?;
        let semantic = semantic_model_id(&requested.semantic_model);
        let family = representation_family_id(&requested.family).map_err(StoreError::from)?;
        Ok(self
            .store
            .resolve_prefix(&nodes, &semantic, &family, 64)?
            .map(|hit| DurableHit {
                cut: hit.token_count as usize,
                manifest_id: hit.manifest_id,
            }))
    }

    /// Resolve and install the deepest authenticated local ancestor. The
    /// caller prefills only `tokens[hit.cut..]` after this returns.
    pub fn restore_deepest(
        &self,
        session: &mut Session,
        tokens: &[u32],
    ) -> Result<Option<DurableHit>, SessionCacheError> {
        let Some(hit) = self.find_deepest(tokens)? else {
            return Ok(None);
        };
        self.restore_manifest(
            session,
            tokens,
            hit.clone(),
            &RestoreCancellation::default(),
        )?;
        Ok(Some(hit))
    }

    /// Restore one already-resolved authenticated manifest. Remote durable
    /// acquisition uses this only after importing and re-resolving the exact
    /// manifest through the receiver's own PrefixNode catalog.
    pub fn restore_manifest(
        &self,
        session: &mut Session,
        tokens: &[u32],
        hit: DurableHit,
        cancellation: &RestoreCancellation,
    ) -> Result<(), SessionCacheError> {
        let cut = hit.cut;
        if cut == 0 || cut > tokens.len() {
            return Err(SessionCacheError::Snapshot(
                "durable prefix result is outside the request".into(),
            ));
        }
        let prefix = &tokens[..cut];
        let runtime = descriptor(&self.config, &self.identity, cut as u32)?;
        let artifact = MuseSessionArtifact::open(
            Arc::clone(&self.store),
            hit.manifest_id,
            &runtime,
            prefix,
            self.store.minimum_readable_key_epoch(),
            RestoreLimits::default(),
        )?;
        let mut sink = StagedMuseSink::default();
        let installed = artifact
            .restore_plan()
            .restore_parallel(&mut sink, cancellation, 4)?;
        let (snapshot, last_logits) =
            sink.into_snapshot(prefix, self.config.kv_dim() as u32, self.config.vocab_size)?;
        // This adapter copies every verified chunk into an owned detached
        // snapshot; the engine never aliases kvpack's pinned source bytes.
        // Release can therefore happen before the live commit. If catalog
        // acknowledgement fails, the previous generation remains untouched.
        installed.engine_free()?;
        if cancellation.is_cancelled() {
            return Err(StoreError::Cancelled.into());
        }
        install_exact_generation(session, &snapshot, &last_logits)?;
        Ok(())
    }
}

#[derive(Default)]
struct StagedMuseSink {
    states: BTreeMap<StateKey, StagedState>,
    committed: bool,
}

struct StagedState {
    plan: RestoreStatePlan,
    bytes: Vec<u8>,
}

impl VerifiedRestoreSink for StagedMuseSink {
    fn begin_restore(
        &mut self,
        _artifact: Id32,
        states: &[RestoreStatePlan],
    ) -> Result<(), StoreError> {
        if !self.states.is_empty() || self.committed {
            return Err(StoreError::State(
                "Muse restore shadow was already initialized",
            ));
        }
        for state in states {
            let length = usize::try_from(state.plaintext_bytes)
                .map_err(|_| StoreError::State("Muse restore plane exceeds usize"))?;
            self.states.insert(
                state.declaration.key.clone(),
                StagedState {
                    plan: state.clone(),
                    bytes: vec![0; length],
                },
            );
        }
        Ok(())
    }

    fn write_verified_chunk(
        &mut self,
        state: &StateKey,
        logical_offset: u64,
        plaintext: &[u8],
    ) -> Result<(), StoreError> {
        let target = self.states.get_mut(state).ok_or(StoreError::State(
            "restore targeted an undeclared Muse plane",
        ))?;
        let start = usize::try_from(logical_offset)
            .map_err(|_| StoreError::State("Muse restore offset exceeds usize"))?;
        let end = start
            .checked_add(plaintext.len())
            .ok_or(StoreError::State("Muse restore write overflows usize"))?;
        let destination = target
            .bytes
            .get_mut(start..end)
            .ok_or(StoreError::State("Muse restore write exceeds its shadow"))?;
        destination.copy_from_slice(plaintext);
        Ok(())
    }

    fn commit_restore(&mut self) -> Result<(), StoreError> {
        self.committed = true;
        Ok(())
    }

    fn abort_restore(&mut self) {
        self.states.clear();
        self.committed = false;
    }
}

impl StagedMuseSink {
    fn into_snapshot(
        self,
        tokens: &[u32],
        elements_per_token: u32,
        vocab_size: usize,
    ) -> Result<(SessionCacheSnapshot, Vec<f32>), SessionCacheError> {
        if !self.committed {
            return Err(SessionCacheError::Snapshot(
                "Muse restore shadow was not committed".into(),
            ));
        }
        let mut keys: Vec<Option<StagedState>> = (0..MUSE_LAYER_COUNT).map(|_| None).collect();
        let mut values: Vec<Option<StagedState>> = (0..MUSE_LAYER_COUNT).map(|_| None).collect();
        let mut exact_logits = None;
        for (key, state) in self.states {
            if key.layer == MUSE_EXACT_LOGITS_LAYER && key.state_name == MUSE_EXACT_LOGITS_STATE {
                if exact_logits.replace(state).is_some() {
                    return Err(SessionCacheError::Snapshot(
                        "restore contains duplicate exact-final logits".into(),
                    ));
                }
                continue;
            }
            let layer = key.layer as usize;
            if layer >= MUSE_LAYER_COUNT {
                return Err(SessionCacheError::Snapshot(
                    "restore contains an out-of-range Muse layer".into(),
                ));
            }
            match key.state_name.as_str() {
                "attn.k" if keys[layer].is_none() => keys[layer] = Some(state),
                "attn.v" if values[layer].is_none() => values[layer] = Some(state),
                _ => {
                    return Err(SessionCacheError::Snapshot(
                        "restore contains a duplicate or unknown Muse plane".into(),
                    ))
                }
            }
        }
        let mut layers = Vec::with_capacity(MUSE_LAYER_COUNT);
        for layer in 0..MUSE_LAYER_COUNT {
            let key = keys[layer]
                .take()
                .ok_or_else(|| SessionCacheError::Snapshot("restore is missing K".into()))?;
            let value = values[layer]
                .take()
                .ok_or_else(|| SessionCacheError::Snapshot("restore is missing V".into()))?;
            if key.plan.declaration.logical_start != value.plan.declaration.logical_start
                || key.plan.declaration.logical_count != value.plan.declaration.logical_count
            {
                return Err(SessionCacheError::Snapshot(
                    "restored K/V logical ranges disagree".into(),
                ));
            }
            layers.push(CachePlaneSnapshot {
                layer: layer as u32,
                logical_start: key.plan.declaration.logical_start,
                logical_count: key.plan.declaration.logical_count,
                encoding: PlaneEncoding::F16Le,
                key: key.bytes.into(),
                value: value.bytes.into(),
            });
        }
        let snapshot = SessionCacheSnapshot {
            position: tokens.len() as u64,
            tokens: tokens.to_vec().into(),
            elements_per_token,
            layers: layers.into(),
        };
        snapshot.validate().map_err(SessionCacheError::Snapshot)?;
        let last_logits = exact_logits
            .ok_or_else(|| {
                SessionCacheError::Snapshot(
                    "restore is missing authenticated exact-final logits".into(),
                )
            })
            .and_then(|state| {
                if state.bytes.len() != vocab_size.saturating_mul(4) {
                    return Err(SessionCacheError::Snapshot(
                        "restored exact-final logits have the wrong byte size".into(),
                    ));
                }
                let logits = state
                    .bytes
                    .chunks_exact(4)
                    .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
                    .collect::<Vec<_>>();
                if logits.iter().any(|value| !value.is_finite()) {
                    return Err(SessionCacheError::Snapshot(
                        "restored exact-final logits contain nonfinite values".into(),
                    ));
                }
                Ok(logits)
            })?;
        Ok((snapshot, last_logits))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kvpack::{create_store_key_random, load_store_key, StoreConfig};
    use muser_engine::config::{layer_kind, MUSE_HEAD_COUNT, MUSE_KV_HEAD_COUNT};

    fn config() -> MuseConfig {
        MuseConfig {
            n_layers: MUSE_LAYER_COUNT,
            hidden_dim: 2_048,
            n_heads: MUSE_HEAD_COUNT,
            n_kv_heads: MUSE_KV_HEAD_COUNT,
            head_dim: 128,
            value_head_dim: 128,
            intermediate_dim: 6_144,
            vocab_size: 128,
            context_length: 131_072,
            rms_eps: 1e-5,
            post_norm_eps: 1e-8,
            rope_base_swa: 500_000.0,
            rope_base_full: 10_000.0,
            rope_dim: 128,
            sliding_window: 2_048,
            layer_kinds: (0..MUSE_LAYER_COUNT)
                .map(|layer| layer_kind(layer).unwrap())
                .collect(),
            final_logit_softcap: 20.0,
            logit_scale: 0.196_116_13,
            qk_scale_factor: 3.87,
            bos_token_id: Some(1),
            eos_tokens: vec![2, 3],
        }
    }

    fn identity() -> MuseIdentity {
        MuseIdentity {
            model_sha256: [1; 32],
            adapter_sha256: [0; 32],
            tokenizer_sha256: [2; 32],
            chat_template_sha256: [3; 32],
            context_policy_sha256: [4; 32],
            model_revision: "test-muse".into(),
            tokenizer_revision: "test-tokenizer".into(),
            // Exercise publication and restore under the P-series product
            // identity. Q4_K/NVFP4 separation is asserted below.
            weight_precision: "nvfp4".into(),
        }
    }

    fn snapshot(position: usize) -> SessionCacheSnapshot {
        let elements = 256usize;
        let tokens: Arc<[u32]> = (0..position as u32).collect::<Vec<_>>().into();
        let layers = (0..MUSE_LAYER_COUNT)
            .map(|layer| {
                let count = if layer_kind(layer).unwrap().is_swa() {
                    position.min(2_048)
                } else {
                    position
                };
                let start = position - count;
                let bytes = count * elements * 2;
                CachePlaneSnapshot {
                    layer: layer as u32,
                    logical_start: start as u64,
                    logical_count: count as u64,
                    encoding: PlaneEncoding::F16Le,
                    key: vec![layer as u8; bytes].into(),
                    value: vec![layer.wrapping_add(1) as u8; bytes].into(),
                }
            })
            .collect::<Vec<_>>();
        SessionCacheSnapshot {
            position: position as u64,
            tokens,
            elements_per_token: elements as u32,
            layers: layers.into(),
        }
    }

    #[test]
    fn durable_authority_resolves_exact_and_deepest_aligned_ancestor() {
        let temp = tempfile::tempdir().unwrap();
        let key_path = temp.path().join("keys/root.key");
        create_store_key_random(&key_path, temp.path()).unwrap();
        let store = Arc::new(
            LocalStore::open(
                StoreConfig {
                    object_root: temp.path().join("objects"),
                    catalog_path: temp.path().join("catalog/catalog.sqlite"),
                    operator_tenant_id: b"muser-durable-test".to_vec(),
                    key_epoch: 1,
                    minimum_readable_key_epoch: 1,
                    catalog_epoch: 1,
                    quota_bytes: 256 * 1024 * 1024,
                    staging_quota_bytes: 256 * 1024 * 1024,
                    endurance_bytes_per_five_minutes: 256 * 1024 * 1024,
                },
                load_store_key(&key_path, temp.path()).unwrap(),
            )
            .unwrap(),
        );
        let cache = DurableCache::new(store, config(), identity());
        let cut = snapshot(256);
        let logits = (0..128)
            .map(|index| index as f32 / 128.0)
            .collect::<Vec<_>>();
        let receipt = cache.save_snapshot(&cut, &logits, [9; 32], 1).unwrap();
        let runtime = descriptor(&cache.config, &cache.identity, 256).unwrap();
        let artifact = MuseSessionArtifact::open(
            Arc::clone(&cache.store),
            receipt.manifest_id,
            &runtime,
            &cut.tokens,
            1,
            RestoreLimits::default(),
        )
        .unwrap();
        let mut sink = StagedMuseSink::default();
        let installed = artifact
            .restore_plan()
            .restore_parallel(&mut sink, &RestoreCancellation::default(), 2)
            .unwrap();
        let (_, restored_logits) = sink.into_snapshot(&cut.tokens, 256, 128).unwrap();
        installed.engine_free().unwrap();
        assert_eq!(restored_logits, logits);

        // Even a completely verified KV payload is not an exact Muse
        // generation without its authenticated cut-scoped distribution.
        let mut incomplete = StagedMuseSink::default();
        let installed = artifact
            .restore_plan()
            .restore_parallel(&mut incomplete, &RestoreCancellation::default(), 2)
            .unwrap();
        incomplete.states.remove(&StateKey::new(
            MUSE_EXACT_LOGITS_LAYER,
            MUSE_EXACT_LOGITS_STATE,
        ));
        let error = incomplete.into_snapshot(&cut.tokens, 256, 128).unwrap_err();
        installed.engine_free().unwrap();
        assert!(error
            .to_string()
            .contains("missing authenticated exact-final logits"));

        let cancellation = RestoreCancellation::default();
        cancellation.cancel();
        let mut cancelled = StagedMuseSink::default();
        assert!(artifact
            .restore_plan()
            .restore_parallel(&mut cancelled, &cancellation, 2)
            .is_err());
        assert!(cancelled.states.is_empty());
        assert!(!cancelled.committed);
        let exact = cache.find_deepest(&cut.tokens).unwrap().unwrap();
        assert_eq!(exact.cut, 256);
        assert_eq!(exact.manifest_id, receipt.manifest_id);
        let extended: Vec<u32> = (0..257).collect();
        let ancestor = cache.find_deepest(&extended).unwrap().unwrap();
        assert_eq!(ancestor.cut, 256);
        assert_eq!(ancestor.manifest_id, receipt.manifest_id);

        // A completed non-aligned prompt/turn boundary remains exact-only.
        // Its partial final prefix block cannot become an ancestor authority;
        // the earlier aligned manifest remains the deepest reusable cut.
        let final_cut = snapshot(257);
        let final_receipt = cache
            .save_snapshot(&final_cut, &logits, [10; 32], 2)
            .unwrap();
        let exact_final = cache.find_deepest(&final_cut.tokens).unwrap().unwrap();
        assert_eq!(exact_final.cut, 257);
        assert_eq!(exact_final.manifest_id, final_receipt.manifest_id);
        let descendant: Vec<u32> = (0..258).collect();
        let aligned_ancestor = cache.find_deepest(&descendant).unwrap().unwrap();
        assert_eq!(aligned_ancestor.cut, 256);
        assert_eq!(aligned_ancestor.manifest_id, receipt.manifest_id);

        // Model/layout/tokenizer identity is part of the keyed prefix
        // authority. The same token witness under another identity is a miss,
        // never a semantically plausible restore candidate.
        let mut wrong_identity = identity();
        wrong_identity.model_sha256[0] ^= 1;
        let incompatible = DurableCache::new(
            Arc::clone(&cache.store),
            cache.config.clone(),
            wrong_identity,
        );
        assert!(incompatible.find_deepest(&cut.tokens).unwrap().is_none());

        let mut wrong_precision = identity();
        wrong_precision.weight_precision = "q4_k_xl".into();
        let incompatible_precision = DurableCache::new(
            Arc::clone(&cache.store),
            cache.config.clone(),
            wrong_precision,
        );
        assert!(incompatible_precision
            .find_deepest(&cut.tokens)
            .unwrap()
            .is_none());

        // Authenticated object corruption is detected before a restore sink
        // can expose a committed generation.
        let manifest_hex = receipt
            .manifest_id
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let manifest_path = temp
            .path()
            .join("objects/manifests")
            .join(&manifest_hex[..2])
            .join(format!("{manifest_hex}.kvpack"));
        let mut corrupted = std::fs::read(&manifest_path).unwrap();
        let offset = corrupted.len() / 2;
        corrupted[offset] ^= 1;
        std::fs::write(&manifest_path, corrupted).unwrap();
        assert!(MuseSessionArtifact::open(
            Arc::clone(&cache.store),
            receipt.manifest_id,
            &runtime,
            &cut.tokens,
            1,
            RestoreLimits::default(),
        )
        .is_err());
    }

    #[test]
    fn durable_publication_schedule_includes_interval_boundaries_and_explicit_cuts() {
        assert!(should_publish_full(2_048, PublishReason::Interval));
        assert!(should_publish_full(513, PublishReason::PromptBoundary));
        assert!(should_publish_full(777, PublishReason::TurnBoundary));
        assert!(should_publish_full(1, PublishReason::Explicit));
        assert!(!should_publish_full(513, PublishReason::Interval));
        assert!(!should_publish_full(0, PublishReason::Explicit));
    }
}
