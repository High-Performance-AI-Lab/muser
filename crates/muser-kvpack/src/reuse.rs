//! Ordered exact-prefix reuse: current session, resident, durable, then remote.

use kvpack::RestoreCancellation;
use muser_engine::Session;

use crate::remote::{RemoteCache, RemotePrefixError};
use crate::resident::{ResidentError, ResidentHit, ResidentRadix};
use crate::session::{install_exact_generation, DurableCache, DurableHit, SessionCacheError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheSource {
    CurrentSession,
    Resident,
    Durable,
    Remote,
    Miss,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefixReuseResult {
    pub source: CacheSource,
    pub matched_tokens: usize,
}

/// What the reuse ladder can do for a prompt a remote producer would
/// otherwise prefill.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoteReuseAction {
    /// The ladder holds the prompt minus at most the boundary token the
    /// receiver decodes locally: skip the remote transfer entirely.
    ServeLocal,
    /// The ladder holds a cut-aligned strict prefix: leaving it installed in
    /// the session arms the handoff as a delta (the producer's `prefix_cut`
    /// is validated against the session's held tokens at admission, and a
    /// full producer answer atomically replaces them, so arming never
    /// grafts unverified state).
    ArmDelta,
    /// Nothing the remote handoff could build on: run a full transfer.
    FullTransfer,
}

/// A reuse-ladder offer for a remote-served prompt, resolved and installed
/// by [`PrefixReuse::prepare_remote`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RemoteReuseOffer {
    pub action: RemoteReuseAction,
    pub source: CacheSource,
    pub matched_tokens: usize,
}

/// Classification of a ladder hit for the remote handoff. The receiver
/// holds back the final prompt token, so a hit reaching `prompt - 1` is
/// already full. A delta handoff may begin only on an aligned boundary, so
/// a shorter prefix is useful to the producer only when it lands on one.
fn remote_action(matched: usize, prompt: usize, cut_align: usize) -> RemoteReuseAction {
    if prompt < 2 {
        return RemoteReuseAction::FullTransfer;
    }
    if matched >= prompt - 1 {
        RemoteReuseAction::ServeLocal
    } else if matched > 0 && cut_align > 0 && matched.is_multiple_of(cut_align) {
        RemoteReuseAction::ArmDelta
    } else {
        RemoteReuseAction::FullTransfer
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PrefixReuseError {
    #[error(transparent)]
    Resident(#[from] ResidentError),
    #[error(transparent)]
    Durable(#[from] SessionCacheError),
    #[error(transparent)]
    Remote(#[from] RemotePrefixError),
    #[error(transparent)]
    Engine(#[from] muser_engine::EngineError),
}

/// The route chosen before any session state is installed. Splitting the
/// decision from the install keeps the ladder's policy — which tier may
/// serve, and how deep — testable without engine weights.
#[derive(Debug)]
enum ReusePlan {
    CurrentSession(usize),
    Resident(ResidentHit),
    Durable(DurableHit),
    RemoteOrMiss,
}

pub struct PrefixReuse {
    resident: ResidentRadix,
    durable: Option<DurableCache>,
    remote: Option<RemoteCache>,
}

impl PrefixReuse {
    pub fn new(resident_capacity_bytes: u64) -> Result<Self, ResidentError> {
        Ok(Self {
            resident: ResidentRadix::new(resident_capacity_bytes)?,
            durable: None,
            remote: None,
        })
    }

    pub fn with_durable(mut self, durable: DurableCache) -> Self {
        self.durable = Some(durable);
        self
    }

    pub fn set_durable(&mut self, durable: DurableCache) {
        self.durable = Some(durable);
    }

    pub fn with_remote(mut self, remote: RemoteCache) -> Self {
        self.remote = Some(remote);
        self
    }

    pub fn set_remote(&mut self, remote: RemoteCache) {
        self.remote = Some(remote);
    }

    pub fn has_remote(&self) -> bool {
        self.remote.is_some()
    }

    pub fn has_durable(&self) -> bool {
        self.durable.is_some()
    }

    pub fn resident_mut(&mut self) -> &mut ResidentRadix {
        &mut self.resident
    }

    pub fn publish_resident(&mut self, session: &Session) -> Result<bool, PrefixReuseError> {
        let snapshot = session.export_cache_snapshot()?;
        Ok(self.resident.insert_with_logits(
            self.resident_identity(),
            &snapshot,
            session.cached_logits(),
        )?)
    }

    pub fn publish_durable(
        &self,
        session: &Session,
        publication_generation: u64,
    ) -> Result<bool, PrefixReuseError> {
        let Some(durable) = &self.durable else {
            return Ok(false);
        };
        let mut digest = sha2::Sha256::new();
        use sha2::Digest as _;
        digest.update(b"muser-durable-prompt-v1\0");
        for token in session.token_history() {
            digest.update(token.to_le_bytes());
        }
        let idempotency_key = digest.finalize().into();
        durable.save(session, idempotency_key, publication_generation)?;
        Ok(true)
    }

    pub fn prepare(
        &mut self,
        session: &mut Session,
        tokens: &[u32],
    ) -> Result<PrefixReuseResult, PrefixReuseError> {
        self.prepare_with_cancellation(session, tokens, &RestoreCancellation::default())
    }

    pub fn prepare_with_cancellation(
        &mut self,
        session: &mut Session,
        tokens: &[u32],
        cancellation: &RestoreCancellation,
    ) -> Result<PrefixReuseResult, PrefixReuseError> {
        match self.plan(session.token_history(), tokens)? {
            ReusePlan::CurrentSession(matched_tokens) => Ok(PrefixReuseResult {
                source: CacheSource::CurrentSession,
                matched_tokens,
            }),
            ReusePlan::Resident(hit) => {
                let cut = hit.cut;
                Self::install_resident_hit(session, hit)?;
                Ok(PrefixReuseResult {
                    source: CacheSource::Resident,
                    matched_tokens: cut,
                })
            }
            ReusePlan::Durable(hit) => {
                // New Muse full manifests authenticate the cut-scoped final
                // target distribution alongside KV. This makes a durable exact
                // hit immediately usable for greedy or sampled generation; an
                // ancestor still prefills only its suffix and overwrites logits.
                let cut = hit.cut;
                self.install_durable_hit(session, tokens, hit)?;
                Ok(PrefixReuseResult {
                    source: CacheSource::Durable,
                    matched_tokens: cut,
                })
            }
            ReusePlan::RemoteOrMiss => {
                if let Some(remote) = &self.remote {
                    if let Some(hit) = remote.restore_deepest(session, tokens, cancellation)? {
                        if hit.cut == tokens.len() && session.cached_logits().is_none() {
                            return Err(muser_engine::EngineError::InvalidCacheSnapshot(
                                "remote exact hit omitted final logits".into(),
                            )
                            .into());
                        }
                        return Ok(PrefixReuseResult {
                            source: CacheSource::Remote,
                            matched_tokens: hit.cut,
                        });
                    }
                }
                Ok(PrefixReuseResult {
                    source: CacheSource::Miss,
                    matched_tokens: 0,
                })
            }
        }
    }

    /// Consult the ladder for a prompt the remote producer would otherwise
    /// prefill, installing the winning tier into `session` with the same
    /// generation/rollback discipline as [`Self::prepare`]. `cut_align` is
    /// the handoff's delta-cut alignment: a partial hit that lands off it is
    /// useless to the producer and deliberately left uninstalled, so the
    /// full-transfer path can reset exactly as before. `Ok(None)` is a clean
    /// miss — issue the full remote transfer.
    pub fn prepare_remote(
        &mut self,
        session: &mut Session,
        tokens: &[u32],
        cut_align: usize,
    ) -> Result<Option<RemoteReuseOffer>, PrefixReuseError> {
        let plan = self.plan(session.token_history(), tokens)?;
        let (source, matched) = match &plan {
            ReusePlan::CurrentSession(matched) => (CacheSource::CurrentSession, *matched),
            ReusePlan::Resident(hit) => (CacheSource::Resident, hit.cut),
            ReusePlan::Durable(hit) => (CacheSource::Durable, hit.cut),
            ReusePlan::RemoteOrMiss => {
                // kvpack's own authenticated remote tier still outranks the
                // handoff; a hit there installs like any other tier.
                let Some(remote) = &self.remote else {
                    return Ok(None);
                };
                let Some(hit) =
                    remote.restore_deepest(session, tokens, &RestoreCancellation::default())?
                else {
                    return Ok(None);
                };
                if hit.cut == tokens.len() && session.cached_logits().is_none() {
                    return Err(muser_engine::EngineError::InvalidCacheSnapshot(
                        "remote exact hit omitted final logits".into(),
                    )
                    .into());
                }
                return Ok(Some(RemoteReuseOffer {
                    action: remote_action(hit.cut, tokens.len(), cut_align),
                    source: CacheSource::Remote,
                    matched_tokens: hit.cut,
                }));
            }
        };
        let action = remote_action(matched, tokens.len(), cut_align);
        if action != RemoteReuseAction::FullTransfer {
            match plan {
                ReusePlan::CurrentSession(_) => {}
                ReusePlan::Resident(hit) => Self::install_resident_hit(session, hit)?,
                ReusePlan::Durable(hit) => self.install_durable_hit(session, tokens, hit)?,
                ReusePlan::RemoteOrMiss => unreachable!("remote-or-miss returned above"),
            }
        }
        Ok(Some(RemoteReuseOffer {
            action,
            source,
            matched_tokens: matched,
        }))
    }

    fn install_resident_hit(
        session: &mut Session,
        hit: ResidentHit,
    ) -> Result<(), PrefixReuseError> {
        let snapshot = hit.snapshot.materialize()?;
        if hit.exact {
            install_exact_generation(
                session,
                &snapshot,
                hit.snapshot
                    .last_logits()
                    .expect("exact hit checked in plan"),
            )?;
        } else {
            session.install_cache_snapshot(&snapshot)?;
        }
        Ok(())
    }

    fn install_durable_hit(
        &self,
        session: &mut Session,
        tokens: &[u32],
        hit: DurableHit,
    ) -> Result<(), PrefixReuseError> {
        let durable = self.durable.as_ref().expect("durable plan has a tier");
        let cut = hit.cut;
        durable.restore_manifest(session, tokens, hit, &RestoreCancellation::default())?;
        if cut == tokens.len() && session.cached_logits().is_none() {
            return Err(muser_engine::EngineError::InvalidCacheSnapshot(
                "durable exact hit omitted final logits".into(),
            )
            .into());
        }
        Ok(())
    }

    /// Resolve the reuse route without touching session cache state. The
    /// durable tier is the sole authentication authority: when it is
    /// configured, its deepest authenticated cut caps what the
    /// unauthenticated resident tier may serve, so a resident entry can
    /// never serve a deeper cut than the durable chain has authenticated for
    /// the same identity. Without a durable tier there is nothing to
    /// authenticate against and the resident tier keeps its legacy uncapped
    /// behavior.
    fn plan(&mut self, current: &[u32], tokens: &[u32]) -> Result<ReusePlan, PrefixReuseError> {
        if !current.is_empty()
            && current.len() <= tokens.len()
            && tokens[..current.len()] == *current
        {
            return Ok(ReusePlan::CurrentSession(current.len()));
        }
        // Resolved before the resident lookup so it can cap it; a durable
        // catalog failure fails closed instead of serving unauthenticated
        // resident state.
        let authenticated = match &self.durable {
            Some(durable) => durable.find_deepest(tokens)?,
            None => None,
        };
        let cap = self
            .durable
            .as_ref()
            .map(|_| authenticated.as_ref().map_or(0, |hit| hit.cut));
        let identity = self.resident_identity();
        let resident = match cap {
            Some(cap) => self.resident.lookup_capped(identity, tokens, cap),
            None => self.resident.lookup(identity, tokens),
        };
        if let Some(hit) = resident {
            // Exact-final state is useful only when its target distribution
            // was captured with the KV cut. Older/generic entries remain
            // eligible as aligned ancestors but cannot masquerade as exact.
            // The cap applies to every resident hit, witnessed exact finals
            // included: logits witness that some generation ended here, but
            // no resident provenance class authenticates one, so nothing
            // resident may stand beyond the deepest durable cut.
            if !hit.exact || hit.snapshot.last_logits().is_some() {
                return Ok(ReusePlan::Resident(hit));
            }
        }
        if let Some(hit) = authenticated {
            return Ok(ReusePlan::Durable(hit));
        }
        Ok(ReusePlan::RemoteOrMiss)
    }

    /// Resident entries are scoped to the durable tier's identity when one is
    /// configured; resident-only deployments have no `MuseIdentity` and stay
    /// unscoped, preserving their previous behavior.
    fn resident_identity(&self) -> Option<[u8; 32]> {
        self.durable
            .as_ref()
            .map(|durable| durable.identity().digest())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kvpack::{create_store_key_random, load_store_key, LocalStore, StoreConfig};
    use muser_engine::cache::{CachePlaneSnapshot, PlaneEncoding, SessionCacheSnapshot};
    use muser_engine::config::{
        layer_kind, MuseConfig, MUSE_HEAD_COUNT, MUSE_KV_HEAD_COUNT, MUSE_LAYER_COUNT,
    };
    use std::sync::Arc;

    use crate::layout::MuseIdentity;

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

    fn logits() -> Vec<f32> {
        (0..128).map(|index| index as f32 / 128.0).collect()
    }

    fn durable_fixture() -> (tempfile::TempDir, Arc<LocalStore>) {
        let temp = tempfile::tempdir().unwrap();
        let key_path = temp.path().join("keys/root.key");
        create_store_key_random(&key_path, temp.path()).unwrap();
        let store = Arc::new(
            LocalStore::open(
                StoreConfig {
                    object_root: temp.path().join("objects"),
                    catalog_path: temp.path().join("catalog/catalog.sqlite"),
                    operator_tenant_id: b"muser-reuse-test".to_vec(),
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
        (temp, store)
    }

    #[test]
    fn resident_ancestor_clamps_to_the_deepest_authenticated_cut() {
        let (_temp, store) = durable_fixture();
        let durable = DurableCache::new(Arc::clone(&store), config(), identity());
        let digest = durable.identity().digest();
        // The durable chain authenticates the 256-token cut and no more.
        let cut = snapshot(256);
        durable.save_snapshot(&cut, &logits(), [9; 32], 1).unwrap();
        let mut reuse = PrefixReuse::new(256 * 1024 * 1024)
            .unwrap()
            .with_durable(durable);

        // Resident holds the authenticated cut and a deeper cut nothing
        // authenticated; the ladder must serve the shallower one.
        let deep = snapshot(512);
        assert!(reuse
            .resident_mut()
            .insert_with_logits(Some(digest), &cut, None)
            .unwrap());
        assert!(reuse
            .resident_mut()
            .insert_with_logits(Some(digest), &deep, None)
            .unwrap());
        let query: Vec<u32> = (0..600).collect();
        match reuse.plan(&[], &query).unwrap() {
            ReusePlan::Resident(hit) => assert_eq!((hit.cut, hit.exact), (256, false)),
            other => panic!("expected the clamped resident cut, got {other:?}"),
        }

        // With no resident entry inside the envelope at all, the deepest
        // servable prefix is the durable cut itself.
        let durable = DurableCache::new(store, config(), identity());
        let mut reuse = PrefixReuse::new(256 * 1024 * 1024)
            .unwrap()
            .with_durable(durable);
        assert!(reuse
            .resident_mut()
            .insert_with_logits(Some(digest), &deep, None)
            .unwrap());
        match reuse.plan(&[], &query).unwrap() {
            ReusePlan::Durable(hit) => assert_eq!(hit.cut, 256),
            other => panic!("expected the durable cut, got {other:?}"),
        }
    }

    #[test]
    fn witnessed_exact_hit_does_not_stand_beyond_the_authenticated_cut() {
        let (_temp, store) = durable_fixture();
        let durable = DurableCache::new(store, config(), identity());
        let digest = durable.identity().digest();
        let cut = snapshot(256);
        durable.save_snapshot(&cut, &logits(), [9; 32], 1).unwrap();
        let mut reuse = PrefixReuse::new(256 * 1024 * 1024)
            .unwrap()
            .with_durable(durable);

        // Witnessed final logits do not authenticate the entry: an exact
        // resident cut deeper than the durable chain is still refused.
        let exact = snapshot(600);
        assert!(reuse
            .resident_mut()
            .insert_with_logits(Some(digest), &exact, Some(&logits()))
            .unwrap());
        match reuse.plan(&[], &exact.tokens).unwrap() {
            ReusePlan::Durable(hit) => assert_eq!(hit.cut, 256),
            other => panic!("expected the durable cut, got {other:?}"),
        }
    }

    #[test]
    fn resident_entries_stay_bound_to_the_identity_that_published_them() {
        let (_temp, store) = durable_fixture();
        let durable = DurableCache::new(Arc::clone(&store), config(), identity());
        let digest = durable.identity().digest();
        let deep = snapshot(512);
        durable.save_snapshot(&deep, &logits(), [9; 32], 1).unwrap();
        let mut reuse = PrefixReuse::new(256 * 1024 * 1024)
            .unwrap()
            .with_durable(durable);
        assert!(reuse
            .resident_mut()
            .insert_with_logits(Some(digest), &deep, None)
            .unwrap());
        let query: Vec<u32> = (0..600).collect();
        match reuse.plan(&[], &query).unwrap() {
            ReusePlan::Resident(hit) => assert_eq!((hit.cut, hit.exact), (512, false)),
            other => panic!("expected the resident cut, got {other:?}"),
        }

        // Swapping the durable tier to another identity re-scopes every
        // lookup: the entry above authenticated nothing for this identity
        // and its resident cut is unreachable.
        let mut other_identity = identity();
        other_identity.weight_precision = "q4_k_xl".into();
        reuse.set_durable(DurableCache::new(store, config(), other_identity));
        assert!(matches!(
            reuse.plan(&[], &query).unwrap(),
            ReusePlan::RemoteOrMiss
        ));
    }

    #[test]
    fn without_a_durable_tier_resident_behavior_is_unchanged() {
        let mut reuse = PrefixReuse::new(256 * 1024 * 1024).unwrap();
        let cut = snapshot(256);
        let deep = snapshot(512);
        assert!(reuse
            .resident_mut()
            .insert_with_logits(None, &cut, None)
            .unwrap());
        assert!(reuse
            .resident_mut()
            .insert_with_logits(None, &deep, None)
            .unwrap());
        let query: Vec<u32> = (0..600).collect();
        match reuse.plan(&[], &query).unwrap() {
            ReusePlan::Resident(hit) => assert_eq!((hit.cut, hit.exact), (512, false)),
            other => panic!("expected the deep resident cut, got {other:?}"),
        }
        // A live session prefix still outranks every tier.
        assert!(matches!(
            reuse.plan(&query[..256], &query).unwrap(),
            ReusePlan::CurrentSession(256)
        ));
    }

    #[test]
    fn remote_action_classification() {
        use RemoteReuseAction::*;
        // The receiver holds back the boundary token, so a hit reaching
        // `prompt - 1` is already full.
        assert_eq!(remote_action(511, 512, 256), ServeLocal);
        assert_eq!(remote_action(512, 512, 256), ServeLocal);
        // A shorter prefix arms a delta only on the handoff's cut alignment.
        assert_eq!(remote_action(256, 512, 256), ArmDelta);
        assert_eq!(remote_action(255, 512, 256), FullTransfer);
        assert_eq!(remote_action(0, 512, 256), FullTransfer);
        // Degenerate prompts never take the reuse path.
        assert_eq!(remote_action(0, 0, 256), FullTransfer);
        assert_eq!(remote_action(1, 1, 256), FullTransfer);
        assert_eq!(remote_action(1, 2, 256), ServeLocal);
    }

    #[test]
    fn a_full_resident_hit_skips_the_remote_transfer() {
        let mut reuse = PrefixReuse::new(256 * 1024 * 1024).unwrap();
        let prompt = snapshot(512);
        assert!(reuse
            .resident_mut()
            .insert_with_logits(None, &prompt, Some(&logits()))
            .unwrap());
        let tokens: Vec<u32> = (0..512).collect();
        match reuse.plan(&[], &tokens).unwrap() {
            ReusePlan::Resident(hit) => {
                assert_eq!(
                    remote_action(hit.cut, tokens.len(), 256),
                    RemoteReuseAction::ServeLocal
                )
            }
            other => panic!("expected the exact resident hit, got {other:?}"),
        }
    }

    #[test]
    fn a_radix_identity_mismatch_falls_through_to_a_full_transfer() {
        let (_temp, store) = durable_fixture();
        let durable = DurableCache::new(Arc::clone(&store), config(), identity());
        let digest = durable.identity().digest();
        let cut = snapshot(512);
        durable.save_snapshot(&cut, &logits(), [9; 32], 1).unwrap();
        let mut reuse = PrefixReuse::new(256 * 1024 * 1024)
            .unwrap()
            .with_durable(durable);
        assert!(reuse
            .resident_mut()
            .insert_with_logits(Some(digest), &cut, Some(&logits()))
            .unwrap());
        // Another identity structurally misses the resident entry: the
        // ladder offers nothing and the server runs a full transfer.
        let mut other_identity = identity();
        other_identity.weight_precision = "q4_k_xl".into();
        reuse.set_durable(DurableCache::new(store, config(), other_identity));
        assert!(matches!(
            reuse.plan(&[], &cut.tokens).unwrap(),
            ReusePlan::RemoteOrMiss
        ));
    }

    #[test]
    fn a_partial_resident_hit_arms_a_delta_only_on_the_cut_alignment() {
        let mut reuse = PrefixReuse::new(256 * 1024 * 1024).unwrap();
        let aligned = snapshot(256);
        assert!(reuse
            .resident_mut()
            .insert_with_logits(None, &aligned, None)
            .unwrap());
        let query: Vec<u32> = (0..600).collect();
        match reuse.plan(&[], &query).unwrap() {
            ReusePlan::Resident(hit) => {
                assert_eq!(
                    remote_action(hit.cut, query.len(), 256),
                    RemoteReuseAction::ArmDelta
                )
            }
            other => panic!("expected the aligned resident cut, got {other:?}"),
        }

        // The radix retains unaligned cuts for exact-hit-only lookup, so an
        // unaligned partial can only arrive from the live session; it must
        // still refuse to arm a delta.
        let mut reuse = PrefixReuse::new(256 * 1024 * 1024).unwrap();
        match reuse.plan(&query[..300], &query).unwrap() {
            ReusePlan::CurrentSession(matched) => assert_eq!(
                remote_action(matched, query.len(), 256),
                RemoteReuseAction::FullTransfer
            ),
            other => panic!("expected the live session prefix, got {other:?}"),
        }
    }

    #[test]
    fn exact_final_hit_still_requires_witnessed_logits() {
        let mut reuse = PrefixReuse::new(256 * 1024 * 1024).unwrap();
        let final_cut = snapshot(513);
        assert!(reuse.resident_mut().insert(None, &final_cut).unwrap());
        assert!(matches!(
            reuse.plan(&[], &final_cut.tokens).unwrap(),
            ReusePlan::RemoteOrMiss
        ));
        assert!(reuse
            .resident_mut()
            .insert_with_logits(None, &final_cut, Some(&logits()))
            .unwrap());
        match reuse.plan(&[], &final_cut.tokens).unwrap() {
            ReusePlan::Resident(hit) => assert_eq!((hit.cut, hit.exact), (513, true)),
            other => panic!("expected the witnessed exact cut, got {other:?}"),
        }
    }
}
