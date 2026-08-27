//! Authenticated remote durable-prefix acquisition.
//!
//! This is deliberately separate from GX10 producer prefill. A remote prefix
//! authority resolves the same kvpack `PrefixNode` chain as a local durable
//! store, then streams the authenticated manifest object and its ordered
//! content-addressed chunks into a private local import. Only after kvpack has
//! authenticated and atomically published the complete import do we build a
//! detached Muse restore and replace the live session generation.

use std::sync::Arc;

use kvpack::{
    AuthenticatedPublicationSource, LocalStore, RestoreCancellation, StoreError, UploadState,
};
use kvpack_core::{representation_family_id, semantic_model_id, Id32, PrefixNode};
use muser_engine::Session;
use sha2::{Digest, Sha256};

use crate::layout::descriptor;
use crate::session::{DurableCache, DurableHit, SessionCacheError};

/// Exact request sent to a remote durable-prefix authority. Semantic and
/// representation identities are derived locally; a remote peer cannot
/// substitute its own identity declarations.
#[derive(Debug, Clone)]
pub struct RemotePrefixRequest {
    pub semantic_id: Id32,
    pub family_id: Id32,
    pub prefix_nodes: Vec<PrefixNode>,
    pub requested_tokens: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RemotePrefixHit {
    pub cut: u64,
    pub prefix_id: Id32,
    pub manifest_id: Id32,
    pub recompute_tokens: u64,
    pub publication_generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RemoteFetchReceipt {
    pub cut: usize,
    pub manifest_id: Id32,
    pub transferred_chunks: u64,
    pub transferred_bytes: u64,
    pub resumed_at_chunk: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RemoteChunk {
    pub ordinal: u64,
    pub object_key: Id32,
    pub object_bytes: u32,
}

/// One lease-protected, ordered artifact stream. Network implementations map
/// this to the kvpack gateway manifest/chunk endpoints; the loopback adapter
/// below uses the exact same publication-source contract without a socket.
pub trait RemoteArtifactReader {
    fn manifest_id(&self) -> Id32;
    fn publication_generation(&self) -> u64;
    fn expected_bytes(&self) -> u64;
    fn manifest_object(&self) -> &[u8];
    fn chunk_count(&self) -> u64;
    fn read_chunk(&mut self, ordinal: u64) -> Result<(RemoteChunk, Vec<u8>), RemotePrefixError>;
}

/// Transport boundary for a remote durable cache. Implementations must bind
/// resolve/open to an authenticated peer. The receiver still authenticates
/// every returned kvpack object, so this interface is discovery/transport,
/// never a second cache identity authority.
pub trait RemotePrefixAuthority: Send + Sync {
    fn resolve(
        &self,
        request: &RemotePrefixRequest,
    ) -> Result<Option<RemotePrefixHit>, RemotePrefixError>;
    fn open(
        &self,
        hit: &RemotePrefixHit,
    ) -> Result<Box<dyn RemoteArtifactReader>, RemotePrefixError>;
}

#[derive(Debug, thiserror::Error)]
pub enum RemotePrefixError {
    #[error(transparent)]
    Kvpack(#[from] StoreError),
    #[error(transparent)]
    Session(#[from] SessionCacheError),
    #[error("remote prefix authority: {0}")]
    Authority(String),
    #[error("remote prefix response is not bound to the exact request: {0}")]
    Binding(&'static str),
}

/// A remote-prefix client with a private local authenticated import store.
/// Imported bytes never become engine-visible until `restore_deepest` has
/// completed all identity, manifest, range, chunk and cancellation checks.
pub struct RemoteCache {
    authority: Arc<dyn RemotePrefixAuthority>,
    staging: DurableCache,
}

impl RemoteCache {
    pub fn new(authority: Arc<dyn RemotePrefixAuthority>, staging: DurableCache) -> Self {
        Self { authority, staging }
    }

    pub fn staging(&self) -> &DurableCache {
        &self.staging
    }

    /// Resolve and import the deepest exact remote prefix without touching an
    /// engine session. This split is also the offline qualifier hook.
    pub fn fetch_deepest(
        &self,
        tokens: &[u32],
        cancellation: &RestoreCancellation,
    ) -> Result<Option<RemoteFetchReceipt>, RemotePrefixError> {
        if tokens.is_empty() {
            return Ok(None);
        }
        if cancellation.is_cancelled() {
            return Err(StoreError::Cancelled.into());
        }
        let requested_count = u32::try_from(tokens.len())
            .map_err(|_| SessionCacheError::Snapshot("remote request exceeds u32".into()))?;
        let runtime = descriptor(
            self.staging.config(),
            self.staging.identity(),
            requested_count,
        )
        .map_err(SessionCacheError::from)?;
        let (_, nodes) = self.staging.store().derive_input_cut(
            &runtime.semantic_model,
            &runtime.family,
            tokens,
            &[],
        )?;
        let semantic_id = semantic_model_id(&runtime.semantic_model);
        let family_id = representation_family_id(&runtime.family).map_err(StoreError::from)?;
        let request = RemotePrefixRequest {
            semantic_id,
            family_id,
            prefix_nodes: nodes,
            requested_tokens: tokens.len() as u64,
        };
        let Some(hit) = self.authority.resolve(&request)? else {
            return Ok(None);
        };
        validate_hit(&request, &hit)?;
        if cancellation.is_cancelled() {
            return Err(StoreError::Cancelled.into());
        }
        let mut source = self.authority.open(&hit)?;
        if source.manifest_id() != hit.manifest_id
            || source.publication_generation() != hit.publication_generation
            || source.expected_bytes() == 0
            || source.chunk_count() == 0
        {
            return Err(RemotePrefixError::Binding(
                "artifact stream metadata differs from resolved hit",
            ));
        }
        let idempotency = remote_import_idempotency(&hit);
        let context = kvpack_core::ValidationContext::default();
        let initial = self.staging.store().begin_authenticated_import(
            &idempotency,
            &hit.manifest_id,
            source.expected_bytes(),
            hit.publication_generation,
        )?;
        if initial.manifest_id != hit.manifest_id
            || initial.publication_generation != hit.publication_generation
        {
            return Err(RemotePrefixError::Binding(
                "local import reservation changed remote identity",
            ));
        }
        let resumed_at_chunk = initial.next_chunk_ordinal;
        if initial.state != UploadState::Published {
            if cancellation.is_cancelled() {
                return Err(StoreError::Cancelled.into());
            }
            if let Err(error) = self.staging.store().stage_authenticated_manifest(
                &idempotency,
                source.manifest_object(),
                &context,
            ) {
                let _ = self
                    .staging
                    .store()
                    .quarantine_authenticated_import(&idempotency);
                return Err(error.into());
            }
            let mut ordinal = initial.next_chunk_ordinal;
            while ordinal < source.chunk_count() {
                if cancellation.is_cancelled() {
                    // Retain a resumable, engine-invisible partial import.
                    return Err(StoreError::Cancelled.into());
                }
                let (chunk, bytes) = source.read_chunk(ordinal)?;
                if chunk.ordinal != ordinal
                    || chunk.object_key == [0; 32]
                    || bytes.len() != chunk.object_bytes as usize
                {
                    let _ = self
                        .staging
                        .store()
                        .quarantine_authenticated_import(&idempotency);
                    return Err(RemotePrefixError::Binding(
                        "remote chunk ordinal, identity, or byte count changed",
                    ));
                }
                if let Err(error) = self.staging.store().put_authenticated_import_chunk(
                    &idempotency,
                    ordinal,
                    &chunk.object_key,
                    &bytes,
                    &context,
                ) {
                    let _ = self
                        .staging
                        .store()
                        .quarantine_authenticated_import(&idempotency);
                    return Err(error.into());
                }
                ordinal += 1;
            }
            if cancellation.is_cancelled() {
                return Err(StoreError::Cancelled.into());
            }
            self.staging
                .store()
                .seal_authenticated_import(&idempotency, &context)?;
            let published = self
                .staging
                .store()
                .commit_authenticated_import(&idempotency, &context)?;
            if published.manifest_id != hit.manifest_id {
                return Err(RemotePrefixError::Binding(
                    "published import changed remote manifest identity",
                ));
            }
        }
        // Re-resolve in the receiver's authenticated PrefixNode catalog. This
        // prevents a transport response from becoming an alternate authority.
        let local = self
            .staging
            .find_deepest(tokens)?
            .ok_or(RemotePrefixError::Binding(
                "imported manifest is not discoverable through exact PrefixNode authority",
            ))?;
        if local.cut != hit.cut as usize || local.manifest_id != hit.manifest_id {
            return Err(RemotePrefixError::Binding(
                "imported local prefix resolution differs from remote hit",
            ));
        }
        Ok(Some(RemoteFetchReceipt {
            cut: local.cut,
            manifest_id: local.manifest_id,
            transferred_chunks: source.chunk_count().saturating_sub(resumed_at_chunk),
            transferred_bytes: source.expected_bytes(),
            resumed_at_chunk,
        }))
    }

    /// Fetch, fully authenticate and atomically install a remote exact state.
    /// The caller prefills only `tokens[receipt.cut..]` afterward.
    pub fn restore_deepest(
        &self,
        session: &mut Session,
        tokens: &[u32],
        cancellation: &RestoreCancellation,
    ) -> Result<Option<RemoteFetchReceipt>, RemotePrefixError> {
        let Some(receipt) = self.fetch_deepest(tokens, cancellation)? else {
            return Ok(None);
        };
        if cancellation.is_cancelled() {
            return Err(StoreError::Cancelled.into());
        }
        self.staging.restore_manifest(
            session,
            tokens,
            DurableHit {
                cut: receipt.cut,
                manifest_id: receipt.manifest_id,
            },
            cancellation,
        )?;
        Ok(Some(receipt))
    }
}

fn validate_hit(
    request: &RemotePrefixRequest,
    hit: &RemotePrefixHit,
) -> Result<(), RemotePrefixError> {
    if hit.cut == 0
        || hit.cut > request.requested_tokens
        || hit.recompute_tokens != request.requested_tokens - hit.cut
        || hit.manifest_id == [0; 32]
        || hit.publication_generation == 0
    {
        return Err(RemotePrefixError::Binding(
            "remote cut, suffix, manifest, or generation is invalid",
        ));
    }
    let node = request
        .prefix_nodes
        .iter()
        .find(|node| node.token_count == hit.cut)
        .ok_or(RemotePrefixError::Binding(
            "remote cut is absent from the exact request chain",
        ))?;
    if node.id != hit.prefix_id || (hit.cut != request.requested_tokens && !node.reusable) {
        return Err(RemotePrefixError::Binding(
            "remote cut is not the exact/reusable request node",
        ));
    }
    Ok(())
}

fn remote_import_idempotency(hit: &RemotePrefixHit) -> Id32 {
    let mut digest = Sha256::new();
    digest.update(b"muser/remote-durable-import/v1\0");
    digest.update(hit.manifest_id);
    digest.update(hit.prefix_id);
    digest.update(hit.cut.to_le_bytes());
    digest.update(hit.publication_generation.to_le_bytes());
    digest.finalize().into()
}

/// In-process adapter used by the fail-closed loopback qualifier. Production
/// network code implements `RemotePrefixAuthority` over the authenticated
/// kvpack gateway/Handoff transport and returns the same ordered objects.
pub struct LoopbackRemoteStore {
    store: Arc<LocalStore>,
    publication_generation: u64,
}

impl LoopbackRemoteStore {
    pub fn new(store: Arc<LocalStore>, publication_generation: u64) -> Result<Self, StoreError> {
        if publication_generation == 0 {
            return Err(StoreError::Expectation(
                "remote publication generation is zero",
            ));
        }
        Ok(Self {
            store,
            publication_generation,
        })
    }
}

impl RemotePrefixAuthority for LoopbackRemoteStore {
    fn resolve(
        &self,
        request: &RemotePrefixRequest,
    ) -> Result<Option<RemotePrefixHit>, RemotePrefixError> {
        let Some(hit) = self.store.resolve_prefix(
            &request.prefix_nodes,
            &request.semantic_id,
            &request.family_id,
            64,
        )?
        else {
            return Ok(None);
        };
        let node = request
            .prefix_nodes
            .iter()
            .find(|node| node.token_count == hit.token_count)
            .ok_or(RemotePrefixError::Binding(
                "source catalog returned a cut outside the request chain",
            ))?;
        Ok(Some(RemotePrefixHit {
            cut: hit.token_count,
            prefix_id: node.id,
            manifest_id: hit.manifest_id,
            recompute_tokens: hit.recompute_tokens,
            publication_generation: self.publication_generation,
        }))
    }

    fn open(
        &self,
        hit: &RemotePrefixHit,
    ) -> Result<Box<dyn RemoteArtifactReader>, RemotePrefixError> {
        let source = self.store.authenticated_publication_source(
            &hit.manifest_id,
            &kvpack_core::ValidationContext::default(),
        )?;
        Ok(Box::new(LoopbackArtifactReader {
            source,
            publication_generation: hit.publication_generation,
        }))
    }
}

struct LoopbackArtifactReader {
    source: AuthenticatedPublicationSource,
    publication_generation: u64,
}

impl RemoteArtifactReader for LoopbackArtifactReader {
    fn manifest_id(&self) -> Id32 {
        self.source.manifest_id()
    }

    fn publication_generation(&self) -> u64 {
        self.publication_generation
    }

    fn expected_bytes(&self) -> u64 {
        self.source.expected_bytes()
    }

    fn manifest_object(&self) -> &[u8] {
        self.source.manifest_object()
    }

    fn chunk_count(&self) -> u64 {
        self.source.chunk_count()
    }

    fn read_chunk(&mut self, ordinal: u64) -> Result<(RemoteChunk, Vec<u8>), RemotePrefixError> {
        let (chunk, bytes) = self.source.read_chunk_object(ordinal)?;
        Ok((
            RemoteChunk {
                ordinal: chunk.ordinal(),
                object_key: chunk.object_key(),
                object_bytes: chunk.object_bytes(),
            },
            bytes,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kvpack::{create_store_key_random, load_store_key, StoreConfig};
    use muser_engine::cache::{CachePlaneSnapshot, PlaneEncoding, SessionCacheSnapshot};
    use muser_engine::config::{
        layer_kind, MuseConfig, MUSE_HEAD_COUNT, MUSE_KV_HEAD_COUNT, MUSE_LAYER_COUNT,
    };

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
            weight_precision: "q4_k_xl".into(),
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

    struct Stores {
        _temp: tempfile::TempDir,
        source: Arc<LocalStore>,
        destination: Arc<LocalStore>,
    }

    fn stores(shared_key: bool, label: &str) -> Stores {
        let temp = tempfile::tempdir().unwrap();
        let source_root = temp.path().join("source");
        let destination_root = temp.path().join("destination");
        let source_key_path = source_root.join("keys/root.key");
        create_store_key_random(&source_key_path, &source_root).unwrap();
        let destination_key_path = destination_root.join("keys/root.key");
        if !shared_key {
            create_store_key_random(&destination_key_path, &destination_root).unwrap();
        }
        let make_config = |root: &std::path::Path| StoreConfig {
            object_root: root.join("objects"),
            catalog_path: root.join("catalog/catalog.sqlite"),
            operator_tenant_id: format!("muser-remote-{label}").into_bytes(),
            key_epoch: 1,
            minimum_readable_key_epoch: 1,
            catalog_epoch: 1,
            quota_bytes: 512 * 1024 * 1024,
            staging_quota_bytes: 512 * 1024 * 1024,
            endurance_bytes_per_five_minutes: 512 * 1024 * 1024,
        };
        let source = Arc::new(
            LocalStore::open(
                make_config(&source_root),
                load_store_key(&source_key_path, &source_root).unwrap(),
            )
            .unwrap(),
        );
        let destination_key = if shared_key {
            load_store_key(&source_key_path, &source_root).unwrap()
        } else {
            load_store_key(&destination_key_path, &destination_root).unwrap()
        };
        let destination =
            Arc::new(LocalStore::open(make_config(&destination_root), destination_key).unwrap());
        Stores {
            _temp: temp,
            source,
            destination,
        }
    }

    fn publish(store: Arc<LocalStore>, cut: usize) -> DurableCache {
        let cache = DurableCache::new(store, config(), identity());
        let logits = (0..128)
            .map(|index| index as f32 / 128.0)
            .collect::<Vec<_>>();
        cache
            .save_snapshot(&snapshot(cut), &logits, [9; 32], 1)
            .unwrap();
        cache
    }

    fn remote_cache(
        source: Arc<dyn RemotePrefixAuthority>,
        destination: Arc<LocalStore>,
    ) -> RemoteCache {
        RemoteCache::new(source, DurableCache::new(destination, config(), identity()))
    }

    #[test]
    fn loopback_imports_exact_and_all_required_suffix_boundaries() {
        let stores = stores(true, "matrix");
        publish(Arc::clone(&stores.source), 256);
        let source: Arc<dyn RemotePrefixAuthority> =
            Arc::new(LoopbackRemoteStore::new(Arc::clone(&stores.source), 1).unwrap());
        let cache = remote_cache(source, Arc::clone(&stores.destination));
        let cancellation = RestoreCancellation::default();

        let exact_tokens = (0..256u32).collect::<Vec<_>>();
        let exact = cache
            .fetch_deepest(&exact_tokens, &cancellation)
            .unwrap()
            .unwrap();
        assert_eq!(exact.cut, 256);
        assert!(exact.transferred_chunks > 0);
        assert_eq!(
            cache
                .staging()
                .find_deepest(&exact_tokens)
                .unwrap()
                .unwrap()
                .cut,
            256
        );

        for suffix in [1usize, 255, 256, 257, 2_047] {
            let tokens = (0..(256 + suffix) as u32).collect::<Vec<_>>();
            let hit = cache
                .fetch_deepest(&tokens, &cancellation)
                .unwrap()
                .unwrap();
            assert_eq!(hit.cut, 256, "suffix {suffix}");
            // The immutable artifact was already imported by the exact hit;
            // every later remote resolution becomes an authenticated replay.
            assert_eq!(hit.transferred_chunks, 0, "suffix {suffix}");
        }
    }

    #[test]
    fn wrong_identity_key_corruption_and_cancellation_fail_before_publication() {
        // Wrong semantic/model identity is an exact miss, not a plausible hit.
        let wrong_identity_stores = stores(true, "wrong-identity");
        publish(Arc::clone(&wrong_identity_stores.source), 256);
        let mut incompatible = identity();
        incompatible.model_sha256[0] ^= 1;
        let cache = RemoteCache::new(
            Arc::new(
                LoopbackRemoteStore::new(Arc::clone(&wrong_identity_stores.source), 1).unwrap(),
            ),
            DurableCache::new(
                Arc::clone(&wrong_identity_stores.destination),
                config(),
                incompatible,
            ),
        );
        let tokens = (0..256u32).collect::<Vec<_>>();
        assert!(cache
            .fetch_deepest(&tokens, &RestoreCancellation::default())
            .unwrap()
            .is_none());
        assert_eq!(
            wrong_identity_stores.destination.stat().unwrap().manifests,
            0
        );

        // A receiver with another tenant key rejects the authenticated
        // manifest before catalog publication.
        let wrong_key_stores = stores(false, "wrong-key");
        publish(Arc::clone(&wrong_key_stores.source), 256);
        let cache = remote_cache(
            Arc::new(LoopbackRemoteStore::new(Arc::clone(&wrong_key_stores.source), 1).unwrap()),
            Arc::clone(&wrong_key_stores.destination),
        );
        assert!(!matches!(
            cache.fetch_deepest(&tokens, &RestoreCancellation::default()),
            Ok(Some(_))
        ));
        assert_eq!(wrong_key_stores.destination.stat().unwrap().manifests, 0);

        // Cancellation is checked before resolve/open/import. The destination
        // therefore remains the previous (empty) generation/catalog.
        let cancelled_stores = stores(true, "cancelled");
        publish(Arc::clone(&cancelled_stores.source), 256);
        let cache = remote_cache(
            Arc::new(LoopbackRemoteStore::new(Arc::clone(&cancelled_stores.source), 1).unwrap()),
            Arc::clone(&cancelled_stores.destination),
        );
        let cancellation = RestoreCancellation::default();
        cancellation.cancel();
        assert!(matches!(
            cache.fetch_deepest(&tokens, &cancellation),
            Err(RemotePrefixError::Kvpack(StoreError::Cancelled))
        ));
        assert_eq!(cancelled_stores.destination.stat().unwrap().manifests, 0);

        // A changed manifest byte is quarantined and cannot become a prefix
        // authority or reach the engine commit path.
        let corrupted_stores = stores(true, "corrupted");
        publish(Arc::clone(&corrupted_stores.source), 256);
        let base: Arc<dyn RemotePrefixAuthority> =
            Arc::new(LoopbackRemoteStore::new(Arc::clone(&corrupted_stores.source), 1).unwrap());
        let cache = remote_cache(
            Arc::new(TamperingAuthority { inner: base }),
            Arc::clone(&corrupted_stores.destination),
        );
        assert!(cache
            .fetch_deepest(&tokens, &RestoreCancellation::default())
            .is_err());
        assert_eq!(corrupted_stores.destination.stat().unwrap().manifests, 0);
        assert!(cache.staging().find_deepest(&tokens).unwrap().is_none());
    }

    struct TamperingAuthority {
        inner: Arc<dyn RemotePrefixAuthority>,
    }

    impl RemotePrefixAuthority for TamperingAuthority {
        fn resolve(
            &self,
            request: &RemotePrefixRequest,
        ) -> Result<Option<RemotePrefixHit>, RemotePrefixError> {
            self.inner.resolve(request)
        }

        fn open(
            &self,
            hit: &RemotePrefixHit,
        ) -> Result<Box<dyn RemoteArtifactReader>, RemotePrefixError> {
            let inner = self.inner.open(hit)?;
            let mut manifest = inner.manifest_object().to_vec();
            let offset = manifest.len() / 2;
            manifest[offset] ^= 1;
            Ok(Box::new(TamperingReader { inner, manifest }))
        }
    }

    struct TamperingReader {
        inner: Box<dyn RemoteArtifactReader>,
        manifest: Vec<u8>,
    }

    impl RemoteArtifactReader for TamperingReader {
        fn manifest_id(&self) -> Id32 {
            self.inner.manifest_id()
        }
        fn publication_generation(&self) -> u64 {
            self.inner.publication_generation()
        }
        fn expected_bytes(&self) -> u64 {
            self.inner.expected_bytes()
        }
        fn manifest_object(&self) -> &[u8] {
            &self.manifest
        }
        fn chunk_count(&self) -> u64 {
            self.inner.chunk_count()
        }
        fn read_chunk(
            &mut self,
            ordinal: u64,
        ) -> Result<(RemoteChunk, Vec<u8>), RemotePrefixError> {
            self.inner.read_chunk(ordinal)
        }
    }
}
