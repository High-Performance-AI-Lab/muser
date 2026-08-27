//! Stable standalone engine API.

use std::path::PathBuf;
use std::sync::Arc;

use crate::config::{MuseConfig, MuseConfigError};
use crate::loader;
use crate::reference::MuseModel;
use crate::tokenizer::BpeTokenizer;
use crate::weights::MuseWeights;

#[cfg(all(target_os = "macos", feature = "metal"))]
type DFlashPromptPrefill = (
    Vec<f32>,
    Vec<f32>,
    crate::metal::dflash::DFlashPromptPipelineStats,
);

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_path: PathBuf,
}

impl ModelConfig {
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SessionConfig {
    pub max_context: usize,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self { max_context: 4_096 }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error(transparent)]
    Model(#[from] MuseConfigError),
    #[error("session max_context must be in 1..={model_limit}, got {requested}")]
    InvalidMaxContext {
        requested: usize,
        model_limit: usize,
    },
    #[error("token {token} is outside vocabulary 0..{vocab_size}")]
    InvalidToken { token: u32, vocab_size: usize },
    #[error("request would extend context to {requested}, beyond session limit {limit}")]
    ContextOverflow { requested: usize, limit: usize },
    #[error("prefill batch is empty")]
    EmptyPrefill,
    #[error("embedding {index} has dimension {actual}, expected {expected}")]
    InvalidEmbedding {
        index: usize,
        actual: usize,
        expected: usize,
    },
    #[error("embedding position witnesses are missing or non-canonical")]
    InvalidEmbeddingWitnesses,
    #[error("speculative verification is not available until the DFlash stage")]
    SpeculativeVerificationUnavailable,
    #[error("speculative verification requires a completed prefill or decode")]
    MissingVerificationState,
    #[error("invalid cache snapshot: {0}")]
    InvalidCacheSnapshot(String),
    #[error("speculative commit length {accepted} exceeds the {evaluated} evaluated candidates")]
    InvalidSpeculativeCommit { accepted: usize, evaluated: usize },
    #[error(
        "restored logits have length {actual}, expected {expected}, or contain nonfinite values"
    )]
    InvalidRestoredLogits { actual: usize, expected: usize },
    #[error("model produced nonfinite logits")]
    NonfiniteLogits,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[error(transparent)]
    Metal(#[from] crate::decode::MetalModelError),
}

/// Detached remote-prefill KV generation. Authenticated tiles are written as
/// they arrive; the live session is unchanged until [`Session::commit_remote_kv_install`].
pub struct RemoteKvInstall {
    tokens: Arc<[u32]>,
    inner: RemoteKvInner,
}

/// Fully validated target handle whose final live swap is infallible.
pub struct PreparedRemoteKvInstall {
    tokens: Arc<[u32]>,
    inner: RemoteKvInner,
}

enum RemoteKvInner {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    Metal(crate::decode::MetalRemoteKvInstall),
    Unsupported,
}

impl RemoteKvInstall {
    /// Scatter one verified f16_le K or V tile into the detached generation.
    pub fn write_f16_tile(
        &mut self,
        layer: usize,
        is_key: bool,
        logical_start: u64,
        logical_count: u64,
        bytes: &[u8],
    ) -> Result<(), EngineError> {
        match &mut self.inner {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            RemoteKvInner::Metal(install) => install
                .write_f16_tile(layer, is_key, logical_start, logical_count, bytes)
                .map_err(EngineError::from),
            RemoteKvInner::Unsupported => {
                let _ = (layer, is_key, logical_start, logical_count, bytes);
                Err(EngineError::InvalidCacheSnapshot(
                    "remote KV tile install requires the Metal session".into(),
                ))
            }
        }
    }

    pub fn validate_complete(&self) -> Result<(), EngineError> {
        match &self.inner {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            RemoteKvInner::Metal(install) => install.validate_complete().map_err(EngineError::from),
            RemoteKvInner::Unsupported => Err(EngineError::InvalidCacheSnapshot(
                "remote KV tile install requires the Metal session".into(),
            )),
        }
    }
}

/// One ring plane's origin pair: the first held logical row and its physical
/// slot. `physical == logical % capacity` is the sequentially-built-ring
/// invariant (ENG-003); the prefix copy checks it instead of assuming it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RingOrigin {
    pub logical: usize,
    pub physical: usize,
}

/// Span of the held prefix a delta install copies into the detached
/// generation, plus the logical position the first suffix tile must land on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DeltaPrefixSpan {
    /// Logical rows `[copy_start, copy_end)` are copied from the live plane;
    /// empty when the sliding window slid past the cut and the whole window
    /// re-arrives as tiles.
    pub copy_start: usize,
    pub copy_end: usize,
    /// Write cursor the install resumes at: the first suffix tile position.
    pub resume: usize,
}

/// Per-plane delta plan. `origin` is the detached plane's logical origin,
/// `position` the full prompt length, and `[live_origin, live_end)` the span
/// the live ring still holds. Fails closed on a cut that leaves no suffix,
/// names tokens the session does not hold, or asks a wrapped ring for rows
/// it has already overwritten.
pub(crate) fn delta_prefix_span(
    origin: usize,
    position: usize,
    prefix_cut: usize,
    live_origin: usize,
    live_end: usize,
) -> Result<DeltaPrefixSpan, String> {
    if prefix_cut == 0 || prefix_cut >= position {
        return Err("delta prefix cut must leave a nonempty suffix".into());
    }
    if prefix_cut > live_end {
        return Err("the live session does not hold the named delta prefix".into());
    }
    let (copy_start, copy_end) = if prefix_cut > origin {
        if live_origin > origin {
            return Err("the live ring no longer holds the delta window origin".into());
        }
        (origin, prefix_cut)
    } else {
        (origin, origin)
    };
    Ok(DeltaPrefixSpan {
        copy_start,
        copy_end,
        resume: prefix_cut.max(origin),
    })
}

/// Copy logical rows `[logical_start, logical_end)` between two ring planes
/// that share one capacity and row layout. Each side's physical row comes
/// from its own origin, so the copy stays exact whether or not the live ring
/// has wrapped; any plane whose origin pair violates the sequential build
/// invariant fails closed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn copy_ring_prefix_rows(
    source: &[u16],
    destination: &mut [u16],
    capacity: usize,
    kv_dim: usize,
    head_dim: usize,
    head_major: bool,
    source_origin: RingOrigin,
    destination_origin: RingOrigin,
    logical_start: usize,
    logical_end: usize,
) -> Result<(), String> {
    if capacity == 0 || head_dim == 0 || kv_dim == 0 || !kv_dim.is_multiple_of(head_dim) {
        return Err("ring plane row geometry is degenerate".into());
    }
    let elements = capacity
        .checked_mul(kv_dim)
        .ok_or_else(|| "ring plane size overflow".to_string())?;
    if source.len() != elements || destination.len() != elements {
        return Err("ring plane storage disagrees with its capacity".into());
    }
    if source_origin.physical != source_origin.logical % capacity
        || destination_origin.physical != destination_origin.logical % capacity
    {
        return Err("ring plane origin violates the sequential build invariant".into());
    }
    let source_held_end = source_origin
        .logical
        .checked_add(capacity)
        .ok_or_else(|| "ring plane logical range overflow".to_string())?;
    let destination_held_end = destination_origin
        .logical
        .checked_add(capacity)
        .ok_or_else(|| "ring plane logical range overflow".to_string())?;
    if logical_start > logical_end
        || logical_end - logical_start > capacity
        || logical_start < source_origin.logical
        || logical_start < destination_origin.logical
        || logical_end > source_held_end
        || logical_end > destination_held_end
    {
        return Err("ring prefix copy range is outside the held rows".into());
    }
    for logical in logical_start..logical_end {
        let source_physical =
            (source_origin.physical + (logical - source_origin.logical)) % capacity;
        let destination_physical =
            (destination_origin.physical + (logical - destination_origin.logical)) % capacity;
        if head_major {
            for head in 0..kv_dim / head_dim {
                let source_offset = (head * capacity + source_physical) * head_dim;
                let destination_offset = (head * capacity + destination_physical) * head_dim;
                destination[destination_offset..destination_offset + head_dim]
                    .copy_from_slice(&source[source_offset..source_offset + head_dim]);
            }
        } else {
            let source_offset = source_physical * kv_dim;
            let destination_offset = destination_physical * kv_dim;
            destination[destination_offset..destination_offset + kv_dim]
                .copy_from_slice(&source[source_offset..source_offset + kv_dim]);
        }
    }
    Ok(())
}

/// Immutable Muse model weights, tokenizer, and model metadata.
///
/// Create independent CPU sessions with [`Model::new_session`]. On macOS,
/// the opt-in `metal` feature additionally exposes
/// `Model::new_metal_session` and `Model::new_metal_sessions`.
pub struct Model {
    config: MuseConfig,
    weights: MuseWeights,
    tokenizer: Arc<BpeTokenizer>,
    tokenizer_metadata_sha256: [u8; 32],
    chat_template: Arc<str>,
    chat_template_sha256: [u8; 32],
    bos_token_id: Option<u32>,
    add_bos_token: bool,
    weight_precision: String,
}

impl Model {
    pub fn load(config: ModelConfig) -> Result<Self, EngineError> {
        let loaded = loader::load_components(&config.model_path)?;
        Ok(Self {
            config: loaded.config,
            weights: loaded.weights,
            tokenizer: Arc::new(loaded.tokenizer),
            tokenizer_metadata_sha256: loaded.tokenizer_metadata_sha256,
            chat_template: Arc::from(loaded.chat_template),
            chat_template_sha256: loaded.chat_template_sha256,
            bos_token_id: loaded.bos_token_id,
            add_bos_token: loaded.add_bos_token,
            weight_precision: loaded.weight_precision,
        })
    }

    pub fn config(&self) -> &MuseConfig {
        &self.config
    }

    pub fn tokenizer_metadata_sha256(&self) -> [u8; 32] {
        self.tokenizer_metadata_sha256
    }

    pub fn chat_template(&self) -> &str {
        &self.chat_template
    }

    pub fn chat_template_sha256(&self) -> [u8; 32] {
        self.chat_template_sha256
    }

    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    pub fn adds_bos_token(&self) -> bool {
        self.add_bos_token
    }

    pub fn weight_precision(&self) -> &str {
        &self.weight_precision
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text)
    }

    /// Encode with explicit control over special-token parsing. Template text
    /// keeps `parse_special = true`; untrusted request content passes `false`
    /// so a spelled-out control marker cannot inject a control token.
    pub fn encode_with_options(&self, text: &str, parse_special: bool) -> Vec<u32> {
        self.tokenizer.encode_with_options(text, parse_special)
    }

    pub fn decode_tokens(&self, tokens: &[u32]) -> String {
        self.tokenizer.decode_all(tokens)
    }

    pub fn token_bytes(&self, token: u32) -> &[u8] {
        self.tokenizer.get_token_bytes(token)
    }

    /// Create a UTF-8-safe incremental decoder for token streaming. The
    /// server owns this alongside (not inside) a mutable inference session,
    /// so split multibyte codepoints are never emitted as replacement text.
    pub fn streaming_detokenizer(&self) -> crate::tokenizer::StreamingDetokenizer<'_> {
        self.tokenizer.streaming_detokenizer()
    }

    pub fn new_session(&self, config: SessionConfig) -> Result<Session, EngineError> {
        self.validate_session_config(config)?;
        Ok(Session {
            backend: SessionBackend::Cpu(Box::new(MuseModel::new(
                self.config.clone(),
                self.weights.clone(),
                config.max_context,
            ))),
            tokenizer: Arc::clone(&self.tokenizer),
            max_context: config.max_context,
            token_history: Vec::new(),
            last_logits: None,
        })
    }

    /// Construct the standalone Metal backend explicitly. The CPU oracle
    /// remains available through `new_session` for correctness comparisons.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn new_metal_session(&self, config: SessionConfig) -> Result<Session, EngineError> {
        self.validate_session_config(config)?;
        Ok(Session {
            backend: SessionBackend::Metal(Box::new(crate::decode::MetalMuseModel::new(
                self.config.clone(),
                self.weights.clone(),
                config.max_context,
            )?)),
            tokenizer: Arc::clone(&self.tokenizer),
            max_context: config.max_context,
            token_history: Vec::new(),
            last_logits: None,
        })
    }

    /// Create one scheduler-owned resident group sharing the immutable Metal
    /// context, PSOs, mapped target weights, and uploaded vector weights.
    /// Sequence-local KV/activation/speculative state remains disjoint.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn new_metal_sessions(
        &self,
        config: SessionConfig,
        count: usize,
    ) -> Result<Vec<Session>, EngineError> {
        self.validate_session_config(config)?;
        if count == 0 {
            return Ok(Vec::new());
        }
        crate::decode::MetalMuseModel::new_sequence_group(
            self.config.clone(),
            self.weights.clone(),
            config.max_context,
            count,
        )
        .map_err(EngineError::from)
        .map(|models| {
            models
                .into_iter()
                .map(|model| Session {
                    backend: SessionBackend::Metal(Box::new(model)),
                    tokenizer: Arc::clone(&self.tokenizer),
                    max_context: config.max_context,
                    token_history: Vec::new(),
                    last_logits: None,
                })
                .collect()
        })
    }

    fn validate_session_config(&self, config: SessionConfig) -> Result<(), EngineError> {
        if config.max_context == 0 || config.max_context > self.config.context_length {
            return Err(EngineError::InvalidMaxContext {
                requested: config.max_context,
                model_limit: self.config.context_length,
            });
        }
        Ok(())
    }

    /// Borrow the target model's embedding table for the DFlash assistant.
    #[doc(hidden)]
    pub fn embed_tokens(&self, tokens: &[u32]) -> Result<Vec<f32>, EngineError> {
        if let Some(&token) = tokens
            .iter()
            .find(|&&token| token as usize >= self.config.vocab_size)
        {
            return Err(EngineError::InvalidToken {
                token,
                vocab_size: self.config.vocab_size,
            });
        }
        let embedding = self.weights.view("token_embd.weight")?;
        let mut output = vec![0.0; tokens.len() * self.config.hidden_dim];
        for (row, &token) in tokens.iter().enumerate() {
            crate::weights::dequant_row(
                &embedding,
                token as usize,
                &mut output[row * self.config.hidden_dim..(row + 1) * self.config.hidden_dim],
            );
        }
        Ok(output)
    }

    /// Apply the target LM head, including Muse's scale and softcap.
    #[doc(hidden)]
    pub fn project_hidden(&self, hidden: &[f32]) -> Result<Vec<f32>, EngineError> {
        if !hidden.len().is_multiple_of(self.config.hidden_dim) {
            return Err(EngineError::InvalidEmbedding {
                index: 0,
                actual: hidden.len() % self.config.hidden_dim,
                expected: self.config.hidden_dim,
            });
        }
        let rows = hidden.len() / self.config.hidden_dim;
        let mut logits = vec![0.0; rows * self.config.vocab_size];
        crate::weights::matmul(
            &self.weights.view("output.weight")?,
            hidden,
            rows,
            &mut logits,
        );
        for value in &mut logits {
            *value *= self.config.logit_scale;
            if self.config.final_logit_softcap > 0.0 {
                let cap = self.config.final_logit_softcap;
                *value = cap * (*value / cap).tanh();
            }
        }
        Ok(logits)
    }
}

#[derive(Debug, Clone)]
pub struct EmbeddingSegment {
    pub vectors: Vec<Vec<f32>>,
    /// Exact position witnesses carried by cache snapshots. These are not
    /// embedded or sampled; they bind non-token decoder positions to the
    /// request identity so a detached multimodal cache has one witness per
    /// logical row.
    pub position_witnesses: Vec<u32>,
}

/// Reserved outside the Muse vocabulary. It can occur only as a cache
/// position witness for a projected embedding row, never as model input.
pub const EMBEDDING_POSITION_WITNESS: u32 = i32::MAX as u32;

impl EmbeddingSegment {
    pub fn new(vectors: Vec<Vec<f32>>) -> Self {
        Self {
            position_witnesses: vec![EMBEDDING_POSITION_WITNESS; vectors.len()],
            vectors,
        }
    }
}

#[derive(Debug, Clone)]
pub enum PrefillSegment {
    Tokens(Vec<u32>),
    Embeddings(EmbeddingSegment),
}

#[derive(Debug, Clone, Default)]
pub struct PrefillBatch {
    pub segments: Vec<PrefillSegment>,
}

impl PrefillBatch {
    pub fn tokens(tokens: impl Into<Vec<u32>>) -> Self {
        Self {
            segments: vec![PrefillSegment::Tokens(tokens.into())],
        }
    }
}

#[derive(Debug, Clone)]
pub struct PrefillResult {
    /// Only the final decoder row is contractual — read it through
    /// [`PrefillResult::last_logits`], never by indexing a prompt position.
    ///
    /// The two backends deliberately produce different amounts of it: the CPU
    /// oracle returns every prompt row (`tokens_processed * vocab_size`)
    /// because its matmul computes them anyway, while the Metal prefill copies
    /// the last row only (`vocab_size`) rather than spending bandwidth on rows
    /// no caller reads. Nothing in the engine depends on the difference, and
    /// this wave does not change it.
    pub logits: Vec<f32>,
    pub tokens_processed: usize,
    pub vocab_size: usize,
}

impl PrefillResult {
    /// The distribution for the position after the last prefilled token.
    pub fn last_logits(&self) -> &[f32] {
        let start = self.logits.len().saturating_sub(self.vocab_size);
        &self.logits[start..]
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DecodeInput {
    pub token_id: u32,
}

#[derive(Debug, Clone)]
pub struct DecodeResult {
    pub input_token: u32,
    pub next_token: u32,
    pub logits: Vec<f32>,
    /// Opt-in host-path timing for the non-notarial streamed-decode
    /// diagnostic. Ordinary serving never populates this field.
    #[doc(hidden)]
    pub diagnostics: Option<DecodeDiagnostics>,
}

/// Result of one bounded, target-only GPU-resident greedy block. This is
/// private serving plumbing; public sampler and logprob contracts continue to
/// use [`DecodeResult`].
#[doc(hidden)]
pub struct GreedyBlockResult {
    pub consumed_tokens: Vec<u32>,
    pub next_token: u32,
    pub final_logits: Vec<f32>,
    pub cancelled: bool,
}

/// Per-token host timeline emitted only when
/// `MUSER_STREAM_DECODE_PROFILE=1`. These timings describe orchestration
/// around the unchanged Metal graph; they are not qualification evidence.
#[derive(Debug, Clone, Default)]
pub struct DecodeDiagnostics {
    pub model_prepare_ns: u64,
    pub model_encode_ns: u64,
    pub encoder_end_ns: u64,
    pub command_commit_ns: u64,
    pub gpu_wait_ns: u64,
    pub logits_readback_ns: u64,
    pub finite_scan_ns: u64,
    pub argmax_ns: u64,
    pub result_clone_ns: u64,
}

#[derive(Debug, Clone)]
pub struct SpeculativeBatch {
    pub draft_tokens: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct VerificationResult {
    pub accepted: usize,
    pub replacement: Option<u32>,
}

/// Mutable inference state for one sequence.
///
/// A session owns its KV cache, token history, retained next-token logits,
/// and context limit. Call [`Session::prefill`] once or more, then pass each
/// selected token to [`Session::decode`] to advance the sequence.
pub struct Session {
    backend: SessionBackend,
    tokenizer: Arc<BpeTokenizer>,
    max_context: usize,
    token_history: Vec<u32>,
    last_logits: Option<Vec<f32>>,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) struct PendingDFlashVerification {
    metal: crate::decode::PendingMetalDFlashVerify,
    checkpoint: crate::decode::MetalSpeculativeCheckpoint,
    draft_tokens: Vec<u32>,
    starting_logits: Vec<f32>,
    capture_width: usize,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
type DFlashVerificationOverlap = (PendingDFlashVerification, Vec<f32>, Option<Vec<f32>>);

enum SessionBackend {
    Cpu(Box<MuseModel>),
    #[cfg(all(target_os = "macos", feature = "metal"))]
    Metal(Box<crate::decode::MetalMuseModel>),
}

impl Session {
    pub fn prefill(&mut self, batch: PrefillBatch) -> Result<PrefillResult, EngineError> {
        let mut logits = Vec::new();
        let mut positions = 0usize;
        for segment in batch.segments {
            match segment {
                PrefillSegment::Tokens(tokens) => {
                    if tokens.is_empty() {
                        continue;
                    }
                    self.validate_tokens(&tokens)?;
                    self.ensure_capacity(tokens.len())?;
                    positions += tokens.len();
                    logits.extend(self.forward(&tokens)?);
                }
                PrefillSegment::Embeddings(segment) => {
                    if segment.vectors.is_empty() {
                        continue;
                    }
                    let expected = self.hidden_dim();
                    if let Some((index, vector)) = segment
                        .vectors
                        .iter()
                        .enumerate()
                        .find(|(_, vector)| vector.len() != expected)
                    {
                        return Err(EngineError::InvalidEmbedding {
                            index,
                            actual: vector.len(),
                            expected,
                        });
                    }
                    if segment.position_witnesses.len() != segment.vectors.len()
                        || segment
                            .position_witnesses
                            .iter()
                            .any(|&witness| witness != EMBEDDING_POSITION_WITNESS)
                    {
                        return Err(EngineError::InvalidEmbeddingWitnesses);
                    }
                    self.ensure_capacity(segment.vectors.len())?;
                    let count = segment.vectors.len();
                    let witnesses = segment.position_witnesses;
                    let flat = segment.vectors.into_iter().flatten().collect::<Vec<_>>();
                    logits.extend(self.forward_embeddings(&flat)?);
                    self.token_history.extend_from_slice(&witnesses);
                    positions += count;
                }
            }
        }
        if positions == 0 {
            return Err(EngineError::EmptyPrefill);
        }
        let final_logits = final_logit_row(&logits, self.vocab_size())?;
        ensure_finite_logits(&final_logits)?;
        self.last_logits = Some(final_logits);
        Ok(PrefillResult {
            logits,
            tokens_processed: positions,
            vocab_size: self.vocab_size(),
        })
    }

    pub fn decode(&mut self, input: DecodeInput) -> Result<DecodeResult, EngineError> {
        self.validate_tokens(&[input.token_id])?;
        self.ensure_capacity(1)?;
        // The decode path refills the retained distribution in place, so a
        // token costs one vocabulary-sized copy for the result instead of two
        // fresh allocations.
        let mut logits = self.last_logits.take().unwrap_or_default();
        if let Err(error) = self.forward_into(&[input.token_id], &mut logits) {
            // A failed forward leaves the buffer untouched, so the
            // distribution installed before the call is still the current one.
            self.last_logits = (!logits.is_empty()).then_some(logits);
            return Err(error);
        }
        // Fail closed: a broken row installs no distribution at all.
        #[cfg(all(target_os = "macos", feature = "metal"))]
        let mut diagnostics = crate::decode::take_stream_decode_diagnostics();
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        let mut diagnostics: Option<DecodeDiagnostics> = None;
        let finite_started = diagnostics.as_ref().map(|_| std::time::Instant::now());
        ensure_finite_logits(&logits)?;
        if let (Some(diagnostics), Some(started)) = (&mut diagnostics, finite_started) {
            diagnostics.finite_scan_ns = elapsed_ns(started);
        }
        let argmax_started = diagnostics.as_ref().map(|_| std::time::Instant::now());
        let next_token = argmax(&logits) as u32;
        if let (Some(diagnostics), Some(started)) = (&mut diagnostics, argmax_started) {
            diagnostics.argmax_ns = elapsed_ns(started);
        }
        let clone_started = diagnostics.as_ref().map(|_| std::time::Instant::now());
        let result_logits = logits.clone();
        if let (Some(diagnostics), Some(started)) = (&mut diagnostics, clone_started) {
            diagnostics.result_clone_ns = elapsed_ns(started);
        }
        let result = DecodeResult {
            input_token: input.token_id,
            next_token,
            logits: result_logits,
            diagnostics,
        };
        self.last_logits = Some(logits);
        Ok(result)
    }

    /// Execute at most the engine's bounded speculative width as a dependent
    /// GPU greedy chain. The first token is selected from the currently
    /// retained distribution; `on_token` receives subsequent selected tokens
    /// as their producing command buffers complete.
    #[doc(hidden)]
    pub fn decode_greedy_block(
        &mut self,
        first_token: u32,
        token_count: usize,
        excluded_tokens: &[u32],
        on_token: impl FnMut(u32) -> bool,
    ) -> Result<GreedyBlockResult, EngineError> {
        self.validate_tokens(&[first_token])?;
        self.ensure_capacity(token_count)?;
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let SessionBackend::Metal(model) = &mut self.backend else {
                return Err(EngineError::InvalidCacheSnapshot(
                    "GPU-resident greedy decode requires Metal".into(),
                ));
            };
            let result = model.forward_greedy_streaming(
                first_token,
                token_count,
                excluded_tokens,
                on_token,
            )?;
            if result.cancelled {
                model.reset();
                self.token_history.clear();
                self.last_logits = None;
            } else {
                ensure_finite_logits(&result.final_logits)?;
                self.token_history
                    .extend_from_slice(&result.consumed_tokens);
                self.last_logits = Some(result.final_logits.clone());
            }
            Ok(GreedyBlockResult {
                consumed_tokens: result.consumed_tokens,
                next_token: result.next_token,
                final_logits: result.final_logits,
                cancelled: result.cancelled,
            })
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = (excluded_tokens, on_token);
            Err(EngineError::InvalidCacheSnapshot(
                "GPU-resident greedy decode requires Metal".into(),
            ))
        }
    }

    /// Submit one decode row from each resident Metal sequence as a packed
    /// accelerator batch. The caller retains independent Session objects;
    /// only immutable weights and this one physical command graph are shared.
    #[doc(hidden)]
    pub fn decode_group(
        sessions: &mut [&mut Session],
        inputs: &[DecodeInput],
    ) -> Result<Vec<DecodeResult>, EngineError> {
        if sessions.is_empty() || sessions.len() > 4 || sessions.len() != inputs.len() {
            return Err(EngineError::InvalidCacheSnapshot(
                "decode group must contain matching 1..=4 sessions and inputs".into(),
            ));
        }
        for (session, input) in sessions.iter().zip(inputs) {
            session.validate_tokens(&[input.token_id])?;
            session.ensure_capacity(1)?;
        }
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let mut models = Vec::with_capacity(sessions.len());
            for session in sessions.iter_mut() {
                match &mut session.backend {
                    SessionBackend::Metal(model) => models.push(model.as_mut()),
                    SessionBackend::Cpu(_) => {
                        return Err(EngineError::InvalidCacheSnapshot(
                            "packed decode requires Metal sessions".into(),
                        ));
                    }
                }
            }
            let tokens = inputs
                .iter()
                .map(|input| input.token_id)
                .collect::<Vec<_>>();
            let logits = crate::decode::MetalMuseModel::forward_decode_group(&mut models, &tokens)?;
            let mut results = Vec::with_capacity(logits.len());
            for ((session, input), logits) in
                sessions.iter_mut().zip(inputs).zip(logits.into_iter())
            {
                ensure_finite_logits(&logits)?;
                let next_token = argmax(&logits) as u32;
                session.token_history.push(input.token_id);
                session.last_logits = Some(logits.clone());
                results.push(DecodeResult {
                    input_token: input.token_id,
                    next_token,
                    logits,
                    diagnostics: None,
                });
            }
            Ok(results)
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        {
            let _ = inputs;
            Err(EngineError::InvalidCacheSnapshot(
                "packed decode requires the Metal feature on macOS".into(),
            ))
        }
    }

    /// L2-normalized final nonpadding decoder hidden state. The returned
    /// dimension is exactly the model hidden width (6656 for pinned Muse).
    pub fn embedding(&mut self, tokens: &[u32]) -> Result<Vec<f32>, EngineError> {
        if tokens.is_empty() {
            return Err(EngineError::EmptyPrefill);
        }
        self.validate_tokens(tokens)?;
        self.ensure_capacity(tokens.len())?;
        let (logits, hidden) = match &mut self.backend {
            SessionBackend::Cpu(model) => model.forward_final_hidden(tokens),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            SessionBackend::Metal(model) => model.forward_final_hidden(tokens)?,
        };
        self.token_history.extend_from_slice(tokens);
        self.last_logits = Some(final_logit_row(&logits, self.vocab_size())?);
        let width = self.hidden_dim();
        let mut embedding = hidden[hidden.len() - width..].to_vec();
        let norm = embedding
            .iter()
            .map(|value| (*value as f64) * (*value as f64))
            .sum::<f64>()
            .sqrt();
        if !norm.is_finite() || norm == 0.0 {
            return Err(EngineError::InvalidCacheSnapshot(
                "final hidden state has an invalid L2 norm".into(),
            ));
        }
        for value in &mut embedding {
            *value = (*value as f64 / norm) as f32;
        }
        Ok(embedding)
    }

    /// Qualification-only teacher-forced sink matching llama-bench's pinned
    /// no-sampler and no-per-token-host-readback policy. Returned IDs are the
    /// teacher inputs as a workload witness, not generated output; greedy
    /// equality is checked by the correctness lane. This is not part of the
    /// public model serving contract.
    #[doc(hidden)]
    pub fn teacher_forced_decode(&mut self, tokens: &[u32]) -> Result<Vec<u32>, EngineError> {
        self.validate_tokens(tokens)?;
        self.ensure_capacity(tokens.len())?;
        if matches!(&self.backend, SessionBackend::Cpu(_)) {
            for &token_id in tokens {
                self.forward(&[token_id])?;
            }
            return Ok(tokens.to_vec());
        }
        match &mut self.backend {
            SessionBackend::Cpu(_) => unreachable!("CPU route returned above"),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            SessionBackend::Metal(model) => {
                let generated = model.forward_teacher_forced(tokens)?;
                self.token_history.extend_from_slice(tokens);
                self.last_logits = None;
                Ok(generated)
            }
        }
    }

    pub fn verify_batch(
        &mut self,
        batch: SpeculativeBatch,
    ) -> Result<VerificationResult, EngineError> {
        self.verify_batch_capturing_layers(&batch.draft_tokens, &[])
            .map(|(verification, _)| verification)
    }

    pub fn reset(&mut self) {
        match &mut self.backend {
            SessionBackend::Cpu(model) => model.reset(),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            SessionBackend::Metal(model) => model.reset(),
        }
        self.token_history.clear();
        self.last_logits = None;
    }

    pub fn position(&self) -> usize {
        match &self.backend {
            SessionBackend::Cpu(model) => model.n_past,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            SessionBackend::Metal(model) => model.position(),
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text)
    }

    /// Session-side `Model::encode_with_options`.
    pub fn encode_with_options(&self, text: &str, parse_special: bool) -> Vec<u32> {
        self.tokenizer.encode_with_options(text, parse_special)
    }

    pub fn decode_tokens(&self, tokens: &[u32]) -> String {
        self.tokenizer.decode_all(tokens)
    }

    pub fn token_history(&self) -> &[u32] {
        &self.token_history
    }

    /// Internal cache export reserved for `muser-kvpack`.
    #[doc(hidden)]
    pub fn export_cache_snapshot(&self) -> Result<crate::cache::SessionCacheSnapshot, EngineError> {
        match &self.backend {
            SessionBackend::Cpu(model) => model
                .export_cache_snapshot(&self.token_history)
                .map_err(EngineError::InvalidCacheSnapshot),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            SessionBackend::Metal(model) => Ok(model.export_cache_snapshot(&self.token_history)?),
        }
    }

    /// Validate and stage a complete snapshot before replacing live state.
    #[doc(hidden)]
    pub fn install_cache_snapshot(
        &mut self,
        snapshot: &crate::cache::SessionCacheSnapshot,
    ) -> Result<(), EngineError> {
        snapshot
            .validate()
            .map_err(EngineError::InvalidCacheSnapshot)?;
        if snapshot.position as usize > self.max_context {
            return Err(EngineError::ContextOverflow {
                requested: snapshot.position as usize,
                limit: self.max_context,
            });
        }
        match &mut self.backend {
            SessionBackend::Cpu(model) => model
                .install_cache_snapshot(snapshot)
                .map_err(EngineError::InvalidCacheSnapshot)?,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            SessionBackend::Metal(model) => model.install_cache_snapshot(snapshot)?,
        }
        self.token_history.clear();
        self.token_history.extend_from_slice(&snapshot.tokens);
        self.last_logits = None;
        Ok(())
    }

    /// Allocate a detached KV generation that can accept authenticated tiles
    /// while the live cache stays at the previous generation.
    #[doc(hidden)]
    pub fn begin_remote_kv_install(
        &self,
        tokens: Arc<[u32]>,
    ) -> Result<RemoteKvInstall, EngineError> {
        if tokens.is_empty() || tokens.len() > self.max_context {
            return Err(EngineError::ContextOverflow {
                requested: tokens.len().max(1),
                limit: self.max_context,
            });
        }
        match &self.backend {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            SessionBackend::Metal(model) => Ok(RemoteKvInstall {
                tokens: Arc::clone(&tokens),
                inner: RemoteKvInner::Metal(model.begin_remote_kv_install(tokens)?),
            }),
            SessionBackend::Cpu(_) => Ok(RemoteKvInstall {
                tokens,
                inner: RemoteKvInner::Unsupported,
            }),
        }
    }

    /// Delta variant of [`Session::begin_remote_kv_install`]: the detached
    /// generation covers the full prompt, the held prefix `[0, prefix_cut)`
    /// is copied from the live cache planes with the exact ring mapping, and
    /// the write cursors resume at the cut so only suffix tiles are accepted.
    /// `prefix_cut == 0` is byte-identical to the full install.
    #[doc(hidden)]
    pub fn begin_remote_kv_install_delta(
        &self,
        tokens: Arc<[u32]>,
        prefix_cut: u64,
    ) -> Result<RemoteKvInstall, EngineError> {
        if prefix_cut == 0 {
            return self.begin_remote_kv_install(tokens);
        }
        if tokens.is_empty() || tokens.len() > self.max_context {
            return Err(EngineError::ContextOverflow {
                requested: tokens.len().max(1),
                limit: self.max_context,
            });
        }
        let cut = usize::try_from(prefix_cut).map_err(|_| {
            EngineError::InvalidCacheSnapshot("delta prefix cut exceeds platform usize".into())
        })?;
        if cut >= tokens.len() || cut > self.position() {
            return Err(EngineError::InvalidCacheSnapshot(format!(
                "delta prefix cut {cut} must leave a suffix the live session already holds"
            )));
        }
        match &self.backend {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            SessionBackend::Metal(model) => Ok(RemoteKvInstall {
                tokens: Arc::clone(&tokens),
                inner: RemoteKvInner::Metal(model.begin_remote_kv_install_delta(tokens, cut)?),
            }),
            SessionBackend::Cpu(_) => Ok(RemoteKvInstall {
                tokens,
                inner: RemoteKvInner::Unsupported,
            }),
        }
    }

    /// Atomically replace live KV with a fully filled remote generation.
    #[doc(hidden)]
    pub fn commit_remote_kv_install(
        &mut self,
        install: RemoteKvInstall,
    ) -> Result<(), EngineError> {
        let prepared = self.prepare_remote_kv_install(install)?;
        self.commit_prepared_remote_kv_install(prepared);
        Ok(())
    }

    #[doc(hidden)]
    pub fn prepare_remote_kv_install(
        &self,
        install: RemoteKvInstall,
    ) -> Result<PreparedRemoteKvInstall, EngineError> {
        if install.tokens.len() > self.max_context {
            return Err(EngineError::ContextOverflow {
                requested: install.tokens.len(),
                limit: self.max_context,
            });
        }
        install.validate_complete()?;
        match (&self.backend, &install.inner) {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            (SessionBackend::Metal(_), RemoteKvInner::Metal(_)) => Ok(PreparedRemoteKvInstall {
                tokens: install.tokens,
                inner: install.inner,
            }),
            _ => Err(EngineError::InvalidCacheSnapshot(
                "remote KV tile install requires the Metal session".into(),
            )),
        }
    }

    #[doc(hidden)]
    pub fn commit_prepared_remote_kv_install(&mut self, install: PreparedRemoteKvInstall) {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let (SessionBackend::Metal(model), RemoteKvInner::Metal(metal)) =
            (&mut self.backend, install.inner)
        {
            model.commit_prepared_remote_kv_install(metal);
            self.token_history.clear();
            self.token_history.extend_from_slice(&install.tokens);
            self.last_logits = None;
            return;
        }
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        let _ = install;
        unreachable!("PreparedRemoteKvInstall can only be built for its matching Metal session")
    }

    /// Target hidden-state capture used by the standalone DFlash assistant.
    /// The CPU oracle is the qualification reference; the Metal batch capture
    /// is installed separately with the GPU assistant path.
    #[doc(hidden)]
    pub fn prefill_capturing_layers(
        &mut self,
        tokens: &[u32],
        layer_ids: &[usize],
    ) -> Result<(Vec<f32>, Vec<f32>), EngineError> {
        self.validate_tokens(tokens)?;
        self.ensure_capacity(tokens.len())?;
        match &mut self.backend {
            SessionBackend::Cpu(model) => {
                let (logits, hidden) = model.forward_capturing_layers(tokens, layer_ids);
                self.token_history.extend_from_slice(tokens);
                self.last_logits = Some(final_logit_row(&logits, model.cfg.vocab_size)?);
                Ok((logits, hidden))
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            SessionBackend::Metal(model) => {
                let (logits, hidden) = model.forward_capturing_layers(tokens, layer_ids)?;
                self.token_history.extend_from_slice(tokens);
                self.last_logits = Some(final_logit_row(&logits, model.cfg.vocab_size)?);
                Ok((logits, hidden))
            }
        }
    }

    /// Capture DFlash target layers across an ordered token/embedding prefill.
    /// The returned hidden rows follow decoder position order regardless of
    /// segment type, with one row per selected layer at every position.
    #[doc(hidden)]
    pub fn prefill_batch_capturing_layers(
        &mut self,
        batch: PrefillBatch,
        layer_ids: &[usize],
    ) -> Result<(Vec<f32>, Vec<f32>), EngineError> {
        let mut logits = Vec::new();
        let mut hidden = Vec::new();
        let mut positions = 0usize;
        for segment in batch.segments {
            match segment {
                PrefillSegment::Tokens(tokens) => {
                    if tokens.is_empty() {
                        continue;
                    }
                    positions += tokens.len();
                    let (next_logits, rows) = self.prefill_capturing_layers(&tokens, layer_ids)?;
                    logits = next_logits;
                    hidden.extend_from_slice(&rows);
                }
                PrefillSegment::Embeddings(segment) => {
                    if segment.vectors.is_empty() {
                        continue;
                    }
                    let expected = self.hidden_dim();
                    if let Some((index, vector)) = segment
                        .vectors
                        .iter()
                        .enumerate()
                        .find(|(_, vector)| vector.len() != expected)
                    {
                        return Err(EngineError::InvalidEmbedding {
                            index,
                            actual: vector.len(),
                            expected,
                        });
                    }
                    if segment.position_witnesses.len() != segment.vectors.len()
                        || segment
                            .position_witnesses
                            .iter()
                            .any(|&witness| witness != EMBEDDING_POSITION_WITNESS)
                    {
                        return Err(EngineError::InvalidEmbeddingWitnesses);
                    }
                    let count = segment.vectors.len();
                    self.ensure_capacity(count)?;
                    let flat = segment.vectors.into_iter().flatten().collect::<Vec<_>>();
                    let (next_logits, rows) = match &mut self.backend {
                        SessionBackend::Cpu(model) => {
                            model.forward_embeddings_capturing_layers(&flat, layer_ids)
                        }
                        #[cfg(all(target_os = "macos", feature = "metal"))]
                        SessionBackend::Metal(model) => {
                            model.forward_embeddings_capturing_layers(&flat, layer_ids)?
                        }
                    };
                    self.token_history
                        .extend_from_slice(&segment.position_witnesses);
                    self.last_logits = Some(final_logit_row(&next_logits, self.vocab_size())?);
                    positions += count;
                    logits = next_logits;
                    hidden.extend_from_slice(&rows);
                }
            }
        }
        if positions == 0 {
            return Err(EngineError::EmptyPrefill);
        }
        Ok((logits, hidden))
    }

    /// Token-only Metal prompt path for the standalone greedy DFlash lane.
    /// Returns `None` unless both target and assistant have Metal backends, so
    /// the CPU oracle and multimodal batches keep the independent legacy path.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn prefill_dflash_prompt_pipelined(
        &mut self,
        tokens: &[u32],
        layer_ids: &[usize],
        forward: &mut crate::dflash::DFlashForward,
        cache: &mut crate::dflash::DFlashContextKvCache,
    ) -> Result<Option<DFlashPromptPrefill>, EngineError> {
        let Some(assistant) = forward.metal_forward.as_mut() else {
            return Ok(None);
        };
        self.validate_tokens(tokens)?;
        self.ensure_capacity(tokens.len())?;
        let SessionBackend::Metal(model) = &mut self.backend else {
            return Ok(None);
        };
        let (logits, newest_hidden, stats) =
            model.forward_dflash_prompt_pipelined(tokens, layer_ids, assistant, cache)?;
        self.token_history.extend_from_slice(tokens);
        self.last_logits = Some(final_logit_row(&logits, model.cfg.vocab_size)?);
        Ok(Some((logits, newest_hidden, stats)))
    }

    #[doc(hidden)]
    pub fn greedy_next_token(&self) -> Result<u32, EngineError> {
        self.last_logits
            .as_ref()
            .map(|logits| argmax(logits) as u32)
            .ok_or(EngineError::MissingVerificationState)
    }

    /// Last target distribution associated with the installed cache cut.
    /// This is private cache plumbing, not part of the public generation API.
    #[doc(hidden)]
    pub fn cached_logits(&self) -> Option<&[f32]> {
        self.last_logits.as_deref()
    }

    /// Apply the target output matrix to DFlash hidden rows on the session's
    /// active backend. Metal uses retained quantized-weight GPU scratch;
    /// CPU remains the independent correctness oracle.
    #[doc(hidden)]
    pub fn project_dflash_hidden(
        &mut self,
        target: &Model,
        hidden: &[f32],
    ) -> Result<Vec<f32>, EngineError> {
        match &mut self.backend {
            SessionBackend::Cpu(_) => target.project_hidden(hidden),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            SessionBackend::Metal(model) => Ok(model.project_dflash_hidden(hidden)?),
        }
    }

    #[doc(hidden)]
    pub fn project_dflash_output(
        &mut self,
        target: &Model,
        output: &crate::dflash::DFlashDraftOutput,
    ) -> Result<Vec<f32>, EngineError> {
        match &mut self.backend {
            SessionBackend::Cpu(_) => target.project_hidden(&output.hidden_states),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            SessionBackend::Metal(model) => match output.metal_hidden_states.as_ref() {
                Some(buffer) => Ok(model.project_dflash_buffer(buffer, output.n_draft_tokens + 1)?),
                None => Ok(model.project_dflash_hidden(&output.hidden_states)?),
            },
        }
    }

    /// Greedy-only DFlash output projection. Metal retains the full logits on
    /// device and returns one reduced token ID per assistant row; CPU remains
    /// the scalar first-maximum oracle. Sampled speculation intentionally uses
    /// `project_dflash_output` because it requires the full proposal density.
    #[doc(hidden)]
    pub fn project_dflash_argmax(
        &mut self,
        target: &Model,
        output: &crate::dflash::DFlashDraftOutput,
    ) -> Result<Vec<u32>, EngineError> {
        match &mut self.backend {
            SessionBackend::Cpu(_) => {
                let logits = target.project_hidden(&output.hidden_states)?;
                Ok(logits
                    .chunks_exact(target.config().vocab_size)
                    .map(|row| argmax(row) as u32)
                    .collect())
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            SessionBackend::Metal(model) => match output.metal_hidden_states.as_ref() {
                Some(buffer) => {
                    Ok(model.project_dflash_buffer_argmax(buffer, output.n_draft_tokens + 1)?)
                }
                None => Ok(model.project_dflash_hidden_argmax(&output.hidden_states)?),
            },
        }
    }

    /// Attach a cryptographically/exactly witnessed final distribution to an
    /// already validated cache installation. Resident exact-final hits use
    /// this so first-token generation does not recompute or fabricate state.
    #[doc(hidden)]
    pub fn install_restored_logits(&mut self, logits: &[f32]) -> Result<(), EngineError> {
        let expected = self.vocab_size();
        if logits.len() != expected || logits.iter().any(|value| !value.is_finite()) {
            return Err(EngineError::InvalidRestoredLogits {
                actual: logits.len(),
                expected,
            });
        }
        self.last_logits = Some(logits.to_vec());
        Ok(())
    }

    /// Evaluate a candidate sequence against the target without changing the
    /// live session.  Row zero is the distribution before `tokens[0]`; each
    /// following row is the distribution after consuming the corresponding
    /// token.  This is the legacy sampled route: it rolls the batch back in
    /// full, so the caller must replay the accepted prefix itself.  The live
    /// sampled route uses [`Session::verify_sampled_capturing_layers`], which
    /// commits that prefix out of the same batch.
    #[doc(hidden)]
    pub fn evaluate_tokens_transactional(
        &mut self,
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>, EngineError> {
        self.validate_tokens(tokens)?;
        self.ensure_capacity(tokens.len())?;
        let starting_logits = self
            .last_logits
            .clone()
            .ok_or(EngineError::MissingVerificationState)?;

        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let SessionBackend::Metal(model) = &mut self.backend {
            let checkpoint = model.speculative_checkpoint(tokens.len())?;
            let evaluated = model.forward_batch_all_logits_capturing(tokens, &[]);
            return match evaluated {
                Ok((flat_logits, _)) => {
                    model.commit_speculative_prefix(checkpoint, 0)?;
                    let mut rows = Vec::with_capacity(tokens.len() + 1);
                    rows.push(starting_logits);
                    rows.extend(
                        flat_logits
                            .chunks_exact(model.cfg.vocab_size)
                            .map(|row| row.to_vec()),
                    );
                    Ok(rows)
                }
                Err(error) => {
                    // The checkpoint contains every potentially overwritten
                    // SWA row, so setup/completion failures can still restore
                    // the prior generation before the error escapes.
                    model.commit_speculative_prefix(checkpoint, 0)?;
                    Err(error.into())
                }
            };
        }
        let snapshot = self.export_cache_snapshot()?;
        let evaluated = (|| {
            let mut rows = Vec::with_capacity(tokens.len() + 1);
            rows.push(starting_logits.clone());
            for &token in tokens {
                rows.push(self.forward(&[token])?);
            }
            Ok(rows)
        })();

        // Installation validates a detached snapshot before it touches live
        // state.  Prefer a restoration error over a forward error: returning
        // with an uncertain session would violate the transactional contract.
        self.install_cache_snapshot(&snapshot)?;
        self.last_logits = Some(starting_logits);
        evaluated
    }

    /// Transactional sampled verification with target-layer capture in the
    /// same batch. `decide` receives exactly the row layout of
    /// [`Session::evaluate_tokens_transactional`] — row zero is the
    /// distribution before `candidates[0]` — and returns how many leading
    /// candidates to commit; the acceptance rule itself stays with the caller.
    /// Nothing is committed until `decide` returns, and every failure path
    /// restores the cut that was live before the call. Returned hidden rows
    /// cover only the committed prefix, in token-major DFlash layout.
    #[doc(hidden)]
    pub fn verify_sampled_capturing_layers<E, F>(
        &mut self,
        candidates: &[u32],
        layer_ids: &[usize],
        decide: F,
    ) -> Result<(usize, Vec<f32>), E>
    where
        E: From<EngineError>,
        F: FnOnce(&[Vec<f32>]) -> Result<usize, E>,
    {
        if candidates.is_empty() {
            return Err(EngineError::EmptyPrefill.into());
        }
        self.validate_tokens(candidates)?;
        self.ensure_capacity(candidates.len())?;
        let starting_logits = self
            .last_logits
            .clone()
            .ok_or(EngineError::MissingVerificationState)?;
        let capture_width = layer_ids.len() * self.hidden_dim();
        let vocab = self.vocab_size();

        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let SessionBackend::Metal(model) = &mut self.backend {
            let checkpoint = model
                .speculative_checkpoint(candidates.len())
                .map_err(EngineError::from)?;
            let evaluated = (|| {
                let (flat_logits, hidden) = model
                    .forward_batch_all_logits_capturing(candidates, layer_ids)
                    .map_err(EngineError::from)?;
                if flat_logits.len() != candidates.len().saturating_mul(vocab) {
                    return Err(E::from(EngineError::InvalidRestoredLogits {
                        actual: flat_logits.len(),
                        expected: candidates.len().saturating_mul(vocab),
                    }));
                }
                let mut rows = Vec::with_capacity(candidates.len() + 1);
                rows.push(starting_logits.clone());
                rows.extend(flat_logits.chunks_exact(vocab).map(<[f32]>::to_vec));
                let accepted = decide(&rows)?;
                if accepted > candidates.len() {
                    return Err(E::from(EngineError::InvalidSpeculativeCommit {
                        accepted,
                        evaluated: candidates.len(),
                    }));
                }
                Ok((rows, hidden, accepted))
            })();
            // The checkpoint spans every row this batch could have overwritten,
            // so a decision failure still restores the previous generation.
            let (rows, mut hidden, accepted) = match evaluated {
                Ok(value) => value,
                Err(error) => {
                    model
                        .commit_speculative_prefix(checkpoint, 0)
                        .map_err(EngineError::from)?;
                    return Err(error);
                }
            };
            model
                .commit_speculative_prefix(checkpoint, accepted)
                .map_err(EngineError::from)?;
            hidden.truncate(accepted * capture_width);
            self.token_history
                .extend_from_slice(&candidates[..accepted]);
            self.last_logits = Some(rows[accepted].clone());
            return Ok((accepted, hidden));
        }

        let snapshot = self.export_cache_snapshot()?;
        let evaluated = (|| -> Result<(Vec<Vec<f32>>, Vec<f32>), EngineError> {
            let mut rows = Vec::with_capacity(candidates.len() + 1);
            rows.push(starting_logits.clone());
            let mut hidden = Vec::with_capacity(candidates.len() * capture_width);
            for &token in candidates {
                let (logits, captured) = self.prefill_capturing_layers(&[token], layer_ids)?;
                rows.push(final_logit_row(&logits, vocab)?);
                hidden.extend_from_slice(&captured);
            }
            Ok((rows, hidden))
        })();
        let (rows, mut hidden) = match evaluated {
            Ok(value) => value,
            Err(error) => {
                self.restore_evaluated_batch(&snapshot, starting_logits)?;
                return Err(error.into());
            }
        };
        let accepted = match decide(&rows) {
            Ok(accepted) if accepted <= candidates.len() => accepted,
            Ok(accepted) => {
                self.restore_evaluated_batch(&snapshot, starting_logits)?;
                return Err(EngineError::InvalidSpeculativeCommit {
                    accepted,
                    evaluated: candidates.len(),
                }
                .into());
            }
            Err(error) => {
                self.restore_evaluated_batch(&snapshot, starting_logits)?;
                return Err(error);
            }
        };
        // A rejected row must never stay live: restore the pre-call cut and
        // replay the committed prefix only. A full accept is already live.
        if accepted != candidates.len() {
            self.restore_evaluated_batch(&snapshot, starting_logits)?;
            if accepted > 0 {
                self.forward(&candidates[..accepted])?;
            }
        }
        hidden.truncate(accepted * capture_width);
        self.last_logits = Some(rows[accepted].clone());
        Ok((accepted, hidden))
    }

    /// Undo a speculatively evaluated batch on the snapshot-based backend.
    fn restore_evaluated_batch(
        &mut self,
        snapshot: &crate::cache::SessionCacheSnapshot,
        starting_logits: Vec<f32>,
    ) -> Result<(), EngineError> {
        self.install_cache_snapshot(snapshot)?;
        self.last_logits = Some(starting_logits);
        Ok(())
    }

    /// Transactional greedy verification with target-layer capture. No
    /// rejected cache row becomes live; returned hidden rows cover only the
    /// accepted prefix and use token-major DFlash layout.
    #[doc(hidden)]
    pub fn verify_batch_capturing_layers(
        &mut self,
        draft_tokens: &[u32],
        layer_ids: &[usize],
    ) -> Result<(VerificationResult, Vec<f32>), EngineError> {
        if draft_tokens.is_empty() {
            return Ok((
                VerificationResult {
                    accepted: 0,
                    replacement: None,
                },
                Vec::new(),
            ));
        }
        self.validate_tokens(draft_tokens)?;
        self.ensure_capacity(draft_tokens.len())?;
        let starting_logits = self
            .last_logits
            .clone()
            .ok_or(EngineError::MissingVerificationState)?;

        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let SessionBackend::Metal(model) = &mut self.backend {
            let trace = crate::dflash::cycle_trace_enabled();
            let trace_started = std::time::Instant::now();
            let checkpoint = model.speculative_checkpoint(draft_tokens.len())?;
            let checkpoint_ns = trace_started.elapsed().as_nanos() as u64;
            // Port Ferrite's complete small-batch verifier: one target graph
            // produces every candidate logit row and every selected hidden row.
            // The earlier rejection was caused by untracked shared buffers, not
            // this graph. After restoring tracked synchronization, live four-
            // and eight-token cross-round POCs are target-exact with 100% draft
            // acceptance, while avoiding one command-buffer round trip per
            // candidate.
            let forward_started = std::time::Instant::now();
            let evaluated = model.forward_batch_all_logits_capturing(draft_tokens, layer_ids);
            let forward_ns = forward_started.elapsed().as_nanos() as u64;
            let (flat_logits, mut hidden) = match evaluated {
                Ok(result) => result,
                Err(error) => {
                    model.commit_speculative_prefix(checkpoint, 0)?;
                    return Err(error.into());
                }
            };
            let vocab = model.cfg.vocab_size;
            let decision_started = std::time::Instant::now();
            let decision = match greedy_verification_decision(
                &starting_logits,
                &flat_logits,
                draft_tokens,
                vocab,
            ) {
                Ok(decision) => decision,
                Err(error) => {
                    model.commit_speculative_prefix(checkpoint, 0)?;
                    return Err(error);
                }
            };
            let decision_ns = decision_started.elapsed().as_nanos() as u64;
            let accepted = decision.accepted;
            let commit_started = std::time::Instant::now();
            model.commit_speculative_prefix(checkpoint, accepted)?;
            let commit_ns = commit_started.elapsed().as_nanos() as u64;
            if trace {
                eprintln!(
                    "dflash-cycle-verify candidates={} checkpoint_ns={} forward_ns={} decision_ns={} commit_ns={}",
                    draft_tokens.len(), checkpoint_ns, forward_ns, decision_ns, commit_ns,
                );
            }
            hidden.truncate(accepted * layer_ids.len() * model.cfg.hidden_dim);
            self.token_history
                .extend_from_slice(&draft_tokens[..accepted]);
            self.last_logits = Some(if accepted == 0 {
                starting_logits
            } else {
                flat_logits[(accepted - 1) * vocab..accepted * vocab].to_vec()
            });
            return Ok((decision, hidden));
        }

        let snapshot = self.export_cache_snapshot()?;
        let mut logits = starting_logits.clone();
        let mut accepted = 0usize;
        let mut hidden = Vec::new();
        let mut replacement = None;
        match &mut self.backend {
            SessionBackend::Cpu(model) => {
                for &draft in draft_tokens {
                    let target = argmax(&logits) as u32;
                    if target != draft {
                        replacement = Some(target);
                        break;
                    }
                    let (next_logits, rows) = model.forward_capturing_layers(&[draft], layer_ids);
                    logits = next_logits;
                    hidden.extend_from_slice(&rows);
                    self.token_history.push(draft);
                    accepted += 1;
                }
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            SessionBackend::Metal(_) => unreachable!("Metal verifier returned above"),
        }
        if accepted != draft_tokens.len() {
            self.install_cache_snapshot(&snapshot)?;
            self.last_logits = Some(starting_logits);
            if accepted > 0 {
                logits = self.forward(&draft_tokens[..accepted])?;
            }
        }
        self.last_logits = Some(final_logit_row(&logits, self.vocab_size())?);
        Ok((
            VerificationResult {
                accepted,
                replacement,
            },
            hidden,
        ))
    }

    /// Submit the target suffix after all requested DFlash capture layers
    /// have completed. This is intentionally private to the exact Mirror-SD
    /// scheduler: callers must finish or abort the pending verification before
    /// touching target generation state again.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn begin_dflash_verification_overlap(
        &mut self,
        draft_tokens: &[u32],
        layer_ids: &[usize],
        capture_fc: Option<&dyn crate::dflash::DFlashProjectionBackend>,
    ) -> Result<DFlashVerificationOverlap, EngineError> {
        if draft_tokens.is_empty() {
            return Err(EngineError::InvalidCacheSnapshot(
                "Mirror-SD verification batch is empty".into(),
            ));
        }
        self.validate_tokens(draft_tokens)?;
        self.ensure_capacity(draft_tokens.len())?;
        let starting_logits = self
            .last_logits
            .clone()
            .ok_or(EngineError::MissingVerificationState)?;
        let SessionBackend::Metal(model) = &mut self.backend else {
            return Err(EngineError::SpeculativeVerificationUnavailable);
        };
        let checkpoint = model.speculative_checkpoint(draft_tokens.len())?;
        let split = layer_ids
            .iter()
            .copied()
            .max()
            .and_then(|layer| layer.checked_add(1))
            .ok_or_else(|| {
                EngineError::InvalidCacheSnapshot(
                    "Mirror-SD requires at least one target capture layer".into(),
                )
            })?;
        let (metal, hidden, target_projection) =
            match model.begin_dflash_verify_suffix(draft_tokens, layer_ids, split, capture_fc) {
                Ok(value) => value,
                Err(error) => {
                    model.commit_speculative_prefix(checkpoint, 0)?;
                    return Err(error.into());
                }
            };
        let capture_width = layer_ids.len() * model.cfg.hidden_dim;
        Ok((
            PendingDFlashVerification {
                metal,
                checkpoint,
                draft_tokens: draft_tokens.to_vec(),
                starting_logits,
                capture_width,
            },
            hidden,
            target_projection,
        ))
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn finish_dflash_verification_overlap(
        &mut self,
        pending: PendingDFlashVerification,
        mut hidden: Vec<f32>,
    ) -> Result<(VerificationResult, Vec<f32>, u64, u64), EngineError> {
        let PendingDFlashVerification {
            metal,
            checkpoint,
            draft_tokens,
            starting_logits,
            capture_width,
        } = pending;
        let SessionBackend::Metal(model) = &mut self.backend else {
            return Err(EngineError::SpeculativeVerificationUnavailable);
        };
        let (flat_logits, target_ns, capture_fc_ns) = match model.finish_dflash_verify_suffix(metal)
        {
            Ok(value) => value,
            Err(error) => {
                model.commit_speculative_prefix(checkpoint, 0)?;
                return Err(error.into());
            }
        };
        let vocab = model.cfg.vocab_size;
        let decision = match greedy_verification_decision(
            &starting_logits,
            &flat_logits,
            &draft_tokens,
            vocab,
        ) {
            Ok(decision) => decision,
            Err(error) => {
                model.commit_speculative_prefix(checkpoint, 0)?;
                return Err(error);
            }
        };
        let accepted = decision.accepted;
        if std::env::var_os("MUSER_DFLASH_MIRROR_DIAG").is_some() {
            let target_tokens = std::iter::once(argmax(&starting_logits) as u32)
                .chain(
                    flat_logits
                        .chunks_exact(vocab)
                        .take(draft_tokens.len().saturating_sub(1))
                        .map(|row| argmax(row) as u32),
                )
                .collect::<Vec<_>>();
            eprintln!(
                "[muser-mirror-diag] candidates={draft_tokens:?} target={target_tokens:?} accepted={} replacement={:?}",
                decision.accepted, decision.replacement
            );
        }
        model.commit_speculative_prefix(checkpoint, accepted)?;
        hidden.truncate(accepted * capture_width);
        self.token_history
            .extend_from_slice(&draft_tokens[..accepted]);
        self.last_logits = Some(if accepted == 0 {
            starting_logits
        } else {
            flat_logits[(accepted - 1) * vocab..accepted * vocab].to_vec()
        });
        Ok((decision, hidden, target_ns, capture_fc_ns))
    }

    fn ensure_capacity(&self, additional: usize) -> Result<(), EngineError> {
        let requested = self.position().saturating_add(additional);
        if requested > self.max_context {
            return Err(EngineError::ContextOverflow {
                requested,
                limit: self.max_context,
            });
        }
        Ok(())
    }

    fn validate_tokens(&self, tokens: &[u32]) -> Result<(), EngineError> {
        let vocab_size = self.vocab_size();
        if let Some(&token) = tokens.iter().find(|&&token| token as usize >= vocab_size) {
            return Err(EngineError::InvalidToken { token, vocab_size });
        }
        Ok(())
    }

    fn vocab_size(&self) -> usize {
        match &self.backend {
            SessionBackend::Cpu(model) => model.cfg.vocab_size,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            SessionBackend::Metal(model) => model.cfg.vocab_size,
        }
    }

    fn hidden_dim(&self) -> usize {
        match &self.backend {
            SessionBackend::Cpu(model) => model.cfg.hidden_dim,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            SessionBackend::Metal(model) => model.cfg.hidden_dim,
        }
    }

    fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>, EngineError> {
        let mut logits = Vec::new();
        self.forward_into(tokens, &mut logits)?;
        Ok(logits)
    }

    /// `forward` into a caller-owned buffer so repeated decode steps reuse one
    /// vocabulary-sized allocation.
    fn forward_into(&mut self, tokens: &[u32], logits: &mut Vec<f32>) -> Result<(), EngineError> {
        match &mut self.backend {
            SessionBackend::Cpu(model) => *logits = model.forward(tokens, None),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            SessionBackend::Metal(model) => model.forward_into(tokens, logits)?,
        }
        self.token_history.extend_from_slice(tokens);
        Ok(())
    }

    fn forward_embeddings(&mut self, embeddings: &[f32]) -> Result<Vec<f32>, EngineError> {
        match &mut self.backend {
            SessionBackend::Cpu(model) => Ok(model.forward_embeddings(embeddings, None)),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            SessionBackend::Metal(model) => Ok(model.forward_embeddings(embeddings)?),
        }
    }
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for index in 1..values.len() {
        if values[index] > values[best] {
            best = index;
        }
    }
    best
}

fn elapsed_ns(started: std::time::Instant) -> u64 {
    started.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

fn ensure_finite_logits(logits: &[f32]) -> Result<(), EngineError> {
    if logits.is_empty() || logits.iter().any(|value| !value.is_finite()) {
        return Err(EngineError::NonfiniteLogits);
    }
    Ok(())
}

fn greedy_verification_decision(
    starting_logits: &[f32],
    flat_logits: &[f32],
    draft_tokens: &[u32],
    vocab_size: usize,
) -> Result<VerificationResult, EngineError> {
    if starting_logits.len() != vocab_size
        || flat_logits.len() != draft_tokens.len().saturating_mul(vocab_size)
    {
        return Err(EngineError::InvalidRestoredLogits {
            actual: flat_logits.len(),
            expected: draft_tokens.len().saturating_mul(vocab_size),
        });
    }
    for (index, &draft) in draft_tokens.iter().enumerate() {
        let row = if index == 0 {
            starting_logits
        } else {
            &flat_logits[(index - 1) * vocab_size..index * vocab_size]
        };
        let target = argmax(row) as u32;
        if target != draft {
            return Ok(VerificationResult {
                accepted: index,
                replacement: Some(target),
            });
        }
    }
    Ok(VerificationResult {
        accepted: draft_tokens.len(),
        replacement: None,
    })
}

fn final_logit_row(values: &[f32], vocab_size: usize) -> Result<Vec<f32>, EngineError> {
    if vocab_size == 0 || values.len() < vocab_size || !values.len().is_multiple_of(vocab_size) {
        return Err(EngineError::InvalidRestoredLogits {
            actual: values.len(),
            expected: vocab_size,
        });
    }
    Ok(values[values.len() - vocab_size..].to_vec())
}

#[cfg(test)]
mod verification_tests {
    use super::greedy_verification_decision;

    fn row(winner: usize) -> Vec<f32> {
        let mut values = vec![-10.0; 32];
        values[winner] = 10.0;
        values
    }

    #[test]
    fn split_verifier_vlen15_full_accept_uses_pre_token_rows() {
        let drafts = (1..=15).collect::<Vec<u32>>();
        let mut flat = Vec::new();
        for winner in 2..=16 {
            flat.extend(row(winner));
        }
        let decision = greedy_verification_decision(&row(1), &flat, &drafts, 32).unwrap();
        assert_eq!(decision.accepted, drafts.len());
        assert_eq!(decision.replacement, None);
    }

    #[test]
    fn split_verifier_vlen15_rejects_first_draft_mismatch_without_off_by_one() {
        let mut drafts = (1..=15).collect::<Vec<u32>>();
        drafts[1] = 27;
        let mut flat = Vec::new();
        for winner in 2..=16 {
            flat.extend(row(winner));
        }
        let decision = greedy_verification_decision(&row(1), &flat, &drafts, 32).unwrap();
        assert_eq!(decision.accepted, 1);
        assert_eq!(decision.replacement, Some(2));
    }

    #[test]
    fn split_verifier_vlen15_rejection_replays_the_exact_replacement_stream() {
        let exact = (1..=16).collect::<Vec<u32>>();
        let mut candidates = (1..=15).collect::<Vec<u32>>();
        candidates[1] = 27;
        let mut first_logits = Vec::new();
        for winner in 2..=16 {
            first_logits.extend(row(winner));
        }
        let first = greedy_verification_decision(&row(1), &first_logits, &candidates, 32).unwrap();
        let mut generated = candidates[..first.accepted].to_vec();
        let replacement = first.replacement.unwrap();
        assert_eq!(replacement, exact[first.accepted]);

        let remainder = exact[first.accepted..].to_vec();
        let mut replay_logits = Vec::new();
        for winner in 3..=17 {
            replay_logits.extend(row(winner));
        }
        let replay = greedy_verification_decision(&row(2), &replay_logits, &remainder, 32).unwrap();
        generated.extend_from_slice(&remainder[..replay.accepted]);
        assert_eq!(generated, exact);
    }
}

#[cfg(test)]
mod delta_install_tests {
    use super::{copy_ring_prefix_rows, delta_prefix_span, RingOrigin};

    #[test]
    fn delta_span_requires_a_nonempty_suffix_and_a_held_prefix() {
        // cut 0 is the full install, never the delta path.
        assert!(delta_prefix_span(0, 300, 0, 0, 256).is_err());
        assert!(delta_prefix_span(0, 300, 300, 0, 300).is_err());
        assert!(delta_prefix_span(0, 300, 512, 0, 512).is_err());
        // The session cannot vouch for tokens it does not hold.
        assert!(delta_prefix_span(0, 300, 256, 0, 200).is_err());
    }

    #[test]
    fn delta_span_copies_up_to_the_cut_and_resumes_there() {
        // NoPE plane: origin 0, the whole prefix is copied, tiles resume at
        // the cut.
        let span = delta_prefix_span(0, 300, 256, 0, 256).unwrap();
        assert_eq!((span.copy_start, span.copy_end, span.resume), (0, 256, 256));
        // SWA plane with the window still covering the cut: copy from the
        // window origin, resume at the cut.
        let span = delta_prefix_span(52, 2100, 256, 0, 256).unwrap();
        assert_eq!(
            (span.copy_start, span.copy_end, span.resume),
            (52, 256, 256)
        );
        // Window slid past the cut: nothing to copy, the whole window
        // re-arrives as tiles and the cursor resumes at the origin.
        let span = delta_prefix_span(2052, 4100, 512, 4, 512).unwrap();
        assert_eq!(
            (span.copy_start, span.copy_end, span.resume),
            (2052, 2052, 2052)
        );
    }

    #[test]
    fn delta_span_fails_closed_when_the_live_ring_overwrote_the_origin() {
        // The delta needs [52, 256) but the live ring already slid to 100.
        assert!(delta_prefix_span(52, 2100, 256, 100, 300).is_err());
    }

    fn ring(capacity: usize, kv_dim: usize, origin_logical: usize, fill: u16) -> Vec<u16> {
        // A sequentially-built ring: logical row l lands at physical l %
        // capacity. Fill marks rows so a misplaced copy is visible.
        let mut plane = vec![0u16; capacity * kv_dim];
        let rotation = origin_logical % capacity;
        for logical_offset in 0..capacity {
            let physical = (rotation + logical_offset) % capacity;
            for element in 0..kv_dim {
                plane[physical * kv_dim + element] =
                    fill.wrapping_add((origin_logical + logical_offset) as u16);
            }
        }
        plane
    }

    #[test]
    fn prefix_copy_matches_a_sequentially_built_ring() {
        // Live ring wrapped: capacity 8, it holds logical [6, 14). The delta
        // plane starts at origin 10; the copy must land physical-identical to
        // a ring that grew there sequentially (ENG-003).
        let live = ring(8, 3, 6, 1000);
        let mut detached = vec![0u16; 8 * 3];
        copy_ring_prefix_rows(
            &live,
            &mut detached,
            8,
            3,
            3,
            false,
            RingOrigin {
                logical: 6,
                physical: 6,
            },
            RingOrigin {
                logical: 10,
                physical: 2,
            },
            10,
            14,
        )
        .unwrap();
        for logical in 10..14 {
            let physical = logical % 8;
            let expected = 1000u16.wrapping_add(logical as u16);
            assert_eq!(
                &detached[physical * 3..physical * 3 + 3],
                &[expected; 3],
                "logical row {logical}"
            );
        }
        // Rows outside the copied span stay untouched: physical rows {2,3,4,5}
        // were written, {0,1,6,7} must still be zero.
        assert!(detached[..2 * 3].iter().all(|&x| x == 0));
        assert_eq!(&detached[6 * 3..6 * 3 + 3], &[0; 3]);
        assert_eq!(&detached[7 * 3..7 * 3 + 3], &[0; 3]);
    }

    #[test]
    fn prefix_copy_supports_the_head_major_layout() {
        // Head-major planes scatter one logical row across per-head row
        // blocks; the copy must hit the same physical slot in each block.
        let capacity = 4;
        let kv_dim = 4;
        let head_dim = 2;
        let mut live = vec![0u16; capacity * kv_dim];
        let origin = RingOrigin {
            logical: 0,
            physical: 0,
        };
        for logical in 0..4 {
            for head in 0..kv_dim / head_dim {
                let offset = (head * capacity + logical) * head_dim;
                live[offset] = 100 * logical as u16 + head as u16;
                live[offset + 1] = 100 * logical as u16 + head as u16;
            }
        }
        let mut detached = vec![0u16; capacity * kv_dim];
        copy_ring_prefix_rows(
            &live,
            &mut detached,
            capacity,
            kv_dim,
            head_dim,
            true,
            origin,
            origin,
            0,
            4,
        )
        .unwrap();
        assert_eq!(detached, live);
    }

    #[test]
    fn prefix_copy_fails_closed_on_a_non_sequential_origin_pair() {
        let live = ring(8, 2, 6, 7);
        let mut detached = vec![0u16; 16];
        // physical != logical % capacity: not a sequentially-built ring.
        let result = copy_ring_prefix_rows(
            &live,
            &mut detached,
            8,
            2,
            2,
            false,
            RingOrigin {
                logical: 6,
                physical: 6,
            },
            RingOrigin {
                logical: 10,
                physical: 0,
            },
            10,
            14,
        );
        assert!(result.is_err());
        assert!(detached.iter().all(|&x| x == 0));
    }

    #[test]
    fn prefix_copy_rejects_ranges_outside_the_held_rows() {
        let live = ring(8, 2, 0, 7);
        let mut detached = vec![0u16; 16];
        let origin = RingOrigin {
            logical: 0,
            physical: 0,
        };
        assert!(
            copy_ring_prefix_rows(&live, &mut detached, 8, 2, 2, false, origin, origin, 0, 9)
                .is_err()
        );
        let mut wrong_size = vec![0u16; 15];
        assert!(copy_ring_prefix_rows(
            &live,
            &mut wrong_size,
            8,
            2,
            2,
            false,
            origin,
            origin,
            0,
            1
        )
        .is_err());
    }
}
