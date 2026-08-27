use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::sampling::{
    distribution_ordered, sample_distribution_mt_ordered, verify_full_speculative_mt_ordered,
    Mt19937, SamplingParams, SpeculativeDecision,
};
use crate::{EngineError, Model, PrefillBatch, Session};

use super::{
    DFlashContextKvCache, DFlashError, DFlashForward, DFlashHiddenCache, DFlashKvCache,
    DFlashProjectionBackend, DFlashWeights,
};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DFlashSpecStats {
    /// Prompt/boundary target prefill, DFlash context preparation, and the
    /// first exact speculative round required before any token can be
    /// published. Draft/verify counters retain that first round too: phase
    /// telemetry is nested work attribution, not a partition of wall time.
    pub prefill_ns: u64,
    pub rounds: usize,
    pub drafted_tokens: usize,
    pub accepted_draft_tokens: usize,
    /// Assistant proposals actually submitted to target verification, in
    /// deterministic round order. Qualification compares this trace exactly.
    pub draft_token_trace: Vec<u32>,
    /// Accepted assistant proposals only (the target-owned seed is excluded).
    pub accepted_prefix_trace: Vec<u32>,
    /// Accepted assistant proposal count for every verification round.
    pub accepted_prefix_counts: Vec<usize>,
    /// Proposals submitted in every verification round, parallel to
    /// `accepted_prefix_counts`. Backs the windowed disable decision.
    pub drafted_counts: Vec<usize>,
    /// First round index the decision window may reach. Advances when the
    /// gate reopens so re-qualification judges fresh evidence only.
    pub window_floor: usize,
    /// Committed-token count at which a disabled request may speculate again.
    pub requalify_at_tokens: usize,
    /// Gate closures this request. Backs the re-qualification backoff.
    pub disable_events: usize,
    pub committed_tokens: usize,
    pub target_only_fallback_tokens: usize,
    /// Effective draft context geometry this request ran with, stamped from
    /// the live cache so every receipt self-identifies. Conditioning the
    /// draft on the wrong window silently invalidated a whole campaign's
    /// spec numbers (2026-08-21); the geometry now travels with the result.
    pub draft_sink_size: usize,
    pub draft_sliding_window: usize,
    pub speculation_disabled: bool,
    /// Committed-token count at the instant `should_disable_speculation`
    /// first tripped for this request. `None` while speculation has never
    /// been disabled; set exactly once, the same round `speculation_disabled`
    /// flips to `true`.
    pub speculation_disabled_at_tokens: Option<usize>,
    pub last_accepted_run: usize,
    /// Target forward batches submitted for verification and prefix commit.
    /// Every single-pass round submits one; the legacy sampled replay route
    /// submits two, because it re-runs the committed prefix after a full
    /// rollback.
    pub target_batches: usize,
    /// Assistant work including target embedding and target LM-head projection.
    pub draft_ns: u64,
    /// Full target verification, accepted-prefix capture, and rollback work.
    pub target_verify_ns: u64,
    /// Target-only adaptive-fallback decode work.
    pub fallback_target_ns: u64,
    /// Mirror-SD attempts launched after the final exact target capture layer.
    pub mirror_overlap_attempts: usize,
    /// Provisional rounds retained because all candidates and the predicted
    /// target bonus matched exactly.
    pub mirror_overlap_commits: usize,
    /// Provisional rounds discarded and replayed from the CPU KV shadow.
    pub mirror_overlap_rollbacks: usize,
    /// Requests which stopped attempting conditional overlap after the first
    /// miss. A miss advances public Core ML's private MLState provisionally;
    /// the next exact draft must replay the authoritative CPU shadow. Repeating
    /// that growing-context replay is more expensive than the bounded overlap
    /// can hide, so the circuit stays open for the rest of the request.
    pub mirror_overlap_circuit_breaks: usize,
    /// Wall time spent producing provisional ANE drafts. Most or all of this
    /// should be hidden by the target suffix on a successful implementation.
    pub mirror_overlap_draft_ns: u64,
    /// End-to-end wall time of conditional-overlap attempts.
    pub mirror_overlap_wall_ns: u64,
    /// Conservative work hidden by overlap: target time plus provisional
    /// draft time minus observed attempt wall time.
    pub mirror_overlap_hidden_ns: u64,
    /// Public-CoreML FC-slice prediction wall time inside the staged target
    /// verifier. This was previously invisible in both `draft_ns` and
    /// `target_verify_ns`, obscuring the dominant v8 overlap cost.
    pub mirror_capture_fc_ns: u64,
    /// Diagnostic per-cycle trace, populated only when
    /// `MUSER_DFLASH_CYCLE_TRACE=1` was set at generation start. Empty
    /// otherwise; never consulted by any route or acceptance decision.
    pub cycle_trace: Vec<DFlashCycleTrace>,
}

/// Diagnostic economics for one speculative verification cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct DFlashCycleTrace {
    /// Assistant block forward including embedding and LM-head projection.
    pub draft_ns: u64,
    /// Target verification batch, acceptance decision, and prefix commit.
    pub verify_ns: u64,
    /// Whole loop iteration; the residual over draft+verify is host work
    /// (context reads, candidate assembly, commit callback).
    pub cycle_ns: u64,
    /// Assistant proposals submitted to the verifier this cycle.
    pub drafted: usize,
    /// Target-accepted assistant proposals this cycle (excludes the seed).
    pub accepted: usize,
}

/// Verification rounds the disable decision looks at. Recent evidence only.
const DISABLE_WINDOW_ROUNDS: usize = 8;
/// Rounds excluded from the decision. The draft conditions on target hidden
/// states, and the rounds immediately after prefill are the coldest of the
/// request; judging the whole response on them is what produced the
/// 2026-08-21 natural-text collapse.
const DISABLE_WARMUP_ROUNDS: usize = 2;
/// Proposals the window must hold before it can close the gate. Preserves
/// the original sample-size intent, now scoped to the window.
const DISABLE_MIN_WINDOW_DRAFTS: usize = 32;
/// Windowed acceptance below which another five-layer draft round cannot
/// amortize.
const DISABLE_ACCEPTANCE_FLOOR: f64 = 0.25;
/// Committed tokens before a disabled request may speculate again.
const REQUALIFY_BASE_TOKENS: usize = 64;
/// Consecutive disables double the cooldown: content that genuinely drafts
/// badly stops retrying quickly, content that was merely cold recovers fast.
const REQUALIFY_MAX_SHIFT: usize = 3;
const REQUALIFY_MAX_TOKENS: usize = 512;

/// Diagnostic-only override for the draft context sink/window. Absent env
/// keeps the shipped geometry.
fn diag_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

pub(crate) fn cycle_trace_enabled() -> bool {
    std::env::var_os("MUSER_DFLASH_CYCLE_TRACE").as_deref() == Some(std::ffi::OsStr::new("1"))
}

/// Diagnostic-only kill switch for the acceptance gate, used to measure the
/// gate against the draft it actually runs with. The gate is ON unless
/// `MUSER_DFLASH_GATE=off` is set explicitly; any other value keeps it on.
fn acceptance_gate_enabled() -> bool {
    std::env::var_os("MUSER_DFLASH_GATE").as_deref() != Some(std::ffi::OsStr::new("off"))
}

impl DFlashSpecStats {
    pub fn acceptance_rate(&self) -> f64 {
        if self.drafted_tokens == 0 {
            0.0
        } else {
            self.accepted_draft_tokens as f64 / self.drafted_tokens as f64
        }
    }

    /// Acceptance over the recent decision window, or `None` while the
    /// window holds too few proposals to judge.
    pub fn window_acceptance(&self) -> Option<f64> {
        let len = self.accepted_prefix_counts.len();
        let start = len
            .saturating_sub(DISABLE_WINDOW_ROUNDS)
            .max(self.window_floor);
        if start >= len {
            return None;
        }
        let drafted: usize = self.drafted_counts[start..].iter().sum();
        if drafted < DISABLE_MIN_WINDOW_DRAFTS {
            return None;
        }
        let accepted: usize = self.accepted_prefix_counts[start..].iter().sum();
        Some(accepted as f64 / drafted as f64)
    }

    fn should_disable_speculation(&self) -> bool {
        // The decision is made on RECENT evidence, never on the cumulative
        // rate: a cumulative rate cannot recover once drafting stops, which
        // made the old gate a one-way latch (2026-08-21 root cause). The
        // first rounds after prefill run with the coldest draft
        // conditioning and are excluded from the sample.
        if self.rounds <= DISABLE_WARMUP_ROUNDS {
            return false;
        }
        self.window_acceptance()
            .is_some_and(|rate| rate < DISABLE_ACCEPTANCE_FLOOR)
    }

    /// Close the gate when recent acceptance cannot amortize another
    /// five-layer draft round. Call once per completed verification round.
    pub(crate) fn update_speculation_gate(&mut self) {
        if !acceptance_gate_enabled() {
            return;
        }
        if self.speculation_disabled || !self.should_disable_speculation() {
            return;
        }
        self.speculation_disabled = true;
        self.disable_events += 1;
        if self.speculation_disabled_at_tokens.is_none() {
            self.speculation_disabled_at_tokens = Some(self.committed_tokens);
        }
        let shift = (self.disable_events - 1).min(REQUALIFY_MAX_SHIFT);
        let cooldown = (REQUALIFY_BASE_TOKENS << shift).min(REQUALIFY_MAX_TOKENS);
        self.requalify_at_tokens = self.committed_tokens + cooldown;
    }

    /// True when this round must fall back to plain target decode. Reopens
    /// the gate once the cooldown has elapsed, so a request whose opening
    /// rounds drafted badly can re-qualify on fresh evidence instead of
    /// serving the remaining response one token at a time.
    pub(crate) fn fallback_round(&mut self) -> bool {
        if !self.speculation_disabled {
            return false;
        }
        if self.committed_tokens >= self.requalify_at_tokens {
            self.speculation_disabled = false;
            self.window_floor = self.accepted_prefix_counts.len();
            return false;
        }
        true
    }
}

pub struct DFlashAssistant {
    forward: DFlashForward,
    context_cache: DFlashContextKvCache,
    draft_cache: DFlashKvCache,
}

/// Fully allocated and validated DFlash state. Committing this handle only
/// swaps owned caches and cannot fail.
pub struct PreparedDFlashContext {
    context_cache: DFlashContextKvCache,
    draft_cache: DFlashKvCache,
}

/// Opaque, single-use Metal preparation for a remote verifier's candidate
/// target-feature rows. It contains no authoritative token decision: only the
/// later exact committed-prefix count and carried frontier may finish it.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[doc(hidden)]
#[derive(Debug)]
#[must_use = "a prepared target context must be finished or explicitly discarded"]
pub struct PreparedDFlashTargetContext {
    inner: crate::metal::dflash::PreparedMetalDFlashContext,
    candidate_rows: usize,
}

/// Opaque rollback-capable DFlash append produced from a predicted carried
/// frontier while the remote target is still finishing its signed decision.
/// It becomes authoritative only through `ProvisionalDFlashTargetContext::resolve`.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[doc(hidden)]
#[must_use = "a provisional target context must be resolved or rolled back"]
pub struct ProvisionalDFlashTargetContext<'a> {
    assistant: &'a mut DFlashAssistant,
    checkpoint: Option<super::cache::DFlashContextCheckpoint>,
    predicted_frontier: u32,
    committed_context_rows: usize,
    drafts: Vec<u32>,
}

/// Target fields admitted only after the caller has authenticated and fully
/// validated the remote verifier result. The constructor is the explicit
/// protocol-to-engine trust boundary; the engine additionally checks that the
/// decision proves the complete provisional append and predicted frontier.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuthenticatedDFlashTargetDecision {
    frontier: u32,
    committed_context_rows: usize,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl AuthenticatedDFlashTargetDecision {
    /// Construct only after validating the result authentication, request and
    /// parent identity, candidate commitment, transition head, and hidden-row
    /// commitment. This type deliberately does not perform wire authentication.
    #[doc(hidden)]
    pub fn from_authenticated_fields(frontier: u32, committed_context_rows: usize) -> Self {
        Self {
            frontier,
            committed_context_rows,
        }
    }
}

/// Result of reconciling a private Mirror-SD append with an authenticated
/// target decision. Only `Committed` contains drafts safe to submit.
#[cfg(all(target_os = "macos", feature = "metal"))]
#[doc(hidden)]
#[derive(Debug, PartialEq, Eq)]
#[must_use = "a provisional DFlash resolution must be handled"]
pub enum ProvisionalDFlashResolution {
    Committed(Vec<u32>),
    RolledBack,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl ProvisionalDFlashTargetContext<'_> {
    /// Retain the provisional append only when a separately authenticated
    /// target result proves the entire predicted transition. A mismatch is
    /// rolled back before this exclusive assistant borrow is released.
    #[doc(hidden)]
    pub fn resolve(
        mut self,
        decision: AuthenticatedDFlashTargetDecision,
    ) -> Result<ProvisionalDFlashResolution, DFlashRunError> {
        if mirror_commit_is_exact(
            decision.committed_context_rows,
            self.committed_context_rows,
            decision.frontier,
            self.predicted_frontier,
            true,
        ) {
            let completed = self
                .checkpoint
                .as_ref()
                .ok_or_else(|| {
                    DFlashRunError::Invariant(
                        "provisional DFlash resolution lost its checkpoint".into(),
                    )
                })
                .and_then(|checkpoint| {
                    self.assistant
                        .context_cache
                        .validate_completed_append(checkpoint)
                        .map_err(|error| {
                            DFlashRunError::Invariant(format!(
                                "provisional DFlash append was not complete: {error}"
                            ))
                        })
                });
            if let Err(error) = completed {
                self.rollback_inner()?;
                return Err(error);
            }
            self.checkpoint.take();
            return Ok(ProvisionalDFlashResolution::Committed(std::mem::take(
                &mut self.drafts,
            )));
        }
        self.rollback_inner()?;
        Ok(ProvisionalDFlashResolution::RolledBack)
    }

    /// Explicitly abandon an unresolved prediction. Dropping the handle has
    /// the same fail-closed effect, but this names transport/auth failures.
    #[doc(hidden)]
    pub fn rollback(mut self) -> Result<(), DFlashRunError> {
        self.rollback_inner()
    }

    fn rollback_inner(&mut self) -> Result<(), DFlashRunError> {
        if let Some(checkpoint) = self.checkpoint.take() {
            self.assistant.forward.invalidate_target_context_split();
            if let Err(error) = self.assistant.context_cache.rollback_append(checkpoint) {
                // A failed rollback proof means the derived cache cannot be
                // trusted. Discard it rather than releasing a predicted state
                // as though it were authoritative.
                self.assistant.reset();
                return Err(DFlashRunError::Invariant(format!(
                    "provisional DFlash rollback was quarantined: {error}"
                )));
            }
        }
        Ok(())
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl Drop for ProvisionalDFlashTargetContext<'_> {
    fn drop(&mut self) {
        let _ = self.rollback_inner();
    }
}

/// Stop set for one generation: the assistant's own `eos_token_id` plus the
/// caller's extra IDs. An empty extra list is EOS-only behaviour.
#[derive(Clone, Copy)]
struct StopSet<'a> {
    eos: u32,
    extra: &'a [u32],
}

impl StopSet<'_> {
    fn contains(&self, token: u32) -> bool {
        token == self.eos || self.extra.contains(&token)
    }

    /// Cut the returned stream at the first stop token. Every speculative
    /// route commits whole accepted prefixes, so committed tokens can sit
    /// behind a stop; those are never returned to the caller.
    fn truncate(&self, generated: &mut Vec<u32>, stats: &mut DFlashSpecStats) -> bool {
        let Some(index) = generated.iter().position(|&token| self.contains(token)) else {
            return false;
        };
        generated.truncate(index + 1);
        stats.committed_tokens = generated.len();
        true
    }

    /// Never submit candidates behind a known terminal token to the target.
    /// This makes the target KV, DFlash hidden frontier, returned tokens, and
    /// stream frontier one combined prefix instead of rolling target state
    /// past an invisible speculative suffix.
    fn truncate_candidates(&self, candidates: &mut Vec<u32>) -> bool {
        let Some(index) = candidates.iter().position(|&token| self.contains(token)) else {
            return false;
        };
        candidates.truncate(index + 1);
        true
    }
}

/// Opaque target/DFlash state prepared from one exact prompt.  Preparation is
/// deliberately separate from generation so qualification can time decode
/// without allowing prompt prefill to hide either a gain or a regression.
pub struct DFlashPreparedGreedy {
    seed: u32,
    hidden_cache: DFlashHiddenCache,
    prefill_ns: u64,
}

/// Opaque sampled target/DFlash state prepared from one exact prompt. The RNG
/// draw selecting the first proposal is part of preparation and therefore
/// happens before a caller atomically publishes the paired target+DFlash cut.
pub struct DFlashPreparedSampled {
    seed: u32,
    hidden_cache: DFlashHiddenCache,
    prefill_ns: u64,
}

impl DFlashAssistant {
    fn exact_mirror_overlap_enabled(&self) -> bool {
        cfg!(all(target_os = "macos", feature = "metal"))
            && std::env::var_os("MUSER_DFLASH_MIRROR_OVERLAP").as_deref()
                == Some(std::ffi::OsStr::new("1"))
            && self
                .forward
                .projection_backend
                .as_deref()
                .is_some_and(DFlashProjectionBackend::supports_exact_mirror_overlap)
    }

    fn capture_fc_pipeline_backend(&self) -> Option<Arc<dyn DFlashProjectionBackend>> {
        (std::env::var_os("MUSER_DFLASH_CAPTURE_FC_PIPELINE").as_deref()
            == Some(std::ffi::OsStr::new("1")))
        .then(|| self.forward.projection_backend.clone())
        .flatten()
        .filter(|backend| backend.supports_capture_fc_pipeline())
    }

    pub fn load(path: &Path, target: &Model) -> Result<Self, DFlashError> {
        let weights = DFlashWeights::load(path)?;
        Self::from_weights(weights, target, None)
    }

    pub fn load_with_projection_backend(
        path: &Path,
        target: &Model,
        backend: Arc<dyn DFlashProjectionBackend>,
    ) -> Result<Self, DFlashError> {
        let weights = DFlashWeights::load_with_external_projections(path)?;
        Self::from_weights(weights, target, Some(backend))
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn load_metal(path: &Path, target: &Model) -> Result<Self, DFlashError> {
        let weights = DFlashWeights::load_metal(path)?;
        let metal = crate::metal::dflash::MetalDFlashForward::new(&weights, 131_072)
            .map_err(|error| DFlashError::Projection(error.to_string()))?;
        let mut assistant = Self::from_weights(weights, target, None)?;
        assistant.forward = assistant.forward.with_metal_forward(metal);
        Ok(assistant)
    }

    fn from_weights(
        weights: DFlashWeights,
        target: &Model,
        backend: Option<Arc<dyn DFlashProjectionBackend>>,
    ) -> Result<Self, DFlashError> {
        let c = &weights.config;
        if c.hidden_size != target.config().hidden_dim
            || c.vocab_size != target.config().vocab_size
            || c.num_target_layers != target.config().n_layers
        {
            return Err(DFlashError::Config(format!(
                "assistant identity h/vocab/layers={}/{}/{} does not match target {}/{}/{}",
                c.hidden_size,
                c.vocab_size,
                c.num_target_layers,
                target.config().hidden_dim,
                target.config().vocab_size,
                target.config().n_layers
            )));
        }
        let geometry = c.context_geometry();
        let kv = geometry.elements_per_token;
        let layers = geometry.layers;
        let mut forward = DFlashForward::new(weights);
        if let Some(backend) = backend {
            forward = forward.with_projection_backend(backend);
        }
        Ok(Self {
            forward,
            context_cache: DFlashContextKvCache::new(
                layers,
                kv,
                diag_usize("MUSER_DFLASH_SINK", geometry.sink_size),
                diag_usize("MUSER_DFLASH_WINDOW", geometry.window_size),
            ),
            draft_cache: DFlashKvCache::new(layers, kv),
        })
    }

    pub fn reset(&mut self) {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        self.forward.invalidate_target_context_split();
        let c = self.forward.config();
        let geometry = c.context_geometry();
        let kv = geometry.elements_per_token;
        self.context_cache = DFlashContextKvCache::new(
            geometry.layers,
            kv,
            diag_usize("MUSER_DFLASH_SINK", geometry.sink_size),
            diag_usize("MUSER_DFLASH_WINDOW", geometry.window_size),
        );
        self.draft_cache = DFlashKvCache::new(c.num_hidden_layers, kv);
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn rollback_mirror_append(
        &mut self,
        checkpoint: super::cache::DFlashContextCheckpoint,
    ) -> Result<(), DFlashRunError> {
        if let Err(error) = self.context_cache.rollback_append(checkpoint) {
            self.reset();
            return Err(DFlashRunError::Invariant(format!(
                "Mirror-SD rollback was quarantined: {error}"
            )));
        }
        Ok(())
    }

    #[doc(hidden)]
    pub fn export_context_snapshot(&self) -> super::DFlashContextSnapshot {
        self.context_cache.snapshot()
    }

    /// Actual cache geometry, including an explicitly requested diagnostic
    /// override. Enrollment comparison therefore cannot mistake a diagnostic
    /// cache for the sidecar-declared release identity.
    pub fn context_geometry(&self) -> super::DFlashContextGeometry {
        self.context_cache.geometry()
    }

    #[doc(hidden)]
    pub fn install_context_snapshot(
        &mut self,
        snapshot: &super::DFlashContextSnapshot,
    ) -> Result<(), DFlashError> {
        let prepared = self.prepare_context_snapshot(snapshot)?;
        self.commit_prepared_context(prepared);
        Ok(())
    }

    #[doc(hidden)]
    pub fn prepare_context_snapshot(
        &self,
        snapshot: &super::DFlashContextSnapshot,
    ) -> Result<PreparedDFlashContext, DFlashError> {
        let context_cache = self
            .context_cache
            .prepare_snapshot(snapshot)
            .map_err(DFlashError::Config)?;
        let config = self.forward.config();
        let width = config.num_key_value_heads * config.head_dim;
        Ok(PreparedDFlashContext {
            context_cache,
            draft_cache: DFlashKvCache::new(config.num_hidden_layers, width),
        })
    }

    #[doc(hidden)]
    pub fn commit_prepared_context(&mut self, prepared: PreparedDFlashContext) {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        self.forward.invalidate_target_context_split();
        self.context_cache = prepared.context_cache;
        self.draft_cache = prepared.draft_cache;
    }

    #[doc(hidden)]
    pub fn validate_context_snapshot(
        &self,
        snapshot: &super::DFlashContextSnapshot,
    ) -> Result<(), DFlashError> {
        self.context_cache
            .validate_snapshot_identity(snapshot)
            .map_err(DFlashError::Config)
    }

    /// Execute the seed-independent half of a future greedy DFlash round from
    /// target features streamed before the remote verifier's terminal
    /// decision. `candidate_rows` may include a rejected suffix; preparation
    /// does not mutate the authoritative DFlash context cache.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[doc(hidden)]
    pub fn prepare_target_context_split(
        &mut self,
        target_hidden: &[f32],
        candidate_rows: usize,
    ) -> Result<PreparedDFlashTargetContext, DFlashRunError> {
        let inner = self.forward.prepare_target_context_split(
            target_hidden,
            candidate_rows,
            &self.context_cache,
        )?;
        Ok(PreparedDFlashTargetContext {
            inner,
            candidate_rows,
        })
    }

    /// Finish a split DFlash round only after the verifier authenticates the
    /// exact committed prefix and next carried frontier. Only
    /// `committed_context_rows` from the prepared prefix are attended to and
    /// appended; every prepared row after that boundary is discarded.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[doc(hidden)]
    pub fn finish_prepared_target_context_greedy(
        &mut self,
        target: &Model,
        projection_session: &mut Session,
        prepared: PreparedDFlashTargetContext,
        exact_frontier: u32,
        committed_context_rows: usize,
        verify_length: usize,
    ) -> Result<Vec<u32>, DFlashRunError> {
        if !matches!(verify_length, 3 | 7 | 15) {
            self.forward.discard_target_context_split(prepared.inner)?;
            return Err(DFlashRunError::InvalidVerifyLength(verify_length));
        }
        let c = self.forward.config().clone();
        let mut ids = vec![c.dflash_config.mask_token_id; c.block_size];
        ids[0] = exact_frontier;
        let embeddings = match target.embed_tokens(&ids) {
            Ok(embeddings) => embeddings,
            Err(error) => {
                self.forward.discard_target_context_split(prepared.inner)?;
                return Err(error.into());
            }
        };
        let output = self.forward.finish_target_context_split(
            prepared.inner,
            &embeddings,
            committed_context_rows,
            &mut self.context_cache,
        )?;
        let top = projection_session.project_dflash_argmax(target, &output)?;
        Ok(top[1..=verify_length.min(output.n_draft_tokens)].to_vec())
    }

    /// Speculatively finish a prepared round using the assistant's prediction
    /// of the remote target bonus. The append is locally visible only inside
    /// this assistant and carries a bounded rollback record; callers must not
    /// publish its drafts until the authenticated target result resolves it.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[doc(hidden)]
    pub fn finish_prepared_target_context_provisionally_greedy<'a>(
        &'a mut self,
        target: &Model,
        projection_session: &mut Session,
        prepared: PreparedDFlashTargetContext,
        predicted_frontier: u32,
        verify_length: usize,
    ) -> Result<ProvisionalDFlashTargetContext<'a>, DFlashRunError> {
        let committed_context_rows = prepared.candidate_rows;
        let checkpoint = self.context_cache.checkpoint_append(committed_context_rows);
        match self.finish_prepared_target_context_greedy(
            target,
            projection_session,
            prepared,
            predicted_frontier,
            committed_context_rows,
            verify_length,
        ) {
            Ok(drafts) => Ok(ProvisionalDFlashTargetContext {
                assistant: self,
                checkpoint: Some(checkpoint),
                predicted_frontier,
                committed_context_rows,
                drafts,
            }),
            Err(error) => {
                self.forward.invalidate_target_context_split();
                if let Err(rollback) = self.context_cache.rollback_append(checkpoint) {
                    self.reset();
                    return Err(DFlashRunError::Invariant(format!(
                        "provisional DFlash finish failed ({error}); rollback was quarantined: {rollback}"
                    )));
                }
                Err(error)
            }
        }
    }

    /// Abandon early target-feature work when the verifier closes without a
    /// next frontier (EOS, max-token closure, cancellation, or protocol error).
    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[doc(hidden)]
    pub fn discard_prepared_target_context(
        &mut self,
        prepared: PreparedDFlashTargetContext,
    ) -> Result<(), DFlashRunError> {
        self.forward.discard_target_context_split(prepared.inner)?;
        Ok(())
    }

    pub fn draft_greedy(
        &mut self,
        target: &Model,
        seed: u32,
        target_hidden: &[f32],
        n_ctx: usize,
        verify_length: usize,
    ) -> Result<Vec<u32>, DFlashRunError> {
        let c = self.forward.config().clone();
        if !matches!(verify_length, 3 | 7 | 15) {
            return Err(DFlashRunError::InvalidVerifyLength(verify_length));
        }
        let mut ids = vec![c.dflash_config.mask_token_id; c.block_size];
        ids[0] = seed;
        let embeddings = target.embed_tokens(&ids)?;
        let output = self.forward.forward(
            &embeddings,
            target_hidden,
            n_ctx,
            &mut self.context_cache,
            &mut self.draft_cache,
        )?;
        let logits = target.project_hidden(&output.hidden_states)?;
        if cycle_trace_enabled() {
            // Row 0 is the only block row that saw a real token. A correctly
            // conditioned block head must reconstruct the seed there; if it
            // does not, the conditioning carries no usable signal.
            let echoed = argmax(&logits[..c.vocab_size]) as u32;
            eprintln!(
                "dflash-seed-echo seed={} row0={} match={}",
                seed,
                echoed,
                echoed == seed
            );
        }
        Ok((1..=verify_length.min(output.n_draft_tokens))
            .map(|position| {
                argmax(&logits[position * c.vocab_size..(position + 1) * c.vocab_size]) as u32
            })
            .collect())
    }

    /// Draft greedily while using an already resident target session only for
    /// its accelerator LM-head projection.  The target session's token/KV
    /// state is deliberately untouched: this is the remote-verifier boundary
    /// where authoritative target state lives on another host, while the Mac
    /// still owns DFlash and the matching target embedding/output weights.
    #[doc(hidden)]
    pub fn draft_greedy_with_session_projection(
        &mut self,
        target: &Model,
        projection_session: &mut Session,
        seed: u32,
        target_hidden: &[f32],
        n_ctx: usize,
        verify_length: usize,
    ) -> Result<Vec<u32>, DFlashRunError> {
        self.draft_greedy_for_session(
            target,
            projection_session,
            seed,
            target_hidden,
            n_ctx,
            verify_length,
        )
    }

    /// Append authoritative target rows to DFlash's context cache without
    /// projecting or retaining any draft proposal.  Context K/V is derived
    /// solely from `target_hidden`; the mask-token noise block is required by
    /// the fixed DFlash forward ABI but cannot influence those appended rows.
    /// Remote-target clients use this to reconstruct a prompt before opening
    /// a verifier session, so a local setup failure cannot consume remote
    /// target state.
    #[doc(hidden)]
    pub fn prime_target_context(
        &mut self,
        target: &Model,
        target_hidden: &[f32],
        n_ctx: usize,
    ) -> Result<(), DFlashRunError> {
        let c = self.forward.config().clone();
        let ids = vec![c.dflash_config.mask_token_id; c.block_size];
        let embeddings = target.embed_tokens(&ids)?;
        self.forward.forward(
            &embeddings,
            target_hidden,
            n_ctx,
            &mut self.context_cache,
            &mut self.draft_cache,
        )?;
        Ok(())
    }

    fn draft_greedy_for_session(
        &mut self,
        target: &Model,
        session: &mut Session,
        seed: u32,
        target_hidden: &[f32],
        n_ctx: usize,
        verify_length: usize,
    ) -> Result<Vec<u32>, DFlashRunError> {
        let c = self.forward.config().clone();
        if !matches!(verify_length, 3 | 7 | 15) {
            return Err(DFlashRunError::InvalidVerifyLength(verify_length));
        }
        let mut ids = vec![c.dflash_config.mask_token_id; c.block_size];
        ids[0] = seed;
        let trace = cycle_trace_enabled();
        let embed_started = Instant::now();
        let embeddings = target.embed_tokens(&ids)?;
        let embed_ns = embed_started.elapsed().as_nanos() as u64;
        let forward_started = Instant::now();
        let output = self.forward.forward(
            &embeddings,
            target_hidden,
            n_ctx,
            &mut self.context_cache,
            &mut self.draft_cache,
        )?;
        let forward_ns = forward_started.elapsed().as_nanos() as u64;
        let project_started = Instant::now();
        let top = session.project_dflash_argmax(target, &output)?;
        if trace {
            eprintln!(
                "dflash-cycle-draft embed_ns={} forward_ns={} project_ns={}",
                embed_ns,
                forward_ns,
                project_started.elapsed().as_nanos() as u64,
            );
        }
        if trace {
            // Row 0 is the only block row that saw a real token. A correctly
            // conditioned block head must reconstruct the seed there; if it
            // does not, the conditioning carries no usable signal.
            eprintln!(
                "dflash-seed-echo seed={} row0={} match={} drafts={:?}",
                seed,
                top[0],
                top[0] == seed,
                &top[1..top.len().min(6)]
            );
        }
        Ok(top[1..=verify_length.min(output.n_draft_tokens)].to_vec())
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[allow(clippy::too_many_arguments)]
    fn draft_greedy_from_capture_projection(
        &mut self,
        target: &Model,
        session: &mut Session,
        seed: u32,
        target_hidden: &[f32],
        target_projection: &[f32],
        n_ctx: usize,
        verify_length: usize,
    ) -> Result<Vec<u32>, DFlashRunError> {
        let c = self.forward.config().clone();
        let mut ids = vec![c.dflash_config.mask_token_id; c.block_size];
        ids[0] = seed;
        let embeddings = target.embed_tokens(&ids)?;
        let output = self.forward.forward_with_target_projection(
            &embeddings,
            target_hidden,
            n_ctx,
            &mut self.context_cache,
            &mut self.draft_cache,
            Some(target_projection),
        )?;
        let top = session.project_dflash_argmax(target, &output)?;
        Ok(top[1..=verify_length.min(output.n_draft_tokens)].to_vec())
    }

    /// Draft sampled tokens together with the exact full-vocabulary proposal
    /// distributions used to draw them.  Retaining q is required for the
    /// `max(p-q, 0)` rejection branch; a selected-token-only approximation is
    /// not valid speculative sampling.
    #[allow(clippy::too_many_arguments)]
    pub fn draft_sampled(
        &mut self,
        target: &Model,
        seed: u32,
        target_hidden: &[f32],
        n_ctx: usize,
        verify_length: usize,
        params: SamplingParams,
        rng: &mut Mt19937,
    ) -> Result<(Vec<u32>, Vec<Vec<f32>>), DFlashRunError> {
        let c = self.forward.config().clone();
        if !matches!(verify_length, 3 | 7 | 15) {
            return Err(DFlashRunError::InvalidVerifyLength(verify_length));
        }
        let mut ids = vec![c.dflash_config.mask_token_id; c.block_size];
        ids[0] = seed;
        let embeddings = target.embed_tokens(&ids)?;
        let output = self.forward.forward(
            &embeddings,
            target_hidden,
            n_ctx,
            &mut self.context_cache,
            &mut self.draft_cache,
        )?;
        let logits = target.project_hidden(&output.hidden_states)?;
        let count = verify_length.min(output.n_draft_tokens);
        let mut tokens = Vec::with_capacity(count);
        let mut probabilities = Vec::with_capacity(count);
        for position in 1..=count {
            let row = distribution_ordered(
                &logits[position * c.vocab_size..(position + 1) * c.vocab_size],
                params,
            )?;
            tokens.push(sample_distribution_mt_ordered(
                &row.weights,
                &row.order,
                rng,
            )?);
            probabilities.push(row.probabilities);
        }
        Ok((tokens, probabilities))
    }

    #[allow(clippy::too_many_arguments)]
    fn draft_sampled_for_session(
        &mut self,
        target: &Model,
        session: &mut Session,
        seed: u32,
        target_hidden: &[f32],
        n_ctx: usize,
        verify_length: usize,
        params: SamplingParams,
        rng: &mut Mt19937,
    ) -> Result<(Vec<u32>, Vec<Vec<f32>>), DFlashRunError> {
        let c = self.forward.config().clone();
        if !matches!(verify_length, 3 | 7 | 15) {
            return Err(DFlashRunError::InvalidVerifyLength(verify_length));
        }
        let mut ids = vec![c.dflash_config.mask_token_id; c.block_size];
        ids[0] = seed;
        let embeddings = target.embed_tokens(&ids)?;
        let output = self.forward.forward(
            &embeddings,
            target_hidden,
            n_ctx,
            &mut self.context_cache,
            &mut self.draft_cache,
        )?;
        let logits = session.project_dflash_output(target, &output)?;
        let count = verify_length.min(output.n_draft_tokens);
        let mut tokens = Vec::with_capacity(count);
        let mut probabilities = Vec::with_capacity(count);
        for position in 1..=count {
            let row = distribution_ordered(
                &logits[position * c.vocab_size..(position + 1) * c.vocab_size],
                params,
            )?;
            tokens.push(sample_distribution_mt_ordered(
                &row.weights,
                &row.order,
                rng,
            )?);
            probabilities.push(row.probabilities);
        }
        Ok((tokens, probabilities))
    }

    /// Exact greedy speculative generation. Target state changes only through
    /// verified prefixes; a rejected suffix is restored before the next round.
    pub fn generate_greedy(
        &mut self,
        target: &Model,
        session: &mut Session,
        prompt: &[u32],
        max_new_tokens: usize,
        verify_length: usize,
        extra_stop_tokens: &[u32],
    ) -> Result<(Vec<u32>, DFlashSpecStats), DFlashRunError> {
        let prepared = self.prepare_greedy(target, session, prompt)?;
        self.generate_prepared_greedy(
            target,
            session,
            prepared,
            max_new_tokens,
            verify_length,
            extra_stop_tokens,
        )
    }

    /// Reset and prefill an exact prompt while capturing the target layers
    /// consumed by DFlash.  No output token is committed by this operation.
    pub fn prepare_greedy(
        &mut self,
        target: &Model,
        session: &mut Session,
        prompt: &[u32],
    ) -> Result<DFlashPreparedGreedy, DFlashRunError> {
        if prompt.is_empty() {
            return Err(DFlashRunError::EmptyPrompt);
        }
        self.prepare_greedy_batch(target, session, PrefillBatch::tokens(prompt.to_vec()))
    }

    /// Prepare DFlash from ordered token and projected-image positions.
    pub fn prepare_greedy_batch(
        &mut self,
        target: &Model,
        session: &mut Session,
        prompt: PrefillBatch,
    ) -> Result<DFlashPreparedGreedy, DFlashRunError> {
        let prefill_started = Instant::now();
        self.reset();
        session.reset();
        let layers = self.forward.config().dflash_config.target_layer_ids.clone();
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            let token_prompt = prompt.segments.iter().try_fold(
                Vec::new(),
                |mut accumulated, segment| match segment {
                    crate::PrefillSegment::Tokens(tokens) => {
                        accumulated.extend_from_slice(tokens);
                        Some(accumulated)
                    }
                    crate::PrefillSegment::Embeddings(_) => None,
                },
            );
            if let Some(tokens) = token_prompt.filter(|tokens| tokens.len() > 1) {
                if let Some((prefill_logits, newest_hidden, pipeline)) = session
                    .prefill_dflash_prompt_pipelined(
                        &tokens,
                        &layers,
                        &mut self.forward,
                        &mut self.context_cache,
                    )?
                {
                    let positions = session.position();
                    let expected_prefix = positions - 1;
                    if positions != tokens.len()
                        || self.context_cache.ctx_offset != expected_prefix
                        || newest_hidden.len() != layers.len() * target.config().hidden_dim
                    {
                        return Err(DFlashRunError::Invariant(
                            "pipelined DFlash prompt frontiers or hidden geometry differ".into(),
                        ));
                    }
                    let mut hidden_cache =
                        DFlashHiddenCache::new(layers, target.config().hidden_dim);
                    hidden_cache.begin_capture(1);
                    hidden_cache.write_token_major(&newest_hidden, 1);
                    hidden_cache.commit(1);
                    let vocab = target.config().vocab_size;
                    let seed = argmax(&prefill_logits[prefill_logits.len() - vocab..]) as u32;
                    if std::env::var_os("MUSER_DFLASH_PREPARE_TRACE").is_some() {
                        eprintln!(
                            "dflash-prepare-trace target_capture_ns={}",
                            prefill_started.elapsed().as_nanos()
                        );
                        eprintln!(
                            "dflash-prepare-trace assistant_kv_ns={} prefix_tokens={} pipeline_chunks={} exposed_wait_ns={}",
                            pipeline.assistant_gpu_ns,
                            expected_prefix,
                            pipeline.chunks,
                            pipeline.exposed_wait_ns,
                        );
                    }
                    return Ok(DFlashPreparedGreedy {
                        seed,
                        hidden_cache,
                        prefill_ns: elapsed_ns(prefill_started),
                    });
                }
            }
        }
        let (prefill_logits, prompt_hidden) =
            session.prefill_batch_capturing_layers(prompt, &layers)?;
        if std::env::var_os("MUSER_DFLASH_PREPARE_TRACE").is_some() {
            eprintln!(
                "dflash-prepare-trace target_capture_ns={}",
                prefill_started.elapsed().as_nanos()
            );
        }
        let positions = session.position();
        let mut hidden_cache = DFlashHiddenCache::new(layers, target.config().hidden_dim);
        hidden_cache.begin_capture(positions);
        hidden_cache.write_token_major(&prompt_hidden, positions);
        hidden_cache.commit(positions);
        let vocab = target.config().vocab_size;
        let seed = argmax(&prefill_logits[prefill_logits.len() - vocab..]) as u32;
        // Assistant prompt K/V construction is prefill work, not decode work.
        // Consume every prompt row except the newest through the exact normal
        // forward path; noise outputs are discarded because only target K/V
        // enters the persistent assistant context. Keeping the newest row for
        // the first draft preserves the existing non-empty incremental ABI.
        if positions > 1 {
            let draft_started = Instant::now();
            let config = self.forward.config().clone();
            let mut ids = vec![config.dflash_config.mask_token_id; config.block_size];
            ids[0] = seed;
            let noise = target.embed_tokens(&ids)?;
            let prefix_tokens = positions - 1;
            let target_width = config.dflash_config.target_layer_ids.len() * config.hidden_size;
            self.forward.forward(
                &noise,
                &prompt_hidden[..prefix_tokens * target_width],
                prefix_tokens,
                &mut self.context_cache,
                &mut self.draft_cache,
            )?;
            hidden_cache.retain_current_suffix(1);
            if std::env::var_os("MUSER_DFLASH_PREPARE_TRACE").is_some() {
                eprintln!(
                    "dflash-prepare-trace assistant_kv_ns={} prefix_tokens={}",
                    draft_started.elapsed().as_nanos(),
                    prefix_tokens
                );
            }
        }
        Ok(DFlashPreparedGreedy {
            seed,
            hidden_cache,
            prefill_ns: elapsed_ns(prefill_started),
        })
    }

    /// Generate from a matching prepared target session.  Consuming the
    /// opaque value prevents accidental reuse against a different live cut.
    pub fn generate_prepared_greedy(
        &mut self,
        target: &Model,
        session: &mut Session,
        prepared: DFlashPreparedGreedy,
        max_new_tokens: usize,
        verify_length: usize,
        extra_stop_tokens: &[u32],
    ) -> Result<(Vec<u32>, DFlashSpecStats), DFlashRunError> {
        self.generate_prepared_greedy_streaming(
            target,
            session,
            prepared,
            max_new_tokens,
            verify_length,
            extra_stop_tokens,
            &mut |_| Ok(()),
        )
    }

    /// Generate exact verified prefixes and publish each prefix immediately
    /// after the target and assistant frontiers have committed it.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_prepared_greedy_streaming(
        &mut self,
        target: &Model,
        session: &mut Session,
        prepared: DFlashPreparedGreedy,
        max_new_tokens: usize,
        verify_length: usize,
        extra_stop_tokens: &[u32],
        on_commit: &mut dyn FnMut(&[u32]) -> Result<(), DFlashRunError>,
    ) -> Result<(Vec<u32>, DFlashSpecStats), DFlashRunError> {
        let prefill_ns = prepared.prefill_ns;
        let first_round_started = Instant::now();
        let mut first_round_prepare_ns = None;
        let result = {
            let mut timed_on_commit = |tokens: &[u32]| {
                first_round_prepare_ns.get_or_insert_with(|| elapsed_ns(first_round_started));
                on_commit(tokens)
            };
            self.generate_from_seed(
                target,
                session,
                prepared.seed,
                prepared.hidden_cache,
                max_new_tokens,
                verify_length,
                extra_stop_tokens,
                &mut timed_on_commit,
            )
        };
        let (tokens, mut stats) = result?;
        stats.prefill_ns = reported_prefill_ns(prefill_ns, first_round_prepare_ns);
        Ok((tokens, stats))
    }

    /// Continue exact speculative generation after an authenticated remote
    /// prefix installed both the target KV and DFlash context. The producer
    /// deliberately holds the final prompt token; evaluating that boundary
    /// token locally supplies the exact first-token logits and the one fresh
    /// target-hidden row needed to advance the imported DFlash context.
    #[doc(hidden)]
    pub fn generate_greedy_from_installed(
        &mut self,
        target: &Model,
        session: &mut Session,
        boundary_token: u32,
        max_new_tokens: usize,
        verify_length: usize,
        extra_stop_tokens: &[u32],
    ) -> Result<(Vec<u32>, DFlashSpecStats), DFlashRunError> {
        self.generate_greedy_from_installed_streaming(
            target,
            session,
            boundary_token,
            max_new_tokens,
            verify_length,
            extra_stop_tokens,
            &mut |_| Ok(()),
        )
    }

    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn generate_greedy_from_installed_streaming(
        &mut self,
        target: &Model,
        session: &mut Session,
        boundary_token: u32,
        max_new_tokens: usize,
        verify_length: usize,
        extra_stop_tokens: &[u32],
        on_commit: &mut dyn FnMut(&[u32]) -> Result<(), DFlashRunError>,
    ) -> Result<(Vec<u32>, DFlashSpecStats), DFlashRunError> {
        let prepared = self.prepare_greedy_from_installed(target, session, boundary_token)?;
        self.generate_prepared_greedy_streaming(
            target,
            session,
            prepared,
            max_new_tokens,
            verify_length,
            extra_stop_tokens,
            on_commit,
        )
    }

    /// Consume the held remote boundary token without committing an output
    /// token. The resulting exact target prompt cut may be published to
    /// kvpack before speculative decode extends it.
    pub fn prepare_greedy_from_installed(
        &mut self,
        target: &Model,
        session: &mut Session,
        boundary_token: u32,
    ) -> Result<DFlashPreparedGreedy, DFlashRunError> {
        let prefill_started = Instant::now();
        if self.context_cache.ctx_offset != session.position() {
            return Err(DFlashRunError::Invariant(
                "target and DFlash imported prefix positions differ".into(),
            ));
        }
        let layers = self.forward.config().dflash_config.target_layer_ids.clone();
        let (logits, hidden) = session.prefill_capturing_layers(&[boundary_token], &layers)?;
        let mut hidden_cache = DFlashHiddenCache::new(layers, target.config().hidden_dim);
        hidden_cache.begin_capture(1);
        hidden_cache.write_token_major(&hidden, 1);
        hidden_cache.commit(1);
        let vocab = target.config().vocab_size;
        let seed = argmax(&logits[logits.len() - vocab..]) as u32;
        Ok(DFlashPreparedGreedy {
            seed,
            hidden_cache,
            prefill_ns: elapsed_ns(prefill_started),
        })
    }

    /// Exact sampled speculative generation using the same scalar sampler for
    /// target and assistant.  `random_seed` makes the OpenAI route and the
    /// standalone scalar oracle reproducible.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_sampled(
        &mut self,
        target: &Model,
        session: &mut Session,
        prompt: &[u32],
        max_new_tokens: usize,
        verify_length: usize,
        params: SamplingParams,
        random_seed: u64,
    ) -> Result<(Vec<u32>, DFlashSpecStats), DFlashRunError> {
        if prompt.is_empty() {
            return Err(DFlashRunError::EmptyPrompt);
        }
        self.generate_sampled_batch(
            target,
            session,
            PrefillBatch::tokens(prompt.to_vec()),
            max_new_tokens,
            verify_length,
            params,
            random_seed,
            &[],
        )
    }

    /// Exact sampled speculative generation after an ordered multimodal
    /// prefill. Projected rows are target-hidden context only; every emitted
    /// token is still accepted by the full target distribution.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_sampled_batch(
        &mut self,
        target: &Model,
        session: &mut Session,
        prompt: PrefillBatch,
        max_new_tokens: usize,
        verify_length: usize,
        params: SamplingParams,
        random_seed: u64,
        extra_stop_tokens: &[u32],
    ) -> Result<(Vec<u32>, DFlashSpecStats), DFlashRunError> {
        self.generate_sampled_batch_streaming(
            target,
            session,
            prompt,
            max_new_tokens,
            verify_length,
            params,
            random_seed,
            extra_stop_tokens,
            &mut |_| Ok(()),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_sampled_batch_streaming(
        &mut self,
        target: &Model,
        session: &mut Session,
        prompt: PrefillBatch,
        max_new_tokens: usize,
        verify_length: usize,
        params: SamplingParams,
        random_seed: u64,
        extra_stop_tokens: &[u32],
        on_commit: &mut dyn FnMut(&[u32]) -> Result<(), DFlashRunError>,
    ) -> Result<(Vec<u32>, DFlashSpecStats), DFlashRunError> {
        let seed = u32::try_from(random_seed)
            .map_err(|_| DFlashRunError::InvalidRandomSeed(random_seed))?;
        let mut rng = Mt19937::new(seed);
        self.generate_sampled_batch_streaming_with_rng(
            target,
            session,
            prompt,
            max_new_tokens,
            verify_length,
            params,
            &mut rng,
            extra_stop_tokens,
            on_commit,
        )
    }

    /// Continue sampled DFlash generation on the caller-owned llama-compatible
    /// RNG stream. This is the stateful serving entry point: the stream is
    /// snapshotted beside target/DFlash KV and therefore resumes at the exact
    /// next draw after save, restore, or migration.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_sampled_batch_streaming_with_rng(
        &mut self,
        target: &Model,
        session: &mut Session,
        prompt: PrefillBatch,
        max_new_tokens: usize,
        verify_length: usize,
        params: SamplingParams,
        rng: &mut Mt19937,
        extra_stop_tokens: &[u32],
        on_commit: &mut dyn FnMut(&[u32]) -> Result<(), DFlashRunError>,
    ) -> Result<(Vec<u32>, DFlashSpecStats), DFlashRunError> {
        let prepared = self.prepare_sampled_batch_with_rng(target, session, prompt, params, rng)?;
        self.generate_prepared_sampled_streaming_with_rng(
            target,
            session,
            prepared,
            max_new_tokens,
            verify_length,
            params,
            rng,
            extra_stop_tokens,
            on_commit,
        )
    }

    /// Prepare a sampled prompt without committing an output token. This is
    /// the sampled counterpart of [`Self::prepare_greedy_batch`] and exists so
    /// the server can publish a hidden target+DFlash context atomically.
    pub fn prepare_sampled_batch_with_rng(
        &mut self,
        target: &Model,
        session: &mut Session,
        prompt: PrefillBatch,
        params: SamplingParams,
        rng: &mut Mt19937,
    ) -> Result<DFlashPreparedSampled, DFlashRunError> {
        let prefill_started = Instant::now();
        params.validate()?;
        self.reset();
        session.reset();
        let layers = self.forward.config().dflash_config.target_layer_ids.clone();
        let (prefill_logits, prompt_hidden) =
            session.prefill_batch_capturing_layers(prompt, &layers)?;
        let positions = session.position();
        let mut hidden_cache = DFlashHiddenCache::new(layers, target.config().hidden_dim);
        hidden_cache.begin_capture(positions);
        hidden_cache.write_token_major(&prompt_hidden, positions);
        hidden_cache.commit(positions);
        let vocab = target.config().vocab_size;
        let first = distribution_ordered(&prefill_logits[prefill_logits.len() - vocab..], params)?;
        let seed = sample_distribution_mt_ordered(&first.weights, &first.order, rng)?;
        Ok(DFlashPreparedSampled {
            seed,
            hidden_cache,
            prefill_ns: elapsed_ns(prefill_started),
        })
    }

    /// Continue from a sampled prompt prepared in a hidden generation.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_prepared_sampled_streaming_with_rng(
        &mut self,
        target: &Model,
        session: &mut Session,
        prepared: DFlashPreparedSampled,
        max_new_tokens: usize,
        verify_length: usize,
        params: SamplingParams,
        rng: &mut Mt19937,
        extra_stop_tokens: &[u32],
        on_commit: &mut dyn FnMut(&[u32]) -> Result<(), DFlashRunError>,
    ) -> Result<(Vec<u32>, DFlashSpecStats), DFlashRunError> {
        let first_round_started = Instant::now();
        let mut first_round_prepare_ns = None;
        let result = {
            let mut timed_on_commit = |tokens: &[u32]| {
                first_round_prepare_ns.get_or_insert_with(|| elapsed_ns(first_round_started));
                on_commit(tokens)
            };
            self.generate_sampled_from_seed(
                target,
                session,
                prepared.seed,
                prepared.hidden_cache,
                max_new_tokens,
                verify_length,
                params,
                rng,
                extra_stop_tokens,
                SampledRoute::from_env(),
                &mut timed_on_commit,
            )
        };
        let (tokens, mut stats) = result?;
        stats.prefill_ns = reported_prefill_ns(prepared.prefill_ns, first_round_prepare_ns);
        Ok((tokens, stats))
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_sampled_batch_with_route(
        &mut self,
        target: &Model,
        session: &mut Session,
        prompt: PrefillBatch,
        max_new_tokens: usize,
        verify_length: usize,
        params: SamplingParams,
        random_seed: u64,
        extra_stop_tokens: &[u32],
        route: SampledRoute,
        on_commit: &mut dyn FnMut(&[u32]) -> Result<(), DFlashRunError>,
    ) -> Result<(Vec<u32>, DFlashSpecStats), DFlashRunError> {
        let seed = u32::try_from(random_seed)
            .map_err(|_| DFlashRunError::InvalidRandomSeed(random_seed))?;
        let mut rng = Mt19937::new(seed);
        self.generate_sampled_batch_with_rng_and_route(
            target,
            session,
            prompt,
            max_new_tokens,
            verify_length,
            params,
            &mut rng,
            extra_stop_tokens,
            route,
            on_commit,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_sampled_batch_with_rng_and_route(
        &mut self,
        target: &Model,
        session: &mut Session,
        prompt: PrefillBatch,
        max_new_tokens: usize,
        verify_length: usize,
        params: SamplingParams,
        rng: &mut Mt19937,
        extra_stop_tokens: &[u32],
        route: SampledRoute,
        on_commit: &mut dyn FnMut(&[u32]) -> Result<(), DFlashRunError>,
    ) -> Result<(Vec<u32>, DFlashSpecStats), DFlashRunError> {
        let prefill_started = Instant::now();
        params.validate()?;
        self.reset();
        session.reset();
        let layers = self.forward.config().dflash_config.target_layer_ids.clone();
        let (prefill_logits, prompt_hidden) =
            session.prefill_batch_capturing_layers(prompt, &layers)?;
        let positions = session.position();
        let mut hidden_cache = DFlashHiddenCache::new(layers, target.config().hidden_dim);
        hidden_cache.begin_capture(positions);
        hidden_cache.write_token_major(&prompt_hidden, positions);
        hidden_cache.commit(positions);
        let vocab = target.config().vocab_size;
        let first = distribution_ordered(&prefill_logits[prefill_logits.len() - vocab..], params)?;
        let seed = sample_distribution_mt_ordered(&first.weights, &first.order, rng)?;
        let prefill_ns = elapsed_ns(prefill_started);
        let (tokens, mut stats) = self.generate_sampled_from_seed(
            target,
            session,
            seed,
            hidden_cache,
            max_new_tokens,
            verify_length,
            params,
            rng,
            extra_stop_tokens,
            route,
            on_commit,
        )?;
        stats.prefill_ns = prefill_ns;
        Ok((tokens, stats))
    }

    /// Sampled continuation after an authenticated, atomic target+DFlash
    /// import.  As in the greedy route, the producer holds one boundary token
    /// so the target locally produces the first exact probability row.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn generate_sampled_from_installed(
        &mut self,
        target: &Model,
        session: &mut Session,
        boundary_token: u32,
        max_new_tokens: usize,
        verify_length: usize,
        params: SamplingParams,
        random_seed: u64,
        extra_stop_tokens: &[u32],
    ) -> Result<(Vec<u32>, DFlashSpecStats), DFlashRunError> {
        self.generate_sampled_from_installed_streaming(
            target,
            session,
            boundary_token,
            max_new_tokens,
            verify_length,
            params,
            random_seed,
            extra_stop_tokens,
            &mut |_| Ok(()),
        )
    }

    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn generate_sampled_from_installed_streaming(
        &mut self,
        target: &Model,
        session: &mut Session,
        boundary_token: u32,
        max_new_tokens: usize,
        verify_length: usize,
        params: SamplingParams,
        random_seed: u64,
        extra_stop_tokens: &[u32],
        on_commit: &mut dyn FnMut(&[u32]) -> Result<(), DFlashRunError>,
    ) -> Result<(Vec<u32>, DFlashSpecStats), DFlashRunError> {
        let seed = u32::try_from(random_seed)
            .map_err(|_| DFlashRunError::InvalidRandomSeed(random_seed))?;
        let mut rng = Mt19937::new(seed);
        self.generate_sampled_from_installed_streaming_with_rng(
            target,
            session,
            boundary_token,
            max_new_tokens,
            verify_length,
            params,
            &mut rng,
            extra_stop_tokens,
            on_commit,
        )
    }

    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn generate_sampled_from_installed_streaming_with_rng(
        &mut self,
        target: &Model,
        session: &mut Session,
        boundary_token: u32,
        max_new_tokens: usize,
        verify_length: usize,
        params: SamplingParams,
        rng: &mut Mt19937,
        extra_stop_tokens: &[u32],
        on_commit: &mut dyn FnMut(&[u32]) -> Result<(), DFlashRunError>,
    ) -> Result<(Vec<u32>, DFlashSpecStats), DFlashRunError> {
        let prepared = self.prepare_sampled_from_installed_with_rng(
            target,
            session,
            boundary_token,
            params,
            rng,
        )?;
        self.generate_prepared_sampled_streaming_with_rng(
            target,
            session,
            prepared,
            max_new_tokens,
            verify_length,
            params,
            rng,
            extra_stop_tokens,
            on_commit,
        )
    }

    /// Sampled counterpart of [`Self::prepare_greedy_from_installed`]. The
    /// first distribution draw is part of the prepared state, while no output
    /// token or visible stream frontier is committed.
    pub fn prepare_sampled_from_installed_with_rng(
        &mut self,
        target: &Model,
        session: &mut Session,
        boundary_token: u32,
        params: SamplingParams,
        rng: &mut Mt19937,
    ) -> Result<DFlashPreparedSampled, DFlashRunError> {
        let prefill_started = Instant::now();
        params.validate()?;
        if self.context_cache.ctx_offset != session.position() {
            return Err(DFlashRunError::Invariant(
                "target and DFlash imported prefix positions differ".into(),
            ));
        }
        let layers = self.forward.config().dflash_config.target_layer_ids.clone();
        let (logits, hidden) = session.prefill_capturing_layers(&[boundary_token], &layers)?;
        let mut hidden_cache = DFlashHiddenCache::new(layers, target.config().hidden_dim);
        hidden_cache.begin_capture(1);
        hidden_cache.write_token_major(&hidden, 1);
        hidden_cache.commit(1);
        let first = distribution_ordered(&logits, params)?;
        let seed = sample_distribution_mt_ordered(&first.weights, &first.order, rng)?;
        Ok(DFlashPreparedSampled {
            seed,
            hidden_cache,
            prefill_ns: elapsed_ns(prefill_started),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_from_seed(
        &mut self,
        target: &Model,
        session: &mut Session,
        mut seed: u32,
        mut hidden_cache: DFlashHiddenCache,
        max_new_tokens: usize,
        verify_length: usize,
        extra_stop_tokens: &[u32],
        on_commit: &mut dyn FnMut(&[u32]) -> Result<(), DFlashRunError>,
    ) -> Result<(Vec<u32>, DFlashSpecStats), DFlashRunError> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if verify_length == 15 && self.exact_mirror_overlap_enabled() {
            return self.generate_from_seed_mirror(
                target,
                session,
                seed,
                hidden_cache,
                max_new_tokens,
                extra_stop_tokens,
                on_commit,
            );
        }
        let stops = StopSet {
            eos: self.forward.config().eos_token_id,
            extra: extra_stop_tokens,
        };
        let trace_cycles = cycle_trace_enabled();
        let diagnostic_pre_draft_idle = match std::env::var("MUSER_DFLASH_PRE_DRAFT_IDLE_MS") {
            Ok(value) if !trace_cycles => {
                return Err(DFlashRunError::Invariant(
                    "MUSER_DFLASH_PRE_DRAFT_IDLE_MS requires MUSER_DFLASH_CYCLE_TRACE=1".into(),
                ));
            }
            Ok(value) => {
                let milliseconds = value.parse::<u64>().map_err(|_| {
                    DFlashRunError::Invariant(
                        "MUSER_DFLASH_PRE_DRAFT_IDLE_MS is not an integer".into(),
                    )
                })?;
                if milliseconds > 1_000 {
                    return Err(DFlashRunError::Invariant(
                        "MUSER_DFLASH_PRE_DRAFT_IDLE_MS exceeds 1000".into(),
                    ));
                }
                Some(std::time::Duration::from_millis(milliseconds))
            }
            Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(_)) => {
                return Err(DFlashRunError::Invariant(
                    "MUSER_DFLASH_PRE_DRAFT_IDLE_MS is not UTF-8".into(),
                ));
            }
        };
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut stats = DFlashSpecStats {
            draft_sink_size: self.context_cache.sink_size,
            draft_sliding_window: self.context_cache.window_size,
            ..Default::default()
        };
        while generated.len() < max_new_tokens {
            let cycle_started = Instant::now();
            if stops.contains(seed) {
                session.decode(crate::DecodeInput { token_id: seed })?;
                generated.push(seed);
                stats.committed_tokens += 1;
                on_commit(std::slice::from_ref(&seed))?;
                break;
            }
            if stats.fallback_round() {
                let started = Instant::now();
                let decoded = session.decode(crate::DecodeInput { token_id: seed })?;
                stats.fallback_target_ns =
                    stats.fallback_target_ns.saturating_add(elapsed_ns(started));
                generated.push(seed);
                stats.committed_tokens += 1;
                stats.target_only_fallback_tokens += 1;
                on_commit(std::slice::from_ref(&seed))?;
                if generated.len() == max_new_tokens || stops.contains(seed) {
                    break;
                }
                seed = decoded.next_token;
                continue;
            }
            if let Some(idle) = diagnostic_pre_draft_idle {
                std::thread::sleep(idle);
            }
            let (context, n_ctx) = hidden_cache.read_current_batch();
            let room = max_new_tokens - generated.len();
            let draft_budget = verify_length.min(room.saturating_sub(1));
            let draft_started = Instant::now();
            let mut drafts = if draft_budget == 0 {
                Vec::new()
            } else {
                self.draft_greedy_for_session(
                    target,
                    session,
                    seed,
                    &context,
                    n_ctx,
                    verify_length,
                )?
            };
            stats.draft_ns = stats.draft_ns.saturating_add(elapsed_ns(draft_started));
            let cycle_draft_ns = elapsed_ns(draft_started);
            drafts.truncate(draft_budget);
            let mut candidates = Vec::with_capacity(1 + drafts.len());
            candidates.push(seed);
            candidates.extend_from_slice(&drafts);
            stops.truncate_candidates(&mut candidates);
            hidden_cache.begin_capture(candidates.len());
            let verify_started = Instant::now();
            let (verification, hidden) = session.verify_batch_capturing_layers(
                &candidates,
                hidden_cache.target_layer_ids.as_slice(),
            )?;
            hidden_cache.write_token_major(&hidden, verification.accepted);
            hidden_cache.commit(verification.accepted);
            stats.target_verify_ns = stats
                .target_verify_ns
                .saturating_add(elapsed_ns(verify_started));
            let cycle_verify_ns = elapsed_ns(verify_started);
            let committed = &candidates[..verification.accepted];
            generated.extend_from_slice(committed);
            stats.rounds += 1;
            stats.target_batches += 1;
            stats.committed_tokens += verification.accepted;
            stats.drafted_tokens += drafts.len();
            let accepted_drafts = verification.accepted.saturating_sub(1);
            stats.accepted_draft_tokens += accepted_drafts;
            stats.last_accepted_run = accepted_drafts;
            stats.draft_token_trace.extend_from_slice(&candidates[1..]);
            stats
                .accepted_prefix_trace
                .extend_from_slice(&candidates[1..1 + accepted_drafts]);
            stats.accepted_prefix_counts.push(accepted_drafts);
            stats.drafted_counts.push(drafts.len());
            on_commit(committed)?;
            if trace_cycles {
                eprintln!(
                    "dflash-cycle-accept round={} drafted={} accepted={} committed={}",
                    stats.rounds,
                    drafts.len(),
                    accepted_drafts,
                    verification.accepted,
                );
                stats.cycle_trace.push(DFlashCycleTrace {
                    draft_ns: cycle_draft_ns,
                    verify_ns: cycle_verify_ns,
                    cycle_ns: elapsed_ns(cycle_started),
                    drafted: drafts.len(),
                    accepted: accepted_drafts,
                });
            }
            if stops.truncate(&mut generated, &mut stats) {
                break;
            }
            seed = verification
                .replacement
                .unwrap_or(session.greedy_next_token()?);
            if verification.accepted == 0 {
                return Err(DFlashRunError::Invariant(
                    "target rejected its own greedy seed".into(),
                ));
            }
            stats.update_speculation_gate();
        }
        generated.truncate(max_new_tokens);
        Ok((generated, stats))
    }

    /// Exact conditional Mirror-SD. The fifteenth assistant proposal predicts
    /// the target bonus after a 14-draft verification round. Once target layer
    /// 49 has produced all five exact capture rows, ANE provisionally drafts
    /// the following round while Metal executes layers 50-51 and the LM head.
    /// The provisional assistant state is retained only when every current
    /// candidate and that bonus match; all other outcomes restore the bounded
    /// CPU-shadow checkpoint and force public Core ML to replay it.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[allow(clippy::too_many_arguments)]
    fn generate_from_seed_mirror(
        &mut self,
        target: &Model,
        session: &mut Session,
        mut seed: u32,
        mut hidden_cache: DFlashHiddenCache,
        max_new_tokens: usize,
        extra_stop_tokens: &[u32],
        on_commit: &mut dyn FnMut(&[u32]) -> Result<(), DFlashRunError>,
    ) -> Result<(Vec<u32>, DFlashSpecStats), DFlashRunError> {
        const VERIFIED_DRAFTS: usize = 14;
        const FULL_DRAFTS: usize = VERIFIED_DRAFTS + 1;
        let stops = StopSet {
            eos: self.forward.config().eos_token_id,
            extra: extra_stop_tokens,
        };
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut stats = DFlashSpecStats {
            draft_sink_size: self.context_cache.sink_size,
            draft_sliding_window: self.context_cache.window_size,
            ..Default::default()
        };
        let mut prefetched_drafts: Option<Vec<u32>> = None;
        let mut mirror_overlap_enabled = true;
        let trace_cycles = cycle_trace_enabled();
        while generated.len() < max_new_tokens {
            if stops.contains(seed) {
                session.decode(crate::DecodeInput { token_id: seed })?;
                generated.push(seed);
                stats.committed_tokens += 1;
                on_commit(std::slice::from_ref(&seed))?;
                break;
            }
            if stats.fallback_round() {
                let started = Instant::now();
                let decoded = session.decode(crate::DecodeInput { token_id: seed })?;
                stats.fallback_target_ns =
                    stats.fallback_target_ns.saturating_add(elapsed_ns(started));
                generated.push(seed);
                stats.committed_tokens += 1;
                stats.target_only_fallback_tokens += 1;
                on_commit(std::slice::from_ref(&seed))?;
                if generated.len() == max_new_tokens || stops.contains(seed) {
                    break;
                }
                seed = decoded.next_token;
                continue;
            }

            let room = max_new_tokens - generated.len();
            let (context, n_ctx) = hidden_cache.read_current_batch();
            let draft_started = Instant::now();
            let drafts = match prefetched_drafts.take() {
                Some(drafts) => drafts,
                None => self.draft_greedy_for_session(
                    target,
                    session,
                    seed,
                    &context,
                    n_ctx,
                    FULL_DRAFTS,
                )?,
            };
            stats.draft_ns = stats.draft_ns.saturating_add(elapsed_ns(draft_started));

            // The final short round cannot create useful lookahead. Preserve
            // the ordinary exact verifier rather than advancing provisional
            // state which the request will never consume.
            let candidate_overlap = mirror_overlap_candidate(
                mirror_overlap_enabled,
                drafts.len(),
                room,
                FULL_DRAFTS,
                VERIFIED_DRAFTS,
            );
            let used_drafts = if candidate_overlap {
                VERIFIED_DRAFTS
            } else {
                drafts.len().min(room.saturating_sub(1))
            };
            let mut candidates = Vec::with_capacity(1 + used_drafts);
            candidates.push(seed);
            candidates.extend_from_slice(&drafts[..used_drafts]);
            let contains_stop = stops.truncate_candidates(&mut candidates);
            let can_overlap = candidate_overlap && !contains_stop;
            hidden_cache.begin_capture(candidates.len());
            let verify_started = Instant::now();

            if !can_overlap {
                let (verification, hidden) = session.verify_batch_capturing_layers(
                    &candidates,
                    hidden_cache.target_layer_ids.as_slice(),
                )?;
                hidden_cache.write_token_major(&hidden, verification.accepted);
                hidden_cache.commit(verification.accepted);
                stats.target_verify_ns = stats
                    .target_verify_ns
                    .saturating_add(elapsed_ns(verify_started));
                let committed = &candidates[..verification.accepted];
                generated.extend_from_slice(committed);
                stats.rounds += 1;
                stats.target_batches += 1;
                stats.committed_tokens += verification.accepted;
                stats.drafted_tokens += used_drafts;
                let accepted_drafts = verification.accepted.saturating_sub(1);
                stats.accepted_draft_tokens += accepted_drafts;
                stats.last_accepted_run = accepted_drafts;
                stats.draft_token_trace.extend_from_slice(&candidates[1..]);
                stats
                    .accepted_prefix_trace
                    .extend_from_slice(&candidates[1..1 + accepted_drafts]);
                stats.accepted_prefix_counts.push(accepted_drafts);
                stats.drafted_counts.push(drafts.len());
                if trace_cycles {
                    eprintln!(
                        "dflash-cycle-accept round={} drafted={} accepted={} committed={}",
                        stats.rounds, used_drafts, accepted_drafts, verification.accepted,
                    );
                }
                if verification.accepted == 0 {
                    return Err(DFlashRunError::Invariant(
                        "target rejected its own greedy seed".into(),
                    ));
                }
                on_commit(committed)?;
                if stops.truncate(&mut generated, &mut stats) {
                    break;
                }
                seed = verification
                    .replacement
                    .unwrap_or(session.greedy_next_token()?);
                continue;
            }

            let predicted_bonus = drafts[VERIFIED_DRAFTS];
            let checkpoint = self.context_cache.checkpoint_append(candidates.len());
            let capture_fc = self.capture_fc_pipeline_backend();
            let (pending, captured, target_projection) = session
                .begin_dflash_verification_overlap(
                    &candidates,
                    hidden_cache.target_layer_ids.as_slice(),
                    capture_fc.as_deref(),
                )?;
            stats.mirror_overlap_attempts += 1;
            let provisional_started = Instant::now();
            let provisional = match target_projection.as_deref() {
                Some(projected) => self.draft_greedy_from_capture_projection(
                    target,
                    session,
                    predicted_bonus,
                    &captured,
                    projected,
                    candidates.len(),
                    FULL_DRAFTS,
                ),
                None => self.draft_greedy_for_session(
                    target,
                    session,
                    predicted_bonus,
                    &captured,
                    candidates.len(),
                    FULL_DRAFTS,
                ),
            };
            let provisional_ns = elapsed_ns(provisional_started);
            stats.mirror_overlap_draft_ns =
                stats.mirror_overlap_draft_ns.saturating_add(provisional_ns);
            stats.draft_ns = stats.draft_ns.saturating_add(provisional_ns);
            let finished = session.finish_dflash_verification_overlap(pending, captured);
            let (verification, hidden, target_ns, capture_fc_ns) = match finished {
                Ok(value) => value,
                Err(error) => {
                    self.rollback_mirror_append(checkpoint)?;
                    return Err(error.into());
                }
            };
            let overlap_wall_ns = elapsed_ns(verify_started);
            stats.mirror_overlap_wall_ns =
                stats.mirror_overlap_wall_ns.saturating_add(overlap_wall_ns);
            stats.mirror_overlap_hidden_ns = stats.mirror_overlap_hidden_ns.saturating_add(
                target_ns
                    .saturating_add(provisional_ns)
                    .saturating_sub(overlap_wall_ns),
            );
            stats.target_verify_ns = stats.target_verify_ns.saturating_add(target_ns);
            stats.mirror_capture_fc_ns = stats.mirror_capture_fc_ns.saturating_add(capture_fc_ns);
            hidden_cache.write_token_major(&hidden, verification.accepted);
            hidden_cache.commit(verification.accepted);
            let committed = &candidates[..verification.accepted];
            generated.extend_from_slice(committed);
            stats.rounds += 1;
            stats.target_batches += 1;
            stats.committed_tokens += verification.accepted;
            stats.drafted_tokens += used_drafts;
            let accepted_drafts = verification.accepted.saturating_sub(1);
            stats.accepted_draft_tokens += accepted_drafts;
            stats.last_accepted_run = accepted_drafts;
            stats.draft_token_trace.extend_from_slice(&candidates[1..]);
            stats
                .accepted_prefix_trace
                .extend_from_slice(&candidates[1..1 + accepted_drafts]);
            stats.accepted_prefix_counts.push(accepted_drafts);
            stats.drafted_counts.push(drafts.len());
            if trace_cycles {
                eprintln!(
                    "dflash-cycle-accept round={} drafted={} accepted={} committed={}",
                    stats.rounds, used_drafts, accepted_drafts, verification.accepted,
                );
            }
            if verification.accepted == 0 {
                self.rollback_mirror_append(checkpoint)?;
                return Err(DFlashRunError::Invariant(
                    "target rejected its own greedy seed".into(),
                ));
            }
            if let Err(error) = on_commit(committed) {
                self.rollback_mirror_append(checkpoint)?;
                return Err(error);
            }
            if stops.truncate(&mut generated, &mut stats) {
                stats.mirror_overlap_rollbacks += 1;
                self.rollback_mirror_append(checkpoint)?;
                break;
            }
            let exact_bonus = match verification.replacement {
                Some(token) => token,
                None => match session.greedy_next_token() {
                    Ok(token) => token,
                    Err(error) => {
                        self.rollback_mirror_append(checkpoint)?;
                        return Err(error.into());
                    }
                },
            };
            let exact_commit = mirror_commit_is_exact(
                verification.accepted,
                candidates.len(),
                exact_bonus,
                predicted_bonus,
                provisional.is_ok(),
            );
            if exact_commit {
                stats.mirror_overlap_commits += 1;
                prefetched_drafts = Some(provisional.expect("checked successful provisional"));
            } else {
                stats.mirror_overlap_rollbacks += 1;
                mirror_overlap_enabled = false;
                stats.mirror_overlap_circuit_breaks += 1;
                self.rollback_mirror_append(checkpoint)?;
                // Propagate a real ANE failure only after both target suffix
                // completion and rollback have restored an exact live state.
                provisional?;
            }
            seed = exact_bonus;
            stats.update_speculation_gate();
        }
        generated.truncate(max_new_tokens);
        Ok((generated, stats))
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_sampled_from_seed(
        &mut self,
        target: &Model,
        session: &mut Session,
        mut seed: u32,
        mut hidden_cache: DFlashHiddenCache,
        max_new_tokens: usize,
        verify_length: usize,
        params: SamplingParams,
        rng: &mut Mt19937,
        extra_stop_tokens: &[u32],
        route: SampledRoute,
        on_commit: &mut dyn FnMut(&[u32]) -> Result<(), DFlashRunError>,
    ) -> Result<(Vec<u32>, DFlashSpecStats), DFlashRunError> {
        if !matches!(verify_length, 3 | 7 | 15) {
            return Err(DFlashRunError::InvalidVerifyLength(verify_length));
        }
        let stops = StopSet {
            eos: self.forward.config().eos_token_id,
            extra: extra_stop_tokens,
        };
        let vocab = target.config().vocab_size;
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut stats = DFlashSpecStats {
            draft_sink_size: self.context_cache.sink_size,
            draft_sliding_window: self.context_cache.window_size,
            ..Default::default()
        };
        let trace_cycles = cycle_trace_enabled();
        while generated.len() < max_new_tokens {
            if stops.contains(seed) {
                session.decode(crate::DecodeInput { token_id: seed })?;
                generated.push(seed);
                stats.committed_tokens += 1;
                on_commit(std::slice::from_ref(&seed))?;
                break;
            }
            if stats.fallback_round() {
                let started = Instant::now();
                let decoded = session.decode(crate::DecodeInput { token_id: seed })?;
                stats.fallback_target_ns =
                    stats.fallback_target_ns.saturating_add(elapsed_ns(started));
                generated.push(seed);
                stats.committed_tokens += 1;
                stats.target_only_fallback_tokens += 1;
                on_commit(std::slice::from_ref(&seed))?;
                if generated.len() == max_new_tokens || stops.contains(seed) {
                    break;
                }
                let next = distribution_ordered(&decoded.logits, params)?;
                seed = sample_distribution_mt_ordered(&next.weights, &next.order, rng)?;
                continue;
            }
            let (context, n_ctx) = hidden_cache.read_current_batch();
            let room = max_new_tokens - generated.len();
            let draft_budget = verify_length.min(room.saturating_sub(1));
            let draft_started = Instant::now();
            let (mut drafts, mut q_rows) = if draft_budget == 0 {
                (Vec::new(), Vec::new())
            } else {
                self.draft_sampled_for_session(
                    target,
                    session,
                    seed,
                    &context,
                    n_ctx,
                    verify_length,
                    params,
                    rng,
                )?
            };
            stats.draft_ns = stats.draft_ns.saturating_add(elapsed_ns(draft_started));
            drafts.truncate(draft_budget);
            q_rows.truncate(draft_budget);

            let mut candidates = Vec::with_capacity(1 + drafts.len());
            candidates.push(seed);
            candidates.extend_from_slice(&drafts);
            stops.truncate_candidates(&mut candidates);
            drafts.truncate(candidates.len().saturating_sub(1));
            q_rows.truncate(drafts.len());
            let verify_started = Instant::now();
            // Rejected rows are zeroed by the capture window and never
            // committed, so the next draft reads accepted rows only.
            hidden_cache.begin_capture(candidates.len());
            let (decision, hidden) = match route {
                SampledRoute::Replay => {
                    let target_logits = session.evaluate_tokens_transactional(&candidates)?;
                    let decision = sampled_round_decision(
                        &drafts,
                        &q_rows,
                        &target_logits,
                        params,
                        vocab,
                        rng,
                    )?;
                    let (_, hidden) = session.prefill_capturing_layers(
                        &candidates[..1 + decision.accepted],
                        hidden_cache.target_layer_ids.as_slice(),
                    )?;
                    stats.target_batches += 2;
                    (decision, hidden)
                }
                SampledRoute::SinglePass => {
                    // The accepted prefix is committed out of the verify batch
                    // that captured the DFlash layers; there is no replay.
                    let mut decided = None;
                    let (committed_count, hidden) = session
                        .verify_sampled_capturing_layers::<DFlashRunError, _>(
                            &candidates,
                            hidden_cache.target_layer_ids.as_slice(),
                            |rows| {
                                let decision = sampled_round_decision(
                                    &drafts, &q_rows, rows, params, vocab, rng,
                                )?;
                                let committed_count = 1 + decision.accepted;
                                decided = Some(decision);
                                Ok(committed_count)
                            },
                        )?;
                    stats.target_batches += 1;
                    let decision = decided.ok_or_else(|| {
                        DFlashRunError::Invariant(
                            "sampled verification returned without a decision".into(),
                        )
                    })?;
                    if committed_count != 1 + decision.accepted {
                        return Err(DFlashRunError::Invariant(format!(
                            "target committed {committed_count} rows, expected {}",
                            1 + decision.accepted
                        )));
                    }
                    (decision, hidden)
                }
            };
            let committed_count = 1 + decision.accepted;
            let committed = &candidates[..committed_count];
            // The capture must cover the committed prefix exactly: a short
            // capture would otherwise leave zeroed feature rows behind.
            let expected_hidden = committed_count * hidden_cache.token_width();
            if hidden.len() != expected_hidden {
                return Err(DFlashRunError::Invariant(format!(
                    "target captured {} floats for {committed_count} committed rows, expected {expected_hidden}",
                    hidden.len()
                )));
            }
            hidden_cache.write_token_major(&hidden, committed_count);
            hidden_cache.commit(committed_count);
            stats.target_verify_ns = stats
                .target_verify_ns
                .saturating_add(elapsed_ns(verify_started));
            generated.extend_from_slice(committed);
            stats.rounds += 1;
            stats.committed_tokens += committed_count;
            stats.drafted_tokens += drafts.len();
            stats.accepted_draft_tokens += decision.accepted;
            stats.last_accepted_run = decision.accepted;
            stats.draft_token_trace.extend_from_slice(&candidates[1..]);
            stats
                .accepted_prefix_trace
                .extend_from_slice(&candidates[1..1 + decision.accepted]);
            stats.accepted_prefix_counts.push(decision.accepted);
            stats.drafted_counts.push(drafts.len());
            if trace_cycles {
                eprintln!(
                    "dflash-cycle-accept round={} drafted={} accepted={} committed={}",
                    stats.rounds,
                    drafts.len(),
                    decision.accepted,
                    committed_count,
                );
            }
            on_commit(committed)?;
            if stops.truncate(&mut generated, &mut stats) {
                break;
            }
            seed = decision.next_token;
            stats.update_speculation_gate();
        }
        generated.truncate(max_new_tokens);
        stats.committed_tokens = generated.len();
        Ok((generated, stats))
    }
}

fn elapsed_ns(started: Instant) -> u64 {
    started.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

fn reported_prefill_ns(prompt_prepare_ns: u64, first_round_prepare_ns: Option<u64>) -> u64 {
    prompt_prepare_ns.saturating_add(first_round_prepare_ns.unwrap_or_default())
}

/// Sampled verification structure. `SinglePass` captures the DFlash feature
/// layers inside the transactional verify batch and commits the accepted
/// prefix out of it. `Replay` is the previous route — verify everything, roll
/// back, then re-run the committed prefix — kept for one release behind
/// `MUSER_DFLASH_SAMPLED_REPLAY=1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SampledRoute {
    SinglePass,
    Replay,
}

impl SampledRoute {
    fn from_env() -> Self {
        Self::from_flag(std::env::var_os("MUSER_DFLASH_SAMPLED_REPLAY").as_deref())
    }

    fn from_flag(value: Option<&std::ffi::OsStr>) -> Self {
        if value == Some(std::ffi::OsStr::new("1")) {
            Self::Replay
        } else {
            Self::SinglePass
        }
    }
}

/// Lossless sampled acceptance for one round. `target_rows` is the exact
/// verification batch: row zero is the distribution the seed was drawn from
/// and takes no part in acceptance; rows 1.. are the target distributions
/// after each candidate, which is the geometry `verify_full_speculative`
/// requires. Both sampled routes decide through this one function so the
/// accept/reject rule and RNG draw order cannot diverge between them.
fn sampled_round_decision(
    drafts: &[u32],
    q_rows: &[Vec<f32>],
    target_rows: &[Vec<f32>],
    params: SamplingParams,
    vocab: usize,
    rng: &mut Mt19937,
) -> Result<SpeculativeDecision, DFlashRunError> {
    if target_rows.len() != drafts.len() + 2 {
        return Err(DFlashRunError::Invariant(format!(
            "sampled round produced {} target rows, expected {}",
            target_rows.len(),
            drafts.len() + 2
        )));
    }
    let p_rows = target_rows[1..]
        .iter()
        .map(|logits| {
            if logits.len() != vocab {
                return Err(DFlashRunError::Invariant(format!(
                    "target probability row has {} logits, expected {vocab}",
                    logits.len()
                )));
            }
            Ok(distribution_ordered(logits, params)?)
        })
        .collect::<Result<Vec<_>, DFlashRunError>>()?;
    let probabilities = p_rows
        .iter()
        .map(|row| row.probabilities.clone())
        .collect::<Vec<_>>();
    let orders = p_rows.into_iter().map(|row| row.order).collect::<Vec<_>>();
    Ok(verify_full_speculative_mt_ordered(
        drafts,
        q_rows,
        &probabilities,
        &orders,
        rng,
    )?)
}

fn mirror_commit_is_exact(
    accepted: usize,
    candidates: usize,
    exact_bonus: u32,
    predicted_bonus: u32,
    provisional_succeeded: bool,
) -> bool {
    accepted == candidates && exact_bonus == predicted_bonus && provisional_succeeded
}

fn mirror_overlap_candidate(
    enabled: bool,
    draft_count: usize,
    room: usize,
    full_drafts: usize,
    verified_drafts: usize,
) -> bool {
    enabled && draft_count == full_drafts && room > verified_drafts + 1
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;

    use super::{
        mirror_commit_is_exact, mirror_overlap_candidate, reported_prefill_ns,
        sampled_round_decision, DFlashSpecStats, SampledRoute, StopSet, DISABLE_ACCEPTANCE_FLOOR,
        DISABLE_WARMUP_ROUNDS, REQUALIFY_BASE_TOKENS, REQUALIFY_MAX_TOKENS,
    };
    use crate::dflash::DFlashHiddenCache;
    use crate::sampling::{
        distribution, distribution_ordered, verify_full_speculative_mt_ordered, Mt19937,
        SamplingParams,
    };

    /// Model EOS 2, caller stop 5.
    const STOPS: StopSet<'static> = StopSet {
        eos: 2,
        extra: &[5],
    };

    #[test]
    fn reported_prefill_includes_the_first_publishable_round() {
        assert_eq!(reported_prefill_ns(10, Some(7)), 17);
        assert_eq!(reported_prefill_ns(10, None), 10);
        assert_eq!(reported_prefill_ns(u64::MAX - 1, Some(7)), u64::MAX);
    }

    #[test]
    fn a_committed_block_returns_nothing_past_the_first_stop_token() {
        // One verification round can commit several tokens at once; the ones
        // behind the stop are not part of the response.
        let mut generated = vec![7, 8, 2, 9, 11];
        let mut stats = DFlashSpecStats {
            committed_tokens: 5,
            ..Default::default()
        };
        assert!(STOPS.truncate(&mut generated, &mut stats));
        assert_eq!(generated, vec![7, 8, 2]);
        assert_eq!(stats.committed_tokens, 3);
    }

    #[test]
    fn a_caller_stop_token_ends_the_stream_like_eos() {
        let mut generated = vec![7, 5, 9];
        let mut stats = DFlashSpecStats::default();
        assert!(STOPS.truncate(&mut generated, &mut stats));
        assert_eq!(generated, vec![7, 5]);
    }

    #[test]
    fn verifier_candidates_never_include_a_suffix_behind_stop() {
        let mut candidates = vec![7, 8, 5, 9, 11];
        assert!(STOPS.truncate_candidates(&mut candidates));
        assert_eq!(candidates, vec![7, 8, 5]);
    }

    #[test]
    fn an_empty_stop_list_keeps_eos_only_behaviour() {
        let eos_only = StopSet { eos: 2, extra: &[] };
        let mut generated = vec![7, 5, 9];
        let mut stats = DFlashSpecStats::default();
        assert!(!eos_only.truncate(&mut generated, &mut stats));
        assert_eq!(generated, vec![7, 5, 9]);
    }

    #[test]
    fn a_stream_without_a_stop_token_is_untouched() {
        let mut generated = vec![7, 8, 9];
        let mut stats = DFlashSpecStats {
            committed_tokens: 3,
            ..Default::default()
        };
        assert!(!STOPS.truncate(&mut generated, &mut stats));
        assert_eq!(generated, vec![7, 8, 9]);
        assert_eq!(stats.committed_tokens, 3);
    }

    #[test]
    fn adaptive_fallback_waits_for_frozen_evidence_window() {
        // Sample-size floor and the 25% threshold, now scoped to the
        // recent window instead of the request's cumulative history.
        let mut stats = rounds_of(&[(15, 0), (15, 0)]); // 30 drafted, warmup only
        assert!(!stats.should_disable_speculation());
        stats = rounds_of(&[(15, 0), (15, 0), (2, 0)]); // 32 drafted, past warmup
        assert!(stats.should_disable_speculation());
        stats = rounds_of(&[(15, 4), (15, 4), (2, 0)]); // 8/32 = 25%, not below
        assert!(!stats.should_disable_speculation());
    }

    /// Builds stats from (drafted, accepted) per round.
    fn rounds_of(per_round: &[(usize, usize)]) -> DFlashSpecStats {
        let mut stats = DFlashSpecStats::default();
        for (drafted, accepted) in per_round {
            stats.rounds += 1;
            stats.drafted_tokens += drafted;
            stats.accepted_draft_tokens += accepted;
            stats.committed_tokens += accepted + 1;
            stats.drafted_counts.push(*drafted);
            stats.accepted_prefix_counts.push(*accepted);
        }
        stats
    }

    #[test]
    fn warmup_rounds_never_close_the_gate() {
        // The coldest rounds of a request sit right after prefill; a
        // total miss there must not sentence the whole response.
        let mut stats = rounds_of(&[(15, 0), (15, 0)]);
        stats.drafted_counts.push(15);
        stats.accepted_prefix_counts.push(0);
        stats.drafted_tokens += 15;
        stats.rounds = DISABLE_WARMUP_ROUNDS;
        assert!(!stats.should_disable_speculation());
    }

    #[test]
    fn recent_recovery_reopens_a_request_that_started_badly() {
        // Regression for the 2026-08-21 latch: a request whose opening
        // rounds drafted badly but whose recent rounds draft well must
        // keep speculating. The cumulative rate here is 24%.
        // 20 dead rounds then 8 strong ones: cumulative 96/420 = 22.9%
        // (below the floor), recent window 96/120 = 80%.
        let mut rounds = vec![(15, 0); 20];
        rounds.extend(std::iter::repeat_n((15, 12), 8));
        let stats = rounds_of(&rounds);
        assert!(stats.acceptance_rate() < DISABLE_ACCEPTANCE_FLOOR);
        assert!(!stats.should_disable_speculation());
    }

    #[test]
    fn closed_gate_requalifies_after_the_cooldown() {
        let mut stats = rounds_of(&[(15, 0), (15, 0), (15, 0), (15, 0)]);
        stats.update_speculation_gate();
        assert!(stats.speculation_disabled);
        assert_eq!(stats.disable_events, 1);
        let reopen_at = stats.requalify_at_tokens;
        assert_eq!(reopen_at, stats.committed_tokens + REQUALIFY_BASE_TOKENS);

        // Plain decode until the cooldown elapses, then the gate reopens
        // and judges fresh evidence only.
        assert!(stats.fallback_round());
        stats.committed_tokens = reopen_at;
        assert!(!stats.fallback_round());
        assert!(!stats.speculation_disabled);
        assert_eq!(stats.window_floor, stats.accepted_prefix_counts.len());
        assert!(!stats.should_disable_speculation());
    }

    #[test]
    fn repeated_closures_back_off() {
        let mut stats = rounds_of(&[(15, 0), (15, 0), (15, 0), (15, 0)]);
        stats.update_speculation_gate();
        let first = stats.requalify_at_tokens - stats.committed_tokens;
        stats.speculation_disabled = false;
        stats.window_floor = stats.accepted_prefix_counts.len();
        for (drafted, accepted) in [(15, 0); 4] {
            stats.rounds += 1;
            stats.drafted_counts.push(drafted);
            stats.accepted_prefix_counts.push(accepted);
        }
        stats.update_speculation_gate();
        let second = stats.requalify_at_tokens - stats.committed_tokens;
        assert_eq!(second, first * 2);
        assert!(second <= REQUALIFY_MAX_TOKENS);
    }

    #[test]
    fn mirror_commit_requires_full_exact_target_confirmation() {
        assert!(mirror_commit_is_exact(15, 15, 42, 42, true));
        assert!(!mirror_commit_is_exact(14, 15, 42, 42, true));
        assert!(!mirror_commit_is_exact(15, 15, 43, 42, true));
        assert!(!mirror_commit_is_exact(15, 15, 42, 42, false));
    }

    #[test]
    fn mirror_overlap_circuit_stays_open_after_a_rollback() {
        assert!(mirror_overlap_candidate(true, 15, 256, 15, 14));
        assert!(!mirror_overlap_candidate(false, 15, 256, 15, 14));
        assert!(!mirror_overlap_candidate(true, 14, 256, 15, 14));
        assert!(!mirror_overlap_candidate(true, 15, 15, 15, 14));
    }

    const PARAMS: SamplingParams = SamplingParams {
        temperature: 0.8,
        top_p: 0.95,
        top_k: 0,
        typical_p: 1.0,
        min_p: 0.0,
        top_n_sigma: 0.0,
        min_keep: 0,
    };
    const VOCAB: usize = 8;

    fn logit_row(favoured: usize) -> Vec<f32> {
        let mut row = vec![0.25f32; VOCAB];
        row[favoured] = 4.0;
        row
    }

    /// Row zero is the distribution the seed was drawn from; the rest are the
    /// per-candidate target rows, exactly as both routes hand them over.
    fn target_rows(favoured: &[usize]) -> Vec<Vec<f32>> {
        favoured.iter().copied().map(logit_row).collect()
    }

    fn draft_rows(drafts: &[u32]) -> Vec<Vec<f32>> {
        drafts
            .iter()
            .map(|&token| distribution(&logit_row(token as usize), PARAMS).expect("draft row"))
            .collect()
    }

    /// The single-pass route must draw exactly what the replay route drew:
    /// same seed, same acceptance decision, same bonus token. Both call this
    /// one function, so the check is that the extraction is faithful to the
    /// verifier call it replaced.
    #[test]
    fn sampled_acceptance_is_unchanged_by_the_single_pass_extraction() {
        let drafts = [3u32, 5, 1];
        let q_rows = draft_rows(&drafts);
        let rows = target_rows(&[7, 3, 5, 6, 2]);

        let mut extracted_rng = Mt19937::new(20_250_813);
        let extracted =
            sampled_round_decision(&drafts, &q_rows, &rows, PARAMS, VOCAB, &mut extracted_rng)
                .expect("decision");

        // The pre-change inline sequence, verbatim.
        let mut inline_rng = Mt19937::new(20_250_813);
        let p_rows = rows[1..]
            .iter()
            .map(|logits| distribution_ordered(logits, PARAMS).expect("target row"))
            .collect::<Vec<_>>();
        let probabilities = p_rows
            .iter()
            .map(|row| row.probabilities.clone())
            .collect::<Vec<_>>();
        let orders = p_rows.into_iter().map(|row| row.order).collect::<Vec<_>>();
        let inline = verify_full_speculative_mt_ordered(
            &drafts,
            &q_rows,
            &probabilities,
            &orders,
            &mut inline_rng,
        )
        .expect("decision");

        assert_eq!(extracted.accepted, inline.accepted);
        assert_eq!(extracted.next_token, inline.next_token);
        // Third draft mismatches the target, so the round cannot accept it.
        assert!(extracted.accepted < drafts.len());
    }

    #[test]
    fn a_short_or_malformed_verification_batch_is_rejected_before_sampling() {
        let drafts = [3u32, 5];
        let q_rows = draft_rows(&drafts);
        let mut rng = Mt19937::new(1);
        // One row short of the seed row plus one row per candidate.
        let short = target_rows(&[7, 3, 5]);
        assert!(sampled_round_decision(&drafts, &q_rows, &short, PARAMS, VOCAB, &mut rng).is_err());
        let wide = target_rows(&[7, 3, 5, 6]);
        assert!(
            sampled_round_decision(&drafts, &q_rows, &wide, PARAMS, VOCAB + 1, &mut rng).is_err()
        );
    }

    #[test]
    fn the_replay_route_is_selected_only_by_the_explicit_flag() {
        assert_eq!(SampledRoute::from_flag(None), SampledRoute::SinglePass);
        assert_eq!(
            SampledRoute::from_flag(Some(OsStr::new("1"))),
            SampledRoute::Replay
        );
        for other in ["", "0", "true", "yes"] {
            assert_eq!(
                SampledRoute::from_flag(Some(OsStr::new(other))),
                SampledRoute::SinglePass
            );
        }
    }

    /// Capturing during the verify batch opens a window over every candidate,
    /// while the replay opened one over the committed prefix only. Both must
    /// leave the same committed rows visible to the next draft, and neither
    /// may expose a rejected row.
    #[test]
    fn in_batch_capture_exposes_the_same_rows_the_replay_committed() {
        let layers = vec![1usize, 3];
        let hidden_dim = 2usize;
        let width = layers.len() * hidden_dim;
        let candidates = 4usize;
        let committed = 2usize;
        // Token-major capture for the whole batch; rejected rows are the tail.
        let batch = (0..candidates * width)
            .map(|value| value as f32)
            .collect::<Vec<_>>();

        let mut replayed = DFlashHiddenCache::new(layers.clone(), hidden_dim);
        replayed.begin_capture(committed);
        replayed.write_token_major(&batch[..committed * width], committed);
        replayed.commit(committed);

        let mut single_pass = DFlashHiddenCache::new(layers, hidden_dim);
        single_pass.begin_capture(candidates);
        let mut accepted = batch.clone();
        accepted.truncate(committed * width);
        single_pass.write_token_major(&accepted, committed);
        single_pass.commit(committed);

        assert_eq!(single_pass.n_committed, replayed.n_committed);
        assert_eq!(single_pass.read_all(), replayed.read_all());
        assert_eq!(
            single_pass.read_current_batch(),
            replayed.read_current_batch()
        );
        assert_eq!(single_pass.read_current_batch().1, committed);
    }

    /// Model-conditional: the two routes must produce one identical sampled
    /// stream from one seed, and the single-pass route must reach it with one
    /// target batch per round instead of two.
    #[cfg(feature = "release-real-model")]
    #[test]
    fn sampled_single_pass_matches_the_replay_route_with_half_the_target_batches() {
        let model_path = std::env::var("MUSER_MODEL")
            .expect("release-real-model requires MUSER_MODEL for DFlash route parity");
        let dflash_path = std::env::var("MUSER_DFLASH")
            .expect("release-real-model requires MUSER_DFLASH for DFlash route parity");
        let model = crate::Model::load(crate::ModelConfig::new(model_path)).expect("Muse GGUF");
        let prompt = model.encode("The capital of France is");
        let max_new_tokens = 24usize;
        let verify_length = 7usize;
        let params = SamplingParams {
            temperature: 0.8,
            top_p: 0.95,
            top_k: 40,
            typical_p: 1.0,
            min_p: 0.0,
            top_n_sigma: 0.0,
            min_keep: 0,
        };
        let run = |route| {
            let mut assistant =
                super::DFlashAssistant::load(std::path::Path::new(&dflash_path), &model)
                    .expect("DFlash assistant");
            let mut session = model
                .new_session(crate::SessionConfig {
                    max_context: prompt.len() + max_new_tokens + 32,
                })
                .expect("session");
            let mut published = Vec::new();
            let mut on_commit = |tokens: &[u32]| {
                assert!(!tokens.is_empty());
                published.extend_from_slice(tokens);
                Ok(())
            };
            let result = assistant
                .generate_sampled_batch_with_route(
                    &model,
                    &mut session,
                    crate::PrefillBatch::tokens(prompt.clone()),
                    max_new_tokens,
                    verify_length,
                    params,
                    20_250_813,
                    &[],
                    route,
                    &mut on_commit,
                )
                .expect("sampled generation");
            assert_eq!(
                published, result.0,
                "callbacks must publish each exact prefix"
            );
            assert_eq!(session.position(), prompt.len() + published.len());
            result
        };
        let (replay_tokens, replay_stats) = run(SampledRoute::Replay);
        let (single_tokens, single_stats) = run(SampledRoute::SinglePass);

        assert_eq!(
            replay_tokens, single_tokens,
            "one seed must produce one sampled stream on both routes"
        );
        assert_eq!(replay_stats.rounds, single_stats.rounds);
        assert_eq!(
            replay_stats.accepted_draft_tokens,
            single_stats.accepted_draft_tokens
        );
        assert!(replay_stats.rounds > 0);
        assert_eq!(replay_stats.target_batches, 2 * replay_stats.rounds);
        assert_eq!(single_stats.target_batches, single_stats.rounds);
    }
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..values.len() {
        if values[i] > values[best] {
            best = i
        }
    }
    best
}

#[derive(Debug, thiserror::Error)]
pub enum DFlashRunError {
    #[error(transparent)]
    Engine(#[from] EngineError),
    #[error(transparent)]
    DFlash(#[from] DFlashError),
    #[error(transparent)]
    Sampling(#[from] crate::sampling::SamplingError),
    #[error("verify length must be one of 3, 7, or 15, got {0}")]
    InvalidVerifyLength(usize),
    #[error("DFlash prompt is empty")]
    EmptyPrompt,
    #[error("DFlash random seed must fit the pinned uint32 range, got {0}")]
    InvalidRandomSeed(u64),
    #[error("DFlash invariant failed: {0}")]
    Invariant(String),
}
