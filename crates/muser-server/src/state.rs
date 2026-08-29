//! `ServerState` — the one process-lifetime struct every telemetry field is
//! derived from (`metrics::build_snapshot(&ServerState) -> MetricsSnapshot`).
//!
//! muser-original: no Ferrite source, this is the real-telemetry pass's own
//! plumbing. Deliberately small: it holds exactly the counters this pass
//! wires for real (economics, sessions, event log, request/connection
//! counts, optional real model file size) and nothing that would require
//! faking data to populate (no fake node/cluster telemetry lives here — see
//! `metrics.rs` for where that's assembled and honesty-tagged instead).

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use kvpack_handoff::MultimodalIdentityV2;
use muser_cluster::config::{Nvfp4ProducerMode, ReceiverConfigV2};
use muser_cluster::control::PrefillControlSegmentV2;
use muser_cluster::receiver::{RemoteReceiveReceipt, RemoteReceiver};
use muser_engine::dflash::DFlashAssistant;
use muser_engine::vision::VisionModel;
use muser_engine::{DecodeInput, DecodeResult, Model, ModelConfig, Session, SessionConfig};
use muser_kvpack::economics::EconomicsCounters;
use muser_kvpack::reuse::PrefixReuse;

use crate::session::{EventLog, SessionRegistry};
use crate::session_store::SessionStore;

/// Rolling window every rate in the snapshot is computed over. A lifetime
/// average hides both idle stretches and bursts; ten seconds tracks a live
/// dashboard while still spanning several SSE ticks.
pub(crate) const RATE_WINDOW: Duration = Duration::from_secs(10);

/// Bound on the samples any one rolling window retains.
const RATE_WINDOW_SAMPLES: usize = 4_096;

/// Bound on the retained inter-token gap samples backing `wire.itl_ms`.
const DECODE_GAP_SAMPLES: usize = 4_096;

/// Consecutive remote-prefill failures that open the breaker, and how long
/// the remote route is skipped afterwards. An unreachable producer must not
/// tax every later request with a connect-and-wait before falling back.
pub(crate) const REMOTE_BREAKER_FAILURES: u64 = 3;
pub(crate) const REMOTE_BREAKER_COOLDOWN: Duration = Duration::from_secs(30);
const REMOTE_PRODUCER_BUSY: &str = "remote producer is busy with another prefill";

/// Shared, thread-safe process state for the telemetry server. One instance
/// is created at startup and handed to every connection handler behind an
/// `Arc`.
pub struct ServerState {
    /// Process start instant — backs `uptime_s`/`engine_clock_s`, both real
    /// monotonic-clock readings.
    pub started: Instant,
    /// Real, live cache-economics counters (see
    /// `muser_kvpack::economics`) — all-zero until real kvpack
    /// restore/prefill traffic flows.
    pub economics: EconomicsCounters,
    /// Real, live session table — all-empty until real inference/session
    /// endpoints are wired (Phase 2/4).
    pub sessions: SessionRegistry,
    /// Durable logical sessions, revision CAS, and encrypted bundle storage.
    pub(crate) logical_sessions: SessionStore,
    /// Real, live lifecycle-event ring buffer feeding the dashboard's
    /// `_events` extension.
    pub events: EventLog,
    /// Count of product HTTP requests this process has handled, incremented
    /// once per served request in `httpd`. Telemetry polling is excluded
    /// (see `telemetry_requests`), so this and the windowed
    /// `wire.requests_per_s` describe API traffic rather than the dashboard
    /// watching itself — `docs/telemetry.md`.
    pub requests_total: AtomicU64,
    /// Count of telemetry polls (`/snapshot`, `/metrics`, `/telemetry`),
    /// counted apart from `requests_total` so `wire.requests_per_s`
    /// describes product traffic rather than the dashboard watching itself.
    pub telemetry_requests: AtomicU64,
    /// Request start instants inside [`RATE_WINDOW`], backing the windowed
    /// `wire.requests_per_s`.
    request_window: Mutex<VecDeque<(Instant, u64)>>,
    /// Count of currently-open `GET /telemetry` SSE connections (live
    /// dashboard viewers). Real; exposed as the `_telemetry_viewers`
    /// snapshot extension.
    pub telemetry_viewers: AtomicU64,
    /// Connections currently being served, and requests currently waiting
    /// for the accelerator lease. Both are gauges, not totals: the accept
    /// loop refuses past a connection ceiling and a queued request is shed
    /// with 429 once its bounded wait expires.
    pub active_connections: AtomicU64,
    pub queue_depth: AtomicU64,
    pub overload_rejections: AtomicU64,
    /// Poisoned-lease recoveries since boot. One engine panic costs one
    /// request, never the process, but the recovery latches here so
    /// `GET /healthz` can report the process as degraded.
    pub lock_recoveries: AtomicU64,
    /// Monotonic, request-complete DFlash counters. These are updated only
    /// after target-verified output has been produced, so telemetry never
    /// counts speculative work that escaped verification.
    pub dflash_rounds: AtomicU64,
    pub dflash_drafted: AtomicU64,
    pub dflash_accepted: AtomicU64,
    pub dflash_fallback_tokens: AtomicU64,
    pub dflash_last_accepted_run: AtomicU64,
    /// Requests where `should_disable_speculation`'s <25% acceptance guard
    /// fired (`muser_engine::dflash::spec`). Monotonic like the DFlash
    /// counters above: differencing an isolated single-request snapshot
    /// window recovers 0 or 1 for that request.
    pub dflash_disabled_requests: AtomicU64,
    /// Sum of `committed_tokens` at the instant `dflash_disabled_requests`
    /// fired, one term per firing request. Differenced the same way, this
    /// recovers that request's own committed-token count at the trip point,
    /// or 0 when speculation was never disabled.
    pub dflash_disabled_at_tokens: AtomicU64,
    /// Gate closures summed across requests. Unlike
    /// `dflash_disabled_requests`, this counts every firing, including
    /// those a request later re-qualified out of.
    pub dflash_disable_events: AtomicU64,
    /// Effective draft context geometry of the most recent DFlash request
    /// (last-value, not a sum).
    pub dflash_draft_sink_size: AtomicU64,
    pub dflash_draft_sliding_window: AtomicU64,
    /// Observable runtime-route failures. These counters make fallback
    /// explicit instead of silently relabeling target-only output as ANE or
    /// Metal DFlash output.
    pub dflash_ane_failures: AtomicU64,
    pub dflash_metal_failures: AtomicU64,
    pub durable_generation: AtomicU64,
    pub ttft_ns: Mutex<VecDeque<u64>>,
    /// Inter-token gaps observed between successive emitted decode steps,
    /// and the cumulative count of completion tokens they came from. Only
    /// the incremental decode loop records gaps: a batch-rendered
    /// speculative run emits already-generated tokens back to back, so its
    /// emit spacing is not decode cadence.
    decode_gap_ns: Mutex<VecDeque<u64>>,
    pub completion_tokens_total: AtomicU64,
    pub(crate) last_generation: Mutex<Option<std::time::SystemTime>>,
    pub(crate) active_request: Mutex<Option<(&'static str, std::time::Instant)>>,
    /// Request-phase timings are monotonic completed-sample counters. They
    /// never infer phase splits from traffic rates.
    pub(crate) phase_timings: PhaseTimings,
    pub(crate) last_request_decode_milli_tok_s: AtomicU64,
    decode_window: Mutex<VecDeque<(Instant, u64)>>,
    /// Bounded completed remote-prefill receipts, recorded only after the
    /// receiver has atomically committed the engine generation.
    pub remote_transfers: Mutex<VecDeque<RemoteTransferSample>>,
    pub remote_bytes_received: AtomicU64,
    pub remote_transfer_ns: AtomicU64,
    ingress_window: Mutex<VecDeque<(Instant, u64)>>,
    /// Remote-prefill failure visibility. A silently-degraded disaggregated
    /// route otherwise looks identical to a route nobody exercised: the
    /// transfer list stays short and no counter moves.
    pub remote_receive_failures: AtomicU64,
    pub remote_fallbacks: AtomicU64,
    pub last_remote_error: Mutex<Option<String>>,
    remote_consecutive_failures: AtomicU64,
    remote_cooldown_until: Mutex<Option<Instant>>,
    remote_probe_in_flight: AtomicBool,
    /// `--model` / resolved `--gguf` path, if one was given at startup.
    pub model_path: Option<PathBuf>,
    /// Real on-disk size of `model_path`, read once at startup via
    /// `fs::metadata` — `None` if no model was given or the file couldn't be
    /// stat'd. This is the one field in `cluster` that can honestly be tagged
    /// `measured`; see `metrics::build_snapshot`.
    pub model_bytes: Option<u64>,
    pub(crate) model_sha256: Option<String>,
    /// Where the model bytes came from this run — `"local"` (already on
    /// disk), `"downloaded"` (fetched by `muser up` this run), `"missing"`
    /// (a path was given but not found), or `"none"`. Real provenance, never
    /// a guessed/implied source; surfaced verbatim by `GET /health`.
    pub model_source_label: &'static str,
    /// The URL a `"downloaded"` model was fetched from, when known. `None`
    /// for local / missing / no model. `GET /health` reports it so the
    /// source is never hardcoded or implied.
    pub model_source_url: Option<String>,
    /// Model identity installed by the dashboard-to-inference transition.
    /// Static `serve --model` / `up --node` launches continue to use the
    /// immutable fields above. A setup-only `muser up` starts with no model,
    /// then publishes this complete metadata record immediately before the
    /// prepared inference runtime becomes visible.
    activated_model: OnceLock<ActivatedModel>,
    /// Serializes the one legal setup -> inference transition. `OnceLock`
    /// prevents replacement after publication; this mutex additionally
    /// makes the metadata + runtime publication order atomic to callers.
    runtime_install: Mutex<()>,
    runtime_lifecycle: Mutex<RuntimeLifecycle>,
    /// Node onboarding jobs — the dashboard's "Add node" button. At most one
    /// runs at a time process-wide: the jobs drive ssh and the remote docker
    /// daemon, which two concurrent runs would interleave on. See
    /// `nodes_api`; nothing here touches inference.
    pub(crate) node_jobs: crate::nodes_api::NodeJobs,
    /// The single immutable model plus its bounded pool of independent
    /// resident serving slots.
    pub(crate) inference: InferenceCell,
}

/// The inference runtime is immutable after it is made visible, but a setup
/// dashboard needs to install it once without replacing the HTTP listener.
/// Keeping the familiar `as_ref` / `as_mut` surface makes the static builders
/// and request handlers use the same runtime representation.
pub(crate) struct InferenceCell(OnceLock<InferenceRuntime>);

impl InferenceCell {
    fn new() -> Self {
        Self(OnceLock::new())
    }

    pub(crate) fn as_ref(&self) -> Option<&InferenceRuntime> {
        self.0.get()
    }

    fn as_mut(&mut self) -> Option<&mut InferenceRuntime> {
        self.0.get_mut()
    }

    pub(crate) fn is_some(&self) -> bool {
        self.0.get().is_some()
    }

    pub(crate) fn is_none(&self) -> bool {
        self.0.get().is_none()
    }

    fn set(&self, runtime: InferenceRuntime) -> Result<(), Box<InferenceRuntime>> {
        // Installation is once-only. Box the impossible duplicate value on
        // the error path so the Result itself does not carry an 8+ KiB enum.
        self.0.set(runtime).map_err(Box::new)
    }

    fn into_inner(self) -> Option<InferenceRuntime> {
        self.0.into_inner()
    }
}

struct ActivatedModel {
    path: PathBuf,
    bytes: Option<u64>,
    sha256: Option<String>,
    source_label: &'static str,
    source_url: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RuntimeLifecycleSnapshot {
    pub(crate) phase: &'static str,
    pub(crate) node: Option<String>,
    pub(crate) detail: String,
}

struct RuntimeLifecycle {
    phase: &'static str,
    node: Option<String>,
    detail: String,
}

#[derive(Default)]
pub(crate) struct PhaseCounter {
    pub(crate) total_ns: AtomicU64,
    pub(crate) samples: AtomicU64,
}

impl PhaseCounter {
    fn record(&self, elapsed_ns: u64) {
        self.total_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
        self.samples.fetch_add(1, Ordering::Relaxed);
    }
}

#[derive(Default)]
pub(crate) struct PhaseTimings {
    pub(crate) queue: PhaseCounter,
    pub(crate) prefill: PhaseCounter,
    pub(crate) sampling: PhaseCounter,
    pub(crate) grammar: PhaseCounter,
    pub(crate) detokenization: PhaseCounter,
    pub(crate) enqueue_write: PhaseCounter,
    pub(crate) dflash_draft: PhaseCounter,
    pub(crate) dflash_target_verify: PhaseCounter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendMode {
    Auto,
    Cpu,
    Metal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextPolicy {
    Shift,
    Error,
}

pub struct InferenceRuntime {
    pub(crate) model: Model,
    pub(crate) vision: Option<VisionModel>,
    pub(crate) vision_identity: Option<VisionIdentity>,
    /// Exact assistant artifact identity bound into durable session bundles.
    /// A target-compatible but different draft cannot inherit its cache.
    pub(crate) dflash_identity_sha256: Option<String>,
    /// Independent serving slots. The pool owns admission and makes it
    /// impossible for more than `--parallel` generations to become resident.
    pub(crate) slots: SlotPool,
    /// Decode-step rendezvous. Request threads retain their independent slot
    /// ownership while one elected runner packs up to four ready Metal rows.
    pub(crate) decode_batcher: DecodeBatcher,
    /// DFlash state is sequence-local for exactly the same reason as target
    /// KV/RNG state. Indexes correspond one-for-one with `slots`.
    pub(crate) dflash: Option<Vec<Mutex<DFlashRuntime>>>,
    /// One assistant context paired with `staging`; it is never indexed by a
    /// serving slot and cannot participate in decode before an atomic swap.
    pub(crate) dflash_staging: Option<Mutex<DFlashRuntime>>,
    /// The one full-capacity generation reserved for atomic context rebuilds.
    /// It is deliberately outside `slots`, so it can never admit or decode a
    /// fifth serving request.
    pub(crate) staging: Mutex<Session>,
    pub(crate) backend: &'static str,
    pub(crate) max_context: usize,
    pub(crate) context_policy: ContextPolicy,
    pub(crate) raw_retain_prefix: usize,
    pub(crate) remote_prefill: Option<RemotePrefillRuntime>,
    pub(crate) prefix_reuse: Mutex<PrefixReuse>,
    pub(crate) prefix_cache_enabled: bool,
}

const MAX_QUEUED_REQUESTS: usize = 64;
const DECODE_COALESCE: Duration = Duration::from_micros(250);

struct DecodeBatchState {
    running: bool,
    last_slot: Option<usize>,
    queue: VecDeque<Arc<DecodeJob>>,
}

struct DecodeJob {
    slot: usize,
    session: usize,
    input: DecodeInput,
    result: Mutex<Option<Result<DecodeResult, String>>>,
}

pub(crate) struct DecodeBatcher {
    enabled: bool,
    state: Mutex<DecodeBatchState>,
    ready: Condvar,
    packed_batches: AtomicU64,
    packed_rows: AtomicU64,
    last_width: AtomicU64,
}

impl DecodeBatcher {
    fn new(metal: bool, resident_slots: usize) -> Self {
        Self {
            // A single resident slot can never form a multi-row batch. The
            // 250 us coalescing window only delays every token in the
            // release-relevant parallel-1 latency cell.
            enabled: metal && resident_slots > 1,
            state: Mutex::new(DecodeBatchState {
                running: false,
                last_slot: None,
                queue: VecDeque::new(),
            }),
            ready: Condvar::new(),
            packed_batches: AtomicU64::new(0),
            packed_rows: AtomicU64::new(0),
            last_width: AtomicU64::new(0),
        }
    }

    pub(crate) fn stats(&self) -> (u64, u64, u64) {
        (
            self.packed_batches.load(Ordering::Relaxed),
            self.packed_rows.load(Ordering::Relaxed),
            self.last_width.load(Ordering::Relaxed),
        )
    }

    pub(crate) fn decode(
        &self,
        slot: usize,
        session: &mut Session,
        input: DecodeInput,
    ) -> Result<DecodeResult, String> {
        if !self.enabled {
            return session.decode(input).map_err(|error| error.to_string());
        }
        let job = Arc::new(DecodeJob {
            slot,
            session: std::ptr::from_mut(session) as usize,
            input,
            result: Mutex::new(None),
        });
        {
            let mut state = self
                .state
                .lock()
                .map_err(|_| "decode batch coordinator is poisoned".to_string())?;
            if state.queue.iter().any(|queued| queued.slot == slot) {
                return Err(format!("slot {slot} already has a queued decode row"));
            }
            state.queue.push_back(Arc::clone(&job));
            self.ready.notify_all();
        }

        loop {
            if let Some(result) = job
                .result
                .lock()
                .map_err(|_| "decode batch result is poisoned".to_string())?
                .take()
            {
                return result;
            }
            let elected = {
                let mut state = self
                    .state
                    .lock()
                    .map_err(|_| "decode batch coordinator is poisoned".to_string())?;
                if !state.running {
                    state.running = true;
                    true
                } else {
                    let (next, _) = self
                        .ready
                        .wait_timeout(state, Duration::from_millis(1))
                        .map_err(|_| "decode batch coordinator is poisoned".to_string())?;
                    drop(next);
                    false
                }
            };
            if elected {
                self.run_one_batch();
            }
        }
    }

    fn run_one_batch(&self) {
        let jobs = {
            let mut state = match self.state.lock() {
                Ok(state) => state,
                Err(_) => return,
            };
            if state.queue.len() < 4 {
                let (next, _) = match self.ready.wait_timeout(state, DECODE_COALESCE) {
                    Ok(next) => next,
                    Err(_) => return,
                };
                state = next;
            }
            let mut candidates = state.queue.drain(..).collect::<Vec<_>>();
            candidates.sort_by_key(|job| decode_rotation_key(state.last_slot, job.slot));
            let split = candidates.len().min(4);
            let remainder = candidates.split_off(split);
            state.queue.extend(remainder);
            if let Some(job) = candidates.last() {
                state.last_slot = Some(job.slot);
            }
            candidates
        };

        if jobs.len() == 1 {
            let job = &jobs[0];
            // SAFETY: a DecodeJob exists only while its caller is blocked
            // inside decode, and SlotPermit guarantees exclusive ownership
            // of that Session for the whole call.
            let session = unsafe { &mut *(job.session as *mut Session) };
            let result = session.decode(job.input).map_err(|error| error.to_string());
            if let Ok(mut slot) = job.result.lock() {
                *slot = Some(result);
            }
        } else if !jobs.is_empty() {
            let unique = jobs
                .iter()
                .map(|job| job.slot)
                .collect::<std::collections::BTreeSet<_>>();
            let result = if unique.len() != jobs.len() {
                Err("decode batch contains duplicate resident slots".to_string())
            } else {
                // SAFETY: each job names a distinct, exclusively leased
                // Session and every submitting caller remains blocked until
                // its result below is installed.
                let mut sessions = jobs
                    .iter()
                    .map(|job| unsafe { &mut *(job.session as *mut Session) })
                    .collect::<Vec<_>>();
                let inputs = jobs.iter().map(|job| job.input).collect::<Vec<_>>();
                Session::decode_group(&mut sessions, &inputs).map_err(|error| error.to_string())
            };
            match result {
                Ok(results) => {
                    self.packed_batches.fetch_add(1, Ordering::Relaxed);
                    self.packed_rows
                        .fetch_add(jobs.len() as u64, Ordering::Relaxed);
                    self.last_width.store(jobs.len() as u64, Ordering::Relaxed);
                    for (job, result) in jobs.iter().zip(results) {
                        if let Ok(mut slot) = job.result.lock() {
                            *slot = Some(Ok(result));
                        }
                    }
                }
                Err(error) => {
                    for job in &jobs {
                        if let Ok(mut slot) = job.result.lock() {
                            *slot = Some(Err(error.clone()));
                        }
                    }
                }
            }
        }
        if let Ok(mut state) = self.state.lock() {
            state.running = false;
            self.ready.notify_all();
        }
    }
}

fn decode_rotation_key(last_slot: Option<usize>, slot: usize) -> (usize, usize) {
    match last_slot {
        Some(last) if slot > last => (0, slot),
        Some(_) => (1, slot),
        None => (0, slot),
    }
}

fn split_staging_generation<T>(
    mut generations: Vec<T>,
    serving_count: usize,
) -> Result<(Vec<T>, T), String> {
    if generations.len() != serving_count.saturating_add(1) {
        return Err(format!(
            "resident generation count {} does not equal {serving_count} serving plus one hidden staging",
            generations.len()
        ));
    }
    let staging = generations
        .pop()
        .expect("the validated generation vector includes staging");
    Ok((generations, staging))
}

struct SlotPoolState {
    sessions: Vec<Option<Session>>,
    available: Vec<usize>,
    waiting: usize,
}

/// Bounded admission for the resident target sessions.
///
/// A poisoned accelerator/session lease is not recovered in place. The
/// process is latched unhealthy so an operator restart is required before
/// any further inference, which avoids serving from uncertain GPU state.
pub(crate) struct SlotPool {
    state: Mutex<SlotPoolState>,
    available: Condvar,
    unhealthy: AtomicBool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SlotAcquireError {
    Overloaded,
    Unhealthy,
}

pub(crate) struct SlotPermit<'a> {
    pool: &'a SlotPool,
    index: usize,
    session: Option<Session>,
}

#[derive(Debug, serde::Serialize)]
pub(crate) struct SlotStatus {
    pub id: usize,
    pub is_processing: bool,
    pub n_ctx: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SlotSnapshot {
    pub schema: String,
    pub target: muser_engine::cache::SessionCacheSnapshot,
    pub logits: Vec<f32>,
}

impl SlotPool {
    fn new(sessions: Vec<Session>) -> Self {
        let count = sessions.len();
        Self {
            state: Mutex::new(SlotPoolState {
                sessions: sessions.into_iter().map(Some).collect(),
                available: (0..count).rev().collect(),
                waiting: 0,
            }),
            available: Condvar::new(),
            unhealthy: AtomicBool::new(false),
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self.state.lock() {
            Ok(state) => state.sessions.len(),
            Err(_) => {
                self.latch_unhealthy();
                0
            }
        }
    }

    pub(crate) fn is_healthy(&self) -> bool {
        !self.unhealthy.load(Ordering::Acquire)
    }

    pub(crate) fn latch_unhealthy(&self) {
        self.unhealthy.store(true, Ordering::Release);
        self.available.notify_all();
    }

    pub(crate) fn acquire(&self, timeout: Duration) -> Result<SlotPermit<'_>, SlotAcquireError> {
        if !self.is_healthy() {
            return Err(SlotAcquireError::Unhealthy);
        }
        let deadline = Instant::now() + timeout;
        let mut state = self.state.lock().map_err(|_| {
            self.latch_unhealthy();
            SlotAcquireError::Unhealthy
        })?;
        if let Some(index) = state.available.pop() {
            let session = state.sessions[index].take().ok_or_else(|| {
                self.latch_unhealthy();
                SlotAcquireError::Unhealthy
            })?;
            return Ok(SlotPermit {
                pool: self,
                index,
                session: Some(session),
            });
        }
        if state.waiting >= MAX_QUEUED_REQUESTS {
            return Err(SlotAcquireError::Overloaded);
        }
        state.waiting += 1;
        loop {
            if !self.is_healthy() {
                state.waiting -= 1;
                return Err(SlotAcquireError::Unhealthy);
            }
            let now = Instant::now();
            if now >= deadline {
                state.waiting -= 1;
                return Err(SlotAcquireError::Overloaded);
            }
            let (next, timed) = self
                .available
                .wait_timeout(state, deadline.saturating_duration_since(now))
                .map_err(|_| {
                    self.latch_unhealthy();
                    SlotAcquireError::Unhealthy
                })?;
            state = next;
            if let Some(index) = state.available.pop() {
                state.waiting -= 1;
                let session = state.sessions[index].take().ok_or_else(|| {
                    self.latch_unhealthy();
                    SlotAcquireError::Unhealthy
                })?;
                return Ok(SlotPermit {
                    pool: self,
                    index,
                    session: Some(session),
                });
            }
            if timed.timed_out() {
                state.waiting -= 1;
                return Err(SlotAcquireError::Overloaded);
            }
        }
    }

    pub(crate) fn acquire_specific(
        &self,
        requested: usize,
        timeout: Duration,
    ) -> Result<SlotPermit<'_>, SlotAcquireError> {
        if !self.is_healthy() {
            return Err(SlotAcquireError::Unhealthy);
        }
        let deadline = Instant::now() + timeout;
        let mut state = self.state.lock().map_err(|_| {
            self.latch_unhealthy();
            SlotAcquireError::Unhealthy
        })?;
        if state.sessions.is_empty() {
            self.latch_unhealthy();
            return Err(SlotAcquireError::Unhealthy);
        }
        // The pinned server wraps an explicit slot ID by the resident count.
        let index = requested % state.sessions.len();
        if let Some(position) = state
            .available
            .iter()
            .position(|candidate| *candidate == index)
        {
            state.available.swap_remove(position);
            let session = state.sessions[index].take().ok_or_else(|| {
                self.latch_unhealthy();
                SlotAcquireError::Unhealthy
            })?;
            return Ok(SlotPermit {
                pool: self,
                index,
                session: Some(session),
            });
        }
        if state.waiting >= MAX_QUEUED_REQUESTS {
            return Err(SlotAcquireError::Overloaded);
        }
        state.waiting += 1;
        loop {
            if !self.is_healthy() {
                state.waiting -= 1;
                return Err(SlotAcquireError::Unhealthy);
            }
            let now = Instant::now();
            if now >= deadline {
                state.waiting -= 1;
                return Err(SlotAcquireError::Overloaded);
            }
            let (next, timed) = self
                .available
                .wait_timeout(state, deadline.saturating_duration_since(now))
                .map_err(|_| {
                    self.latch_unhealthy();
                    SlotAcquireError::Unhealthy
                })?;
            state = next;
            if let Some(position) = state
                .available
                .iter()
                .position(|candidate| *candidate == index)
            {
                state.waiting -= 1;
                state.available.swap_remove(position);
                let session = state.sessions[index].take().ok_or_else(|| {
                    self.latch_unhealthy();
                    SlotAcquireError::Unhealthy
                })?;
                return Ok(SlotPermit {
                    pool: self,
                    index,
                    session: Some(session),
                });
            }
            if timed.timed_out() {
                state.waiting -= 1;
                return Err(SlotAcquireError::Overloaded);
            }
        }
    }

    pub(crate) fn status(&self, n_ctx: usize) -> Result<Vec<SlotStatus>, SlotAcquireError> {
        let state = self.state.lock().map_err(|_| {
            self.latch_unhealthy();
            SlotAcquireError::Unhealthy
        })?;
        Ok((0..state.sessions.len())
            .map(|id| SlotStatus {
                id,
                is_processing: !state.available.contains(&id),
                n_ctx,
            })
            .collect())
    }

    pub(crate) fn erase_idle(&self, index: usize) -> Result<(), SlotAcquireError> {
        let mut state = self.state.lock().map_err(|_| {
            self.latch_unhealthy();
            SlotAcquireError::Unhealthy
        })?;
        if index >= state.sessions.len() {
            return Err(SlotAcquireError::Overloaded);
        }
        let position = state
            .available
            .iter()
            .position(|candidate| *candidate == index)
            .ok_or(SlotAcquireError::Overloaded)?;
        state.available.swap_remove(position);
        let mut session = state.sessions[index]
            .take()
            .ok_or(SlotAcquireError::Unhealthy)?;
        drop(state);
        session.reset();
        let mut state = self.state.lock().map_err(|_| {
            self.latch_unhealthy();
            SlotAcquireError::Unhealthy
        })?;
        state.sessions[index] = Some(session);
        state.available.push(index);
        self.available.notify_one();
        Ok(())
    }

    pub(crate) fn snapshot_idle(&self, index: usize) -> Result<SlotSnapshot, String> {
        if !self.is_healthy() {
            return Err("accelerator state is unhealthy".into());
        }
        let state = self.state.lock().map_err(|_| {
            self.latch_unhealthy();
            "accelerator slot registry is poisoned".to_string()
        })?;
        if index >= state.sessions.len() {
            return Err(format!("slot {index} does not exist"));
        }
        if !state.available.contains(&index) {
            return Err(format!("slot {index} is busy"));
        }
        let session = state.sessions[index]
            .as_ref()
            .ok_or_else(|| format!("slot {index} has no resident session"))?;
        let target = session
            .export_cache_snapshot()
            .map_err(|error| format!("slot {index} has no saveable cache: {error}"))?;
        let logits = session
            .cached_logits()
            .ok_or_else(|| format!("slot {index} has no final target distribution"))?
            .to_vec();
        Ok(SlotSnapshot {
            schema: "muser.slot-snapshot.v1".into(),
            target,
            logits,
        })
    }

    pub(crate) fn restore_idle(&self, index: usize, snapshot: &SlotSnapshot) -> Result<(), String> {
        if snapshot.schema != "muser.slot-snapshot.v1" {
            return Err("slot snapshot schema is not supported".into());
        }
        if !self.is_healthy() {
            return Err("accelerator state is unhealthy".into());
        }
        let mut state = self.state.lock().map_err(|_| {
            self.latch_unhealthy();
            "accelerator slot registry is poisoned".to_string()
        })?;
        if index >= state.sessions.len() {
            return Err(format!("slot {index} does not exist"));
        }
        let available = state
            .available
            .iter()
            .position(|candidate| *candidate == index)
            .ok_or_else(|| format!("slot {index} is busy"))?;
        state.available.swap_remove(available);
        let mut session = state.sessions[index]
            .take()
            .ok_or_else(|| format!("slot {index} has no resident session"))?;
        drop(state);

        let restored = session
            .install_cache_snapshot(&snapshot.target)
            .and_then(|()| session.install_restored_logits(&snapshot.logits));
        if restored.is_err() {
            session.reset();
        }
        let mut state = self.state.lock().map_err(|_| {
            self.latch_unhealthy();
            "accelerator slot registry is poisoned".to_string()
        })?;
        if state.sessions[index].replace(session).is_some() {
            self.latch_unhealthy();
            return Err("slot ownership changed during restore".into());
        }
        state.available.push(index);
        self.available.notify_one();
        restored.map_err(|error| format!("slot snapshot is incompatible: {error}"))
    }
}

impl SlotPermit<'_> {
    pub(crate) fn index(&self) -> usize {
        self.index
    }

    pub(crate) fn session_mut(&mut self) -> &mut Session {
        self.session
            .as_mut()
            .expect("resident session remains owned by its permit")
    }
}

impl Drop for SlotPermit<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.pool.latch_unhealthy();
            return;
        }
        if let Ok(mut state) = self.pool.state.lock() {
            let Some(session) = self.session.take() else {
                self.pool.latch_unhealthy();
                return;
            };
            if state.sessions[self.index].replace(session).is_some() {
                self.pool.latch_unhealthy();
                return;
            }
            state.available.push(self.index);
            self.pool.available.notify_one();
        } else {
            self.pool.latch_unhealthy();
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct VisionIdentity {
    pub projector_sha256: String,
    pub preprocessing_sha256: String,
}

pub(crate) const VISION_PREPROCESSING_CONTRACT: &[u8] =
    b"muse-glimmer-vision-v1:lanczos3:max-image-tokens-4096:rgb-normalized:pixel-shuffle-2";

/// One transactional DFlash route plus its lazily loaded release fallback.
///
/// ANE startup does not duplicate the assistant weights merely to keep a
/// dormant Metal copy.  If public CoreML reports a normal inference error,
/// the same pinned artifact is loaded on the Metal route and the request is
/// restarted from an exact target snapshot.  Accelerator-process failures
/// are handled outside the process by the campaign's stop-the-lane policy.
pub(crate) struct DFlashRuntime {
    primary: DFlashAssistant,
    primary_route: &'static str,
    fallback: Option<DFlashAssistant>,
    fallback_path: Option<PathBuf>,
}

impl DFlashRuntime {
    fn single(primary: DFlashAssistant, route: &'static str) -> Self {
        Self {
            primary,
            primary_route: route,
            fallback: None,
            fallback_path: None,
        }
    }

    fn ane(primary: DFlashAssistant, fallback_path: &std::path::Path) -> Self {
        Self {
            primary,
            primary_route: "ane",
            fallback: None,
            fallback_path: Some(fallback_path.to_path_buf()),
        }
    }

    pub(crate) fn primary_mut(&mut self) -> &mut DFlashAssistant {
        &mut self.primary
    }

    pub(crate) fn primary_route(&self) -> &'static str {
        self.primary_route
    }

    /// Return the next DFlash route in the frozen fallback chain.  Loading is
    /// delayed until an ANE inference actually fails so normal ANE residency
    /// is not penalized by a second complete assistant allocation.
    pub(crate) fn fallback_mut(
        &mut self,
        model: &Model,
        target_backend: &'static str,
    ) -> Result<Option<&mut DFlashAssistant>, muser_engine::dflash::DFlashError> {
        let Some(path) = self.fallback_path.as_deref() else {
            return Ok(None);
        };
        if self.fallback.is_none() {
            self.fallback = Some(load_default_dflash(path, model, target_backend)?);
        }
        Ok(self.fallback.as_mut())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemotePrefillMode {
    Auto,
    Required,
}

pub struct RemotePrefillRuntime {
    receiver: RemoteReceiver,
    mode: RemotePrefillMode,
    /// The native producer is one unchunked vLLM job at a time. Admit at
    /// most one control request from this Mac instead of filling the remote
    /// TCP backlog with work that can only time out behind a deep prefill.
    producer_lease: Mutex<()>,
}

#[derive(Debug, Clone)]
pub(crate) struct RemoteTransferSample {
    pub transfer_id: String,
    pub installed_bytes: u64,
    pub transfer_ns: u64,
    pub active_drain_ns: u64,
    pub control_ns: u64,
    pub accept_ns: u64,
    pub hidden_fraction: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum InferenceLoadError {
    #[error(transparent)]
    Engine(#[from] muser_engine::EngineError),
    #[error("the Metal backend is unavailable in this build or on this platform")]
    MetalUnavailable,
    #[error(transparent)]
    Vision(#[from] muser_engine::vision::VisionError),
    #[error("vision identity: {0}")]
    VisionIdentity(String),
    #[error("Metal vision bridge: {0}")]
    VisionBridge(String),
    #[error(transparent)]
    DFlash(#[from] muser_engine::dflash::DFlashError),
    #[error("DFlash identity: {0}")]
    DFlashIdentity(String),
    #[error("CoreML DFlash error: {0}")]
    CoreMl(String),
    #[error("remote prefill configuration: {0}")]
    Remote(String),
    #[error("resident prefix cache: {0}")]
    Resident(String),
    #[error("durable prefix cache: {0}")]
    Durable(String),
    #[error("runtime activation: {0}")]
    Activation(String),
}

impl ServerState {
    pub fn new(model_path: Option<&str>) -> Self {
        Self::new_with_verified_sha256(model_path, None)
    }

    pub fn new_with_verified_sha256(
        model_path: Option<&str>,
        verified_sha256: Option<String>,
    ) -> Self {
        let model_path = model_path.map(PathBuf::from);
        let model_bytes = model_path
            .as_ref()
            .and_then(|p| std::fs::metadata(p).ok())
            .map(|m| m.len());
        let model_sha256 = verified_sha256.or_else(|| {
            model_path
                .as_deref()
                .filter(|_| model_bytes.is_some())
                .and_then(|path| sha256_file(path).ok())
        });
        // Default provenance from what we can see on disk: a stat'd file is
        // `local`, a given-but-absent path is `missing`, no path is `none`.
        // `muser up` overrides this with `.with_provenance("downloaded", url)`
        // when it fetched the bytes itself.
        let model_source_label = match (&model_path, model_bytes) {
            (Some(_), Some(_)) => "local",
            (Some(_), None) => "missing",
            (None, _) => "none",
        };
        ServerState {
            started: Instant::now(),
            economics: EconomicsCounters::new(),
            sessions: SessionRegistry::new(),
            logical_sessions: SessionStore::new(),
            events: EventLog::new(),
            requests_total: AtomicU64::new(0),
            telemetry_requests: AtomicU64::new(0),
            request_window: Mutex::new(VecDeque::new()),
            telemetry_viewers: AtomicU64::new(0),
            active_connections: AtomicU64::new(0),
            queue_depth: AtomicU64::new(0),
            overload_rejections: AtomicU64::new(0),
            lock_recoveries: AtomicU64::new(0),
            dflash_rounds: AtomicU64::new(0),
            dflash_drafted: AtomicU64::new(0),
            dflash_accepted: AtomicU64::new(0),
            dflash_fallback_tokens: AtomicU64::new(0),
            dflash_last_accepted_run: AtomicU64::new(0),
            dflash_disabled_requests: AtomicU64::new(0),
            dflash_disabled_at_tokens: AtomicU64::new(0),
            dflash_disable_events: AtomicU64::new(0),
            dflash_draft_sink_size: AtomicU64::new(0),
            dflash_draft_sliding_window: AtomicU64::new(0),
            dflash_ane_failures: AtomicU64::new(0),
            dflash_metal_failures: AtomicU64::new(0),
            durable_generation: AtomicU64::new(1),
            ttft_ns: Mutex::new(VecDeque::with_capacity(4_096)),
            decode_gap_ns: Mutex::new(VecDeque::with_capacity(DECODE_GAP_SAMPLES)),
            completion_tokens_total: AtomicU64::new(0),
            last_generation: Mutex::new(None),
            active_request: Mutex::new(None),
            phase_timings: PhaseTimings::default(),
            last_request_decode_milli_tok_s: AtomicU64::new(0),
            decode_window: Mutex::new(VecDeque::new()),
            remote_transfers: Mutex::new(VecDeque::with_capacity(64)),
            remote_bytes_received: AtomicU64::new(0),
            remote_transfer_ns: AtomicU64::new(0),
            ingress_window: Mutex::new(VecDeque::new()),
            remote_receive_failures: AtomicU64::new(0),
            remote_fallbacks: AtomicU64::new(0),
            last_remote_error: Mutex::new(None),
            remote_consecutive_failures: AtomicU64::new(0),
            remote_cooldown_until: Mutex::new(None),
            remote_probe_in_flight: AtomicBool::new(false),
            model_path,
            model_bytes,
            model_sha256,
            model_source_label,
            model_source_url: None,
            activated_model: OnceLock::new(),
            runtime_install: Mutex::new(()),
            runtime_lifecycle: Mutex::new(RuntimeLifecycle {
                phase: "setup",
                node: None,
                detail: "Add a prefill node to start the Mac decoder".into(),
            }),
            node_jobs: crate::nodes_api::NodeJobs::new(),
            inference: InferenceCell::new(),
        }
    }

    #[allow(clippy::too_many_arguments)] // Startup's frozen serving contract is intentionally explicit.
    pub fn with_inference(
        mut self,
        model_path: &std::path::Path,
        max_context: usize,
        parallel: usize,
        context_policy: ContextPolicy,
        raw_retain_prefix: usize,
        requested: BackendMode,
        resident_cache_bytes: u64,
        prefix_cache_enabled: bool,
    ) -> Result<Self, InferenceLoadError> {
        let model = Model::load(ModelConfig::new(model_path))?;
        if !(1..=4).contains(&parallel) {
            return Err(InferenceLoadError::Resident(
                "parallel resident slots must be in 1..=4".into(),
            ));
        }
        let config = SessionConfig { max_context };
        let session_generations = parallel + 1;
        let mut sessions = Vec::with_capacity(session_generations);
        let backend = match requested {
            BackendMode::Cpu => {
                for _ in 0..session_generations {
                    sessions.push(model.new_session(config)?);
                }
                "cpu"
            }
            BackendMode::Metal => {
                #[cfg(all(target_os = "macos", feature = "metal"))]
                {
                    sessions = model.new_metal_sessions(config, session_generations)?;
                    "metal"
                }
                #[cfg(not(all(target_os = "macos", feature = "metal")))]
                {
                    return Err(InferenceLoadError::MetalUnavailable);
                }
            }
            BackendMode::Auto => {
                #[cfg(all(target_os = "macos", feature = "metal"))]
                {
                    sessions = model.new_metal_sessions(config, session_generations)?;
                    "metal"
                }
                #[cfg(not(all(target_os = "macos", feature = "metal")))]
                {
                    for _ in 0..session_generations {
                        sessions.push(model.new_session(config)?);
                    }
                    "cpu"
                }
            }
        };
        let (sessions, staging) =
            split_staging_generation(sessions, parallel).map_err(InferenceLoadError::Resident)?;
        self.inference
            .set(InferenceRuntime {
                model,
                vision: None,
                vision_identity: None,
                dflash_identity_sha256: None,
                dflash: None,
                dflash_staging: None,
                staging: Mutex::new(staging),
                slots: SlotPool::new(sessions),
                decode_batcher: DecodeBatcher::new(backend == "metal", parallel),
                backend,
                max_context,
                context_policy,
                raw_retain_prefix,
                remote_prefill: None,
                prefix_reuse: Mutex::new(
                    PrefixReuse::new(resident_cache_bytes)
                        .map_err(|error| InferenceLoadError::Resident(error.to_string()))?,
                ),
                prefix_cache_enabled,
            })
            .map_err(|_| {
                InferenceLoadError::Resident("inference runtime was initialized twice".into())
            })?;
        *self
            .runtime_lifecycle
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = RuntimeLifecycle {
            phase: "ready",
            node: None,
            detail: "Mac decoder is ready".into(),
        };
        Ok(self)
    }

    pub fn with_vision(
        mut self,
        mmproj_path: &std::path::Path,
        mtmd_bridge_path: Option<&std::path::Path>,
    ) -> Result<Self, InferenceLoadError> {
        let runtime = self
            .inference
            .as_mut()
            .ok_or(InferenceLoadError::MetalUnavailable)?;
        #[cfg(all(target_os = "macos", feature = "metal"))]
        let vision = if runtime.backend == "metal" {
            let bridge = mtmd_bridge_path.ok_or_else(|| {
                InferenceLoadError::VisionBridge(
                    "pass --mtmd-bridge with the packaged libmuser_mtmd_bridge.dylib".into(),
                )
            })?;
            VisionModel::load_metal(mmproj_path, bridge)?
        } else {
            VisionModel::load(mmproj_path)?
        };
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        let vision = {
            let _ = mtmd_bridge_path;
            VisionModel::load(mmproj_path)?
        };
        runtime.vision = Some(vision);
        runtime.vision_identity = Some(VisionIdentity {
            projector_sha256: sha256_file(mmproj_path)
                .map_err(InferenceLoadError::VisionIdentity)?,
            preprocessing_sha256: {
                use sha2::{Digest, Sha256};
                format!("{:x}", Sha256::digest(VISION_PREPROCESSING_CONTRACT))
            },
        });
        Ok(self)
    }

    pub fn with_durable_cache(
        mut self,
        config_path: &std::path::Path,
    ) -> Result<Self, InferenceLoadError> {
        let config = muser_kvpack::config::DurableConfigV1::load(config_path)
            .map_err(|error| InferenceLoadError::Durable(error.to_string()))?;
        let model_path = self
            .model_path
            .as_deref()
            .ok_or_else(|| InferenceLoadError::Durable("model path is absent".into()))?;
        let actual = sha256_file(model_path).map_err(InferenceLoadError::Durable)?;
        let expected = config
            .model_sha256()
            .map_err(|error| InferenceLoadError::Durable(error.to_string()))?;
        if actual != hex_digest(&expected) {
            return Err(InferenceLoadError::Durable(
                "configured model SHA-256 differs from the loaded GGUF".into(),
            ));
        }
        let runtime = self
            .inference
            .as_mut()
            .ok_or(InferenceLoadError::MetalUnavailable)?;
        if config.identity.weight_precision != runtime.model.weight_precision() {
            return Err(InferenceLoadError::Durable(format!(
                "configured weight precision {} differs from loaded artifact {}",
                config.identity.weight_precision,
                runtime.model.weight_precision()
            )));
        }
        if !runtime.prefix_cache_enabled {
            return Err(InferenceLoadError::Durable(
                "--kvpack-config requires --prefix-cache on".into(),
            ));
        }
        let durable = config
            .open(runtime.model.config().clone())
            .map_err(|error| InferenceLoadError::Durable(error.to_string()))?;
        runtime
            .prefix_reuse
            .get_mut()
            .map_err(|_| InferenceLoadError::Durable("prefix cache lease was poisoned".into()))?
            .set_durable(durable);
        Ok(self)
    }

    pub fn with_dflash(
        mut self,
        dflash_path: &std::path::Path,
    ) -> Result<Self, InferenceLoadError> {
        let draft_identity =
            dflash_identity(dflash_path).map_err(InferenceLoadError::DFlashIdentity)?;
        let runtime = self
            .inference
            .as_mut()
            .ok_or(InferenceLoadError::MetalUnavailable)?;
        let route = if runtime.backend == "metal" {
            "metal"
        } else {
            "cpu"
        };
        let mut slots = Vec::with_capacity(runtime.slots.len());
        for _ in 0..runtime.slots.len() {
            let assistant = load_default_dflash(dflash_path, &runtime.model, runtime.backend)?;
            slots.push(Mutex::new(DFlashRuntime::single(assistant, route)));
        }
        let staging = load_default_dflash(dflash_path, &runtime.model, runtime.backend)?;
        runtime.dflash = Some(slots);
        runtime.dflash_staging = Some(Mutex::new(DFlashRuntime::single(staging, route)));
        runtime.dflash_identity_sha256 = Some(draft_identity);
        Ok(self)
    }

    #[cfg(all(target_os = "macos", feature = "ane-coreml"))]
    pub fn with_dflash_ane(
        mut self,
        dflash_path: &std::path::Path,
        manifest_path: Option<&std::path::Path>,
    ) -> Result<Self, InferenceLoadError> {
        use std::sync::Arc;

        use muser_engine::dflash::DFlashConfig;
        use muser_engine::dflash_ane::{dflash_artifact_identity, file_sha256, AneDFlashBackend};

        let runtime = self
            .inference
            .as_mut()
            .ok_or(InferenceLoadError::MetalUnavailable)?;
        if runtime.backend != "metal" {
            return Err(InferenceLoadError::CoreMl(
                "ANE DFlash requires the Metal target backend so its fallback route is Metal"
                    .into(),
            ));
        }
        let target_path = self
            .model_path
            .as_deref()
            .ok_or_else(|| InferenceLoadError::CoreMl("model path is absent".into()))?;
        let target_identity = file_sha256(target_path).map_err(InferenceLoadError::CoreMl)?;
        let draft_identity =
            dflash_artifact_identity(dflash_path).map_err(InferenceLoadError::CoreMl)?;
        let config = DFlashConfig::from_artifact(dflash_path)?;
        let default_manifest;
        let manifest_path = if let Some(path) = manifest_path {
            path
        } else {
            default_manifest = if dflash_path.is_file() {
                dflash_path.with_extension("ane").join("manifest.json")
            } else {
                dflash_path.join("ane/manifest.json")
            };
            &default_manifest
        };
        let backend = Arc::new(
            AneDFlashBackend::load(manifest_path, &target_identity, &draft_identity, &config)
                .map_err(InferenceLoadError::CoreMl)?,
        );
        let mut slots = Vec::with_capacity(runtime.slots.len());
        for _ in 0..runtime.slots.len() {
            let assistant = DFlashAssistant::load_with_projection_backend(
                dflash_path,
                &runtime.model,
                backend.clone(),
            )?;
            slots.push(Mutex::new(DFlashRuntime::ane(assistant, dflash_path)));
        }
        let staging =
            DFlashAssistant::load_with_projection_backend(dflash_path, &runtime.model, backend)?;
        runtime.dflash = Some(slots);
        runtime.dflash_staging = Some(Mutex::new(DFlashRuntime::ane(staging, dflash_path)));
        runtime.dflash_identity_sha256 = Some(draft_identity);
        Ok(self)
    }

    /// Record where the model bytes came from (used by `muser up` after it
    /// resolves or downloads the GGUF). Real provenance for `GET /health`;
    /// never a hardcoded/implied source.
    pub fn with_provenance(mut self, label: &'static str, url: Option<String>) -> Self {
        self.model_source_label = label;
        self.model_source_url = url;
        self
    }

    pub fn with_remote_prefill(
        mut self,
        config_path: &std::path::Path,
        mode: RemotePrefillMode,
        dflash_path: Option<&std::path::Path>,
    ) -> Result<Self, InferenceLoadError> {
        require_strict_cross_vendor_qk(std::env::var("MUSER_CROSS_VENDOR_QK").ok().as_deref())?;
        let config = ReceiverConfigV2::load(config_path).map_err(InferenceLoadError::Remote)?;
        let runtime = self
            .inference
            .as_mut()
            .ok_or_else(|| InferenceLoadError::Remote("inference runtime is absent".into()))?;
        validate_remote_dflash_policy(config.producer_mode, dflash_path.is_some())?;
        // Startup already admitted the model bytes before constructing this
        // state. Re-reading a 17-20 GB GGUF here made every remote launch hash
        // the same file twice without adding a new trust boundary.
        let actual_model = self.model_sha256.as_deref().ok_or_else(|| {
            InferenceLoadError::Remote("verified model identity is absent".into())
        })?;
        if actual_model != config.identity.model_sha256 {
            return Err(InferenceLoadError::Remote(
                "cluster model SHA-256 differs from the loaded GGUF".into(),
            ));
        }
        match (dflash_path, config.dflash_identity_sha256.as_deref()) {
            (None, None) => {}
            (Some(path), Some(expected)) => {
                let actual = dflash_identity(path).map_err(InferenceLoadError::Remote)?;
                if actual != expected {
                    return Err(InferenceLoadError::Remote(
                        "cluster DFlash identity differs from the loaded assistant".into(),
                    ));
                }
            }
            _ => {
                return Err(InferenceLoadError::Remote(
                    "remote target+DFlash transfer must be configured on both sides or neither"
                        .into(),
                ))
            }
        }
        let receiver = RemoteReceiver::bind(config).map_err(InferenceLoadError::Remote)?;
        runtime.remote_prefill = Some(RemotePrefillRuntime {
            receiver,
            mode,
            producer_lease: Mutex::new(()),
        });
        Ok(self)
    }

    /// Publish a fully prepared decoder + receiver into an already-running
    /// setup server. Preparation happens off to the side, so request handlers
    /// can observe only "not ready" or the complete runtime — never a model
    /// without its authenticated remote receiver.
    pub(crate) fn install_prepared_runtime(
        &self,
        prepared: ServerState,
        node: &str,
    ) -> Result<(), InferenceLoadError> {
        let _install = self
            .runtime_install
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if self.inference.is_some() || self.activated_model.get().is_some() {
            return Err(InferenceLoadError::Activation(
                "an inference runtime is already installed".into(),
            ));
        }

        let runtime = prepared.inference.into_inner().ok_or_else(|| {
            InferenceLoadError::Activation("prepared inference runtime is absent".into())
        })?;
        let path = prepared.model_path.ok_or_else(|| {
            InferenceLoadError::Activation("prepared model path is absent".into())
        })?;
        if prepared.model_sha256.as_deref().is_none_or(str::is_empty) {
            return Err(InferenceLoadError::Activation(
                "prepared model identity is absent".into(),
            ));
        }
        let metadata = ActivatedModel {
            path,
            bytes: prepared.model_bytes,
            sha256: prepared.model_sha256,
            source_label: prepared.model_source_label,
            source_url: prepared.model_source_url,
        };

        // Publication order is deliberate. A handler that acquires the
        // inference OnceLock is guaranteed to see the metadata set before it.
        self.activated_model.set(metadata).map_err(|_| {
            InferenceLoadError::Activation("model metadata was installed twice".into())
        })?;
        self.inference.set(runtime).map_err(|_| {
            InferenceLoadError::Activation("inference runtime was installed twice".into())
        })?;
        self.set_runtime_lifecycle(
            "ready",
            Some(node),
            "Mac decoder and remote prefill are ready",
        );
        Ok(())
    }

    pub(crate) fn model_path(&self) -> Option<&std::path::Path> {
        self.activated_model
            .get()
            .map(|model| model.path.as_path())
            .or(self.model_path.as_deref())
    }

    pub(crate) fn model_bytes(&self) -> Option<u64> {
        self.activated_model
            .get()
            .and_then(|model| model.bytes)
            .or(self.model_bytes)
    }

    pub(crate) fn model_sha256(&self) -> Option<&str> {
        self.activated_model
            .get()
            .and_then(|model| model.sha256.as_deref())
            .or(self.model_sha256.as_deref())
    }

    pub(crate) fn model_source_label(&self) -> &str {
        self.activated_model
            .get()
            .map(|model| model.source_label)
            .unwrap_or(self.model_source_label)
    }

    pub(crate) fn model_source_url(&self) -> Option<&str> {
        self.activated_model
            .get()
            .and_then(|model| model.source_url.as_deref())
            .or(self.model_source_url.as_deref())
    }

    pub(crate) fn mark_runtime_loading(&self, node: &str, detail: &str) {
        self.set_runtime_lifecycle("loading", Some(node), detail);
    }

    pub(crate) fn mark_runtime_failed(&self, node: &str, detail: &str) {
        self.set_runtime_lifecycle("failed", Some(node), detail);
    }

    pub(crate) fn runtime_lifecycle(&self) -> RuntimeLifecycleSnapshot {
        let lifecycle = self
            .runtime_lifecycle
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        RuntimeLifecycleSnapshot {
            phase: lifecycle.phase,
            node: lifecycle.node.clone(),
            detail: lifecycle.detail.clone(),
        }
    }

    fn set_runtime_lifecycle(&self, phase: &'static str, node: Option<&str>, detail: &str) {
        let mut lifecycle = self
            .runtime_lifecycle
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        lifecycle.phase = phase;
        lifecycle.node = node.map(str::to_string);
        lifecycle.detail = detail.to_string();
    }

    pub fn uptime_s(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }

    /// True after any auxiliary lock recovery or accelerator-slot poison.
    /// Accelerator poison is fail-closed and requires process restart.
    pub fn degraded(&self) -> bool {
        self.lock_recoveries.load(Ordering::Relaxed) > 0
            || self
                .inference
                .as_ref()
                .is_some_and(|runtime| !runtime.slots.is_healthy())
    }

    pub(crate) fn record_lock_recovery(&self) {
        self.lock_recoveries.fetch_add(1, Ordering::Relaxed);
    }

    /// Count one product (non-telemetry) request against the windowed rate.
    pub(crate) fn record_request(&self) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        push_window(&self.request_window, 1);
    }

    pub(crate) fn requests_per_s(&self) -> f64 {
        window_rate(&self.request_window)
    }

    /// One inter-token gap from the incremental decode loop.
    pub(crate) fn record_decode_gap(&self, gap_ns: u64) {
        let mut values = self
            .decode_gap_ns
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if values.len() == DECODE_GAP_SAMPLES {
            values.pop_front();
        }
        values.push_back(gap_ns);
    }

    /// Completion tokens actually produced, from every route including the
    /// batch-rendered speculative ones.
    pub(crate) fn record_decode_tokens(&self, tokens: u64) {
        if tokens == 0 {
            return;
        }
        self.completion_tokens_total
            .fetch_add(tokens, Ordering::Relaxed);
        push_window(&self.decode_window, tokens);
        // Idle dashboards distinguish "never ran" from "ran, now quiet" by
        // this stamp. Raw time only on the hot path; the snapshot reader
        // formats it, never the per-token writer.
        *self
            .last_generation
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Some(std::time::SystemTime::now());
    }

    pub(crate) fn set_active_phase(&self, phase: &'static str) {
        *self
            .active_request
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Some((phase, std::time::Instant::now()));
    }

    pub(crate) fn clear_active(&self) {
        *self
            .active_request
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = None;
    }

    /// (phase, ms in that phase) for the in-flight request, if any.
    pub(crate) fn active_phase(&self) -> Option<(&'static str, u64)> {
        self.active_request
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .map(|(phase, since)| (phase, since.elapsed().as_millis() as u64))
    }

    pub(crate) fn last_generation(&self) -> Option<String> {
        let stamp = *self
            .last_generation
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        stamp.map(|at| {
            let secs = at
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            crate::timefmt::unix_to_rfc3339(secs)
        })
    }

    pub(crate) fn decode_gaps_ns(&self) -> Vec<u64> {
        self.decode_gap_ns
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .iter()
            .copied()
            .collect()
    }

    pub(crate) fn decode_tokens_per_s(&self) -> f64 {
        window_rate(&self.decode_window)
    }

    /// Windowed inbound remote-prefill rate in gigabits per second. Bytes
    /// per window nanosecond is already Gb/s.
    pub(crate) fn ingress_gbps(&self) -> f64 {
        window_total(&self.ingress_window) as f64 * 8.0 / RATE_WINDOW.as_nanos() as f64
    }

    /// Record a failed remote receive and, after
    /// [`REMOTE_BREAKER_FAILURES`] consecutive ones, open the breaker.
    pub(crate) fn record_remote_failure(&self, error: &str) {
        // Local admission did not touch the network and says nothing about
        // producer health. Count the caller's fallback separately, but never
        // open the connectivity breaker because Mac slots contended for the
        // deliberately single-flight node.
        if remote_producer_is_busy(error) {
            return;
        }
        self.remote_receive_failures.fetch_add(1, Ordering::Relaxed);
        *self
            .last_remote_error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(error.to_string());
        let consecutive = self
            .remote_consecutive_failures
            .fetch_add(1, Ordering::Relaxed)
            + 1;
        if consecutive >= REMOTE_BREAKER_FAILURES {
            *self
                .remote_cooldown_until
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) =
                Some(Instant::now() + REMOTE_BREAKER_COOLDOWN);
        }
        // A failed half-open probe re-arms the cool-down at once instead of
        // spending two more doomed requests re-earning it.
        if self.remote_probe_in_flight.swap(false, Ordering::Relaxed) {
            *self
                .remote_cooldown_until
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) =
                Some(Instant::now() + REMOTE_BREAKER_COOLDOWN);
        }
    }

    /// Count one request served locally because the remote route was
    /// unavailable — including requests the breaker skipped outright.
    pub(crate) fn record_remote_fallback(&self) {
        self.remote_fallbacks.fetch_add(1, Ordering::Relaxed);
    }

    /// Whether the remote route may be attempted. Closed while the breaker's
    /// cool-down runs; on expiry exactly one half-open probe is admitted and
    /// concurrent callers keep falling back until the probe resolves.
    /// `Required` mode never consults it.
    pub(crate) fn remote_route_is_open(&self) -> bool {
        let mut until = self
            .remote_cooldown_until
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match *until {
            Some(deadline) if Instant::now() < deadline => false,
            Some(_) => {
                *until = None;
                !self.remote_probe_in_flight.swap(true, Ordering::Relaxed)
            }
            None => !self.remote_probe_in_flight.load(Ordering::Relaxed),
        }
    }

    pub(crate) fn last_remote_error(&self) -> Option<String> {
        self.last_remote_error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    pub(crate) fn record_dflash_stats(&self, stats: &muser_engine::dflash::DFlashSpecStats) {
        if stats.prefill_ns > 0 {
            self.phase_timings.prefill.record(stats.prefill_ns);
        }
        self.dflash_rounds
            .fetch_add(stats.rounds as u64, Ordering::Relaxed);
        self.dflash_drafted
            .fetch_add(stats.drafted_tokens as u64, Ordering::Relaxed);
        self.dflash_accepted
            .fetch_add(stats.accepted_draft_tokens as u64, Ordering::Relaxed);
        self.dflash_fallback_tokens
            .fetch_add(stats.target_only_fallback_tokens as u64, Ordering::Relaxed);
        self.dflash_last_accepted_run
            .store(stats.last_accepted_run as u64, Ordering::Relaxed);
        if stats.speculation_disabled {
            self.dflash_disabled_requests
                .fetch_add(1, Ordering::Relaxed);
            self.dflash_disabled_at_tokens.fetch_add(
                stats.speculation_disabled_at_tokens.unwrap_or(0) as u64,
                Ordering::Relaxed,
            );
        }
        // Counted separately from `speculation_disabled`: a re-qualifying
        // request can fire the gate and still finish with it open.
        self.dflash_disable_events
            .fetch_add(stats.disable_events as u64, Ordering::Relaxed);
        self.dflash_draft_sink_size
            .store(stats.draft_sink_size as u64, Ordering::Relaxed);
        self.dflash_draft_sliding_window
            .store(stats.draft_sliding_window as u64, Ordering::Relaxed);
        if stats.draft_ns > 0 {
            self.phase_timings.dflash_draft.record(stats.draft_ns);
        }
        if stats.target_verify_ns > 0 {
            self.phase_timings
                .dflash_target_verify
                .record(stats.target_verify_ns);
        }
    }

    pub(crate) fn record_phase(&self, phase: &'static str, elapsed_ns: u64) {
        let counter = match phase {
            "queue" => &self.phase_timings.queue,
            "prefill" => &self.phase_timings.prefill,
            "sampling" => &self.phase_timings.sampling,
            "grammar" => &self.phase_timings.grammar,
            "detokenization" => &self.phase_timings.detokenization,
            "enqueue_write" => &self.phase_timings.enqueue_write,
            _ => return,
        };
        counter.record(elapsed_ns);
    }

    pub(crate) fn record_request_decode(&self, tokens: usize, elapsed_ns: u64) {
        if tokens == 0 || elapsed_ns == 0 {
            return;
        }
        let milli_tok_s = (tokens as u128)
            .saturating_mul(1_000_000_000_000)
            .checked_div(elapsed_ns as u128)
            .unwrap_or(0)
            .min(u64::MAX as u128) as u64;
        self.last_request_decode_milli_tok_s
            .store(milli_tok_s, Ordering::Relaxed);
    }

    pub(crate) fn record_dflash_route_failure(&self, route: &str) {
        match route {
            "ane" => {
                self.dflash_ane_failures.fetch_add(1, Ordering::Relaxed);
            }
            "metal" => {
                self.dflash_metal_failures.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
    }

    pub(crate) fn record_ttft(&self, elapsed_ns: u64) {
        let mut values = self
            .ttft_ns
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if values.len() == 4_096 {
            values.pop_front();
        }
        values.push_back(elapsed_ns);
    }

    pub(crate) fn record_remote_transfer(&self, receipt: &RemoteReceiveReceipt) {
        self.remote_bytes_received
            .fetch_add(receipt.installed_bytes, Ordering::Relaxed);
        self.remote_transfer_ns
            .fetch_add(receipt.transfer_commit_ns, Ordering::Relaxed);
        push_window(&self.ingress_window, receipt.installed_bytes);
        // A committed transfer closes the breaker: the run of failures it
        // counts must be consecutive, not cumulative.
        self.remote_consecutive_failures.store(0, Ordering::Relaxed);
        self.remote_probe_in_flight.store(false, Ordering::Relaxed);
        *self
            .remote_cooldown_until
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
        let hidden_fraction = receipt.producer.as_ref().map_or(0.0, |producer| {
            let start = producer
                .transfer_start_unix_ns
                .max(producer.prefill_start_unix_ns);
            let end = producer
                .transfer_acked_unix_ns
                .min(producer.prefill_end_unix_ns);
            let transfer = producer
                .transfer_acked_unix_ns
                .saturating_sub(producer.transfer_start_unix_ns);
            if transfer == 0 {
                0.0
            } else {
                end.saturating_sub(start) as f64 / transfer as f64
            }
        });
        let mut transfers = self
            .remote_transfers
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if transfers.len() == 64 {
            transfers.pop_front();
        }
        transfers.push_back(RemoteTransferSample {
            transfer_id: receipt.transfer_id.clone(),
            installed_bytes: receipt.installed_bytes,
            transfer_ns: receipt.transfer_commit_ns,
            active_drain_ns: receipt.phases.segment_read_ns,
            control_ns: receipt.control_ns,
            accept_ns: receipt.accept_ns,
            hidden_fraction,
        });
    }

    /// The configured remote endpoint pair, when a receiver is bound. Node
    /// labels are derived from the live configuration and the local bind
    /// address, never from a hardcoded demo topology.
    pub(crate) fn remote_endpoints(&self) -> Option<(String, String)> {
        let remote = self.inference.as_ref()?.remote_prefill.as_ref()?;
        Some(remote.endpoints())
    }
}

fn validate_remote_dflash_policy(
    producer_mode: Option<Nvfp4ProducerMode>,
    dflash_configured: bool,
) -> Result<(), InferenceLoadError> {
    if producer_mode == Some(Nvfp4ProducerMode::Native) && dflash_configured {
        return Err(InferenceLoadError::Remote(
            "native NVFP4 fast-lane speculative decode is unqualified; omit --dflash and use plain NVFP4 decode, or route speculative serving to the kquant lane"
                .into(),
        ));
    }
    Ok(())
}

/// Append one sample to a rolling window, dropping everything older than
/// [`RATE_WINDOW`] first so a quiet process never reports stale traffic.
fn push_window(window: &Mutex<VecDeque<(Instant, u64)>>, value: u64) {
    let mut samples = window.lock().unwrap_or_else(|error| error.into_inner());
    prune_window(&mut samples);
    if samples.len() == RATE_WINDOW_SAMPLES {
        samples.pop_front();
    }
    samples.push_back((Instant::now(), value));
}

fn window_total(window: &Mutex<VecDeque<(Instant, u64)>>) -> u64 {
    let mut samples = window.lock().unwrap_or_else(|error| error.into_inner());
    prune_window(&mut samples);
    samples.iter().map(|(_, value)| *value).sum()
}

fn window_rate(window: &Mutex<VecDeque<(Instant, u64)>>) -> f64 {
    window_total(window) as f64 / RATE_WINDOW.as_secs_f64()
}

fn prune_window(samples: &mut VecDeque<(Instant, u64)>) {
    let now = Instant::now();
    while samples
        .front()
        .is_some_and(|(at, _)| now.duration_since(*at) > RATE_WINDOW)
    {
        samples.pop_front();
    }
}

impl RemotePrefillRuntime {
    pub fn mode(&self) -> RemotePrefillMode {
        self.mode
    }

    /// Deepest prompt the remote lane will attempt; deeper prompts prefill
    /// locally instead of starving the producer-wait budget.
    pub fn max_prompt_tokens(&self) -> usize {
        self.receiver.config().remote_max_prompt_tokens
    }

    /// `(src_node, dst_node)` read from the live receiver configuration: the
    /// configured producer control endpoint is the source, the advertised or
    /// bound receive address is the destination.
    fn endpoints(&self) -> (String, String) {
        let config = self.receiver.config();
        let source = config
            .producer_control
            .as_ref()
            .map(|control| control.server_name.clone())
            .unwrap_or_else(|| "unsolicited-producer".to_string());
        let destination = config
            .advertised_receiver_host
            .clone()
            .unwrap_or_else(|| config.listen.to_string());
        (source, destination)
    }

    pub fn receive(
        &self,
        session: &mut Session,
        dflash: Option<&mut DFlashAssistant>,
        prompt_witnesses: &[u32],
        multimodal: Option<(MultimodalIdentityV2, Vec<PrefillControlSegmentV2>)>,
        max_context: usize,
    ) -> Result<RemoteReceiveReceipt, String> {
        let _lease = try_producer_lease(&self.producer_lease)?;
        self.receiver.receive(
            session,
            dflash,
            prompt_witnesses,
            multimodal,
            max_context,
            self.mode == RemotePrefillMode::Required,
        )
    }
}

fn try_producer_lease(gate: &Mutex<()>) -> Result<std::sync::MutexGuard<'_, ()>, String> {
    match gate.try_lock() {
        Ok(lease) => Ok(lease),
        Err(std::sync::TryLockError::WouldBlock) => Err(REMOTE_PRODUCER_BUSY.into()),
        Err(std::sync::TryLockError::Poisoned(_)) => {
            Err("remote producer admission gate is unhealthy; restart muser".into())
        }
    }
}

pub(crate) fn remote_producer_is_busy(error: &str) -> bool {
    error == REMOTE_PRODUCER_BUSY
}

fn sha256_file(path: &std::path::Path) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut input =
        std::fs::File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = input
            .read(&mut buffer)
            .map_err(|error| format!("read {}: {error}", path.display()))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn dflash_identity(path: &std::path::Path) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    if path.is_file() {
        return sha256_file(path);
    }
    let mut digest = Sha256::new();
    digest.update(b"muser-dflash-artifact-v1\0");
    for name in ["config.json", "model.safetensors"] {
        let file = path.join(name);
        let mut input = std::fs::File::open(&file)
            .map_err(|error| format!("open {}: {error}", file.display()))?;
        let mut buffer = [0u8; 1024 * 1024];
        loop {
            let count = input
                .read(&mut buffer)
                .map_err(|error| format!("read {}: {error}", file.display()))?;
            if count == 0 {
                break;
            }
            digest.update(&buffer[..count]);
        }
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn hex_digest(value: &[u8; 32]) -> String {
    value.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn require_strict_cross_vendor_qk(value: Option<&str>) -> Result<(), InferenceLoadError> {
    if value == Some("1") {
        return Ok(());
    }
    Err(InferenceLoadError::Remote(
        "remote CUDA cache consumption requires MUSER_CROSS_VENDOR_QK=1 so Metal uses the pinned cross-vendor math route"
            .into(),
    ))
}

fn load_default_dflash(
    path: &std::path::Path,
    model: &Model,
    target_backend: &str,
) -> Result<DFlashAssistant, muser_engine::dflash::DFlashError> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    if target_backend == "metal" {
        return DFlashAssistant::load_metal(path, model);
    }
    let _ = target_backend;
    DFlashAssistant::load(path, model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_rotation_is_cyclic_after_the_last_served_slot() {
        let order = |last_slot| {
            let mut slots = vec![3usize, 0, 2, 1];
            slots.sort_by_key(|slot| decode_rotation_key(last_slot, *slot));
            slots
        };
        assert_eq!(order(None), [0, 1, 2, 3]);
        assert_eq!(order(Some(1)), [2, 3, 0, 1]);
        assert_eq!(order(Some(3)), [0, 1, 2, 3]);
    }

    #[test]
    fn single_slot_decode_never_pays_the_multi_slot_coalescing_window() {
        assert!(!DecodeBatcher::new(true, 1).enabled);
        assert!(DecodeBatcher::new(true, 2).enabled);
        assert!(DecodeBatcher::new(true, 4).enabled);
        assert!(!DecodeBatcher::new(false, 4).enabled);
    }

    #[test]
    fn slot_admission_is_bounded_and_an_unhealthy_latch_is_terminal() {
        let slots = SlotPool::new(Vec::new());
        slots.state.lock().unwrap().waiting = MAX_QUEUED_REQUESTS;
        assert_eq!(
            slots.acquire(Duration::ZERO).err(),
            Some(SlotAcquireError::Overloaded)
        );

        slots.latch_unhealthy();
        assert!(!slots.is_healthy());
        assert_eq!(
            slots.acquire(Duration::ZERO).err(),
            Some(SlotAcquireError::Unhealthy)
        );
    }

    #[test]
    fn exactly_one_generation_is_hidden_from_serving_admission() {
        let (serving, staging) = split_staging_generation(vec![0, 1, 2, 3, 4], 4).unwrap();
        assert_eq!(serving, [0, 1, 2, 3]);
        assert_eq!(staging, 4);
        assert!(split_staging_generation(vec![0, 1, 2, 3], 4).is_err());
        assert!(split_staging_generation(vec![0, 1, 2, 3, 4, 5], 4).is_err());
    }

    #[test]
    fn no_model_path_means_no_measured_weights_bytes() {
        let s = ServerState::new(None);
        assert!(s.model_bytes.is_none());
    }

    #[test]
    fn missing_model_file_is_honestly_none_not_an_error() {
        let s = ServerState::new(Some("/nonexistent/path/does-not-exist.gguf"));
        assert!(s.model_bytes.is_none());
    }

    #[test]
    fn uptime_advances() {
        let s = ServerState::new(None);
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert!(s.uptime_s() > 0.0);
    }

    #[test]
    fn a_recovered_lease_latches_degraded() {
        let s = ServerState::new(None);
        assert!(!s.degraded());
        s.record_lock_recovery();
        assert!(s.degraded());
    }

    #[test]
    fn the_remote_breaker_opens_after_a_run_of_failures_and_a_commit_closes_it() {
        let s = ServerState::new(None);
        assert!(s.remote_route_is_open());
        for _ in 0..REMOTE_BREAKER_FAILURES {
            s.record_remote_failure("producer unreachable");
        }
        assert!(!s.remote_route_is_open());
        assert_eq!(
            s.remote_receive_failures.load(Ordering::Relaxed),
            REMOTE_BREAKER_FAILURES
        );
        s.record_remote_transfer(&RemoteReceiveReceipt {
            transfer_id: "transfer-1".into(),
            generation: 1,
            installed_segments: 1,
            installed_bytes: 1,
            control_ns: 0,
            accept_ns: 0,
            transfer_commit_ns: 1,
            total_ns: 1,
            producer: None,
            components: Default::default(),
            phases: Default::default(),
        });
        assert!(s.remote_route_is_open());
    }

    #[test]
    fn local_single_flight_contention_does_not_poison_the_remote_breaker() {
        let s = ServerState::new(None);
        for _ in 0..(REMOTE_BREAKER_FAILURES * 3) {
            s.record_remote_failure(REMOTE_PRODUCER_BUSY);
        }
        assert!(s.remote_route_is_open());
        assert_eq!(s.remote_receive_failures.load(Ordering::Relaxed), 0);
        assert_eq!(s.remote_consecutive_failures.load(Ordering::Relaxed), 0);
        assert!(s.last_remote_error.lock().unwrap().is_none());
    }

    #[test]
    fn failures_must_be_consecutive_to_open_the_breaker() {
        let s = ServerState::new(None);
        s.record_remote_failure("first");
        s.record_remote_transfer(&RemoteReceiveReceipt {
            transfer_id: "transfer-2".into(),
            generation: 1,
            installed_segments: 1,
            installed_bytes: 1,
            control_ns: 0,
            accept_ns: 0,
            transfer_commit_ns: 1,
            total_ns: 1,
            producer: None,
            components: Default::default(),
            phases: Default::default(),
        });
        for _ in 0..(REMOTE_BREAKER_FAILURES - 1) {
            s.record_remote_failure("later");
        }
        assert!(s.remote_route_is_open());
    }

    #[test]
    fn an_expired_cooldown_admits_one_half_open_probe() {
        let s = ServerState::new(None);
        for _ in 0..REMOTE_BREAKER_FAILURES {
            s.record_remote_failure("producer unreachable");
        }
        assert!(!s.remote_route_is_open());
        *s.remote_cooldown_until.lock().unwrap() = Some(Instant::now() - Duration::from_secs(1));
        assert!(s.remote_route_is_open());
        assert!(!s.remote_route_is_open());
        // A failed probe re-arms the cool-down immediately.
        s.record_remote_failure("still down");
        assert!(!s.remote_route_is_open());
        // And a committed probe transfer heals the route fully.
        *s.remote_cooldown_until.lock().unwrap() = Some(Instant::now() - Duration::from_secs(1));
        assert!(s.remote_route_is_open());
        s.record_remote_transfer(&RemoteReceiveReceipt {
            transfer_id: "probe".into(),
            generation: 2,
            installed_segments: 1,
            installed_bytes: 1,
            control_ns: 0,
            accept_ns: 0,
            transfer_commit_ns: 1,
            total_ns: 1,
            producer: None,
            components: Default::default(),
            phases: Default::default(),
        });
        assert!(s.remote_route_is_open());
        assert!(s.remote_route_is_open());
    }

    #[test]
    fn remote_cache_math_route_requires_exact_opt_in() {
        assert!(require_strict_cross_vendor_qk(Some("1")).is_ok());
        for value in [
            None,
            Some(""),
            Some("0"),
            Some("true"),
            Some(" 1"),
            Some("1 "),
        ] {
            let error = require_strict_cross_vendor_qk(value).unwrap_err();
            assert!(error
                .to_string()
                .contains("requires MUSER_CROSS_VENDOR_QK=1"));
        }
    }

    #[test]
    fn the_single_flight_producer_gate_refuses_a_tcp_backlog_queue() {
        let gate = Mutex::new(());
        let first = try_producer_lease(&gate).unwrap();
        assert_eq!(
            try_producer_lease(&gate).unwrap_err(),
            "remote producer is busy with another prefill"
        );
        drop(first);
        assert!(try_producer_lease(&gate).is_ok());
    }

    #[test]
    fn native_nvfp4_remote_route_rejects_unqualified_dflash() {
        let error =
            validate_remote_dflash_policy(Some(Nvfp4ProducerMode::Native), true).unwrap_err();
        assert!(error
            .to_string()
            .contains("speculative decode is unqualified"));
        assert!(validate_remote_dflash_policy(Some(Nvfp4ProducerMode::Native), false).is_ok());
        assert!(validate_remote_dflash_policy(Some(Nvfp4ProducerMode::Exact), true).is_ok());
        assert!(validate_remote_dflash_policy(None, true).is_ok());
    }
}
