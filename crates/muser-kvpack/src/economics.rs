//! Cache-economics accounting — the dashboard's hero panel data source.
//!
//! **No direct Ferrite source — this is muser-original**, specified in
//! full by `docs/kvpack-economics.md`. Implements the accounting rules
//! defined there against the real kvpack success edge
//! (`sink.commit_restore()` returning `Ok`, per
//! `kvpack/src/restore/plan.rs`):
//!
//! - A "hit" is a restore that was cryptographically verified **and**
//!   committed into engine memory. A lookup that matched but failed
//!   verification, OOM'd, or was cancelled is a miss with a wasted probe —
//!   it contributes zero saved value. Continuing the *same* live session (no
//!   restore ran — the KV was never evicted) and a disaggregated remote
//!   prefill (real compute, just not local) are both real, useful outcomes,
//!   but neither is a served hit either: see
//!   [`EconomicsCounters::record_session_continuation`] and
//!   [`EconomicsCounters::record_disagg_prefill`], which land in their own
//!   fields and never touch `cache_hits`/`bytes_served_from_cache`/any tier —
//!   the design doc's anti-self-dealing rule (§1.4), generalized past its
//!   original producer-readback case.
//! - `bytes_from_cache` = the exact KV bytes moved out of a tier and
//!   installed. For durable/remote tiers this must be the caller-supplied,
//!   manifest-authenticated `complete_restored_bytes`
//!   ([`RestoreBytes::Manifest`]), **never** inferred from
//!   `tokens × per_token_bytes` — Muse's 39 SWA layers don't grow linearly
//!   past 2048 tokens, so a naive per-token estimate overstates value on
//!   long contexts. The resident tier has no separate manifest read (it's a
//!   live in-process copy), so it may fall back to the token-derived
//!   estimate ([`RestoreBytes::Estimated`]) — labeled as such, never
//!   presented as measured.
//! - `bytes_recomputed` = KV for the suffix still prefilled after a
//!   partial-prefix hit. **The matched prefix is never double-counted as
//!   both saved and recomputed.**
//!
//! Feeds `docs/metrics-schema.md` §2 `Economics`: `cache_hits`,
//! `cache_misses`, `hit_rate`, `bytes_served_from_cache`,
//! `bytes_recomputed`, `restore_speedup` (target/measured range
//! 21.9-30.1x), `derived.{seconds_saved,gflops_avoided,joules_saved}`,
//! tiered by `resident|ssd|remote`.
//!
//! ## Status (real-telemetry pass)
//!
//! This module is now a **real, live counter**, not a stub: every field on
//! [`EconomicsSnapshot`] is produced by [`EconomicsCounters`] from actual
//! `record_restore`/`record_prefill_miss` calls, guarded by a `Mutex` (this
//! is a request-rate structure, not a per-token hot path, so a lock is the
//! right tool — no atomics-with-float-bit-tricks). `muser-server` records real
//! prefix misses and exact resident/durable restores through the shared
//! [`EconomicsCounters`]; an idle server therefore reports zero while a cache
//! workload reports measured traffic. See `docs/telemetry.md` for the
//! field-by-field honesty accounting and `tests` below for a worked example.
//!
//! `gflops_avoided` reports only the defensible linear weight-matmul floor
//! term (`docs/kvpack-economics.md` §1.3's first term); the attention-FLOPs
//! term needs per-layer head/window data this crate doesn't have wired in,
//! so it is left out rather than guessed — the field undercounts, not
//! overclaims.
//!
//! The binary-hit-or-miss model here (a request is either a full restore or
//! a full prefill) is deliberately simpler than the full partial-prefix
//! model in `docs/kvpack-economics.md` §1.2 (matched-prefix + recomputed
//! suffix in the same request). Prefix reuse records restored and recomputed
//! token counts separately; this aggregate retains the simpler request-level
//! presentation so dashboard accounting remains stable.

use std::sync::Mutex;
use std::time::Instant;

use serde::Serialize;

/// KV bytes per layer per cached token: 2 KV heads × 128 head_dim × (K+V) ×
/// 2 B fp16 = 1024 B. `[measured]`, `docs/muser-architecture.md` §0.
pub const KV_BYTES_PER_LAYER_TOKEN: u64 = 1024;
/// The 13 position-free full layers: grow unbounded with context.
/// `[measured]`.
pub const NOPE_LAYERS: u64 = 13;
/// The 39 sliding-window layers: ring-bounded at [`SWA_WINDOW`].
/// `[measured]`.
pub const SWA_LAYERS: u64 = 39;
/// SWA ring window, in tokens. `[measured]`.
pub const SWA_WINDOW: u64 = 2048;

/// EMA smoothing factor for paired, real restore/prefill timings.
const RESTORE_SPEEDUP_EMA_K: f64 = 0.25;

/// Active parameters/token for Muse Glimmer-30B (dense). `[target]` — used
/// only for the linear weight-matmul floor of `gflops_avoided`
/// (`docs/kvpack-economics.md` §1.3); an MoE variant would need
/// active-expert params here instead of the dense total.
const GLIMMER_30B_ACTIVE_PARAMS: f64 = 30e9;

/// The defensible floor term of `gflops_avoided(h)`: prefill weight-matmul
/// FLOPs the engine did not run for `tokens` restored positions. Linear in
/// `tokens`, so unlike `kv_bytes` it needs no SWA-window awareness. Excludes
/// the attention QKᵀ+AV term (`docs/kvpack-economics.md` §1.3's second term)
/// — this crate has no per-layer head/window data wired in to compute it,
/// so the result undercounts rather than guesses.
fn gflops_avoided_linear(tokens: u64) -> f64 {
    2.0 * GLIMMER_30B_ACTIVE_PARAMS * tokens as f64 / 1e9
}

/// Paired wall-clock evidence for one restore. Both durations must describe
/// the same prompt cut, model, engine identity, and starting state.
#[derive(Debug, Clone, Copy)]
pub struct RestoreTiming {
    pub restore_seconds: f64,
    pub local_prefill_seconds: f64,
}

impl RestoreTiming {
    /// Time `restore` with `Instant`, pairing it with a `local_prefill_seconds`
    /// the caller already measured for the identical cut — the same
    /// `Instant::now()`/`elapsed()` bracket `muser-bench/src/kvpack.rs` hand-
    /// rolls at each call site, wrapped once so `restore_speedup` gets fed
    /// real timing without every caller re-deriving it. Returns `restore`'s
    /// own result alongside the timing so callers can still handle a
    /// `Result` before deciding whether to record anything.
    pub fn measure<T>(local_prefill_seconds: f64, restore: impl FnOnce() -> T) -> (T, Self) {
        let started = Instant::now();
        let result = restore();
        let restore_seconds = started.elapsed().as_secs_f64();
        (
            result,
            RestoreTiming {
                restore_seconds,
                local_prefill_seconds,
            },
        )
    }
}

/// Source of `bytes_from_cache` for one [`EconomicsCounters::record_restore`]
/// call. `docs/kvpack-economics.md` §1.2 requires the authenticated manifest
/// byte count wherever one exists; `Estimated` exists only for tiers (today:
/// resident) with no separate manifest read to cite.
#[derive(Debug, Clone, Copy)]
pub enum RestoreBytes {
    /// `complete_restored_bytes` from the authenticated manifest — required
    /// for durable/remote restores. SWA-aware by construction; never a
    /// linear estimate.
    Manifest(u64),
    /// `kv_bytes(tokens)`, used only because no manifest byte count exists
    /// for this restore. Reported as an estimate, not a measurement.
    Estimated,
}

/// Bytes occupied by the 13 NoPE (grow-unbounded) layers for `tokens`
/// live-cached positions.
pub fn nope_bytes(tokens: u64) -> u64 {
    NOPE_LAYERS * KV_BYTES_PER_LAYER_TOKEN * tokens
}

/// Bytes occupied by the 39 SWA (ring-windowed at [`SWA_WINDOW`]) layers for
/// `tokens` live-cached positions.
pub fn swa_bytes(tokens: u64) -> u64 {
    SWA_LAYERS * KV_BYTES_PER_LAYER_TOKEN * tokens.min(SWA_WINDOW)
}

/// Total structural KV bytes (both classes) for `tokens` live-cached
/// positions — the same quantity `docs/kvpack-economics.md` §1.2 calls
/// `kv_bytes(T)`.
pub fn kv_bytes(tokens: u64) -> u64 {
    nope_bytes(tokens) + swa_bytes(tokens)
}

/// Which storage tier served a restore. Product-tier names are intentionally
/// narrower than the audited in-tree kvpack snapshot's internal route types;
/// this is the stable dashboard adapter vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    /// Unified-memory / VRAM resident copy. Fastest; ~memcpy cost.
    Resident,
    /// Local NVMe kvpack artifact.
    Ssd,
    /// Authenticated mTLS-TCP remote path. Release qualification requires a
    /// three-handoff median of at least 3.0 Gbps effective installed-payload
    /// throughput; the tier name makes no generic Ethernet link-class claim.
    /// RDMA/RoCE remains outside the v0.1 feature contract.
    Remote,
}

/// One storage tier's live occupancy + hit count — wire-shape match for
/// `docs/metrics-schema.json` `#/$defs/StorageTier`.
///
/// `resident_bytes` here tracks *cumulative bytes ever served via this
/// tier*, not a live occupancy gauge (that needs real per-tier eviction
/// bookkeeping this crate doesn't implement yet — see `docs/telemetry.md`).
/// `capacity_bytes` is a configured operational budget, not a measurement.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct StorageTier {
    pub resident_bytes: u64,
    pub capacity_bytes: u64,
    pub hits: u64,
}

/// The three tiers `docs/kvpack-economics.md` §3 attributes restores to.
/// Wire-shape match for `Economics.tiers` (`docs/metrics-schema.md`'s
/// `tiers.remote`, renamed from `tiers.rdma_pool`). Carries
/// [`Tier::Remote`]'s stats, which today is the mTLS-TCP remote path, not
/// RDMA — see that variant's doc.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct Tiers {
    pub resident: StorageTier,
    pub ssd: StorageTier,
    pub remote: StorageTier,
}

/// Derived counterfactual value — wire-shape match for `Economics.derived`.
/// All three are "what would recomputing have cost, minus what the restore
/// actually cost" (`docs/kvpack-economics.md` §1.3): `seconds_saved` is real
/// once any timed restore has been recorded. `gflops_avoided` reports only
/// the linear weight-matmul floor term (`2 * P_active * N_h`, `P_active`
/// `[target]` for Glimmer-30B) — the attention-FLOPs term is not computed,
/// so this is a defensible undercount, never the full formula. `joules_saved`
/// has no writer: it needs a hardware power calibration this crate doesn't
/// perform (`docs/kvpack-economics.md` §1.3), so it stays zero and must
/// always be reported `target`/`mock`, never `measured`.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct Derived {
    pub seconds_saved: f64,
    pub gflops_avoided: f64,
    pub joules_saved: f64,
}

/// Wire-shape match for `docs/metrics-schema.json` `#/$defs/Economics`.
/// Returned by [`EconomicsCounters::report`] — this is exactly what
/// `muser-server` embeds as `MetricsSnapshot.economics`, unmodified.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct EconomicsSnapshot {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub hit_rate: f64,
    pub bytes_served_from_cache: u64,
    pub bytes_recomputed: u64,
    pub restore_ops: u64,
    pub prefill_ops: u64,
    pub restore_speedup: f64,
    pub derived: Derived,
    pub tiers: Tiers,
    /// Prompt continuations of the *same* live session: no restore ran (the
    /// KV was never evicted), so this is real and useful but not a served
    /// hit — kept out of `cache_hits` per the design doc's anti-self-dealing
    /// rule (§1.4).
    pub session_continuation_hits: u64,
    /// Disaggregated remote prefills (compute ran on another node, e.g.
    /// GX10, with KV installed over the wire): real prefill work, not a
    /// cache restore, so it is counted here instead of `cache_hits`.
    pub disagg_prefills: u64,
    /// Bytes installed by disaggregated remote prefills — the caller-
    /// supplied authenticated transfer size, never folded into
    /// `bytes_served_from_cache`.
    pub disagg_bytes_installed: u64,
}

/// [`EconomicsCounters::report`]'s return value: the wire snapshot plus the
/// honesty metadata `muser-server` needs to pick `measured` vs `target` for
/// `restore_speedup` — kept out of [`EconomicsSnapshot`] itself so the wire
/// object stays exactly schema-shaped (`additionalProperties: false`).
#[derive(Debug, Clone, Copy)]
pub struct EconomicsReport {
    pub wire: EconomicsSnapshot,
    /// `true` once at least one real, timed restore has updated the
    /// `restore_speedup` EMA — i.e. the number reflects this process's own
    /// measurement rather than the published 21.9-30.1x cross-check range
    /// (`docs/kvpack-economics.md` §1.3), which this module never
    /// substitutes as a value.
    pub restore_speedup_is_measured: bool,
}

#[derive(Debug, Default)]
struct Inner {
    cache_hits: u64,
    cache_misses: u64,
    bytes_served_from_cache: u64,
    bytes_recomputed: u64,
    restore_ops: u64,
    prefill_ops: u64,
    restore_speedup_ema: Option<f64>,
    derived: Derived,
    tiers: Tiers,
    session_continuation_hits: u64,
    disagg_prefills: u64,
    disagg_bytes_installed: u64,
}

/// Live, thread-safe cache-economics counters. One instance lives in
/// `muser-server`'s `ServerState` for the whole process lifetime; every
/// real kvpack restore/prefill event updates it through
/// [`record_restore`](Self::record_restore) /
/// [`record_prefill_miss`](Self::record_prefill_miss). This is real
/// accounting, not a simulation: it starts at all-zero and stays there
/// until real session traffic flows through the server, which is the
/// honest state of the world for a Phase-0/1 scaffold (see module docs).
#[derive(Debug, Default)]
pub struct EconomicsCounters(Mutex<Inner>);

impl EconomicsCounters {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a served cache hit: `tokens` is the matched-cut token count
    /// actually restored (never the recomputed suffix — see
    /// `docs/kvpack-economics.md` §1.2's anti-double-count rule), `tier` is
    /// which storage tier served it, `installed_bytes` is the byte source
    /// for `bytes_served_from_cache` (the authenticated manifest count for
    /// durable/remote tiers — see [`RestoreBytes`] — never a raw token
    /// estimate for those), and `timing` is the real measured wall time of
    /// transfer+verify+install if the caller has it.
    ///
    /// Economics remain zero unless `timing` contains paired, positive real
    /// measurements. A verified hit without timing still updates hit/byte
    /// counters; it never substitutes a precedent or projected speedup.
    pub fn record_restore(
        &self,
        tokens: u64,
        tier: Tier,
        installed_bytes: RestoreBytes,
        timing: Option<RestoreTiming>,
    ) {
        let bytes = match installed_bytes {
            RestoreBytes::Manifest(bytes) => bytes,
            RestoreBytes::Estimated => kv_bytes(tokens),
        };

        let mut inner = self.0.lock().unwrap_or_else(|e| e.into_inner());
        inner.cache_hits += 1;
        inner.restore_ops += 1;
        inner.bytes_served_from_cache += bytes;

        let tier_state = match tier {
            Tier::Resident => &mut inner.tiers.resident,
            Tier::Ssd => &mut inner.tiers.ssd,
            Tier::Remote => &mut inner.tiers.remote,
        };
        tier_state.hits += 1;
        tier_state.resident_bytes += bytes;

        if let Some(timing) = timing
            .filter(|timing| timing.restore_seconds > 0.0 && timing.local_prefill_seconds > 0.0)
        {
            let ratio = timing.local_prefill_seconds / timing.restore_seconds;
            inner.restore_speedup_ema = Some(match inner.restore_speedup_ema {
                Some(prev) => prev + (ratio - prev) * RESTORE_SPEEDUP_EMA_K,
                None => ratio,
            });
            inner.derived.seconds_saved +=
                (timing.local_prefill_seconds - timing.restore_seconds).max(0.0);
            inner.derived.gflops_avoided += gflops_avoided_linear(tokens);
        }
    }

    /// Record a full prefill (cache miss): `tokens` is the token count
    /// actually recomputed.
    pub fn record_prefill_miss(&self, tokens: u64) {
        let bytes = kv_bytes(tokens);
        let mut inner = self.0.lock().unwrap_or_else(|e| e.into_inner());
        inner.cache_misses += 1;
        inner.prefill_ops += 1;
        inner.bytes_recomputed += bytes;
    }

    /// Record only the recomputed suffix of a successful ancestor restore.
    /// This contributes work/bytes but not a second logical cache miss, so a
    /// single request cannot inflate both the hit and miss denominators.
    pub fn record_prefill_suffix(&self, tokens: u64) {
        if tokens == 0 {
            return;
        }
        let bytes = kv_bytes(tokens);
        let mut inner = self.0.lock().unwrap_or_else(|e| e.into_inner());
        inner.prefill_ops += 1;
        inner.bytes_recomputed += bytes;
    }

    /// Record a prompt continuation of the *same* live session: the request
    /// prefix matched what this session already held resident, so no
    /// restore ran (the KV was never evicted). Real and useful, but not a
    /// served hit under `docs/kvpack-economics.md` §1.1's definition — no
    /// `commit_restore` edge fired. Never touches `cache_hits`,
    /// `bytes_served_from_cache`, or any tier.
    pub fn record_session_continuation(&self, tokens: usize) {
        if tokens == 0 {
            return;
        }
        let mut inner = self.0.lock().unwrap_or_else(|e| e.into_inner());
        inner.session_continuation_hits += 1;
    }

    /// Record a disaggregated remote prefill: `tokens` positions were
    /// genuinely prefilled, just on another node (e.g. GX10), with the
    /// resulting KV installed over the wire. Real compute, not a cache
    /// restore, so it is counted here instead of `cache_hits`.
    /// `installed_bytes` is the authenticated transfer size the caller
    /// already has from the receipt. Never touches `cache_hits`,
    /// `bytes_served_from_cache`, or any tier.
    pub fn record_disagg_prefill(&self, tokens: usize, installed_bytes: u64) {
        if tokens == 0 {
            return;
        }
        let mut inner = self.0.lock().unwrap_or_else(|e| e.into_inner());
        inner.disagg_prefills += 1;
        inner.disagg_bytes_installed += installed_bytes;
    }

    /// Snapshot the current counters as the wire `Economics` shape, plus
    /// whether `restore_speedup` reflects a real measurement.
    pub fn report(&self) -> EconomicsReport {
        let inner = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let denom = inner.cache_hits + inner.cache_misses;
        let hit_rate = if denom == 0 {
            0.0
        } else {
            inner.cache_hits as f64 / denom as f64
        };
        let (restore_speedup, restore_speedup_is_measured) = match inner.restore_speedup_ema {
            Some(v) => (v, true),
            None => (0.0, false),
        };
        EconomicsReport {
            wire: EconomicsSnapshot {
                cache_hits: inner.cache_hits,
                cache_misses: inner.cache_misses,
                hit_rate,
                bytes_served_from_cache: inner.bytes_served_from_cache,
                bytes_recomputed: inner.bytes_recomputed,
                restore_ops: inner.restore_ops,
                prefill_ops: inner.prefill_ops,
                restore_speedup,
                derived: inner.derived,
                tiers: inner.tiers,
                session_continuation_hits: inner.session_continuation_hits,
                disagg_prefills: inner.disagg_prefills,
                disagg_bytes_installed: inner.disagg_bytes_installed,
            },
            restore_speedup_is_measured,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_honestly_at_zero() {
        let c = EconomicsCounters::new();
        let r = c.report();
        assert_eq!(r.wire.cache_hits, 0);
        assert_eq!(r.wire.cache_misses, 0);
        assert_eq!(r.wire.hit_rate, 0.0);
        assert_eq!(r.wire.bytes_served_from_cache, 0);
        assert_eq!(r.wire.derived.seconds_saved, 0.0);
        assert!(!r.restore_speedup_is_measured);
        assert_eq!(r.wire.restore_speedup, 0.0);
    }

    #[test]
    fn kv_bytes_matches_the_two_class_split() {
        // Below the SWA window: both classes grow linearly.
        assert_eq!(nope_bytes(100), 13 * 1024 * 100);
        assert_eq!(swa_bytes(100), 39 * 1024 * 100);
        // Past the SWA window: SWA is flat, NoPE keeps growing.
        assert_eq!(swa_bytes(5000), 39 * 1024 * 2048);
        assert_eq!(nope_bytes(5000), 13 * 1024 * 5000);
        assert_eq!(kv_bytes(5000), nope_bytes(5000) + swa_bytes(5000));
    }

    #[test]
    fn record_restore_with_real_timing_is_measured() {
        let c = EconomicsCounters::new();
        // Paired measurements for the identical cut: 0.05s restore versus
        // 12.99s local prefill.
        c.record_restore(
            4000,
            Tier::Resident,
            RestoreBytes::Estimated,
            Some(RestoreTiming {
                restore_seconds: 0.05,
                local_prefill_seconds: 12.99,
            }),
        );
        let r = c.report();
        assert_eq!(r.wire.cache_hits, 1);
        assert_eq!(r.wire.restore_ops, 1);
        assert_eq!(r.wire.bytes_served_from_cache, kv_bytes(4000));
        assert!(r.restore_speedup_is_measured);
        // prefill_equiv/restore = (4000/308)/0.05 =~ 259x for this single
        // fast local hit -- sanity check it's in a plausible ballpark, not
        // asserting an exact float.
        assert!(r.wire.restore_speedup > 50.0);
        assert!(r.wire.derived.seconds_saved > 0.0);
        assert_eq!(r.wire.tiers.resident.hits, 1);
        // The linear weight-matmul floor is real and positive once timing
        // was paired; it must equal the direct formula, not a guess.
        assert_eq!(r.wire.derived.gflops_avoided, gflops_avoided_linear(4000));
        assert!(r.wire.derived.gflops_avoided > 0.0);
    }

    #[test]
    fn record_restore_without_timing_leaves_gflops_avoided_at_zero() {
        // No paired timing means no counterfactual is defensible -- neither
        // seconds_saved nor gflops_avoided may become nonzero from a bare
        // hit count.
        let c = EconomicsCounters::new();
        c.record_restore(4000, Tier::Resident, RestoreBytes::Estimated, None);
        let r = c.report().wire;
        assert_eq!(r.cache_hits, 1);
        assert_eq!(r.derived.gflops_avoided, 0.0);
        assert_eq!(r.derived.seconds_saved, 0.0);
    }

    #[test]
    fn manifest_bytes_are_used_verbatim_not_rederived_from_tokens() {
        // A durable/remote restore's bytes come from the authenticated
        // manifest, which for SWA-aware Muse KV never equals the naive
        // per-token estimate -- assert the counter honors the manifest
        // value even where it disagrees with kv_bytes(tokens).
        let c = EconomicsCounters::new();
        let manifest_bytes = kv_bytes(4000) - 4096;
        c.record_restore(
            4000,
            Tier::Remote,
            RestoreBytes::Manifest(manifest_bytes),
            None,
        );
        let r = c.report().wire;
        assert_eq!(r.bytes_served_from_cache, manifest_bytes);
        assert_ne!(r.bytes_served_from_cache, kv_bytes(4000));
        assert_eq!(r.tiers.remote.hits, 1);
        assert_eq!(r.tiers.remote.resident_bytes, manifest_bytes);
    }

    #[test]
    fn tier_remote_serializes_as_remote() {
        // `Tier::RdmaPool` was renamed to `Tier::Remote`; the wire label
        // must be the honest "remote", not a leftover "rdma_pool"/"rdmaPool".
        assert_eq!(serde_json::to_string(&Tier::Remote).unwrap(), "\"remote\"");
        assert_eq!(
            serde_json::to_string(&Tier::Resident).unwrap(),
            "\"resident\""
        );
        assert_eq!(serde_json::to_string(&Tier::Ssd).unwrap(), "\"ssd\"");
    }

    #[test]
    fn restore_timing_measure_wraps_instant_and_returns_the_result() {
        let (value, timing) = RestoreTiming::measure(1.0, || {
            std::thread::sleep(std::time::Duration::from_millis(1));
            42
        });
        assert_eq!(value, 42);
        assert!(timing.restore_seconds > 0.0);
        assert_eq!(timing.local_prefill_seconds, 1.0);
    }

    #[test]
    fn record_prefill_miss_is_a_real_miss() {
        let c = EconomicsCounters::new();
        c.record_prefill_miss(8000);
        let r = c.report();
        assert_eq!(r.wire.cache_misses, 1);
        assert_eq!(r.wire.prefill_ops, 1);
        assert_eq!(r.wire.bytes_recomputed, kv_bytes(8000));
        assert_eq!(r.wire.hit_rate, 0.0);
    }

    #[test]
    fn restored_suffix_counts_work_without_double_counting_the_request() {
        let c = EconomicsCounters::new();
        c.record_restore(256, Tier::Resident, RestoreBytes::Estimated, None);
        c.record_prefill_suffix(17);
        let r = c.report().wire;
        assert_eq!((r.cache_hits, r.cache_misses), (1, 0));
        assert_eq!(r.prefill_ops, 1);
        assert_eq!(r.bytes_recomputed, kv_bytes(17));
    }

    #[test]
    fn hit_rate_is_hits_over_hits_plus_misses() {
        let c = EconomicsCounters::new();
        c.record_restore(
            1000,
            Tier::Ssd,
            RestoreBytes::Manifest(kv_bytes(1000)),
            None,
        );
        c.record_restore(
            1000,
            Tier::Ssd,
            RestoreBytes::Manifest(kv_bytes(1000)),
            None,
        );
        c.record_prefill_miss(1000);
        let r = c.report();
        assert_eq!(r.wire.cache_hits, 2);
        assert_eq!(r.wire.cache_misses, 1);
        assert!((r.wire.hit_rate - 2.0 / 3.0).abs() < 1e-9);
        // No paired timing was given, so the hit is counted but economics
        // remain unavailable rather than using a modeled fallback.
        assert!(!r.restore_speedup_is_measured);
        assert_eq!(r.wire.restore_speedup, 0.0);
        assert_eq!(r.wire.derived.seconds_saved, 0.0);
    }

    #[test]
    fn session_continuation_and_disagg_prefill_never_move_cache_hits_or_bytes() {
        // Both are real, useful outcomes but neither is a served hit under
        // docs/kvpack-economics.md §1.1/§1.4 -- they must land only in their
        // own fields, never inflate the headline hit/byte counters or any
        // storage tier.
        let c = EconomicsCounters::new();
        c.record_session_continuation(512);
        c.record_disagg_prefill(2048, 9_999);
        let r = c.report().wire;
        assert_eq!(r.cache_hits, 0);
        assert_eq!(r.cache_misses, 0);
        assert_eq!(r.bytes_served_from_cache, 0);
        assert_eq!(r.restore_ops, 0);
        assert_eq!(r.tiers.resident.hits, 0);
        assert_eq!(r.tiers.ssd.hits, 0);
        assert_eq!(r.tiers.remote.hits, 0);
        assert_eq!(r.session_continuation_hits, 1);
        assert_eq!(r.disagg_prefills, 1);
        assert_eq!(r.disagg_bytes_installed, 9_999);
    }

    #[test]
    fn session_continuation_and_disagg_prefill_ignore_zero_tokens() {
        let c = EconomicsCounters::new();
        c.record_session_continuation(0);
        c.record_disagg_prefill(0, 500);
        let r = c.report().wire;
        assert_eq!(r.session_continuation_hits, 0);
        assert_eq!(r.disagg_prefills, 0);
        assert_eq!(r.disagg_bytes_installed, 0);
    }
}
