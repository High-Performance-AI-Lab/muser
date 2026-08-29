//! `GET /snapshot` (poll) + `GET /telemetry` (SSE push) — the dashboard
//! contract. **No direct Ferrite source** — this is muser-original,
//! specified in full by `docs/metrics-schema.md` and
//! `docs/metrics-schema.json` (Draft 2020-12), which this module must
//! satisfy exactly (the dashboard renders entirely from that contract).
//!
//! Emits from `muser-engine`'s live counters + `muser-kvpack`'s
//! `economics` module with no extra bookkeeping: one engine-held state
//! struct (`state::ServerState`), `build_snapshot` serializes all of it.
//!
//! The hero panel this feeds: kvpack cache economics — GB-from-cache vs
//! GB-recomputed = value quantified (`docs/kvpack-economics.md`).
//!
//! ## Status (real-telemetry pass)
//!
//! This module now defines the full `MetricsSnapshot` wire shape and
//! assembles it from real counters where `ServerState` has them (economics,
//! sessions, event log, uptime, request/viewer counts, optionally a real
//! `--model` file size, and committed disaggregation receipts), and
//! clearly-placeholder values elsewhere (node/GPU telemetry and egress) —
//! because those measurements are unavailable. Every section carries an explicit honesty
//! tag in the `_honesty` extension so a live dashboard never has to guess
//! which numbers are real; `docs/telemetry.md` is the field-by-field
//! reference this file must stay in sync with.
//!
//! Unavailable nodes, transfers, and optimizations are emitted as empty
//! collections. The live endpoint never substitutes simulated product data.

use serde::Serialize;

use muser_kvpack::economics::{self as econ, Derived, Tiers};

use crate::session::{SessionEvent, SessionView};
use crate::state::ServerState;
use crate::timefmt;

/// Contract major version. `docs/metrics-schema.json` `#/$defs/schema_version`.
pub const SCHEMA_VERSION: u32 = 1;

/// No model means no observed file size. The zero is tagged mock and is never
/// replaced by a historical estimate.
const DEFAULT_WEIGHTS_BYTES: u64 = 0;

/// Configured resident KV budget used as the occupancy-meter denominator —
/// an operational target, not a measurement.
const KV_CAPACITY_BYTES: u64 = 0;

// ================================================================= enums

macro_rules! wire_enum {
    ($name:ident { $($variant:ident => $rename:literal),+ $(,)? }) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
        pub enum $name {
            $(#[serde(rename = $rename)] $variant),+
        }
    };
}

wire_enum!(NodeRole { Prefill => "prefill", Decode => "decode" });
wire_enum!(NodeKind { Gx10 => "gx10", Gb10 => "gb10", M3Ultra => "m3ultra" });
wire_enum!(NodeState { Idle => "idle", Prefilling => "prefilling", Decoding => "decoding", Offline => "offline" });
wire_enum!(TransferPhase { StreamingNope => "streaming_nope", ShippingSwa => "shipping_swa", Done => "done" });
wire_enum!(TrickName {
    Relocation => "relocation",
    GqaHeadpack => "gqa_headpack",
    RetrievalSpecdec => "retrieval_specdec",
    Disaggregation => "disaggregation",
    AneConcurrency => "ane_concurrency",
    KvpackRestore => "kvpack_restore",
    DispatchBatching => "dispatch_batching",
});
wire_enum!(TrickKind { Multiplier => "multiplier", Percent => "percent", Ratio => "ratio", Flag => "flag", Killed => "killed" });

// The three-way honesty vocabulary the dashboard already renders
// (`t-meas`/`t-tgt`/`t-mock`, footer legend in `web/muser-dashboard.html`).
// `measured` = backed by a real counter on this process (even if currently
// zero for lack of traffic); `target` = a real, citable engineering
// goal/precedent constant, not live-measured by this process; `mock` = a
// placeholder value with no real backing subsystem yet.
//
// (plain `//` here, not `///`: this precedes a macro invocation, not a
// declaration, so rustdoc can't attach a doc comment to it.)
wire_enum!(HonestyTag { Measured => "measured", Target => "target", Mock => "mock" });

// ================================================================ structs

#[derive(Debug, Clone, Serialize)]
pub struct Cluster {
    pub model: String,
    pub quant: String,
    pub weights_bytes: u64,
    pub layers_total: u32,
    pub layers_nope: u32,
    pub layers_swa: u32,
    pub swa_window: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct Node {
    pub id: String,
    pub role: NodeRole,
    pub kind: NodeKind,
    pub state: NodeState,
    pub tok_s: f64,
    pub util: f64,
    pub mem_used: u64,
    pub mem_total: u64,
    pub power_w: f64,
    pub temp_c: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Structural {
    pub nope_growing_bytes: u64,
    pub swa_windowed_bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct PerSessionKv {
    pub session: String,
    pub bytes: u64,
    pub nope_bytes: u64,
    pub swa_bytes: u64,
    pub tokens: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct KvCache {
    pub total_bytes: u64,
    pub tokens_cached: u64,
    pub capacity_bytes: u64,
    pub structural: Structural,
    pub per_session: Vec<PerSessionKv>,
}

/// Wire-shape match for `docs/metrics-schema.json` `#/$defs/Economics`.
/// Field-identical to `muser_kvpack::economics::EconomicsSnapshot` — kept as
/// a distinct type in this module (rather than re-exporting) only so this
/// file is a complete, self-contained reading of the wire contract.
#[derive(Debug, Clone, Serialize)]
pub struct Economics {
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
    /// Reuse that is real work avoided but not a served cache hit: a prompt
    /// continued inside its own live session, and prefill another node
    /// computed. Both are reported beside `cache_hits`, never inside it.
    pub session_continuation_hits: u64,
    pub disagg_prefills: u64,
    pub disagg_bytes_installed: u64,
}

impl From<econ::EconomicsSnapshot> for Economics {
    fn from(e: econ::EconomicsSnapshot) -> Self {
        Economics {
            cache_hits: e.cache_hits,
            cache_misses: e.cache_misses,
            hit_rate: e.hit_rate,
            bytes_served_from_cache: e.bytes_served_from_cache,
            bytes_recomputed: e.bytes_recomputed,
            restore_ops: e.restore_ops,
            prefill_ops: e.prefill_ops,
            restore_speedup: e.restore_speedup,
            derived: e.derived,
            tiers: e.tiers,
            session_continuation_hits: e.session_continuation_hits,
            disagg_prefills: e.disagg_prefills,
            disagg_bytes_installed: e.disagg_bytes_installed,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Transfer {
    pub session: String,
    pub src_node: String,
    pub dst_node: String,
    pub bytes_total: u64,
    pub bytes_sent: u64,
    pub phase: TransferPhase,
    pub throughput_gbps: f64,
    pub active_drain_gbps: f64,
    pub hidden_pct: f64,
    /// The receipt's own phase split (`_`-prefixed extensions, same
    /// convention as `_events`): control-channel round trip and producer
    /// wait, which is where a slow handoff usually spends its time — the
    /// commit itself is already visible as `throughput_gbps`.
    #[serde(rename = "_control_ns")]
    pub control_ns: u64,
    #[serde(rename = "_accept_ns")]
    pub accept_ns: u64,
    #[serde(rename = "_active_drain_ns")]
    pub active_drain_ns: u64,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Percentiles {
    pub p50: f64,
    pub p95: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Wire {
    pub ingress_gbps: f64,
    pub egress_gbps: f64,
    /// Product requests per second over the last [`state::RATE_WINDOW`] —
    /// telemetry polling is excluded, and a lifetime average is not used:
    /// it would keep reporting traffic that stopped minutes ago.
    pub requests_per_s: f64,
    /// Active inference sessions, which is what this gauge has always
    /// counted; open sockets (dashboard viewers included) are reported
    /// separately as `_active_connections`.
    pub connected_clients: u64,
    pub ttft_ms: Percentiles,
    pub itl_ms: Percentiles,
}

#[derive(Debug, Clone, Serialize)]
pub struct TrickContribution {
    pub kind: TrickKind,
    pub value: f64,
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline: Option<String>,
    pub measured: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Trick {
    pub name: TrickName,
    pub active: bool,
    pub measured_contribution: TrickContribution,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct SpecDec {
    pub accept_rate: f64,
    pub accepted_run: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub draft_len: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub per_weight_read_speedup: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cumulative_accepted: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cumulative_drafted: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ane_route_failures: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metal_route_failures: Option<u64>,
}

/// Per-field honesty tags for the `economics` section. `counters` covers
/// the raw monotonic counters (`cache_hits`..`prefill_ops`); `tiers` covers
/// `tiers.*.hits`/`resident_bytes` (real) — `tiers.*.capacity_bytes` is a
/// configured budget, documented as such in `docs/telemetry.md` rather than
/// broken out into its own tag.
#[derive(Debug, Clone, Serialize)]
pub struct EconomicsHonesty {
    pub counters: HonestyTag,
    pub restore_speedup: HonestyTag,
    pub derived: DerivedHonesty,
    pub tiers: HonestyTag,
}

#[derive(Debug, Clone, Serialize)]
pub struct DerivedHonesty {
    pub seconds_saved: HonestyTag,
    pub gflops_avoided: HonestyTag,
    pub joules_saved: HonestyTag,
}

#[derive(Debug, Clone, Serialize)]
pub struct WireHonesty {
    pub connected_clients: HonestyTag,
    pub requests_per_s: HonestyTag,
    pub ingress_gbps: HonestyTag,
    pub egress_gbps: HonestyTag,
    pub ttft_ms: HonestyTag,
    pub itl_ms: HonestyTag,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClusterHonesty {
    /// Layer counts / GQA shape / swa_window / model / quant: fixed,
    /// verified facts from the pinned loaded-model contract, not a live
    /// utilization reading.
    pub arch: HonestyTag,
    /// `measured` only when `--model` was given and the file was actually
    /// stat'd on disk this run; `mock` otherwise.
    pub weights_bytes: HonestyTag,
}

/// Per-section (and, where a section is genuinely mixed, per-field) honesty
/// tags for one [`MetricsSnapshot`]. Carried as the `_honesty` extension —
/// not part of the versioned `docs/metrics-schema.json` contract proper
/// (additive and `_`-prefixed),
/// so a strict schema consumer that doesn't know about it can ignore it.
#[derive(Debug, Clone, Serialize)]
pub struct Honesty {
    pub cluster: ClusterHonesty,
    pub nodes: HonestyTag,
    pub kv: HonestyTag,
    pub economics: EconomicsHonesty,
    pub transfers: HonestyTag,
    pub wire: WireHonesty,
    pub sessions: HonestyTag,
    /// Section-level tag for the tricks *catalogue itself* (i.e. "this list
    /// and its per-trick `measured` flags are accurate"). Per-trick honesty
    /// is already carried by each entry's own
    /// `measured_contribution.measured`.
    pub tricks: HonestyTag,
    pub specdec: HonestyTag,
}

#[derive(Debug, Clone, Serialize)]
pub struct MetricsSnapshot {
    pub schema_version: u32,
    pub generated_at: String,
    pub engine_clock_s: f64,
    pub uptime_s: f64,
    pub cluster: Cluster,
    pub nodes: Vec<Node>,
    pub kv: KvCache,
    pub economics: Economics,
    pub transfers: Vec<Transfer>,
    pub wire: Wire,
    pub sessions: Vec<SessionView>,
    pub tricks: Vec<Trick>,
    pub specdec: SpecDec,
    #[serde(rename = "_events")]
    pub events: Vec<SessionEvent>,
    #[serde(rename = "_honesty")]
    pub honesty: Honesty,
    /// Real, live count of currently-open `GET /telemetry` SSE connections
    /// (see `state::ServerState::telemetry_viewers`, incremented/decremented
    /// by the SSE/WebSocket viewer guards). Not part of `docs/metrics-schema.json`'s
    /// versioned contract — an extra, always-`measured` extension in the
    /// same spirit as `_events`/`_honesty`, included because it's real and
    /// free, not because the dashboard requires it.
    #[serde(rename = "_telemetry_viewers")]
    pub telemetry_viewers: u64,
    /// Telemetry polls, open sockets, and the load-shedding gauges, kept out
    /// of `wire` so the versioned sections keep describing product traffic.
    #[serde(rename = "_telemetry_requests")]
    pub telemetry_requests: u64,
    #[serde(rename = "_active_connections")]
    pub active_connections: u64,
    #[serde(rename = "_queue_depth")]
    pub queue_depth: u64,
    #[serde(rename = "_overload_rejections")]
    pub overload_rejections: u64,
    /// Poisoned-lease recoveries; non-zero means a generation panicked and
    /// the process kept serving.
    #[serde(rename = "_lock_recoveries")]
    pub lock_recoveries: u64,
    #[serde(rename = "_decode")]
    pub decode: Decode,
    #[serde(rename = "_remote")]
    pub remote: RemoteHealth,
    #[serde(rename = "_phases")]
    pub phases: PhaseTelemetry,
    /// See [`DFlashAcceptance`].
    #[serde(rename = "_dflash_acceptance")]
    pub dflash_acceptance: DFlashAcceptance,
}

#[derive(Debug, Clone, Serialize)]
pub struct PhaseMeasurement {
    pub samples: u64,
    pub total_ms: f64,
    pub mean_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct PhaseTelemetry {
    pub queue: PhaseMeasurement,
    pub prefill: PhaseMeasurement,
    pub sampling: PhaseMeasurement,
    pub grammar: PhaseMeasurement,
    pub detokenization: PhaseMeasurement,
    pub enqueue_write: PhaseMeasurement,
    pub dflash_draft: PhaseMeasurement,
    pub dflash_target_verify: PhaseMeasurement,
    pub last_request_decode_tok_s: f64,
}

/// Muser-side DFlash acceptance counters, additive and env-free like the
/// `specdec.cumulative_*` pair: process-monotonic, so an isolated
/// single-request snapshot window (before/after one measured request, the
/// pattern `scripts/representative_dflash_smoke.py` already uses for
/// `_phases.dflash_draft`/`_phases.dflash_target_verify`) recovers that
/// request's own contribution by differencing. Reuses the engine's own
/// `DFlashSpecStats` counters verbatim (`drafted_tokens`,
/// `accepted_draft_tokens`, `rounds`, `speculation_disabled`,
/// `speculation_disabled_at_tokens`) rather than deriving parallel ones, so
/// this telemetry always equals the policy's own view. Carried as the
/// `_dflash_acceptance` extension — not part of the versioned
/// `docs/metrics-schema.json` contract proper (additive and `_`-prefixed),
/// so a strict schema consumer that doesn't know about it can ignore it; see
/// `_telemetry_viewers` above for the precedent.
#[derive(Debug, Clone, Serialize)]
pub struct DFlashAcceptance {
    /// Total drafted/proposed tokens verified against the target.
    pub dflash_drafted_tokens: u64,
    /// Accepted-from-draft tokens, excluding each round's non-draft commit.
    pub dflash_accepted_tokens: u64,
    /// Verify rounds executed.
    pub dflash_rounds: u64,
    /// Requests that FINISHED with speculation disabled; reads as 0/1 once
    /// differenced over one isolated request. Since the gate re-qualifies
    /// (`e3c7464`), a request can close and reopen the gate and still
    /// finish enabled, so this undercounts activity: use
    /// `dflash_disable_events` to see firings.
    pub dflash_disabled: u64,
    /// Sum of the committed-token counts at which the gate closed. A
    /// re-qualifying request can close the gate more than once, so this is
    /// a sum over firings, not a single per-request trip point; divide by
    /// `dflash_disable_events` for a mean.
    pub dflash_disabled_at_tokens: u64,
    /// Gate closures. Counts every firing, including ones the request later
    /// re-qualified out of — the only field that sees a close/reopen cycle.
    pub dflash_disable_events: u64,
    /// Effective draft context geometry of the most recent DFlash request.
    /// Last-value, not additive: a receipt reads the geometry its own run
    /// used, so a wrong window can never again hide behind a clean number.
    pub draft_sink_size: u64,
    pub draft_sliding_window: u64,
}

/// Live completion traffic. `completion_traffic_tok_s_10s` is windowed over
/// [`state::RATE_WINDOW`]; `completion_tokens` is the process total.
#[derive(Debug, Clone, Serialize)]
pub struct Decode {
    pub completion_traffic_tok_s_10s: f64,
    pub completion_tokens: u64,
    /// When the last generation finished, so an idle dashboard can say
    /// "idle since" instead of decaying to a zero indistinguishable from a
    /// server that never served.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_generation_at: Option<String>,
    /// The in-flight request, so the dashboard can show work happening
    /// during prefill instead of appearing dead between token streams.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active: Option<ActivePhase>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ActivePhase {
    pub phase: String,
    pub elapsed_ms: u64,
}

/// Remote-prefill health. A degraded disaggregated route is otherwise
/// indistinguishable from one nobody exercised: both leave `transfers` empty.
#[derive(Debug, Clone, Serialize)]
pub struct RemoteHealth {
    pub receive_failures: u64,
    pub fallbacks: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
}

/// Compatibility-SSE envelope for `GET /telemetry`.
///
/// WebSocket telemetry is protocol v2 and intentionally uses a different
/// hello/snapshot/section-delta shape; see `docs/metrics-schema.md`.
#[derive(Debug, Clone, Serialize)]
pub struct Envelope<T> {
    pub v: u32,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub seq: u64,
    pub t: f64,
    pub data: T,
}

// ================================================================ tables

/// No optimization appears on the live product surface until its independent
/// correctness and performance seal passes in this repository.
pub fn tricks() -> Vec<Trick> {
    Vec::new()
}

/// Hardware topology is unavailable until configured producers and live
/// telemetry exist. Do not manufacture offline nodes from a planned demo.
fn unavailable_nodes() -> Vec<Node> {
    Vec::new()
}

// ============================================================== assembly

/// Assemble a [`MetricsSnapshot`] from live `ServerState`. Pure function of
/// its input: safe to call once per `GET /snapshot` request or once per SSE
/// tick, no hidden mutation.
pub fn build_snapshot(state: &ServerState) -> MetricsSnapshot {
    let uptime_s = state.uptime_s();
    let sessions = state.sessions.list();
    let per_session: Vec<PerSessionKv> = sessions
        .iter()
        .map(|s| PerSessionKv {
            session: s.id.clone(),
            bytes: econ::kv_bytes(s.tokens),
            nope_bytes: econ::nope_bytes(s.tokens),
            swa_bytes: econ::swa_bytes(s.tokens),
            tokens: s.tokens,
        })
        .collect();
    let nope_total: u64 = per_session.iter().map(|p| p.nope_bytes).sum();
    let swa_total: u64 = per_session.iter().map(|p| p.swa_bytes).sum();
    let tokens_cached: u64 = sessions.iter().map(|s| s.tokens).sum();

    let econ_report = state.economics.report();

    let requests_per_s = state.requests_per_s();

    let model_bytes = state.model_bytes();
    // Node labels come from the live receiver configuration; with no remote
    // route configured there is nothing to name and no transfer to label.
    let (src_node, dst_node) = state
        .remote_endpoints()
        .unwrap_or_else(|| ("unconfigured".to_string(), "unconfigured".to_string()));
    let transfers = state
        .remote_transfers
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .iter()
        .map(|sample| Transfer {
            session: sample.transfer_id.clone(),
            src_node: src_node.clone(),
            dst_node: dst_node.clone(),
            bytes_total: sample.installed_bytes,
            bytes_sent: sample.installed_bytes,
            phase: TransferPhase::Done,
            throughput_gbps: if sample.transfer_ns == 0 {
                0.0
            } else {
                sample.installed_bytes as f64 * 8.0 / sample.transfer_ns as f64
            },
            active_drain_gbps: if sample.active_drain_ns == 0 {
                0.0
            } else {
                sample.installed_bytes as f64 * 8.0 / sample.active_drain_ns as f64
            },
            hidden_pct: sample.hidden_fraction,
            control_ns: sample.control_ns,
            accept_ns: sample.accept_ns,
            active_drain_ns: sample.active_drain_ns,
        })
        .collect::<Vec<_>>();
    let ingress_gbps = state.ingress_gbps();
    let ttft = {
        let values = state
            .ttft_ns
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        percentiles_ns(values.iter().copied().collect())
    };
    let itl = percentiles_ns(state.decode_gaps_ns());

    MetricsSnapshot {
        schema_version: SCHEMA_VERSION,
        generated_at: timefmt::now_rfc3339(),
        engine_clock_s: uptime_s,
        uptime_s,
        cluster: Cluster {
            model: "Muse Glimmer-30B".into(),
            quant: "Q4".into(),
            weights_bytes: model_bytes.unwrap_or(DEFAULT_WEIGHTS_BYTES),
            layers_total: 52,
            layers_nope: 13,
            layers_swa: 39,
            swa_window: 2048,
        },
        nodes: unavailable_nodes(),
        kv: KvCache {
            total_bytes: nope_total + swa_total,
            tokens_cached,
            capacity_bytes: KV_CAPACITY_BYTES,
            structural: Structural {
                nope_growing_bytes: nope_total,
                swa_windowed_bytes: swa_total,
            },
            per_session,
        },
        economics: econ_report.wire.into(),
        transfers,
        wire: Wire {
            ingress_gbps,
            egress_gbps: 0.0,
            requests_per_s,
            connected_clients: sessions.len() as u64,
            ttft_ms: ttft,
            itl_ms: itl,
        },
        sessions,
        tricks: tricks(),
        specdec: {
            let drafted = state
                .dflash_drafted
                .load(std::sync::atomic::Ordering::Relaxed);
            let accepted = state
                .dflash_accepted
                .load(std::sync::atomic::Ordering::Relaxed);
            SpecDec {
                accept_rate: if drafted == 0 {
                    0.0
                } else {
                    accepted as f64 / drafted as f64
                },
                accepted_run: state
                    .dflash_last_accepted_run
                    .load(std::sync::atomic::Ordering::Relaxed),
                draft_len: state
                    .inference
                    .as_ref()
                    .is_some_and(|runtime| runtime.dflash.is_some())
                    .then(|| crate::openai::dflash_verify_len() as u64),
                per_weight_read_speedup: None,
                cumulative_accepted: Some(accepted),
                cumulative_drafted: Some(drafted),
                ane_route_failures: Some(
                    state
                        .dflash_ane_failures
                        .load(std::sync::atomic::Ordering::Relaxed),
                ),
                metal_route_failures: Some(
                    state
                        .dflash_metal_failures
                        .load(std::sync::atomic::Ordering::Relaxed),
                ),
            }
        },
        events: state.events.list(),
        telemetry_viewers: state
            .telemetry_viewers
            .load(std::sync::atomic::Ordering::Relaxed),
        telemetry_requests: state
            .telemetry_requests
            .load(std::sync::atomic::Ordering::Relaxed),
        active_connections: state
            .active_connections
            .load(std::sync::atomic::Ordering::Relaxed),
        queue_depth: state.queue_depth.load(std::sync::atomic::Ordering::Relaxed),
        overload_rejections: state
            .overload_rejections
            .load(std::sync::atomic::Ordering::Relaxed),
        lock_recoveries: state
            .lock_recoveries
            .load(std::sync::atomic::Ordering::Relaxed),
        decode: Decode {
            completion_traffic_tok_s_10s: state.decode_tokens_per_s(),
            last_generation_at: state.last_generation(),
            active: state.active_phase().map(|(phase, ms)| ActivePhase {
                phase: phase.to_string(),
                elapsed_ms: ms,
            }),
            completion_tokens: state
                .completion_tokens_total
                .load(std::sync::atomic::Ordering::Relaxed),
        },
        remote: RemoteHealth {
            receive_failures: state
                .remote_receive_failures
                .load(std::sync::atomic::Ordering::Relaxed),
            fallbacks: state
                .remote_fallbacks
                .load(std::sync::atomic::Ordering::Relaxed),
            last_error: state.last_remote_error(),
        },
        phases: PhaseTelemetry {
            queue: phase_measurement(&state.phase_timings.queue),
            prefill: phase_measurement(&state.phase_timings.prefill),
            sampling: phase_measurement(&state.phase_timings.sampling),
            grammar: phase_measurement(&state.phase_timings.grammar),
            detokenization: phase_measurement(&state.phase_timings.detokenization),
            enqueue_write: phase_measurement(&state.phase_timings.enqueue_write),
            dflash_draft: phase_measurement(&state.phase_timings.dflash_draft),
            dflash_target_verify: phase_measurement(&state.phase_timings.dflash_target_verify),
            last_request_decode_tok_s: state
                .last_request_decode_milli_tok_s
                .load(std::sync::atomic::Ordering::Relaxed)
                as f64
                / 1_000.0,
        },
        dflash_acceptance: DFlashAcceptance {
            dflash_drafted_tokens: state
                .dflash_drafted
                .load(std::sync::atomic::Ordering::Relaxed),
            dflash_accepted_tokens: state
                .dflash_accepted
                .load(std::sync::atomic::Ordering::Relaxed),
            dflash_rounds: state
                .dflash_rounds
                .load(std::sync::atomic::Ordering::Relaxed),
            dflash_disabled: state
                .dflash_disabled_requests
                .load(std::sync::atomic::Ordering::Relaxed),
            dflash_disabled_at_tokens: state
                .dflash_disabled_at_tokens
                .load(std::sync::atomic::Ordering::Relaxed),
            dflash_disable_events: state
                .dflash_disable_events
                .load(std::sync::atomic::Ordering::Relaxed),
            draft_sink_size: state
                .dflash_draft_sink_size
                .load(std::sync::atomic::Ordering::Relaxed),
            draft_sliding_window: state
                .dflash_draft_sliding_window
                .load(std::sync::atomic::Ordering::Relaxed),
        },
        honesty: Honesty {
            cluster: ClusterHonesty {
                arch: HonestyTag::Measured,
                weights_bytes: if model_bytes.is_some() {
                    HonestyTag::Measured
                } else {
                    HonestyTag::Mock
                },
            },
            nodes: HonestyTag::Mock,
            kv: HonestyTag::Measured,
            economics: EconomicsHonesty {
                counters: HonestyTag::Measured,
                restore_speedup: if econ_report.restore_speedup_is_measured {
                    HonestyTag::Measured
                } else {
                    HonestyTag::Mock
                },
                derived: DerivedHonesty {
                    seconds_saved: if econ_report.restore_speedup_is_measured {
                        HonestyTag::Measured
                    } else {
                        HonestyTag::Mock
                    },
                    // This is a topology/model-derived linear FLOP floor, not
                    // a hardware counter. The three-value wire vocabulary has
                    // no "derived" tag, so fail closed as unavailable rather
                    // than presenting it as measured.
                    gflops_avoided: HonestyTag::Mock,
                    joules_saved: HonestyTag::Mock,
                },
                tiers: HonestyTag::Measured,
            },
            transfers: HonestyTag::Measured,
            wire: WireHonesty {
                connected_clients: HonestyTag::Measured,
                requests_per_s: HonestyTag::Measured,
                ingress_gbps: HonestyTag::Measured,
                egress_gbps: HonestyTag::Mock,
                ttft_ms: HonestyTag::Measured,
                // Both are now read from live counters this process keeps:
                // inter-token gaps recorded around the decode loop, and a
                // windowed count of completion tokens actually produced.
                itl_ms: HonestyTag::Measured,
            },
            sessions: HonestyTag::Measured,
            tricks: HonestyTag::Measured,
            specdec: HonestyTag::Measured,
        },
    }
}

fn phase_measurement(counter: &crate::state::PhaseCounter) -> PhaseMeasurement {
    let samples = counter.samples.load(std::sync::atomic::Ordering::Relaxed);
    let total_ns = counter.total_ns.load(std::sync::atomic::Ordering::Relaxed);
    PhaseMeasurement {
        samples,
        total_ms: total_ns as f64 / 1_000_000.0,
        mean_ms: if samples == 0 {
            0.0
        } else {
            total_ns as f64 / samples as f64 / 1_000_000.0
        },
    }
}

fn percentiles_ns(mut values: Vec<u64>) -> Percentiles {
    if values.is_empty() {
        return Percentiles { p50: 0.0, p95: 0.0 };
    }
    values.sort_unstable();
    let at = |quantile: f64| {
        let index = ((values.len() - 1) as f64 * quantile).ceil() as usize;
        values[index] as f64 / 1_000_000.0
    };
    Percentiles {
        p50: at(0.50),
        p95: at(0.95),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_state_is_schema_shaped_and_honest() {
        let state = ServerState::new(None);
        let snap = build_snapshot(&state);
        assert_eq!(snap.schema_version, 1);
        assert!(snap.nodes.is_empty());
        assert!(snap.sessions.is_empty());
        assert!(snap.transfers.is_empty());
        assert_eq!(snap.kv.total_bytes, 0);
        assert_eq!(snap.economics.cache_hits, 0);
        assert!(snap.tricks.is_empty());
        assert_eq!(snap.cluster.weights_bytes, 0);
        // No --model means unavailable, represented by zero and tagged mock
        // in schema v1. It is never replaced with a historical estimate.
        assert_eq!(snap.honesty.cluster.weights_bytes, HonestyTag::Mock);
        assert_eq!(snap.honesty.nodes, HonestyTag::Mock);
        assert_eq!(snap.honesty.kv, HonestyTag::Measured);
        assert_eq!(snap.honesty.economics.restore_speedup, HonestyTag::Mock);
        assert_eq!(
            snap.honesty.economics.derived.seconds_saved,
            HonestyTag::Mock
        );
        assert_eq!(
            snap.honesty.economics.derived.gflops_avoided,
            HonestyTag::Mock
        );
    }

    #[test]
    fn a_real_session_shows_up_in_kv_and_wire() {
        let state = ServerState::new(None);
        state
            .sessions
            .create("sx-1", 4096, "m3ultra-0", crate::session::Origin::Resumed);
        let snap = build_snapshot(&state);
        assert_eq!(snap.sessions.len(), 1);
        assert_eq!(snap.wire.connected_clients, 1);
        assert_eq!(snap.kv.tokens_cached, 4096);
        assert_eq!(snap.kv.total_bytes, econ::kv_bytes(4096));
    }

    #[test]
    fn a_real_restore_flips_restore_speedup_to_measured() {
        let state = ServerState::new(None);
        state.economics.record_restore(
            2000,
            econ::Tier::Resident,
            econ::RestoreBytes::Estimated,
            Some(econ::RestoreTiming {
                restore_seconds: 0.02,
                local_prefill_seconds: 1.0,
            }),
        );
        let snap = build_snapshot(&state);
        assert_eq!(snap.honesty.economics.restore_speedup, HonestyTag::Measured);
        assert_eq!(
            snap.honesty.economics.derived.seconds_saved,
            HonestyTag::Measured
        );
        assert_eq!(
            snap.honesty.economics.derived.gflops_avoided,
            HonestyTag::Mock
        );
        assert!(snap.economics.restore_speedup > 1.0);
    }

    #[test]
    fn snapshot_serializes_to_json() {
        let state = ServerState::new(None);
        let snap = build_snapshot(&state);
        let json = serde_json::to_string(&snap).expect("MetricsSnapshot must serialize");
        assert!(json.contains("\"schema_version\":1"));
        assert!(json.contains("\"_honesty\""));
        assert!(json.contains("\"tricks\":[]"));
        // The acceptance extension must carry every counter a receipt
        // differences, including gate firings a request re-qualified out of.
        for field in [
            "\"_dflash_acceptance\"",
            "\"dflash_drafted_tokens\"",
            "\"dflash_accepted_tokens\"",
            "\"dflash_rounds\"",
            "\"dflash_disabled\"",
            "\"dflash_disabled_at_tokens\"",
            "\"dflash_disable_events\"",
            "\"draft_sink_size\"",
            "\"draft_sliding_window\"",
        ] {
            assert!(json.contains(field), "snapshot is missing {field}");
        }
    }

    #[test]
    fn gate_disable_events_survive_a_requalifying_request() {
        let state = ServerState::new(None);
        // A request that closed the gate twice and finished with it open:
        // `dflash_disabled` cannot see it, `dflash_disable_events` must.
        let stats = muser_engine::dflash::DFlashSpecStats {
            disable_events: 2,
            speculation_disabled: false,
            ..Default::default()
        };
        state.record_dflash_stats(&stats);
        let snap = build_snapshot(&state);
        assert_eq!(snap.dflash_acceptance.dflash_disabled, 0);
        assert_eq!(snap.dflash_acceptance.dflash_disable_events, 2);
    }

    #[test]
    fn snapshot_stamps_the_draft_geometry_a_request_ran_with() {
        let state = ServerState::new(None);
        let stats = muser_engine::dflash::DFlashSpecStats {
            draft_sink_size: 64,
            draft_sliding_window: 2_048,
            ..Default::default()
        };
        state.record_dflash_stats(&stats);
        let snap = build_snapshot(&state);
        assert_eq!(snap.dflash_acceptance.draft_sink_size, 64);
        assert_eq!(snap.dflash_acceptance.draft_sliding_window, 2_048);
    }

    #[test]
    fn ttft_window_reports_measured_percentiles() {
        let state = ServerState::new(None);
        for value in [1_000_000, 2_000_000, 9_000_000] {
            state.record_ttft(value);
        }
        let snap = build_snapshot(&state);
        assert_eq!(snap.wire.ttft_ms.p50, 2.0);
        assert_eq!(snap.wire.ttft_ms.p95, 9.0);
        assert_eq!(snap.honesty.wire.ttft_ms, HonestyTag::Measured);
    }

    #[test]
    fn committed_remote_receipt_populates_transfer_and_ingress_metrics() {
        let state = ServerState::new(None);
        state.record_remote_transfer(&muser_cluster::receiver::RemoteReceiveReceipt {
            transfer_id: "transfer-9".into(),
            generation: 9,
            installed_segments: 4,
            installed_bytes: 125_000_000,
            control_ns: 1,
            accept_ns: 1,
            transfer_commit_ns: 1_000_000_000,
            total_ns: 1_000_000_002,
            producer: None,
            components: Default::default(),
            phases: muser_cluster::phase::HandoffPhaseNanos {
                segment_read_ns: 250_000_000,
                ..Default::default()
            },
        });
        let snap = build_snapshot(&state);
        assert_eq!(snap.transfers.len(), 1);
        assert_eq!(snap.transfers[0].session, "transfer-9");
        assert_eq!(snap.transfers[0].throughput_gbps, 1.0);
        assert_eq!(snap.transfers[0].active_drain_gbps, 4.0);
        assert_eq!(snap.transfers[0].active_drain_ns, 250_000_000);
        // The per-transfer rate is the wire rate during the commit; wire
        // ingress is the same bytes averaged over the rolling window.
        assert_eq!(
            snap.wire.ingress_gbps,
            125_000_000.0 * 8.0 / crate::state::RATE_WINDOW.as_nanos() as f64
        );
        assert_eq!(snap.transfers[0].hidden_pct, 0.0);
        assert_eq!(snap.transfers[0].control_ns, 1);
        assert_eq!(snap.transfers[0].accept_ns, 1);
        // Without a configured receiver there is no node pair to name.
        assert_eq!(snap.transfers[0].src_node, "unconfigured");
        assert_eq!(snap.honesty.transfers, HonestyTag::Measured);
        assert_eq!(snap.honesty.wire.ingress_gbps, HonestyTag::Measured);
    }

    #[test]
    fn decode_telemetry_is_measured_from_live_counters() {
        let state = ServerState::new(None);
        for gap in [1_000_000, 2_000_000, 9_000_000] {
            state.record_decode_gap(gap);
        }
        state.record_decode_tokens(30);
        let snap = build_snapshot(&state);
        assert_eq!(snap.wire.itl_ms.p50, 2.0);
        assert_eq!(snap.wire.itl_ms.p95, 9.0);
        assert_eq!(snap.honesty.wire.itl_ms, HonestyTag::Measured);
        assert_eq!(snap.decode.completion_tokens, 30);
        assert_eq!(
            snap.decode.completion_traffic_tok_s_10s,
            30.0 / crate::state::RATE_WINDOW.as_secs_f64()
        );
    }

    #[test]
    fn telemetry_polls_stay_out_of_the_request_rate() {
        let state = ServerState::new(None);
        state.record_request();
        state
            .telemetry_requests
            .fetch_add(9, std::sync::atomic::Ordering::Relaxed);
        let snap = build_snapshot(&state);
        assert_eq!(
            snap.wire.requests_per_s,
            1.0 / crate::state::RATE_WINDOW.as_secs_f64()
        );
        assert_eq!(snap.telemetry_requests, 9);
    }

    #[test]
    fn remote_failures_are_visible_without_a_committed_transfer() {
        let state = ServerState::new(None);
        state.record_remote_failure("producer refused the handshake");
        state.record_remote_fallback();
        let snap = build_snapshot(&state);
        assert!(snap.transfers.is_empty());
        assert_eq!(snap.remote.receive_failures, 1);
        assert_eq!(snap.remote.fallbacks, 1);
        assert_eq!(
            snap.remote.last_error.as_deref(),
            Some("producer refused the handshake")
        );
    }
}
