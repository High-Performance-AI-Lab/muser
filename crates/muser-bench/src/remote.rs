#![recursion_limit = "256"]

//! Live GX10 -> Metal disaggregated-prefill qualification.
//!
//! This executable deliberately uses the same `RemoteReceiver` as serving.
//! Every sample performs cold local recomputation and an authenticated remote
//! install, compares 256 greedy tokens plus every full target-logit row, and
//! retains producer/transport phase times needed to prove real overlap.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant};

use base64::Engine as _;
use kvpack_handoff::MultimodalIdentityV2;
use muser_cluster::config::ReceiverConfigV2;
use muser_cluster::control::{PrefillControlSegmentV2, ProducerPhaseReceiptV1};
use muser_cluster::receiver::{RemoteReceiveReceipt, RemoteReceiver};
use muser_engine::dflash::{DFlashAssistant, DFlashContextSnapshot, DFlashSpecStats};
use muser_engine::vision::VisionModel;
use muser_engine::{
    DecodeInput, EmbeddingSegment, Model, ModelConfig, PrefillBatch, PrefillSegment, Session,
    EMBEDDING_POSITION_WITNESS,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const VISION_PREPROCESSING_CONTRACT: &[u8] =
    b"muse-glimmer-vision-v1:lanczos3:max-image-tokens-4096:rgb-normalized:pixel-shuffle-2";
const LINK_GBPS_MINIMUM: f64 = 3.0;
const DFLASH_ACCEPTANCE_MINIMUM: f64 = 0.95;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Variant {
    Text,
    Multimodal,
    TargetPlusDflash,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum ProducerReceiptProfile {
    Enrolled,
    HistoricalPreStreamingControl,
}

struct Args {
    model: PathBuf,
    prompt_fixture: PathBuf,
    cluster_config: PathBuf,
    variant: Variant,
    dflash: Option<PathBuf>,
    mmproj: Option<PathBuf>,
    mtmd_bridge: Option<PathBuf>,
    image: Option<PathBuf>,
    repetitions: usize,
    output_tokens: usize,
    verify_length: usize,
    identity: String,
    poc: bool,
    p4: bool,
    diagnostic: bool,
    onboarding_native: bool,
    drift_graded: bool,
    reference_once: bool,
    performance_only: bool,
    fast_consumer_math: bool,
    producer_receipt_profile: ProducerReceiptProfile,
    external_producer_receipt: Option<PathBuf>,
    external_producer_receipt_dir: Option<PathBuf>,
    delta_prefix_cut: Option<u64>,
    prefix_prompt_fixture: Option<PathBuf>,
    local_only: bool,
    dry_run: bool,
}

#[derive(Debug, Deserialize)]
struct ExternalProducerClientReceipt {
    schema: String,
    token_count: usize,
    response: ExternalProducerResponse,
}

#[derive(Debug, Deserialize)]
struct ExternalProducerResponse {
    status: String,
    request_id: String,
    prompt_token_count: usize,
    producer_receipt: ExternalProducerReceipt,
}

#[derive(Debug, Deserialize)]
struct ExternalProducerReceipt {
    schema: String,
    producer_mode: Option<String>,
    prompt_token_count: usize,
    token_ids_sha256: String,
    phase_ns: ExternalProducerPhaseNanos,
    handoff: ExternalHandoffReceipt,
}

#[derive(Debug, Deserialize)]
struct ExternalProducerPhaseNanos {
    connector_total: u64,
    d2h_complete_offset: u64,
    /// Present on the streaming (v2) producer: offset of the first segment
    /// send, which must precede D2H completion (the streaming overlap).
    first_segment_sent_offset: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct ExternalHandoffReceipt {
    ack: bool,
    transfer_id: String,
    generation: u64,
    segments: u32,
    payload_bytes: u64,
    payload_wire_ns: u64,
    payload_wire_source: String,
    payload_pacing_bps: u64,
    transfer_start_unix_ns: u64,
    first_segment_sent_unix_ns: u64,
    transfer_acked_unix_ns: u64,
}

#[derive(Clone)]
struct PreparedPrompt {
    batch: PrefillBatch,
    cached_batch: PrefillBatch,
    witnesses: Vec<u32>,
    multimodal: Option<(MultimodalIdentityV2, Vec<PrefillControlSegmentV2>)>,
}

#[derive(Debug)]
struct Generation {
    tokens: Vec<u32>,
    full_logit_digest: String,
    retained_logits: Option<Vec<f32>>,
    /// Prefix cache immediately before the held prompt-boundary token is
    /// decoded. This is retained only for explicitly requested diagnostics.
    cache_snapshot: Option<muser_engine::cache::SessionCacheSnapshot>,
    /// Cache after the final logits row has been produced. The last sampled
    /// token is intentionally not decoded, so output row N's causative KV row
    /// is still addressable without changing generation semantics.
    post_decode_cache_snapshot: Option<muser_engine::cache::SessionCacheSnapshot>,
    first_64_decode_ns: u64,
}

struct DFlashRun {
    tokens: Vec<u32>,
    stats: DFlashSpecStats,
    context: DFlashContextSnapshot,
    receipt: Option<RemoteReceiveReceipt>,
    decode_ns: u64,
}

#[derive(Debug)]
struct DFlashContextDiff {
    layer: usize,
    plane: &'static str,
    index: usize,
    token: usize,
    element: usize,
    local_bits: u32,
    remote_bits: u32,
}

#[derive(Serialize)]
struct PlaneDiff {
    layer: u32,
    logical_start: u64,
    logical_count: u64,
    key_mismatched: usize,
    value_mismatched: usize,
    key_max_abs: f32,
    value_max_abs: f32,
    key_mean_abs: f64,
    value_mean_abs: f64,
    key_mismatched_by_token: Vec<usize>,
    value_mismatched_by_token: Vec<usize>,
    first_key_mismatch: Option<CellDiff>,
    first_value_mismatch: Option<CellDiff>,
}

#[derive(Serialize)]
struct CellDiff {
    index: usize,
    token: usize,
    element: usize,
    local_bits: u16,
    remote_bits: u16,
    local: f32,
    remote: f32,
}

#[derive(Serialize)]
struct Sample<'a> {
    schema: &'static str,
    kind: &'static str,
    identity: &'a str,
    variant: Variant,
    repetition: usize,
    order: [&'static str; 2],
    prompt_positions: usize,
    output_tokens: usize,
    local_ttft_ns: u64,
    remote_ttft_ns: u64,
    ttft_speedup: f64,
    local_first_64_decode_ns: u64,
    remote_first_64_decode_ns: u64,
    remote_decode_ratio: f64,
    installed_bytes: u64,
    installed_segments: u32,
    target_installed_bytes: u64,
    target_installed_segments: u32,
    dflash_installed_bytes: u64,
    dflash_installed_segments: u32,
    target_prepared: bool,
    target_installed: bool,
    dflash_prepared: bool,
    dflash_installed: bool,
    receiver_control_ns: u64,
    receiver_accept_ns: u64,
    receiver_transfer_commit_ns: u64,
    producer_payload_wire_ns: u64,
    /// Installed payload divided by producer kernel TCP busy time.
    producer_payload_gbps: f64,
    installed_payload_gbps: f64,
    /// N-series phase split: receiver socket-drain time for segment frames.
    receiver_segment_drain_ns: Option<u64>,
    /// Segment verify time (vendored HMAC/hash checks), install excluded.
    receiver_verify_ns: Option<u64>,
    /// Sink install time inside the transfer loop.
    receiver_install_ns: Option<u64>,
    /// Seal frame read + seal HMAC verify + sink prepare_commit.
    receiver_seal_ns: Option<u64>,
    /// Atomic engine commit.
    receiver_commit_ns: Option<u64>,
    /// installed_bytes*8 / receiver_segment_drain_ns.
    wire_gbps: Option<f64>,
    /// installed_bytes*8 / (install + seal + commit).
    install_gbps: Option<f64>,
    receiver_segment_phases: Option<Vec<muser_cluster::phase::SegmentPhaseNanos>>,
    producer_export_overhead_ratio: Option<f64>,
    producer_first_tile_prefill_fraction: Option<f64>,
    producer_transfer_hidden_ratio: Option<f64>,
    producer_payload_bytes: Option<u64>,
    generated_tokens_sha256: &'a str,
    full_logit_digest: &'a str,
    exact_tokens: bool,
    token_agreement_rate: f64,
    divergent_tokens: usize,
    first_divergent_token: Option<usize>,
    exact_full_logits: bool,
    remote_local_logit_max_abs: Option<f32>,
    remote_local_logit_mean_abs: Option<f64>,
    local_dflash_acceptance: Option<f64>,
    remote_dflash_acceptance: Option<f64>,
    remote_dflash_acceptance_ratio: Option<f64>,
    exact_dflash_tokens: Option<bool>,
    exact_dflash_trace: Option<bool>,
    dflash_draft_trace_sha256: Option<String>,
    dflash_context_sha256: Option<String>,
    dflash_accepted_prefix_trace_sha256: Option<String>,
    dflash_accepted_prefix_counts: Option<Vec<usize>>,
}

#[derive(Serialize)]
struct FastPerformanceSample<'a> {
    schema: &'static str,
    kind: &'static str,
    identity: &'a str,
    repetition: usize,
    /// Repetition 0 is the preregistered warmup handoff (owner ruling
    /// 2026-08-19, plan §7.3): receipt-validated and bound into the
    /// determinism canonical, but excluded from the stability gates because
    /// the first handoff after producer readiness pays a known one-time
    /// CUDA/allocator warmup (~8%).
    warmup: bool,
    prompt_positions: usize,
    output_tokens: usize,
    remote_ttft_ns: u64,
    remote_first_64_decode_ns: u64,
    installed_bytes: u64,
    installed_segments: u32,
    producer_receipt_profile: ProducerReceiptProfile,
    producer_payload_wire_ns: u64,
    installed_payload_gbps: f64,
    generated_tokens_sha256: String,
    full_logit_digest: &'a str,
    deterministic_against_first: bool,
    receiver_segment_drain_ns: u64,
    receiver_verify_ns: u64,
    receiver_install_ns: u64,
    receiver_seal_ns: u64,
    receiver_commit_ns: u64,
    receiver_seal_read_offset_ns: u64,
    receiver_seal_read_unix_ns: u64,
    receiver_segment_read_offsets_ns: Vec<u64>,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("muser-remote-qualify: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    validate_args(&args)?;
    let diagnostic_mode = args.poc || args.diagnostic;
    let first_divergence_diagnostic = std::env::var_os("MUSER_REMOTE_FIRST_DIVERGENCE").is_some();
    let retain_comparison = diagnostic_mode || args.drift_graded || first_divergence_diagnostic;
    if args.dry_run {
        println!(
            "{}",
            serde_json::json!({
                "schema": "muser.remote-qualify.v1",
                "kind": "dry-run",
                "accelerator_touched": false,
                "model": args.model,
                "prompt_fixture": args.prompt_fixture,
                "cluster_config": args.cluster_config,
                "variant": args.variant,
                "dflash": args.dflash,
                "mmproj": args.mmproj,
                "mtmd_bridge": args.mtmd_bridge,
                "image": args.image,
                "repetitions": args.repetitions,
                "output_tokens": args.output_tokens,
                "verify_length": args.verify_length,
                "identity": args.identity,
                "poc": args.poc,
                "p4": args.p4,
                "diagnostic": args.diagnostic,
                "onboarding_native": args.onboarding_native,
                "drift_graded": args.drift_graded,
                "reference_once": args.reference_once,
                "performance_only": args.performance_only,
                "consumer_math_route": if args.fast_consumer_math {
                    "native-fast"
                } else {
                    "cross-vendor-exact"
                },
                "producer_receipt_profile": args.producer_receipt_profile,
                "external_producer_receipt": args.external_producer_receipt,
                "external_producer_receipt_dir": args.external_producer_receipt_dir,
                "local_only": args.local_only,
                "correctness": if args.performance_only {
                    "fast-output-determinism; exact-reference comparison is separate evidence"
                } else if args.drift_graded {
                    "distributional-token-logit-and-kv-plane-drift"
                } else {
                    "exact-256-tokens-and-all-step-full-target-logit-digest"
                },
                "retain_mismatch_diagnostics": retain_comparison,
                "first_divergence_diagnostic": first_divergence_diagnostic,
                "measurement": "cold-local-versus-live-mtls-hmac-atomic-remote-install",
                "seal_eligible": false,
            })
        );
        return Ok(());
    }
    if std::env::var("MUSER_ACCELERATOR_LEASE").as_deref() != Ok("1") {
        return Err("execution must be a child of accelerator_safe.py".into());
    }

    let prompt_bytes = std::fs::read(&args.prompt_fixture)
        .map_err(|error| format!("cannot read {}: {error}", args.prompt_fixture.display()))?;
    let prompt_tokens = parse_tokens(&prompt_bytes)?;
    if prompt_tokens.len() < 3 {
        return Err("remote qualification prompt requires at least three tokens".into());
    }
    phase("target-model-load:start");
    let model = Model::load(ModelConfig::new(&args.model)).map_err(|error| error.to_string())?;
    phase("target-model-load:done");
    if prompt_tokens
        .iter()
        .any(|token| *token as usize >= model.config().vocab_size)
    {
        return Err("prompt fixture contains an out-of-vocabulary token".into());
    }
    let prepared = prepare_prompt(&args, &model, &prompt_tokens)?;
    let max_context = prepared
        .witnesses
        .len()
        .checked_add(args.output_tokens)
        .ok_or("context length overflow")?;
    if max_context > model.config().context_length {
        return Err(format!(
            "prompt plus output is {max_context}, model limit is {}",
            model.config().context_length
        ));
    }
    if args.local_only {
        let first = run_local_target(&model, &prepared, max_context, args.output_tokens, true)?.0;
        let second = run_local_target(&model, &prepared, max_context, args.output_tokens, true)?.0;
        if first.tokens != second.tokens {
            return Err(format!(
                "local target tokens differ at output row {}",
                first_mismatch(&first.tokens, &second.tokens)
            ));
        }
        if first.full_logit_digest != second.full_logit_digest {
            let difference = retained_logit_diff(&first, &second)?;
            return Err(format!(
                "local target full-logit digest is nondeterministic (max_abs={:e}, mean_abs={:e}, mismatched={}, first_row={}, first_logit={})",
                difference.maximum,
                difference.mean,
                difference.mismatched,
                difference.first_row,
                difference.first_logit,
            ));
        }
        let installed =
            run_local_tile_install_target(&model, &prepared, max_context, args.output_tokens)?;
        if first.tokens != installed.tokens
            || first.full_logit_digest != installed.full_logit_digest
        {
            let difference = retained_logit_diff(&first, &installed)?;
            return Err(format!(
                "local remote-tile install does not replay target logits (token_row={}, max_abs={:e}, mean_abs={:e}, mismatched={}, first_row={}, first_logit={})",
                first_mismatch(&first.tokens, &installed.tokens),
                difference.maximum,
                difference.mean,
                difference.mismatched,
                difference.first_row,
                difference.first_logit,
            ));
        }
        println!(
            "{}",
            serde_json::json!({
                "schema": "muser.remote-qualify.v1",
                "kind": "local-diagnostic",
                "identity": args.identity,
                "prompt_positions": prepared.witnesses.len(),
                "output_tokens": args.output_tokens,
                "exact_tokens": true,
                "exact_full_logits": true,
                "exact_remote_tile_install": true,
                "full_logit_digest": first.full_logit_digest,
                "seal_eligible": false,
            })
        );
        return Ok(());
    }
    let mut config = ReceiverConfigV2::load(&args.cluster_config)?;
    if args.variant != Variant::TargetPlusDflash {
        config.dflash_identity_sha256 = None;
        config.dflash_context_geometry = None;
    }
    let receiver = RemoteReceiver::bind(config)?;
    if let Some(cut) = args.delta_prefix_cut {
        return run_delta_probe(&args, &model, &prepared, max_context, &receiver, cut);
    }
    if args.performance_only {
        let mut assistant = match args.dflash.as_deref() {
            Some(path) => {
                phase("dflash-model-load:start");
                let assistant = load_dflash(path, &model)?;
                phase("dflash-model-load:done");
                Some(assistant)
            }
            None => None,
        };
        return run_fast_performance_only(
            &args,
            &model,
            &prepared,
            max_context,
            &receiver,
            &prompt_bytes,
            assistant.as_mut(),
        );
    }
    let mut assistant = match args.dflash.as_deref() {
        Some(path) => {
            phase("dflash-model-load:start");
            let assistant = load_dflash(path, &model)?;
            phase("dflash-model-load:done");
            Some(assistant)
        }
        None => None,
    };

    let mut local_ttfts = Vec::with_capacity(args.repetitions);
    let mut remote_ttfts = Vec::with_capacity(args.repetitions);
    let mut link_gbps_samples = Vec::with_capacity(args.repetitions);
    let mut wire_gbps_samples = Vec::with_capacity(args.repetitions);
    let mut install_gbps_samples = Vec::with_capacity(args.repetitions);
    let mut remote_dflash_acceptance_samples = Vec::with_capacity(args.repetitions);
    let mut canonical_tokens = None;
    let mut canonical_logits = None;
    let mut canonical_remote_tokens = None;
    let mut canonical_remote_logits = None;
    let mut comparison_exact = true;
    let mut token_agreement_samples = Vec::<f64>::new();
    let mut logit_max_abs_samples = Vec::<f32>::new();
    let mut logit_mean_abs_samples = Vec::<f64>::new();
    // Fast-lane P4 cells compare five native runs with one immutable exact
    // anchor. Recomputing that 275-second validation anchor cannot improve
    // either the native timing statistics or the drift measurement.
    let local_reference = if args.reference_once {
        Some(run_local_target(
            &model,
            &prepared,
            max_context,
            args.output_tokens,
            retain_comparison,
        )?)
    } else {
        None
    };
    for repetition in 0..args.repetitions {
        // Release qualification retains ABBA ordering. POC diagnostics start
        // remote-first so the transferred-cache comparator can report before
        // a redundant full local reference run; POC output is never sealable.
        // Frozen paired order repeats ABBA across the first member of each
        // pair: local, remote, remote, local. Three-sample onboarding uses
        // the first three entries; release benchmark packets use all five.
        let remote_first =
            args.reference_once || diagnostic_mode || matches!(repetition % 4, 1 | 2);
        let mut local_owned = None;
        let (remote, remote_ttft, receipt) = if args.reference_once {
            let remote = run_remote_target(
                &receiver,
                assistant.as_mut(),
                &model,
                &prepared,
                max_context,
                args.output_tokens,
                diagnostic_mode,
                retain_comparison,
            )?;
            (remote.0, remote.1, remote.2)
        } else if remote_first {
            let remote = run_remote_target(
                &receiver,
                assistant.as_mut(),
                &model,
                &prepared,
                max_context,
                args.output_tokens,
                diagnostic_mode,
                retain_comparison,
            )?;
            let local = run_local_target(
                &model,
                &prepared,
                max_context,
                args.output_tokens,
                retain_comparison,
            )?;
            local_owned = Some(local);
            (remote.0, remote.1, remote.2)
        } else {
            let local = run_local_target(
                &model,
                &prepared,
                max_context,
                args.output_tokens,
                retain_comparison,
            )?;
            local_owned = Some(local);
            let remote = run_remote_target(
                &receiver,
                assistant.as_mut(),
                &model,
                &prepared,
                max_context,
                args.output_tokens,
                diagnostic_mode,
                retain_comparison,
            )?;
            (remote.0, remote.1, remote.2)
        };
        let (local, local_ttft) = local_reference
            .as_ref()
            .or(local_owned.as_ref())
            .map(|(generation, ttft)| (generation, *ttft))
            .ok_or("local reference is unavailable")?;
        if std::env::var_os("MUSER_REMOTE_CACHE_DIFF").is_some() {
            emit_cache_diff(
                local
                    .cache_snapshot
                    .as_ref()
                    .ok_or("local cache snapshot was not retained")?,
                remote
                    .cache_snapshot
                    .as_ref()
                    .ok_or("remote cache snapshot was not retained")?,
            )?;
        }
        let divergent_tokens = local
            .tokens
            .iter()
            .zip(&remote.tokens)
            .filter(|(local, remote)| local != remote)
            .count()
            + local.tokens.len().abs_diff(remote.tokens.len());
        let first_divergent_token =
            (local.tokens != remote.tokens).then(|| first_mismatch(&local.tokens, &remote.tokens));
        let compared_tokens = local.tokens.len().max(remote.tokens.len());
        let token_agreement_rate = if compared_tokens == 0 {
            1.0
        } else {
            1.0 - divergent_tokens as f64 / compared_tokens as f64
        };
        let exact_tokens = divergent_tokens == 0;
        if !exact_tokens && !args.drift_graded {
            let index = first_divergent_token.expect("non-exact token streams");
            if first_divergence_diagnostic {
                emit_first_divergence_diagnostic(local, &remote, index, prepared.witnesses.len())?;
            }
            let detail = if diagnostic_mode && !args.drift_graded {
                let difference = retained_logit_diff(local, &remote)?;
                format!(
                    ", local_token={:?}, remote_token={:?}, local_logits={}, remote_logits={}, max_abs={:e}, mean_abs={:e}, mismatched={}, first_logit={}",
                    local.tokens.get(index),
                    remote.tokens.get(index),
                    local.full_logit_digest,
                    remote.full_logit_digest,
                    difference.maximum,
                    difference.mean,
                    difference.mismatched,
                    difference.first_logit,
                )
            } else {
                String::new()
            };
            return Err(format!(
                "remote target tokens differ at output row {index}{detail}"
            ));
        }
        let exact_full_logits = local.full_logit_digest == remote.full_logit_digest;
        let (logit_max_abs, logit_mean_abs) = if exact_full_logits {
            (Some(0.0), Some(0.0))
        } else {
            if !retain_comparison {
                return Err("remote target full-logit digest differs from local".into());
            }
            let difference = retained_logit_diff(local, &remote)?;
            if first_divergence_diagnostic {
                emit_full_logit_divergence_diagnostic(
                    local,
                    &remote,
                    &difference,
                    prepared.witnesses.len(),
                )?;
            }
            let detail = if diagnostic_mode && !args.drift_graded {
                let local_repeat =
                    run_local_target(&model, &prepared, max_context, args.output_tokens, true)?.0;
                let repeat_exact = local.full_logit_digest == local_repeat.full_logit_digest;
                let repeat = if repeat_exact {
                    RetainedLogitDiff {
                        maximum: 0.0,
                        mean: 0.0,
                        mismatched: 0,
                        first_row: 0,
                        first_logit: 0,
                    }
                } else {
                    retained_logit_diff(local, &local_repeat)?
                };
                format!(
                    " (max_abs={:e}, mean_abs={:e}, mismatched={}, first_row={}, first_logit={}, local_repeat_exact={}, local_repeat_max_abs={:e}, local_repeat_mismatched={}, local_repeat_first_row={}, local_repeat_first_logit={})",
                    difference.maximum,
                    difference.mean,
                    difference.mismatched,
                    difference.first_row,
                    difference.first_logit,
                    repeat_exact,
                    repeat.maximum,
                    repeat.mismatched,
                    repeat.first_row,
                    repeat.first_logit,
                )
            } else {
                String::new()
            };
            if !args.drift_graded {
                return Err(format!(
                    "remote target full-logit digest differs from local{detail}"
                ));
            }
            (Some(difference.maximum), Some(difference.mean))
        };
        comparison_exact &= exact_tokens && exact_full_logits;
        token_agreement_samples.push(token_agreement_rate);
        logit_max_abs_samples.push(logit_max_abs.expect("comparison maximum"));
        logit_mean_abs_samples.push(logit_mean_abs.expect("comparison mean"));
        if let Some(tokens) = canonical_tokens.as_ref() {
            if tokens != &local.tokens {
                return Err("generated tokens changed between cold repetitions".into());
            }
        } else {
            canonical_tokens = Some(local.tokens.clone());
            canonical_logits = Some(local.full_logit_digest.clone());
        }
        if let Some(tokens) = canonical_remote_tokens.as_ref() {
            if tokens != &remote.tokens
                || canonical_remote_logits.as_ref() != Some(&remote.full_logit_digest)
            {
                return Err("fast-lane output changed between cold repetitions".into());
            }
        } else {
            canonical_remote_tokens = Some(remote.tokens.clone());
            canonical_remote_logits = Some(remote.full_logit_digest.clone());
        }

        let (local_dflash, remote_dflash, exact_dflash) =
            if let Some(assistant) = assistant.as_mut() {
                let local_dflash = run_local_dflash(
                    assistant,
                    &model,
                    &prepared,
                    max_context,
                    args.output_tokens,
                    args.verify_length,
                )?;
                let remote_dflash = run_remote_dflash(
                    &receiver,
                    assistant,
                    &model,
                    &prepared,
                    max_context,
                    args.output_tokens,
                    args.verify_length,
                    diagnostic_mode,
                )?;
                let exact = local_dflash.tokens == remote_dflash.tokens
                    && local_dflash.tokens == local.tokens;
                if !exact && !args.drift_graded {
                    return Err("remote/local/target DFlash token mismatch".into());
                }
                (Some(local_dflash), Some(remote_dflash), Some(exact))
            } else {
                (None, None, None)
            };

        let token_hash = token_digest(&local.tokens);
        let (
            exact_dflash_trace,
            draft_trace_sha256,
            accepted_trace_sha256,
            accepted_counts,
            context_sha256,
        ) = match (local_dflash.as_ref(), remote_dflash.as_ref()) {
            (Some(local_run), Some(remote_run)) => {
                let local_stats = &local_run.stats;
                let remote_stats = &remote_run.stats;
                validate_dflash_trace(local_stats)?;
                validate_dflash_trace(remote_stats)?;
                // The installed draft context is gated on raw bits, not on
                // trace equality: a low-mantissa divergence can leave the
                // proposal trace intact on one cell while corrupting
                // another. Compare always.
                let local_context_sha = dflash_context_digest(&local_run.context);
                let remote_context_sha = dflash_context_digest(&remote_run.context);
                let context_difference =
                    first_dflash_context_diff(&local_run.context, &remote_run.context)?;
                if let Some(difference) = context_difference.as_ref().filter(|_| !args.drift_graded)
                {
                    return Err(format!(
                            "remote DFlash assistant context differs: layer={} plane={} index={} token={} element={} local_bits={:#010x} remote_bits={:#010x} local_sha={} remote_sha={}",
                            difference.layer,
                            difference.plane,
                            difference.index,
                            difference.token,
                            difference.element,
                            difference.local_bits,
                            difference.remote_bits,
                            local_context_sha,
                            remote_context_sha,
                        ));
                }
                let trace_exact = local_stats.draft_token_trace == remote_stats.draft_token_trace
                    && local_stats.accepted_prefix_trace == remote_stats.accepted_prefix_trace
                    && local_stats.accepted_prefix_counts == remote_stats.accepted_prefix_counts;
                if !args.drift_graded
                    && (local_stats.draft_token_trace != remote_stats.draft_token_trace
                        || local_stats.accepted_prefix_trace != remote_stats.accepted_prefix_trace
                        || local_stats.accepted_prefix_counts
                            != remote_stats.accepted_prefix_counts)
                {
                    let draft_index = first_mismatch(
                        &local_stats.draft_token_trace,
                        &remote_stats.draft_token_trace,
                    );
                    let accepted_index = first_mismatch(
                        &local_stats.accepted_prefix_trace,
                        &remote_stats.accepted_prefix_trace,
                    );
                    let count_index = local_stats
                        .accepted_prefix_counts
                        .iter()
                        .zip(&remote_stats.accepted_prefix_counts)
                        .position(|(local, remote)| local != remote)
                        .unwrap_or(
                            local_stats
                                .accepted_prefix_counts
                                .len()
                                .min(remote_stats.accepted_prefix_counts.len()),
                        );
                    let local_context_sha = dflash_context_digest(&local_run.context);
                    let remote_context_sha = dflash_context_digest(&remote_run.context);
                    let context_diff =
                        first_dflash_context_diff(&local_run.context, &remote_run.context)?;
                    let context_diff = context_diff.map(|difference| {
                            format!(
                                "layer={} plane={} index={} token={} element={} local_bits={:#010x} remote_bits={:#010x}",
                                difference.layer,
                                difference.plane,
                                difference.index,
                                difference.token,
                                difference.element,
                                difference.local_bits,
                                difference.remote_bits,
                            )
                        });
                    let first_link_gbps = receipt
                        .producer
                        .as_ref()
                        .filter(|producer| producer.payload_wire_ns != 0)
                        .map(|producer| {
                            receipt.installed_bytes as f64 * 8.0 / producer.payload_wire_ns as f64
                        });
                    let second_receipt = remote_run
                        .receipt
                        .as_ref()
                        .ok_or("remote DFlash run lacks its transfer receipt")?;
                    let second_link_gbps = second_receipt
                        .producer
                        .as_ref()
                        .filter(|producer| producer.payload_wire_ns != 0)
                        .map(|producer| {
                            second_receipt.installed_bytes as f64 * 8.0
                                / producer.payload_wire_ns as f64
                        });
                    return Err(format!(
                            "remote DFlash assistant trace differs: draft local={} remote={} first_index={} local_token={:?} remote_token={:?}; accepted local={} remote={} first_index={} local_token={:?} remote_token={:?}; counts first_round={} local={:?} remote={:?}; acceptance local={}/{}={:.9} remote={}/{}={:.9}; context local={} remote={} first_diff={:?}; target_transfer_link_gbps={:?}; dflash_transfer_link_gbps={:?}",
                            token_digest(&local_stats.draft_token_trace),
                            token_digest(&remote_stats.draft_token_trace),
                            draft_index,
                            local_stats.draft_token_trace.get(draft_index),
                            remote_stats.draft_token_trace.get(draft_index),
                            token_digest(&local_stats.accepted_prefix_trace),
                            token_digest(&remote_stats.accepted_prefix_trace),
                            accepted_index,
                            local_stats.accepted_prefix_trace.get(accepted_index),
                            remote_stats.accepted_prefix_trace.get(accepted_index),
                            count_index,
                            local_stats.accepted_prefix_counts,
                            remote_stats.accepted_prefix_counts,
                            local_stats.accepted_draft_tokens,
                            local_stats.drafted_tokens,
                            local_stats.acceptance_rate(),
                            remote_stats.accepted_draft_tokens,
                            remote_stats.drafted_tokens,
                            remote_stats.acceptance_rate(),
                            local_context_sha,
                            remote_context_sha,
                            context_diff,
                            first_link_gbps,
                            second_link_gbps,
                        ));
                }
                (
                    Some(trace_exact && context_difference.is_none()),
                    Some(token_digest(&local_stats.draft_token_trace)),
                    Some(token_digest(&local_stats.accepted_prefix_trace)),
                    Some(local_stats.accepted_prefix_counts.clone()),
                    Some(local_context_sha),
                )
            }
            (None, None) => (None, None, None, None, None),
            _ => return Err("DFlash trace evidence is incomplete".into()),
        };
        let component_segments = receipt
            .components
            .target_segments
            .checked_add(receipt.components.dflash_segments)
            .ok_or("installed component segment count overflow")?;
        let component_bytes = receipt
            .components
            .target_bytes
            .checked_add(receipt.components.dflash_bytes)
            .ok_or("installed component byte count overflow")?;
        if !receipt.components.target_prepared
            || !receipt.components.target_installed
            || receipt.components.target_segments == 0
            || receipt.components.target_bytes == 0
            || component_segments != receipt.installed_segments
            || component_bytes != receipt.installed_bytes
        {
            return Err("remote receipt does not prove the complete target install".into());
        }
        let combined = args.variant == Variant::TargetPlusDflash;
        if combined
            != (receipt.components.dflash_prepared
                && receipt.components.dflash_installed
                && receipt.components.dflash_segments > 0
                && receipt.components.dflash_bytes > 0)
        {
            return Err("remote receipt does not prove the requested DFlash install".into());
        }
        if !combined
            && (receipt.components.dflash_segments != 0 || receipt.components.dflash_bytes != 0)
        {
            return Err("non-DFlash transfer installed unexpected DFlash payload".into());
        }
        let external_producer_path = args.external_producer_receipt.clone().or_else(|| {
            args.external_producer_receipt_dir
                .as_ref()
                .map(|directory| directory.join(format!("{}-client.json", receipt.transfer_id)))
        });
        let external_producer = external_producer_path
            .as_deref()
            .map(|path| {
                load_external_producer_phase(
                    path,
                    &receipt,
                    &prepared.witnesses,
                    args.producer_receipt_profile,
                )
            })
            .transpose()?;
        let producer = receipt
            .producer
            .as_ref()
            .or(external_producer.as_ref())
            .ok_or("remote receipt lacks producer phase evidence")?;
        if producer.payload_bytes != receipt.installed_bytes {
            return Err("producer payload bytes do not bind the installed transfer".into());
        }
        let phases = phase_metrics(producer)?;
        if receipt.transfer_commit_ns == 0 {
            return Err("remote receipt contains zero transfer time".into());
        }
        let producer_payload_gbps =
            receipt.installed_bytes as f64 * 8.0 / producer.payload_wire_ns as f64;
        // N-series phase split: the receiver side of the same transfer.
        let phase = &receipt.phases;
        let (drain_ns, verify_ns, install_ns, seal_ns, commit_ns) = (
            phase.segment_read_ns,
            phase.verify_ns(),
            phase.sink_install_ns,
            phase.seal_ns,
            phase.commit_ns,
        );
        let wire_gbps =
            (drain_ns != 0).then(|| receipt.installed_bytes as f64 * 8.0 / drain_ns as f64);
        let installed_payload_gbps = producer_payload_gbps;
        let install_span = install_ns.saturating_add(seal_ns).saturating_add(commit_ns);
        let install_gbps =
            (install_span != 0).then(|| receipt.installed_bytes as f64 * 8.0 / install_span as f64);
        let (export_ratio, first_tile_fraction, hidden_ratio) =
            (Some(phases.0), Some(phases.1), Some(phases.2));
        let local_acceptance = local_dflash.as_ref().map(|run| run.stats.acceptance_rate());
        let remote_acceptance = remote_dflash
            .as_ref()
            .map(|run| run.stats.acceptance_rate());
        let acceptance_ratio = local_acceptance
            .zip(remote_acceptance)
            .map(|(local, remote)| {
                if local == 0.0 {
                    if remote == 0.0 {
                        1.0
                    } else {
                        f64::INFINITY
                    }
                } else {
                    remote / local
                }
            });
        let order = if args.reference_once {
            ["reference", "remote"]
        } else if remote_first {
            ["remote", "local"]
        } else {
            ["local", "remote"]
        };
        println!(
            "{}",
            serde_json::to_string(&Sample {
                schema: "muser.remote-qualify.v1",
                kind: "sample",
                identity: &args.identity,
                variant: args.variant,
                repetition,
                order,
                prompt_positions: prepared.witnesses.len(),
                output_tokens: args.output_tokens,
                local_ttft_ns: local_ttft,
                remote_ttft_ns: remote_ttft,
                ttft_speedup: local_ttft as f64 / remote_ttft as f64,
                local_first_64_decode_ns: local.first_64_decode_ns,
                remote_first_64_decode_ns: remote.first_64_decode_ns,
                remote_decode_ratio: remote.first_64_decode_ns as f64
                    / local.first_64_decode_ns as f64,
                installed_bytes: receipt.installed_bytes,
                installed_segments: receipt.installed_segments,
                target_installed_bytes: receipt.components.target_bytes,
                target_installed_segments: receipt.components.target_segments,
                dflash_installed_bytes: receipt.components.dflash_bytes,
                dflash_installed_segments: receipt.components.dflash_segments,
                target_prepared: receipt.components.target_prepared,
                target_installed: receipt.components.target_installed,
                dflash_prepared: receipt.components.dflash_prepared,
                dflash_installed: receipt.components.dflash_installed,
                receiver_control_ns: receipt.control_ns,
                receiver_accept_ns: receipt.accept_ns,
                receiver_transfer_commit_ns: receipt.transfer_commit_ns,
                producer_payload_wire_ns: producer.payload_wire_ns,
                producer_payload_gbps,
                installed_payload_gbps,
                receiver_segment_drain_ns: Some(drain_ns),
                receiver_verify_ns: Some(verify_ns),
                receiver_install_ns: Some(install_ns),
                receiver_seal_ns: Some(seal_ns),
                receiver_commit_ns: Some(commit_ns),
                wire_gbps,
                install_gbps,
                receiver_segment_phases: Some(phase.segments.clone()),
                producer_export_overhead_ratio: export_ratio,
                producer_first_tile_prefill_fraction: first_tile_fraction,
                producer_transfer_hidden_ratio: hidden_ratio,
                producer_payload_bytes: Some(producer.payload_bytes),
                generated_tokens_sha256: &token_hash,
                full_logit_digest: &local.full_logit_digest,
                exact_tokens,
                token_agreement_rate,
                divergent_tokens,
                first_divergent_token,
                exact_full_logits,
                remote_local_logit_max_abs: logit_max_abs,
                remote_local_logit_mean_abs: logit_mean_abs,
                local_dflash_acceptance: local_acceptance,
                remote_dflash_acceptance: remote_acceptance,
                remote_dflash_acceptance_ratio: acceptance_ratio,
                exact_dflash_tokens: exact_dflash,
                exact_dflash_trace,
                dflash_draft_trace_sha256: draft_trace_sha256,
                dflash_context_sha256: context_sha256,
                dflash_accepted_prefix_trace_sha256: accepted_trace_sha256,
                dflash_accepted_prefix_counts: accepted_counts,
            })
            .map_err(|error| error.to_string())?
        );
        local_ttfts.push(local_ttft);
        remote_ttfts.push(remote_ttft);
        link_gbps_samples.push(installed_payload_gbps);
        if let Some(value) = wire_gbps {
            wire_gbps_samples.push(value);
        }
        if let Some(value) = install_gbps {
            install_gbps_samples.push(value);
        }
        if let Some(value) = remote_acceptance {
            remote_dflash_acceptance_samples.push(value);
        }
    }

    let local_cv = cv(&local_ttfts);
    let remote_cv = cv(&remote_ttfts);
    let link_cv = cv_f64(&link_gbps_samples);
    let stability_limit = if args.p4 { 0.02 } else { 0.03 };
    let mut sorted_link = link_gbps_samples.clone();
    sorted_link.sort_by(f64::total_cmp);
    let link_median = sorted_link[sorted_link.len() / 2];
    // Owner ruling 2026-08-19 (plan §7.4): the link leg is a per-repetition
    // achieved-rate floor, not a rate CV. The CV leg was calibrated in the
    // pinned-flat pacing regime, where the producer busy-time metric measured
    // the pacer (CV 0.4%); above the pin it measures benign physical jitter
    // (CV 5-10% at 5.5-7.3 Gbps) while end-to-end stability is already gated
    // by the TTFT CV legs. The rate CV stays in the receipt for audit.
    let link_floor_ok = link_gbps_samples
        .iter()
        .all(|value| *value >= LINK_GBPS_MINIMUM);
    let link_minimum = link_gbps_samples
        .iter()
        .copied()
        .reduce(f64::min)
        .ok_or("missing link samples")?;
    let dflash_acceptance_minimum = remote_dflash_acceptance_samples
        .iter()
        .copied()
        .reduce(f64::min);
    let speedups = local_ttfts
        .iter()
        .zip(&remote_ttfts)
        .map(|(&local, &remote)| local as f64 / remote as f64)
        .collect::<Vec<_>>();
    println!(
        "{}",
        serde_json::json!({
            "schema": "muser.remote-qualify.v1",
            "kind": "summary",
            "identity": args.identity,
            "poc": args.poc,
            "p4": args.p4,
            "diagnostic": args.diagnostic,
            "onboarding_native": args.onboarding_native,
            "drift_graded": args.drift_graded,
            "reference_once": args.reference_once,
            "performance_only": args.performance_only,
            "variant": args.variant,
            "prompt_positions": prepared.witnesses.len(),
            "prompt_file_sha256": format!("{:x}", Sha256::digest(prompt_bytes)),
            "output_tokens": args.output_tokens,
            "local_ttft_raw_ns": local_ttfts,
            "remote_ttft_raw_ns": remote_ttfts,
            "local_ttft_cv": local_cv,
            "remote_ttft_cv": remote_cv,
            "installed_payload_gbps": link_gbps_samples,
            "installed_payload_gbps_median": link_median,
            "wire_gbps_samples": wire_gbps_samples,
            "wire_gbps_median": median_f64(&wire_gbps_samples),
            "install_gbps_samples": install_gbps_samples,
            "install_gbps_median": median_f64(&install_gbps_samples),
            "installed_payload_gbps_cv": link_cv,
            "installed_payload_gbps_min": link_minimum,
            "stability_cv_maximum": stability_limit,
            "installed_payload_gbps_minimum": LINK_GBPS_MINIMUM,
            "remote_dflash_acceptance": remote_dflash_acceptance_samples,
            "remote_dflash_acceptance_minimum": dflash_acceptance_minimum,
            "remote_dflash_acceptance_required": DFLASH_ACCEPTANCE_MINIMUM,
            "ttft_speedup_mean": speedups.iter().sum::<f64>() / speedups.len() as f64,
            "generated_tokens_sha256": token_digest(canonical_tokens.as_deref().expect("sample")),
            "fast_generated_tokens_sha256": token_digest(
                canonical_remote_tokens.as_deref().expect("fast sample")
            ),
            "full_logit_digest": canonical_logits.expect("sample"),
            "fast_full_logit_digest": canonical_remote_logits.expect("fast sample"),
            "exact_remote_local": comparison_exact,
            "exact_tokens": token_agreement_samples.iter().all(|value| *value == 1.0),
            "token_agreement_rate": token_agreement_samples,
            "logit_max_abs": logit_max_abs_samples,
            "logit_mean_abs": logit_mean_abs_samples,
            "stable": local_cv <= stability_limit && remote_cv <= stability_limit
                && link_floor_ok
                && (args.drift_graded || args.variant != Variant::TargetPlusDflash
                    || dflash_acceptance_minimum.is_some_and(|value| {
                        value >= DFLASH_ACCEPTANCE_MINIMUM
                    })),
            "seal_eligible": false,
            "reason": "cell evidence requires complete 4-depth x 3-variant packet evaluation",
        })
    );
    Ok(())
}

fn run_fast_performance_only(
    args: &Args,
    model: &Model,
    prepared: &PreparedPrompt,
    max_context: usize,
    receiver: &RemoteReceiver,
    prompt_bytes: &[u8],
    assistant: Option<&mut DFlashAssistant>,
) -> Result<(), String> {
    if args.variant == Variant::TargetPlusDflash {
        return run_fast_dflash_performance_only(
            args,
            model,
            prepared,
            max_context,
            receiver,
            prompt_bytes,
            assistant.ok_or("DFlash performance cell has no assistant")?,
        );
    }
    if assistant.is_some() {
        return Err("target-only performance cell received a DFlash assistant".into());
    }
    let mut remote_ttfts = Vec::with_capacity(args.repetitions);
    let mut link_gbps_samples = Vec::with_capacity(args.repetitions);
    let mut canonical_tokens: Option<Vec<u32>> = None;
    let mut canonical_logits: Option<String> = None;
    let mut warmup_ttft: Option<u64> = None;
    let schedule = fast_performance_schedule(args.p4, args.repetitions);
    for (repetition, warmup) in schedule {
        let (remote, remote_ttft, receipt) = run_remote_target(
            receiver,
            None,
            model,
            prepared,
            max_context,
            args.output_tokens,
            false,
            false,
        )?;
        let deterministic = match (&canonical_tokens, &canonical_logits) {
            (Some(tokens), Some(logits)) => {
                tokens == &remote.tokens && logits == &remote.full_logit_digest
            }
            (None, None) => true,
            _ => return Err("fast performance determinism state is incomplete".into()),
        };
        if !deterministic {
            return Err("fast-lane output changed between performance repetitions".into());
        }
        if canonical_tokens.is_none() {
            canonical_tokens = Some(remote.tokens.clone());
            canonical_logits = Some(remote.full_logit_digest.clone());
        }
        if !receipt.components.target_prepared
            || !receipt.components.target_installed
            || receipt.components.target_segments == 0
            || receipt.components.target_bytes == 0
            || receipt.components.dflash_segments != 0
            || receipt.components.dflash_bytes != 0
        {
            return Err("fast performance receipt does not prove a target-only install".into());
        }
        let producer_path = args
            .external_producer_receipt_dir
            .as_ref()
            .ok_or("fast performance requires an external producer receipt directory")?
            .join(format!("{}-client.json", receipt.transfer_id));
        let external = load_external_producer_phase(
            &producer_path,
            &receipt,
            &prepared.witnesses,
            args.producer_receipt_profile,
        )?;
        if external.payload_bytes != receipt.installed_bytes || external.payload_wire_ns == 0 {
            return Err("fast performance producer receipt does not bind the transfer".into());
        }
        phase_metrics(&external)?;
        let payload_gbps = receipt.installed_bytes as f64 * 8.0 / external.payload_wire_ns as f64;
        println!(
            "{}",
            serde_json::to_string(&FastPerformanceSample {
                schema: "muser.remote-qualify.v1",
                kind: "fast-performance-sample",
                identity: &args.identity,
                repetition,
                warmup,
                prompt_positions: prepared.witnesses.len(),
                output_tokens: args.output_tokens,
                remote_ttft_ns: remote_ttft,
                remote_first_64_decode_ns: remote.first_64_decode_ns,
                installed_bytes: receipt.installed_bytes,
                installed_segments: receipt.installed_segments,
                producer_receipt_profile: args.producer_receipt_profile,
                producer_payload_wire_ns: external.payload_wire_ns,
                installed_payload_gbps: payload_gbps,
                generated_tokens_sha256: token_digest(&remote.tokens),
                full_logit_digest: &remote.full_logit_digest,
                deterministic_against_first: deterministic,
                receiver_segment_drain_ns: receipt.phases.segment_read_ns,
                receiver_verify_ns: receipt.phases.verify_ns(),
                receiver_install_ns: receipt.phases.sink_install_ns,
                receiver_seal_ns: receipt.phases.seal_ns,
                receiver_commit_ns: receipt.phases.commit_ns,
                receiver_seal_read_offset_ns: receipt.phases.seal_read_offset_ns,
                receiver_seal_read_unix_ns: receipt.phases.seal_read_unix_ns,
                receiver_segment_read_offsets_ns: receipt
                    .phases
                    .segments
                    .iter()
                    .map(|segment| segment.read_started_offset_ns)
                    .collect(),
            })
            .map_err(|error| error.to_string())?
        );
        if warmup {
            warmup_ttft = Some(remote_ttft);
        } else {
            remote_ttfts.push(remote_ttft);
            link_gbps_samples.push(payload_gbps);
        }
    }
    let remote_cv = cv(&remote_ttfts);
    let link_cv = cv_f64(&link_gbps_samples);
    let link_median = median_f64(&link_gbps_samples).ok_or("missing link samples")?;
    // Owner ruling 2026-08-19 (plan §7.4): per-repetition achieved-rate floor,
    // not a rate CV — see the full-qualification summary for the rationale.
    let link_minimum = link_gbps_samples
        .iter()
        .copied()
        .reduce(f64::min)
        .ok_or("missing link samples")?;
    let mut sorted_ttft = remote_ttfts.clone();
    sorted_ttft.sort_unstable();
    let ttft_median = sorted_ttft[sorted_ttft.len() / 2];
    let ttft_target_applicable = prepared.witnesses.len() <= 2048;
    let stable = fast_performance_stable(
        prepared.witnesses.len(),
        remote_cv,
        link_minimum,
        ttft_median,
    );
    println!(
        "{}",
        serde_json::json!({
            "schema": "muser.remote-qualify.v1",
            "kind": "fast-performance-summary",
            "identity": args.identity,
            "performance_only": true,
            "reference_comparison": null,
            "prompt_positions": prepared.witnesses.len(),
            "prompt_file_sha256": format!("{:x}", Sha256::digest(prompt_bytes)),
            "output_tokens": args.output_tokens,
            "remote_ttft_raw_ns": remote_ttfts,
            "remote_ttft_warmup_ns": warmup_ttft,
            "warmup_repetitions": usize::from(args.p4),
            "remote_ttft_median_ns": ttft_median,
            "remote_ttft_cv": remote_cv,
            "remote_ttft_target_ns": 4_000_000_000_u64,
            "remote_ttft_target_applicable": ttft_target_applicable,
            "installed_payload_gbps": link_gbps_samples,
            "installed_payload_gbps_median": link_median,
            "installed_payload_gbps_cv": link_cv,
            "installed_payload_gbps_min": link_minimum,
            "installed_payload_gbps_minimum": LINK_GBPS_MINIMUM,
            "producer_receipt_profile": args.producer_receipt_profile,
            "fast_generated_tokens_sha256": token_digest(
                canonical_tokens.as_deref().expect("fast sample")
            ),
            "fast_full_logit_digest": canonical_logits.expect("fast sample"),
            "deterministic": true,
            "stable": stable,
            "seal_eligible": false,
            "reason": "performance-only packet; drift is bound by a separate exact-reference cell",
        })
    );
    Ok(())
}

fn fast_performance_schedule(p4: bool, repetitions: usize) -> Vec<(usize, bool)> {
    // A P4 merit packet has one preregistered warmup before its counted
    // repetitions. A
    // diagnostic performance cell is intentionally one measured handoff so
    // operational soak callers can retain state after every ordinal.
    let warmups = usize::from(p4);
    (0..repetitions + warmups)
        .map(|repetition| (repetition, repetition < warmups))
        .collect()
}

fn fast_performance_stable(
    prompt_positions: usize,
    remote_cv: f64,
    link_minimum: f64,
    ttft_median: u64,
) -> bool {
    remote_cv <= 0.02
        && link_minimum >= LINK_GBPS_MINIMUM
        && (prompt_positions > 2048 || ttft_median <= 4_000_000_000)
}

#[allow(clippy::too_many_arguments)]
fn run_fast_dflash_performance_only(
    args: &Args,
    model: &Model,
    prepared: &PreparedPrompt,
    max_context: usize,
    receiver: &RemoteReceiver,
    prompt_bytes: &[u8],
    assistant: &mut DFlashAssistant,
) -> Result<(), String> {
    let mut decode_tps_samples = Vec::with_capacity(args.repetitions);
    let mut acceptance_samples = Vec::with_capacity(args.repetitions);
    let mut link_gbps_samples = Vec::with_capacity(args.repetitions);
    let mut canonical_tokens = None;
    let mut canonical_trace = None;
    let mut canonical_counts = None;
    for repetition in 0..args.repetitions {
        let run = run_remote_dflash(
            receiver,
            assistant,
            model,
            prepared,
            max_context,
            args.output_tokens,
            args.verify_length,
            false,
        )?;
        validate_dflash_trace(&run.stats)?;
        let deterministic = match (&canonical_tokens, &canonical_trace, &canonical_counts) {
            (Some(tokens), Some(trace), Some(counts)) => {
                tokens == &run.tokens
                    && trace == &run.stats.draft_token_trace
                    && counts == &run.stats.accepted_prefix_counts
            }
            (None, None, None) => true,
            _ => return Err("fast DFlash determinism state is incomplete".into()),
        };
        if !deterministic {
            return Err("fast-lane DFlash output or proposal trace changed".into());
        }
        if canonical_tokens.is_none() {
            canonical_tokens = Some(run.tokens.clone());
            canonical_trace = Some(run.stats.draft_token_trace.clone());
            canonical_counts = Some(run.stats.accepted_prefix_counts.clone());
        }
        let receipt = run
            .receipt
            .as_ref()
            .ok_or("fast DFlash run lacks its transfer receipt")?;
        let component_segments = receipt
            .components
            .target_segments
            .checked_add(receipt.components.dflash_segments)
            .ok_or("installed component segment count overflow")?;
        let component_bytes = receipt
            .components
            .target_bytes
            .checked_add(receipt.components.dflash_bytes)
            .ok_or("installed component byte count overflow")?;
        if !receipt.components.target_prepared
            || !receipt.components.target_installed
            || !receipt.components.dflash_prepared
            || !receipt.components.dflash_installed
            || receipt.components.target_segments == 0
            || receipt.components.dflash_segments == 0
            || component_segments != receipt.installed_segments
            || component_bytes != receipt.installed_bytes
        {
            return Err("fast DFlash receipt does not prove the complete paired install".into());
        }
        let producer_path = args
            .external_producer_receipt_dir
            .as_ref()
            .ok_or("fast DFlash performance requires producer receipts")?
            .join(format!("{}-client.json", receipt.transfer_id));
        let external = load_external_producer_phase(
            &producer_path,
            receipt,
            &prepared.witnesses,
            args.producer_receipt_profile,
        )?;
        if external.payload_bytes != receipt.installed_bytes || external.payload_wire_ns == 0 {
            return Err("fast DFlash producer receipt does not bind the transfer".into());
        }
        phase_metrics(&external)?;
        if run.decode_ns == 0 {
            return Err("fast DFlash decode duration is zero".into());
        }
        let decode_tps = args.output_tokens as f64 * 1_000_000_000.0 / run.decode_ns as f64;
        let acceptance = run.stats.acceptance_rate();
        let payload_gbps = receipt.installed_bytes as f64 * 8.0 / external.payload_wire_ns as f64;
        println!(
            "{}",
            serde_json::json!({
                "schema": "muser.remote-qualify.v1",
                "kind": "fast-dflash-performance-sample",
                "identity": args.identity,
                "repetition": repetition,
                "prompt_positions": prepared.witnesses.len(),
                "output_tokens": args.output_tokens,
                "verify_length": args.verify_length,
                "spec_decode_ns": run.decode_ns,
                "spec_decode_tokens_per_second": decode_tps,
                "acceptance_rate": acceptance,
                "accepted_draft_tokens": run.stats.accepted_draft_tokens,
                "drafted_tokens": run.stats.drafted_tokens,
                "rounds": run.stats.rounds,
                "target_batches": run.stats.target_batches,
                "draft_ns": run.stats.draft_ns,
                "target_verify_ns": run.stats.target_verify_ns,
                "fallback_target_ns": run.stats.fallback_target_ns,
                "mirror_capture_fc_ns": run.stats.mirror_capture_fc_ns,
                "cycle_trace": run.stats.cycle_trace,
                "installed_bytes": receipt.installed_bytes,
                "installed_segments": receipt.installed_segments,
                "producer_payload_wire_ns": external.payload_wire_ns,
                "installed_payload_gbps": payload_gbps,
                "generated_tokens_sha256": token_digest(&run.tokens),
                "draft_trace_sha256": token_digest(&run.stats.draft_token_trace),
                "deterministic_against_first": deterministic,
                "consumer_math_route": if args.fast_consumer_math {
                    "native-fast"
                } else {
                    "cross-vendor-exact"
                },
            })
        );
        decode_tps_samples.push(decode_tps);
        acceptance_samples.push(acceptance);
        link_gbps_samples.push(payload_gbps);
    }
    let decode_tps_median =
        median_f64(&decode_tps_samples).ok_or("missing DFlash speed samples")?;
    let acceptance_minimum = acceptance_samples
        .iter()
        .copied()
        .reduce(f64::min)
        .ok_or("missing DFlash acceptance samples")?;
    let link_median = median_f64(&link_gbps_samples).ok_or("missing DFlash link samples")?;
    let decode_tps_cv = cv_f64(&decode_tps_samples);
    let link_cv = cv_f64(&link_gbps_samples);
    // Owner ruling 2026-08-19 (plan §7.4): per-repetition achieved-rate floor,
    // not a rate CV — see the full-qualification summary for the rationale.
    let link_minimum = link_gbps_samples
        .iter()
        .copied()
        .reduce(f64::min)
        .ok_or("missing DFlash link samples")?;
    println!(
        "{}",
        serde_json::json!({
            "schema": "muser.remote-qualify.v1",
            "kind": "fast-dflash-performance-summary",
            "identity": args.identity,
            "performance_only": true,
            "consumer_math_route": if args.fast_consumer_math {
                "native-fast"
            } else {
                "cross-vendor-exact"
            },
            "qualification": if args.p4 { "p4" } else { "diagnostic" },
            "reference_comparison": null,
            "prompt_positions": prepared.witnesses.len(),
            "prompt_file_sha256": format!("{:x}", Sha256::digest(prompt_bytes)),
            "output_tokens": args.output_tokens,
            "verify_length": args.verify_length,
            "spec_decode_tokens_per_second": decode_tps_samples,
            "spec_decode_tokens_per_second_median": decode_tps_median,
            "spec_decode_tokens_per_second_cv": decode_tps_cv,
            "acceptance_rate": acceptance_samples,
            "acceptance_rate_minimum": acceptance_minimum,
            "acceptance_rate_required": DFLASH_ACCEPTANCE_MINIMUM,
            "installed_payload_gbps": link_gbps_samples,
            "installed_payload_gbps_median": link_median,
            "installed_payload_gbps_cv": link_cv,
            "installed_payload_gbps_min": link_minimum,
            "installed_payload_gbps_minimum": LINK_GBPS_MINIMUM,
            "generated_tokens_sha256": token_digest(
                canonical_tokens.as_deref().expect("DFlash sample")
            ),
            "draft_trace_sha256": token_digest(
                canonical_trace.as_deref().expect("DFlash trace")
            ),
            "deterministic": true,
            "stable": args.p4
                && decode_tps_cv <= 0.02
                && link_minimum >= LINK_GBPS_MINIMUM
                && acceptance_minimum >= DFLASH_ACCEPTANCE_MINIMUM,
            "seal_eligible": false,
            "reason": "speculative product-speed packet; drift is bound by separate exact-reference evidence",
        })
    );
    Ok(())
}

fn run_local_tile_install_target(
    model: &Model,
    prepared: &PreparedPrompt,
    max_context: usize,
    output_tokens: usize,
) -> Result<Generation, String> {
    if prepared.witnesses.contains(&EMBEDDING_POSITION_WITNESS) {
        return Err("local remote-tile diagnostic only supports token prompts".into());
    }
    let mut source = new_metal_session(model, max_context)?;
    source
        .prefill(prepared.cached_batch.clone())
        .map_err(|error| error.to_string())?;
    let snapshot = source
        .export_cache_snapshot()
        .map_err(|error| error.to_string())?;
    let mut restored = new_metal_session(model, max_context)?;
    let mut install = restored
        .begin_remote_kv_install(Arc::clone(&snapshot.tokens))
        .map_err(|error| error.to_string())?;
    for plane in snapshot.layers.iter() {
        install
            .write_f16_tile(
                plane.layer as usize,
                true,
                plane.logical_start,
                plane.logical_count,
                &plane.key,
            )
            .map_err(|error| error.to_string())?;
        install
            .write_f16_tile(
                plane.layer as usize,
                false,
                plane.logical_start,
                plane.logical_count,
                &plane.value,
            )
            .map_err(|error| error.to_string())?;
    }
    let prepared_install = restored
        .prepare_remote_kv_install(install)
        .map_err(|error| error.to_string())?;
    restored.commit_prepared_remote_kv_install(prepared_install);
    let boundary = *prepared.witnesses.last().expect("validated");
    let logits = restored
        .decode(DecodeInput { token_id: boundary })
        .map_err(|error| error.to_string())?
        .logits;
    generate_target(model, &mut restored, logits, output_tokens, true)
}

fn run_local_target(
    model: &Model,
    prepared: &PreparedPrompt,
    max_context: usize,
    output_tokens: usize,
    retain_logits: bool,
) -> Result<(Generation, u64), String> {
    phase("local-target-prefill:start");
    let mut session = new_metal_session(model, max_context)?;
    let started = Instant::now();
    session
        .prefill(prepared.cached_batch.clone())
        .map_err(|error| error.to_string())?;
    let cache_snapshot = if retain_logits
        && (std::env::var_os("MUSER_REMOTE_CACHE_DIFF").is_some()
            || std::env::var_os("MUSER_REMOTE_FIRST_DIVERGENCE").is_some())
    {
        Some(
            session
                .export_cache_snapshot()
                .map_err(|error| error.to_string())?,
        )
    } else {
        None
    };
    let boundary = *prepared.witnesses.last().expect("validated");
    if boundary == EMBEDDING_POSITION_WITNESS {
        return Err("local prompt boundary is not a token".into());
    }
    let logits = session
        .decode(DecodeInput { token_id: boundary })
        .map_err(|error| error.to_string())?
        .logits;
    let ttft = nanos(started.elapsed().as_nanos());
    let mut generation =
        generate_target(model, &mut session, logits, output_tokens, retain_logits)?;
    generation.cache_snapshot = cache_snapshot;
    let result = (generation, ttft);
    phase("local-target-prefill-decode:done");
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn run_remote_target(
    receiver: &RemoteReceiver,
    dflash: Option<&mut DFlashAssistant>,
    model: &Model,
    prepared: &PreparedPrompt,
    max_context: usize,
    output_tokens: usize,
    wait_without_control: bool,
    retain_logits: bool,
) -> Result<(Generation, u64, RemoteReceiveReceipt), String> {
    // N2 serial condition: hold a short quiesce so any residue from the
    // previous rep's local lane (allocator pressure, deferred GPU teardown)
    // settles before the receiver starts draining the next handoff.
    if std::env::var_os("MUSER_REMOTE_QUALIFY_SERIAL").is_some() {
        phase("remote-target-prefill-export-receive:serial-quiesce");
        std::thread::sleep(std::time::Duration::from_secs(5));
    }
    phase("remote-target-prefill-export-receive:start");
    let session_started = std::time::Instant::now();
    let mut session = new_metal_session(model, max_context)?;
    phase(&format!(
        "remote-target-prefill-export-receive:session-create:{}ns",
        session_started.elapsed().as_nanos()
    ));
    let receipt = receiver.receive(
        &mut session,
        dflash,
        &prepared.witnesses,
        prepared.multimodal.clone(),
        max_context,
        wait_without_control,
    )?;
    phase("remote-target-prefill-export-receive:done");
    let boundary = *prepared.witnesses.last().expect("validated");
    if boundary == EMBEDDING_POSITION_WITNESS {
        return Err("remote prompt boundary is not a token".into());
    }
    let cache_snapshot = if retain_logits
        && (std::env::var_os("MUSER_REMOTE_CACHE_DIFF").is_some()
            || std::env::var_os("MUSER_REMOTE_FIRST_DIVERGENCE").is_some())
        && !prepared.witnesses.contains(&EMBEDDING_POSITION_WITNESS)
    {
        Some(
            session
                .export_cache_snapshot()
                .map_err(|error| error.to_string())?,
        )
    } else {
        None
    };
    let decode_started = Instant::now();
    let logits = session
        .decode(DecodeInput { token_id: boundary })
        .map_err(|error| error.to_string())?
        .logits;
    let ttft = nanos(
        (receipt.control_ns as u128)
            .saturating_add(receipt.accept_ns as u128)
            .saturating_add(receipt.transfer_commit_ns as u128)
            .saturating_add(decode_started.elapsed().as_nanos()),
    );
    let mut generation =
        generate_target(model, &mut session, logits, output_tokens, retain_logits)?;
    generation.cache_snapshot = cache_snapshot;
    let result = (generation, ttft, receipt);
    phase("remote-target-decode:done");
    Ok(result)
}

/// Delta-handoff evidence cell: install a producer-produced prefix with a
/// full handoff, receive the same longer prompt's suffix through a prefix-cut
/// handoff onto the held prefix, and prove the output is identical to a full
/// handoff of the same prompt while the suffix handoff moved fewer bytes.
///
/// The producer holds back the last prompt token of every request (the
/// receiver decodes it locally as the boundary), so the prefix request asks
/// for `cut + 1` tokens: the handoff then covers all `cut` prefix positions
/// and every installed position in both the delta and reference sessions is
/// producer-computed. Folding the cut boundary locally instead would give
/// position `cut - 1` a different (Mac-computed) provenance than the full
/// reference handoff and the bit-exact logit comparison could never pass.
fn run_delta_probe(
    args: &Args,
    model: &Model,
    prepared: &PreparedPrompt,
    max_context: usize,
    receiver: &RemoteReceiver,
    cut: u64,
) -> Result<(), String> {
    let cut = usize::try_from(cut).map_err(|_| "delta prefix cut overflows usize")?;
    let prefix_bytes = std::fs::read(args.prefix_prompt_fixture.as_deref().expect("validated"))
        .map_err(|error| format!("cannot read the prefix fixture: {error}"))?;
    let prefix_tokens = parse_tokens(&prefix_bytes)?;
    if prefix_tokens.len() != cut {
        return Err(format!(
            "prefix fixture holds {} tokens but --delta-prefix-cut is {cut}",
            prefix_tokens.len()
        ));
    }
    if prefix_tokens != prepared.witnesses[..cut] {
        return Err("the prefix fixture is not a prefix of the prompt fixture".into());
    }
    if cut >= prepared.witnesses.len() {
        return Err("the delta prefix cut leaves no suffix in the prompt fixture".into());
    }
    phase("remote-delta-prefix-receive:start");
    let mut session = new_metal_session(model, max_context)?;
    // The producer request carries cut + 1 tokens (see the cell doc comment),
    // so the handoff installs the whole prefix and no local fold decode is
    // needed at the cut. The receiver itself holds back the last witness, so
    // the prefix receive is armed with cut + 1 witnesses to expect a
    // cut-token manifest.
    let prefix_receipt = receiver.receive(
        &mut session,
        None,
        &prepared.witnesses[..=cut],
        None,
        max_context,
        true,
    )?;
    phase("remote-delta-prefix-receive:done");
    phase("remote-delta-receive:start");
    let delta_receipt = receiver.receive(
        &mut session,
        None,
        &prepared.witnesses,
        None,
        max_context,
        true,
    )?;
    phase("remote-delta-receive:done");
    let boundary = *prepared.witnesses.last().expect("validated");
    if boundary == EMBEDDING_POSITION_WITNESS {
        return Err("remote prompt boundary is not a token".into());
    }
    let logits = session
        .decode(DecodeInput { token_id: boundary })
        .map_err(|error| error.to_string())?
        .logits;
    let delta = generate_target(model, &mut session, logits, args.output_tokens, true)?;
    phase("remote-delta-reference-receive:start");
    let mut reference_session = new_metal_session(model, max_context)?;
    let reference_receipt = receiver.receive(
        &mut reference_session,
        None,
        &prepared.witnesses,
        None,
        max_context,
        true,
    )?;
    phase("remote-delta-reference-receive:done");
    let reference_logits = reference_session
        .decode(DecodeInput { token_id: boundary })
        .map_err(|error| error.to_string())?
        .logits;
    let reference = generate_target(
        model,
        &mut reference_session,
        reference_logits,
        args.output_tokens,
        true,
    )?;
    let exact =
        delta.tokens == reference.tokens && delta.full_logit_digest == reference.full_logit_digest;
    println!(
        "{}",
        serde_json::json!({
            "schema": "muser.remote-qualify.v1",
            "kind": "delta-probe",
            "identity": args.identity,
            "prefix_cut": cut,
            "prefix_payload_bytes": prefix_receipt.installed_bytes,
            "delta_payload_bytes": delta_receipt.installed_bytes,
            "reference_payload_bytes": reference_receipt.installed_bytes,
            "delta_share_of_full": delta_receipt.installed_bytes as f64
                / reference_receipt.installed_bytes as f64,
            "delta_tokens_sha256": token_digest(&delta.tokens),
            "reference_tokens_sha256": token_digest(&reference.tokens),
            "exact_against_full_handoff": exact,
            "seal_eligible": false,
        })
    );
    if !exact {
        return Err("delta handoff output differs from the full handoff".into());
    }
    Ok(())
}

fn emit_cache_diff(
    local: &muser_engine::cache::SessionCacheSnapshot,
    remote: &muser_engine::cache::SessionCacheSnapshot,
) -> Result<(), String> {
    if local.position != remote.position
        || local.tokens != remote.tokens
        || local.elements_per_token != remote.elements_per_token
        || local.layers.len() != remote.layers.len()
    {
        return Err("remote/local cache diagnostic geometry differs".into());
    }
    let layer_zero_token_one_probe = if std::env::var_os("MUSER_REMOTE_CACHE_PROBE").is_some() {
        let local_layer = local
            .layers
            .first()
            .ok_or("local cache diagnostic has no layers")?;
        let remote_layer = remote
            .layers
            .first()
            .ok_or("remote cache diagnostic has no layers")?;
        Some(serde_json::json!({
            "layer": local_layer.layer,
            "token": 1,
            "local_key": plane_row_probe(&local_layer.key, local_layer.logical_count, 1, 16)?,
            "remote_key": plane_row_probe(&remote_layer.key, remote_layer.logical_count, 1, 16)?,
        }))
    } else {
        None
    };
    let mut layers = Vec::with_capacity(local.layers.len());
    for (local, remote) in local.layers.iter().zip(remote.layers.iter()) {
        let (key_mismatched, key_max_abs, key_mean_abs) = plane_diff(&local.key, &remote.key)?;
        let (value_mismatched, value_max_abs, value_mean_abs) =
            plane_diff(&local.value, &remote.value)?;
        let row_bytes = usize::try_from(local.logical_count)
            .ok()
            .filter(|count| *count != 0)
            .and_then(|count| local.key.len().checked_div(count))
            .ok_or("remote cache diagnostic row geometry is invalid")?;
        layers.push(PlaneDiff {
            layer: local.layer,
            logical_start: local.logical_start,
            logical_count: local.logical_count,
            key_mismatched,
            value_mismatched,
            key_max_abs,
            value_max_abs,
            key_mean_abs,
            value_mean_abs,
            key_mismatched_by_token: plane_mismatches_by_row(&local.key, &remote.key, row_bytes)?,
            value_mismatched_by_token: plane_mismatches_by_row(
                &local.value,
                &remote.value,
                row_bytes,
            )?,
            first_key_mismatch: first_plane_mismatch(&local.key, &remote.key, row_bytes)?,
            first_value_mismatch: first_plane_mismatch(&local.value, &remote.value, row_bytes)?,
        });
    }
    println!(
        "{}",
        serde_json::json!({
            "schema": "muser.remote-cache-diagnostic.v1",
            "kind": "poc-diagnostic",
            "position": local.position,
            "elements_per_token": local.elements_per_token,
            "exact_cache_bytes": layers.iter().all(|layer| {
                layer.key_mismatched == 0 && layer.value_mismatched == 0
            }),
            "layer_zero_token_one_probe": layer_zero_token_one_probe,
            "layers": layers,
            "seal_eligible": false,
        })
    );
    Ok(())
}

/// Emit bounded, non-verdict diagnostics for the first strict token refusal.
/// The ordinary target-token inequality remains the gate; this record only
/// makes the already-observed refusal attributable after the live session is
/// gone. It is enabled solely by `MUSER_REMOTE_FIRST_DIVERGENCE`.
fn emit_first_divergence_diagnostic(
    local: &Generation,
    remote: &Generation,
    output_row: usize,
    prompt_positions: usize,
) -> Result<(), String> {
    let local_prefix = local
        .cache_snapshot
        .as_ref()
        .ok_or("first-divergence local prefix cache was not retained")?;
    let remote_prefix = remote
        .cache_snapshot
        .as_ref()
        .ok_or("first-divergence remote prefix cache was not retained")?;
    let local_post = local
        .post_decode_cache_snapshot
        .as_ref()
        .ok_or("first-divergence local post-decode cache was not retained")?;
    let remote_post = remote
        .post_decode_cache_snapshot
        .as_ref()
        .ok_or("first-divergence remote post-decode cache was not retained")?;
    let causative_logical_row = prompt_positions
        .checked_add(output_row)
        .and_then(|position| position.checked_sub(1))
        .ok_or("first-divergence KV row underflow")?;
    let local_prefix_sha256 = session_cache_digest(local_prefix);
    let remote_prefix_sha256 = session_cache_digest(remote_prefix);
    let local_kv_row_sha256 = cache_row_digest(local_post, causative_logical_row)?;
    let remote_kv_row_sha256 = cache_row_digest(remote_post, causative_logical_row)?;
    println!(
        "{}",
        serde_json::json!({
            "schema": "muser.remote-first-divergence.v1",
            "kind": "diagnostic-only",
            "verdict_gate": "target-token-inequality-unchanged",
            "output_row": output_row,
            "local_token": local.tokens.get(output_row),
            "remote_token": remote.tokens.get(output_row),
            "local_top_logits": retained_top_logits(local, output_row, 8)?,
            "remote_top_logits": retained_top_logits(remote, output_row, 8)?,
            "prefix_cache": {
                "local_position": local_prefix.position,
                "remote_position": remote_prefix.position,
                "local_sha256": local_prefix_sha256,
                "remote_sha256": remote_prefix_sha256,
                "exact": local_prefix_sha256 == remote_prefix_sha256,
            },
            "causative_kv_row": {
                "logical_row": causative_logical_row,
                "meaning": "token consumed to produce the first divergent logit row",
                "local_session_position": local_post.position,
                "remote_session_position": remote_post.position,
                "local_sha256": local_kv_row_sha256,
                "remote_sha256": remote_kv_row_sha256,
                "exact": local_kv_row_sha256 == remote_kv_row_sha256,
            },
            "seal_eligible": false,
        })
    );
    Ok(())
}

/// Emit bounded, non-verdict diagnostics when strict target tokens agree but
/// the immediately following full-logit digest gate refuses. The gate and its
/// return path are unchanged; this only retains the already-computed delta and
/// the cache row that caused its first differing output row.
fn emit_full_logit_divergence_diagnostic(
    local: &Generation,
    remote: &Generation,
    difference: &RetainedLogitDiff,
    prompt_positions: usize,
) -> Result<(), String> {
    println!(
        "{}",
        full_logit_divergence_diagnostic_value(local, remote, difference, prompt_positions,)?
    );
    Ok(())
}

fn full_logit_divergence_diagnostic_value(
    local: &Generation,
    remote: &Generation,
    difference: &RetainedLogitDiff,
    prompt_positions: usize,
) -> Result<serde_json::Value, String> {
    if local.tokens != remote.tokens {
        return Err("full-logit diagnostic requires exact target tokens".into());
    }
    let local_prefix = local
        .cache_snapshot
        .as_ref()
        .ok_or("full-logit divergence local prefix cache was not retained")?;
    let remote_prefix = remote
        .cache_snapshot
        .as_ref()
        .ok_or("full-logit divergence remote prefix cache was not retained")?;
    let local_post = local
        .post_decode_cache_snapshot
        .as_ref()
        .ok_or("full-logit divergence local post-decode cache was not retained")?;
    let remote_post = remote
        .post_decode_cache_snapshot
        .as_ref()
        .ok_or("full-logit divergence remote post-decode cache was not retained")?;
    let causative_logical_row = prompt_positions
        .checked_add(difference.first_row)
        .and_then(|position| position.checked_sub(1))
        .ok_or("full-logit divergence KV row underflow")?;
    let local_prefix_sha256 = session_cache_digest(local_prefix);
    let remote_prefix_sha256 = session_cache_digest(remote_prefix);
    let local_kv_row_sha256 = cache_row_digest(local_post, causative_logical_row)?;
    let remote_kv_row_sha256 = cache_row_digest(remote_post, causative_logical_row)?;
    Ok(serde_json::json!({
        "schema": "muser.remote-full-logit-divergence.v1",
        "kind": "diagnostic-only",
        "verdict_gate": "target-full-logit-digest-unchanged",
        "target_tokens_exact": true,
        "first_row": difference.first_row,
        "first_logit": difference.first_logit,
        "mismatched_logits": difference.mismatched,
        "maximum_abs_delta": difference.maximum,
        "mean_abs_delta": difference.mean,
        "local_full_logit_sha256": local.full_logit_digest,
        "remote_full_logit_sha256": remote.full_logit_digest,
        "local_top_logits": retained_top_logits(local, difference.first_row, 8)?,
        "remote_top_logits": retained_top_logits(remote, difference.first_row, 8)?,
        "prefix_cache": {
            "local_position": local_prefix.position,
            "remote_position": remote_prefix.position,
            "local_sha256": local_prefix_sha256,
            "remote_sha256": remote_prefix_sha256,
            "exact": local_prefix_sha256 == remote_prefix_sha256,
        },
        "prefix_last_row_layers": cache_row_layer_difference(
            local_prefix,
            remote_prefix,
            prompt_positions.checked_sub(2).ok_or("full-logit prefix row underflow")?,
        )?,
        "causative_kv_row": {
            "logical_row": causative_logical_row,
            "meaning": "token consumed to produce the first differing full-logit row",
            "local_session_position": local_post.position,
            "remote_session_position": remote_post.position,
            "local_sha256": local_kv_row_sha256,
            "remote_sha256": remote_kv_row_sha256,
            "exact": local_kv_row_sha256 == remote_kv_row_sha256,
            "layers": cache_row_layer_difference(
                local_post,
                remote_post,
                causative_logical_row,
            )?,
        },
        "seal_eligible": false,
    }))
}

fn retained_top_logits(
    generation: &Generation,
    row: usize,
    count: usize,
) -> Result<Vec<serde_json::Value>, String> {
    let logits = generation
        .retained_logits
        .as_deref()
        .ok_or("first-divergence logits were not retained")?;
    let row_width = logits
        .len()
        .checked_div(generation.tokens.len())
        .filter(|width| {
            *width != 0 && width.saturating_mul(generation.tokens.len()) == logits.len()
        })
        .ok_or("first-divergence retained logit rows are not rectangular")?;
    let values = logits
        .chunks_exact(row_width)
        .nth(row)
        .ok_or("first-divergence logit row is absent")?;
    let mut tokens = (0..values.len()).collect::<Vec<_>>();
    tokens.sort_unstable_by(|left, right| {
        values[*right]
            .total_cmp(&values[*left])
            .then_with(|| left.cmp(right))
    });
    Ok(tokens
        .into_iter()
        .take(count)
        .enumerate()
        .map(|(rank, token)| {
            serde_json::json!({
                "rank": rank,
                "token": token,
                "value": values[token],
                "bits": format!("{:08x}", values[token].to_bits()),
            })
        })
        .collect())
}

fn session_cache_digest(snapshot: &muser_engine::cache::SessionCacheSnapshot) -> String {
    let mut digest = Sha256::new();
    digest.update(b"muser-session-cache-diagnostic-v1\0");
    digest.update(snapshot.position.to_le_bytes());
    digest.update(snapshot.elements_per_token.to_le_bytes());
    for token in snapshot.tokens.iter() {
        digest.update(token.to_le_bytes());
    }
    for plane in snapshot.layers.iter() {
        digest.update(plane.layer.to_le_bytes());
        digest.update(plane.logical_start.to_le_bytes());
        digest.update(plane.logical_count.to_le_bytes());
        digest.update((plane.encoding.width_bytes() as u64).to_le_bytes());
        digest.update(plane.key.as_ref());
        digest.update(plane.value.as_ref());
    }
    format!("{:x}", digest.finalize())
}

fn cache_row_digest(
    snapshot: &muser_engine::cache::SessionCacheSnapshot,
    logical_row: usize,
) -> Result<String, String> {
    let logical_row = u64::try_from(logical_row)
        .map_err(|_| "first-divergence KV row exceeds u64".to_string())?;
    let mut digest = Sha256::new();
    digest.update(b"muser-session-cache-row-diagnostic-v1\0");
    digest.update(logical_row.to_le_bytes());
    for plane in snapshot.layers.iter() {
        let row_offset = logical_row
            .checked_sub(plane.logical_start)
            .filter(|offset| *offset < plane.logical_count)
            .ok_or_else(|| {
                format!(
                    "first-divergence KV row {logical_row} is absent from layer {} range {}..{}",
                    plane.layer,
                    plane.logical_start,
                    plane.logical_start.saturating_add(plane.logical_count),
                )
            })?;
        let row_bytes = usize::try_from(snapshot.elements_per_token)
            .ok()
            .and_then(|elements| elements.checked_mul(plane.encoding.width_bytes()))
            .ok_or("first-divergence KV row byte size overflow")?;
        let start = usize::try_from(row_offset)
            .ok()
            .and_then(|offset| offset.checked_mul(row_bytes))
            .ok_or("first-divergence KV row offset overflow")?;
        let end = start
            .checked_add(row_bytes)
            .ok_or("first-divergence KV row end overflow")?;
        let key = plane
            .key
            .get(start..end)
            .ok_or("first-divergence key row is truncated")?;
        let value = plane
            .value
            .get(start..end)
            .ok_or("first-divergence value row is truncated")?;
        digest.update(plane.layer.to_le_bytes());
        digest.update((plane.encoding.width_bytes() as u64).to_le_bytes());
        digest.update(b"K");
        digest.update(key);
        digest.update(b"V");
        digest.update(value);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn cache_row_layer_difference(
    local: &muser_engine::cache::SessionCacheSnapshot,
    remote: &muser_engine::cache::SessionCacheSnapshot,
    logical_row: usize,
) -> Result<serde_json::Value, String> {
    if local.elements_per_token != remote.elements_per_token
        || local.layers.len() != remote.layers.len()
    {
        return Err("cache row layer diagnostic geometry differs".into());
    }
    let logical_row = u64::try_from(logical_row)
        .map_err(|_| "cache row layer diagnostic row exceeds u64".to_string())?;
    let mut mismatching_layers = 0usize;
    let mut first = None;
    for (local_plane, remote_plane) in local.layers.iter().zip(remote.layers.iter()) {
        if local_plane.layer != remote_plane.layer
            || local_plane.encoding != remote_plane.encoding
            || local_plane.logical_start != remote_plane.logical_start
            || local_plane.logical_count != remote_plane.logical_count
        {
            return Err(format!(
                "cache row layer diagnostic geometry differs at layer {}",
                local_plane.layer
            ));
        }
        let row_bytes = usize::try_from(local.elements_per_token)
            .ok()
            .and_then(|elements| elements.checked_mul(local_plane.encoding.width_bytes()))
            .ok_or("cache row layer diagnostic byte size overflow")?;
        let offset = logical_row
            .checked_sub(local_plane.logical_start)
            .filter(|offset| *offset < local_plane.logical_count)
            .and_then(|offset| usize::try_from(offset).ok())
            .and_then(|offset| offset.checked_mul(row_bytes))
            .ok_or_else(|| {
                format!(
                    "cache row layer diagnostic row {logical_row} is absent from layer {}",
                    local_plane.layer
                )
            })?;
        let end = offset
            .checked_add(row_bytes)
            .ok_or("cache row layer diagnostic end overflow")?;
        let local_key = local_plane
            .key
            .get(offset..end)
            .ok_or("cache row layer diagnostic local key is truncated")?;
        let remote_key = remote_plane
            .key
            .get(offset..end)
            .ok_or("cache row layer diagnostic remote key is truncated")?;
        let local_value = local_plane
            .value
            .get(offset..end)
            .ok_or("cache row layer diagnostic local value is truncated")?;
        let remote_value = remote_plane
            .value
            .get(offset..end)
            .ok_or("cache row layer diagnostic remote value is truncated")?;
        if local_key != remote_key || local_value != remote_value {
            mismatching_layers += 1;
            if first.is_none() {
                first = Some(serde_json::json!({
                    "layer": local_plane.layer,
                    "encoding_width_bytes": local_plane.encoding.width_bytes(),
                    "key": cache_plane_row_difference(
                        local_key,
                        remote_key,
                        local_plane.encoding.width_bytes(),
                    )?,
                    "value": cache_plane_row_difference(
                        local_value,
                        remote_value,
                        local_plane.encoding.width_bytes(),
                    )?,
                }));
            }
        }
    }
    Ok(serde_json::json!({
        "logical_row": logical_row,
        "exact": mismatching_layers == 0,
        "mismatching_layers": mismatching_layers,
        "first_mismatch": first,
    }))
}

fn cache_plane_row_difference(
    local: &[u8],
    remote: &[u8],
    width: usize,
) -> Result<serde_json::Value, String> {
    if width == 0 || local.len() != remote.len() || !local.len().is_multiple_of(width) {
        return Err("cache plane row diagnostic geometry differs".into());
    }
    let differences = local
        .chunks_exact(width)
        .zip(remote.chunks_exact(width))
        .enumerate()
        .filter(|(_, (left, right))| left != right)
        .collect::<Vec<_>>();
    let first = differences.first().map(|(index, (left, right))| {
        let bits = |bytes: &[u8]| {
            bytes
                .iter()
                .rev()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        };
        serde_json::json!({
            "element": index,
            "local_bits": bits(left),
            "remote_bits": bits(right),
        })
    });
    Ok(serde_json::json!({
        "exact": differences.is_empty(),
        "mismatching_elements": differences.len(),
        "first_mismatch": first,
        "local_sha256": format!("{:x}", Sha256::digest(local)),
        "remote_sha256": format!("{:x}", Sha256::digest(remote)),
    }))
}

fn plane_row_probe(
    plane: &[u8],
    logical_count: u64,
    token: usize,
    count: usize,
) -> Result<Vec<serde_json::Value>, String> {
    let row_bytes = usize::try_from(logical_count)
        .ok()
        .filter(|rows| *rows != 0)
        .and_then(|rows| plane.len().checked_div(rows))
        .ok_or("remote cache probe row geometry is invalid")?;
    let row = plane
        .chunks_exact(row_bytes)
        .nth(token)
        .ok_or("remote cache probe token is absent")?;
    Ok(row
        .chunks_exact(2)
        .take(count)
        .enumerate()
        .map(|(element, bytes)| {
            let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
            serde_json::json!({
                "element": element,
                "bits": bits,
                "value": half::f16::from_bits(bits).to_f32(),
            })
        })
        .collect())
}

fn first_plane_mismatch(
    local: &[u8],
    remote: &[u8],
    row_bytes: usize,
) -> Result<Option<CellDiff>, String> {
    if row_bytes == 0
        || !row_bytes.is_multiple_of(2)
        || local.len() != remote.len()
        || !local.len().is_multiple_of(row_bytes)
    {
        return Err("remote/local cache diagnostic row size differs".into());
    }
    for (index, (left, right)) in local
        .chunks_exact(2)
        .zip(remote.chunks_exact(2))
        .enumerate()
    {
        let local_bits = u16::from_le_bytes([left[0], left[1]]);
        let remote_bits = u16::from_le_bytes([right[0], right[1]]);
        if local_bits != remote_bits {
            let elements_per_row = row_bytes / 2;
            return Ok(Some(CellDiff {
                index,
                token: index / elements_per_row,
                element: index % elements_per_row,
                local_bits,
                remote_bits,
                local: half::f16::from_bits(local_bits).to_f32(),
                remote: half::f16::from_bits(remote_bits).to_f32(),
            }));
        }
    }
    Ok(None)
}

fn plane_mismatches_by_row(
    local: &[u8],
    remote: &[u8],
    row_bytes: usize,
) -> Result<Vec<usize>, String> {
    if row_bytes == 0
        || !row_bytes.is_multiple_of(2)
        || local.len() != remote.len()
        || !local.len().is_multiple_of(row_bytes)
    {
        return Err("remote/local cache diagnostic row size differs".into());
    }
    Ok(local
        .chunks_exact(row_bytes)
        .zip(remote.chunks_exact(row_bytes))
        .map(|(left, right)| {
            left.chunks_exact(2)
                .zip(right.chunks_exact(2))
                .filter(|(left, right)| left != right)
                .count()
        })
        .collect())
}

fn plane_diff(local: &[u8], remote: &[u8]) -> Result<(usize, f32, f64), String> {
    if local.len() != remote.len() || !local.len().is_multiple_of(2) {
        return Err("remote/local cache diagnostic plane size differs".into());
    }
    let mut mismatched = 0usize;
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    for (left, right) in local.chunks_exact(2).zip(remote.chunks_exact(2)) {
        let left_bits = u16::from_le_bytes([left[0], left[1]]);
        let right_bits = u16::from_le_bytes([right[0], right[1]]);
        mismatched += usize::from(left_bits != right_bits);
        let delta = (half::f16::from_bits(left_bits).to_f32()
            - half::f16::from_bits(right_bits).to_f32())
        .abs();
        max_abs = max_abs.max(delta);
        sum_abs += f64::from(delta);
    }
    let elements = local.len() / 2;
    Ok((
        mismatched,
        max_abs,
        if elements == 0 {
            0.0
        } else {
            sum_abs / elements as f64
        },
    ))
}

struct RetainedLogitDiff {
    maximum: f32,
    mean: f64,
    mismatched: usize,
    first_row: usize,
    first_logit: usize,
}

fn retained_logit_diff(
    local: &Generation,
    remote: &Generation,
) -> Result<RetainedLogitDiff, String> {
    let rows = local.tokens.len();
    let local = local
        .retained_logits
        .as_deref()
        .ok_or("POC local full logits were not retained")?;
    let remote = remote
        .retained_logits
        .as_deref()
        .ok_or("POC remote full logits were not retained")?;
    if local.len() != remote.len() || local.is_empty() {
        return Err("POC remote/local full-logit geometry differs".into());
    }
    let row_width = local
        .len()
        .checked_div(rows)
        .filter(|width| *width != 0 && width.saturating_mul(rows) == local.len())
        .ok_or("POC retained logit rows are not rectangular")?;
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    let mut mismatched = 0usize;
    let mut first = None;
    for (index, (&left, &right)) in local.iter().zip(remote).enumerate() {
        let difference = (left - right).abs();
        if !difference.is_finite() {
            return Err("POC remote/local logit difference is nonfinite".into());
        }
        max_abs = max_abs.max(difference);
        sum_abs += f64::from(difference);
        if left.to_bits() != right.to_bits() {
            mismatched += 1;
            first.get_or_insert(index);
        }
    }
    let first = first.ok_or("POC logit digests differ without a bitwise element mismatch")?;
    Ok(RetainedLogitDiff {
        maximum: max_abs,
        mean: sum_abs / local.len() as f64,
        mismatched,
        first_row: first / row_width,
        first_logit: first % row_width,
    })
}

fn run_local_dflash(
    assistant: &mut DFlashAssistant,
    model: &Model,
    prepared: &PreparedPrompt,
    max_context: usize,
    output_tokens: usize,
    verify_length: usize,
) -> Result<DFlashRun, String> {
    phase("local-dflash-prefill:start");
    let mut session = new_metal_session(model, max_context)?;
    let prepared = assistant
        .prepare_greedy_batch(model, &mut session, prepared.batch.clone())
        .map_err(|error| error.to_string())?;
    let context = assistant.export_context_snapshot();
    phase("local-dflash-prefill:done");
    phase("local-dflash-decode:start");
    let decode_started = Instant::now();
    let (tokens, stats) = assistant
        .generate_prepared_greedy(
            model,
            &mut session,
            prepared,
            output_tokens,
            verify_length,
            &[],
        )
        .map_err(|error| error.to_string())?;
    phase("local-dflash-decode:done");
    Ok(DFlashRun {
        tokens,
        stats,
        context,
        receipt: None,
        decode_ns: nanos(decode_started.elapsed().as_nanos()),
    })
}

#[allow(clippy::too_many_arguments)]
fn run_remote_dflash(
    receiver: &RemoteReceiver,
    assistant: &mut DFlashAssistant,
    model: &Model,
    prepared: &PreparedPrompt,
    max_context: usize,
    output_tokens: usize,
    verify_length: usize,
    wait_without_control: bool,
) -> Result<DFlashRun, String> {
    if std::env::var_os("MUSER_REMOTE_QUALIFY_SERIAL").is_some() {
        phase("remote-dflash-prefill-export-receive:serial-quiesce");
        std::thread::sleep(std::time::Duration::from_secs(5));
    }
    phase("remote-dflash-prefill-export-receive:start");
    let mut session = new_metal_session(model, max_context)?;
    assistant.reset();
    let receipt = receiver.receive(
        &mut session,
        Some(assistant),
        &prepared.witnesses,
        prepared.multimodal.clone(),
        max_context,
        wait_without_control,
    )?;
    let context = assistant.export_context_snapshot();
    phase("remote-dflash-prefill-export-receive:done");
    let boundary = *prepared.witnesses.last().expect("validated");
    phase("remote-dflash-decode:start");
    let decode_started = Instant::now();
    let (tokens, stats) = assistant
        .generate_greedy_from_installed(
            model,
            &mut session,
            boundary,
            output_tokens,
            verify_length,
            &[],
        )
        .map_err(|error| error.to_string())?;
    phase("remote-dflash-decode:done");
    Ok(DFlashRun {
        tokens,
        stats,
        context,
        receipt: Some(receipt),
        decode_ns: nanos(decode_started.elapsed().as_nanos()),
    })
}

fn generate_target(
    model: &Model,
    session: &mut Session,
    mut logits: Vec<f32>,
    count: usize,
    retain_logits: bool,
) -> Result<Generation, String> {
    let mut tokens = Vec::with_capacity(count);
    let mut digest = Sha256::new();
    let mut retained = retain_logits.then(|| Vec::with_capacity(count * logits.len()));
    let decode_started = Instant::now();
    let mut first_64_decode_ns = 0;
    for index in 0..count {
        reject_invalid_logits(&logits, model.config().vocab_size, index)?;
        for value in &logits {
            digest.update(value.to_bits().to_le_bytes());
        }
        if let Some(rows) = retained.as_mut() {
            rows.extend_from_slice(&logits);
        }
        let token = argmax(&logits) as u32;
        if model.config().eos_tokens.contains(&token) {
            return Err(format!("early EOS at output row {index}"));
        }
        tokens.push(token);
        if index == 63 {
            first_64_decode_ns = nanos(decode_started.elapsed().as_nanos());
        }
        if index + 1 < count {
            logits = session
                .decode(DecodeInput { token_id: token })
                .map_err(|error| error.to_string())?
                .logits;
        }
    }
    if count < 64 {
        first_64_decode_ns = nanos(decode_started.elapsed().as_nanos());
    }
    let post_decode_cache_snapshot =
        if retain_logits && std::env::var_os("MUSER_REMOTE_FIRST_DIVERGENCE").is_some() {
            Some(
                session
                    .export_cache_snapshot()
                    .map_err(|error| error.to_string())?,
            )
        } else {
            None
        };
    Ok(Generation {
        tokens,
        full_logit_digest: format!("{:x}", digest.finalize()),
        retained_logits: retained,
        cache_snapshot: None,
        post_decode_cache_snapshot,
        first_64_decode_ns,
    })
}

fn reject_invalid_logits(logits: &[f32], expected: usize, index: usize) -> Result<(), String> {
    if logits.len() != expected {
        return Err(format!(
            "invalid target logits at output row {index}: len {} expected {expected}",
            logits.len()
        ));
    }
    let nans = logits.iter().filter(|value| value.is_nan()).count();
    let infs = logits.iter().filter(|value| value.is_infinite()).count();
    if nans + infs > 0 {
        return Err(format!(
            "invalid target logits at output row {index}: {nans} NaN, {infs} Inf (of {expected})"
        ));
    }
    Ok(())
}

fn prepare_prompt(args: &Args, model: &Model, tokens: &[u32]) -> Result<PreparedPrompt, String> {
    if args.variant != Variant::Multimodal {
        return Ok(PreparedPrompt {
            batch: PrefillBatch::tokens(tokens.to_vec()),
            cached_batch: PrefillBatch::tokens(tokens[..tokens.len() - 1].to_vec()),
            witnesses: tokens.to_vec(),
            multimodal: None,
        });
    }
    let mmproj = args.mmproj.as_deref().expect("validated");
    let bridge = args.mtmd_bridge.as_deref().expect("validated");
    let image_path = args.image.as_deref().expect("validated");
    let encoded = std::fs::read(image_path)
        .map_err(|error| format!("cannot read {}: {error}", image_path.display()))?;
    let vision = load_vision(mmproj, bridge)?;
    if vision.config.output_dim != model.config().hidden_dim {
        return Err(format!(
            "vision output width {} differs from target hidden width {}",
            vision.config.output_dim,
            model.config().hidden_dim
        ));
    }
    let pixels = vision
        .preprocess_bytes(&encoded)
        .map_err(|error| error.to_string())?;
    let projected = vision
        .projected_token_count(&pixels)
        .map_err(|error| error.to_string())?;
    let embeddings = vision
        .encode_accelerated(&encoded, &pixels)
        .map_err(|error| error.to_string())?;
    if embeddings.len() != projected {
        return Err("vision projector row count differs from geometry".into());
    }
    let text_positions = tokens
        .len()
        .checked_sub(projected)
        .ok_or("multimodal image has more positions than the requested context")?;
    if text_positions < 3 {
        return Err("multimodal context leaves fewer than three text tokens".into());
    }
    let split = text_positions - 2;
    let prefix = tokens[..split].to_vec();
    let suffix = tokens[tokens.len() - 2..].to_vec();
    let image_sha = Sha256::digest(&encoded);
    let mut image_sequence = Sha256::new();
    image_sequence.update(image_sha);
    let identity = MultimodalIdentityV2 {
        projector_sha256: sha256_file(mmproj)?,
        preprocessing_sha256: format!("{:x}", Sha256::digest(VISION_PREPROCESSING_CONTRACT)),
        image_sequence_sha256: format!("{:x}", image_sequence.finalize()),
    };
    let mut witnesses = prefix.clone();
    witnesses.extend(std::iter::repeat_n(EMBEDDING_POSITION_WITNESS, projected));
    witnesses.extend_from_slice(&suffix);
    Ok(PreparedPrompt {
        batch: PrefillBatch {
            segments: vec![
                PrefillSegment::Tokens(prefix.clone()),
                PrefillSegment::Embeddings(EmbeddingSegment::new(embeddings.clone())),
                PrefillSegment::Tokens(suffix.clone()),
            ],
        },
        cached_batch: PrefillBatch {
            segments: vec![
                PrefillSegment::Tokens(prefix.clone()),
                PrefillSegment::Embeddings(EmbeddingSegment::new(embeddings)),
                PrefillSegment::Tokens(suffix[..suffix.len() - 1].to_vec()),
            ],
        },
        witnesses,
        multimodal: Some((
            identity,
            vec![
                PrefillControlSegmentV2::Tokens { token_ids: prefix },
                PrefillControlSegmentV2::Image {
                    data_base64: base64::engine::general_purpose::STANDARD.encode(&encoded),
                    sha256: format!("{image_sha:x}"),
                    projected_tokens: projected as u32,
                },
                PrefillControlSegmentV2::Tokens { token_ids: suffix },
            ],
        )),
    })
}

fn phase_metrics(receipt: &ProducerPhaseReceiptV1) -> Result<(f64, f64, f64), String> {
    let prefill = receipt
        .prefill_end_unix_ns
        .checked_sub(receipt.prefill_start_unix_ns)
        .ok_or("producer prefill phase underflow")?;
    let export = receipt
        .state_saved_unix_ns
        .checked_sub(receipt.prefill_end_unix_ns)
        .ok_or("producer export phase underflow")?;
    let first = receipt
        .first_segment_sent_unix_ns
        .saturating_sub(receipt.prefill_start_unix_ns);
    let transfer = receipt
        .transfer_acked_unix_ns
        .checked_sub(receipt.transfer_start_unix_ns)
        .ok_or("producer transfer phase underflow")?;
    if prefill == 0 || transfer == 0 {
        return Err("producer receipt contains a zero-length measured phase".into());
    }
    let overlap_end = receipt
        .transfer_acked_unix_ns
        .min(receipt.prefill_end_unix_ns);
    let overlap = overlap_end.saturating_sub(receipt.transfer_start_unix_ns);
    Ok((
        export as f64 / prefill as f64,
        first as f64 / prefill as f64,
        overlap as f64 / transfer as f64,
    ))
}

fn load_external_producer_phase(
    path: &Path,
    receiver: &RemoteReceiveReceipt,
    prompt_tokens: &[u32],
    profile: ProducerReceiptProfile,
) -> Result<ProducerPhaseReceiptV1, String> {
    // The external Spark client and receiver finish concurrently. Wait only
    // for the atomic, exclusive-create client receipt, never for accelerator
    // work. In normal operation the file is already present.
    let deadline = Instant::now() + Duration::from_secs(5);
    let bytes = loop {
        match std::fs::read(path) {
            Ok(bytes) => break bytes,
            Err(error)
                if error.kind() == std::io::ErrorKind::NotFound && Instant::now() < deadline =>
            {
                std::thread::sleep(Duration::from_millis(25));
            }
            Err(error) => {
                return Err(format!(
                    "read external producer receipt {}: {error}",
                    path.display()
                ))
            }
        }
    };
    parse_external_producer_phase(&bytes, receiver, prompt_tokens, profile)
}

fn parse_external_producer_phase(
    bytes: &[u8],
    receiver: &RemoteReceiveReceipt,
    prompt_tokens: &[u32],
    profile: ProducerReceiptProfile,
) -> Result<ProducerPhaseReceiptV1, String> {
    let client: ExternalProducerClientReceipt = serde_json::from_slice(bytes)
        .map_err(|error| format!("parse external producer receipt: {error}"))?;
    let response = client.response;
    let producer = response.producer_receipt;
    let handoff = producer.handoff;
    let valid_producer = match producer.schema.as_str() {
        "muser.spark-nvfp4-prefill.v1" => producer
            .producer_mode
            .as_deref()
            .is_none_or(|mode| mode == "exact"),
        "muser.spark-nvfp4-prefill.v2" => producer.producer_mode.as_deref() == Some("native"),
        _ => false,
    };
    if client.schema != "muser.spark-nvfp4-prefill-client.v1"
        || response.status != "ok"
        || !valid_producer
    {
        return Err("external producer receipt has an invalid schema or status".into());
    }
    if client.token_count != prompt_tokens.len()
        || response.prompt_token_count != prompt_tokens.len()
        || producer.prompt_token_count != prompt_tokens.len()
        || producer.token_ids_sha256 != token_digest(prompt_tokens)
    {
        return Err("external producer receipt does not bind the prompt tokens".into());
    }
    if response.request_id != receiver.transfer_id
        || handoff.transfer_id != receiver.transfer_id
        || handoff.generation != receiver.generation
        || handoff.segments != receiver.installed_segments
        || handoff.payload_bytes != receiver.installed_bytes
        || !handoff.ack
    {
        return Err("external producer receipt does not bind the installed transfer".into());
    }
    if producer.phase_ns.connector_total == 0
        || producer.phase_ns.d2h_complete_offset == 0
        || producer.phase_ns.d2h_complete_offset > producer.phase_ns.connector_total
        || handoff.payload_wire_ns == 0
        || handoff.payload_wire_source != "linux-tcp-info-busy-time-v1"
        || handoff.payload_pacing_bps < 4_000_000_000
        || handoff.first_segment_sent_unix_ns < handoff.transfer_start_unix_ns
        || handoff.transfer_acked_unix_ns < handoff.first_segment_sent_unix_ns
    {
        return Err("external producer receipt contains invalid phase evidence".into());
    }
    let prefill_end_unix_ns = handoff
        .transfer_start_unix_ns
        .checked_add(producer.phase_ns.d2h_complete_offset)
        .ok_or("external producer prefill timestamp overflow")?;
    match profile {
        ProducerReceiptProfile::Enrolled => {
            if producer.schema != "muser.spark-nvfp4-prefill.v2" {
                return Err("enrolled producer requires a v2 native producer receipt".into());
            }
            // Streaming producer: the first segment must go on the wire no later
            // than D2H completion — that overlap is the point of the v2 seam.
            let first_sent = producer
                .phase_ns
                .first_segment_sent_offset
                .ok_or("streaming producer receipt omits first_segment_sent_offset")?;
            if first_sent == 0 || first_sent > producer.phase_ns.d2h_complete_offset {
                return Err(
                    "streaming producer sent its first segment after D2H completion".into(),
                );
            }
        }
        ProducerReceiptProfile::HistoricalPreStreamingControl => {
            if producer.schema != "muser.spark-nvfp4-prefill.v2" {
                return Err("pre-streaming control requires a v2 native producer receipt".into());
            }
            if producer.phase_ns.first_segment_sent_offset.is_some() {
                return Err(
                    "pre-streaming control unexpectedly includes streaming first-segment evidence"
                        .into(),
                );
            }
            if prefill_end_unix_ns > handoff.first_segment_sent_unix_ns {
                return Err(
                    "pre-streaming control sent its first segment before D2H completion".into(),
                );
            }
        }
    }
    let prefill_tokens = u32::try_from(prompt_tokens.len())
        .map_err(|_| "external producer prompt token count exceeds u32")?;
    let streaming = producer.schema == "muser.spark-nvfp4-prefill.v2"
        && profile == ProducerReceiptProfile::Enrolled;
    Ok(ProducerPhaseReceiptV1 {
        // The connector timer starts immediately before vLLM invokes the
        // exact cache export path; D2H completion is recorded as its offset.
        prefill_start_unix_ns: handoff.transfer_start_unix_ns,
        prefill_end_unix_ns,
        // Under the streaming (v2) seam, state is fully saved when the last
        // layer's D2H lands — segments have already streamed by then, so the
        // deferred design's "export after prefill" phase does not exist.
        state_saved_unix_ns: if streaming {
            prefill_end_unix_ns
        } else {
            handoff.first_segment_sent_unix_ns
        },
        transfer_start_unix_ns: handoff.transfer_start_unix_ns,
        first_segment_sent_unix_ns: handoff.first_segment_sent_unix_ns,
        transfer_acked_unix_ns: handoff.transfer_acked_unix_ns,
        prefill_tokens,
        payload_bytes: handoff.payload_bytes,
        payload_wire_ns: handoff.payload_wire_ns,
    })
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn new_metal_session(model: &Model, max_context: usize) -> Result<Session, String> {
    model
        .new_metal_session(muser_engine::SessionConfig { max_context })
        .map_err(|error| error.to_string())
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn new_metal_session(_model: &Model, _max_context: usize) -> Result<Session, String> {
    Err("remote qualification requires macOS and the metal feature".into())
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn load_dflash(path: &Path, model: &Model) -> Result<DFlashAssistant, String> {
    DFlashAssistant::load_metal(path, model).map_err(|error| error.to_string())
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn load_dflash(_path: &Path, _model: &Model) -> Result<DFlashAssistant, String> {
    Err("remote DFlash qualification requires macOS and the metal feature".into())
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn load_vision(path: &Path, bridge: &Path) -> Result<VisionModel, String> {
    VisionModel::load_metal(path, bridge).map_err(|error| error.to_string())
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn load_vision(_path: &Path, _bridge: &Path) -> Result<VisionModel, String> {
    Err("remote multimodal qualification requires macOS and the metal feature".into())
}

fn validate_args(args: &Args) -> Result<(), String> {
    if usize::from(args.poc)
        + usize::from(args.p4)
        + usize::from(args.diagnostic)
        + usize::from(args.onboarding_native)
        > 1
    {
        return Err(
            "--poc, --p4, --diagnostic, and --onboarding-native are mutually exclusive".into(),
        );
    }
    if args.local_only && (!args.diagnostic || args.variant != Variant::Text) {
        return Err("--local-only requires --diagnostic --variant text".into());
    }
    if args.drift_graded
        && (!(args.diagnostic || args.p4 || args.onboarding_native) || args.local_only)
    {
        return Err(
            "--drift-graded requires a live diagnostic, P4, or native onboarding cell".into(),
        );
    }
    if args.reference_once
        && (!args.drift_graded
            || !(args.p4 || args.onboarding_native)
            || args.variant != Variant::Text)
    {
        return Err("--reference-once requires drift-graded P4/native-onboarding text".into());
    }
    if args.onboarding_native
        && (!args.drift_graded
            || !args.reference_once
            || args.variant != Variant::Text
            || args.performance_only
            || args.fast_consumer_math
            || args.external_producer_receipt.is_some()
            || args.external_producer_receipt_dir.is_some()
            || args.delta_prefix_cut.is_some()
            || args.local_only)
    {
        return Err(
            "--onboarding-native requires --drift-graded --reference-once --variant text and no alternate receipt/performance mode"
                .into(),
        );
    }
    if args.performance_only
        && (!(args.p4 || args.diagnostic)
            || !matches!(args.variant, Variant::Text | Variant::TargetPlusDflash)
            || args.reference_once
            || args.drift_graded
            || args.external_producer_receipt_dir.is_none())
    {
        return Err(
            "--performance-only requires live P4/diagnostic text/DFlash and a receipt directory"
                .into(),
        );
    }
    if args.fast_consumer_math && !args.performance_only {
        return Err("--fast-consumer-math requires --performance-only".into());
    }
    if args.producer_receipt_profile == ProducerReceiptProfile::HistoricalPreStreamingControl
        && (!args.p4 || !args.performance_only || args.variant != Variant::Text)
    {
        return Err(
            "--pre-streaming-control requires --p4 --performance-only --variant text".into(),
        );
    }
    if args.external_producer_receipt.is_some()
        && (!args.diagnostic || args.local_only || args.repetitions != 1)
    {
        return Err("--external-producer-receipt requires one live --diagnostic repetition".into());
    }
    if args.external_producer_receipt.is_some() && args.external_producer_receipt_dir.is_some() {
        return Err(
            "--external-producer-receipt and --external-producer-receipt-dir are mutually exclusive"
                .into(),
        );
    }
    if args.external_producer_receipt_dir.is_some()
        && (!(args.p4 || args.diagnostic) || args.local_only)
    {
        return Err("--external-producer-receipt-dir requires a live P4/diagnostic cell".into());
    }
    if args.delta_prefix_cut.is_some() != args.prefix_prompt_fixture.is_some() {
        return Err("--delta-prefix-cut and --prefix-prompt-fixture are a pair".into());
    }
    if let Some(cut) = args.delta_prefix_cut {
        if cut == 0 || cut % 256 != 0 {
            return Err("--delta-prefix-cut must be positive and 256-aligned".into());
        }
        if args.poc
            || args.p4
            || args.diagnostic
            || args.onboarding_native
            || args.performance_only
            || args.local_only
        {
            return Err("--delta-prefix-cut is a standalone probe, not a packet modifier".into());
        }
    }
    if !args.dry_run && !args.local_only && !args.fast_consumer_math {
        require_strict_cross_vendor_qk(std::env::var("MUSER_CROSS_VENDOR_QK").ok().as_deref())?;
    }
    if !args.dry_run
        && args.fast_consumer_math
        && std::env::var_os("MUSER_CROSS_VENDOR_QK").is_some()
    {
        return Err("--fast-consumer-math requires MUSER_CROSS_VENDOR_QK to be unset".into());
    }
    if args.onboarding_native {
        if args.repetitions != 3 || args.output_tokens != 256 {
            return Err(
                "native onboarding requires exactly 3 repetitions and 256 output tokens".into(),
            );
        }
    } else if args.poc {
        if args.repetitions != 1 || !(1..=16).contains(&args.output_tokens) {
            return Err("remote POC requires one repetition and 1..=16 output tokens".into());
        }
    } else if args.p4 {
        if !valid_p4_geometry(args.performance_only, args.repetitions, args.output_tokens) {
            if args.performance_only {
                return Err(
                    "remote P4 performance-only qualification requires at least 3 repetitions and 1..=256 output tokens"
                        .into(),
                );
            }
            return Err(
                "remote P4 qualification requires exactly 5 repetitions and 256 output tokens"
                    .into(),
            );
        }
    } else if args.diagnostic {
        if args.repetitions != 1 || !(1..=256).contains(&args.output_tokens) {
            return Err(
                "remote diagnostic requires one repetition and 1..=256 output tokens".into(),
            );
        }
    } else if args.delta_prefix_cut.is_some() {
        if args.repetitions != 1 || !(1..=256).contains(&args.output_tokens) {
            return Err(
                "the delta probe runs one cell of 1..=256 output tokens (three handoffs)".into(),
            );
        }
    } else if args.repetitions != 3 || args.output_tokens != 256 {
        return Err(
            "remote qualification requires exactly 3 repetitions and 256 output tokens".into(),
        );
    }
    if !matches!(args.verify_length, 3 | 7 | 15) {
        return Err("verify length must be one of 3, 7, or 15".into());
    }
    let dflash = args.dflash.is_some();
    let vision = args.mmproj.is_some() || args.mtmd_bridge.is_some() || args.image.is_some();
    match args.variant {
        Variant::Text if dflash || vision => {
            Err("text variant does not accept DFlash/vision artifacts".into())
        }
        Variant::Multimodal
            if dflash
                || args.mmproj.is_none()
                || args.mtmd_bridge.is_none()
                || args.image.is_none() =>
        {
            Err("multimodal variant requires --mmproj, --mtmd-bridge, and --image only".into())
        }
        Variant::TargetPlusDflash if !dflash || vision => {
            Err("target-plus-dflash variant requires --dflash and no vision artifacts".into())
        }
        _ => Ok(()),
    }
}

fn parse_args() -> Result<Args, String> {
    let mut model = None;
    let mut prompt_fixture = None;
    let mut cluster_config = None;
    let mut variant = None;
    let mut dflash = None;
    let mut mmproj = None;
    let mut mtmd_bridge = None;
    let mut image = None;
    let mut repetitions = 3;
    let mut output_tokens = 256;
    let mut verify_length = 7;
    let mut identity = None;
    let mut poc = false;
    let mut p4 = false;
    let mut diagnostic = false;
    let mut onboarding_native = false;
    let mut drift_graded = false;
    let mut reference_once = false;
    let mut performance_only = false;
    let mut fast_consumer_math = false;
    let mut producer_receipt_profile = ProducerReceiptProfile::Enrolled;
    let mut external_producer_receipt = None;
    let mut external_producer_receipt_dir = None;
    let mut delta_prefix_cut = None;
    let mut prefix_prompt_fixture = None;
    let mut local_only = false;
    let mut dry_run = false;
    let mut values = std::env::args().skip(1);
    while let Some(flag) = values.next() {
        let value = |values: &mut std::iter::Skip<std::env::Args>, flag: &str| {
            values
                .next()
                .ok_or_else(|| format!("{flag} requires a value"))
        };
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(value(&mut values, &flag)?)),
            "--prompt-token-fixture" => {
                prompt_fixture = Some(PathBuf::from(value(&mut values, &flag)?))
            }
            "--cluster-config" => cluster_config = Some(PathBuf::from(value(&mut values, &flag)?)),
            "--variant" => {
                variant = Some(match value(&mut values, &flag)?.as_str() {
                    "text" => Variant::Text,
                    "multimodal" => Variant::Multimodal,
                    "target-plus-dflash" => Variant::TargetPlusDflash,
                    other => return Err(format!("unknown remote variant {other}")),
                })
            }
            "--dflash" => dflash = Some(PathBuf::from(value(&mut values, &flag)?)),
            "--mmproj" => mmproj = Some(PathBuf::from(value(&mut values, &flag)?)),
            "--mtmd-bridge" => mtmd_bridge = Some(PathBuf::from(value(&mut values, &flag)?)),
            "--image" => image = Some(PathBuf::from(value(&mut values, &flag)?)),
            "--repetitions" => repetitions = parse_usize(&value(&mut values, &flag)?, &flag)?,
            "--output-tokens" => output_tokens = parse_usize(&value(&mut values, &flag)?, &flag)?,
            "--verify-length" => verify_length = parse_usize(&value(&mut values, &flag)?, &flag)?,
            "--identity" => identity = Some(value(&mut values, &flag)?),
            "--poc" => poc = true,
            "--p4" => p4 = true,
            "--diagnostic" => diagnostic = true,
            "--onboarding-native" => onboarding_native = true,
            "--drift-graded" => drift_graded = true,
            "--reference-once" => reference_once = true,
            "--performance-only" => performance_only = true,
            "--fast-consumer-math" => fast_consumer_math = true,
            "--pre-streaming-control" => {
                producer_receipt_profile = ProducerReceiptProfile::HistoricalPreStreamingControl
            }
            "--external-producer-receipt" => {
                external_producer_receipt = Some(PathBuf::from(value(&mut values, &flag)?))
            }
            "--external-producer-receipt-dir" => {
                external_producer_receipt_dir = Some(PathBuf::from(value(&mut values, &flag)?))
            }
            "--delta-prefix-cut" => {
                delta_prefix_cut = Some(parse_usize(&value(&mut values, &flag)?, &flag)? as u64)
            }
            "--prefix-prompt-fixture" => {
                prefix_prompt_fixture = Some(PathBuf::from(value(&mut values, &flag)?))
            }
            "--local-only" => local_only = true,
            "--dry-run" => dry_run = true,
            _ => return Err(format!("unknown argument {flag}")),
        }
    }
    Ok(Args {
        model: model.ok_or("--model is required")?,
        prompt_fixture: prompt_fixture.ok_or("--prompt-token-fixture is required")?,
        cluster_config: cluster_config.ok_or("--cluster-config is required")?,
        variant: variant.ok_or("--variant is required")?,
        dflash,
        mmproj,
        mtmd_bridge,
        image,
        repetitions,
        output_tokens,
        verify_length,
        identity: identity.ok_or("--identity is required")?,
        poc,
        p4,
        diagnostic,
        onboarding_native,
        drift_graded,
        reference_once,
        performance_only,
        fast_consumer_math,
        producer_receipt_profile,
        external_producer_receipt,
        external_producer_receipt_dir,
        delta_prefix_cut,
        prefix_prompt_fixture,
        local_only,
        dry_run,
    })
}

fn parse_usize(value: &str, flag: &str) -> Result<usize, String> {
    value
        .parse()
        .map_err(|_| format!("invalid {flag}: {value}"))
}

fn parse_tokens(bytes: &[u8]) -> Result<Vec<u32>, String> {
    std::str::from_utf8(bytes)
        .map_err(|error| error.to_string())?
        .split_ascii_whitespace()
        .map(|value| value.parse::<u32>().map_err(|error| error.to_string()))
        .collect()
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1).then_with(|| right.0.cmp(&left.0)))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn first_mismatch(left: &[u32], right: &[u32]) -> usize {
    left.iter()
        .zip(right)
        .position(|(left, right)| left != right)
        .unwrap_or(left.len().min(right.len()))
}

fn token_digest(tokens: &[u32]) -> String {
    let mut digest = Sha256::new();
    for token in tokens {
        digest.update(token.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn phase(message: &str) {
    eprintln!("muser-remote-qualify: phase={message}");
    let _ = std::io::stderr().flush();
}

fn dflash_context_digest(snapshot: &DFlashContextSnapshot) -> String {
    let mut digest = Sha256::new();
    for value in [
        snapshot.position,
        snapshot.sink_size,
        snapshot.window_size,
        snapshot.elements_per_token,
        snapshot.layers.len(),
    ] {
        digest.update(value.to_le_bytes());
    }
    for (key, value) in &snapshot.layers {
        for plane in [key, value] {
            digest.update(plane.len().to_le_bytes());
            for item in plane {
                digest.update(item.to_bits().to_le_bytes());
            }
        }
    }
    format!("{:x}", digest.finalize())
}

fn first_dflash_context_diff(
    local: &DFlashContextSnapshot,
    remote: &DFlashContextSnapshot,
) -> Result<Option<DFlashContextDiff>, String> {
    if local.position != remote.position
        || local.sink_size != remote.sink_size
        || local.window_size != remote.window_size
        || local.elements_per_token != remote.elements_per_token
        || local.layers.len() != remote.layers.len()
    {
        return Err(format!(
            "DFlash context geometry differs: local=({},{},{},{},{}) remote=({},{},{},{},{})",
            local.position,
            local.sink_size,
            local.window_size,
            local.elements_per_token,
            local.layers.len(),
            remote.position,
            remote.sink_size,
            remote.window_size,
            remote.elements_per_token,
            remote.layers.len(),
        ));
    }
    let width = local.elements_per_token;
    if width == 0 {
        return Err("DFlash context width is zero".into());
    }
    for (layer, ((local_key, local_value), (remote_key, remote_value))) in
        local.layers.iter().zip(&remote.layers).enumerate()
    {
        for (plane, local, remote) in [
            ("key", local_key, remote_key),
            ("value", local_value, remote_value),
        ] {
            if local.len() != remote.len() {
                return Err(format!(
                    "DFlash context layer {layer} {plane} length differs: local={} remote={}",
                    local.len(),
                    remote.len()
                ));
            }
            if let Some((index, (&left, &right))) = local
                .iter()
                .zip(remote)
                .enumerate()
                .find(|(_, (left, right))| left.to_bits() != right.to_bits())
            {
                return Ok(Some(DFlashContextDiff {
                    layer,
                    plane,
                    index,
                    token: index / width,
                    element: index % width,
                    local_bits: left.to_bits(),
                    remote_bits: right.to_bits(),
                }));
            }
        }
    }
    Ok(None)
}

fn validate_dflash_trace(stats: &DFlashSpecStats) -> Result<(), String> {
    if stats.accepted_prefix_counts.len() != stats.rounds
        || stats.accepted_prefix_counts.iter().sum::<usize>() != stats.accepted_draft_tokens
        || stats.accepted_prefix_trace.len() != stats.accepted_draft_tokens
    {
        return Err("DFlash assistant trace counters are internally inconsistent".into());
    }
    Ok(())
}

fn valid_p4_geometry(performance_only: bool, repetitions: usize, output_tokens: usize) -> bool {
    if performance_only {
        repetitions >= 3 && (1..=256).contains(&output_tokens)
    } else {
        repetitions == 5 && output_tokens == 256
    }
}

fn cv(samples: &[u64]) -> f64 {
    let mean = samples.iter().map(|&value| value as f64).sum::<f64>() / samples.len() as f64;
    let variance = samples
        .iter()
        .map(|&value| {
            let delta = value as f64 - mean;
            delta * delta
        })
        .sum::<f64>()
        / samples.len() as f64;
    if mean == 0.0 {
        0.0
    } else {
        variance.sqrt() / mean
    }
}

fn nanos(value: u128) -> u64 {
    value.min(u64::MAX as u128) as u64
}

fn median_f64(samples: &[f64]) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    Some(sorted[sorted.len() / 2])
}

fn cv_f64(samples: &[f64]) -> f64 {
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance = samples
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / samples.len() as f64;
    if mean == 0.0 {
        0.0
    } else {
        variance.sqrt() / mean
    }
}

fn require_strict_cross_vendor_qk(value: Option<&str>) -> Result<(), String> {
    if value == Some("1") {
        return Ok(());
    }
    Err(
        "remote CUDA cache qualification requires MUSER_CROSS_VENDOR_QK=1 so the Metal consumer uses the pinned cross-vendor math route"
            .into(),
    )
}

#[cfg(test)]
mod tests {
    use super::{
        cache_row_digest, dflash_context_digest, fast_performance_schedule,
        fast_performance_stable, first_dflash_context_diff, full_logit_divergence_diagnostic_value,
        parse_external_producer_phase, require_strict_cross_vendor_qk, retained_logit_diff,
        retained_top_logits, session_cache_digest, token_digest, valid_p4_geometry,
        DFlashContextSnapshot, Generation, ProducerReceiptProfile,
    };
    use muser_cluster::muse_sink::ComponentInstallEvidence;
    use muser_cluster::phase::HandoffPhaseNanos;
    use muser_cluster::receiver::RemoteReceiveReceipt;
    use muser_engine::cache::{CachePlaneSnapshot, PlaneEncoding, SessionCacheSnapshot};
    use std::sync::Arc;

    #[test]
    fn remote_qualification_math_route_requires_exact_opt_in() {
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
            assert!(error.contains("requires MUSER_CROSS_VENDOR_QK=1"));
        }
    }

    #[test]
    fn first_divergence_top_logits_are_ranked_and_bit_stamped() {
        let generation = Generation {
            tokens: vec![7, 8],
            full_logit_digest: String::new(),
            retained_logits: Some(vec![0.5, 1.0, -1.0, 2.0, 4.0, 3.0]),
            cache_snapshot: None,
            post_decode_cache_snapshot: None,
            first_64_decode_ns: 0,
        };
        let top = retained_top_logits(&generation, 1, 2).unwrap();
        assert_eq!(top[0]["token"], 1);
        assert_eq!(top[0]["value"], 4.0);
        assert_eq!(top[0]["bits"], "40800000");
        assert_eq!(top[1]["token"], 2);
        assert_eq!(top[1]["value"], 3.0);
    }

    #[test]
    fn first_divergence_cache_digest_names_the_exact_logical_row() {
        let snapshot = SessionCacheSnapshot {
            position: 3,
            tokens: Arc::from([1_u32, 2, 3]),
            elements_per_token: 2,
            layers: Arc::from([CachePlaneSnapshot {
                layer: 0,
                logical_start: 1,
                logical_count: 2,
                encoding: PlaneEncoding::F16Le,
                key: Arc::from([1_u8, 0, 2, 0, 3, 0, 4, 0]),
                value: Arc::from([5_u8, 0, 6, 0, 7, 0, 8, 0]),
            }]),
        };
        let row_one = cache_row_digest(&snapshot, 1).unwrap();
        let row_two = cache_row_digest(&snapshot, 2).unwrap();
        assert_ne!(row_one, row_two);
        assert!(cache_row_digest(&snapshot, 0)
            .unwrap_err()
            .contains("absent from layer 0"));

        let mut changed = snapshot.clone();
        changed.tokens = Arc::from([1_u32, 2, 4]);
        assert_ne!(
            session_cache_digest(&snapshot),
            session_cache_digest(&changed)
        );
    }

    #[test]
    fn full_logit_divergence_record_is_bounded_and_does_not_change_the_gate() {
        let prefix = SessionCacheSnapshot {
            position: 3,
            tokens: Arc::from([1_u32, 2, 3]),
            elements_per_token: 2,
            layers: Arc::from([CachePlaneSnapshot {
                layer: 0,
                logical_start: 1,
                logical_count: 2,
                encoding: PlaneEncoding::F16Le,
                key: Arc::from([1_u8, 0, 2, 0, 3, 0, 4, 0]),
                value: Arc::from([5_u8, 0, 6, 0, 7, 0, 8, 0]),
            }]),
        };
        let local_post = SessionCacheSnapshot {
            position: 5,
            tokens: Arc::from([1_u32, 2, 3, 7, 8]),
            elements_per_token: 2,
            layers: Arc::from([CachePlaneSnapshot {
                layer: 0,
                logical_start: 2,
                logical_count: 3,
                encoding: PlaneEncoding::F16Le,
                key: Arc::from([1_u8, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0]),
                value: Arc::from([7_u8, 0, 8, 0, 9, 0, 10, 0, 11, 0, 12, 0]),
            }]),
        };
        let mut remote_post = local_post.clone();
        remote_post.layers = Arc::from([CachePlaneSnapshot {
            layer: 0,
            logical_start: 2,
            logical_count: 3,
            encoding: PlaneEncoding::F16Le,
            key: Arc::from([1_u8, 0, 2, 0, 13, 0, 4, 0, 5, 0, 6, 0]),
            value: Arc::from([7_u8, 0, 8, 0, 9, 0, 10, 0, 11, 0, 12, 0]),
        }]);
        let local = Generation {
            tokens: vec![7, 8],
            full_logit_digest: "local".into(),
            retained_logits: Some(vec![1.0, 0.0, -1.0, 2.0, 1.0, 0.0]),
            cache_snapshot: Some(prefix.clone()),
            post_decode_cache_snapshot: Some(local_post),
            first_64_decode_ns: 0,
        };
        let remote = Generation {
            tokens: vec![7, 8],
            full_logit_digest: "remote".into(),
            retained_logits: Some(vec![1.0, 0.0, -1.0, 2.0, 1.0, 0.25]),
            cache_snapshot: Some(prefix),
            post_decode_cache_snapshot: Some(remote_post),
            first_64_decode_ns: 0,
        };
        let difference = retained_logit_diff(&local, &remote).unwrap();
        let record =
            full_logit_divergence_diagnostic_value(&local, &remote, &difference, 3).unwrap();
        assert_eq!(record["schema"], "muser.remote-full-logit-divergence.v1");
        assert_eq!(record["kind"], "diagnostic-only");
        assert_eq!(record["verdict_gate"], "target-full-logit-digest-unchanged");
        assert_eq!(record["target_tokens_exact"], true);
        assert_eq!(record["first_row"], 1);
        assert_eq!(record["first_logit"], 2);
        assert_eq!(record["mismatched_logits"], 1);
        assert_eq!(record["prefix_cache"]["exact"], true);
        assert_eq!(record["prefix_last_row_layers"]["exact"], true);
        assert_eq!(record["causative_kv_row"]["logical_row"], 3);
        assert_eq!(record["causative_kv_row"]["exact"], false);
        assert_eq!(
            record["causative_kv_row"]["layers"]["first_mismatch"]["layer"],
            0
        );
        assert_eq!(
            record["causative_kv_row"]["layers"]["first_mismatch"]["key"]["first_mismatch"]
                ["element"],
            0
        );
        assert_eq!(
            record["causative_kv_row"]["layers"]["first_mismatch"]["value"]["exact"],
            true
        );
        assert_eq!(record["seal_eligible"], false);
    }

    #[test]
    fn performance_warmup_is_p4_only() {
        assert_eq!(
            fast_performance_schedule(true, 5),
            vec![
                (0, true),
                (1, false),
                (2, false),
                (3, false),
                (4, false),
                (5, false),
            ]
        );
        assert_eq!(fast_performance_schedule(false, 1), vec![(0, false)]);
    }

    #[test]
    fn p4_performance_geometry_does_not_weaken_full_qualification() {
        assert!(valid_p4_geometry(true, 3, 256));
        assert!(valid_p4_geometry(true, 3, 48));
        assert!(valid_p4_geometry(true, 5, 256));
        assert!(!valid_p4_geometry(true, 2, 256));
        assert!(!valid_p4_geometry(true, 3, 0));
        assert!(!valid_p4_geometry(true, 3, 257));

        assert!(valid_p4_geometry(false, 5, 256));
        assert!(!valid_p4_geometry(false, 3, 256));
        assert!(!valid_p4_geometry(false, 5, 48));
    }

    #[test]
    fn fast_performance_ttft_target_applies_only_to_2048_class() {
        assert!(!fast_performance_stable(2048, 0.01, 3.0, 4_000_000_001));
        assert!(fast_performance_stable(8192, 0.01, 3.0, 4_000_000_001));
        assert!(!fast_performance_stable(8192, 0.021, 3.0, 1));
        assert!(!fast_performance_stable(8192, 0.01, 2.999, 1));
    }

    #[test]
    fn dflash_context_evidence_finds_the_first_bit_mismatch() {
        let local = DFlashContextSnapshot {
            position: 2,
            sink_size: 1,
            window_size: 1,
            elements_per_token: 2,
            layers: vec![(vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0])],
        };
        let mut remote = local.clone();
        remote.layers[0].0[3] = f32::from_bits(local.layers[0].0[3].to_bits() + 1);

        assert_ne!(
            dflash_context_digest(&local),
            dflash_context_digest(&remote)
        );
        let difference = first_dflash_context_diff(&local, &remote).unwrap().unwrap();
        assert_eq!(difference.layer, 0);
        assert_eq!(difference.plane, "key");
        assert_eq!(difference.index, 3);
        assert_eq!(difference.token, 1);
        assert_eq!(difference.element, 1);
    }

    #[test]
    fn dflash_context_evidence_rejects_geometry_drift() {
        let local = DFlashContextSnapshot {
            position: 1,
            sink_size: 1,
            window_size: 1,
            elements_per_token: 1,
            layers: vec![(vec![1.0], vec![2.0])],
        };
        let mut remote = local.clone();
        remote.position = 2;
        assert!(first_dflash_context_diff(&local, &remote)
            .unwrap_err()
            .contains("geometry differs"));
    }

    #[test]
    fn external_producer_receipt_is_bound_to_prompt_and_install() {
        let tokens = [1_u32, 2, 3];
        let receiver = RemoteReceiveReceipt {
            transfer_id: "request-7".into(),
            generation: 7,
            installed_segments: 4,
            installed_bytes: 1024,
            control_ns: 1,
            accept_ns: 1,
            transfer_commit_ns: 1,
            total_ns: 1,
            producer: None,
            components: ComponentInstallEvidence::default(),
            phases: HandoffPhaseNanos::default(),
        };
        let receipt = serde_json::json!({
            "schema": "muser.spark-nvfp4-prefill-client.v1",
            "token_count": tokens.len(),
            "response": {
                "status": "ok",
                "request_id": "request-7",
                "prompt_token_count": tokens.len(),
                "producer_receipt": {
                    "schema": "muser.spark-nvfp4-prefill.v1",
                    "prompt_token_count": tokens.len(),
                    "token_ids_sha256": token_digest(&tokens),
                    "phase_ns": {
                        "connector_total": 900,
                        "d2h_complete_offset": 600
                    },
                    "handoff": {
                        "ack": true,
                        "transfer_id": "request-7",
                        "generation": 7,
                        "segments": 4,
                        "payload_bytes": 1024,
                        "payload_wire_ns": 200,
                        "payload_wire_source": "linux-tcp-info-busy-time-v1",
                        "payload_pacing_bps": 4_000_000_000_u64,
                        "transfer_start_unix_ns": 1000,
                        "first_segment_sent_unix_ns": 1700,
                        "transfer_acked_unix_ns": 2000
                    }
                }
            }
        });
        let bytes = serde_json::to_vec(&receipt).unwrap();
        assert!(parse_external_producer_phase(
            &bytes,
            &receiver,
            &tokens,
            ProducerReceiptProfile::Enrolled,
        )
        .unwrap_err()
        .contains("enrolled producer requires a v2 native"));

        let mut native = receipt.clone();
        native["response"]["producer_receipt"]["schema"] =
            serde_json::json!("muser.spark-nvfp4-prefill.v2");
        native["response"]["producer_receipt"]["producer_mode"] = serde_json::json!("native");
        // The streaming seam sends its first segment before D2H completes.
        native["response"]["producer_receipt"]["phase_ns"]["first_segment_sent_offset"] =
            serde_json::json!(300);
        let phase = parse_external_producer_phase(
            &serde_json::to_vec(&native).unwrap(),
            &receiver,
            &tokens,
            ProducerReceiptProfile::Enrolled,
        )
        .unwrap();
        assert_eq!(phase.prefill_start_unix_ns, 1000);
        assert_eq!(phase.prefill_end_unix_ns, 1600);
        assert_eq!(phase.state_saved_unix_ns, 1600);
        assert_eq!(phase.payload_bytes, 1024);
        // A v2 receipt that sends late — or omits the streaming evidence —
        // is rejected.
        let mut late = native.clone();
        late["response"]["producer_receipt"]["phase_ns"]["first_segment_sent_offset"] =
            serde_json::json!(700);
        assert!(parse_external_producer_phase(
            &serde_json::to_vec(&late).unwrap(),
            &receiver,
            &tokens,
            ProducerReceiptProfile::Enrolled,
        )
        .unwrap_err()
        .contains("first segment after D2H completion"));
        let mut missing = native.clone();
        missing["response"]["producer_receipt"]["phase_ns"]
            .as_object_mut()
            .unwrap()
            .remove("first_segment_sent_offset");
        assert!(parse_external_producer_phase(
            &serde_json::to_vec(&missing).unwrap(),
            &receiver,
            &tokens,
            ProducerReceiptProfile::Enrolled,
        )
        .unwrap_err()
        .contains("omits first_segment_sent_offset"));

        // The historical pre-streaming control is admitted only when the
        // caller explicitly selects that profile. It must prove the opposite
        // schedule: no streaming offset, and D2H complete before the first
        // segment reaches the wire.
        let pre_streaming = parse_external_producer_phase(
            &serde_json::to_vec(&missing).unwrap(),
            &receiver,
            &tokens,
            ProducerReceiptProfile::HistoricalPreStreamingControl,
        )
        .unwrap();
        assert_eq!(pre_streaming.prefill_end_unix_ns, 1600);
        assert_eq!(pre_streaming.state_saved_unix_ns, 1700);
        assert!(parse_external_producer_phase(
            &serde_json::to_vec(&native).unwrap(),
            &receiver,
            &tokens,
            ProducerReceiptProfile::HistoricalPreStreamingControl,
        )
        .unwrap_err()
        .contains("unexpectedly includes streaming"));
        let mut overlapping_control = missing.clone();
        overlapping_control["response"]["producer_receipt"]["handoff"]
            ["first_segment_sent_unix_ns"] = serde_json::json!(1500);
        assert!(parse_external_producer_phase(
            &serde_json::to_vec(&overlapping_control).unwrap(),
            &receiver,
            &tokens,
            ProducerReceiptProfile::HistoricalPreStreamingControl,
        )
        .unwrap_err()
        .contains("before D2H completion"));
        assert!(parse_external_producer_phase(
            &bytes,
            &receiver,
            &tokens,
            ProducerReceiptProfile::HistoricalPreStreamingControl,
        )
        .unwrap_err()
        .contains("requires a v2 native"));

        native["response"]["producer_receipt"]["producer_mode"] = serde_json::Value::Null;
        assert!(parse_external_producer_phase(
            &serde_json::to_vec(&native).unwrap(),
            &receiver,
            &tokens,
            ProducerReceiptProfile::Enrolled,
        )
        .unwrap_err()
        .contains("invalid schema or status"));

        let mut wrong = receipt;
        wrong["response"]["producer_receipt"]["handoff"]["generation"] = serde_json::json!(8);
        assert!(parse_external_producer_phase(
            &serde_json::to_vec(&wrong).unwrap(),
            &receiver,
            &tokens,
            ProducerReceiptProfile::Enrolled,
        )
        .unwrap_err()
        .contains("does not bind the installed transfer"));
    }
}
