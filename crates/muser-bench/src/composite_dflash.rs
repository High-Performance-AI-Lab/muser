#![recursion_limit = "256"]

//! End-to-end Mac DFlash -> persistent GX Dudeman verifier qualification.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use half::f16;
use kvpack_handoff::{canonical_json, MacKey};
use muser_cluster::security::load_mac_key;
use muser_engine::dflash::{
    AuthenticatedDFlashTargetDecision, DFlashAssistant, ProvisionalDFlashResolution,
};
use muser_engine::{Model, ModelConfig, SessionConfig};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const REQUEST_SCHEMA: &str = "muser.composite-verifier-rpc-request.v2";
const PROVISIONAL_SCHEMA: &str = "muser.composite-verifier-rpc-provisional.v1";
const FINAL_SCHEMA: &str = "muser.composite-verifier-rpc-final.v2";
const REQUEST_DOMAIN: &[u8] = b"muser-composite-verifier-rpc-request-v2";
const PROVISIONAL_DOMAIN: &[u8] = b"muser-composite-verifier-rpc-provisional-v1";
const FINAL_DOMAIN: &[u8] = b"muser-composite-verifier-rpc-final-v2";
const PROVISIONAL_COMMITMENT_DOMAIN: &str = "muser-composite-verifier-provisional-commitment-v1";
const VERIFIER_IDENTITY_DOMAIN: &str = "muser-composite-verifier-identity-v1";
const TARGET_LAYERS: usize = 5;
const HIDDEN_SIZE: usize = 6_656;
const ROW_FLOATS: usize = TARGET_LAYERS * HIDDEN_SIZE;
const F16_ROW_BYTES: usize = ROW_FLOATS * 2;
const MAX_HEADER_BYTES: usize = 1 << 20;
const MAX_CANDIDATES: usize = 65;
const MAX_PAYLOAD_BYTES: usize = MAX_CANDIDATES * F16_ROW_BYTES;
const PRODUCT_BAR_TPS: f64 = 107.9;
const VERIFIER_CACHE_BLOCK_SIZE: usize = 16;

struct Args {
    model: PathBuf,
    dflash: PathBuf,
    prompt_fixture: PathBuf,
    prompt_tokens: usize,
    initial_hidden: PathBuf,
    initial_hidden_sha256: String,
    expected_verifier_identity_sha256: String,
    expected_model_sha256: String,
    expected_dflash_sha256: String,
    hmac_key_file: PathBuf,
    ggml_metallib: PathBuf,
    server_host: String,
    server_port: u16,
    session_id: String,
    output_tokens: usize,
    verify_length: usize,
    mirror_overlap: bool,
    timeout: Duration,
    output: PathBuf,
    identity: String,
}

#[derive(Serialize)]
struct RequestCore<'a> {
    schema: &'static str,
    session_id: &'a str,
    request_id: &'a str,
    command: &'a str,
    base_head_sha256: &'a str,
    candidates: &'a [u32],
    sent_unix_ms: u64,
}

#[derive(Serialize)]
struct RequestEnvelope<'a> {
    core: &'a RequestCore<'a>,
    hmac_sha256: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FinalEnvelope {
    core: FinalCore,
    hmac_sha256: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FinalCore {
    schema: String,
    accepted_drafts: usize,
    base_head_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    candidate_count: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    candidate_tokens_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    capture_timing: Option<CaptureTiming>,
    committed_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    committed_payload_bytes: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    committed_payload_sha256: Option<String>,
    committed_tokens: Vec<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    frontier_out: Option<u32>,
    new_head_sha256: String,
    num_cached_tokens: Option<usize>,
    output_height: usize,
    payload_bytes: usize,
    payload_dtype: String,
    payload_row_bytes: usize,
    payload_rows: usize,
    payload_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    provisional_sha256: Option<String>,
    replayed: bool,
    request_id: String,
    session_id: String,
    status: String,
    target_tokens: Vec<u32>,
    transcript_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    verifier_identity: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    verifier_identity_sha256: Option<String>,
    wall_ns: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LayerTiming {
    arrival_offset_ns: u64,
    copy_is_async: bool,
    copy_enqueued_offset_ns: u64,
    layer: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CaptureTiming {
    capture_started_ns: u64,
    finish_completed_offset_ns: u64,
    finish_started_offset_ns: u64,
    generate_finished_offset_ns: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    host_ready_callback_completed_offset_ns: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    host_ready_callback_started_offset_ns: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    host_ready_offset_ns: Option<u64>,
    last_layer_arrival_to_generate_finish_ns: u64,
    last_layer_copy_enqueue_to_generate_finish_ns: u64,
    layer_timings: Vec<LayerTiming>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    payload_ready_offset_ns: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    provisional_send_completed_offset_ns: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    provisional_send_started_offset_ns: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProvisionalEnvelope {
    core: ProvisionalCore,
    hmac_sha256: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProvisionalCore {
    schema: String,
    base_head_sha256: String,
    candidate_count: usize,
    candidate_tokens_sha256: String,
    frame_kind: String,
    host_ready_offset_ns: u64,
    payload_bytes: usize,
    payload_dtype: String,
    payload_ready_offset_ns: u64,
    payload_row_bytes: usize,
    payload_rows: usize,
    payload_sha256: String,
    provisional_sha256: String,
    replayed: bool,
    request_id: String,
    session_id: String,
}

#[derive(Serialize)]
struct RoundReceipt {
    round: usize,
    request_id: String,
    candidate_count: usize,
    drafted: usize,
    accepted_drafts: usize,
    committed_count: usize,
    apc_parent_lag: usize,
    initial_draft_ns: u64,
    request_to_provisional_ns: u64,
    provisional_decode_ns: u64,
    provisional_prepare_ns: u64,
    provisional_draft_ns: u64,
    final_wait_ns: u64,
    post_final_finish_ns: u64,
    mirror_attempted: bool,
    mirror_committed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    mirror_predicted_frontier: Option<u32>,
    verifier_wall_ns: u64,
    rpc_ns: u64,
    response_payload_bytes: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    capture_timing: Option<CaptureTiming>,
    base_head_sha256: String,
    new_head_sha256: String,
    frontier_out: u32,
}

struct RoundExpectation<'a> {
    session_id: &'a str,
    base_head: &'a str,
    candidates: &'a [u32],
    prior_height: usize,
    prior_transcript_len: usize,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("muser-composite-dflash-qualify: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    validate_args(&args)?;
    if let Some(existing) = std::env::var_os("MUSER_GGML_METALLIB") {
        if Path::new(&existing) != args.ggml_metallib {
            return Err("MUSER_GGML_METALLIB differs from --ggml-metallib".into());
        }
    } else {
        std::env::set_var("MUSER_GGML_METALLIB", &args.ggml_metallib);
    }
    let ggml_metallib_sha256 = sha256_file(&args.ggml_metallib)?;
    let model_sha256 = sha256_file(&args.model)?;
    let dflash_sha256 = sha256_file(&args.dflash)?;
    if model_sha256 != args.expected_model_sha256 {
        return Err("local target model differs from --expected-model-sha256".into());
    }
    if dflash_sha256 != args.expected_dflash_sha256 {
        return Err("local DFlash model differs from --expected-dflash-sha256".into());
    }
    let key = load_mac_key(&args.hmac_key_file).map_err(|error| error.to_string())?;
    let prompt = read_tokens(&args.prompt_fixture, args.prompt_tokens)?;
    let (mut initial_hidden, initial_hidden_digest) = read_hidden(
        &args.initial_hidden,
        args.prompt_tokens
            .checked_mul(ROW_FLOATS)
            .ok_or("initial hidden element count overflow")?,
    )?;
    if initial_hidden_digest != args.initial_hidden_sha256 {
        return Err(format!(
            "initial hidden digest {initial_hidden_digest} differs from {}",
            args.initial_hidden_sha256
        ));
    }

    let model = Model::load(ModelConfig::new(&args.model)).map_err(|error| error.to_string())?;
    let mut assistant =
        DFlashAssistant::load_metal(&args.dflash, &model).map_err(|error| error.to_string())?;

    // This session owns only the resident Metal embedding/LM-head machinery.
    // Its token/KV state is intentionally never populated or consulted.
    let mut projection_session = model
        .new_metal_session(SessionConfig { max_context: 1 })
        .map_err(|error| error.to_string())?;

    // Warm every DFlash/LM-head kernel before publishing a remote session,
    // then discard the one-row context mutation. Product throughput evidence
    // must not charge one-time shader/page warmup to its first decode round.
    let warmup_started = Instant::now();
    assistant
        .draft_greedy_with_session_projection(
            &model,
            &mut projection_session,
            0,
            &initial_hidden[..ROW_FLOATS],
            1,
            args.verify_length,
        )
        .map_err(|error| error.to_string())?;
    assistant.reset();
    let dflash_warmup_ns = elapsed_ns(warmup_started);

    let mut stream = connect(&args)?;
    let open_started = Instant::now();
    let open_id = "open-000000";
    let (opened, hidden_payload) = transact_final(
        &mut stream,
        &key,
        &args.session_id,
        open_id,
        "open",
        &"0".repeat(64),
        &[],
    )?;
    let open_rpc_ns = elapsed_ns(open_started);
    require_status(&opened, open_id, "opened")?;
    let verifier_identity = opened
        .verifier_identity
        .as_ref()
        .ok_or("open response omitted verifier identity")?;
    let verifier_identity_digest = verifier_identity_sha256(verifier_identity)?;
    if opened.base_head_sha256 != "0".repeat(64)
        || opened.session_id != args.session_id
        || opened.verifier_identity_sha256.as_deref() != Some(verifier_identity_digest.as_str())
        || verifier_identity_digest != args.expected_verifier_identity_sha256
        || opened.committed_count != 0
        || opened.output_height != 0
        || opened.transcript_sha256 != token_digest(&prompt)
        || validate_final_payload_abi(&opened, &hidden_payload, 1).is_err()
    {
        return Err("open response does not bind the composite genesis".into());
    }
    let mut frontier = opened
        .frontier_out
        .ok_or("open response omitted the carried frontier")?;
    let mut head = opened.new_head_sha256.clone();

    // Reconstruct the exact DFlash prompt context only after the potentially
    // long remote composite-open call. Besides keeping setup outside decode
    // timing, this leaves Metal resident immediately before the first round
    // instead of paying an avoidable post-idle wakeup on the critical path.
    let prefix_rows = args.prompt_tokens - 1;
    let prime_started = Instant::now();
    assistant
        .prime_target_context(
            &model,
            &initial_hidden[..prefix_rows * ROW_FLOATS],
            prefix_rows,
        )
        .map_err(|error| error.to_string())?;
    let prime_ns = elapsed_ns(prime_started);
    initial_hidden.clear();
    initial_hidden.shrink_to_fit();

    let initial_hidden_dtype = opened.payload_dtype.clone();
    let mut initial_hidden_payload = Some(hidden_payload);
    let mut prefetched_drafts: Option<Vec<u32>> = None;

    publish_profile_ready_marker()?;

    let decode_started = Instant::now();
    let mut generated = Vec::with_capacity(args.output_tokens);
    let mut transcript = prompt;
    let mut output_height = 0usize;
    let mut rounds = Vec::new();
    let mut drafted_total = 0usize;
    let mut accepted_total = 0usize;
    let mut dflash_total_ns = 0u64;
    let mut verifier_wall_total_ns = 0u64;
    let mut rpc_total_ns = 0u64;
    let mut request_to_provisional_total_ns = 0u64;
    let mut provisional_decode_total_ns = 0u64;
    let mut provisional_prepare_total_ns = 0u64;
    let mut provisional_draft_total_ns = 0u64;
    let mut final_wait_total_ns = 0u64;
    let mut post_final_finish_total_ns = 0u64;
    let mut mirror_overlap_attempts = 0usize;
    let mut mirror_overlap_commits = 0usize;
    let mut mirror_overlap_rollbacks = 0usize;
    let mut mirror_overlap_enabled = args.mirror_overlap;
    let mut payload_total_bytes = initial_hidden_payload.as_ref().map_or(0, Vec::len);

    while generated.len() < args.output_tokens {
        let remaining = args.output_tokens - generated.len();
        let (drafts, initial_draft_ns) = if remaining > 1 {
            match prefetched_drafts.take() {
                Some(drafts) => (drafts, 0),
                None => {
                    let payload = initial_hidden_payload
                        .take()
                        .ok_or("missing both prefetched drafts and initial hidden row")?;
                    let hidden = decode_hidden(&payload, &initial_hidden_dtype)?;
                    let draft_started = Instant::now();
                    let drafts = assistant
                        .draft_greedy_with_session_projection(
                            &model,
                            &mut projection_session,
                            frontier,
                            &hidden,
                            1,
                            args.verify_length,
                        )
                        .map_err(|error| error.to_string())?;
                    (drafts, elapsed_ns(draft_started))
                }
            }
        } else {
            (Vec::new(), 0)
        };
        let mirror_verified_drafts = mirror_verified_drafts(
            mirror_overlap_enabled,
            args.verify_length,
            drafts.len(),
            remaining,
        );
        let mirror_attempted = mirror_verified_drafts.is_some();
        let used_drafts = if let Some(verified) = mirror_verified_drafts {
            verified
        } else {
            drafts.len().min(remaining.saturating_sub(1))
        };
        let mirror_predicted_frontier =
            held_back_mirror_frontier(mirror_attempted, &drafts, used_drafts)?;
        let mut candidates = Vec::with_capacity(1 + used_drafts);
        candidates.push(frontier);
        candidates.extend_from_slice(&drafts[..used_drafts]);
        let request_id = format!("round-{:06}", rounds.len());
        let rpc_started = Instant::now();
        send_request(
            &mut stream,
            &key,
            &args.session_id,
            &request_id,
            "verify",
            &head,
            &candidates,
        )?;
        let provisional_receive_started = Instant::now();
        let (provisional, provisional_payload) = receive_provisional(&mut stream, &key)?;
        let request_to_provisional_ns = elapsed_ns(provisional_receive_started);
        validate_provisional(
            &provisional,
            &provisional_payload,
            &args.session_id,
            &request_id,
            &head,
            &candidates,
        )?;
        let provisional_decode_started = Instant::now();
        let provisional_hidden = decode_hidden(&provisional_payload, &provisional.payload_dtype)?;
        let provisional_decode_ns = elapsed_ns(provisional_decode_started);
        let provisional_prepare_started = Instant::now();
        let prepared = assistant
            .prepare_target_context_split(&provisional_hidden, candidates.len())
            .map_err(|error| error.to_string())?;
        let provisional_prepare_ns = elapsed_ns(provisional_prepare_started);
        let provisional_draft_started = Instant::now();
        let mut prepared = Some(prepared);
        let mut provisional_context = if let Some(predicted_frontier) = mirror_predicted_frontier {
            mirror_overlap_attempts += 1;
            Some(
                assistant
                    .finish_prepared_target_context_provisionally_greedy(
                        &model,
                        &mut projection_session,
                        prepared.take().expect("prepared context is present"),
                        predicted_frontier,
                        args.verify_length,
                    )
                    .map_err(|error| error.to_string())?,
            )
        } else {
            None
        };
        let provisional_draft_ns = elapsed_ns(provisional_draft_started);
        let final_wait_started = Instant::now();
        let final_frame = receive_final(&mut stream, &key);
        let final_wait_ns = elapsed_ns(final_wait_started);
        let (response, final_payload) = match final_frame {
            Ok(frame) => frame,
            Err(error) => {
                if let Some(provisional) = provisional_context.take() {
                    provisional.rollback().map_err(|rollback| {
                        format!("{error}; Mirror rollback failed: {rollback}")
                    })?;
                    return Err(error);
                }
                drop(provisional_context);
                if let Some(prepared) = prepared.take() {
                    assistant
                        .discard_prepared_target_context(prepared)
                        .map_err(|discard| {
                            format!("{error}; prepared discard failed: {discard}")
                        })?;
                }
                return Err(error);
            }
        };
        let rpc_ns = elapsed_ns(rpc_started);
        let validated = (|| {
            require_status(&response, &request_id, "verified")?;
            let apc_parent_lag = validate_round(
                &response,
                &final_payload,
                &provisional,
                &provisional_payload,
                &RoundExpectation {
                    session_id: &args.session_id,
                    base_head: &head,
                    candidates: &candidates,
                    prior_height: output_height,
                    prior_transcript_len: transcript.len(),
                },
            )?;
            let next_frontier = response
                .frontier_out
                .ok_or("verified response omitted the carried frontier")?;
            let mut next_transcript = transcript.clone();
            next_transcript.extend_from_slice(&response.committed_tokens);
            if response.transcript_sha256 != token_digest(&next_transcript) {
                return Err("verified response transcript digest differs".into());
            }
            let hidden_sha256 = response
                .committed_payload_sha256
                .as_deref()
                .ok_or("verified response omitted committed hidden digest")?;
            let expected_head = transition_head(
                &head,
                &response.committed_tokens,
                next_frontier,
                response.output_height,
                hidden_sha256,
            )?;
            if response.new_head_sha256 != expected_head {
                return Err("verified response head transition differs".into());
            }
            Ok((next_frontier, next_transcript, apc_parent_lag))
        })();
        let (next_frontier, next_transcript, apc_parent_lag) = match validated {
            Ok(value) => value,
            Err(error) => {
                if let Some(provisional) = provisional_context.take() {
                    provisional.rollback().map_err(|rollback| {
                        format!("{error}; Mirror rollback failed: {rollback}")
                    })?;
                    return Err(error);
                }
                drop(provisional_context);
                if let Some(prepared) = prepared.take() {
                    assistant
                        .discard_prepared_target_context(prepared)
                        .map_err(|discard| {
                            format!("{error}; prepared discard failed: {discard}")
                        })?;
                }
                return Err(error);
            }
        };
        transcript = next_transcript;
        generated.extend_from_slice(&response.committed_tokens);
        let post_final_finish_started = Instant::now();
        let needs_next_draft = args.output_tokens.saturating_sub(generated.len()) > 1;
        let mirror_prefetch = if let Some(provisional) = provisional_context.take() {
            let decision = AuthenticatedDFlashTargetDecision::from_authenticated_fields(
                next_frontier,
                response.committed_count,
            );
            match provisional
                .resolve(decision)
                .map_err(|error| error.to_string())?
            {
                ProvisionalDFlashResolution::Committed(drafts) => Some(drafts),
                ProvisionalDFlashResolution::RolledBack => None,
            }
        } else {
            None
        };
        drop(provisional_context);
        let mirror_committed = mirror_prefetch.is_some();
        if mirror_attempted {
            if mirror_committed {
                mirror_overlap_commits += 1;
            } else {
                mirror_overlap_rollbacks += 1;
                mirror_overlap_enabled = false;
            }
        }
        if needs_next_draft {
            prefetched_drafts = match mirror_prefetch {
                Some(drafts) => Some(drafts),
                None => {
                    let prepared = match prepared.take() {
                        Some(prepared) => prepared,
                        None => assistant
                            .prepare_target_context_split(&provisional_hidden, candidates.len())
                            .map_err(|error| error.to_string())?,
                    };
                    Some(
                        assistant
                            .finish_prepared_target_context_greedy(
                                &model,
                                &mut projection_session,
                                prepared,
                                next_frontier,
                                response.committed_count,
                                args.verify_length,
                            )
                            .map_err(|error| error.to_string())?,
                    )
                }
            };
        } else if let Some(prepared) = prepared.take() {
            assistant
                .discard_prepared_target_context(prepared)
                .map_err(|error| error.to_string())?;
        }
        let post_final_finish_ns = elapsed_ns(post_final_finish_started);
        drafted_total += used_drafts;
        accepted_total += response.accepted_drafts;
        dflash_total_ns = dflash_total_ns
            .saturating_add(initial_draft_ns)
            .saturating_add(provisional_prepare_ns)
            .saturating_add(provisional_draft_ns)
            .saturating_add(post_final_finish_ns);
        verifier_wall_total_ns = verifier_wall_total_ns.saturating_add(response.wall_ns);
        rpc_total_ns = rpc_total_ns.saturating_add(rpc_ns);
        request_to_provisional_total_ns =
            request_to_provisional_total_ns.saturating_add(request_to_provisional_ns);
        provisional_decode_total_ns =
            provisional_decode_total_ns.saturating_add(provisional_decode_ns);
        provisional_prepare_total_ns =
            provisional_prepare_total_ns.saturating_add(provisional_prepare_ns);
        provisional_draft_total_ns =
            provisional_draft_total_ns.saturating_add(provisional_draft_ns);
        final_wait_total_ns = final_wait_total_ns.saturating_add(final_wait_ns);
        post_final_finish_total_ns =
            post_final_finish_total_ns.saturating_add(post_final_finish_ns);
        payload_total_bytes = payload_total_bytes.saturating_add(provisional_payload.len());
        rounds.push(RoundReceipt {
            round: rounds.len(),
            request_id,
            candidate_count: candidates.len(),
            drafted: used_drafts,
            accepted_drafts: response.accepted_drafts,
            committed_count: response.committed_count,
            apc_parent_lag,
            initial_draft_ns,
            request_to_provisional_ns,
            provisional_decode_ns,
            provisional_prepare_ns,
            provisional_draft_ns,
            final_wait_ns,
            post_final_finish_ns,
            mirror_attempted,
            mirror_committed,
            mirror_predicted_frontier,
            verifier_wall_ns: response.wall_ns,
            rpc_ns,
            response_payload_bytes: provisional_payload.len() + final_payload.len(),
            capture_timing: response.capture_timing.clone(),
            base_head_sha256: head.clone(),
            new_head_sha256: response.new_head_sha256.clone(),
            frontier_out: next_frontier,
        });
        output_height = response.output_height;
        head = response.new_head_sha256;
        frontier = next_frontier;
    }
    let decode_wall_ns = elapsed_ns(decode_started);

    let close_id = "close-000000";
    let (closed, close_payload) = transact_final(
        &mut stream,
        &key,
        &args.session_id,
        close_id,
        "close",
        &head,
        &[],
    )?;
    require_status(&closed, close_id, "closed")?;
    if !close_payload.is_empty()
        || closed.session_id != args.session_id
        || closed.committed_count != 0
        || closed.new_head_sha256 != head
        || closed.output_height != output_height
        || closed.transcript_sha256 != token_digest(&transcript)
        || validate_final_payload_abi(&closed, &close_payload, 0).is_err()
    {
        return Err("close response differs from the committed local cut".into());
    }

    let output_tps = args.output_tokens as f64 * 1_000_000_000.0 / decode_wall_ns as f64;
    let acceptance_rate = if drafted_total == 0 {
        0.0
    } else {
        accepted_total as f64 / drafted_total as f64
    };
    let receipt = json!({
        "schema": "muser.composite-dflash-qualification.v2",
        "created_unix_ms": unix_ms()?,
        "identity": args.identity,
        "session_id": args.session_id,
        "model": args.model,
        "model_sha256": model_sha256,
        "dflash": args.dflash,
        "dflash_sha256": dflash_sha256,
        "ggml_metallib": args.ggml_metallib,
        "ggml_metallib_sha256": ggml_metallib_sha256,
        "prompt_fixture": args.prompt_fixture,
        "prompt_tokens": args.prompt_tokens,
        "initial_hidden": args.initial_hidden,
        "initial_hidden_sha256": initial_hidden_digest,
        "expected_verifier_identity_sha256": args.expected_verifier_identity_sha256,
        "verifier_identity": verifier_identity,
        "verifier_identity_sha256": verifier_identity_digest,
        "server": format!("{}:{}", args.server_host, args.server_port),
        "output_tokens": args.output_tokens,
        "verify_length": args.verify_length,
        "mirror_overlap": args.mirror_overlap,
        "generated_tokens_sha256": token_digest(&generated),
        "final_transcript_sha256": token_digest(&transcript),
        "final_head_sha256": head,
        "final_frontier": frontier,
        "round_count": rounds.len(),
        "drafted_tokens": drafted_total,
        "accepted_draft_tokens": accepted_total,
        "acceptance_rate": acceptance_rate,
        "accepted_prefix_counts": rounds.iter().map(|round| round.accepted_drafts).collect::<Vec<_>>(),
        "open_rpc_ns": open_rpc_ns,
        "open_target_ns": opened.wall_ns,
        "dflash_warmup_ns": dflash_warmup_ns,
        "dflash_prime_ns": prime_ns,
        "decode_wall_ns": decode_wall_ns,
        "dflash_total_ns": dflash_total_ns,
        "verifier_wall_total_ns": verifier_wall_total_ns,
        "rpc_total_ns": rpc_total_ns,
        "request_to_provisional_total_ns": request_to_provisional_total_ns,
        "provisional_decode_total_ns": provisional_decode_total_ns,
        "provisional_prepare_total_ns": provisional_prepare_total_ns,
        "provisional_draft_total_ns": provisional_draft_total_ns,
        "final_wait_total_ns": final_wait_total_ns,
        "post_final_finish_total_ns": post_final_finish_total_ns,
        "mirror_overlap_attempts": mirror_overlap_attempts,
        "mirror_overlap_commits": mirror_overlap_commits,
        "mirror_overlap_rollbacks": mirror_overlap_rollbacks,
        "mirror_overlap_circuit_open_at_end": mirror_overlap_enabled,
        "response_payload_bytes": payload_total_bytes,
        "effective_output_tps": output_tps,
        "qualified_product_bar_tps": PRODUCT_BAR_TPS,
        "point_estimate_beats_product_bar": output_tps > PRODUCT_BAR_TPS,
        "rounds": rounds,
        "transport": "direct-tcp-authenticated-provisional-plus-bound-final-research-v2",
        "seal_eligible": false,
        "reason": "single-session research qualification; promotion requires paired strata, confidence bound, V2 sidecar, TLS, terminal and fault gates",
    });
    write_json_exclusive(&args.output, &receipt)?;
    println!(
        "{}",
        serde_json::to_string(&receipt).map_err(|error| error.to_string())?
    );
    Ok(())
}

fn validate_round(
    response: &FinalCore,
    final_payload: &[u8],
    provisional: &ProvisionalCore,
    provisional_payload: &[u8],
    expected: &RoundExpectation<'_>,
) -> Result<usize, String> {
    validate_verify_capture_timing(
        response
            .capture_timing
            .as_ref()
            .ok_or("verified response omitted capture timing")?,
        provisional,
    )?;
    let committed_bytes = response
        .committed_count
        .checked_mul(F16_ROW_BYTES)
        .ok_or("committed hidden byte count overflow")?;
    let committed_payload = provisional_payload
        .get(..committed_bytes)
        .ok_or("committed hidden prefix exceeds provisional payload")?;
    let candidate_digest = token_digest(expected.candidates);
    let committed_payload_digest = sha256_hex(committed_payload);
    let apc_parent_lag =
        verifier_cache_parent_lag(response.num_cached_tokens, expected.prior_transcript_len)?;
    if response.session_id != expected.session_id
        || response.base_head_sha256 != expected.base_head
        || response.candidate_count != Some(expected.candidates.len())
        || response.candidate_tokens_sha256.as_deref() != Some(candidate_digest.as_str())
        || response.provisional_sha256.as_deref() != Some(provisional.provisional_sha256.as_str())
        || response.target_tokens.len() != expected.candidates.len()
        || response.committed_count == 0
        || response.committed_count > expected.candidates.len()
        || response.committed_tokens != expected.candidates[..response.committed_count]
        || response.accepted_drafts + 1 != response.committed_count
        || response.output_height != expected.prior_height + response.committed_count
        || response.committed_payload_bytes != Some(committed_bytes)
        || response.committed_payload_sha256.as_deref() != Some(committed_payload_digest.as_str())
        || validate_final_payload_abi(response, final_payload, 0).is_err()
    {
        return Err("verified final/provisional geometry differs".into());
    }
    let expected_accepted = expected.candidates[1..]
        .iter()
        .zip(response.target_tokens.iter())
        .take_while(|(draft, target)| draft == target)
        .count();
    if response.accepted_drafts != expected_accepted
        || response.frontier_out
            != response
                .target_tokens
                .get(response.accepted_drafts)
                .copied()
    {
        return Err("verified response acceptance/frontier differs".into());
    }
    Ok(apc_parent_lag)
}

fn verifier_cache_parent_lag(
    cached_tokens: Option<usize>,
    authenticated_parent_tokens: usize,
) -> Result<usize, String> {
    let cached_tokens = cached_tokens.ok_or("verified response omitted prefix-cache cut")?;
    if cached_tokens > authenticated_parent_tokens
        || !cached_tokens.is_multiple_of(VERIFIER_CACHE_BLOCK_SIZE)
    {
        return Err("verified prefix-cache cut exceeds or misaligns the parent".into());
    }
    let lag = authenticated_parent_tokens - cached_tokens;
    if lag >= VERIFIER_CACHE_BLOCK_SIZE {
        return Err("verified prefix-cache lag exceeds one partial block".into());
    }
    Ok(lag)
}

fn validate_verify_capture_timing(
    timing: &CaptureTiming,
    provisional: &ProvisionalCore,
) -> Result<(), String> {
    let expected_layers = [1, 13, 25, 37, 49];
    if timing.layer_timings.len() != expected_layers.len()
        || timing
            .layer_timings
            .iter()
            .zip(expected_layers)
            .any(|(sample, layer)| {
                sample.layer != layer
                    || sample.arrival_offset_ns > sample.copy_enqueued_offset_ns
                    || !sample.copy_is_async
            })
    {
        return Err("verified capture layer timing differs".into());
    }
    let last = timing
        .layer_timings
        .last()
        .ok_or("verified capture timing omitted layer 49")?;
    let host_ready = timing
        .host_ready_offset_ns
        .ok_or("verified capture timing omitted host-ready offset")?;
    let payload_ready = timing
        .payload_ready_offset_ns
        .ok_or("verified capture timing omitted payload-ready offset")?;
    let callback_started = timing
        .host_ready_callback_started_offset_ns
        .ok_or("verified capture timing omitted callback-start offset")?;
    let send_started = timing
        .provisional_send_started_offset_ns
        .ok_or("verified capture timing omitted send-start offset")?;
    let send_completed = timing
        .provisional_send_completed_offset_ns
        .ok_or("verified capture timing omitted send-complete offset")?;
    let callback_completed = timing
        .host_ready_callback_completed_offset_ns
        .ok_or("verified capture timing omitted callback-complete offset")?;
    if last.copy_enqueued_offset_ns > host_ready
        || host_ready > payload_ready
        || payload_ready > callback_started
        || callback_started > send_started
        || send_started > send_completed
        || send_completed > callback_completed
        || callback_completed > timing.generate_finished_offset_ns
        || timing.generate_finished_offset_ns > timing.finish_started_offset_ns
        || timing.finish_started_offset_ns > timing.finish_completed_offset_ns
        || provisional.host_ready_offset_ns != host_ready
        || provisional.payload_ready_offset_ns != payload_ready
        || timing.last_layer_arrival_to_generate_finish_ns
            != timing
                .generate_finished_offset_ns
                .saturating_sub(last.arrival_offset_ns)
        || timing.last_layer_copy_enqueue_to_generate_finish_ns
            != timing
                .generate_finished_offset_ns
                .saturating_sub(last.copy_enqueued_offset_ns)
    {
        return Err("verified capture timing is not monotonic or bound".into());
    }
    Ok(())
}

fn validate_provisional(
    provisional: &ProvisionalCore,
    payload: &[u8],
    session_id: &str,
    request_id: &str,
    base_head: &str,
    candidates: &[u32],
) -> Result<(), String> {
    if provisional.schema != PROVISIONAL_SCHEMA
        || provisional.frame_kind != "verify_hidden_provisional"
        || provisional.session_id != session_id
        || provisional.request_id != request_id
        || provisional.base_head_sha256 != base_head
        || provisional.candidate_count != candidates.len()
        || provisional.candidate_tokens_sha256 != token_digest(candidates)
        || provisional.replayed
        || provisional.host_ready_offset_ns > provisional.payload_ready_offset_ns
        || provisional.payload_dtype != "f16_le"
        || provisional.payload_row_bytes != F16_ROW_BYTES
        || provisional.payload_rows != candidates.len()
        || provisional.payload_bytes != payload.len()
        || payload.len() != candidates.len().saturating_mul(F16_ROW_BYTES)
        || provisional.payload_sha256 != sha256_hex(payload)
        || provisional.provisional_sha256 != provisional_commitment_sha256(provisional)?
    {
        return Err("provisional frame identity, ABI, or commitment differs".into());
    }
    Ok(())
}

fn provisional_commitment_sha256(provisional: &ProvisionalCore) -> Result<String, String> {
    let mut core = serde_json::to_value(provisional).map_err(|error| error.to_string())?;
    core.as_object_mut()
        .ok_or("provisional core did not serialize as an object")?
        .remove("provisional_sha256")
        .ok_or("provisional commitment field disappeared")?;
    let commitment = json!({
        "core": core,
        "domain": PROVISIONAL_COMMITMENT_DOMAIN,
    });
    Ok(sha256_hex(
        &canonical_json(&commitment).map_err(|error| error.to_string())?,
    ))
}

fn verifier_identity_sha256(identity: &Value) -> Result<String, String> {
    let commitment = json!({
        "domain": VERIFIER_IDENTITY_DOMAIN,
        "identity": identity,
    });
    Ok(sha256_hex(
        &canonical_json(&commitment).map_err(|error| error.to_string())?,
    ))
}

fn validate_final_payload_abi(
    response: &FinalCore,
    payload: &[u8],
    expected_rows: usize,
) -> Result<(), String> {
    if response.payload_dtype != "f16_le"
        || response.payload_row_bytes != F16_ROW_BYTES
        || response.payload_rows != expected_rows
        || response.payload_bytes != payload.len()
        || payload.len() != expected_rows.saturating_mul(F16_ROW_BYTES)
        || response.payload_sha256 != sha256_hex(payload)
    {
        return Err("authenticated hidden-payload ABI differs".into());
    }
    Ok(())
}

fn require_status(response: &FinalCore, request_id: &str, status: &str) -> Result<(), String> {
    if response.request_id != request_id || response.status != status || response.replayed {
        return Err(format!(
            "response identity/status differs for {request_id}: {}",
            response.status
        ));
    }
    if let Some(error) = response.error.as_deref() {
        return Err(format!("remote verifier rejected {request_id}: {error}"));
    }
    Ok(())
}

fn send_request(
    stream: &mut TcpStream,
    key: &MacKey,
    session_id: &str,
    request_id: &str,
    command: &str,
    base_head: &str,
    candidates: &[u32],
) -> Result<(), String> {
    let core = RequestCore {
        schema: REQUEST_SCHEMA,
        session_id,
        request_id,
        command,
        base_head_sha256: base_head,
        candidates,
        // This bounded direct protocol has no durable client journal. Keep the
        // signed intent byte-identical across process restart so the server's
        // exact-replay table can reconcile a lost response. The production V3
        // gateway carries authenticated live deadlines in its durable WAL.
        sent_unix_ms: 0,
    };
    let core_bytes = canonical_json(&core).map_err(|error| error.to_string())?;
    let tag = key
        .tag_domain_hex(REQUEST_DOMAIN, &core_bytes)
        .map_err(|error| error.to_string())?;
    let frame = canonical_json(&RequestEnvelope {
        core: &core,
        hmac_sha256: tag,
    })
    .map_err(|error| error.to_string())?;
    if frame.is_empty() || frame.len() > MAX_HEADER_BYTES {
        return Err("request header exceeds the closed bound".into());
    }
    stream
        .write_all(&(frame.len() as u64).to_be_bytes())
        .and_then(|()| stream.write_all(&frame))
        .map_err(|error| error.to_string())
}

fn transact_final(
    stream: &mut TcpStream,
    key: &MacKey,
    session_id: &str,
    request_id: &str,
    command: &str,
    base_head: &str,
    candidates: &[u32],
) -> Result<(FinalCore, Vec<u8>), String> {
    send_request(
        stream, key, session_id, request_id, command, base_head, candidates,
    )?;
    receive_final(stream, key)
}

fn read_frame_header(stream: &mut TcpStream) -> Result<(Vec<u8>, usize), String> {
    let header_len = read_u64(stream)? as usize;
    let payload_len = read_u64(stream)? as usize;
    if header_len == 0 || header_len > MAX_HEADER_BYTES || payload_len > MAX_PAYLOAD_BYTES {
        return Err("response frame exceeds the closed bound".into());
    }
    let mut header = vec![0u8; header_len];
    stream
        .read_exact(&mut header)
        .map_err(|error| error.to_string())?;
    Ok((header, payload_len))
}

fn read_frame_payload(stream: &mut TcpStream, payload_len: usize) -> Result<Vec<u8>, String> {
    let mut payload = vec![0u8; payload_len];
    stream
        .read_exact(&mut payload)
        .map_err(|error| error.to_string())?;
    Ok(payload)
}

fn receive_provisional(
    stream: &mut TcpStream,
    key: &MacKey,
) -> Result<(ProvisionalCore, Vec<u8>), String> {
    let (header, payload_len) = read_frame_header(stream)?;
    let envelope: ProvisionalEnvelope =
        serde_json::from_slice(&header).map_err(|error| error.to_string())?;
    if canonical_json(&envelope).map_err(|error| error.to_string())? != header {
        return Err("provisional header is not canonical JSON".into());
    }
    let core = canonical_json(&envelope.core).map_err(|error| error.to_string())?;
    key.verify_domain_hex(PROVISIONAL_DOMAIN, &core, &envelope.hmac_sha256)
        .map_err(|error| error.to_string())?;
    if envelope.core.schema != PROVISIONAL_SCHEMA
        || envelope.core.payload_bytes != payload_len
        || envelope.core.payload_rows != envelope.core.candidate_count
        || payload_len
            != envelope
                .core
                .payload_rows
                .saturating_mul(envelope.core.payload_row_bytes)
    {
        return Err("provisional schema or authenticated payload length differs".into());
    }
    let payload = read_frame_payload(stream, payload_len)?;
    if envelope.core.payload_sha256 != sha256_hex(&payload) {
        return Err("provisional payload digest differs".into());
    }
    Ok((envelope.core, payload))
}

fn receive_final(stream: &mut TcpStream, key: &MacKey) -> Result<(FinalCore, Vec<u8>), String> {
    let (header, payload_len) = read_frame_header(stream)?;
    let envelope: FinalEnvelope =
        serde_json::from_slice(&header).map_err(|error| error.to_string())?;
    if canonical_json(&envelope).map_err(|error| error.to_string())? != header {
        return Err("final header is not canonical JSON".into());
    }
    let core = canonical_json(&envelope.core).map_err(|error| error.to_string())?;
    key.verify_domain_hex(FINAL_DOMAIN, &core, &envelope.hmac_sha256)
        .map_err(|error| error.to_string())?;
    if envelope.core.schema != FINAL_SCHEMA
        || envelope.core.payload_bytes != payload_len
        || payload_len
            != envelope
                .core
                .payload_rows
                .saturating_mul(envelope.core.payload_row_bytes)
    {
        return Err("final schema or authenticated payload length differs".into());
    }
    let payload = read_frame_payload(stream, payload_len)?;
    if envelope.core.payload_sha256 != sha256_hex(&payload) {
        return Err("final payload digest differs".into());
    }
    Ok((envelope.core, payload))
}

fn transition_head(
    base_head: &str,
    committed_tokens: &[u32],
    frontier_out: u32,
    output_height: usize,
    hidden_sha256: &str,
) -> Result<String, String> {
    let value = json!({
        "base_head_sha256": base_head,
        "committed_tokens": committed_tokens,
        "frontier_out": frontier_out,
        "hidden_sha256": hidden_sha256,
        "output_height": output_height,
        "schema": "muser.composite-verifier-head.v1",
    });
    Ok(sha256_hex(
        &canonical_json(&value).map_err(|error| error.to_string())?,
    ))
}

fn connect(args: &Args) -> Result<TcpStream, String> {
    let mut addresses = (args.server_host.as_str(), args.server_port)
        .to_socket_addrs()
        .map_err(|error| error.to_string())?;
    let address = addresses
        .next()
        .ok_or("server address resolved to nothing")?;
    let stream = TcpStream::connect_timeout(&address, args.timeout)
        .map_err(|error| format!("connect {address}: {error}"))?;
    stream
        .set_read_timeout(Some(args.timeout))
        .and_then(|()| stream.set_write_timeout(Some(args.timeout)))
        .and_then(|()| stream.set_nodelay(true))
        .map_err(|error| error.to_string())?;
    Ok(stream)
}

fn read_u64(stream: &mut TcpStream) -> Result<u64, String> {
    let mut bytes = [0u8; 8];
    stream
        .read_exact(&mut bytes)
        .map_err(|error| error.to_string())?;
    Ok(u64::from_be_bytes(bytes))
}

fn read_hidden(path: &Path, expected_floats: usize) -> Result<(Vec<f32>, String), String> {
    let expected_bytes = expected_floats
        .checked_mul(4)
        .ok_or("initial hidden byte count overflow")?;
    let mut bytes = Vec::new();
    File::open(path)
        .and_then(|mut file| file.read_to_end(&mut bytes))
        .map_err(|error| error.to_string())?;
    if bytes.len() != expected_bytes {
        return Err(format!(
            "initial hidden has {} bytes, expected {expected_bytes}",
            bytes.len()
        ));
    }
    let digest = sha256_hex(&bytes);
    let values = decode_hidden(&bytes, "f32_le")?;
    Ok((values, digest))
}

fn decode_hidden(bytes: &[u8], dtype: &str) -> Result<Vec<f32>, String> {
    match dtype {
        "f16_le" => {
            if !bytes.len().is_multiple_of(2) {
                return Err("hidden payload is not f16 aligned".into());
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|chunk| {
                    f16::from_bits(u16::from_le_bytes(
                        chunk.try_into().expect("two-byte chunk"),
                    ))
                    .to_f32()
                })
                .collect())
        }
        "f32_le" => {
            if !bytes.len().is_multiple_of(4) {
                return Err("hidden payload is not f32 aligned".into());
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
                .collect())
        }
        other => Err(format!("unsupported hidden payload dtype {other}")),
    }
}

fn read_tokens(path: &Path, count: usize) -> Result<Vec<u32>, String> {
    let text = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
    let tokens = text
        .split_whitespace()
        .take(count)
        .map(|value| value.parse::<u32>().map_err(|error| error.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    if tokens.len() != count || tokens.iter().any(|&token| token >= 202_048) {
        return Err("prompt fixture token count or vocabulary differs".into());
    }
    Ok(tokens)
}

fn token_digest(tokens: &[u32]) -> String {
    let mut digest = Sha256::new();
    for token in tokens {
        digest.update(token.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn elapsed_ns(started: Instant) -> u64 {
    started.elapsed().as_nanos().min(u64::MAX as u128) as u64
}

fn mirror_verified_drafts(
    enabled: bool,
    verify_length: usize,
    draft_count: usize,
    remaining: usize,
) -> Option<usize> {
    (enabled
        && verify_length == 15
        && draft_count == verify_length
        && remaining > verify_length + 1)
        .then_some(verify_length - 1)
}

fn held_back_mirror_frontier(
    mirror_attempted: bool,
    drafts: &[u32],
    used_drafts: usize,
) -> Result<Option<u32>, String> {
    if !mirror_attempted {
        return Ok(None);
    }
    drafts
        .get(used_drafts)
        .copied()
        .map(Some)
        .ok_or_else(|| "Mirror overlap omitted its held-back frontier proposal".into())
}

fn unix_ms() -> Result<u64, String> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_millis()
        .min(u64::MAX as u128) as u64)
}

fn publish_profile_ready_marker() -> Result<(), String> {
    let Some(path) = std::env::var_os("MUSER_COMPOSITE_PROFILE_READY_FILE") else {
        return Ok(());
    };
    let path = PathBuf::from(path);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|error| format!("cannot publish profile marker {}: {error}", path.display()))?;
    file.write_all(b"ready\n")
        .and_then(|()| file.sync_all())
        .map_err(|error| format!("cannot seal profile marker {}: {error}", path.display()))?;
    let delay_ms = std::env::var("MUSER_COMPOSITE_PROFILE_ATTACH_DELAY_MS")
        .map_err(|error| format!("profile marker requires attach delay: {error}"))?
        .parse::<u64>()
        .map_err(|_| "profile attach delay is not an integer".to_string())?;
    if !(100..=10_000).contains(&delay_ms) {
        return Err("profile attach delay must be within 100..=10000 ms".into());
    }
    std::thread::sleep(Duration::from_millis(delay_ms));
    Ok(())
}

fn write_json_exclusive(path: &Path, value: &Value) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| error.to_string())?;
    serde_json::to_writer_pretty(&mut file, value).map_err(|error| error.to_string())?;
    file.write_all(b"\n").map_err(|error| error.to_string())?;
    file.sync_all().map_err(|error| error.to_string())
}

fn parse_args() -> Result<Args, String> {
    let mut options = BTreeMap::<String, OsString>::new();
    let mut arguments = std::env::args_os().skip(1);
    while let Some(flag) = arguments.next() {
        let flag = flag
            .into_string()
            .map_err(|_| "argument flag is not UTF-8".to_string())?;
        if !flag.starts_with("--") || flag == "--" {
            return Err(format!("unexpected argument {flag}"));
        }
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        if options.insert(flag.clone(), value).is_some() {
            return Err(format!("duplicate argument {flag}"));
        }
    }
    let mut take = |name: &str| {
        options
            .remove(name)
            .ok_or_else(|| format!("missing required argument {name}"))
    };
    let model = PathBuf::from(take("--model")?);
    let dflash = PathBuf::from(take("--dflash")?);
    let prompt_fixture = PathBuf::from(take("--prompt-fixture")?);
    let initial_hidden = PathBuf::from(take("--initial-hidden")?);
    let hmac_key_file = PathBuf::from(take("--hmac-key-file")?);
    let ggml_metallib = PathBuf::from(take("--ggml-metallib")?);
    let output = PathBuf::from(take("--output")?);
    let initial_hidden_sha256 = utf8(take("--initial-hidden-sha256")?)?;
    let expected_verifier_identity_sha256 = utf8(take("--expected-verifier-identity-sha256")?)?;
    let expected_model_sha256 = utf8(take("--expected-model-sha256")?)?;
    let expected_dflash_sha256 = utf8(take("--expected-dflash-sha256")?)?;
    let server_host = utf8(take("--server-host")?)?;
    let server_port = parse_number::<u16>(take("--server-port")?, "server port")?;
    let session_id = utf8(take("--session-id")?)?;
    let identity = utf8(take("--identity")?)?;
    let prompt_tokens = optional_number(&mut options, "--prompt-tokens", 2_048)?;
    let output_tokens = optional_number(&mut options, "--output-tokens", 256)?;
    let verify_length = optional_number(&mut options, "--verify-length", 15)?;
    let mirror_overlap = optional_number::<u8>(&mut options, "--mirror-overlap", 0)?;
    if mirror_overlap > 1 {
        return Err("--mirror-overlap must be 0 or 1".into());
    }
    let timeout_seconds = optional_number(&mut options, "--timeout-seconds", 120u64)?;
    if let Some(flag) = options.keys().next() {
        return Err(format!("unknown argument {flag}"));
    }
    Ok(Args {
        model,
        dflash,
        prompt_fixture,
        prompt_tokens,
        initial_hidden,
        initial_hidden_sha256,
        expected_verifier_identity_sha256,
        expected_model_sha256,
        expected_dflash_sha256,
        hmac_key_file,
        ggml_metallib,
        server_host,
        server_port,
        session_id,
        output_tokens,
        verify_length,
        mirror_overlap: mirror_overlap == 1,
        timeout: Duration::from_secs(timeout_seconds),
        output,
        identity,
    })
}

fn optional_number<T>(
    options: &mut BTreeMap<String, OsString>,
    name: &str,
    default: T,
) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    options
        .remove(name)
        .map_or(Ok(default), |value| parse_number(value, name))
}

fn parse_number<T>(value: OsString, label: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    utf8(value)?
        .parse::<T>()
        .map_err(|error| format!("invalid {label}: {error}"))
}

fn utf8(value: OsString) -> Result<String, String> {
    value
        .into_string()
        .map_err(|_| "argument value is not UTF-8".into())
}

fn validate_args(args: &Args) -> Result<(), String> {
    for (label, path) in [
        ("model", &args.model),
        ("dflash", &args.dflash),
        ("prompt fixture", &args.prompt_fixture),
        ("initial hidden", &args.initial_hidden),
        ("HMAC key", &args.hmac_key_file),
        ("GGML metallib", &args.ggml_metallib),
    ] {
        let metadata = std::fs::symlink_metadata(path)
            .map_err(|error| format!("{label} {}: {error}", path.display()))?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(format!("{label} must be a regular non-symlink file"));
        }
    }
    if !args.output.is_absolute() || args.output.exists() {
        return Err("output must be a new absolute path".into());
    }
    if !is_sha256(&args.initial_hidden_sha256)
        || !is_sha256(&args.expected_verifier_identity_sha256)
        || !is_sha256(&args.expected_model_sha256)
        || !is_sha256(&args.expected_dflash_sha256)
    {
        return Err("expected SHA-256 values must be lowercase hexadecimal".into());
    }
    if args.prompt_tokens < 2 || args.output_tokens == 0 {
        return Err("prompt/output token geometry is invalid".into());
    }
    if !matches!(args.verify_length, 3 | 7 | 15) {
        return Err("verify length must be 3, 7, or 15".into());
    }
    if args.mirror_overlap && args.verify_length != 15 {
        return Err("mirror overlap requires verify length 15".into());
    }
    if !is_wire_id(&args.session_id) || args.identity.is_empty() {
        return Err("session/identity string is invalid".into());
    }
    Ok(())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_wire_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._:-".contains(&byte))
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|error| error.to_string())?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[cfg(test)]
mod tests {
    use super::{
        decode_hidden, provisional_commitment_sha256, sha256_hex, token_digest,
        validate_provisional, validate_round, verifier_cache_parent_lag, verifier_identity_sha256,
        CaptureTiming, FinalCore, LayerTiming, ProvisionalCore, ProvisionalEnvelope,
        RoundExpectation, F16_ROW_BYTES, FINAL_SCHEMA, PROVISIONAL_DOMAIN, PROVISIONAL_SCHEMA,
    };
    use kvpack_handoff::{canonical_json, MacKey};
    use serde_json::json;

    #[test]
    fn mirror_geometry_holds_back_only_the_fifteenth_proposal() {
        assert_eq!(super::mirror_verified_drafts(true, 15, 15, 17), Some(14));
        assert_eq!(super::mirror_verified_drafts(true, 15, 15, 16), None);
        assert_eq!(super::mirror_verified_drafts(true, 15, 14, 17), None);
        assert_eq!(super::mirror_verified_drafts(false, 15, 15, 17), None);
        assert_eq!(
            super::held_back_mirror_frontier(false, &[1, 2, 3], 3)
                .expect("disabled Mirror is lazy"),
            None
        );
        assert_eq!(
            super::held_back_mirror_frontier(true, &[1, 2, 3], 2).expect("held-back proposal"),
            Some(3)
        );
    }

    #[test]
    fn mirror_finite_length_schedule_never_overshoots() {
        for expected in [1usize, 15, 16, 17, 18, 511, 512] {
            let mut remaining = expected;
            let mut committed = 0usize;
            while remaining != 0 {
                let drafts = remaining.saturating_sub(1).min(15);
                let verified =
                    super::mirror_verified_drafts(true, 15, drafts, remaining).unwrap_or(drafts);
                let round = 1 + verified;
                assert!(round <= remaining);
                committed += round;
                remaining -= round;
            }
            assert_eq!(committed, expected);
        }
    }

    fn provisional(candidates: &[u32], payload: &[u8]) -> ProvisionalCore {
        let mut core = ProvisionalCore {
            schema: PROVISIONAL_SCHEMA.into(),
            base_head_sha256: "a".repeat(64),
            candidate_count: candidates.len(),
            candidate_tokens_sha256: token_digest(candidates),
            frame_kind: "verify_hidden_provisional".into(),
            host_ready_offset_ns: 100,
            payload_bytes: payload.len(),
            payload_dtype: "f16_le".into(),
            payload_ready_offset_ns: 125,
            payload_row_bytes: F16_ROW_BYTES,
            payload_rows: candidates.len(),
            payload_sha256: sha256_hex(payload),
            provisional_sha256: String::new(),
            replayed: false,
            request_id: "verify-1".into(),
            session_id: "session-a".into(),
        };
        core.provisional_sha256 = provisional_commitment_sha256(&core).expect("commitment");
        core
    }

    fn final_core(
        candidates: &[u32],
        provisional: &ProvisionalCore,
        provisional_payload: &[u8],
    ) -> FinalCore {
        let committed_count = candidates.len();
        FinalCore {
            schema: FINAL_SCHEMA.into(),
            accepted_drafts: committed_count - 1,
            base_head_sha256: provisional.base_head_sha256.clone(),
            candidate_count: Some(candidates.len()),
            candidate_tokens_sha256: Some(token_digest(candidates)),
            capture_timing: Some(CaptureTiming {
                capture_started_ns: 1_000,
                finish_completed_offset_ns: 170,
                finish_started_offset_ns: 160,
                generate_finished_offset_ns: 150,
                host_ready_callback_completed_offset_ns: Some(145),
                host_ready_callback_started_offset_ns: Some(130),
                host_ready_offset_ns: Some(100),
                last_layer_arrival_to_generate_finish_ns: 80,
                last_layer_copy_enqueue_to_generate_finish_ns: 75,
                layer_timings: [1, 13, 25, 37, 49]
                    .into_iter()
                    .enumerate()
                    .map(|(index, layer)| LayerTiming {
                        arrival_offset_ns: 10 + index as u64 * 15,
                        copy_is_async: true,
                        copy_enqueued_offset_ns: 15 + index as u64 * 15,
                        layer,
                    })
                    .collect(),
                payload_ready_offset_ns: Some(125),
                provisional_send_completed_offset_ns: Some(140),
                provisional_send_started_offset_ns: Some(135),
            }),
            committed_count,
            committed_payload_bytes: Some(committed_count * F16_ROW_BYTES),
            committed_payload_sha256: Some(sha256_hex(
                &provisional_payload[..committed_count * F16_ROW_BYTES],
            )),
            committed_tokens: candidates.to_vec(),
            error: None,
            frontier_out: Some(99),
            new_head_sha256: "b".repeat(64),
            num_cached_tokens: Some(2_048),
            output_height: committed_count,
            payload_bytes: 0,
            payload_dtype: "f16_le".into(),
            payload_row_bytes: F16_ROW_BYTES,
            payload_rows: 0,
            payload_sha256: sha256_hex(&[]),
            provisional_sha256: Some(provisional.provisional_sha256.clone()),
            replayed: false,
            request_id: provisional.request_id.clone(),
            session_id: provisional.session_id.clone(),
            status: "verified".into(),
            target_tokens: vec![candidates[1], 99],
            transcript_sha256: "c".repeat(64),
            verifier_identity: None,
            verifier_identity_sha256: None,
            wall_ns: 1,
        }
    }

    #[test]
    fn decodes_little_endian_f16_hidden_payload() {
        let payload = [0x00, 0x3c, 0x00, 0xc0, 0x00, 0x38];
        assert_eq!(
            decode_hidden(&payload, "f16_le").expect("f16 payload"),
            vec![1.0, -2.0, 0.5]
        );
    }

    #[test]
    fn hidden_payload_dtype_is_closed() {
        assert!(decode_hidden(&[0, 0], "bf16_le").is_err());
        assert!(decode_hidden(&[0], "f16_le").is_err());
        assert!(decode_hidden(&[0, 0], "f32_le").is_err());
    }

    #[test]
    fn provisional_commitment_matches_python_protocol_vector() {
        let candidates = [10, 11];
        let payload = vec![0; candidates.len() * F16_ROW_BYTES];
        let core = provisional(&candidates, &payload);
        assert_eq!(
            core.candidate_tokens_sha256,
            "50ac472466c102b9f97990af92e6c7acc1e76efbfd9b10904c4e2cfd533b0ca8"
        );
        assert_eq!(
            core.payload_sha256,
            "0c4f8dafe910c111d1bcd5e946e1f047d6289bc6ccd99371f76b67b6d8d20283"
        );
        assert_eq!(
            core.provisional_sha256,
            "1497efb6174f1d5969340982d37710b4513c82766eef857ab8de3263157aca6f"
        );
        validate_provisional(
            &core,
            &payload,
            "session-a",
            "verify-1",
            &"a".repeat(64),
            &candidates,
        )
        .expect("exact provisional");
    }

    #[test]
    fn provisional_hmac_envelope_matches_python_wire_vector() {
        let candidates = [10, 11];
        let payload = vec![0; candidates.len() * F16_ROW_BYTES];
        let core = provisional(&candidates, &payload);
        let key = MacKey::from_bytes(std::array::from_fn(|index| index as u8));
        let core_bytes = canonical_json(&core).expect("canonical core");
        let hmac_sha256 = key
            .tag_domain_hex(PROVISIONAL_DOMAIN, &core_bytes)
            .expect("provisional tag");
        assert_eq!(
            hmac_sha256,
            "928227191afcfcdc3eae927bde93ad93824ca1b0934741f06a9d125c5d93f000"
        );
        let header =
            canonical_json(&ProvisionalEnvelope { core, hmac_sha256 }).expect("canonical envelope");
        assert_eq!(header.len(), 772);
        assert_eq!(
            sha256_hex(&header),
            "2f939a2a4ba1d3230dd759cb8a673cfd98788942ca4436fd75a881172e0f6bb6"
        );
    }

    #[test]
    fn verifier_identity_matches_python_protocol_vector() {
        let identity = json!({
            "bundle_root_sha256": "a".repeat(64),
            "hidden_abi": {
                "dtype": "f16_le",
                "hidden_size": 6656,
                "layout": "token-major-selected-layer-major-hidden",
                "target_layers": [1, 13, 25, 37, 49],
            },
            "source_checkpoint_artifact_sha256": "b".repeat(64),
            "source_checkpoint_revision": "source-rev",
            "target_checkpoint_artifact_sha256": "c".repeat(64),
            "target_checkpoint_revision": "target-rev",
        });
        assert_eq!(
            verifier_identity_sha256(&identity).expect("identity digest"),
            "0513acbb5e7f9594a93562a3cc5ad2ac80336f18e25be60858cb331fb8f64dd3"
        );
    }

    #[test]
    fn verifier_cache_cut_may_lag_only_one_aligned_block_tail() {
        assert_eq!(
            verifier_cache_parent_lag(Some(2_048), 2_063).expect("partial block"),
            15
        );
        assert_eq!(
            verifier_cache_parent_lag(Some(2_064), 2_064).expect("aligned parent"),
            0
        );
        assert!(verifier_cache_parent_lag(Some(2_064), 2_063).is_err());
        assert!(verifier_cache_parent_lag(Some(2_049), 2_063).is_err());
        assert!(verifier_cache_parent_lag(Some(2_048), 2_064).is_err());
    }

    #[test]
    fn provisional_and_final_bind_exact_committed_hidden_prefix() {
        let candidates = [10, 11];
        let mut payload = vec![0; candidates.len() * F16_ROW_BYTES];
        payload[F16_ROW_BYTES] = 7;
        let core = provisional(&candidates, &payload);
        let mut final_core = final_core(&candidates, &core, &payload);
        validate_round(
            &final_core,
            &[],
            &core,
            &payload,
            &RoundExpectation {
                session_id: "session-a",
                base_head: &"a".repeat(64),
                candidates: &candidates,
                prior_height: 0,
                prior_transcript_len: 2_048,
            },
        )
        .expect("bound final");

        final_core.committed_payload_sha256 = Some("0".repeat(64));
        assert!(validate_round(
            &final_core,
            &[],
            &core,
            &payload,
            &RoundExpectation {
                session_id: "session-a",
                base_head: &"a".repeat(64),
                candidates: &candidates,
                prior_height: 0,
                prior_transcript_len: 2_048,
            },
        )
        .is_err());

        let mut corrupted = payload;
        corrupted[0] ^= 1;
        assert!(validate_provisional(
            &core,
            &corrupted,
            "session-a",
            "verify-1",
            &"a".repeat(64),
            &candidates,
        )
        .is_err());
    }

    #[test]
    fn rejection_and_single_candidate_bind_only_the_committed_row() {
        for (candidates, target_tokens, frontier) in
            [(vec![10, 11], vec![77, 99], 77), (vec![10], vec![88], 88)]
        {
            let payload = vec![3; candidates.len() * F16_ROW_BYTES];
            let core = provisional(&candidates, &payload);
            let mut final_core = final_core(&[10, 11], &core, &vec![3; 2 * F16_ROW_BYTES]);
            final_core.accepted_drafts = 0;
            final_core.candidate_count = Some(candidates.len());
            final_core.candidate_tokens_sha256 = Some(token_digest(&candidates));
            final_core.committed_count = 1;
            final_core.committed_payload_bytes = Some(F16_ROW_BYTES);
            final_core.committed_payload_sha256 = Some(sha256_hex(&payload[..F16_ROW_BYTES]));
            final_core.committed_tokens = vec![10];
            final_core.frontier_out = Some(frontier);
            final_core.output_height = 1;
            final_core.target_tokens = target_tokens;
            final_core.provisional_sha256 = Some(core.provisional_sha256.clone());
            validate_round(
                &final_core,
                &[],
                &core,
                &payload,
                &RoundExpectation {
                    session_id: "session-a",
                    base_head: &"a".repeat(64),
                    candidates: &candidates,
                    prior_height: 0,
                    prior_transcript_len: 2_048,
                },
            )
            .expect("rejected/single candidate geometry");
        }
    }
}
