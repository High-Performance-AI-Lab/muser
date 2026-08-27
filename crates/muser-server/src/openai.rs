//! Narrow OpenAI chat surface adapted from Ferrite's `openai.rs` contract.
//! Muser keeps the request/response/SSE shapes and UTF-8-safe streaming, but
//! routes directly into its one Muse `Model`/`Session` instead of Ferrite's
//! model manager, route VM, cascade, or speculative scheduler.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use base64::Engine as _;
use kvpack_handoff::MultimodalIdentityV2;
use muser_cluster::control::PrefillControlSegmentV2;
use muser_cluster::schedule::PREFIX_CUT_ALIGN;
use muser_engine::dflash::{
    DFlashAssistant, DFlashPreparedGreedy, DFlashPreparedSampled, DFlashRunError, DFlashSpecStats,
};
use muser_engine::sampling::{
    sample_discrete_distribution_mt, sample_discrete_distribution_mt_ordered,
    sample_distribution_mt_ordered, Mt19937, SamplingParams,
};
use muser_engine::vision::PreprocessedImage;
use muser_engine::{DecodeInput, EmbeddingSegment, Model, PrefillBatch, PrefillSegment, Session};
use muser_kvpack::economics::{RestoreBytes, Tier};
use muser_kvpack::reuse::{CacheSource, RemoteReuseAction, RemoteReuseOffer};
use muser_kvpack::session::DURABLE_FULL_INTERVAL;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::grammar::{json_object_gbnf, json_schema_to_gbnf, quoted_literal, GrammarMatcher};
use crate::session::Origin;
use crate::session_store::{BeginMutation, CachedGeneration, SamplerStateSnapshot, SessionBundle};
use crate::state::{
    ContextPolicy, InferenceRuntime, RemotePrefillMode, RemotePrefillRuntime, ServerState,
    SlotAcquireError, SlotPermit,
};

const MODEL_ID: &str = "muse-glimmer-30b";
const DEFAULT_MAX_TOKENS: usize = 256;
const MAX_OUTPUT_TOKENS: usize = 131_072;
/// The one server-side source of truth for the DFlash verification length.
/// Length 7 is the ledger's fixed-window product selection: the 2026-08-21
/// natural-text sweep gave it the best decode on both measured cells and
/// materially more robust recovery through cold acceptance patches than
/// length 15. Both speculative routes and `metrics::build_snapshot`'s
/// `specdec.draft_len` read
/// [`dflash_verify_len`], so the reported draft length can never drift from
/// the length the route actually ran.
pub const DFLASH_VERIFY_LEN: usize = 7;
/// Bounded wait for the process-wide accelerator lease. Past this the
/// request is shed with 429: an unbounded queue turns one slow generation
/// into a pile of clients waiting past their own timeouts.
const LEASE_WAIT: Duration = Duration::from_secs(3);
static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(0);

/// The effective verification length, `MUSER_DFLASH_VERIFY_LEN` overriding
/// the ledger default for tuning runs. Read once: a length that changed
/// mid-process would make the reported `draft_len` a lie for earlier runs.
pub fn dflash_verify_len() -> usize {
    static VERIFY_LEN: OnceLock<usize> = OnceLock::new();
    *VERIFY_LEN.get_or_init(|| {
        std::env::var("MUSER_DFLASH_VERIFY_LEN")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DFLASH_VERIFY_LEN)
    })
}

/// Seed an unseeded request from the OS entropy source. A fixed default made
/// every sampled request without an explicit `seed` replay one fixed
/// continuation; the effective seed is echoed back so a client can still pin
/// the run it just got.
pub fn entropy_seed() -> u64 {
    use rand::RngCore as _;
    u64::from(rand::rngs::OsRng.next_u32())
}

pub fn deserialize_seed<'de, D>(deserializer: D) -> Result<Option<u64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error as _;
    match Option::<serde_json::Value>::deserialize(deserializer)? {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::Number(number)) if number.as_i64() == Some(-1) => Ok(None),
        Some(serde_json::Value::Number(number)) => number
            .as_u64()
            .filter(|seed| *seed <= u64::from(u32::MAX))
            .map(Some)
            .ok_or_else(|| D::Error::custom("seed must be -1 or an unsigned 32-bit integer")),
        Some(_) => Err(D::Error::custom(
            "seed must be -1 or an unsigned 32-bit integer",
        )),
    }
}

pub fn deserialize_slot_id<'de, D>(deserializer: D) -> Result<Option<usize>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<i64>::deserialize(deserializer)?;
    match value {
        None | Some(-1) => Ok(None),
        Some(value) if value >= 0 => usize::try_from(value)
            .map(Some)
            .map_err(|_| serde::de::Error::custom("id_slot is too large")),
        Some(_) => Err(serde::de::Error::custom(
            "id_slot must be -1 or a nonnegative integer",
        )),
    }
}

fn canonical_sampler_name(name: &str) -> Option<&'static str> {
    match name.to_ascii_lowercase().as_str() {
        "dry" => Some("dry"),
        "top_k" | "top-k" | "topk" => Some("top_k"),
        "top_p" | "top-p" | "topp" | "nucleus" => Some("top_p"),
        "top_n_sigma" | "top-n-sigma" | "topnsigma" => Some("top_n_sigma"),
        "typ_p" | "typ-p" | "typp" | "typ" => Some("typ_p"),
        "min_p" | "min-p" | "minp" => Some("min_p"),
        "temperature" | "temp" => Some("temperature"),
        "xtc" => Some("xtc"),
        "penalties" => Some("penalties"),
        "adaptive_p" | "adaptive-p" | "adaptivep" => Some("adaptive_p"),
        "infill" => Some("infill"),
        _ => None,
    }
}

pub fn deserialize_sampler_sequence<'de, D>(
    deserializer: D,
) -> Result<Option<Vec<String>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de::Error as _;
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    let Some(value) = value else {
        return Ok(None);
    };
    let names = match value {
        serde_json::Value::Array(values) => values
            .into_iter()
            .map(|value| {
                value
                    .as_str()
                    .and_then(canonical_sampler_name)
                    .map(str::to_owned)
                    .ok_or_else(|| D::Error::custom(format!("unsupported sampler {value}")))
            })
            .collect::<Result<Vec<_>, _>>()?,
        serde_json::Value::String(chars) => chars
            .chars()
            .map(|character| {
                let name = match character {
                    'd' => "dry",
                    'k' => "top_k",
                    'y' => "typ_p",
                    'p' => "top_p",
                    's' => "top_n_sigma",
                    'm' => "min_p",
                    't' => "temperature",
                    'x' => "xtc",
                    'e' => "penalties",
                    'a' => "adaptive_p",
                    'i' => "infill",
                    _ => {
                        return Err(D::Error::custom(format!(
                            "unsupported sampler character {character:?}"
                        )))
                    }
                };
                Ok(name.to_owned())
            })
            .collect::<Result<Vec<_>, _>>()?,
        _ => {
            return Err(D::Error::custom(
                "samplers must be an array of names or a compact character string",
            ))
        }
    };
    Ok(Some(names))
}

pub fn deserialize_logit_bias<'de, D>(
    deserializer: D,
) -> Result<Option<std::collections::HashMap<String, f32>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    let Some(value) = value else {
        return Ok(None);
    };
    fn bias<E: serde::de::Error>(value: &serde_json::Value) -> Result<f32, E> {
        match value {
            serde_json::Value::Bool(false) => Ok(f32::NEG_INFINITY),
            serde_json::Value::Number(number) => number
                .as_f64()
                .filter(|value| {
                    value.is_finite() && *value >= f32::MIN as f64 && *value <= f32::MAX as f64
                })
                .map(|value| value as f32)
                .ok_or_else(|| E::custom("logit_bias value must be a finite float or false")),
            _ => Err(E::custom(
                "logit_bias value must be a finite float or false",
            )),
        }
    }
    let mut result = std::collections::HashMap::new();
    match value {
        serde_json::Value::Object(entries) => {
            for (target, value) in entries {
                result.insert(target, bias::<D::Error>(&value)?);
            }
        }
        serde_json::Value::Array(entries) => {
            for entry in entries {
                let serde_json::Value::Array(pair) = entry else {
                    return Err(serde::de::Error::custom(
                        "logit_bias array entries must be [token, bias] pairs",
                    ));
                };
                if pair.len() != 2 {
                    return Err(serde::de::Error::custom(
                        "logit_bias array entries must contain exactly two values",
                    ));
                }
                let target = match &pair[0] {
                    serde_json::Value::String(value) => value.clone(),
                    serde_json::Value::Number(value) if value.as_u64().is_some() => {
                        value.as_u64().expect("checked").to_string()
                    }
                    _ => {
                        return Err(serde::de::Error::custom(
                            "logit_bias token must be a nonnegative integer or string",
                        ))
                    }
                };
                result.insert(target, bias::<D::Error>(&pair[1])?);
            }
        }
        _ => {
            return Err(serde::de::Error::custom(
                "logit_bias must be an object or an array of [token, bias] pairs",
            ))
        }
    }
    Ok(Some(result))
}

fn default_cache_prompt() -> bool {
    true
}

fn default_parallel_tool_calls() -> bool {
    true
}

fn default_add_generation_prompt() -> bool {
    true
}

fn default_model() -> String {
    MODEL_ID.to_owned()
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct ChatRequest {
    #[serde(default = "default_model")]
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub stream: bool,
    pub stream_options: Option<StreamOptions>,
    pub max_tokens: Option<usize>,
    pub max_completion_tokens: Option<usize>,
    /// Pinned llama.cpp generation deadline. Values <= 0 disable it; a
    /// positive deadline is checked only when a generated token contains a
    /// newline, measured from the first generated token.
    pub t_max_predict_ms: Option<i64>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub typical_p: Option<f32>,
    pub min_p: Option<f32>,
    pub top_n_sigma: Option<f32>,
    pub min_keep: Option<usize>,
    #[serde(default)]
    pub ignore_eos: bool,
    #[serde(default, deserialize_with = "deserialize_logit_bias")]
    pub logit_bias: Option<std::collections::HashMap<String, f32>>,
    pub repeat_penalty: Option<f32>,
    pub repeat_last_n: Option<i32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub dry_multiplier: Option<f32>,
    pub dry_base: Option<f32>,
    pub dry_allowed_length: Option<usize>,
    pub dry_penalty_last_n: Option<i32>,
    pub dry_sequence_breakers: Option<Vec<String>>,
    pub mirostat: Option<u8>,
    pub mirostat_tau: Option<f32>,
    pub mirostat_eta: Option<f32>,
    pub adaptive_target: Option<f32>,
    pub adaptive_decay: Option<f32>,
    pub dynatemp_range: Option<f32>,
    pub dynatemp_exponent: Option<f32>,
    pub xtc_probability: Option<f32>,
    pub xtc_threshold: Option<f32>,
    #[serde(default, deserialize_with = "deserialize_sampler_sequence")]
    pub samplers: Option<Vec<String>>,
    #[serde(default)]
    pub reasoning_control: bool,
    #[serde(skip)]
    pub reasoning_end_signal: Option<Arc<AtomicBool>>,
    #[serde(default, deserialize_with = "deserialize_seed")]
    pub seed: Option<u64>,
    pub n: Option<u32>,
    /// Source-pinned physical slot selection and prompt-cache controls.
    #[serde(default, deserialize_with = "deserialize_slot_id")]
    pub id_slot: Option<usize>,
    #[serde(default = "default_cache_prompt")]
    pub cache_prompt: bool,
    pub stop: Option<StopField>,
    pub tools: Option<Vec<ToolDefinition>>,
    pub tool_choice: Option<ToolChoice>,
    /// Forwarded to the immutable GGUF template. This is source-compatible
    /// with llama.cpp and is distinct from arbitrary template replacement,
    /// which Muser intentionally rejects.
    #[serde(default = "default_add_generation_prompt")]
    pub add_generation_prompt: bool,
    /// The pinned Muse template supports multiple ATEM invokes in one
    /// assistant turn.  OpenAI defaults this field to true; when false both
    /// constrained decoding and post-generation validation permit one invoke.
    #[serde(default = "default_parallel_tool_calls")]
    pub parallel_tool_calls: bool,
    /// Parsed only to be rejected: silently ignoring a requested output
    /// contract would hand the client text that does not honour it.
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<usize>,
    pub response_format: Option<serde_json::Value>,
    /// Native llama.cpp constrained-decoding inputs. They are mutually
    /// exclusive with each other and with OpenAI `response_format`.
    pub grammar: Option<String>,
    pub json_schema: Option<serde_json::Value>,
    /// Private qualification extension. It bypasses chat templating so Muser
    /// and llama.cpp can receive the same audited token IDs for TTFT cells.
    #[serde(default)]
    pub muser_prompt_token_ids: Option<Vec<u32>>,
    /// Fail closed unless this request is running on the target-only, local,
    /// cache-disabled baseline route required by the release matrix.
    #[serde(default)]
    pub muser_baseline_ttft: bool,
    /// Muser stateful-generation extension. These three values are all-or-none.
    pub session_id: Option<String>,
    pub expected_revision: Option<u64>,
    #[serde(skip)]
    pub idempotency_key: Option<String>,
    /// Canonical digest of the caller-supplied JSON body. The HTTP layer
    /// fills this before applying server defaults so an idempotency key can
    /// never replay a different mutation at the same session revision.
    #[serde(skip)]
    pub idempotency_request_sha256: Option<[u8; 32]>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct Message {
    pub role: String,
    pub content: MessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<MessageToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recipient: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_turn: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
    Null(()),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct MessageToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: MessageToolFunction,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct MessageToolFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ToolFunctionDefinition,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct ToolFunctionDefinition {
    pub name: String,
    pub description: Option<String>,
    pub parameters: serde_json::Value,
    pub strict: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub enum ToolChoice {
    Mode(String),
    Named(NamedToolChoice),
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct NamedToolChoice {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: NamedToolChoiceFunction,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct NamedToolChoiceFunction {
    pub name: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type", deny_unknown_fields)]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct ImageUrl {
    pub url: String,
}

#[derive(Debug, Deserialize, Default, Clone)]
#[serde(deny_unknown_fields)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum StopField {
    One(String),
    Many(Vec<String>),
}

impl StopField {
    fn values(&self) -> Vec<String> {
        match self {
            Self::One(value) => vec![value.clone()],
            Self::Many(values) => values.clone(),
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            Self::One(value) => value.is_empty(),
            Self::Many(values) => values.is_empty(),
        }
    }
}

#[derive(Debug, Serialize, Clone, Copy)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub prompt_tokens_details: PromptTokensDetails,
}

#[derive(Debug, Serialize, Clone, Copy)]
pub struct PromptTokensDetails {
    pub cached_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
    /// The sampling seed this request actually ran with, whether the client
    /// pinned it or the server drew it from entropy. Replaying the exact run
    /// is impossible without it.
    pub system_fingerprint: String,
    #[serde(skip)]
    pub muser_seed: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub muser_session_revision: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: AssistantMessage,
    pub finish_reason: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChoiceLogprobs>,
}

#[derive(Debug, Serialize)]
pub struct AssistantMessage {
    pub role: &'static str,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ParsedToolCall>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ParsedToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: ParsedFunctionCall,
}

#[derive(Debug, Clone, Serialize)]
pub struct ParsedFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Serialize)]
pub struct ChatChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: String,
    pub choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Carried on the opening role chunk so a streamed run reports the same
    /// effective seed a non-streamed one does.
    #[serde(skip)]
    pub muser_seed: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: ChunkDelta,
    pub finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Option<String>>,
}

#[derive(Debug)]
pub struct Generated {
    pub text: String,
    pub usage: Usage,
    pub finish_reason: &'static str,
    pub stop_type: &'static str,
    pub stopping_word: String,
    pub seed: u64,
    pub session_revision: Option<u64>,
    pub logprobs: Option<ChoiceLogprobs>,
    /// Every selected completion token, including an EOG token that is
    /// observable in usage/logprobs but deliberately never decoded into KV.
    pub sampled_tokens: Vec<u32>,
    pub context: Vec<u32>,
    pub slot_id: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChoiceLogprobs {
    pub content: Vec<TokenLogprob>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TokenLogprob {
    pub id: u32,
    pub token: String,
    pub logprob: f64,
    pub bytes: Vec<u8>,
    pub top_logprobs: Vec<TopLogprob>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TopLogprob {
    pub id: u32,
    pub token: String,
    pub logprob: f64,
    pub bytes: Vec<u8>,
}

/// One committed streaming fragment and, when requested, the target-verified
/// distribution for the token whose commit made that fragment publishable.
/// Empty fragments are cancellation probes and never carry a logprob.
pub struct StreamEvent<'a> {
    pub text: &'a str,
    /// The generated token responsible for this stream chunk. Native llama
    /// streaming emits one chunk per sampled token even when stop/UTF-8
    /// buffering makes `text` empty.
    pub token: Option<u32>,
    pub logprob: Option<&'a TokenLogprob>,
}

#[derive(Debug, thiserror::Error)]
pub enum ChatError {
    #[error("invalid request: {0}")]
    BadRequest(String),
    #[error("model '{0}' is not loaded")]
    ModelUnavailable(String),
    #[error("generation failed: {0}")]
    Engine(String),
    #[error("client disconnected")]
    Cancelled,
    #[error("the accelerator lease is busy; retry shortly")]
    Overloaded,
    #[error("the accelerator state is unhealthy; restart the server")]
    Unavailable,
    #[error("session conflict: {0}")]
    Conflict(String),
}

impl ChatError {
    pub fn status(&self) -> (u16, &'static str, &'static str) {
        match self {
            Self::BadRequest(_) => (400, "Bad Request", "invalid_request_error"),
            Self::ModelUnavailable(_) => (404, "Not Found", "model_not_found"),
            Self::Engine(_) => (500, "Internal Server Error", "generation_error"),
            Self::Cancelled => (499, "Client Closed Request", "cancelled"),
            Self::Overloaded => (429, "Too Many Requests", "rate_limit_exceeded"),
            Self::Unavailable => (503, "Service Unavailable", "engine_unavailable"),
            Self::Conflict(_) => (409, "Conflict", "conflict"),
        }
    }

    pub fn json(&self) -> serde_json::Value {
        let (_, _, kind) = self.status();
        serde_json::json!({"error": {"type": kind, "message": self.to_string()}})
    }
}

pub fn new_request_identity() -> (String, u64) {
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let counter = REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed);
    (format!("chatcmpl-{created}{counter:04}"), created)
}

struct ParsedAtemOutput {
    content: String,
    reasoning: String,
    tool_calls: Vec<ParsedToolCall>,
}

#[derive(Debug)]
pub enum AtemStreamEvent {
    Content(String),
    Reasoning(String),
    ToolCall { index: usize, call: ParsedToolCall },
}

#[derive(Debug)]
struct AtemActivePhase {
    recipient: String,
    body_start: usize,
    emitted_until: usize,
}

/// Incremental parser for the GGUF's Muse/ATEM assistant wire format.
///
/// Plain assistant text is passed through as soon as the short recipient
/// discriminator is resolved. Structured reasoning/content is released while
/// it is generated, retaining only a possible split `<|eom|>` suffix. Tool
/// calls are released one complete invoke at a time (before later phases or
/// generation finish), so malformed XML can never be presented as ordinary
/// assistant text.
#[derive(Debug, Default)]
pub struct AtemStreamParser {
    raw: String,
    structured: Option<bool>,
    cursor: usize,
    plain_emitted: usize,
    active: Option<AtemActivePhase>,
    tool_calls_emitted: usize,
}

impl AtemStreamParser {
    pub fn push(&mut self, piece: &str) -> Result<Vec<AtemStreamEvent>, String> {
        self.raw.push_str(piece);
        self.drain(false)
    }

    pub fn finish(&mut self) -> Result<Vec<AtemStreamEvent>, String> {
        let events = self.drain(true)?;
        if self.structured == Some(true) {
            // Full validation catches an invalid recipient, dangling invoke,
            // duplicate parameter, or malformed terminal phase even if an
            // earlier valid reasoning delta was already delivered.
            parse_atem_output(&self.raw)?;
        }
        Ok(events)
    }

    /// Finish a streamed response without exposing a length-truncated ATEM
    /// recipient header as user-visible text. Pinned llama-server retains the
    /// same raw header in its nonstream response but emits no content delta
    /// for the unresolved streaming prefix.
    pub fn finish_stream(&mut self) -> Result<Vec<AtemStreamEvent>, String> {
        if self.structured.is_none() {
            let trimmed = self.raw.trim_start();
            let structured_prefix = trimmed.starts_with("to=")
                || trimmed.starts_with("<|start|>assistant")
                || "to=".starts_with(trimmed)
                || "<|start|>assistant".starts_with(trimmed);
            if structured_prefix && !trimmed.contains("<|message|>") {
                return Ok(Vec::new());
            }
        }
        self.finish()
    }

    pub fn is_structured(&self) -> bool {
        self.structured == Some(true)
    }

    fn drain(&mut self, finishing: bool) -> Result<Vec<AtemStreamEvent>, String> {
        let mut events = Vec::new();
        if self.structured.is_none() {
            let trimmed = self.raw.trim_start();
            let possible = ["to=", "<|start|>assistant"];
            if possible.iter().any(|prefix| prefix.starts_with(trimmed)) && !finishing {
                return Ok(events);
            }
            if (trimmed.starts_with("to=") || trimmed.starts_with("<|start|>assistant"))
                && !trimmed.contains("<|message|>")
                && !finishing
            {
                return Ok(events);
            }
            self.structured = Some(
                (trimmed.starts_with("to=") || trimmed.starts_with("<|start|>assistant"))
                    && trimmed.contains("<|message|>"),
            );
        }
        if self.structured == Some(false) {
            if self.plain_emitted < self.raw.len() {
                events.push(AtemStreamEvent::Content(
                    self.raw[self.plain_emitted..].to_owned(),
                ));
                self.plain_emitted = self.raw.len();
            }
            return Ok(events);
        }

        const PHASE_END: &str = "<|eom|>";
        loop {
            if self.active.is_none() {
                if self.cursor == self.raw.len() {
                    break;
                }
                let mut header = self.cursor;
                if self.raw[header..].starts_with("<|start|>assistant") {
                    header += "<|start|>assistant".len();
                } else if "<|start|>assistant".starts_with(&self.raw[header..]) && !finishing {
                    break;
                }
                let recipient;
                if self.raw[header..].starts_with(" to=") {
                    let rest = header + " to=".len();
                    let Some(relative) = self.raw[rest..].find("<|message|>") else {
                        if finishing {
                            return Err("assistant recipient header omits <|message|>".into());
                        }
                        break;
                    };
                    recipient = self.raw[rest..rest + relative].trim().to_owned();
                    if recipient.is_empty() {
                        return Err("assistant recipient is empty".into());
                    }
                    header = rest + relative + "<|message|>".len();
                } else if self.raw[header..].starts_with("<|message|>") {
                    recipient = "user".into();
                    header += "<|message|>".len();
                } else if !finishing {
                    break;
                } else {
                    return Err("structured assistant phase omits a recipient header".into());
                }
                self.active = Some(AtemActivePhase {
                    recipient,
                    body_start: header,
                    emitted_until: header,
                });
            }

            let phase = self.active.as_mut().expect("phase was installed");
            let phase_end = self.raw[phase.body_start..]
                .find(PHASE_END)
                .map(|offset| phase.body_start + offset);
            if phase.recipient == "self" || phase.recipient == "user" {
                let safe_end = match phase_end {
                    Some(end) => end,
                    None if finishing => self.raw.len(),
                    None => self.raw.len() - marker_suffix_len(&self.raw, PHASE_END),
                };
                if safe_end > phase.emitted_until {
                    let text = self.raw[phase.emitted_until..safe_end].to_owned();
                    events.push(if phase.recipient == "self" {
                        AtemStreamEvent::Reasoning(text)
                    } else {
                        AtemStreamEvent::Content(text)
                    });
                    phase.emitted_until = safe_end;
                }
            }
            let Some(end) = phase_end else {
                if finishing && phase.recipient != "self" && phase.recipient != "user" {
                    let calls = parse_atem_calls(&self.raw[phase.body_start..])?;
                    validate_atem_recipient(&phase.recipient, &calls)?;
                    for call in calls {
                        let index = self.tool_calls_emitted;
                        self.tool_calls_emitted += 1;
                        events.push(AtemStreamEvent::ToolCall { index, call });
                    }
                }
                break;
            };
            if phase.recipient != "self" && phase.recipient != "user" {
                let calls = parse_atem_calls(&self.raw[phase.body_start..end])?;
                validate_atem_recipient(&phase.recipient, &calls)?;
                for call in calls {
                    let index = self.tool_calls_emitted;
                    self.tool_calls_emitted += 1;
                    events.push(AtemStreamEvent::ToolCall { index, call });
                }
            }
            self.cursor = end + PHASE_END.len();
            self.active = None;
        }
        Ok(events)
    }
}

fn marker_suffix_len(text: &str, marker: &str) -> usize {
    let limit = text.len().min(marker.len().saturating_sub(1));
    // Compare bytes: markers are ASCII, so a match can only begin on a char
    // boundary, and byte slicing never panics on a multibyte tail.
    let text = text.as_bytes();
    let marker = marker.as_bytes();
    (1..=limit)
        .rev()
        .find(|&length| marker.starts_with(&text[text.len() - length..]))
        .unwrap_or(0)
}

/// Keep the marker-detection window to its last 64 bytes, rounding the cut
/// down to a char boundary so multibyte characters never split.
fn trim_phase_tail(phase_tail: &mut String) {
    let mut split = phase_tail.len() - 64;
    while !phase_tail.is_char_boundary(split) {
        split -= 1;
    }
    phase_tail.drain(..split);
}

fn parse_atem_output(text: &str) -> Result<ParsedAtemOutput, String> {
    let mut output = ParsedAtemOutput {
        content: String::new(),
        reasoning: String::new(),
        tool_calls: Vec::new(),
    };
    let cleaned = text.replace("<|eot|>", "");
    for raw_phase in cleaned.split("<|eom|>") {
        let mut phase = raw_phase.trim_start_matches("<|start|>assistant");
        let mut recipient = "user";
        if let Some(rest) = phase.strip_prefix(" to=") {
            let (name, body) = rest
                .split_once("<|message|>")
                .ok_or_else(|| "assistant recipient header omits <|message|>".to_string())?;
            recipient = name.trim();
            phase = body;
        } else if let Some(body) = phase.strip_prefix("<|message|>") {
            phase = body;
        }
        if phase.is_empty() {
            continue;
        }
        if recipient == "self" {
            output.reasoning.push_str(phase);
        } else if phase.contains("<atem:function_calls>") {
            let calls = parse_atem_calls(phase)?;
            validate_atem_recipient(recipient, &calls)?;
            output.tool_calls.extend(calls);
        } else if recipient == "user" {
            output.content.push_str(phase);
        } else {
            return Err(format!(
                "assistant emitted recipient {recipient:?} without an ATEM function-call block"
            ));
        }
    }
    Ok(output)
}

pub fn atem_delta_chunks(
    id: &str,
    created: u64,
    model: &str,
    text: &str,
) -> Result<Vec<serde_json::Value>, ChatError> {
    atem_delta_chunks_indexed(id, created, model, text, 0)
}

pub fn atem_delta_chunks_indexed(
    id: &str,
    created: u64,
    model: &str,
    text: &str,
    choice_index: u32,
) -> Result<Vec<serde_json::Value>, ChatError> {
    let parsed = parse_atem_output(text)
        .map_err(|error| ChatError::Engine(format!("malformed Muse ATEM output: {error}")))?;
    let mut chunks = Vec::new();
    if !parsed.reasoning.is_empty() {
        chunks.push(serde_json::json!({
            "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": choice_index, "delta": {"reasoning_content": parsed.reasoning}, "finish_reason": null}]
        }));
    }
    for (index, call) in parsed.tool_calls.into_iter().enumerate() {
        chunks.push(serde_json::json!({
            "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": choice_index, "delta": {"tool_calls": [{
                "index": index, "id": call.id, "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments}
            }]}, "finish_reason": null}]
        }));
    }
    if !parsed.content.is_empty() {
        chunks.push(serde_json::json!({
            "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
            "choices": [{"index": choice_index, "delta": {"content": parsed.content}, "finish_reason": null}]
        }));
    }
    Ok(chunks)
}

pub fn atem_finish_reason(text: &str, fallback: &'static str) -> &'static str {
    match parse_atem_output(text) {
        Ok(parsed) if !parsed.tool_calls.is_empty() => "tool_calls",
        _ => fallback,
    }
}

fn parse_atem_calls(text: &str) -> Result<Vec<ParsedToolCall>, String> {
    let trimmed = text.trim();
    let mut cursor = trimmed
        .strip_prefix("<atem:function_calls>")
        .and_then(|inner| inner.strip_suffix("</atem:function_calls>"))
        .ok_or_else(|| {
            "ATEM function-call phase must contain one closed outer block".to_string()
        })?;
    let mut calls = Vec::new();
    loop {
        cursor = cursor.trim_start();
        if cursor.is_empty() {
            break;
        }
        cursor = cursor
            .strip_prefix("<atem:invoke name=\"")
            .ok_or_else(|| "unexpected text outside an ATEM invoke".to_string())?;
        let quote = cursor
            .find("\">")
            .ok_or_else(|| "malformed ATEM invoke name".to_string())?;
        let name = &cursor[..quote];
        if name.is_empty()
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
        {
            return Err("ATEM invoke name is outside the closed function-name grammar".into());
        }
        cursor = &cursor[quote + 2..];
        let end = cursor
            .find("</atem:invoke>")
            .ok_or_else(|| "unterminated ATEM invoke".to_string())?;
        let mut body = &cursor[..end];
        let mut parameters = serde_json::Map::new();
        loop {
            body = body.trim_start();
            if body.is_empty() {
                break;
            }
            body = body
                .strip_prefix("<atem:parameter name=\"")
                .ok_or_else(|| "unexpected text outside an ATEM parameter".to_string())?;
            let name_end = body
                .find("\">")
                .ok_or_else(|| "malformed ATEM parameter name".to_string())?;
            let parameter_name = &body[..name_end];
            if parameter_name.is_empty() {
                return Err("ATEM parameter name is empty".into());
            }
            body = &body[name_end + 2..];
            let value_end = body
                .find("</atem:parameter>")
                .ok_or_else(|| "unterminated ATEM parameter".to_string())?;
            let raw = &body[..value_end];
            if raw.contains("<atem:") || raw.contains("</atem:") {
                return Err("ATEM parameter value contains a structural tag".into());
            }
            let value = serde_json::from_str(raw)
                .unwrap_or_else(|_| serde_json::Value::String(raw.to_owned()));
            if parameters.insert(parameter_name.into(), value).is_some() {
                return Err(format!("duplicate ATEM parameter {parameter_name:?}"));
            }
            body = &body[value_end + "</atem:parameter>".len()..];
        }
        let arguments = serde_json::to_string(&parameters).expect("JSON map serializes");
        let digest = Sha256::digest(format!("{name}\0{arguments}\0{}", calls.len()).as_bytes());
        let call_id = digest[..12]
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        calls.push(ParsedToolCall {
            id: format!("call_{call_id}"),
            kind: "function",
            function: ParsedFunctionCall {
                name: name.into(),
                arguments,
            },
        });
        cursor = &cursor[end + "</atem:invoke>".len()..];
    }
    if calls.is_empty() {
        return Err("ATEM function-call block contains no invokes".into());
    }
    Ok(calls)
}

fn validate_atem_recipient(recipient: &str, calls: &[ParsedToolCall]) -> Result<(), String> {
    if recipient == "tool"
        || calls
            .first()
            .is_some_and(|call| call.function.name == recipient)
    {
        return Ok(());
    }
    Err(format!(
        "assistant recipient {recipient:?} does not match the first ATEM invoke"
    ))
}

pub fn role_chunk(id: &str, created: u64, model: &str, seed: u64) -> ChatChunk {
    role_chunk_indexed(id, created, model, seed, 0)
}

pub fn role_chunk_indexed(id: &str, created: u64, model: &str, seed: u64, index: u32) -> ChatChunk {
    let mut opening = chunk(
        id,
        created,
        model,
        Some("assistant"),
        None,
        None,
        None,
        index,
    );
    // The frozen llama stream opens with {role:"assistant", content:null}.
    // A nested option distinguishes that explicit null from omitted content
    // on reasoning/tool/terminal deltas.
    opening.choices[0].delta.content = Some(None);
    opening.muser_seed = Some(seed);
    opening
}

pub fn content_chunk(id: &str, created: u64, model: &str, content: String) -> ChatChunk {
    content_chunk_indexed(id, created, model, content, 0)
}

pub fn content_chunk_indexed(
    id: &str,
    created: u64,
    model: &str,
    content: String,
    index: u32,
) -> ChatChunk {
    chunk(id, created, model, None, Some(content), None, None, index)
}

pub fn terminal_chunk(
    id: &str,
    created: u64,
    model: &str,
    finish_reason: &'static str,
) -> ChatChunk {
    terminal_chunk_indexed(id, created, model, finish_reason, 0)
}

pub fn terminal_chunk_indexed(
    id: &str,
    created: u64,
    model: &str,
    finish_reason: &'static str,
    index: u32,
) -> ChatChunk {
    chunk(
        id,
        created,
        model,
        None,
        None,
        Some(finish_reason),
        None,
        index,
    )
}

pub fn usage_chunk(id: &str, created: u64, model: &str, usage: Usage) -> ChatChunk {
    chunk(id, created, model, None, None, None, Some(usage), 0)
}

#[allow(clippy::too_many_arguments)]
fn chunk(
    id: &str,
    created: u64,
    model: &str,
    role: Option<&'static str>,
    content: Option<String>,
    finish_reason: Option<&'static str>,
    usage: Option<Usage>,
    index: u32,
) -> ChatChunk {
    ChatChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        system_fingerprint: "muser-v0.1".into(),
        choices: if usage.is_some() {
            Vec::new()
        } else {
            vec![ChunkChoice {
                index,
                delta: ChunkDelta {
                    role,
                    content: content.map(Some),
                },
                finish_reason,
            }]
        },
        usage,
        muser_seed: None,
    }
}

pub fn response(id: String, created: u64, model: String, generated: Generated) -> ChatResponse {
    response_many(id, created, model, vec![generated])
}

pub fn response_many(
    id: String,
    created: u64,
    model: String,
    generated: Vec<Generated>,
) -> ChatResponse {
    let seed = generated.first().map_or(0, |value| value.seed);
    let prompt_tokens = generated
        .first()
        .map_or(0, |value| value.usage.prompt_tokens);
    // Pinned llama-server reports usage for one completion even when `n`
    // creates multiple choices; it does not sum the parallel alternatives.
    let completion_tokens = generated
        .first()
        .map_or(0, |value| value.usage.completion_tokens);
    let session_revision = generated.first().and_then(|value| value.session_revision);
    let choices = generated
        .into_iter()
        .enumerate()
        .map(|(index, generated)| {
            let parsed = parse_atem_output(&generated.text).unwrap_or_else(|_| ParsedAtemOutput {
                content: generated.text.clone(),
                reasoning: String::new(),
                tool_calls: Vec::new(),
            });
            let finish_reason = if parsed.tool_calls.is_empty() {
                generated.finish_reason
            } else {
                "tool_calls"
            };
            ChatChoice {
                index: index as u32,
                message: AssistantMessage {
                    role: "assistant",
                    // Pinned llama-server retains an empty string for an
                    // ATEM-only assistant message; OpenAI's nullable variant
                    // is valid in the abstract API but is not byte-contract
                    // compatible with this frozen source.
                    content: Some(parsed.content),
                    reasoning_content: (!parsed.reasoning.is_empty()).then_some(parsed.reasoning),
                    tool_calls: (!parsed.tool_calls.is_empty()).then_some(parsed.tool_calls),
                },
                finish_reason,
                logprobs: generated.logprobs,
            }
        })
        .collect();
    ChatResponse {
        id,
        object: "chat.completion",
        created,
        model,
        choices,
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            prompt_tokens_details: PromptTokensDetails { cached_tokens: 0 },
        },
        system_fingerprint: format!("muser-seed-{seed}"),
        muser_seed: seed,
        muser_session_revision: session_revision,
    }
}

/// Everything that can reject a request before a single byte is committed to
/// the wire. The streaming route must run this before its 200/SSE header: an
/// error discovered afterwards can only be delivered as an in-stream frame
/// under a status code that already claimed success.
pub fn precheck(state: &ServerState, request: &ChatRequest) -> Result<(), ChatError> {
    validate_request(request)?;
    if state.inference.is_none() {
        return Err(ChatError::ModelUnavailable(request.model.clone()));
    }
    Ok(())
}

/// Generate greedily and invoke `emit` only with UTF-8-complete text that is
/// known not to belong to a configured stop sequence. A failed callback is a
/// connection cancellation; the session is reset before this function exits.
/// An empty `emit` is a cancellation probe: it never reaches the wire, and
/// the caller answers it from the state of its own connection.
pub fn generate(
    state: &ServerState,
    request: &ChatRequest,
    session_id: &str,
    mut emit: impl FnMut(&str) -> Result<(), ChatError>,
) -> Result<Generated, ChatError> {
    generate_events(state, request, session_id, |event| emit(event.text))
}

pub fn generate_events(
    state: &ServerState,
    request: &ChatRequest,
    session_id: &str,
    mut emit: impl FnMut(StreamEvent<'_>) -> Result<(), ChatError>,
) -> Result<Generated, ChatError> {
    let generation_started = Instant::now();
    state.set_active_phase("prefill");
    // Clears the in-flight marker on every exit path, panics included.
    struct ActivePhaseGuard<'a>(&'a ServerState);
    impl Drop for ActivePhaseGuard<'_> {
        fn drop(&mut self) {
            self.0.clear_active();
        }
    }
    let _active_guard = ActivePhaseGuard(state);
    let mut first_content_emitted = false;
    let mut first_token_at: Option<Instant> = None;
    let mut last_emit_at: Option<Instant> = None;
    let mut measured_emit = |text: &str, token: Option<u32>, logprob: Option<&TokenLogprob>| {
        if !text.is_empty() {
            if !first_content_emitted {
                first_content_emitted = true;
                first_token_at = Some(Instant::now());
                state.record_ttft(
                    generation_started
                        .elapsed()
                        .as_nanos()
                        .min(u64::MAX as u128) as u64,
                );
            }
            let now = Instant::now();
            if let Some(previous) = last_emit_at {
                state.record_decode_gap(
                    now.duration_since(previous)
                        .as_nanos()
                        .min(u64::MAX as u128) as u64,
                );
            }
            last_emit_at = Some(now);
        }
        let write_started = Instant::now();
        let result = emit(StreamEvent {
            text,
            token,
            logprob,
        });
        state.record_phase(
            "enqueue_write",
            write_started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        );
        result
    };
    validate_request(request)?;
    let runtime = state
        .inference
        .as_ref()
        .ok_or_else(|| ChatError::ModelUnavailable(request.model.clone()))?;
    if request.muser_baseline_ttft
        && (runtime.prefix_cache_enabled
            || runtime.remote_prefill.is_some()
            || runtime.dflash.is_some())
    {
        return Err(ChatError::BadRequest(
            "baseline TTFT requires cache, remote prefill, and DFlash to be disabled".into(),
        ));
    }
    let requested = request
        .max_completion_tokens
        .or(request.max_tokens)
        .unwrap_or(DEFAULT_MAX_TOKENS)
        .min(MAX_OUTPUT_TOKENS);
    let CanonicalPrefill {
        prefill: prepared_prefill,
        replay_messages,
        mut shifted,
    } = prepare_with_context_policy(runtime, request, requested)?;
    let prompt_positions = prepared_prefill.positions;
    state.sessions.create(
        session_id,
        prompt_positions as u64,
        format!("{}-decode", runtime.backend),
        Origin::Created,
    );
    let _active_session = ActiveSession {
        state,
        id: session_id,
    };
    let prompt_witnesses = &prepared_prefill.witnesses;
    let available = runtime.max_context.saturating_sub(prompt_positions);
    if available == 0 && requested > 0 {
        return Err(ChatError::BadRequest(format!(
            "prompt uses {} tokens, exhausting the {} token session",
            prompt_positions, runtime.max_context
        )));
    }
    let max_tokens = requested.min(available);
    let sampling = sampling_params(request)?;
    let sampler_config_sha256 = sampler_config_sha256(request);
    let mut random_seed = request.seed.unwrap_or_else(entropy_seed);
    let stop = request
        .stop
        .as_ref()
        .map(StopField::values)
        .unwrap_or_default();
    // Stop and grammar eligibility must come from the GGUF identity, not
    // guessed spellings. Muse's primary EOS has an empty token piece, while
    // EOT is a separately declared end-of-generation token. Re-encoding a
    // hand-written name therefore omitted the primary EOS and made a
    // completed grammar choose the lower-probability EOT instead of matching
    // llama.cpp.
    let eos = runtime.model.config().eos_tokens.clone();
    let grammar_source = constrained_grammar_source(request)?;
    let grammar_sha256 = grammar_source
        .as_ref()
        .map(|source| Sha256::digest(source.as_bytes()).into());
    let mut grammar = grammar_source
        .as_ref()
        .map(|source| GrammarMatcher::parse(source, "root").map_err(ChatError::BadRequest))
        .transpose()?;

    let remote_route = selected_remote_route(state, runtime);

    let stateful = match (
        request.session_id.as_deref(),
        request.expected_revision,
        request.idempotency_key.as_deref(),
        request.idempotency_request_sha256,
    ) {
        (Some(id), Some(revision), Some(key), Some(request_sha256)) => {
            Some((id, revision, key, request_sha256))
        }
        (None, None, None, _) => None,
        _ => {
            return Err(ChatError::BadRequest(
                "session_id, expected_revision, Idempotency-Key, and a canonical request identity are all required for stateful generation"
                    .into(),
            ))
        }
    };
    let mut previous_bundle = None;
    if let Some((id, revision, key, request_sha256)) = stateful {
        match state
            .logical_sessions
            .begin(id, revision, key, request_sha256)
            .map_err(ChatError::Conflict)?
        {
            BeginMutation::Replay(cached) => {
                measured_emit(&cached.text, None, None)?;
                return Ok(Generated {
                    text: cached.text,
                    usage: Usage {
                        prompt_tokens: cached.prompt_tokens,
                        completion_tokens: cached.completion_tokens,
                        total_tokens: cached.prompt_tokens + cached.completion_tokens,
                        prompt_tokens_details: PromptTokensDetails { cached_tokens: 0 },
                    },
                    finish_reason: if cached.finish_reason == "stop" {
                        "stop"
                    } else {
                        "length"
                    },
                    stop_type: if cached.finish_reason == "stop" {
                        "eos"
                    } else {
                        "limit"
                    },
                    stopping_word: String::new(),
                    seed: cached.seed,
                    session_revision: Some(cached.revision),
                    logprobs: None,
                    sampled_tokens: cached.sampled_tokens,
                    context: cached.context,
                    slot_id: None,
                });
            }
            BeginMutation::Started(bundle) => previous_bundle = bundle,
        }
    }
    struct MutationGuard<'a> {
        state: &'a ServerState,
        id: Option<&'a str>,
        active: bool,
    }
    impl Drop for MutationGuard<'_> {
        fn drop(&mut self) {
            if self.active {
                if let Some(id) = self.id {
                    self.state.logical_sessions.abort(id);
                }
            }
        }
    }
    let mut mutation_guard = MutationGuard {
        state,
        id: stateful.map(|(id, _, _, _)| id),
        active: stateful.is_some(),
    };

    if let Some(bundle) = &previous_bundle {
        // A restored logical session continues the exact sampler stream that
        // was committed with its KV/logit frontier. A fresh client seed only
        // applies when creating a new frontier, never midway through one.
        random_seed = bundle.rng_seed;
        if bundle.grammar_sha256 != grammar_sha256 {
            return Err(ChatError::Conflict(
                "stored grammar identity differs from the continuing request".into(),
            ));
        }
        if bundle.sampler_config_sha256 != sampler_config_sha256 {
            return Err(ChatError::Conflict(
                "stored sampler configuration differs from the continuing request".into(),
            ));
        }
        grammar = bundle.grammar_state.clone();
    }
    let mut sampler = RequestSamplerState::new(request, random_seed as u32);
    if let Some(bundle) = &previous_bundle {
        sampler.restore(&bundle.sampler_state)?;
        let expected_dflash = bundle
            .dflash
            .as_ref()
            .and(runtime.dflash_identity_sha256.as_deref());
        let expected_vision = (!bundle.vision_rows.is_empty())
            .then_some(runtime.vision_identity.as_ref())
            .flatten();
        if bundle.model_sha256 != state.model_sha256.as_deref().unwrap_or_default()
            || bundle.tokenizer_sha256 != runtime.model.tokenizer_metadata_sha256()
            || bundle.template_sha256 != runtime.model.chat_template_sha256()
            || bundle.layout_abi != "muse-kv-layout-v1"
            || bundle.dflash_identity_sha256.as_deref() != expected_dflash
            || bundle.vision_projector_sha256.as_deref()
                != expected_vision.map(|identity| identity.projector_sha256.as_str())
            || bundle.vision_preprocessing_sha256.as_deref()
                != expected_vision.map(|identity| identity.preprocessing_sha256.as_str())
        {
            return Err(ChatError::Conflict(
                "session bundle model/template/layout/assistant/vision identity differs from this server"
                    .into(),
            ));
        }
        let exact_frontier = validate_session_lineage(
            bundle,
            request,
            &prepared_prefill.witnesses,
            runtime.raw_retain_prefix,
        )?;
        if !shifted && !exact_frontier {
            if runtime.context_policy == ContextPolicy::Error {
                return Err(ChatError::BadRequest(
                    "session continuation requires a context rebuild but context policy is error"
                        .into(),
                ));
            }
            shifted = true;
        }
    }
    let committed_context_epoch = next_context_epoch(
        previous_bundle
            .as_ref()
            .map_or(0, |bundle| bundle.context_epoch),
        shifted,
    )?;

    let mut slot = acquire_slot(state, runtime, request.id_slot)?;
    let slot_index = slot.index();
    let session = slot.session_mut();
    let mut restored_positions = None;
    let mut committed_vision_rows = previous_bundle
        .as_ref()
        .filter(|_| !shifted)
        .map_or_else(Vec::new, |bundle| bundle.vision_rows.clone());
    let mut shifted_dflash_prepared = None;
    if shifted {
        // Do not start a potentially long staging prefill for a client that
        // disconnected while waiting for its serving-slot lease.
        measured_emit("", None, None)?;
        let batch = prepared_prefill.materialize(runtime)?;
        let shifted_vision_rows = vision_rows(&batch);
        let mut staging = match runtime.staging.try_lock() {
            Ok(staging) => staging,
            Err(std::sync::TryLockError::WouldBlock) => return Err(ChatError::Overloaded),
            Err(std::sync::TryLockError::Poisoned(_)) => {
                runtime.slots.latch_unhealthy();
                return Err(ChatError::Unavailable);
            }
        };
        staging.reset();
        if max_tokens > 0 && dflash_sampling_compatible(request) {
            match (runtime.dflash.as_ref(), runtime.dflash_staging.as_ref()) {
                (Some(dflash_slots), Some(hidden_dflash)) => {
                    let mut hidden_dflash =
                        hidden_dflash.try_lock().map_err(|error| match error {
                            std::sync::TryLockError::WouldBlock => ChatError::Overloaded,
                            std::sync::TryLockError::Poisoned(_) => accelerator_failure(runtime),
                        })?;
                    let mut live_dflash = dflash_slots[slot_index].lock().map_err(|_| {
                        runtime.slots.latch_unhealthy();
                        ChatError::Unavailable
                    })?;
                    let prepared = match sampling {
                        Some(params) => hidden_dflash
                            .primary_mut()
                            .prepare_sampled_batch_with_rng(
                                &runtime.model,
                                &mut staging,
                                batch,
                                params,
                                &mut sampler.distribution_rng,
                            )
                            .map(ShiftedDFlashPrepared::Sampled),
                        None => hidden_dflash
                            .primary_mut()
                            .prepare_greedy_batch(&runtime.model, &mut staging, batch)
                            .map(ShiftedDFlashPrepared::Greedy),
                    }
                    .map_err(|_| accelerator_failure(runtime))?;
                    swap_staging_pair_on_success(
                        session,
                        &mut staging,
                        &mut *live_dflash,
                        &mut *hidden_dflash,
                        Ok::<(), ChatError>(()),
                    )?;
                    shifted_dflash_prepared = Some(prepared);
                }
                (None, None) => {
                    let prepared = staging
                        .prefill(batch)
                        .map(|_| ())
                        .map_err(|_| accelerator_failure(runtime));
                    swap_staging_on_success(session, &mut staging, prepared)?;
                }
                _ => return Err(accelerator_failure(runtime)),
            }
        } else {
            let prepared = staging
                .prefill(batch)
                .map(|_| ())
                .map_err(|_| accelerator_failure(runtime));
            swap_staging_on_success(session, &mut staging, prepared)?;
        }
        // The old serving generation is now the hidden owner. Empty it only
        // after the infallible ownership swap; no failure path can have touched
        // the live session that was committed before this rebuild.
        staging.reset();
        restored_positions = Some(prompt_positions);
        committed_vision_rows = shifted_vision_rows;
    }
    if let Some(bundle) = previous_bundle.as_ref().filter(|_| !shifted) {
        session
            .install_cache_snapshot(&bundle.target)
            .and_then(|()| session.install_restored_logits(&bundle.target_logits))
            .map_err(|error| ChatError::Engine(error.to_string()))?;
        restored_positions = Some(bundle.target.position as usize);
    }
    let mut committed_dflash = None;
    let mut result = (|| {
        // The lease can have been waited on for seconds; a client that left
        // in the meantime must not start a generation.
        measured_emit("", None, None)?;
        let mut dflash_route_exhausted = false;
        // Set when the reuse ladder served the whole remote prompt locally:
        // the local DFlash routes below would rebuild the installed prefix.
        let mut remote_reuse_hit = false;
        // Each speculative backend attempt starts at the same exact draw.
        // A route that fails before publishing a prefix must not perturb the
        // target-only fallback or a later backend's sampled continuation.
        let dflash_rng_checkpoint = sampler.distribution_rng.clone();
        if let Some(prepared) = shifted_dflash_prepared.take() {
            let dflash_slots = runtime
                .dflash
                .as_ref()
                .expect("shifted DFlash preparation requires serving assistants");
            let mut dflash = dflash_slots[slot_index].lock().map_err(|_| {
                runtime.slots.latch_unhealthy();
                ChatError::Unavailable
            })?;
            match generate_dflash_prepared_shift(
                state,
                dflash.primary_mut(),
                &runtime.model,
                session,
                prepared,
                max_tokens,
                sampling,
                random_seed,
                &mut sampler.distribution_rng,
                &eos,
                session_id,
                prompt_positions,
                &mut |token, text| measured_emit(text, token, None),
            ) {
                Ok((mut generated, stats)) => {
                    generated.context = session.token_history().to_vec();
                    committed_dflash = Some(dflash.primary_mut().export_context_snapshot());
                    state.record_dflash_stats(&stats);
                    return Ok(generated);
                }
                Err(error) => {
                    return Err(error.into_error(runtime));
                }
            }
        }
        // A compatible shifted request was already prepared and returned from
        // the paired hidden target+DFlash path above. Incompatible constrained
        // or logprob requests deliberately remain exact target-only.
        if max_tokens > 0 && !shifted && dflash_sampling_compatible(request) {
            if let Some(dflash_slots) = runtime.dflash.as_ref() {
                let dflash = &dflash_slots[slot_index];
                let prompt_tokens = prepared_prefill.token_only();
                let dflash_prefill = match prompt_tokens.as_ref() {
                    Some(tokens) => PrefillBatch::tokens(tokens.clone()),
                    None => {
                        let batch = prepared_prefill.materialize(runtime)?;
                        committed_vision_rows = vision_rows(&batch);
                        batch
                    }
                };
                let mut dflash = dflash.lock().map_err(|_| {
                    runtime.slots.latch_unhealthy();
                    ChatError::Unavailable
                })?;
                if let Some(snapshot) = previous_bundle
                    .as_ref()
                    .and_then(|bundle| bundle.dflash.as_ref())
                {
                    dflash
                        .primary_mut()
                        .validate_context_snapshot(snapshot)
                        .map_err(|error| ChatError::Conflict(error.to_string()))?;
                }
                let mut route_failures = Vec::new();
                if let Some(remote) = remote_route {
                    if prompt_witnesses.len() < 2 {
                        if remote.mode() == RemotePrefillMode::Required {
                            return Err(ChatError::BadRequest(
                                "remote prefill requires at least two prompt tokens".into(),
                            ));
                        }
                    } else {
                        let boundary = *prompt_witnesses.last().expect("length checked");
                        if boundary == muser_engine::EMBEDDING_POSITION_WITNESS {
                            return Err(ChatError::BadRequest(
                                "remote DFlash prompt must end at a text boundary".into(),
                            ));
                        }
                        // The reuse ladder is consulted before any remote
                        // work: an exact local cut skips the handoff and is
                        // served from the resident cache, an aligned prefix
                        // stays installed to arm a delta, and anything else
                        // pays the full transfer as before.
                        let reuse = consult_reuse_before_remote(
                            state,
                            runtime,
                            session,
                            prompt_tokens.as_deref(),
                        )?;
                        let (skip_remote, armed_cut) = match reuse {
                            Some(RemoteHandoffReuse::SkipRemote { matched, source }) => {
                                record_reuse_source(state, source, matched);
                                restored_positions = Some(matched);
                                remote_reuse_hit = true;
                                (true, 0)
                            }
                            // The armed prefix stays in the session: the
                            // receiver validates the producer's `prefix_cut`
                            // against it, and a full answer atomically
                            // replaces it.
                            Some(RemoteHandoffReuse::ArmDelta { prefix_cut, .. }) => {
                                dflash.primary_mut().reset();
                                (false, prefix_cut)
                            }
                            None => {
                                session.reset();
                                dflash.primary_mut().reset();
                                (false, 0)
                            }
                        };
                        let receive = (!skip_remote).then(|| {
                            remote.receive(
                                session,
                                Some(dflash.primary_mut()),
                                prompt_witnesses,
                                prepared_prefill.remote_multimodal.clone(),
                                runtime.max_context,
                            )
                        });
                        match receive {
                            None => {}
                            Some(Ok(receipt)) => {
                                state.record_remote_transfer(&receipt);
                                // Both fallback attempts start from detached,
                                // authenticated state. A failed ANE attempt can
                                // never leave rows in the live target generation.
                                let target_snapshot = session
                                    .export_cache_snapshot()
                                    .map_err(|error| ChatError::Engine(error.to_string()))?;
                                let dflash_snapshot =
                                    dflash.primary_mut().export_context_snapshot();
                                let mut prompt_published = false;
                                let mut publish_prompt = |session: &Session| {
                                    publish_remote_prompt_cut(
                                        runtime,
                                        state,
                                        session,
                                        prompt_positions,
                                        receipt.installed_bytes,
                                        armed_cut,
                                        &mut prompt_published,
                                    )
                                };
                                sampler.distribution_rng = dflash_rng_checkpoint.clone();
                                match generate_dflash_installed(
                                    state,
                                    dflash.primary_mut(),
                                    &runtime.model,
                                    session,
                                    boundary,
                                    max_tokens,
                                    sampling,
                                    random_seed,
                                    &mut sampler.distribution_rng,
                                    &eos,
                                    session_id,
                                    prompt_positions,
                                    &mut publish_prompt,
                                    &mut |token, text| measured_emit(text, token, None),
                                ) {
                                    Ok((mut generated, stats)) => {
                                        generated.context = session.token_history().to_vec();
                                        committed_dflash =
                                            Some(dflash.primary_mut().export_context_snapshot());
                                        state.record_dflash_stats(&stats);
                                        return Ok(generated);
                                    }
                                    Err(error) => {
                                        let detail = error.to_string();
                                        if let Some(terminal) = error.into_terminal(runtime) {
                                            return Err(terminal);
                                        }
                                        let route = dflash.primary_route();
                                        state.record_dflash_route_failure(route);
                                        route_failures.push(format!("{route}: {detail}"));
                                        session.install_cache_snapshot(&target_snapshot).map_err(
                                            |restore| ChatError::Engine(restore.to_string()),
                                        )?;
                                    }
                                }
                                let fallback =
                                    match dflash.fallback_mut(&runtime.model, runtime.backend) {
                                        Ok(value) => value,
                                        Err(error) => {
                                            state.record_dflash_route_failure("metal");
                                            route_failures.push(format!("metal load: {error}"));
                                            None
                                        }
                                    };
                                if let Some(fallback) = fallback {
                                    fallback.reset();
                                    fallback
                                        .install_context_snapshot(&dflash_snapshot)
                                        .map_err(|error| ChatError::Engine(error.to_string()))?;
                                    sampler.distribution_rng = dflash_rng_checkpoint.clone();
                                    match generate_dflash_installed(
                                        state,
                                        fallback,
                                        &runtime.model,
                                        session,
                                        boundary,
                                        max_tokens,
                                        sampling,
                                        random_seed,
                                        &mut sampler.distribution_rng,
                                        &eos,
                                        session_id,
                                        prompt_positions,
                                        &mut publish_prompt,
                                        &mut |token, text| measured_emit(text, token, None),
                                    ) {
                                        Ok((mut generated, stats)) => {
                                            generated.context = session.token_history().to_vec();
                                            committed_dflash =
                                                Some(fallback.export_context_snapshot());
                                            state.record_dflash_stats(&stats);
                                            return Ok(generated);
                                        }
                                        Err(error) => {
                                            let detail = error.to_string();
                                            if let Some(terminal) = error.into_terminal(runtime) {
                                                return Err(terminal);
                                            }
                                            state.record_dflash_route_failure("metal");
                                            route_failures.push(format!("metal: {detail}"));
                                        }
                                    }
                                }
                                if remote.mode() == RemotePrefillMode::Required {
                                    return Err(ChatError::Engine(format!(
                                        "required combined remote DFlash routes failed: {}",
                                        route_failures.join("; ")
                                    )));
                                }
                                session.reset();
                            }
                            Some(Err(error)) if remote.mode() == RemotePrefillMode::Required => {
                                state.record_remote_failure(&error);
                                return Err(ChatError::Engine(format!(
                                    "required remote prefill failed: {error}"
                                )));
                            }
                            Some(Err(error)) => {
                                state.record_remote_failure(&error);
                                state.record_remote_fallback();
                                session.reset();
                                dflash.primary_mut().reset();
                            }
                        }
                    }
                }
                // A warm reuse hit installed the prompt already; the local
                // DFlash routes would reset and rebuild it.
                if !remote_reuse_hit {
                    sampler.distribution_rng = dflash_rng_checkpoint.clone();
                    match generate_dflash_local(
                        state,
                        dflash.primary_mut(),
                        &runtime.model,
                        session,
                        dflash_prefill.clone(),
                        max_tokens,
                        sampling,
                        random_seed,
                        &mut sampler.distribution_rng,
                        &eos,
                        session_id,
                        prompt_positions,
                        &mut |token, text| measured_emit(text, token, None),
                    ) {
                        Ok((mut generated, stats)) => {
                            generated.context = session.token_history().to_vec();
                            committed_dflash = Some(dflash.primary_mut().export_context_snapshot());
                            state.record_dflash_stats(&stats);
                            return Ok(generated);
                        }
                        Err(error) => {
                            let detail = error.to_string();
                            if let Some(terminal) = error.into_terminal(runtime) {
                                return Err(terminal);
                            }
                            let route = dflash.primary_route();
                            state.record_dflash_route_failure(route);
                            route_failures.push(format!("{route}: {detail}"));
                        }
                    }
                    let fallback = match dflash.fallback_mut(&runtime.model, runtime.backend) {
                        Ok(value) => value,
                        Err(error) => {
                            state.record_dflash_route_failure("metal");
                            route_failures.push(format!("metal load: {error}"));
                            None
                        }
                    };
                    if let Some(fallback) = fallback {
                        sampler.distribution_rng = dflash_rng_checkpoint.clone();
                        match generate_dflash_local(
                            state,
                            fallback,
                            &runtime.model,
                            session,
                            dflash_prefill.clone(),
                            max_tokens,
                            sampling,
                            random_seed,
                            &mut sampler.distribution_rng,
                            &eos,
                            session_id,
                            prompt_positions,
                            &mut |token, text| measured_emit(text, token, None),
                        ) {
                            Ok((mut generated, stats)) => {
                                generated.context = session.token_history().to_vec();
                                committed_dflash = Some(fallback.export_context_snapshot());
                                state.record_dflash_stats(&stats);
                                return Ok(generated);
                            }
                            Err(error) => {
                                let detail = error.to_string();
                                if let Some(terminal) = error.into_terminal(runtime) {
                                    return Err(terminal);
                                }
                                state.record_dflash_route_failure("metal");
                                route_failures.push(format!("metal: {detail}"));
                            }
                        }
                    }
                    // The final route is exact target-only generation. It restarts
                    // from the prompt below and is visible through failure counters;
                    // no partial assistant token has been streamed to the client.
                    let _ = route_failures;
                    sampler.distribution_rng = dflash_rng_checkpoint;
                    dflash_route_exhausted = true;
                }
            }
        }
        let prompt_token_ids = prepared_prefill.token_only();
        let target_prefill_started = Instant::now();
        if restored_positions.is_none() && request.cache_prompt {
            if let Some(tokens) = prompt_token_ids.as_deref() {
                let retained = session.token_history();
                if !retained.is_empty()
                    && tokens.starts_with(retained)
                    && session.cached_logits().is_some()
                {
                    restored_positions = Some(retained.len());
                }
            }
        }
        if restored_positions.is_none() {
            session.reset();
        }
        let mut logits = None;
        // A cut-aligned tier prefix installed by the ladder consult below
        // arms the remote handoff as a delta instead of paying a local
        // suffix prefill.
        let mut armed_prefix_cut = None;

        if restored_positions == Some(prompt_positions) {
            logits = Some(
                session
                    .cached_logits()
                    .ok_or_else(|| {
                        ChatError::Engine("restored session omitted final logits".into())
                    })?
                    .to_vec(),
            );
        } else if let (Some(matched), Some(tokens)) =
            (restored_positions, prompt_token_ids.as_deref())
        {
            logits = if matched == tokens.len() {
                Some(
                    session
                        .cached_logits()
                        .ok_or_else(|| {
                            ChatError::Engine("restored session omitted final logits".into())
                        })?
                        .to_vec(),
                )
            } else if runtime.prefix_cache_enabled {
                None
            } else {
                state.economics.record_session_continuation(matched);
                state
                    .economics
                    .record_prefill_suffix((tokens.len() - matched) as u64);
                Some(prefill_token_suffix(
                    runtime, state, session, tokens, matched,
                )?)
            };
        }

        // Lookup order is local exact state before network work. A resident
        // ancestor computes only its suffix; an exact-final hit must carry
        // the witnessed final target distribution or it is rejected.
        if runtime.prefix_cache_enabled {
            if let Some(tokens) = prompt_token_ids.as_deref() {
                let reuse = recovered_lock(state, &runtime.prefix_reuse)
                    .0
                    .prepare(session, tokens)
                    .map_err(|error| ChatError::Engine(error.to_string()))?;
                if reuse.source != CacheSource::Miss {
                    record_reuse_source(state, reuse.source, reuse.matched_tokens);
                    logits = if reuse.matched_tokens == tokens.len() {
                        Some(
                            session
                                .cached_logits()
                                .ok_or_else(|| {
                                    ChatError::Engine(
                                        "exact prefix hit omitted its final target logits".into(),
                                    )
                                })?
                                .to_vec(),
                        )
                    } else if let Some(cut) = remote_route.and_then(|_| {
                        arm_remote_delta(reuse.source, reuse.matched_tokens, tokens.len())
                    }) {
                        // The installed aligned prefix stays in the session
                        // and arms the handoff below as a delta; only the
                        // suffix crosses the wire. A full producer answer
                        // atomically replaces the held cut.
                        armed_prefix_cut = Some(cut);
                        None
                    } else {
                        state
                            .economics
                            .record_prefill_suffix((tokens.len() - reuse.matched_tokens) as u64);
                        Some(prefill_token_suffix(
                            runtime,
                            state,
                            session,
                            tokens,
                            reuse.matched_tokens,
                        )?)
                    };
                }
            }
        }

        // Remote is consulted only after an exact local miss. It holds the
        // final prompt token so the Mac obtains the first logits locally.
        if logits.is_none() && restored_positions.is_none() {
            if let Some(remote) = remote_route {
                if prompt_witnesses.len() >= 2 {
                    let boundary = *prompt_witnesses.last().expect("length checked");
                    if boundary == muser_engine::EMBEDDING_POSITION_WITNESS {
                        return Err(ChatError::Engine(
                            "remote prompt ended with an embedding instead of a text boundary"
                                .into(),
                        ));
                    }
                    // An armed delta keeps the installed prefix in the
                    // session: the receiver validates the producer's
                    // `prefix_cut` against it, and a full answer atomically
                    // replaces it. Anything else starts from an empty
                    // session as before.
                    if armed_prefix_cut.is_none() {
                        session.reset();
                    }
                    match remote.receive(
                        session,
                        None,
                        prompt_witnesses,
                        prepared_prefill.remote_multimodal.clone(),
                        runtime.max_context,
                    ) {
                        Ok(receipt) => {
                            state.record_remote_transfer(&receipt);
                            publish_durable_cut(runtime, state, session)?;
                            let decoded = runtime
                                .decode_batcher
                                .decode(slot_index, session, DecodeInput { token_id: boundary })
                                .map_err(|_| accelerator_failure(runtime))?;
                            logits = Some(decoded.logits);
                            // Disaggregated prefill is work another node did,
                            // not a cache tier this node read: it is attributed
                            // to the handoff with the bytes actually installed.
                            // An armed delta prefilled only the suffix remotely.
                            state.economics.record_disagg_prefill(
                                prompt_witnesses.len() - 1 - armed_prefix_cut.unwrap_or(0),
                                receipt.installed_bytes,
                            );
                            state.economics.record_prefill_suffix(1);
                        }
                        Err(error) if remote.mode() == RemotePrefillMode::Required => {
                            state.record_remote_failure(&error);
                            return Err(ChatError::Engine(format!(
                                "required remote prefill failed: {error}"
                            )));
                        }
                        Err(error) => {
                            state.record_remote_failure(&error);
                            state.record_remote_fallback();
                            session.reset();
                        }
                    }
                } else if remote.mode() == RemotePrefillMode::Required {
                    return Err(ChatError::BadRequest(
                        "required remote prefill needs at least two decoder positions".into(),
                    ));
                }
            }
        }

        if logits.is_none() {
            session.reset();
            logits = Some(if let Some(tokens) = prompt_token_ids.as_deref() {
                prefill_token_suffix(runtime, state, session, tokens, 0)?
            } else {
                let batch = prepared_prefill.materialize(runtime)?;
                committed_vision_rows = vision_rows(&batch);
                session
                    .prefill(batch)
                    .map_err(|_| accelerator_failure(runtime))?
                    .last_logits()
                    .to_vec()
            });
            state.economics.record_prefill_miss(prompt_positions as u64);
        }

        // Publish the completed prompt cut before decode extends the session.
        // Publication failures are loud: silently serving while omitting a
        // promised cache generation would make cache metrics untrustworthy.
        if runtime.prefix_cache_enabled && prompt_token_ids.is_some() {
            let (mut cache, _) = recovered_lock(state, &runtime.prefix_reuse);
            cache
                .publish_resident(session)
                .map_err(|error| ChatError::Engine(error.to_string()))?;
            if cache.has_durable() {
                let generation = state.durable_generation.fetch_add(1, Ordering::Relaxed);
                cache
                    .publish_durable(session, generation)
                    .map_err(|error| ChatError::Engine(error.to_string()))?;
            }
        }
        let mut current_logits = logits.expect("all local/remote paths assign logits");
        // Prefill can run for seconds on a long prompt; decoding for a client
        // that hung up in the meantime only holds the lease against the next.
        measured_emit("", None, None)?;
        state.record_phase(
            "prefill",
            target_prefill_started
                .elapsed()
                .as_nanos()
                .min(u64::MAX as u128) as u64,
        );
        if max_tokens == 0 {
            return Ok(Generated {
                text: String::new(),
                usage: Usage {
                    prompt_tokens: prompt_positions,
                    completion_tokens: 0,
                    total_tokens: prompt_positions,
                    prompt_tokens_details: PromptTokensDetails { cached_tokens: 0 },
                },
                finish_reason: "length",
                stop_type: "limit",
                stopping_word: String::new(),
                seed: random_seed,
                session_revision: None,
                logprobs: request.logprobs.unwrap_or(false).then_some(ChoiceLogprobs {
                    content: Vec::new(),
                }),
                sampled_tokens: Vec::new(),
                context: session.token_history().to_vec(),
                slot_id: Some(slot_index),
            });
        }
        let reasoning_end_tokens = runtime
            .model
            .encode_with_options("<|eom|>", true)
            .into_iter()
            .collect::<VecDeque<_>>();
        if request.reasoning_control && reasoning_end_tokens.is_empty() {
            return Err(ChatError::Engine(
                "pinned tokenizer cannot encode the reasoning-end marker".into(),
            ));
        }
        let mut forced_reasoning = VecDeque::new();
        let mut reasoning_closed = false;
        let sampling_started = Instant::now();
        let mut next = take_forced_reasoning_token(
            request,
            &reasoning_end_tokens,
            &mut forced_reasoning,
            reasoning_closed,
        )
        .map_or_else(
            || {
                sample_or_argmax(
                    &current_logits,
                    sampling,
                    request,
                    session.token_history(),
                    &mut sampler,
                    grammar.as_ref(),
                    &eos,
                    &runtime.model,
                    state,
                )
            },
            Ok,
        )?;
        state.record_phase(
            "sampling",
            sampling_started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        );
        let mut detokenizer = runtime.model.streaming_detokenizer();
        let mut stop_filter = StopFilter::new(stop);
        let mut atem_completion = request.tools.as_ref().map(|_| AtemStreamParser::default());
        let mut text = String::new();
        let mut phase_tail = String::new();
        let mut completion_tokens = 0usize;
        let mut sampled_tokens = Vec::with_capacity(max_tokens);
        let mut token_logprobs = Vec::new();
        let mut last_stream_logprob = None;
        let mut stopped = false;
        let mut stopped_on_eos = false;
        let mut stopped_on_time = false;
        let mut prediction_started = None;
        let stream_decode_profile =
            std::env::var("MUSER_STREAM_DECODE_PROFILE").as_deref() == Ok("1");

        let gpu_greedy = runtime.backend == "metal"
            && runtime.slots.len() == 1
            && request.session_id.is_none()
            && sampling.is_none()
            && request.ignore_eos
            && request.stop.as_ref().is_none_or(StopField::is_empty)
            && grammar.is_none()
            && request.tools.is_none()
            && !request.reasoning_control
            && request.logprobs != Some(true)
            && request.t_max_predict_ms.is_none_or(|value| value <= 0)
            && std::env::var_os("MUSER_CROSS_VENDOR_QK").is_none()
            && !stream_decode_profile;

        if gpu_greedy {
            const GREEDY_BLOCK: usize = 16;
            let excluded = eos.to_vec();
            let mut remaining = max_tokens;
            while remaining > 0 {
                let block = remaining.min(GREEDY_BLOCK);
                sampled_tokens.push(next);
                completion_tokens += 1;
                prediction_started.get_or_insert_with(Instant::now);
                if completion_tokens == 1 {
                    state.set_active_phase("decode");
                }
                state.record_decode_tokens(1);
                state
                    .sessions
                    .set_tokens(session_id, (prompt_positions + completion_tokens) as u64);
                let detokenize_started = Instant::now();
                let piece = detokenizer.push_token(next);
                text.push_str(&piece);
                state.record_phase(
                    "detokenization",
                    detokenize_started
                        .elapsed()
                        .as_nanos()
                        .min(u64::MAX as u128) as u64,
                );
                measured_emit(&piece, Some(next), None)?;

                let mut callback_error = None;
                let result = session
                    .decode_greedy_block(next, block, &excluded, |produced| {
                        sampled_tokens.push(produced);
                        completion_tokens += 1;
                        state.record_decode_tokens(1);
                        state
                            .sessions
                            .set_tokens(session_id, (prompt_positions + completion_tokens) as u64);
                        let detokenize_started = Instant::now();
                        let piece = detokenizer.push_token(produced);
                        text.push_str(&piece);
                        state.record_phase(
                            "detokenization",
                            detokenize_started
                                .elapsed()
                                .as_nanos()
                                .min(u64::MAX as u128) as u64,
                        );
                        if let Err(error) = measured_emit(&piece, Some(produced), None) {
                            callback_error = Some(error);
                            false
                        } else {
                            true
                        }
                    })
                    .map_err(|_| accelerator_failure(runtime))?;
                if let Some(error) = callback_error {
                    return Err(error);
                }
                if result.cancelled {
                    return Err(ChatError::Cancelled);
                }
                debug_assert_eq!(result.consumed_tokens.len(), block);
                next = result.next_token;
                remaining -= block;
            }
        } else {
            for index in 0..max_tokens {
                sampled_tokens.push(next);
                if !request.ignore_eos && eos.contains(&next) {
                    // llama-server counts the selected EOG token as a prediction
                    // and exposes its empty piece (including logprobs) before the
                    // terminal frame. It does not decode that invisible token
                    // back into KV state.
                    completion_tokens += 1;
                    prediction_started.get_or_insert_with(Instant::now);
                    let current_logprob = if request.logprobs == Some(true) {
                        Some(build_token_logprob(
                            &runtime.model,
                            &current_logits,
                            next,
                            request.top_logprobs.unwrap_or(0),
                            &eos,
                        )?)
                    } else {
                        None
                    };
                    state.record_decode_tokens(1);
                    state
                        .sessions
                        .set_tokens(session_id, (prompt_positions + completion_tokens) as u64);
                    measured_emit("", Some(next), current_logprob.as_ref())?;
                    if let Some(logprob) = current_logprob {
                        token_logprobs.push(logprob);
                    }
                    stopped = true;
                    stopped_on_eos = true;
                    break;
                }
                if let Some(grammar) = grammar.as_mut() {
                    grammar
                        .accept_token(next, runtime.model.token_bytes(next))
                        .map_err(|error| {
                            ChatError::Engine(format!(
                                "selected token violated constrained grammar: {error}"
                            ))
                        })?;
                }
                completion_tokens += 1;
                prediction_started.get_or_insert_with(Instant::now);
                let current_logprob = if request.logprobs == Some(true) {
                    Some(build_token_logprob(
                        &runtime.model,
                        &current_logits,
                        next,
                        request.top_logprobs.unwrap_or(0),
                        &eos,
                    )?)
                } else {
                    None
                };
                if completion_tokens == 1 {
                    state.set_active_phase("decode");
                }
                state.record_decode_tokens(1);
                state
                    .sessions
                    .set_tokens(session_id, (prompt_positions + completion_tokens) as u64);
                let detokenize_started = Instant::now();
                let piece = detokenizer.push_token(next);
                phase_tail.push_str(&piece);
                if phase_tail.contains("<|eom|>") {
                    reasoning_closed = true;
                    forced_reasoning.clear();
                }
                if phase_tail.len() > 64 {
                    trim_phase_tail(&mut phase_tail);
                }
                state.record_phase(
                    "detokenization",
                    detokenize_started
                        .elapsed()
                        .as_nanos()
                        .min(u64::MAX as u128) as u64,
                );
                let mut tool_phase_complete = false;
                if let Some(parser) = atem_completion.as_mut() {
                    let events = parser.push(&piece).map_err(|error| {
                        ChatError::Engine(format!("malformed Muse ATEM output: {error}"))
                    })?;
                    for event in events {
                        if let AtemStreamEvent::ToolCall { index, call } = event {
                            validate_streamed_atem_call_indexed(request, index, &call)?;
                            tool_phase_complete = true;
                        }
                    }
                }
                let mut stream_text = String::new();
                if stop_filter.push(&piece, |safe| {
                    text.push_str(safe);
                    stream_text.push_str(safe);
                    Ok(())
                })? {
                    stopped = true;
                }
                // A closed ATEM function-call phase is a semantic terminal even
                // though Muse also uses <|eom|> between its private reasoning and
                // public action phases. Pinned llama-server stops at the former's
                // completed tool phase, not at the first EOM and not at a later
                // generic EOT. Waiting for the normal EOS set caused valid calls
                // to continue into a second assistant phase until max_tokens.
                stopped |= tool_phase_complete;
                if !stopped
                    && piece.contains('\n')
                    && request.t_max_predict_ms.is_some_and(|limit| {
                        limit > 0
                            && prediction_started.is_some_and(|started| {
                                started.elapsed() >= Duration::from_millis(limit as u64)
                            })
                    })
                {
                    stopped = true;
                    stopped_on_time = true;
                }
                let emit_started = stream_decode_profile.then(Instant::now);
                measured_emit(&stream_text, Some(next), current_logprob.as_ref())?;
                let emit_ns = emit_started
                    .map(|started| started.elapsed().as_nanos().min(u64::MAX as u128) as u64);
                if let Some(logprob) = current_logprob {
                    last_stream_logprob = Some(logprob.clone());
                    token_logprobs.push(logprob);
                }
                // Commit every selected, visible token to target state before a
                // stop or length boundary can finish the request. Previously the
                // terminal token was emitted but never consumed, leaving saved
                // state one token behind its advertised revision.
                let decode_started = stream_decode_profile.then(Instant::now);
                let decoded = runtime
                    .decode_batcher
                    .decode(slot_index, session, DecodeInput { token_id: next })
                    .map_err(|_| accelerator_failure(runtime))?;
                let decode_total_ns = decode_started
                    .map(|started| started.elapsed().as_nanos().min(u64::MAX as u128) as u64);
                let decode_diagnostics = decoded.diagnostics.clone();
                if stopped {
                    break;
                }
                let mut sampling_after_decode_ns = None;
                if index + 1 < max_tokens {
                    current_logits = decoded.logits;
                    let sampling_started = Instant::now();
                    next = take_forced_reasoning_token(
                        request,
                        &reasoning_end_tokens,
                        &mut forced_reasoning,
                        reasoning_closed,
                    )
                    .map_or_else(
                        || {
                            sample_or_argmax(
                                &current_logits,
                                sampling,
                                request,
                                session.token_history(),
                                &mut sampler,
                                grammar.as_ref(),
                                &eos,
                                &runtime.model,
                                state,
                            )
                        },
                        Ok,
                    )?;
                    let elapsed =
                        sampling_started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
                    state.record_phase("sampling", elapsed);
                    sampling_after_decode_ns = Some(elapsed);
                }
                if stream_decode_profile {
                    let diagnostics = decode_diagnostics.ok_or_else(|| {
                        ChatError::Engine(
                            "stream decode profiling requested but engine diagnostics are absent"
                                .into(),
                        )
                    })?;
                    let accounted_engine_ns = diagnostics
                        .model_prepare_ns
                        .saturating_add(diagnostics.model_encode_ns)
                        .saturating_add(diagnostics.encoder_end_ns)
                        .saturating_add(diagnostics.command_commit_ns)
                        .saturating_add(diagnostics.gpu_wait_ns)
                        .saturating_add(diagnostics.logits_readback_ns)
                        .saturating_add(diagnostics.finite_scan_ns)
                        .saturating_add(diagnostics.argmax_ns)
                        .saturating_add(diagnostics.result_clone_ns);
                    eprintln!(
                        "[muser-stream-decode-profile] {}",
                        serde_json::json!({
                            "schema": "muser.stream-decode-profile.v1",
                            "session_id": session_id,
                            "token_index": index,
                            "input_token": decoded.input_token,
                            "engine_argmax_token": decoded.next_token,
                            "decode_total_ns": decode_total_ns,
                            "batcher_unaccounted_ns": decode_total_ns
                                .map(|total| total.saturating_sub(accounted_engine_ns)),
                            "emit_ns": emit_ns,
                            "sampling_after_decode_ns": sampling_after_decode_ns,
                            "model_prepare_ns": diagnostics.model_prepare_ns,
                            "model_encode_ns": diagnostics.model_encode_ns,
                            "encoder_end_ns": diagnostics.encoder_end_ns,
                            "command_commit_ns": diagnostics.command_commit_ns,
                            "gpu_wait_ns": diagnostics.gpu_wait_ns,
                            "logits_readback_ns": diagnostics.logits_readback_ns,
                            "finite_scan_ns": diagnostics.finite_scan_ns,
                            "argmax_ns": diagnostics.argmax_ns,
                            "result_clone_ns": diagnostics.result_clone_ns,
                        })
                    );
                }
            }
        }

        if !stopped {
            let detokenize_started = Instant::now();
            let tail = detokenizer.flush();
            state.record_phase(
                "detokenization",
                detokenize_started
                    .elapsed()
                    .as_nanos()
                    .min(u64::MAX as u128) as u64,
            );
            if stop_filter.push(&tail, |safe| {
                text.push_str(safe);
                measured_emit(safe, None, last_stream_logprob.as_ref())
            })? {
                stopped = true;
            } else {
                stop_filter.finish(|safe| {
                    text.push_str(safe);
                    measured_emit(safe, None, last_stream_logprob.as_ref())
                })?;
            }
        }
        if dflash_route_exhausted {
            state
                .dflash_fallback_tokens
                .fetch_add(completion_tokens as u64, Ordering::Relaxed);
        }
        Ok(Generated {
            text,
            usage: Usage {
                prompt_tokens: prompt_positions,
                completion_tokens,
                total_tokens: prompt_positions + completion_tokens,
                prompt_tokens_details: PromptTokensDetails { cached_tokens: 0 },
            },
            finish_reason: if stopped && !stopped_on_time {
                "stop"
            } else {
                "length"
            },
            stop_type: if stopped_on_eos {
                "eos"
            } else if stop_filter.matched_stop().is_some() {
                "word"
            } else {
                "limit"
            },
            stopping_word: stop_filter.matched_stop().unwrap_or_default().into(),
            seed: random_seed,
            session_revision: None,
            logprobs: request.logprobs.unwrap_or(false).then_some(ChoiceLogprobs {
                content: token_logprobs,
            }),
            sampled_tokens,
            context: session.token_history().to_vec(),
            slot_id: None,
        })
    })();
    if let Ok(generated) = &mut result {
        generated.slot_id = Some(slot_index);
    }
    if let Ok(generated) = &result {
        if let Err(error) = validate_generated_atem(
            request,
            &generated.text,
            generated.finish_reason == "length",
        ) {
            result = Err(error);
        }
    }
    let mut commit_error = None;
    if let (Ok(generated), Some((id, expected_revision, key, request_sha256))) =
        (&mut result, stateful)
    {
        let committed = (|| -> Result<u64, ChatError> {
            let target = session
                .export_cache_snapshot()
                .map_err(|error| ChatError::Engine(error.to_string()))?;
            let target_logits = session
                .cached_logits()
                .ok_or_else(|| ChatError::Engine("committed session omitted target logits".into()))?
                .to_vec();
            let bundle = SessionBundle {
                schema: "muser.session-bundle.v3".into(),
                session_id: id.into(),
                revision: expected_revision,
                context_epoch: committed_context_epoch,
                model_sha256: state.model_sha256.clone().unwrap_or_default(),
                tokenizer_sha256: runtime.model.tokenizer_metadata_sha256(),
                template_sha256: runtime.model.chat_template_sha256(),
                layout_abi: "muse-kv-layout-v1".into(),
                dflash_identity_sha256: committed_dflash
                    .as_ref()
                    .and(runtime.dflash_identity_sha256.clone()),
                vision_projector_sha256: (!committed_vision_rows.is_empty())
                    .then(|| {
                        runtime
                            .vision_identity
                            .as_ref()
                            .map(|identity| identity.projector_sha256.clone())
                    })
                    .flatten(),
                vision_preprocessing_sha256: (!committed_vision_rows.is_empty())
                    .then(|| {
                        runtime
                            .vision_identity
                            .as_ref()
                            .map(|identity| identity.preprocessing_sha256.clone())
                    })
                    .flatten(),
                position_witnesses: target.tokens.to_vec(),
                rng_seed: random_seed,
                sampler_state: sampler.snapshot(),
                sampler_config_sha256,
                sampler_history: session.token_history().to_vec(),
                detokenizer_pending: String::new(),
                stop_matcher_pending: String::new(),
                grammar_state: grammar.clone(),
                grammar_sha256,
                target,
                target_logits,
                dflash: committed_dflash,
                canonical_replay_plan_json: serde_json::to_string(&replay_messages)
                    .expect("messages serialize"),
                vision_rows: committed_vision_rows,
            };
            let cached = CachedGeneration {
                text: generated.text.clone(),
                prompt_tokens: generated.usage.prompt_tokens,
                completion_tokens: generated.usage.completion_tokens,
                finish_reason: generated.finish_reason.into(),
                seed: generated.seed,
                revision: expected_revision,
                context: session.token_history().to_vec(),
                sampled_tokens: generated.sampled_tokens.clone(),
            };
            state
                .logical_sessions
                .commit(id, expected_revision, key, request_sha256, bundle, cached)
                .map_err(ChatError::Conflict)
        })();
        match committed {
            Ok(revision) => {
                generated.session_revision = Some(revision);
                mutation_guard.active = false;
            }
            Err(error) => commit_error = Some(error),
        }
    }
    if let Some(error) = commit_error {
        result = Err(error);
    }
    // Successful requests deliberately leave the exact committed target KV,
    // logits, and token history resident so the llama-compatible physical
    // slot save action can snapshot that frontier. Every fresh stateless
    // prefill resets or explicitly reuses this state above; failed requests
    // must never return a partial or unvalidated frontier to the idle pool.
    if result.is_err() {
        session.reset();
    }
    if let (Ok(generated), Some(started)) = (&result, first_token_at) {
        state.record_request_decode(
            generated.usage.completion_tokens,
            started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        );
    }
    result
}

/// Admit a request to one of the bounded resident slots.
fn acquire_slot<'a>(
    state: &ServerState,
    runtime: &'a InferenceRuntime,
    requested: Option<usize>,
) -> Result<SlotPermit<'a>, ChatError> {
    let queue_started = Instant::now();
    state.queue_depth.fetch_add(1, Ordering::Relaxed);
    let acquired = match requested {
        Some(index) => runtime.slots.acquire_specific(index, LEASE_WAIT),
        None => runtime.slots.acquire(LEASE_WAIT),
    };
    state.queue_depth.fetch_sub(1, Ordering::Relaxed);
    state.record_phase(
        "queue",
        queue_started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
    );
    match acquired.map_err(slot_error_to_chat) {
        Ok(permit) => Ok(permit),
        Err(ChatError::Overloaded) => {
            state.overload_rejections.fetch_add(1, Ordering::Relaxed);
            Err(ChatError::Overloaded)
        }
        Err(error) => Err(error),
    }
}

fn slot_error_to_chat(error: SlotAcquireError) -> ChatError {
    match error {
        SlotAcquireError::Overloaded => ChatError::Overloaded,
        SlotAcquireError::Unhealthy => ChatError::Unavailable,
    }
}

/// A target forward can have submitted work or advanced KV before reporting
/// an error. Its state is therefore not safe to reset and reuse in-process:
/// latch the whole resident pool and make every current and later request see
/// the release contract's restart-required 503.
fn accelerator_failure(runtime: &InferenceRuntime) -> ChatError {
    runtime.slots.latch_unhealthy();
    ChatError::Unavailable
}

fn swap_staging_on_success<T, E>(
    live: &mut T,
    staging: &mut T,
    prepared: Result<(), E>,
) -> Result<(), E> {
    prepared?;
    std::mem::swap(live, staging);
    Ok(())
}

fn swap_staging_pair_on_success<T, U, E>(
    live_target: &mut T,
    staging_target: &mut T,
    live_dflash: &mut U,
    staging_dflash: &mut U,
    prepared: Result<(), E>,
) -> Result<(), E> {
    prepared?;
    std::mem::swap(live_target, staging_target);
    std::mem::swap(live_dflash, staging_dflash);
    Ok(())
}

/// Take an auxiliary lease, recovering it if a panic poisoned it. The caller
/// resets whatever engine state that lease guards; the returned flag says
/// whether it must.
fn recovered_lock<'a, T>(state: &ServerState, lease: &'a Mutex<T>) -> (MutexGuard<'a, T>, bool) {
    match lease.lock() {
        Ok(guard) => (guard, false),
        Err(poisoned) => {
            state.record_lock_recovery();
            (poisoned.into_inner(), true)
        }
    }
}

/// The remote prefill route this request may use. `Auto` skips remote while
/// the breaker's cool-down runs: after a run of failures, paying connect and
/// producer-wait on every request only delays the local prefill that ends up
/// serving it anyway. `Required` has no local fallback and is never skipped.
fn selected_remote_route<'a>(
    state: &ServerState,
    runtime: &'a InferenceRuntime,
) -> Option<&'a RemotePrefillRuntime> {
    let remote = runtime.remote_prefill.as_ref()?;
    if remote.mode() == RemotePrefillMode::Required || state.remote_route_is_open() {
        return Some(remote);
    }
    state.record_remote_fallback();
    None
}

/// The reuse ladder's answer for a prompt the remote producer would
/// otherwise prefill.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RemoteHandoffReuse {
    /// The prompt minus at most the held boundary token is installed from a
    /// local tier: skip the remote transfer entirely and decode locally.
    SkipRemote { matched: usize, source: CacheSource },
    /// A cut-aligned prefix is installed and stays in the session, arming
    /// the handoff as a delta: the producer's `prefix_cut` is validated
    /// against these held tokens at admission, and a full producer answer
    /// atomically replaces them, so arming can never graft unverified state.
    ArmDelta {
        prefix_cut: usize,
        source: CacheSource,
    },
}

/// Consult the reuse ladder before a remote handoff. `Ok(None)` runs the
/// full transfer exactly as before. A ladder error (e.g. a durable catalog
/// failure) fails the request closed rather than serving state the ladder
/// refused to authenticate.
fn consult_reuse_before_remote(
    state: &ServerState,
    runtime: &InferenceRuntime,
    session: &mut Session,
    tokens: Option<&[u32]>,
) -> Result<Option<RemoteHandoffReuse>, ChatError> {
    let Some(tokens) = tokens else {
        return Ok(None);
    };
    if !runtime.prefix_cache_enabled || tokens.len() < 2 {
        return Ok(None);
    }
    let offer = recovered_lock(state, &runtime.prefix_reuse)
        .0
        .prepare_remote(session, tokens, PREFIX_CUT_ALIGN as usize)
        .map_err(|error| ChatError::Engine(error.to_string()))?;
    Ok(remote_handoff_reuse(offer))
}

fn remote_handoff_reuse(offer: Option<RemoteReuseOffer>) -> Option<RemoteHandoffReuse> {
    match offer? {
        RemoteReuseOffer {
            action: RemoteReuseAction::ServeLocal,
            source,
            matched_tokens,
        } => Some(RemoteHandoffReuse::SkipRemote {
            matched: matched_tokens,
            source,
        }),
        RemoteReuseOffer {
            action: RemoteReuseAction::ArmDelta,
            source,
            matched_tokens,
        } => Some(RemoteHandoffReuse::ArmDelta {
            prefix_cut: matched_tokens,
            source,
        }),
        RemoteReuseOffer {
            action: RemoteReuseAction::FullTransfer,
            ..
        } => None,
    }
}

/// A partial ladder hit arms a delta handoff only when it came from a
/// fetched tier — a live session continuation prefills its suffix locally
/// for less than a handoff costs — and only when the cut lands on the
/// producer's alignment with at least the held boundary left to transfer.
fn arm_remote_delta(source: CacheSource, matched: usize, prompt: usize) -> Option<usize> {
    let fetched = matches!(
        source,
        CacheSource::Resident | CacheSource::Durable | CacheSource::Remote
    );
    let aligned = matched > 0 && matched.is_multiple_of(PREFIX_CUT_ALIGN as usize);
    (fetched && aligned && matched + 1 < prompt).then_some(matched)
}

/// Attribute a reuse-ladder hit to the tier that served it. A current-
/// session continuation fetched nothing, so counting it as a hit would
/// inflate the one number the economics panel exists to report. Restores
/// report matched positions, not the manifest byte count, so restored bytes
/// are the documented estimate rather than a measurement.
fn record_reuse_source(state: &ServerState, source: CacheSource, matched_tokens: usize) {
    match source {
        CacheSource::CurrentSession => state.economics.record_session_continuation(matched_tokens),
        CacheSource::Resident => state.economics.record_restore(
            matched_tokens as u64,
            Tier::Resident,
            RestoreBytes::Estimated,
            None,
        ),
        CacheSource::Durable => state.economics.record_restore(
            matched_tokens as u64,
            Tier::Ssd,
            RestoreBytes::Estimated,
            None,
        ),
        CacheSource::Remote => state.economics.record_restore(
            matched_tokens as u64,
            Tier::Remote,
            RestoreBytes::Estimated,
            None,
        ),
        CacheSource::Miss => {}
    }
}

struct ActiveSession<'a> {
    state: &'a ServerState,
    id: &'a str,
}

impl Drop for ActiveSession<'_> {
    fn drop(&mut self) {
        self.state.sessions.remove(self.id);
    }
}

/// Prefill a token suffix while capturing every durable interval and the
/// penultimate cut needed to regenerate exact-final logits. This is the only
/// safe way to retain an earlier full SWA tail: it is captured before the ring
/// advances, never reconstructed from a later window.
fn prefill_token_suffix(
    runtime: &InferenceRuntime,
    state: &ServerState,
    session: &mut Session,
    tokens: &[u32],
    matched: usize,
) -> Result<Vec<f32>, ChatError> {
    if matched >= tokens.len() {
        return session
            .cached_logits()
            .map(ToOwned::to_owned)
            .ok_or_else(|| ChatError::Engine("exact prefix state omitted final logits".into()));
    }
    let durable = runtime.prefix_cache_enabled
        && recovered_lock(state, &runtime.prefix_reuse).0.has_durable();
    let interval = DURABLE_FULL_INTERVAL as usize;
    let penultimate = tokens.len().saturating_sub(1);
    let mut cursor = matched;
    let mut logits = None;
    if durable && cursor > 0 && (cursor == penultimate || cursor.is_multiple_of(interval)) {
        publish_durable_cut(runtime, state, session)?;
    }
    while cursor < tokens.len() {
        let next_interval = cursor
            .checked_div(interval)
            .and_then(|block| block.checked_add(1))
            .and_then(|block| block.checked_mul(interval))
            .unwrap_or(tokens.len());
        let mut end = next_interval.min(tokens.len());
        if durable && cursor < penultimate && end > penultimate {
            end = penultimate;
        }
        if end == cursor {
            end = tokens.len();
        }
        logits = Some(
            session
                .prefill(PrefillBatch::tokens(tokens[cursor..end].to_vec()))
                .map_err(|_| accelerator_failure(runtime))?
                .last_logits()
                .to_vec(),
        );
        cursor = end;
        if durable
            && cursor < tokens.len()
            && (cursor == penultimate || cursor.is_multiple_of(interval))
        {
            publish_durable_cut(runtime, state, session)?;
        }
    }
    logits.ok_or_else(|| ChatError::Engine("token suffix produced no logits".into()))
}

fn publish_durable_cut(
    runtime: &InferenceRuntime,
    state: &ServerState,
    session: &Session,
) -> Result<(), ChatError> {
    if !runtime.prefix_cache_enabled || session.position() == 0 {
        return Ok(());
    }
    let (cache, _) = recovered_lock(state, &runtime.prefix_reuse);
    if cache.has_durable() {
        let generation = state.durable_generation.fetch_add(1, Ordering::Relaxed);
        cache
            .publish_durable(session, generation)
            .map_err(|error| ChatError::Engine(error.to_string()))?;
    }
    Ok(())
}

fn publish_remote_prompt_cut(
    runtime: &InferenceRuntime,
    state: &ServerState,
    session: &Session,
    prompt_positions: usize,
    installed_bytes: u64,
    held_prefix: usize,
    published: &mut bool,
) -> Result<(), ChatError> {
    if *published {
        return Ok(());
    }
    if session.position() != prompt_positions || session.cached_logits().is_none() {
        return Err(ChatError::Engine(
            "remote DFlash boundary did not produce an exact prompt cut".into(),
        ));
    }
    if runtime.prefix_cache_enabled {
        recovered_lock(state, &runtime.prefix_reuse)
            .0
            .publish_resident(session)
            .map_err(|error| ChatError::Engine(error.to_string()))?;
        publish_durable_cut(runtime, state, session)?;
    }
    // An armed delta prefilled only the suffix remotely; the held prefix
    // was restored from a local tier, not shipped.
    state.economics.record_disagg_prefill(
        prompt_positions.saturating_sub(1 + held_prefix),
        installed_bytes,
    );
    state.economics.record_prefill_suffix(1);
    *published = true;
    Ok(())
}

/// The speculative routes run the same verification length the snapshot
/// reports, and stop on the same token set the incremental loop does: a
/// draft that runs past EOS would otherwise emit tokens the local route
/// never would.
#[allow(clippy::too_many_arguments)]
fn run_dflash_local(
    assistant: &mut DFlashAssistant,
    model: &Model,
    session: &mut Session,
    prompt: PrefillBatch,
    max_tokens: usize,
    sampling: Option<SamplingParams>,
    rng: &mut Mt19937,
    extra_stop_tokens: &[u32],
    on_commit: &mut dyn FnMut(&[u32]) -> Result<(), DFlashRunError>,
) -> Result<(Vec<u32>, DFlashSpecStats), DFlashRunError> {
    let verify_length = dflash_verify_len();
    match sampling {
        Some(params) => assistant.generate_sampled_batch_streaming_with_rng(
            model,
            session,
            prompt,
            max_tokens,
            verify_length,
            params,
            rng,
            extra_stop_tokens,
            on_commit,
        ),
        None => {
            let prepared = assistant.prepare_greedy_batch(model, session, prompt)?;
            assistant.generate_prepared_greedy_streaming(
                model,
                session,
                prepared,
                max_tokens,
                verify_length,
                extra_stop_tokens,
                on_commit,
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_dflash_installed(
    assistant: &mut DFlashAssistant,
    model: &Model,
    session: &mut Session,
    boundary_token: u32,
    max_tokens: usize,
    sampling: Option<SamplingParams>,
    rng: &mut Mt19937,
    extra_stop_tokens: &[u32],
    on_commit: &mut dyn FnMut(&[u32]) -> Result<(), DFlashRunError>,
) -> Result<(Vec<u32>, DFlashSpecStats), DFlashRunError> {
    let verify_length = dflash_verify_len();
    match sampling {
        Some(params) => assistant.generate_sampled_from_installed_streaming_with_rng(
            model,
            session,
            boundary_token,
            max_tokens,
            verify_length,
            params,
            rng,
            extra_stop_tokens,
            on_commit,
        ),
        None => assistant.generate_greedy_from_installed_streaming(
            model,
            session,
            boundary_token,
            max_tokens,
            verify_length,
            extra_stop_tokens,
            on_commit,
        ),
    }
}

/// Couple each target-verified DFlash prefix to its observable stream
/// frontier. The engine callback runs only after target KV, target logits and
/// the DFlash hidden frontier commit the exact same prefix.
type DFlashStreamEmitter<'a> = dyn FnMut(Option<u32>, &str) -> Result<(), ChatError> + 'a;

#[allow(clippy::too_many_arguments)]
fn stream_dflash(
    state: &ServerState,
    model: &Model,
    prompt_positions: usize,
    session_id: &str,
    seed: u64,
    eos: &[u32],
    emit: &mut DFlashStreamEmitter<'_>,
    run: impl FnOnce(
        &mut dyn FnMut(&[u32]) -> Result<(), DFlashRunError>,
    ) -> Result<(Vec<u32>, DFlashSpecStats), DFlashRunError>,
) -> Result<(Generated, DFlashSpecStats), DFlashAttemptError> {
    let mut detokenizer = model.streaming_detokenizer();
    let mut text = String::new();
    let mut completion_tokens = 0usize;
    let mut sampled_tokens = Vec::new();
    let mut stopped = false;
    let mut callback_error = None;
    let mut committed = false;
    let engine_result = {
        let mut on_commit = |tokens: &[u32]| {
            committed |= !tokens.is_empty();
            for &token in tokens {
                sampled_tokens.push(token);
                completion_tokens += 1;
                if completion_tokens == 1 {
                    state.set_active_phase("decode");
                }
                state.record_decode_tokens(1);
                state
                    .sessions
                    .set_tokens(session_id, (prompt_positions + completion_tokens) as u64);
                if eos.contains(&token) {
                    stopped = true;
                    if let Err(error) = emit(Some(token), "") {
                        callback_error = Some(error);
                        return Err(DFlashRunError::Invariant(
                            "verified-prefix stream callback failed".into(),
                        ));
                    }
                    continue;
                }
                let detokenize_started = Instant::now();
                let piece = detokenizer.push_token(token);
                state.record_phase(
                    "detokenization",
                    detokenize_started
                        .elapsed()
                        .as_nanos()
                        .min(u64::MAX as u128) as u64,
                );
                text.push_str(&piece);
                if let Err(error) = emit(Some(token), &piece) {
                    callback_error = Some(error);
                    return Err(DFlashRunError::Invariant(
                        "verified-prefix stream callback failed".into(),
                    ));
                }
            }
            Ok(())
        };
        run(&mut on_commit)
    };
    if let Some(error) = callback_error {
        return Err(DFlashAttemptError {
            error,
            committed,
            callback_failed: true,
        });
    }
    let (_, stats) = engine_result.map_err(|error| DFlashAttemptError {
        error: ChatError::Engine(error.to_string()),
        committed,
        callback_failed: false,
    })?;
    let detokenize_started = Instant::now();
    let tail = detokenizer.flush();
    state.record_phase(
        "detokenization",
        detokenize_started
            .elapsed()
            .as_nanos()
            .min(u64::MAX as u128) as u64,
    );
    if !tail.is_empty() {
        text.push_str(&tail);
        // A flush is decoder text, not a newly sampled token. Fold it into
        // the last token event at the caller only when one exists; Muse's
        // tokenizer normally completes UTF-8 during `push_token`.
        emit(None, &tail).map_err(|error| DFlashAttemptError {
            error,
            committed,
            callback_failed: true,
        })?;
    }
    Ok((
        Generated {
            text,
            usage: Usage {
                prompt_tokens: prompt_positions,
                completion_tokens,
                total_tokens: prompt_positions + completion_tokens,
                prompt_tokens_details: PromptTokensDetails { cached_tokens: 0 },
            },
            finish_reason: if stopped { "stop" } else { "length" },
            stop_type: if stopped { "eos" } else { "limit" },
            stopping_word: String::new(),
            seed,
            session_revision: None,
            logprobs: None,
            sampled_tokens,
            context: Vec::new(),
            slot_id: None,
        },
        stats,
    ))
}

#[derive(Debug)]
struct DFlashAttemptError {
    error: ChatError,
    committed: bool,
    callback_failed: bool,
}

enum ShiftedDFlashPrepared {
    Greedy(DFlashPreparedGreedy),
    Sampled(DFlashPreparedSampled),
}

impl std::fmt::Display for DFlashAttemptError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.fmt(formatter)
    }
}

impl DFlashAttemptError {
    /// A stream callback failure is ordinary request cancellation. Any engine
    /// failure after a committed prefix leaves accelerator health uncertain.
    fn into_terminal(self, runtime: &InferenceRuntime) -> Option<ChatError> {
        if self.callback_failed {
            return Some(self.error);
        }
        if self.committed {
            runtime.slots.latch_unhealthy();
            return Some(ChatError::Unavailable);
        }
        None
    }

    fn into_error(self, runtime: &InferenceRuntime) -> ChatError {
        if self.callback_failed {
            return self.error;
        }
        if self.committed {
            runtime.slots.latch_unhealthy();
            return ChatError::Unavailable;
        }
        self.error
    }
}

#[allow(clippy::too_many_arguments)]
fn generate_dflash_prepared_shift(
    state: &ServerState,
    assistant: &mut DFlashAssistant,
    model: &Model,
    session: &mut Session,
    prepared: ShiftedDFlashPrepared,
    max_tokens: usize,
    sampling: Option<SamplingParams>,
    seed: u64,
    rng: &mut Mt19937,
    eos: &[u32],
    session_id: &str,
    prompt_positions: usize,
    emit: &mut DFlashStreamEmitter<'_>,
) -> Result<(Generated, DFlashSpecStats), DFlashAttemptError> {
    stream_dflash(
        state,
        model,
        prompt_positions,
        session_id,
        seed,
        eos,
        emit,
        |on_commit| match (prepared, sampling) {
            (ShiftedDFlashPrepared::Greedy(prepared), None) => assistant
                .generate_prepared_greedy_streaming(
                    model,
                    session,
                    prepared,
                    max_tokens,
                    dflash_verify_len(),
                    eos,
                    on_commit,
                ),
            (ShiftedDFlashPrepared::Sampled(prepared), Some(params)) => assistant
                .generate_prepared_sampled_streaming_with_rng(
                    model,
                    session,
                    prepared,
                    max_tokens,
                    dflash_verify_len(),
                    params,
                    rng,
                    eos,
                    on_commit,
                ),
            _ => Err(DFlashRunError::Invariant(
                "shifted DFlash preparation mode changed before commit".into(),
            )),
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn generate_dflash_local(
    state: &ServerState,
    assistant: &mut DFlashAssistant,
    model: &Model,
    session: &mut Session,
    prompt: PrefillBatch,
    max_tokens: usize,
    sampling: Option<SamplingParams>,
    seed: u64,
    rng: &mut Mt19937,
    eos: &[u32],
    session_id: &str,
    prompt_positions: usize,
    emit: &mut DFlashStreamEmitter<'_>,
) -> Result<(Generated, DFlashSpecStats), DFlashAttemptError> {
    stream_dflash(
        state,
        model,
        prompt_positions,
        session_id,
        seed,
        eos,
        emit,
        |on_commit| {
            run_dflash_local(
                assistant, model, session, prompt, max_tokens, sampling, rng, eos, on_commit,
            )
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn generate_dflash_installed(
    state: &ServerState,
    assistant: &mut DFlashAssistant,
    model: &Model,
    session: &mut Session,
    boundary_token: u32,
    max_tokens: usize,
    sampling: Option<SamplingParams>,
    seed: u64,
    rng: &mut Mt19937,
    eos: &[u32],
    session_id: &str,
    prompt_positions: usize,
    publish_prompt: &mut dyn FnMut(&Session) -> Result<(), ChatError>,
    emit: &mut DFlashStreamEmitter<'_>,
) -> Result<(Generated, DFlashSpecStats), DFlashAttemptError> {
    let prepared = match sampling {
        Some(params) => assistant
            .prepare_sampled_from_installed_with_rng(model, session, boundary_token, params, rng)
            .map(ShiftedDFlashPrepared::Sampled),
        None => assistant
            .prepare_greedy_from_installed(model, session, boundary_token)
            .map(ShiftedDFlashPrepared::Greedy),
    }
    .map_err(|error| DFlashAttemptError {
        error: ChatError::Engine(error.to_string()),
        committed: false,
        callback_failed: false,
    })?;
    publish_prompt(session).map_err(|error| DFlashAttemptError {
        error,
        committed: false,
        callback_failed: true,
    })?;
    generate_dflash_prepared_shift(
        state,
        assistant,
        model,
        session,
        prepared,
        max_tokens,
        sampling,
        seed,
        rng,
        eos,
        session_id,
        prompt_positions,
        emit,
    )
}

fn validate_request(request: &ChatRequest) -> Result<(), ChatError> {
    if request.model != MODEL_ID {
        return Err(ChatError::ModelUnavailable(request.model.clone()));
    }
    if request.messages.is_empty() {
        return Err(ChatError::BadRequest("messages must not be empty".into()));
    }
    let session_fields = usize::from(request.session_id.is_some())
        + usize::from(request.expected_revision.is_some())
        + usize::from(request.idempotency_key.is_some());
    if !matches!(session_fields, 0 | 3) {
        return Err(ChatError::BadRequest(
            "session_id, expected_revision, and Idempotency-Key are all required for stateful generation"
                .into(),
        ));
    }
    if request.muser_baseline_ttft && request.muser_prompt_token_ids.is_none() {
        return Err(ChatError::BadRequest(
            "baseline TTFT requires muser_prompt_token_ids".into(),
        ));
    }
    let n = request.n.unwrap_or(1);
    if !(1..=4).contains(&n) {
        return Err(ChatError::BadRequest("n must be in 1..=4".into()));
    }
    if n != 1 && request.session_id.is_some() {
        return Err(ChatError::BadRequest(
            "stateful generation requires n=1".into(),
        ));
    }
    validate_tools(request)?;
    validate_sampler_chain(request)?;
    if request.top_logprobs.is_some_and(|value| value > 20) {
        return Err(ChatError::BadRequest(
            "top_logprobs must be at most 20".into(),
        ));
    }
    if request.top_logprobs.is_some() && request.logprobs != Some(true) {
        return Err(ChatError::BadRequest(
            "top_logprobs requires logprobs=true".into(),
        ));
    }
    if request.stream && request.logprobs == Some(true) && request.tools.is_some() {
        return Err(ChatError::BadRequest(
            "logprobs is not supported with tools + stream".into(),
        ));
    }
    if request.seed.is_some_and(|seed| seed > u64::from(u32::MAX)) {
        return Err(ChatError::BadRequest(
            "seed must fit the pinned llama.cpp uint32 range".into(),
        ));
    }
    if request.t_max_predict_ms.is_some_and(|value| value < -1) {
        return Err(ChatError::BadRequest(
            "t_max_predict_ms must be -1 or a nonnegative integer".into(),
        ));
    }
    constrained_matcher(request)?;
    sampling_params(request)?;
    if let Some(StopField::Many(values)) = &request.stop {
        if values.len() > 4 {
            return Err(ChatError::BadRequest(
                "stop accepts at most four strings".into(),
            ));
        }
    }
    if let Some(breakers) = &request.dry_sequence_breakers {
        if breakers.is_empty()
            || breakers
                .iter()
                .any(|breaker| breaker.is_empty() || breaker.len() > 40)
        {
            return Err(ChatError::BadRequest(
                "dry_sequence_breakers must contain 1 or more strings of 1..=40 bytes".into(),
            ));
        }
    }
    Ok(())
}

const PINNED_SAMPLER_ORDER: &[&str] = &[
    "penalties",
    "dry",
    "top_n_sigma",
    "top_k",
    "typ_p",
    "top_p",
    "min_p",
    "xtc",
    "temperature",
    "adaptive_p",
];

fn validate_sampler_chain(request: &ChatRequest) -> Result<(), ChatError> {
    let Some(samplers) = &request.samplers else {
        return Ok(());
    };
    for sampler in samplers {
        if sampler == "infill" {
            return Err(ChatError::BadRequest(
                "sampler 'infill' is outside the v0.1 feature contract".into(),
            ));
        }
        if !PINNED_SAMPLER_ORDER.contains(&sampler.as_str()) {
            return Err(ChatError::BadRequest(format!(
                "unsupported sampler {sampler:?}"
            )));
        }
    }
    Ok(())
}

fn sampler_enabled(request: &ChatRequest, name: &str) -> bool {
    request.samplers.as_ref().map_or_else(
        || PINNED_SAMPLER_ORDER[..PINNED_SAMPLER_ORDER.len() - 1].contains(&name),
        |samplers| samplers.iter().any(|sampler| sampler == name),
    )
}

fn constrained_matcher(request: &ChatRequest) -> Result<Option<GrammarMatcher>, ChatError> {
    constrained_grammar_source(request)?
        .map(|source| GrammarMatcher::parse(&source, "root").map_err(ChatError::BadRequest))
        .transpose()
}

fn constrained_grammar_source(request: &ChatRequest) -> Result<Option<String>, ChatError> {
    let selected = usize::from(request.grammar.is_some())
        + usize::from(request.json_schema.is_some())
        + usize::from(request.response_format.is_some());
    if selected > 1 {
        return Err(ChatError::BadRequest(
            "grammar, json_schema, and response_format are mutually exclusive".into(),
        ));
    }
    let mut source = if let Some(grammar) = &request.grammar {
        if grammar.trim().is_empty() {
            return Err(ChatError::BadRequest("grammar must not be empty".into()));
        }
        Some(grammar.clone())
    } else if let Some(schema) = &request.json_schema {
        Some(json_schema_to_gbnf(schema).map_err(ChatError::BadRequest)?)
    } else if let Some(format) = &request.response_format {
        let object = format
            .as_object()
            .ok_or_else(|| ChatError::BadRequest("response_format must be an object".into()))?;
        let kind = object
            .get("type")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| ChatError::BadRequest("response_format.type is required".into()))?;
        match kind {
            "text" => {
                reject_unknown_keys(object, &["type"], "response_format")?;
                None
            }
            "json_object" => {
                reject_unknown_keys(object, &["type"], "response_format")?;
                Some(json_object_gbnf())
            }
            "json_schema" => {
                reject_unknown_keys(object, &["type", "json_schema"], "response_format")?;
                let envelope = object
                    .get("json_schema")
                    .and_then(serde_json::Value::as_object)
                    .ok_or_else(|| {
                        ChatError::BadRequest("response_format.json_schema is required".into())
                    })?;
                reject_unknown_keys(
                    envelope,
                    &["name", "description", "schema", "strict"],
                    "response_format.json_schema",
                )?;
                if envelope.get("name").is_none_or(|value| !value.is_string()) {
                    return Err(ChatError::BadRequest(
                        "response_format.json_schema.name must be a string".into(),
                    ));
                }
                if envelope
                    .get("description")
                    .is_some_and(|value| !value.is_string())
                    || envelope
                        .get("strict")
                        .is_some_and(|value| !value.is_boolean())
                {
                    return Err(ChatError::BadRequest(
                        "response_format.json_schema description/strict have invalid types".into(),
                    ));
                }
                let schema = envelope.get("schema").ok_or_else(|| {
                    ChatError::BadRequest("response_format.json_schema.schema is required".into())
                })?;
                Some(json_schema_to_gbnf(schema).map_err(ChatError::BadRequest)?)
            }
            other => {
                return Err(ChatError::BadRequest(format!(
                    "unsupported response_format.type {other:?}"
                )))
            }
        }
    } else {
        None
    };
    // llama-server applies response constraints to the public assistant
    // phase, while Muse may first emit a private `to=self` reasoning phase.
    // Native completion/raw-token requests still constrain byte zero onward.
    if request.muser_prompt_token_ids.is_none() {
        source = source.map(|source| muse_chat_response_grammar(&source));
    }
    if source.is_none()
        && request.tool_choice.as_ref().is_some_and(|choice| {
            matches!(choice, ToolChoice::Mode(mode) if mode == "required")
                || matches!(choice, ToolChoice::Named(_))
        })
    {
        source = Some(required_tool_gbnf(request)?);
    }
    Ok(source)
}

fn muse_chat_response_grammar(source: &str) -> String {
    let mut suffix = 0usize;
    let response_root = loop {
        let candidate = format!("muser-chat-response-root-{suffix}");
        if !grammar_has_identifier(source, &candidate) {
            break candidate;
        }
        suffix += 1;
    };
    let renamed = rename_grammar_identifier(source, "root", &response_root);
    format!(
        "root ::= muser-chat-reasoning-phase? muser-chat-answer-phase\n\
         muser-chat-reasoning-phase ::= muser-chat-assistant-prefix \" to=self<|message|>\" muser-chat-any* \"<|eom|>\"\n\
         muser-chat-answer-phase ::= muser-chat-assistant-prefix (\" to=user<|message|>\" | \"<|message|>\")? {response_root}\n\
         muser-chat-assistant-prefix ::= \"<|start|>assistant\"?\n\
         muser-chat-any ::= [\\U00000000-\\uD7FF\\uE000-\\U0010FFFF]\n\
         {renamed}"
    )
}

fn grammar_has_identifier(source: &str, target: &str) -> bool {
    rename_grammar_identifier(source, target, "") != source
}

fn rename_grammar_identifier(source: &str, from: &str, to: &str) -> String {
    let characters = source.chars().collect::<Vec<_>>();
    let mut output = String::with_capacity(source.len() + to.len());
    let mut position = 0usize;
    let mut quote = None;
    let mut escaped = false;
    while position < characters.len() {
        let character = characters[position];
        if let Some(end) = quote {
            output.push(character);
            position += 1;
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == end {
                quote = None;
            }
            continue;
        }
        if character == '"' {
            quote = Some('"');
            output.push(character);
            position += 1;
            continue;
        }
        if character == '[' {
            quote = Some(']');
            output.push(character);
            position += 1;
            continue;
        }
        if character.is_ascii_alphanumeric() || character == '_' || character == '-' {
            let start = position;
            position += 1;
            while position < characters.len()
                && (characters[position].is_ascii_alphanumeric()
                    || characters[position] == '_'
                    || characters[position] == '-')
            {
                position += 1;
            }
            let identifier = characters[start..position].iter().collect::<String>();
            output.push_str(if identifier == from { to } else { &identifier });
        } else {
            output.push(character);
            position += 1;
        }
    }
    output
}

fn required_tool_gbnf(request: &ChatRequest) -> Result<String, ChatError> {
    let selected = request
        .tool_choice
        .as_ref()
        .and_then(|choice| match choice {
            ToolChoice::Named(choice) => Some(choice.function.name.as_str()),
            ToolChoice::Mode(_) => None,
        });
    let tools = request
        .tools
        .as_ref()
        .into_iter()
        .flatten()
        .filter(|tool| selected.is_none_or(|name| tool.function.name == name))
        .collect::<Vec<_>>();
    if tools.is_empty() {
        return Err(ChatError::BadRequest(
            "required tool grammar has no eligible function".into(),
        ));
    }
    let recipients = tools
        .iter()
        .map(|tool| quoted_literal(&tool.function.name))
        .chain(std::iter::once(quoted_literal("tool")))
        .collect::<Vec<_>>()
        .join(" | ");
    let mut invokes = Vec::new();
    for (index, tool) in tools.iter().enumerate() {
        let properties = tool
            .function
            .parameters
            .get("properties")
            .and_then(serde_json::Value::as_object);
        let parameter_names = properties
            .into_iter()
            .flat_map(|properties| properties.keys())
            .map(|name| quoted_literal(name))
            .collect::<Vec<_>>();
        let additional = !matches!(
            tool.function.parameters.get("additionalProperties"),
            Some(serde_json::Value::Bool(false))
        );
        let parameter_name = if additional {
            "[A-Za-z_] [A-Za-z0-9_.-]*".into()
        } else if parameter_names.is_empty() {
            String::new()
        } else {
            format!("({})", parameter_names.join(" | "))
        };
        let parameter_rule = if parameter_name.is_empty() {
            String::new()
        } else {
            let rule_name = format!("tool-parameter-{index}");
            invokes.push((
                rule_name.clone(),
                format!(
                    "\"<atem:parameter name=\\\"\" {parameter_name} \"\\\">\" parameter-char* \"</atem:parameter>\" layout"
                ),
            ));
            format!("{rule_name}*")
        };
        let rule_name = format!("tool-invoke-{index}");
        invokes.push((
            rule_name.clone(),
            format!(
                "\"<atem:invoke name=\\\"\" {} \"\\\">\" layout {parameter_rule} \"</atem:invoke>\" layout",
                quoted_literal(&tool.function.name)
            ),
        ));
    }
    let invoke_names = invokes
        .iter()
        .filter(|(name, _)| name.starts_with("tool-invoke-"))
        .map(|(name, _)| name.as_str())
        .collect::<Vec<_>>()
        .join(" | ");
    let invoke_expression = if request.parallel_tool_calls {
        format!("({invoke_names})+")
    } else {
        format!("({invoke_names})")
    };
    let mut grammar = format!(
        "root ::= reasoning-phase? assistant-prefix \" to=\" ({recipients}) \"<|message|><atem:function_calls>\" layout {invoke_expression} \"</atem:function_calls><|eom|>\"\n\
         reasoning-phase ::= assistant-prefix \" to=self<|message|>\" any* \"<|eom|>\"\n\
         assistant-prefix ::= \"<|start|>assistant\"?\n\
         any ::= [\\U00000000-\\uD7FF\\uE000-\\U0010FFFF]\n\
         parameter-char ::= [^<]\n\
         layout ::= [ \\t\\n\\r]*\n"
    );
    for (name, expression) in invokes {
        grammar.push_str(&format!("{name} ::= {expression}\n"));
    }
    Ok(grammar)
}

fn reject_unknown_keys(
    object: &serde_json::Map<String, serde_json::Value>,
    allowed: &[&str],
    field: &str,
) -> Result<(), ChatError> {
    if let Some(key) = object.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(ChatError::BadRequest(format!(
            "unknown field {field}.{key}"
        )));
    }
    Ok(())
}

fn validate_tools(request: &ChatRequest) -> Result<(), ChatError> {
    let mut names = Vec::new();
    if let Some(tools) = &request.tools {
        for (index, tool) in tools.iter().enumerate() {
            if tool.kind != "function"
                || tool.function.name.is_empty()
                || !tool.function.parameters.is_object()
            {
                return Err(ChatError::BadRequest(format!(
                    "tools[{index}] must be a named function with object parameters"
                )));
            }
            let grammar = json_schema_to_gbnf(&tool.function.parameters).map_err(|error| {
                ChatError::BadRequest(format!(
                    "tools[{index}].function.parameters cannot be constrained: {error}"
                ))
            })?;
            GrammarMatcher::parse(&grammar, "root").map_err(|error| {
                ChatError::BadRequest(format!(
                    "tools[{index}].function.parameters grammar is invalid: {error}"
                ))
            })?;
            names.push(tool.function.name.as_str());
        }
    }
    if let Some(choice) = &request.tool_choice {
        if let ToolChoice::Mode(mode) = choice {
            if !matches!(mode.as_str(), "none" | "auto" | "required") {
                return Err(ChatError::BadRequest(format!(
                    "tool_choice mode {mode:?} is unsupported"
                )));
            }
            if mode == "required" && names.is_empty() {
                return Err(ChatError::BadRequest(
                    "tool_choice='required' needs at least one tool".into(),
                ));
            }
        } else if let ToolChoice::Named(selected) = choice {
            if selected.kind != "function" || !names.contains(&selected.function.name.as_str()) {
                return Err(ChatError::BadRequest(format!(
                    "tool_choice names unknown function {:?}",
                    selected.function.name
                )));
            }
        }
    }
    Ok(())
}

fn validate_generated_atem(
    request: &ChatRequest,
    text: &str,
    allow_incomplete: bool,
) -> Result<(), ChatError> {
    let required = request.tool_choice.as_ref().is_some_and(|choice| {
        matches!(choice, ToolChoice::Mode(mode) if mode == "required")
            || matches!(choice, ToolChoice::Named(_))
    });
    let parsed = match parse_atem_output(text) {
        Ok(parsed) => parsed,
        // A length stop can cut between any two ATEM header tokens. The
        // pinned route returns that partial assistant content; treating it as
        // a poisoned engine makes low max_tokens requests spuriously fail.
        // Required tool calls remain fail-closed because no valid invocation
        // was completed.
        Err(_) if allow_incomplete && !required => return Ok(()),
        Err(error) => {
            return Err(ChatError::Engine(format!(
                "malformed Muse ATEM output: {error}"
            )))
        }
    };
    if required && parsed.tool_calls.is_empty() {
        return Err(ChatError::Engine(
            "model did not emit the required tool call".into(),
        ));
    }
    if !request.parallel_tool_calls && parsed.tool_calls.len() > 1 {
        return Err(ChatError::Engine(
            "model emitted multiple tool calls with parallel_tool_calls=false".into(),
        ));
    }
    for (index, call) in parsed.tool_calls.iter().enumerate() {
        validate_streamed_atem_call_indexed(request, index, call)?;
    }
    Ok(())
}

/// Validate a complete streamed invocation before it crosses the socket
/// boundary. Full-output validation still runs at generation completion, but
/// schema-invalid arguments or a disallowed function must never be emitted as
/// an apparently valid `delta.tool_calls` frame first.
pub fn validate_streamed_atem_call(
    request: &ChatRequest,
    call: &ParsedToolCall,
) -> Result<(), ChatError> {
    validate_streamed_atem_call_indexed(request, 0, call)
}

pub fn validate_streamed_atem_call_indexed(
    request: &ChatRequest,
    index: usize,
    call: &ParsedToolCall,
) -> Result<(), ChatError> {
    if !request.parallel_tool_calls && index > 0 {
        return Err(ChatError::Engine(
            "model emitted multiple tool calls with parallel_tool_calls=false".into(),
        ));
    }
    if matches!(
        request.tool_choice.as_ref(),
        Some(ToolChoice::Mode(mode)) if mode == "none"
    ) {
        return Err(ChatError::Engine(
            "model emitted a tool call with tool_choice='none'".into(),
        ));
    }
    if let Some(selected) = request
        .tool_choice
        .as_ref()
        .and_then(|choice| match choice {
            ToolChoice::Named(choice) => Some(choice.function.name.as_str()),
            ToolChoice::Mode(_) => None,
        })
    {
        if call.function.name != selected {
            return Err(ChatError::Engine(format!(
                "model emitted a tool other than required function {selected:?}"
            )));
        }
    }
    let tool = request
        .tools
        .as_ref()
        .into_iter()
        .flatten()
        .find(|tool| tool.function.name == call.function.name)
        .ok_or_else(|| {
            ChatError::Engine("model emitted a function that was not supplied in tools".into())
        })?;
    let grammar = json_schema_to_gbnf(&tool.function.parameters)
        .map_err(|error| ChatError::Engine(format!("tool schema: {error}")))?;
    let mut matcher = GrammarMatcher::parse(&grammar, "root")
        .map_err(|error| ChatError::Engine(format!("tool schema grammar: {error}")))?;
    if matcher
        .accept_bytes(call.function.arguments.as_bytes())
        .is_err()
        || !matcher.is_accepting()
    {
        return Err(ChatError::Engine(format!(
            "model arguments for function {:?} violate its JSON Schema",
            call.function.name
        )));
    }
    Ok(())
}

fn sampling_params(request: &ChatRequest) -> Result<Option<SamplingParams>, ChatError> {
    // Frozen source defaults from common_params_sampling at llama.cpp
    // 89e0aa6fd362. An explicit zero remains greedy; omission is sampled.
    let temperature = request.temperature.unwrap_or(0.8);
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(ChatError::BadRequest(
            "temperature must be finite and non-negative".into(),
        ));
    }
    let top_p = request.top_p.unwrap_or(0.95);
    if !(top_p.is_finite() && (0.0..=1.0).contains(&top_p)) {
        return Err(ChatError::BadRequest(
            "top_p must be finite and in [0, 1]".into(),
        ));
    }
    let top_k = request.top_k.unwrap_or(40);
    if top_k > i32::MAX as usize {
        return Err(ChatError::BadRequest(
            "top_k must fit the pinned llama.cpp int32 range".into(),
        ));
    }
    if request
        .min_keep
        .is_some_and(|value| value > i32::MAX as usize)
    {
        return Err(ChatError::BadRequest(
            "min_keep must fit the pinned llama.cpp int32 range".into(),
        ));
    }
    if request
        .dry_allowed_length
        .is_some_and(|value| value > i32::MAX as usize)
    {
        return Err(ChatError::BadRequest(
            "dry_allowed_length must fit the pinned llama.cpp int32 range".into(),
        ));
    }
    if request.repeat_last_n.is_some_and(|value| value < 0) {
        return Err(ChatError::BadRequest(
            "repeat_last_n must be non-negative".into(),
        ));
    }
    if request.dry_penalty_last_n.is_some_and(|value| value < 0) {
        return Err(ChatError::BadRequest(
            "dry_penalty_last_n must be non-negative".into(),
        ));
    }
    let chain_enabled = request.mirostat.unwrap_or(0) == 0;
    let params = SamplingParams {
        temperature: if request.mirostat.unwrap_or(0) != 0
            || sampler_enabled(request, "temperature")
        {
            temperature
        } else {
            1.0
        },
        top_p: if chain_enabled { top_p } else { 1.0 },
        top_k: if chain_enabled { top_k } else { 0 },
        typical_p: if chain_enabled {
            request.typical_p.unwrap_or(1.0)
        } else {
            1.0
        },
        min_p: if chain_enabled {
            request.min_p.unwrap_or(0.05)
        } else {
            0.0
        },
        top_n_sigma: if chain_enabled {
            request.top_n_sigma.unwrap_or(-1.0).max(0.0)
        } else {
            0.0
        },
        min_keep: request.min_keep.unwrap_or(0),
    };
    params
        .validate()
        .map_err(|error| ChatError::BadRequest(error.to_string()))?;
    for (name, value, minimum, maximum) in [
        (
            "repeat_penalty",
            request.repeat_penalty.unwrap_or(1.0),
            f32::EPSILON,
            f32::MAX,
        ),
        (
            "presence_penalty",
            request.presence_penalty.unwrap_or(0.0),
            -2.0,
            2.0,
        ),
        (
            "frequency_penalty",
            request.frequency_penalty.unwrap_or(0.0),
            -2.0,
            2.0,
        ),
        (
            "dry_multiplier",
            request.dry_multiplier.unwrap_or(0.0),
            0.0,
            f32::MAX,
        ),
        (
            "dry_base",
            request.dry_base.unwrap_or(1.75),
            -f32::MAX,
            f32::MAX,
        ),
        (
            "dynatemp_range",
            request.dynatemp_range.unwrap_or(0.0),
            0.0,
            f32::MAX,
        ),
        (
            "dynatemp_exponent",
            request.dynatemp_exponent.unwrap_or(1.0),
            f32::EPSILON,
            f32::MAX,
        ),
        (
            "xtc_probability",
            request.xtc_probability.unwrap_or(0.0),
            0.0,
            1.0,
        ),
        (
            "xtc_threshold",
            request.xtc_threshold.unwrap_or(0.1),
            0.0,
            1.0,
        ),
        (
            "mirostat_tau",
            request.mirostat_tau.unwrap_or(5.0),
            f32::EPSILON,
            f32::MAX,
        ),
        (
            "mirostat_eta",
            request.mirostat_eta.unwrap_or(0.1),
            f32::EPSILON,
            f32::MAX,
        ),
        (
            "adaptive_target",
            request.adaptive_target.unwrap_or(-1.0),
            -1.0,
            1.0,
        ),
        (
            "adaptive_decay",
            request.adaptive_decay.unwrap_or(0.9),
            0.0,
            0.99,
        ),
    ] {
        if !value.is_finite() || value < minimum || value > maximum {
            return Err(ChatError::BadRequest(format!(
                "{name} must be finite and in [{minimum}, {maximum}]"
            )));
        }
    }
    if request.mirostat.unwrap_or(0) > 2 {
        return Err(ChatError::BadRequest("mirostat must be 0, 1, or 2".into()));
    }
    if request.logit_bias.as_ref().is_some_and(|biases| {
        biases
            .values()
            .any(|value| value.is_nan() || *value == f32::INFINITY)
    }) {
        return Err(ChatError::BadRequest(
            "logit_bias values must be finite or false".into(),
        ));
    }
    if params.temperature == 0.0
        && request.dynatemp_range.unwrap_or(0.0) == 0.0
        && request.mirostat.unwrap_or(0) == 0
        && request.adaptive_target.unwrap_or(-1.0) < 0.0
    {
        Ok(None)
    } else {
        Ok(Some(params))
    }
}

fn sampler_config_sha256(request: &ChatRequest) -> [u8; 32] {
    let logit_bias = request.logit_bias.as_ref().map(|biases| {
        biases
            .iter()
            .map(|(token, bias)| (token.clone(), *bias))
            .collect::<BTreeMap<_, _>>()
    });
    let canonical = serde_json::json!({
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "typical_p": request.typical_p,
        "min_p": request.min_p,
        "top_n_sigma": request.top_n_sigma,
        "min_keep": request.min_keep,
        "logit_bias": logit_bias,
        "repeat_penalty": request.repeat_penalty,
        "repeat_last_n": request.repeat_last_n,
        "presence_penalty": request.presence_penalty,
        "frequency_penalty": request.frequency_penalty,
        "dry_multiplier": request.dry_multiplier,
        "dry_base": request.dry_base,
        "dry_allowed_length": request.dry_allowed_length,
        "dry_penalty_last_n": request.dry_penalty_last_n,
        "dry_sequence_breakers": request.dry_sequence_breakers,
        "mirostat": request.mirostat,
        "mirostat_tau": request.mirostat_tau,
        "mirostat_eta": request.mirostat_eta,
        "adaptive_target": request.adaptive_target,
        "adaptive_decay": request.adaptive_decay,
        "dynatemp_range": request.dynatemp_range,
        "dynatemp_exponent": request.dynatemp_exponent,
        "xtc_probability": request.xtc_probability,
        "xtc_threshold": request.xtc_threshold,
        "samplers": request.samplers,
        "ignore_eos": request.ignore_eos,
    });
    Sha256::digest(serde_json::to_vec(&canonical).expect("sampler config serializes")).into()
}

fn dflash_sampling_compatible(request: &ChatRequest) -> bool {
    // Arbitrary text stops can span token boundaries. Until the grammar/stop
    // matcher participates in the verifier's pre-commit decision, keep those
    // requests on target-only so no invisible verified suffix can advance KV.
    !request.ignore_eos
        && request.stop.as_ref().is_none_or(StopField::is_empty)
        && request.grammar.is_none()
        && request.json_schema.is_none()
        && request.response_format.is_none()
        && request.samplers.is_none()
        && request.logprobs != Some(true)
        && request.logit_bias.is_none()
        && request.t_max_predict_ms.is_none_or(|value| value <= 0)
        && request.repeat_penalty.unwrap_or(1.0) == 1.0
        && request.presence_penalty.unwrap_or(0.0) == 0.0
        && request.frequency_penalty.unwrap_or(0.0) == 0.0
        && request.dry_multiplier.unwrap_or(0.0) == 0.0
        && request.mirostat.unwrap_or(0) == 0
        && request.dynatemp_range.unwrap_or(0.0) == 0.0
        && request.xtc_probability.unwrap_or(0.0) == 0.0
}

struct AdaptiveSamplerState {
    target: f32,
    decay: f32,
    weighted_sum: f32,
    total_weight: f32,
    rng: Mt19937,
    pending: Option<(u32, f32)>,
}

impl AdaptiveSamplerState {
    fn new(request: &ChatRequest, seed: u32) -> Self {
        let target = request.adaptive_target.unwrap_or(-1.0);
        let decay = request.adaptive_decay.unwrap_or(0.9).clamp(0.0, 0.99);
        Self {
            target,
            decay,
            weighted_sum: target / (1.0 - decay),
            total_weight: 1.0 / (1.0 - decay),
            rng: Mt19937::new(seed),
            pending: None,
        }
    }

    fn sample(&mut self, original: &[f32], order: &[u32]) -> Result<u32, ChatError> {
        let target = self.target.clamp(0.0, 1.0);
        let adapted = (2.0 * target - self.weighted_sum / self.total_weight).clamp(0.0, 1.0);
        let mut transformed = vec![0.0; original.len()];
        let mut maximum = f32::NEG_INFINITY;
        for (index, probability) in original.iter().copied().enumerate() {
            if probability <= 0.0 {
                continue;
            }
            let distance = ((probability - adapted) / 0.3).abs();
            let logit = 5.0 - 10.0 * distance * distance / (1.0 + distance);
            transformed[index] = logit;
            maximum = maximum.max(logit);
        }
        if !maximum.is_finite() {
            return Err(ChatError::Engine(
                "adaptive_p received an empty distribution".into(),
            ));
        }
        for (probability, &source) in transformed.iter_mut().zip(original) {
            *probability = if source > 0.0 {
                (*probability - maximum).exp()
            } else {
                0.0
            };
        }
        let selected = sample_discrete_distribution_mt_ordered(&transformed, order, &mut self.rng)
            .map_err(|error| ChatError::Engine(error.to_string()))?;
        self.pending = Some((selected, original[selected as usize]));
        Ok(selected)
    }

    fn accept(&mut self, token: u32) {
        if let Some((pending, probability)) = self.pending.take() {
            if pending == token {
                self.weighted_sum = probability + self.decay * self.weighted_sum;
                self.total_weight = 1.0 + self.decay * self.total_weight;
            }
        }
    }
}

struct RequestSamplerState {
    distribution_rng: Mt19937,
    xtc_rng: Mt19937,
    mirostat_rng: Mt19937,
    mirostat_mu: f32,
    adaptive: AdaptiveSamplerState,
}

impl RequestSamplerState {
    fn new(request: &ChatRequest, seed: u32) -> Self {
        Self {
            distribution_rng: Mt19937::new(seed),
            xtc_rng: Mt19937::new(seed),
            mirostat_rng: Mt19937::new(seed),
            mirostat_mu: request.mirostat_tau.unwrap_or(5.0) * 2.0,
            adaptive: AdaptiveSamplerState::new(request, seed),
        }
    }

    fn snapshot(&self) -> SamplerStateSnapshot {
        SamplerStateSnapshot {
            distribution_rng: self.distribution_rng.snapshot(),
            xtc_rng: self.xtc_rng.snapshot(),
            mirostat_rng: self.mirostat_rng.snapshot(),
            adaptive_rng: self.adaptive.rng.snapshot(),
            mirostat_mu: self.mirostat_mu,
            adaptive_weighted_sum: self.adaptive.weighted_sum,
            adaptive_total_weight: self.adaptive.total_weight,
        }
    }

    fn restore(&mut self, snapshot: &SamplerStateSnapshot) -> Result<(), ChatError> {
        self.distribution_rng = Mt19937::from_snapshot(&snapshot.distribution_rng)
            .map_err(|error| ChatError::Conflict(error.to_string()))?;
        self.xtc_rng = Mt19937::from_snapshot(&snapshot.xtc_rng)
            .map_err(|error| ChatError::Conflict(error.to_string()))?;
        self.mirostat_rng = Mt19937::from_snapshot(&snapshot.mirostat_rng)
            .map_err(|error| ChatError::Conflict(error.to_string()))?;
        self.adaptive.rng = Mt19937::from_snapshot(&snapshot.adaptive_rng)
            .map_err(|error| ChatError::Conflict(error.to_string()))?;
        if !snapshot.mirostat_mu.is_finite()
            || !snapshot.adaptive_weighted_sum.is_finite()
            || !snapshot.adaptive_total_weight.is_finite()
            || snapshot.adaptive_total_weight <= 0.0
        {
            return Err(ChatError::Conflict(
                "stored sampler scalar state is invalid".into(),
            ));
        }
        self.mirostat_mu = snapshot.mirostat_mu;
        self.adaptive.weighted_sum = snapshot.adaptive_weighted_sum;
        self.adaptive.total_weight = snapshot.adaptive_total_weight;
        self.adaptive.pending = None;
        Ok(())
    }
}

fn take_forced_reasoning_token(
    request: &ChatRequest,
    marker: &VecDeque<u32>,
    pending: &mut VecDeque<u32>,
    closed: bool,
) -> Option<u32> {
    if closed {
        pending.clear();
        return None;
    }
    if pending.is_empty()
        && request
            .reasoning_end_signal
            .as_ref()
            .is_some_and(|signal| signal.swap(false, Ordering::AcqRel))
    {
        pending.extend(marker.iter().copied());
    }
    pending.pop_front()
}

// Sampling intentionally receives the live request, RNG, grammar, EOS set,
// and tokenizer identity together so no caller can apply those stateful
// constraints in a different order.
#[allow(clippy::too_many_arguments)]
fn sample_or_argmax(
    logits: &[f32],
    params: Option<SamplingParams>,
    request: &ChatRequest,
    history: &[u32],
    sampler: &mut RequestSamplerState,
    grammar: Option<&GrammarMatcher>,
    eos: &[u32],
    model: &Model,
    state: &ServerState,
) -> Result<u32, ChatError> {
    let mut adjusted = logits.to_vec();
    if request.ignore_eos {
        for &token in eos {
            if let Some(logit) = adjusted.get_mut(token as usize) {
                *logit = f32::NEG_INFINITY;
            }
        }
    }
    if let Some(biases) = &request.logit_bias {
        for (target, bias) in biases {
            let tokens = match target.parse::<usize>() {
                Ok(token) => vec![token],
                Err(_) => model
                    .encode_with_options(target, true)
                    .into_iter()
                    .map(|token| token as usize)
                    .collect(),
            };
            if tokens.is_empty() {
                return Err(ChatError::BadRequest(format!(
                    "logit_bias string target '{target}' produced no tokens"
                )));
            }
            for token in tokens {
                let value = adjusted.get_mut(token).ok_or_else(|| {
                    ChatError::BadRequest(format!("logit_bias token {token} is out of vocabulary"))
                })?;
                *value += *bias;
            }
        }
    }
    let repeat_last_n = request.repeat_last_n.unwrap_or(64);
    let history = if repeat_last_n == 0 {
        &[][..]
    } else if repeat_last_n < 0 {
        history
    } else {
        &history[history.len().saturating_sub(repeat_last_n as usize)..]
    };
    let mut counts = std::collections::HashMap::<u32, usize>::new();
    for token in history {
        *counts.entry(*token).or_default() += 1;
    }
    match params {
        Some(params) => {
            // Pinned llama-server uses grammar rejection sampling: run the
            // ordinary chain first, accept an eligible result immediately,
            // and only rerun with a grammar mask after a rejected result. The
            // rejected draw deliberately advances every stochastic sampler.
            let mut sample = |grammar_first| {
                if request.mirostat.unwrap_or(0) != 0 {
                    sample_mirostat(
                        &adjusted,
                        request,
                        sampler,
                        grammar_first,
                        eos,
                        model,
                        state,
                    )
                } else {
                    sample_standard_chain(
                        &adjusted,
                        params,
                        request,
                        history,
                        &counts,
                        sampler,
                        grammar_first,
                        eos,
                        model,
                        state,
                    )
                }
            };
            let first = sample(None)?;
            let selected = if grammar.is_none_or(|grammar| {
                let grammar_started = Instant::now();
                let allowed = grammar_allows(grammar, model, first, eos);
                state.record_phase(
                    "grammar",
                    grammar_started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                );
                allowed
            }) {
                first
            } else {
                sample(grammar)?
            };
            if sampler_enabled(request, "adaptive_p") && sampler.adaptive.target >= 0.0 {
                sampler.adaptive.accept(selected);
            }
            Ok(selected)
        }
        None => {
            if let Some(grammar) = grammar {
                let grammar_started = Instant::now();
                let mut ranked = (0..adjusted.len()).collect::<Vec<_>>();
                ranked.sort_unstable_by(|left, right| {
                    adjusted[*right]
                        .total_cmp(&adjusted[*left])
                        .then_with(|| left.cmp(right))
                });
                let selected = ranked
                    .into_iter()
                    .find(|token| grammar_allows(grammar, model, *token as u32, eos))
                    .map(|token| token as u32)
                    .ok_or_else(|| {
                        ChatError::Engine("constrained grammar rejected every token".into())
                    });
                state.record_phase(
                    "grammar",
                    grammar_started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                );
                selected
            } else {
                Ok(argmax(&adjusted) as u32)
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn sample_standard_chain(
    logits: &[f32],
    params: SamplingParams,
    request: &ChatRequest,
    history: &[u32],
    counts: &std::collections::HashMap<u32, usize>,
    sampler: &mut RequestSamplerState,
    grammar_first: Option<&GrammarMatcher>,
    eos: &[u32],
    model: &Model,
    state: &ServerState,
) -> Result<u32, ChatError> {
    let mut logits = logits.to_vec();
    if let Some(grammar) = grammar_first {
        let started = Instant::now();
        for (token, logit) in logits.iter_mut().enumerate() {
            if logit.is_finite() && !grammar_allows(grammar, model, token as u32, eos) {
                *logit = f32::NEG_INFINITY;
            }
        }
        state.record_phase(
            "grammar",
            started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        );
        if logits.iter().all(|logit| *logit == f32::NEG_INFINITY) {
            return Err(ChatError::Engine(
                "constrained grammar rejected every sampled token".into(),
            ));
        }
    }

    let chain = request.samplers.as_ref().map_or_else(
        || PINNED_SAMPLER_ORDER.to_vec(),
        |stages| stages.iter().map(String::as_str).collect::<Vec<_>>(),
    );
    let mut candidates = logits
        .iter()
        .copied()
        .enumerate()
        .map(|(token, logit)| (token as u32, logit))
        .collect::<Vec<_>>();
    let mut sorted = false;
    let mut adaptive = false;
    for stage in chain {
        match stage {
            "penalties" => {
                let repeat = request.repeat_penalty.unwrap_or(1.0);
                let presence = request.presence_penalty.unwrap_or(0.0);
                let frequency = request.frequency_penalty.unwrap_or(0.0);
                for (token, logit) in &mut candidates {
                    let Some(count) = counts.get(token) else {
                        continue;
                    };
                    if repeat != 1.0 {
                        *logit = if *logit <= 0.0 {
                            *logit * repeat
                        } else {
                            *logit / repeat
                        };
                    }
                    *logit -= presence + frequency * *count as f32;
                }
                sorted = false;
            }
            "dry" => {
                let mut penalty = vec![0.0; logits.len()];
                apply_dry_penalty(&mut penalty, history, request, model);
                for (token, logit) in &mut candidates {
                    *logit += penalty[*token as usize];
                }
                sorted = false;
            }
            "top_n_sigma" if params.top_n_sigma > 0.0 && candidates.len() > 1 => {
                let finite = candidates
                    .iter()
                    .map(|entry| entry.1)
                    .filter(|value| *value != f32::NEG_INFINITY)
                    .collect::<Vec<_>>();
                let maximum = finite.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mean = finite.iter().copied().sum::<f32>() / finite.len() as f32;
                let variance = finite
                    .iter()
                    .map(|value| (*value - mean).powi(2))
                    .sum::<f32>()
                    / finite.len() as f32;
                let threshold = maximum - params.top_n_sigma * variance.sqrt();
                for (_, logit) in &mut candidates {
                    if *logit < threshold {
                        *logit = f32::NEG_INFINITY;
                    }
                }
            }
            "top_k" if params.top_k > 0 => {
                sort_sampler_candidates(&mut candidates);
                candidates.truncate(params.top_k.min(candidates.len()));
                sorted = true;
            }
            "typ_p" if params.typical_p < 1.0 => {
                if !sorted {
                    sort_sampler_candidates(&mut candidates);
                }
                let probabilities = candidate_softmax(&candidates)?;
                let entropy = probabilities
                    .iter()
                    .map(|probability| -probability * probability.ln())
                    .sum::<f32>();
                let mut typical = candidates
                    .into_iter()
                    .zip(probabilities)
                    .collect::<Vec<_>>();
                typical.sort_unstable_by(|left, right| {
                    (-left.1.ln() - entropy)
                        .abs()
                        .total_cmp(&(-right.1.ln() - entropy).abs())
                        .then_with(|| left.0 .0.cmp(&right.0 .0))
                });
                let mut cumulative = 0.0;
                let mut keep = typical.len();
                for (index, (_, probability)) in typical.iter().enumerate() {
                    cumulative += probability;
                    if cumulative > params.typical_p && index + 1 >= params.min_keep {
                        keep = index + 1;
                        break;
                    }
                }
                typical.truncate(keep.max(1));
                candidates = typical.into_iter().map(|entry| entry.0).collect();
                sorted = false;
            }
            "top_p" if params.top_p < 1.0 => {
                if !sorted {
                    sort_sampler_candidates(&mut candidates);
                }
                let probabilities = candidate_softmax(&candidates)?;
                let mut cumulative = 0.0;
                let mut keep = candidates.len();
                for (index, probability) in probabilities.into_iter().enumerate() {
                    cumulative += probability;
                    if cumulative >= params.top_p && index + 1 >= params.min_keep {
                        keep = index + 1;
                        break;
                    }
                }
                candidates.truncate(keep.max(1));
                sorted = true;
            }
            "min_p" if params.min_p > 0.0 => {
                let maximum = candidates
                    .iter()
                    .map(|entry| entry.1)
                    .fold(f32::NEG_INFINITY, f32::max);
                let threshold = maximum + params.min_p.ln();
                if !sorted {
                    let filtered = candidates
                        .iter()
                        .copied()
                        .filter(|entry| entry.1 >= threshold)
                        .collect::<Vec<_>>();
                    if !filtered.is_empty() && filtered.len() >= params.min_keep {
                        candidates = filtered;
                    } else {
                        sort_sampler_candidates(&mut candidates);
                        sorted = true;
                    }
                }
                if sorted {
                    let matching = candidates
                        .iter()
                        .take_while(|entry| entry.1 >= threshold)
                        .count();
                    candidates.truncate(matching.max(params.min_keep).max(1).min(candidates.len()));
                }
            }
            "xtc"
                if request.xtc_probability.unwrap_or(0.0) > 0.0
                    && request.xtc_threshold.unwrap_or(0.1) <= 0.5
                    && candidates.len() >= 2
                    && sampler.xtc_rng.uniform_f32() <= request.xtc_probability.unwrap_or(0.0) =>
            {
                sort_sampler_candidates(&mut candidates);
                let probabilities = candidate_softmax(&candidates)?;
                let above = probabilities
                    .iter()
                    .take_while(|probability| **probability >= request.xtc_threshold.unwrap_or(0.1))
                    .count();
                if above > 1 && candidates.len().saturating_sub(above - 1) >= params.min_keep {
                    candidates.drain(..above - 1);
                }
                sorted = true;
            }
            "temperature" => {
                let mut temperature = params.temperature;
                let range = request.dynatemp_range.unwrap_or(0.0);
                if range > 0.0 && candidates.len() > 1 {
                    // llama_sampler_temp_ext calls softmax with sorting
                    // enabled before accumulating entropy. Besides defining
                    // the f32 accumulation order, this prevents the first
                    // token-ID candidate from being mistaken for the maximum.
                    sort_sampler_candidates(&mut candidates);
                    sorted = true;
                    let probabilities = candidate_softmax(&candidates)?;
                    let entropy = probabilities
                        .iter()
                        .filter(|probability| **probability > 0.0)
                        .map(|probability| -probability * probability.ln())
                        .sum::<f32>()
                        / (-(1.0f32 / candidates.len() as f32).ln());
                    let minimum = (temperature - range).max(0.0);
                    let maximum = temperature + range;
                    temperature = minimum
                        + (maximum - minimum)
                            * entropy.powf(request.dynatemp_exponent.unwrap_or(1.0));
                }
                if temperature <= 0.0 {
                    let mut maximum = 0;
                    for index in 1..candidates.len() {
                        if candidates[index].1 > candidates[maximum].1 {
                            maximum = index;
                        }
                    }
                    for (index, (_, logit)) in candidates.iter_mut().enumerate() {
                        *logit = if index == maximum {
                            0.0
                        } else {
                            f32::NEG_INFINITY
                        };
                    }
                } else if temperature != 1.0 {
                    for (_, logit) in &mut candidates {
                        *logit /= temperature;
                    }
                }
            }
            "adaptive_p" => adaptive = true,
            _ => {}
        }
    }
    if candidates.is_empty() {
        return Err(ChatError::Engine(
            "sampler chain removed every candidate".into(),
        ));
    }
    let weights_by_candidate = sampler_weights(&candidates)?;
    let mut probabilities = vec![0.0; logits.len()];
    let mut sampling_weights = vec![0.0; logits.len()];
    let candidate_order = candidates.iter().map(|entry| entry.0).collect::<Vec<_>>();
    for ((token, _), weight) in candidates.iter().zip(weights_by_candidate) {
        probabilities[*token as usize] = weight;
        sampling_weights[*token as usize] = weight;
    }
    normalize_probabilities_ordered(&mut probabilities, &candidate_order)?;
    if adaptive && sampler.adaptive.target >= 0.0 {
        sampler.adaptive.sample(&probabilities, &candidate_order)
    } else {
        sample_distribution_mt_ordered(
            &sampling_weights,
            &candidate_order,
            &mut sampler.distribution_rng,
        )
        .map_err(|error| ChatError::Engine(error.to_string()))
    }
}

fn normalize_probabilities(probabilities: &mut [f32]) -> Result<(), ChatError> {
    let total = probabilities.iter().copied().sum::<f32>();
    if !total.is_finite() || total <= 0.0 {
        return Err(ChatError::Engine(
            "sampler removed every candidate probability".into(),
        ));
    }
    for probability in probabilities {
        *probability /= total;
    }
    Ok(())
}

fn sort_sampler_candidates(candidates: &mut [(u32, f32)]) {
    candidates.sort_unstable_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
}

fn sampler_weights(candidates: &[(u32, f32)]) -> Result<Vec<f32>, ChatError> {
    let maximum = candidates
        .iter()
        .map(|entry| entry.1)
        .fold(f32::NEG_INFINITY, f32::max);
    if !maximum.is_finite() {
        return Err(ChatError::Engine(
            "sampler chain retained no finite candidate".into(),
        ));
    }
    let weights = candidates
        .iter()
        .map(|entry| (entry.1 - maximum).exp())
        .collect::<Vec<_>>();
    if weights.iter().copied().sum::<f32>() <= 0.0 {
        return Err(ChatError::Engine(
            "sampler chain produced an empty distribution".into(),
        ));
    }
    Ok(weights)
}

fn normalize_probabilities_ordered(
    probabilities: &mut [f32],
    order: &[u32],
) -> Result<(), ChatError> {
    let total = order
        .iter()
        .map(|token| probabilities[*token as usize])
        .sum::<f32>();
    if !total.is_finite() || total <= 0.0 {
        return Err(ChatError::Engine(
            "sampler removed every candidate probability".into(),
        ));
    }
    for &token in order {
        probabilities[token as usize] /= total;
    }
    Ok(())
}

fn candidate_softmax(candidates: &[(u32, f32)]) -> Result<Vec<f32>, ChatError> {
    let maximum = candidates
        .iter()
        .map(|candidate| candidate.1)
        .fold(f32::NEG_INFINITY, f32::max);
    if !maximum.is_finite() {
        return Err(ChatError::Engine("sampler has no finite candidates".into()));
    }
    let mut probabilities = candidates
        .iter()
        .map(|candidate| (candidate.1 - maximum).exp())
        .collect::<Vec<_>>();
    normalize_probabilities(&mut probabilities)?;
    Ok(probabilities)
}

#[allow(clippy::too_many_arguments)]
fn sample_mirostat(
    logits: &[f32],
    request: &ChatRequest,
    sampler: &mut RequestSamplerState,
    grammar: Option<&GrammarMatcher>,
    eos: &[u32],
    model: &Model,
    state: &ServerState,
) -> Result<u32, ChatError> {
    let temperature = request.temperature.unwrap_or(0.8);
    let grammar_started = Instant::now();
    let eligible = logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(token, logit)| {
            logit.is_finite()
                && grammar.is_none_or(|grammar| grammar_allows(grammar, model, *token as u32, eos))
        })
        .map(|(token, logit)| (token as u32, logit))
        .collect::<Vec<_>>();
    if grammar.is_some() {
        state.record_phase(
            "grammar",
            grammar_started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
        );
    }
    if eligible.is_empty() {
        return Err(ChatError::Engine(
            "constrained grammar rejected every sampled token".into(),
        ));
    }
    if temperature <= 0.0 {
        let mut selected = eligible[0];
        for &candidate in &eligible[1..] {
            if candidate.1 > selected.1 {
                selected = candidate;
            }
        }
        let _ = sample_discrete_distribution_mt(&[1.0], &mut sampler.mirostat_rng)
            .map_err(|error| ChatError::Engine(error.to_string()))?;
        sampler.mirostat_mu +=
            request.mirostat_eta.unwrap_or(0.1) * request.mirostat_tau.unwrap_or(5.0);
        return Ok(selected.0);
    }
    let mut candidates = eligible
        .into_iter()
        .map(|(token, logit)| (token, logit / temperature))
        .collect::<Vec<_>>();
    candidates.sort_unstable_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    let mode = request.mirostat.unwrap_or(0);
    let tau = request.mirostat_tau.unwrap_or(5.0);
    let eta = request.mirostat_eta.unwrap_or(0.1);
    let mut probabilities = candidate_softmax(&candidates)?;
    if mode == 1 {
        let mut sum_ti_bi = 0.0f32;
        let mut sum_ti_sq = 0.0f32;
        for index in 0..99.min(candidates.len().saturating_sub(1)) {
            let t_i = (((index + 2) as f32) / ((index + 1) as f32)).ln();
            let b_i = (probabilities[index] / probabilities[index + 1]).ln();
            sum_ti_bi += t_i * b_i;
            sum_ti_sq += t_i * t_i;
        }
        let s_hat = sum_ti_bi / sum_ti_sq;
        let epsilon_hat = s_hat - 1.0;
        let k = ((epsilon_hat * 2.0f32.powf(sampler.mirostat_mu))
            / (1.0 - (logits.len() as f32).powf(-epsilon_hat)))
        .powf(s_hat.recip());
        let keep = (k as usize).max(1).min(candidates.len());
        candidates.truncate(keep);
        probabilities = candidate_softmax(&candidates)?;
    } else {
        let keep = probabilities
            .iter()
            .position(|probability| -probability.log2() > sampler.mirostat_mu)
            .unwrap_or(probabilities.len())
            .max(1);
        candidates.truncate(keep);
        probabilities = candidate_softmax(&candidates)?;
    }
    let selected = sample_discrete_distribution_mt(&probabilities, &mut sampler.mirostat_rng)
        .map_err(|error| ChatError::Engine(error.to_string()))? as usize;
    let surprise = -probabilities[selected].log2();
    sampler.mirostat_mu -= eta * (surprise - tau);
    Ok(candidates[selected].0)
}

fn grammar_allows(grammar: &GrammarMatcher, model: &Model, token: u32, eos: &[u32]) -> bool {
    if eos.contains(&token) {
        return grammar.is_accepting();
    }
    grammar.allows_token(token, model.token_bytes(token))
}

fn build_token_logprob(
    model: &Model,
    logits: &[f32],
    chosen: u32,
    top_n: usize,
    eos: &[u32],
) -> Result<TokenLogprob, ChatError> {
    if chosen as usize >= logits.len()
        || logits
            .iter()
            .any(|value| value.is_nan() || *value == f32::INFINITY)
    {
        return Err(ChatError::Engine(
            "target logprob distribution is invalid".into(),
        ));
    }
    // Pinned llama-server partially sorts the candidate array *before* its
    // scalar f32 softmax. The unselected tail is therefore accumulated in
    // libc++ heap-selection order, not token-ID order. With a 202k vocabulary
    // that rounding difference alone can move every reported logprob by more
    // than the 1e-4 public tolerance even when raw logits are within 5e-6.
    let ranked = source_partial_sort_order(logits, top_n);
    let maximum = ranked.first().filter(|_| top_n > 0).map_or_else(
        || logits.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        |token| logits[*token],
    );
    if !maximum.is_finite() {
        return Err(ChatError::Engine(
            "target logprob distribution has no finite candidate".into(),
        ));
    }
    let mut partition = 0.0f32;
    for &token in &ranked {
        partition += (logits[token] - maximum).exp();
    }
    if !partition.is_finite() || partition <= 0.0 {
        return Err(ChatError::Engine(
            "target logprob distribution cannot be normalized".into(),
        ));
    }
    let logprob = |token: u32| {
        // Keep the source's expf/divide/logf sequence instead of folding it
        // into logit-logsumexp; the intermediate f32 rounding is observable.
        let probability = (logits[token as usize] - maximum).exp() / partition;
        let value = probability.ln();
        if value.is_finite() {
            value as f64
        } else {
            f32::MIN as f64
        }
    };
    let entry = |token: u32| {
        // llama_token_to_piece(..., special = false) exposes every EOG
        // control token as an empty piece. The GGUF vocabulary still stores
        // its printable spelling, which is useful for template encoding but
        // must not leak into completion logprobs or stream bytes.
        let bytes = if eos.contains(&token) {
            Vec::new()
        } else {
            model.token_bytes(token).to_vec()
        };
        TopLogprob {
            id: token,
            token: String::from_utf8_lossy(&bytes).into_owned(),
            logprob: logprob(token),
            bytes,
        }
    };
    let top_logprobs = ranked
        .into_iter()
        .take(top_n)
        .map(|token| entry(token as u32))
        .collect();
    let chosen_entry = entry(chosen);
    Ok(TokenLogprob {
        id: chosen_entry.id,
        token: chosen_entry.token,
        logprob: chosen_entry.logprob,
        bytes: chosen_entry.bytes,
        top_logprobs,
    })
}

/// Reproduce libc++ `partial_sort(first, first + n, last, logit_descending)`
/// through its heap-selection phase. Sorting the selected prefix afterwards
/// cannot affect the already-permuted tail and gives the same strict ordering
/// for finite unequal logits used by the source server.
fn source_partial_sort_order(logits: &[f32], top_n: usize) -> Vec<usize> {
    let mut order = (0..logits.len()).collect::<Vec<_>>();
    let selected = top_n.min(order.len());
    if selected == 0 {
        return order;
    }
    let precedes = |left: usize, right: usize| logits[left] > logits[right];
    if selected > 1 {
        for start in (0..=((selected - 2) / 2)).rev() {
            source_sift_down(&mut order[..selected], start, &precedes);
        }
    }
    for index in selected..order.len() {
        if precedes(order[index], order[0]) {
            order.swap(index, 0);
            source_sift_down(&mut order[..selected], 0, &precedes);
        }
    }
    order[..selected].sort_unstable_by(|left, right| {
        logits[*right]
            .total_cmp(&logits[*left])
            .then_with(|| left.cmp(right))
    });
    order
}

fn source_sift_down(
    heap: &mut [usize],
    mut start: usize,
    precedes: &impl Fn(usize, usize) -> bool,
) {
    let len = heap.len();
    if len < 2 || (len - 2) / 2 < start {
        return;
    }
    let mut child = 2 * start + 1;
    if child + 1 < len && precedes(heap[child], heap[child + 1]) {
        child += 1;
    }
    if precedes(heap[child], heap[start]) {
        return;
    }
    let top = heap[start];
    loop {
        heap[start] = heap[child];
        start = child;
        if (len - 2) / 2 < child {
            break;
        }
        child = 2 * child + 1;
        if child + 1 < len && precedes(heap[child], heap[child + 1]) {
            child += 1;
        }
        if precedes(heap[child], top) {
            break;
        }
    }
    heap[start] = top;
}

fn apply_dry_penalty(logits: &mut [f32], history: &[u32], request: &ChatRequest, model: &Model) {
    let multiplier = request.dry_multiplier.unwrap_or(0.0);
    if multiplier <= 0.0 || history.len() < 2 {
        return;
    }
    let allowed = request.dry_allowed_length.unwrap_or(2);
    // Pinned llama.cpp replaces values below one with the configured
    // default instead of rejecting the request or exponentiating them.
    let base = request
        .dry_base
        .filter(|value| *value >= 1.0)
        .unwrap_or(1.75);
    let last_n = request.dry_penalty_last_n.unwrap_or(64);
    let history = if last_n < 0 {
        history
    } else {
        &history[history.len().saturating_sub(last_n as usize)..]
    };
    let default_breakers = vec!["\n".into(), ":".into(), "\"".into(), "*".into()];
    let breakers = request
        .dry_sequence_breakers
        .as_deref()
        .unwrap_or(default_breakers.as_slice());
    let mut bytes = Vec::new();
    let mut boundaries = Vec::with_capacity(history.len() + 1);
    boundaries.push(0);
    for &token in history {
        bytes.extend_from_slice(model.token_bytes(token));
        boundaries.push(bytes.len());
    }
    let mut repetition_limit = history.len();
    for breaker in breakers {
        if breaker.is_empty() {
            continue;
        }
        let needle = breaker.as_bytes();
        if let Some(start) = bytes
            .windows(needle.len())
            .rposition(|window| window == needle)
        {
            let end = start + needle.len();
            let tokens_after = boundaries
                .iter()
                .position(|boundary| *boundary >= end)
                .map_or(0, |boundary| history.len().saturating_sub(boundary));
            repetition_limit = repetition_limit.min(tokens_after);
        }
    }
    if repetition_limit < allowed {
        return;
    }
    let max_match = history.len().saturating_sub(1).min(repetition_limit);
    let mut longest = std::collections::HashMap::<u32, usize>::new();
    for length in allowed..=max_match {
        let suffix = &history[history.len() - length..];
        for start in 0..history.len().saturating_sub(length) {
            if history[start..start + length] == *suffix {
                if let Some(&next) = history.get(start + length) {
                    longest
                        .entry(next)
                        .and_modify(|current| *current = (*current).max(length))
                        .or_insert(length);
                }
            }
        }
    }
    for (token, length) in longest {
        if breakers.iter().any(|breaker| {
            !breaker.is_empty()
                && model
                    .token_bytes(token)
                    .windows(breaker.len())
                    .any(|window| window == breaker.as_bytes())
        }) {
            continue;
        }
        if let Some(logit) = logits.get_mut(token as usize) {
            let exponent = length.saturating_sub(allowed).min(1024) as i32;
            *logit -= multiplier * base.powi(exponent);
        }
    }
}

enum PreparedSegment {
    Tokens(Vec<u32>),
    Image {
        encoded: Vec<u8>,
        preprocessed: PreprocessedImage,
        projected_tokens: usize,
    },
}

struct PreparedPrefill {
    segments: Vec<PreparedSegment>,
    witnesses: Vec<u32>,
    positions: usize,
    remote_multimodal: Option<(MultimodalIdentityV2, Vec<PrefillControlSegmentV2>)>,
}

struct CanonicalPrefill {
    prefill: PreparedPrefill,
    replay_messages: Vec<Message>,
    shifted: bool,
}

fn prepare_with_context_policy(
    runtime: &InferenceRuntime,
    request: &ChatRequest,
    output_reserve: usize,
) -> Result<CanonicalPrefill, ChatError> {
    let capacity = retained_context_capacity(runtime.max_context, output_reserve)?;
    if let Some(tokens) = request.muser_prompt_token_ids.as_deref() {
        let shifted = tokens.len() > capacity;
        let mut retained = tokens.to_vec();
        if shifted {
            if runtime.context_policy == ContextPolicy::Error {
                return Err(ChatError::BadRequest(format!(
                    "raw prompt plus output reserve exceeds context limit {}",
                    runtime.max_context
                )));
            }
            retained = compact_raw_prompt(tokens, capacity, runtime.raw_retain_prefix)?;
        }
        return Ok(CanonicalPrefill {
            prefill: build_exact_token_prefill(&retained)?,
            replay_messages: Vec::new(),
            shifted,
        });
    }

    let full = build_prefill(
        runtime,
        &request.messages,
        request.tools.as_ref(),
        request.add_generation_prompt,
    )?;
    if full.positions <= capacity {
        return Ok(CanonicalPrefill {
            prefill: full,
            replay_messages: request.messages.clone(),
            shifted: false,
        });
    }
    if runtime.context_policy == ContextPolicy::Error {
        return Err(ChatError::BadRequest(format!(
            "chat prompt plus output reserve exceeds context limit {}",
            runtime.max_context
        )));
    }

    let (systems, mut turns) = shift_chat_units(&request.messages)?;
    if turns.is_empty() {
        return Err(ChatError::BadRequest(
            "system content plus output reserve cannot fit without a complete turn".into(),
        ));
    }
    loop {
        let messages = systems
            .iter()
            .cloned()
            .chain(turns.iter().flatten().cloned())
            .collect::<Vec<_>>();
        let prefill = build_prefill(
            runtime,
            &messages,
            request.tools.as_ref(),
            request.add_generation_prompt,
        )?;
        if prefill.positions <= capacity {
            return Ok(CanonicalPrefill {
                prefill,
                replay_messages: messages,
                shifted: true,
            });
        }
        if turns.len() == 1 {
            return Err(ChatError::BadRequest(
                "system content, newest complete turn, and output reserve cannot fit".into(),
            ));
        }
        turns.remove(0);
    }
}

fn retained_context_capacity(
    max_context: usize,
    output_reserve: usize,
) -> Result<usize, ChatError> {
    let capacity = max_context.checked_sub(output_reserve).ok_or_else(|| {
        ChatError::BadRequest(format!(
            "output reserve {output_reserve} cannot fit the {} token context",
            max_context
        ))
    })?;
    if capacity == 0 {
        return Err(ChatError::BadRequest(
            "minimum retained context plus output reserve cannot fit".into(),
        ));
    }
    Ok(capacity)
}

fn compact_raw_prompt(
    tokens: &[u32],
    capacity: usize,
    retain_prefix: usize,
) -> Result<Vec<u32>, ChatError> {
    debug_assert!(tokens.len() > capacity);
    if retain_prefix >= capacity {
        return Err(ChatError::BadRequest(format!(
            "configured raw prefix {retain_prefix}, a nonempty newest suffix, and output reserve cannot fit"
        )));
    }
    let suffix = capacity - retain_prefix;
    let mut compact = Vec::with_capacity(capacity);
    compact.extend_from_slice(&tokens[..retain_prefix]);
    compact.extend_from_slice(&tokens[tokens.len() - suffix..]);
    Ok(compact)
}

/// A shift may remove only complete units beginning at a user boundary.
/// Assistant calls, tool results, and image-bearing messages remain attached
/// to their turn and therefore move or disappear as one replay unit.
fn complete_chat_turns(messages: &[Message]) -> Vec<Vec<Message>> {
    let mut turns: Vec<Vec<Message>> = Vec::new();
    for message in messages {
        if message.role == "user" || turns.is_empty() {
            turns.push(vec![message.clone()]);
        } else {
            turns
                .last_mut()
                .expect("created turn")
                .push(message.clone());
        }
    }
    turns
}

fn shift_chat_units(messages: &[Message]) -> Result<(Vec<Message>, Vec<Vec<Message>>), ChatError> {
    let system_count = messages
        .iter()
        .take_while(|message| message.role == "system")
        .count();
    if messages[system_count..]
        .iter()
        .any(|message| message.role == "system")
    {
        return Err(ChatError::BadRequest(
            "context shift requires all system messages to precede conversation turns".into(),
        ));
    }
    Ok((
        messages[..system_count].to_vec(),
        complete_chat_turns(&messages[system_count..]),
    ))
}

fn message_values(messages: &[Message]) -> Vec<serde_json::Value> {
    messages
        .iter()
        .map(|message| serde_json::to_value(message).expect("message serializes"))
        .collect()
}

fn contains_contiguous<T: PartialEq>(haystack: &[T], needle: &[T]) -> bool {
    needle.is_empty()
        || (needle.len() <= haystack.len()
            && haystack
                .windows(needle.len())
                .any(|window| window == needle))
}

/// Validate the caller's unabridged continuation before using either an exact
/// live frontier or a compacted replay plan. After an earlier shift, the
/// retained newest turns may occur after client-retained, already-dropped
/// history; they must still appear as one exact ordered run under identical
/// leading system content.
fn validate_session_lineage(
    bundle: &SessionBundle,
    request: &ChatRequest,
    prompt_witnesses: &[u32],
    raw_retain_prefix: usize,
) -> Result<bool, ChatError> {
    let previous_plan: Vec<Message> = serde_json::from_str(&bundle.canonical_replay_plan_json)
        .map_err(|_| ChatError::Conflict("stored canonical replay plan is invalid".into()))?;
    let exact_frontier = prompt_witnesses.starts_with(&bundle.position_witnesses);
    if let Some(tokens) = request.muser_prompt_token_ids.as_deref() {
        if !previous_plan.is_empty() {
            return Err(ChatError::Conflict(
                "raw continuation cannot replace a committed chat replay plan".into(),
            ));
        }
        let previous = bundle.target.tokens.as_ref();
        let extends = if bundle.context_epoch == 0 {
            tokens.starts_with(previous)
        } else {
            let prefix = raw_retain_prefix.min(previous.len());
            tokens.starts_with(&previous[..prefix])
                && contains_contiguous(&tokens[prefix.min(tokens.len())..], &previous[prefix..])
        };
        if !extends {
            return Err(ChatError::Conflict(
                "raw prompt does not extend the committed session frontier".into(),
            ));
        }
        return Ok(exact_frontier);
    }

    if previous_plan.is_empty() {
        return Err(ChatError::Conflict(
            "chat continuation cannot replace a committed raw replay plan".into(),
        ));
    }
    let (previous_systems, _) = shift_chat_units(&previous_plan).map_err(|_| {
        ChatError::Conflict("stored replay plan has non-leading system content".into())
    })?;
    let (current_systems, _) = shift_chat_units(&request.messages)?;
    let previous_values = message_values(&previous_plan);
    let current_values = message_values(&request.messages);
    let extends = if bundle.context_epoch == 0 || current_values.starts_with(&previous_values) {
        current_values.starts_with(&previous_values)
    } else {
        let previous_system_count = previous_systems.len();
        let current_system_count = current_systems.len();
        message_values(&previous_systems) == message_values(&current_systems)
            && contains_contiguous(
                &current_values[current_system_count..],
                &previous_values[previous_system_count..],
            )
    };
    if !extends {
        return Err(ChatError::Conflict(
            "canonical replay plan does not extend the committed session frontier".into(),
        ));
    }
    Ok(exact_frontier)
}

fn next_context_epoch(previous: u64, rebuilt: bool) -> Result<u64, ChatError> {
    previous
        .checked_add(u64::from(rebuilt))
        .ok_or_else(|| ChatError::Conflict("session context epoch overflow".into()))
}

impl PreparedPrefill {
    fn token_only(&self) -> Option<Vec<u32>> {
        let mut tokens = Vec::new();
        for segment in &self.segments {
            match segment {
                PreparedSegment::Tokens(value) => tokens.extend_from_slice(value),
                PreparedSegment::Image { .. } => return None,
            }
        }
        Some(tokens)
    }

    /// Run the expensive fifty-block vision graph only if local prefill is
    /// actually selected. A successful GX10 handoff therefore does not hide
    /// a complete projector execution inside the measured Mac TTFT.
    fn materialize(&self, runtime: &InferenceRuntime) -> Result<PrefillBatch, ChatError> {
        let mut segments = Vec::with_capacity(self.segments.len());
        for segment in &self.segments {
            match segment {
                PreparedSegment::Tokens(tokens) => {
                    segments.push(PrefillSegment::Tokens(tokens.clone()));
                }
                PreparedSegment::Image {
                    encoded,
                    preprocessed,
                    projected_tokens,
                } => {
                    let vision = runtime.vision.as_ref().ok_or_else(|| {
                        ChatError::Engine("prepared image lost its loaded vision model".into())
                    })?;
                    let vectors = vision
                        .encode_accelerated(encoded, preprocessed)
                        .map_err(|error| ChatError::Engine(error.to_string()))?;
                    if vectors.len() != *projected_tokens {
                        return Err(ChatError::Engine(format!(
                            "vision projector emitted {} rows, geometry promised {projected_tokens}",
                            vectors.len()
                        )));
                    }
                    segments.push(PrefillSegment::Embeddings(EmbeddingSegment::new(vectors)));
                }
            }
        }
        Ok(PrefillBatch { segments })
    }
}

fn vision_rows(batch: &PrefillBatch) -> Vec<Vec<f32>> {
    batch
        .segments
        .iter()
        .filter_map(|segment| match segment {
            PrefillSegment::Embeddings(embeddings) => Some(embeddings.vectors.iter().cloned()),
            PrefillSegment::Tokens(_) => None,
        })
        .flatten()
        .collect()
}

fn build_exact_token_prefill(tokens: &[u32]) -> Result<PreparedPrefill, ChatError> {
    if tokens.is_empty() {
        return Err(ChatError::BadRequest(
            "muser_prompt_token_ids must not be empty".into(),
        ));
    }
    Ok(PreparedPrefill {
        segments: vec![PreparedSegment::Tokens(tokens.to_vec())],
        witnesses: tokens.to_vec(),
        positions: tokens.len(),
        remote_multimodal: None,
    })
}

/// Prepare ordered token/image segments. Image projector rows occupy ordinary
/// decoder positions between Muse's official image sentinels, but projection
/// itself stays lazy so a remote hit can improve real end-to-end TTFT.
fn build_prefill(
    runtime: &InferenceRuntime,
    messages: &[Message],
    tools: Option<&Vec<ToolDefinition>>,
    add_generation_prompt: bool,
) -> Result<PreparedPrefill, ChatError> {
    let mut segments = Vec::new();
    let mut control_segments = Vec::new();
    let mut image_digests = Vec::new();
    let mut pending = PendingText::default();
    if runtime.model.adds_bos_token() {
        let bos = runtime.model.bos_token_id().ok_or_else(|| {
            ChatError::Engine("GGUF requests BOS insertion without a BOS token ID".into())
        })?;
        segments.push(PreparedSegment::Tokens(vec![bos]));
        control_segments.push(PrefillControlSegmentV2::Tokens {
            token_ids: vec![bos],
        });
    }

    enum Insertion {
        Text(String),
        Image(ImageUrl),
    }
    let mut insertions = Vec::<(String, Insertion)>::new();
    let mut template_messages = Vec::with_capacity(messages.len());
    let mut next_marker = 0usize;
    let mut marker = || {
        let value = format!("\u{e000}muser-content-{next_marker}\u{e001}");
        next_marker += 1;
        value
    };
    for message in messages {
        if !matches!(
            message.role.as_str(),
            "system" | "user" | "assistant" | "tool"
        ) {
            return Err(ChatError::BadRequest(format!(
                "unsupported message role '{}'",
                message.role
            )));
        }
        // The pinned Muse template renders an assistant message that carries
        // tool_calls as ATEM blocks only; its content is never rendered.
        // Sentinel-guarding that content would trip the dropped-content
        // check below, and the template ignores it either way.
        let template_renders_content = !(message.role == "assistant"
            && message
                .tool_calls
                .as_ref()
                .is_some_and(|calls| !calls.is_empty()));
        let content = match &message.content {
            _ if !template_renders_content => serde_json::Value::Null,
            MessageContent::Text(text) => {
                let sentinel = marker();
                insertions.push((sentinel.clone(), Insertion::Text(text.clone())));
                serde_json::Value::String(sentinel)
            }
            MessageContent::Parts(parts) => {
                let mut rendered_parts = Vec::with_capacity(parts.len());
                for part in parts {
                    match part {
                        ContentPart::Text { text } => {
                            let sentinel = marker();
                            insertions.push((sentinel.clone(), Insertion::Text(text.clone())));
                            rendered_parts.push(serde_json::json!({
                                "type": "text",
                                "text": sentinel,
                            }));
                        }
                        ContentPart::ImageUrl { image_url } => {
                            let sentinel = marker();
                            insertions
                                .push((sentinel.clone(), Insertion::Image(image_url.clone())));
                            rendered_parts.push(serde_json::json!({
                                "type": "text",
                                "text": sentinel,
                            }));
                        }
                    }
                }
                serde_json::Value::Array(rendered_parts)
            }
            MessageContent::Null(()) => serde_json::Value::Null,
        };
        let mut template_message = serde_json::to_value(message).expect("message serializes");
        crate::chat_template::normalize_tool_call_arguments(&mut template_message);
        template_message["content"] = content;
        template_messages.push(template_message);
    }
    let timestamp = crate::timefmt::now_rfc3339();
    let tools = tools.map(|value| serde_json::to_value(value).expect("tools serialize"));
    let rendered = crate::chat_template::render_with_options(
        runtime.model.chat_template(),
        &serde_json::Value::Array(template_messages),
        tools.as_ref(),
        &timestamp[..10],
        add_generation_prompt,
    )
    .map_err(|error| ChatError::BadRequest(format!("Muse chat template: {error}")))?;
    let mut cursor = rendered.as_str();
    for (sentinel, insertion) in insertions {
        let offset = cursor.find(&sentinel).ok_or_else(|| {
            ChatError::Engine("Muse chat template dropped or reordered request content".into())
        })?;
        pending.scaffold(&cursor[..offset]);
        match insertion {
            Insertion::Text(text) => pending.content(&text),
            Insertion::Image(image_url) => {
                flush_text_segment(
                    &runtime.model,
                    &mut pending,
                    &mut segments,
                    &mut control_segments,
                );
                let vision = runtime.vision.as_ref().ok_or_else(|| {
                    ChatError::BadRequest(
                        "image input requires the server to be started with --mmproj".into(),
                    )
                })?;
                let encoded = decode_image_data_url(&image_url.url)?;
                let image = vision
                    .preprocess_bytes(&encoded)
                    .map_err(|error| ChatError::BadRequest(error.to_string()))?;
                let projected_tokens = vision
                    .projected_token_count(&image)
                    .map_err(|error| ChatError::BadRequest(error.to_string()))?;
                let image_digest = Sha256::digest(&encoded);
                image_digests.extend_from_slice(&image_digest);
                control_segments.push(PrefillControlSegmentV2::Image {
                    data_base64: base64::engine::general_purpose::STANDARD.encode(&encoded),
                    sha256: format!("{image_digest:x}"),
                    projected_tokens: projected_tokens as u32,
                });
                segments.push(PreparedSegment::Image {
                    encoded,
                    preprocessed: image,
                    projected_tokens,
                });
            }
        }
        cursor = &cursor[offset + sentinel.len()..];
    }
    pending.scaffold(cursor);
    flush_text_segment(
        &runtime.model,
        &mut pending,
        &mut segments,
        &mut control_segments,
    );
    let positions = segments
        .iter()
        .map(|segment| match segment {
            PreparedSegment::Tokens(tokens) => tokens.len(),
            PreparedSegment::Image {
                projected_tokens, ..
            } => *projected_tokens,
        })
        .sum();
    if positions == 0 {
        return Err(ChatError::BadRequest(
            "prompt tokenized to an empty sequence".into(),
        ));
    }
    let remote_multimodal = if image_digests.is_empty() {
        None
    } else {
        let identity = runtime.vision_identity.as_ref().ok_or_else(|| {
            ChatError::Engine("vision model is loaded without a release identity".into())
        })?;
        Some((
            MultimodalIdentityV2 {
                projector_sha256: identity.projector_sha256.clone(),
                preprocessing_sha256: identity.preprocessing_sha256.clone(),
                image_sequence_sha256: format!("{:x}", Sha256::digest(&image_digests)),
            },
            control_segments,
        ))
    };
    let mut witnesses = Vec::with_capacity(positions);
    for segment in &segments {
        match segment {
            PreparedSegment::Tokens(tokens) => witnesses.extend_from_slice(tokens),
            PreparedSegment::Image {
                projected_tokens, ..
            } => witnesses.extend(std::iter::repeat_n(
                muser_engine::EMBEDDING_POSITION_WITNESS,
                *projected_tokens,
            )),
        }
    }
    Ok(PreparedPrefill {
        segments,
        witnesses,
        positions,
        remote_multimodal,
    })
}

/// Prompt text waiting to be tokenized, split into runs by who authored it.
/// Server-authored chat scaffolding is encoded with special-token parsing on;
/// client-authored message content is encoded with it off, so a message body
/// containing `<|im_end|>` becomes literal text instead of closing its turn
/// and opening another role.
#[derive(Default)]
struct PendingText {
    runs: Vec<(String, bool)>,
}

impl PendingText {
    fn scaffold(&mut self, text: &str) {
        self.push(text, true);
    }

    fn content(&mut self, text: &str) {
        self.push(text, false);
    }

    fn push(&mut self, text: &str, parse_special: bool) {
        if text.is_empty() {
            return;
        }
        match self.runs.last_mut() {
            Some((run, mode)) if *mode == parse_special => run.push_str(text),
            _ => self.runs.push((text.to_string(), parse_special)),
        }
    }

    fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }

    fn take_tokens(&mut self, model: &muser_engine::Model) -> Vec<u32> {
        let mut tokens = Vec::new();
        for (text, parse_special) in self.runs.drain(..) {
            tokens.extend_from_slice(&model.encode_with_options(&text, parse_special));
        }
        tokens
    }
}

fn flush_text_segment(
    model: &muser_engine::Model,
    pending: &mut PendingText,
    segments: &mut Vec<PreparedSegment>,
    control_segments: &mut Vec<PrefillControlSegmentV2>,
) {
    if pending.is_empty() {
        return;
    }
    let tokens = pending.take_tokens(model);
    if !tokens.is_empty() {
        control_segments.push(PrefillControlSegmentV2::Tokens {
            token_ids: tokens.clone(),
        });
        segments.push(PreparedSegment::Tokens(tokens));
    }
}

fn decode_image_data_url(url: &str) -> Result<Vec<u8>, ChatError> {
    const MAX_ENCODED_BYTES: usize = 48 * 1024 * 1024;
    const MAX_DECODED_BYTES: usize = 32 * 1024 * 1024;
    if url.len() > MAX_ENCODED_BYTES {
        return Err(ChatError::BadRequest("image data URL is too large".into()));
    }
    let (header, payload) = url
        .split_once(',')
        .ok_or_else(|| ChatError::BadRequest("image_url must be a base64 data:image URL".into()))?;
    if !header.starts_with("data:image/") || !header.ends_with(";base64") {
        return Err(ChatError::BadRequest(
            "image_url must be a base64 data:image URL".into(),
        ));
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload)
        .map_err(|_| ChatError::BadRequest("image_url contains invalid base64".into()))?;
    if bytes.len() > MAX_DECODED_BYTES {
        return Err(ChatError::BadRequest("decoded image is too large".into()));
    }
    Ok(bytes)
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0usize;
    for index in 1..values.len() {
        if values[index] > values[best] {
            best = index;
        }
    }
    best
}

struct StopFilter {
    stops: Vec<String>,
    pending: String,
    hold_bytes: usize,
    matched_stop: Option<String>,
}

impl StopFilter {
    fn new(stops: Vec<String>) -> Self {
        let hold_bytes = stops
            .iter()
            .map(String::len)
            .max()
            .unwrap_or(0)
            .saturating_sub(1);
        Self {
            stops: stops.into_iter().filter(|stop| !stop.is_empty()).collect(),
            pending: String::new(),
            hold_bytes,
            matched_stop: None,
        }
    }

    fn push(
        &mut self,
        piece: &str,
        mut emit: impl FnMut(&str) -> Result<(), ChatError>,
    ) -> Result<bool, ChatError> {
        self.pending.push_str(piece);
        if let Some((index, matched)) = self
            .stops
            .iter()
            .filter_map(|stop| self.pending.find(stop).map(|index| (index, stop.clone())))
            .min_by_key(|(index, _)| *index)
        {
            if index > 0 {
                emit(&self.pending[..index])?;
            }
            self.matched_stop = Some(matched);
            self.pending.clear();
            return Ok(true);
        }
        let mut safe = self.pending.len().saturating_sub(self.hold_bytes);
        while safe > 0 && !self.pending.is_char_boundary(safe) {
            safe -= 1;
        }
        if safe > 0 {
            emit(&self.pending[..safe])?;
            self.pending.drain(..safe);
        }
        Ok(false)
    }

    fn finish(
        &mut self,
        mut emit: impl FnMut(&str) -> Result<(), ChatError>,
    ) -> Result<(), ChatError> {
        if !self.pending.is_empty() {
            emit(&self.pending)?;
            self.pending.clear();
        }
        Ok(())
    }

    fn matched_stop(&self) -> Option<&str> {
        self.matched_stop.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_full_ladder_hit_maps_to_skipping_the_remote_transfer() {
        let offer = |action, source, matched_tokens| {
            Some(RemoteReuseOffer {
                action,
                source,
                matched_tokens,
            })
        };
        assert_eq!(
            remote_handoff_reuse(offer(
                RemoteReuseAction::ServeLocal,
                CacheSource::Resident,
                511
            )),
            Some(RemoteHandoffReuse::SkipRemote {
                matched: 511,
                source: CacheSource::Resident,
            })
        );
        assert_eq!(
            remote_handoff_reuse(offer(
                RemoteReuseAction::ArmDelta,
                CacheSource::Durable,
                256
            )),
            Some(RemoteHandoffReuse::ArmDelta {
                prefix_cut: 256,
                source: CacheSource::Durable,
            })
        );
        // A ladder miss and an unusable (unaligned) partial both run the
        // full transfer exactly as before.
        assert_eq!(remote_handoff_reuse(None), None);
        assert_eq!(
            remote_handoff_reuse(offer(
                RemoteReuseAction::FullTransfer,
                CacheSource::Resident,
                300
            )),
            None
        );
    }

    #[test]
    fn delta_arming_requires_a_fetched_aligned_prefix() {
        // The live session's own continuation prefills its suffix locally
        // for less than a handoff costs.
        assert_eq!(
            arm_remote_delta(CacheSource::CurrentSession, 256, 600),
            None
        );
        assert_eq!(arm_remote_delta(CacheSource::Resident, 256, 600), Some(256));
        assert_eq!(arm_remote_delta(CacheSource::Durable, 512, 600), Some(512));
        // Off the producer's cut alignment a partial hit arms nothing.
        assert_eq!(arm_remote_delta(CacheSource::Resident, 300, 600), None);
        // A hit reaching the held boundary token is served locally instead.
        assert_eq!(arm_remote_delta(CacheSource::Resident, 599, 600), None);
        assert_eq!(arm_remote_delta(CacheSource::Resident, 0, 600), None);
    }

    fn lineage_bundle(messages: &[Message], tokens: &[u32], context_epoch: u64) -> SessionBundle {
        let rng = muser_engine::sampling::Mt19937::new(7).snapshot();
        SessionBundle {
            schema: "muser.session-bundle.v3".into(),
            session_id: "lineage".into(),
            revision: 1,
            context_epoch,
            model_sha256: "00".repeat(32),
            tokenizer_sha256: [1; 32],
            template_sha256: [2; 32],
            layout_abi: "muse-kv-layout-v1".into(),
            dflash_identity_sha256: None,
            vision_projector_sha256: None,
            vision_preprocessing_sha256: None,
            target: muser_engine::cache::SessionCacheSnapshot {
                position: tokens.len() as u64,
                tokens: std::sync::Arc::from(tokens),
                elements_per_token: 1,
                layers: std::sync::Arc::from([]),
            },
            target_logits: vec![0.0],
            dflash: None,
            position_witnesses: tokens.to_vec(),
            rng_seed: 7,
            sampler_state: crate::session_store::SamplerStateSnapshot {
                distribution_rng: rng.clone(),
                xtc_rng: rng.clone(),
                mirostat_rng: rng.clone(),
                adaptive_rng: rng,
                mirostat_mu: 10.0,
                adaptive_weighted_sum: 0.0,
                adaptive_total_weight: 1.0,
            },
            sampler_config_sha256: [3; 32],
            sampler_history: tokens.to_vec(),
            detokenizer_pending: String::new(),
            stop_matcher_pending: String::new(),
            grammar_state: None,
            grammar_sha256: None,
            canonical_replay_plan_json: serde_json::to_string(messages).unwrap(),
            vision_rows: Vec::new(),
        }
    }

    #[test]
    fn logprob_partial_sort_matches_pinned_libcxx_tail_order() {
        let logits = [
            1.0, -2.0, 5.0, 3.0, 0.5, 7.0, -4.0, 2.0, 6.0, -1.0, 4.0, 0.0,
        ];
        assert_eq!(
            source_partial_sort_order(&logits, 4),
            [5, 8, 2, 10, 1, 4, 6, 0, 7, 9, 3, 11]
        );
    }

    #[test]
    fn stop_filter_holds_cross_token_suffix() {
        let mut filter = StopFilter::new(vec!["END".into()]);
        let mut output = String::new();
        assert!(!filter
            .push("hello E", |s| {
                output.push_str(s);
                Ok(())
            })
            .unwrap());
        assert!(filter
            .push("ND ignored", |s| {
                output.push_str(s);
                Ok(())
            })
            .unwrap());
        assert_eq!(output, "hello ");
    }

    #[test]
    fn shed_requests_are_retryable_rate_limit_errors() {
        let (status, _, kind) = ChatError::Overloaded.status();
        assert_eq!(status, 429);
        assert_eq!(kind, "rate_limit_exceeded");
    }

    #[test]
    fn poisoned_accelerator_errors_are_restart_required_service_unavailable() {
        let (status, _, kind) = slot_error_to_chat(SlotAcquireError::Unhealthy).status();
        assert_eq!(status, 503);
        assert_eq!(kind, "engine_unavailable");
    }

    #[test]
    fn failed_staging_rebuild_never_replaces_the_live_generation() {
        let mut live = "committed";
        let mut staging = "partial";
        let failure = swap_staging_on_success(&mut live, &mut staging, Err("prefill failed"));
        assert_eq!(failure, Err("prefill failed"));
        assert_eq!(live, "committed");
        assert_eq!(staging, "partial");

        swap_staging_on_success::<_, &str>(&mut live, &mut staging, Ok(())).unwrap();
        assert_eq!(live, "partial");
        assert_eq!(staging, "committed");
    }

    #[test]
    fn failed_paired_staging_keeps_both_live_frontiers() {
        let mut live_target = "target-live";
        let mut staging_target = "target-partial";
        let mut live_dflash = "dflash-live";
        let mut staging_dflash = "dflash-partial";
        let failure = swap_staging_pair_on_success(
            &mut live_target,
            &mut staging_target,
            &mut live_dflash,
            &mut staging_dflash,
            Err("dflash prepare failed"),
        );
        assert_eq!(failure, Err("dflash prepare failed"));
        assert_eq!((live_target, live_dflash), ("target-live", "dflash-live"));
        assert_eq!(
            (staging_target, staging_dflash),
            ("target-partial", "dflash-partial")
        );

        swap_staging_pair_on_success::<_, _, &str>(
            &mut live_target,
            &mut staging_target,
            &mut live_dflash,
            &mut staging_dflash,
            Ok(()),
        )
        .unwrap();
        assert_eq!(
            (live_target, live_dflash),
            ("target-partial", "dflash-partial")
        );
        assert_eq!(
            (staging_target, staging_dflash),
            ("target-live", "dflash-live")
        );
    }

    #[test]
    fn the_verification_length_has_one_source() {
        // The ledger's fixed-window selection; `metrics` reports this same
        // call, so a tuning override can never disagree with what the route ran.
        assert_eq!(DFLASH_VERIFY_LEN, 7);
        assert!(dflash_verify_len() > 0);
    }

    #[test]
    fn entropy_seeds_differ_between_requests() {
        assert_ne!(entropy_seed(), entropy_seed());
    }

    #[test]
    fn pinned_seed_parser_accepts_random_sentinel_and_rejects_other_negatives() {
        #[derive(Deserialize)]
        struct SeedOnly {
            #[serde(default, deserialize_with = "deserialize_seed")]
            seed: Option<u64>,
        }
        assert_eq!(
            serde_json::from_str::<SeedOnly>(r#"{"seed":-1}"#)
                .unwrap()
                .seed,
            None
        );
        assert_eq!(
            serde_json::from_str::<SeedOnly>(r#"{"seed":4294967295}"#)
                .unwrap()
                .seed,
            Some(u64::from(u32::MAX))
        );
        assert!(serde_json::from_str::<SeedOnly>(r#"{"seed":-2}"#).is_err());
        assert!(serde_json::from_str::<SeedOnly>(r#"{"seed":4294967296}"#).is_err());
    }

    #[test]
    fn pinned_slot_and_logit_bias_forms_are_strict() {
        let defaulted: ChatRequest = serde_json::from_value(serde_json::json!({
            "messages": [{"role":"user", "content":"hi"}]
        }))
        .unwrap();
        assert_eq!(defaulted.model, MODEL_ID);
        let idle: ChatRequest = serde_json::from_value(serde_json::json!({
            "model": MODEL_ID,
            "messages": [{"role":"user", "content":"hi"}],
            "id_slot": -1,
            "logit_bias": [[7, false], ["literal", -0.25]]
        }))
        .unwrap();
        assert_eq!(idle.id_slot, None);
        assert!(idle.cache_prompt);
        assert!(idle.add_generation_prompt);
        let biases = idle.logit_bias.unwrap();
        assert_eq!(biases["7"], f32::NEG_INFINITY);
        assert_eq!(biases["literal"], -0.25);
        assert!(serde_json::from_value::<ChatRequest>(serde_json::json!({
            "model": MODEL_ID,
            "messages": [{"role":"user", "content":"hi"}],
            "id_slot": -2
        }))
        .is_err());
        assert!(serde_json::from_value::<ChatRequest>(serde_json::json!({
            "model": MODEL_ID,
            "messages": [{"role":"user", "content":"hi"}],
            "logit_bias": [[1, true]]
        }))
        .is_err());
        let continuation: ChatRequest = serde_json::from_value(serde_json::json!({
            "model": MODEL_ID,
            "messages": [{"role":"assistant", "content":"partial"}],
            "add_generation_prompt": false
        }))
        .unwrap();
        assert!(!continuation.add_generation_prompt);
    }

    #[test]
    fn sampler_softmax_does_not_assume_token_zero_is_the_maximum() {
        let probabilities = candidate_softmax(&[(0, -1_000.0), (1, 0.0), (2, -1.0)]).unwrap();
        assert_eq!(probabilities[0], 0.0);
        assert!(probabilities[1] > probabilities[2]);
        assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn stateful_basic_sampling_is_dflash_compatible() {
        let mut request: ChatRequest = serde_json::from_value(serde_json::json!({
            "model": MODEL_ID,
            "messages": [{"role":"user", "content":"continue"}],
            "session_id":"session-1",
            "expected_revision":3,
            "temperature":0.8,
            "top_p":0.95
        }))
        .unwrap();
        request.idempotency_key = Some("request-4".into());
        assert!(dflash_sampling_compatible(&request));

        let mut constrained: ChatRequest = serde_json::from_value(serde_json::json!({
            "model": MODEL_ID,
            "messages": [{"role":"user", "content":"continue"}],
            "session_id":"session-1",
            "expected_revision":3,
            "temperature":0.8,
            "grammar":"root ::= \"ok\""
        }))
        .unwrap();
        constrained.idempotency_key = Some("request-4".into());
        assert!(!dflash_sampling_compatible(&constrained));
    }

    #[test]
    fn reasoning_control_injects_one_complete_marker_and_never_reopens() {
        let signal = Arc::new(AtomicBool::new(true));
        let request = ChatRequest {
            model: MODEL_ID.into(),
            messages: Vec::new(),
            stream: true,
            stream_options: None,
            max_tokens: None,
            max_completion_tokens: None,
            t_max_predict_ms: None,
            temperature: None,
            top_p: None,
            top_k: None,
            typical_p: None,
            min_p: None,
            top_n_sigma: None,
            min_keep: None,
            ignore_eos: false,
            logit_bias: None,
            repeat_penalty: None,
            repeat_last_n: None,
            presence_penalty: None,
            frequency_penalty: None,
            dry_multiplier: None,
            dry_base: None,
            dry_allowed_length: None,
            dry_penalty_last_n: None,
            dry_sequence_breakers: None,
            mirostat: None,
            mirostat_tau: None,
            mirostat_eta: None,
            adaptive_target: None,
            adaptive_decay: None,
            dynatemp_range: None,
            dynatemp_exponent: None,
            xtc_probability: None,
            xtc_threshold: None,
            samplers: None,
            reasoning_control: true,
            reasoning_end_signal: Some(Arc::clone(&signal)),
            seed: None,
            n: None,
            id_slot: None,
            cache_prompt: false,
            stop: None,
            tools: None,
            tool_choice: None,
            add_generation_prompt: true,
            parallel_tool_calls: true,
            logprobs: None,
            top_logprobs: None,
            response_format: None,
            grammar: None,
            json_schema: None,
            muser_prompt_token_ids: None,
            muser_baseline_ttft: false,
            session_id: None,
            expected_revision: None,
            idempotency_key: None,
            idempotency_request_sha256: None,
        };
        let marker = VecDeque::from([7, 8, 9]);
        let mut pending = VecDeque::new();
        assert_eq!(
            take_forced_reasoning_token(&request, &marker, &mut pending, false),
            Some(7)
        );
        assert_eq!(
            take_forced_reasoning_token(&request, &marker, &mut pending, false),
            Some(8)
        );
        assert_eq!(
            take_forced_reasoning_token(&request, &marker, &mut pending, false),
            Some(9)
        );
        assert_eq!(
            take_forced_reasoning_token(&request, &marker, &mut pending, false),
            None
        );
        signal.store(true, Ordering::Release);
        assert_eq!(
            take_forced_reasoning_token(&request, &marker, &mut pending, true),
            None
        );
        assert!(pending.is_empty());
    }

    #[test]
    fn atem_reasoning_parallel_calls_and_final_content_parse() {
        let parsed = parse_atem_output(
            " to=self<|message|>think<|eom|><|start|>assistant to=weather.get<|message|>\
             <atem:function_calls>\n<atem:invoke name=\"weather.get\">\n\
             <atem:parameter name=\"city\">Zurich</atem:parameter>\n</atem:invoke>\n\
             <atem:invoke name=\"clock.get\">\n<atem:parameter name=\"offset\">2</atem:parameter>\n\
             </atem:invoke>\n</atem:function_calls><|eom|>\
             <|start|>assistant to=user<|message|>done",
        )
        .unwrap();
        assert_eq!(parsed.reasoning, "think");
        assert_eq!(parsed.content, "done");
        assert_eq!(parsed.tool_calls.len(), 2);
        assert_eq!(parsed.tool_calls[0].function.name, "weather.get");
        assert_eq!(
            parsed.tool_calls[0].function.arguments,
            r#"{"city":"Zurich"}"#
        );
        assert_eq!(parsed.tool_calls[1].function.arguments, r#"{"offset":2}"#);
        assert!(parse_atem_output(
            " to=clock.get<|message|><atem:function_calls>\
             <atem:invoke name=\"weather.get\"></atem:invoke>\
             </atem:function_calls><|eom|>"
        )
        .is_err());
    }

    #[test]
    fn required_tool_choice_installs_a_closed_atem_decode_grammar() {
        let request: ChatRequest = serde_json::from_value(serde_json::json!({
            "model": MODEL_ID,
            "messages": [{"role":"user", "content":"weather?"}],
            "tools": [{
                "type":"function",
                "function": {
                    "name":"weather.get",
                    "parameters": {
                        "type":"object",
                        "properties":{"city":{"type":"string"}},
                        "required":["city"],
                        "additionalProperties":false
                    }
                }
            }],
            "tool_choice":"required"
        }))
        .unwrap();
        let source = constrained_grammar_source(&request)
            .unwrap()
            .expect("required tool grammar");
        let accepts = |text: &str| {
            let mut matcher = GrammarMatcher::parse(&source, "root").unwrap();
            matcher.accept_bytes(text.as_bytes()).is_ok() && matcher.is_accepting()
        };
        assert!(accepts(
            " to=weather.get<|message|><atem:function_calls>\n\
             <atem:invoke name=\"weather.get\"><atem:parameter name=\"city\">\"Zurich\"\
             </atem:parameter></atem:invoke></atem:function_calls><|eom|>"
        ));
        assert!(!accepts(
            " to=shell.exec<|message|><atem:function_calls><atem:invoke name=\"shell.exec\">\
             </atem:invoke></atem:function_calls><|eom|>"
        ));
        assert!(!accepts(
            " to=weather.get<|message|><atem:function_calls><atem:invoke name=\"weather.get\">"
        ));

        let mut serial_request = request.clone();
        serial_request.parallel_tool_calls = false;
        let serial_source = constrained_grammar_source(&serial_request)
            .unwrap()
            .expect("serial required-tool grammar");
        let mut serial = GrammarMatcher::parse(&serial_source, "root").unwrap();
        let duplicate = " to=weather.get<|message|><atem:function_calls>\
             <atem:invoke name=\"weather.get\"><atem:parameter name=\"city\">\"Zurich\"\
             </atem:parameter></atem:invoke>\
             <atem:invoke name=\"weather.get\"><atem:parameter name=\"city\">\"Bern\"\
             </atem:parameter></atem:invoke></atem:function_calls><|eom|>";
        assert!(serial.accept_bytes(duplicate.as_bytes()).is_err() || !serial.is_accepting());
    }

    #[test]
    fn chat_schema_constrains_public_answer_after_muse_reasoning() {
        let request: ChatRequest = serde_json::from_value(serde_json::json!({
            "model": MODEL_ID,
            "messages": [{"role":"user", "content":"answer as JSON"}],
            "response_format": {
                "type":"json_schema",
                "json_schema": {
                    "name":"answer",
                    "strict":true,
                    "schema": {
                        "type":"object",
                        "properties":{"ok":{"type":"boolean"}},
                        "required":["ok"],
                        "additionalProperties":false
                    }
                }
            }
        }))
        .unwrap();
        let source = constrained_grammar_source(&request)
            .unwrap()
            .expect("chat response grammar");
        let accepts = |text: &str| {
            let mut matcher = GrammarMatcher::parse(&source, "root").unwrap();
            matcher.accept_bytes(text.as_bytes()).is_ok() && matcher.is_accepting()
        };
        assert!(accepts(
            " to=self<|message|>think first<|eom|><|start|>assistant to=user<|message|>{ \"ok\": true }"
        ));
        assert!(accepts("{ \"ok\": false }"));
        assert!(!accepts(
            " to=self<|message|>think first<|eom|><|start|>assistant to=user<|message|>{ \"ok\": 7 }"
        ));

        let mut raw = request;
        raw.muser_prompt_token_ids = Some(vec![19873]);
        let bare = constrained_grammar_source(&raw)
            .unwrap()
            .expect("raw response grammar");
        let mut matcher = GrammarMatcher::parse(&bare, "root").unwrap();
        assert!(matcher.accept_bytes(b" to=self<|message|>think").is_err());
    }

    #[test]
    fn streamed_tool_arguments_are_validated_before_emission() {
        let request: ChatRequest = serde_json::from_value(serde_json::json!({
            "model": MODEL_ID,
            "messages": [{"role":"user", "content":"weather?"}],
            "tools": [{
                "type":"function",
                "function": {
                    "name":"weather.get",
                    "parameters": {
                        "type":"object",
                        "properties":{"city":{"type":"string"}},
                        "required":["city"],
                        "additionalProperties":false
                    }
                }
            }],
            "tool_choice":"required"
        }))
        .unwrap();
        let invalid = ParsedToolCall {
            id: "call-invalid".into(),
            kind: "function",
            function: ParsedFunctionCall {
                name: "weather.get".into(),
                arguments: r#"{"city":7}"#.into(),
            },
        };
        assert!(validate_streamed_atem_call(&request, &invalid).is_err());
        let valid = ParsedToolCall {
            id: "call-valid".into(),
            kind: "function",
            function: ParsedFunctionCall {
                name: "weather.get".into(),
                arguments: r#"{"city":"Zurich"}"#.into(),
            },
        };
        validate_streamed_atem_call(&request, &valid).unwrap();
    }

    #[test]
    fn malformed_atem_is_not_silently_returned_as_text() {
        assert!(parse_atem_output(
            " to=weather.get<|message|><atem:function_calls><atem:invoke name=\"weather.get\">"
        )
        .is_err());
    }

    #[test]
    fn length_truncated_recipient_header_remains_compatible_content() {
        let request: ChatRequest = serde_json::from_value(serde_json::json!({
            "model": MODEL_ID,
            "messages": [{"role":"user", "content":"hello"}]
        }))
        .unwrap();
        validate_generated_atem(&request, " to=self", true).unwrap();
        assert!(validate_generated_atem(&request, " to=self", false).is_err());

        let required: ChatRequest = serde_json::from_value(serde_json::json!({
            "model": MODEL_ID,
            "messages": [{"role":"user", "content":"hello"}],
            "tools": [{"type":"function", "function":{"name":"clock.now", "parameters":{"type":"object"}}}],
            "tool_choice":"required"
        }))
        .unwrap();
        assert!(validate_generated_atem(&required, " to=clock.now", true).is_err());

        let mut streamed = AtemStreamParser::default();
        assert!(streamed.push(" to=self").unwrap().is_empty());
        assert!(streamed.finish_stream().unwrap().is_empty());
    }

    #[test]
    fn atem_stream_parser_emits_reasoning_calls_and_content_before_finish() {
        let mut parser = AtemStreamParser::default();
        assert!(parser.push(" to=se").unwrap().is_empty());
        let reasoning = parser.push("lf<|message|>rea").unwrap();
        assert!(matches!(
            reasoning.as_slice(),
            [AtemStreamEvent::Reasoning(text)] if text == "rea"
        ));
        let reasoning = parser.push("son<|eo").unwrap();
        assert!(matches!(
            reasoning.as_slice(),
            [AtemStreamEvent::Reasoning(text)] if text == "son"
        ));
        let calls = parser
            .push(
                "m|><|start|>assistant to=weather.get<|message|>\
                 <atem:function_calls><atem:invoke name=\"weather.get\">\
                 <atem:parameter name=\"city\">Zurich</atem:parameter>\
                 </atem:invoke></atem:function_calls><|eom|>",
            )
            .unwrap();
        assert!(matches!(
            calls.as_slice(),
            [AtemStreamEvent::ToolCall { index: 0, call }]
                if call.function.name == "weather.get"
                    && call.function.arguments == r#"{"city":"Zurich"}"#
        ));
        let content = parser
            .push("<|start|>assistant to=user<|message|>do")
            .unwrap();
        assert!(matches!(
            content.as_slice(),
            [AtemStreamEvent::Content(text)] if text == "do"
        ));
        let content = parser.push("ne").unwrap();
        assert!(matches!(
            content.as_slice(),
            [AtemStreamEvent::Content(text)] if text == "ne"
        ));
        assert!(parser.finish().unwrap().is_empty());
        assert!(parser.is_structured());
    }

    #[test]
    fn atem_stream_parser_passes_plain_text_and_fails_dangling_tools() {
        let mut plain = AtemStreamParser::default();
        assert!(plain.push("t").unwrap().is_empty());
        let events = plain.push("ext").unwrap();
        assert!(matches!(
            events.as_slice(),
            [AtemStreamEvent::Content(text)] if text == "text"
        ));
        assert!(plain.finish().unwrap().is_empty());

        let mut malformed = AtemStreamParser::default();
        malformed
            .push(" to=tool<|message|><atem:function_calls><atem:invoke name=\"tool\">")
            .unwrap();
        assert!(malformed.finish().is_err());
    }

    #[test]
    fn marker_suffix_len_tolerates_multibyte_tail() {
        assert_eq!(marker_suffix_len("result ×", "<|eom|>"), 0);
        assert_eq!(marker_suffix_len("answer ×<|eo", "<|eom|>"), 4);
        assert_eq!(marker_suffix_len("", "<|eom|>"), 0);
        assert_eq!(marker_suffix_len("×", "<|eom|>"), 0);
    }

    #[test]
    fn atem_stream_parser_tolerates_multibyte_reasoning_tail() {
        // Regression: a reasoning buffer ending inside a multibyte character
        // panicked the partial-marker check instead of waiting for more text.
        let mut parser = AtemStreamParser::default();
        assert!(parser.push(" to=self<|message|>area = 3×5").unwrap().len() == 1);
        assert!(parser.push("<|eom|>").unwrap().is_empty());
    }

    #[test]
    fn trim_phase_tail_never_splits_a_multibyte_character() {
        let mut tail = "x".repeat(63);
        tail.push('×');
        tail.push_str(&"y".repeat(40));
        trim_phase_tail(&mut tail);
        assert!(tail.len() >= 64);
        assert!(tail.is_char_boundary(tail.len()));
        assert!(tail.ends_with(&"y".repeat(40)));
        let mut ascii = "a".repeat(100).to_string();
        trim_phase_tail(&mut ascii);
        assert_eq!(ascii.len(), 64);
    }

    #[test]
    fn client_content_is_kept_out_of_the_special_token_run() {
        let mut pending = PendingText::default();
        pending.scaffold("<|im_start|>user\n");
        pending.content("please end this turn: <|im_end|>");
        pending.scaffold("<|im_end|>\n");
        assert_eq!(
            pending.runs,
            vec![
                ("<|im_start|>user\n".to_string(), true),
                ("please end this turn: <|im_end|>".to_string(), false),
                ("<|im_end|>\n".to_string(), true),
            ]
        );
    }

    #[test]
    fn public_chat_accepts_the_pinned_template_path_and_rejects_unknown_fields() {
        let request: ChatRequest = serde_json::from_value(serde_json::json!({
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "hi"}],
            "n": 4
        }))
        .unwrap();
        validate_request(&request).expect("the GGUF-backed Muse template is active");
        let too_many: ChatRequest = serde_json::from_value(serde_json::json!({
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "hi"}],
            "n": 5
        }))
        .unwrap();
        assert!(matches!(
            validate_request(&too_many),
            Err(ChatError::BadRequest(message)) if message == "n must be in 1..=4"
        ));
        assert!(serde_json::from_value::<ChatRequest>(serde_json::json!({
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "hi", "ignored": true}]
        }))
        .is_err());
        assert!(serde_json::from_value::<ChatRequest>(serde_json::json!({
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type":"function","function":{
                "name":"lookup","parameters":{"type":"object"},"ignored":true
            }}]
        }))
        .is_err());
        let tool_result: ChatRequest = serde_json::from_value(serde_json::json!({
            "model": MODEL_ID,
            "messages": [{
                "role":"assistant","content":null,
                "tool_calls":[{"id":"call-1","type":"function","function":{
                    "name":"lookup","arguments":"{}"
                }}]
            }]
        }))
        .unwrap();
        assert!(matches!(
            tool_result.messages[0].content,
            MessageContent::Null(())
        ));

        let ordered: ChatRequest = serde_json::from_value(serde_json::json!({
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "hi"}],
            "samplers": ["temp", "top-k", "temperature", "penalties"]
        }))
        .unwrap();
        assert_eq!(
            ordered.samplers.as_deref(),
            Some(
                ["temperature", "top_k", "temperature", "penalties"]
                    .map(str::to_string)
                    .as_slice()
            )
        );
        validate_request(&ordered).unwrap();

        let compact: ChatRequest = serde_json::from_value(serde_json::json!({
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "hi"}],
            "samplers": "tkkt"
        }))
        .unwrap();
        assert_eq!(
            compact.samplers,
            Some(
                vec!["temperature", "top_k", "top_k", "temperature"]
                    .into_iter()
                    .map(str::to_string)
                    .collect()
            )
        );
        let infill: ChatRequest = serde_json::from_value(serde_json::json!({
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": "hi"}],
            "samplers": ["infill"]
        }))
        .unwrap();
        assert!(validate_request(&infill).is_err());
    }

    #[test]
    fn raw_context_shift_keeps_exact_prefix_and_newest_suffix() {
        let tokens = (0..20).collect::<Vec<u32>>();
        assert_eq!(
            compact_raw_prompt(&tokens, 8, 3).unwrap(),
            vec![0, 1, 2, 15, 16, 17, 18, 19]
        );
        assert!(matches!(
            compact_raw_prompt(&tokens, 4, 99),
            Err(ChatError::BadRequest(message)) if message.contains("nonempty newest suffix")
        ));
        assert_eq!(
            compact_raw_prompt(&tokens, 4, 0).unwrap(),
            vec![16, 17, 18, 19]
        );
    }

    #[test]
    fn chat_context_shift_groups_tool_and_image_units_with_whole_turns() {
        let messages: Vec<Message> = serde_json::from_value(serde_json::json!([
            {"role":"user","content":[{"type":"text","text":"first"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]},
            {"role":"assistant","content":"","tool_calls":[{"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},
            {"role":"tool","tool_call_id":"call-1","content":"result"},
            {"role":"assistant","content":"answer"},
            {"role":"user","content":"second"},
            {"role":"assistant","content":"newest"}
        ])).unwrap();
        let expected = serde_json::to_value(&messages).unwrap();
        let turns = complete_chat_turns(&messages);
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].len(), 4);
        assert_eq!(turns[1].len(), 2);
        assert_eq!(serde_json::to_value(turns.concat()).unwrap(), expected);
    }

    #[test]
    fn context_shift_preserves_every_valid_system_message_and_rejects_late_systems() {
        let messages: Vec<Message> = serde_json::from_value(serde_json::json!([
            {"role":"system","content":"policy"},
            {"role":"system","content":"persona"},
            {"role":"user","content":"old"},
            {"role":"assistant","content":"answer"},
            {"role":"user","content":"new"}
        ]))
        .unwrap();
        let (systems, turns) = shift_chat_units(&messages).unwrap();
        assert_eq!(systems.len(), 2);
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0].len(), 2);
        assert_eq!(turns[1].len(), 1);

        let late_system: Vec<Message> = serde_json::from_value(serde_json::json!([
            {"role":"user","content":"old"},
            {"role":"system","content":"must never be dropped"},
            {"role":"user","content":"new"}
        ]))
        .unwrap();
        assert!(matches!(
            shift_chat_units(&late_system),
            Err(ChatError::BadRequest(message)) if message.contains("all system messages")
        ));
    }

    #[test]
    fn shifted_chat_lineage_accepts_retained_turns_but_not_substitution() {
        let previous: Vec<Message> = serde_json::from_value(serde_json::json!([
            {"role":"system","content":"policy"},
            {"role":"user","content":"retained"},
            {"role":"assistant","content":"retained answer"}
        ]))
        .unwrap();
        let current: ChatRequest = serde_json::from_value(serde_json::json!({
            "messages":[
                {"role":"system","content":"policy"},
                {"role":"user","content":"old client-retained turn"},
                {"role":"assistant","content":"old answer"},
                {"role":"user","content":"retained"},
                {"role":"assistant","content":"retained answer"},
                {"role":"user","content":"new"}
            ]
        }))
        .unwrap();
        let shifted = lineage_bundle(&previous, &[10, 11, 12], 1);
        assert!(!validate_session_lineage(&shifted, &current, &[99], 2).unwrap());

        let unshifted = lineage_bundle(&previous, &[10, 11, 12], 0);
        assert!(validate_session_lineage(&unshifted, &current, &[99], 2).is_err());

        let substituted: ChatRequest = serde_json::from_value(serde_json::json!({
            "messages":[
                {"role":"system","content":"policy"},
                {"role":"user","content":"different"},
                {"role":"assistant","content":"retained answer"},
                {"role":"user","content":"new"}
            ]
        }))
        .unwrap();
        assert!(validate_session_lineage(&shifted, &substituted, &[99], 2).is_err());
    }

    #[test]
    fn shifted_raw_lineage_accepts_prefix_and_retained_suffix_only() {
        let bundle = lineage_bundle(&[], &[0, 1, 8, 9, 10], 1);
        let mut request: ChatRequest = serde_json::from_value(serde_json::json!({
            "messages":[],
            "muser_prompt_token_ids":[0,1,2,3,4,5,8,9,10,11]
        }))
        .unwrap();
        assert!(!validate_session_lineage(&bundle, &request, &[0, 1, 2], 2).unwrap());
        request.muser_prompt_token_ids = Some(vec![0, 1, 2, 3, 8, 7, 10, 11]);
        assert!(validate_session_lineage(&bundle, &request, &[0, 1, 2], 2).is_err());
    }

    #[test]
    fn context_epoch_is_checked_and_advances_only_for_rebuilds() {
        assert_eq!(next_context_epoch(7, false).unwrap(), 7);
        assert_eq!(next_context_epoch(7, true).unwrap(), 8);
        assert!(next_context_epoch(u64::MAX, true).is_err());
    }

    #[test]
    fn output_reserve_always_leaves_a_nonempty_retained_context() {
        assert_eq!(retained_context_capacity(16, 15).unwrap(), 1);
        for reserve in [16, 17] {
            let error = retained_context_capacity(16, reserve).unwrap_err();
            let (status, _, _) = error.status();
            assert_eq!(status, 400);
        }
    }
}
