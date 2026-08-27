//! Durable, authenticated carried-frontier protocol for a remote verifier.
//!
//! Ordered token decisions are single-writer state. Immutable capture/KV
//! fragments may arrive in any order, but cannot become visible until their
//! complete authenticated closure is staged. A commit is written and fsynced
//! before renderer activation; recovery finishes any committed activation
//! before admitting more work. CRDT/set convergence is therefore payload
//! transport only and never chooses a token branch.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

use kvpack_handoff::{canonical_json, sha256_hex, MacKey};
use muser_engine::sampling::{
    Mt19937, Mt19937Snapshot, SparseProbabilityEntry, SparseProbabilityRow,
};
use ring::signature::{Ed25519KeyPair, KeyPair, UnparsedPublicKey, ED25519};
use serde::{Deserialize, Serialize};

pub const VERIFIER_SESSION_PROTOCOL_V2: &str = "muser-verifier-session-v2";
pub const VERIFIER_ROUND_PROTOCOL_V2: &str = "muser-verifier-round-v2";
/// Round v3 adds an authenticated verify/close intent.  The v2 representation
/// remains readable for historical verify rounds and retains its exact
/// canonical serialization (the default `Verify` intent is omitted).
pub const VERIFIER_ROUND_PROTOCOL_V3: &str = "muser-verifier-round-v3";
pub const VERIFIER_RESULT_PROTOCOL_V2: &str = "muser-verifier-result-v2";
pub const VERIFIER_SESSION_MAC_DOMAIN_V2: &[u8] = b"muser-verifier-session-v2";
pub const VERIFIER_ROUND_MAC_DOMAIN_V2: &[u8] = b"muser-verifier-round-v2";
pub const VERIFIER_ROUND_MAC_DOMAIN_V3: &[u8] = b"muser-verifier-round-v3";
pub const VERIFIER_JOURNAL_MAC_DOMAIN_V2: &[u8] = b"muser-verifier-journal-v2";
pub const STATELESS_SAMPLER_V2: &str = "muser-stateless-sampler-v2";
pub const MAX_DRAFTS_V2: usize = 64;
pub const MAX_SPARSE_SUPPORT_V2: usize = 4_096;
pub const MAX_CONTEXT_TOKENS_V2: usize = 131_072;
pub const MAX_FRAGMENT_BYTES_V2: u64 = 512 * 1024 * 1024;
pub const MAX_TOTAL_FRAGMENT_BYTES_V2: u64 = 512 * 1024 * 1024;
pub const MAX_FRAGMENTS_V2: usize = 4_096;
pub const MAX_CLOCK_SKEW_MS_V2: u64 = 30_000;

#[derive(Debug, thiserror::Error)]
pub enum VerifierV2Error {
    #[error("verifier V2 validation: {0}")]
    Validation(String),
    #[error("verifier V2 authentication failed")]
    Authentication,
    #[error("verifier V2 authority lease is not live")]
    LeaseNotLive,
    #[error("verifier V2 request changed intent")]
    ChangedIntent,
    #[error("verifier V2 parent is stale")]
    StaleHead,
    #[error("verifier V2 parent already has an in-flight child")]
    BusyParent,
    #[error("verifier V2 session is terminal")]
    Terminal,
    #[error("verifier V2 closure is incomplete")]
    Incomplete,
    #[error("verifier V2 retry is outside retained history")]
    ResyncRequired,
    #[error("verifier V2 renderer: {0}")]
    Renderer(String),
    #[error("verifier V2 I/O: {0}")]
    Io(String),
}

impl PartialEq for VerifierV2Error {
    fn eq(&self, other: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
            && self.to_string() == other.to_string()
    }
}

impl Eq for VerifierV2Error {}

impl From<io::Error> for VerifierV2Error {
    fn from(value: io::Error) -> Self {
        Self::Io(value.to_string())
    }
}

type Result<T> = std::result::Result<T, VerifierV2Error>;

fn invalid(message: impl Into<String>) -> VerifierV2Error {
    VerifierV2Error::Validation(message.into())
}

fn canonical<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    canonical_json(value).map_err(|error| invalid(error.to_string()))
}

fn validate_digest(name: &str, digest: &str) -> Result<()> {
    if digest.len() != 64
        || digest
            .bytes()
            .any(|byte| !byte.is_ascii_digit() && !(b'a'..=b'f').contains(&byte))
    {
        return Err(invalid(format!("{name} is not lowercase SHA-256")));
    }
    Ok(())
}

fn validate_identifier(name: &str, value: &str) -> Result<()> {
    if value.is_empty()
        || value.len() > 128
        || value
            .bytes()
            .any(|byte| !(byte.is_ascii_alphanumeric() || b"._:-".contains(&byte)))
    {
        return Err(invalid(format!("{name} is invalid")));
    }
    Ok(())
}

fn token_digest(tokens: &[u32]) -> String {
    let mut bytes = Vec::with_capacity(tokens.len() * 4);
    for token in tokens {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    sha256_hex(&bytes)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CompositeIdentityV2 {
    pub semantic_product_sha256: String,
    pub prefix_checkpoint_sha256: String,
    pub target_checkpoint_sha256: String,
    pub target_engine_sha256: String,
    pub target_sampler_sha256: String,
    pub draft_checkpoint_sha256: String,
    pub draft_engine_sha256: String,
    pub tokenizer_sha256: String,
    pub vocabulary_sha256: String,
    pub target_genesis_root_sha256: String,
    pub draft_genesis_root_sha256: String,
    pub portable_kv_abi: String,
}

impl CompositeIdentityV2 {
    fn validate(&self) -> Result<()> {
        for (name, digest) in [
            ("semantic product", &self.semantic_product_sha256),
            ("prefix checkpoint", &self.prefix_checkpoint_sha256),
            ("target checkpoint", &self.target_checkpoint_sha256),
            ("target engine", &self.target_engine_sha256),
            ("target sampler", &self.target_sampler_sha256),
            ("draft checkpoint", &self.draft_checkpoint_sha256),
            ("draft engine", &self.draft_engine_sha256),
            ("tokenizer", &self.tokenizer_sha256),
            ("vocabulary", &self.vocabulary_sha256),
            ("target genesis", &self.target_genesis_root_sha256),
            ("draft genesis", &self.draft_genesis_root_sha256),
        ] {
            validate_digest(name, digest)?;
        }
        validate_identifier("portable KV ABI", &self.portable_kv_abi)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CouplingPolicyV2 {
    Greedy,
    SparseMaximal { max_support: u32 },
    SharedGumbel,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SamplerStateV2 {
    Stateless { implementation: String },
    Mt19937 { state: Vec<u32>, index: u32 },
}

impl SamplerStateV2 {
    pub fn stateless() -> Self {
        Self::Stateless {
            implementation: STATELESS_SAMPLER_V2.into(),
        }
    }

    pub fn from_mt(snapshot: &Mt19937Snapshot) -> Result<Self> {
        let index = u32::try_from(snapshot.index).map_err(|_| invalid("MT index overflow"))?;
        let state = Self::Mt19937 {
            state: snapshot.state.clone(),
            index,
        };
        state.validate()?;
        Ok(state)
    }

    pub fn to_mt(&self) -> Result<Mt19937> {
        let Self::Mt19937 { state, index } = self else {
            return Err(invalid("sampler state is not MT19937"));
        };
        Mt19937::from_snapshot(&Mt19937Snapshot {
            state: state.clone(),
            index: *index as usize,
        })
        .map_err(|error| invalid(error.to_string()))
    }

    pub fn digest(&self) -> Result<String> {
        Ok(sha256_hex(&canonical(self)?))
    }

    fn validate(&self) -> Result<()> {
        match self {
            Self::Stateless { implementation } if implementation == STATELESS_SAMPLER_V2 => Ok(()),
            Self::Mt19937 { state, index } if state.len() == 624 && *index <= 624 => Ok(()),
            _ => Err(invalid("sampler state geometry or implementation differs")),
        }
    }

    fn validate_for(&self, policy: &CouplingPolicyV2) -> Result<()> {
        self.validate()?;
        match (policy, self) {
            (CouplingPolicyV2::SparseMaximal { .. }, Self::Mt19937 { .. }) => Ok(()),
            (CouplingPolicyV2::Greedy | CouplingPolicyV2::SharedGumbel, Self::Stateless { .. }) => {
                Ok(())
            }
            _ => Err(invalid("sampler state does not match coupling policy")),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReasonV2 {
    Eos,
    /// The caller's output-token limit or exhaustion of the model context.
    /// Both stop at the already-committed boundary before another frontier is
    /// evaluated; the authenticated Close intent distinguishes this from EOS.
    MaxTokens,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RoundIntentV2 {
    /// Evaluate the carried frontier and any supplied draft candidates.
    #[default]
    Verify,
    /// Close at the already-committed parent boundary without evaluating or
    /// publishing the carried frontier. EOS is target-observed during Verify;
    /// a writer may request only cancellation or its authenticated token cap.
    Close { reason: FinishReasonV2 },
}

impl RoundIntentV2 {
    fn is_verify(&self) -> bool {
        matches!(self, Self::Verify)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum FrontierV2 {
    Open { token: u32, output_ordinal: u64 },
    Closed { reason: FinishReasonV2 },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum FragmentKindV2 {
    TargetHidden,
    TargetKvDelta,
    TargetProbabilityEvidence,
    SamplerWitness,
    Snapshot,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FragmentCoverageV2 {
    CommittedInputs,
    FreshTargetRows,
    RoundSingleton,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FragmentRequirementV2 {
    pub component_id: String,
    pub kind: FragmentKindV2,
    pub coverage: FragmentCoverageV2,
    pub payload_abi: String,
    pub bytes_per_logical_row: u64,
    pub required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FragmentDescriptorV2 {
    pub ordinal: u32,
    pub component_id: String,
    pub kind: FragmentKindV2,
    pub logical_start: u64,
    pub logical_count: u64,
    pub payload_abi: String,
    pub bytes_per_logical_row: u64,
    pub byte_len: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SessionCoreV2 {
    pub protocol: String,
    pub session_id: String,
    pub client_incarnation: String,
    pub log_writer_authority_id: String,
    pub authority_lease_id: String,
    pub authority_term: u64,
    pub target_executor_id: String,
    pub target_public_key: [u8; 32],
    pub created_unix_ms: u64,
    pub expires_unix_ms: u64,
    pub hmac_key_id: String,
    pub hmac_key_epoch: u64,
    pub identity: CompositeIdentityV2,
    pub coupling_policy: CouplingPolicyV2,
    pub sampler_config_sha256: String,
    pub vocab_size: u32,
    pub max_context_tokens: u32,
    pub max_drafts: u32,
    pub max_total_fragment_bytes: u64,
    pub initial_evaluated_tokens: Vec<u32>,
    pub initial_output_height: u64,
    pub initial_frontier: FrontierV2,
    pub initial_sampler_state: SamplerStateV2,
    pub fragment_requirements: Vec<FragmentRequirementV2>,
    pub genesis_head_sha256: String,
}

impl SessionCoreV2 {
    fn validate_without_head(&self, expected_key_id: &str, minimum_epoch: u64) -> Result<()> {
        if self.protocol != VERIFIER_SESSION_PROTOCOL_V2 {
            return Err(invalid("wrong session protocol"));
        }
        for (name, value) in [
            ("session id", &self.session_id),
            ("client incarnation", &self.client_incarnation),
            ("log writer", &self.log_writer_authority_id),
            ("authority lease", &self.authority_lease_id),
            ("target executor", &self.target_executor_id),
            ("HMAC key id", &self.hmac_key_id),
        ] {
            validate_identifier(name, value)?;
        }
        if self.hmac_key_id != expected_key_id
            || self.hmac_key_epoch < minimum_epoch
            || self.authority_term == 0
            || self.created_unix_ms >= self.expires_unix_ms
            || self.vocab_size == 0
            || self.max_context_tokens == 0
            || self.max_context_tokens as usize > MAX_CONTEXT_TOKENS_V2
            || self.initial_evaluated_tokens.is_empty()
            || self.initial_evaluated_tokens.len() > self.max_context_tokens as usize
            || self.max_drafts == 0
            || self.max_drafts as usize > MAX_DRAFTS_V2
            || self.max_total_fragment_bytes == 0
            || self.max_total_fragment_bytes > MAX_TOTAL_FRAGMENT_BYTES_V2
            || self.target_public_key == [0; 32]
            || self
                .initial_evaluated_tokens
                .iter()
                .any(|token| *token >= self.vocab_size)
        {
            return Err(invalid("invalid session lifetime or geometry"));
        }
        match (&self.coupling_policy, &self.initial_frontier) {
            (
                CouplingPolicyV2::SparseMaximal { max_support },
                FrontierV2::Open {
                    token,
                    output_ordinal,
                },
            ) if *max_support > 0
                && *max_support as usize <= MAX_SPARSE_SUPPORT_V2
                && *token < self.vocab_size
                && *output_ordinal == self.initial_output_height => {}
            (
                CouplingPolicyV2::Greedy | CouplingPolicyV2::SharedGumbel,
                FrontierV2::Open {
                    token,
                    output_ordinal,
                },
            ) if *token < self.vocab_size && *output_ordinal == self.initial_output_height => {}
            _ => return Err(invalid("invalid initial policy or frontier")),
        }
        self.initial_sampler_state
            .validate_for(&self.coupling_policy)?;
        validate_digest("sampler config", &self.sampler_config_sha256)?;
        self.identity.validate()?;
        validate_requirements(&self.fragment_requirements)
    }

    pub fn genesis_digest(&self) -> Result<String> {
        let mut normalized = self.clone();
        normalized.genesis_head_sha256 = "0".repeat(64);
        Ok(sha256_hex(&canonical(&normalized)?))
    }

    fn validate(&self, expected_key_id: &str, minimum_epoch: u64) -> Result<()> {
        self.validate_without_head(expected_key_id, minimum_epoch)?;
        validate_digest("genesis head", &self.genesis_head_sha256)?;
        if self.genesis_head_sha256 != self.genesis_digest()? {
            return Err(invalid(
                "session genesis head is not derived from genesis state",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AuthenticatedSessionV2 {
    pub core: SessionCoreV2,
    pub hmac_sha256: String,
}

impl AuthenticatedSessionV2 {
    pub fn sign(mut core: SessionCoreV2, key: &MacKey) -> Result<Self> {
        core.genesis_head_sha256 = "0".repeat(64);
        core.validate_without_head(&core.hmac_key_id, core.hmac_key_epoch)?;
        core.genesis_head_sha256 = core.genesis_digest()?;
        let hmac_sha256 = key
            .tag_domain_hex(VERIFIER_SESSION_MAC_DOMAIN_V2, &canonical(&core)?)
            .map_err(|_| VerifierV2Error::Authentication)?;
        Ok(Self { core, hmac_sha256 })
    }

    pub fn verify_historical(
        &self,
        key: &MacKey,
        expected_key_id: &str,
        minimum_epoch: u64,
    ) -> Result<()> {
        self.core.validate(expected_key_id, minimum_epoch)?;
        validate_digest("session HMAC", &self.hmac_sha256)?;
        key.verify_domain_hex(
            VERIFIER_SESSION_MAC_DOMAIN_V2,
            &canonical(&self.core)?,
            &self.hmac_sha256,
        )
        .map_err(|_| VerifierV2Error::Authentication)
    }

    pub fn verify_live(
        &self,
        key: &MacKey,
        now_unix_ms: u64,
        expected_key_id: &str,
        minimum_epoch: u64,
    ) -> Result<()> {
        self.verify_historical(key, expected_key_id, minimum_epoch)?;
        if now_unix_ms > self.core.expires_unix_ms
            || self.core.created_unix_ms > now_unix_ms.saturating_add(MAX_CLOCK_SKEW_MS_V2)
        {
            return Err(invalid("session is expired or created in the future"));
        }
        Ok(())
    }

    pub fn record_digest(&self) -> Result<String> {
        Ok(sha256_hex(&canonical(self)?))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SparseWeightEntryV2 {
    pub token: u32,
    pub weight_f32_bits: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SparseWeightRowV2 {
    pub vocab_size: u32,
    pub entries: Vec<SparseWeightEntryV2>,
}

impl SparseWeightRowV2 {
    fn validate(&self, vocab_size: u32, max_support: usize) -> Result<f64> {
        if self.vocab_size != vocab_size
            || self.entries.is_empty()
            || self.entries.len() > max_support
        {
            return Err(invalid("sparse weight row geometry differs"));
        }
        let mut seen = BTreeSet::new();
        let mut total = 0.0f64;
        for entry in &self.entries {
            let weight = f32::from_bits(entry.weight_f32_bits);
            if entry.token >= vocab_size
                || !seen.insert(entry.token)
                || !weight.is_finite()
                || weight <= 0.0
            {
                return Err(invalid("sparse weight row has an invalid entry"));
            }
            total += weight as f64;
        }
        if !total.is_finite() || total <= 0.0 {
            return Err(invalid("sparse weight row has invalid total"));
        }
        Ok(total)
    }

    fn sample(&self, rng: &mut Mt19937, vocab_size: u32, max_support: usize) -> Result<u32> {
        let total = self.validate(vocab_size, max_support)?;
        let target = rng.uniform_f64() * total;
        let mut cumulative = 0.0f64;
        for entry in &self.entries {
            cumulative += f32::from_bits(entry.weight_f32_bits) as f64;
            if cumulative >= target {
                return Ok(entry.token);
            }
        }
        Ok(self.entries.last().expect("validated nonempty row").token)
    }

    pub fn probabilities(
        &self,
        vocab_size: u32,
        max_support: usize,
    ) -> Result<SparseProbabilityRow> {
        let total = self.validate(vocab_size, max_support)?;
        let row = SparseProbabilityRow {
            vocab_size,
            entries: self
                .entries
                .iter()
                .map(|entry| SparseProbabilityEntry {
                    token: entry.token,
                    probability: (f32::from_bits(entry.weight_f32_bits) as f64 / total) as f32,
                })
                .collect(),
        };
        row.validate_bounded(max_support)
            .map_err(|error| invalid(error.to_string()))?;
        Ok(row)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum DraftEvidenceV2 {
    None,
    SparseMaximal { q_rows: Vec<SparseWeightRowV2> },
    SharedGumbel { witness_sha256: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RoundCoreV2 {
    pub protocol: String,
    /// Absent in historical round-v2 JSON, where Verify was the only intent.
    /// Omitting the default preserves those records' canonical bytes and HMAC.
    #[serde(default, skip_serializing_if = "RoundIntentV2::is_verify")]
    pub intent: RoundIntentV2,
    pub session_id: String,
    pub session_genesis_sha256: String,
    pub authority_term: u64,
    pub request_id: String,
    pub created_unix_ms: u64,
    pub expires_unix_ms: u64,
    pub hmac_key_id: String,
    pub hmac_key_epoch: u64,
    pub base_output_height: u64,
    pub base_head_sha256: String,
    pub base_evaluated_tokens: Vec<u32>,
    pub base_tokens_sha256: String,
    pub frontier_in: FrontierV2,
    pub draft_tokens: Vec<u32>,
    pub draft_evidence: DraftEvidenceV2,
    pub base_sampler_state: SamplerStateV2,
    pub post_draft_sampler_state: SamplerStateV2,
}

impl RoundCoreV2 {
    fn mac_domain(&self) -> Result<&'static [u8]> {
        match &*self.protocol {
            VERIFIER_ROUND_PROTOCOL_V2 => Ok(VERIFIER_ROUND_MAC_DOMAIN_V2),
            VERIFIER_ROUND_PROTOCOL_V3 => Ok(VERIFIER_ROUND_MAC_DOMAIN_V3),
            _ => Err(invalid("wrong round protocol")),
        }
    }

    fn validate_historical(&self, session: &SessionCoreV2) -> Result<()> {
        let compatible_protocol = matches!(
            (&*self.protocol, &self.intent),
            (VERIFIER_ROUND_PROTOCOL_V2, RoundIntentV2::Verify) | (VERIFIER_ROUND_PROTOCOL_V3, _)
        );
        if !compatible_protocol
            || self.session_id != session.session_id
            || self.session_genesis_sha256 != session.genesis_head_sha256
            || self.authority_term != session.authority_term
            || self.hmac_key_id != session.hmac_key_id
            || self.hmac_key_epoch != session.hmac_key_epoch
            || self.created_unix_ms < session.created_unix_ms
            || self.created_unix_ms >= self.expires_unix_ms
            || self.expires_unix_ms > session.expires_unix_ms
        {
            return Err(invalid("round addresses the wrong session or lifetime"));
        }
        validate_identifier("request id", &self.request_id)?;
        validate_digest("base head", &self.base_head_sha256)?;
        validate_digest("base transcript", &self.base_tokens_sha256)?;
        if token_digest(&self.base_evaluated_tokens) != self.base_tokens_sha256
            || self.base_evaluated_tokens.len() > session.max_context_tokens as usize
            || self
                .base_evaluated_tokens
                .iter()
                .chain(&self.draft_tokens)
                .any(|token| *token >= session.vocab_size)
            || self.draft_tokens.len() > session.max_drafts as usize
        {
            return Err(invalid("round transcript or candidate geometry differs"));
        }
        let generated =
            self.base_output_height
                .checked_sub(session.initial_output_height)
                .ok_or_else(|| invalid("round output precedes genesis"))? as usize;
        let context_inputs = match &self.intent {
            RoundIntentV2::Verify => 1 + self.draft_tokens.len(),
            RoundIntentV2::Close { .. } => 0,
        };
        if self.base_evaluated_tokens.len()
            != session
                .initial_evaluated_tokens
                .len()
                .saturating_add(generated)
            || !self
                .base_evaluated_tokens
                .starts_with(&session.initial_evaluated_tokens)
            || self
                .base_evaluated_tokens
                .len()
                .checked_add(context_inputs)
                .is_none_or(|end| end > session.max_context_tokens as usize)
        {
            return Err(invalid("round transcript length or context limit differs"));
        }
        let FrontierV2::Open {
            token,
            output_ordinal,
        } = self.frontier_in
        else {
            return Err(VerifierV2Error::Terminal);
        };
        if token >= session.vocab_size || output_ordinal != self.base_output_height {
            return Err(invalid("round carried frontier differs"));
        }
        self.base_sampler_state
            .validate_for(&session.coupling_policy)?;
        self.post_draft_sampler_state
            .validate_for(&session.coupling_policy)?;
        if let RoundIntentV2::Close { reason } = &self.intent {
            if matches!(reason, FinishReasonV2::Eos)
                || !self.draft_tokens.is_empty()
                || !matches!(self.draft_evidence, DraftEvidenceV2::None)
                || self.base_sampler_state != self.post_draft_sampler_state
            {
                return Err(invalid("close intent or sampler boundary differs"));
            }
            return Ok(());
        }
        match (&session.coupling_policy, &self.draft_evidence) {
            (CouplingPolicyV2::Greedy, DraftEvidenceV2::None) => {
                if self.base_sampler_state != self.post_draft_sampler_state {
                    return Err(invalid("greedy round changed stateless sampler state"));
                }
            }
            (CouplingPolicyV2::SharedGumbel, DraftEvidenceV2::SharedGumbel { witness_sha256 }) => {
                validate_digest("shared-Gumbel witness", witness_sha256)?;
                if self.base_sampler_state != self.post_draft_sampler_state {
                    return Err(invalid(
                        "shared-Gumbel round changed stateless sampler state",
                    ));
                }
            }
            (
                CouplingPolicyV2::SparseMaximal { max_support },
                DraftEvidenceV2::SparseMaximal { q_rows },
            ) => {
                if q_rows.len() != self.draft_tokens.len() {
                    return Err(invalid("sparse q row count differs from draft count"));
                }
                let mut rng = self.base_sampler_state.to_mt()?;
                for (row, token) in q_rows.iter().zip(&self.draft_tokens) {
                    if row.sample(&mut rng, session.vocab_size, *max_support as usize)? != *token {
                        return Err(invalid("draft token was not sampled from its q row"));
                    }
                }
                if SamplerStateV2::from_mt(&rng.snapshot())? != self.post_draft_sampler_state {
                    return Err(invalid("post-draft MT snapshot does not replay"));
                }
            }
            _ => {
                return Err(invalid(
                    "draft evidence does not match fixed session policy",
                ))
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AuthenticatedRoundV2 {
    pub core: RoundCoreV2,
    pub hmac_sha256: String,
}

impl AuthenticatedRoundV2 {
    pub fn sign(core: RoundCoreV2, session: &SessionCoreV2, key: &MacKey) -> Result<Self> {
        core.validate_historical(session)?;
        let domain = core.mac_domain()?;
        let hmac_sha256 = key
            .tag_domain_hex(domain, &canonical(&core)?)
            .map_err(|_| VerifierV2Error::Authentication)?;
        Ok(Self { core, hmac_sha256 })
    }

    pub fn verify_historical(&self, session: &SessionCoreV2, key: &MacKey) -> Result<()> {
        self.core.validate_historical(session)?;
        validate_digest("round HMAC", &self.hmac_sha256)?;
        key.verify_domain_hex(
            self.core.mac_domain()?,
            &canonical(&self.core)?,
            &self.hmac_sha256,
        )
        .map_err(|_| VerifierV2Error::Authentication)
    }

    pub fn verify_live(
        &self,
        session: &SessionCoreV2,
        key: &MacKey,
        now_unix_ms: u64,
    ) -> Result<()> {
        self.verify_historical(session, key)?;
        if self.core.protocol != VERIFIER_ROUND_PROTOCOL_V3 {
            return Err(invalid(
                "round-v2 is historical-only; new live work requires round-v3",
            ));
        }
        if now_unix_ms > self.core.expires_unix_ms
            || self.core.created_unix_ms > now_unix_ms.saturating_add(MAX_CLOCK_SKEW_MS_V2)
        {
            return Err(invalid("round is expired or created in the future"));
        }
        Ok(())
    }

    pub fn intent_sha256(&self) -> Result<String> {
        Ok(sha256_hex(&canonical(&self.core)?))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ResultDecisionV2 {
    Open {
        accepted_drafts: u32,
        frontier_out: u32,
    },
    Closed {
        /// False closes at the carried-frontier boundary without evaluating
        /// or emitting it (cancel/max-token before a new round). In that case
        /// `accepted_drafts` must be zero.
        commit_frontier: bool,
        accepted_drafts: u32,
        reason: FinishReasonV2,
    },
}

impl ResultDecisionV2 {
    fn accepted_drafts(&self) -> u32 {
        match self {
            Self::Open {
                accepted_drafts, ..
            }
            | Self::Closed {
                accepted_drafts, ..
            } => *accepted_drafts,
        }
    }

    fn committed_input_count(&self) -> Result<usize> {
        match self {
            Self::Open {
                accepted_drafts, ..
            } => Ok(*accepted_drafts as usize + 1),
            Self::Closed {
                commit_frontier: true,
                accepted_drafts,
                ..
            } => Ok(*accepted_drafts as usize + 1),
            Self::Closed {
                commit_frontier: false,
                accepted_drafts: 0,
                ..
            } => Ok(0),
            Self::Closed {
                commit_frontier: false,
                ..
            } => Err(invalid(
                "terminal decision cannot accept drafts without its frontier",
            )),
        }
    }

    fn validate_for_intent(
        &self,
        protocol: &str,
        intent: &RoundIntentV2,
        base_sampler_state: &SamplerStateV2,
        post_verify_sampler_state: &SamplerStateV2,
    ) -> Result<()> {
        // Round-v2 had no authenticated intent and allowed every structurally
        // valid terminal decision. Preserve historical journal verification;
        // `verify_live` refuses v2 for all newly admitted work.
        if protocol == VERIFIER_ROUND_PROTOCOL_V2 && matches!(intent, RoundIntentV2::Verify) {
            return Ok(());
        }
        match (intent, self) {
            (RoundIntentV2::Verify, Self::Open { .. }) => Ok(()),
            (
                RoundIntentV2::Verify,
                Self::Closed {
                    commit_frontier: true,
                    reason: FinishReasonV2::Eos,
                    ..
                },
            ) => Ok(()),
            (
                RoundIntentV2::Close { reason: requested },
                Self::Closed {
                    commit_frontier: false,
                    accepted_drafts: 0,
                    reason: returned,
                },
            ) if requested == returned && base_sampler_state == post_verify_sampler_state => Ok(()),
            (RoundIntentV2::Close { .. }, _) => {
                Err(invalid("result does not honor authenticated close intent"))
            }
            (RoundIntentV2::Verify, Self::Closed { .. }) => {
                Err(invalid("verify may close only on target-observed EOS"))
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ResultCoreV2 {
    pub protocol: String,
    pub session_id: String,
    pub session_genesis_sha256: String,
    pub authority_term: u64,
    pub target_executor_id: String,
    pub request_id: String,
    pub request_intent_sha256: String,
    pub base_output_height: u64,
    pub base_head_sha256: String,
    pub decision: ResultDecisionV2,
    pub committed_tokens: Vec<u32>,
    pub new_output_height: u64,
    pub new_evaluated_tokens_sha256: String,
    pub new_frontier: FrontierV2,
    pub post_verify_sampler_state: SamplerStateV2,
    pub fragments_sha256: String,
    pub fragments: Vec<FragmentDescriptorV2>,
    pub new_head_sha256: String,
    pub completed_unix_ms: u64,
}

pub struct TargetSigningKeyV2(Ed25519KeyPair);

impl TargetSigningKeyV2 {
    pub fn from_seed(seed: [u8; 32]) -> Result<Self> {
        Ed25519KeyPair::from_seed_unchecked(&seed)
            .map(Self)
            .map_err(|_| invalid("invalid Ed25519 target seed"))
    }

    pub fn public_key(&self) -> [u8; 32] {
        self.0
            .public_key()
            .as_ref()
            .try_into()
            .expect("Ed25519 public keys are 32 bytes")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AuthenticatedResultV2 {
    pub core: ResultCoreV2,
    pub target_signature_hex: String,
}

impl AuthenticatedResultV2 {
    pub fn sign(core: ResultCoreV2, signer: &TargetSigningKeyV2) -> Result<Self> {
        let signature = signer.0.sign(&canonical(&core)?);
        Ok(Self {
            core,
            target_signature_hex: hex_encode(signature.as_ref()),
        })
    }

    pub fn verify_against(
        &self,
        request: &AuthenticatedRoundV2,
        session: &SessionCoreV2,
    ) -> Result<()> {
        let signature = hex_decode(&self.target_signature_hex, 64)?;
        UnparsedPublicKey::new(&ED25519, session.target_public_key)
            .verify(&canonical(&self.core)?, &signature)
            .map_err(|_| VerifierV2Error::Authentication)?;
        let core = &self.core;
        core.decision.validate_for_intent(
            &request.core.protocol,
            &request.core.intent,
            &request.core.base_sampler_state,
            &core.post_verify_sampler_state,
        )?;
        let accepted = core.decision.accepted_drafts() as usize;
        let FrontierV2::Open {
            token: frontier_in,
            output_ordinal,
        } = request.core.frontier_in
        else {
            return Err(VerifierV2Error::Terminal);
        };
        if accepted > request.core.draft_tokens.len() {
            return Err(invalid("accepted draft count exceeds request"));
        }
        let committed_input_count = core.decision.committed_input_count()?;
        let mut expected_committed = Vec::with_capacity(committed_input_count);
        if committed_input_count != 0 {
            expected_committed.push(frontier_in);
            expected_committed.extend_from_slice(&request.core.draft_tokens[..accepted]);
        }
        let expected_height = request
            .core
            .base_output_height
            .checked_add(expected_committed.len() as u64)
            .ok_or_else(|| invalid("result height overflow"))?;
        if core.protocol != VERIFIER_RESULT_PROTOCOL_V2
            || core.session_id != session.session_id
            || core.session_genesis_sha256 != session.genesis_head_sha256
            || core.authority_term != session.authority_term
            || core.target_executor_id != session.target_executor_id
            || core.request_id != request.core.request_id
            || core.request_intent_sha256 != request.intent_sha256()?
            || core.base_output_height != request.core.base_output_height
            || core.base_head_sha256 != request.core.base_head_sha256
            || output_ordinal != core.base_output_height
            || core.committed_tokens != expected_committed
            || core.new_output_height != expected_height
            || core.completed_unix_ms < request.core.created_unix_ms
            || core.completed_unix_ms > request.core.expires_unix_ms
        {
            return Err(invalid("result does not match its authenticated request"));
        }
        match (&core.decision, &core.new_frontier) {
            (
                ResultDecisionV2::Open { frontier_out, .. },
                FrontierV2::Open {
                    token,
                    output_ordinal,
                },
            ) if frontier_out == token
                && *token < session.vocab_size
                && *output_ordinal == expected_height => {}
            (
                ResultDecisionV2::Closed { reason, .. },
                FrontierV2::Closed {
                    reason: frontier_reason,
                },
            ) if reason == frontier_reason => {}
            _ => return Err(invalid("result terminal/frontier transition differs")),
        }
        core.post_verify_sampler_state
            .validate_for(&session.coupling_policy)?;
        validate_digest("new transcript", &core.new_evaluated_tokens_sha256)?;
        let mut transcript = request.core.base_evaluated_tokens.clone();
        transcript.extend_from_slice(&core.committed_tokens);
        if token_digest(&transcript) != core.new_evaluated_tokens_sha256 {
            return Err(invalid("result transcript commitment differs"));
        }
        validate_descriptors(&core.fragments, session.max_total_fragment_bytes)?;
        validate_closure_geometry(core, request, session)?;
        if fragment_set_digest(&core.fragments)? != core.fragments_sha256 {
            return Err(invalid("result fragment commitment differs"));
        }
        let expected_head = transition_head(core)?;
        if core.new_head_sha256 != expected_head {
            return Err(invalid("result head transition differs"));
        }
        Ok(())
    }

    pub fn record_digest(&self) -> Result<String> {
        Ok(sha256_hex(&canonical(self)?))
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn hex_decode(value: &str, expected_bytes: usize) -> Result<Vec<u8>> {
    if value.len() != expected_bytes * 2
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(VerifierV2Error::Authentication);
    }
    (0..expected_bytes)
        .map(|index| {
            u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
                .map_err(|_| VerifierV2Error::Authentication)
        })
        .collect()
}

fn validate_requirements(requirements: &[FragmentRequirementV2]) -> Result<()> {
    if requirements.is_empty() || requirements.len() > 64 {
        return Err(invalid("fragment requirement count differs"));
    }
    let mut components = BTreeSet::new();
    for requirement in requirements {
        validate_identifier("fragment component", &requirement.component_id)?;
        validate_identifier("fragment payload ABI", &requirement.payload_abi)?;
        if !components.insert((requirement.component_id.clone(), requirement.kind))
            || requirement.bytes_per_logical_row == 0
            || requirement.bytes_per_logical_row > MAX_FRAGMENT_BYTES_V2
        {
            return Err(invalid("fragment requirement is duplicate or invalid"));
        }
    }
    Ok(())
}

fn validate_descriptors(descriptors: &[FragmentDescriptorV2], session_limit: u64) -> Result<()> {
    if descriptors.len() > MAX_FRAGMENTS_V2 {
        return Err(invalid("too many result fragments"));
    }
    let mut total = 0u64;
    for (index, descriptor) in descriptors.iter().enumerate() {
        validate_identifier("fragment component", &descriptor.component_id)?;
        validate_identifier("fragment payload ABI", &descriptor.payload_abi)?;
        validate_digest("fragment", &descriptor.sha256)?;
        if descriptor.ordinal as usize != index
            || descriptor.logical_count == 0
            || descriptor.bytes_per_logical_row == 0
            || descriptor.byte_len == 0
            || descriptor.byte_len > MAX_FRAGMENT_BYTES_V2
            || descriptor
                .logical_count
                .checked_mul(descriptor.bytes_per_logical_row)
                != Some(descriptor.byte_len)
        {
            return Err(invalid("fragment descriptor geometry differs"));
        }
        total = total
            .checked_add(descriptor.byte_len)
            .ok_or_else(|| invalid("fragment bytes overflow"))?;
    }
    if total > session_limit || total > MAX_TOTAL_FRAGMENT_BYTES_V2 {
        return Err(invalid("fragment set exceeds byte limit"));
    }
    Ok(())
}

fn validate_closure_geometry(
    result: &ResultCoreV2,
    request: &AuthenticatedRoundV2,
    session: &SessionCoreV2,
) -> Result<()> {
    for requirement in &session.fragment_requirements {
        let rows = result
            .fragments
            .iter()
            .filter(|descriptor| {
                descriptor.component_id == requirement.component_id
                    && descriptor.kind == requirement.kind
            })
            .collect::<Vec<_>>();
        let (start, count) = match (&request.core.intent, requirement.coverage) {
            (RoundIntentV2::Close { .. }, _) => (request.core.base_output_height, 0),
            (_, FragmentCoverageV2::CommittedInputs) => (
                request.core.base_output_height,
                result.committed_tokens.len() as u64,
            ),
            (_, FragmentCoverageV2::FreshTargetRows) => (
                request.core.base_output_height + 1,
                request.core.draft_tokens.len() as u64 + 1,
            ),
            (_, FragmentCoverageV2::RoundSingleton) => (request.core.base_output_height, 1),
        };
        if rows.is_empty() {
            if count == 0 {
                continue;
            }
            if requirement.required {
                return Err(invalid("required result component is absent"));
            }
            continue;
        }
        if rows.iter().any(|descriptor| {
            descriptor.payload_abi != requirement.payload_abi
                || descriptor.bytes_per_logical_row != requirement.bytes_per_logical_row
        }) {
            return Err(invalid("fragment payload ABI differs from session"));
        }
        let mut cursor = start;
        for descriptor in rows {
            if descriptor.logical_start != cursor {
                return Err(invalid("fragment logical coverage has a gap or overlap"));
            }
            cursor = cursor
                .checked_add(descriptor.logical_count)
                .ok_or_else(|| invalid("fragment logical range overflow"))?;
        }
        if cursor != start + count {
            return Err(invalid("fragment logical coverage is incomplete"));
        }
    }
    if result.fragments.iter().any(|descriptor| {
        !session.fragment_requirements.iter().any(|requirement| {
            requirement.component_id == descriptor.component_id
                && requirement.kind == descriptor.kind
        })
    }) {
        return Err(invalid("result has an undeclared component"));
    }
    Ok(())
}

fn fragment_set_digest(descriptors: &[FragmentDescriptorV2]) -> Result<String> {
    Ok(sha256_hex(&canonical(&descriptors)?))
}

#[derive(Serialize)]
struct HeadTransitionV2<'a> {
    protocol: &'static str,
    session_id: &'a str,
    session_genesis_sha256: &'a str,
    authority_term: u64,
    base_head_sha256: &'a str,
    base_output_height: u64,
    committed_tokens: &'a [u32],
    new_output_height: u64,
    new_evaluated_tokens_sha256: &'a str,
    new_frontier: &'a FrontierV2,
    post_verify_sampler_state: &'a SamplerStateV2,
    fragments_sha256: &'a str,
    request_intent_sha256: &'a str,
}

fn transition_head(core: &ResultCoreV2) -> Result<String> {
    Ok(sha256_hex(&canonical(&HeadTransitionV2 {
        protocol: VERIFIER_RESULT_PROTOCOL_V2,
        session_id: &core.session_id,
        session_genesis_sha256: &core.session_genesis_sha256,
        authority_term: core.authority_term,
        base_head_sha256: &core.base_head_sha256,
        base_output_height: core.base_output_height,
        committed_tokens: &core.committed_tokens,
        new_output_height: core.new_output_height,
        new_evaluated_tokens_sha256: &core.new_evaluated_tokens_sha256,
        new_frontier: &core.new_frontier,
        post_verify_sampler_state: &core.post_verify_sampler_state,
        fragments_sha256: &core.fragments_sha256,
        request_intent_sha256: &core.request_intent_sha256,
    })?))
}

pub fn build_result(
    request: &AuthenticatedRoundV2,
    session: &SessionCoreV2,
    decision: ResultDecisionV2,
    post_verify_sampler_state: SamplerStateV2,
    fragments: Vec<FragmentDescriptorV2>,
    completed_unix_ms: u64,
    signer: &TargetSigningKeyV2,
) -> Result<AuthenticatedResultV2> {
    decision.validate_for_intent(
        &request.core.protocol,
        &request.core.intent,
        &request.core.base_sampler_state,
        &post_verify_sampler_state,
    )?;
    let accepted = decision.accepted_drafts() as usize;
    let FrontierV2::Open {
        token: frontier_in, ..
    } = request.core.frontier_in
    else {
        return Err(VerifierV2Error::Terminal);
    };
    if accepted > request.core.draft_tokens.len()
        || completed_unix_ms > request.core.expires_unix_ms
    {
        return Err(invalid("target decision or completion time differs"));
    }
    post_verify_sampler_state.validate_for(&session.coupling_policy)?;
    let committed_input_count = decision.committed_input_count()?;
    let mut committed_tokens = Vec::with_capacity(committed_input_count);
    if committed_input_count != 0 {
        committed_tokens.push(frontier_in);
        committed_tokens.extend_from_slice(&request.core.draft_tokens[..accepted]);
    }
    let new_output_height = request
        .core
        .base_output_height
        .checked_add(committed_tokens.len() as u64)
        .ok_or_else(|| invalid("result height overflow"))?;
    let new_frontier = match &decision {
        ResultDecisionV2::Open { frontier_out, .. } => FrontierV2::Open {
            token: *frontier_out,
            output_ordinal: new_output_height,
        },
        ResultDecisionV2::Closed { reason, .. } => FrontierV2::Closed {
            reason: reason.clone(),
        },
    };
    let mut transcript = request.core.base_evaluated_tokens.clone();
    transcript.extend_from_slice(&committed_tokens);
    let fragments_sha256 = fragment_set_digest(&fragments)?;
    let mut core = ResultCoreV2 {
        protocol: VERIFIER_RESULT_PROTOCOL_V2.into(),
        session_id: session.session_id.clone(),
        session_genesis_sha256: session.genesis_head_sha256.clone(),
        authority_term: session.authority_term,
        target_executor_id: session.target_executor_id.clone(),
        request_id: request.core.request_id.clone(),
        request_intent_sha256: request.intent_sha256()?,
        base_output_height: request.core.base_output_height,
        base_head_sha256: request.core.base_head_sha256.clone(),
        decision,
        committed_tokens,
        new_output_height,
        new_evaluated_tokens_sha256: token_digest(&transcript),
        new_frontier,
        post_verify_sampler_state,
        fragments_sha256,
        fragments,
        new_head_sha256: "0".repeat(64),
        completed_unix_ms,
    };
    core.new_head_sha256 = transition_head(&core)?;
    let result = AuthenticatedResultV2::sign(core, signer)?;
    result.verify_against(request, session)?;
    Ok(result)
}

#[derive(Debug, Clone)]
pub struct VerifiedFragmentV2 {
    pub descriptor: FragmentDescriptorV2,
    pub payload: Vec<u8>,
}

#[derive(Debug)]
pub struct VerifiedClosureV2 {
    result_sha256: String,
    fragments: Vec<VerifiedFragmentV2>,
}

impl VerifiedClosureV2 {
    pub fn fragments(&self) -> &[VerifiedFragmentV2] {
        &self.fragments
    }

    fn verify_result(&self, result: &AuthenticatedResultV2) -> Result<()> {
        if self.result_sha256 != result.record_digest()?
            || self.fragments.len() != result.core.fragments.len()
            || self
                .fragments
                .iter()
                .zip(&result.core.fragments)
                .any(|(fragment, descriptor)| fragment.descriptor != *descriptor)
        {
            return Err(invalid("verified closure belongs to a different result"));
        }
        Ok(())
    }
}

pub struct FragmentAssemblerV2 {
    result_sha256: String,
    descriptors: Vec<FragmentDescriptorV2>,
    payloads: Vec<Option<Vec<u8>>>,
}

impl FragmentAssemblerV2 {
    pub fn new(
        result: &AuthenticatedResultV2,
        request: &AuthenticatedRoundV2,
        session: &SessionCoreV2,
    ) -> Result<Self> {
        result.verify_against(request, session)?;
        Ok(Self {
            result_sha256: result.record_digest()?,
            descriptors: result.core.fragments.clone(),
            payloads: vec![None; result.core.fragments.len()],
        })
    }

    pub fn insert(&mut self, ordinal: u32, payload: Vec<u8>) -> Result<bool> {
        let index = ordinal as usize;
        let descriptor = self
            .descriptors
            .get(index)
            .ok_or_else(|| invalid("fragment ordinal is out of range"))?;
        if payload.len() as u64 != descriptor.byte_len || sha256_hex(&payload) != descriptor.sha256
        {
            return Err(invalid("fragment payload digest or length differs"));
        }
        if let Some(existing) = &self.payloads[index] {
            if existing != &payload {
                return Err(invalid("duplicate fragment has different bytes"));
            }
            return Ok(false);
        }
        self.payloads[index] = Some(payload);
        Ok(true)
    }

    pub fn missing_ordinals(&self) -> Vec<u32> {
        self.payloads
            .iter()
            .enumerate()
            .filter_map(|(index, payload)| payload.is_none().then_some(index as u32))
            .collect()
    }

    pub fn finish(self) -> Result<VerifiedClosureV2> {
        if self.payloads.iter().any(Option::is_none) {
            return Err(VerifierV2Error::Incomplete);
        }
        Ok(VerifiedClosureV2 {
            result_sha256: self.result_sha256,
            fragments: self
                .descriptors
                .into_iter()
                .zip(self.payloads)
                .map(|(descriptor, payload)| VerifiedFragmentV2 {
                    descriptor,
                    payload: payload.expect("closure checked complete"),
                })
                .collect(),
        })
    }
}

/// External registry/fencing service consulted at every admission and commit.
pub trait AuthorityFenceV2 {
    fn permits(
        &self,
        session_id: &str,
        log_writer_authority_id: &str,
        authority_lease_id: &str,
        authority_term: u64,
        target_executor_id: &str,
        now_unix_ms: u64,
    ) -> bool;
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct RendererStageReceiptV2 {
    pub result_sha256: String,
    pub render_root_sha256: String,
    pub stage_token: String,
    pub staged_unix_ms: u64,
}

pub trait VerifierRendererV2 {
    /// Persist an invisible render keyed by the result digest. Implementations
    /// must be idempotent because a crash can occur after staging but before
    /// the journal records the returned receipt.
    fn stage(
        &mut self,
        result: &AuthenticatedResultV2,
        closure: &VerifiedClosureV2,
        now_unix_ms: u64,
    ) -> std::result::Result<RendererStageReceiptV2, String>;

    /// Make a staged render visible. Implementations must be idempotent: the
    /// journal deliberately writes its commit WAL before activation, so a
    /// crash after activation but before the active receipt replays this call.
    fn activate(&mut self, receipt: &RendererStageReceiptV2) -> std::result::Result<(), String>;
}

// The durable journal follows below. Keeping filesystem mechanics in this
// module ensures the safety claims are executable rather than comments.

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct JournalEnvelopeV2 {
    protocol: String,
    kind: String,
    session_record_sha256: String,
    payload: serde_json::Value,
    hmac_sha256: String,
}

#[derive(Serialize)]
struct JournalUnsignedV2<'a> {
    protocol: &'static str,
    kind: &'a str,
    session_record_sha256: &'a str,
    payload: &'a serde_json::Value,
}

fn journal_envelope<T: Serialize>(
    kind: &str,
    session_record_sha256: &str,
    payload: &T,
    key: &MacKey,
) -> Result<JournalEnvelopeV2> {
    validate_identifier("journal kind", kind)?;
    validate_digest("journal session", session_record_sha256)?;
    let payload = serde_json::to_value(payload).map_err(|error| invalid(error.to_string()))?;
    let unsigned = JournalUnsignedV2 {
        protocol: VERIFIER_JOURNAL_PROTOCOL_V2,
        kind,
        session_record_sha256,
        payload: &payload,
    };
    let hmac_sha256 = key
        .tag_domain_hex(VERIFIER_JOURNAL_MAC_DOMAIN_V2, &canonical(&unsigned)?)
        .map_err(|_| VerifierV2Error::Authentication)?;
    Ok(JournalEnvelopeV2 {
        protocol: VERIFIER_JOURNAL_PROTOCOL_V2.into(),
        kind: kind.into(),
        session_record_sha256: session_record_sha256.into(),
        payload,
        hmac_sha256,
    })
}

const VERIFIER_JOURNAL_PROTOCOL_V2: &str = "muser-verifier-journal-v2";

fn decode_journal<T: for<'de> Deserialize<'de>>(
    envelope: &JournalEnvelopeV2,
    expected_kind: &str,
    session_record_sha256: &str,
    key: &MacKey,
) -> Result<T> {
    if envelope.protocol != VERIFIER_JOURNAL_PROTOCOL_V2
        || envelope.kind != expected_kind
        || envelope.session_record_sha256 != session_record_sha256
    {
        return Err(invalid("journal envelope identity differs"));
    }
    validate_digest("journal HMAC", &envelope.hmac_sha256)?;
    let unsigned = JournalUnsignedV2 {
        protocol: VERIFIER_JOURNAL_PROTOCOL_V2,
        kind: &envelope.kind,
        session_record_sha256: &envelope.session_record_sha256,
        payload: &envelope.payload,
    };
    key.verify_domain_hex(
        VERIFIER_JOURNAL_MAC_DOMAIN_V2,
        &canonical(&unsigned)?,
        &envelope.hmac_sha256,
    )
    .map_err(|_| VerifierV2Error::Authentication)?;
    serde_json::from_value(envelope.payload.clone()).map_err(|error| invalid(error.to_string()))
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ReservationRecordV2 {
    request: AuthenticatedRoundV2,
    reserved_unix_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct PreparedRecordV2 {
    request_intent_sha256: String,
    result: AuthenticatedResultV2,
    prepared_unix_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct StagedRecordV2 {
    request_id: String,
    request_intent_sha256: String,
    result_sha256: String,
    renderer: RendererStageReceiptV2,
    staged_unix_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct CommitRecordV2 {
    request: AuthenticatedRoundV2,
    result: AuthenticatedResultV2,
    renderer: RendererStageReceiptV2,
    committed_unix_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ActiveRecordV2 {
    request_id: String,
    result_sha256: String,
    new_head_sha256: String,
    activated_unix_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct AckRecordV2 {
    request_id: String,
    result_sha256: String,
    applied_head_sha256: String,
    acknowledged_unix_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct TombstoneRecordV2 {
    request_id: String,
    result_sha256: String,
    retired_unix_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReserveOutcomeV2 {
    Reserved,
    Prepared(Box<AuthenticatedResultV2>),
    Replay(Box<AuthenticatedResultV2>),
    Stale {
        output_height: u64,
        head_sha256: String,
        frontier: FrontierV2,
        sampler_state: SamplerStateV2,
        transcript_sha256: String,
    },
}

/// The one unfinished child of the journal's current head, reconstructed from
/// durable records after restart. Callers use this instead of attempting to
/// recreate request IDs, timestamps, draft tokens, or sampler evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PendingRoundV2 {
    Reserved {
        request: AuthenticatedRoundV2,
    },
    Prepared {
        request: AuthenticatedRoundV2,
        result: AuthenticatedResultV2,
    },
    Staged {
        request: AuthenticatedRoundV2,
        result: AuthenticatedResultV2,
        renderer: RendererStageReceiptV2,
    },
}

pub struct DurableVerifierJournalV2 {
    root: PathBuf,
    _lock: File,
    key: MacKey,
    session: AuthenticatedSessionV2,
    session_record_sha256: String,
    evaluated_tokens: Vec<u32>,
    output_height: u64,
    head_sha256: String,
    frontier: FrontierV2,
    sampler_state: SamplerStateV2,
    reservations: BTreeMap<String, ReservationRecordV2>,
    prepared: BTreeMap<String, PreparedRecordV2>,
    staged: BTreeMap<String, StagedRecordV2>,
    completions: BTreeMap<String, AuthenticatedResultV2>,
    acknowledgements: BTreeMap<String, AckRecordV2>,
    tombstones: BTreeMap<String, TombstoneRecordV2>,
    pending_activation: Option<CommitRecordV2>,
}

impl DurableVerifierJournalV2 {
    pub fn create(
        root: &Path,
        session: AuthenticatedSessionV2,
        key: MacKey,
        now_unix_ms: u64,
        expected_key_id: &str,
        minimum_epoch: u64,
    ) -> Result<Self> {
        if !root.is_absolute() || root.exists() || root.is_symlink() {
            return Err(invalid("journal destination must be a new absolute path"));
        }
        session.verify_live(&key, now_unix_ms, expected_key_id, minimum_epoch)?;
        fs::create_dir(root)?;
        fs::set_permissions(root, fs::Permissions::from_mode(0o700))?;
        for name in [
            "reservations",
            "prepared",
            "staged",
            "commits",
            "active",
            "acks",
            "tombstones",
            "fragments",
        ] {
            let path = root.join(name);
            fs::create_dir(&path)?;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o700))?;
        }
        let lock = open_lock(root)?;
        write_exclusive(&root.join("session.json"), &canonical(&session)?)?;
        sync_directory(root)?;
        Self::from_session(root, lock, key, session)
    }

    pub fn open(
        root: &Path,
        key: MacKey,
        expected_key_id: &str,
        minimum_epoch: u64,
    ) -> Result<Self> {
        validate_journal_root(root)?;
        let lock = open_lock(root)?;
        let session: AuthenticatedSessionV2 =
            read_json(&root.join("session.json"), 4 * 1024 * 1024)?;
        session.verify_historical(&key, expected_key_id, minimum_epoch)?;
        let mut journal = Self::from_session(root, lock, key, session)?;
        journal.restore_records()?;
        Ok(journal)
    }

    fn from_session(
        root: &Path,
        lock: File,
        key: MacKey,
        session: AuthenticatedSessionV2,
    ) -> Result<Self> {
        let session_record_sha256 = session.record_digest()?;
        Ok(Self {
            root: root.to_path_buf(),
            _lock: lock,
            key,
            session: session.clone(),
            session_record_sha256,
            evaluated_tokens: session.core.initial_evaluated_tokens.clone(),
            output_height: session.core.initial_output_height,
            head_sha256: session.core.genesis_head_sha256.clone(),
            frontier: session.core.initial_frontier.clone(),
            sampler_state: session.core.initial_sampler_state.clone(),
            reservations: BTreeMap::new(),
            prepared: BTreeMap::new(),
            staged: BTreeMap::new(),
            completions: BTreeMap::new(),
            acknowledgements: BTreeMap::new(),
            tombstones: BTreeMap::new(),
            pending_activation: None,
        })
    }

    pub fn session(&self) -> &SessionCoreV2 {
        &self.session.core
    }

    /// Return the complete authenticated session record for target admission
    /// or reconnection. `session()` remains the convenient core-only accessor.
    pub fn authenticated_session(&self) -> &AuthenticatedSessionV2 {
        &self.session
    }

    pub fn current_head(&self) -> (u64, &str, &FrontierV2, &SamplerStateV2) {
        (
            self.output_height,
            &self.head_sha256,
            &self.frontier,
            &self.sampler_state,
        )
    }

    pub fn current_transcript_sha256(&self) -> String {
        token_digest(&self.evaluated_tokens)
    }

    pub fn evaluated_tokens(&self) -> &[u32] {
        &self.evaluated_tokens
    }

    /// Return an activated completion retained for exact replay. PREPARED,
    /// staged, and WAL-only results are intentionally invisible here.
    pub fn completed_result(&self, request_id: &str) -> Option<&AuthenticatedResultV2> {
        self.completions.get(request_id)
    }

    pub fn pending_activation(&self) -> bool {
        self.pending_activation.is_some()
    }

    /// Recover the exact unfinished request at the current parent and its
    /// durable phase. A WAL-committed activation must be recovered first; it is
    /// already a linearized decision rather than a Reserved/Prepared/Staged
    /// request.
    pub fn pending_round(&self) -> Result<Option<PendingRoundV2>> {
        if self.pending_activation.is_some() {
            return Err(invalid(
                "journal has a WAL-committed activation; recover it before resuming rounds",
            ));
        }
        let candidates = self
            .reservations
            .iter()
            .filter(|(request_id, reservation)| {
                !self.completions.contains_key(*request_id)
                    && !self.tombstones.contains_key(*request_id)
                    && self.matches_current_parent(&reservation.request)
            })
            .collect::<Vec<_>>();
        if candidates.len() > 1 {
            return Err(invalid(
                "journal has multiple unfinished children of the current head",
            ));
        }
        let Some((request_id, reservation)) = candidates.first().copied() else {
            return Ok(None);
        };
        let request = reservation.request.clone();
        let Some(prepared) = self.prepared.get(request_id) else {
            return Ok(Some(PendingRoundV2::Reserved { request }));
        };
        let result = prepared.result.clone();
        let Some(staged) = self.staged.get(request_id) else {
            return Ok(Some(PendingRoundV2::Prepared { request, result }));
        };
        Ok(Some(PendingRoundV2::Staged {
            request,
            result,
            renderer: staged.renderer.clone(),
        }))
    }

    pub fn reserve<F: AuthorityFenceV2>(
        &mut self,
        request: &AuthenticatedRoundV2,
        fence: &F,
        now_unix_ms: u64,
    ) -> Result<ReserveOutcomeV2> {
        if self.pending_activation.is_some() {
            return Err(invalid("journal has an unactivated committed result"));
        }
        request.verify_historical(&self.session.core, &self.key)?;
        let intent = request.intent_sha256()?;
        if let Some(tombstone) = self.tombstones.get(&request.core.request_id) {
            if tombstone.request_id == request.core.request_id {
                return Err(VerifierV2Error::ResyncRequired);
            }
        }
        if let Some(result) = self.completions.get(&request.core.request_id) {
            if result.core.request_intent_sha256 != intent {
                return Err(VerifierV2Error::ChangedIntent);
            }
            return Ok(ReserveOutcomeV2::Replay(Box::new(result.clone())));
        }
        if let Some(prepared) = self.prepared.get(&request.core.request_id) {
            if prepared.request_intent_sha256 != intent {
                return Err(VerifierV2Error::ChangedIntent);
            }
            return Ok(ReserveOutcomeV2::Prepared(Box::new(
                prepared.result.clone(),
            )));
        }
        if let Some(reservation) = self.reservations.get(&request.core.request_id) {
            if reservation.request.intent_sha256()? != intent {
                return Err(VerifierV2Error::ChangedIntent);
            }
            return Ok(ReserveOutcomeV2::Reserved);
        }
        self.check_live_fence(fence, now_unix_ms)?;
        if matches!(self.frontier, FrontierV2::Closed { .. }) {
            return Err(VerifierV2Error::Terminal);
        }
        request.verify_live(&self.session.core, &self.key, now_unix_ms)?;
        if !self.matches_current_parent(request) {
            return Ok(ReserveOutcomeV2::Stale {
                output_height: self.output_height,
                head_sha256: self.head_sha256.clone(),
                frontier: self.frontier.clone(),
                sampler_state: self.sampler_state.clone(),
                transcript_sha256: token_digest(&self.evaluated_tokens),
            });
        }
        if self.reservations.values().any(|reservation| {
            reservation.request.core.base_head_sha256 == self.head_sha256
                && reservation.request.core.request_id != request.core.request_id
        }) {
            return Err(VerifierV2Error::BusyParent);
        }
        let record = ReservationRecordV2 {
            request: request.clone(),
            reserved_unix_ms: now_unix_ms,
        };
        self.write_slot(
            "reservations",
            &request.core.request_id,
            "reservation",
            &record,
        )?;
        self.reservations
            .insert(request.core.request_id.clone(), record);
        Ok(ReserveOutcomeV2::Reserved)
    }

    pub fn record_prepared<F: AuthorityFenceV2>(
        &mut self,
        request: &AuthenticatedRoundV2,
        result: AuthenticatedResultV2,
        fence: &F,
        now_unix_ms: u64,
    ) -> Result<()> {
        self.check_live_fence(fence, now_unix_ms)?;
        result.verify_against(request, &self.session.core)?;
        let intent = request.intent_sha256()?;
        let reservation = self
            .reservations
            .get(&request.core.request_id)
            .ok_or_else(|| invalid("result has no durable reservation"))?;
        if reservation.request != *request || !self.matches_current_parent(request) {
            return Err(VerifierV2Error::StaleHead);
        }
        if let Some(existing) = self.prepared.get(&request.core.request_id) {
            return if existing.result == result {
                Ok(())
            } else {
                Err(VerifierV2Error::ChangedIntent)
            };
        }
        let record = PreparedRecordV2 {
            request_intent_sha256: intent,
            result,
            prepared_unix_ms: now_unix_ms,
        };
        self.write_slot("prepared", &request.core.request_id, "prepared", &record)?;
        self.prepared
            .insert(request.core.request_id.clone(), record);
        Ok(())
    }

    pub fn stage_result<R: VerifierRendererV2>(
        &mut self,
        request: &AuthenticatedRoundV2,
        closure: &VerifiedClosureV2,
        renderer: &mut R,
        now_unix_ms: u64,
    ) -> Result<RendererStageReceiptV2> {
        let prepared = self
            .prepared
            .get(&request.core.request_id)
            .ok_or_else(|| invalid("cannot stage before PREPARED"))?
            .clone();
        closure.verify_result(&prepared.result)?;
        if let Some(existing) = self.staged.get(&request.core.request_id) {
            if existing.result_sha256 == prepared.result.record_digest()? {
                return Ok(existing.renderer.clone());
            }
            return Err(VerifierV2Error::ChangedIntent);
        }
        for fragment in closure.fragments() {
            self.write_fragment(fragment)?;
        }
        let receipt = renderer
            .stage(&prepared.result, closure, now_unix_ms)
            .map_err(VerifierV2Error::Renderer)?;
        validate_stage_receipt(&receipt, &prepared.result)?;
        let record = StagedRecordV2 {
            request_id: request.core.request_id.clone(),
            request_intent_sha256: prepared.request_intent_sha256,
            result_sha256: prepared.result.record_digest()?,
            renderer: receipt.clone(),
            staged_unix_ms: now_unix_ms,
        };
        self.write_slot("staged", &request.core.request_id, "staged", &record)?;
        self.staged.insert(request.core.request_id.clone(), record);
        Ok(receipt)
    }

    pub fn commit_and_activate<F: AuthorityFenceV2, R: VerifierRendererV2>(
        &mut self,
        request: &AuthenticatedRoundV2,
        fence: &F,
        renderer: &mut R,
        now_unix_ms: u64,
    ) -> Result<AuthenticatedResultV2> {
        if self.pending_activation.is_some() {
            self.recover_pending(renderer, now_unix_ms)?;
        }
        self.check_live_fence(fence, now_unix_ms)?;
        if let Some(result) = self.completions.get(&request.core.request_id) {
            return Ok(result.clone());
        }
        let prepared = self
            .prepared
            .get(&request.core.request_id)
            .ok_or_else(|| invalid("cannot commit before PREPARED"))?
            .clone();
        let staged = self
            .staged
            .get(&request.core.request_id)
            .ok_or_else(|| invalid("cannot commit before durable renderer stage"))?
            .clone();
        if prepared.request_intent_sha256 != request.intent_sha256()?
            || staged.result_sha256 != prepared.result.record_digest()?
            || !self.matches_current_parent(request)
        {
            return Err(VerifierV2Error::StaleHead);
        }
        let commit = CommitRecordV2 {
            request: request.clone(),
            result: prepared.result.clone(),
            renderer: staged.renderer,
            committed_unix_ms: now_unix_ms,
        };
        self.write_slot("commits", &request.core.request_id, "commit", &commit)?;
        self.pending_activation = Some(commit);
        self.recover_pending(renderer, now_unix_ms)?;
        self.completions
            .get(&request.core.request_id)
            .cloned()
            .ok_or_else(|| invalid("activation did not publish completion"))
    }

    /// Finish a result which was already durably linearized before a crash.
    /// No live lease is consulted: a WAL commit is the authority decision.
    /// Uncommitted PREPARED work still requires the current fence.
    pub fn recover_pending<R: VerifierRendererV2>(
        &mut self,
        renderer: &mut R,
        now_unix_ms: u64,
    ) -> Result<()> {
        let Some(commit) = self.pending_activation.clone() else {
            return Ok(());
        };
        validate_stage_receipt(&commit.renderer, &commit.result)?;
        renderer
            .activate(&commit.renderer)
            .map_err(VerifierV2Error::Renderer)?;
        let active = ActiveRecordV2 {
            request_id: commit.request.core.request_id.clone(),
            result_sha256: commit.result.record_digest()?,
            new_head_sha256: commit.result.core.new_head_sha256.clone(),
            activated_unix_ms: now_unix_ms,
        };
        self.write_active(&active)?;
        self.apply_active(&commit.request, &commit.result)?;
        self.pending_activation = None;
        Ok(())
    }

    pub fn acknowledge(
        &mut self,
        request_id: &str,
        applied_head_sha256: &str,
        now_unix_ms: u64,
    ) -> Result<()> {
        validate_digest("applied head", applied_head_sha256)?;
        let result = self
            .completions
            .get(request_id)
            .ok_or_else(|| invalid("cannot ACK an unknown completion"))?;
        if applied_head_sha256 != result.core.new_head_sha256 {
            return Err(invalid("ACK head differs from completion"));
        }
        let ack = AckRecordV2 {
            request_id: request_id.into(),
            result_sha256: result.record_digest()?,
            applied_head_sha256: applied_head_sha256.into(),
            acknowledged_unix_ms: now_unix_ms,
        };
        if let Some(existing) = self.acknowledgements.get(request_id) {
            return if existing == &ack || existing.result_sha256 == ack.result_sha256 {
                Ok(())
            } else {
                Err(VerifierV2Error::ChangedIntent)
            };
        }
        self.write_slot("acks", request_id, "ack", &ack)?;
        self.acknowledgements.insert(request_id.into(), ack);
        Ok(())
    }

    pub fn garbage_collect(
        &mut self,
        request_id: &str,
        now_unix_ms: u64,
        retry_retention_ms: u64,
    ) -> Result<usize> {
        let result = self
            .completions
            .get(request_id)
            .ok_or_else(|| invalid("cannot retire an unknown completion"))?
            .clone();
        let result_sha256 = result.record_digest()?;
        if let Some(tombstone) = self.tombstones.get(request_id) {
            // The tombstone is the durable proof that ACK and retention were
            // checked before a previous cleanup attempt. Reusing it makes GC
            // recoverable after a crash between any two unlink operations.
            if tombstone.result_sha256 != result_sha256 {
                return Err(VerifierV2Error::ChangedIntent);
            }
        } else {
            let ack = self
                .acknowledgements
                .get(request_id)
                .ok_or_else(|| invalid("cannot retire an unacknowledged completion"))?;
            if ack.result_sha256 != result_sha256
                || now_unix_ms
                    < result
                        .core
                        .completed_unix_ms
                        .saturating_add(retry_retention_ms)
            {
                return Err(invalid("completion retry retention has not elapsed"));
            }
            let tombstone = TombstoneRecordV2 {
                request_id: request_id.into(),
                result_sha256,
                retired_unix_ms: now_unix_ms,
            };
            self.write_slot("tombstones", request_id, "tombstone", &tombstone)?;
            self.tombstones.insert(request_id.into(), tombstone);
        }

        let protected = self
            .completions
            .iter()
            .filter(|(other, _)| other.as_str() != request_id)
            .flat_map(|(_, completion)| {
                completion
                    .core
                    .fragments
                    .iter()
                    .map(|item| item.sha256.clone())
            })
            .chain(
                self.prepared
                    .iter()
                    .filter(|(other, _)| other.as_str() != request_id)
                    .flat_map(|(_, prepared)| {
                        prepared
                            .result
                            .core
                            .fragments
                            .iter()
                            .map(|item| item.sha256.clone())
                    }),
            )
            .collect::<BTreeSet<_>>();
        let mut removed = 0usize;
        for fragment in &result.core.fragments {
            if !protected.contains(&fragment.sha256) {
                let path = self.fragment_path(&fragment.sha256);
                if path.exists() {
                    fs::remove_file(path)?;
                    removed += 1;
                }
            }
        }
        // Commit/active records are the compact ordered transcript and must
        // survive unless a separately authenticated snapshot compacts them.
        // GC removes retry/evidence state only.
        for directory in ["reservations", "prepared", "staged", "acks"] {
            let path = self.slot_path(directory, request_id);
            if path.exists() {
                fs::remove_file(path)?;
            }
        }
        sync_directory(&self.root)?;
        self.reservations.remove(request_id);
        self.prepared.remove(request_id);
        self.staged.remove(request_id);
        self.completions.remove(request_id);
        self.acknowledgements.remove(request_id);
        Ok(removed)
    }

    fn check_live_fence<F: AuthorityFenceV2>(&self, fence: &F, now_unix_ms: u64) -> Result<()> {
        let core = &self.session.core;
        if !fence.permits(
            &core.session_id,
            &core.log_writer_authority_id,
            &core.authority_lease_id,
            core.authority_term,
            &core.target_executor_id,
            now_unix_ms,
        ) {
            return Err(VerifierV2Error::LeaseNotLive);
        }
        Ok(())
    }

    fn matches_current_parent(&self, request: &AuthenticatedRoundV2) -> bool {
        request.core.base_output_height == self.output_height
            && request.core.base_head_sha256 == self.head_sha256
            && request.core.frontier_in == self.frontier
            && request.core.base_sampler_state == self.sampler_state
            && request.core.base_evaluated_tokens == self.evaluated_tokens
            && request.core.base_tokens_sha256 == token_digest(&self.evaluated_tokens)
    }

    fn apply_active(
        &mut self,
        request: &AuthenticatedRoundV2,
        result: &AuthenticatedResultV2,
    ) -> Result<()> {
        result.verify_against(request, &self.session.core)?;
        if let Some(existing) = self.completions.get(&request.core.request_id) {
            return if existing == result {
                Ok(())
            } else {
                Err(VerifierV2Error::ChangedIntent)
            };
        }
        if !self.matches_current_parent(request) {
            return Err(VerifierV2Error::StaleHead);
        }
        self.evaluated_tokens
            .extend_from_slice(&result.core.committed_tokens);
        self.output_height = result.core.new_output_height;
        self.head_sha256.clone_from(&result.core.new_head_sha256);
        self.frontier.clone_from(&result.core.new_frontier);
        self.sampler_state
            .clone_from(&result.core.post_verify_sampler_state);
        self.completions
            .insert(request.core.request_id.clone(), result.clone());
        Ok(())
    }

    fn restore_records(&mut self) -> Result<()> {
        self.tombstones = self.read_slots("tombstones", "tombstone")?;
        self.reservations = self.read_slots("reservations", "reservation")?;
        self.prepared = self.read_slots("prepared", "prepared")?;
        self.staged = self.read_slots("staged", "staged")?;
        self.acknowledgements = self.read_slots("acks", "ack")?;
        // Tombstone publication is the GC linearization point. Ignore any
        // pre-commit records left by a crash during the subsequent unlink
        // sequence; the retained commit/active chain remains authoritative.
        self.reservations
            .retain(|request_id, _| !self.tombstones.contains_key(request_id));
        self.prepared
            .retain(|request_id, _| !self.tombstones.contains_key(request_id));
        self.staged
            .retain(|request_id, _| !self.tombstones.contains_key(request_id));
        for reservation in self.reservations.values() {
            reservation
                .request
                .verify_historical(&self.session.core, &self.key)?;
        }
        for (request_id, prepared) in &self.prepared {
            let reservation = self
                .reservations
                .get(request_id)
                .ok_or_else(|| invalid("PREPARED record has no reservation"))?;
            if prepared.request_intent_sha256 != reservation.request.intent_sha256()? {
                return Err(invalid("PREPARED intent differs from reservation"));
            }
            prepared
                .result
                .verify_against(&reservation.request, &self.session.core)?;
        }
        for (request_id, staged) in &self.staged {
            let prepared = self
                .prepared
                .get(request_id)
                .ok_or_else(|| invalid("staged render has no PREPARED result"))?;
            if staged.request_intent_sha256 != prepared.request_intent_sha256
                || staged.result_sha256 != prepared.result.record_digest()?
            {
                return Err(invalid("staged render differs from PREPARED result"));
            }
            validate_stage_receipt(&staged.renderer, &prepared.result)?;
            self.verify_fragment_cas(&prepared.result)?;
        }
        let mut commits: BTreeMap<String, CommitRecordV2> = self.read_slots("commits", "commit")?;
        let active: BTreeMap<String, ActiveRecordV2> = self.read_slots("active", "active")?;
        if active
            .keys()
            .any(|request_id| !commits.contains_key(request_id))
        {
            return Err(invalid("active record has no WAL commit"));
        }
        while !commits.is_empty() {
            let candidates = commits
                .iter()
                .filter(|(_, commit)| self.matches_current_parent(&commit.request))
                .map(|(request_id, _)| request_id.clone())
                .collect::<Vec<_>>();
            if candidates.len() != 1 {
                return Err(invalid("journal commit chain is not uniquely causal"));
            }
            let request_id = &candidates[0];
            let commit = commits
                .remove(request_id)
                .expect("causal candidate came from commit map");
            commit
                .request
                .verify_historical(&self.session.core, &self.key)?;
            commit
                .result
                .verify_against(&commit.request, &self.session.core)?;
            validate_stage_receipt(&commit.renderer, &commit.result)?;
            if !self
                .tombstones
                .contains_key(&commit.request.core.request_id)
            {
                self.verify_fragment_cas(&commit.result)?;
            }
            let result_sha256 = commit.result.record_digest()?;
            match active.get(&commit.request.core.request_id) {
                Some(record)
                    if record.result_sha256 == result_sha256
                        && record.new_head_sha256 == commit.result.core.new_head_sha256 =>
                {
                    self.apply_active(&commit.request, &commit.result)?;
                }
                Some(_) => return Err(invalid("active record differs from WAL commit")),
                None if self.pending_activation.is_none() && commits.is_empty() => {
                    // The WAL decision is durable but invisible until renderer
                    // recovery succeeds. No child can follow an inactive head.
                    self.pending_activation = Some(commit);
                }
                None => return Err(invalid("journal commit/active chain is ambiguous")),
            }
        }
        let completed_ids = self.completions.keys().cloned().collect::<Vec<_>>();
        for request_id in completed_ids {
            self.reservations.remove(&request_id);
            self.prepared.remove(&request_id);
            self.staged.remove(&request_id);
        }
        for (request_id, ack) in &self.acknowledgements {
            let completion = self
                .completions
                .get(request_id)
                .ok_or_else(|| invalid("ACK has no active completion"))?;
            if ack.result_sha256 != completion.record_digest()?
                || ack.applied_head_sha256 != completion.core.new_head_sha256
            {
                return Err(invalid("ACK differs from active completion"));
            }
        }
        for (request_id, tombstone) in &self.tombstones {
            let completion = self
                .completions
                .get(request_id)
                .ok_or_else(|| invalid("tombstone has no retained commit"))?;
            if tombstone.result_sha256 != completion.record_digest()? {
                return Err(invalid("tombstone differs from retained commit"));
            }
        }
        Ok(())
    }

    fn write_fragment(&self, fragment: &VerifiedFragmentV2) -> Result<()> {
        let path = self.fragment_path(&fragment.descriptor.sha256);
        if path.exists() {
            let metadata = fs::symlink_metadata(&path)?;
            if !metadata.file_type().is_file()
                || metadata.file_type().is_symlink()
                || metadata.permissions().mode() & 0o077 != 0
            {
                return Err(invalid("fragment CAS object mode differs"));
            }
            let payload = fs::read(&path)?;
            if payload != fragment.payload {
                return Err(invalid("content-addressed fragment collision"));
            }
            return Ok(());
        }
        write_exclusive(&path, &fragment.payload)
    }

    fn verify_fragment_cas(&self, result: &AuthenticatedResultV2) -> Result<()> {
        for descriptor in &result.core.fragments {
            let path = self.fragment_path(&descriptor.sha256);
            let metadata = fs::symlink_metadata(&path)?;
            if !metadata.file_type().is_file()
                || metadata.file_type().is_symlink()
                || metadata.permissions().mode() & 0o077 != 0
                || metadata.len() != descriptor.byte_len
            {
                return Err(invalid("staged fragment CAS object mode or length differs"));
            }
            let payload = fs::read(path)?;
            if sha256_hex(&payload) != descriptor.sha256 {
                return Err(invalid("staged fragment CAS digest differs"));
            }
        }
        Ok(())
    }

    fn write_active(&self, active: &ActiveRecordV2) -> Result<()> {
        let path = self.slot_path("active", &active.request_id);
        if path.exists() {
            let envelope: JournalEnvelopeV2 = read_json(&path, 16 * 1024 * 1024)?;
            let existing: ActiveRecordV2 =
                decode_journal(&envelope, "active", &self.session_record_sha256, &self.key)?;
            if existing.request_id == active.request_id
                && existing.result_sha256 == active.result_sha256
                && existing.new_head_sha256 == active.new_head_sha256
            {
                return Ok(());
            }
            return Err(VerifierV2Error::ChangedIntent);
        }
        self.write_slot("active", &active.request_id, "active", active)
    }

    fn fragment_path(&self, digest: &str) -> PathBuf {
        self.root.join("fragments").join(format!("{digest}.bin"))
    }

    fn slot_path(&self, directory: &str, request_id: &str) -> PathBuf {
        self.root
            .join(directory)
            .join(format!("{}.json", sha256_hex(request_id.as_bytes())))
    }

    fn write_slot<T: Serialize>(
        &self,
        directory: &str,
        request_id: &str,
        kind: &str,
        payload: &T,
    ) -> Result<()> {
        let envelope = journal_envelope(kind, &self.session_record_sha256, payload, &self.key)?;
        let path = self.slot_path(directory, request_id);
        if path.exists() {
            let existing: JournalEnvelopeV2 = read_json(&path, 16 * 1024 * 1024)?;
            if canonical(&existing)? == canonical(&envelope)? {
                return Ok(());
            }
            return Err(VerifierV2Error::ChangedIntent);
        }
        write_exclusive(&path, &canonical(&envelope)?)
    }

    fn read_slots<T>(&self, directory: &str, kind: &str) -> Result<BTreeMap<String, T>>
    where
        T: for<'de> Deserialize<'de> + RequestIdentityV2,
    {
        let mut records = BTreeMap::new();
        let directory_path = self.root.join(directory);
        for entry in fs::read_dir(&directory_path)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) != Some("json") {
                return Err(invalid("journal directory contains an unexpected object"));
            }
            let envelope: JournalEnvelopeV2 = read_json(&path, 16 * 1024 * 1024)?;
            let record: T =
                decode_journal(&envelope, kind, &self.session_record_sha256, &self.key)?;
            let request_id = record.request_id().to_owned();
            if self.slot_path(directory, &request_id) != path
                || records.insert(request_id, record).is_some()
            {
                return Err(invalid("journal slot filename or identity differs"));
            }
        }
        Ok(records)
    }
}

trait RequestIdentityV2 {
    fn request_id(&self) -> &str;
}

impl RequestIdentityV2 for ReservationRecordV2 {
    fn request_id(&self) -> &str {
        &self.request.core.request_id
    }
}
impl RequestIdentityV2 for PreparedRecordV2 {
    fn request_id(&self) -> &str {
        &self.result.core.request_id
    }
}
impl RequestIdentityV2 for StagedRecordV2 {
    fn request_id(&self) -> &str {
        &self.request_id
    }
}
impl RequestIdentityV2 for CommitRecordV2 {
    fn request_id(&self) -> &str {
        &self.request.core.request_id
    }
}
impl RequestIdentityV2 for ActiveRecordV2 {
    fn request_id(&self) -> &str {
        &self.request_id
    }
}
impl RequestIdentityV2 for AckRecordV2 {
    fn request_id(&self) -> &str {
        &self.request_id
    }
}
impl RequestIdentityV2 for TombstoneRecordV2 {
    fn request_id(&self) -> &str {
        &self.request_id
    }
}

fn validate_stage_receipt(
    receipt: &RendererStageReceiptV2,
    result: &AuthenticatedResultV2,
) -> Result<()> {
    validate_digest("renderer result", &receipt.result_sha256)?;
    validate_digest("renderer root", &receipt.render_root_sha256)?;
    validate_identifier("renderer stage token", &receipt.stage_token)?;
    if receipt.result_sha256 != result.record_digest()?
        || receipt.render_root_sha256 != result.core.fragments_sha256
    {
        return Err(invalid("renderer stage belongs to a different result"));
    }
    Ok(())
}

fn validate_journal_root(root: &Path) -> Result<()> {
    if !root.is_absolute() {
        return Err(invalid("journal root must be absolute"));
    }
    let metadata = fs::symlink_metadata(root)?;
    if !metadata.file_type().is_dir()
        || metadata.file_type().is_symlink()
        || metadata.permissions().mode() & 0o077 != 0
    {
        return Err(invalid(
            "journal root must be a private non-symlink directory",
        ));
    }
    for name in [
        "reservations",
        "prepared",
        "staged",
        "commits",
        "active",
        "acks",
        "tombstones",
        "fragments",
    ] {
        let directory = fs::symlink_metadata(root.join(name))?;
        if !directory.file_type().is_dir()
            || directory.file_type().is_symlink()
            || directory.permissions().mode() & 0o077 != 0
        {
            return Err(invalid("journal subdirectory mode differs"));
        }
    }
    Ok(())
}

fn open_lock(root: &Path) -> Result<File> {
    let path = root.join(".lock");
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .mode(0o600)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)?;
    // SAFETY: flock only reads the valid descriptor and does not retain a
    // pointer. The File stays owned by the journal for the lock lifetime.
    let status = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if status != 0 {
        return Err(VerifierV2Error::Io("journal is already locked".into()));
    }
    Ok(file)
}

fn write_exclusive(path: &Path, payload: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)?;
    file.write_all(payload)?;
    file.sync_all()?;
    sync_directory(
        path.parent()
            .ok_or_else(|| invalid("journal object has no parent"))?,
    )
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path, maximum_bytes: u64) -> Result<T> {
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.file_type().is_file()
        || metadata.file_type().is_symlink()
        || metadata.len() == 0
        || metadata.len() > maximum_bytes
        || metadata.permissions().mode() & 0o077 != 0
    {
        return Err(invalid("journal object mode or size differs"));
    }
    let bytes = fs::read(path)?;
    serde_json::from_slice(&bytes).map_err(|error| invalid(error.to_string()))
}

fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    const NOW: u64 = 1_787_060_000_000;

    fn digest(byte: u8) -> String {
        format!("{byte:02x}").repeat(32)
    }

    fn identity() -> CompositeIdentityV2 {
        CompositeIdentityV2 {
            semantic_product_sha256: digest(1),
            prefix_checkpoint_sha256: digest(2),
            target_checkpoint_sha256: digest(3),
            target_engine_sha256: digest(4),
            target_sampler_sha256: digest(5),
            draft_checkpoint_sha256: digest(6),
            draft_engine_sha256: digest(7),
            tokenizer_sha256: digest(8),
            vocabulary_sha256: digest(9),
            target_genesis_root_sha256: digest(10),
            draft_genesis_root_sha256: digest(11),
            portable_kv_abi: "muser-handoff-v2-f16le".into(),
        }
    }

    fn session(
        key: &MacKey,
        signer: &TargetSigningKeyV2,
        policy: CouplingPolicyV2,
        max_context_tokens: u32,
    ) -> AuthenticatedSessionV2 {
        let initial_sampler_state = match policy {
            CouplingPolicyV2::SparseMaximal { .. } => {
                SamplerStateV2::from_mt(&Mt19937::new(0x1234_5678).snapshot()).unwrap()
            }
            _ => SamplerStateV2::stateless(),
        };
        AuthenticatedSessionV2::sign(
            SessionCoreV2 {
                protocol: VERIFIER_SESSION_PROTOCOL_V2.into(),
                session_id: "session-a".into(),
                client_incarnation: "mac-boot-a".into(),
                log_writer_authority_id: "mac-a".into(),
                authority_lease_id: "lease-a".into(),
                authority_term: 7,
                target_executor_id: "gx10-a".into(),
                target_public_key: signer.public_key(),
                created_unix_ms: NOW,
                expires_unix_ms: NOW + 10_000,
                hmac_key_id: "request-key-a".into(),
                hmac_key_epoch: 3,
                identity: identity(),
                coupling_policy: policy,
                sampler_config_sha256: digest(12),
                vocab_size: 202_048,
                max_context_tokens,
                max_drafts: 15,
                max_total_fragment_bytes: 1 << 20,
                initial_evaluated_tokens: vec![1, 2, 3],
                initial_output_height: 0,
                initial_frontier: FrontierV2::Open {
                    token: 101,
                    output_ordinal: 0,
                },
                initial_sampler_state,
                fragment_requirements: vec![FragmentRequirementV2 {
                    component_id: "target-hidden".into(),
                    kind: FragmentKindV2::TargetHidden,
                    coverage: FragmentCoverageV2::CommittedInputs,
                    payload_abi: "f32-row-v1".into(),
                    bytes_per_logical_row: 4,
                    required: true,
                }],
                genesis_head_sha256: digest(0),
            },
            key,
        )
        .unwrap()
    }

    fn sparse_row(token: u32) -> SparseWeightRowV2 {
        SparseWeightRowV2 {
            vocab_size: 202_048,
            entries: vec![SparseWeightEntryV2 {
                token,
                weight_f32_bits: 1.0f32.to_bits(),
            }],
        }
    }

    fn request(
        journal: &DurableVerifierJournalV2,
        key: &MacKey,
        request_id: &str,
        drafts: Vec<u32>,
    ) -> AuthenticatedRoundV2 {
        let (draft_evidence, post_draft_sampler_state) = match &journal.session().coupling_policy {
            CouplingPolicyV2::SparseMaximal { max_support } => {
                let rows = drafts.iter().copied().map(sparse_row).collect::<Vec<_>>();
                let mut rng = journal.sampler_state.to_mt().unwrap();
                for (row, token) in rows.iter().zip(&drafts) {
                    assert_eq!(
                        row.sample(&mut rng, 202_048, *max_support as usize)
                            .unwrap(),
                        *token
                    );
                }
                (
                    DraftEvidenceV2::SparseMaximal { q_rows: rows },
                    SamplerStateV2::from_mt(&rng.snapshot()).unwrap(),
                )
            }
            CouplingPolicyV2::Greedy => (DraftEvidenceV2::None, journal.sampler_state.clone()),
            CouplingPolicyV2::SharedGumbel => (
                DraftEvidenceV2::SharedGumbel {
                    witness_sha256: digest(88),
                },
                journal.sampler_state.clone(),
            ),
        };
        AuthenticatedRoundV2::sign(
            RoundCoreV2 {
                protocol: VERIFIER_ROUND_PROTOCOL_V3.into(),
                intent: RoundIntentV2::Verify,
                session_id: journal.session().session_id.clone(),
                session_genesis_sha256: journal.session().genesis_head_sha256.clone(),
                authority_term: journal.session().authority_term,
                request_id: request_id.into(),
                created_unix_ms: NOW + 1,
                expires_unix_ms: NOW + 5_000,
                hmac_key_id: journal.session().hmac_key_id.clone(),
                hmac_key_epoch: journal.session().hmac_key_epoch,
                base_output_height: journal.output_height,
                base_head_sha256: journal.head_sha256.clone(),
                base_evaluated_tokens: journal.evaluated_tokens.clone(),
                base_tokens_sha256: token_digest(&journal.evaluated_tokens),
                frontier_in: journal.frontier.clone(),
                draft_tokens: drafts,
                draft_evidence,
                base_sampler_state: journal.sampler_state.clone(),
                post_draft_sampler_state,
            },
            journal.session(),
            key,
        )
        .unwrap()
    }

    fn close_request(
        journal: &DurableVerifierJournalV2,
        key: &MacKey,
        request_id: &str,
        reason: FinishReasonV2,
    ) -> AuthenticatedRoundV2 {
        AuthenticatedRoundV2::sign(
            RoundCoreV2 {
                protocol: VERIFIER_ROUND_PROTOCOL_V3.into(),
                intent: RoundIntentV2::Close { reason },
                session_id: journal.session().session_id.clone(),
                session_genesis_sha256: journal.session().genesis_head_sha256.clone(),
                authority_term: journal.session().authority_term,
                request_id: request_id.into(),
                created_unix_ms: NOW + 1,
                expires_unix_ms: NOW + 5_000,
                hmac_key_id: journal.session().hmac_key_id.clone(),
                hmac_key_epoch: journal.session().hmac_key_epoch,
                base_output_height: journal.output_height,
                base_head_sha256: journal.head_sha256.clone(),
                base_evaluated_tokens: journal.evaluated_tokens.clone(),
                base_tokens_sha256: token_digest(&journal.evaluated_tokens),
                frontier_in: journal.frontier.clone(),
                draft_tokens: Vec::new(),
                draft_evidence: DraftEvidenceV2::None,
                base_sampler_state: journal.sampler_state.clone(),
                post_draft_sampler_state: journal.sampler_state.clone(),
            },
            journal.session(),
            key,
        )
        .unwrap()
    }

    fn fragments(base: u64, committed: usize) -> (Vec<FragmentDescriptorV2>, Vec<Vec<u8>>) {
        let payloads = (0..committed)
            .map(|index| (index as u32 + 10).to_le_bytes().to_vec())
            .collect::<Vec<_>>();
        let descriptors = payloads
            .iter()
            .enumerate()
            .map(|(index, payload)| FragmentDescriptorV2 {
                ordinal: index as u32,
                component_id: "target-hidden".into(),
                kind: FragmentKindV2::TargetHidden,
                logical_start: base + index as u64,
                logical_count: 1,
                payload_abi: "f32-row-v1".into(),
                bytes_per_logical_row: 4,
                byte_len: 4,
                sha256: sha256_hex(payload),
            })
            .collect();
        (descriptors, payloads)
    }

    fn open_result(
        request: &AuthenticatedRoundV2,
        session: &SessionCoreV2,
        signer: &TargetSigningKeyV2,
        accepted_drafts: u32,
        frontier_out: u32,
    ) -> (AuthenticatedResultV2, Vec<Vec<u8>>) {
        let (descriptors, payloads) = fragments(
            request.core.base_output_height,
            accepted_drafts as usize + 1,
        );
        let result = build_result(
            request,
            session,
            ResultDecisionV2::Open {
                accepted_drafts,
                frontier_out,
            },
            request.core.post_draft_sampler_state.clone(),
            descriptors,
            NOW + 3,
            signer,
        )
        .unwrap();
        (result, payloads)
    }

    fn closure(
        result: &AuthenticatedResultV2,
        request: &AuthenticatedRoundV2,
        session: &SessionCoreV2,
        payloads: &[Vec<u8>],
    ) -> VerifiedClosureV2 {
        let mut assembler = FragmentAssemblerV2::new(result, request, session).unwrap();
        for (ordinal, payload) in payloads.iter().enumerate().rev() {
            assert!(assembler.insert(ordinal as u32, payload.clone()).unwrap());
            assert!(!assembler.insert(ordinal as u32, payload.clone()).unwrap());
        }
        assembler.finish().unwrap()
    }

    struct Fence {
        live: bool,
        term: u64,
    }

    impl AuthorityFenceV2 for Fence {
        fn permits(
            &self,
            session_id: &str,
            log_writer_authority_id: &str,
            authority_lease_id: &str,
            authority_term: u64,
            target_executor_id: &str,
            _now_unix_ms: u64,
        ) -> bool {
            self.live
                && self.term == authority_term
                && session_id == "session-a"
                && log_writer_authority_id == "mac-a"
                && authority_lease_id == "lease-a"
                && target_executor_id == "gx10-a"
        }
    }

    #[derive(Default)]
    struct Renderer {
        fail_activation: bool,
        staged: BTreeSet<String>,
        activated: BTreeSet<String>,
    }

    impl VerifierRendererV2 for Renderer {
        fn stage(
            &mut self,
            result: &AuthenticatedResultV2,
            closure: &VerifiedClosureV2,
            now_unix_ms: u64,
        ) -> std::result::Result<RendererStageReceiptV2, String> {
            closure
                .verify_result(result)
                .map_err(|error| error.to_string())?;
            let result_sha256 = result.record_digest().map_err(|error| error.to_string())?;
            self.staged.insert(result_sha256.clone());
            Ok(RendererStageReceiptV2 {
                result_sha256,
                render_root_sha256: result.core.fragments_sha256.clone(),
                stage_token: result.core.request_id.clone(),
                staged_unix_ms: now_unix_ms,
            })
        }

        fn activate(
            &mut self,
            receipt: &RendererStageReceiptV2,
        ) -> std::result::Result<(), String> {
            if self.fail_activation {
                return Err("injected activation crash".into());
            }
            self.activated.insert(receipt.result_sha256.clone());
            Ok(())
        }
    }

    fn create_journal(
        directory: &TempDir,
        key: &MacKey,
        signer: &TargetSigningKeyV2,
        policy: CouplingPolicyV2,
    ) -> DurableVerifierJournalV2 {
        let root = directory.path().join("journal");
        DurableVerifierJournalV2::create(
            &root,
            session(key, signer, policy, 128),
            key.clone(),
            NOW,
            "request-key-a",
            3,
        )
        .unwrap()
    }

    #[test]
    fn sparse_request_replays_actual_draws_and_target_signature_is_directional() {
        let key = MacKey::from_bytes([0x31; 32]);
        let signer = TargetSigningKeyV2::from_seed([0x41; 32]).unwrap();
        let directory = TempDir::new().unwrap();
        let journal = create_journal(
            &directory,
            &key,
            &signer,
            CouplingPolicyV2::SparseMaximal { max_support: 40 },
        );
        let request = request(&journal, &key, "request-a", vec![102, 103]);
        request
            .verify_live(journal.session(), &key, NOW + 2)
            .unwrap();
        let DraftEvidenceV2::SparseMaximal { q_rows } = &request.core.draft_evidence else {
            panic!("wrong evidence")
        };
        assert_eq!(
            q_rows[0].probabilities(202_048, 40).unwrap().entries[0].probability,
            1.0
        );

        let (result, _) = open_result(&request, journal.session(), &signer, 1, 999);
        result.verify_against(&request, journal.session()).unwrap();
        let mut forged = result.clone();
        forged.target_signature_hex.replace_range(0..2, "00");
        assert_eq!(
            forged.verify_against(&request, journal.session()),
            Err(VerifierV2Error::Authentication)
        );

        let mut bad = request.core.clone();
        bad.draft_tokens[0] = 104;
        assert!(AuthenticatedRoundV2::sign(bad, journal.session(), &key).is_err());
        let mut bad_state = request.core.clone();
        let SamplerStateV2::Mt19937 { index, .. } = &mut bad_state.post_draft_sampler_state else {
            panic!("wrong sampler state")
        };
        *index = index.saturating_sub(1);
        assert!(AuthenticatedRoundV2::sign(bad_state, journal.session(), &key).is_err());
    }

    #[test]
    fn durable_round_replays_closes_fragments_and_gcs_only_after_ack() {
        let key = MacKey::from_bytes([0x32; 32]);
        let signer = TargetSigningKeyV2::from_seed([0x42; 32]).unwrap();
        let directory = TempDir::new().unwrap();
        let root = directory.path().join("journal");
        let mut journal = create_journal(
            &directory,
            &key,
            &signer,
            CouplingPolicyV2::SparseMaximal { max_support: 40 },
        );
        let fence = Fence {
            live: true,
            term: 7,
        };
        let round = request(&journal, &key, "request-a", vec![102, 103]);
        assert_eq!(
            journal.reserve(&round, &fence, NOW + 2).unwrap(),
            ReserveOutcomeV2::Reserved
        );
        let sibling = request(&journal, &key, "request-b", vec![102]);
        assert_eq!(
            journal.reserve(&sibling, &fence, NOW + 2),
            Err(VerifierV2Error::BusyParent)
        );
        let (result, payloads) = open_result(&round, journal.session(), &signer, 1, 999);
        journal
            .record_prepared(&round, result.clone(), &fence, NOW + 3)
            .unwrap();
        assert!(matches!(
            journal.reserve(&round, &fence, NOW + 4).unwrap(),
            ReserveOutcomeV2::Prepared(_)
        ));
        let closure = closure(&result, &round, journal.session(), &payloads);
        let mut renderer = Renderer::default();
        journal
            .stage_result(&round, &closure, &mut renderer, NOW + 4)
            .unwrap();
        let visible = journal
            .commit_and_activate(&round, &fence, &mut renderer, NOW + 5)
            .unwrap();
        assert_eq!(visible, result);
        assert_eq!(journal.output_height, 2);
        assert_eq!(journal.evaluated_tokens, vec![1, 2, 3, 101, 102]);
        assert!(matches!(
            journal.reserve(&round, &fence, NOW + 6).unwrap(),
            ReserveOutcomeV2::Replay(_)
        ));
        assert!(journal.garbage_collect("request-a", NOW + 6, 1).is_err());
        journal
            .acknowledge("request-a", &result.core.new_head_sha256, NOW + 6)
            .unwrap();
        assert_eq!(
            journal
                .garbage_collect("request-a", NOW + 20_000, 1)
                .unwrap(),
            2
        );
        assert_eq!(
            journal.reserve(&round, &fence, NOW + 20_001),
            Err(VerifierV2Error::ResyncRequired)
        );
        drop(journal);

        let mut restored =
            DurableVerifierJournalV2::open(&root, key.clone(), "request-key-a", 3).unwrap();
        assert_eq!(restored.output_height, 2);
        assert_eq!(restored.evaluated_tokens, vec![1, 2, 3, 101, 102]);
        assert_eq!(
            restored.reserve(&round, &fence, NOW + 20_001),
            Err(VerifierV2Error::ResyncRequired)
        );
    }

    #[test]
    fn restart_exposes_authenticated_session_and_exact_pending_phase() {
        let key = MacKey::from_bytes([0x38; 32]);
        let signer = TargetSigningKeyV2::from_seed([0x48; 32]).unwrap();
        let directory = TempDir::new().unwrap();
        let root = directory.path().join("journal");
        let fence = Fence {
            live: true,
            term: 7,
        };
        let mut journal = create_journal(&directory, &key, &signer, CouplingPolicyV2::Greedy);
        let authenticated_session = journal.authenticated_session().clone();
        assert_eq!(journal.pending_round().unwrap(), None);

        let round = request(&journal, &key, "request-resume", vec![102, 103]);
        journal.reserve(&round, &fence, NOW + 2).unwrap();
        drop(journal);

        let mut journal =
            DurableVerifierJournalV2::open(&root, key.clone(), "request-key-a", 3).unwrap();
        assert_eq!(journal.authenticated_session(), &authenticated_session);
        assert_eq!(
            journal.pending_round().unwrap(),
            Some(PendingRoundV2::Reserved {
                request: round.clone()
            })
        );

        let (result, payloads) = open_result(&round, journal.session(), &signer, 1, 999);
        journal
            .record_prepared(&round, result.clone(), &fence, NOW + 3)
            .unwrap();
        drop(journal);

        let mut journal =
            DurableVerifierJournalV2::open(&root, key.clone(), "request-key-a", 3).unwrap();
        assert_eq!(
            journal.pending_round().unwrap(),
            Some(PendingRoundV2::Prepared {
                request: round.clone(),
                result: result.clone(),
            })
        );

        let verified = closure(&result, &round, journal.session(), &payloads);
        let mut renderer = Renderer::default();
        let receipt = journal
            .stage_result(&round, &verified, &mut renderer, NOW + 4)
            .unwrap();
        drop(journal);

        let journal = DurableVerifierJournalV2::open(&root, key, "request-key-a", 3).unwrap();
        assert_eq!(
            journal.pending_round().unwrap(),
            Some(PendingRoundV2::Staged {
                request: round,
                result,
                renderer: receipt,
            })
        );
    }

    #[test]
    fn round_v2_verify_bytes_remain_compatible_and_v2_close_fails_closed() {
        let key = MacKey::from_bytes([0x39; 32]);
        let signer = TargetSigningKeyV2::from_seed([0x49; 32]).unwrap();
        let directory = TempDir::new().unwrap();
        let journal = create_journal(&directory, &key, &signer, CouplingPolicyV2::Greedy);
        let mut legacy_core = request(&journal, &key, "request-v2", vec![102]).core;
        legacy_core.protocol = VERIFIER_ROUND_PROTOCOL_V2.into();
        let round = AuthenticatedRoundV2::sign(legacy_core, journal.session(), &key).unwrap();
        assert_eq!(round.core.protocol, VERIFIER_ROUND_PROTOCOL_V2);
        assert_eq!(round.core.intent, RoundIntentV2::Verify);
        let core = serde_json::to_value(&round.core).unwrap();
        assert!(core.get("intent").is_none());
        let bytes = canonical(&round).unwrap();
        let restored: AuthenticatedRoundV2 = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(restored, round);
        restored.verify_historical(journal.session(), &key).unwrap();
        assert!(restored
            .verify_live(journal.session(), &key, NOW + 2)
            .is_err());

        // V2 had no authenticated request intent, so historical terminal
        // results remain valid. They are recoverable but cannot be admitted as
        // new work after the V3 boundary.
        let legacy_terminal = build_result(
            &round,
            journal.session(),
            ResultDecisionV2::Closed {
                commit_frontier: false,
                accepted_drafts: 0,
                reason: FinishReasonV2::MaxTokens,
            },
            round.core.base_sampler_state.clone(),
            Vec::new(),
            NOW + 3,
            &signer,
        )
        .unwrap();
        legacy_terminal
            .verify_against(&round, journal.session())
            .unwrap();

        let mut incompatible = round.core;
        incompatible.intent = RoundIntentV2::Close {
            reason: FinishReasonV2::MaxTokens,
        };
        incompatible.draft_tokens.clear();
        assert!(AuthenticatedRoundV2::sign(incompatible, journal.session(), &key).is_err());
    }

    #[test]
    fn authenticated_close_can_stop_at_full_context_without_evaluating_frontier() {
        let key = MacKey::from_bytes([0x3a; 32]);
        let signer = TargetSigningKeyV2::from_seed([0x4a; 32]).unwrap();
        let directory = TempDir::new().unwrap();
        let root = directory.path().join("journal");
        let mut full_context_session = session(
            &key,
            &signer,
            CouplingPolicyV2::SparseMaximal { max_support: 40 },
            3,
        )
        .core;
        full_context_session.fragment_requirements = vec![
            FragmentRequirementV2 {
                component_id: "target-hidden".into(),
                kind: FragmentKindV2::TargetHidden,
                coverage: FragmentCoverageV2::FreshTargetRows,
                payload_abi: "f32-row-v1".into(),
                bytes_per_logical_row: 4,
                required: true,
            },
            FragmentRequirementV2 {
                component_id: "sampler-witness".into(),
                kind: FragmentKindV2::SamplerWitness,
                coverage: FragmentCoverageV2::RoundSingleton,
                payload_abi: "f32-row-v1".into(),
                bytes_per_logical_row: 4,
                required: true,
            },
        ];
        let mut journal = DurableVerifierJournalV2::create(
            &root,
            AuthenticatedSessionV2::sign(full_context_session, &key).unwrap(),
            key.clone(),
            NOW,
            "request-key-a",
            3,
        )
        .unwrap();
        assert_eq!(journal.evaluated_tokens().len(), 3);

        let close = close_request(
            &journal,
            &key,
            "request-full-context",
            FinishReasonV2::MaxTokens,
        );
        close.verify_live(journal.session(), &key, NOW + 2).unwrap();
        let mut tampered_close = close.clone();
        tampered_close.core.intent = RoundIntentV2::Close {
            reason: FinishReasonV2::Cancelled,
        };
        assert_eq!(
            tampered_close.verify_historical(journal.session(), &key),
            Err(VerifierV2Error::Authentication)
        );

        let mut verify = close.core.clone();
        verify.intent = RoundIntentV2::Verify;
        verify.draft_evidence = DraftEvidenceV2::SparseMaximal { q_rows: Vec::new() };
        assert!(AuthenticatedRoundV2::sign(verify, journal.session(), &key).is_err());
        let mut eos_close = close.core.clone();
        eos_close.intent = RoundIntentV2::Close {
            reason: FinishReasonV2::Eos,
        };
        assert!(AuthenticatedRoundV2::sign(eos_close, journal.session(), &key).is_err());
        assert!(build_result(
            &close,
            journal.session(),
            ResultDecisionV2::Open {
                accepted_drafts: 0,
                frontier_out: 777,
            },
            close.core.base_sampler_state.clone(),
            Vec::new(),
            NOW + 3,
            &signer,
        )
        .is_err());
        assert!(build_result(
            &close,
            journal.session(),
            ResultDecisionV2::Closed {
                commit_frontier: false,
                accepted_drafts: 0,
                reason: FinishReasonV2::Cancelled,
            },
            close.core.base_sampler_state.clone(),
            Vec::new(),
            NOW + 3,
            &signer,
        )
        .is_err());

        let fence = Fence {
            live: true,
            term: 7,
        };
        journal.reserve(&close, &fence, NOW + 2).unwrap();
        let result = build_result(
            &close,
            journal.session(),
            ResultDecisionV2::Closed {
                commit_frontier: false,
                accepted_drafts: 0,
                reason: FinishReasonV2::MaxTokens,
            },
            close.core.base_sampler_state.clone(),
            Vec::new(),
            NOW + 3,
            &signer,
        )
        .unwrap();
        journal
            .record_prepared(&close, result.clone(), &fence, NOW + 3)
            .unwrap();
        let verified = FragmentAssemblerV2::new(&result, &close, journal.session())
            .unwrap()
            .finish()
            .unwrap();
        let mut renderer = Renderer::default();
        journal
            .stage_result(&close, &verified, &mut renderer, NOW + 4)
            .unwrap();
        journal
            .commit_and_activate(&close, &fence, &mut renderer, NOW + 5)
            .unwrap();
        assert_eq!(journal.evaluated_tokens(), &[1, 2, 3]);
        assert_eq!(
            journal.current_head().2,
            &FrontierV2::Closed {
                reason: FinishReasonV2::MaxTokens,
            }
        );
    }

    #[test]
    fn crash_after_prepare_and_commit_before_activation_recovers_exactly_once() {
        let key = MacKey::from_bytes([0x33; 32]);
        let signer = TargetSigningKeyV2::from_seed([0x43; 32]).unwrap();
        let directory = TempDir::new().unwrap();
        let root = directory.path().join("journal");
        let fence = Fence {
            live: true,
            term: 7,
        };
        let mut journal = create_journal(
            &directory,
            &key,
            &signer,
            CouplingPolicyV2::SparseMaximal { max_support: 40 },
        );
        let request = request(&journal, &key, "request-crash", vec![102]);
        journal.reserve(&request, &fence, NOW + 2).unwrap();
        let (result, payloads) = open_result(&request, journal.session(), &signer, 1, 777);
        journal
            .record_prepared(&request, result.clone(), &fence, NOW + 3)
            .unwrap();
        drop(journal);

        let mut restored =
            DurableVerifierJournalV2::open(&root, key.clone(), "request-key-a", 3).unwrap();
        assert!(matches!(
            restored.reserve(&request, &fence, NOW + 4).unwrap(),
            ReserveOutcomeV2::Prepared(_)
        ));
        let closure = closure(&result, &request, restored.session(), &payloads);
        let mut renderer = Renderer::default();
        restored
            .stage_result(&request, &closure, &mut renderer, NOW + 4)
            .unwrap();
        renderer.fail_activation = true;
        assert!(matches!(
            restored.commit_and_activate(&request, &fence, &mut renderer, NOW + 5),
            Err(VerifierV2Error::Renderer(_))
        ));
        assert!(restored.pending_activation());
        drop(restored);

        let mut recovered =
            DurableVerifierJournalV2::open(&root, key.clone(), "request-key-a", 3).unwrap();
        assert!(recovered.pending_activation());
        let mut healthy_renderer = Renderer::default();
        recovered
            .recover_pending(&mut healthy_renderer, NOW + 6)
            .unwrap();
        assert!(!recovered.pending_activation());
        assert_eq!(recovered.output_height, 2);
        assert!(matches!(
            recovered.reserve(&request, &fence, NOW + 7).unwrap(),
            ReserveOutcomeV2::Replay(_)
        ));
    }

    #[test]
    fn terminal_without_evaluating_frontier_has_no_bogus_token_or_render_row() {
        let key = MacKey::from_bytes([0x34; 32]);
        let signer = TargetSigningKeyV2::from_seed([0x44; 32]).unwrap();
        let directory = TempDir::new().unwrap();
        let mut journal = create_journal(&directory, &key, &signer, CouplingPolicyV2::Greedy);
        let fence = Fence {
            live: true,
            term: 7,
        };
        let request = close_request(
            &journal,
            &key,
            "request-terminal",
            FinishReasonV2::MaxTokens,
        );
        journal.reserve(&request, &fence, NOW + 2).unwrap();
        let result = build_result(
            &request,
            journal.session(),
            ResultDecisionV2::Closed {
                commit_frontier: false,
                accepted_drafts: 0,
                reason: FinishReasonV2::MaxTokens,
            },
            SamplerStateV2::stateless(),
            Vec::new(),
            NOW + 3,
            &signer,
        )
        .unwrap();
        journal
            .record_prepared(&request, result.clone(), &fence, NOW + 3)
            .unwrap();
        let closure = FragmentAssemblerV2::new(&result, &request, journal.session())
            .unwrap()
            .finish()
            .unwrap();
        let mut renderer = Renderer::default();
        journal
            .stage_result(&request, &closure, &mut renderer, NOW + 4)
            .unwrap();
        journal
            .commit_and_activate(&request, &fence, &mut renderer, NOW + 5)
            .unwrap();
        assert_eq!(journal.output_height, 0);
        assert_eq!(journal.evaluated_tokens, vec![1, 2, 3]);
        assert_eq!(
            journal.frontier,
            FrontierV2::Closed {
                reason: FinishReasonV2::MaxTokens
            }
        );

        let mut new_core = request.core.clone();
        new_core.request_id = "request-after-terminal".into();
        new_core.base_head_sha256 = journal.head_sha256.clone();
        let new_request = AuthenticatedRoundV2::sign(new_core, journal.session(), &key).unwrap();
        assert_eq!(
            journal.reserve(&new_request, &fence, NOW + 6),
            Err(VerifierV2Error::Terminal)
        );
    }

    #[test]
    fn causal_restore_handles_a_zero_height_terminal_transition() {
        let key = MacKey::from_bytes([0x36; 32]);
        let signer = TargetSigningKeyV2::from_seed([0x46; 32]).unwrap();
        let directory = TempDir::new().unwrap();
        let root = directory.path().join("journal");
        let fence = Fence {
            live: true,
            term: 7,
        };
        let mut journal = create_journal(&directory, &key, &signer, CouplingPolicyV2::Greedy);

        let first = request(&journal, &key, "zzz-open", Vec::new());
        journal.reserve(&first, &fence, NOW + 2).unwrap();
        let (first_result, first_payloads) =
            open_result(&first, journal.session(), &signer, 0, 777);
        journal
            .record_prepared(&first, first_result.clone(), &fence, NOW + 3)
            .unwrap();
        let first_closure = closure(&first_result, &first, journal.session(), &first_payloads);
        let mut renderer = Renderer::default();
        journal
            .stage_result(&first, &first_closure, &mut renderer, NOW + 3)
            .unwrap();
        journal
            .commit_and_activate(&first, &fence, &mut renderer, NOW + 4)
            .unwrap();

        // This result and its parent both have new_output_height=1. Restore
        // must follow base-head causality rather than sorting by height or ID.
        let terminal = close_request(&journal, &key, "aaa-terminal", FinishReasonV2::Cancelled);
        journal.reserve(&terminal, &fence, NOW + 4).unwrap();
        let terminal_result = build_result(
            &terminal,
            journal.session(),
            ResultDecisionV2::Closed {
                commit_frontier: false,
                accepted_drafts: 0,
                reason: FinishReasonV2::Cancelled,
            },
            SamplerStateV2::stateless(),
            Vec::new(),
            NOW + 5,
            &signer,
        )
        .unwrap();
        journal
            .record_prepared(&terminal, terminal_result.clone(), &fence, NOW + 5)
            .unwrap();
        let terminal_closure =
            FragmentAssemblerV2::new(&terminal_result, &terminal, journal.session())
                .unwrap()
                .finish()
                .unwrap();
        journal
            .stage_result(&terminal, &terminal_closure, &mut renderer, NOW + 5)
            .unwrap();
        journal
            .commit_and_activate(&terminal, &fence, &mut renderer, NOW + 6)
            .unwrap();
        drop(journal);

        let restored = DurableVerifierJournalV2::open(&root, key, "request-key-a", 3).unwrap();
        assert_eq!(restored.output_height, 1);
        assert_eq!(restored.evaluated_tokens, vec![1, 2, 3, 101]);
        assert_eq!(
            restored.frontier,
            FrontierV2::Closed {
                reason: FinishReasonV2::Cancelled
            }
        );
    }

    #[test]
    fn tombstone_resumes_cleanup_after_a_gc_crash() {
        let key = MacKey::from_bytes([0x37; 32]);
        let signer = TargetSigningKeyV2::from_seed([0x47; 32]).unwrap();
        let directory = TempDir::new().unwrap();
        let root = directory.path().join("journal");
        let fence = Fence {
            live: true,
            term: 7,
        };
        let mut journal = create_journal(&directory, &key, &signer, CouplingPolicyV2::Greedy);
        let round = request(&journal, &key, "request-gc-crash", Vec::new());
        journal.reserve(&round, &fence, NOW + 2).unwrap();
        let (result, payloads) = open_result(&round, journal.session(), &signer, 0, 777);
        journal
            .record_prepared(&round, result.clone(), &fence, NOW + 3)
            .unwrap();
        let verified = closure(&result, &round, journal.session(), &payloads);
        let mut renderer = Renderer::default();
        journal
            .stage_result(&round, &verified, &mut renderer, NOW + 3)
            .unwrap();
        journal
            .commit_and_activate(&round, &fence, &mut renderer, NOW + 4)
            .unwrap();
        journal
            .acknowledge(
                &round.core.request_id,
                &result.core.new_head_sha256,
                NOW + 5,
            )
            .unwrap();

        // Simulate the GC process dying immediately after its durable
        // tombstone and one unlink. PREPARED/STAGED and the fragment remain.
        let tombstone = TombstoneRecordV2 {
            request_id: round.core.request_id.clone(),
            result_sha256: result.record_digest().unwrap(),
            retired_unix_ms: NOW + 20_000,
        };
        journal
            .write_slot(
                "tombstones",
                &round.core.request_id,
                "tombstone",
                &tombstone,
            )
            .unwrap();
        fs::remove_file(journal.slot_path("reservations", &round.core.request_id)).unwrap();
        drop(journal);

        let mut restored = DurableVerifierJournalV2::open(&root, key, "request-key-a", 3).unwrap();
        assert_eq!(
            restored.reserve(&round, &fence, NOW + 20_001),
            Err(VerifierV2Error::ResyncRequired)
        );
        assert_eq!(
            restored
                .garbage_collect(&round.core.request_id, NOW + 20_001, 1)
                .unwrap(),
            1
        );
        drop(restored);
        let restored = DurableVerifierJournalV2::open(
            &root,
            MacKey::from_bytes([0x37; 32]),
            "request-key-a",
            3,
        )
        .unwrap();
        assert_eq!(restored.output_height, 1);
    }

    #[test]
    fn lease_context_abi_and_fragment_tampering_fail_closed() {
        let key = MacKey::from_bytes([0x35; 32]);
        let signer = TargetSigningKeyV2::from_seed([0x45; 32]).unwrap();
        let directory = TempDir::new().unwrap();
        let mut journal = create_journal(
            &directory,
            &key,
            &signer,
            CouplingPolicyV2::SparseMaximal { max_support: 40 },
        );
        let request = request(&journal, &key, "request-a", vec![102]);
        assert_eq!(
            journal.reserve(
                &request,
                &Fence {
                    live: false,
                    term: 7
                },
                NOW + 2
            ),
            Err(VerifierV2Error::LeaseNotLive)
        );
        assert_eq!(
            journal.reserve(
                &request,
                &Fence {
                    live: true,
                    term: 8
                },
                NOW + 2
            ),
            Err(VerifierV2Error::LeaseNotLive)
        );

        let (mut descriptors, _) = fragments(0, 2);
        descriptors[0].payload_abi = "wrong-abi".into();
        assert!(build_result(
            &request,
            journal.session(),
            ResultDecisionV2::Open {
                accepted_drafts: 1,
                frontier_out: 900,
            },
            request.core.post_draft_sampler_state.clone(),
            descriptors,
            NOW + 3,
            &signer,
        )
        .is_err());

        let (result, payloads) = open_result(&request, journal.session(), &signer, 1, 900);
        let mut assembler = FragmentAssemblerV2::new(&result, &request, journal.session()).unwrap();
        assert!(assembler.insert(0, b"bad!".to_vec()).is_err());
        assert_eq!(assembler.missing_ordinals(), vec![0, 1]);
        assert_eq!(payloads.len(), 2);

        let tiny_session = session(
            &key,
            &signer,
            CouplingPolicyV2::SparseMaximal { max_support: 40 },
            3,
        );
        let mut core = request.core.clone();
        core.session_genesis_sha256 = tiny_session.core.genesis_head_sha256.clone();
        core.base_head_sha256 = tiny_session.core.genesis_head_sha256.clone();
        assert!(AuthenticatedRoundV2::sign(core, &tiny_session.core, &key).is_err());
    }
}
