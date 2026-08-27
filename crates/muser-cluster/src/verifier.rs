//! Authenticated, policy-neutral round log for a remote speculative verifier.
//!
//! This module is deliberately separate from the qualified one-shot Handoff
//! V2 protocol.  It encodes the distributed-systems boundary discovered in
//! the NVFP4 speculative investigation:
//!
//! - the committed token/frontier state is ordered and single-writer;
//! - target and draft KV are checkpoint-native derived renders;
//! - immutable evidence/capture/KV chunks may arrive duplicated or out of
//!   order, but are installable only after an authenticated closure;
//! - callers can validate a prepared result and complete object closure before
//!   applying it to the in-memory [`VerifierRoundLogV1`].
//!
//! The protocol supports both sparse maximal coupling and communication-free
//! shared-Gumbel coupling.  It does not declare either policy qualified.  Its
//! HMAC threat model is trusted peers plus corruption/replay protection: a Mac
//! that knows the symmetric session key can forge a GX result.  A deployment
//! requiring proposer/target role separation must replace result HMACs with a
//! target-only signature or a trusted gateway.  Likewise, the signed authority
//! term is an identity fence, not a lease service; a serving integration must
//! check the bound authority/lease against a live external registry on every
//! new admission.
//!
//! This is a protocol-shape prototype, not a complete serving state machine. It
//! does not yet carry a reconstructible token transcript or sampler snapshot,
//! reserve in-flight requests durably, encode terminal/EOS state, bind fragment
//! payload ABI, or atomically couple head CAS with renderer activation. It
//! machine-checks network/object closure, not filesystem durability or renderer
//! installation. No serving route uses this experimental module today.

use std::collections::BTreeMap;

use kvpack_handoff::{canonical_json, sha256_hex, MacKey};
use serde::{Deserialize, Serialize};

pub const VERIFIER_SESSION_PROTOCOL_V1: &str = "muser-verifier-session-v1";
pub const VERIFIER_ROUND_PROTOCOL_V1: &str = "muser-verifier-round-v1";
pub const VERIFIER_RESULT_PROTOCOL_V1: &str = "muser-verifier-result-v1";
pub const VERIFIER_SESSION_MAC_DOMAIN_V1: &[u8] = b"muser-verifier-session-v1";
pub const VERIFIER_ROUND_MAC_DOMAIN_V1: &[u8] = b"muser-verifier-round-v1";
pub const VERIFIER_RESULT_MAC_DOMAIN_V1: &[u8] = b"muser-verifier-result-v1";
pub const MAX_VERIFIER_DRAFTS_V1: usize = 64;
pub const MAX_VERIFIER_FRAGMENTS_V1: usize = 4_096;
pub const MAX_VERIFIER_FRAGMENT_BYTES_V1: u64 = 512 * 1024 * 1024;
pub const MAX_VERIFIER_TOTAL_FRAGMENT_BYTES_V1: u64 = 512 * 1024 * 1024;
pub const MAX_VERIFIER_CLOCK_SKEW_MS_V1: u64 = 30_000;
pub const MAX_VERIFIER_FRAGMENT_REQUIREMENTS_V1: usize = 64;
pub const MAX_VERIFIER_SPARSE_SUPPORT_V1: u32 = 4_096;
pub const MAX_VERIFIER_DRAFT_EVIDENCE_BYTES_V1: usize = 4 * 1024 * 1024;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum VerifierProtocolError {
    #[error("verifier validation: {0}")]
    Validation(String),
    #[error("verifier authentication failed")]
    Authentication,
    #[error("request id was reused with different intent")]
    ChangedIntent,
    #[error("round is based on a stale verifier head")]
    StaleHead,
    #[error("result closure is incomplete")]
    Incomplete,
}

type Result<T> = std::result::Result<T, VerifierProtocolError>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct VerifierCompositeIdentityV1 {
    /// User-visible product identity.  This must change when RedHat replaces
    /// Dudeman as the authoritative continuation model.
    pub semantic_product_sha256: String,
    pub prefix_checkpoint_sha256: String,
    pub target_checkpoint_sha256: String,
    pub target_engine_sha256: String,
    /// Includes the finite sampler implementation, float/PRF contract, and
    /// processor order; a heterogeneous failover cannot silently change it.
    pub target_sampler_sha256: String,
    pub draft_checkpoint_sha256: String,
    pub draft_engine_sha256: String,
    pub tokenizer_sha256: String,
    pub vocabulary_sha256: String,
    pub target_genesis_root_sha256: String,
    /// The Mac draft genesis is distinct even when both sides began with the
    /// same portable RedHat prefix bytes.
    pub draft_genesis_root_sha256: String,
    pub portable_kv_abi: String,
}

impl VerifierCompositeIdentityV1 {
    fn validate(&self) -> Result<()> {
        for (name, value) in [
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
            validate_digest(name, value)?;
        }
        validate_identifier("portable KV ABI", &self.portable_kv_abi, 128)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct VerifierSessionCoreV1 {
    pub protocol: String,
    pub session_id: String,
    pub client_incarnation: String,
    /// These values are issued and fenced by an external lease registry.  The
    /// pure transcript layer binds them but cannot itself determine liveness.
    pub authority_id: String,
    pub authority_lease_id: String,
    pub authority_term: u64,
    pub created_unix_ms: u64,
    pub expires_unix_ms: u64,
    pub hmac_key_id: String,
    pub hmac_key_epoch: u64,
    pub identity: VerifierCompositeIdentityV1,
    pub sampler_seed: [u8; 32],
    pub coupling_policy: VerifierCouplingPolicyV1,
    /// Temperature/top-k/top-p/penalties/grammar, processor order, and all
    /// policy-specific RNG rules are committed by this digest.
    pub sampler_config_sha256: String,
    /// Digest of the initial MT snapshot for sparse-maximal, or of the
    /// canonical stateless sentinel for greedy/shared-Gumbel.
    pub initial_sampler_state_sha256: String,
    pub vocab_size: u32,
    pub max_context_tokens: u64,
    pub max_total_fragment_bytes: u64,
    pub max_drafts: u32,
    /// Zero outside sparse-maximal sessions; otherwise every p/q row is
    /// rejected if its positive support exceeds this authenticated cap.
    pub max_sparse_support: u32,
    /// Model position immediately after the initial committed/evaluated cut.
    pub initial_model_position: u64,
    /// Number of generated tokens already evaluated, committed, and emitted.
    pub initial_output_height: u64,
    /// Absolute generated-output ordinal of the carried, unprocessed frontier.
    pub initial_frontier_ordinal: u64,
    pub initial_head_sha256: String,
    pub initial_frontier: u32,
    pub fragment_requirements: Vec<VerifierFragmentRequirementV1>,
}

impl VerifierSessionCoreV1 {
    fn validate_historical(&self, expected_key_id: &str, minimum_epoch: u64) -> Result<()> {
        if self.protocol != VERIFIER_SESSION_PROTOCOL_V1 {
            return validation("wrong verifier session protocol");
        }
        validate_identifier("session id", &self.session_id, 128)?;
        validate_identifier("client incarnation", &self.client_incarnation, 128)?;
        validate_identifier("authority id", &self.authority_id, 128)?;
        validate_identifier("authority lease id", &self.authority_lease_id, 128)?;
        validate_identifier("HMAC key id", &self.hmac_key_id, 128)?;
        if self.hmac_key_id != expected_key_id || self.hmac_key_epoch < minimum_epoch {
            return validation("wrong verifier HMAC key id or stale epoch");
        }
        if self.authority_term == 0 || self.created_unix_ms >= self.expires_unix_ms {
            return validation("invalid verifier authority term or lifetime");
        }
        if self.vocab_size == 0
            || self.max_context_tokens == 0
            || self.initial_model_position > self.max_context_tokens
            || self.max_total_fragment_bytes == 0
            || self.max_total_fragment_bytes > MAX_VERIFIER_TOTAL_FRAGMENT_BYTES_V1
            || self.max_drafts == 0
            || self.max_drafts as usize > MAX_VERIFIER_DRAFTS_V1
            || self.initial_frontier >= self.vocab_size
            || self.initial_frontier_ordinal != self.initial_output_height
        {
            return validation("invalid verifier geometry or frontier ordinal");
        }
        match self.coupling_policy {
            VerifierCouplingPolicyV1::SparseMaximal
                if self.max_sparse_support == 0
                    || self.max_sparse_support > MAX_VERIFIER_SPARSE_SUPPORT_V1 =>
            {
                return validation("sparse-maximal support cap is invalid");
            }
            VerifierCouplingPolicyV1::SparseMaximal => {}
            _ if self.max_sparse_support != 0 => {
                return validation("non-sparse verifier session has a sparse support cap");
            }
            _ => {}
        }
        validate_digest("initial head", &self.initial_head_sha256)?;
        validate_digest("initial sampler state", &self.initial_sampler_state_sha256)?;
        validate_digest("sampler config", &self.sampler_config_sha256)?;
        validate_fragment_requirements(&self.fragment_requirements)?;
        self.identity.validate()
    }

    fn validate(&self, now_unix_ms: u64, expected_key_id: &str, minimum_epoch: u64) -> Result<()> {
        self.validate_historical(expected_key_id, minimum_epoch)?;
        if now_unix_ms > self.expires_unix_ms
            || self.created_unix_ms > now_unix_ms.saturating_add(MAX_VERIFIER_CLOCK_SKEW_MS_V1)
        {
            return validation("verifier session is expired or created too far in the future");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AuthenticatedVerifierSessionV1 {
    pub core: VerifierSessionCoreV1,
    pub hmac_sha256: String,
}

impl AuthenticatedVerifierSessionV1 {
    pub fn sign(core: VerifierSessionCoreV1, key: &MacKey) -> Result<Self> {
        core.validate_historical(&core.hmac_key_id, core.hmac_key_epoch)?;
        let canonical = canonical(&core)?;
        let hmac_sha256 = key
            .tag_domain_hex(VERIFIER_SESSION_MAC_DOMAIN_V1, &canonical)
            .map_err(|_| VerifierProtocolError::Authentication)?;
        Ok(Self { core, hmac_sha256 })
    }

    pub fn verify(
        &self,
        key: &MacKey,
        now_unix_ms: u64,
        expected_key_id: &str,
        minimum_epoch: u64,
    ) -> Result<()> {
        self.core
            .validate(now_unix_ms, expected_key_id, minimum_epoch)?;
        validate_digest("session HMAC", &self.hmac_sha256)?;
        key.verify_domain_hex(
            VERIFIER_SESSION_MAC_DOMAIN_V1,
            &canonical(&self.core)?,
            &self.hmac_sha256,
        )
        .map_err(|_| VerifierProtocolError::Authentication)
    }

    /// Verify a durable session genesis for restore/audit without treating its
    /// serving expiry as a retention deadline.
    pub fn verify_historical(
        &self,
        key: &MacKey,
        expected_key_id: &str,
        minimum_epoch: u64,
    ) -> Result<()> {
        self.core
            .validate_historical(expected_key_id, minimum_epoch)?;
        validate_digest("session HMAC", &self.hmac_sha256)?;
        key.verify_domain_hex(
            VERIFIER_SESSION_MAC_DOMAIN_V1,
            &canonical(&self.core)?,
            &self.hmac_sha256,
        )
        .map_err(|_| VerifierProtocolError::Authentication)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VerifierCouplingPolicyV1 {
    GreedyEquality,
    SparseMaximal,
    SharedGumbel,
}

/// Exact wire representation of one normalized sparse q row. Probabilities
/// travel as IEEE-754 bits so JSON number formatting cannot alter acceptance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SparseQRowWireV1 {
    pub vocab_size: u32,
    /// Source sampler order is retained; fixed-seed residual selection can be
    /// observably order-sensitive even when token probabilities are equal.
    pub entries: Vec<SparseQEntryWireV1>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SparseQEntryWireV1 {
    pub token: u32,
    pub probability_f32_bits: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum VerifierDraftEvidenceV1 {
    None,
    SparseMaximal {
        q_rows: Vec<SparseQRowWireV1>,
    },
    SharedGumbel {
        draft_sampler_witness_sha256: String,
    },
}

impl SparseQRowWireV1 {
    fn validate(&self, vocab_size: u32, max_support: u32) -> Result<()> {
        if self.vocab_size != vocab_size
            || self.entries.is_empty()
            || self.entries.len() > max_support as usize
        {
            return validation("sparse q row has invalid vocabulary or support geometry");
        }
        let mut seen = std::collections::BTreeSet::new();
        let mut total = 0.0f64;
        for entry in &self.entries {
            let probability = f32::from_bits(entry.probability_f32_bits);
            if entry.token >= vocab_size
                || !seen.insert(entry.token)
                || !probability.is_finite()
                || probability <= 0.0
            {
                return validation("sparse q row contains an invalid entry");
            }
            total += probability as f64;
        }
        if !total.is_finite() || (total - 1.0).abs() > 1e-6 {
            return validation("sparse q row is not canonically normalized");
        }
        Ok(())
    }

    fn contains(&self, token: u32) -> bool {
        self.entries.iter().any(|entry| entry.token == token)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct VerifyRoundCoreV1 {
    pub protocol: String,
    pub session_id: String,
    pub authority_term: u64,
    pub request_id: String,
    pub created_unix_ms: u64,
    pub expires_unix_ms: u64,
    pub hmac_key_id: String,
    pub hmac_key_epoch: u64,
    pub base_output_height: u64,
    pub base_head_sha256: String,
    pub frontier_ordinal: u64,
    pub frontier_in: u32,
    pub draft_tokens: Vec<u32>,
    pub coupling_policy: VerifierCouplingPolicyV1,
    /// Digest of the inline evidence below. A serving sparse-MT protocol must
    /// additionally carry or close over the actual canonical sampler snapshot;
    /// this skeleton does not make a digest invertible.
    pub draft_evidence_sha256: String,
    pub draft_evidence: VerifierDraftEvidenceV1,
    pub base_sampler_state_sha256: String,
    /// Digest of the state after Mac produced the declared drafts. A serving
    /// integration must verify this transition from an actual authenticated
    /// snapshot and draw schedule, not from these digest bytes alone.
    pub post_draft_sampler_state_sha256: String,
}

impl VerifyRoundCoreV1 {
    fn validate_historical(&self, session: &VerifierSessionCoreV1) -> Result<()> {
        if self.protocol != VERIFIER_ROUND_PROTOCOL_V1
            || self.session_id != session.session_id
            || self.authority_term != session.authority_term
            || self.coupling_policy != session.coupling_policy
        {
            return validation("round addresses the wrong session, authority term, or policy");
        }
        validate_identifier("request id", &self.request_id, 128)?;
        validate_identifier("round HMAC key id", &self.hmac_key_id, 128)?;
        if self.hmac_key_id != session.hmac_key_id
            || self.hmac_key_epoch != session.hmac_key_epoch
            || self.created_unix_ms < session.created_unix_ms
            || self.created_unix_ms >= self.expires_unix_ms
            || self.expires_unix_ms > session.expires_unix_ms
        {
            return validation("round has invalid key identity or lifetime");
        }
        if self.draft_tokens.len() > session.max_drafts as usize
            || self.draft_tokens.len() > MAX_VERIFIER_DRAFTS_V1
            || self.base_output_height < session.initial_output_height
            || self.frontier_in >= session.vocab_size
            || self
                .draft_tokens
                .iter()
                .any(|token| *token >= session.vocab_size)
        {
            return validation("round candidates exceed session geometry");
        }
        let generated_since_genesis = self
            .base_output_height
            .checked_sub(session.initial_output_height)
            .ok_or_else(|| {
                VerifierProtocolError::Validation("round precedes session genesis".into())
            })?;
        let evaluated_candidate_count = u64::try_from(self.draft_tokens.len() + 1)
            .map_err(|_| VerifierProtocolError::Validation("candidate count overflow".into()))?;
        let requested_model_end = session
            .initial_model_position
            .checked_add(generated_since_genesis)
            .and_then(|position| position.checked_add(evaluated_candidate_count))
            .ok_or_else(|| VerifierProtocolError::Validation("model position overflow".into()))?;
        if requested_model_end > session.max_context_tokens {
            return validation("round candidates exceed the authenticated context limit");
        }
        validate_digest("base head", &self.base_head_sha256)?;
        validate_digest("draft evidence", &self.draft_evidence_sha256)?;
        let evidence_bytes = canonical(&self.draft_evidence)?;
        if evidence_bytes.len() > MAX_VERIFIER_DRAFT_EVIDENCE_BYTES_V1
            || sha256_hex(&evidence_bytes) != self.draft_evidence_sha256
        {
            return validation("inline draft evidence exceeds its cap or differs from its digest");
        }
        match (&self.coupling_policy, &self.draft_evidence) {
            (VerifierCouplingPolicyV1::GreedyEquality, VerifierDraftEvidenceV1::None) => {}
            (
                VerifierCouplingPolicyV1::SharedGumbel,
                VerifierDraftEvidenceV1::SharedGumbel {
                    draft_sampler_witness_sha256,
                },
            ) => validate_digest("draft sampler witness", draft_sampler_witness_sha256)?,
            (
                VerifierCouplingPolicyV1::SparseMaximal,
                VerifierDraftEvidenceV1::SparseMaximal { q_rows },
            ) => {
                if q_rows.len() != self.draft_tokens.len() {
                    return validation("sparse q row count differs from draft count");
                }
                for (row, token) in q_rows.iter().zip(&self.draft_tokens) {
                    row.validate(session.vocab_size, session.max_sparse_support)?;
                    if !row.contains(*token) {
                        return validation("draft token is absent from its declared q support");
                    }
                }
            }
            _ => return validation("draft evidence does not match the fixed session policy"),
        }
        validate_digest("base sampler state", &self.base_sampler_state_sha256)?;
        validate_digest(
            "post-draft sampler state",
            &self.post_draft_sampler_state_sha256,
        )
    }

    fn validate(&self, session: &VerifierSessionCoreV1, now_unix_ms: u64) -> Result<()> {
        self.validate_historical(session)?;
        if now_unix_ms > self.expires_unix_ms
            || self.created_unix_ms > now_unix_ms.saturating_add(MAX_VERIFIER_CLOCK_SKEW_MS_V1)
        {
            return validation("round is expired or created too far in the future");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AuthenticatedVerifyRoundV1 {
    pub core: VerifyRoundCoreV1,
    pub hmac_sha256: String,
}

impl AuthenticatedVerifyRoundV1 {
    pub fn sign(
        core: VerifyRoundCoreV1,
        session: &VerifierSessionCoreV1,
        key: &MacKey,
    ) -> Result<Self> {
        core.validate_historical(session)?;
        let hmac_sha256 = key
            .tag_domain_hex(VERIFIER_ROUND_MAC_DOMAIN_V1, &canonical(&core)?)
            .map_err(|_| VerifierProtocolError::Authentication)?;
        Ok(Self { core, hmac_sha256 })
    }

    pub fn verify(
        &self,
        session: &VerifierSessionCoreV1,
        key: &MacKey,
        now_unix_ms: u64,
    ) -> Result<()> {
        self.core.validate(session, now_unix_ms)?;
        validate_digest("round HMAC", &self.hmac_sha256)?;
        key.verify_domain_hex(
            VERIFIER_ROUND_MAC_DOMAIN_V1,
            &canonical(&self.core)?,
            &self.hmac_sha256,
        )
        .map_err(|_| VerifierProtocolError::Authentication)
    }

    /// Verify a durable historical request without treating its admission
    /// deadline as a replay deadline.  This is used only to return or restore
    /// an already-journaled completion; new work must call [`Self::verify`].
    pub fn verify_historical(&self, session: &VerifierSessionCoreV1, key: &MacKey) -> Result<()> {
        self.core.validate_historical(session)?;
        validate_digest("round HMAC", &self.hmac_sha256)?;
        key.verify_domain_hex(
            VERIFIER_ROUND_MAC_DOMAIN_V1,
            &canonical(&self.core)?,
            &self.hmac_sha256,
        )
        .map_err(|_| VerifierProtocolError::Authentication)
    }

    pub fn intent_sha256(&self) -> Result<String> {
        Ok(sha256_hex(&canonical(&self.core)?))
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum VerifierFragmentKindV1 {
    TargetHidden,
    TargetKvDelta,
    TargetProbabilityEvidence,
    SamplerWitness,
    Snapshot,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VerifierFragmentCoverageV1 {
    /// Exactly one logical row for every committed/evaluated input.
    CommittedInputs,
    /// Exactly one record rooted at the round's base output height.
    RoundSingleton,
    /// The D+1 fresh target rows after evaluating `frontier_in` and then each
    /// of D drafts; the last row is the all-accepted bonus when every draft
    /// matches. The T0 witness that selected `frontier_in` belongs to the
    /// authenticated parent and is not materialized again on a cache hit.
    FreshDraftDistributionsPlusBonus,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct VerifierFragmentRequirementV1 {
    pub component_id: String,
    pub kind: VerifierFragmentKindV1,
    pub coverage: VerifierFragmentCoverageV1,
    pub required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct VerifierFragmentDescriptorV1 {
    pub ordinal: u32,
    pub component_id: String,
    pub kind: VerifierFragmentKindV1,
    pub logical_start: u64,
    pub logical_count: u64,
    pub byte_len: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct VerifyResultCoreV1 {
    pub protocol: String,
    pub session_id: String,
    pub authority_term: u64,
    pub request_id: String,
    pub request_intent_sha256: String,
    pub base_output_height: u64,
    pub base_head_sha256: String,
    pub base_sampler_state_sha256: String,
    pub accepted_drafts: u32,
    /// Evaluated inputs committed this round: carried frontier plus accepted
    /// drafts.  `frontier_out` is explicitly excluded.
    pub commit_input_count: u32,
    pub committed_tokens: Vec<u32>,
    pub frontier_out: u32,
    pub frontier_out_ordinal: u64,
    pub new_output_height: u64,
    pub new_head_sha256: String,
    pub post_verify_sampler_state_sha256: String,
    pub fragments_sha256: String,
    pub fragments: Vec<VerifierFragmentDescriptorV1>,
    pub completed_unix_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AuthenticatedVerifyResultV1 {
    pub core: VerifyResultCoreV1,
    pub hmac_sha256: String,
}

impl AuthenticatedVerifyResultV1 {
    fn sign(core: VerifyResultCoreV1, key: &MacKey) -> Result<Self> {
        let hmac_sha256 = key
            .tag_domain_hex(VERIFIER_RESULT_MAC_DOMAIN_V1, &canonical(&core)?)
            .map_err(|_| VerifierProtocolError::Authentication)?;
        Ok(Self { core, hmac_sha256 })
    }

    pub fn verify_tag(&self, key: &MacKey) -> Result<()> {
        validate_digest("result HMAC", &self.hmac_sha256)?;
        key.verify_domain_hex(
            VERIFIER_RESULT_MAC_DOMAIN_V1,
            &canonical(&self.core)?,
            &self.hmac_sha256,
        )
        .map_err(|_| VerifierProtocolError::Authentication)
    }

    pub fn verify_against(
        &self,
        request: &AuthenticatedVerifyRoundV1,
        session: &VerifierSessionCoreV1,
        key: &MacKey,
    ) -> Result<()> {
        self.verify_tag(key)?;
        let core = &self.core;
        let round = &request.core;
        if core.protocol != VERIFIER_RESULT_PROTOCOL_V1
            || core.session_id != round.session_id
            || core.session_id != session.session_id
            || core.authority_term != round.authority_term
            || core.request_id != round.request_id
            || core.request_intent_sha256 != request.intent_sha256()?
            || core.base_output_height != round.base_output_height
            || core.base_head_sha256 != round.base_head_sha256
            || core.base_sampler_state_sha256 != round.base_sampler_state_sha256
            || core.accepted_drafts as usize > round.draft_tokens.len()
            || core.commit_input_count != core.accepted_drafts + 1
            || core.committed_tokens.len() != core.commit_input_count as usize
            || core.committed_tokens.first() != Some(&round.frontier_in)
            || core.committed_tokens[1..] != round.draft_tokens[..core.accepted_drafts as usize]
            || core.frontier_out >= session.vocab_size
            || core.frontier_out_ordinal != core.new_output_height
            || core.base_output_height < session.initial_output_height
            || core.completed_unix_ms < round.created_unix_ms
            || core.completed_unix_ms > round.expires_unix_ms
            || core.completed_unix_ms > session.expires_unix_ms
            || core.new_output_height
                != core
                    .base_output_height
                    .checked_add(core.commit_input_count as u64)
                    .ok_or_else(|| {
                        VerifierProtocolError::Validation("output height overflow".into())
                    })?
        {
            return validation("result does not match its authenticated round");
        }
        validate_digest("new head", &core.new_head_sha256)?;
        validate_digest(
            "post-verify sampler state",
            &core.post_verify_sampler_state_sha256,
        )?;
        validate_digest("fragment set", &core.fragments_sha256)?;
        validate_fragments(&core.fragments)?;
        validate_fragment_closure(core, round, session)?;
        if fragment_set_sha256(&core.fragments)? != core.fragments_sha256 {
            return validation("result fragment commitment differs");
        }
        let generated_since_genesis = core
            .new_output_height
            .checked_sub(session.initial_output_height)
            .ok_or_else(|| {
                VerifierProtocolError::Validation("result precedes session genesis".into())
            })?;
        let new_model_position = session
            .initial_model_position
            .checked_add(generated_since_genesis)
            .ok_or_else(|| VerifierProtocolError::Validation("model position overflow".into()))?;
        if new_model_position > session.max_context_tokens {
            return validation("result exceeds the authenticated context limit");
        }
        let expected_head = transition_head_sha256(
            &core.session_id,
            core.authority_term,
            &core.base_head_sha256,
            core.base_output_height,
            &core.committed_tokens,
            core.frontier_out,
            core.frontier_out_ordinal,
            round.coupling_policy,
            &core.request_intent_sha256,
            &core.post_verify_sampler_state_sha256,
            &core.fragments_sha256,
        )?;
        if expected_head != core.new_head_sha256 {
            return validation("result head transition differs");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyRoundAdmissionV1 {
    New {
        intent_sha256: String,
    },
    Replay(Box<AuthenticatedVerifyResultV1>),
    Stale {
        output_height: u64,
        head_sha256: String,
        frontier_ordinal: u64,
        frontier: u32,
        sampler_state_sha256: String,
    },
}

#[derive(Debug, Clone)]
struct CompletionV1 {
    intent_sha256: String,
    result: AuthenticatedVerifyResultV1,
}

/// In-memory projection intended to sit behind a future durable round journal.
///
/// This type stores completed results only. It does not implement the durable
/// PREPARED reservation needed to suppress recomputation between prepare and
/// apply, nor the atomic staged-render activation required at the head CAS.
pub struct VerifierRoundLogV1 {
    session: VerifierSessionCoreV1,
    live_admission: bool,
    output_height: u64,
    head_sha256: String,
    frontier_ordinal: u64,
    frontier: u32,
    sampler_state_sha256: String,
    completions: BTreeMap<String, CompletionV1>,
}

impl VerifierRoundLogV1 {
    pub fn open(
        session: &AuthenticatedVerifierSessionV1,
        key: &MacKey,
        now_unix_ms: u64,
        expected_key_id: &str,
        minimum_epoch: u64,
    ) -> Result<Self> {
        session.verify(key, now_unix_ms, expected_key_id, minimum_epoch)?;
        Ok(Self::genesis(session, true))
    }

    /// Open an expired session only for durable-chain reconstruction, audit,
    /// and exact completion replay. New verifier work remains fail-closed.
    pub fn open_historical(
        session: &AuthenticatedVerifierSessionV1,
        key: &MacKey,
        expected_key_id: &str,
        minimum_epoch: u64,
    ) -> Result<Self> {
        session.verify_historical(key, expected_key_id, minimum_epoch)?;
        Ok(Self::genesis(session, false))
    }

    fn genesis(session: &AuthenticatedVerifierSessionV1, live_admission: bool) -> Self {
        Self {
            session: session.core.clone(),
            live_admission,
            output_height: session.core.initial_output_height,
            head_sha256: session.core.initial_head_sha256.clone(),
            frontier_ordinal: session.core.initial_frontier_ordinal,
            frontier: session.core.initial_frontier,
            sampler_state_sha256: session.core.initial_sampler_state_sha256.clone(),
            completions: BTreeMap::new(),
        }
    }

    pub fn session(&self) -> &VerifierSessionCoreV1 {
        &self.session
    }

    pub fn current_cut(&self) -> (u64, &str, u64, u32) {
        (
            self.output_height,
            &self.head_sha256,
            self.frontier_ordinal,
            self.frontier,
        )
    }

    pub fn current_sampler_state_sha256(&self) -> &str {
        &self.sampler_state_sha256
    }

    pub fn admit(
        &self,
        request: &AuthenticatedVerifyRoundV1,
        key: &MacKey,
        now_unix_ms: u64,
    ) -> Result<VerifyRoundAdmissionV1> {
        request.verify_historical(&self.session, key)?;
        let intent = request.intent_sha256()?;
        if let Some(completion) = self.completions.get(&request.core.request_id) {
            if completion.intent_sha256 != intent {
                return Err(VerifierProtocolError::ChangedIntent);
            }
            return Ok(VerifyRoundAdmissionV1::Replay(Box::new(
                completion.result.clone(),
            )));
        }
        if !self.live_admission {
            return validation("historical verifier log cannot admit new work");
        }
        // Expiry prevents new computation, but does not prevent exact replay
        // of a durable completion checked above.
        request.verify(&self.session, key, now_unix_ms)?;
        if request.core.base_output_height != self.output_height
            || request.core.base_head_sha256 != self.head_sha256
            || request.core.frontier_ordinal != self.frontier_ordinal
            || request.core.frontier_in != self.frontier
            || request.core.base_sampler_state_sha256 != self.sampler_state_sha256
        {
            return Ok(VerifyRoundAdmissionV1::Stale {
                output_height: self.output_height,
                head_sha256: self.head_sha256.clone(),
                frontier_ordinal: self.frontier_ordinal,
                frontier: self.frontier,
                sampler_state_sha256: self.sampler_state_sha256.clone(),
            });
        }
        Ok(VerifyRoundAdmissionV1::New {
            intent_sha256: intent,
        })
    }

    /// Construct and authenticate a result without mutating the live head.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_result(
        &self,
        request: &AuthenticatedVerifyRoundV1,
        accepted_drafts: usize,
        frontier_out: u32,
        post_verify_sampler_state_sha256: String,
        fragments: Vec<VerifierFragmentDescriptorV1>,
        completed_unix_ms: u64,
        key: &MacKey,
    ) -> Result<AuthenticatedVerifyResultV1> {
        match self.admit(request, key, completed_unix_ms)? {
            VerifyRoundAdmissionV1::New { .. } => {}
            VerifyRoundAdmissionV1::Replay(result) => return Ok(*result),
            VerifyRoundAdmissionV1::Stale { .. } => return Err(VerifierProtocolError::StaleHead),
        }
        if accepted_drafts > request.core.draft_tokens.len()
            || frontier_out >= self.session.vocab_size
            || completed_unix_ms > request.core.expires_unix_ms
        {
            return validation("invalid verifier decision or completion time");
        }
        validate_digest(
            "post-verify sampler state",
            &post_verify_sampler_state_sha256,
        )?;
        validate_fragments(&fragments)?;
        let accepted_drafts_u32 = u32::try_from(accepted_drafts)
            .map_err(|_| VerifierProtocolError::Validation("accepted count overflow".into()))?;
        let commit_input_count = accepted_drafts_u32 + 1;
        let mut committed_tokens = Vec::with_capacity(commit_input_count as usize);
        committed_tokens.push(request.core.frontier_in);
        committed_tokens.extend_from_slice(&request.core.draft_tokens[..accepted_drafts]);
        let new_output_height = request
            .core
            .base_output_height
            .checked_add(commit_input_count as u64)
            .ok_or_else(|| VerifierProtocolError::Validation("output height overflow".into()))?;
        let request_intent_sha256 = request.intent_sha256()?;
        let fragments_sha256 = fragment_set_sha256(&fragments)?;
        let new_head_sha256 = transition_head_sha256(
            &request.core.session_id,
            request.core.authority_term,
            &request.core.base_head_sha256,
            request.core.base_output_height,
            &committed_tokens,
            frontier_out,
            new_output_height,
            request.core.coupling_policy,
            &request_intent_sha256,
            &post_verify_sampler_state_sha256,
            &fragments_sha256,
        )?;
        let core = VerifyResultCoreV1 {
            protocol: VERIFIER_RESULT_PROTOCOL_V1.into(),
            session_id: request.core.session_id.clone(),
            authority_term: request.core.authority_term,
            request_id: request.core.request_id.clone(),
            request_intent_sha256,
            base_output_height: request.core.base_output_height,
            base_head_sha256: request.core.base_head_sha256.clone(),
            base_sampler_state_sha256: request.core.base_sampler_state_sha256.clone(),
            accepted_drafts: accepted_drafts_u32,
            commit_input_count,
            committed_tokens,
            frontier_out,
            frontier_out_ordinal: new_output_height,
            new_output_height,
            new_head_sha256,
            post_verify_sampler_state_sha256,
            fragments_sha256,
            fragments,
            completed_unix_ms,
        };
        let result = AuthenticatedVerifyResultV1::sign(core, key)?;
        result.verify_against(request, &self.session, key)?;
        Ok(result)
    }

    /// Apply a transcript result after authenticating/assembling every required
    /// immutable fragment. The opaque closure enforces only network/object
    /// closure. A production wrapper must durably reserve and journal the
    /// request, stage renderer changes invisibly, and atomically coordinate
    /// head CAS, render activation, and publication; callers must not mutate a
    /// live renderer before this method's stale-head check.
    pub fn apply_durable_result(
        &mut self,
        request: &AuthenticatedVerifyRoundV1,
        result: AuthenticatedVerifyResultV1,
        closure: &VerifiedResultClosureV1,
        key: &MacKey,
        now_unix_ms: u64,
    ) -> Result<()> {
        request.verify_historical(&self.session, key)?;
        result.verify_against(request, &self.session, key)?;
        closure.verify_result(&result)?;
        if result.core.completed_unix_ms > now_unix_ms.saturating_add(MAX_VERIFIER_CLOCK_SKEW_MS_V1)
        {
            return validation("durable result completion is too far in the future");
        }
        let intent = request.intent_sha256()?;
        if let Some(completion) = self.completions.get(&request.core.request_id) {
            if completion.intent_sha256 != intent || completion.result != result {
                return Err(VerifierProtocolError::ChangedIntent);
            }
            return Ok(());
        }
        if request.core.base_output_height != self.output_height
            || request.core.base_head_sha256 != self.head_sha256
            || request.core.frontier_ordinal != self.frontier_ordinal
            || request.core.frontier_in != self.frontier
            || request.core.base_sampler_state_sha256 != self.sampler_state_sha256
        {
            return Err(VerifierProtocolError::StaleHead);
        }
        self.output_height = result.core.new_output_height;
        self.head_sha256.clone_from(&result.core.new_head_sha256);
        self.frontier_ordinal = result.core.frontier_out_ordinal;
        self.frontier = result.core.frontier_out;
        self.sampler_state_sha256
            .clone_from(&result.core.post_verify_sampler_state_sha256);
        self.completions.insert(
            request.core.request_id.clone(),
            CompletionV1 {
                intent_sha256: intent,
                result,
            },
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FragmentInsertV1 {
    Inserted,
    Duplicate,
}

#[derive(Debug, Clone)]
pub struct VerifiedVerifierFragmentV1 {
    pub descriptor: VerifierFragmentDescriptorV1,
    pub payload: Vec<u8>,
}

/// Opaque proof that every byte named by one authenticated result manifest was
/// received and digest-checked.  Only [`VerifierFragmentAssemblerV1::finish`]
/// can construct it, so advancing the token head cannot accidentally bypass
/// closure verification.
#[derive(Debug)]
pub struct VerifiedResultClosureV1 {
    result_record_sha256: String,
    fragments: Vec<VerifiedVerifierFragmentV1>,
}

impl VerifiedResultClosureV1 {
    pub fn fragments(&self) -> &[VerifiedVerifierFragmentV1] {
        &self.fragments
    }

    fn verify_result(&self, result: &AuthenticatedVerifyResultV1) -> Result<()> {
        if self.result_record_sha256 != result_record_sha256(result)?
            || self.fragments.len() != result.core.fragments.len()
            || self
                .fragments
                .iter()
                .zip(&result.core.fragments)
                .any(|(fragment, descriptor)| &fragment.descriptor != descriptor)
        {
            return validation("verified closure belongs to a different result");
        }
        Ok(())
    }
}

/// Sparse, arbitrary-arrival staging for one authenticated result closure.
pub struct VerifierFragmentAssemblerV1 {
    result_record_sha256: String,
    descriptors: Vec<VerifierFragmentDescriptorV1>,
    payloads: Vec<Option<Vec<u8>>>,
    received_bytes: u64,
    max_total_bytes: u64,
}

impl VerifierFragmentAssemblerV1 {
    pub fn new(
        result: &AuthenticatedVerifyResultV1,
        request: &AuthenticatedVerifyRoundV1,
        session: &VerifierSessionCoreV1,
        key: &MacKey,
        max_total_bytes: u64,
    ) -> Result<Self> {
        result.verify_against(request, session, key)?;
        if max_total_bytes == 0 {
            return validation("receiver fragment limit must be positive");
        }
        let declared = result
            .core
            .fragments
            .iter()
            .try_fold(0u64, |sum, descriptor| {
                sum.checked_add(descriptor.byte_len).ok_or_else(|| {
                    VerifierProtocolError::Validation("fragment bytes overflow".into())
                })
            })?;
        let effective_limit = max_total_bytes
            .min(session.max_total_fragment_bytes)
            .min(MAX_VERIFIER_TOTAL_FRAGMENT_BYTES_V1);
        if declared > effective_limit {
            return validation("fragment closure exceeds receiver byte limit");
        }
        Ok(Self {
            result_record_sha256: result_record_sha256(result)?,
            descriptors: result.core.fragments.clone(),
            payloads: vec![None; result.core.fragments.len()],
            received_bytes: 0,
            max_total_bytes: effective_limit,
        })
    }

    pub fn insert(&mut self, ordinal: u32, payload: Vec<u8>) -> Result<FragmentInsertV1> {
        let index = ordinal as usize;
        let descriptor = self.descriptors.get(index).ok_or_else(|| {
            VerifierProtocolError::Validation("undeclared fragment ordinal".into())
        })?;
        if descriptor.ordinal != ordinal
            || payload.len() as u64 != descriptor.byte_len
            || sha256_hex(&payload) != descriptor.sha256
        {
            return validation("fragment length or digest differs from manifest");
        }
        if let Some(existing) = &self.payloads[index] {
            return if existing == &payload {
                Ok(FragmentInsertV1::Duplicate)
            } else {
                validation("duplicate fragment ordinal has different bytes")
            };
        }
        self.received_bytes = self
            .received_bytes
            .checked_add(descriptor.byte_len)
            .ok_or_else(|| VerifierProtocolError::Validation("received bytes overflow".into()))?;
        if self.received_bytes > self.max_total_bytes {
            return validation("received fragments exceed receiver byte limit");
        }
        self.payloads[index] = Some(payload);
        Ok(FragmentInsertV1::Inserted)
    }

    pub fn missing_ordinals(&self) -> Vec<u32> {
        self.payloads
            .iter()
            .enumerate()
            .filter_map(|(index, payload)| payload.is_none().then_some(index as u32))
            .collect()
    }

    pub fn is_complete(&self) -> bool {
        self.payloads.iter().all(Option::is_some)
    }

    pub fn finish(self) -> Result<VerifiedResultClosureV1> {
        if !self.is_complete() {
            return Err(VerifierProtocolError::Incomplete);
        }
        let fragments = self
            .descriptors
            .into_iter()
            .zip(self.payloads)
            .map(|(descriptor, payload)| VerifiedVerifierFragmentV1 {
                descriptor,
                payload: payload.expect("closure was checked complete"),
            })
            .collect();
        Ok(VerifiedResultClosureV1 {
            result_record_sha256: self.result_record_sha256,
            fragments,
        })
    }
}

fn validate_fragments(fragments: &[VerifierFragmentDescriptorV1]) -> Result<()> {
    if fragments.len() > MAX_VERIFIER_FRAGMENTS_V1 {
        return validation("too many verifier fragments");
    }
    let mut ranges: BTreeMap<(String, VerifierFragmentKindV1), Vec<(u64, u64)>> = BTreeMap::new();
    for (index, descriptor) in fragments.iter().enumerate() {
        if descriptor.ordinal as usize != index
            || descriptor.logical_count == 0
            || descriptor.byte_len == 0
            || descriptor.byte_len > MAX_VERIFIER_FRAGMENT_BYTES_V1
        {
            return validation("invalid verifier fragment ordinal or bounds");
        }
        validate_identifier("fragment component id", &descriptor.component_id, 128)?;
        validate_digest("fragment", &descriptor.sha256)?;
        let end = descriptor
            .logical_start
            .checked_add(descriptor.logical_count)
            .ok_or_else(|| VerifierProtocolError::Validation("fragment range overflow".into()))?;
        let key = (descriptor.component_id.clone(), descriptor.kind);
        let component_ranges = ranges.entry(key).or_default();
        if component_ranges
            .iter()
            .any(|(start, prior_end)| descriptor.logical_start < *prior_end && *start < end)
        {
            return validation("overlapping verifier fragment ranges");
        }
        component_ranges.push((descriptor.logical_start, end));
    }
    let total_bytes = fragments.iter().try_fold(0u64, |sum, descriptor| {
        sum.checked_add(descriptor.byte_len)
            .ok_or_else(|| VerifierProtocolError::Validation("fragment bytes overflow".into()))
    })?;
    if total_bytes > MAX_VERIFIER_TOTAL_FRAGMENT_BYTES_V1 {
        return validation("verifier fragment set exceeds the protocol byte limit");
    }
    Ok(())
}

fn validate_fragment_requirements(requirements: &[VerifierFragmentRequirementV1]) -> Result<()> {
    if requirements.is_empty()
        || requirements.len() > MAX_VERIFIER_FRAGMENT_REQUIREMENTS_V1
        || !requirements.iter().any(|requirement| requirement.required)
    {
        return validation("fragment schema must contain a bounded required component");
    }
    let mut seen = std::collections::BTreeSet::new();
    for requirement in requirements {
        validate_identifier(
            "fragment requirement component id",
            &requirement.component_id,
            128,
        )?;
        if !seen.insert((requirement.component_id.clone(), requirement.kind)) {
            return validation("duplicate fragment schema component");
        }
    }
    Ok(())
}

fn validate_fragment_closure(
    result: &VerifyResultCoreV1,
    round: &VerifyRoundCoreV1,
    session: &VerifierSessionCoreV1,
) -> Result<()> {
    let total_bytes = result.fragments.iter().try_fold(0u64, |sum, descriptor| {
        sum.checked_add(descriptor.byte_len)
            .ok_or_else(|| VerifierProtocolError::Validation("fragment bytes overflow".into()))
    })?;
    if total_bytes > session.max_total_fragment_bytes {
        return validation("result exceeds the session fragment byte limit");
    }

    for descriptor in &result.fragments {
        if !session.fragment_requirements.iter().any(|requirement| {
            requirement.component_id == descriptor.component_id
                && requirement.kind == descriptor.kind
        }) {
            return validation("result contains a component outside the session schema");
        }
    }

    for requirement in &session.fragment_requirements {
        let mut ranges = result
            .fragments
            .iter()
            .filter(|descriptor| {
                descriptor.component_id == requirement.component_id
                    && descriptor.kind == requirement.kind
            })
            .map(|descriptor| {
                descriptor
                    .logical_start
                    .checked_add(descriptor.logical_count)
                    .map(|end| (descriptor.logical_start, end))
                    .ok_or_else(|| {
                        VerifierProtocolError::Validation("fragment range overflow".into())
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        if ranges.is_empty() {
            if requirement.required {
                return validation("result omits a required fragment component");
            }
            continue;
        }
        ranges.sort_unstable();
        let (expected_start, expected_count) = match requirement.coverage {
            VerifierFragmentCoverageV1::CommittedInputs => {
                (result.base_output_height, result.commit_input_count as u64)
            }
            VerifierFragmentCoverageV1::RoundSingleton => (result.base_output_height, 1),
            VerifierFragmentCoverageV1::FreshDraftDistributionsPlusBonus => {
                let start = result.base_output_height.checked_add(1).ok_or_else(|| {
                    VerifierProtocolError::Validation("fresh target evidence start overflow".into())
                })?;
                let count = u64::try_from(round.draft_tokens.len() + 1).map_err(|_| {
                    VerifierProtocolError::Validation("candidate evidence count overflow".into())
                })?;
                (start, count)
            }
        };
        let expected_end = expected_start.checked_add(expected_count).ok_or_else(|| {
            VerifierProtocolError::Validation("fragment coverage overflow".into())
        })?;
        let mut cursor = expected_start;
        for (start, end) in ranges {
            if start != cursor || end > expected_end {
                return validation("fragment component coverage is gapped or out of range");
            }
            cursor = end;
        }
        if cursor != expected_end {
            return validation("fragment component coverage is incomplete");
        }
    }
    Ok(())
}

fn fragment_set_sha256(fragments: &[VerifierFragmentDescriptorV1]) -> Result<String> {
    Ok(sha256_hex(&canonical(&fragments)?))
}

fn result_record_sha256(result: &AuthenticatedVerifyResultV1) -> Result<String> {
    Ok(sha256_hex(&canonical(result)?))
}

#[allow(clippy::too_many_arguments)]
fn transition_head_sha256(
    session_id: &str,
    authority_term: u64,
    parent_head_sha256: &str,
    base_output_height: u64,
    committed_tokens: &[u32],
    frontier_out: u32,
    frontier_out_ordinal: u64,
    coupling_policy: VerifierCouplingPolicyV1,
    request_intent_sha256: &str,
    post_verify_sampler_state_sha256: &str,
    fragments_sha256: &str,
) -> Result<String> {
    #[derive(Serialize)]
    struct Transition<'a> {
        domain: &'static str,
        session_id: &'a str,
        authority_term: u64,
        parent_head_sha256: &'a str,
        base_output_height: u64,
        committed_tokens: &'a [u32],
        frontier_out: u32,
        frontier_out_ordinal: u64,
        coupling_policy: VerifierCouplingPolicyV1,
        request_intent_sha256: &'a str,
        post_verify_sampler_state_sha256: &'a str,
        fragments_sha256: &'a str,
    }
    let transition = Transition {
        domain: "muser-verifier-head-v1",
        session_id,
        authority_term,
        parent_head_sha256,
        base_output_height,
        committed_tokens,
        frontier_out,
        frontier_out_ordinal,
        coupling_policy,
        request_intent_sha256,
        post_verify_sampler_state_sha256,
        fragments_sha256,
    };
    Ok(sha256_hex(&canonical(&transition)?))
}

fn canonical<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    canonical_json(value)
        .map_err(|error| VerifierProtocolError::Validation(format!("canonical JSON: {error}")))
}

fn validate_digest(name: &str, value: &str) -> Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return validation(format!("{name} is not lowercase SHA-256"));
    }
    Ok(())
}

fn validate_identifier(name: &str, value: &str, max_len: usize) -> Result<()> {
    if value.is_empty()
        || value.len() > max_len
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"-_.".contains(&byte))
    {
        return validation(format!("invalid {name}"));
    }
    Ok(())
}

fn validation<T>(message: impl Into<String>) -> Result<T> {
    Err(VerifierProtocolError::Validation(message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: u64 = 1_000;

    fn digest(byte: u8) -> String {
        format!("{byte:02x}").repeat(32)
    }

    fn identity() -> VerifierCompositeIdentityV1 {
        VerifierCompositeIdentityV1 {
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
            portable_kv_abi: "muse-kv-v2".into(),
        }
    }

    fn session(key: &MacKey) -> AuthenticatedVerifierSessionV1 {
        AuthenticatedVerifierSessionV1::sign(
            VerifierSessionCoreV1 {
                protocol: VERIFIER_SESSION_PROTOCOL_V1.into(),
                session_id: "session-a".into(),
                client_incarnation: "mac-boot-a".into(),
                authority_id: "gx10-a".into(),
                authority_lease_id: "lease-a".into(),
                authority_term: 1,
                created_unix_ms: NOW,
                expires_unix_ms: NOW + 10_000,
                hmac_key_id: "key-a".into(),
                hmac_key_epoch: 7,
                identity: identity(),
                sampler_seed: [0x42; 32],
                coupling_policy: VerifierCouplingPolicyV1::SparseMaximal,
                sampler_config_sha256: digest(17),
                initial_sampler_state_sha256: digest(14),
                vocab_size: 202_048,
                max_context_tokens: 32_768,
                max_total_fragment_bytes: 16 * 1024 * 1024,
                max_drafts: 15,
                max_sparse_support: 40,
                initial_model_position: 2_048,
                initial_output_height: 0,
                initial_frontier_ordinal: 0,
                initial_head_sha256: digest(12),
                initial_frontier: 101,
                fragment_requirements: vec![VerifierFragmentRequirementV1 {
                    component_id: "target-hidden".into(),
                    kind: VerifierFragmentKindV1::TargetHidden,
                    coverage: VerifierFragmentCoverageV1::CommittedInputs,
                    required: true,
                }],
            },
            key,
        )
        .unwrap()
    }

    fn request(
        log: &VerifierRoundLogV1,
        key: &MacKey,
        request_id: &str,
        drafts: Vec<u32>,
    ) -> AuthenticatedVerifyRoundV1 {
        let (height, head, ordinal, frontier) = log.current_cut();
        let draft_evidence = sparse_evidence(&drafts);
        AuthenticatedVerifyRoundV1::sign(
            VerifyRoundCoreV1 {
                protocol: VERIFIER_ROUND_PROTOCOL_V1.into(),
                session_id: log.session().session_id.clone(),
                authority_term: log.session().authority_term,
                request_id: request_id.into(),
                created_unix_ms: NOW + 1,
                expires_unix_ms: NOW + 1_000,
                hmac_key_id: log.session().hmac_key_id.clone(),
                hmac_key_epoch: log.session().hmac_key_epoch,
                base_output_height: height,
                base_head_sha256: head.into(),
                frontier_ordinal: ordinal,
                frontier_in: frontier,
                draft_tokens: drafts,
                coupling_policy: VerifierCouplingPolicyV1::SparseMaximal,
                draft_evidence_sha256: sha256_hex(&canonical(&draft_evidence).unwrap()),
                draft_evidence,
                base_sampler_state_sha256: log.current_sampler_state_sha256().into(),
                post_draft_sampler_state_sha256: digest(15),
            },
            log.session(),
            key,
        )
        .unwrap()
    }

    fn sparse_evidence(drafts: &[u32]) -> VerifierDraftEvidenceV1 {
        VerifierDraftEvidenceV1::SparseMaximal {
            q_rows: drafts
                .iter()
                .map(|token| SparseQRowWireV1 {
                    vocab_size: 202_048,
                    entries: vec![SparseQEntryWireV1 {
                        token: *token,
                        probability_f32_bits: 1.0f32.to_bits(),
                    }],
                })
                .collect(),
        }
    }

    fn fragments() -> (Vec<VerifierFragmentDescriptorV1>, Vec<Vec<u8>>) {
        let payloads = vec![b"hidden-row-0".to_vec(), b"hidden-row-1".to_vec()];
        let descriptors = payloads
            .iter()
            .enumerate()
            .map(|(index, payload)| VerifierFragmentDescriptorV1 {
                ordinal: index as u32,
                component_id: "target-hidden".into(),
                kind: VerifierFragmentKindV1::TargetHidden,
                logical_start: index as u64,
                logical_count: 1,
                byte_len: payload.len() as u64,
                sha256: sha256_hex(payload),
            })
            .collect();
        (descriptors, payloads)
    }

    fn assembled_closure(
        result: &AuthenticatedVerifyResultV1,
        request: &AuthenticatedVerifyRoundV1,
        session: &VerifierSessionCoreV1,
        key: &MacKey,
        payloads: &[Vec<u8>],
    ) -> VerifiedResultClosureV1 {
        let mut assembler =
            VerifierFragmentAssemblerV1::new(result, request, session, key, 1_024).unwrap();
        for (ordinal, payload) in payloads.iter().enumerate().rev() {
            assembler.insert(ordinal as u32, payload.clone()).unwrap();
        }
        assembler.finish().unwrap()
    }

    #[test]
    fn exact_retry_replays_but_changed_intent_and_stale_parent_fail_closed() {
        let key = MacKey::from_bytes([0x55; 32]);
        let genesis = session(&key);
        let mut log = VerifierRoundLogV1::open(&genesis, &key, NOW, "key-a", 7).unwrap();
        let first = request(&log, &key, "request-1", vec![102, 103, 104]);
        assert!(matches!(
            log.admit(&first, &key, NOW + 2).unwrap(),
            VerifyRoundAdmissionV1::New { .. }
        ));
        let (descriptors, payloads) = fragments();
        let result = log
            .prepare_result(&first, 1, 999, digest(16), descriptors, NOW + 3, &key)
            .unwrap();
        let closure = assembled_closure(&result, &first, log.session(), &key, &payloads);
        // Preparation alone does not advance a correctness-bearing head.
        assert_eq!(log.current_cut().0, 0);
        log.apply_durable_result(&first, result.clone(), &closure, &key, NOW + 4)
            .unwrap();
        assert_eq!(
            log.current_cut(),
            (2, result.core.new_head_sha256.as_str(), 2, 999)
        );
        assert_eq!(
            log.admit(&first, &key, NOW + 5).unwrap(),
            VerifyRoundAdmissionV1::Replay(Box::new(result.clone()))
        );
        // Reapplying the exact durable completion is idempotent.
        log.apply_durable_result(&first, result, &closure, &key, NOW + 5)
            .unwrap();

        let mut changed_core = first.core.clone();
        changed_core.draft_tokens.push(105);
        changed_core.draft_evidence = sparse_evidence(&changed_core.draft_tokens);
        changed_core.draft_evidence_sha256 =
            sha256_hex(&canonical(&changed_core.draft_evidence).unwrap());
        let changed = AuthenticatedVerifyRoundV1::sign(changed_core, log.session(), &key).unwrap();
        assert_eq!(
            log.admit(&changed, &key, NOW + 5),
            Err(VerifierProtocolError::ChangedIntent)
        );

        let stale = request_from_cut(log.session(), &key, "request-stale", 0, &digest(12), 0, 101);
        assert!(matches!(
            log.admit(&stale, &key, NOW + 5).unwrap(),
            VerifyRoundAdmissionV1::Stale {
                output_height: 2,
                ..
            }
        ));
    }

    fn request_from_cut(
        session: &VerifierSessionCoreV1,
        key: &MacKey,
        request_id: &str,
        height: u64,
        head: &str,
        ordinal: u64,
        frontier: u32,
    ) -> AuthenticatedVerifyRoundV1 {
        let drafts = vec![102];
        let draft_evidence = sparse_evidence(&drafts);
        AuthenticatedVerifyRoundV1::sign(
            VerifyRoundCoreV1 {
                protocol: VERIFIER_ROUND_PROTOCOL_V1.into(),
                session_id: session.session_id.clone(),
                authority_term: session.authority_term,
                request_id: request_id.into(),
                created_unix_ms: NOW + 1,
                expires_unix_ms: NOW + 1_000,
                hmac_key_id: session.hmac_key_id.clone(),
                hmac_key_epoch: session.hmac_key_epoch,
                base_output_height: height,
                base_head_sha256: head.into(),
                frontier_ordinal: ordinal,
                frontier_in: frontier,
                draft_tokens: drafts,
                coupling_policy: session.coupling_policy,
                draft_evidence_sha256: sha256_hex(&canonical(&draft_evidence).unwrap()),
                draft_evidence,
                base_sampler_state_sha256: session.initial_sampler_state_sha256.clone(),
                post_draft_sampler_state_sha256: digest(15),
            },
            session,
            key,
        )
        .unwrap()
    }

    #[test]
    fn chunks_reassemble_out_of_order_absorb_duplicates_and_require_closure() {
        let key = MacKey::from_bytes([0x55; 32]);
        let genesis = session(&key);
        let log = VerifierRoundLogV1::open(&genesis, &key, NOW, "key-a", 7).unwrap();
        let round = request(&log, &key, "request-1", vec![102]);
        let (descriptors, payloads) = fragments();
        let result = log
            .prepare_result(&round, 1, 999, digest(16), descriptors, NOW + 3, &key)
            .unwrap();
        let mut assembler =
            VerifierFragmentAssemblerV1::new(&result, &round, log.session(), &key, 1_024).unwrap();
        assert_eq!(assembler.missing_ordinals(), vec![0, 1]);
        assert_eq!(
            assembler.insert(1, payloads[1].clone()).unwrap(),
            FragmentInsertV1::Inserted
        );
        assert_eq!(assembler.missing_ordinals(), vec![0]);
        assert_eq!(
            assembler.insert(1, payloads[1].clone()).unwrap(),
            FragmentInsertV1::Duplicate
        );
        assert!(matches!(
            assembler.finish(),
            Err(VerifierProtocolError::Incomplete)
        ));

        let mut assembler =
            VerifierFragmentAssemblerV1::new(&result, &round, log.session(), &key, 1_024).unwrap();
        assembler.insert(1, payloads[1].clone()).unwrap();
        assembler.insert(0, payloads[0].clone()).unwrap();
        let closure = assembler.finish().unwrap();
        assert_eq!(closure.fragments()[0].payload, payloads[0]);
        assert_eq!(closure.fragments()[1].payload, payloads[1]);
    }

    #[test]
    fn fresh_target_rows_begin_after_the_durable_parent_frontier_witness() {
        let key = MacKey::from_bytes([0x55; 32]);
        let mut core = session(&key).core;
        core.fragment_requirements = vec![VerifierFragmentRequirementV1 {
            component_id: "target-probability".into(),
            kind: VerifierFragmentKindV1::TargetProbabilityEvidence,
            coverage: VerifierFragmentCoverageV1::FreshDraftDistributionsPlusBonus,
            required: true,
        }];
        let genesis = AuthenticatedVerifierSessionV1::sign(core, &key).unwrap();
        let log = VerifierRoundLogV1::open(&genesis, &key, NOW, "key-a", 7).unwrap();
        let round = request(&log, &key, "request-fresh-rows", vec![102, 103]);

        let descriptors = |first_start: u64| {
            (0..3)
                .map(|index| {
                    let payload = vec![index as u8];
                    VerifierFragmentDescriptorV1 {
                        ordinal: index,
                        component_id: "target-probability".into(),
                        kind: VerifierFragmentKindV1::TargetProbabilityEvidence,
                        logical_start: first_start + index as u64,
                        logical_count: 1,
                        byte_len: payload.len() as u64,
                        sha256: sha256_hex(&payload),
                    }
                })
                .collect::<Vec<_>>()
        };

        // T0 at output ordinal zero belongs to the parent. Fresh T1, T2 and
        // the bonus occupy ordinals one through three.
        log.prepare_result(&round, 1, 999, digest(16), descriptors(1), NOW + 3, &key)
            .unwrap();
        assert!(log
            .prepare_result(&round, 1, 999, digest(16), descriptors(0), NOW + 3, &key,)
            .is_err());
    }

    #[test]
    fn authentication_and_manifest_tampering_are_rejected() {
        let key = MacKey::from_bytes([0x55; 32]);
        let other = MacKey::from_bytes([0x56; 32]);
        let genesis = session(&key);
        assert_eq!(
            genesis.verify(&other, NOW, "key-a", 7),
            Err(VerifierProtocolError::Authentication)
        );
        let log = VerifierRoundLogV1::open(&genesis, &key, NOW, "key-a", 7).unwrap();
        let round = request(&log, &key, "request-1", vec![102]);
        let (descriptors, payloads) = fragments();
        let mut result = log
            .prepare_result(&round, 1, 999, digest(16), descriptors, NOW + 3, &key)
            .unwrap();
        result.core.fragments[0].sha256 = sha256_hex(b"tampered");
        assert_eq!(
            VerifierFragmentAssemblerV1::new(&result, &round, log.session(), &key, 1_024).err(),
            Some(VerifierProtocolError::Authentication)
        );

        let valid = log
            .prepare_result(&round, 1, 999, digest(16), fragments().0, NOW + 3, &key)
            .unwrap();
        let mut assembler =
            VerifierFragmentAssemblerV1::new(&valid, &round, log.session(), &key, 1_024).unwrap();
        assert!(assembler.insert(0, b"wrong bytes".to_vec()).is_err());
        assert_eq!(assembler.missing_ordinals(), vec![0, 1]);
        assert_eq!(payloads.len(), 2);
    }

    #[test]
    fn expired_requests_replay_and_expired_sessions_restore_without_new_admission() {
        let key = MacKey::from_bytes([0x55; 32]);
        let genesis = session(&key);
        let mut live = VerifierRoundLogV1::open(&genesis, &key, NOW, "key-a", 7).unwrap();
        let round = request(&live, &key, "request-1", vec![102]);
        let (descriptors, payloads) = fragments();
        let result = live
            .prepare_result(&round, 1, 999, digest(16), descriptors, NOW + 3, &key)
            .unwrap();
        let closure = assembled_closure(&result, &round, live.session(), &key, &payloads);
        live.apply_durable_result(&round, result.clone(), &closure, &key, NOW + 4)
            .unwrap();

        // Request admission expired, but its exact completion remains replayable.
        assert!(matches!(
            live.admit(&round, &key, NOW + 2_000).unwrap(),
            VerifyRoundAdmissionV1::Replay(_)
        ));
        assert!(VerifierRoundLogV1::open(
            &genesis,
            &key,
            genesis.core.expires_unix_ms + 1,
            "key-a",
            7,
        )
        .is_err());

        let mut restored = VerifierRoundLogV1::open_historical(&genesis, &key, "key-a", 7).unwrap();
        restored
            .apply_durable_result(
                &round,
                result,
                &closure,
                &key,
                genesis.core.expires_unix_ms + 1,
            )
            .unwrap();
        assert!(matches!(
            restored
                .admit(&round, &key, genesis.core.expires_unix_ms + 1)
                .unwrap(),
            VerifyRoundAdmissionV1::Replay(_)
        ));
        let new_round = request(&restored, &key, "request-2", vec![103]);
        assert!(matches!(
            restored.admit(&new_round, &key, NOW + 5),
            Err(VerifierProtocolError::Validation(_))
        ));
    }

    #[test]
    fn policy_context_and_fragment_schema_fail_closed() {
        let key = MacKey::from_bytes([0x55; 32]);
        let genesis = session(&key);
        let log = VerifierRoundLogV1::open(&genesis, &key, NOW, "key-a", 7).unwrap();
        let round = request(&log, &key, "request-1", vec![102]);

        let mut switched = round.core.clone();
        switched.coupling_policy = VerifierCouplingPolicyV1::SharedGumbel;
        assert!(AuthenticatedVerifyRoundV1::sign(switched, log.session(), &key).is_err());

        let mut scaled_q = round.core.clone();
        let VerifierDraftEvidenceV1::SparseMaximal { q_rows } = &mut scaled_q.draft_evidence else {
            unreachable!()
        };
        q_rows[0].entries[0].probability_f32_bits = 0.5f32.to_bits();
        scaled_q.draft_evidence_sha256 = sha256_hex(&canonical(&scaled_q.draft_evidence).unwrap());
        assert!(AuthenticatedVerifyRoundV1::sign(scaled_q, log.session(), &key).is_err());

        let mut near_limit_core = genesis.core.clone();
        near_limit_core.initial_model_position = near_limit_core.max_context_tokens - 1;
        let near_limit = AuthenticatedVerifierSessionV1::sign(near_limit_core, &key).unwrap();
        assert!(
            AuthenticatedVerifyRoundV1::sign(round.core.clone(), &near_limit.core, &key).is_err()
        );

        assert!(log
            .prepare_result(&round, 1, 999, digest(16), Vec::new(), NOW + 3, &key)
            .is_err());
        let (mut descriptors, _) = fragments();
        descriptors.pop();
        assert!(log
            .prepare_result(&round, 1, 999, digest(16), descriptors, NOW + 3, &key)
            .is_err());
    }

    #[test]
    fn head_and_apply_are_bound_to_the_exact_verified_fragment_closure() {
        let key = MacKey::from_bytes([0x55; 32]);
        let genesis = session(&key);
        let mut log = VerifierRoundLogV1::open(&genesis, &key, NOW, "key-a", 7).unwrap();
        let round = request(&log, &key, "request-1", vec![102]);
        let (descriptors_a, payloads_a) = fragments();
        let result_a = log
            .prepare_result(&round, 1, 999, digest(16), descriptors_a, NOW + 3, &key)
            .unwrap();
        let closure_a = assembled_closure(&result_a, &round, log.session(), &key, &payloads_a);

        let payloads_b = [b"other-hidden-0".to_vec(), b"other-hidden-1".to_vec()];
        let descriptors_b = payloads_b
            .iter()
            .enumerate()
            .map(|(index, payload)| VerifierFragmentDescriptorV1 {
                ordinal: index as u32,
                component_id: "target-hidden".into(),
                kind: VerifierFragmentKindV1::TargetHidden,
                logical_start: index as u64,
                logical_count: 1,
                byte_len: payload.len() as u64,
                sha256: sha256_hex(payload),
            })
            .collect();
        let result_b = log
            .prepare_result(&round, 1, 999, digest(16), descriptors_b, NOW + 3, &key)
            .unwrap();
        assert_ne!(result_a.core.new_head_sha256, result_b.core.new_head_sha256);
        assert!(log
            .apply_durable_result(&round, result_b, &closure_a, &key, NOW + 4)
            .is_err());
        log.apply_durable_result(&round, result_a, &closure_a, &key, NOW + 4)
            .unwrap();
    }
}
