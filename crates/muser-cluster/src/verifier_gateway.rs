//! Non-serving verifier gateway core over [`crate::verifier_v2`].
//!
//! The gateway owns ordering and authentication, while a
//! [`TargetEvaluatorV1`] returns raw, unsigned model output.  The gateway
//! durably reserves the request before model execution, converts the raw
//! output into the target-only signed V2 result, persists that exact result
//! and its content-addressed fragments, records PREPARED, stages an invisible
//! render, commits and activates it, and only then constructs a reply.
//!
//! This is not a listener.  Production mTLS, socket admission, timeouts,
//! connection ownership, and target-runtime integration remain explicit
//! wiring gaps; see [`crate::verifier_gateway_codec`] for the required ALPN.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::os::fd::AsRawFd;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use kvpack_handoff::{canonical_json, decode_canonical_json, sha256_hex, MacKey};
use serde::{Deserialize, Serialize};

use crate::verifier_gateway_adapter::VerifiedGatewayRoundResultV1;
use crate::verifier_gateway_codec::{
    AuthenticatedGatewaySourceRequestV1, GatewaySessionFailureReasonV1, GatewaySessionFailureV1,
    GatewaySourceCommandV1, PendingRoundSummaryV1, SourceInstallReceiptV1, VerifierGatewayFrameV1,
    VerifierGatewayMessageV1, VerifierGatewayReplyV1, DEFAULT_VERIFIER_GATEWAY_PAYLOAD_BYTES_V1,
    VERIFIER_SOURCE_INSTALL_RECEIPT_PROTOCOL_V1,
};
use crate::verifier_v2::{
    build_result, AuthenticatedResultV2, AuthenticatedRoundV2, AuthorityFenceV2, CouplingPolicyV2,
    DurableVerifierJournalV2, FragmentAssemblerV2, FragmentDescriptorV2, FragmentKindV2,
    FragmentRequirementV2, PendingRoundV2, RendererStageReceiptV2, ReserveOutcomeV2,
    ResultDecisionV2, SamplerStateV2, SessionCoreV2, TargetSigningKeyV2, VerifiedClosureV2,
    VerifierRendererV2, VerifierV2Error, MAX_FRAGMENTS_V2, VERIFIER_ROUND_PROTOCOL_V3,
};

const GATEWAY_STORE_PROTOCOL_V1: &str = "muser-verifier-gateway-store-v2";
const GATEWAY_ACTIVATION_INTENT_PROTOCOL_V1: &str = "muser-verifier-fenced-activation-v1";
const GATEWAY_STORE_MAX_JSON_V1: usize = 64 * 1024 * 1024;
const MAX_RAW_FRAGMENT_OCCURRENCES_V1: usize = MAX_FRAGMENTS_V2 * 2;
const MAX_PENDING_ACTIVATION_INTENTS_V1: usize = 4_096;
const MAX_SOURCE_RECEIPT_CLOCK_SKEW_MS_V1: u64 = 30_000;

#[derive(Debug, thiserror::Error)]
pub enum VerifierGatewayErrorV1 {
    #[error("verifier gateway validation: {0}")]
    Validation(String),
    #[error(transparent)]
    Evaluator(#[from] TargetEvaluationErrorV1),
    #[error(transparent)]
    Codec(#[from] crate::verifier_gateway_codec::VerifierGatewayCodecErrorV1),
    #[error(transparent)]
    Verifier(#[from] VerifierV2Error),
    #[error("verifier gateway I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("verifier gateway clock: {0}")]
    Clock(String),
    #[error("verifier source durable install: {0}")]
    SourceInstall(String),
    #[error(transparent)]
    ActivationFence(#[from] GatewayActivationFenceErrorV1),
}

pub type GatewayResultV1<T> = std::result::Result<T, VerifierGatewayErrorV1>;

fn invalid(message: impl Into<String>) -> VerifierGatewayErrorV1 {
    VerifierGatewayErrorV1::Validation(message.into())
}

#[derive(Debug, Clone, Copy, thiserror::Error, PartialEq, Eq)]
pub enum GatewayActivationFenceErrorV1 {
    #[error("activation fence is temporarily unavailable")]
    Unavailable,
    #[error("activation fence observed a conflicting head")]
    Conflict,
}

/// Durable intent placed before the local commit WAL. A production fence must
/// hold exclusive writer authority from `acquire_activation` through
/// `publish_committed`; a crash after the local WAL is reconciled through the
/// same expected-head transition before the gateway serves again.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FencedActivationIntentV1 {
    pub protocol: String,
    pub session_record_sha256: String,
    pub authority_term: u64,
    pub request_id: String,
    pub request_intent_sha256: String,
    pub result_sha256: String,
    pub expected_head_sha256: String,
    pub committed_head_sha256: String,
    pub renderer: RendererStageReceiptV2,
    pub created_unix_ms: u64,
}

pub trait GatewayActivationPermitV1: AuthorityFenceV2 {
    /// Publish the head transition while the exclusive fencing permit remains
    /// held. Implementations must be idempotent for exact recovery.
    fn publish_committed(
        &mut self,
        intent: &FencedActivationIntentV1,
    ) -> std::result::Result<(), GatewayActivationFenceErrorV1>;
}

/// External linearization boundary. A Boolean lease probe is insufficient:
/// this interface requires an owned permit spanning the journal WAL and
/// renderer activation, plus restart reconciliation for WAL-first crashes.
pub trait GatewayActivationFenceV1: AuthorityFenceV2 {
    fn acquire_activation<'a>(
        &'a mut self,
        intent: &FencedActivationIntentV1,
        now_unix_ms: u64,
    ) -> std::result::Result<Box<dyn GatewayActivationPermitV1 + 'a>, GatewayActivationFenceErrorV1>;

    fn reconcile_committed(
        &mut self,
        intent: &FencedActivationIntentV1,
        now_unix_ms: u64,
    ) -> std::result::Result<(), GatewayActivationFenceErrorV1>;
}

struct ActivationPermitFenceV1<'a>(&'a dyn GatewayActivationPermitV1);

impl AuthorityFenceV2 for ActivationPermitFenceV1<'_> {
    fn permits(
        &self,
        session_id: &str,
        log_writer_authority_id: &str,
        authority_lease_id: &str,
        authority_term: u64,
        target_executor_id: &str,
        now_unix_ms: u64,
    ) -> bool {
        self.0.permits(
            session_id,
            log_writer_authority_id,
            authority_lease_id,
            authority_term,
            target_executor_id,
            now_unix_ms,
        )
    }
}

/// Lets the V2 journal durably linearize its commit without exposing the
/// renderer before the external head CAS succeeds. The real renderer is
/// activated only after `publish_committed` accepts the exact intent.
struct DeferredActivationRendererV1 {
    expected: RendererStageReceiptV2,
    observed: bool,
}

impl DeferredActivationRendererV1 {
    fn new(expected: RendererStageReceiptV2) -> Self {
        Self {
            expected,
            observed: false,
        }
    }
}

impl VerifierRendererV2 for DeferredActivationRendererV1 {
    fn stage(
        &mut self,
        _result: &AuthenticatedResultV2,
        _closure: &VerifiedClosureV2,
        _now_unix_ms: u64,
    ) -> std::result::Result<RendererStageReceiptV2, String> {
        Err("deferred activation renderer cannot stage".into())
    }

    fn activate(&mut self, receipt: &RendererStageReceiptV2) -> std::result::Result<(), String> {
        if receipt != &self.expected {
            return Err("deferred activation receipt differs".into());
        }
        self.observed = true;
        Ok(())
    }
}

enum FencedActivationOutcomeV1 {
    Activated,
    SessionFailed(GatewaySessionFailureV1),
}

fn activate_with_external_fence_v1<R, F>(
    journal: &mut DurableVerifierJournalV2,
    renderer: &mut R,
    fence: &mut F,
    request: &AuthenticatedRoundV2,
    result: &AuthenticatedResultV2,
    intent: &FencedActivationIntentV1,
    now_unix_ms: u64,
) -> GatewayResultV1<AuthenticatedResultV2>
where
    R: VerifierRendererV2,
    F: GatewayActivationFenceV1,
{
    let mut permit = fence.acquire_activation(intent, now_unix_ms)?;
    let session = journal.session();
    if !permit.permits(
        &session.session_id,
        &session.log_writer_authority_id,
        &session.authority_lease_id,
        session.authority_term,
        &session.target_executor_id,
        now_unix_ms,
    ) {
        return Err(VerifierV2Error::LeaseNotLive.into());
    }
    let permit_fence = ActivationPermitFenceV1(permit.as_ref());
    let mut deferred = DeferredActivationRendererV1::new(intent.renderer.clone());
    let committed =
        journal.commit_and_activate(request, &permit_fence, &mut deferred, now_unix_ms)?;
    if !deferred.observed || committed != *result {
        return Err(invalid(
            "journal activation differs from its durable fenced intent",
        ));
    }
    permit.publish_committed(intent)?;
    renderer
        .activate(&intent.renderer)
        .map_err(VerifierV2Error::Renderer)?;
    drop(permit);
    Ok(committed)
}

fn recover_with_external_fence_v1<R, F>(
    journal: &mut DurableVerifierJournalV2,
    renderer: &mut R,
    fence: &mut F,
    request: &AuthenticatedRoundV2,
    result: &AuthenticatedResultV2,
    intent: &FencedActivationIntentV1,
    now_unix_ms: u64,
) -> GatewayResultV1<()>
where
    R: VerifierRendererV2,
    F: GatewayActivationFenceV1,
{
    // The core commit WAL is already an immutable decision. Recovery must not
    // depend on the expired writer lease: the external registry atomically
    // reconciles only this exact expected-head transition, or reports a
    // conflict before any real renderer visibility.
    fence.reconcile_committed(intent, now_unix_ms)?;
    let mut deferred = DeferredActivationRendererV1::new(intent.renderer.clone());
    journal.recover_pending(&mut deferred, now_unix_ms)?;
    if !deferred.observed || journal.completed_result(&request.core.request_id) != Some(result) {
        return Err(invalid(
            "recovered commit WAL differs from its durable fenced intent",
        ));
    }
    renderer
        .activate(&intent.renderer)
        .map_err(VerifierV2Error::Renderer)?;
    Ok(())
}

/// One opaque model fragment before the gateway derives its byte length and
/// digest. Occurrences may arrive in any order and exact duplicates are
/// absorbed; ordinals become the canonical signed descriptor order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawTargetFragmentV1 {
    pub ordinal: u32,
    pub component_id: String,
    pub kind: FragmentKindV2,
    pub logical_start: u64,
    pub logical_count: u64,
    pub payload_abi: String,
    pub bytes_per_logical_row: u64,
    pub payload: Vec<u8>,
}

/// Unsigned target output. Implementations cannot manufacture an
/// `AuthenticatedResultV2`; only the gateway holding the target Ed25519 key
/// crosses that directional authority boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawTargetEvaluationV1 {
    pub decision: ResultDecisionV2,
    pub post_verify_sampler_state: SamplerStateV2,
    pub fragments: Vec<RawTargetFragmentV1>,
}

/// Closed model/runtime identity that an evaluator adapter must derive from
/// its loaded target before the gateway will sign anything it returns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetEvaluatorCapabilitiesV1 {
    pub target_executor_id: String,
    pub semantic_product_sha256: String,
    pub target_checkpoint_sha256: String,
    pub target_engine_sha256: String,
    pub target_sampler_sha256: String,
    pub target_genesis_root_sha256: String,
    pub tokenizer_sha256: String,
    pub vocabulary_sha256: String,
    pub portable_kv_abi: String,
    pub sampler_config_sha256: String,
    pub coupling_policy: CouplingPolicyV2,
    pub vocab_size: u32,
    pub max_context_tokens: u32,
    pub max_drafts: u32,
    pub max_total_fragment_bytes: u64,
    pub fragment_requirements: Vec<FragmentRequirementV2>,
}

impl TargetEvaluatorCapabilitiesV1 {
    fn matches_session(&self, session: &SessionCoreV2) -> bool {
        self.target_executor_id == session.target_executor_id
            && self.semantic_product_sha256 == session.identity.semantic_product_sha256
            && self.target_checkpoint_sha256 == session.identity.target_checkpoint_sha256
            && self.target_engine_sha256 == session.identity.target_engine_sha256
            && self.target_sampler_sha256 == session.identity.target_sampler_sha256
            && self.target_genesis_root_sha256 == session.identity.target_genesis_root_sha256
            && self.tokenizer_sha256 == session.identity.tokenizer_sha256
            && self.vocabulary_sha256 == session.identity.vocabulary_sha256
            && self.portable_kv_abi == session.identity.portable_kv_abi
            && self.sampler_config_sha256 == session.sampler_config_sha256
            && self.coupling_policy == session.coupling_policy
            && self.vocab_size == session.vocab_size
            && self.max_context_tokens == session.max_context_tokens
            && self.max_drafts == session.max_drafts
            && self.max_total_fragment_bytes == session.max_total_fragment_bytes
            && self.fragment_requirements == session.fragment_requirements
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TargetEvaluationErrorV1 {
    #[error("transient target evaluation failure: {0}")]
    Transient(String),
    #[error("permanent target evaluation failure: {0}")]
    Permanent(String),
}

/// Typestate proving that the gateway durably reserved this exact round before
/// invoking target semantics. Construction is private to the gateway.
pub struct ReservedTargetRequestV1<'a> {
    session: &'a SessionCoreV2,
    request: &'a AuthenticatedRoundV2,
}

impl<'a> ReservedTargetRequestV1<'a> {
    pub fn session(&self) -> &'a SessionCoreV2 {
        self.session
    }

    pub fn request(&self) -> &'a AuthenticatedRoundV2 {
        self.request
    }
}

pub trait TargetEvaluatorV1 {
    fn capabilities(&self) -> &TargetEvaluatorCapabilitiesV1;

    /// Evaluate the exact authenticated carried-frontier request. For a Close
    /// intent the adapter must avoid model execution and return the matching
    /// boundary closure. The gateway validates every resulting transition.
    fn evaluate(
        &mut self,
        reserved: ReservedTargetRequestV1<'_>,
    ) -> std::result::Result<RawTargetEvaluationV1, TargetEvaluationErrorV1>;
}

pub trait GatewayClockV1 {
    fn now_unix_ms(&mut self) -> std::result::Result<u64, String>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SystemGatewayClockV1;

impl GatewayClockV1 for SystemGatewayClockV1 {
    fn now_unix_ms(&mut self) -> std::result::Result<u64, String> {
        let elapsed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| error.to_string())?;
        u64::try_from(elapsed.as_millis()).map_err(|_| "Unix time overflow".into())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatewayDrivePhaseV1 {
    Reserved,
    Evaluated,
    Prepared,
    Staged,
    Activated,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct GatewayStoreMetadataV1 {
    protocol: String,
    session_record_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct StoredGatewayResultV1 {
    protocol: String,
    session_record_sha256: String,
    request: AuthenticatedRoundV2,
    result: AuthenticatedResultV2,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct StoredRoundAdmissionV1 {
    protocol: String,
    session_record_sha256: String,
    source_request: AuthenticatedGatewaySourceRequestV1,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct StoredSourceAckV1 {
    protocol: String,
    session_record_sha256: String,
    source_request: AuthenticatedGatewaySourceRequestV1,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct StoredSessionFailureV1 {
    protocol: String,
    session_record_sha256: String,
    failure: GatewaySessionFailureV1,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct StoredActivationIntentV1 {
    protocol: String,
    session_record_sha256: String,
    intent: FencedActivationIntentV1,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct StoredActivationCompleteV1 {
    protocol: String,
    session_record_sha256: String,
    request_id: String,
    result_sha256: String,
    completed_unix_ms: u64,
}

/// Private sidecar retaining the exact post-evaluation result and response
/// fragments across PREPARED/activation/reply crashes.  The V2 journal remains
/// the ordered authority; this store is immutable repair material only.
pub struct DurableGatewayFragmentStoreV1 {
    root: PathBuf,
    _lock: File,
    session_record_sha256: String,
}

impl DurableGatewayFragmentStoreV1 {
    pub fn create(root: &Path, session_record_sha256: &str) -> GatewayResultV1<Self> {
        validate_digest("gateway store session", session_record_sha256)?;
        if !root.is_absolute() || root.exists() || root.is_symlink() {
            return Err(invalid(
                "gateway store destination must be a new absolute path",
            ));
        }
        fs::create_dir(root)?;
        fs::set_permissions(root, fs::Permissions::from_mode(0o700))?;
        for directory in ["admissions", "results", "fragments", "acks", "fences"] {
            let path = root.join(directory);
            fs::create_dir(&path)?;
            fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
        }
        let metadata = GatewayStoreMetadataV1 {
            protocol: GATEWAY_STORE_PROTOCOL_V1.into(),
            session_record_sha256: session_record_sha256.into(),
        };
        write_exclusive(&root.join("metadata.json"), &canonical(&metadata)?)?;
        sync_directory(root)?;
        let lock = open_lock(root)?;
        Ok(Self {
            root: root.into(),
            _lock: lock,
            session_record_sha256: session_record_sha256.into(),
        })
    }

    pub fn open(root: &Path, session_record_sha256: &str) -> GatewayResultV1<Self> {
        validate_digest("gateway store session", session_record_sha256)?;
        validate_private_directory(root)?;
        validate_private_directory(&root.join("admissions"))?;
        validate_private_directory(&root.join("results"))?;
        validate_private_directory(&root.join("fragments"))?;
        validate_private_directory(&root.join("acks"))?;
        validate_private_directory(&root.join("fences"))?;
        let metadata: GatewayStoreMetadataV1 =
            read_canonical(&root.join("metadata.json"), GATEWAY_STORE_MAX_JSON_V1)?;
        if metadata.protocol != GATEWAY_STORE_PROTOCOL_V1
            || metadata.session_record_sha256 != session_record_sha256
        {
            return Err(invalid("gateway store belongs to another session"));
        }
        let lock = open_lock(root)?;
        Ok(Self {
            root: root.into(),
            _lock: lock,
            session_record_sha256: session_record_sha256.into(),
        })
    }

    pub fn session_record_sha256(&self) -> &str {
        &self.session_record_sha256
    }

    fn persist_round_admission(
        &self,
        source_request: &AuthenticatedGatewaySourceRequestV1,
        key: &MacKey,
    ) -> GatewayResultV1<()> {
        source_request.verify(key)?;
        if source_request.core.session_record_sha256 != self.session_record_sha256 {
            return Err(invalid("round admission addresses another session"));
        }
        let GatewaySourceCommandV1::SubmitRound { round, .. } = &source_request.core.command else {
            return Err(invalid("round admission record is not a submission"));
        };
        let stored = StoredRoundAdmissionV1 {
            protocol: GATEWAY_STORE_PROTOCOL_V1.into(),
            session_record_sha256: self.session_record_sha256.clone(),
            source_request: source_request.clone(),
        };
        let path = self.admission_path(&round.core.request_id);
        if path.exists() {
            let existing: StoredRoundAdmissionV1 =
                read_canonical(&path, GATEWAY_STORE_MAX_JSON_V1)?;
            if existing != stored {
                return Err(invalid(
                    "round request ID changed its authenticated source admission",
                ));
            }
            return Ok(());
        }
        write_exclusive(&path, &canonical(&stored)?)?;
        sync_directory(&self.root.join("admissions"))
    }

    fn has_round_admission(&self, request_id: &str) -> GatewayResultV1<bool> {
        validate_identifier("round admission", request_id)?;
        Ok(self.admission_path(request_id).exists())
    }

    fn round_admission(
        &self,
        request_id: &str,
        key: &MacKey,
    ) -> GatewayResultV1<AuthenticatedGatewaySourceRequestV1> {
        validate_identifier("round admission", request_id)?;
        let stored: StoredRoundAdmissionV1 =
            read_canonical(&self.admission_path(request_id), GATEWAY_STORE_MAX_JSON_V1)?;
        stored.source_request.verify(key)?;
        let GatewaySourceCommandV1::SubmitRound { round, .. } = &stored.source_request.core.command
        else {
            return Err(invalid("stored round admission is not a submission"));
        };
        if stored.protocol != GATEWAY_STORE_PROTOCOL_V1
            || stored.session_record_sha256 != self.session_record_sha256
            || stored.source_request.core.session_record_sha256 != self.session_record_sha256
            || round.core.request_id != request_id
        {
            return Err(invalid("stored round admission identity differs"));
        }
        Ok(stored.source_request)
    }

    fn persist(
        &self,
        request: &AuthenticatedRoundV2,
        result: &AuthenticatedResultV2,
        payloads: &BTreeMap<u32, Vec<u8>>,
        session: &SessionCoreV2,
        key: &MacKey,
    ) -> GatewayResultV1<()> {
        request.verify_historical(session, key)?;
        result.verify_against(request, session)?;
        if payloads.len() != result.core.fragments.len() {
            return Err(invalid("stored fragment count differs from signed result"));
        }
        for descriptor in &result.core.fragments {
            let payload = payloads
                .get(&descriptor.ordinal)
                .ok_or_else(|| invalid("stored fragment ordinal is absent"))?;
            if payload.len() as u64 != descriptor.byte_len
                || sha256_hex(payload) != descriptor.sha256
            {
                return Err(invalid("stored fragment payload differs from descriptor"));
            }
            self.write_fragment(&descriptor.sha256, payload)?;
        }
        sync_directory(&self.root.join("fragments"))?;
        let stored = StoredGatewayResultV1 {
            protocol: GATEWAY_STORE_PROTOCOL_V1.into(),
            session_record_sha256: self.session_record_sha256.clone(),
            request: request.clone(),
            result: result.clone(),
        };
        let bytes = canonical(&stored)?;
        if bytes.len() > GATEWAY_STORE_MAX_JSON_V1 {
            return Err(invalid("stored result manifest exceeds bounds"));
        }
        let path = self.result_path(&request.core.request_id);
        if path.exists() {
            let existing: StoredGatewayResultV1 = read_canonical(&path, GATEWAY_STORE_MAX_JSON_V1)?;
            if existing != stored {
                return Err(invalid("stored request ID changed target result"));
            }
            return Ok(());
        }
        write_exclusive(&path, &bytes)?;
        sync_directory(&self.root.join("results"))
    }

    fn load_for_request(
        &self,
        request: &AuthenticatedRoundV2,
        session: &SessionCoreV2,
        key: &MacKey,
    ) -> GatewayResultV1<Option<StoredGatewayResultV1>> {
        let Some(stored) = self.load(&request.core.request_id, session, key)? else {
            return Ok(None);
        };
        if stored.request != *request {
            return Err(invalid("stored request ID changed authenticated intent"));
        }
        Ok(Some(stored))
    }

    fn load(
        &self,
        request_id: &str,
        session: &SessionCoreV2,
        key: &MacKey,
    ) -> GatewayResultV1<Option<StoredGatewayResultV1>> {
        validate_identifier("stored request", request_id)?;
        let path = self.result_path(request_id);
        if !path.exists() {
            return Ok(None);
        }
        let stored: StoredGatewayResultV1 = read_canonical(&path, GATEWAY_STORE_MAX_JSON_V1)?;
        if stored.protocol != GATEWAY_STORE_PROTOCOL_V1
            || stored.session_record_sha256 != self.session_record_sha256
            || stored.request.core.request_id != request_id
        {
            return Err(invalid("stored result identity differs"));
        }
        stored.request.verify_historical(session, key)?;
        stored.result.verify_against(&stored.request, session)?;
        for descriptor in &stored.result.core.fragments {
            self.read_fragment(descriptor)?;
        }
        Ok(Some(stored))
    }

    fn payloads(&self, stored: &StoredGatewayResultV1) -> GatewayResultV1<BTreeMap<u32, Vec<u8>>> {
        stored
            .result
            .core
            .fragments
            .iter()
            .map(|descriptor| Ok((descriptor.ordinal, self.read_fragment(descriptor)?)))
            .collect()
    }

    fn persist_source_ack(
        &self,
        source_request: &AuthenticatedGatewaySourceRequestV1,
        key: &MacKey,
    ) -> GatewayResultV1<()> {
        source_request.verify(key)?;
        if source_request.core.session_record_sha256 != self.session_record_sha256 {
            return Err(invalid("source ACK addresses another session"));
        }
        let GatewaySourceCommandV1::SourceAck { receipt } = &source_request.core.command else {
            return Err(invalid("durable source ACK record is not an ACK"));
        };
        let stored = StoredSourceAckV1 {
            protocol: GATEWAY_STORE_PROTOCOL_V1.into(),
            session_record_sha256: self.session_record_sha256.clone(),
            source_request: source_request.clone(),
        };
        let path = self.ack_path(&receipt.applied_head_sha256);
        if path.exists() {
            let existing: StoredSourceAckV1 = read_canonical(&path, GATEWAY_STORE_MAX_JSON_V1)?;
            if existing != stored {
                return Err(invalid("source ACK changed durable receipt"));
            }
            return Ok(());
        }
        write_exclusive(&path, &canonical(&stored)?)?;
        sync_directory(&self.root.join("acks"))
    }

    fn has_source_ack(&self, head_sha256: &str, key: &MacKey) -> GatewayResultV1<bool> {
        validate_digest("source ACK head", head_sha256)?;
        let path = self.ack_path(head_sha256);
        if !path.exists() {
            return Ok(false);
        }
        let stored: StoredSourceAckV1 = read_canonical(&path, GATEWAY_STORE_MAX_JSON_V1)?;
        stored.source_request.verify(key)?;
        let GatewaySourceCommandV1::SourceAck { receipt } = &stored.source_request.core.command
        else {
            return Err(invalid("stored source ACK record is not an ACK"));
        };
        if stored.protocol != GATEWAY_STORE_PROTOCOL_V1
            || stored.session_record_sha256 != self.session_record_sha256
            || stored.source_request.core.session_record_sha256 != self.session_record_sha256
            || receipt.protocol != VERIFIER_SOURCE_INSTALL_RECEIPT_PROTOCOL_V1
            || receipt.applied_head_sha256 != head_sha256
            || receipt.source_state_bytes == 0
        {
            return Err(invalid("stored source ACK identity differs"));
        }
        validate_digest("stored source state", &receipt.source_state_sha256)?;
        validate_digest(
            "stored source transcript",
            &receipt.installed_transcript_sha256,
        )?;
        Ok(true)
    }

    fn persist_session_failure(&self, failure: &GatewaySessionFailureV1) -> GatewayResultV1<()> {
        validate_session_failure(failure)?;
        let stored = StoredSessionFailureV1 {
            protocol: GATEWAY_STORE_PROTOCOL_V1.into(),
            session_record_sha256: self.session_record_sha256.clone(),
            failure: failure.clone(),
        };
        let path = self.root.join("session-failure.json");
        if path.exists() {
            let existing: StoredSessionFailureV1 =
                read_canonical(&path, GATEWAY_STORE_MAX_JSON_V1)?;
            if existing != stored {
                return Err(invalid("durable session failure changed identity"));
            }
            return Ok(());
        }
        write_exclusive(&path, &canonical(&stored)?)?;
        sync_directory(&self.root)
    }

    fn session_failure(&self) -> GatewayResultV1<Option<GatewaySessionFailureV1>> {
        let path = self.root.join("session-failure.json");
        if !path.exists() {
            return Ok(None);
        }
        let stored: StoredSessionFailureV1 = read_canonical(&path, GATEWAY_STORE_MAX_JSON_V1)?;
        if stored.protocol != GATEWAY_STORE_PROTOCOL_V1
            || stored.session_record_sha256 != self.session_record_sha256
        {
            return Err(invalid(
                "durable session failure belongs to another session",
            ));
        }
        validate_session_failure(&stored.failure)?;
        Ok(Some(stored.failure))
    }

    fn persist_activation_intent(&self, intent: &FencedActivationIntentV1) -> GatewayResultV1<()> {
        validate_activation_intent(intent, &self.session_record_sha256)?;
        let stored = StoredActivationIntentV1 {
            protocol: GATEWAY_STORE_PROTOCOL_V1.into(),
            session_record_sha256: self.session_record_sha256.clone(),
            intent: intent.clone(),
        };
        let path = self.activation_intent_path(&intent.request_id);
        if path.exists() {
            let existing: StoredActivationIntentV1 =
                read_canonical(&path, GATEWAY_STORE_MAX_JSON_V1)?;
            if existing != stored {
                return Err(invalid("activation intent changed identity"));
            }
            return Ok(());
        }
        write_exclusive(&path, &canonical(&stored)?)?;
        sync_directory(&self.root.join("fences"))
    }

    fn persist_activation_complete(
        &self,
        intent: &FencedActivationIntentV1,
        completed_unix_ms: u64,
    ) -> GatewayResultV1<()> {
        validate_activation_intent(intent, &self.session_record_sha256)?;
        if completed_unix_ms == 0 {
            return Err(invalid("activation completion time differs"));
        }
        let stored = StoredActivationCompleteV1 {
            protocol: GATEWAY_STORE_PROTOCOL_V1.into(),
            session_record_sha256: self.session_record_sha256.clone(),
            request_id: intent.request_id.clone(),
            result_sha256: intent.result_sha256.clone(),
            completed_unix_ms,
        };
        let path = self.activation_complete_path(&intent.request_id);
        if path.exists() {
            let existing: StoredActivationCompleteV1 =
                read_canonical(&path, GATEWAY_STORE_MAX_JSON_V1)?;
            if existing.request_id != stored.request_id
                || existing.result_sha256 != stored.result_sha256
                || existing.protocol != stored.protocol
                || existing.session_record_sha256 != stored.session_record_sha256
                || existing.completed_unix_ms == 0
            {
                return Err(invalid("activation completion changed identity"));
            }
            return Ok(());
        }
        write_exclusive(&path, &canonical(&stored)?)?;
        sync_directory(&self.root.join("fences"))
    }

    fn pending_activation_intents(&self) -> GatewayResultV1<Vec<FencedActivationIntentV1>> {
        let mut paths =
            fs::read_dir(self.root.join("fences"))?.collect::<std::result::Result<Vec<_>, _>>()?;
        if paths.len() > MAX_PENDING_ACTIVATION_INTENTS_V1.saturating_mul(2) {
            return Err(invalid("activation fence record count exceeds bounds"));
        }
        paths.sort_by_key(std::fs::DirEntry::file_name);
        let mut pending = Vec::new();
        for entry in paths {
            let file_name = entry
                .file_name()
                .into_string()
                .map_err(|_| invalid("activation fence filename is not UTF-8"))?;
            let Some(stem) = file_name.strip_suffix(".intent.json") else {
                continue;
            };
            validate_digest("activation fence filename", stem)?;
            let stored: StoredActivationIntentV1 =
                read_canonical(&entry.path(), GATEWAY_STORE_MAX_JSON_V1)?;
            if stored.protocol != GATEWAY_STORE_PROTOCOL_V1
                || stored.session_record_sha256 != self.session_record_sha256
            {
                return Err(invalid("activation intent belongs to another session"));
            }
            validate_activation_intent(&stored.intent, &self.session_record_sha256)?;
            if self.activation_complete(&stored.intent)? {
                continue;
            }
            pending.push(stored.intent);
        }
        if pending.len() > MAX_PENDING_ACTIVATION_INTENTS_V1 {
            return Err(invalid("pending activation intent count exceeds bounds"));
        }
        Ok(pending)
    }

    fn activation_complete(&self, intent: &FencedActivationIntentV1) -> GatewayResultV1<bool> {
        let path = self.activation_complete_path(&intent.request_id);
        if !path.exists() {
            return Ok(false);
        }
        let stored: StoredActivationCompleteV1 = read_canonical(&path, GATEWAY_STORE_MAX_JSON_V1)?;
        if stored.protocol != GATEWAY_STORE_PROTOCOL_V1
            || stored.session_record_sha256 != self.session_record_sha256
            || stored.request_id != intent.request_id
            || stored.result_sha256 != intent.result_sha256
            || stored.completed_unix_ms == 0
        {
            return Err(invalid("activation completion identity differs"));
        }
        Ok(true)
    }

    fn read_fragment(&self, descriptor: &FragmentDescriptorV2) -> GatewayResultV1<Vec<u8>> {
        validate_digest("stored fragment", &descriptor.sha256)?;
        let path = self.fragment_path(&descriptor.sha256);
        validate_private_file(&path, Some(descriptor.byte_len))?;
        let payload = fs::read(path)?;
        if sha256_hex(&payload) != descriptor.sha256 {
            return Err(invalid("stored fragment content digest differs"));
        }
        Ok(payload)
    }

    fn write_fragment(&self, digest: &str, payload: &[u8]) -> GatewayResultV1<()> {
        validate_digest("stored fragment", digest)?;
        if sha256_hex(payload) != digest {
            return Err(invalid("fragment CAS key differs from payload"));
        }
        let path = self.fragment_path(digest);
        if path.exists() {
            validate_private_file(&path, Some(payload.len() as u64))?;
            if fs::read(path)? != payload {
                return Err(invalid("fragment CAS collision"));
            }
            return Ok(());
        }
        write_exclusive(&path, payload)
    }

    fn result_path(&self, request_id: &str) -> PathBuf {
        self.root
            .join("results")
            .join(format!("{}.json", sha256_hex(request_id.as_bytes())))
    }

    fn admission_path(&self, request_id: &str) -> PathBuf {
        self.root
            .join("admissions")
            .join(format!("{}.json", sha256_hex(request_id.as_bytes())))
    }

    fn fragment_path(&self, digest: &str) -> PathBuf {
        self.root.join("fragments").join(format!("{digest}.bin"))
    }

    fn ack_path(&self, head_sha256: &str) -> PathBuf {
        self.root.join("acks").join(format!("{head_sha256}.json"))
    }

    fn activation_intent_path(&self, request_id: &str) -> PathBuf {
        self.root
            .join("fences")
            .join(format!("{}.intent.json", sha256_hex(request_id.as_bytes())))
    }

    fn activation_complete_path(&self, request_id: &str) -> PathBuf {
        self.root.join("fences").join(format!(
            "{}.complete.json",
            sha256_hex(request_id.as_bytes())
        ))
    }
}

pub struct VerifierGatewayV1<E, R, F, C = SystemGatewayClockV1> {
    journal: DurableVerifierJournalV2,
    store: DurableGatewayFragmentStoreV1,
    source_key: MacKey,
    target_signer: TargetSigningKeyV2,
    evaluator: E,
    renderer: R,
    fence: F,
    clock: C,
    session_record_sha256: String,
}

pub struct VerifierGatewayRuntimeV1<E, R, F, C = SystemGatewayClockV1> {
    pub evaluator: E,
    pub renderer: R,
    pub fence: F,
    pub clock: C,
}

impl<E, R, F, C> VerifierGatewayV1<E, R, F, C>
where
    E: TargetEvaluatorV1,
    R: VerifierRendererV2,
    F: GatewayActivationFenceV1,
    C: GatewayClockV1,
{
    pub fn new(
        journal: DurableVerifierJournalV2,
        store: DurableGatewayFragmentStoreV1,
        source_key: MacKey,
        target_signer: TargetSigningKeyV2,
        runtime: VerifierGatewayRuntimeV1<E, R, F, C>,
    ) -> GatewayResultV1<Self> {
        let VerifierGatewayRuntimeV1 {
            evaluator,
            renderer,
            fence,
            clock,
        } = runtime;
        let authenticated_session = journal.authenticated_session();
        let session = &authenticated_session.core;
        authenticated_session.verify_historical(
            &source_key,
            &session.hmac_key_id,
            session.hmac_key_epoch,
        )?;
        if target_signer.public_key() != session.target_public_key {
            return Err(invalid("target signer differs from admitted session"));
        }
        if !evaluator.capabilities().matches_session(session) {
            return Err(invalid(
                "target evaluator capabilities differ from admitted session",
            ));
        }
        let session_record_sha256 = authenticated_session.record_digest()?;
        if store.session_record_sha256() != session_record_sha256 {
            return Err(invalid("gateway store differs from admitted session"));
        }
        Ok(Self {
            journal,
            store,
            source_key,
            target_signer,
            evaluator,
            renderer,
            fence,
            clock,
            session_record_sha256,
        })
    }

    #[cfg(test)]
    fn journal(&self) -> &DurableVerifierJournalV2 {
        &self.journal
    }

    pub fn handle_source(
        &mut self,
        request: &AuthenticatedGatewaySourceRequestV1,
    ) -> GatewayResultV1<Vec<VerifierGatewayFrameV1>> {
        self.handle_source_with_hook(request, &mut |_| Ok(()))
    }

    fn handle_source_with_hook<H>(
        &mut self,
        request: &AuthenticatedGatewaySourceRequestV1,
        hook: &mut H,
    ) -> GatewayResultV1<Vec<VerifierGatewayFrameV1>>
    where
        H: FnMut(GatewayDrivePhaseV1) -> GatewayResultV1<()>,
    {
        request.verify(&self.source_key)?;
        if request.core.session_record_sha256 != self.session_record_sha256 {
            return Err(invalid("source request addresses another session"));
        }
        if let Some(failure) = self.store.session_failure()? {
            return Ok(self.session_failed_frames(failure));
        }
        self.recover_pending_activation()?;
        self.reconcile_committed_activations()?;
        if let Some(failure) = self.refresh_session_failure()? {
            return Ok(self.session_failed_frames(failure));
        }
        match &request.core.command {
            GatewaySourceCommandV1::AdmitSession { session } => {
                if session.as_ref() != self.journal.authenticated_session() {
                    return Err(invalid("source admitted a different session record"));
                }
                Ok(vec![VerifierGatewayFrameV1::reply(
                    VerifierGatewayReplyV1::SessionAccepted {
                        session: session.clone(),
                        session_record_sha256: self.session_record_sha256.clone(),
                        pending: self.pending_summary()?,
                    },
                )])
            }
            GatewaySourceCommandV1::SubmitRound { round, .. } => {
                if round.core.protocol != VERIFIER_ROUND_PROTOCOL_V3 {
                    return Err(invalid("gateway admits only live round-v3 requests"));
                }
                self.drive_round(request, round, hook)
            }
            GatewaySourceCommandV1::FetchMissing {
                request_id,
                result_sha256,
                ordinals,
            } => self.fetch_missing(request_id, result_sha256, ordinals),
            GatewaySourceCommandV1::SourceAck { .. } => self.source_ack(request),
        }
    }

    fn drive_round<H>(
        &mut self,
        source_request: &AuthenticatedGatewaySourceRequestV1,
        request: &AuthenticatedRoundV2,
        hook: &mut H,
    ) -> GatewayResultV1<Vec<VerifierGatewayFrameV1>>
    where
        H: FnMut(GatewayDrivePhaseV1) -> GatewayResultV1<()>,
    {
        self.require_source_ack_before_new_child(request)?;
        self.validate_mirror_prediction(source_request, request)?;
        let reserve_unix_ms = self.now_unix_ms()?;
        if !self.store.has_round_admission(&request.core.request_id)? {
            request.verify_live(self.journal.session(), &self.source_key, reserve_unix_ms)?;
            self.require_current_parent_for_new_admission(request)?;
        }
        self.store
            .persist_round_admission(source_request, &self.source_key)?;
        let (result, replayed) =
            match self
                .journal
                .reserve(request, &self.fence, reserve_unix_ms)?
            {
                ReserveOutcomeV2::Stale {
                    output_height,
                    head_sha256,
                    frontier,
                    sampler_state,
                    transcript_sha256,
                } => {
                    return Ok(vec![VerifierGatewayFrameV1::reply(
                        VerifierGatewayReplyV1::Stale {
                            request_id: request.core.request_id.clone(),
                            output_height,
                            head_sha256,
                            frontier,
                            sampler_state,
                            transcript_sha256,
                        },
                    )]);
                }
                ReserveOutcomeV2::Replay(result) => (*result, true),
                ReserveOutcomeV2::Prepared(result) => {
                    let stored = self.require_stored(request)?;
                    if stored.result != *result {
                        return Err(invalid("PREPARED result differs from gateway repair store"));
                    }
                    let result = *result;
                    if let FencedActivationOutcomeV1::SessionFailed(failure) =
                        self.stage_and_commit(request, &result, hook)?
                    {
                        return Ok(self.session_failed_frames(failure));
                    }
                    (result, false)
                }
                ReserveOutcomeV2::Reserved => {
                    hook(GatewayDrivePhaseV1::Reserved)?;
                    let stored = match self.store.load_for_request(
                        request,
                        self.journal.session(),
                        &self.source_key,
                    )? {
                        Some(stored) => stored,
                        None => {
                            let evaluation =
                                match self.evaluator.evaluate(ReservedTargetRequestV1 {
                                    session: self.journal.session(),
                                    request,
                                }) {
                                    Ok(evaluation) => evaluation,
                                    Err(TargetEvaluationErrorV1::Transient(message)) => {
                                        return Err(
                                            TargetEvaluationErrorV1::Transient(message).into()
                                        );
                                    }
                                    Err(TargetEvaluationErrorV1::Permanent(_)) => {
                                        let failed_unix_ms = self.now_unix_ms()?;
                                        let failure = self.latch_session_failure(
                                            request,
                                            GatewaySessionFailureReasonV1::PermanentEvaluator,
                                            failed_unix_ms,
                                        )?;
                                        return Ok(self.session_failed_frames(failure));
                                    }
                                };
                            let completed_unix_ms = self.now_unix_ms()?;
                            if completed_unix_ms > request.core.expires_unix_ms {
                                let failure = self.latch_session_failure(
                                    request,
                                    GatewaySessionFailureReasonV1::ExpiredReserved,
                                    completed_unix_ms,
                                )?;
                                return Ok(self.session_failed_frames(failure));
                            }
                            let (result, payloads) = self.materialize_target_result(
                                request,
                                evaluation,
                                completed_unix_ms,
                            )?;
                            self.store.persist(
                                request,
                                &result,
                                &payloads,
                                self.journal.session(),
                                &self.source_key,
                            )?;
                            hook(GatewayDrivePhaseV1::Evaluated)?;
                            self.require_stored(request)?
                        }
                    };
                    let prepared_unix_ms = self.now_unix_ms()?;
                    self.journal.record_prepared(
                        request,
                        stored.result.clone(),
                        &self.fence,
                        prepared_unix_ms,
                    )?;
                    hook(GatewayDrivePhaseV1::Prepared)?;
                    if let FencedActivationOutcomeV1::SessionFailed(failure) =
                        self.stage_and_commit(request, &stored.result, hook)?
                    {
                        return Ok(self.session_failed_frames(failure));
                    }
                    (stored.result, false)
                }
            };
        self.committed_frames(&result, replayed)
    }

    fn materialize_target_result(
        &self,
        request: &AuthenticatedRoundV2,
        evaluation: RawTargetEvaluationV1,
        now_unix_ms: u64,
    ) -> GatewayResultV1<(AuthenticatedResultV2, BTreeMap<u32, Vec<u8>>)> {
        if evaluation.fragments.len() > MAX_RAW_FRAGMENT_OCCURRENCES_V1 {
            return Err(invalid(
                "raw target fragment occurrence count exceeds bounds",
            ));
        }
        let mut unique = BTreeMap::<u32, RawTargetFragmentV1>::new();
        for fragment in evaluation.fragments {
            if fragment.payload.len() as u64 > DEFAULT_VERIFIER_GATEWAY_PAYLOAD_BYTES_V1 {
                return Err(invalid(
                    "raw target fragment exceeds the gateway wire payload limit",
                ));
            }
            if let Some(existing) = unique.get(&fragment.ordinal) {
                if existing != &fragment {
                    return Err(invalid("duplicate raw target fragment differs"));
                }
                continue;
            }
            unique.insert(fragment.ordinal, fragment);
        }
        if unique.len() > MAX_FRAGMENTS_V2
            || unique
                .keys()
                .copied()
                .enumerate()
                .any(|(expected, ordinal)| ordinal as usize != expected)
        {
            return Err(invalid("raw target fragment ordinals are not contiguous"));
        }
        let descriptors = unique
            .values()
            .map(|fragment| FragmentDescriptorV2 {
                ordinal: fragment.ordinal,
                component_id: fragment.component_id.clone(),
                kind: fragment.kind,
                logical_start: fragment.logical_start,
                logical_count: fragment.logical_count,
                payload_abi: fragment.payload_abi.clone(),
                bytes_per_logical_row: fragment.bytes_per_logical_row,
                byte_len: fragment.payload.len() as u64,
                sha256: sha256_hex(&fragment.payload),
            })
            .collect::<Vec<_>>();
        let payloads = unique
            .into_iter()
            .map(|(ordinal, fragment)| (ordinal, fragment.payload))
            .collect::<BTreeMap<_, _>>();
        let result = build_result(
            request,
            self.journal.session(),
            evaluation.decision,
            evaluation.post_verify_sampler_state,
            descriptors,
            now_unix_ms,
            &self.target_signer,
        )?;
        Ok((result, payloads))
    }

    fn stage_and_commit<H>(
        &mut self,
        request: &AuthenticatedRoundV2,
        result: &AuthenticatedResultV2,
        hook: &mut H,
    ) -> GatewayResultV1<FencedActivationOutcomeV1>
    where
        H: FnMut(GatewayDrivePhaseV1) -> GatewayResultV1<()>,
    {
        let closure = self.stored_closure(request, result)?;
        let staged_unix_ms = self.now_unix_ms()?;
        let renderer =
            self.journal
                .stage_result(request, &closure, &mut self.renderer, staged_unix_ms)?;
        hook(GatewayDrivePhaseV1::Staged)?;
        let committed_unix_ms = self.now_unix_ms()?;
        let intent = FencedActivationIntentV1 {
            protocol: GATEWAY_ACTIVATION_INTENT_PROTOCOL_V1.into(),
            session_record_sha256: self.session_record_sha256.clone(),
            authority_term: request.core.authority_term,
            request_id: request.core.request_id.clone(),
            request_intent_sha256: request.intent_sha256()?,
            result_sha256: result.record_digest()?,
            expected_head_sha256: request.core.base_head_sha256.clone(),
            committed_head_sha256: result.core.new_head_sha256.clone(),
            renderer,
            created_unix_ms: committed_unix_ms,
        };
        self.store.persist_activation_intent(&intent)?;
        let committed = match activate_with_external_fence_v1(
            &mut self.journal,
            &mut self.renderer,
            &mut self.fence,
            request,
            result,
            &intent,
            committed_unix_ms,
        ) {
            Ok(committed) => committed,
            Err(VerifierGatewayErrorV1::ActivationFence(
                GatewayActivationFenceErrorV1::Conflict,
            )) => {
                let failure = self.latch_session_failure(
                    request,
                    GatewaySessionFailureReasonV1::ActivationFenceConflict,
                    committed_unix_ms,
                )?;
                return Ok(FencedActivationOutcomeV1::SessionFailed(failure));
            }
            Err(error) => return Err(error),
        };
        let activated_unix_ms = self.now_unix_ms()?;
        self.store
            .persist_activation_complete(&intent, activated_unix_ms)?;
        hook(GatewayDrivePhaseV1::Activated)?;
        if committed != *result {
            return Err(invalid("activated result differs from PREPARED result"));
        }
        Ok(FencedActivationOutcomeV1::Activated)
    }

    fn recover_pending_activation(&mut self) -> GatewayResultV1<()> {
        if !self.journal.pending_activation() {
            return Ok(());
        }
        let mut intents = self.store.pending_activation_intents()?.into_iter();
        let intent = intents
            .next()
            .ok_or_else(|| invalid("pending commit WAL lacks its activation intent"))?;
        if intents.next().is_some() {
            return Err(invalid(
                "pending commit WAL has multiple unresolved activation intents",
            ));
        }
        let stored = self
            .store
            .load(&intent.request_id, self.journal.session(), &self.source_key)?
            .ok_or_else(|| invalid("pending commit WAL lacks its durable result"))?;
        let request = stored.request;
        let result = stored.result;
        if intent.request_intent_sha256 != request.intent_sha256()?
            || intent.result_sha256 != result.record_digest()?
            || intent.expected_head_sha256 != request.core.base_head_sha256
            || intent.committed_head_sha256 != result.core.new_head_sha256
            || intent.renderer.result_sha256 != intent.result_sha256
            || intent.renderer.render_root_sha256 != result.core.fragments_sha256
        {
            return Err(invalid(
                "pending commit WAL differs from its activation intent",
            ));
        }
        let now_unix_ms = self.now_unix_ms()?;
        let recovered = recover_with_external_fence_v1(
            &mut self.journal,
            &mut self.renderer,
            &mut self.fence,
            &request,
            &result,
            &intent,
            now_unix_ms,
        );
        match recovered {
            Ok(()) => {}
            Err(VerifierGatewayErrorV1::ActivationFence(
                GatewayActivationFenceErrorV1::Conflict,
            )) => {
                self.latch_session_failure(
                    &request,
                    GatewaySessionFailureReasonV1::ActivationFenceConflict,
                    now_unix_ms,
                )?;
                return Ok(());
            }
            Err(error) => return Err(error),
        }
        let activated_unix_ms = self.now_unix_ms()?;
        self.store
            .persist_activation_complete(&intent, activated_unix_ms)
    }

    fn reconcile_committed_activations(&mut self) -> GatewayResultV1<()> {
        for intent in self.store.pending_activation_intents()? {
            let Some(result) = self.journal.completed_result(&intent.request_id) else {
                // An intent is persisted before acquiring the external permit.
                // No local completion means the exact request may safely retry.
                continue;
            };
            if result.record_digest()? != intent.result_sha256
                || result.core.new_head_sha256 != intent.committed_head_sha256
                || result.core.base_head_sha256 != intent.expected_head_sha256
                || result.core.fragments_sha256 != intent.renderer.render_root_sha256
            {
                return Err(invalid(
                    "completed result differs from durable activation intent",
                ));
            }
            let now_unix_ms = self.now_unix_ms()?;
            match self.fence.reconcile_committed(&intent, now_unix_ms) {
                Ok(()) => {
                    self.renderer
                        .activate(&intent.renderer)
                        .map_err(VerifierV2Error::Renderer)?;
                    let activated_unix_ms = self.now_unix_ms()?;
                    self.store
                        .persist_activation_complete(&intent, activated_unix_ms)?;
                }
                Err(GatewayActivationFenceErrorV1::Unavailable) => {
                    return Err(GatewayActivationFenceErrorV1::Unavailable.into());
                }
                Err(GatewayActivationFenceErrorV1::Conflict) => {
                    let request = self
                        .store
                        .load(&intent.request_id, self.journal.session(), &self.source_key)?
                        .ok_or_else(|| invalid("conflicting activation lost its request"))?
                        .request;
                    self.latch_session_failure(
                        &request,
                        GatewaySessionFailureReasonV1::ActivationFenceConflict,
                        now_unix_ms,
                    )?;
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    fn refresh_session_failure(&mut self) -> GatewayResultV1<Option<GatewaySessionFailureV1>> {
        if let Some(failure) = self.store.session_failure()? {
            return Ok(Some(failure));
        }
        let Some(PendingRoundV2::Reserved { request }) = self.journal.pending_round()? else {
            return Ok(None);
        };
        if self
            .store
            .load_for_request(&request, self.journal.session(), &self.source_key)?
            .is_some()
        {
            // Evaluation completed and its exact result is durable. Crossing
            // the admission deadline does not authorize a sibling or resample.
            return Ok(None);
        }
        let now_unix_ms = self.now_unix_ms()?;
        if now_unix_ms <= request.core.expires_unix_ms {
            return Ok(None);
        }
        self.latch_session_failure(
            &request,
            GatewaySessionFailureReasonV1::ExpiredReserved,
            now_unix_ms,
        )
        .map(Some)
    }

    fn latch_session_failure(
        &self,
        request: &AuthenticatedRoundV2,
        reason: GatewaySessionFailureReasonV1,
        failed_unix_ms: u64,
    ) -> GatewayResultV1<GatewaySessionFailureV1> {
        if let Some(existing) = self.store.session_failure()? {
            if existing.reason == reason
                && existing.request_id == request.core.request_id
                && existing.request_intent_sha256 == request.intent_sha256()?
                && existing.base_head_sha256 == request.core.base_head_sha256
                && existing.request_expires_unix_ms == request.core.expires_unix_ms
            {
                return Ok(existing);
            }
            return Err(invalid("durable session failure changed identity"));
        }
        let failure = GatewaySessionFailureV1 {
            reason,
            request_id: request.core.request_id.clone(),
            request_intent_sha256: request.intent_sha256()?,
            base_head_sha256: request.core.base_head_sha256.clone(),
            request_expires_unix_ms: request.core.expires_unix_ms,
            failed_unix_ms,
        };
        self.store.persist_session_failure(&failure)?;
        Ok(failure)
    }

    fn session_failed_frames(
        &self,
        failure: GatewaySessionFailureV1,
    ) -> Vec<VerifierGatewayFrameV1> {
        vec![VerifierGatewayFrameV1::reply(
            VerifierGatewayReplyV1::SessionFailed {
                session_record_sha256: self.session_record_sha256.clone(),
                failure,
            },
        )]
    }

    fn stored_closure(
        &self,
        request: &AuthenticatedRoundV2,
        result: &AuthenticatedResultV2,
    ) -> GatewayResultV1<VerifiedClosureV2> {
        let stored = self.require_stored(request)?;
        if stored.result != *result {
            return Err(invalid("stored closure belongs to another result"));
        }
        let payloads = self.store.payloads(&stored)?;
        let mut assembler = FragmentAssemblerV2::new(result, request, self.journal.session())?;
        for (ordinal, payload) in payloads.into_iter().rev() {
            assembler.insert(ordinal, payload)?;
        }
        Ok(assembler.finish()?)
    }

    fn require_stored(
        &self,
        request: &AuthenticatedRoundV2,
    ) -> GatewayResultV1<StoredGatewayResultV1> {
        self.store
            .load_for_request(request, self.journal.session(), &self.source_key)?
            .ok_or_else(|| invalid("durable gateway result or fragments are absent"))
    }

    fn committed_frames(
        &self,
        result: &AuthenticatedResultV2,
        replayed: bool,
    ) -> GatewayResultV1<Vec<VerifierGatewayFrameV1>> {
        if self.journal.completed_result(&result.core.request_id) != Some(result) {
            return Err(invalid("gateway cannot reply before result activation"));
        }
        let stored = self
            .store
            .load(
                &result.core.request_id,
                self.journal.session(),
                &self.source_key,
            )?
            .ok_or_else(|| invalid("committed result is absent from repair store"))?;
        if stored.result != *result {
            return Err(invalid("committed result differs from repair store"));
        }
        let admission = self
            .store
            .round_admission(&result.core.request_id, &self.source_key)?;
        let GatewaySourceCommandV1::SubmitRound { round, .. } = &admission.core.command else {
            return Err(invalid("committed result lacks a round admission"));
        };
        if round.intent_sha256()? != result.core.request_intent_sha256 {
            return Err(invalid("committed source admission differs from result"));
        }
        let result_sha256 = result.record_digest()?;
        let source_admission_sha256 = admission.record_digest()?;
        let mut frames = vec![VerifierGatewayFrameV1::reply(
            VerifierGatewayReplyV1::RoundCommitted {
                result: Box::new(result.clone()),
                result_sha256: result_sha256.clone(),
                source_admission_sha256,
                replayed,
            },
        )];
        for descriptor in &result.core.fragments {
            frames.push(VerifierGatewayFrameV1::fragment(
                VerifierGatewayReplyV1::Fragment {
                    request_id: result.core.request_id.clone(),
                    result_sha256: result_sha256.clone(),
                    descriptor: descriptor.clone(),
                },
                self.store.read_fragment(descriptor)?,
            ));
        }
        Ok(frames)
    }

    fn fetch_missing(
        &self,
        request_id: &str,
        requested_result_sha256: &str,
        ordinals: &[u32],
    ) -> GatewayResultV1<Vec<VerifierGatewayFrameV1>> {
        let stored = self
            .store
            .load(request_id, self.journal.session(), &self.source_key)?
            .ok_or_else(|| invalid("missing-fragment request names an unknown result"))?;
        if self.journal.completed_result(request_id) != Some(&stored.result) {
            return Err(invalid(
                "missing-fragment repair requires an activated completion",
            ));
        }
        let result_sha256 = stored.result.record_digest()?;
        if result_sha256 != requested_result_sha256 {
            return Err(invalid("missing-fragment request changed result identity"));
        }
        let mut frames = Vec::with_capacity(ordinals.len() + 1);
        for ordinal in ordinals {
            let descriptor = stored
                .result
                .core
                .fragments
                .get(*ordinal as usize)
                .filter(|descriptor| descriptor.ordinal == *ordinal)
                .ok_or_else(|| invalid("missing-fragment ordinal is outside result"))?;
            frames.push(VerifierGatewayFrameV1::fragment(
                VerifierGatewayReplyV1::Fragment {
                    request_id: request_id.into(),
                    result_sha256: result_sha256.clone(),
                    descriptor: descriptor.clone(),
                },
                self.store.read_fragment(descriptor)?,
            ));
        }
        frames.push(VerifierGatewayFrameV1::reply(
            VerifierGatewayReplyV1::FetchComplete {
                request_id: request_id.into(),
                result_sha256,
                delivered_ordinals: ordinals.to_vec(),
            },
        ));
        Ok(frames)
    }

    fn source_ack(
        &mut self,
        source_request: &AuthenticatedGatewaySourceRequestV1,
    ) -> GatewayResultV1<Vec<VerifierGatewayFrameV1>> {
        let GatewaySourceCommandV1::SourceAck { receipt } = &source_request.core.command else {
            return Err(invalid("source ACK handler received another command"));
        };
        let request_id = &receipt.request_id;
        let requested_result_sha256 = &receipt.result_sha256;
        let applied_head_sha256 = &receipt.applied_head_sha256;
        let stored = self
            .store
            .load(request_id, self.journal.session(), &self.source_key)?
            .ok_or_else(|| invalid("source ACK names an unknown result"))?;
        if self.journal.completed_result(request_id) != Some(&stored.result) {
            return Err(invalid("source ACK requires an activated completion"));
        }
        let result_sha256 = stored.result.record_digest()?;
        let acknowledged_unix_ms = self.now_unix_ms()?;
        if result_sha256 != *requested_result_sha256
            || stored.result.core.new_head_sha256 != *applied_head_sha256
            || stored.result.core.fragments_sha256 != receipt.render_root_sha256
            || stored.result.core.new_output_height != receipt.installed_output_height
            || stored.result.core.new_evaluated_tokens_sha256 != receipt.installed_transcript_sha256
            || receipt.durable_unix_ms < stored.result.core.completed_unix_ms
            || receipt.durable_unix_ms
                > acknowledged_unix_ms.saturating_add(MAX_SOURCE_RECEIPT_CLOCK_SKEW_MS_V1)
        {
            return Err(invalid(
                "source ACK changed result, head, transcript, or durable render identity",
            ));
        }
        self.journal
            .acknowledge(request_id, applied_head_sha256, acknowledged_unix_ms)?;
        self.store
            .persist_source_ack(source_request, &self.source_key)?;
        Ok(vec![VerifierGatewayFrameV1::reply(
            VerifierGatewayReplyV1::SourceAcked {
                request_id: request_id.into(),
                result_sha256,
                applied_head_sha256: applied_head_sha256.into(),
            },
        )])
    }

    fn pending_summary(&self) -> GatewayResultV1<Option<PendingRoundSummaryV1>> {
        self.journal
            .pending_round()?
            .map(|pending| match pending {
                PendingRoundV2::Reserved { request } => Ok(PendingRoundSummaryV1::Reserved {
                    request: Box::new(request),
                }),
                PendingRoundV2::Prepared { request, result } => {
                    Ok(PendingRoundSummaryV1::Prepared {
                        request: Box::new(request),
                        result: Box::new(result),
                    })
                }
                PendingRoundV2::Staged {
                    request,
                    result,
                    renderer,
                } => Ok(PendingRoundSummaryV1::Staged {
                    request: Box::new(request),
                    result: Box::new(result),
                    renderer,
                }),
            })
            .transpose()
    }

    fn validate_mirror_prediction(
        &self,
        source_request: &AuthenticatedGatewaySourceRequestV1,
        request: &AuthenticatedRoundV2,
    ) -> GatewayResultV1<()> {
        let GatewaySourceCommandV1::SubmitRound {
            round,
            mirror_prediction,
        } = &source_request.core.command
        else {
            return Err(invalid("Mirror admission is not a round submission"));
        };
        if round.as_ref() != request {
            return Err(invalid("Mirror admission carries another round"));
        }
        let Some(prediction) = mirror_prediction else {
            return Ok(());
        };
        if self.journal.session().coupling_policy != CouplingPolicyV2::Greedy
            || request.core.intent != crate::verifier_v2::RoundIntentV2::Verify
            || prediction.predicted_frontier_token >= self.journal.session().vocab_size
        {
            return Err(invalid(
                "Mirror prediction requires a live greedy verification round",
            ));
        }
        Ok(())
    }

    fn require_current_parent_for_new_admission(
        &self,
        request: &AuthenticatedRoundV2,
    ) -> GatewayResultV1<()> {
        let (output_height, head_sha256, frontier, sampler_state) = self.journal.current_head();
        if request.core.base_output_height != output_height
            || request.core.base_head_sha256 != head_sha256
            || request.core.frontier_in != *frontier
            || request.core.base_sampler_state != *sampler_state
            || request.core.base_evaluated_tokens != self.journal.evaluated_tokens()
            || request.core.base_tokens_sha256 != self.journal.current_transcript_sha256()
        {
            return Err(invalid(
                "new source admission does not name the current parent",
            ));
        }
        if self.journal.pending_round()?.is_some() {
            return Err(invalid(
                "new source admission cannot replace the parent's pending child",
            ));
        }
        Ok(())
    }

    fn require_source_ack_before_new_child(
        &self,
        request: &AuthenticatedRoundV2,
    ) -> GatewayResultV1<()> {
        let (_, current_head, _, _) = self.journal.current_head();
        if request.core.base_head_sha256 != current_head
            || current_head == self.journal.session().genesis_head_sha256
        {
            return Ok(());
        }
        let same_pending = self
            .journal
            .pending_round()?
            .is_some_and(|pending| match pending {
                PendingRoundV2::Reserved { request: pending }
                | PendingRoundV2::Prepared {
                    request: pending, ..
                }
                | PendingRoundV2::Staged {
                    request: pending, ..
                } => pending == *request,
            });
        if !same_pending && !self.store.has_source_ack(current_head, &self.source_key)? {
            return Err(invalid(
                "source must durably ACK the current head before a child is admitted",
            ));
        }
        Ok(())
    }

    fn now_unix_ms(&mut self) -> GatewayResultV1<u64> {
        self.clock
            .now_unix_ms()
            .map_err(VerifierGatewayErrorV1::Clock)
    }
}

/// Source-side bounded adapter around the V2 fragment assembler. It accepts
/// exact duplicates and arbitrary delivery order, and generates authenticated
/// FetchMissing/SourceAck control requests for reconnect repair.
pub struct VerifierSourceAssemblerV1 {
    request_id: String,
    result_sha256: String,
    request: AuthenticatedRoundV2,
    result: AuthenticatedResultV2,
    assembler: FragmentAssemblerV2,
}

/// Source integration boundary. Returning success asserts that the exact
/// closure and its reconstructible state manifest have been installed and
/// fsynced. Network receipt alone must never implement this trait.
pub trait DurableSourceInstallerV1 {
    fn install_durably(
        &mut self,
        request: &AuthenticatedRoundV2,
        result: &AuthenticatedResultV2,
        closure: &VerifiedClosureV2,
    ) -> std::result::Result<SourceInstallReceiptV1, String>;
}

/// Typestate available only after a source installer returned a durable,
/// result-bound receipt. It may produce repeatable ACKs for reconnect retry.
pub struct DurablyInstalledSourceResultV1 {
    receipt: SourceInstallReceiptV1,
}

impl DurablyInstalledSourceResultV1 {
    pub fn receipt(&self) -> &SourceInstallReceiptV1 {
        &self.receipt
    }

    pub fn source_ack_request(
        &self,
        session_record_sha256: &str,
        key: &MacKey,
    ) -> GatewayResultV1<AuthenticatedGatewaySourceRequestV1> {
        Ok(AuthenticatedGatewaySourceRequestV1::sign(
            session_record_sha256.into(),
            GatewaySourceCommandV1::SourceAck {
                receipt: self.receipt.clone(),
            },
            key,
        )?)
    }
}

impl VerifierSourceAssemblerV1 {
    pub fn new(verified: VerifiedGatewayRoundResultV1) -> GatewayResultV1<Self> {
        let (session, request, result) = verified.into_parts();
        let result_sha256 = result.record_digest()?;
        let assembler = FragmentAssemblerV2::new(&result, &request, &session.core)?;
        Ok(Self {
            request_id: request.core.request_id.clone(),
            result_sha256,
            request,
            result,
            assembler,
        })
    }

    pub fn insert_frame(&mut self, frame: &VerifierGatewayFrameV1) -> GatewayResultV1<bool> {
        let VerifierGatewayMessageV1::Gateway { reply } = &frame.message else {
            return Err(invalid("source assembler expected a fragment frame"));
        };
        let VerifierGatewayReplyV1::Fragment {
            request_id,
            result_sha256,
            descriptor,
        } = reply.as_ref()
        else {
            return Err(invalid("source assembler expected a fragment frame"));
        };
        if request_id != &self.request_id
            || result_sha256 != &self.result_sha256
            || self.result.core.fragments.get(descriptor.ordinal as usize) != Some(descriptor)
        {
            return Err(invalid("source fragment addresses another result"));
        }
        Ok(self
            .assembler
            .insert(descriptor.ordinal, frame.payload.clone())?)
    }

    pub fn missing_ordinals(&self) -> Vec<u32> {
        self.assembler.missing_ordinals()
    }

    pub fn fetch_missing_request(
        &self,
        session_record_sha256: &str,
        key: &MacKey,
    ) -> GatewayResultV1<AuthenticatedGatewaySourceRequestV1> {
        Ok(AuthenticatedGatewaySourceRequestV1::sign(
            session_record_sha256.into(),
            GatewaySourceCommandV1::FetchMissing {
                request_id: self.request_id.clone(),
                result_sha256: self.result_sha256.clone(),
                ordinals: self.missing_ordinals(),
            },
            key,
        )?)
    }

    pub fn install_durably<I: DurableSourceInstallerV1>(
        self,
        installer: &mut I,
    ) -> GatewayResultV1<DurablyInstalledSourceResultV1> {
        let closure = self.assembler.finish()?;
        let receipt = installer
            .install_durably(&self.request, &self.result, &closure)
            .map_err(VerifierGatewayErrorV1::SourceInstall)?;
        validate_source_install_receipt(
            &receipt,
            &self.request_id,
            &self.result_sha256,
            &self.result,
        )?;
        Ok(DurablyInstalledSourceResultV1 { receipt })
    }

    pub fn finish(self) -> GatewayResultV1<VerifiedClosureV2> {
        Ok(self.assembler.finish()?)
    }
}

fn validate_source_install_receipt(
    receipt: &SourceInstallReceiptV1,
    request_id: &str,
    result_sha256: &str,
    result: &AuthenticatedResultV2,
) -> GatewayResultV1<()> {
    validate_digest("source durable state", &receipt.source_state_sha256)?;
    validate_digest(
        "source durable transcript",
        &receipt.installed_transcript_sha256,
    )?;
    if receipt.protocol != VERIFIER_SOURCE_INSTALL_RECEIPT_PROTOCOL_V1
        || receipt.request_id != request_id
        || receipt.result_sha256 != result_sha256
        || receipt.applied_head_sha256 != result.core.new_head_sha256
        || receipt.render_root_sha256 != result.core.fragments_sha256
        || receipt.source_state_bytes == 0
        || receipt.installed_output_height != result.core.new_output_height
        || receipt.installed_transcript_sha256 != result.core.new_evaluated_tokens_sha256
        || receipt.durable_unix_ms < result.core.completed_unix_ms
    {
        return Err(invalid(
            "source durable-install receipt differs from assembled result",
        ));
    }
    Ok(())
}

fn canonical<T: Serialize>(value: &T) -> GatewayResultV1<Vec<u8>> {
    canonical_json(value).map_err(|error| invalid(error.to_string()))
}

fn read_canonical<T: serde::de::DeserializeOwned + Serialize>(
    path: &Path,
    limit: usize,
) -> GatewayResultV1<T> {
    validate_private_file(path, None)?;
    let metadata = fs::metadata(path)?;
    let length = usize::try_from(metadata.len()).map_err(|_| invalid("file length overflow"))?;
    if length == 0 || length > limit {
        return Err(invalid("gateway store JSON length exceeds bounds"));
    }
    let bytes = fs::read(path)?;
    decode_canonical_json(&bytes, limit).map_err(|error| invalid(error.to_string()))
}

fn write_exclusive(path: &Path, bytes: &[u8]) -> GatewayResultV1<()> {
    let mut options = OpenOptions::new();
    options
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
    let mut file = options.open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    if let Some(parent) = path.parent() {
        sync_directory(parent)?;
    }
    Ok(())
}

fn open_lock(root: &Path) -> GatewayResultV1<File> {
    let path = root.join("LOCK");
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)?;
    let status = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if status != 0 {
        return Err(invalid("gateway store is already locked"));
    }
    Ok(file)
}

fn sync_directory(path: &Path) -> GatewayResultV1<()> {
    File::open(path)?.sync_all()?;
    Ok(())
}

fn validate_private_directory(path: &Path) -> GatewayResultV1<()> {
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_dir()
        || metadata.file_type().is_symlink()
        || metadata.permissions().mode() & 0o077 != 0
    {
        return Err(invalid("gateway store directory mode differs"));
    }
    Ok(())
}

fn validate_private_file(path: &Path, expected_len: Option<u64>) -> GatewayResultV1<()> {
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.is_file()
        || metadata.file_type().is_symlink()
        || metadata.permissions().mode() & 0o077 != 0
        || expected_len.is_some_and(|length| metadata.len() != length)
    {
        return Err(invalid("gateway store file mode or length differs"));
    }
    Ok(())
}

fn validate_identifier(label: &str, value: &str) -> GatewayResultV1<()> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"-_.:".contains(&byte))
    {
        return Err(invalid(format!("{label} identifier differs")));
    }
    Ok(())
}

fn validate_digest(label: &str, value: &str) -> GatewayResultV1<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(invalid(format!("{label} digest differs")));
    }
    Ok(())
}

fn validate_session_failure(failure: &GatewaySessionFailureV1) -> GatewayResultV1<()> {
    validate_identifier("failed request", &failure.request_id)?;
    validate_digest("failed request intent", &failure.request_intent_sha256)?;
    validate_digest("failed request parent", &failure.base_head_sha256)?;
    if failure.request_expires_unix_ms == 0 || failure.failed_unix_ms == 0 {
        return Err(invalid("failed session timestamps differ"));
    }
    if failure.reason == GatewaySessionFailureReasonV1::ExpiredReserved
        && failure.failed_unix_ms <= failure.request_expires_unix_ms
    {
        return Err(invalid("expired reservation failure predates expiry"));
    }
    Ok(())
}

fn validate_activation_intent(
    intent: &FencedActivationIntentV1,
    session_record_sha256: &str,
) -> GatewayResultV1<()> {
    if intent.protocol != GATEWAY_ACTIVATION_INTENT_PROTOCOL_V1
        || intent.session_record_sha256 != session_record_sha256
        || intent.authority_term == 0
        || intent.created_unix_ms == 0
    {
        return Err(invalid("activation intent protocol or authority differs"));
    }
    validate_identifier("activation request", &intent.request_id)?;
    validate_digest("activation request intent", &intent.request_intent_sha256)?;
    validate_digest("activation result", &intent.result_sha256)?;
    validate_digest("activation expected head", &intent.expected_head_sha256)?;
    validate_digest("activation committed head", &intent.committed_head_sha256)?;
    validate_digest("activation renderer result", &intent.renderer.result_sha256)?;
    validate_digest(
        "activation renderer root",
        &intent.renderer.render_root_sha256,
    )?;
    validate_identifier("activation renderer token", &intent.renderer.stage_token)?;
    if intent.expected_head_sha256 == intent.committed_head_sha256 {
        return Err(invalid("activation intent does not advance the head"));
    }
    if intent.renderer.result_sha256 != intent.result_sha256
        || intent.renderer.stage_token != intent.request_id
        || intent.renderer.staged_unix_ms == 0
    {
        return Err(invalid("activation renderer receipt differs"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use tempfile::TempDir;

    use super::*;
    use crate::verifier_gateway_codec::{
        read_verifier_gateway_frame_v1, write_verifier_gateway_frame_v1,
        MirrorPredictionCommitmentV1, VerifierGatewayFrameLimitsV1,
    };
    use crate::verifier_v2::{
        AuthenticatedSessionV2, CompositeIdentityV2, CouplingPolicyV2, DraftEvidenceV2,
        FragmentCoverageV2, FragmentRequirementV2, FrontierV2, RendererStageReceiptV2, RoundCoreV2,
        RoundIntentV2, SessionCoreV2, VERIFIER_ROUND_PROTOCOL_V3, VERIFIER_SESSION_PROTOCOL_V2,
    };

    const NOW: u64 = 10_000;
    const TARGET_SEED: [u8; 32] = [0x41; 32];

    fn digest(byte: u8) -> String {
        format!("{byte:02x}").repeat(32)
    }

    fn token_digest(tokens: &[u32]) -> String {
        let mut bytes = Vec::with_capacity(tokens.len() * 4);
        for token in tokens {
            bytes.extend_from_slice(&token.to_le_bytes());
        }
        sha256_hex(&bytes)
    }

    fn signed_session(key: &MacKey, signer: &TargetSigningKeyV2) -> AuthenticatedSessionV2 {
        AuthenticatedSessionV2::sign(
            SessionCoreV2 {
                protocol: VERIFIER_SESSION_PROTOCOL_V2.into(),
                session_id: "session-a".into(),
                client_incarnation: "client-a".into(),
                log_writer_authority_id: "mac-a".into(),
                authority_lease_id: "lease-a".into(),
                authority_term: 7,
                target_executor_id: "gx10-a".into(),
                target_public_key: signer.public_key(),
                created_unix_ms: NOW,
                expires_unix_ms: NOW + 1_000_000,
                hmac_key_id: "request-key-a".into(),
                hmac_key_epoch: 3,
                identity: CompositeIdentityV2 {
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
                    portable_kv_abi: "kvpack-test-v1".into(),
                },
                coupling_policy: CouplingPolicyV2::Greedy,
                sampler_config_sha256: digest(12),
                vocab_size: 1_024,
                max_context_tokens: 128,
                max_drafts: 8,
                max_total_fragment_bytes: 1 << 20,
                initial_evaluated_tokens: vec![1, 2, 3],
                initial_output_height: 0,
                initial_frontier: FrontierV2::Open {
                    token: 101,
                    output_ordinal: 0,
                },
                initial_sampler_state: SamplerStateV2::stateless(),
                fragment_requirements: vec![FragmentRequirementV2 {
                    component_id: "target-hidden".into(),
                    kind: FragmentKindV2::TargetHidden,
                    coverage: FragmentCoverageV2::CommittedInputs,
                    payload_abi: "u32-row-v1".into(),
                    bytes_per_logical_row: 4,
                    required: true,
                }],
                genesis_head_sha256: digest(0),
            },
            key,
        )
        .unwrap()
    }

    fn round(
        journal: &DurableVerifierJournalV2,
        key: &MacKey,
        request_id: &str,
    ) -> AuthenticatedRoundV2 {
        let (output_height, head, frontier, sampler) = journal.current_head();
        AuthenticatedRoundV2::sign(
            RoundCoreV2 {
                protocol: VERIFIER_ROUND_PROTOCOL_V3.into(),
                intent: RoundIntentV2::Verify,
                session_id: journal.session().session_id.clone(),
                session_genesis_sha256: journal.session().genesis_head_sha256.clone(),
                authority_term: journal.session().authority_term,
                request_id: request_id.into(),
                created_unix_ms: NOW + 1,
                expires_unix_ms: NOW + 500_000,
                hmac_key_id: journal.session().hmac_key_id.clone(),
                hmac_key_epoch: journal.session().hmac_key_epoch,
                base_output_height: output_height,
                base_head_sha256: head.into(),
                base_evaluated_tokens: journal.evaluated_tokens().to_vec(),
                base_tokens_sha256: token_digest(journal.evaluated_tokens()),
                frontier_in: frontier.clone(),
                draft_tokens: vec![102, 103],
                draft_evidence: DraftEvidenceV2::None,
                base_sampler_state: sampler.clone(),
                post_draft_sampler_state: sampler.clone(),
            },
            journal.session(),
            key,
        )
        .unwrap()
    }

    fn capabilities(session: &SessionCoreV2) -> TargetEvaluatorCapabilitiesV1 {
        TargetEvaluatorCapabilitiesV1 {
            target_executor_id: session.target_executor_id.clone(),
            semantic_product_sha256: session.identity.semantic_product_sha256.clone(),
            target_checkpoint_sha256: session.identity.target_checkpoint_sha256.clone(),
            target_engine_sha256: session.identity.target_engine_sha256.clone(),
            target_sampler_sha256: session.identity.target_sampler_sha256.clone(),
            target_genesis_root_sha256: session.identity.target_genesis_root_sha256.clone(),
            tokenizer_sha256: session.identity.tokenizer_sha256.clone(),
            vocabulary_sha256: session.identity.vocabulary_sha256.clone(),
            portable_kv_abi: session.identity.portable_kv_abi.clone(),
            sampler_config_sha256: session.sampler_config_sha256.clone(),
            coupling_policy: session.coupling_policy.clone(),
            vocab_size: session.vocab_size,
            max_context_tokens: session.max_context_tokens,
            max_drafts: session.max_drafts,
            max_total_fragment_bytes: session.max_total_fragment_bytes,
            fragment_requirements: session.fragment_requirements.clone(),
        }
    }

    #[derive(Clone)]
    struct SharedTestState {
        evaluator_calls: Arc<AtomicUsize>,
        stage_calls: Arc<AtomicUsize>,
        activate_calls: Arc<AtomicUsize>,
        live_fence: Arc<AtomicBool>,
        activation_held: Arc<AtomicBool>,
        activation_head: Arc<Mutex<String>>,
        fail_fence_publish_once: Arc<AtomicBool>,
        permanent_evaluator_failure: Arc<AtomicBool>,
        full_acceptance: Arc<AtomicBool>,
        fail_activation_once: Arc<AtomicBool>,
        clock: Arc<AtomicU64>,
        activated: Arc<Mutex<BTreeSet<String>>>,
    }

    impl Default for SharedTestState {
        fn default() -> Self {
            Self {
                evaluator_calls: Arc::new(AtomicUsize::new(0)),
                stage_calls: Arc::new(AtomicUsize::new(0)),
                activate_calls: Arc::new(AtomicUsize::new(0)),
                live_fence: Arc::new(AtomicBool::new(true)),
                activation_held: Arc::new(AtomicBool::new(false)),
                activation_head: Arc::new(Mutex::new(digest(0))),
                fail_fence_publish_once: Arc::new(AtomicBool::new(false)),
                permanent_evaluator_failure: Arc::new(AtomicBool::new(false)),
                full_acceptance: Arc::new(AtomicBool::new(false)),
                fail_activation_once: Arc::new(AtomicBool::new(false)),
                clock: Arc::new(AtomicU64::new(NOW + 2)),
                activated: Arc::new(Mutex::new(BTreeSet::new())),
            }
        }
    }

    struct TestEvaluator {
        capabilities: TargetEvaluatorCapabilitiesV1,
        shared: SharedTestState,
    }

    impl TargetEvaluatorV1 for TestEvaluator {
        fn capabilities(&self) -> &TargetEvaluatorCapabilitiesV1 {
            &self.capabilities
        }

        fn evaluate(
            &mut self,
            reserved: ReservedTargetRequestV1<'_>,
        ) -> std::result::Result<RawTargetEvaluationV1, TargetEvaluationErrorV1> {
            self.shared.evaluator_calls.fetch_add(1, Ordering::SeqCst);
            if self
                .shared
                .permanent_evaluator_failure
                .load(Ordering::SeqCst)
            {
                return Err(TargetEvaluationErrorV1::Permanent(
                    "injected permanent evaluator failure".into(),
                ));
            }
            let request = reserved.request();
            let base = request.core.base_output_height;
            let second = RawTargetFragmentV1 {
                ordinal: 1,
                component_id: "target-hidden".into(),
                kind: FragmentKindV2::TargetHidden,
                logical_start: base + 1,
                logical_count: 1,
                payload_abi: "u32-row-v1".into(),
                bytes_per_logical_row: 4,
                payload: 11u32.to_le_bytes().to_vec(),
            };
            let first = RawTargetFragmentV1 {
                ordinal: 0,
                component_id: "target-hidden".into(),
                kind: FragmentKindV2::TargetHidden,
                logical_start: base,
                logical_count: 1,
                payload_abi: "u32-row-v1".into(),
                bytes_per_logical_row: 4,
                payload: 10u32.to_le_bytes().to_vec(),
            };
            let full_acceptance = self.shared.full_acceptance.load(Ordering::SeqCst);
            let accepted_drafts = if full_acceptance {
                request.core.draft_tokens.len() as u32
            } else {
                1
            };
            let mut fragments = vec![second.clone(), first, second];
            if full_acceptance {
                fragments.insert(
                    0,
                    RawTargetFragmentV1 {
                        ordinal: 2,
                        component_id: "target-hidden".into(),
                        kind: FragmentKindV2::TargetHidden,
                        logical_start: base + 2,
                        logical_count: 1,
                        payload_abi: "u32-row-v1".into(),
                        bytes_per_logical_row: 4,
                        payload: 12u32.to_le_bytes().to_vec(),
                    },
                );
            }
            Ok(RawTargetEvaluationV1 {
                decision: ResultDecisionV2::Open {
                    accepted_drafts,
                    frontier_out: 777,
                },
                post_verify_sampler_state: request.core.base_sampler_state.clone(),
                // Out of order with an exact duplicate: normalization must be
                // deterministic and duplicate-insensitive.
                fragments,
            })
        }
    }

    struct TestRenderer {
        shared: SharedTestState,
    }

    struct TestSourceInstaller {
        receipt: Option<SourceInstallReceiptV1>,
    }

    impl DurableSourceInstallerV1 for TestSourceInstaller {
        fn install_durably(
            &mut self,
            request: &AuthenticatedRoundV2,
            result: &AuthenticatedResultV2,
            closure: &VerifiedClosureV2,
        ) -> std::result::Result<SourceInstallReceiptV1, String> {
            if request.core.request_id != result.core.request_id
                || closure.fragments().len() != result.core.fragments.len()
            {
                return Err("source installer received another closure".into());
            }
            self.receipt
                .take()
                .ok_or_else(|| "source installer receipt was already consumed".into())
        }
    }

    impl VerifierRendererV2 for TestRenderer {
        fn stage(
            &mut self,
            result: &AuthenticatedResultV2,
            closure: &VerifiedClosureV2,
            now_unix_ms: u64,
        ) -> std::result::Result<RendererStageReceiptV2, String> {
            if closure.fragments().len() != result.core.fragments.len() {
                return Err("closure length differs".into());
            }
            self.shared.stage_calls.fetch_add(1, Ordering::SeqCst);
            Ok(RendererStageReceiptV2 {
                result_sha256: result.record_digest().map_err(|error| error.to_string())?,
                render_root_sha256: result.core.fragments_sha256.clone(),
                stage_token: result.core.request_id.clone(),
                staged_unix_ms: now_unix_ms,
            })
        }

        fn activate(
            &mut self,
            receipt: &RendererStageReceiptV2,
        ) -> std::result::Result<(), String> {
            self.shared.activate_calls.fetch_add(1, Ordering::SeqCst);
            if self
                .shared
                .fail_activation_once
                .swap(false, Ordering::SeqCst)
            {
                return Err("injected activation crash".into());
            }
            self.shared
                .activated
                .lock()
                .unwrap()
                .insert(receipt.result_sha256.clone());
            Ok(())
        }
    }

    struct TestFence {
        shared: SharedTestState,
    }

    struct TestActivationPermit {
        shared: SharedTestState,
        intent: FencedActivationIntentV1,
    }

    impl Drop for TestActivationPermit {
        fn drop(&mut self) {
            self.shared.activation_held.store(false, Ordering::SeqCst);
        }
    }

    impl AuthorityFenceV2 for TestActivationPermit {
        fn permits(
            &self,
            session_id: &str,
            log_writer_authority_id: &str,
            authority_lease_id: &str,
            authority_term: u64,
            target_executor_id: &str,
            _now_unix_ms: u64,
        ) -> bool {
            self.shared.activation_held.load(Ordering::SeqCst)
                && session_id == "session-a"
                && log_writer_authority_id == "mac-a"
                && authority_lease_id == "lease-a"
                && authority_term == self.intent.authority_term
                && target_executor_id == "gx10-a"
        }
    }

    impl GatewayActivationPermitV1 for TestActivationPermit {
        fn publish_committed(
            &mut self,
            intent: &FencedActivationIntentV1,
        ) -> std::result::Result<(), GatewayActivationFenceErrorV1> {
            if intent != &self.intent {
                return Err(GatewayActivationFenceErrorV1::Conflict);
            }
            if self
                .shared
                .fail_fence_publish_once
                .swap(false, Ordering::SeqCst)
            {
                return Err(GatewayActivationFenceErrorV1::Unavailable);
            }
            let mut head = self.shared.activation_head.lock().unwrap();
            if *head == intent.committed_head_sha256 {
                return Ok(());
            }
            if *head != intent.expected_head_sha256 {
                return Err(GatewayActivationFenceErrorV1::Conflict);
            }
            head.clone_from(&intent.committed_head_sha256);
            Ok(())
        }
    }

    impl AuthorityFenceV2 for TestFence {
        fn permits(
            &self,
            session_id: &str,
            log_writer_authority_id: &str,
            authority_lease_id: &str,
            authority_term: u64,
            target_executor_id: &str,
            _now_unix_ms: u64,
        ) -> bool {
            self.shared.live_fence.load(Ordering::SeqCst)
                && session_id == "session-a"
                && log_writer_authority_id == "mac-a"
                && authority_lease_id == "lease-a"
                && authority_term == 7
                && target_executor_id == "gx10-a"
        }
    }

    impl GatewayActivationFenceV1 for TestFence {
        fn acquire_activation<'a>(
            &'a mut self,
            intent: &FencedActivationIntentV1,
            _now_unix_ms: u64,
        ) -> std::result::Result<
            Box<dyn GatewayActivationPermitV1 + 'a>,
            GatewayActivationFenceErrorV1,
        > {
            if !self.shared.live_fence.load(Ordering::SeqCst) || intent.authority_term != 7 {
                return Err(GatewayActivationFenceErrorV1::Unavailable);
            }
            if self
                .shared
                .activation_held
                .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                .is_err()
            {
                return Err(GatewayActivationFenceErrorV1::Unavailable);
            }
            let head = self.shared.activation_head.lock().unwrap().clone();
            if head != intent.expected_head_sha256 && head != intent.committed_head_sha256 {
                self.shared.activation_held.store(false, Ordering::SeqCst);
                return Err(GatewayActivationFenceErrorV1::Conflict);
            }
            Ok(Box::new(TestActivationPermit {
                shared: self.shared.clone(),
                intent: intent.clone(),
            }))
        }

        fn reconcile_committed(
            &mut self,
            intent: &FencedActivationIntentV1,
            _now_unix_ms: u64,
        ) -> std::result::Result<(), GatewayActivationFenceErrorV1> {
            let mut head = self.shared.activation_head.lock().unwrap();
            if *head == intent.committed_head_sha256 {
                return Ok(());
            }
            if *head != intent.expected_head_sha256 {
                return Err(GatewayActivationFenceErrorV1::Conflict);
            }
            head.clone_from(&intent.committed_head_sha256);
            Ok(())
        }
    }

    struct TestClock {
        shared: SharedTestState,
    }

    impl GatewayClockV1 for TestClock {
        fn now_unix_ms(&mut self) -> std::result::Result<u64, String> {
            Ok(self.shared.clock.fetch_add(1, Ordering::SeqCst))
        }
    }

    type TestGateway = VerifierGatewayV1<TestEvaluator, TestRenderer, TestFence, TestClock>;

    #[derive(Clone)]
    struct TestPaths {
        journal: PathBuf,
        store: PathBuf,
    }

    fn gateway_parts(
        journal: DurableVerifierJournalV2,
        store: DurableGatewayFragmentStoreV1,
        key: &MacKey,
        shared: &SharedTestState,
    ) -> TestGateway {
        let signer = TargetSigningKeyV2::from_seed(TARGET_SEED).unwrap();
        let evaluator = TestEvaluator {
            capabilities: capabilities(journal.session()),
            shared: shared.clone(),
        };
        VerifierGatewayV1::new(
            journal,
            store,
            key.clone(),
            signer,
            VerifierGatewayRuntimeV1 {
                evaluator,
                renderer: TestRenderer {
                    shared: shared.clone(),
                },
                fence: TestFence {
                    shared: shared.clone(),
                },
                clock: TestClock {
                    shared: shared.clone(),
                },
            },
        )
        .unwrap()
    }

    fn create_gateway(
        directory: &TempDir,
        key: &MacKey,
        shared: &SharedTestState,
    ) -> (
        TestGateway,
        AuthenticatedSessionV2,
        AuthenticatedRoundV2,
        String,
        TestPaths,
    ) {
        let paths = TestPaths {
            journal: directory.path().join("journal"),
            store: directory.path().join("gateway-store"),
        };
        let signer = TargetSigningKeyV2::from_seed(TARGET_SEED).unwrap();
        let session = signed_session(key, &signer);
        shared
            .activation_head
            .lock()
            .unwrap()
            .clone_from(&session.core.genesis_head_sha256);
        let session_record_sha256 = session.record_digest().unwrap();
        let journal = DurableVerifierJournalV2::create(
            &paths.journal,
            session.clone(),
            key.clone(),
            NOW,
            "request-key-a",
            3,
        )
        .unwrap();
        let request = round(&journal, key, "round-a");
        let store =
            DurableGatewayFragmentStoreV1::create(&paths.store, &session_record_sha256).unwrap();
        (
            gateway_parts(journal, store, key, shared),
            session,
            request,
            session_record_sha256,
            paths,
        )
    }

    fn reopen_gateway(
        paths: &TestPaths,
        key: &MacKey,
        shared: &SharedTestState,
        session_record_sha256: &str,
    ) -> TestGateway {
        let journal =
            DurableVerifierJournalV2::open(&paths.journal, key.clone(), "request-key-a", 3)
                .unwrap();
        let store =
            DurableGatewayFragmentStoreV1::open(&paths.store, session_record_sha256).unwrap();
        gateway_parts(journal, store, key, shared)
    }

    fn source_request(
        session_record_sha256: &str,
        command: GatewaySourceCommandV1,
        key: &MacKey,
    ) -> AuthenticatedGatewaySourceRequestV1 {
        AuthenticatedGatewaySourceRequestV1::sign(session_record_sha256.into(), command, key)
            .unwrap()
    }

    fn submit_request(
        session_record_sha256: &str,
        request: &AuthenticatedRoundV2,
        key: &MacKey,
    ) -> AuthenticatedGatewaySourceRequestV1 {
        source_request(
            session_record_sha256,
            GatewaySourceCommandV1::SubmitRound {
                round: Box::new(request.clone()),
                mirror_prediction: None,
            },
            key,
        )
    }

    fn mirror_submit_request(
        session_record_sha256: &str,
        request: &AuthenticatedRoundV2,
        predicted_frontier_token: u32,
        key: &MacKey,
    ) -> AuthenticatedGatewaySourceRequestV1 {
        let mirror_prediction = MirrorPredictionCommitmentV1::for_round(
            request,
            predicted_frontier_token,
            40,
            41,
            digest(22),
        )
        .unwrap();
        source_request(
            session_record_sha256,
            GatewaySourceCommandV1::SubmitRound {
                round: Box::new(request.clone()),
                mirror_prediction: Some(Box::new(mirror_prediction)),
            },
            key,
        )
    }

    fn committed_result(frames: &[VerifierGatewayFrameV1]) -> AuthenticatedResultV2 {
        let VerifierGatewayMessageV1::Gateway { reply } = &frames[0].message else {
            panic!("first response was not from gateway")
        };
        let VerifierGatewayReplyV1::RoundCommitted { result, .. } = reply.as_ref() else {
            panic!("first response was not a committed result")
        };
        result.as_ref().clone()
    }

    fn is_replay(frames: &[VerifierGatewayFrameV1]) -> bool {
        let VerifierGatewayMessageV1::Gateway { reply } = &frames[0].message else {
            return false;
        };
        matches!(
            reply.as_ref(),
            VerifierGatewayReplyV1::RoundCommitted { replayed: true, .. }
        )
    }

    fn session_failure(frames: &[VerifierGatewayFrameV1]) -> GatewaySessionFailureV1 {
        let VerifierGatewayMessageV1::Gateway { reply } = &frames[0].message else {
            panic!("failure response was not from gateway")
        };
        let VerifierGatewayReplyV1::SessionFailed { failure, .. } = reply.as_ref() else {
            panic!("response was not a durable session failure")
        };
        failure.clone()
    }

    fn source_install_receipt(
        request: &AuthenticatedRoundV2,
        result: &AuthenticatedResultV2,
    ) -> SourceInstallReceiptV1 {
        SourceInstallReceiptV1 {
            protocol: VERIFIER_SOURCE_INSTALL_RECEIPT_PROTOCOL_V1.into(),
            request_id: request.core.request_id.clone(),
            result_sha256: result.record_digest().unwrap(),
            applied_head_sha256: result.core.new_head_sha256.clone(),
            render_root_sha256: result.core.fragments_sha256.clone(),
            source_state_sha256: digest(21),
            source_state_bytes: 4_096,
            installed_output_height: result.core.new_output_height,
            installed_transcript_sha256: result.core.new_evaluated_tokens_sha256.clone(),
            durable_unix_ms: NOW + 100,
        }
    }

    fn wire_round_trip(frame: &VerifierGatewayFrameV1) -> VerifierGatewayFrameV1 {
        let mut wire = Vec::new();
        write_verifier_gateway_frame_v1(&mut wire, frame, VerifierGatewayFrameLimitsV1::default())
            .unwrap();
        read_verifier_gateway_frame_v1(
            &mut wire.as_slice(),
            VerifierGatewayFrameLimitsV1::default(),
        )
        .unwrap()
    }

    #[test]
    fn loopback_repairs_drop_reorder_duplicate_then_durably_acks_and_replays() {
        let key = MacKey::from_bytes([0x31; 32]);
        let shared = SharedTestState::default();
        let directory = TempDir::new().unwrap();
        let (mut gateway, session, request, session_digest, _) =
            create_gateway(&directory, &key, &shared);

        let admit = source_request(
            &session_digest,
            GatewaySourceCommandV1::AdmitSession {
                session: Box::new(session.clone()),
            },
            &key,
        );
        let admitted = gateway.handle_source(&admit).unwrap();
        assert!(matches!(
            &admitted[0].message,
            VerifierGatewayMessageV1::Gateway { reply }
                if matches!(reply.as_ref(), VerifierGatewayReplyV1::SessionAccepted { pending: None, .. })
        ));

        let submit = submit_request(&session_digest, &request, &key);
        let mut phases = Vec::new();
        let frames = gateway
            .handle_source_with_hook(&submit, &mut |phase| {
                phases.push(phase);
                Ok(())
            })
            .unwrap();
        assert_eq!(
            phases,
            vec![
                GatewayDrivePhaseV1::Reserved,
                GatewayDrivePhaseV1::Evaluated,
                GatewayDrivePhaseV1::Prepared,
                GatewayDrivePhaseV1::Staged,
                GatewayDrivePhaseV1::Activated,
            ]
        );
        assert_eq!(frames.len(), 3);
        let frames = frames.iter().map(wire_round_trip).collect::<Vec<_>>();
        let result = committed_result(&frames);
        result
            .verify_against(&request, gateway.journal().session())
            .unwrap();
        assert_eq!(shared.evaluator_calls.load(Ordering::SeqCst), 1);

        let VerifierGatewayMessageV1::Gateway { reply } = frames[0].message.clone() else {
            panic!("committed frame direction changed")
        };
        let verified =
            VerifiedGatewayRoundResultV1::from_committed_reply(&session, &submit, *reply, &key)
                .unwrap();
        assert!(verified.exact_mirror_commit().is_err());
        let mut source = VerifierSourceAssemblerV1::new(verified).unwrap();
        assert!(source.insert_frame(&frames[2]).unwrap());
        assert!(!source.insert_frame(&frames[2]).unwrap());
        assert_eq!(source.missing_ordinals(), vec![0]);

        let fetch = source.fetch_missing_request(&session_digest, &key).unwrap();
        let fetch_wire = wire_round_trip(&VerifierGatewayFrameV1::source(fetch));
        let VerifierGatewayMessageV1::Source { request: fetch } = fetch_wire.message else {
            panic!("fetch wire direction changed")
        };
        let repaired = gateway.handle_source(&fetch).unwrap();
        assert_eq!(repaired.len(), 2);
        assert!(source.insert_frame(&wire_round_trip(&repaired[0])).unwrap());
        assert!(source.missing_ordinals().is_empty());

        let child = round(gateway.journal(), &key, "round-before-ack");
        let child_submit = submit_request(&session_digest, &child, &key);
        assert!(matches!(
            gateway.handle_source(&child_submit),
            Err(VerifierGatewayErrorV1::Validation(_))
        ));

        let receipt = source_install_receipt(&request, &result);
        let installed = source
            .install_durably(&mut TestSourceInstaller {
                receipt: Some(receipt),
            })
            .unwrap();
        let ack = installed.source_ack_request(&session_digest, &key).unwrap();
        let acked = gateway.handle_source(&ack).unwrap();
        assert!(matches!(
            &acked[0].message,
            VerifierGatewayMessageV1::Gateway { reply }
                if matches!(reply.as_ref(), VerifierGatewayReplyV1::SourceAcked { .. })
        ));
        gateway.handle_source(&ack).unwrap();

        shared.live_fence.store(false, Ordering::SeqCst);
        let replay = gateway.handle_source(&submit).unwrap();
        assert!(is_replay(&replay));
        assert_eq!(committed_result(&replay), result);
        assert_eq!(shared.evaluator_calls.load(Ordering::SeqCst), 1);

        let mut forged = result;
        forged.target_signature_hex.replace_range(0..2, "00");
        assert!(forged
            .verify_against(&request, gateway.journal().session())
            .is_err());
    }

    #[test]
    fn every_gateway_phase_reopens_without_resampling_or_early_reply() {
        for crash_phase in [
            GatewayDrivePhaseV1::Reserved,
            GatewayDrivePhaseV1::Evaluated,
            GatewayDrivePhaseV1::Prepared,
            GatewayDrivePhaseV1::Staged,
            GatewayDrivePhaseV1::Activated,
        ] {
            let key = MacKey::from_bytes([0x32; 32]);
            let shared = SharedTestState::default();
            let directory = TempDir::new().unwrap();
            let (mut gateway, session, request, session_digest, paths) =
                create_gateway(&directory, &key, &shared);
            let submit = submit_request(&session_digest, &request, &key);
            let crashed = gateway.handle_source_with_hook(&submit, &mut |phase| {
                if phase == crash_phase {
                    Err(invalid("injected gateway crash"))
                } else {
                    Ok(())
                }
            });
            assert!(crashed.is_err(), "phase {crash_phase:?} returned a reply");
            let calls_before_reopen = shared.evaluator_calls.load(Ordering::SeqCst);
            assert_eq!(
                calls_before_reopen,
                usize::from(crash_phase != GatewayDrivePhaseV1::Reserved)
            );
            drop(gateway);

            let mut gateway = reopen_gateway(&paths, &key, &shared, &session_digest);
            let admit = source_request(
                &session_digest,
                GatewaySourceCommandV1::AdmitSession {
                    session: Box::new(session.clone()),
                },
                &key,
            );
            let state = gateway.handle_source(&admit).unwrap();
            let VerifierGatewayMessageV1::Gateway { reply } = &state[0].message else {
                panic!("admission direction changed")
            };
            let VerifierGatewayReplyV1::SessionAccepted { pending, .. } = reply.as_ref() else {
                panic!("admission response changed")
            };
            match crash_phase {
                GatewayDrivePhaseV1::Reserved | GatewayDrivePhaseV1::Evaluated => {
                    assert!(
                        matches!(pending, Some(PendingRoundSummaryV1::Reserved { request: pending }) if pending.as_ref() == &request)
                    );
                }
                GatewayDrivePhaseV1::Prepared => {
                    assert!(
                        matches!(pending, Some(PendingRoundSummaryV1::Prepared { request: pending, .. }) if pending.as_ref() == &request)
                    );
                }
                GatewayDrivePhaseV1::Staged => {
                    assert!(
                        matches!(pending, Some(PendingRoundSummaryV1::Staged { request: pending, .. }) if pending.as_ref() == &request)
                    );
                }
                GatewayDrivePhaseV1::Activated => assert!(pending.is_none()),
            }
            let completed = gateway.handle_source(&submit).unwrap();
            assert_eq!(committed_result(&completed).core.new_output_height, 2);
            assert_eq!(shared.evaluator_calls.load(Ordering::SeqCst), 1);
            assert_eq!(gateway.journal().current_head().0, 2);
        }
    }

    #[test]
    fn post_cas_renderer_crash_recovers_and_stale_fence_cannot_compute_new_work() {
        let key = MacKey::from_bytes([0x33; 32]);
        let shared = SharedTestState::default();
        shared.fail_activation_once.store(true, Ordering::SeqCst);
        let directory = TempDir::new().unwrap();
        let (mut gateway, _, request, session_digest, paths) =
            create_gateway(&directory, &key, &shared);
        let submit = submit_request(&session_digest, &request, &key);
        assert!(matches!(
            gateway.handle_source(&submit),
            Err(VerifierGatewayErrorV1::Verifier(VerifierV2Error::Renderer(
                _
            )))
        ));
        assert!(!gateway.journal().pending_activation());
        let committed_before_render = gateway
            .journal()
            .completed_result(&request.core.request_id)
            .unwrap();
        assert_eq!(
            *shared.activation_head.lock().unwrap(),
            committed_before_render.core.new_head_sha256
        );
        drop(gateway);

        let mut gateway = reopen_gateway(&paths, &key, &shared, &session_digest);
        let completed = gateway.handle_source(&submit).unwrap();
        assert!(is_replay(&completed));
        assert_eq!(shared.evaluator_calls.load(Ordering::SeqCst), 1);
        assert_eq!(shared.activate_calls.load(Ordering::SeqCst), 2);

        let key = MacKey::from_bytes([0x34; 32]);
        let shared = SharedTestState::default();
        shared.live_fence.store(false, Ordering::SeqCst);
        let directory = TempDir::new().unwrap();
        let (mut gateway, _, request, session_digest, _) =
            create_gateway(&directory, &key, &shared);
        let submit = submit_request(&session_digest, &request, &key);
        assert!(matches!(
            gateway.handle_source(&submit),
            Err(VerifierGatewayErrorV1::Verifier(
                VerifierV2Error::LeaseNotLive
            ))
        ));
        assert_eq!(shared.evaluator_calls.load(Ordering::SeqCst), 0);
        assert_eq!(gateway.journal().pending_round().unwrap(), None);
        shared.live_fence.store(true, Ordering::SeqCst);
        gateway.handle_source(&submit).unwrap();
        assert_eq!(shared.evaluator_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn expired_reservation_and_permanent_evaluator_failure_latch_across_restart() {
        let key = MacKey::from_bytes([0x35; 32]);
        let shared = SharedTestState::default();
        let directory = TempDir::new().unwrap();
        let (mut gateway, _, request, session_digest, paths) =
            create_gateway(&directory, &key, &shared);
        let submit = submit_request(&session_digest, &request, &key);
        let crashed = gateway.handle_source_with_hook(&submit, &mut |phase| {
            if phase == GatewayDrivePhaseV1::Reserved {
                Err(invalid("injected crash after reservation"))
            } else {
                Ok(())
            }
        });
        assert!(crashed.is_err());
        assert_eq!(shared.evaluator_calls.load(Ordering::SeqCst), 0);
        shared
            .clock
            .store(request.core.expires_unix_ms + 1, Ordering::SeqCst);
        let failed = gateway.handle_source(&submit).unwrap();
        let failure = session_failure(&failed);
        assert_eq!(
            failure.reason,
            GatewaySessionFailureReasonV1::ExpiredReserved
        );
        assert!(failure.failed_unix_ms > failure.request_expires_unix_ms);
        wire_round_trip(&failed[0]);
        drop(gateway);

        let mut gateway = reopen_gateway(&paths, &key, &shared, &session_digest);
        assert_eq!(
            session_failure(&gateway.handle_source(&submit).unwrap()),
            failure
        );
        assert_eq!(shared.evaluator_calls.load(Ordering::SeqCst), 0);

        let key = MacKey::from_bytes([0x36; 32]);
        let shared = SharedTestState::default();
        shared
            .permanent_evaluator_failure
            .store(true, Ordering::SeqCst);
        let directory = TempDir::new().unwrap();
        let (mut gateway, _, request, session_digest, paths) =
            create_gateway(&directory, &key, &shared);
        let submit = submit_request(&session_digest, &request, &key);
        let failed = gateway.handle_source(&submit).unwrap();
        let failure = session_failure(&failed);
        assert_eq!(
            failure.reason,
            GatewaySessionFailureReasonV1::PermanentEvaluator
        );
        assert_eq!(shared.evaluator_calls.load(Ordering::SeqCst), 1);
        drop(gateway);

        let mut gateway = reopen_gateway(&paths, &key, &shared, &session_digest);
        assert_eq!(
            session_failure(&gateway.handle_source(&submit).unwrap()),
            failure
        );
        assert_eq!(shared.evaluator_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn external_head_publish_outage_reconciles_exactly_but_conflict_fails_session() {
        let key = MacKey::from_bytes([0x37; 32]);
        let shared = SharedTestState::default();
        shared.fail_fence_publish_once.store(true, Ordering::SeqCst);
        let directory = TempDir::new().unwrap();
        let (mut gateway, _, request, session_digest, paths) =
            create_gateway(&directory, &key, &shared);
        let base_head = request.core.base_head_sha256.clone();
        let submit = submit_request(&session_digest, &request, &key);
        assert!(matches!(
            gateway.handle_source(&submit),
            Err(VerifierGatewayErrorV1::ActivationFence(
                GatewayActivationFenceErrorV1::Unavailable
            ))
        ));
        let committed = gateway
            .journal()
            .completed_result(&request.core.request_id)
            .unwrap()
            .clone();
        assert_eq!(*shared.activation_head.lock().unwrap(), base_head);
        assert_eq!(shared.activate_calls.load(Ordering::SeqCst), 0);
        drop(gateway);

        // Recovery of an already-durable local result is read-only with
        // respect to the old lease: it reconciles the exact head transition,
        // installs the renderer, and then exact replay remains available.
        shared.live_fence.store(false, Ordering::SeqCst);
        let mut gateway = reopen_gateway(&paths, &key, &shared, &session_digest);
        let replay = gateway.handle_source(&submit).unwrap();
        assert!(is_replay(&replay));
        assert_eq!(committed_result(&replay), committed);
        assert_eq!(
            *shared.activation_head.lock().unwrap(),
            committed.core.new_head_sha256
        );
        assert_eq!(shared.evaluator_calls.load(Ordering::SeqCst), 1);
        assert_eq!(shared.activate_calls.load(Ordering::SeqCst), 1);

        let key = MacKey::from_bytes([0x38; 32]);
        let shared = SharedTestState::default();
        let directory = TempDir::new().unwrap();
        let (mut gateway, _, request, session_digest, paths) =
            create_gateway(&directory, &key, &shared);
        *shared.activation_head.lock().unwrap() = digest(99);
        let submit = submit_request(&session_digest, &request, &key);
        let failed = gateway.handle_source(&submit).unwrap();
        let failure = session_failure(&failed);
        assert_eq!(
            failure.reason,
            GatewaySessionFailureReasonV1::ActivationFenceConflict
        );
        assert!(gateway
            .journal()
            .completed_result(&request.core.request_id)
            .is_none());
        assert_eq!(shared.activate_calls.load(Ordering::SeqCst), 0);
        assert_eq!(shared.evaluator_calls.load(Ordering::SeqCst), 1);
        drop(gateway);

        let mut gateway = reopen_gateway(&paths, &key, &shared, &session_digest);
        assert_eq!(
            session_failure(&gateway.handle_source(&submit).unwrap()),
            failure
        );
        assert_eq!(shared.evaluator_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn source_ack_requires_exact_durable_install_identity_before_child() {
        let key = MacKey::from_bytes([0x39; 32]);
        let shared = SharedTestState::default();
        let directory = TempDir::new().unwrap();
        let (mut gateway, _, request, session_digest, _) =
            create_gateway(&directory, &key, &shared);
        let submit = submit_request(&session_digest, &request, &key);
        let result = committed_result(&gateway.handle_source(&submit).unwrap());
        let valid = source_install_receipt(&request, &result);

        for receipt in [
            SourceInstallReceiptV1 {
                installed_output_height: valid.installed_output_height + 1,
                ..valid.clone()
            },
            SourceInstallReceiptV1 {
                installed_transcript_sha256: digest(98),
                ..valid.clone()
            },
            SourceInstallReceiptV1 {
                durable_unix_ms: result.core.completed_unix_ms.saturating_sub(1),
                ..valid.clone()
            },
            SourceInstallReceiptV1 {
                durable_unix_ms: NOW + MAX_SOURCE_RECEIPT_CLOCK_SKEW_MS_V1 + 10_000,
                ..valid.clone()
            },
        ] {
            let ack = source_request(
                &session_digest,
                GatewaySourceCommandV1::SourceAck { receipt },
                &key,
            );
            assert!(matches!(
                gateway.handle_source(&ack),
                Err(VerifierGatewayErrorV1::Validation(_))
            ));
        }

        assert!(AuthenticatedGatewaySourceRequestV1::sign(
            session_digest.clone(),
            GatewaySourceCommandV1::SourceAck {
                receipt: SourceInstallReceiptV1 {
                    source_state_bytes: 0,
                    ..valid.clone()
                },
            },
            &key,
        )
        .is_err());

        let child = round(gateway.journal(), &key, "child-after-source-install");
        let child_submit = submit_request(&session_digest, &child, &key);
        assert!(gateway.handle_source(&child_submit).is_err());

        let ack = source_request(
            &session_digest,
            GatewaySourceCommandV1::SourceAck { receipt: valid },
            &key,
        );
        gateway.handle_source(&ack).unwrap();
        let ack_path = gateway.store.ack_path(&result.core.new_head_sha256);
        let authentic_ack = fs::read(&ack_path).unwrap();
        let mut forged_ack: StoredSourceAckV1 =
            decode_canonical_json(&authentic_ack, GATEWAY_STORE_MAX_JSON_V1).unwrap();
        let replacement = if forged_ack.source_request.hmac_sha256.starts_with("00") {
            "ff"
        } else {
            "00"
        };
        forged_ack
            .source_request
            .hmac_sha256
            .replace_range(0..2, replacement);
        fs::write(&ack_path, canonical_json(&forged_ack).unwrap()).unwrap();
        shared.live_fence.store(true, Ordering::SeqCst);
        assert!(matches!(
            gateway.handle_source(&child_submit),
            Err(VerifierGatewayErrorV1::Codec(
                crate::verifier_gateway_codec::VerifierGatewayCodecErrorV1::Authentication
            ))
        ));
        fs::write(&ack_path, authentic_ack).unwrap();
        gateway.handle_source(&child_submit).unwrap();
        assert_eq!(shared.evaluator_calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn pending_commit_wal_reacquires_external_fence_before_renderer_recovery() {
        let key = MacKey::from_bytes([0x3a; 32]);
        let shared = SharedTestState::default();
        let directory = TempDir::new().unwrap();
        let (mut gateway, _, request, session_digest, paths) =
            create_gateway(&directory, &key, &shared);
        let submit = submit_request(&session_digest, &request, &key);
        let stopped = gateway.handle_source_with_hook(&submit, &mut |phase| {
            if phase == GatewayDrivePhaseV1::Staged {
                Err(invalid("stop before gateway activation intent"))
            } else {
                Ok(())
            }
        });
        assert!(stopped.is_err());
        let Some(PendingRoundV2::Staged {
            request: pending_request,
            result,
            renderer,
        }) = gateway.journal().pending_round().unwrap()
        else {
            panic!("gateway did not leave a staged round")
        };
        let intent = FencedActivationIntentV1 {
            protocol: GATEWAY_ACTIVATION_INTENT_PROTOCOL_V1.into(),
            session_record_sha256: session_digest.clone(),
            authority_term: pending_request.core.authority_term,
            request_id: pending_request.core.request_id.clone(),
            request_intent_sha256: pending_request.intent_sha256().unwrap(),
            result_sha256: result.record_digest().unwrap(),
            expected_head_sha256: pending_request.core.base_head_sha256.clone(),
            committed_head_sha256: result.core.new_head_sha256.clone(),
            renderer,
            created_unix_ms: NOW + 50,
        };
        gateway.store.persist_activation_intent(&intent).unwrap();

        // Model the process dying after the core commit WAL fsync but before
        // its Active receipt. The production gateway's deferred renderer does
        // not expose bytes at this point; this direct V2 call creates the exact
        // recovery state that the gateway must safely consume.
        shared.fail_activation_once.store(true, Ordering::SeqCst);
        assert!(matches!(
            gateway.journal.commit_and_activate(
                &pending_request,
                &gateway.fence,
                &mut gateway.renderer,
                NOW + 51,
            ),
            Err(VerifierV2Error::Renderer(_))
        ));
        assert!(gateway.journal().pending_activation());
        assert_eq!(
            *shared.activation_head.lock().unwrap(),
            pending_request.core.base_head_sha256
        );
        drop(gateway);

        shared.live_fence.store(false, Ordering::SeqCst);
        let mut gateway = reopen_gateway(&paths, &key, &shared, &session_digest);
        let replay = gateway.handle_source(&submit).unwrap();
        assert!(is_replay(&replay));
        assert!(!gateway.journal().pending_activation());
        assert_eq!(
            *shared.activation_head.lock().unwrap(),
            committed_result(&replay).core.new_head_sha256
        );
        assert_eq!(shared.evaluator_calls.load(Ordering::SeqCst), 1);
        assert_eq!(shared.activate_calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn mirror_commit_typestate_requires_full_exact_target_signed_acceptance() {
        let key = MacKey::from_bytes([0x3b; 32]);
        let shared = SharedTestState::default();
        shared.full_acceptance.store(true, Ordering::SeqCst);
        let directory = TempDir::new().unwrap();
        let (mut gateway, session, request, session_digest, _) =
            create_gateway(&directory, &key, &shared);
        let different_request = round(gateway.journal(), &key, "different-round");
        let different_submit =
            mirror_submit_request(&session_digest, &different_request, 777, &key);
        let submit = mirror_submit_request(&session_digest, &request, 777, &key);
        let frames = gateway.handle_source(&submit).unwrap();
        let VerifierGatewayMessageV1::Gateway { reply } = frames[0].message.clone() else {
            panic!("committed frame direction changed")
        };
        assert!(VerifiedGatewayRoundResultV1::from_committed_reply(
            &session,
            &different_submit,
            *reply.clone(),
            &key,
        )
        .is_err());
        let verified =
            VerifiedGatewayRoundResultV1::from_committed_reply(&session, &submit, *reply, &key)
                .unwrap();
        let changed_prediction = mirror_submit_request(&session_digest, &request, 778, &key);
        assert!(matches!(
            gateway.handle_source(&changed_prediction),
            Err(VerifierGatewayErrorV1::Validation(_))
        ));
        let mirror = verified.exact_mirror_commit().unwrap();
        assert_eq!(mirror.request_id(), request.core.request_id);
        assert_eq!(mirror.base_head_sha256(), request.core.base_head_sha256);
        assert_eq!(mirror.committed_context_rows(), 3);
        assert_eq!(mirror.frontier_token(), 777);
        assert_eq!(mirror.target_hidden_sha256().len(), 3);
        assert_eq!(mirror.provisional_parent_cache_revision(), 40);
        assert_eq!(mirror.provisional_cache_revision(), 41);
        assert_eq!(mirror.provisional_state_sha256(), digest(22));
    }

    #[test]
    fn gateway_constructor_rejects_loaded_target_capability_mismatch() {
        let key = MacKey::from_bytes([0x3c; 32]);
        let shared = SharedTestState::default();
        let directory = TempDir::new().unwrap();
        let (gateway, _, _, session_digest, paths) = create_gateway(&directory, &key, &shared);
        drop(gateway);
        let journal =
            DurableVerifierJournalV2::open(&paths.journal, key.clone(), "request-key-a", 3)
                .unwrap();
        let store = DurableGatewayFragmentStoreV1::open(&paths.store, &session_digest).unwrap();
        let mut wrong_capabilities = capabilities(journal.session());
        wrong_capabilities.target_checkpoint_sha256 = digest(99);
        let result = VerifierGatewayV1::new(
            journal,
            store,
            key,
            TargetSigningKeyV2::from_seed(TARGET_SEED).unwrap(),
            VerifierGatewayRuntimeV1 {
                evaluator: TestEvaluator {
                    capabilities: wrong_capabilities,
                    shared: shared.clone(),
                },
                renderer: TestRenderer {
                    shared: shared.clone(),
                },
                fence: TestFence {
                    shared: shared.clone(),
                },
                clock: TestClock { shared },
            },
        );
        assert!(matches!(result, Err(VerifierGatewayErrorV1::Validation(_))));
    }
}
