//! Canonical, bounded framing for the experimental remote-verifier gateway.
//!
//! This module is deliberately transport-only.  It does not bind a socket or
//! claim a serving endpoint.  A production caller must first establish TLS
//! 1.3 mutual authentication, the exact [`MUSER_VERIFIER_GATEWAY_ALPN`], and
//! the configured peer-certificate pin through `crate::security`, then run
//! this codec over that authenticated stream.

use std::io::{Read, Write};

use kvpack_handoff::{canonical_json, decode_canonical_json, sha256_hex, MacKey};
use serde::{Deserialize, Serialize};

use crate::verifier_v2::{
    AuthenticatedResultV2, AuthenticatedRoundV2, AuthenticatedSessionV2, FragmentDescriptorV2,
    FrontierV2, RendererStageReceiptV2, SamplerStateV2, MAX_FRAGMENTS_V2, MAX_FRAGMENT_BYTES_V2,
};

// The Rust type names retain their experimental V1 suffix, but the wire was
// deliberately bumped when durable source-state receipts and terminal
// session-failure replies were added.  ALPN prevents an older peer from
// silently accepting the expanded grammar.
pub const MUSER_VERIFIER_GATEWAY_ALPN: &[u8] = b"muser-verifier-gateway-v2";
pub const VERIFIER_GATEWAY_PROTOCOL_V1: &str = "muser-verifier-gateway-v2";
pub const VERIFIER_GATEWAY_SOURCE_PROTOCOL_V1: &str = "muser-verifier-gateway-source-v2";
pub const VERIFIER_GATEWAY_SOURCE_MAC_DOMAIN_V1: &[u8] = b"muser-verifier-gateway-source-v2";
pub const VERIFIER_SOURCE_INSTALL_RECEIPT_PROTOCOL_V1: &str = "muser-verifier-source-install-v2";
pub const VERIFIER_MIRROR_PREDICTION_PROTOCOL_V1: &str = "muser-mirror-prediction-v1";

const VERIFIER_GATEWAY_MAGIC_V1: &[u8; 8] = b"MUVERF1\0";
pub const MAX_VERIFIER_GATEWAY_HEADER_BYTES_V1: usize = 64 * 1024 * 1024;
pub const DEFAULT_VERIFIER_GATEWAY_HEADER_BYTES_V1: usize = 32 * 1024 * 1024;
pub const DEFAULT_VERIFIER_GATEWAY_PAYLOAD_BYTES_V1: u64 = 64 * 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum VerifierGatewayCodecErrorV1 {
    #[error("verifier gateway codec I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("verifier gateway codec validation: {0}")]
    Validation(String),
    #[error("verifier gateway source authentication failed")]
    Authentication,
}

pub type CodecResultV1<T> = std::result::Result<T, VerifierGatewayCodecErrorV1>;

fn invalid(message: impl Into<String>) -> VerifierGatewayCodecErrorV1 {
    VerifierGatewayCodecErrorV1::Validation(message.into())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerifierGatewayFrameLimitsV1 {
    pub max_header_bytes: usize,
    pub max_payload_bytes: u64,
}

impl Default for VerifierGatewayFrameLimitsV1 {
    fn default() -> Self {
        Self {
            max_header_bytes: DEFAULT_VERIFIER_GATEWAY_HEADER_BYTES_V1,
            max_payload_bytes: DEFAULT_VERIFIER_GATEWAY_PAYLOAD_BYTES_V1,
        }
    }
}

impl VerifierGatewayFrameLimitsV1 {
    pub fn validate(self) -> CodecResultV1<Self> {
        if self.max_header_bytes == 0
            || self.max_header_bytes > MAX_VERIFIER_GATEWAY_HEADER_BYTES_V1
            || self.max_payload_bytes > MAX_FRAGMENT_BYTES_V2
        {
            return Err(invalid("frame limits exceed the closed protocol bounds"));
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AuthenticatedGatewaySourceRequestV1 {
    pub core: GatewaySourceRequestCoreV1,
    pub hmac_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GatewaySourceRequestCoreV1 {
    pub protocol: String,
    pub session_record_sha256: String,
    pub command: GatewaySourceCommandV1,
}

/// Source-authenticated commitment to the provisional local DFlash state and
/// its predicted next frontier. The gateway persists the exact outer request
/// before reserving model work, so this value cannot be selected after seeing
/// the target result or changed on retry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct MirrorPredictionCommitmentV1 {
    pub protocol: String,
    pub request_id: String,
    pub request_intent_sha256: String,
    pub base_head_sha256: String,
    pub predicted_frontier_token: u32,
    pub provisional_parent_cache_revision: u64,
    pub provisional_cache_revision: u64,
    pub provisional_state_sha256: String,
}

impl MirrorPredictionCommitmentV1 {
    pub fn for_round(
        round: &AuthenticatedRoundV2,
        predicted_frontier_token: u32,
        provisional_parent_cache_revision: u64,
        provisional_cache_revision: u64,
        provisional_state_sha256: String,
    ) -> CodecResultV1<Self> {
        let commitment = Self {
            protocol: VERIFIER_MIRROR_PREDICTION_PROTOCOL_V1.into(),
            request_id: round.core.request_id.clone(),
            request_intent_sha256: round
                .intent_sha256()
                .map_err(|error| invalid(error.to_string()))?,
            base_head_sha256: round.core.base_head_sha256.clone(),
            predicted_frontier_token,
            provisional_parent_cache_revision,
            provisional_cache_revision,
            provisional_state_sha256,
        };
        commitment.validate_for(round)?;
        Ok(commitment)
    }

    fn validate_for(&self, round: &AuthenticatedRoundV2) -> CodecResultV1<()> {
        if self.protocol != VERIFIER_MIRROR_PREDICTION_PROTOCOL_V1
            || self.request_id != round.core.request_id
            || self.request_intent_sha256
                != round
                    .intent_sha256()
                    .map_err(|error| invalid(error.to_string()))?
            || self.base_head_sha256 != round.core.base_head_sha256
            || self.provisional_cache_revision
                != self
                    .provisional_parent_cache_revision
                    .checked_add(1)
                    .ok_or_else(|| invalid("Mirror cache revision overflow"))?
        {
            return Err(invalid("Mirror prediction commitment differs from round"));
        }
        validate_identifier("Mirror prediction request", &self.request_id)?;
        validate_digest("Mirror prediction intent", &self.request_intent_sha256)?;
        validate_digest("Mirror prediction parent", &self.base_head_sha256)?;
        validate_digest("Mirror provisional state", &self.provisional_state_sha256)
    }
}

/// Source-owned proof that the committed closure was durably installed, not
/// merely received into an in-memory assembler. The outer source-request HMAC
/// authenticates this assertion.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SourceInstallReceiptV1 {
    pub protocol: String,
    pub request_id: String,
    pub result_sha256: String,
    pub applied_head_sha256: String,
    pub render_root_sha256: String,
    /// Digest of a source-owned, fsynced snapshot or reconstructible CAS
    /// manifest. The gateway may release its repair copy only after the source
    /// authenticates this independent recovery point.
    pub source_state_sha256: String,
    pub source_state_bytes: u64,
    pub installed_output_height: u64,
    pub installed_transcript_sha256: String,
    pub durable_unix_ms: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GatewaySessionFailureReasonV1 {
    ExpiredReserved,
    PermanentEvaluator,
    ActivationFenceConflict,
}

/// Durable terminal state for a session whose current parent can no longer
/// choose a safe child. Starting a new authenticated session is required; the
/// failed request is retained for diagnosis and exact retry rejection.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct GatewaySessionFailureV1 {
    pub reason: GatewaySessionFailureReasonV1,
    pub request_id: String,
    pub request_intent_sha256: String,
    pub base_head_sha256: String,
    pub request_expires_unix_ms: u64,
    pub failed_unix_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum GatewaySourceCommandV1 {
    /// Prove possession of the session key and require an exact session record.
    AdmitSession {
        session: Box<AuthenticatedSessionV2>,
    },
    /// Admit or exactly retry one V3 verifier round.
    SubmitRound {
        round: Box<AuthenticatedRoundV2>,
        mirror_prediction: Option<Box<MirrorPredictionCommitmentV1>>,
    },
    /// Retransmit only the source's still-missing committed fragments.
    FetchMissing {
        request_id: String,
        result_sha256: String,
        ordinals: Vec<u32>,
    },
    /// Confirm that the source durably installed the exact committed head.
    SourceAck { receipt: SourceInstallReceiptV1 },
}

impl AuthenticatedGatewaySourceRequestV1 {
    pub fn sign(
        session_record_sha256: String,
        command: GatewaySourceCommandV1,
        key: &MacKey,
    ) -> CodecResultV1<Self> {
        let core = GatewaySourceRequestCoreV1 {
            protocol: VERIFIER_GATEWAY_SOURCE_PROTOCOL_V1.into(),
            session_record_sha256,
            command,
        };
        core.validate()?;
        let hmac_sha256 = key
            .tag_domain_hex(
                VERIFIER_GATEWAY_SOURCE_MAC_DOMAIN_V1,
                &canonical_json(&core).map_err(|error| invalid(error.to_string()))?,
            )
            .map_err(|_| VerifierGatewayCodecErrorV1::Authentication)?;
        Ok(Self { core, hmac_sha256 })
    }

    pub fn verify(&self, key: &MacKey) -> CodecResultV1<()> {
        self.core.validate()?;
        validate_digest("source request HMAC", &self.hmac_sha256)?;
        key.verify_domain_hex(
            VERIFIER_GATEWAY_SOURCE_MAC_DOMAIN_V1,
            &canonical_json(&self.core).map_err(|error| invalid(error.to_string()))?,
            &self.hmac_sha256,
        )
        .map_err(|_| VerifierGatewayCodecErrorV1::Authentication)
    }

    pub fn record_digest(&self) -> CodecResultV1<String> {
        Ok(sha256_hex(
            &canonical_json(self).map_err(|error| invalid(error.to_string()))?,
        ))
    }
}

impl GatewaySourceRequestCoreV1 {
    fn validate(&self) -> CodecResultV1<()> {
        if self.protocol != VERIFIER_GATEWAY_SOURCE_PROTOCOL_V1 {
            return Err(invalid("source request protocol differs"));
        }
        validate_digest("session record", &self.session_record_sha256)?;
        match &self.command {
            GatewaySourceCommandV1::AdmitSession { session } => {
                let digest = session
                    .record_digest()
                    .map_err(|error| invalid(error.to_string()))?;
                if digest != self.session_record_sha256 {
                    return Err(invalid("admitted session digest differs"));
                }
            }
            GatewaySourceCommandV1::SubmitRound {
                round,
                mirror_prediction,
            } => {
                validate_identifier("round request", &round.core.request_id)?;
                if let Some(mirror_prediction) = mirror_prediction {
                    mirror_prediction.validate_for(round)?;
                }
            }
            GatewaySourceCommandV1::FetchMissing {
                request_id,
                result_sha256,
                ordinals,
            } => {
                validate_identifier("fetch request", request_id)?;
                validate_digest("fetch result", result_sha256)?;
                if ordinals.len() > MAX_FRAGMENTS_V2
                    || ordinals.windows(2).any(|window| window[0] >= window[1])
                {
                    return Err(invalid(
                        "missing fragment ordinals are not a bounded strict set",
                    ));
                }
            }
            GatewaySourceCommandV1::SourceAck { receipt } => {
                if receipt.protocol != VERIFIER_SOURCE_INSTALL_RECEIPT_PROTOCOL_V1
                    || receipt.durable_unix_ms == 0
                {
                    return Err(invalid("source install receipt protocol or time differs"));
                }
                validate_identifier("ACK request", &receipt.request_id)?;
                validate_digest("ACK result", &receipt.result_sha256)?;
                validate_digest("ACK head", &receipt.applied_head_sha256)?;
                validate_digest("ACK render root", &receipt.render_root_sha256)?;
                validate_digest("ACK source state", &receipt.source_state_sha256)?;
                validate_digest(
                    "ACK installed transcript",
                    &receipt.installed_transcript_sha256,
                )?;
                if receipt.source_state_bytes == 0 {
                    return Err(invalid("source install receipt has no durable state"));
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "phase", rename_all = "snake_case", deny_unknown_fields)]
pub enum PendingRoundSummaryV1 {
    Reserved {
        request: Box<AuthenticatedRoundV2>,
    },
    Prepared {
        request: Box<AuthenticatedRoundV2>,
        result: Box<AuthenticatedResultV2>,
    },
    Staged {
        request: Box<AuthenticatedRoundV2>,
        result: Box<AuthenticatedResultV2>,
        renderer: RendererStageReceiptV2,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum VerifierGatewayReplyV1 {
    SessionAccepted {
        session: Box<AuthenticatedSessionV2>,
        session_record_sha256: String,
        pending: Option<PendingRoundSummaryV1>,
    },
    /// Emitted only after the commit WAL, renderer activation, and active
    /// receipt have all completed. The nested result is target-only signed.
    RoundCommitted {
        result: Box<AuthenticatedResultV2>,
        result_sha256: String,
        source_admission_sha256: String,
        replayed: bool,
    },
    Fragment {
        request_id: String,
        result_sha256: String,
        descriptor: FragmentDescriptorV2,
    },
    FetchComplete {
        request_id: String,
        result_sha256: String,
        delivered_ordinals: Vec<u32>,
    },
    SourceAcked {
        request_id: String,
        result_sha256: String,
        applied_head_sha256: String,
    },
    SessionFailed {
        session_record_sha256: String,
        failure: GatewaySessionFailureV1,
    },
    Stale {
        request_id: String,
        output_height: u64,
        head_sha256: String,
        frontier: FrontierV2,
        sampler_state: SamplerStateV2,
        transcript_sha256: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "direction", rename_all = "snake_case", deny_unknown_fields)]
pub enum VerifierGatewayMessageV1 {
    Source {
        request: Box<AuthenticatedGatewaySourceRequestV1>,
    },
    Gateway {
        reply: Box<VerifierGatewayReplyV1>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifierGatewayFrameV1 {
    pub message: VerifierGatewayMessageV1,
    pub payload: Vec<u8>,
}

impl VerifierGatewayFrameV1 {
    pub fn source(request: AuthenticatedGatewaySourceRequestV1) -> Self {
        Self {
            message: VerifierGatewayMessageV1::Source {
                request: Box::new(request),
            },
            payload: Vec::new(),
        }
    }

    pub fn reply(reply: VerifierGatewayReplyV1) -> Self {
        Self {
            message: VerifierGatewayMessageV1::Gateway {
                reply: Box::new(reply),
            },
            payload: Vec::new(),
        }
    }

    pub fn fragment(reply: VerifierGatewayReplyV1, payload: Vec<u8>) -> Self {
        Self {
            message: VerifierGatewayMessageV1::Gateway {
                reply: Box::new(reply),
            },
            payload,
        }
    }

    fn validate(&self) -> CodecResultV1<()> {
        validate_message_header(&self.message, self.payload.len() as u64)?;
        match &self.message {
            VerifierGatewayMessageV1::Source { request } => {
                request.core.validate()?;
            }
            VerifierGatewayMessageV1::Gateway { reply } => {
                if let VerifierGatewayReplyV1::Fragment { descriptor, .. } = reply.as_ref() {
                    if self.payload.len() as u64 != descriptor.byte_len
                        || sha256_hex(&self.payload) != descriptor.sha256
                    {
                        return Err(invalid("fragment frame payload differs"));
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct VerifierGatewayFrameHeaderV1 {
    protocol: String,
    payload_bytes: u64,
    payload_sha256: String,
    message: VerifierGatewayMessageV1,
}

pub fn write_verifier_gateway_frame_v1(
    writer: &mut impl Write,
    frame: &VerifierGatewayFrameV1,
    limits: VerifierGatewayFrameLimitsV1,
) -> CodecResultV1<()> {
    let limits = limits.validate()?;
    frame.validate()?;
    if frame.payload.len() as u64 > limits.max_payload_bytes {
        return Err(invalid("frame payload exceeds configured bounds"));
    }
    let header = VerifierGatewayFrameHeaderV1 {
        protocol: VERIFIER_GATEWAY_PROTOCOL_V1.into(),
        payload_bytes: frame.payload.len() as u64,
        payload_sha256: sha256_hex(&frame.payload),
        message: frame.message.clone(),
    };
    let header = canonical_json(&header).map_err(|error| invalid(error.to_string()))?;
    if header.is_empty() || header.len() > limits.max_header_bytes {
        return Err(invalid("frame header exceeds configured bounds"));
    }
    let header_len = u32::try_from(header.len()).map_err(|_| invalid("frame header overflow"))?;
    writer.write_all(VERIFIER_GATEWAY_MAGIC_V1)?;
    writer.write_all(&header_len.to_be_bytes())?;
    writer.write_all(&(frame.payload.len() as u64).to_be_bytes())?;
    writer.write_all(&header)?;
    writer.write_all(&frame.payload)?;
    writer.flush()?;
    Ok(())
}

pub fn read_verifier_gateway_frame_v1(
    reader: &mut impl Read,
    limits: VerifierGatewayFrameLimitsV1,
) -> CodecResultV1<VerifierGatewayFrameV1> {
    let limits = limits.validate()?;
    let mut magic = [0u8; 8];
    let mut header_len = [0u8; 4];
    let mut payload_len = [0u8; 8];
    reader.read_exact(&mut magic)?;
    reader.read_exact(&mut header_len)?;
    reader.read_exact(&mut payload_len)?;
    if &magic != VERIFIER_GATEWAY_MAGIC_V1 {
        return Err(invalid("frame magic differs"));
    }
    let header_len = u32::from_be_bytes(header_len) as usize;
    let payload_len = u64::from_be_bytes(payload_len);
    if header_len == 0
        || header_len > limits.max_header_bytes
        || payload_len > limits.max_payload_bytes
    {
        return Err(invalid("frame lengths exceed configured bounds"));
    }
    let mut header_bytes = vec![0u8; header_len];
    reader.read_exact(&mut header_bytes)?;
    let header: VerifierGatewayFrameHeaderV1 =
        decode_canonical_json(&header_bytes, limits.max_header_bytes)
            .map_err(|error| invalid(error.to_string()))?;
    if header.protocol != VERIFIER_GATEWAY_PROTOCOL_V1
        || header.payload_bytes != payload_len
        || !is_digest(&header.payload_sha256)
    {
        return Err(invalid("frame header identity or payload length differs"));
    }
    validate_message_header(&header.message, payload_len)?;
    let payload_len = usize::try_from(payload_len).map_err(|_| invalid("payload overflow"))?;
    let mut payload = vec![0u8; payload_len];
    reader.read_exact(&mut payload)?;
    if sha256_hex(&payload) != header.payload_sha256 {
        return Err(invalid("frame payload commitment differs"));
    }
    let frame = VerifierGatewayFrameV1 {
        message: header.message,
        payload,
    };
    frame.validate()?;
    Ok(frame)
}

fn validate_message_header(
    message: &VerifierGatewayMessageV1,
    payload_bytes: u64,
) -> CodecResultV1<()> {
    match message {
        VerifierGatewayMessageV1::Source { request } => {
            request.core.validate()?;
            if payload_bytes != 0 {
                return Err(invalid("source control frame carries a payload"));
            }
        }
        VerifierGatewayMessageV1::Gateway { reply } => {
            validate_reply(reply)?;
            match reply.as_ref() {
                VerifierGatewayReplyV1::Fragment { descriptor, .. }
                    if payload_bytes == descriptor.byte_len => {}
                VerifierGatewayReplyV1::Fragment { .. } => {
                    return Err(invalid("fragment preamble length differs from descriptor"));
                }
                _ if payload_bytes != 0 => {
                    return Err(invalid("control reply carries a payload"));
                }
                _ => {}
            }
        }
    }
    Ok(())
}

fn validate_reply(reply: &VerifierGatewayReplyV1) -> CodecResultV1<()> {
    match reply {
        VerifierGatewayReplyV1::SessionAccepted {
            session,
            session_record_sha256,
            pending,
        } => {
            validate_digest("accepted session", session_record_sha256)?;
            if session
                .record_digest()
                .map_err(|error| invalid(error.to_string()))?
                != *session_record_sha256
            {
                return Err(invalid("accepted session record digest differs"));
            }
            if let Some(pending) = pending {
                match pending {
                    PendingRoundSummaryV1::Reserved { request } => {
                        validate_identifier("pending request", &request.core.request_id)?;
                    }
                    PendingRoundSummaryV1::Prepared { request, result } => {
                        validate_pending_pair(request, result)?;
                    }
                    PendingRoundSummaryV1::Staged {
                        request,
                        result,
                        renderer,
                    } => {
                        validate_pending_pair(request, result)?;
                        validate_digest("pending renderer result", &renderer.result_sha256)?;
                        validate_digest("pending renderer root", &renderer.render_root_sha256)?;
                        validate_identifier("pending renderer token", &renderer.stage_token)?;
                        if renderer.result_sha256
                            != result
                                .record_digest()
                                .map_err(|error| invalid(error.to_string()))?
                            || renderer.render_root_sha256 != result.core.fragments_sha256
                        {
                            return Err(invalid("pending renderer receipt differs"));
                        }
                    }
                }
            }
        }
        VerifierGatewayReplyV1::RoundCommitted {
            result,
            result_sha256,
            source_admission_sha256,
            ..
        } => {
            validate_digest("committed result", result_sha256)?;
            validate_digest("committed source admission", source_admission_sha256)?;
            validate_identifier("committed request", &result.core.request_id)?;
            if result
                .record_digest()
                .map_err(|error| invalid(error.to_string()))?
                != *result_sha256
            {
                return Err(invalid("committed result digest differs"));
            }
        }
        VerifierGatewayReplyV1::Fragment {
            request_id,
            result_sha256,
            descriptor,
        } => {
            validate_identifier("fragment request", request_id)?;
            validate_digest("fragment result", result_sha256)?;
            validate_identifier("fragment component", &descriptor.component_id)?;
            validate_identifier("fragment ABI", &descriptor.payload_abi)?;
            validate_digest("fragment payload", &descriptor.sha256)?;
            if descriptor.ordinal as usize >= MAX_FRAGMENTS_V2
                || descriptor.byte_len == 0
                || descriptor.byte_len > MAX_FRAGMENT_BYTES_V2
            {
                return Err(invalid("fragment descriptor exceeds wire bounds"));
            }
        }
        VerifierGatewayReplyV1::FetchComplete {
            request_id,
            result_sha256,
            delivered_ordinals,
        } => {
            validate_identifier("fetch completion request", request_id)?;
            validate_digest("fetch completion result", result_sha256)?;
            if delivered_ordinals.len() > MAX_FRAGMENTS_V2
                || delivered_ordinals
                    .windows(2)
                    .any(|window| window[0] >= window[1])
            {
                return Err(invalid("fetch completion ordinal set differs"));
            }
        }
        VerifierGatewayReplyV1::SourceAcked {
            request_id,
            result_sha256,
            applied_head_sha256,
        } => {
            validate_identifier("ACKed request", request_id)?;
            validate_digest("ACKed result", result_sha256)?;
            validate_digest("ACKed head", applied_head_sha256)?;
        }
        VerifierGatewayReplyV1::SessionFailed {
            session_record_sha256,
            failure,
        } => {
            validate_digest("failed session", session_record_sha256)?;
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
        }
        VerifierGatewayReplyV1::Stale {
            request_id,
            head_sha256,
            transcript_sha256,
            ..
        } => {
            validate_identifier("stale request", request_id)?;
            validate_digest("stale head", head_sha256)?;
            validate_digest("stale transcript", transcript_sha256)?;
        }
    }
    Ok(())
}

fn validate_pending_pair(
    request: &AuthenticatedRoundV2,
    result: &AuthenticatedResultV2,
) -> CodecResultV1<()> {
    validate_identifier("pending request", &request.core.request_id)?;
    if request.core.request_id != result.core.request_id
        || request
            .intent_sha256()
            .map_err(|error| invalid(error.to_string()))?
            != result.core.request_intent_sha256
    {
        return Err(invalid("pending request/result identity differs"));
    }
    Ok(())
}

fn validate_identifier(label: &str, value: &str) -> CodecResultV1<()> {
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

fn is_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_digest(label: &str, value: &str) -> CodecResultV1<()> {
    if !is_digest(value) {
        return Err(invalid(format!("{label} digest differs")));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_rejects_oversize_before_body_and_noncanonical_headers() {
        let limits = VerifierGatewayFrameLimitsV1 {
            max_header_bytes: 128,
            max_payload_bytes: 32,
        };
        let mut oversized = Vec::new();
        oversized.extend_from_slice(VERIFIER_GATEWAY_MAGIC_V1);
        oversized.extend_from_slice(&129u32.to_be_bytes());
        oversized.extend_from_slice(&0u64.to_be_bytes());
        assert!(matches!(
            read_verifier_gateway_frame_v1(&mut oversized.as_slice(), limits),
            Err(VerifierGatewayCodecErrorV1::Validation(_))
        ));

        let raw = br#"{ "message" : {}, "payload_bytes" : 0, "payload_sha256" : "00", "protocol" : "muser-verifier-gateway-v1" }"#;
        let mut noncanonical = Vec::new();
        noncanonical.extend_from_slice(VERIFIER_GATEWAY_MAGIC_V1);
        noncanonical.extend_from_slice(&(raw.len() as u32).to_be_bytes());
        noncanonical.extend_from_slice(&0u64.to_be_bytes());
        noncanonical.extend_from_slice(raw);
        assert!(read_verifier_gateway_frame_v1(
            &mut noncanonical.as_slice(),
            VerifierGatewayFrameLimitsV1::default()
        )
        .is_err());

        // The decoded message shape must be checked before allocating or
        // attempting to read its declared payload.  This control reply omits
        // the advertised byte; an I/O error would prove the reader tried to
        // consume the body before rejecting the invalid shape.
        let control_with_payload = VerifierGatewayFrameHeaderV1 {
            protocol: VERIFIER_GATEWAY_PROTOCOL_V1.into(),
            payload_bytes: 1,
            payload_sha256: "00".repeat(32),
            message: VerifierGatewayMessageV1::Gateway {
                reply: Box::new(VerifierGatewayReplyV1::SourceAcked {
                    request_id: "request-a".into(),
                    result_sha256: "11".repeat(32),
                    applied_head_sha256: "22".repeat(32),
                }),
            },
        };
        let header = canonical_json(&control_with_payload).unwrap();
        let mut no_body = Vec::new();
        no_body.extend_from_slice(VERIFIER_GATEWAY_MAGIC_V1);
        no_body.extend_from_slice(&(header.len() as u32).to_be_bytes());
        no_body.extend_from_slice(&1u64.to_be_bytes());
        no_body.extend_from_slice(&header);
        assert!(matches!(
            read_verifier_gateway_frame_v1(
                &mut no_body.as_slice(),
                VerifierGatewayFrameLimitsV1::default()
            ),
            Err(VerifierGatewayCodecErrorV1::Validation(_))
        ));
    }
}
