//! Source-side typestates for consuming a committed verifier result.
//!
//! A raw target signature proves model semantics, but it does not by itself
//! prove that the source authenticated the exact request/session or received
//! the result through the gateway's post-activation reply.  These capabilities
//! close that accidental-misuse gap before fragment installation or Mirror-SD
//! resolution is allowed.

use kvpack_handoff::MacKey;

use crate::verifier_gateway_codec::{
    AuthenticatedGatewaySourceRequestV1, GatewaySourceCommandV1, MirrorPredictionCommitmentV1,
    VerifierGatewayReplyV1,
};
use crate::verifier_v2::{
    AuthenticatedResultV2, AuthenticatedRoundV2, AuthenticatedSessionV2, CouplingPolicyV2,
    FragmentKindV2, FrontierV2, ResultDecisionV2, RoundIntentV2, SessionCoreV2,
};

#[derive(Debug, thiserror::Error)]
pub enum VerifierGatewayAdapterErrorV1 {
    #[error("gateway adapter expected a post-activation committed result")]
    NotCommitted,
    #[error("gateway adapter authentication or semantic binding failed: {0}")]
    Binding(String),
    #[error("gateway result is not an exact greedy Mirror-SD commit")]
    NotExactMirrorCommit,
}

pub type AdapterResultV1<T> = std::result::Result<T, VerifierGatewayAdapterErrorV1>;

fn binding(error: impl ToString) -> VerifierGatewayAdapterErrorV1 {
    VerifierGatewayAdapterErrorV1::Binding(error.to_string())
}

/// Crate-owned capability that binds an authenticated source admission to a
/// directional target signature and a post-activation gateway reply. The
/// reply itself relies on the authenticated transport; it is not a standalone
/// gateway signature and must not be reconstructed by downstream crates.
pub struct VerifiedGatewayRoundResultV1 {
    session: AuthenticatedSessionV2,
    request: AuthenticatedRoundV2,
    mirror_prediction: Option<MirrorPredictionCommitmentV1>,
    result: AuthenticatedResultV2,
    result_sha256: String,
}

impl VerifiedGatewayRoundResultV1 {
    pub(crate) fn from_committed_reply(
        session: &AuthenticatedSessionV2,
        source_request: &AuthenticatedGatewaySourceRequestV1,
        reply: VerifierGatewayReplyV1,
        source_key: &MacKey,
    ) -> AdapterResultV1<Self> {
        session
            .verify_historical(
                source_key,
                &session.core.hmac_key_id,
                session.core.hmac_key_epoch,
            )
            .map_err(binding)?;
        source_request.verify(source_key).map_err(binding)?;
        if source_request.core.session_record_sha256 != session.record_digest().map_err(binding)? {
            return Err(binding("source admission addresses another session"));
        }
        let GatewaySourceCommandV1::SubmitRound {
            round: request,
            mirror_prediction,
        } = &source_request.core.command
        else {
            return Err(binding("source admission is not a round submission"));
        };
        request
            .verify_historical(&session.core, source_key)
            .map_err(binding)?;
        let VerifierGatewayReplyV1::RoundCommitted {
            result,
            result_sha256,
            source_admission_sha256,
            ..
        } = reply
        else {
            return Err(VerifierGatewayAdapterErrorV1::NotCommitted);
        };
        result
            .verify_against(request, &session.core)
            .map_err(binding)?;
        if result.record_digest().map_err(binding)? != result_sha256
            || source_request.record_digest().map_err(binding)? != source_admission_sha256
        {
            return Err(binding(
                "committed result or source admission digest differs",
            ));
        }
        Ok(Self {
            session: session.clone(),
            request: request.as_ref().clone(),
            mirror_prediction: mirror_prediction.as_deref().cloned(),
            result: *result,
            result_sha256,
        })
    }

    pub fn session(&self) -> &SessionCoreV2 {
        &self.session.core
    }

    pub fn request(&self) -> &AuthenticatedRoundV2 {
        &self.request
    }

    pub fn result(&self) -> &AuthenticatedResultV2 {
        &self.result
    }

    pub fn result_sha256(&self) -> &str {
        &self.result_sha256
    }

    pub fn exact_mirror_commit(&self) -> AdapterResultV1<VerifiedMirrorCommitV1> {
        let Some(prediction) = &self.mirror_prediction else {
            return Err(VerifierGatewayAdapterErrorV1::NotExactMirrorCommit);
        };
        if self.session.core.coupling_policy != CouplingPolicyV2::Greedy
            || self.request.core.intent != RoundIntentV2::Verify
        {
            return Err(VerifierGatewayAdapterErrorV1::NotExactMirrorCommit);
        }
        let ResultDecisionV2::Open {
            accepted_drafts,
            frontier_out,
        } = self.result.core.decision
        else {
            return Err(VerifierGatewayAdapterErrorV1::NotExactMirrorCommit);
        };
        let committed_context_rows = self.request.core.draft_tokens.len() + 1;
        let expected_ordinal = self
            .request
            .core
            .base_output_height
            .checked_add(committed_context_rows as u64)
            .ok_or_else(|| binding("Mirror output ordinal overflow"))?;
        let FrontierV2::Open {
            token,
            output_ordinal,
        } = self.result.core.new_frontier
        else {
            return Err(VerifierGatewayAdapterErrorV1::NotExactMirrorCommit);
        };
        if accepted_drafts as usize != self.request.core.draft_tokens.len()
            || self.result.core.committed_tokens.len() != committed_context_rows
            || frontier_out != prediction.predicted_frontier_token
            || token != prediction.predicted_frontier_token
            || output_ordinal != expected_ordinal
        {
            return Err(VerifierGatewayAdapterErrorV1::NotExactMirrorCommit);
        }
        let target_hidden_sha256 = self
            .result
            .core
            .fragments
            .iter()
            .filter(|descriptor| descriptor.kind == FragmentKindV2::TargetHidden)
            .map(|descriptor| descriptor.sha256.clone())
            .collect::<Vec<_>>();
        if target_hidden_sha256.is_empty() {
            return Err(VerifierGatewayAdapterErrorV1::NotExactMirrorCommit);
        }
        Ok(VerifiedMirrorCommitV1 {
            request_id: self.request.core.request_id.clone(),
            request_intent_sha256: self.request.intent_sha256().map_err(binding)?,
            result_sha256: self.result_sha256.clone(),
            base_head_sha256: self.result.core.base_head_sha256.clone(),
            committed_head_sha256: self.result.core.new_head_sha256.clone(),
            fragments_sha256: self.result.core.fragments_sha256.clone(),
            target_hidden_sha256,
            committed_context_rows,
            frontier_token: token,
            frontier_output_ordinal: output_ordinal,
            provisional_parent_cache_revision: prediction.provisional_parent_cache_revision,
            provisional_cache_revision: prediction.provisional_cache_revision,
            provisional_state_sha256: prediction.provisional_state_sha256.clone(),
        })
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        AuthenticatedSessionV2,
        AuthenticatedRoundV2,
        AuthenticatedResultV2,
    ) {
        (self.session, self.request, self.result)
    }
}

/// Engine-neutral capability consumed by the Mac adapter only after a full,
/// exact, target-signed gamma14 hit. Fields are private so token/count scalars
/// cannot be confused with a verified distributed transition.
pub struct VerifiedMirrorCommitV1 {
    request_id: String,
    request_intent_sha256: String,
    result_sha256: String,
    base_head_sha256: String,
    committed_head_sha256: String,
    fragments_sha256: String,
    target_hidden_sha256: Vec<String>,
    committed_context_rows: usize,
    frontier_token: u32,
    frontier_output_ordinal: u64,
    provisional_parent_cache_revision: u64,
    provisional_cache_revision: u64,
    provisional_state_sha256: String,
}

impl VerifiedMirrorCommitV1 {
    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    pub fn request_intent_sha256(&self) -> &str {
        &self.request_intent_sha256
    }

    pub fn result_sha256(&self) -> &str {
        &self.result_sha256
    }

    pub fn base_head_sha256(&self) -> &str {
        &self.base_head_sha256
    }

    pub fn committed_head_sha256(&self) -> &str {
        &self.committed_head_sha256
    }

    pub fn fragments_sha256(&self) -> &str {
        &self.fragments_sha256
    }

    pub fn target_hidden_sha256(&self) -> &[String] {
        &self.target_hidden_sha256
    }

    pub fn committed_context_rows(&self) -> usize {
        self.committed_context_rows
    }

    pub fn frontier_token(&self) -> u32 {
        self.frontier_token
    }

    pub fn frontier_output_ordinal(&self) -> u64 {
        self.frontier_output_ordinal
    }

    pub fn provisional_parent_cache_revision(&self) -> u64 {
        self.provisional_parent_cache_revision
    }

    pub fn provisional_cache_revision(&self) -> u64 {
        self.provisional_cache_revision
    }

    pub fn provisional_state_sha256(&self) -> &str {
        &self.provisional_state_sha256
    }
}
