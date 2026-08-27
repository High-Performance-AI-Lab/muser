//! Exact request/model/component admission before any handoff bytes reach an
//! engine-owned shadow.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use kvpack_handoff::{
    BeginManifestV2, CommittedGeneration, ComponentKindV2, ExactIdentityV1, HandoffError,
    HandoffSinkV2, MultimodalIdentityV2, SegmentDescriptorV2, ValidatedBeginV2, VerifiedSealV2,
    VerifiedSegmentV2,
};

/// Namespace every receiver-minted request id carries. Request-scoped transfer
/// ids are `<request id>-<generation>`, so this is also the prefix of every
/// transfer id a request-aware producer stamps.
pub const REQUEST_ID_NAMESPACE: &str = "muser-";

/// A transfer id belongs to this request when it carries the request id the
/// receiver put on the control channel. Producers that predate the stamped id
/// mint an opaque id instead and stay admissible during the launch window; an
/// id inside the receiver's own namespace that names a different request is
/// always a stale connection.
pub fn transfer_id_matches(expected_prefix: &str, transfer_id: &str) -> bool {
    transfer_id.starts_with(expected_prefix) || !transfer_id.starts_with(REQUEST_ID_NAMESPACE)
}

#[derive(Debug, Clone)]
pub struct BeginExpectationsV2 {
    pub identity: ExactIdentityV1,
    pub prompt_token_ids: Vec<u32>,
    pub target_cache_identity_sha256: String,
    pub dflash_identity_sha256: Option<String>,
    pub multimodal: Option<MultimodalIdentityV2>,
    /// Request id the receiver sent on the control channel, when it has one.
    pub expected_transfer_id_prefix: Option<String>,
    pub max_context: usize,
    /// Delta claim lifted from the raw begin frame (the vendored typed
    /// manifest drops the unknown `prefix_cut` field; see transport.rs):
    /// tokens of the armed prompt the receiving session already holds. Zero
    /// is today's full transfer.
    pub prefix_cut: u64,
    /// Token history the receiving session holds at arm time; a nonzero cut
    /// must name exactly this prefix of the armed prompt.
    pub held_token_ids: Vec<u32>,
}

/// Set when the wrapped sink rejected the transfer, so the receiver can count
/// engine-install failures apart from verification failures. Identity checks
/// happen before the sink is called and never set it.
#[derive(Debug, Clone, Default)]
pub struct InstallFailureFlag(Arc<AtomicBool>);

impl InstallFailureFlag {
    pub fn is_set(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }

    fn set(&self) {
        self.0.store(true, Ordering::SeqCst)
    }
}

impl BeginExpectationsV2 {
    pub fn validate(&self, manifest: &BeginManifestV2) -> kvpack_handoff::Result<()> {
        // Generation zero can never become live, and a transfer id from another
        // request is a stale producer. Both are refused here, before any sink
        // work, so they can never fail an already-committed generation.
        if manifest.generation == 0 {
            return Err(validation("handoff generation zero is never installable"));
        }
        if let Some(prefix) = &self.expected_transfer_id_prefix {
            if !transfer_id_matches(prefix, &manifest.transfer_id) {
                return Err(validation("handoff transfer id names a different request"));
            }
        }
        if manifest.identity != self.identity {
            return Err(validation(
                "model/tokenizer/template/adapter identity mismatch",
            ));
        }
        if manifest.prompt_token_ids != self.prompt_token_ids {
            return Err(validation(
                "remote prompt tokens differ from the exact request",
            ));
        }
        if manifest.prompt_token_ids.is_empty()
            || manifest.prompt_token_ids.len() > self.max_context
        {
            return Err(validation(
                "remote prefix is outside the armed context bound",
            ));
        }
        if manifest.multimodal != self.multimodal {
            return Err(validation(
                "multimodal identity differs from the armed request",
            ));
        }
        let target = manifest
            .components
            .iter()
            .filter(|component| component.kind == ComponentKindV2::TargetKv)
            .collect::<Vec<_>>();
        if target.len() != 1
            || !target[0].required
            || target[0].identity_sha256 != self.target_cache_identity_sha256
        {
            return Err(validation("target cache component identity mismatch"));
        }
        let dflash = manifest
            .components
            .iter()
            .filter(|component| component.kind == ComponentKindV2::DflashContext)
            .collect::<Vec<_>>();
        match (&self.dflash_identity_sha256, dflash.as_slice()) {
            (None, []) => {}
            (Some(expected), [component])
                if component.required && component.identity_sha256 == *expected => {}
            _ => {
                return Err(validation(
                    "DFlash component identity or requirement mismatch",
                ))
            }
        }
        if manifest
            .components
            .iter()
            .any(|component| component.kind == ComponentKindV2::VisionContext)
        {
            return Err(validation(
                "vision state is represented by the multimodal identity, not a cache component",
            ));
        }
        if self.prefix_cut != 0 {
            self.validate_prefix_cut(manifest)?;
        }
        Ok(())
    }

    /// Delta admission: a begin carrying a nonzero prefix cut is a span
    /// transfer over `[cut, position)`. Fail closed unless the cut is
    /// 256-aligned, leaves a nonempty suffix, names a prefix the receiving
    /// session holds exactly, and — for declared schedules — the declared
    /// target segments equal the span schedule for the cut. Deferred streams
    /// declare no segments; the sink re-checks their target stream against
    /// the span schedule at prepare time.
    fn validate_prefix_cut(&self, manifest: &BeginManifestV2) -> kvpack_handoff::Result<()> {
        let position = manifest.prompt_token_ids.len() as u64;
        let cut = self.prefix_cut;
        if !cut.is_multiple_of(crate::schedule::PREFIX_CUT_ALIGN) || cut >= position {
            return Err(validation(
                "delta prefix cut is not a 256-aligned cut inside the prompt",
            ));
        }
        let cut = cut as usize;
        if self.held_token_ids.len() < cut
            || self.held_token_ids[..cut] != manifest.prompt_token_ids[..cut]
        {
            return Err(validation(
                "delta prefix cut names a prefix the receiving session does not hold",
            ));
        }
        if !manifest.deferred_segments && !declared_schedule_is_span(manifest, self.prefix_cut) {
            return Err(validation(
                "declared target schedule is not the span schedule for the prefix cut",
            ));
        }
        Ok(())
    }
}

pub struct CheckedSinkV2<S> {
    expectations: BeginExpectationsV2,
    inner: S,
    install_failed: InstallFailureFlag,
}

impl<S> CheckedSinkV2<S> {
    pub fn new(expectations: BeginExpectationsV2, inner: S) -> Self {
        Self {
            expectations,
            inner,
            install_failed: InstallFailureFlag::default(),
        }
    }

    /// Clone the install-failure flag before the sink is moved into a receive.
    pub fn install_failures(&self) -> InstallFailureFlag {
        self.install_failed.clone()
    }

    fn installing<T>(&self, result: kvpack_handoff::Result<T>) -> kvpack_handoff::Result<T> {
        if result.is_err() {
            self.install_failed.set();
        }
        result
    }
}

impl<S: HandoffSinkV2> HandoffSinkV2 for CheckedSinkV2<S> {
    fn begin(&mut self, begin: &ValidatedBeginV2) -> kvpack_handoff::Result<()> {
        self.expectations.validate(begin.manifest())?;
        let result = self.inner.begin(begin);
        self.installing(result)
    }

    fn segment_ready(&mut self, segment: VerifiedSegmentV2) -> kvpack_handoff::Result<()> {
        let result = self.inner.segment_ready(segment);
        self.installing(result)
    }

    fn prepare_commit(&mut self, seal: &VerifiedSealV2) -> kvpack_handoff::Result<()> {
        let result = self.inner.prepare_commit(seal);
        self.installing(result)
    }

    fn commit(&mut self) -> kvpack_handoff::Result<CommittedGeneration> {
        let result = self.inner.commit();
        self.installing(result)
    }

    fn abort(&mut self) {
        self.inner.abort()
    }
}

fn validation(message: &str) -> HandoffError {
    HandoffError::Validation(message.into())
}

/// Declared-mode span check: the manifest's target-component segments must be
/// exactly the span schedule for the cut. DFlash segments are cut-independent
/// and stay shape-checked by the sink at prepare time.
fn declared_schedule_is_span(manifest: &BeginManifestV2, prefix_cut: u64) -> bool {
    let position = manifest.prompt_token_ids.len() as u64;
    let Some(expected) = crate::schedule::muse_schedule_span(position, prefix_cut, None) else {
        return false;
    };
    let target_ids: std::collections::BTreeSet<&str> = manifest
        .components
        .iter()
        .filter(|component| component.kind == ComponentKindV2::TargetKv)
        .map(|component| component.id.as_str())
        .collect();
    let declared: Vec<&SegmentDescriptorV2> = manifest
        .segments
        .iter()
        .filter(|segment| target_ids.contains(segment.component_id.as_str()))
        .collect();
    declared.len() == expected.segments.len()
        && declared
            .iter()
            .zip(&expected.segments)
            .all(|(declared, intent)| {
                declared.sequence == intent.sequence
                    && declared.role == intent.role
                    && declared.layer == intent.layer
                    && declared.logical_start == intent.logical_start
                    && declared.logical_count == intent.logical_count
                    && declared.element_type == intent.element_type
                    && declared.elements_per_token == intent.elements_per_token
            })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kvpack_handoff::{ComponentV2, HmacIdentityV2, LIVE_HANDOFF_PROTOCOL_V2};

    fn material() -> (BeginExpectationsV2, BeginManifestV2) {
        let identity = ExactIdentityV1 {
            adapter_sha256: "a".repeat(64),
            chat_template_sha256: "b".repeat(64),
            context_policy_sha256: "c".repeat(64),
            model_revision: "m".into(),
            model_sha256: "d".repeat(64),
            tokenizer_revision: "t".into(),
            tokenizer_sha256: "e".repeat(64),
        };
        let expectations = BeginExpectationsV2 {
            identity: identity.clone(),
            prompt_token_ids: vec![1, 2],
            target_cache_identity_sha256: "f".repeat(64),
            dflash_identity_sha256: None,
            multimodal: None,
            expected_transfer_id_prefix: Some("muser-77-1".into()),
            max_context: 8,
            prefix_cut: 0,
            held_token_ids: Vec::new(),
        };
        let manifest = BeginManifestV2 {
            protocol: LIVE_HANDOFF_PROTOCOL_V2.into(),
            transfer_id: "muser-77-1-5".into(),
            generation: 5,
            created_unix_ms: 10,
            expires_unix_ms: 30,
            identity,
            prompt_token_ids: vec![1, 2],
            multimodal: None,
            hmac: HmacIdentityV2 {
                key_id: "loop-key".into(),
                epoch: 4,
            },
            components: vec![ComponentV2 {
                id: "target".into(),
                kind: ComponentKindV2::TargetKv,
                required: true,
                identity_sha256: "f".repeat(64),
            }],
            deferred_segments: false,
            segments: Vec::new(),
        };
        (expectations, manifest)
    }

    #[test]
    fn armed_request_admits_only_its_own_transfer() {
        let (expectations, mut manifest) = material();
        expectations.validate(&manifest).unwrap();
        manifest.transfer_id = "muser-77-2-5".into();
        assert!(expectations.validate(&manifest).is_err());
        // A producer that predates the stamped id keeps its opaque id.
        manifest.transfer_id = "9f3c".repeat(8);
        expectations.validate(&manifest).unwrap();
    }

    #[test]
    fn generation_zero_is_refused_before_any_sink_work() {
        let (expectations, mut manifest) = material();
        manifest.generation = 0;
        assert!(expectations.validate(&manifest).is_err());
    }

    #[test]
    fn unsolicited_producer_is_admitted_without_a_prefix() {
        let (mut expectations, mut manifest) = material();
        expectations.expected_transfer_id_prefix = None;
        manifest.transfer_id = "muser-99-9-1".into();
        expectations.validate(&manifest).unwrap();
    }

    /// A declared-mode delta: `position` prompt tokens, `cut` of them already
    /// held by the receiving session, target segments set to the span
    /// schedule for the cut.
    fn delta_material(position: u32, cut: u64) -> (BeginExpectationsV2, BeginManifestV2) {
        let (mut expectations, mut manifest) = material();
        let prompt: Vec<u32> = (0..position).collect();
        expectations.prompt_token_ids = prompt.clone();
        expectations.max_context = position as usize;
        expectations.prefix_cut = cut;
        expectations.held_token_ids = prompt[..cut as usize].to_vec();
        manifest.prompt_token_ids = prompt;
        manifest.segments = crate::schedule::muse_schedule_span(position as u64, cut, None)
            .unwrap()
            .segments
            .iter()
            .map(|intent| SegmentDescriptorV2 {
                sequence: intent.sequence,
                component_id: intent.component_id.clone(),
                role: intent.role,
                layer: intent.layer,
                logical_start: intent.logical_start,
                logical_count: intent.logical_count,
                element_type: intent.element_type.clone(),
                elements_per_token: intent.elements_per_token,
                byte_len: intent.logical_count * intent.elements_per_token as u64 * 2,
                sha256: "0".repeat(64),
            })
            .collect();
        (expectations, manifest)
    }

    #[test]
    fn delta_cut_admits_an_exact_held_prefix() {
        let (expectations, manifest) = delta_material(300, 256);
        expectations.validate(&manifest).unwrap();
    }

    #[test]
    fn delta_cut_must_be_256_aligned_and_leave_a_suffix() {
        for cut in [128, 300, 512] {
            let (mut expectations, manifest) = delta_material(300, 256);
            expectations.prefix_cut = cut;
            assert!(expectations.validate(&manifest).is_err(), "cut {cut}");
        }
    }

    #[test]
    fn delta_cut_requires_the_session_to_hold_the_prefix() {
        let (mut expectations, manifest) = delta_material(300, 256);
        expectations.held_token_ids[10] ^= 1;
        assert!(expectations.validate(&manifest).is_err());
        let (mut expectations, manifest) = delta_material(300, 256);
        expectations.held_token_ids.truncate(200);
        assert!(expectations.validate(&manifest).is_err());
        let (mut expectations, mut manifest) = delta_material(300, 256);
        manifest.prompt_token_ids[10] ^= 1;
        expectations.prompt_token_ids[10] ^= 1;
        assert!(expectations.validate(&manifest).is_err());
    }

    #[test]
    fn delta_declared_schedule_must_be_the_span_schedule() {
        let (expectations, mut manifest) = delta_material(300, 256);
        // A full-coverage NoPE tile where the span schedule starts at the cut.
        manifest.segments[0].logical_start = 0;
        manifest.segments[0].logical_count = 300;
        assert!(expectations.validate(&manifest).is_err());
        let (expectations, mut manifest) = delta_material(300, 256);
        manifest.segments.remove(1);
        assert!(expectations.validate(&manifest).is_err());
        let (expectations, mut manifest) = delta_material(300, 256);
        manifest.deferred_segments = true;
        manifest.segments.clear();
        // Deferred streams declare no segments; the sink re-checks the target
        // stream against the span schedule at prepare time.
        expectations.validate(&manifest).unwrap();
    }
}
