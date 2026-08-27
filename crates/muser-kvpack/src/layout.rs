//! Muse's sole durable layout/identity derivation.

use kvpack::{
    bind_weights_scalar_math_v2, derive_portable_prefill_descriptor_v2, ExportStateDeclaration,
    PortablePrefillDescriptorInputV2, PortablePrefillDescriptorV1, PreRopeKernelPinV1,
    WeightsScalarMathV1, MUSE_EXACT_LOGITS_LAYER, MUSE_EXACT_LOGITS_STATE, PORTABLE_PREFILL_ABI_V2,
};
use kvpack_core::{
    CacheKind, Codec, DType, FamilyState, Layout, StateKey, StaticDimension, TokenAxisRule,
};
use muser_engine::config::{
    MuseConfig, MUSE_HEAD_DIM, MUSE_KV_HEAD_COUNT, MUSE_LAYER_COUNT, MUSE_MAX_CONTEXT,
    MUSE_SWA_WINDOW,
};
use sha2::{Digest, Sha256};

pub const MUSE_LAYOUT_NAME: &str = "muse-glimmer-30b";
pub const MUSER_CACHE_ABI: &str = "muser-muse-glimmer-f16-logits-v2";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MuseIdentity {
    pub model_sha256: [u8; 32],
    pub adapter_sha256: [u8; 32],
    pub tokenizer_sha256: [u8; 32],
    pub chat_template_sha256: [u8; 32],
    pub context_policy_sha256: [u8; 32],
    pub model_revision: String,
    pub tokenizer_revision: String,
    pub weight_precision: String,
}

impl MuseIdentity {
    /// One digest covering every identity field. The resident radix scopes
    /// its keys by this digest, so an entry written under another identity —
    /// or under none — is structurally unreachable from a lookup, never a
    /// best-effort hit.
    pub fn digest(&self) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(b"muser-resident-identity-v1\0");
        for field in [
            self.model_sha256,
            self.adapter_sha256,
            self.tokenizer_sha256,
            self.chat_template_sha256,
            self.context_policy_sha256,
        ] {
            digest.update(field);
        }
        for text in [
            &self.model_revision,
            &self.tokenizer_revision,
            &self.weight_precision,
        ] {
            digest.update((text.len() as u64).to_le_bytes());
            digest.update(text.as_bytes());
        }
        digest.finalize().into()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LayoutError {
    #[error("Muser Muse geometry disagrees with the qualified kvpack layout: {0}")]
    Geometry(&'static str),
    #[error(transparent)]
    Kvpack(#[from] kvpack::StoreError),
}

pub fn descriptor(
    cfg: &MuseConfig,
    identity: &MuseIdentity,
    cached_token_count: u32,
) -> Result<PortablePrefillDescriptorV1, LayoutError> {
    validate_geometry(cfg, cached_token_count)?;
    let input = PortablePrefillDescriptorInputV2 {
        model_sha256: identity.model_sha256,
        adapter_sha256: identity.adapter_sha256,
        tokenizer_sha256: identity.tokenizer_sha256,
        chat_template_sha256: identity.chat_template_sha256,
        context_policy_sha256: identity.context_policy_sha256,
        model_revision: identity.model_revision.clone(),
        tokenizer_revision: identity.tokenizer_revision.clone(),
        producer_engine_abi: MUSER_CACHE_ABI.into(),
        consumer_engine_abi: MUSER_CACHE_ABI.into(),
        portable_abi: PORTABLE_PREFILL_ABI_V2.into(),
        compute_precision: "float16".into(),
        kv_precision: "float16".into(),
        weight_precision: identity.weight_precision.clone(),
        cached_token_count,
        max_context_tokens: MUSE_MAX_CONTEXT as u32,
        layout_name: MUSE_LAYOUT_NAME.into(),
        transform: None,
        prerope_kernel_pin: None::<PreRopeKernelPinV1>,
    };
    let descriptor = derive_portable_prefill_descriptor_v2(&input)?;
    let mut descriptor = bind_weights_scalar_math_v2(
        descriptor,
        &WeightsScalarMathV1 {
            qk_scale_factor_bits: f64::from(cfg.qk_scale_factor).to_bits(),
            output_multiplier_bits: f64::from(cfg.logit_scale).to_bits(),
            final_logit_softcapping_bits: f64::from(cfg.final_logit_softcap).to_bits(),
            post_norm_eps_bits: f64::from(cfg.post_norm_eps).to_bits(),
        },
    )?;
    let vocab = u64::try_from(cfg.vocab_size)
        .map_err(|_| LayoutError::Geometry("vocabulary exceeds u64"))?;
    let key = StateKey::new(MUSE_EXACT_LOGITS_LAYER, MUSE_EXACT_LOGITS_STATE);
    descriptor.family.states.push(FamilyState {
        key: key.clone(),
        cache_kind: CacheKind::OrdinaryKv,
        dtype: DType::F32,
        codec: Codec::Raw,
        codec_version: 1,
        layout: Layout::Contiguous,
        token_axis_rule: TokenAxisRule::TailWindow,
        token_axis: 0,
        elements_per_token: vocab,
        dimensions: vec![StaticDimension::Token, StaticDimension::Fixed(vocab)],
        dependencies: Vec::new(),
    });
    descriptor.states.push(ExportStateDeclaration {
        key,
        strides: vec![vocab, 1],
        atomic_group: MUSE_EXACT_LOGITS_LAYER + 1,
    });
    let logits_bytes = vocab
        .checked_mul(4)
        .ok_or(LayoutError::Geometry("logits byte size overflow"))?;
    descriptor.bytes_per_state = descriptor.bytes_per_state.max(logits_bytes);
    descriptor.restored_bytes = descriptor
        .restored_bytes
        .checked_add(logits_bytes)
        .ok_or(LayoutError::Geometry("descriptor byte size overflow"))?;
    kvpack_core::validate_family(&descriptor.family).map_err(kvpack::StoreError::from)?;
    Ok(descriptor)
}

fn validate_geometry(cfg: &MuseConfig, cached: u32) -> Result<(), LayoutError> {
    if cfg.n_layers != MUSE_LAYER_COUNT {
        return Err(LayoutError::Geometry("layer count"));
    }
    if cfg.n_kv_heads != MUSE_KV_HEAD_COUNT || cfg.head_dim != MUSE_HEAD_DIM {
        return Err(LayoutError::Geometry("KV head geometry"));
    }
    if cfg.sliding_window != MUSE_SWA_WINDOW {
        return Err(LayoutError::Geometry("SWA window"));
    }
    if cfg.context_length != MUSE_MAX_CONTEXT {
        return Err(LayoutError::Geometry("maximum context"));
    }
    if cached == 0 || cached as usize > cfg.context_length {
        return Err(LayoutError::Geometry("cached token count"));
    }
    if cfg
        .layer_kinds
        .iter()
        .enumerate()
        .any(|(layer, kind)| kind.is_swa() != (layer % 4 != 3))
    {
        return Err(LayoutError::Geometry("39-SWA/13-NoPE partition"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> MuseIdentity {
        MuseIdentity {
            model_sha256: [1; 32],
            adapter_sha256: [0; 32],
            tokenizer_sha256: [2; 32],
            chat_template_sha256: [3; 32],
            context_policy_sha256: [4; 32],
            model_revision: "test-muse".into(),
            tokenizer_revision: "test-tokenizer".into(),
            weight_precision: "nvfp4".into(),
        }
    }

    #[test]
    fn identity_digest_covers_every_field() {
        let base = identity();
        let digest = base.digest();
        assert_eq!(digest, base.digest());

        let mut wrong_model = base.clone();
        wrong_model.model_sha256[0] ^= 1;
        assert_ne!(digest, wrong_model.digest());

        let mut wrong_precision = base.clone();
        wrong_precision.weight_precision = "q4_k_xl".into();
        assert_ne!(digest, wrong_precision.digest());

        let mut wrong_revision = base.clone();
        wrong_revision.tokenizer_revision = "other-tokenizer".into();
        assert_ne!(digest, wrong_revision.digest());

        // Length-prefix framing: a field boundary shift is a different
        // identity even when the concatenated bytes read the same.
        let mut shifted = base.clone();
        shifted.model_revision = "test-mus".into();
        shifted.tokenizer_revision = "etest-tokenizer".into();
        assert_ne!(digest, shifted.digest());
    }
}
