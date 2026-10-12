//! Fail-closed Muse GGUF loading and QK-norm verification.

use std::path::Path;

use crate::config::{MuseConfig, MuseConfigError, QkNormProbe};
use crate::gguf::GgufFile;
use crate::reference::MuseModel;
use crate::tokenizer::BpeTokenizer;
use crate::weights::MuseWeights;

pub struct LoadedComponents {
    pub config: MuseConfig,
    pub weights: MuseWeights,
    pub tokenizer: BpeTokenizer,
    pub tokenizer_metadata_sha256: [u8; 32],
    pub chat_template: String,
    pub chat_template_sha256: [u8; 32],
    pub bos_token_id: Option<u32>,
    pub add_bos_token: bool,
    pub weight_precision: String,
}

pub fn load_components(path: &Path) -> Result<LoadedComponents, MuseConfigError> {
    let gguf = GgufFile::parse_path(path)
        .map_err(|error| MuseConfigError::Geometry(format!("gguf parse: {error}")))?;
    let weight_precision = weight_precision(&gguf)?;
    let weights = MuseWeights::open(path, &gguf)?;
    let probe = probe_qk_norms(&weights, &gguf)?;
    let config = MuseConfig::from_gguf(&gguf, &probe)?;
    if !probe.is_synthesized_scalar() {
        return Err(MuseConfigError::Geometry(format!(
            "attn_q_norm/attn_k_norm are not the expected constant broadcasts \
             (q_const={}, k_const={}, k_value={}); this checkpoint may carry a \
             genuinely learned QK-norm",
            probe.q_norm_is_constant, probe.k_norm_is_constant, probe.k_norm_value
        )));
    }

    let pre_type = gguf.meta_str("tokenizer.ggml.pre").unwrap_or("default");
    let tokenizer = BpeTokenizer::new(gguf.vocab(), gguf.merges(), pre_type, &gguf.token_types());
    if tokenizer.vocab_size() != config.vocab_size {
        return Err(MuseConfigError::Geometry(format!(
            "tokenizer vocab {} does not match model vocab {}",
            tokenizer.vocab_size(),
            config.vocab_size
        )));
    }
    let chat_template = gguf
        .chat_template()
        .filter(|template| !template.is_empty())
        .ok_or_else(|| MuseConfigError::MissingKey("tokenizer.chat_template".into()))?
        .to_owned();
    let chat_template_sha256 = gguf
        .chat_template_sha256()
        .ok_or_else(|| MuseConfigError::MissingKey("tokenizer.chat_template".into()))?;

    Ok(LoadedComponents {
        config,
        weights,
        tokenizer,
        tokenizer_metadata_sha256: gguf.tokenizer_metadata_sha256(),
        chat_template,
        chat_template_sha256,
        bos_token_id: gguf.meta_u32("tokenizer.ggml.bos_token_id"),
        add_bos_token: gguf
            .meta_bool("tokenizer.ggml.add_bos_token")
            .unwrap_or(false),
        weight_precision,
    })
}

fn weight_precision(gguf: &GgufFile) -> Result<String, MuseConfigError> {
    let has_nvfp4 = gguf
        .tensors
        .iter()
        .any(|tensor| tensor.dtype == crate::gguf::GgmlType::NVFP4_E2M1);
    let declared = gguf.meta_str("muser.weight_precision");
    match (has_nvfp4, declared) {
        (true, Some("nvfp4")) => Ok("nvfp4".into()),
        (true, other) => Err(MuseConfigError::Geometry(format!(
            "native NVFP4 tensors require muser.weight_precision=nvfp4, got {other:?}"
        ))),
        (false, Some("nvfp4")) => Err(MuseConfigError::Geometry(
            "muser.weight_precision=nvfp4 but the artifact has no native NVFP4 tensors".into(),
        )),
        (false, None | Some("q4_k_xl")) => Ok("q4_k_xl".into()),
        (false, Some(other)) => Err(MuseConfigError::Geometry(format!(
            "unsupported muser.weight_precision={other}"
        ))),
    }
}

pub fn load(path: &Path, max_seq: usize) -> Result<MuseModel, MuseConfigError> {
    let loaded = load_components(path)?;
    Ok(MuseModel::new(loaded.config, loaded.weights, max_seq))
}

fn probe_qk_norms(weights: &MuseWeights, gguf: &GgufFile) -> Result<QkNormProbe, MuseConfigError> {
    let n_layers = gguf
        .meta_u32("muse-glimmer.block_count")
        .ok_or_else(|| MuseConfigError::MissingKey("muse-glimmer.block_count".into()))?
        as usize;
    let mut q_value = f32::NAN;
    let mut k_value = f32::NAN;
    let mut q_const = true;
    let mut k_const = true;

    for layer in 0..n_layers {
        for (name, value, is_const) in [
            (
                format!("blk.{layer}.attn_q_norm.weight"),
                &mut q_value,
                &mut q_const,
            ),
            (
                format!("blk.{layer}.attn_k_norm.weight"),
                &mut k_value,
                &mut k_const,
            ),
        ] {
            let data = weights.f32_vec(&name)?;
            let first = *data.first().unwrap_or(&f32::NAN);
            if value.is_nan() {
                *value = first;
            }
            if data.iter().any(|candidate| *candidate != *value) {
                *is_const = false;
            }
        }
    }

    Ok(QkNormProbe {
        q_norm_value: q_value,
        k_norm_value: k_value,
        q_norm_is_constant: q_const,
        k_norm_is_constant: k_const,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{GgmlType, MetadataValue, TensorInfo};
    use std::collections::HashMap;

    fn fixture(dtype: GgmlType, precision: Option<&str>) -> GgufFile {
        let mut metadata = HashMap::new();
        if let Some(value) = precision {
            metadata.insert(
                "muser.weight_precision".into(),
                MetadataValue::Str(value.into()),
            );
        }
        GgufFile {
            version: 3,
            metadata,
            tensors: vec![TensorInfo {
                name: "matrix".into(),
                shape: vec![16, 1],
                dtype,
                offset: 0,
            }],
            data_offset: 0,
        }
    }

    #[test]
    fn native_nvfp4_precision_is_explicit_and_fail_closed() {
        assert_eq!(
            weight_precision(&fixture(GgmlType::NVFP4_E2M1, Some("nvfp4"))).unwrap(),
            "nvfp4"
        );
        assert!(weight_precision(&fixture(GgmlType::NVFP4_E2M1, None)).is_err());
        assert!(weight_precision(&fixture(GgmlType::F16, Some("nvfp4"))).is_err());
        assert_eq!(
            weight_precision(&fixture(GgmlType::Q4_K, None)).unwrap(),
            "q4_k_xl"
        );
    }
}
