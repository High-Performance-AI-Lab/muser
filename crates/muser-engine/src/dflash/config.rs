use std::collections::BTreeSet;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::DFlashError;
use crate::gguf::GgufFile;

/// The context-cache shape bound to one enrolled DFlash identity.
///
/// The trained window comes from the sidecar metadata. The 64-row sink is
/// part of Muser's DFlash cache ABI rather than GGUF metadata, so enrollment
/// stamps it into both peers' identity configs explicitly. Receivers must
/// never infer either value from a local fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DFlashContextGeometry {
    pub layers: usize,
    pub elements_per_token: usize,
    pub sink_size: usize,
    pub window_size: usize,
}

impl DFlashContextGeometry {
    pub fn validate(self) -> Result<(), String> {
        if self.layers == 0 {
            return Err("DFlash context geometry layers must be positive".into());
        }
        if self.elements_per_token == 0 {
            return Err("DFlash context geometry elements_per_token must be positive".into());
        }
        if self.sink_size == 0 {
            return Err("DFlash context geometry sink_size must be positive".into());
        }
        if self.window_size == 0 {
            return Err("DFlash context geometry window_size must be positive".into());
        }
        self.buffered_byte_limit()?;
        Ok(())
    }

    /// Exact maximum for the two f32 planes in every declared layer.
    ///
    /// Using the identity-derived bound keeps a 2048-row window bounded
    /// without retaining the old receiver's unrelated 512 MiB ceiling.
    pub fn buffered_byte_limit(self) -> Result<u64, String> {
        let bytes = self
            .layers
            .checked_mul(2)
            .and_then(|value| value.checked_mul(self.sink_size + self.window_size))
            .and_then(|value| value.checked_mul(self.elements_per_token))
            .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| "DFlash context geometry buffered byte limit overflow".to_string())?;
        u64::try_from(bytes)
            .map_err(|_| "DFlash context geometry buffered byte limit exceeds u64".to_string())
    }
}

/// The sink span used by the released DFlash cache ABI. It is made explicit
/// in every newly enrolled combined identity; it is not a receiver default.
pub const DFLASH_CONTEXT_SINK_SIZE: usize = 64;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DFlashSpecificConfig {
    pub mask_token_id: u32,
    pub target_layer_ids: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DFlashConfig {
    pub architectures: Vec<String>,
    #[serde(default = "default_block_size")]
    pub block_size: usize,
    #[serde(default)]
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub hidden_size: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub num_target_layers: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    /// Trained sliding-window span for the draft's context attention, from
    /// `dflash.attention.sliding_window`. The draft must be conditioned on
    /// exactly this many trailing target rows: measured 2026-08-21, feeding
    /// it half (the previously hardcoded 1024) or far more (32768) collapses
    /// natural-text acceptance from 72.5% to 2.2%.
    #[serde(default = "serde_default_sliding_window")]
    pub sliding_window: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub dflash_config: DFlashSpecificConfig,
    #[serde(default)]
    pub dtype: Option<String>,
}

/// Resolve the draft's trained sliding window, warning loudly when the
/// sidecar does not declare one.
///
/// Conditioning the draft on the wrong window is not a small error: a
/// sidecar trained at 2048 and run at 1024 collapsed natural-text
/// acceptance from ~72% to ~2% (2026-08-21). A sidecar trained at some
/// other width with the key absent would silently reproduce that, so the
/// fallback announces itself. Fail-closed rejection is an open owner
/// ruling, not taken here: older artifacts predate the key.
fn resolve_sliding_window(declared: Option<u32>) -> usize {
    match declared
        .map(|value| value as usize)
        .filter(|value| *value > 0)
    {
        Some(value) => value,
        None => {
            let fallback = default_sliding_window();
            eprintln!(
                "muser-dflash: WARNING dflash.attention.sliding_window absent/zero; \
                 defaulting to {fallback} — draft conditioning may be wrong if the \
                 sidecar is trained otherwise"
            );
            fallback
        }
    }
}

/// serde's default for a `config.json` artifact that omits the key. Routed
/// through the same warning as the GGUF path: this load path is reachable
/// via `from_artifact` for directory-style artifacts.
fn serde_default_sliding_window() -> usize {
    resolve_sliding_window(None)
}

fn default_sliding_window() -> usize {
    2_048
}

fn default_block_size() -> usize {
    16
}

impl DFlashConfig {
    pub fn from_artifact(path: &Path) -> Result<Self, DFlashError> {
        if path.is_file() {
            let gguf =
                GgufFile::parse_path(path).map_err(|error| DFlashError::Gguf(error.to_string()))?;
            return Self::from_gguf(&gguf);
        }
        Self::from_file(&path.join("config.json"))
    }

    pub fn from_file(path: &Path) -> Result<Self, DFlashError> {
        let data = std::fs::read_to_string(path).map_err(|e| DFlashError::Io(path.into(), e))?;
        let mut config: Self =
            serde_json::from_str(&data).map_err(|e| DFlashError::Config(e.to_string()))?;
        if config.sliding_window == 0 {
            // Present-but-zero never reaches serde's default, so it needs the
            // same warning as an absent key.
            config.sliding_window = resolve_sliding_window(None);
        }
        config.validate()?;
        Ok(config)
    }

    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Geometry the engine will use for this exact sidecar identity.
    pub fn context_geometry(&self) -> DFlashContextGeometry {
        DFlashContextGeometry {
            layers: self.num_hidden_layers,
            elements_per_token: self.num_key_value_heads * self.head_dim,
            sink_size: DFLASH_CONTEXT_SINK_SIZE,
            window_size: self.sliding_window,
        }
    }

    #[doc(hidden)]
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, DFlashError> {
        let architecture = gguf.meta_str("general.architecture");
        if architecture != Some("dflash") {
            return Err(DFlashError::Config(format!(
                "official assistant must have general.architecture=dflash, got {architecture:?}"
            )));
        }
        let required_u32 = |key: &str| {
            gguf.meta_u32(key).ok_or_else(|| {
                DFlashError::Config(format!("missing or invalid GGUF metadata {key}"))
            })
        };
        let required_f32 = |key: &str| {
            gguf.meta_f32(key).ok_or_else(|| {
                DFlashError::Config(format!("missing or invalid GGUF metadata {key}"))
            })
        };
        let raw_target_layers = gguf
            .meta_u32_array("dflash.target_layers")
            .ok_or_else(|| DFlashError::Config("missing dflash.target_layers".into()))?;
        // llama.cpp's official converter writes target hidden-state numbers
        // as one-based layer outputs. The SafeTensors training config is
        // zero-based, which is the convention used by Muser capture hooks.
        let target_layer_ids = raw_target_layers
            .into_iter()
            .map(|layer| {
                layer
                    .checked_sub(1)
                    .map(|value| value as usize)
                    .ok_or_else(|| {
                        DFlashError::Config(
                            "dflash.target_layers contains invalid one-based layer 0".into(),
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let key_length = required_u32("dflash.attention.key_length")? as usize;
        let value_length = required_u32("dflash.attention.value_length")? as usize;
        if key_length != value_length {
            return Err(DFlashError::Config(format!(
                "DFlash key/value head dimensions differ: {key_length} != {value_length}"
            )));
        }
        let vocab_size = gguf
            .meta_array_len("tokenizer.ggml.tokens")
            .filter(|count| *count > 0)
            .ok_or_else(|| DFlashError::Config("missing tokenizer.ggml.tokens".into()))?;
        let config = Self {
            architectures: vec!["DFlashDraftModel".into()],
            block_size: required_u32("dflash.block_size")? as usize,
            bos_token_id: gguf.meta_u32("tokenizer.ggml.bos_token_id").unwrap_or(0),
            eos_token_id: required_u32("tokenizer.ggml.eos_token_id")?,
            hidden_size: required_u32("dflash.embedding_length")? as usize,
            head_dim: key_length,
            intermediate_size: required_u32("dflash.feed_forward_length")? as usize,
            num_attention_heads: required_u32("dflash.attention.head_count")? as usize,
            num_hidden_layers: required_u32("dflash.block_count")? as usize,
            num_key_value_heads: required_u32("dflash.attention.head_count_kv")? as usize,
            // Muse Glimmer is the only target in this standalone engine.
            num_target_layers: 52,
            vocab_size,
            max_position_embeddings: required_u32("dflash.context_length")? as usize,
            sliding_window: resolve_sliding_window(
                gguf.meta_u32("dflash.attention.sliding_window"),
            ),
            rms_norm_eps: required_f32("dflash.attention.layer_norm_rms_epsilon")? as f64,
            rope_theta: required_f32("dflash.rope.freq_base")? as f64,
            dflash_config: DFlashSpecificConfig {
                mask_token_id: required_u32("tokenizer.ggml.mask_token_id")?,
                target_layer_ids,
            },
            dtype: Some("gguf-kquant".into()),
        };
        config.validate()?;
        Ok(config)
    }

    pub(super) fn validate(&self) -> Result<(), DFlashError> {
        if self.architectures.first().map(String::as_str) != Some("DFlashDraftModel") {
            return Err(DFlashError::Config(
                "architecture must be DFlashDraftModel".into(),
            ));
        }
        if self.num_hidden_layers != 5 {
            return Err(DFlashError::Config(format!(
                "release assistant must have exactly 5 layers, got {}",
                self.num_hidden_layers
            )));
        }
        if self.hidden_size == 0 || self.head_dim == 0 || self.block_size < 2 {
            return Err(DFlashError::Config(
                "zero geometry or block_size < 2".into(),
            ));
        }
        if self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
        {
            return Err(DFlashError::Config("invalid GQA head geometry".into()));
        }
        if self.dflash_config.target_layer_ids.is_empty() {
            return Err(DFlashError::Config("target_layer_ids is empty".into()));
        }
        if self
            .dflash_config
            .target_layer_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .len()
            != self.dflash_config.target_layer_ids.len()
        {
            return Err(DFlashError::Config(
                "target_layer_ids contains duplicates".into(),
            ));
        }
        if self
            .dflash_config
            .target_layer_ids
            .iter()
            .any(|&layer| layer >= self.num_target_layers)
        {
            return Err(DFlashError::Config(
                "target layer lies outside target model".into(),
            ));
        }
        Ok(())
    }

    pub fn expected_tensor_names(&self) -> Vec<String> {
        let mut names = vec![
            "fc.weight".into(),
            "hidden_norm.weight".into(),
            "norm.weight".into(),
        ];
        for layer in 0..self.num_hidden_layers {
            let p = format!("layers.{layer}");
            for suffix in [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            ] {
                names.push(format!("{p}.{suffix}"));
            }
        }
        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::MetadataValue;
    use std::collections::HashMap;

    #[test]
    fn config_json_without_the_key_falls_back_to_the_declared_default() {
        // from_artifact reaches this path for directory-style artifacts, so
        // the serde default must land on the same value the GGUF path does.
        let json = r#"{"architectures":["DFlashDraftModel"],"block_size":16,"eos_token_id":2,"hidden_size":64,"head_dim":16,"intermediate_size":128,"num_attention_heads":4,"num_hidden_layers":4,"num_key_value_heads":2,"num_target_layers":52,"vocab_size":100,"max_position_embeddings":1024,"rms_norm_eps":1e-6,"rope_theta":10000.0,"dflash_config":{"mask_token_id":99,"target_layer_ids":[1,9,17,25,33]}}"#;
        let parsed: DFlashConfig = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.sliding_window, 2_048);

        let declared = json.replace(
            "\"rope_theta\":10000.0",
            "\"rope_theta\":10000.0,\"sliding_window\":4096",
        );
        let parsed: DFlashConfig = serde_json::from_str(&declared).unwrap();
        assert_eq!(parsed.sliding_window, 4_096);
    }

    #[test]
    fn sliding_window_falls_back_loudly_when_undeclared() {
        // A sidecar without the key must still load, at the documented
        // default, and must not do so silently.
        assert_eq!(resolve_sliding_window(None), 2_048);
        assert_eq!(resolve_sliding_window(Some(0)), 2_048);
        // A declared window is honoured verbatim, including a non-default
        // one -- the case the warning exists to protect.
        assert_eq!(resolve_sliding_window(Some(2_048)), 2_048);
        assert_eq!(resolve_sliding_window(Some(4_096)), 4_096);
    }

    #[test]
    fn release_shape_rejects_non_five_layer_assistant() {
        let json = r#"{"architectures":["DFlashDraftModel"],"block_size":16,"eos_token_id":2,"hidden_size":64,"head_dim":16,"intermediate_size":128,"num_attention_heads":4,"num_hidden_layers":4,"num_key_value_heads":2,"num_target_layers":52,"vocab_size":100,"max_position_embeddings":1024,"rms_norm_eps":1e-6,"rope_theta":10000.0,"dflash_config":{"mask_token_id":99,"target_layer_ids":[1,9,17,25,33]}}"#;
        let parsed: DFlashConfig = serde_json::from_str(json).unwrap();
        assert!(parsed.validate().is_err());
    }

    #[test]
    fn context_geometry_uses_the_declared_window_and_has_an_exact_bound() {
        let geometry = DFlashContextGeometry {
            layers: 5,
            elements_per_token: 8 * 128,
            sink_size: 64,
            window_size: 2_048,
        };
        geometry.validate().unwrap();
        assert_eq!(geometry.buffered_byte_limit().unwrap(), 86_507_520);

        let legacy = DFlashContextGeometry {
            window_size: 1_024,
            ..geometry
        };
        assert_eq!(legacy.buffered_byte_limit().unwrap(), 44_564_480);
    }

    #[test]
    fn official_gguf_metadata_converts_one_based_target_layers() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".into(),
            MetadataValue::Str("dflash".into()),
        );
        for (key, value) in [
            ("dflash.block_size", 16),
            ("tokenizer.ggml.bos_token_id", 1),
            ("tokenizer.ggml.eos_token_id", 2),
            ("tokenizer.ggml.mask_token_id", 99),
            ("dflash.embedding_length", 64),
            ("dflash.attention.key_length", 16),
            ("dflash.attention.value_length", 16),
            ("dflash.feed_forward_length", 128),
            ("dflash.attention.head_count", 4),
            ("dflash.block_count", 5),
            ("dflash.attention.head_count_kv", 2),
            ("dflash.context_length", 1024),
        ] {
            metadata.insert(key.into(), MetadataValue::U32(value));
        }
        metadata.insert(
            "dflash.attention.layer_norm_rms_epsilon".into(),
            MetadataValue::F32(1e-6),
        );
        metadata.insert("dflash.rope.freq_base".into(), MetadataValue::F32(10_000.0));
        metadata.insert(
            "dflash.target_layers".into(),
            MetadataValue::Array(vec![
                MetadataValue::U32(1),
                MetadataValue::U32(10),
                MetadataValue::U32(52),
            ]),
        );
        metadata.insert(
            "tokenizer.ggml.tokens".into(),
            MetadataValue::Array(vec![MetadataValue::Str(String::new()); 100]),
        );
        let gguf = GgufFile {
            version: 3,
            metadata,
            tensors: Vec::new(),
            data_offset: 0,
        };
        let config = DFlashConfig::from_gguf(&gguf).unwrap();
        assert_eq!(config.dflash_config.target_layer_ids, [0, 9, 51]);
        assert_eq!(config.num_hidden_layers, 5);
        assert_eq!(config.dtype.as_deref(), Some("gguf-kquant"));
    }
}
