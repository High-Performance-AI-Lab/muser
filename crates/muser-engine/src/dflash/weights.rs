use std::io::Write;
use std::mem::size_of;
use std::path::Path;

use super::{DFlashConfig, DFlashError};

pub struct DFlashWeights {
    pub config: DFlashConfig,
    pub fc_weight: Vec<f32>,
    pub hidden_norm_weight: Vec<f32>,
    pub norm_weight: Vec<f32>,
    pub layers: Vec<DFlashLayerWeights>,
    /// Retain the official GGUF mapping so Metal can consume the k-quant
    /// projection bytes directly. The f32 fields above remain the CPU oracle
    /// and SafeTensors compatibility path; production Metal must not expand
    /// and stream them for each speculative round.
    pub(crate) gguf_weights: Option<crate::weights::MuseWeights>,
}

pub struct DFlashLayerWeights {
    pub input_layernorm_weight: Vec<f32>,
    pub post_attention_layernorm_weight: Vec<f32>,
    pub q_proj_weight: Vec<f32>,
    pub k_proj_weight: Vec<f32>,
    pub v_proj_weight: Vec<f32>,
    pub o_proj_weight: Vec<f32>,
    pub q_norm_weight: Vec<f32>,
    pub k_norm_weight: Vec<f32>,
    pub gate_proj_weight: Vec<f32>,
    pub up_proj_weight: Vec<f32>,
    pub down_proj_weight: Vec<f32>,
}

impl DFlashWeights {
    pub fn load(path: &Path) -> Result<Self, DFlashError> {
        if path.is_file() {
            return Self::load_gguf(path);
        }
        Self::load_safetensors(path)
    }

    /// Load the official assistant without expanding its projection matrices.
    /// Norm vectors remain f32; all large matrices stay mmap'd in their GGUF
    /// k-quant representation and are consumed directly by Metal.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn load_metal(path: &Path) -> Result<Self, DFlashError> {
        if !path.is_file() {
            return Self::load_safetensors(path);
        }
        let (mut weights, mapped) = Self::load_gguf_projection_shell(path)?;
        weights.gguf_weights = Some(mapped);
        Ok(weights)
    }

    /// Load only the CPU-side norm vectors when every dense projection is
    /// supplied by an external backend such as public Core ML.  Expanding the
    /// official 1.5 GiB k-quant assistant into ~6 GiB of f32 matrices wastes
    /// startup time and resident memory, and none of those bytes are read by
    /// the projection-backend forward path.
    pub(crate) fn load_with_external_projections(path: &Path) -> Result<Self, DFlashError> {
        if !path.is_file() {
            return Self::load_safetensors(path);
        }
        let (weights, _mapped) = Self::load_gguf_projection_shell(path)?;
        Ok(weights)
    }

    fn load_gguf_projection_shell(
        path: &Path,
    ) -> Result<(Self, crate::weights::MuseWeights), DFlashError> {
        let gguf = crate::gguf::GgufFile::parse_path(path)
            .map_err(|error| DFlashError::Gguf(error.to_string()))?;
        let config = DFlashConfig::from_gguf(&gguf)?;
        let mapped = crate::weights::MuseWeights::open(path, &gguf)
            .map_err(|error| DFlashError::Gguf(error.to_string()))?;
        Self::validate_quantized_gguf_layouts(&mapped, &config)?;
        let load = |name: &str| {
            mapped
                .f32_vec(name)
                .map_err(|error| DFlashError::Gguf(error.to_string()))
        };
        let hidden_norm_weight = load("enc.output_norm.weight")?;
        let norm_weight = load("output_norm.weight")?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            let p = format!("blk.{layer}");
            let one = |suffix: &str| load(&format!("{p}.{suffix}.weight"));
            layers.push(DFlashLayerWeights {
                input_layernorm_weight: one("attn_norm")?,
                post_attention_layernorm_weight: one("ffn_norm")?,
                q_norm_weight: one("attn_q_norm")?,
                k_norm_weight: one("attn_k_norm")?,
                q_proj_weight: Vec::new(),
                k_proj_weight: Vec::new(),
                v_proj_weight: Vec::new(),
                o_proj_weight: Vec::new(),
                gate_proj_weight: Vec::new(),
                up_proj_weight: Vec::new(),
                down_proj_weight: Vec::new(),
            });
        }
        Ok((
            Self {
                config,
                fc_weight: Vec::new(),
                hidden_norm_weight,
                norm_weight,
                layers,
                gguf_weights: None,
            },
            mapped,
        ))
    }

    fn validate_quantized_gguf_layouts(
        mapped: &crate::weights::MuseWeights,
        config: &DFlashConfig,
    ) -> Result<(), DFlashError> {
        let h = config.hidden_size;
        let q = config.num_attention_heads * config.head_dim;
        let kv = config.num_key_value_heads * config.head_dim;
        let inter = config.intermediate_size;
        let sampled = config.dflash_config.target_layer_ids.len();
        let mut expected = vec![("fc.weight".to_string(), sampled * h, h)];
        for layer in 0..config.num_hidden_layers {
            for (suffix, input, output) in [
                ("attn_q", h, q),
                ("attn_k", h, kv),
                ("attn_v", h, kv),
                ("attn_output", q, h),
                ("ffn_gate", h, inter),
                ("ffn_up", h, inter),
                ("ffn_down", inter, h),
            ] {
                expected.push((format!("blk.{layer}.{suffix}.weight"), input, output));
            }
        }
        for (name, input, output) in expected {
            let layout = mapped
                .layout(&name)
                .map_err(|error| DFlashError::Gguf(error.to_string()))?;
            if (layout.n_in, layout.n_out) != (input, output) {
                return Err(DFlashError::Config(format!(
                    "{name}: expected [{input}, {output}], got [{}, {}]",
                    layout.n_in, layout.n_out
                )));
            }
            if !matches!(
                layout.dtype,
                crate::gguf::GgmlType::Q4_K
                    | crate::gguf::GgmlType::Q5_K
                    | crate::gguf::GgmlType::Q6_K
            ) {
                return Err(DFlashError::Config(format!(
                    "{name}: Metal DFlash requires Q4_K/Q5_K/Q6_K, got {:?}",
                    layout.dtype
                )));
            }
        }
        Ok(())
    }

    fn load_safetensors(model_dir: &Path) -> Result<Self, DFlashError> {
        let config = DFlashConfig::from_file(&model_dir.join("config.json"))?;
        let path = model_dir.join("model.safetensors");
        if !path.exists() {
            return Err(DFlashError::MissingFile(path));
        }
        let mut file = std::fs::File::open(&path).map_err(|e| DFlashError::Io(path.clone(), e))?;
        let st = crate::safetensors::SafeTensorsFile::parse(&mut file)
            .map_err(|e| DFlashError::SafeTensors(e.to_string()))?;
        for name in config.expected_tensor_names() {
            if !st.tensors.contains_key(&name) {
                return Err(DFlashError::MissingTensor(name));
            }
        }
        let load = |file: &mut std::fs::File, name: &str| {
            st.read_tensor_f32(file, name)
                .map_err(|e| DFlashError::SafeTensors(e.to_string()))
        };
        let fc_weight = load(&mut file, "fc.weight")?;
        let hidden_norm_weight = load(&mut file, "hidden_norm.weight")?;
        let norm_weight = load(&mut file, "norm.weight")?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            let p = format!("layers.{layer}");
            let mut one = |suffix: &str| load(&mut file, &format!("{p}.{suffix}"));
            layers.push(DFlashLayerWeights {
                input_layernorm_weight: one("input_layernorm.weight")?,
                post_attention_layernorm_weight: one("post_attention_layernorm.weight")?,
                q_proj_weight: one("self_attn.q_proj.weight")?,
                k_proj_weight: one("self_attn.k_proj.weight")?,
                v_proj_weight: one("self_attn.v_proj.weight")?,
                o_proj_weight: one("self_attn.o_proj.weight")?,
                q_norm_weight: one("self_attn.q_norm.weight")?,
                k_norm_weight: one("self_attn.k_norm.weight")?,
                gate_proj_weight: one("mlp.gate_proj.weight")?,
                up_proj_weight: one("mlp.up_proj.weight")?,
                down_proj_weight: one("mlp.down_proj.weight")?,
            });
        }
        let weights = Self {
            config,
            fc_weight,
            hidden_norm_weight,
            norm_weight,
            layers,
            gguf_weights: None,
        };
        weights.validate_shapes()?;
        Ok(weights)
    }

    fn load_gguf(path: &Path) -> Result<Self, DFlashError> {
        let gguf = crate::gguf::GgufFile::parse_path(path)
            .map_err(|error| DFlashError::Gguf(error.to_string()))?;
        let config = DFlashConfig::from_gguf(&gguf)?;
        let mapped = crate::weights::MuseWeights::open(path, &gguf)
            .map_err(|error| DFlashError::Gguf(error.to_string()))?;
        let load = |name: &str| {
            mapped
                .f32_vec(name)
                .map_err(|error| DFlashError::Gguf(error.to_string()))
        };
        let fc_weight = load("fc.weight")?;
        let hidden_norm_weight = load("enc.output_norm.weight")?;
        let norm_weight = load("output_norm.weight")?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer in 0..config.num_hidden_layers {
            let p = format!("blk.{layer}");
            let one = |suffix: &str| load(&format!("{p}.{suffix}.weight"));
            layers.push(DFlashLayerWeights {
                input_layernorm_weight: one("attn_norm")?,
                post_attention_layernorm_weight: one("ffn_norm")?,
                q_proj_weight: one("attn_q")?,
                k_proj_weight: one("attn_k")?,
                v_proj_weight: one("attn_v")?,
                o_proj_weight: one("attn_output")?,
                q_norm_weight: one("attn_q_norm")?,
                k_norm_weight: one("attn_k_norm")?,
                gate_proj_weight: one("ffn_gate")?,
                up_proj_weight: one("ffn_up")?,
                down_proj_weight: one("ffn_down")?,
            });
        }
        let weights = Self {
            config,
            fc_weight,
            hidden_norm_weight,
            norm_weight,
            layers,
            gguf_weights: Some(mapped),
        };
        weights.validate_shapes()?;
        Ok(weights)
    }

    fn validate_shapes(&self) -> Result<(), DFlashError> {
        let c = &self.config;
        let h = c.hidden_size;
        let q = c.num_attention_heads * c.head_dim;
        let kv = c.num_key_value_heads * c.head_dim;
        let inter = c.intermediate_size;
        let exact = |name: &str, got: usize, expected: usize| {
            if got == expected {
                Ok(())
            } else {
                Err(DFlashError::Config(format!(
                    "{name}: expected {expected} elements, got {got}"
                )))
            }
        };
        exact(
            "fc.weight",
            self.fc_weight.len(),
            c.dflash_config.target_layer_ids.len() * h * h,
        )?;
        exact("hidden_norm.weight", self.hidden_norm_weight.len(), h)?;
        exact("norm.weight", self.norm_weight.len(), h)?;
        for (index, layer) in self.layers.iter().enumerate() {
            for (name, got, expected) in [
                ("input_layernorm", layer.input_layernorm_weight.len(), h),
                (
                    "post_attention_layernorm",
                    layer.post_attention_layernorm_weight.len(),
                    h,
                ),
                ("q_proj", layer.q_proj_weight.len(), q * h),
                ("k_proj", layer.k_proj_weight.len(), kv * h),
                ("v_proj", layer.v_proj_weight.len(), kv * h),
                ("o_proj", layer.o_proj_weight.len(), h * q),
                ("q_norm", layer.q_norm_weight.len(), c.head_dim),
                ("k_norm", layer.k_norm_weight.len(), c.head_dim),
                ("gate_proj", layer.gate_proj_weight.len(), inter * h),
                ("up_proj", layer.up_proj_weight.len(), inter * h),
                ("down_proj", layer.down_proj_weight.len(), h * inter),
            ] {
                exact(&format!("layer {index} {name}"), got, expected)?;
            }
        }
        Ok(())
    }

    pub fn param_count(&self) -> usize {
        self.fc_weight.len()
            + self.hidden_norm_weight.len()
            + self.norm_weight.len()
            + self
                .layers
                .iter()
                .map(|l| {
                    l.input_layernorm_weight.len()
                        + l.post_attention_layernorm_weight.len()
                        + l.q_proj_weight.len()
                        + l.k_proj_weight.len()
                        + l.v_proj_weight.len()
                        + l.o_proj_weight.len()
                        + l.q_norm_weight.len()
                        + l.k_norm_weight.len()
                        + l.gate_proj_weight.len()
                        + l.up_proj_weight.len()
                        + l.down_proj_weight.len()
                })
                .sum::<usize>()
    }
}

/// Stream one release projection from the official GGUF as row-major little-
/// endian f32. This is the narrow bridge used by the public-CoreML exporter;
/// it deliberately dequantizes one matrix at a time instead of materializing
/// the complete assistant.
#[doc(hidden)]
pub fn write_gguf_projection_f32(
    path: &Path,
    canonical_name: &str,
    mut output: impl Write,
) -> Result<(usize, usize), DFlashError> {
    let gguf = crate::gguf::GgufFile::parse_path(path)
        .map_err(|error| DFlashError::Gguf(error.to_string()))?;
    let config = DFlashConfig::from_gguf(&gguf)?;
    let gguf_name = canonical_projection_name(canonical_name, config.num_hidden_layers)?;
    let mapped = crate::weights::MuseWeights::open(path, &gguf)
        .map_err(|error| DFlashError::Gguf(error.to_string()))?;
    let tensor = mapped
        .view(&gguf_name)
        .map_err(|error| DFlashError::Gguf(error.to_string()))?;
    let mut row = vec![0.0f32; tensor.n_in];
    let mut encoded = vec![0u8; tensor.n_in * size_of::<f32>()];
    for row_index in 0..tensor.n_out {
        crate::weights::dequant_row(&tensor, row_index, &mut row);
        for (bytes, value) in encoded.chunks_exact_mut(4).zip(&row) {
            bytes.copy_from_slice(&value.to_le_bytes());
        }
        output
            .write_all(&encoded)
            .map_err(|error| DFlashError::Io(path.to_path_buf(), error))?;
    }
    Ok((tensor.n_out, tensor.n_in))
}

fn canonical_projection_name(name: &str, layers: usize) -> Result<String, DFlashError> {
    if name == "fc.weight" {
        return Ok(name.into());
    }
    if let Some(rest) = name.strip_prefix("layers.") {
        let parts = rest.split('.').collect::<Vec<_>>();
        if let [layer, "post_attention_layernorm", "weight"] = parts.as_slice() {
            let layer = layer
                .parse::<usize>()
                .ok()
                .filter(|value| *value < layers)
                .ok_or_else(|| DFlashError::MissingTensor(name.into()))?;
            return Ok(format!("blk.{layer}.ffn_norm.weight"));
        }
    }
    let mut parts = name.split('.');
    let valid = parts.next() == Some("layers");
    let layer = parts.next().and_then(|value| value.parse::<usize>().ok());
    let family = parts.next();
    let projection = parts.next();
    let suffix = parts.next();
    if !valid || suffix != Some("weight") || parts.next().is_some() {
        return Err(DFlashError::MissingTensor(name.into()));
    }
    let layer = layer
        .filter(|value| *value < layers)
        .ok_or_else(|| DFlashError::MissingTensor(name.into()))?;
    let suffix = match (family, projection) {
        (Some("self_attn"), Some("q_proj")) => "attn_q",
        (Some("self_attn"), Some("k_proj")) => "attn_k",
        (Some("self_attn"), Some("v_proj")) => "attn_v",
        (Some("self_attn"), Some("o_proj")) => "attn_output",
        (Some("mlp"), Some("gate_proj")) => "ffn_gate",
        (Some("mlp"), Some("up_proj")) => "ffn_up",
        (Some("mlp"), Some("down_proj")) => "ffn_down",
        _ => return Err(DFlashError::MissingTensor(name.into())),
    };
    Ok(format!("blk.{layer}.{suffix}.weight"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn metadata_string(bytes: &mut Vec<u8>, key: &str, value: &str) {
        string(bytes, key);
        bytes.extend_from_slice(&8u32.to_le_bytes());
        string(bytes, value);
    }

    fn metadata_u32(bytes: &mut Vec<u8>, key: &str, value: u32) {
        string(bytes, key);
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn metadata_f32(bytes: &mut Vec<u8>, key: &str, value: f32) {
        string(bytes, key);
        bytes.extend_from_slice(&6u32.to_le_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn metadata_empty_strings(bytes: &mut Vec<u8>, key: &str, count: usize) {
        string(bytes, key);
        bytes.extend_from_slice(&9u32.to_le_bytes());
        bytes.extend_from_slice(&8u32.to_le_bytes());
        bytes.extend_from_slice(&(count as u64).to_le_bytes());
        for _ in 0..count {
            string(bytes, "");
        }
    }

    fn tiny_official_gguf() -> Vec<u8> {
        let mut tensors = vec![
            ("fc.weight".to_string(), vec![10, 2]),
            ("enc.output_norm.weight".to_string(), vec![2]),
            ("output_norm.weight".to_string(), vec![2]),
        ];
        for layer in 0..5 {
            let p = format!("blk.{layer}");
            for (suffix, shape) in [
                ("attn_norm.weight", vec![2]),
                ("ffn_norm.weight", vec![2]),
                ("attn_q.weight", vec![2, 2]),
                ("attn_k.weight", vec![2, 1]),
                ("attn_v.weight", vec![2, 1]),
                ("attn_output.weight", vec![2, 2]),
                ("attn_q_norm.weight", vec![1]),
                ("attn_k_norm.weight", vec![1]),
                ("ffn_gate.weight", vec![2, 3]),
                ("ffn_up.weight", vec![2, 3]),
                ("ffn_down.weight", vec![3, 2]),
            ] {
                tensors.push((format!("{p}.{suffix}"), shape));
            }
        }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0x4655_4747u32.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&17u64.to_le_bytes());
        metadata_string(&mut bytes, "general.architecture", "dflash");
        for (key, value) in [
            ("dflash.block_size", 16),
            ("tokenizer.ggml.bos_token_id", 1),
            ("tokenizer.ggml.eos_token_id", 2),
            ("tokenizer.ggml.mask_token_id", 99),
            ("dflash.embedding_length", 2),
            ("dflash.attention.key_length", 1),
            ("dflash.attention.value_length", 1),
            ("dflash.feed_forward_length", 3),
            ("dflash.attention.head_count", 2),
            ("dflash.block_count", 5),
            ("dflash.attention.head_count_kv", 1),
            ("dflash.context_length", 1024),
        ] {
            metadata_u32(&mut bytes, key, value);
        }
        metadata_f32(&mut bytes, "dflash.attention.layer_norm_rms_epsilon", 1e-6);
        metadata_f32(&mut bytes, "dflash.rope.freq_base", 10_000.0);
        metadata_empty_strings(&mut bytes, "tokenizer.ggml.tokens", 100);
        string(&mut bytes, "dflash.target_layers");
        bytes.extend_from_slice(&9u32.to_le_bytes());
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&5u64.to_le_bytes());
        for layer in [1u32, 10, 20, 30, 52] {
            bytes.extend_from_slice(&layer.to_le_bytes());
        }
        let mut offset = 0u64;
        for (name, shape) in &tensors {
            string(&mut bytes, name);
            bytes.extend_from_slice(&(shape.len() as u32).to_le_bytes());
            for dimension in shape {
                bytes.extend_from_slice(&(*dimension as u64).to_le_bytes());
            }
            bytes.extend_from_slice(&0u32.to_le_bytes());
            bytes.extend_from_slice(&offset.to_le_bytes());
            offset += shape.iter().product::<usize>() as u64 * 4;
        }
        let aligned = bytes.len().div_ceil(32) * 32;
        bytes.resize(aligned, 0);
        for index in 0..offset / 4 {
            bytes.extend_from_slice(&(index as f32 / 100.0).to_le_bytes());
        }
        bytes
    }

    #[test]
    fn loads_official_gguf_tensor_names_and_geometry() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("dflash-kquant.gguf");
        std::fs::write(&path, tiny_official_gguf()).unwrap();
        let weights = DFlashWeights::load(&path).unwrap();
        assert_eq!(
            weights.config.dflash_config.target_layer_ids,
            [0, 9, 19, 29, 51]
        );
        assert_eq!(weights.fc_weight.len(), 20);
        assert_eq!(weights.layers.len(), 5);
        assert_eq!(weights.layers[4].down_proj_weight.len(), 6);
        assert!(weights.param_count() > 100);
    }

    #[test]
    fn streams_one_canonical_projection_from_official_gguf() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("dflash-kquant.gguf");
        std::fs::write(&path, tiny_official_gguf()).unwrap();
        let mut bytes = Vec::new();
        let shape =
            write_gguf_projection_f32(&path, "layers.4.mlp.down_proj.weight", &mut bytes).unwrap();
        assert_eq!(shape, (2, 3));
        assert_eq!(bytes.len(), 6 * 4);
        assert!(bytes
            .chunks_exact(4)
            .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
            .all(f32::is_finite));

        bytes.clear();
        let shape = write_gguf_projection_f32(
            &path,
            "layers.4.post_attention_layernorm.weight",
            &mut bytes,
        )
        .unwrap();
        assert_eq!(shape, (1, 2));
        assert_eq!(bytes.len(), 2 * 4);
    }
}
