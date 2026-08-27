//! Release contract and serialized runner for public-CoreML DFlash shards.
//!
//! The release backend deliberately accepts a much narrower artifact surface
//! than Core ML itself: ordered INT8 1x1-convolution programs, each smaller
//! than 250 MiB, with exact model identities and content digests.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::sync::Mutex;

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::coreml::CoreMlModel;
use crate::dflash::{
    DFlashAttentionProjections, DFlashConfig, DFlashFusedAttentionInput,
    DFlashFusedAttentionOutput, DFlashProjectionBackend, DFlashStatefulAttentionInput,
};

mod attention_state;
use attention_state::LoadedStatefulAttention;
mod fused_attention;
use fused_attention::LoadedFusedAttention;

pub const ANE_MANIFEST_VERSION: u32 = 9;
pub const MAX_SHARD_BYTES: u64 = 250 * 1024 * 1024;

pub fn file_sha256(path: &Path) -> Result<String, String> {
    let mut digest = Sha256::new();
    update_digest_from_file(&mut digest, path)?;
    Ok(format!("{:x}", digest.finalize()))
}

pub fn dflash_artifact_identity(model_dir: &Path) -> Result<String, String> {
    if model_dir.is_file() {
        return file_sha256(model_dir);
    }
    let mut digest = Sha256::new();
    digest.update(b"muser-dflash-artifact-v1\0");
    update_digest_from_file(&mut digest, &model_dir.join("config.json"))?;
    update_digest_from_file(&mut digest, &model_dir.join("model.safetensors"))?;
    Ok(format!("{:x}", digest.finalize()))
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AneDFlashManifest {
    pub version: u32,
    pub backend: String,
    pub compute_units: String,
    pub weight_dtype: String,
    pub projection_operator: String,
    pub target_identity: String,
    pub dflash_identity: String,
    pub dflash_source_format: String,
    pub extractor_sha256: Option<String>,
    pub assistant_layers: usize,
    pub block_size: usize,
    pub shards: Vec<AneShardManifest>,
    #[serde(default)]
    pub ffn_shards: Vec<AneFfnShardManifest>,
    #[serde(default)]
    pub tail_shards: Vec<AneTailShardManifest>,
    #[serde(default)]
    pub attention_shards: Vec<AneAttentionShardManifest>,
    #[serde(default)]
    pub fused_attention_shards: Vec<AneFusedAttentionShardManifest>,
}

impl DFlashProjectionBackend for AneDFlashBackend {
    fn fused_stateful_attention_layer(
        &self,
        layer: usize,
        input: DFlashFusedAttentionInput<'_>,
    ) -> Result<Option<DFlashFusedAttentionOutput>, String> {
        // The v9 graph has a fixed 16-row ABI. Prefill may present more rows;
        // retain the independently receipted v8 QKV + attention path instead
        // of turning an optional optimization into a geometry failure.
        if input.target_rows > input.block_size {
            return Ok(None);
        }
        let _guard = self
            .inference_lock
            .lock()
            .map_err(|_| "ANE inference lock poisoned")?;
        let Some(shard) = self
            .fused_attention_shards
            .iter()
            .find(|shard| shard.layer() == layer)
        else {
            return Ok(None);
        };
        shard.predict(input).map(Some)
    }

    fn supports_exact_mirror_overlap(&self) -> bool {
        // Mirror-SD keeps the target accelerator permit across provisional
        // drafting. Only v8's capture-FC route can draft without reacquiring
        // that permit; v7/v9 must use the ordinary prepare/verify path.
        exact_mirror_overlap_supported(
            self.manifest.version,
            self.attention_shards.len(),
            self.manifest.assistant_layers,
        )
    }

    fn supports_capture_fc_pipeline(&self) -> bool {
        self.manifest.version == 8
    }

    fn project_capture_fc_slice(
        &self,
        capture: usize,
        input: &[f32],
    ) -> Result<Option<Vec<f32>>, String> {
        AneDFlashBackend::project_capture_fc_slice(self, capture, input).map(Some)
    }

    fn project(&self, name: &str, input: &[f32]) -> Result<Vec<f32>, String> {
        AneDFlashBackend::project(self, name, input)
    }

    fn project_group(&self, names: &[&str], input: &[f32]) -> Result<Vec<Vec<f32>>, String> {
        AneDFlashBackend::project_group(self, names, input)
    }

    fn fused_ffn(&self, layer: usize, input: &[f32]) -> Result<Option<Vec<f32>>, String> {
        AneDFlashBackend::fused_ffn(self, layer, input)
    }

    fn has_fused_layer_tail(&self, layer: usize) -> bool {
        self.tail_shards
            .iter()
            .any(|shard| shard.manifest.layer == layer)
    }

    fn fused_layer_tail(
        &self,
        layer: usize,
        attention: &[f32],
        residual: &[f32],
    ) -> Result<Option<Vec<f32>>, String> {
        AneDFlashBackend::fused_layer_tail(self, layer, attention, residual)
    }

    fn fused_layer_tail_into(
        &self,
        layer: usize,
        attention: &[f32],
        residual: &[f32],
        output: &mut [f32],
    ) -> Result<bool, String> {
        AneDFlashBackend::fused_layer_tail_into(self, layer, attention, residual, output)
    }

    fn attention_projections(
        &self,
        layer: usize,
        noise: &[f32],
        target: &[f32],
    ) -> Result<DFlashAttentionProjections, String> {
        AneDFlashBackend::attention_projections(self, layer, noise, target)
    }

    fn stateful_attention(
        &self,
        layer: usize,
        input: DFlashStatefulAttentionInput<'_>,
    ) -> Result<Option<Vec<f32>>, String> {
        AneDFlashBackend::stateful_attention(self, layer, input)
    }

    fn stateful_attention_into(
        &self,
        layer: usize,
        input: DFlashStatefulAttentionInput<'_>,
        output: &mut [f32],
    ) -> Result<bool, String> {
        AneDFlashBackend::stateful_attention_into(self, layer, input, output)
    }
}

fn exact_mirror_overlap_supported(version: u32, attention_layers: usize, layers: usize) -> bool {
    version == 8 && attention_layers == layers
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AneShardComponentManifest {
    pub projection: String,
    /// Channel offset in the physical Core ML result.
    pub output_offset: usize,
    /// Channel offset in the logical projection result.
    pub projection_offset: usize,
    pub output_width: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AneShardManifest {
    pub order: usize,
    pub path: PathBuf,
    pub input_name: String,
    pub output_name: String,
    pub projection: String,
    pub input_offset: usize,
    pub input_width: usize,
    pub output_offset: usize,
    pub output_width: usize,
    pub input_shape: Vec<usize>,
    pub input_elements: usize,
    pub output_elements: usize,
    pub bytes: u64,
    pub sha256: String,
    #[serde(default)]
    pub components: Vec<AneShardComponentManifest>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AneFfnShardManifest {
    pub order: usize,
    pub layer: usize,
    pub path: PathBuf,
    pub input_name: String,
    pub output_name: String,
    pub intermediate_offset: usize,
    pub intermediate_width: usize,
    pub input_shape: Vec<usize>,
    pub input_elements: usize,
    pub output_elements: usize,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AneTailShardManifest {
    pub order: usize,
    pub layer: usize,
    pub head: bool,
    pub path: PathBuf,
    pub input_name: String,
    pub output_name: String,
    pub intermediate_offset: usize,
    pub intermediate_width: usize,
    pub input_shape: Vec<usize>,
    pub input_elements: usize,
    pub output_elements: usize,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AneAttentionShardManifest {
    pub order: usize,
    pub layer: usize,
    pub path: PathBuf,
    pub max_context: usize,
    pub query_size: usize,
    pub chunk: usize,
    pub kv_write_chunk: usize,
    pub kv_join: String,
    pub state_group_kv_heads: usize,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AneFusedAttentionShardManifest {
    pub order: usize,
    pub layer: usize,
    pub path: PathBuf,
    pub max_context: usize,
    pub block_size: usize,
    pub query_size: usize,
    pub kv_write_chunk: usize,
    pub kv_join: String,
    pub state_group_kv_heads: usize,
    pub bundle_channels: usize,
    pub bytes: u64,
    pub sha256: String,
}

struct LoadedShard {
    manifest: AneShardManifest,
    model: CoreMlModel,
    input_scratch: Mutex<Vec<f32>>,
    output_scratch: Mutex<Vec<f32>>,
}

struct LoadedFfnShard {
    manifest: AneFfnShardManifest,
    model: CoreMlModel,
    input_scratch: Mutex<Vec<f32>>,
    output_scratch: Mutex<Vec<f32>>,
}

struct LoadedTailShard {
    manifest: AneTailShardManifest,
    model: CoreMlModel,
    input_scratch: Mutex<Vec<f32>>,
    output_scratch: Mutex<Vec<f32>>,
}

struct TailLogicalScratch {
    post_attention: Vec<f32>,
    normed: Vec<f32>,
}

/// A public-CoreML shard graph. One lock spans the complete graph so two
/// requests can never overlap on the Neural Engine.
pub struct AneDFlashBackend {
    manifest: AneDFlashManifest,
    shards: Vec<LoadedShard>,
    ffn_shards: Vec<LoadedFfnShard>,
    tail_shards: Vec<LoadedTailShard>,
    attention_shards: Vec<LoadedStatefulAttention>,
    fused_attention_shards: Vec<LoadedFusedAttention>,
    tail_logical_scratch: Mutex<TailLogicalScratch>,
    inference_lock: Mutex<()>,
}

impl AneDFlashBackend {
    pub fn load(
        manifest_path: &Path,
        expected_target: &str,
        expected_dflash: &str,
        config: &DFlashConfig,
    ) -> Result<Self, String> {
        let bytes = std::fs::read(manifest_path)
            .map_err(|e| format!("read ANE manifest {}: {e}", manifest_path.display()))?;
        let manifest: AneDFlashManifest = serde_json::from_slice(&bytes)
            .map_err(|e| format!("parse ANE manifest {}: {e}", manifest_path.display()))?;
        manifest.validate(expected_target, expected_dflash, config)?;
        let root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
        let mut shards = Vec::with_capacity(manifest.shards.len());
        for spec in &manifest.shards {
            let path = resolve_beneath(root, &spec.path)?;
            let actual_bytes = tree_size(&path)?;
            if actual_bytes != spec.bytes {
                return Err(format!(
                    "ANE shard {} byte count mismatch: manifest {}, actual {actual_bytes}",
                    spec.order, spec.bytes
                ));
            }
            if actual_bytes > MAX_SHARD_BYTES {
                return Err(format!(
                    "ANE shard {} is {actual_bytes} bytes, over {MAX_SHARD_BYTES}",
                    spec.order
                ));
            }
            let actual_digest = tree_sha256(&path)?;
            if actual_digest != spec.sha256 {
                return Err(format!("ANE shard {} SHA-256 mismatch", spec.order));
            }
            let model = CoreMlModel::load(
                &path,
                &spec.input_name,
                &spec.output_name,
                &spec.input_shape,
                &[
                    1,
                    spec.output_width,
                    spec.input_shape.get(2).copied().ok_or_else(|| {
                        format!("ANE shard {} input shape has no token axis", spec.order)
                    })?,
                    1,
                ],
            )?;
            shards.push(LoadedShard {
                manifest: spec.clone(),
                model,
                input_scratch: Mutex::new(vec![0.0; spec.input_elements]),
                output_scratch: Mutex::new(vec![0.0; spec.output_elements]),
            });
        }
        let mut ffn_shards = Vec::with_capacity(manifest.ffn_shards.len());
        for spec in &manifest.ffn_shards {
            let path = resolve_beneath(root, &spec.path)?;
            let actual_bytes = tree_size(&path)?;
            if actual_bytes != spec.bytes || actual_bytes > MAX_SHARD_BYTES {
                return Err(format!(
                    "ANE fused FFN shard {}/{} byte count is invalid",
                    spec.layer, spec.order
                ));
            }
            if tree_sha256(&path)? != spec.sha256 {
                return Err(format!(
                    "ANE fused FFN shard {}/{} SHA-256 mismatch",
                    spec.layer, spec.order
                ));
            }
            let model = CoreMlModel::load(
                &path,
                &spec.input_name,
                &spec.output_name,
                &spec.input_shape,
                &[
                    1,
                    spec.output_elements
                        / spec.input_shape.get(2).copied().ok_or_else(|| {
                            format!(
                                "ANE fused FFN shard {}/{} input shape has no token axis",
                                spec.layer, spec.order
                            )
                        })?,
                    spec.input_shape[2],
                    1,
                ],
            )?;
            ffn_shards.push(LoadedFfnShard {
                manifest: spec.clone(),
                model,
                input_scratch: Mutex::new(vec![0.0; spec.input_elements]),
                output_scratch: Mutex::new(vec![0.0; spec.output_elements]),
            });
        }
        let mut tail_shards = Vec::with_capacity(manifest.tail_shards.len());
        for spec in &manifest.tail_shards {
            let path = resolve_beneath(root, &spec.path)?;
            let actual_bytes = tree_size(&path)?;
            if actual_bytes != spec.bytes || actual_bytes > MAX_SHARD_BYTES {
                return Err(format!(
                    "ANE fused tail shard {}/{} byte count is invalid",
                    spec.layer, spec.order
                ));
            }
            if tree_sha256(&path)? != spec.sha256 {
                return Err(format!(
                    "ANE fused tail shard {}/{} SHA-256 mismatch",
                    spec.layer, spec.order
                ));
            }
            let batch = spec.input_shape.get(2).copied().ok_or_else(|| {
                format!(
                    "ANE fused tail shard {}/{} input shape has no token axis",
                    spec.layer, spec.order
                )
            })?;
            let output_channels = spec.output_elements / batch;
            let model = CoreMlModel::load(
                &path,
                &spec.input_name,
                &spec.output_name,
                &spec.input_shape,
                &[1, output_channels, batch, 1],
            )?;
            tail_shards.push(LoadedTailShard {
                manifest: spec.clone(),
                model,
                input_scratch: Mutex::new(vec![0.0; spec.input_elements]),
                output_scratch: Mutex::new(vec![0.0; spec.output_elements]),
            });
        }
        let mut attention_shards = Vec::with_capacity(manifest.attention_shards.len());
        for spec in &manifest.attention_shards {
            let path = resolve_beneath(root, &spec.path)?;
            let actual_bytes = tree_size(&path)?;
            if actual_bytes != spec.bytes || actual_bytes > MAX_SHARD_BYTES {
                return Err(format!(
                    "ANE stateful attention layer {} byte count is invalid",
                    spec.layer
                ));
            }
            if tree_sha256(&path)? != spec.sha256 {
                return Err(format!(
                    "ANE stateful attention layer {} SHA-256 mismatch",
                    spec.layer
                ));
            }
            attention_shards.push(LoadedStatefulAttention::load(path, spec.clone(), config)?);
        }
        let mut fused_attention_shards = Vec::with_capacity(manifest.fused_attention_shards.len());
        for spec in &manifest.fused_attention_shards {
            let path = resolve_beneath(root, &spec.path)?;
            let actual_bytes = tree_size(&path)?;
            if actual_bytes != spec.bytes || actual_bytes > MAX_SHARD_BYTES {
                return Err(format!(
                    "ANE fused attention layer {} byte count is invalid",
                    spec.layer
                ));
            }
            if tree_sha256(&path)? != spec.sha256 {
                return Err(format!(
                    "ANE fused attention layer {} SHA-256 mismatch",
                    spec.layer
                ));
            }
            fused_attention_shards.push(LoadedFusedAttention::load(path, spec.clone(), config)?);
        }
        Ok(Self {
            manifest,
            shards,
            ffn_shards,
            tail_shards,
            attention_shards,
            fused_attention_shards,
            tail_logical_scratch: Mutex::new(TailLogicalScratch {
                post_attention: vec![0.0; config.block_size * config.hidden_size],
                normed: vec![0.0; config.block_size * config.hidden_size],
            }),
            inference_lock: Mutex::new(()),
        })
    }

    pub fn manifest(&self) -> &AneDFlashManifest {
        &self.manifest
    }

    /// Execute one declared projection. The graph-level lock remains held for
    /// the call, enforcing the one-CoreML-inference-at-a-time contract. The
    /// server's outer target-session lease prevents another request from
    /// interleaving between projections in the same draft round.
    fn project_v1_unlocked(&self, projection: &str, input: &[f32]) -> Result<Vec<f32>, String> {
        let shards = self
            .shards
            .iter()
            .filter(|shard| shard.manifest.projection == projection)
            .collect::<Vec<_>>();
        if shards.is_empty() {
            return Err(format!("ANE projection {projection} is absent"));
        }
        let batch = self.manifest.block_size;
        let input_width = shards
            .iter()
            .map(|shard| shard.manifest.input_offset + shard.manifest.input_width)
            .max()
            .unwrap_or(0);
        let output_width = shards
            .iter()
            .map(|shard| shard.manifest.output_offset + shard.manifest.output_width)
            .max()
            .unwrap_or(0);
        if input.is_empty() || !input.len().is_multiple_of(input_width) {
            return Err(format!(
                "ANE projection {projection} expected rows of {input_width} inputs, got {}",
                input.len()
            ));
        }
        let rows = input.len() / input_width;
        let mut output = vec![0.0f32; rows * output_width];
        for start in (0..rows).step_by(batch) {
            let count = (rows - start).min(batch);
            for shard in &shards {
                let spec = &shard.manifest;
                let mut padded = shard
                    .input_scratch
                    .lock()
                    .map_err(|_| format!("ANE shard {} input scratch lock poisoned", spec.order))?;
                padded.fill(0.0);
                for row in 0..count {
                    let source = (start + row) * input_width + spec.input_offset;
                    for column in 0..spec.input_width {
                        let destination = coreml_index(
                            self.manifest.version,
                            row,
                            column,
                            batch,
                            spec.input_width,
                        );
                        padded[destination] = input[source + column];
                    }
                }
                let mut projected = shard.output_scratch.lock().map_err(|_| {
                    format!("ANE shard {} output scratch lock poisoned", spec.order)
                })?;
                shard
                    .model
                    .predict_into(&padded, &spec.input_shape, &mut projected)?;
                for row in 0..count {
                    let destination = (start + row) * output_width + spec.output_offset;
                    for column in 0..spec.output_width {
                        let source = coreml_index(
                            self.manifest.version,
                            row,
                            column,
                            batch,
                            spec.output_width,
                        );
                        output[destination + column] += projected[source];
                    }
                }
            }
        }
        Ok(output)
    }

    fn project_group_unlocked(
        &self,
        projections: &[&str],
        input: &[f32],
    ) -> Result<Vec<Vec<f32>>, String> {
        if projections.is_empty() {
            return Err("ANE projection group is empty".into());
        }
        if self.manifest.version == 1 {
            return projections
                .iter()
                .map(|projection| self.project_v1_unlocked(projection, input))
                .collect();
        }
        let requested = projections.iter().copied().collect::<BTreeSet<_>>();
        if requested.len() != projections.len() {
            return Err("ANE projection group contains duplicates".into());
        }
        let physical_group = self
            .shards
            .iter()
            .map(|shard| shard.manifest.projection.as_str())
            .find(|group| {
                let available = self
                    .shards
                    .iter()
                    .filter(|shard| shard.manifest.projection == *group)
                    .flat_map(|shard| shard.manifest.components.iter())
                    .map(|component| component.projection.as_str())
                    .collect::<BTreeSet<_>>();
                requested.is_subset(&available)
            })
            .ok_or_else(|| format!("ANE fused projection group {projections:?} is absent"))?;
        let shards = self
            .shards
            .iter()
            .filter(|shard| shard.manifest.projection == physical_group)
            .collect::<Vec<_>>();
        let input_width = shards
            .iter()
            .map(|shard| shard.manifest.input_offset + shard.manifest.input_width)
            .max()
            .unwrap_or(0);
        if input_width == 0 || input.is_empty() || !input.len().is_multiple_of(input_width) {
            return Err(format!(
                "ANE fused projection {physical_group} expected rows of {input_width} inputs, got {}",
                input.len()
            ));
        }
        let rows = input.len() / input_width;
        let mut outputs = projections
            .iter()
            .map(|projection| {
                let width = shards
                    .iter()
                    .flat_map(|shard| shard.manifest.components.iter())
                    .filter(|component| component.projection == *projection)
                    .map(|component| component.projection_offset + component.output_width)
                    .max()
                    .unwrap_or(0);
                vec![0.0f32; rows * width]
            })
            .collect::<Vec<_>>();
        let batch = shards[0]
            .manifest
            .input_shape
            .get(2)
            .copied()
            .unwrap_or(self.manifest.block_size);
        if shards
            .iter()
            .any(|shard| shard.manifest.input_shape.get(2).copied() != Some(batch))
        {
            return Err(format!(
                "ANE fused projection {physical_group} mixes token capacities"
            ));
        }
        for start in (0..rows).step_by(batch) {
            let count = (rows - start).min(batch);
            for shard in &shards {
                let spec = &shard.manifest;
                let mut padded = shard
                    .input_scratch
                    .lock()
                    .map_err(|_| format!("ANE shard {} input scratch lock poisoned", spec.order))?;
                padded.fill(0.0);
                for row in 0..count {
                    let source = (start + row) * input_width + spec.input_offset;
                    for column in 0..spec.input_width {
                        let destination = coreml_index(
                            self.manifest.version,
                            row,
                            column,
                            batch,
                            spec.input_width,
                        );
                        padded[destination] = input[source + column];
                    }
                }
                let mut projected = shard.output_scratch.lock().map_err(|_| {
                    format!("ANE shard {} output scratch lock poisoned", spec.order)
                })?;
                shard
                    .model
                    .predict_into(&padded, &spec.input_shape, &mut projected)?;
                for component in &spec.components {
                    let Some(result_index) = projections
                        .iter()
                        .position(|projection| *projection == component.projection)
                    else {
                        continue;
                    };
                    let logical_width = outputs[result_index].len() / rows;
                    for row in 0..count {
                        let destination =
                            (start + row) * logical_width + component.projection_offset;
                        for column in 0..component.output_width {
                            let source = coreml_index(
                                self.manifest.version,
                                row,
                                component.output_offset + column,
                                batch,
                                spec.output_width,
                            );
                            outputs[result_index][destination + column] += projected[source];
                        }
                    }
                }
            }
        }
        Ok(outputs)
    }

    pub fn attention_projections(
        &self,
        layer: usize,
        noise: &[f32],
        target: &[f32],
    ) -> Result<DFlashAttentionProjections, String> {
        let _guard = self
            .inference_lock
            .lock()
            .map_err(|_| "ANE inference lock poisoned")?;
        let q_name = format!("layers.{layer}.q_proj");
        let k_name = format!("layers.{layer}.k_proj");
        let v_name = format!("layers.{layer}.v_proj");
        let physical_group = format!("layers.{layer}.qkv_fused");
        let input_width = self
            .shards
            .iter()
            .filter(|shard| shard.manifest.projection == physical_group)
            .map(|shard| shard.manifest.input_offset + shard.manifest.input_width)
            .max()
            .ok_or_else(|| format!("ANE attention layer {layer} is absent"))?;
        if noise.is_empty()
            || target.is_empty()
            || !noise.len().is_multiple_of(input_width)
            || !target.len().is_multiple_of(input_width)
        {
            return Err(format!(
                "ANE attention layer {layer} input geometry is invalid"
            ));
        }
        let noise_rows = noise.len() / input_width;
        let target_rows = target.len() / input_width;
        let capacity = self
            .shards
            .iter()
            .find(|shard| shard.manifest.projection == physical_group)
            .and_then(|shard| shard.manifest.input_shape.get(2))
            .copied()
            .unwrap_or(self.manifest.block_size);
        if noise_rows + target_rows > capacity {
            let mut noise_values =
                self.project_group_unlocked(&[&q_name, &k_name, &v_name], noise)?;
            let mut target_values = self.project_group_unlocked(&[&k_name, &v_name], target)?;
            return Ok(DFlashAttentionProjections {
                q: noise_values.remove(0),
                k: noise_values.remove(0),
                v: noise_values.remove(0),
                k_target: target_values.remove(0),
                v_target: target_values.remove(0),
            });
        }
        let mut combined = Vec::with_capacity(noise.len() + target.len());
        combined.extend_from_slice(noise);
        combined.extend_from_slice(target);
        let values = self.project_group_unlocked(&[&q_name, &k_name, &v_name], &combined)?;
        let split = |value: &[f32]| {
            let width = value.len() / (noise_rows + target_rows);
            let boundary = noise_rows * width;
            (value[..boundary].to_vec(), value[boundary..].to_vec())
        };
        let (q, _) = split(&values[0]);
        let (k, k_target) = split(&values[1]);
        let (v, v_target) = split(&values[2]);
        Ok(DFlashAttentionProjections {
            q,
            k,
            v,
            k_target,
            v_target,
        })
    }

    pub fn project_capture_fc_slice(
        &self,
        capture: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        if self.manifest.version < 8 {
            return Err("ANE capture FC slices require a v8 manifest".into());
        }
        let _guard = self
            .inference_lock
            .lock()
            .map_err(|_| "ANE inference lock poisoned")?;
        let shards = self
            .shards
            .iter()
            .filter(|shard| shard.manifest.projection == "fc")
            .collect::<Vec<_>>();
        let shard = shards
            .get(capture)
            .ok_or_else(|| format!("ANE capture FC slice {capture} is absent"))?;
        let spec = &shard.manifest;
        if input.is_empty() || !input.len().is_multiple_of(spec.input_width) {
            return Err(format!(
                "ANE capture FC slice {capture} expected rows of {}, got {} inputs",
                spec.input_width,
                input.len()
            ));
        }
        let rows = input.len() / spec.input_width;
        let batch = spec.input_shape[2];
        if rows > batch {
            return Err(format!(
                "ANE capture FC slice {capture} has {rows} rows, capacity is {batch}"
            ));
        }
        let mut padded = shard
            .input_scratch
            .lock()
            .map_err(|_| format!("ANE capture FC slice {capture} input lock poisoned"))?;
        padded.fill(0.0);
        for row in 0..rows {
            for column in 0..spec.input_width {
                padded[coreml_index(self.manifest.version, row, column, batch, spec.input_width)] =
                    input[row * spec.input_width + column];
            }
        }
        let mut projected = shard
            .output_scratch
            .lock()
            .map_err(|_| format!("ANE capture FC slice {capture} output lock poisoned"))?;
        shard
            .model
            .predict_into(&padded, &spec.input_shape, &mut projected)?;
        let mut output = vec![0.0; rows * spec.output_width];
        for row in 0..rows {
            for column in 0..spec.output_width {
                output[row * spec.output_width + column] = projected
                    [coreml_index(self.manifest.version, row, column, batch, spec.output_width)];
            }
        }
        Ok(output)
    }

    pub fn project(&self, projection: &str, input: &[f32]) -> Result<Vec<f32>, String> {
        let _guard = self
            .inference_lock
            .lock()
            .map_err(|_| "ANE inference lock poisoned")?;
        if self.manifest.version == 1 {
            self.project_v1_unlocked(projection, input)
        } else {
            self.project_group_unlocked(&[projection], input)
                .map(|mut outputs| outputs.remove(0))
        }
    }

    pub fn project_group(
        &self,
        projections: &[&str],
        input: &[f32],
    ) -> Result<Vec<Vec<f32>>, String> {
        let _guard = self
            .inference_lock
            .lock()
            .map_err(|_| "ANE inference lock poisoned")?;
        self.project_group_unlocked(projections, input)
    }

    pub fn fused_ffn(&self, layer: usize, input: &[f32]) -> Result<Option<Vec<f32>>, String> {
        let _guard = self
            .inference_lock
            .lock()
            .map_err(|_| "ANE inference lock poisoned")?;
        let shards = self
            .ffn_shards
            .iter()
            .filter(|shard| shard.manifest.layer == layer)
            .collect::<Vec<_>>();
        if shards.is_empty() {
            return Ok(None);
        }
        let batch = self.manifest.block_size;
        let hidden = shards[0].manifest.output_elements / batch;
        if input.len() != batch * hidden {
            return Err(format!(
                "ANE fused FFN layer {layer} expected {} inputs, got {}",
                batch * hidden,
                input.len()
            ));
        }
        let mut output = vec![0.0f32; batch * hidden];
        for shard in shards {
            let spec = &shard.manifest;
            let mut padded = shard.input_scratch.lock().map_err(|_| {
                format!(
                    "ANE fused FFN shard {layer}/{} input scratch lock poisoned",
                    spec.order
                )
            })?;
            for row in 0..batch {
                for column in 0..hidden {
                    padded[coreml_index(self.manifest.version, row, column, batch, hidden)] =
                        input[row * hidden + column];
                }
            }
            let mut projected = shard.output_scratch.lock().map_err(|_| {
                format!(
                    "ANE fused FFN shard {layer}/{} output scratch lock poisoned",
                    spec.order
                )
            })?;
            shard
                .model
                .predict_into(&padded, &spec.input_shape, &mut projected)?;
            for row in 0..batch {
                for column in 0..hidden {
                    output[row * hidden + column] +=
                        projected[coreml_index(self.manifest.version, row, column, batch, hidden)];
                }
            }
        }
        Ok(Some(output))
    }

    pub fn fused_layer_tail(
        &self,
        layer: usize,
        attention: &[f32],
        residual: &[f32],
    ) -> Result<Option<Vec<f32>>, String> {
        let _guard = self
            .inference_lock
            .lock()
            .map_err(|_| "ANE inference lock poisoned")?;
        let mut output = vec![0.0; residual.len()];
        if !self.fused_layer_tail_unlocked(layer, attention, residual, &mut output)? {
            return Ok(None);
        }
        Ok(Some(output))
    }

    pub fn fused_layer_tail_into(
        &self,
        layer: usize,
        attention: &[f32],
        residual: &[f32],
        output: &mut [f32],
    ) -> Result<bool, String> {
        let _guard = self
            .inference_lock
            .lock()
            .map_err(|_| "ANE inference lock poisoned")?;
        self.fused_layer_tail_unlocked(layer, attention, residual, output)
    }

    fn fused_layer_tail_unlocked(
        &self,
        layer: usize,
        attention: &[f32],
        residual: &[f32],
        result: &mut [f32],
    ) -> Result<bool, String> {
        let Some(head) = self
            .tail_shards
            .iter()
            .find(|shard| shard.manifest.layer == layer && shard.manifest.head)
        else {
            return Ok(false);
        };
        let batch = self.manifest.block_size;
        if attention.is_empty() || !attention.len().is_multiple_of(batch) {
            return Err(format!(
                "ANE fused tail layer {layer} attention geometry is invalid"
            ));
        }
        let head_output_blocks = if self.manifest.version >= 6 { 2 } else { 3 };
        let hidden = head.manifest.output_elements / (head_output_blocks * batch);
        let attention_width = attention.len() / batch;
        let input_channels = attention_width + hidden;
        if hidden == 0
            || residual.len() != batch * hidden
            || result.len() != batch * hidden
            || head.manifest.input_shape.get(1).copied() != Some(input_channels)
        {
            return Err(format!(
                "ANE fused tail layer {layer} input geometry is invalid"
            ));
        }
        let mut packed = head
            .input_scratch
            .lock()
            .map_err(|_| format!("ANE fused tail shard {layer}/0 input scratch lock poisoned"))?;
        for row in 0..batch {
            for column in 0..attention_width {
                packed[coreml_index(self.manifest.version, row, column, batch, input_channels)] =
                    attention[row * attention_width + column];
            }
            for column in 0..hidden {
                packed[coreml_index(
                    self.manifest.version,
                    row,
                    attention_width + column,
                    batch,
                    input_channels,
                )] = residual[row * hidden + column];
            }
        }
        let mut head_output = head
            .output_scratch
            .lock()
            .map_err(|_| format!("ANE fused tail shard {layer}/0 output scratch lock poisoned"))?;
        head.model
            .predict_into(&packed, &head.manifest.input_shape, &mut head_output)?;
        let mut logical = self
            .tail_logical_scratch
            .lock()
            .map_err(|_| "ANE fused tail logical scratch lock poisoned")?;
        for row in 0..batch {
            for column in 0..hidden {
                if self.manifest.version >= 6 {
                    result[row * hidden + column] = head_output
                        [coreml_index(self.manifest.version, row, column, batch, 2 * hidden)];
                    logical.normed[row * hidden + column] = head_output[coreml_index(
                        self.manifest.version,
                        row,
                        hidden + column,
                        batch,
                        2 * hidden,
                    )];
                } else {
                    logical.post_attention[row * hidden + column] = head_output
                        [coreml_index(self.manifest.version, row, column, batch, 3 * hidden)];
                    logical.normed[row * hidden + column] = head_output[coreml_index(
                        self.manifest.version,
                        row,
                        hidden + column,
                        batch,
                        3 * hidden,
                    )];
                    result[row * hidden + column] = head_output[coreml_index(
                        self.manifest.version,
                        row,
                        2 * hidden + column,
                        batch,
                        3 * hidden,
                    )];
                }
            }
        }
        drop(head_output);
        drop(packed);
        for shard in self
            .tail_shards
            .iter()
            .filter(|shard| shard.manifest.layer == layer && !shard.manifest.head)
        {
            let spec = &shard.manifest;
            let mut input = shard.input_scratch.lock().map_err(|_| {
                format!(
                    "ANE fused tail shard {layer}/{} input scratch lock poisoned",
                    spec.order
                )
            })?;
            for row in 0..batch {
                for column in 0..hidden {
                    input[coreml_index(self.manifest.version, row, column, batch, hidden)] =
                        logical.normed[row * hidden + column];
                }
            }
            let mut shard_output = shard.output_scratch.lock().map_err(|_| {
                format!(
                    "ANE fused tail shard {layer}/{} output scratch lock poisoned",
                    spec.order
                )
            })?;
            shard
                .model
                .predict_into(&input, &spec.input_shape, &mut shard_output)?;
            for row in 0..batch {
                for column in 0..hidden {
                    result[row * hidden + column] += shard_output
                        [coreml_index(self.manifest.version, row, column, batch, hidden)];
                }
            }
        }
        if self.manifest.version < 6 {
            for (value, residual) in result.iter_mut().zip(&logical.post_attention) {
                *value += residual;
            }
        }
        Ok(true)
    }

    pub fn stateful_attention(
        &self,
        layer: usize,
        input: DFlashStatefulAttentionInput<'_>,
    ) -> Result<Option<Vec<f32>>, String> {
        let _guard = self
            .inference_lock
            .lock()
            .map_err(|_| "ANE inference lock poisoned")?;
        let Some(attention) = self
            .attention_shards
            .iter()
            .find(|attention| attention.layer() == layer)
        else {
            return Ok(None);
        };
        attention.predict(input).map(Some)
    }

    pub fn stateful_attention_into(
        &self,
        layer: usize,
        input: DFlashStatefulAttentionInput<'_>,
        output: &mut [f32],
    ) -> Result<bool, String> {
        let _guard = self
            .inference_lock
            .lock()
            .map_err(|_| "ANE inference lock poisoned")?;
        let Some(attention) = self
            .attention_shards
            .iter()
            .find(|attention| attention.layer() == layer)
        else {
            return Ok(false);
        };
        attention.predict_into(input, output)?;
        Ok(true)
    }

    pub fn with_exclusive<T>(
        &self,
        operation: impl FnOnce(&dyn Fn(&str, &[f32]) -> Result<Vec<f32>, String>) -> Result<T, String>,
    ) -> Result<T, String> {
        let _guard = self
            .inference_lock
            .lock()
            .map_err(|_| "ANE inference lock poisoned")?;
        operation(&|projection, input| {
            if self.manifest.version == 1 {
                self.project_v1_unlocked(projection, input)
            } else {
                self.project_group_unlocked(&[projection], input)
                    .map(|mut outputs| outputs.remove(0))
            }
        })
    }
}

impl AneDFlashManifest {
    fn validate(
        &self,
        expected_target: &str,
        expected_dflash: &str,
        config: &DFlashConfig,
    ) -> Result<(), String> {
        if !(1..=ANE_MANIFEST_VERSION).contains(&self.version)
            || self.backend != "public_coreml"
            || self.compute_units != "CPU_AND_NE"
            || self.weight_dtype != "int8"
            || self.projection_operator != "conv1x1"
        {
            return Err("ANE manifest is outside the release artifact contract".into());
        }
        if self.target_identity != expected_target || self.dflash_identity != expected_dflash {
            return Err("ANE target/DFlash identity mismatch".into());
        }
        match (self.dflash_source_format.as_str(), &self.extractor_sha256) {
            ("official-gguf", Some(digest)) if is_lower_sha256(digest) => {}
            ("development-safetensors", None) => {}
            _ => return Err("ANE DFlash source/extractor provenance is invalid".into()),
        }
        if self.assistant_layers != 5
            || self.assistant_layers != config.num_hidden_layers
            || self.block_size != config.block_size
        {
            return Err("ANE assistant geometry mismatch".into());
        }
        let expected = if self.version >= 5 {
            expected_attention_projections(config)
        } else if self.version >= 3 {
            expected_linear_projections(config)
        } else {
            expected_projections(config)
        };
        let geometry = projection_geometry(config);
        let mut found = BTreeSet::new();
        for (order, shard) in self.shards.iter().enumerate() {
            if shard.order != order
                || shard.input_name.is_empty()
                || shard.output_name.is_empty()
                || shard.projection.is_empty()
                || shard.input_shape.is_empty()
                || shard.input_shape.contains(&0)
                || shard.input_elements == 0
                || shard.output_elements == 0
                || shard.input_width == 0
                || shard.output_width == 0
                || shard.bytes == 0
                || shard.bytes > MAX_SHARD_BYTES
                || !is_lower_sha256(&shard.sha256)
            {
                return Err(format!("invalid ANE shard contract at order {order}"));
            }
            if self.version == 1 {
                if !shard.components.is_empty() {
                    return Err(format!("legacy ANE shard {order} has fused components"));
                }
                found.insert(shard.projection.clone());
            } else {
                if shard.components.is_empty() {
                    return Err(format!("fused ANE shard {order} has no components"));
                }
                found.extend(
                    shard
                        .components
                        .iter()
                        .map(|component| component.projection.clone()),
                );
                if !partitions_exact(
                    shard
                        .components
                        .iter()
                        .map(|component| (component.output_offset, component.output_width)),
                    shard.output_width,
                ) {
                    return Err(format!("ANE shard {order} physical outputs are not exact"));
                }
            }
            let input_elements = shard
                .input_shape
                .iter()
                .try_fold(1usize, |n, d| n.checked_mul(*d))
                .ok_or_else(|| format!("ANE shard {order} input shape overflow"))?;
            if input_elements != shard.input_elements {
                return Err(format!("ANE shard {order} input shape mismatch"));
            }
            let physical_input_width = if self.version == 1 {
                geometry
                    .get(&shard.projection)
                    .map(|geometry| geometry.0)
                    .ok_or_else(|| format!("ANE shard {order} has unknown projection"))?
            } else {
                let mut widths = shard.components.iter().map(|component| {
                    geometry
                        .get(&component.projection)
                        .map(|geometry| geometry.0)
                        .ok_or_else(|| {
                            format!("ANE shard {order} has unknown component projection")
                        })
                });
                let first = widths
                    .next()
                    .ok_or_else(|| format!("ANE shard {order} is empty"))??;
                if widths.any(|width| width != Ok(first)) {
                    return Err(format!(
                        "ANE shard {order} components have different inputs"
                    ));
                }
                first
            };
            let physical_output_valid = if self.version == 1 {
                let output_width = geometry[&shard.projection].1;
                shard.output_offset.saturating_add(shard.output_width) <= output_width
            } else {
                shard.components.iter().all(|component| {
                    geometry.get(&component.projection).is_some_and(|geometry| {
                        component
                            .projection_offset
                            .saturating_add(component.output_width)
                            <= geometry.1
                    })
                })
            };
            let token_capacity = if self.version >= 4 && shard.projection.ends_with(".qkv_fused") {
                2 * config.block_size
            } else {
                config.block_size
            };
            let expected_input_shape = if self.version >= 4 {
                vec![1, shard.input_width, token_capacity, 1]
            } else {
                vec![config.block_size, shard.input_width, 1, 1]
            };
            if shard.input_offset.saturating_add(shard.input_width) > physical_input_width
                || !physical_output_valid
                || shard.input_shape != expected_input_shape
                || shard.output_elements != token_capacity * shard.output_width
            {
                return Err(format!("ANE shard {order} projection geometry mismatch"));
            }
        }
        if found != expected {
            let missing = expected.difference(&found).cloned().collect::<Vec<_>>();
            let extra = found.difference(&expected).cloned().collect::<Vec<_>>();
            return Err(format!(
                "ANE projection set mismatch: missing {missing:?}, extra {extra:?}"
            ));
        }
        for projection in &expected {
            let &(input_width, output_width) = &geometry[projection];
            let parts = self
                .shards
                .iter()
                .flat_map(|shard| {
                    if self.version == 1 && &shard.projection == projection {
                        vec![(shard, shard.output_offset, shard.output_width)]
                    } else {
                        shard
                            .components
                            .iter()
                            .filter(|component| &component.projection == projection)
                            .map(|component| {
                                (shard, component.projection_offset, component.output_width)
                            })
                            .collect()
                    }
                })
                .collect::<Vec<_>>();
            let output_partition = parts
                .iter()
                .all(|(shard, _, _)| shard.input_offset == 0 && shard.input_width == input_width)
                && partitions_exact(
                    parts.iter().map(|(_, offset, width)| (*offset, *width)),
                    output_width,
                );
            let input_partition = parts
                .iter()
                .all(|(_, offset, width)| *offset == 0 && *width == output_width)
                && partitions_exact(
                    parts
                        .iter()
                        .map(|(shard, _, _)| (shard.input_offset, shard.input_width)),
                    input_width,
                );
            if !output_partition && !input_partition {
                return Err(format!(
                    "ANE projection {projection} slices are not an exact input or output partition"
                ));
            }
        }
        if self.version >= 5 {
            if !self.ffn_shards.is_empty() {
                return Err("ANE v5+ manifests use fused tails, not standalone FFN shards".into());
            }
            self.validate_fused_tails(config)?;
        } else if self.version < 3 {
            if !self.ffn_shards.is_empty() {
                return Err("legacy ANE manifests cannot contain fused FFN shards".into());
            }
            if !self.tail_shards.is_empty() {
                return Err("legacy ANE manifests cannot contain fused tail shards".into());
            }
        } else {
            if !self.tail_shards.is_empty() {
                return Err("ANE v3/v4 manifests cannot contain fused tail shards".into());
            }
            self.validate_fused_ffn(config)?;
        }
        if self.version >= 7 {
            self.validate_stateful_attention(config)?;
        } else if !self.attention_shards.is_empty() {
            return Err("ANE v1-v6 manifests cannot contain stateful attention".into());
        }
        if self.version == 8 {
            self.validate_capture_fc(config)?;
        }
        if self.version >= 9 {
            self.validate_fused_attention(config)?;
        } else if !self.fused_attention_shards.is_empty() {
            return Err("ANE v1-v8 manifests cannot contain fused attention layers".into());
        }
        Ok(())
    }

    fn validate_fused_attention(&self, config: &DFlashConfig) -> Result<(), String> {
        let expected_bundle =
            (config.num_attention_heads + 2 * config.num_key_value_heads) * config.head_dim;
        if self.fused_attention_shards.len() != config.num_hidden_layers
            || self.tail_shards.is_empty()
        {
            return Err(
                "ANE v9 fused attention must cover every layer and retain fused tails".into(),
            );
        }
        for (order, shard) in self.fused_attention_shards.iter().enumerate() {
            if shard.order != order
                || shard.layer != order
                || shard.max_context != 64 + 1024
                || shard.block_size != config.block_size
                || shard.query_size != config.block_size
                || shard.kv_write_chunk != config.block_size
                || shard.kv_join != "split"
                // One 8-head state is rejected by ANEF for the fused graph.
                // Two 4-head state groups retain one prediction while the
                // captured compute plan places every operation on ANE.
                || shard.state_group_kv_heads != 4
                || shard.bundle_channels != expected_bundle
                || shard.bytes == 0
                || shard.bytes > MAX_SHARD_BYTES
                || !is_lower_sha256(&shard.sha256)
            {
                return Err(format!(
                    "ANE v9 fused attention layer {order} contract is invalid"
                ));
            }
        }
        Ok(())
    }

    fn validate_capture_fc(&self, config: &DFlashConfig) -> Result<(), String> {
        let hidden = config.hidden_size;
        let expected_parts = config.dflash_config.target_layer_ids.len();
        let parts = self
            .shards
            .iter()
            .filter(|shard| shard.projection == "fc")
            .collect::<Vec<_>>();
        if parts.len() != expected_parts
            || parts.iter().enumerate().any(|(index, shard)| {
                shard.input_offset != index * hidden
                    || shard.input_width != hidden
                    || shard.output_width != hidden
                    || shard.components.len() != 1
                    || shard.components[0].projection != "fc"
                    || shard.components[0].projection_offset != 0
                    || shard.components[0].output_offset != 0
                    || shard.components[0].output_width != hidden
            })
        {
            return Err(
                "ANE v8 capture FC must contain one ordered full-output input slice per target layer"
                    .into(),
            );
        }
        Ok(())
    }

    fn validate_stateful_attention(&self, config: &DFlashConfig) -> Result<(), String> {
        if self.attention_shards.len() != config.num_hidden_layers {
            return Err("ANE stateful attention does not cover every assistant layer".into());
        }
        for (order, shard) in self.attention_shards.iter().enumerate() {
            if shard.order != order
                || shard.layer != order
                || shard.max_context != 64 + 1024
                || shard.query_size != 16
                || shard.chunk != 16
                || shard.kv_write_chunk != 16
                || shard.kv_join != "split"
                || shard.state_group_kv_heads != 8
                || shard.bytes == 0
                || shard.bytes > MAX_SHARD_BYTES
                || !is_lower_sha256(&shard.sha256)
            {
                return Err(format!(
                    "invalid ANE stateful attention contract at order {order}"
                ));
            }
        }
        Ok(())
    }

    fn validate_fused_tails(&self, config: &DFlashConfig) -> Result<(), String> {
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        for layer in 0..config.num_hidden_layers {
            let mut shards = self
                .tail_shards
                .iter()
                .filter(|shard| shard.layer == layer)
                .collect::<Vec<_>>();
            shards.sort_by_key(|shard| shard.order);
            if shards.is_empty()
                || !shards[0].head
                || shards.iter().skip(1).any(|shard| shard.head)
                || !partitions_exact(
                    shards
                        .iter()
                        .map(|shard| (shard.intermediate_offset, shard.intermediate_width)),
                    intermediate,
                )
            {
                return Err(format!("ANE fused tail layer {layer} is incomplete"));
            }
            for (order, shard) in shards.into_iter().enumerate() {
                let input_channels = if shard.head {
                    config.num_attention_heads * config.head_dim + hidden
                } else {
                    hidden
                };
                let output_channels = if shard.head {
                    if self.version >= 6 {
                        2 * hidden
                    } else {
                        3 * hidden
                    }
                } else {
                    hidden
                };
                if shard.order != order
                    || shard.input_name.is_empty()
                    || shard.output_name.is_empty()
                    || shard.input_shape != vec![1, input_channels, config.block_size, 1]
                    || shard.input_elements != config.block_size * input_channels
                    || shard.output_elements != config.block_size * output_channels
                    || shard.bytes == 0
                    || shard.bytes > MAX_SHARD_BYTES
                    || !is_lower_sha256(&shard.sha256)
                {
                    return Err(format!(
                        "invalid ANE fused tail shard contract at layer {layer} order {order}"
                    ));
                }
            }
        }
        if self
            .tail_shards
            .iter()
            .any(|shard| shard.layer >= config.num_hidden_layers)
        {
            return Err("ANE fused tail contains an out-of-range layer".into());
        }
        Ok(())
    }

    fn validate_fused_ffn(&self, config: &DFlashConfig) -> Result<(), String> {
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        for layer in 0..config.num_hidden_layers {
            let shards = self
                .ffn_shards
                .iter()
                .filter(|shard| shard.layer == layer)
                .collect::<Vec<_>>();
            if shards.is_empty()
                || !partitions_exact(
                    shards
                        .iter()
                        .map(|shard| (shard.intermediate_offset, shard.intermediate_width)),
                    intermediate,
                )
            {
                return Err(format!(
                    "ANE fused FFN layer {layer} is not an exact intermediate partition"
                ));
            }
            for (order, shard) in shards.into_iter().enumerate() {
                let expected_input_shape = if self.version >= 4 {
                    vec![1, hidden, config.block_size, 1]
                } else {
                    vec![config.block_size, hidden, 1, 1]
                };
                if shard.order != order
                    || shard.input_name.is_empty()
                    || shard.output_name.is_empty()
                    || shard.input_shape != expected_input_shape
                    || shard.input_elements != config.block_size * hidden
                    || shard.output_elements != config.block_size * hidden
                    || shard.bytes == 0
                    || shard.bytes > MAX_SHARD_BYTES
                    || !is_lower_sha256(&shard.sha256)
                {
                    return Err(format!(
                        "invalid ANE fused FFN shard contract at layer {layer} order {order}"
                    ));
                }
            }
        }
        if self
            .ffn_shards
            .iter()
            .any(|shard| shard.layer >= config.num_hidden_layers)
        {
            return Err("ANE fused FFN contains an out-of-range layer".into());
        }
        Ok(())
    }
}

/// CoreML v4 follows the empirically validated ANE layout `[1, C, T, 1]`.
/// Legacy manifests used `[T, C, 1, 1]`; retain their row-major indexing so
/// already-built development artifacts remain readable.
fn coreml_index(
    manifest_version: u32,
    token: usize,
    channel: usize,
    tokens: usize,
    channels: usize,
) -> usize {
    if manifest_version >= 4 {
        channel * tokens + token
    } else {
        token * channels + channel
    }
}

fn partitions_exact(parts: impl Iterator<Item = (usize, usize)>, extent: usize) -> bool {
    let mut parts = parts.collect::<Vec<_>>();
    parts.sort_unstable();
    let mut cursor = 0usize;
    for (offset, width) in parts {
        if offset != cursor || width == 0 {
            return false;
        }
        let Some(next) = cursor.checked_add(width) else {
            return false;
        };
        cursor = next;
    }
    cursor == extent
}

fn expected_projections(config: &DFlashConfig) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    names.insert("fc".into());
    for layer in 0..config.num_hidden_layers {
        for suffix in [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ] {
            names.insert(format!("layers.{layer}.{suffix}"));
        }
    }
    names
}

fn expected_linear_projections(config: &DFlashConfig) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    names.insert("fc".into());
    for layer in 0..config.num_hidden_layers {
        for suffix in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            names.insert(format!("layers.{layer}.{suffix}"));
        }
    }
    names
}

fn expected_attention_projections(config: &DFlashConfig) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    names.insert("fc".into());
    for layer in 0..config.num_hidden_layers {
        for suffix in ["q_proj", "k_proj", "v_proj"] {
            names.insert(format!("layers.{layer}.{suffix}"));
        }
    }
    names
}

fn projection_geometry(config: &DFlashConfig) -> BTreeMap<String, (usize, usize)> {
    let h = config.hidden_size;
    let q = config.num_attention_heads * config.head_dim;
    let kv = config.num_key_value_heads * config.head_dim;
    let inter = config.intermediate_size;
    let mut values = BTreeMap::new();
    values.insert(
        "fc".into(),
        (config.dflash_config.target_layer_ids.len() * h, h),
    );
    for layer in 0..config.num_hidden_layers {
        for (suffix, input, output) in [
            ("q_proj", h, q),
            ("k_proj", h, kv),
            ("v_proj", h, kv),
            ("o_proj", q, h),
            ("gate_proj", h, inter),
            ("up_proj", h, inter),
            ("down_proj", inter, h),
        ] {
            values.insert(format!("layers.{layer}.{suffix}"), (input, output));
        }
    }
    values
}

fn resolve_beneath(root: &Path, relative: &Path) -> Result<PathBuf, String> {
    if relative.is_absolute()
        || relative
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err(format!("unsafe ANE shard path {}", relative.display()));
    }
    let root = root
        .canonicalize()
        .map_err(|e| format!("canonicalize ANE root {}: {e}", root.display()))?;
    let candidate = root.join(relative);
    let canonical = candidate
        .canonicalize()
        .map_err(|e| format!("canonicalize ANE shard {}: {e}", candidate.display()))?;
    if !canonical.starts_with(&root) {
        return Err(format!(
            "ANE shard escapes artifact root: {}",
            relative.display()
        ));
    }
    Ok(canonical)
}

fn tree_entries(path: &Path) -> Result<Vec<(PathBuf, PathBuf)>, String> {
    if path.is_file() {
        return Ok(vec![(PathBuf::from("."), path.to_owned())]);
    }
    if !path.is_dir() {
        return Err(format!(
            "ANE shard is not a file or directory: {}",
            path.display()
        ));
    }
    fn visit(root: &Path, dir: &Path, out: &mut Vec<(PathBuf, PathBuf)>) -> Result<(), String> {
        let mut entries = std::fs::read_dir(dir)
            .map_err(|e| format!("read ANE shard directory {}: {e}", dir.display()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("read ANE shard directory {}: {e}", dir.display()))?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let file_type = entry
                .file_type()
                .map_err(|e| format!("stat ANE shard entry: {e}"))?;
            let child = entry.path();
            if file_type.is_symlink() {
                return Err(format!("ANE shard contains symlink: {}", child.display()));
            }
            if file_type.is_dir() {
                visit(root, &child, out)?;
            } else if file_type.is_file() {
                let relative = child
                    .strip_prefix(root)
                    .map_err(|_| "ANE shard traversal escaped root")?
                    .to_owned();
                out.push((relative, child));
            } else {
                return Err(format!(
                    "ANE shard contains special file: {}",
                    child.display()
                ));
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    visit(path, path, &mut out)?;
    Ok(out)
}

fn tree_size(path: &Path) -> Result<u64, String> {
    tree_entries(path)?
        .into_iter()
        .try_fold(0u64, |sum, (_, file)| {
            let len = file
                .metadata()
                .map_err(|e| format!("stat ANE shard file {}: {e}", file.display()))?
                .len();
            sum.checked_add(len)
                .ok_or_else(|| "ANE shard size overflow".into())
        })
}

fn tree_sha256(path: &Path) -> Result<String, String> {
    let mut digest = Sha256::new();
    for (relative, file) in tree_entries(path)? {
        let relative = relative
            .to_str()
            .ok_or_else(|| format!("ANE shard path is not UTF-8: {}", relative.display()))?;
        let relative = relative.as_bytes();
        digest.update((relative.len() as u64).to_le_bytes());
        digest.update(relative);
        let file_len = file
            .metadata()
            .map_err(|e| format!("stat ANE shard file {}: {e}", file.display()))?
            .len();
        digest.update(file_len.to_le_bytes());
        update_digest_from_file(&mut digest, &file)?;
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn update_digest_from_file(digest: &mut Sha256, path: &Path) -> Result<(), String> {
    let mut input =
        File::open(path).map_err(|e| format!("open artifact file {}: {e}", path.display()))?;
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = input
            .read(&mut buffer)
            .map_err(|e| format!("read artifact file {}: {e}", path.display()))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(())
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_gguf_identity_is_its_file_digest() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("dflash-kquant.gguf");
        std::fs::write(&path, b"official assistant bytes").unwrap();
        assert_eq!(
            dflash_artifact_identity(&path).unwrap(),
            file_sha256(&path).unwrap()
        );
    }

    fn manifest() -> AneDFlashManifest {
        AneDFlashManifest {
            version: 1,
            backend: "public_coreml".into(),
            compute_units: "CPU_AND_NE".into(),
            weight_dtype: "int8".into(),
            projection_operator: "conv1x1".into(),
            target_identity: "target".into(),
            dflash_identity: "draft".into(),
            dflash_source_format: "official-gguf".into(),
            extractor_sha256: Some("b".repeat(64)),
            assistant_layers: 5,
            block_size: 16,
            shards: Vec::new(),
            ffn_shards: Vec::new(),
            tail_shards: Vec::new(),
            attention_shards: Vec::new(),
            fused_attention_shards: Vec::new(),
        }
    }

    fn config() -> DFlashConfig {
        serde_json::from_str(r#"{"architectures":["DFlashDraftModel"],"block_size":16,"eos_token_id":2,"hidden_size":64,"head_dim":16,"intermediate_size":128,"num_attention_heads":4,"num_hidden_layers":5,"num_key_value_heads":2,"num_target_layers":52,"vocab_size":100,"max_position_embeddings":1024,"rms_norm_eps":1e-6,"rope_theta":10000.0,"dflash_config":{"mask_token_id":99,"target_layer_ids":[1,9,17,25,33]}}"#).unwrap()
    }

    fn fill_shards(value: &mut AneDFlashManifest) {
        let config = config();
        let bs = config.block_size;
        let geometry = projection_geometry(&config);
        for projection in expected_projections(&config) {
            let &(input_width, output_width) = &geometry[&projection];
            let order = value.shards.len();
            value.shards.push(AneShardManifest {
                order,
                path: format!("{order}.mlpackage").into(),
                input_name: "input".into(),
                output_name: "output".into(),
                projection,
                input_offset: 0,
                input_width,
                output_offset: 0,
                output_width,
                input_shape: vec![bs, input_width, 1, 1],
                input_elements: input_width * bs,
                output_elements: output_width * bs,
                bytes: 42,
                sha256: "a".repeat(64),
                components: Vec::new(),
            });
        }
    }

    #[test]
    fn release_contract_accepts_ordered_public_int8_conv_graph() {
        let mut value = manifest();
        fill_shards(&mut value);
        value.validate("target", "draft", &config()).unwrap();
    }

    #[test]
    fn release_contract_accepts_fused_qkv_components() {
        let mut value = manifest();
        fill_shards(&mut value);
        value.version = 2;
        for shard in &mut value.shards {
            shard.components.push(AneShardComponentManifest {
                projection: shard.projection.clone(),
                output_offset: 0,
                projection_offset: shard.output_offset,
                output_width: shard.output_width,
            });
            shard.output_offset = 0;
        }
        let names = ["layers.0.q_proj", "layers.0.k_proj", "layers.0.v_proj"];
        let mut fused = value
            .shards
            .iter()
            .find(|shard| shard.projection == names[0])
            .unwrap()
            .clone();
        fused.projection = "layers.0.qkv_fused".into();
        fused.components.clear();
        fused.output_width = 0;
        for name in names {
            let shard = value
                .shards
                .iter()
                .find(|shard| shard.projection == name)
                .unwrap();
            fused.components.push(AneShardComponentManifest {
                projection: name.into(),
                output_offset: fused.output_width,
                projection_offset: 0,
                output_width: shard.output_width,
            });
            fused.output_width += shard.output_width;
        }
        fused.output_elements = fused.output_width * value.block_size;
        value
            .shards
            .retain(|shard| !names.contains(&shard.projection.as_str()));
        value.shards.push(fused);
        for (order, shard) in value.shards.iter_mut().enumerate() {
            shard.order = order;
        }
        value.validate("target", "draft", &config()).unwrap();
    }

    #[test]
    fn release_contract_accepts_fused_ffn_partitions() {
        let config = config();
        let mut value = manifest();
        fill_shards(&mut value);
        value.version = 3;
        value.shards.retain(|shard| {
            !["gate_proj", "up_proj", "down_proj"]
                .iter()
                .any(|suffix| shard.projection.ends_with(suffix))
        });
        for (order, shard) in value.shards.iter_mut().enumerate() {
            shard.order = order;
            shard.components.push(AneShardComponentManifest {
                projection: shard.projection.clone(),
                output_offset: 0,
                projection_offset: shard.output_offset,
                output_width: shard.output_width,
            });
            shard.output_offset = 0;
        }
        for layer in 0..config.num_hidden_layers {
            for (order, (offset, width)) in [(0, 64), (64, 64)].into_iter().enumerate() {
                value.ffn_shards.push(AneFfnShardManifest {
                    order,
                    layer,
                    path: format!("ffn-{layer}-{order}.mlpackage").into(),
                    input_name: "input".into(),
                    output_name: "output".into(),
                    intermediate_offset: offset,
                    intermediate_width: width,
                    input_shape: vec![config.block_size, config.hidden_size, 1, 1],
                    input_elements: config.block_size * config.hidden_size,
                    output_elements: config.block_size * config.hidden_size,
                    bytes: 42,
                    sha256: "c".repeat(64),
                });
            }
        }
        value.validate("target", "draft", &config).unwrap();
    }

    #[test]
    fn release_contract_accepts_ane_native_nctw_layout() {
        let config = config();
        let mut value = manifest();
        fill_shards(&mut value);
        value.version = 4;
        value.shards.retain(|shard| {
            !["gate_proj", "up_proj", "down_proj"]
                .iter()
                .any(|suffix| shard.projection.ends_with(suffix))
        });
        for (order, shard) in value.shards.iter_mut().enumerate() {
            shard.order = order;
            shard.input_shape = vec![1, shard.input_width, config.block_size, 1];
            shard.components.push(AneShardComponentManifest {
                projection: shard.projection.clone(),
                output_offset: 0,
                projection_offset: shard.output_offset,
                output_width: shard.output_width,
            });
            shard.output_offset = 0;
        }
        for layer in 0..config.num_hidden_layers {
            for (order, (offset, width)) in [(0, 64), (64, 64)].into_iter().enumerate() {
                value.ffn_shards.push(AneFfnShardManifest {
                    order,
                    layer,
                    path: format!("ffn-v4-{layer}-{order}.mlpackage").into(),
                    input_name: "input".into(),
                    output_name: "output".into(),
                    intermediate_offset: offset,
                    intermediate_width: width,
                    input_shape: vec![1, config.hidden_size, config.block_size, 1],
                    input_elements: config.block_size * config.hidden_size,
                    output_elements: config.block_size * config.hidden_size,
                    bytes: 42,
                    sha256: "d".repeat(64),
                });
            }
        }
        value.validate("target", "draft", &config).unwrap();
        assert_eq!(coreml_index(4, 3, 2, 16, 64), 35);
        assert_eq!(coreml_index(3, 3, 2, 16, 64), 194);
    }

    #[test]
    fn release_contract_accepts_fused_layer_tails() {
        let config = config();
        let mut value = manifest();
        fill_shards(&mut value);
        value.version = 5;
        value.shards.retain(|shard| {
            shard.projection == "fc"
                || ["q_proj", "k_proj", "v_proj"]
                    .iter()
                    .any(|suffix| shard.projection.ends_with(suffix))
        });
        for (order, shard) in value.shards.iter_mut().enumerate() {
            shard.order = order;
            shard.input_shape = vec![1, shard.input_width, config.block_size, 1];
            shard.components.push(AneShardComponentManifest {
                projection: shard.projection.clone(),
                output_offset: 0,
                projection_offset: shard.output_offset,
                output_width: shard.output_width,
            });
            shard.output_offset = 0;
        }
        let attention_width = config.num_attention_heads * config.head_dim;
        for layer in 0..config.num_hidden_layers {
            for (order, (offset, width)) in [(0, 64), (64, 64)].into_iter().enumerate() {
                let head = order == 0;
                let input_channels = if head {
                    attention_width + config.hidden_size
                } else {
                    config.hidden_size
                };
                let output_channels = if head {
                    3 * config.hidden_size
                } else {
                    config.hidden_size
                };
                value.tail_shards.push(AneTailShardManifest {
                    order,
                    layer,
                    head,
                    path: format!("tail-v5-{layer}-{order}.mlpackage").into(),
                    input_name: "input".into(),
                    output_name: "output".into(),
                    intermediate_offset: offset,
                    intermediate_width: width,
                    input_shape: vec![1, input_channels, config.block_size, 1],
                    input_elements: config.block_size * input_channels,
                    output_elements: config.block_size * output_channels,
                    bytes: 42,
                    sha256: "e".repeat(64),
                });
            }
        }
        value.validate("target", "draft", &config).unwrap();
        value.version = 6;
        for shard in &mut value.tail_shards {
            if shard.head {
                shard.output_elements = 2 * config.block_size * config.hidden_size;
            }
        }
        value.validate("target", "draft", &config).unwrap();
        value.version = 7;
        assert!(value.validate("target", "draft", &config).is_err());
        value.attention_shards = (0..config.num_hidden_layers)
            .map(|layer| AneAttentionShardManifest {
                order: layer,
                layer,
                path: "stateful-attention.mlpackage".into(),
                max_context: 1088,
                query_size: 16,
                chunk: 16,
                kv_write_chunk: 16,
                kv_join: "split".into(),
                state_group_kv_heads: 8,
                bytes: 42,
                sha256: "f".repeat(64),
            })
            .collect();
        value.validate("target", "draft", &config).unwrap();
        value.attention_shards[0].state_group_kv_heads = 4;
        assert!(value.validate("target", "draft", &config).is_err());
        value.attention_shards[0].state_group_kv_heads = 8;
        value.attention_shards[0].kv_write_chunk = 4;
        assert!(value.validate("target", "draft", &config).is_err());
    }

    #[test]
    fn release_contract_rejects_identity_and_route_changes() {
        let mut value = manifest();
        fill_shards(&mut value);
        value.compute_units = "ALL".into();
        assert!(value.validate("target", "draft", &config()).is_err());
        let mut value = manifest();
        fill_shards(&mut value);
        assert!(value.validate("different", "draft", &config()).is_err());
    }

    #[test]
    fn release_contract_rejects_graph_gaps_and_oversize_shards() {
        let mut value = manifest();
        fill_shards(&mut value);
        value.shards.pop();
        assert!(value.validate("target", "draft", &config()).is_err());
        let mut value = manifest();
        fill_shards(&mut value);
        value.shards[0].bytes = MAX_SHARD_BYTES + 1;
        assert!(value.validate("target", "draft", &config()).is_err());
    }

    #[test]
    fn release_contract_accepts_exact_accumulating_input_slices() {
        let mut value = manifest();
        fill_shards(&mut value);
        let fc = value
            .shards
            .iter()
            .position(|shard| shard.projection == "fc")
            .unwrap();
        let mut second = value.shards[fc].clone();
        value.shards[fc].input_width = 160;
        value.shards[fc].input_shape[1] = 160;
        value.shards[fc].input_elements = 160 * value.block_size;
        second.input_offset = 160;
        second.input_width = 160;
        second.input_shape[1] = 160;
        second.input_elements = 160 * value.block_size;
        value.shards.insert(fc + 1, second);
        for (order, shard) in value.shards.iter_mut().enumerate() {
            shard.order = order;
        }
        value.validate("target", "draft", &config()).unwrap();
        value.shards[fc + 1].input_offset = 159;
        assert!(value.validate("target", "draft", &config()).is_err());
    }

    #[test]
    fn v8_capture_fc_requires_one_ordered_slice_per_target_layer() {
        let config = config();
        let mut value = manifest();
        fill_shards(&mut value);
        let fc = value
            .shards
            .iter()
            .position(|shard| shard.projection == "fc")
            .unwrap();
        let template = value.shards.remove(fc);
        let mut parts = Vec::new();
        for index in 0..config.dflash_config.target_layer_ids.len() {
            let mut shard = template.clone();
            shard.input_offset = index * config.hidden_size;
            shard.input_width = config.hidden_size;
            shard.output_width = config.hidden_size;
            shard.components = vec![AneShardComponentManifest {
                projection: "fc".into(),
                output_offset: 0,
                projection_offset: 0,
                output_width: config.hidden_size,
            }];
            parts.push(shard);
        }
        value.shards.splice(fc..fc, parts);
        value.validate_capture_fc(&config).unwrap();
        value.shards[fc + 1].input_offset -= 1;
        assert!(value.validate_capture_fc(&config).is_err());
    }

    #[test]
    fn v9_fused_attention_requires_complete_exact_layer_geometry() {
        let config = config();
        let mut value = manifest();
        value.tail_shards.push(AneTailShardManifest {
            order: 0,
            layer: 0,
            head: true,
            path: "tail.mlpackage".into(),
            input_name: "input".into(),
            output_name: "output".into(),
            intermediate_offset: 0,
            intermediate_width: 1,
            input_shape: vec![1, 1, 16, 1],
            input_elements: 16,
            output_elements: 16,
            bytes: 1,
            sha256: "a".repeat(64),
        });
        let bundle_channels =
            (config.num_attention_heads + 2 * config.num_key_value_heads) * config.head_dim;
        value.fused_attention_shards = (0..config.num_hidden_layers)
            .map(|layer| AneFusedAttentionShardManifest {
                order: layer,
                layer,
                path: format!("fused-{layer}.mlpackage").into(),
                max_context: 1088,
                block_size: config.block_size,
                query_size: config.block_size,
                kv_write_chunk: config.block_size,
                kv_join: "split".into(),
                state_group_kv_heads: 4,
                bundle_channels,
                bytes: 42,
                sha256: "b".repeat(64),
            })
            .collect();
        value.validate_fused_attention(&config).unwrap();
        value.fused_attention_shards[2].bundle_channels -= 1;
        assert!(value.validate_fused_attention(&config).is_err());
    }

    #[test]
    fn mirror_overlap_is_confined_to_the_deadlock_free_v8_capture_route() {
        assert!(exact_mirror_overlap_supported(8, 5, 5));
        assert!(!exact_mirror_overlap_supported(7, 5, 5));
        assert!(!exact_mirror_overlap_supported(8, 4, 5));
        assert!(!exact_mirror_overlap_supported(9, 5, 5));
    }
}
