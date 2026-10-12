use std::sync::Arc;

use super::cache::{DFlashContextKvCache, DFlashKvCache};
use super::ops::rms_norm;
use super::projection::project_to;
use super::{DFlashConfig, DFlashError, DFlashProjectionBackend, DFlashWeights};

pub struct DFlashDraftOutput {
    pub hidden_states: Vec<f32>,
    pub n_draft_tokens: usize,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) metal_hidden_states: Option<crate::metal::buffer::GpuBuffer>,
}

pub struct DFlashForward {
    pub(crate) config: DFlashConfig,
    pub(crate) weights: DFlashWeights,
    pub(crate) hidden: Vec<f32>,
    pub(crate) residual: Vec<f32>,
    pub(crate) normed: Vec<f32>,
    pub(crate) target_proj: Vec<f32>,
    pub(crate) q: Vec<f32>,
    pub(crate) k: Vec<f32>,
    pub(crate) v: Vec<f32>,
    pub(crate) k_target: Vec<f32>,
    pub(crate) v_target: Vec<f32>,
    pub(crate) k_cat: Vec<f32>,
    pub(crate) v_cat: Vec<f32>,
    pub(crate) attn: Vec<f32>,
    pub(crate) projected: Vec<f32>,
    pub(crate) gate: Vec<f32>,
    pub(crate) up: Vec<f32>,
    pub(crate) ffn: Vec<f32>,
    pub(crate) n_ctx: usize,
    pub(crate) projection_backend: Option<Arc<dyn DFlashProjectionBackend>>,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) metal_forward: Option<crate::metal::dflash::MetalDFlashForward>,
}

impl DFlashForward {
    pub fn new(weights: DFlashWeights) -> Self {
        let c = &weights.config;
        let (bs, h) = (c.block_size, c.hidden_size);
        let q = c.num_attention_heads * c.head_dim;
        let kv = c.num_key_value_heads * c.head_dim;
        let inter = c.intermediate_size;
        Self {
            config: c.clone(),
            weights,
            hidden: vec![0.; bs * h],
            residual: vec![0.; bs * h],
            normed: vec![0.; bs * h],
            target_proj: vec![0.; bs * h],
            q: vec![0.; bs * q],
            k: vec![0.; bs * kv],
            v: vec![0.; bs * kv],
            k_target: vec![0.; bs * kv],
            v_target: vec![0.; bs * kv],
            k_cat: vec![0.; 2 * bs * kv],
            v_cat: vec![0.; 2 * bs * kv],
            attn: vec![0.; bs * q],
            projected: vec![0.; bs * h],
            gate: vec![0.; bs * inter],
            up: vec![0.; bs * inter],
            ffn: vec![0.; bs * h],
            n_ctx: bs,
            projection_backend: None,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            metal_forward: None,
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn with_metal_forward(mut self, forward: crate::metal::dflash::MetalDFlashForward) -> Self {
        self.metal_forward = Some(forward);
        self
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn prepare_target_context_split(
        &mut self,
        target_hidden: &[f32],
        n_ctx: usize,
        cache: &DFlashContextKvCache,
    ) -> Result<crate::metal::dflash::PreparedMetalDFlashContext, DFlashError> {
        self.metal_forward
            .as_mut()
            .ok_or_else(|| {
                DFlashError::Config(
                    "prepared DFlash context requires the Metal forward backend".into(),
                )
            })?
            .prepare_target_context(target_hidden, n_ctx, cache)
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn finish_target_context_split(
        &mut self,
        prepared: crate::metal::dflash::PreparedMetalDFlashContext,
        noise_embedding: &[f32],
        selected_context_rows: usize,
        cache: &mut DFlashContextKvCache,
    ) -> Result<DFlashDraftOutput, DFlashError> {
        self.metal_forward
            .as_mut()
            .ok_or_else(|| {
                DFlashError::Config(
                    "prepared DFlash context requires the Metal forward backend".into(),
                )
            })?
            .finish_prepared(prepared, noise_embedding, selected_context_rows, cache)
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn discard_target_context_split(
        &mut self,
        prepared: crate::metal::dflash::PreparedMetalDFlashContext,
    ) -> Result<(), DFlashError> {
        self.metal_forward
            .as_mut()
            .ok_or_else(|| {
                DFlashError::Config(
                    "prepared DFlash context requires the Metal forward backend".into(),
                )
            })?
            .discard_prepared(prepared)
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn invalidate_target_context_split(&mut self) {
        if let Some(forward) = self.metal_forward.as_mut() {
            forward.invalidate_prepared();
        }
    }

    pub fn with_projection_backend(mut self, backend: Arc<dyn DFlashProjectionBackend>) -> Self {
        self.projection_backend = Some(backend);
        self
    }

    pub fn config(&self) -> &DFlashConfig {
        &self.config
    }

    pub fn forward(
        &mut self,
        noise_embedding: &[f32],
        target_hidden: &[f32],
        n_ctx: usize,
        cache: &mut DFlashContextKvCache,
        draft_cache: &mut DFlashKvCache,
    ) -> Result<DFlashDraftOutput, DFlashError> {
        self.forward_with_target_projection(
            noise_embedding,
            target_hidden,
            n_ctx,
            cache,
            draft_cache,
            None,
        )
    }

    pub(crate) fn forward_with_target_projection(
        &mut self,
        noise_embedding: &[f32],
        target_hidden: &[f32],
        n_ctx: usize,
        cache: &mut DFlashContextKvCache,
        _draft_cache: &mut DFlashKvCache,
        target_projection: Option<&[f32]>,
    ) -> Result<DFlashDraftOutput, DFlashError> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Some(forward) = self.metal_forward.as_mut() {
            if target_projection.is_some() {
                return Err(DFlashError::Config(
                    "capture-FC projection is only valid for the public-CoreML backend".into(),
                ));
            }
            return forward.forward(noise_embedding, target_hidden, n_ctx, cache);
        }
        let (bs, h) = (self.config.block_size, self.config.hidden_size);
        let sampled = self.config.dflash_config.target_layer_ids.len();
        if noise_embedding.len() != bs * h {
            return Err(DFlashError::Config(format!(
                "noise embedding expected {}, got {}",
                bs * h,
                noise_embedding.len()
            )));
        }
        if n_ctx == 0 || target_hidden.len() != n_ctx * sampled * h {
            return Err(DFlashError::Config(
                "invalid target hidden-state batch".into(),
            ));
        }
        self.n_ctx = n_ctx;
        self.target_proj.resize(n_ctx * h, 0.);
        self.k_target.resize(n_ctx * cache.layout().width(), 0.);
        self.v_target.resize(n_ctx * cache.layout().width(), 0.);
        if let Some(projected) = target_projection {
            if projected.len() != n_ctx * h {
                return Err(DFlashError::Config(format!(
                    "capture-FC projection expected {} elements, got {}",
                    n_ctx * h,
                    projected.len()
                )));
            }
            self.target_proj.copy_from_slice(projected);
        } else {
            project_to(
                self.projection_backend.as_deref(),
                "fc",
                target_hidden,
                &self.weights.fc_weight,
                &mut self.target_proj,
                n_ctx,
                h,
                sampled * h,
            )?;
        }
        rms_norm(
            &mut self.target_proj,
            &self.weights.hidden_norm_weight,
            n_ctx,
            h,
            self.config.rms_norm_eps,
        );
        self.hidden.copy_from_slice(noise_embedding);
        for layer in 0..self.config.num_hidden_layers {
            self.forward_layer(layer, cache)?;
        }
        cache.advance_round(n_ctx);
        rms_norm(
            &mut self.hidden,
            &self.weights.norm_weight,
            bs,
            h,
            self.config.rms_norm_eps,
        );
        Ok(DFlashDraftOutput {
            hidden_states: self.hidden.clone(),
            n_draft_tokens: bs - 1,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            metal_hidden_states: None,
        })
    }

    fn forward_layer(
        &mut self,
        layer: usize,
        cache: &mut DFlashContextKvCache,
    ) -> Result<(), DFlashError> {
        let (bs, h, inter) = (
            self.config.block_size,
            self.config.hidden_size,
            self.config.intermediate_size,
        );
        self.residual.copy_from_slice(&self.hidden);
        self.normed.copy_from_slice(&self.hidden);
        rms_norm(
            &mut self.normed,
            &self.weights.layers[layer].input_layernorm_weight,
            bs,
            h,
            self.config.rms_norm_eps,
        );
        self.forward_attention(layer, cache)?;
        if let Some(backend) = self.projection_backend.as_deref() {
            if backend.has_fused_layer_tail(layer) {
                let wrote_output = backend
                    .fused_layer_tail_into(layer, &self.attn, &self.residual, &mut self.hidden)
                    .map_err(DFlashError::Projection)?;
                if !wrote_output {
                    return Err(DFlashError::Projection(format!(
                        "fused layer {layer} tail was declared but returned no output"
                    )));
                }
                return Ok(());
            }
        }
        let w = &self.weights.layers[layer];
        for i in 0..bs * h {
            self.hidden[i] = self.residual[i] + self.projected[i];
        }
        self.residual.copy_from_slice(&self.hidden);
        self.normed.copy_from_slice(&self.hidden);
        rms_norm(
            &mut self.normed,
            &w.post_attention_layernorm_weight,
            bs,
            h,
            self.config.rms_norm_eps,
        );
        if let Some(backend) = self.projection_backend.as_deref() {
            if let Some(output) = backend
                .fused_ffn(layer, &self.normed)
                .map_err(DFlashError::Projection)?
            {
                if output.len() != bs * h {
                    return Err(DFlashError::Projection(format!(
                        "fused layer {layer} FFN returned {} elements, expected {}",
                        output.len(),
                        bs * h
                    )));
                }
                self.ffn.copy_from_slice(&output);
                for i in 0..bs * h {
                    self.hidden[i] = self.residual[i] + self.ffn[i];
                }
                return Ok(());
            }
        }
        let gate_name = format!("layers.{layer}.gate_proj");
        let up_name = format!("layers.{layer}.up_proj");
        if let Some(backend) = self.projection_backend.as_deref() {
            let values = backend
                .project_group(&[&gate_name, &up_name], &self.normed)
                .map_err(DFlashError::Projection)?;
            if values.len() != 2 || values[0].len() != bs * inter || values[1].len() != bs * inter {
                return Err(DFlashError::Projection(format!(
                    "gate/up group returned invalid geometry {:?}",
                    values.iter().map(Vec::len).collect::<Vec<_>>()
                )));
            }
            self.gate.copy_from_slice(&values[0]);
            self.up.copy_from_slice(&values[1]);
        } else {
            project_to(
                None,
                &gate_name,
                &self.normed,
                &w.gate_proj_weight,
                &mut self.gate,
                bs,
                inter,
                h,
            )?;
            project_to(
                None,
                &up_name,
                &self.normed,
                &w.up_proj_weight,
                &mut self.up,
                bs,
                inter,
                h,
            )?;
        }
        for i in 0..bs * inter {
            let g = self.gate[i];
            self.gate[i] = g / (1. + (-g).exp()) * self.up[i];
        }
        project_to(
            self.projection_backend.as_deref(),
            &format!("layers.{layer}.down_proj"),
            &self.gate,
            &w.down_proj_weight,
            &mut self.ffn,
            bs,
            h,
            inter,
        )?;
        for i in 0..bs * h {
            self.hidden[i] = self.residual[i] + self.ffn[i];
        }
        Ok(())
    }
}
