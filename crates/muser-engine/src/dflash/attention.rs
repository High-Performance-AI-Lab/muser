use super::cache::DFlashContextKvCache;
use super::ops::{attention, head_norm, rope};
use super::projection::{project_to, DFlashFusedAttentionInput, DFlashStatefulAttentionInput};
use super::{DFlashError, DFlashForward};

impl DFlashForward {
    pub(crate) fn forward_attention(
        &mut self,
        layer: usize,
        cache: &mut DFlashContextKvCache,
    ) -> Result<(), DFlashError> {
        let c = &self.config;
        let (bs, h, hd, heads, kv_heads) = (
            c.block_size,
            c.hidden_size,
            c.head_dim,
            c.num_attention_heads,
            c.num_key_value_heads,
        );
        let q_dim = heads * hd;
        let kv_dim = kv_heads * hd;
        let (ctx_len, ctx_offset, n_ctx) = (cache.ctx_len, cache.ctx_offset, self.n_ctx);
        if let Some(backend) = self.projection_backend.as_deref() {
            if let Some(output) = backend
                .fused_stateful_attention_layer(
                    layer,
                    DFlashFusedAttentionInput {
                        noise_normed: &self.normed,
                        target_projected: &self.target_proj,
                        cached_key: cache.layer_k(layer),
                        cached_value: cache.layer_v(layer),
                        block_size: bs,
                        target_rows: n_ctx,
                        hidden_size: h,
                        attention_heads: heads,
                        key_value_heads: kv_heads,
                        head_dim: hd,
                        context_position: ctx_offset,
                        context_rows: ctx_len,
                        sink_size: cache.sink_size,
                        window_size: cache.window_size,
                        rope_theta: c.rope_theta,
                        cache_identity: cache.identity(),
                        cache_revision: cache.revision(),
                    },
                )
                .map_err(DFlashError::Projection)?
            {
                let attention = bs * q_dim;
                let target = n_ctx * kv_dim;
                if output.attention.len() != attention
                    || output.target_key.len() != target
                    || output.target_value.len() != target
                {
                    return Err(DFlashError::Projection(format!(
                        "fused stateful attention layer {layer} returned geometry {}/{}/{}, expected {attention}/{target}/{target}",
                        output.attention.len(),
                        output.target_key.len(),
                        output.target_value.len(),
                    )));
                }
                self.attn[..attention].copy_from_slice(&output.attention);
                self.k_target[..target].copy_from_slice(&output.target_key);
                self.v_target[..target].copy_from_slice(&output.target_value);
                cache.append_layer(layer, &output.target_key, &output.target_value, n_ctx);
                if !backend.has_fused_layer_tail(layer) {
                    return Err(DFlashError::Projection(format!(
                        "fused stateful attention layer {layer} requires a fused tail"
                    )));
                }
                return Ok(());
            }
        }
        let full = ctx_len + n_ctx + bs;
        let required = full * kv_dim;
        self.k_cat.resize(required, 0.);
        self.v_cat.resize(required, 0.);
        let w = &self.weights.layers[layer];
        if let Some(backend) = self.projection_backend.as_deref() {
            let values = backend
                .attention_projections(layer, &self.normed, &self.target_proj)
                .map_err(DFlashError::Projection)?;
            copy_group(
                &[values.q, values.k, values.v],
                [&mut self.q, &mut self.k, &mut self.v],
                [bs * q_dim, bs * kv_dim, bs * kv_dim],
                "noise Q/K/V",
            )?;
            copy_group(
                &[values.k_target, values.v_target],
                [&mut self.k_target, &mut self.v_target],
                [n_ctx * kv_dim, n_ctx * kv_dim],
                "target K/V",
            )?;
        } else {
            let q_name = format!("layers.{layer}.q_proj");
            let k_name = format!("layers.{layer}.k_proj");
            let v_name = format!("layers.{layer}.v_proj");
            project_to(
                None,
                &q_name,
                &self.normed,
                &w.q_proj_weight,
                &mut self.q,
                bs,
                q_dim,
                h,
            )?;
            project_to(
                None,
                &k_name,
                &self.normed,
                &w.k_proj_weight,
                &mut self.k,
                bs,
                kv_dim,
                h,
            )?;
            project_to(
                None,
                &v_name,
                &self.normed,
                &w.v_proj_weight,
                &mut self.v,
                bs,
                kv_dim,
                h,
            )?;
            project_to(
                None,
                &k_name,
                &self.target_proj,
                &w.k_proj_weight,
                &mut self.k_target,
                n_ctx,
                kv_dim,
                h,
            )?;
            project_to(
                None,
                &v_name,
                &self.target_proj,
                &w.v_proj_weight,
                &mut self.v_target,
                n_ctx,
                kv_dim,
                h,
            )?;
        }
        let cached = 0..ctx_len * kv_dim;
        let fresh = cached.end..cached.end + n_ctx * kv_dim;
        let noise = fresh.end..fresh.end + bs * kv_dim;
        if ctx_len > 0 {
            self.k_cat[cached.clone()].copy_from_slice(cache.k_rows(layer, 0, ctx_len));
            self.v_cat[cached].copy_from_slice(cache.v_rows(layer, 0, ctx_len));
        }
        self.k_cat[fresh.clone()].copy_from_slice(&self.k_target[..n_ctx * kv_dim]);
        self.v_cat[fresh.clone()].copy_from_slice(&self.v_target[..n_ctx * kv_dim]);
        self.k_cat[noise.clone()].copy_from_slice(&self.k[..bs * kv_dim]);
        self.v_cat[noise.clone()].copy_from_slice(&self.v[..bs * kv_dim]);
        head_norm(&mut self.q, &w.q_norm_weight, bs, heads, hd, c.rms_norm_eps);
        head_norm(
            &mut self.k_cat[fresh.start..noise.end],
            &w.k_norm_weight,
            n_ctx + bs,
            kv_heads,
            hd,
            c.rms_norm_eps,
        );
        rope(
            &mut self.k_cat[fresh.clone()],
            n_ctx,
            kv_heads,
            hd,
            ctx_offset,
            c.rope_theta,
        );
        rope(
            &mut self.k_cat[noise.clone()],
            bs,
            kv_heads,
            hd,
            ctx_offset + n_ctx,
            c.rope_theta,
        );
        rope(&mut self.q, bs, heads, hd, ctx_offset + n_ctx, c.rope_theta);
        let accelerated_attention = if let Some(backend) = self.projection_backend.as_deref() {
            backend
                .stateful_attention_into(
                    layer,
                    DFlashStatefulAttentionInput {
                        query: &self.q,
                        noise_key: &self.k_cat[noise.clone()],
                        noise_value: &self.v_cat[noise.clone()],
                        target_key: &self.k_cat[fresh.clone()],
                        target_value: &self.v_cat[fresh.clone()],
                        cached_key: cache.layer_k(layer),
                        cached_value: cache.layer_v(layer),
                        block_size: bs,
                        target_rows: n_ctx,
                        attention_heads: heads,
                        key_value_heads: kv_heads,
                        head_dim: hd,
                        context_position: ctx_offset,
                        context_rows: ctx_len,
                        sink_size: cache.sink_size,
                        window_size: cache.window_size,
                        cache_identity: cache.identity(),
                        cache_revision: cache.revision(),
                    },
                    &mut self.attn,
                )
                .map_err(DFlashError::Projection)?
        } else {
            false
        };
        if !accelerated_attention {
            attention(
                &self.q,
                &self.k_cat[..required],
                &self.v_cat[..required],
                &mut self.attn,
                bs,
                full,
                heads,
                kv_heads,
                hd,
            );
        }
        // Keep the exact CPU shadow even when Core ML owns live attention
        // state. It is the source for snapshots, resynchronization, and the
        // transactional ANE -> Metal fallback.
        cache.append_layer(layer, &self.k_cat[fresh.clone()], &self.v_cat[fresh], n_ctx);
        if !self
            .projection_backend
            .as_deref()
            .is_some_and(|backend| backend.has_fused_layer_tail(layer))
        {
            project_to(
                self.projection_backend.as_deref(),
                &format!("layers.{layer}.o_proj"),
                &self.attn,
                &w.o_proj_weight,
                &mut self.projected,
                bs,
                h,
                q_dim,
            )?;
        }
        Ok(())
    }
}

fn copy_group<const N: usize>(
    values: &[Vec<f32>],
    outputs: [&mut Vec<f32>; N],
    expected: [usize; N],
    label: &str,
) -> Result<(), DFlashError> {
    if values.len() != N {
        return Err(DFlashError::Projection(format!(
            "{label} returned {} projections, expected {N}",
            values.len()
        )));
    }
    for ((value, output), expected) in values.iter().zip(outputs).zip(expected) {
        if value.len() != expected {
            return Err(DFlashError::Projection(format!(
                "{label} returned {} elements, expected {expected}",
                value.len()
            )));
        }
        output[..expected].copy_from_slice(value);
    }
    Ok(())
}
