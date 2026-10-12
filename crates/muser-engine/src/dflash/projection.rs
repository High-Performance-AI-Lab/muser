use super::ops::matmul;
use super::DFlashError;

pub struct DFlashAttentionProjections {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub k_target: Vec<f32>,
    pub v_target: Vec<f32>,
}

/// Inputs for one backend-owned QKV + stateful-attention call. The backend
/// returns exact post-RoPE target K/V as well as the pre-output-projection
/// attention rows so the CPU shadow remains authoritative and the existing
/// fused tail can finish the layer without repeating QKV work.
pub struct DFlashFusedAttentionInput<'a> {
    pub noise_normed: &'a [f32],
    pub target_projected: &'a [f32],
    pub cached_key: &'a [f32],
    pub cached_value: &'a [f32],
    pub block_size: usize,
    pub target_rows: usize,
    pub hidden_size: usize,
    pub attention_heads: usize,
    pub key_value_heads: usize,
    pub head_dim: usize,
    pub context_position: usize,
    pub context_rows: usize,
    pub sink_size: usize,
    pub window_size: usize,
    pub rope_theta: f64,
    pub cache_identity: u64,
    pub cache_revision: u64,
}

pub struct DFlashFusedAttentionOutput {
    pub attention: Vec<f32>,
    pub target_key: Vec<f32>,
    pub target_value: Vec<f32>,
}

/// Exact, already-normalized attention rows presented to a backend which owns
/// persistent DFlash K/V state. Tensor rows are token-major; a Core ML backend
/// is responsible for converting them to its declared NCHW/head-major ABI.
pub struct DFlashStatefulAttentionInput<'a> {
    pub query: &'a [f32],
    pub noise_key: &'a [f32],
    pub noise_value: &'a [f32],
    pub target_key: &'a [f32],
    pub target_value: &'a [f32],
    pub cached_key: &'a [f32],
    pub cached_value: &'a [f32],
    pub block_size: usize,
    pub target_rows: usize,
    pub attention_heads: usize,
    pub key_value_heads: usize,
    pub head_dim: usize,
    pub context_position: usize,
    pub context_rows: usize,
    pub sink_size: usize,
    pub window_size: usize,
    pub cache_identity: u64,
    pub cache_revision: u64,
}

/// Replace only DFlash's dense projections while retaining the exact CPU
/// norms, RoPE, dual-context attention, cache, and verifier semantics.
pub trait DFlashProjectionBackend: Send + Sync {
    /// One-call QKV projection, normalization/RoPE, state update, and GQA.
    /// `None` retains the projection + attention path used by v1-v8.
    fn fused_stateful_attention_layer(
        &self,
        _layer: usize,
        _input: DFlashFusedAttentionInput<'_>,
    ) -> Result<Option<DFlashFusedAttentionOutput>, String> {
        Ok(None)
    }

    /// True only for the public-CoreML backend whose state can be restored by
    /// replaying the authoritative CPU shadow after a Mirror-SD miss.
    fn supports_exact_mirror_overlap(&self) -> bool {
        false
    }

    /// Whether `fc` is represented as one ordered input slice per exact
    /// target capture layer. This is the artifact half of the default-off v8
    /// pipeline; callers still require staged target execution.
    fn supports_capture_fc_pipeline(&self) -> bool {
        false
    }

    /// Project one `[tokens, hidden]` target-layer capture into its additive
    /// contribution to DFlash `fc`. Returning `None` keeps the v7 whole-FC
    /// route.
    fn project_capture_fc_slice(
        &self,
        _capture: usize,
        _input: &[f32],
    ) -> Result<Option<Vec<f32>>, String> {
        Ok(None)
    }

    fn project(&self, name: &str, input: &[f32]) -> Result<Vec<f32>, String>;

    /// Execute projections which consume the same rows. Backends may fuse the
    /// physical graph; the default preserves the exact independent route.
    fn project_group(&self, names: &[&str], input: &[f32]) -> Result<Vec<Vec<f32>>, String> {
        names.iter().map(|name| self.project(name, input)).collect()
    }

    /// Project the noise Q/K/V rows and target K/V rows for one assistant
    /// layer. Public Core ML overrides this to concatenate both row sets into
    /// one wider-NCTW prediction; other backends preserve the independent
    /// projection route.
    fn attention_projections(
        &self,
        layer: usize,
        noise: &[f32],
        target: &[f32],
    ) -> Result<DFlashAttentionProjections, String> {
        let q_name = format!("layers.{layer}.q_proj");
        let k_name = format!("layers.{layer}.k_proj");
        let v_name = format!("layers.{layer}.v_proj");
        let mut noise_values = self.project_group(&[&q_name, &k_name, &v_name], noise)?;
        let mut target_values = self.project_group(&[&k_name, &v_name], target)?;
        if noise_values.len() != 3 || target_values.len() != 2 {
            return Err("assistant attention projection group has invalid arity".into());
        }
        Ok(DFlashAttentionProjections {
            q: noise_values.remove(0),
            k: noise_values.remove(0),
            v: noise_values.remove(0),
            k_target: target_values.remove(0),
            v_target: target_values.remove(0),
        })
    }

    /// Optionally execute attention against backend-owned persistent K/V
    /// state. `None` preserves the exact CPU attention/cache path. Returning
    /// `Some` must still leave the caller's CPU shadow authoritative so
    /// snapshots and Metal fallback remain transactional.
    fn stateful_attention(
        &self,
        _layer: usize,
        _input: DFlashStatefulAttentionInput<'_>,
    ) -> Result<Option<Vec<f32>>, String> {
        Ok(None)
    }

    /// Allocation-free form of [`Self::stateful_attention`]. Backends with a
    /// resident output buffer should override this so the draft runner does
    /// not allocate and then copy a full `[T,H,D]` tensor on every layer.
    /// The default keeps third-party/legacy backends source-compatible.
    fn stateful_attention_into(
        &self,
        layer: usize,
        input: DFlashStatefulAttentionInput<'_>,
        output: &mut [f32],
    ) -> Result<bool, String> {
        let Some(values) = self.stateful_attention(layer, input)? else {
            return Ok(false);
        };
        if values.len() != output.len() {
            return Err(format!(
                "stateful attention returned {} elements, expected {}",
                values.len(),
                output.len()
            ));
        }
        output.copy_from_slice(&values);
        Ok(true)
    }

    /// Run a complete gated FFN when the backend carries a fused graph for
    /// it. `None` keeps the projection-by-projection path used by Metal and
    /// legacy Core ML artifacts. Public-CoreML v3 uses this hook to keep the
    /// gate, SiLU, multiply, and down projection inside three resident ANE
    /// output-partition graphs instead of crossing the host six times.
    fn fused_ffn(&self, _layer: usize, _input: &[f32]) -> Result<Option<Vec<f32>>, String> {
        Ok(None)
    }

    /// Whether this backend owns the complete post-attention tail for a
    /// layer. When true, the ordinary output projection is skipped and
    /// `fused_layer_tail` receives the attention rows plus the layer residual.
    fn has_fused_layer_tail(&self, _layer: usize) -> bool {
        false
    }

    /// Run output projection, post-attention residual/RMSNorm, and the gated
    /// FFN as one backend-owned tail. Public-CoreML v5+ uses two bounded ANE
    /// programs per layer; other backends retain the existing path.
    fn fused_layer_tail(
        &self,
        _layer: usize,
        _attention: &[f32],
        _residual: &[f32],
    ) -> Result<Option<Vec<f32>>, String> {
        Ok(None)
    }

    /// Allocation-free form of [`Self::fused_layer_tail`]. The default keeps
    /// compatibility while resident backends can write the next hidden block
    /// directly into the draft arena.
    fn fused_layer_tail_into(
        &self,
        layer: usize,
        attention: &[f32],
        residual: &[f32],
        output: &mut [f32],
    ) -> Result<bool, String> {
        let Some(values) = self.fused_layer_tail(layer, attention, residual)? else {
            return Ok(false);
        };
        if values.len() != output.len() {
            return Err(format!(
                "fused layer tail returned {} elements, expected {}",
                values.len(),
                output.len()
            ));
        }
        output.copy_from_slice(&values);
        Ok(true)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn project_to(
    backend: Option<&dyn DFlashProjectionBackend>,
    name: &str,
    input: &[f32],
    weights: &[f32],
    output: &mut [f32],
    rows: usize,
    output_width: usize,
    input_width: usize,
) -> Result<(), DFlashError> {
    if let Some(backend) = backend {
        let projected = backend
            .project(name, input)
            .map_err(DFlashError::Projection)?;
        if projected.len() != rows * output_width {
            return Err(DFlashError::Projection(format!(
                "{name} returned {} elements, expected {}",
                projected.len(),
                rows * output_width
            )));
        }
        output[..projected.len()].copy_from_slice(&projected);
    } else {
        matmul(input, weights, output, rows, output_width, input_width);
    }
    Ok(())
}
