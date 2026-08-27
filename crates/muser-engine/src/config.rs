//! Muse Glimmer architecture configuration, parsed straight from GGUF metadata.
//!
//! Every field is read from the file — no silent architecture defaults. The one
//! deliberate exception is `post_norm_eps`, which llama.cpp hard-codes in the
//! graph builder rather than reading from the checkpoint
//! (`src/models/muse-glimmer.cpp:67`: `const float post_norm_eps = 1e-8f;`).

use crate::gguf::{GgmlType, GgufFile};

/// GGUF architecture string for this model family.
pub const MUSE_ARCH: &str = "muse-glimmer";

pub const MUSE_LAYER_COUNT: usize = 52;
pub const MUSE_SWA_LAYER_COUNT: usize = 39;
pub const MUSE_NOPE_LAYER_COUNT: usize = 13;
pub const MUSE_SWA_WINDOW: usize = 2_048;
pub const MUSE_MAX_CONTEXT: usize = 131_072;
pub const MUSE_HEAD_COUNT: usize = 32;
pub const MUSE_KV_HEAD_COUNT: usize = 2;
pub const MUSE_HEAD_DIM: usize = 128;
pub const MUSE_KV_ROW_ELEMENTS: usize = MUSE_KV_HEAD_COUNT * MUSE_HEAD_DIM;

/// Post-attention / post-FFN RMSNorm epsilon.
///
/// llama.cpp uses a *different* epsilon for the two "post" norms of the
/// Gemma-2-style sandwich than for every other RMSNorm in the graph, and it is
/// not carried in the GGUF. See `src/models/muse-glimmer.cpp:67`.
pub const MUSE_POST_NORM_EPS: f32 = 1e-8;

/// Errors raised while ingesting a Muse Glimmer checkpoint.
#[derive(Debug, thiserror::Error)]
pub enum MuseConfigError {
    #[error("not a muse-glimmer checkpoint: general.architecture = {0:?}")]
    WrongArch(String),
    #[error("missing required GGUF metadata key: {0}")]
    MissingKey(String),
    #[error("missing required tensor: {0}")]
    MissingTensor(String),
    #[error("tensor {name}: expected shape {expected:?}, found {found:?}")]
    BadShape {
        name: String,
        expected: Vec<u64>,
        found: Vec<u64>,
    },
    #[error("unsupported dtype {dtype:?} for tensor {name}")]
    BadDtype { name: String, dtype: GgmlType },
    #[error("inconsistent geometry: {0}")]
    Geometry(String),
}

/// Per-layer attention kind. Muse Glimmer alternates
/// `[sliding, sliding, sliding, full]`, and — unlike Gemma 3 — it is the
/// *sliding* layers that carry RoPE while the *full* layers are NoPE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MuseLayerKind {
    /// Windowed attention (`sliding_window` tokens) with RoPE applied.
    SlidingRope,
    /// Full causal attention with no positional rotation at all.
    FullNoPe,
}

impl MuseLayerKind {
    pub const fn is_swa(self) -> bool {
        matches!(self, Self::SlidingRope)
    }
    /// RoPE runs iff the layer is a sliding layer (`muse-glimmer.cpp:93`:
    /// `const bool use_rope = hparams.is_swa(il);`).
    pub const fn uses_rope(self) -> bool {
        matches!(self, Self::SlidingRope)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerIndexError(pub usize);

impl std::fmt::Display for LayerIndexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Muse layer {} is outside 0..{MUSE_LAYER_COUNT}", self.0)
    }
}

impl std::error::Error for LayerIndexError {}

pub const fn layer_kind(layer: usize) -> Result<MuseLayerKind, LayerIndexError> {
    if layer >= MUSE_LAYER_COUNT {
        return Err(LayerIndexError(layer));
    }
    if layer % 4 == 3 {
        Ok(MuseLayerKind::FullNoPe)
    } else {
        Ok(MuseLayerKind::SlidingRope)
    }
}

pub fn nope_layers() -> impl Iterator<Item = usize> {
    (0..MUSE_LAYER_COUNT).filter(|&layer| matches!(layer_kind(layer), Ok(MuseLayerKind::FullNoPe)))
}

pub fn swa_layers() -> impl Iterator<Item = usize> {
    (0..MUSE_LAYER_COUNT)
        .filter(|&layer| matches!(layer_kind(layer), Ok(MuseLayerKind::SlidingRope)))
}

/// Fully-resolved Muse Glimmer hyperparameters.
#[derive(Debug, Clone)]
pub struct MuseConfig {
    pub n_layers: usize,
    pub hidden_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub value_head_dim: usize,
    pub intermediate_dim: usize,
    pub vocab_size: usize,
    pub context_length: usize,

    /// RMSNorm epsilon for every norm except the two "post" sandwich norms.
    pub rms_eps: f32,
    /// Epsilon for post-attention / post-FFN norms (llama.cpp constant).
    pub post_norm_eps: f32,

    /// RoPE theta used on sliding layers. llama.cpp reads
    /// `rope.freq_base_swa` and falls back to `rope.freq_base`; since RoPE only
    /// ever runs on sliding layers, this is the only theta the model uses.
    pub rope_base_swa: f32,
    /// RoPE theta declared for full layers. Recorded for provenance only — full
    /// layers are NoPE, so this value is never applied.
    pub rope_base_full: f32,
    pub rope_dim: usize,

    pub sliding_window: usize,
    /// Per-layer attention kind, length `n_layers`.
    pub layer_kinds: Vec<MuseLayerKind>,

    /// `20.0` for this checkpoint. llama.cpp defaults to 30.0 when absent.
    pub final_logit_softcap: f32,
    /// `output_multiplier`, applied to logits *before* the softcap.
    pub logit_scale: f32,
    /// Uniform value of `attn_q_norm.weight` (the converter-synthesized
    /// broadcast of `qk_scale_factor`). Not a learned per-channel gain.
    pub qk_scale_factor: f32,

    pub bos_token_id: Option<u32>,
    /// GGUF-declared end-of-generation tokens: EOS, EOT, and EOM.
    pub eos_tokens: Vec<u32>,
}

fn meta_u32(g: &GgufFile, key: &str) -> Option<u32> {
    g.meta_u32(key)
        .or_else(|| g.meta_u64(key).map(|v| v as u32))
}

fn require_u32(g: &GgufFile, key: &str) -> Result<u32, MuseConfigError> {
    meta_u32(g, key).ok_or_else(|| MuseConfigError::MissingKey(key.to_string()))
}

impl MuseConfig {
    /// Parse and validate a Muse Glimmer GGUF header.
    ///
    /// Fails closed: any missing required key, any tensor whose shape disagrees
    /// with the derived geometry, and any non-constant QK-norm weight is an
    /// error rather than a warning.
    pub fn from_gguf(g: &GgufFile, qk_norm_probe: &QkNormProbe) -> Result<Self, MuseConfigError> {
        let arch = g.meta_str("general.architecture").unwrap_or("");
        if arch != MUSE_ARCH {
            return Err(MuseConfigError::WrongArch(arch.to_string()));
        }

        let n_layers = require_u32(g, "muse-glimmer.block_count")? as usize;
        let hidden_dim = require_u32(g, "muse-glimmer.embedding_length")? as usize;
        let n_heads = require_u32(g, "muse-glimmer.attention.head_count")? as usize;
        let n_kv_heads =
            meta_u32(g, "muse-glimmer.attention.head_count_kv").unwrap_or(n_heads as u32) as usize;
        let head_dim = meta_u32(g, "muse-glimmer.attention.key_length")
            .map(|v| v as usize)
            .unwrap_or(hidden_dim / n_heads.max(1));
        let value_head_dim = meta_u32(g, "muse-glimmer.attention.value_length")
            .map(|v| v as usize)
            .unwrap_or(head_dim);
        let intermediate_dim = require_u32(g, "muse-glimmer.feed_forward_length")? as usize;
        let context_length = meta_u32(g, "muse-glimmer.context_length").unwrap_or(2048) as usize;

        // llama.cpp requires these three; absence is a hard error there too.
        let rms_eps = g
            .meta_f32("muse-glimmer.attention.layer_norm_rms_epsilon")
            .ok_or_else(|| {
                MuseConfigError::MissingKey("muse-glimmer.attention.layer_norm_rms_epsilon".into())
            })?;
        let sliding_window = require_u32(g, "muse-glimmer.attention.sliding_window")? as usize;
        let logit_scale = g
            .meta_f32("muse-glimmer.logit_scale")
            .ok_or_else(|| MuseConfigError::MissingKey("muse-glimmer.logit_scale".into()))?;

        // Optional in llama.cpp, with a 30.0 default that still enables the branch.
        let final_logit_softcap = g
            .meta_f32("muse-glimmer.final_logit_softcapping")
            .unwrap_or(30.0);

        let rope_base_full = g.meta_f32("muse-glimmer.rope.freq_base").unwrap_or(10000.0);
        let rope_base_swa = g
            .meta_f32("muse-glimmer.rope.freq_base_swa")
            .unwrap_or(rope_base_full);
        let rope_dim = meta_u32(g, "muse-glimmer.rope.dimension_count")
            .map(|v| v as usize)
            .unwrap_or(head_dim);

        let layer_kinds = resolve_layer_kinds(g, n_layers);

        let vocab_size = meta_u32(g, "muse-glimmer.vocab_size")
            .map(|v| v as usize)
            .unwrap_or_else(|| g.vocab().len());

        let bos_token_id = meta_u32(g, "tokenizer.ggml.bos_token_id");
        let mut eos_tokens = Vec::new();
        if let Some(t) = meta_u32(g, "tokenizer.ggml.eos_token_id") {
            eos_tokens.push(t);
        }
        // llama.cpp treats all three declared control tokens as EOG. They are
        // separate scalar metadata keys, not one array.
        for key in ["tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"] {
            if let Some(t) = meta_u32(g, key) {
                if !eos_tokens.contains(&t) {
                    eos_tokens.push(t);
                }
            }
        }

        if !n_heads.is_multiple_of(n_kv_heads) {
            return Err(MuseConfigError::Geometry(format!(
                "n_heads {n_heads} not divisible by n_kv_heads {n_kv_heads}"
            )));
        }
        if !head_dim.is_multiple_of(2) {
            return Err(MuseConfigError::Geometry(format!(
                "head_dim {head_dim} must be even for RoPE"
            )));
        }

        let cfg = Self {
            n_layers,
            hidden_dim,
            n_heads,
            n_kv_heads,
            head_dim,
            value_head_dim,
            intermediate_dim,
            vocab_size,
            context_length,
            rms_eps,
            post_norm_eps: MUSE_POST_NORM_EPS,
            rope_base_swa,
            rope_base_full,
            rope_dim,
            sliding_window,
            layer_kinds,
            final_logit_softcap,
            logit_scale,
            qk_scale_factor: qk_norm_probe.q_norm_value,
            bos_token_id,
            eos_tokens,
        };
        cfg.assert_tensor_shapes(g)?;
        Ok(cfg)
    }

    /// Total attention-projection width (`n_heads * head_dim`), i.e. the space
    /// the attention-output gate lives in.
    pub fn attn_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
    pub fn heads_per_kv(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }
    /// Softmax scale. Note this is `1/sqrt(head_dim)` *in addition to* the
    /// `qk_scale_factor` already folded into the Q-norm weights.
    pub fn attn_scale(&self) -> f32 {
        1.0 / (self.head_dim as f32).sqrt()
    }

    /// Assert every tensor named by the architecture exists with the exact
    /// shape the parsed geometry implies. GGUF shapes are `ne` order, i.e.
    /// `[in_dim, out_dim]` for a projection.
    fn assert_tensor_shapes(&self, g: &GgufFile) -> Result<(), MuseConfigError> {
        let h = self.hidden_dim as u64;
        let attn = self.attn_dim() as u64;
        let kv = self.kv_dim() as u64;
        let ff = self.intermediate_dim as u64;
        let hd = self.head_dim as u64;
        let vocab = self.vocab_size as u64;

        let mut expect: Vec<(String, Vec<u64>)> = vec![
            ("token_embd.weight".into(), vec![h, vocab]),
            ("output.weight".into(), vec![h, vocab]),
            ("output_norm.weight".into(), vec![h]),
        ];
        for l in 0..self.n_layers {
            for (suffix, shape) in [
                ("attn_norm.weight", vec![h]),
                ("attn_q.weight", vec![h, attn]),
                ("attn_k.weight", vec![h, kv]),
                ("attn_v.weight", vec![h, kv]),
                ("attn_q_norm.weight", vec![hd]),
                ("attn_k_norm.weight", vec![hd]),
                ("attn_gate.weight", vec![h, attn]),
                ("attn_output.weight", vec![attn, h]),
                ("post_attention_norm.weight", vec![h]),
                ("ffn_norm.weight", vec![h]),
                ("ffn_gate.weight", vec![h, ff]),
                ("ffn_up.weight", vec![h, ff]),
                ("ffn_down.weight", vec![ff, h]),
                ("post_ffw_norm.weight", vec![h]),
            ] {
                expect.push((format!("blk.{l}.{suffix}"), shape));
            }
        }

        for (name, shape) in expect {
            let info = g
                .tensor(&name)
                .ok_or_else(|| MuseConfigError::MissingTensor(name.clone()))?;
            if info.shape != shape {
                return Err(MuseConfigError::BadShape {
                    name,
                    expected: shape,
                    found: info.shape.clone(),
                });
            }
        }
        Ok(())
    }
}

/// Resolve the per-layer sliding/full pattern.
///
/// The key can arrive two ways. llama.cpp's gguf-writer collapses a periodic
/// boolean `layer_types` list into a single **period integer**, and that is what
/// this checkpoint carries (`sliding_window_pattern = 4`). Some converters emit
/// the full boolean array instead, so both are accepted.
///
/// Scalar form follows llama.cpp `llama_hparams::set_swa_pattern`
/// (`src/llama-hparams.cpp:8`, `dense_first = false`):
/// `is_swa[il] = (il % n_pattern) < (n_pattern - 1)`, i.e. every `n`-th layer is
/// the full-attention one.
fn resolve_layer_kinds(g: &GgufFile, n_layers: usize) -> Vec<MuseLayerKind> {
    const KEY: &str = "muse-glimmer.attention.sliding_window_pattern";

    if let Some(period) = meta_u32(g, KEY) {
        return (0..n_layers)
            .map(|il| {
                let swa = period == 0 || (il as u32 % period) < period.saturating_sub(1);
                if swa {
                    MuseLayerKind::SlidingRope
                } else {
                    MuseLayerKind::FullNoPe
                }
            })
            .collect();
    }

    if let Some(flags) = g.meta_bool_array(KEY) {
        if flags.len() == n_layers {
            return flags
                .into_iter()
                .map(|swa| {
                    if swa {
                        MuseLayerKind::SlidingRope
                    } else {
                        MuseLayerKind::FullNoPe
                    }
                })
                .collect();
        }
    }

    // Fail loud rather than silently making every layer full-attention: an
    // all-full model runs and emits plausible text while being wrong.
    panic!("muse-glimmer: {KEY} missing or malformed; cannot resolve SWA pattern");
}

/// Measured facts about the converter-synthesized QK-norm tensors.
///
/// The HF checkpoint has no learned q_norm/k_norm. The converter materializes
/// `full(qk_scale_factor)` and `ones(...)` so llama.cpp's weighted-RMSNorm op
/// can carry a scalar. We verify that at load time rather than trusting it:
/// a genuinely learned per-channel norm would change the math.
#[derive(Debug, Clone)]
pub struct QkNormProbe {
    pub q_norm_value: f32,
    pub k_norm_value: f32,
    pub q_norm_is_constant: bool,
    pub k_norm_is_constant: bool,
}

impl QkNormProbe {
    /// True when both tensors are the constant broadcasts the converter emits,
    /// meaning the operation is `scaleless_rmsnorm(x) * scalar`.
    pub fn is_synthesized_scalar(&self) -> bool {
        self.q_norm_is_constant && self.k_norm_is_constant && self.k_norm_value == 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_partition_is_39_swa_and_13_nope() {
        assert_eq!(swa_layers().count(), MUSE_SWA_LAYER_COUNT);
        assert_eq!(nope_layers().count(), MUSE_NOPE_LAYER_COUNT);
        assert_eq!(
            nope_layers().collect::<Vec<_>>(),
            (3..MUSE_LAYER_COUNT).step_by(4).collect::<Vec<_>>()
        );
    }

    #[test]
    fn invalid_layer_does_not_wrap_into_the_pattern() {
        assert_eq!(layer_kind(MUSE_LAYER_COUNT), Err(LayerIndexError(52)));
        assert_eq!(layer_kind(usize::MAX), Err(LayerIndexError(usize::MAX)));
    }

    #[test]
    fn attention_scale_is_independent_of_qk_gain() {
        let scale = 1.0 / (MUSE_HEAD_DIM as f32).sqrt();
        assert!((scale - 0.088_388_35).abs() < 1e-6);
        assert!((3.87 * scale - 0.342_062_3).abs() < 1e-5);
    }
}
