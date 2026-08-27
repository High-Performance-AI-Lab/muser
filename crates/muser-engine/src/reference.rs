//! Muse Glimmer reference forward pass (CPU, f32).
//!
//! Transcribed op-for-op from llama.cpp `src/models/muse-glimmer.cpp`
//! (commit `d2f83055d`), which is the golden spec for this architecture.
//! Activation layout matches ggml exactly — the feature dim is contiguous and
//! tokens are the outer index — so a capture from here lines up element-for-
//! element with `llama-eval-callback` output.
//!
//! Traps this file exists to get right, all of which produce a model that runs
//! and emits plausible text when got wrong:
//!   * `attn_scale` is `1/sqrt(head_dim)` **on top of** the 3.87 folded into
//!     the Q-norm weights — not instead of it.
//!   * Post-attention and post-FFN norms use eps `1e-8`, every other norm uses
//!     the GGUF's `1e-5`.
//!   * RoPE runs on *sliding* layers; *full* layers are NoPE. This is inverted
//!     relative to Gemma 3.
//!   * The attention gate is driven by the pre-attention normed hidden state,
//!     and multiplies between SDPA and `o_proj`.
//!   * The logit scale is applied *before* the softcap, so the two do not
//!     commute.

use crate::quant::silu_fast;

use crate::cache::{
    f32s_to_le_bytes, le_bytes_to_f32s, CachePlaneSnapshot, PlaneEncoding, SessionCacheSnapshot,
};
use crate::capture::Capture;
use crate::config::{MuseConfig, MuseLayerKind};
use crate::weights::{matmul, MuseWeights, TensorView};

/// Rolling KV cache in ggml order: `[pos][kv_head][head_dim]`.
pub struct KvCache {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub len: usize,
    kv_dim: usize,
}

impl KvCache {
    pub fn new(n_layers: usize, max_seq: usize, kv_dim: usize) -> Vec<Self> {
        (0..n_layers)
            .map(|_| Self {
                k: vec![0.0; max_seq * kv_dim],
                v: vec![0.0; max_seq * kv_dim],
                len: 0,
                kv_dim,
            })
            .collect()
    }
    fn append(&mut self, k: &[f32], v: &[f32], n_tokens: usize) {
        let off = self.len * self.kv_dim;
        self.k[off..off + n_tokens * self.kv_dim].copy_from_slice(&k[..n_tokens * self.kv_dim]);
        self.v[off..off + n_tokens * self.kv_dim].copy_from_slice(&v[..n_tokens * self.kv_dim]);
        self.len += n_tokens;
    }
}

/// Weightless RMSNorm over the trailing `dim` axis, in place.
fn rms_norm(x: &mut [f32], dim: usize, eps: f32) {
    for row in x.chunks_mut(dim) {
        let mut ss = 0.0f32;
        for &v in row.iter() {
            ss += v * v;
        }
        let scale = 1.0 / (ss / dim as f32 + eps).sqrt();
        for v in row.iter_mut() {
            *v *= scale;
        }
    }
}

/// RMSNorm followed by an elementwise weight multiply — llama.cpp `build_norm`.
fn rms_norm_mul(x: &mut [f32], dim: usize, eps: f32, w: &[f32]) {
    rms_norm(x, dim, eps);
    for row in x.chunks_mut(dim) {
        for (v, &wi) in row.iter_mut().zip(w.iter()) {
            *v *= wi;
        }
    }
}

/// `LLAMA_ROPE_TYPE_NORM`: rotate the interleaved pairs `(x[2i], x[2i+1])`.
///
/// The converter un-permutes Q/K at conversion time precisely so this
/// interleaved form is the correct one for GGUF weights; an engine that keeps
/// HF's `rotate_half` layout must not apply the same permute.
fn rope_norm_inplace(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    pos: usize,
    base: f32,
) {
    let theta_scale = base.powf(-2.0 / rope_dim as f32);
    for h in 0..n_heads {
        let head = &mut x[h * head_dim..(h + 1) * head_dim];
        let mut theta = pos as f32;
        let mut i = 0;
        while i < rope_dim {
            let (sin_t, cos_t) = theta.sin_cos();
            let x0 = head[i];
            let x1 = head[i + 1];
            head[i] = x0 * cos_t - x1 * sin_t;
            head[i + 1] = x0 * sin_t + x1 * cos_t;
            theta *= theta_scale;
            i += 2;
        }
    }
}

/// The full model: config, mapped weights, KV cache.
pub struct MuseModel {
    pub cfg: MuseConfig,
    pub weights: MuseWeights,
    pub kv: Vec<KvCache>,
    /// Number of tokens already committed to the cache.
    pub n_past: usize,
}

impl MuseModel {
    pub fn new(cfg: MuseConfig, weights: MuseWeights, max_seq: usize) -> Self {
        let kv = KvCache::new(cfg.n_layers, max_seq, cfg.kv_dim());
        Self {
            cfg,
            weights,
            kv,
            n_past: 0,
        }
    }

    pub fn reset(&mut self) {
        for c in self.kv.iter_mut() {
            c.len = 0;
        }
        self.n_past = 0;
    }

    pub(crate) fn export_cache_snapshot(
        &self,
        tokens: &[u32],
    ) -> Result<SessionCacheSnapshot, String> {
        if self.n_past == 0 || tokens.len() != self.n_past {
            return Err("CPU cache export requires the exact nonempty token history".into());
        }
        let kv_dim = self.cfg.kv_dim();
        let mut layers = Vec::with_capacity(self.kv.len());
        for (layer, cache) in self.kv.iter().enumerate() {
            if cache.len != self.n_past {
                return Err(format!(
                    "CPU cache layer {layer} ends at {}, expected {}",
                    cache.len, self.n_past
                ));
            }
            let count = if self.cfg.layer_kinds[layer].is_swa() {
                self.n_past.min(self.cfg.sliding_window)
            } else {
                self.n_past
            };
            let start = self.n_past - count;
            let element_start = start * kv_dim;
            let element_end = self.n_past * kv_dim;
            layers.push(CachePlaneSnapshot {
                layer: layer as u32,
                logical_start: start as u64,
                logical_count: count as u64,
                encoding: PlaneEncoding::F32Le,
                key: f32s_to_le_bytes(&cache.k[element_start..element_end]),
                value: f32s_to_le_bytes(&cache.v[element_start..element_end]),
            });
        }
        let snapshot = SessionCacheSnapshot {
            position: self.n_past as u64,
            tokens: tokens.to_vec().into(),
            elements_per_token: kv_dim as u32,
            layers: layers.into(),
        };
        snapshot.validate_for_window(self.cfg.sliding_window)?;
        Ok(snapshot)
    }

    pub(crate) fn install_cache_snapshot(
        &mut self,
        snapshot: &SessionCacheSnapshot,
    ) -> Result<(), String> {
        snapshot.validate_for_window(self.cfg.sliding_window)?;
        if snapshot.elements_per_token as usize != self.cfg.kv_dim()
            || snapshot.position as usize > self.kv[0].k.len() / self.cfg.kv_dim()
            || snapshot.encoding() != Some(PlaneEncoding::F32Le)
        {
            return Err("CPU cache snapshot geometry or encoding mismatch".into());
        }
        let kv_dim = self.cfg.kv_dim();
        let max_seq = self.kv[0].k.len() / kv_dim;
        let mut detached = KvCache::new(self.cfg.n_layers, max_seq, kv_dim);
        for plane in snapshot.layers.iter() {
            let layer = plane.layer as usize;
            let key = le_bytes_to_f32s(&plane.key)?;
            let value = le_bytes_to_f32s(&plane.value)?;
            let start = plane.logical_start as usize * kv_dim;
            let end = start + key.len();
            detached[layer].k[start..end].copy_from_slice(&key);
            detached[layer].v[start..end].copy_from_slice(&value);
            detached[layer].len = snapshot.position as usize;
        }
        self.kv = detached;
        self.n_past = snapshot.position as usize;
        Ok(())
    }

    fn w(&self, name: &str) -> TensorView<'_> {
        self.weights
            .view(name)
            .unwrap_or_else(|e| panic!("muse: missing weight {name}: {e}"))
    }

    /// Run `tokens` as one batch starting at the current cache position and
    /// return the post-softcap logits for every token, `[n_tokens][vocab]`.
    ///
    /// When `cap` is `Some`, records the same intermediate tensors llama.cpp's
    /// eval callback sees, under the same node names.
    pub fn forward(&mut self, tokens: &[u32], cap: Option<&mut Capture>) -> Vec<f32> {
        self.forward_inputs(tokens, None, cap, None, None)
    }

    /// Run already-projected vision embeddings through the decoder at their
    /// exact insertion positions. The vectors enter at the same point as
    /// token embeddings, including Muse's weightless entry RMSNorm.
    pub fn forward_embeddings(
        &mut self,
        embeddings: &[f32],
        cap: Option<&mut Capture>,
    ) -> Vec<f32> {
        assert_eq!(embeddings.len() % self.cfg.hidden_dim, 0);
        self.forward_inputs(&[], Some(embeddings), cap, None, None)
    }

    /// Run target tokens and retain full layer-output rows selected by the
    /// DFlash artifact. Returned storage is token-major
    /// `[token][selected-layer][hidden]`.
    pub fn forward_capturing_layers(
        &mut self,
        tokens: &[u32],
        layer_ids: &[usize],
    ) -> (Vec<f32>, Vec<f32>) {
        let mut captured = Vec::new();
        let logits = self.forward_inputs(tokens, None, None, Some(&mut captured), None);
        let mut output = Vec::with_capacity(tokens.len() * layer_ids.len() * self.cfg.hidden_dim);
        for token in 0..tokens.len() {
            for &layer in layer_ids {
                let (_, rows) = captured
                    .iter()
                    .find(|(candidate, _)| *candidate == layer)
                    .expect("requested validated Muse capture layer");
                let start = token * self.cfg.hidden_dim;
                output.extend_from_slice(&rows[start..start + self.cfg.hidden_dim]);
            }
        }
        (logits, output)
    }

    /// Vision-embedding counterpart of [`Self::forward_capturing_layers`].
    /// DFlash consumes positions, not token identities, so projected image
    /// rows participate in the same token-major target-layer stream.
    pub fn forward_embeddings_capturing_layers(
        &mut self,
        embeddings: &[f32],
        layer_ids: &[usize],
    ) -> (Vec<f32>, Vec<f32>) {
        assert_eq!(embeddings.len() % self.cfg.hidden_dim, 0);
        let count = embeddings.len() / self.cfg.hidden_dim;
        let mut captured = Vec::new();
        let logits = self.forward_inputs(&[], Some(embeddings), None, Some(&mut captured), None);
        let mut output = Vec::with_capacity(count * layer_ids.len() * self.cfg.hidden_dim);
        for token in 0..count {
            for &layer in layer_ids {
                let (_, rows) = captured
                    .iter()
                    .find(|(candidate, _)| *candidate == layer)
                    .expect("requested validated Muse capture layer");
                let start = token * self.cfg.hidden_dim;
                output.extend_from_slice(&rows[start..start + self.cfg.hidden_dim]);
            }
        }
        (logits, output)
    }

    /// Final, output-normalized decoder rows used by the public embedding
    /// contract. This is captured before the LM head and after every decoder
    /// block and the model's exact output RMSNorm.
    pub fn forward_final_hidden(&mut self, tokens: &[u32]) -> (Vec<f32>, Vec<f32>) {
        let mut hidden = Vec::new();
        let logits = self.forward_inputs(tokens, None, None, None, Some(&mut hidden));
        (logits, hidden)
    }

    fn forward_inputs(
        &mut self,
        tokens: &[u32],
        embeddings: Option<&[f32]>,
        mut cap: Option<&mut Capture>,
        mut hidden_capture: Option<&mut Vec<(usize, Vec<f32>)>>,
        final_hidden: Option<&mut Vec<f32>>,
    ) -> Vec<f32> {
        let cfg = self.cfg.clone();
        let t = embeddings
            .map(|values| values.len() / cfg.hidden_dim)
            .unwrap_or(tokens.len());
        let h_dim = cfg.hidden_dim;
        let ffn_dim = cfg.intermediate_dim;
        let attn_dim = cfg.attn_dim();
        let kv_dim = cfg.kv_dim();
        let head_dim = cfg.head_dim;
        let start_pos = self.n_past;

        // ── embeddings ────────────────────────────────────────────────────
        // Weightless RMSNorm on the token embeddings; no sqrt(n_embd) scale.
        let mut hidden = vec![0.0f32; t * h_dim];
        if let Some(embeddings) = embeddings {
            hidden.copy_from_slice(embeddings);
        } else {
            let embd = self.w("token_embd.weight");
            for (i, &tok) in tokens.iter().enumerate() {
                crate::weights::dequant_row(
                    &embd,
                    tok as usize,
                    &mut hidden[i * h_dim..(i + 1) * h_dim],
                );
            }
        }
        if let Some(c) = cap.as_deref_mut() {
            c.record("embd", &hidden, &[h_dim, t]);
        }
        rms_norm(&mut hidden, h_dim, cfg.rms_eps);
        if let Some(c) = cap.as_deref_mut() {
            c.record("embd_norm", &hidden, &[h_dim, t]);
        }

        let mut q = vec![0.0f32; t * attn_dim];
        let mut k = vec![0.0f32; t * kv_dim];
        let mut v = vec![0.0f32; t * kv_dim];
        let mut gate = vec![0.0f32; t * attn_dim];
        let mut attn_out = vec![0.0f32; t * attn_dim];
        let mut proj = vec![0.0f32; t * h_dim];
        let mut ffn_a = vec![0.0f32; t * ffn_dim];
        let mut ffn_b = vec![0.0f32; t * ffn_dim];

        for il in 0..cfg.n_layers {
            let kind = cfg.layer_kinds[il];
            let residual = hidden.clone();

            // ── pre-attention norm ────────────────────────────────────────
            let mut x = hidden.clone();
            let attn_norm_w = self
                .weights
                .f32_vec(&format!("blk.{il}.attn_norm.weight"))
                .unwrap();
            rms_norm_mul(&mut x, h_dim, cfg.rms_eps, &attn_norm_w);
            if let Some(c) = cap.as_deref_mut() {
                c.record(&format!("attn_norm-{il}"), &x, &[h_dim, t]);
            }

            // ── Q/K/V + gate, all from the same normed input ──────────────
            matmul(&self.w(&format!("blk.{il}.attn_q.weight")), &x, t, &mut q);
            matmul(&self.w(&format!("blk.{il}.attn_k.weight")), &x, t, &mut k);
            matmul(&self.w(&format!("blk.{il}.attn_v.weight")), &x, t, &mut v);
            if let Some(c) = cap.as_deref_mut() {
                c.record(&format!("Qcur-{il}"), &q, &[head_dim, cfg.n_heads, t]);
                c.record(&format!("Kcur-{il}"), &k, &[head_dim, cfg.n_kv_heads, t]);
                c.record(&format!("Vcur-{il}"), &v, &[head_dim, cfg.n_kv_heads, t]);
            }
            matmul(
                &self.w(&format!("blk.{il}.attn_gate.weight")),
                &x,
                t,
                &mut gate,
            );
            if let Some(c) = cap.as_deref_mut() {
                c.record(&format!("attn_gate_proj-{il}"), &gate, &[attn_dim, t]);
            }

            // ── per-head QK norm ──────────────────────────────────────────
            // Weight is a constant broadcast of qk_scale_factor (Q) / 1.0 (K);
            // the eps is the GGUF's, not the post-norm 1e-8.
            let q_norm_w = self
                .weights
                .f32_vec(&format!("blk.{il}.attn_q_norm.weight"))
                .unwrap();
            let k_norm_w = self
                .weights
                .f32_vec(&format!("blk.{il}.attn_k_norm.weight"))
                .unwrap();
            rms_norm_mul(&mut q, head_dim, cfg.rms_eps, &q_norm_w);
            rms_norm_mul(&mut k, head_dim, cfg.rms_eps, &k_norm_w);
            if let Some(c) = cap.as_deref_mut() {
                c.record(
                    &format!("Qcur_normed-{il}"),
                    &q,
                    &[head_dim, cfg.n_heads, t],
                );
                c.record(
                    &format!("Kcur_normed-{il}"),
                    &k,
                    &[head_dim, cfg.n_kv_heads, t],
                );
            }

            // ── RoPE, sliding layers only ─────────────────────────────────
            if kind.uses_rope() {
                for ti in 0..t {
                    let pos = start_pos + ti;
                    rope_norm_inplace(
                        &mut q[ti * attn_dim..(ti + 1) * attn_dim],
                        cfg.n_heads,
                        head_dim,
                        cfg.rope_dim,
                        pos,
                        cfg.rope_base_swa,
                    );
                    rope_norm_inplace(
                        &mut k[ti * kv_dim..(ti + 1) * kv_dim],
                        cfg.n_kv_heads,
                        head_dim,
                        cfg.rope_dim,
                        pos,
                        cfg.rope_base_swa,
                    );
                }
                if let Some(c) = cap.as_deref_mut() {
                    c.record(&format!("Qcur_rope-{il}"), &q, &[head_dim, cfg.n_heads, t]);
                    c.record(
                        &format!("Kcur_rope-{il}"),
                        &k,
                        &[head_dim, cfg.n_kv_heads, t],
                    );
                }
            }

            // ── attention ─────────────────────────────────────────────────
            self.kv[il].append(&k, &v, t);
            self.attention(il, kind, &q, start_pos, &mut attn_out);
            if let Some(c) = cap.as_deref_mut() {
                c.record(&format!("attn_out-{il}"), &attn_out, &[attn_dim, t]);
            }

            // ── sigmoid gate, then o_proj ─────────────────────────────────
            for g in gate.iter_mut() {
                *g = 1.0 / (1.0 + (-*g).exp());
            }
            if let Some(c) = cap.as_deref_mut() {
                c.record(&format!("attn_gate_sig-{il}"), &gate, &[attn_dim, t]);
            }
            for (a, g) in attn_out.iter_mut().zip(gate.iter()) {
                *a *= *g;
            }
            matmul(
                &self.w(&format!("blk.{il}.attn_output.weight")),
                &attn_out,
                t,
                &mut proj,
            );
            if let Some(c) = cap.as_deref_mut() {
                c.record(&format!("attn_o_proj-{il}"), &proj, &[h_dim, t]);
            }

            // ── post-attention norm (eps 1e-8) + residual ─────────────────
            let post_attn_w = self
                .weights
                .f32_vec(&format!("blk.{il}.post_attention_norm.weight"))
                .unwrap();
            rms_norm_mul(&mut proj, h_dim, cfg.post_norm_eps, &post_attn_w);
            if let Some(c) = cap.as_deref_mut() {
                c.record(&format!("attn_post_norm-{il}"), &proj, &[h_dim, t]);
            }
            for (hv, (p, r)) in hidden.iter_mut().zip(proj.iter().zip(residual.iter())) {
                *hv = *p + *r;
            }
            if let Some(c) = cap.as_deref_mut() {
                c.record(&format!("ffn_inp-{il}"), &hidden, &[h_dim, t]);
            }

            // ── FFN ───────────────────────────────────────────────────────
            let ffn_residual = hidden.clone();
            let mut y = hidden.clone();
            let ffn_norm_w = self
                .weights
                .f32_vec(&format!("blk.{il}.ffn_norm.weight"))
                .unwrap();
            rms_norm_mul(&mut y, h_dim, cfg.rms_eps, &ffn_norm_w);
            if let Some(c) = cap.as_deref_mut() {
                c.record(&format!("ffn_norm-{il}"), &y, &[h_dim, t]);
            }
            matmul(
                &self.w(&format!("blk.{il}.ffn_gate.weight")),
                &y,
                t,
                &mut ffn_a,
            );
            if let Some(c) = cap.as_deref_mut() {
                c.record(&format!("ffn_gate-{il}"), &ffn_a, &[ffn_dim, t]);
            }
            matmul(
                &self.w(&format!("blk.{il}.ffn_up.weight")),
                &y,
                t,
                &mut ffn_b,
            );
            if let Some(c) = cap.as_deref_mut() {
                c.record(&format!("ffn_up-{il}"), &ffn_b, &[ffn_dim, t]);
            }
            for (a, b) in ffn_a.iter_mut().zip(ffn_b.iter()) {
                *a = silu_fast(*a) * *b;
            }
            if let Some(c) = cap.as_deref_mut() {
                c.record(&format!("ffn_swiglu-{il}"), &ffn_a, &[ffn_dim, t]);
            }
            matmul(
                &self.w(&format!("blk.{il}.ffn_down.weight")),
                &ffn_a,
                t,
                &mut proj,
            );
            if let Some(c) = cap.as_deref_mut() {
                c.record(&format!("ffn_out-{il}"), &proj, &[h_dim, t]);
            }

            // ── post-FFN norm (eps 1e-8) + residual ───────────────────────
            let post_ffw_w = self
                .weights
                .f32_vec(&format!("blk.{il}.post_ffw_norm.weight"))
                .unwrap();
            rms_norm_mul(&mut proj, h_dim, cfg.post_norm_eps, &post_ffw_w);
            if let Some(c) = cap.as_deref_mut() {
                c.record(&format!("ffn_post_norm-{il}"), &proj, &[h_dim, t]);
            }
            for (hv, (p, r)) in hidden.iter_mut().zip(proj.iter().zip(ffn_residual.iter())) {
                *hv = *p + *r;
            }
            if let Some(c) = cap.as_deref_mut() {
                c.record(&format!("l_out-{il}"), &hidden, &[h_dim, t]);
            }
            if let Some(captured) = hidden_capture.as_deref_mut() {
                captured.push((il, hidden.clone()));
            }
        }

        self.n_past += t;

        // ── final norm → lm_head → scale → softcap ────────────────────────
        let output_norm_w = self.weights.f32_vec("output_norm.weight").unwrap();
        rms_norm_mul(&mut hidden, h_dim, cfg.rms_eps, &output_norm_w);
        if let Some(output) = final_hidden {
            output.extend_from_slice(&hidden);
        }
        if let Some(c) = cap.as_deref_mut() {
            c.record("result_norm", &hidden, &[h_dim, t]);
        }

        let mut logits = vec![0.0f32; t * cfg.vocab_size];
        matmul(&self.w("output.weight"), &hidden, t, &mut logits);
        for l in logits.iter_mut() {
            *l *= cfg.logit_scale;
        }
        if cfg.final_logit_softcap > 0.0 {
            let cap_v = cfg.final_logit_softcap;
            let inv = 1.0 / cap_v;
            for l in logits.iter_mut() {
                *l = cap_v * (*l * inv).tanh();
            }
        }
        if let Some(c) = cap {
            c.record("result_output", &logits, &[cfg.vocab_size, t]);
        }
        logits
    }

    /// Grouped-query causal attention with optional sliding window.
    ///
    /// Mask follows llama.cpp `LLAMA_SWA_TYPE_STANDARD`: a key at position `p0`
    /// is visible to a query at `p1` iff `p1 >= p0` and `p1 - p0 < n_swa`.
    fn attention(
        &self,
        il: usize,
        kind: MuseLayerKind,
        q: &[f32],
        start_pos: usize,
        out: &mut [f32],
    ) {
        let cfg = &self.cfg;
        let head_dim = cfg.head_dim;
        let heads_per_kv = cfg.heads_per_kv();
        let kv_dim = cfg.kv_dim();
        let attn_dim = cfg.attn_dim();
        let scale = cfg.attn_scale();
        let cache = &self.kv[il];
        let window = if kind.is_swa() { cfg.sliding_window } else { 0 };

        use rayon::prelude::*;
        out.par_chunks_mut(attn_dim)
            .enumerate()
            .for_each(|(ti, out_row)| {
                let pos = start_pos + ti;
                let lo = if window > 0 {
                    pos.saturating_sub(window - 1)
                } else {
                    0
                };
                let hi = pos; // inclusive
                let n_vis = hi + 1 - lo;
                let mut scores = vec![0.0f32; n_vis];
                for h in 0..cfg.n_heads {
                    let kvh = h / heads_per_kv;
                    let qh = &q[ti * attn_dim + h * head_dim..ti * attn_dim + (h + 1) * head_dim];
                    let mut max = f32::NEG_INFINITY;
                    for (j, s) in scores.iter_mut().enumerate() {
                        let koff = (lo + j) * kv_dim + kvh * head_dim;
                        let kv = &cache.k[koff..koff + head_dim];
                        let mut acc = 0.0f32;
                        for d in 0..head_dim {
                            acc += qh[d] * kv[d];
                        }
                        let sv = acc * scale;
                        *s = sv;
                        if sv > max {
                            max = sv;
                        }
                    }
                    let mut sum = 0.0f32;
                    for s in scores.iter_mut() {
                        *s = (*s - max).exp();
                        sum += *s;
                    }
                    let inv = 1.0 / sum;
                    let dst = &mut out_row[h * head_dim..(h + 1) * head_dim];
                    dst.fill(0.0);
                    for (j, &p) in scores.iter().enumerate() {
                        let w = p * inv;
                        let voff = (lo + j) * kv_dim + kvh * head_dim;
                        let vv = &cache.v[voff..voff + head_dim];
                        for d in 0..head_dim {
                            dst[d] += w * vv[d];
                        }
                    }
                }
            });
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "release-real-model")]
    use super::MuseModel;

    fn argmax(values: &[f32]) -> u32 {
        values
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index as u32)
            .expect("nonempty logits")
    }

    /// The oracle every GPU route is diffed against must be reproducible on
    /// its own: two runs of the same tiny greedy decode are bit-identical, so
    /// a parity failure can never be blamed on the reference drifting.
    #[cfg(feature = "release-real-model")]
    #[test]
    fn cpu_greedy_decode_repeats_bit_exactly() {
        let path = std::env::var("MUSER_MODEL")
            .expect("release-real-model requires MUSER_MODEL for the CPU oracle");
        let loaded = crate::loader::load_components(std::path::Path::new(&path))
            .expect("Muse GGUF must load");
        let prompt = [200_000u32, 19_873, 24];
        let steps = 4usize;

        let run = || {
            let mut model = MuseModel::new(
                loaded.config.clone(),
                loaded.weights.clone(),
                prompt.len() + steps,
            );
            let mut logits = model.forward(&prompt, None);
            let mut generated = Vec::with_capacity(steps);
            for _ in 0..steps {
                let next = argmax(&logits[logits.len() - loaded.config.vocab_size..]);
                generated.push(next);
                logits = model.forward(&[next], None);
            }
            (generated, logits)
        };

        let (first_tokens, first_logits) = run();
        let (second_tokens, second_logits) = run();
        assert_eq!(first_tokens, second_tokens, "greedy tokens must repeat");
        assert_eq!(
            first_logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            second_logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "reference logits must repeat bit for bit"
        );
    }
}
