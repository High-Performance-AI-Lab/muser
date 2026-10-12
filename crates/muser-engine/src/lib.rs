//! `muser-engine` — the Muse Glimmer-30B forward path.
//!
//! One model, one forward path, no model router or VM interpreter. See
//! `docs/muser-architecture.md` for the current runtime and
//! `docs/extraction-manifest.md` for historical Ferrite provenance.
//!
//! Model facts asserted from the pinned GGUF contract:
//! - 52 layers = 39 SWA (window 2048) + 13 NoPE full layers `{3,7,...,51}`.
//! - GQA 32:2, head_dim 128; the 13 NoPE layers are position-free (relocate
//!   = memcpy) — the whole kvpack free lunch.
//! - Parameterless QK-RMSNorm, sigmoid attention-output gate, Gemma-2-style
//!   sandwich norms (dual eps: 1e-5 / 1e-8 on the two post-norms), softcap 20
//!   logits, dual EOS.
//! - The pinned target artifact is 16,756,681,056 bytes on disk; residency is
//!   workload- and page-cache-dependent.
//!
//! Build shape: this crate compiles
//! CPU-only (`--no-default-features`, the `reference` oracle + `gguf` +
//! `quant`) on any host. The `metal` feature adds the Mac GPU decode/prefill
//! drivers and only makes sense under `target_os = "macos"`.
//!
//! # CPU load, prefill, and decode
//!
//! `Model` owns immutable, memory-mapped weights and tokenizer state;
//! `Session` owns the mutable KV cache and token history for one sequence.
//!
//! ```no_run
//! use muser_engine::{DecodeInput, Model, ModelConfig, PrefillBatch, SessionConfig};
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let model = Model::load(ModelConfig::new("/models/muse-glimmer.gguf"))?;
//!     let prompt = model.encode("Explain why KV caching helps decode.");
//!     let mut session = model.new_session(SessionConfig {
//!         max_context: prompt.len() + 32,
//!     })?;
//!
//!     let prefill = session.prefill(PrefillBatch::tokens(prompt))?;
//!     assert_eq!(prefill.last_logits().len(), model.config().vocab_size);
//!
//!     let first_token = session.greedy_next_token()?;
//!     let decoded = session.decode(DecodeInput {
//!         token_id: first_token,
//!     })?;
//!     println!("{}", model.decode_tokens(&[first_token, decoded.next_token]));
//!     Ok(())
//! }
//! ```

#![allow(dead_code)] // extraction is incremental; completed modules are tested

pub mod api;
pub use api::{
    DecodeDiagnostics, DecodeInput, DecodeResult, EmbeddingSegment, EngineError, Model,
    ModelConfig, PrefillBatch, PrefillResult, PrefillSegment, PreparedRemoteKvInstall,
    RemoteKvInstall, Session, SessionConfig, SpeculativeBatch, VerificationResult,
    EMBEDDING_POSITION_WITNESS,
};

/// Cache interchange reserved for the sibling `muser-kvpack` crate. It is
/// intentionally outside the stable engine API re-exports above.
#[doc(hidden)]
pub mod cache;

/// GGUF v3 parser (muse metadata only).
///
/// PULL-CLEAN from `crates/ferrite-inference/src/gguf.rs` +
/// `gguf/{parser,reader,metadata,types,accessors}.rs`. The muse loader
/// already uses only this parser. **Drop** the `#[cfg(test)]` untrusted-offset
/// `cache.rs` sidecar — not needed and not a trust boundary muser wants.
pub mod gguf;

/// `MuseConfig` + layer-kind resolver + `QkNormProbe`.
///
/// PULL-CLEAN from `crates/ferrite-inference/src/muse/config.rs`
/// (op-for-op transcribed from llama.cpp `muse-glimmer.cpp`, the golden
/// oracle). Encodes the 52-layer SWA/NoPE pattern, GQA 32:2 shapes, and the
/// fail-closed QK-norm verification (GGUF `attn_q_norm`/`attn_k_norm` are
/// converter-synthesized constant broadcasts of qk_scale_factor ≈ 3.87 / 1.0
/// — `QkNormProbe` fails closed if a real learned norm ever appears).
pub mod config;

/// mmap'd tensor views + quantized dot/dequant + matmul.
///
/// PULL-CLEAN from `crates/ferrite-inference/src/muse/weights.rs`:
/// `TensorView`, `dot_row`, `matmul`, `matmul_rows`. The reference documents
/// "prefill of T tokens ≈ one token's DRAM traffic" — the batching insight
/// `prefill.rs` implements against.
pub mod weights;

/// Quantized dot/dequant kernels, muse dtypes only.
///
/// PULL-AND-SIMPLIFY from `crates/ferrite-inference/src/quant/
/// {q4k,q5..,blocks,helpers,dispatch}.rs`. Keep only what the muse GGUF
/// actually uses: Q4_K, Q5_K, Q6_K, Q8_0, Q4_0, F16, F32. **Drop** the
/// IQ2/IQ3/MLX/codebook/subspace quant zoo (dead weight for a one-model
/// engine).
pub mod quant;

/// CPU f32 forward oracle — **the correctness gate**.
///
/// PULL-CLEAN, keep verbatim, from
/// `crates/ferrite-inference/src/muse/forward.rs:113-441`. This is the
/// bit-level spec for every trap in the architecture (inverted-Gemma
/// RoPE/NoPE split, sigmoid gate placement, dual-eps sandwich norms,
/// softcap-after-scale ordering, GPT-J RoPE pairing). `decode.rs` and
/// `prefill.rs` are transcriptions of this graph with GPU encoders swapped
/// in per op — this file is the spec they're checked against, not a draft
/// of it.
pub mod reference;

mod rope_nco;

/// Node-named intermediate-activation recorder (parity harness plumbing).
///
/// PULL-CLEAN from `crates/ferrite-inference/src/muse/capture.rs`. Used by
/// both `reference.rs` and the GPU drivers so `tests/muse_golden.rs` can
/// diff op-named node by node instead of only comparing final logits.
pub mod capture;

/// GGUF loader entry point + fail-closed QK-norm probe invocation.
///
/// PULL-CLEAN from `crates/ferrite-inference/src/muse.rs:54-112`
/// (`load()` + `probe_qk_norms`).
pub mod loader;

/// GGUF BPE tokenizer + muse chat template + dual-EOS (`eos` + `eot`).
///
/// PULL-AND-SIMPLIFY from `crates/ferrite-inference/src/tokenizer.rs` +
/// GGUF vocab accessors. Muse chat template only — no other model's
/// template branches carried over.
pub mod tokenizer;

/// Official Muse Glimmer 50-block vision graph and CPU oracle, ported from
/// pinned llama.cpp `0b1bad14ff20`'s mtmd implementation.
pub mod vision;

/// Standalone five-layer DFlash assistant and transactional verifier support.
/// The loader and CPU oracle are direct extractions of Ferrite's proven
/// DFlash implementation. Development SafeTensors directories and the pinned
/// official k-quant GGUF sidecar share one validated loader; no Ferrite crate
/// is linked at runtime.
pub mod dflash;

/// Canonical scalar temperature/top-k/top-p and full speculative sampling.
pub mod sampling;

mod safetensors;

/// Public-CoreML-only ANE runtime. No private Apple symbols are referenced.
#[cfg(all(target_os = "macos", feature = "ane-coreml"))]
pub mod coreml;
#[cfg(all(target_os = "macos", feature = "ane-coreml"))]
pub mod dflash_ane;
#[cfg(all(target_os = "macos", feature = "ane-coreml"))]
pub mod target_ane;

// The two-class KV allocator (13 NoPE growing planes + 39 SWA ring-windowed
// planes) lives on the shipping decode path as `decode::MetalKvPlane`. The
// standalone `kv::{ring,global_arena}` transcription of Ferrite's arena
// mechanics was consumed by nothing but its own tests, so wrap, partial
// accept, and rollback are covered against the shipping planes instead.

/// Metal runtime substrate + per-op encoders. Only meaningful on macOS.
///
/// PULL-AND-SIMPLIFY from `ferrite-metal-core::{context,buffer,pso_cache,
/// fast_metal_ffi,barrier_tracker}`: device/command-buffer/PSO-cache
/// harness + runtime shader compile (`include_str!` the `.metal` sources in
/// `shaders/`, `newLibraryWithSource` on first use, cache PSOs — no Xcode
/// step, pure-source checkout). Keep the substrate; **drop** the
/// route-registry/receipt/override machinery that substrate carries in
/// Ferrite (VM plumbing, unused here).
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metal;

/// Single-token GPU decode driver — muse-fixed, no VM.
///
/// **REIMPLEMENT.** Ferrite's GPU muse decode lives fused into the
/// multi-arch "VM" execution engine (`forward_gpu/engine_decode/*`, ~40
/// files: `vm_forward`, `vm_attn_ops`, `decode_oracle`, `logit_cert`,
/// `kernel_selector`, `route_registry`...) which the catalogue marks **DEAD
/// for muse** — the handwritten path is what passed 32/32 parity. Do not
/// lift the VM. `reference.rs` is the exact spec: 52 layers, per layer
/// {pre-norm → q/k/v/gate proj → per-head QK-norm → (SWA: rope) →
/// flash-attn(window|full) → sigmoid-gate → o_proj →
/// post-attn-norm(1e-8)+resid → ffn-norm → gate/up/silu/down →
/// post-ffw-norm(1e-8)+resid} → final-norm → lm_head → ×1/√26 → softcap 20.
/// This module is that graph with GPU encoders swapped in per op — a
/// transcription against a golden capture, not research.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod decode;

/// Batched GPU prefill driver — muse-fixed, no VM.
///
/// **REIMPLEMENT**, same rationale as `decode.rs`, replacing Ferrite's
/// `forward_gpu/engine_prefill/*`. Weight-row-reuse matmul; see
/// `weights.rs`'s "prefill of T tokens ≈ one token's DRAM traffic" note.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod prefill;
