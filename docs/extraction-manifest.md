# muser — Ferrite→muser Extraction Manifest

Historical provenance ledger for the Ferrite-to-Muser extraction. It records
the original port plan and subsequent extraction receipts; imperative wording
such as "pull", "reimplement", and "leave behind" describes decisions made
during that port, not unfinished work. `docs/muser-architecture.md` describes
the current runtime and `release/feature-contract-v1.json` defines the release
boundary.

Provenance surveyed (read-only, Ferrite private repo):
`ferrite-rs-linked-main-backup-20260810`, workstream
`muse-onboard-20260811` (+ `gqa-headpack`, `retrieval-specdec`,
`muse-relocate-ferrite`), and `MUSE-IMPLEMENTATION-CATALOGUE.md`.

Extracted modules retain source anchors in their doc comments. This file is
the historical index and is not a current implementation-status checklist.

---

## 2. EXTRACTION MANIFEST

Legend: **PULL-CLEAN** (copy near-verbatim; already muse-focused & self-contained) ·
**PULL-AND-SIMPLIFY** (copy, then delete the non-muse arms / hardcode muse) · **REIMPLEMENT**
(too tangled to lift; the muse reference tells you exactly what to write).

### A. The muse forward path — the crown jewels (PULL-CLEAN)

These files on `muse-onboard-20260811` are *already* a clean, self-contained,
muse-only slice. They are the highest-value extraction and carry almost no Ferrite baggage.

| Ferrite source (branch) | → muser | action | note |
|---|---|---|---|
| `crates/ferrite-inference/src/muse/forward.rs` | `muser-engine/src/reference.rs` | **PULL-CLEAN** | CPU f32 oracle; the correctness gate. Keep verbatim. |
| `crates/ferrite-inference/src/muse/config.rs` | `.../config.rs` | **PULL-CLEAN** | `MuseConfig`, layer-kind resolver, `QkNormProbe`, shape assertions. |
| `crates/ferrite-inference/src/muse/weights.rs` | `.../weights.rs` | **PULL-CLEAN** | mmap `TensorView`, `dot_row`, `matmul`, `matmul_rows`. |
| `crates/ferrite-inference/src/muse.rs` | `.../loader.rs` | **PULL-CLEAN** | `load()` + `probe_qk_norms` (fail-closed QK-norm check). |
| `crates/ferrite-inference/src/muse/capture.rs` | `.../capture.rs` | **PULL-CLEAN** | node-named intermediate recorder for parity. |
| `crates/ferrite-inference/tests/muse_golden.rs` | `muser-engine/tests/muse_golden.rs` | **PULL-CLEAN** | **the anchor**: bit-level capture parity vs llama-eval-callback. |
| `crates/ferrite-inference/src/layer_config/tests/muse.rs` | `.../tests/` | **PULL-CLEAN** | 160-line muse geometry fixtures/tests. |

### B. GGUF + quant (PULL-AND-SIMPLIFY)

| Ferrite source | → muser | action | note |
|---|---|---|---|
| `crates/ferrite-inference/src/gguf.rs` + `gguf/{parser,reader,metadata,types,accessors}.rs` | `muser-engine/src/gguf/` | **PULL-CLEAN** | v3 parser; muse loader already uses only this. Drop the `#[cfg(test)] cache.rs` untrusted-sidecar. |
| `crates/ferrite-inference/src/quant/{q4k,q5..,blocks,helpers,dispatch}.rs` + `dequant_block`, `dot_q4_k/q5_k/q6_k/q8/q4_0_f32`, `silu_fast`, `f16_to_f32` | `muser-engine/src/quant/` | **PULL-AND-SIMPLIFY** | Keep only the dtypes muse GGUF uses (Q4_K/Q5_K/Q6_K/Q8_0/Q4_0/F16/F32). **Drop** IQ2/IQ3/MLX/codebook/subspace quant zoo. |

### C. Metal kernels the muse forward needs (mixed)

The **shaders** are portable and small — pull the muse-relevant `.metal` files clean. The
**Rust encode/dispatch layer** in Ferrite is fused into the multi-arch VM engine — reimplement
a thin muse-fixed driver against the shaders (§ D).

| Ferrite shader (`crates/ferrite-metal-shaders/shaders/`) | muse role | action |
|---|---|---|
| `rope.metal` (**branch version** — adds norm-layout `rope_inplace`) | NORM/GPT-J RoPE on SWA layers | **PULL-CLEAN** |
| `sigmoid_gate.metal` | attn-output sigmoid gate | **PULL-CLEAN** |
| `rms_norm_llamacpp.metal`, `fused_residual_rms_norm_llamacpp.metal`, `rms_norm_per_head.metal` | sandwich norms + per-head QK-norm (dual eps) | **PULL-AND-SIMPLIFY** (keep muse eps paths) |
| `flash_attn_ext_vec_llamacpp.metal` (+ `..._f16.metal`, `..._reduce.metal`) | flash decode, split-K, online softmax | **PULL-AND-SIMPLIFY**: specialize DK=DV template **256→128** (muse head_dim), keep the bit-exact llama arithmetic ordering |
| `flash_attn_decode_vec_f16_gqa_lazyrope.metal`, `flash_attn_decode_vec_geop_dense_gqa.metal` | **GQA head-packing** decode (the +23.6% lever) + lazy-rope | **PULL-AND-SIMPLIFY** (the NHPTG=4 Q-packing port; still BUILDING) |
| `flash_attn_prefill_q4_dk64.metal` / prefill variants | Mac-local prefill flash | **PULL-AND-SIMPLIFY** → dk128 |
| `matmul_q4k_*`, `ffn_fused*.metal`, `embed_gather.metal`, `matvec_q6k_llama.metal` | proj/FFN/LM-head matmuls, embed | **PULL-AND-SIMPLIFY** (keep muse dtypes/shapes) |
| `rope_kv.rs` encode (`.../encode_batch/elementwise_ops/rope_kv.rs`, **branch**) | norm-layout RoPE+KV write encoder | **PULL-AND-SIMPLIFY** |

| Metal runtime substrate | → muser | action |
|---|---|---|
| `ferrite-metal-core::{context,buffer,pso_cache,fast_metal_ffi,barrier_tracker}` | `muser-engine/src/metal/` | **PULL-AND-SIMPLIFY** — this is the device/command-buffer/PSO-cache harness + runtime shader compile. Keep the substrate, drop the route-registry/receipt/override machinery. |

### D. The GPU forward driver (REIMPLEMENT — the deliberate rebuild)

Ferrite's GPU muse forward lives in `forward_gpu/engine_prefill/*` and `forward_gpu/
engine_decode/*` (≈80 files) and is **fused into the multi-arch "VM" execution engine**
(`vm_forward`, `vm_attn_ops`, `vm_planning`, kernel_selector, route_receipts, koopman/
geoprecision/grassmann shadows, oracle/certificate scaffolding…). Per the catalogue the
**VM exec path is DEAD for muse** (handwritten path is what passed 32/32 parity).

→ **Do not lift the VM.** Reimplement two small, straight-line, muse-fixed drivers:
- `decode.rs` — single-token decode: 52 layers, per layer emit {pre-norm → q/k/v/gate proj →
  per-head QK-norm → (SWA:rope) → flash-attn(window|full) → sigmoid-gate → o_proj →
  post-attn-norm(1e-8)+resid → ffn-norm → gate/up/silu/down → post-ffw-norm(1e-8)+resid} →
  final-norm → lm_head → ×1/√26 → softcap 20. **The reference `forward.rs` is the exact
  spec** — decode is that graph with GPU encoders swapped in per op.
- `prefill.rs` — the batched form (weight-row-reuse matmul; the reference `matmul` already
  documents "prefill of T tokens ≈ one token's DRAM traffic").

Everything the driver must do is already pinned, op-named, and eps-tagged in the reference +
`capture.rs`. This is a *transcription against a golden capture*, not research.

| Ferrite source | → muser | action |
|---|---|---|
| `forward_gpu/engine_decode/*` (VM decode) | `muser-engine/src/decode.rs` | **REIMPLEMENT** (muse-fixed, no VM) |
| `forward_gpu/engine_prefill/*` (VM prefill) | `muser-engine/src/prefill.rs` | **REIMPLEMENT** |
| `forward_gpu/kvpack_bridge/*` | `muser-kvpack` glue | **PULL-AND-SIMPLIFY** (keep save/restore API, drop remote-VM coupling) |

### E. KV memory (REIMPLEMENT — clean two-class allocator)

Ferrite's live KV owner is `kv::BucketArena` via `forward_gpu/arena.rs`; `paged_kv.rs` is
explicitly *legacy telemetry, not the runtime owner*. The **ring modulus for the 39 SWA
layers is NOT wired** (OOB hazard past 2048; catalogue VALIDATE). muser wants a **purpose-
built two-class allocator** matching muse exactly:
- **13 NoPE full planes**: grow-only, position-free → `memcpy`-relocatable (kvpack export
  ships these as-is; stream during prefill).
- **39 SWA windowed planes**: true ring modulo `sliding_window=2048`, wraparound correct
  (this is the piece Ferrite left stubbed; muser gets it right from day one).

| Ferrite source | → muser | action |
|---|---|---|
| `crates/ferrite-inference/src/kv/{block_arena,global_arena,eviction_manager}.rs` | superseded: shipping KV owner is `muser-engine/src/decode.rs` (`MetalKvPlane` + `MetalSpeculativeCheckpoint`); the pulled `kv/` arena modules were deleted 2026-08-13 as dead code | **PULLED, THEN RETIRED** |
| `paged_kv.rs` | — | **LEAVE BEHIND** (legacy telemetry seam) |

### F. Tokenizer + chat template (PULL-AND-SIMPLIFY)

| Ferrite source | → muser | action |
|---|---|---|
| `crates/ferrite-inference/src/tokenizer.rs` + GGUF vocab accessors | `muser-engine/src/tokenizer.rs` | **PULL-AND-SIMPLIFY** — GGUF BPE, dual-EOS (`eos`+`eot`), muse chat template only |

### G. Server (PULL-AND-SIMPLIFY, aggressively)

Ferrite's `ferrite-server` is large (cascade/speculation routers, constrained decode,
matryoshka, spark-prefill protocols, model-manager multi-model slots…). muser serves **one
model**. Keep the HTTP/OpenAI scaffold; drop the rest.

| Ferrite source | → muser | action |
|---|---|---|
| `ferrite-server/src/{api/,openai.rs,repl.rs}` | `muser-server/src/{api/,openai.rs}` | **PULL-AND-SIMPLIFY** (OpenAI `/v1/chat/completions` + native `/generate`, SSE stream) |
| `ferrite-server/src/api/session.rs`, `stages/session.rs` | `muser-server/src/session.rs` | **PULL-AND-SIMPLIFY** (kvpack save/load endpoints — the differentiator; llama.cpp ships this **DISABLED** for this arch) |
| `ferrite-server/src/state/sampling.rs` | `muser-server/src/sampling.rs` | **PULL-CLEAN** |
| model-manager / multi-model slots / cascade / speculation-router / spark-prefill-protocol | — | **LEAVE BEHIND** (single-model server; no VM) |
| `main/spark_prefill*` (GX10↔Mac disagg protocol) | `muser-cluster/src/producer.rs` | **PULL-AND-SIMPLIFY** (the disagg wire) |

### Sharp edges to LEAVE BEHIND (do not extract)

- **The whole `forward_gpu` VM** (`vm_*`, `engine_*`, `kernel_selector`, `route_registry/
  receipt`, `koopman*`, `geoprecision/grassmann/gauge_structure` shadows, `decode_oracle`,
  `logit_cert`, `frozen_replay`) — muse never uses it; it's the multi-arch interpreter.
- **Multi-arch dispatch everywhere** — hardcode muse. No `layer_config` arch-switch, no
  arch-generic shard/component machinery, no Gemma/Qwen/Kimi/Deepseek branches.
- **The DEAD registry** (catalogue §DEAD): sub-Q4 weight codecs, sub-1.4× KV codecs, far-field
  attn approximations, gate contextual-sparsity, query-group rank-sharing, parallel/Jacobi
  decode, Grassmannian/subspace KV. Proven not to help muse.
- **IQ/MLX/codebook quant zoo**, `expert_*`/MoE (muse is dense), `conv1d`/`kda`/`gdn`
  (linear-attn) shaders, vision tower (text demo), `subspace_cache`, `constrained_decode`/
  json-constraint (not on the demo path).
- **kvpack legacy crates** and the `#[cfg(test)]` gguf untrusted-offset sidecar.

### Stage 2 shader extraction receipt

At initial extraction, the following files were byte-for-byte copies from
`a85048a90fd448585beb9c1b14a819e54a4f16f9` under
`crates/ferrite-metal-shaders/shaders/`. They live under
`crates/muser-engine/src/shaders/ferrite/`; the standalone driver binds only
the Muse routes and does not import Ferrite's runtime or route registry.

| file | SHA-256 |
|---|---|
| `_q4k_helpers.metal` | `83aa9907818c7bc4a4d7dac35b4e365309ee4a716c654c41db7d5201f24a71bd` |
| `ffn_fused.metal` | `f641ffa022f741b5060048e3d321a09905c95e2d8fce9e066a70af26391246d2` |
| `ffn_fused_normed_quant.metal` | `7d4de7d13bcb1040eafb7eb76543e35e704ab224d8e1ba69a9754b16f05c2ea0` |
| `ffn_fused_q4k_hidden.metal` | `852eeccd6f8f07214b0113b3874f178cec21619cb1ad36547325bf56120de4e5` |
| `ffn_fused_tail.metal` | `2c3b5817e0740cdd0566a90b47c1a128f9bd08f02c0117f4aa53f4b562a65547` |
| `flash_attn_decode_vec.metal` | `e53102e0e6221620c7be8c8a8710784c1a8dda20766e1d5b3189d919d92636a8` |
| `flash_attn_decode_vec_contiguous_f16.metal` | `f4defc60993cc2285ca6d86acdcf2e4643fe306dcc7df3374748095424a149ab` |
| `flash_attn_prefill_q4.metal` | `efddb38a54176c4b7e7ff3c6af82f7e1fcbeb65f0bb54a9386d8682d1ffa5483` |
| `fused_residual_rms_norm_llamacpp.metal` | `80da0a23fc407940e2ff47713fde7d9fe6f52a083ec88a2b964922b142cd070d` |
| `matmul.metal` | `785ad6efc9e81e98fe33498c2da1522154c02103507259cb35fca5f1480608f9` |
| `rms_norm_llamacpp.metal` | `76baa3a5c9dbb881e78aaee966d55d769b98fdeddcd3ffc813a77256ca4f37fd` |
| `rms_norm_per_head.metal` | `897a82d7d1c4ad585de55dbb4c84fcb2aa91b581edbee975a59931e5290717d9` |
| `rmsnorm_batch_tail.metal` | `a1a8f2aab1463f533326bd89a65d335a2702f487d7a7b0c8653caf98301202e8` |
| `rope.metal` | `fdcce735f90037f3f5d1fe6761bfb761de7d43fcded563efc1afff882556306f` |
| `sigmoid_gate.metal` | `f57638e56abed09194f4d13a9e3bb5c58ddad258fa6e0f1a4ad1ada26fcae481` |

The parity-restoration pass additionally pulled two accepted prefill
primitives from the same Ferrite revision:

| Muser file | Ferrite source | source SHA-256 | Muser SHA-256 | adaptation |
|---|---|---|---|---|
| `flash_attn_v2.metal` | `flash_attn_v2.metal` | `4f04e91934e3bcb21fadc40114b7687e70c0b90944ee639a121a9b1d0b1d706c` | `44fc4900b7954c38bfee77e4e64fec4042202a8a9c2a7f4d0046de40cbdb9061` | Adds token-major/head-major selection, a logical cache base for compact SWA tails, and per-query exact window masks; arithmetic and DK128 FA2 tiling remain Ferrite's. |
| `batch_sgm_q4_aligned.metal` | helpers from `batch_q5_sgm.metal` plus the aligned kernel from `batch_sgm_q4.metal` | `batch_sgm_q4.metal`: `227a48e02fbb532b496ecb67d07c85911458dd86fafc807586b4ab1b74004a4e` | `cd1b6fd5a24d1c428133b670a3aadb60ea180a9f966644b6fac4facfc1126287` | Keeps only shared Q4_K helpers and `matmul_q4k_batch_sgm_aligned`; no QKVG or FFN consolidation. |
| `argmax_f32.metal` | `matmul_misc_postprocess.metal` kernels `argmax_f32_phase1/phase2` from the accepted GPU-greedy lineage (`be6ea89a4`, split through `56be14fa0`) | extracted functions | `30950ca9b2afb0b6902af271095937ba2d6267b65e7d6d29e3e6668444d2f61b` | Retains only the exact two-phase first-maximum reduction; Muser dispatches it once per DFlash row so greedy drafting returns 16 IDs instead of copying 16 full vocabulary rows. |

`rmsnorm_batch_tail.metal` is the one subsequently adapted in Muser: its
current SHA-256 is
`24379a087307ec732f09feb2875dd7d90f7246c0202a82e476fcd04197cdba16`.
The adaptation restores Ferrite's accepted four-SIMD-group/vectorized ordinary
norm geometry and its 32-SIMD-group 6,656-wide fused-tail geometry, then adds a
Muse-specific dual-epsilon tail that performs
`post_norm(1e-8)+residual` followed by the next `pre_norm(1e-5)` without an
intermediate dispatch. The corrected 1,024-thread form reproduces the pinned
llama.cpp f32x4 norm primitive's 32-SIMD-group reductions, `1/sqrt` scaling
and expression order, and forces the same f32 device-memory publication/reread
boundary between the two operations. It removes both intermediate dispatches
at every layer boundary without changing the retained full-logit digest;
`MUSER_NO_FUSED_PREFILL_DUAL_NORM=1` retains the split diagnostic control.
The original extraction hash remains above as the source witness.

The straight-line driver adaptations retain explicit source anchors:
`6c9630146` (Muse graph/routing), `45e5d6720` (NORM RoPE and weightless
entry norm), `ac2c9c55e` (dependency ordering), `8cb734ee1` (one serial decode
command buffer), `897a6256b` (Q4_K SiLU gate+up 4r2s), and `a85048a90`
(cache-interleaved F16 split-K). The ring-address translation is intentionally
Muser-owned because ordinary Ferrite cache indexing used absolute positions;
Muser always resolves logical rows through explicit origin metadata.

The batched prefill driver uses the same accepted serial-command dependency
shape as decode: one command buffer and one compute encoder per 512-token
chunk, with tracked shared resources and retained activation/token/capture
arenas. This replaces the initial correctness implementation's per-operator
encoder creation without changing the Muse graph or cache placement.

After the 2,048-token SWA wrap, the driver materializes the explicit logical
ring followed by the current chunk into a detached 2,560-row F16 staging tail,
runs the same accepted Ferrite FA2 tile on that tail, and only then commits the
live ring. The gather kernel is Muser-owned (`muse_reference.metal` SHA-256
`cd57e8aaf9650d9dfdb5c683ebf4f7d535b387d57acd4fe57fdf1eb6c77a5090`);
it never derives physical placement from an absolute token position.

The release substrate also retains Ferrite's public `MTLResidencySet` owner
and default-untracked Metal allocation policy. The immutable mmap GGUF arena
is attached to the session command queue for its lifetime; complete-token and
complete-chunk serial encoders own ordinary activation dependencies, while
split-K producer/reducer transitions retain explicit resource barriers. Token
ID staging is retained rather than allocating a Metal buffer per decode step.

The standalone decode route now also carries the accepted production
scheduling details from the read-only Ferrite expansion: a concurrent encoder,
explicit graph-boundary resource barriers, concurrent Q/K/V/gate and FFN
gate/up projection intervals, fast-math compilation (matching Ferrite and
llama.cpp), four-SIMD-group ordinary norms, 32-SIMD-group fused hidden tails,
and fused entry/final graph ownership.
Q4_K/Q5_K/Q6_K projection math is dispatched through the pinned upstream
llama.cpp metallib where available; the official target's Q6_K tensors fail
closed at model load when `MUSER_GGML_METALLIB` is absent instead of panicking
mid-token.

The qualification-only teacher-forced sink exactly matches the pinned llama
fixture: it executes output norm and the complete LM head but performs no
sampling, argmax, or full-vocabulary CPU readback in the timed region. Greedy
tokens and full logits remain correctness-lane outputs, never throughput-lane
work.

Dirty Ferrite QKVG and FFN dispatch-consolidation experiments are intentionally
not imported. Their retained 2026-08-11 A/B records regressed the valid control
(roughly 29.0--29.6 ms/token), and the release campaign excludes both routes.

### Stage 3 comparator and evidence extraction receipt

The qualification apparatus ports Ferrite's reviewed comparison contracts,
not its runtime. `scripts/evaluate_logits.py` reduces the row alignment and
floor-aware rank logic from `scripts/qwen25_logit_parity.py` at
`51ad7e7ef1540fe3051520a47cac7be104f0aca7`. The exact pre-quantization
top-two/target-NLL producer in `scripts/llama_bench_fixture.patch` comes from
`scripts/bench_vs_llama/comparator/llama-bench-fixtures.patch` at
`17f9e96c804e5d54e25954a3731d051cd7897d79`; the strict sibling validator in
`scripts/llama_perplexity_evidence.py` comes from
`scripts/bench_vs_llama/comparator/llama_perplexity_evidence.py` at
`58ff91891bce10a14356ebd129f980d01c019a67`. Names and schemas are changed to
Muser. Each comparator bundle accepts only the exact fresh llama.cpp revision
named at build time, embeds that 40-hex revision and the tracked patch digest,
and authenticates the patched benchmark, perplexity, and server binaries in a
v3 source receipt.

The campaign ordering, append-only packet rules, alternating A/B runs,
fingerprint checks, and no-retry accelerator discipline follow the canonical
Ferrite runbook at `3a84665393b65320edf7d240d525ef683e1f7656`.
No Ferrite crate, process, or library is linked into Muser or the comparator.

Muse vision acceleration is the one narrow native extraction from the pinned
upstream comparator source rather than Ferrite: `native/mtmd/` exposes only
raw RGB to projected decoder embeddings from llama.cpp's official 50-block
`mtmd` Muse graph. `scripts/build_mtmd_bridge.sh` requires the exact fresh
qualification revision, rejects modified mtmd inputs, packages the complete
vision-only dynamic-library closure, and emits source/binary hashes in a v2
receipt. Muser retains tokenization, text inference, KV state, scheduling,
verification, and product serving; no Ferrite runtime or llama text engine is
linked. The release apparatus launches the
receipt-authenticated llama-server only as an isolated external comparator for
TTFT and native DFlash measurements.

### DFlash accelerator extraction receipt

The five-layer Metal forward in `metal/dflash.rs` is the Muse-fixed transplant
of Ferrite's complete accepted DFlash GPU lineage, including target hidden
conditioning, dual-context attention, retained activation arenas, GPU-resident
sink/window caches, one-command-buffer layer execution, device-output handoff,
and bounded verification (`f332877600`, `84e4a8018`, `d26c51434`, `3063a4762`,
`8e2b9cc4f`, `60f76e63f`). Long first batches grow the arena on demand and
compact their CPU shadow directly to sink+window instead of allocating ten
full-prompt planes. The official GGUF route retains the mmap'd Q4_K/Q6_K
projection bytes and dispatches Muser's extracted quantized kernels directly;
it loads only norm vectors as f32 and never expands the 1.5 GiB assistant into
an f32 production weight set.
The target LM-head hot path follows Ferrite's production
`DFlashGpuLmHeadProjector` lineage (`0b2f3d144`): Muser reuses its own mapped
Q4_K/Q5_K output weight and retained session scratch, so speculative rounds do
not fall through to a CPU output projection. Greedy drafting additionally uses
Ferrite's two-phase GPU argmax lineage (`be6ea89a4`, `56be14fa0`) and transfers
only one token ID per assistant row; sampled drafting retains full logits for
the exact proposal density. The CPU projection remains the oracle only.
The retained experimental/post-release public-CoreML ANE artifacts include the
variable-length target conditioning `fc` matrix plus all 35 layer projections
as INT8 1x1-convolution shards; target output projection remains on Muser Metal
and every token is still committed only by the full target verifier. These
research artifacts are not v0.1 identity, candidate, qualification, or `auto`
route inputs.
