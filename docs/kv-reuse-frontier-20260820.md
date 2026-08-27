# Beyond Prefix Caching: the KV-Reuse Frontier

Research report, 2026-08-20/21. Seven passes: four literature sweeps (non-prefix systems,
cross-model transfer, cache theory/math, positional algebra), one cross-disciplinary ideation
pass, one original theory pass, one adversarial review pass (corrections applied; engine facts
verified against the muser codebase). Evidence tags: **[EXACT]** provable identity;
**[MEAS]** measured, scope noted; **[HYP]** ours to test. arXiv IDs inline; index at the end.

Scope note: the analysis is cross-family. kvpack targets many model families; Muse
(= the published RNoPE-SWA layout, Cohere Command A, 2501.18795) is one worked row of the
taxonomy, at the easy corner of it.

---

## 0. TL;DR — the code, cracked as far as it cracks today

1. **RoPE's coordinate is solved; positional *semantics* is not.** Re-anchoring a cached key is
   one exact rotation (group action, production-deployed for years: llama.cpp K-shift,
   FlashInfer pre-RoPE caches). But position also leaks into the *values* via the causal mask
   (implicit position), and that part is unmeasured — it is barrier #5 below, not a RoPE
   problem. Stop thinking about rotations; start thinking about what the values remember.
2. **Reads compose exactly; representations don't — and only exact reads are free.** Softmax
   attention over concatenated caches decomposes exactly into per-segment (max-logit, exp-sum,
   partial-output) triples. But Q is never cached: evaluating cross-segment coupling above the
   first mixing layer requires the conditioned forward pass you were avoiding. The honest
   accounting: **exact composition of B onto A costs a full prefill of B in A's context — you
   save A's prefill and nothing else. Every non-prefix composition of representations is
   approximate by construction**, and the science is how much approximation the acceptance gate
   admits.
3. **Contextualization is concentrated and largely repairable — but not the way the famous
   papers say.** "15% token recompute suffices" (CacheBlend) holds only on loosely coupled RAG
   QA; multi-key retrieval needs 70–90% (CacheClip); on cross-chunk binding, token recompute
   caps at 36% recovery while a rank-64 feature-space patch recovers 97% (Kamera — and our own
   algebra *predicts* that shape, §3). Ranked, replicated barriers: missing cross-segment
   conditioning > duplicate attention sinks > softmax calibration > magnitude drift (does not
   even exist under QK-norm architectures like Muse) > implicit position (never isolated).
4. **The economics sell time, not bytes — and the right metric is break-even reuse count.** A
   cache is ~3,000× larger than the text it memoizes, and its information content is the text's
   (the decoder of that "compression" is the model itself — no free headroom there). Restore
   beats *local decode-box* prefill by large factors, but against an overnight GX10 prefill the
   honest crossover is ~20–40×, and TTFT on composed contexts is dominated by the SWA warm-up,
   not I/O. What justifies a 100 TB shipped library is **cold-start insurance for air-gapped
   deployment** — you can't know tomorrow's access distribution, so you ship everything — plus
   Zipf-head serving where reuse counts are high (production data: top 5% of chunks serve 60%
   of requests; 79% are never re-accessed — Cache-Craft).
5. **The story is cross-family; each architecture sits at a different point on one map.**
   Composition = exact per-family mechanics (rotation group for dense RoPE, re-indexing for
   ALiBi, latent append for MLA, scan composition for linear/SSM — with a caveat, §2.5,
   multiset union for NoPE planes) + the shared contextualization obstacle. Repair machinery
   and gates are built once. Nearly all published repair evidence comes from dense-RoPE models
   — the common case. Muse sits at the easy corner: its growing cache is 13 position-free NoPE
   planes, so composition concentrates in 13 of 52 layers — which also carry the heaviest sinks
   (0.33 mass) and do the retrieval. MLA is nearly as privileged. Gated-hybrid SSM is the hard
   corner: there, contextualization contaminates the *mechanics* too.
6. **The appliance is byte-feasible today; capacity varies ~25× by family.** 100 TB holds
   ~780M tokens of Llama-8B cache, ~1.4–3B of MLA, ~7.5B of Muse long-range cache at f16 —
   ~28B at 4-bit, i.e. several English Wikipedias *in the 4-bit branch* (at f16, roughly one).
   The open questions are hit rate and composition fidelity, not capacity.

---

## 1. The vision, quantified honestly

Target: overnight batch prefill; morning jobs run faster; ship a primed appliance (iodyne Pro
Data class, ~100 TB) to a ship, a station, an air-gapped site.

**What a cached token costs [EXACT arithmetic].** Muse long-range (NoPE) cache: 13 layers ×
1,024 B = 13,312 B/token — a ~3,000× amplification over ~4.3 B/token of UTF-8 (~6,000× over
17.6-bit token ids at vocab 202,048). Dense Llama-3-8B: ~128 KB/token; 70B: ~320 KB/token; MLA:
~2–4 KB/token. Muse's advantage decomposes as ~4× from GQA-2 and ~4× from NoPE-only growth —
state both factors or reviewers will call double-counting. The bound "cache entropy ≤ text
entropy" (2604.15356) is true and useless here: its decoder is the model. **The appliance sells
time (and joules), not bytes.**

**Break-even is a reuse count, not a bandwidth.** Every library token is prefilled once; you
recover its cost only on reuse. Per-request restore-vs-prefill on the *decode box* favors
restore by orders of magnitude (2P ≈ 60 GFLOP/token vs a 13 KB read), but the honest fleet
comparison is against the overnight GX10 producer at 5–20k tok/s: crossover ~20–40×. And
per-request TTFT on a composed context is dominated by the SWA warm-up (~5–9 s at W = 2048 on
M3 Ultra-class compute), not by the restore (~0.6 s for a full 131k NoPE image at 3 GB/s). The
binding constraints are reuse rate and warm-up, never bandwidth.

**Capacity and I/O, with assumptions labeled.** 100 TB / 13,312 B ≈ 7.5B tokens f16; ≈ 28B at
KIVI-4-bit (kvpack carries the codec with honest error bounds). English Wikipedia ≈ 6B tokens
≈ 80 TB f16 — marginal at f16, comfortable at 4-bit. 3 GB/s is an *assumption* (TB4 single-port
tops at ~2.6–2.8 GB/s; multi-path or TB5 needed), and kvpack's integrity model means every
restored byte is MAC-verified — the number that matters is **verified end-to-end restore
throughput, measured** (E6). Energy [HYP]: prefill ~tens of mJ/token vs NVMe read ~tens of
µJ/token — a plausible 10³–10⁴× that one wall-plug measurement converts from `mock` to measured
in muser's economics (`joules_saved` currently has no writer).

**Hit rate is a policy problem with real data.** Production trace (Cache-Craft, 2502.15734):
exact-prefix hits cover 8% of requests / 18% of tokens; top 5% of chunks appear in 60% of
requests; 79% never re-accessed in a month. Fit Zipf, size with Che's approximation, and define
"90% primed" as **fraction of prefill FLOPs avoided** — never request rate. The Zipf head fits
in far less than 100 TB; the full-library size is justified by air-gapped cold-start insurance,
and should be argued as such.

---

## 2. The algebra: what is exactly true (with its edges)

1. **Log-sum-exp merge monoid [EXACT].** For any partition of the cache, softmax attention is
   the associative merge of per-segment triples (m, l, o) — flash-decoding / Hydragen
   (2402.05099). The merged output obeys o′ = (1−β)·o_S + β·ō_ext with β = σ(ℓ_ext − ℓ_S):
   this identity is *exact per layer*, not first-order. Edges: (a) *read-time deletion* of a
   segment is trivial (don't read its rows) — subtracting from a stored aggregate is not
   generally possible (the decode query isn't known at store time) and is catastrophic
   cancellation in f16 when the deleted mass dominates; (b) FlashAttention computes each
   segment's per-(token, head) LSE at prefill and discards it — **store it** (64 B/layer-token:
   6% of f16 KV, 25% of 4-bit — store as offset-bf16 or f32, since raw f16 error ~0.03 on
   logits reaching ~55 corrupts the merge weight); (c) using it needs the *other* side's
   ℓ_ext, which needs B's queries — cheap at the first mixing layer, conditioned-stack-priced
   above it (see §3's accounting). The sidecar is genuinely free at prefill and genuinely
   decisive at the first NoPE layer; above that it is an estimator, not a bound.
2. **RoPE group action [EXACT].** R(Δ) re-anchors a cached key exactly; one hop from a
   canonical anchor, trig in f32, never chained. Two pairing conventions (NeoX vs GPT-J) —
   wrong-convention rotation is the classic silent-garbage bug (cacheweaver's canary exists for
   it). kvpack's `canonical-kv-prerope-v2` (pre-RoPE f32 keys, consumer rotates once in its
   pinned kernel) is the FlashInfer-style endgame: natively position-free caches.
3. **Frequency surgery [EXACT transformation; semantically approximate; novelty unverified].**
   PI/NTK/YaRN/LongRoPE rotate in the same planes, so a cache prefilled under config A rewrites
   to config B by one composite rotation per plane. Caveats: YaRN's logit temperature is not a
   rotation; the rewritten K is self-consistent with B but the V and hidden states were built
   under A's attention pattern — end-to-end quality unmeasured; and llama.cpp's K-shift with
   changed frequency parameters arguably makes this folklore. Cheap experiment, modest paper.
4. **NoPE multiset structure — Prop 1′ [EXACT, tightly scoped].** At a NoPE layer with no
   positional logit term, no length-dependent temperature, and the mask given by set
   membership, the attention output depends on the visible rows only as a multiset. Scope:
   (a) mask-preserving permutations only — appending or interleaving *new* tokens has real
   order semantics via the intra-batch causal mask; (b) exact in exact arithmetic, ~1e-3
   relative in f16 online softmax (row order changes the block partition), so a permuted cache
   is RECONCILED, never EXACT, under the seal regime; (c) **licenses mechanics, not quality** —
   it says the engine won't misindex, not that the distribution matches a joint prefill.
   The tension to resolve empirically: implicit position lives in the *content* (hidden
   variance decreases with position — 2203.16634, 2404.12224, 2509.21042), so after a splice
   both segments' rows still encode "I begin at 0" — the duplicate-sink pathology replayed in
   the implicit-position channel. If real, the predicted fix is an **affine offset in feature
   space**, not row surgery (E1).
5. **Linear/SSM scan composition [EXACT for LTI only].** S(A·B) = Λ(B)·S(A) + S(B) holds with
   Λ computable from B alone for *fixed-decay* recurrences (S4/S5, RetNet). For the gated
   variants the industry actually ships (GDN, Mamba-2, GLA, RWKV-6/7), Λ(B) = Π diag(α_t) with
   α_t a function of B's *contextualized* inputs — computable from B alone at layer 0 only.
   **For gated hybrids, contextualization contaminates the mechanics term itself**: the
   mechanics/contextualization factorization that holds cleanly for attention degrades here.
   Consistent with the measured caution (naive recurrent-state composition: 46.6% of full
   quality; single-state re-init recovers 86.8% — LinearKV 2608.11231). Engineering notes for
   an executable algebra: ship the conv1d boundary tokens (kernel_width−1), and keep Λ in log
   space (a product of sub-unit decays underflows f16).
6. **Gauge freedom [EXACT] — and QK-norm collapses it to Procrustes.** Per head, logits are
   invariant under K→KA, Q→QA⁻ᵀ (V→VB with W_O absorbing B⁻¹): a cache is defined only up to a
   per-head gauge. **KV-space reconstruction error is not gauge-invariant — which is the exact
   reason it fails to predict transfer quality; attention outputs are gauge-invariant.** With
   per-head QK-RMSNorm (Muse: K-norm weight exactly 1.0, verified in the engine), the QK-side
   gauge group collapses from GL(d) to **O(d)** — so an orthogonal-Procrustes-constrained
   mapper is not a heuristic but the *exact residual symmetry* of the QK side (V side keeps
   GL). Bonus engine fact: with QK-norm and the folded scale, Muse attention logits are exactly
   ≈43.8·cos(q,k), bounded — cosine acceptance gates are dimensionally native and
   calibration-free, and cached K rows have per-head RMS exactly 1 (magnitude drift is
   architecturally impossible here).

Framing result on the headroom: **MLA caches ~2% of MHA-equivalent bytes with no loss — chosen
at training time** (2405.04434). Post-hoc SVD tops out ~2×, and naive rank reduction loses to
quantization everywhere tested (rank-32 → LAMBADA 0.4%, 2604.11501) — §3 derives why.

**The architecture taxonomy** (each family = its exact mechanics + the shared obstacle; the
algebra is per-*layer*, not per-model — Muse's is a 52-vector: 39 × `window_truncate+reanchor`,
13 × `multiset_union`):

| Family | Position action on cached K | Growing planes | Exact composition mechanic | Caveat |
|---|---|---|---|---|
| Dense RoPE (Llama/Qwen) | R(Δ), all layers | all | re-anchor + append | YaRN/NTK models: scale factor is length-dependent; temperature term is not a rotation |
| SWA + global-RoPE (Gemma-2/3) | R(Δ) on global planes | global only | re-anchor global; SWA tail bounded | — |
| SWA + NoPE (Muse / Command A / SmolLM3 / Llama-4-iRoPE) | none on growing planes | 13 NoPE planes | multiset append (Prop 1′ scope) | implicit position in values, unmeasured |
| MLA (DeepSeek) | thin 64-dim k_R channel | latent c_i | latent append; rotate k_R (Irminsul 2605.05696 ships this) | — |
| ALiBi / bias | none (score-time bias) | all | re-indexing — and the inter-segment *gap* is a free per-segment log-prior on attention mass: an exact, explicit β knob (APE's scaling, for free) | ecosystem faded |
| Gated linear/SSM hybrids | n/a (state) | recurrent state | scan compose — **LTI only**; gated Λ needs contextualized inputs | mechanics and contextualization entangled |

Missing from the table's axes on purpose-noted grounds: sink/normalization structure — the #1
and #2 measured barriers — is family-universal, which is why the repair toolbox (§4) is shared.

**kvpack implication:** promote the per-layer valid-transformation vector — `reanchor(Δ) |
permute | window_truncate | scan_compose | reindex` with parameters (pairing convention, freq
base and scaling type, decay parameterization, conv boundary width) — from implicit identity
metadata to an executable `cache_algebra` manifest field. kvpack already carries ~80% as
identity material (`RepresentationFamilyId`, `TokenAxisRule`, `rotation.rs`, `mla-latent`).
One composition engine then serves every family. Validity metadata must hash the full
non-weight surface too: RoPE base/scaling, window pattern and NoPE polarity (muser's config
notes bad converters get `sliding_window_pattern` wrong — and `rope_base_full` is parsed but
unused today), attention scale, softcap, QK-norm params, numeric format.

---

## 3. What actually breaks — with the accounting stated up front

**The exact-composition accounting [EXACT].** Q is never cached. Cross-segment coupling at
mixing layer ℓ needs B's queries at ℓ, i.e. B's hidden states at ℓ, i.e. B's forward pass
conditioned on A up to ℓ. At the first mixing layer that is cheap (for Muse: layers 0–2 are
SWA, so the first cross-segment-relevant layer is NoPE layer 3 — three layers of B ≈ 6% of a
prefill plus one |B|×|A| QK pass). At all higher layers it is the joint prefill you were
avoiding. Hence: exact composition of B onto A costs a full prefill of B in A's context; the
only exactly reusable non-prefix object is a read. All economics run on the approximate path.

**The error object.** Δ = KV(B|A) − KV(B|∅). Layer-1 Δ = 0 exactly. Per layer, the exact merge
identity o′ = (1−β)o_B + β·ō_A(q) splits the damage into a scalar attenuation and an additive
injection, which yields three *derivations* of measured results:

- **Why quantization beats rank reduction [DERIVED].** Through the softmax Jacobian
  (∂p/∂z = diag(p) − ppᵀ): independent per-row errors (quantization) concentrate as
  √(Σp_i²) — small under diffuse attention; errors *shared across rows* (low-rank truncation)
  add coherently as Σp_i = O(1). Exactly the measured dominance (2604.11501).
- **Why the deficit is token-diffuse and feature-low-rank [DERIVED].** The injected term
  β_i·ō_A(q_i) lives in span(V_A) — low-rank in features, spread over tokens. That is Kamera's
  headline finding (rank-64 patch recovers 97% of answer flips; token recompute caps at 36%),
  predicted by the algebra. Family-2 repair is the theoretically favored primitive.
- **Why temperature+scale recovers so much, and where it stops [DERIVED].** A splice has two
  sinks where a joint prefill has one. At Command A's measured 0.33 NoPE sink mass, merged sink
  mass ≈ 0.5–0.66 vs the true 0.33: a near-uniform ~40% attenuation of content-carrying mass at
  every NoPE layer — a scalar error, exactly what APE's temperature+scale removes (98% RAG
  recovery, zero recompute) — while the residual *content* injection is what it cannot remove
  (the collapse to ~34% on multi-value retrieval).

**Barriers ranked by evidence [MEAS]:**
1. Missing cross-segment conditioning — universal 7–18 F1 deficit on multi-hop QA across 11
   systems and 3 models (2603.20218).
2. Duplicate sinks at segment starts — 4 independent groups; sinks are structural (softmax
   simplex necessity 2603.11487; over-mixing control 2504.02732). Fixes that work: one shared
   sink prefix deduplicated at splice; recompute a constant handful of chunk-initial tokens;
   trained header tokens.
3. Softmax calibration — the derived attenuation above; temperature control on full-attention
   layers is mandatory equipment; NoPE layers have no distance decay to protect them when the
   support widens (entropy inflection, 2404.12224).
4. Magnitude drift — measured elsewhere (key norms grow with position, APE); **architecturally
   absent under per-head QK-norm** (Muse: cached K rows have RMS exactly 1).
5. Implicit position in "position-free" layers — robustly probed, never isolated for splicing;
   E1 decides. Prediction registered now: after a splice, both segments' rows probe as "I begin
   at 0," and the fix is a learned affine offset in feature space.

**The recompute-fraction ledger** (quote no number without a task-interdependence qualifier):
~0–2% (self-contained chunks — EPIC regime) · ~15% (loosely coupled RAG QA, F1/Rouge —
CacheBlend regime, never independently replicated) · ~30% (production RAG — Cache-Craft, with a
per-chunk predictor) · 70–100%-or-retrain (aggregation, multi-key, cross-chunk binding —
CacheClip). Non-monotonic hazard: 20–60% ratios can fragment multi-token entities and score
worse than either extreme.

**Framing challenges (build the instrument first):** mean perplexity dilutes splice damage on a
30-token answer by ~10³, and the damage is *selectively relational* (cross-boundary
coreference, compare, contradict, attribute) — no standard benchmark isolates it (E0). A
spliced cache is exactly Fusion-in-Decoder — a known-good architecture — so monolithic prefill
is a chosen reference, not ground truth; test whether spliced-no-repair ever *beats* monolithic
(it eliminates lost-in-the-middle by construction). And every published multi-request PIC eval
runs synthetic streams against unoptimized baselines — the honest bar is scheduled prefix
caching + reordering (97.5% of oracle from reordering alone — 2606.19667, an unrelated system
that happens to be named "CacheWeaver") and muser's own disagg lane (E0b).

---

## 4. The repair toolbox (family-independent by construction)

Every mechanism below operates on the shared obstacle, never on position mechanics. Evidence
direction: families 0–3 were validated on dense-RoPE models — the hardest-mechanics common
case — and transfer toward SWA+NoPE/MLA with the mechanics load removed.

- **Family 0 — distributional (zero recompute):** one shared sink prefix (deduplicated at
  splice), temperature/scale on full-attention layers (§3 derives why this floor is high),
  positions locally exact + globally monotone (the decode-side remapping literature —
  Self-Extend, DCA, ReRoPE — licenses the query lying about distance under exactly those two
  constraints). Floor ~93–98% on retrieval-shaped tasks; collapses on cross-chunk binding.
- **Family 1 — token-sparse recompute:** select by cross-segment attention mass β at the first
  mixing layer (LSE sidecar + 3 layers of B + one QK pass — cheap, §2.1), recompute in ≥8-token
  windows (entity-fragmentation guard). A 135M auxiliary model's last-layer attention is a
  measurably better selector than the big model's own layer 1 (CacheClip) — replicate. Scope
  honestly: β above the first mixing layer is an estimator; and *precomputed pairwise* repair
  artifacts are quadratic in library size — they exist only for a **fixed-A regime** (system
  prompts, agent scaffolds, tool descriptions) or curated bundles.
- **Family 2 — feature-space low-rank patch (the linker), theoretically favored (§3):** the
  injection lives in span(V_A); a precomputed rank-r patch GEMM'd into the cache recovers 97%
  where token recompute caps at 36% (Kamera, multimodal — **text is open**). Each segment
  exports a rank-r sketch of its (K,V) measure; compose = one cheap attention against the
  sketch + patch. Same fixed-A/bundle scoping as Family 1. Breakeven ≈ 9 reuses in Kamera's
  accounting.
- **Family 3 — trained tolerance (the quality ceiling):** fine-tune to accept independently
  encoded blocks (Block-Attention/TurboRAG: parity at −98.7% TTFT; KVLink +4% over best
  training-free; Prompt Choreography for agents). Costs weight access, re-tuning per release,
  and changes the model identity — hence every kvpack cache identity: v0.2+ for muser. Cheap
  entry: LoRA context-dropout splice-robustness tune.

**Acceptance gates (the honesty layer).** Two regimes, because first-order sensitivity vanishes
exactly where routing flips live:
- *Diffuse heads* (p_max < τ): first-order bound in the centered metric — Cov_q(q), not
  E[qqᵀ] (the mean query direction produces uniform logit shifts, which the softmax Jacobian
  annihilates — and value-centered: key error on rows whose value ≈ the attention mean is free).
- *Peaked heads* (p_max ≥ τ — the ~5% retrieval heads, which in RNoPE-class models live in the
  NoPE layers): a **margin criterion** — does the perturbation preserve top-k logit order? On
  QK-normed engines this is native cosine units (Muse: one logit unit = 0.023 cosine).
Gate composed caches on per-layer attention-output agreement over a probe set + final-logit KL;
never on KV-space error, which is not even gauge-invariant (§2.6). This slots into the
cacheweaver invariant — composed caches are **RECONCILED, never EXACT**, admitted on measured
evidence, full-prefill fallback on gate failure — and VeriCache-style lazy verification
(decode speculatively, verify against ground truth computed off the TTFT path) upgrades
RECONCILED to verified-exact after the fact.

---

## 5. Cross-model and synthesized caches: proven vs vibes

**Proven (≥2 groups or deployed):** same-model position-shifted reuse + selective recompute
(CacheBlend lineage); same-architecture fine-tune pairs with layer-selective recompute
(DroidSpeak ~3.1× — and its killer negative: naive full reuse between a base model and its own
instruct tune collapses to base-model quality); exact-by-construction reuse (activated-LoRA:
base KV exactly valid, up to 100× TTFT; PrefillShare/ICaRus: frozen prefill module + specialized
decode, within 1% of full fine-tuning); hidden states as transfer currency (HCache ≥6× restore;
KV exactly reconstructible from the residual stream in pre-norm transformers — 2603.19664, ≤4B,
sliding-window breaks it; note the direction: X→KV is trivial, KV→X is impossible — 256 dims
per layer cannot recover 6656); cross-scale attention-*map* similarity (IAM/SmallKV/
SpecPrefill).

**Promising, single-source:** NVIDIA closed-form mapper (97.6% retention Qwen3 14B→32B; mapper
1–3.4B params / 4–12 GB per direction-pair; 2/6 pairs fail, no a-priori predictor, no code);
C2C fusers (peer-reviewed, toy receivers); Apple KV Prediction (the honest small→large
synthesis number: 61–66% retention at ~2× FLOPs).

**Vibes:** "cross-model transfer works" unqualified; "low-rank structure → low-rank transfer"
(§3 derives the failure); "negligible loss" on F1/Rouge only (judge workloads silently diverge
decisions at stable end-task metrics — 2601.08343).

**Unpublished directions worth owning:** (1) fit mappers in the gauge-invariant/centered metric
instead of MSE — and under QK-norm the QK-side map is *provably* orthogonal, so Procrustes is
derivable, not heuristic; (2) residual-stream transfer — its merit is **self-consistency** (the
target manufactures K/V with its own W_K/W_V, staying on-manifold), *not* cost: per-layer
h-maps are d_model² (≈4.4e7 params/layer at 6656) and infeasible at restore bandwidth, whereas
per-plane KV-space maps are d_kv² (256² — ~3.4 MB total for a Muse-class model) and trivial;
(3) head-function-aware budgets — recompute the ~5% retrieval heads exactly, map the rest;
(4) factored mappers to collapse the 4–12 GB sidecar (a 13 MB adapter already works for the
communication use case — LCF).

**Cache validity under weight updates — the corrected theorem and the loophole.** A cache stays
exactly valid iff the update touches nothing in the **causal cone** of any cached K/V: all
weights strictly below the deepest cached layer, plus W_K, W_V, *and the QK-norm parameters and
qk-scale* at cached layers. Outside the cone: lm_head, final norm, **and the entire post-KV
part of the deepest layer — W_O, MLP, attention gate, post-attention norms**. Consequences:
- **Release policy beats migration research:** restrict fine-tune updates to the top of the
  cone and every shipped cache stays exactly valid — the appliance survives model releases
  with zero migration. (Check embedding/lm_head tying first; tied weights make "lm_head only"
  the empty set. Embedding updates invalidate everything.)
- "Caches survive LoRA" is **exactly false, approximately open**: with GQA, W_K/W_V are a
  fraction of a percent of layer params, so a generic attention LoRA almost surely touches
  W_Q/W_O — in the cone. Whether stale caches pass the gate anyway is an empirical question:
  **run the unmigrated null first** (an afternoon; often kills the need for any mapper).
- Migration mappers, properly costed: they must be O(r·d_kv) per plane and run at ≥ restore
  bandwidth. Per-plane linear maps for a Muse-class model: ~3.4 MB f16 total, <1 TFLOPS at
  restore rate — feasible; per-layer residual maps are not. NVIDIA's mappers are huge because
  they cross *different* representation geometries; consecutive checkpoints of one model
  should need Procrustes-on-K + linear-on-V (8k params/head). Library migration economics:
  rewriting 60 TB ≈ 11 h at 3 GB/s vs ~250 h to re-prefill — ~20×, not 10³.

**The third cost model — trained caches:** Cartridges (2506.06266): train a per-corpus KV
"cartridge" offline by self-distillation — ICL-parity at 38.6× less KV memory. Overnight
priming as *training*. The right tool for the truly cold tier; composable with everything
above.

---

## 6. Worked programs per family (kvpack is the substrate; Muse is one row)

One composition engine: read `cache_algebra`, apply exact mechanics, run §4, gate on evidence.

**Dense RoPE (Llama/Qwen — the common case; all the repair literature lives here).** Store
pre-RoPE or position-0-anchored (kvpack `canonical-kv-prerope-v2`; FlashInfer proves fused
rotate-at-read is free) → natively position-free caches. Composition = re-anchor + shared sink
+ boundary repair + temperature. All planes grow: 100 TB ≈ 780M tokens (8B) / ~310M (70B) — the
appliance holds a hot working set, not a corpus; Cartridge/cold tiers matter most here.
Frequency surgery (§2.3) = "your cache survives the context-window upgrade," pending the
end-to-end measurement and a novelty check.

**SWA + global-RoPE (Gemma-2/3 class).** As dense RoPE, but only global planes grow — kvpack's
v2 layout tables (`window_tokens`) already express the split.

**MLA (DeepSeek class).** Latent is position-free; only the 64-dim k_R channel rotates.
Muse-grade easy composition at the best capacity-per-token in the zoo (~2–4 KB/token). Irminsul
(2605.05696) already ships non-prefix MLA reuse. kvpack's `mla-latent` class is the container.

**Gated linear/SSM hybrids (GDN + attention).** Honest v1: cache attention planes per the rules
above; checkpoint recurrent state at exact-prefix cuts only; treat cross-segment state
composition as open research (§2.5 — the factorization itself degrades; LTI-only exactness;
log-space decays; conv boundary tokens).

**ALiBi.** Position-free bytes plus an explicit exact β knob (the inter-segment gap as a
per-segment log-prior). No ecosystem demand; the row proves the abstraction.

### The Muse program (RNoPE-SWA row, worked in full)

Structural claims, adversarially corrected:
- **A′ [EXACT — the storage claim]:** no SWA row older than W = 2048 is ever *read* at decode.
  Stored long-range cache = the 13 NoPE planes (13.3 KB/token); SWA storage is bounded at
  39·2048·1024 B ≈ 82 MB regardless of context. Corroborated at the attention level: needle
  mass concentrates in NoPE layers; SWA near-zero at range (Command A analysis).
- **A″ — retired.** "Long-range influence flows exclusively through NoPE planes" is false as an
  influence claim: information injected at a NoPE layer propagates into subsequent tokens' SWA
  rows; the composed SWA receptive field is 39·W ≈ 80k tokens. Correct statement: **Δ is
  injected only at NoPE layers but carried by all layers.** SWA rows are discardable because
  they are never read at range (the mask), not because they are context-free.
- **Free win, previously unclaimed:** at 131k context, NoPE-only decode reads 1.74 GB of KV per
  token vs 6.9 GB for a full 52-plane cache — RNoPE is a **~4× decode-bandwidth win at range**,
  on top of the storage win.
- **The SWA warm-up, costed honestly [corrected — this changes the engineering priority].**
  Exact tail reconstruction requires re-prefilling min(N, 39·W) ≈ 80k tokens — ~72% of a full
  131k prefill. The *approximate* W = 2048 warm-up costs ~1.9% of a prefill ≈ 5–9 s on M3
  Ultra-class compute — which **dominates the ~0.6 s restore by ~10×**; complexity is
  O(W·d²·L) + O(W·N·d_kv·L_NoPE): effectively O(W) only below N ≈ 280k (inside Muse's cap, so
  the claim survives in weakened form). And the reason to recompute the tail is **validity**
  (an isolated segment's tail is wrong for any composition where it isn't first), not the
  82 MB: storing a valid tail is strictly better whenever one exists. Product consequence:
  **memoize compositions, not documents** — 82 MB per hot bundle is nothing (materialized-view
  selection under a byte budget is submodular; lazy greedy gives the (1−1/e) guarantee).
- Composition mechanics: concatenate NoPE multisets (Prop 1′ scope) + one shared sink +
  temperature on the 13 layers + β-guided boundary repair at layer 3 + warm-up per above.
  Repair effort concentrates in 13 layers; E0c tests whether it concentrates further in the
  *early* NoPE layers (residual-norm growth predicts ‖Δ‖/‖x‖ falls with depth).

**Appliance design sketch (favorable branches, each labeled):** NoPE-only packs [A′, EXACT] ·
4-bit KIVI [MEAS] with dithering [HYP — falsifiable signature: the advantage should grow with
context length] · Cov_q-shaped bit allocation [HYP] — do **not** stack rank projection on
quantization (§3 derives why; the "~1.5 KB/token → ~65B tokens" endpoint is [HYP], not a design
point) · content-defined-chunk Merkle keying for dedup beyond prefixes [HYP — dies if E4's
blast radius is unbounded, and at NoPE layers a *flat* raw-ΔKV profile is expected; measure
attention-output-weighted deltas] · LSE sidecar [EXACT at first NoPE layer] · three tiers: exact
hot / multipole-summary warm (random-feature moments per cold doc; error multiplied by the
small external mass) / Cartridge cold. kvpack is the identity substrate; cacheweaver's
provenance regime is the admission law; `cache_algebra` (§2) makes the engine model-generic.

**What this gives the launch story:** kvpack as the composition substrate — exact per-family
mechanics, measured-evidence admission, RECONCILED provenance, lazy verification to exact, and
a release policy (top-of-cone fine-tuning) under which shipped caches survive model updates.
Nobody ships that.

---

## 7. The claim ledger (widely cited, nobody proved)

1. "15% recompute recovers full quality" — regime-bound; 70–90% on RULER multi-key;
   non-monotonic entity corruption between; never independently replicated even at home.
2. "CAG replaces RAG" — toy KBs, BERTScore, loses its own largest-KB row.
3. "LLMs tolerate discontinuous positions without quality loss" — true for weakly interacting
   modules; the repair literature exists because it is false in general.
4. "98% of sequential performance" (APE) — collapses to ~34% on multi-value retrieval; §3
   derives both the success and the collapse.
5. "Chunk-start sinks are THE problem" (EPIC) — partial; contradicted by the diffuse-deficit
   measurement (Kamera); no adjudicating text experiment exists.
6. "500× prompt compression" — at 62–73% retained QA, a number rarely quoted with it.
7. All TTFT speedups measured against unoptimized prefill — never against scheduled prefix
   caching + chunked prefill, never against a disagg lane (E0b is our answer).
8. "Caches survive LoRA" — exactly false (causal cone); approximately open (run the null).

Ours, pre-registered as falsifiable (§9): β-repair needing >40% recompute on multi-key kills
Family 1; an uncorrectable implicit-position collision kills NoPE splicing; scheduled prefix
caching meeting TTFT targets at realistic sizes kills the *latency* product (leaving the
insurance product).

---

## 8. Frontier bets, ranked (post-review)

All bets except #2 are family-generic.

1. **The null baseline + relational diagnostic (E0/E0b/E0c).** Cheapest possible falsifiers of
   the entire program, and the instrument every other result reports into. A program that
   skips these manufactures its own evidence.
2. **The RNoPE-SWA splice ablation** (shared sink + boundary recompute + temperature + affine
   offset arm), scored on the relational diagnostic. Both surveyed open holes point here; we
   own the ideal engine; Prop 1′ + the sink-attenuation derivation give it a theory to test.
3. **Feature-space low-rank repair on text** (Kamera replication + the span(V_A) prediction).
   If it transfers, the token-recompute debate dissolves and the repair artifact becomes a
   shippable pack section — scoped to fixed-A/bundle regimes.
4. **Metric-aware everything** — Cov_q-shaped quantization/projection, margin gates on peaked
   heads, Procrustes-constrained mappers (exact under QK-norm). One calibration pass; touches
   storage, transfer, and admission at once.
5. **LSE sidecar** — free at prefill, decisive at the first NoPE layer, estimator above it;
   gives kvpack a principled `RepairPlan` object. (Demoted from #1 by the review's scoping.)
6. **Top-of-cone release policy + the unmigrated null** — makes cache migration research
   optional; the strongest product story in §5. Residual-stream transfer stays a research bet
   for its self-consistency property, with the corrected (inverted) cost rationale.
7. **The corpus-appliance systems paper** — Zipf/Che sizing on real traces, CDC-dedup ratio,
   verified end-to-end restore throughput, wall-plug joules, break-even reuse counts. The
   end-to-end result nobody has published; closest to the lab's launch narrative.
8. **Agent-scaffold KV modules, no fine-tune** — the fixed-A regime par excellence (tool
   descriptions, scaffolds: near-independent modules, reuse counts in the millions); the
   easiest unclaimed regime, and the one where Families 1–2's precomputed artifacts are
   actually well-posed.
9. **Frequency surgery on rotated caches** — exact transformation, semantically approximate,
   one kernel + one eval + one novelty check.

---

## 9. The decisive first week (kill-or-fund order, falsifiers pre-registered)

- **E0 — the relational diagnostic (build first).** ~200 items: cross-boundary coreference,
  A-vs-B comparison, contradiction detection, source attribution — each with monolithic-cache
  control; report per-category composed/monolithic ratio + answer-span NLL (+ mean PPL only to
  expose its dilution). Decides the FiD question too: does spliced-no-repair ever beat
  monolithic?
- **E0b — the null-reuse control.** At matched TTFT budget: scheduled prefix caching + chunked
  prefill on the disagg lane vs retrieve-and-prefill vs no reuse. If prefix caching meets the
  target, non-prefix composition has no latency economics regardless of the algebra.
- **E0c — measure Δ before repairing it.** Joint vs isolated prefill of identical segments;
  pre-registered predictions: Δ injected at NoPE layers and carried elsewhere; Δ low-rank in
  feature space (span(V_A)); ‖Δ‖/‖x‖ decreasing with depth (repair belongs at layers 3, 7, 11).
- **E1 — NoPE position-freeness, corrected design.** (i) *Suffix-invariance*: prefill [doc] vs
  [doc, filler]; the doc's NoPE rows must be bitwise identical on the CPU oracle (causality) —
  this, not offset-shifting, is the true no-position-term test; offset-shift confounds
  contextualization and would fail for the wrong reason. (ii) *Prefix-sensitivity*: [filler,
  doc] — nonzero by design; its magnitude IS Δ. (iii) *Implicit-position probe* with the
  registered hypothesis: after splicing, B's rows probe to positions 0..|B| ("both documents
  think they're first"); then test the affine-offset fix directly and re-score on E0.
- **E2 — the splice curve.** Two-document compose vs joint prefill; sink dedup as a controlled
  factor; selectors compared at **equal wall-clock** (β costs 3 layers + a QK pass; CacheBlend's
  costs layer 1); order control (A·B vs B·A); length-matched monolithic control; **interference
  control** (splice an irrelevant B, ask A-only questions — the production failure nobody
  measures); a rank-r feature-patch arm (Kamera-on-text); repair-fraction sweep with the
  non-monotonic entity-corruption check.
- **E3 — SWA warm-up, reframed for validity.** Sweep W ∈ {2k … 80k} (the exactness horizon is
  39·W): error vs the 4-bit noise floor, TTFT including warm-up. Expect exactness at ~72% of a
  prefill and the real question to be how fast error decays below the floor.
- **E4 — edit blast radius, measured in the right metric.** Attention-output-weighted ΔKV vs
  distance and layer (raw ‖ΔKV‖ at NoPE layers is expected flat by construction — pre-register
  that a flat *weighted* profile kills the CDC-dedup branch and the incremental-sync story).
- **E5 — Cov_q spectra + gate calibration.** Per-head centered query covariance; effective
  rank; cross-domain stability (instability → per-domain calibration); validate the two-regime
  gate on our engine.
- **E6 — the economics receipts.** *Verified* end-to-end restore throughput (MAC included);
  TTFT including warm-up; wall-plug joules/token prefill vs restore (converts the `mock`
  field); Zipf fit + Che sizing; **break-even reuse count** as the headline, not bandwidth.
- **E7 — cross-family calibration.** Run E0+E2 on a dense-RoPE model (Qwen-class; kvpack
  already pins its rotation tables) with pre-RoPE storage + re-anchoring: anchors our results
  to the literature's home regime and proves the `cache_algebra` engine is model-generic.

Family scope: E0, E0b, E0c, E2, E4, E5, E6 are family-generic; E1, E3 are the RNoPE-specific
branches; E7 is the generalization control. Everything else (multipole tier, dithering,
Rao-Blackwellized caches, linker tokens, materialized-view selection, percolation-shaped
repair) branches off these results.

---

## 10. Reference index (arXiv IDs)

Theory/structure: 2501.12352 (test-time regression) · 2605.20271 · 2102.11174 (fast weights) ·
2405.21060 (SSD) · 2008.02217 (Hopfield) · 2402.05099 (Hydragen/LSE merge) · 2104.09864 (RoPE) ·
2604.15356 (cache entropy ≤ text entropy) · 2502.15955 (Θ(nd) barrier) · 2605.25085 (power-law
sensitivity / Wyner-Ziv) · 2607.01520 (minimax compression risk) · 2604.11501 (quantization vs
rank; routing flips) · 2512.05916 (KQ-SVD) · 2510.00636 (Expected Attention) · 2405.04434 (MLA) ·
2503.18893 (xKV) · 2410.03111 (LoRC depth propagation) · 2602.05929 (KV-CoRE).
Sinks/heads: 2309.17453 (StreamingLLM) · 2410.10781 · 2504.02732 · 2603.11487 (sink necessity) ·
2402.17762 · 2512.22213 (secondary sinks) · 2404.15574 (retrieval heads) · 2410.10819
(DuoAttention) · 2406.02069 (PyramidKV) · 2406.10774 (Quest) · 2602.02579 (ProphetKV).
Non-prefix systems: 2405.16444 (CacheBlend) · 2410.15332 (EPIC) · 2502.15734 (Cache-Craft) ·
2510.10129 (CacheClip) · 2606.23581 (Kamera) · 2502.05431 (APE) · 2311.04934 (PromptCache) ·
2409.15355 (Block-Attention) · 2410.07590 (TurboRAG) · 2502.16002 (KVLink) · 2512.23049 (Prompt
Choreography) · 2412.15605 (CAG) · 2506.06266 (Cartridges) · 2603.20218 (11-system comparison) ·
2604.13226 (KV Packet) · 2606.19667 (reordering "CacheWeaver" — unrelated to our repo) ·
2412.10319 (SCBench) · 2407.00079 (Mooncake) · 2310.07240 (CacheGen) · 2404.12457 (RAGCache) ·
2507.07400 (KVFlow) · 2601.08670 (PCED) · 2509.01092 (REFRAG).
Cross-model/synthesized: 2608.03893 (NVIDIA anchor) · 2510.03215 (C2C) · 2411.02820 (DroidSpeak)
· 2601.06123 (LatentAlign) · 2605.22863 (LCF) · 2507.11953 (IAM) · 2508.02751 (SmallKV) ·
2502.02789 (SpecPrefill) · 2410.03960 (SwiftKV) · 2410.08391 (KV Prediction) · 2405.05254 (YOCO)
· 2405.12981 (CLA) · 2602.12029 (PrefillShare) · 2603.13281 (ICaRus) · 2512.17910 (aLoRA) ·
2410.05004 (HCache) · 2603.19664 (residual stream) · 2605.17613 (VeriCache) · 2601.08343 (judge
divergence) · 2403.05527 (GEAR).
Positional: 2401.01325 (Self-Extend) · 2402.17463 (DCA) · 2308.16137 (LM-Infinite) · 2309.00071
(YaRN) · 2306.15595 (PI) · 2402.13753 (LongRoPE) · 2203.16634 (Haviv NoPE) · 2305.19466
(Kazemnejad) · 2404.12224 (variance/entropy) · 2509.21042 (Behind RoPE) · 2501.18795 (RNoPE-SWA
/ Command A) · 2503.19786 (Gemma 3) · 2605.05696 (Irminsul, MLA reuse) · 2608.11231 (LinearKV).
Quantization: 2402.02750 (KIVI) · 2401.18079 (KVQuant) · 2605.06675 · 2605.08317 · 2608.14191.
Engine facts verified in-repo: d_model 6656, vocab 202048, per-head QK-RMSNorm (K-norm weight
1.0, pre-RoPE, pre-cache), attention scale folded (logits ≈ 43.8·cos), lm-head-only softcap, no
RoPE scaling; `rope_base_full` parsed-unused and `sliding_window_pattern` converter hazard noted
in config.
