# Release provenance

This file freezes the inputs for the first Muse Glimmer release program. It
does not claim that any hardware seal has passed.

> **Recovery note (2026-08-14):** references below to an individual “seal” or
> the old eight-seal chain describe historical, ineligible apparatus. Current
> authority is `release/release-lock.json`, the 15 mandatory unsealed lanes,
> one exact-identity readiness receipt, and one fresh atomic final bundle as
> documented in `docs/private-release.md`.
>
> This is a chronological evidence ledger. Earlier diagnoses and source-state
> bullets are not silently rewritten when later evidence supersedes them; the
> last dated override plus the tracked machine-readable contracts controls.
> In the current recovery tree, Cargo resolves kvpack only from the audited
> `third_party/kvpack` snapshot and ignores sibling checkouts.
>
> **v0.1 scope override (2026-08-14):** public-CoreML ANE is experimental and
> post-release. It is not a mandatory lane, release identity input, seal member,
> or candidate artifact; v0.1 `auto` routing is permanently Metal. The ANE
> results below remain chronological POC evidence, not release blockers.

## Source state

- Muser standalone safety checkpoint: `cdaf126` (`main`, no configured public
  remote at checkpoint time).
- Ferrite extraction reference:
  `a85048a90fd448585beb9c1b14a819e54a4f16f9`, private research
  repository. The source worktree was dirty and was used
  read-only. A copied file must retain its original license header and record
  the source path and commit in `docs/extraction-manifest.md`.
- CPU text extraction lineage:
  `83cfd55584dde68a9affca9c76af6a6124a3cf32`. Muser carries the GGUF parser,
  Muse config/loader, mmap weights, CPU quant math, tokenizer, capture, and
  reference forward graph without a Ferrite runtime dependency.
- kvpack release branch: `release/muser-alpha2`, with standalone crates
  `kvpack-core`, `kvpack`, and `kvpack-handoff` at `0.1.0-alpha.2`. Muser pins
  that local release source and uses its authenticated Handoff V2 receiver.
- kvpack handoff migration provenance:
  `d776d9fd5e110ac0e5bfdca2e3b76eeb15fa54ae`; Muse interleaved-RoPE and exact
  relocation fixes were manually retained while incompatible cache authority
  experiments were excluded.

## Toolchain

- Rust `1.93.1`, pinned by `rust-toolchain.toml`.
- Freeze host: Darwin arm64. Accelerator identity and OS build must be added
  to each benchmark packet by the safe harness; this document is not a
  benchmark receipt.
- The fresh llama.cpp comparator is source-pinned to
  `89e0aa6fd362617d9073e0dafc18e41241521572`. The v3 receipt authenticates the
  fixture patch and the `llama-bench`, `llama-server`, and
  `llama-perplexity` binaries; the build step did not execute them.
- The Muse `mtmd` bridge and its complete shared-library closure were built in
  an isolated clone of that same commit. Its v2 receipt authenticates every
  packaged dylib and records that the bridge was not executed during build.

## Model inputs

The authoritative revision, filenames, and SHA-256 values are machine-readable
in `docs/release-artifacts.json`. Weights remain outside git and are never
bundled. Validate local files with `scripts/validate_release_artifacts.py`.

## Claim status at freeze

### Live-state override (2026-08-12)

This section supersedes operational statements inherited from earlier
handoffs. Handoff notes are discovery hints, not release evidence. Current
source, direct process/hardware probes, and retained run receipts take
precedence, in that order.

- The GX10 producer is reachable and was directly observed idle (GB10 at 0%
  utilization, with no Muser, llama.cpp, Python workload, or workload
  container). It is not blocked by the stale-process report in the original
  handoff.
- The CUDA-to-kvpack producer is not being reimplemented. Muser's
  `scripts/gx10/llamacpp/llamacpp_session_send.py` descends from the
  qualified llama.cpp C-API/session exporter in the kvpack Spark lane.
  It preserves that implementation's cell-position-aware K/V plane parsing,
  V-layout handling, hybrid full/SWA block mapping, and canonical logical
  ranges. Muser extends it with direct 512-token NoPE streaming, exact logical
  SWA-tail gathering through llama.cpp cell metadata, multimodal witnesses,
  and atomic target-plus-DFlash state. The live 8,191-token run has exercised
  the producer's 2,560-cell physical wrap, which the original prototype spike had
  explicitly left untested.
- Cross-device transport and placement are proven through exact transmitted
  bytes, authenticated atomic commit, exact greedy target tokens, and exact
  DFlash tokens. Full target logits still differ from a cold local Metal
  recomputation because the CUDA-produced K/V already differs at layer 0.
  This is a producer/consumer numerical-parity problem, not a missing kvpack
  exporter or a wire-layout problem.
- Metal DFlash target synchronization is fixed. The measured regression began
  at the change that globally enabled untracked shared-buffer hazards; the
  current tracked-buffer path restores 100% acceptance in the focused POC.
  The earlier handoff's batch-verifier diagnosis was incomplete.
- Public-CoreML ANE execution is functional and target-exact in the focused
  POC, with every convolution assigned to the Neural Engine. It is currently
  slower than Metal DFlash (432.243 ms versus 174.583 ms), so the ANE speed
  claim remains false.
- The earlier `MLPredictionOptions.outputBackings` stop is superseded: the
  corrected public output-backing path has since executed successfully in
  separately identified guarded cells. The latest Mac CoreML lane stopped at
  a 28,901-byte, weight-free, 1,088-position stateful-attention package, which
  failed ANEF compilation with error `-14`. This excludes model weights,
  package size, and the earlier 384-position pilot geometry as sufficient
  causes. This is a package/compiler verdict, not an accelerator-availability
  issue. Subsequent work in this turn is offline; the next accelerator
  continuation will test the separately identified manual-GQA topology copied
  from the compiler-proven `ane-book` stateful transformer path.

- Muse target on CPU: the historical development GGUF passes geometry and
  finite-logit checks and matches the pinned fresh-llama `p1` fixture for 32
  greedy tokens. The pinned 16,756,681,056-byte official target is now acquired
  at SHA-256
  `7e9b74b7c8875e9e265695df9613bf6290f2392e479ce740495a129019c488d8`;
  its standalone loader/tokenizer passes an exact round trip and its first CPU
  forward emits the complete 202,048-logit vocabulary. Five diverse 32-step
  CPU fixtures now retain 160 finite full-logit rows and their digests. The
  official CPU route also completes an exact 64-token free-running record with
  generated-token digest
  `95995afebb6e2de066c5bbc65a85519b20f9bf92c16b502b0ceade266532eb9a`.
  A 64-token Metal record now matches both the standalone CPU route and the
  source-pinned fresh-llama top-1 oracle exactly (token digest
  `95995afebb6e2de066c5bbc65a85519b20f9bf92c16b502b0ceade266532eb9a`).
  A fresh complete five-fixture CPU packet under the corrected binary32
  quantized-row validator is sealed at receipt SHA-256
  `0c1094e9eb8e6ebb0ab8ac6613d57f640a32f8004fe57b55b9dbb31a39e764ce`.
  All 160 rows validate and the aggregate gates pass: 158/160 top-1
  (`0.9875`), `0.949375` mean top-10 overlap, `0.00478154` relative
  target-NLL error, and zero nonfinite values. The formerly invalid row
  exceeded a `0.501`-bin bound by `6.60e-7`; the corrected `0.51`-bin bound
  remains below one encoded probability step.
  The complete long-context correctness packet remains outstanding, so this
  is not a release seal.
- Muse target on Metal: implemented from the accepted Ferrite kernel lineage.
  A live four-token DFlash cell on the M3 Ultra
  exposed a pre-existing synchronization regression to commit `b9678d4`:
  enabling `HazardTrackingModeUntracked` for every shared buffer before all
  cross-encoder dependencies had explicit synchronization changed target
  hidden states while final greedy IDs still matched. The identical historical
  cell accepted 3/3 drafts at `a0e1627` and 0/6 immediately after `b9678d4`.
  Restoring tracked shared buffers and Ferrite's accepted sequential hidden-
  capture verifier restores 3/3 acceptance on the current quantized DFlash
  route, with exact target output, 89.003 ms target-only decode, 172.793 ms
  DFlash decode, 44.511 ms drafting, and 127.850 ms target verification. This
  is a focused POC, not a Metal or DFlash seal.
- Muse vision: the official graph and CPU oracle are implemented; the pinned
  1,400,328,928-byte official artifact is acquired at SHA-256
  `f48b452316f9b213758e8659444029b961a24a07f99a1abb2a9f88b06f7c00c6`.
  Its standalone CPU load validates 50 blocks, 1536-wide embeddings, 16×96
  heads, patch size 14, a 32×32 learned position grid, and 6656-wide target
  projection. The four-fixture qualifier is now executable as one bounded
  packet: it rejects incorrect source dimensions, compares pinned-upstream
  preprocessing with the Rust CPU oracle at the `1/255` pixel bound, checks
  projected embeddings at cosine `0.999` and relative-L2 `0.01`, and requires
  exact 64-token decoder equality. Its receipt authenticates the complete
  image insertion position range rather than substituting a row count. Each
  paired Muser/fresh-llama TTFT cell owns its loopback server lifecycle,
  disables prefix reuse, records three/five raw samples, and requires both
  engines to report the same installed prompt-position count. The evaluator
  still requires at least three stable cells including high resolution, with
  every stable cell at or above llama.cpp. This is executor readiness only;
  no four-fixture hardware packet or vision seal exists yet.
- Official Muse DFlash: CPU, Metal, exact sampled verification, and adaptive
  fallback are implemented. The pinned 1,631,205,312-byte GGUF is acquired and
  matches SHA-256 `27d9a805fa29b943cfb6ad4843367cd4eaaaf06bd452d8cc3e00a2cd18a677bc`;
  a CPU structural load found 2,555,985,152 finite parameters with the expected
  five layers and target captures `[1,13,25,37,49]`. The
  qualification matrix includes adjacent alternating runs of native
  llama-DFlash and rejects any stable Muser cell below parity. Muser consumes
  the same pinned `dflash-kquant.gguf` sidecar directly, including one-based
  official target-layer metadata conversion and GGUF tensor-name validation.
- Public-CoreML ANE DFlash: public-API runtime and reproducible shard exporter
  are implemented against the pinned assistant. An initial 36-matrix export was
  rejected after the public ANE compiler failed on 19,968-channel FFN graphs.
  The first resident layout used 70 exact 6,656-channel input/output
  partitions. The v4 artifact contract applied the public-CoreML topology and
  runtime laws retained in the sibling `ane-book`: `[1,C,T,1]` tensors,
  INT8-per-tensor 1x1 convolutions, fixed resident models, reused
  input `MLMultiArray`/feature-provider storage, stride-aware public
  `MLMultiArray` output reads, serialized inference, and fewer, fatter graphs
  only where the public compiler accepts them. Q/K/V share
  one physical convolution and each compiler-safe FFN partition kept
  gate/SiLU/multiply/down inside Core ML. Noise Q/K/V and target K/V rows share
  one `[1,C,32,1]` prediction per layer, reducing the planned artifact to 26
  packages and 26 hot predictions per assistant forward. The v5 source
  contract now also fuses output projection, residual add, post-attention
  RMSNorm, and the first FFN partition into each layer's tail head; one
  continuation program completes the remaining 10,752 intermediate channels.
  An official-artifact dry run, after correcting the assumed square output
  projection to the measured 4,096-to-6,656 geometry, plans exactly 16
  predictions: one FC, five QKV, and ten tail programs. The two tail payloads
  contain 211,288,064 and 214,695,936 raw INT8 weight bytes, below the 250 MiB
  ceiling. The v5 manifest validator and a synthetic public-CoreML graph
  conversion passed before the live artifact result recorded below. The
  v1-v4 readers remain compatible. The official ANE route loads only the assistant's f32
  norm vectors; its k-quant projection matrices are not redundantly expanded
  beside the compiled INT8 packages. Conversion is pinned by
  `scripts/coreml-requirements.lock` to Python 3.12-compatible coremltools 9.0
  and NumPy 2.1.3, following `ane-book` revision
  `3cf5969eda414832e0cb6c58e3372400fc3c6277`. The v4 `MLComputePlan` receipt
  at SHA-256
  `9676b2f0730e2b49668696b7b96d531fb02458f80e5c68832a03ebe01016d2db`
  records `CPU_AND_NE`, 26 packages, and every convolution assigned to
  `MLNeuralEngineComputeDevice`. After the target hazard fix, a focused live
  end-to-end packet on the resident k-quant Metal assistant produced exact
  target output and 100% Metal and ANE draft acceptance. ANE took 432.243 ms
  versus 174.583 ms for Metal DFlash; target verification differed by -0.37%.
  The public-CoreML execution path works and satisfies the verification-tax
  POC, but the required ANE-over-Metal speed gate still fails; the complete
  packet and speed seal remain outstanding.
  A subsequent official v5 artifact did compile and its complete 16-package
  `MLComputePlan` receipt assigned every convolution to
  `MLNeuralEngineComputeDevice`. Its focused live POC remained target-exact
  with 100% draft acceptance and -0.076% target-verification tax, but took
  474.893 ms versus 181.514 ms for Metal DFlash. Runtime diagnostics reported
  exactly five ANE compiler fallbacks, matching the five tail-head programs;
  v5 is therefore rejected as a performance route. The v6 experiment retains
  16 calls but emits only `post_attention + partial0` and the normalized hidden
  state from each tail head, reducing its public output from three hidden-width
  blocks to two before the continuation FFN. Its official artifact has manifest
  SHA-256
  `56c784df6d1ccb2e35349b1ef664e46c7a4951026889ce78df4d0ab48f4a7d95`;
  the complete `CPU_AND_NE` compute-plan receipt has SHA-256
  `ffa3064145d2a9de90016a18d1728c7e0acb9cec7b7a21155695a3768b46b1ee`
  and assigns all 16 packages to `MLNeuralEngineComputeDevice`, with no compile
  fallback. A live target-exact POC retained 100% Metal and ANE acceptance but
  measured 438.876 ms for ANE versus 176.187 ms for Metal. Direct warm-package
  profiling then measured only 72.826 ms across the complete FC, five QKV, and
  ten tail invocation sequence, localizing most of the end-to-end cost to
  repeated prompt work and host/runtime orchestration rather than convolution.
  Preparing the first 65 prompt rows outside the decode timer reduced the
  target-exact ANE result to 238.637 ms, while Metal measured 153.681 ms and
  target-only 89.300 ms; acceptance remained 100% and verification tax was
  0.378%. This is a real improvement but still only 0.644x Metal DFlash, so the
  16-call split is rejected for release performance. The next ANE route must
  move the complete assistant-layer attention/KV/tail boundary into resident
  stateful Core ML rather than tune this split further. Prompt preparation is
  not a TTFT claim and will be charged in later server qualification.
  Re-evaluating the already-ported Ferrite all-position target verifier after
  the tracked-buffer repair overturned its stale rejection: live four- and
  eight-token cross-round POCs remained target-exact with 100% Metal and ANE
  acceptance. It reduced the four-token target verification interval from
  roughly 128 ms to 106 ms and is now the default transactional verifier.
  The eight-token cell still measured 263.783 ms for Metal DFlash versus
  206.596 ms target-only, because two four-position target batches consumed
  212.663 ms; small-batch target execution remains the shared DFlash/ANE
  bottleneck. A direct GGML `mul_mm` probe preserved exactness but increased
  that verification interval to 389.179 ms, so the existing per-row upstream
  GEMV dispatch remains accepted and the 32-wide batch tile is rejected for
  verification-sized batches.
  The assistant's intended 15-token verification length changes the end-to-end
  result decisively: one live 16-output-token POC was target-exact with 100%
  acceptance and measured 441.734 ms target-only, 224.901 ms Metal DFlash, and
  316.480 ms ANE DFlash. Metal therefore reached 1.964x target-only and ANE
  reached 1.396x target-only, while ANE remained only 0.711x Metal and incurred
  1.699% target-verification tax. The 16-position target batch took 199.456 ms
  on the Metal-DFlash arm and 202.845 ms on the ANE arm. This POC proves the
  repaired Metal differentiated route and validates length 15 as the leading
  tuning candidate; it does not replace the frozen eight-prompt tuning packet.
  It also makes the remaining ANE constraint explicit: because both arms share
  roughly 200 ms of target verification, a resident whole-layer ANE draft and
  a materially faster 16-position target batch are both needed for the required
  1.10x ANE-over-Metal ratio.
  Extending Ferrite's imported high-occupancy Q4_K batch kernel from multiples
  of 32 to multiples of 16 preserved the same target-token digest and 100%
  acceptance in a second length-15 POC. Target verification fell from 199.456
  ms to 193.097 ms and Metal DFlash fell from 224.901 ms to 217.908 ms, reaching
  2.023x target-only. ANE DFlash remained exact at 309.619 ms and 1.424x
  target-only, but only 0.704x Metal. This one-repetition POC accepts the kernel
  routing change, not a stability or release-performance claim.
  The registered Muse investigation also scoped ANE beyond drafting: fixed
  2048-token SWA target-decode work was the ANE-compatible side of the repeating
  three-SWA/one-NoPE topology, while the 13 growing NoPE layers remained on
  Metal. An early analytic rejection assumed only 28--45 GB/s of M3 Ultra ANE
  bandwidth; the later on-device concurrency probe measured roughly 97 GB/s of
  concurrent ANE contribution, 1.42x aggregate over its synthetic Metal stream,
  and 0.02--0.03% Metal tax. Target-decode ANE partitioning is therefore an
  active empirical POC lane again. It must choose the share by whole-layer and
  boundary measurements rather than blindly move all 39 SWA layers; pure hybrid
  prefill remains rejected by the earlier 20--50x loss.
  A subsequent real 16-position verifier profile resolved the target-split
  question at operation granularity. The exact graph reported 985 dispatches
  and 192.212 ms of GPU work: FFN down/gate/up consumed 111.664 ms, Q/K/V/gate
  consumed 58.053 ms, attention only 1.399 ms, output projections 9.388 ms,
  and the batch LM head 5.754 ms. A direct port of Ferrite's generic Q6_K SGM
  batch kernel preserved exact target output but regressed adjacent verifier
  time from 193.596 ms to 195.339 ms and was removed. Together with the public
  CoreML layer-tail result (roughly 10.36 ms ANE versus 2.90 ms Metal), these
  measurements reject serial target-layer offload and fine-grained per-layer
  ANE weight splitting for this release: CoreML synchronization dominates
  before enough Metal weight traffic can be removed. The primary ANE lane is
  therefore the complete five-layer DFlash graph with CoreML-owned attention
  state, following the stateful transformer pattern in `ane-book`; projection-
  only CoreML remains qualification scaffolding rather than the release path.
  The first official layer-0 stateful pilot did not pass that qualification:
  both 384- and 128-position state variants failed ANEF compilation with error
  `-14`. Rewriting the 16-query attention as four internal T=4 subgraphs inside
  the same prediction failed identically, excluding both state size and the
  T=16 softmax axis as sufficient causes. Those negative packages are retained
  in external evidence storage and are not release artifacts. The next isolated
  graph removes the eight-way GQA branch expansion by repeating K/V heads once
  and issuing one public CoreML fused scaled-dot-product-attention operation per
  T=4 query slice. Its official-weight Torch trace is finite, and skip-load MIL
  conversion reduces the optimized graph to 309 operations with four fused
  attention operations instead of eight expanded softmax branches. It still has
  no CoreML residency or timing claim. Guarded qualification of that exact
  68,212,273-byte INT8 package still failed ANEF with error `-14`; therefore
  fused SDPA and lower branch count are not sufficient. Inspection then found
  that the supposedly T=4 graph had retained a T=16 KV-state scatter. The next
  offline-qualified variant partitions both K and V state writes into four
  T=4 contractions, matching the largest stateful path actually proven by
  `ane-book`; its official-weight output is finite and its skip-load MIL graph
  contains eight state-write matmuls plus four fused attention operations. It
  has not entered a hardware cell as a combined weighted package. A second
  isolation artifact now
  removes the already-proven QKV/output convolutions entirely and accepts exact
  normalized/rotated Q/K/V tensors. It retains one prediction for all 16 rows,
  four T=4 state-write contractions, and four fused attention operations; the
  resulting skip-load MIL graph is 178 operations. The exact release-sized
  1,088-position version exported successfully as a 28,901-byte package, but
  its guarded MLComputePlan load failed ANEF with the same error `-14`. This
  cleanly excludes projection weights and isolates fused SDPA plus stateful
  cache mutation as the rejected compiler surface. The next offline-qualified
  graph now matches `ane-book` rather than approximating it: explicit GQA
  score/softmax/value products per KV head, four-token query and state-write
  chunks, and no fused SDPA. Its official-geometry Torch output is finite; its
  optimized MIL graph has 722 operations, including 72 matmuls, 32 softmaxes,
  and two state updates. The release-sized package exported successfully, but
  its guarded `CPU_AND_NE` MLComputePlan load still failed ANEF with `-14` at
  `muser-receipt://dflash-attention-only-manual-1088-plan-guarded/20260812T160507Z-f68e7b46c92f4c1b9efc5e1d7260c97e.command.log`.
  This rules out fused SDPA as the remaining cause. The rejected graph still
  differed from `ane-book` by concatenating mutable MLState with the 16
  external noise K/V rows before contraction. The next isolated topology
  contracts state and noise K/V separately, concatenates only their score
  tensors before the shared softmax, then sums the two probability/value
  products. At full 1,088-row geometry its optimized graph is finite with 136
  matmuls, 32 softmaxes, and two state updates. Against the concatenated Torch
  reference with populated state and writes it has max absolute error
  `1.9073486e-6`, mean absolute error `8.7054104e-8`, and cosine
  `0.99998945`. The full-geometry split-KV package also exported successfully,
  but its guarded MLComputePlan load failed with public error `-14`.
  Unified-log correlation provides the missing internal cause at
  `2026-08-12 18:11:47.300` local time:
  `ANECompilerService: RegAlloc: failed to allocate intermediate tensors.`
  The public `-14` value is therefore not a usable root-cause classification.
  The same service reported `live input tensor ... not used in network` for
  the fused-SDPA package and compiler return `22` (`EINVAL`) for the manual
  concatenated-KV package; `ane-book` separately records `-14` for weight and
  process-memory ceilings. `apple-ml-re` confirms the private compiler has
  distinct SDPA, matmul, softmax, state/ring-writer, and intermediate-buffer
  validation surfaces, but exposes no public error-code map.

  Offline inspection with `scripts/coreml_mil_tensor_pressure.py` explains the
  RegAlloc failure. Four T=4 write contractions materialized four full
  `[1,8,1088,128]` K intermediates and four matching V intermediates. Each is
  2,228,224 bytes; the two four-way stack tensors are 8,912,896 bytes each.
  Conservative textual SSA liveness peaks at 25.500 MiB for both manual
  packages. The fused-SDPA graph independently peaks at 25.915 MiB because
  repeated K and V each materialize `[32,1,1104,128]` tensors of 9,043,968
  bytes. The corrected graph now uses the single full-width mask-times-KV
  contraction used by the passing `ane-book` stateful exporters while keeping
  attention queries in T=4 slices. It removes both cache-sized stacks and all
  but the write-mask reduction: 1,280 optimized MIL operations, 130 matmuls,
  32 softmaxes, and two state updates. With populated state and one-hot writes,
  its Torch output is bit-identical to the rejected four-write graph. This
  release-sized single-write package still failed its guarded MLComputePlan
  load. Unified logs at `2026-08-12 18:19:49` again identify ANEF register
  allocation failure; the evidence packet is
  `muser-receipt://dflash-attention-only-split-manual-q4-write16-1088-plan-guarded/20260812T161937Z-fa13ed59b8da4e669ef728b1867cdd81.command.log`.
  The change nevertheless halves conservative source liveness to 12.750 MiB,
  showing that removal of the stack tensors was real but insufficient.

  The next compiler-isolation graph uses GQA's independence explicitly. It
  divides the eight KV heads into two four-head K/V MLState pairs and completes
  both groups inside one Core ML prediction; output heads are concatenated in
  their original order, so there is no host state copy or extra prediction.
  With populated state and one-hot writes, the grouped implementation is
  bit-identical to the original eight-head state for both attention output and
  post-write K/V state. Skip-load conversion accepts all four state updates at
  the official 1,088-row geometry and yields 1,366 optimized MIL operations,
  including 132 matmuls and 32 softmaxes. The improved textual-MIL diagnostic
  measures a 6.565 MiB conservative peak: six live `[1,4,1088,128]` tensors
  plus small query/output slices. The release-sized package exported, but its
  guarded MLComputePlan load still failed. Unified logs at
  `2026-08-12 18:32:23` again report `RegAlloc: failed to allocate intermediate
  tensors`; the exact command log is
  `muser-receipt://dflash-attention-only-split-manual-q4-write16-group4-plan-guarded/20260812T163210Z-e18f987af6bb4ab19303d24e512ab2a9.command.log`.
  This disproves the simple hypothesis that conservative live byte count alone
  explains ANEF allocation.

  Comparison with the passing `ane-book` Phi/HyMT verifier graphs exposes the
  remaining structural difference: the rejected Muse procedure unrolls four
  T=4 query chunks and contains 132 matmuls, whereas the proven public compiler
  surface is one externally invoked T=4 stateful program. The next isolation
  artifact therefore keeps all 16 noise K/V rows and one 16-row target-state
  write but accepts only four query rows per prediction. The first call writes
  the target block; three later calls reuse the same MLState with an empty write
  mask. With populated state, four such predictions are bit-identical to the
  single rejected T=16 prediction for both the concatenated output and final
  K/V state. The full-eight-KV-head form is closest to `ane-book`: 434 optimized
  MIL operations, 34 matmuls, eight softmaxes, and two state updates. The
  lower-pressure four-head grouping has 508 operations and 36 matmuls. This
  T=4 program passed the public compiler gate. ANECompilerService completed in
  roughly 0.4 seconds and reported 9,323,468 bytes of procedure DRAM. The
  persisted `CPU_AND_NE` MLComputePlan assigns all 145 non-constant operations
  to `MLNeuralEngineComputeDevice`, with estimated cost entirely on ANE. The
  guarded command initially exited nonzero because the receipt validator only
  recognized convolutions or fused SDPA as qualifying compute; manual attention
  has neither despite every matmul, softmax, state operation, and elementwise
  operation preferring ANE. The validator now uses the stricter correct rule:
  every non-constant operation must prefer a Neural Engine device. Evidence is
  retained under
  `muser-receipt://dflash-stateful-attention-only-t4-manual-write16-fullstate-1088-pilot/compute-plan.json`.
  A faster follow-on topology is now offline-qualified rather than accepting
  the 20-call attention route as final. The earlier rejected T=16 manual graph
  unrolled four query chunks and produced 130 matmuls. Contracting all 16 query
  rows in each per-KV-head matmul produces the same compact graph as external
  T=4: 434 optimized operations, 34 matmuls, eight softmaxes, and two state
  updates. With populated state and writes it is bit-identical to the four-
  chunk T=16 reference for output and final K/V. Its guarded compiler cell
  passes: all 145 runtime operations prefer Neural Engine, estimated cost is
  entirely ANE, compilation completes in roughly 0.4 seconds, and procedure
  DRAM is 9,670,604 bytes. Stateful attention therefore falls from twenty calls
  to five and the complete v7 path from 36 predictions to 21 without
  duplicating projection work. The clean receipt is
  `muser-receipt://dflash-stateful-attention-only-t16-singlechunk-fullstate-1088-pilot/compute-plan.json`.
  Muser carries the corresponding FP16 public-CoreML runtime,
  per-layer MLState ownership, exact sink-plus-window physical ring mapping,
  snapshot replay, and transactional CPU shadow needed to consume a passing
  package. The runtime retains query-slice support for experimental artifacts,
  but the release manifest binds one 16-row contraction and therefore one
  attention prediction per layer.
  The same manual-GQA construction is now also applied to the
  official weighted layer-0 pilot at the full 1,088-row geometry. Its offline
  concatenated-KV MIL graph is finite and contains 929 operations. The new
  split-KV weighted graph is also finite. Its grouped-state successor is
  bit-identical to the original eight-head state for the full post-projection
  hidden output and post-write K/V state. The optimized graph contains 1,573
  operations: 132 matmuls, 32 softmaxes, two INT8-eligible QKV/output
  convolutions, exact norms/RoPE, and four state updates. This official-weight
  variant began as T=16 scaffolding. Its external-T4 successor is now also
  prepared offline: a one-hot query selector retains the complete 16-row
  noise/target projection inputs while selecting four Q and residual rows per
  call. Four state-reusing calls are bit-identical to one T=16 weighted call
  for post-projection hidden output and final K/V state. The optimized official
  layer-0 graph contains 659 operations, including two INT8-eligible
  convolutions, 36 matmuls, eight softmaxes, and two state updates. Its guarded
  public compile also passes: ANECompilerService completed in roughly 0.6
  seconds with 92,889,480 bytes of procedure DRAM. MLComputePlan assigns both
  convolutions and all 249 runtime operations to Neural Engine. Three
  `constexpr_blockwise_shift_scale` nodes report no device and zero cost because
  they materialize compressed weights during compilation; the receipt validator
  now excludes `constexpr_*` alongside constants instead of misclassifying them
  as runtime fallback. The first receipt retaining that raw distinction is
  `muser-receipt://dflash-stateful-attention-layer0-t4-manual-write16-fullstate-1088-pilot/compute-plan.json`.
  Release manifests now bind a single 16-row query contraction, the single
  16-row state write, and full eight-KV-head state so an older high-pressure
  package cannot be sealed accidentally.
  The complete 21-call v7 artifact exported successfully. Its first strict
  whole-artifact plan exposed one CPU-preferred `cast` at the FC package
  boundary; estimated cost is `0.002430` versus `0.576422` on ANE (0.42%). The
  FC convolution and every substantive operation in all 21 package entries
  remain ANE-preferred. Receipts preserve
  `all_ane_compute_resident=false` rather than hiding that boundary, and add a
  separate fail-closed `all_ane_compute_qualified`: only cast operators may be
  non-ANE and their combined estimated cost must be at most 1%. V7 runtime
  qualification requires this explicit field; older receipts cannot silently
  satisfy the new contract.
  The first v7 end-to-end POC then failed before prediction with an explicit
  runtime geometry error: the 65-row prompt context was presented as one new
  target segment, while the Core ML state-write input has block capacity 16.
  The runtime now streams arbitrary new target segments into MLState in
  16-row tiles before the query call, preserving absolute sink/window ring
  placement. Segments of at most 16 rows still combine their state write with
  the query, so steady decode remains 21 calls; only prompt/context catch-up
  pays the additional tile calls. A focused 33-row test covers two full tiles
  plus the final partial tile and verifies packed head layout and physical
  slots.
  The first corrected live v7 cell then completed end to end: its output was
  target-exact, ANE draft acceptance was 100%, and the 21-call ANE route took
  193.472 ms versus 312.491 ms for the Metal assistant, a 1.615x ANE-over-
  Metal speedup. This verification-length-3 POC is intentionally too short to
  amortize speculation and is not a target-only speed or stability claim. The
  immediately following verification-length-15 cell failed the exact-output
  gate before emitting a sample. Offline triage identified the unqualified
  bespoke batch-16 target-verifier kernel as the route difference from the
  retained exact 193.097 ms Ferrite-derived verifier; the hardware lane stopped
  and that route was removed. The guarded rerun with the retained Ferrite tile
  was target-exact with 100% acceptance on both assistants: target-only took
  439.264 ms, Metal DFlash 218.091 ms, and v7 ANE DFlash 276.216 ms. Thus v7
  is 1.590x target-only but only 0.790x Metal, leaving ANE host/runtime draft
  overhead as the release blocker. Enabling the separately ported Ferrite fused
  Q4_K FFN remained exact and measured 458.160, 218.448, and 275.101 ms
  respectively; its ANE target-verification tax was 1.752%, inside the 2%
  gate, but the ANE-over-Metal speed gate still fails.
  Per-call profiling localized the steady v7 draft round to 63.631 ms inside
  Core ML prediction and only 0.571 ms in measured input/output handling;
  serial package micro-optimization cannot satisfy the ANE-over-Metal gate.
  The exact Ferrite-derived conditional-overlap POC therefore splits target
  verification after layer 49 and provisionally drafts from the predicted
  bonus token, committing only on a complete match. Its first live run failed
  exactness at output token 1 (target 5342 versus provisional 27756) and the
  hardware lane stopped. Offline diagnosis found that the suffix graph had
  consumed layer 49's temporary normalized buffer across command buffers.
  The repaired split rematerializes layer 50's attention norm from the
  authoritative residual and now has deterministic full-accept, mismatch,
  replacement, and rollback tests; no speed claim exists until a guarded
  diagnostic rerun passes exactness.
- The exact target verifier now has an offline-compiled batch-16 specialization
  of Ferrite's accepted Q4_K simdgroup-matrix kernel. It retains Ferrite's
  dequantization and matrix arithmetic, stages both 32-row halves of the same
  64-row weight tile with 64 threads, and removes the two simdgroups that only
  multiplied the original 32-wide tile's padded upper batch half. The Metal
  compiler and `muser-engine` feature build accept it. It has no correctness or
  performance claim until the next guarded verifier cell compares it with the
  retained 193.097 ms exact baseline.
- Greedy DFlash LM-head reduction now uses Ferrite's accepted two-phase Metal
  argmax, so each draft round reads 16 token IDs rather than copying roughly
  12.9 MB of full-vocabulary logits to the host. Exact sampled speculation
  continues to retain the full assistant proposal distributions.
- Authenticated GX10-to-Mac live handoff: target, atomic target-plus-DFlash,
  and identity-bound multimodal producer/consumer paths are implemented. The
  multimodal exporter reuses upstream llama.cpp's `mtmd` graph, emits exact
  embedding-position witnesses, streams NoPE tiles during prefill, and gathers
  the final logical SWA tail from llama.cpp cell metadata without a target
  session save. It was statically compiled on the live host. A live
  authenticated target transfer committed 104 ordered segments and 3,461,120
  bytes in the initial source-run POC. (Those counts are target-only; they do
  not include the ten DFlash planes.) The source-pinned release
  container was subsequently built as
  `sha256:18b1779a1fc54b870aba7a9d265464b17bde5be51139f7af1b04b3cbc225b07c`
  from llama.cpp `89e0aa6fd362617d9073e0dafc18e41241521572` and adapter digest
  `bf566cad9940f7d3172d72c719b81b3054942a615833551a1041d9ba65347700`.
  A guarded real-CUDA container POC over the identical official target and
  DFlash artifacts emitted a 3,464,248-byte target session and a
  1,333,194-byte DFlash session; its 65-position combined prefill took 330 ms
  and state export took 3 ms. This is producer functionality evidence, not a
  long-context TTFT seal.
  The refreshed default-CUDA image
  `sha256:44b858a710e245b7080bd944161619f9e76cba0a284b08df058995e6260fce8e`
  reproduced both session files byte-for-byte. DFlash state uses llama.cpp's
  actual two-memory-module serialization (a zero-KV encoder-state block
  followed by the five-layer KV block); the sender now parses that form and
  materializes the complete combined generation as 114 segments and
  6,123,520 bytes. The same default image also completed an 8,191-token
  target-plus-DFlash prefill across the 2,048-token SWA window and the
  producer's 2,560-cell physical wrap: CUDA prefill took 9.604 s, state save
  took 0.169 s, and the sender validated 514 ordered segments carrying
  235,392,000 bytes. The live M3 Ultra receiver subsequently ACKed two fresh
  65-position combined generations, each with 114 segments and 6,123,520
  transmitted bytes. Atomic target-plus-DFlash installation produced exact
  target tokens and exact remote/local/target DFlash tokens. The deliberately
  staggered POC includes receiver wait time and is not TTFT evidence.
  Exact qualification remains blocked by producer/consumer math drift already
  present in layer-0 projected K/V; the transfer bytes and layout validate, but
  the remote full-logit digest does not yet equal local Muser.
  A sealed forced-cuBLAS diagnostic image
  (`sha256:f4440b3f8fdedd04afb6ef3c0d32f3000c8b721016748fda80299e8218e926d1`)
  reduced mean absolute KV drift against the llama CPU reference from
  `0.0280716` to `0.0210094`, but only `1.03556%` of the 1,730,560 exported
  fp16 values were byte-identical. Explicit fp32 cuBLAS compute did not improve
  the result (`0.0211141` mean absolute drift, `1.03845%` identical). The
  cuBLAS route is therefore rejected as an exactness fix; MMQ contributes to
  the drift but is not its sole cause.
  A subsequent first-boundary probe on the idle GX10 localized the divergence.
  For the identical two-token prompt and official target, llama.cpp CPU versus
  CUDA layer-0 `attn_norm` sums were `262.807312` and `262.807343`, while the
  first quantized Q/K projection outputs had already diverged: `Qcur_normed`
  sums were `-13.132085` versus `-16.220798`, and `Kcur_normed` sums were
  `27.896976` versus `27.721706`. The difference therefore begins in
  quantized QKV projection/reduction semantics before attention; kvpack,
  transport, RoPE, and receiver ring placement are downstream of it. The
  retained CPU/CUDA probe logs have SHA-256 values
  `3e89ecef25aac390c962227bfc91662cfb305f1a8e1d7b5d88867e26133dd712`
  and `fcc3584382e2e473e1c351bfc03258ead844979a6254178fe14c158dee86bec4`.
  A focused layer-0 CPU-math oracle now pins the token embedding and every
  learned layer-0 tensor to the host, disables KQV offload, and prevents the
  integrated CUDA scheduler from silently re-offloading host operations for
  batches of 32 or more. An initial comparison against an older retained CPU
  session reported six K and two V adjacent-fp16 differences; that comparison
  crossed producer revisions and is not parity evidence. Re-running both
  routes from pinned llama.cpp `89e0aa6fd362617d9073e0dafc18e41241521572`
  made raw-f32 `attn_norm-0`, `Kcur-0`, `Kcur_normed-0`, `Kcur_rope-0`, and
  `Vcur-0` byte-identical. All 16,640 layer-0 K and 16,640 V fp16 values then
  matched exactly, as did the complete 65-row `l_out-0` tensor (SHA-256
  `0105624e57001ac54161e36b03f125bc8914021cee335bd204ef06882a3ae3ec`).
  The focused oracle prefill took 444--452 ms, so this route is a correctness
  oracle rather than the production solution. The source-pinned image is
  `sha256:425854b14471f5584293dd8c197040417be2a1f95b497bafe44f60aced3caa63`;
  its build receipt, exported session, and exporter log have SHA-256 values
  `c6601b0a79ece80446128d259ed15208c3e30d94ee67ef1116d5acbbccd1bd49`,
  `4667af5fe16a2c3506937d2db90ca6e1c2183b6cd483553b7b03c6a4077f6106`,
  and `79d912acaab0c073ffae76f4b3994f119e7560229344bfcfded3815d44c1df08`.
  This proves that pinned same-identity CPU arithmetic closes the complete
  first-layer boundary exactly and that transport and attention layout are not
  blockers. A held-boundary-token decode then captured all 202,048 raw f32
  logits: target-only CPU and layer-0-oracle-plus-CUDA retained the same greedy
  token (`965`) but no logit was byte-identical (mean absolute error
  `0.1983705`, maximum `0.7338862`; SHA-256
  `694fa451bb99b8b3b1e74a89d8209d75c285e0156fad62a7f7ad9eb19cb8ead3`
  versus
  `183278ecc230ddea80885dc08a9cd02d03e26d697bcd4237f5a8cbbd80ec32ea`).
  Full-logit equality therefore remains unclaimed until compatible CUDA
  reductions replace the oracle route across every layer.
  The first default-off CUDA compatibility kernel now closes one complete
  quantized projection exactly.  For pinned batch 65,
  `blk.0.attn_k.weight` produces all 16,640 raw-f32 `Kcur-0` cells
  byte-identically to the host oracle (SHA-256
  `fd1fd3b7c39d147252202a91c076a750a1f235dc4fd45e8eb5f3f7c82ba4d157`).
  The initial generic reduction retained eight float lanes and missed 13,584
  cells by at most `3.8146973e-6`.  The pinned ARM/SVE implementation instead
  reduces each 256-value Q8_K/Q4_K block in integer arithmetic and applies its
  minimum bias and dot contribution as two ordered scalar fused operations.
  Matching that order and fusion made the CUDA projection byte-exact while
  retaining the exact input norm digest
  `3929f5c93278bd093863201292d21f6c9665a7a27d2006a6cd6d75bd58f9c903`.
  A default-off one-cell trace records Q8_K scale/hash, decoded weight scales,
  integer bias/dot, and both fused boundaries for all 26 blocks.  The image is
  `sha256:2b95694e5e55f956415739faa111d501101b750741d6c39907b2dbb06c321ee8`;
  its build receipt has SHA-256
  `2a47fad15f94d2cf8be5bf0fc104cef5c4b51d64707ca156792c110581963d3b`.
  The same arithmetic is now warp-parallel: eight output-row warps share each
  exact Q8_K activation block, reduce only associative integers in parallel,
  and preserve the two scalar fused boundaries in lane zero.  Repeating the
  complete 65-row layer-0 K boundary retained the exact input and output
  digests above and completed prefill in `0.145 s`.  The parallel image is
  `sha256:02031e0c23ed8b44bc33f21968e10b46dd8eb91f121a89684c94a2937eda6d98`;
  its build receipt has SHA-256
  `1a296e87b4cbcb54d9d89fbc71c6446db19760d9ba2ba4f3f977674bb0e37920`.
  The sequential Q6_K companion then made all 16,640 raw-f32 layer-0
  `Vcur-0` cells byte-identical as well (SHA-256
  `d7feff11c24d26426e8dd7a78ac502b1f1c877fc2c2b0bf7a9331caeaf2f3b55`).
  It uses the same Q8_K quantizer, reconstructs the signed six-bit weights and
  their 16-value scales into one exact integer super-block dot, and matches
  the ARM/SVE scale multiply plus ordered scalar FMA.  Its image is
  `sha256:b0b41ef47667bbac0faf14ff612a3dd9b7940d863417a9cb96a959d771b7fb73`;
  the build receipt has SHA-256
  `3bebcfe65e1f1563b4b0b355e322cc1b05f7acf45185c00267f4df60168e68d0`.
  Its eight-warp parallel form likewise shares the exact Q8_K block and
  reduces signed integer products before the ordered lane-zero FMA.  The
  complete layer-0 V boundary retained the input and all-16,640-cell output
  digests above while completing the 65-token prefill in `0.143 s`.  The
  parallel image is
  `sha256:505c52f6933be25b271d55cac3a2766464c7818c8e8cb1de3fd338b5e8d45cad`;
  its build receipt has SHA-256
  `42c3351ec3f399053fef6342e64c724fb1c06227562cce10759f427d6d6255fd`.
  A complete layer-0 boundary comparison with the parallel Q4_K/Q6_K kernels
  and exact RoPE stayed byte-identical through `attn_norm`, Q norm, K norm,
  and K RoPE.  The first remaining divergence is `attn_out`: pinned CPU flash
  attention has SHA-256
  `b1f50c29006fcaa6bae991295e509c247871da7c2db7f98e4f72a405a5b96629`
  versus CUDA
  `f7ff3bb23f29cbfa710011e0bf03c2686d702cbc8c2ff0109c2c52167e1d283e`
  (16 of 266,240 f32 cells exact, mean absolute error `0.000144991`, maximum
  `0.0126100`).  Disabling flash attention did not close the boundary: the
  CPU/CUDA result had 4,134 exact cells, mean absolute error `0.000106764`,
  and maximum `0.00568581`.  A first CUDA replay of CPU's online-softmax and
  fp16 accumulator ordering was worse and was removed rather than retained.
  The parallel compatibility layer completed in `0.339 s` versus `0.474 s`
  for the host layer-0 oracle.
  Flash-off attention is now byte-identical through `attn_out-0`.  CPU QK
  uses `ggml_vec_dot_f16` because permuted Q is not contiguous, so llamafile
  is skipped and src1 f32 is rounded to f16; CPU PV uses llamafile tinyBLAS
  F16×F32 (`tinyBLAS<4, float32x4_t, …, ggml_fp16_t, float, float>`) because
  softmax is contiguous, keeping B in f32.  The CUDA compatibility kernel
  follows that split.  On image
  `sha256:2265a753aa6cbb8b7fa4d180daa9fc8dcf914e46105c558306c50be4fd0f0d67`
  (`muser-gx10-prefill:0.1.0-alpha.1-attn-pv-tinyblas`, receipt SHA-256
  `fd185dada0eae454e4db10ee34d8f82c9a407884060f834d13fbb4468d8bcc37`) the
  266,240-cell layer-0 dumps match the host oracle at SHA-256
  `8d68cd30eda0e5369a6755a854238677c3eaba2183c3c465079ac8f8f7dba207` (`kq-0`),
  `f6c0da903cfb02a1e44ebdd4b811ecd1cbb98b489b7b7912f2d6cf0d8e90b8a3`
  (`kq_soft_max-0`),
  `11eab68345b245d9b15ea257c0adfa05a141d48e3a6cc8d6e8a0d22312089deb` (`kqv-0`),
  and `b90077a53fbf05284fa217ed099ea4b4df99ee0386d2b81de9d93dd264213bab`
  (`attn_out-0`).  Flash-on `GGML_OP_FLASH_ATTN_EXT` replays that same proven
  f32 QK+softmax+PV boundary rather than native CPU flash or the deleted
  fp16 online-softmax path; CUDA flash-on `attn_out-0` retains SHA-256
  `b90077a53fbf05284fa217ed099ea4b4df99ee0386d2b81de9d93dd264213bab` and
  therefore does not match native CPU flash
  `b1f50c29006fcaa6bae991295e509c247871da7c2db7f98e4f72a405a5b96629`.
  Default CUDA RMS already matches `attn_norm`; `MUSER_CUDA_CPU_ORDER_RMS`
  stays unset.  `attn_gate_proj-0` is also exact at SHA-256
  `0c724f5a5992e0d4d07825451ceca3042e614f593649af677925ebbef209974d`.
  The next split is scalar CPU sigmoid (`1/(1+expf(-x))` against glibc 2.39
  `expf`, not `ggml_v_expf`): 144,189 / 266,240 cells exact versus CUDA
  `expf`, mean absolute error `5.64119e-9`, maximum `1.19209e-7`.  Replaying
  glibc 2.39 `__expf` on CUDA closed that boundary: `attn_gate_sig-0` SHA-256
  `d25450841d6be760ca3d9dcfb0057d69bec1d4336770ae42243faeb800fa47f8`,
  `attn_gated-0`
  `524c2ea3b4c3076eb8ab75a39d0167d25e8913ebdbb73ef9d60810d3af170172`, and
  `attn_o_proj-0`
  `358abe4f181f52450ab1b2cd4f2195b8a337dbf4291b06437044794bb732899b` are
  byte-identical.  The following split is post-attn RMS at eps `1e-8`
  (`attn_post_norm-0` 233,872 / 432,640 exact, maximum `7.62939e-6`).  Default
  CUDA RMS (f32 reduction + `rsqrtf`) remains the 1e-5 path; a 2-D double-sum
  kernel is required only for 1e-8.  Enabling that kernel on 3-D QK-norm
  broke the already-exact `attn_out` path and is rejected.  Later SiLU,
  residual add, scale-then-tanh, and all-layer parallel Q4/Q6 remain required
  before `result_output` qualification.
  The following RoPE compatibility boundary is now exact too.  The CUDA path
  consumes a host-libc table generated by the CPU cache's iterative theta
  order, supports the fused `ROPE + VIEW + SET_ROWS` destination, and mirrors
  the pinned ARM vector contraction asymmetry: the even result rounds
  `x0*cos` before its fused subtraction, while the odd result rounds `x1*cos`
  before its fused addition.  Replaying the complete boundary on the GX10 made
  all 16,640 raw-f32 `Kcur_rope-0` cells byte-identical (SHA-256
  `83527c6d448b07e194b5ff251366dfe425cc16d637e1a7c8c56f8bb2a0a22c69`).
  Its default-off image is
  `sha256:f6e23e82b22a8f92e6380f85b645daff08a28dcd323402e5e015f03df1daeaa0`;
  the build receipt has SHA-256
  `b0bb8d6f273bf389c6de1f9faac6dc22ef2661f8863367c72a3c87279bc98ecf`.
  The model's sole Q5_K tensor, `output.weight`, now has an exact CUDA
  compatibility boundary as well.  With identical pinned normalized input,
  the host and CUDA paths produced all 202,048 raw-f32 lm-head cells
  byte-identically (SHA-256
  `bcc4571e5bf817796794977f03fd32b45d147b749ba0554b86718a9b06cd6534`).
  A block trace showed that the ARM Q5_K contraction rounds the minimum
  product first and then fuses the scaled integer dot with its subtraction,
  the opposite fused operand from Q4_K.  Matching that boundary closed the
  prior 1--4 ulp drift.  The default-off warp-parallel image is
  `sha256:42dfd913e3789e46988d6d006aeb542d5cd7f618dbf48955ea08f4b04d70db62`;
  its build receipt has SHA-256
  `d018e7861d5b1e3ccdf83d837803b7d08d5ea5e343ef9e54f0121b53117f7071`.
  The raw projection is exact, but the ordinary CUDA logit scale/tanh remains
  a separate non-exact boundary (post-transform SHA-256
  `a68e8b8be0da6c3ba797caf6825cb4b8ea55f1cc9879b166f3149df2274fed68`
  versus the host's
  `daeb8e6935a0acd7c5641f6d036d5bb37d3b0cf4fef2f70459a58fcdef86ec37`).
  The controlled exactness run still spends about 14 seconds in its deliberate
  all-CPU 52-layer upstream oracle.  An adjacent normal-CUDA A/B isolated the
  warp-parallel Q5_K output route itself: both default and compatibility runs
  completed the 65-token prefill in `0.140 s`, with identical normalized-input
  digests.  This single measurement shows no millisecond-resolution overhead;
  it is focused POC evidence rather than a performance seal.
  A type-checked prefix selector can apply the same Q4_K/Q6_K arithmetic to a
  complete layer for localization.  On layer 0 it retained exact K projection,
  K normalization, K RoPE, and V projection digests, but serializing every
  attention and FFN projection increased the 65-token prefill to `3.355 s`.
  This prefix route is therefore an oracle and is rejected as the production
  all-layer implementation; production kernels must parallelize rows/tokens
  while preserving the now-proven integer reductions and scalar fusion order.
  These are focused quantized-arithmetic results, not yet a remote-parity claim:
  Q4_K and Q6_K compatibility must cover every layer, and the remaining
  non-learned CUDA boundaries must close, before full-logit qualification.
  A reproducible container builder now archives the exact fresh llama.cpp
  commit, applies both accepted patches, builds the CUDA exporter without using
  a GPU, and emits an arm64 image receipt. Container-bound producer schemas
  authenticate that receipt and address the image by immutable SHA-256 ID.
  A fresh read-only preflight at `2026-08-12T14:57:25+02:00` reached the GX10
  directly: its NVIDIA GB10 reported 0% utilization, and no Muser, llama.cpp,
  Python, or workload container process was running. Live correctness and
  performance seals remain outstanding pending the cross-device math-parity
  fix.
- Exact-prefix restore audit (source and CPU-only qualification, 2026-08-12):
  the resident radix is not a placeholder. It structurally interns immutable
  256-row plane chunks, retains witnessed exact-final logits, permits
  non-aligned cuts only as exact hits, and chooses aligned deepest ancestors
  for suffix prefill. The durable path uses kvpack's sole
  `PrefixNode`/`LocalStore` authority, verifies every state into an owned
  detached shadow, requires the cut-scoped exact-final distribution, and
  installs only after complete manifest/chunk/range verification. A late
  kvpack resource-release acknowledgement previously happened after the live
  engine install; it now happens before live mutation because the adapter has
  already copied all source bytes. Resident and durable exact-final commits
  additionally checkpoint the prior generation until both KV and final logits
  have installed, rolling back on a late distribution mismatch. The remote V2
  sink independently prepares complete target and optional DFlash shadows and
  validates their identities before commit; text remote prefill intentionally
  retains one boundary token for local first-logit computation rather than
  transporting a CUDA-derived final distribution. These paths are operational
  implementations. Their 8K--131K corruption matrix, full-logit digests, and
  cache-hit performance thresholds remain unsealed hardware evidence.
- Benchmark lifecycle: warm Muser and fresh-llama servers are created inside
  each serialized accelerator lease and exit through private loopback-only,
  token-authenticated routes. Both own a pre-load self-deadline; the campaign
  never signals or kills them.
- Baseline and differentiated release seals: not run.
  A later throwaway `--smoke` under identity
  `sha256:7ab8b3e939843b25d4f4dc3afb2fe9bfc012051355d19fd96358e6ba37931451`
  is not a seal identity. It did pass exact Metal correctness and all three
  adjacent pairs with both CVs ≤ 2%: PP128 439.306 ms versus 453.890 ms
  (1.033×), TG512 1.852 s versus 1.892 s (1.021×), TTFT128 447.927 ms versus
  556.449 ms (1.242×). Winning production paths in that packet were prefill
  `concurrent-qkvg-ffn-v1`, decode FFN `upstream-split-gate-up-v1`, one
  retained concurrent command buffer per teacher-forced token, and TTFT
  warmup `one-uncached-request-after-ready-before-timing-v1`. Fused decode
  FFN remains opt-in. This smoke does not replace a later one-identity
  `--full` packet.

### Live-state override (2026-08-13)

GX10 CUDA versus the true ARM CPU oracle is now closed through
`result_output` on a fair dump set. This supersedes the earlier statement
that all-layer logits remained unqualified. It does not claim CUDA versus
Metal bit-identity, and it is not a campaign seal.

- True CPU oracle remains `n_gpu_layers=0 no_host=1 use_extra_bufts=0` with
  `--tensor-cpu ".*" --threads 4 --flash-attn 0`. CUDA dumps use
  `--flash-attn 0` plus the default-off compatibility flags
  `--cuda-cpu-order-qk-prefix blk. --cuda-cpu-order-q4k-tensor
  token_embd.weight --cuda-cpu-order-q5k-tensor output.weight
  --cuda-cpu-order-rope --cuda-cpu-order-attn --cuda-cpu-order-nonlearned
  --cuda-cpu-order-rms`. Held-token dumps are `--dump-decode-only` after
  `--skip-tail 1`. Failed C-`/` Q4_K image
  `muser-gx10-prefill:0.1.0-alpha.1-q4k-cpu-div` stays tainted; nvcc
  `-use_fast_math` turns device `/` into `__fdividef`.
- Layer 27 was the first Q4_K FFN break (`ffn_up-27` / `ffn_gate-27`) after
  exact `l_out-26`. ARM `quantize_row_q8_K_ref` contracts `iscale*x +
  12582912.f` into FMA inside inlined `nearest_int`; the CUDA kernel used
  separate `__fmul_rn` then `__fadd_rn`. One Q8_K value in super-block 6
  differed (`qhash` `deba13b6` versus `5a43faa9`). Matching
  `__fmaf_rn(iscale, x, 12582912.0f)` made `ffn_up-27`, `ffn_gate-27`, and
  `l_out-27` byte-identical, then `l_out-51` and `result_norm`.
- The remaining split was Q5_K `output.weight`. ARM
  `ggml_vec_dot_q5_K_q8_K` compiles to `sumf += fma(d, dot, -(dmin*sumi))`
  (`fmul` / `fnmsub` / `fadd`). Two running FMAs into the accumulator
  matched the CUDA dump and missed the CPU dump by about 1.5e-5. Matching
  that association closed `result_lm_head` and the scale/tanh
  `result_output` path, including `muse-glimmer.final_logit_softcapping=20`
  and `logit_scale=0.196116`.
- Qualified dumps, all byte-identical CUDA versus true CPU, SHA-256:
  T=1 held-token `result_output`
  `3d041164abe5acfbeeb1bbbd460911fe99c1f90a63ae90cdd9c7f40fae3283f7`;
  T=1 `result_lm_head`
  `7f5855e3608d972db42bbda5555d725bd76667e2c275b0581858bc357cedb548`;
  T=1 `l_out-51`
  `c948e9c6ff3eed385aa27958e2b38eaa6847ed9ba0540e317a4b056a006197d5`;
  T=65 held-token `result_output`
  `da365289e160b51dab8f94e8a77b59c856d4750fad781793a26c2231bd3d71b2`;
  T=65 prefill `l_out-0` (shape 6656×65)
  `70f631366281e5e730fc278deb484d20948c8052199144e73f5ce5e4c4d5a1d6`.
  Flash-on `GGML_OP_FLASH_ATTN_EXT` `result_output` matches flash-off at
  both T=1 and T=65 (same SHA-256 as the flash-off files above). Probe
  top-8 greedy ids and 4-decimal scores matched on every exact dump pair.
- Compatibility image
  `muser-gx10-prefill:0.1.0-alpha.1-q5k-assoc`,
  `image_id`
  `sha256:42aa09b87324292eb6732032ac545ba132bc103fb4628f0d0a2a67ce64073ccf`,
  `adapter_sha256`
  `25b2ca24b29a92b47ca57c2ce6cc82abde450d8553d57f44aa95892ebc790b02`,
  receipt
  `muser-build-receipt://gx10-container-89e0aa6-q5k-assoc-20260813T0435.json`
  (file SHA-256
  `491d691d4f4f2a2ff76688b53e04b218640e589fa6806fa2a8fa48601bf3bc64`),
  llama.cpp `89e0aa6fd362617d9073e0dafc18e41241521572`. Compatibility
  kernels remain default-off. Q6_K FMA / `__fdiv_rn` nearest_int, 3-D RMS
  isolation, and the 21-call v7 attention topology are unchanged.
- This is focused producer-math evidence, not a remote-parity qualification
  and not a Metal-vs-CUDA claim. The retired eight-seal apparatus never
  authorized v0.1; the 15-lane unsealed matrix, readiness receipt, atomic
  final campaign, and RC verification remain outstanding. Do not tag.
- The first 66-token teacher-forced Metal seal attempt under identity
  `sha256:e83fd9010f2b71864aecd44dcb23030eff43de5cdfcbb0710e77fff530ab3f25`
  matched llama top-1 on all 32 scored rows (mean top-10 overlap 0.984,
  relative target-NLL 0.00218) and matched the Muser CPU argmax on every
  row. It missed the 0.5 CPU/Metal all-vocab abs-logit gate: max abs 1.031
  on one high-confidence decode row (hidden-logit rms ≈ 11.7, cosine
  0.99992, target-token abs 0.27). One-token and prompt-prefill golden
  tests remain gated at 0.5. The 32-row campaign envelope is 1.1 so the
  all-vocab gate still fails closed on a broken residual while accepting
  Q8_K-Metal versus f32-CPU tail mass on confident rows. Run-id
  `overnight-20260813-correctness` is tainted. Do not reuse it.
- Greedy case `diverse-p1` then emitted special tokens `200007` / `200022` /
  `200023` with no exact UTF-8 decode/re-encode. llama-perplexity already
  consumes `MUSER_COMPARATOR_TOKEN_FIXTURE`; the `-f` corpus is a dummy.
  `greedy_campaign.py` now writes that dummy instead of requiring a
  round-trip render. Run-id `overnight-20260813b-greedy` is tainted.

Historical Ferrite measurements are research provenance, not Muser product
claims. A value becomes a Muser claim only through a retained, complete seal
packet produced after correctness succeeds.

## Accepted and rejected routes

The accepted target route is the release program's fixed Muse-only Metal path.
Rejected research routes are not compiled into the engine: GQA head-packing,
retrieval prompt-lookup speculation, VM/registry routing, paged-Q8, MoE,
multi-architecture dispatch, and private ANE APIs. Speculation is limited to
the official five-layer DFlash assistant. Its independent correctness path is
implemented; its release performance gates remain unsealed.

## Freeze status (2026-08-13)

A full-team review of the working tree completed. Seven freeze commits were
created on `main`, in order:

1. `2d64231` — docs: 2026-08-13 provenance live-state override, schema and
   telemetry notes
2. `03164bc` — web: glass dashboard live-only rewrite (restores closing style
   tag)
3. `598bc3c` — engine: llama FA vec decode integration, flash_attn_v2 f32
   output accumulator, cache/api updates
4. `6edfd67` — server: unified serve/up flow, openai session handling, live
   metrics counters
5. `aed44ec` — cluster,kvpack: streaming tile receive path against
   kvpack-v0.1.0-alpha.2-rc1, economics updates
6. `e065063` — bench,scripts: campaign fixes and GX10 producer parity tooling
7. `82437b8` — docs: import dashboard assets from feat/one-click-deploy
   (screenshot needs re-capture post-fixes)

`kvpack`/`kvpack-core`/`kvpack-handoff` (the separate coordinated worktree
this source bundle pins, see "Source state" above) were committed and tagged
`kvpack-v0.1.0-alpha.2-rc1` at commit `70c34c7` in that worktree.

### ANE mirror-overlap rerun (2026-08-13)

Evidence directory:
`muser-receipt://ane-v8-mirror-overlap-vlen15-256`. The v8
mirror-overlap route at verify length 15, 256-token window: exact-token
output, draft acceptance dropped from 100% at short context to **93.3%** as
context grew, ANE-over-Metal throughput **0.797×** (still slower than the
Metal assistant), and ANE target-verification tax **5.13%** against the
program's historical 2% gate. **The historical verification-tax gate fails.**
ANE is now experimental/post-release and this result does not block v0.1. This
supersedes the v7 guarded-rerun numbers earlier in this document (1.590×
target-only / 0.790× Metal, 1.752% tax) as the current ANE state; the v7
numbers remain valid history for that narrower POC, not the launch-path
verdict.

### ANE v9 fused-attention POC (2026-08-14)

The retained warm three-repetition, 256-token packet is
`muser-receipt://ane-v9-fused-sg4-256x3-20260814`. All
repetitions produced the exact same target-token digest; ANE/Metal draft
acceptance was 238/259 (91.89%). ANE raw times were 5.118, 5.113, and 5.073
seconds (CV 0.40%), versus Metal at 4.185, 4.204, and 4.260 seconds (CV
0.75%). The resulting ANE/Metal throughput ratio was 0.8266x. Mean target
verification tax was -0.156%, inside the historical 2% POC gate, but the ANE
route remained slower than Metal. The implemented v9 path and receipts are
preserved; further call-count fusion stopped when ANE moved outside the v0.1
release boundary. No v0.1 seal, candidate member, or launch claim derives from
this packet.

### Artifact hash restamp

The release-prep recovery restamped every Muser-side shader digest in
`docs/extraction-manifest.md`, including the previously stale
`batch_sgm_q4_aligned.metal` record. `restamp_extraction_manifest.py --check`
now treats any future mismatch as a release-blocking integrity failure.

### Wrap-restore exactness (2026-08-13, matrix-20260814 chain)

- The greedy campaign's ring gate previously invoked a deleted module's
  test and could self-skip; it now runs
  `decode::tests::real_model_wrap_boundaries_and_detached_restore_replay_exactly`
  with `MUSER_MODEL` and refuses skipped runs.
- That gate exposed a foundation-era gap: detached restores installed SWA
  rows at physical origin 0 while a live ring is rotated, so a wrapped
  session's suffix could never replay bitwise (float accumulation order
  differs with physical scan order). Restores now install at
  `origin_logical % capacity`; the 2,560-cut replay test passes for the
  first time on this hardware. Freeze-commit worktree run confirms the
  old behavior also failed the gate: this predates the 2026-08-13 waves.
- Known-failing model-conditional golden, retained failing:
  `decode::tests::real_model_detached_restore_replays_exact_suffix`
  NaNs the restored replay at a 3-token/max_context=3 raw
  `MetalKvSnapshot` restore and predates the waves (freeze worktree
  fails identically). Campaign snapshot-replay gates use the session
  path at real depths and pass; the raw tiny-context path needs a root
  cause before any claim touches it.
- The remote wire sink still installs at physical origin 0; at positions
  that are not window multiples this contributes ulp-level scan-order
  divergence on top of the open producer-math parity item. Fold into the
  remote-parity fix.

## Recovery override (2026-08-14)

The “freeze status” above is historical. Commit `11119bd` is explicitly
non-releasable, sealing remains disabled, and no evidence from that campaign
authorizes a tag or publication. The current source is a recovery worktree;
only a future clean frozen commit with zero open findings can enter readiness.

- The raw tiny-context detached restore and the remote SWA physical-origin
  defects described above are fixed and covered by mandatory tests.
- The integer-phase Q2.30 NCO oracle was independently rerun across ARM CPU,
  Metal, and GX10 CUDA over the complete 2^28 stride sweep plus one million
  adversarial phases. All three produced stride hash
  `f440e6e27c6c3f04`, adversarial hash `1a5b32d13ce785ab`, and combined hash
  `0e9c19b3b953c4af`.
- The canonical NCO table and pinned rotation order make Q RoPE 4096/4096
  and K RoPE 256/256 bit-exact. A held-token oracle is exact through all 52
  layer outputs and all 202,048 logits.
- A retained 65-token CUDA prefill oracle then localized the production
  discrepancy to Metal FA2 attention. The strict CPU-order prefill kernel now
  matches CUDA exactly for layer-0 attention output, gated attention, output
  projection, FFN input, and final layer output. Evidence is retained at
  `muser-build-receipt://gx10-prefill65-nco-attention-20260814`.
- A containment stop exposed that the resident GX control process could exit
  on SIGTERM before its warm CUDA container cleanup ran. SIGTERM now unwinds
  past per-request error handlers through the daemon's cleanup block. A live
  GX lifecycle check reached `warm exporter ready`, terminated the controller,
  printed `muser-prefilld: stopped`, and left no listener, container, or CUDA
  compute client. This is operational evidence only, not a qualification.
- `GX-003` remains open. The mandatory fresh 2,048-position/256-output,
  target-plus-DFlash, three-repetition qualification has not passed yet; the
  node stays unqualified and no throughput or TTFT claim is authorized.
