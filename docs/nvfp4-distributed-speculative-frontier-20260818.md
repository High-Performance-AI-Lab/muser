# Distributed NVFP4 speculative decode frontier

Status: **bounded non-serving research prototype; product promotion no-go**.
Research and prototype record, 2026-08-18. The shipped matrix remains unchanged:
native NVFP4 speculative requests still fail closed and the 107.9 tok/s
speculative lane remains kquant-only.

## Decision

The prior statement that a GX10 verifier requires checkpoint unification was
too strong.

Lossless speculative decoding does not require the drafter and target to use
the same checkpoint. It requires one endpoint to execute the authoritative
target transition. The other endpoint may use any approximation. The real
fork is therefore target identity:

1. **Attempt to preserve today's composite model lineage.** Load Dudeman on
   GX10 as a second resident model, import the same portable RedHat-produced
   prefill KV bytes installed by the Mac, and use Dudeman for every
   authoritative continuation transition. Mac DFlash remains the proposer.
   The inverse importer and first composite-genesis packet now exist: GX10
   consumed all 52 authenticated RedHat layers, reported exactly 2,047 external
   tokens, evaluated the held boundary under Dudeman, and retained the source
   bundle root. CUDA/vLLM remains a distinct target-engine epoch until broader
   oracle and product qualification passes.
2. **Make RedHat the target.** Keep the existing RedHat session on GX10 and use
   Mac Dudeman/DFlash only to propose. This is algorithmically valid but is a
   new product lane: output semantics, quality evidence, DFlash acceptance,
   and fallback identity all change.

No RedHat-to-Dudeman KV translation is involved in either architecture. Target
and draft KV remain checkpoint-native derived renders. The shared semantic
truth is an authenticated ordered token/frontier log.

The lead candidate had a credible, narrow performance opening. Starting
from the authenticated RedHat bundle, 31 warm prefix-cached GX10 Dudeman M16
runs with all five f32 DFlash target layers copied into pinned host memory and
hashed measured 107.152 ms median and 107.947 ms p95. After charging the
already measured 26.9 ms Mac draft, 0.78 ms RTT, approximately 4.37 ms to move
2,129,920 capture bytes at the measured link rate, and approximately 0.01 ms
for sparse q, full acceptance projects to 114.93 tok/s at median target wall
and 114.28 tok/s at p95 target wall. That screen established the preregistered
bar: at least 99.151% IID per-edge acceptance at median and 99.229% at p95 to
beat 107.9 tok/s. The following end-to-end runs test and reject that assumption
outside the all-accept control.

## End-to-end linear-lane verdict

Those decisive acceptance runs are now complete, and they reject the linear
M16 candidate for general product serving. The direct research path used the
real RedHat portable genesis, persistent GX Dudeman target, authenticated
provisional f16 feature frame, bound final decision, carried frontier, and Mac
DFlash rollback/commit path. The standard trace is a genuine positive control:
it committed every proposal and beat the existing product bar. None of the
three organic content strata was competitive:

| Trace | Output | Rounds | Accepted / drafted | Acceptance | Measured tok/s | Verifier-only ceiling |
|---|---:|---:|---:|---:|---:|---:|
| Standard | 512 | 35 | 477 / 477 | 100.00% | **110.59** | 125.61 |
| Documentation | 256 | 109 | 147 / 1,592 | 9.23% | 15.53 | 20.15 |
| Python | 256 | 55 | 201 / 764 | 26.31% | 11.17 | 40.04 |
| Rust | 256 | 39 | 217 / 570 | 38.07% | 15.41 | 55.96 |

The verifier-only ceiling is `output_tokens / sum(GX verifier wall)` and
therefore grants zero time to DFlash, feature decode, transport, installation,
or scheduling. It is the important rejection bound: all three organic traces
remain below 107.9 tok/s even under physically impossible zero-cost Mac and
network assumptions. Documentation also remains below the 35.5 tok/s native
plain-decode floor under that assumption. Python and Rust wall measurements
overlapped unrelated local validation work, but that can only affect their
reported end-to-end point estimates; it does not weaken the remote-only upper
bound or the accepted-prefix evidence.

Mirror-SD correctly retained 34/34 speculative transactions on the standard
trace. Documentation and Python missed on their first attempt. Rust is the
adversarial classifier case: its first 14 proposals and held frontier were
correct, then its second Mirror attempt failed. A one-round admission probe is
therefore unsafe. After a miss, the prototype deliberately opens its Mirror
circuit and uses exact rollback/recompute; this preserves state but cannot
repair the fundamental number of GX weight-streams per emitted token.

Adaptive width is safe but not a performance escape: the same GX session can
verify width 7, width 3, or the singleton `[frontier]` without changing target
semantics. Singleton costs one Dudeman weight stream per emitted token and is
only a roughly 9–10 tok/s safety mode. A future authenticated GX target-only
burst can avoid one RPC per singleton, but autoregressive dependence still
requires one target transition per token; it is an availability fallback, not
a route to 107.9 tok/s.

Silently switching to Mac plain decode is not exact. Mac/Metal and GX/vLLM are
distinct target-engine epochs, and the V2 session binds the target executor and
engine digest. A legitimate switch must retain a live Mac RedHat-prefix state,
replay only the authenticated accepted suffix, require the Mac next token to
equal the signed unprocessed GX frontier, atomically publish a signed GX→Mac
epoch seam, and discard GX-derived DFlash state. Even granting that unbuilt
handoff zero cost, one failed gamma-14 probe followed by the measured 35.491
tok/s Mac lane projects to only about 35.16 tok/s over 512 tokens, below the
documented greater-than-35.5 fallback gate.

The documentation run reused one byte-identical first-round server completion
after a client-side bounds panic exposed and fixed an eager-index bug. Its
acceptance counts and signed target result are exact; its aggregate decode wall
is not used as a clean latency claim. The other three sessions were fresh. The
retained receipt SHA-256 values are:

| Trace | Mac receipt | GX service receipt |
|---|---|---|
| Standard | `d4ae629f6256b9e288885cda9b74de16233d9f94ecca017c3887cfb0c231cbb5` | `82561ced353ee46ed2a78060be09d4f0fc0416859b634f5e23dd1cd4efd066cb` |
| Documentation | `4b1c70e6a75ee0fca475afce57dfc649aad68ab0283b22c538f89e7970eba787` | `d463f72f966003906458753f1d803e9cfe2d4af3e81146f0629afb739d95ff3c` |
| Python | `efa4d48d058c69ad010626b167ede218d70df00e0a3d7cdaaa9392fd7bdbe95b` | `9104239fd461b694d1ff3bfcbcfdecfe67e19d5989ffc80b9e3bccdc6ec18979` |
| Rust | `10f6269b8c7d47e0486b07467480b1336296d1808ad66c6c8f479ec1e2c5e09d` | `4068d8bcdc6a4efa7c2a1ac986cf493e3ea5accb5c73eb62b930e7ccbf7e431a` |

This closes the simple-chain attempt, not the checkpoint-mismatch question.
Checkpoint unification is still unnecessary; the remaining performance
question is whether a hardware-efficient token tree can turn otherwise idle GX
batch arithmetic into enough authoritative path coverage. That experiment must
beat the ceilings above with measured emitted tokens per evaluated tree node.

## What was implemented

- `sampling.rs` contains a typed carried-frontier decision, bounded sparse
  maximal coupling, and a finite shared-Gumbel reference sampler.
- For one fixed two-draft, three-token fixture, honest sparse rows produced from
  the full ordered distribution are seed-for-seed identical to the existing
  dense MT19937 helper across 1,000 seeds. Sparse values travel as exact f32
  bits and normalized rows have fixed support limits; broad K=40 property and
  integration tests remain a gate.
- `muser-cluster::verifier_v2` carries the complete evaluated-token transcript
  and actual MT19937 snapshot, derives the genesis head from the signed session,
  replays every sparse-q proposal draw, and fixes one coupling policy for the
  session. The earlier `verifier` module remains as the V1 research record.
- V2 separates log-writer and target-executor identity. Mac requests and the
  journal use domain-separated HMAC; GX results use an Ed25519 target-only
  signature, so possession of the request key cannot forge verifier output.
- V2 implements `Open { frontier } | Closed { reason }`, including cancellation
  or max-token closure without evaluating or publishing the pending frontier.
- V2 has a private, locked, fsync-backed filesystem journal with durable
  reservation and PREPARED records, content-addressed fragment closure,
  invisible idempotent renderer staging, commit WAL, idempotent activation,
  ACK, retry retention, tombstone, resumable GC, and historical restart.
  Live authority fencing is required at admission, PREPARED, and commit; an
  already-WAL'd decision is completed after a crash without being resampled.
- Fragment requirements bind kind, ABI, exact bytes per logical row, row
  coverage, aggregate caps, and the manifest root incorporated into the new
  head. Delivery may be duplicated or reordered without changing visibility.
- kvpack's zeroizing `MacKey` has a documented vendored patch exposing a
  length-delimited domain-separated HMAC operation.
- `benchmark_native_verifier.py` receipts carried-frontier geometry, exact
  parent cache hits, native kernel selection, optional five-layer capture, and
  now an authenticated RedHat portable-KV genesis rather than Dudeman replay.
- `composite_bundle.py` and `composite_connector.py` implement a closed,
  HMAC-authenticated 52-layer portable bundle plus the exact inverse
  interleaved-key→NeoX cache install, including the non-block-aligned 2,047-row
  cut. `qualify_composite_kv.py` and the full-logit oracle retain import,
  sequential, and block receipts.
- `analyze_checkpoint_top1_agreement.py` turns retained E2 top-token traces
  into a reproducible mismatch and throughput-risk screen.

Nothing is wired into the serving route. The V2 journal's authority registry,
renderer, network service, and model execution are explicit integration traits
or callers, not deployed implementations. The prototype intentionally cannot
remove the existing native+DFlash fail-closed gate.

## The target identity invariant

For target distribution `p` and draft distribution `q`, ordinary maximal
coupling accepts proposal `x` with:

```text
min(1, p(x) / q(x))
```

On rejection it samples from normalized `max(p - q, 0)`. `p` and `q` need not
come from the same weights. The final distribution is `p` as long as the
authoritative endpoint actually computes `p` and the coupled sampler is
correct. This is the core result behind
[Leviathan, Kalman, and Matias](https://proceedings.mlr.press/v202/leviathan23a.html)
and the independent
[DeepMind formulation](https://arxiv.org/abs/2302.01318).

That result does **not** allow RedHat logits to certify a Dudeman decision. If
the observable target must remain Dudeman, Dudeman must execute on Mac, GX10,
or a split of the two. Consensus, gossip, CRDTs, and KV translation do not
change that semantic boundary.

The current product already has a composite genesis: RedHat creates portable
prefix KV and Dudeman continues from those bytes on Mac. The lead candidate
now imports those bytes into GX10 Dudeman. The retained bundle root is
`e9ff6e91bcc258320a78c95b659d95adcbeb7009733ce01e4dcd35a413475634`;
the scheduler and receipt both report the exact 2,047-token external cut, so
no Dudeman prompt rows were silently substituted. The continuation identity is:

```text
RedHat prefill checkpoint + portable KV ABI + prefix root
    followed by
Dudeman continuation checkpoint + verifier arithmetic/sampler identity
```

The first 16-row oracle deliberately compared sequential and block execution
from that same composite genesis. All 16 greedy argmaxes matched, including the
actual singleton bonus witness, but no hidden or logit row was bit-identical.
Maximum observed absolute errors were 0.375 in the five captured hidden layers
and 0.0625 in logits. Therefore the evidence supports a named GX block target
for greedy qualification; it does not establish Mac↔GX identity or sampled
distribution equivalence.

## Coupling policy decision

### Default: sparse maximal coupling

The shipped sampler defaults to `top_k=40`. After truncation, each q row has at
most 40 positive entries. Fifteen rows represented as
`(token_id:u32, probability_bits:u32)` are only:

```text
15 × 40 × 8 = 4,800 bytes
```

At approximately 3.9 Gbps the binary payload is around 0.01 ms, before envelope
overhead, and remains negligible beside the five f32 target-hidden rows. For
exactly normalized p/q, maximal coupling has optimal `1-TV(p,q)` collision.
The prototype more narrowly matches the existing finite dense helper when an
honest complete sparse q row is actually sampled from the bound MT state; its
f32 rows are only within a normalization tolerance. It avoids moving a
202,048-wide matrix and remains the sampled MVP hypothesis for trusted peers.

The V2 signed request carries source-ordered positive f32 **weights** inline,
not approximately normalized probabilities. It also carries the full base and
post-draft MT19937 states. GX reconstructs the same f64 normalizer, replays one
`uniform_f64` draw per row, checks every proposed token, derives the f32
probability row, and rejects a mismatched post-draft snapshot. This proves
consistency with the q bytes supplied by the authenticated trusted Mac. It
does not prove that those q bytes came from the declared neural model. An
untrusted proposer still requires target-owned shared Gumbel,
attested/recomputed q, or another proof mechanism; a target-only result
signature is not enough.

### Secondary: shared Gumbel

[Daliri, Musco, and Suresh](https://arxiv.org/abs/2408.07978) show that draft
and target can use the same public Gumbel field, sample locally, and accept the
matching prefix without exchanging q. Target output then depends on target
probabilities plus the shared field, not on which drafter was used. This is
useful when q evidence is untrusted or seeded output must be invariant to a
changed drafter.

It is not the default here:

- 4.8 KB of q evidence is not a meaningful bottleneck;
- Gumbel collision cannot exceed maximal coupling and may be lower;
- Gumbel-Max changes the current inverse-CDF MT sampler semantics;
- scalar SHA-256 plus logarithms is a reference, not a wide-support kernel;
- heterogeneous `libm` is not a replay contract. Target failover needs a
  bit-specified transform or a new sampler implementation epoch.

The public field is keyed by `(sampler version, seed, absolute output ordinal,
token_id)`, never by request, round, or draft width. For explicit `top_k=0`,
the service must use a separately qualified dense/interactive fallback or a
new shared-Gumbel sampler; bounded sparse evidence fails closed.

## Carried-frontier state machine

Muser carries a target-selected token that is not yet evaluated or emitted.
For `D` drafts a round evaluates:

```text
[frontier_in, draft_0, ..., draft_(D-1)]
```

It reasons over `D+2` target witnesses: durable parent witness `T0` for
`frontier_in`, fresh `T1..TD` after successive candidates, and fresh
`T(D+1)` as the all-matched bonus. `T0` must equal `frontier_in`; draft `i` is
compared with `T(i+1)`. Thus only `D+1` rows are freshly materialized in a
prefix-cache hit. The carried frontier plus matching drafts commit. The
mismatch or bonus becomes `frontier_out` and stays absent from emitted tokens,
target KV, and DFlash state until processed as candidate zero next round. The
V2's `FreshTargetRows` coverage starts at the first fresh post-frontier witness
and spans exactly `D+1` rows. The request binds the durable parent head, full
transcript, carried frontier, and sampler state; the parent witness is the
carried token selected by the preceding target decision.

This explicit geometry prevents a common off-by-one: publishing a replacement
whose target KV row and five DFlash features do not yet exist.

## Distributed state and reconciliation

### Canonical state

The V2 correctness-bearing object is:

```text
(session, authority term, output height, token-head hash,
 frontier ordinal/token, sampler-state digest)
```

RedHat or Dudeman target KV and Mac DFlash/draft KV are separate derived
renders. Each is reconstructible from its model-specific genesis root plus the
committed token transcript. The transcript is tiny beside 32k KV and should
normally be retained for the session lifetime.

The log-writer authority and target executor are distinct roles even when one
process temporarily holds both. V2 names both, binds the lease ID and term, and
consults an external live fence before admission, PREPARED publication, and
commit. The production lease registry remains to be connected.

### Preferred proposed service: Mac-owned log, GX soft state

Mac would own the durable authenticated round log. Each request would name the
exact parent head and carry either the committed token prefix or an
authenticated transcript/snapshot object plus candidates. GX uses
automatic prefix caching only as rebuildable soft state. On eviction or reboot
it re-prefills or restores a model-native kvpack snapshot.

V2 implements the simple reconstructible form: every request carries the full
evaluated transcript and actual sampler state, and the retained commit chain
reconstructs both on restart. Snapshot/CAS compaction is deliberately absent;
it is an optimization required before very long-lived multi-session service,
not a correctness prerequisite for the bounded lane.

One vLLM change is mandatory: cache hits must be capped at the authenticated
parent cut. A rejected branch contains mathematically valid KV, but if its
candidate positions become cache hits the verifier may not produce their
prompt-logprob or selected-hidden rows. The service needs either
`max_cache_hit_tokens=parent_committed_len` or a content-addressed artifact
side cache for those rows. The benchmark avoids ambiguity with a unique first
suffix token and asserts exactly 2,048 cached parent tokens.

This design can avoid a remote semantic COMMIT/rollback protocol once those
objects and terminal semantics exist. Persistent GX sessions remain a
performance optimization, not the source of truth.

### Stateful fallback

If public prefix caching cannot expose the required rows, use one GX session
actor as the linearization point. A request names `(term, base_head,
request_id)`. The actor records request-to-result durably before replying;
exact retries return the completion and competing same-parent intents fail
stale. This follows [RIFL](https://web.stanford.edu/~ouster/cgi-bin/papers/rifl.pdf).

V2 durably reserves `(parent, request_id, intent)` before compute. An identical
retry returns Reserved, PREPARED, or the exact completion; changed intent and
competing same-parent work fail closed. Its implemented transaction is:

```text
PREPARED reservation -> immutable staged render + fragment closure
    -> durable result -> fenced head CAS + render activation -> emit -> ACK
```

The renderer stages invisibly and idempotently. The journal fsyncs its commit
WAL before idempotent activation, writes an active receipt, and only then makes
the new head available to a caller. This closes the local crash window; the
real KV/DFlash renderer and client stream-resume boundary are still unconnected
and must honor the trait contract.

The signed term is not itself a lease. V2 therefore requires a live registry
callback at admission, PREPARED, and commit. Multiple target authorities would
still require Raft/Paxos for the ordered head, not a CRDT merge. A delayed
old-term result cannot enter the WAL; only a decision already fsynced there is
completed without consulting a new lease.

### What Baquero contributes

Baquero and Brito's
[Cache Merging as a Convergent Replicated State](https://arxiv.org/abs/2607.01308)
supplies the state/render separation and content-addressed set pattern:

- immutable fragments are named by content;
- duplicates are harmless;
- replicas converge by set union;
- a deterministic render is independent of fragment delivery permutation.

This design extends that pattern with chunk assembly, authentication, complete
manifest visibility, leases, snapshots, and completion records; those are not
claims from Baquero's paper. It is used for hidden rows, probability evidence,
KV deltas, and snapshots. It is **not** used to combine RedHat and Dudeman KV or
select between token branches. Autoregressive token transitions are ordered
and non-commutative. CRDT union may retain competing children for diagnosis,
but a fenced authority chooses the only visible child.

[Merkle-CRDTs](https://arxiv.org/abs/2004.00107) and
[delta-CRDTs](https://arxiv.org/abs/1603.01529) motivate missing-object repair
after loss or partition. The current code implements a one-result arbitrary-
arrival assembler, not Merkle inventory exchange or gossip. None of these
mechanisms authenticates an authority or makes two model transition functions
equivalent.

### Fragment closure and security limits

Each descriptor binds component, kind, logical range, length, and SHA-256. The
session fixes required components and coverage rules. The assembler:

- accepts arbitrary arrival order;
- absorbs an identical duplicate;
- rejects a different duplicate, overlap, gap, undeclared component, oversize
  closure, or wrong digest;
- creates an opaque closure only after all required bytes arrive;
- requires that closure before the token head advances;
- binds the complete fragment manifest into the new head.

V2 persists every fragment in a private content-addressed store, requires an
opaque closure bound to the signed result, and accepts an invisible renderer
stage receipt before writing the commit WAL. Requirements bind an ABI identifier
and exact bytes per logical row. The actual f32 capture/KV decoder must still
register and enforce its concrete ABI; the protocol cannot inspect model
semantics inside correctly sized opaque bytes.

Request and local-journal HMAC remains a trusted-peer boundary, but results now
carry a GX-only Ed25519 signature. That prevents a proposer holding the request
key from forging target output. It still does not make a dishonest q safe for
sparse maximal coupling; use a target-owned policy or verifiable q as described
above.

V2 pins fragments and request-to-result records until an authenticated
durable-install ACK and retry horizon pass, then publishes a tombstone before
resumable cleanup. Late retries return `RESYNC_REQUIRED`; commit/active history
is retained until a future authenticated snapshot compacts it. Safe model-state
compaction still needs a complete model-native snapshot and applied watermark.
Gossip suspicion must never fence an authority or authorize deletion.

## Experimental evidence

### GX native verifier screen

Valid cells use NVIDIA GB10, vLLM commit `6adad087`, native
`FlashInferCutlassNvFp4LinearKernel`, a 2,048-token parent, and exactly the
requested cache hit. The 4/8/16 cells use five repetitions; the exploratory
32/64 cells use three:

| Target | Inputs | Median wall | Range | CV | Full-accept raw rate |
|---|---:|---:|---:|---:|---:|
| RedHat | 4 | 107.338 ms | 105.514–108.362 | 1.013% | 37.27 tok/s |
| RedHat | 8 | 107.747 ms | 106.117–108.327 | 0.896% | 74.25 tok/s |
| RedHat | 16 | 108.903 ms | 108.175–110.362 | 0.803% | 146.92 tok/s |
| RedHat | 32 | 113.728 ms | 112.308–115.232 | 1.050% | 281.37 tok/s |
| RedHat | 64 | 118.335 ms | 117.511–118.396 | 0.342% | 540.84 tok/s |
| Dudeman | 4 | 101.488 ms | 99.845–102.599 | 0.942% | 39.41 tok/s |
| Dudeman | 8 | 102.242 ms | 100.877–102.440 | 0.633% | 78.25 tok/s |
| Dudeman | 16 | 104.473 ms | 103.916–104.870 | 0.313% | 153.15 tok/s |
| Dudeman + five f32 captures | 16 | 110.359 ms | 108.931–112.294 | 1.184% | 144.98 tok/s |
| Dudeman + five f32 captures | 32 | 117.870 ms | 116.006–118.553 | 0.917% | 271.49 tok/s |
| Dudeman + five f32 captures | 64 | 132.772 ms | 131.457–133.178 | 0.554% | 482.03 tok/s |
| Composite RedHat→Dudeman + five f32 in-memory captures | 16 | 107.152 ms | 105.163–108.055 | 0.705% | 149.32 raw tok/s |

The capture cell materializes 2,129,920 bytes; median capture finish, including
transpose, hash, write, and fsync, is 5.061 ms. A fused RPC should stream
layer-major rows and avoid disk/transpose, but it must be measured rather than
credited in advance.

The composite row is that measurement: it retains D2H copies, token-major
transpose, and SHA-256 but stops at the pinned-host serving boundary rather
than writing a disposable file. Median capture finish is 1.037 ms. A separate
15-run fsync control is retained: its median target wall was 111.753 ms and one
40.496 ms capture-finish outlier raised target p95 to 123.648 ms. The economics
gate continues to charge network transfer separately; neither packet assumes
zero-copy wire time.

The captured 32- and 64-input cells materialize 4,259,840 and 8,519,680 bytes,
with median finish costs of 6.887 and 10.863 ms. Despite doubling twice from
16 inputs, total wall rises only 7.511 ms and then 14.903 ms. RedHat without
capture similarly rises from 108.903 ms at 16 to 118.335 ms at 64. This is
consistent with batch amortization and motivates the weight-stream-bound tree
hypothesis. A fused verifier
should decide the selected target path on GX and return captures only for that
path; these screens conservatively materialize captures for every candidate.
They do **not** establish tree-attention correctness, DFlash branch coverage,
or end-to-end tree throughput.

These are greedy teacher-forced timing screens. They include the scheduler,
top-1 prompt-logprob materialization, and bonus row. They do not qualify
sampled top-k materialization, sparse coupling, the network service, or token
correctness. An earlier approximately 1.25 s packet had
`num_cached_tokens=0` and is retained only as an invalid recomputation control.

Retained receipts under `muser-receipt://frontier-verifier-20260818/`:

- `redhat-native-prefix-hit-r5-v2.json`
- `dudeman-native-prefix-hit-r5-v2.json`
- `dudeman-native-prefix-hit-capture-r5-v1.json`
- `redhat-native-prefix-hit-wide-r3-v1.json`
- `dudeman-native-prefix-hit-capture-wide-r3-v1.json`

Composite-genesis receipts are retained under
`muser-receipt://composite-verifier-20260818/`:

- `redhat-prefix-2047-export.json`
- `dudeman-redhat-prefix-import.json`
- `composite-target-sequential-m16.json`
- `composite-target-block-m16.json`
- `composite-target-m16-comparison.json`
- `composite-dudeman-verifier-m16.json` (fsync control)
- `composite-dudeman-verifier-m16-memory.json`
- `composite-dudeman-verifier-m16-memory-economics.json`

### Cross-checkpoint direct-chain risk

The retained E2 teacher-forced top-token arrays compare 190,449 rows. RedHat
and Dudeman disagree on 15,457 (8.116%):

| Content | Rows | Mismatch | Match | IID M16 expected output | Projected rate |
|---|---:|---:|---:|---:|---:|
| Rust | 63,483 | 3.705% | 96.295% | 12.238 | 86.79 tok/s |
| Python | 63,483 | 5.216% | 94.784% | 11.036 | 78.27 tok/s |
| Docs | 63,483 | 15.428% | 84.572% | 6.038 | 42.82 tok/s |
| Aggregate | 190,449 | 8.116% | 91.884% | 9.141 | 64.83 tok/s |

The projection uses the RedHat 108.903 ms target screen plus 32.1 ms fixed
overhead and `E[L]=(1-alpha^16)/(1-alpha)`. It is provisional:
teacher-forced Dudeman top-1 is not a DFlash proposal trace, and row matches
are not independent. It is nevertheless strong evidence against shipping a
simple RedHat-target chain at 107.9 tok/s, especially on documentation.
Receipt: `redhat-vs-dudeman-top1-agreement.json` in the directory above.

The existing Mac Dudeman qualification accepted 239/239 DFlash proposals on
the standard 2,048→256 cell, and the later native target M16 cell accepted
30/30. That is useful lineage for dual Dudeman, not proof of broad or
cross-device 99.46% acceptance.

## Architecture attempts and disposition

This campaign did not stop at the first plausible design. The attempts below
were developed independently and then challenged across the distributed-state,
sampler, model-semantics, and hardware boundaries.

| # | Attempt | Evidence class | What it buys | Current disposition |
|---:|---|---|---|---|
| 1 | Optimize the existing Mac Dudeman target | Measured | No distributed target state and unchanged semantics | Rejected as the primary route: the retained M16 verifier is 227.864 ms GPU / 239.564 ms wall, so its full-accept ceiling is below 107.9 tok/s once drafting is included. Keep only as fallback and a larger-shape/tree experiment. |
| 2 | Second resident Dudeman verifier on GX10, linear chain | Authenticated composite import, sequential/block oracle, end-to-end traces | Preserves checkpoint/composite lineage and reuses DFlash | **Rejected for general serving.** The all-accept control reaches 110.59 tok/s, but docs/Python/Rust have only 9.23%/26.31%/38.07% proposal acceptance and verifier-only ceilings of 20.15/40.04/55.96 tok/s. Keep the machinery for the tree experiment. |
| 3 | Existing RedHat GX10 session as target, Mac Dudeman/DFlash as proposer | Algorithm + measured mismatch screen | No second checkpoint and the simplest target state | Algorithmically correct but a new RedHat-decode product lane. Direct chains are unlikely to meet the speed gate because retained RedHat/Dudeman top-1 agreement collapses on docs. |
| 4 | Shared-Gumbel RedHat target | Literature-derived + reference implementation | No q transfer; drafter-invariant target output and stronger containment of malformed q | Retain as an experimental sampler/policy. It does not preserve Dudeman semantics and gives up some maximal-coupling acceptance. |
| 5 | Sparse maximal coupling | Linear sampler plus V2 request replay | Dense-helper-equivalent linear verification with only a few KB of q evidence at top-k 40 | Default trusted-peer MVP. Actual MT state and each q draw are replayed; randomized K=40 target integration remains. |
| 6 | Stateless GX verifier with authenticated Mac log and prefix-cache soft state | Reconstructible V2 transcript/journal; service unimplemented | Crash recovery becomes replay; rejected branches need no semantic rollback; no durable remote model session is required | Preferred service architecture. Remaining work is GX RPC/model execution, parent-cut cache control, the real renderer, and snapshot compaction. |
| 7 | Persistent GX session actor with request-result journal | Durable V2 local state machine | Avoids prefix replay and works when verifier hooks cannot coexist with public prefix caching | Protocol transaction is implemented and fault-tested locally; persistent model-state integration remains a fallback. |
| 8 | One-seam Dudeman layer split between Mac and GX | Analytical estimate + literature | Keeps both machines doing target work and moves one `[M,6656]` boundary rather than 52 collectives | Contingency if full Dudeman on GX fails capacity or latency. Sweep cuts 2/14/26; do not assume a half split is balanced. |
| 9 | Larger Mac chain / Sequoia token tree | Analytical estimate + literature | Preserves current semantics without a Dudeman port | Bounded experiment only. Mac needs at least a 1.6–1.8× target-row throughput gain before branch waste, so it is a high bar. |
| 10 | Uncertainty-aware shared-Gumbel/Sequoia tree on GX | Literature-derived + measured batch curve | Uses otherwise idle batch arithmetic to cover near-tie cross-checkpoint branches | **Only remaining performance experiment.** Wide GX timing supplies a hardware rationale; target-winner rank, tree masks, branch coverage, and end-to-end throughput remain unmeasured. It is not a product claim. |
| 11 | RedHat GX drafter, Mac Dudeman target | Invariant rejection + estimate | No target identity change and checkpoint mismatch is harmless | Safe but rejected as the primary performance route: the Mac verifier ceiling remains. It can contribute branches to attempt 9. |
| 12 | Dudeman LM-head-only GX offload | Analytical estimate | Small protocol spike and one network seam | Useful to exercise fencing/closure, but the estimated ceiling is roughly 77–88 tok/s. Not a final route. |
| 13 | Certified Dudeman-vs-RedHat argmax interval | Research hypothesis | Could let RedHat certify only provably identical greedy steps | Research spike only. Bounds across 52 layers will probably become vacuous, and argmax certificates do not implement sampled decoding. |
| 14 | Cross-checkpoint KV translator, CRDT token-branch union, or RedHat certifying Dudeman | Rejected by invariant + published approximate systems | Appears to avoid a target implementation | **Rejected as unsafe.** Published translators are approximate; set convergence is not semantic composition; the wrong transition function cannot certify the target. |

## Frontier attempt: uncertainty-aware token trees

The key new opportunity is to spend GX arithmetic rather than network
bandwidth. An NVFP4 verifier is dominated by reading approximately one model's
weights at small batch sizes. A 32–64-node tree can therefore cost much less
than two or four 16-row chains.

Mac constructs a shallow tree under its Dudeman/DFlash distribution, branching
only at low-margin nodes. For the drafter-invariant version, each node ranks
children with the public Gumbel field. GX evaluates the tree with tree
attention in one pass, follows the unique authoritative target winner at each
level, commits the longest present path, and carries the first absent target
winner as the next frontier. Branches never vote and are never merged into the
ordered log.

For target-owned shared Gumbel, this can preserve the target marginal because
the GX target chooses its unique path; extra draft branches only increase the
chance that the chosen path is present, provided every node has the correct
ancestor mask, position, and absolute-ordinal field. This is a design argument,
not a tested implementation. Sparse maximal coupling cannot simply accept any
top-b sibling: multi-proposal residual mass and MT draw order need a
topology-specific proof and target-only property tests. The required experiment
is not generic beam search: it must measure target-winner rank under actual
DFlash proposals, construct a hardware-aware topology, and report emitted depth
per evaluated node.

The retained GX batch curve gives a strict admission screen before that kernel
is built. Charging measured five-capture target wall, 26.9 ms linear DFlash
work, 0.79 ms RTT/protocol overhead, and 0.273 ms of selected-path f16 wire time
per emitted row gives:

```text
rate = 1000 * E / (target_wall_N + 27.69 + 0.273 * E)
```

| Nodes | Target wall | Required mean emitted/call | Minimum path depth | Required node efficiency |
|---:|---:|---:|---:|---:|
| 24 | 114.115 ms (interpolated) | 15.77 | 16 | 65.7% |
| 32 | 117.870 ms | 16.18 | 17 | 50.6% |
| 48 | 125.321 ms (interpolated) | 17.01 | 18 | 35.4% |
| 64 | 132.772 ms | 17.84 | 18 | 27.9% |

This exposes an important topology constraint: a 16-deep tree cannot beat
107.9 tok/s on the conservative 32/48/64 capture curve even with perfect path
coverage. The tree must be deeper as well as wider. Forty-eight nodes is the
first sensible implementation target; 64 is justified only if its additional
16 nodes add at least 0.83 emitted token per call. Relative to the measured
linear traces, documentation still needs roughly a 1.85–2.35x node-efficiency
improvement at 64/48 nodes. That is plausible only if its early authoritative
mismatches are predominantly low-rank DFlash alternatives and their descendant
paths remain predictable. The current receipts contain no q-rank or descendant
branch data, so claiming a tree win now would be speculation.

## Remaining product blockers and exact next steps

These are deliberate merge boundaries for a research prototype, not deferred
serving details:

- no GX RPC binds V2 to a resident Dudeman engine, sparse target distributions,
  selected-layer captures, target KV rollback, or the composite importer;
- the bounded gateway's lease-free recovery reconciles external head CAS before
  real renderer activation. Production needs one owned recovery permit through
  activation, or an external Pending→Ready state that blocks child admission;
- `VerifiedMirrorCommitV1` is not connected to an engine consumer, while the
  research engine/bench can still construct a scalar target decision. The
  production adapter must be the only constructor before this boundary can be
  seal-eligible;
- the production authority registry and real idempotent KV/DFlash renderer are
  not implemented; local tests use explicit fake fence/renderer objects;
- client stream publication/resume is outside the journal, so exactly-once
  visible token delivery still needs an integration boundary after activation;
- automatic prefix-cache hits are not capped at the authenticated parent cut;
- sparse q is consistent with the authenticated Mac's supplied weights and MT
  draws, but is not attested as output of the declared draft model;
- the retained commit chain has no authenticated snapshot compaction or
  multi-GX consensus/failover path; admission history also reaches its bounded
  hard stop at roughly 4,096 rounds without GC;
- sampled target rows, every rejection index, target KV/capture cuts, and broad
  randomized K=40 equivalence have not been exercised against GX model output;
- the measured linear accepted-prefix distributions fail the economics gate on
  all three organic strata; the simple chain is closed rather than awaiting
  more tuning;
- no prompt-only classifier has a safe admission proof. Rust fully passes the
  first Mirror round and fails the second, while a later failure after visible
  output cannot silently switch from the named GX target engine to the Mac
  target engine without defining and requalifying new observable semantics.

The next performance attempt is deliberately narrow:

1. Record the rank of each authoritative first-mismatch token under the actual
   DFlash distribution for standard, docs, Python, Rust, and the 24-task golden
   set. Follow the authoritative non-top-1 branch and record descendant ranks;
   first-edge recall alone is insufficient. Preregister top-2/top-4/top-8
   path-coverage thresholds before building serving code.
2. Implement one target-owned shared-Gumbel tree policy first. Bind absolute
   output ordinals, ancestor masks, positions, candidate-DAG root, and the
   carried frontier; prove target-only equivalence under arbitrary tree shape
   and packet order. Do not generalize linear sparse maximal coupling to trees
   without a separate residual proof.
3. Offline-simulate frozen 24/32/48/64-node topologies and reject any size whose
   mean emitted path on any organic stratum is below 15.77/16.18/17.01/17.84.
   Implement 48 first; advance to 64 only if the extra nodes add at least 0.83
   token/call. The final gate is paired end-to-end emitted tokens per wall
   second, not node throughput. Each organic stratum must exceed 107.9 tok/s
   with a preregistered lower confidence bound; otherwise reject trees too.
4. If tree coverage fails, keep the shipped kquant speculative lane and native
   plain NVFP4 lane. An adaptive fallback may select a lane before generation,
   but it must not change target authority after publishing tokens. Buffering a
   speculative probe or maintaining a fully qualified Mac target shadow are
   separate product policies with explicit TTFT/memory/quality gates.

## Product promotion gates

The checked-in work is useful only as a bounded research scaffold while these
gates are pursued. The shipped fail-closed behavior remains in force until
every safety gate and one performance route passes.

1. **Composite-genesis and target-oracle exactness.** The actual
   RedHat-kvpack→vLLM Dudeman import and first 2,047-row packet are implemented;
   Dudeman prompt replay is no longer the timing or oracle genesis. Name the GX
   target engine as its own epoch. Against a non-speculative GX
   target-only oracle, match probability/sampler output, five target captures,
   target KV deltas, rollback cut, and next step at every rejection index.
   Separately measure Mac↔GX equivalence before claiming today's Mac observable
   semantics. Include 2k/8k/32k and SWA wrap.
2. **Fused verifier service.** V2 now provides the transcript, actual sampler
   state, terminal state, payload closure, durable PREPARED/commit/activation,
   ACK, retention, and resumable GC primitives. Connect them to native Dudeman
   residency, bounded target top-k materialization, block/tree candidates,
   selected-layer/KV streaming, a real authority fence and renderer, client
   emission, and authenticated parent-cut prefix caching. Do not infer service
   latency from the Python screen.
3. **Real acceptance.** Linear 256-token documentation/Python/Rust runs and a
   512-token all-accept control are complete and reject the chain. Before any
   tree implementation is promotable, add the 24-task agentic set and report
   target-winner rank plus the whole accepted-path histogram for candidate
   trees 24/32/48/64; do not replace it with an IID estimate or mean token
   agreement.
4. **End-to-end economics.** At contexts 2k/8k/32k and sessions 1/2/4, measure
   p50 and p95 verifier, draft, capture, transport, install, and effective
   output rate against a paired contemporaneous 107.9-lane control. Preregister
   a lower confidence-bound gate, not a point estimate. The competitive gate
   is greater than 107.9 tok/s; the minimum useful fallback is greater than
   35.5 tok/s. Account for both resident checkpoints and model-native KV.
5. **Semantic and failure qualification.** Rerun D2 and the P-series exact
   seams. Actual q-draw replay is covered in V2; add randomized K=40
   dense/sparse properties, every rejection index, zero/tiny mass, final RNG
   state, and tree-mask/
   residual proofs. Inject lost, duplicated, corrupted, and reordered
   fragments; crash at PREPARE, compute, journal, staged render, head CAS,
   activation, reply, and stream emission; test concurrent parents, stale
   terms, key rotation, retry after expiry, EOS at every position, cancellation,
   max-token closure, and context overflow. Assert one visible child and no
   unprocessed frontier in KV or output.
6. **Authority hardening.** V2 separates log-writer and target-executor
   identities, uses Ed25519 target-only result signatures, and calls a live
   fence at admission/PREPARED/commit. Connect that interface to the production
   registry and stream publication. Multi-GX failover requires an ordered
   consensus head; Merkle/CRDT repair remains payload-only. Sparse q from an
   untrusted proposer additionally needs attestation/recomputation or a
   target-owned policy.

Reasonable rejection is justified if Dudeman cannot reside beside RedHat with
the required 32k/session capacity, if the fused M16 target plus measured fixed
cost exceeds `E[emitted]/107.9`, if broad acceptance is below the derived
threshold even with a hardware-efficient tree, or if composite-genesis
next-step equality cannot be made exact. In that event, keep kquant
speculative, retain native non-speculative serving, and publish RedHat-target
speculation only as a separately named and requalified lane if it clears its
own quality/performance gates.

## Research basis

- Lossless speculative decoding: [Leviathan et al.](https://proceedings.mlr.press/v202/leviathan23a.html)
  and [Chen et al.](https://arxiv.org/abs/2302.01318).
- Communication-free coupling: [Daliri, Musco, and Suresh](https://arxiv.org/abs/2408.07978).
- Hardware-aware token trees: [Sequoia](https://arxiv.org/abs/2402.12374) and
  [SpecInfer](https://arxiv.org/abs/2305.09781).
- Distributed speculation and pipelining: [DSI](https://proceedings.mlr.press/v262/timor24a.html),
  [FlowSpec](https://arxiv.org/abs/2507.02620), and
  [PipeSpec](https://arxiv.org/abs/2505.01572).
- Disaggregated prefill/decode and KV movement: [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)
  and [Mooncake](https://www.usenix.org/conference/fast25/presentation/qin).
- Approximate cross-model KV reuse, explicitly unsuitable for the exact lane:
  [DroidSpeak](https://www.usenix.org/conference/nsdi26/presentation/liu-yuhan).
- Exactly-once RPC and ordered authority: [RIFL](https://web.stanford.edu/~ouster/cgi-bin/papers/rifl.pdf)
  and [Raft](https://raft.github.io/raft.pdf).
- Out-of-order immutable reconstruction: [Baquero and Brito](https://arxiv.org/abs/2607.01308),
  [Merkle-CRDTs](https://arxiv.org/abs/2004.00107), and
  [delta-CRDTs](https://arxiv.org/abs/1603.01529).

The local Baquero program and transport notes used in this campaign are
`../kvpack/docs/DISTRIBUTED_KVPACK_PROGRAM.md` and
`../kvpack/docs/DISTRIBUTED_CONSISTENCY_AND_REPLICATION.md` relative to this
checkout's parent directory.
