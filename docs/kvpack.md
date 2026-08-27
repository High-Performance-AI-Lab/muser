# kvpack: exact KV state, moved safely

kvpack is the format and protocol family that moves a prefilled KV cache
between producer and consumer — the thing that makes disaggregated prefill
([`disaggregated-prefill.md`](disaggregated-prefill.md)) more than a
benchmark trick. The stance is simple: **exactness is the product; speed is
the consequence.** Restored state is proven byte-identical to what was
saved, and proven to carry the exact computation identity it was produced
under — the artifact digests, tokenizer, template, adapter, and math route
that generated it. A receiver installs state only from the producer
identity it enrolled with, and whenever any of those proofs fails, the
state is refused — loudly. A minimal, hash-pinned snapshot of the
upstream workspace lives at `third_party/kvpack` with recorded provenance
(`python3 scripts/audit_vendored_kvpack.py`); muser's adapter, transport,
and security model are in-tree. The upstream project lives at
[High-Performance-AI-Lab/kvpack](https://github.com/High-Performance-AI-Lab/kvpack).

## Why it exists

A 65k-token prompt's KV cache is hundreds of megabytes; a 131k prompt's is
nearly a gigabyte. Moving or reusing that state between machines and
processes raises three questions, and kvpack answers each mechanically
rather than by convention:

1. **Is this cache the cache I asked for?** — keyed, fail-closed identity.
2. **Have I seen this request before?** — replay protection.
3. **Did anyone touch it in flight?** — layered integrity and authenticated
   sealing.

## The reuse ladder

Each rung works on more machines and costs less, and all of them are
measured (5-rep conventions, retained receipts; see
[`benchmarks.md`](benchmarks.md)):

| Rung | What happens | Result |
|---|---|---|
| **Warm prefix, one Mac** | Local resident/durable cache answers from installed state | **~65 ms** to resume a shallow warm prefix |
| **Warm prefix at depth, remote producer** | Producer holds the exact prefix; no compute, no transfer | **0.613 s** at 65,536 tokens (vs 68.6 s cold), **1.057 s** at 130,815 (vs 147.8 s), bit-identical output |
| **Delta handoff** | Only the missing suffix crosses the wire | **54.2851%** of full payload bytes; output SHA-256 exactly equal to a full handoff |

The miss controls matter as much as the hits: an unrelated 8,192-token
prompt through the same path stayed valid (~12.9 s) — the fast path is
genuine reuse, not a cache that answers everything.

## What moves

### Full handoff

The producer packs the KV cache with the identities that generated it —
model, revision, quantization, adapter, tokenizer, chat template, context
policy, engine ABI — and the receiver installs it only after those
identities match its own exactly.

Across the disaggregated boundary this is deliberately *relational*, not
self-identical: producer and receiver are different engines on different
hardware, so what the receiver checks is "is this the exact enrolled
producer?" — and what makes installing that foreign-runtime state sound is
not trust but the pinned cross-vendor math route, qualified by a three-way
Metal/CPU/CUDA oracle that produces identical digests over hundreds of
millions of inputs, plus a shared, pinned representation schema.

### Delta handoff — pay for what's new

When the receiver (or producer-side cache) already holds an exact prefix,
only the suffix crosses the wire:

| Arm (65,536-token request, 32,768 held) | Payload bytes |
|---|---:|
| Full handoff | 954,190,848 |
| Delta handoff | 517,983,232 — **54.2851%** |

Admission is fail-closed: the delta span must sit against the session's
*exact* held prefix (256-aligned cut, nonempty suffix, span-schedule
match).

### Warm reuse — don't ship anything

When the producer's radix cache already holds the exact prefix, no producer
compute and no transfer happen at all; the receiver answers from its own
installed state (table above).

## The security model, in full

### Data plane — what kvpack itself enforces

- **Fail-closed identity.** The cache identity binds eight runtime inputs
  (model SHA-256, revision, quantization, adapter ID, tokenizer SHA-256,
  chat-template SHA-256, context-policy SHA-256, engine ABI) into a keyed
  namespace; the prefix ID is a keyed HMAC over that namespace, the exact
  token count, and every token. Any difference is a miss or rejection —
  never a best-effort restore that might be subtly wrong.
- **Layered integrity.** Record headers and payloads are SHA-256 hashed,
  object IDs are content-derived, the terminal commit carries an ordered
  inventory, a canonical Merkle root binds it, and the footer seals the
  header digest plus every byte of the file. Truncation, single-bit flips,
  reordering, substitution, and length games are rejected — verified by
  exhaustive truncation and bit-flip conformance corpora across Rust,
  Python, and C99 reference implementations that produce byte-identical
  packs.
- **Crash-safe publication.** Packs are append-only immutable files; the
  commit is written last; pack sets publish through an exclusive atomic
  rename. A torn write or SIGKILL mid-write (injected in tests) can never
  replace the last known-good generation.
- **Privacy by construction.** Cache keys are keyed HMACs; raw prompt text
  never appears in paths or telemetry; token blocks stay inside the pack
  because a restorer must prove the prefix bytes.
- **Optional authenticated encryption at rest.** The `kvenc` envelope
  (ChaCha20-Poly1305, HKDF-derived keys) is the boundary against an
  attacker who can rewrite an entire pack and its metadata.
- **Bounded, abortable restore.** Positional bounded reads re-verify each
  payload into caller-owned buffers; memory admission caps allocation; a
  failed restore aborts and resets engine state rather than leaving
  partially installed cache live.

### Transport — what muser's Handoff V2 adds on top

- **Authenticated channel**: mutual TLS with HMAC-sealed manifests; every
  segment verified before admission.
- **Replay ledger.** Every remote request carries a generation number; the
  receiver's durable ledger refuses anything at or below its high-water
  mark. A stale or duplicated request never reaches the decoder.
- **Cross-vendor math binding.** The receiver only installs KV produced
  under the cross-vendor math route it shares with the producer
  (`MUSER_CROSS_VENDOR_QK`) — foreign-derivation KV is rejected by
  construction, not by hope.
- **Fail-closed producer.** The producer exits (75) on any engine-touched
  error rather than serving suspect state; a supervisor performs the full
  restart ritual and latches off after three consecutive failed starts. A
  killed producer was detected and recovered automatically in testing and
  resumed bit-identical payloads.

### Proven live — refusal receipts, not claims

Each of these was demonstrated against real hardware, and the refusal
artifacts are retained in the campaign ledger:

- **Stale generation refused.** A request fired exactly one generation
  below the live watermark received the explicit stale/replayed refusal; a
  fresh generation one above it served normally.
- **Foreign identity refused.** A well-formed config with a flipped
  adapter digest was rejected with an explicit identity-mismatch error — a
  tampered config cannot smuggle foreign KV into a receiver.
- **Nothing installs silently.** Receipts bind every attempt — command,
  exit status, retained log — including the refused ones. A producer
  timeout is recorded as *invalid evidence*, never counted as a refusal.
- **Stability under load.** Eight consecutive 130,815-token handoffs ran
  with zero producer deaths and deterministic output.

The honest threat-model line: pack hashes prove integrity, not
authenticity — a whole-file rewrite attacker needs the `kvenc` envelope or
the transport-layer authentication above. The specs are equally explicit
about what kvpack does *not* guarantee; engines remain responsible for
proving restored bytes are correct state for their runtime.

## Economics

kvpack also models when moving KV is worth it versus recomputing: transfer
cost scales with payload bytes, recompute with prompt tokens and producer
throughput. The model and measured crossover analysis live in
[`kvpack-economics.md`](kvpack-economics.md); the ladder above is the
practical outcome — at deep prompts, reuse in any form beats recompute by
orders of magnitude, and a cold handoff's wire cost is amortized by the
3.75–4.26× TTFT payoff of NVFP4 tensor-core prefill.
