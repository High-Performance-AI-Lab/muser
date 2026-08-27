# Muser architecture

This document describes the current recovery tree. It is not a launch claim
or a release receipt. The executable release boundary is
`release/feature-contract-v1.json` and the closed finding record is
`release/findings-v1.json`. `release/release-lock.json` keeps sealing,
candidate creation, and publication disabled; it permits only the exact
non-release `v0.1.0-beta.1` marker tag after a separate operator go.

Historical Ferrite extraction provenance is retained separately in
`docs/extraction-manifest.md`. Muser has no Ferrite runtime dependency.

## Product boundary

Muser v0.1 serves one pinned Muse Glimmer model. Text, vision, Metal DFlash,
GX10 prefill/storage, the dashboard, logical
sessions, and decode/storage migration are in scope. LoRA, model routing or
hot-swap, infill, reranking, Responses and Anthropic APIs, built-in tool
execution, and llama.cpp's Web UI are out of scope.

The compatibility reference is llama.cpp commit
`89e0aa6fd362617d9073e0dafc18e41241521572`. Compatibility claims are limited
to `release/llama-server-compat-v1.json`; security policy, identifiers, clocks,
paths, timings, build fingerprints, and documented Muser metrics may differ.

## Model and engine

The GGUF is the source of runtime geometry and identity. Startup validates the
pinned revision, byte size, and SHA-256 before loading. The tokenizer identity
is a canonical hash of the relevant GGUF metadata and the chat-template
identity is a hash of the exact 7,167-byte GGUF template. A path override
changes location, never model identity.

The fixed text graph has 52 layers in the repeating
`[sliding, sliding, sliding, full]` pattern:

- 39 sliding-attention layers use interleaved-pair RoPE and a 2,048-token KV
  ring;
- 13 full-attention layers use no positional rotation and retain growing KV;
- GQA geometry is 32 query heads to 2 KV heads with head dimension 128;
- attention output uses the model's sigmoid gate before `o_proj`;
- the two post norms use the llama.cpp graph constant `1e-8`; other norms use
  the GGUF epsilon;
- final logits apply `1/sqrt(26)` before the `tanh` soft cap at 20.

`muser-engine` contains an independent CPU f32 oracle plus the Metal
prefill/decode implementation. The target and DFlash engines expose prepared
state handles; a combined remote install prepares and verifies both before an
infallible publication step. If rollback or accelerator state becomes
uncertain, the engine latches unhealthy and serving returns 503 until restart.

The official vision path projects image embeddings through the pinned
50-block `mtmd` package and inserts those rows into the text context. DFlash
drafts are always verified by target distributions. The public-CoreML ANE
route is retained as an explicit experimental post-release path. It is absent
from v0.1 qualification, sealing, identity, and candidate contents, and auto
routing is permanently Metal for v0.1.

## Slots and scheduling

One scheduler owns one accelerator and between one and four resident slots.
`--parallel` accepts `1..=4` and defaults to four on the release configuration.
Each slot owns independent target KV, DFlash state, logits, RNG, sampler and
grammar state, detokenizer/stop state, and cancellation state. Immutable
weights, Metal pipelines, and the DFlash executor are shared. Experimental ANE
`MLState` exists only when that backend is explicitly selected.

Admission is bounded at 64 waiting requests. Decode is favored over prefill;
ready decode rows rendezvous into submissions of up to four sequences, while
prefill is chunked. Tokenization, sampling, grammar/tool parsing, disk, TLS,
and socket writes stay outside the accelerator owner. Output channels are
bounded: disconnect cancels the request, and a continuously blocked writer is
cancelled without holding the accelerator.

There are at most four serving slots. Context shift and restore use a staging
generation and publish only after the replacement state is complete; staging
is not a fifth concurrently serving slot.

## Context and sessions

The model limit is 131,072 positions per slot. Context policy is `shift` by
default or `error`. Chat shifting preserves system content and whole newest
turn/tool/image units. Raw shifting preserves the configured prefix plus the
newest suffix. A request is rejected if the minimum retained unit plus output
reserve cannot fit.

At most 64 logical sessions are tracked. Stateful generation requires a
session ID, expected revision, and `Idempotency-Key`; revision conflicts or a
busy session return 409. Authenticated encrypted bundles bind exact model,
tokenizer, template, layout, and state identities together with target/DFlash
state, sampler state, replay messages, vision rows, context epoch, and
revision.

Migration is two phase. Decode-node copy/move uses authenticated HTTPS between
identically qualified Muser decoders; storage-tier copy/move uses enrolled
kvpack storage. The destination durably commits before a move can delete the
source, and transfer status is idempotently queryable after ambiguous failures.
GX10 is not a decode destination.

## HTTP and security boundary

`muser-server` uses Tokio, Axum/Hyper, and rustls. Request framing is handled
by the framework and the service applies strict JSON content types,
route-specific body limits, deadlines, bounded connection/body budgets, and
bounded output channels.

Authorization is deliberately asymmetric:

- loopback inference is keyless;
- loopback management needs bearer auth or a same-origin dashboard session;
- dashboard login exchanges the API key for a Secure, HttpOnly,
  SameSite=Strict cookie, and cookie-authenticated mutations additionally need
  an exact Origin match and CSRF token;
- a nonloopback bind is refused before listening unless a certificate,
  mode-0600 private key, and mode-0600 API-key file are supplied;
- every LAN inference, telemetry, WebSocket, session, cancellation, and node
  management request needs authentication.

Default CORS is none. The dashboard is same-origin and has no `file://` or
cross-origin endpoint override. WebSocket authentication uses either the
dashboard cookie or a 30-second single-use ticket; long-lived bearer keys are
not placed in URLs. `muser tls init` and `muser tls issue` implement the
separate local-CA operator workflow with explicit SANs.

## Public serving surface

The active router implements the frozen llama-compatible completion, chat,
tokenizer/template, embedding, slot, model/property, and health routes; the
Ollama-compatible `/api/generate` and `/generate` aliases; logical session and
migration routes; `/snapshot` JSON; `/metrics` Prometheus text; authenticated
`/stream` WebSocket telemetry; and temporary `/telemetry` SSE keyframes.
Request DTOs reject unknown fields. Intentional rejections are listed in the
compatibility contract.

The dashboard reads only live process state. Fields without a measurement are
tagged `mock` and rendered unavailable. `tricks[]` remains empty until an
optimization has a qualifying release packet; historical Ferrite results are
never inserted as live Muser measurements.

## Durable and remote KV

The audited kvpack source is vendored under `third_party/kvpack`; Cargo ignores
sibling checkouts. `muser-kvpack` provides the Muse layout and accounting
adapter for resident and authenticated durable reuse.

GX10 Handoff V2 uses mutually authenticated TLS plus an HMAC-sealed manifest.
Enrollment generates each TLS private key on the machine where it remains;
the HMAC is a shared secret transferred over known-host-verified SSH.
Replay admission durably reserves the generation with file and directory
fsync before target+DFlash publication and ACK. Any durability failure
degrades the route until repair and restart.

The v0.1 topology is one Mac decoder and one Spark/GX10 producer. The transport
is mTLS over TCP, not RDMA/RoCE. Qualification is a property of the enrolled
producer identity. The combined kquant lane requires three ordered exact
target-plus-DFlash repetitions at a 2,048-position prompt and 256 generated
tokens. The native tensor-core NVFP4 lane requires three ordered text
repetitions under its exact-token and declared bounded-logit policy, with no
DFlash identity. The integer-dot producer remains available through
`MUSER_NVFP4_EXACT=1` as the native lane's verification anchor. Exact and
native producers derive different target-cache identities; an unknown recipe
is refused at enrollment. Multi-producer scheduling and node discovery are not
implemented.

Both declared recipes passed the complete CLI wizard live on 2026-08-24:
native/text in attempt 9 and combined target-plus-DFlash in attempt 31. Every
run completed three ordered handoffs and the healthy transition, and the
canonical resident was restored afterward. Native NVFP4 speculative decode
remains fail-closed under Fallback B; plain NVFP4 decode serves the native lane
and speculative serving remains kquant-only.

The shipped lane matrix is:

| Lane | Prefill | Decode | Speculative | Intended use |
|---|---|---|---|---|
| Native NVFP4 | Spark tensor-core FP4 | Mac NVFP4 weights with FP16 KV, 35.491 tok/s | Rejected (fail-closed) | Fast product lane |
| kquant/reference | Reference path | kquant, 35.440 tok/s | 107.9 tok/s | Speculative and explicit reference lock |
| Exact NVFP4 flag (`MUSER_NVFP4_EXACT=1`) | Integer-dot verification producer | Mac NVFP4 | Verification only | Deterministic anchor |

Native decode is parity-within-noise with kquant, never claimed faster. The
native lane serves without a context cap through the measured 32k range; the
published sensitivity is confined to high-entropy numeric/digest/tabular
documentation content, and the kquant lane is the explicit reference lock for
that class.

The experimental distributed-verifier design no longer assumes checkpoint
unification. It imports the same RedHat-produced prefix KV genesis into
independent Mac and GX10 Dudeman continuation renders, keeps an authenticated
token/frontier log as canonical state, and uses content-addressed set
convergence only for out-of-order payload reconstruction. The direct prototype
proved the exact 2,047-row external cache cut, authenticated early f16 feature
streaming, carried-frontier commit, and exact Mac DFlash rollback. Its linear
policy is nevertheless rejected: the all-accept control reached 110.59 tok/s,
while docs/Python/Rust reached only 15.53/11.17/15.41 tok/s; their physically
impossible zero-Mac-cost verifier ceilings are 20.15/40.04/55.96 tok/s. The V2
typed protocol carries the full token transcript and MT state, replays sparse
proposal draws, has terminal transitions and GX-only result signatures, and
implements a durable PREPARED/staged-render/WAL/activation/ACK transaction.
The production authority, renderer, executor, and stream boundary remain
unwired, so this research does not change the fail-closed lane. See
`docs/nvfp4-distributed-speculative-frontier-20260818.md` for the alternatives,
hardware screens, and promotion gates.

## Historical performance diagnosis

An earlier NVFP4 engineering packet measured 3.881 s cold disaggregated TTFT for
a 2,048-token prompt, including 1.87 s native producer compute, versus
approximately 6.5 s local Mac serving prefill. Warm kvpack prefix reuse is
64.631 ms. Plain Mac NVFP4 decode is 35.491 tok/s, parity within noise with
the adjacent 35.440 tok/s kquant measurement. These are dated engineering
results with the qualification boundaries recorded in
`docs/nvfp4-fast-lane-evidence-20260817.md`; the current public matrix and
receipt scopes are in `docs/benchmarks.md`.

The paragraph below is retained as the earlier kquant/llama diagnostic, not as
the current disaggregated product baseline.

A single non-notarial 2,048-prompt/256-output comparison against pinned
llama.cpp measured exact generated tokens, Muser prefill at 6,432.152 ms versus
6,663.143 ms (about 3.6% faster), and Muser decode at 9,795.246 ms versus
7,653.574 ms (about 22% slower by throughput ratio). This is an engineering
sample, not a qualified product result or launch claim.

The bounded one-token phase diagnostic attributes the production/legacy gap to
104 separated exact norm-boundary closure groups, 39 SWA staging groups, 52
KV-publication/attention closure splits, and one bookkeeping copy. Only the
copy has been removed bit-exactly; the available 104-group norm fusion changes
logprobs beyond contract and is rejected. Full numbers and the instrumentation
limits are in `docs/decode-dispatch-gap-20260815.md`.

## Release evidence

Ordinary tests and qualification receipts do not authorize release. The
release flow freezes one clean identity, runs all mandatory lanes unsealed,
creates readiness only at zero open findings, then freshly reruns the full
matrix into one atomic seal bundle. Any post-seal source, dependency,
artifact, configuration, documentation, binary, or candidate change
invalidates the campaign. Tagging and publication remain owner actions after
both candidate verifiers pass.

See `docs/private-release.md` for the procedure and
`docs/launch-claims.md` for the only language eligible for launch copy after
the corresponding final evidence exists.
