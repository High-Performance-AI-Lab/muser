# Disaggregated prefill: the GB10 → Mac lane

## Roles, not machines

"Producer" and "consumer" are **roles**, not machines:

- the **producer** is the process prefill runs in;
- the **consumer** (receiver) is the process that decodes.

The handoff protocol, identity binding, and kvpack state format are defined
between those two roles — not between vendors or hosts. A producer on the
same machine (a second process over loopback), or on any other hardware, is
the same protocol with a different placement. What today's qualified
deployment adds is a *placement*: a remote NVIDIA producer, because
tensor-core NVFP4 prefill and unified-memory decode each win on different
silicon. The placement is a technicality; the role split is the
architecture. (Colocated producers are not yet qualified — see
[`docs/release-todo-20260823.md`](release-todo-20260823.md) §10.)

## The idea

Big-prompt inference is two different jobs stapled together. **Prefill** —
processing the whole prompt into KV cache — is a throughput job that loves
tensor cores and FP4 arithmetic. **Decode** — emitting tokens one at a time —
is a latency job that loves unified memory close to the user. Doing both on
one chip means each job fights the other for memory and compute.

muser's disaggregated lane splits them along the line that matters:

```text
   ┌────────────────────────┐         kvpack handoff          ┌──────────────────┐
   │  producer role         │  ─────────────────────────────► │  consumer role   │
   │  (today: GB10 node,    │   authenticated TLS + HMAC,     │  (today: Mac,    │
   │  vLLM, NVFP4 weights)  │   replay ledger, fail-closed    │  muser, Metal)   │
   │  prefills your prompt  │                                 │  decodes         │
   └────────────────────────┘                                 └──────────────────┘
```

The producer prefills your 130k-token prompt in NVFP4 on tensor cores,
packs the resulting KV cache, and ships it to the consumer, which picks up
decoding exactly where prefill left off. Everything after the handoff is
ordinary muser decoding. DFlash is available on the combined kquant lane;
the native/text lane rejects a DFlash identity by contract.

## What you get

Time-to-first-token falls 3.75–4.26× at every measured depth, because
NVFP4 prefill on tensor cores simply outruns anything a Mac can do locally
on a long prompt, even accounting for shipping the KV over the wire:

| Depth | Local TTFT | Remote TTFT | Payoff |
|---:|---:|---:|---:|
| 2,048 | 6.48 s | 1.52 s | 4.26× |
| 32,768 | 114.3 s | 30.5 s | 3.75× |
| 130,815 | 570.1 s | 137.4 s | 4.15× |

Full tables, CVs, and link floors: [`benchmarks.md`](benchmarks.md). Output
is deterministic across reps, and reuse makes the common cases nearly free:
a warm prefix answers in **0.61–1.06 s**, and a delta handoff moves **54.2851%
of the bytes** for a bit-identical result ([`kvpack.md`](kvpack.md)).

## What you need

- **A producer node**: aarch64 + NVIDIA GPU (GB10-class in the lab), NVIDIA
  driver, Docker, SSH with key auth. muser's **Add node** wizard
  (dashboard, or `muser node add user@host`) does the whole pipeline:
  preflight, pinned runtime deploy, lane-specific model acquisition with
  SHA-256 verification, enrollment key provisioning, producer start, and the
  lane-declared three-handoff qualification recipe before the node is marked
  healthy.
- **A wired 10GbE path for the measured numbers.** The reported matrix used
  a path with a ~9.4 Gbps raw single-stream ceiling. Re-prove the current
  route after any topology change. WiFi works for management but is never a
  measurement path.
- **EEE disabled on that link** — and this is measured, not superstition.
  With Energy-Efficient Ethernet active, the deep-payload burst after LPI
  idle produced discrete ~6.4 s retransmission blackouts and per-rep link
  rates collapsing to 0.68–1.73 Gbps. Disabling EEE on the direct link is
  enrolled production guidance, verified before and after every campaign.

Then serve with remote prefill:

```sh
MUSER_CROSS_VENDOR_QK=1 MUSER_GGML_METALLIB=/path/to/llama.metallib \
  muser serve \
  --model ~/.muser/models/muse-glimmer-30B-kquant-17gb.gguf \
  --prefill remote \
  --cluster-config ~/.muser/nodes/<name>/cluster.json
```

`MUSER_CROSS_VENDOR_QK=1` pins the cross-vendor math route the CUDA producer
bakes in — the Mac must derive Q/K exactly the way the producer did, or the
KV is foreign by construction, and the receiver refuses it.

## Correctness: what "correct" means here

1. **Token-level:** disaggregated cells are deterministic across reps and
   differentially checked against the local lane; the spec lane is
   token-exact against llama.cpp's own draft-dflash route.
2. **Quality at depth:** native-vs-kquant perplexity and calibrated
   top-token gates pass on code and prose at every measured depth. One
   content class (high-entropy documentation text) exceeds its band at
   65,536 on the NVFP4 route — published as a content-local sensitivity;
   the kquant lane is the reference route and remains selectable.
3. **Adversarial:** stale replay generations, identity-mismatched configs,
   and tampered manifests are refused end-to-end on live hardware — the
   refusal itself is the passing test. See [`kvpack.md`](kvpack.md) §Security.
4. **Stability:** eight consecutive 130,815-token handoffs ran with zero
   producer deaths and deterministic output.

## Operating it

- The producer **fails closed** (exit 75) on any engine-touched error — it
  never serves degraded state. A supervised restart ritual cleans the
  startup receipt, RoPE cache, and socket, waits for readiness, and latches
  off after three consecutive failed starts. A killed producer was detected
  and recovered with no operator action in testing, resuming bit-identical
  payloads.
- Every handoff runs against a replay ledger: each request carries a fresh
  generation number above the ledger's high-water mark; anything below it is
  a replay attempt and is refused.
- Diagnostic tooling (raw link ceiling, fsync-tail probes, per-rep phase
  reports, restart/supervision) lives in `scripts/gx10/` — see
  `scripts/gx10/README.md`.

## Honest limitations

- **One producer placement qualified today**: one remote NVIDIA producer per
  consumer, no multi-producer scheduler. The roles themselves are
  host-agnostic — colocated or same-vendor producers are future placements,
  not architecture changes. Scale-out is roadmap.
- **Multimodal requests use local prefill** — remote-multimodal handoffs
  are not qualified, and the server says so by falling back rather than
  pretending.
- **The link matters.** Over a shared/worse-than-10GbE path your payoff
  shrinks toward the wire; the merit gates (≥3 Gbps payload floor, ≤2% TTFT
  CV) exist so a degraded link fails visibly instead of quietly.
- The GB10 is a **prefill and storage node, never a decode destination**;
  remote speculative decoding was measured and rejected
  ([`nvfp4-distributed-speculative-frontier-20260818.md`](nvfp4-distributed-speculative-frontier-20260818.md)).
