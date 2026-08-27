# Benchmarks

Every number muser publishes, its methodology, and where its receipt lives.
The authoritative record is the campaign ledger
[`goal-parity-ledger-2026-08.md`](goal-parity-ledger-2026-08.md) — this page
is its public summary. Where a number below has a caveat, the caveat is part
of the claim.

## Methodology

- **Comparator:** a source-pinned llama.cpp build (commit
  `89e0aa6fd362…`, `flash_attn_ext` prefill route) with the same pinned
  model and, in speculative lanes, llama.cpp's own `draft-dflash` route.
- **Ratios are `llama ÷ muser`** — above 1.0 means muser is faster.
- **Exactness is a gate, not a hope:** every performance rep is also a
  token-exactness check on synthetic fixtures. Cells that fail exactness are
  not reported as passes.
- **Repetitions:** five counted reps per cell after conventions (one
  discarded warmup handoff in disaggregated cells; 60 s cooldowns in spec
  matrices), means reported with coefficient of variation (CV).
- **Hardware:** one Apple Silicon Mac (decode) + one GB10 node (remote
  prefill), wired 10GbE point-to-point with EEE disabled per measured
  ruling. Remote-payload floors are gated at ≥3 Gbps; TTFT CV at ≤2%.

## 1. Plain local decode and prefill (no speculation)

Five reps per depth, exact-token every cell, zero failures
(ledger: "Phase 2 non-spec context matrix", 2026-08-20).

| Depth | Decode | Prefill |
|---:|---:|---:|
| 2,048 | 1.0504 | 1.0397 |
| 8,192 | 1.0429 | 1.0208 |
| 16,384 | 1.0414 | 1.0185 |
| 32,768 | 1.0479 | 1.0171 |
| 65,536 | 1.0274 | 1.0163 |
| 131,008 | 1.0277 | 1.0139 |

Plain NVFP4 local decode measured 35.49 tok/s versus kquant's 35.44 —
parity within noise, never claimed faster.

## 2. DFlash speculative decode (local)

Measured at the window-fixed binary. The synthetic matrix ran at
verify-length 15 (pinned by the comparison harness); serving defaults to
verify-length 7 — the frozen tuning choice, for the natural-text reasons
below. Reps: 5/5 exact per depth at 2,048, 16,384, 32,768, and 131,008; the
† cells (8,192, 65,536) are single-rep diagnostics. Decode ratios are
per-round speedups; **wall clock is the accounting-invariant column**
(ledger: "Spec re-measurement at the fixed window" 2026-08-21; "Deep
synthetic restatement" 2026-08-22; "Spec-prefill funded-fix requalification"
2026-08-23).

| Depth | Decode | Prefill | Wall |
|---:|---:|---:|---:|
| 2,048 | 1.2369 | 1.0083 | 1.0711 |
| 8,192 | 1.214† | — | 1.022† |
| 16,384 | 1.2032 | 1.0066 | 1.0160 |
| 32,768 | 1.1962 | 1.0020 | 1.0069 |
| 65,536 | 1.188† | — | 1.006† |
| 131,008 | —* | 1.0246 | **1.0254** |

\* At 48 output tokens the decode phase is four speculative rounds, and the
phase boundary is asymmetric across engines — muser's decode excludes its
first verified round while llama's eval includes its first. No short-output
decode figure in this series is an accounting-neutral per-round speedup
(ledger amendment, 2026-08-23). The funded prefill fix's robust headline is
the **131,008 wall crossing parity for the first time**: 0.9768 → 0.9840 →
1.0254 across the fix lineage.

**Natural text — the honest edges.** On real corpora (not synthetic
fixtures), cross-engine outputs diverge, so speed stands without an
exactness gate. Spec decode wins on python-like content (16,384: 1.186;
8,192 suffix: 1.321) and **loses on high-acceptance shallow text** (rust at
2,048: 0.931, improving to 0.945 at verify-length 7), where llama's lighter
draft wins. This asymmetry is why the serving verify-length was frozen at 7:
it is the best decode and the most robust acceptance on natural text
(ledger: "Verify length is the wrong default", 2026-08-21).

## 3. Disaggregated prefill (GB10 NVFP4 → Mac)

TTFT payoff = local-prefill TTFT ÷ remote-prefill TTFT, five counted reps
per depth after one discarded warmup handoff, deterministic output in every
rep (ledger: "Phase 4 disaggregated GX10→Mac context matrix", 2026-08-20;
"EEE A/B at 130815", 2026-08-21).

| Depth | Local TTFT | Remote TTFT | Payoff |
|---:|---:|---:|---:|
| 2,048 | 6.48 s | 1.520 s | 4.26× |
| 8,192 | 26.77 s | 6.564 s | 4.08× |
| 16,384 | 54.79 s | 14.140 s | 3.87× |
| 32,768 | 114.31 s | 30.489 s | 3.75× |
| 65,536 | 247.88 s | 64.239 s | 3.86× |
| 130,815 | 570.12 s | 137.405 s | **4.149×** |

The 130,815 row is the EEE-off arm (CV 0.576%, ≥6.995 Gbps per-rep payload
floor). EEE on the point-to-point link caused discrete retransmission
blackouts that violated the link floor — disabling it is enrolled production
guidance, not a benchmark trick. A post-network-rebuild requalification of
the 2,048 cell reproduced 1.536 s median TTFT (CV 0.322%, ≥6.459 Gbps
payload floor).

**Sustained load:** eight consecutive 130,815-token handoffs, back to back,
all deterministic, zero producer restarts or deaths (ledger: "eight-handoff
deep soak", 2026-08-23).

**Quality at depth:** native-vs-kquant relative perplexity and calibrated
top-token gates pass on code and prose corpora at both depths. One content
class — high-entropy documentation/digest text — exceeds its calibrated
top-token band at 65,536 on the NVFP4 route; it is published as a
content-local sensitivity (not replicated across corpora, not persistent at
depth), and the kquant lane remains available as the reference route.
Users needing reference-quant behavior on that class can select kquant
explicitly.

## 4. kvpack reuse effects

Ledger: "Kvpack ladder stage-5 isolated-depth verdict" and "stage-6
delta-witness verdict", 2026-08-22.

**Warm prefix reuse** (producer already holds the exact prefix; no producer
compute, receiver answers from cache):

| Depth | Cold TTFT | Warm TTFT | Output |
|---:|---:|---:|---|
| 65,536 | 68.62 s | **0.613 s** | bit-identical |
| 130,815 | 147.83 s | **1.057 s** | bit-identical |

Miss controls (8,192-token unrelated prompts through the same path) stayed
valid at ~12.9 s, proving the warm path is reuse, not cache-forever.

**Delta handoff** (32,768-token prefix held, 65,536-token request):

| Arm | Payload bytes | Output SHA-256 |
|---|---:|---|
| Full handoff | 954,190,848 | `2526a55d…19778` |
| Delta handoff | 517,983,232 (**54.2851%**) | identical |

## 5. What we measured and rejected

- **Distributed speculative decoding across the wire**: 110.59 tok/s only on
  an all-accept control; real proposal acceptance collapsed to 9–38%, with
  measured verifier ceilings below the local spec bar. Rejected for serving;
  receipts in
  [`nvfp4-distributed-speculative-frontier-20260818.md`](nvfp4-distributed-speculative-frontier-20260818.md).
- **Weight-only NVFP4 verification**: no-go (fails exactness gates).
- **ANE/CoreML drafting**: experimental, post-release, never auto-selected.

## Reproducing

Lab comparison tooling lives in `scripts/` (`representative_target_smoke.py`,
`representative_dflash_smoke.py`, `qualify_nvfp4_fast.py`), all designed to
run both engines under a shared accelerator lease with exactness gates — see
[`quickstart.md §6`](quickstart.md). Receipt paths for every table above are
recorded in the ledger entries cited with each table.
