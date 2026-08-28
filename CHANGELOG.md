# Changelog

All notable changes to Muser are recorded here. Performance figures retain
the scope and methodology of the linked benchmark sections.

## 0.1.0-beta.1 — 2026-08-28

- Fresh remote nodes select the shipped NVFP4 vLLM producer lane. Onboarding
  now resolves the split Mac consumer, source-pinned Metal runtime, immutable
  GX10 image, and checkpoint without operator-supplied build artifacts.
- Receiver loss cancels abandoned producer work, verifies the warm engine is
  idle before reuse, and restarts the exact image when bounded cancellation
  cannot prove recovery.
- Native prompt budgets scale through the receipted 128k path. The slower
  kquant+DFlash lane remains available as explicitly selected research and
  keeps its measured 4,096-token onboarding envelope.

- The [six-depth plain local matrix](docs/benchmarks.md#1-plain-local-decode-and-prefill-no-speculation)
  matched or beat the pinned llama.cpp comparator on mean decode and prefill
  at every tested depth on synthetic exact-token fixtures.
- The [local kquant DFlash matrix](docs/benchmarks.md#2-dflash-speculative-decode-local)
  measured 1.19–1.24× decode throughput at its reported depths, with the
  131,008/48 packet reaching 1.0254× end-to-end wall time.
- [Disaggregated GB10 NVFP4 prefill](docs/benchmarks.md#3-disaggregated-prefill-gb10-nvfp4--mac)
  measured a 3.75–4.26× TTFT payoff across the reported matrix.
- [Resident kvpack reuse](docs/benchmarks.md#4-kvpack-reuse-effects) returned
  first tokens in 0.61 s and 1.06 s at the two reported depths, versus 68.6 s
  and 147.8 s cold; the measured delta handoff moved 54.2851% of full payload
  bytes.
