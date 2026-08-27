# kvpack cache-economics accounting

This document describes the counters emitted by the current Muser process. It
is not a benchmark result. Historical Ferrite ratios are not substituted for
missing Muser measurements.

## What counts as a hit

A cache hit is recorded only after authenticated state has been restored and
committed into an engine slot. A lookup match that fails verification or
installation contributes no saved value.

Two useful events are deliberately not cache hits:

- continuing the same live session, where no restore ran, increments
  `session_continuation_hits`;
- installing a newly computed GX10 prefill increments `disagg_prefills` and
  `disagg_bytes_installed`.

Neither event changes `cache_hits`, `bytes_served_from_cache`, or a storage-tier
hit counter.

## Byte accounting

Muse has 13 growing NoPE layers and 39 sliding layers with a 2,048-token
window. Each K+V row in one layer is 1,024 bytes. For `N` positions:

```text
nope_bytes(N) = 13 * N * 1,024
swa_bytes(N)  = 39 * min(N, 2,048) * 1,024
kv_bytes(N)   = nope_bytes(N) + swa_bytes(N)
```

Durable and remote restores must use the authenticated manifest's installed
byte count. The resident tier has no separate manifest and is explicitly
allowed to use the topology-derived estimate. A partial-prefix request records
the matched restore and recomputed suffix separately; the request does not
inflate both the hit and miss denominator.

## Live fields

`EconomicsCounters` supplies the `economics` object in `/snapshot`:

- `cache_hits`, `cache_misses`, `hit_rate`;
- `bytes_served_from_cache`, `bytes_recomputed`;
- `restore_ops`, `prefill_ops`;
- tier counters for `resident`, `ssd`, and `remote`;
- `session_continuation_hits`, `disagg_prefills`, and
  `disagg_bytes_installed`;
- `restore_speedup` and the derived values below.

The `remote` tier names authenticated remote cache reuse. A fresh GX10 prefill
handoff is reported in the separate disaggregation fields. The v0.1 transport
is mTLS-TCP, not RDMA/RoCE.

## Timing and derived values

`restore_speedup` becomes measured only after a caller records positive,
paired wall-clock durations for a restore and the identical local-prefill cut.
Before that it is zero and `_honesty.economics.restore_speedup` is `mock`.

For a timed restore:

```text
restore_speedup = local_prefill_seconds / restore_seconds
seconds_saved   = max(0, local_prefill_seconds - restore_seconds)
```

The process maintains an exponential moving average for the speedup and a
monotonic sum for seconds saved. `gflops_avoided` is emitted only for timed
restores and uses the conservative linear weight-matmul floor
`2 * 30e9 * restored_tokens`; it omits attention FLOPs. The 30B parameter
value is a model-derived input, so this field is a derived estimate rather
than a directly measured hardware counter.

`joules_saved` has no calibrated power source. It remains zero and is always
tagged `mock`.

## Honesty rules

- Idle zeros from live counters remain measured zeros.
- Manifest bytes are not replaced by `tokens * bytes_per_token` when a
  manifest exists.
- Same-session continuation and fresh disaggregated prefill never inflate the
  cache hit rate.
- Untimed restores may update hit/byte counters, but never synthesize a
  speedup, seconds-saved, or GFLOPs value.
- Telemetry proves that an event occurred; release performance wording still
  requires the complete retained qualification packet in
  `docs/launch-claims.md`.
