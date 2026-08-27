# Live telemetry and honesty accounting

Muser serves one metrics contract from real process state. Every section has
an `_honesty` tag: `measured` means a live counter or verified build fact,
`target` means a documented release threshold, and `mock` means the process
has no backing measurement. A zero from a live counter remains `measured`.

## Endpoints

`muser serve --model MODEL` binds the dashboard and inference API on one port:

| Route | Method | Result |
|---|---|---|
| `/snapshot` | GET | Current `MetricsSnapshot` JSON |
| `/metrics` | GET | Prometheus text exposition |
| `/stream` | GET (WebSocket) | Authenticated schema-v2 hello, full snapshots, changed-field deltas, resync snapshots, and WebSocket ping frames |
| `/telemetry` | GET | One full SSE snapshot each second |
| `/health`, `/healthz` | GET | Runtime/model health and liveness |
| `/v1/models` | GET | OpenAI-compatible model list |
| `/v1/chat/completions` | POST | Bounded JSON or SSE text/vision inference |

Inference uses up to four isolated resident slots behind one decode-first
accelerator scheduler and a bounded 64-request admission queue. Client
disconnects stop that request at the next emission boundary and reset only
its slot before the lease is released.

## Measurement provenance

| Section or field | Tag | Backing source |
|---|---|---|
| `cluster` architecture | `measured` | Loaded Muse geometry: 52 layers, 13 NoPE, 39 SWA, window 2048 |
| `cluster.weights_bytes` | `measured` when a model is statted, otherwise `mock` | `fs::metadata` on the selected GGUF |
| `nodes[]` | `mock` | Empty until public M3/GX10 utilization, power, and temperature collection is wired |
| `kv.*`, `sessions[]`, `wire.connected_clients` | `measured` | Active OpenAI request registry and exact Muse KV byte formulas |
| cache counters and tiers | `measured` | `muser-kvpack` resident/durable/remote restore and prefill call sites |
| `economics.restore_speedup` | `measured` after a timed restore, otherwise `mock` | Measured restore versus local-prefill duration |
| `economics.derived.seconds_saved` | `measured` only after a paired timing, otherwise `mock` | Accumulated positive local-prefill time minus restore time |
| `economics.derived.gflops_avoided` | `mock` | Model/topology-derived linear floor, not a measured hardware counter |
| `economics.derived.joules_saved` | `mock` | Requires a hardware power calibration; no modeled power is presented as measured |
| `transfers[]` | `measured` | Bounded receipts retained only after authenticated Handoff V2 engine commit |
| `transfers[]._control_ns`, `transfers[]._accept_ns` | `measured` | Receipt's own control-channel round trip and producer-wait split |
| `wire.ingress_gbps` | `measured` | Committed bytes over a 10s rolling window (telemetry routes excluded) |
| `wire.ttft_ms` | `measured` | Request receipt to first non-empty generated content |
| `wire.requests_per_s` | `measured` | HTTP request count over a 10s rolling window (telemetry routes excluded) |
| `wire.itl_ms` | `measured` | Inter-token latency percentiles from per-token emission gaps recorded around the decode loop |
| `wire.egress_gbps` | `mock` | No complete live measurement is retained yet |
| `economics.session_continuation_hits` | `measured` | Same-live-session continuations (no restore ran) — kept out of `cache_hits` so it never inflates the headline hit rate |
| `economics.disagg_prefills`, `economics.disagg_bytes_installed` | `measured` | Disaggregated remote prefills and bytes installed — kept out of `cache_hits`/`tiers`; not a kvpack cache hit |
| `economics.tiers.remote` | `measured` | Renamed from `rdma_pool`: the shipping transport is mTLS-TCP, not RDMA/RoCE |
| `specdec.*` | `measured` | Target-verified DFlash drafted/accepted/fallback counters and route failures |
| `_telemetry_viewers` | `measured` | RAII count of open telemetry SSE connections |
| `_telemetry_requests`, `_active_connections` | `measured` | Telemetry poll volume and open-socket count, kept out of `wire.requests_per_s`/`wire.connected_clients` |
| `_queue_depth`, `_overload_rejections`, `_lock_recoveries` | `measured` | Load-shedding gauges and poisoned-lease recovery counter |
| `_decode.completion_traffic_tok_s_10s`, `_decode.completion_tokens` | `measured` | Ten-second completion traffic rate and process-lifetime completion-token total |
| `_remote.receive_failures`, `_remote.fallbacks`, `_remote.last_error` | `measured` | Remote-prefill health — a degraded disaggregated route is otherwise indistinguishable from one nobody exercised, since both leave `transfers[]` empty |

See `docs/metrics-schema.md` §2 (WIRE / IO, DISAGGREGATION) and `docs/metrics-schema.json` for
the exact wire shape of the `_`-prefixed extensions above.

Completed transfer entries contain the authenticated transfer ID, committed
byte count, receiver-side throughput, and the percentage of transfer time
overlapping producer prefill. They never appear before atomic target (or
target+DFlash) installation succeeds. Failed, cancelled, or corrupt transfers
leave both the engine generation and these completed-transfer metrics
unchanged.

`/stream` currently emits one-second `section_delta` objects containing all
changed top-level snapshot fields, plus a full snapshot every ten ticks. It
does not emit a separate event type for each section. Authentication errors are
HTTP failures before upgrade; application-level WebSocket error frames are not
currently advertised. See `docs/metrics-schema.md` for the exact v1 SSE and v2
WebSocket envelopes.

## Dashboard

`muser serve` embeds `web/muser-dashboard.html` at `GET /`. The page is
live-only and same-origin: it fetches `/snapshot` once, then subscribes to
`/telemetry` SSE keyframes. `file://` and cross-origin endpoint overrides are
disabled during containment.

The renderer never falls back to simulated data after a connection failure.
It reports the endpoint as disconnected and keeps honesty badges from the
last valid snapshot. Mock fields (node GPU telemetry, egress, joules)
render as unavailable rather than as zeros that look live.

## Release claims

Telemetry proves that a route ran; it does not by itself make a performance
claim. Baseline, kvpack, vision, DFlash, and remote speedups may appear in
release material only after their complete retained qualification packets pass
the evaluators in `scripts/`. ANE is experimental/post-release, never selected
by v0.1 `auto`, and has no v0.1 speed claim. Historical Ferrite measurements
are provenance, not Muser product measurements.
