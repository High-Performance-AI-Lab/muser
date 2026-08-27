# Muser metrics and telemetry contract

The machine-readable snapshot schema is `docs/metrics-schema.json` (JSON
Schema Draft 2020-12). This document describes the current server behavior;
historical designs and target topology are not part of the live payload.

Honesty tags use three values:

- `measured`: a live counter, duration, or verified loaded-model fact;
- `target`: a threshold or modeled goal, never an observed result;
- `mock`: no backing measurement is available; the dashboard renders the
  value unavailable.

An idle live counter may be a measured zero. Performance claims still require
the retained release qualification described in `docs/launch-claims.md`.

## Transports

| Route | Result |
|---|---|
| `GET /snapshot` | one `MetricsSnapshot` JSON object |
| `GET /metrics` | Prometheus text exposition |
| `GET /telemetry` | compatibility SSE; one schema-v1 snapshot envelope per second |
| `GET /stream` | authenticated WebSocket telemetry schema v2 |

`/stream` accepts a Secure dashboard cookie or a 30-second single-use ticket.
A long-lived bearer key is never placed in the WebSocket URL.

### SSE envelope

Each `/telemetry` event is named `snapshot` and its data is:

```json
{"v":1,"type":"snapshot","seq":0,"t":0.0,"data":{}}
```

`seq` increases per connection and `t` is process uptime in seconds. `data`
is a full `MetricsSnapshot`.

### WebSocket schema v2

After upgrade the server emits:

1. `{"v":2,"type":"hello","schema":"muser.telemetry.v2",
   "snapshot_interval_s":10,"ping_interval_s":5}`;
2. a full `snapshot` frame immediately and every ten sequence ticks;
3. `section_delta` frames at intervening one-second ticks. Their `data` is an
   object containing every top-level snapshot field whose value changed;
4. WebSocket Ping control frames every five sequence ticks.

Snapshot and delta frames have `v`, `type`, `seq`, and `data`. They do not
carry the schema-v1 SSE `t` field. A client treats a full snapshot as the
resynchronization boundary. Authentication failures are HTTP responses before
upgrade; the current server does not advertise application-level WebSocket
error frames.

## `MetricsSnapshot`

The stable fields are:

```text
schema_version, generated_at, engine_clock_s, uptime_s,
cluster, nodes, kv, economics, transfers, wire, sessions, tricks, specdec
```

The snapshot also contains documented `_`-prefixed extensions used by the
dashboard: `_events`, `_honesty`, `_telemetry_viewers`,
`_telemetry_requests`, `_active_connections`, `_queue_depth`,
`_overload_rejections`, `_lock_recoveries`, `_decode`, `_remote`, and
`_phases`. These are additive and clients must ignore extensions they do not
understand.

### Cluster and nodes

`cluster` reports the loaded Muse geometry: 52 layers, 13 full NoPE layers,
39 sliding layers, and a 2,048-token window. `weights_bytes` is measured only
when the selected GGUF was statted; without a model it is a placeholder and is
tagged `mock`.

`nodes[]` is currently empty and tagged `mock`: M3/GX10 utilization, memory,
power, temperature, and token-rate collection are not wired to this payload.
The separate node-management API and registry do not manufacture telemetry
node cards.

### KV and sessions

`kv` is derived from active inference-session rows and the Muse topology. It
contains total bytes/tokens, a configured capacity, the NoPE/SWA structural
split, and per-session entries. `sessions[]` and `_events` come from the live
inference registry.

These dashboard session rows are observability records. The canonical logical
session CRUD/save/restore/migration API has its own revisioned objects under
`/v1/sessions`.

### Cache economics

`economics` contains live cache hit/miss, byte, operation, and storage-tier
counters. Same-live-session continuations and newly computed GX10 prefills are
separate fields and do not count as cache hits. See
`docs/kvpack-economics.md` for the accounting rules.

`restore_speedup` is measured only after a positive paired restore/local-
prefill timing. `seconds_saved` shares that measurement condition.
`gflops_avoided` is a topology/model-derived floor rather than a hardware
measurement, and `joules_saved` is unavailable until power calibration.

### Transfers

`transfers[]` contains completed authenticated Handoff V2 installs only. Each
entry reports the transfer identity/endpoints, committed byte count,
receiver-side throughput, overlap fraction, and the receipt's control and
accept durations. Failed, cancelled, or corrupt transfers do not create an
entry. The live list does not expose in-progress per-layer shipping phases;
completed entries have phase `done`.

The remote transport is mTLS-TCP. The 3.0 Gbps value is a release
qualification minimum, not a live default and not a "10GbE-class" claim.

### Wire, decode, and phases

`wire.requests_per_s`, `wire.ingress_gbps`, and
`_decode.completion_traffic_tok_s_10s` are ten-second rolling rates.
`wire.ttft_ms` and `wire.itl_ms` are process-local percentile samples.
`wire.egress_gbps` has no complete measurement and is tagged `mock`.

`_decode.completion_tokens` is a lifetime token counter;
`_decode.last_generation_at` and `_decode.active` describe recent/in-flight
generation. `_queue_depth`, `_overload_rejections`, and the connection/viewer
counters are live process counters.

`_phases` exposes completed sample counts, total milliseconds, and means for
queue, prefill, sampling, grammar, detokenization, bounded enqueue/write,
DFlash drafting, and DFlash target verification. Its
`last_request_decode_tok_s` is request-local throughput, distinct from the
ten-second traffic rate.

Prometheus exports the same phase totals and sample counts, request-local
decode throughput, and continuous-batching counters. Packed-batch counters
increase only for submissions containing two to four ready sequence rows.

### DFlash and optimization claims

`specdec` contains live DFlash drafted/accepted counts, the last accepted run,
configured draft length, and Metal plus explicitly selected experimental-ANE
route-failure counters. ANE is post-release, never selected by v0.1 `auto`,
and has no v0.1 qualification or speed card. Its
`per_weight_read_speedup` is absent because no live producer supplies it.

`tricks[]` is intentionally empty. No optimization card appears until its
independent correctness and performance qualification passes for the release
identity. Historical Ferrite results are provenance, not live Muser metrics.

## Versioning and validation

`MetricsSnapshot.schema_version` and SSE `v` are 1. WebSocket frames use
protocol `v: 2` and name `muser.telemetry.v2` in the hello. Additive
`_`-prefixed snapshot extensions are ignorable; incompatible stable snapshot
shape changes require a schema-version bump.

Validate the JSON source and server DTO tests with:

```sh
python3 -m json.tool docs/metrics-schema.json >/dev/null
cargo test -p muser-server --no-default-features metrics
```
