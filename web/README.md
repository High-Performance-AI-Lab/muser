# Muser status dashboard

`muser-dashboard.html` is a dependency-free glass UI served by `muser` at `/`
and `/dashboard`. It renders the live `MetricsSnapshot` from `GET /snapshot`
and `GET /telemetry` (SSE keyframes). It is same-origin only. Standalone
`file://` endpoint overrides are disabled during security containment.

Numbers come from this process: cluster geometry, KV occupancy, kvpack
economics (including session-continuation and disaggregated-prefill reuse,
shown separately from served cache hits), committed Handoff V2 transfers
(with their control/accept phase timings when present), TTFT, ITL, decode
tokens/s, queue depth, remote-prefill failure/fallback counters, request
rate, sessions, events, and DFlash draft/accept counters. GPU util/power/temp
and wire egress are tagged `mock` and shown as unavailable. Sealed speed
claims stay empty until a retained qualification packet populates `tricks[]`.
Requests/s and the ingress gauge are already 10s rolling windows on the wire
(telemetry polls excluded) — the client renders them as-is, no client-side
re-windowing.

Public-CoreML ANE counters describe only the explicitly selected experimental,
post-release backend. v0.1 `auto` is Metal and the dashboard never presents an
ANE optimization or speed claim as release-qualified.

The page never falls back to simulated data. When `nodes[]` is empty (no
GPU topology wired up yet) the topology panel says so instead of drawing
placeholder node cards, and the disaggregation pipeline only draws the
prefill box once a committed transfer/receipt has been seen. A disconnect
marks the badge as disconnected; a connection that stops advancing frames
without erroring (a stalled SSE stream) is caught separately — the badge
flips to "stale · last update Ns ago" and panels dim after ~4s with no new
frame. Field-by-field provenance is in `docs/telemetry.md`.

The kvpack economics "remote" tier is read from `tiers.remote` (the
server-side rename from `rdma_pool` has fully landed).

## Chat

The "Talk to the model" panel is a live chat client for the engine itself.
It posts `{stream: true}` to the same-origin `POST /v1/chat/completions` and
appends each SSE chunk as it arrives — the engine emits one chunk per sampled
token, so the visible token count is a count, not an estimate, and the
per-stream TTFT / tok/s line is measured from chunk arrival times (with the
server's `usage` figures shown when the stream reports them). Reasoning
deltas (`reasoning_content`) stream into a collapsible section that folds
itself when the answer's first token lands. **Stop** aborts the fetch and
the server cancels the generation on client disconnect; a stopped stream
keeps its partial text as the assistant turn. Loopback serving is keyless;
on a non-loopback serve the panel asks once for the API key and keeps it in
page memory only — never a URL, storage, or log. Model text reaches the DOM
only via `textContent`.

## Add-node wizard

The topology panel's "+ Add node" button opens a modal that posts
`{host,user,name?,key_path?}` to `POST /v1/nodes`. The modal also accepts the
server API key. HTTPS exchanges it for the server's Secure dashboard session
and CSRF token; loopback HTTP retains the bearer only in page memory. Neither
path places the key in a URL or raw log. A 202 switches the modal to
a live checklist of seven `muser.node-progress.v2` labels (preflight, deploy,
model, enroll, daemon, netqual, smoke) driven by six executable pipeline
stages—`netqual` is emitted by `smoke`—and a bearer-authenticated fetch stream
to `GET /v1/nodes/<name>/progress`. Every step/status/detail string is
`esc()`-ed before it touches the DOM — it has crossed ssh and a remote shell
to get here. The raw JSON lines stay available behind a "Raw log" toggle, and
a failed run offers "Copy log". `GET /v1/nodes` polls every 5s, only while
the topology panel is in view, to draw node cards (state, daemon liveness,
link quality, image short-sha) below the topology grid; the pipeline canvas
now also draws the GX10 box, labeled with the node's name, once a registry
node reaches `healthy` — not only on a committed transfer receipt.

The dashboard is fully self-contained: no CDN, external stylesheet, or font.
Its network calls are same-origin only: `/snapshot`, `/telemetry`,
same-origin `POST /v1/chat/completions` (the chat pane), optional HTTPS
`/v1/dashboard/login`, and the `/v1/nodes` onboarding/progress/status
routes.
