# One-button node onboarding — operator reference

This is the honest reference for what `muser node add <user@host>` (CLI) and
`POST /v1/nodes` (dashboard's **Add node** button) actually do. It replaces
the ~20-field manual ritual previously required to bring up a GX10 prefill
node by hand (see `scripts/gx10/llamacpp/muser_prefilld.py`'s config surface
for what that ritual used to require).

Nothing here is aspirational: every step below is a real action against a
real remote host, over real SSH, with real files on disk on both ends. If a
step is not implemented yet, it is not described as done.

Operational authorization is deliberately separate from capability. On
2026-08-24, fresh CLI enrollments passed all seven progress labels for both
declared recipes: native/text in attempt 9 and combined target-plus-DFlash in
attempt 31. Both transitioned to `healthy`, and both restored the canonical
resident afterward. Those retained runs do not authorize a later agent to run
`node add`, `daemon`, or `smoke`; each remains a mutating operator action.

The dashboard wizard requires the API key configured with the server's
`--api-key-file`. On HTTPS it exchanges the key for a Secure, HttpOnly,
SameSite session and uses the returned CSRF token for the mutation. On
loopback HTTP it retains the bearer only in page memory and uses authenticated
fetch for progress, so the key is never put in a URL or onboarding log. The
CLI path does not use the HTTP management API.

## What "Add node" requires up front

- SSH reachability to the node with **key-based auth already working**
  (agent or keychain identity, or an explicit `--key`). BatchMode is always
  on: if the node would prompt for a password, onboarding fails closed
  rather than hang or fall back to one.
- The node itself: **aarch64**, an **NVIDIA driver**, and **docker**
  present and reachable over that same SSH session. Preflight checks for
  all three before anything is copied or run.

Nothing else is asked for. No manual key exchange, no hand-typed lane paths,
no separately-run llama.cpp prefill daemon config.

## Six executable stages, seven progress labels

The pipeline has six executable stages. `smoke` emits both `netqual` and
`smoke` progress rows, so the dashboard displays seven labels. There is no
separate `muser node netqual` subcommand. Progress uses
(`{"schema":"muser.node-progress.v2","step":"...","status":"start|ok|fail|info|planned","detail":"...","data":{...}}`)
emitted one JSON object per line on stdout by the CLI, and relayed verbatim
as SSE `data:` events by `GET /v1/nodes/<name>/progress`. The six stages are
independently re-runnable via `muser node <stage> <name>`.

### 1. `preflight`

Confirms the node is reachable over `ssh -o BatchMode=yes` (optionally with
`-i <key_path>`), then confirms `uname -m` is `aarch64`, an NVIDIA driver is
loadable, and `docker` answers. Writes nothing but a registry row: `state`
moves to `preflight-ok`, or to `error` with `last_error` set to the first
failing check. No files are copied to the node at this step.

### 2. `deploy`

Pushes the pinned runtime container (`container_image`, a `sha256:...` id —
never a mutable tag) to the node's docker and records the local
`container_receipt` proving what was pushed and verified. Creates the
node's `lane_dir`, a remote absolute path that everything else in this
pipeline (model weights, sockets, the daemon's working state) lives under.
Registry: `container_image`, `container_receipt`, `lane_dir` written;
`state` becomes `deployed`.

### 3. `model`

Gets the pinned target and DFlash GGUFs onto the node under `lane_dir`,
verifying their byte sizes and SHA-256 values against the same pinned artifact
manifest used locally (see `docs/release-artifacts.json`). Refuses to hand an
artifact to the daemon on a mismatch.

### 4. `enroll`

Provisions the security material for the disaggregated handoff, in both
directions:

- generates the node TLS private key on the node inside a versioned `0700`
  staging directory, retrieves and verifies only its CSR, and returns the
  signed public certificate; the node TLS private key never leaves GX10;
- exchanges pinned leaf certificates so both sides of the future mTLS
  connection know exactly which peer certificate to accept (see "Security
  model" below);
- mints an HMAC key and records its `hmac_key_id` and incremented
  `hmac_epoch` in the registry. The HMAC is deliberately a shared secret:
  it is transferred to GX10 over the already authenticated, known-host-
  verified SSH channel.

Registry: `pki_dir`, `hmac_key_id`, `hmac_epoch` written; `state` becomes
`enrolled`.

### 5. `daemon`

Starts the remote producer daemon (the GX10-side half of Handoff V2,
`muser-cluster`'s producer path) inside the deployed container, pointed at
the enrolled PKI material and the model placed under `lane_dir`. This is
the step that arms the node to actually serve prefill traffic; it does not
by itself prove the node is reachable end to end — that's `smoke`.

### 6a. `netqual` progress inside `smoke`

The `smoke` stage first measures median TCP-connect RTT. Installed-payload
throughput is derived from each authenticated handoff's committed bytes and
receiver transfer time. The median across three repetitions must be at least
3.0 Gbps. Registry fields `netqual_gbps` and `netqual_rtt_ms` are written as
measured floats, never modeled values.

### 6b. `smoke`

Runs the qualification recipe declared by the enrolled producer identity,
always as exactly three ordered Handoff V2 exchanges for a 2,048-position
prompt and 256 output tokens:

- `native/text`: exact target tokens and the identity's bounded full-logit
  drift policy; no DFlash identity or token trace is admitted;
- `kquant/target-plus-dflash`: exact target tokens, exact required full
  logits, and exact DFlash tokens and trace.

An identity with no known recipe is refused at enrollment. Every retained
sample must bind the run identity and report positive installed
segments/bytes/timing. The three recipe passes and the 3.0 Gbps median are
mandatory. Only then is `state=healthy` durably written and the terminal
progress event emitted. `muser node add` exits `0` only if this stage passed;
every other outcome is a non-zero exit with `last_error` retained.

## Where things live

| What | Where | Written by |
|---|---|---|
| Node registry | `~/.muser/nodes.toml` (atomic temp+rename, `[[node]]` array-of-tables) | every step, on state transitions |
| Per-node PKI | `~/.muser/nodes/<name>/pki` (`0700` dir, `0600` key files) | `enroll` |
| Remote lane | `<lane_dir>` on the node (deploy-chosen absolute path) | `deploy`, then written into by `model`, `daemon` |
| Replay/generation ledger | local path from `muser-cluster`'s `replay_ledger` config, one entry per `hmac_key_id:hmac_epoch` pair, highest admitted generation only | the receiver (Mac) side of every `smoke` and later production handoff |

## Starting the production consumer

Enrollment writes the Mac receiver configuration to
`~/.muser/nodes/<name>/cluster.json` (or the same path below
`$MUSER_HOME`). The GX10 producer uses the pinned CUDA compatibility graph,
so the Mac decoder must use the matching exact Metal graph when it consumes
those KV planes:

```sh
MUSER_CROSS_VENDOR_QK=1 muser serve \
  --model ~/.muser/models/muse-glimmer-30B-kquant-17gb.gguf \
  --prefill remote \
  --cluster-config ~/.muser/nodes/<name>/cluster.json
```

`--prefill auto` has the same requirement. Both modes refuse startup when
`MUSER_CROSS_VENDOR_QK` is absent or not exactly `1`; they never install a
strict CUDA cache into the ordinary Metal math route. The managed GX10
producer container and the onboarding smoke child already set the same flag.
There is no separate Mac serving daemon installed by onboarding in v0.1, so
operators or service managers launching `muser serve` must preserve this
environment setting.

### Why the generation ledger must never reset

Handoff V2's replay admission (`crates/muser-cluster/src/security.rs`,
`ReplayLedger`) tracks, per `hmac_key_id:hmac_epoch`, the highest KV
generation number it has ever admitted, and refuses anything at or below
that number. Generation `0` is refused unconditionally. This is the only
thing standing between the receiver and a replayed handoff: HMAC and mTLS
prove the message came from a key/cert this receiver trusts, but only the
ledger proves the message isn't a valid, correctly-signed *replay* of one
already installed. Deleting or truncating the ledger file — including as a
side effect of re-running `enroll` against an existing node without
minting a new `hmac_epoch` — resets the "highest admitted" counter to
zero and makes every previously captured, validly-signed handoff message
replayable again. If a node's PKI is ever regenerated, its `hmac_epoch`
must be bumped so the ledger starts a fresh, disjoint counter space rather
than silently reusing one whose history was just discarded.

## Security model

- **mTLS both ways.** Both the Mac (consumer) and the node (producer)
  present a client certificate; each side verifies the other's chain and
  additionally pins the peer's leaf certificate by SHA-256
  (`leaf_sha256_pins`, TLS 1.3-only, exact ALPN). A valid-but-unpinned
  certificate is rejected the same as an invalid one.
- **Pinned leaves**, not just CA trust: `enroll` records the pin for the
  specific certificate it just issued, not a CA that could later sign a
  different one.
- **HMAC seals** the KV artifact's canonical manifest on top of the TLS
  channel, keyed by `hmac_key_id`/`hmac_epoch` and checked against the
  generation ledger above before any engine-side install happens.
- **TLS private keys never leave the machine that generated them.** The Mac
  key remains on the Mac and the node key remains on GX10; only a CSR and
  certificates cross during enrollment. The HMAC key is a different thing:
  it is intentionally a shared secret copied over authenticated SSH.
  Nothing under `pki_dir` is written into
  `~/.muser/nodes.toml` — the registry's `pki_dir` field is a path, not a
  copy.
- **Authenticated management API.** `POST /v1/nodes`, `GET /v1/nodes`, and
  `GET /v1/nodes/<name>/progress` require bearer authentication or a
  same-origin dashboard session; mutations also require Origin and CSRF.
  LAN access exists only on a native-TLS listener with the mandatory API-key
  file. The enrolled node does not receive dashboard credentials.

## Failure modes and re-run semantics

Every step is idempotent: re-running `muser node <step> <name>` (or the
full `muser node add` pipeline) after a partial failure re-does only the
work that step is responsible for and does not require unwinding earlier
steps by hand. Concretely:

- A failed `preflight` leaves the registry row (if any) at `state=error`
  with `last_error` set; nothing remote was touched.
- A failed `deploy` can be re-run; it re-pushes the pinned image and is a
  no-op on the node's docker if that exact `sha256:...` id is already
  present.
- A failed `model` re-verifies whatever partial file exists under
  `lane_dir` and resumes or restarts the transfer; it never hands a
  hash-mismatched file to `daemon`.
- Re-running `enroll` against an already-`enrolled` node **mints a new
  `hmac_epoch`** rather than reusing the old one. The registry is durably set
  to `needs-reenrollment` before secrets change, and the remote versioned
  directory becomes active through one symlink rename only after its key,
  certificate, CA, HMAC and config all validate. An interrupted activation
  therefore never restores `healthy`; rerunning resumes the same epoch stage.
- A failed `daemon` can be rerun independently. Link or qualification failure
  is retried by rerunning `muser node smoke <name>` against the already-
  deployed container and enrolled PKI.
- A failed `smoke` durably records `state=error`; an accelerator-busy failure
  records `state=blocked`. It preserves `last_error` and never leaves or
  restores `healthy`. `healthy` is set only after the exact qualification and
  registry save both pass.
- `muser node status [--json]` reads the registry and performs a bounded
  one-second live daemon TCP probe. Remembered state and current reachability
  are reported separately, so a previously healthy but stopped daemon is not
  shown as live.

## v1 limits

These are explicit, not hidden gaps:

- **One node at a time.** `Add node` onboards a single prefill node per
  invocation; there is no batch/fleet onboarding in v1.
- **Single-producer serving topology today.** Regardless of how many
  nodes are `healthy` in the registry, the current serving path is 1x Mac
  decode + 1x GX10 prefill (see `docs/launch-claims.md` #8); onboarding a
  second node registers it but does not make it a second concurrent
  producer.
- **Manual host entry — no discovery.** `Add node` requires the operator
  to type or paste `user@host`; there is no LAN scan, mDNS, or inventory
  import.
- **No revocation flow yet.** There is no `muser node remove` /
  de-enroll path that walks back a node's certificates, HMAC key, or
  registry row. Removing a node today means manually editing
  `~/.muser/nodes.toml` and the node's `pki_dir`, and is not a supported,
  tested operation.
