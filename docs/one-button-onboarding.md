# One-button node onboarding — operator reference

This is the honest reference for what `muser node add <user@host>` (CLI) and
`POST /v1/nodes` (dashboard's **Add node** button) actually do. In the signed
macOS image, double-clicking **Muser.app** starts `muser up`, opens that
dashboard, and leaves Terminal visible for progress and Ctrl-C. It replaces
the ~20-field manual ritual previously required to bring up a GX10 prefill
node by hand (see `scripts/gx10/llamacpp/muser_prefilld.py`'s config surface
for what that ritual used to require).

Nothing here is aspirational: every step below is a real action against a
real remote host, over real SSH, with real files on disk on both ends. If a
step is not implemented yet, it is not described as done.

Operational authorization is deliberately separate from capability. On
2026-08-24, fresh CLI enrollments passed all seven progress labels for both
declared recipes: native/text in attempt 9 and combined target-plus-DFlash in
attempt 31. The native/text recipe is the shipped default; the combined
kquant recipe is explicitly selected research. Both transitioned to `healthy`,
and both restored the canonical resident afterward. Those retained runs do not
authorize a later operator session to run `node add`, `daemon`, or `smoke`;
each remains a mutating operator action.

On the default keyless loopback server, opening the dashboard at a literal
loopback address creates a bounded HttpOnly, SameSite session automatically;
the wizard uses its CSRF token and asks for no API key. A configured or LAN
server still requires `--api-key-file`: HTTPS exchanges it for a Secure,
HttpOnly session, while keyed loopback HTTP retains the bearer only in page
memory. No credential is put in a URL or onboarding log. The CLI path does
not use the HTTP management API.

## What "Add node" requires up front

- A **96 GiB Apple Silicon Mac** for the released four-slot, 131K decoder.
  Live preflight checks the platform and physical unified-memory floor before
  downloading artifacts or changing the GX10.
- SSH reachability to the node with **key-based auth already working**
  (agent or keychain identity, or an explicit `--key`). BatchMode is always
  on: if the node would prompt for a password, onboarding fails closed
  rather than hang or fall back to one.
- A **direct data path in both directions**: Mac→GX10 TCP 29591 and
  GX10→Mac TCP 29590. Handoff traffic is not tunneled through SSH;
  `ProxyJump`/`ProxyCommand` targets are rejected because `$SSH_CLIENT`
  would identify the intermediary instead of the Mac.
- The node itself: **aarch64 GB10/SM121**, at least **96 GiB memory** and
  **64 GiB free disk**, a 580-series or newer **NVIDIA driver**, **docker**,
  `curl`, `sha256sum`, `zstd`, and **systemd**, all reachable over that same
  SSH session. A non-root user service also needs systemd lingering so it
  survives SSH logout and starts after reboot. The daemon stage enables
  lingering with passwordless `sudo`, or fails with the exact one-time
  `sudo loginctl enable-linger <user>` command. Preflight checks the complete
  set before anything is copied or run. It inspects Docker's real storage
  root as well as `$HOME`; when they are different filesystems the floors are
  48 GiB for checkpoint/archive staging and 32 GiB for Docker image storage.
- At least **48 GiB free on the Mac during first install**. The final native GGUF
  is 19.6 GB; independently verified release chunks are retained across a
  retry and removed after the atomic assembly succeeds.

Nothing else is asked for. No manual key exchange, no hand-typed lane paths,
no separately-run llama.cpp prefill daemon config.

## Six onboarding stages, plus in-process activation

The pipeline has six executable stages. `smoke` emits both `netqual` and
`smoke` progress rows, so the CLI path displays seven labels. When Add Node is
running inside a fresh `muser up` setup server, the dashboard displays an
eighth `activate` row while that same process loads the Mac decoder. There is
no separate `muser node netqual` subcommand. Progress uses
(`{"schema":"muser.node-progress.v2","step":"...","status":"start|ok|fail|info|planned","detail":"...","data":{...}}`)
emitted one JSON object per line on stdout by the CLI, and relayed verbatim
as SSE `data:` events by `GET /v1/nodes/<name>/progress`. The six stages are
independently re-runnable via `muser node <stage> <name>`.
The separate `muser node qualify <name>` command runs the publication-grade
three-repetition evidence cell and is not part of blocking setup.

### 1. `preflight`

Confirms the node is reachable over `ssh -o BatchMode=yes` (optionally with
`-i <key_path>`), honors the effective `HostName` from `ssh -G`, then confirms
`uname -m` is `aarch64`, an NVIDIA driver is loadable, and `docker` answers.
Before any download or daemon initialization it opens a one-shot listener on
the real Mac receiver port and requires the GX10 to call the `$SSH_CLIENT`
address back. Bad routing, macOS firewall refusal, and an already-running
receiver therefore fail in preflight rather than after a cold startup in smoke.
It also refuses more than five seconds of wall-clock skew because mTLS
validity windows and the producer's absolute request deadlines span both
machines.
Writes nothing remotely; locally, `state` moves to `preflight-ok`, or to
`error` with `last_error` set to the first failing check.

### 2. `deploy`

Pulls the pinned runtime container and requires its exact `sha256:...` image
ID, never merely a mutable tag. If the registry is unavailable or private, it
resume-downloads the public split image archive, verifies every chunk and the
complete compressed stream, loads it, and requires the same image ID. It then
records the local `container_receipt` proving what was verified. Creates the
node's `lane_dir`, a remote absolute path that everything else in this pipeline
(model weights, sockets, the daemon's working state) lives under.
Registry: `container_image`, `container_receipt`, `runtime_sha256`, and
`lane_dir` are written. `runtime_sha256` covers every control script staged
outside the image, so an upgraded client cannot mistake an old control plane
for the current one. `state` becomes `deployed`.

The native image is the immutable vLLM/CUDA dependency root, not a baked copy
of whichever Muser source happened to exist when it was built. The daemon
bind-mounts the enrolled `resident_producer.py`, `request_producer.py`, sender
helpers, and complete `muser_vllm` package read-only over that image. Before
touching CUDA it recomputes the same ordered runtime digest as the Mac deployer
and compares it with the enrollment. A partial upload, symlink substitution,
old overlay, or image/runtime mismatch therefore fails before engine startup.
This is why users need Docker on the GX10 but do not need a Dockerfile,
Compose recipe, Python environment, or manual `docker` command.

### 3. `model`

For the shipped native lane, resume-downloads the pinned native NVFP4 decode
GGUF to the Mac when absent, verifies every release chunk and the assembled
artifact, and acquires every file of the immutable RedHatAI NVFP4 checkpoint
on the node. It checks byte sizes, per-file SHA-256 values, and both aggregate
identities. For the explicitly selected llama.cpp research lane, it installs
the pinned target and DFlash GGUFs. No mismatched artifact reaches the daemon.
The canonical local decoder path is recorded as `consumer_model_path`,
including when onboarding used a custom `--model-dir`.

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
For non-root accounts it verifies systemd lingering before installing the
user unit; the pipeline does not silently substitute an unsupervised tmux
session.

The native runtime keeps the 131,072-token request contract while initializing
vLLM with `max_num_batched_tokens=8192`. Longer prompts use the qualified
chunked-prefill connector; it retains each completed scheduler chunk and
exports only the final complete KV state. The connector refuses preemption,
resume, missing chunks, or a non-final export rather than installing partial
state.

Muser supplies an explicit KV-cache budget, disables optional JIT/CuteDSL
warmups, loads the immutable checkpoint, initializes the 8K scheduler shape,
allocates the 128K KV budget, and performs a 2,048-token first-request warmup.
In the qualified cold-start receipt, the producer reached ready in 187
seconds: weights finished at 108 seconds, KV/kernel warmup began at 115, and
first-request warmup began at 153. Final clean-bundle and canonical receipts
put the observed range at 187–206 seconds. Those are observations, not a
universal ETA. The progress stream reports five sanitized milestones and an elapsed
heartbeat every 15 seconds. Weight deserialization, CUDA engine creation, KV
allocation, and a real warmup cannot be disabled while leaving the same ready
engine contract.

### 6a. `netqual` progress inside `smoke`

The `smoke` stage first measures median TCP-connect RTT. Installed-payload
throughput is derived from the authenticated handoff's committed bytes and
measured payload time and must be at least 3.0 Gbps. Registry fields
`netqual_gbps` and `netqual_rtt_ms` are written as measured floats, never
modeled values. The research kquant lane retains its three-sample median.

### 6b. `smoke`

Runs the operational recipe declared by the enrolled producer identity:

- `native/text` (the shipped default): one 2,048-position Handoff V2 exchange,
  an authenticated and atomically installed target-KV receipt, and eight
  decoded tokens on Metal. It deliberately reports no local-reference
  comparison;
- `kquant/target-plus-dflash` (research lane only): exact target tokens,
  exact required full logits, and exact DFlash tokens and trace across the
  historical three 2,048/256 exchanges.

An identity with no known recipe is refused at enrollment. Every retained
sample must bind the run identity and report positive installed
segments/bytes/timing. The operational recipe and the 3.0 Gbps gate are
mandatory. Only then is `state=healthy` durably written and the terminal
progress event emitted. `muser node add` exits `0` only if this stage passed;
every other outcome is a non-zero exit with `last_error` retained.

### Full qualification (explicit)

`muser node qualify <name>` runs the identity-bound three ordered 2,048/256
local-reference exchanges. For native/text it requires exact target tokens,
deterministic remote output, and the declared bounded full-logit drift. For
the kquant research lane it also requires exact DFlash tokens and trace. This
is the evidence command for reproducing a qualification claim; it is not a
prerequisite for using a node whose operational smoke passed.

## Where things live

| What | Where | Written by |
|---|---|---|
| Node registry | `~/.muser/nodes.toml` (atomic temp+rename, `[[node]]` array-of-tables) | every step, on state transitions |
| Per-node PKI | `~/.muser/nodes/<name>/pki` (`0700` dir, `0600` key files) | `enroll` |
| Remote lane | `<lane_dir>` on the node (deploy-chosen absolute path) | `deploy`, then written into by `model`, `daemon` |
| Replay/generation ledger | local path from `muser-cluster`'s `replay_ledger` config, one entry per `hmac_key_id:hmac_epoch` pair, highest admitted generation only | the receiver (Mac) side of every `smoke` and later production handoff |

## Starting and activating the production consumer

The normal interactive entry point is one long-lived command:

```sh
muser up
```

With no compatible enrollment, `up` binds the final dashboard/API listener in
setup mode. Add Node runs behind that listener; after smoke succeeds, an
in-process activator resolves the just-enrolled decoder and receiver config,
loads the Metal slots, and atomically publishes the inference runtime. The
listener and dashboard session do not change. Health is live throughout:
`/healthz` reports setup/loading/ready, while inference endpoints return 503
with the current phase until the runtime is ready. A failed activation leaves
the setup dashboard alive with the exact error so it can be retried.

With an existing compatible enrollment, bare `up` automatically selects the
newest healthy native node. `up --node <name>` is the explicit multi-node
selector. Both require remote prefill and the strict cross-vendor Metal route;
there is no silent local fallback. `Ctrl-C` owns only the Mac process lifecycle:
the remote user-systemd producer remains supervised and warm. The lower-level
`serve --model ... --prefill auto|remote --cluster-config ...` surface remains
available to service managers.

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

Every step is convergent: re-running `muser node <step> <name>` (or the full
`muser node add` pipeline) after a partial failure re-does only the
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
- A failed `daemon` can be rerun independently. Link or handoff failure
  is retried by rerunning `muser node smoke <name>` against the already-
  deployed container and enrolled PKI.
- A failed `smoke` durably records `state=error`; an accelerator-busy failure
  records `state=blocked`. It preserves `last_error` and never leaves or
  restores `healthy`. `healthy` is set only after the authenticated
  operational handoff, bounded decode, link gate, and registry save pass.
- Re-adding a healthy native node whose container image, staged-runtime
  digest, decoder path, and enrollment all match this build does not rerun
  deploy, enrollment, or daemon startup. It proves one authenticated
  operational handoff and keeps the warm producer. If the registered port is
  offline, Add Node enters the repair pipeline. If a process is listening but
  the authenticated handoff fails, it is left untouched; `--repair` is
  required to authorize redeployment, key rotation, and restart.
- Node mutation/qualification and remote serving hold one cross-process
  topology lease under `$MUSER_HOME`. A second CLI or dashboard process fails
  immediately instead of racing the registry, rotating live enrollment, or
  restarting the single-flight producer during a request.
- `muser node status [--json]` reads the registry and performs a bounded
  one-second live daemon TCP probe. Remembered state and current reachability
  are reported separately, so a previously healthy but stopped daemon is not
  shown as live.

## v1 limits

These are explicit, not hidden gaps:

- **One topology operation at a time.** `Add node` onboards a single prefill
  node per invocation; there is no batch/fleet onboarding in v1. Separate
  processes are serialized by the same topology lease.
- **Single-producer serving topology today.** Regardless of how many
  nodes are `healthy` in the registry, the current serving path is 1x Mac
  decode + 1x GX10 prefill (see `docs/launch-claims.md` #8); onboarding a
  second node registers it but does not make it a second concurrent
  producer. One physical GX10 cannot be registered twice under different SSH
  aliases because both rows would fight over the same listener and enrollment.
- **Manual host entry — no discovery.** `Add node` requires the operator
  to type or paste `user@host`; there is no LAN scan, mDNS, or inventory
  import.
- **No revocation flow yet.** There is no `muser node remove` /
  de-enroll path that walks back a node's certificates, HMAC key, or
  registry row. Removing a node today means manually editing
  `~/.muser/nodes.toml` and the node's `pki_dir`, and is not a supported,
  tested operation.
