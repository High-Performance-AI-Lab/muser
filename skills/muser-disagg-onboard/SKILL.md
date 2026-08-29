---
name: muser-disagg-onboard
description: Qualify, deeply diagnose, and safely recover a Muser disaggregated prefill node after ordinary release startup. Use for explicit `muser node qualify`, enrollment repair, native or combined handoff qualification, GX10 diagnostics, producer restoration, and node health evidence. For the normal dashboard Add node and released NVFP4 first run, use `muser-release-up`.
---

# Onboard a disaggregated prefill node

Work from the repository root. Read `AGENTS.md`,
`docs/one-button-onboarding.md`, and `scripts/gx10/README.md` before touching a
node. Use the topology and resident identity supplied by the operator; do not
infer them from historical receipts.

This is the maintainer qualification and recovery path, not the ordinary
one-button install. The shipped default is the native NVFP4 lane. Do not set
`MUSER_CROSS_VENDOR_QK` or select the kquant/llama.cpp research lane unless
the task explicitly calls for that experiment.

## Enforce the boundaries first

- Obtain explicit authorization for node mutation and for displacing any
  resident producer.
- Never read, print, copy, or archive anything below
  `~/.muser/**/secrets` or a node's `pki` directory. Passing an existing
  config path to its owning program is allowed.
- Serialize the complete attempt with `scripts/accelerator_safe.py`. Check
  `/tmp/ferrite.gpu.lock`; wait or coordinate when another campaign owns it.
- Enforce one GPU producer. Capture node state before mutation and after
  restoration. If two producers can run concurrently, abort and restore.
- Use a fresh node name, enrollment, HMAC epoch, and evidence directory for
  each full attempt. Never reset or truncate a replay ledger.
- Preserve every identity, TLS, HMAC, replay, exactness, and healthy-state
  refusal.

## Prepare the receiver

Build the CLI and the Metal qualifier used by the smoke gate:

```sh
cargo build --locked --release -p muser-server
cargo build --locked --release -p muser-bench --features metal \
  --bin muser-remote-qualify
```

Set `MUSER_GGML_METALLIB` and, when needed,
`MUSER_GGML_METALLIB_RECEIPT` to a receipt-bound qualified metallib. Validate
the pinned model files with `scripts/validate_release_artifacts.py`. Do not
replace a pinned image digest, model digest, or metallib receipt with a
mutable tag or filename assumption.

## Capture the starting state

Before onboarding, retain an operator-readable receipt containing:

- route/interface selection and SSH reachability;
- the resident container image digest, state, start time, and restart count;
- producer socket and readiness receipt presence;
- GPU lease holder and supervisor state;
- any required link property, such as EEE state;
- the exact restoration command and expected final resident identity.

Do not include private-key or shared-secret contents.

## Plan the pipeline

Start with the CLI dry-run; it performs no SSH or remote writes:

```sh
./target/release/muser node add <user@host> \
  --name <fresh-node-name> \
  --producer <native-or-llamacpp> \
  --container-receipt /path/to/pinned-container-receipt.json \
  --model-dir /path/to/pinned-models \
  --ggml-metallib "$MUSER_GGML_METALLIB" \
  --ggml-metallib-receipt "$MUSER_GGML_METALLIB_RECEIPT" \
  --dry-run --json
```

Select `native` for the text-only native NVFP4 recipe. Select `llamacpp` for
the combined target-plus-DFlash recipe. An unknown lane or a lane without a
qualification recipe must fail at enrollment.

Review the plan, coordinate the resident swap, then place the real onboarding
command under one whole-attempt accelerator lease. Retain its JSON-lines
progress stream.

## Require every stage

The six executable stages produce seven progress labels:

1. `preflight`: key-only SSH, `aarch64`, NVIDIA driver, Docker, capacity.
2. `deploy`: exact pinned runtime image and lane directory.
3. `model`: pinned target/DFlash filenames, byte sizes, and SHA-256.
4. `enroll`: node-local TLS key, pinned peer leaves, fresh HMAC key/epoch.
5. `daemon`: the single remote producer starts and listens.
6. `smoke`: emits `netqual`, then runs the lane-derived exactness recipe.

The release enrollment smoke gate performs one production-path operational
handoff before publishing a healthy node. The explicit maintainer command
`muser node qualify <name>` performs the full three-repetition qualification.
For every qualification sample, retain the identity, installed bytes and
segments, timing, target-token and required-logit exactness, plus DFlash-token
exactness for an explicitly requested combined research lane. The node may
transition to `healthy` only after its selected gate passes.

Verify remembered and live state independently:

```sh
./target/release/muser node status --json
```

Require the enrolled node to report remembered `state: "healthy"`, a live
daemon, a measured qualification rate at or above the configured gate, and no
`last_error`. A zero exit or listening socket alone is insufficient.

## Start through the enrolled node

Use the released selector rather than reconstructing a private cluster config
or model command:

```sh
./target/release/muser up --node <name> --no-open
```

Prove remote use with a real completion, a new `/snapshot` transfer, increased
disaggregated-prefill and installed-byte counters, and no new receive failure,
fallback, or last error. Do not add local DFlash to a native enrollment.

## Diagnose before changing code

Use the dependency-free tools documented in `scripts/gx10/README.md`:

- `tcp_probe.py` for the current raw link ceiling in both directions;
- `durable_fsync_probe.py` for replay-ledger durable-write tail latency;
- `handoff_report.py` for per-repetition producer, wire, and receiver phases;
- `restart_resident_producer.py` for the complete fail-closed restart ritual;
- `supervise_resident_producer.py` for unattended recovery with a failure
  latch.

These diagnostics do not authorize node access. Do not copy a whole source
file from repository HEAD into an older container; extract its version and
make only an authorized minimal change.

## Restore and close

Remove or quiesce the transient producer using the documented operator
procedure. If the resident was displaced, run the full restart ritual rather
than a bare container restart. Compare final state with the starting receipt:
same resident image and role, socket, lease, supervisor, route properties,
and readiness. Stop and report if restoration cannot be proved.

Retain the full progress log, wrapper receipt, per-handoff receipts,
before/after node state, and restoration receipt in a fresh append-only
evidence directory. Public statements must stay within
`docs/launch-claims.md`; blind onboarding and final qualification remain
operator-side.
