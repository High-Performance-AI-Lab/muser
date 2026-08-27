---
name: muser-disagg-onboard
description: Enroll, qualify, serve through, diagnose, and safely recover a Muser disaggregated prefill node. Use for `muser node add`, the dashboard Add node wizard, native or combined handoff qualification, remote prefill, GX10 diagnostics, producer restoration, and node health evidence.
---

# Onboard a disaggregated prefill node

Work from the repository root. Read `AGENTS.md`,
`docs/one-button-onboarding.md`, and `scripts/gx10/README.md` before touching a
node. Use the topology and resident identity supplied by the operator; do not
infer them from historical receipts.

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

Native/text qualification requires three ordered exact text handoffs.
Combined qualification requires three ordered exact target-plus-DFlash
handoffs. The progress text must name the chosen recipe. For every sample,
retain the identity, installed bytes/segments, timing, target-token and
required-logit exactness, plus DFlash-token exactness for the combined lane.
The node may transition to `healthy` only after all three samples and the
throughput gate pass.

Verify remembered and live state independently:

```sh
./target/release/muser node status --json
```

Require the enrolled node to report remembered `state: "healthy"`, a live
daemon, a measured qualification rate at or above the configured gate, and no
`last_error`. A zero exit or listening socket alone is insufficient.

## Serve through the enrolled node

Use the enrolled cluster config without reading the sibling PKI directory:

```sh
MUSER_CROSS_VENDOR_QK=1 \
MUSER_GGML_METALLIB=/absolute/path/to/llama.metallib \
  ./target/release/muser serve \
  --model "${MUSER_HOME:-$HOME/.muser}/models/muse-glimmer-30B-kquant-17gb.gguf" \
  --prefill remote \
  --cluster-config "${MUSER_HOME:-$HOME/.muser}/nodes/<name>/cluster.json"
```

`MUSER_CROSS_VENDOR_QK` must be exactly `1`; `remote` and `auto` fail closed
otherwise. Do not add local `--dflash` for a native enrollment. Prove remote
use with a new `/snapshot` transfer, increased disaggregated-prefill and
installed-byte counters, and no new receive failure, fallback, or last error.

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
