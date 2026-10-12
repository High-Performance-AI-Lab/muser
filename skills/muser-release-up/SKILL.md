---
name: muser-release-up
description: Install, start, and verify the released Muser NVFP4 Mac + GX10 topology through the one-button `muser up` workflow. Use for first-run setup, the dashboard Add node flow, source-clone startup, warm relaunch, the first prompt, or diagnosing why the shipped engine is not ready. Use `muser-local-up` for the explicit Mac-only research lane and `muser-disagg-onboard` for operator qualification or node recovery.
---

# Stand up the released Muser engine

Work from the repository root. Read `AGENTS.md`, `docs/install.md`,
`docs/quickstart.md`, and `docs/one-button-onboarding.md` before changing the
flow. The released path is a Mac decoder plus a remote GX10 NVFP4 prefill
producer. Do not substitute the kquant research lane, a manual vLLM command,
or a hand-written container recipe.

## Establish the topology

- Run the Muser CLI on Apple Silicon. Keep its default listener on
  `127.0.0.1:4949` unless the task explicitly requires a secured network bind.
- Use an operator-authorized `user@host` SSH target for the GX10. Confirm the
  target before any remote mutation. Do not infer a host from old receipts.
- Require key-only SSH, the documented remote prerequisites, and enough local
  model capacity. Downloads are resumable and digest checked; never bypass an
  artifact, image, model, metallib, TLS, HMAC, or identity refusal.
- Do not read or print files under a Muser `secrets` or `pki` directory. Pass
  existing paths only to the program that owns them.

## Start the release path

For an installed release bundle, run:

```sh
./bin/muser up
```

For a source clone, build the two binaries used by onboarding, then run the
same workflow:

```sh
cargo build --release --locked -p muser-server --bin muser
cargo build --release --locked -p muser-bench \
  --bin muser-remote-qualify --features metal
./target/release/muser up
```

On a fresh install, open the dashboard, choose **Add node**, and enter
`user@host`. Keep the process running: after onboarding passes, that same
listener becomes the inference server. There is no setup-server stop and
restart.

For a headless first run:

```sh
./target/release/muser node add user@host
./target/release/muser up
```

Use bare `up` to select the newest compatible healthy native NVFP4 node. Use
`up --node <name>` only to choose among enrolled nodes. Never add `--local`
unless the user explicitly requests the Mac-only research lane.

## Interpret cold startup honestly

Separate download time from engine startup. Once artifacts are present, a
cold producer still loads weights, initializes CUDA/vLLM, allocates KV, and
warms a real request. Follow the dashboard milestones and their elapsed time;
do not report a hang while a named stage is advancing. A matching supervised
producer remains warm across normal Mac relaunches.

Do not disable the qualified warmup, reduce the 131K serving contract, or
change vLLM flags merely to make the progress display finish sooner. If a
stage stops advancing, retain the named stage and error before diagnosing it.

## Prove the first prompt

An open port or completed progress bar is not sufficient. Require all four
HTTP checks, and inspect `/snapshot` immediately before and after the
completion:

```sh
curl -fsS http://127.0.0.1:4949/healthz
curl -fsS http://127.0.0.1:4949/health
curl -fsS http://127.0.0.1:4949/v1/models
curl -fsS http://127.0.0.1:4949/snapshot
curl -fsS http://127.0.0.1:4949/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Reply with: muser ready"}],"max_tokens":32}'
curl -fsS http://127.0.0.1:4949/snapshot
```

Also inspect the enrolled node:

```sh
./target/release/muser node status --json
```

Require healthy engine responses, a model entry, generated text, a healthy
live node, and remote-transfer/install counters advancing for the prompt.
Fail if the request silently fell back to local compute or if a receive,
identity, replay, or producer error appeared.

## Relaunch and diagnose

For a normal warm relaunch, use the same `muser up` command. Do not re-enroll
or redeploy a compatible healthy producer. Diagnose a failure at the stage
that reported it:

- use `muser-local-up` only for explicit Mac-only startup or Metal diagnosis;
- use `muser-disagg-onboard` for maintainer qualification, enrollment repair,
  producer recovery, or deep GX10 diagnostics;
- use `muser-bench-ladder` only for controlled performance evidence.

On a shared development machine, place the complete accelerator attempt
behind `scripts/accelerator_safe.py` and review its dry-run first. This lab
serialization rule does not require an end user on their own idle Mac and
GX10 to wrap the released command.

Report the exact command, selected node, milestone that failed or completed,
health results, and first-prompt evidence. Do not turn an ad hoc run into a
new performance claim.
