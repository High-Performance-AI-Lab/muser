---
name: muser-local-up
description: Build, start, and verify Muser locally on Apple Silicon or the CPU correctness path. Use when asked to install Muser, acquire the pinned model, run `muser up` or `muser serve`, enable local DFlash, or diagnose local startup and HTTP health without weakening artifact, metallib, bind, or API gates.
---

# Run Muser locally

Work from the repository root. Read `AGENTS.md`, then
`docs/quickstart.md`. Treat `docs/release-artifacts.json` as the model trust
root and `release/feature-contract-v1.json` as the implemented feature
boundary.

## Choose the execution path

- Use the default `muser-server` build for Apple Silicon Metal serving.
- Use `--no-default-features` only for CPU correctness and tests. It is not
  the production serving-performance path.
- Add `--dflash` only when both the pinned target and DFlash artifacts are
  available.
- Never start Metal, llama.cpp, Core ML, or another accelerator process
  directly on a shared host. Put the complete command behind
  `scripts/accelerator_safe.py`; review its dry-run before adding
  `--execute`.

## Build

Build the local serving CLI:

```sh
cargo build --locked --release -p muser-server
```

Build the CPU-safe path when the task is correctness-only:

```sh
cargo build --locked --release -p muser-server --no-default-features
```

After source changes, run the repository battery documented in `AGENTS.md`.
Compilation alone is not permission to start an engine.

## Supply the pinned metallib

Metal startup is fail-closed. Set `MUSER_GGML_METALLIB` to the qualified
llama.cpp metallib and keep its source receipt beside it as
`source-receipt.json`, or set `MUSER_GGML_METALLIB_RECEIPT` explicitly.
Compare the receipt's revision, byte count, and SHA-256 with the artifact and
the frozen feature contract before use.

If a development metallib must be built, use only:

```sh
scripts/compile_llama_metallib.sh \
  --llama-dir /path/to/llama.cpp \
  --revision 89e0aa6fd362617d9073e0dafc18e41241521572 \
  --output /fresh/output/llama.metallib \
  --receipt /fresh/output/source-receipt.json
```

Do not hand-author a missing receipt or substitute a metallib from another
llama.cpp revision.

## Acquire and run

`muser up` is the smallest local path. It resolves the manifest-pinned Hugging
Face repository and filename, streams into `$MUSER_HOME/models` (default
`~/.muser/models`), verifies byte size and SHA-256, then serves:

```sh
MUSER_GGML_METALLIB=/absolute/path/to/llama.metallib \
  ./target/release/muser up --no-open
```

Hugging Face is transport, never identity. Do not bypass a repository,
revision, size, or digest refusal.

Use `serve` for explicit model, backend, or DFlash selection:

```sh
MUSER_GGML_METALLIB=/absolute/path/to/llama.metallib \
  ./target/release/muser serve \
  --model /absolute/path/to/muse-glimmer-30B-kquant-17gb.gguf \
  --backend metal
```

```sh
MUSER_GGML_METALLIB=/absolute/path/to/llama.metallib \
  ./target/release/muser serve \
  --model /absolute/path/to/muse-glimmer-30B-kquant-17gb.gguf \
  --dflash /absolute/path/to/dflash-kquant.gguf \
  --dflash-backend metal
```

When accelerator serialization applies, these `up` or `serve` commands must
be the child command after `scripts/accelerator_safe.py --`, not a separate
shell process.

## Preserve the HTTP boundary

The default listener is `127.0.0.1:4949`. Loopback inference is keyless;
management routes need `--api-key-file`. A non-loopback bind requires a TLS
certificate, a mode-0600 TLS key, and a mode-0600 API-key file. Do not relax
that refusal. Request DTOs are strict, so unknown fields must continue to
fail.

## Prove success

Require all four checks, not just an open port:

```sh
curl -fsS http://127.0.0.1:4949/healthz
curl -fsS http://127.0.0.1:4949/health
curl -fsS http://127.0.0.1:4949/v1/models
curl -fsS http://127.0.0.1:4949/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"health check"}],"max_tokens":8}'
```

Require `/healthz` to report `ok: true`, `/health` to report status `ok`, a
model entry, and an HTTP 200 completion with generated content. For a
model-backed Metal process, also require `accelerator_in_use: true`.

For DFlash, compare `/snapshot` before and after a long-enough completion and
require the speculative round/draft counters to advance. That proves route
use, not a performance claim.

## Finish with evidence

Retain the accelerator wrapper's command log and result receipt in a fresh
evidence directory. Report the model identity, metallib source receipt,
selected lane, health responses, and any refusal. Use `docs/benchmarks.md`
and `docs/launch-claims.md` for public numbers; do not create a new claim from
an ad hoc run.

Blind end-user testing and the final release judgment remain operator-side.
