# Quickstart

From zero to talking to Muse Glimmer on your Mac, then to speculative
decoding, then — optionally — to the disaggregated GB10 prefill lane.

## 1. Requirements

- macOS on Apple Silicon (M-series). Metal is the serving path.
- Rust (see `rust-toolchain.toml` for the pinned version).
- The pinned Muse Glimmer GGUF. `muser up` can resolve and verify it for
  you, or point `--model` at a file you already have. Pinned filenames and
  SHA-256 values live in [`release-artifacts.json`](release-artifacts.json);
  validate downloads with `python3 scripts/validate_release_artifacts.py`.
- For Metal builds: the source-pinned llama.cpp metallib. Set
  `MUSER_GGML_METALLIB=/absolute/path/to/llama.metallib` — a Metal-enabled
  checkout **fails closed** without it, because Q6_K tensors route through
  the source-pinned llama kernels. (A `--no-default-features` build needs no
  metallib, but it is the CPU correctness path, not the serving path.)

## 2. Build and serve

```sh
cargo build --release

# One-button path: resolves the pinned Hugging Face repo id, streams only the
# manifest-pinned GGUF into $MUSER_HOME/models (default ~/.muser/models),
# verifies byte size and SHA-256, serves, and opens the dashboard. Ctrl-C stops.
MUSER_GGML_METALLIB=/path/to/llama.metallib ./target/release/muser up

# The repository selector is explicit when desired. A different repo id is
# refused because Hugging Face is transport, not the model trust root.
MUSER_GGML_METALLIB=/path/to/llama.metallib ./target/release/muser up \
  --hf-repo meta-models/Muse-Glimmer-30B-GGUF
```

Or explicitly:

```sh
MUSER_GGML_METALLIB=/path/to/llama.metallib ./target/release/muser serve \
  --model /path/to/muse-glimmer-30B-kquant-17gb.gguf
```

The server listens on `127.0.0.1:4949` by default. Loopback inference is
keyless. Management routes require `--api-key-file`; a **non-loopback bind
is refused** until you supply `--tls-cert`, a mode-0600 `--tls-key`, and a
mode-0600 `--api-key-file`. `muser tls init` / `muser tls issue` provide a
local-CA workflow if you don't bring certificates.

## 3. Talk to it

OpenAI-compatible:

```sh
curl http://127.0.0.1:4949/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":64}'
```

The llama.cpp-compatible surface is also implemented: completion and chat
aliases, `/tokenize`, `/detokenize`, `/apply-template`, embedding aliases,
`/slots`, `/props`, and health aliases — plus the exact `/api/generate` and
`/generate` pair. The route contract, including intentional rejections, is
[`release/llama-server-compat-v1.json`](../release/llama-server-compat-v1.json).
Request DTOs are strict: unknown fields return errors instead of being
silently ignored.

Sessions (`/v1/sessions` CRUD/save/restore with encrypted, tamper-evident
bundles), Prometheus `/metrics`, authenticated `/stream` WebSocket telemetry,
and the live dashboard (every number tagged `measured`/`target`/`mock`) are
available in the same server. See [`telemetry.md`](telemetry.md).

## 4. Turn on DFlash speculative decoding

```sh
MUSER_GGML_METALLIB=/path/to/llama.metallib ./target/release/muser serve \
  --model /path/to/muse-glimmer-30B-kquant-17gb.gguf \
  --dflash /path/to/dflash-kquant.gguf
```

Speculation is exact-token: every committed prefix is verified by the target
model, and muser's spec route is differentially tested against llama.cpp's
own `draft-dflash` route. The serving verify-length is **7**, frozen from
measured natural-text acceptance (long verify lengths stall through
low-acceptance patches; 7 recovers fastest and decodes best). Expect roughly
1.19–1.24× decode versus plain llama.cpp on the reported fixed synthetic
fixtures; see [`benchmarks.md`](benchmarks.md) for the exact depths and the
natural-text cases where that does not hold.

Native NVFP4 speculative decode is deliberately fail-closed — speculation is
a kquant-lane feature. Plain NVFP4 decode serves at parity-within-noise of
kquant.

## 5. Go disaggregated (optional)

**What you need:** a GB10-class (or other NVIDIA aarch64) node reachable over
SSH with key auth, running an NVIDIA driver and Docker; and, for the
measured performance above, a **wired point-to-point 10GbE link** between the
Mac and the node (the lab's numbers were measured with EEE disabled on that
link — see [`disaggregated-prefill.md`](disaggregated-prefill.md) for why
that matters).

**One pipeline from the dashboard:** open the dashboard, click **Add node**,
give it `user@host`. muser runs preflight (SSH, aarch64, driver, docker),
deploys the pinned runtime, acquires and SHA-verifies the lane's pinned model
artifacts, and provisions enrollment-v2 keys (TLS key generated and retained
on the node;
HMAC shared secret over authenticated SSH), starts the producer, and runs the
enrolled lane's qualification recipe: three ordered 2,048/256 native-text or
combined target-plus-DFlash handoffs. The node is marked healthy only after
that recipe passes. The same pipeline is available as `muser node add
<user@host>`. Step-by-step reference:
[`one-button-onboarding.md`](one-button-onboarding.md).

**Serve with remote prefill:**

```sh
MUSER_CROSS_VENDOR_QK=1 MUSER_GGML_METALLIB=/path/to/llama.metallib \
  ./target/release/muser serve \
  --model ~/.muser/models/muse-glimmer-30B-kquant-17gb.gguf \
  --prefill remote \
  --cluster-config ~/.muser/nodes/<name>/cluster.json
```

`MUSER_CROSS_VENDOR_QK=1` selects the cross-vendor math route the CUDA
producer is pinned to; `--prefill auto` and `remote` fail closed without it.
Long prompts then prefill on the NVFP4 producer and hand the KV to your Mac
over an authenticated transport — the reported matrix measured 3.75–4.26×
lower time-to-first-token, deterministic output, and warm prefix reuse at
0.61 s and 1.06 s at the two reported depths. Multimodal requests fall back
to local prefill.

## 6. Verify your own numbers (optional)

The lab's comparisons run both engines under a shared accelerator lease
(`scripts/accelerator_safe.py`, dry-run by default) with exact-token gates.
Representative tooling you can point at your own models and fixtures:

```sh
python3 scripts/representative_target_smoke.py --help   # muser vs llama.cpp, plain
python3 scripts/representative_dflash_smoke.py --help   # spec lane comparison
python3 scripts/qualify_nvfp4_fast.py --help            # disaggregated lane cells
```

The full methodology is documented in [`benchmarks.md`](benchmarks.md).
