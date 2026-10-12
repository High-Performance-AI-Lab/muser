# Quickstart

From zero to the shipped topology: NVFP4 prefill on a GB10/GX10, authenticated
KV handoff, and Metal decode on the Mac. The local kquant/DFlash lane is a
separate research option.

## 1. Requirements

- macOS on Apple Silicon. The released four-slot, 131K configuration is
  qualified on a 96 GB M3 Ultra; smaller-memory Macs are not yet a supported
  claim.
- 48 GiB free for first-time native artifact assembly (or pass `--model-dir`
  on a larger volume when adding the node).
- Rust (see `rust-toolchain.toml` for the pinned version) when building from
  source. The Apple Silicon release bundle includes both required binaries.
- The pinned native Muse Glimmer decoder. Add Node downloads and verifies it
  automatically; no model path is requested. Pinned filenames and SHA-256
  values live in [`release-artifacts.json`](release-artifacts.json).
- The source-pinned llama.cpp metallib used by a few Metal kernels. Muser
  downloads the 7 MB binary and its source receipt on first use, then pins
  both SHA-256 values. `MUSER_GGML_METALLIB=/absolute/path/to/llama.metallib`
  remains an override and is held to the same receipt. A
  `--no-default-features` build is the CPU correctness path.

## 2. Install or build

The native release bundle is the shortest route; see [`install.md`](install.md).
From a source clone:

```sh
cargo build --release --locked -p muser-server --bin muser
cargo build --release --locked -p muser-bench --bin muser-remote-qualify --features metal

# One-button shipped path: opens setup when no NVFP4 enrollment exists, then
# stays alive as the inference server after Add node succeeds.
./target/release/muser up
```

For the Mac-only kquant research lane, opt in explicitly:

```sh
./target/release/muser up --local
# or: ./target/release/muser serve --model /path/to/model.gguf
```

The server listens on `127.0.0.1:4949` by default. Loopback inference is
keyless. When no API key is configured, a dashboard loaded from a literal
loopback address receives a bounded same-origin session automatically, so Add
Node works without a separate login step. A **non-loopback bind is refused**
until you supply `--tls-cert`, a mode-0600 `--tls-key`, and a mode-0600
`--api-key-file`. `muser tls init` / `muser tls issue` provide a local-CA
workflow if you don't bring certificates.

## 3. Talk to it

OpenAI-compatible:

```sh
curl http://127.0.0.1:4949/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Reply with exactly: Remote prefill is working."}],"max_tokens":256}'
```

Muse Glimmer returns reasoning in `reasoning_content` and the answer in
`content`; budget for both. Greedy `temperature: 0` can make this checkpoint
repeat its reasoning, so the first-run example uses the qualified default
sampler.

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
./target/release/muser serve \
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

## 5. Add the GX10 producer

**What you need:** a GB10/SM121 aarch64 node with at least 96 GiB memory and
64 GiB free disk when home and Docker share a filesystem (or 48 GiB under
home plus 32 GiB in Docker storage when split), reachable over SSH with key
auth, running a 580-series or
newer NVIDIA driver, Docker, curl, sha256sum, and zstd; and, for the
measured performance above, a **wired point-to-point 10GbE link** between the
Mac and the node (the lab's numbers were measured with EEE disabled on that
link — see [`disaggregated-prefill.md`](disaggregated-prefill.md) for why
that matters). TCP 29591 must be reachable Mac→GX10 and TCP 29590 GX10→Mac;
SSH jump/proxy hosts are not the data plane. Preflight proves the reverse
callback before downloading or starting the producer. The node also needs
systemd. For a non-root SSH account, user lingering must be enabled so the
producer survives logout and starts after reboot; onboarding enables it with
passwordless `sudo`, or stops with the exact one-time
`sudo loginctl enable-linger <user>` command.

**One pipeline from the dashboard:** open the dashboard, click **Add node**,
give it `user@host`. muser runs preflight (SSH, aarch64, driver, docker),
deploys the pinned runtime, acquires and SHA-verifies the lane's pinned model
artifacts (including a resumable 11-part download of the native Mac GGUF when
it is absent), and provisions enrollment-v2 keys (TLS key generated and retained
on the node;
HMAC shared secret over authenticated SSH), starts the producer, and runs one
authenticated 2,048/8 native NVFP4 handoff through the production receiver.
The node is marked healthy only after KV installs and the Mac decodes the
bounded continuation. The publication-grade three-repetition 2,048/256
local-reference cell remains available as `muser node qualify <name>`; it
does not block first use. Fresh nodes select this lane without a flag;
`--producer llamacpp` explicitly selects the kquant+DFlash research lane. The
same pipeline is available as `muser node add <user@host>`. Step-by-step reference:
[`one-button-onboarding.md`](one-button-onboarding.md).

**No second launch:** if you started with `muser up`, its setup listener stays
bound while onboarding runs. After the handoff passes, the same process loads
the verified Mac decoder and the Add Node dialog ends on **Start Mac decoder**.
Close the dialog and prompt immediately; do not stop or restart the server.

For a headless setup, run:

```sh
./target/release/muser node add user@host
./target/release/muser up
```

Bare `up` automatically selects the newest compatible healthy native
enrollment. `up --node <name>` selects a specific node when several exist.
Both resolve the verified Mac decoder and receiver configuration, require
remote prefill (no silent local fallback), and select the cross-vendor math
route automatically. The lower-level `serve --prefill auto|remote
--cluster-config ...` interface remains available for service managers.

With artifacts already present, the first cold producer activation still has
real compute work. The current connector keeps a 131,072-token request contract
with qualified chunked prefill while vLLM initializes an 8,192-token scheduler
shape, allocates the full KV budget, and runs a 2,048-token first-request
warmup. Final cold receipts for this runtime reached ready in 187–206 seconds;
the qualified 187-second run finished weights at 108 seconds. The dashboard
names each milestone and keeps an
elapsed heartbeat moving. A repeat Add Node against the same image, runtime
digest, decoder, and enrollment keeps the producer warm and runs only the
operational handoff. Use `node add ... --repair` when a live node must actually
be redeployed and restarted.

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
