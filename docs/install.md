# Install Muser

The Apple Silicon release archive contains the native Mac runtime and the
pinned GX10 onboarding control plane. Model weights and the producer image are
separate, resumable downloads. The released four-slot, 131K configuration is
qualified on a 96 GB M3 Ultra; smaller-memory Macs are not yet a supported
claim.

## Requirements

- A 96 GB Apple Silicon Mac running macOS 15 or newer, with at least 48 GiB
  free during first-time model assembly.
- A directly reachable GB10/GX10 with at least 96 GiB memory and 64 GiB free
  when home and Docker share a filesystem. With split storage, preflight
  requires 48 GiB under home and 32 GiB in Docker's actual storage root.
  NVIDIA driver 580 or newer, Docker, systemd, `curl`, `sha256sum`, and `zstd`.
- Passwordless key-based SSH from the Mac to the GX10. The SSH target must work
  non-interactively as `user@host`.
- Direct TCP reachability Mac→GX10 on 29591 and GX10→Mac on 29590. SSH jump
  hosts are not the KV data path.

## One button

The official Apple Silicon release is a Developer ID-signed, notarized, and
stapled disk image. Download the `.dmg` and adjacent `.sha256` file, verify it,
and open it:

```bash
shasum -a 256 -c muser-*-macos-arm64.dmg.sha256
open muser-*-macos-arm64.dmg
```

The image contains one **Muser.app**. Double-click it. The signed application
opens Terminal for progress, actionable errors, and Ctrl-C, then opens the
dashboard and runs the same command as the terminal path:

```bash
./bin/muser up
```

That is the complete interactive workflow. On a new installation the command
opens the local dashboard in setup mode. Choose **Add node**, enter the same
`user@host` target you use with SSH, and leave the process running. Muser
checks both machines, downloads and verifies missing artifacts, enrolls the
node, starts the producer, proves a real handoff, then loads the Mac decoder in
the same process and on the same HTTP port. When **Start Mac decoder** turns
green, close the dialog and send a prompt. There is no setup-server
Ctrl-C/restart transition.

On later launches the same bare command selects the newest compatible healthy
NVFP4 enrollment and resumes it automatically:

```bash
./bin/muser up
```

The dashboard and OpenAI-compatible API listen on
`http://127.0.0.1:4949`. The default loopback dashboard creates a bounded local
session automatically; there is no sign-in or API-key step. `Ctrl-C` stops the
Mac server, but the supervised GX10 producer stays warm for the next launch.
The admitted producer identity is copied atomically into
`~/.muser/nodes/<name>/`; node state never points back into the disk image or
source checkout used for setup.

For a headless terminal, the equivalent explicit workflow is:

```bash
./bin/muser node add user@host
./bin/muser up
```

The node name is optional, and `up --node <name>` is only needed when more than
one compatible enrollment exists and a specific one should be selected.

The app wrapper is deliberate. Finder routes a directly opened command-line
tool through Terminal as a document, a path Apple documents as vulnerable to
an unconditional Gatekeeper block. `Muser.app` is assessed as an application;
its native launcher then asks Terminal to execute a separately signed and
notarized Mach-O helper. The official disk image does not expose or reopen an
internal shell document. See Apple's
[Gatekeeper guidance](https://developer.apple.com/forums/thread/706379).

## What takes time on the first run

Downloads depend on the network and are not included in engine startup time.
They are resumable and admitted only after their pinned byte counts and
SHA-256 identities match:

- the 19.6 GB native Mac decoder;
- the immutable NVFP4 checkpoint on the GX10;
- the exact producer image (roughly 10 GB on the tested node), fetched by
  immutable image ID or by the equally pinned public split-archive fallback.

After those artifacts are already present, a genuinely cold producer still
has real work to do. In the current real GX10 onboarding receipt it became
ready in **187 seconds**: weight loading started at 27 seconds and finished at
108 seconds, KV/kernel warmup began at 115 seconds, first-request warmup began
at 153 seconds, and the service reported ready at 187 seconds. Hardware and
filesystem state can move those numbers, so the UI reports observed milestones
and elapsed time rather than pretending to know a percentage. Final cold
receipts for the same runtime ranged from **187 to 206 seconds**; the clean
public-bundle runs reached ready at 199 and 206 seconds, and the canonical
restore at 189 seconds.

The current runtime keeps the 131,072-token serving contract while limiting
vLLM's startup scheduler shape to 8,192 tokens. Longer prompts use the
qualified chunked-prefill connector, which exports only the final complete KV
state. This removes the old 131K dummy-batch startup cost; it does not make
weight deserialization, CUDA engine initialization, KV allocation, or the
first real kernel/request warmup optional. Muser supplies an explicit KV-cache
budget so vLLM can skip memory-capacity discovery, and disables optional JIT
and CuteDSL warmups. There is no supported safe switch that skips the remaining
work while leaving the engine ready to serve the same contract.

The Add Node dialog exposes that work as five milestones—engine setup, weight
load, 8K chunk initialization, 128K KV allocation, and first-request warmup.
The active milestone animates and receives a sanitized heartbeat every 15
seconds. Raw container logs and credentials are never sent to the browser.

Users do not write a container recipe, build vLLM, run `docker` commands, copy
certificates, or install a remote process by hand. Muser pins the dependency
image, mounts the release's verified runtime overlay read-only, and installs a
supervised user-systemd service. Re-adding a matching healthy node preserves
the warm producer, migrates any receipt path written by an older installer,
and does not rotate enrollment; `node add --repair` is the explicit
redeploy/restart path.

## Source builds and official binaries

From a source clone, build both binaries and then use the same one-command
workflow:

```bash
cargo build --release --locked -p muser-server --bin muser
cargo build --release --locked -p muser-bench --bin muser-remote-qualify --features metal
./target/release/muser up
```

`scripts/build_user_bundle.sh` creates the deterministic unsigned input. It is
useful for local/source testing but is not an official download. A release
operator with an existing Developer ID identity and `notarytool` keychain
profile produces the exact downloadable image with:

```bash
scripts/notarize_user_bundle.sh \
  --archive dist/muser-0.1.0-beta.1-macos-arm64.tar.gz \
  --identity "Developer ID Application: NAME (TEAMID)" \
  --keychain-profile PROFILE
```

Credentials are never accepted as arguments. The script signs the app,
launcher, and both helper binaries from the inside out with hardened runtime
and trusted timestamps; checks Apple's pre-submission policy; signs and
submits the exact disk image; validates that Apple's accepted ticket names
every code component; and staples the ticket. The independent gate then
requires checksums, signatures, `syspolicy_check`, Gatekeeper, the runtime
manifest, and an executable app-layout check. It emits the `.dmg`, its
`.sha256`, and a receipt binding the source archive, accepted log, submitted
image, final stapled bytes, team, and notary submission. Without all three
outputs and a passing gate, the binary must not be published as official.
