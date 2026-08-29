# Onboarding release-readiness map

This page maps the public Mac + GX10 NVFP4 onboarding promise to the failure
that could break it, the behavior Muser now requires, and the evidence that
guards that behavior. It is a review aid, not a substitute for the live
hardware qualification described in [`benchmarks.md`](benchmarks.md).

The supported first-run entry point is `muser up` (or **Muser.app** in the
signed disk image), followed by one **Add node** field containing
`user@host`. The same process and HTTP listener must become the inference
server. A successful wizard is therefore a stronger claim than “the remote
daemon started”: it includes a verified handoff, Metal decode, and activation
of the Mac decoder.

## Requirement-to-evidence map

| Risk | Required product behavior | Regression and release evidence |
|---|---|---|
| Wrong lane becomes the public default | A fresh node and bare `muser up` select native NVFP4. The kquant lane requires an explicit research selection. | Registry/default-selection tests in `node/registry.rs`, `node/mod.rs`, and `up.rs`; the native operational smoke refuses kquant evidence. |
| Setup requires a hidden restart | The setup listener remains bound while Add Node runs, then installs the decoder into that same server process. Inference returns a truthful starting response until activation completes. | Same-job activation and terminal-failure tests in `nodes_api.rs`; lifecycle and HTTP tests in `axum_httpd.rs` and `state.rs`; real clean-home bundle activation. |
| Browser opens before the server listens | Browser launch waits for a successful bounded listener probe and maps wildcard binds to a connectable address. | `up.rs` listener and wildcard-host tests. |
| SSH fails late or prompts invisibly | Preflight uses non-interactive key authentication, bounded connect time, host-key checking, and actionable errors before deployment. The KV callback path is checked independently from SSH. | Parser/argument tests in `node/ssh.rs`; callback-address tests in `node/preflight.rs`; real fresh-host preflight. |
| A machine runs out of storage midway | Mac memory and model-volume space are checked before acquisition. The GX10 check distinguishes its home filesystem from Docker's actual storage root and applies shared/split capacity floors. | `node/preflight.rs` validation tests and structured remote probe; clean-home hardware onboarding. |
| A large download restarts from zero or publishes partial bytes | Mac model parts and the public image fallback resume independently, carry bounded idle heartbeats, and become visible only after pinned length and SHA-256 verification. A mismatched remote file is moved aside rather than silently reused. | `node/model.rs`, `node/deploy.rs`, `bootstrap_node.sh`, and bootstrap idempotency/mismatch tests; anonymous range-read gate for every first-install asset. |
| A private or unavailable container registry blocks onboarding | The exact immutable image is tried first; an anonymously readable, equally pinned split archive is the automatic fallback. Docker must report the required final image ID. | `node/deploy.rs`, the native identity manifests, and `check_public_onboarding_assets.sh`. |
| Remote Python drifts from the release | The dependency image stays immutable while the release runtime is mounted read-only. The ordered runtime-overlay digest is checked before CUDA is touched and is stored in enrollment. | Runtime digest tests in `node/deploy.rs`, overlay tests in `test_nvfp4_runtime_overlay.py`, and staged-runtime refusal tests in `test_gx10_native_prefilld.py`. |
| Enrollment reports success with incomplete trust material | Enrollment binds the image, checkpoint, consumer, runtime digest, endpoint identity, fresh HMAC key, and both TLS leaves. Secrets remain path references and mode-restricted files. | Config-shape and identity tests in `node/enroll.rs`, `node/artifacts.rs`, and `test_gx10_native_prefilld.py`; the operational smoke uses the enrolled trust material. |
| vLLM appears frozen during real initialization | The UI receives closed, sanitized milestones for engine setup, weights, 8K chunk initialization, 128K KV allocation, and first-request warmup, plus an elapsed heartbeat every 15 seconds. Raw container output never reaches the browser. | `node/daemon.rs`, `bootstrap_node.sh`, and milestone/sanitization tests in `test_bootstrap_node.py`. |
| Startup profiles a 128K dummy batch | vLLM keeps the 131,072-token request contract but profiles an 8,192-token scheduler shape with an explicit KV budget and optional warmups disabled. | Runtime-overlay identity test and the real cold-start receipts documented in `install.md`. |
| Long prompts export incomplete KV | Scheduler chunks accumulate internally; only the final complete prompt state can activate a handoff. Preemption, resume, missing context, and warmup export fail closed. | `test_vllm_chunked_connector.py`; a real 16,384-token, two-chunk handoff; the retained 128K qualification. |
| Killing the Mac leaves the producer occupied | The producer watches the duplex control connection. Receiver loss cancels promptly; uncertain cancellation restarts the exact container and proves idle before another request. | Abort/recovery tests in `test_gx10_native_prefilld.py` and repeated live disconnect recovery. |
| One busy producer becomes a TCP backlog of timeouts | Serving admits one producer request. Contention returns retryable HTTP 429 with `producer_busy` and `Retry-After`, without poisoning the remote-health breaker. | Admission and breaker tests in `state.rs`, `openai.rs`, and the HTTP server suite. |
| A fixed timeout rejects legitimate deep prompts | Transfer budgets scale with admitted prompt depth and identify the failed phase and elapsed work. The native serving envelope remains unlimited within the qualified 128K contract. | Timeout configuration tests and the long-context native handoff qualification. |
| A retry redeploys a healthy warm node | Exact image, overlay, enrollment, decoder, and live-control identity permit a fast authenticated rejoin. `--repair` is required for a mutating recovery. | Fast-rejoin tests in `node/mod.rs`; repeated Add Node and day-two hardware runs. |
| Ejecting an installer leaves topology state pointing into it | The admitted producer manifest is atomically copied to mode-restricted per-node state before its path enters the registry. A warm fast rejoin migrates older transient paths without credential rotation or producer restart. | Durable-copy, source-removal, migration, atomicity, and symlink-refusal tests in `node/deploy.rs` and `node/registry.rs`; exact-bundle repair and status receipt. |
| Every launch rehashes 19.6 GB | Successful consumer validation is bound to file identity and digest. An unchanged file reuses the stamp; replacement or metadata change forces verification. | Validation-stamp tests in `node/model.rs`; measured day-two launch. |
| The wizard turns green before inference works | `healthy` is written only after authenticated operational handoff and bounded Metal continuation; interactive Add Node then waits for Mac decoder activation. Any failed stage is terminal and retained. | `node/smoke.rs`, `nodes_api.rs`, registry-state tests, and the clean-home first prompt. |
| Finder blocks a signed CLI or shell script as an untrusted document | The official disk image exposes only `Muser.app`; after normal application assessment its launcher asks Terminal to execute a separately signed native helper. No quarantined shell document is opened. | Native-helper resolution tests in `muser-launcher`, bundle/notary component gates, a Terminal Mach-O execution probe, and Apple's documented Gatekeeper application path. |
| An unsigned binary is presented as official | CI builds a deterministic unsigned input only. Publication requires inside-out Developer ID signatures, hardened runtime, Apple acceptance covering every code component, a stapled disk-image ticket, exact checksums, an immutable receipt, and independent Gatekeeper plus `syspolicy_check` verification. | `notarize_user_bundle.sh`, `verify_notarized_user_bundle.sh`, and the public bundle CI job. |
| A known dependency advisory ships unnoticed | CI refreshes the RustSec database and rejects vulnerabilities and yanked lockfile entries. | The pinned `cargo-audit` job and a zero-warning audit of `Cargo.lock`. |

## What the cold-start measurements mean

Download time is network-dependent and is reported separately. With all
artifacts present but the producer genuinely cold, final runs of the current
runtime reached ready in 187–206 seconds. The qualified 187-second run
finished weight loading at 108 seconds, began KV/kernel warmup at 115 seconds,
and began the real first-request warmup at 153 seconds. A warm supervised
producer avoids that startup on normal Mac restarts.

There is no supported switch that removes weight deserialization, CUDA engine
creation, KV allocation, and a real warmup while preserving the same ready
contract. The optimization is smaller startup shape plus chunked long-context
serving, not a fake ready signal.

## Scope boundary

The release claim is one 96 GB Apple Silicon decode host and one GB10/GX10
NVFP4 prefill node. Manual `user@host` entry, a single active producer, and the
absence of a de-enrollment command are explicit v1 limits in
[`one-button-onboarding.md`](one-button-onboarding.md#v1-limits). They are not
silently represented as discovery, fleet scheduling, or multi-producer HA.
