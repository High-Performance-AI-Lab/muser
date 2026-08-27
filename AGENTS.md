# AGENTS.md — working agreements for the muser repository

Muser is a standalone Muse Glimmer (52-layer, ~30B) inference engine: Apple
Silicon Metal decode, a kquant DFlash speculative lane, and a disaggregated
lane where a remote NVIDIA GB10 ("GX10") node prefills NVFP4 and hands the KV
to the Mac over an authenticated Handoff V2 transport. Rust workspace +
Python tooling under `scripts/`.

## Hard rules (fail-closed culture)

- The release lock (`release/release-lock.json`) is authoritative: no seals,
  tags, or candidates while it is in containment. Findings live in
  `release/findings-v1.json`; the feature boundary is
  `release/feature-contract-v1.json`. Do not "fix" these casually — changes
  to them change the campaign identity.
- Never weaken a fail-closed check to make a run pass. If a gate rejects
  your evidence, the evidence is wrong until proven otherwise.
- Metal, llama.cpp, Core ML, and accelerator runs go through
  `scripts/accelerator_safe.py` (dry-run by default; `--execute` to run).
  It holds `/tmp/ferrite.gpu.lock`; never bypass it on a shared machine.
- Evidence lives on an append-only external evidence volume (export scripts
  take theirs via `MUSER_RESULTS_DIR`). The durability lesson from 2026-08-18: **operational state (replay ledger,
  sockets, locks) belongs on the internal disk** — the evidence volume's
  directory-fsync tail produces bimodal ~1 s stalls in commit paths.
- Secrets hygiene: never read or copy files under `~/.muser/**/secrets` or
  pki dirs; cluster configs reference them by path.
- Prefer minimal, reviewable diffs. Match the surrounding code's style.
  Comments and docs must describe what the code does after your change.

## Build and test

```sh
cargo test --workspace --no-default-features   # CPU-safe suite
python3 -m unittest discover -s scripts/tests  # Python suite
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
```

Notes: `muser-bench` has `default = []` — binaries that touch Metal need
`--features metal` or they compile with stub fallbacks and fail at runtime
(e.g. `muser-remote-qualify`). Prefer serial workspace test runs locally;
parallel Metal tests can flake.

## Commits and pushes

- Commit and push work as you go (`git push origin main`); do not batch a
  day's work into one unrecorded state.
- Message convention: `type(scope): summary` — seen types: `feat`, `fix`,
  `docs`, `perf`, `research`, `tools`, `style`, `evidence`, `ci`, `bench`
  (e.g. `perf(gx10): raise handoff pacing ceiling to 8 Gbps`).
- Do not rewrite or force-push shared history.

## The GX10 lane (operator cheat sheet for agents)

- Producer node: use your own node's SSH alias and wired point-to-point
  path — Wi-Fi must never carry a measurement. Re-prove your raw link
  ceiling (`scripts/gx10/tcp_probe.py`) before relying on any throughput
  reference. The resident vLLM NVFP4 producer runs as a docker container;
  the Mac side receiver binds per `~/.muser/nodes/<name>/cluster.json`.
- The producer exits fail-closed (status 75) after engine-touched errors.
  A bare `docker restart` is not enough (stale O_EXCL startup receipt, RoPE
  cache, socket) — use the restart tool below.
- Container-file edits: extract the file from the container, modify, verify,
  `docker cp` back — never copy a file from repo HEAD wholesale into a
  container built from an older commit (the `muser_vllm` package drifts).

### Diagnostic tools (`scripts/gx10/`, documented in `scripts/gx10/README.md`)

- `tcp_probe.py` — raw link ceiling, both directions; ~9.4 Gbps is healthy
  on the lab's 10GbE point-to-point.
- `durable_fsync_probe.py <dir>` — tail-latency check for the reserve
  pattern (write+fsync+rename+dir-fsync); run it before pointing a replay
  ledger at a volume. Exit 1 past `--max-tail-ms`.
- `handoff_report.py --out-dir <qualifier out> [--log <wrapper log>]` —
  per-rep phase table (producer and receiver phases, wire rate, TTFT, CV)
  from retained receipts; the docstring maps phase patterns to causes.
- `restart_resident_producer.py --container <name> [--dry-run]` — the full
  fail-closed producer restart ritual with readiness wait; runs on the node.
- `supervise_resident_producer.py --container <name>` — runs the ritual
  automatically on container death and latches off after three consecutive
  failed starts; the unattended form of the restart tool.

Tests: `scripts/tests/test_gx10_diagnostics.py`.

## Pointers

- Architecture: `docs/muser-architecture.md`; launch claims (what may be
  said publicly): `docs/launch-claims.md`; sealing plan for the
  disaggregated lane: `docs/disaggregated-prefill-sealing-plan-20260818.md`.
- The campaign ledger (`docs/goal-parity-ledger-2026-08.md`) records every
  measured verdict with evidence paths — append to it, don't edit history.
- Vendored kvpack lives in `third_party/kvpack` with pinned provenance;
  audit via `scripts/audit_vendored_kvpack.py`.
