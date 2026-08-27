# GX10 operator diagnostics

Small, dependency-free tools for diagnosing the disaggregated GX10→Mac
prefill lane. None of them touch the GPU, the model, or the release
machinery; they are safe to run against a live lab link.

Use your own producer node's SSH alias and your wired point-to-point path
between the Mac and the node. Wi-Fi must never carry a measurement: verify
direct same-subnet routes in both directions before probing.

| Tool | Question it answers |
|---|---|
| `tcp_probe.py` | Is the raw wire healthy? (throughput ceiling, both directions) |
| `durable_fsync_probe.py` | Is this volume safe for the replay ledger? (durable-write tail latency) |
| `handoff_report.py` | Where did each repetition's time go? (per-phase table from retained receipts) |
| `restart_resident_producer.py` | Bring back a fail-closed (exit 75) vLLM producer correctly. |
| `supervise_resident_producer.py` | Keep it back: restart ritual + readiness wait in a loop, with a failure latch. |

## Diagnostic flow

A handoff looks slow or unstable. Work from the bottom up:

1. **Wire**: run `tcp_probe.py` between the nodes (server on one side,
   client on the other; then swap). The pre-migration direct-link reference
   was ~9.4 Gbps single-stream; re-anchor it on the current fabric. If raw TCP
   is slow, fix the physical/driver
   layer first — nothing above it can help. On the GB10, check the
   ConnectX-7 driver is ≥ 580.142 (earlier drivers throttle the 200GbE
   ports to ~13 Gbps).
2. **Durability tail**: if TTFT is bimodal by ~1 s on random reps, run
   `durable_fsync_probe.py` against the directory that hosts
   `replay_ledger` in the receiver's cluster config. A max above ~100 ms
   means operational state is on a slow/busy volume — move it to the
   internal disk. (This exact misplacement cost the fast lane its
   stability gate; see the sealing plan's W1 findings.)
3. **Phase attribution**: run `handoff_report.py` on the packet's out-dir
   plus the wrapper command log. The docstring's field guide maps each
   phase pattern to its likely cause (producer aging, pacing pin, Mac-side
   setup, ledger reserve).
4. **Producer lifecycle**: if the resident producer died (container exited
   75 — a refused receiver connection is enough), use
   `restart_resident_producer.py`; a bare `docker restart` fails on the
   stale O_EXCL startup artifacts unless they are moved aside. For
   unattended operation, run `supervise_resident_producer.py` — it performs
   the ritual automatically and latches off after three consecutive failed
   starts instead of flapping.

## Conventions

Every script documents its purpose, usage, and exit codes in its module
docstring (`--help` prints the short form). `--json` is provided where the
output feeds other tooling. Tests live in
`scripts/tests/test_gx10_diagnostics.py` and run with the standard suite:

```sh
python3 -m unittest discover -s scripts/tests
```
