#!/usr/bin/env python3
"""Per-rep phase report for GX10 handoff qualification runs.

Purpose
-------
Turn one qualification packet's retained evidence into the table you need to
answer "where did the time go, per repetition". It reads the producer client
receipts in a qualifier out-dir (`f-*-client.json`) and, when given the
accelerator wrapper's command log, the receiver-side phases from the
`fast-performance-sample` records.

Usage
-----
    python3 scripts/gx10/handoff_report.py --out-dir /path/to/out-p4 \
        --log /path/to/p4-wrapper/YYYYMMDDThhmmssZ-<id>.command.log

Both arguments accept any run of the NVFP4 fast lane; either alone is fine
(you get fewer columns).

How to read it
--------------
- `wire` stable + `seal` bimodal by ~1 s, receiver `seal_off` constant:
  the stall is after the byte stream — historically the replay ledger's
  directory fsync on a slow volume. Run durable_fsync_probe.py on the
  ledger's directory.
- `d2h` or `first_layer` growing across reps: producer-side slowdown
  (thermals, allocator); check `docker logs` for the resident producer.
- `wire` far below what tcp_probe.py measures raw: the sender's pacing pin
  or a genuinely sick link, in that order.
- Everything flat but TTFT high: look at Mac-side session setup and the
  first local decode, not the handoff.

Exit status is always 0; this is a reporting tool, not a gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys


def cv(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.fmean(values)
    return statistics.stdev(values) / mean if mean else 0.0


def load_receipts(out_dir: Path) -> list[dict]:
    receipts = []
    for path in sorted(out_dir.glob("*-client.json")):
        receipts.append(json.loads(path.read_text()))
    if not receipts:
        raise SystemExit(f"no *-client.json receipts under {out_dir}")
    receipts.sort(key=lambda r: r["response"]["producer_receipt"]["handoff"]["generation"])
    return receipts


def load_samples(log: Path) -> dict[int, dict]:
    samples: dict[int, dict] = {}
    for line in log.read_text().splitlines():
        if "fast-performance-sample" not in line:
            continue
        record = json.loads(line[line.index("{"):])
        samples[record["repetition"]] = record
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, required=True, help="qualifier out dir with *-client.json receipts")
    parser.add_argument("--log", type=Path, help="accelerator wrapper command log (adds receiver phases and TTFT)")
    args = parser.parse_args()
    receipts = load_receipts(args.out_dir)
    samples = load_samples(args.log) if args.log else {}

    header = (
        f"{'gen':>5} {'ttft':>7} {'cut':>5} {'sched':>6} {'first':>6} {'d2h':>7} {'hash':>6} "
        f"{'pack':>6} {'seal':>7} {'wire':>6} {'gbps':>6} | {'drain':>6} {'verify':>6} "
        f"{'install':>7} {'commit':>6} {'seal_off':>8}"
    )
    print(header)
    print("-" * len(header))
    ttfts: list[float] = []
    for repetition, receipt in enumerate(receipts):
        producer = receipt["response"]["producer_receipt"]
        handoff = producer["handoff"]
        phases = producer.get("phase_ns", {})
        ms = lambda key: phases.get(key, 0) / 1e6
        wire_ms = handoff["payload_wire_ns"] / 1e6
        gbps = handoff["payload_bytes"] * 8 / handoff["payload_wire_ns"]
        sample = samples.get(repetition, {})
        ttft = sample.get("remote_ttft_ns", 0) / 1e9
        if ttft:
            ttfts.append(ttft)
        receiver = (
            f"{sample.get('receiver_segment_drain_ns', 0) / 1e6:6.0f} "
            f"{sample.get('receiver_verify_ns', 0) / 1e6:6.0f} "
            f"{sample.get('receiver_install_ns', 0) / 1e6:7.0f} "
            f"{sample.get('receiver_commit_ns', 0) / 1e6:6.0f} "
            f"{sample.get('receiver_seal_read_offset_ns', 0) / 1e6:8.0f}"
            if sample
            else f"{'-':>6} {'-':>6} {'-':>7} {'-':>6} {'-':>8}"
        )
        prefix_cut = receipt["response"].get("prefix_cut", 0)
        print(
            f"{handoff['generation']:>5} {ttft:7.3f} {prefix_cut:>5} {ms('scheduled_to_connector_start'):6.0f} "
            f"{ms('first_layer_offset'):6.0f} {ms('d2h_complete_offset'):7.0f} "
            f"{ms('host_materialize_hash'):6.0f} {ms('pack_send'):6.0f} {ms('seal'):7.0f} "
            f"{wire_ms:6.0f} {gbps:6.2f} | {receiver}"
        )
    if ttfts:
        print(
            f"\nttft: median={statistics.median(ttfts):.3f}s cv={cv(ttfts):.4f} "
            f"min={min(ttfts):.3f} max={max(ttfts):.3f} (gate: cv <= 0.02)"
        )
    print("all times are milliseconds unless suffixed with s; gbps is payload bits / producer wire-busy time")
    return 0


if __name__ == "__main__":
    sys.exit(main())
