#!/usr/bin/env python3
"""Durable-write tail-latency probe for replay-ledger placement.

Purpose
-------
The GX10 handoff's commit path durably reserves each generation with
write + fsync + rename + directory-fsync before the receiver ACKs. On a
healthy local volume that reserve is sub-millisecond; on a busy or slow
volume (for example a big external evidence disk under load) the directory
fsync has a bimodal tail of ~0.7-1.0 s, which lands directly in TTFT and
presents as a random-looking bimodal stall in the handoff. This probe
reproduces the exact reserve pattern and reports the tail.

Usage
-----
    python3 scripts/gx10/durable_fsync_probe.py /path/to/dir
    python3 scripts/gx10/durable_fsync_probe.py /path/to/dir --iterations 100 \
        --max-tail-ms 50        # exit 1 if the worst reserve exceeds 50 ms

Run it against the directory that will host `replay_ledger` in the receiver
cluster config. The current convention: the ledger lives on the internal
disk (fast), while the external evidence volume holds evidence only.

Interpreting
------------
- median ~0.1-0.3 ms and max < ~5 ms: healthy; the ledger may live here.
- a max (or p99) of hundreds of ms: do NOT put operational durability state
  on this volume; point `replay_ledger` at the internal disk instead. This
  exact failure cost the fast lane its stability gate on 2026-08-18.

The probe writes and deletes small temporary files in the target directory;
it never touches existing files.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time


def probe(directory: Path, iterations: int, payload_bytes: int) -> list[float]:
    """Time each write+fsync+rename+directory-fsync reserve cycle, in ms."""
    samples_ms: list[float] = []
    dir_fd = os.open(directory, os.O_RDONLY)
    try:
        for index in range(iterations):
            temporary = directory / f".durable-fsync-probe-{os.getpid()}-{index}.tmp"
            final = directory / f"durable-fsync-probe-{os.getpid()}-{index}"
            started = time.perf_counter()
            with temporary.open("wb") as stream:
                stream.write(b"\0" * payload_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.rename(final)
            os.fsync(dir_fd)
            samples_ms.append((time.perf_counter() - started) * 1000)
            final.unlink()
    finally:
        os.close(dir_fd)
    return samples_ms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path, help="directory to probe (created files are temporary)")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--bytes", type=int, default=4096, dest="payload_bytes")
    parser.add_argument(
        "--max-tail-ms",
        type=float,
        default=100.0,
        help="exit 1 if the worst reserve exceeds this (default 100 ms)",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()
    if args.iterations <= 0 or args.payload_bytes <= 0:
        parser.error("iterations and bytes must be positive")
    if not args.directory.is_dir():
        parser.error(f"{args.directory} is not a directory")
    samples = sorted(probe(args.directory, args.iterations, args.payload_bytes))
    median = statistics.median(samples)
    p99 = samples[min(len(samples) - 1, int(len(samples) * 0.99))]
    worst = samples[-1]
    verdict = worst <= args.max_tail_ms
    if args.json:
        print(
            json.dumps(
                {
                    "schema": "muser.durable-fsync-probe.v1",
                    "directory": str(args.directory),
                    "iterations": args.iterations,
                    "payload_bytes": args.payload_bytes,
                    "min_ms": round(samples[0], 3),
                    "median_ms": round(median, 3),
                    "p99_ms": round(p99, 3),
                    "max_ms": round(worst, 3),
                    "max_tail_ms": args.max_tail_ms,
                    "verdict": "pass" if verdict else "fail",
                },
                sort_keys=True,
            )
        )
    else:
        print(
            f"reserves: n={len(samples)} min={samples[0]:.2f} median={median:.2f} "
            f"p99={p99:.2f} max={worst:.2f} ms -> "
            + ("PASS" if verdict else f"FAIL (tail beyond {args.max_tail_ms} ms; keep the replay ledger off this volume)"),
            flush=True,
        )
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
