#!/usr/bin/env python3
"""Supervisor for the resident GX10 vLLM producer container.

Purpose
-------
The resident producer exits fail-closed (status 75) after any engine-touched
error — a refused receiver connection included — and a bare `docker restart`
does not bring it back (stale O_EXCL startup receipt, RoPE cache, and socket
must be moved aside first; see restart_resident_producer.py). Left alone,
one transient failure takes the lane down until an operator notices. This
supervisor owns that lifecycle: it watches the container, performs the full
restart ritual on death, waits for the fresh startup receipt (real readiness,
not process liveness), and latches off after too many consecutive failures
rather than flapping forever.

Run it on the GX10 itself, typically under systemd or tmux:

    python3 supervise_resident_producer.py --container muser-redhat-native-f1-83ad7a4

Options
-------
--container NAME               container to supervise (required)
--max-consecutive-failures N   latch off after N failed starts (default 3)
--backoff-seconds N            base backoff between restarts (default 30,
                               doubled per consecutive failure)
--readiness-timeout SECONDS    per-restart readiness deadline (default 600)
--once                         supervise a single restart cycle, then exit
                               (useful for smoke tests)

Exit status: 0 on clean stop (SIGTERM/SIGINT) or --once success; 1 when the
failure latch trips or the container cannot be revived.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import time

from restart_resident_producer import container_layout, plan


def docker(*args: str) -> str:
    result = subprocess.run(
        ["docker", *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    if result.returncode != 0:
        raise RuntimeError(f"docker {' '.join(args)} failed: {(result.stdout or '').strip()}")
    return result.stdout or ""


def container_state(container: str) -> str:
    return docker("inspect", container, "--format", "{{.State.Status}}").strip()


def log(message: str) -> None:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"{stamp} supervisor: {message}", flush=True)


def decide(consecutive_failures: int, max_failures: int) -> str:
    """Latch off at the failure ceiling; otherwise restart."""
    return "latch" if consecutive_failures >= max_failures else "restart"


def await_readiness(container: str, receipt: Path, timeout: int) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if receipt.exists():
            return True
        if container_state(container) != "running":
            return False
        time.sleep(5)
    return False


def supervise(container: str, max_failures: int, backoff: int, timeout: int, once: bool) -> int:
    layout = container_layout(container)
    receipt = Path(layout["receipt"])
    consecutive = 0
    log(f"supervising {container} (receipt: {receipt})")
    while True:
        state = container_state(container)
        if state != "running":
            log(f"container is {state}; performing the restart ritual")
            for verb, source, destination in plan(layout, dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")):
                source = Path(source)
                if verb == "move-aside":
                    if source.exists():
                        source.rename(destination)
                elif source.exists():
                    source.unlink()
            docker("restart", container)
        if await_readiness(container, receipt, timeout):
            if consecutive:
                log("producer ready; failure counter reset")
            consecutive = 0
            if once:
                return 0
        else:
            consecutive += 1
            log(
                f"restart did not reach readiness (consecutive failures: {consecutive}); "
                + (decide(consecutive, max_failures) == "latch" and "latching off" or "backing off")
            )
            if decide(consecutive, max_failures) == "latch":
                log(f"last container log lines:\n{docker('logs', '--tail', '15', container)}")
                return 1
            time.sleep(backoff * (1 << (consecutive - 1)))
            continue
        # Steady state: watch for the next death.
        while container_state(container) == "running":
            time.sleep(5)
        log("container exited while serving")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--container", required=True)
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    parser.add_argument("--backoff-seconds", type=int, default=30)
    parser.add_argument("--readiness-timeout", type=int, default=600)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.max_consecutive_failures < 1 or args.backoff_seconds < 1 or args.readiness_timeout < 10:
        parser.error("failures/backoff must be positive and the readiness timeout >= 10")
    try:
        return supervise(
            args.container,
            args.max_consecutive_failures,
            args.backoff_seconds,
            args.readiness_timeout,
            args.once,
        )
    except KeyboardInterrupt:
        log("supervisor stopped by operator; container left running")
        return 0


if __name__ == "__main__":
    sys.exit(main())
