#!/usr/bin/env python3
"""Restart a resident GX10 vLLM producer and wait for real readiness.

Purpose
-------
The resident producer exits fail-closed (status 75) after an engine-touched
error — including a receiver that is not listening yet — and a bare
`docker restart` is NOT enough to bring it back: the startup receipt and the
RoPE cache are created with O_EXCL, and a stale producer.sock blocks the new
bind. This script performs the full restart ritual and then waits for the
fresh startup receipt, which only appears after the model is loaded and the
warmup has run.

Run it on the GX10 itself (docker socket is local), e.g. from the Mac:

    ssh <node> python3 ~/.muser/lane/gx10/restart_resident_producer.py \
        --container muser-redhat-native-f1-83ad7a4

Options
-------
--container NAME     container to restart (required)
--timeout SECONDS    readiness deadline (default 600; model load ~2-3 min)
--dry-run            print the plan (files moved aside, socket removed) and
                     exit without touching anything

Behavior
--------
1. Derives the container's work and receipt directories from `docker
   inspect` mounts and the startup receipt / RoPE cache / socket names from
   its command line — no hardcoded lane layout.
2. Probes the target container's accelerator lease and refuses if another
   process holds it, with a `fuser` holder hint.
3. Moves the previous startup receipt and RoPE cache aside with a
   timestamped suffix (nothing is deleted), removes the stale socket.
4. `docker restart`, then polls for the fresh receipt. If the container
   exits first, the last log lines are printed and the exit status is 1.

Exit status: 0 when ready, 1 on death/timeout/bad arguments.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time


def docker(*args: str, capture: bool = True) -> str:
    result = subprocess.run(
        ["docker", *args],
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise SystemExit(f"docker {' '.join(args)} failed: {(result.stdout or '').strip()}")
    return result.stdout or ""


def container_layout(container: str) -> dict[str, Path | str]:
    """Derive host-side paths from the container's mounts and command line."""
    inspect = json.loads(docker("inspect", container))[0]
    cmd: list[str] = inspect["Config"]["Cmd"] or []
    mounts = {m["Destination"]: Path(m["Source"]) for m in inspect["Mounts"]}
    for required in ("/run/muser/work", "/receipts"):
        if required not in mounts:
            raise SystemExit(f"{container}: no mount for {required}; not a resident producer?")

    def cmd_value(flag: str) -> str:
        if flag not in cmd:
            raise SystemExit(f"{container}: command line lacks {flag}")
        return cmd[cmd.index(flag) + 1]

    def host_path(container_path: str) -> Path:
        for destination, source in mounts.items():
            if container_path == destination:
                return source
            if container_path.startswith(destination + "/"):
                return source / container_path[len(destination) + 1 :]
        raise SystemExit(f"{container}: no mount maps {container_path}")

    return {
        "work": mounts["/run/muser/work"],
        "receipt": host_path(cmd_value("--startup-receipt")),
        "rope_cache": host_path(cmd_value("--rope-cache-output")),
        "sock": host_path(cmd_value("--sock")),
        "lease_file": host_path(cmd_value("--lease-file")),
    }


def plan(layout: dict[str, Path | str], stamp: str) -> list[tuple[str, Path, Path | None]]:
    """The restart ritual as (verb, source, destination) rows."""
    rope_cache = Path(layout["rope_cache"])
    receipt = Path(layout["receipt"])
    sock = Path(layout["sock"])
    rows: list[tuple[str, Path, Path | None]] = [
        ("move-aside", rope_cache, rope_cache.with_name(f"{rope_cache.name}.stale-{stamp}")),
        ("move-aside", receipt, receipt.with_name(f"{receipt.name}.stale-{stamp}")),
        ("remove", sock, None),
    ]
    return rows


def accelerator_lease_is_free(lease_file: Path) -> bool:
    """Return whether the lease can be acquired without waiting, then release it."""
    handle = lease_file.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return True
    finally:
        handle.close()


def lease_holder_hint(lease_file: Path) -> str:
    """Return best-effort `fuser` output without weakening a held-lease refusal."""
    try:
        result = subprocess.run(
            ["fuser", "-v", str(lease_file)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except OSError as error:
        return f"fuser unavailable: {error}"
    output = (result.stdout or "").strip()
    return output or f"fuser exited {result.returncode} without output"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--container", required=True)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    layout = container_layout(args.container)
    receipt = Path(layout["receipt"])
    lease_file = Path(layout["lease_file"])
    rows = plan(layout, dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    for verb, source, destination in rows:
        target = f" -> {destination}" if destination else ""
        print(f"{verb}: {source}{target}", flush=True)
    if args.dry_run:
        print("dry run; nothing changed", flush=True)
        return 0
    if not accelerator_lease_is_free(lease_file):
        print(
            f"refusing to restart {args.container}: accelerator lease {lease_file} "
            "is held; stop the holding producer first",
            file=sys.stderr,
            flush=True,
        )
        print(f"holder hint:\n{lease_holder_hint(lease_file)}", file=sys.stderr, flush=True)
        return 1
    for verb, source, destination in rows:
        if verb == "move-aside":
            if source.exists():
                source.rename(destination)
        elif source.exists():
            source.unlink()
    docker("restart", args.container, capture=False)
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        if receipt.exists():
            print(f"ready: fresh startup receipt at {receipt}", flush=True)
            return 0
        state = docker("inspect", args.container, "--format", "{{.State.Status}}").strip()
        if state != "running":
            print(f"container exited while starting; last log lines:", flush=True)
            print(docker("logs", "--tail", "15", args.container), flush=True)
            return 1
        time.sleep(5)
    print(f"timeout after {args.timeout}s waiting for {receipt}", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
