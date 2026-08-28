#!/usr/bin/env python3
"""Dry-run-first, serialized wrapper for every accelerator invocation."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import time
import uuid


LOCK_PATH = Path("/tmp/ferrite.gpu.lock")
LEASE_FD_ENV = "MUSER_ACCELERATOR_LEASE_FD"
FORBIDDEN = {"xctrace", "gputrace", "kill", "killall", "pkill"}
GPU_PROCESS = re.compile(
    # muser-console is the telemetry dashboard: it never touches the
    # accelerator, so it is excluded the way `kill` is — categorically.
    r"(?:^|/)(?:llama-(?:bench|cli|server)|ferrite(?:-[^/]*)?|muser(?:-(?!console)[^/ ]*)?|xctrace"
    r"|(?:coreml[^/ ]*|export_dflash[^/ ]*coreml[^/ ]*|export_muse_target_coreml"
    r"|evaluate_ane)\.py)(?:$|\s)",
    re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run after printing the plan; default is dry-run",
    )
    parser.add_argument(
        "--identity", required=True, help="frozen engine/model/hardware identity"
    )
    parser.add_argument(
        "--cell", required=True, help="unique correctness or benchmark cell name"
    )
    parser.add_argument(
        "--out-dir", type=Path, help="append-only evidence directory"
    )
    parser.add_argument(
        "--result-receipt",
        type=Path,
        help="append-only atomic result receipt path (default: <out-dir>/<run-id>.result.json)",
    )
    parser.add_argument(
        "--allow-profiler",
        action="store_true",
        help=(
            "allow only a direct gputrace headless-profile attach-launched command "
            "while this wrapper holds the accelerator lease"
        ),
    )
    parser.add_argument(
        "--share-lease",
        action="store_true",
        help=(
            "pass the held lock descriptor to a nested accelerator_safe wrapper; "
            "the nested wrapper verifies ownership before borrowing it"
        ),
    )
    parser.add_argument("--quiet-seconds", type=int, default=10)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    return args


def default_out_dir() -> Path:
    override = os.environ.get("MUSER_RESULTS_DIR")
    if override:
        external = Path(override)
        if external.is_dir() and os.access(external, os.W_OK):
            return external
        raise SystemExit(f"MUSER_RESULTS_DIR is not a writable directory: {external}")
    return Path(tempfile.gettempdir()) / "muser-results"


def validate_command(command: list[str], allow_profiler: bool = False) -> None:
    lowered = {Path(token).name.lower() for token in command}
    if allow_profiler:
        required = {
            "--attach-launched",
            "--attach-after-file",
            "--process",
            "--out-dir",
        }
        if (
            Path(command[0]).name.lower() != "gputrace"
            or command[1:2] != ["headless-profile"]
            or not required.issubset(command)
            or "--" not in command
        ):
            raise SystemExit(
                "--allow-profiler requires a direct gputrace headless-profile "
                "attach-launched command with a launched child"
            )
        lowered.remove("gputrace")
    rejected = sorted(lowered & FORBIDDEN)
    if rejected:
        raise SystemExit(
            f"forbidden accelerator command component: {', '.join(rejected)}"
        )


def active_gpu_processes() -> list[str]:
    """Return candidate accelerator users, excluding this wrapper process."""
    listing = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,comm=,args="],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    parsed: list[tuple[int, int, str, str]] = []
    for line in listing.splitlines():
        fields = line.strip().split(maxsplit=3)
        if len(fields) == 4:
            parsed.append((int(fields[0]), int(fields[1]), fields[2], fields[3]))

    parents = {pid: ppid for pid, ppid, _, _ in parsed}
    own = {os.getpid()}
    pid = os.getppid()
    while pid > 1 and pid not in own:
        own.add(pid)
        pid = parents.get(pid, 1)
    violations = []
    for pid, _, executable, command in parsed:
        if pid in own or not process_uses_accelerator(executable, command):
            continue
        if _is_accelerator_free_server(command):
            continue
        violations.append(f"{pid} {command}")
    return violations


def process_uses_accelerator(executable: str, command: str) -> bool:
    """Classify the live executable, not text embedded in a shell wrapper.

    Shells and build tools often carry an entire child command in ``args``;
    the eventual child is independently visible to ``ps``. Matching wrapper
    text produced false positives for CPU-only work in a directory named
    ``ferrite-rs``. Python is the one intentional exception because its
    executable name does not identify the accelerator script it is running.
    """
    name = Path(executable).name
    if GPU_PROCESS.search(name):
        return True
    first = command.split(maxsplit=1)[0] if command else ""
    if GPU_PROCESS.search(Path(first).name):
        return True
    return name.lower().startswith("python") and GPU_PROCESS.search(command) is not None


_SERVE_PORT = re.compile(r"(?:^|/)muser\s+serve\b.*?--port\s+(\d+)")


def _is_accelerator_free_server(command: str) -> bool:
    """A modelless `muser serve` (dashboard only) holds no accelerator; its
    /healthz says so explicitly. Anything unreachable or with a model loaded
    stays a violation — verification, not trust in the process name."""
    match = _SERVE_PORT.search(command)
    if match is None:
        return False
    import json as _json
    import urllib.request

    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{match.group(1)}/healthz", timeout=1
        ) as response:
            health = _json.loads(response.read().decode())
    except Exception:
        return False
    return health.get("accelerator_in_use") is False


def append_record(path: Path, record: dict[str, object]) -> None:
    encoded = (json.dumps(record, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        written = 0
        while written < len(encoded):
            written += os.write(fd, encoded[written:])
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_result(path: Path, record: dict[str, object]) -> None:
    """Publish one immutable receipt with file and directory durability."""
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to replace result receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    encoded = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        written = 0
        while written < len(encoded):
            written += os.write(fd, encoded[written:])
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.rename(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def fsync_directory(path: Path) -> None:
    directory = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def under_temp_dir(path: Path) -> bool:
    resolved = path.resolve()
    temp_root = Path(tempfile.gettempdir()).resolve()
    return resolved == temp_root or temp_root in resolved.parents


def inherited_lease_fd() -> int | None:
    """Return a verified inherited lease descriptor, if one was delegated.

    The environment is only a locator. Acceptance is tied to the open file
    description: a separate descriptor for the same path, or an unlocked
    descriptor with a forged environment variable, is refused.
    """
    raw = os.environ.get(LEASE_FD_ENV)
    if raw is None:
        return None
    if os.environ.get("MUSER_ACCELERATOR_LEASE") != "1":
        raise RuntimeError(
            f"{LEASE_FD_ENV} is set without MUSER_ACCELERATOR_LEASE=1"
        )
    if not raw.isascii() or not raw.isdecimal():
        raise RuntimeError(f"invalid inherited accelerator lease descriptor: {raw!r}")
    descriptor = int(raw)
    if descriptor <= 2:
        raise RuntimeError("inherited accelerator lease descriptor must be above stderr")
    verify_inherited_lease(descriptor)
    return descriptor


def verify_inherited_lease(descriptor: int) -> None:
    try:
        inherited = os.fstat(descriptor)
        expected = LOCK_PATH.stat()
    except OSError as error:
        raise RuntimeError("inherited accelerator lease descriptor is not open") from error
    if not stat.S_ISREG(inherited.st_mode) or (
        inherited.st_dev,
        inherited.st_ino,
    ) != (expected.st_dev, expected.st_ino):
        raise RuntimeError(
            f"inherited accelerator lease descriptor is not {LOCK_PATH}"
        )

    # A new open file description must conflict, while reasserting LOCK_EX on
    # the inherited one must succeed. Together these checks distinguish the
    # actual owning description from a second descriptor for the same inode.
    probe = LOCK_PATH.open("a+")
    try:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(probe, fcntl.LOCK_UN)
            raise RuntimeError("inherited accelerator lease descriptor is not locked")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "inherited accelerator lease descriptor does not own the lock"
            ) from error
    finally:
        probe.close()


def main() -> int:
    args = parse_args()
    validate_command(args.command, args.allow_profiler)
    out_dir = args.out_dir or default_out_dir()
    if under_temp_dir(out_dir):
        print(
            f"WARNING: evidence out-dir {out_dir} resolves under the system temp "
            "directory and is not durable evidence storage.",
            file=sys.stderr,
        )
        if args.execute:
            raise SystemExit(
                f"refusing --execute: out-dir {out_dir} resolves under "
                f"{tempfile.gettempdir()}; pass a durable --out-dir"
            )
    planned = {
        "mode": "execute" if args.execute else "dry-run",
        "identity": args.identity,
        "cell": args.cell,
        "command": args.command,
        "lock": str(LOCK_PATH),
        "quiet_seconds_before": args.quiet_seconds,
        "quiet_seconds_after": args.quiet_seconds,
        "out_dir": str(out_dir),
        "expected_records": ["command.log", "result.json", "records.jsonl"],
        "automatic_retry": False,
        "profiler_authorized": args.allow_profiler,
        "lease_source": "inherited" if LEASE_FD_ENV in os.environ else "acquired",
        "lease_shared_with_child": args.share_lease,
    }
    print(json.dumps(planned, indent=2, sort_keys=True))
    if not args.execute:
        return 0
    if args.quiet_seconds < 10:
        raise SystemExit("execution requires quiet periods of at least 10 seconds")

    out_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = args.result_receipt
    if receipt_path is not None and receipt_path.parent.resolve() != out_dir.resolve():
        raise RuntimeError("--result-receipt must be directly under --out-dir")
    run_id = (
        f"{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-"
        f"{uuid.uuid4().hex}"
    )
    log_path = out_dir / f"{run_id}.command.log"
    receipt_path = receipt_path or out_dir / f"{run_id}.result.json"
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    exit_status = 3
    lock_stream = None
    try:
        lock_descriptor = inherited_lease_fd()
        if lock_descriptor is None:
            lock_stream = LOCK_PATH.open("a+")
            lock_descriptor = lock_stream.fileno()
            try:
                fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    f"accelerator lease unavailable: {LOCK_PATH}"
                ) from error
        active = active_gpu_processes()
        if active:
            raise RuntimeError("another GPU process is present:\n" + "\n".join(active))

        time.sleep(args.quiet_seconds)
        child_env = os.environ.copy()
        child_env["MUSER_ACCELERATOR_LEASE"] = "1"
        child_env["MUSER_ACCELERATOR_CELL"] = args.cell
        pass_fds: tuple[int, ...] = ()
        if args.share_lease:
            child_env[LEASE_FD_ENV] = str(lock_descriptor)
            pass_fds = (lock_descriptor,)
        else:
            child_env.pop(LEASE_FD_ENV, None)
        with log_path.open("xb") as log:
            result = subprocess.run(
                args.command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=child_env,
                pass_fds=pass_fds,
            )
            log.flush()
            os.fsync(log.fileno())
        fsync_directory(out_dir)
        exit_status = result.returncode
        time.sleep(args.quiet_seconds)
    except BaseException as error:
        # Admission failures are still executions from the caller's point of
        # view. Retain their exact diagnostic and publish the same bound
        # result shape as a child-process failure; callers must never infer
        # success or retry class from missing evidence.
        exit_status = 75 if isinstance(error, (RuntimeError, BlockingIOError)) else 3
        if not log_path.exists():
            with log_path.open("xb") as log:
                log.write((str(error) + "\n").encode())
                log.flush()
                os.fsync(log.fileno())
            fsync_directory(out_dir)
    finally:
        if lock_stream is not None:
            lock_stream.close()

    finished = dt.datetime.now(dt.timezone.utc).isoformat()
    record = {
        **planned,
        "mode": "executed",
        "run_id": run_id,
        "started_at": started,
        "finished_at": finished,
        "exit_code": exit_status,
        "log": str(log_path),
    }
    receipt = {
        "schema": "muser.accelerator-result.v1",
        "run_id": run_id,
        "identity": args.identity,
        "cell": args.cell,
        "command": args.command,
        "exit_status": exit_status,
        "command_log": str(log_path.resolve()),
        "started_at": started,
        "finished_at": finished,
        "lease_source": planned["lease_source"],
        "lease_shared_with_child": args.share_lease,
    }
    publish_result(receipt_path, receipt)
    record["result_receipt"] = str(receipt_path)
    append_record(out_dir / "records.jsonl", record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return exit_status


if __name__ == "__main__":
    sys.exit(main())
