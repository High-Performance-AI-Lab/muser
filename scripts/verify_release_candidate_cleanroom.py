#!/usr/bin/env python3
"""Rebuild and smoke-test a structurally verified candidate from its source archive."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to replace clean-room report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def artifact(path: Path, record: object, name: str) -> dict:
    if not path.is_file() or path.is_symlink() or not isinstance(record, dict):
        raise RuntimeError(f"{name} artifact is missing, unsafe, or unreceipted")
    size = path.stat().st_size
    digest = sha256(path)
    if size != record.get("bytes") or digest != record.get("sha256"):
        raise RuntimeError(f"{name} artifact differs from the candidate manifest")
    return {"path": str(path.resolve()), "bytes": size, "sha256": digest}


def extract_source(archive_path: Path, destination: Path) -> Path:
    if not archive_path.is_file() or archive_path.is_symlink():
        raise RuntimeError("candidate source archive is missing or unsafe")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            relative = PurePosixPath(member.name)
            if (
                relative.is_absolute()
                or not relative.parts
                or any(part in {"", ".", ".."} for part in relative.parts)
                or member.issym()
                or member.islnk()
                or not (member.isfile() or member.isdir())
            ):
                raise RuntimeError(f"unsafe source archive member: {member.name!r}")
        archive.extractall(destination, members=members, filter="data")
    source = destination / "muser"
    if not source.is_dir() or source.is_symlink():
        raise RuntimeError("candidate source archive has no safe muser root")
    return source


def request_json(url: str, value: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(value, separators=(",", ":")).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"runtime smoke returned HTTP {response.status}")
        result = json.loads(response.read())
    if not isinstance(result, dict):
        raise RuntimeError("runtime smoke returned a non-object response")
    return result


def choose_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
        stream.bind(("127.0.0.1", 0))
        return int(stream.getsockname()[1])


def runtime_smoke(
    binary: Path,
    candidate: Path,
    models: dict[str, Path],
    log_path: Path,
) -> dict:
    port = choose_loopback_port()
    token = secrets.token_hex(32)
    command = [
        str(binary), "serve", "--host", "127.0.0.1", "--port", str(port),
        "--model", str(models["target"]), "--backend", "metal",
        "--mmproj", str(models["vision"]),
        "--mtmd-bridge", str(candidate / "lib/mtmd/libmuser_mtmd_bridge.dylib"),
        "--dflash", str(models["dflash"]),
        "--dflash-backend", "auto", "--prefill", "local", "--prefix-cache", "on",
        "--benchmark-shutdown-token", token,
        "--benchmark-deadline-seconds", "180",
    ]
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            command,
            cwd=candidate,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        base = f"http://127.0.0.1:{port}"
        ready = False
        choices: list[object] = []
        try:
            deadline = time.monotonic() + 150
            while time.monotonic() < deadline and process.poll() is None:
                try:
                    with urllib.request.urlopen(f"{base}/healthz", timeout=2) as response:
                        ready = response.status == 200
                    if ready:
                        break
                except (OSError, urllib.error.URLError):
                    time.sleep(0.5)
            if not ready:
                process.wait(timeout=40)
                raise RuntimeError(
                    f"candidate server did not become healthy (exit {process.returncode})"
                )
            response = request_json(
                f"{base}/v1/chat/completions",
                {
                    "model": "muse-glimmer-30b",
                    "messages": [
                        {"role": "user", "content": "Reply with the word ready."}
                    ],
                    "max_tokens": 1,
                    "temperature": 0,
                    "stream": False,
                },
                120,
            )
            value = response.get("choices")
            if not isinstance(value, list) or len(value) != 1:
                raise RuntimeError("candidate smoke response has no single completion choice")
            choices = value
        finally:
            if ready and process.poll() is None:
                try:
                    shutdown = urllib.request.Request(
                        f"{base}/__muser/benchmark/shutdown",
                        data=token.encode(),
                        method="POST",
                    )
                    with urllib.request.urlopen(shutdown, timeout=5) as reply:
                        if reply.status != 200:
                            raise RuntimeError("candidate refused cooperative shutdown")
                except (OSError, urllib.error.URLError):
                    # The hard in-process deadline remains the final authority.
                    pass
            if process.poll() is None:
                process.wait(timeout=190)
            log.flush()
            os.fsync(log.fileno())
        exit_code = process.returncode
    if exit_code != 0:
        raise RuntimeError(f"candidate server exited with {exit_code}")
    return {
        "loopback_only": True,
        "completion_choices": len(choices),
        "server_exit_code": exit_code,
        "log": str(log_path),
        "log_sha256": sha256(log_path),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--candidate", type=Path, required=True)
    value.add_argument("--target", type=Path, required=True)
    value.add_argument("--vision", type=Path, required=True)
    value.add_argument("--dflash", type=Path, required=True)
    value.add_argument("--out", type=Path, required=True)
    value.add_argument("--execute", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    plan = {
        "schema": "muser.release-candidate-cleanroom.v1",
        "mode": "execute" if args.execute else "plan",
        "candidate": str(args.candidate),
        "out": str(args.out),
        "offline_cargo": True,
        "loopback_runtime": True,
        "requires_accelerator_lease": True,
    }
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    try:
        if os.environ.get("MUSER_ACCELERATOR_LEASE") != "1":
            raise RuntimeError("execution requires scripts/accelerator_safe.py --execute")
        candidate = args.candidate.resolve()
        if not candidate.is_dir() or candidate.is_symlink():
            raise RuntimeError("candidate is missing or unsafe")
        structural = subprocess.run(
            [sys.executable, str(ROOT / "scripts/verify_release_candidate.py"), str(candidate)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
        )
        structural_report = json.loads(structural.stdout)
        if structural.returncode != 0 or structural_report.get("status") != "passed":
            raise RuntimeError("structural candidate verification failed")
        manifest = json.loads(
            (candidate / "evidence/release-artifacts.json").read_text(encoding="utf-8")
        )
        model_paths = {
            "target": args.target.resolve(),
            "vision": args.vision.resolve(),
            "dflash": args.dflash.resolve(),
        }
        artifacts = {
            name: artifact(path, manifest.get("artifacts", {}).get(name), name)
            for name, path in model_paths.items()
        }
        with tempfile.TemporaryDirectory(prefix="muser-candidate-cleanroom-") as temporary:
            work = Path(temporary)
            source = extract_source(candidate / "source/muser-with-kvpack.tar.gz", work)
            subprocess.run(
                [sys.executable, str(source / "scripts/audit_vendored_kvpack.py")],
                cwd=source,
                check=True,
                stdin=subprocess.DEVNULL,
            )
            environment = os.environ.copy()
            environment["CARGO_NET_OFFLINE"] = "true"
            environment["CARGO_TARGET_DIR"] = str(work / "target")
            subprocess.run(
                [
                    "cargo", "build", "--manifest-path", str(source / "Cargo.toml"),
                    "--release", "--locked", "--offline", "--all-features",
                    "-p", "muser-server",
                ],
                cwd=source,
                env=environment,
                check=True,
                stdin=subprocess.DEVNULL,
            )
            rebuilt = work / "target/release/muser"
            candidate_binary = candidate / "bin/muser"
            rebuilt_sha = sha256(rebuilt)
            candidate_sha = sha256(candidate_binary)
            if rebuilt_sha != candidate_sha:
                raise RuntimeError("clean-room binary is not byte-identical to the candidate")
            version = subprocess.run(
                [str(rebuilt), "--version"],
                cwd=work,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout.strip()
            smoke = runtime_smoke(
                rebuilt, candidate, model_paths, args.out.with_suffix(".server.log")
            )
        report = {
            **plan,
            "mode": "executed",
            "status": "passed",
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "identity": structural_report.get("identity"),
            "structural_files_verified": structural_report.get("files_verified"),
            "artifacts": artifacts,
            "candidate_binary_sha256": candidate_sha,
            "rebuilt_binary_sha256": rebuilt_sha,
            "byte_identical_rebuild": True,
            "version": version,
            "runtime": smoke,
        }
        atomic_json(args.out, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"clean-room candidate verification failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
