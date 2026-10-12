#!/usr/bin/env python3
"""Integrated, non-notarial Muser human test runner.

`--plan` only renders the ordered work. `--check` performs a complete
read-only preflight and writes one private receipt. `--execute` repeats that
preflight, requires it to match the supplied receipt exactly, then runs every
enabled section sequentially under one accelerator lease.
"""

from __future__ import annotations

import argparse
import fcntl
import http.client
import json
import os
from pathlib import Path
import secrets
import ssl
import subprocess
import sys
import time
from typing import Any
from urllib.parse import urlsplit

import accelerator_safe
from human_test_common import (
    RECEIPT_SCHEMA,
    PreflightError,
    artifact_path,
    atomic_json,
    canonical_bytes,
    load_config,
    preflight,
    rehash_artifacts,
    sha256_bytes,
    strict_json_file,
    validate_config_shape,
)


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
ORDER = (
    "dashboard", "real_model", "migration", "batching",
    "target_comparator", "dflash", "gx",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--receipt", type=Path,
        help="new receipt for --check; existing matching receipt for --execute",
    )
    return parser.parse_args()


def plan(config: dict[str, Any], config_hash: str) -> dict[str, Any]:
    validate_config_shape(config)
    enabled = [name for name in ORDER if config["sections"][name]]
    return {
        "schema": "muser.human-test.plan.v1",
        "run_id": config["run_id"],
        "config_sha256": config_hash,
        "mode": "non-notarial-human-test",
        "accelerator_touched": False,
        "ordered_sections": enabled,
        "single_local_accelerator_lease": True,
        "gx_mode": "read-only enrolled node status; never node smoke",
        "browser_hold": config["browser_hold"],
        "output_root": config["output_root"],
        "prohibited_outputs": ["seal", "readiness", "candidate", "tag", "publish"],
    }


def checked_receipt(
    config: dict[str, Any], config_hash: str, *, accelerator_lock_held: bool = False,
) -> dict[str, Any]:
    snapshot = preflight(
        config, config_hash, ROOT, accelerator_lock_held=accelerator_lock_held,
    )
    return {
        "schema": RECEIPT_SCHEMA,
        "status": "passed",
        "run_id": config["run_id"],
        "config_sha256": config_hash,
        "snapshot": snapshot,
        "seal_eligible": False,
        "accelerator_touched": False,
    }


def require_receipt(path: Path, expected: dict[str, Any]) -> None:
    try:
        actual = strict_json_file(path)
    except PreflightError as error:
        raise PreflightError(f"cannot read check receipt {path}: {error}") from error
    if actual.get("schema") != RECEIPT_SCHEMA or actual.get("status") != "passed":
        raise PreflightError("execution requires a passing human-test check receipt")
    if canonical_bytes(actual) != canonical_bytes(expected):
        raise PreflightError(
            "check receipt no longer matches the exact config, artifacts, scripts, host, ports, and idle state"
        )


def origin(port: int, tls: bool = False) -> str:
    return f"{'https' if tls else 'http'}://127.0.0.1:{port}"


def clean_environment(extra: dict[str, str] | None = None) -> dict[str, str]:
    allowed = ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TMPDIR")
    environment = {name: os.environ[name] for name in allowed if name in os.environ}
    if os.environ.get("MUSER_ACCELERATOR_LEASE") == "1":
        environment["MUSER_ACCELERATOR_LEASE"] = "1"
    environment.update(extra or {})
    return environment


def run_logged(
    name: str,
    command: list[str],
    output: Path,
    *,
    environment: dict[str, str] | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"RUN  {name}: {' '.join(command)}", flush=True)
    with output.open("xb") as log:
        result = subprocess.run(
            command, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, env=environment or clean_environment(), check=False,
        )
    if result.returncode:
        tail = output.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"{name} failed with {result.returncode}\n{tail}")
    print(f"PASS {name}", flush=True)


class Server:
    def __init__(
        self,
        config: dict[str, Any],
        output: Path,
        port: int,
        *,
        tls_prefix: str | None = None,
        home_name: str,
        parallel: int = 1,
    ) -> None:
        self.config = config
        self.output = output
        self.port = port
        self.tls_prefix = tls_prefix
        self.home = output / home_name
        self.parallel = parallel
        self.token = secrets.token_hex(32)
        self.process: subprocess.Popen[bytes] | None = None

    @property
    def url(self) -> str:
        return origin(self.port, self.tls_prefix is not None)

    @property
    def ca(self) -> Path | None:
        if self.tls_prefix is None:
            return None
        return artifact_path(self.config, f"tls_{self.tls_prefix}_ca")

    def command(self) -> list[str]:
        command = [
            str(artifact_path(self.config, "muser")), "serve",
            "--host", "127.0.0.1", "--port", str(self.port),
            "--api-key-file", str(artifact_path(self.config, "api_key")),
            "--model", str(artifact_path(self.config, "model")),
            "--backend", "metal", "--parallel", str(self.parallel),
            "--max-context", "4096",
            "--dflash", str(artifact_path(self.config, "dflash")),
            "--dflash-backend", "metal",
            "--mmproj", str(artifact_path(self.config, "mmproj")),
            "--mtmd-bridge", str(artifact_path(self.config, "mtmd_bridge")),
            "--benchmark-shutdown-token", self.token,
            "--benchmark-deadline-seconds", "7200",
        ]
        if self.tls_prefix is not None:
            command.extend([
                "--tls-cert", str(artifact_path(self.config, f"tls_{self.tls_prefix}_cert")),
                "--tls-key", str(artifact_path(self.config, f"tls_{self.tls_prefix}_key")),
            ])
        return command

    def start(self) -> None:
        self.home.mkdir(parents=True, mode=0o700, exist_ok=False)
        environment = clean_environment({
            "MUSER_HOME": str(self.home),
            "MUSER_GGML_METALLIB": str(artifact_path(self.config, "metallib")),
            "MUSER_DECODE_MIGRATION_CA": str(
                artifact_path(self.config, "tls_destination_ca")
            ),
        })
        log = (self.output / f"server-{self.port}.log").open("xb")
        self.process = subprocess.Popen(
            self.command(), stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, env=environment,
        )
        deadline = time.monotonic() + 1800
        parsed = urlsplit(self.url)
        context = ssl.create_default_context(cafile=str(self.ca)) if self.ca else None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"server {self.port} exited during startup")
            try:
                if context is None:
                    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=2)
                else:
                    connection = http.client.HTTPSConnection(
                        parsed.hostname, parsed.port, timeout=2, context=context,
                    )
                connection.request("GET", "/healthz")
                response = connection.getresponse()
                response.read()
                connection.close()
                if response.status == 200:
                    return
            except OSError:
                pass
            time.sleep(0.2)
        raise RuntimeError(f"server {self.port} did not become ready")

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        parsed = urlsplit(self.url)
        context = ssl.create_default_context(cafile=str(self.ca)) if self.ca else None
        try:
            if context is None:
                connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=10)
            else:
                connection = http.client.HTTPSConnection(
                    parsed.hostname, parsed.port, timeout=10, context=context,
                )
            connection.request(
                "POST", "/__muser/benchmark/shutdown", body=self.token.encode(),
                headers={"Content-Type": "application/octet-stream"},
            )
            response = connection.getresponse()
            response.read()
            connection.close()
            if response.status != 200:
                raise RuntimeError(f"cooperative shutdown returned HTTP {response.status}")
            self.process.wait(timeout=30)
            return
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            pass
        self.process.terminate()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)

    def __enter__(self) -> "Server":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


def dashboard(config: dict[str, Any], output: Path) -> None:
    environment = clean_environment({
        "MUSER_SMOKE_BIN": str(artifact_path(config, "muser")),
        "MUSER_SMOKE_PORT": str(config["ports"]["dashboard"]),
        "MUSER_SMOKE_HOLD": "0",
        "TMPDIR": str(output / "dashboard-tmp"),
        "CARGO_TARGET_DIR": str(output / "unused-dashboard-target"),
    })
    (output / "dashboard-tmp").mkdir(parents=True)
    run_logged(
        "dashboard", ["bash", str(SCRIPTS / "smoke_local_dashboard.sh")],
        output / "dashboard.log", environment=environment,
    )


def real_model(config: dict[str, Any], output: Path) -> None:
    server = Server(
        config, output, config["ports"]["engine"],
        home_name="engine-home", parallel=4,
    )
    with server:
        run_logged("real-model", [
            sys.executable, str(SCRIPTS / "smoke_real_model.py"),
            "--base-url", server.url,
            "--api-key-file", str(artifact_path(config, "api_key")),
            "--image", str(artifact_path(config, "vision_image")),
            "--output", str(output / "real-model.json"),
        ], output / "real-model.log")


def migration(config: dict[str, Any], output: Path) -> None:
    source = Server(
        config, output, config["ports"]["migration_source"],
        tls_prefix="source", home_name="migration-source-home",
    )
    destination = Server(
        config, output, config["ports"]["migration_destination"],
        tls_prefix="destination", home_name="migration-destination-home",
    )
    with destination, source:
        run_logged("migration", [
            sys.executable, str(SCRIPTS / "smoke_decode_migration.py"),
            "--source-url", source.url, "--destination-url", destination.url,
            "--source-api-key-file", str(artifact_path(config, "api_key")),
            "--destination-api-key-file", str(artifact_path(config, "api_key")),
            "--source-ca-file", str(source.ca),
            "--destination-ca-file", str(destination.ca),
            "--output", str(output / "migration.json"),
        ], output / "migration.log")


def batching(config: dict[str, Any], output: Path) -> None:
    environment = clean_environment({
        "MUSER_GGML_METALLIB": str(artifact_path(config, "metallib")),
    })
    run_logged("continuous-batching", [
        sys.executable, str(SCRIPTS / "continuous_batching_smoke.py"),
        "--model", str(artifact_path(config, "model")),
        "--muser-server", str(artifact_path(config, "muser")),
        "--prompt-token-fixture", str(artifact_path(config, "prompt_fixture")),
        "--output", str(output / "batching.json"),
        "--identity", config["run_id"],
        "--base-port", str(config["ports"]["batching_base"]),
    ], output / "batching.log", environment=environment)


def target_comparator(config: dict[str, Any], output: Path) -> None:
    run_logged("target-comparator", [
        sys.executable, str(SCRIPTS / "representative_target_smoke.py"),
        "--model", str(artifact_path(config, "model")),
        "--prompt-token-fixture", str(artifact_path(config, "prompt_fixture")),
        "--muser-server", str(artifact_path(config, "muser")),
        "--expected-muser-sha256", config["artifacts"]["muser"]["sha256"],
        "--muser-metallib", str(artifact_path(config, "metallib")),
        "--llama-server", str(artifact_path(config, "llama_server")),
        "--llama-receipt", str(artifact_path(config, "llama_receipt")),
        "--output", str(output / "target-comparator.json"),
        "--identity", config["run_id"],
        "--muser-url", origin(config["ports"]["target_muser"]),
        "--llama-url", origin(config["ports"]["target_llama"]),
    ], output / "target-comparator.log")


def dflash(config: dict[str, Any], output: Path) -> None:
    environment = clean_environment({
        "MUSER_GGML_METALLIB": str(artifact_path(config, "metallib")),
    })
    run_logged("muser-dflash", [
        str(artifact_path(config, "muser_dflash")),
        "--model", str(artifact_path(config, "model")),
        "--dflash", str(artifact_path(config, "dflash")),
        "--prompt-token-fixture", str(artifact_path(config, "prompt_fixture")),
        "--repetitions", "1", "--output-tokens", "256",
        "--verify-length", "15", "--sampled-check-tokens", "32",
        "--target-backend", "metal", "--assistant-backend", "metal",
        "--identity", config["run_id"],
    ], output / "muser-dflash.log", environment=environment)
    run_logged("llama-dflash", [
        sys.executable, str(SCRIPTS / "bench_llama_dflash.py"),
        "--server-binary", str(artifact_path(config, "llama_server")),
        "--model", str(artifact_path(config, "model")),
        "--dflash", str(artifact_path(config, "dflash")),
        "--prompt-token-fixture", str(artifact_path(config, "prompt_fixture")),
        "--depth", "2048", "--verify-length", "15",
        "--repetitions", "1", "--human-smoke",
        "--identity", config["run_id"],
        "--base-url", origin(config["ports"]["dflash"]),
    ], output / "dflash.log")
    def records(path: Path) -> list[dict[str, Any]]:
        values = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                values.append(value)
        return values

    muser_sample = next(
        value for value in records(output / "muser-dflash.log")
        if value.get("kind") == "sample"
    )
    llama_sample = next(
        value for value in records(output / "dflash.log")
        if value.get("kind") == "sample"
    )
    report = {
        "schema": "muser.human-dflash-smoke.v1",
        "notarial": False,
        "seal_eligible": False,
        "identity": config["run_id"],
        "verify_length": 15,
        "output_tokens": 256,
        "exact_target_match": muser_sample.get("exact_target_match"),
        "acceptance_rate": muser_sample.get("acceptance_rate"),
        "target_to_muser_dflash_ratio": muser_sample.get("speedup"),
        "llama_to_muser_dflash_time_ratio": (
            llama_sample["elapsed_ns"] / muser_sample["dflash_ns"]
        ),
        "muser_dflash_ns": muser_sample["dflash_ns"],
        "llama_dflash_ns": llama_sample["elapsed_ns"],
    }
    atomic_json(output / "dflash-human-summary.json", report)


def gx(config: dict[str, Any], output: Path) -> None:
    # This suite is diagnostic and may not change node qualification state.
    # Exact cluster/container/model identities were bound by preflight; the
    # existing status driver performs only a live read-only probe.
    run_logged("gx10-status", [
        str(artifact_path(config, "muser")), "node", "status", "--json",
    ], output / "gx10.log")


RUNNERS = {
    "dashboard": dashboard,
    "real_model": real_model,
    "migration": migration,
    "batching": batching,
    "target_comparator": target_comparator,
    "dflash": dflash,
    "gx": gx,
}


SECTION_ARTIFACTS = {
    "dashboard": ("muser",),
    "real_model": ("muser", "metallib", "model", "dflash", "mmproj", "mtmd_bridge", "mtmd_receipt", "vision_image", "api_key"),
    "migration": ("muser", "metallib", "model", "dflash", "mmproj", "mtmd_bridge", "mtmd_receipt", "api_key", "tls_source_cert", "tls_source_key", "tls_source_ca", "tls_destination_cert", "tls_destination_key", "tls_destination_ca"),
    "batching": ("muser", "metallib", "model", "prompt_fixture"),
    "target_comparator": ("muser", "metallib", "metallib_receipt", "model", "prompt_fixture", "llama_server", "llama_receipt"),
    "dflash": ("muser_dflash", "metallib", "model", "dflash", "prompt_fixture", "llama_server"),
    "gx": ("muser", "gx_cluster_config", "gx_container_receipt", "gx_node_registry", "model", "dflash"),
}


CHECKLIST = """Muser human browser checklist

1. Open {url} on this Mac; the dashboard must render without console errors.
2. Choose Sign in, paste the private API key from {key}, and do not copy it elsewhere.
3. Confirm Live Telemetry connects, snapshot sections populate, and metrics update.
4. Start one short chat and one streamed chat; confirm incremental text and final usage.
5. Exercise vision with the configured image and confirm a non-empty description.
6. Create/save/restore a session and verify the visible revision advances exactly once.
7. Disconnect one stream; another request must remain responsive.
8. Press Ctrl-C in this terminal when finished. This is human evidence, never a seal.
"""


def execute_locked(config: dict[str, Any]) -> None:
    output = Path(config["output_root"])
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    checklist = CHECKLIST.format(
        url=origin(config["ports"]["engine"]),
        key=artifact_path(config, "api_key"),
    )
    (output / "BROWSER_CHECKLIST.txt").write_text(checklist, encoding="utf-8")
    completed: list[str] = []
    result: dict[str, Any] = {
        "schema": "muser.human-test.result.v1", "run_id": config["run_id"],
        "status": "running", "seal_eligible": False, "completed": completed,
    }
    try:
        owners = accelerator_safe.active_gpu_processes()
        if owners:
            raise RuntimeError(f"accelerator became busy: {', '.join(owners)}")
        os.environ["MUSER_ACCELERATOR_LEASE"] = "1"
        for name in ORDER:
            if config["sections"][name]:
                rehash_artifacts(config, SECTION_ARTIFACTS[name])
                RUNNERS[name](config, output)
                completed.append(name)
        if config["browser_hold"]:
            print(checklist, flush=True)
            rehash_artifacts(config, SECTION_ARTIFACTS["real_model"])
            hold = Server(
                config, output, config["ports"]["engine"],
                home_name="browser-home", parallel=4,
            )
            with hold:
                try:
                    while True:
                        time.sleep(3600)
                except KeyboardInterrupt:
                    pass
        result["status"] = "passed"
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = str(error)
        raise
    finally:
        os.environ.pop("MUSER_ACCELERATOR_LEASE", None)
        result["result_sha256"] = sha256_bytes(canonical_bytes(result))
        atomic_json(output / "RESULT.json", result)


def main() -> int:
    args = parse_args()
    try:
        config, config_hash = load_config(args.config)
        if args.plan:
            print(json.dumps(plan(config, config_hash), indent=2, sort_keys=True))
            return 0
        if args.receipt is None:
            raise PreflightError("--receipt is required with --check and --execute")
        receipt = args.receipt.absolute()
        if receipt.resolve(strict=False) != receipt or receipt.is_symlink():
            raise PreflightError("receipt path must be normalized and may not traverse symlinks")
        output = Path(config["output_root"])
        if receipt == args.config.absolute() or receipt == output or output in receipt.parents:
            raise PreflightError("receipt must be disjoint from the config and outside output_root")
        if args.check:
            expected = checked_receipt(config, config_hash)
            atomic_json(receipt, expected)
            print(json.dumps(expected, indent=2, sort_keys=True))
            return 0
        descriptor = os.open(accelerator_safe.LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise PreflightError("accelerator lease is held by another process") from error
            # The same exclusive descriptor remains held through the final
            # section and browser hold. Re-running the full snapshot here
            # closes the receipt-to-execution race.
            expected = checked_receipt(
                config, config_hash, accelerator_lock_held=True,
            )
            require_receipt(receipt, expected)
            execute_locked(config)
        finally:
            os.close(descriptor)
        return 0
    except (PreflightError, RuntimeError, OSError) as error:
        print(f"human-test: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
