#!/usr/bin/env python3
"""Strict configuration and preflight helpers for the human test runner."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import stat
import subprocess
import sys
from typing import Any

import accelerator_safe

SCHEMA = "muser.human-test.config.v1"
RECEIPT_SCHEMA = "muser.human-test.check.v1"
PINNED_LLAMA = "89e0aa6fd362617d9073e0dafc18e41241521572"
EXPECTED_MUSER_VERSION = "muser 0.1.0-beta.1"
NO_ANE_BUILD_MARKER = b"this binary was built without the ane-coreml feature"
PINNED_ARTIFACTS = {
    "model": (16_756_681_056, "7e9b74b7c8875e9e265695df9613bf6290f2392e479ce740495a129019c488d8"),
    "mmproj": (1_400_328_928, "f48b452316f9b213758e8659444029b961a24a07f99a1abb2a9f88b06f7c00c6"),
    "dflash": (1_631_205_312, "27d9a805fa29b943cfb6ad4843367cd4eaaaf06bd452d8cc3e00a2cd18a677bc"),
}
DRIVERS = (
    "smoke_local_dashboard.sh", "smoke_real_model.py", "smoke_decode_migration.py",
    "continuous_batching_smoke.py", "representative_target_smoke.py", "bench_llama_dflash.py",
)
PORT_NAMES = (
    "dashboard", "engine", "migration_source", "migration_destination",
    "batching_base", "target_muser", "target_llama", "dflash",
)
REQUIRED_ARTIFACTS = (
    "muser", "muser_dflash", "metallib", "metallib_receipt", "model", "dflash",
    "mmproj", "mtmd_bridge", "mtmd_receipt", "vision_image", "api_key",
    "llama_server", "llama_receipt", "prompt_fixture", "tls_source_cert",
    "tls_source_key", "tls_source_ca", "tls_destination_cert", "tls_destination_key",
    "tls_destination_ca", "gx_cluster_config", "gx_container_receipt",
    "gx_node_registry", "release_manifest",
)


class PreflightError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise PreflightError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    try:
        if path.is_symlink() or path.resolve(strict=False) != path.absolute():
            raise PreflightError("config path must be normalized and may not traverse symlinks")
        raw = path.read_bytes()
        value = json.loads(raw, object_pairs_hook=unique_object)
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError(f"cannot read config {path}: {error}") from error
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise PreflightError(f"config schema must be {SCHEMA}")
    allowed = {
        "schema", "run_id", "requirements", "artifacts", "ports",
        "output_root", "sections", "browser_hold",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise PreflightError(f"unknown config fields: {', '.join(unknown)}")
    if not isinstance(value.get("run_id"), str) or not value["run_id"]:
        raise PreflightError("run_id must be a non-empty string")
    return value, sha256_bytes(canonical_bytes(value))


def _artifact(config: dict[str, Any], name: str) -> dict[str, Any]:
    artifacts = config.get("artifacts")
    if not isinstance(artifacts, dict) or not isinstance(artifacts.get(name), dict):
        raise PreflightError(f"missing artifact declaration: {name}")
    item = artifacts[name]
    allowed = {"path", "bytes", "sha256", "mode"}
    unknown = sorted(set(item) - allowed)
    if unknown:
        raise PreflightError(f"artifact {name} has unknown fields: {', '.join(unknown)}")
    return item


def artifact_path(config: dict[str, Any], name: str) -> Path:
    path = _artifact(config, name).get("path")
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise PreflightError(f"artifact {name} path must be absolute")
    result = Path(path)
    if result.resolve(strict=False) != result:
        raise PreflightError(f"artifact {name} path traverses a symlink or is not normalized")
    return result


def _check_artifact(config: dict[str, Any], name: str) -> dict[str, Any]:
    item = _artifact(config, name)
    path = artifact_path(config, name)
    if not path.is_file() or path.is_symlink():
        raise PreflightError(f"artifact {name} is missing or unsafe: {path}")
    expected_bytes = item.get("bytes")
    expected_hash = item.get("sha256")
    expected_mode = item.get("mode")
    if not isinstance(expected_bytes, int) or expected_bytes < 0:
        raise PreflightError(f"artifact {name} bytes must be an integer")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise PreflightError(f"artifact {name} sha256 must be 64 hex characters")
    try:
        int(expected_hash, 16)
    except ValueError as error:
        raise PreflightError(f"artifact {name} sha256 is not hexadecimal") from error
    if not isinstance(expected_mode, str) or len(expected_mode) != 4:
        raise PreflightError(f"artifact {name} mode must be four octal digits")
    try:
        wanted_mode = int(expected_mode, 8)
    except ValueError as error:
        raise PreflightError(f"artifact {name} mode is not octal") from error
    actual = path.stat()
    if actual.st_size != expected_bytes:
        raise PreflightError(f"artifact {name} byte-size mismatch")
    actual_mode = stat.S_IMODE(actual.st_mode)
    if actual_mode != wanted_mode:
        raise PreflightError(
            f"artifact {name} mode mismatch: {actual_mode:04o} != {wanted_mode:04o}"
        )
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise PreflightError(f"artifact {name} SHA-256 mismatch")
    return {
        "path": str(path), "bytes": actual.st_size,
        "sha256": actual_hash, "mode": f"{actual_mode:04o}",
    }


def _command(args: list[str], label: str) -> str:
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-2000:]
        raise PreflightError(f"{label} failed: {detail}")
    return result.stdout


def _check_host(config: dict[str, Any], output: Path) -> dict[str, Any]:
    requirements = config.get("requirements")
    if not isinstance(requirements, dict):
        raise PreflightError("requirements must be an object")
    allowed = {"min_ram_bytes", "min_free_disk_bytes"}
    unknown = sorted(set(requirements) - allowed)
    if unknown:
        raise PreflightError(f"unknown requirements: {', '.join(unknown)}")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise PreflightError("human model smoke requires arm64 macOS")
    ambient = sorted(name for name in os.environ if name.startswith("MUSER_"))
    if ambient:
        raise PreflightError(
            "ambient MUSER_* variables are forbidden; configuration must be explicit: "
            + ", ".join(ambient)
        )
    ram = int(_command(["sysctl", "-n", "hw.memsize"], "RAM query").strip())
    min_ram = requirements.get("min_ram_bytes")
    min_disk = requirements.get("min_free_disk_bytes")
    if not isinstance(min_ram, int) or not isinstance(min_disk, int):
        raise PreflightError("RAM/disk requirements must be integers")
    probe = output.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if ram < min_ram:
        raise PreflightError(f"RAM {ram} is below required {min_ram}")
    if free < min_disk:
        raise PreflightError(f"free disk {free} is below required {min_disk}")
    # Actual free space is deliberately not receipt-bound: it changes between
    # check and execute. Both modes enforce the live threshold, while the
    # stable snapshot binds the requirement that was checked.
    return {
        "os": "macOS", "architecture": "arm64",
        "min_ram_bytes": min_ram, "min_free_disk_bytes": min_disk,
        "requirements_met": True,
    }


def _check_ports(config: dict[str, Any]) -> dict[str, int]:
    ports = config.get("ports")
    if not isinstance(ports, dict) or set(ports) != set(PORT_NAMES):
        raise PreflightError("ports must contain exactly: " + ", ".join(PORT_NAMES))
    values: dict[str, int] = {}
    for name in PORT_NAMES:
        value = ports[name]
        if not isinstance(value, int) or not 1024 <= value <= 65535:
            raise PreflightError(f"port {name} is outside 1024..65535")
        values[name] = value
    all_ports = {name: value for name, value in values.items() if name != "batching_base"}
    all_ports.update({f"batching_{parallel}": values["batching_base"] + parallel for parallel in (1, 2, 4)})
    if len(set(all_ports.values())) != len(all_ports):
        raise PreflightError("all configured ports must be distinct")
    sockets = []
    try:
        for name, value in all_ports.items():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
            try:
                sock.bind(("127.0.0.1", value))
            except OSError as error:
                sock.close()
                raise PreflightError(f"port {name} ({value}) is unavailable: {error}") from error
            sockets.append(sock)
    finally:
        for sock in sockets:
            sock.close()
    return values


def _check_accelerator_idle() -> dict[str, Any]:
    descriptor = os.open(accelerator_safe.LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise PreflightError("accelerator lease is held by another process") from error
        owners = accelerator_safe.active_gpu_processes()
        if owners:
            raise PreflightError(f"accelerator process is active: {', '.join(owners)}")
    finally:
        os.close(descriptor)
    return {"lease_path": str(accelerator_safe.LOCK_PATH), "active_owners": []}


def _check_binary(config: dict[str, Any]) -> dict[str, Any]:
    binary = artifact_path(config, "muser")
    description = _command(["file", str(binary)], "binary architecture")
    links = _command(["otool", "-L", str(binary)], "binary link audit")
    help_text = _command([str(binary), "serve", "--help"], "binary serve help")
    if "Mach-O 64-bit executable arm64" not in description:
        raise PreflightError("Muser binary is not an arm64 Mach-O executable")
    if "Metal.framework" not in links:
        raise PreflightError("Muser binary is not linked with the default Metal feature")
    required = ("[default: auto]", "auto, cpu, metal", "--dflash-backend")
    if any(value not in help_text for value in required):
        raise PreflightError("Muser binary does not expose the expected default-Metal CLI")
    if NO_ANE_BUILD_MARKER not in binary.read_bytes():
        raise PreflightError("Muser binary does not prove the default no-ANE feature identity")
    version = _command([str(binary), "--version"], "binary version").strip()
    if version != EXPECTED_MUSER_VERSION:
        raise PreflightError(f"Muser binary version mismatch: {version}")
    qualifier = artifact_path(config, "muser_dflash")
    qualifier_description = _command(["file", str(qualifier)], "DFlash qualifier architecture")
    qualifier_links = _command(["otool", "-L", str(qualifier)], "DFlash qualifier link audit")
    build_info_raw = _command([str(qualifier), "--build-info"], "DFlash qualifier build-info")
    try:
        build_info = json.loads(build_info_raw)
    except json.JSONDecodeError as error:
        raise PreflightError("Muser DFlash qualifier build-info is not JSON") from error
    if (
        "Mach-O 64-bit executable arm64" not in qualifier_description
        or "Metal.framework" not in qualifier_links
    ):
        raise PreflightError("Muser DFlash qualifier is not the expected Metal executable")
    validate_dflash_build_info(build_info)
    return {
        "mach_o_arm64": True, "metal_linked": True,
        "backend_default": "auto", "ane_coreml": False, "version": version,
        "dflash_qualifier": build_info,
    }


def validate_dflash_build_info(value: Any) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "version", "metal_feature"}
        or value.get("schema") != "muser.dflash-qualify.build-info.v1"
        or value.get("version") != "0.1.0-beta.1"
        or value.get("metal_feature") is not True
    ):
        raise PreflightError("Muser DFlash qualifier does not prove its own metal feature is enabled")


def _check_receipts(config: dict[str, Any]) -> dict[str, Any]:
    manifest = strict_json_file(artifact_path(config, "release_manifest"))
    expected_manifest = {
        "target": PINNED_ARTIFACTS["model"],
        "vision": PINNED_ARTIFACTS["mmproj"],
        "dflash": PINNED_ARTIFACTS["dflash"],
    }
    if manifest.get("schema") != "muser.release-artifacts.v2":
        raise PreflightError("normative release artifact manifest schema is invalid")
    for role, expected_pin in expected_manifest.items():
        value = manifest.get("artifacts", {}).get(role, {})
        if (value.get("bytes"), value.get("sha256")) != expected_pin:
            raise PreflightError(f"normative release manifest has the wrong {role} identity")
    path_roles = {"model": "target", "mmproj": "vision", "dflash": "dflash"}
    for artifact_name, role in path_roles.items():
        if artifact_path(config, artifact_name).name != manifest["artifacts"][role].get("filename"):
            raise PreflightError(f"{artifact_name} does not use its canonical manifest filename")
    llama = strict_json_file(artifact_path(config, "llama_receipt"))
    llama_artifact = _artifact(config, "llama_server")
    expected = llama.get("artifacts", {}).get("llama-server", {})
    if (
        llama.get("schema") != "muser.llama_comparator.source_receipt.v3"
        or llama.get("executed") is not False
        or llama.get("source_commit") != PINNED_LLAMA
        or llama.get("build", {}).get("metal") is not True
        or expected.get("bytes") != llama_artifact.get("bytes")
        or expected.get("sha256") != llama_artifact.get("sha256")
    ):
        raise PreflightError("llama comparator receipt/source/binary pins do not match")
    metal = strict_json_file(artifact_path(config, "metallib_receipt"))
    metallib = _artifact(config, "metallib")
    if (
        metal.get("schema") != "muser.llama_metallib.source_receipt.v1"
        or metal.get("source_commit") != PINNED_LLAMA
        or metal.get("artifact_name") != Path(metallib["path"]).name
        or metal.get("binary_size_bytes") != metallib.get("bytes")
        or metal.get("binary_sha256") != metallib.get("sha256")
    ):
        raise PreflightError("metallib receipt/source/artifact pins do not match")
    gx = strict_json_file(artifact_path(config, "gx_container_receipt"))
    if gx.get("schema") != "muser.gx10-container.receipt.v1" or gx.get("status") != "built":
        raise PreflightError("GX container receipt is not a built v1 receipt")
    cluster = strict_json_file(artifact_path(config, "gx_cluster_config"))
    if (
        cluster.get("identity", {}).get("model_sha256") != PINNED_ARTIFACTS["model"][1]
        or cluster.get("dflash_identity_sha256") != PINNED_ARTIFACTS["dflash"][1]
    ):
        raise PreflightError("GX cluster config does not bind the exact target and DFlash identities")
    if gx.get("adapter_sha256") != cluster.get("identity", {}).get("adapter_sha256"):
        raise PreflightError("GX container and enrolled cluster adapter identities differ")
    if config["sections"]["gx"]:
        expected_cluster = Path(os.environ["HOME"]) / ".muser/nodes/gx10/cluster.json"
        expected_registry = Path(os.environ["HOME"]) / ".muser/nodes.toml"
        if artifact_path(config, "gx_cluster_config") != expected_cluster:
            raise PreflightError("enabled GX status must bind the exact enrolled gx10 cluster config")
        if artifact_path(config, "gx_node_registry") != expected_registry:
            raise PreflightError("enabled GX status must bind the exact node registry")
    return {
        "llama_source": PINNED_LLAMA, "comparator_schema": llama["schema"],
        "metallib_schema": metal["schema"], "gx_schema": gx["schema"],
        "artifact_manifest_schema": manifest["schema"],
    }


def _check_mtmd_package(config: dict[str, Any]) -> dict[str, Any]:
    receipt_path = artifact_path(config, "mtmd_receipt")
    bridge = artifact_path(config, "mtmd_bridge")
    receipt = strict_json_file(receipt_path)
    declared = receipt.get("artifacts")
    if (
        receipt.get("schema") != "muser.mtmd_bridge.receipt.v2"
        or receipt.get("status") != "built"
        or receipt.get("llama_commit") != PINNED_LLAMA
        or receipt.get("executed") is not False
        or not isinstance(declared, dict)
        or "libmuser_mtmd_bridge.dylib" not in declared
        or bridge.parent != receipt_path.parent
    ):
        raise PreflightError("mtmd package receipt or pinned source identity is invalid")
    package = {}
    for name, expected in sorted(declared.items()):
        path = receipt_path.parent / name
        if path.resolve(strict=False) != path or not path.is_file() or path.is_symlink():
            raise PreflightError(f"mtmd dependency is missing or unsafe: {path}")
        if not isinstance(expected, dict):
            raise PreflightError(f"mtmd receipt entry is not an object: {name}")
        actual = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if actual["bytes"] != expected.get("bytes") or actual["sha256"] != expected.get("sha256"):
            raise PreflightError(f"mtmd dependency differs from its receipt: {name}")
        package[name] = actual
    if package[bridge.name]["sha256"] != _artifact(config, "mtmd_bridge")["sha256"]:
        raise PreflightError("configured mtmd bridge differs from its package receipt")
    return package


def _check_tls(config: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for prefix in ("source", "destination"):
        cert = artifact_path(config, f"tls_{prefix}_cert")
        key = artifact_path(config, f"tls_{prefix}_key")
        ca = artifact_path(config, f"tls_{prefix}_ca")
        _command(["openssl", "verify", "-CAfile", str(ca), str(cert)], f"{prefix} CA chain")
        _command(["openssl", "x509", "-in", str(cert), "-noout", "-checkip", "127.0.0.1"], f"{prefix} IP SAN")
        cert_key = _command(["openssl", "x509", "-in", str(cert), "-pubkey", "-noout"], f"{prefix} certificate public key")
        private_key = _command(["openssl", "pkey", "-in", str(key), "-pubout"], f"{prefix} private key")
        if cert_key != private_key:
            raise PreflightError(f"{prefix} TLS certificate and private key do not match")
        result[prefix] = {
            "ip_san": "127.0.0.1", "key_matches": True,
            "outbound_trust": "MUSER_DECODE_MIGRATION_CA",
        }
    return result


def _check_fixture_and_key(config: dict[str, Any]) -> dict[str, Any]:
    key_raw = artifact_path(config, "api_key").read_text(encoding="utf-8")
    key = key_raw.strip()
    if len(key) < 32 or any(character.isspace() for character in key):
        raise PreflightError("API key must contain one non-whitespace value of at least 32 characters")
    try:
        tokens = [int(item) for item in artifact_path(config, "prompt_fixture").read_bytes().split()]
    except ValueError as error:
        raise PreflightError("prompt fixture contains a non-integer token") from error
    if len(tokens) != 2048 or any(not 0 <= token <= 0xFFFFFFFF for token in tokens):
        raise PreflightError("prompt fixture must contain exactly 2048 valid u32 token IDs")
    return {"api_key_characters": len(key), "prompt_tokens": len(tokens)}


def _check_drivers(repo: Path) -> dict[str, Any]:
    tools = ("bash", "cargo", "curl", "file", "jq", "node", "openssl", "otool", "python3")
    resolved_tools = {}
    for name in tools:
        path = shutil.which(name)
        if path is None:
            raise PreflightError(f"required human-test tool is unavailable: {name}")
        resolved_tools[name] = str(Path(path).resolve())
    result = {"tools": resolved_tools}
    for name in DRIVERS:
        path = repo / "scripts" / name
        if not path.is_file() or path.is_symlink():
            raise PreflightError(f"required driver is missing or unsafe: {path}")
        if path.suffix == ".py":
            try:
                compile(path.read_bytes(), str(path), "exec")
            except SyntaxError as error:
                raise PreflightError(f"syntax check {name} failed: {error}") from error
        else:
            _command(["bash", "-n", str(path)], f"syntax check {name}")
        result[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    for name in ("human_test.py", "human_test_common.py", "accelerator_safe.py"):
        path = repo / "scripts" / name
        try:
            compile(path.read_bytes(), str(path), "exec")
        except SyntaxError as error:
            raise PreflightError(f"syntax check {name} failed: {error}") from error
        result[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return result


def validate_config_shape(config: dict[str, Any]) -> None:
    sections = config.get("sections")
    names = {"dashboard", "real_model", "migration", "batching", "target_comparator", "dflash", "gx"}
    if not isinstance(sections, dict) or set(sections) != names:
        raise PreflightError("sections must contain exactly: " + ", ".join(sorted(names)))
    if not all(isinstance(value, bool) for value in sections.values()):
        raise PreflightError("every section value must be boolean")
    if not isinstance(config.get("browser_hold"), bool):
        raise PreflightError("browser_hold must be boolean")
    output = config.get("output_root")
    if not isinstance(output, str) or not Path(output).is_absolute():
        raise PreflightError("output_root must be absolute")
    for name in REQUIRED_ARTIFACTS:
        _artifact(config, name)
    artifacts = config["artifacts"]
    allowed = set(REQUIRED_ARTIFACTS)
    unknown = sorted(set(artifacts) - allowed)
    if unknown:
        raise PreflightError(f"unknown artifacts: {', '.join(unknown)}")


def strict_json_file(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PreflightError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=unique)
    except (OSError, json.JSONDecodeError) as error:
        raise PreflightError(f"invalid JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        raise PreflightError(f"JSON file is not an object: {path}")
    return value


def rehash_artifacts(config: dict[str, Any], names: tuple[str, ...]) -> None:
    for name in names:
        _check_artifact(config, name)
    if "mtmd_bridge" in names or "mtmd_receipt" in names:
        _check_mtmd_package(config)


def _execution_contract(config: dict[str, Any], repo: Path) -> dict[str, Any]:
    """Render stable child boundaries; explicitly mark ephemeral values."""
    artifact = lambda name: str(artifact_path(config, name))
    scripts = repo / "scripts"
    ports = config["ports"]
    output = config["output_root"]
    server = lambda port, parallel, tls: [
        artifact("muser"), "serve", "--host", "127.0.0.1", "--port", str(port),
        "--api-key-file", artifact("api_key"), "--model", artifact("model"),
        "--backend", "metal", "--parallel", str(parallel), "--max-context", "4096",
        "--dflash", artifact("dflash"), "--dflash-backend", "metal",
        "--mmproj", artifact("mmproj"), "--mtmd-bridge", artifact("mtmd_bridge"),
        "--benchmark-shutdown-token", "<ephemeral-64-hex>",
        "--benchmark-deadline-seconds", "7200",
        *(
            ["--tls-cert", artifact(f"tls_{tls}_cert"), "--tls-key", artifact(f"tls_{tls}_key")]
            if tls else []
        ),
    ]
    contracts: dict[str, Any] = {
        "dashboard": {
            "argv": ["bash", str(scripts / "smoke_local_dashboard.sh")],
            "generated_private_inputs": ["api-key", "local CA key", "TLS key"],
            "retention": f"{output}/dashboard-tmp/muser-human-smoke.<mktemp>",
        },
        "real_model": {
            "server": server(ports["engine"], 4, None),
            "driver": [
                sys.executable, str(scripts / "smoke_real_model.py"),
                "--base-url", f"http://127.0.0.1:{ports['engine']}",
                "--api-key-file", artifact("api_key"),
                "--image", artifact("vision_image"),
                "--output", f"{output}/real-model.json",
            ],
        },
        "migration": {
            "source_server": server(ports["migration_source"], 1, "source"),
            "destination_server": server(ports["migration_destination"], 1, "destination"),
            "outbound_ca": artifact("tls_destination_ca"),
            "driver": [
                sys.executable, str(scripts / "smoke_decode_migration.py"),
                "--source-url", f"https://127.0.0.1:{ports['migration_source']}",
                "--destination-url", f"https://127.0.0.1:{ports['migration_destination']}",
                "--source-api-key-file", artifact("api_key"),
                "--destination-api-key-file", artifact("api_key"),
                "--source-ca-file", artifact("tls_source_ca"),
                "--destination-ca-file", artifact("tls_destination_ca"),
                "--output", f"{output}/migration.json",
            ],
        },
        "batching": {
            "driver": [
                sys.executable, str(scripts / "continuous_batching_smoke.py"),
                "--model", artifact("model"), "--muser-server", artifact("muser"),
                "--prompt-token-fixture", artifact("prompt_fixture"),
                "--output", f"{output}/batching.json", "--identity", config["run_id"],
                "--base-port", str(ports["batching_base"]),
            ],
        },
        "target_comparator": {
            "driver": [
                sys.executable, str(scripts / "representative_target_smoke.py"),
                "--model", artifact("model"),
                "--prompt-token-fixture", artifact("prompt_fixture"),
                "--muser-server", artifact("muser"),
                "--expected-muser-sha256", config["artifacts"]["muser"]["sha256"],
                "--muser-metallib", artifact("metallib"),
                "--llama-server", artifact("llama_server"),
                "--llama-receipt", artifact("llama_receipt"),
                "--output", f"{output}/target-comparator.json",
                "--identity", config["run_id"],
                "--muser-url", f"http://127.0.0.1:{ports['target_muser']}",
                "--llama-url", f"http://127.0.0.1:{ports['target_llama']}",
            ],
        },
        "dflash": {
            "muser_driver": [
                artifact("muser_dflash"), "--model", artifact("model"),
                "--dflash", artifact("dflash"),
                "--prompt-token-fixture", artifact("prompt_fixture"),
                "--repetitions", "1", "--output-tokens", "256",
                "--verify-length", "15", "--sampled-check-tokens", "32",
                "--target-backend", "metal",
                "--assistant-backend", "metal", "--identity", config["run_id"],
            ],
            "llama_driver": [
                sys.executable, str(scripts / "bench_llama_dflash.py"),
                "--server-binary", artifact("llama_server"),
                "--model", artifact("model"), "--dflash", artifact("dflash"),
                "--prompt-token-fixture", artifact("prompt_fixture"),
                "--depth", "2048", "--verify-length", "15",
                "--repetitions", "1", "--human-smoke",
                "--identity", config["run_id"],
                "--base-url", f"http://127.0.0.1:{ports['dflash']}",
            ],
        },
        "gx": {
            "mode": "read-only-status-only",
            "cluster_config": artifact("gx_cluster_config"),
            "container_receipt": artifact("gx_container_receipt"),
            "driver": [artifact("muser"), "node", "status", "--json"],
        },
    }
    return {
        "order": [name for name in ("dashboard", "real_model", "migration", "batching", "target_comparator", "dflash", "gx") if config["sections"][name]],
        "environment_allowlist": ["HOME", "LANG", "LC_ALL", "LOGNAME", "PATH", "TMPDIR", "USER"],
        "injected_environment": [
            "MUSER_ACCELERATOR_LEASE=1", "MUSER_GGML_METALLIB=<pinned>",
            "MUSER_HOME=<section-private>", "MUSER_DECODE_MIGRATION_CA=<pinned>",
        ],
        "sections": {name: contracts[name] for name in contracts if config["sections"][name]},
    }


def preflight(
    config: dict[str, Any],
    config_hash: str,
    repo: Path,
    *,
    accelerator_lock_held: bool = False,
) -> dict[str, Any]:
    validate_config_shape(config)
    output = Path(config["output_root"])
    if output.resolve(strict=False) != output:
        raise PreflightError("output_root traverses a symlink or is not normalized")
    if output.exists():
        raise PreflightError(f"output root already exists (no replacement): {output}")
    ancestor = output.parent
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    if not ancestor.is_dir() or ancestor.is_symlink() or not os.access(ancestor, os.W_OK):
        raise PreflightError(f"output ancestor is not a writable non-symlink directory: {ancestor}")
    host = _check_host(config, output)
    artifacts = {name: _check_artifact(config, name) for name in sorted(config["artifacts"])}
    for name, expected in PINNED_ARTIFACTS.items():
        if (artifacts[name]["bytes"], artifacts[name]["sha256"]) != expected:
            raise PreflightError(f"{name} does not match the immutable v0.1 artifact identity")
    if stat.S_IMODE(artifact_path(config, "api_key").stat().st_mode) != 0o600:
        raise PreflightError("API key must be mode 0600")
    for name in ("tls_source_key", "tls_destination_key"):
        if stat.S_IMODE(artifact_path(config, name).stat().st_mode) != 0o600:
            raise PreflightError(f"{name} must be mode 0600")
    binary = _check_binary(config)
    receipts = _check_receipts(config)
    mtmd_package = _check_mtmd_package(config)
    tls = _check_tls(config)
    inputs = _check_fixture_and_key(config)
    ports = _check_ports(config)
    if accelerator_lock_held:
        owners = accelerator_safe.active_gpu_processes()
        if owners:
            raise PreflightError(f"accelerator process is active: {', '.join(owners)}")
        accelerator = {
            "lease_path": str(accelerator_safe.LOCK_PATH),
            "active_owners": [],
        }
    else:
        accelerator = _check_accelerator_idle()
    drivers = _check_drivers(repo)
    execution = _execution_contract(config, repo)
    snapshot = {
        "config_sha256": config_hash, "host": host, "artifacts": artifacts,
        "binary": binary, "receipts": receipts, "ports": ports,
        "tls": tls, "inputs": inputs, "mtmd_package": mtmd_package,
        "accelerator": accelerator,
        "drivers": drivers, "execution": execution,
    }
    snapshot["snapshot_sha256"] = sha256_bytes(canonical_bytes(snapshot))
    return snapshot


def atomic_json(path: Path, value: Any) -> None:
    if path.exists():
        raise PreflightError(f"refusing to replace {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
