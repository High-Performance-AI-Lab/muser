#!/usr/bin/env python3
"""Verify a private Muser candidate without executing bundled code."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import struct
import tarfile


MUSER_VERSION = "0.1.0-beta.1"
MUSER_TAG = f"muser-v{MUSER_VERSION}"
KVPACK_VERSION = "0.1.0-alpha.2"
KVPACK_TAG = f"kvpack-v{KVPACK_VERSION}"
MANDATORY = frozenset({
    "correctness", "sampled", "greedy", "kvpack", "session", "vision",
    "baseline", "dflash", "remote", "serving", "onboarding",
    "api-parity", "continuous-batching", "migration", "security",
})
MODEL_SUFFIXES = {".gguf", ".ggml", ".safetensors", ".onnx", ".kvpack"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(value: str) -> PurePosixPath | None:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "." in path.parts:
        return None
    return path


def tree_receipt(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"unsafe package tree: {path}")
    for child in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
        if child.is_symlink() or not (child.is_file() or child.is_dir()):
            raise ValueError(f"unsafe package entry: {child}")
        if not child.is_file():
            continue
        relative = child.relative_to(path).as_posix().encode()
        size = child.stat().st_size
        total += size
        digest.update(struct.pack("<Q", len(relative)))
        digest.update(relative)
        digest.update(struct.pack("<Q", size))
        with child.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return total, digest.hexdigest()


def inspect_source_archive(path: Path, allowed_roots: set[str], failures: list[str]) -> set[str]:
    roots: set[str] = set()
    try:
        with tarfile.open(path, "r:gz") as archive:
            for member in archive:
                relative = safe_relative(member.name)
                if relative is None:
                    failures.append(f"unsafe source archive path: {member.name!r}")
                    continue
                roots.add(relative.parts[0])
                if relative.parts[0] not in allowed_roots:
                    failures.append(f"unexpected source archive root: {relative.parts[0]}")
                if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                    failures.append(f"unsafe source archive member type: {member.name}")
                if member.isfile() and PurePosixPath(member.name).suffix.lower() in MODEL_SUFFIXES:
                    failures.append(f"model/cache artifact in source archive: {member.name}")
                if relative.parts[0] == "kvpack-muser-alpha2" and len(relative.parts) >= 3:
                    if relative.parts[1] == "crates" and relative.parts[2] not in {
                        "kvpack-core", "kvpack", "kvpack-handoff"
                    }:
                        failures.append(f"unexpected kvpack crate in source archive: {member.name}")
    except (OSError, tarfile.TarError) as error:
        failures.append(f"cannot inspect source archive {path.name}: {error}")
    return roots


def load_json(path: Path, label: str, failures: list[str]) -> dict:
    try:
        if not path.is_file() or path.is_symlink():
            raise ValueError("file is missing or unsafe")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("top-level value is not an object")
        return value
    except (OSError, ValueError, TypeError) as error:
        failures.append(f"invalid {label}: {error}")
        return {}


def canonical_identity_digest(value: dict) -> str | None:
    payload = dict(value)
    claimed = payload.pop("digest", None)
    if not isinstance(claimed, str):
        return None
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    actual = hashlib.sha256(encoded).hexdigest()
    return claimed if claimed == actual else None


def command_sha256(command: list[str]) -> str:
    encoded = json.dumps(command, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def lane_execution_config_sha256(record: dict) -> str:
    runners = record.get(
        "readiness_runners", ["scripts/run_unsealed_release_matrix.py"]
    )
    encoded = json.dumps(
        {"argv": record.get("argv"), "readiness_runners": runners},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_atomic_seal_bundle(
    root: Path, identity: object, receipt: dict, failures: list[str]
) -> None:
    bundle = root / "evidence" / "atomic-seal-bundle"
    if not bundle.is_dir() or bundle.is_symlink():
        failures.append("atomic seal bundle is missing or unsafe")
        return

    result_path = bundle / "RESULT.json"
    manifest_path = bundle / "MANIFEST.json"
    readiness_path = bundle / "release-readiness.json"
    config_path = bundle / "matrix-config.json"
    result = load_json(result_path, "atomic seal result", failures)
    manifest = load_json(manifest_path, "atomic seal manifest", failures)
    readiness = load_json(readiness_path, "release-readiness receipt", failures)
    config = load_json(config_path, "release matrix configuration", failures)

    expected_members = manifest.get("members")
    actual_members = {
        path.relative_to(bundle).as_posix(): sha256(path)
        for path in sorted(bundle.rglob("*"))
        if path.is_file() and path.name != "MANIFEST.json"
    }
    if not isinstance(expected_members, dict) or expected_members != actual_members:
        failures.append("atomic seal bundle member hashes differ from its manifest")

    campaign = manifest.get("identity")
    if (
        manifest.get("schema") != "muser.atomic-seal-bundle.v1"
        or not isinstance(campaign, dict)
        or canonical_identity_digest(campaign) != identity
    ):
        failures.append("atomic seal manifest has an invalid campaign identity")

    lanes = result.get("lanes")
    if (
        result.get("schema") != "muser.final-seal-result.v1"
        or result.get("status") != "passed"
        or result.get("mode") != "seal"
        or result.get("published") is not True
        or result.get("fresh_re_evaluation") is not True
        or result.get("identity") != identity
        or not isinstance(lanes, dict)
        or set(lanes) != MANDATORY
    ):
        failures.append("atomic seal result is outside the final campaign contract")

    if (
        readiness.get("schema") != "muser.release-readiness.v1"
        or readiness.get("status") != "passed"
        or readiness.get("identity") != identity
        or not isinstance(readiness.get("lanes"), dict)
        or set(readiness["lanes"]) != MANDATORY
        or result.get("readiness_sha256") != sha256(readiness_path)
    ):
        failures.append("release-readiness receipt does not authorize the candidate")
    if (
        config.get("schema") != "muser.unsealed-matrix-config.v1"
        or not isinstance(config.get("lanes"), dict)
        or set(config["lanes"]) != MANDATORY
        or readiness.get("matrix_config_sha256") != sha256(config_path)
        or result.get("matrix_config_sha256") != sha256(config_path)
    ):
        failures.append("atomic bundle matrix configuration is invalid or mismatched")

    for lane in sorted(MANDATORY):
        report_path = bundle / "lanes" / f"{lane}.json"
        log_path = bundle / "logs" / f"{lane}.log"
        report = load_json(report_path, f"final {lane} lane report", failures)
        provenance = report.get("execution_provenance")
        command = provenance.get("command") if isinstance(provenance, dict) else None
        template = (
            config.get("lanes", {}).get(lane, {}).get("argv")
            if isinstance(config.get("lanes"), dict)
            and isinstance(config["lanes"].get(lane), dict)
            else None
        )
        if (
            report.get("schema") != "muser.unsealed-qualification.v1"
            or report.get("lane") != lane
            or report.get("status") != "passed"
            or report.get("seal_eligible") is not False
            or report.get("identity") != identity
            or not isinstance(lanes, dict)
            or lanes.get(lane) != (sha256(report_path) if report_path.is_file() else None)
        ):
            failures.append(f"invalid final unsealed qualification lane: {lane}")
        if not log_path.is_file() or log_path.is_symlink():
            failures.append(f"final qualification log is missing or unsafe: {lane}")
        if (
            not isinstance(provenance, dict)
            or provenance.get("schema")
            not in {
                "muser.lane-execution-provenance.v1",
                "muser.lane-execution-provenance.v2",
            }
            or provenance.get("matrix_config_sha256") != sha256(config_path)
            or provenance.get("command_template") != template
            or not isinstance(command, list)
            or not command
            or not all(isinstance(value, str) for value in command)
            or provenance.get("command_sha256")
            != (command_sha256(command) if isinstance(command, list) else None)
            or provenance.get("runner") != "scripts/atomic_seal_campaign.py"
            or provenance.get("log_name") != log_path.name
            or provenance.get("log_sha256")
            != (sha256(log_path) if log_path.is_file() else None)
            or re.fullmatch(
                r"[0-9a-f]{64}", str(provenance.get("evaluator_report_sha256", ""))
            )
            is None
            or (
                provenance.get("schema") == "muser.lane-execution-provenance.v2"
                and provenance.get("lane_execution_config_sha256")
                != lane_execution_config_sha256(config["lanes"][lane])
            )
        ):
            failures.append(f"invalid final lane execution provenance: {lane}")

    if (
        receipt.get("atomic_seal_identity") != identity
        or receipt.get("atomic_seal_result_sha256")
        != (sha256(result_path) if result_path.is_file() else None)
    ):
        failures.append("release receipt does not bind the exact atomic seal result")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    root = args.candidate.resolve()
    failures: list[str] = []
    if not root.is_dir() or root.is_symlink():
        failures.append("candidate root is missing or unsafe")
        files: list[Path] = []
    else:
        files = sorted(path for path in root.rglob("*") if path.is_file())
        for path in root.rglob("*"):
            if path.is_symlink() or not (path.is_file() or path.is_dir()):
                failures.append(f"unsafe candidate entry: {path.relative_to(root)}")

    sums_path = root / "SHA256SUMS"
    expected: dict[str, str] = {}
    if not sums_path.is_file() or sums_path.is_symlink():
        failures.append("SHA256SUMS is missing or unsafe")
    else:
        for line_number, line in enumerate(sums_path.read_text(encoding="utf-8").splitlines(), 1):
            match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
            if match is None or safe_relative(line[66:]) is None:
                failures.append(f"invalid SHA256SUMS line {line_number}")
                continue
            digest, relative = match.groups()
            if relative in expected:
                failures.append(f"duplicate SHA256SUMS path: {relative}")
            expected[relative] = digest
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in files
        if path != sums_path
    }
    if set(expected) != actual_paths:
        failures.append(
            "SHA256SUMS path set differs from candidate files "
            f"(missing={sorted(actual_paths - set(expected))}, extra={sorted(set(expected) - actual_paths)})"
        )
    for relative, digest in expected.items():
        path = root / relative
        if path.is_file() and sha256(path) != digest:
            failures.append(f"SHA-256 mismatch: {relative}")

    required = {
        "source/muser-with-kvpack.tar.gz",
        "source/gx10-producer.tar.gz",
        "bin/muser",
        "lib/mtmd/libmuser_mtmd_bridge.dylib",
        "evidence/campaign-identity.json",
        "evidence/gx10-container.json",
        "evidence/llama-comparator.json",
        "evidence/release-artifacts.json",
        "evidence/atomic-seal-bundle/RESULT.json",
        "evidence/atomic-seal-bundle/MANIFEST.json",
        "evidence/atomic-seal-bundle/release-readiness.json",
        "evidence/atomic-seal-bundle/matrix-config.json",
        "licenses/LICENSE-APACHE",
        "licenses/LICENSE-MIT",
        "licenses/NOTICE",
        "licenses/llama.cpp-LICENSE",
        "licenses/kvpack-LICENSE-APACHE",
        "licenses/kvpack-LICENSE-MIT",
        "licenses/kvpack-THIRD_PARTY_NOTICES.md",
        "licenses/rust-dependencies.json",
        "sbom.spdx.json",
        "demo/README.md",
        "demo/run.sh",
        "demo/request.sh",
        "demo/verify-models.py",
        "release-receipt.json",
    } | {
        f"evidence/atomic-seal-bundle/{kind}/{name}.{suffix}"
        for name in MANDATORY
        for kind, suffix in (("lanes", "json"), ("logs", "log"))
    }
    for relative in sorted(required):
        path = root / relative
        if not path.is_file() or path.is_symlink():
            failures.append(f"required candidate file is missing or unsafe: {relative}")
    for path in files:
        if path.suffix.lower() in MODEL_SUFFIXES:
            failures.append(f"model/cache artifact bundled: {path.relative_to(root)}")
    for forbidden in (root / "lib" / "ane", root / "evidence" / "coreml-compute-plan.json"):
        if forbidden.exists() or forbidden.is_symlink():
            failures.append(
                f"experimental ANE artifact is forbidden in v0.1 candidate: "
                f"{forbidden.relative_to(root)}"
            )

    receipt = {}
    try:
        receipt = json.loads((root / "release-receipt.json").read_text(encoding="utf-8"))
        if (
            receipt.get("schema") != "muser.release-candidate.receipt.v1"
            or receipt.get("status") != "private-candidate"
            or receipt.get("version") != MUSER_VERSION
            or receipt.get("distribution") != "private"
            or receipt.get("public_release_authorized") is not False
            or receipt.get("private_tags") != {"muser": MUSER_TAG, "kvpack": KVPACK_TAG}
            or receipt.get("model_weights_bundled") is not False
            or receipt.get("kvpack_release_crates")
            != ["kvpack", "kvpack-core", "kvpack-handoff"]
            or "not affiliated" not in receipt.get("affiliation", "")
        ):
            failures.append("release receipt is outside the private candidate contract")
    except (OSError, ValueError, TypeError) as error:
        failures.append(f"invalid release receipt: {error}")

    identity = receipt.get("identity")
    validate_atomic_seal_bundle(root, identity, receipt, failures)

    try:
        sbom = json.loads((root / "sbom.spdx.json").read_text())
        package_versions = {(p.get("name"), p.get("versionInfo")) for p in sbom.get("packages", [])}
        if (
            sbom.get("spdxVersion") != "SPDX-2.3"
            or not sbom.get("relationships")
            or not all((name, MUSER_VERSION) in package_versions for name in (
                "muser-engine", "muser-server", "muser-kvpack", "muser-cluster", "muser-bench"
            ))
            or not all((name, KVPACK_VERSION) in package_versions for name in (
                "kvpack", "kvpack-core", "kvpack-handoff"
            ))
            or ("llama.cpp", receipt.get("llama_source_commit")) not in package_versions
        ):
            failures.append("SPDX SBOM omits a required release component or relationship")
    except (OSError, ValueError, TypeError) as error:
        failures.append(f"invalid SPDX SBOM: {error}")

    muser_roots = inspect_source_archive(
        root / "source" / "muser-with-kvpack.tar.gz",
        {"muser", "kvpack-muser-alpha2"},
        failures,
    )
    if muser_roots != {"muser", "kvpack-muser-alpha2"}:
        failures.append(f"combined source archive roots are incomplete: {sorted(muser_roots)}")
    gx10_roots = inspect_source_archive(
        root / "source" / "gx10-producer.tar.gz", {"muser"}, failures
    )
    if gx10_roots != {"muser"}:
        failures.append(f"GX10 source archive roots are incomplete: {sorted(gx10_roots)}")

    for relative in ("demo/run.sh", "demo/request.sh", "demo/verify-models.py", "bin/muser"):
        path = root / relative
        if path.is_file() and not os.access(path, os.X_OK):
            failures.append(f"candidate executable bit is missing: {relative}")
    report = {
        "schema": "muser.release-candidate.verification.v1",
        "status": "failed" if failures else "passed",
        "candidate": str(root),
        "files_verified": len(actual_paths),
        "identity": identity,
        "failures": failures,
        "executes_bundled_code": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
