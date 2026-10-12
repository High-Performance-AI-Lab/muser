#!/usr/bin/env python3
"""Fail closed on release hygiene that can be checked without hardware."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MAX_TRACKED_BYTES = 5 * 1024 * 1024
REQUIRED = [
    "LICENSE-APACHE",
    "LICENSE-MIT",
    "NOTICE",
    "licenses/llama.cpp-LICENSE",
    "third_party/llama.cpp.LICENSE",
    "docs/release-artifacts.json",
    "docs/release-model-metadata.json",
    "scripts/build_release_candidate.py",
    "scripts/build_feature_freeze.py",
    "scripts/verify_release_candidate.py",
    "scripts/verify_release_candidate_cleanroom.py",
    "scripts/release_demo.py",
    "release/feature-contract-v1.json",
    "release/findings-v1.json",
    "release/llama-server-compat-v1.json",
    "release/release-lock.json",
    "release/dflash-route-policy-v1.json",
    "release/dflash-tuning-v1.json",
    "scripts/release_identity.py",
    "scripts/release_readiness.py",
    "scripts/atomic_seal_campaign.py",
    "scripts/release_host_preflight.py",
    "scripts/run_unsealed_release_matrix.py",
    "scripts/freeze_dflash_tuning.py",
    "third_party/kvpack/provenance.json",
    "third_party/metal/provenance.json",
]

CONTAINMENT_MARKER_TAG_POLICY = {
    "class": "non-release-marker",
    "allowed_tags": ["v0.1.0-beta.1"],
    "operator_go_required": True,
    "creates_seal": False,
    "creates_candidate": False,
    "creates_publication": False,
}


def containment_lock_is_safe(lock: dict) -> bool:
    if lock.get("state") != "containment":
        return True
    return (
        lock.get("sealing_enabled") is False
        and lock.get("candidate_creation_enabled") is False
        and lock.get("tagging_enabled") is True
        and lock.get("tagging_policy") == CONTAINMENT_MARKER_TAG_POLICY
        and lock.get("publishing_enabled") is False
    )


def git_files() -> list[Path]:
    output = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    return [ROOT / raw.decode() for raw in output.split(b"\0") if raw]


def main() -> int:
    failures: list[str] = []
    files = git_files()
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            failures.append(f"missing required release file: {relative}")

    for path in files:
        relative = path.relative_to(ROOT)
        if path.is_file() and path.stat().st_size > MAX_TRACKED_BYTES:
            failures.append(f"tracked file exceeds 5 MiB without an allowlist: {relative}")
        if path.suffix.lower() in {".gguf", ".ggml", ".safetensors", ".onnx", ".kvpack"}:
            failures.append(f"model/cache artifact would be tracked: {relative}")
        if path.is_file() and path.stat().st_size <= MAX_TRACKED_BYTES:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            personal_path = re.compile("/" + "Users" + r"/[^/\s]+/")
            if personal_path.search(text):
                failures.append(f"personal absolute path in {relative}")

    metadata = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    packages = json.loads(metadata)["packages"]
    ferrite_packages = sorted(p["name"] for p in packages if p["name"].startswith("ferrite-"))
    if ferrite_packages:
        failures.append("Ferrite runtime packages present: " + ", ".join(ferrite_packages))
    workspace_packages = [package for package in packages if package["name"].startswith("muser-")]
    if not workspace_packages or {
        package["version"] for package in workspace_packages
    } != {"0.1.0-beta.1"}:
        failures.append("Muser workspace packages are not all version 0.1.0-beta.1")
    if any(package.get("license") != "Apache-2.0 OR MIT" for package in workspace_packages):
        failures.append("Muser workspace package license metadata is inconsistent")
    root = ROOT.resolve()
    for package in packages:
        if package.get("source") is not None:
            continue
        manifest = Path(package["manifest_path"]).resolve()
        try:
            manifest.relative_to(root)
        except ValueError:
            failures.append(
                f"dependency package escapes the Muser tree: {package['name']} ({manifest})"
            )

    metal_root = ROOT / "third_party" / "metal"
    metal_provenance_path = metal_root / "provenance.json"
    try:
        metal_provenance = json.loads(metal_provenance_path.read_text(encoding="utf-8"))
        metal_manifest = (metal_root / "Cargo.toml").read_text(encoding="utf-8")
        source_hash = hashlib.sha256()
        for source in sorted((metal_root / "src").rglob("*")):
            if source.is_file():
                source_hash.update(source.relative_to(metal_root).as_posix().encode())
                source_hash.update(b"\0")
                source_hash.update(hashlib.sha256(source.read_bytes()).digest())
    except (OSError, json.JSONDecodeError) as error:
        failures.append(f"vendored metal provenance is invalid: {error}")
    else:
        if (
            metal_provenance.get("schema") != "muser.vendored-crate-provenance.v1"
            or metal_provenance.get("crate") != "metal"
            or metal_provenance.get("version") != "0.33.0"
            or metal_provenance.get("crates_io_archive_sha256")
            != "c7047791b5bc903b8cd963014b355f71dc9864a9a0b727057676c1dcae5cbc15"
            or metal_provenance.get("source_tree_sha256") != source_hash.hexdigest()
        ):
            failures.append("vendored metal identity or source tree does not match provenance")
        if 'package = "pastey"' not in metal_manifest or 'version = "0.2.3"' not in metal_manifest:
            failures.append("vendored metal does not carry the declared pastey patch")

    lock_path = ROOT / "release" / "release-lock.json"
    if lock_path.is_file():
        try:
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            failures.append(f"release lock is invalid: {error}")
        else:
            if lock.get("schema") != "muser.release-lock.v1":
                failures.append("release lock schema is invalid")
            if not containment_lock_is_safe(lock):
                failures.append(
                    "containment lock permits more than the exact operator-gated beta marker"
                )

    findings_path = ROOT / "release" / "findings-v1.json"
    if findings_path.is_file():
        try:
            findings = json.loads(findings_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            failures.append(f"findings register is invalid: {error}")
        else:
            records = findings.get("findings")
            if findings.get("schema") != "muser.findings.v1" or not isinstance(records, list):
                failures.append("findings register schema is invalid")
            else:
                identifiers = [record.get("id") for record in records]
                if len(identifiers) != len(set(identifiers)):
                    failures.append("findings register has duplicate ids")
                if any(record.get("status") not in {"open", "closed"} for record in records):
                    failures.append("findings register has an invalid status")
                if any("waiver" in record for record in records):
                    failures.append("findings register contains a waiver")

    lock_import = "from release_lock import"
    for relative in (
        "scripts/run_seal_chain.py",
        "scripts/campaign.py",
        "scripts/build_release_candidate.py",
        "scripts/evaluate_ane.py",
        "scripts/evaluate_baseline.py",
        "scripts/evaluate_dflash.py",
        "scripts/evaluate_kvpack.py",
        "scripts/evaluate_remote.py",
        "scripts/evaluate_vision.py",
    ):
        path = ROOT / relative
        if path.is_file() and lock_import not in path.read_text(encoding="utf-8"):
            failures.append(f"seal/candidate entry point does not import release lock: {relative}")

    notice = (ROOT / "NOTICE").read_text(encoding="utf-8") if (ROOT / "NOTICE").is_file() else ""
    for required_notice in ("llama.cpp", "kvpack", "not affiliated", "Meta"):
        if required_notice not in notice:
            failures.append(f"NOTICE omits required attribution/policy text: {required_notice}")

    bundled_llama_license = ROOT / "licenses" / "llama.cpp-LICENSE"
    source_llama_license = ROOT / "third_party" / "llama.cpp.LICENSE"
    if (
        bundled_llama_license.is_symlink()
        or source_llama_license.is_symlink()
        or not bundled_llama_license.is_file()
        or not source_llama_license.is_file()
        or bundled_llama_license.read_bytes() != source_llama_license.read_bytes()
    ):
        failures.append("bundled llama.cpp license is missing, unsafe, or not exact")

    ignored = subprocess.run(
        ["git", "check-ignore", "results/probe", "scratchpad/probe", "logs/probe"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
    )
    if ignored.returncode != 0:
        failures.append("results/, scratchpad/, and logs/ must all be ignored")

    report = {"status": "failed" if failures else "passed", "files_checked": len(files), "failures": failures}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
