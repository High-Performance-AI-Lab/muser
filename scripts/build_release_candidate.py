#!/usr/bin/env python3
"""Build an append-only candidate from one independently verified atomic seal bundle."""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile

from build_gx10_container import (
    CUDA_IMAGE as GX10_CUDA_IMAGE,
    FILES as GX10_BUILD_INPUTS,
    adapter_digest as gx10_adapter_digest,
)
from release_lock import ReleaseLocked, require_candidate_enabled


ROOT = Path(__file__).resolve().parents[1]
KVPACK_ROOT = ROOT / "third_party" / "kvpack"
ARTIFACT_MANIFEST = ROOT / "docs" / "release-artifacts.json"
LLAMA_LICENSE = ROOT / "licenses" / "llama.cpp-LICENSE"
MUSER_VERSION = "0.1.0-beta.1"
MUSER_PRIVATE_TAG = f"muser-v{MUSER_VERSION}"
KVPACK_VERSION = "0.1.0-alpha.2"
KVPACK_PRIVATE_TAG = f"kvpack-v{KVPACK_VERSION}"
KVPACK_RELEASE_CRATES = frozenset({"kvpack-core", "kvpack", "kvpack-handoff"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--identity-receipt", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--muser-binary", type=Path, required=True)
    parser.add_argument("--mtmd-package", type=Path, required=True)
    parser.add_argument("--gx10-container-receipt", type=Path, required=True)
    parser.add_argument("--llama-comparator-receipt", type=Path, required=True)
    parser.add_argument("--seal-bundle", type=Path, required=True)
    parser.add_argument(
        "--seal", action="append", default=[], metavar="NAME=PATH",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--version", default=MUSER_VERSION)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_files(path: Path) -> list[tuple[str, Path]]:
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError(f"release tree is missing or unsafe: {path}")
    result: list[tuple[str, Path]] = []
    for child in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
        if child.is_symlink():
            raise RuntimeError(f"release tree contains a symlink: {child}")
        if child.is_file():
            result.append((child.relative_to(path).as_posix(), child))
        elif not child.is_dir():
            raise RuntimeError(f"release tree contains a special entry: {child}")
    return result


def tree_receipt(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    for relative, source in tree_files(path):
        encoded = relative.encode("utf-8")
        size = source.stat().st_size
        total += size
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
        digest.update(struct.pack("<Q", size))
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return total, digest.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, text=True, stdout=subprocess.PIPE
    ).stdout.strip()


def require_clean(root: Path, label: str) -> str:
    if not (root / ".git").exists():
        raise RuntimeError(f"{label} is not a git worktree: {root}")
    status = git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError(f"{label} worktree is dirty")
    return git(root, "rev-parse", "HEAD")


def validate_atomic_seal_bundle(path: Path, identity: str) -> tuple[Path, dict]:
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError("atomic seal bundle is missing or unsafe")
    root = path.resolve()
    result_path = root / "RESULT.json"
    manifest_path = root / "MANIFEST.json"
    if not result_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("atomic seal bundle omits RESULT.json or MANIFEST.json")
    result = json.loads(result_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if (
        result.get("schema") != "muser.final-seal-result.v1"
        or result.get("status") != "passed"
        or result.get("published") is not True
        or result.get("identity") != identity
        or manifest.get("schema") != "muser.atomic-seal-bundle.v1"
        or manifest.get("identity", {}).get("digest") != identity
    ):
        raise RuntimeError("atomic seal bundle does not authorize this identity")
    expected = manifest.get("members")
    if not isinstance(expected, dict):
        raise RuntimeError("atomic seal manifest has no member map")
    actual = {
        item.relative_to(root).as_posix(): sha256(item)
        for item in sorted(root.rglob("*"))
        if item.is_file() and item.name != "MANIFEST.json"
    }
    if actual != expected:
        raise RuntimeError("atomic seal bundle member hashes differ from its manifest")
    return root, result


def validate_json_receipt(path: Path, schema: str, identity: str) -> dict:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"missing or unsafe receipt: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != schema:
        raise RuntimeError(f"{path.name} has schema {value.get('schema')!r}, expected {schema}")
    if value.get("identity") != identity:
        raise RuntimeError(f"{path.name} belongs to a different qualification identity")
    if value.get("status") != "passed" or value.get("seal_eligible") is not True:
        raise RuntimeError(f"{path.name} is not a passing seal")
    return value


def validate_campaign_identity(
    path: Path | None, identity: str, muser_commit: str, binary: Path
) -> dict:
    if path is None or not path.is_file() or path.is_symlink():
        raise RuntimeError("a regular --identity-receipt from the qualification run is required")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "muser.campaign-identity.v3":
        raise RuntimeError("qualification identity receipt is not v3")
    encoded_value = dict(value)
    claimed_digest = encoded_value.pop("digest", None)
    encoded = json.dumps(encoded_value, sort_keys=True, separators=(",", ":")).encode()
    actual_digest = hashlib.sha256(encoded).hexdigest()
    if claimed_digest != identity or actual_digest != identity:
        raise RuntimeError("qualification identity receipt digest does not match --identity")
    if value.get("source", {}).get("commit") != muser_commit:
        raise RuntimeError("release source commit differs from the qualified source commit")
    expected = value.get("binaries", {}).get("muser_server", {})
    if (
        expected.get("bytes") != binary.stat().st_size
        or expected.get("sha256") != sha256(binary)
    ):
        raise RuntimeError("release binary differs from the qualified Muser server")
    pinned = value.get("files", {}).get("docs/release-artifacts.json", {})
    if pinned.get("sha256") != sha256(ARTIFACT_MANIFEST):
        raise RuntimeError("qualification used a different official artifact manifest")
    return value


def validate_gx10_receipt(path: Path, llama_commit: str) -> dict:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("GX10 container receipt is missing or unsafe")
    value = json.loads(path.read_text(encoding="utf-8"))
    image_id = value.get("image_id")
    if (
        value.get("schema") != "muser.gx10-container.receipt.v1"
        or value.get("status") != "built"
        or value.get("executed") is not False
        or value.get("architecture") != "arm64"
        or value.get("source_commit") != llama_commit
        or value.get("cuda_image") != GX10_CUDA_IMAGE
        or value.get("cuda_matmul") not in {"default", "force-cublas", "force-mmq"}
        or not isinstance(image_id, str)
        or not image_id.startswith("sha256:")
        or not isinstance(value.get("image_bytes"), int)
        or value["image_bytes"] <= 0
    ):
        raise RuntimeError("GX10 container receipt is outside the release contract")
    expected_inputs = {
        name: sha256(ROOT / "scripts" / "gx10" / "llamacpp" / name)
        for name in GX10_BUILD_INPUTS
    }
    if (
        value.get("adapter_sha256") != gx10_adapter_digest(expected_inputs)
        or value.get("build_inputs") != expected_inputs
    ):
        raise RuntimeError("GX10 container differs from the bundled adapter source")
    return value


def validate_artifact_manifest(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"missing or unsafe release artifact manifest: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema") != "muser.release-artifacts.v2"
        or value.get("repository") != "meta-models/Muse-Glimmer-30B-GGUF"
        or not isinstance(value.get("revision"), str)
        or len(value["revision"]) != 40
        or set(value.get("artifacts", {})) != {"target", "vision", "dflash"}
    ):
        raise RuntimeError("release artifact manifest has an invalid identity contract")
    for name, artifact in value["artifacts"].items():
        digest = artifact.get("sha256") if isinstance(artifact, dict) else None
        filename = artifact.get("filename") if isinstance(artifact, dict) else None
        revision = artifact.get("revision") if isinstance(artifact, dict) else None
        url = artifact.get("url") if isinstance(artifact, dict) else None
        size = artifact.get("bytes") if isinstance(artifact, dict) else None
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or revision != value["revision"]
            or not isinstance(url, str)
            or f"/resolve/{revision}/{filename}?download=true" not in url
            or not isinstance(size, int)
            or size <= 0
        ):
            raise RuntimeError(f"release artifact manifest entry {name!r} is invalid")
    return value


def validate_mtmd_package(path: Path) -> dict:
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError(f"missing or unsafe mtmd package: {path}")
    for child in path.rglob("*"):
        if child.is_symlink() or not (child.is_file() or child.is_dir()):
            raise RuntimeError(f"mtmd package contains an unsafe entry: {child}")
    receipt_path = path / "receipt.json"
    if not receipt_path.is_file():
        raise RuntimeError("mtmd package receipt is missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema") != "muser.mtmd_bridge.receipt.v2"
        or receipt.get("status") != "built"
        or receipt.get("bridge_abi") != "muser-mtmd-muse-vision-v1"
        or not isinstance(receipt.get("artifacts"), dict)
    ):
        raise RuntimeError("mtmd package receipt is outside the release contract")
    artifacts = receipt["artifacts"]
    if "libmuser_mtmd_bridge.dylib" not in artifacts:
        raise RuntimeError("mtmd package omits the Muser bridge")
    for name, expected in artifacts.items():
        artifact = path / name
        if (
            Path(name).name != name
            or not artifact.is_file()
            or artifact.is_symlink()
            or artifact.stat().st_size != expected.get("bytes")
            or sha256(artifact) != expected.get("sha256")
        ):
            raise RuntimeError(f"mtmd package artifact differs from receipt: {name}")
    actual = {child.name for child in path.iterdir() if child.suffix == ".dylib"}
    if actual != set(artifacts):
        raise RuntimeError("mtmd package contains unreceipted dylibs")
    return receipt


def tracked(root: Path) -> list[Path]:
    raw = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=True, stdout=subprocess.PIPE
    ).stdout
    return [Path(value.decode()) for value in raw.split(b"\0") if value]


def kvpack_release_paths(root: Path) -> list[Path]:
    """Retain repository support files but expose only the alpha.2 crate set."""
    provenance_path = root / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (
        provenance.get("schema") != "muser.vendored-source.v1"
        or provenance.get("upstream_commit")
        != "70c34c7d790dbfc9c1271727dd34ea0e863404d2"
        or provenance.get("patches") != []
        or not isinstance(provenance.get("files"), dict)
    ):
        raise RuntimeError("vendored kvpack provenance is outside the release contract")
    paths: list[Path] = []
    for raw, expected in provenance["files"].items():
        path = Path(raw)
        if len(path.parts) >= 2 and path.parts[0] == "crates":
            if path.parts[1] not in KVPACK_RELEASE_CRATES:
                continue
        source = root / path
        if (
            not source.is_file()
            or source.is_symlink()
            or sha256(source) != expected
        ):
            raise RuntimeError(f"vendored kvpack file differs from provenance: {path}")
        paths.append(path)
    paths.append(Path("provenance.json"))
    present = {
        path.parts[1]
        for path in paths
        if len(path.parts) >= 2 and path.parts[0] == "crates"
    }
    if present != KVPACK_RELEASE_CRATES:
        raise RuntimeError(
            f"kvpack release crate set is {sorted(present)}, "
            f"expected {sorted(KVPACK_RELEASE_CRATES)}"
        )
    return paths


def deterministic_tar(
    output: Path, roots: list[tuple[str, Path, list[Path]]], epoch: int
) -> None:
    with output.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=epoch) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for prefix, root, paths in roots:
                    for relative in sorted(paths):
                        source = root / relative
                        if not source.is_file() or source.is_symlink():
                            raise RuntimeError(f"release source entry is not a regular file: {source}")
                        info = tarfile.TarInfo(f"{prefix}/{relative.as_posix()}")
                        info.size = source.stat().st_size
                        info.mode = 0o755 if os.access(source, os.X_OK) else 0o644
                        info.mtime = epoch
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        with source.open("rb") as stream:
                            archive.addfile(info, stream)


def cargo_metadata() -> dict:
    return json.loads(
        subprocess.run(
            ["cargo", "metadata", "--format-version", "1", "--locked"],
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout
    )


def spdx_id(package_id: str) -> str:
    digest = hashlib.sha256(package_id.encode()).hexdigest()[:20]
    return f"SPDXRef-Package-{digest}"


def write_spdx(
    path: Path,
    metadata: dict,
    identity: str,
    commit: str,
    created: str,
    llama_commit: str,
) -> None:
    packages = []
    package_ids: dict[str, str] = {}
    for package in sorted(metadata["packages"], key=lambda p: p["id"]):
        identifier = spdx_id(package["id"])
        package_ids[package["id"]] = identifier
        source = package.get("source")
        if isinstance(source, str) and source.startswith("registry+"):
            download = f"https://crates.io/crates/{package['name']}/{package['version']}"
        else:
            download = "NOASSERTION"
        packages.append(
            {
                "SPDXID": identifier,
                "name": package["name"],
                "versionInfo": package["version"],
                "downloadLocation": download,
                "filesAnalyzed": False,
                "licenseDeclared": package.get("license") or "NOASSERTION",
                "licenseConcluded": "NOASSERTION",
                "supplier": "NOASSERTION",
            }
        )
    llama_id = "SPDXRef-Package-llama.cpp"
    packages.append(
        {
            "SPDXID": llama_id,
            "name": "llama.cpp",
            "versionInfo": llama_commit,
            "downloadLocation": "https://github.com/ggml-org/llama.cpp",
            "filesAnalyzed": False,
            "licenseDeclared": "MIT",
            "licenseConcluded": "NOASSERTION",
            "supplier": "Organization: The ggml authors",
        }
    )
    relationships = []
    for node in metadata.get("resolve", {}).get("nodes", []):
        source = package_ids.get(node.get("id"))
        if source is None:
            continue
        for dependency in sorted(node.get("deps", []), key=lambda value: value.get("pkg", "")):
            target = package_ids.get(dependency.get("pkg"))
            if target is not None:
                relationships.append(
                    {
                        "spdxElementId": source,
                        "relationshipType": "DEPENDS_ON",
                        "relatedSpdxElement": target,
                    }
                )
    workspace_ids = [
        package_ids[package_id]
        for package_id in metadata.get("workspace_members", [])
        if package_id in package_ids
    ]
    relationships.extend(
        {
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relationshipType": "DESCRIBES",
            "relatedSpdxElement": identifier,
        }
        for identifier in [*workspace_ids, llama_id]
    )
    namespace_digest = hashlib.sha256(f"{identity}\0{commit}".encode()).hexdigest()
    document = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "muser-release-sbom",
        "documentNamespace": f"https://muser.invalid/spdx/{namespace_digest}",
        "creationInfo": {"created": created, "creators": ["Tool: muser-release-builder-v1"]},
        "packages": packages,
        "relationships": relationships,
    }
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


def collect_cargo_licenses(destination: Path, metadata: dict) -> dict:
    destination.mkdir()
    inventory = []
    for package in sorted(metadata["packages"], key=lambda value: value["id"]):
        record = {
            "name": package["name"],
            "version": package["version"],
            "license": package.get("license") or "NOASSERTION",
            "source": package.get("source") or "path",
            "license_files": [],
        }
        if package.get("source"):
            package_root = Path(package["manifest_path"]).parent
            if not package_root.is_dir() or package_root.is_symlink():
                raise RuntimeError(f"Cargo source package is missing or unsafe: {package_root}")
            identifier = spdx_id(package["id"]).removeprefix("SPDXRef-Package-")
            for candidate in sorted(package_root.iterdir(), key=lambda value: value.name.lower()):
                lower = candidate.name.lower()
                is_notice = (
                    lower.startswith(("license", "copying", "notice"))
                    or lower in {"copyright", "unlicense"}
                )
                if not is_notice:
                    continue
                if not candidate.is_file() or candidate.is_symlink():
                    raise RuntimeError(f"Cargo license entry is unsafe: {candidate}")
                if candidate.stat().st_size > 2 * 1024 * 1024:
                    raise RuntimeError(f"Cargo license entry is unexpectedly large: {candidate}")
                safe_name = re.sub(r"[^A-Za-z0-9_.+-]", "_", candidate.name)
                output_name = f"{package['name']}-{package['version']}-{identifier}-{safe_name}"
                shutil.copyfile(candidate, destination / output_name)
                record["license_files"].append(f"rust/{output_name}")
        inventory.append(record)
    return {
        "schema": "muser.rust-dependency-notices.v1",
        "packages": inventory,
    }


def demo_text() -> str:
    return """#!/bin/sh
set -eu
ROOT=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
: "${MUSER_MODEL:?set MUSER_MODEL to the pinned target GGUF}"
: "${MUSER_MMPROJ:?set MUSER_MMPROJ to the pinned mmproj GGUF}"
: "${MUSER_DFLASH:?set MUSER_DFLASH to the pinned DFlash GGUF}"
python3 "$ROOT/demo/verify-models.py" \\
  --manifest "$ROOT/evidence/release-artifacts.json" \\
  --target "$MUSER_MODEL" --vision "$MUSER_MMPROJ" --dflash "$MUSER_DFLASH"
exec "$ROOT/bin/muser" serve \\
  --model "$MUSER_MODEL" --mmproj "$MUSER_MMPROJ" \\
  --mtmd-bridge "$ROOT/lib/mtmd/libmuser_mtmd_bridge.dylib" \\
  --dflash "$MUSER_DFLASH" \\
  --backend metal --dflash-backend auto --prefill local --prefix-cache on
"""


def demo_request_text() -> str:
    return """#!/bin/sh
set -eu
MUSER_URL=${MUSER_URL:-http://127.0.0.1:4949}
curl --fail --silent --show-error "$MUSER_URL/healthz"
printf '\n'
curl --fail --silent --show-error --no-buffer \\
  -H 'content-type: application/json' \\
  --data '{"model":"muse-glimmer-30b","messages":[{"role":"user","content":"Explain why exact KV restoration matters in one sentence."}],"max_tokens":64,"stream":true}' \\
  "$MUSER_URL/v1/chat/completions"
"""


def demo_readme_text() -> str:
    return """# Private candidate demo

This candidate does not include model weights. Set `MUSER_MODEL`,
`MUSER_MMPROJ`, and `MUSER_DFLASH` to the three files named in
`../evidence/release-artifacts.json`, then run `./run.sh`. The launcher hashes
all three files before starting the loopback-only server. DFlash auto routing
is frozen to Metal for v0.1; experimental ANE artifacts are not bundled.

In another terminal, run `./request.sh` for a streaming OpenAI-compatible text
request. The live dashboard and measured health state are at
`http://127.0.0.1:4949/`. No performance number is embedded here; release
claims live only in the bundled seal receipts.

Muser is independent and is not affiliated with, sponsored by, or endorsed by
Meta or the Muse model authors.
"""


def main() -> int:
    args = parse_args()
    if not args.dry_run:
        try:
            require_candidate_enabled("release candidate build")
        except ReleaseLocked as error:
            raise SystemExit(str(error)) from None
    plan = {
        "schema": "muser.release-candidate.plan.v1",
        "mode": "dry-run" if args.dry_run else "build",
        "identity": args.identity,
        "identity_receipt": str(args.identity_receipt) if args.identity_receipt else None,
        "version": args.version,
        "private_tags": {
            "muser": MUSER_PRIVATE_TAG,
            "kvpack": KVPACK_PRIVATE_TAG,
        },
        "output_dir": str(args.output_dir),
        "atomic_seal_bundle": str(args.seal_bundle),
        "outputs": [
            "source/muser-with-kvpack.tar.gz",
            "source/gx10-producer.tar.gz",
            "bin/muser",
            "lib/mtmd/*.dylib",
            "evidence/*.json",
            "licenses/*",
            "sbom.spdx.json",
            "demo/{README.md,run.sh,request.sh,verify-models.py}",
            "SHA256SUMS",
            "release-receipt.json",
        ],
        "models_bundled": False,
        "artifact_manifest": "evidence/release-artifacts.json",
        "kvpack_release_crates": sorted(KVPACK_RELEASE_CRATES),
        "publishes_or_tags": False,
    }
    if args.dry_run:
        if args.version != MUSER_VERSION:
            raise RuntimeError(f"private candidate version must be {MUSER_VERSION}")
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if args.version != MUSER_VERSION:
        raise RuntimeError(f"private candidate version must be {MUSER_VERSION}")
    if args.seal:
        raise RuntimeError("individual --seal inputs are retired; use --seal-bundle only")
    seal_bundle, seal_result = validate_atomic_seal_bundle(args.seal_bundle, args.identity)
    muser_commit = require_clean(ROOT, "Muser")
    kvpack_provenance = json.loads((KVPACK_ROOT / "provenance.json").read_text())
    kvpack_commit = kvpack_provenance.get("upstream_commit")
    if kvpack_commit != "70c34c7d790dbfc9c1271727dd34ea0e863404d2":
        raise RuntimeError("vendored kvpack commit differs from the release pin")
    artifact_manifest = validate_artifact_manifest(ARTIFACT_MANIFEST)
    mtmd_receipt = validate_mtmd_package(args.mtmd_package.resolve())
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise RuntimeError(f"refusing to replace release output: {args.output_dir}")
    binary = args.muser_binary.resolve()
    if not binary.is_file() or binary.is_symlink():
        raise RuntimeError("Muser binary is missing or unsafe")
    file_kind = subprocess.run(
        ["file", "-b", str(binary)], check=True, text=True, stdout=subprocess.PIPE
    ).stdout
    if "Mach-O" not in file_kind or "arm64" not in file_kind:
        raise RuntimeError(f"Muser binary is not Mach-O arm64: {file_kind.strip()}")
    campaign_identity = validate_campaign_identity(
        args.identity_receipt, args.identity, muser_commit, binary
    )
    llama_receipt = json.loads(args.llama_comparator_receipt.read_text())
    if (
        llama_receipt.get("schema") != "muser.llama_comparator.source_receipt.v3"
        or llama_receipt.get("executed") is not False
        or llama_receipt.get("build", {}).get("metal") is not True
        or not isinstance(llama_receipt.get("source_commit"), str)
        or len(llama_receipt["source_commit"]) != 40
    ):
        raise RuntimeError("llama comparator is not an unexecuted Metal v3 build")
    if mtmd_receipt.get("llama_commit") != llama_receipt["source_commit"]:
        raise RuntimeError("mtmd and llama comparator use different upstream commits")
    comparator_identity = campaign_identity.get("artifacts", {}).get(
        "llama_comparator_receipt", {}
    )
    if (
        comparator_identity.get("sha256") != sha256(args.llama_comparator_receipt)
        or comparator_identity.get("source_commit") != llama_receipt["source_commit"]
    ):
        raise RuntimeError("release comparator differs from the qualified comparator")
    gx10_receipt = validate_gx10_receipt(
        args.gx10_container_receipt.resolve(), llama_receipt["source_commit"]
    )

    metadata = cargo_metadata()
    workspace_versions = {
        package["version"]
        for package in metadata["packages"]
        if package["id"] in metadata["workspace_members"]
    }
    if workspace_versions != {MUSER_VERSION}:
        raise RuntimeError(
            f"Muser workspace versions are {sorted(workspace_versions)}, expected {MUSER_VERSION}"
        )

    commit_epoch = int(git(ROOT, "show", "-s", "--format=%ct", muser_commit))
    created = dt.datetime.fromtimestamp(commit_epoch, dt.timezone.utc).isoformat()
    output_parent = args.output_dir.parent.resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".muser-release-", dir=output_parent))
    try:
        for directory in ("source", "bin", "lib", "evidence", "licenses", "demo"):
            (stage / directory).mkdir()
        deterministic_tar(
            stage / "source" / "muser-with-kvpack.tar.gz",
            [
                ("muser", ROOT, tracked(ROOT)),
                (
                    "kvpack-muser-alpha2",
                    KVPACK_ROOT,
                    kvpack_release_paths(KVPACK_ROOT),
                ),
            ],
            commit_epoch,
        )
        gx10_paths = [
            path for path in tracked(ROOT)
            if path.is_relative_to(Path("scripts/gx10"))
        ]
        deterministic_tar(
            stage / "source" / "gx10-producer.tar.gz",
            [("muser", ROOT, gx10_paths)],
            commit_epoch,
        )
        shutil.copy2(binary, stage / "bin" / "muser")
        os.chmod(stage / "bin" / "muser", 0o755)
        shutil.copytree(
            args.mtmd_package.resolve(),
            stage / "lib" / "mtmd",
            symlinks=False,
        )
        shutil.copytree(seal_bundle, stage / "evidence" / "atomic-seal-bundle")
        shutil.copy2(args.identity_receipt, stage / "evidence" / "campaign-identity.json")
        shutil.copy2(args.llama_comparator_receipt, stage / "evidence" / "llama-comparator.json")
        shutil.copy2(args.gx10_container_receipt, stage / "evidence" / "gx10-container.json")
        shutil.copy2(ARTIFACT_MANIFEST, stage / "evidence" / "release-artifacts.json")
        for name in ("LICENSE-APACHE", "LICENSE-MIT", "NOTICE"):
            shutil.copy2(ROOT / name, stage / "licenses" / name)
        shutil.copy2(LLAMA_LICENSE, stage / "licenses" / "llama.cpp-LICENSE")
        for source, output in (
            (KVPACK_ROOT / "LICENSE-APACHE", "kvpack-LICENSE-APACHE"),
            (KVPACK_ROOT / "LICENSE-MIT", "kvpack-LICENSE-MIT"),
            (KVPACK_ROOT / "THIRD_PARTY_NOTICES.md", "kvpack-THIRD_PARTY_NOTICES.md"),
        ):
            if not source.is_file() or source.is_symlink():
                raise RuntimeError(f"kvpack release notice is missing or unsafe: {source}")
            shutil.copy2(source, stage / "licenses" / output)
        rust_notices = collect_cargo_licenses(stage / "licenses" / "rust", metadata)
        (stage / "licenses" / "rust-dependencies.json").write_text(
            json.dumps(rust_notices, indent=2, sort_keys=True) + "\n"
        )
        spdx_created = created.replace("+00:00", "Z")
        write_spdx(
            stage / "sbom.spdx.json",
            metadata,
            args.identity,
            muser_commit,
            spdx_created,
            llama_receipt["source_commit"],
        )
        demo = stage / "demo" / "run.sh"
        demo.write_text(demo_text())
        demo.chmod(0o755)
        request = stage / "demo" / "request.sh"
        request.write_text(demo_request_text())
        request.chmod(0o755)
        (stage / "demo" / "README.md").write_text(demo_readme_text())
        shutil.copy2(ROOT / "scripts" / "release_demo.py", stage / "demo" / "verify-models.py")
        (stage / "demo" / "verify-models.py").chmod(0o755)
        receipt = {
            "schema": "muser.release-candidate.receipt.v1",
            "status": "private-candidate",
            "version": args.version,
            "distribution": "private",
            "public_release_authorized": False,
            "private_tags": {
                "muser": MUSER_PRIVATE_TAG,
                "kvpack": KVPACK_PRIVATE_TAG,
            },
            "identity": args.identity,
            "muser_commit": muser_commit,
            "kvpack_commit": kvpack_commit,
            "kvpack_release_crates": sorted(KVPACK_RELEASE_CRATES),
            "created_at": created,
            "model_weights_bundled": False,
            "artifact_repository": artifact_manifest["repository"],
            "artifact_revision": artifact_manifest["revision"],
            "artifact_manifest_sha256": sha256(ARTIFACT_MANIFEST),
            "llama_source_commit": llama_receipt.get("source_commit"),
            "muser_binary_sha256": sha256(binary),
            "campaign_identity_receipt_sha256": sha256(args.identity_receipt),
            "gx10_container_receipt_sha256": sha256(args.gx10_container_receipt),
            "gx10_image_id": gx10_receipt.get("image_id"),
            "gx10_adapter_sha256": gx10_receipt.get("adapter_sha256"),
            "gx10_cuda_matmul": gx10_receipt.get("cuda_matmul"),
            "mtmd_llama_commit": mtmd_receipt.get("llama_commit"),
            "atomic_seal_result_sha256": sha256(seal_bundle / "RESULT.json"),
            "atomic_seal_identity": seal_result["identity"],
            "affiliation": "independent; not affiliated with or endorsed by Meta or the Muse model authors",
        }
        (stage / "release-receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
        files = sorted(
            path for path in stage.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS"
        )
        sums = "".join(
            f"{sha256(path)}  {path.relative_to(stage).as_posix()}\n" for path in files
        )
        (stage / "SHA256SUMS").write_text(sums)
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "verify_release_candidate.py"), str(stage)],
            check=True,
            stdout=subprocess.PIPE,
        )
        os.rename(stage, args.output_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(json.dumps({**plan, "mode": "built", "muser_commit": muser_commit,
                      "kvpack_commit": kvpack_commit}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
