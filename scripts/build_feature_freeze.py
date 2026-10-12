#!/usr/bin/env python3
"""Build frozen release binaries from a clean offline clone and receipt them."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from release_identity import identity, sha256
from release_readiness import MANDATORY, atomic_json


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str) -> dict:
    path = ROOT / relative
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"frozen input is missing or unsafe: {relative}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"frozen input is not an object: {relative}")
    return value


def validate_frozen_state(matrix_path: Path) -> dict:
    contract = load("release/feature-contract-v1.json")
    findings = load("release/findings-v1.json")
    tuning = load("release/dflash-tuning-v1.json")
    route = load("release/dflash-route-policy-v1.json")
    lock = load("release/release-lock.json")
    if contract.get("status") != "frozen":
        raise RuntimeError("feature contract is not frozen")
    open_findings = [
        item.get("id")
        for item in findings.get("findings", [])
        if not isinstance(item, dict) or item.get("status") != "closed"
    ]
    if open_findings:
        raise RuntimeError(f"open findings block feature freeze: {open_findings}")
    if (
        tuning.get("status") != "frozen"
        or tuning.get("selected_verify_length") not in tuning.get("allowed_verify_lengths", [])
    ):
        raise RuntimeError("DFlash tuning is not frozen to an allowed value")
    ane_gate = route.get("ane_gate")
    if (
        set(route) != {"schema", "status", "auto_route", "ane_gate", "policy"}
        or route.get("schema") != "muser.dflash-route-policy.v1"
        or route.get("status") != "v0.1-metal-only"
        or not isinstance(route.get("policy"), str)
        or not route["policy"].strip()
        or not isinstance(ane_gate, dict)
        or set(ane_gate) != {"required", "passed", "same_build_receipt"}
        or ane_gate.get("required") is not False
        or ane_gate.get("passed") is not False
        or ane_gate.get("same_build_receipt") is not None
        or route.get("auto_route") != "metal"
    ):
        raise RuntimeError("v0.1 DFlash route policy is not frozen to Metal-only auto routing")
    if lock.get("sealing_enabled") is not False:
        raise RuntimeError("feature freeze must be built while sealing remains disabled")
    if not matrix_path.is_file() or matrix_path.is_symlink():
        raise RuntimeError("reviewed release matrix is missing or unsafe")
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if (
        matrix.get("schema") != "muser.unsealed-matrix-config.v1"
        or not isinstance(matrix.get("lanes"), dict)
        or set(matrix["lanes"]) != MANDATORY
    ):
        raise RuntimeError("reviewed release matrix does not contain all mandatory lanes")
    return matrix


def binary_names(source: Path) -> list[str]:
    raw = subprocess.run(
        ["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"],
        cwd=source,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    metadata = json.loads(raw)
    names = {
        target["name"]
        for package in metadata.get("packages", [])
        for target in package.get("targets", [])
        if "bin" in target.get("kind", [])
    }
    if "muser" not in names:
        raise RuntimeError("release workspace has no muser server binary")
    return sorted(names)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--matrix-config", type=Path, required=True)
    value.add_argument("--out", type=Path, required=True)
    value.add_argument("--execute", action="store_true")
    return value


def fsync_tree(root: Path) -> None:
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    directories = sorted(
        (item for item in root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    )
    for path in directories + [root]:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def main() -> int:
    args = parser().parse_args()
    plan = {
        "schema": "muser.feature-freeze.plan.v1",
        "mode": "execute" if args.execute else "plan",
        "matrix_config": str(args.matrix_config),
        "out": str(args.out),
        "clean_clone": True,
        "offline_build": True,
        "seals_emitted": False,
    }
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    try:
        matrix_path = args.matrix_config.resolve()
        validate_frozen_state(matrix_path)
        if args.out.is_symlink():
            raise RuntimeError("feature-freeze output may not be a symlink")
        output = args.out.resolve()
        if output == ROOT or ROOT in output.parents:
            raise RuntimeError("feature-freeze output must be outside the source tree")
        if output.exists() or output.is_symlink():
            raise RuntimeError(f"refusing to replace feature-freeze output: {args.out}")
        if subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout:
            raise RuntimeError("feature freeze requires a clean committed worktree")
        output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
        try:
            with tempfile.TemporaryDirectory(
                prefix=f".{output.name}.work-", dir=output.parent
            ) as work_directory:
                work = Path(work_directory)
                clone = work / "source"
                subprocess.run(
                    [
                        "git", "clone", "--local", "--no-hardlinks", "--quiet",
                        str(ROOT), str(clone),
                    ],
                    check=True,
                    stdin=subprocess.DEVNULL,
                )
                subprocess.run(
                    [sys.executable, str(clone / "scripts/audit_vendored_kvpack.py")],
                    cwd=clone,
                    check=True,
                    stdin=subprocess.DEVNULL,
                )
                target = work / "target"
                environment = os.environ.copy()
                environment["CARGO_NET_OFFLINE"] = "true"
                environment["CARGO_TARGET_DIR"] = str(target)
                subprocess.run(
                    [
                        "cargo", "build", "--workspace", "--all-features", "--release",
                        "--locked", "--offline", "--bins",
                    ],
                    cwd=clone,
                    env=environment,
                    check=True,
                    stdin=subprocess.DEVNULL,
                )
                output_binaries = stage / "binaries"
                output_binaries.mkdir()
                paths: dict[str, Path] = {}
                for name in binary_names(clone):
                    source = target / "release" / name
                    if not source.is_file() or source.is_symlink():
                        raise RuntimeError(f"clean clone did not produce release binary {name}")
                    destination = output_binaries / name
                    shutil.copy2(source, destination)
                    paths[name] = destination
            campaign = identity(paths)
            shutil.copy2(matrix_path, stage / "matrix-config.json")
            atomic_json(stage / "campaign-identity.json", campaign)
            receipt = {
                "schema": "muser.feature-freeze.v1",
                "status": "passed",
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "identity": campaign["digest"],
                "campaign_identity": campaign,
                "campaign_identity_sha256": sha256(stage / "campaign-identity.json"),
                "source_commit": campaign["source"]["commit"],
                "source_tree": campaign["source"]["tree"],
                "matrix_config_sha256": sha256(stage / "matrix-config.json"),
                "binaries": {
                    name: {
                        "path": f"binaries/{name}",
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                    for name, path in sorted(paths.items())
                },
                "clean_clone": True,
                "offline_build": True,
                "seals_emitted": False,
            }
            atomic_json(stage / "RESULT.json", receipt)
            fsync_tree(stage)
            os.rename(stage, output)
            descriptor = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"feature freeze failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
