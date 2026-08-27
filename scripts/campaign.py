#!/usr/bin/env python3
"""Resumable, fail-fast Muse qualification campaign driver.

Dry-run is side-effect free. Smoke materializes deterministic fixtures and
executes one guarded correctness cell followed by a small parser/route sample.
Kvpack-full follows correctness. Baseline-full additionally requires kvpack
and vision seals. DFlash tuning and qualification require the baseline seal,
and qualification consumes a frozen tuning receipt. A failed command
permanently taints that run ID; its evidence is retained, and only a complete
packet can be sealed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
from typing import Any
import zlib

import release_matrix
from release_identity import identity as release_identity_v3
from release_lock import load_release_lock


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
BODY = (19873, 24, 10676, 768, 1085, 13634, 2304, 1509)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--kvpack-full", action="store_true")
    mode.add_argument("--vision-full", action="store_true")
    mode.add_argument("--dflash-tune", action="store_true")
    mode.add_argument("--dflash-full", action="store_true")
    mode.add_argument("--ane-full", action="store_true")
    mode.add_argument("--remote-full", action="store_true")
    parser.add_argument(
        "--dry-run-stage",
        choices=("complete", "smoke", "baseline", "kvpack", "vision", "dflash", "ane", "remote"),
        default="complete",
        help="limit a dry-run to the exact stage that will be launched",
    )
    parser.add_argument(
        "--depths",
        default="",
        help=(
            "diagnostic depth subset, no seals. With --smoke: prefill/decode/TTFT "
            "instead of 128/512. With --dflash-full: dflash-DEPTH-p1/p2 vs llama "
            "at verify-length 7. Same idea as Ferrite bench_vs_llama --tier 1 --depths."
        ),
    )
    parser.add_argument("--identity", required=True)
    parser.add_argument("--run-id", default="dry-run")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--muser-bench", default="target/release/muser-bench")
    parser.add_argument("--muser-greedy", default="target/release/muser-greedy-evidence")
    parser.add_argument("--muser-forward", default="target/release/muser-forward-evidence")
    parser.add_argument("--muser-fixture", default="target/release/muser-token-fixture")
    parser.add_argument("--muser-kvpack", default="target/release/muser-kvpack-qualify")
    parser.add_argument("--muser-dflash", default="target/release/muser-dflash-qualify")
    parser.add_argument("--muser-vision", default="target/release/muser-vision-qualify")
    parser.add_argument("--muser-ane", default="target/release/muser-ane-qualify")
    parser.add_argument("--muser-remote", default="target/release/muser-remote-qualify")
    parser.add_argument("--muser-server", default="target/release/muser")
    parser.add_argument("--dflash", type=Path)
    parser.add_argument("--mmproj", type=Path)
    parser.add_argument("--mtmd-bridge", type=Path)
    parser.add_argument("--ane-manifest", type=Path)
    parser.add_argument("--coreml-plan-receipt", type=Path)
    parser.add_argument("--cluster-config", type=Path)
    parser.add_argument("--llama-bench", default="llama-bench")
    parser.add_argument("--llama-perplexity", default="llama-perplexity")
    parser.add_argument("--llama-server", default="llama-server")
    parser.add_argument("--llama-receipt", type=Path)
    parser.add_argument("--ggml-metallib", type=Path)
    parser.add_argument("--ggml-metallib-receipt", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("results/baseline-seal"))
    parser.add_argument("--bos-token", type=int, default=200000)
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=202048,
        help="tokenizer vocabulary used for deterministic fixtures (official Muse: 202048)",
    )
    parser.add_argument("--correctness-receipt", type=Path)
    parser.add_argument("--greedy-correctness-receipt", type=Path)
    parser.add_argument("--baseline-seal", type=Path)
    parser.add_argument("--kvpack-seal", type=Path)
    parser.add_argument("--vision-seal", type=Path)
    parser.add_argument("--dflash-tuning-freeze", type=Path)
    parser.add_argument("--dflash-seal", type=Path)
    parser.add_argument("--muser-url", default="http://127.0.0.1:4949")
    parser.add_argument("--llama-url", default="http://127.0.0.1:8080")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_binary(raw: str) -> Path | None:
    candidate = Path(raw)
    if candidate.parent != Path(".") or "/" in raw:
        resolved = candidate if candidate.is_absolute() else ROOT / candidate
        return resolved.resolve() if resolved.is_file() else None
    found = shutil.which(raw)
    return Path(found).resolve() if found else None


def sysctl(name: str) -> str | None:
    try:
        return subprocess.run(
            ["sysctl", "-n", name],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def hardware_identity() -> dict[str, Any]:
    """Static machine identity: safe to fold into the frozen campaign digest."""
    try:
        macos_build = subprocess.run(
            ["sw_vers", "-buildVersion"], check=True, text=True, stdout=subprocess.PIPE,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        macos_build = None
    return {
        "chip_model": sysctl("machdep.cpu.brand_string"),
        "macos_build": macos_build,
    }


def hardware_observed() -> dict[str, Any]:
    """Best-effort, point-in-time thermal/power state.

    Excluded from the identity digest: it fluctuates run to run and must
    never gate whether a later stage matches an earlier stage's identity.
    """
    return {
        "thermal_pressure_level": sysctl("kern.thermal_state"),
        "power_source": sysctl("hw.acpower") or None,
    }


def git_output(*arguments: str, text: bool = True) -> str | bytes:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, text=text,
        stdout=subprocess.PIPE,
    ).stdout


def source_snapshot() -> dict[str, Any]:
    tracked_diff = git_output("diff", "--binary", "HEAD", "--", text=False)
    assert isinstance(tracked_diff, bytes)
    raw_untracked = git_output("ls-files", "--others", "--exclude-standard", "-z", text=False)
    assert isinstance(raw_untracked, bytes)
    untracked: list[dict[str, Any]] = []
    for raw_path in sorted(field for field in raw_untracked.split(b"\0") if field):
        path_text = raw_path.decode("utf-8")
        path = ROOT / path_text
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"untracked source is not a regular file: {path_text}")
        untracked.append(
            {"path": path_text, "bytes": path.stat().st_size, "sha256": sha256(path)}
        )
    return {
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "tracked_diff_bytes": len(tracked_diff),
        "untracked": untracked,
        "dirty": bool(tracked_diff or untracked),
    }


def validate_llama_receipt(
    receipt_path: Path,
    llama_artifact: dict[str, Any] | None,
    perplexity_artifact: dict[str, Any] | None,
    server_artifact: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, list[str]]:
    failures: list[str] = []
    if not receipt_path.is_file() or receipt_path.is_symlink():
        return None, [f"missing or unsafe llama receipt: {receipt_path}"]
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return None, [f"invalid llama receipt: {error}"]
    if receipt.get("schema") != "muser.llama_comparator.source_receipt.v3":
        failures.append("wrong llama comparator receipt schema")
    source_commit = receipt.get("source_commit")
    if not isinstance(source_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        failures.append("llama receipt has no exact source commit")
    patch_path = ROOT / "scripts" / "llama_bench_fixture.patch"
    if receipt.get("patch_sha256") != sha256(patch_path):
        failures.append("llama receipt patch digest differs from tracked apparatus")
    expected = receipt.get("artifacts", {}).get("llama-bench", {})
    if llama_artifact is None or any(
        expected.get(key) != llama_artifact.get(key) for key in ("bytes", "sha256")
    ):
        failures.append("llama-bench binary differs from its source receipt")
    expected_perplexity = receipt.get("artifacts", {}).get("llama-perplexity", {})
    if perplexity_artifact is None or any(
        expected_perplexity.get(key) != perplexity_artifact.get(key)
        for key in ("bytes", "sha256")
    ):
        failures.append("llama-perplexity binary differs from its source receipt")
    expected_server = receipt.get("artifacts", {}).get("llama-server", {})
    if server_artifact is None or any(
        expected_server.get(key) != server_artifact.get(key)
        for key in ("bytes", "sha256")
    ):
        failures.append("llama-server binary differs from its source receipt")
    summary = {
        "basename": receipt_path.name,
        "bytes": receipt_path.stat().st_size,
        "sha256": sha256(receipt_path),
        "source_commit": source_commit,
        "source_tree": receipt.get("source_tree"),
        "patch_sha256": receipt.get("patch_sha256"),
        "patched_source_sha256": receipt.get("patched_source_sha256"),
        "patched_perplexity_source_sha256": receipt.get(
            "patched_perplexity_source_sha256"
        ),
        "patched_server_source_sha256": receipt.get("patched_server_source_sha256"),
    }
    return summary, failures


def validate_metallib_receipt(
    receipt_path: Path,
    metallib_artifact: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, list[str]]:
    failures: list[str] = []
    if not receipt_path.is_file() or receipt_path.is_symlink():
        return None, [f"missing or unsafe GGML metallib receipt: {receipt_path}"]
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return None, [f"invalid GGML metallib receipt: {error}"]
    if receipt.get("schema") != "muser.llama_metallib.source_receipt.v1":
        failures.append("wrong GGML metallib receipt schema")
    source_commit = receipt.get("source_commit")
    if not isinstance(source_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        failures.append("GGML metallib receipt has no exact source commit")
    if metallib_artifact is None or (
        receipt.get("binary_size_bytes") != metallib_artifact.get("bytes")
        or receipt.get("binary_sha256") != metallib_artifact.get("sha256")
    ):
        failures.append("GGML metallib differs from its source receipt")
    summary = {
        "basename": receipt_path.name,
        "bytes": receipt_path.stat().st_size,
        "sha256": sha256(receipt_path),
        "source_commit": source_commit,
        "source_tree": receipt.get("source_tree"),
        "merged_source_sha256": receipt.get("merged_source_sha256"),
        "sdk_version": receipt.get("sdk_version"),
        "xcode_version": receipt.get("xcode_version"),
    }
    return summary, failures


def static_identity(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    artifacts: dict[str, Any] = {}
    identity_inputs: dict[str, Path] = {}
    for name, path in [
        ("model", args.model),
        ("muser_bench", resolve_binary(args.muser_bench)),
        ("muser_greedy", resolve_binary(args.muser_greedy)),
        ("muser_forward", resolve_binary(args.muser_forward)),
        ("muser_fixture", resolve_binary(args.muser_fixture)),
        ("muser_kvpack", resolve_binary(args.muser_kvpack)),
        ("muser_dflash", resolve_binary(args.muser_dflash)),
        ("muser_vision", resolve_binary(args.muser_vision)),
        ("muser_remote", resolve_binary(args.muser_remote)),
        ("muser_server", resolve_binary(args.muser_server)),
        ("llama_bench", resolve_binary(args.llama_bench)),
        ("llama_perplexity", resolve_binary(args.llama_perplexity)),
        ("llama_server", resolve_binary(args.llama_server)),
    ]:
        if path is None or not Path(path).is_file():
            blockers.append(f"missing {name}: {path or getattr(args, name, '')}")
            artifacts[name] = None
            continue
        resolved = Path(path).resolve()
        identity_inputs[name] = resolved
        artifacts[name] = {
            "basename": resolved.name,
            "bytes": resolved.stat().st_size,
            "sha256": sha256(resolved),
        }
    if args.ane_full or (args.dry_run and args.dry_run_stage == "ane"):
        path = resolve_binary(args.muser_ane)
        if path is None or not path.is_file():
            blockers.append(f"missing muser_ane: {args.muser_ane}")
            artifacts["muser_ane"] = None
        else:
            identity_inputs["muser_ane"] = path
            artifacts["muser_ane"] = {
                "basename": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    else:
        artifacts["muser_ane"] = None
    if args.dflash is not None and args.dflash.is_file():
        resolved_dflash = args.dflash.resolve()
        identity_inputs["dflash_artifact"] = resolved_dflash
        artifacts["dflash"] = {
            "basename": resolved_dflash.name,
            "bytes": resolved_dflash.stat().st_size,
            "sha256": sha256(resolved_dflash),
            "format": "official-gguf",
        }
    elif (
        args.dflash is not None
        and args.dflash.is_dir()
        and (args.dflash / "config.json").is_file()
        and (args.dflash / "model.safetensors").is_file()
    ):
        resolved_dflash = args.dflash.resolve()
        config = resolved_dflash / "config.json"
        weights = resolved_dflash / "model.safetensors"
        digest = hashlib.sha256(b"muser-dflash-artifact-v1\0")
        total = 0
        for path in (config, weights):
            total += path.stat().st_size
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
        artifacts["dflash"] = {
            "basename": resolved_dflash.name,
            "bytes": total,
            "sha256": digest.hexdigest(),
            "format": "development-safetensors",
        }
    else:
        artifacts["dflash"] = None
        if (
            (args.dry_run and args.dry_run_stage in ("complete", "dflash", "ane", "remote"))
            or args.dflash_tune or args.dflash_full
            or args.ane_full or args.remote_full
        ):
            blockers.append(f"missing official DFlash artifact: {args.dflash}")
    for name, path in (("mmproj", args.mmproj), ("mtmd_bridge", args.mtmd_bridge)):
        if path is not None and path.is_file():
            resolved = path.resolve()
            identity_inputs[f"{name}_artifact"] = resolved
            artifacts[name] = {
                "basename": resolved.name,
                "bytes": resolved.stat().st_size,
                "sha256": sha256(resolved),
            }
        else:
            artifacts[name] = None
            if (
                (args.dry_run and args.dry_run_stage in ("complete", "vision", "remote"))
                or args.vision_full or args.remote_full
            ):
                blockers.append(f"missing official vision artifact {name}: {path}")
    for name, path in (
        ("ane_manifest", args.ane_manifest),
        ("coreml_plan_receipt", args.coreml_plan_receipt),
    ):
        if path is not None and path.is_file():
            resolved = path.resolve()
            identity_inputs[name] = resolved
            artifacts[name] = {
                "basename": resolved.name,
                "bytes": resolved.stat().st_size,
                "sha256": sha256(resolved),
            }
        else:
            artifacts[name] = None
            if (
                (args.dry_run and args.dry_run_stage == "ane") or args.ane_full
            ):
                blockers.append(f"missing ANE qualification artifact {name}: {path}")
    if args.cluster_config is not None and args.cluster_config.is_file():
        resolved = args.cluster_config.resolve()
        identity_inputs["cluster_config"] = resolved
        artifacts["cluster_config"] = {
            "basename": resolved.name,
            "bytes": resolved.stat().st_size,
            "sha256": sha256(resolved),
        }
    else:
        artifacts["cluster_config"] = None
        if (
            (args.dry_run and args.dry_run_stage in ("complete", "remote"))
            or args.remote_full
        ):
            blockers.append(f"missing remote cluster config: {args.cluster_config}")
    metallib = args.ggml_metallib
    if metallib is None and os.environ.get("MUSER_GGML_METALLIB"):
        metallib = Path(os.environ["MUSER_GGML_METALLIB"])
    if metallib is not None:
        if not metallib.is_file():
            blockers.append(f"missing MUSER_GGML_METALLIB: {metallib}")
            artifacts["ggml_metallib"] = None
        else:
            metallib = metallib.resolve()
            identity_inputs["ggml_metallib"] = metallib
            artifacts["ggml_metallib"] = {
                "basename": metallib.name,
                "bytes": metallib.stat().st_size,
                "sha256": sha256(metallib),
            }
    else:
        artifacts["ggml_metallib"] = None
        blockers.append("--ggml-metallib is required for reproducible Metal routes")

    if args.ggml_metallib_receipt is not None:
        metallib_summary, metallib_failures = validate_metallib_receipt(
            args.ggml_metallib_receipt.resolve(), artifacts.get("ggml_metallib")
        )
        artifacts["ggml_metallib_receipt"] = metallib_summary
        blockers.extend(metallib_failures)
    else:
        artifacts["ggml_metallib_receipt"] = None
        blockers.append("--ggml-metallib-receipt is required for reproducible Metal routes")

    release_manifest_path = ROOT / "docs" / "release-artifacts.json"
    release_manifest = json.loads(release_manifest_path.read_text())
    pinned = release_manifest["artifacts"]
    required_pins: list[tuple[str, str]] = []
    if any(
        (
            args.dry_run,
            args.full,
            args.kvpack_full,
            args.vision_full,
            args.dflash_tune,
            args.dflash_full,
            args.ane_full,
            args.remote_full,
        )
    ):
        required_pins.append(("model", "target"))
    if args.vision_full or args.remote_full:
        required_pins.append(("mmproj", "vision"))
    if args.dry_run and args.dry_run_stage in ("complete", "vision", "remote"):
        required_pins.append(("mmproj", "vision"))
    if args.dflash_tune or args.dflash_full or args.ane_full or args.remote_full:
        required_pins.append(("dflash", "dflash"))
    if args.dry_run and args.dry_run_stage in ("complete", "dflash", "ane", "remote"):
        required_pins.append(("dflash", "dflash"))
    for artifact_name, pin_name in required_pins:
        actual = artifacts.get(artifact_name)
        expected = pinned[pin_name]
        if not isinstance(actual, dict) or (
            actual.get("basename") != expected["filename"]
            or actual.get("sha256") != expected["sha256"]
        ):
            blockers.append(
                f"{artifact_name} is not pinned official {expected['filename']} "
                f"at release revision {release_manifest['revision']}"
            )

    if args.llama_receipt is not None:
        receipt_summary, receipt_failures = validate_llama_receipt(
            args.llama_receipt.resolve(),
            artifacts.get("llama_bench"),
            artifacts.get("llama_perplexity"),
            artifacts.get("llama_server"),
        )
        artifacts["llama_comparator_receipt"] = receipt_summary
        blockers.extend(receipt_failures)
        metallib_summary = artifacts.get("ggml_metallib_receipt")
        if (
            isinstance(receipt_summary, dict)
            and isinstance(metallib_summary, dict)
            and metallib_summary.get("source_commit") != receipt_summary.get("source_commit")
        ):
            blockers.append("GGML metallib and llama comparator source commits differ")
    else:
        artifacts["llama_comparator_receipt"] = None
        blockers.append("--llama-receipt is required for the complete campaign identity")

    snapshot = source_snapshot()
    if snapshot["dirty"]:
        if args.smoke:
            print(
                "WARNING: source tree is dirty; --smoke evidence is diagnostic-only "
                "and not seal-eligible.",
                file=sys.stderr,
            )
        elif not args.dry_run:
            blockers.append("dirty source tree: seal-eligible stages require a clean tree")
    try:
        identity = release_identity_v3(identity_inputs)
        if args.dry_run:
            # Planning output is never a final campaign identity, even when
            # the worktree happens to be clean enough to compute its digest.
            # Keep this explicit so dry-run semantics do not depend on ambient
            # repository dirtiness.
            identity["preview_only"] = True
    except RuntimeError as error:
        # A dry-run remains a side-effect-free planning operation on a dirty
        # development tree. It cannot claim a final identity.
        blockers.append(str(error))
        identity = {
            "schema": "muser.campaign-identity.v3",
            "source": {
                "commit": str(git_output("rev-parse", "HEAD")).strip(),
                "tree": str(git_output("rev-parse", "HEAD^{tree}")).strip(),
            },
            "digest": "unavailable-unfrozen-worktree",
            "preview_only": True,
        }
    return identity, blockers


def matrix_args(args: argparse.Namespace, identity_digest: str) -> argparse.Namespace:
    muser = resolve_binary(args.muser_bench)
    llama = resolve_binary(args.llama_bench)
    muser_server = resolve_binary(args.muser_server)
    llama_server = resolve_binary(args.llama_server)
    return argparse.Namespace(
        identity=identity_digest,
        model=str(args.model.resolve()),
        muser_bench=str(muser or args.muser_bench),
        muser_kvpack=str(resolve_binary(args.muser_kvpack) or args.muser_kvpack),
        muser_dflash=str(resolve_binary(args.muser_dflash) or args.muser_dflash),
        muser_vision=str(resolve_binary(args.muser_vision) or args.muser_vision),
        muser_ane=str(resolve_binary(args.muser_ane) or args.muser_ane),
        muser_remote=str(resolve_binary(args.muser_remote) or args.muser_remote),
        dflash=str(args.dflash.resolve() if args.dflash else "official-dflash-artifact-absent"),
        mmproj=str(args.mmproj.resolve() if args.mmproj else "official-mmproj-artifact-absent"),
        mtmd_bridge=str(
            args.mtmd_bridge.resolve() if args.mtmd_bridge else "mtmd-bridge-package-absent"
        ),
        ane_manifest=str(
            args.ane_manifest.resolve() if args.ane_manifest else "ane-manifest-absent"
        ),
        coreml_plan_receipt=str(
            args.coreml_plan_receipt.resolve()
            if args.coreml_plan_receipt
            else "coreml-plan-receipt-absent"
        ),
        cluster_config=str(
            args.cluster_config.resolve() if args.cluster_config else "cluster-config-absent"
        ),
        llama_bench=str(llama or args.llama_bench),
        muser_server=str(muser_server or args.muser_server),
        llama_server=str(llama_server or args.llama_server),
        out_dir=str(args.out_dir.resolve()),
        muser_url=args.muser_url,
        llama_url=args.llama_url,
    )


def parse_diagnostic_depths(raw: str) -> list[int]:
    if not raw.strip():
        return []
    depths = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not depths or any(depth <= 0 for depth in depths):
        raise SystemExit("--depths must contain positive integers")
    return list(dict.fromkeys(depths))


def diagnostic_baseline_cells(
    baseline_cells: list[dict[str, Any]],
    ttft_cells: list[dict[str, Any]],
    depths: list[int],
) -> list[dict[str, Any]]:
    if not depths:
        return [
            baseline_cells[0],
            baseline_cells[10],  # PP128, TG512
            ttft_cells[0],
        ]
    wanted = (
        {f"prefill-{depth}" for depth in depths}
        | {f"decode-{depth}" for depth in depths}
        | {f"ttft-{depth}" for depth in depths}
    )
    selected = [
        cell for cell in baseline_cells + ttft_cells if cell["cell"] in wanted
    ]
    missing = wanted - {cell["cell"] for cell in selected}
    if missing:
        raise SystemExit("unknown diagnostic cells: " + ", ".join(sorted(missing)))
    return selected


def diagnostic_dflash_cells(
    dflash_cells: list[dict[str, Any]],
    depths: list[int],
) -> list[dict[str, Any]]:
    if not depths:
        return dflash_cells
    wanted = {f"dflash-{depth}-p{variant}" for depth in depths for variant in (1, 2)}
    selected = [cell for cell in dflash_cells if cell["cell"] in wanted]
    missing = wanted - {cell["cell"] for cell in selected}
    if missing:
        raise SystemExit("unknown diagnostic DFlash cells: " + ", ".join(sorted(missing)))
    return selected


def fixture_tokens(count: int, bos: int, vocab_size: int) -> list[int]:
    if count == 0:
        return []
    return [
        bos if index == 0 else BODY[(index - 1) % len(BODY)] % vocab_size
        for index in range(count)
    ]


def variant_fixture_tokens(count: int, bos: int, vocab_size: int, variant: int) -> list[int]:
    """Deterministic, disjoint token fixtures without tokenizer ambiguity."""
    if count == 0:
        return []
    salt = (7919 * variant) % vocab_size
    return [
        bos if index == 0 else (BODY[(index - 1 + variant) % len(BODY)] + salt) % vocab_size
        for index in range(count)
    ]


def fixture_bytes(tokens: list[int]) -> bytes:
    return ("\n".join(map(str, tokens)) + ("\n" if tokens else "")).encode()


def publish_exact(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise RuntimeError(f"existing fixture differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        written = 0
        while written < len(payload):
            written += os.write(fd, payload[written:])
        os.fsync(fd)
    finally:
        os.close(fd)


def materialize_fixtures(args: argparse.Namespace) -> dict[str, str]:
    directory = args.out_dir.resolve() / "fixtures"
    result: dict[str, str] = {}
    kvpack_depths = {cut + suffix for cut in [8192, 16384, 32768, 65536, 128768]
                     for suffix in [1, 255, 256, 257, 2047]}
    for depth in sorted(
        (set(release_matrix.PREFILL + release_matrix.DECODE) | kvpack_depths) - {0}
    ):
        path = directory / f"prompt-{depth}.txt"
        payload = fixture_bytes(fixture_tokens(depth, args.bos_token, args.vocab_size))
        publish_exact(path, payload)
        result[path.name] = hashlib.sha256(payload).hexdigest()
    decode = directory / "decode-64.txt"
    payload = fixture_bytes(fixture_tokens(64, args.bos_token, args.vocab_size))
    publish_exact(decode, payload)
    result[decode.name] = hashlib.sha256(payload).hexdigest()
    for prefix, depths in (
        ("dflash-tune", release_matrix.DFLASH_TUNE_DEPTHS),
        ("dflash", release_matrix.DFLASH_DEPTHS),
    ):
        variant_offset = 100 if prefix == "dflash-tune" else 200
        for depth in depths:
            for variant in (1, 2):
                path = directory / f"{prefix}-{depth}-p{variant}.txt"
                payload = fixture_bytes(
                    variant_fixture_tokens(
                        depth, args.bos_token, args.vocab_size, variant_offset + variant
                    )
                )
                publish_exact(path, payload)
                result[path.name] = hashlib.sha256(payload).hexdigest()
    for variant, (name, width, height) in enumerate(
        [
            ("low-square", 224, 224),
            ("wide", 1024, 256),
            ("tall", 256, 1024),
            ("high-resolution", 2048, 1536),
        ],
        start=1,
    ):
        path = directory / f"vision-{name}.png"
        payload = png_fixture(width, height, variant)
        publish_exact(path, payload)
        result[path.name] = hashlib.sha256(payload).hexdigest()
    return result


def png_fixture(width: int, height: int, variant: int) -> bytes:
    """Deterministic RGB fixture with gradients, edges, and fine texture."""
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload)) + kind + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    rows = bytearray()
    for y in range(height):
        rows.append(0)  # PNG filter None: fixture bytes are independently auditable.
        row = bytearray(width * 3)
        for x in range(width):
            offset = x * 3
            checker = 71 if ((x // 17) ^ (y // 19) ^ variant) & 1 else 0
            row[offset] = (x * 13 + y * 3 + checker + variant * 29) & 0xFF
            row[offset + 1] = (x * 5 + y * 11 + checker * 2 + variant * 47) & 0xFF
            row[offset + 2] = (x * 7 + y * 17 + checker * 3 + variant * 61) & 0xFF
        rows.extend(row)
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return signature + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b"")


def append_record(path: Path, record: dict[str, Any]) -> None:
    encoded = (json.dumps(record, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        written = 0
        while written < len(encoded):
            written += os.write(fd, encoded[written:])
        os.fsync(fd)
    finally:
        os.close(fd)


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def plan_digest(command: list[str], identity: str) -> str:
    payload = json.dumps(
        {"command": command, "identity": identity},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def parse_json_lines(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def coefficient_of_variation(samples: list[int | float]) -> float:
    mean = sum(samples) / len(samples)
    variance = sum((sample - mean) ** 2 for sample in samples) / len(samples)
    return variance ** 0.5 / mean


def position_range_sha256(start: int, end: int) -> str:
    digest = hashlib.sha256()
    for position in range(start, end):
        digest.update(position.to_bytes(8, "little"))
    return "sha256:" + digest.hexdigest()


def baseline_cell_shape(cell: str) -> tuple[str, int]:
    match = re.fullmatch(r"(prefill|decode|ttft)-(\d+)", cell)
    if match is None:
        raise RuntimeError(f"invalid baseline cell name: {cell}")
    return match.group(1), int(match.group(2))


def dflash_cell_shape(cell: str) -> tuple[int, int]:
    qualify = re.fullmatch(r"dflash-(\d+)-p([12])", cell)
    if qualify is not None:
        return int(qualify.group(1)), int(qualify.group(2))
    tune = re.fullmatch(r"dflash-tune-(\d+)-p([12])-v(\d+)", cell)
    if tune is not None:
        return int(tune.group(1)), int(tune.group(2))
    raise RuntimeError(f"invalid dflash cell name: {cell}")


def normalize_evidence(
    engine: str, log: Path, cell: str | None = None, identity: str | None = None
) -> dict[str, Any]:
    if engine == "correctness":
        text = log.read_text(errors="replace")
        if "test result: ok." not in text:
            raise RuntimeError("correctness smoke did not report a passing Rust test")
        return {"raw_ns": [], "cv": 0.0, "fingerprint": {"kind": "correctness-smoke"}}
    records = parse_json_lines(log)
    if engine in ("ttft-muser", "ttft-llama"):
        if cell is None or identity is None:
            raise RuntimeError("TTFT normalization requires its planned cell and identity")
        surface, depth = baseline_cell_shape(cell)
        if surface != "ttft":
            raise RuntimeError(f"TTFT evidence attached to non-TTFT cell {cell}")
        expected_samples = 5
        samples = [
            record for record in records
            if record.get("schema") == "muser.server-ttft.v2"
            and record.get("kind") == "sample"
        ]
        summaries = [
            record for record in records
            if record.get("schema") == "muser.server-ttft.v2"
            and record.get("kind") == "summary"
        ]
        if len(samples) != expected_samples or len(summaries) != 1:
            raise RuntimeError("TTFT log has the wrong sample or summary count")
        if [record.get("repetition") for record in samples] != list(range(expected_samples)):
            raise RuntimeError("TTFT repetitions are absent, duplicated, or reordered")
        raw_ns = [record.get("elapsed_ns") for record in samples]
        if not all(isinstance(value, int) and value > 0 for value in raw_ns):
            raise RuntimeError("TTFT raw nanoseconds are missing or non-positive")
        summary = summaries[0]
        if summary.get("raw_ns") != raw_ns or summary.get("seal_eligible") is not True:
            raise RuntimeError("TTFT summary does not seal the exact requested prompt depth")
        expected_engine = engine.removeprefix("ttft-")
        if any(
            record.get("engine") != expected_engine
            or record.get("identity") != identity
            or record.get("depth") != depth
            for record in samples
        ):
            raise RuntimeError("TTFT log contains a mixed engine route")
        prompt_counts = summary.get("reported_prompt_tokens")
        if (
            summary.get("engine") != expected_engine
            or summary.get("identity") != identity
            or summary.get("depth") != depth
            or not isinstance(prompt_counts, list)
            or prompt_counts != [depth] * expected_samples
        ):
            raise RuntimeError("TTFT summary does not prove the exact prompt depth")
        content_digests = summary.get("first_content_digests")
        if (
            not isinstance(content_digests, list)
            or len(content_digests) != expected_samples
            or len(set(content_digests)) != 1
        ):
            raise RuntimeError("TTFT repetitions did not produce one exact first token")
        return {
            "raw_ns": raw_ns,
            "cv": coefficient_of_variation(raw_ns),
            "full_recompute_ns": summary.get("full_recompute_ns"),
            "source_prefill_ns": summary.get("source_prefill_ns"),
            "publication_overhead_ratio": summary.get("publication_overhead_ratio"),
            "miss_overhead_ratio": summary.get("miss_overhead_ratio"),
            "speedup_geomean_cell": summary.get("speedup_geomean_cell"),
            "fingerprint": {
                "schema": summary.get("schema"),
                "engine": expected_engine,
                "prompt_sha256": summary.get("prompt_sha256"),
                "reported_prompt_tokens": depth,
                "first_content_sha256": content_digests[0],
                "server_lifecycle": summary.get("server_lifecycle"),
                "cache": "disabled",
            },
        }
    if engine == "kvpack":
        if cell is None or identity is None:
            raise RuntimeError("kvpack normalization requires its planned cell and identity")
        exact = re.fullmatch(r"kvpack-(resident|durable|remote)-exact-(\d+)", cell)
        ancestor = re.fullmatch(
            r"kvpack-(resident|durable|remote)-ancestor-(\d+)-s(\d+)", cell
        )
        if exact is not None:
            expected_source = exact.group(1)
            expected_lookup = "exact-final"
            expected_prompt = int(exact.group(2))
            expected_cut = expected_prompt
            expected_suffix = 0
        elif ancestor is not None:
            expected_source = ancestor.group(1)
            expected_lookup = "deepest-ancestor"
            expected_cut = int(ancestor.group(2))
            expected_suffix = int(ancestor.group(3))
            expected_prompt = expected_cut + expected_suffix
        else:
            raise RuntimeError(f"invalid kvpack cell name: {cell}")
        samples = [
            record for record in records
            if record.get("schema") == "muser.kvpack-qualify.v1"
            and record.get("kind") == "sample"
        ]
        summaries = [
            record for record in records
            if record.get("schema") == "muser.kvpack-qualify.v1"
            and record.get("kind") == "summary"
        ]
        if len(samples) != 3 or len(summaries) != 1:
            raise RuntimeError("kvpack log requires exactly three samples and one summary")
        if [record.get("repetition") for record in samples] != [0, 1, 2]:
            raise RuntimeError("kvpack repetitions are absent, duplicated, or reordered")
        expected_route = (
            expected_source,
            expected_lookup,
            expected_prompt,
            expected_cut,
            expected_suffix,
        )
        if any(
            record.get("identity") != identity
            or (
                record.get("source"),
                record.get("lookup"),
                record.get("prompt_tokens"),
                record.get("published_cut"),
                record.get("suffix_tokens"),
            )
            != expected_route
            or record.get("matched_tokens") != expected_cut
            for record in samples
        ):
            raise RuntimeError("kvpack log contains a mixed or unplanned restore route")
        raw_ns = [record.get("restore_to_first_logits_ns") for record in samples]
        if not all(isinstance(value, int) and value > 0 for value in raw_ns):
            raise RuntimeError("kvpack restore nanoseconds are missing or non-positive")
        summary = summaries[0]
        if (
            summary.get("identity") != identity
            or (
                summary.get("source"),
                summary.get("lookup"),
                summary.get("prompt_tokens"),
                summary.get("published_cut"),
                summary.get("suffix_tokens"),
            )
            != expected_route
        ):
            raise RuntimeError("kvpack summary differs from its planned restore route")
        if summary.get("raw_restore_ns") != raw_ns or summary.get("seal_eligible") is not True:
            raise RuntimeError("kvpack summary failed correctness, overhead, or speed gates")
        full_recompute_ns = summary.get("full_recompute_ns")
        source_prefill_ns = summary.get("source_prefill_ns")
        publication_ns = summary.get("publication_ns")
        miss_lookup_ns = summary.get("miss_lookup_ns")
        if not all(
            isinstance(value, int) and value > 0
            for value in (
                full_recompute_ns,
                source_prefill_ns,
                publication_ns,
                miss_lookup_ns,
            )
        ):
            raise RuntimeError("kvpack summary timing components are absent or invalid")
        if any(record.get("full_recompute_ns") != full_recompute_ns for record in samples):
            raise RuntimeError("kvpack samples disagree with the full recompute timing")
        publication_ratio = summary.get("publication_overhead_ratio")
        miss_ratio = summary.get("miss_overhead_ratio")
        speedup = summary.get("speedup_geomean_cell")
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0
            for value in (publication_ratio, miss_ratio, speedup)
        ):
            raise RuntimeError("kvpack summary ratios are absent or invalid")
        mean_restore_ns = sum(raw_ns) / len(raw_ns)
        expected_ratios = (
            publication_ns / source_prefill_ns,
            miss_lookup_ns / full_recompute_ns,
            full_recompute_ns / mean_restore_ns,
        )
        if any(
            abs(actual - expected) > max(1e-12, abs(expected) * 1e-12)
            for actual, expected in zip(
                (publication_ratio, miss_ratio, speedup), expected_ratios
            )
        ):
            raise RuntimeError("kvpack summary ratios do not match raw timing components")
        if summary.get("correctness") != "exact-64-tokens-and-all-step-full-logit-digest":
            raise RuntimeError("kvpack summary lacks the exact correctness contract")
        if len({json.dumps(record.get("token_ids")) for record in samples}) != 1:
            raise RuntimeError("kvpack repetitions produced mixed token vectors")
        if len({record.get("full_logit_digest") for record in samples}) != 1:
            raise RuntimeError("kvpack repetitions produced mixed full-logit digests")
        token_ids = samples[0].get("token_ids")
        if (
            not isinstance(token_ids, list)
            or len(token_ids) != 64
            or not all(
                isinstance(token, int)
                and not isinstance(token, bool)
                and 0 <= token <= 0xFFFFFFFF
                for token in token_ids
            )
        ):
            raise RuntimeError("kvpack sample does not contain exactly 64 u32 tokens")
        token_digest = hashlib.sha256(
            b"".join(token.to_bytes(4, "little") for token in token_ids)
        ).hexdigest()
        logit_digest = samples[0].get("full_logit_digest")
        if (
            summary.get("generated_tokens_sha256") != token_digest
            or summary.get("full_logit_digest") != logit_digest
            or not isinstance(logit_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", logit_digest) is None
        ):
            raise RuntimeError("kvpack summary correctness digests do not match its samples")
        return {
            "raw_ns": raw_ns,
            "cv": coefficient_of_variation(raw_ns),
            "full_recompute_ns": full_recompute_ns,
            "source_prefill_ns": source_prefill_ns,
            "publication_ns": publication_ns,
            "publication_overhead_ratio": publication_ratio,
            "miss_lookup_ns": miss_lookup_ns,
            "miss_overhead_ratio": miss_ratio,
            "speedup_geomean_cell": speedup,
            "fingerprint": {
                "source": summary.get("source"),
                "lookup": summary.get("lookup"),
                "prompt_tokens": summary.get("prompt_tokens"),
                "published_cut": summary.get("published_cut"),
                "suffix_tokens": summary.get("suffix_tokens"),
                "generated_tokens_sha256": summary.get("generated_tokens_sha256"),
                "full_logit_digest": summary.get("full_logit_digest"),
            },
        }
    if engine == "dflash":
        if cell is None or identity is None:
            raise RuntimeError("DFlash normalization requires its planned cell and identity")
        depth, _variant = dflash_cell_shape(cell)
        samples = [
            record for record in records
            if record.get("schema") == "muser.dflash-qualify.v1"
            and record.get("kind") == "sample"
        ]
        summaries = [
            record for record in records
            if record.get("schema") == "muser.dflash-qualify.v1"
            and record.get("kind") == "summary"
        ]
        tuning = cell.startswith("dflash-tune-")
        expected_samples = 3 if tuning else 5
        if len(samples) != expected_samples or len(summaries) != 1:
            raise RuntimeError(
                f"DFlash log requires exactly {expected_samples} paired samples and one summary"
            )
        if [record.get("repetition") for record in samples] != list(range(expected_samples)):
            raise RuntimeError("DFlash repetitions are missing, duplicated, or reordered")
        expected_order = [
            ["dflash", "target-only"] if index % 4 in (1, 2)
            else ["target-only", "dflash"]
            for index in range(expected_samples)
        ]
        if [record.get("order") for record in samples] != expected_order:
            raise RuntimeError("DFlash A/B ordering is not the frozen ABBA order")
        if any(
            record.get("identity") != identity
            or record.get("prompt_tokens") != depth
            or record.get("output_tokens") != 256
            or record.get("verify_length") not in (3, 7, 15)
            or not isinstance(record.get("drafted_tokens"), int)
            or record.get("drafted_tokens", 0) <= 0
            or not isinstance(record.get("accepted_draft_tokens"), int)
            or not 0
            <= record.get("accepted_draft_tokens", -1)
            <= record.get("drafted_tokens", -1)
            or not isinstance(record.get("fallback_tokens"), int)
            or not 0 <= record.get("fallback_tokens", -1) <= 256
            for record in samples
        ):
            raise RuntimeError("DFlash samples contain a mixed identity, route, or geometry")
        if any(record.get("exact_target_match") is not True for record in samples):
            raise RuntimeError("DFlash output did not exactly match target-only")
        target_ns = [record.get("target_only_ns") for record in samples]
        dflash_ns = [record.get("dflash_ns") for record in samples]
        if not all(isinstance(value, int) and value > 0 for value in target_ns + dflash_ns):
            raise RuntimeError("DFlash paired raw nanoseconds are missing or non-positive")
        summary = summaries[0]
        if (
            summary.get("identity") != identity
            or summary.get("prompt_tokens") != depth
            or summary.get("output_tokens") != 256
            or summary.get("verify_length") not in (3, 7, 15)
            or any(
                record.get("verify_length") != summary.get("verify_length")
                for record in samples
            )
            or summary.get("target_only_raw_ns") != target_ns
            or summary.get("dflash_raw_ns") != dflash_ns
            or summary.get("exact_target_match") is not True
            or summary.get("warmup_policy")
            != "one-untimed-target-plus-dflash-pair-v1"
            or summary.get("measurement_order") != "abba-first-engine-v1"
        ):
            raise RuntimeError("DFlash summary differs from its paired samples")
        if (
            summary.get("sampled_scalar_oracle")
            != "muser-engine-scalar-full-distribution-v1"
            or summary.get("sampled_scalar_match") is not True
            or summary.get("sampled_tokens") != 32
            or not isinstance(summary.get("sampled_seed"), int)
            or summary.get("sampled_temperature_milli") != 800
            or summary.get("sampled_top_p_milli") != 950
            or summary.get("sampled_top_k") != 50
            or not isinstance(summary.get("sampled_generated_tokens_sha256"), str)
            or not isinstance(summary.get("sampled_drafted_tokens"), int)
            or summary.get("sampled_drafted_tokens", 0) <= 0
            or not isinstance(summary.get("sampled_accepted_draft_tokens"), int)
            or not isinstance(summary.get("sampled_fallback_tokens"), int)
        ):
            raise RuntimeError("DFlash sampled scalar verification is missing or incomplete")
        if len({record.get("generated_tokens_sha256") for record in samples}) != 1:
            raise RuntimeError("DFlash repetitions produced mixed token vectors")
        return {
            "raw_ns": dflash_ns,
            "target_only_raw_ns": target_ns,
            "cv": coefficient_of_variation(dflash_ns),
            "target_only_cv": coefficient_of_variation(target_ns),
            "speedups": [target / draft for target, draft in zip(target_ns, dflash_ns)],
            "measurement_order": expected_order,
            "fingerprint": {
                "prompt_tokens": summary.get("prompt_tokens"),
                "prompt_file_sha256": summary.get("prompt_file_sha256"),
                "output_tokens": summary.get("output_tokens"),
                "verify_length": summary.get("verify_length"),
                "target_backend": summary.get("target_backend"),
                "assistant_backend": summary.get("assistant_backend"),
                "generated_tokens_sha256": samples[0].get("generated_tokens_sha256"),
                "sampled_scalar_oracle": summary.get("sampled_scalar_oracle"),
                "sampled_tokens": summary.get("sampled_tokens"),
                "sampled_seed": summary.get("sampled_seed"),
                "sampled_temperature_milli": summary.get(
                    "sampled_temperature_milli"
                ),
                "sampled_top_p_milli": summary.get("sampled_top_p_milli"),
                "sampled_top_k": summary.get("sampled_top_k"),
                "sampled_generated_tokens_sha256": summary.get(
                    "sampled_generated_tokens_sha256"
                ),
                "sampled_drafted_tokens": summary.get("sampled_drafted_tokens"),
            },
        }
    if engine == "llama-dflash":
        if cell is None or identity is None:
            raise RuntimeError(
                "llama DFlash normalization requires its planned cell and identity"
            )
        depth, _variant = dflash_cell_shape(cell)
        samples = [
            record for record in records
            if record.get("schema") == "muser.llama-dflash.v1"
            and record.get("kind") == "sample"
        ]
        summaries = [
            record for record in records
            if record.get("schema") == "muser.llama-dflash.v1"
            and record.get("kind") == "summary"
        ]
        if len(samples) != 5 or len(summaries) != 1:
            raise RuntimeError("llama DFlash log requires five samples and one summary")
        if [record.get("repetition") for record in samples] != list(range(5)):
            raise RuntimeError("llama DFlash repetitions are missing, duplicated, or reordered")
        if any(
            record.get("identity") != identity
            or record.get("depth") != depth
            or record.get("output_tokens") != 256
            or record.get("verify_length") not in (3, 7, 15)
            or not isinstance(record.get("drafted_tokens"), int)
            or record.get("drafted_tokens", 0) <= 0
            for record in samples
        ):
            raise RuntimeError("llama DFlash samples contain a mixed identity, route, or geometry")
        raw_ns = [record.get("elapsed_ns") for record in samples]
        if not all(isinstance(value, int) and value > 0 for value in raw_ns):
            raise RuntimeError("llama DFlash raw decode nanoseconds are missing")
        summary = summaries[0]
        if (
            summary.get("identity") != identity
            or summary.get("depth") != depth
            or summary.get("output_tokens") != 256
            or summary.get("verify_length") not in (3, 7, 15)
            or any(
                record.get("verify_length") != summary.get("verify_length")
                for record in samples
            )
            or summary.get("raw_ns") != raw_ns
            or summary.get("seal_eligible") is not True
        ):
            raise RuntimeError("llama DFlash summary differs from its samples")
        digests = {record.get("generated_tokens_sha256") for record in samples}
        if len(digests) != 1 or None in digests:
            raise RuntimeError("llama DFlash repetitions produced mixed token vectors")
        return {
            "raw_ns": raw_ns,
            "cv": coefficient_of_variation(raw_ns),
            "fingerprint": {
                "prompt_tokens": summary.get("depth"),
                "prompt_file_sha256": summary.get("prompt_file_sha256"),
                "output_tokens": summary.get("output_tokens"),
                "verify_length": summary.get("verify_length"),
                "generated_tokens_sha256": next(iter(digests)),
                "route": "llama-draft-dflash",
            },
        }
    if engine == "vision":
        samples = [
            record for record in records
            if record.get("schema") == "muser.vision-qualify.v1"
            and record.get("kind") == "sample"
        ]
        summaries = [
            record for record in records
            if record.get("schema") == "muser.vision-qualify.v1"
            and record.get("kind") == "summary"
        ]
        if len(samples) != 3 or len(summaries) != 1:
            raise RuntimeError("vision log requires exactly three samples and one summary")
        raw_ns = [record.get("elapsed_ns") for record in samples]
        if not all(isinstance(value, int) and value > 0 for value in raw_ns):
            raise RuntimeError("vision raw nanoseconds are missing or non-positive")
        summary = summaries[0]
        if summary.get("raw_ns") != raw_ns:
            raise RuntimeError("vision summary differs from its samples")
        insertion_start = summary.get("insertion_start")
        insertion_end = summary.get("insertion_end")
        insertion_count = summary.get("insertion_count")
        projected_tokens = summary.get("projected_tokens")
        suffix_tokens = summary.get("suffix_tokens")
        installed_positions = summary.get("installed_positions")
        if (
            summary.get("max_pixel_error", 1.0) > 1 / 255
            or summary.get("embedding_cosine", 0.0) < 0.999
            or summary.get("embedding_relative_l2", 1.0) > 0.01
            or summary.get("exact_decoder_tokens") is not True
            or summary.get("output_tokens") != 64
            or not all(
                isinstance(value, int) and value >= 0
                for value in (
                    insertion_start, insertion_end, insertion_count, projected_tokens,
                    suffix_tokens, installed_positions,
                )
            )
            or insertion_start != summary.get("prefix_tokens")
            or insertion_end - insertion_start != insertion_count
            or insertion_count != projected_tokens
            or installed_positions != insertion_end + suffix_tokens
            or summary.get("insertion_positions_sha256")
            != position_range_sha256(insertion_start, insertion_end)
        ):
            raise RuntimeError("vision correctness thresholds did not pass")
        if summary.get("route") != "mtmd-metal:muser-mtmd-muse-vision-v1":
            raise RuntimeError("vision qualification used an unexpected route")
        return {
            "raw_ns": raw_ns,
            "cv": coefficient_of_variation(raw_ns),
            "fingerprint": {
                key: summary.get(key) for key in [
                    "fixture", "route", "target_backend", "image_sha256",
                    "preprocessing_sha256", "upstream_preprocessing_sha256",
                    "cpu_embeddings_sha256", "accelerated_embeddings_sha256",
                    "decoder_tokens_sha256", "source_width", "source_height",
                    "width", "height", "projected_tokens",
                    "max_pixel_error", "embedding_cosine", "embedding_relative_l2",
                    "exact_decoder_tokens", "insertion_start", "insertion_end",
                    "insertion_count", "insertion_positions_sha256", "prefix_tokens",
                    "suffix_tokens", "installed_positions",
                ]
            },
        }
    if engine == "ane":
        samples = [
            record for record in records
            if record.get("schema") == "muser.ane-qualify.v1"
            and record.get("kind") == "sample"
        ]
        summaries = [
            record for record in records
            if record.get("schema") == "muser.ane-qualify.v1"
            and record.get("kind") == "summary"
        ]
        if len(samples) != 3 or len(summaries) != 1:
            raise RuntimeError("ANE log requires exactly three paired samples and one summary")
        if any(record.get("exact_target_match") is not True for record in samples):
            raise RuntimeError("ANE or Metal DFlash output differs from target-only")
        target_ns = [record.get("target_only_ns") for record in samples]
        metal_ns = [record.get("metal_dflash_ns") for record in samples]
        ane_ns = [record.get("ane_dflash_ns") for record in samples]
        verify_taxes = [record.get("target_verification_tax") for record in samples]
        metal_verify_ns = [record.get("metal_target_verify_ns") for record in samples]
        ane_verify_ns = [record.get("ane_target_verify_ns") for record in samples]
        phase_fields = (
            "metal_prefill_ns", "ane_prefill_ns", "metal_draft_ns", "ane_draft_ns",
            "metal_fallback_target_ns", "ane_fallback_target_ns", "metal_rounds",
            "ane_rounds", "metal_drafted_tokens", "ane_drafted_tokens",
            "metal_accepted_draft_tokens", "ane_accepted_draft_tokens",
            "ane_mirror_capture_fc_ns",
        )
        phase_values = {
            field: [record.get(field) for record in samples] for field in phase_fields
        }
        if not all(
            isinstance(value, int) and value > 0 for value in target_ns + metal_ns + ane_ns
        ) or not all(
            isinstance(value, int) and value > 0 for value in metal_verify_ns + ane_verify_ns
        ) or not all(isinstance(value, (int, float)) for value in verify_taxes) or any(
            not all(isinstance(value, int) and value >= 0 for value in values)
            for values in phase_values.values()
        ) or any(
            value <= 0
            for field in ("metal_rounds", "ane_rounds")
            for value in phase_values[field]
        ):
            raise RuntimeError("ANE paired timings are absent or invalid")
        summary = summaries[0]
        if (
            summary.get("target_only_raw_ns") != target_ns
            or summary.get("metal_dflash_raw_ns") != metal_ns
            or summary.get("ane_dflash_raw_ns") != ane_ns
            or summary.get("target_verification_taxes") != verify_taxes
            or summary.get("compute_units") != "CPU_AND_NE"
            or summary.get("exact_target_match") is not True
        ):
            raise RuntimeError("ANE summary differs from its exact paired samples")
        if len({record.get("generated_tokens_sha256") for record in samples}) != 1:
            raise RuntimeError("ANE repetitions produced mixed token vectors")
        return {
            "raw_ns": ane_ns,
            "metal_dflash_raw_ns": metal_ns,
            "target_only_raw_ns": target_ns,
            "cv": coefficient_of_variation(ane_ns),
            "metal_dflash_cv": coefficient_of_variation(metal_ns),
            "target_only_cv": coefficient_of_variation(target_ns),
            "speedups": [metal / ane for metal, ane in zip(metal_ns, ane_ns)],
            "verification_taxes": verify_taxes,
            "metal_target_verify_ns": metal_verify_ns,
            "ane_target_verify_ns": ane_verify_ns,
            "phase_timings": phase_values,
            "fingerprint": {
                key: summary.get(key) for key in [
                    "prompt_tokens", "prompt_file_sha256", "output_tokens",
                    "verify_length", "target_identity", "dflash_identity",
                    "manifest_sha256", "compute_plan_receipt_sha256", "compute_units",
                ]
            },
        }
    if engine == "remote":
        if cell is None or identity is None:
            raise RuntimeError("remote normalization requires its planned cell and identity")
        shape = re.fullmatch(r"remote-(text|multimodal|target-plus-dflash)-(\d+)", cell)
        if shape is None:
            raise RuntimeError(f"invalid remote cell name: {cell}")
        expected_variant, depth_text = shape.groups()
        depth = int(depth_text)
        expected_output_tokens = 48 if depth == 131008 else 256
        samples = [
            record for record in records
            if record.get("schema") == "muser.remote-qualify.v1"
            and record.get("kind") == "sample"
        ]
        summaries = [
            record for record in records
            if record.get("schema") == "muser.remote-qualify.v1"
            and record.get("kind") == "summary"
        ]
        if len(samples) != 3 or len(summaries) != 1:
            raise RuntimeError("remote log requires exactly three paired samples and one summary")
        if [record.get("repetition") for record in samples] != [0, 1, 2]:
            raise RuntimeError("remote repetitions are absent, duplicated, or reordered")
        if [record.get("order") for record in samples] != [
            ["local", "remote"],
            ["remote", "local"],
            ["remote", "local"],
        ]:
            raise RuntimeError("remote paired timing is not in frozen ABBA order")
        if any(
            record.get("exact_tokens") is not True
            or record.get("exact_full_logits") is not True
            for record in samples
        ):
            raise RuntimeError("remote output or full logits differ from local")
        if any(
            record.get("identity") != identity
            or record.get("variant") != expected_variant
            or record.get("prompt_positions") != depth
            or record.get("output_tokens") != expected_output_tokens
            for record in samples
        ):
            raise RuntimeError("remote samples contain a mixed identity, route, or geometry")
        local = [record.get("local_ttft_ns") for record in samples]
        remote = [record.get("remote_ttft_ns") for record in samples]
        local_decode = [record.get("local_first_64_decode_ns") for record in samples]
        remote_decode = [record.get("remote_first_64_decode_ns") for record in samples]
        if not all(
            isinstance(value, int) and value > 0
            for value in local + remote + local_decode + remote_decode
        ):
            raise RuntimeError("remote paired timings are absent or invalid")
        summary = summaries[0]
        if (
            summary.get("local_ttft_raw_ns") != local
            or summary.get("remote_ttft_raw_ns") != remote
            or summary.get("exact_remote_local") is not True
            or summary.get("output_tokens") != expected_output_tokens
            or summary.get("identity") != identity
            or summary.get("prompt_positions") != depth
            or summary.get("stable") is not True
        ):
            raise RuntimeError("remote summary differs from paired samples")
        variants = {record.get("variant") for record in samples}
        if variants != {summary.get("variant")}:
            raise RuntimeError("remote log contains mixed variants")
        if len({record.get("generated_tokens_sha256") for record in samples}) != 1:
            raise RuntimeError("remote repetitions produced mixed token vectors")
        if len({record.get("full_logit_digest") for record in samples}) != 1:
            raise RuntimeError("remote repetitions produced mixed full-logit digests")
        token_digest = samples[0].get("generated_tokens_sha256")
        logit_digest = samples[0].get("full_logit_digest")
        if (
            not isinstance(token_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", token_digest) is None
            or not isinstance(logit_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", logit_digest) is None
            or summary.get("generated_tokens_sha256") != token_digest
            or summary.get("full_logit_digest") != logit_digest
        ):
            raise RuntimeError("remote correctness digests are malformed or unbound")
        producer_fields = (
            "producer_export_overhead_ratio",
            "producer_first_tile_prefill_fraction",
            "producer_transfer_hidden_ratio",
        )
        if any(
            not isinstance(record.get(field), (int, float))
            for record in samples for field in producer_fields
        ):
            raise RuntimeError("remote sample lacks producer overlap receipt")
        variant = summary.get("variant")
        installed_bytes = [record.get("installed_bytes") for record in samples]
        installed_segments = [record.get("installed_segments") for record in samples]
        transfer_ns = [record.get("receiver_transfer_commit_ns") for record in samples]
        payload_wire_ns = [record.get("producer_payload_wire_ns") for record in samples]
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in installed_bytes
            + installed_segments
            + transfer_ns
            + payload_wire_ns
        ):
            raise RuntimeError("remote installed payload or phase timing is absent")
        if any(
            not isinstance(record.get("producer_payload_bytes"), int)
            or isinstance(record.get("producer_payload_bytes"), bool)
            or record.get("producer_payload_bytes") != record.get("installed_bytes")
            for record in samples
        ):
            raise RuntimeError("producer payload bytes do not bind the installed transfer")
        component_fields = (
            "target_installed_bytes", "target_installed_segments",
            "dflash_installed_bytes", "dflash_installed_segments",
        )
        if any(
            not isinstance(record.get(field), int)
            or isinstance(record.get(field), bool)
            or record.get(field) < 0
            for record in samples for field in component_fields
        ):
            raise RuntimeError("remote component install counters are absent")
        if any(
            record.get("target_prepared") is not True
            or record.get("target_installed") is not True
            or record.get("target_installed_bytes", 0) <= 0
            or record.get("target_installed_segments", 0) <= 0
            or record["target_installed_bytes"] + record["dflash_installed_bytes"]
            != record["installed_bytes"]
            or record["target_installed_segments"] + record["dflash_installed_segments"]
            != record["installed_segments"]
            for record in samples
        ):
            raise RuntimeError("remote target component was not completely installed")
        if variant == "target-plus-dflash":
            if any(
                record.get("exact_dflash_tokens") is not True
                or record.get("exact_dflash_trace") is not True
                or record.get("dflash_prepared") is not True
                or record.get("dflash_installed") is not True
                or record.get("dflash_installed_bytes", 0) <= 0
                or record.get("dflash_installed_segments", 0) <= 0
                or not isinstance(record.get("local_dflash_acceptance"), (int, float))
                or not isinstance(record.get("remote_dflash_acceptance"), (int, float))
                or not isinstance(record.get("remote_dflash_acceptance_ratio"), (int, float))
                or re.fullmatch(
                    r"[0-9a-f]{64}", str(record.get("dflash_draft_trace_sha256"))
                ) is None
                or re.fullmatch(
                    r"[0-9a-f]{64}",
                    str(record.get("dflash_accepted_prefix_trace_sha256")),
                ) is None
                or not isinstance(record.get("dflash_accepted_prefix_counts"), list)
                or not record.get("dflash_accepted_prefix_counts")
                or not all(
                    isinstance(value, int) and not isinstance(value, bool) and value >= 0
                    for value in record.get("dflash_accepted_prefix_counts", [])
                )
                for record in samples
            ):
                raise RuntimeError("combined remote sample lacks exact DFlash evidence")
            trace_identity = (
                samples[0]["dflash_draft_trace_sha256"],
                samples[0]["dflash_accepted_prefix_trace_sha256"],
                samples[0]["dflash_accepted_prefix_counts"],
            )
            if any(
                (
                    record["dflash_draft_trace_sha256"],
                    record["dflash_accepted_prefix_trace_sha256"],
                    record["dflash_accepted_prefix_counts"],
                ) != trace_identity
                for record in samples[1:]
            ):
                raise RuntimeError("DFlash assistant trace changed between repetitions")
            acceptance = [record["remote_dflash_acceptance"] for record in samples]
            if any(value < 0.95 for value in acceptance):
                raise RuntimeError("remote DFlash acceptance is below 95%")
            if (
                summary.get("remote_dflash_acceptance") != acceptance
                or summary.get("remote_dflash_acceptance_minimum") != min(acceptance)
                or summary.get("remote_dflash_acceptance_required") != 0.95
            ):
                raise RuntimeError("remote summary does not bind DFlash acceptance")
        elif any(
            record.get("exact_dflash_tokens") is not None
            or record.get("exact_dflash_trace") is not None
            for record in samples
        ):
            raise RuntimeError("non-DFlash remote variant contains DFlash evidence")
        elif any(
            record.get("dflash_prepared") is not False
            or record.get("dflash_installed") is not False
            or record.get("dflash_installed_bytes") != 0
            or record.get("dflash_installed_segments") != 0
            for record in samples
        ):
            raise RuntimeError("non-DFlash remote variant installed a DFlash component")
        throughputs = [
            byte_count * 8.0 / elapsed
            for byte_count, elapsed in zip(installed_bytes, payload_wire_ns)
        ]
        if any(
            not isinstance(record.get("installed_payload_gbps"), (int, float))
            or isinstance(record.get("installed_payload_gbps"), bool)
            or not math.isclose(
                record["installed_payload_gbps"], expected, rel_tol=1e-12, abs_tol=0.0
            )
            for record, expected in zip(samples, throughputs)
        ):
            raise RuntimeError("remote link throughput is not derived from installed payload")
        link_cv = coefficient_of_variation(throughputs)
        link_median = sorted(throughputs)[len(throughputs) // 2]
        if (
            summary.get("installed_payload_gbps") != throughputs
            or not math.isclose(
                summary.get("installed_payload_gbps_cv", math.inf),
                link_cv,
                rel_tol=1e-12,
                abs_tol=0.0,
            )
            or not math.isclose(
                summary.get("installed_payload_gbps_median", math.inf),
                link_median,
                rel_tol=1e-12,
                abs_tol=0.0,
            )
            or summary.get("installed_payload_gbps_minimum") != 3.0
        ):
            raise RuntimeError("remote summary does not bind link throughput evidence")
        return {
            "raw_ns": remote,
            "local_ttft_raw_ns": local,
            "cv": coefficient_of_variation(remote),
            "local_ttft_cv": coefficient_of_variation(local),
            "ttft_speedups": [left / right for left, right in zip(local, remote)],
            "local_first_64_decode_ns": local_decode,
            "remote_first_64_decode_ns": remote_decode,
            "decode_ratios": [right / left for left, right in zip(local_decode, remote_decode)],
            "producer_export_overhead_ratios": [
                record["producer_export_overhead_ratio"] for record in samples
            ],
            "producer_first_tile_prefill_fractions": [
                record["producer_first_tile_prefill_fraction"] for record in samples
            ],
            "producer_transfer_hidden_ratios": [
                record["producer_transfer_hidden_ratio"] for record in samples
            ],
            "dflash_acceptance_ratios": [
                record.get("remote_dflash_acceptance_ratio") for record in samples
            ] if variant == "target-plus-dflash" else None,
            "dflash_acceptance": [
                record.get("remote_dflash_acceptance") for record in samples
            ] if variant == "target-plus-dflash" else None,
            "installed_payload_gbps": throughputs,
            "installed_payload_gbps_cv": link_cv,
            "fingerprint": {
                "variant": variant,
                "prompt_positions": summary.get("prompt_positions"),
                "prompt_file_sha256": summary.get("prompt_file_sha256"),
                "output_tokens": summary.get("output_tokens"),
                "generated_tokens_sha256": samples[0].get("generated_tokens_sha256"),
                "full_logit_digest": samples[0].get("full_logit_digest"),
                "installed_bytes": installed_bytes,
                "installed_segments": installed_segments,
                "dflash_draft_trace_sha256": samples[0].get("dflash_draft_trace_sha256"),
                "dflash_accepted_prefix_trace_sha256": samples[0].get(
                    "dflash_accepted_prefix_trace_sha256"
                ),
                "dflash_accepted_prefix_counts": samples[0].get(
                    "dflash_accepted_prefix_counts"
                ),
            },
        }
    if engine in ("vision-ttft-muser", "vision-ttft-llama"):
        expected_engine = engine.removeprefix("vision-ttft-")
        expected_samples = 3 if expected_engine == "muser" else 5
        samples = [
            record for record in records
            if record.get("schema") == "muser.vision-server-ttft.v1"
            and record.get("kind") == "sample"
        ]
        summaries = [
            record for record in records
            if record.get("schema") == "muser.vision-server-ttft.v1"
            and record.get("kind") == "summary"
        ]
        if len(samples) != expected_samples or len(summaries) != 1:
            raise RuntimeError("vision TTFT log has the wrong sample or summary count")
        if [record.get("repetition") for record in samples] != list(range(expected_samples)):
            raise RuntimeError("vision TTFT repetitions are absent, duplicated, or reordered")
        raw_ns = [record.get("elapsed_ns") for record in samples]
        summary = summaries[0]
        if (
            not all(isinstance(value, int) and value > 0 for value in raw_ns)
            or summary.get("raw_ns") != raw_ns
            or summary.get("engine") != expected_engine
            or summary.get("seal_eligible") is not True
            or summary.get("server_lifecycle")
            != "leased-start-ready-exact-requests-cooperative-exit"
        ):
            raise RuntimeError("vision TTFT summary is invalid or mixed-route")
        return {
            "raw_ns": raw_ns,
            "cv": coefficient_of_variation(raw_ns),
            "fingerprint": {
                "engine": expected_engine,
                "fixture": summary.get("fixture"),
                "image_sha256": summary.get("image_sha256"),
                "first_content_digests": summary.get("first_content_digests"),
                "reported_prompt_tokens": summary.get("reported_prompt_tokens"),
                "server_lifecycle": summary.get("server_lifecycle"),
                "cache": "disabled",
            },
        }
    if engine == "muser":
        if cell is None or identity is None:
            raise RuntimeError("Muser normalization requires its planned cell and identity")
        surface, depth = baseline_cell_shape(cell)
        if surface == "ttft":
            raise RuntimeError(f"synthetic Muser evidence attached to TTFT cell {cell}")
        samples = [record for record in records if record.get("kind") == "sample"]
        summaries = [record for record in records if record.get("kind") == "summary"]
        if len(samples) != 5 or len(summaries) != 1:
            raise RuntimeError("Muser log lacks exactly 5 samples and 1 summary")
        if [record.get("repetition") for record in samples] != list(range(5)):
            raise RuntimeError("Muser repetitions are absent, duplicated, or reordered")
        expected_tokens = depth if surface == "prefill" else 64
        if any(
            record.get("schema") != "muser-bench.v1"
            or record.get("surface") != surface
            or record.get("depth") != depth
            or record.get("measured_tokens") != expected_tokens
            for record in samples
        ):
            raise RuntimeError("Muser evidence contradicts its planned surface or depth")
        raw_ns = [record.get("elapsed_ns") for record in samples]
        if not all(isinstance(value, int) and value > 0 for value in raw_ns):
            raise RuntimeError("Muser raw nanoseconds are missing or non-positive")
        if (
            summaries[0].get("raw_ns") != raw_ns
            or summaries[0].get("surface") != surface
            or summaries[0].get("depth") != depth
        ):
            raise RuntimeError("Muser summary raw_ns does not match samples")
        if len({record.get("token_digest") for record in samples}) != 1:
            raise RuntimeError("Muser repetitions produced mixed results")
        fingerprints = [record.get("fingerprint") for record in samples]
        if any(value != fingerprints[0] for value in fingerprints[1:]):
            raise RuntimeError("Muser repetitions used mixed fingerprints")
        fingerprint = fingerprints[0]
        if not isinstance(fingerprint, dict):
            raise RuntimeError("Muser fingerprint is absent")
        required = {
            "identity": identity,
            "backend": "metal-reference",
            "kv": "f16",
            "flash_attention_requested": "on",
            "flash_attention_active": True,
            "warmup_policy": (
                "full-logical-prompt-once-before-timing-v1"
                if surface == "prefill"
                else "full-teacher-block-before-timing-v1"
            ),
        }
        if any(fingerprint.get(key) != value for key, value in required.items()):
            raise RuntimeError(f"Muser route fingerprint contradicts request: {fingerprint}")
        return {
            "raw_ns": raw_ns,
            "cv": coefficient_of_variation(raw_ns),
            "fingerprint": fingerprint,
        }

    if cell is None:
        raise RuntimeError("llama normalization requires its planned cell")
    surface, depth = baseline_cell_shape(cell)
    if surface == "ttft":
        raise RuntimeError(f"synthetic llama evidence attached to TTFT cell {cell}")
    candidates = [record for record in records if isinstance(record.get("samples_ns"), list)]
    if len(candidates) != 1:
        raise RuntimeError("llama log lacks exactly one JSONL result")
    record = candidates[0]
    raw_ns = record["samples_ns"]
    if len(raw_ns) != 5 or not all(isinstance(value, int) and value > 0 for value in raw_ns):
        raise RuntimeError("llama result lacks exactly 5 positive raw samples_ns")
    expected = {
        "n_threads": 20,
        "n_batch": 2048,
        "n_ubatch": 512,
        "n_gpu_layers": 99,
        "type_k": "f16",
        "type_v": "f16",
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"llama route fields contradict request: {record}")
    if record.get("flash_attn") not in (1, True):
        raise RuntimeError("llama FlashAttention was not active")
    expected_warmup = (
        "full_logical_prompt_once_v1"
        if surface == "prefill"
        else "decode_depth_construct_once_not_warmup_v1"
    )
    if record.get("prefill_warmup_policy") != expected_warmup or record.get("no_warmup") is not False:
        raise RuntimeError("llama evidence lacks the canonical warmup")
    expected_shape = (
        {"n_prompt": depth, "n_depth": 0, "n_gen": 0}
        if surface == "prefill"
        else {"n_prompt": 0, "n_depth": depth, "n_gen": 64}
    )
    if any(record.get(key) != value for key, value in expected_shape.items()):
        raise RuntimeError(f"llama evidence contradicts its planned surface or depth: {record}")
    if (
        not re.fullmatch(r"[0-9a-f]{40}", str(record.get("comparator_upstream_commit")))
        or record.get("comparator_patch_sha256")
        != sha256(ROOT / "scripts" / "llama_bench_fixture.patch")
    ):
        raise RuntimeError("llama evidence has the wrong comparator identity")
    return {
        "raw_ns": raw_ns,
        "cv": coefficient_of_variation(raw_ns),
        "fingerprint": {
            key: record.get(key)
            for key in [
                "build_commit", "build_number", "comparator_upstream_commit",
                "comparator_patch_sha256", "n_batch", "n_ubatch", "n_threads",
                "n_gpu_layers", "type_k", "type_v", "flash_attn",
                "prompt_fixture_file_sha256", "prompt_tokens_sha256",
                "decode_fixture_file_sha256", "decode_tokens_sha256",
                "workload_sha256",
                "prefill_warmup_policy", "decode_warmup_policy",
                "decode_repetition_policy", "no_warmup",
            ]
        },
    }


def validate_correctness_receipts(
    numerical_path: Path | None, greedy_path: Path | None, identity: str
) -> None:
    path = numerical_path
    if path is None:
        raise RuntimeError("--full requires --correctness-receipt")
    receipt = json.loads(path.read_text())
    if receipt.get("schema") != "muser.correctness.receipt.v1":
        raise RuntimeError("wrong correctness receipt schema")
    if receipt.get("status") != "passed" or receipt.get("identity") != identity:
        raise RuntimeError("correctness receipt is not a pass for this exact identity")
    if greedy_path is None:
        raise RuntimeError("--full requires --greedy-correctness-receipt")
    greedy = json.loads(greedy_path.read_text())
    if (
        greedy.get("schema") != "muser.greedy-correctness.receipt.v1"
        or greedy.get("status") != "passed"
        or greedy.get("seal_eligible") is not True
        or greedy.get("identity") != identity
        or greedy.get("exact_cases") != 11
        or greedy.get("snapshot_replay_depths")
        != [8192, 16384, 32768, 65536, 131008]
        or greedy.get("ring_import_cuts")
        != [2047, 2048, 2049, 2559, 2560, 2561]
        or not isinstance(greedy.get("ring_import_log_sha256"), str)
    ):
        raise RuntimeError("greedy correctness receipt is not a complete pass for this identity")


def validate_baseline_seal(path: Path | None, identity: str) -> None:
    if path is None:
        raise RuntimeError("--kvpack-full requires --baseline-seal")
    seal = json.loads(path.read_text())
    if (
        seal.get("schema") != "muser.baseline.seal.v1"
        or seal.get("status") != "passed"
        or seal.get("identity") != identity
    ):
        raise RuntimeError("baseline seal is not a pass for this exact identity")


def validate_stage_seal(
    path: Path | None, identity: str, schema: str, option: str
) -> None:
    if path is None:
        raise RuntimeError(f"{option} is required by the strict stage order")
    seal = json.loads(path.read_text())
    if (
        seal.get("schema") != schema
        or seal.get("status") != "passed"
        or seal.get("identity") != identity
    ):
        raise RuntimeError(f"{option} is not a pass for this exact identity")


def frozen_dflash_verify_length(path: Path | None) -> int:
    if path is None:
        raise RuntimeError("--dflash-full requires --dflash-tuning-freeze")
    expected = (ROOT / "release/dflash-tuning-v1.json").resolve()
    if path.resolve() != expected or not expected.is_file() or expected.is_symlink():
        raise RuntimeError("DFlash tuning must use tracked release/dflash-tuning-v1.json")
    receipt = json.loads(path.read_text())
    selected = receipt.get("selected_verify_length")
    if (
        receipt.get("schema") != "muser.dflash-tuning-freeze.v1"
        or receipt.get("status") != "frozen"
        or selected not in (3, 7, 15)
    ):
        raise RuntimeError("tracked DFlash tuning selection is not frozen")
    return selected


def wrapped_execute(command: list[str]) -> list[str]:
    result = list(command)
    if len(result) < 2 or Path(result[1]).name != "accelerator_safe.py":
        raise RuntimeError("matrix command bypasses accelerator_safe.py")
    result.insert(2, "--execute")
    return result


def command_engine(command: list[str]) -> str:
    child_index = command.index("--") + 1
    child = Path(command[child_index]).name
    names = [Path(token).name for token in command]
    if child == "llama-bench":
        return "llama"
    if child == "muser-kvpack-qualify":
        return "kvpack"
    if child == "muser-dflash-qualify":
        return "dflash"
    if child == "muser-vision-qualify":
        return "vision"
    if child == "muser-ane-qualify":
        return "ane"
    if child == "muser-remote-qualify":
        return "remote"
    if "bench_vision_server.py" in names:
        return "vision-ttft-" + command[command.index("--engine") + 1]
    if "bench_llama_dflash.py" in names:
        return "llama-dflash"
    if "bench_server_ttft.py" in names or "--engine" in command:
        return "ttft-" + command[command.index("--engine") + 1]
    return "muser"


def execute_plan(
    args: argparse.Namespace,
    identity: dict[str, Any],
    commands: list[tuple[str, str, list[str]]],
) -> int:
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = out_dir / f"campaign-{args.run_id}.jsonl"
    existing = read_records(ledger)
    completed = {
        record.get("plan_digest")
        for record in existing
        if record.get("status") == "passed"
        and record.get("identity") == identity["digest"]
    }
    for cell, engine, command in commands:
        digest = plan_digest(command, identity["digest"])
        if digest in completed:
            print(f"resume: {cell}/{engine} already passed")
            continue
        started = dt.datetime.now(dt.timezone.utc).isoformat()
        result_receipt = out_dir / f"accelerator-{args.run_id}-{digest}.result.json"
        executing_command = wrapped_execute(command)
        separator = executing_command.index("--")
        executing_command[separator:separator] = ["--result-receipt", str(result_receipt)]
        result = subprocess.run(
            executing_command, cwd=ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        base = {
            "schema": "muser.campaign.cell.v1",
            "run_id": args.run_id,
            "identity": identity["digest"],
            "cell": cell,
            "engine": engine,
            "plan_digest": digest,
            "started_at": started,
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "wrapper_stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
        }
        try:
            retained, log = retained_accelerator_result(
                result_receipt, executing_command, result.returncode
            )
            if retained["exit_status"] != 0:
                raise RuntimeError("accelerator child failed according to its retained receipt")
            normalized = normalize_evidence(engine, log, cell, identity["digest"])
        except Exception as error:
            append_record(ledger, {**base, "status": "failed", "error": str(error)})
            print(f"{cell}/{engine}: {error}", file=sys.stderr)
            return 1
        append_record(
            ledger,
            {
                **base,
                "status": "passed",
                "log_sha256": sha256(log),
                **normalized,
            },
        )
        print(f"passed: {cell}/{engine} cv={normalized['cv']:.6f}")
    return 0


def retained_accelerator_result(
    receipt_path: Path, wrapper_command: list[str], process_exit_status: int
) -> tuple[dict[str, Any], Path]:
    if not receipt_path.is_absolute():
        receipt_path = ROOT / receipt_path
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise RuntimeError("accelerator result receipt is not a retained regular file")
    receipt = json.loads(receipt_path.read_text())
    expected_keys = {
        "schema", "run_id", "identity", "cell", "command", "exit_status",
        "command_log", "started_at", "finished_at",
    }
    if set(receipt) != expected_keys or receipt.get("schema") != "muser.accelerator-result.v1":
        raise RuntimeError("accelerator result receipt has an unknown or incomplete shape")
    separator = wrapper_command.index("--")
    expected_child = wrapper_command[separator + 1 :]
    expected_identity = wrapper_command[wrapper_command.index("--identity") + 1]
    expected_cell = wrapper_command[wrapper_command.index("--cell") + 1]
    if (
        receipt.get("identity") != expected_identity
        or receipt.get("cell") != expected_cell
        or receipt.get("command") != expected_child
        or receipt.get("exit_status") != process_exit_status
    ):
        raise RuntimeError("accelerator result receipt is bound to a different run")
    run_id = receipt.get("run_id")
    if not isinstance(run_id, str) or re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{32}", run_id) is None:
        raise RuntimeError("accelerator result receipt has an invalid run id")
    log = Path(str(receipt.get("command_log", "")))
    if (
        not log.is_absolute()
        or log.is_symlink()
        or not log.is_file()
        or log.name != f"{run_id}.command.log"
        or log.parent.resolve() != receipt_path.parent.resolve()
    ):
        raise RuntimeError("accelerator result command log is not bound to its run")
    return receipt, log


def main() -> int:
    args = parse_args()
    # Campaigns collect raw/unsealed evidence in every lock state. They have
    # no authority to publish a seal; only atomic_seal_campaign.py does.
    load_release_lock()
    if args.ggml_metallib is not None:
        os.environ["MUSER_GGML_METALLIB"] = str(args.ggml_metallib.resolve())
    if not RUN_ID.fullmatch(args.run_id):
        raise SystemExit("unsafe --run-id")
    if not 0 <= args.bos_token < args.vocab_size:
        raise SystemExit("--bos-token must be inside --vocab-size")
    identity, blockers = static_identity(args)
    margs = matrix_args(args, identity["digest"])
    baseline_cells = release_matrix.cells(margs)
    dflash_tune_cells = release_matrix.dflash_tune_cells(margs)
    dflash_cells = release_matrix.dflash_cells(margs)
    vision_cells = release_matrix.vision_cells(margs)
    ane_cells = release_matrix.ane_cells(margs)
    remote_cells = release_matrix.remote_cells(margs)
    ttft_cells = release_matrix.ttft_cells(margs)
    depths = parse_diagnostic_depths(args.depths)
    diagnostic_dflash = args.dflash_full or (
        args.dry_run and args.dry_run_stage == "dflash"
    )
    if depths and not args.smoke and not diagnostic_dflash and not (
        args.dry_run and args.dry_run_stage == "smoke"
    ):
        raise SystemExit("--depths is only valid with --smoke or --dflash-full")
    smoke_cells = diagnostic_baseline_cells(baseline_cells, ttft_cells, depths)
    dflash_probe_cells = diagnostic_dflash_cells(dflash_cells, depths)
    cells = (
        baseline_cells
        + ttft_cells
        + release_matrix.kvpack_cells(margs)
        + dflash_tune_cells
        + dflash_cells
        + vision_cells
        + remote_cells
    )
    if args.dry_run:
        dry_run_cells = {
            "complete": cells,
            "smoke": smoke_cells,
            "baseline": baseline_cells + ttft_cells,
            "kvpack": release_matrix.kvpack_cells(margs),
            "vision": vision_cells,
            "dflash": (
                dflash_probe_cells if depths else dflash_tune_cells + dflash_cells
            ),
            "ane": ane_cells,
            "remote": remote_cells,
        }
        planned_cells = dry_run_cells[args.dry_run_stage]
    elif args.smoke:
        planned_cells = smoke_cells
    elif args.full:
        planned_cells = baseline_cells + ttft_cells
    else:
        planned_cells = cells
    report = {
        "schema": "muser.campaign.plan.v1",
        "mode": (
            "dry-run" if args.dry_run else "smoke" if args.smoke
            else "baseline-full" if args.full else "kvpack-full" if args.kvpack_full
            else "vision-full" if args.vision_full else "dflash-tune" if args.dflash_tune
            else "dflash-full" if args.dflash_full else "ane-full" if args.ane_full
            else "remote-full"
        ),
        "accelerator_touched": False,
        "dry_run_stage": args.dry_run_stage if args.dry_run else None,
        "run_id": args.run_id,
        "identity": identity,
        "blockers": blockers,
        "cells": planned_cells,
        "cell_count": len(planned_cells),
        "commands": sum(len(cell["commands"]) for cell in planned_cells),
        "seal_eligible": False,
    }
    if args.dry_run:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if blockers:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1

    identity_payload = (json.dumps(identity, indent=2, sort_keys=True) + "\n").encode()
    publish_exact(args.out_dir.resolve() / f"identity-{args.run_id}.json", identity_payload)
    fixtures = materialize_fixtures(args)
    print(json.dumps({"identity": identity["digest"], "fixtures": fixtures}, sort_keys=True))
    sealed = not depths
    if sealed and (
        args.full or args.kvpack_full or args.vision_full or args.dflash_tune
        or args.dflash_full or args.ane_full or args.remote_full
    ):
        validate_correctness_receipts(
            args.correctness_receipt,
            args.greedy_correctness_receipt,
            identity["digest"],
        )
    if args.full:
        validate_stage_seal(
            args.kvpack_seal, identity["digest"], "muser.kvpack.seal.v1", "--kvpack-seal"
        )
        validate_stage_seal(
            args.vision_seal, identity["digest"], "muser.vision.seal.v1", "--vision-seal"
        )
    if sealed and (
        args.dflash_tune or args.dflash_full or args.ane_full or args.remote_full
    ):
        validate_baseline_seal(args.baseline_seal, identity["digest"])
    if args.ane_full:
        validate_stage_seal(
            args.dflash_seal,
            identity["digest"],
            "muser.dflash.seal.v1",
            "--dflash-seal",
        )
    if args.remote_full:
        validate_stage_seal(
            args.vision_seal,
            identity["digest"],
            "muser.vision.seal.v1",
            "--vision-seal",
        )

    if args.full:
        selected = baseline_cells + release_matrix.ttft_cells(margs)
    elif args.kvpack_full:
        selected = release_matrix.kvpack_cells(margs)
    elif args.vision_full:
        selected = vision_cells
    elif args.dflash_tune:
        selected = dflash_tune_cells
    elif args.dflash_full:
        verify_length = (
            7
            if depths
            else frozen_dflash_verify_length(args.dflash_tuning_freeze)
        )
        selected = diagnostic_dflash_cells(
            release_matrix.dflash_cells(margs, verify_length),
            depths,
        )
    elif args.ane_full:
        selected = release_matrix.ane_cells(
            margs,
            frozen_dflash_verify_length(args.dflash_tuning_freeze),
        )
    elif args.remote_full:
        selected = release_matrix.remote_cells(
            margs,
            frozen_dflash_verify_length(args.dflash_tuning_freeze),
        )
    else:
        selected = smoke_cells
    commands: list[tuple[str, str, list[str]]] = []
    if args.smoke and not depths:
        correctness = release_matrix.wrapped(
            identity["digest"], "correctness-smoke", str(args.out_dir.resolve()),
            [
                "env", f"MUSER_MODEL={args.model.resolve()}",
                f"MUSER_MODEL_SHA256={sha256(args.model.resolve())}",
                "cargo", "test", "--release",
                "-p", "muser-engine", "--features", "metal,release-real-model", "--test", "muse_golden",
                "metal_prefill_and_decode_match_exact_llama_greedy_fixture", "--", "--exact", "--nocapture",
            ],
        )
        commands.append(("correctness-smoke", "correctness", correctness))
    for cell in selected:
        for command in cell["commands"]:
            commands.append((cell["cell"], command_engine(command), command))
    return execute_plan(args, identity, commands)


if __name__ == "__main__":
    sys.exit(main())
