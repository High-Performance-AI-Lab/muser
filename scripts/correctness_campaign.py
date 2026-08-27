#!/usr/bin/env python3
"""Produce the mandatory 32-row Muse teacher-forced correctness receipt.

Execute this only as one child of `accelerator_safe.py`: CPU reference, Metal
candidate, and pinned llama.cpp run under one lease with quiet intervals and
no automatic retry.  Dry-run performs static identity checks and writes
nothing.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from release_lock import ReleaseLocked, require_sealing_enabled


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--identity", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--muser-forward", type=Path, required=True)
    parser.add_argument("--muser-fixture", type=Path, required=True)
    parser.add_argument("--llama-perplexity", type=Path, required=True)
    parser.add_argument("--llama-receipt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--quiet-seconds", type=int, default=10)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path, label: str, blockers: list[str]) -> dict | None:
    if not path.is_file() or path.is_symlink():
        blockers.append(f"missing or unsafe {label}: {path}")
        return None
    return {"basename": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def static_plan(args: argparse.Namespace) -> dict:
    blockers: list[str] = []
    artifacts = {
        "model": artifact(args.model, "model", blockers),
        "muser_forward": artifact(args.muser_forward, "Muser forward producer", blockers),
        "muser_fixture": artifact(args.muser_fixture, "Muser fixture producer", blockers),
        "llama_perplexity": artifact(args.llama_perplexity, "llama-perplexity", blockers),
        "llama_receipt": artifact(args.llama_receipt, "llama receipt", blockers),
    }
    if artifacts["llama_receipt"] is not None:
        try:
            receipt = json.loads(args.llama_receipt.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            blockers.append(f"invalid llama receipt: {error}")
        else:
            expected = receipt.get("artifacts", {}).get("llama-perplexity", {})
            actual = artifacts["llama_perplexity"] or {}
            if (
                receipt.get("schema") != "muser.llama_comparator.source_receipt.v3"
                or receipt.get("executed") is not False
                or expected.get("bytes") != actual.get("bytes")
                or expected.get("sha256") != actual.get("sha256")
            ):
                blockers.append("llama-perplexity differs from its unexecuted v3 build receipt")
    lane = args.out_dir.resolve() / f"correctness-{args.run_id}"
    return {
        "schema": "muser.correctness.plan.v1",
        "mode": "dry-run" if args.dry_run else "execute",
        "accelerator_touched": False,
        "identity": args.identity,
        "run_id": args.run_id,
        "lane_dir": str(lane),
        "artifacts": artifacts,
        "blockers": blockers,
        "policy": {
            "context_tokens": 66,
            "scored_rows": 32,
            "cpu_metal_max_logit_error": 1.1,
            "relative_target_nll_error": 0.005,
            "minimum_top1_agreement": 0.985,
            "minimum_mean_top10_overlap": 0.90,
            "llama_threads": 20,
            "llama_gpu_layers": 99,
            "kv": "f16",
            "flash_attention": True,
            "quiet_seconds": args.quiet_seconds,
            "automatic_retry": False,
        },
        "seal_eligible": False,
    }


def run_logged(
    command: list[str],
    log: Path,
    quiet: int,
    accelerator: bool,
    environment: dict[str, str] | None = None,
) -> None:
    if accelerator:
        time.sleep(quiet)
    child_environment = os.environ.copy()
    if environment is not None:
        child_environment.update(environment)
    with log.open("xb") as output:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            env=child_environment,
        )
        output.flush()
        os.fsync(output.fileno())
    if accelerator:
        time.sleep(quiet)
    if result.returncode != 0:
        raise RuntimeError(f"child exited {result.returncode}: {' '.join(command[:2])}")


def publish(path: Path, value: dict) -> None:
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def execute(args: argparse.Namespace, plan: dict) -> int:
    if plan["blockers"]:
        raise RuntimeError("static blockers: " + "; ".join(plan["blockers"]))
    if os.environ.get("MUSER_ACCELERATOR_LEASE") != "1":
        raise RuntimeError("execution must be a child of accelerator_safe.py")
    if args.quiet_seconds < 10:
        raise RuntimeError("execution requires quiet intervals of at least ten seconds")
    lane = Path(plan["lane_dir"])
    lane.mkdir(parents=True, exist_ok=False)
    tokens = lane / "teacher-66.tokens"
    corpus = lane / "teacher-66.txt"
    fixture_log = lane / "fixture.jsonl"
    cpu_rows = lane / "cpu.rows.jsonl"
    metal_rows = lane / "metal.rows.jsonl"
    cpu_logits = lane / "cpu.logits.f32"
    metal_logits = lane / "metal.logits.f32"
    llama_logits = lane / "llama.logits.u16"
    llama_log = lane / "llama-perplexity.log"
    evaluation = lane / "evaluation.json"
    comparator_receipt = json.loads(args.llama_receipt.read_text(encoding="utf-8"))
    upstream_commit = comparator_receipt["source_commit"]
    patch_sha256 = comparator_receipt["patch_sha256"]
    run_logged(
        [
            str(args.muser_fixture), "--model", str(args.model),
            "--tokens-out", str(tokens), "--corpus-out", str(corpus),
            "--count", "66", "--identity", args.identity,
        ],
        fixture_log,
        args.quiet_seconds,
        False,
    )
    common = [
        "--model", str(args.model), "--token-fixture", str(tokens),
        "--top-k", "10", "--identity", args.identity,
    ]
    run_logged(
        [str(args.muser_forward), *common, "--backend", "cpu", "--logits-out", str(cpu_logits)],
        cpu_rows,
        args.quiet_seconds,
        False,
    )
    run_logged(
        [str(args.muser_forward), *common, "--backend", "metal", "--logits-out", str(metal_logits)],
        metal_rows,
        args.quiet_seconds,
        True,
    )
    run_logged(
        [
            str(args.llama_perplexity), "-m", str(args.model), "-f", str(corpus),
            "-c", "66", "-b", "66", "-ub", "66", "-t", "20", "-ngl", "99",
            "-fa", "1", "-ctk", "f16", "-ctv", "f16", "--chunks", "1",
            "--save-all-logits", str(llama_logits),
        ],
        llama_log,
        args.quiet_seconds,
        True,
        {"MUSER_COMPARATOR_TOKEN_FIXTURE": str(tokens)},
    )
    evaluator_log = lane / "evaluator.log"
    run_logged(
        [
            sys.executable, str(ROOT / "scripts" / "evaluate_logits.py"),
            "--candidate-rows", str(metal_rows), "--candidate-logits", str(metal_logits),
            "--cpu-logits", str(cpu_logits), "--llama-logits", str(llama_logits),
            "--llama-upstream-commit", upstream_commit,
            "--llama-patch-sha256", patch_sha256,
            "--model-transformer-layers", "52",
            "--maximum-cpu-logit-error", "1.1",
            "--identity", args.identity, "--report", str(evaluation),
        ],
        evaluator_log,
        args.quiet_seconds,
        False,
    )
    verdict = json.loads(evaluation.read_text(encoding="utf-8"))
    if verdict.get("status") != "passed" or verdict.get("seal_eligible") is not True:
        raise RuntimeError("logit evaluator did not produce a passing seal-eligible verdict")
    evidence = {}
    for path in sorted(lane.iterdir()):
        if path.name == "receipt.json":
            continue
        evidence[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    receipt = {
        "schema": "muser.correctness.receipt.v1",
        "status": "passed",
        "identity": args.identity,
        "run_id": args.run_id,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "plan": plan,
        "evaluation": verdict,
        "evidence": evidence,
    }
    publish(lane / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def main() -> int:
    args = parse_args()
    if not args.dry_run:
        try:
            require_sealing_enabled("correctness receipt creation")
        except ReleaseLocked as error:
            raise SystemExit(str(error)) from None
    if not RUN_ID.fullmatch(args.run_id):
        raise SystemExit("unsafe --run-id")
    plan = static_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    try:
        return execute(args, plan)
    except Exception as error:
        print(f"correctness_campaign: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
