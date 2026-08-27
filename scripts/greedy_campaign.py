#!/usr/bin/env python3
"""Produce the exact 64-token long-context Muser/llama correctness receipt."""

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
CASES = (
    ("diverse-p1", "p1", 128),
    ("diverse-p2", "p2", 512),
    ("diverse-p3", "p3", 2_048),
    ("swa-2047", "swa", 2_047),
    ("swa-2048", "swa", 2_048),
    ("swa-2049", "swa", 2_049),
    ("long-8k", "long", 8_192),
    ("long-16k", "long", 16_384),
    ("long-32k", "long", 32_768),
    ("long-64k", "long", 65_536),
    ("long-131008", "long", 131_008),
)
SNAPSHOT_REPLAY_DEPTHS = (8_192, 16_384, 32_768, 65_536, 131_008)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--identity", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--muser-greedy", type=Path, required=True)
    parser.add_argument("--muser-fixture", type=Path, required=True)
    parser.add_argument("--muser-forward", type=Path, required=True)
    parser.add_argument("--llama-perplexity", type=Path, required=True)
    parser.add_argument("--llama-receipt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--quiet-seconds", type=int, default=10)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path, label: str, blockers: list[str]) -> dict | None:
    if not path.is_file() or path.is_symlink():
        blockers.append(f"missing or unsafe {label}: {path}")
        return None
    return {"basename": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def static_plan(args: argparse.Namespace) -> tuple[dict, dict | None]:
    blockers: list[str] = []
    artifacts = {
        "model": artifact(args.model, "model", blockers),
        "muser_greedy": artifact(args.muser_greedy, "Muser greedy producer", blockers),
        "muser_fixture": artifact(args.muser_fixture, "Muser fixture producer", blockers),
        "muser_forward": artifact(args.muser_forward, "Muser corpus renderer", blockers),
        "llama_perplexity": artifact(args.llama_perplexity, "llama-perplexity", blockers),
        "llama_receipt": artifact(args.llama_receipt, "llama receipt", blockers),
    }
    receipt = None
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
                blockers.append("llama-perplexity differs from its unexecuted v3 receipt")
    lane = args.out_dir.resolve() / f"greedy-{args.run_id}"
    return ({
        "schema": "muser.greedy-campaign.plan.v1",
        "mode": "dry-run" if args.dry_run else "execute",
        "accelerator_touched": False,
        "identity": args.identity,
        "run_id": args.run_id,
        "lane_dir": str(lane),
        "artifacts": artifacts,
        "blockers": blockers,
        "cases": [
            {
                "id": case_id,
                "fixture_id": fixture_id,
                "prompt_tokens": depth,
                "output_tokens": 64,
                "combined_context": depth + 64,
            }
            for case_id, fixture_id, depth in CASES
        ],
        "policy": {
            "muser_backend": "metal",
            "llama_threads": 20,
            "llama_gpu_layers": 99,
            "batch": 2_048,
            "ubatch": 512,
            "kv": "f16",
            "flash_attention": True,
            "same_engine_repetitions": 1,
            "cross_engine_result": "exact-only",
            "detached_snapshot_replay": "exact-tokens-and-all-step-full-logit-digest",
            "snapshot_replay_depths": list(SNAPSHOT_REPLAY_DEPTHS),
            "ring_import_cuts": [2_047, 2_048, 2_049, 2_559, 2_560, 2_561],
            "automatic_retry": False,
            "quiet_seconds": args.quiet_seconds,
        },
        "seal_eligible": False,
    }, receipt)


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


def token_file(path: Path) -> list[int]:
    return [int(value) for value in path.read_bytes().split()]


def publish_tokens(path: Path, tokens: list[int]) -> None:
    payload = ("\n".join(map(str, tokens)) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def publish_json(path: Path, value: dict) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def execute(args: argparse.Namespace, plan: dict, comparator: dict) -> int:
    if plan["blockers"]:
        raise RuntimeError("static blockers: " + "; ".join(plan["blockers"]))
    if os.environ.get("MUSER_ACCELERATOR_LEASE") != "1":
        raise RuntimeError("execution must be a child of accelerator_safe.py")
    if args.quiet_seconds < 10:
        raise RuntimeError("execution requires quiet intervals of at least ten seconds")
    lane = Path(plan["lane_dir"])
    lane.mkdir(parents=True, exist_ok=False)
    ring_log = lane / "ring-import-boundaries.log"
    run_logged(
        [
            "cargo", "test", "--release", "-p", "muser-engine", "--lib",
            "--features", "metal",
            "decode::tests::real_model_wrap_boundaries_and_detached_restore_replay_exactly",
            "--", "--exact", "--nocapture",
        ],
        ring_log,
        args.quiet_seconds,
        False,
        environment={"MUSER_MODEL": str(args.model)},
    )
    ring_text = ring_log.read_text(encoding="utf-8", errors="replace")
    if "test result: ok. 1 passed;" not in ring_text:
        raise RuntimeError("ring/import boundary test did not execute exactly once")
    if "skipped: set MUSER_MODEL" in ring_text:
        raise RuntimeError(
            "ring/import boundary test skipped itself instead of running; "
            "MUSER_MODEL did not reach the cargo test environment"
        )
    evaluations = []
    for case_id, fixture_id, depth in CASES:
        case = lane / case_id
        case.mkdir()
        prompt = case / "prompt.tokens"
        prompt_corpus = case / "prompt.txt"
        fixture_log = case / "fixture.jsonl"
        generated = case / "generated.tokens"
        muser_log = case / "muser-greedy.jsonl"
        combined = case / "combined.tokens"
        combined_corpus = case / "combined.txt"
        llama_logits = case / "llama.logits.u16"
        llama_log = case / "llama-perplexity.log"
        evaluation = case / "evaluation.json"
        run_logged(
            [
                str(args.muser_fixture), "--model", str(args.model),
                "--tokens-out", str(prompt), "--corpus-out", str(prompt_corpus),
                "--count", str(depth), "--fixture-id", fixture_id,
                "--identity", args.identity,
            ],
            fixture_log,
            args.quiet_seconds,
            False,
        )
        muser_command = [
                str(args.muser_greedy), "--model", str(args.model),
                "--prompt-token-fixture", str(prompt), "--output-tokens", "64",
                "--backend", "metal", "--identity", args.identity,
                "--tokens-out", str(generated),
        ]
        if depth in SNAPSHOT_REPLAY_DEPTHS:
            muser_command.append("--snapshot-replay")
        run_logged(
            muser_command,
            muser_log,
            args.quiet_seconds,
            True,
        )
        publish_tokens(combined, token_file(prompt) + token_file(generated))
        # llama-perplexity still requires -f. The comparator patch consumes
        # MUSER_COMPARATOR_TOKEN_FIXTURE and does not tokenize this file.
        # Greedy special tokens have no exact UTF-8 round-trip.
        dummy = b"x\n"
        descriptor = os.open(combined_corpus, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(dummy)
            handle.flush()
            os.fsync(handle.fileno())
        context = depth + 64
        batch = min(2_048, context)
        ubatch = min(512, context)
        run_logged(
            [
                str(args.llama_perplexity), "-m", str(args.model),
                "-f", str(combined_corpus), "-c", str(context),
                "-b", str(batch), "-ub", str(ubatch), "-t", "20",
                "-ngl", "99", "-fa", "1", "-ctk", "f16", "-ctv", "f16",
                "--chunks", "1", "--save-all-logits", str(llama_logits),
            ],
            llama_log,
            args.quiet_seconds,
            True,
            {"MUSER_COMPARATOR_TOKEN_FIXTURE": str(combined)},
        )
        evaluator_log = case / "evaluator.log"
        evaluator_command = [
                sys.executable, str(ROOT / "scripts" / "evaluate_greedy.py"),
                "--muser-evidence", str(muser_log),
                "--prompt-fixture", str(prompt),
                "--generated-fixture", str(generated),
                "--llama-logits", str(llama_logits),
                "--identity", args.identity,
                "--llama-upstream-commit", comparator["source_commit"],
                "--llama-patch-sha256", comparator["patch_sha256"],
                "--context-length", str(context),
                "--batch-size", str(batch), "--ubatch-size", str(ubatch),
                "--threads", "20", "--model-transformer-layers", "52",
                "--report", str(evaluation),
        ]
        if depth in SNAPSHOT_REPLAY_DEPTHS:
            evaluator_command.append("--require-snapshot-replay")
        run_logged(
            evaluator_command,
            evaluator_log,
            args.quiet_seconds,
            False,
        )
        verdict = json.loads(evaluation.read_text(encoding="utf-8"))
        if verdict.get("status") != "passed" or verdict.get("seal_eligible") is not True:
            raise RuntimeError(f"{case_id}: exact greedy evaluator did not pass")
        evaluations.append({
            "case": case_id,
            "prompt_tokens": depth,
            "snapshot_position": verdict["snapshot_position"],
            "snapshot_replay_all_logits_sha256": verdict[
                "snapshot_replay_all_logits_sha256"
            ],
            "evaluation_sha256": sha256(evaluation),
            "combined_tokens_sha256": verdict["combined_tokens_sha256"],
            "muser_all_logits_sha256": verdict["muser_all_logits_sha256"],
            "llama_artifacts": verdict["llama_artifacts"],
        })
    receipt = {
        "schema": "muser.greedy-correctness.receipt.v1",
        "status": "passed",
        "identity": args.identity,
        "run_id": args.run_id,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "plan": plan,
        "cases": evaluations,
        "exact_cases": len(evaluations),
        "snapshot_replay_depths": [
            case["prompt_tokens"] for case in evaluations
            if case["prompt_tokens"] in SNAPSHOT_REPLAY_DEPTHS
        ],
        "ring_import_cuts": [2_047, 2_048, 2_049, 2_559, 2_560, 2_561],
        "ring_import_log_sha256": sha256(ring_log),
        "seal_eligible": (
            len(evaluations) == len(CASES)
            and all(
                case["snapshot_position"] == case["prompt_tokens"]
                for case in evaluations
                if case["prompt_tokens"] in SNAPSHOT_REPLAY_DEPTHS
            )
        ),
    }
    publish_json(lane / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def main() -> int:
    args = parse_args()
    if not args.dry_run:
        try:
            require_sealing_enabled("greedy correctness receipt creation")
        except ReleaseLocked as error:
            raise SystemExit(str(error)) from None
    if not RUN_ID.fullmatch(args.run_id):
        raise SystemExit("unsafe --run-id")
    plan, comparator = static_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    try:
        if comparator is None:
            raise RuntimeError("llama comparator receipt is unavailable")
        return execute(args, plan, comparator)
    except Exception as error:
        print(f"greedy_campaign: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
