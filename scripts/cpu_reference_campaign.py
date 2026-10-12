#!/usr/bin/env python3
"""Qualify standalone Muse CPU logits against a fresh CPU-only llama.cpp.

This is the fresh-comparator correctness lane used before the Stage 6 baseline
seal, not the historical Stage 1 acceptance gate and not a Metal/performance
seal. It runs five deterministic 66-token fixtures, retains complete raw
evidence, and publishes only after every fixture passes the numerical and rank
gates.
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


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
FIXTURES = ("p1", "p2", "p3", "swa", "long")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--identity", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--muser-forward", type=Path, required=True)
    parser.add_argument("--muser-fixture", type=Path, required=True)
    parser.add_argument("--llama-perplexity", type=Path, required=True)
    parser.add_argument("--llama-receipt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path, label: str, blockers: list[str]) -> dict | None:
    if not path.is_file() or path.is_symlink():
        blockers.append(f"missing or unsafe {label}: {path}")
        return None
    return {"basename": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def plan(args: argparse.Namespace) -> tuple[dict, dict | None]:
    blockers: list[str] = []
    artifacts = {
        "model": artifact(args.model, "model", blockers),
        "muser_forward": artifact(args.muser_forward, "Muser forward producer", blockers),
        "muser_fixture": artifact(args.muser_fixture, "Muser fixture producer", blockers),
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
                or receipt.get("build", {}).get("metal") is not False
                or expected.get("bytes") != actual.get("bytes")
                or expected.get("sha256") != actual.get("sha256")
            ):
                blockers.append("llama-perplexity is not its CPU-only unexecuted v3 build")
    lane = args.out_dir.resolve() / f"cpu-reference-{args.run_id}"
    fixtures = FIXTURES[:1] if args.smoke else FIXTURES
    return {
        "schema": "muser.cpu-reference.plan.v1",
        "mode": "dry-run" if args.dry_run else "execute",
        "accelerator_touched": False,
        "identity": args.identity,
        "run_id": args.run_id,
        "lane_dir": str(lane),
        "qualification_mode": "smoke" if args.smoke else "full",
        "fixtures": list(fixtures),
        "artifacts": artifacts,
        "blockers": blockers,
        "policy": {
            "context_tokens": 66,
            "scored_rows_per_fixture": 32,
            "llama_threads": 20,
            "llama_gpu_layers": 0,
            "flash_attention": True,
            "kv": "f16",
            "automatic_retry": False,
        },
        "seal_eligible": False,
    }, receipt


def run_logged(
    command: list[str],
    log: Path,
    environment: dict[str, str] | None = None,
    accepted_returncodes: frozenset[int] = frozenset({0}),
) -> None:
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
    if result.returncode not in accepted_returncodes:
        raise RuntimeError(f"child exited {result.returncode}: {' '.join(command[:2])}")


def publish(path: Path, value: dict) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def aggregate_verdicts(cases: list[dict]) -> dict:
    gates = {
        "maximum_cpu_logit_error": 0.5,
        "maximum_relative_target_nll_error": 0.005,
        "minimum_mean_top10_overlap": 0.9,
        "minimum_top1_agreement": 0.985,
        "nonfinite_values": 0,
    }
    failures: list[str] = []
    rows = 0
    top1_matches = 0
    top10_weighted = 0.0
    candidate_nll = 0.0
    llama_nll = 0.0
    maximum_cpu_error = 0.0
    maximum_centered_error = 0.0
    nonfinite = 0
    for case in cases:
        verdict = case.get("evaluation")
        reference = verdict.get("llama_reference") if isinstance(verdict, dict) else None
        raw = verdict.get("raw_cpu_candidate") if isinstance(verdict, dict) else None
        if not isinstance(reference, dict) or not isinstance(raw, dict):
            failures.append(f"{case.get('fixture', 'unknown')}: invalid evaluator evidence")
            continue
        count = reference.get("scored_rows")
        agreement = reference.get("top1_agreement")
        overlap = reference.get("mean_top10_overlap")
        if (
            not isinstance(count, int)
            or count <= 0
            or not isinstance(agreement, (int, float))
            or not isinstance(overlap, (int, float))
        ):
            failures.append(f"{case.get('fixture', 'unknown')}: invalid rank metrics")
            continue
        rows += count
        top1_matches += round(float(agreement) * count)
        top10_weighted += float(overlap) * count
        candidate_nll += float(reference["candidate_target_nll_sum"])
        llama_nll += float(reference["llama_target_nll_sum"])
        maximum_centered_error = max(
            maximum_centered_error,
            float(reference["maximum_centered_logit_error"]),
        )
        maximum_cpu_error = max(maximum_cpu_error, float(raw["maximum_absolute_error"]))
        nonfinite += int(raw["nonfinite_values"])
    top1 = top1_matches / rows if rows else 0.0
    top10 = top10_weighted / rows if rows else 0.0
    relative_nll = abs(candidate_nll - llama_nll) / max(abs(llama_nll), 1e-30)
    if nonfinite != 0:
        failures.append("nonfinite CPU logits")
    if maximum_cpu_error > gates["maximum_cpu_logit_error"]:
        failures.append("CPU maximum absolute logit error")
    if relative_nll > gates["maximum_relative_target_nll_error"]:
        failures.append("relative target-NLL error")
    if top1 < gates["minimum_top1_agreement"]:
        failures.append("top-1 agreement")
    if top10 < gates["minimum_mean_top10_overlap"]:
        failures.append("mean top-10 overlap")
    return {
        "schema": "muser.cpu-reference.aggregate.v1",
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "gates": gates,
        "scored_rows": rows,
        "top1_matches": top1_matches,
        "top1_agreement": top1,
        "mean_top10_overlap": top10,
        "candidate_target_nll_sum": candidate_nll,
        "llama_target_nll_sum": llama_nll,
        "relative_target_nll_error": relative_nll,
        "maximum_centered_logit_error": maximum_centered_error,
        "maximum_cpu_logit_error": maximum_cpu_error,
        "nonfinite_values": nonfinite,
    }


def execute(args: argparse.Namespace, static: dict, comparator: dict) -> int:
    if static["blockers"]:
        raise RuntimeError("static blockers: " + "; ".join(static["blockers"]))
    lane = Path(static["lane_dir"])
    lane.mkdir(parents=True, exist_ok=False)
    cases = []
    for fixture_id in static["fixtures"]:
        case = lane / fixture_id
        case.mkdir()
        tokens = case / "teacher.tokens"
        corpus = case / "teacher.txt"
        fixture_log = case / "fixture.jsonl"
        rows = case / "muser.rows.jsonl"
        logits = case / "muser.logits.f32"
        llama_logits = case / "llama.logits.u16"
        llama_log = case / "llama-perplexity.log"
        evaluation = case / "evaluation.json"
        run_logged(
            [
                str(args.muser_fixture), "--model", str(args.model),
                "--tokens-out", str(tokens), "--corpus-out", str(corpus),
                "--count", "66", "--fixture-id", fixture_id,
                "--identity", args.identity,
            ],
            fixture_log,
        )
        run_logged(
            [
                str(args.muser_forward), "--model", str(args.model),
                "--token-fixture", str(tokens), "--backend", "cpu",
                "--top-k", "10", "--identity", args.identity,
                "--logits-out", str(logits),
            ],
            rows,
        )
        run_logged(
            [
                str(args.llama_perplexity), "-m", str(args.model),
                "-f", str(corpus), "-c", "66", "-b", "66", "-ub", "66",
                "-t", "20", "-ngl", "0", "-fa", "1", "-ctk", "f16",
                "-ctv", "f16", "--chunks", "1", "--save-all-logits",
                str(llama_logits),
            ],
            llama_log,
            {"MUSER_COMPARATOR_TOKEN_FIXTURE": str(tokens)},
        )
        evaluator_log = case / "evaluator.log"
        run_logged(
            [
                sys.executable, str(ROOT / "scripts" / "evaluate_logits.py"),
                "--candidate-rows", str(rows), "--candidate-logits", str(logits),
                "--cpu-logits", str(logits), "--llama-logits", str(llama_logits),
                "--llama-route", "cpu-only",
                "--llama-upstream-commit", comparator["source_commit"],
                "--llama-patch-sha256", comparator["patch_sha256"],
                "--model-transformer-layers", "52", "--identity", args.identity,
                "--report", str(evaluation),
            ],
            evaluator_log,
            accepted_returncodes=frozenset({0, 1}),
        )
        verdict = json.loads(evaluation.read_text(encoding="utf-8"))
        evidence = {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(case.iterdir())
        }
        cases.append({
            "fixture": fixture_id,
            "evaluation": verdict,
            "evidence": evidence,
        })
    full = static["qualification_mode"] == "full"
    aggregate = aggregate_verdicts(cases)
    if not full:
        smoke_failures = [
            failure
            for failure in aggregate["failures"]
            if failure in {
                "nonfinite CPU logits",
                "CPU maximum absolute logit error",
                "mean top-10 overlap",
            }
            or failure.endswith("invalid evaluator evidence")
            or failure.endswith("invalid rank metrics")
        ]
        aggregate = {**aggregate, "status": "passed" if not smoke_failures else "failed",
                     "failures": smoke_failures, "smoke_thresholds_only": True}
    passed = aggregate["status"] == "passed"
    receipt = {
        "schema": (
            "muser.cpu-reference.receipt.v1"
            if full
            else "muser.cpu-reference.smoke.v1"
        ),
        "status": "passed" if passed else "failed",
        "seal_eligible": passed and full and len(cases) == len(FIXTURES),
        "identity": args.identity,
        "run_id": args.run_id,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "plan": static,
        "aggregate": aggregate,
        "cases": cases,
    }
    publish(lane / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if passed else 1


def main() -> int:
    args = parse_args()
    if not RUN_ID.fullmatch(args.run_id):
        raise SystemExit("unsafe --run-id")
    static, comparator = plan(args)
    if args.dry_run:
        print(json.dumps(static, indent=2, sort_keys=True))
        return 0
    try:
        if comparator is None:
            raise RuntimeError("llama comparator receipt is unavailable")
        return execute(args, static, comparator)
    except Exception as error:
        print(f"cpu_reference_campaign: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
