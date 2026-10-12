#!/usr/bin/env python3
"""Fail-closed evaluator for one complete Muse baseline campaign packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any

import release_matrix
from release_lock import force_unsealed


RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def publish(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to replace evaluation artifact: {path}")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        written = 0
        while written < len(payload):
            written += os.write(fd, payload[written:])
        os.fsync(fd)
    finally:
        os.close(fd)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean(values: list[int]) -> float:
    return sum(values) / len(values)


def cv(values: list[int]) -> float:
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / len(values)) / average


def expected_abba_order(index: int, left: str, right: str) -> list[str]:
    return [right, left] if index % 4 in (1, 2) else [left, right]


def normalized_digest(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value)
    return text.removeprefix("sha256:")


def workload_matches(muser: dict[str, Any], llama: dict[str, Any]) -> bool:
    left = muser["fingerprint"]
    right = llama["fingerprint"]
    return all(
        normalized_digest(left.get(muser_key)) == normalized_digest(right.get(llama_key))
        for muser_key, llama_key in [
            ("prompt_fixture_sha256", "prompt_fixture_file_sha256"),
            ("prompt_tokens_sha256", "prompt_tokens_sha256"),
            ("decode_fixture_sha256", "decode_fixture_file_sha256"),
            ("decode_tokens_sha256", "decode_tokens_sha256"),
            ("workload_sha256", "workload_sha256"),
        ]
    )


def workload_is_bound(
    muser: dict[str, Any], llama: dict[str, Any], surface: str, depth: int
) -> bool:
    left = muser.get("fingerprint", {})
    right = llama.get("fingerprint", {})
    required = [("workload_sha256", "workload_sha256")]
    if surface == "prefill" or depth > 0:
        required.extend(
            [
                ("prompt_fixture_sha256", "prompt_fixture_file_sha256"),
                ("prompt_tokens_sha256", "prompt_tokens_sha256"),
            ]
        )
    if surface == "decode":
        required.extend(
            [
                ("decode_fixture_sha256", "decode_fixture_file_sha256"),
                ("decode_tokens_sha256", "decode_tokens_sha256"),
            ]
        )
    return all(
        re.fullmatch(r"[0-9a-f]{64}", str(normalized_digest(left.get(left_key))))
        is not None
        and normalized_digest(left.get(left_key))
        == normalized_digest(right.get(right_key))
        for left_key, right_key in required
    )


def route_projection(engine: str, fingerprint: dict[str, Any]) -> dict[str, Any]:
    if engine == "muser":
        keys = [
            "identity", "backend", "kv", "flash_attention_requested",
            "flash_attention_active", "matvec_route", "ggml_metallib_sha256",
            "prefill_attention_route", "prefill_q4_route", "prefill_dispatch_route",
        ]
    else:
        keys = [
            "build_commit", "build_number", "comparator_upstream_commit",
            "comparator_patch_sha256", "n_batch", "n_ubatch", "n_threads",
            "n_gpu_layers", "type_k", "type_v", "flash_attn",
        ]
    return {key: fingerprint.get(key) for key in keys}


def prior_attempts(out_dir: Path, identity_digest: str | None, run_id: str) -> list[dict[str, Any]]:
    """Surface every other run-id sharing this identity so a retry-until-green
    seal cannot silently hide earlier failed attempts."""
    attempts: list[dict[str, Any]] = []
    for identity_file in sorted(out_dir.glob("identity-*.json")):
        other_run_id = identity_file.name.removeprefix("identity-").removesuffix(".json")
        if other_run_id == run_id:
            continue
        try:
            other_identity = json.loads(identity_file.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if other_identity.get("digest") != identity_digest:
            continue
        evaluation_file = out_dir / f"evaluation-{other_run_id}.json"
        status = None
        if evaluation_file.is_file():
            try:
                status = json.loads(evaluation_file.read_text()).get("status")
            except (OSError, json.JSONDecodeError):
                status = "unreadable"
        attempts.append({"run_id": other_run_id, "status": status})
    return attempts


def evaluate(out_dir: Path, run_id: str) -> dict[str, Any]:
    identity_path = out_dir / f"identity-{run_id}.json"
    ledger_path = out_dir / f"campaign-{run_id}.jsonl"
    failures: list[str] = []
    if not identity_path.is_file():
        return {"status": "failed", "failures": ["missing identity receipt"]}
    if not ledger_path.is_file():
        return {"status": "failed", "failures": ["missing campaign ledger"]}
    identity = json.loads(identity_path.read_text())
    identity_digest = identity.get("digest")
    records = load_jsonl(ledger_path)
    failed_records = [record for record in records if record.get("status") != "passed"]
    if failed_records:
        failures.append("run is permanently tainted by one or more failed records")

    expected = {
        (f"prefill-{depth}", engine)
        for depth in release_matrix.PREFILL
        for engine in ("muser", "llama")
    } | {
        (f"decode-{depth}", engine)
        for depth in release_matrix.DECODE
        for engine in ("muser", "llama")
    } | {
        (f"ttft-{depth}", engine)
        for depth in release_matrix.TTFT
        for engine in ("ttft-muser", "ttft-llama")
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    unexpected: list[tuple[str, str]] = []
    for record in records:
        key = (str(record.get("cell")), str(record.get("engine")))
        if key not in expected:
            unexpected.append(key)
            continue
        if record.get("status") == "passed":
            grouped.setdefault(key, []).append(record)
        if record.get("identity") != identity_digest:
            failures.append(f"identity mismatch in {key[0]}/{key[1]}")
    if unexpected:
        failures.append(f"unexpected records: {sorted(set(unexpected))}")
    missing = sorted(key for key in expected if key not in grouped)
    duplicate = sorted(key for key, values in grouped.items() if len(values) != 1)
    if missing:
        failures.append(f"missing records: {missing}")
    if duplicate:
        failures.append(f"duplicate passing records: {duplicate}")

    pair_results: list[dict[str, Any]] = []
    routes: dict[str, set[str]] = {"muser": set(), "llama": set()}
    stable_by_surface: dict[str, set[int]] = {
        "prefill": set(),
        "decode": set(),
        "ttft": set(),
    }
    if not missing and not duplicate:
        baseline_index = 0
        for surface, depths in [
            ("prefill", release_matrix.PREFILL),
            ("decode", release_matrix.DECODE),
        ]:
            for depth in depths:
                cell = f"{surface}-{depth}"
                muser = grouped[(cell, "muser")][0]
                llama = grouped[(cell, "llama")][0]
                observed_order = [
                    str(record.get("engine"))
                    for record in records
                    if record.get("cell") == cell and record.get("status") == "passed"
                ]
                if observed_order != expected_abba_order(
                    baseline_index, "muser", "llama"
                ):
                    failures.append(f"{cell} is not in the frozen ABBA command order")
                baseline_index += 1
                for engine, record in [("muser", muser), ("llama", llama)]:
                    projection = route_projection(engine, record["fingerprint"])
                    routes[engine].add(json.dumps(projection, sort_keys=True))
                if not workload_matches(muser, llama) or not workload_is_bound(
                    muser, llama, surface, depth
                ):
                    failures.append(f"mixed workload in {cell}")
                muser_samples = muser.get("raw_ns")
                llama_samples = llama.get("raw_ns")
                if not isinstance(muser_samples, list) or len(muser_samples) != 5:
                    failures.append(f"{cell} lacks 5 Muser samples")
                    continue
                if not isinstance(llama_samples, list) or len(llama_samples) != 5:
                    failures.append(f"{cell} lacks 5 llama samples")
                    continue
                muser_cv = cv(muser_samples)
                llama_cv = cv(llama_samples)
                stable = muser_cv <= 0.03 and llama_cv <= 0.03
                ratio = mean(llama_samples) / mean(muser_samples)
                if not stable:
                    failures.append(f"{cell} CV exceeds 3%")
                else:
                    stable_by_surface[surface].add(depth)
                if ratio < 1.0:
                    failures.append(f"baseline regression in {cell}: {ratio:.9f}x")
                pair_results.append(
                    {
                        "cell": cell,
                        "surface": surface,
                        "depth": depth,
                        "muser_raw_ns": muser_samples,
                        "llama_raw_ns": llama_samples,
                        "muser_cv": muser_cv,
                        "llama_cv": llama_cv,
                        "stable": stable,
                        "muser_over_llama_throughput": ratio if stable else None,
                        "classification": "claim" if stable else "retained-no-claim",
                    }
                )
        for ttft_index, depth in enumerate(release_matrix.TTFT):
            cell = f"ttft-{depth}"
            muser = grouped[(cell, "ttft-muser")][0]
            llama = grouped[(cell, "ttft-llama")][0]
            observed_order = [
                str(record.get("engine"))
                for record in records
                if record.get("cell") == cell and record.get("status") == "passed"
            ]
            if observed_order != expected_abba_order(
                ttft_index, "ttft-muser", "ttft-llama"
            ):
                failures.append(f"{cell} is not in the frozen ABBA command order")
            muser_fingerprint = muser.get("fingerprint", {})
            llama_fingerprint = llama.get("fingerprint", {})
            if (
                muser_fingerprint.get("prompt_sha256")
                != llama_fingerprint.get("prompt_sha256")
                or re.fullmatch(
                    r"[0-9a-f]{64}", str(muser_fingerprint.get("prompt_sha256"))
                )
                is None
                or muser_fingerprint.get("reported_prompt_tokens") != depth
                or llama_fingerprint.get("reported_prompt_tokens") != depth
                or muser_fingerprint.get("cache") != "disabled"
                or llama_fingerprint.get("cache") != "disabled"
            ):
                failures.append(f"mixed workload or route in {cell}")
            muser_samples = muser.get("raw_ns")
            llama_samples = llama.get("raw_ns")
            if not isinstance(muser_samples, list) or len(muser_samples) != 5:
                failures.append(f"{cell} lacks 5 Muser samples")
                continue
            if not isinstance(llama_samples, list) or len(llama_samples) != 5:
                failures.append(f"{cell} lacks 5 llama samples")
                continue
            muser_cv = cv(muser_samples)
            llama_cv = cv(llama_samples)
            stable = muser_cv <= 0.03 and llama_cv <= 0.03
            ratio = mean(llama_samples) / mean(muser_samples)
            if not stable:
                failures.append(f"{cell} CV exceeds 3%")
            else:
                stable_by_surface["ttft"].add(depth)
            if ratio < 1.0:
                failures.append(f"baseline regression in {cell}: {ratio:.9f}x")
            pair_results.append(
                {
                    "cell": cell,
                    "surface": "ttft",
                    "depth": depth,
                    "muser_raw_ns": muser_samples,
                    "llama_raw_ns": llama_samples,
                    "muser_cv": muser_cv,
                    "llama_cv": llama_cv,
                    "stable": stable,
                    "muser_over_llama_throughput": ratio if stable else None,
                    "classification": "claim" if stable else "retained-no-claim",
                }
            )

    for engine, values in routes.items():
        if len(values) > 1:
            failures.append(f"mixed {engine} routes across packet")
    required_stable = {
        "prefill": {128, 8192, 32768, 131072},
        "decode": {0, 8192, 32768, 131008},
        "ttft": {128, 8192, 32768, 131008},
    }
    for surface in ("prefill", "decode", "ttft"):
        stable = stable_by_surface[surface]
        if len(stable) != 9:
            failures.append(f"{surface} has only {len(stable)}/9 stable pairs")
        absent = sorted(required_stable[surface] - stable)
        if absent:
            failures.append(f"{surface} lacks required stable depths: {absent}")

    return {
        "schema": "muser.baseline.evaluation.v1",
        "run_id": run_id,
        "identity": identity_digest,
        "status": "passed" if not failures else "failed",
        "seal_eligible": not failures,
        "failures": failures,
        "records_seen": len(records),
        "expected_records": len(expected),
        "stable_counts": {key: len(value) for key, value in stable_by_surface.items()},
        "prior_attempts": prior_attempts(out_dir, identity_digest, run_id),
        "pairs": pair_results,
    }


def main() -> int:
    args = parse_args()
    if not RUN_ID.fullmatch(args.run_id):
        raise SystemExit("unsafe --run-id")
    out_dir = args.out_dir.resolve()
    evaluation = evaluate(out_dir, args.run_id)
    force_unsealed(evaluation, lane="baseline")
    payload = (json.dumps(evaluation, indent=2, sort_keys=True) + "\n").encode()
    evaluation_path = out_dir / f"evaluation-{args.run_id}.json"
    publish(evaluation_path, payload)
    if evaluation["seal_eligible"] is True:
        seal = {
            "schema": "muser.baseline.seal.v1",
            "run_id": args.run_id,
            "identity": evaluation["identity"],
            "evaluation_sha256": digest_bytes(payload),
            "status": "passed",
        }
        publish(
            out_dir / f"seal-{args.run_id}.json",
            (json.dumps(seal, indent=2, sort_keys=True) + "\n").encode(),
        )
    print(json.dumps(evaluation, indent=2, sort_keys=True))
    return 0 if evaluation["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
