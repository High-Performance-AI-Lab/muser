#!/usr/bin/env python3
"""Seal only a complete, correct, stable four-fixture vision packet."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

from packet_integrity import collect_unique_packet, publish_new
from release_lock import force_unsealed


FIXTURES = ("low-square", "wide", "tall", "high-resolution")
FIXTURE_DIMENSIONS = {
    "low-square": (224, 224),
    "wide": (1024, 256),
    "tall": (256, 1024),
    "high-resolution": (2048, 1536),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def mean(values: list[int]) -> float:
    return sum(values) / len(values)


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def position_digest(start: int, end: int) -> str:
    digest = hashlib.sha256()
    for position in range(start, end):
        digest.update(position.to_bytes(8, "little"))
    return "sha256:" + digest.hexdigest()


def valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def main() -> int:
    args = parse_args()
    records = [
        json.loads(line) for line in args.ledger.read_text().splitlines() if line.strip()
    ]
    expected_keys = {
        (f"vision-{fixture}", engine)
        for fixture in FIXTURES
        for engine in ("vision", "vision-ttft-muser", "vision-ttft-llama")
    }
    relevant, failures = collect_unique_packet(
        records,
        expected_keys,
        identity=args.identity,
        key=lambda record: (record.get("cell"), record.get("engine")),
        label="vision",
    )
    cells: dict[str, dict[str, object]] = {}
    stable_speedups: list[float] = []
    stable_fixtures: list[str] = []
    for fixture in FIXTURES:
        name = f"vision-{fixture}"
        by_engine = {
            engine: relevant[(name, engine)]
            for engine in ("vision", "vision-ttft-muser", "vision-ttft-llama")
            if (name, engine) in relevant
            and relevant[(name, engine)].get("status") == "passed"
        }
        expected = {"vision", "vision-ttft-muser", "vision-ttft-llama"}
        if set(by_engine) != expected:
            failures.append(
                f"{name} incomplete engines: missing {sorted(expected - set(by_engine))}"
            )
            continue
        qualifier = by_engine["vision"]
        muser = by_engine["vision-ttft-muser"]
        llama = by_engine["vision-ttft-llama"]
        qfp = qualifier.get("fingerprint", {})
        mfp = muser.get("fingerprint", {})
        lfp = llama.get("fingerprint", {})
        if any(fp.get("fixture") != fixture for fp in (qfp, mfp, lfp)):
            failures.append(f"{name} contains a mixed fixture identity")
            continue
        image_digests = {qfp.get("image_sha256"), mfp.get("image_sha256"), lfp.get("image_sha256")}
        if len(image_digests) != 1 or None in image_digests:
            failures.append(f"{name} contains mixed image bytes")
            continue
        if (
            qfp.get("route") != "mtmd-metal:muser-mtmd-muse-vision-v1"
            or (qfp.get("source_width"), qfp.get("source_height"))
            != FIXTURE_DIMENSIONS[fixture]
            or qfp.get("max_pixel_error", 1.0) > 1 / 255
            or qfp.get("embedding_cosine", 0.0) < 0.999
            or qfp.get("embedding_relative_l2", 1.0) > 0.01
            or qfp.get("exact_decoder_tokens") is not True
            or not valid_sha256(qfp.get("decoder_tokens_sha256"))
        ):
            failures.append(f"{name} failed a correctness threshold")
            continue
        insertion_start = qfp.get("insertion_start")
        insertion_end = qfp.get("insertion_end")
        insertion_count = qfp.get("insertion_count")
        projected_tokens = qfp.get("projected_tokens")
        prefix_tokens = qfp.get("prefix_tokens")
        suffix_tokens = qfp.get("suffix_tokens")
        installed_positions = qfp.get("installed_positions")
        if (
            not all(
                isinstance(value, int) and value >= 0
                for value in (
                    insertion_start, insertion_end, insertion_count, projected_tokens,
                    prefix_tokens, suffix_tokens, installed_positions,
                )
            )
            or insertion_start != prefix_tokens
            or insertion_end - insertion_start != insertion_count
            or insertion_count != projected_tokens
            or installed_positions != insertion_end + suffix_tokens
            or qfp.get("insertion_positions_sha256")
            != position_digest(insertion_start, insertion_end)
        ):
            failures.append(f"{name} failed exact insertion-position evidence")
            continue
        muser_prompt_counts = mfp.get("reported_prompt_tokens")
        llama_prompt_counts = lfp.get("reported_prompt_tokens")
        if (
            mfp.get("server_lifecycle")
            != "leased-start-ready-exact-requests-cooperative-exit"
            or lfp.get("server_lifecycle")
            != "leased-start-ready-exact-requests-cooperative-exit"
            or not isinstance(muser_prompt_counts, list)
            or len(muser_prompt_counts) != 3
            or not all(isinstance(value, int) and value > 0 for value in muser_prompt_counts)
            or len(set(muser_prompt_counts)) != 1
            or not isinstance(llama_prompt_counts, list)
            or len(llama_prompt_counts) != 5
            or not all(isinstance(value, int) and value > 0 for value in llama_prompt_counts)
            or len(set(llama_prompt_counts)) != 1
            or muser_prompt_counts[0] != llama_prompt_counts[0]
            or muser_prompt_counts[0] != installed_positions
        ):
            failures.append(f"{name} failed server insertion-position evidence")
            continue
        muser_raw = muser.get("raw_ns")
        llama_raw = llama.get("raw_ns")
        if (
            not isinstance(muser_raw, list) or len(muser_raw) != 3
            or not isinstance(llama_raw, list) or len(llama_raw) != 5
            or not all(isinstance(value, int) and value > 0 for value in muser_raw + llama_raw)
        ):
            failures.append(f"{name} has invalid raw timing samples")
            continue
        stable = muser.get("cv", 1.0) <= 0.02 and llama.get("cv", 1.0) <= 0.02
        speedup = mean(llama_raw) / mean(muser_raw)
        if stable:
            stable_fixtures.append(fixture)
            stable_speedups.append(speedup)
            if speedup < 1.0:
                failures.append(f"{name} stable Muser/llama speedup is {speedup:.6f}x")
        cells[fixture] = {
            "stable": stable,
            "muser_cv": muser.get("cv"),
            "llama_cv": llama.get("cv"),
            "speedup": speedup,
            "projected_tokens": qfp.get("projected_tokens"),
            "max_pixel_error": qfp.get("max_pixel_error"),
            "embedding_cosine": qfp.get("embedding_cosine"),
            "embedding_relative_l2": qfp.get("embedding_relative_l2"),
            "image_sha256": qfp.get("image_sha256"),
            "decoder_tokens_sha256": qfp.get("decoder_tokens_sha256"),
        }
    if len(stable_fixtures) < 3:
        failures.append(f"only {len(stable_fixtures)}/4 vision cells are stable")
    if "high-resolution" not in stable_fixtures:
        failures.append("high-resolution vision cell is not stable")
    receipt = {
        "schema": "muser.vision.seal.v1",
        "status": "passed" if not failures and len(cells) == 4 else "failed",
        "identity": args.identity,
        "fixtures": cells,
        "stable_fixtures": stable_fixtures,
        "geometric_mean_speedup": geometric_mean(stable_speedups) if stable_speedups else None,
        "failures": failures,
        "ledger_sha256": hashlib.sha256(args.ledger.read_bytes()).hexdigest(),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "seal_eligible": not failures and len(cells) == 4,
    }
    force_unsealed(receipt, lane="vision")
    encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    publish_new(args.out, encoded)
    print(encoded, end="")
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
