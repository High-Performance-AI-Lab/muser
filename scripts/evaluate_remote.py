#!/usr/bin/env python3
"""Seal only a complete live GX10 disaggregated-prefill packet."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re

from packet_integrity import collect_unique_packet, publish_new
from release_lock import force_unsealed


DEPTHS = (8192, 32768, 65536, 131008)
VARIANTS = ("text",)
TTFT_MINIMUM = {8192: 1.10, 32768: 2.10, 65536: 2.25, 131008: 2.50}
LINK_GBPS_MINIMUM = 3.0
CV_MAXIMUM = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--baseline-ledger", type=Path, required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def read_ledger(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean(values: list[int]) -> float:
    return sum(values) / len(values)


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def main() -> int:
    args = parse_args()
    records = read_ledger(args.ledger)
    baseline = read_ledger(args.baseline_ledger)
    expected = {f"remote-{variant}-{depth}" for depth in DEPTHS for variant in VARIANTS}
    relevant, failures = collect_unique_packet(
        records,
        expected,
        identity=args.identity,
        key=lambda record: (
            record.get("cell") if record.get("engine") == "remote" else None
        ),
        label="remote",
    )
    llama, packet_failures = collect_unique_packet(
        baseline,
        set(DEPTHS),
        identity=args.identity,
        key=lambda record: (
            int(str(record.get("cell")).removeprefix("ttft-"))
            if record.get("engine") == "ttft-llama"
            and str(record.get("cell")).removeprefix("ttft-").isdigit()
            else None
        ),
        label="fresh-llama TTFT",
    )
    failures.extend(packet_failures)
    cells: dict[str, dict[str, object]] = {}
    text_speedups: list[float] = []
    for depth in DEPTHS:
        for variant in VARIANTS:
            name = f"remote-{variant}-{depth}"
            record = relevant.get(name)
            if record is None:
                continue
            if record.get("status") != "passed":
                failures.append(f"{name} did not pass")
                continue
            fingerprint = record.get("fingerprint")
            if (
                not isinstance(fingerprint, dict)
                or fingerprint.get("variant") != variant
                or fingerprint.get("prompt_positions") != depth
                or fingerprint.get("output_tokens") != (48 if depth == 131008 else 256)
                or re.fullmatch(r"[0-9a-f]{64}", str(fingerprint.get("generated_tokens_sha256"))) is None
                or re.fullmatch(r"[0-9a-f]{64}", str(fingerprint.get("full_logit_digest"))) is None
            ):
                failures.append(f"{name} has an invalid route/correctness fingerprint")
                continue
            remote_raw = record.get("raw_ns")
            local_raw = record.get("local_ttft_raw_ns")
            decode_ratios = record.get("decode_ratios")
            export_ratios = record.get("producer_export_overhead_ratios")
            first_tile = record.get("producer_first_tile_prefill_fractions")
            hidden = record.get("producer_transfer_hidden_ratios")
            link_gbps = record.get("installed_payload_gbps")
            vectors = (
                remote_raw, local_raw, decode_ratios, export_ratios, first_tile, hidden,
                link_gbps,
            )
            if any(not isinstance(values, list) or len(values) != 3 for values in vectors):
                failures.append(f"{name} lacks three complete cold paired samples")
                continue
            if not all(
                isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0
                for values in vectors for value in values
            ) or not all(value > 0 for value in remote_raw + local_raw):
                failures.append(f"{name} contains invalid raw measurements")
                continue
            if not all(value > 0 for value in link_gbps):
                failures.append(f"{name} contains invalid installed-payload throughput")
                continue
            remote_cv = math.sqrt(
                sum((value - mean(remote_raw)) ** 2 for value in remote_raw) / len(remote_raw)
            ) / mean(remote_raw)
            local_cv = math.sqrt(
                sum((value - mean(local_raw)) ** 2 for value in local_raw) / len(local_raw)
            ) / mean(local_raw)
            link_cv = math.sqrt(
                sum((value - mean(link_gbps)) ** 2 for value in link_gbps) / len(link_gbps)
            ) / mean(link_gbps)
            if remote_cv > CV_MAXIMUM or local_cv > CV_MAXIMUM:
                failures.append(f"{name} is unstable")
            median_link = sorted(link_gbps)[len(link_gbps) // 2]
            if median_link < LINK_GBPS_MINIMUM:
                failures.append(f"{name} link median is below 3.0 Gbps")
            if max(export_ratios) > 0.05:
                failures.append(f"{name} export overhead exceeds 5%")
            if max(first_tile) >= 0.25:
                failures.append(f"{name} first tile was not sent before 25% of prefill")
            if depth >= 32768 and min(hidden) < 0.95:
                failures.append(f"{name} hid less than 95% of transfer")
            if max(decode_ratios) > 1.02:
                failures.append(f"{name} remote steady decode regressed by more than 2%")
            acceptance = record.get("dflash_acceptance_ratios")
            absolute_acceptance = record.get("dflash_acceptance")
            if acceptance is not None or absolute_acceptance is not None:
                failures.append(f"{name} contains an unexpected DFlash route")
            text_speedup = None
            if variant == "text" and depth in llama:
                llama_raw = llama[depth].get("raw_ns")
                if (
                    not isinstance(llama_raw, list)
                    or len(llama_raw) != 5
                    or not all(isinstance(value, int) and value > 0 for value in llama_raw)
                ):
                    failures.append(f"ttft-{depth}/llama has invalid raw measurements")
                else:
                    text_speedup = mean(llama_raw) / mean(remote_raw)
                    text_speedups.append(text_speedup)
                    if text_speedup < TTFT_MINIMUM[depth]:
                        failures.append(
                            f"{name} fresh-llama TTFT advantage is only {text_speedup:.6f}x"
                        )
            cells[name] = {
                "remote_cv": remote_cv,
                "local_cv": local_cv,
                "link_gbps_cv": link_cv,
                "median_installed_payload_gbps": median_link,
                "fresh_llama_ttft_speedup": text_speedup,
                "maximum_export_overhead_ratio": max(export_ratios),
                "maximum_first_tile_prefill_fraction": max(first_tile),
                "minimum_transfer_hidden_ratio": min(hidden),
                "maximum_remote_decode_ratio": max(decode_ratios),
                "minimum_dflash_acceptance_ratio": None,
                "minimum_dflash_acceptance": None,
                "generated_tokens_sha256": fingerprint.get("generated_tokens_sha256"),
                "full_logit_digest": fingerprint.get("full_logit_digest"),
            }
    receipt = {
        "schema": "muser.remote-prefill.seal.v1",
        "status": "passed" if not failures and len(cells) == 4 else "failed",
        "identity": args.identity,
        "cells": cells,
        "text_fresh_llama_ttft_geometric_mean_speedup": (
            geometric_mean(text_speedups) if text_speedups else None
        ),
        "failures": failures,
        "ledger_sha256": hashlib.sha256(args.ledger.read_bytes()).hexdigest(),
        "baseline_ledger_sha256": hashlib.sha256(args.baseline_ledger.read_bytes()).hexdigest(),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "seal_eligible": not failures and len(cells) == 4,
    }
    force_unsealed(receipt, lane="remote")
    encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    publish_new(args.out, encoded)
    print(encoded, end="")
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
