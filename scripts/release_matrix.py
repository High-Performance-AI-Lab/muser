#!/usr/bin/env python3
"""Print the frozen baseline matrix and exact dry-run commands as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PREFILL = [128, 512, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
DECODE = [0, 512, 2048, 4096, 8192, 16384, 32768, 65536, 131008]
TTFT = [128, 512, 2048, 4096, 8192, 16384, 32768, 65536, 131008]
REMOTE = [8192, 32768, 65536, 131008]
REMOTE_VARIANTS = ("text",)
DFLASH_DEPTHS = [512, 2048, 8192, 32768]
DFLASH_TUNE_DEPTHS = [256, 4096]
VISION_FIXTURES = ["low-square", "wide", "tall", "high-resolution"]
ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--model", default="muse-glimmer-30B-kquant-17gb.gguf")
    parser.add_argument("--muser-bench", default="muser-bench")
    parser.add_argument("--muser-kvpack", default="muser-kvpack-qualify")
    parser.add_argument("--muser-dflash", default="muser-dflash-qualify")
    parser.add_argument("--muser-vision", default="muser-vision-qualify")
    parser.add_argument("--muser-ane", default="muser-ane-qualify")
    parser.add_argument("--muser-remote", default="muser-remote-qualify")
    parser.add_argument("--dflash", default="muse-glimmer-dflash")
    parser.add_argument("--mmproj", default="muse-glimmer-mmproj.gguf")
    parser.add_argument("--mtmd-bridge", default="libmuser_mtmd_bridge.dylib")
    parser.add_argument("--ane-manifest", default="muse-glimmer-dflash/ane/manifest.json")
    parser.add_argument("--coreml-plan-receipt", default="results/coreml-plan.json")
    parser.add_argument("--cluster-config", default="muser-cluster.json")
    parser.add_argument("--llama-bench", default="llama-bench")
    parser.add_argument("--muser-server", default="muser")
    parser.add_argument("--llama-server", default="llama-server")
    parser.add_argument("--out-dir", default="results/baseline-seal")
    parser.add_argument("--muser-url", default="http://127.0.0.1:4949")
    parser.add_argument("--llama-url", default="http://127.0.0.1:8080")
    parser.add_argument(
        "--comparator-sweep",
        action="store_true",
        help=(
            "also emit a llama-only -ub 512/1024/2048 micro-sweep at the frozen "
            "PP8192 depth, recorded alongside the packet so an 'untuned baseline' "
            "objection is pre-answered with evidence rather than argued from prose"
        ),
    )
    return parser.parse_args()


def wrapped(identity: str, cell: str, out_dir: str, command: list[str]) -> list[str]:
    return [
        "python3",
        str(ROOT / "scripts" / "accelerator_safe.py"),
        "--identity",
        identity,
        "--cell",
        cell,
        "--out-dir",
        out_dir,
        "--",
        *command,
    ]


def cells(args: argparse.Namespace) -> list[dict[str, object]]:
    common_llama = [
        "-m", args.model, "-r", "5", "-t", "20", "-ngl", "99",
        "-b", "2048", "-ub", "512", "-ctk", "f16", "-ctv", "f16", "-fa", "1",
        "-o", "jsonl",
    ]
    records = []
    index = 0
    for surface, depths in [("prefill", PREFILL), ("decode", DECODE)]:
        for depth in depths:
            cell = f"{surface}-{depth}"
            if surface == "prefill":
                prompt_fixture = str(Path(args.out_dir) / "fixtures" / f"prompt-{depth}.txt")
                muser = [
                    args.muser_bench, "--model", args.model, "--surface", surface,
                    "--tokens", str(depth), "--repetitions", "5", "--kv", "f16",
                    "--flash-attention", "on", "--backend", "metal",
                    "--identity", args.identity,
                    "--prompt-token-fixture", prompt_fixture,
                ]
                llama = [
                    args.llama_bench, *common_llama, "-p", str(depth), "-n", "0",
                    "--prompt-token-fixture", prompt_fixture,
                ]
            else:
                decode_fixture = str(Path(args.out_dir) / "fixtures" / "decode-64.txt")
                muser = [
                    args.muser_bench, "--model", args.model, "--surface", surface,
                    "--start-depth", str(depth), "--teacher-forced", "64",
                    "--repetitions", "5", "--kv", "f16", "--flash-attention", "on",
                    "--backend", "metal", "--identity", args.identity,
                    "--decode-token-fixture", decode_fixture,
                ]
                llama = [
                    args.llama_bench, *common_llama, "-p", "0", "-d", str(depth), "-n", "64",
                    "--decode-token-fixture", decode_fixture,
                ]
                if depth > 0:
                    prompt_fixture = str(Path(args.out_dir) / "fixtures" / f"prompt-{depth}.txt")
                    muser.extend(["--prompt-token-fixture", prompt_fixture])
                    llama.extend(["--prompt-token-fixture", prompt_fixture])
            engines = [("muser", muser), ("llama", llama)]
            if index % 4 in (1, 2):
                engines.reverse()
            records.append(
                {
                    "cell": cell,
                    "surface": surface,
                    "depth": depth,
                    "order": [name for name, _ in engines],
                    "commands": [wrapped(args.identity, f"{cell}-{name}", args.out_dir, command) for name, command in engines],
                    "expected": {"correctness_first": True, "muser_repetitions": 5, "llama_repetitions": 5},
                }
            )
            index += 1
    return records


COMPARATOR_SWEEP_UBATCH = [512, 1024, 2048]
COMPARATOR_SWEEP_DEPTH = 8192


def comparator_sweep_cells(args: argparse.Namespace) -> list[dict[str, object]]:
    """Opt-in llama-only -ub micro-sweep at a fixed depth.

    Not part of the sealed muser-vs-llama pairing: this only characterizes
    llama-bench's own sensitivity to -ub so the frozen -ub 512 comparator
    setting is a documented, evidenced choice rather than an unexamined one.
    """
    prompt_fixture = str(Path(args.out_dir) / "fixtures" / f"prompt-{COMPARATOR_SWEEP_DEPTH}.txt")
    records = []
    for ubatch in COMPARATOR_SWEEP_UBATCH:
        cell = f"comparator-sweep-prefill-{COMPARATOR_SWEEP_DEPTH}-ub{ubatch}"
        llama = [
            args.llama_bench, "-m", args.model, "-r", "5", "-t", "20", "-ngl", "99",
            "-b", "2048", "-ub", str(ubatch), "-ctk", "f16", "-ctv", "f16", "-fa", "1",
            "-o", "jsonl", "-p", str(COMPARATOR_SWEEP_DEPTH), "-n", "0",
            "--prompt-token-fixture", prompt_fixture,
        ]
        records.append({
            "cell": cell,
            "surface": "prefill",
            "depth": COMPARATOR_SWEEP_DEPTH,
            "ubatch": ubatch,
            "order": ["llama"],
            "commands": [wrapped(args.identity, cell, args.out_dir, llama)],
            "expected": {"llama_repetitions": 5},
        })
    return records


def ttft_cells(args: argparse.Namespace) -> list[dict[str, object]]:
    output = []
    for index, depth in enumerate(TTFT):
        prompt = str(Path(args.out_dir) / "fixtures" / f"prompt-{depth}.txt")
        engines = [
            ("muser", args.muser_url, 5),
            ("llama", args.llama_url, 5),
        ]
        if index % 4 in (1, 2):
            engines.reverse()
        commands = []
        for engine, url, repetitions in engines:
            server_binary = args.muser_server if engine == "muser" else args.llama_server
            command = [
                "python3", str(ROOT / "scripts" / "bench_server_ttft.py"),
                "--base-url", url, "--model", "muse-glimmer-30b",
                "--server-binary", server_binary, "--model-path", args.model,
                "--prompt-file", prompt, "--depth", str(depth),
                "--repetitions", str(repetitions), "--identity", args.identity,
                "--engine", engine,
            ]
            commands.append(wrapped(args.identity, f"ttft-{depth}-{engine}", args.out_dir, command))
        output.append({
            "cell": f"ttft-{depth}",
            "depth": depth,
            "order": [engine for engine, _, _ in engines],
            "commands": commands,
            "cache": "disabled",
            "measurement": "request-send-complete-to-first-nonempty-sse-content",
        })
    return output


def kvpack_cells(args: argparse.Namespace) -> list[dict[str, object]]:
    output = []
    fixture_dir = Path(args.out_dir) / "fixtures"
    store_base = Path(args.out_dir) / "kvpack-stores" / args.identity.replace(":", "-")
    exact_depths = [8192, 16384, 32768, 65536, 131008]
    ancestor_cuts = [8192, 16384, 32768, 65536, 128768]
    suffixes = [1, 255, 256, 257, 2047]
    for source in ("resident", "durable", "remote"):
        for depth in exact_depths:
            cell = f"kvpack-{source}-exact-{depth}"
            command = [
                args.muser_kvpack, "--model", args.model,
                "--prompt-token-fixture", str(fixture_dir / f"prompt-{depth}.txt"),
                "--source", source, "--lookup", "exact-final", "--suffix", "0",
                "--repetitions", "3", "--identity", args.identity,
            ]
            if source in ("durable", "remote"):
                command.extend(["--store-root", str(store_base / cell)])
            output.append({
                "cell": cell,
                "source": source,
                "lookup": "exact-final",
                "prompt_tokens": depth,
                "suffix": 0,
                "commands": [wrapped(args.identity, cell, args.out_dir, command)],
            })
        for cut in ancestor_cuts:
            for suffix in suffixes:
                depth = cut + suffix
                cell = f"kvpack-{source}-ancestor-{cut}-s{suffix}"
                command = [
                    args.muser_kvpack, "--model", args.model,
                    "--prompt-token-fixture", str(fixture_dir / f"prompt-{depth}.txt"),
                    "--source", source, "--lookup", "deepest-ancestor",
                    "--suffix", str(suffix), "--repetitions", "3",
                    "--identity", args.identity,
                ]
                if source in ("durable", "remote"):
                    command.extend(["--store-root", str(store_base / cell)])
                output.append({
                    "cell": cell,
                    "source": source,
                    "lookup": "deepest-ancestor",
                    "published_cut": cut,
                    "prompt_tokens": depth,
                    "suffix": suffix,
                    "commands": [wrapped(args.identity, cell, args.out_dir, command)],
                })
    return output


def vision_cells(args: argparse.Namespace) -> list[dict[str, object]]:
    output = []
    fixture_dir = Path(args.out_dir) / "fixtures"
    for index, fixture in enumerate(VISION_FIXTURES):
        image = str(fixture_dir / f"vision-{fixture}.png")
        qualifier = [
            args.muser_vision, "--model", args.model,
            "--mmproj", args.mmproj, "--mtmd-bridge", args.mtmd_bridge,
            "--image", image, "--fixture", fixture,
            "--repetitions", "3", "--output-tokens", "64",
            "--target-backend", "metal", "--identity", args.identity,
        ]
        servers = [
            ("muser", args.muser_url, 3),
            ("llama", args.llama_url, 5),
        ]
        if index % 2:
            servers.reverse()
        commands = [
            wrapped(args.identity, f"vision-{fixture}-qualifier", args.out_dir, qualifier)
        ]
        for engine, url, repetitions in servers:
            command = [
                "python3", str(ROOT / "scripts" / "bench_vision_server.py"),
                "--base-url", url, "--server-binary",
                args.muser_server if engine == "muser" else args.llama_server,
                "--model-path", args.model, "--mmproj", args.mmproj,
                "--model", "muse-glimmer-30b",
                "--image", image, "--fixture", fixture,
                "--repetitions", str(repetitions), "--identity", args.identity,
                "--engine", engine,
            ]
            if engine == "muser":
                command.extend(["--mtmd-bridge", args.mtmd_bridge])
            commands.append(
                wrapped(args.identity, f"vision-{fixture}-{engine}", args.out_dir, command)
            )
        output.append({
            "cell": f"vision-{fixture}",
            "fixture": fixture,
            "image": image,
            "order": ["qualifier", *[engine for engine, _, _ in servers]],
            "commands": commands,
            "correctness": {
                "pixel_error_max": 1 / 255,
                "embedding_cosine_min": 0.999,
                "embedding_relative_l2_max": 0.01,
                "decoder_tokens": 64,
            },
        })
    return output


def dflash_tune_cells(args: argparse.Namespace) -> list[dict[str, object]]:
    output = []
    fixture_dir = Path(args.out_dir) / "fixtures"
    for depth in DFLASH_TUNE_DEPTHS:
        for variant in (1, 2):
            for verify_length in (3, 7, 15):
                cell = f"dflash-tune-{depth}-p{variant}-v{verify_length}"
                command = [
                    args.muser_dflash, "--model", args.model,
                    "--dflash", args.dflash,
                    "--prompt-token-fixture",
                    str(fixture_dir / f"dflash-tune-{depth}-p{variant}.txt"),
                    "--repetitions", "3", "--output-tokens", "256",
                    "--verify-length", str(verify_length),
                    "--target-backend", "metal", "--assistant-backend", "metal",
                    "--identity", args.identity,
                ]
                output.append({
                    "cell": cell,
                    "phase": "tune",
                    "prompt_tokens": depth,
                    "prompt_variant": variant,
                    "verify_length": verify_length,
                    "commands": [wrapped(args.identity, cell, args.out_dir, command)],
                })
    return output


def dflash_cells(args: argparse.Namespace, verify_length: int = 7) -> list[dict[str, object]]:
    output = []
    fixture_dir = Path(args.out_dir) / "fixtures"
    index = 0
    for depth in DFLASH_DEPTHS:
        for variant in (1, 2):
            cell = f"dflash-{depth}-p{variant}"
            prompt = str(fixture_dir / f"dflash-{depth}-p{variant}.txt")
            muser = [
                args.muser_dflash, "--model", args.model,
                "--dflash", args.dflash,
                "--prompt-token-fixture", prompt,
                "--repetitions", "5", "--output-tokens", "256",
                "--verify-length", str(verify_length),
                "--target-backend", "metal", "--assistant-backend", "metal",
                "--identity", args.identity,
            ]
            llama = [
                "python3", str(ROOT / "scripts" / "bench_llama_dflash.py"),
                "--server-binary", args.llama_server, "--model", args.model,
                "--dflash", args.dflash, "--prompt-token-fixture", prompt,
                "--depth", str(depth), "--repetitions", "5", "--output-tokens", "256",
                "--verify-length", str(verify_length), "--identity", args.identity,
                "--base-url", args.llama_url,
            ]
            engines = [("muser", muser), ("llama", llama)]
            if index % 4 in (1, 2):
                engines.reverse()
            output.append({
                "cell": cell,
                "phase": "qualification",
                "prompt_tokens": depth,
                "prompt_variant": variant,
                "verify_length": verify_length,
                "order": [name for name, _ in engines],
                "commands": [
                    wrapped(args.identity, f"{cell}-{name}", args.out_dir, command)
                    for name, command in engines
                ],
            })
            index += 1
    return output


NATURAL_FIXTURE = ROOT / "scripts" / "fixtures" / "dflash_natural.txt"


def dflash_natural_cells(args: argparse.Namespace, verify_length: int = 7) -> list[dict[str, object]]:
    """Natural-prose companion to dflash_cells' periodic synthetic fixture.

    A 9-unique-token periodic body makes DFlash acceptance trivially
    predictable. These cells run the identical qualifier and comparator
    commands against a checked-in prose fixture (tokenized at campaign time
    with `muser-bench tokenize`) so acceptance-rate evidence reflects
    natural-text entropy. Reported separately from the synthetic-exactness
    cells; never substitutes for them.
    """
    output = []
    fixture_dir = Path(args.out_dir) / "fixtures"
    index = 0
    for depth in DFLASH_DEPTHS:
        cell = f"dflash-natural-{depth}"
        prompt = str(fixture_dir / f"dflash-natural-{depth}.txt")
        muser = [
            args.muser_dflash, "--model", args.model,
            "--dflash", args.dflash,
            "--prompt-token-fixture", prompt,
            "--repetitions", "5", "--output-tokens", "256",
            "--verify-length", str(verify_length),
            "--target-backend", "metal", "--assistant-backend", "metal",
            "--identity", args.identity,
        ]
        llama = [
            "python3", str(ROOT / "scripts" / "bench_llama_dflash.py"),
            "--server-binary", args.llama_server, "--model", args.model,
            "--dflash", args.dflash, "--prompt-token-fixture", prompt,
            "--depth", str(depth), "--repetitions", "5", "--output-tokens", "256",
            "--verify-length", str(verify_length), "--identity", args.identity,
            "--base-url", args.llama_url,
        ]
        engines = [("muser", muser), ("llama", llama)]
        if index % 4 in (1, 2):
            engines.reverse()
        output.append({
            "cell": cell,
            "phase": "qualification-natural",
            "prompt_tokens": depth,
            "prompt_source": str(NATURAL_FIXTURE),
            "verify_length": verify_length,
            "order": [name for name, _ in engines],
            "commands": [
                wrapped(args.identity, f"{cell}-{name}", args.out_dir, command)
                for name, command in engines
            ],
        })
        index += 1
    return output


def ane_cells(args: argparse.Namespace, verify_length: int = 7) -> list[dict[str, object]]:
    output = []
    fixture_dir = Path(args.out_dir) / "fixtures"
    for depth in DFLASH_DEPTHS:
        for variant in (1, 2):
            cell = f"ane-{depth}-p{variant}"
            command = [
                args.muser_ane, "--model", args.model,
                "--dflash", args.dflash, "--manifest", args.ane_manifest,
                "--compute-plan-receipt", args.coreml_plan_receipt,
                "--prompt-token-fixture",
                str(fixture_dir / f"dflash-{depth}-p{variant}.txt"),
                "--repetitions", "3", "--output-tokens", "256",
                "--verify-length", str(verify_length), "--identity", args.identity,
            ]
            output.append({
                "cell": cell,
                "phase": "ane-qualification",
                "prompt_tokens": depth,
                "prompt_variant": variant,
                "verify_length": verify_length,
                "commands": [wrapped(args.identity, cell, args.out_dir, command)],
            })
    return output


def remote_cells(args: argparse.Namespace, verify_length: int = 7) -> list[dict[str, object]]:
    output = []
    fixture_dir = Path(args.out_dir) / "fixtures"
    for depth in REMOTE:
        for variant in REMOTE_VARIANTS:
            cell = f"remote-{variant}-{depth}"
            output_tokens = 48 if depth == 131008 else 256
            command = [
                args.muser_remote, "--model", args.model,
                "--prompt-token-fixture", str(fixture_dir / f"prompt-{depth}.txt"),
                "--cluster-config", args.cluster_config,
                "--variant", variant, "--repetitions", "3",
                "--output-tokens", str(output_tokens), "--verify-length", str(verify_length),
                "--identity", args.identity,
            ]
            output.append({
                "cell": cell,
                "phase": "remote-qualification",
                "prompt_tokens": depth,
                "output_tokens": output_tokens,
                "variant": variant,
                "verify_length": verify_length,
                "commands": [wrapped(args.identity, cell, args.out_dir, command)],
            })
    return output


def main() -> int:
    args = parse_args()
    baseline_cells = cells(args)
    server_ttft_cells = ttft_cells(args)
    prefix_cells = kvpack_cells(args)
    vision_qualification = vision_cells(args)
    dflash_tuning = dflash_tune_cells(args)
    dflash_qualification = dflash_cells(args)
    dflash_natural_qualification = dflash_natural_cells(args)
    ane_qualification = ane_cells(args)
    remote_qualification = remote_cells(args)
    sweep_cells = comparator_sweep_cells(args) if args.comparator_sweep else []
    report = {
        "mode": "dry-run",
        "accelerator_touched": False,
        "identity": args.identity,
        "model": args.model,
        "workload": {
            "schema": "decimal-u32-lines-v1",
            "generator": "muser-cyclic-body-v1",
            "decode_tokens": 64,
            "fixture_directory": str(Path(args.out_dir) / "fixtures"),
        },
        # Kept as a top-level compatibility alias for existing packet readers.
        "cells": baseline_cells,
        "lanes": {
            "correctness": {
                "required": True,
                "status": "numerical-and-long-greedy-executors-ready-unsealed",
                "greedy_tokens": 64,
                "depths": [
                    128, 512, 2047, 2048, 2049,
                    8192, 16384, 32768, 65536, 131008,
                ],
                "fixtures": ["diverse-p1", "diverse-p2", "diverse-p3", "swa-crossing", "long"],
                "numerical_rows": 32,
                "long_greedy_cases": 11,
                "snapshot_replay_depths": [8192, 16384, 32768, 65536, 131008],
            },
            "synthetic_baseline": {
                "required": True,
                "status": "executor-ready-metal-reference-unsealed",
                "cells": baseline_cells,
            },
            "warm_server_ttft": {
                "required": True,
                "status": "executor-ready-unsealed",
                "depths": TTFT,
                "repetitions": {"muser": 5, "llama": 5},
                "measurement": "request-complete-to-first-SSE-token",
                "cache": "disabled",
                "cells": server_ttft_cells,
            },
            "vision": {
                "required": True,
                "status": "executor-ready-pinned-official-artifact-structurally-qualified-unsealed",
                "fixtures": VISION_FIXTURES,
                "cells": vision_qualification,
            },
            "kvpack": {
                "required": True,
                "status": "real-model-executor-ready-unsealed",
                "sources": ["resident", "durable", "remote"],
                "lookups": ["exact-final", "deepest-ancestor"],
                "suffix_lengths": [1, 255, 256, 257, 2047],
                "depths": [8192, 16384, 32768, 65536, 131008],
                "cells": prefix_cells,
            },
            "dflash_metal": {
                "required": True,
                "status": "paired-executor-ready-pinned-official-artifact-structurally-qualified-unsealed",
                "prompts": 8,
                "depths": 4,
                "output_tokens": 256,
                "verification_lengths": [3, 7, 15],
                "tuning_cells": dflash_tuning,
                "qualification_cells": dflash_qualification,
                "natural_fixture_qualification_cells": dflash_natural_qualification,
                "natural_fixture_source": str(NATURAL_FIXTURE),
            },
            "dflash_ane": {
                "required": False,
                "status": "experimental-post-release-excluded-from-v0.1",
                "compute_units": "CPU_AND_NE",
                "prompts": 8,
                "depths": 4,
                "output_tokens": 256,
                "qualification_cells": ane_qualification,
            },
            "remote_prefill": {
                "required": True,
                "status": "executor-ready-live-host-evidence-required-unsealed",
                "contexts": REMOTE,
                "output_tokens_by_depth": {
                    str(depth): 48 if depth == 131008 else 256 for depth in REMOTE
                },
                "cold_repetitions": 3,
                "variants": list(REMOTE_VARIANTS),
                "cells": remote_qualification,
            },
        },
        "seal": {
            "eligible": False,
            "reason": "mandatory qualification packets are incomplete",
            "mixed_routes_allowed": False,
            "partial_packets_allowed": False,
        },
        "comparator_sweep": {
            "enabled": args.comparator_sweep,
            "required": False,
            "note": "llama-only -ub sensitivity check; never gates the seal",
            "ubatch_values": COMPARATOR_SWEEP_UBATCH,
            "depth": COMPARATOR_SWEEP_DEPTH,
            "cells": sweep_cells,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
