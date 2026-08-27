#!/usr/bin/env python3
"""Build the frozen native-versus-exact NVFP4 drift fixture packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any


SCHEMA = "muser.nvfp4-drift-fixtures.v1"
BOS_TOKEN = 200_000
VOCAB_SIZE = 202_048


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_exclusive(path: Path, value: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def tokenize(bench: Path, model: Path, text_path: Path) -> list[int]:
    completed = subprocess.run(
        [str(bench), "tokenize", "--model", str(model), "--file", str(text_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(completed.stdout)
    if value.get("schema") != "muser-bench.tokenize.v1":
        raise RuntimeError("muser-bench returned an unknown tokenizer receipt")
    tokens = value.get("tokens")
    if (
        not isinstance(tokens, list)
        or value.get("token_count") != len(tokens)
        or any(not isinstance(token, int) or not 0 <= token < VOCAB_SIZE for token in tokens)
    ):
        raise RuntimeError("muser-bench returned invalid tokenizer output")
    return tokens


def source_packet(paths: list[Path], character_limit: int) -> tuple[str, list[dict[str, Any]]]:
    chunks: list[str] = []
    sources: list[dict[str, Any]] = []
    size = 0
    for path in paths:
        content = path.read_text(encoding="utf-8", errors="strict")
        header = f"\n\n===== {path.as_posix()} =====\n"
        remaining = character_limit - size
        if remaining <= len(header):
            break
        selected = content[: remaining - len(header)]
        if not selected:
            continue
        chunks.append(header + selected)
        sources.append(
            {
                "path": path.as_posix(),
                "sha256": sha256(path),
                "selected_bytes": len(selected.encode("utf-8")),
                "complete": len(selected) == len(content),
            }
        )
        size += len(header) + len(selected)
        if size >= character_limit:
            break
    return "".join(chunks), sources


def agentic_packet(tasks_path: Path) -> tuple[str, list[str]]:
    rows = []
    ids = []
    for line_number, line in enumerate(tasks_path.read_text(encoding="utf-8").splitlines(), 1):
        task = json.loads(line)
        task_id = task.get("id")
        tools = task.get("tools")
        if not isinstance(task_id, str) or not isinstance(tools, list):
            raise ValueError(f"agentic task line {line_number} has an unknown shape")
        ids.append(task_id)
        rows.append(
            {
                "id": task_id,
                "category": task.get("category"),
                "difficulty": task.get("difficulty"),
                "prompt": task.get("prompt"),
                "tools": [
                    {
                        "name": tool.get("name"),
                        "description": tool.get("description"),
                        "parameters": tool.get("parameters"),
                    }
                    for tool in tools
                ],
            }
        )
    if not rows:
        raise ValueError("agentic golden set is empty")
    return "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", ids


def read_original(path: Path) -> list[int]:
    tokens = [int(value) for value in path.read_bytes().split()]
    if len(tokens) < 2 or any(not 0 <= token < VOCAB_SIZE for token in tokens):
        raise ValueError(f"invalid retained token fixture {path}")
    return tokens


def parse_original(value: str) -> tuple[str, Path]:
    fixture_id, separator, raw_path = value.partition("=")
    if not separator or not fixture_id.replace("-", "").isalnum() or not raw_path:
        raise argparse.ArgumentTypeError("original fixture must be ID=PATH")
    return fixture_id, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--bench", required=True, type=Path)
    parser.add_argument("--agentic-tasks", required=True, type=Path)
    parser.add_argument("--original", required=True, action="append", type=parse_original)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    for path in (args.repo, args.model, args.bench, args.agentic_tasks):
        if not path.exists():
            parser.error(f"required input does not exist: {path}")
    if args.output_dir.exists():
        parser.error("output directory already exists")
    args.output_dir.mkdir(parents=True)

    code_paths = sorted((args.repo / "crates").rglob("*.rs")) + sorted(
        (args.repo / "scripts").rglob("*.py")
    )
    code_text, code_sources = source_packet(code_paths, 96_000)
    long_paths = sorted((args.repo / "docs").rglob("*.md")) + code_paths
    long_text, long_sources = source_packet(long_paths, 240_000)
    agentic_text, agentic_ids = agentic_packet(args.agentic_tasks)
    text_rows = {
        "code": (code_text, 8_192),
        "agentic": (agentic_text, None),
        "long-context": (long_text, 32_768),
    }
    fixtures: list[dict[str, Any]] = []
    build_rows: list[dict[str, Any]] = []
    for fixture_id, (content, target_count) in text_rows.items():
        text_path = args.output_dir / f"{fixture_id}.txt"
        write_exclusive(text_path, content)
        tokens = tokenize(args.bench, args.model, text_path)
        if target_count is not None:
            needed = target_count - 1
            if len(tokens) < needed:
                raise RuntimeError(
                    f"{fixture_id} text produced {len(tokens)} tokens, fewer than {needed}"
                )
            tokens = tokens[:needed]
        tokens = [BOS_TOKEN, *tokens]
        token_path = args.output_dir / f"{fixture_id}.tokens"
        write_exclusive(token_path, " ".join(map(str, tokens)) + "\n")
        fixtures.append(
            {
                "id": fixture_id,
                "regime": fixture_id,
                "token_file": token_path.name,
                "output_tokens": 256,
            }
        )
        build_rows.append(
            {
                "id": fixture_id,
                "text_file": text_path.name,
                "text_sha256": sha256(text_path),
                "token_file": token_path.name,
                "token_file_sha256": sha256(token_path),
                "token_count": len(tokens),
            }
        )

    original_ids: set[str] = set()
    for fixture_id, source in args.original:
        if fixture_id in original_ids or fixture_id in text_rows:
            parser.error(f"duplicate fixture id: {fixture_id}")
        original_ids.add(fixture_id)
        tokens = read_original(source)
        destination = args.output_dir / f"{fixture_id}.tokens"
        shutil.copyfile(source, destination)
        fixtures.append(
            {
                "id": fixture_id,
                "regime": "original",
                "token_file": destination.name,
                "output_tokens": 256,
            }
        )
        build_rows.append(
            {
                "id": fixture_id,
                "source": str(source),
                "source_sha256": sha256(source),
                "token_file": destination.name,
                "token_file_sha256": sha256(destination),
                "token_count": len(tokens),
            }
        )

    manifest = {"schema": SCHEMA, "fixtures": fixtures}
    manifest_path = args.output_dir / "manifest.json"
    write_exclusive(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema": "muser.nvfp4-drift-fixture-build.v1",
        "manifest": manifest_path.name,
        "manifest_sha256": sha256(manifest_path),
        "tokenizer_model": str(args.model),
        "tokenizer_model_sha256": sha256(args.model),
        "agentic_tasks": str(args.agentic_tasks),
        "agentic_tasks_sha256": sha256(args.agentic_tasks),
        "agentic_task_ids": agentic_ids,
        "code_sources": code_sources,
        "long_context_sources": long_sources,
        "fixtures": build_rows,
        "seal_eligible": False,
    }
    receipt_path = args.output_dir / "build-receipt.json"
    write_exclusive(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "receipt": str(receipt_path)}))


if __name__ == "__main__":
    main()
