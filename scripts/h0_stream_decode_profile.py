#!/usr/bin/env python3
"""Summarize opt-in streamed-decode host timing events fail-closed."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
from pathlib import Path

MARKER = "[muser-stream-decode-profile] "
SCHEMA = "muser.stream-decode-profile.v1"
REQUIRED_INTS = (
    "token_index",
    "input_token",
    "engine_argmax_token",
    "decode_total_ns",
    "batcher_unaccounted_ns",
    "emit_ns",
    "model_prepare_ns",
    "model_encode_ns",
    "encoder_end_ns",
    "command_commit_ns",
    "gpu_wait_ns",
    "logits_readback_ns",
    "finite_scan_ns",
    "argmax_ns",
    "result_clone_ns",
)
COMPONENTS = (
    "model_prepare_ns",
    "model_encode_ns",
    "encoder_end_ns",
    "command_commit_ns",
    "gpu_wait_ns",
    "logits_readback_ns",
    "finite_scan_ns",
    "argmax_ns",
    "result_clone_ns",
    "batcher_unaccounted_ns",
    "sampling_after_decode_ns",
    "emit_ns",
    "decode_total_ns",
    "serialized_host_total_ns",
)


def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def parse_events(log_path: Path) -> list[dict[str, object]]:
    if not log_path.is_file() or log_path.is_symlink():
        raise ValueError("server log must be a regular non-symlink file")
    events: list[dict[str, object]] = []
    for line_number, line in enumerate(log_path.read_text(errors="strict").splitlines(), 1):
        if MARKER not in line:
            continue
        payload = line.split(MARKER, 1)[1]
        try:
            value = json.loads(payload, object_pairs_hook=reject_duplicate_keys)
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"invalid profile event at line {line_number}: {error}") from error
        if not isinstance(value, dict) or value.get("schema") != SCHEMA:
            raise ValueError(f"invalid profile schema at line {line_number}")
        expected = {"schema", "session_id", "sampling_after_decode_ns", *REQUIRED_INTS}
        if set(value) != expected:
            raise ValueError(f"profile event has unknown or missing fields at line {line_number}")
        if not isinstance(value["session_id"], str) or not value["session_id"]:
            raise ValueError(f"session_id must be a nonempty string at line {line_number}")
        for field in REQUIRED_INTS:
            value[field] = nonnegative_int(value[field], f"line {line_number} {field}")
        sampling = value["sampling_after_decode_ns"]
        if sampling is not None:
            value["sampling_after_decode_ns"] = nonnegative_int(
                sampling, f"line {line_number} sampling_after_decode_ns"
            )
        events.append(value)
    if not events:
        raise ValueError("server log contains no streamed-decode profile events")
    return events


def select_measured_session(
    events: list[dict[str, object]], minimum_tokens: int
) -> list[dict[str, object]]:
    sessions: dict[str, list[dict[str, object]]] = {}
    for event in events:
        sessions.setdefault(str(event["session_id"]), []).append(event)
    candidates = [values for values in sessions.values() if len(values) >= minimum_tokens]
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one session with >= {minimum_tokens} tokens, got {len(candidates)}"
        )
    selected = candidates[0]
    indexes = [int(event["token_index"]) for event in selected]
    if indexes != list(range(len(selected))):
        raise ValueError("measured profile token indexes are not contiguous from zero")
    if any(event["sampling_after_decode_ns"] is None for event in selected[:-1]):
        raise ValueError("only the final measured token may omit post-decode sampling")
    return selected


def distribution(values: list[int]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty component")
    ordered = sorted(values)
    return {
        "samples": len(values),
        "min_ms": ordered[0] / 1_000_000,
        "median_ms": statistics.median(ordered) / 1_000_000,
        "max_ms": ordered[-1] / 1_000_000,
        "mean_ms": statistics.fmean(ordered) / 1_000_000,
    }


def summarize(events: list[dict[str, object]], minimum_tokens: int) -> dict[str, object]:
    measured = select_measured_session(events, minimum_tokens)
    rows: list[dict[str, int | None]] = []
    for event in measured:
        row = {field: int(event[field]) for field in REQUIRED_INTS}
        row["sampling_after_decode_ns"] = (
            None
            if event["sampling_after_decode_ns"] is None
            else int(event["sampling_after_decode_ns"])
        )
        row["serialized_host_total_ns"] = sum(
            int(row[field])
            for field in (
                "model_prepare_ns",
                "model_encode_ns",
                "encoder_end_ns",
                "command_commit_ns",
                "logits_readback_ns",
                "finite_scan_ns",
                "argmax_ns",
                "result_clone_ns",
                "batcher_unaccounted_ns",
                "emit_ns",
            )
        ) + int(row["sampling_after_decode_ns"] or 0)
        rows.append(row)
    components: dict[str, object] = {}
    for component in COMPONENTS:
        values = [int(row[component]) for row in rows if row[component] is not None]
        components[component] = distribution(values)
    ranked = sorted(
        (
            {
                "component": name,
                "median_ms": float(components[name]["median_ms"]),
            }
            for name in COMPONENTS
            if name not in {"gpu_wait_ns", "decode_total_ns", "serialized_host_total_ns"}
        ),
        key=lambda item: (-item["median_ms"], item["component"]),
    )
    if any(not math.isfinite(item["median_ms"]) for item in ranked):
        raise ValueError("profile summary produced a nonfinite component")
    return {
        "schema": "muser.stream-decode-profile-summary.v1",
        "kind": "non-notarial-diagnostic",
        "accelerator_touched": True,
        "seal_eligible": False,
        "session_id": measured[0]["session_id"],
        "tokens": len(measured),
        "components": components,
        "serialized_rank": ranked,
    }


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-tokens", type=int, default=64)
    args = parser.parse_args()
    if args.minimum_tokens < 64:
        parser.error("--minimum-tokens must be at least 64")
    try:
        result = summarize(parse_events(args.server_log), args.minimum_tokens)
        if args.output.exists():
            raise ValueError("output already exists")
        atomic_json(args.output, result)
    except (OSError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
