#!/usr/bin/env python3
"""Bind one retained native-text remote packet without rerunning its GPU cells."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any
from urllib.parse import unquote, urlparse

from release_identity import sha256
from release_readiness import (
    atomic_json,
    bind_lane_report,
    lane_execution_config,
    substitute_lane_command,
    validate_lane_report,
    validate_lane_payload,
)
from run_nvfp4_text_matrix import (
    COUNTED_REPETITIONS,
    DEPTHS,
    LINK_GBPS_MINIMUM,
    TTFT_CV_MAXIMUM,
    json_records,
    output_tokens,
    validate_fast_records,
)


ROOT = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source", type=Path, required=True)
    execution = value.add_mutually_exclusive_group(required=True)
    execution.add_argument("--execution-record", type=Path)
    execution.add_argument("--session-log", type=Path)
    value.add_argument("--session-ordinal", type=int)
    value.add_argument("--lane-config", type=Path)
    value.add_argument("--out", type=Path, required=True)
    return value


def safe_file(raw: object, *, parent: Path | None = None) -> Path:
    if not isinstance(raw, str):
        raise RuntimeError("retained evidence path is not a string")
    path = Path(raw)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise RuntimeError(f"retained evidence path is missing or unsafe: {path}")
    resolved = path.resolve()
    if parent is not None and resolved.parent != parent.resolve():
        raise RuntimeError(f"retained evidence escaped its cell directory: {path}")
    return resolved


def checked_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"retained JSON is not an object: {path}")
    return value


def command_value(command: list[str], flag: str) -> str:
    try:
        index = command.index(flag)
        return command[index + 1]
    except (ValueError, IndexError) as error:
        raise RuntimeError(f"retained accelerator command lacks {flag}") from error


def validate_retained_packet(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"source report is missing or unsafe: {path}")
    report = checked_json(path)
    identity = report.get("identity")
    if not isinstance(identity, str):
        raise RuntimeError("source report has no identity")
    validate_lane_payload(report, "remote", identity)
    if (
        report.get("variant") != "text"
        or report.get("depths") != list(DEPTHS)
        or report.get("counted_repetitions") != COUNTED_REPETITIONS
        or report.get("warmup_repetitions_per_depth") != 1
        or report.get("ttft_cv_maximum") != TTFT_CV_MAXIMUM
        or report.get("installed_payload_gbps_minimum") != LINK_GBPS_MINIMUM
        or report.get("release_lock_state") != "containment"
    ):
        raise RuntimeError("source report is outside the native-text packet contract")
    evidence_dir = Path(str(report.get("evidence_dir", "")))
    if (
        not evidence_dir.is_absolute()
        or not evidence_dir.is_dir()
        or evidence_dir.is_symlink()
    ):
        raise RuntimeError("source report has an unsafe evidence directory")
    cells = report.get("cells")
    if not isinstance(cells, dict) or set(cells) != {str(depth) for depth in DEPTHS}:
        raise RuntimeError("source report has an incomplete depth set")

    next_generation = report.get("initial_replay_high_water")
    if not isinstance(next_generation, int):
        raise RuntimeError("source report has no initial replay watermark")
    checked_cells: dict[str, Any] = {}
    for depth in DEPTHS:
        cell = cells[str(depth)]
        if not isinstance(cell, dict):
            raise RuntimeError(f"depth {depth} receipt is not an object")
        cell_dir = (evidence_dir / f"remote-text-{depth}").resolve()
        if not cell_dir.is_dir() or cell_dir.is_symlink():
            raise RuntimeError(f"depth {depth} evidence directory is unsafe")
        accelerator_path = safe_file(cell.get("accelerator_result"), parent=cell_dir)
        log_path = safe_file(cell.get("command_log"), parent=cell_dir)
        control_path = safe_file(cell.get("control"), parent=cell_dir / "qualify")
        for label, retained, evidence_path in (
            (
                "accelerator result",
                cell.get("accelerator_result_sha256"),
                accelerator_path,
            ),
            ("command log", cell.get("command_log_sha256"), log_path),
            ("control", cell.get("control_sha256"), control_path),
        ):
            if retained != sha256(evidence_path):
                raise RuntimeError(f"depth {depth} {label} hash differs")

        accelerator = checked_json(accelerator_path)
        command = accelerator.get("command")
        if not isinstance(command, list) or not all(
            isinstance(value, str) for value in command
        ):
            raise RuntimeError(f"depth {depth} accelerator command is invalid")
        expected_first = next_generation + 1
        if (
            accelerator.get("schema") != "muser.accelerator-result.v1"
            or accelerator.get("identity") != identity
            or accelerator.get("cell") != f"remote-text-{depth}"
            or accelerator.get("exit_status") != 0
            or Path(str(accelerator.get("command_log", ""))).resolve() != log_path
            or command_value(command, "--identity") != identity
            or command_value(command, "--first-generation") != str(expected_first)
            or command_value(command, "--variant") != "text"
            or command_value(command, "--repetitions") != str(COUNTED_REPETITIONS)
            or command_value(command, "--output-tokens") != str(output_tokens(depth))
            or "--performance-only" not in command
        ):
            raise RuntimeError(f"depth {depth} accelerator receipt differs from the packet")

        control = checked_json(control_path)
        if (
            control.get("schema") != "muser.nvfp4-fast-control.v1"
            or control.get("identity") != identity
            or control.get("mode") != "p4"
            or control.get("variant") != "text"
            or control.get("repetitions") != COUNTED_REPETITIONS
            or control.get("output_tokens") != output_tokens(depth)
            or control.get("performance_only") is not True
            or control.get("producer_receipt_profile") != "enrolled"
            or control.get("exit_status") != 0
            or control.get("eee_check_error") is not None
            or "EEE status: disabled" not in str(control.get("eee_before", ""))
            or "EEE status: disabled" not in str(control.get("eee_after", ""))
            or len(control.get("producer_receipts", [])) != COUNTED_REPETITIONS + 1
        ):
            raise RuntimeError(f"depth {depth} control receipt is invalid")
        producer_hashes = []
        for offset, retained in enumerate(control["producer_receipts"]):
            if not isinstance(retained, dict):
                raise RuntimeError(f"depth {depth} producer receipt is invalid")
            producer_path = safe_file(retained.get("path"), parent=control_path.parent)
            producer_sha256 = sha256(producer_path)
            producer = checked_json(producer_path)
            response = producer.get("response")
            producer_receipt = (
                response.get("producer_receipt")
                if isinstance(response, dict)
                else None
            )
            handoff = (
                producer_receipt.get("handoff")
                if isinstance(producer_receipt, dict)
                else None
            )
            if (
                retained.get("sha256") != producer_sha256
                or producer.get("schema") != "muser.spark-nvfp4-prefill-client.v1"
                or not isinstance(response, dict)
                or response.get("status") != "ok"
                or not isinstance(producer_receipt, dict)
                or producer_receipt.get("schema")
                != "muser.spark-nvfp4-prefill.v2"
                or not isinstance(handoff, dict)
                or handoff.get("generation") != expected_first + offset
                or handoff.get("ack") is not True
                or handoff.get("streaming_target") is not True
            ):
                raise RuntimeError(
                    f"depth {depth} producer receipt {offset} is not enrolled v2"
                )
            producer_hashes.append(producer_sha256)

        measured = validate_fast_records(
            json_records(log_path), identity=identity, depth=depth
        )
        for key, value in measured.items():
            if cell.get(key) != value:
                raise RuntimeError(f"depth {depth} retained result differs at {key}")
        expected_last = expected_first + COUNTED_REPETITIONS
        if (
            cell.get("first_generation") != expected_first
            or cell.get("last_generation") != expected_last
        ):
            raise RuntimeError(f"depth {depth} generation range is not contiguous")
        next_generation = expected_last
        checked_cells[str(depth)] = {
            "accelerator_result_sha256": sha256(accelerator_path),
            "command_log_sha256": sha256(log_path),
            "control_sha256": sha256(control_path),
            "producer_receipt_sha256": producer_hashes,
            "first_generation": expected_first,
            "last_generation": expected_last,
        }

    if report.get("final_replay_high_water") != next_generation:
        raise RuntimeError("source report final replay watermark differs from its cells")
    for name in ("node-state-before.json", "node-state-after.json"):
        state_path = safe_file(str(evidence_dir / name), parent=evidence_dir)
        state = checked_json(state_path)
        if (
            state.get("schema") != "muser.gx10-node-state.v1"
            or state.get("status") != "passed"
        ):
            raise RuntimeError(f"retained node-state receipt failed: {name}")
    return {
        "identity": identity,
        "source_report": str(path.resolve()),
        "source_report_sha256": sha256(path),
        "cells": checked_cells,
        "node_state_before_sha256": sha256(evidence_dir / "node-state-before.json"),
        "node_state_after_sha256": sha256(evidence_dir / "node-state-after.json"),
    }


def validate_execution_record(
    path: Path, *, template: list[str], identity: str, source: Path
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("execution record is missing or unsafe")
    record = checked_json(path)
    command = record.get("command")
    if (
        record.get("schema") != "muser.retained-command-execution.v1"
        or record.get("status") != "completed"
        or record.get("exit_code") != 0
        or record.get("cwd") != str(ROOT)
        or not isinstance(command, list)
        or not command
        or not all(isinstance(value, str) for value in command)
        or command != substitute_lane_command(template, identity, source.resolve())
    ):
        raise RuntimeError("execution record does not bind the successful source command")
    return record


def execution_from_session(path: Path, ordinal: int | None) -> dict[str, Any]:
    if ordinal is None:
        raise RuntimeError("--session-ordinal is required with --session-log")
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("session log is missing or unsafe")
    selected: dict[str, Any] | None = None
    selected_line: bytes | None = None
    with path.open("rb") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("ordinal") == ordinal:
                selected = event
                selected_line = line
                break
    if selected is None or selected_line is None:
        raise RuntimeError(f"session ordinal {ordinal} was not found")
    item = selected.get("payload", {}).get("item", {})
    cwd = item.get("cwd")
    if isinstance(cwd, str) and cwd.startswith("file:"):
        cwd = unquote(urlparse(cwd).path)
    record = {
        "schema": "muser.retained-command-execution.v1",
        "execution_id": item.get("id"),
        "command": item.get("command"),
        "cwd": cwd,
        "status": item.get("status"),
        "exit_code": item.get("exit_code"),
        "duration": item.get("duration"),
        "recorded_at": selected.get("timestamp"),
        "source_session": str(path.resolve()),
        "source_ordinal": ordinal,
        "source_event_sha256": hashlib.sha256(selected_line).hexdigest(),
    }
    return record


def derive_lane_config(
    execution: dict[str, Any], *, identity: str, source: Path, evidence_dir: Path
) -> dict[str, Any]:
    command = execution.get("command")
    if not isinstance(command, list) or not all(
        isinstance(value, str) for value in command
    ):
        raise RuntimeError("execution record has no command to template")
    template = []
    replacements = {
        identity: "{identity}",
        str(source.resolve()): "{output}",
        str(evidence_dir.resolve()): "{output_dir}/remote-evidence",
    }
    observed = {value: 0 for value in replacements}
    for argument in command:
        templated = argument
        for original, replacement in replacements.items():
            count = templated.count(original)
            observed[original] += count
            templated = templated.replace(original, replacement)
        template.append(templated)
    if any(count != 1 for count in observed.values()):
        raise RuntimeError("source command cannot be converted to one exact lane template")
    return {
        "schema": "muser.unsealed-lane-config.v1",
        "lane": "remote",
        "argv": template,
        "readiness_runners": [
            "scripts/run_unsealed_release_matrix.py",
            "scripts/run_nvfp4_text_matrix.py",
        ],
    }


def copy_new(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to replace bound receipt: {target}")
    with source.open("rb") as incoming, temporary.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    temporary.replace(target)
    descriptor = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    args = parser().parse_args()
    try:
        packet = validate_retained_packet(args.source)
        if args.session_log is not None:
            execution = execution_from_session(args.session_log, args.session_ordinal)
            execution_path = args.out.with_name("remote-execution-record.json")
            atomic_json(execution_path, execution)
        else:
            if args.session_ordinal is not None:
                raise RuntimeError("--session-ordinal requires --session-log")
            assert args.execution_record is not None
            execution_path = args.execution_record
            execution = checked_json(execution_path)
        if args.lane_config is None:
            source_report = checked_json(args.source)
            lane_record = derive_lane_config(
                execution,
                identity=packet["identity"],
                source=args.source,
                evidence_dir=Path(source_report["evidence_dir"]),
            )
            lane_config_path = args.out.with_name("remote-lane-config.json")
            atomic_json(lane_config_path, lane_record)
        else:
            lane_config_path = args.lane_config
            lane_record = checked_json(lane_config_path)
        if lane_record.get("lane") not in (None, "remote"):
            raise RuntimeError("lane config is not for the remote lane")
        execution_config = lane_execution_config(lane_record)
        execution = validate_execution_record(
            execution_path,
            template=execution_config["argv"],
            identity=packet["identity"],
            source=args.source,
        )
        log_path = args.out.with_name("remote.log")
        atomic_json(
            log_path,
            {
                "schema": "muser.retained-lane-binding-log.v1",
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "lane": "remote",
                "status": "passed",
                "packet_validation": packet,
                "lane_config": str(lane_config_path.resolve()),
                "lane_config_sha256": sha256(lane_config_path),
                "execution_record": str(execution_path.resolve()),
                "execution_record_sha256": sha256(execution_path),
                "execution_id": execution.get("execution_id"),
                "accelerator_rerun": False,
            },
        )
        copy_new(args.source, args.out)
        bind_lane_report(
            args.out,
            "remote",
            packet["identity"],
            matrix_config_sha256=None,
            lane_config=lane_record,
            command_template=execution_config["argv"],
            command=execution["command"],
            log_path=log_path,
            runner="scripts/run_nvfp4_text_matrix.py",
            evaluator_output_path=args.source,
        )
        validate_lane_report(
            args.out,
            "remote",
            packet["identity"],
            lane_config=lane_record,
            command_template=execution_config["argv"],
            command=execution["command"],
            log_path=log_path,
            runner=execution_config["readiness_runners"],
        )
        print(json.dumps(checked_json(args.out), indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as error:
        print(f"remote receipt binding failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
