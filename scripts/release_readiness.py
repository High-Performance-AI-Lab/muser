#!/usr/bin/env python3
"""Emit release-readiness only from one clean identity and every passing unsealed lane."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys

from release_identity import identity, parse_named, sha256

ROOT = Path(__file__).resolve().parents[1]
MANDATORY = {
    "correctness", "sampled", "greedy", "kvpack", "session", "vision",
    "baseline", "dflash", "remote", "serving", "onboarding",
    "api-parity", "continuous-batching", "migration", "security",
}


def command_sha256(command: list[str]) -> str:
    encoded = json.dumps(command, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def lane_execution_config(record: dict) -> dict:
    """Return the stable, execution-affecting part of one reviewed lane record."""
    template = record.get("argv")
    if not isinstance(template, list) or not template or not all(
        isinstance(value, str) for value in template
    ):
        raise RuntimeError("release lane has no closed command template")
    runners = record.get(
        "readiness_runners", ["scripts/run_unsealed_release_matrix.py"]
    )
    if not isinstance(runners, list) or not runners or not all(
        isinstance(value, str) and value for value in runners
    ):
        raise RuntimeError("release lane has no closed readiness runner list")
    return {"argv": template, "readiness_runners": runners}


def lane_execution_config_sha256(record: dict) -> str:
    encoded = json.dumps(
        lane_execution_config(record),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def substitute_lane_command(template: list[str], campaign: str, output: Path) -> list[str]:
    if not template or not all(isinstance(value, str) for value in template):
        raise RuntimeError("release lane has no closed command template")
    if not any("{output}" in value for value in template):
        raise RuntimeError("release lane command does not bind its exact output")
    command = []
    for value in template:
        resolved = value.replace("{identity}", campaign)
        resolved = resolved.replace("{output_dir}", str(output.parent))
        resolved = resolved.replace("{output}", str(output))
        command.append(resolved)
    if any(
        "{identity}" in value
        or "{output}" in value
        or "{output_dir}" in value
        for value in command
    ):
        raise RuntimeError("unresolved release lane command placeholder")
    return command


def validate_lane_payload(report: dict, lane: str, campaign: str) -> None:
    if lane not in MANDATORY:
        raise RuntimeError(f"unknown release lane: {lane}")
    if (
        report.get("schema") != "muser.unsealed-qualification.v1"
        or report.get("lane") != lane
        or report.get("status") != "passed"
        or report.get("seal_eligible") is not False
        or report.get("identity") != campaign
    ):
        raise RuntimeError(
            f"lane {lane} is not a passing lane-bound unsealed report for this identity"
        )


def bind_lane_report(
    path: Path,
    lane: str,
    campaign: str,
    *,
    matrix_config_sha256: str | None,
    lane_config: dict | None = None,
    command_template: list[str],
    command: list[str],
    log_path: Path,
    runner: str,
    evaluator_output_path: Path | None = None,
) -> dict:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"lane report is missing or unsafe: {lane}={path}")
    if not log_path.is_file() or log_path.is_symlink():
        raise RuntimeError(f"lane log is missing or unsafe: {lane}={log_path}")
    original_sha256 = sha256(path)
    evaluator_output = (evaluator_output_path or path).absolute()
    if (
        not evaluator_output.is_file()
        or evaluator_output.is_symlink()
        or sha256(evaluator_output) != original_sha256
    ):
        raise RuntimeError("evaluator output differs from the report being bound")
    report = json.loads(path.read_text(encoding="utf-8"))
    validate_lane_payload(report, lane, campaign)
    provenance = {
        "schema": (
            "muser.lane-execution-provenance.v2"
            if lane_config is not None
            else "muser.lane-execution-provenance.v1"
        ),
        "command_template": command_template,
        "command": command,
        "command_sha256": command_sha256(command),
        "evaluator_report_sha256": original_sha256,
        "evaluator_output": str(evaluator_output),
        "log_name": log_path.name,
        "log_sha256": sha256(log_path),
        "runner": runner,
    }
    if matrix_config_sha256 is not None:
        provenance["matrix_config_sha256"] = matrix_config_sha256
    if lane_config is not None:
        if lane_execution_config(lane_config)["argv"] != command_template:
            raise RuntimeError("lane command template differs from its execution config")
        provenance["lane_execution_config_sha256"] = lane_execution_config_sha256(
            lane_config
        )
    report["execution_provenance"] = provenance
    atomic_json(path, report, replace=True)
    return report


def validate_lane_report(
    path: Path,
    lane: str,
    campaign: str,
    *,
    matrix_config_sha256: str | None = None,
    lane_config: dict | None = None,
    command_template: list[str] | None = None,
    command: list[str] | None = None,
    log_path: Path | None = None,
    runner: str | list[str] | tuple[str, ...] | None = None,
) -> dict:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"lane report is missing or unsafe: {lane}={path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    validate_lane_payload(report, lane, campaign)
    provenance = report.get("execution_provenance")
    recorded_command = provenance.get("command") if isinstance(provenance, dict) else None
    if (
        not isinstance(provenance, dict)
        or provenance.get("schema")
        not in {
            "muser.lane-execution-provenance.v1",
            "muser.lane-execution-provenance.v2",
        }
        or not isinstance(recorded_command, list)
        or not recorded_command
        or not all(isinstance(value, str) for value in recorded_command)
        or provenance.get("command_sha256") != command_sha256(recorded_command)
        or not isinstance(provenance.get("evaluator_report_sha256"), str)
        or len(provenance["evaluator_report_sha256"]) != 64
        or (
            provenance.get("schema") == "muser.lane-execution-provenance.v2"
            and (
                not isinstance(provenance.get("evaluator_output"), str)
                or not Path(provenance["evaluator_output"]).is_absolute()
            )
        )
        or not isinstance(provenance.get("log_sha256"), str)
        or len(provenance["log_sha256"]) != 64
    ):
        raise RuntimeError(f"lane {lane} has missing or invalid execution provenance")
    if provenance.get("schema") == "muser.lane-execution-provenance.v2":
        lane_config_sha256 = provenance.get("lane_execution_config_sha256")
        if not isinstance(lane_config_sha256, str) or len(lane_config_sha256) != 64:
            raise RuntimeError(f"lane {lane} has missing or invalid execution provenance")
    evaluator_output = Path(provenance.get("evaluator_output", str(path.absolute())))
    if evaluator_output != path.absolute() and (
        not evaluator_output.is_file()
        or evaluator_output.is_symlink()
        or sha256(evaluator_output) != provenance["evaluator_report_sha256"]
    ):
        raise RuntimeError(f"lane {lane} retained evaluator output is missing or mismatched")
    if (
        lane_config is not None
        and provenance.get("schema") == "muser.lane-execution-provenance.v2"
        and provenance.get("lane_execution_config_sha256")
        != lane_execution_config_sha256(lane_config)
    ):
        raise RuntimeError(f"lane {lane} belongs to a different lane execution config")
    if matrix_config_sha256 is not None and provenance.get("matrix_config_sha256") != matrix_config_sha256:
        raise RuntimeError(f"lane {lane} belongs to a different matrix configuration")
    if command_template is not None and provenance.get("command_template") != command_template:
        raise RuntimeError(f"lane {lane} command template differs from the reviewed matrix")
    if command is not None and recorded_command != command:
        raise RuntimeError(f"lane {lane} executed a different command")
    allowed_runners = [runner] if isinstance(runner, str) else runner
    if allowed_runners is not None and provenance.get("runner") not in allowed_runners:
        raise RuntimeError(f"lane {lane} was emitted by the wrong matrix runner")
    if log_path is not None:
        if (
            not log_path.is_file()
            or log_path.is_symlink()
            or provenance.get("log_name") != log_path.name
            or provenance.get("log_sha256") != sha256(log_path)
        ):
            raise RuntimeError(f"lane {lane} retained log is missing or mismatched")
    return report


def atomic_json(path: Path, value: dict, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists() and not replace:
        temporary.unlink(missing_ok=True)
        raise FileExistsError(f"refusing to replace existing receipt: {path}")
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--binary", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--matrix-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        lanes = parse_named(args.lane)
        missing = MANDATORY - lanes.keys()
        extra = lanes.keys() - MANDATORY
        if missing or extra:
            raise RuntimeError(f"lane set mismatch; missing={sorted(missing)} extra={sorted(extra)}")
        findings = json.loads((ROOT / "release/findings-v1.json").read_text())
        open_findings = [item["id"] for item in findings["findings"] if item["status"] != "closed"]
        if open_findings:
            raise RuntimeError(f"open findings block readiness: {open_findings}")
        campaign = identity(parse_named(args.binary))
        if not args.matrix_config.is_file() or args.matrix_config.is_symlink():
            raise RuntimeError("release matrix configuration is missing or unsafe")
        matrix = json.loads(args.matrix_config.read_text(encoding="utf-8"))
        if matrix.get("schema") != "muser.unsealed-matrix-config.v1":
            raise RuntimeError("unsupported release matrix configuration")
        if not isinstance(matrix.get("lanes"), dict) or set(matrix["lanes"]) != MANDATORY:
            raise RuntimeError("release matrix configuration does not contain every mandatory lane")
        matrix_sha256 = sha256(args.matrix_config)
        reports = {}
        for name, path in sorted(lanes.items()):
            record = matrix["lanes"][name]
            template = record.get("argv") if isinstance(record, dict) else None
            if not isinstance(template, list):
                raise RuntimeError(f"lane {name} has no reviewed execution command")
            provenance = json.loads(path.read_text(encoding="utf-8")).get(
                "execution_provenance", {}
            )
            matrix_binding = (
                matrix_sha256
                if provenance.get("matrix_config_sha256") is not None
                else None
            )
            validate_lane_report(
                path,
                name,
                campaign["digest"],
                matrix_config_sha256=matrix_binding,
                lane_config=record,
                command_template=template,
                command=substitute_lane_command(
                    template,
                    campaign["digest"],
                    Path(provenance.get("evaluator_output", str(path.absolute()))),
                ),
                log_path=path.with_name(f"{name}.log"),
                runner=lane_execution_config(record)["readiness_runners"],
            )
            reports[name] = {"path": str(path.resolve()), "sha256": sha256(path)}
        receipt = {
            "schema": "muser.release-readiness.v1",
            "status": "passed",
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "identity": campaign["digest"],
            "campaign_identity": campaign,
            "matrix_config": str(args.matrix_config.resolve()),
            "matrix_config_sha256": matrix_sha256,
            "open_findings": [],
            "lanes": reports,
        }
        atomic_json(args.out, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError) as error:
        print(f"release readiness blocked: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
