#!/usr/bin/env python3
"""Freshly re-evaluate and atomically expose one complete final seal bundle."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from release_identity import identity, parse_named, sha256
from release_lock import ReleaseLocked, require_sealing_enabled
from release_readiness import (
    MANDATORY,
    atomic_json,
    bind_lane_report,
    substitute_lane_command,
    validate_lane_report,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    mode = value.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="nonvalidating preview only")
    mode.add_argument("--check", action="store_true", help="complete read-only validation")
    mode.add_argument("--dry-run", action="store_true", help="alias for --check")
    value.add_argument("--readiness", type=Path, required=True)
    value.add_argument("--matrix-config", type=Path, required=True)
    value.add_argument("--binary", action="append", default=[], metavar="NAME=PATH")
    value.add_argument("--out", type=Path, required=True)
    return value


def closed_argv(record: object, field: str, lane: str) -> list[str]:
    argv = record.get(field) if isinstance(record, dict) else None
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
        raise RuntimeError(f"lane {lane} has no closed {field} list")
    return argv


def substitute(argv: list[str], campaign: str, output: Path | None) -> list[str]:
    values = [item.replace("{identity}", campaign) for item in argv]
    if output is not None:
        values = [
            item.replace("{output_dir}", str(output.parent)).replace(
                "{output}", str(output)
            )
            for item in values
        ]
    if any(
        "{identity}" in item
        or "{output}" in item
        or "{output_dir}" in item
        for item in values
    ):
        raise RuntimeError("unresolved final-campaign command placeholder")
    return values


def validate(args: argparse.Namespace) -> tuple[dict, dict, dict]:
    campaign = identity(parse_named(args.binary))
    if not args.readiness.is_file() or args.readiness.is_symlink():
        raise RuntimeError("release-readiness receipt is missing or unsafe")
    if not args.matrix_config.is_file() or args.matrix_config.is_symlink():
        raise RuntimeError("final matrix configuration is missing or unsafe")
    readiness = json.loads(args.readiness.read_text(encoding="utf-8"))
    if (
        readiness.get("schema") != "muser.release-readiness.v1"
        or readiness.get("status") != "passed"
        or readiness.get("identity") != campaign["digest"]
    ):
        raise RuntimeError("readiness receipt does not authorize this exact identity")
    if readiness.get("matrix_config_sha256") != sha256(args.matrix_config):
        raise RuntimeError("final matrix configuration differs from readiness")
    config = json.loads(args.matrix_config.read_text(encoding="utf-8"))
    if config.get("schema") != "muser.unsealed-matrix-config.v1":
        raise RuntimeError("unsupported final matrix configuration")
    lanes = config.get("lanes")
    if not isinstance(lanes, dict) or set(lanes) != MANDATORY:
        raise RuntimeError(
            f"final lane set mismatch; missing={sorted(MANDATORY - set(lanes or {}))} "
            f"extra={sorted(set(lanes or {}) - MANDATORY)}"
        )
    if set(readiness.get("lanes", {})) != MANDATORY:
        raise RuntimeError("readiness receipt does not contain every mandatory lane")
    for lane, lane_receipt in readiness["lanes"].items():
        if not isinstance(lane_receipt, dict):
            raise RuntimeError(f"readiness lane {lane} has no retained report binding")
        report_path = Path(str(lane_receipt.get("path", "")))
        if (
            not report_path.is_file()
            or report_path.is_symlink()
            or lane_receipt.get("sha256") != sha256(report_path)
        ):
            raise RuntimeError(f"readiness lane {lane} report is missing or changed")
        template = closed_argv(lanes[lane], "argv", lane)
        provenance = json.loads(report_path.read_text(encoding="utf-8")).get(
            "execution_provenance", {}
        )
        validate_lane_report(
            report_path,
            lane,
            campaign["digest"],
            matrix_config_sha256=(
                sha256(args.matrix_config)
                if provenance.get("matrix_config_sha256") is not None
                else None
            ),
            lane_config=lanes[lane],
            command_template=template,
            command=substitute_lane_command(
                template,
                campaign["digest"],
                Path(provenance.get("evaluator_output", str(report_path.absolute()))),
            ),
            log_path=report_path.with_name(f"{lane}.log"),
            runner=lanes[lane].get(
                "readiness_runners", ["scripts/run_unsealed_release_matrix.py"]
            ),
        )
    for lane, record in lanes.items():
        check = closed_argv(record, "check_argv", lane)
        execute = closed_argv(record, "argv", lane)
        if any("{output}" in item for item in check):
            raise RuntimeError(f"lane {lane} read-only check may not bind an output")
        if not any("{output}" in item for item in execute):
            raise RuntimeError(f"lane {lane} final command does not bind its exact output")
        substitute(check, campaign["digest"], None)
        substitute(execute, campaign["digest"], Path("FINAL-OUTPUT"))
    return readiness, config, campaign


def run_checks(config: dict, campaign: dict) -> None:
    for lane in sorted(MANDATORY):
        command = substitute(
            closed_argv(config["lanes"][lane], "check_argv", lane),
            campaign["digest"],
            None,
        )
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"lane {lane} read-only validation failed with exit {completed.returncode}")


def fsync_tree(root: Path) -> None:
    for path in sorted((item for item in root.rglob("*") if item.is_file())):
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    directories = [item for item in root.rglob("*") if item.is_dir()]
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True) + [root]:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def run_campaign(args: argparse.Namespace, readiness: dict, config: dict, campaign: dict) -> dict:
    require_sealing_enabled("atomic final seal campaign")
    if args.out.exists():
        raise RuntimeError(f"refusing to overwrite final seal bundle: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.out.name}.tmp-", dir=args.out.parent))
    config_sha256 = sha256(args.matrix_config)
    try:
        lane_dir = temporary / "lanes"
        log_dir = temporary / "logs"
        lane_dir.mkdir()
        log_dir.mkdir()
        hashes: dict[str, str] = {}
        for lane in sorted(MANDATORY):
            output = lane_dir / f"{lane}.json"
            command_template = closed_argv(config["lanes"][lane], "argv", lane)
            command = substitute(
                command_template,
                campaign["digest"],
                output,
            )
            log_path = log_dir / f"{lane}.log"
            with log_path.open("xb") as log:
                completed = subprocess.run(
                    command,
                    cwd=Path(__file__).resolve().parents[1],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                log.flush()
                os.fsync(log.fileno())
            if completed.returncode != 0:
                raise RuntimeError(f"fresh lane {lane} failed with exit {completed.returncode}")
            bind_lane_report(
                output,
                lane,
                campaign["digest"],
                matrix_config_sha256=config_sha256,
                lane_config=config["lanes"][lane],
                command_template=command_template,
                command=command,
                log_path=log_path,
                runner="scripts/atomic_seal_campaign.py",
            )
            validate_lane_report(
                output,
                lane,
                campaign["digest"],
                matrix_config_sha256=config_sha256,
                lane_config=config["lanes"][lane],
                command_template=command_template,
                command=command,
                log_path=log_path,
                runner="scripts/atomic_seal_campaign.py",
            )
            hashes[lane] = sha256(output)

        receipt_target = temporary / "release-readiness.json"
        config_target = temporary / "matrix-config.json"
        shutil.copyfile(args.readiness, receipt_target)
        shutil.copyfile(args.matrix_config, config_target)
        result = {
            "schema": "muser.final-seal-result.v1",
            "status": "passed",
            "mode": "seal",
            "published": True,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "identity": campaign["digest"],
            "readiness_sha256": sha256(receipt_target),
            "matrix_config_sha256": sha256(config_target),
            "lanes": hashes,
            "fresh_re_evaluation": True,
        }
        atomic_json(temporary / "RESULT.json", result)
        manifest = {
            "schema": "muser.atomic-seal-bundle.v1",
            "identity": campaign,
            "readiness": readiness["identity"],
            "members": {
                str(path.relative_to(temporary)): sha256(path)
                for path in sorted(temporary.rglob("*"))
                if path.is_file() and path.name != "MANIFEST.json"
            },
        }
        atomic_json(temporary / "MANIFEST.json", manifest)
        fsync_tree(temporary)
        os.rename(temporary, args.out)
        descriptor = os.open(args.out.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return result
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    args = parser().parse_args()
    if args.plan:
        print(json.dumps({
            "mode": "plan", "validating": False, "out": str(args.out),
            "readiness": str(args.readiness), "matrix_config": str(args.matrix_config),
        }, indent=2, sort_keys=True))
        return 0
    try:
        readiness, config, campaign = validate(args)
        if args.check or args.dry_run:
            run_checks(config, campaign)
            print(json.dumps({
                "schema": "muser.final-seal-result.v1", "status": "passed",
                "mode": "check", "published": False, "identity": campaign["digest"],
                "all_lane_checks_executed": True,
            }, indent=2, sort_keys=True))
            return 0
        result = run_campaign(args, readiness, config, campaign)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError, ReleaseLocked) as error:
        print(f"final seal blocked: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"final seal internal error: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
