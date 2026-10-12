#!/usr/bin/env python3
"""Run every mandatory release lane and retain only exact unsealed reports."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys

from release_identity import identity
from release_readiness import (
    MANDATORY,
    atomic_json,
    bind_lane_report,
    substitute_lane_command,
    validate_lane_report,
)
from release_identity import sha256

ROOT = Path(__file__).resolve().parents[1]


def required_path(variable: str) -> Path:
    raw = os.environ.get(variable)
    if not raw:
        raise RuntimeError(f"{variable} is required")
    return Path(raw).expanduser().resolve()


def main() -> int:
    try:
        results = required_path("MUSER_RELEASE_RESULTS")
        config_path = required_path("MUSER_RELEASE_MATRIX_CONFIG")
        if not results.is_dir() or results.is_symlink():
            raise RuntimeError("release results directory is missing or unsafe")
        config = json.loads(config_path.read_text())
        config_sha256 = sha256(config_path)
        if config.get("schema") != "muser.unsealed-matrix-config.v1":
            raise RuntimeError("unsupported unsealed matrix config")
        lanes = config.get("lanes")
        if not isinstance(lanes, dict) or set(lanes) != MANDATORY:
            raise RuntimeError(
                f"matrix lane set mismatch; missing={sorted(MANDATORY - set(lanes or {}))} "
                f"extra={sorted(set(lanes or {}) - MANDATORY)}"
            )
        binaries = {
            name: (ROOT / path).resolve()
            for name, path in config.get("binaries", {}).items()
        }
        campaign = identity(binaries)
        run_root = results / f"unsealed-{campaign['digest']}"
        run_root.mkdir(mode=0o700, exist_ok=False)
        reports = {}
        for name in sorted(MANDATORY):
            record = lanes[name]
            argv = record.get("argv") if isinstance(record, dict) else None
            if not isinstance(argv, list) or not argv or not all(isinstance(v, str) for v in argv):
                raise RuntimeError(f"lane {name} has no closed argv list")
            output = run_root / f"{name}.json"
            command = substitute_lane_command(argv, campaign["digest"], output)
            log_path = run_root / f"{name}.log"
            with log_path.open("xb") as log:
                completed = subprocess.run(
                    command, cwd=ROOT, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, check=False,
                )
                log.flush()
                os.fsync(log.fileno())
            if completed.returncode != 0:
                raise RuntimeError(f"lane {name} failed with exit {completed.returncode}")
            bind_lane_report(
                output,
                name,
                campaign["digest"],
                matrix_config_sha256=config_sha256,
                lane_config=record,
                command_template=argv,
                command=command,
                log_path=log_path,
                runner="scripts/run_unsealed_release_matrix.py",
            )
            validate_lane_report(
                output,
                name,
                campaign["digest"],
                matrix_config_sha256=config_sha256,
                lane_config=record,
                command_template=argv,
                command=command,
                log_path=log_path,
                runner="scripts/run_unsealed_release_matrix.py",
            )
            reports[name] = str(output)
        summary = {
            "schema": "muser.unsealed-release-matrix.v1",
            "status": "passed",
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "identity": campaign["digest"],
            "campaign_identity": campaign,
            "reports": reports,
            "seals_emitted": False,
        }
        atomic_json(run_root / "RESULT.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"unsealed release matrix failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
