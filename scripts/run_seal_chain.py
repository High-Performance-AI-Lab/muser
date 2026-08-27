#!/usr/bin/env python3
"""Retired legacy individual-seal driver; only a nonvalidating notice remains."""

from __future__ import annotations

import argparse
import json

# Audit marker: release tooling treats this historical filename as a mutator.
from release_lock import ReleaseLocked  # noqa: F401


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run-plan", action="store_true")
    args = parser.parse_args()
    if not args.dry_run_plan:
        parser.error("legacy execution is retired; use atomic_seal_campaign.py")
    print(json.dumps({
        "schema": "muser.retired-seal-chain.v1",
        "mode": "plan",
        "validating": False,
        "seals_emitted": False,
        "replacement": "scripts/atomic_seal_campaign.py",
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
