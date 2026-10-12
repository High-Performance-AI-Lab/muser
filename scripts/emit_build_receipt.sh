#!/usr/bin/env bash
# scripts/emit_build_receipt.sh
# Records a build-provenance receipt for a built muser binary.
# Usage: scripts/emit_build_receipt.sh <path-to-binary> [--features <features>] [--profile <profile>]
set -euo pipefail

BINARY="$1"; shift
FEATURES=""
PROFILE="release"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --features) FEATURES="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

[[ -f "$BINARY" ]] || { echo "no such binary: $BINARY" >&2; exit 1; }

RECEIPT_ROOT="${MUSER_RECEIPT_ROOT:?set MUSER_RECEIPT_ROOT to the append-only receipts root}"
mkdir -p "$RECEIPT_ROOT"

GIT_COMMIT=$(git rev-parse HEAD)
GIT_DIRTY=false
git diff --quiet --ignore-submodules HEAD -- || GIT_DIRTY=true
git diff --quiet --cached --ignore-submodules HEAD -- || GIT_DIRTY=true
[[ -n "$(git status --porcelain --untracked-files=no)" ]] && GIT_DIRTY=true

RUSTC_VV=$(rustc -Vv)
BINARY_SHA=$(shasum -a 256 "$BINARY" | awk '{print $1}')
BINARY_BYTES=$(stat -f%z "$BINARY" 2>/dev/null || stat -c%s "$BINARY")
BUILT_AT=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)
RECEIPT_ID="$(basename "$BINARY")-${GIT_COMMIT:0:12}-$(date -u +%Y%m%dT%H%M%SZ)"

OUT="${RECEIPT_ROOT}/${RECEIPT_ID}.json"
python3 - "$OUT" <<PYEOF
import json, sys
out = sys.argv[1]
receipt = {
    "schema": "muser.build-receipt.v1",
    "binary": {
        "path": "${BINARY}",
        "sha256": "${BINARY_SHA}",
        "bytes": ${BINARY_BYTES},
    },
    "git": {
        "commit": "${GIT_COMMIT}",
        "dirty": "${GIT_DIRTY}" == "true",
    },
    "rustc_vv": """${RUSTC_VV}""",
    "features": "${FEATURES}",
    "profile": "${PROFILE}",
    "built_at": "${BUILT_AT}",
}
with open(out, "w") as f:
    json.dump(receipt, f, indent=2, sort_keys=True)
    f.write("\n")
PYEOF

echo "wrote ${OUT}"
