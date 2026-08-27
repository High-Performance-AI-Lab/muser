#!/usr/bin/env bash
# Prove a committed fresh clone builds offline with no sibling checkout.

set -eu

MUSER_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/muser-clean-clone.XXXXXX")"

cleanup() {
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

fail() {
  echo "FAIL: $1" >&2
  exit 1
}

[ -d "$MUSER_SRC/.git" ] || fail "Muser source is not a git repository"
[ -z "$(git -C "$MUSER_SRC" status --porcelain=v1 --untracked-files=all)" ] \
  || fail "clean-clone qualification requires a clean committed worktree"

git clone --local --quiet "$MUSER_SRC" "$WORKDIR/muser" \
  || fail "local fresh clone failed"

python3 "$WORKDIR/muser/scripts/audit_vendored_kvpack.py" \
  || fail "vendored kvpack audit failed in fresh clone"

CARGO_NET_OFFLINE=true CARGO_TARGET_DIR="$WORKDIR/target" \
  cargo check --manifest-path "$WORKDIR/muser/Cargo.toml" \
  --workspace --all-targets --locked \
  || fail "offline cargo check failed in fresh clone"

echo "PASS: fresh clone builds offline from in-tree dependencies"
