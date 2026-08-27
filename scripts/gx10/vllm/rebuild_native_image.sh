#!/usr/bin/env bash
# Rebuild the resident NVFP4 vLLM producer image on the GX10 and receipt it.
#
# Purpose
# -------
# The producer image bakes scripts/gx10/{llamacpp,vllm} at build time, so any
# change to the sender, connector, or producer needs an image rebuild to take
# effect. This script runs the full rebuild pipeline from the Mac:
#
#   1. stage the current repo's scripts/gx10 tree as the build context,
#   2. docker build with the pinned Dockerfile (pinned base digest + pinned
#      vLLM wheel SHA-256 — see the Dockerfile),
#   3. produce the image receipt and the adapter identity receipt,
#   4. print the new adapter_sha256.
#
# The adapter hash covers the sender/connector sources, so it changes with
# them. After a rebuild you must, in order:
#   a. recreate the resident container from the new image (same mounts/env/
#      cmd; the supervisor then keeps it alive),
#   b. set the new adapter_sha256 in the producer's work config.json AND in
#      the Mac-side cluster config — the receiver rejects a Begin whose
#      adapter identity differs (fail-closed, verified 2026-08-19).
#
# Usage
# -----
#   scripts/gx10/vllm/rebuild_native_image.sh                 # tag = git sha
#   scripts/gx10/vllm/rebuild_native_image.sh --tag mytag --node <alias>
#
# Requires: repo working tree committed (the tag defaults to the HEAD sha),
# ssh access to the node, docker on the node. No GPU is touched by the build.
set -euo pipefail

NODE=""
TAG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --node) NODE="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$NODE" ]] || { echo "--node <alias> is required" >&2; exit 2; }
ROOT="$(git rev-parse --show-toplevel)"
TAG="${TAG:-$(git -C "$ROOT" rev-parse --short HEAD)}"
if [[ -n "$(git -C "$ROOT" status --porcelain -- scripts/gx10)" ]]; then
  echo "refusing to build from a dirty scripts/gx10 tree (commit first)" >&2
  exit 1
fi

BUILD_DIR=".muser/lane/gx10/build/nvfp4-vllm-${TAG}-$(date -u +%Y%m%d)"
IMAGE="muser/gx10-vllm-native:${TAG}"

echo "== staging context on ${NODE}:${BUILD_DIR}"
tar czf - -C "$ROOT" scripts/gx10 | ssh "$NODE" "mkdir -p ~/${BUILD_DIR} && tar xzf - -C ~/${BUILD_DIR}"

echo "== building ${IMAGE} on ${NODE}"
ssh "$NODE" "cd ~/${BUILD_DIR} && docker build -f scripts/gx10/vllm/Dockerfile -t ${IMAGE} ."

echo "== receipting"
ssh "$NODE" "python3 ~/${BUILD_DIR}/scripts/gx10/vllm/receipt_image.py \
  --image ${IMAGE} --source-root ~/${BUILD_DIR} \
  --output ~/.muser/lane/gx10/receipts/nvfp4-vllm-image-${TAG}.json && \
  python3 ~/${BUILD_DIR}/scripts/gx10/vllm/receipt_adapter.py \
  --image-receipt ~/.muser/lane/gx10/receipts/nvfp4-vllm-image-${TAG}.json \
  --output ~/.muser/lane/gx10/receipts/spark-producer-adapter-${TAG}.json" >/dev/null

ssh "$NODE" "python3 -c \"import json; r=json.load(open('${HOME}/.muser/lane/gx10/receipts/spark-producer-adapter-${TAG}.json')); print('adapter_sha256:', r['adapter_sha256'])\"" 2>/dev/null \
  || ssh "$NODE" "python3 -c \"import json,os; r=json.load(open(os.path.expanduser('~/.muser/lane/gx10/receipts/spark-producer-adapter-${TAG}.json'))); print('adapter_sha256:', r['adapter_sha256'])\""
echo "== done: ${IMAGE} (receipts under ~/.muser/lane/gx10/receipts/)"
echo "next: recreate the resident container from ${IMAGE} and set this adapter_sha256 in the producer config.json and the Mac cluster config"
