#!/bin/sh
# Narrow llama.cpp GX10 producer for Muser Handoff V2.
set -eu

WORK=${MUSER_GX10_WORK:?set MUSER_GX10_WORK}
FIXTURE=${1:?pass the token fixture file name}
SERIAL=${2:-$(date +%s)}
LLAMA_SRC=${LLAMA_SRC:?set LLAMA_SRC}
MODEL_HOST_DIR=${MODEL_HOST_DIR:?set MODEL_HOST_DIR}
MODEL_CONT=${MODEL_CONT:?set MODEL_CONT}
RECEIVER_HOST=${RECEIVER_HOST:?set RECEIVER_HOST}
HMAC_KEY_FILE=${HMAC_KEY_FILE:?set HMAC_KEY_FILE}

case "$FIXTURE" in
    *[!A-Za-z0-9._-]*|'') echo "invalid file name: $FIXTURE" >&2; exit 2 ;;
esac

: "${SERVER_LEAF_SHA256:?set SERVER_LEAF_SHA256}"
: "${MODEL_SHA256:?set MODEL_SHA256}"
: "${MODEL_REVISION:?set MODEL_REVISION}"
: "${TOKENIZER_REVISION:?set TOKENIZER_REVISION}"
: "${TOKENIZER_SHA256:?set TOKENIZER_SHA256}"
: "${CHAT_TEMPLATE_SHA256:?set CHAT_TEMPLATE_SHA256}"
: "${CONTEXT_POLICY_SHA256:?set CONTEXT_POLICY_SHA256}"
: "${ADAPTER_SHA256:?set ADAPTER_SHA256}"
: "${TARGET_CACHE_IDENTITY_SHA256:?set TARGET_CACHE_IDENTITY_SHA256}"
: "${HMAC_KEY_ID:?set HMAC_KEY_ID}"
: "${HMAC_EPOCH:?set HMAC_EPOCH}"
: "${GENERATION:?set GENERATION}"

CUDA_IMAGE=${CUDA_IMAGE:-nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04}
RECEIVER_PORT=${RECEIVER_PORT:-29590}
SERVER_NAME=${SERVER_NAME:-muser-prefilld}
MAX_CONTEXT=${MAX_CONTEXT:-131072}
SESSION=kv-session-$SERIAL.bin
DFLASH_SESSION=dflash-session-$SERIAL.bin

cleanup() { rm -f "$WORK/$SESSION" "$WORK/$DFLASH_SESSION"; }
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

set -- \
    --model "$MODEL_CONT" \
    --tokens "/run/muser/$FIXTURE" \
    --out "/run/muser/$SESSION" \
    --n-ctx "$MAX_CONTEXT" \
    --n-batch 2048 \
    --n-ubatch 512 \
    --flash-attn 1 \
    --skip-tail 1

if [ -n "${DFLASH_MODEL_CONT:-}" ]; then
    : "${DFLASH_IDENTITY_SHA256:?set DFLASH_IDENTITY_SHA256}"
    : "${DFLASH_KV_HEADS:?set DFLASH_KV_HEADS}"
    : "${DFLASH_HEAD_DIM:?set DFLASH_HEAD_DIM}"
    set -- "$@" \
        --draft-model "$DFLASH_MODEL_CONT" \
        --draft-out "/run/muser/$DFLASH_SESSION"
fi

docker run --rm --gpus all \
    -v "$LLAMA_SRC:/src" \
    -v "$MODEL_HOST_DIR:/models:ro" \
    -v "$WORK:/run/muser" \
    -w /src \
    "$CUDA_IMAGE" \
    /src/build/bin/spark_kv_export "$@"

set -- \
    --session "$WORK/$SESSION" \
    --prompt-token-fixture "$WORK/$FIXTURE" \
    --receiver-host "$RECEIVER_HOST" \
    --receiver-port "$RECEIVER_PORT" \
    --server-name "$SERVER_NAME" \
    --ca-cert "$WORK/pki/ca.cert.pem" \
    --client-cert "$WORK/pki/gx10.cert.pem" \
    --client-key "$WORK/pki/gx10.key.pem" \
    --server-leaf-sha256 "$SERVER_LEAF_SHA256" \
    --hmac-key-file "$HMAC_KEY_FILE" \
    --hmac-key-id "$HMAC_KEY_ID" \
    --hmac-epoch "$HMAC_EPOCH" \
    --generation "$GENERATION" \
    --model-sha256 "$MODEL_SHA256" \
    --model-revision "$MODEL_REVISION" \
    --tokenizer-revision "$TOKENIZER_REVISION" \
    --tokenizer-sha256 "$TOKENIZER_SHA256" \
    --chat-template-sha256 "$CHAT_TEMPLATE_SHA256" \
    --context-policy-sha256 "$CONTEXT_POLICY_SHA256" \
    --adapter-sha256 "$ADAPTER_SHA256" \
    --target-cache-identity-sha256 "$TARGET_CACHE_IDENTITY_SHA256"

if [ -n "${DFLASH_MODEL_CONT:-}" ]; then
    set -- "$@" \
        --dflash-session "$WORK/$DFLASH_SESSION" \
        --dflash-identity-sha256 "$DFLASH_IDENTITY_SHA256" \
        --dflash-kv-heads "$DFLASH_KV_HEADS" \
        --dflash-head-dim "$DFLASH_HEAD_DIM"
fi

python3 "$WORK/llamacpp/muser_v2_send.py" "$@"
