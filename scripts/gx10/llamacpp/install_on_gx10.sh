#!/bin/sh
# Install the pinned Muser llama.cpp adapter and build its export helper.
set -eu

# This host build path only applies muser_streaming_kv.patch and
# muser_logical_swa.patch against a floating CUDA_IMAGE tag, so it silently
# diverges from Dockerfile's receipted 3-patch set pinned by image digest
# (muser_cuda_metal_compat.patch included, container_receipt-verified by
# muser_prefilld.py). Building here produces a binary the receiver's
# container_receipt check will refuse to arm. Use the container path unless
# you know exactly why you need this one.
if [ "${FORCE_HOST_BUILD:-0}" != 1 ]; then
    cat >&2 <<'EOF'
============================================================
 install_on_gx10.sh is DEPRECATED.
 It applies only 2 of the 3 adapter patches and does not pin
 CUDA_IMAGE by digest, so its output is not the receipted
 container image muser_prefilld.py requires (container_receipt
 schema muser.gx10-container.receipt.v1). Build and ship the
 container image from scripts/gx10/llamacpp/Dockerfile instead.

 Set FORCE_HOST_BUILD=1 to run this script anyway.
============================================================
EOF
    exit 2
fi

GX10=${MUSER_GX10_HOST:?set MUSER_GX10_HOST}
KEY=${MUSER_GX10_SSH_KEY:?set MUSER_GX10_SSH_KEY}
WORK=${MUSER_GX10_WORK:?set MUSER_GX10_WORK}
LLAMA_REMOTE=${MUSER_GX10_LLAMA_DIR:-llama.cpp}
CUDA_IMAGE=${CUDA_IMAGE:-nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04}
SCRIPT_DIR=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)

case "$WORK" in
    *[!A-Za-z0-9_./-]*|*..*|'') echo "unsafe remote work path" >&2; exit 2 ;;
esac

ssh -i "$KEY" -o BatchMode=yes "$GX10" "mkdir -p '$WORK/llamacpp'"
scp -q -i "$KEY" \
    "$SCRIPT_DIR/spark_kv_export.cpp" \
    "$SCRIPT_DIR/muser_streaming_kv.patch" \
    "$SCRIPT_DIR/muser_logical_swa.patch" \
    "$SCRIPT_DIR/llamacpp_session_send.py" \
    "$SCRIPT_DIR/protocol.py" \
    "$SCRIPT_DIR/muser_v2_send.py" \
    "$SCRIPT_DIR/muser_prefilld.py" \
    "$SCRIPT_DIR/muser-prefilld" \
    "$SCRIPT_DIR/muse-glimmer-30b.layout.json" \
    "$SCRIPT_DIR/muser_prefill_producer.sh" \
    "$GX10:$WORK/llamacpp/"

ssh -i "$KEY" -o BatchMode=yes "$GX10" "
    cd \$HOME/$LLAMA_REMOTE && \
    if grep -q 'bool values_are_transposed() const;' src/llama-kv-cache.h; then :; \
    else git apply --check \$HOME/$WORK/llamacpp/muser_streaming_kv.patch && \
         git apply \$HOME/$WORK/llamacpp/muser_streaming_kv.patch; fi && \
    if grep -q 'get_cells_for_positions' src/llama-kv-cache.h; then :; \
    else git apply --check \$HOME/$WORK/llamacpp/muser_logical_swa.patch && \
         git apply \$HOME/$WORK/llamacpp/muser_logical_swa.patch; fi && \
    docker run --rm \
        -v \$HOME/$LLAMA_REMOTE:/src \
        -v \$HOME/$WORK:/run/muser \
        -w /src '$CUDA_IMAGE' \
        sh -lc 'apt-get update -qq && \
          DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cmake >/dev/null && \
          cmake --build build --target mtmd -j2 && \
          g++ -O2 -std=c++17 -pthread /run/muser/llamacpp/spark_kv_export.cpp \
            -I/src/include -I/src/src -I/src/ggml/include -I/src/tools/mtmd \
            -L/src/build/bin -lmtmd -lllama -lggml -lggml-base -lggml-cpu -lggml-cuda \
            -L/usr/local/cuda/lib64/stubs -lcuda \
            -Wl,-rpath,/src/build/bin \
            -o /src/build/bin/spark_kv_export'
"
