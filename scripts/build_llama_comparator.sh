#!/usr/bin/env bash
# Build an append-only, source-pinned llama.cpp comparator bundle without
# executing any accelerator binary. The Muser fixture patch is applied only in
# a temporary clone, leaving the supplied upstream checkout untouched.
set -euo pipefail

usage() {
    cat <<'EOF'
usage: build_llama_comparator.sh \
  --llama-dir PATH --revision COMMIT --output-dir PATH [--patch PATH] [--metal on|off]

The requested revision must equal checkout HEAD. The output directory must not
exist. The bundle contains llama-bench, llama-server, llama-perplexity, and
source-receipt.json.
EOF
}

LLAMA_DIR=""
REVISION=""
OUTPUT_DIR=""
PATCH_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/llama_bench_fixture.patch"
METAL="on"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --llama-dir) LLAMA_DIR="${2-}"; shift 2 ;;
        --revision) REVISION="${2-}"; shift 2 ;;
        --output-dir) OUTPUT_DIR="${2-}"; shift 2 ;;
        --patch) PATCH_PATH="${2-}"; shift 2 ;;
        --metal) METAL="${2-}"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "$METAL" != "on" && "$METAL" != "off" ]]; then
    echo "error: --metal must be on or off" >&2
    exit 2
fi

if [[ -z "$LLAMA_DIR" || -z "$REVISION" || -z "$OUTPUT_DIR" ]]; then
    usage >&2
    exit 2
fi
if [[ ! -e "$LLAMA_DIR/.git" ]]; then
    echo "error: llama.cpp source is not a git checkout" >&2
    exit 1
fi
if [[ ! -f "$PATCH_PATH" || -L "$PATCH_PATH" ]]; then
    echo "error: comparator patch is absent or a symlink: $PATCH_PATH" >&2
    exit 1
fi

ACTUAL_REVISION="$(git -C "$LLAMA_DIR" rev-parse HEAD)"
REQUIRED_REVISION="$(git -C "$LLAMA_DIR" rev-parse --verify "${REVISION}^{commit}")"
if [[ "$ACTUAL_REVISION" != "$REQUIRED_REVISION" ]]; then
    echo "error: llama.cpp revision mismatch" >&2
    echo "  required: $REQUIRED_REVISION" >&2
    echo "  actual:   $ACTUAL_REVISION" >&2
    exit 1
fi
if [[ -e "$OUTPUT_DIR" || -L "$OUTPUT_DIR" ]]; then
    echo "error: refusing to replace comparator output: $OUTPUT_DIR" >&2
    exit 1
fi

OUTPUT_PARENT="$(dirname "$OUTPUT_DIR")"
mkdir -p "$OUTPUT_PARENT"
OUTPUT_PARENT="$(cd "$OUTPUT_PARENT" && pwd -P)"
OUTPUT_DIR="$OUTPUT_PARENT/$(basename "$OUTPUT_DIR")"
TEMP_DIR="$(mktemp -d "$OUTPUT_PARENT/.muser-llama-comparator.XXXXXX")"
cleanup() {
    rm -rf -- "$TEMP_DIR"
}
trap cleanup EXIT

SOURCE_DIR="$TEMP_DIR/source"
BUILD_DIR="$TEMP_DIR/build"
STAGE_DIR="$TEMP_DIR/stage"
git clone --quiet --no-hardlinks --no-checkout "$LLAMA_DIR" "$SOURCE_DIR"
git -C "$SOURCE_DIR" checkout --quiet --detach "$REQUIRED_REVISION"
git -C "$SOURCE_DIR" apply --whitespace=nowarn --check "$PATCH_PATH"
git -C "$SOURCE_DIR" apply --whitespace=nowarn "$PATCH_PATH"

PATCH_SHA256="$(shasum -a 256 "$PATCH_PATH" | awk '{print $1}')"
SOURCE_TREE="$(git -C "$SOURCE_DIR" rev-parse "${REQUIRED_REVISION}^{tree}")"
ORIGIN_URL="$(git -C "$LLAMA_DIR" remote get-url origin 2>/dev/null || true)"
PATCHED_BENCH_SOURCE_SHA256="$(shasum -a 256 "$SOURCE_DIR/tools/llama-bench/llama-bench.cpp" | awk '{print $1}')"
PATCHED_PERPLEXITY_SOURCE_SHA256="$(shasum -a 256 "$SOURCE_DIR/tools/perplexity/perplexity.cpp" | awk '{print $1}')"
PATCHED_SERVER_SOURCE_SHA256="$(shasum -a 256 "$SOURCE_DIR/tools/server/server.cpp" | awk '{print $1}')"
DIRTY_PATHS="$(git -C "$SOURCE_DIR" status --short | awk '{print $2}')"
EXPECTED_DIRTY_PATHS=$'tools/llama-bench/llama-bench.cpp\ntools/perplexity/perplexity.cpp\ntools/server/server.cpp'
if [[ "$DIRTY_PATHS" != "$EXPECTED_DIRTY_PATHS" ]]; then
    echo "error: comparator patch modified an unexpected path" >&2
    git -C "$SOURCE_DIR" status --short >&2
    exit 1
fi

CXX_RELEASE_FLAGS="-O3 -DNDEBUG -DMUSER_COMPARATOR_UPSTREAM_COMMIT=${REQUIRED_REVISION} -DMUSER_COMPARATOR_PATCH_SHA256=${PATCH_SHA256}"
if [[ "$METAL" == "on" ]]; then
    METAL_CMAKE=ON
    METAL_JSON=true
else
    METAL_CMAKE=OFF
    METAL_JSON=false
fi
cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_BUILD_TOOLS=ON \
    -DGGML_METAL="$METAL_CMAKE" \
    -DGGML_CCACHE=OFF \
    -DCMAKE_C_FLAGS_RELEASE="-O3 -DNDEBUG" \
    -DCMAKE_CXX_FLAGS_RELEASE="$CXX_RELEASE_FLAGS"
cmake --build "$BUILD_DIR" --target llama-bench llama-server llama-perplexity -j "$(sysctl -n hw.logicalcpu)"

mkdir "$STAGE_DIR"
for binary in llama-bench llama-server llama-perplexity; do
    if [[ ! -x "$BUILD_DIR/bin/$binary" ]]; then
        echo "error: build did not produce $binary" >&2
        exit 1
    fi
    cp "$BUILD_DIR/bin/$binary" "$STAGE_DIR/$binary"
done

BENCH_SIZE="$(stat -f '%z' "$STAGE_DIR/llama-bench")"
BENCH_SHA256="$(shasum -a 256 "$STAGE_DIR/llama-bench" | awk '{print $1}')"
SERVER_SIZE="$(stat -f '%z' "$STAGE_DIR/llama-server")"
SERVER_SHA256="$(shasum -a 256 "$STAGE_DIR/llama-server" | awk '{print $1}')"
PERPLEXITY_SIZE="$(stat -f '%z' "$STAGE_DIR/llama-perplexity")"
PERPLEXITY_SHA256="$(shasum -a 256 "$STAGE_DIR/llama-perplexity" | awk '{print $1}')"
CMAKE_VERSION="$(cmake --version | head -1)"
CXX_VERSION="$(c++ --version | head -1)"
SDK_VERSION="$(xcrun --sdk macosx --show-sdk-version)"

python3 - \
    "$STAGE_DIR/source-receipt.json" "$REQUIRED_REVISION" "$SOURCE_TREE" \
    "$ORIGIN_URL" "$PATCH_SHA256" "$PATCHED_BENCH_SOURCE_SHA256" \
    "$PATCHED_PERPLEXITY_SOURCE_SHA256" "$PATCHED_SERVER_SOURCE_SHA256" \
    "$BENCH_SIZE" "$BENCH_SHA256" "$SERVER_SIZE" "$SERVER_SHA256" \
    "$PERPLEXITY_SIZE" "$PERPLEXITY_SHA256" \
    "$CMAKE_VERSION" "$CXX_VERSION" "$SDK_VERSION" "$METAL_JSON" <<'PY'
import datetime as dt
import json
import pathlib
import sys

(
    receipt_arg,
    source_commit,
    source_tree,
    origin_url,
    patch_sha256,
    patched_source_sha256,
    patched_perplexity_source_sha256,
    patched_server_source_sha256,
    bench_size,
    bench_sha256,
    server_size,
    server_sha256,
    perplexity_size,
    perplexity_sha256,
    cmake_version,
    cxx_version,
    sdk_version,
    metal_json,
) = sys.argv[1:]
receipt = {
    "schema": "muser.llama_comparator.source_receipt.v3",
    "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "source_commit": source_commit,
    "source_tree": source_tree,
    "origin_url": origin_url,
    "patch_name": "llama_bench_fixture.patch",
    "patch_sha256": patch_sha256,
    "patched_source_sha256": patched_source_sha256,
    "patched_perplexity_source_sha256": patched_perplexity_source_sha256,
    "patched_server_source_sha256": patched_server_source_sha256,
    "artifacts": {
        "llama-bench": {"bytes": int(bench_size), "sha256": bench_sha256},
        "llama-server": {"bytes": int(server_size), "sha256": server_sha256},
        "llama-perplexity": {
            "bytes": int(perplexity_size),
            "sha256": perplexity_sha256,
        },
    },
    "build": {
        "type": "Release",
        "static_libraries": True,
        "metal": metal_json == "true",
        "tests": False,
        "examples": False,
        "tools": True,
        "server": True,
        "embedded_ui": False,
        "cmake_version": cmake_version,
        "cxx_version": cxx_version,
        "macos_sdk_version": sdk_version,
    },
    "executed": False,
}
pathlib.Path(receipt_arg).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY

# Publishing the prepared directory with rename keeps incomplete builds out of
# the requested append-only destination.
python3 - "$STAGE_DIR" "$OUTPUT_DIR" <<'PY'
import os
import sys
os.rename(sys.argv[1], sys.argv[2])
PY

echo "llama.cpp comparator: $OUTPUT_DIR"
echo "source revision: $REQUIRED_REVISION"
echo "patch SHA-256: $PATCH_SHA256"
echo "llama-bench SHA-256: $BENCH_SHA256"
echo "llama-server SHA-256: $SERVER_SHA256"
echo "llama-perplexity SHA-256: $PERPLEXITY_SHA256"
