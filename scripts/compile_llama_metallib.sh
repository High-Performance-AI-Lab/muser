#!/usr/bin/env bash
# Build an append-only GGML Metal library from a verified llama.cpp checkout.
# This produces a development artifact; correctness and release eligibility are
# established separately by Muser's guarded parity and benchmark packets.
set -euo pipefail

usage() {
    cat <<'EOF'
usage: compile_llama_metallib.sh \
  --llama-dir PATH --revision COMMIT --output PATH [--receipt PATH]

The requested revision must equal checkout HEAD. The three upstream Metal
inputs must be clean in both the index and working tree. Existing outputs are
never replaced.
EOF
}

LLAMA_DIR=""
REVISION=""
OUTPUT=""
RECEIPT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --llama-dir) LLAMA_DIR="${2-}"; shift 2 ;;
        --revision) REVISION="${2-}"; shift 2 ;;
        --output) OUTPUT="${2-}"; shift 2 ;;
        --receipt) RECEIPT="${2-}"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$LLAMA_DIR" || -z "$REVISION" || -z "$OUTPUT" ]]; then
    usage >&2
    exit 2
fi
if [[ -z "$RECEIPT" ]]; then
    RECEIPT="${OUTPUT}.source-receipt.json"
fi

SOURCE_PATHS=(
    ggml/src/ggml-metal/ggml-metal.metal
    ggml/src/ggml-metal/ggml-metal-impl.h
    ggml/src/ggml-common.h
)
for relative in "${SOURCE_PATHS[@]}"; do
    if [[ ! -f "$LLAMA_DIR/$relative" ]]; then
        echo "error: missing llama.cpp Metal input: $relative" >&2
        exit 1
    fi
done
if [[ ! -e "$LLAMA_DIR/.git" ]]; then
    echo "error: llama.cpp source is not a git checkout" >&2
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
if ! git -C "$LLAMA_DIR" diff --quiet -- "${SOURCE_PATHS[@]}" \
    || ! git -C "$LLAMA_DIR" diff --cached --quiet -- "${SOURCE_PATHS[@]}"; then
    echo "error: verified llama.cpp Metal inputs have local modifications" >&2
    exit 1
fi

OUTPUT_PARENT="$(dirname "$OUTPUT")"
RECEIPT_PARENT="$(dirname "$RECEIPT")"
mkdir -p "$OUTPUT_PARENT" "$RECEIPT_PARENT"
OUTPUT_PARENT="$(cd "$OUTPUT_PARENT" && pwd -P)"
RECEIPT_PARENT="$(cd "$RECEIPT_PARENT" && pwd -P)"
OUTPUT="$OUTPUT_PARENT/$(basename "$OUTPUT")"
RECEIPT="$RECEIPT_PARENT/$(basename "$RECEIPT")"
if [[ -e "$OUTPUT" || -L "$OUTPUT" ]]; then
    echo "error: refusing to replace existing metallib: $OUTPUT" >&2
    exit 1
fi
if [[ -e "$RECEIPT" || -L "$RECEIPT" ]]; then
    echo "error: refusing to replace existing receipt: $RECEIPT" >&2
    exit 1
fi

TEMP_DIR="$(mktemp -d "$OUTPUT_PARENT/.muser-llama-metallib.XXXXXX")"
cleanup() {
    rm -rf -- "$TEMP_DIR"
}
trap cleanup EXIT
MERGED="$TEMP_DIR/llama-merged.metal"
AIR="$TEMP_DIR/llama-merged.air"
BUILT_OUTPUT="$TEMP_DIR/llama.metallib"

python3 - "$LLAMA_DIR" "$MERGED" <<'PY'
import re
import sys

llama_dir, output = sys.argv[1:]
with open(f"{llama_dir}/ggml/src/ggml-metal/ggml-metal.metal") as handle:
    source = handle.read()
with open(f"{llama_dir}/ggml/src/ggml-common.h") as handle:
    common = handle.read()
with open(f"{llama_dir}/ggml/src/ggml-metal/ggml-metal-impl.h") as handle:
    implementation = handle.read()

source = source.replace('#include "ggml-metal-impl.h"', implementation)
source = re.sub(r'#include "ggml-common.h"', common, source)
source = source.replace("// __embed_ggml-common.h__", common)
with open(output, "w") as handle:
    handle.write(source)
print(f"merged {len(source)} characters")
PY

xcrun -sdk macosx metal -std=metal3.1 -c "$MERGED" -o "$AIR"
xcrun -sdk macosx metallib "$AIR" -o "$BUILT_OUTPUT"

SIZE="$(stat -f '%z' "$BUILT_OUTPUT")"
SHA256="$(shasum -a 256 "$BUILT_OUTPUT" | awk '{print $1}')"
SOURCE_TREE="$(git -C "$LLAMA_DIR" rev-parse "${ACTUAL_REVISION}^{tree}")"
ORIGIN_URL="$(git -C "$LLAMA_DIR" remote get-url origin 2>/dev/null || true)"
MERGED_SHA256="$(shasum -a 256 "$MERGED" | awk '{print $1}')"
SDK_VERSION="$(xcrun --sdk macosx --show-sdk-version)"
METAL_COMPILER="$(basename "$(xcrun --find metal)")"
XCODE_VERSION="$(xcodebuild -version | tr '\n' ' ')"
RECEIPT_TEMP="$TEMP_DIR/source-receipt.json"

python3 - \
    "$RECEIPT_TEMP" "$LLAMA_DIR" "$ACTUAL_REVISION" "$SOURCE_TREE" \
    "$(basename "$OUTPUT")" "$SIZE" "$SHA256" "$MERGED_SHA256" \
    "$SDK_VERSION" "$METAL_COMPILER" "$XCODE_VERSION" "$ORIGIN_URL" <<'PY'
import hashlib
import json
import pathlib
import sys

(
    receipt_arg,
    llama_dir_arg,
    source_commit,
    source_tree,
    artifact_name,
    size_arg,
    binary_sha256,
    merged_sha256,
    sdk_version,
    metal_compiler,
    xcode_version,
    origin_url,
) = sys.argv[1:]
llama_dir = pathlib.Path(llama_dir_arg)
source_paths = (
    "ggml/src/ggml-metal/ggml-metal.metal",
    "ggml/src/ggml-metal/ggml-metal-impl.h",
    "ggml/src/ggml-common.h",
)
receipt = {
    "schema": "muser.llama_metallib.source_receipt.v1",
    "artifact_name": artifact_name,
    "binary_sha256": binary_sha256,
    "binary_size_bytes": int(size_arg),
    "source_commit": source_commit,
    "source_tree": source_tree,
    "origin_url": origin_url,
    "source_files_sha256": {
        relative: hashlib.sha256((llama_dir / relative).read_bytes()).hexdigest()
        for relative in source_paths
    },
    "merged_source_sha256": merged_sha256,
    "sdk_version": sdk_version,
    "metal_compiler": metal_compiler,
    "xcode_version": xcode_version.strip(),
    "metal_standard": "metal3.1",
    "build_steps": ["metal -c", "metallib"],
}
pathlib.Path(receipt_arg).write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n"
)
PY

# Both publications use hard links from the same filesystem and therefore
# cannot overwrite artifacts created by a competing build.
ln "$BUILT_OUTPUT" "$OUTPUT"
ln "$RECEIPT_TEMP" "$RECEIPT"

echo "llama.cpp metallib: $OUTPUT ($SIZE bytes)"
echo "source revision: $ACTUAL_REVISION"
echo "SHA-256: $SHA256"
echo "source receipt: $RECEIPT"
