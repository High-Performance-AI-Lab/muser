#!/bin/sh
# Build an append-only Muse mtmd bridge from the exact qualification llama.cpp.
set -eu

ROOT=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
LLAMA_SRC=${MUSER_LLAMA_SRC:-}
OUT=${MUSER_MTMD_OUT:-}
REVISION=${MUSER_LLAMA_REVISION:-}
DRY_RUN=0

usage() {
    echo "usage: build_mtmd_bridge.sh --llama-dir PATH --revision COMMIT --output PATH [--dry-run]" >&2
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --llama-dir) LLAMA_SRC=${2-}; shift 2 ;;
        --revision) REVISION=${2-}; shift 2 ;;
        --output) OUT=${2-}; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
done

case "$REVISION" in
    ''|*[!0-9a-f]* ) echo "revision must be exact lowercase hex" >&2; exit 2 ;;
esac
if [ "${#REVISION}" -ne 40 ] || [ -z "$LLAMA_SRC" ] || [ -z "$OUT" ]; then
    usage
    exit 2
fi

if [ "$DRY_RUN" -eq 1 ]; then
    python3 - "$LLAMA_SRC" "$REVISION" "$OUT" <<'PY'
import json, sys
source, revision, output = sys.argv[1:]
print(json.dumps({
    "schema": "muser.mtmd-bridge.plan.v1",
    "mode": "dry-run",
    "llama_dir": source,
    "llama_revision": revision,
    "output": output,
    "bridge_abi": "muser-mtmd-muse-vision-v1",
    "accelerator_touched": False,
}, indent=2, sort_keys=True))
PY
    exit 0
fi

test -e "$LLAMA_SRC/.git"
test -f "$LLAMA_SRC/tools/mtmd/mtmd.h"
if [ -e "$OUT" ] || [ -L "$OUT" ]; then
    echo "output already exists: $OUT" >&2
    exit 2
fi

COMMIT=$(git -C "$LLAMA_SRC" rev-parse HEAD)
REQUIRED=$(git -C "$LLAMA_SRC" rev-parse --verify "$REVISION^{commit}")
if [ "$COMMIT" != "$REQUIRED" ]; then
    echo "llama.cpp is $COMMIT, expected $REQUIRED" >&2
    exit 2
fi
PARENT=$(dirname "$OUT")
mkdir -p "$PARENT"
PARENT=$(CDPATH='' cd -- "$PARENT" && pwd -P)
OUT="$PARENT/$(basename "$OUT")"
STAGE=$(mktemp -d "$PARENT/.muser-mtmd.XXXXXX")
cleanup() { rm -rf -- "$STAGE"; }
trap cleanup EXIT HUP INT TERM

SOURCE_DIR="$STAGE/source"
BUILD_DIR="$STAGE/build"
PACKAGE_DIR="$STAGE/package"
BRIDGE_SOURCE="$STAGE/muser_mtmd_bridge.cpp"
git clone --quiet --no-hardlinks --no-checkout "$LLAMA_SRC" "$SOURCE_DIR"
git -C "$SOURCE_DIR" checkout --quiet --detach "$REQUIRED"
cp "$ROOT/native/mtmd/muser_mtmd_bridge.cpp" "$BRIDGE_SOURCE"

cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TOOLS=OFF \
    -DLLAMA_BUILD_MTMD=ON \
    -DGGML_METAL=ON \
    -DGGML_CCACHE=OFF
cmake --build "$BUILD_DIR" --target mtmd -j "$(sysctl -n hw.logicalcpu)"

mkdir "$PACKAGE_DIR"

c++ -O3 -std=c++17 -dynamiclib \
    "$BRIDGE_SOURCE" \
    -I"$SOURCE_DIR/include" -I"$SOURCE_DIR/ggml/include" -I"$SOURCE_DIR/tools/mtmd" \
    -L"$BUILD_DIR/bin" -lmtmd -lllama -lggml \
    -Wl,-install_name,@rpath/libmuser_mtmd_bridge.dylib \
    -Wl,-rpath,@loader_path \
    -o "$PACKAGE_DIR/libmuser_mtmd_bridge.dylib"

for library in libmtmd.0.dylib libllama.0.dylib libggml.0.dylib \
    libggml-base.0.dylib libggml-cpu.0.dylib libggml-blas.0.dylib libggml-metal.0.dylib
do
    source=$(readlink "$BUILD_DIR/bin/$library" || true)
    if [ -n "$source" ]; then
        cp "$BUILD_DIR/bin/$source" "$PACKAGE_DIR/$library"
    else
        cp "$BUILD_DIR/bin/$library" "$PACKAGE_DIR/$library"
    fi
done

for library in "$PACKAGE_DIR"/*.dylib; do
    install_name_tool -add_rpath @loader_path "$library" 2>/dev/null || true
done

TREE=$(git -C "$LLAMA_SRC" rev-parse "$COMMIT^{tree}")
ORIGIN=$(git -C "$LLAMA_SRC" remote get-url origin 2>/dev/null || true)
python3 - "$PACKAGE_DIR" "$BRIDGE_SOURCE" "$SOURCE_DIR" "$COMMIT" "$TREE" "$ORIGIN" <<'PY'
import hashlib, json, pathlib, platform, subprocess, sys
stage = pathlib.Path(sys.argv[1]); bridge = pathlib.Path(sys.argv[2]); llama = pathlib.Path(sys.argv[3])
commit, tree, origin = sys.argv[4:]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()
artifacts={p.name:{"bytes":p.stat().st_size,"sha256":sha(p)} for p in sorted(stage.glob('*.dylib'))}
source_paths=("tools/mtmd/mtmd.h","tools/mtmd/mtmd.cpp","include/llama.h")
receipt={
    "schema":"muser.mtmd_bridge.receipt.v2",
    "status":"built",
    "llama_commit":commit,
    "llama_tree":tree,
    "llama_origin":origin,
    "bridge_abi":"muser-mtmd-muse-vision-v1",
    "bridge_source_sha256":sha(bridge),
    "llama_sources_sha256":{name:sha(llama/name) for name in source_paths},
    "artifacts":artifacts,
    "platform":platform.platform(),
    "compiler":subprocess.run(['c++','--version'],text=True,stdout=subprocess.PIPE,check=True).stdout.splitlines()[0],
    "executed":False,
}
(stage/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
(stage/'SHA256SUMS').write_text(''.join(f"{v['sha256']}  {k}\n" for k,v in sorted(artifacts.items())))
PY

mv "$PACKAGE_DIR" "$OUT"
trap - EXIT HUP INT TERM
rm -rf -- "$STAGE"
echo "$OUT/receipt.json"
