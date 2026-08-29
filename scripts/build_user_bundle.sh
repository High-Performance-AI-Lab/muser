#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: scripts/build_user_bundle.sh [--output DIR] [--skip-build]" >&2
}

output=""
skip_build=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            output=$2
            shift 2
            ;;
        --skip-build)
            skip_build=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage
            exit 2
            ;;
    esac
done

root=$(git rev-parse --show-toplevel)
cd "$root"

if [[ $(uname -s) != Darwin || $(uname -m) != arm64 ]]; then
    echo "user bundle requires an arm64 macOS build host" >&2
    exit 2
fi

version=$(sed -n 's/^version = "\([^"]*\)"/\1/p' Cargo.toml | head -n 1)
[[ -n "$version" ]] || { echo "workspace version is missing" >&2; exit 2; }
output=${output:-"$root/dist"}
mkdir -p "$output"
output=$(cd "$output" && pwd -P)

# Rust keeps source locations for panics even in optimized binaries. Without
# remapping, dependency paths disclose the build account and make otherwise
# identical bundles depend on the checkout location. Use encoded flags so a
# path cannot be split on whitespace, and deliberately ignore ambient
# RUSTFLAGS for the release artifact.
rust_sysroot=$(rustc --print sysroot)
release_rustflags=()
builder_root=""
case "$rust_sysroot" in
    */.rustup/toolchains/*)
        builder_root=${rust_sysroot%%/.rustup/toolchains/*}
        release_rustflags+=("--remap-path-prefix=$builder_root=/build-home")
        release_rustflags+=("--remap-path-prefix=$builder_root/.cargo=/cargo")
        ;;
esac
# rustc uses the last matching remap. Keep the narrower toolchain and source
# roots after the broad build-account rule.
release_rustflags+=("--remap-path-prefix=$rust_sysroot=/rust/sysroot")
release_rustflags+=("--remap-path-prefix=$root=/muser/source")
encoded_rustflags=""
for flag in "${release_rustflags[@]}"; do
    [[ -z "$encoded_rustflags" ]] || encoded_rustflags+=$'\x1f'
    encoded_rustflags+=$flag
done
export CARGO_ENCODED_RUSTFLAGS=$encoded_rustflags
unset RUSTFLAGS

if [[ $skip_build -eq 0 ]]; then
    cargo build --release --locked -p muser-server --bin muser
    cargo build --release --locked -p muser-bench --bin muser-remote-qualify \
        --features metal
    cargo build --release --locked -p muser-launcher --bin muser-launcher
fi

for binary in \
    target/release/muser \
    target/release/muser-remote-qualify \
    target/release/muser-launcher
do
    [[ -x "$binary" ]] || {
        echo "$binary is absent; build it or remove --skip-build" >&2
        exit 2
    }
done

scratch=$(mktemp -d "${TMPDIR:-/tmp}/muser-user-bundle.XXXXXX")
trap 'rm -rf "$scratch"' EXIT
name="muser-${version}-macos-arm64"
stage="$scratch/$name"
mkdir -p \
    "$stage/bin" \
    "$stage/docs" \
    "$stage/release" \
    "$stage/scripts/gx10/llamacpp" \
    "$stage/scripts/gx10/vllm/muser_vllm"

install -m 0755 target/release/muser "$stage/bin/muser"
install -m 0755 target/release/muser-remote-qualify "$stage/bin/muser-remote-qualify"
install -m 0755 target/release/muser-launcher "$stage/bin/muser-launcher"
install -m 0755 "packaging/Start Muser.command" "$stage/Start Muser.command"
short_version=${version%%-*}
[[ $short_version =~ ^[0-9]+(\.[0-9]+){1,2}$ ]] || {
    echo "workspace version cannot form an Apple bundle version: $version" >&2
    exit 2
}
app="$stage/Muser.app"
app_contents="$app/Contents"
app_runtime="$app_contents/Resources/muser"
mkdir -p "$app_contents/Helpers" "$app_contents/MacOS" "$app_runtime"
sed \
    -e "s/__MUSER_SHORT_VERSION__/$short_version/g" \
    -e "s/__MUSER_BUNDLE_VERSION__/$short_version/g" \
    packaging/Muser.Info.plist > "$app_contents/Info.plist"

# ld derives LC_UUID from link inputs, including checkout-specific metadata,
# even after every embedded source path is remapped. First establish the final
# ad-hoc signature layout and identifier, then canonicalize only the staged
# copy from its executable content and sign the changed UUID once more. The
# publication step later replaces this with Developer ID; target/release
# itself remains untouched.
codesign --force --sign - --identifier org.high-performance-ai-lab.muser \
    "$stage/bin/muser"
python3 scripts/normalize_macho_uuid.py "$stage/bin/muser"
codesign --force --sign - --identifier org.high-performance-ai-lab.muser \
    "$stage/bin/muser"
python3 scripts/normalize_macho_uuid.py --check "$stage/bin/muser"
codesign --verify --strict --verbose=2 "$stage/bin/muser"

codesign --force --sign - \
    --identifier org.high-performance-ai-lab.muser-remote-qualify \
    "$stage/bin/muser-remote-qualify"
python3 scripts/normalize_macho_uuid.py "$stage/bin/muser-remote-qualify"
codesign --force --sign - \
    --identifier org.high-performance-ai-lab.muser-remote-qualify \
    "$stage/bin/muser-remote-qualify"
python3 scripts/normalize_macho_uuid.py --check "$stage/bin/muser-remote-qualify"
codesign --verify --strict --verbose=2 "$stage/bin/muser-remote-qualify"

codesign --force --sign - \
    --identifier org.high-performance-ai-lab.muser-launcher \
    "$stage/bin/muser-launcher"
python3 scripts/normalize_macho_uuid.py "$stage/bin/muser-launcher"
codesign --force --sign - \
    --identifier org.high-performance-ai-lab.muser-launcher \
    "$stage/bin/muser-launcher"
python3 scripts/normalize_macho_uuid.py --check "$stage/bin/muser-launcher"
codesign --verify --strict --verbose=2 "$stage/bin/muser-launcher"

# The remap is a release invariant, not a best-effort compiler option. Scan
# what will actually ship and fail before archiving if this host is named in
# either executable.
for binary in \
    "$stage/bin/muser" \
    "$stage/bin/muser-remote-qualify" \
    "$stage/bin/muser-launcher"
do
    string_dump="$scratch/$(basename "$binary").strings"
    strings "$binary" > "$string_dump"
    if grep -F "$root" "$string_dump" >/dev/null \
        || { [[ -n "$builder_root" ]] && grep -F "$builder_root" "$string_dump" >/dev/null; }; then
        echo "$(basename "$binary") contains an unremapped build-host path" >&2
        exit 1
    fi
    while IFS= read -r dependency; do
        case "$dependency" in
            /System/Library/*|/usr/lib/*) ;;
            *)
                echo "$(basename "$binary") has a non-system runtime dependency: $dependency" >&2
                exit 1
                ;;
        esac
    done < <(otool -L "$binary" | tail -n +2 | awk '{print $1}')
done
install -m 0644 CHANGELOG.md LICENSE-APACHE LICENSE-MIT NOTICE README.md "$stage/"
bundle_docs=(
    benchmarks.md
    disaggregated-prefill.md
    install.md
    kvpack.md
    launch-claims.md
    muser-architecture.md
    nvfp4-distributed-speculative-frontier-20260818.md
    one-button-onboarding.md
    onboarding-readiness.md
    quickstart.md
    release-artifacts.json
    release-model-metadata.json
    telemetry.md
)
for file in "${bundle_docs[@]}"; do
    install -m 0644 "docs/$file" "$stage/docs/$file"
done
install -m 0644 release/nvfp4-runtime-identity-v1.json \
    release/nvfp4-runtime-overlay-v2.json \
    release/llama-server-compat-v1.json "$stage/release/"
install -m 0755 scripts/accelerator_safe.py "$stage/scripts/accelerator_safe.py"
install -m 0755 scripts/gx10/bootstrap_node.sh "$stage/scripts/gx10/bootstrap_node.sh"

llamacpp_runtime=(
    muser_prefilld.py
    muser-prefilld
    muser-prefilld.service
    muser_v2_send.py
    llamacpp_session_send.py
    protocol.py
    muser_prefill_producer.sh
    muse-glimmer-30b.layout.json
)
for file in "${llamacpp_runtime[@]}"; do
    mode=0644
    case "$file" in
        muser-prefilld|muser_prefilld.py|muser_prefill_producer.sh) mode=0755 ;;
    esac
    install -m "$mode" "scripts/gx10/llamacpp/$file" "$stage/scripts/gx10/llamacpp/$file"
done

native_runtime=(
    muser_native_prefilld.py
    resident_producer.py
    request_producer.py
    Dockerfile
    native_onboarding_identity_v1.json
)
for file in "${native_runtime[@]}"; do
    install -m 0644 "scripts/gx10/vllm/$file" "$stage/scripts/gx10/vllm/$file"
done
while IFS= read -r file; do
    install -m 0644 "$file" "$stage/scripts/gx10/vllm/muser_vllm/$(basename "$file")"
done < <(find scripts/gx10/vllm/muser_vllm -maxdepth 1 -type f -name '*.py' | LC_ALL=C sort)

# Finder must assess a conventional application, not a command or shell script
# handed to Terminal as a document. The application asks Terminal to execute a
# separately signed Mach-O helper; that helper sets the verified resource root
# and replaces itself with muser. Publication replaces these ad-hoc signatures
# with Developer ID.
install -m 0755 "$stage/bin/muser-launcher" "$app_contents/MacOS/Muser"
install -m 0755 "$stage/bin/muser-launcher" \
    "$app_contents/Helpers/muser-terminal"
install -m 0755 "$stage/bin/muser" "$app_contents/Helpers/muser"
install -m 0755 "$stage/bin/muser-remote-qualify" \
    "$app_contents/Helpers/muser-remote-qualify"
install -m 0644 \
    "$stage/CHANGELOG.md" \
    "$stage/LICENSE-APACHE" \
    "$stage/LICENSE-MIT" \
    "$stage/NOTICE" \
    "$stage/README.md" \
    "$app_runtime/"
cp -R "$stage/docs" "$stage/release" "$stage/scripts" "$app_runtime/"
(
    cd "$app_runtime"
    find . -type f -print0 \
        | LC_ALL=C sort -z \
        | xargs -0 shasum -a 256 \
        > "$scratch/app-SHA256SUMS"
)
install -m 0644 "$scratch/app-SHA256SUMS" "$app_runtime/SHA256SUMS"
codesign --force --sign - \
    --identifier org.high-performance-ai-lab.muser-terminal \
    "$app_contents/Helpers/muser-terminal"
codesign --force --sign - --identifier org.high-performance-ai-lab.muser "$app"
codesign --verify --deep --strict --verbose=2 "$app"
"$app_contents/MacOS/Muser" --check-bundle
"$app_contents/Helpers/muser-terminal" --check-bundle

# The extracted CLI tree remains a self-contained runtime root. Muser.app
# carries the same assets under Resources and its signed Terminal helper sets
# the explicit runtime root before invoking the engine.
manifest="$scratch/SHA256SUMS"
(
    cd "$stage"
    find . -type f -print0 \
        | LC_ALL=C sort -z \
        | xargs -0 shasum -a 256 \
        > "$manifest"
)
install -m 0644 "$manifest" "$stage/SHA256SUMS"

archive="$output/$name.tar.gz"
archive_tar="$scratch/$name.tar"
# Normalize every archive entry and feed tar a sorted path list. The payload
# checksums already bind content; fixed mtimes keep two assemblies of the same
# built binaries and source tree byte-for-byte identical as well.
find "$stage" -exec touch -h -t 200001010000.00 {} +
(
    cd "$scratch"
    find "$name" -print0 \
        | LC_ALL=C sort -z \
        | COPYFILE_DISABLE=1 tar --null --no-recursion -cf "$archive_tar" -T -
)
# Apple gzip otherwise stamps the wrapper with the current time even when
# every tar member is normalized.
gzip -n -c "$archive_tar" > "$archive"
(
    cd "$output"
    shasum -a 256 "$(basename "$archive")" > "$(basename "$archive").sha256"
)
echo "$archive"
