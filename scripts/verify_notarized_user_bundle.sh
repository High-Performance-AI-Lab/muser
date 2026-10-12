#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "usage: scripts/verify_notarized_user_bundle.sh <muser-*-macos-arm64.dmg>" >&2
}

if [[ $# -eq 1 && ( $1 == -h || $1 == --help ) ]]; then
    usage
    exit 0
fi
if [[ $# -ne 1 ]]; then
    usage
    exit 2
fi
if [[ $(uname -s) != Darwin || $(uname -m) != arm64 ]]; then
    echo "notarized user-bundle verification requires arm64 macOS" >&2
    exit 2
fi

image=$1
[[ -f "$image" ]] || { echo "disk image is absent: $image" >&2; exit 2; }
image_dir=$(cd "$(dirname "$image")" && pwd -P)
image="$image_dir/$(basename "$image")"
checksum="$image.sha256"
receipt="$image.notarization.json"
[[ -f "$checksum" ]] || { echo "checksum is absent: $checksum" >&2; exit 2; }
[[ -f "$receipt" ]] || { echo "notarization receipt is absent: $receipt" >&2; exit 2; }

for command in codesign hdiutil plutil python3 shasum spctl syspolicy_check xcrun; do
    command -v "$command" >/dev/null || {
        echo "required command is absent: $command" >&2
        exit 2
    }
done

(
    cd "$image_dir"
    shasum -a 256 -c "$(basename "$checksum")"
)

expected_team=$(python3 - "$receipt" "$image" <<'PY'
import hashlib
import json
import pathlib
import sys

receipt_path = pathlib.Path(sys.argv[1])
image_path = pathlib.Path(sys.argv[2])
data = json.loads(receipt_path.read_text(encoding="utf-8"))
if data.get("schema") != "muser.apple-notarization.v2":
    raise SystemExit("notarization receipt has the wrong schema")
if data.get("status") != "Accepted":
    raise SystemExit("notarization receipt is not Accepted")
if data.get("artifact") != image_path.name:
    raise SystemExit("notarization receipt names a different artifact")
if data.get("entrypoint") != "Muser.app":
    raise SystemExit("notarization receipt names the wrong entry point")
if data.get("bundle_identifier") != "org.high-performance-ai-lab.muser":
    raise SystemExit("notarization receipt names the wrong bundle identifier")
log_digest = data.get("notary_log_sha256")
if not isinstance(log_digest, str) or len(log_digest) != 64:
    raise SystemExit("notarization receipt omits the accepted Apple log digest")
if not isinstance(data.get("notary_ticket_components"), int) or data[
    "notary_ticket_components"
] < 5:
    raise SystemExit("notarization receipt omits required ticket components")
digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
if data.get("artifact_sha256") != digest:
    raise SystemExit("notarization receipt does not bind the disk image")
team = data.get("team_identifier")
if not isinstance(team, str) or not team or team == "not set":
    raise SystemExit("notarization receipt omits the Developer ID team")
print(team)
PY
)

signature_info=$(codesign -dv --verbose=4 "$image" 2>&1)
grep -Fq "Authority=Developer ID Application:" <<<"$signature_info" || {
    echo "disk image is not signed by a Developer ID Application certificate" >&2
    exit 1
}
grep -Fq "TeamIdentifier=$expected_team" <<<"$signature_info" || {
    echo "disk-image signature team differs from the notarization receipt" >&2
    exit 1
}
codesign --verify --strict --verbose=2 "$image"
xcrun stapler validate "$image"
spctl --assess --type open --context context:primary-signature --verbose=2 "$image"

scratch=$(mktemp -d "${TMPDIR:-/tmp}/muser-notarized-verify.XXXXXX")
mountpoint="$scratch/mount"
mounted=0
cleanup() {
    if [[ $mounted -eq 1 ]]; then
        hdiutil detach "$mountpoint" -quiet >/dev/null 2>&1 || true
    fi
    rm -rf "$scratch"
}
trap cleanup EXIT
mkdir -p "$mountpoint"
hdiutil attach -readonly -nobrowse -mountpoint "$mountpoint" "$image" >/dev/null
mounted=1

entry_count=$(find "$mountpoint" -mindepth 1 -maxdepth 1 -print | wc -l | tr -d ' ')
[[ $entry_count == 1 ]] || {
    echo "notarized image must contain exactly Muser.app" >&2
    exit 1
}
app="$mountpoint/Muser.app"
contents="$app/Contents"
runtime_root="$contents/Resources/muser"
[[ -d "$app" && -f "$contents/Info.plist" \
    && -f "$runtime_root/SHA256SUMS" ]] || {
    echo "notarized image does not contain the signed Muser application" >&2
    exit 1
}
plutil -lint "$contents/Info.plist" >/dev/null
bundle_identifier=$(plutil -extract CFBundleIdentifier raw -o - "$contents/Info.plist")
[[ $bundle_identifier == org.high-performance-ai-lab.muser ]] || {
    echo "Muser.app has the wrong bundle identifier" >&2
    exit 1
}
(
    cd "$runtime_root"
    shasum -a 256 -c SHA256SUMS
)

app_signature=$(codesign -dv --verbose=4 "$app" 2>&1)
grep -Fq "Authority=Developer ID Application:" <<<"$app_signature" || {
    echo "Muser.app is not Developer ID signed" >&2
    exit 1
}
grep -Fq "TeamIdentifier=$expected_team" <<<"$app_signature" || {
    echo "Muser.app is signed by the wrong team" >&2
    exit 1
}
codesign --verify --deep --strict --verbose=2 "$app"
spctl --assess --type execute --verbose=2 "$app"
syspolicy_check distribution "$app"

for binary in \
    "$contents/MacOS/Muser" \
    "$contents/Helpers/muser" \
    "$contents/Helpers/muser-terminal" \
    "$contents/Helpers/muser-remote-qualify"
do
    [[ -x "$binary" ]] || { echo "signed binary is absent: $binary" >&2; exit 1; }
    signature_info=$(codesign -dv --verbose=4 "$binary" 2>&1)
    grep -Fq "Authority=Developer ID Application:" <<<"$signature_info" || {
        echo "$(basename "$binary") is not Developer ID signed" >&2
        exit 1
    }
    grep -Fq "TeamIdentifier=$expected_team" <<<"$signature_info" || {
        echo "$(basename "$binary") is signed by the wrong team" >&2
        exit 1
    }
    codesign --verify --strict --verbose=2 "$binary"
    spctl --assess --type execute --verbose=2 "$binary"
done

"$contents/MacOS/Muser" --check-bundle
"$contents/Helpers/muser-terminal" --check-bundle
(
    cd "$scratch"
    MUSER_HOME="$scratch/muser-home" MUSER_REPO_ROOT="$runtime_root" \
        "$contents/Helpers/muser" --help >/dev/null
    "$contents/Helpers/muser-terminal" --help >/dev/null
)

echo "verified notarized Muser bundle: $image"
