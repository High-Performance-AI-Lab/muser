#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
usage: scripts/notarize_user_bundle.sh \
  --archive muser-*-macos-arm64.tar.gz \
  --identity "Developer ID Application: ..." \
  --keychain-profile PROFILE [--output DIR] [--timeout 30m]

The keychain profile must already exist through `xcrun notarytool
store-credentials`. Credentials are never accepted on this command line.
EOF
}

archive=""
identity=""
keychain_profile=""
output=""
timeout="30m"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --archive)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            archive=$2
            shift 2
            ;;
        --identity)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            identity=$2
            shift 2
            ;;
        --keychain-profile)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            keychain_profile=$2
            shift 2
            ;;
        --output)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            output=$2
            shift 2
            ;;
        --timeout)
            [[ $# -ge 2 ]] || { usage; exit 2; }
            timeout=$2
            shift 2
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

[[ -n "$archive" && -n "$identity" && -n "$keychain_profile" ]] || {
    usage
    exit 2
}
if [[ $(uname -s) != Darwin || $(uname -m) != arm64 ]]; then
    echo "signing and notarization require arm64 macOS" >&2
    exit 2
fi
if [[ ! $timeout =~ ^[0-9]+[smh]?$ ]]; then
    echo "--timeout must be an integer with optional s, m, or h suffix" >&2
    exit 2
fi

for command in \
    codesign hdiutil plutil python3 security shasum spctl syspolicy_check tar xcrun
do
    command -v "$command" >/dev/null || {
        echo "required command is absent: $command" >&2
        exit 2
    }
done

[[ -f "$archive" ]] || { echo "unsigned archive is absent: $archive" >&2; exit 2; }
archive_dir=$(cd "$(dirname "$archive")" && pwd -P)
archive="$archive_dir/$(basename "$archive")"
archive_checksum="$archive.sha256"
[[ -f "$archive_checksum" ]] || {
    echo "unsigned archive checksum is absent: $archive_checksum" >&2
    exit 2
}
case "$(basename "$archive")" in
    muser-*-macos-arm64.tar.gz) ;;
    *) echo "unsigned archive name does not match the user-bundle contract" >&2; exit 2 ;;
esac

output=${output:-$archive_dir}
mkdir -p "$output"
output=$(cd "$output" && pwd -P)
bundle_name=$(basename "$archive" .tar.gz)
official_name="$bundle_name.dmg"
official="$output/$official_name"
official_checksum="$official.sha256"
official_receipt="$official.notarization.json"
for path in "$official" "$official_checksum" "$official_receipt"; do
    [[ ! -e "$path" ]] || {
        echo "refusing to overwrite existing release output: $path" >&2
        exit 2
    }
done

available_identities=$(security find-identity -v -p codesigning)
grep -Fq "\"$identity\"" <<<"$available_identities" || {
        echo "the requested Developer ID signing identity is unavailable" >&2
        exit 2
    }

(
    cd "$archive_dir"
    shasum -a 256 -c "$(basename "$archive_checksum")"
)
source_archive_sha256=$(shasum -a 256 "$archive" | awk '{print $1}')

scratch=$(mktemp -d "${TMPDIR:-/tmp}/muser-notarize.XXXXXX")
trap 'rm -rf "$scratch"' EXIT
extract="$scratch/extract"
mkdir -p "$extract"
tar -xzf "$archive" -C "$extract"
entry_count=$(find "$extract" -mindepth 1 -maxdepth 1 -print | wc -l | tr -d ' ')
[[ $entry_count == 1 && -d "$extract/$bundle_name" ]] || {
    echo "unsigned archive must contain exactly $bundle_name" >&2
    exit 1
}
root="$extract/$bundle_name"
(
    cd "$root"
    shasum -a 256 -c SHA256SUMS
)

# A Finder-opened command-line tool or shell script triggers document
# Gatekeeper and can be blocked even when nested in another archive. The
# deterministic bundle therefore exposes a conventional application; its
# Terminal entry point is a native signed helper, not an opened shell document.
image_root="$scratch/image-root"
app="$image_root/Muser.app"
mkdir -p "$image_root"
[[ -d "$root/Muser.app" ]] || {
    echo "unsigned archive does not contain Muser.app" >&2
    exit 1
}
cp -R "$root/Muser.app" "$app"
contents="$app/Contents"
runtime_root="$contents/Resources/muser"
for input in \
    "$contents/Info.plist" \
    "$contents/MacOS/Muser" \
    "$contents/Helpers/muser" \
    "$contents/Helpers/muser-terminal" \
    "$contents/Helpers/muser-remote-qualify" \
    "$runtime_root/SHA256SUMS"
do
    [[ -f "$input" ]] || { echo "application input is absent: $input" >&2; exit 1; }
done
[[ -x "$contents/MacOS/Muser" && -x "$contents/Helpers/muser" \
    && -x "$contents/Helpers/muser-terminal" \
    && -x "$contents/Helpers/muser-remote-qualify" \
    ]] || {
    echo "one or more application inputs are not executable" >&2
    exit 1
}
plutil -lint "$contents/Info.plist" >/dev/null
if grep -Fq '__MUSER_' "$contents/Info.plist"; then
    echo "application Info.plist still contains a build placeholder" >&2
    exit 1
fi
(
    cd "$runtime_root"
    shasum -a 256 -c SHA256SUMS
)

team_identifier=""
for binary in \
    "$contents/Helpers/muser" \
    "$contents/Helpers/muser-terminal" \
    "$contents/Helpers/muser-remote-qualify"
do
    [[ -x "$binary" ]] || { echo "release binary is absent: $binary" >&2; exit 1; }
    codesign --force --options runtime --timestamp --sign "$identity" "$binary"
    codesign --verify --strict --verbose=2 "$binary"
    signature_info=$(codesign -dv --verbose=4 "$binary" 2>&1)
    grep -Fq "Authority=Developer ID Application:" <<<"$signature_info" || {
        echo "$(basename "$binary") did not receive a Developer ID signature" >&2
        exit 1
    }
    binary_team=$(awk -F= '$1 == "TeamIdentifier" { print $2 }' <<<"$signature_info")
    [[ -n "$binary_team" && "$binary_team" != "not set" ]] || {
        echo "$(basename "$binary") signature has no team identifier" >&2
        exit 1
    }
    if [[ -z "$team_identifier" ]]; then
        team_identifier=$binary_team
    elif [[ "$team_identifier" != "$binary_team" ]]; then
        echo "release binaries were signed by different teams" >&2
        exit 1
    fi
done

codesign --force --options runtime --timestamp --sign "$identity" "$app"
codesign --verify --deep --strict --verbose=2 "$app"
syspolicy_check notary-submission "$app"
app_signature=$(codesign -dv --verbose=4 "$app" 2>&1)
grep -Fq "Authority=Developer ID Application:" <<<"$app_signature" || {
    echo "Muser.app did not receive a Developer ID signature" >&2
    exit 1
}
grep -Fq "TeamIdentifier=$team_identifier" <<<"$app_signature" || {
    echo "Muser.app and its helpers were signed by different teams" >&2
    exit 1
}
"$contents/MacOS/Muser" --check-bundle
"$contents/Helpers/muser-terminal" --check-bundle

image="$scratch/$official_name"
version=${bundle_name#muser-}
version=${version%-macos-arm64}
hdiutil create -quiet -fs 'Journaled HFS+' -format UDZO \
    -volname "Muser $version" -srcfolder "$image_root" "$image"
codesign --force --timestamp --sign "$identity" "$image"
codesign --verify --strict --verbose=2 "$image"
image_signature=$(codesign -dv --verbose=4 "$image" 2>&1)
grep -Fq "Authority=Developer ID Application:" <<<"$image_signature" || {
    echo "disk image did not receive a Developer ID signature" >&2
    exit 1
}
grep -Fq "TeamIdentifier=$team_identifier" <<<"$image_signature" || {
    echo "disk image and binaries were signed by different teams" >&2
    exit 1
}
submitted_sha256=$(shasum -a 256 "$image" | awk '{print $1}')

notary_result="$scratch/notary-result.json"
if ! xcrun notarytool submit "$image" \
    --keychain-profile "$keychain_profile" \
    --wait --timeout "$timeout" --no-progress --output-format json \
    > "$notary_result"; then
    echo "Apple notarization submission failed" >&2
    exit 1
fi
read -r submission_id status created_date < <(
    python3 - "$notary_result" <<'PY'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(data.get("id", ""), data.get("status", ""), data.get("createdDate", ""))
PY
)
[[ -n "$submission_id" ]] || {
    echo "Apple notarization response omitted its submission ID" >&2
    exit 1
}
if [[ "$status" != "Accepted" ]]; then
    log="$scratch/notary-log.json"
    xcrun notarytool log "$submission_id" "$log" \
        --keychain-profile "$keychain_profile" >/dev/null 2>&1 || true
    python3 - "$status" "$log" <<'PY'
import json
import pathlib
import sys

print(f"Apple notarization status is {sys.argv[1] or 'unknown'}", file=sys.stderr)
path = pathlib.Path(sys.argv[2])
if path.is_file():
    data = json.loads(path.read_text(encoding="utf-8"))
    for issue in data.get("issues", []):
        message = issue.get("message", "notarization issue")
        severity = issue.get("severity", "error")
        print(f"{severity}: {message}", file=sys.stderr)
PY
    exit 1
fi

notary_log="$scratch/notary-log.json"
xcrun notarytool log "$submission_id" "$notary_log" \
    --keychain-profile "$keychain_profile" >/dev/null
read -r notary_log_sha256 ticket_component_count < <(
    python3 - "$notary_log" "$submitted_sha256" <<'PY'
import hashlib
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected_sha = sys.argv[2]
data = json.loads(path.read_text(encoding="utf-8"))
if data.get("status") != "Accepted":
    raise SystemExit("Apple notarization log is not Accepted")
if data.get("sha256") != expected_sha:
    raise SystemExit("Apple notarization log names different submitted bytes")
issues = data.get("issues")
if issues not in (None, []):
    raise SystemExit("Apple accepted the image with unexpected logged issues")
tickets = data.get("ticketContents")
if not isinstance(tickets, list):
    raise SystemExit("Apple notarization log omitted ticket contents")
paths = [item.get("path", "") for item in tickets if isinstance(item, dict)]
required = [
    "Muser.app",
    "Muser.app/Contents/MacOS/Muser",
    "Muser.app/Contents/Helpers/muser",
    "Muser.app/Contents/Helpers/muser-terminal",
    "Muser.app/Contents/Helpers/muser-remote-qualify",
]
for suffix in required:
    if not any(value.endswith(suffix) for value in paths):
        raise SystemExit(f"Apple notarization ticket omitted {suffix}")
digest = hashlib.sha256(path.read_bytes()).hexdigest()
print(digest, len(tickets))
PY
)

# A disk image supports an attached ticket. The artifact users download is
# the post-staple image, and the adjacent receipt binds both its final digest
# and the digest that Apple accepted before stapling.
xcrun stapler staple "$image"
xcrun stapler validate "$image"
artifact_sha256=$(shasum -a 256 "$image" | awk '{print $1}')

publish="$scratch/publish"
mkdir -p "$publish"
install -m 0644 "$image" "$publish/$official_name"
(
    cd "$publish"
    shasum -a 256 "$official_name" > "$official_name.sha256"
)
python3 - \
    "$publish/$official_name.notarization.json" \
    "$official_name" \
    "$source_archive_sha256" \
    "$submitted_sha256" \
    "$artifact_sha256" \
    "$team_identifier" \
    "$submission_id" \
    "$notary_log_sha256" \
    "$ticket_component_count" \
    "$created_date" <<'PY'
import json
import pathlib
import sys

(
    output,
    artifact,
    source_sha,
    submitted_sha,
    artifact_sha,
    team,
    submission_id,
    notary_log_sha,
    ticket_component_count,
    created_date,
) = sys.argv[1:]
receipt = {
    "schema": "muser.apple-notarization.v2",
    "artifact": artifact,
    "artifact_sha256": artifact_sha,
    "bundle_identifier": "org.high-performance-ai-lab.muser",
    "entrypoint": "Muser.app",
    "notary_log_sha256": notary_log_sha,
    "notary_ticket_components": int(ticket_component_count),
    "source_archive_sha256": source_sha,
    "submitted_image_sha256": submitted_sha,
    "team_identifier": team,
    "submission_id": submission_id,
    "status": "Accepted",
}
if created_date:
    receipt["created_at"] = created_date
pathlib.Path(output).write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

"$(dirname "$0")/verify_notarized_user_bundle.sh" "$publish/$official_name"

install -m 0644 "$publish/$official_name" "$official"
install -m 0644 "$publish/$official_name.sha256" "$official_checksum"
install -m 0644 "$publish/$official_name.notarization.json" "$official_receipt"
echo "$official"
echo "$official_checksum"
echo "$official_receipt"
