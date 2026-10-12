#!/usr/bin/env bash
set -euo pipefail

root=$(git rev-parse --show-toplevel)
identity="$root/scripts/gx10/vllm/native_onboarding_identity_v1.json"

for tool in curl jq; do
    command -v "$tool" >/dev/null || {
        echo "public onboarding asset gate needs $tool" >&2
        exit 2
    }
done

checked=0
probe() {
    local label=$1
    local url=$2
    local code
    local bytes
    local result
    result=$(curl --silent --show-error --location --fail --max-time 60 \
        --range 0-0 --output /dev/null --write-out '%{http_code} %{size_download}' "$url")
    read -r code bytes <<<"$result"
    if [[ "$code" != 206 || "$bytes" != 1 ]]; then
        echo "$label is not anonymously range-readable (HTTP $code, $bytes bytes)" >&2
        exit 1
    fi
    printf 'ok  %s\n' "$label"
    checked=$((checked + 1))
}

while IFS=$'\t' read -r filename url; do
    probe "producer image/$filename" "$url"
done < <(jq -r '.producer_image.archive.parts[] | [.filename, .url] | @tsv' "$identity")

while IFS=$'\t' read -r filename url; do
    probe "Mac decoder/$filename" "$url"
done < <(jq -r '.consumer.parts[] | [.filename, .url] | @tsv' "$identity")

checkpoint_repository=$(jq -r '.checkpoint.repository' "$identity")
checkpoint_revision=$(jq -r '.checkpoint.revision' "$identity")
while IFS= read -r filename; do
    probe "NVFP4 checkpoint/$filename" \
        "https://huggingface.co/$checkpoint_repository/resolve/$checkpoint_revision/$filename?download=true"
done < <(jq -r '.checkpoint.files[].filename' "$identity")

runtime_base='https://github.com/High-Performance-AI-Lab/muser/releases/download/nvfp4-consumer-d5109a1-v1'
probe 'Metal runtime' "$runtime_base/llama-metal-89e0aa6fd362.metallib"
probe 'Metal source receipt' "$runtime_base/llama-metal-89e0aa6fd362-source-receipt.json"

video_url='https://github.com/user-attachments/assets/02b6e368-fe46-4167-a7f0-1380e0ce2a47'
if ! grep --fixed-strings --line-regexp --quiet "$video_url" "$root/README.md"; then
    echo 'root README does not contain the native GitHub video attachment' >&2
    exit 1
fi
video_payload=$(jq --compact-output --null-input \
    --arg text "$video_url" \
    '{text: $text, mode: "gfm", context: "High-Performance-AI-Lab/muser"}')
video_render=$(curl --silent --show-error --fail --max-time 60 \
    --request POST \
    --header 'Accept: application/vnd.github+json' \
    --header 'Content-Type: application/json' \
    --header 'X-GitHub-Api-Version: 2022-11-28' \
    --data "$video_payload" \
    https://api.github.com/markdown)
if [[ "$video_render" != *'<video '* ]]; then
    echo 'root README attachment does not render as a native GitHub video player' >&2
    exit 1
fi
printf 'ok  onboarding video player\n'
checked=$((checked + 1))

printf 'public onboarding asset gate passed (%d assets)\n' "$checked"
