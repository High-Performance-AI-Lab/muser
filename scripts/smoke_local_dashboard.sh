#!/usr/bin/env bash
set -euo pipefail

# CPU-only, local product smoke for the secured dashboard and management API.
# Set MUSER_SMOKE_HOLD=1 to leave the server running for the manual browser
# checks printed at the end. Temporary PKI and evidence are intentionally kept.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${CARGO_TARGET_DIR:-$(mktemp -d)/target-dashboard-smoke}"
TMP_BASE="${TMPDIR:-/tmp}"
PORT="${MUSER_SMOKE_PORT:-4957}"
BASE="https://localhost:${PORT}"
mkdir -p "${TARGET_DIR}" "${TMP_BASE}"
SMOKE_DIR="$(mktemp -d "${TMP_BASE%/}/muser-human-smoke.XXXXXX")"
SERVER_PID=""

finish() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap finish EXIT INT TERM

for command in cargo curl jq node openssl; do
  command -v "${command}" >/dev/null || {
    printf 'missing required command: %s\n' "${command}" >&2
    exit 2
  }
done

export CARGO_TARGET_DIR="${TARGET_DIR}"
export TMPDIR="${TMP_BASE}"

cd "${ROOT_DIR}"
if [[ -n "${MUSER_SMOKE_BIN:-}" ]]; then
  BIN="${MUSER_SMOKE_BIN}"
  [[ -x "${BIN}" && -f "${BIN}" ]] || {
    printf 'MUSER_SMOKE_BIN is not an executable regular file: %s\n' "${BIN}" >&2
    exit 2
  }
else
  cargo build --locked -p muser-server --no-default-features
  BIN="${TARGET_DIR}/debug/muser"
fi

openssl rand -hex -out "${SMOKE_DIR}/api-key" 32
chmod 600 "${SMOKE_DIR}/api-key"
"${BIN}" tls init --dir "${SMOKE_DIR}/pki"
"${BIN}" tls issue \
  --dir "${SMOKE_DIR}/pki" \
  --name localhost \
  --san localhost \
  --san 127.0.0.1 \
  --out-dir "${SMOKE_DIR}/server"

"${BIN}" serve \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --tls-cert "${SMOKE_DIR}/server/server.pem" \
  --tls-key "${SMOKE_DIR}/server/server-key.pem" \
  --api-key-file "${SMOKE_DIR}/api-key" \
  --backend cpu >"${SMOKE_DIR}/server.log" 2>&1 &
SERVER_PID=$!

CA="${SMOKE_DIR}/pki/ca.pem"
KEY="$(tr -d '\r\n' < "${SMOKE_DIR}/api-key")"
for _ in $(seq 1 100); do
  if curl --silent --fail --cacert "${CA}" "${BASE}/healthz" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    printf 'server exited during startup\n' >&2
    tail -n 40 "${SMOKE_DIR}/server.log" >&2
    exit 1
  fi
  sleep 0.05
done
curl --silent --fail --cacert "${CA}" "${BASE}/healthz" >/dev/null

expect_code() {
  local label="$1" expected="$2" actual="$3"
  if [[ "${actual}" != "${expected}" ]]; then
    printf '%s: expected HTTP %s, got %s\n' "${label}" "${expected}" "${actual}" >&2
    exit 1
  fi
  printf 'ok  %-28s HTTP %s\n' "${label}" "${actual}"
}

code="$(curl --silent --show-error --cacert "${CA}" \
  -o "${SMOKE_DIR}/dashboard.html" -w '%{http_code}' "${BASE}/")"
expect_code "dashboard asset" 200 "${code}"
node -e '
const fs=require("fs"),vm=require("vm");
const html=fs.readFileSync(process.argv[1],"utf8");
const match=html.match(/<script>([\s\S]*?)<\/script>/);
if(!match) throw new Error("dashboard has no inline script");
new vm.Script(match[1], {filename:"muser-dashboard.inline.js"});
if(!html.includes("id=\"loginBtn\"")) throw new Error("dashboard sign-in control missing");
' "${SMOKE_DIR}/dashboard.html"
printf 'ok  %-28s parsed\n' "dashboard JavaScript"

code="$(curl --silent --show-error --cacert "${CA}" \
  -o "${SMOKE_DIR}/unauth.json" -w '%{http_code}' "${BASE}/snapshot")"
expect_code "unauthenticated snapshot" 401 "${code}"

code="$(curl --silent --show-error --cacert "${CA}" \
  -H "Authorization: Bearer ${KEY}" \
  -o "${SMOKE_DIR}/snapshot.json" -w '%{http_code}' "${BASE}/snapshot")"
expect_code "bearer snapshot" 200 "${code}"
jq -e '.schema_version == 1 and .cluster.weights_bytes == 0' \
  "${SMOKE_DIR}/snapshot.json" >/dev/null

read -r login_code login_http < <(
  curl --silent --show-error --http2 --cacert "${CA}" \
    -H "Authorization: Bearer ${KEY}" \
    -H "Origin: ${BASE}" \
    -c "${SMOKE_DIR}/cookies.txt" \
    -o "${SMOKE_DIR}/login.json" \
    -w '%{http_code} %{http_version}\n' \
    -X POST "${BASE}/v1/dashboard/login"
)
expect_code "HTTP/2 dashboard login" 200 "${login_code}"
[[ "${login_http}" == "2" ]] || {
  printf 'dashboard login did not negotiate HTTP/2\n' >&2
  exit 1
}
CSRF="$(jq -er '.csrf_token | select(length > 20)' "${SMOKE_DIR}/login.json")"

code="$(curl --silent --show-error --cacert "${CA}" \
  -b "${SMOKE_DIR}/cookies.txt" \
  -o "${SMOKE_DIR}/snapshot-cookie.json" -w '%{http_code}' "${BASE}/snapshot")"
expect_code "cookie snapshot" 200 "${code}"
code="$(curl --silent --show-error --cacert "${CA}" \
  -b "${SMOKE_DIR}/cookies.txt" \
  -o "${SMOKE_DIR}/metrics.txt" -w '%{http_code}' "${BASE}/metrics")"
expect_code "cookie Prometheus metrics" 200 "${code}"
grep -q '^completion_traffic_tok_s_10s ' "${SMOKE_DIR}/metrics.txt"

set +e
curl --silent --max-time 2 --cacert "${CA}" \
  -b "${SMOKE_DIR}/cookies.txt" \
  -o "${SMOKE_DIR}/telemetry.sse" "${BASE}/telemetry"
telemetry_rc=$?
set -e
[[ "${telemetry_rc}" == 0 || "${telemetry_rc}" == 28 ]]
grep -q '^event: snapshot' "${SMOKE_DIR}/telemetry.sse"
printf 'ok  %-28s snapshot received\n' "telemetry SSE"

auth_headers=(
  -b "${SMOKE_DIR}/cookies.txt"
  -H "Origin: ${BASE}"
  -H "x-csrf-token: ${CSRF}"
)
code="$(curl --silent --show-error --cacert "${CA}" \
  "${auth_headers[@]}" -H 'Content-Type: application/json' \
  --data '{"id":"human-smoke"}' \
  -o "${SMOKE_DIR}/session-create.json" -w '%{http_code}' \
  "${BASE}/v1/sessions")"
expect_code "session create" 201 "${code}"
jq -e '.id == "human-smoke" and .revision == 0' \
  "${SMOKE_DIR}/session-create.json" >/dev/null

for endpoint in "/v1/sessions" "/v1/sessions/human-smoke"; do
  code="$(curl --silent --show-error --cacert "${CA}" \
    -b "${SMOKE_DIR}/cookies.txt" \
    -o "${SMOKE_DIR}/session-read.json" -w '%{http_code}' "${BASE}${endpoint}")"
  expect_code "session read ${endpoint}" 200 "${code}"
done

code="$(curl --silent --show-error --cacert "${CA}" \
  "${auth_headers[@]}" -X POST \
  -o "${SMOKE_DIR}/ticket.json" -w '%{http_code}' "${BASE}/v1/ws-tickets")"
expect_code "WebSocket ticket" 200 "${code}"
TICKET="$(jq -er '.ticket | select(length > 20)' "${SMOKE_DIR}/ticket.json")"
NODE_EXTRA_CA_CERTS="${CA}" node -e '
const ws=new WebSocket(`wss://localhost:'"${PORT}"'/stream?ticket=${encodeURIComponent(process.argv[1])}`);
const timer=setTimeout(()=>process.exit(2),5000);
const frames=[];
ws.onmessage=event=>{
  frames.push(JSON.parse(event.data).type);
  if(frames.length===2){
    clearTimeout(timer);
    if(frames[0]!=="hello" || frames[1]!=="snapshot") process.exit(3);
    process.exit(0);
  }
};
ws.onerror=()=>process.exit(1);
' "${TICKET}"
printf 'ok  %-28s hello,snapshot\n' "WebSocket schema v2"

code="$(curl --silent --show-error --http1.1 --cacert "${CA}" \
  -H 'Connection: Upgrade' -H 'Upgrade: websocket' \
  -H 'Sec-WebSocket-Version: 13' \
  -H 'Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==' \
  -o "${SMOKE_DIR}/ticket-reuse.json" -w '%{http_code}' \
  "${BASE}/stream?ticket=${TICKET}")"
expect_code "single-use ticket reuse" 401 "${code}"

code="$(curl --silent --show-error --cacert "${CA}" \
  -H 'Content-Type: application/json' \
  --data '{"messages":[],"bogus":1}' \
  -o "${SMOKE_DIR}/strict-chat.json" -w '%{http_code}' \
  "${BASE}/v1/chat/completions")"
expect_code "strict unknown request field" 400 "${code}"
jq -e '.error.message | contains("unknown field `bogus`")' \
  "${SMOKE_DIR}/strict-chat.json" >/dev/null

code="$(curl --silent --show-error --cacert "${CA}" \
  -H 'Content-Type: application/json; charset=utf-8' \
  --data '{"messages":[]}' \
  -o "${SMOKE_DIR}/strict-content-type.json" -w '%{http_code}' \
  "${BASE}/v1/chat/completions")"
expect_code "strict JSON content type" 415 "${code}"

code="$(curl --silent --show-error --cacert "${CA}" \
  "${auth_headers[@]}" -X DELETE \
  -o "${SMOKE_DIR}/session-delete.json" -w '%{http_code}' \
  "${BASE}/v1/sessions/human-smoke")"
expect_code "session delete" 204 "${code}"

printf '\nAutomated local dashboard/API smoke passed.\n'
printf 'Evidence: %s\n' "${SMOKE_DIR}"
printf 'Manual browser check:\n'
printf '  1. Trust %s in a temporary/test keychain.\n' "${CA}"
printf '  2. Open %s and choose Sign in.\n' "${BASE}"
printf '  3. Paste the key from %s; verify Live Telemetry and no console errors.\n' "${SMOKE_DIR}/api-key"

if [[ "${MUSER_SMOKE_HOLD:-0}" == "1" ]]; then
  printf 'Server is running (PID %s); press Ctrl-C when finished.\n' "${SERVER_PID}"
  wait "${SERVER_PID}"
fi
