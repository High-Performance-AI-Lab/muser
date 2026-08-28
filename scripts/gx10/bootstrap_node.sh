#!/usr/bin/env bash
# Idempotent GX10 node bootstrap: uploaded once, driven repeatedly by the
# orchestrator over SSH. Every subcommand below is safe to re-run and must
# converge to the same end state instead of erroring on "already done".
#
# Contract (see docs referenced by the onboarding effort):
#   - probe:  always prints exactly one JSON line (own schema) so the
#             orchestrator can parse it without depending on --json.
#   - model/daemon/stop/status: human text on stdout by default; with
#     --json, ONLY muser.node-progress.v2 lines go to stdout (the
#     orchestrator relays stdout verbatim as SSE) - anything else goes to
#     stderr so it never corrupts that stream.
#   - exit codes: 0 ok, 1 operational failure, 2 usage error, 3 (model
#     subcommand only) "no local copy and no --source: upload me".
#
# Deliberately no `set -e`: a trap swallowing *which* step failed is worse
# than explicit checks. `set -u -o pipefail` still catch unset vars and
# masked pipeline failures without hiding the failing command.
set -u -o pipefail

# Plain bash + coreutils + curl + docker + systemctl only - no new deps, and
# nothing here may ssh out or touch a GPU (bootstrap runs ON the node).
readonly PROGRESS_SCHEMA="muser.node-progress.v2"
readonly PROBE_SCHEMA="muser.node-probe.v1"
readonly DAEMON_STATUS_SCHEMA="muser.node-daemon-status.v1"
readonly DAEMON_UNIT="muser-prefilld.service"
readonly TMUX_SESSION="muser-prefilld"
readonly WARM_POLL_SECONDS=60
readonly NATIVE_WARM_POLL_SECONDS=900

json_mode=0
dry_run=0

print_usage() {
    cat <<'EOF'
usage: bootstrap_node.sh [--json] [--dry-run] <subcommand> [args...]

subcommands:
  probe
      Print one muser.node-probe.v1 JSON line: arch, driver_version,
      docker_ok, disk_free_gib, mem_free_gib.

  model --dir D --name N --bytes B --sha256 H [--source URL]
      Ensure D/N exists and matches sha256 H. Downloads from --source when
      given; otherwise exits 3 ("upload me") if no verified copy exists.

  daemon --lane LANE --model M [--dflash F] [--systemd|--tmux]
  daemon --native --lane LANE --checkpoint DIR [--systemd|--tmux]
      Install and start the prefill producer for LANE, verifying it opens
      its configured listen port (60s llama.cpp, 900s native warm start).

  stop [--systemd|--tmux]
  status [--json]
      Operability commands for the installed daemon.

global flags (must precede the subcommand):
  --json      emit muser.node-progress.v2 JSON lines instead of text
  --dry-run   report the plan without mutating state (model/daemon/stop)
EOF
}

usage_error() {
    printf 'error: %s\n' "$1" >&2
    print_usage >&2
    exit 2
}

json_escape() {
    local s=$1
    s=${s//\\/\\\\}
    s=${s//\"/\\\"}
    s=${s//$'\n'/\\n}
    s=${s//$'\r'/\\r}
    s=${s//$'\t'/\\t}
    printf '%s' "$s"
}

# emit STEP STATUS DETAIL [DATA_JSON]
# STATUS is start|ok|fail|info|planned per the progress protocol. In --json mode
# this is the ONLY thing allowed to touch stdout; in text mode it narrates
# to stderr so stdout stays free for a subcommand's actual result text.
emit() {
    local step=$1 status=$2 detail=$3 data=${4:-}
    if [ "$json_mode" = 1 ]; then
        if [ -n "$data" ]; then
            printf '{"schema":"%s","step":"%s","status":"%s","detail":"%s","data":%s}\n' \
                "$PROGRESS_SCHEMA" "$step" "$status" "$(json_escape "$detail")" "$data"
        else
            printf '{"schema":"%s","step":"%s","status":"%s","detail":"%s"}\n' \
                "$PROGRESS_SCHEMA" "$step" "$status" "$(json_escape "$detail")"
        fi
    else
        printf '[%s] %s: %s\n' "$step" "$status" "$detail" >&2
    fi
}

# plan STEP TEXT [DATA_JSON] - dry-run reporting. Text mode prints a plain
# "PLAN: ..." line to stdout (that IS the subcommand's result in dry-run);
# json mode folds it into a planned event so stdout stays pure JSON.
plan() {
    local step=$1 text=$2 data=${3:-}
    if [ "$json_mode" = 1 ]; then
        emit "$step" planned "$text" "$data"
    else
        printf 'PLAN: %s\n' "$text"
    fi
}

# fail STEP DETAIL [DATA_JSON] [EXIT_CODE=1]
fail() {
    local step=$1 detail=$2 data=${3:-} code=${4:-1}
    emit "$step" fail "$detail" "$data"
    exit "$code"
}

require_cmd() {
    local bin=$1 step=$2 hint=$3
    command -v "$bin" >/dev/null 2>&1 || fail "$step" "required command not found: $bin ($hint)"
}

model_sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum -- "$1" 2>/dev/null | awk '{print $1}'
    else
        shasum -a 256 -- "$1" 2>/dev/null | awk '{print $1}'
    fi
}

# systemd_quote VALUE -> a double-quoted token safe to splice into a
# systemd ExecStart= line (systemd's own quoting dialect: \\ \" and a
# literal $ must be doubled so it is not treated as variable expansion).
systemd_quote() {
    local s=$1
    s=${s//\\/\\\\}
    s=${s//\"/\\\"}
    s=${s//\$/\$\$}
    printf '"%s"' "$s"
}

# sh_quote VALUE -> a single-quoted token safe to splice into a POSIX
# shell command string (used for the tmux fallback pane command).
sh_quote() {
    local s=$1
    printf "'%s'" "${s//\'/\'\\\'\'}"
}

render_systemd_argv() {
    local out="" tok
    for tok in "$@"; do
        out="$out $(systemd_quote "$tok")"
    done
    printf '%s' "${out# }"
}

render_shell_argv() {
    local out="" tok
    for tok in "$@"; do
        out="$out $(sh_quote "$tok")"
    done
    printf '%s' "${out# }"
}

# extract_json_string_field FILE FIELD / extract_json_int_field FILE FIELD
# Handoff configs are our own flat, non-nested JSON (see muser_prefilld.py
# load_config); a targeted grep avoids pulling in a JSON parser dependency.
extract_json_string_field() {
    grep -o "\"$2\"[[:space:]]*:[[:space:]]*\"[^\"]*\"" "$1" 2>/dev/null \
        | head -n1 | sed -E 's/.*:[[:space:]]*"([^"]*)"/\1/'
}

extract_json_int_field() {
    grep -o "\"$2\"[[:space:]]*:[[:space:]]*[0-9]\{1,\}" "$1" 2>/dev/null \
        | head -n1 | sed -E 's/.*:[[:space:]]*([0-9]+)/\1/'
}

poll_listen_port() {
    local host=$1 port=$2 timeout=$3
    local deadline=$((SECONDS + timeout))
    while [ "$SECONDS" -lt "$deadline" ]; do
        if (exec 3<>"/dev/tcp/$host/$port") 2>/dev/null; then
            exec 3>&- 3<&-
            return 0
        fi
        sleep 1
    done
    return 1
}

daemon_log_tail() {
    local mode=$1 lane=$2
    if [ "$mode" = "tmux" ]; then
        tail -n 50 -- "$lane/prefilld-console.log" 2>/dev/null
    else
        local -a jc=(journalctl -u "$DAEMON_UNIT" -n 50 --no-pager)
        [ "$(id -u)" = "0" ] || jc=(journalctl --user -u "$DAEMON_UNIT" -n 50 --no-pager)
        "${jc[@]}" 2>/dev/null
    fi
}

# ---------------------------------------------------------------- probe --
cmd_probe() {
    local dir="/"
    while [ $# -gt 0 ]; do
        case "$1" in
            --dir) [ $# -ge 2 ] || usage_error "--dir requires a path"; dir=$2; shift 2 ;;
            *) usage_error "probe: unknown argument: $1" ;;
        esac
    done

    local arch
    arch=$(uname -m)

    local driver_version="null"
    if command -v nvidia-smi >/dev/null 2>&1; then
        local raw
        raw=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:]')
        [ -n "$raw" ] && driver_version="\"$(json_escape "$raw")\""
    fi

    local docker_ok=false
    if command -v docker >/dev/null 2>&1; then
        if command -v timeout >/dev/null 2>&1; then
            timeout 5 docker info >/dev/null 2>&1 && docker_ok=true
        else
            docker info >/dev/null 2>&1 && docker_ok=true
        fi
    fi

    local disk_free_gib="null" mem_free_gib="null"
    local disk_kib
    disk_kib=$(df -Pk "$dir" 2>/dev/null | awk 'NR==2 {print $4}')
    [ -n "${disk_kib:-}" ] && disk_free_gib=$(awk -v k="$disk_kib" 'BEGIN{printf "%.1f", k/1048576}')
    local mem_kib
    mem_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null)
    [ -n "${mem_kib:-}" ] && mem_free_gib=$(awk -v k="$mem_kib" 'BEGIN{printf "%.1f", k/1048576}')

    local payload
    payload=$(printf '{"schema":"%s","arch":"%s","driver_version":%s,"docker_ok":%s,"disk_free_gib":%s,"mem_free_gib":%s}' \
        "$PROBE_SCHEMA" "$(json_escape "$arch")" "$driver_version" "$docker_ok" "$disk_free_gib" "$mem_free_gib")

    # probe always emits exactly one JSON line, --json or not: the
    # orchestrator parses it directly rather than waiting on a start/ok pair.
    if [ "$json_mode" = 1 ]; then
        emit preflight ok "node probe complete" "$payload"
    else
        printf '%s\n' "$payload"
    fi
}

# ---------------------------------------------------------------- model --
cmd_model() {
    local dir="" name="" bytes="" sha256="" source=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --dir) [ $# -ge 2 ] || usage_error "--dir requires a path"; dir=$2; shift 2 ;;
            --name) [ $# -ge 2 ] || usage_error "--name requires a value"; name=$2; shift 2 ;;
            --bytes) [ $# -ge 2 ] || usage_error "--bytes requires a value"; bytes=$2; shift 2 ;;
            --sha256) [ $# -ge 2 ] || usage_error "--sha256 requires a value"; sha256=$2; shift 2 ;;
            --source) [ $# -ge 2 ] || usage_error "--source requires a URL"; source=$2; shift 2 ;;
            *) usage_error "model: unknown argument: $1" ;;
        esac
    done
    [ -n "$dir" ] || fail model "model requires --dir" "" 2
    [ -n "$name" ] || fail model "model requires --name" "" 2
    [ -n "$bytes" ] || fail model "model requires --bytes" "" 2
    case "$bytes" in *[!0-9]*|0) fail model "--bytes must be a positive integer" "" 2 ;; esac
    [ -n "$sha256" ] || fail model "model requires --sha256" "" 2
    if [ ${#sha256} -ne 64 ]; then
        fail model "--sha256 must be 64 lowercase hex characters" "" 2
    fi
    # Spell out the alphabet: locale collation can make the range `a-f`
    # admit uppercase letters on the stock macOS Bash used by public CI.
    case "$sha256" in
        *[!0123456789abcdef]*) fail model "--sha256 must be 64 lowercase hex characters" "" 2 ;;
    esac

    emit model start "verifying $name in $dir"

    if [ "$dry_run" = 1 ]; then
        [ -d "$dir" ] || plan model "would create model directory $dir"
    else
        mkdir -p -- "$dir" || fail model "cannot create model directory: $dir (check permissions/free disk)"
    fi

    local target="$dir/$name"
    if [ -f "$target" ]; then
        local actual actual_bytes
        actual_bytes=$(wc -c < "$target" | tr -d '[:space:]')
        actual=$(model_sha256 "$target")
        if [ "$actual_bytes" = "$bytes" ] && [ -n "$actual" ] && [ "$actual" = "$sha256" ]; then
            if [ "$dry_run" = 1 ]; then
                plan model "verified existing model would remain unchanged at $target"
            else
                emit model ok "model already present and verified" \
                    "$(printf '{"path":"%s"}' "$(json_escape "$target")")"
            fi
            return 0
        fi
        if [ "$dry_run" = 1 ]; then
            plan model "would move aside mismatched file at $target (never deleted)"
        else
            local stamp aside
            stamp=$(date -u +%Y%m%dT%H%M%SZ)
            aside="$target.mismatch-$stamp"
            mv -- "$target" "$aside" || fail model "could not move aside mismatched file: $target"
            emit model info "moved mismatched file to $aside"
        fi
    fi

    if [ -z "$source" ]; then
        # Deliberate, not a failure: signals the orchestrator to scp the
        # model up itself and re-invoke this subcommand to verify it.
        emit model info "no --source and no verified copy at $target; upload required"
        exit 3
    fi

    if [ "$dry_run" = 1 ]; then
        plan model "would curl -fL -C - --retry 5 from $source to $target.partial, verify sha256, then rename to $target" \
            "$(printf '{"source":"%s","path":"%s"}' "$(json_escape "$source")" "$(json_escape "$target")")"
        return 0
    fi

    require_cmd curl model "install curl to fetch models"
    local partial="$target.partial"
    if ! curl -fL -C - --retry 5 --retry-connrefused -o "$partial" -- "$source"; then
        fail model "download failed from $source (network/URL problem; check connectivity and --source)"
    fi
    local downloaded
    local downloaded_bytes
    downloaded_bytes=$(wc -c < "$partial" | tr -d '[:space:]')
    if [ "$downloaded_bytes" != "$bytes" ]; then
        rm -f -- "$partial"
        fail model "downloaded file size mismatch (got $downloaded_bytes, want $bytes); removed partial download"
    fi
    downloaded=$(model_sha256 "$partial")
    if [ -z "$downloaded" ] || [ "$downloaded" != "$sha256" ]; then
        rm -f -- "$partial"
        fail model "downloaded file sha256 mismatch (got '${downloaded:-<none>}', want $sha256); removed partial download"
    fi
    mv -- "$partial" "$target" || fail model "atomic rename failed for $target"
    emit model ok "model downloaded and verified" \
        "$(printf '{"path":"%s"}' "$(json_escape "$target")")"
}

# --------------------------------------------------------------- daemon --
daemon_install_systemd() {
    local lane=$1 template=$2
    shift 2
    local -a exec_argv=("$@")

    [ -f "$template" ] || fail daemon "unit template missing from lane payload: $template (deploy step incomplete)" "" 1

    local root_mode=0
    [ "$(id -u)" = "0" ] && root_mode=1
    local unit_dir env_dir wanted_by
    local -a systemctl_cmd=(systemctl)
    if [ "$root_mode" = 1 ]; then
        unit_dir="/etc/systemd/system"
        env_dir="/etc/muser"
        wanted_by="multi-user.target"
    else
        # --user managers have no multi-user.target; default.target is the
        # user-session equivalent for `systemctl --user enable`.
        unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
        env_dir="${XDG_CONFIG_HOME:-$HOME/.config}/muser"
        systemctl_cmd=(systemctl --user)
        wanted_by="default.target"
    fi
    local unit_path="$unit_dir/$DAEMON_UNIT"
    local env_path="$env_dir/gx10-prefilld.env"
    local exec_start
    exec_start=$(render_systemd_argv "${exec_argv[@]}")

    if [ "$dry_run" = 1 ]; then
        plan daemon "would install $unit_path (root=$root_mode) with ExecStart=$exec_start and WorkingDirectory=$lane, then daemon-reload + enable --now" \
            "$(printf '{"unit_path":"%s","root_mode":%s,"exec_start":"%s"}' \
                "$(json_escape "$unit_path")" "$root_mode" "$(json_escape "$exec_start")")"
        return 0
    fi

    mkdir -p -- "$unit_dir" "$env_dir" || fail daemon "cannot create systemd/env directories under $unit_dir / $env_dir (permissions?)"
    # Never clobber an existing env file: it may hold an operator-set
    # MUSER_LIVE_BATCH_FULL toggle (see muser_prefilld.py); only create it
    # if genuinely missing, and keep it 0600 (EnvironmentFile, not secrets,
    # but no reason to widen perms).
    if [ ! -f "$env_path" ]; then
        : > "$env_path" || fail daemon "cannot create env file: $env_path"
        chmod 0600 "$env_path" 2>/dev/null || true
    fi

    local content
    content=$(cat -- "$template") || fail daemon "cannot read unit template: $template"
    content=${content//@@MUSER_WORKING_DIRECTORY@@/$lane}
    content=${content//@@MUSER_EXEC_START@@/$exec_start}
    content=${content//@@MUSER_ENV_FILE@@/$env_path}
    content=${content//@@MUSER_WANTED_BY@@/$wanted_by}

    local tmp_unit
    tmp_unit=$(mktemp "$unit_dir/.muser-prefilld.service.XXXXXX") || fail daemon "cannot create temp unit file in $unit_dir"
    if ! printf '%s\n' "$content" > "$tmp_unit"; then
        rm -f -- "$tmp_unit"
        fail daemon "cannot write unit file content to $tmp_unit"
    fi
    if ! mv -- "$tmp_unit" "$unit_path"; then
        rm -f -- "$tmp_unit"
        fail daemon "cannot install unit file at $unit_path"
    fi

    if ! "${systemctl_cmd[@]}" daemon-reload; then
        fail daemon "systemctl daemon-reload failed (is systemd PID 1? DGX OS should have it)"
    fi
    if ! "${systemctl_cmd[@]}" enable "$DAEMON_UNIT"; then
        fail daemon "systemctl enable failed; check: ${systemctl_cmd[*]} status $DAEMON_UNIT"
    fi
    # A previous fail-closed enrollment can legitimately exhaust systemd's
    # start burst while the operator repairs its configuration. Clear only
    # this unit's failure latch before activating the newly installed,
    # independently validated generation.
    if ! "${systemctl_cmd[@]}" reset-failed "$DAEMON_UNIT"; then
        fail daemon "systemctl reset-failed failed; check: ${systemctl_cmd[*]} status $DAEMON_UNIT"
    fi
    # restart, not enable --now: enable --now is a no-op on an already-active
    # service, which left a daemon serving certificates its lane had since
    # replaced (caught live: pin mismatch after a re-enroll).
    if ! "${systemctl_cmd[@]}" restart "$DAEMON_UNIT"; then
        local tail
        tail=$(daemon_log_tail systemd "$lane")
        fail daemon "systemctl restart failed; log tail follows:
$tail"
    fi
    emit daemon info "unit installed and started at $unit_path"
}

daemon_install_tmux() {
    local lane=$1
    shift
    local -a exec_argv=("$@")
    local console_log="$lane/prefilld-console.log"
    local shell_cmd
    shell_cmd=$(render_shell_argv "${exec_argv[@]}")
    local pane_cmd
    pane_cmd="$shell_cmd 2>&1 | tee -a $(sh_quote "$console_log")"

    if [ "$dry_run" = 1 ]; then
        plan daemon "would (re)create tmux session $TMUX_SESSION in $lane running: $pane_cmd" \
            "$(printf '{"session":"%s","console_log":"%s"}' "$TMUX_SESSION" "$(json_escape "$console_log")")"
        return 0
    fi

    require_cmd tmux daemon "install tmux, or use --systemd"
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        emit daemon info "existing tmux session $TMUX_SESSION found; recreating for a clean restart"
        tmux kill-session -t "$TMUX_SESSION" || fail daemon "could not kill existing tmux session $TMUX_SESSION"
    fi
    if ! tmux new-session -d -s "$TMUX_SESSION" -c "$lane" "$pane_cmd"; then
        fail daemon "tmux new-session failed for $TMUX_SESSION"
    fi
    emit daemon info "tmux session $TMUX_SESSION started; console log $console_log"
}

cmd_daemon() {
    local lane="" model="" dflash="" checkpoint="" native=0 mode="systemd"
    while [ $# -gt 0 ]; do
        case "$1" in
            --lane) [ $# -ge 2 ] || usage_error "--lane requires a path"; lane=$2; shift 2 ;;
            --model) [ $# -ge 2 ] || usage_error "--model requires a path"; model=$2; shift 2 ;;
            --dflash) [ $# -ge 2 ] || usage_error "--dflash requires a path"; dflash=$2; shift 2 ;;
            --checkpoint) [ $# -ge 2 ] || usage_error "--checkpoint requires a path"; checkpoint=$2; shift 2 ;;
            --native) native=1; shift ;;
            --systemd) mode="systemd"; shift ;;
            --tmux) mode="tmux"; shift ;;
            *) usage_error "daemon: unknown argument: $1" ;;
        esac
    done
    [ -n "$lane" ] || fail daemon "daemon requires --lane" "" 2
    if [ "$native" = 1 ]; then
        [ -n "$checkpoint" ] || fail daemon "native daemon requires --checkpoint" "" 2
        [ -z "$model" ] || fail daemon "native daemon does not accept --model" "" 2
        [ -z "$dflash" ] || fail daemon "native daemon does not accept --dflash" "" 2
    else
        [ -n "$model" ] || fail daemon "daemon requires --model" "" 2
        [ -z "$checkpoint" ] || fail daemon "llama daemon does not accept --checkpoint" "" 2
    fi
    [ -d "$lane" ] || fail daemon "lane directory does not exist: $lane (run the deploy step first)"
    lane=$(cd -- "$lane" && pwd) || fail daemon "cannot resolve lane directory: $lane"

    # The deploy step pushes the llamacpp runtime into LANE/llamacpp (see
    # deploy.rs RUNTIME_FILES / MAKE_LANE); enroll.rs writes handoff.json
    # and the pki/ secrets directly under LANE. WorkingDirectory stays LANE
    # itself per the daemon contract - harmless either way since every path
    # baked into ExecStart below is absolute, never cwd-relative.
    local payload_dir="$lane/llamacpp"
    local prefilld_bin="$payload_dir/muser-prefilld"
    local unit_template="$payload_dir/$DAEMON_UNIT"
    local handoff="$lane/handoff.json"

    if [ "$native" = 1 ]; then
        prefilld_bin="$lane/vllm/muser_native_prefilld.py"
        [ -f "$prefilld_bin" ] || fail daemon "native prefilld missing from lane payload: $prefilld_bin (deploy step incomplete)"
        [ -d "$checkpoint" ] || fail daemon "checkpoint directory not found: $checkpoint (run the model step first)"
    else
        [ -f "$payload_dir/muser_prefilld.py" ] || fail daemon "muser_prefilld.py missing from lane payload: $payload_dir (deploy step incomplete)"
        [ -f "$prefilld_bin" ] || fail daemon "muser-prefilld launcher missing from lane payload: $prefilld_bin"
        [ -f "$model" ] || fail daemon "model file not found: $model (run the model step first)"
        if [ -n "$dflash" ] && [ ! -f "$dflash" ]; then
            fail daemon "dflash file not found: $dflash"
        fi
    fi
    [ -f "$handoff" ] || fail daemon "handoff config missing: $handoff (enroll step must run first)"

    local listen_host listen_port poll_host
    listen_host=$(extract_json_string_field "$handoff" listen_host)
    listen_port=$(extract_json_int_field "$handoff" listen_port)
    [ -n "$listen_host" ] || fail daemon "handoff config missing listen_host: $handoff"
    [ -n "$listen_port" ] || fail daemon "handoff config missing listen_port: $handoff"
    # A bind-any address (enroll.rs writes "0.0.0.0") isn't a valid dial
    # target on every stack; this poll runs on the node itself right after
    # starting the daemon, so the loopback equivalent is always reachable.
    case "$listen_host" in
        0.0.0.0) poll_host="127.0.0.1" ;;
        ::|::0|0:0:0:0:0:0:0:0) poll_host="::1" ;;
        *) poll_host=$listen_host ;;
    esac

    emit daemon start "installing $mode daemon for lane $lane"

    local -a exec_argv
    if [ "$native" = 1 ]; then
        exec_argv=(python3 "$prefilld_bin" --handoff-config "$handoff")
    else
        exec_argv=("$prefilld_bin" --model "$model")
        [ -n "$dflash" ] && exec_argv+=(--dflash "$dflash")
        exec_argv+=(--handoff-config "$handoff")
    fi

    case "$mode" in
        systemd) daemon_install_systemd "$lane" "$unit_template" "${exec_argv[@]}" ;;
        tmux) daemon_install_tmux "$lane" "${exec_argv[@]}" ;;
        *) fail daemon "unknown daemon mode: $mode" "" 2 ;;
    esac

    if [ "$dry_run" = 1 ]; then
        emit daemon planned "dry-run plan only; daemon not started"
        return 0
    fi

    local poll_seconds=$WARM_POLL_SECONDS
    [ "$native" = 1 ] && poll_seconds=$NATIVE_WARM_POLL_SECONDS
    if ! poll_listen_port "$poll_host" "$listen_port" "$poll_seconds"; then
        local tail
        tail=$(daemon_log_tail "$mode" "$lane")
        fail daemon "daemon did not open $listen_host:$listen_port within ${poll_seconds}s; log tail follows:
$tail"
    fi
    emit daemon ok "daemon listening on $listen_host:$listen_port" \
        "$(printf '{"listen_host":"%s","listen_port":%s}' "$(json_escape "$listen_host")" "$listen_port")"
}

# ----------------------------------------------------------------- stop --
cmd_stop() {
    local mode="systemd"
    while [ $# -gt 0 ]; do
        case "$1" in
            --systemd) mode="systemd"; shift ;;
            --tmux) mode="tmux"; shift ;;
            *) usage_error "stop: unknown argument: $1" ;;
        esac
    done
    emit daemon start "stopping $mode daemon"
    if [ "$dry_run" = 1 ]; then
        plan daemon "would stop the $mode daemon (idempotent no-op if not running)"
        return 0
    fi
    if [ "$mode" = "tmux" ]; then
        if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
            tmux kill-session -t "$TMUX_SESSION" || fail daemon "could not kill tmux session $TMUX_SESSION"
        fi
        emit daemon ok "tmux session stopped (or was not running)"
    else
        local -a systemctl_cmd=(systemctl)
        [ "$(id -u)" = "0" ] || systemctl_cmd=(systemctl --user)
        "${systemctl_cmd[@]}" disable --now "$DAEMON_UNIT" >/dev/null 2>&1
        emit daemon ok "systemd unit stopped and disabled (or was not installed)"
    fi
}

# --------------------------------------------------------------- status --
cmd_status() {
    local mode="systemd"
    while [ $# -gt 0 ]; do
        case "$1" in
            --systemd) mode="systemd"; shift ;;
            --tmux) mode="tmux"; shift ;;
            *) usage_error "status: unknown argument: $1" ;;
        esac
    done
    local running=false label
    if [ "$mode" = "tmux" ]; then
        label="tmux session $TMUX_SESSION"
        if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
            running=true
        fi
    else
        label="systemd unit $DAEMON_UNIT"
        local -a systemctl_cmd=(systemctl)
        [ "$(id -u)" = "0" ] || systemctl_cmd=(systemctl --user)
        if "${systemctl_cmd[@]}" is-active --quiet "$DAEMON_UNIT" 2>/dev/null; then
            running=true
        fi
    fi
    local payload
    payload=$(printf '{"schema":"%s","mode":"%s","running":%s}' "$DAEMON_STATUS_SCHEMA" "$mode" "$running")
    local word="stopped"
    [ "$running" = true ] && word="running"
    if [ "$json_mode" = 1 ]; then
        emit daemon info "$label is $word" "$payload"
    else
        printf '%s: %s\n' "$label" "$word"
    fi
}

# ------------------------------------------------------------------ main --
main() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --json) json_mode=1; shift ;;
            --dry-run) dry_run=1; shift ;;
            --help|-h) print_usage; exit 0 ;;
            --) shift; break ;;
            -*) usage_error "unknown global option: $1" ;;
            *) break ;;
        esac
    done
    local sub=${1:-}
    [ -n "$sub" ] || usage_error "missing subcommand"
    shift
    case "$sub" in
        probe) cmd_probe "$@" ;;
        model) cmd_model "$@" ;;
        daemon) cmd_daemon "$@" ;;
        stop) cmd_stop "$@" ;;
        status) cmd_status "$@" ;;
        *) usage_error "unknown subcommand: $sub" ;;
    esac
}

main "$@"
