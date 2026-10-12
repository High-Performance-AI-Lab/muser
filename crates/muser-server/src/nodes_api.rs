//! Node onboarding API — the server half of the dashboard's "Add node"
//! button.
//!
//! The button is a thin shell around the CLI: `POST /v1/nodes` spawns this
//! same binary as `muser node add <user@host>` and relays the child's
//! progress protocol (`muser.node-progress.v2`, one JSON object per stdout
//! line) to the browser as Server-Sent Events. The pipeline itself —
//! preflight, runtime deploy, model staging, keys/certs, daemon, verified
//! smoke handoff — lives in the CLI, so a terminal user and the button run
//! byte-identical code and there is exactly one implementation to trust.
//!
//! Three routes, all behind bearer or same-origin dashboard authentication;
//! LAN access additionally requires the native-TLS listener policy:
//! - `POST /v1/nodes` — start one onboarding job (202, or 409 while another
//!   job is running).
//! - `GET  /v1/nodes` — registry entries (`~/.muser/nodes.toml`) plus a live
//!   daemon TCP probe and the running-job flag.
//! - `GET  /v1/nodes/<name>/progress` — SSE: ring-buffer replay, then a live
//!   tail of the child's stdout until it exits.
//!
//! Two hard constraints shape the code below:
//!
//! 1. **One job at a time, globally.** Onboarding jobs contend for ssh and
//!    for the remote's docker daemon; two concurrent runs would interleave
//!    on the same remote host. [`NodeJobs`] holds at most one running job.
//! 2. **No new dependencies.** The registry is read by the small
//!    `[[node]]`-shaped TOML reader in this file rather than by adding a
//!    parser to the dependency set, and it is deliberately lenient: an entry
//!    it cannot understand is skipped, never fatal, so one malformed line
//!    cannot blank the dashboard's node list.

use std::collections::VecDeque;
use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::{Duration, Instant};

use crate::state::ServerState;

/// Where a deployed node's prefill daemon listens. `GET /v1/nodes` reports
/// reachability from a real TCP connect, never from the registry's own
/// `state` field — a registry says what the last run believed, a connect
/// says what is true now.
pub(crate) const DAEMON_PORT: u16 = 29591;
const DAEMON_PROBE_TIMEOUT: Duration = Duration::from_secs(1);

/// Progress lines retained per job for SSE replay. A full onboarding run
/// emits a few dozen; 500 keeps a whole verbose run without bounding on the
/// child's chattiness.
const PROGRESS_RING: usize = 500;
/// Finished jobs kept addressable so a browser that subscribes after the
/// run ended still gets the replay and the terminal event.
const RECENT_JOBS: usize = 8;
/// Inactivity ceiling on one onboarding run. A wedged ssh would otherwise
/// hold the single global job slot forever and make the button permanently
/// return 409. The clock resets on every progress event, so a slow run that
/// is still reporting (a cold smoke loads the target model per repetition)
/// is never killed mid-flight; only silence for this long is.
const JOB_DEADLINE: Duration = Duration::from_secs(30 * 60);
/// How often the reaper checks the child (cheap `waitpid(WNOHANG)`).
const REAP_POLL: Duration = Duration::from_millis(500);
/// Idle gap after which the progress stream emits an SSE comment, matching
/// the house streaming style in `httpd`.
const HEARTBEAT: Duration = Duration::from_secs(10);
/// Bound on the child's stderr we retain for the terminal event's detail.
const STDERR_TAIL_LINES: usize = 20;
/// Reconnection delay handed to the browser on the terminal event, in
/// milliseconds — an hour, i.e. "this run is over, do not come back".
const TERMINAL_RETRY_MS: u64 = 3_600_000;

const PROGRESS_SCHEMA: &str = "muser.node-progress.v2";
const JOB_SCHEMA: &str = "muser.node-job.v1";

/// Optional continuation supplied by `muser up`'s setup server. The regular
/// `muser serve` management surface remains enrollment-only; the one-button
/// launcher uses this callback to install the Mac decoder after the child has
/// exited successfully and released the cross-process topology lock.
pub(crate) type ActivationReporter = Arc<dyn Fn(&str, &str) + Send + Sync>;
pub(crate) type NodeActivator =
    Arc<dyn Fn(&str, ActivationReporter) -> Result<(), String> + Send + Sync>;

// ===================================================================== paths

/// The same Muser home the node CLI resolves: `$MUSER_HOME` when set,
/// otherwise `~/.muser`. The server and the child it spawns must call the
/// same resolver or the button can write one registry while the dashboard
/// reads another. `None` means neither location can be resolved, which makes
/// node management report as unavailable in the `/health` ledger.
pub(crate) fn muser_home() -> Option<PathBuf> {
    crate::node::muser_home().ok()
}

/// The shared registry (contract: atomically written by the CLI, read here).
pub(crate) fn registry_path() -> Option<PathBuf> {
    muser_home().map(|home| home.join("nodes.toml"))
}

fn node_dir(name: &str) -> Option<PathBuf> {
    muser_home().map(|home| home.join("nodes").join(name))
}

fn progress_log_path(name: &str) -> Option<PathBuf> {
    node_dir(name).map(|dir| dir.join("progress.log"))
}

/// Whether this process can actually run an onboarding job. Reported to
/// `GET /health` so the ledger states a capability this process has, not one
/// the source tree merely contains.
pub(crate) fn available() -> bool {
    muser_home().is_some() && std::env::current_exe().is_ok()
}

// ================================================================== registry

/// One `[[node]]` entry, exactly the shared registry shape. Everything past
/// the identity fields is optional because a `draft` entry is written before
/// the pipeline has produced any of it.
#[derive(Debug, Default, Clone, PartialEq)]
pub(crate) struct NodeEntry {
    pub(crate) name: String,
    pub(crate) host: String,
    pub(crate) user: String,
    pub(crate) role: Option<String>,
    pub(crate) state: Option<String>,
    pub(crate) key_path: Option<String>,
    pub(crate) lane_dir: Option<String>,
    pub(crate) container_image: Option<String>,
    pub(crate) container_receipt: Option<String>,
    pub(crate) runtime_sha256: Option<String>,
    pub(crate) consumer_model_path: Option<String>,
    pub(crate) pki_dir: Option<String>,
    pub(crate) hmac_key_id: Option<String>,
    pub(crate) hmac_epoch: Option<i64>,
    pub(crate) enrollment_version: Option<i64>,
    pub(crate) netqual_gbps: Option<f64>,
    pub(crate) connect_host: Option<String>,
    pub(crate) netqual_rtt_ms: Option<f64>,
    pub(crate) last_error: Option<String>,
    pub(crate) updated: Option<String>,
}

/// A scalar as the registry can spell it. The registry holds no arrays,
/// no inline tables, and no multi-line strings, so this is the whole grammar.
#[derive(Debug, Clone, PartialEq)]
enum Scalar {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
}

impl Scalar {
    fn as_string(&self) -> Option<String> {
        match self {
            Scalar::Str(value) => Some(value.clone()),
            Scalar::Int(value) => Some(value.to_string()),
            Scalar::Float(value) => Some(value.to_string()),
            Scalar::Bool(value) => Some(value.to_string()),
        }
    }

    fn as_int(&self) -> Option<i64> {
        match self {
            Scalar::Int(value) => Some(*value),
            Scalar::Str(value) => value.trim().parse().ok(),
            _ => None,
        }
    }

    fn as_float(&self) -> Option<f64> {
        match self {
            Scalar::Float(value) => Some(*value),
            Scalar::Int(value) => Some(*value as f64),
            Scalar::Str(value) => value.trim().parse().ok(),
            _ => None,
        }
    }
}

/// Read `~/.muser/nodes.toml`. A missing file is an empty registry, not an
/// error: the button is how the first node gets created.
pub(crate) fn read_registry() -> Vec<NodeEntry> {
    let Some(path) = registry_path() else {
        return Vec::new();
    };
    match fs::read_to_string(&path) {
        Ok(text) => parse_registry(&text),
        Err(_) => Vec::new(),
    }
}

/// Parse exactly the `[[node]]` array-of-tables shape of the shared
/// registry. Lenient by construction: any line that is not a recognised
/// `key = scalar` inside a `[[node]]` table is ignored, and an entry missing
/// the identity fields (`name`, `host`, `user`) is dropped rather than
/// surfaced half-formed.
fn parse_registry(text: &str) -> Vec<NodeEntry> {
    let mut entries: Vec<NodeEntry> = Vec::new();
    let mut current: Option<NodeEntry> = None;
    let mut in_node = false;

    let lines: Vec<&str> = text.lines().collect();
    let mut index = 0;
    while index < lines.len() {
        let line = lines[index].trim();
        index += 1;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with("[[") {
            if let Some(entry) = current.take() {
                push_entry(&mut entries, entry);
            }
            in_node = table_name(line) == "node";
            if in_node {
                current = Some(NodeEntry::default());
            }
            continue;
        }
        if line.starts_with('[') {
            // Any other table ends the current entry; keys under it belong
            // to a section this reader does not model.
            if let Some(entry) = current.take() {
                push_entry(&mut entries, entry);
            }
            in_node = false;
            continue;
        }
        if !in_node {
            continue;
        }
        let Some((key, rest)) = line.split_once('=') else {
            continue;
        };
        let rest = rest.trim();
        // A multi-line string is not part of the registry shape, but a
        // remote error message could still drive a writer to emit one. Skip
        // the whole value: its body must never be read as further keys.
        if let Some(delimiter) = rest
            .strip_prefix("\"\"\"")
            .map(|_| "\"\"\"")
            .or_else(|| rest.strip_prefix("'''").map(|_| "'''"))
        {
            if !rest[3..].contains(delimiter) {
                while index < lines.len() && !lines[index].contains(delimiter) {
                    index += 1;
                }
                index = (index + 1).min(lines.len());
            }
            continue;
        }
        let Some(value) = parse_scalar(rest) else {
            continue;
        };
        if let Some(entry) = current.as_mut() {
            assign(entry, key.trim(), &value);
        }
    }
    if let Some(entry) = current.take() {
        push_entry(&mut entries, entry);
    }
    entries
}

/// Strip the brackets off a `[[table]]` / `[table]` header.
fn table_name(line: &str) -> &str {
    line.trim_matches(|c| c == '[' || c == ']').trim()
}

fn push_entry(entries: &mut Vec<NodeEntry>, entry: NodeEntry) {
    if entry.name.is_empty() || entry.host.is_empty() || entry.user.is_empty() {
        return;
    }
    entries.push(entry);
}

fn assign(entry: &mut NodeEntry, key: &str, value: &Scalar) {
    // Bare keys may be quoted in TOML; the registry writes them bare, but
    // accepting both costs one trim.
    let key = key.trim_matches('"');
    match key {
        "name" => entry.name = value.as_string().unwrap_or_default(),
        "host" => entry.host = value.as_string().unwrap_or_default(),
        "user" => entry.user = value.as_string().unwrap_or_default(),
        "role" => entry.role = value.as_string(),
        "state" => entry.state = value.as_string(),
        "key_path" => entry.key_path = value.as_string(),
        "lane_dir" => entry.lane_dir = value.as_string(),
        "container_image" => entry.container_image = value.as_string(),
        "container_receipt" => entry.container_receipt = value.as_string(),
        "runtime_sha256" => entry.runtime_sha256 = value.as_string(),
        "consumer_model_path" => entry.consumer_model_path = value.as_string(),
        "pki_dir" => entry.pki_dir = value.as_string(),
        "hmac_key_id" => entry.hmac_key_id = value.as_string(),
        "hmac_epoch" => entry.hmac_epoch = value.as_int(),
        "enrollment_version" => entry.enrollment_version = value.as_int(),
        "netqual_gbps" => entry.netqual_gbps = value.as_float(),
        "connect_host" => entry.connect_host = value.as_string(),
        "netqual_rtt_ms" => entry.netqual_rtt_ms = value.as_float(),
        "last_error" => entry.last_error = value.as_string(),
        "updated" => entry.updated = value.as_string(),
        _ => {}
    }
}

/// One TOML scalar: basic string (with the escapes the registry can emit),
/// literal string, integer, float, or boolean. Anything else — arrays,
/// inline tables, dates — returns `None` and the line is skipped.
fn parse_scalar(text: &str) -> Option<Scalar> {
    let mut chars = text.chars().peekable();
    match chars.peek() {
        Some('"') => parse_basic_string(text).map(Scalar::Str),
        Some('\'') => parse_literal_string(text).map(Scalar::Str),
        _ => {
            // Unquoted: the value ends at whitespace or a trailing comment.
            let token = text
                .split('#')
                .next()
                .unwrap_or("")
                .split_whitespace()
                .next()
                .unwrap_or("");
            if token.is_empty() {
                return None;
            }
            match token {
                "true" => return Some(Scalar::Bool(true)),
                "false" => return Some(Scalar::Bool(false)),
                _ => {}
            }
            let clean = token.replace('_', "");
            if let Ok(value) = clean.parse::<i64>() {
                return Some(Scalar::Int(value));
            }
            clean.parse::<f64>().ok().map(Scalar::Float)
        }
    }
}

fn parse_basic_string(text: &str) -> Option<String> {
    let mut chars = text.chars();
    if chars.next() != Some('"') {
        return None;
    }
    let mut out = String::new();
    while let Some(c) = chars.next() {
        match c {
            '"' => return Some(out),
            '\\' => match chars.next()? {
                'n' => out.push('\n'),
                't' => out.push('\t'),
                'r' => out.push('\r'),
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                'u' => {
                    let hex: String = (0..4).filter_map(|_| chars.next()).collect();
                    let code = u32::from_str_radix(&hex, 16).ok()?;
                    out.push(char::from_u32(code)?);
                }
                other => out.push(other),
            },
            _ => out.push(c),
        }
    }
    None // unterminated
}

fn parse_literal_string(text: &str) -> Option<String> {
    let rest = text.strip_prefix('\'')?;
    let end = rest.find('\'')?;
    Some(rest[..end].to_string())
}

// ================================================================= job state

/// Exit status of a finished onboarding job.
#[derive(Debug, Clone)]
pub(crate) struct JobExit {
    pub(crate) code: Option<i32>,
    pub(crate) ok: bool,
    pub(crate) detail: String,
}

#[derive(Default)]
struct Progress {
    /// Retained progress lines, oldest first (bounded by [`PROGRESS_RING`]).
    lines: VecDeque<String>,
    /// Absolute index of `lines.front()`. Subscribers hold absolute cursors,
    /// so eviction never silently rewinds or duplicates a stream.
    first: u64,
    exit: Option<JobExit>,
}

/// One onboarding run: its bounded progress ring and, once reaped, its exit.
pub(crate) struct NodeJob {
    pub(crate) name: String,
    progress: Mutex<Progress>,
    changed: Condvar,
}

/// What a subscriber gets for one poll: new lines, the cursor to resume
/// from, and the exit status if the job has finished.
pub(crate) struct ProgressSlice {
    pub(crate) lines: Vec<String>,
    pub(crate) cursor: u64,
    pub(crate) exit: Option<JobExit>,
}

impl NodeJob {
    fn new(name: String) -> Self {
        NodeJob {
            name,
            progress: Mutex::new(Progress::default()),
            changed: Condvar::new(),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Progress> {
        // A panicked writer must not take the progress stream down with it:
        // the retained lines stay readable and the job still reaches a
        // terminal event.
        self.progress.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn push(&self, line: String) {
        let mut progress = self.lock();
        if progress.lines.len() >= PROGRESS_RING {
            progress.lines.pop_front();
            progress.first += 1;
        }
        progress.lines.push_back(line);
        drop(progress);
        self.changed.notify_all();
    }

    fn finish(&self, exit: JobExit) {
        let mut progress = self.lock();
        progress.exit = Some(exit);
        drop(progress);
        self.changed.notify_all();
    }

    fn slice_locked(progress: &Progress, cursor: u64) -> ProgressSlice {
        let end = progress.first + progress.lines.len() as u64;
        // A cursor older than the retained window resumes at the oldest
        // retained line; a cursor ahead of the stream (impossible for our
        // own replies, but cheap to survive) yields nothing.
        let start = cursor.max(progress.first).min(end);
        let offset = (start - progress.first) as usize;
        ProgressSlice {
            lines: progress.lines.iter().skip(offset).cloned().collect(),
            cursor: end,
            exit: progress.exit.clone(),
        }
    }

    /// Everything at or after `cursor`, without blocking.
    pub(crate) fn since(&self, cursor: u64) -> ProgressSlice {
        Self::slice_locked(&self.lock(), cursor)
    }

    /// Everything at or after `cursor`, waiting up to `timeout` for the
    /// first new line. Returns an empty slice on timeout so the caller can
    /// emit its heartbeat.
    pub(crate) fn wait(&self, cursor: u64, timeout: Duration) -> ProgressSlice {
        let progress = self.lock();
        let end = progress.first + progress.lines.len() as u64;
        if cursor < end || progress.exit.is_some() {
            return Self::slice_locked(&progress, cursor);
        }
        let (progress, _) = self
            .changed
            .wait_timeout(progress, timeout)
            .unwrap_or_else(PoisonError::into_inner);
        Self::slice_locked(&progress, cursor)
    }
}

/// The process-wide onboarding job table: at most one running job, plus a
/// bounded tail of finished ones so a late subscriber still finds its
/// stream.
pub(crate) struct NodeJobs {
    inner: Mutex<JobTable>,
}

#[derive(Default)]
struct JobTable {
    running: Option<Arc<NodeJob>>,
    recent: VecDeque<Arc<NodeJob>>,
}

impl Default for NodeJobs {
    fn default() -> Self {
        Self::new()
    }
}

impl NodeJobs {
    pub(crate) fn new() -> Self {
        NodeJobs {
            inner: Mutex::new(JobTable::default()),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, JobTable> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    pub(crate) fn running_name(&self) -> Option<String> {
        self.lock().running.as_ref().map(|job| job.name.clone())
    }

    /// Claim the single global job slot. Returns the running job's name when
    /// the slot is taken — jobs contend for ssh and the remote docker
    /// daemon, so a second concurrent run is refused, not queued.
    fn reserve(&self, name: &str) -> Result<Arc<NodeJob>, String> {
        let mut table = self.lock();
        if let Some(running) = table.running.as_ref() {
            return Err(running.name.clone());
        }
        let job = Arc::new(NodeJob::new(name.to_string()));
        table.running = Some(Arc::clone(&job));
        Ok(job)
    }

    fn retire(&self, job: &Arc<NodeJob>) {
        let mut table = self.lock();
        if table
            .running
            .as_ref()
            .is_some_and(|running| Arc::ptr_eq(running, job))
        {
            table.running = None;
        }
        table.recent.push_back(Arc::clone(job));
        while table.recent.len() > RECENT_JOBS {
            table.recent.pop_front();
        }
    }

    /// The newest job for `name`, running or recently finished.
    pub(crate) fn find(&self, name: &str) -> Option<Arc<NodeJob>> {
        let table = self.lock();
        if let Some(running) = table.running.as_ref().filter(|job| job.name == name) {
            return Some(Arc::clone(running));
        }
        table
            .recent
            .iter()
            .rev()
            .find(|job| job.name == name)
            .map(Arc::clone)
    }
}

// ============================================================ POST /v1/nodes

/// A validated `POST /v1/nodes` body.
struct AddRequest {
    name: String,
    host: String,
    user: String,
    key_path: Option<String>,
    dry_run: bool,
}

/// An HTTP reply as the dispatch in `httpd` writes it.
pub(crate) struct Reply {
    pub(crate) code: u16,
    pub(crate) reason: &'static str,
    pub(crate) body: String,
}

fn error_reply(code: u16, reason: &'static str, kind: &str, message: &str) -> Reply {
    Reply {
        code,
        reason,
        body: serde_json::json!({"error": {"type": kind, "message": message}}).to_string(),
    }
}

/// `POST /v1/nodes` — validate, claim the global job slot, spawn
/// `muser node add` and return 202 immediately. The run itself is watched by
/// a background thread; the browser follows it on the progress stream.
pub(crate) fn create(
    state: &Arc<ServerState>,
    body: &[u8],
    activator: Option<NodeActivator>,
) -> Reply {
    let request = match parse_add_request(body) {
        Ok(request) => request,
        Err(message) => return error_reply(400, "Bad Request", "invalid_request_error", &message),
    };
    if !available() {
        return error_reply(
            503,
            "Service Unavailable",
            "node_management_unavailable",
            "node management needs a resolvable home directory ($HOME or $MUSER_HOME) for \
             ~/.muser/nodes.toml and the per-node key material",
        );
    }
    let exe = match std::env::current_exe() {
        Ok(exe) => exe,
        Err(error) => {
            return error_reply(
                503,
                "Service Unavailable",
                "node_management_unavailable",
                &format!("could not locate the muser binary to run the pipeline: {error}"),
            )
        }
    };

    let job = match state.node_jobs.reserve(&request.name) {
        Ok(job) => job,
        Err(running) => {
            // Same-name and different-name collisions get the same status
            // and both name the job holding the slot, so the dashboard can
            // point at the stream that is actually live.
            let message = if running == request.name {
                format!("an onboarding job for {running} is already running")
            } else {
                format!(
                    "onboarding runs one node at a time (ssh + remote docker are shared); \
                     {running} is running"
                )
            };
            return Reply {
                code: 409,
                reason: "Conflict",
                body: serde_json::json!({
                    "error": {"type": "node_job_running", "message": message},
                    "running": running,
                })
                .to_string(),
            };
        }
    };

    let activates_inference = activator.is_some();
    let state = Arc::clone(state);
    let job_thread = Arc::clone(&job);
    std::thread::spawn(move || {
        run_job(&job_thread, exe, request, activator);
        // The slot is released only after the child is reaped, so the 409
        // and the running-job flag stay true for the whole run.
        state.node_jobs.retire(&job_thread);
    });

    Reply {
        code: 202,
        reason: "Accepted",
        body: serde_json::json!({
            "name": job.name,
            "progress": format!("/v1/nodes/{}/progress", job.name),
            "activates_inference": activates_inference,
        })
        .to_string(),
    }
}

fn parse_add_request(body: &[u8]) -> Result<AddRequest, String> {
    let value: serde_json::Value =
        serde_json::from_slice(body).map_err(|error| format!("invalid JSON body: {error}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| "body must be a JSON object".to_string())?;

    let host = string_field(object, "host")?.ok_or_else(|| "\"host\" is required".to_string())?;
    let user = string_field(object, "user")?.ok_or_else(|| "\"user\" is required".to_string())?;
    let name = match string_field(object, "name")? {
        Some(name) => name,
        None => default_name(&host),
    };
    let key_path = string_field(object, "key_path")?;
    let dry_run = match object.get("dry_run") {
        None | Some(serde_json::Value::Null) => false,
        Some(serde_json::Value::Bool(value)) => *value,
        Some(_) => return Err("\"dry_run\" must be a boolean".into()),
    };

    // Every field below is spliced into a child process argv (and from there
    // into ssh arguments), so the charsets are allow-lists, and nothing may
    // start with `-` and be read as a flag by the child's own parser.
    if !is_valid_host(&host) {
        return Err(
            "\"host\" must be a hostname or address: letters, digits, '.', '-', '_', ':'".into(),
        );
    }
    if !is_valid_user(&user) {
        return Err("\"user\" must be a login name: letters, digits, '.', '-', '_'".into());
    }
    if !is_valid_name(&name) {
        return Err(
            "\"name\" must be 1-64 characters of letters, digits, '-' or '_' (it is a registry \
             key and a directory name)"
                .into(),
        );
    }
    if let Some(key) = key_path.as_deref() {
        if key.is_empty()
            || key.len() > 4096
            || key.starts_with('-')
            || key.chars().any(char::is_control)
        {
            return Err("\"key_path\" must be a plain filesystem path".into());
        }
    }

    Ok(AddRequest {
        name,
        host,
        user,
        key_path,
        dry_run,
    })
}

fn string_field(
    object: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Result<Option<String>, String> {
    match object.get(key) {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(value)) => {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                Ok(None)
            } else {
                Ok(Some(trimmed.to_string()))
            }
        }
        Some(_) => Err(format!("\"{key}\" must be a string")),
    }
}

/// Default a node's name from its host's first label, so the common case
/// (`POST {"host":"producer-1","user":"producer-1"}`) needs no name at all.
fn default_name(host: &str) -> String {
    let label: String = host
        .split('.')
        .next()
        .unwrap_or("")
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        // Bounded here so a long hostname yields a usable default rather
        // than a 400 about a name the caller never wrote.
        .take(64)
        .collect();
    let label = label.trim_matches('-').to_string();
    if label.is_empty() {
        "node".to_string()
    } else {
        label
    }
}

fn is_valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && !name.starts_with('-')
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

fn is_valid_host(host: &str) -> bool {
    !host.is_empty()
        && host.len() <= 255
        && !host.starts_with('-')
        && host
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_' | ':'))
}

fn is_valid_user(user: &str) -> bool {
    !user.is_empty()
        && user.len() <= 64
        && !user.starts_with('-')
        && user
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_'))
}

// ============================================================== job execution

/// One synthetic progress line, in the shared protocol, for the things the
/// server itself has to say (a child that would not start, a missing
/// transcript). The CLI owns every other line on the stream.
fn server_line(step: &str, status: &str, detail: &str) -> String {
    serde_json::json!({
        "schema": PROGRESS_SCHEMA,
        "step": step,
        "status": status,
        "detail": detail,
    })
    .to_string()
}

struct ChildOutcome {
    code: Option<i32>,
    stderr_tail: Vec<String>,
    killed: bool,
}

fn run_job(
    job: &Arc<NodeJob>,
    exe: PathBuf,
    request: AddRequest,
    activator: Option<NodeActivator>,
) {
    let transcript = open_progress_log(&job.name);
    if transcript.is_none() {
        job.push(server_line(
            "preflight",
            "info",
            "progress log unavailable — streaming without an on-disk transcript",
        ));
    }
    let log: SharedLog = Arc::new(Mutex::new(transcript));

    let target = format!("{}@{}", request.user, request.host);
    // The child is this same binary (`current_exe`), so `--json` — the flag
    // that switches the CLI's progress protocol to one JSON object per line
    // — is always available to it.
    let mut args: Vec<OsString> = vec!["node".into(), "add".into(), target.into()];
    args.push("--name".into());
    args.push(request.name.clone().into());
    if let Some(key) = request.key_path.as_deref() {
        args.push("--key".into());
        args.push(key.into());
    }
    if request.dry_run {
        args.push("--dry-run".into());
    }
    args.push("--json".into());

    let outcome = run_child(job, &exe, &args, &log);
    let mut exit = summarize(&outcome);
    activate_after_onboarding(job, &request, activator, &mut exit);
    if !exit.ok {
        eprintln!("muser-server: node {}: {}", job.name, exit.detail);
    }
    job.finish(exit);
}

fn activate_after_onboarding(
    job: &Arc<NodeJob>,
    request: &AddRequest,
    activator: Option<NodeActivator>,
    exit: &mut JobExit,
) {
    if !exit.ok || request.dry_run {
        return;
    }
    let Some(activate) = activator else {
        return;
    };
    job.push(server_line(
        "activate",
        "start",
        "starting the Mac decoder on this dashboard",
    ));
    let reporter_job = Arc::clone(job);
    let reporter: ActivationReporter = Arc::new(move |status, detail| {
        reporter_job.push(server_line("activate", status, detail));
    });
    match activate(&request.name, reporter) {
        Ok(()) => {
            job.push(server_line(
                "activate",
                "ok",
                "Mac decoder and remote prefill are ready on this dashboard",
            ));
            exit.detail = "node onboarding and Mac decoder activation finished".to_string();
        }
        Err(error) => {
            job.push(server_line("activate", "fail", &error));
            *exit = JobExit {
                code: Some(1),
                ok: false,
                detail: format!(
                    "node enrolled and smoke-tested, but Mac decoder activation failed: {error}"
                ),
            };
        }
    }
}

fn summarize(outcome: &ChildOutcome) -> JobExit {
    if outcome.killed {
        return JobExit {
            code: outcome.code,
            ok: false,
            detail: format!(
                "onboarding made no progress for {} minutes and was stopped",
                JOB_DEADLINE.as_secs() / 60
            ),
        };
    }
    match outcome.code {
        Some(0) => JobExit {
            code: Some(0),
            ok: true,
            detail: "node onboarding finished; the smoke handoff passed".to_string(),
        },
        Some(code) => {
            let tail = outcome
                .stderr_tail
                .last()
                .map(|line| format!(": {line}"))
                .unwrap_or_default();
            JobExit {
                code: Some(code),
                ok: false,
                detail: format!("`muser node add` exited {code}{tail}"),
            }
        }
        None => JobExit {
            code: None,
            ok: false,
            detail: "`muser node add` was terminated by a signal".to_string(),
        },
    }
}

type SharedLog = Arc<Mutex<Option<fs::File>>>;

fn run_child(job: &Arc<NodeJob>, exe: &Path, args: &[OsString], log: &SharedLog) -> ChildOutcome {
    let mut child = match Command::new(exe)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => {
            job.push(server_line(
                "preflight",
                "fail",
                &format!("could not start `muser node add`: {error}"),
            ));
            return ChildOutcome {
                code: None,
                stderr_tail: vec![error.to_string()],
                killed: false,
            };
        }
    };

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    let last_activity = Arc::new(Mutex::new(Instant::now()));
    let stdout_job = Arc::clone(job);
    let stdout_log = Arc::clone(log);
    let stdout_activity = Arc::clone(&last_activity);
    let stdout_pump = stdout.map(|stdout| {
        std::thread::spawn(move || pump_stdout(stdout, &stdout_job, &stdout_log, &stdout_activity))
    });

    let stderr_tail = Arc::new(Mutex::new(VecDeque::<String>::new()));
    let stderr_sink = Arc::clone(&stderr_tail);
    let stderr_pump =
        stderr.map(|stderr| std::thread::spawn(move || pump_stderr(stderr, &stderr_sink)));

    let (code, killed) = reap(&mut child, &last_activity);
    // Join the pumps before reporting: the child's last progress line must
    // be on the stream before the terminal event that follows it.
    if let Some(handle) = stdout_pump {
        let _ = handle.join();
    }
    if let Some(handle) = stderr_pump {
        let _ = handle.join();
    }

    let stderr_tail = stderr_tail
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .iter()
        .cloned()
        .collect();
    ChildOutcome {
        code,
        stderr_tail,
        killed,
    }
}

/// Wait for the child, enforcing [`JOB_DEADLINE`]. Polling rather than a
/// blocking wait keeps the deadline on the same thread that owns the handle,
/// so the kill needs no second owner of the `Child`.
fn reap(child: &mut Child, last_activity: &Arc<Mutex<Instant>>) -> (Option<i32>, bool) {
    let mut killed = false;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return (status.code(), killed),
            Ok(None) => {}
            Err(_) => return (None, killed),
        }
        let idle_since = *last_activity.lock().expect("activity clock poisoned");
        if !killed && idle_since.elapsed() >= JOB_DEADLINE {
            let _ = child.kill();
            killed = true;
        }
        std::thread::sleep(REAP_POLL);
    }
}

/// Relay the child's stdout: progress objects go to the ring (and so to
/// every SSE subscriber) and to the on-disk transcript; anything else is
/// transcript-only, so a stray `println!` cannot inject a non-JSON `data:`
/// frame into a browser's `EventSource`.
fn pump_stdout(
    stdout: impl Read,
    job: &Arc<NodeJob>,
    log: &SharedLog,
    last_activity: &Arc<Mutex<Instant>>,
) {
    for line in BufReader::new(stdout).lines() {
        let Ok(line) = line else { break };
        let line = line.trim_end().to_string();
        if line.is_empty() {
            continue;
        }
        *last_activity.lock().expect("activity clock poisoned") = Instant::now();
        if line.starts_with('{') {
            append_log(log, &line);
            job.push(line);
        } else {
            append_log(log, &format!("# stdout: {line}"));
        }
    }
}

fn pump_stderr(stderr: impl Read, tail: &Mutex<VecDeque<String>>) {
    for line in BufReader::new(stderr).lines() {
        let Ok(line) = line else { break };
        let line = line.trim_end().to_string();
        if line.is_empty() {
            continue;
        }
        eprintln!("muser-server: node add: {line}");
        let mut tail = tail.lock().unwrap_or_else(PoisonError::into_inner);
        if tail.len() >= STDERR_TAIL_LINES {
            tail.pop_front();
        }
        tail.push_back(line);
    }
}

/// `~/.muser/nodes/<name>/progress.log`, appended to. The directory is the
/// same one the pipeline keeps key material in, so it is created 0700 and
/// the transcript 0600.
fn open_progress_log(name: &str) -> Option<fs::File> {
    let dir = node_dir(name)?;
    fs::create_dir_all(&dir).ok()?;
    let _ = fs::set_permissions(&dir, fs::Permissions::from_mode(0o700));
    let path = progress_log_path(name)?;
    OpenOptions::new()
        .create(true)
        .append(true)
        .mode(0o600)
        .open(path)
        .ok()
}

fn append_log(log: &SharedLog, line: &str) {
    let mut transcript = log.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(file) = transcript.as_mut() {
        let _ = writeln!(file, "{line}");
    }
}

/// The tail of a finished run's transcript, for a browser that subscribes
/// after the process that ran it is gone (a server restart, most often).
fn replay_log(name: &str) -> Option<Vec<String>> {
    let path = progress_log_path(name)?;
    let text = fs::read_to_string(path).ok()?;
    let lines: Vec<String> = text
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with('{'))
        .map(str::to_string)
        .collect();
    if lines.is_empty() {
        return None;
    }
    let start = lines.len().saturating_sub(PROGRESS_RING);
    Some(lines[start..].to_vec())
}

// ============================================================= GET /v1/nodes

/// `GET /v1/nodes` — the registry, plus what is true right now: a real TCP
/// connect to each node's daemon port and the running-job flag.
pub(crate) fn list(state: &ServerState) -> Reply {
    let entries = read_registry();
    let running = state.node_jobs.running_name();
    let probes = probe_all(&entries);

    let nodes: Vec<serde_json::Value> = entries
        .iter()
        .zip(probes)
        .map(|(entry, reachable)| {
            serde_json::json!({
                "name": entry.name,
                "host": entry.host,
                "user": entry.user,
                "role": entry.role,
                "state": if entry.enrollment_version.unwrap_or(0) < 2
                    && matches!(entry.state.as_deref(), Some("healthy" | "enrolled"))
                { Some("needs-reenrollment") } else { entry.state.as_deref() },
                "key_path": entry.key_path,
                "lane_dir": entry.lane_dir,
                "container_image": entry.container_image,
                "container_receipt": entry.container_receipt,
                "runtime_sha256": entry.runtime_sha256,
                "consumer_model_path": entry.consumer_model_path,
                "pki_dir": entry.pki_dir,
                "hmac_key_id": entry.hmac_key_id,
                "hmac_epoch": entry.hmac_epoch,
                "enrollment_version": entry.enrollment_version,
                "netqual_gbps": entry.netqual_gbps,
                "netqual_rtt_ms": entry.netqual_rtt_ms,
                "last_error": entry.last_error,
                "updated": entry.updated,
                // Live, this instant — a TCP connect, not a remembered
                // registry field.
                "daemon_alive": reachable,
                "daemon_port": DAEMON_PORT,
                "job_running": running.as_deref() == Some(entry.name.as_str()),
            })
        })
        .collect();

    Reply {
        code: 200,
        reason: "OK",
        body: serde_json::json!({
            "nodes": nodes,
            "running_job": running,
            "registry": registry_path().map(|path| path.display().to_string()),
            "daemon_probe_timeout_ms": DAEMON_PROBE_TIMEOUT.as_millis() as u64,
        })
        .to_string(),
    }
}

/// Probe every node in parallel: a serial walk would pay the connect
/// timeout once per unreachable node on a route the dashboard polls.
fn probe_all(entries: &[NodeEntry]) -> Vec<bool> {
    if entries.is_empty() {
        return Vec::new();
    }
    std::thread::scope(|scope| {
        let handles: Vec<_> = entries
            .iter()
            .map(|entry| {
                scope.spawn(move || {
                    probe_daemon(entry.connect_host.as_deref().unwrap_or(&entry.host))
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap_or(false))
            .collect()
    })
}

fn probe_daemon(host: &str) -> bool {
    let Ok(addrs) = (host, DAEMON_PORT).to_socket_addrs() else {
        return false;
    };
    // Resolution can hand back both families; two attempts bound the wall
    // clock at twice the connect timeout.
    addrs
        .take(2)
        .any(|addr| TcpStream::connect_timeout(&addr, DAEMON_PROBE_TIMEOUT).is_ok())
}

// ================================================ GET /v1/nodes/<name>/progress

/// Where a progress stream's lines come from: a live (or recently finished)
/// job in this process, or the on-disk transcript of a run this process did
/// not host.
pub(crate) enum ProgressSource {
    Job(Arc<NodeJob>),
    Transcript(Vec<String>),
}

/// `None` when neither a job nor a transcript exists for `name` — the route
/// answers 404 rather than opening a stream that can never carry anything.
pub(crate) fn progress_source(state: &ServerState, name: &str) -> Option<ProgressSource> {
    if !is_valid_name(name) {
        return None;
    }
    if let Some(job) = state.node_jobs.find(name) {
        return Some(ProgressSource::Job(job));
    }
    replay_log(name).map(ProgressSource::Transcript)
}

/// Stream one job's progress as SSE: ring-buffer replay first (so a browser
/// that subscribes late still sees the whole run), then a live tail until
/// the child exits, then one terminal `end` event carrying the exit status.
/// Heartbeat comments every [`HEARTBEAT`] keep an idle step — a multi-minute
/// container pull — from looking like a dead connection.
pub(crate) fn stream_progress(
    mut stream: TcpStream,
    source: ProgressSource,
    name: &str,
) -> io::Result<()> {
    stream.write_all(crate::httpd::SSE_HEADERS.as_bytes())?;
    stream.set_write_timeout(Some(Duration::from_secs(5)))?;

    produce_progress_frames(source, name, |payload| {
        stream.write_all(payload.as_bytes())?;
        stream.flush()
    })
}

/// Transport-neutral progress producer used by both the legacy CLI socket
/// adapter and Axum's bounded async body channel. A failed callback is a
/// disconnected/slow client and stops following the child immediately.
pub(crate) fn produce_progress_frames(
    source: ProgressSource,
    name: &str,
    mut emit: impl FnMut(String) -> io::Result<()>,
) -> io::Result<()> {
    let job = match source {
        ProgressSource::Transcript(lines) => {
            // Nothing live to follow: replay the transcript and close with a
            // terminal event that says so, rather than holding the socket.
            let detail = "replayed from the on-disk transcript; no job for this node is running \
                          in this server";
            emit(frame(&lines))?;
            emit(terminal_event(name, None, detail))?;
            return Ok(());
        }
        ProgressSource::Job(job) => job,
    };

    let mut cursor = 0u64;
    let replay = job.since(cursor);
    cursor = replay.cursor;
    if !replay.lines.is_empty() {
        emit(frame(&replay.lines))?;
    }
    if let Some(exit) = replay.exit {
        emit(terminal_event(name, Some(&exit), &exit.detail))?;
        return Ok(());
    }

    loop {
        let slice = job.wait(cursor, HEARTBEAT);
        cursor = slice.cursor;
        if slice.lines.is_empty() {
            if let Some(exit) = slice.exit {
                emit(terminal_event(name, Some(&exit), &exit.detail))?;
                return Ok(());
            }
            emit(": ping\n\n".to_string())?;
            continue;
        }
        emit(frame(&slice.lines))?;
        if let Some(exit) = slice.exit {
            emit(terminal_event(name, Some(&exit), &exit.detail))?;
            return Ok(());
        }
    }
}

/// The child's lines, relayed verbatim, as one buffer: a batch of frames
/// costs one syscall and one segment instead of one per line.
fn frame(lines: &[String]) -> String {
    let mut out = String::new();
    for line in lines {
        out.push_str("data: ");
        out.push_str(line);
        out.push_str("\n\n");
    }
    out
}

fn terminal_event(name: &str, exit: Option<&JobExit>, detail: &str) -> String {
    let status = match exit {
        Some(exit) if exit.ok => "ok",
        Some(_) => "fail",
        None => "idle",
    };
    let body = serde_json::json!({
        "schema": JOB_SCHEMA,
        "name": name,
        "status": status,
        "exit_code": exit.and_then(|exit| exit.code),
        "detail": detail,
    });
    // The stream ends when the run does, and an `EventSource` reconnects on
    // any close — including this deliberate one. Raising the reconnection
    // time on the last event stops a finished run from being replayed to the
    // browser every few seconds; a mid-run drop still reconnects promptly,
    // because only this frame carries the long `retry`.
    format!("retry: {}\nevent: end\ndata: {body}\n\n", TERMINAL_RETRY_MS)
}

#[cfg(test)]
mod tests {
    use super::*;

    const REGISTRY: &str = r#"
# muser node registry
[[node]]
name = "gx10"
host = "producer-1"
user = "producer-1"
role = "prefill"
state = "healthy"
key_path = "/home/x/.ssh/id_ed25519"
lane_dir = "/opt/muser/lane"
container_image = "sha256:abc123"
container_receipt = "/home/x/.muser/nodes/gx10/container.json"
runtime_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
consumer_model_path = "/models/native.gguf"
pki_dir = "/home/x/.muser/nodes/gx10/pki"
hmac_key_id = "k1"
hmac_epoch = 7
netqual_gbps = 18.5
netqual_rtt_ms = 0.42
updated = "2026-08-13T21:00:00Z"

[[node]]
name = "spare"
host = "10.0.0.4"
user = "ops"
role = "prefill"
state = "draft"
"#;

    #[test]
    fn registry_parses_the_full_node_shape() {
        let entries = parse_registry(REGISTRY);
        assert_eq!(entries.len(), 2);
        let first = &entries[0];
        assert_eq!(first.name, "gx10");
        assert_eq!(first.host, "producer-1");
        assert_eq!(first.user, "producer-1");
        assert_eq!(first.state.as_deref(), Some("healthy"));
        assert_eq!(first.container_image.as_deref(), Some("sha256:abc123"));
        assert_eq!(
            first.runtime_sha256.as_deref(),
            Some("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
        );
        assert_eq!(
            first.consumer_model_path.as_deref(),
            Some("/models/native.gguf")
        );
        assert_eq!(first.hmac_epoch, Some(7));
        assert_eq!(first.netqual_gbps, Some(18.5));
        assert_eq!(first.netqual_rtt_ms, Some(0.42));
        assert_eq!(first.updated.as_deref(), Some("2026-08-13T21:00:00Z"));
        // Optional fields a draft entry has not earned yet stay absent
        // instead of arriving as empty strings.
        let second = &entries[1];
        assert_eq!(second.name, "spare");
        assert!(second.key_path.is_none());
        assert!(second.hmac_epoch.is_none());
        assert!(second.netqual_gbps.is_none());
    }

    #[test]
    fn registry_reader_survives_lines_it_does_not_model() {
        // A newer writer adding keys, sections, or comments must not blank
        // the dashboard's node list.
        let text = r#"
version = 2
[[node]]
name = "gx10" # trailing comment
host = 'producer-1'
user = "producer-1"
future_key = ["not", "modelled"]
hmac_epoch = 3

[settings]
name = "not-a-node"
host = "nowhere"
user = "nobody"
"#;
        let entries = parse_registry(text);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "gx10");
        assert_eq!(entries[0].host, "producer-1");
        assert_eq!(entries[0].hmac_epoch, Some(3));
    }

    #[test]
    fn registry_reader_drops_entries_without_an_identity() {
        let text =
            "[[node]]\nstate = \"draft\"\n\n[[node]]\nname = \"ok\"\nhost = \"h\"\nuser = \"u\"\n";
        let entries = parse_registry(text);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "ok");
    }

    #[test]
    fn registry_strings_keep_their_escapes() {
        let text = concat!(
            "[[node]]\n",
            "name = \"gx10\"\n",
            "host = \"h\"\n",
            "user = \"u\"\n",
            "last_error = \"ssh said \\\"permission denied\\\"\\nretry with -i\"\n",
            "lane_dir = \"/opt/a\\\\b\"\n",
        );
        let entries = parse_registry(text);
        assert_eq!(
            entries[0].last_error.as_deref(),
            Some("ssh said \"permission denied\"\nretry with -i")
        );
        assert_eq!(entries[0].lane_dir.as_deref(), Some("/opt/a\\b"));
    }

    #[test]
    fn registry_reader_ignores_an_unterminated_string() {
        let text = "[[node]]\nname = \"gx10\"\nhost = \"h\"\nuser = \"u\"\nlast_error = \"oops\n";
        let entries = parse_registry(text);
        assert_eq!(entries.len(), 1);
        assert!(entries[0].last_error.is_none());
    }

    #[test]
    fn a_multi_line_value_cannot_smuggle_keys_into_an_entry() {
        // `last_error` carries remote text; a writer that spilled it into a
        // multi-line string must not let its body be read as registry keys.
        let text = concat!(
            "[[node]]\n",
            "name = \"gx10\"\n",
            "host = \"h\"\n",
            "user = \"u\"\n",
            "last_error = \"\"\"\n",
            "ssh failed\n",
            "host = \"attacker\"\n",
            "\"\"\"\n",
            "state = \"error\"\n",
        );
        let entries = parse_registry(text);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].host, "h");
        assert_eq!(entries[0].state.as_deref(), Some("error"));
    }

    #[test]
    fn registry_reader_handles_crlf_and_an_empty_file() {
        let text = "[[node]]\r\nname = \"gx10\"\r\nhost = \"h\"\r\nuser = \"u\"\r\n";
        assert_eq!(parse_registry(text).len(), 1);
        assert!(parse_registry("").is_empty());
    }

    fn job_with(lines: usize) -> NodeJob {
        let job = NodeJob::new("gx10".to_string());
        for index in 0..lines {
            job.push(format!("{{\"n\":{index}}}"));
        }
        job
    }

    #[test]
    fn replay_from_zero_returns_every_retained_line() {
        let job = job_with(3);
        let slice = job.since(0);
        assert_eq!(slice.lines.len(), 3);
        assert_eq!(slice.cursor, 3);
        assert!(slice.exit.is_none());
    }

    #[test]
    fn a_cursor_only_yields_lines_it_has_not_seen() {
        let job = job_with(3);
        let first = job.since(0);
        job.push("{\"n\":3}".to_string());
        let next = job.since(first.cursor);
        assert_eq!(next.lines, vec!["{\"n\":3}".to_string()]);
        assert_eq!(next.cursor, 4);
        // Re-reading at the same cursor yields nothing new.
        assert!(job.since(next.cursor).lines.is_empty());
    }

    #[test]
    fn the_ring_keeps_the_newest_lines_and_a_stale_cursor_resumes_at_the_oldest() {
        let overflow = PROGRESS_RING + 25;
        let job = job_with(overflow);
        let slice = job.since(0);
        assert_eq!(slice.lines.len(), PROGRESS_RING);
        assert_eq!(slice.cursor, overflow as u64);
        assert_eq!(
            slice.lines[0],
            format!("{{\"n\":{}}}", overflow - PROGRESS_RING)
        );
        assert_eq!(
            slice.lines[PROGRESS_RING - 1],
            format!("{{\"n\":{}}}", overflow - 1)
        );
        // A subscriber whose cursor fell out of the window resumes at the
        // oldest retained line rather than replaying evicted ones.
        let stale = job.since(1);
        assert_eq!(stale.lines.len(), PROGRESS_RING);
        assert_eq!(stale.cursor, overflow as u64);
    }

    #[test]
    fn a_cursor_past_the_stream_yields_nothing() {
        let job = job_with(2);
        let slice = job.since(99);
        assert!(slice.lines.is_empty());
        assert_eq!(slice.cursor, 2);
    }

    #[test]
    fn the_exit_status_reaches_a_subscriber_with_the_last_lines() {
        let job = job_with(1);
        job.finish(JobExit {
            code: Some(0),
            ok: true,
            detail: "done".into(),
        });
        let slice = job.since(0);
        assert_eq!(slice.lines.len(), 1);
        let exit = slice.exit.expect("exit must be visible");
        assert!(exit.ok);
        assert_eq!(exit.code, Some(0));
        // A finished job never blocks a waiter.
        let waited = job.wait(slice.cursor, Duration::from_secs(30));
        assert!(waited.lines.is_empty());
        assert!(waited.exit.is_some());
    }

    #[test]
    fn wait_returns_early_when_a_line_arrives() {
        let job = Arc::new(job_with(0));
        let writer = Arc::clone(&job);
        let handle = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            writer.push("{\"late\":true}".to_string());
        });
        let slice = job.wait(0, Duration::from_secs(30));
        handle.join().expect("writer thread");
        assert_eq!(slice.lines, vec!["{\"late\":true}".to_string()]);
    }

    #[test]
    fn only_one_job_holds_the_global_slot() {
        let jobs = NodeJobs::new();
        let first = jobs.reserve("gx10").expect("first job takes the slot");
        assert_eq!(jobs.reserve("other").err().as_deref(), Some("gx10"));
        assert_eq!(jobs.reserve("gx10").err().as_deref(), Some("gx10"));
        assert_eq!(jobs.running_name().as_deref(), Some("gx10"));
        jobs.retire(&first);
        assert!(jobs.running_name().is_none());
        // A retired job stays addressable so a late subscriber still finds
        // its replay and terminal event.
        assert!(jobs.find("gx10").is_some());
        assert!(jobs.reserve("gx10").is_ok(), "the slot is free again");
    }

    #[test]
    fn retired_jobs_are_bounded() {
        let jobs = NodeJobs::new();
        for index in 0..(RECENT_JOBS + 3) {
            let name = format!("n{index}");
            let job = jobs.reserve(&name).expect("slot is free");
            jobs.retire(&job);
        }
        assert!(jobs.find("n0").is_none());
        assert!(jobs.find(&format!("n{}", RECENT_JOBS + 2)).is_some());
    }

    #[test]
    fn add_requests_are_validated_before_anything_is_spawned() {
        let ok = parse_add_request(br#"{"host":"producer-1","user":"producer-1"}"#)
            .expect("minimal body");
        assert_eq!(ok.name, "producer-1");
        assert_eq!(ok.user, "producer-1");
        assert!(!ok.dry_run);
        assert!(ok.key_path.is_none());

        let named = parse_add_request(
            br#"{"host":"10.0.0.4","user":"ops","name":"gx10","key_path":"/k/id","dry_run":true}"#,
        )
        .expect("full body");
        assert_eq!(named.name, "gx10");
        assert_eq!(named.key_path.as_deref(), Some("/k/id"));
        assert!(named.dry_run);

        for body in [
            &br#"{}"#[..],
            &br#"{"user":"muser"}"#[..],
            &br#"{"host":"h"}"#[..],
            &br#"not json"#[..],
            &br#"[{"host":"h","user":"u"}]"#[..],
            // Argv and path injection: every one of these would otherwise
            // reach ssh or a directory name.
            &br#"{"host":"h; rm -rf /","user":"u"}"#[..],
            &br#"{"host":"-oProxyCommand=x","user":"u"}"#[..],
            &br#"{"host":"h","user":"u","name":"../../etc"}"#[..],
            &br#"{"host":"h","user":"u","name":"has space"}"#[..],
            &br#"{"host":"h","user":"u","key_path":"-oIdentityFile=x"}"#[..],
            &br#"{"host":"h","user":"u","dry_run":"yes"}"#[..],
            &br#"{"host":42,"user":"u"}"#[..],
        ] {
            assert!(
                parse_add_request(body).is_err(),
                "body must be refused: {}",
                String::from_utf8_lossy(body)
            );
        }
    }

    #[test]
    fn a_hostless_name_default_is_still_a_legal_registry_key() {
        assert_eq!(default_name("PRODUCER-1.local"), "producer-1");
        assert_eq!(default_name("10.0.0.4"), "10");
        assert_eq!(default_name("...."), "node");
        assert!(is_valid_name(&default_name("weird!!host")));
    }

    #[test]
    fn progress_frames_are_one_buffer_and_terminal_events_name_their_status() {
        let frames = frame(&["{\"a\":1}".to_string(), "{\"b\":2}".to_string()]);
        assert_eq!(frames, "data: {\"a\":1}\n\ndata: {\"b\":2}\n\n");

        let ok = terminal_event(
            "gx10",
            Some(&JobExit {
                code: Some(0),
                ok: true,
                detail: "done".into(),
            }),
            "done",
        );
        // The long `retry` rides on the terminal frame so a finished run is
        // not replayed to a reconnecting EventSource every few seconds.
        assert!(ok.starts_with(&format!("retry: {TERMINAL_RETRY_MS}\nevent: end\ndata: {{")));
        assert!(ok.ends_with("\n\n"));
        assert!(ok.contains("\"status\":\"ok\""));
        assert!(ok.contains("\"exit_code\":0"));

        let failed = terminal_event(
            "gx10",
            Some(&JobExit {
                code: Some(2),
                ok: false,
                detail: "smoke failed".into(),
            }),
            "smoke failed",
        );
        assert!(failed.contains("\"status\":\"fail\""));
        assert!(failed.contains("\"exit_code\":2"));

        let idle = terminal_event("gx10", None, "transcript");
        assert!(idle.contains("\"status\":\"idle\""));
        assert!(idle.contains("\"exit_code\":null"));
    }

    #[test]
    fn server_lines_carry_the_shared_progress_schema() {
        let line = server_line("preflight", "fail", "could not start");
        let value: serde_json::Value = serde_json::from_str(&line).expect("valid JSON");
        assert_eq!(value["schema"], PROGRESS_SCHEMA);
        assert_eq!(value["step"], "preflight");
        assert_eq!(value["status"], "fail");
        assert_eq!(value["detail"], "could not start");
    }

    #[test]
    fn successful_onboarding_activates_the_same_dashboard_job() {
        let job = Arc::new(NodeJob::new("gx10".to_string()));
        let request = AddRequest {
            name: "gx10".to_string(),
            host: "gx10.example".to_string(),
            user: "muser".to_string(),
            key_path: None,
            dry_run: false,
        };
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let calls_for_activation = Arc::clone(&calls);
        let activator: NodeActivator = Arc::new(move |name, reporter| {
            assert_eq!(name, "gx10");
            calls_for_activation.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            reporter("info", "loading verified Metal slots");
            Ok(())
        });
        let mut exit = JobExit {
            code: Some(0),
            ok: true,
            detail: "smoke passed".to_string(),
        };

        activate_after_onboarding(&job, &request, Some(activator), &mut exit);

        assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert!(exit.ok);
        assert_eq!(
            exit.detail,
            "node onboarding and Mac decoder activation finished"
        );
        let lines = job.since(0).lines.join("\n");
        assert!(lines.contains(r#""step":"activate","status":"start""#));
        assert!(lines.contains("loading verified Metal slots"));
        assert!(lines.contains(r#""step":"activate","status":"ok""#));
    }

    #[test]
    fn activation_failure_is_terminal_and_dry_run_never_activates() {
        let job = Arc::new(NodeJob::new("gx10".to_string()));
        let mut request = AddRequest {
            name: "gx10".to_string(),
            host: "gx10.example".to_string(),
            user: "muser".to_string(),
            key_path: None,
            dry_run: true,
        };
        let activator: NodeActivator = Arc::new(|_, _| panic!("dry run activated inference"));
        let mut exit = JobExit {
            code: Some(0),
            ok: true,
            detail: "planned".to_string(),
        };
        activate_after_onboarding(&job, &request, Some(activator), &mut exit);
        assert!(job.since(0).lines.is_empty());
        assert!(exit.ok);

        request.dry_run = false;
        let activator: NodeActivator =
            Arc::new(|_, _| Err("receiver configuration disappeared".to_string()));
        activate_after_onboarding(&job, &request, Some(activator), &mut exit);
        assert!(!exit.ok);
        assert_eq!(exit.code, Some(1));
        assert!(exit.detail.contains("activation failed"));
        assert!(job
            .since(0)
            .lines
            .join("\n")
            .contains(r#""step":"activate","status":"fail""#));
    }
}
