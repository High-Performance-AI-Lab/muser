//! `muser node` — the onboarding orchestrator.
//!
//! This is the engine the dashboard's "Add node" button drives. One command
//! takes a bare `user@host` to a disaggregated prefill node that has proven
//! itself with a real, exactly-verified remote prefill:
//!
//! | step | what it establishes |
//! |------|---------------------|
//! | `preflight` | aarch64, an NVIDIA driver, a reachable docker daemon, disk, memory |
//! | `deploy`    | the exact pinned producer image and lane runtime |
//! | `model`     | lane weights acquired and SHA-256 verified on the node |
//! | `enroll`    | lab CA, both TLS leaves, a fresh HMAC key, and both handoff configs |
//! | `daemon`    | the resident producer installed, started and listening |
//! | `netqual` + `smoke` | a bounded authenticated handoff, Metal decode, and what the link did |
//!
//! Three properties hold across all of them:
//!
//! - **Progress is a protocol.** Every step emits `muser.node-progress.v2`
//!   lines (`progress.rs`), which the server relays verbatim as SSE.
//! - **`--dry-run` touches nothing.** No SSH, no local writes, no child
//!   processes — each step prints the exact argv it would have run.
//! - **State is one file.** `~/.muser/nodes.toml`, written temp+rename
//!   (`registry.rs`). Key material lives in `pki_dir` at 0600, never here.
//!
//! `muser node add` exits 0 only when the operational smoke passed. Full
//! three-repetition evidence is an explicit `muser node qualify` command.

pub mod artifacts;
pub mod daemon;
pub mod deploy;
pub mod enroll;
pub mod model;
pub mod pki;
pub mod preflight;
pub mod progress;
pub mod registry;
pub mod smoke;
pub mod ssh;
pub mod status;

use std::net::ToSocketAddrs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use crate::cli::{NodeAddArgs, NodeArgs, NodeCommand, NodeCommonArgs, NodeStepArgs};

use self::artifacts::{ContainerReceipt, NativeIdentity, Release};
use self::progress::{Progress, Status, Step};
use self::registry::{
    NodeEntry, OperationLock, ProducerKind, Registry, DAEMON_PORT, STATE_HEALTHY,
};
use self::ssh::Ssh;

pub type Result<T> = std::result::Result<T, String>;
type StepRunner = fn(&Ctx, &mut NodeEntry) -> Result<()>;

const REJOIN_PROBE_TIMEOUT: Duration = Duration::from_secs(1);
const REJOIN_ERROR_PREFIX: &str = "live rejoin probe failed without changing the node: ";

/// Everything a step needs that is not the node itself.
pub struct Ctx {
    pub progress: Progress,
    pub dry_run: bool,
    pub muser_home: PathBuf,
    pub repo_root: PathBuf,
    pub container_receipt: Option<PathBuf>,
    pub model_dir_override: Option<PathBuf>,
    pub ggml_metallib: Option<PathBuf>,
    pub ggml_metallib_receipt: Option<PathBuf>,
    pub model_source_base: Option<String>,
    pub prompt_fixture: Option<PathBuf>,
    pub lane_dir_override: Option<String>,
    /// In-process proof that the immediately preceding model stage hashed the
    /// native consumer. `node enroll` run on its own still rehashes; the full
    /// pipeline does not read the same 19.6 GB twice.
    verified_native_consumer: Mutex<Option<(PathBuf, String)>>,
}

impl Ctx {
    fn new(common: &NodeCommonArgs) -> Result<Self> {
        Ok(Self {
            progress: Progress::new(common.json),
            dry_run: common.dry_run,
            muser_home: muser_home()?,
            repo_root: repo_root()?,
            container_receipt: common.container_receipt.clone(),
            model_dir_override: common.model_dir.clone(),
            ggml_metallib: common
                .ggml_metallib
                .clone()
                .or_else(|| std::env::var_os("MUSER_GGML_METALLIB").map(PathBuf::from)),
            ggml_metallib_receipt: common
                .ggml_metallib_receipt
                .clone()
                .or_else(|| std::env::var_os("MUSER_GGML_METALLIB_RECEIPT").map(PathBuf::from)),
            model_source_base: common.model_source_base.clone(),
            prompt_fixture: common.prompt_fixture.clone(),
            lane_dir_override: common.lane_dir.clone(),
            verified_native_consumer: Mutex::new(None),
        })
    }

    pub fn ssh(&self, entry: &NodeEntry) -> Result<Ssh> {
        Ssh::new(
            &entry.user,
            &entry.host,
            entry.key_path.as_deref().map(Path::new),
        )
    }

    pub fn release(&self) -> Result<Release> {
        Release::load(&self.repo_root)
    }

    pub fn receipt(&self) -> Result<ContainerReceipt> {
        match &self.container_receipt {
            Some(path) => ContainerReceipt::load(path),
            None => ContainerReceipt::newest(&artifacts::receipts_dir()?),
        }
    }

    pub fn native_identity(&self) -> Result<NativeIdentity> {
        NativeIdentity::load(&self.repo_root)
    }

    /// Where this Mac keeps the pinned GGUFs — `muser up`'s default weights
    /// path decides it unless `--model-dir` says otherwise.
    pub fn model_dir(&self) -> Result<PathBuf> {
        if let Some(path) = &self.model_dir_override {
            return Ok(path.clone());
        }
        let path = crate::model::default_model_path().map_err(|error| error.to_string())?;
        Ok(path.parent().unwrap_or(Path::new(".")).to_path_buf())
    }

    /// The URL the node fetches an artifact from, or empty for the
    /// scp-from-this-Mac fallback.
    pub fn model_source(&self, filename: &str) -> String {
        match &self.model_source_base {
            Some(base) if !base.is_empty() => {
                format!("{}/{filename}", base.trim_end_matches('/'))
            }
            _ => String::new(),
        }
    }

    /// The qualification binary that carries the production receiver.
    pub fn qualify_binary(&self) -> PathBuf {
        match std::env::var_os("MUSER_REMOTE_QUALIFY") {
            Some(path) => PathBuf::from(path),
            None => std::env::current_exe()
                .ok()
                .and_then(|path| {
                    path.parent()
                        .map(|parent| parent.join("muser-remote-qualify"))
                })
                .filter(|path| path.is_file())
                .unwrap_or_else(|| self.repo_root.join("target/release/muser-remote-qualify")),
        }
    }

    pub fn pinned_metallib(&self) -> Result<PathBuf> {
        if let Some(configured) = &self.ggml_metallib {
            let path = configured
                .canonicalize()
                .map_err(|error| format!("resolve pinned GGML metallib: {error}"))?;
            let receipt = self
                .ggml_metallib_receipt
                .clone()
                .unwrap_or_else(|| path.with_file_name("source-receipt.json"));
            crate::model::validate_metallib(&path, &receipt)
                .map_err(|error| format!("verify pinned GGML metallib: {error}"))?;
            return Ok(path);
        }
        if self.dry_run {
            return crate::model::default_metallib_path().map_err(|error| error.to_string());
        }
        self.progress.emit(
            Step::Smoke,
            Status::Info,
            "resolving the pinned 7 MB llama.cpp Metal runtime",
        );
        crate::model::ensure_metallib(None).map_err(|error| error.to_string())
    }

    pub fn remember_native_consumer(&self, path: &Path, sha256: &str) {
        *self
            .verified_native_consumer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) =
            Some((path.to_path_buf(), sha256.to_string()));
    }

    pub fn native_consumer_was_verified(&self, path: &Path, sha256: &str) -> bool {
        self.verified_native_consumer
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_ref()
            .is_some_and(|(verified_path, verified_sha256)| {
                verified_path == path && verified_sha256 == sha256
            })
    }
}

pub fn run(args: NodeArgs) -> Result<()> {
    match args.command {
        NodeCommand::Add(args) => add(args),
        NodeCommand::Preflight(args) => single(args, Step::Preflight, preflight::run),
        NodeCommand::Deploy(args) => single(args, Step::Deploy, deploy::run),
        NodeCommand::Model(args) => single(args, Step::Model, model::run),
        NodeCommand::Enroll(args) => single(args, Step::Enroll, enroll::run),
        NodeCommand::Daemon(args) => single(args, Step::Daemon, daemon::run),
        NodeCommand::Smoke(args) => single(args, Step::Smoke, smoke::run),
        NodeCommand::Qualify(args) => single(args, Step::Smoke, smoke::qualify),
        NodeCommand::Status(args) => status::run(&muser_home()?, args.json),
    }
}

/// The whole pipeline. A step's failure stops the run, is recorded on the
/// entry, and leaves the exit code non-zero — the button never reports a
/// node ready on the strength of five steps out of six.
fn add(args: NodeAddArgs) -> Result<()> {
    let mut ctx = Ctx::new(&args.common)?;
    let (user, host) = ssh::parse_target(&args.target)?;
    let name = match &args.name {
        Some(name) => name.clone(),
        None => default_name(&host),
    };
    ssh::validate_name(&name)?;
    let _operation_lock = if ctx.dry_run {
        None
    } else {
        Some(OperationLock::acquire(
            &ctx.muser_home,
            &format!("onboarding or repairing node {name}"),
        )?)
    };

    let mut registry = Registry::load(&ctx.muser_home)?;
    let retired =
        reject_duplicate_control_endpoint(&mut registry, &name, &user, &host, args.repair)?;
    if !retired.is_empty() {
        ctx.progress.emit(
            Step::Preflight,
            Status::Info,
            &format!(
                "repair retired stale registry alias{} {}; credentials were left on disk",
                if retired.len() == 1 { "" } else { "es" },
                retired.join(", ")
            ),
        );
    }
    let mut entry = match registry.get(&name) {
        Some(existing) if existing.user == user && existing.host == host => existing.clone(),
        Some(existing) => {
            return Err(format!(
                "node {name} already names {}@{} — pass --name for a second node",
                existing.user, existing.host
            ))
        }
        None => NodeEntry::draft(
            &name,
            &user,
            &host,
            &ctx.muser_home,
            ctx.lane_dir_override.as_deref(),
        ),
    };
    // A dashboard re-run has no `--model-dir` field. Preserve the directory
    // that the previous verified model stage recorded instead of silently
    // reverting to ~/.muser/models and downloading another 19.6 GB copy.
    inherit_recorded_model_dir(&mut ctx, &entry);
    if let Some(key) = &args.key {
        entry.key_path = Some(key.display().to_string());
    }
    // An explicit `--producer` restates the lane (llamacpp is stored as
    // `None`, the registry's pre-native shape); no flag leaves an existing
    // entry on the lane it was enrolled for. A new draft already selects
    // native explicitly.
    let producer_changed = args
        .producer
        .is_some_and(|producer| producer != entry.producer_kind());
    if let Some(producer) = args.producer {
        entry.producer = (producer != ProducerKind::Llamacpp).then_some(producer);
    }
    let lane_changed = ctx
        .lane_dir_override
        .as_ref()
        .is_some_and(|lane| lane != &entry.lane_dir);
    if let Some(lane) = &ctx.lane_dir_override {
        ssh::validate_remote_path(lane)?;
        entry.lane_dir = lane.clone();
    }

    if !args.repair
        && fast_rejoin_eligible(&ctx, &entry, producer_changed || lane_changed)?
        && fast_rejoin(&ctx, &mut registry, &mut entry, &args.target)?
    {
        return Ok(());
    }

    ctx.progress.emit_data(
        Step::Preflight,
        Status::Info,
        &format!(
            "{} {name} ({}@{}) — preflight, deploy, model, enroll, daemon, netqual, smoke",
            if args.repair {
                "repairing"
            } else {
                "onboarding"
            },
            entry.user,
            entry.host
        ),
        serde_json::json!({ "name": name, "dry_run": ctx.dry_run, "lane_dir": entry.lane_dir }),
    );

    let steps: [(Step, StepRunner); 6] = [
        (Step::Preflight, preflight::run),
        (Step::Deploy, deploy::run),
        (Step::Model, model::run),
        (Step::Enroll, enroll::run),
        (Step::Daemon, daemon::run),
        (Step::Smoke, smoke::run),
    ];
    for (step, run_step) in steps {
        match run_step(&ctx, &mut entry) {
            Ok(()) => {
                if step == Step::Smoke && !ctx.dry_run {
                    entry.touch(registry::STATE_HEALTHY);
                    entry.last_error = None;
                }
                persist(&ctx, &mut registry, &entry)?;
            }
            Err(error) => {
                ctx.progress.emit(step, Status::Fail, &error);
                entry.touch(failure_state(&error));
                entry.last_error = Some(record_error(&error));
                persist(&ctx, &mut registry, &entry)?;
                return Err(error);
            }
        }
    }
    if ctx.dry_run {
        ctx.progress.plan(
            Step::Smoke,
            &format!("finish the onboarding plan for {name}; state remains unchanged"),
        );
        return Ok(());
    }
    ctx.progress.emit_data(
        Step::Smoke,
        Status::Ok,
        &format!("{name} is enrolled — start inference with `muser up`"),
        serde_json::json!({
            "name": name,
            "state": entry.state,
            "next_command": "muser up",
        }),
    );
    Ok(())
}

/// Every resident producer owns the same host TCP port. Registering one
/// physical GX10 twice under two aliases would create two apparently healthy
/// topology rows whose deploy/restart operations fight over port 29591 and
/// whose last enrollment wins. Resolve OpenSSH aliases and refuse that shape
/// before any remote mutation.
fn reject_duplicate_control_endpoint(
    registry: &mut Registry,
    name: &str,
    user: &str,
    host: &str,
    repair: bool,
) -> Result<Vec<String>> {
    let requested = Ssh::new(user, host, None)?.effective_host();
    let duplicates = registry
        .nodes
        .iter()
        .filter(|entry| {
            if entry.name == name {
                return false;
            }
            let registered = entry.connect_host.clone().unwrap_or_else(|| {
                Ssh::new(
                    &entry.user,
                    &entry.host,
                    entry.key_path.as_deref().map(Path::new),
                )
                .map(|ssh| ssh.effective_host())
                .unwrap_or_else(|_| entry.host.clone())
            });
            same_control_endpoint(&requested, &registered)
        })
        .map(|entry| {
            (
                entry.name.clone(),
                entry.state.clone(),
                entry.user.clone(),
                entry.host.clone(),
            )
        })
        .collect::<Vec<_>>();
    if duplicates.is_empty() {
        return Ok(Vec::new());
    }

    let protected = duplicates
        .iter()
        .find(|(_, state, _, _)| state == registry::STATE_HEALTHY);
    if let Some((duplicate_name, _, duplicate_user, duplicate_host)) = protected {
        return Err(format!(
            "{} resolves to the producer endpoint already registered as {}; one GX10 owns one control listener on port {DAEMON_PORT} — re-add it as `muser node add {}@{} --name {}` instead of creating a duplicate",
            host, duplicate_name, duplicate_user, duplicate_host, duplicate_name
        ));
    }
    if !repair {
        let stale = duplicates
            .iter()
            .map(|(duplicate_name, state, _, _)| format!("{duplicate_name} ({state})"))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "{host} resolves to stale registration {stale}; rerun this exact command with --repair to retire the stale registry row before activating {name}"
        ));
    }

    let retired = duplicates
        .into_iter()
        .map(|(duplicate_name, _, _, _)| duplicate_name)
        .collect::<Vec<_>>();
    registry
        .nodes
        .retain(|entry| !retired.iter().any(|name| name == &entry.name));
    Ok(retired)
}

fn same_control_endpoint(left: &str, right: &str) -> bool {
    if left.eq_ignore_ascii_case(right) {
        return true;
    }
    let Ok(left) = (left, DAEMON_PORT).to_socket_addrs() else {
        return false;
    };
    let Ok(right) = (right, DAEMON_PORT).to_socket_addrs() else {
        return false;
    };
    let left = left.map(|address| address.ip()).collect::<Vec<_>>();
    right
        .map(|address| address.ip())
        .any(|address| left.contains(&address))
}

/// A repeated Add Node is a health operation when the exact deployed state
/// still matches this build. It must not rotate credentials or restart a
/// warm 128K producer merely because the operator opened the wizard twice.
/// A real authenticated handoff is still required before success is shown.
fn fast_rejoin_eligible(ctx: &Ctx, entry: &NodeEntry, topology_changed: bool) -> Result<bool> {
    if topology_changed
        || entry.producer_kind() != ProducerKind::Native
        || entry.enrollment_version != 2
        || entry.hmac_epoch <= 0
        || entry.hmac_key_id.is_empty()
        || !(entry.state == STATE_HEALTHY
            || entry
                .last_error
                .as_deref()
                .is_some_and(|error| error.starts_with(REJOIN_ERROR_PREFIX)))
        || !enroll::cluster_config(&ctx.muser_home, &entry.name).is_file()
    {
        return Ok(false);
    }
    let identity = ctx.native_identity()?;
    if entry.container_image.as_deref() != Some(identity.image_id.as_str()) {
        return Ok(false);
    }
    let consumer = ctx.model_dir()?.join(&identity.consumer.filename);
    let consumer = match consumer.canonicalize() {
        Ok(path) => path,
        Err(_) => return Ok(false),
    };
    if entry.consumer_model_path.as_deref() != Some(consumer.to_string_lossy().as_ref()) {
        return Ok(false);
    }
    let runtime_sha256 = deploy::runtime_sha256(&ctx.repo_root, entry.producer_kind())?;
    Ok(entry.runtime_sha256.as_deref() == Some(runtime_sha256.as_str()))
}

/// Returns `true` when the live proof completed (or was planned by a dry
/// run), `false` when the producer is offline and the normal repair pipeline
/// should continue. A listening-but-invalid endpoint fails closed and needs
/// an explicit `--repair`; silently rotating a live node on any probe error
/// would turn local GPU contention or a bad endpoint into remote mutation.
fn fast_rejoin(
    ctx: &Ctx,
    registry: &mut Registry,
    entry: &mut NodeEntry,
    target: &str,
) -> Result<bool> {
    ctx.progress.emit(
        Step::Preflight,
        Status::Start,
        "checking the existing native producer before changing it",
    );
    if ctx.dry_run {
        ctx.progress.plan(
            Step::Deploy,
            "keep the matching image and control runtime in place",
        );
        ctx.progress
            .plan(Step::Model, "reuse the enrolled NVFP4 artifacts");
        ctx.progress
            .plan(Step::Enroll, "keep the active enrollment and keys");
        ctx.progress
            .plan(Step::Daemon, "keep the resident producer warm");
        smoke::run(ctx, entry)?;
        ctx.progress.plan(
            Step::Smoke,
            &format!(
                "finish the rejoin plan for {}; state remains unchanged",
                entry.name
            ),
        );
        return Ok(true);
    }

    let ssh = ctx.ssh(entry)?;
    if let Err(error) = ssh.tcp_probe(DAEMON_PORT, REJOIN_PROBE_TIMEOUT) {
        ctx.progress.emit(
            Step::Daemon,
            Status::Info,
            &format!(
                "the registered producer is offline ({error}); continuing with the repair pipeline"
            ),
        );
        return Ok(false);
    }

    ctx.progress.emit(
        Step::Preflight,
        Status::Ok,
        "registered target and enrollment match this release",
    );
    ctx.progress.emit(
        Step::Deploy,
        Status::Ok,
        "image and staged control-runtime digest match; copied nothing",
    );
    ctx.progress.emit(
        Step::Model,
        Status::Ok,
        "using the enrolled NVFP4 artifacts for the live handoff",
    );
    ctx.progress.emit(
        Step::Enroll,
        Status::Ok,
        "enrollment v2 is active; kept the existing credentials",
    );
    ctx.progress.emit(
        Step::Daemon,
        Status::Ok,
        "resident producer is listening; kept it warm",
    );

    // Older builds recorded the identity manifest inside their checkout or
    // installer. Migrate it before saving this otherwise no-op rejoin, while
    // the current bundle is present and its compiled roots have validated it.
    deploy::persist_current_receipt(ctx, entry)?;

    if let Err(error) = smoke::run(ctx, entry) {
        let detail = format!(
            "{error}; the listening producer was left unchanged — rerun `muser node add {target} --name {} --repair` only if you intend to redeploy, rotate enrollment, and restart it",
            entry.name
        );
        ctx.progress.emit(Step::Smoke, Status::Fail, &detail);
        entry.touch(failure_state(&error));
        entry.last_error = Some(format!("{REJOIN_ERROR_PREFIX}{}", record_error(&error)));
        persist(ctx, registry, entry)?;
        return Err(detail);
    }

    entry.touch(STATE_HEALTHY);
    entry.last_error = None;
    persist(ctx, registry, entry)?;
    ctx.progress.emit_data(
        Step::Smoke,
        Status::Ok,
        &format!(
            "{} is enrolled; live handoff passed without restarting the producer — start inference with `muser up`",
            entry.name
        ),
        serde_json::json!({
            "name": entry.name,
            "state": entry.state,
            "reused_warm_producer": true,
            "next_command": "muser up",
        }),
    );
    Ok(true)
}

/// One step, against a node the registry already knows.
fn single(
    args: NodeStepArgs,
    step: Step,
    run_step: fn(&Ctx, &mut NodeEntry) -> Result<()>,
) -> Result<()> {
    let mut ctx = Ctx::new(&args.common)?;
    ssh::validate_name(&args.name)?;
    let _operation_lock = if ctx.dry_run {
        None
    } else {
        Some(OperationLock::acquire(
            &ctx.muser_home,
            &format!("running node step for {}", args.name),
        )?)
    };
    let mut registry = Registry::load(&ctx.muser_home)?;
    let mut entry = registry
        .get(&args.name)
        .cloned()
        .ok_or_else(|| format!("no node named {} — run `muser node add` first", args.name))?;
    inherit_recorded_model_dir(&mut ctx, &entry);
    if let Some(lane) = &ctx.lane_dir_override {
        ssh::validate_remote_path(lane)?;
        entry.lane_dir = lane.clone();
    }
    match run_step(&ctx, &mut entry) {
        Ok(()) => {
            if step == Step::Smoke && !ctx.dry_run {
                entry.touch(registry::STATE_HEALTHY);
                entry.last_error = None;
            }
            persist(&ctx, &mut registry, &entry)?;
            if step == Step::Smoke && !ctx.dry_run {
                ctx.progress.emit_data(
                    Step::Smoke,
                    Status::Ok,
                    &format!(
                        "{} is enrolled — start inference with `muser up`",
                        entry.name
                    ),
                    serde_json::json!({
                        "name": entry.name,
                        "state": entry.state,
                        "next_command": "muser up",
                    }),
                );
            }
            Ok(())
        }
        Err(error) => {
            entry.touch(failure_state(&error));
            entry.last_error = Some(record_error(&error));
            persist(&ctx, &mut registry, &entry)?;
            Err(error)
        }
    }
}

fn inherit_recorded_model_dir(ctx: &mut Ctx, entry: &NodeEntry) {
    if ctx.model_dir_override.is_none() {
        ctx.model_dir_override = entry
            .consumer_model_path
            .as_deref()
            .map(Path::new)
            .and_then(Path::parent)
            .map(Path::to_path_buf);
    }
}

/// A dry run is a plan, and a plan does not write the registry.
fn persist(ctx: &Ctx, registry: &mut Registry, entry: &NodeEntry) -> Result<()> {
    if ctx.dry_run {
        return Ok(());
    }
    registry.upsert(entry.clone());
    registry.save(&ctx.muser_home)
}

/// `~/.muser`, or `$MUSER_HOME` when a test or a second lab needs its own.
pub fn muser_home() -> Result<PathBuf> {
    if let Some(explicit) = std::env::var_os("MUSER_HOME") {
        return Ok(PathBuf::from(explicit));
    }
    let home = std::env::var_os("HOME").ok_or("HOME is unset, so ~/.muser cannot be located")?;
    Ok(PathBuf::from(home).join(".muser"))
}

/// Fully resolved local half of one enrolled native topology. This is the
/// bridge between the onboarding registry and `muser up`: users name
/// the node once instead of copying model, TLS, HMAC, and cluster paths out
/// of progress logs.
pub(crate) struct ServingTarget {
    pub name: String,
    pub model_path: PathBuf,
    pub model_sha256: String,
    pub model_validation_current: bool,
    pub cluster_config: PathBuf,
}

pub(crate) fn serving_target(name: &str) -> Result<ServingTarget> {
    ssh::validate_name(name)?;
    let home = muser_home()?;
    let registry = Registry::load(&home)?;
    let entry = registry
        .get(name)
        .ok_or_else(|| format!("no node named {name} — run `muser node add user@host` first"))?;
    if entry.producer_kind() != ProducerKind::Native {
        return Err(format!(
            "node {name} uses the kquant research lane; `muser up` ships the native NVFP4 lane"
        ));
    }
    if entry.state != registry::STATE_HEALTHY || entry.enrollment_version != 2 {
        return Err(format!(
            "node {name} is {} rather than healthy enrollment v2 — rerun `muser node add {}@{} --name {name}`",
            entry.state, entry.user, entry.host
        ));
    }

    let root = repo_root()?;
    let identity = NativeIdentity::load(&root)?;
    if entry.container_image.as_deref() != Some(identity.image_id.as_str()) {
        return Err(format!(
            "node {name} is not deployed with this release's native image — rerun `muser node add {}@{} --name {name}`",
            entry.user, entry.host
        ));
    }
    let expected_runtime = deploy::runtime_sha256(&root, ProducerKind::Native)?;
    if entry.runtime_sha256.as_deref() != Some(expected_runtime.as_str()) {
        return Err(format!(
            "node {name}'s staged runtime is older than this build — rerun `muser node add {}@{} --name {name}`",
            entry.user, entry.host
        ));
    }

    let recorded_model = entry.consumer_model_path.as_deref().ok_or_else(|| {
        format!(
            "node {name} predates automatic decoder selection — rerun `muser node add {}@{} --name {name}` once",
            entry.user, entry.host
        )
    })?;
    let model_path = PathBuf::from(recorded_model)
        .canonicalize()
        .map_err(|error| format!("resolve node {name} decoder {recorded_model}: {error}"))?;
    if model_path.file_name().and_then(|value| value.to_str())
        != Some(identity.consumer.filename.as_str())
    {
        return Err(format!(
            "node {name}'s recorded decoder is not the enrolled native artifact"
        ));
    }
    let current_validation =
        model::consumer_validation_stamp(&model_path, &identity.consumer.sha256).ok();
    let model_validation_current =
        entry.consumer_validation.as_deref() == current_validation.as_deref();
    let cluster_config = enroll::cluster_config(&home, name);
    if !cluster_config.is_file() {
        return Err(format!(
            "node {name}'s receiver configuration is missing — rerun `muser node add {}@{} --name {name}`",
            entry.user, entry.host
        ));
    }
    Ok(ServingTarget {
        name: name.to_string(),
        model_path,
        model_sha256: identity.consumer.sha256,
        model_validation_current,
        cluster_config,
    })
}

/// Persist the cheap stat-bound receipt after a compatibility fallback had
/// to re-hash an artifact written by an older Muser release. New onboarding
/// records this during the model step, so ordinary `muser up` never enters
/// this path.
pub(crate) fn remember_consumer_validation(
    name: &str,
    path: &Path,
    expected_sha256: &str,
) -> Result<()> {
    let home = muser_home()?;
    let mut registry = Registry::load(&home)?;
    let entry = registry
        .nodes
        .iter_mut()
        .find(|entry| entry.name == name)
        .ok_or_else(|| format!("no node named {name}"))?;
    let recorded = entry
        .consumer_model_path
        .as_deref()
        .ok_or_else(|| format!("node {name} has no recorded consumer"))?;
    let recorded = PathBuf::from(recorded)
        .canonicalize()
        .map_err(|error| format!("resolve node {name} consumer: {error}"))?;
    if recorded != path {
        return Err(format!("node {name} consumer changed during validation"));
    }
    entry.consumer_validation = Some(model::consumer_validation_stamp(path, expected_sha256)?);
    registry.save(&home)
}

/// Pick the most recently updated release-compatible native enrollment for
/// bare `muser up`. The shipped topology is single-producer today, so the
/// newest healthy entry is the least surprising recovery target after a
/// repair or re-enrollment. Stale/incomplete entries are skipped and remain
/// visible in the dashboard for repair.
pub(crate) fn default_serving_node() -> Result<Option<String>> {
    let home = muser_home()?;
    let registry = Registry::load(&home)?;
    let mut candidates = registry
        .nodes
        .into_iter()
        .filter(|entry| {
            entry.producer_kind() == ProducerKind::Native
                && entry.state == STATE_HEALTHY
                && entry.enrollment_version == 2
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| right.updated.cmp(&left.updated));
    for entry in candidates {
        if serving_target(&entry.name).is_ok() {
            return Ok(Some(entry.name));
        }
    }
    Ok(None)
}

/// The repository this binary's scripts and pins live in. `muser up` runs
/// from a clone, so the working directory is the first place to look.
fn repo_root() -> Result<PathBuf> {
    if let Some(explicit) = std::env::var_os("MUSER_REPO_ROOT") {
        return Ok(PathBuf::from(explicit));
    }
    let marker = Path::new("docs/release-artifacts.json");
    let mut candidates = Vec::new();
    if let Ok(directory) = std::env::current_dir() {
        candidates.push(directory);
    }
    if let Ok(executable) = std::env::current_exe() {
        candidates.push(executable.clone());
        if let Ok(canonical) = executable.canonicalize() {
            if canonical != executable {
                candidates.push(canonical);
            }
        }
    }
    for candidate in candidates {
        for ancestor in candidate.ancestors() {
            if ancestor.join(marker).is_file() {
                return Ok(ancestor.to_path_buf());
            }
        }
    }
    Err(
        "cannot find the muser repository (no docs/release-artifacts.json above the working \
         directory) — set MUSER_REPO_ROOT"
            .into(),
    )
}

/// `gx10.lab.local` becomes `gx10`; a bare address keeps its own text.
fn default_name(host: &str) -> String {
    host.split('.').next().unwrap_or(host).to_string()
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;

    fn workspace_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .expect("workspace root")
            .to_path_buf()
    }

    #[test]
    fn the_default_name_is_the_hosts_first_label() {
        assert_eq!(default_name("gx10.lab.local"), "gx10");
        assert_eq!(default_name("gx10"), "gx10");
    }

    #[test]
    fn duplicate_control_endpoints_are_detected_after_resolution() {
        assert!(same_control_endpoint("127.0.0.1", "127.0.0.1"));
        assert!(same_control_endpoint("LOCALHOST", "localhost"));
        assert!(!same_control_endpoint("127.0.0.1", "127.0.0.2"));
    }

    #[test]
    fn explicit_repair_retires_only_stale_duplicate_rows() {
        let mut stale =
            NodeEntry::draft("old", "muser", "old-alias", Path::new("/tmp/muser"), None);
        stale.connect_host = Some("127.0.0.1".into());
        stale.touch(registry::STATE_ERROR);
        let mut registry = Registry { nodes: vec![stale] };

        let error = reject_duplicate_control_endpoint(
            &mut registry,
            "current",
            "muser",
            "127.0.0.1",
            false,
        )
        .unwrap_err();
        assert!(error.contains("--repair"));
        assert_eq!(registry.nodes.len(), 1);

        let retired =
            reject_duplicate_control_endpoint(&mut registry, "current", "muser", "127.0.0.1", true)
                .unwrap();
        assert_eq!(retired, vec!["old"]);
        assert!(registry.nodes.is_empty());
    }

    #[test]
    fn repair_never_retires_a_healthy_duplicate_row() {
        let mut healthy = NodeEntry::draft(
            "active",
            "muser",
            "active-alias",
            Path::new("/tmp/muser"),
            None,
        );
        healthy.connect_host = Some("127.0.0.1".into());
        healthy.touch(registry::STATE_HEALTHY);
        let mut registry = Registry {
            nodes: vec![healthy],
        };

        let error =
            reject_duplicate_control_endpoint(&mut registry, "other", "muser", "127.0.0.1", true)
                .unwrap_err();
        assert!(error.contains("already registered as active"));
        assert_eq!(registry.nodes.len(), 1);
    }

    #[test]
    fn model_source_is_empty_unless_a_base_url_was_given() {
        let mut ctx = test_ctx();
        assert_eq!(ctx.model_source("a.gguf"), "");
        ctx.model_source_base = Some("https://mirror.example/muse/".into());
        assert_eq!(
            ctx.model_source("a.gguf"),
            "https://mirror.example/muse/a.gguf"
        );
    }

    #[test]
    fn the_model_directory_defaults_to_the_up_launcher_weights_path() {
        let ctx = test_ctx();
        assert_eq!(
            ctx.model_dir().unwrap(),
            crate::model::default_model_path()
                .unwrap()
                .parent()
                .unwrap()
        );
    }

    #[test]
    fn an_existing_node_reuses_its_recorded_model_volume() {
        let mut ctx = test_ctx();
        let mut entry = NodeEntry::draft("gx10", "muser", "gx10.local", Path::new("/tmp"), None);
        entry.consumer_model_path = Some("/Volumes/models/native.gguf".into());
        inherit_recorded_model_dir(&mut ctx, &entry);
        assert_eq!(
            ctx.model_dir_override.as_deref(),
            Some(Path::new("/Volumes/models"))
        );

        ctx.model_dir_override = Some(PathBuf::from("/explicit"));
        inherit_recorded_model_dir(&mut ctx, &entry);
        assert_eq!(
            ctx.model_dir_override.as_deref(),
            Some(Path::new("/explicit"))
        );
    }

    #[test]
    fn fast_rejoin_requires_the_exact_runtime_and_active_enrollment() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let home =
            std::env::temp_dir().join(format!("muser-fast-rejoin-{}-{unique}", std::process::id()));
        let cluster = enroll::cluster_config(&home, "gx10");
        std::fs::create_dir_all(cluster.parent().unwrap()).unwrap();
        std::fs::write(&cluster, b"{}\n").unwrap();

        let mut ctx = test_ctx();
        ctx.muser_home = home.clone();
        ctx.repo_root = workspace_root();
        ctx.model_dir_override = Some(home.join("models"));
        let identity = ctx.native_identity().unwrap();
        let runtime = deploy::runtime_sha256(&ctx.repo_root, ProducerKind::Native).unwrap();
        let consumer = ctx.model_dir().unwrap().join(&identity.consumer.filename);
        std::fs::create_dir_all(consumer.parent().unwrap()).unwrap();
        std::fs::write(&consumer, b"test-only").unwrap();
        let mut entry = NodeEntry::draft("gx10", "muser", "gx10.local", &home, None);
        entry.state = STATE_HEALTHY.into();
        entry.container_image = Some(identity.image_id);
        entry.runtime_sha256 = Some(runtime);
        entry.consumer_model_path = Some(consumer.canonicalize().unwrap().display().to_string());
        entry.enrollment_version = 2;
        entry.hmac_epoch = 1;
        entry.hmac_key_id = "muser-gx10-e1".into();

        assert!(fast_rejoin_eligible(&ctx, &entry, false).unwrap());
        assert!(!fast_rejoin_eligible(&ctx, &entry, true).unwrap());
        entry.runtime_sha256 = Some("0".repeat(64));
        assert!(!fast_rejoin_eligible(&ctx, &entry, false).unwrap());

        std::fs::remove_dir_all(home).unwrap();
    }

    fn test_ctx() -> Ctx {
        Ctx {
            progress: Progress::new(true),
            dry_run: true,
            muser_home: PathBuf::from("/tmp/muser-node-test"),
            repo_root: PathBuf::from("."),
            container_receipt: None,
            model_dir_override: None,
            ggml_metallib: None,
            ggml_metallib_receipt: None,
            model_source_base: None,
            prompt_fixture: None,
            lane_dir_override: None,
            verified_native_consumer: Mutex::new(None),
        }
    }
}

/// What the registry (and therefore the node card) records for a failure.
/// Blocked-class refusals get an operator sentence — which processes hold
/// the machine, what to do — because the raw quiet-scan output is PIDs and
/// full command lines; forensics stay in the progress and command logs.
fn record_error(error: &str) -> String {
    if failure_state(error) == registry::STATE_BLOCKED {
        let mut culprits: Vec<String> = Vec::new();
        for line in error.lines().skip(1) {
            let mut fields = line.split_whitespace();
            let Some(pid) = fields.next() else { continue };
            if pid.parse::<u32>().is_err() {
                continue;
            }
            for field in fields {
                let name = field.rsplit('/').next().unwrap_or(field);
                if name.starts_with("muser") || name == "llama-server" || name == "llama-bench" {
                    if !culprits.contains(&name.to_string()) {
                        culprits.push(name.to_string());
                    }
                    break;
                }
            }
        }
        let holders = if culprits.is_empty() {
            "another accelerator process".to_string()
        } else {
            culprits.join(", ")
        };
        return format!(
            "the accelerator is busy ({holders} is running); stop it and rerun \
             this step — a modelless `muser serve` (no --model) only serves \
             management routes and may stay up during onboarding"
        );
    }
    flatten_error(error)
}

/// The registry reader speaks single-line strings only, and a card wants one
/// line anyway: newlines become separators and the tail is bounded.
fn flatten_error(error: &str) -> String {
    let mut flat = error
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" | ");
    if flat.len() > 300 {
        flat.truncate(300);
        flat.push_str("...");
    }
    flat
}

/// A refusal by the accelerator lease or quiet-machine scan is a scheduling
/// condition, not a node fault; the card should say "blocked", not "error".
fn failure_state(error: &str) -> &'static str {
    if error.contains("another GPU process") || error.contains("accelerator lease") {
        registry::STATE_BLOCKED
    } else {
        registry::STATE_ERROR
    }
}
