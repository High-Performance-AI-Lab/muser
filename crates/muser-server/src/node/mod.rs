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
//! | `netqual` + `smoke` | three real 2048/256 lane-specific handoffs, and what the link did |
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
//! `muser node add` exits 0 only when the smoke step passed.

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

use std::path::{Path, PathBuf};

use crate::cli::{NodeAddArgs, NodeArgs, NodeCommand, NodeCommonArgs, NodeStepArgs};

use self::artifacts::{ContainerReceipt, NativeIdentity, Release};
use self::progress::{Progress, Status, Step};
use self::registry::{NodeEntry, ProducerKind, Registry};
use self::ssh::Ssh;

pub type Result<T> = std::result::Result<T, String>;
type StepRunner = fn(&Ctx, &mut NodeEntry) -> Result<()>;

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
            None => ContainerReceipt::newest(&artifacts::receipts_dir()),
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
            None => self.repo_root.join("target/release/muser-remote-qualify"),
        }
    }

    pub fn pinned_metallib(&self) -> Result<PathBuf> {
        use sha2::{Digest as _, Sha256};

        let path = self
            .ggml_metallib
            .as_ref()
            .ok_or_else(|| {
                "GX10 qualification requires --ggml-metallib or MUSER_GGML_METALLIB".to_string()
            })?
            .canonicalize()
            .map_err(|error| format!("resolve pinned GGML metallib: {error}"))?;
        let metadata = std::fs::symlink_metadata(&path)
            .map_err(|error| format!("inspect {}: {error}", path.display()))?;
        if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
            return Err(format!("{} is not a regular metallib", path.display()));
        }
        let receipt_path = self
            .ggml_metallib_receipt
            .clone()
            .unwrap_or_else(|| path.with_file_name("source-receipt.json"));
        let receipt: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&receipt_path)
                .map_err(|error| format!("read {}: {error}", receipt_path.display()))?,
        )
        .map_err(|error| format!("parse {}: {error}", receipt_path.display()))?;
        let bytes =
            std::fs::read(&path).map_err(|error| format!("read {}: {error}", path.display()))?;
        let digest = format!("{:x}", Sha256::digest(&bytes));
        if receipt.get("schema").and_then(serde_json::Value::as_str)
            != Some("muser.llama_metallib.source_receipt.v1")
            || receipt
                .get("source_commit")
                .and_then(serde_json::Value::as_str)
                != Some(artifacts::LLAMA_CPP_COMMIT)
            || receipt
                .get("binary_size_bytes")
                .and_then(serde_json::Value::as_u64)
                != Some(bytes.len() as u64)
            || receipt
                .get("binary_sha256")
                .and_then(serde_json::Value::as_str)
                != Some(digest.as_str())
        {
            return Err(format!(
                "{} does not bind {} to pinned llama.cpp {}",
                receipt_path.display(),
                path.display(),
                artifacts::LLAMA_CPP_COMMIT
            ));
        }
        Ok(path)
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
        NodeCommand::Status(args) => status::run(&muser_home()?, args.json),
    }
}

/// The whole pipeline. A step's failure stops the run, is recorded on the
/// entry, and leaves the exit code non-zero — the button never reports a
/// node ready on the strength of five steps out of six.
fn add(args: NodeAddArgs) -> Result<()> {
    let ctx = Ctx::new(&args.common)?;
    let (user, host) = ssh::parse_target(&args.target)?;
    let name = match &args.name {
        Some(name) => name.clone(),
        None => default_name(&host),
    };
    ssh::validate_name(&name)?;

    let mut registry = Registry::load(&ctx.muser_home)?;
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
    if let Some(key) = &args.key {
        entry.key_path = Some(key.display().to_string());
    }
    // An explicit `--producer` restates the lane (llamacpp is stored as
    // `None`, the registry's pre-native shape); no flag leaves an existing
    // entry on the lane it was enrolled for.
    if let Some(producer) = args.producer {
        entry.producer = (producer != ProducerKind::Llamacpp).then_some(producer);
    }
    if let Some(lane) = &ctx.lane_dir_override {
        ssh::validate_remote_path(lane)?;
        entry.lane_dir = lane.clone();
    }

    ctx.progress.emit_data(
        Step::Preflight,
        Status::Info,
        &format!(
            "onboarding {name} ({}@{}) — preflight, deploy, model, enroll, daemon, netqual, smoke",
            entry.user, entry.host
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
        &format!("{name} is ready for disaggregated prefill"),
        serde_json::json!({ "name": name, "state": entry.state }),
    );
    Ok(())
}

/// One step, against a node the registry already knows.
fn single(
    args: NodeStepArgs,
    step: Step,
    run_step: fn(&Ctx, &mut NodeEntry) -> Result<()>,
) -> Result<()> {
    let ctx = Ctx::new(&args.common)?;
    ssh::validate_name(&args.name)?;
    let mut registry = Registry::load(&ctx.muser_home)?;
    let mut entry = registry
        .get(&args.name)
        .cloned()
        .ok_or_else(|| format!("no node named {} — run `muser node add` first", args.name))?;
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
                    &format!("{} is ready for disaggregated prefill", entry.name),
                    serde_json::json!({ "name": entry.name, "state": entry.state }),
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

/// The repository this binary's scripts and pins live in. `muser up` runs
/// from a clone, so the working directory is the first place to look.
fn repo_root() -> Result<PathBuf> {
    if let Some(explicit) = std::env::var_os("MUSER_REPO_ROOT") {
        return Ok(PathBuf::from(explicit));
    }
    let marker = Path::new("docs/release-artifacts.json");
    let candidates = std::env::current_dir()
        .ok()
        .into_iter()
        .chain(std::env::current_exe().ok());
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

    #[test]
    fn the_default_name_is_the_hosts_first_label() {
        assert_eq!(default_name("gx10.lab.local"), "gx10");
        assert_eq!(default_name("gx10"), "gx10");
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
