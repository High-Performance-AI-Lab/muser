//! `~/.muser/nodes.toml` — the shared node registry.
//!
//! Written temp+rename so a reader (the server's `GET /v1/nodes`) never
//! observes a half-written file, and never a file that lost its previous
//! contents to a crash mid-write. Key material never lands here: the
//! registry points at `pki_dir`, which holds the keys at 0600/0700.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::Result;
use crate::timefmt::now_rfc3339;

pub const STATE_DRAFT: &str = "draft";
pub const STATE_PREFLIGHT_OK: &str = "preflight-ok";
pub const STATE_DEPLOYED: &str = "deployed";
pub const STATE_ENROLLED: &str = "enrolled";
pub const STATE_NEEDS_REENROLLMENT: &str = "needs-reenrollment";
pub const STATE_HEALTHY: &str = "healthy";
pub const STATE_ERROR: &str = "error";
/// A precondition refusal (accelerator lease, quiet-machine scan), not a
/// node fault: the pipeline can simply be rerun when the machine frees up.
pub const STATE_BLOCKED: &str = "blocked";

/// The producer lane a node is enrolled for. `muser node add --producer`
/// selects it; every later step reads it back from the registry entry, so an
/// individually rerun step cannot drift onto the other lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum ProducerKind {
    /// The llama.cpp sealed-exporter lane: combined target+DFlash transfers.
    Llamacpp,
    /// The NVFP4 vLLM resident-producer lane: plain decode only — DFlash is
    /// refused at serve time (`state.rs:validate_remote_dflash_policy`).
    Native,
}

/// Exact qualification contract selected by the enrolled producer lane.
/// Keeping this exhaustive beside `ProducerKind` means adding a lane without
/// choosing a recipe is a compile error; an unknown serialized lane is
/// refused while loading the registry, before enrollment can mint keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualificationRecipe {
    KquantTargetPlusDflash,
    NativeText,
}

impl QualificationRecipe {
    pub fn variant(self) -> &'static str {
        match self {
            Self::KquantTargetPlusDflash => "target-plus-dflash",
            Self::NativeText => "text",
        }
    }

    pub fn public_name(self) -> &'static str {
        match self {
            Self::KquantTargetPlusDflash => "kquant/target-plus-dflash",
            Self::NativeText => "native/text",
        }
    }

    pub fn includes_dflash(self) -> bool {
        matches!(self, Self::KquantTargetPlusDflash)
    }
}

impl ProducerKind {
    pub fn qualification_recipe(self) -> QualificationRecipe {
        match self {
            Self::Llamacpp => QualificationRecipe::KquantTargetPlusDflash,
            Self::Native => QualificationRecipe::NativeText,
        }
    }
}

/// The resident producer daemon's control port. The live probe in
/// `muser node status` and the server's `GET /v1/nodes` both use it.
pub const DAEMON_PORT: u16 = 29591;

/// The Mac receiver's listen port, written into the node's cluster config.
pub const RECEIVER_PORT: u16 = 29590;

/// Cross-process lease for the one-producer topology. The dashboard already
/// serializes jobs inside one server process, but a separate CLI or serving
/// process could otherwise race the registry, rotate enrollment underneath a
/// live receiver, or restart the producer during a handoff. `flock` is
/// released by the kernel even after a crash; the small file is only a human
/// readable holder receipt, not the lock itself.
pub(crate) struct OperationLock {
    _file: std::fs::File,
}

impl Drop for OperationLock {
    fn drop(&mut self) {
        use std::os::fd::AsRawFd as _;

        // `flock` follows the open file description, so a descriptor briefly
        // duplicated by a concurrently spawned process can otherwise retain
        // this lease after the owning Rust value closes its copy. An explicit
        // unlock releases that shared lock immediately; closing the file then
        // remains the kernel-backed crash fallback.
        // SAFETY: `_file` owns a live descriptor for the whole destructor.
        let _ = unsafe { libc::flock(self._file.as_raw_fd(), libc::LOCK_UN) };
    }
}

impl OperationLock {
    pub(crate) fn acquire(home: &Path, purpose: &str) -> Result<Self> {
        use std::io::{Read as _, Seek as _, Write as _};
        use std::os::fd::AsRawFd as _;
        use std::os::unix::fs::OpenOptionsExt as _;

        std::fs::create_dir_all(home)
            .map_err(|error| format!("create {}: {error}", home.display()))?;
        let path = home.join("node-operation.lock");
        let mut options = std::fs::OpenOptions::new();
        options
            .read(true)
            .write(true)
            .create(true)
            .mode(0o600)
            // Rust already requests close-on-exec on supported Unix hosts;
            // restate it as a topology invariant for crash-time inheritance.
            .custom_flags(libc::O_CLOEXEC);
        let mut file = options
            .open(&path)
            .map_err(|error| format!("open topology lock {}: {error}", path.display()))?;
        set_mode(&path, 0o600)?;

        // SAFETY: `file` owns a live descriptor for the duration of the call;
        // flock neither retains a pointer nor accesses Rust memory.
        let locked = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if locked != 0 {
            let error = std::io::Error::last_os_error();
            let mut holder = String::new();
            let _ = file.seek(std::io::SeekFrom::Start(0));
            let _ = file.read_to_string(&mut holder);
            let holder = holder.trim();
            let holder = if holder.is_empty() {
                "another muser process".to_string()
            } else {
                holder.to_string()
            };
            return Err(format!(
                "node topology is busy ({holder}); stop that operation or server before changing or probing the enrolled producer ({error})"
            ));
        }

        file.set_len(0)
            .map_err(|error| format!("truncate topology lock {}: {error}", path.display()))?;
        file.write_all(format!("pid {}: {purpose}\n", std::process::id()).as_bytes())
            .map_err(|error| format!("write topology lock {}: {error}", path.display()))?;
        file.sync_data()
            .map_err(|error| format!("sync topology lock {}: {error}", path.display()))?;
        Ok(Self { _file: file })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeEntry {
    pub name: String,
    pub host: String,
    pub user: String,
    /// Only one role exists today: this node prefills for a Mac decoder.
    pub role: String,
    pub state: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key_path: Option<String>,
    pub lane_dir: String,
    /// The enrolled producer lane. Absent means llama.cpp — every registry
    /// written before the native NVFP4 lane existed — so old files need no
    /// migration. Fresh entries explicitly select the shipped native lane.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub producer: Option<ProducerKind>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container_image: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container_receipt: Option<String>,
    /// Digest of the complete control runtime last staged by `node deploy`:
    /// bootstrap, lane scripts, and (for native) the muser_vllm package. A
    /// matching container alone is not enough to skip deployment after a
    /// client upgrade because these files live outside the image.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_sha256: Option<String>,
    /// Exact local decoder artifact verified by the model stage. Native
    /// onboarding may use a non-default `--model-dir`; recording the path is
    /// what lets `muser up --node <name>` start the same enrolled consumer
    /// without asking the user to reconstruct it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub consumer_model_path: Option<String>,
    /// Stat-bound receipt for the last complete SHA-256 verification of the
    /// local decoder. `muser up` compares this closed stamp in microseconds
    /// and re-hashes only when the file's identity or timestamps changed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub consumer_validation: Option<String>,
    pub pki_dir: String,
    pub hmac_key_id: String,
    pub hmac_epoch: i64,
    #[serde(default)]
    pub enrollment_version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub netqual_gbps: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub netqual_rtt_ms: Option<f64>,
    /// The address TCP consumers dial. Set when `host` is an ssh-config
    /// alias getaddrinfo cannot resolve (ssh still dials `host`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connect_host: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
    pub updated: String,
}

impl NodeEntry {
    /// A fresh `draft` entry. `lane_dir` is a well-formed absolute guess
    /// from the user name; preflight replaces it with the node's real
    /// `$HOME`-derived path.
    pub fn draft(name: &str, user: &str, host: &str, home: &Path, lane_dir: Option<&str>) -> Self {
        Self {
            name: name.to_string(),
            host: host.to_string(),
            user: user.to_string(),
            role: "prefill".into(),
            state: STATE_DRAFT.into(),
            key_path: None,
            lane_dir: lane_dir
                .map(str::to_string)
                .unwrap_or_else(|| format!("/home/{user}/.muser/lane/{name}")),
            producer: Some(ProducerKind::Native),
            container_image: None,
            container_receipt: None,
            runtime_sha256: None,
            consumer_model_path: None,
            consumer_validation: None,
            pki_dir: node_dir(home, name).join("pki").display().to_string(),
            hmac_key_id: String::new(),
            hmac_epoch: 0,
            enrollment_version: 0,
            netqual_gbps: None,
            netqual_rtt_ms: None,
            connect_host: None,
            last_error: None,
            updated: now_rfc3339(),
        }
    }

    pub fn touch(&mut self, state: &str) {
        self.state = state.to_string();
        self.updated = now_rfc3339();
    }

    /// The lane this entry is enrolled for. Entries without the field
    /// predate the native NVFP4 lane and are llama.cpp by construction.
    pub fn producer_kind(&self) -> ProducerKind {
        self.producer.unwrap_or(ProducerKind::Llamacpp)
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Registry {
    #[serde(default, rename = "node")]
    pub nodes: Vec<NodeEntry>,
}

impl Registry {
    pub fn path(home: &Path) -> PathBuf {
        home.join("nodes.toml")
    }

    /// A missing registry is an empty registry — the first `muser node add`
    /// on a fresh Mac must not have to create it first.
    pub fn load(home: &Path) -> Result<Self> {
        let path = Self::path(home);
        let text = match std::fs::read_to_string(&path) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Self::default())
            }
            Err(error) => return Err(format!("read {}: {error}", path.display())),
        };
        let mut registry: Self =
            toml::from_str(&text).map_err(|error| format!("parse {}: {error}", path.display()))?;
        for entry in &mut registry.nodes {
            if entry.enrollment_version < 2
                && matches!(entry.state.as_str(), STATE_ENROLLED | STATE_HEALTHY)
            {
                entry.state = STATE_NEEDS_REENROLLMENT.into();
                entry.last_error = Some(
                    "legacy enrollment must rotate through node-local-key enrollment v2".into(),
                );
            }
        }
        Ok(registry)
    }

    pub fn save(&self, home: &Path) -> Result<()> {
        let path = Self::path(home);
        std::fs::create_dir_all(home)
            .map_err(|error| format!("create {}: {error}", home.display()))?;
        let text =
            toml::to_string_pretty(self).map_err(|error| format!("encode registry: {error}"))?;
        let temporary = path.with_extension(format!("toml.tmp.{}", std::process::id()));
        write_private(&temporary, text.as_bytes())?;
        std::fs::rename(&temporary, &path).map_err(|error| {
            let _ = std::fs::remove_file(&temporary);
            format!(
                "rename {} -> {}: {error}",
                temporary.display(),
                path.display()
            )
        })?;
        let directory = std::fs::File::open(home)
            .map_err(|error| format!("open registry directory {}: {error}", home.display()))?;
        directory
            .sync_all()
            .map_err(|error| format!("sync registry directory {}: {error}", home.display()))
    }

    pub fn get(&self, name: &str) -> Option<&NodeEntry> {
        self.nodes.iter().find(|entry| entry.name == name)
    }

    pub fn upsert(&mut self, entry: NodeEntry) {
        match self
            .nodes
            .iter_mut()
            .find(|existing| existing.name == entry.name)
        {
            Some(existing) => *existing = entry,
            None => self.nodes.push(entry),
        }
    }
}

pub fn node_dir(home: &Path, name: &str) -> PathBuf {
    home.join("nodes").join(name)
}

/// Write a file only this user can read, fsync'd before it is used.
pub fn write_private(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(path)
        .map_err(|error| format!("open {}: {error}", path.display()))?;
    file.write_all(bytes)
        .map_err(|error| format!("write {}: {error}", path.display()))?;
    file.sync_all()
        .map_err(|error| format!("sync {}: {error}", path.display()))?;
    // An existing file keeps its old mode through `create(true)`; restate it.
    set_mode(path, 0o600)
}

/// Atomically replace a private state file without ever following an
/// existing symlink. This is for durable receipts and other non-key state
/// whose readers must see either the old complete value or the new complete
/// value. The temporary is created with O_EXCL in the destination directory,
/// fsync'd, renamed, and followed by a directory fsync.
pub fn write_private_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write as _;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMPORARY: AtomicU64 = AtomicU64::new(0);

    let parent = path
        .parent()
        .ok_or_else(|| format!("{} has no parent directory", path.display()))?;
    create_private_dir(parent)?;
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if !metadata.file_type().is_file() => {
            return Err(format!(
                "refusing to replace non-regular private state path {}",
                path.display()
            ))
        }
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(format!("inspect {}: {error}", path.display())),
    }

    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| format!("{} has no UTF-8 filename", path.display()))?;
    let mut temporary = None;
    let mut temporary_file = None;
    for _ in 0..128 {
        let nonce = NEXT_TEMPORARY.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(".{filename}.tmp.{}.{nonce}", std::process::id()));
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            options.mode(0o600);
        }
        match options.open(&candidate) {
            Ok(file) => {
                temporary = Some(candidate);
                temporary_file = Some(file);
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(format!("create {}: {error}", candidate.display())),
        }
    }
    let temporary = temporary.ok_or_else(|| {
        format!(
            "could not reserve a unique temporary beside {} after 128 attempts",
            path.display()
        )
    })?;
    let mut temporary_file = temporary_file.expect("temporary path and file are set together");
    let result = (|| -> Result<()> {
        temporary_file
            .write_all(bytes)
            .map_err(|error| format!("write {}: {error}", temporary.display()))?;
        temporary_file
            .sync_all()
            .map_err(|error| format!("sync {}: {error}", temporary.display()))?;
        drop(temporary_file);
        set_mode(&temporary, 0o600)?;
        std::fs::rename(&temporary, path).map_err(|error| {
            format!(
                "rename {} -> {}: {error}",
                temporary.display(),
                path.display()
            )
        })?;
        let directory = std::fs::File::open(parent)
            .map_err(|error| format!("open state directory {}: {error}", parent.display()))?;
        directory
            .sync_all()
            .map_err(|error| format!("sync state directory {}: {error}", parent.display()))
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result
}

pub fn create_private_dir(path: &Path) -> Result<()> {
    std::fs::create_dir_all(path).map_err(|error| format!("create {}: {error}", path.display()))?;
    set_mode(path, 0o700)
}

fn set_mode(path: &Path, mode: u32) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))
            .map_err(|error| format!("chmod {}: {error}", path.display()))?;
    }
    #[cfg(not(unix))]
    {
        let _ = (path, mode);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    static NEXT_TEST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    #[test]
    fn a_draft_round_trips_through_the_array_of_tables() {
        let home = Path::new("/tmp/muser-registry-test-home");
        let entry = NodeEntry::draft("gx10", "muser", "gx10.local", home, None);
        let mut registry = Registry::default();
        registry.upsert(entry);
        let text = toml::to_string_pretty(&registry).unwrap();
        assert!(text.contains("[[node]]"));
        assert!(text.contains("role = \"prefill\""));
        assert!(text.contains("state = \"draft\""));
        // Unset optionals stay absent: TOML has no null.
        assert!(!text.contains("netqual_gbps"));
        let parsed: Registry = toml::from_str(&text).unwrap();
        assert_eq!(parsed.nodes.len(), 1);
        assert_eq!(parsed.nodes[0].lane_dir, "/home/muser/.muser/lane/gx10");
    }

    /// The server reads this file while the CLI writes it, so the write has
    /// to be a rename over a complete file — never a truncate-and-fill.
    #[test]
    fn saving_is_atomic_and_leaves_no_temporary_behind() {
        let home = std::env::temp_dir().join(format!("muser-registry-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);
        let mut registry = Registry::default();
        let mut entry = NodeEntry::draft("gx10", "muser", "gx10.local", &home, None);
        entry.netqual_gbps = Some(9.4);
        entry.enrollment_version = 2;
        entry.touch(STATE_HEALTHY);
        registry.upsert(entry);
        registry.save(&home).expect("first save");
        registry.save(&home).expect("overwriting save");

        let leftovers = std::fs::read_dir(&home)
            .expect("registry directory")
            .flatten()
            .filter(|entry| entry.file_name().to_string_lossy().contains(".tmp."))
            .count();
        assert_eq!(leftovers, 0, "a temporary file survived the rename");

        let loaded = Registry::load(&home).expect("reload");
        assert_eq!(loaded.nodes.len(), 1);
        assert_eq!(loaded.nodes[0].state, STATE_HEALTHY);
        assert_eq!(loaded.nodes[0].netqual_gbps, Some(9.4));
        assert!(loaded.nodes[0].last_error.is_none());
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn private_atomic_write_replaces_only_complete_regular_files() {
        let home = std::env::temp_dir().join(format!(
            "muser-private-atomic-{}-{}",
            std::process::id(),
            NEXT_TEST.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let path = home.join("nodes/gx10/identity.json");
        write_private_atomic(&path, b"first").unwrap();
        write_private_atomic(&path, b"second").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"second");
        let leftovers = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .flatten()
            .filter(|entry| entry.file_name().to_string_lossy().contains(".tmp."))
            .count();
        assert_eq!(leftovers, 0);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            assert_eq!(
                std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        let _ = std::fs::remove_dir_all(&home);
    }

    #[cfg(unix)]
    #[test]
    fn private_atomic_write_refuses_a_symlink_without_touching_its_target() {
        use std::os::unix::fs::symlink;

        let root = std::env::temp_dir().join(format!(
            "muser-private-atomic-symlink-{}-{}",
            std::process::id(),
            NEXT_TEST.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let directory = root.join("nodes/gx10");
        std::fs::create_dir_all(&directory).unwrap();
        let outside = root.join("outside.json");
        std::fs::write(&outside, b"do not replace").unwrap();
        let path = directory.join("identity.json");
        symlink(&outside, &path).unwrap();

        let error = write_private_atomic(&path, b"replacement").unwrap_err();
        assert!(error.contains("non-regular private state path"), "{error}");
        assert_eq!(std::fs::read(&outside).unwrap(), b"do not replace");
        assert!(std::fs::symlink_metadata(&path)
            .unwrap()
            .file_type()
            .is_symlink());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_missing_registry_is_an_empty_registry() {
        let home =
            std::env::temp_dir().join(format!("muser-registry-absent-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);
        assert!(Registry::load(&home)
            .expect("absent is not an error")
            .nodes
            .is_empty());
    }

    #[test]
    fn topology_operations_are_single_process_across_open_files() {
        let home =
            std::env::temp_dir().join(format!("muser-node-operation-lock-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);
        let first = OperationLock::acquire(&home, "first test operation").unwrap();
        let error = OperationLock::acquire(&home, "racing test operation")
            .err()
            .expect("a second descriptor must not acquire the topology lock");
        assert!(error.contains("node topology is busy"), "{error}");
        assert!(error.contains("first test operation"), "{error}");
        drop(first);
        OperationLock::acquire(&home, "after release").unwrap();
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn dropping_the_owner_unlocks_an_inherited_descriptor() {
        use std::os::fd::{AsRawFd as _, FromRawFd as _};

        let home = std::env::temp_dir().join(format!(
            "muser-node-operation-inherited-lock-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&home);
        let first = OperationLock::acquire(&home, "descriptor inheritance test").unwrap();
        let duplicate = unsafe { libc::dup(first._file.as_raw_fd()) };
        assert!(
            duplicate >= 0,
            "dup topology lock: {}",
            std::io::Error::last_os_error()
        );
        // SAFETY: `dup` returned a new owned descriptor, transferred here.
        let inherited = unsafe { std::fs::File::from_raw_fd(duplicate) };

        drop(first);
        let after_release = OperationLock::acquire(&home, "after inherited release").unwrap();
        drop(after_release);
        drop(inherited);
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn upsert_replaces_rather_than_duplicates() {
        let home = Path::new("/tmp/muser-registry-test-home");
        let mut registry = Registry::default();
        registry.upsert(NodeEntry::draft("gx10", "muser", "a.local", home, None));
        registry.upsert(NodeEntry::draft("gx10", "muser", "b.local", home, None));
        assert_eq!(registry.nodes.len(), 1);
        assert_eq!(registry.nodes[0].host, "b.local");
    }

    #[test]
    fn a_fresh_node_explicitly_selects_the_native_release_lane() {
        let home = Path::new("/tmp/muser-registry-test-home");
        let entry = NodeEntry::draft("gx10", "muser", "gx10.local", home, None);
        assert_eq!(entry.producer_kind(), ProducerKind::Native);
        let mut registry = Registry::default();
        registry.upsert(entry);
        let text = toml::to_string_pretty(&registry).unwrap();
        assert!(text.contains("producer = \"native\""));
        let parsed: Registry = toml::from_str(&text).unwrap();
        assert_eq!(parsed.nodes[0].producer_kind(), ProducerKind::Native);
    }

    #[test]
    fn a_legacy_entry_without_a_lane_remains_llamacpp() {
        let parsed: Registry = toml::from_str(
            r#"
[[node]]
name = "gx10"
host = "gx10.local"
user = "muser"
role = "prefill"
state = "draft"
lane_dir = "/home/muser/.muser/lane/gx10"
pki_dir = "/tmp/pki"
hmac_key_id = ""
hmac_epoch = 0
enrollment_version = 0
updated = "2026-08-28T00:00:00Z"
"#,
        )
        .unwrap();
        assert_eq!(parsed.nodes[0].producer_kind(), ProducerKind::Llamacpp);
    }

    #[test]
    fn the_native_lane_round_trips_and_unknown_lanes_are_refused() {
        let home = Path::new("/tmp/muser-registry-test-home");
        let mut entry = NodeEntry::draft("gx10", "muser", "gx10.local", home, None);
        entry.producer = Some(ProducerKind::Native);
        let mut registry = Registry::default();
        registry.upsert(entry);
        let text = toml::to_string_pretty(&registry).unwrap();
        assert!(text.contains("producer = \"native\""));
        let parsed: Registry = toml::from_str(&text).unwrap();
        assert_eq!(parsed.nodes[0].producer_kind(), ProducerKind::Native);
        // A lane this build does not know is a parse error, never a default.
        let foreign = text.replace("producer = \"native\"", "producer = \"tpu\"");
        assert!(toml::from_str::<Registry>(&foreign).is_err());
    }

    #[test]
    fn every_declared_lane_selects_a_recipe_and_unknown_lane_fails_before_enrollment() {
        assert_eq!(
            ProducerKind::Llamacpp.qualification_recipe(),
            QualificationRecipe::KquantTargetPlusDflash
        );
        assert_eq!(
            ProducerKind::Native.qualification_recipe(),
            QualificationRecipe::NativeText
        );

        let home = std::env::temp_dir().join(format!(
            "muser-registry-unknown-lane-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&home);
        std::fs::create_dir_all(&home).unwrap();
        std::fs::write(
            Registry::path(&home),
            r#"[[node]]
name = "future"
host = "gx10"
user = "muser"
role = "prefill"
state = "draft"
lane_dir = "/tmp/lane"
producer = "unknown"
pki_dir = "/tmp/pki"
hmac_key_id = ""
hmac_epoch = 0
enrollment_version = 0
updated = "now"
"#,
        )
        .unwrap();
        let error = Registry::load(&home).unwrap_err();
        assert!(error.contains("unknown variant"), "{error}");
        let _ = std::fs::remove_dir_all(&home);
    }

    #[test]
    fn legacy_healthy_nodes_require_reenrollment_v2() {
        let home =
            std::env::temp_dir().join(format!("muser-registry-legacy-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&home);
        let mut registry = Registry::default();
        let mut entry = NodeEntry::draft("gx10", "muser", "gx10.local", &home, None);
        entry.touch(STATE_HEALTHY);
        registry.upsert(entry);
        registry.save(&home).unwrap();
        let loaded = Registry::load(&home).unwrap();
        assert_eq!(loaded.nodes[0].state, STATE_NEEDS_REENROLLMENT);
        assert!(loaded.nodes[0]
            .last_error
            .as_deref()
            .unwrap()
            .contains("node-local-key"));
        let _ = std::fs::remove_dir_all(&home);
    }
}
