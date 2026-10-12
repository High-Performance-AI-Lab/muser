//! Step 3 — the pinned weights, on the node.
//!
//! `scripts/gx10/bootstrap_node.sh model` owns the install; this step
//! uploads it and drives it once per artifact. Its contract:
//!
//! ```text
//! bootstrap_node.sh model --dir <remote dir> --name <filename> \
//!                         --bytes <size> --sha256 <hex> [--source <url>]
//! ```
//!
//! With a `--source` the node fetches the file itself, resumably
//! (`curl -C -`). Without one the script verifies what is already there and
//! exits 3 — "upload me" — at which point this step scp's the file from the
//! Mac's model directory and asks again. Either way the last word is the
//! SHA-256 the script checks, so a truncated transfer is never mistaken for
//! an install.

use std::io::{BufReader, Read as _};
use std::path::Path;

use sha2::{Digest as _, Sha256};

use super::artifacts::Artifact;
use super::progress::{Status, Step};
use super::registry::{NodeEntry, ProducerKind};
use super::{Ctx, Result};

pub const BOOTSTRAP: &str = "bootstrap_node.sh";
const CONSUMER_VALIDATION_SCHEMA: &str = "muser.model-validation.v1";

/// `bootstrap_node.sh model`'s "no verified copy and no source" exit.
const UPLOAD_REQUIRED: i32 = 3;

// `--source` is omitted rather than passed empty: remote arguments travel
// through a shell command line, where an empty argument disappears and
// shifts every argument after it.
const DRIVE: &str = r#"set -eu
if [ -n "${6:-}" ]; then
    bash "$1/bootstrap_node.sh" model --dir "$2" --name "$3" --bytes "$4" --sha256 "$5" --source "$6"
else
    bash "$1/bootstrap_node.sh" model --dir "$2" --name "$3" --bytes "$4" --sha256 "$5"
fi
"#;

pub fn run(ctx: &Ctx, entry: &mut NodeEntry) -> Result<()> {
    if entry.producer_kind() == ProducerKind::Native {
        return run_native(ctx, entry);
    }
    let ssh = ctx.ssh(entry)?;
    let lane = entry.lane_dir.clone();
    let models = format!("{lane}/models");
    let release = ctx.release()?;
    let local_dir = ctx.model_dir()?;
    let bootstrap_local = ctx.repo_root.join("scripts/gx10").join(BOOTSTRAP);

    ctx.progress.emit(
        Step::Model,
        Status::Start,
        &format!(
            "installing the pinned weights for revision {}",
            release.revision
        ),
    );

    let wanted = [("target", &release.target), ("dflash", &release.dflash)];

    if ctx.dry_run {
        ctx.progress.plan_command(
            Step::Model,
            &format!("upload {BOOTSTRAP} to {lane}"),
            &ssh.scp_argv(&bootstrap_local, &format!("{lane}/{BOOTSTRAP}")),
        );
        for (role, artifact) in wanted {
            let source = ctx.model_source(&artifact.filename);
            let bytes = artifact.bytes.to_string();
            let mut args = vec![
                lane.as_str(),
                models.as_str(),
                artifact.filename.as_str(),
                bytes.as_str(),
                artifact.sha256.as_str(),
            ];
            if !source.is_empty() {
                args.push(source.as_str());
            }
            ctx.progress.plan_command(
                Step::Model,
                &format!(
                    "install the {role} GGUF {} ({}) into {models} from {}",
                    artifact.filename,
                    &artifact.sha256[..12],
                    if source.is_empty() {
                        "this Mac over scp"
                    } else {
                        &source
                    }
                ),
                &ssh.argv(&args),
            );
            if source.is_empty() {
                ctx.progress.plan_command(
                    Step::Model,
                    "scp it only if the node's copy fails its SHA-256",
                    &ssh.scp_argv(
                        &local_dir.join(&artifact.filename),
                        &format!("{models}/{}", artifact.filename),
                    ),
                );
            }
        }
        ctx.progress
            .plan(Step::Model, "finish without moving any weights");
        return Ok(());
    }

    if !bootstrap_local.is_file() {
        return Err(format!(
            "{} is missing — the node bootstrap script has not been installed in this tree",
            bootstrap_local.display()
        ));
    }
    ssh.scp(&bootstrap_local, &format!("{lane}/{BOOTSTRAP}"))?;

    for (role, artifact) in wanted {
        install(ctx, entry, role, artifact, &models, &local_dir, None)?;
    }

    let local_target = local_dir.join(&release.target.filename);
    entry.consumer_model_path = local_target
        .canonicalize()
        .ok()
        .map(|path| path.display().to_string());
    entry.updated = crate::timefmt::now_rfc3339();
    entry.last_error = None;
    ctx.progress.emit(
        Step::Model,
        Status::Ok,
        &format!("target and DFlash weights verified on {}", entry.name),
    );
    Ok(())
}

fn run_native(ctx: &Ctx, entry: &mut NodeEntry) -> Result<()> {
    let ssh = ctx.ssh(entry)?;
    let lane = entry.lane_dir.clone();
    let identity = ctx.native_identity()?;
    let models = format!("{lane}/models/{}", identity.checkpoint_directory);
    let local_dir = ctx.model_dir()?;
    let consumer = local_dir.join(&identity.consumer.filename);
    let checkpoint_local = local_dir.join(&identity.checkpoint_directory);
    let bootstrap_local = ctx.repo_root.join("scripts/gx10").join(BOOTSTRAP);

    ctx.progress.emit(
        Step::Model,
        Status::Start,
        &format!(
            "acquiring the pinned NVFP4 checkpoint {}",
            identity.checkpoint_revision
        ),
    );

    if ctx.dry_run {
        ctx.progress.plan(
            Step::Model,
            &format!(
                "verify or resume-download the Mac decode artifact {} ({} bytes, {}, {} parts)",
                consumer.display(),
                identity.consumer.bytes,
                &identity.consumer.sha256[..12],
                identity.consumer_parts.len()
            ),
        );
        for file in &identity.checkpoint_files {
            let artifact = identity.checkpoint_artifact(file);
            let source = native_source(ctx, &artifact);
            ctx.progress.plan_command(
                Step::Model,
                &format!(
                    "install checkpoint file {} ({}) into {models} from {source}",
                    file.filename,
                    &file.sha256[..12]
                ),
                &ssh.argv(&[
                    &lane,
                    &models,
                    &file.filename,
                    &file.bytes.to_string(),
                    &file.sha256,
                    &source,
                ]),
            );
        }
        ctx.progress
            .plan(Step::Model, "finish without moving any weights");
        return Ok(());
    }

    if !bootstrap_local.is_file() {
        return Err(format!("{} is missing", bootstrap_local.display()));
    }

    let consumer_validation_current = consumer_validation_is_current(
        entry.consumer_validation.as_deref(),
        &consumer,
        &identity.consumer.sha256,
    );
    if consumer_validation_current {
        ctx.progress.emit(
            Step::Model,
            Status::Info,
            "using the unchanged decoder's prior full SHA-256 verification",
        );
    }

    // The consumer lives on this Mac and the checkpoint lives on the GX10.
    // Their digest checks (or first download streams) have no shared mutable
    // state, and enrollment has not begun yet, so overlap them. Both results
    // are joined and required before any trust material is rotated.
    std::thread::scope(|scope| -> Result<()> {
        let consumer_check = scope.spawn(|| {
            if consumer_validation_current {
                Ok(())
            } else {
                acquire_native_consumer(ctx, &identity, &consumer)
            }
        });
        let checkpoint_check = (|| -> Result<()> {
            ssh.scp(&bootstrap_local, &format!("{lane}/{BOOTSTRAP}"))?;
            for file in &identity.checkpoint_files {
                let artifact = identity.checkpoint_artifact(file);
                let source = native_source(ctx, &artifact);
                install(
                    ctx,
                    entry,
                    "NVFP4 checkpoint",
                    &artifact,
                    &models,
                    &checkpoint_local,
                    Some(&source),
                )?;
            }
            Ok(())
        })();
        let consumer_check = consumer_check
            .join()
            .map_err(|_| "native Mac decode artifact worker panicked".to_string())?;
        consumer_check?;
        checkpoint_check
    })?;

    ctx.remember_native_consumer(&consumer, &identity.consumer.sha256);
    entry.consumer_validation = Some(consumer_validation_stamp(
        &consumer,
        &identity.consumer.sha256,
    )?);
    entry.consumer_model_path = Some(
        consumer
            .canonicalize()
            .map_err(|error| format!("resolve native Mac decode artifact: {error}"))?
            .display()
            .to_string(),
    );
    ctx.progress.emit(
        Step::Model,
        Status::Info,
        &format!(
            "native Mac decode artifact verified ({})",
            &identity.consumer.sha256[..12]
        ),
    );

    entry.updated = crate::timefmt::now_rfc3339();
    entry.last_error = None;
    ctx.progress.emit_data(
        Step::Model,
        Status::Ok,
        "NVFP4 checkpoint and native Mac decode artifact verified",
        serde_json::json!({
            "checkpoint_artifact_sha256": identity.checkpoint_artifact_sha256,
            "checkpoint_bytes": identity.checkpoint_total_bytes,
            "consumer_sha256": identity.consumer.sha256,
        }),
    );
    Ok(())
}

fn native_source(ctx: &Ctx, artifact: &Artifact) -> String {
    let configured = ctx.model_source(&artifact.filename);
    if configured.is_empty() {
        artifact.url.clone()
    } else {
        configured
    }
}

fn acquire_native_consumer(
    ctx: &Ctx,
    identity: &super::artifacts::NativeIdentity,
    target: &Path,
) -> Result<()> {
    match std::fs::symlink_metadata(target) {
        Ok(_) => {
            ctx.progress.emit(
                Step::Model,
                Status::Info,
                &format!(
                    "native Mac decode artifact is present; verifying {:.1} GB",
                    identity.consumer.bytes as f64 / 1e9
                ),
            );
            let announced = std::cell::Cell::new((0_u64, std::time::Instant::now()));
            return verify_regular_file_with_progress(
                target,
                identity.consumer.bytes,
                &identity.consumer.sha256,
                "native Mac decode artifact",
                |done, total| {
                    let gib = done / (1024 * 1024 * 1024);
                    let (last_gib, last_update) = announced.get();
                    if gib > last_gib
                        || done == total
                        || last_update.elapsed() >= std::time::Duration::from_secs(15)
                    {
                        announced.set((gib, std::time::Instant::now()));
                        ctx.progress.emit(
                            Step::Model,
                            Status::Info,
                            &format!(
                                "verifying native Mac decode artifact {:.1}/{:.1} GB",
                                done as f64 / 1e9,
                                total as f64 / 1e9
                            ),
                        );
                    }
                },
            );
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(format!(
                "inspect native Mac decode artifact {}: {error}",
                target.display()
            ))
        }
    }

    ctx.progress.emit(
        Step::Model,
        Status::Info,
        &format!(
            "native Mac decode artifact is absent; downloading {} pinned parts ({:.1} GB total)",
            identity.consumer_parts.len(),
            identity.consumer.bytes as f64 / 1e9
        ),
    );
    let parts = identity
        .consumer_parts
        .iter()
        .map(|part| {
            let configured = ctx.model_source(&part.filename);
            crate::model::PinnedArtifact {
                filename: part.filename.clone(),
                revision: part.revision.clone(),
                url: if configured.is_empty() {
                    part.url.clone()
                } else {
                    configured
                },
                bytes: part.bytes,
                sha256: part.sha256.clone(),
            }
        })
        .collect::<Vec<_>>();
    let announced = std::cell::Cell::new((0_u64, std::time::Instant::now()));
    crate::model::download_pinned_parts(
        &parts,
        target,
        identity.consumer.bytes,
        &identity.consumer.sha256,
        |done, total| {
            let gib = done / (1024 * 1024 * 1024);
            let (last_gib, last_update) = announced.get();
            if gib > last_gib
                || done == total
                || last_update.elapsed() >= std::time::Duration::from_secs(15)
            {
                announced.set((gib, std::time::Instant::now()));
                ctx.progress.emit(
                    Step::Model,
                    Status::Info,
                    &format!(
                        "native Mac decode download {:.1}/{:.1} GB",
                        done as f64 / 1e9,
                        total as f64 / 1e9
                    ),
                );
            }
        },
    )
    .map_err(|error| format!("download native Mac decode artifact: {error}"))?;
    Ok(())
}

pub(super) fn verify_regular_file(
    path: &Path,
    bytes: u64,
    expected: &str,
    label: &str,
) -> Result<()> {
    verify_regular_file_with_progress(path, bytes, expected, label, |_, _| {})
}

fn verify_regular_file_with_progress(
    path: &Path,
    bytes: u64,
    expected: &str,
    label: &str,
    mut progress: impl FnMut(u64, u64),
) -> Result<()> {
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|error| format!("inspect {label} {}: {error}", path.display()))?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Err(format!("{label} is not a regular file: {}", path.display()));
    }
    if metadata.len() != bytes {
        return Err(format!(
            "{label} byte count mismatch: expected {bytes}, got {} at {}",
            metadata.len(),
            path.display()
        ));
    }
    let file = std::fs::File::open(path)
        .map_err(|error| format!("open {label} {}: {error}", path.display()))?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut hashed = 0_u64;
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|error| format!("hash {label} {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
        hashed = hashed.saturating_add(read as u64);
        progress(hashed, bytes);
    }
    let actual = format!("{:x}", digest.finalize());
    if actual != expected {
        return Err(format!(
            "{label} SHA-256 mismatch: expected {expected}, got {actual}"
        ));
    }
    Ok(())
}

/// A cheap, closed receipt for one already-completed content hash. Device,
/// inode, mtime and ctime jointly catch replacement and in-place edits; ctime
/// cannot be restored by an unprivileged caller after changing the bytes.
/// The registry itself is private to the same account that runs Muser, so the
/// boundary here is accidental/stale data rather than a hostile local owner.
pub(super) fn consumer_validation_stamp(path: &Path, expected: &str) -> Result<String> {
    if expected.len() != 64
        || !expected
            .bytes()
            .all(|value| value.is_ascii_digit() || (b'a'..=b'f').contains(&value))
    {
        return Err("consumer validation digest is not lowercase SHA-256".into());
    }
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|error| format!("inspect validated consumer {}: {error}", path.display()))?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Err(format!(
            "validated consumer is not a regular file: {}",
            path.display()
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;
        Ok(format!(
            "{CONSUMER_VALIDATION_SCHEMA}:{expected}:{}:{}:{}:{}:{}:{}:{}",
            metadata.len(),
            metadata.dev(),
            metadata.ino(),
            metadata.mtime(),
            metadata.mtime_nsec(),
            metadata.ctime(),
            metadata.ctime_nsec(),
        ))
    }
    #[cfg(not(unix))]
    {
        let modified = metadata
            .modified()
            .map_err(|error| format!("read consumer modified time: {error}"))?
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|_| "consumer modified time predates Unix epoch".to_string())?;
        Ok(format!(
            "{CONSUMER_VALIDATION_SCHEMA}:{expected}:{}:0:0:{}:{}:0:0",
            metadata.len(),
            modified.as_secs(),
            modified.subsec_nanos(),
        ))
    }
}

fn consumer_validation_is_current(recorded: Option<&str>, path: &Path, expected: &str) -> bool {
    recorded.is_some() && consumer_validation_stamp(path, expected).ok().as_deref() == recorded
}

fn install(
    ctx: &Ctx,
    entry: &NodeEntry,
    role: &str,
    artifact: &Artifact,
    models: &str,
    local_dir: &Path,
    source_override: Option<&str>,
) -> Result<()> {
    let ssh = ctx.ssh(entry)?;
    let configured = ctx.model_source(&artifact.filename);
    let source = if configured.is_empty() {
        source_override.unwrap_or_default().to_string()
    } else {
        configured
    };
    ctx.progress.emit(
        Step::Model,
        Status::Info,
        &format!("{role}: {} ({})", artifact.filename, &artifact.sha256[..12]),
    );
    let bytes = artifact.bytes.to_string();
    let args = [
        entry.lane_dir.as_str(),
        models,
        artifact.filename.as_str(),
        bytes.as_str(),
        artifact.sha256.as_str(),
        source.as_str(),
    ];
    let relay = |line: &str| ctx.progress.emit(Step::Model, Status::Info, line);
    let outcome = ssh.exec(DRIVE, &args, Some(&relay))?;
    if outcome.code == 0 {
        ctx.progress.emit(
            Step::Model,
            Status::Info,
            &format!("{role}: installed and SHA-256 verified on the node"),
        );
        return Ok(());
    }
    if outcome.code != UPLOAD_REQUIRED {
        return Err(format!(
            "{role}: the node could not install {} — {}",
            artifact.filename,
            outcome.failure(&ssh.target())
        ));
    }
    if !source.is_empty() {
        return Err(format!(
            "{role}: {} was neither fetched from {source} nor verified on the node",
            artifact.filename
        ));
    }

    let local = local_dir.join(&artifact.filename);
    if !local.is_file() {
        return Err(format!(
            "{role}: {} is absent locally; place the pinned artifact at that \
             path (docs/release-artifacts.json names it) or pass \
             --model-source-base",
            local.display()
        ));
    }
    ctx.progress.emit(
        Step::Model,
        Status::Info,
        &format!("{role}: copying {} to {models}", local.display()),
    );
    ssh.scp(&local, &format!("{models}/{}", artifact.filename))?;
    ssh.run_relayed(DRIVE, &args, &relay).map_err(|error| {
        format!(
            "{role}: {} failed its SHA-256 after the copy — {error}",
            artifact.filename
        )
    })?;
    ctx.progress.emit(
        Step::Model,
        Status::Info,
        &format!("{role}: installed and verified"),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consumer_validation_stamp_changes_on_replacement_and_digest_identity() {
        let root = std::env::temp_dir().join(format!(
            "muser-consumer-stamp-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("consumer.gguf");
        std::fs::write(&path, b"same-size-bytes").unwrap();
        let digest = "a".repeat(64);
        let original = consumer_validation_stamp(&path, &digest).unwrap();
        assert_eq!(consumer_validation_stamp(&path, &digest).unwrap(), original);
        assert_ne!(
            consumer_validation_stamp(&path, &"b".repeat(64)).unwrap(),
            original
        );

        let replacement = root.join("replacement.gguf");
        std::fs::write(&replacement, b"same-size-bytes").unwrap();
        std::fs::rename(&replacement, &path).unwrap();
        assert_ne!(consumer_validation_stamp(&path, &digest).unwrap(), original);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn prior_consumer_verification_is_reused_only_for_an_unchanged_file() {
        let root = std::env::temp_dir().join(format!(
            "muser-consumer-reuse-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("decoder.gguf");
        std::fs::write(&path, b"verified decoder").unwrap();
        let digest = "a".repeat(64);
        let stamp = consumer_validation_stamp(&path, &digest).unwrap();

        assert!(consumer_validation_is_current(Some(&stamp), &path, &digest));
        assert!(!consumer_validation_is_current(None, &path, &digest));
        assert!(!consumer_validation_is_current(
            Some(&stamp),
            &path,
            &"b".repeat(64)
        ));

        std::fs::write(&path, b"replaced decoder").unwrap();
        assert!(!consumer_validation_is_current(
            Some(&stamp),
            &path,
            &digest
        ));
        std::fs::remove_dir_all(root).unwrap();
    }
}
