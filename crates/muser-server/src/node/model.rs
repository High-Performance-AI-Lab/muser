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
                "verify the Mac decode artifact {} ({} bytes, {})",
                consumer.display(),
                identity.consumer.bytes,
                &identity.consumer.sha256[..12]
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

    verify_regular_file(
        &consumer,
        identity.consumer.bytes,
        &identity.consumer.sha256,
        "native Mac decode artifact",
    )?;
    ctx.progress.emit(
        Step::Model,
        Status::Info,
        &format!(
            "native Mac decode artifact verified ({})",
            &identity.consumer.sha256[..12]
        ),
    );
    if !bootstrap_local.is_file() {
        return Err(format!("{} is missing", bootstrap_local.display()));
    }
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

pub(super) fn verify_regular_file(
    path: &Path,
    bytes: u64,
    expected: &str,
    label: &str,
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
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|error| format!("hash {label} {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    let actual = format!("{:x}", digest.finalize());
    if actual != expected {
        return Err(format!(
            "{label} SHA-256 mismatch: expected {expected}, got {actual}"
        ));
    }
    Ok(())
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
