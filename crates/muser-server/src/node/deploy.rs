//! Step 2 — the pinned producer runtime.
//!
//! Two halves: the sealed exporter *image* (verified present by exact image
//! ID, built by `scripts/build_gx10_container.py` if it is not) and the lane
//! *runtime* (`muser_prefilld.py` and the scripts it drives), pushed into
//! `lane_dir/llamacpp`. The receiver refuses to arm an exporter whose
//! receipt does not match its config, so the image ID recorded here is the
//! one the handoff config will name.

use std::path::{Path, PathBuf};
use std::process::Command;

use super::artifacts::ContainerReceipt;
use super::progress::{Status, Step};
use super::registry::{NodeEntry, ProducerKind, STATE_DEPLOYED};
use super::{Ctx, Result};

/// The lane runtime: the set `install_on_gx10.sh` shipped, plus the systemd
/// unit template `bootstrap_node.sh daemon` instantiates from
/// `LANE/llamacpp/muser-prefilld.service`.
pub const RUNTIME_FILES: [&str; 8] = [
    "muser_prefilld.py",
    "muser-prefilld",
    "muser-prefilld.service",
    "muser_v2_send.py",
    "llamacpp_session_send.py",
    "protocol.py",
    "muser_prefill_producer.sh",
    "muse-glimmer-30b.layout.json",
];

/// The NVFP4 vLLM producer runtime, staged into `lane_dir/vllm` only for
/// `--producer native`: the resident-producer pair and the container recipe.
/// The `muser_vllm` package they import from travels with them, walked from
/// the tree rather than listed — a module the producer starts importing can
/// never be left behind by a stale pin.
pub const VLLM_RUNTIME_FILES: [&str; 4] = [
    "muser_native_prefilld.py",
    "resident_producer.py",
    "request_producer.py",
    "Dockerfile",
];

/// The Python package the vLLM resident producer imports from.
pub const VLLM_PACKAGE: &str = "muser_vllm";

const IMAGE_PRESENT: &str = r#"set -u
docker image inspect --format '{{.Id}}' "$1" 2>/dev/null || true
"#;

const PULL_PINNED_IMAGE: &str = r#"set -eu
docker pull "$1" >&2
docker image inspect --format '{{.Id}}' "$1"
"#;

/// Public-release fallback for nodes that cannot anonymously pull GHCR. Each
/// chunk is resumable and independently verified before the complete zstd
/// stream is hashed and admitted to Docker. The final image ID is still the
/// same immutable identity; the archive is transport, never a second build.
const LOAD_PINNED_IMAGE_ARCHIVE: &str = r#"set -eu
lane=$1
expected_image=$2
archive_sha=$3
archive_bytes=$4
shift 4
cache="$lane/image-cache"
list="$cache/archive-parts.list"
umask 077
mkdir -p "$cache"
: > "$list"
while [ "$#" -ge 4 ]; do
    name=$1
    bytes=$2
    sha=$3
    url=$4
    shift 4
    path="$cache/$name"
    valid=0
    if [ -f "$path" ] && [ ! -L "$path" ]; then
        actual_bytes=$(wc -c < "$path" | tr -d ' ')
        if [ "$actual_bytes" = "$bytes" ] && printf '%s  %s\n' "$sha" "$path" | sha256sum --check --strict --status; then
            valid=1
        fi
    fi
    if [ "$valid" -ne 1 ]; then
        if [ -e "$path" ] && { [ ! -f "$path" ] || [ -L "$path" ]; }; then
            echo "refusing non-regular image cache path: $path" >&2
            exit 2
        fi
        curl --fail --location --retry 5 --continue-at - --output "$path" "$url" || {
            rm -f -- "$path"
            curl --fail --location --retry 5 --output "$path" "$url"
        }
        actual_bytes=$(wc -c < "$path" | tr -d ' ')
        [ "$actual_bytes" = "$bytes" ]
        printf '%s  %s\n' "$sha" "$path" | sha256sum --check --strict --status
    fi
    printf '%s\n' "$path" >> "$list"
    echo "verified image archive chunk $name"
done
[ "$#" -eq 0 ]
actual_bytes=0
while IFS= read -r path; do
    size=$(wc -c < "$path" | tr -d ' ')
    actual_bytes=$((actual_bytes + size))
done < "$list"
[ "$actual_bytes" = "$archive_bytes" ]
actual_sha=$(while IFS= read -r path; do cat "$path"; done < "$list" | sha256sum | awk '{print $1}')
[ "$actual_sha" = "$archive_sha" ]
while IFS= read -r path; do cat "$path"; done < "$list" | zstd -dc | docker load >&2
loaded=$(docker image inspect --format '{{.Id}}' "$expected_image")
[ "$loaded" = "$expected_image" ]
while IFS= read -r path; do rm -f -- "$path"; done < "$list"
rm -f -- "$list"
printf '%s\n' "$loaded"
"#;

const MAKE_LANE: &str = r#"set -eu
umask 077
mkdir -p "$1" "$1/llamacpp" "$1/pki" "$1/models" "$1/work"
"#;

/// Native mode only: the vLLM lane directory (and its package) beside
/// `llamacpp/`, which the default lane continues to own alone.
const MAKE_VLLM_LANE: &str = r#"set -eu
umask 077
mkdir -p "$1/vllm/muser_vllm"
"#;

pub fn run(ctx: &Ctx, entry: &mut NodeEntry) -> Result<()> {
    if entry.producer_kind() == ProducerKind::Native {
        return run_native(ctx, entry);
    }
    let ssh = ctx.ssh(entry)?;
    let lane = entry.lane_dir.clone();
    ctx.progress.emit(
        Step::Deploy,
        Status::Start,
        "resolving the pinned producer image",
    );

    let receipt = ctx.receipt()?;
    ctx.progress.emit_data(
        Step::Deploy,
        Status::Info,
        &format!(
            "receipt {} pins {} (adapter {})",
            receipt.path.display(),
            receipt.image_id,
            &receipt.adapter_sha256[..12]
        ),
        serde_json::json!({
            "container_receipt": receipt.path,
            "container_image": receipt.image_id,
            "image_tag": receipt.image_tag,
            "source_commit": receipt.source_commit,
        }),
    );

    if ctx.dry_run {
        ctx.progress.plan_command(
            Step::Deploy,
            &format!("create the lane at {lane}"),
            &ssh.argv(&[&lane]),
        );
        ctx.progress.plan_command(
            Step::Deploy,
            &format!("check for image {} on the node", receipt.image_id),
            &ssh.argv(&[&receipt.image_id]),
        );
        ctx.progress.plan_command(
            Step::Deploy,
            "build the image if it is absent",
            &build_argv(ctx, entry, &receipt, Path::new("<new-receipt>.json")),
        );
        for name in RUNTIME_FILES {
            ctx.progress.plan_command(
                Step::Deploy,
                &format!("push {name} into {lane}/llamacpp"),
                &ssh.scp_argv(
                    &ctx.repo_root.join("scripts/gx10/llamacpp").join(name),
                    &format!("{lane}/llamacpp/{name}"),
                ),
            );
        }
        if entry.producer_kind() == ProducerKind::Native {
            for name in VLLM_RUNTIME_FILES {
                ctx.progress.plan_command(
                    Step::Deploy,
                    &format!("push {name} into {lane}/vllm"),
                    &ssh.scp_argv(
                        &ctx.repo_root.join("scripts/gx10/vllm").join(name),
                        &format!("{lane}/vllm/{name}"),
                    ),
                );
            }
            ctx.progress.plan(
                Step::Deploy,
                &format!("push the {VLLM_PACKAGE} package into {lane}/vllm"),
            );
        }
        ctx.progress.plan_command(
            Step::Deploy,
            &format!("push {} into {lane}", super::model::BOOTSTRAP),
            &ssh.scp_argv(
                &bootstrap(ctx),
                &format!("{lane}/{}", super::model::BOOTSTRAP),
            ),
        );
        ctx.progress
            .plan(Step::Deploy, "finish without deploying anything");
        return Ok(());
    }

    ssh.run(MAKE_LANE, &[&lane])?;
    let present = !ssh
        .run(IMAGE_PRESENT, &[&receipt.image_id])?
        .trim()
        .is_empty();
    let receipt = if present {
        ctx.progress.emit(
            Step::Deploy,
            Status::Info,
            &format!("image {} is already on the node", receipt.image_id),
        );
        receipt
    } else {
        ctx.progress.emit(
            Step::Deploy,
            Status::Info,
            &format!(
                "image {} is absent — building it from llama.cpp {}",
                receipt.image_id, receipt.source_commit
            ),
        );
        build(ctx, entry, &receipt)?
    };

    for name in RUNTIME_FILES {
        let local = ctx.repo_root.join("scripts/gx10/llamacpp").join(name);
        if !local.is_file() {
            return Err(format!("lane runtime file {} is missing", local.display()));
        }
        ssh.scp(&local, &format!("{lane}/llamacpp/{name}"))?;
    }
    // The bootstrap script is what the model and daemon steps drive, so it
    // lands here rather than only alongside its first caller.
    ssh.scp(
        &bootstrap(ctx),
        &format!("{lane}/{}", super::model::BOOTSTRAP),
    )?;
    ctx.progress.emit(
        Step::Deploy,
        Status::Info,
        &format!(
            "{} lane runtime files installed in {lane}/llamacpp",
            RUNTIME_FILES.len()
        ),
    );

    if entry.producer_kind() == ProducerKind::Native {
        ssh.run(MAKE_VLLM_LANE, &[&lane])?;
        for name in VLLM_RUNTIME_FILES {
            let local = ctx.repo_root.join("scripts/gx10/vllm").join(name);
            if !local.is_file() {
                return Err(format!(
                    "vLLM lane runtime file {} is missing",
                    local.display()
                ));
            }
            ssh.scp(&local, &format!("{lane}/vllm/{name}"))?;
        }
        let modules = vllm_package_modules(&ctx.repo_root)?;
        let module_count = modules.len();
        for local in &modules {
            let name = local
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| format!("vLLM package module {} has no name", local.display()))?;
            ssh.scp(local, &format!("{lane}/vllm/{VLLM_PACKAGE}/{name}"))?;
        }
        ctx.progress.emit(
            Step::Deploy,
            Status::Info,
            &format!(
                "vLLM native runtime installed in {lane}/vllm ({} files, {module_count} {VLLM_PACKAGE} modules)",
                VLLM_RUNTIME_FILES.len()
            ),
        );
    }

    entry.container_image = Some(receipt.image_id.clone());
    entry.container_receipt = Some(receipt.path.display().to_string());
    entry.touch(STATE_DEPLOYED);
    entry.last_error = None;
    ctx.progress.emit_data(
        Step::Deploy,
        Status::Ok,
        &format!("producer runtime deployed ({})", receipt.image_id),
        serde_json::json!({ "container_image": receipt.image_id }),
    );
    Ok(())
}

fn run_native(ctx: &Ctx, entry: &mut NodeEntry) -> Result<()> {
    let ssh = ctx.ssh(entry)?;
    let lane = entry.lane_dir.clone();
    let identity = ctx.native_identity()?;
    ctx.progress.emit(
        Step::Deploy,
        Status::Start,
        "resolving the pinned native NVFP4 producer image",
    );
    ctx.progress.emit_data(
        Step::Deploy,
        Status::Info,
        &format!(
            "native identity pins {} (adapter {})",
            identity.image_id,
            &identity.adapter_sha256[..12]
        ),
        serde_json::json!({
            "runtime_identity": identity.path,
            "container_image": identity.image_id,
            "image_tag": identity.image_tag,
            "vllm_commit": identity.vllm_commit,
        }),
    );

    if ctx.dry_run {
        ctx.progress.plan_command(
            Step::Deploy,
            &format!("create the native lane at {lane}"),
            &ssh.argv(&[&lane]),
        );
        ctx.progress.plan_command(
            Step::Deploy,
            &format!("check for exact image {}", identity.image_id),
            &ssh.argv(&[&identity.image_id]),
        );
        ctx.progress.plan_command(
            Step::Deploy,
            "pull the pinned tag only if absent, then require the exact image ID",
            &ssh.argv(&[&identity.image_tag]),
        );
        ctx.progress.plan(
            Step::Deploy,
            &format!(
                "if the registry pull is unavailable, resume {} public archive chunks and load the same exact image ID",
                identity.image_archive_parts.len()
            ),
        );
        for name in RUNTIME_FILES {
            ctx.progress.plan_command(
                Step::Deploy,
                &format!("push {name} into {lane}/llamacpp"),
                &ssh.scp_argv(
                    &ctx.repo_root.join("scripts/gx10/llamacpp").join(name),
                    &format!("{lane}/llamacpp/{name}"),
                ),
            );
        }
        for name in VLLM_RUNTIME_FILES {
            ctx.progress.plan_command(
                Step::Deploy,
                &format!("push {name} into {lane}/vllm"),
                &ssh.scp_argv(
                    &ctx.repo_root.join("scripts/gx10/vllm").join(name),
                    &format!("{lane}/vllm/{name}"),
                ),
            );
        }
        ctx.progress
            .plan(Step::Deploy, "finish without deploying anything");
        return Ok(());
    }

    ssh.run(MAKE_LANE, &[&lane])?;
    ssh.run(MAKE_VLLM_LANE, &[&lane])?;
    let present = ssh
        .run(IMAGE_PRESENT, &[&identity.image_id])?
        .trim()
        .to_string();
    if present != identity.image_id {
        ctx.progress.emit(
            Step::Deploy,
            Status::Info,
            &format!(
                "exact native image is absent; pulling {} and verifying its immutable ID",
                identity.image_tag
            ),
        );
        let pull = ssh.exec(PULL_PINNED_IMAGE, &[&identity.image_tag], None)?;
        let pulled = pull.stdout.trim();
        if pull.code != 0 || pulled != identity.image_id {
            ctx.progress.emit(
                Step::Deploy,
                Status::Info,
                "registry pull unavailable or mismatched; using the pinned public image archive",
            );
            load_native_image_archive(ctx, entry, &identity)?;
        }
    } else {
        ctx.progress.emit(
            Step::Deploy,
            Status::Info,
            &format!(
                "exact native image {} is already on the node",
                identity.image_id
            ),
        );
    }

    for name in RUNTIME_FILES {
        let local = ctx.repo_root.join("scripts/gx10/llamacpp").join(name);
        if !local.is_file() {
            return Err(format!("lane runtime file {} is missing", local.display()));
        }
        ssh.scp(&local, &format!("{lane}/llamacpp/{name}"))?;
    }
    for name in VLLM_RUNTIME_FILES {
        let local = ctx.repo_root.join("scripts/gx10/vllm").join(name);
        if !local.is_file() {
            return Err(format!(
                "native runtime file {} is missing",
                local.display()
            ));
        }
        ssh.scp(&local, &format!("{lane}/vllm/{name}"))?;
    }
    for local in vllm_package_modules(&ctx.repo_root)? {
        let name = local
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| format!("vLLM package module {} has no name", local.display()))?;
        ssh.scp(&local, &format!("{lane}/vllm/{VLLM_PACKAGE}/{name}"))?;
    }
    ssh.scp(
        &bootstrap(ctx),
        &format!("{lane}/{}", super::model::BOOTSTRAP),
    )?;

    entry.container_image = Some(identity.image_id.clone());
    entry.container_receipt = Some(identity.path.display().to_string());
    entry.touch(STATE_DEPLOYED);
    entry.last_error = None;
    ctx.progress.emit_data(
        Step::Deploy,
        Status::Ok,
        &format!("native NVFP4 runtime deployed ({})", identity.image_id),
        serde_json::json!({
            "container_image": identity.image_id,
            "checkpoint_artifact_sha256": identity.checkpoint_artifact_sha256,
        }),
    );
    Ok(())
}

fn load_native_image_archive(
    ctx: &Ctx,
    entry: &NodeEntry,
    identity: &super::artifacts::NativeIdentity,
) -> Result<()> {
    let ssh = ctx.ssh(entry)?;
    let mut args = vec![
        entry.lane_dir.clone(),
        identity.image_id.clone(),
        identity.image_archive.sha256.clone(),
        identity.image_archive.bytes.to_string(),
    ];
    for part in &identity.image_archive_parts {
        let configured = ctx.model_source(&part.filename);
        args.extend([
            part.filename.clone(),
            part.bytes.to_string(),
            part.sha256.clone(),
            if configured.is_empty() {
                part.url.clone()
            } else {
                configured
            },
        ]);
    }
    let refs = args.iter().map(String::as_str).collect::<Vec<_>>();
    let relay = |line: &str| ctx.progress.emit(Step::Deploy, Status::Info, line);
    let output = ssh.run_relayed(LOAD_PINNED_IMAGE_ARCHIVE, &refs, &relay)?;
    if output.lines().last() != Some(identity.image_id.as_str()) {
        return Err(format!(
            "public image archive did not load exact image {}",
            identity.image_id
        ));
    }
    Ok(())
}

fn bootstrap(ctx: &Ctx) -> std::path::PathBuf {
    ctx.repo_root
        .join("scripts/gx10")
        .join(super::model::BOOTSTRAP)
}

/// Every module of the local `muser_vllm` package, sorted so the push order
/// is stable. A missing or empty package is a hard error: the resident
/// producer imports from it, and a partial push would fail on the node hours
/// later instead of here.
fn vllm_package_modules(repo_root: &Path) -> Result<Vec<PathBuf>> {
    let dir = repo_root.join("scripts/gx10/vllm").join(VLLM_PACKAGE);
    let entries = std::fs::read_dir(&dir)
        .map_err(|error| format!("vLLM lane package {} is unreadable: {error}", dir.display()))?;
    let mut modules = Vec::new();
    for entry in entries {
        let path = entry
            .map_err(|error| format!("list {}: {error}", dir.display()))?
            .path();
        if path.is_file() && path.extension() == Some(std::ffi::OsStr::new("py")) {
            modules.push(path);
        }
    }
    modules.sort();
    if modules.is_empty() {
        return Err(format!(
            "vLLM lane package {} holds no Python modules",
            dir.display()
        ));
    }
    Ok(modules)
}

/// `build_gx10_container.py` writes its own authenticated receipt; the
/// pipeline never fabricates one, it just reads back what the builder wrote.
fn build(ctx: &Ctx, entry: &NodeEntry, receipt: &ContainerReceipt) -> Result<ContainerReceipt> {
    let output = super::artifacts::receipts_dir()?.join(format!(
        "gx10-container-{}-{}-{}.json",
        &receipt.source_commit[..7],
        entry.name,
        crate::timefmt::now_rfc3339().replace([':', '-'], "")
    ));
    let argv = build_argv(ctx, entry, receipt, &output);
    let status = Command::new(&argv[0])
        .args(&argv[1..])
        .status()
        .map_err(|error| format!("spawn {}: {error}", argv[0]))?;
    if !status.success() {
        return Err(format!(
            "build_gx10_container.py exited {}",
            status
                .code()
                .map(|code| code.to_string())
                .unwrap_or_else(|| "by signal".into())
        ));
    }
    let built = ContainerReceipt::load(&output)?;
    if built.adapter_sha256 != receipt.adapter_sha256 {
        return Err(format!(
            "built adapter {} differs from the pinned receipt's {}",
            built.adapter_sha256, receipt.adapter_sha256
        ));
    }
    Ok(built)
}

fn build_argv(
    ctx: &Ctx,
    entry: &NodeEntry,
    receipt: &ContainerReceipt,
    output: &Path,
) -> Vec<String> {
    // The builder's own `--host` grammar is a plain SSH alias, so the node's
    // user must come from the operator's ssh config, not from this argv.
    vec![
        "python3".into(),
        ctx.repo_root
            .join("scripts/build_gx10_container.py")
            .display()
            .to_string(),
        "--host".into(),
        entry.host.clone(),
        "--llama-dir".into(),
        "llama.cpp".into(),
        "--llama-revision".into(),
        receipt.source_commit.clone(),
        "--image-tag".into(),
        receipt.image_tag.clone(),
        "--cuda-matmul".into(),
        receipt.cuda_matmul.clone(),
        "--output".into(),
        output.display().to_string(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workspace_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .expect("workspace root")
            .to_path_buf()
    }

    /// Everything the native lane stages must be in this tree: a missing
    /// artifact is a deploy-time error, never a node-side surprise.
    #[test]
    fn the_native_lane_runtime_is_complete_in_this_tree() {
        let root = workspace_root();
        for name in VLLM_RUNTIME_FILES {
            assert!(
                root.join("scripts/gx10/vllm").join(name).is_file(),
                "scripts/gx10/vllm/{name} is missing"
            );
        }
        let modules = vllm_package_modules(&root).expect("the muser_vllm package walks");
        let names: Vec<_> = modules
            .iter()
            .map(|path| path.file_name().unwrap().to_str().unwrap().to_string())
            .collect();
        for required in ["__init__.py", "connector.py", "receipt.py"] {
            assert!(
                names.iter().any(|name| name == required),
                "the muser_vllm package lost {required}"
            );
        }
        // Sorted, and the bytecode cache is never part of the push.
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
        assert!(!names.iter().any(|name| name.ends_with(".pyc")));
    }

    #[test]
    fn the_native_lane_remote_scripts_are_valid_shell() {
        for script in [MAKE_VLLM_LANE, LOAD_PINNED_IMAGE_ARCHIVE] {
            let mut child = std::process::Command::new("bash")
                .arg("-n")
                .stdin(std::process::Stdio::piped())
                .spawn()
                .unwrap();
            use std::io::Write as _;
            child
                .stdin
                .as_mut()
                .unwrap()
                .write_all(script.as_bytes())
                .unwrap();
            assert!(child.wait().unwrap().success());
        }
    }
}
