//! Step 1 — is this machine a node at all?
//!
//! One SSH round trip runs every probe and prints `key=value` lines; the
//! answers are reported one detail line per probe so a failure names the
//! probe that failed, not "preflight failed".

use super::progress::{Status, Step};
use super::registry::{NodeEntry, STATE_PREFLIGHT_OK};
use super::ssh::{validate_remote_path, Ssh};
use super::{Ctx, Result};

/// Peak first-install space for the image (including the public archive
/// fallback), checkpoint, and Docker extraction scratch.
pub const REQUIRED_DISK_GIB: u64 = 64;
/// The released lane promises the 131072-token topology, not merely a shallow
/// demo. Reject hardware that cannot hold that resident working set.
const REQUIRED_MEMORY_GIB: u64 = 96;
/// Available memory is load-sensitive, so this remains a warning; the daemon
/// owns the definitive accelerator lease and startup check.
const ADVISORY_AVAILABLE_MEMORY_GIB: u64 = 48;
const REQUIRED_LOCAL_DISK_GIB: u64 = 48;

const PROBE: &str = r#"set -u
printf 'home=%s\n' "$HOME"
printf 'arch=%s\n' "$(uname -m)"
printf 'kernel=%s\n' "$(uname -sr)"
printf 'ssh_client=%s\n' "${SSH_CLIENT%% *}"
if command -v nvidia-smi >/dev/null 2>&1; then
    printf 'nvidia_smi=%s\n' "$(command -v nvidia-smi)"
    printf 'driver=%s\n' "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1)"
    printf 'gpu=%s\n' "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)"
    printf 'compute_cap=%s\n' "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1)"
else
    printf 'nvidia_smi=\n'
fi
if command -v docker >/dev/null 2>&1; then
    printf 'docker=%s\n' "$(command -v docker)"
    if docker info >/dev/null 2>&1; then printf 'docker_daemon=1\n'; else printf 'docker_daemon=0\n'; fi
else
    printf 'docker=\n'
    printf 'docker_daemon=0\n'
fi
if command -v curl >/dev/null 2>&1 && command -v sha256sum >/dev/null 2>&1 && command -v zstd >/dev/null 2>&1; then
    printf 'artifact_tools=1\n'
else
    printf 'artifact_tools=0\n'
fi
if command -v systemctl >/dev/null 2>&1; then printf 'systemctl=1\n'; else printf 'systemctl=0\n'; fi
printf 'disk_kib=%s\n' "$(df -Pk "$HOME" | awk 'NR==2 {print $4}')"
printf 'mem_total_kib=%s\n' "$(awk '/MemTotal/ {print $2}' /proc/meminfo 2>/dev/null)"
printf 'mem_kib=%s\n' "$(awk '/MemAvailable/ {print $2}' /proc/meminfo 2>/dev/null)"
"#;

#[derive(Debug, Default, Clone)]
pub struct Probes {
    pub home: String,
    pub arch: String,
    pub kernel: String,
    pub ssh_client: String,
    pub nvidia_smi: String,
    pub driver: String,
    pub gpu: String,
    pub compute_cap: String,
    pub docker: String,
    pub docker_daemon: bool,
    pub artifact_tools: bool,
    pub systemctl: bool,
    pub disk_kib: u64,
    pub mem_total_kib: u64,
    pub mem_kib: u64,
}

pub fn run(ctx: &Ctx, entry: &mut NodeEntry) -> Result<()> {
    let ssh = ctx.ssh(entry)?;
    ctx.progress.emit(
        Step::Preflight,
        Status::Start,
        &format!("probing {}", ssh.target()),
    );
    if ctx.dry_run {
        ctx.progress.plan_command(
            Step::Preflight,
            &format!(
                "probe {} for aarch64, SM121, driver 580+, docker, artifact tools, \
                 {REQUIRED_DISK_GIB} GiB disk, and {REQUIRED_MEMORY_GIB} GiB memory",
                ssh.target()
            ),
            &ssh.argv(&[]),
        );
        ctx.progress.plan(
            Step::Preflight,
            &format!(
                "record the probe results and set state={STATE_PREFLIGHT_OK} for {}",
                entry.name
            ),
        );
        if entry.producer_kind() == super::registry::ProducerKind::Native {
            ctx.progress.plan(
                Step::Preflight,
                &format!(
                    "require {REQUIRED_LOCAL_DISK_GIB} GiB free for first-time native Mac artifact assembly"
                ),
            );
        }
        ctx.progress
            .plan(Step::Preflight, "finish without probing the node");
        return Ok(());
    }

    check_local_artifact_capacity(ctx, entry)?;

    let probes = parse(&ssh.run(PROBE, &[])?);
    ctx.progress.emit_data(
        Step::Preflight,
        Status::Info,
        &format!("{} runs {} ({})", entry.name, probes.arch, probes.kernel),
        serde_json::json!({ "arch": probes.arch, "kernel": probes.kernel }),
    );
    if probes.arch != "aarch64" {
        return Err(format!(
            "node architecture is {}, and the pinned producer container is arm64-only",
            probes.arch
        ));
    }
    if probes.nvidia_smi.is_empty() {
        return Err(
            "nvidia-smi is not on the node's PATH — this is not a CUDA prefill node".into(),
        );
    }
    if probes.driver.is_empty() {
        return Err("nvidia-smi reported no driver version — the driver is not loaded".into());
    }
    let driver_major = probes
        .driver
        .split('.')
        .next()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0);
    if driver_major < 580 {
        return Err(format!(
            "NVIDIA driver {} is older than the pinned CUDA 13.3 runtime's 580-series floor",
            probes.driver
        ));
    }
    if probes.compute_cap != "12.1" {
        return Err(format!(
            "GPU {} reports compute capability {}; the released native image is qualified for GB10/SM121 (12.1)",
            probes.gpu, probes.compute_cap
        ));
    }
    ctx.progress.emit_data(
        Step::Preflight,
        Status::Info,
        &format!("NVIDIA driver {} on {}", probes.driver, probes.gpu),
        serde_json::json!({
            "driver": probes.driver,
            "gpu": probes.gpu,
            "compute_capability": probes.compute_cap,
        }),
    );
    if probes.docker.is_empty() {
        return Err("docker is not on the node's PATH".into());
    }
    if !probes.docker_daemon {
        return Err(format!(
            "{} is present but `docker info` failed — the daemon is unreachable for {}",
            probes.docker, entry.user
        ));
    }
    if !probes.artifact_tools {
        return Err(
            "the node needs curl, sha256sum, and zstd for pinned artifact installation".into(),
        );
    }
    ctx.progress.emit_data(
        Step::Preflight,
        Status::Info,
        &format!("docker at {} with a reachable daemon", probes.docker),
        serde_json::json!({ "docker": probes.docker, "systemctl": probes.systemctl }),
    );

    let disk_gib = probes.disk_kib / (1024 * 1024);
    if disk_gib < REQUIRED_DISK_GIB {
        return Err(format!(
            "{}'s filesystem has {disk_gib} GiB free; the pinned weights plus the exporter image need {REQUIRED_DISK_GIB} GiB",
            probes.home
        ));
    }
    ctx.progress.emit_data(
        Step::Preflight,
        Status::Info,
        &format!("{disk_gib} GiB free under {}", probes.home),
        serde_json::json!({ "disk_gib": disk_gib, "required_gib": REQUIRED_DISK_GIB }),
    );
    let total_mem_gib = probes.mem_total_kib / (1024 * 1024);
    if total_mem_gib < REQUIRED_MEMORY_GIB {
        return Err(format!(
            "the node has {total_mem_gib} GiB memory; the released 131072-token GB10 lane requires {REQUIRED_MEMORY_GIB} GiB"
        ));
    }
    let mem_gib = probes.mem_kib / (1024 * 1024);
    let memory_detail = format!("{mem_gib} GiB memory available");
    if mem_gib < ADVISORY_AVAILABLE_MEMORY_GIB {
        ctx.progress.emit_data(
            Step::Preflight,
            Status::Info,
            &format!("WARN: {memory_detail} (under {ADVISORY_AVAILABLE_MEMORY_GIB} GiB)"),
            serde_json::json!({
                "mem_gib": mem_gib,
                "total_mem_gib": total_mem_gib,
                "advisory_gib": ADVISORY_AVAILABLE_MEMORY_GIB,
            }),
        );
    } else {
        ctx.progress.emit_data(
            Step::Preflight,
            Status::Info,
            &memory_detail,
            serde_json::json!({ "mem_gib": mem_gib, "total_mem_gib": total_mem_gib }),
        );
    }

    // The lane lives under the node's real home, whatever the account's home
    // actually is; the drafted `/home/<user>/...` guess is only a placeholder.
    if ctx.lane_dir_override.is_none() {
        let lane = format!(
            "{}/.muser/lane/{}",
            probes.home.trim_end_matches('/'),
            entry.name
        );
        validate_remote_path(&lane)?;
        if lane != entry.lane_dir {
            ctx.progress.emit(
                Step::Preflight,
                Status::Info,
                &format!("lane directory resolves to {lane}"),
            );
            entry.lane_dir = lane;
        }
    }
    let effective = ssh.effective_host();
    entry.connect_host = (effective != entry.host).then_some(effective);
    entry.touch(STATE_PREFLIGHT_OK);
    entry.last_error = None;
    ctx.progress.emit(
        Step::Preflight,
        Status::Ok,
        &format!("{} is a usable prefill node", entry.name),
    );
    Ok(())
}

fn check_local_artifact_capacity(ctx: &Ctx, entry: &NodeEntry) -> Result<()> {
    if entry.producer_kind() != super::registry::ProducerKind::Native {
        return Ok(());
    }
    let identity = ctx.native_identity()?;
    let model_dir = ctx.model_dir()?;
    let target = model_dir.join(&identity.consumer.filename);
    if std::fs::symlink_metadata(&target)
        .ok()
        .is_some_and(|metadata| {
            metadata.file_type().is_file()
                && !metadata.file_type().is_symlink()
                && metadata.len() == identity.consumer.bytes
        })
    {
        return Ok(());
    }
    let existing = model_dir
        .ancestors()
        .find(|path| path.is_dir())
        .ok_or_else(|| {
            format!(
                "no existing parent for model directory {}",
                model_dir.display()
            )
        })?;
    let output = std::process::Command::new("df")
        .args(["-Pk"])
        .arg(existing)
        .output()
        .map_err(|error| format!("inspect free space under {}: {error}", existing.display()))?;
    if !output.status.success() {
        return Err(format!(
            "df could not inspect free space under {}: {}",
            existing.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let available_kib = String::from_utf8_lossy(&output.stdout)
        .lines()
        .rfind(|line| !line.trim().is_empty())
        .and_then(|line| line.split_whitespace().nth(3))
        .and_then(|value| value.parse::<u64>().ok())
        .ok_or_else(|| format!("df returned no free-space value for {}", existing.display()))?;
    let available_gib = available_kib / (1024 * 1024);
    if available_gib < REQUIRED_LOCAL_DISK_GIB {
        return Err(format!(
            "the Mac has {available_gib} GiB free under {}; first-time native artifact assembly needs {REQUIRED_LOCAL_DISK_GIB} GiB — free space or pass --model-dir on a larger volume",
            existing.display()
        ));
    }
    ctx.progress.emit_data(
        Step::Preflight,
        Status::Info,
        &format!("{available_gib} GiB free for first-time native Mac artifact assembly"),
        serde_json::json!({
            "local_disk_gib": available_gib,
            "local_required_gib": REQUIRED_LOCAL_DISK_GIB,
            "model_dir": model_dir,
        }),
    );
    Ok(())
}

/// The Mac's address as the node sees it — what the node dials back on.
/// Taken from `$SSH_CLIENT`, so it is the route that already works rather
/// than a guess from a local interface list.
pub fn advertised_receiver_host(ssh: &Ssh) -> Result<String> {
    let probes = parse(&ssh.run(PROBE, &[])?);
    if probes.ssh_client.is_empty() {
        return Err("the node reported no $SSH_CLIENT address for this Mac".into());
    }
    Ok(probes.ssh_client)
}

fn parse(output: &str) -> Probes {
    let mut probes = Probes::default();
    for line in output.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let value = value.trim();
        match key {
            "home" => probes.home = value.to_string(),
            "arch" => probes.arch = value.to_string(),
            "kernel" => probes.kernel = value.to_string(),
            "ssh_client" => probes.ssh_client = value.to_string(),
            "nvidia_smi" => probes.nvidia_smi = value.to_string(),
            "driver" => probes.driver = value.to_string(),
            "gpu" => probes.gpu = value.to_string(),
            "compute_cap" => probes.compute_cap = value.to_string(),
            "docker" => probes.docker = value.to_string(),
            "docker_daemon" => probes.docker_daemon = value == "1",
            "artifact_tools" => probes.artifact_tools = value == "1",
            "systemctl" => probes.systemctl = value == "1",
            "disk_kib" => probes.disk_kib = value.parse().unwrap_or(0),
            "mem_total_kib" => probes.mem_total_kib = value.parse().unwrap_or(0),
            "mem_kib" => probes.mem_kib = value.parse().unwrap_or(0),
            _ => {}
        }
    }
    probes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_output_parses_into_answers() {
        let probes = parse(
            "home=/home/muser\narch=aarch64\ndriver=580.1\ngpu=NVIDIA GB10\ncompute_cap=12.1\n\
             docker=/usr/bin/docker\ndocker_daemon=1\nartifact_tools=1\n\
             disk_kib=104857600\nmem_total_kib=134217728\nmem_kib=33554432\n\
             systemctl=1\nssh_client=10.0.0.2\n",
        );
        assert_eq!(probes.home, "/home/muser");
        assert_eq!(probes.arch, "aarch64");
        assert!(probes.docker_daemon);
        assert!(probes.systemctl);
        assert_eq!(probes.disk_kib / (1024 * 1024), 100);
        assert_eq!(probes.ssh_client, "10.0.0.2");
    }

    #[test]
    fn a_missing_probe_is_absent_rather_than_wrong() {
        let probes = parse("arch=x86_64\nnvidia_smi=\ndocker=\n");
        assert!(probes.nvidia_smi.is_empty());
        assert!(probes.docker.is_empty());
        assert_eq!(probes.disk_kib, 0);
    }
}
