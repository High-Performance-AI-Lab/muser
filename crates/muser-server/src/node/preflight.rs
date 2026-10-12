//! Step 1 — is this machine a node at all?
//!
//! One SSH round trip runs every probe and prints `key=value` lines; the
//! answers are reported one detail line per probe so a failure names the
//! probe that failed, not "preflight failed".

use super::progress::{Status, Step};
use super::registry::{NodeEntry, RECEIVER_PORT, STATE_PREFLIGHT_OK};
use super::ssh::{validate_remote_path, Ssh};
use super::{Ctx, Result};

/// Peak first-install space for the image (including the public archive
/// fallback), checkpoint, and Docker extraction scratch.
pub const REQUIRED_DISK_GIB: u64 = 64;
const REQUIRED_SPLIT_HOME_DISK_GIB: u64 = 48;
const REQUIRED_SPLIT_DOCKER_DISK_GIB: u64 = 32;
/// The released lane promises the 131072-token topology, not merely a shallow
/// demo. Reject hardware that cannot hold that resident working set.
const REQUIRED_MEMORY_GIB: u64 = 96;
/// Available memory is load-sensitive, so this remains a warning; the daemon
/// owns the definitive accelerator lease and startup check.
const ADVISORY_AVAILABLE_MEMORY_GIB: u64 = 48;
const REQUIRED_LOCAL_DISK_GIB: u64 = 48;
const REQUIRED_LOCAL_MEMORY_GIB: u64 = 96;

const CALLBACK_PROBE: &str = r#"set -eu
host="$1"
port="$2"
case "$host" in
    *:*) url="http://[$host]:$port/muser-preflight" ;;
    *) url="http://$host:$port/muser-preflight" ;;
esac
curl --noproxy '*' --fail --silent --show-error --connect-timeout 3 --max-time 5 "$url" >/dev/null
printf 'callback-ok\n'
"#;

const PROBE: &str = r#"set -u
printf 'home=%s\n' "$HOME"
printf 'arch=%s\n' "$(uname -m)"
printf 'kernel=%s\n' "$(uname -sr)"
printf 'ssh_client=%s\n' "${SSH_CLIENT%% *}"
printf 'unix_seconds=%s\n' "$(date +%s)"
printf 'uid=%s\n' "$(id -u)"
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
    if docker info >/dev/null 2>&1; then
        printf 'docker_daemon=1\n'
        docker_root=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null)
        printf 'docker_root=%s\n' "$docker_root"
        printf 'docker_device=%s\n' "$(df -Pk "$docker_root" 2>/dev/null | awk 'NR==2 {print $1}')"
        printf 'docker_disk_kib=%s\n' "$(df -Pk "$docker_root" 2>/dev/null | awk 'NR==2 {print $4}')"
    else
        printf 'docker_daemon=0\n'
    fi
else
    printf 'docker=\n'
    printf 'docker_daemon=0\n'
fi
if command -v curl >/dev/null 2>&1 && command -v sha256sum >/dev/null 2>&1 && command -v zstd >/dev/null 2>&1; then
    printf 'artifact_tools=1\n'
else
    printf 'artifact_tools=0\n'
fi
if command -v systemctl >/dev/null 2>&1; then
    printf 'systemctl=1\n'
    if systemctl --user show-environment >/dev/null 2>&1; then printf 'user_systemd=1\n'; else printf 'user_systemd=0\n'; fi
else
    printf 'systemctl=0\n'
    printf 'user_systemd=0\n'
fi
if command -v loginctl >/dev/null 2>&1; then
    printf 'linger=%s\n' "$(loginctl show-user "$(id -un)" -p Linger --value 2>/dev/null || printf unknown)"
else
    printf 'linger=unknown\n'
fi
printf 'home_device=%s\n' "$(df -Pk "$HOME" | awk 'NR==2 {print $1}')"
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
    pub unix_seconds: u64,
    pub uid: u64,
    pub nvidia_smi: String,
    pub driver: String,
    pub gpu: String,
    pub compute_cap: String,
    pub docker: String,
    pub docker_daemon: bool,
    pub docker_root: String,
    pub docker_device: String,
    pub docker_disk_kib: u64,
    pub artifact_tools: bool,
    pub systemctl: bool,
    pub user_systemd: bool,
    pub linger: String,
    pub home_device: String,
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
                    "require an Apple Silicon Mac with {REQUIRED_LOCAL_MEMORY_GIB} GiB memory and \
                     {REQUIRED_LOCAL_DISK_GIB} GiB free for first-time native artifact assembly"
                ),
            );
        }
        ctx.progress
            .plan(Step::Preflight, "finish without probing the node");
        return Ok(());
    }

    ssh.require_direct_data_path()?;
    check_local_platform(ctx)?;
    check_local_artifact_capacity(ctx, entry)?;

    let probes = parse(&ssh.run(PROBE, &[])?);
    ctx.progress.emit_data(
        Step::Preflight,
        Status::Info,
        &format!("{} runs {} ({})", entry.name, probes.arch, probes.kernel),
        serde_json::json!({ "arch": probes.arch, "kernel": probes.kernel }),
    );
    let local_unix_seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|error| format!("read Mac wall clock: {error}"))?
        .as_secs();
    let clock_skew_seconds = local_unix_seconds.abs_diff(probes.unix_seconds);
    if probes.unix_seconds == 0 || clock_skew_seconds > 5 {
        return Err(format!(
            "Mac and GX10 clocks differ by {clock_skew_seconds}s; mTLS certificate validity and control deadlines require synchronized clocks — enable network time on both machines"
        ));
    }
    ctx.progress.emit_data(
        Step::Preflight,
        Status::Info,
        &format!("Mac/GX10 clock skew is within {clock_skew_seconds}s"),
        serde_json::json!({ "clock_skew_seconds": clock_skew_seconds }),
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
    validate_callback_route(&ssh, &probes.ssh_client)?;
    ctx.progress.emit_data(
        Step::Preflight,
        Status::Info,
        &format!(
            "GX10 can call this Mac back at {}:{RECEIVER_PORT}",
            probes.ssh_client
        ),
        serde_json::json!({
            "advertised_receiver_host": probes.ssh_client,
            "receiver_port": RECEIVER_PORT,
        }),
    );
    if !probes.systemctl {
        return Err(
            "systemd is required for the supervised producer service; the advertised tmux fallback is not a persistent release lifecycle"
                .into(),
        );
    }
    if probes.uid != 0 && probes.linger == "yes" && !probes.user_systemd {
        return Err(
            "systemd lingering is enabled but the user manager is unavailable over SSH; check the node's PAM/systemd user session before onboarding"
                .into(),
        );
    }
    if probes.uid != 0 && probes.linger != "yes" {
        ctx.progress.emit_data(
            Step::Preflight,
            Status::Info,
            "WARN: systemd user lingering is disabled; the daemon stage will enable it with passwordless sudo or stop with the exact one-time command",
            serde_json::json!({
                "user_systemd": probes.user_systemd,
                "linger": probes.linger,
            }),
        );
    }
    ctx.progress.emit_data(
        Step::Preflight,
        Status::Info,
        &format!("docker at {} with a reachable daemon", probes.docker),
        serde_json::json!({ "docker": probes.docker, "systemctl": probes.systemctl }),
    );

    let disk_gib = probes.disk_kib / (1024 * 1024);
    let split_docker_storage = !probes.home_device.is_empty()
        && !probes.docker_device.is_empty()
        && probes.home_device != probes.docker_device;
    let required_home_gib = if split_docker_storage {
        REQUIRED_SPLIT_HOME_DISK_GIB
    } else {
        REQUIRED_DISK_GIB
    };
    if disk_gib < required_home_gib {
        return Err(format!(
            "{}'s filesystem has {disk_gib} GiB free; the pinned checkpoint and image archive need {required_home_gib} GiB",
            probes.home
        ));
    }
    let docker_disk_gib = probes.docker_disk_kib / (1024 * 1024);
    if probes.docker_root.is_empty()
        || probes.docker_device.is_empty()
        || probes.docker_disk_kib == 0
    {
        return Err(
            "docker is reachable but its storage root/free space could not be inspected".into(),
        );
    }
    if split_docker_storage && docker_disk_gib < REQUIRED_SPLIT_DOCKER_DISK_GIB {
        return Err(format!(
            "Docker storage at {} has {docker_disk_gib} GiB free; loading the pinned producer image needs {REQUIRED_SPLIT_DOCKER_DISK_GIB} GiB",
            probes.docker_root
        ));
    }
    ctx.progress.emit_data(
        Step::Preflight,
        Status::Info,
        &if split_docker_storage {
            format!(
                "{disk_gib} GiB free under {}; {docker_disk_gib} GiB in Docker storage",
                probes.home
            )
        } else {
            format!("{disk_gib} GiB free for checkpoint and Docker storage")
        },
        serde_json::json!({
            "disk_gib": disk_gib,
            "required_gib": required_home_gib,
            "docker_root": probes.docker_root,
            "docker_disk_gib": docker_disk_gib,
            "docker_required_gib": if split_docker_storage {
                REQUIRED_SPLIT_DOCKER_DISK_GIB
            } else {
                REQUIRED_DISK_GIB
            },
            "split_docker_storage": split_docker_storage,
        }),
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

fn check_local_platform(ctx: &Ctx) -> Result<()> {
    if std::env::consts::OS != "macos" || std::env::consts::ARCH != "aarch64" {
        return Err(format!(
            "the released decoder requires Apple Silicon macOS; this client is {} {}",
            std::env::consts::OS,
            std::env::consts::ARCH
        ));
    }
    let output = std::process::Command::new("/usr/sbin/sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .map_err(|error| format!("inspect Mac unified memory: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "sysctl could not inspect Mac unified memory: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let bytes = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<u64>()
        .map_err(|_| "sysctl returned no numeric Mac memory size".to_string())?;
    let memory_gib = validate_local_memory(bytes)?;
    ctx.progress.emit_data(
        Step::Preflight,
        Status::Info,
        &format!("Apple Silicon Mac with {memory_gib} GiB unified memory"),
        serde_json::json!({
            "local_arch": std::env::consts::ARCH,
            "local_memory_gib": memory_gib,
            "local_required_memory_gib": REQUIRED_LOCAL_MEMORY_GIB,
        }),
    );
    Ok(())
}

fn validate_local_memory(bytes: u64) -> Result<u64> {
    let memory_gib = bytes / (1024 * 1024 * 1024);
    if memory_gib < REQUIRED_LOCAL_MEMORY_GIB {
        return Err(format!(
            "this Mac has {memory_gib} GiB unified memory; the released four-slot 131K decoder \
             requires {REQUIRED_LOCAL_MEMORY_GIB} GiB"
        ));
    }
    Ok(memory_gib)
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
    let existing = free_space_probe_path(&model_dir)?;
    let output = std::process::Command::new("df")
        .args(["-Pk"])
        .arg(&existing)
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

/// Resolve the nearest existing model-directory ancestor before asking `df`.
/// A common Mac layout keeps `~/.muser` on the internal disk and makes only
/// `~/.muser/models` a symlink to a larger volume. Probing the symlink inode
/// reports the internal disk and can falsely reject hundreds of GiB that are
/// available where the artifact will actually be assembled.
fn free_space_probe_path(model_dir: &std::path::Path) -> Result<std::path::PathBuf> {
    let existing = model_dir
        .ancestors()
        .find(|path| path.is_dir())
        .ok_or_else(|| {
            format!(
                "no existing parent for model directory {}",
                model_dir.display()
            )
        })?;
    existing.canonicalize().map_err(|error| {
        format!(
            "resolve model storage under {} before checking free space: {error}",
            existing.display()
        )
    })
}

/// The Mac's address as the node sees it — what the node dials back on.
/// Taken from `$SSH_CLIENT`, so it is the route that already works rather
/// than a guess from a local interface list.
pub fn advertised_receiver_host(ssh: &Ssh) -> Result<String> {
    let probes = parse(&ssh.run(PROBE, &[])?);
    validate_callback_address(&probes.ssh_client)
}

/// Before downloads or the cold producer initialization, prove
/// that the reverse half of the topology is routable and that the Mac
/// firewall permits the receiver port. The actual receiver later replaces
/// this tiny one-shot HTTP listener on the same wildcard address and port.
fn validate_callback_route(ssh: &Ssh, host: &str) -> Result<()> {
    use std::io::Write as _;
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, TcpListener};
    use std::thread;
    use std::time::{Duration, Instant};

    let host = validate_callback_address(host)?;
    let ip: IpAddr = host
        .parse()
        .map_err(|_| "the SSH client address is not a numeric IP address".to_string())?;
    let bind = SocketAddr::new(
        match ip {
            IpAddr::V4(_) => IpAddr::V4(Ipv4Addr::UNSPECIFIED),
            IpAddr::V6(_) => IpAddr::V6(Ipv6Addr::UNSPECIFIED),
        },
        RECEIVER_PORT,
    );
    let listener = TcpListener::bind(bind).map_err(|error| {
        format!(
            "cannot open the Mac receiver probe on {bind}: {error} — stop any existing `muser up` process and allow incoming connections for Muser"
        )
    })?;
    listener
        .set_nonblocking(true)
        .map_err(|error| format!("make receiver callback probe nonblocking: {error}"))?;
    let accepting = thread::spawn(move || -> Result<()> {
        let deadline = Instant::now() + Duration::from_secs(8);
        loop {
            match listener.accept() {
                Ok((mut stream, _)) => {
                    stream
                        .write_all(
                            b"HTTP/1.1 204 No Content\r\nConnection: close\r\nContent-Length: 0\r\n\r\n",
                        )
                        .map_err(|error| format!("answer receiver callback probe: {error}"))?;
                    return Ok(());
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    if Instant::now() >= deadline {
                        return Err("the GX10 did not reach the Mac receiver port within 8s".into());
                    }
                    thread::sleep(Duration::from_millis(25));
                }
                Err(error) => return Err(format!("accept receiver callback probe: {error}")),
            }
        }
    });
    let port = RECEIVER_PORT.to_string();
    let remote = ssh.run(CALLBACK_PROBE, &[&host, &port]);
    let accepted = accepting
        .join()
        .map_err(|_| "receiver callback probe thread panicked".to_string())?;
    remote?;
    accepted?;
    Ok(())
}

fn validate_callback_address(value: &str) -> Result<String> {
    let ip = value.parse::<std::net::IpAddr>().map_err(|_| {
        "the node reported no valid numeric $SSH_CLIENT address for this Mac".to_string()
    })?;
    if ip.is_unspecified() || ip.is_loopback() || ip.is_multicast() {
        return Err(format!(
            "the node reported unusable $SSH_CLIENT address {ip} for this Mac"
        ));
    }
    Ok(ip.to_string())
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
            "unix_seconds" => probes.unix_seconds = value.parse().unwrap_or(0),
            "uid" => probes.uid = value.parse().unwrap_or(u64::MAX),
            "nvidia_smi" => probes.nvidia_smi = value.to_string(),
            "driver" => probes.driver = value.to_string(),
            "gpu" => probes.gpu = value.to_string(),
            "compute_cap" => probes.compute_cap = value.to_string(),
            "docker" => probes.docker = value.to_string(),
            "docker_daemon" => probes.docker_daemon = value == "1",
            "docker_root" => probes.docker_root = value.to_string(),
            "docker_device" => probes.docker_device = value.to_string(),
            "docker_disk_kib" => probes.docker_disk_kib = value.parse().unwrap_or(0),
            "artifact_tools" => probes.artifact_tools = value == "1",
            "systemctl" => probes.systemctl = value == "1",
            "user_systemd" => probes.user_systemd = value == "1",
            "linger" => probes.linger = value.to_string(),
            "home_device" => probes.home_device = value.to_string(),
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
             docker_root=/var/lib/docker\ndocker_device=/dev/nvme0n1p2\ndocker_disk_kib=104857600\n\
             home_device=/dev/nvme0n1p2\ndisk_kib=104857600\nmem_total_kib=134217728\nmem_kib=33554432\n\
             systemctl=1\nuser_systemd=1\nlinger=yes\nuid=1000\nssh_client=10.0.0.2\nunix_seconds=1787961600\n",
        );
        assert_eq!(probes.home, "/home/muser");
        assert_eq!(probes.arch, "aarch64");
        assert!(probes.docker_daemon);
        assert_eq!(probes.docker_root, "/var/lib/docker");
        assert_eq!(probes.docker_device, "/dev/nvme0n1p2");
        assert!(probes.systemctl);
        assert!(probes.user_systemd);
        assert_eq!(probes.linger, "yes");
        assert_eq!(probes.uid, 1000);
        assert_eq!(probes.disk_kib / (1024 * 1024), 100);
        assert_eq!(probes.docker_disk_kib / (1024 * 1024), 100);
        assert_eq!(probes.home_device, "/dev/nvme0n1p2");
        assert_eq!(probes.ssh_client, "10.0.0.2");
        assert_eq!(probes.unix_seconds, 1_787_961_600);
    }

    #[test]
    fn a_missing_probe_is_absent_rather_than_wrong() {
        let probes = parse("arch=x86_64\nnvidia_smi=\ndocker=\n");
        assert!(probes.nvidia_smi.is_empty());
        assert!(probes.docker.is_empty());
        assert_eq!(probes.disk_kib, 0);
    }

    #[test]
    fn callback_addresses_are_numeric_and_nonlocal() {
        assert_eq!(
            validate_callback_address("192.0.2.113").unwrap(),
            "192.0.2.113"
        );
        assert_eq!(
            validate_callback_address("2001:db8::2").unwrap(),
            "2001:db8::2"
        );
        assert!(validate_callback_address("").is_err());
        assert!(validate_callback_address("localhost").is_err());
        assert!(validate_callback_address("127.0.0.1").is_err());
        assert!(validate_callback_address("::").is_err());
    }

    #[test]
    fn local_memory_floor_fails_before_onboarding() {
        assert_eq!(validate_local_memory(96 * 1024 * 1024 * 1024).unwrap(), 96);
        let error = validate_local_memory(64 * 1024 * 1024 * 1024).unwrap_err();
        assert!(error.contains("four-slot 131K decoder"));
        assert!(error.contains("requires 96 GiB"));
    }

    #[cfg(unix)]
    #[test]
    fn free_space_probe_follows_a_models_directory_symlink() {
        use std::os::unix::fs::symlink;

        let root = std::env::temp_dir().join(format!(
            "muser-preflight-model-link-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let target = root.join("external-models");
        let link = root.join("home").join("models");
        std::fs::create_dir_all(&target).unwrap();
        std::fs::create_dir_all(link.parent().unwrap()).unwrap();
        symlink(&target, &link).unwrap();

        assert_eq!(
            free_space_probe_path(&link).unwrap(),
            target.canonicalize().unwrap()
        );

        std::fs::remove_dir_all(&root).unwrap();
    }
}
