//! Step 5 — the resident producer.
//!
//! `bootstrap_node.sh daemon` owns the supervised systemd install; this step
//! hands it the paths from the enrolment and then believes only the port. A
//! daemon that "started"
//! but never listened has not started.

use std::time::{Duration, Instant};

use super::progress::{Status, Step};
use super::registry::{
    create_private_dir, node_dir, NodeEntry, ProducerKind, DAEMON_PORT, STATE_ENROLLED,
};
use super::{Ctx, Result};

/// How long the producer gets to hold its GPU lease, start its warm
/// exporter container, and bind.
const PORT_WAIT: Duration = Duration::from_secs(60);
const TIMEOUT_SECONDS: u64 = 900;
const PROBE_TIMEOUT: Duration = Duration::from_secs(1);
const POLL_INTERVAL: Duration = Duration::from_secs(2);

// The handoff config is not passed: `bootstrap_node.sh daemon` reads
// `LANE/handoff.json`, which the enrol step wrote, and takes its listen
// host/port from there.
const DRIVE: &str = r#"set -eu
bash "$1/bootstrap_node.sh" daemon --lane "$1" --model "$2" --dflash "$3"
"#;

const DRIVE_NATIVE: &str = r#"set -eu
bash "$1/bootstrap_node.sh" --json daemon --native --lane "$1" --checkpoint "$2"
"#;

pub fn run(ctx: &Ctx, entry: &mut NodeEntry) -> Result<()> {
    if entry.producer_kind() == ProducerKind::Native {
        return run_native(ctx, entry);
    }
    let ssh = ctx.ssh(entry)?;
    let lane = entry.lane_dir.clone();
    let release = ctx.release()?;
    let model = format!("{lane}/models/{}", release.target.filename);
    let dflash = format!("{lane}/models/{}", release.dflash.filename);

    ctx.progress.emit(
        Step::Daemon,
        Status::Start,
        "installing and starting the producer daemon",
    );

    if ctx.dry_run {
        ctx.progress.plan_command(
            Step::Daemon,
            &format!("install and start the persistent systemd daemon from {lane}"),
            &ssh.argv(&[&lane, &model, &dflash]),
        );
        ctx.progress.plan(
            Step::Daemon,
            &format!(
                "wait up to {}s for {}:{DAEMON_PORT} to accept a TCP connection",
                PORT_WAIT.as_secs(),
                entry.host
            ),
        );
        ctx.progress
            .plan(Step::Daemon, "finish without starting a daemon");
        return Ok(());
    }

    let relay = |line: &str| ctx.progress.emit(Step::Daemon, Status::Info, line);
    ssh.run_relayed(DRIVE, &[&lane, &model, &dflash], &relay)?;
    ctx.progress.emit(
        Step::Daemon,
        Status::Info,
        &format!(
            "bootstrap reported the daemon installed; waiting for {}:{DAEMON_PORT}",
            entry.host
        ),
    );

    let deadline = Instant::now() + PORT_WAIT;
    let mut last = format!("{}:{DAEMON_PORT} was never probed", entry.host);
    loop {
        match ssh.tcp_probe(DAEMON_PORT, PROBE_TIMEOUT) {
            Ok(elapsed) => {
                ctx.progress.emit_data(
                    Step::Daemon,
                    Status::Ok,
                    &format!(
                        "{}:{DAEMON_PORT} is listening ({} ms to connect)",
                        entry.host,
                        millis(elapsed)
                    ),
                    serde_json::json!({ "port": DAEMON_PORT, "connect_ms": millis(elapsed) }),
                );
                // Listening proves only the enrolled transport is available;
                // `healthy` remains owned by a real authenticated handoff,
                // bounded Metal decode, and the durable registry commit.
                entry.touch(STATE_ENROLLED);
                entry.last_error = None;
                return Ok(());
            }
            Err(error) => last = error,
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "{}:{DAEMON_PORT} did not listen within {}s — {last}",
                entry.host,
                PORT_WAIT.as_secs()
            ));
        }
        std::thread::sleep(POLL_INTERVAL);
    }
}

#[derive(Debug, PartialEq)]
struct BootstrapProgress {
    detail: String,
    data: Option<serde_json::Value>,
}

fn sanitized_native_startup(value: &serde_json::Value) -> Option<serde_json::Value> {
    let startup = value.get("data")?.get("native_startup")?;
    let phase = startup.get("phase")?.as_str()?;
    let completed = startup.get("completed")?.as_u64()?;
    let total = startup.get("total")?.as_u64()?;
    let elapsed_seconds = startup.get("elapsed_seconds")?.as_u64()?;
    let expected_completed = match phase {
        "engine-setup" => 0,
        "weights" => 1,
        "batch-profile" => 2,
        "kv-warmup" => 3,
        "request-warmup" => 4,
        "ready" => 5,
        _ => return None,
    };
    if completed != expected_completed || total != 5 || elapsed_seconds > TIMEOUT_SECONDS {
        return None;
    }
    // Rebuild the object rather than relaying the remote value. This is the
    // complete public vocabulary; arbitrary container fields never cross the
    // SSH boundary into the browser's progress transcript.
    Some(serde_json::json!({
        "native_startup": {
            "phase": phase,
            "completed": completed,
            "total": total,
            "elapsed_seconds": elapsed_seconds,
        }
    }))
}

fn bootstrap_progress(line: &str) -> Option<BootstrapProgress> {
    let value = serde_json::from_str::<serde_json::Value>(line).ok()?;
    if value.get("schema").and_then(serde_json::Value::as_str)
        != Some(super::progress::PROGRESS_SCHEMA)
        || value.get("step").and_then(serde_json::Value::as_str) != Some("daemon")
    {
        return None;
    }
    let detail = value
        .get("detail")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)?;
    let data = sanitized_native_startup(&value);
    Some(BootstrapProgress { detail, data })
}

fn run_native(ctx: &Ctx, entry: &mut NodeEntry) -> Result<()> {
    let ssh = ctx.ssh(entry)?;
    let lane = entry.lane_dir.clone();
    let identity = ctx.native_identity()?;
    let checkpoint = format!("{lane}/models/{}", identity.checkpoint_directory);

    ctx.progress.emit(
        Step::Daemon,
        Status::Start,
        "installing and starting the native NVFP4 producer daemon",
    );
    if ctx.dry_run {
        ctx.progress.plan_command(
            Step::Daemon,
            &format!("start exact image {} from {checkpoint}", identity.image_id),
            &ssh.argv(&[&lane, &checkpoint]),
        );
        ctx.progress.plan(
            Step::Daemon,
            &format!(
                "wait up to {}s for authenticated control on {}:{DAEMON_PORT}",
                TIMEOUT_SECONDS, entry.host
            ),
        );
        ctx.progress
            .plan(Step::Daemon, "finish without starting a daemon");
        return Ok(());
    }

    let relay = |line: &str| {
        if let Some(progress) = bootstrap_progress(line) {
            if let Some(data) = progress.data {
                ctx.progress
                    .emit_data(Step::Daemon, Status::Info, &progress.detail, data);
            } else {
                ctx.progress
                    .emit(Step::Daemon, Status::Info, &progress.detail);
            }
        }
    };
    ssh.run_relayed(DRIVE_NATIVE, &[&lane, &checkpoint], &relay)?;
    let deadline = Instant::now() + Duration::from_secs(TIMEOUT_SECONDS);
    let mut last = format!("{}:{DAEMON_PORT} was never probed", entry.host);
    loop {
        match ssh.tcp_probe(DAEMON_PORT, PROBE_TIMEOUT) {
            Ok(elapsed) => {
                let local = node_dir(&ctx.muser_home, &entry.name);
                create_private_dir(&local)?;
                let local_rope = local.join("native-rope-cache-f32le.bin");
                ssh.scp_from(
                    &format!("{lane}/work/native-rope-cache-f32le.bin"),
                    &local_rope,
                )?;
                super::model::verify_regular_file(
                    &local_rope,
                    identity.rope_cache_bytes,
                    &identity.rope_cache_sha256,
                    "native RoPE cache",
                )?;
                ctx.progress.emit_data(
                    Step::Daemon,
                    Status::Ok,
                    &format!(
                        "native producer control is listening ({} ms); RoPE cache retained",
                        millis(elapsed)
                    ),
                    serde_json::json!({
                        "port": DAEMON_PORT,
                        "connect_ms": millis(elapsed),
                        "container_image": identity.image_id,
                        "rope_cache": local_rope,
                    }),
                );
                entry.touch(STATE_ENROLLED);
                entry.last_error = None;
                return Ok(());
            }
            Err(error) => last = error,
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "native producer did not listen within {TIMEOUT_SECONDS}s — {last}"
            ));
        }
        std::thread::sleep(POLL_INTERVAL);
    }
}

pub fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_bootstrap_progress_is_sanitized_without_relaying_arbitrary_output() {
        let line = serde_json::json!({
            "schema": super::super::progress::PROGRESS_SCHEMA,
            "step": "daemon",
            "status": "info",
            "detail": "weights loaded; initializing the 8K chunked-prefill scheduler (99s)",
            "data": {
                "native_startup": {
                    "phase": "batch-profile",
                    "completed": 2,
                    "total": 5,
                    "elapsed_seconds": 99,
                    "remote_only": "must not cross",
                },
                "arbitrary": "must not cross",
            },
        })
        .to_string();
        let progress = bootstrap_progress(&line).expect("sanitized bootstrap event");
        assert_eq!(
            progress.detail,
            "weights loaded; initializing the 8K chunked-prefill scheduler (99s)"
        );
        assert_eq!(
            progress.data,
            Some(serde_json::json!({
                "native_startup": {
                    "phase": "batch-profile",
                    "completed": 2,
                    "total": 5,
                    "elapsed_seconds": 99,
                }
            }))
        );
        assert!(bootstrap_progress("raw detached-container log").is_none());
        assert!(bootstrap_progress(
            r#"{"schema":"muser.node-progress.v2","step":"model","detail":"wrong step"}"#
        )
        .is_none());
    }

    #[test]
    fn malformed_native_startup_data_is_not_relayed() {
        let line = serde_json::json!({
            "schema": super::super::progress::PROGRESS_SCHEMA,
            "step": "daemon",
            "status": "info",
            "detail": "safe milestone text",
            "data": {
                "native_startup": {
                    "phase": "made-up-phase",
                    "completed": 2,
                    "total": 5,
                    "elapsed_seconds": 9,
                }
            },
        })
        .to_string();
        let progress = bootstrap_progress(&line).expect("detail remains usable");
        assert_eq!(progress.detail, "safe milestone text");
        assert_eq!(progress.data, None);
    }
}
