//! Step 6 — a real prefill, and what the link actually did.
//!
//! The smoke test is not a ping: `muser-remote-qualify` runs the *production*
//! receiver against the node, recomputes the same prefix locally, and
//! refuses to print a passing sample unless the remote and local token
//! streams are identical. Exactness is by construction, so this step only
//! has to establish that a sample was produced and read its numbers.
//!
//! It runs under `scripts/accelerator_safe.py`, which owns the machine-wide
//! GPU lease. A node onboarding therefore queues behind whatever else holds
//! the accelerator rather than colliding with it.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use sha2::{Digest as _, Sha256};

use super::artifacts::NativeQualification;
use super::daemon::millis;
use super::progress::{Status, Step};
use super::registry::{node_dir, write_private, NodeEntry, QualificationRecipe, DAEMON_PORT};
use super::{Ctx, Result};

/// The prefill this step proves, in positions.
pub const SMOKE_PROMPT_TOKENS: usize = 2048;
/// Decoded after the install, enough to compare a real token stream.
// The qualify executor enforces its blessed protocol (exactly 3 repetitions,
// 256 output tokens); the smoke runs that protocol rather than a lighter one
// so the button's final step is literally a production qualification cell.
const SMOKE_OUTPUT_TOKENS: usize = 256;
const SMOKE_REPETITIONS: usize = 3;
/// Hard minimum median effective installed-payload throughput.
const EXPECTED_GBPS: f64 = 3.0;
const RTT_SAMPLES: usize = 5;
const PROBE_TIMEOUT: Duration = Duration::from_secs(1);

/// The qualification variants this gate runs, in `muser-remote-qualify`'s
/// `--variant` grammar.
const VARIANT_TARGET_PLUS_DFLASH: &str = "target-plus-dflash";
const VARIANT_TEXT: &str = "text";

fn recipe_progress(recipe: QualificationRecipe) -> String {
    match recipe {
        QualificationRecipe::KquantTargetPlusDflash => format!(
            "running recipe {}: three ordered {SMOKE_PROMPT_TOKENS}/{SMOKE_OUTPUT_TOKENS} token, full-logit, and DFlash exact handoffs",
            recipe.public_name()
        ),
        QualificationRecipe::NativeText => format!(
            "running recipe {}: three ordered {SMOKE_PROMPT_TOKENS}/{SMOKE_OUTPUT_TOKENS} exact-token handoffs with bounded full-logit drift",
            recipe.public_name()
        ),
    }
}

pub fn run(ctx: &Ctx, entry: &mut NodeEntry) -> Result<()> {
    let ssh = ctx.ssh(entry)?;
    let local = node_dir(&ctx.muser_home, &entry.name);
    let cluster = super::enroll::cluster_config(&ctx.muser_home, &entry.name);
    let fixture = ctx
        .prompt_fixture
        .clone()
        .unwrap_or_else(|| local.join("smoke-2048.tokens"));
    let model_dir = ctx.model_dir()?;
    // The identity is recorded verbatim in the qualification receipt, so it
    // stays free of separators an evidence path would have to escape.
    let identity = format!(
        "node-{}-{}",
        entry.name,
        crate::timefmt::now_rfc3339().replace([':', '-'], "")
    );
    let producer = entry.producer_kind();
    let recipe = producer.qualification_recipe();
    let native_identity = (producer == super::registry::ProducerKind::Native)
        .then(|| ctx.native_identity())
        .transpose()?;
    let release = if native_identity.is_none() {
        Some(ctx.release()?)
    } else {
        None
    };
    let model = match &native_identity {
        Some(identity) => model_dir.join(&identity.consumer.filename),
        None => model_dir.join(&release.as_ref().expect("combined release").target.filename),
    };
    let variant = recipe.variant();
    // The llama.cpp lane ships combined target+DFlash transfers and
    // verification needs the DFlash weights; the native lane is plain
    // decode only, so the gate must not ask for a DFlash artifact.
    let dflash = match recipe {
        QualificationRecipe::KquantTargetPlusDflash => {
            let dflash = model_dir.join(
                &release
                    .as_ref()
                    .expect("combined recipe has a release")
                    .dflash
                    .filename,
            );
            if !dflash.is_file() {
                return Err(format!(
                    "{} is absent; the enrolled lane ships combined target+DFlash transfers and verification needs the DFlash weights",
                    dflash.display()
                ));
            }
            Some(dflash)
        }
        QualificationRecipe::NativeText => None,
    };
    let argv = qualify_argv(
        ctx,
        entry,
        &model,
        dflash.as_deref(),
        &fixture,
        &cluster,
        &local,
        &identity,
        variant,
    );

    ctx.progress.emit(
        Step::Netqual,
        Status::Start,
        &format!("measuring the link to {}", entry.host),
    );
    ctx.progress
        .emit(Step::Smoke, Status::Start, &recipe_progress(recipe));

    if ctx.dry_run {
        ctx.progress.plan(
            Step::Netqual,
            &format!(
                "time {RTT_SAMPLES} TCP connects to {}:{DAEMON_PORT} and take the median",
                entry.host
            ),
        );
        ctx.progress.plan(
            Step::Smoke,
            &format!(
                "write a deterministic {SMOKE_PROMPT_TOKENS}-position fixture to {}",
                fixture.display()
            ),
        );
        ctx.progress.plan_command(
            Step::Smoke,
            &format!("qualify {} against {}", entry.name, cluster.display()),
            &argv,
        );
        ctx.progress.plan(
            Step::Netqual,
            &format!(
                "derive Gb/s from the sample's installed bytes and authenticated payload-wire time, refuse under {EXPECTED_GBPS:.1} Gbps"
            ),
        );
        ctx.progress.plan(
            Step::Smoke,
            &format!(
                "durably persist state={} for {} only after qualification passes",
                super::registry::STATE_HEALTHY,
                entry.name
            ),
        );
        ctx.progress.plan(
            Step::Smoke,
            "bind the exact receipted llama.cpp metallib into the local qualifier",
        );
        return Ok(());
    }

    ensure_metal_qualifier(ctx)?;

    let mut rtts = Vec::with_capacity(RTT_SAMPLES);
    for _ in 0..RTT_SAMPLES {
        rtts.push(millis(ssh.tcp_probe(DAEMON_PORT, PROBE_TIMEOUT)?));
    }
    rtts.sort_by(f64::total_cmp);
    let rtt_ms = rtts[rtts.len() / 2];
    entry.netqual_rtt_ms = Some(rtt_ms);
    ctx.progress.emit_data(
        Step::Netqual,
        Status::Info,
        &format!("median TCP RTT {rtt_ms:.2} ms over {RTT_SAMPLES} connects"),
        serde_json::json!({ "netqual_rtt_ms": rtt_ms, "samples": rtts }),
    );

    if ctx.prompt_fixture.is_none() && ensure_default_fixture(&fixture, recipe)? {
        ctx.progress.emit(
            Step::Smoke,
            Status::Info,
            &format!(
                "wrote the {} {SMOKE_PROMPT_TOKENS}-position fixture to {}",
                recipe.public_name(),
                fixture.display()
            ),
        );
    }
    if !model.is_file() {
        return Err(format!(
            "{} is absent; the smoke prefill recomputes the same prefix locally and needs the target weights",
            model.display()
        ));
    }
    if let Some(identity) = &native_identity {
        let bytes = std::fs::read(&fixture)
            .map_err(|error| format!("read native qualification fixture: {error}"))?;
        let digest = format!("{:x}", Sha256::digest(&bytes));
        if digest != identity.qualification.prompt_sha256 {
            return Err(format!(
                "native qualification fixture SHA-256 mismatch: expected {}, got {digest}",
                identity.qualification.prompt_sha256
            ));
        }
    }
    let metallib = ctx.pinned_metallib()?;

    // One-button semantics: a busy machine is waited out, not handed to the
    // user as an errand. ~2 minutes of patience covers a dashboard's model
    // server being stopped or a short benchmark cell draining.
    const BUSY_RETRIES: usize = 12;
    const BUSY_WAIT: std::time::Duration = std::time::Duration::from_secs(10);
    let mut output = None;
    for attempt in 0..=BUSY_RETRIES {
        let receipt_path = local
            .join("smoke")
            .join(format!("{identity}-attempt-{attempt}.result.json"));
        let attempt_argv = with_result_receipt(&argv, &receipt_path)?;
        let mut command = Command::new(&attempt_argv[0]);
        command
            .args(&attempt_argv[1..])
            .env("MUSER_GGML_METALLIB", &metallib)
            .env("MUSER_CROSS_VENDOR_QK", "1");
        if native_identity.is_some() {
            command.env(
                "MUSER_CROSS_VENDOR_ROPE_CACHE",
                local.join("native-rope-cache-f32le.bin"),
            );
        }
        let candidate = command
            .output()
            .map_err(|error| format!("spawn {}: {error}", attempt_argv[0]))?;
        let retained = read_accelerator_result(&receipt_path, &attempt_argv)?;
        let command_log = std::fs::read_to_string(&retained.command_log)
            .map_err(|error| format!("read {}: {error}", retained.command_log.display()))?;
        let busy = retained.exit_status != 0 && busy_class(&command_log).is_some();
        if busy && attempt < BUSY_RETRIES {
            ctx.progress.emit(
                Step::Smoke,
                Status::Info,
                &format!(
                    "the GPU is busy; waiting {}s and retrying ({}/{BUSY_RETRIES})",
                    BUSY_WAIT.as_secs(),
                    attempt + 1
                ),
            );
            std::thread::sleep(BUSY_WAIT);
            continue;
        }
        output = Some((candidate, retained, command_log));
        break;
    }
    let (output, retained, command_log) = output.expect("loop always assigns before break");
    if output.status.code() != Some(retained.exit_status) {
        return Err("accelerator wrapper exit differs from its retained receipt".into());
    }
    if retained.exit_status != 0 {
        let detail = command_log
            .lines()
            .rev()
            .take(3)
            .collect::<Vec<_>>()
            .join(" | ");
        return Err(format!("remote qualification failed: {detail}"));
    }
    let samples = parse_samples(
        &command_log,
        &identity,
        recipe,
        native_identity.as_ref().map(|value| &value.qualification),
    )?;
    let sample = &samples[0];
    ctx.progress.emit_data(
        Step::Smoke,
        Status::Ok,
        &format!(
            "exact tokens over {} installed bytes in {:.1} ms",
            sample.installed_bytes,
            sample.transfer_commit_ns as f64 / 1e6
        ),
        serde_json::json!({
            "installed_bytes": sample.installed_bytes,
            "installed_segments": sample.installed_segments,
            "transfer_commit_ns": sample.transfer_commit_ns,
            "exact_tokens": true,
        }),
    );

    // `receiver_transfer_commit_ns` includes producer prefill/export and is
    // the right end-to-end latency receipt, but it is not a link duration.
    // The producer's authenticated TCP_INFO busy-time is the payload-wire
    // clock used by the qualifier's unchanged 3 Gbps gate.
    let mut throughputs = samples
        .iter()
        .map(installed_payload_gbps)
        .collect::<Vec<_>>();
    throughputs.sort_by(f64::total_cmp);
    let gbps = throughputs[throughputs.len() / 2];
    if gbps < EXPECTED_GBPS {
        return Err(format!(
            "median installed-payload throughput {gbps:.2} Gbps is below the required {EXPECTED_GBPS:.1} Gbps minimum"
        ));
    }
    ctx.progress.emit_data(
        Step::Netqual,
        Status::Ok,
        &format!("{gbps:.2} Gbps, {rtt_ms:.2} ms median RTT"),
        serde_json::json!({ "netqual_gbps": gbps, "netqual_rtt_ms": rtt_ms }),
    );

    entry.netqual_gbps = Some(gbps);
    entry.netqual_rtt_ms = Some(rtt_ms);
    Ok(())
}

struct SmokeSample {
    repetition: u64,
    installed_bytes: u64,
    installed_segments: u64,
    transfer_commit_ns: u64,
    payload_wire_ns: u64,
}

/// Bytes x 8 bits over nanoseconds is already gigabits per second.
fn installed_payload_gbps(sample: &SmokeSample) -> f64 {
    sample.installed_bytes as f64 * 8.0 / sample.payload_wire_ns as f64
}

/// `muser-remote-qualify` prints one JSON object per line: samples, then a
/// summary. Only a sample carries the transfer numbers.
/// Is this failure the machine being busy (lease held, quiet-scan refusal)
/// rather than a real fault?
fn busy_class(text: &str) -> Option<()> {
    (text.contains("another GPU process") || text.contains("accelerator lease")).then_some(())
}

#[derive(serde::Deserialize)]
struct AcceleratorResult {
    schema: String,
    command: Vec<String>,
    exit_status: i32,
    command_log: PathBuf,
}

fn with_result_receipt(argv: &[String], path: &std::path::Path) -> Result<Vec<String>> {
    let split = argv
        .iter()
        .position(|value| value == "--")
        .ok_or_else(|| "accelerator wrapper argv has no command separator".to_string())?;
    let mut result = argv.to_vec();
    result.splice(
        split..split,
        ["--result-receipt".to_string(), path.display().to_string()],
    );
    Ok(result)
}

fn read_accelerator_result(path: &std::path::Path, argv: &[String]) -> Result<AcceleratorResult> {
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|error| format!("result receipt {} is missing: {error}", path.display()))?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Err(format!(
            "result receipt {} is not a regular file",
            path.display()
        ));
    }
    let value: AcceleratorResult = serde_json::from_slice(
        &std::fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?,
    )
    .map_err(|error| format!("parse {}: {error}", path.display()))?;
    let split = argv
        .iter()
        .position(|item| item == "--")
        .ok_or_else(|| "accelerator wrapper argv has no command separator".to_string())?;
    if value.schema != "muser.accelerator-result.v1" || value.command != argv[split + 1..] {
        return Err("accelerator result is bound to a different command".into());
    }
    let log = std::fs::symlink_metadata(&value.command_log).map_err(|error| {
        format!(
            "command log {} is missing: {error}",
            value.command_log.display()
        )
    })?;
    if !log.file_type().is_file() || log.file_type().is_symlink() {
        return Err(format!(
            "command log {} is not a regular file",
            value.command_log.display()
        ));
    }
    Ok(value)
}

fn parse_samples(
    stdout: &str,
    identity: &str,
    recipe: QualificationRecipe,
    native: Option<&NativeQualification>,
) -> Result<Vec<SmokeSample>> {
    let variant = recipe.variant();
    let mut samples = Vec::new();
    let mut summary = None;
    for line in stdout.lines() {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if value.get("schema").and_then(|value| value.as_str()) != Some("muser.remote-qualify.v1") {
            continue;
        }
        if value.get("kind").and_then(|kind| kind.as_str()) == Some("summary") {
            if summary.replace(value).is_some() {
                return Err("remote qualification printed duplicate summaries".into());
            }
            continue;
        }
        if value.get("kind").and_then(|kind| kind.as_str()) != Some("sample") {
            continue;
        }
        if value.get("identity").and_then(|value| value.as_str()) != Some(identity)
            || value.get("variant").and_then(|value| value.as_str()) != Some(variant)
            || value
                .get("prompt_positions")
                .and_then(serde_json::Value::as_u64)
                != Some(SMOKE_PROMPT_TOKENS as u64)
            || value
                .get("output_tokens")
                .and_then(serde_json::Value::as_u64)
                != Some(SMOKE_OUTPUT_TOKENS as u64)
        {
            return Err("qualification sample has a mixed run identity or geometry".into());
        }
        if value
            .get("exact_tokens")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
        {
            return Err("the qualification sample did not report exact tokens".into());
        }
        let exact_full_logits = value
            .get("exact_full_logits")
            .and_then(serde_json::Value::as_bool)
            .ok_or_else(|| "qualification sample omits target full-logit evidence".to_string())?;
        if recipe == QualificationRecipe::KquantTargetPlusDflash && !exact_full_logits {
            return Err("combined qualification sample lacks exact target full logits".into());
        }
        if let Some(rule) = native {
            let maximum = value
                .get("remote_local_logit_max_abs")
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| "native sample omits maximum full-logit drift".to_string())?;
            let mean = value
                .get("remote_local_logit_mean_abs")
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| "native sample omits mean full-logit drift".to_string())?;
            if !maximum.is_finite()
                || !mean.is_finite()
                || maximum > rule.full_logits.maximum_absolute
                || mean > rule.full_logits.maximum_mean_absolute
            {
                return Err(format!(
                    "native full-logit drift exceeds its identity rule: max={maximum}, mean={mean}"
                ));
            }
        }
        // DFlash exactness is required exactly when the lane ships DFlash:
        // a combined sample must prove it, a text sample must not claim it.
        if value
            .get("exact_dflash_tokens")
            .and_then(serde_json::Value::as_bool)
            != (variant == VARIANT_TARGET_PLUS_DFLASH).then_some(true)
        {
            return Err("qualification sample's DFlash evidence does not match its lane".into());
        }
        let field = |name: &str| -> u64 {
            value
                .get(name)
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
        };
        let installed_bytes = field("installed_bytes");
        let installed_segments = field("installed_segments");
        let transfer_commit_ns = field("receiver_transfer_commit_ns");
        let payload_wire_ns = field("producer_payload_wire_ns");
        if installed_bytes == 0
            || installed_segments == 0
            || transfer_commit_ns == 0
            || payload_wire_ns == 0
        {
            return Err(
                "the qualification sample has no positive installed payload or timing".into(),
            );
        }
        samples.push(SmokeSample {
            repetition: field("repetition"),
            installed_bytes,
            installed_segments,
            transfer_commit_ns,
            payload_wire_ns,
        });
    }
    if samples.len() != SMOKE_REPETITIONS
        || samples
            .iter()
            .map(|sample| sample.repetition)
            .collect::<Vec<_>>()
            != (0..SMOKE_REPETITIONS as u64).collect::<Vec<_>>()
    {
        return Err("remote qualification requires exactly three ordered samples".into());
    }
    let summary = summary.ok_or_else(|| "remote qualification printed no summary".to_string())?;
    // `stable` is the sealing campaign's aggregate: for the combined lane it
    // also includes the campaign's speculative-acceptance floor. Enrollment
    // has a different, identity-declared recipe: exactly three ordered token,
    // full-logit, and DFlash-token comparisons (checked above), plus the link
    // floor checked by `run`. Keep the campaign verdict intact, but do not
    // silently substitute it for the enrolled recipe's verdict.
    if summary.get("identity").and_then(|value| value.as_str()) != Some(identity)
        || summary.get("variant").and_then(|value| value.as_str()) != Some(variant)
        || summary
            .get("exact_tokens")
            .and_then(serde_json::Value::as_bool)
            != Some(true)
        || (recipe == QualificationRecipe::KquantTargetPlusDflash
            && summary
                .get("exact_remote_local")
                .and_then(serde_json::Value::as_bool)
                != Some(true))
    {
        return Err(
            "remote qualification summary did not pass the enrolled recipe's exactness gate".into(),
        );
    }
    if let Some(rule) = native {
        let agreements = summary
            .get("token_agreement_rate")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| "native summary omits token agreement rates".to_string())?;
        let maxima = summary
            .get("logit_max_abs")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| "native summary omits maximum logit drift".to_string())?;
        let means = summary
            .get("logit_mean_abs")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| "native summary omits mean logit drift".to_string())?;
        if agreements.len() != SMOKE_REPETITIONS
            || maxima.len() != SMOKE_REPETITIONS
            || means.len() != SMOKE_REPETITIONS
            || agreements.iter().any(|value| value.as_f64() != Some(1.0))
            || maxima.iter().any(|value| {
                value
                    .as_f64()
                    .is_none_or(|value| value > rule.full_logits.maximum_absolute)
            })
            || means.iter().any(|value| {
                value
                    .as_f64()
                    .is_none_or(|value| value > rule.full_logits.maximum_mean_absolute)
            })
            || summary
                .get("fast_generated_tokens_sha256")
                .and_then(serde_json::Value::as_str)
                .is_none_or(|value| value.len() != 64)
        {
            return Err("native qualification summary violates its identity-bound recipe".into());
        }
    }
    Ok(samples)
}

/// A deterministic 2048-position prompt. The point of the smoke prefill is
/// the transfer and the exact-token comparison, not the text: the same ids
/// every time keep two runs against one node comparable.
fn prompt_fixture(recipe: QualificationRecipe) -> String {
    match recipe {
        QualificationRecipe::KquantTargetPlusDflash => (1..=SMOKE_PROMPT_TOKENS)
            .map(|token| token.to_string())
            .collect::<Vec<_>>()
            .join("\n"),
        QualificationRecipe::NativeText => {
            const BODY: [u32; 8] = [19_873, 24, 10_676, 768, 1_085, 13_634, 2_304, 1_509];
            let mut tokens = vec![200_000u32];
            while tokens.len() < SMOKE_PROMPT_TOKENS {
                tokens.extend(BODY);
            }
            tokens.truncate(SMOKE_PROMPT_TOKENS);
            tokens
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join("\n")
                + "\n"
        }
    }
}

/// Keep the generated fixture coupled to the selected lane. A node can be
/// re-enrolled from the historical kquant lane to native while retaining its
/// local node directory; reusing the old recipe's bytes makes the immutable
/// native qualification gate fail before the first handoff.
fn ensure_default_fixture(path: &Path, recipe: QualificationRecipe) -> Result<bool> {
    let wanted = prompt_fixture(recipe);
    if std::fs::read(path).ok().as_deref() == Some(wanted.as_bytes()) {
        return Ok(false);
    }
    write_private(path, wanted.as_bytes())?;
    Ok(true)
}

/// A workspace-wide `cargo build --release` does not activate package-local
/// features on `muser-bench`, even though the server itself enables Metal.
/// Build the exact executor the onboarding gate needs before taking the GPU
/// lease, so a clean clone cannot discover the feature mismatch at runtime.
fn ensure_metal_qualifier(ctx: &Ctx) -> Result<()> {
    let binary = ctx.qualify_binary();
    if std::env::var_os("MUSER_REMOTE_QUALIFY").is_some() {
        return binary
            .is_file()
            .then_some(())
            .ok_or_else(|| format!("MUSER_REMOTE_QUALIFY is not a file: {}", binary.display()));
    }
    if !cfg!(target_os = "macos") {
        return Err("node qualification requires macOS Metal on the consumer".into());
    }
    let output = Command::new("cargo")
        .args([
            "build",
            "--release",
            "--package",
            "muser-bench",
            "--bin",
            "muser-remote-qualify",
            "--features",
            "metal",
        ])
        .current_dir(&ctx.repo_root)
        .output()
        .map_err(|error| format!("build Metal remote qualifier: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "build Metal remote qualifier failed: {}",
            String::from_utf8_lossy(&output.stderr)
                .chars()
                .rev()
                .take(2048)
                .collect::<String>()
                .chars()
                .rev()
                .collect::<String>()
        ));
    }
    if !binary.is_file() {
        return Err(format!(
            "Metal remote qualifier build did not produce {}",
            binary.display()
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn qualify_argv(
    ctx: &Ctx,
    entry: &NodeEntry,
    model: &Path,
    dflash: Option<&Path>,
    fixture: &Path,
    cluster: &Path,
    local: &Path,
    identity: &str,
    variant: &str,
) -> Vec<String> {
    let mut argv: Vec<String> = vec![
        "python3".into(),
        ctx.repo_root
            .join("scripts/accelerator_safe.py")
            .display()
            .to_string(),
        "--execute".into(),
        "--identity".into(),
        identity.into(),
        "--cell".into(),
        format!("node-smoke-{}", entry.name),
        "--out-dir".into(),
        local.join("smoke").display().to_string(),
        "--".into(),
        ctx.qualify_binary().display().to_string(),
        "--model".into(),
        model.display().to_string(),
        "--prompt-token-fixture".into(),
        fixture.display().to_string(),
        "--cluster-config".into(),
        cluster.display().to_string(),
        "--variant".into(),
        variant.into(),
    ];
    // Exactly the combined lane carries a DFlash artifact; the qualify
    // executor refuses `--dflash` on any other variant.
    if let Some(dflash) = dflash {
        argv.push("--dflash".into());
        argv.push(dflash.display().to_string());
    }
    if entry.producer_kind() == super::registry::ProducerKind::Native {
        argv.extend([
            "--onboarding-native".into(),
            "--drift-graded".into(),
            "--reference-once".into(),
        ]);
    }
    argv.extend([
        "--repetitions".into(),
        SMOKE_REPETITIONS.to_string(),
        "--output-tokens".into(),
        SMOKE_OUTPUT_TOKENS.to_string(),
        "--identity".into(),
        identity.into(),
    ]);
    if ctx.dry_run {
        argv.push("--dry-run".into());
    }
    argv
}

#[cfg(test)]
mod tests {
    use super::super::progress::Progress;
    use super::super::registry::{NodeEntry, ProducerKind};
    use super::*;

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

    fn combined_stream(
        identity: &str,
        exact_full_logits: bool,
        exact_dflash_tokens: bool,
        exact_remote_local: bool,
        stable: bool,
    ) -> String {
        let mut lines = Vec::new();
        for repetition in 0..SMOKE_REPETITIONS {
            lines.push(
                serde_json::json!({
                    "schema": "muser.remote-qualify.v1",
                    "kind": "sample",
                    "identity": identity,
                    "variant": "target-plus-dflash",
                    "repetition": repetition,
                    "prompt_positions": 2048,
                    "output_tokens": 256,
                    "exact_tokens": true,
                    "exact_full_logits": exact_full_logits,
                    "exact_dflash_tokens": exact_dflash_tokens,
                    "installed_bytes": 1_000_000,
                    "installed_segments": 4,
                    "receiver_transfer_commit_ns": 20_000_000,
                    "producer_payload_wire_ns": 2_000_000,
                })
                .to_string(),
            );
        }
        lines.push(
            serde_json::json!({
                "schema": "muser.remote-qualify.v1",
                "kind": "summary",
                "identity": identity,
                "variant": "target-plus-dflash",
                "exact_tokens": true,
                "exact_remote_local": exact_remote_local,
                "stable": stable,
            })
            .to_string(),
        );
        lines.join("\n")
    }

    #[test]
    fn the_fixture_is_exactly_the_advertised_prefill_length() {
        for recipe in [
            QualificationRecipe::KquantTargetPlusDflash,
            QualificationRecipe::NativeText,
        ] {
            let fixture = prompt_fixture(recipe);
            assert_eq!(
                fixture.split_ascii_whitespace().count(),
                SMOKE_PROMPT_TOKENS
            );
        }
        assert_eq!(
            format!(
                "{:x}",
                Sha256::digest(prompt_fixture(QualificationRecipe::NativeText).as_bytes())
            ),
            "149ac0d9c37c957823e53c0637b52a38f2ac601089dbda9f98eec4bc5f369030"
        );
    }

    #[test]
    fn switching_lanes_replaces_the_generated_fixture() {
        let root =
            std::env::temp_dir().join(format!("muser-lane-fixture-switch-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("smoke-2048.tokens");
        assert!(
            ensure_default_fixture(&path, QualificationRecipe::KquantTargetPlusDflash).unwrap()
        );
        assert!(ensure_default_fixture(&path, QualificationRecipe::NativeText).unwrap());
        assert!(!ensure_default_fixture(&path, QualificationRecipe::NativeText).unwrap());
        assert_eq!(
            format!("{:x}", Sha256::digest(std::fs::read(&path).unwrap())),
            "149ac0d9c37c957823e53c0637b52a38f2ac601089dbda9f98eec4bc5f369030"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn progress_names_the_enrolled_recipe() {
        let combined = ProducerKind::Llamacpp.qualification_recipe();
        let native = ProducerKind::Native.qualification_recipe();
        assert_eq!(combined.variant(), VARIANT_TARGET_PLUS_DFLASH);
        assert_eq!(native.variant(), VARIANT_TEXT);
        assert!(recipe_progress(combined).contains("recipe kquant/target-plus-dflash"));
        assert!(recipe_progress(native).contains("recipe native/text"));
        assert!(recipe_progress(native).contains("three ordered 2048/256"));
    }

    #[test]
    fn the_sample_line_is_picked_out_of_the_stream() {
        let identity = "node-gx10-test";
        // The campaign aggregate may be false (for example because its
        // acceptance floor is not part of onboarding) while every gate in
        // the enrolled three-handoff recipe is exact.
        let stdout = combined_stream(identity, true, true, true, false);
        let sample = parse_samples(
            &stdout,
            identity,
            QualificationRecipe::KquantTargetPlusDflash,
            None,
        )
        .unwrap();
        let sample = &sample[0];
        assert_eq!(sample.installed_bytes, 1_000_000);
        assert_eq!(sample.installed_segments, 4);
        assert_eq!(sample.transfer_commit_ns, 20_000_000);
        assert_eq!(sample.payload_wire_ns, 2_000_000);
        // Link throughput uses authenticated payload-wire time, not the
        // end-to-end prefill/export/commit span: 1 MB in 2 ms is 4 Gbps.
        assert!((installed_payload_gbps(sample) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn combined_summary_remote_local_mismatch_is_still_refused() {
        let identity = "node-gx10-test";
        let error = parse_samples(
            &combined_stream(identity, true, true, false, true),
            identity,
            QualificationRecipe::KquantTargetPlusDflash,
            None,
        )
        .err()
        .expect("summary mismatch must be refused");
        assert!(error.contains("exactness gate"));
    }

    #[test]
    fn combined_sample_exactness_is_still_fail_closed() {
        let identity = "node-gx10-test";
        for stdout in [
            combined_stream(identity, false, true, true, true),
            combined_stream(identity, true, false, true, true),
        ] {
            assert!(parse_samples(
                &stdout,
                identity,
                QualificationRecipe::KquantTargetPlusDflash,
                None,
            )
            .is_err());
        }
    }

    #[test]
    fn a_stream_without_a_sample_is_a_failure() {
        assert!(parse_samples(
            r#"{"kind":"summary"}"#,
            "test",
            QualificationRecipe::KquantTargetPlusDflash,
            None,
        )
        .is_err());
        assert!(parse_samples(
            r#"{"kind":"sample","exact_tokens":false}"#,
            "test",
            QualificationRecipe::KquantTargetPlusDflash,
            None,
        )
        .is_err());
    }

    /// The native NVFP4 lane's gate: the text variant, no DFlash artifact on
    /// the command line, and samples that carry no DFlash evidence.
    #[test]
    fn the_native_lane_runs_the_text_variant() {
        let entry = NodeEntry {
            producer: Some(ProducerKind::Native),
            ..NodeEntry::draft("gx10", "muser", "gx10.local", Path::new("/tmp"), None)
        };
        let argv = qualify_argv(
            &test_ctx(),
            &entry,
            Path::new("/models/target.gguf"),
            None,
            Path::new("/fixture.tokens"),
            Path::new("/cluster.json"),
            Path::new("/local"),
            "node-gx10-test",
            entry.producer_kind().qualification_recipe().variant(),
        );
        let value_after = |flag: &str| {
            argv.windows(2)
                .find(|pair| pair[0] == flag)
                .map(|pair| pair[1].clone())
        };
        assert_eq!(value_after("--variant").as_deref(), Some("text"));
        assert!(!argv.iter().any(|arg| arg == "--dflash"));
        assert_eq!(value_after("--repetitions").as_deref(), Some("3"));
        assert_eq!(value_after("--output-tokens").as_deref(), Some("256"));
        for flag in ["--onboarding-native", "--drift-graded", "--reference-once"] {
            assert!(argv.iter().any(|argument| argument == flag));
        }

        let identity = "node-gx10-native";
        let runtime = super::super::artifacts::NativeIdentity::load(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .ancestors()
                .nth(2)
                .unwrap(),
        )
        .unwrap();
        let mut lines = Vec::new();
        for repetition in 0..3 {
            lines.push(
                serde_json::json!({
                    "schema": "muser.remote-qualify.v1",
                    "kind": "sample",
                    "identity": identity,
                    "variant": "text",
                    "repetition": repetition,
                    "prompt_positions": 2048,
                    "output_tokens": 256,
                    "exact_tokens": true,
                    "exact_full_logits": false,
                    "remote_local_logit_max_abs": 10.884401321411133,
                    "remote_local_logit_mean_abs": 1.233788776361515,
                    "exact_dflash_tokens": null,
                    "installed_bytes": 1_000_000,
                    "installed_segments": 4,
                    "receiver_transfer_commit_ns": 2_000_000,
                    "producer_payload_wire_ns": 2_000_000,
                })
                .to_string(),
            );
        }
        lines.push(
            serde_json::json!({
                "schema": "muser.remote-qualify.v1",
                "kind": "summary",
                "identity": identity,
                "variant": "text",
                "exact_tokens": true,
                "exact_remote_local": false,
                "stable": false,
                "token_agreement_rate": [1.0, 1.0, 1.0],
                "logit_max_abs": [10.884401321411133, 10.884401321411133, 10.884401321411133],
                "logit_mean_abs": [1.233788776361515, 1.233788776361515, 1.233788776361515],
                "fast_generated_tokens_sha256": "a".repeat(64),
            })
            .to_string(),
        );
        let samples = parse_samples(
            &lines.join("\n"),
            identity,
            QualificationRecipe::NativeText,
            Some(&runtime.qualification),
        )
        .unwrap();
        assert_eq!(samples.len(), SMOKE_REPETITIONS);
    }

    /// A lane and its evidence must agree: DFlash evidence on the text lane
    /// — or a combined-variant sample anywhere near the text gate — is a
    /// hard failure, never a shrug.
    #[test]
    fn a_text_sample_claiming_dflash_evidence_is_refused() {
        let identity = "node-gx10-native";
        let mut lines = Vec::new();
        for repetition in 0..3 {
            lines.push(
                serde_json::json!({
                    "schema": "muser.remote-qualify.v1",
                    "kind": "sample",
                    "identity": identity,
                    "variant": "text",
                    "repetition": repetition,
                    "prompt_positions": 2048,
                    "output_tokens": 256,
                    "exact_tokens": true,
                    "exact_full_logits": true,
                    "exact_dflash_tokens": true,
                    "installed_bytes": 1_000_000,
                    "installed_segments": 4,
                    "receiver_transfer_commit_ns": 2_000_000,
                    "producer_payload_wire_ns": 2_000_000,
                })
                .to_string(),
            );
        }
        assert!(parse_samples(
            &lines.join("\n"),
            identity,
            QualificationRecipe::NativeText,
            None,
        )
        .is_err());
    }

    #[test]
    fn the_llamacpp_lane_keeps_its_combined_gate() {
        let mut entry = NodeEntry::draft("gx10", "muser", "gx10.local", Path::new("/tmp"), None);
        entry.producer = None;
        assert_eq!(entry.producer_kind(), ProducerKind::Llamacpp);
        assert_eq!(
            entry.producer_kind().qualification_recipe().variant(),
            VARIANT_TARGET_PLUS_DFLASH
        );
        let argv = qualify_argv(
            &test_ctx(),
            &entry,
            Path::new("/models/target.gguf"),
            Some(Path::new("/models/dflash.gguf")),
            Path::new("/fixture.tokens"),
            Path::new("/cluster.json"),
            Path::new("/local"),
            "node-gx10-test",
            entry.producer_kind().qualification_recipe().variant(),
        );
        let variant_at = argv.iter().position(|arg| arg == "--variant").unwrap();
        let dflash_at = argv.iter().position(|arg| arg == "--dflash").unwrap();
        assert_eq!(argv[variant_at + 1], "target-plus-dflash");
        assert_eq!(argv[dflash_at + 1], "/models/dflash.gguf");
        // A combined-variant sample is not interchangeable with a text one.
        let mut lines = Vec::new();
        for repetition in 0..3 {
            lines.push(
                serde_json::json!({
                    "schema": "muser.remote-qualify.v1",
                    "kind": "sample",
                    "identity": "node-gx10-test",
                    "variant": "text",
                    "repetition": repetition,
                    "prompt_positions": 2048,
                    "output_tokens": 256,
                    "exact_tokens": true,
                    "exact_full_logits": true,
                    "exact_dflash_tokens": null,
                    "installed_bytes": 1_000_000,
                    "installed_segments": 4,
                    "receiver_transfer_commit_ns": 2_000_000,
                    "producer_payload_wire_ns": 2_000_000,
                })
                .to_string(),
            );
        }
        assert!(parse_samples(
            &lines.join("\n"),
            "node-gx10-test",
            QualificationRecipe::KquantTargetPlusDflash,
            None,
        )
        .is_err());
    }
}
