//! Real-boundary qualification for Muse target-decoder ANE partitioning.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use muser_engine::decode::MetalMuseModel;
use muser_engine::loader::load_components;
use muser_engine::target_ane::{MuseTargetAneTail, MuseTargetTailResult};
use serde::Serialize;
use sha2::{Digest, Sha256};

const BATCH: usize = 16;
const HIDDEN: usize = 6_656;

struct Args {
    model: PathBuf,
    manifest: PathBuf,
    compute_plan_receipt: PathBuf,
    token_fixture: PathBuf,
    layer: usize,
    repetitions: usize,
    identity: String,
    dry_run: bool,
}

#[derive(Serialize)]
struct ErrorMetrics {
    cosine: f64,
    relative_l2: f64,
    max_abs: f32,
    mean_abs: f64,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("muser-target-ane-qualify: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    if args.repetitions == 0 || args.layer % 4 == 3 || args.layer >= 52 {
        return Err("repetitions must be positive and layer must be a Muse SWA layer".into());
    }
    if args.dry_run {
        println!(
            "{}",
            serde_json::json!({
                "schema": "muser.target-ane-qualify.v1",
                "kind": "dry-run",
                "accelerator_touched": false,
                "model": args.model,
                "manifest": args.manifest,
                "compute_plan_receipt": args.compute_plan_receipt,
                "token_fixture": args.token_fixture,
                "layer": args.layer,
                "batch": BATCH,
                "repetitions": args.repetitions,
                "identity": args.identity,
                "routes": ["muse-metal-swa-tail", "public-coreml-cpu-and-ne-swa-tail"],
                "seal_eligible": false,
            })
        );
        return Ok(());
    }

    let tokens = parse_tokens(&std::fs::read(&args.token_fixture).map_err(|error| {
        format!(
            "read token fixture {}: {error}",
            args.token_fixture.display()
        )
    })?)?;
    if tokens.len() < BATCH {
        return Err("token fixture contains fewer than 16 tokens".into());
    }
    let tokens = &tokens[..BATCH];
    let target_identity = file_sha256(&args.model)?;
    validate_plan_receipt(&args.compute_plan_receipt, &args.manifest, &target_identity)?;
    let components = load_components(&args.model).map_err(|error| error.to_string())?;
    if components.config.hidden_dim != HIDDEN
        || components.config.n_layers != 52
        || args.layer >= components.config.n_layers
        || !components.config.layer_kinds[args.layer].is_swa()
    {
        return Err("loaded model does not have the pinned Muse target geometry".into());
    }
    let post_ffw_norm = components
        .weights
        .f32_vec(&format!("blk.{}.post_ffw_norm.weight", args.layer))
        .map_err(|error| error.to_string())?;
    let max_context = BATCH;
    let mut metal = MetalMuseModel::new(components.config, components.weights, max_context)
        .map_err(|error| error.to_string())?;
    let ane = MuseTargetAneTail::load(&args.manifest, &target_identity)?;
    if ane.layer() != args.layer {
        return Err("Core ML artifact layer differs from requested layer".into());
    }

    // One production forward captures both sides of the real handoff. It is
    // deliberately outside every timed sample.
    let capture = metal
        .forward_batch_capturing_swa_tail(tokens, args.layer)
        .map_err(|error| error.to_string())?;
    if capture
        .attention
        .iter()
        .chain(&capture.residual)
        .any(|x| !x.is_finite())
    {
        return Err("production Metal handoff contains a nonfinite value".into());
    }

    // Compile/warm both routes before paired timing.
    let warm_metal = metal
        .run_swa_tail_metal(args.layer, &capture.attention, &capture.residual)
        .map_err(|error| error.to_string())?;
    let warm_ane = ane.run(&capture.attention, &capture.residual)?;
    let warm_ane_hidden = finish_post_ffw(&warm_ane, &post_ffw_norm);
    let metal_capture_error = metrics(&capture.metal_hidden, &warm_metal.hidden)?;
    let ane_error = metrics(&capture.metal_hidden, &warm_ane_hidden)?;

    let mut metal_ns = Vec::with_capacity(args.repetitions);
    let mut ane_ns = Vec::with_capacity(args.repetitions);
    let mut ane_coreml_ns = Vec::with_capacity(args.repetitions);
    let mut canonical_ane = warm_ane_hidden;
    for repetition in 0..args.repetitions {
        let ane_first = repetition % 2 == 1;
        let (metal_sample, ane_sample, ane_coreml_sample, ane_hidden) = if ane_first {
            let (ane_total, ane_coreml, hidden) = run_ane(&ane, &capture, &post_ffw_norm)?;
            let metal_result = metal
                .run_swa_tail_metal(args.layer, &capture.attention, &capture.residual)
                .map_err(|error| error.to_string())?;
            (metal_result.wall_ns, ane_total, ane_coreml, hidden)
        } else {
            let metal_result = metal
                .run_swa_tail_metal(args.layer, &capture.attention, &capture.residual)
                .map_err(|error| error.to_string())?;
            let (ane_total, ane_coreml, hidden) = run_ane(&ane, &capture, &post_ffw_norm)?;
            (metal_result.wall_ns, ane_total, ane_coreml, hidden)
        };
        let repeat_error = metrics(&canonical_ane, &ane_hidden)?;
        if repeat_error.max_abs != 0.0 {
            return Err("public Core ML target-tail output changed between repetitions".into());
        }
        canonical_ane = ane_hidden;
        metal_ns.push(metal_sample);
        ane_ns.push(ane_sample);
        ane_coreml_ns.push(ane_coreml_sample);
        println!(
            "{}",
            serde_json::json!({
                "schema": "muser.target-ane-qualify.v1",
                "kind": "sample",
                "identity": args.identity,
                "layer": args.layer,
                "batch": BATCH,
                "repetition": repetition,
                "order": if ane_first { ["ane", "metal"] } else { ["metal", "ane"] },
                "metal_tail_ns": metal_sample,
                "ane_coreml_ns": ane_coreml_sample,
                "ane_tail_total_ns": ane_sample,
                "ane_vs_metal_speedup": metal_sample as f64 / ane_sample as f64,
            })
        );
    }
    let metal_cv = cv(&metal_ns);
    let ane_cv = cv(&ane_ns);
    let mean_speedup = paired_mean_speedup(&metal_ns, &ane_ns);
    println!(
        "{}",
        serde_json::json!({
            "schema": "muser.target-ane-qualify.v1",
            "kind": "summary",
            "identity": args.identity,
            "model_sha256": target_identity,
            "manifest_sha256": file_sha256(&args.manifest)?,
            "compute_plan_receipt_sha256": file_sha256(&args.compute_plan_receipt)?,
            "layer": args.layer,
            "layer_kind": "swa_rope_2048",
            "batch": BATCH,
            "metal_tail_raw_ns": metal_ns,
            "ane_coreml_raw_ns": ane_coreml_ns,
            "ane_tail_total_raw_ns": ane_ns,
            "metal_cv": metal_cv,
            "ane_cv": ane_cv,
            "mean_ane_vs_metal_speedup": mean_speedup,
            "metal_replay_vs_production_capture": metal_capture_error,
            "ane_int8_vs_production_metal": ane_error,
            "compute_units": "CPU_AND_NE",
            "mlcomputeplan_all_conv_resident_on_neural_engine": true,
            "stable": metal_cv <= 0.03 && ane_cv <= 0.03,
            "finite": true,
            "seal_eligible": false,
            "reason": "single-layer empirical partition POC; full target-token integration is still required",
        })
    );
    Ok(())
}

fn run_ane(
    ane: &MuseTargetAneTail,
    capture: &muser_engine::decode::MuseSwaTailCapture,
    post_ffw_norm: &[f32],
) -> Result<(u64, u64, Vec<f32>), String> {
    let total_started = Instant::now();
    let coreml_started = Instant::now();
    let result = ane.run(&capture.attention, &capture.residual)?;
    let coreml_ns = nanos(coreml_started.elapsed().as_nanos());
    let hidden = finish_post_ffw(&result, post_ffw_norm);
    Ok((nanos(total_started.elapsed().as_nanos()), coreml_ns, hidden))
}

fn finish_post_ffw(result: &MuseTargetTailResult, weight: &[f32]) -> Vec<f32> {
    debug_assert_eq!(weight.len(), HIDDEN);
    debug_assert_eq!(result.ffn_input.len(), BATCH * HIDDEN);
    debug_assert_eq!(result.down_projection.len(), BATCH * HIDDEN);
    let mut hidden = result.ffn_input.clone();
    for token in 0..BATCH {
        let row = &result.down_projection[token * HIDDEN..(token + 1) * HIDDEN];
        let inv_rms = (row.iter().map(|value| value * value).sum::<f32>() / HIDDEN as f32 + 1.0e-8)
            .sqrt()
            .recip();
        for channel in 0..HIDDEN {
            hidden[token * HIDDEN + channel] += row[channel] * inv_rms * weight[channel];
        }
    }
    hidden
}

fn metrics(reference: &[f32], candidate: &[f32]) -> Result<ErrorMetrics, String> {
    if reference.len() != candidate.len()
        || reference.is_empty()
        || reference
            .iter()
            .chain(candidate)
            .any(|value| !value.is_finite())
    {
        return Err("comparison tensors differ in shape or contain nonfinite values".into());
    }
    let mut dot = 0.0f64;
    let mut ref_sq = 0.0f64;
    let mut candidate_sq = 0.0f64;
    let mut diff_sq = 0.0f64;
    let mut abs_sum = 0.0f64;
    let mut max_abs = 0.0f32;
    for (&a, &b) in reference.iter().zip(candidate) {
        let a64 = a as f64;
        let b64 = b as f64;
        let diff = (a - b).abs();
        dot += a64 * b64;
        ref_sq += a64 * a64;
        candidate_sq += b64 * b64;
        diff_sq += (a64 - b64) * (a64 - b64);
        abs_sum += diff as f64;
        max_abs = max_abs.max(diff);
    }
    Ok(ErrorMetrics {
        cosine: dot / (ref_sq.sqrt() * candidate_sq.sqrt()),
        relative_l2: (diff_sq / ref_sq).sqrt(),
        max_abs,
        mean_abs: abs_sum / reference.len() as f64,
    })
}

fn validate_plan_receipt(path: &Path, manifest: &Path, target: &str) -> Result<(), String> {
    let manifest_sha256 = file_sha256(manifest)?;
    let receipt: serde_json::Value = serde_json::from_slice(
        &std::fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?,
    )
    .map_err(|error| format!("parse {}: {error}", path.display()))?;
    let packages = receipt
        .get("target_packages")
        .and_then(|value| value.as_array())
        .ok_or("compute-plan receipt has no target packages")?;
    if receipt.get("schema").and_then(|value| value.as_str())
        != Some("muser-coreml-compute-plan-v4")
        || receipt
            .get("plan_compute_units")
            .and_then(|value| value.as_str())
            != Some("CPU_AND_NE")
        || receipt
            .get("all_conv_resident_on_neural_engine")
            .and_then(|value| value.as_bool())
            != Some(true)
        || receipt
            .get("target_identity")
            .and_then(|value| value.as_str())
            != Some(target)
        || receipt
            .get("manifest_sha256")
            .and_then(|value| value.as_str())
            != Some(manifest_sha256.as_str())
        || packages.len() != 2
        || packages.iter().any(|package| {
            package
                .get("conv_resident_on_neural_engine")
                .and_then(|value| value.as_bool())
                != Some(true)
        })
    {
        return Err(
            "Core ML target compute-plan receipt is absent, non-resident, or wrong-identity".into(),
        );
    }
    Ok(())
}

fn parse_args() -> Result<Args, String> {
    let mut model = None;
    let mut manifest = None;
    let mut receipt = None;
    let mut tokens = None;
    let mut layer = 0usize;
    let mut repetitions = 3usize;
    let mut identity = None;
    let mut dry_run = false;
    let mut values = std::env::args().skip(1);
    while let Some(flag) = values.next() {
        let next = |values: &mut std::iter::Skip<std::env::Args>, flag: &str| {
            values
                .next()
                .ok_or_else(|| format!("{flag} requires a value"))
        };
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(next(&mut values, &flag)?)),
            "--manifest" => manifest = Some(PathBuf::from(next(&mut values, &flag)?)),
            "--compute-plan-receipt" => receipt = Some(PathBuf::from(next(&mut values, &flag)?)),
            "--token-fixture" => tokens = Some(PathBuf::from(next(&mut values, &flag)?)),
            "--layer" => {
                layer = next(&mut values, &flag)?
                    .parse()
                    .map_err(|_| "bad --layer")?
            }
            "--repetitions" => {
                repetitions = next(&mut values, &flag)?
                    .parse()
                    .map_err(|_| "bad --repetitions")?
            }
            "--identity" => identity = Some(next(&mut values, &flag)?),
            "--dry-run" => dry_run = true,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(Args {
        model: model.ok_or("--model is required")?,
        manifest: manifest.ok_or("--manifest is required")?,
        compute_plan_receipt: receipt.ok_or("--compute-plan-receipt is required")?,
        token_fixture: tokens.ok_or("--token-fixture is required")?,
        layer,
        repetitions,
        identity: identity.ok_or("--identity is required")?,
        dry_run,
    })
}

fn parse_tokens(bytes: &[u8]) -> Result<Vec<u32>, String> {
    std::str::from_utf8(bytes)
        .map_err(|error| error.to_string())?
        .split_whitespace()
        .map(|word| word.parse::<u32>().map_err(|error| error.to_string()))
        .collect()
}

fn file_sha256(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 8 * 1024 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn cv(samples: &[u64]) -> f64 {
    let mean = samples.iter().map(|&value| value as f64).sum::<f64>() / samples.len() as f64;
    let variance = samples
        .iter()
        .map(|&value| {
            let delta = value as f64 - mean;
            delta * delta
        })
        .sum::<f64>()
        / samples.len() as f64;
    variance.sqrt() / mean
}

fn paired_mean_speedup(baseline: &[u64], candidate: &[u64]) -> f64 {
    baseline
        .iter()
        .zip(candidate)
        .map(|(&base, &new)| base as f64 / new as f64)
        .sum::<f64>()
        / baseline.len() as f64
}

fn nanos(value: u128) -> u64 {
    value.min(u64::MAX as u128) as u64
}
