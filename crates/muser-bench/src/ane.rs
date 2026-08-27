//! Paired Metal-DFlash versus public-CoreML ANE qualification.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use muser_engine::dflash::{DFlashAssistant, DFlashSpecStats};
use muser_engine::{DecodeInput, Model, ModelConfig, PrefillBatch, Session};
use serde::Serialize;
use sha2::{Digest, Sha256};

struct Args {
    model: PathBuf,
    dflash: PathBuf,
    manifest: PathBuf,
    compute_plan_receipt: PathBuf,
    prompt_fixture: PathBuf,
    repetitions: usize,
    output_tokens: usize,
    verify_length: usize,
    identity: String,
    poc: bool,
    dry_run: bool,
}

#[derive(Serialize)]
struct Sample<'a> {
    schema: &'static str,
    kind: &'static str,
    identity: &'a str,
    prompt_tokens: usize,
    output_tokens: usize,
    repetition: usize,
    order: [&'static str; 3],
    target_only_ns: u64,
    metal_dflash_ns: u64,
    ane_dflash_ns: u64,
    ane_vs_metal_speedup: f64,
    metal_vs_target_speedup: f64,
    ane_vs_target_speedup: f64,
    metal_target_verify_ns: u64,
    ane_target_verify_ns: u64,
    metal_prefill_ns: u64,
    ane_prefill_ns: u64,
    metal_draft_ns: u64,
    ane_draft_ns: u64,
    metal_fallback_target_ns: u64,
    ane_fallback_target_ns: u64,
    metal_rounds: usize,
    ane_rounds: usize,
    metal_drafted_tokens: usize,
    ane_drafted_tokens: usize,
    metal_accepted_draft_tokens: usize,
    ane_accepted_draft_tokens: usize,
    target_verification_tax: f64,
    metal_acceptance_rate: f64,
    ane_acceptance_rate: f64,
    ane_mirror_overlap_attempts: usize,
    ane_mirror_overlap_commits: usize,
    ane_mirror_overlap_rollbacks: usize,
    ane_mirror_overlap_circuit_breaks: usize,
    ane_mirror_overlap_draft_ns: u64,
    ane_mirror_overlap_wall_ns: u64,
    ane_mirror_overlap_hidden_ns: u64,
    ane_mirror_capture_fc_ns: u64,
    generated_tokens_sha256: String,
    exact_target_match: bool,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("muser-ane-qualify: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    validate(&args)?;
    if args.dry_run {
        println!(
            "{}",
            serde_json::json!({
                "schema": "muser.ane-qualify.v1",
                "kind": "dry-run",
                "accelerator_touched": false,
                "model": args.model,
                "dflash": args.dflash,
                "manifest": args.manifest,
                "compute_plan_receipt": args.compute_plan_receipt,
                "prompt_fixture": args.prompt_fixture,
                "repetitions": args.repetitions,
                "output_tokens": args.output_tokens,
                "verify_length": args.verify_length,
                "identity": args.identity,
                "poc": args.poc,
                "routes": ["target-only-metal", "dflash-metal", "dflash-public-coreml-cpu-and-ne"],
                "seal_eligible": false,
            })
        );
        return Ok(());
    }

    let prompt_bytes = std::fs::read(&args.prompt_fixture)
        .map_err(|error| format!("cannot read {}: {error}", args.prompt_fixture.display()))?;
    let prompt = parse_tokens(&prompt_bytes)?;
    if prompt.is_empty() {
        return Err("prompt fixture is empty".into());
    }
    let model = Model::load(ModelConfig::new(&args.model)).map_err(|error| error.to_string())?;
    let limit = prompt
        .len()
        .checked_add(args.output_tokens)
        .ok_or("context length overflow")?;
    if limit > model.config().context_length
        || prompt
            .iter()
            .any(|&token| token as usize >= model.config().vocab_size)
    {
        return Err("prompt/output exceeds target model geometry".into());
    }
    let (mut metal, mut ane, target_identity, draft_identity, plan_sha256) =
        load_assistants(&args, &model)?;

    let mut target_raw = Vec::with_capacity(args.repetitions);
    let mut metal_raw = Vec::with_capacity(args.repetitions);
    let mut ane_raw = Vec::with_capacity(args.repetitions);
    let mut verify_taxes = Vec::with_capacity(args.repetitions);
    let mut canonical: Option<Vec<u32>> = None;
    for repetition in 0..args.repetitions {
        let order = match repetition % 3 {
            0 => ["target-only", "metal-dflash", "ane-dflash"],
            1 => ["metal-dflash", "ane-dflash", "target-only"],
            _ => ["ane-dflash", "target-only", "metal-dflash"],
        };
        let mut target = None;
        let mut metal_result = None;
        let mut ane_result = None;
        for route in order {
            match route {
                "target-only" => target = Some(run_target(&args, &model, &prompt, limit)?),
                "metal-dflash" => {
                    metal_result = Some(run_dflash(&args, &model, &mut metal, &prompt, limit)?)
                }
                "ane-dflash" => {
                    ane_result = Some(run_dflash(&args, &model, &mut ane, &prompt, limit)?)
                }
                _ => unreachable!(),
            }
        }
        let (target_tokens, target_ns) = target.ok_or("target route did not execute")?;
        let (metal_tokens, metal_ns, metal_stats) =
            metal_result.ok_or("Metal DFlash route did not execute")?;
        let (ane_tokens, ane_ns, ane_stats) = ane_result.ok_or("ANE route did not execute")?;
        if target_tokens.len() != args.output_tokens
            || metal_tokens != target_tokens
            || ane_tokens != target_tokens
        {
            return Err(format!(
                "DFlash output differs from exact target-only output: target_len={} metal_len={} ane_len={} metal_first_mismatch={:?} ane_first_mismatch={:?} target_digest={} metal_digest={} ane_digest={} metal_acceptance={:.6} ane_acceptance={:.6}",
                target_tokens.len(),
                metal_tokens.len(),
                ane_tokens.len(),
                first_mismatch(&target_tokens, &metal_tokens),
                first_mismatch(&target_tokens, &ane_tokens),
                token_digest(&target_tokens),
                token_digest(&metal_tokens),
                token_digest(&ane_tokens),
                metal_stats.acceptance_rate(),
                ane_stats.acceptance_rate(),
            ));
        }
        if let Some(previous) = canonical.as_ref() {
            if previous != &target_tokens {
                return Err("target output changed between repetitions".into());
            }
        } else {
            canonical = Some(target_tokens.clone());
        }
        if metal_stats.target_verify_ns == 0 || ane_stats.target_verify_ns == 0 {
            return Err("DFlash target-verification timing is absent".into());
        }
        let verification_tax =
            ane_stats.target_verify_ns as f64 / metal_stats.target_verify_ns as f64 - 1.0;
        target_raw.push(target_ns);
        metal_raw.push(metal_ns);
        ane_raw.push(ane_ns);
        verify_taxes.push(verification_tax);
        println!(
            "{}",
            serde_json::to_string(&Sample {
                schema: "muser.ane-qualify.v1",
                kind: "sample",
                identity: &args.identity,
                prompt_tokens: prompt.len(),
                output_tokens: args.output_tokens,
                repetition,
                order,
                target_only_ns: target_ns,
                metal_dflash_ns: metal_ns,
                ane_dflash_ns: ane_ns,
                ane_vs_metal_speedup: metal_ns as f64 / ane_ns as f64,
                metal_vs_target_speedup: target_ns as f64 / metal_ns as f64,
                ane_vs_target_speedup: target_ns as f64 / ane_ns as f64,
                metal_target_verify_ns: metal_stats.target_verify_ns,
                ane_target_verify_ns: ane_stats.target_verify_ns,
                metal_prefill_ns: metal_stats.prefill_ns,
                ane_prefill_ns: ane_stats.prefill_ns,
                metal_draft_ns: metal_stats.draft_ns,
                ane_draft_ns: ane_stats.draft_ns,
                metal_fallback_target_ns: metal_stats.fallback_target_ns,
                ane_fallback_target_ns: ane_stats.fallback_target_ns,
                metal_rounds: metal_stats.rounds,
                ane_rounds: ane_stats.rounds,
                metal_drafted_tokens: metal_stats.drafted_tokens,
                ane_drafted_tokens: ane_stats.drafted_tokens,
                metal_accepted_draft_tokens: metal_stats.accepted_draft_tokens,
                ane_accepted_draft_tokens: ane_stats.accepted_draft_tokens,
                target_verification_tax: verification_tax,
                metal_acceptance_rate: metal_stats.acceptance_rate(),
                ane_acceptance_rate: ane_stats.acceptance_rate(),
                ane_mirror_overlap_attempts: ane_stats.mirror_overlap_attempts,
                ane_mirror_overlap_commits: ane_stats.mirror_overlap_commits,
                ane_mirror_overlap_rollbacks: ane_stats.mirror_overlap_rollbacks,
                ane_mirror_overlap_circuit_breaks: ane_stats.mirror_overlap_circuit_breaks,
                ane_mirror_overlap_draft_ns: ane_stats.mirror_overlap_draft_ns,
                ane_mirror_overlap_wall_ns: ane_stats.mirror_overlap_wall_ns,
                ane_mirror_overlap_hidden_ns: ane_stats.mirror_overlap_hidden_ns,
                ane_mirror_capture_fc_ns: ane_stats.mirror_capture_fc_ns,
                generated_tokens_sha256: token_digest(&target_tokens),
                exact_target_match: true,
            })
            .map_err(|error| error.to_string())?
        );
    }

    let target_cv = cv(&target_raw);
    let metal_cv = cv(&metal_raw);
    let ane_cv = cv(&ane_raw);
    println!(
        "{}",
        serde_json::json!({
            "schema": "muser.ane-qualify.v1",
            "kind": "summary",
            "identity": args.identity,
            "poc": args.poc,
            "prompt_tokens": prompt.len(),
            "prompt_file_sha256": format!("{:x}", Sha256::digest(prompt_bytes)),
            "output_tokens": args.output_tokens,
            "verify_length": args.verify_length,
            "target_identity": target_identity,
            "dflash_identity": draft_identity,
            "manifest_sha256": file_sha256(&args.manifest)?,
            "compute_plan_receipt_sha256": plan_sha256,
            "compute_units": "CPU_AND_NE",
            "target_only_raw_ns": target_raw,
            "metal_dflash_raw_ns": metal_raw,
            "ane_dflash_raw_ns": ane_raw,
            "target_only_cv": target_cv,
            "metal_dflash_cv": metal_cv,
            "ane_dflash_cv": ane_cv,
            "mean_ane_vs_metal_speedup": paired_mean_speedup(&metal_raw, &ane_raw),
            "mean_metal_vs_target_speedup": paired_mean_speedup(&target_raw, &metal_raw),
            "mean_ane_vs_target_speedup": paired_mean_speedup(&target_raw, &ane_raw),
            "mean_target_verification_tax": verify_taxes.iter().sum::<f64>() / verify_taxes.len() as f64,
            "target_verification_taxes": verify_taxes,
            "stable": target_cv <= 0.03 && metal_cv <= 0.03 && ane_cv <= 0.03,
            "exact_target_match": true,
            "seal_eligible": false,
            "reason": "cell evidence requires the complete eight-prompt ANE packet evaluation",
        })
    );
    Ok(())
}

fn first_mismatch(expected: &[u32], actual: &[u32]) -> Option<(usize, Option<u32>, Option<u32>)> {
    let common = expected.len().min(actual.len());
    if let Some(index) = (0..common).find(|&index| expected[index] != actual[index]) {
        return Some((index, Some(expected[index]), Some(actual[index])));
    }
    (expected.len() != actual.len()).then(|| {
        let index = common;
        (
            index,
            expected.get(index).copied(),
            actual.get(index).copied(),
        )
    })
}

#[cfg(all(target_os = "macos", feature = "ane-coreml"))]
fn load_assistants(
    args: &Args,
    model: &Model,
) -> Result<(DFlashAssistant, DFlashAssistant, String, String, String), String> {
    use std::sync::Arc;

    use muser_engine::dflash::DFlashConfig;
    use muser_engine::dflash_ane::{
        dflash_artifact_identity, file_sha256 as engine_file_sha256, AneDFlashBackend,
    };

    let target_identity = engine_file_sha256(&args.model)?;
    let draft_identity = dflash_artifact_identity(&args.dflash)?;
    let config = DFlashConfig::from_artifact(&args.dflash).map_err(|error| error.to_string())?;
    let backend = Arc::new(AneDFlashBackend::load(
        &args.manifest,
        &target_identity,
        &draft_identity,
        &config,
    )?);
    validate_plan_receipt(
        &args.compute_plan_receipt,
        &args.manifest,
        &target_identity,
        &draft_identity,
    )?;
    let metal =
        DFlashAssistant::load_metal(&args.dflash, model).map_err(|error| error.to_string())?;
    let ane = DFlashAssistant::load_with_projection_backend(&args.dflash, model, backend)
        .map_err(|error| error.to_string())?;
    Ok((
        metal,
        ane,
        target_identity,
        draft_identity,
        file_sha256(&args.compute_plan_receipt)?,
    ))
}

#[cfg(not(all(target_os = "macos", feature = "ane-coreml")))]
fn load_assistants(
    _args: &Args,
    _model: &Model,
) -> Result<(DFlashAssistant, DFlashAssistant, String, String, String), String> {
    Err("ANE qualification requires macOS and the ane-coreml feature".into())
}

#[cfg(all(target_os = "macos", feature = "ane-coreml"))]
fn validate_plan_receipt(
    path: &PathBuf,
    manifest: &PathBuf,
    target: &str,
    draft: &str,
) -> Result<(), String> {
    let manifest_sha256 = file_sha256(manifest)?;
    let receipt: serde_json::Value = serde_json::from_slice(
        &std::fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?,
    )
    .map_err(|error| format!("parse {}: {error}", path.display()))?;
    let v7_residency_valid = receipt
        .get("manifest_version")
        .and_then(|value| value.as_u64())
        .is_none_or(|version| {
            version < 7
                || receipt
                    .get("all_ane_compute_qualified")
                    .and_then(|value| value.as_bool())
                    == Some(true)
        });
    if !matches!(
        receipt.get("schema").and_then(|value| value.as_str()),
        Some("muser-coreml-compute-plan-v3" | "muser-coreml-compute-plan-v4")
    ) || receipt
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
            .get("dflash_identity")
            .and_then(|value| value.as_str())
            != Some(draft)
        || receipt
            .get("manifest_sha256")
            .and_then(|value| value.as_str())
            != Some(manifest_sha256.as_str())
        || !v7_residency_valid
    {
        return Err(
            "CoreML compute-plan receipt is absent, non-resident, or wrong-identity".into(),
        );
    }
    Ok(())
}

fn run_target(
    args: &Args,
    model: &Model,
    prompt: &[u32],
    limit: usize,
) -> Result<(Vec<u32>, u64), String> {
    let mut session = new_metal_session(model, limit)?;
    let mut logits = Vec::new();
    for chunk in prompt.chunks(2_048) {
        logits = session
            .prefill(PrefillBatch::tokens(chunk.to_vec()))
            .map_err(|error| error.to_string())?
            .last_logits()
            .to_vec();
    }
    let started = Instant::now();
    let mut next = argmax(&logits) as u32;
    let mut tokens = Vec::with_capacity(args.output_tokens);
    for index in 0..args.output_tokens {
        tokens.push(next);
        if index + 1 < args.output_tokens {
            next = session
                .decode(DecodeInput { token_id: next })
                .map_err(|error| error.to_string())?
                .next_token;
        }
    }
    Ok((tokens, nanos(started.elapsed().as_nanos())))
}

fn run_dflash(
    args: &Args,
    model: &Model,
    assistant: &mut DFlashAssistant,
    prompt: &[u32],
    limit: usize,
) -> Result<(Vec<u32>, u64, DFlashSpecStats), String> {
    let mut session = new_metal_session(model, limit)?;
    let prepared = assistant
        .prepare_greedy(model, &mut session, prompt)
        .map_err(|error| error.to_string())?;
    let started = Instant::now();
    let (tokens, stats) = assistant
        .generate_prepared_greedy(
            model,
            &mut session,
            prepared,
            args.output_tokens,
            args.verify_length,
            &[],
        )
        .map_err(|error| error.to_string())?;
    Ok((tokens, nanos(started.elapsed().as_nanos()), stats))
}

fn parse_args() -> Result<Args, String> {
    let mut model = None;
    let mut dflash = None;
    let mut manifest = None;
    let mut plan = None;
    let mut prompt = None;
    let mut repetitions = 3usize;
    let mut output_tokens = 256usize;
    let mut verify_length = 7usize;
    let mut identity = None;
    let mut poc = false;
    let mut dry_run = false;
    let mut values = std::env::args().skip(1);
    while let Some(flag) = values.next() {
        let value = |values: &mut std::iter::Skip<std::env::Args>, flag: &str| {
            values
                .next()
                .ok_or_else(|| format!("{flag} requires a value"))
        };
        match flag.as_str() {
            "--model" => model = Some(PathBuf::from(value(&mut values, &flag)?)),
            "--dflash" => dflash = Some(PathBuf::from(value(&mut values, &flag)?)),
            "--manifest" => manifest = Some(PathBuf::from(value(&mut values, &flag)?)),
            "--compute-plan-receipt" => plan = Some(PathBuf::from(value(&mut values, &flag)?)),
            "--prompt-token-fixture" => prompt = Some(PathBuf::from(value(&mut values, &flag)?)),
            "--repetitions" => repetitions = parse_usize(&value(&mut values, &flag)?, &flag)?,
            "--output-tokens" => output_tokens = parse_usize(&value(&mut values, &flag)?, &flag)?,
            "--verify-length" => verify_length = parse_usize(&value(&mut values, &flag)?, &flag)?,
            "--identity" => identity = Some(value(&mut values, &flag)?),
            "--poc" => poc = true,
            "--dry-run" => dry_run = true,
            _ => return Err(format!("unknown argument {flag}")),
        }
    }
    Ok(Args {
        model: model.ok_or("--model is required")?,
        dflash: dflash.ok_or("--dflash is required")?,
        manifest: manifest.ok_or("--manifest is required")?,
        compute_plan_receipt: plan.ok_or("--compute-plan-receipt is required")?,
        prompt_fixture: prompt.ok_or("--prompt-token-fixture is required")?,
        repetitions,
        output_tokens,
        verify_length,
        identity: identity.ok_or("--identity is required")?,
        poc,
        dry_run,
    })
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn new_metal_session(model: &Model, limit: usize) -> Result<Session, String> {
    model
        .new_metal_session(muser_engine::SessionConfig { max_context: limit })
        .map_err(|error| error.to_string())
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn new_metal_session(_model: &Model, _limit: usize) -> Result<Session, String> {
    Err("ANE qualification requires macOS and the metal feature".into())
}

fn validate(args: &Args) -> Result<(), String> {
    if args.poc {
        if args.repetitions != 1 || !(1..=16).contains(&args.output_tokens) {
            return Err("ANE POC requires one repetition and 1..=16 output tokens".into());
        }
    } else if args.repetitions != 3 || args.output_tokens != 256 {
        return Err("ANE qualification requires exactly 3 repetitions and 256 tokens".into());
    }
    if !matches!(args.verify_length, 3 | 7 | 15) {
        return Err("verify length must be one of 3, 7, or 15".into());
    }
    Ok(())
}

fn parse_tokens(bytes: &[u8]) -> Result<Vec<u32>, String> {
    let text = std::str::from_utf8(bytes).map_err(|error| error.to_string())?;
    text.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(line, value)| {
            value
                .trim()
                .parse::<u32>()
                .map_err(|error| format!("invalid token at line {}: {error}", line + 1))
        })
        .collect()
}

fn parse_usize(value: &str, flag: &str) -> Result<usize, String> {
    value
        .parse()
        .map_err(|_| format!("invalid {flag}: {value}"))
}

fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for index in 1..values.len() {
        if values[index] > values[best] {
            best = index;
        }
    }
    best
}

fn token_digest(tokens: &[u32]) -> String {
    let mut digest = Sha256::new();
    for token in tokens {
        digest.update(token.to_le_bytes());
    }
    format!("sha256:{:x}", digest.finalize())
}

fn file_sha256(path: &PathBuf) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
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
    if mean == 0.0 {
        0.0
    } else {
        variance.sqrt() / mean
    }
}

fn paired_mean_speedup(baseline: &[u64], candidate: &[u64]) -> f64 {
    baseline
        .iter()
        .zip(candidate)
        .map(|(&left, &right)| left as f64 / right as f64)
        .sum::<f64>()
        / baseline.len() as f64
}

fn nanos(value: u128) -> u64 {
    value.min(u64::MAX as u128) as u64
}
