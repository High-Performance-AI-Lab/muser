//! Paired target-only versus exact target-verified DFlash qualification.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use muser_engine::dflash::DFlashAssistant;
use muser_engine::sampling::SamplingParams;
use muser_engine::{DecodeInput, Model, ModelConfig, PrefillBatch, Session, SessionConfig};
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Backend {
    Cpu,
    Metal,
}

impl Backend {
    fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu-reference",
            Self::Metal => "metal",
        }
    }
}

struct Args {
    model: PathBuf,
    dflash: PathBuf,
    prompt_fixture: PathBuf,
    repetitions: usize,
    output_tokens: usize,
    verify_length: usize,
    target_backend: Backend,
    assistant_backend: Backend,
    identity: String,
    sampled_check_tokens: usize,
    sampled_check_seed: u64,
    dry_run: bool,
}

const SAMPLED_TEMPERATURE: f32 = 0.8;
const SAMPLED_TOP_P: f32 = 0.95;
const SAMPLED_TOP_K: usize = 50;

#[derive(Serialize)]
struct BuildInfo {
    schema: &'static str,
    version: &'static str,
    metal_feature: bool,
}

fn build_info() -> BuildInfo {
    BuildInfo {
        schema: "muser.dflash-qualify.build-info.v1",
        version: env!("CARGO_PKG_VERSION"),
        metal_feature: cfg!(feature = "metal"),
    }
}

#[derive(Serialize)]
struct Sample<'a> {
    schema: &'static str,
    kind: &'static str,
    identity: &'a str,
    prompt_tokens: usize,
    output_tokens: usize,
    repetition: usize,
    order: [&'static str; 2],
    target_only_ns: u64,
    dflash_ns: u64,
    target_prefill_ns: u64,
    dflash_prefill_ns: u64,
    target_ttft_ns: u64,
    dflash_ttft_ns: u64,
    speedup: f64,
    verify_length: usize,
    drafted_tokens: usize,
    accepted_draft_tokens: usize,
    acceptance_rate: f64,
    fallback_tokens: usize,
    draft_ns: u64,
    target_verify_ns: u64,
    fallback_target_ns: u64,
    rounds: usize,
    target_batches: usize,
    accepted_prefix_counts: &'a [usize],
    cycle_trace: &'a [muser_engine::dflash::DFlashCycleTrace],
    draft_token_trace_sha256: String,
    accepted_prefix_trace_sha256: String,
    generated_tokens_sha256: String,
    exact_target_match: bool,
}

fn main() -> ExitCode {
    let mut arguments = std::env::args_os().skip(1);
    if arguments.next().as_deref() == Some(std::ffi::OsStr::new("--build-info"))
        && arguments.next().is_none()
    {
        println!(
            "{}",
            serde_json::to_string(&build_info()).expect("serialize closed build-info")
        );
        return ExitCode::SUCCESS;
    }
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("muser-dflash-qualify: {error}");
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
                "schema": "muser.dflash-qualify.v1",
                "kind": "dry-run",
                "accelerator_touched": false,
                "model": args.model,
                "dflash": args.dflash,
                "prompt_fixture": args.prompt_fixture,
                "repetitions": args.repetitions,
                "output_tokens": args.output_tokens,
                "verify_length": args.verify_length,
                "target_backend": args.target_backend.name(),
                "assistant_backend": args.assistant_backend.name(),
                "identity": args.identity,
                "measurement": "paired in-process prefill, first-token, and decode timing after identical prompt preparation",
                "warmup_policy": "one-untimed-target-plus-dflash-pair-v1",
                "measurement_order": "abba-first-engine-v1",
                "correctness": "exact-target-token-equality plus ordered speculative round trace",
                "sampled_correctness": {
                    "oracle": "muser-engine-scalar-full-distribution-v1",
                    "tokens": args.sampled_check_tokens,
                    "seed": args.sampled_check_seed,
                    "temperature_milli": 800,
                    "top_p_milli": 950,
                    "top_k": SAMPLED_TOP_K,
                    "replays": 2,
                },
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
    if limit > model.config().context_length {
        return Err(format!(
            "prompt plus output is {limit}, model limit is {}",
            model.config().context_length
        ));
    }
    if prompt
        .iter()
        .any(|&token| token as usize >= model.config().vocab_size)
    {
        return Err("prompt fixture contains an out-of-vocabulary token".into());
    }

    let mut assistant = load_assistant(&args, &model)?;
    let mut target_samples = Vec::with_capacity(args.repetitions);
    let mut dflash_samples = Vec::with_capacity(args.repetitions);
    let mut canonical_tokens: Option<Vec<u32>> = None;

    // One complete pair warms both routes. It is deliberately excluded from
    // raw timing evidence, but correctness still has to hold.
    let (warm_target, _, _, _) = run_target(&args, &model, &prompt, limit)?;
    let (warm_dflash, _, _, _) = run_dflash(&args, &model, &mut assistant, &prompt, limit)?;
    if warm_target != warm_dflash {
        return Err("DFlash warmup differs from target-only".into());
    }

    for repetition in 0..args.repetitions {
        let dflash_first = matches!(repetition % 4, 1 | 2);
        let (
            target,
            target_ns,
            target_prefill_ns,
            target_ttft_ns,
            speculative,
            dflash_ns,
            dflash_ttft_ns,
            stats,
        ) = if dflash_first {
            let (tokens, ns, ttft, stats) =
                run_dflash(&args, &model, &mut assistant, &prompt, limit)?;
            let (target, target_ns, target_prefill_ns, target_ttft_ns) =
                run_target(&args, &model, &prompt, limit)?;
            (
                target,
                target_ns,
                target_prefill_ns,
                target_ttft_ns,
                tokens,
                ns,
                ttft,
                stats,
            )
        } else {
            let (target, target_ns, target_prefill_ns, target_ttft_ns) =
                run_target(&args, &model, &prompt, limit)?;
            let (tokens, ns, ttft, stats) =
                run_dflash(&args, &model, &mut assistant, &prompt, limit)?;
            (
                target,
                target_ns,
                target_prefill_ns,
                target_ttft_ns,
                tokens,
                ns,
                ttft,
                stats,
            )
        };
        if target.len() != args.output_tokens || speculative.len() != args.output_tokens {
            return Err(format!(
                "early EOS or short output: target={} DFlash={} expected={}",
                target.len(),
                speculative.len(),
                args.output_tokens
            ));
        }
        if target != speculative {
            let mismatch = target
                .iter()
                .zip(&speculative)
                .position(|(left, right)| left != right)
                .unwrap_or(target.len().min(speculative.len()));
            return Err(format!(
                "DFlash differs from target-only at output token {mismatch}"
            ));
        }
        if let Some(canonical) = canonical_tokens.as_ref() {
            if canonical != &target {
                return Err("target-only output changed between repetitions".into());
            }
        } else {
            canonical_tokens = Some(target.clone());
        }
        target_samples.push(target_ns);
        dflash_samples.push(dflash_ns);
        let order = if dflash_first {
            ["dflash", "target-only"]
        } else {
            ["target-only", "dflash"]
        };
        println!(
            "{}",
            serde_json::to_string(&Sample {
                schema: "muser.dflash-qualify.v1",
                kind: "sample",
                identity: &args.identity,
                prompt_tokens: prompt.len(),
                output_tokens: args.output_tokens,
                repetition,
                order,
                target_only_ns: target_ns,
                dflash_ns,
                target_prefill_ns,
                dflash_prefill_ns: stats.prefill_ns,
                target_ttft_ns,
                dflash_ttft_ns,
                speedup: target_ns as f64 / dflash_ns as f64,
                verify_length: args.verify_length,
                drafted_tokens: stats.drafted_tokens,
                accepted_draft_tokens: stats.accepted_draft_tokens,
                acceptance_rate: stats.acceptance_rate(),
                fallback_tokens: stats.target_only_fallback_tokens,
                draft_ns: stats.draft_ns,
                target_verify_ns: stats.target_verify_ns,
                fallback_target_ns: stats.fallback_target_ns,
                rounds: stats.rounds,
                target_batches: stats.target_batches,
                accepted_prefix_counts: &stats.accepted_prefix_counts,
                cycle_trace: &stats.cycle_trace,
                draft_token_trace_sha256: token_digest(&stats.draft_token_trace),
                accepted_prefix_trace_sha256: token_digest(&stats.accepted_prefix_trace),
                generated_tokens_sha256: token_digest(&target),
                exact_target_match: true,
            })
            .map_err(|error| error.to_string())?
        );
    }

    let target_cv = cv(&target_samples);
    let dflash_cv = cv(&dflash_samples);
    let speedups = target_samples
        .iter()
        .zip(&dflash_samples)
        .map(|(&target, &draft)| target as f64 / draft as f64)
        .collect::<Vec<_>>();
    // Exercise the production sampled route outside every timed sample.  The
    // route uses muser-engine's published scalar full-distribution verifier
    // (including max(p-q, 0) rejection), and two fresh-state executions must
    // be bit-for-bit reproducible for the frozen seed and sampling policy.
    let sampled = run_sampled_scalar_check(&args, &model, &mut assistant, &prompt, limit)?;
    println!(
        "{}",
        serde_json::json!({
            "schema": "muser.dflash-qualify.v1",
            "kind": "summary",
            "identity": args.identity,
            "prompt_tokens": prompt.len(),
            "prompt_file_sha256": format!("{:x}", Sha256::digest(prompt_bytes)),
            "output_tokens": args.output_tokens,
            "verify_length": args.verify_length,
            "target_backend": args.target_backend.name(),
            "assistant_backend": args.assistant_backend.name(),
            "target_only_raw_ns": target_samples,
            "dflash_raw_ns": dflash_samples,
            "target_only_cv": target_cv,
            "dflash_cv": dflash_cv,
            "mean_speedup": speedups.iter().sum::<f64>() / speedups.len() as f64,
            "stable": target_cv <= 0.03 && dflash_cv <= 0.03,
            "warmup_policy": "one-untimed-target-plus-dflash-pair-v1",
            "measurement_order": "abba-first-engine-v1",
            "exact_target_match": true,
            "sampled_scalar_oracle": "muser-engine-scalar-full-distribution-v1",
            "sampled_scalar_match": true,
            "sampled_tokens": args.sampled_check_tokens,
            "sampled_seed": args.sampled_check_seed,
            "sampled_temperature_milli": 800,
            "sampled_top_p_milli": 950,
            "sampled_top_k": SAMPLED_TOP_K,
            "sampled_generated_tokens_sha256": token_digest(&sampled.tokens),
            "sampled_drafted_tokens": sampled.stats.drafted_tokens,
            "sampled_accepted_draft_tokens": sampled.stats.accepted_draft_tokens,
            "sampled_acceptance_rate": sampled.stats.acceptance_rate(),
            "sampled_accepted_prefix_counts": sampled.stats.accepted_prefix_counts,
            "sampled_fallback_tokens": sampled.stats.target_only_fallback_tokens,
            "seal_eligible": false,
            "reason": "cell evidence requires complete eight-prompt packet evaluation",
        })
    );
    Ok(())
}

struct SampledCheck {
    tokens: Vec<u32>,
    stats: muser_engine::dflash::DFlashSpecStats,
}

fn run_sampled_scalar_check(
    args: &Args,
    model: &Model,
    assistant: &mut DFlashAssistant,
    prompt: &[u32],
    limit: usize,
) -> Result<SampledCheck, String> {
    let params = SamplingParams {
        temperature: SAMPLED_TEMPERATURE,
        top_p: SAMPLED_TOP_P,
        top_k: SAMPLED_TOP_K,
        typical_p: 1.0,
        min_p: 0.0,
        top_n_sigma: 0.0,
        min_keep: 0,
    };
    let mut canonical: Option<SampledCheck> = None;
    for _ in 0..2 {
        let mut session = new_session(model, args.target_backend, limit)?;
        let (tokens, stats) = assistant
            .generate_sampled(
                model,
                &mut session,
                prompt,
                args.sampled_check_tokens,
                args.verify_length,
                params,
                args.sampled_check_seed,
            )
            .map_err(|error| error.to_string())?;
        if tokens.len() != args.sampled_check_tokens {
            return Err(format!(
                "sampled scalar check produced {} tokens, expected {}",
                tokens.len(),
                args.sampled_check_tokens
            ));
        }
        if stats.drafted_tokens == 0 || stats.committed_tokens != tokens.len() {
            return Err(
                "sampled scalar check did not exercise a complete speculative route".into(),
            );
        }
        if let Some(expected) = canonical.as_ref() {
            if tokens != expected.tokens
                || stats.drafted_tokens != expected.stats.drafted_tokens
                || stats.accepted_draft_tokens != expected.stats.accepted_draft_tokens
                || stats.target_only_fallback_tokens != expected.stats.target_only_fallback_tokens
            {
                return Err(
                    "sampled scalar check changed between identical fresh-state replays".into(),
                );
            }
        } else {
            canonical = Some(SampledCheck { tokens, stats });
        }
    }
    canonical.ok_or_else(|| "sampled scalar check produced no replay".into())
}

fn run_target(
    args: &Args,
    model: &Model,
    prompt: &[u32],
    limit: usize,
) -> Result<(Vec<u32>, u64, u64, u64), String> {
    let mut session = new_session(model, args.target_backend, limit)?;
    let mut logits = Vec::new();
    let prefill_started = Instant::now();
    for chunk in prompt.chunks(2_048) {
        logits = session
            .prefill(PrefillBatch::tokens(chunk.to_vec()))
            .map_err(|error| error.to_string())?
            .last_logits()
            .to_vec();
    }
    let prefill_ns = nanos(prefill_started.elapsed().as_nanos());
    let started = Instant::now();
    let mut next = argmax(&logits) as u32;
    let mut tokens = Vec::with_capacity(args.output_tokens);
    let mut first_token_ns = None;
    for index in 0..args.output_tokens {
        tokens.push(next);
        first_token_ns.get_or_insert_with(|| nanos(started.elapsed().as_nanos()));
        if index + 1 < args.output_tokens {
            next = session
                .decode(DecodeInput { token_id: next })
                .map_err(|error| error.to_string())?
                .next_token;
        }
    }
    Ok((
        tokens,
        nanos(started.elapsed().as_nanos()),
        prefill_ns,
        prefill_ns.saturating_add(first_token_ns.unwrap_or_default()),
    ))
}

fn run_dflash(
    args: &Args,
    model: &Model,
    assistant: &mut DFlashAssistant,
    prompt: &[u32],
    limit: usize,
) -> Result<(Vec<u32>, u64, u64, muser_engine::dflash::DFlashSpecStats), String> {
    let mut session = new_session(model, args.target_backend, limit)?;
    let prepared = assistant
        .prepare_greedy(model, &mut session, prompt)
        .map_err(|error| error.to_string())?;
    let started = Instant::now();
    let mut first_token_ns = None;
    let (tokens, stats) = assistant
        .generate_prepared_greedy_streaming(
            model,
            &mut session,
            prepared,
            args.output_tokens,
            args.verify_length,
            &[],
            &mut |_| {
                first_token_ns.get_or_insert_with(|| nanos(started.elapsed().as_nanos()));
                Ok(())
            },
        )
        .map_err(|error| error.to_string())?;
    let ttft_ns = stats
        .prefill_ns
        .saturating_add(first_token_ns.unwrap_or_default());
    Ok((tokens, nanos(started.elapsed().as_nanos()), ttft_ns, stats))
}

fn load_assistant(args: &Args, model: &Model) -> Result<DFlashAssistant, String> {
    match args.assistant_backend {
        Backend::Cpu => {
            DFlashAssistant::load(&args.dflash, model).map_err(|error| error.to_string())
        }
        Backend::Metal => {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                DFlashAssistant::load_metal(&args.dflash, model).map_err(|error| error.to_string())
            }
            #[cfg(not(all(target_os = "macos", feature = "metal")))]
            {
                Err("--assistant-backend metal requires macOS and the metal feature".into())
            }
        }
    }
}

fn new_session(model: &Model, backend: Backend, limit: usize) -> Result<Session, String> {
    let config = SessionConfig { max_context: limit };
    match backend {
        Backend::Cpu => model.new_session(config).map_err(|error| error.to_string()),
        Backend::Metal => {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                model
                    .new_metal_session(config)
                    .map_err(|error| error.to_string())
            }
            #[cfg(not(all(target_os = "macos", feature = "metal")))]
            {
                Err("--target-backend metal requires macOS and the metal feature".into())
            }
        }
    }
}

fn parse_args() -> Result<Args, String> {
    let mut model = None;
    let mut dflash = None;
    let mut prompt_fixture = None;
    let mut repetitions = 3usize;
    let mut output_tokens = 256usize;
    let mut verify_length = 7usize;
    let mut target_backend = Backend::Metal;
    let mut assistant_backend = Backend::Metal;
    let mut identity = None;
    let mut sampled_check_tokens = 32usize;
    let mut sampled_check_seed = 0x4d55_5345_5244_464c_u64;
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
            "--prompt-token-fixture" => {
                prompt_fixture = Some(PathBuf::from(value(&mut values, &flag)?))
            }
            "--repetitions" => repetitions = parse_usize(&value(&mut values, &flag)?, &flag)?,
            "--output-tokens" => output_tokens = parse_usize(&value(&mut values, &flag)?, &flag)?,
            "--verify-length" => verify_length = parse_usize(&value(&mut values, &flag)?, &flag)?,
            "--target-backend" => target_backend = parse_backend(&value(&mut values, &flag)?)?,
            "--assistant-backend" => {
                assistant_backend = parse_backend(&value(&mut values, &flag)?)?
            }
            "--identity" => identity = Some(value(&mut values, &flag)?),
            "--sampled-check-tokens" => {
                sampled_check_tokens = parse_usize(&value(&mut values, &flag)?, &flag)?
            }
            "--sampled-check-seed" => {
                sampled_check_seed = value(&mut values, &flag)?
                    .parse()
                    .map_err(|_| format!("invalid {flag}"))?
            }
            "--dry-run" => dry_run = true,
            _ => return Err(format!("unknown argument {flag}")),
        }
    }
    Ok(Args {
        model: model.ok_or("--model is required")?,
        dflash: dflash.ok_or("--dflash is required")?,
        prompt_fixture: prompt_fixture.ok_or("--prompt-token-fixture is required")?,
        repetitions,
        output_tokens,
        verify_length,
        target_backend,
        assistant_backend,
        identity: identity.ok_or("--identity is required")?,
        sampled_check_tokens,
        sampled_check_seed,
        dry_run,
    })
}

fn validate(args: &Args) -> Result<(), String> {
    if args.repetitions == 0 || args.output_tokens == 0 || args.sampled_check_tokens == 0 {
        return Err("repetitions, output tokens, and sampled check tokens must be positive".into());
    }
    if args.sampled_check_tokens > args.output_tokens {
        return Err("sampled check tokens cannot exceed output tokens".into());
    }
    if !matches!(args.verify_length, 3 | 7 | 15) {
        return Err("verify length must be one of 3, 7, or 15".into());
    }
    Ok(())
}

fn parse_backend(value: &str) -> Result<Backend, String> {
    match value {
        "cpu" => Ok(Backend::Cpu),
        "metal" => Ok(Backend::Metal),
        _ => Err(format!("backend must be cpu or metal, got {value}")),
    }
}

fn parse_usize(value: &str, flag: &str) -> Result<usize, String> {
    value
        .parse()
        .map_err(|_| format!("invalid {flag}: {value}"))
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

fn nanos(value: u128) -> u64 {
    value.min(u64::MAX as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_info_reports_the_qualifier_crates_own_metal_cfg() {
        let info = build_info();
        assert_eq!(info.schema, "muser.dflash-qualify.build-info.v1");
        assert_eq!(info.version, env!("CARGO_PKG_VERSION"));
        assert_eq!(
            info.metal_feature,
            cfg!(feature = "metal"),
            "build-info must report this crate's cfg, not dependency feature unification"
        );
    }
}
