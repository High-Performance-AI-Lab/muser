//! Exact free-running Muse greedy evidence over an audited token fixture.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

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
    prompt_fixture: PathBuf,
    output_tokens: usize,
    backend: Backend,
    identity: String,
    tokens_out: Option<PathBuf>,
    snapshot_replay: bool,
    dry_run: bool,
}

#[derive(Serialize)]
struct Candidate {
    token_id: u32,
    logit: f32,
}

#[derive(Serialize)]
struct Decision<'a> {
    schema: &'static str,
    kind: &'static str,
    identity: &'a str,
    backend: &'static str,
    index: usize,
    position: usize,
    target_position: usize,
    selected_token_id: u32,
    candidates: [Candidate; 2],
    logits_sha256: String,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("muser-greedy-evidence: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    if args.dry_run {
        println!(
            "{}",
            serde_json::json!({
                "schema": "muser.greedy-evidence.v1",
                "kind": "dry-run",
                "accelerator_touched": false,
                "model": args.model,
                "prompt_fixture": args.prompt_fixture,
                "output_tokens": args.output_tokens,
                "backend": args.backend.name(),
                "identity": args.identity,
                "tokens_out": args.tokens_out,
                "snapshot_replay": args.snapshot_replay,
                "eos_policy": "fail-on-early-eos",
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
    if prompt
        .iter()
        .any(|token| *token as usize >= model.config().vocab_size)
    {
        return Err("prompt fixture contains a token outside the Muse vocabulary".into());
    }
    let context = prompt
        .len()
        .checked_add(args.output_tokens)
        .ok_or("context length overflow")?;
    if context > model.config().context_length {
        return Err("prompt plus output exceeds the model context".into());
    }
    let mut session = new_session(&model, args.backend, context)?;
    let mut logits = Vec::new();
    for chunk in prompt.chunks(2_048) {
        logits = session
            .prefill(PrefillBatch::tokens(chunk.to_vec()))
            .map_err(|error| error.to_string())?
            .last_logits()
            .to_vec();
    }
    let replay_state = if args.snapshot_replay {
        Some((
            session
                .export_cache_snapshot()
                .map_err(|error| error.to_string())?,
            logits.clone(),
        ))
    } else {
        None
    };
    let mut generated = Vec::with_capacity(args.output_tokens);
    let mut all_logits = Sha256::new();
    for index in 0..args.output_tokens {
        if logits.len() != model.config().vocab_size || logits.iter().any(|x| !x.is_finite()) {
            let nans = logits.iter().filter(|value| value.is_nan()).count();
            let infs = logits.iter().filter(|value| value.is_infinite()).count();
            return Err(format!(
                "nonfinite or incomplete logits at output row {index}: len {} expected {}, {nans} NaN, {infs} Inf",
                logits.len(),
                model.config().vocab_size
            ));
        }
        let (first, second) = top2(&logits)?;
        let selected = first as u32;
        if model.config().eos_tokens.contains(&selected) {
            return Err(format!(
                "early EOS token {selected} at output row {index}; exact horizon is {}",
                args.output_tokens
            ));
        }
        let bytes = f32_bytes(&logits);
        let row_hash = Sha256::digest(&bytes);
        all_logits.update(&bytes);
        println!(
            "{}",
            serde_json::to_string(&Decision {
                schema: "muser.greedy-evidence.v1",
                kind: "decision",
                identity: &args.identity,
                backend: args.backend.name(),
                index,
                position: prompt.len() - 1 + index,
                target_position: prompt.len() + index,
                selected_token_id: selected,
                candidates: [
                    Candidate {
                        token_id: first as u32,
                        logit: logits[first],
                    },
                    Candidate {
                        token_id: second as u32,
                        logit: logits[second],
                    },
                ],
                logits_sha256: format!("{row_hash:x}"),
            })
            .map_err(|error| error.to_string())?
        );
        generated.push(selected);
        if index + 1 < args.output_tokens {
            logits = session
                .decode(DecodeInput { token_id: selected })
                .map_err(|error| error.to_string())?
                .logits;
        }
    }
    let all_logits_sha256 = format!("{:x}", all_logits.finalize());
    drop(session);
    let (snapshot_position, replay_token_digest, replay_logit_digest, replay_exact) =
        if let Some((snapshot, restored_logits)) = replay_state {
            let mut replay = new_session(&model, args.backend, context)?;
            replay
                .install_cache_snapshot(&snapshot)
                .map_err(|error| error.to_string())?;
            replay
                .install_restored_logits(&restored_logits)
                .map_err(|error| error.to_string())?;
            let (tokens, digest) =
                generate_digest(&model, &mut replay, restored_logits, args.output_tokens)?;
            let token_digest = token_digest(&tokens);
            (
                Some(snapshot.position),
                Some(token_digest),
                Some(digest.clone()),
                tokens == generated && digest == all_logits_sha256,
            )
        } else {
            (None, None, None, true)
        };
    if args.snapshot_replay && !replay_exact {
        return Err("detached snapshot replay changed tokens or full logits".into());
    }
    if let Some(path) = args.tokens_out.as_deref() {
        publish_tokens(path, &generated)?;
    }
    println!(
        "{}",
        serde_json::json!({
            "schema": "muser.greedy-evidence.v1",
            "kind": "summary",
            "identity": args.identity,
            "backend": args.backend.name(),
            "prompt_tokens": prompt.len(),
            "output_tokens": generated.len(),
            "prompt_file_sha256": format!("{:x}", Sha256::digest(&prompt_bytes)),
            "prompt_tokens_sha256": token_digest(&prompt),
            "generated_token_ids": generated,
            "generated_tokens_sha256": token_digest(&generated),
            "all_logits_sha256": all_logits_sha256,
            "snapshot_replay_requested": args.snapshot_replay,
            "snapshot_position": snapshot_position,
            "snapshot_replay_generated_tokens_sha256": replay_token_digest,
            "snapshot_replay_all_logits_sha256": replay_logit_digest,
            "snapshot_replay_exact": replay_exact,
            "nonfinite_values": 0,
            "seal_eligible": replay_exact,
        })
    );
    Ok(())
}

fn new_session(model: &Model, backend: Backend, max_context: usize) -> Result<Session, String> {
    match backend {
        Backend::Cpu => model
            .new_session(SessionConfig { max_context })
            .map_err(|error| error.to_string()),
        Backend::Metal => {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                model
                    .new_metal_session(SessionConfig { max_context })
                    .map_err(|error| error.to_string())
            }
            #[cfg(not(all(target_os = "macos", feature = "metal")))]
            {
                Err("--backend metal requires macOS and the metal feature".into())
            }
        }
    }
}

fn generate_digest(
    model: &Model,
    session: &mut Session,
    mut logits: Vec<f32>,
    output_tokens: usize,
) -> Result<(Vec<u32>, String), String> {
    let mut generated = Vec::with_capacity(output_tokens);
    let mut digest = Sha256::new();
    for index in 0..output_tokens {
        if logits.len() != model.config().vocab_size || logits.iter().any(|x| !x.is_finite()) {
            return Err(format!("invalid restored logits at output row {index}"));
        }
        digest.update(f32_bytes(&logits));
        let selected = top2(&logits)?.0 as u32;
        if model.config().eos_tokens.contains(&selected) {
            return Err(format!(
                "early EOS in restored replay at output row {index}"
            ));
        }
        generated.push(selected);
        if index + 1 < output_tokens {
            logits = session
                .decode(DecodeInput { token_id: selected })
                .map_err(|error| error.to_string())?
                .logits;
        }
    }
    Ok((generated, format!("{:x}", digest.finalize())))
}

fn top2(logits: &[f32]) -> Result<(usize, usize), String> {
    if logits.len() < 2 {
        return Err("vocabulary has fewer than two logits".into());
    }
    let mut first = 0usize;
    let mut second = 1usize;
    if better(logits, second, first) {
        std::mem::swap(&mut first, &mut second);
    }
    for token in 2..logits.len() {
        if better(logits, token, first) {
            second = first;
            first = token;
        } else if better(logits, token, second) {
            second = token;
        }
    }
    Ok((first, second))
}

fn better(logits: &[f32], left: usize, right: usize) -> bool {
    logits[left] > logits[right] || (logits[left] == logits[right] && left < right)
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    bytes
}

fn parse_tokens(bytes: &[u8]) -> Result<Vec<u32>, String> {
    let text = std::str::from_utf8(bytes).map_err(|error| error.to_string())?;
    text.split(|character: char| character == ',' || character.is_ascii_whitespace())
        .filter(|field| !field.is_empty())
        .map(|field| field.parse::<u32>().map_err(|error| error.to_string()))
        .collect()
}

fn token_digest(tokens: &[u32]) -> String {
    let mut digest = Sha256::new();
    for token in tokens {
        digest.update(token.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn publish_tokens(path: &Path, tokens: &[u32]) -> Result<(), String> {
    let payload = tokens
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("cannot create {}: {error}", path.display()))?;
    file.write_all(payload.as_bytes())
        .and_then(|_| file.flush())
        .map_err(|error| error.to_string())?;
    file.sync_all().map_err(|error| error.to_string())
}

fn parse_args() -> Result<Args, String> {
    let argv = std::env::args().skip(1).collect::<Vec<_>>();
    let mut model = None;
    let mut prompt_fixture = None;
    let mut output_tokens = 64usize;
    let mut backend = Backend::Cpu;
    let mut identity = "unsealed-local".to_string();
    let mut tokens_out = None;
    let mut snapshot_replay = false;
    let mut dry_run = false;
    let mut index = 0;
    while index < argv.len() {
        let value = |name: &str, index: usize| {
            argv.get(index + 1)
                .cloned()
                .ok_or_else(|| format!("{name} requires a value"))
        };
        match argv[index].as_str() {
            "--model" => model = Some(PathBuf::from(value("--model", index)?)),
            "--prompt-token-fixture" => {
                prompt_fixture = Some(PathBuf::from(value("--prompt-token-fixture", index)?))
            }
            "--output-tokens" => {
                output_tokens = value("--output-tokens", index)?
                    .parse()
                    .map_err(|_| "--output-tokens must be an integer")?
            }
            "--backend" => {
                backend = match value("--backend", index)?.as_str() {
                    "cpu" => Backend::Cpu,
                    "metal" => Backend::Metal,
                    other => return Err(format!("unknown backend {other:?}")),
                }
            }
            "--identity" => identity = value("--identity", index)?,
            "--tokens-out" => tokens_out = Some(PathBuf::from(value("--tokens-out", index)?)),
            "--snapshot-replay" => {
                snapshot_replay = true;
                index += 1;
                continue;
            }
            "--dry-run" => {
                dry_run = true;
                index += 1;
                continue;
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
        index += 2;
    }
    if output_tokens == 0 {
        return Err("--output-tokens must be positive".into());
    }
    Ok(Args {
        model: model.ok_or("--model is required")?,
        prompt_fixture: prompt_fixture.ok_or("--prompt-token-fixture is required")?,
        output_tokens,
        backend,
        identity,
        tokens_out,
        snapshot_replay,
        dry_run,
    })
}
