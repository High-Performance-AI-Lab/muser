//! Teacher-forced full-logit evidence for the Muse correctness co-gate.
//!
//! The row contract follows Ferrite's proven forward-evidence shape, reduced
//! to the one Muse model.  An even N-token fixture scores positions N/2..
//! N-2, exactly matching llama-perplexity's saved-logit window.  Raw f32 rows
//! are optional append-only evidence for the CPU/Metal <= 0.5 logit gate.

use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use muser_engine::{DecodeInput, Model, ModelConfig, PrefillBatch, SessionConfig};
use serde::Serialize;
use sha2::{Digest, Sha256};

const MAGIC: &[u8; 8] = b"MUSLOG1\0";
const VERSION: u32 = 1;

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
    token_fixture: PathBuf,
    backend: Backend,
    top_k: usize,
    logits_out: Option<PathBuf>,
    render_text_out: Option<PathBuf>,
    identity: String,
    dry_run: bool,
}

#[derive(Serialize)]
struct Candidate {
    token: usize,
    logit: f32,
}

#[derive(Serialize)]
struct Row<'a> {
    schema: &'static str,
    kind: &'static str,
    identity: &'a str,
    backend: &'static str,
    window: usize,
    pos: usize,
    input_token: u32,
    target_token: u32,
    target_nll: f64,
    top_k: Vec<Candidate>,
    logits_sha256: String,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("muser-forward-evidence: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    if args.dry_run {
        println!(
            "{}",
            serde_json::to_string(&serde_json::json!({
                "schema": "muser.forward-evidence.v1",
                "kind": "dry-run",
                "accelerator_touched": false,
                "model": args.model,
                "token_fixture": args.token_fixture,
                "backend": args.backend.name(),
                "top_k": args.top_k,
                "logits_out": args.logits_out,
                "render_text_out": args.render_text_out,
                "identity": args.identity,
            }))
            .map_err(|error| error.to_string())?
        );
        return Ok(());
    }

    let tokens = load_tokens(&args.token_fixture)?;
    if tokens.len() < 2 {
        return Err("token fixture must contain at least two tokens".into());
    }
    let model = Model::load(ModelConfig::new(&args.model)).map_err(|error| error.to_string())?;
    if tokens
        .iter()
        .any(|token| *token as usize >= model.config().vocab_size)
    {
        return Err("teacher fixture contains a token outside the Muse vocabulary".into());
    }
    if args.top_k > model.config().vocab_size {
        return Err("--top-k exceeds the Muse vocabulary".into());
    }
    if let Some(path) = args.render_text_out.as_deref() {
        let bos = model
            .config()
            .bos_token_id
            .ok_or("Muse model does not declare a BOS token")?;
        if tokens.first() != Some(&bos) {
            return Err("rendered llama corpus fixture must begin with the Muse BOS token".into());
        }
        let text = model.decode_tokens(&tokens[1..]);
        if model.encode(&text) != tokens[1..] {
            return Err(
                "token fixture has no exact UTF-8 decode/re-encode representation for llama".into(),
            );
        }
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|error| format!("cannot create corpus {}: {error}", path.display()))?;
        file.write_all(text.as_bytes())
            .and_then(|_| file.flush())
            .map_err(|error| error.to_string())?;
        file.sync_all().map_err(|error| error.to_string())?;
        println!(
            "{}",
            serde_json::to_string(&serde_json::json!({
                "schema": "muser.forward-evidence.v1",
                "kind": "rendered-corpus",
                "identity": args.identity,
                "token_count": tokens.len(),
                "tokens_sha256": token_digest(&tokens),
                "text_sha256": format!("{:x}", Sha256::digest(text.as_bytes())),
                "path": path,
                "exact_round_trip": true,
            }))
            .map_err(|error| error.to_string())?
        );
        return Ok(());
    }
    if tokens.len() < 4 || tokens.len() % 2 != 0 {
        return Err("teacher fixture must contain an even number of at least four tokens".into());
    }
    let mut session = match args.backend {
        Backend::Cpu => model
            .new_session(SessionConfig {
                max_context: tokens.len(),
            })
            .map_err(|error| error.to_string())?,
        Backend::Metal => {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            {
                model
                    .new_metal_session(SessionConfig {
                        max_context: tokens.len(),
                    })
                    .map_err(|error| error.to_string())?
            }
            #[cfg(not(all(target_os = "macos", feature = "metal")))]
            {
                return Err("--backend metal requires macOS and the metal feature".into());
            }
        }
    };
    let scored_start = tokens.len() / 2;
    session
        .prefill(PrefillBatch::tokens(tokens[..scored_start].to_vec()))
        .map_err(|error| error.to_string())?;

    let row_count = tokens.len() - 1 - scored_start;
    let mut raw = args
        .logits_out
        .as_deref()
        .map(|path| {
            open_raw(
                path,
                tokens.len(),
                model.config().vocab_size,
                row_count,
                &tokens,
            )
        })
        .transpose()?;
    let mut summed_target_nll = 0.0f64;
    let mut evidence_digest = Sha256::new();
    for position in scored_start..tokens.len() - 1 {
        let decoded = session
            .decode(DecodeInput {
                token_id: tokens[position],
            })
            .map_err(|error| error.to_string())?;
        if decoded.logits.len() != model.config().vocab_size
            || decoded.logits.iter().any(|value| !value.is_finite())
        {
            return Err(format!(
                "nonfinite or incomplete logits at position {position}"
            ));
        }
        let bytes = f32_bytes(&decoded.logits);
        let row_digest = Sha256::digest(&bytes);
        evidence_digest.update(&bytes);
        if let Some(writer) = raw.as_mut() {
            writer
                .write_all(&bytes)
                .map_err(|error| error.to_string())?;
        }
        let target_nll = target_nll(&decoded.logits, tokens[position + 1] as usize)?;
        summed_target_nll += target_nll;
        let top_k = top_k(&decoded.logits, args.top_k)
            .into_iter()
            .map(|token| Candidate {
                token,
                logit: decoded.logits[token],
            })
            .collect();
        let row = Row {
            schema: "muser.forward-evidence.v1",
            kind: "row",
            identity: &args.identity,
            backend: args.backend.name(),
            window: 0,
            pos: position,
            input_token: tokens[position],
            target_token: tokens[position + 1],
            target_nll,
            top_k,
            logits_sha256: format!("{row_digest:x}"),
        };
        println!(
            "{}",
            serde_json::to_string(&row).map_err(|error| error.to_string())?
        );
    }
    if let Some(mut writer) = raw {
        writer.flush().map_err(|error| error.to_string())?;
        writer
            .get_ref()
            .sync_all()
            .map_err(|error| error.to_string())?;
    }
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "schema": "muser.forward-evidence.v1",
            "kind": "summary",
            "identity": args.identity,
            "backend": args.backend.name(),
            "context_length": tokens.len(),
            "vocab_size": model.config().vocab_size,
            "scored_rows": row_count,
            "summed_target_nll": summed_target_nll,
            "all_logits_sha256": format!("{:x}", evidence_digest.finalize()),
            "raw_logits": args.logits_out,
            "nonfinite_values": 0,
            "seal_eligible": true,
        }))
        .map_err(|error| error.to_string())?
    );
    Ok(())
}

fn open_raw(
    path: &Path,
    context: usize,
    vocab: usize,
    rows: usize,
    tokens: &[u32],
) -> Result<BufWriter<std::fs::File>, String> {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("cannot create raw logits {}: {error}", path.display()))?;
    let mut writer = BufWriter::new(file);
    writer.write_all(MAGIC).map_err(|error| error.to_string())?;
    for value in [VERSION, context as u32, vocab as u32, rows as u32] {
        writer
            .write_all(&value.to_le_bytes())
            .map_err(|error| error.to_string())?;
    }
    for token in tokens {
        writer
            .write_all(&token.to_le_bytes())
            .map_err(|error| error.to_string())?;
    }
    Ok(writer)
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    bytes
}

fn target_nll(logits: &[f32], target: usize) -> Result<f64, String> {
    let maximum = logits
        .iter()
        .copied()
        .reduce(f32::max)
        .ok_or("empty logits")? as f64;
    let partition = logits
        .iter()
        .map(|value| ((*value as f64) - maximum).exp())
        .sum::<f64>();
    let value = maximum + partition.ln() - logits[target] as f64;
    if !value.is_finite() || value < 0.0 {
        return Err("target NLL is nonfinite or negative".into());
    }
    Ok(value)
}

fn top_k(logits: &[f32], count: usize) -> Vec<usize> {
    let mut indices = (0..logits.len()).collect::<Vec<_>>();
    indices.select_nth_unstable_by(count - 1, |left, right| {
        logits[*right]
            .total_cmp(&logits[*left])
            .then_with(|| left.cmp(right))
    });
    indices.truncate(count);
    indices.sort_unstable_by(|left, right| {
        logits[*right]
            .total_cmp(&logits[*left])
            .then_with(|| left.cmp(right))
    });
    indices
}

fn load_tokens(path: &Path) -> Result<Vec<u32>, String> {
    let bytes = std::fs::read(path)
        .map_err(|error| format!("cannot read token fixture {}: {error}", path.display()))?;
    let text = std::str::from_utf8(&bytes)
        .map_err(|error| format!("token fixture is not UTF-8: {error}"))?;
    text.split(|character: char| character == ',' || character.is_ascii_whitespace())
        .filter(|field| !field.is_empty())
        .map(|field| {
            field
                .parse::<u32>()
                .map_err(|error| format!("invalid token {field:?}: {error}"))
        })
        .collect()
}

fn token_digest(tokens: &[u32]) -> String {
    let mut digest = Sha256::new();
    for token in tokens {
        digest.update(token.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn parse_args() -> Result<Args, String> {
    let argv = std::env::args().skip(1).collect::<Vec<_>>();
    let mut model = None;
    let mut token_fixture = None;
    let mut backend = Backend::Cpu;
    let mut top_k = 10usize;
    let mut logits_out = None;
    let mut render_text_out = None;
    let mut identity = "unsealed-local".to_string();
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
            "--token-fixture" => {
                token_fixture = Some(PathBuf::from(value("--token-fixture", index)?))
            }
            "--backend" => {
                backend = match value("--backend", index)?.as_str() {
                    "cpu" => Backend::Cpu,
                    "metal" => Backend::Metal,
                    other => return Err(format!("unknown backend {other:?}")),
                }
            }
            "--top-k" => {
                top_k = value("--top-k", index)?
                    .parse()
                    .map_err(|_| "--top-k must be an integer")?
            }
            "--logits-out" => logits_out = Some(PathBuf::from(value("--logits-out", index)?)),
            "--render-text-out" => {
                render_text_out = Some(PathBuf::from(value("--render-text-out", index)?))
            }
            "--identity" => identity = value("--identity", index)?,
            "--dry-run" => {
                dry_run = true;
                index += 1;
                continue;
            }
            unknown => return Err(format!("unknown argument {unknown:?}")),
        }
        index += 2;
    }
    if top_k == 0 {
        return Err("--top-k must be positive".into());
    }
    if render_text_out.is_some() && logits_out.is_some() {
        return Err("--render-text-out and --logits-out are mutually exclusive".into());
    }
    Ok(Args {
        model: model.ok_or("--model is required")?,
        token_fixture: token_fixture.ok_or("--token-fixture is required")?,
        backend,
        top_k,
        logits_out,
        render_text_out,
        identity,
        dry_run,
    })
}
