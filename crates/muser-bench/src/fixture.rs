//! Build an exact shared Muser/llama.cpp teacher corpus without touching an
//! accelerator. llama-perplexity adds BOS itself, so the text encodes to the
//! fixture tail while the decimal-u32 fixture explicitly carries BOS.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

use muser_engine::{Model, ModelConfig};
use sha2::{Digest, Sha256};

const SOURCES: &[(&str, &str)] = &[
    (
        "p1",
        "Muse Glimmer follows an exact instruction and reports the result plainly. The quick brown fox crosses the quiet valley. 0123456789.\n",
    ),
    (
        "p2",
        "A careful systems engineer checks every boundary twice: cache identity, logical position, and deterministic output all remain explicit.\n",
    ),
    (
        "p3",
        "In a small observatory, three instruments record the same event independently before their measurements are compared. Alpha beta gamma.\n",
    ),
    (
        "swa",
        // Plain repeated chants near-tie an end-of-text special against the
        // loop continuation at deep cycle boundaries (measured: muser +0.196
        // for 200007 vs llama +0.133 for 182290 at position 2067 - same
        // top-2, opposite order), so an exact-64 gate cannot sit on them.
        // The swa stream is therefore generated as numbered facts (see run());
        // this entry documents the template and keeps the id registered.
        "Ring fact 1: the window keeps the newest tokens and slot order never lies. ",
    ),
    (
        // Counted stream (see run()); this entry documents the template. The
        // UTF-8 words are load-bearing: the case exists to cross tokenizer
        // byte-fallback and multi-byte merges at depth.
        "long",
        "Chapter 1: Zürich stays exact and naïve stays naïve across numbered chapters. ",
    ),
];

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("muser-token-fixture: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut model_path = None;
    let mut tokens_out = None;
    let mut corpus_out = None;
    let mut count = 66usize;
    let mut identity = "unsealed-local".to_string();
    let mut fixture_id = "p1".to_string();
    let mut dry_run = false;
    let argv = std::env::args().skip(1).collect::<Vec<_>>();
    let mut index = 0;
    while index < argv.len() {
        let value = |name: &str, index: usize| {
            argv.get(index + 1)
                .cloned()
                .ok_or_else(|| format!("{name} requires a value"))
        };
        match argv[index].as_str() {
            "--model" => model_path = Some(PathBuf::from(value("--model", index)?)),
            "--tokens-out" => tokens_out = Some(PathBuf::from(value("--tokens-out", index)?)),
            "--corpus-out" => corpus_out = Some(PathBuf::from(value("--corpus-out", index)?)),
            "--count" => {
                count = value("--count", index)?
                    .parse()
                    .map_err(|_| "--count must be an integer")?
            }
            "--identity" => identity = value("--identity", index)?,
            "--fixture-id" => fixture_id = value("--fixture-id", index)?,
            "--dry-run" => {
                dry_run = true;
                index += 1;
                continue;
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
        index += 2;
    }
    let model_path = model_path.ok_or("--model is required")?;
    let tokens_out = tokens_out.ok_or("--tokens-out is required")?;
    let corpus_out = corpus_out.ok_or("--corpus-out is required")?;
    if count == 0 {
        return Err("--count must be positive".into());
    }
    let source = SOURCES
        .iter()
        .find_map(|(id, source)| (*id == fixture_id).then_some(*source))
        .ok_or("--fixture-id must be p1, p2, p3, swa, or long")?;
    if dry_run {
        println!(
            "{}",
            serde_json::json!({
                "schema": "muser.token-fixture.v1",
                "kind": "dry-run",
                "accelerator_touched": false,
                "model": model_path,
                "tokens_out": tokens_out,
                "corpus_out": corpus_out,
                "count": count,
                "identity": identity,
                "fixture_id": fixture_id,
            })
        );
        return Ok(());
    }
    let model = Model::load(ModelConfig::new(model_path)).map_err(|error| error.to_string())?;
    let bos = model
        .config()
        .bos_token_id
        .ok_or("Muse model has no BOS token")?;
    // Counted streams: an incrementing counter makes every greedy
    // continuation near-forced, so the exact-64 cross-engine gates never sit
    // on a marginal continue-vs-end ranking at depth (measured: plain chants
    // near-tie an end-of-text special around cycle boundaries - swa at
    // position 2067 by 0.13-0.20 logits, long-8192 at index 14 by 0.027).
    let source = match fixture_id.as_str() {
        "swa" => {
            let mut text = String::with_capacity(count * 64);
            for index in 1..=count {
                text.push_str(&format!(
                    "Ring fact {index}: the window keeps the newest tokens and slot order never lies. "
                ));
            }
            text
        }
        "long" => {
            let mut text = String::with_capacity(count * 64);
            for index in 1..=count {
                text.push_str(&format!(
                    "Chapter {index}: Zürich stays exact and naïve stays naïve across numbered chapters. "
                ));
            }
            text
        }
        _ => source.repeat(count),
    };
    let encoded = model.encode(&source);
    if encoded.len() < count - 1 {
        return Err("fixed corpus did not produce enough tokens".into());
    }
    let tail = &encoded[..count - 1];
    let corpus = model.decode_tokens(tail);
    if model.encode(&corpus) != tail {
        return Err("fixed corpus truncation is not an exact tokenizer round trip".into());
    }
    let mut tokens = Vec::with_capacity(count);
    tokens.push(bos);
    tokens.extend_from_slice(tail);
    let token_bytes = tokens
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    publish(&tokens_out, token_bytes.as_bytes())?;
    publish(&corpus_out, corpus.as_bytes())?;
    let mut semantic = Sha256::new();
    for token in &tokens {
        semantic.update(token.to_le_bytes());
    }
    println!(
        "{}",
        serde_json::json!({
            "schema": "muser.token-fixture.v1",
            "kind": "fixture",
            "identity": identity,
            "fixture_id": fixture_id,
            "count": count,
            "bos_token": bos,
            "tokens_sha256": format!("{:x}", semantic.finalize()),
            "tokens_file_sha256": format!("{:x}", Sha256::digest(token_bytes.as_bytes())),
            "corpus_sha256": format!("{:x}", Sha256::digest(corpus.as_bytes())),
            "exact_tail_round_trip": true,
        })
    );
    Ok(())
}

fn publish(path: &std::path::Path, payload: &[u8]) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("cannot create {}: {error}", path.display()))?;
    file.write_all(payload)
        .and_then(|_| file.flush())
        .map_err(|error| error.to_string())?;
    file.sync_all().map_err(|error| error.to_string())
}
