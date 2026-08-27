//! Real-model resident/durable exact-prefix qualification.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;
use std::{fs::File, io::Read};

use kvpack::{create_store_key_random, load_store_key, LocalStore, StoreConfig};
use muser_engine::{DecodeInput, Model, ModelConfig, PrefillBatch, Session};
use muser_kvpack::layout::MuseIdentity;
use muser_kvpack::remote::{LoopbackRemoteStore, RemoteCache};
use muser_kvpack::reuse::{CacheSource, PrefixReuse};
use muser_kvpack::session::DurableCache;
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
enum Source {
    Resident,
    Durable,
    Remote,
}

#[derive(Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Lookup {
    ExactFinal,
    DeepestAncestor,
}

struct Args {
    model: PathBuf,
    prompt_fixture: PathBuf,
    source: Source,
    lookup: Lookup,
    suffix: usize,
    repetitions: usize,
    store_root: Option<PathBuf>,
    resident_capacity_bytes: u64,
    identity: String,
    dry_run: bool,
}

#[derive(Serialize)]
struct Sample<'a> {
    schema: &'static str,
    kind: &'static str,
    identity: &'a str,
    source: Source,
    lookup: Lookup,
    repetition: usize,
    prompt_tokens: usize,
    published_cut: usize,
    matched_tokens: usize,
    suffix_tokens: usize,
    restore_to_first_logits_ns: u64,
    full_recompute_ns: u64,
    token_ids: &'a [u32],
    full_logit_digest: &'a str,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("muser-kvpack-qualify: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let planned_cut = match args.lookup {
        Lookup::ExactFinal => "prompt",
        Lookup::DeepestAncestor => "prompt-minus-suffix",
    };
    if args.dry_run {
        println!(
            "{}",
            serde_json::json!({
                "schema": "muser.kvpack-qualify.v1",
                "kind": "dry-run",
                "accelerator_touched": false,
                "model": args.model,
                "prompt_fixture": args.prompt_fixture,
                "source": args.source,
                "lookup": args.lookup,
                "suffix": args.suffix,
                "published_cut": planned_cut,
                "repetitions": args.repetitions,
                "store_root": args.store_root,
                "resident_capacity_bytes": args.resident_capacity_bytes,
                "identity": args.identity,
                "backend": "metal",
                "output_tokens": 64,
                "seal_eligible": false,
            })
        );
        return Ok(());
    }
    if std::env::var("MUSER_ACCELERATOR_LEASE").as_deref() != Ok("1") {
        return Err("execution must be a child of accelerator_safe.py".into());
    }
    let prompt_bytes = std::fs::read(&args.prompt_fixture)
        .map_err(|error| format!("cannot read {}: {error}", args.prompt_fixture.display()))?;
    let prompt = parse_tokens(&prompt_bytes)?;
    if prompt.len() < 2 {
        return Err("kvpack prompt requires at least two tokens".into());
    }
    let published_cut = match args.lookup {
        Lookup::ExactFinal => prompt.len(),
        Lookup::DeepestAncestor => prompt
            .len()
            .checked_sub(args.suffix)
            .ok_or("suffix exceeds prompt")?,
    };
    if published_cut == 0 {
        return Err("published cut must be positive".into());
    }
    if args.lookup == Lookup::DeepestAncestor && args.suffix == 0 {
        return Err("deepest-ancestor lookup requires a nonzero suffix".into());
    }
    if args.source == Source::Resident
        && args.lookup == Lookup::DeepestAncestor
        && !published_cut.is_multiple_of(256)
    {
        return Err("resident ancestor cut must align to 256 tokens".into());
    }
    if matches!(args.source, Source::Durable | Source::Remote) && args.store_root.is_none() {
        return Err("durable/remote qualification requires --store-root".into());
    }
    let model = Model::load(ModelConfig::new(&args.model)).map_err(|error| error.to_string())?;
    if prompt.len() + 64 > model.config().context_length
        || prompt
            .iter()
            .any(|token| *token as usize >= model.config().vocab_size)
    {
        return Err("prompt geometry is outside the Muse model contract".into());
    }
    let mut reuse =
        PrefixReuse::new(args.resident_capacity_bytes).map_err(|error| error.to_string())?;
    let mut remote_source_cache = None;
    if matches!(args.source, Source::Durable | Source::Remote) {
        let root = args.store_root.as_deref().expect("checked");
        if root.exists() {
            return Err(format!(
                "refusing to reuse durable store root {}",
                root.display()
            ));
        }
        std::fs::create_dir_all(root).map_err(|error| error.to_string())?;
        let key_path = root.join("keys/root.key");
        create_store_key_random(&key_path, root).map_err(|error| error.to_string())?;
        let store_root = if args.source == Source::Remote {
            root.join("source")
        } else {
            root.to_path_buf()
        };
        let store = Arc::new(
            LocalStore::open(
                StoreConfig {
                    object_root: store_root.join("objects"),
                    catalog_path: store_root.join("catalog/catalog.sqlite"),
                    operator_tenant_id: b"muser-kvpack-qualification-v1".to_vec(),
                    key_epoch: 1,
                    minimum_readable_key_epoch: 1,
                    catalog_epoch: 1,
                    quota_bytes: 16 * 1024 * 1024 * 1024,
                    staging_quota_bytes: 8 * 1024 * 1024 * 1024,
                    endurance_bytes_per_five_minutes: 64 * 1024 * 1024 * 1024,
                },
                load_store_key(&key_path, root).map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.to_string())?,
        );
        let model_digest: [u8; 32] = file_sha256(&args.model)?;
        let domain = |label: &[u8]| -> [u8; 32] {
            let mut digest = Sha256::new();
            digest.update(label);
            digest.update(model_digest);
            digest.finalize().into()
        };
        let durable = DurableCache::new(
            Arc::clone(&store),
            model.config().clone(),
            MuseIdentity {
                model_sha256: model_digest,
                adapter_sha256: [0; 32],
                tokenizer_sha256: domain(b"muser/tokenizer-from-gguf/v1\0"),
                chat_template_sha256: domain(b"muser/chat-template-from-gguf/v1\0"),
                context_policy_sha256: domain(b"muser/context-policy-131072-swa2048/v1\0"),
                model_revision: args.identity.clone(),
                tokenizer_revision: args.identity.clone(),
                weight_precision: "q4_k_xl".into(),
            },
        );
        if args.source == Source::Durable {
            reuse.set_durable(durable);
        } else {
            let staging_root = root.join("receiver");
            let staging_store = Arc::new(
                LocalStore::open(
                    StoreConfig {
                        object_root: staging_root.join("objects"),
                        catalog_path: staging_root.join("catalog/catalog.sqlite"),
                        operator_tenant_id: b"muser-kvpack-qualification-v1".to_vec(),
                        key_epoch: 1,
                        minimum_readable_key_epoch: 1,
                        catalog_epoch: 1,
                        quota_bytes: 16 * 1024 * 1024 * 1024,
                        staging_quota_bytes: 8 * 1024 * 1024 * 1024,
                        endurance_bytes_per_five_minutes: 64 * 1024 * 1024 * 1024,
                    },
                    load_store_key(&key_path, root).map_err(|error| error.to_string())?,
                )
                .map_err(|error| error.to_string())?,
            );
            let staging = DurableCache::new(
                staging_store,
                model.config().clone(),
                durable.identity().clone(),
            );
            let authority = Arc::new(
                LoopbackRemoteStore::new(Arc::clone(&store), 1)
                    .map_err(|error| error.to_string())?,
            );
            reuse.set_remote(RemoteCache::new(authority, staging));
            remote_source_cache = Some(durable);
        }
    }

    let mut miss_session = new_metal_session(&model, prompt.len() + 64)?;
    let miss_started = Instant::now();
    let miss = reuse
        .prepare(&mut miss_session, &prompt)
        .map_err(|error| error.to_string())?;
    let miss_lookup_ns = nanos(miss_started.elapsed().as_nanos());
    if miss.source != CacheSource::Miss || miss.matched_tokens != 0 {
        return Err("empty prefix authority returned a semantic false positive".into());
    }

    let mut source_session = new_metal_session(&model, prompt.len() + 64)?;
    let source_started = Instant::now();
    prefill_chunked(&mut source_session, &prompt[..published_cut])?;
    let source_prefill_ns = nanos(source_started.elapsed().as_nanos());
    let publish_started = Instant::now();
    match args.source {
        Source::Resident => {
            if !reuse
                .publish_resident(&source_session)
                .map_err(|error| error.to_string())?
            {
                return Err("resident source was immediately evicted".into());
            }
        }
        Source::Durable => {
            if !reuse
                .publish_durable(&source_session, 1)
                .map_err(|error| error.to_string())?
            {
                return Err("durable source was not published".into());
            }
        }
        Source::Remote => {
            remote_source_cache
                .as_ref()
                .ok_or("remote durable source was not configured")?
                .save(&source_session, [11; 32], 1)
                .map_err(|error| error.to_string())?;
        }
    }
    let publication_ns = nanos(publish_started.elapsed().as_nanos());

    let mut reference = new_metal_session(&model, prompt.len() + 64)?;
    let full_started = Instant::now();
    let reference_logits = prefill_chunked(&mut reference, &prompt)?;
    let full_recompute_ns = nanos(full_started.elapsed().as_nanos());
    let (reference_tokens, reference_digest) =
        generate64(&model, &mut reference, reference_logits)?;

    let mut restore_samples = Vec::with_capacity(args.repetitions);
    for repetition in 0..args.repetitions {
        let mut restored = new_metal_session(&model, prompt.len() + 64)?;
        let started = Instant::now();
        let hit = reuse
            .prepare(&mut restored, &prompt)
            .map_err(|error| error.to_string())?;
        let expected_source = match args.source {
            Source::Resident => CacheSource::Resident,
            Source::Durable => CacheSource::Durable,
            Source::Remote => CacheSource::Remote,
        };
        if hit.source != expected_source || hit.matched_tokens != published_cut {
            return Err(format!(
                "restore route mismatch: source={:?} matched={}",
                hit.source, hit.matched_tokens
            ));
        }
        let logits = if hit.matched_tokens == prompt.len() {
            restored
                .cached_logits()
                .ok_or("resident exact hit omitted final logits")?
                .to_vec()
        } else {
            prefill_chunked(&mut restored, &prompt[hit.matched_tokens..])?
        };
        let elapsed = nanos(started.elapsed().as_nanos());
        let (tokens, digest) = generate64(&model, &mut restored, logits)?;
        if tokens != reference_tokens || digest != reference_digest {
            return Err(format!(
                "restored generation differs at repetition {repetition}"
            ));
        }
        restore_samples.push(elapsed);
        println!(
            "{}",
            serde_json::to_string(&Sample {
                schema: "muser.kvpack-qualify.v1",
                kind: "sample",
                identity: &args.identity,
                source: args.source,
                lookup: args.lookup,
                repetition,
                prompt_tokens: prompt.len(),
                published_cut,
                matched_tokens: hit.matched_tokens,
                suffix_tokens: prompt.len() - hit.matched_tokens,
                restore_to_first_logits_ns: elapsed,
                full_recompute_ns,
                token_ids: &tokens,
                full_logit_digest: &digest,
            })
            .map_err(|error| error.to_string())?
        );
    }
    let mean = restore_samples
        .iter()
        .map(|value| *value as f64)
        .sum::<f64>()
        / restore_samples.len() as f64;
    let variance = restore_samples
        .iter()
        .map(|value| (*value as f64 - mean).powi(2))
        .sum::<f64>()
        / restore_samples.len() as f64;
    println!(
        "{}",
        serde_json::json!({
            "schema": "muser.kvpack-qualify.v1",
            "kind": "summary",
            "identity": args.identity,
            "source": args.source,
            "lookup": args.lookup,
            "prompt_tokens": prompt.len(),
            "published_cut": published_cut,
            "suffix_tokens": prompt.len() - published_cut,
            "raw_restore_ns": restore_samples,
            "restore_cv": if mean == 0.0 { 0.0 } else { variance.sqrt() / mean },
            "full_recompute_ns": full_recompute_ns,
            "source_prefill_ns": source_prefill_ns,
            "publication_ns": publication_ns,
            "publication_overhead_ratio": publication_ns as f64 / source_prefill_ns as f64,
            "miss_lookup_ns": miss_lookup_ns,
            "miss_overhead_ratio": miss_lookup_ns as f64 / full_recompute_ns as f64,
            "speedup_geomean_cell": full_recompute_ns as f64 / mean,
            "generated_tokens_sha256": token_digest(&reference_tokens),
            "full_logit_digest": reference_digest,
            "correctness": "exact-64-tokens-and-all-step-full-logit-digest",
            "seal_eligible": publication_ns as f64 <= source_prefill_ns as f64 * 0.05
                && miss_lookup_ns as f64 <= full_recompute_ns as f64 * 0.02
                && full_recompute_ns as f64 > mean,
        })
    );
    Ok(())
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn new_metal_session(model: &Model, max_context: usize) -> Result<Session, String> {
    model
        .new_metal_session(muser_engine::SessionConfig { max_context })
        .map_err(|error| error.to_string())
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn new_metal_session(_model: &Model, _max_context: usize) -> Result<Session, String> {
    Err("kvpack qualification requires macOS and the metal feature".into())
}

fn prefill_chunked(session: &mut Session, tokens: &[u32]) -> Result<Vec<f32>, String> {
    let mut logits = None;
    for chunk in tokens.chunks(2_048) {
        logits = Some(
            session
                .prefill(PrefillBatch::tokens(chunk.to_vec()))
                .map_err(|error| error.to_string())?
                .last_logits()
                .to_vec(),
        );
    }
    logits.ok_or_else(|| "cannot prefill an empty token suffix".into())
}

fn generate64(
    model: &Model,
    session: &mut Session,
    mut logits: Vec<f32>,
) -> Result<(Vec<u32>, String), String> {
    let mut tokens = Vec::with_capacity(64);
    let mut digest = Sha256::new();
    for index in 0..64 {
        if logits.len() != model.config().vocab_size || logits.iter().any(|x| !x.is_finite()) {
            return Err(format!("invalid logits at generation row {index}"));
        }
        for value in &logits {
            digest.update(value.to_bits().to_le_bytes());
        }
        let token = logits
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1).then_with(|| right.0.cmp(&left.0)))
            .map(|(token, _)| token as u32)
            .ok_or("empty logits")?;
        if model.config().eos_tokens.contains(&token) {
            return Err(format!("early EOS at generation row {index}"));
        }
        tokens.push(token);
        if index != 63 {
            logits = session
                .decode(DecodeInput { token_id: token })
                .map_err(|error| error.to_string())?
                .logits;
        }
    }
    Ok((tokens, format!("{:x}", digest.finalize())))
}

fn parse_tokens(bytes: &[u8]) -> Result<Vec<u32>, String> {
    std::str::from_utf8(bytes)
        .map_err(|error| error.to_string())?
        .split_ascii_whitespace()
        .map(|value| value.parse::<u32>().map_err(|error| error.to_string()))
        .collect()
}

fn file_sha256(path: &Path) -> Result<[u8; 32], String> {
    let mut file = File::open(path).map_err(|error| error.to_string())?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 8 << 20];
    loop {
        let read = file.read(&mut buffer).map_err(|error| error.to_string())?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(digest.finalize().into())
}

fn token_digest(tokens: &[u32]) -> String {
    let mut digest = Sha256::new();
    for token in tokens {
        digest.update(token.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn nanos(value: u128) -> u64 {
    value.min(u64::MAX as u128) as u64
}

fn parse_args() -> Result<Args, String> {
    let argv = std::env::args().skip(1).collect::<Vec<_>>();
    let mut model = None;
    let mut prompt_fixture = None;
    let mut source = None;
    let mut lookup = None;
    let mut suffix = 0usize;
    let mut repetitions = 3usize;
    let mut store_root = None;
    let mut resident_capacity_bytes = 8 * 1024 * 1024 * 1024u64;
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
            "--prompt-token-fixture" => {
                prompt_fixture = Some(PathBuf::from(value("--prompt-token-fixture", index)?))
            }
            "--source" => {
                source = Some(match value("--source", index)?.as_str() {
                    "resident" => Source::Resident,
                    "durable" => Source::Durable,
                    "remote" => Source::Remote,
                    other => return Err(format!("unknown source {other:?}")),
                })
            }
            "--lookup" => {
                lookup = Some(match value("--lookup", index)?.as_str() {
                    "exact-final" => Lookup::ExactFinal,
                    "deepest-ancestor" => Lookup::DeepestAncestor,
                    other => return Err(format!("unknown lookup {other:?}")),
                })
            }
            "--suffix" => {
                suffix = value("--suffix", index)?
                    .parse()
                    .map_err(|_| "--suffix must be an integer")?
            }
            "--repetitions" => {
                repetitions = value("--repetitions", index)?
                    .parse()
                    .map_err(|_| "--repetitions must be an integer")?
            }
            "--store-root" => store_root = Some(PathBuf::from(value("--store-root", index)?)),
            "--resident-capacity-bytes" => {
                resident_capacity_bytes = value("--resident-capacity-bytes", index)?
                    .parse()
                    .map_err(|_| "--resident-capacity-bytes must be an integer")?
            }
            "--identity" => identity = value("--identity", index)?,
            "--dry-run" => {
                dry_run = true;
                index += 1;
                continue;
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
        index += 2;
    }
    if repetitions != 3 {
        return Err("kvpack qualification requires exactly three repetitions".into());
    }
    Ok(Args {
        model: model.ok_or("--model is required")?,
        prompt_fixture: prompt_fixture.ok_or("--prompt-token-fixture is required")?,
        source: source.ok_or("--source is required")?,
        lookup: lookup.ok_or("--lookup is required")?,
        suffix,
        repetitions,
        store_root,
        resident_capacity_bytes,
        identity,
        dry_run,
    })
}
