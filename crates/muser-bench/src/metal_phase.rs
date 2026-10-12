//! One-shot, non-notarial Metal phase diagnostic.
//!
//! This binary deliberately has no warm-up or repetition loop. After strict
//! artifact and fixture validation it prefills the pinned prompt, calls the
//! production one-row `decode` graph exactly once, restores that exact prefix,
//! then calls the legacy `teacher_forced_decode` graph exactly once. The two
//! engine phase profilers print separately bracketed reports to stderr.

use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use muser_engine::decode::MetalMuseModel;
use muser_engine::{DecodeInput, Model, ModelConfig, PrefillBatch, SessionConfig};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const RELEASE_ARTIFACTS: &str = include_str!("../../../docs/release-artifacts.json");
const PROMPT_CAPTURE_BOUNDARIES: [&str; 21] = [
    "embedding",
    "entry_norm",
    "attn_norm-0",
    "Qcur-0",
    "Kcur-0",
    "Vcur-0",
    "attn_gate_proj-0",
    "Qcur_normed-0",
    "Kcur_normed-0",
    "Qcur_rope-0",
    "Kcur_rope-0",
    "attn_out-0",
    "attn_gated-0",
    "attn_o_proj-0",
    "ffn_inp-0",
    "ffn_norm-0",
    "ffn_gate-0",
    "ffn_up-0",
    "ffn_swiglu-0",
    "ffn_out-0",
    "l_out-0",
];

#[derive(Debug)]
struct Args {
    model: PathBuf,
    prompt: PathBuf,
    prompt_positions: usize,
    prompt_file_sha256: String,
    prompt_tokens_sha256: String,
    teacher_token: u32,
    production_logits_out: Option<PathBuf>,
    capture_layer: Option<usize>,
    capture_dir: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArtifactManifest {
    schema: String,
    revision: String,
    repository: String,
    artifacts: Artifacts,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Artifacts {
    target: Artifact,
    vision: Artifact,
    dflash: Artifact,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Artifact {
    filename: String,
    revision: String,
    url: String,
    bytes: u64,
    sha256: String,
}

#[derive(Serialize)]
struct ResultRecord<'a> {
    schema: &'static str,
    kind: &'static str,
    purpose: &'static str,
    model_revision: &'a str,
    model_sha256: &'a str,
    model_bytes: u64,
    artifact_manifest_sha256: String,
    prompt_file_sha256: &'a str,
    prompt_tokens_sha256: &'a str,
    prompt_positions: usize,
    teacher_token: u32,
    teacher_token_witness_sha256: String,
    production_graph: &'static str,
    production_elapsed_ns: u64,
    production_next_token: u32,
    production_logits_sha256: String,
    production_logits_path: Option<String>,
    production_logits_bytes: Option<usize>,
    capture_layer: Option<usize>,
    capture_files: Option<usize>,
    production_decode_calls: u8,
    legacy_graph: &'static str,
    legacy_elapsed_ns: u64,
    legacy_teacher_forced_calls: u8,
    batch_phase_profile_stderr: bool,
    legacy_phase_profile_stderr: bool,
    accelerator_touched: bool,
    notarial: bool,
    qualification_eligible: bool,
    readiness_eligible: bool,
    seal_eligible: bool,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("muser-metal-phase-diagnostic: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let args = parse_args(std::env::args().skip(1))?;
    let live_trace = match std::env::var_os("MUSER_METAL_LIVE_TRACE") {
        None => false,
        Some(value) if value == "1" => true,
        Some(_) => return Err("MUSER_METAL_LIVE_TRACE must be absent or exactly 1".into()),
    };
    let phase_profile = std::env::var_os("MUSER_METAL_PHASE_PROFILE");
    let batch_phase_profile = std::env::var_os("MUSER_METAL_BATCH_PHASE_PROFILE");
    if live_trace {
        if phase_profile.is_some() || batch_phase_profile.is_some() {
            return Err("live trace mode forbids the isolated phase profilers".into());
        }
    } else if phase_profile.as_deref() != Some(std::ffi::OsStr::new("1"))
        || batch_phase_profile.as_deref() != Some(std::ffi::OsStr::new("1"))
    {
        return Err("both Metal phase profiler variables must be exactly 1".into());
    }

    let manifest = parse_manifest()?;
    validate_manifest(&manifest)?;
    validate_regular_file(&args.model, "model")?;
    validate_regular_file(&args.prompt, "prompt fixture")?;
    let metadata = std::fs::metadata(&args.model)
        .map_err(|error| format!("stat model {}: {error}", args.model.display()))?;
    if metadata.len() != manifest.artifacts.target.bytes {
        return Err(format!(
            "model size {} differs from pinned {}",
            metadata.len(),
            manifest.artifacts.target.bytes
        ));
    }
    let model_sha256 = sha256_file(&args.model)?;
    if model_sha256 != manifest.artifacts.target.sha256 {
        return Err(format!(
            "model SHA-256 {model_sha256} differs from pinned {}",
            manifest.artifacts.target.sha256
        ));
    }

    let prompt_bytes = std::fs::read(&args.prompt)
        .map_err(|error| format!("read prompt fixture {}: {error}", args.prompt.display()))?;
    let prompt_file_sha256 = sha256_bytes(&prompt_bytes);
    if prompt_file_sha256 != args.prompt_file_sha256 {
        return Err(format!(
            "prompt file SHA-256 {prompt_file_sha256} differs from expected {}",
            args.prompt_file_sha256
        ));
    }
    let tokens = parse_fixture_bytes(&prompt_bytes, &args.prompt)?;
    if tokens.len() != args.prompt_positions {
        return Err(format!(
            "prompt fixture has {} positions, expected {}",
            tokens.len(),
            args.prompt_positions
        ));
    }
    let prompt_tokens_sha256 = digest_tokens(&tokens);
    if prompt_tokens_sha256 != args.prompt_tokens_sha256 {
        return Err(format!(
            "prompt token SHA-256 {prompt_tokens_sha256} differs from expected {}",
            args.prompt_tokens_sha256
        ));
    }
    let max_context = tokens
        .len()
        .checked_add(1)
        .ok_or("prompt context length overflow")?;

    let capture_files =
        if let (Some(layer), Some(directory)) = (args.capture_layer, args.capture_dir.as_ref()) {
            validate_empty_real_directory(directory)?;
            let loaded = muser_engine::loader::load_components(&args.model)
                .map_err(|error| error.to_string())?;
            if layer >= loaded.config.n_layers {
                return Err(format!("capture layer {layer} is outside the model"));
            }
            let mut diagnostic = MetalMuseModel::new(loaded.config, loaded.weights, max_context)
                .map_err(|error| error.to_string())?;
            let prompt_capture = diagnostic
                .forward_capturing_debug_layer(&tokens[..512], layer)
                .map_err(|error| error.to_string())?;
            for (name, values) in prompt_capture
                .iter()
                .filter(|(name, _)| PROMPT_CAPTURE_BOUNDARIES.contains(name))
            {
                write_logits_exclusive(&directory.join(format!("prompt.{name}.f32")), values)?;
            }
            diagnostic
                .forward(&tokens[512..])
                .map_err(|error| error.to_string())?;
            let captures = diagnostic
                .forward_capturing_debug_layer(&[args.teacher_token], layer)
                .map_err(|error| error.to_string())?;
            for (name, values) in &captures {
                write_logits_exclusive(&directory.join(format!("{name}.f32")), values)?;
            }
            Some(captures.len())
        } else {
            None
        };

    let model = Model::load(ModelConfig::new(&args.model)).map_err(|error| error.to_string())?;
    let vocab_size = model.config().vocab_size;
    if let Some((index, token)) = tokens
        .iter()
        .enumerate()
        .find(|(_, token)| **token as usize >= vocab_size)
    {
        return Err(format!(
            "prompt token {index}={token} is outside vocabulary 0..{vocab_size}"
        ));
    }
    if args.teacher_token as usize >= vocab_size {
        return Err(format!(
            "teacher token {} is outside vocabulary 0..{vocab_size}",
            args.teacher_token
        ));
    }
    if max_context > model.config().context_length {
        return Err(format!(
            "prompt plus teacher token requires {max_context} positions, model supports {}",
            model.config().context_length
        ));
    }

    let mut session = model
        .new_metal_session(SessionConfig { max_context })
        .map_err(|error| error.to_string())?;
    for chunk in tokens.chunks(2_048) {
        session
            .prefill(PrefillBatch::tokens(chunk.to_vec()))
            .map_err(|error| error.to_string())?;
    }
    let prefix = session
        .export_cache_snapshot()
        .map_err(|error| error.to_string())?;
    if let (Some(layer), Some(directory)) = (args.capture_layer, args.capture_dir.as_ref()) {
        let plane = prefix
            .layers
            .get(layer)
            .ok_or("captured cache layer is absent")?;
        write_bytes_exclusive(&directory.join("prefix.key.f16"), &plane.key)?;
        write_bytes_exclusive(&directory.join("prefix.value.f16"), &plane.value)?;
    }
    capture_rendezvous(live_trace)?;
    eprintln!(
        "muser-metal-phase-diagnostic: graph=production-forward-batch-one-row phase=start position={}",
        session.position()
    );
    let production_started = Instant::now();
    let production = session
        .decode(DecodeInput {
            token_id: args.teacher_token,
        })
        .map_err(|error| error.to_string())?;
    let production_elapsed_ns =
        u64::try_from(production_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    if production.input_token != args.teacher_token || session.position() != max_context {
        return Err("one-row production decode produced an invalid state witness".into());
    }
    if let Some(path) = args.production_logits_out.as_ref() {
        write_logits_exclusive(path, &production.logits)?;
    }
    eprintln!("muser-metal-phase-diagnostic: graph=production-forward-batch-one-row phase=done");

    session
        .install_cache_snapshot(&prefix)
        .map_err(|error| error.to_string())?;
    if session.position() != tokens.len() {
        return Err("restored comparison prefix has the wrong position".into());
    }
    eprintln!(
        "muser-metal-phase-diagnostic: graph=legacy-forward-token phase=start position={}",
        session.position()
    );
    let legacy_started = Instant::now();
    let witness = session
        .teacher_forced_decode(&[args.teacher_token])
        .map_err(|error| error.to_string())?;
    let legacy_elapsed_ns = u64::try_from(legacy_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    if witness != [args.teacher_token] || session.position() != max_context {
        return Err("one-token teacher-forced state transition produced an invalid witness".into());
    }
    eprintln!("muser-metal-phase-diagnostic: graph=legacy-forward-token phase=done");

    let record = ResultRecord {
        schema: "muser.metal-phase-diagnostic.v1",
        kind: "non-notarial-diagnostic",
        purpose: "production-versus-legacy-one-token-metal-phase-attribution-only",
        model_revision: &manifest.revision,
        model_sha256: &model_sha256,
        model_bytes: metadata.len(),
        artifact_manifest_sha256: sha256_bytes(RELEASE_ARTIFACTS.as_bytes()),
        prompt_file_sha256: &prompt_file_sha256,
        prompt_tokens_sha256: &prompt_tokens_sha256,
        prompt_positions: tokens.len(),
        teacher_token: args.teacher_token,
        teacher_token_witness_sha256: digest_tokens(&witness),
        production_graph: "forward-batch-one-row",
        production_elapsed_ns,
        production_next_token: production.next_token,
        production_logits_sha256: digest_logits(&production.logits),
        production_logits_path: args
            .production_logits_out
            .as_ref()
            .map(|path| path.display().to_string()),
        production_logits_bytes: args
            .production_logits_out
            .as_ref()
            .map(|_| production.logits.len() * std::mem::size_of::<f32>()),
        capture_layer: args.capture_layer,
        capture_files,
        production_decode_calls: 1,
        legacy_graph: "forward-token-teacher-forced-one-row",
        legacy_elapsed_ns,
        legacy_teacher_forced_calls: 1,
        batch_phase_profile_stderr: !live_trace,
        legacy_phase_profile_stderr: !live_trace,
        accelerator_touched: true,
        notarial: false,
        qualification_eligible: false,
        readiness_eligible: false,
        seal_eligible: false,
    };
    println!(
        "{}",
        serde_json::to_string(&record).map_err(|error| error.to_string())?
    );
    Ok(())
}

fn capture_rendezvous(live_trace: bool) -> Result<(), String> {
    let Some(path) = std::env::var_os("MUSER_METAL_CAPTURE_READY_FILE") else {
        if std::env::var_os("MUSER_METAL_CAPTURE_PAUSE_MS").is_some() {
            return Err(
                "MUSER_METAL_CAPTURE_PAUSE_MS requires MUSER_METAL_CAPTURE_READY_FILE".into(),
            );
        }
        return if live_trace {
            Err("live trace mode requires a capture-ready rendezvous".into())
        } else {
            Ok(())
        };
    };
    if !live_trace {
        return Err("capture-ready rendezvous requires MUSER_METAL_LIVE_TRACE=1".into());
    }
    let path = PathBuf::from(path);
    let pause_ms = std::env::var("MUSER_METAL_CAPTURE_PAUSE_MS")
        .map_err(|_| "MUSER_METAL_CAPTURE_PAUSE_MS is required for capture rendezvous")?
        .parse::<u64>()
        .map_err(|_| "MUSER_METAL_CAPTURE_PAUSE_MS must be an integer")?;
    if !(1_000..=30_000).contains(&pause_ms) {
        return Err("MUSER_METAL_CAPTURE_PAUSE_MS must be in 1000..=30000".into());
    }
    let parent = path
        .parent()
        .filter(|value| !value.as_os_str().is_empty())
        .ok_or("MUSER_METAL_CAPTURE_READY_FILE must have a parent directory")?;
    let parent_metadata = std::fs::symlink_metadata(parent)
        .map_err(|error| format!("stat capture-ready parent {}: {error}", parent.display()))?;
    if parent_metadata.file_type().is_symlink() || !parent_metadata.is_dir() {
        return Err(format!(
            "capture-ready parent {} must be a real directory",
            parent.display()
        ));
    }
    let mut marker = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|error| format!("create capture-ready marker {}: {error}", path.display()))?;
    marker
        .write_all(b"{\"schema\":\"muser.metal-capture-ready.v1\",\"ready\":true}\n")
        .map_err(|error| format!("write capture-ready marker {}: {error}", path.display()))?;
    marker
        .sync_all()
        .map_err(|error| format!("fsync capture-ready marker {}: {error}", path.display()))?;
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| format!("fsync capture-ready parent {}: {error}", parent.display()))?;
    eprintln!(
        "muser-metal-phase-diagnostic: capture-ready={} pause_ms={pause_ms}",
        path.display()
    );
    std::thread::sleep(Duration::from_millis(pause_ms));
    Ok(())
}

fn parse_args(argv: impl Iterator<Item = String>) -> Result<Args, String> {
    let argv = argv.collect::<Vec<_>>();
    let mut model = None;
    let mut prompt = None;
    let mut prompt_positions = None;
    let mut prompt_file_sha256 = None;
    let mut prompt_tokens_sha256 = None;
    let mut teacher_token = None;
    let mut production_logits_out = None;
    let mut capture_layer = None;
    let mut capture_dir = None;
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
                prompt = Some(PathBuf::from(value("--prompt-token-fixture", index)?))
            }
            "--prompt-positions" => {
                prompt_positions = Some(parse_positive_usize(
                    &value("--prompt-positions", index)?,
                    "--prompt-positions",
                )?)
            }
            "--prompt-file-sha256" => {
                prompt_file_sha256 = Some(parse_sha256(
                    &value("--prompt-file-sha256", index)?,
                    "--prompt-file-sha256",
                )?)
            }
            "--prompt-tokens-sha256" => {
                prompt_tokens_sha256 = Some(parse_sha256(
                    &value("--prompt-tokens-sha256", index)?,
                    "--prompt-tokens-sha256",
                )?)
            }
            "--teacher-token" => {
                teacher_token = Some(
                    value("--teacher-token", index)?
                        .parse::<u32>()
                        .map_err(|error| format!("--teacher-token must be u32: {error}"))?,
                )
            }
            "--production-logits-out" => {
                production_logits_out =
                    Some(PathBuf::from(value("--production-logits-out", index)?))
            }
            "--capture-layer" => {
                capture_layer = Some(
                    value("--capture-layer", index)?
                        .parse::<usize>()
                        .map_err(|error| format!("--capture-layer must be usize: {error}"))?,
                )
            }
            "--capture-dir" => capture_dir = Some(PathBuf::from(value("--capture-dir", index)?)),
            other => return Err(format!("unknown argument {other:?}")),
        }
        index += 2;
    }
    if capture_layer.is_some() != capture_dir.is_some() {
        return Err("--capture-layer and --capture-dir must be supplied together".into());
    }
    Ok(Args {
        model: model.ok_or("--model is required")?,
        prompt: prompt.ok_or("--prompt-token-fixture is required")?,
        prompt_positions: prompt_positions.ok_or("--prompt-positions is required")?,
        prompt_file_sha256: prompt_file_sha256.ok_or("--prompt-file-sha256 is required")?,
        prompt_tokens_sha256: prompt_tokens_sha256.ok_or("--prompt-tokens-sha256 is required")?,
        teacher_token: teacher_token.ok_or("--teacher-token is required")?,
        production_logits_out,
        capture_layer,
        capture_dir,
    })
}

fn validate_empty_real_directory(path: &Path) -> Result<(), String> {
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|error| format!("stat capture directory {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(format!(
            "capture directory {} must be a real directory",
            path.display()
        ));
    }
    if std::fs::read_dir(path)
        .map_err(|error| format!("read capture directory {}: {error}", path.display()))?
        .next()
        .is_some()
    {
        return Err(format!(
            "capture directory {} must be empty",
            path.display()
        ));
    }
    Ok(())
}

fn parse_manifest() -> Result<ArtifactManifest, String> {
    serde_json::from_str(RELEASE_ARTIFACTS)
        .map_err(|error| format!("embedded release artifact manifest is invalid: {error}"))
}

fn validate_manifest(manifest: &ArtifactManifest) -> Result<(), String> {
    if manifest.schema != "muser.release-artifacts.v2" || manifest.repository.is_empty() {
        return Err("embedded release artifact manifest has an unsupported identity".into());
    }
    for (name, artifact) in [
        ("target", &manifest.artifacts.target),
        ("vision", &manifest.artifacts.vision),
        ("dflash", &manifest.artifacts.dflash),
    ] {
        parse_sha256(&artifact.sha256, "manifest artifact SHA-256")?;
        if artifact.filename.is_empty()
            || artifact.bytes == 0
            || artifact.revision != manifest.revision
            || !artifact.url.contains(&format!("/{}/", manifest.revision))
        {
            return Err(format!("embedded {name} artifact identity is incomplete"));
        }
    }
    Ok(())
}

fn validate_regular_file(path: &Path, label: &str) -> Result<(), String> {
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|error| format!("stat {label} {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!("{label} {} is not a regular file", path.display()));
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let file = File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|error| format!("read {}: {error}", path.display()))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn digest_tokens(tokens: &[u32]) -> String {
    let mut digest = Sha256::new();
    for token in tokens {
        digest.update(token.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn digest_logits(logits: &[f32]) -> String {
    let mut digest = Sha256::new();
    for value in logits {
        digest.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn write_logits_exclusive(path: &Path, logits: &[f32]) -> Result<(), String> {
    let bytes = logits
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect::<Vec<_>>();
    write_bytes_exclusive(path, &bytes)
}

fn write_bytes_exclusive(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .filter(|value| !value.as_os_str().is_empty())
        .ok_or("--production-logits-out must have a parent directory")?;
    let parent_metadata = std::fs::symlink_metadata(parent)
        .map_err(|error| format!("stat logits parent {}: {error}", parent.display()))?;
    if parent_metadata.file_type().is_symlink() || !parent_metadata.is_dir() {
        return Err(format!(
            "logits parent {} must be a real directory",
            parent.display()
        ));
    }
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("create logits output {}: {error}", path.display()))?;
    output
        .write_all(bytes)
        .map_err(|error| format!("write diagnostic output {}: {error}", path.display()))?;
    output
        .sync_all()
        .map_err(|error| format!("fsync logits output {}: {error}", path.display()))?;
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| format!("fsync logits parent {}: {error}", parent.display()))?;
    Ok(())
}

fn parse_fixture_bytes(bytes: &[u8], path: &Path) -> Result<Vec<u32>, String> {
    let text = std::str::from_utf8(bytes)
        .map_err(|error| format!("prompt fixture {} is not UTF-8: {error}", path.display()))?;
    let tokens = text
        .split(|character: char| character == ',' || character.is_ascii_whitespace())
        .filter(|field| !field.is_empty())
        .map(|field| {
            field.parse::<u32>().map_err(|error| {
                format!(
                    "prompt fixture {} has invalid token {field:?}: {error}",
                    path.display()
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if tokens.is_empty() {
        return Err(format!("prompt fixture {} is empty", path.display()));
    }
    Ok(tokens)
}

fn parse_positive_usize(value: &str, name: &str) -> Result<usize, String> {
    let value = value
        .parse::<usize>()
        .map_err(|error| format!("{name} must be an integer: {error}"))?;
    if value == 0 {
        return Err(format!("{name} must be positive"));
    }
    Ok(value)
}

fn parse_sha256(value: &str, name: &str) -> Result<String, String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!(
            "{name} must be 64 lowercase hexadecimal characters"
        ));
    }
    Ok(value.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_manifest_is_strict_and_pinned() {
        let manifest = parse_manifest().unwrap();
        validate_manifest(&manifest).unwrap();
        assert_eq!(manifest.artifacts.target.bytes, 16_756_681_056);
        assert_eq!(
            manifest.artifacts.target.sha256,
            "7e9b74b7c8875e9e265695df9613bf6290f2392e479ce740495a129019c488d8"
        );
    }

    #[test]
    fn prompt_fixture_and_token_digest_are_canonical() {
        let tokens = parse_fixture_bytes(b"1, 2\n4294967295\n", Path::new("fixture")).unwrap();
        assert_eq!(tokens, [1, 2, u32::MAX]);
        assert_eq!(
            digest_tokens(&tokens),
            "0a1ce634879f6a487527c9a185b1a4a3de7f41238ad4139e3c6d9e0da723628c"
        );
        assert!(parse_fixture_bytes(b"", Path::new("empty")).is_err());
        assert!(parse_fixture_bytes(b"1 nope", Path::new("bad")).is_err());
    }

    #[test]
    fn cli_requires_all_identity_pins_and_rejects_unknowns() {
        let valid = [
            "--model",
            "model.gguf",
            "--prompt-token-fixture",
            "prompt.tokens",
            "--prompt-positions",
            "2048",
            "--prompt-file-sha256",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "--prompt-tokens-sha256",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "--teacher-token",
            "42",
            "--production-logits-out",
            "out/logits.f32",
        ]
        .into_iter()
        .map(str::to_owned);
        let args = parse_args(valid).unwrap();
        assert_eq!(args.prompt_positions, 2048);
        assert_eq!(args.teacher_token, 42);
        assert_eq!(
            args.production_logits_out.as_deref(),
            Some(Path::new("out/logits.f32"))
        );
        assert!(parse_args(["--unknown".into(), "x".into()].into_iter()).is_err());
    }

    #[test]
    fn driver_contains_exactly_one_call_to_each_profiled_graph() {
        let source = include_str!("metal_phase.rs");
        let production = [".decode(", "DecodeInput"].concat();
        let legacy = [".teacher_forced", "_decode(&["].concat();
        assert_eq!(source.matches(&production).count(), 1);
        assert_eq!(source.matches(&legacy).count(), 1);
    }

    #[test]
    fn capture_rendezvous_is_fail_closed_and_bounded() {
        let source = include_str!("metal_phase.rs");
        assert!(source.contains("create_new(true)"));
        assert!(source.contains("1_000..=30_000"));
        assert!(source.contains("muser.metal-capture-ready.v1"));
        assert!(source.contains("live trace mode forbids the isolated phase profilers"));
        assert!(source.contains("live trace mode requires a capture-ready rendezvous"));
    }

    #[test]
    fn prompt_capture_retains_entry_boundaries_before_layer_zero() {
        assert!(PROMPT_CAPTURE_BOUNDARIES.contains(&"embedding"));
        assert!(PROMPT_CAPTURE_BOUNDARIES.contains(&"entry_norm"));
        assert!(PROMPT_CAPTURE_BOUNDARIES.contains(&"attn_norm-0"));
    }
}
