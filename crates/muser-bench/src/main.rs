use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use muser_engine::gguf::GgufFile;
use muser_engine::{Model, ModelConfig, PrefillBatch, SessionConfig};
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "lowercase")]
enum Surface {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Backend {
    Cpu,
    Metal,
}

impl Backend {
    fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu-reference",
            Self::Metal => "metal-reference",
        }
    }
}

#[derive(Debug)]
struct Args {
    model: PathBuf,
    surface: Surface,
    tokens: usize,
    start_depth: usize,
    teacher_forced: usize,
    repetitions: usize,
    kv: String,
    flash_attention: String,
    identity: String,
    backend: Backend,
    prompt_fixture: Option<PathBuf>,
    decode_fixture: Option<PathBuf>,
    dry_run: bool,
}

#[derive(Serialize)]
struct Fingerprint<'a> {
    identity: &'a str,
    backend: &'static str,
    kv: &'a str,
    flash_attention_requested: &'a str,
    flash_attention_active: bool,
    matvec_route: &'static str,
    prefill_attention_route: &'static str,
    prefill_q4_route: &'static str,
    prefill_norm_route: &'static str,
    prefill_command_buffer_route: &'static str,
    prefill_dispatch_route: &'static str,
    decode_ffn_route: &'static str,
    ggml_metallib_sha256: Option<&'a str>,
    prompt_fixture_sha256: Option<&'a str>,
    prompt_tokens_sha256: Option<&'a str>,
    decode_fixture_sha256: Option<&'a str>,
    decode_tokens_sha256: Option<&'a str>,
    workload_sha256: &'a str,
    warmup_policy: &'static str,
    repetition_state_policy: &'static str,
    seal_eligible: bool,
}

struct RouteIdentity {
    matvec_route: &'static str,
    prefill_attention_route: &'static str,
    prefill_q4_route: &'static str,
    prefill_norm_route: &'static str,
    prefill_command_buffer_route: &'static str,
    prefill_dispatch_route: &'static str,
    decode_ffn_route: &'static str,
    ggml_metallib_sha256: Option<String>,
}

struct Workload {
    prompt: Option<Vec<u32>>,
    decode: Option<Vec<u32>>,
    prompt_file_sha256: Option<String>,
    prompt_tokens_sha256: Option<String>,
    decode_file_sha256: Option<String>,
    decode_tokens_sha256: Option<String>,
    workload_sha256: String,
}

#[derive(Serialize)]
struct Sample<'a> {
    schema: &'static str,
    kind: &'static str,
    surface: Surface,
    depth: usize,
    repetition: usize,
    elapsed_ns: u64,
    measured_tokens: usize,
    token_digest: String,
    fingerprint: Fingerprint<'a>,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("muser-bench: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.first().map(String::as_str) == Some("tokenize") {
        return run_tokenize(argv[1..].to_vec());
    }
    if argv.first().map(String::as_str) == Some("metadata") {
        return run_metadata(argv[1..].to_vec());
    }
    let args = parse_args()?;
    validate(&args)?;
    let route = route_identity(args.backend)?;
    if args.dry_run {
        let report = serde_json::json!({
            "schema": "muser-bench.v1",
            "kind": "dry-run",
            "accelerator_touched": false,
            "surface": args.surface,
            "depth": depth(&args),
            "repetitions": args.repetitions,
            "measured_tokens": measured_tokens(&args),
            "backend": args.backend.name(),
            "matvec_route": route.matvec_route,
            "prefill_attention_route": route.prefill_attention_route,
            "prefill_q4_route": route.prefill_q4_route,
            "prefill_norm_route": route.prefill_norm_route,
            "prefill_command_buffer_route": route.prefill_command_buffer_route,
            "prefill_dispatch_route": route.prefill_dispatch_route,
            "decode_ffn_route": route.decode_ffn_route,
            "ggml_metallib_sha256": route.ggml_metallib_sha256,
            "prompt_fixture": args.prompt_fixture,
            "decode_fixture": args.decode_fixture,
            "workload_policy": "fixture-files-sha256-at-execution",
            "seal_eligible": false,
        });
        println!(
            "{}",
            serde_json::to_string(&report).map_err(|error| error.to_string())?
        );
        return Ok(());
    }

    let workload = load_workload(&args)?;
    let model = Model::load(ModelConfig::new(&args.model)).map_err(|error| error.to_string())?;
    validate_fixture_vocab(&model, &workload)?;
    let bos = model.config().bos_token_id.unwrap_or(0);
    let mut prefill = match args.surface {
        Surface::Prefill => Some(PreparedPrefill::new(&model, &args, bos, &workload)?),
        Surface::Decode => None,
    };
    let mut decode = match args.surface {
        Surface::Decode => Some(PreparedDecode::new(&model, &args, bos, &workload)?),
        Surface::Prefill => None,
    };
    let mut samples = Vec::with_capacity(args.repetitions);
    for repetition in 0..args.repetitions {
        let (elapsed_ns, digest) = match args.surface {
            Surface::Prefill => prefill.as_mut().expect("prepared prefill").run()?,
            Surface::Decode => decode.as_mut().expect("prepared decode").run()?,
        };
        samples.push(elapsed_ns);
        let record = Sample {
            schema: "muser-bench.v1",
            kind: "sample",
            surface: args.surface,
            depth: depth(&args),
            repetition,
            elapsed_ns,
            measured_tokens: measured_tokens(&args),
            token_digest: format!("fnv1a64:{digest:016x}"),
            fingerprint: Fingerprint {
                identity: &args.identity,
                backend: args.backend.name(),
                kv: &args.kv,
                flash_attention_requested: &args.flash_attention,
                flash_attention_active: args.backend == Backend::Metal,
                matvec_route: route.matvec_route,
                prefill_attention_route: route.prefill_attention_route,
                prefill_q4_route: route.prefill_q4_route,
                prefill_norm_route: route.prefill_norm_route,
                prefill_command_buffer_route: route.prefill_command_buffer_route,
                prefill_dispatch_route: route.prefill_dispatch_route,
                decode_ffn_route: route.decode_ffn_route,
                ggml_metallib_sha256: route.ggml_metallib_sha256.as_deref(),
                prompt_fixture_sha256: workload.prompt_file_sha256.as_deref(),
                prompt_tokens_sha256: workload.prompt_tokens_sha256.as_deref(),
                decode_fixture_sha256: workload.decode_file_sha256.as_deref(),
                decode_tokens_sha256: workload.decode_tokens_sha256.as_deref(),
                workload_sha256: &workload.workload_sha256,
                warmup_policy: match args.surface {
                    Surface::Prefill => "full-logical-prompt-once-before-timing-v1",
                    Surface::Decode => "full-teacher-block-before-timing-v1",
                },
                repetition_state_policy: match args.surface {
                    Surface::Prefill => "reset-same-session-each-repetition-v1",
                    Surface::Decode => "exact-prefix-state-restore-each-repetition-v1",
                },
                seal_eligible: false,
            },
        };
        println!(
            "{}",
            serde_json::to_string(&record).map_err(|error| error.to_string())?
        );
    }
    let mean = samples.iter().map(|&value| value as f64).sum::<f64>() / samples.len() as f64;
    let variance = samples
        .iter()
        .map(|&value| {
            let delta = value as f64 - mean;
            delta * delta
        })
        .sum::<f64>()
        / samples.len() as f64;
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "schema": "muser-bench.v1",
            "kind": "summary",
            "surface": args.surface,
            "depth": depth(&args),
            "raw_ns": samples,
            "mean_ns": mean,
            "cv": if mean == 0.0 { 0.0 } else { variance.sqrt() / mean },
            "seal_eligible": false,
            "reason": if args.backend == Backend::Metal {
                "Metal reference route is measurable but unsealed until flash attention and baseline gates pass"
            } else {
                "CPU reference executor is correctness/smoke only"
            }
        }))
        .map_err(|error| error.to_string())?
    );
    Ok(())
}

fn run_metadata(argv: Vec<String>) -> Result<(), String> {
    let [flag, path] = argv.as_slice() else {
        return Err("metadata requires exactly --model PATH".into());
    };
    if flag != "--model" {
        return Err(format!("unknown metadata argument {flag:?}"));
    }
    let path = PathBuf::from(path);
    let gguf = GgufFile::parse_path(&path).map_err(|error| error.to_string())?;
    let template = gguf
        .chat_template()
        .ok_or("GGUF has no tokenizer.chat_template")?;
    let template_sha = gguf
        .chat_template_sha256()
        .ok_or("GGUF has no tokenizer.chat_template")?;
    let vocab = gguf.vocab();
    let token = |key: &str| {
        gguf.meta_u32(key).map(|id| {
            serde_json::json!({
                "id": id,
                "text": vocab.get(id as usize),
            })
        })
    };
    println!(
        "{}",
        serde_json::to_string(&serde_json::json!({
            "schema": "muser.gguf-metadata-identity.v1",
            "model": path,
            "tokenizer_metadata_sha256": hex_bytes(&gguf.tokenizer_metadata_sha256()),
            "chat_template_sha256": hex_bytes(&template_sha),
            "chat_template_bytes": template.len(),
            "chat_template": template,
            "bos_token": token("tokenizer.ggml.bos_token_id"),
            "eos_token": token("tokenizer.ggml.eos_token_id"),
            "eot_token": token("tokenizer.ggml.eot_token_id"),
        }))
        .map_err(|error| error.to_string())?
    );
    Ok(())
}

fn hex_bytes(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut value = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut value, "{byte:02x}").expect("writing to String cannot fail");
    }
    value
}

fn route_identity(backend: Backend) -> Result<RouteIdentity, String> {
    if backend == Backend::Cpu {
        return Ok(RouteIdentity {
            matvec_route: "cpu-reference",
            prefill_attention_route: "cpu-reference",
            prefill_q4_route: "cpu-reference",
            prefill_norm_route: "cpu-reference",
            prefill_command_buffer_route: "cpu-reference",
            prefill_dispatch_route: "cpu-reference",
            decode_ffn_route: "cpu-reference",
            ggml_metallib_sha256: None,
        });
    }
    let Some(path) = std::env::var_os("MUSER_GGML_METALLIB").map(PathBuf::from) else {
        return Ok(RouteIdentity {
            matvec_route: "muser-local-q4k-q5k",
            prefill_attention_route: "ferrite-fa2-dk128-ring-staged-v2",
            prefill_q4_route: "ferrite-q4k-sgm-aligned-v1",
            prefill_norm_route: prefill_norm_route(),
            prefill_command_buffer_route: prefill_command_buffer_route(),
            prefill_dispatch_route: prefill_dispatch_route(),
            decode_ffn_route: decode_ffn_route(),
            ggml_metallib_sha256: None,
        });
    };
    let bytes = std::fs::read(&path).map_err(|error| {
        format!(
            "cannot fingerprint GGML metallib {}: {error}",
            path.display()
        )
    })?;
    let digest = Sha256::digest(bytes);
    Ok(RouteIdentity {
        matvec_route: "llama-ggml-metallib",
        prefill_attention_route: "ferrite-fa2-dk128-ring-staged-v2",
        prefill_q4_route: "ferrite-q4k-sgm-aligned-v1",
        prefill_norm_route: prefill_norm_route(),
        prefill_command_buffer_route: prefill_command_buffer_route(),
        prefill_dispatch_route: prefill_dispatch_route(),
        decode_ffn_route: decode_ffn_route(),
        ggml_metallib_sha256: Some(format!("sha256:{digest:x}")),
    })
}

fn prefill_norm_route() -> &'static str {
    if std::env::var_os("MUSER_NO_FUSED_PREFILL_DUAL_NORM").is_some() {
        "split-dual-eps-v1"
    } else {
        "ferrite-fused-dual-eps-v1"
    }
}

fn prefill_command_buffer_route() -> &'static str {
    "ferrite-unretained-v1"
}

fn prefill_dispatch_route() -> &'static str {
    if std::env::var_os("MUSER_SERIAL_PREFILL_DISPATCH").is_some() {
        "serial-v1"
    } else {
        "concurrent-qkvg-ffn-v1"
    }
}

fn decode_ffn_route() -> &'static str {
    if std::env::var_os("MUSER_FERRITE_FFN_GATE_UP").is_some() {
        "ferrite-q4k-gate-up-4r2s-v1"
    } else {
        "upstream-split-gate-up-v1"
    }
}

struct PreparedPrefill {
    session: muser_engine::Session,
    tokens: Vec<u32>,
}

impl PreparedPrefill {
    fn new(model: &Model, args: &Args, bos: u32, workload: &Workload) -> Result<Self, String> {
        let mut session = new_session(model, args.backend, args.tokens)?;
        let tokens = workload
            .prompt
            .clone()
            .unwrap_or_else(|| synthetic_tokens(args.tokens, bos, model.config().vocab_size));
        // Match llama-bench: fault the complete logical prompt and all batch
        // scratch once, then time identical resets on the same live engine.
        prefill_chunked(&mut session, &tokens)?;
        session.reset();
        Ok(Self { session, tokens })
    }

    fn run(&mut self) -> Result<(u64, u64), String> {
        self.session.reset();
        let started = Instant::now();
        let logits = prefill_chunked(&mut self.session, &self.tokens)?;
        if logits.iter().any(|value| !value.is_finite()) {
            return Err("prefill produced nonfinite logits".into());
        }
        let elapsed = nanos(started.elapsed().as_nanos());
        Ok((elapsed, digest_logits(&logits)))
    }
}

struct PreparedDecode {
    session: muser_engine::Session,
    prefix: Option<muser_engine::cache::SessionCacheSnapshot>,
    teacher: Vec<u32>,
}

impl PreparedDecode {
    fn new(model: &Model, args: &Args, bos: u32, workload: &Workload) -> Result<Self, String> {
        let limit = args
            .start_depth
            .checked_add(args.teacher_forced)
            .ok_or("context length overflow")?
            .max(1);
        let mut session = new_session(model, args.backend, limit)?;
        if args.start_depth > 0 {
            let prefix = workload.prompt.clone().unwrap_or_else(|| {
                synthetic_tokens(args.start_depth, bos, model.config().vocab_size)
            });
            prefill_chunked(&mut session, &prefix)?;
        }
        let teacher = workload.decode.clone().unwrap_or_else(|| {
            synthetic_tokens(args.teacher_forced, bos, model.config().vocab_size)
        });
        let prefix = if session.position() == 0 {
            None
        } else {
            Some(
                session
                    .export_cache_snapshot()
                    .map_err(|error| error.to_string())?,
            )
        };
        // The timed decode graph is one retained command buffer of
        // `teacher_forced` tokens. Warm that same graph once, then restore.
        // A single teacher token compiles a different encoder and leaves the
        // first timed sample paying a ~50 ms command-buffer tax.
        session
            .teacher_forced_decode(&teacher)
            .map_err(|error| error.to_string())?;
        restore_decode_prefix(&mut session, prefix.as_ref())?;
        Ok(Self {
            session,
            prefix,
            teacher,
        })
    }

    fn run(&mut self) -> Result<(u64, u64), String> {
        // Every repetition restores the identical prefix before timing. The
        // same live session, queues, pipelines, and residency sets are reused,
        // matching llama-bench's construct-once/restore-many lifecycle.
        restore_decode_prefix(&mut self.session, self.prefix.as_ref())?;
        let started = Instant::now();
        let generated = self
            .session
            .teacher_forced_decode(&self.teacher)
            .map_err(|error| error.to_string())?;
        let mut digest = FNV_OFFSET;
        for token_id in generated {
            digest = fnv_bytes(digest, &token_id.to_le_bytes());
        }
        Ok((nanos(started.elapsed().as_nanos()), digest))
    }
}

fn restore_decode_prefix(
    session: &mut muser_engine::Session,
    prefix: Option<&muser_engine::cache::SessionCacheSnapshot>,
) -> Result<(), String> {
    // Restore detached prefix state rather than inferring physical ring
    // placement from the logical position.
    if let Some(prefix) = prefix {
        session
            .install_cache_snapshot(prefix)
            .map_err(|error| error.to_string())?;
    } else {
        session.reset();
    }
    Ok(())
}

fn prefill_chunked(
    session: &mut muser_engine::Session,
    tokens: &[u32],
) -> Result<Vec<f32>, String> {
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
    logits.ok_or_else(|| "prefill fixture is empty".into())
}

fn load_workload(args: &Args) -> Result<Workload, String> {
    let prompt = args
        .prompt_fixture
        .as_ref()
        .map(|path| load_fixture(path))
        .transpose()?;
    let decode = args
        .decode_fixture
        .as_ref()
        .map(|path| load_fixture(path))
        .transpose()?;
    let expected_prompt = match args.surface {
        Surface::Prefill => args.tokens,
        Surface::Decode => args.start_depth,
    };
    if let Some((tokens, _)) = prompt.as_ref() {
        if tokens.len() != expected_prompt {
            return Err(format!(
                "prompt fixture has {} tokens; expected {expected_prompt}",
                tokens.len()
            ));
        }
    }
    if let Some((tokens, _)) = decode.as_ref() {
        if matches!(args.surface, Surface::Prefill) || tokens.len() != args.teacher_forced {
            return Err(format!(
                "decode fixture has {} tokens; expected {} for decode surface",
                tokens.len(),
                args.teacher_forced
            ));
        }
    }
    if args.prompt_fixture.is_some() != (expected_prompt > 0) {
        return Err(if expected_prompt > 0 {
            "fixture mode requires --prompt-token-fixture".into()
        } else {
            "a zero-depth cell must not provide a prompt fixture".into()
        });
    }
    if matches!(args.surface, Surface::Decode) != args.decode_fixture.is_some() {
        return Err("fixture mode requires a decode fixture only on decode cells".into());
    }

    let (prompt_tokens, prompt_file_sha256) = prompt
        .map(|(tokens, digest)| (Some(tokens), Some(digest)))
        .unwrap_or((None, None));
    let (decode_tokens, decode_file_sha256) = decode
        .map(|(tokens, digest)| (Some(tokens), Some(digest)))
        .unwrap_or((None, None));
    let prompt_tokens_sha256 = prompt_tokens.as_deref().map(digest_tokens);
    let decode_tokens_sha256 = decode_tokens.as_deref().map(digest_tokens);
    let mut workload = Sha256::new();
    workload.update(b"muser-bench-workload-v1\0");
    for tokens in [prompt_tokens.as_deref(), decode_tokens.as_deref()] {
        let len = tokens.map_or(0, <[u32]>::len);
        workload.update((len as u64).to_le_bytes());
        if let Some(tokens) = tokens {
            for token in tokens {
                workload.update(token.to_le_bytes());
            }
        }
    }
    Ok(Workload {
        prompt: prompt_tokens,
        decode: decode_tokens,
        prompt_file_sha256,
        prompt_tokens_sha256,
        decode_file_sha256,
        decode_tokens_sha256,
        workload_sha256: format!("sha256:{:x}", workload.finalize()),
    })
}

fn load_fixture(path: &std::path::Path) -> Result<(Vec<u32>, String), String> {
    let bytes = std::fs::read(path)
        .map_err(|error| format!("cannot read fixture {}: {error}", path.display()))?;
    let text = std::str::from_utf8(&bytes)
        .map_err(|error| format!("fixture {} is not UTF-8: {error}", path.display()))?;
    let tokens = text
        .split(|character: char| character == ',' || character.is_ascii_whitespace())
        .filter(|field| !field.is_empty())
        .map(|field| {
            field.parse::<u32>().map_err(|error| {
                format!(
                    "fixture {} has bad token {field:?}: {error}",
                    path.display()
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((tokens, format!("sha256:{:x}", Sha256::digest(bytes))))
}

fn digest_tokens(tokens: &[u32]) -> String {
    let mut digest = Sha256::new();
    for token in tokens {
        digest.update(token.to_le_bytes());
    }
    format!("sha256:{:x}", digest.finalize())
}

fn validate_fixture_vocab(model: &Model, workload: &Workload) -> Result<(), String> {
    let vocab_size = model.config().vocab_size;
    for (name, tokens) in [("prompt", &workload.prompt), ("decode", &workload.decode)] {
        if let Some(tokens) = tokens {
            if let Some((index, token)) = tokens
                .iter()
                .enumerate()
                .find(|(_, token)| **token as usize >= vocab_size)
            {
                return Err(format!(
                    "{name} fixture token {index}={token} is outside vocabulary 0..{vocab_size}"
                ));
            }
        }
    }
    Ok(())
}

fn new_session(
    model: &Model,
    backend: Backend,
    max_context: usize,
) -> Result<muser_engine::Session, String> {
    let config = SessionConfig { max_context };
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
                let _ = (model, config);
                Err("--backend metal requires macOS and muser-bench --features metal".into())
            }
        }
    }
}

fn synthetic_tokens(count: usize, bos: u32, vocab_size: usize) -> Vec<u32> {
    const BODY: [u32; 8] = [19873, 24, 10676, 768, 1085, 13634, 2304, 1509];
    (0..count)
        .map(|index| {
            if index == 0 {
                bos
            } else {
                BODY[(index - 1) % BODY.len()] % vocab_size as u32
            }
        })
        .collect()
}

fn digest_logits(logits: &[f32]) -> u64 {
    logits.iter().fold(FNV_OFFSET, |digest, value| {
        fnv_bytes(digest, &value.to_bits().to_le_bytes())
    })
}

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

fn fnv_bytes(mut digest: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        digest ^= u64::from(*byte);
        digest = digest.wrapping_mul(FNV_PRIME);
    }
    digest
}

fn nanos(value: u128) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn depth(args: &Args) -> usize {
    match args.surface {
        Surface::Prefill => args.tokens,
        Surface::Decode => args.start_depth,
    }
}

fn measured_tokens(args: &Args) -> usize {
    match args.surface {
        Surface::Prefill => args.tokens,
        Surface::Decode => args.teacher_forced,
    }
}

fn validate(args: &Args) -> Result<(), String> {
    if args.repetitions == 0 {
        return Err("--repetitions must be positive".into());
    }
    if args.kv != "f16" {
        return Err("only --kv f16 is valid for the release route".into());
    }
    if !matches!(args.flash_attention.as_str(), "on" | "off") {
        return Err("--flash-attention must be on or off".into());
    }
    match (args.backend, args.flash_attention.as_str()) {
        (Backend::Metal, "off") => {
            return Err("the Metal route is FlashAttention-only; use --flash-attention on".into())
        }
        (Backend::Cpu, "on") => return Err("the CPU oracle requires --flash-attention off".into()),
        _ => {}
    }
    match args.surface {
        Surface::Prefill if args.tokens == 0 => Err("--tokens must be positive".into()),
        Surface::Decode if args.teacher_forced == 0 => {
            Err("--teacher-forced must be positive".into())
        }
        _ => Ok(()),
    }
}

fn parse_args() -> Result<Args, String> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut model = None;
    let mut surface = None;
    let mut tokens = 0;
    let mut start_depth = 0;
    let mut teacher_forced = 64;
    let mut repetitions = 3;
    let mut kv = "f16".to_string();
    let mut flash_attention = "off".to_string();
    let mut identity = "unsealed-local".to_string();
    let mut backend = Backend::Cpu;
    let mut prompt_fixture = None;
    let mut decode_fixture = None;
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
            "--surface" => {
                surface = Some(match value("--surface", index)?.as_str() {
                    "prefill" => Surface::Prefill,
                    "decode" => Surface::Decode,
                    other => return Err(format!("unsupported surface {other:?}")),
                })
            }
            "--tokens" => tokens = parse_usize(&value("--tokens", index)?, "--tokens")?,
            "--start-depth" => {
                start_depth = parse_usize(&value("--start-depth", index)?, "--start-depth")?
            }
            "--teacher-forced" => {
                teacher_forced =
                    parse_usize(&value("--teacher-forced", index)?, "--teacher-forced")?
            }
            "--repetitions" => {
                repetitions = parse_usize(&value("--repetitions", index)?, "--repetitions")?
            }
            "--kv" => kv = value("--kv", index)?,
            "--flash-attention" => flash_attention = value("--flash-attention", index)?,
            "--identity" => identity = value("--identity", index)?,
            "--backend" => {
                backend = match value("--backend", index)?.as_str() {
                    "cpu" => Backend::Cpu,
                    "metal" => Backend::Metal,
                    other => return Err(format!("unsupported backend {other:?}")),
                }
            }
            "--prompt-token-fixture" => {
                prompt_fixture = Some(PathBuf::from(value("--prompt-token-fixture", index)?))
            }
            "--decode-token-fixture" => {
                decode_fixture = Some(PathBuf::from(value("--decode-token-fixture", index)?))
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
    Ok(Args {
        model: model.ok_or("--model is required")?,
        surface: surface.ok_or("--surface is required")?,
        tokens,
        start_depth,
        teacher_forced,
        repetitions,
        kv,
        flash_attention,
        identity,
        backend,
        prompt_fixture,
        decode_fixture,
        dry_run,
    })
}

/// CPU-only tokenizer probe: loads a model's tokenizer (no GPU session) and
/// prints the token IDs for a text file, so campaign fixtures can be
/// generated from checked-in prose rather than only synthetic bodies.
fn run_tokenize(argv: Vec<String>) -> Result<(), String> {
    let mut model_path = None;
    let mut file_path = None;
    let mut index = 0;
    while index < argv.len() {
        match argv[index].as_str() {
            "--model" => {
                model_path = Some(PathBuf::from(
                    argv.get(index + 1)
                        .cloned()
                        .ok_or("--model requires a value")?,
                ));
                index += 2;
            }
            "--file" => {
                file_path = Some(PathBuf::from(
                    argv.get(index + 1)
                        .cloned()
                        .ok_or("--file requires a value")?,
                ));
                index += 2;
            }
            other => return Err(format!("unknown tokenize argument {other:?}")),
        }
    }
    let model_path = model_path.ok_or("tokenize requires --model")?;
    let file_path = file_path.ok_or("tokenize requires --file")?;
    let text = std::fs::read_to_string(&file_path).map_err(|error| error.to_string())?;
    let model = Model::load(ModelConfig::new(&model_path)).map_err(|error| error.to_string())?;
    let tokens = model.encode(&text);
    let report = serde_json::json!({
        "schema": "muser-bench.tokenize.v1",
        "accelerator_touched": false,
        "file": file_path,
        "token_count": tokens.len(),
        "tokens": tokens,
    });
    println!(
        "{}",
        serde_json::to_string(&report).map_err(|error| error.to_string())?
    );
    Ok(())
}

fn parse_usize(value: &str, name: &str) -> Result<usize, String> {
    value
        .parse()
        .map_err(|error| format!("bad {name}: {error}"))
}
