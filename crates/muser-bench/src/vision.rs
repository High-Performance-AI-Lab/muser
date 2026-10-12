//! Four-fixture Muse vision qualification against the independent CPU oracle.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use muser_engine::vision::{PreprocessedImage, VisionModel};
use muser_engine::{
    DecodeInput, EmbeddingSegment, Model, ModelConfig, PrefillBatch, PrefillSegment, Session,
    SessionConfig,
};
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
    mmproj: PathBuf,
    mtmd_bridge: PathBuf,
    image: PathBuf,
    fixture: String,
    repetitions: usize,
    output_tokens: usize,
    target_backend: Backend,
    identity: String,
    dry_run: bool,
}

#[derive(Serialize)]
struct Sample<'a> {
    schema: &'static str,
    kind: &'static str,
    identity: &'a str,
    fixture: &'a str,
    repetition: usize,
    elapsed_ns: u64,
    projected_tokens: usize,
    embeddings_sha256: &'a str,
    route: &'static str,
}

struct DecoderEvidence {
    tokens: Vec<u32>,
    insertion_start: usize,
    insertion_end: usize,
    installed_positions: usize,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("muser-vision-qualify: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    if args.repetitions == 0 || args.output_tokens == 0 {
        return Err("repetitions and output tokens must be positive".into());
    }
    if args.dry_run {
        println!(
            "{}",
            serde_json::json!({
                "schema": "muser.vision-qualify.v1",
                "kind": "dry-run",
                "accelerator_touched": false,
                "model": args.model,
                "mmproj": args.mmproj,
                "mtmd_bridge": args.mtmd_bridge,
                "image": args.image,
                "fixture": args.fixture,
                "repetitions": args.repetitions,
                "output_tokens": args.output_tokens,
                "target_backend": args.target_backend.name(),
                "identity": args.identity,
                "correctness": [
                    "upstream-preprocessing-vs-rust-oracle",
                    "projected-embedding-cosine-and-relative-l2",
                    "exact-decoder-token-equality",
                ],
                "seal_eligible": false,
            })
        );
        return Ok(());
    }

    let encoded = std::fs::read(&args.image)
        .map_err(|error| format!("cannot read {}: {error}", args.image.display()))?;
    let source = image::load_from_memory(&encoded)
        .map_err(|error| format!("cannot decode {}: {error}", args.image.display()))?;
    let source_width = source.width() as usize;
    let source_height = source.height() as usize;
    validate_fixture_geometry(&args.fixture, source_width, source_height)?;
    let model = Model::load(ModelConfig::new(&args.model)).map_err(|error| error.to_string())?;
    let vision = load_vision(&args)?;
    if vision.config.output_dim != model.config().hidden_dim {
        return Err(format!(
            "vision output width {} differs from target hidden width {}",
            vision.config.output_dim,
            model.config().hidden_dim
        ));
    }

    let cpu_pixels = vision
        .preprocess_bytes(&encoded)
        .map_err(|error| error.to_string())?;
    let upstream_pixels = vision
        .preprocess_upstream(&encoded, &cpu_pixels)
        .map_err(|error| error.to_string())?;
    let max_pixel_error = max_pixel_error(&vision, &cpu_pixels, &upstream_pixels)?;
    if max_pixel_error > 1.0 / 255.0 {
        return Err(format!(
            "preprocessing max pixel error {max_pixel_error} exceeds 1/255"
        ));
    }

    let cpu_embeddings = vision
        .encode(&cpu_pixels)
        .map_err(|error| error.to_string())?;
    let expected_tokens = vision
        .projected_token_count(&cpu_pixels)
        .map_err(|error| error.to_string())?;
    if cpu_embeddings.len() != expected_tokens {
        return Err("CPU oracle projected-token count differs from geometry".into());
    }

    let mut raw_ns = Vec::with_capacity(args.repetitions);
    let mut accelerated: Option<Vec<Vec<f32>>> = None;
    let mut accelerated_digest = String::new();
    for repetition in 0..args.repetitions {
        let started = Instant::now();
        let embeddings = vision
            .encode_accelerated(&encoded, &cpu_pixels)
            .map_err(|error| error.to_string())?;
        let elapsed_ns = nanos(started.elapsed().as_nanos());
        if embeddings.len() != expected_tokens {
            return Err(format!(
                "accelerated graph emitted {} rows, expected {expected_tokens}",
                embeddings.len()
            ));
        }
        let digest = float_digest(&embeddings);
        if let Some(canonical) = accelerated.as_ref() {
            if canonical != &embeddings {
                return Err("accelerated embeddings changed between repetitions".into());
            }
        } else {
            accelerated_digest = digest.clone();
            accelerated = Some(embeddings.clone());
        }
        raw_ns.push(elapsed_ns);
        println!(
            "{}",
            serde_json::to_string(&Sample {
                schema: "muser.vision-qualify.v1",
                kind: "sample",
                identity: &args.identity,
                fixture: &args.fixture,
                repetition,
                elapsed_ns,
                projected_tokens: expected_tokens,
                embeddings_sha256: &digest,
                route: vision.route_identity(),
            })
            .map_err(|error| error.to_string())?
        );
    }
    let accelerated = accelerated.ok_or("no accelerated vision sample")?;
    let (cosine, relative_l2) = embedding_error(&cpu_embeddings, &accelerated)?;
    if cosine < 0.999 || relative_l2 > 0.01 {
        return Err(format!(
            "embedding parity failed: cosine={cosine} relative_l2={relative_l2}"
        ));
    }

    let cpu_decoder = decoder_tokens(&model, &cpu_embeddings, &args)?;
    let accelerated_decoder = decoder_tokens(&model, &accelerated, &args)?;
    if cpu_decoder.tokens != accelerated_decoder.tokens
        || cpu_decoder.tokens.len() != args.output_tokens
    {
        let mismatch = cpu_decoder
            .tokens
            .iter()
            .zip(&accelerated_decoder.tokens)
            .position(|(left, right)| left != right)
            .unwrap_or(
                cpu_decoder
                    .tokens
                    .len()
                    .min(accelerated_decoder.tokens.len()),
            );
        return Err(format!(
            "decoder output differs at token {mismatch}: oracle={} accelerated={}",
            cpu_decoder.tokens.len(),
            accelerated_decoder.tokens.len()
        ));
    }
    if cpu_decoder.insertion_start != accelerated_decoder.insertion_start
        || cpu_decoder.insertion_end != accelerated_decoder.insertion_end
        || cpu_decoder.installed_positions != accelerated_decoder.installed_positions
    {
        return Err("CPU and accelerated decoder insertion positions differ".into());
    }
    let insertion_positions_sha256 =
        position_digest(cpu_decoder.insertion_start, cpu_decoder.insertion_end);

    let coefficient_of_variation = cv(&raw_ns);
    println!(
        "{}",
        serde_json::json!({
            "schema": "muser.vision-qualify.v1",
            "kind": "summary",
            "identity": args.identity,
            "fixture": args.fixture,
            "route": vision.route_identity(),
            "target_backend": args.target_backend.name(),
            "image_sha256": format!("{:x}", Sha256::digest(&encoded)),
            "preprocessing_sha256": float_digest_flat(&cpu_pixels.pixels),
            "upstream_preprocessing_sha256": float_digest_flat(&upstream_pixels.pixels),
            "cpu_embeddings_sha256": float_digest(&cpu_embeddings),
            "accelerated_embeddings_sha256": accelerated_digest,
            "decoder_tokens_sha256": token_digest(&cpu_decoder.tokens),
            "source_width": source_width,
            "source_height": source_height,
            "width": cpu_pixels.width,
            "height": cpu_pixels.height,
            "projected_tokens": expected_tokens,
            "output_tokens": args.output_tokens,
            "max_pixel_error": max_pixel_error,
            "embedding_cosine": cosine,
            "embedding_relative_l2": relative_l2,
            "exact_decoder_tokens": true,
            "insertion_start": cpu_decoder.insertion_start,
            "insertion_end": cpu_decoder.insertion_end,
            "insertion_count": cpu_decoder.insertion_end - cpu_decoder.insertion_start,
            "insertion_positions_sha256": insertion_positions_sha256,
            "prefix_tokens": cpu_decoder.insertion_start,
            "suffix_tokens": cpu_decoder.installed_positions - cpu_decoder.insertion_end,
            "installed_positions": cpu_decoder.installed_positions,
            "raw_ns": raw_ns,
            "cv": coefficient_of_variation,
            "stable": coefficient_of_variation <= 0.02,
            "seal_eligible": false,
            "reason": "cell evidence requires a complete four-fixture packet and llama.cpp latency comparison",
        })
    );
    Ok(())
}

fn load_vision(args: &Args) -> Result<VisionModel, String> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        VisionModel::load_metal(&args.mmproj, &args.mtmd_bridge).map_err(|error| error.to_string())
    }
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        let _ = args;
        Err("vision qualification requires macOS and the metal feature".into())
    }
}

fn decoder_tokens(
    model: &Model,
    embeddings: &[Vec<f32>],
    args: &Args,
) -> Result<DecoderEvidence, String> {
    let prefix = model.encode("<|im_start|>user\n<|image_start|>");
    let suffix =
        model.encode("<|image_end|>\nDescribe the image.<|im_end|>\n<|im_start|>assistant\n");
    let insertion_start = prefix.len();
    let insertion_end = insertion_start + embeddings.len();
    let positions = insertion_end + suffix.len();
    let limit = positions
        .checked_add(args.output_tokens)
        .ok_or("vision context size overflow")?;
    let mut session = new_session(model, args.target_backend, limit)?;
    let result = session
        .prefill(PrefillBatch {
            segments: vec![
                PrefillSegment::Tokens(prefix),
                PrefillSegment::Embeddings(EmbeddingSegment::new(embeddings.to_vec())),
                PrefillSegment::Tokens(suffix),
            ],
        })
        .map_err(|error| error.to_string())?;
    if result.tokens_processed != positions {
        return Err(format!(
            "decoder installed {} positions, expected {positions}",
            result.tokens_processed
        ));
    }
    let mut next = argmax(result.last_logits()) as u32;
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
    Ok(DecoderEvidence {
        tokens,
        insertion_start,
        insertion_end,
        installed_positions: result.tokens_processed,
    })
}

fn validate_fixture_geometry(fixture: &str, width: usize, height: usize) -> Result<(), String> {
    let expected = match fixture {
        "low-square" => (224, 224),
        "wide" => (1024, 256),
        "tall" => (256, 1024),
        "high-resolution" => (2048, 1536),
        _ => return Err(format!("unknown vision fixture {fixture:?}")),
    };
    if (width, height) != expected {
        return Err(format!(
            "vision fixture {fixture:?} is {width}x{height}, expected {}x{}",
            expected.0, expected.1
        ));
    }
    Ok(())
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

fn max_pixel_error(
    vision: &VisionModel,
    left: &PreprocessedImage,
    right: &PreprocessedImage,
) -> Result<f64, String> {
    if left.width != right.width
        || left.height != right.height
        || left.pixels.len() != right.pixels.len()
    {
        return Err("preprocessing shapes differ".into());
    }
    let plane = left.width * left.height;
    let mut maximum = 0.0f64;
    for channel in 0..3 {
        let scale = vision.config.image_std[channel] as f64;
        for index in 0..plane {
            let error = ((left.pixels[channel * plane + index]
                - right.pixels[channel * plane + index]) as f64
                * scale)
                .abs();
            maximum = maximum.max(error);
        }
    }
    Ok(maximum)
}

fn embedding_error(left: &[Vec<f32>], right: &[Vec<f32>]) -> Result<(f64, f64), String> {
    if left.len() != right.len()
        || left.iter().map(Vec::len).collect::<Vec<_>>()
            != right.iter().map(Vec::len).collect::<Vec<_>>()
    {
        return Err("embedding shapes differ".into());
    }
    let mut dot = 0.0f64;
    let mut left_norm = 0.0f64;
    let mut right_norm = 0.0f64;
    let mut squared_error = 0.0f64;
    for (&a, &b) in left.iter().flatten().zip(right.iter().flatten()) {
        let a = a as f64;
        let b = b as f64;
        dot += a * b;
        left_norm += a * a;
        right_norm += b * b;
        squared_error += (a - b) * (a - b);
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        return Err("zero-norm vision embeddings".into());
    }
    Ok((
        dot / (left_norm.sqrt() * right_norm.sqrt()),
        (squared_error / left_norm).sqrt(),
    ))
}

fn parse_args() -> Result<Args, String> {
    let mut model = None;
    let mut mmproj = None;
    let mut mtmd_bridge = None;
    let mut image = None;
    let mut fixture = None;
    let mut repetitions = 3usize;
    let mut output_tokens = 64usize;
    let mut target_backend = Backend::Metal;
    let mut identity = None;
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
            "--mmproj" => mmproj = Some(PathBuf::from(value(&mut values, &flag)?)),
            "--mtmd-bridge" => mtmd_bridge = Some(PathBuf::from(value(&mut values, &flag)?)),
            "--image" => image = Some(PathBuf::from(value(&mut values, &flag)?)),
            "--fixture" => fixture = Some(value(&mut values, &flag)?),
            "--repetitions" => repetitions = parse_usize(&value(&mut values, &flag)?, &flag)?,
            "--output-tokens" => output_tokens = parse_usize(&value(&mut values, &flag)?, &flag)?,
            "--target-backend" => target_backend = parse_backend(&value(&mut values, &flag)?)?,
            "--identity" => identity = Some(value(&mut values, &flag)?),
            "--dry-run" => dry_run = true,
            _ => return Err(format!("unknown argument {flag}")),
        }
    }
    Ok(Args {
        model: model.ok_or("--model is required")?,
        mmproj: mmproj.ok_or("--mmproj is required")?,
        mtmd_bridge: mtmd_bridge.ok_or("--mtmd-bridge is required")?,
        image: image.ok_or("--image is required")?,
        fixture: fixture.ok_or("--fixture is required")?,
        repetitions,
        output_tokens,
        target_backend,
        identity: identity.ok_or("--identity is required")?,
        dry_run,
    })
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

fn argmax(values: &[f32]) -> usize {
    let mut best = 0;
    for index in 1..values.len() {
        if values[index] > values[best] {
            best = index;
        }
    }
    best
}

fn float_digest(rows: &[Vec<f32>]) -> String {
    let mut digest = Sha256::new();
    for value in rows.iter().flatten() {
        digest.update(value.to_le_bytes());
    }
    format!("sha256:{:x}", digest.finalize())
}

fn float_digest_flat(values: &[f32]) -> String {
    let mut digest = Sha256::new();
    for value in values {
        digest.update(value.to_le_bytes());
    }
    format!("sha256:{:x}", digest.finalize())
}

fn token_digest(tokens: &[u32]) -> String {
    let mut digest = Sha256::new();
    for token in tokens {
        digest.update(token.to_le_bytes());
    }
    format!("sha256:{:x}", digest.finalize())
}

fn position_digest(start: usize, end: usize) -> String {
    let mut digest = Sha256::new();
    for position in start..end {
        digest.update((position as u64).to_le_bytes());
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
