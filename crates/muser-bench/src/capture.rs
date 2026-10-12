//! CPU activation fingerprints for layer-local fresh-llama parity diagnosis.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;

use muser_engine::capture::Capture;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("muser-capture-evidence: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut argv = std::env::args().skip(1);
    let mut model = None;
    let mut fixture = None;
    let mut output = None;
    while let Some(argument) = argv.next() {
        let value = argv
            .next()
            .ok_or_else(|| format!("{argument} requires a value"))?;
        match argument.as_str() {
            "--model" => model = Some(PathBuf::from(value)),
            "--token-fixture" => fixture = Some(PathBuf::from(value)),
            "--out" => output = Some(PathBuf::from(value)),
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    let model = model.ok_or("--model is required")?;
    let fixture = fixture.ok_or("--token-fixture is required")?;
    let output = output.ok_or("--out is required")?;
    let tokens = std::fs::read_to_string(&fixture)
        .map_err(|error| format!("cannot read {}: {error}", fixture.display()))?
        .split_whitespace()
        .map(|field| {
            field
                .parse::<u32>()
                .map_err(|error| format!("invalid token {field:?}: {error}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if tokens.is_empty() {
        return Err("token fixture is empty".into());
    }
    let mut oracle =
        muser_engine::loader::load(&model, tokens.len()).map_err(|error| error.to_string())?;
    if tokens
        .iter()
        .any(|token| *token as usize >= oracle.cfg.vocab_size)
    {
        return Err("token fixture contains an out-of-vocabulary token".into());
    }
    let mut capture = Capture::default();
    let logits = oracle.forward(&tokens, Some(&mut capture));
    if logits.iter().any(|value| !value.is_finite()) {
        return Err("captured forward produced nonfinite logits".into());
    }
    let payload = capture.to_json();
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)
        .map_err(|error| format!("cannot create {}: {error}", output.display()))?;
    file.write_all(payload.as_bytes())
        .and_then(|_| file.write_all(b"\n"))
        .and_then(|_| file.flush())
        .map_err(|error| error.to_string())?;
    file.sync_all().map_err(|error| error.to_string())?;
    Ok(())
}
