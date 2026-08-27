//! M0 offline diagnostic for the GX10 DFlash draft-context divergence.
//!
//! Recomputes the draft context K at layer 0 / token 0 / head 0 / element 0
//! from the real Mac target-prefill hidden states and the draft weights, in
//! plain f32 on the CPU, then evaluates the k_norm head mean and the final
//! element under both candidate epsilons (the engine hardcodes 1e-6; the
//! draft GGUF metadata and llama.cpp both use the artifact's 1e-5).
//! Compares against the retained v3 diagnostic bits:
//!   local  0x3ea8f873
//!   remote 0x3ea8f85f

use std::path::{Path, PathBuf};
use std::{fs::OpenOptions, io::Write};

use muser_engine::api::{Model, ModelConfig, PrefillBatch, SessionConfig};
use muser_engine::dflash::DFlashWeights;
use muser_engine::metal::dflash::MetalDFlashForward;
use sha2::{Digest, Sha256};

const LOCAL_BITS: u32 = 0x3ea8f873;
const REMOTE_BITS: u32 = 0x3ea8f85f;

fn rms_serial(values: &[f32], eps: f32) -> f32 {
    let mut sum = 0.0f32;
    for &v in values {
        sum = f32::mul_add(v, v, sum);
    }
    1.0 / (sum / values.len() as f32 + eps).sqrt()
}

fn read_f32_le(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|error| error.to_string())?;
    if !bytes.len().is_multiple_of(std::mem::size_of::<f32>()) {
        return Err(format!(
            "{} has {} bytes, not an integral number of f32 values",
            path.display(),
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect())
}

fn write_f32_le(path: &Path, values: &[f32]) -> Result<(String, usize), String> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| error.to_string())?;
    let mut digest = Sha256::new();
    for value in values {
        let bytes = value.to_le_bytes();
        file.write_all(&bytes).map_err(|error| error.to_string())?;
        digest.update(bytes);
    }
    file.sync_all().map_err(|error| error.to_string())?;
    Ok((format!("{:x}", digest.finalize()), values.len() * 4))
}

fn main() -> Result<(), String> {
    let mut model_path = None;
    let mut dflash_path = None;
    let mut fixture = None;
    let mut hidden_output = None;
    let mut hidden_input = None;
    let mut boundary_output_dir = None;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model_path = Some(PathBuf::from(&args[i]));
            }
            "--dflash" => {
                i += 1;
                dflash_path = Some(PathBuf::from(&args[i]));
            }
            "--prompt-token-fixture" => {
                i += 1;
                fixture = Some(PathBuf::from(&args[i]));
            }
            "--hidden-output" => {
                i += 1;
                hidden_output = Some(PathBuf::from(&args[i]));
            }
            "--hidden-input" => {
                i += 1;
                hidden_input = Some(PathBuf::from(&args[i]));
            }
            "--boundary-output-dir" => {
                i += 1;
                boundary_output_dir = Some(PathBuf::from(&args[i]));
            }
            other => return Err(format!("unknown argument {other}")),
        }
        i += 1;
    }
    let dflash_path = dflash_path.ok_or("--dflash required")?;

    let weights = DFlashWeights::load(&dflash_path).map_err(|e| e.to_string())?;
    let layer_ids = weights.config.dflash_config.target_layer_ids.clone();
    let hidden = weights.config.hidden_size;
    let draft_eps = weights.config.rms_norm_eps as f32;
    println!("target_layer_ids={layer_ids:?} hidden={hidden} gguf_eps={draft_eps}");

    if let Some(input_path) = hidden_input {
        if model_path.is_some() || fixture.is_some() || hidden_output.is_some() {
            return Err(
                "--hidden-input cannot combine with --model, --prompt-token-fixture, or --hidden-output"
                    .into(),
            );
        }
        let output_dir =
            boundary_output_dir.ok_or("--boundary-output-dir is required with --hidden-input")?;
        std::fs::create_dir(&output_dir).map_err(|error| error.to_string())?;
        let target_hidden = read_f32_le(&input_path)?;
        let row_width = layer_ids.len() * hidden;
        if target_hidden.is_empty() || !target_hidden.len().is_multiple_of(row_width) {
            return Err(format!(
                "hidden input has {} values, expected a positive multiple of {row_width}",
                target_hidden.len()
            ));
        }
        let n_context = target_hidden.len() / row_width;
        let mut forward =
            MetalDFlashForward::new(&weights, n_context).map_err(|error| error.to_string())?;
        let probe = forward
            .probe_context_boundaries(&target_hidden, n_context, 0)
            .map_err(|error| error.to_string())?;
        let mut outputs = serde_json::Map::new();
        for (name, values) in [
            ("fc_out", probe.fc_out),
            ("enc_norm_out", probe.enc_norm_out),
            ("k_projected_layer0", probe.k_projected_layer0),
            ("v_projected_layer0", probe.v_projected_layer0),
            ("k_normed_layer0", probe.k_normed_layer0),
            ("k_rope_layer0", probe.k_rope_layer0),
        ] {
            let path = output_dir.join(format!("{name}.f32"));
            let first_bits = values
                .first()
                .ok_or_else(|| format!("{name} is unexpectedly empty"))?
                .to_bits();
            let (sha256, bytes) = write_f32_le(&path, &values)?;
            outputs.insert(
                name.into(),
                serde_json::json!({
                    "path": path,
                    "sha256": sha256,
                    "bytes": bytes,
                    "first_bits": format!("0x{first_bits:08x}"),
                }),
            );
        }
        println!(
            "{}",
            serde_json::json!({
                "schema": "muser.mac-dflash-context-boundaries.v1",
                "input": input_path,
                "context_tokens": n_context,
                "target_layers": layer_ids,
                "hidden_size": hidden,
                "outputs": outputs,
                "seal_eligible": false,
            })
        );
        return Ok(());
    }
    if boundary_output_dir.is_some() {
        return Err("--boundary-output-dir requires --hidden-input".into());
    }

    let model_path = model_path.ok_or("--model required")?;
    let fixture = fixture.ok_or("--prompt-token-fixture required")?;
    let tokens = std::fs::read_to_string(&fixture)
        .map_err(|e| e.to_string())?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            line.trim()
                .parse::<u32>()
                .map_err(|_| "fixture line is not a token id".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if tokens.len() < 2 {
        return Err("fixture needs a cached prefix and boundary token".into());
    }
    let token0 = tokens[0];

    let model = Model::load(ModelConfig::new(&model_path)).map_err(|e| e.to_string())?;
    let mut session = model
        .new_metal_session(SessionConfig { max_context: 4096 })
        .map_err(|e| e.to_string())?;
    if let Some(output) = hidden_output {
        let cached = &tokens[..tokens.len() - 1];
        let (_logits, rows) = session
            .prefill_batch_capturing_layers(PrefillBatch::tokens(cached.to_vec()), &layer_ids)
            .map_err(|e| e.to_string())?;
        let expected = cached.len() * layer_ids.len() * hidden;
        if rows.len() != expected {
            return Err(format!(
                "captured {} target hidden values, expected {expected}",
                rows.len()
            ));
        }
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&output)
            .map_err(|e| e.to_string())?;
        let mut digest = Sha256::new();
        for value in rows {
            let bytes = value.to_le_bytes();
            file.write_all(&bytes).map_err(|e| e.to_string())?;
            digest.update(bytes);
        }
        file.sync_all().map_err(|e| e.to_string())?;
        println!(
            "{}",
            serde_json::json!({
                "schema": "muser.mac-dflash-target-features.v1",
                "output": output,
                "sha256": format!("{:x}", digest.finalize()),
                "bytes": expected * std::mem::size_of::<f32>(),
                "cached_tokens": cached.len(),
                "target_layers": layer_ids,
                "hidden_size": hidden,
                "dtype": "f32_le",
                "layout": "token-major-selected-layer-major-hidden",
                "seal_eligible": false,
            })
        );
        return Ok(());
    }
    let (_logits, rows) = session
        .prefill_batch_capturing_layers(PrefillBatch::tokens(vec![token0]), &layer_ids)
        .map_err(|e| e.to_string())?;
    assert_eq!(rows.len(), layer_ids.len() * hidden);

    // fc: concat the five layer rows, project 33280 -> 6656 (plain f32).
    let fc_in = &rows[..];
    let fc_out: Vec<f32> = weights
        .fc_weight
        .chunks_exact(fc_in.len())
        .map(|row| row.iter().zip(fc_in).map(|(a, b)| a * b).sum())
        .collect();
    assert_eq!(fc_out.len(), hidden);

    // hidden_norm (enc.output_norm.weight) with the GGUF eps.
    let inv = rms_serial(&fc_out, draft_eps);
    let normed: Vec<f32> = fc_out
        .iter()
        .zip(&weights.hidden_norm_weight)
        .map(|(v, w)| v * inv * w)
        .collect();

    // k_proj layer 0, head 0 (first 128 outputs).
    let layer0 = &weights.layers[0];
    let krow: Vec<f32> = layer0
        .k_proj_weight
        .chunks_exact(hidden)
        .take(128)
        .map(|row| row.iter().zip(&normed).map(|(a, b)| a * b).sum())
        .collect();
    let mean: f32 = krow.iter().map(|v| v * v).sum::<f32>() / 128.0;
    println!("k_norm head0 mean128 = {mean:.8}");

    for eps in [1e-6f32, 1e-5] {
        let scale = 1.0 / (mean + eps).sqrt();
        let k0 = krow[0] * scale * layer0.k_norm_weight[0];
        let bits = k0.to_bits();
        println!(
            "eps={eps:.0e}  K[0,0,0,0]={k0:.9} bits=0x{bits:08x}  vs local 0x{LOCAL_BITS:08x} ({} ulp)  remote 0x{REMOTE_BITS:08x} ({} ulp)",
            bits as i64 - LOCAL_BITS as i64,
            bits as i64 - REMOTE_BITS as i64,
        );
    }
    Ok(())
}
