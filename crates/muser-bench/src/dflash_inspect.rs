//! CPU-only structural load gate for the official DFlash artifact.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use muser_engine::dflash::DFlashWeights;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("muser-dflash-inspect: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut values = std::env::args().skip(1);
    let mut artifact: Option<PathBuf> = None;
    while let Some(flag) = values.next() {
        match flag.as_str() {
            "--artifact" => {
                artifact = Some(PathBuf::from(
                    values.next().ok_or("--artifact requires a path")?,
                ));
            }
            _ => return Err(format!("unknown argument {flag}")),
        }
    }
    let artifact = artifact.ok_or("--artifact is required")?;
    let bytes = artifact
        .metadata()
        .map_err(|error| format!("cannot stat {}: {error}", artifact.display()))?
        .len();
    let started = Instant::now();
    let weights = DFlashWeights::load(&artifact).map_err(|error| error.to_string())?;
    let config = &weights.config;
    println!(
        "{}",
        serde_json::json!({
            "schema": "muser.dflash-artifact-inspection.v1",
            "artifact": artifact,
            "artifact_bytes": bytes,
            "format": config.dtype,
            "load_ns": u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            "parameter_count": weights.param_count(),
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "head_dim": config.head_dim,
            "attention_heads": config.num_attention_heads,
            "kv_heads": config.num_key_value_heads,
            "assistant_layers": config.num_hidden_layers,
            "target_layers": config.dflash_config.target_layer_ids,
            "block_size": config.block_size,
            "vocab_size": config.vocab_size,
            "max_position_embeddings": config.max_position_embeddings,
            "finite_weights": weights.fc_weight.iter()
                .chain(&weights.hidden_norm_weight)
                .chain(&weights.norm_weight)
                .chain(weights.layers.iter().flat_map(|layer| {
                    layer.input_layernorm_weight.iter()
                        .chain(&layer.post_attention_layernorm_weight)
                        .chain(&layer.q_proj_weight)
                        .chain(&layer.k_proj_weight)
                        .chain(&layer.v_proj_weight)
                        .chain(&layer.o_proj_weight)
                        .chain(&layer.q_norm_weight)
                        .chain(&layer.k_norm_weight)
                        .chain(&layer.gate_proj_weight)
                        .chain(&layer.up_proj_weight)
                        .chain(&layer.down_proj_weight)
                }))
                .all(|value| value.is_finite()),
        })
    );
    Ok(())
}
