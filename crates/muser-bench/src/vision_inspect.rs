//! CPU-only structural load gate for the pinned official Muse mmproj.

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;
use std::{fs::File, io::Read};

use muser_engine::vision::VisionModel;
use sha2::{Digest, Sha256};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("muser-vision-inspect: {error}");
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
    let artifact_bytes = artifact
        .metadata()
        .map_err(|error| format!("stat {}: {error}", artifact.display()))?
        .len();
    let artifact_sha256 = file_sha256(&artifact)?;
    let started = Instant::now();
    let model = VisionModel::load(&artifact).map_err(|error| error.to_string())?;
    let config = &model.config;
    println!(
        "{}",
        serde_json::json!({
            "schema": "muser.vision-artifact-inspection.v1",
            "artifact": artifact,
            "artifact_bytes": artifact_bytes,
            "artifact_sha256": artifact_sha256,
            "load_ns": u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            "route": model.route_identity(),
            "embedding_dim": config.embedding_dim,
            "intermediate_dim": config.intermediate_dim,
            "blocks": config.n_layers,
            "heads": config.n_heads,
            "head_dim": config.head_dim(),
            "patch_size": config.patch_size,
            "output_dim": config.output_dim,
            "position_grid": config.position_grid,
            "image_mean": config.image_mean,
            "image_std": config.image_std,
        })
    );
    Ok(())
}

fn file_sha256(path: &std::path::Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 8 * 1024 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| format!("read {}: {error}", path.display()))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}
