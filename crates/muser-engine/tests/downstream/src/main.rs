use std::path::Path;

use muser_engine::{DecodeInput, EngineError, Model, ModelConfig, PrefillBatch, SessionConfig};

fn check_cpu_api(model_path: &Path) -> Result<(), EngineError> {
    let model = Model::load(ModelConfig::new(model_path))?;
    let prompt = model.encode("downstream API check");
    let mut session = model.new_session(SessionConfig {
        max_context: prompt.len() + 2,
    })?;
    session.prefill(PrefillBatch::tokens(prompt))?;
    let token = session.greedy_next_token()?;
    session.decode(DecodeInput { token_id: token })?;
    Ok(())
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn check_metal_api(model_path: &Path) -> Result<(), EngineError> {
    let model = Model::load(ModelConfig::new(model_path))?;
    let _session = model.new_metal_session(SessionConfig::default())?;
    Ok(())
}

fn main() {
    let _cpu_api: fn(&Path) -> Result<(), EngineError> = check_cpu_api;
    #[cfg(all(target_os = "macos", feature = "metal"))]
    let _metal_api: fn(&Path) -> Result<(), EngineError> = check_metal_api;
}
