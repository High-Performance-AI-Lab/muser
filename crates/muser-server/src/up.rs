//! `muser up` — resolve the pinned model and launch the engine plus dashboard.
//!
//! What this module actually does, end to end:
//! 1. Print the banner.
//! 2. Resolve the GGUF: use the local file if present, else download it
//!    from the pinned immutable URL with a real progress bar, verifying the
//!    manifest revision, size, and SHA-256 — see `model.rs`.
//! 3. Print an honest live/unavailable ledger.
//! 4. Try to open the dashboard in a browser, then print the one-line
//!    "muser ready" banner.
//! 5. Bind and serve — see `httpd.rs` — until Ctrl+C.
//!
//! `docs/muser-architecture.md` is the source of truth for what's wired;
//! this module's own terminal output stays in sync with it rather than
//! aspirationally describing Phase 2-5 as done.

use std::path::PathBuf;
use std::sync::Arc;

use console::style;

use crate::banner;
use crate::cli::UpArgs;
use crate::config::{ConfigError, FileConfig};
use crate::httpd::{self, ServeError};
use crate::model::{self, ModelError, ModelSource, ResolveRequest};
use crate::state::ServerState;

#[derive(Debug)]
pub enum UpError {
    Config(ConfigError),
    Model(ModelError),
    Serve(ServeError),
    Inference(crate::state::InferenceLoadError),
    Policy(String),
}

impl From<ConfigError> for UpError {
    fn from(e: ConfigError) -> Self {
        UpError::Config(e)
    }
}
impl From<ModelError> for UpError {
    fn from(e: ModelError) -> Self {
        UpError::Model(e)
    }
}
impl From<ServeError> for UpError {
    fn from(e: ServeError) -> Self {
        UpError::Serve(e)
    }
}
impl From<crate::state::InferenceLoadError> for UpError {
    fn from(e: crate::state::InferenceLoadError) -> Self {
        UpError::Inference(e)
    }
}

impl UpError {
    /// The message to print before exiting non-zero. `ModelError` gets the
    /// long, "here's exactly what to do" treatment (see
    /// `model::friendly_error_message`); the rest get their `Display`.
    pub fn friendly_message(&self) -> String {
        match self {
            UpError::Model(e) => model::friendly_error_message(e),
            UpError::Config(e) => e.to_string(),
            UpError::Serve(e) => e.to_string(),
            UpError::Inference(e) => e.to_string(),
            UpError::Policy(e) => e.clone(),
        }
    }
}

struct EffectiveConfig {
    gguf_path: PathBuf,
    hf_repository: String,
    host: String,
    port: u16,
    security: httpd::SecurityConfig,
}

impl EffectiveConfig {
    /// Precedence, highest first: CLI flag / env var (already folded into
    /// `args` by clap's `env = "MUSER_..."`) > `muser.toml` > built-in
    /// default.
    fn resolve(args: &UpArgs, file_cfg: &FileConfig) -> Result<Self, ModelError> {
        let gguf_path = args
            .gguf
            .clone()
            .or_else(|| file_cfg.gguf_path.clone())
            .map(Ok)
            .unwrap_or_else(model::default_model_path)?;
        let hf_repository = args
            .hf_repo
            .clone()
            .or_else(|| file_cfg.hf_repo.clone())
            .map(Ok)
            .unwrap_or_else(model::pinned_repository_id)?;
        let host = args
            .host
            .clone()
            .or_else(|| file_cfg.host.clone())
            .unwrap_or_else(|| "127.0.0.1".to_string());
        // The dashboard is served same-origin; no file:// endpoint override
        // is supported during containment.
        let port = args.port.or(file_cfg.port).unwrap_or(4949);
        let security = httpd::SecurityConfig {
            tls_cert: args.tls_cert.clone().or_else(|| file_cfg.tls_cert.clone()),
            tls_key: args.tls_key.clone().or_else(|| file_cfg.tls_key.clone()),
            api_key_file: args
                .api_key_file
                .clone()
                .or_else(|| file_cfg.api_key_file.clone()),
        };

        Ok(EffectiveConfig {
            gguf_path,
            hf_repository,
            host,
            port,
            security,
        })
    }
}

pub fn run(args: UpArgs) -> Result<(), UpError> {
    banner::print_banner();

    let file_cfg = FileConfig::load(args.config.as_deref())?;
    let effective = EffectiveConfig::resolve(&args, &file_cfg)?;
    httpd::validate_bind_security(&effective.host, effective.port, &effective.security)
        .map_err(UpError::Policy)?;
    model::activate_metallib()?;

    println!("{}", style("Resolving Muse Glimmer-30B weights").bold());
    let resolved = model::resolve(ResolveRequest {
        repository: &effective.hf_repository,
        target_path: &effective.gguf_path,
    })?;
    println!();

    print_status_ledger();

    let scheme = if effective.security.tls_cert.is_some() {
        "https"
    } else {
        "http"
    };
    let url = format!("{scheme}://{}:{}", effective.host, effective.port);
    if !args.no_open {
        try_open_browser(&url);
    }

    println!(
        "{} muser ready \u{2192} open the dashboard at {}",
        style("\u{2705}").green(),
        style(&url).bold().underlined()
    );
    println!("   (Ctrl+C to stop)");
    println!();

    // Hand the resolved model to the telemetry server as real, honest
    // provenance: ServerState re-stats the file for `cluster.weights_bytes`
    // (tagged measured), and `.with_provenance` records where the bytes came
    // from for `GET /health` — never a hardcoded/implied source.
    let (source_label, source_url) = match &resolved.source {
        ModelSource::Local => ("local", None),
        ModelSource::Downloaded { url } => ("downloaded", Some(url.clone())),
    };
    let verified_sha256 = model::pinned_artifact(model::TARGET_ARTIFACT)?.sha256;
    let state = Arc::new(
        ServerState::new_with_verified_sha256(resolved.path.to_str(), Some(verified_sha256))
            .with_provenance(source_label, source_url)
            .with_inference(
                &resolved.path,
                131_072,
                4,
                crate::state::ContextPolicy::Shift,
                256,
                crate::state::BackendMode::Auto,
                8 * 1024 * 1024 * 1024,
                true,
            )?,
    );
    httpd::serve_secure(&effective.host, effective.port, state, effective.security)?;
    Ok(())
}

fn print_status_ledger() {
    println!("{}", style("what's real right now:").bold());
    for line in [
        "GGUF resolution / download, with size + checksum verification",
        "an Axum/rustls HTTP server answering strict inference, session, telemetry, health, and management routes",
        "up to four isolated Muse slots with OpenAI-compatible JSON/SSE streaming and continuous decode batching",
        "live telemetry from real process counters — kv / economics / sessions / TTFT / inter-token latency (honestly zero until traffic), every field _honesty-tagged measured/target/mock",
        "the live-only status dashboard (web/muser-dashboard.html); no simulated metrics",
    ] {
        println!("  {} {}", style("\u{2713}").green(), line);
    }
    println!("{}", style("not measured by this process:").bold());
    for line in [
        "node utilization, power, and temperature telemetry",
        "wire egress telemetry",
    ] {
        println!("  {} {}", style("\u{25cb}").yellow(), line);
    }
    println!();
}

fn try_open_browser(url: &str) {
    #[cfg(target_os = "macos")]
    {
        match std::process::Command::new("open").arg(url).status() {
            Ok(status) if status.success() => {
                println!("  opening {url} in your default browser...");
            }
            _ => {
                println!("  (couldn't auto-open a browser — open {url} yourself)");
            }
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        println!("  open {url} in your browser to see the dashboard");
    }
}
