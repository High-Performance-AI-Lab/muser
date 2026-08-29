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

use std::net::ToSocketAddrs as _;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

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
    node: Option<String>,
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
        let node = args.node.clone().or_else(|| file_cfg.node.clone());
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
            node,
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
    // The public release is the native NVFP4 topology. Bare `muser up`
    // therefore resumes the newest compatible enrollment or opens the setup
    // dashboard. Local inference remains explicit via --local, --gguf,
    // --hf-repo, or a local-model entry in muser.toml.
    let local_requested = args.local
        || args.gguf.is_some()
        || args.hf_repo.is_some()
        || (effective.node.is_none()
            && (file_cfg.gguf_path.is_some() || file_cfg.hf_repo.is_some()));
    let selected_node = if local_requested {
        None
    } else if let Some(name) = effective.node.clone() {
        Some(name)
    } else {
        crate::node::default_serving_node().map_err(UpError::Policy)?
    };

    // A setup process may install a remote receiver after HTTP worker threads
    // exist, so select the qualified cross-vendor route before any of those
    // threads start. Environment mutation later would be unsafe.
    crate::prepare_remote_prefill(!local_requested).map_err(UpError::Policy)?;
    httpd::validate_bind_security(&effective.host, effective.port, &effective.security)
        .map_err(UpError::Policy)?;
    model::activate_metallib()?;

    let startup = Instant::now();
    if !local_requested && selected_node.is_none() {
        return run_setup_dashboard(&args, effective, startup);
    }

    // Keep enrollment, daemon restarts, qualification, and a live required
    // receiver mutually exclusive across separate CLI/dashboard processes.
    // The producer itself is single-flight; rotating it under this server
    // would turn an otherwise bounded request into an unexplained failure.
    let _topology_lock = selected_node
        .as_deref()
        .map(|name| {
            let home = crate::node::muser_home()?;
            crate::node::registry::OperationLock::acquire(
                &home,
                &format!("serving required remote-prefill node {name}"),
            )
        })
        .transpose()
        .map_err(UpError::Policy)?;
    let (state, remote_name) = if let Some(name) = selected_node.as_deref() {
        println!(
            "{}",
            style(format!("Resuming enrolled NVFP4 node {name}")).bold()
        );
        let report = |detail: &str| println!("  {detail}");
        (build_remote_runtime(name, &report)?, Some(name))
    } else {
        println!("{}", style("Resolving Muse Glimmer-30B weights").bold());
        let resolved = model::resolve(ResolveRequest {
            repository: &effective.hf_repository,
            target_path: &effective.gguf_path,
        })?;
        let source = match &resolved.source {
            ModelSource::Local => ("local", None),
            ModelSource::Downloaded { url } => ("downloaded", Some(url.clone())),
        };
        println!();
        println!("{}", style("Loading the Metal decoder").bold());
        let verified_sha256 = model::pinned_artifact(model::TARGET_ARTIFACT)?.sha256;
        let model_text = resolved
            .path
            .to_str()
            .ok_or_else(|| UpError::Policy("model path is not valid UTF-8".into()))?;
        let state = ServerState::new_with_verified_sha256(Some(model_text), Some(verified_sha256))
            .with_provenance(source.0, source.1)
            .with_inference(
                &resolved.path,
                131_072,
                4,
                crate::state::ContextPolicy::Shift,
                256,
                crate::state::BackendMode::Auto,
                8 * 1024 * 1024 * 1024,
                true,
            )?;
        (state, None)
    };
    let state = Arc::new(state);

    print_status_ledger(remote_name);

    let scheme = if effective.security.tls_cert.is_some() {
        "https"
    } else {
        "http"
    };
    let url = format!("{scheme}://{}:{}", effective.host, effective.port);
    if !args.no_open {
        open_browser_when_ready(&url, &effective.host, effective.port);
    }

    println!(
        "{} muser ready in {:.1}s \u{2192} open the dashboard at {}",
        style("\u{2705}").green(),
        startup.elapsed().as_secs_f64(),
        style(&url).bold().underlined()
    );
    println!("   (Ctrl+C to stop)");
    println!();

    httpd::serve_secure(&effective.host, effective.port, state, effective.security)?;
    Ok(())
}

fn build_remote_runtime(name: &str, report: &dyn Fn(&str)) -> Result<ServerState, UpError> {
    report("checking the enrolled runtime and receiver configuration");
    let target = crate::node::serving_target(name).map_err(UpError::Policy)?;
    let cluster = muser_cluster::config::ReceiverConfigV2::load(&target.cluster_config)
        .map_err(|error| UpError::Policy(format!("node {name} receiver configuration: {error}")))?;
    if cluster.identity.model_sha256 != target.model_sha256 {
        return Err(UpError::Policy(format!(
            "node {name} receiver identity differs from its enrolled decoder"
        )));
    }
    let verified = if target.model_validation_current {
        report("using the unchanged decoder's prior full SHA-256 verification");
        target.model_sha256.clone()
    } else {
        report("decoder changed or predates validation stamps; verifying its full SHA-256 once");
        let verified =
            model::validate_configured_artifact(&target.model_path, &target.model_sha256)?;
        crate::node::remember_consumer_validation(name, &target.model_path, &verified)
            .map_err(UpError::Policy)?;
        verified
    };
    let model_text = target
        .model_path
        .to_str()
        .ok_or_else(|| UpError::Policy("model path is not valid UTF-8".into()))?;
    report("loading four Metal decoder slots");
    let state = ServerState::new_with_verified_sha256(Some(model_text), Some(verified))
        .with_provenance("enrolled-native", None)
        .with_inference(
            &target.model_path,
            131_072,
            4,
            crate::state::ContextPolicy::Shift,
            256,
            crate::state::BackendMode::Auto,
            8 * 1024 * 1024 * 1024,
            true,
        )?;
    report("binding the authenticated remote-prefill receiver");
    state
        .with_remote_prefill(
            &target.cluster_config,
            crate::state::RemotePrefillMode::Required,
            None,
        )
        .map_err(UpError::Inference)
}

fn run_setup_dashboard(
    args: &UpArgs,
    effective: EffectiveConfig,
    startup: Instant,
) -> Result<(), UpError> {
    let state = Arc::new(ServerState::new(None));
    let activation_state = Arc::clone(&state);
    let serving_lock: Arc<Mutex<Option<crate::node::registry::OperationLock>>> =
        Arc::new(Mutex::new(None));
    let activation_lock = Arc::clone(&serving_lock);
    let activator: crate::nodes_api::NodeActivator = Arc::new(move |name, reporter| {
        if activation_state.inference.is_some() {
            return Err("the Mac decoder is already running".into());
        }
        activation_state.mark_runtime_loading(name, "checking the enrolled topology");

        let (heartbeat_stop, heartbeat_receiver) = std::sync::mpsc::channel::<()>();
        let heartbeat_reporter = Arc::clone(&reporter);
        let heartbeat = std::thread::spawn(move || {
            while heartbeat_receiver
                .recv_timeout(std::time::Duration::from_secs(15))
                .is_err()
            {
                heartbeat_reporter("info", "the Mac decoder is still loading");
            }
        });

        let result = (|| -> Result<(), String> {
            reporter("info", "transferring the topology from setup to serving");
            let home = crate::node::muser_home()?;
            let lock = crate::node::registry::OperationLock::acquire(
                &home,
                &format!("serving required remote-prefill node {name}"),
            )?;
            let state_for_progress = Arc::clone(&activation_state);
            let reporter_for_progress = Arc::clone(&reporter);
            let progress = |detail: &str| {
                state_for_progress.mark_runtime_loading(name, detail);
                reporter_for_progress("info", detail);
            };
            let prepared =
                build_remote_runtime(name, &progress).map_err(|error| error.friendly_message())?;
            activation_state
                .install_prepared_runtime(prepared, name)
                .map_err(|error| error.to_string())?;
            *activation_lock
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(lock);
            Ok(())
        })();

        let _ = heartbeat_stop.send(());
        let _ = heartbeat.join();
        if let Err(error) = &result {
            activation_state.mark_runtime_failed(name, error);
        }
        result
    });

    let scheme = if effective.security.tls_cert.is_some() {
        "https"
    } else {
        "http"
    };
    let url = format!("{scheme}://{}:{}", effective.host, effective.port);
    if !args.no_open {
        open_browser_when_ready(&url, &effective.host, effective.port);
    }
    println!(
        "{} setup ready in {:.1}s \u{2192} open {} and choose Add node",
        style("\u{2705}").green(),
        startup.elapsed().as_secs_f64(),
        style(&url).bold().underlined()
    );
    println!("   The same dashboard becomes the inference server automatically; no restart.");
    println!("   (Ctrl+C to stop)");
    println!();

    httpd::serve_secure_with_node_activation(
        &effective.host,
        effective.port,
        state,
        activator,
        effective.security,
    )?;
    // The lock is intentionally kept live until the HTTP server exits.
    drop(serving_lock);
    Ok(())
}

fn print_status_ledger(remote_node: Option<&str>) {
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
    if let Some(name) = remote_node {
        println!(
            "  {} authenticated NVFP4 prefill through enrolled node {} (required; no silent local fallback)",
            style("\u{2713}").green(),
            name
        );
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

fn open_browser_when_ready(url: &str, host: &str, port: u16) {
    let url = url.to_string();
    let host = dashboard_connect_host(host).to_string();
    std::thread::spawn(move || {
        let deadline = Instant::now() + std::time::Duration::from_secs(10);
        while Instant::now() < deadline {
            let listening = (host.as_str(), port)
                .to_socket_addrs()
                .ok()
                .is_some_and(|addresses| {
                    addresses.into_iter().any(|address| {
                        std::net::TcpStream::connect_timeout(
                            &address,
                            std::time::Duration::from_millis(250),
                        )
                        .is_ok()
                    })
                });
            if listening {
                try_open_browser(&url);
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        eprintln!("muser: dashboard did not begin listening within 10s; open {url} manually");
    });
}

fn dashboard_connect_host(host: &str) -> &str {
    match host {
        "0.0.0.0" => "127.0.0.1",
        "::" | "[::]" => "::1",
        _ => host,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn browser_probe_uses_a_connectable_address_for_wildcard_binds() {
        assert_eq!(dashboard_connect_host("0.0.0.0"), "127.0.0.1");
        assert_eq!(dashboard_connect_host("::"), "::1");
        assert_eq!(dashboard_connect_host("127.0.0.1"), "127.0.0.1");
        assert_eq!(dashboard_connect_host("muser.local"), "muser.local");
    }
}
