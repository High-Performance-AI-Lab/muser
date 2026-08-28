#![recursion_limit = "256"]

//! `muser-server` — HTTP/WS serving for a single model, muser's way, plus
//! the `muser up` one-click launcher.
//!
//! Extraction source (PULL-AND-SIMPLIFY, aggressively) for the OpenAI/session
//! surface: `ferrite-server/src/{api/,openai.rs,repl.rs}`. Ferrite's server is
//! large (cascade/speculation routers, constrained decode, matryoshka,
//! spark-prefill protocols, model-manager multi-model slots...) because it
//! serves many models many ways. muser serves **one model**
//! (Muse Glimmer-30B). Keep the HTTP/OpenAI scaffold; **drop the rest**:
//! model-manager, multi-model slots, cascade, speculation-router,
//! spark-prefill-protocol are all **LEFT BEHIND** (single-model server, no
//! VM to route through).
//!
//! ## What's real in this unified build
//!
//! This crate is the reconciliation of two sibling lanes:
//!
//! - **one-click deploy** (`cli`, `up`, `model`, `config`, `banner`):
//!   `muser up` resolves the pinned GGUF (local file or immutable manifest URL
//!   with a real `indicatif` progress bar, size, and SHA-256 verification),
//!   prints an honest real-vs-stub ledger and a banner, opens the dashboard,
//!   and serves until Ctrl+C.
//! - **real telemetry** (`httpd`, `metrics`, `state`, `session`, `timefmt`,
//!   and `muser_kvpack::economics`): Tokio/Axum/Hyper with rustls answers
//!   `GET /snapshot`, Prometheus `GET /metrics`, authenticated WebSocket
//!   `GET /stream`, compatibility SSE `GET /telemetry`, and `GET /healthz`
//!   from real `Mutex`/atomic
//!   guarded process counters, each field carrying a `_honesty` tag
//!   (measured/target/mock) so a live dashboard never has to guess which
//!   numbers are real — see `docs/telemetry.md`.
//!
//! The single `httpd` server carries both surfaces. With `--model`, OpenAI
//! chat performs real CPU/Metal text inference, vision embedding insertion,
//! exact prefix reuse, optional DFlash/ANE drafting, and optional authenticated
//! GX10 prefill through isolated resident slots and one decode-first shared
//! accelerator scheduler. Product claims stay
//! gated on the corresponding release packets.
#![allow(dead_code)]

mod grammar;
mod openai;
mod resumable_stream;
mod session;
mod session_store;

#[path = "axum_httpd.rs"]
mod httpd;
mod metrics;
mod nodes_api;
mod state;
mod timefmt;
mod tls;

mod banner;
mod chat_template;
mod cli;
mod config;
mod model;
mod node;
mod up;

use std::sync::Arc;

use console::style;

use crate::cli::{Cli, Command, ServeArgs};
use crate::state::ServerState;

fn main() {
    let cli = <Cli as clap::Parser>::parse();

    let result = match cli.command {
        Command::Up(args) => up::run(args).map_err(|e| e.friendly_message()),
        Command::Serve(args) => {
            if let Some(seconds) = args.benchmark_deadline_seconds.filter(|value| *value > 0) {
                arm_benchmark_process_deadline(seconds);
            }
            run_serve(args)
        }
        // `muser node add` exits non-zero unless the smoke step passed, so
        // the dashboard's button never reports a node ready on a partial run.
        Command::Node(args) => node::run(args),
        Command::Tls(args) => tls::run(args),
    };

    if let Err(message) = result {
        eprintln!();
        eprintln!("{} {}", style("muser: error:").red().bold(), message);
        eprintln!();
        std::process::exit(1);
    }
}

/// Qualification processes own their hard deadline from before model load.
/// A hung load therefore cannot leave an orphaned accelerator process after
/// the coordinator has stopped waiting. Normal serving never arms this path.
fn arm_benchmark_process_deadline(seconds: u64) {
    std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_secs(seconds));
        eprintln!("muser-server: qualification self-deadline expired");
        std::process::exit(124);
    });
}

/// `muser serve` — the scriptable, no-frills entry point. Binds the same
/// telemetry-and-deploy HTTP surface `muser up` serves, straight from flat
/// `--host` / `--port` / `--model` flags, with none of `up`'s banner,
/// download, or browser-open orchestration. A given `--model` is stat'd for
/// the real on-disk weights size (`cluster.weights_bytes` tagged `measured`);
/// and loaded into the selected standalone CPU or Metal backend.
fn run_serve(args: ServeArgs) -> Result<(), String> {
    let security = httpd::SecurityConfig {
        tls_cert: args.tls_cert.clone(),
        tls_key: args.tls_key.clone(),
        api_key_file: args.api_key_file.clone(),
    };
    httpd::validate_bind_security(&args.host, args.port, &security)?;
    if matches!(args.prefill, cli::PrefillArg::Local) && args.cluster_config.is_some() {
        return Err("--cluster-config requires --prefill auto or remote".into());
    }
    if !matches!(args.prefill, cli::PrefillArg::Local) && args.cluster_config.is_none() {
        return Err("--prefill auto or remote requires --cluster-config".into());
    }
    if args.ane_manifest.is_some() && args.dflash.is_none() {
        return Err("--ane-manifest requires --dflash".into());
    }
    if args.model.is_some() && !matches!(args.backend, cli::BackendArg::Cpu) {
        model::activate_metallib()
            .map_err(|error| format!("resolve pinned Metal runtime: {error}"))?;
    }
    let configured_remote_model_sha256 = args
        .cluster_config
        .as_deref()
        .map(|path| {
            muser_cluster::config::ReceiverConfigV2::load(path)
                .map(|config| config.identity.model_sha256)
                .map_err(|error| format!("remote prefill configuration: {error}"))
        })
        .transpose()?;
    let verified_model_sha256 = args
        .model
        .as_deref()
        .map(|path| {
            let path = std::path::Path::new(path);
            match configured_remote_model_sha256.as_deref() {
                Some(expected) => model::validate_configured_artifact(path, expected),
                None => model::validate_pinned_artifact(path, model::TARGET_ARTIFACT)
                    .map(|artifact| artifact.sha256),
            }
            .map_err(|error| format!("target model identity verification failed: {error}"))
        })
        .transpose()?;
    if let Some(path) = args.mmproj.as_deref() {
        model::validate_pinned_artifact(path, "vision")
            .map_err(|error| format!("vision artifact identity verification failed: {error}"))?;
    }
    if let Some(path) = args.dflash.as_deref() {
        model::validate_pinned_artifact(path, "dflash")
            .map_err(|error| format!("DFlash artifact identity verification failed: {error}"))?;
    }
    let mut state =
        ServerState::new_with_verified_sha256(args.model.as_deref(), verified_model_sha256);
    if let Some(path) = args.model.as_deref() {
        state = state
            .with_inference(
                std::path::Path::new(path),
                args.max_context,
                usize::from(args.parallel),
                args.context_policy.into(),
                args.raw_retain_prefix,
                args.backend.into(),
                args.resident_cache_bytes,
                args.prefix_cache == cli::PrefixCacheArg::On,
            )
            .map_err(|error| error.to_string())?;
        if let Some(mmproj) = args.mmproj.as_deref() {
            state = state
                .with_vision(mmproj, args.mtmd_bridge.as_deref())
                .map_err(|error| error.to_string())?;
        }
        if let Some(kvpack) = args.kvpack_config.as_deref() {
            state = state
                .with_durable_cache(kvpack)
                .map_err(|error| error.to_string())?;
        }
        if let Some(dflash) = args.dflash.as_deref() {
            state = match args.dflash_backend {
                cli::DflashBackendArg::Metal => state.with_dflash(dflash),
                cli::DflashBackendArg::Ane => {
                    with_dflash_ane(state, dflash, args.ane_manifest.as_deref())
                }
                cli::DflashBackendArg::Auto => {
                    auto_dflash_route()?;
                    if args.ane_manifest.is_some() {
                        return Err(
                            "--ane-manifest cannot override the v0.1 Metal-only auto route; use the experimental --dflash-backend ane explicitly"
                                .into(),
                        );
                    }
                    state.with_dflash(dflash)
                }
            }
            .map_err(|error| error.to_string())?;
        }
        if let Some(cluster) = args.cluster_config.as_deref() {
            let mode = match args.prefill {
                cli::PrefillArg::Auto => state::RemotePrefillMode::Auto,
                cli::PrefillArg::Remote => state::RemotePrefillMode::Required,
                cli::PrefillArg::Local => unreachable!("cluster config rejected above"),
            };
            state = state
                .with_remote_prefill(cluster, mode, args.dflash.as_deref())
                .map_err(|error| error.to_string())?;
        }
    } else if args.mmproj.is_some() || args.dflash.is_some() || args.kvpack_config.is_some() {
        return Err("--mmproj, --dflash, and --kvpack-config require --model".into());
    } else if !matches!(args.prefill, cli::PrefillArg::Local) {
        return Err("remote prefill requires --model".into());
    }
    let state = Arc::new(state);
    match (&args.model, state.model_bytes) {
        (Some(path), Some(bytes)) => println!(
            "muser serve: loaded {path} ({bytes} bytes) for real inference"
        ),
        (Some(path), None) => println!(
            "muser serve: --model {path} could not be stat'd — weights unavailable (zero, tagged mock in schema v1)"
        ),
        (None, _) => println!(
            "muser serve: no --model given — weights unavailable (zero, tagged mock in schema v1)"
        ),
    }
    match (
        args.benchmark_shutdown_token.as_deref(),
        args.benchmark_deadline_seconds,
    ) {
        (None, None) => httpd::serve_secure(&args.host, args.port, state, security)
            .map_err(|error| error.to_string()),
        (Some(token), Some(deadline_seconds)) => {
            let host = args
                .host
                .parse::<std::net::IpAddr>()
                .map_err(|_| "qualification server requires a numeric loopback --host")?;
            if !host.is_loopback() {
                return Err("qualification server may bind only to a loopback address".into());
            }
            if token.len() < 32 || deadline_seconds == 0 {
                return Err(
                    "qualification shutdown token must be at least 32 bytes and deadline must be positive"
                        .into(),
                );
            }
            httpd::serve_for_benchmark(
                &args.host,
                args.port,
                state,
                token,
                deadline_seconds,
                security,
            )
            .map_err(|error| error.to_string())
        }
        _ => Err(
            "--benchmark-shutdown-token and --benchmark-deadline-seconds must be supplied together"
                .into(),
        ),
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct DFlashRoutePolicy {
    schema: String,
    status: DFlashRouteStatus,
    auto_route: DFlashAutoRoute,
    ane_gate: DFlashAneGate,
    policy: String,
}

#[derive(serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum DFlashRouteStatus {
    #[serde(rename = "v0.1-metal-only")]
    V0_1MetalOnly,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "lowercase")]
enum DFlashAutoRoute {
    Metal,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct DFlashAneGate {
    required: bool,
    passed: bool,
    same_build_receipt: Option<String>,
}

fn auto_dflash_route() -> Result<&'static str, String> {
    parse_auto_dflash_route(include_str!("../../../release/dflash-route-policy-v1.json"))
}

fn parse_auto_dflash_route(input: &str) -> Result<&'static str, String> {
    let policy: DFlashRoutePolicy = serde_json::from_str(input)
        .map_err(|error| format!("invalid frozen DFlash route policy: {error}"))?;
    if policy.schema != "muser.dflash-route-policy.v1" {
        return Err("invalid frozen DFlash route-policy schema".into());
    }
    if policy.policy.trim().is_empty()
        || policy.status != DFlashRouteStatus::V0_1MetalOnly
        || policy.ane_gate.required
        || policy.ane_gate.passed
        || policy.ane_gate.same_build_receipt.is_some()
    {
        return Err("frozen v0.1 DFlash route policy is not Metal-only".into());
    }
    match policy.auto_route {
        DFlashAutoRoute::Metal => Ok("metal"),
    }
}

fn with_dflash_ane(
    state: ServerState,
    dflash: &std::path::Path,
    manifest: Option<&std::path::Path>,
) -> Result<ServerState, state::InferenceLoadError> {
    #[cfg(all(target_os = "macos", feature = "ane-coreml"))]
    {
        state.with_dflash_ane(dflash, manifest)
    }
    #[cfg(not(all(target_os = "macos", feature = "ane-coreml")))]
    {
        let _ = (state, dflash, manifest);
        Err(state::InferenceLoadError::CoreMl(
            "this binary was built without the ane-coreml feature".into(),
        ))
    }
}

#[cfg(test)]
mod route_policy_tests {
    use super::parse_auto_dflash_route;

    const POLICY_TEXT: &str = "v0.1 auto routing is permanently Metal";

    fn policy(status: &str, route: &str, required: bool, passed: bool, receipt: &str) -> String {
        format!(
            r#"{{
                "schema":"muser.dflash-route-policy.v1",
                "status":{status:?},
                "auto_route":{route:?},
                "ane_gate":{{"required":{required},"passed":{passed},"same_build_receipt":{receipt}}},
                "policy":{POLICY_TEXT:?}
            }}"#
        )
    }

    #[test]
    fn v0_1_policy_routes_auto_to_metal() {
        let value = policy("v0.1-metal-only", "metal", false, false, "null");
        assert_eq!(parse_auto_dflash_route(&value).unwrap(), "metal");
    }

    #[test]
    fn v0_1_auto_refuses_ane_and_qualification_evidence() {
        for invalid in [
            policy("v0.1-metal-only", "ane", false, false, "null"),
            policy("v0.1-metal-only", "metal", true, false, "null"),
            policy(
                "v0.1-metal-only",
                "metal",
                false,
                true,
                r#""ane-lane.json""#,
            ),
            policy("ane-qualified", "metal", false, false, "null"),
        ] {
            assert!(parse_auto_dflash_route(&invalid).is_err());
        }
    }

    #[test]
    fn metal_auto_route_rejects_qualification_claims_and_unknown_fields() {
        let unknown = policy("v0.1-metal-only", "metal", false, false, "null").replacen(
            "{",
            r#"{"unexpected":1,"#,
            1,
        );
        assert!(parse_auto_dflash_route(&unknown).is_err());
    }
}
