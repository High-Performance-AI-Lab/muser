//! Command-line surface for the `muser` binary.
//!
//! `muser up` is the one-click launcher (Deliverable A, `feat/one-click-deploy`):
//! resolve the pinned GGUF (local file or immutable manifest download), start whatever
//! of the engine is real today and print one clear
//! "ready" line pointing at the dashboard. See `up.rs` for the orchestration
//! and `docs/muser-architecture.md` for what "real today" currently means.

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

use crate::node::registry::ProducerKind;
use crate::state::BackendMode;

#[derive(Debug, Parser)]
#[command(
    name = "muser",
    version,
    about = "muser — inference you can see through.",
    long_about = "muser — a standalone Muse Glimmer engine.\n\nText, vision, Metal DFlash, exact prefix reuse, and authenticated GX10 handoff\nare the v0.1 boundary. Public-CoreML ANE routing is experimental and explicitly\nselected only; auto remains Metal. Product claims remain gated on the mandatory\nunsealed qualification matrix and one atomic final bundle."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Resolve the pinned model and launch the engine plus dashboard.
    Up(UpArgs),

    /// Bind the HTTP/telemetry surface directly from flat flags — the
    /// scriptable, no-frills path (no banner, no download, no browser-open).
    ///
    /// Serves the same routes `muser up` does (`/`, `/snapshot`,
    /// `/telemetry`, `/metrics`, `/health`, `/healthz`, `/v1/models`,
    /// `POST /v1/chat/completions`). When `--model` is present the model is
    /// loaded and the endpoint performs real bounded inference.
    Serve(ServeArgs),

    /// Bring a disaggregated prefill node up: preflight, deploy, model,
    /// enroll, daemon, then smoke (which also emits netqual). This is the
    /// engine the dashboard's "Add node" button drives — see `node/mod.rs`.
    Node(NodeArgs),

    /// Create the private local CA or issue a server certificate with
    /// explicit DNS/IP subject alternative names.
    Tls(TlsArgs),
}

#[derive(Debug, Parser)]
pub struct TlsArgs {
    #[command(subcommand)]
    pub command: TlsCommand,
}

#[derive(Debug, Subcommand)]
pub enum TlsCommand {
    /// Create a separate local Muser CA. Existing CA material is never overwritten.
    Init(TlsInitArgs),
    /// Issue a server certificate from the local CA. At least one explicit SAN is required.
    Issue(TlsIssueArgs),
}

#[derive(Debug, Parser)]
pub struct TlsInitArgs {
    /// PKI directory. Defaults to $MUSER_HOME/pki, otherwise ~/.muser/pki.
    #[arg(long, value_name = "DIR")]
    pub dir: Option<PathBuf>,
}

#[derive(Debug, Parser)]
pub struct TlsIssueArgs {
    /// PKI directory holding ca.pem and ca-key.pem.
    #[arg(long, value_name = "DIR")]
    pub dir: Option<PathBuf>,
    /// Filename-safe certificate name.
    #[arg(long, value_name = "NAME")]
    pub name: String,
    /// Explicit DNS name or IP address. Repeat for every required SAN.
    #[arg(long, value_name = "DNS_OR_IP", required = true, action = clap::ArgAction::Append)]
    pub san: Vec<String>,
    /// Output directory. Defaults to <PKI_DIR>/issued/<NAME>.
    #[arg(long, value_name = "DIR")]
    pub out_dir: Option<PathBuf>,
}

#[derive(Debug, Parser)]
pub struct NodeArgs {
    #[command(subcommand)]
    pub command: NodeCommand,
}

#[derive(Debug, Subcommand)]
pub enum NodeCommand {
    /// Run the whole pipeline against a new or existing node. Exits 0 only
    /// if the smoke step passed.
    Add(NodeAddArgs),

    /// Probe the node's architecture, GPU, container runtime, disk and memory.
    Preflight(NodeStepArgs),

    /// Verify (or build) the pinned producer image and push the lane runtime.
    Deploy(NodeStepArgs),

    /// Install and verify the pinned target + DFlash GGUFs on the node.
    Model(NodeStepArgs),

    /// Mint the lab PKI, HMAC key and both handoff configs, then install the
    /// node-side half.
    Enroll(NodeStepArgs),

    /// Install and start the resident producer daemon, then wait for its port.
    Daemon(NodeStepArgs),

    /// Run a real remote prefill through the production receiver and record
    /// the measured link quality.
    Smoke(NodeStepArgs),

    /// Registry contents plus a live daemon probe per node.
    Status(NodeStatusArgs),
}

/// Options every pipeline step accepts. `--json` switches the progress
/// protocol (`muser.node-progress.v2`) from human lines to one JSON object
/// per line, which is what the server relays as SSE.
#[derive(Debug, Clone, Parser)]
pub struct NodeCommonArgs {
    /// Print exactly what each step would do and touch nothing — no SSH, no
    /// local writes, no child processes.
    #[arg(long)]
    pub dry_run: bool,

    /// Emit the progress protocol as JSON lines on stdout.
    #[arg(long)]
    pub json: bool,

    /// Pinned producer container receipt. Defaults to the newest
    /// `muser-gx10-prefill` receipt under the release-receipts directory.
    #[arg(long, value_name = "PATH")]
    pub container_receipt: Option<PathBuf>,

    /// Local directory holding the pinned GGUFs. Defaults to the directory
    /// of `muser up`'s default weights path.
    #[arg(long, value_name = "PATH")]
    pub model_dir: Option<PathBuf>,

    /// Override for the pinned llama.cpp Metal library. By default muser
    /// downloads and verifies the 7 MB release artifact automatically. May
    /// also be supplied as `MUSER_GGML_METALLIB`.
    #[arg(long, value_name = "PATH")]
    pub ggml_metallib: Option<PathBuf>,

    /// Source receipt for `--ggml-metallib`. Defaults to
    /// `source-receipt.json` beside the library.
    #[arg(long, value_name = "PATH")]
    pub ggml_metallib_receipt: Option<PathBuf>,

    /// Optional artifact mirror. The native lane otherwise downloads its Mac
    /// consumer chunks plus immutable-revision Hugging Face checkpoint and
    /// verifies every digest; the llama.cpp lane uses the scp fallback.
    #[arg(long, value_name = "URL")]
    pub model_source_base: Option<String>,

    /// Prompt token fixture for the smoke prefill. Defaults to a generated
    /// 2048-position fixture under the node's local directory.
    #[arg(long, value_name = "PATH")]
    pub prompt_fixture: Option<PathBuf>,

    /// Absolute lane directory on the node. Defaults to
    /// `/home/<user>/.muser/lane/<name>`, corrected from the remote `$HOME`
    /// by preflight.
    #[arg(long, value_name = "PATH")]
    pub lane_dir: Option<String>,
}

#[derive(Debug, Parser)]
pub struct NodeAddArgs {
    /// The node, as `user@host`.
    #[arg(value_name = "USER@HOST")]
    pub target: String,

    /// Registry name. Defaults to the host's first label.
    #[arg(long, value_name = "NAME")]
    pub name: Option<String>,

    /// Optional SSH identity file. Agent/keychain auth is used otherwise;
    /// passwords are never attempted (BatchMode=yes).
    #[arg(long, value_name = "PATH")]
    pub key: Option<PathBuf>,

    /// Producer lane to enroll. Fresh nodes default to the shipped NVFP4
    /// vLLM `native` lane; `llamacpp` selects the kquant+DFlash research
    /// lane. Unset leaves an existing node's lane unchanged.
    #[arg(long, value_enum, value_name = "LANE")]
    pub producer: Option<ProducerKind>,

    #[command(flatten)]
    pub common: NodeCommonArgs,
}

#[derive(Debug, Parser)]
pub struct NodeStepArgs {
    /// Registry name of an existing node.
    #[arg(value_name = "NAME")]
    pub name: String,

    #[command(flatten)]
    pub common: NodeCommonArgs,
}

#[derive(Debug, Parser)]
pub struct NodeStatusArgs {
    /// Emit the registry plus each live probe as a JSON array.
    #[arg(long)]
    pub json: bool,
}

#[derive(Debug, Parser)]
pub struct ServeArgs {
    /// Host/interface to bind on.
    #[arg(
        long,
        value_name = "HOST",
        env = "MUSER_HOST",
        default_value = "127.0.0.1"
    )]
    pub host: String,

    /// Port to bind on. Loopback is the default; LAN requires native TLS and auth.
    #[arg(long, value_name = "PORT", env = "MUSER_PORT", default_value_t = 4949)]
    pub port: u16,

    /// PEM certificate chain for native TLS. Nonloopback binds require this,
    /// --tls-key, and --api-key-file before the socket is opened.
    #[arg(long, value_name = "PEM", requires = "tls_key")]
    pub tls_cert: Option<PathBuf>,

    /// Mode-0600 PEM private key for native TLS.
    #[arg(long, value_name = "PEM", requires = "tls_cert")]
    pub tls_key: Option<PathBuf>,

    /// Mode-0600 file containing one API key. Required for every LAN bind
    /// and for loopback management routes.
    #[arg(long, value_name = "PATH")]
    pub api_key_file: Option<PathBuf>,

    /// Path to the Muse Glimmer GGUF.
    #[arg(long, value_name = "PATH")]
    pub model: Option<String>,

    /// Inference backend. Auto selects Metal on a Metal-enabled macOS build
    /// and CPU elsewhere; it never silently falls back after load failure.
    #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
    pub backend: BackendArg,

    /// Per-session context limit, up to the model's 131072-token limit.
    #[arg(long, default_value_t = 131_072)]
    pub max_context: usize,

    /// Resident decode slots. Each slot retains the full configured context.
    #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u8).range(1..=4))]
    pub parallel: u8,

    /// Context overflow behavior. Shift rebuilds complete newest turns in a
    /// staging generation; error rejects the request.
    #[arg(long, value_enum, default_value_t = ContextPolicyArg::Shift)]
    pub context_policy: ContextPolicyArg,

    /// Raw-mode prefix tokens retained when context policy is shift.
    #[arg(long, default_value_t = 256)]
    pub raw_retain_prefix: usize,

    /// Byte budget for the compressed resident exact-prefix radix.
    #[arg(long, default_value_t = 8 * 1024 * 1024 * 1024u64)]
    pub resident_cache_bytes: u64,

    /// Optional authenticated kvpack LocalStore configuration.
    #[arg(long, value_name = "PATH")]
    pub kvpack_config: Option<PathBuf>,

    /// Exact prefix reuse policy. Baseline TTFT qualification pins this off.
    #[arg(long, value_enum, default_value_t = PrefixCacheArg::On)]
    pub prefix_cache: PrefixCacheArg,

    /// Optional official vision projector artifact.
    #[arg(long, value_name = "PATH")]
    pub mmproj: Option<PathBuf>,

    /// Packaged in-process mtmd Metal bridge. Required with --mmproj when
    /// the selected target backend is Metal; ignored by the CPU oracle.
    #[arg(long, value_name = "DYLIB", env = "MUSER_MTMD_BRIDGE")]
    pub mtmd_bridge: Option<PathBuf>,

    /// Optional five-layer DFlash assistant artifact.
    #[arg(long, value_name = "PATH")]
    pub dflash: Option<PathBuf>,

    #[arg(long, value_enum, default_value_t = DflashBackendArg::Auto)]
    pub dflash_backend: DflashBackendArg,

    /// Explicit exported Core ML manifest for the experimental post-release
    /// ANE DFlash route. It is never selected by v0.1 auto routing.
    #[arg(long, value_name = "PATH")]
    pub ane_manifest: Option<PathBuf>,

    /// Prefill route. Auto/remote consumes CUDA-produced KV and therefore
    /// requires MUSER_CROSS_VENDOR_QK=1 for the pinned exact Metal math route.
    #[arg(long, value_enum, default_value_t = PrefillArg::Local)]
    pub prefill: PrefillArg,

    /// Enrolled receiver configuration. Requires --prefill auto or remote;
    /// remote startup fails closed unless MUSER_CROSS_VENDOR_QK=1 is set.
    #[arg(long, value_name = "PATH")]
    pub cluster_config: Option<PathBuf>,

    /// Private qualification-only token for cooperative server shutdown.
    /// Requires --benchmark-deadline-seconds and a loopback bind.
    #[arg(long, hide = true, value_name = "TOKEN")]
    pub benchmark_shutdown_token: Option<String>,

    /// Qualification-only self-deadline. The server exits its accept loop
    /// itself; the campaign coordinator never signals or kills it.
    #[arg(long, hide = true, value_name = "SECONDS")]
    pub benchmark_deadline_seconds: Option<u64>,
}

#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum BackendArg {
    #[default]
    Auto,
    Cpu,
    Metal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ContextPolicyArg {
    Shift,
    Error,
}

impl From<ContextPolicyArg> for crate::state::ContextPolicy {
    fn from(value: ContextPolicyArg) -> Self {
        match value {
            ContextPolicyArg::Shift => Self::Shift,
            ContextPolicyArg::Error => Self::Error,
        }
    }
}

impl From<BackendArg> for BackendMode {
    fn from(value: BackendArg) -> Self {
        match value {
            BackendArg::Auto => Self::Auto,
            BackendArg::Cpu => Self::Cpu,
            BackendArg::Metal => Self::Metal,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum DflashBackendArg {
    #[default]
    Auto,
    Metal,
    /// Experimental post-release public-CoreML path; excluded from v0.1.
    Ane,
}

#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum PrefillArg {
    #[default]
    Local,
    Auto,
    Remote,
}

#[derive(Debug, Clone, Copy, ValueEnum, Default, PartialEq, Eq)]
pub enum PrefixCacheArg {
    #[default]
    On,
    Off,
}

#[derive(Debug, Parser)]
pub struct UpArgs {
    /// Path to the local Muse Glimmer GGUF.
    ///
    /// This changes location only. Model revision, size, and SHA-256 remain
    /// pinned by docs/release-artifacts.json.
    #[arg(long, value_name = "PATH")]
    pub gguf: Option<PathBuf>,

    /// Hugging Face repository identifier to resolve through the embedded
    /// release registry. Only the repository pinned by
    /// docs/release-artifacts.json is admitted.
    #[arg(long, value_name = "REPO_ID", env = "MUSER_HF_REPO")]
    pub hf_repo: Option<String>,

    /// Optional TOML config file. CLI flags and env vars both win over it;
    /// it wins over built-in defaults. Defaults to `./muser.toml` if present.
    #[arg(long, value_name = "PATH")]
    pub config: Option<PathBuf>,

    /// Host/interface to bind the server + dashboard on.
    #[arg(long, value_name = "HOST", env = "MUSER_HOST")]
    pub host: Option<String>,

    /// Port to bind the server + dashboard on.
    #[arg(long, value_name = "PORT", env = "MUSER_PORT")]
    pub port: Option<u16>,

    /// PEM certificate chain for native TLS. Required with --tls-key for a
    /// nonloopback bind and for Secure dashboard-session cookies.
    #[arg(long, value_name = "PEM", requires = "tls_key")]
    pub tls_cert: Option<PathBuf>,

    /// Mode-0600 PEM private key for native TLS.
    #[arg(long, value_name = "PEM", requires = "tls_cert")]
    pub tls_key: Option<PathBuf>,

    /// Mode-0600 API-key file. Required for nonloopback serving and for
    /// dashboard node management; loopback inference remains keyless.
    #[arg(long, value_name = "PATH")]
    pub api_key_file: Option<PathBuf>,

    /// Don't try to open the dashboard in a browser automatically.
    #[arg(long)]
    pub no_open: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// clap's own derive invariants (no duplicate flags, valid subcommands,
    /// etc.) — cheap insurance that the CLI definition stays well-formed.
    #[test]
    fn cli_definition_is_valid() {
        <Cli as clap::CommandFactory>::command().debug_assert();
    }

    #[test]
    fn up_parses_flags() {
        let cli = Cli::try_parse_from([
            "muser",
            "up",
            "--gguf",
            "/tmp/m.gguf",
            "--hf-repo",
            "meta-models/Muse-Glimmer-30B-GGUF",
            "--host",
            "0.0.0.0",
            "--port",
            "9999",
            "--no-open",
        ])
        .expect("`muser up` with flags must parse");
        match cli.command {
            Command::Up(a) => {
                assert_eq!(a.gguf.as_deref(), Some(std::path::Path::new("/tmp/m.gguf")));
                assert_eq!(
                    a.hf_repo.as_deref(),
                    Some("meta-models/Muse-Glimmer-30B-GGUF")
                );
                assert_eq!(a.host.as_deref(), Some("0.0.0.0"));
                assert_eq!(a.port, Some(9999));
                assert!(a.tls_cert.is_none());
                assert!(a.tls_key.is_none());
                assert!(a.api_key_file.is_none());
                assert!(a.no_open);
            }
            other => panic!("expected Up, got {other:?}"),
        }
    }

    #[test]
    fn serve_defaults_to_the_dashboard_live_fallback_port() {
        let cli = Cli::try_parse_from(["muser", "serve"]).expect("bare `muser serve` must parse");
        match cli.command {
            Command::Serve(a) => {
                assert_eq!(a.host, "127.0.0.1");
                assert_eq!(a.port, 4949);
                assert!(a.model.is_none());
            }
            other => panic!("expected Serve, got {other:?}"),
        }
    }

    #[test]
    fn up_accepts_the_same_native_tls_and_auth_files_as_serve() {
        let cli = Cli::try_parse_from([
            "muser",
            "up",
            "--tls-cert",
            "/pki/server.pem",
            "--tls-key",
            "/pki/server-key.pem",
            "--api-key-file",
            "/pki/api-key",
        ])
        .expect("secure `muser up` flags must parse");
        match cli.command {
            Command::Up(args) => {
                assert_eq!(
                    args.tls_cert.as_deref(),
                    Some(std::path::Path::new("/pki/server.pem"))
                );
                assert_eq!(
                    args.tls_key.as_deref(),
                    Some(std::path::Path::new("/pki/server-key.pem"))
                );
                assert_eq!(
                    args.api_key_file.as_deref(),
                    Some(std::path::Path::new("/pki/api-key"))
                );
            }
            other => panic!("expected Up, got {other:?}"),
        }
    }

    #[test]
    fn serve_parses_flat_flags() {
        let cli = Cli::try_parse_from([
            "muser",
            "serve",
            "--host",
            "0.0.0.0",
            "--port",
            "8080",
            "--model",
            "/tmp/m.gguf",
        ])
        .expect("`muser serve` with flat flags must parse");
        match cli.command {
            Command::Serve(a) => {
                assert_eq!(a.host, "0.0.0.0");
                assert_eq!(a.port, 8080);
                assert_eq!(a.model.as_deref(), Some("/tmp/m.gguf"));
                assert!(a.ane_manifest.is_none());
                assert_eq!(a.context_policy, ContextPolicyArg::Shift);
                assert_eq!(a.raw_retain_prefix, 256);
            }
            other => panic!("expected Serve, got {other:?}"),
        }
    }

    #[test]
    fn serve_parses_private_cooperative_benchmark_lifecycle() {
        let token = "0123456789abcdef0123456789abcdef";
        let cli = Cli::try_parse_from([
            "muser",
            "serve",
            "--benchmark-shutdown-token",
            token,
            "--benchmark-deadline-seconds",
            "1800",
        ])
        .expect("qualification lifecycle flags must parse");
        match cli.command {
            Command::Serve(args) => {
                assert_eq!(args.benchmark_shutdown_token.as_deref(), Some(token));
                assert_eq!(args.benchmark_deadline_seconds, Some(1800));
            }
            other => panic!("expected Serve, got {other:?}"),
        }
    }

    #[test]
    fn a_subcommand_is_required() {
        assert!(Cli::try_parse_from(["muser"]).is_err());
    }

    #[test]
    fn node_add_parses_the_one_button_surface() {
        let cli = Cli::try_parse_from([
            "muser",
            "node",
            "add",
            "muser@gx10.local",
            "--name",
            "gx10",
            "--key",
            "/k/id_ed25519",
            "--dry-run",
            "--json",
        ])
        .expect("`muser node add` must parse");
        match cli.command {
            Command::Node(NodeArgs {
                command: NodeCommand::Add(args),
            }) => {
                assert_eq!(args.target, "muser@gx10.local");
                assert_eq!(args.name.as_deref(), Some("gx10"));
                assert_eq!(
                    args.key.as_deref(),
                    Some(std::path::Path::new("/k/id_ed25519"))
                );
                assert!(args.common.dry_run);
                assert!(args.common.json);
            }
            other => panic!("expected Node/Add, got {other:?}"),
        }
    }

    #[test]
    fn node_add_parses_the_producer_lane_flag() {
        let parsed = |args: &[&str]| -> Option<ProducerKind> {
            let cli = Cli::try_parse_from(args).expect("`muser node add` must parse");
            match cli.command {
                Command::Node(NodeArgs {
                    command: NodeCommand::Add(args),
                }) => args.producer,
                other => panic!("expected Node/Add, got {other:?}"),
            }
        };
        // A fresh node resolves no flag to native when its registry entry is
        // created; clap deliberately leaves the option absent here so an
        // existing node keeps its enrolled lane.
        assert_eq!(parsed(&["muser", "node", "add", "muser@gx10.local"]), None);
        assert_eq!(
            parsed(&[
                "muser",
                "node",
                "add",
                "muser@gx10.local",
                "--producer",
                "llamacpp"
            ]),
            Some(ProducerKind::Llamacpp)
        );
        assert_eq!(
            parsed(&[
                "muser",
                "node",
                "add",
                "muser@gx10.local",
                "--producer",
                "native"
            ]),
            Some(ProducerKind::Native)
        );
        // A lane this build does not know is a hard refusal, not a default.
        assert!(Cli::try_parse_from([
            "muser",
            "node",
            "add",
            "muser@gx10.local",
            "--producer",
            "tpu"
        ])
        .is_err());
    }

    #[test]
    fn every_pipeline_step_is_individually_runnable() {
        for step in ["preflight", "deploy", "model", "enroll", "daemon", "smoke"] {
            let cli = Cli::try_parse_from(["muser", "node", step, "gx10"])
                .unwrap_or_else(|error| panic!("`muser node {step} <name>` must parse: {error}"));
            let Command::Node(NodeArgs { command }) = cli.command else {
                panic!("expected a node subcommand");
            };
            let name = match command {
                NodeCommand::Preflight(args)
                | NodeCommand::Deploy(args)
                | NodeCommand::Model(args)
                | NodeCommand::Enroll(args)
                | NodeCommand::Daemon(args)
                | NodeCommand::Smoke(args) => args.name,
                other => panic!("expected a single-step subcommand, got {other:?}"),
            };
            assert_eq!(name, "gx10");
        }
    }

    #[test]
    fn node_status_takes_only_its_json_flag() {
        let cli = Cli::try_parse_from(["muser", "node", "status", "--json"])
            .expect("`muser node status --json` must parse");
        match cli.command {
            Command::Node(NodeArgs {
                command: NodeCommand::Status(args),
            }) => assert!(args.json),
            other => panic!("expected Node/Status, got {other:?}"),
        }
        assert!(Cli::try_parse_from(["muser", "node", "status", "gx10"]).is_err());
    }
}
