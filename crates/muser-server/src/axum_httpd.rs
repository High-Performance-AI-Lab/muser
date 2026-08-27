//! Tokio/Axum/Hyper HTTP substrate. Hyper owns request framing; application
//! code only sees validated requests and bounded bodies.

use std::cell::Cell;
use std::collections::HashMap;
use std::convert::Infallible;
use std::future::Future;
use std::io;
use std::net::{SocketAddr, ToSocketAddrs};
use std::path::{Path as FsPath, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use axum::body::{Body, Bytes};
use axum::extract::ws::{Message as WebSocketMessage, WebSocket, WebSocketUpgrade};
use axum::extract::{ConnectInfo, DefaultBodyLimit, OriginalUri, Path, Query, Request, State};
use axum::http::header::{AUTHORIZATION, CACHE_CONTROL, CONTENT_TYPE, HOST, ORIGIN};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Extension, Json, Router};
use base64::Engine as _;
use sha2::{Digest as _, Sha256};
use tokio::io::AsyncWriteExt as _;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt as _;
use tower::limit::ConcurrencyLimitLayer;
use tower::Service;
use tower_http::timeout::RequestBodyTimeoutLayer;

use crate::metrics::{self, Envelope};
use crate::resumable_stream::{ReadSnapshot, StreamManager};
use crate::state::ServerState;
use crate::{nodes_api, openai};

/// Retained only for the legacy node-progress writer while that producer is
/// migrated to the async frame callback. It is not used to parse requests.
pub(crate) const SSE_HEADERS: &str = concat!(
    "HTTP/1.1 200 OK\r\n",
    "Content-Type: text/event-stream\r\n",
    "Cache-Control: no-cache\r\n",
    "Connection: keep-alive\r\n",
    "X-Accel-Buffering: no\r\n",
    "\r\n",
);

const DASHBOARD_HTML: &str = include_str!("../../../web/muser-dashboard.html");
const MAX_BODY: usize = 64 * 1024 * 1024;
const STREAM_CHANNEL_DEPTH: usize = 64;
const SLOW_CLIENT_GRACE: Duration = Duration::from_secs(5);

#[derive(Debug, thiserror::Error)]
#[error("could not bind {host}:{port}: {source}")]
pub struct ServeError {
    host: String,
    port: u16,
    #[source]
    source: io::Error,
}

#[derive(Clone)]
struct AppState {
    server: Arc<ServerState>,
    benchmark: Option<Arc<BenchmarkControl>>,
    api_key: Option<Arc<[u8]>>,
    lan: bool,
    tls: bool,
    dashboard_sessions: Arc<std::sync::Mutex<HashMap<String, DashboardSession>>>,
    websocket_tickets: Arc<std::sync::Mutex<HashMap<String, Instant>>>,
    streams: StreamManager,
    reasoning_controls: Arc<std::sync::Mutex<HashMap<String, Arc<AtomicBool>>>>,
}

#[derive(Clone)]
struct DashboardSession {
    csrf: String,
    origin: String,
    expires: Instant,
}

struct ReasoningControlRegistration {
    id: String,
    signal: Arc<AtomicBool>,
    controls: Arc<std::sync::Mutex<HashMap<String, Arc<AtomicBool>>>>,
}

impl Drop for ReasoningControlRegistration {
    fn drop(&mut self) {
        let mut controls = self
            .controls
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if controls
            .get(&self.id)
            .is_some_and(|current| Arc::ptr_eq(current, &self.signal))
        {
            controls.remove(&self.id);
        }
    }
}

struct DisconnectCancellation {
    signal: Arc<AtomicBool>,
    armed: bool,
}

#[derive(Clone)]
struct ConnectionCancellation(Arc<AtomicBool>);

type ConnectionRequests = Arc<std::sync::Mutex<Vec<Weak<AtomicBool>>>>;

struct ConnectionLifetime(ConnectionRequests);

impl Drop for ConnectionLifetime {
    fn drop(&mut self) {
        let requests = self.0.lock().unwrap_or_else(|error| error.into_inner());
        for request in requests.iter().filter_map(Weak::upgrade) {
            request.store(true, Ordering::Release);
        }
    }
}

#[derive(Clone)]
struct ConnectionService<S> {
    inner: S,
    requests: ConnectionRequests,
    _lifetime: Arc<ConnectionLifetime>,
}

impl<S, B> Service<axum::http::Request<B>> for ConnectionService<S>
where
    S: Service<axum::http::Request<B>>,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = S::Future;

    fn poll_ready(&mut self, context: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(context)
    }

    fn call(&mut self, mut request: axum::http::Request<B>) -> Self::Future {
        // Cancellation belongs to a request, not its HTTP/1 keep-alive or
        // HTTP/2 connection. Dropping one handler must never cancel another
        // request multiplexed on the same connection. The connection keeps
        // weak registrations only so a real disconnect can still signal all
        // in-flight workers immediately.
        let cancellation = ConnectionCancellation(Arc::new(AtomicBool::new(false)));
        {
            let mut requests = self
                .requests
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            requests.retain(|signal| signal.strong_count() > 0);
            requests.push(Arc::downgrade(&cancellation.0));
        }
        request.extensions_mut().insert(cancellation);
        self.inner.call(request)
    }
}

#[derive(Clone)]
struct ConnectionMakeService<M> {
    inner: M,
}

impl<M> ConnectionMakeService<M> {
    fn new(inner: M) -> Self {
        Self { inner }
    }
}

impl<M, T> Service<T> for ConnectionMakeService<M>
where
    M: Service<T>,
    M::Future: Send + 'static,
    M::Response: Send + 'static,
    M::Error: 'static,
{
    type Response = ConnectionService<M::Response>;
    type Error = M::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, context: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(context)
    }

    fn call(&mut self, target: T) -> Self::Future {
        let future = self.inner.call(target);
        Box::pin(async move {
            let inner = future.await?;
            let requests = Arc::new(std::sync::Mutex::new(Vec::new()));
            Ok(ConnectionService {
                inner,
                requests: Arc::clone(&requests),
                _lifetime: Arc::new(ConnectionLifetime(requests)),
            })
        })
    }
}

impl DisconnectCancellation {
    fn new(signal: Arc<AtomicBool>) -> Self {
        Self {
            signal,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for DisconnectCancellation {
    fn drop(&mut self) {
        if self.armed {
            self.signal.store(true, Ordering::Release);
        }
    }
}

struct BenchmarkControl {
    shutdown_token: Vec<u8>,
    stop: AtomicBool,
}

pub fn validate_bind_host(host: &str, port: u16) -> Result<(), String> {
    validate_bind_security(host, port, &SecurityConfig::default())
}

#[derive(Clone, Debug, Default)]
pub struct SecurityConfig {
    pub tls_cert: Option<PathBuf>,
    pub tls_key: Option<PathBuf>,
    pub api_key_file: Option<PathBuf>,
}

pub fn validate_bind_security(
    host: &str,
    port: u16,
    security: &SecurityConfig,
) -> Result<(), String> {
    let addresses = (host, port)
        .to_socket_addrs()
        .map_err(|error| format!("cannot resolve bind host {host:?}: {error}"))?
        .collect::<Vec<_>>();
    if addresses.is_empty() {
        return Err(format!("bind host {host:?} resolved to no addresses"));
    }
    let lan = addresses.iter().any(|address| !address.ip().is_loopback());
    let tls_pair = security.tls_cert.is_some() && security.tls_key.is_some();
    if security.tls_cert.is_some() != security.tls_key.is_some() {
        return Err("--tls-cert and --tls-key must be supplied together".into());
    }
    if lan && (!tls_pair || security.api_key_file.is_none()) {
        return Err(
            "nonloopback serving requires --tls-cert, --tls-key, and --api-key-file".into(),
        );
    }
    if let Some(key) = security.tls_key.as_deref() {
        require_private_file(key, "TLS private key")?;
    }
    if let Some(key) = security.api_key_file.as_deref() {
        require_private_file(key, "API-key file")?;
        let value = std::fs::read(key)
            .map_err(|error| format!("read API-key file {}: {error}", key.display()))?;
        if value.iter().all(u8::is_ascii_whitespace) || value.len() > 4096 {
            return Err("API-key file must contain 1..=4096 non-whitespace bytes".into());
        }
    }
    Ok(())
}

#[cfg(unix)]
fn require_private_file(path: &FsPath, label: &str) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt as _;
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|error| format!("inspect {label} {}: {error}", path.display()))?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Err(format!("{label} {} must be a regular file", path.display()));
    }
    if metadata.permissions().mode() & 0o077 != 0 {
        return Err(format!(
            "{label} {} must have mode 0600 or stricter",
            path.display()
        ));
    }
    Ok(())
}

#[cfg(not(unix))]
fn require_private_file(path: &FsPath, label: &str) -> Result<(), String> {
    if !path.is_file() {
        return Err(format!("{label} {} must be a regular file", path.display()));
    }
    Ok(())
}

pub fn serve(host: &str, port: u16, state: Arc<ServerState>) -> Result<(), ServeError> {
    serve_inner(host, port, state, None, SecurityConfig::default())
}

pub fn serve_secure(
    host: &str,
    port: u16,
    state: Arc<ServerState>,
    security: SecurityConfig,
) -> Result<(), ServeError> {
    serve_inner(host, port, state, None, security)
}

pub fn serve_for_benchmark(
    host: &str,
    port: u16,
    state: Arc<ServerState>,
    shutdown_token: &str,
    deadline_seconds: u64,
    security: SecurityConfig,
) -> Result<(), ServeError> {
    let control = Arc::new(BenchmarkControl {
        shutdown_token: shutdown_token.as_bytes().to_vec(),
        stop: AtomicBool::new(false),
    });
    let deadline = Arc::clone(&control);
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_secs(deadline_seconds));
        deadline.stop.store(true, Ordering::Release);
    });
    serve_inner(host, port, state, Some(control), security)
}

fn serve_inner(
    host: &str,
    port: u16,
    server: Arc<ServerState>,
    benchmark: Option<Arc<BenchmarkControl>>,
    security: SecurityConfig,
) -> Result<(), ServeError> {
    // The workspace contains clients selecting ring and axum-server selecting
    // aws-lc, so rustls cannot infer a global provider from Cargo features.
    // Choose the already-qualified ring provider explicitly before parsing
    // any key material.
    let _ = rustls::crypto::ring::default_provider().install_default();
    validate_bind_security(host, port, &security).map_err(|message| ServeError {
        host: host.into(),
        port,
        source: io::Error::new(io::ErrorKind::PermissionDenied, message),
    })?;
    let address = (host, port)
        .to_socket_addrs()
        .map_err(|source| ServeError {
            host: host.into(),
            port,
            source,
        })?
        .next()
        .ok_or_else(|| ServeError {
            host: host.into(),
            port,
            source: io::Error::new(io::ErrorKind::AddrNotAvailable, "bind host resolved empty"),
        })?;
    let lan = !address.ip().is_loopback();
    let api_key = security
        .api_key_file
        .as_deref()
        .map(read_api_key)
        .transpose()
        .map_err(|source| ServeError {
            host: host.into(),
            port,
            source,
        })?
        .map(Arc::from);
    let tls_paths = security.tls_cert.zip(security.tls_key);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|source| ServeError {
            host: host.into(),
            port,
            source,
        })?;
    runtime.block_on(async move {
        let tls = tls_paths.is_some();
        let app_state = AppState {
            server,
            benchmark: benchmark.clone(),
            api_key,
            lan,
            tls,
            dashboard_sessions: Arc::new(std::sync::Mutex::new(HashMap::new())),
            websocket_tickets: Arc::new(std::sync::Mutex::new(HashMap::new())),
            streams: StreamManager::default(),
            reasoning_controls: Arc::new(std::sync::Mutex::new(HashMap::new())),
        };
        let app = router(app_state);
        if let Some((cert, key)) = tls_paths {
            let config = axum_server::tls_rustls::RustlsConfig::from_pem_file(cert, key)
                .await
                .map_err(|source| ServeError {
                    host: host.into(),
                    port,
                    source,
                })?;
            let handle = axum_server::Handle::new();
            let shutdown_handle = handle.clone();
            tokio::spawn(async move {
                shutdown_signal(benchmark).await;
                shutdown_handle.graceful_shutdown(Some(Duration::from_secs(5)));
            });
            println!("muser-server: listening on https://{address} (Axum/Hyper/rustls)");
            let mut serving = axum_server::bind_rustls(address, config).handle(handle);
            serving
                .http_builder()
                .http1()
                .timer(hyper_util::rt::TokioTimer::new())
                .header_read_timeout(Duration::from_secs(10))
                .max_buf_size(16 * 1024);
            serving
                .serve(ConnectionMakeService::new(
                    app.into_make_service_with_connect_info::<SocketAddr>(),
                ))
                .await
                .map_err(|source| ServeError {
                    host: host.into(),
                    port,
                    source,
                })
        } else {
            let handle = axum_server::Handle::new();
            let shutdown_handle = handle.clone();
            tokio::spawn(async move {
                shutdown_signal(benchmark).await;
                shutdown_handle.graceful_shutdown(Some(Duration::from_secs(5)));
            });
            println!("muser-server: listening on http://{address} (Axum/Hyper)");
            let mut serving = axum_server::bind(address).handle(handle);
            serving
                .http_builder()
                .http1()
                .timer(hyper_util::rt::TokioTimer::new())
                .header_read_timeout(Duration::from_secs(10))
                .max_buf_size(16 * 1024);
            serving
                .serve(ConnectionMakeService::new(
                    app.into_make_service_with_connect_info::<SocketAddr>(),
                ))
                .await
                .map_err(|source| ServeError {
                    host: host.into(),
                    port,
                    source,
                })
        }
    })
}

fn read_api_key(path: &FsPath) -> io::Result<Vec<u8>> {
    let value = std::fs::read(path)?;
    let start = value
        .iter()
        .position(|byte| !byte.is_ascii_whitespace())
        .unwrap_or(value.len());
    let end = value
        .iter()
        .rposition(|byte| !byte.is_ascii_whitespace())
        .map_or(start, |index| index + 1);
    Ok(value[start..end].to_vec())
}

async fn shutdown_signal(benchmark: Option<Arc<BenchmarkControl>>) {
    if let Some(control) = benchmark {
        while !control.stop.load(Ordering::Acquire) {
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
        return;
    }
    let _ = tokio::signal::ctrl_c().await;
}

fn router(state: AppState) -> Router {
    let application = Router::new()
        .route("/", get(dashboard))
        .route("/dashboard", get(dashboard))
        .route("/snapshot", get(snapshot))
        .route("/metrics", get(prometheus_metrics))
        .route("/telemetry", get(telemetry))
        .route("/health", get(health))
        .route("/v1/health", get(health))
        .route("/healthz", get(healthz))
        .route("/models", get(models))
        .route("/v1/models", get(models))
        .route("/props", get(props).post(props_update))
        .route("/slots", get(slots_get).post(slots_action))
        .route("/slots/{id}", post(slots_compat_action))
        .route("/tokenize", post(tokenize))
        .route("/detokenize", post(detokenize))
        .route("/apply-template", post(apply_template))
        .route("/embedding", post(embeddings))
        .route("/embeddings", post(embeddings))
        .route("/v1/embeddings", post(embeddings))
        .route("/completion", post(completion))
        .route("/completions", post(completion))
        .route("/v1/completions", post(completion))
        .route("/api/generate", post(ollama_generate))
        .route("/generate", post(ollama_generate))
        .route("/v1/chat/completions", post(chat))
        .route("/chat/completions", post(chat))
        .route(
            "/v1/stream",
            get(resumable_stream_get).delete(resumable_stream_delete),
        )
        .route("/v1/streams/lookup", post(resumable_stream_lookup))
        .route("/v1/chat/completions/control", post(reasoning_control))
        .route("/v1/dashboard/login", post(dashboard_login))
        .route("/v1/ws-tickets", post(websocket_ticket))
        .route("/v1/sessions", get(sessions_list).post(sessions_create))
        .route("/v1/sessions/{id}", get(session_get).delete(session_delete))
        .route("/v1/sessions/{id}/save", post(session_save))
        .route("/v1/sessions/{id}/restore", post(session_restore))
        .route("/v1/sessions/{id}/migrate", post(session_migrate))
        .route(
            "/v1/session-transfers/{transfer_id}",
            get(session_transfer_get),
        )
        .route(
            "/__muser/v1/session-transfers/prepare",
            post(session_transfer_prepare),
        )
        .route(
            "/__muser/v1/session-transfers/{transfer_id}/commit",
            post(session_transfer_commit),
        )
        .route("/stream", get(websocket_stream))
        .route("/v1/nodes", get(nodes_list).post(nodes_create))
        .route("/v1/nodes/{name}/progress", get(nodes_progress))
        .route("/__muser/benchmark/shutdown", post(benchmark_shutdown))
        .layer(DefaultBodyLimit::max(MAX_BODY))
        .layer(RequestBodyTimeoutLayer::new(Duration::from_secs(30)))
        .layer(ConcurrencyLimitLayer::new(256));
    let transfer_payload = Router::new()
        .route(
            "/__muser/v1/session-transfers/{transfer_id}/payload",
            put(session_transfer_payload),
        )
        .layer(DefaultBodyLimit::disable())
        .layer(RequestBodyTimeoutLayer::new(Duration::from_secs(60 * 60)))
        .layer(ConcurrencyLimitLayer::new(4));
    application
        .merge(transfer_payload)
        .layer(middleware::from_fn(normalize_request_authority))
        .with_state(state)
}

/// HTTP/2 carries the request authority in `:authority`, represented by
/// Hyper as the URI authority, and need not include an HTTP/1 `Host` header.
/// Normalize that protocol-level authority into `Host` before route auth so
/// dashboard origin binding behaves identically on HTTP/1.1 and HTTP/2.
/// A peer that supplies both forms with different values is ambiguous and is
/// rejected rather than allowing either value to win.
async fn normalize_request_authority(mut request: Request, next: Next) -> Response {
    if let Err(message) = normalize_authority(&mut request) {
        return error_json(StatusCode::BAD_REQUEST, "invalid_request_error", message);
    }
    next.run(request).await
}

fn normalize_authority(request: &mut Request) -> Result<(), &'static str> {
    let Some(authority) = request.uri().authority().map(|value| value.as_str()) else {
        return Ok(());
    };
    match single_header(request.headers(), HOST.as_str()) {
        Some(host) if host == authority => Ok(()),
        Some(_) => Err("request Host and HTTP/2 authority must match exactly"),
        None if request.headers().contains_key(HOST) => {
            Err("request Host must contain one valid authority")
        }
        None => {
            let value = HeaderValue::from_str(authority)
                .map_err(|_| "request authority must be a valid Host value")?;
            request.headers_mut().insert(HOST, value);
            Ok(())
        }
    }
}

async fn dashboard() -> impl IntoResponse {
    let mut response = Html(DASHBOARD_HTML).into_response();
    response
        .headers_mut()
        .insert(CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response
}

async fn snapshot(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if !valid_management_auth(&state, &headers, false) {
        return auth_required();
    }
    state
        .server
        .telemetry_requests
        .fetch_add(1, Ordering::Relaxed);
    Json(metrics::build_snapshot(&state.server)).into_response()
}

async fn prometheus_metrics(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if !valid_management_auth(&state, &headers, false) {
        return auth_required();
    }
    state
        .server
        .telemetry_requests
        .fetch_add(1, Ordering::Relaxed);
    let snapshot = metrics::build_snapshot(&state.server);
    let mut body = format!(concat!(
        "# TYPE completion_traffic_tok_s_10s gauge\ncompletion_traffic_tok_s_10s {}\n",
        "# TYPE muser_queue_depth gauge\nmuser_queue_depth {}\n",
        "# TYPE muser_overload_rejections_total counter\nmuser_overload_rejections_total {}\n",
        "# TYPE muser_completion_tokens_total counter\nmuser_completion_tokens_total {}\n",
        "# TYPE muser_ttft_milliseconds gauge\nmuser_ttft_milliseconds{{quantile=\"0.50\"}} {}\nmuser_ttft_milliseconds{{quantile=\"0.95\"}} {}\n",
        "# TYPE muser_itl_milliseconds gauge\nmuser_itl_milliseconds{{quantile=\"0.50\"}} {}\nmuser_itl_milliseconds{{quantile=\"0.95\"}} {}\n",
        "# TYPE muser_dflash_acceptance_ratio gauge\nmuser_dflash_acceptance_ratio {}\n"
    ), snapshot.decode.completion_traffic_tok_s_10s, snapshot.queue_depth,
       snapshot.overload_rejections, snapshot.decode.completion_tokens,
       snapshot.wire.ttft_ms.p50, snapshot.wire.ttft_ms.p95,
       snapshot.wire.itl_ms.p50, snapshot.wire.itl_ms.p95,
       snapshot.specdec.accept_rate);
    body.push_str("# TYPE muser_phase_seconds_total counter\n");
    body.push_str("# TYPE muser_phase_samples_total counter\n");
    for (name, phase) in [
        ("queue", &snapshot.phases.queue),
        ("prefill", &snapshot.phases.prefill),
        ("sampling", &snapshot.phases.sampling),
        ("grammar", &snapshot.phases.grammar),
        ("detokenization", &snapshot.phases.detokenization),
        ("enqueue_write", &snapshot.phases.enqueue_write),
        ("dflash_draft", &snapshot.phases.dflash_draft),
        (
            "dflash_target_verify",
            &snapshot.phases.dflash_target_verify,
        ),
    ] {
        body.push_str(&format!(
            "muser_phase_seconds_total{{phase=\"{name}\"}} {}\n\
             muser_phase_samples_total{{phase=\"{name}\"}} {}\n",
            phase.total_ms / 1_000.0,
            phase.samples,
        ));
    }
    body.push_str(&format!(
        "# TYPE muser_request_decode_tok_s gauge\n\
         muser_request_decode_tok_s {}\n",
        snapshot.phases.last_request_decode_tok_s,
    ));
    let (packed_batches, packed_rows, last_width) = state
        .server
        .inference
        .as_ref()
        .map(|runtime| runtime.decode_batcher.stats())
        .unwrap_or_default();
    body.push_str(&format!(
        "# TYPE muser_decode_packed_batches_total counter\n\
         muser_decode_packed_batches_total {packed_batches}\n\
         # TYPE muser_decode_packed_rows_total counter\n\
         muser_decode_packed_rows_total {packed_rows}\n\
         # TYPE muser_decode_batch_width_last gauge\n\
         muser_decode_batch_width_last {last_width}\n",
    ));
    let mut response = body.into_response();
    response.headers_mut().insert(
        CONTENT_TYPE,
        HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
    );
    response
}

async fn telemetry(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if !valid_management_auth(&state, &headers, false) {
        return auth_required();
    }
    state
        .server
        .telemetry_requests
        .fetch_add(1, Ordering::Relaxed);
    state
        .server
        .telemetry_viewers
        .fetch_add(1, Ordering::Relaxed);
    let (sender, receiver) = mpsc::channel::<Result<Bytes, Infallible>>(4);
    tokio::spawn(async move {
        struct Viewer(Arc<ServerState>);
        impl Drop for Viewer {
            fn drop(&mut self) {
                self.0.telemetry_viewers.fetch_sub(1, Ordering::Relaxed);
            }
        }
        let _viewer = Viewer(Arc::clone(&state.server));
        let mut seq = 0u64;
        let mut tick = tokio::time::interval(Duration::from_secs(1));
        loop {
            tick.tick().await;
            let snap = metrics::build_snapshot(&state.server);
            let frame = Envelope {
                v: metrics::SCHEMA_VERSION,
                kind: "snapshot",
                seq,
                t: snap.uptime_s,
                data: snap,
            };
            let json = serde_json::to_string(&frame)
                .unwrap_or_else(|error| format!("{{\"error\":\"{error}\"}}"));
            if sender
                .send(Ok(Bytes::from(format!(
                    "event: snapshot\ndata: {json}\n\n"
                ))))
                .await
                .is_err()
            {
                break;
            }
            seq += 1;
        }
    });
    sse_response(Body::from_stream(ReceiverStream::new(receiver)))
}

async fn health(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if state.lan && !valid_bearer(&state, &headers) {
        return auth_required();
    }
    state.server.record_request();
    let healthy = state
        .server
        .inference
        .as_ref()
        .is_some_and(|runtime| runtime.slots.is_healthy());
    if healthy {
        Json(serde_json::json!({"status": "ok"})).into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": {
                "code": 503,
                "message": if state.server.inference.is_some() { "Inference engine unavailable" } else { "Loading model" },
                "type": "unavailable_error"
            }})),
        )
            .into_response()
    }
}

async fn healthz(State(state): State<AppState>) -> Response {
    state.server.record_request();
    let engine_healthy = state
        .server
        .inference
        .as_ref()
        .is_none_or(|runtime| runtime.slots.is_healthy());
    let status = if engine_healthy {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (
        status,
        Json(serde_json::json!({"ok": engine_healthy, "degraded": state.server.degraded(), "accelerator_in_use": state.server.inference.is_some()})),
    )
        .into_response()
}

async fn models(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if state.lan && !valid_bearer(&state, &headers) {
        return auth_required();
    }
    state.server.record_request();
    let capabilities = if state
        .server
        .inference
        .as_ref()
        .is_some_and(|runtime| runtime.vision.is_some())
    {
        serde_json::json!(["completion", "multimodal"])
    } else {
        serde_json::json!(["completion"])
    };
    let model_id = "muse-glimmer-30b";
    let meta = state.server.inference.as_ref().map(|runtime| {
        let config = runtime.model.config();
        serde_json::json!({
            "vocab_type": 2,
            "n_vocab": config.vocab_size,
            "n_ctx": runtime.max_context,
            "n_ctx_train": config.context_length,
            "n_embd": config.hidden_dim,
            "n_params": 27_854_794_240_u64,
            "size": state.server.model_bytes.unwrap_or_default(),
            "ftype": "Q4_K - Medium"
        })
    });
    Json(serde_json::json!({"models": [{
        "name": model_id,
        "model": model_id,
        "modified_at": "",
        "size": "",
        "digest": "",
        "type": "model",
        "description": "",
        "tags": [""],
        "capabilities": capabilities,
        "parameters": "",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": "",
            "families": [""],
            "parameter_size": "",
            "quantization_level": ""
        }
    }],
    "object": "list",
    "data": [{
        "id": model_id,
        "aliases": [model_id],
        "tags": [],
        "object": "model",
        "created": 0,
        "owned_by": "muser",
        "meta": meta
    }]}))
    .into_response()
}

async fn props(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if state.lan && !valid_bearer(&state, &headers) {
        return auth_required();
    }
    state.server.record_request();
    let Some(runtime) = state.server.inference.as_ref() else {
        return error_json(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_not_loaded",
            "model is not loaded",
        );
    };
    let default_params = default_generation_params(runtime.max_context);
    Json(serde_json::json!({
        "model_path": state.server.model_path,
        "total_slots": runtime.slots.len(),
        "default_generation_settings": {"params": default_params, "n_ctx": runtime.max_context},
        "model_alias": "muse-glimmer-30b",
        "model_ftype": "Q4_K - Medium",
        "chat_template": runtime.model.chat_template(),
        "chat_template_caps": {
            "supports_string_content": true,
            "supports_typed_content": false,
            "supports_tools": true,
            "supports_tool_calls": true,
            "supports_parallel_tool_calls": true,
            "supports_system_role": true,
            "supports_preserve_reasoning": true,
            "supports_object_arguments": true
        },
        "modalities": {"vision": runtime.vision.is_some(), "video": false, "audio": false},
        "media_marker": "<__media_muser_v0_1__>",
        "endpoint_slots": true,
        "endpoint_props": true,
        "endpoint_metrics": true,
        "ui": true,
        "ui_settings": {},
        "bos_token": "<|begin_of_text|>",
        "eos_token": "<|end_of_text|>",
        "build_info": format!("muser-{}-{}", env!("CARGO_PKG_VERSION"), runtime.backend),
        "is_sleeping": false,
        "cors_proxy_enabled": false
    }))
    .into_response()
}

async fn props_update(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if !valid_management_auth(&state, &headers, true) {
        return auth_required();
    }
    state.server.record_request();
    error_json(
        StatusCode::NOT_IMPLEMENTED,
        "not_supported_error",
        "changing global properties is not enabled by Muser v0.1",
    )
}

fn default_generation_params(_n_ctx: usize) -> serde_json::Value {
    serde_json::json!({
        "n_predict": -1,
        "seed": u32::MAX,
        "temperature": 0.8,
        "dynatemp_range": 0.0,
        "dynatemp_exponent": 1.0,
        "top_k": 40,
        "top_p": 0.95,
        "min_p": 0.05,
        "top_n_sigma": -1.0,
        "xtc_probability": 0.0,
        "xtc_threshold": 0.1,
        "typical_p": 1.0,
        "repeat_last_n": 64,
        "repeat_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "dry_multiplier": 0.0,
        "dry_base": 1.75,
        "dry_allowed_length": 2,
        "dry_penalty_last_n": 64,
        "mirostat": 0,
        "mirostat_tau": 5.0,
        "mirostat_eta": 0.1,
        "adaptive_target": -1.0,
        "adaptive_decay": 0.9,
        "max_tokens": -1,
        "n_keep": 0,
        "n_discard": 0,
        "ignore_eos": false,
        "stream": false,
        "n_probs": 0,
        "min_keep": 0,
        "chat_format": "Content-only",
        "reasoning_format": "none",
        "reasoning_in_content": false,
        "generation_prompt": "",
        "samplers": ["penalties", "dry", "top_n_sigma", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature"],
        "speculative.types": "none",
        "timings_per_token": false,
        "post_sampling_probs": false,
        "backend_sampling": false,
        "lora": []
    })
}

fn native_generation_settings(
    request: &openai::ChatRequest,
    n_predict: i64,
    reported_seed: u64,
) -> serde_json::Value {
    let mut biases = request
        .logit_bias
        .as_ref()
        .map(|biases| {
            biases
                .iter()
                .map(|(token, bias)| serde_json::json!([token, bias]))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    biases.sort_by(|left, right| left[0].as_str().cmp(&right[0].as_str()));
    let stop = match request.stop.as_ref() {
        None => Vec::new(),
        Some(openai::StopField::One(value)) => vec![value.clone()],
        Some(openai::StopField::Many(values)) => values.clone(),
    };
    serde_json::json!({
        "seed": reported_seed,
        "temperature": request.temperature.unwrap_or(0.8),
        "dynatemp_range": request.dynatemp_range.unwrap_or(0.0),
        "dynatemp_exponent": request.dynatemp_exponent.unwrap_or(1.0),
        "top_k": request.top_k.unwrap_or(40),
        "top_p": request.top_p.unwrap_or(0.95),
        "min_p": request.min_p.unwrap_or(0.05),
        "top_n_sigma": request.top_n_sigma.unwrap_or(-1.0),
        "xtc_probability": request.xtc_probability.unwrap_or(0.0),
        "xtc_threshold": request.xtc_threshold.unwrap_or(0.1),
        "typical_p": request.typical_p.unwrap_or(1.0),
        "repeat_last_n": request.repeat_last_n.unwrap_or(64),
        "repeat_penalty": request.repeat_penalty.unwrap_or(1.0),
        "presence_penalty": request.presence_penalty.unwrap_or(0.0),
        "frequency_penalty": request.frequency_penalty.unwrap_or(0.0),
        "dry_multiplier": request.dry_multiplier.unwrap_or(0.0),
        "dry_base": request.dry_base.unwrap_or(1.75),
        "dry_allowed_length": request.dry_allowed_length.unwrap_or(2),
        "dry_penalty_last_n": request.dry_penalty_last_n.unwrap_or(64),
        "dry_sequence_breakers": request.dry_sequence_breakers.clone().unwrap_or_else(|| vec!["\n".into(), ":".into(), "\"".into(), "*".into()]),
        "mirostat": request.mirostat.unwrap_or(0),
        "mirostat_tau": request.mirostat_tau.unwrap_or(5.0),
        "mirostat_eta": request.mirostat_eta.unwrap_or(0.1),
        "adaptive_target": request.adaptive_target.unwrap_or(-1.0),
        "adaptive_decay": request.adaptive_decay.unwrap_or(0.9),
        "stop": stop,
        "max_tokens": n_predict,
        "n_predict": n_predict,
        "n_keep": 0,
        "n_discard": 0,
        "ignore_eos": request.ignore_eos,
        "stream": request.stream,
        "logit_bias": biases,
        "n_probs": request.top_logprobs.unwrap_or(0),
        "min_keep": request.min_keep.unwrap_or(0),
        "grammar": request.grammar.clone().unwrap_or_default(),
        "grammar_lazy": false,
        "grammar_triggers": [],
        "preserved_tokens": [],
        "chat_format": "Content-only",
        "reasoning_format": "deepseek",
        "reasoning_in_content": false,
        "generation_prompt": "",
        "samplers": request.samplers.clone().unwrap_or_else(|| vec!["penalties".into(), "dry".into(), "top_n_sigma".into(), "top_k".into(), "typ_p".into(), "top_p".into(), "min_p".into(), "xtc".into(), "temperature".into()]),
        "speculative.types": "none",
        "timings_per_token": false,
        "post_sampling_probs": false,
        "backend_sampling": false,
        "lora": []
    })
}

#[derive(serde::Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct SlotsQuery {
    fail_on_no_slot: Option<String>,
}

async fn slots_get(
    State(state): State<AppState>,
    Query(query): Query<SlotsQuery>,
    headers: HeaderMap,
) -> Response {
    if state.lan && !valid_bearer(&state, &headers) {
        return auth_required();
    }
    state.server.record_request();
    let Some(runtime) = state.server.inference.as_ref() else {
        return error_json(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_not_loaded",
            "model is not loaded",
        );
    };
    match runtime.slots.status(runtime.max_context) {
        Ok(slots) => {
            if query.fail_on_no_slot.is_some() && slots.iter().all(|slot| slot.is_processing) {
                return error_json(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "unavailable_error",
                    "no slot available",
                );
            }
            let slots = slots
                .into_iter()
                .map(|slot| serde_json::json!({
                    "id": slot.id,
                    "id_task": if slot.is_processing { 0 } else { -1 },
                    "n_ctx": slot.n_ctx,
                    "speculative": runtime.dflash.is_some(),
                    "is_processing": slot.is_processing,
                    "n_prompt_tokens": 0,
                    "n_prompt_tokens_processed": 0,
                    "n_prompt_tokens_cache": 0,
                    "params": default_generation_params(slot.n_ctx),
                    "next_token": [{"has_next_token": true, "has_new_line": false, "n_remain": -1, "n_decoded": 0}]
                }))
                .collect::<Vec<_>>();
            Json(slots).into_response()
        }
        Err(_) => error_json(
            StatusCode::SERVICE_UNAVAILABLE,
            "engine_unavailable",
            "accelerator state is unhealthy",
        ),
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SlotActionRequest {
    id: Option<usize>,
    action: String,
    session_id: Option<String>,
}

async fn slots_action(State(state): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    if !valid_management_auth(&state, &headers, true) {
        return auth_required();
    }
    if !exact_json_content_type(&headers) {
        return error_json(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "invalid_request_error",
            "Content-Type must be exactly application/json",
        );
    }
    let request: SlotActionRequest = match strict_json(&body) {
        Ok(request) => request,
        Err(response) => return response,
    };
    state.server.record_request();
    match request.action.as_str() {
        "erase" => {
            let Some(runtime) = state.server.inference.as_ref() else {
                return error_json(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "model_not_loaded",
                    "model is not loaded",
                );
            };
            let Some(id) = request.id else {
                return error_json(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    "field 'id' is required for erase",
                );
            };
            match runtime.slots.erase_idle(id) {
                Ok(()) => Json(serde_json::json!({"id": id, "action": "erase", "ok": true}))
                    .into_response(),
                Err(crate::state::SlotAcquireError::Overloaded) => error_json(
                    StatusCode::CONFLICT,
                    "conflict",
                    "slot is busy or does not exist",
                ),
                Err(crate::state::SlotAcquireError::Unhealthy) => error_json(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "engine_unavailable",
                    "accelerator state is unhealthy",
                ),
            }
        }
        "save" => {
            let Some(id) = request.session_id.as_deref() else {
                return error_json(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    "field 'session_id' is required for save",
                );
            };
            match state.server.logical_sessions.save(id) {
                Ok(path) => {
                    Json(serde_json::json!({"action": "save", "session_id": id, "path": path}))
                        .into_response()
                }
                Err(error) => error_json(StatusCode::CONFLICT, "session_error", &error),
            }
        }
        "restore" => {
            let Some(id) = request.session_id.as_deref() else {
                return error_json(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    "field 'session_id' is required for restore",
                );
            };
            match state.server.logical_sessions.restore(id) {
                Ok(session) => Json(serde_json::json!({"action": "restore", "session": session}))
                    .into_response(),
                Err(error) => error_json(StatusCode::BAD_REQUEST, "session_restore_error", &error),
            }
        }
        other => error_json(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            &format!("unsupported slot action '{other}'"),
        ),
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SlotCompatQuery {
    action: String,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SlotCompatFile {
    filename: String,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SlotCompatEnvelope {
    schema: String,
    model_sha256: String,
    tokenizer_sha256: [u8; 32],
    template_sha256: [u8; 32],
    layout_abi: String,
    slot: crate::state::SlotSnapshot,
}

/// Source-pinned llama.cpp slot action shape: `POST /slots/{id}?action=...`.
/// Muser's canonical logical-session API remains separate; this compatibility
/// route saves/restores the exact resident target KV and final logits for the
/// named physical slot.
async fn slots_compat_action(
    State(state): State<AppState>,
    Path(id): Path<usize>,
    Query(query): Query<SlotCompatQuery>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if !valid_management_auth(&state, &headers, true) {
        return auth_required();
    }
    let Some(runtime) = state.server.inference.as_ref() else {
        return error_json(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_not_loaded",
            "model is not loaded",
        );
    };
    state.server.record_request();
    if query.action == "erase" {
        return match runtime.slots.erase_idle(id) {
            Ok(()) => Json(serde_json::json!({"id_slot": id})).into_response(),
            Err(crate::state::SlotAcquireError::Overloaded) => error_json(
                StatusCode::CONFLICT,
                "conflict",
                "slot is busy or does not exist",
            ),
            Err(crate::state::SlotAcquireError::Unhealthy) => error_json(
                StatusCode::SERVICE_UNAVAILABLE,
                "engine_unavailable",
                "accelerator state is unhealthy",
            ),
        };
    }
    if !matches!(query.action.as_str(), "save" | "restore") {
        return error_json(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "Invalid action",
        );
    }
    if !exact_json_content_type(&headers) {
        return error_json(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "invalid_request_error",
            "Content-Type must be exactly application/json",
        );
    }
    let request: SlotCompatFile = match strict_json(&body) {
        Ok(request) => request,
        Err(response) => return response,
    };
    let path = match slot_compat_path(&request.filename) {
        Ok(path) => path,
        Err(error) => return error_json(StatusCode::BAD_REQUEST, "invalid_request_error", &error),
    };
    if query.action == "save" {
        let slot = match runtime.slots.snapshot_idle(id) {
            Ok(snapshot) => snapshot,
            Err(error) => return error_json(StatusCode::CONFLICT, "slot_error", &error),
        };
        let n_saved = slot.target.position;
        let envelope = SlotCompatEnvelope {
            schema: "muser.slot-file.v1".into(),
            model_sha256: state.server.model_sha256.clone().unwrap_or_default(),
            tokenizer_sha256: runtime.model.tokenizer_metadata_sha256(),
            template_sha256: runtime.model.chat_template_sha256(),
            layout_abi: "muse-kv-layout-v1".into(),
            slot,
        };
        if let Some(parent) = path.parent() {
            if let Err(error) = std::fs::create_dir_all(parent) {
                return error_json(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "slot_error",
                    &error.to_string(),
                );
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt as _;
                if let Err(error) =
                    std::fs::set_permissions(parent, std::fs::Permissions::from_mode(0o700))
                {
                    return error_json(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "slot_error",
                        &error.to_string(),
                    );
                }
            }
        }
        return match crate::session_store::atomic_private_file(&path, |file| {
            postcard::to_io(&envelope, file)
                .map(|_| ())
                .map_err(|error| error.to_string())
        }) {
            Ok(()) => Json(serde_json::json!({"id_slot": id, "filename": request.filename, "n_saved": n_saved})).into_response(),
            Err(error) => error_json(StatusCode::INTERNAL_SERVER_ERROR, "slot_error", &error),
        };
    }

    let metadata = match std::fs::symlink_metadata(&path) {
        Ok(metadata) if metadata.file_type().is_file() => metadata,
        Ok(_) => {
            return error_json(
                StatusCode::BAD_REQUEST,
                "slot_error",
                "slot snapshot is not a regular file",
            )
        }
        Err(error) => return error_json(StatusCode::NOT_FOUND, "slot_error", &error.to_string()),
    };
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;
        if metadata.mode() & 0o077 != 0 {
            return error_json(
                StatusCode::BAD_REQUEST,
                "slot_error",
                "slot snapshot permissions must exclude group and other access",
            );
        }
    }
    let file = match std::fs::File::open(&path) {
        Ok(file) => file,
        Err(error) => return error_json(StatusCode::NOT_FOUND, "slot_error", &error.to_string()),
    };
    let mut scratch = [0u8; 16 * 1024];
    let envelope: SlotCompatEnvelope = match postcard::from_io((file, &mut scratch)) {
        Ok((envelope, _)) => envelope,
        Err(error) => {
            return error_json(
                StatusCode::BAD_REQUEST,
                "slot_error",
                &format!("slot snapshot is malformed: {error}"),
            )
        }
    };
    if envelope.schema != "muser.slot-file.v1"
        || envelope.model_sha256 != state.server.model_sha256.as_deref().unwrap_or_default()
        || envelope.tokenizer_sha256 != runtime.model.tokenizer_metadata_sha256()
        || envelope.template_sha256 != runtime.model.chat_template_sha256()
        || envelope.layout_abi != "muse-kv-layout-v1"
    {
        return error_json(
            StatusCode::CONFLICT,
            "slot_error",
            "slot snapshot model, tokenizer, template, or layout identity differs",
        );
    }
    let n_restored = envelope.slot.target.position;
    match runtime.slots.restore_idle(id, &envelope.slot) {
        Ok(()) => Json(serde_json::json!({"id_slot": id, "filename": request.filename, "n_restored": n_restored})).into_response(),
        Err(error) => error_json(StatusCode::CONFLICT, "slot_error", &error),
    }
}

fn slot_compat_path(filename: &str) -> Result<PathBuf, String> {
    if filename.is_empty()
        || filename.len() > 255
        || filename == "."
        || filename == ".."
        || filename
            .bytes()
            .any(|byte| !(byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-')))
    {
        return Err("Invalid filename".into());
    }
    let root = std::env::var_os("MUSER_HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .filter(|value| !value.is_empty())
                .map(|home| PathBuf::from(home).join(".muser"))
        })
        .ok_or_else(|| "MUSER_HOME or HOME is required for slot snapshots".to_string())?;
    Ok(root.join("slots").join(filename))
}

#[derive(serde::Deserialize)]
#[serde(untagged)]
enum TokenizeContent {
    Text(String),
    Mixed(Vec<TokenizePart>),
}

#[derive(serde::Deserialize)]
#[serde(untagged)]
enum TokenizePart {
    Text(String),
    Token(u32),
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TokenizeRequest {
    content: Option<TokenizeContent>,
    #[serde(default)]
    add_special: bool,
    #[serde(default = "default_true")]
    parse_special: bool,
    #[serde(default)]
    with_pieces: bool,
}

async fn tokenize(State(state): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    if let Some(response) = inference_json_preflight(&state, &headers, &body, 4 * 1024 * 1024) {
        return response;
    }
    let request: TokenizeRequest = match strict_json(&body) {
        Ok(request) => request,
        Err(response) => return response,
    };
    let Some(runtime) = state.server.inference.as_ref() else {
        return error_json(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_not_loaded",
            "model is not loaded",
        );
    };
    let mut tokens = Vec::new();
    match request.content {
        None => {}
        Some(TokenizeContent::Text(text)) => {
            tokens = runtime
                .model
                .encode_with_options(&text, request.parse_special);
            if request.add_special && runtime.model.adds_bos_token() {
                if let Some(bos) = runtime.model.bos_token_id() {
                    tokens.insert(0, bos);
                }
            }
        }
        Some(TokenizeContent::Mixed(parts)) => {
            let mut first = true;
            for part in parts {
                match part {
                    TokenizePart::Text(text) => {
                        let mut encoded = runtime
                            .model
                            .encode_with_options(&text, request.parse_special);
                        if first && request.add_special && runtime.model.adds_bos_token() {
                            if let Some(bos) = runtime.model.bos_token_id() {
                                encoded.insert(0, bos);
                            }
                        }
                        tokens.extend(encoded);
                    }
                    TokenizePart::Token(token) => tokens.push(token),
                }
                first = false;
            }
        }
    }
    if tokens
        .iter()
        .any(|token| *token as usize >= runtime.model.config().vocab_size)
    {
        return error_json(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "content contains an out-of-vocabulary token ID",
        );
    }
    if request.with_pieces {
        let tokens = tokens
            .into_iter()
            .map(|id| {
                let bytes = runtime.model.token_bytes(id);
                let piece = std::str::from_utf8(bytes).map_or_else(
                    |_| serde_json::json!(bytes),
                    |piece| serde_json::Value::String(piece.into()),
                );
                serde_json::json!({"id": id, "piece": piece})
            })
            .collect::<Vec<_>>();
        Json(serde_json::json!({"tokens": tokens})).into_response()
    } else {
        Json(serde_json::json!({"tokens": tokens})).into_response()
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct DetokenizeRequest {
    tokens: Vec<u32>,
}

async fn detokenize(State(state): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    if let Some(response) = inference_json_preflight(&state, &headers, &body, 16 * 1024 * 1024) {
        return response;
    }
    let request: DetokenizeRequest = match strict_json(&body) {
        Ok(request) => request,
        Err(response) => return response,
    };
    let Some(runtime) = state.server.inference.as_ref() else {
        return error_json(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_not_loaded",
            "model is not loaded",
        );
    };
    if request
        .tokens
        .iter()
        .any(|token| *token as usize >= runtime.model.config().vocab_size)
    {
        return error_json(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "tokens contains an out-of-vocabulary token ID",
        );
    }
    Json(serde_json::json!({"content": runtime.model.decode_tokens(&request.tokens)}))
        .into_response()
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ApplyTemplateRequest {
    messages: Vec<openai::Message>,
    tools: Option<Vec<openai::ToolDefinition>>,
    #[serde(default = "default_true")]
    add_generation_prompt: bool,
}

fn default_true() -> bool {
    true
}

async fn apply_template(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if let Some(response) = inference_json_preflight(&state, &headers, &body, 16 * 1024 * 1024) {
        return response;
    }
    let request: ApplyTemplateRequest = match strict_json(&body) {
        Ok(request) => request,
        Err(response) => return response,
    };
    let Some(runtime) = state.server.inference.as_ref() else {
        return error_json(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_not_loaded",
            "model is not loaded",
        );
    };
    let messages = serde_json::to_value(&request.messages).expect("messages serialize");
    let date = &crate::timefmt::now_rfc3339()[..10];
    let tools = request
        .tools
        .as_ref()
        .map(|value| serde_json::to_value(value).expect("tools serialize"));
    let rendered = match crate::chat_template::render_with_options(
        runtime.model.chat_template(),
        &messages,
        tools.as_ref(),
        date,
        request.add_generation_prompt,
    ) {
        Ok(rendered) => rendered,
        Err(error) => {
            return error_json(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                &format!("Muse chat template: {error}"),
            )
        }
    };
    Json(serde_json::json!({"prompt": rendered})).into_response()
}

fn inference_json_preflight(
    state: &AppState,
    headers: &HeaderMap,
    body: &Bytes,
    limit: usize,
) -> Option<Response> {
    state.server.record_request();
    if state.lan && !valid_bearer(state, headers) {
        return Some(auth_required());
    }
    if !exact_json_content_type(headers) {
        return Some(error_json(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "invalid_request_error",
            "Content-Type must be exactly application/json",
        ));
    }
    if body.len() > limit {
        return Some(error_json(
            StatusCode::PAYLOAD_TOO_LARGE,
            "invalid_request_error",
            "request body exceeds the route-specific limit",
        ));
    }
    None
}

#[allow(clippy::result_large_err)] // Axum Response is the route-native error value.
fn strict_json<T: serde::de::DeserializeOwned>(body: &[u8]) -> Result<T, Response> {
    serde_json::from_slice(body).map_err(|error| {
        error_json(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            &format!("invalid JSON request: {error}"),
        )
    })
}

#[allow(clippy::result_large_err)] // Axum Response is the route-native error value.
fn canonical_json_sha256(body: &[u8]) -> Result<[u8; 32], Response> {
    let value: serde_json::Value = strict_json(body)?;
    // serde_json is built with preserve_order, so canonical bytes must not
    // depend on the parsed map's insertion order: sort object keys
    // recursively before hashing.
    let canonical = serde_json::to_vec(&sorted_json(&value)).map_err(|error| {
        error_json(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            &format!("cannot canonicalize JSON request: {error}"),
        )
    })?;
    Ok(Sha256::digest(canonical).into())
}

fn sorted_json(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => {
            let mut entries: Vec<_> = map.iter().collect();
            entries.sort_by(|left, right| left.0.cmp(right.0));
            serde_json::Value::Object(
                entries
                    .into_iter()
                    .map(|(key, value)| (key.clone(), sorted_json(value)))
                    .collect(),
            )
        }
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.iter().map(sorted_json).collect())
        }
        scalar => scalar.clone(),
    }
}

fn hex_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CompletionRequest {
    model: Option<String>,
    prompt: serde_json::Value,
    #[serde(default)]
    verbose: bool,
    #[serde(default)]
    timings_per_token: bool,
    stream_options: Option<openai::StreamOptions>,
    #[serde(default)]
    return_tokens: bool,
    #[serde(default)]
    stream: bool,
    #[serde(alias = "n_predict", alias = "max_completion_tokens")]
    max_tokens: Option<i64>,
    t_max_predict_ms: Option<i64>,
    response_fields: Option<Vec<String>>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    #[serde(alias = "typ_p")]
    typical_p: Option<f32>,
    min_p: Option<f32>,
    top_n_sigma: Option<f32>,
    min_keep: Option<usize>,
    #[serde(default)]
    ignore_eos: bool,
    #[serde(default, deserialize_with = "openai::deserialize_logit_bias")]
    logit_bias: Option<std::collections::HashMap<String, f32>>,
    repeat_penalty: Option<f32>,
    repeat_last_n: Option<i32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    dry_multiplier: Option<f32>,
    dry_base: Option<f32>,
    dry_allowed_length: Option<usize>,
    dry_penalty_last_n: Option<i32>,
    dry_sequence_breakers: Option<Vec<String>>,
    mirostat: Option<u8>,
    mirostat_tau: Option<f32>,
    mirostat_eta: Option<f32>,
    adaptive_target: Option<f32>,
    adaptive_decay: Option<f32>,
    dynatemp_range: Option<f32>,
    dynatemp_exponent: Option<f32>,
    xtc_probability: Option<f32>,
    xtc_threshold: Option<f32>,
    #[serde(default, deserialize_with = "openai::deserialize_sampler_sequence")]
    samplers: Option<Vec<String>>,
    #[serde(default, deserialize_with = "openai::deserialize_seed")]
    seed: Option<u64>,
    #[serde(alias = "n_cmpl")]
    n: Option<u32>,
    #[serde(default, deserialize_with = "openai::deserialize_slot_id")]
    id_slot: Option<usize>,
    #[serde(default = "default_true")]
    cache_prompt: bool,
    stop: Option<openai::StopField>,
    /// OpenAI completions and the pinned llama server use an integer count.
    #[serde(alias = "n_probs")]
    logprobs: Option<usize>,
    grammar: Option<String>,
    json_schema: Option<serde_json::Value>,
}

fn completion_prompts(
    model: &muser_engine::Model,
    prompt: serde_json::Value,
) -> Result<Vec<(Vec<u32>, serde_json::Value)>, String> {
    fn one(
        model: &muser_engine::Model,
        value: serde_json::Value,
    ) -> Result<(Vec<u32>, serde_json::Value), String> {
        let original = value.clone();
        let mut tokens = Vec::new();
        match value {
            serde_json::Value::String(text) => {
                tokens = model.encode_with_options(&text, true);
                if model.adds_bos_token() {
                    if let Some(bos) = model.bos_token_id() {
                        tokens.insert(0, bos);
                    }
                }
            }
            serde_json::Value::Array(parts) => {
                let mut first = true;
                for part in parts {
                    match part {
                        serde_json::Value::String(text) => {
                            let mut encoded = model.encode_with_options(&text, true);
                            if first && model.adds_bos_token() {
                                if let Some(bos) = model.bos_token_id() {
                                    encoded.insert(0, bos);
                                }
                            }
                            tokens.extend(encoded);
                        }
                        serde_json::Value::Number(number) => {
                            let token = number
                                .as_u64()
                                .and_then(|value| u32::try_from(value).ok())
                                .ok_or_else(|| {
                                    "prompt token IDs must be unsigned 32-bit integers".to_string()
                                })?;
                            tokens.push(token);
                        }
                        _ => {
                            return Err(
                                "a prompt sequence may contain only strings and token IDs".into()
                            )
                        }
                    }
                    first = false;
                }
            }
            serde_json::Value::Object(_) => return Err(
                "multimodal completion prompt objects are not supported; use chat content parts"
                    .into(),
            ),
            _ => return Err("prompt must be a string or token/string array".into()),
        }
        if tokens.is_empty() {
            return Err("prompt must contain at least one token".into());
        }
        if tokens
            .iter()
            .any(|token| *token as usize >= model.config().vocab_size)
        {
            return Err("prompt contains an out-of-vocabulary token ID".into());
        }
        Ok((tokens, original))
    }

    let is_multiple = match &prompt {
        serde_json::Value::Array(values) => {
            values.iter().all(serde_json::Value::is_string)
                || values.iter().any(|value| {
                    matches!(
                        value,
                        serde_json::Value::Array(_) | serde_json::Value::Object(_)
                    )
                })
        }
        _ => false,
    };
    if is_multiple {
        let serde_json::Value::Array(values) = prompt else {
            unreachable!()
        };
        if values.is_empty() {
            return Err("prompt array must not be empty".into());
        }
        values.into_iter().map(|value| one(model, value)).collect()
    } else {
        Ok(vec![one(model, prompt)?])
    }
}

fn select_response_fields(value: serde_json::Value, paths: &[String]) -> serde_json::Value {
    if paths.is_empty() {
        return value;
    }
    let mut selected = serde_json::Map::new();
    for path in paths {
        let mut current = &value;
        let mut valid = true;
        for key in path.split('/') {
            match current.as_object().and_then(|object| object.get(key)) {
                Some(next) => current = next,
                None => {
                    valid = false;
                    break;
                }
            }
        }
        if valid {
            selected.insert(path.clone(), current.clone());
        }
    }
    serde_json::Value::Object(selected)
}

fn compatibility_timings(
    prompt_tokens: usize,
    completion_tokens: usize,
    prompt_ms: f64,
    predicted_ms: f64,
) -> serde_json::Value {
    let prompt_per_token_ms = if prompt_tokens == 0 {
        0.0
    } else {
        prompt_ms / prompt_tokens as f64
    };
    let predicted_per_token_ms = if completion_tokens == 0 {
        0.0
    } else {
        predicted_ms / completion_tokens as f64
    };
    serde_json::json!({
        "cache_n": 0,
        "prompt_n": prompt_tokens,
        "prompt_ms": prompt_ms,
        "prompt_per_token_ms": prompt_per_token_ms,
        "prompt_per_second": if prompt_per_token_ms == 0.0 { 0.0 } else { 1_000.0 / prompt_per_token_ms },
        "predicted_n": completion_tokens,
        "predicted_ms": predicted_ms,
        "predicted_per_token_ms": predicted_per_token_ms,
        "predicted_per_second": if predicted_per_token_ms == 0.0 { 0.0 } else { 1_000.0 / predicted_per_token_ms }
    })
}

async fn completion(
    State(state): State<AppState>,
    Extension(connection): Extension<ConnectionCancellation>,
    OriginalUri(uri): OriginalUri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if let Some(response) = inference_json_preflight(&state, &headers, &body, 16 * 1024 * 1024) {
        return response;
    }
    let request: CompletionRequest = match strict_json(&body) {
        Ok(request) => request,
        Err(response) => return response,
    };
    let Some(runtime) = state.server.inference.as_ref() else {
        return error_json(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_not_loaded",
            "model is not loaded",
        );
    };
    let native = uri.path() != "/v1/completions";
    let prompts = match completion_prompts(&runtime.model, request.prompt) {
        Ok(prompts) => prompts,
        Err(error) => return error_json(StatusCode::BAD_REQUEST, "invalid_request_error", &error),
    };
    let model = request.model.unwrap_or_else(|| "muse-glimmer-30b".into());
    let seed_explicit = request.seed.is_some();
    let reported_seed = request.seed.unwrap_or(u64::from(u32::MAX));
    let seed = request.seed.unwrap_or_else(openai::entropy_seed);
    let n_predict_report = request.max_tokens.unwrap_or(-1);
    let unlimited_output = n_predict_report == -1;
    let max_context = runtime.max_context;
    let max_tokens = match n_predict_report {
        -1 => Some(max_context.saturating_sub(prompts[0].0.len())),
        value if value >= 0 => match usize::try_from(value) {
            Ok(value) => Some(value),
            Err(_) => {
                return error_json(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    "n_predict is too large for this platform",
                )
            }
        },
        _ => {
            return error_json(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "n_predict must be -1 or a nonnegative integer",
            )
        }
    };
    let generation = openai::ChatRequest {
        model: model.clone(),
        messages: vec![openai::Message {
            role: "user".into(),
            content: openai::MessageContent::Text(String::new()),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            reasoning_content: None,
            recipient: None,
            end_turn: None,
        }],
        stream: request.stream,
        stream_options: request.stream_options,
        max_tokens,
        max_completion_tokens: None,
        t_max_predict_ms: request.t_max_predict_ms,
        temperature: request.temperature,
        top_p: request.top_p,
        top_k: request.top_k,
        typical_p: request.typical_p,
        min_p: request.min_p,
        top_n_sigma: request.top_n_sigma,
        min_keep: request.min_keep,
        ignore_eos: request.ignore_eos,
        logit_bias: request.logit_bias,
        repeat_penalty: request.repeat_penalty,
        repeat_last_n: request.repeat_last_n,
        presence_penalty: request.presence_penalty,
        frequency_penalty: request.frequency_penalty,
        dry_multiplier: request.dry_multiplier,
        dry_base: request.dry_base,
        dry_allowed_length: request.dry_allowed_length,
        dry_penalty_last_n: request.dry_penalty_last_n,
        dry_sequence_breakers: request.dry_sequence_breakers,
        mirostat: request.mirostat,
        mirostat_tau: request.mirostat_tau,
        mirostat_eta: request.mirostat_eta,
        adaptive_target: request.adaptive_target,
        adaptive_decay: request.adaptive_decay,
        dynatemp_range: request.dynatemp_range,
        dynatemp_exponent: request.dynatemp_exponent,
        xtc_probability: request.xtc_probability,
        xtc_threshold: request.xtc_threshold,
        samplers: request.samplers,
        reasoning_control: false,
        reasoning_end_signal: None,
        seed: Some(seed),
        n: request.n,
        id_slot: request.id_slot,
        cache_prompt: request.cache_prompt,
        stop: request.stop,
        tools: None,
        tool_choice: None,
        add_generation_prompt: true,
        parallel_tool_calls: true,
        logprobs: request.logprobs.map(|count| count > 0),
        top_logprobs: request.logprobs,
        response_format: None,
        grammar: request.grammar,
        json_schema: request.json_schema,
        muser_prompt_token_ids: Some(prompts[0].0.clone()),
        muser_baseline_ttft: false,
        session_id: None,
        expected_revision: None,
        idempotency_key: None,
        idempotency_request_sha256: None,
    };
    if let Err(error) = openai::precheck(&state.server, &generation) {
        return chat_error(error);
    }
    let (id, created) = openai::new_request_identity();
    let choice_count = generation.n.unwrap_or(1);
    let return_tokens = request.return_tokens;
    let verbose = request.verbose;
    let _timings_per_token = request.timings_per_token;
    let response_fields = request.response_fields.unwrap_or_default();
    let prompt_tokens_total = prompts.first().map_or(0, |(tokens, _)| tokens.len());
    if !generation.stream {
        let server = Arc::clone(&state.server);
        let worker_cancelled = Arc::clone(&connection.0);
        let mut disconnect = DisconnectCancellation::new(Arc::clone(&connection.0));
        let joined = tokio::task::spawn_blocking(move || {
            let result_count = prompts.len() * choice_count as usize;
            let mut choices = Vec::with_capacity(result_count);
            let mut native_results = Vec::with_capacity(result_count);
            let mut completion_tokens = 0usize;
            let mut total_prompt_ms = 0.0f64;
            let mut total_predicted_ms = 0.0f64;
            for (prompt_index, (prompt_tokens, _prompt_json)) in prompts.into_iter().enumerate() {
              for choice_index in 0..choice_count {
                if worker_cancelled.load(Ordering::Acquire) {
                    return Err(openai::ChatError::Cancelled);
                }
                let index = prompt_index as u32 * choice_count + choice_index;
                let mut choice_request = generation.clone();
                choice_request.n = Some(1);
                choice_request.seed = Some(u64::from((seed as u32).wrapping_add(index)));
                choice_request.muser_prompt_token_ids = Some(prompt_tokens.clone());
                if unlimited_output {
                    choice_request.max_tokens =
                        Some(max_context.saturating_sub(prompt_tokens.len()));
                }
                let mut text = String::new();
                let started = Instant::now();
                let mut first_piece_at = None;
                let generated = openai::generate(&server, &choice_request, &id, |piece| {
                    if worker_cancelled.load(Ordering::Acquire) {
                        return Err(openai::ChatError::Cancelled);
                    }
                    if !piece.is_empty() {
                        first_piece_at.get_or_insert_with(Instant::now);
                    }
                    text.push_str(piece);
                    Ok(())
                })?;
                let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
                let prompt_ms = first_piece_at
                    .map_or(elapsed_ms, |at| at.duration_since(started).as_secs_f64() * 1_000.0);
                let predicted_ms = (elapsed_ms - prompt_ms).max(0.001);
                if index == 0 {
                    total_prompt_ms = prompt_ms;
                    total_predicted_ms = predicted_ms;
                    completion_tokens = generated.usage.completion_tokens;
                }
                if native {
                    let tokens = if return_tokens {
                        generated.sampled_tokens.clone()
                    } else {
                        Vec::new()
                    };
                    let completion_probabilities = generated
                        .logprobs
                        .as_ref()
                        .map(|logprobs| {
                            serde_json::to_value(&logprobs.content)
                                .expect("logprobs serialize")
                        })
                        .unwrap_or_else(|| serde_json::Value::Array(Vec::new()));
                    let per_token_ms = if generated.usage.completion_tokens == 0 {
                        0.0
                    } else {
                        predicted_ms / generated.usage.completion_tokens as f64
                    };
                    let mut value = serde_json::json!({
                        "index": index,
                        "content": text,
                        "tokens": tokens,
                        "id_slot": generated.slot_id.unwrap_or(0),
                        "stop": true,
                        "model": model,
                        "tokens_predicted": generated.usage.completion_tokens,
                        "tokens_evaluated": generated.usage.prompt_tokens,
                        "generation_settings": native_generation_settings(
                            &choice_request,
                            n_predict_report,
                            if seed_explicit { choice_request.seed.unwrap_or(reported_seed) } else { reported_seed }
                        ),
                        "prompt": server.inference.as_ref().expect("inference runtime").model.decode_tokens(&prompt_tokens),
                        "has_new_line": text.contains('\n'),
                        "truncated": false,
                        "stop_type": generated.stop_type,
                        "stopping_word": generated.stopping_word,
                        "tokens_cached": generated.context.len().saturating_sub(1),
                        "timings": {
                            "cache_n": 0,
                            "prompt_n": generated.usage.prompt_tokens,
                            "prompt_ms": prompt_ms,
                            "prompt_per_token_ms": if generated.usage.prompt_tokens == 0 { 0.0 } else { prompt_ms / generated.usage.prompt_tokens as f64 },
                            "prompt_per_second": if prompt_ms == 0.0 { 0.0 } else { generated.usage.prompt_tokens as f64 * 1_000.0 / prompt_ms },
                            "predicted_n": generated.usage.completion_tokens,
                            "predicted_ms": predicted_ms,
                            "predicted_per_token_ms": per_token_ms,
                            "predicted_per_second": if per_token_ms == 0.0 { 0.0 } else { 1_000.0 / per_token_ms }
                        }
                    });
                    if !completion_probabilities
                        .as_array()
                        .is_some_and(Vec::is_empty)
                    {
                        value["completion_probabilities"] = completion_probabilities;
                    }
                    native_results.push(select_response_fields(value, &response_fields));
                }
                choices.push(serde_json::json!({
                    "text": text, "index": index, "logprobs": generated.logprobs,
                    "finish_reason": generated.finish_reason
                }));
              }
            }
            if native {
                return Ok::<_, openai::ChatError>(if native_results.len() == 1 {
                    native_results.pop().expect("one native result")
                } else {
                    serde_json::Value::Array(native_results)
                });
            }
            let mut response = serde_json::json!({
                "id": id, "object": "text_completion", "created": created,
                "model": model, "system_fingerprint": format!("muser-seed-{seed:016x}"),
                "choices": choices,
                "usage": {"prompt_tokens": prompt_tokens_total, "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens_total + completion_tokens,
                    "prompt_tokens_details": {"cached_tokens": 0}}
            });
            response["timings"] = compatibility_timings(
                prompt_tokens_total,
                completion_tokens,
                total_prompt_ms,
                total_predicted_ms,
            );
            if verbose {
                response["__verbose"] = serde_json::json!({"muser": true});
            }
            Ok::<_, openai::ChatError>(response)
        })
        .await;
        disconnect.disarm();
        return match joined {
            Ok(Ok(response)) => Json(response).into_response(),
            Ok(Err(error)) => chat_error(error),
            Err(error) => error_json(
                StatusCode::INTERNAL_SERVER_ERROR,
                "generation_error",
                &format!("generation task failed: {error}"),
            ),
        };
    }

    let (sender, receiver) = mpsc::channel::<Result<Bytes, Infallible>>(STREAM_CHANNEL_DEPTH);
    let server = Arc::clone(&state.server);
    tokio::task::spawn_blocking(move || {
        let send = |value: serde_json::Value| -> Result<(), openai::ChatError> {
            send_bounded(&sender, Bytes::from(format!("data: {value}\n\n")))
        };
        for (prompt_index, (prompt_tokens, _)) in prompts.into_iter().enumerate() {
            for choice_index in 0..choice_count {
                let index = prompt_index as u32 * choice_count + choice_index;
                let mut choice_request = generation.clone();
                choice_request.n = Some(1);
                choice_request.seed = Some(u64::from((seed as u32).wrapping_add(index)));
                choice_request.muser_prompt_token_ids = Some(prompt_tokens.clone());
                if unlimited_output {
                    choice_request.max_tokens =
                        Some(max_context.saturating_sub(prompt_tokens.len()));
                }
                let started = Instant::now();
                let mut first_token_at = None;
                let mut streamed_tokens = 0usize;
                let mut streamed_text = String::new();
                let generated = openai::generate_events(&server, &choice_request, &id, |event| {
                    if native {
                        if event.token.is_some() || !event.text.is_empty() {
                            if event.token.is_some() {
                                streamed_tokens += 1;
                                first_token_at.get_or_insert_with(Instant::now);
                            }
                            streamed_text.push_str(event.text);
                            // Pinned llama.cpp always returns the current raw
                            // token in native streaming mode, independent of the
                            // non-streaming `return_tokens` switch.
                            let tokens = event.token.into_iter().collect::<Vec<_>>();
                            let mut value = serde_json::json!({
                                "index": index, "content": event.text, "tokens": tokens,
                                "id_slot": choice_request.id_slot.map_or(-1_i64, |slot| slot as i64), "stop": false,
                                "tokens_predicted": streamed_tokens, "tokens_evaluated": prompt_tokens.len()
                            });
                            if let Some(entry) = event.logprob {
                                value["completion_probabilities"] = serde_json::json!([entry]);
                            }
                            send(value)?;
                        }
                    } else if !event.text.is_empty() {
                        let logprobs = event
                            .logprob
                            .map(|entry| serde_json::json!({"content": [entry]}));
                        send(serde_json::json!({
                            "id": id, "object": "text_completion", "created": created,
                            "model": model, "system_fingerprint": "muser-v0.1",
                            "choices": [{"text": event.text, "index": index,
                                "logprobs": logprobs, "finish_reason": null}]
                        }))?;
                    }
                    Ok(())
                });
                match generated {
                    Ok(generated) => {
                        let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
                        let prompt_ms = first_token_at.map_or(elapsed_ms, |at| {
                            at.duration_since(started).as_secs_f64() * 1_000.0
                        });
                        let predicted_ms = (elapsed_ms - prompt_ms).max(0.001);
                        let predicted_per_token_ms = if generated.usage.completion_tokens == 0 {
                            0.0
                        } else {
                            predicted_ms / generated.usage.completion_tokens as f64
                        };
                        let terminal = if native {
                            serde_json::json!({
                                "index": index, "content": "", "tokens": [],
                                "id_slot": generated.slot_id.unwrap_or(0), "stop": true,
                                "model": model,
                                "tokens_predicted": generated.usage.completion_tokens,
                                "tokens_evaluated": generated.usage.prompt_tokens,
                                "generation_settings": native_generation_settings(
                                    &choice_request,
                                    n_predict_report,
                                    if seed_explicit { choice_request.seed.unwrap_or(reported_seed) } else { reported_seed }
                                ),
                                "prompt": server.inference.as_ref().expect("inference runtime").model.decode_tokens(&prompt_tokens),
                                "has_new_line": streamed_text.contains('\n'),
                                "truncated": false,
                                "stop_type": generated.stop_type,
                                "stopping_word": generated.stopping_word,
                                "tokens_cached": generated.context.len().saturating_sub(1),
                                "timings": {
                                    "cache_n": 0,
                                    "prompt_n": generated.usage.prompt_tokens,
                                    "prompt_ms": prompt_ms,
                                    "prompt_per_token_ms": if generated.usage.prompt_tokens == 0 { 0.0 } else { prompt_ms / generated.usage.prompt_tokens as f64 },
                                    "prompt_per_second": if prompt_ms == 0.0 { 0.0 } else { generated.usage.prompt_tokens as f64 * 1_000.0 / prompt_ms },
                                    "predicted_n": generated.usage.completion_tokens,
                                    "predicted_ms": predicted_ms,
                                    "predicted_per_token_ms": predicted_per_token_ms,
                                    "predicted_per_second": if predicted_per_token_ms == 0.0 { 0.0 } else { 1_000.0 / predicted_per_token_ms }
                                }
                            })
                        } else {
                            serde_json::json!({
                                "id": id, "object": "text_completion", "created": created,
                                "model": model, "system_fingerprint": "muser-v0.1",
                                "choices": [{"text": "", "index": index,
                                    "logprobs": null, "finish_reason": generated.finish_reason}],
                                "usage": generated.usage,
                                "timings": compatibility_timings(
                                    generated.usage.prompt_tokens,
                                    generated.usage.completion_tokens,
                                    prompt_ms,
                                    predicted_ms
                                )
                            })
                        };
                        let _ = send(terminal);
                    }
                    Err(openai::ChatError::Cancelled) => return,
                    Err(error) => {
                        let _ = send(error.json());
                        return;
                    }
                }
            }
        }
        if !native {
            let _ = sender.try_send(Ok(Bytes::from_static(b"data: [DONE]\n\n")));
        }
    });
    sse_response(Body::from_stream(ReceiverStream::new(receiver)))
}

fn send_bounded(
    sender: &mpsc::Sender<Result<Bytes, Infallible>>,
    bytes: Bytes,
) -> Result<(), openai::ChatError> {
    let started = Instant::now();
    let mut item = Ok(bytes);
    loop {
        match sender.try_send(item) {
            Ok(()) => return Ok(()),
            Err(mpsc::error::TrySendError::Full(returned))
                if started.elapsed() < SLOW_CLIENT_GRACE =>
            {
                item = returned;
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(_) => return Err(openai::ChatError::Cancelled),
        }
    }
}

fn send_resumable_bounded(
    sender: &mpsc::Sender<Result<Bytes, Infallible>>,
    bytes: Bytes,
    resumable: bool,
    live_client: &Cell<bool>,
    grace: Duration,
) -> Result<(), openai::ChatError> {
    if !live_client.get() {
        return Ok(());
    }
    let started = Instant::now();
    let mut item = Ok(bytes);
    loop {
        match sender.try_send(item) {
            Ok(()) => return Ok(()),
            Err(mpsc::error::TrySendError::Full(returned)) if started.elapsed() < grace => {
                item = returned;
                std::thread::sleep(Duration::from_millis(5));
            }
            // Resumability only changes the handling of an actually closed
            // socket. A connected client that remains backpressured for the
            // full grace period still cancels this request, as required by
            // the serving isolation contract.
            Err(mpsc::error::TrySendError::Full(_)) => return Err(openai::ChatError::Cancelled),
            Err(mpsc::error::TrySendError::Closed(_)) if resumable => {
                live_client.set(false);
                return Ok(());
            }
            Err(mpsc::error::TrySendError::Closed(_)) => return Err(openai::ChatError::Cancelled),
        }
    }
}

#[derive(serde::Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct OllamaOptions {
    #[serde(default, deserialize_with = "openai::deserialize_seed")]
    seed: Option<u64>,
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    typical_p: Option<f32>,
    min_p: Option<f32>,
    top_n_sigma: Option<f32>,
    min_keep: Option<usize>,
    #[serde(default)]
    ignore_eos: bool,
    repeat_penalty: Option<f32>,
    repeat_last_n: Option<i32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    dry_multiplier: Option<f32>,
    dry_base: Option<f32>,
    dry_allowed_length: Option<usize>,
    dry_penalty_last_n: Option<i32>,
    dry_sequence_breakers: Option<Vec<String>>,
    mirostat: Option<u8>,
    mirostat_tau: Option<f32>,
    mirostat_eta: Option<f32>,
    adaptive_target: Option<f32>,
    adaptive_decay: Option<f32>,
    dynatemp_range: Option<f32>,
    dynatemp_exponent: Option<f32>,
    xtc_probability: Option<f32>,
    xtc_threshold: Option<f32>,
    #[serde(default, deserialize_with = "openai::deserialize_sampler_sequence")]
    samplers: Option<Vec<String>>,
    num_predict: Option<usize>,
    stop: Option<Vec<String>>,
}

/// Presence marker for fields that are part of an upstream request shape but
/// intentionally unsupported by Muser. `Option<Value>` cannot distinguish an
/// absent field from an explicit JSON null, which would silently accept an
/// unsupported request.
#[derive(Default)]
struct UnsupportedField(bool);

impl<'de> serde::Deserialize<'de> for UnsupportedField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let _ = <serde::de::IgnoredAny as serde::Deserialize>::deserialize(deserializer)?;
        Ok(Self(true))
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct OllamaGenerateRequest {
    model: String,
    #[serde(default)]
    prompt: String,
    system: Option<String>,
    images: Option<Vec<String>>,
    #[serde(default = "default_true")]
    stream: bool,
    #[serde(default)]
    raw: bool,
    context: Option<Vec<u32>>,
    options: Option<OllamaOptions>,
    format: Option<serde_json::Value>,
    logprobs: Option<bool>,
    top_logprobs: Option<usize>,
    #[serde(default)]
    suffix: UnsupportedField,
    #[serde(default)]
    keep_alive: UnsupportedField,
    #[serde(default)]
    template: UnsupportedField,
}

fn unsupported_ollama_field(request: &OllamaGenerateRequest) -> Option<&'static str> {
    [
        ("suffix", request.suffix.0),
        ("keep_alive", request.keep_alive.0),
        ("template", request.template.0),
    ]
    .into_iter()
    .find_map(|(name, present)| present.then_some(name))
}

async fn ollama_generate(
    State(state): State<AppState>,
    Extension(connection): Extension<ConnectionCancellation>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if let Some(response) = inference_json_preflight(&state, &headers, &body, 64 * 1024 * 1024) {
        return response;
    }
    let mut request: OllamaGenerateRequest = match strict_json(&body) {
        Ok(request) => request,
        Err(response) => return response,
    };
    if let Some(name) = unsupported_ollama_field(&request) {
        return error_json(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            &format!("unsupported field '{name}'"),
        );
    }
    let (response_format, json_schema) = match request.format.take() {
        None => (None, None),
        Some(serde_json::Value::String(value)) if value == "json" => {
            (Some(serde_json::json!({"type": "json_object"})), None)
        }
        Some(value @ serde_json::Value::Object(_)) => (None, Some(value)),
        Some(_) => {
            return error_json(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "format must be 'json' or a JSON Schema object",
            )
        }
    };
    let Some(runtime) = state.server.inference.as_ref() else {
        return error_json(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_not_loaded",
            "model is not loaded",
        );
    };
    if request.model != "muse-glimmer-30b" {
        return error_json(
            StatusCode::NOT_FOUND,
            "model_not_found",
            &format!("model '{}' is not loaded", request.model),
        );
    }
    let options = request.options.unwrap_or_default();
    let mut messages = Vec::new();
    if let Some(system) = request.system {
        messages.push(openai::Message {
            role: "system".into(),
            content: openai::MessageContent::Text(system),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            reasoning_content: None,
            recipient: None,
            end_turn: None,
        });
    }
    let content = if let Some(images) = request.images {
        let mut parts = vec![openai::ContentPart::Text {
            text: request.prompt.clone(),
        }];
        for image in images {
            parts.push(openai::ContentPart::ImageUrl {
                image_url: openai::ImageUrl {
                    url: format!("data:image/unknown;base64,{image}"),
                },
            });
        }
        openai::MessageContent::Parts(parts)
    } else {
        openai::MessageContent::Text(request.prompt.clone())
    };
    messages.push(openai::Message {
        role: "user".into(),
        content,
        name: None,
        tool_call_id: None,
        tool_calls: None,
        reasoning_content: None,
        recipient: None,
        end_turn: None,
    });
    let exact_prompt = if request.raw || request.context.is_some() {
        let mut tokens = request.context.unwrap_or_default();
        tokens.extend(runtime.model.encode_with_options(&request.prompt, true));
        if tokens.is_empty() {
            return error_json(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "raw/context prompt must contain at least one token",
            );
        }
        Some(tokens)
    } else {
        None
    };
    let seed = options.seed.unwrap_or_else(openai::entropy_seed);
    let generation = openai::ChatRequest {
        model: request.model.clone(),
        messages,
        stream: request.stream,
        stream_options: None,
        max_tokens: options.num_predict,
        max_completion_tokens: None,
        t_max_predict_ms: None,
        temperature: options.temperature,
        top_p: options.top_p,
        top_k: options.top_k,
        typical_p: options.typical_p,
        min_p: options.min_p,
        top_n_sigma: options.top_n_sigma,
        min_keep: options.min_keep,
        ignore_eos: options.ignore_eos,
        logit_bias: None,
        repeat_penalty: options.repeat_penalty,
        repeat_last_n: options.repeat_last_n,
        presence_penalty: options.presence_penalty,
        frequency_penalty: options.frequency_penalty,
        dry_multiplier: options.dry_multiplier,
        dry_base: options.dry_base,
        dry_allowed_length: options.dry_allowed_length,
        dry_penalty_last_n: options.dry_penalty_last_n,
        dry_sequence_breakers: options.dry_sequence_breakers,
        mirostat: options.mirostat,
        mirostat_tau: options.mirostat_tau,
        mirostat_eta: options.mirostat_eta,
        adaptive_target: options.adaptive_target,
        adaptive_decay: options.adaptive_decay,
        dynatemp_range: options.dynatemp_range,
        dynatemp_exponent: options.dynatemp_exponent,
        xtc_probability: options.xtc_probability,
        xtc_threshold: options.xtc_threshold,
        samplers: options.samplers,
        reasoning_control: false,
        reasoning_end_signal: None,
        seed: Some(seed),
        n: Some(1),
        id_slot: None,
        cache_prompt: false,
        stop: options.stop.map(openai::StopField::Many),
        tools: None,
        tool_choice: None,
        add_generation_prompt: true,
        parallel_tool_calls: true,
        logprobs: request.logprobs,
        top_logprobs: request.top_logprobs,
        response_format,
        grammar: None,
        json_schema,
        muser_prompt_token_ids: exact_prompt,
        muser_baseline_ttft: false,
        session_id: None,
        expected_revision: None,
        idempotency_key: None,
        idempotency_request_sha256: None,
    };
    if let Err(error) = openai::precheck(&state.server, &generation) {
        return chat_error(error);
    }
    let (id, _) = openai::new_request_identity();
    let model = request.model;
    let stream = request.stream;
    let started = Instant::now();
    if !stream {
        let server = Arc::clone(&state.server);
        let worker_cancelled = Arc::clone(&connection.0);
        let mut disconnect = DisconnectCancellation::new(Arc::clone(&connection.0));
        let joined = tokio::task::spawn_blocking(move || {
            let mut response_text = String::new();
            let generated = openai::generate(&server, &generation, &id, |piece| {
                if worker_cancelled.load(Ordering::Acquire) {
                    return Err(openai::ChatError::Cancelled);
                }
                response_text.push_str(piece);
                Ok(())
            })?;
            let (response_text, thinking) = ollama_visible_output(&response_text)?;
            let elapsed = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
            Ok::<_, openai::ChatError>(ollama_final(
                &model,
                response_text,
                thinking,
                generated.finish_reason,
                generated.usage,
                generated.context,
                elapsed,
                generated.logprobs,
            ))
        })
        .await;
        disconnect.disarm();
        return match joined {
            Ok(Ok(value)) => Json(value).into_response(),
            Ok(Err(error)) => chat_error(error),
            Err(error) => error_json(
                StatusCode::INTERNAL_SERVER_ERROR,
                "generation_error",
                &format!("generation task failed: {error}"),
            ),
        };
    }
    let (sender, receiver) = mpsc::channel::<Result<Bytes, Infallible>>(STREAM_CHANNEL_DEPTH);
    let server = Arc::clone(&state.server);
    tokio::task::spawn_blocking(move || {
        let mut atem = openai::AtemStreamParser::default();
        let generated = openai::generate_events(&server, &generation, &id, |event| {
            let events = atem.push(event.text).map_err(|error| {
                openai::ChatError::Engine(format!("malformed Muse ATEM output: {error}"))
            })?;
            for parsed in events {
                let mut value = serde_json::json!({
                    "model": model, "created_at": crate::timefmt::now_rfc3339(),
                    "response": "", "done": false
                });
                match parsed {
                    openai::AtemStreamEvent::Content(content) => value["response"] = content.into(),
                    openai::AtemStreamEvent::Reasoning(thinking) => {
                        value["thinking"] = thinking.into()
                    }
                    openai::AtemStreamEvent::ToolCall { .. } => {
                        return Err(openai::ChatError::Engine(
                            "Ollama generate emitted an unexpected tool call".into(),
                        ));
                    }
                }
                if let Some(logprob) = event.logprob {
                    value["logprobs"] = serde_json::json!([logprob]);
                }
                send_bounded(&sender, Bytes::from(format!("{value}\n")))?;
            }
            Ok(())
        });
        match generated {
            Ok(generated) => {
                let mut response_text = String::new();
                let mut thinking = String::new();
                match atem.finish_stream() {
                    Ok(events) => {
                        for event in events {
                            match event {
                                openai::AtemStreamEvent::Content(value) => {
                                    response_text.push_str(&value)
                                }
                                openai::AtemStreamEvent::Reasoning(value) => {
                                    thinking.push_str(&value)
                                }
                                openai::AtemStreamEvent::ToolCall { .. } => {}
                            }
                        }
                    }
                    Err(error) => {
                        let _ = send_bounded(
                            &sender,
                            Bytes::from(format!("{}\n", openai::ChatError::Engine(error).json())),
                        );
                        return;
                    }
                }
                let elapsed = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
                let value = ollama_final(
                    &model,
                    response_text,
                    thinking,
                    generated.finish_reason,
                    generated.usage,
                    generated.context,
                    elapsed,
                    generated.logprobs,
                );
                let _ = send_bounded(&sender, Bytes::from(format!("{value}\n")));
            }
            Err(openai::ChatError::Cancelled) => {}
            Err(error) => {
                let _ = send_bounded(&sender, Bytes::from(format!("{}\n", error.json())));
            }
        }
    });
    let mut response = Body::from_stream(ReceiverStream::new(receiver)).into_response();
    response.headers_mut().insert(
        CONTENT_TYPE,
        HeaderValue::from_static("application/x-ndjson"),
    );
    response
        .headers_mut()
        .insert(CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response
}

fn ollama_visible_output(text: &str) -> Result<(String, String), openai::ChatError> {
    let mut parser = openai::AtemStreamParser::default();
    let mut response = String::new();
    let mut thinking = String::new();
    let mut events = parser.push(text).map_err(|error| {
        openai::ChatError::Engine(format!("malformed Muse ATEM output: {error}"))
    })?;
    events.extend(parser.finish_stream().map_err(|error| {
        openai::ChatError::Engine(format!("malformed Muse ATEM output: {error}"))
    })?);
    for event in events {
        match event {
            openai::AtemStreamEvent::Content(value) => response.push_str(&value),
            openai::AtemStreamEvent::Reasoning(value) => thinking.push_str(&value),
            openai::AtemStreamEvent::ToolCall { .. } => {
                return Err(openai::ChatError::Engine(
                    "Ollama generate emitted an unexpected tool call".into(),
                ));
            }
        }
    }
    Ok((response, thinking))
}

#[allow(clippy::too_many_arguments)]
fn ollama_final(
    model: &str,
    response: String,
    thinking: String,
    finish_reason: &str,
    usage: openai::Usage,
    context: Vec<u32>,
    elapsed_ns: u64,
    logprobs: Option<openai::ChoiceLogprobs>,
) -> serde_json::Value {
    let mut value = serde_json::json!({
        "model": model,
        "created_at": crate::timefmt::now_rfc3339(),
        "response": response,
        "done": true,
        "done_reason": finish_reason,
        "context": context,
        "total_duration": elapsed_ns,
        "load_duration": 0,
        "prompt_eval_count": usage.prompt_tokens,
        "prompt_eval_duration": 0,
        "eval_count": usage.completion_tokens,
        "eval_duration": elapsed_ns
    });
    if let Some(logprobs) = logprobs {
        value["logprobs"] = serde_json::to_value(logprobs).expect("logprobs serialize");
    }
    if !thinking.is_empty() {
        value["thinking"] = thinking.into();
    }
    value
}

#[derive(serde::Deserialize)]
#[serde(untagged)]
enum EmbeddingInput {
    Text(String),
    Texts(Vec<String>),
    Tokens(Vec<u32>),
    TokenBatches(Vec<Vec<u32>>),
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct EmbeddingRequest {
    model: Option<String>,
    input: Option<EmbeddingInput>,
    content: Option<EmbeddingInput>,
    encoding_format: Option<String>,
    dimensions: Option<usize>,
    embd_normalize: Option<i32>,
    user: Option<String>,
}

async fn embeddings(
    State(state): State<AppState>,
    OriginalUri(uri): OriginalUri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if let Some(response) = inference_json_preflight(&state, &headers, &body, 16 * 1024 * 1024) {
        return response;
    }
    let request: EmbeddingRequest = match strict_json(&body) {
        Ok(request) => request,
        Err(response) => return response,
    };
    let _ = request.user;
    let native = uri.path() != "/v1/embeddings";
    let input = match (request.input, request.content) {
        (Some(input), None) | (None, Some(input)) => input,
        (None, None) => {
            return error_json(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "\"input\" or \"content\" must be provided",
            )
        }
        (Some(_), Some(_)) => {
            return error_json(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "input and content are mutually exclusive",
            )
        }
    };
    if request.embd_normalize.is_some_and(|value| value != 2) {
        return error_json(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "Muse embeddings require embd_normalize=2 (L2 normalization)",
        );
    }
    let model = request.model.unwrap_or_else(|| "muse-glimmer-30b".into());
    if model != "muse-glimmer-30b" {
        return error_json(
            StatusCode::NOT_FOUND,
            "model_not_found",
            &format!("model '{model}' is not loaded"),
        );
    }
    let encoding = request.encoding_format.as_deref().unwrap_or("float");
    if !matches!(encoding, "float" | "base64") {
        return error_json(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "encoding_format must be 'float' or 'base64'",
        );
    }
    let Some(runtime) = state.server.inference.as_ref() else {
        return error_json(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_not_loaded",
            "model is not loaded",
        );
    };
    let width = runtime.model.config().hidden_dim;
    if request
        .dimensions
        .is_some_and(|dimensions| dimensions != width)
    {
        return error_json(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            &format!("dimensions truncation is unsupported; dimensions must equal {width}"),
        );
    }
    let encode_text = |text: &str| {
        let mut tokens = runtime.model.encode_with_options(text, true);
        if runtime.model.adds_bos_token() {
            if let Some(bos) = runtime.model.bos_token_id() {
                tokens.insert(0, bos);
            }
        }
        tokens
    };
    let inputs = match input {
        EmbeddingInput::Text(text) => vec![encode_text(&text)],
        EmbeddingInput::Texts(texts) => texts.iter().map(|text| encode_text(text)).collect(),
        EmbeddingInput::Tokens(tokens) => vec![tokens],
        EmbeddingInput::TokenBatches(tokens) => tokens,
    };
    if inputs.is_empty() || inputs.iter().any(Vec::is_empty) {
        return error_json(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "input must contain at least one nonempty sequence",
        );
    }
    let token_count = inputs.iter().map(Vec::len).sum::<usize>();
    let base64 = encoding == "base64";
    let server = Arc::clone(&state.server);
    match tokio::task::spawn_blocking(move || {
        let runtime = server
            .inference
            .as_ref()
            .ok_or(openai::ChatError::Unavailable)?;
        let mut permit =
            runtime
                .slots
                .acquire(Duration::from_secs(3))
                .map_err(|error| match error {
                    crate::state::SlotAcquireError::Overloaded => openai::ChatError::Overloaded,
                    crate::state::SlotAcquireError::Unhealthy => openai::ChatError::Unavailable,
                })?;
        let session = permit.session_mut();
        let mut data = Vec::with_capacity(inputs.len());
        for (index, tokens) in inputs.iter().enumerate() {
            session.reset();
            let vector = session
                .embedding(tokens)
                .map_err(|error| openai::ChatError::Engine(error.to_string()))?;
            if native {
                data.push(serde_json::json!({
                    "index": index, "embedding": [vector]
                }));
                continue;
            }
            let embedding = if base64 {
                let mut bytes = Vec::with_capacity(vector.len() * 4);
                for value in vector {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                serde_json::Value::String(base64::engine::general_purpose::STANDARD.encode(bytes))
            } else {
                serde_json::to_value(vector).expect("finite normalized embedding")
            };
            let mut item = serde_json::json!({
                "object": "embedding", "embedding": embedding, "index": index
            });
            if base64 {
                item["encoding_format"] = serde_json::Value::String("base64".into());
            }
            data.push(item);
        }
        session.reset();
        Ok::<_, openai::ChatError>(if native {
            serde_json::Value::Array(data)
        } else {
            serde_json::json!({
                "object": "list", "data": data, "model": model,
                "usage": {"prompt_tokens": token_count, "total_tokens": token_count}
            })
        })
    })
    .await
    {
        Ok(Ok(value)) => Json(value).into_response(),
        Ok(Err(error)) => chat_error(error),
        Err(error) => error_json(
            StatusCode::INTERNAL_SERVER_ERROR,
            "generation_error",
            &format!("embedding task failed: {error}"),
        ),
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ResumableStreamQuery {
    conv_id: String,
    #[serde(default)]
    from: usize,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ReasoningControlRequest {
    id: String,
    action: String,
}

async fn reasoning_control(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    state.server.record_request();
    if state.lan && !valid_bearer(&state, &headers) {
        return auth_required();
    }
    if !exact_json_content_type(&headers) {
        return error_json(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "invalid_request_error",
            "Content-Type must be exactly application/json",
        );
    }
    let request: ReasoningControlRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(error) => {
            return error_json(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                &format!("invalid JSON request: {error}"),
            )
        }
    };
    if request.id.is_empty() {
        return error_json(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "missing completion id",
        );
    }
    if request.action != "reasoning_end" {
        return error_json(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "unknown control action",
        );
    }
    let signal = state
        .reasoning_controls
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(&request.id)
        .cloned();
    match signal {
        Some(signal) => {
            signal.store(true, Ordering::Release);
            Json(serde_json::json!({"success": true})).into_response()
        }
        None => Json(serde_json::json!({
            "success": false,
            "message": "no active completion for this id"
        }))
        .into_response(),
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct StreamLookupRequest {
    conversation_ids: Vec<String>,
}

async fn resumable_stream_get(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<ResumableStreamQuery>,
) -> Response {
    state.server.record_request();
    if state.lan && !valid_bearer(&state, &headers) {
        return auth_required();
    }
    if !valid_conversation_id(&query.conv_id) {
        return error_json(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "conv_id must contain 1..=256 visible ASCII characters",
        );
    }
    let Some(session) = state.streams.get(&query.conv_id) else {
        return error_json(
            StatusCode::NOT_FOUND,
            "not_found_error",
            "Stream not found or expired",
        );
    };
    if matches!(session.snapshot(query.from), ReadSnapshot::Lost) {
        return error_json(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "Stream offset lost, please restart",
        );
    }
    let (sender, receiver) = mpsc::channel::<Result<Bytes, Infallible>>(STREAM_CHANNEL_DEPTH);
    tokio::spawn(async move {
        let mut offset = query.from;
        loop {
            match session.snapshot(offset) {
                ReadSnapshot::Lost => {
                    let frame = "data: {\"error\":{\"message\":\"Stream offset lost, please restart\",\"type\":\"invalid_request_error\"}}\n\n";
                    let _ = sender.send(Ok(Bytes::from_static(frame.as_bytes()))).await;
                    break;
                }
                ReadSnapshot::Data(bytes) => {
                    offset = offset.saturating_add(bytes.len());
                    if sender.send(Ok(Bytes::from(bytes))).await.is_err() {
                        break;
                    }
                }
                ReadSnapshot::Pending => session.changed().await,
                ReadSnapshot::Done => break,
            }
        }
    });
    sse_response(Body::from_stream(ReceiverStream::new(receiver)))
}

async fn resumable_stream_lookup(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    state.server.record_request();
    if state.lan && !valid_bearer(&state, &headers) {
        return auth_required();
    }
    if !exact_json_content_type(&headers) {
        return error_json(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "invalid_request_error",
            "Content-Type must be exactly application/json",
        );
    }
    let request: StreamLookupRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(error) => {
            return error_json(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                &format!("invalid JSON request: {error}"),
            )
        }
    };
    if request.conversation_ids.len() > 64
        || request
            .conversation_ids
            .iter()
            .any(|id| !valid_conversation_id(id))
    {
        return error_json(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "conversation_ids must contain at most 64 valid IDs",
        );
    }
    Json(state.streams.lookup(&request.conversation_ids)).into_response()
}

async fn resumable_stream_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<ResumableStreamQuery>,
) -> Response {
    state.server.record_request();
    if state.lan && !valid_bearer(&state, &headers) {
        return auth_required();
    }
    if !valid_conversation_id(&query.conv_id) || query.from != 0 {
        return error_json(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "DELETE requires a valid conv_id and no nonzero from offset",
        );
    }
    state.streams.evict_and_cancel(&query.conv_id);
    StatusCode::NO_CONTENT.into_response()
}

fn valid_conversation_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 256
        && id
            .bytes()
            .all(|byte| byte.is_ascii_graphic() && !matches!(byte, b'?' | b'#'))
}

async fn chat(
    State(state): State<AppState>,
    Extension(connection): Extension<ConnectionCancellation>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    state.server.record_request();
    if state.lan && !valid_bearer(&state, &headers) {
        return auth_required();
    }
    if !exact_json_content_type(&headers) {
        return error_json(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "invalid_request_error",
            "Content-Type must be exactly application/json",
        );
    }
    let mut request: openai::ChatRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(error) => {
            return error_json(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                &format!("invalid JSON request: {error}"),
            )
        }
    };
    request.idempotency_key = headers
        .get("idempotency-key")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    request.idempotency_request_sha256 = match canonical_json_sha256(&body) {
        Ok(digest) => Some(digest),
        Err(response) => return response,
    };
    if let Err(error) = openai::precheck(&state.server, &request) {
        return chat_error(error);
    }
    let conversation_id = headers
        .get("x-conversation-id")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    if conversation_id
        .as_deref()
        .is_some_and(|id| !valid_conversation_id(id))
    {
        return error_json(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "X-Conversation-Id must contain 1..=256 visible ASCII characters",
        );
    }
    if conversation_id.is_some() && !request.stream {
        return error_json(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "X-Conversation-Id requires stream=true",
        );
    }
    let seed = request.seed.unwrap_or_else(openai::entropy_seed);
    request.seed = Some(seed);
    let (id, created) = openai::new_request_identity();
    let reasoning_registration = if request.reasoning_control {
        let signal = Arc::new(AtomicBool::new(false));
        request.reasoning_end_signal = Some(Arc::clone(&signal));
        state
            .reasoning_controls
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(id.clone(), Arc::clone(&signal));
        Some(ReasoningControlRegistration {
            id: id.clone(),
            signal,
            controls: Arc::clone(&state.reasoning_controls),
        })
    } else {
        None
    };
    let choice_count = request.n.unwrap_or(1);
    if !request.stream {
        let server = Arc::clone(&state.server);
        let worker_cancelled = Arc::clone(&connection.0);
        let mut disconnect = DisconnectCancellation::new(Arc::clone(&connection.0));
        let joined = tokio::task::spawn_blocking(move || {
            let _reasoning_registration = reasoning_registration;
            let mut choices = Vec::with_capacity(choice_count as usize);
            let started = Instant::now();
            let mut first_piece_at = None;
            for index in 0..choice_count {
                if worker_cancelled.load(Ordering::Acquire) {
                    return Err(openai::ChatError::Cancelled);
                }
                let mut choice_request = request.clone();
                choice_request.n = Some(1);
                choice_request.seed = Some(u64::from((seed as u32).wrapping_add(index)));
                let mut text = String::new();
                let mut generated = openai::generate(&server, &choice_request, &id, |piece| {
                    if worker_cancelled.load(Ordering::Acquire) {
                        return Err(openai::ChatError::Cancelled);
                    }
                    if !piece.is_empty() {
                        first_piece_at.get_or_insert_with(Instant::now);
                    }
                    text.push_str(piece);
                    Ok(())
                })?;
                generated.text = text;
                choices.push(generated);
            }
            let response = openai::response_many(id, created, request.model, choices);
            let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
            let prompt_ms = first_piece_at.map_or(elapsed_ms, |at| {
                at.duration_since(started).as_secs_f64() * 1_000.0
            });
            let mut value = serde_json::to_value(&response).expect("chat response serializes");
            value["timings"] = compatibility_timings(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                prompt_ms,
                (elapsed_ms - prompt_ms).max(0.001),
            );
            Ok::<_, openai::ChatError>(value)
        })
        .await;
        disconnect.disarm();
        return match joined {
            Ok(Ok(response)) => Json(response).into_response(),
            Ok(Err(error)) => chat_error(error),
            Err(error) => error_json(
                StatusCode::INTERNAL_SERVER_ERROR,
                "generation_error",
                &format!("generation task failed: {error}"),
            ),
        };
    }

    let model = request.model.clone();
    let resumable = conversation_id.map(|id| state.streams.create_or_replace(id));
    let (sender, receiver) = mpsc::channel::<Result<Bytes, Infallible>>(STREAM_CHANNEL_DEPTH);
    let server = Arc::clone(&state.server);
    tokio::task::spawn_blocking(move || {
        let _reasoning_registration = reasoning_registration;
        let _finish = resumable.as_ref().map(|session| session.finish_guard());
        let live_client = Cell::new(true);
        let send = |frame: String| -> Result<(), openai::ChatError> {
            if let Some(session) = &resumable {
                if session.is_cancelled() || !session.append(frame.as_bytes()) {
                    return Err(openai::ChatError::Cancelled);
                }
            }
            send_resumable_bounded(
                &sender,
                Bytes::from(frame),
                resumable.is_some(),
                &live_client,
                SLOW_CLIENT_GRACE,
            )
        };
        let stream_started = Instant::now();
        for choice_index in 0..choice_count {
            let choice_seed = seed.wrapping_add(choice_index as u64);
            if send(sse_json(&openai::role_chunk_indexed(
                &id,
                created,
                &model,
                choice_seed,
                choice_index,
            )))
            .is_err()
            {
                return;
            }
            let mut choice_request = request.clone();
            choice_request.n = Some(1);
            choice_request.seed = Some(choice_seed);
            let mut raw_output = String::new();
            let mut atem = openai::AtemStreamParser::default();
            let send_events =
                |events: Vec<openai::AtemStreamEvent>, logprob: Option<&openai::TokenLogprob>| {
                    let event_count = events.len();
                    for (event_index, event) in events.into_iter().enumerate() {
                        let mut value = match event {
                            openai::AtemStreamEvent::Content(content) => serde_json::json!({
                                "id": id, "object": "chat.completion.chunk", "created": created,
                                "model": model, "system_fingerprint": "muser-v0.1",
                                "choices": [{"index": choice_index,
                                    "delta": {"content": content}, "finish_reason": null}]
                            }),
                            openai::AtemStreamEvent::Reasoning(reasoning) => serde_json::json!({
                                "id": id, "object": "chat.completion.chunk", "created": created,
                                "model": model, "system_fingerprint": "muser-v0.1",
                                "choices": [{"index": choice_index,
                                    "delta": {"reasoning_content": reasoning}, "finish_reason": null}]
                            }),
                            openai::AtemStreamEvent::ToolCall { index, call } => {
                                openai::validate_streamed_atem_call_indexed(
                                    &choice_request,
                                    index,
                                    &call,
                                )?;
                                serde_json::json!({
                                    "id": id, "object": "chat.completion.chunk", "created": created,
                                    "model": model, "system_fingerprint": "muser-v0.1",
                                    "choices": [{"index": choice_index,
                                        "delta": {"tool_calls": [{"index": index, "id": call.id,
                                            "type": call.kind, "function": {"name": call.function.name,
                                                "arguments": call.function.arguments}}]},
                                        "finish_reason": null}]
                                })
                            }
                        };
                        if event_index + 1 == event_count {
                            if let Some(logprob) = logprob {
                                value["choices"][0]["logprobs"] =
                                    serde_json::json!({"content": [logprob]});
                            }
                        }
                        send(format!("data: {value}\n\n"))?;
                    }
                    Ok::<(), openai::ChatError>(())
                };
            let generated = openai::generate_events(&server, &choice_request, &id, |event| {
                raw_output.push_str(event.text);
                let events = atem.push(event.text).map_err(|error| {
                    openai::ChatError::Engine(format!("malformed Muse ATEM output: {error}"))
                })?;
                send_events(events, event.logprob)
            });
            match generated {
                Ok(generated) => {
                    match atem.finish_stream() {
                        Ok(events) => {
                            if send_events(events, None).is_err() {
                                return;
                            }
                        }
                        Err(error) => {
                            let error = openai::ChatError::Engine(format!(
                                "malformed Muse ATEM output: {error}"
                            ));
                            let _ = send(format!("data: {}\n\n", error.json()));
                            return;
                        }
                    }
                    let _ = send(sse_json(&openai::terminal_chunk_indexed(
                        &id,
                        created,
                        &model,
                        openai::atem_finish_reason(&raw_output, generated.finish_reason),
                        choice_index,
                    )));
                    if request
                        .stream_options
                        .as_ref()
                        .is_some_and(|options| options.include_usage)
                    {
                        let elapsed_ms = stream_started.elapsed().as_secs_f64() * 1_000.0;
                        let usage = generated.usage;
                        let mut value =
                            serde_json::to_value(openai::usage_chunk(&id, created, &model, usage))
                                .expect("usage chunk serializes");
                        value["timings"] = compatibility_timings(
                            usage.prompt_tokens,
                            usage.completion_tokens,
                            elapsed_ms,
                            0.001,
                        );
                        if send(format!("data: {value}\n\n")).is_err() {
                            return;
                        }
                    }
                }
                Err(openai::ChatError::Cancelled) => return,
                Err(error) => {
                    let _ = send(format!("data: {}\n\n", error.json()));
                    return;
                }
            }
        }
        let _ = send("data: [DONE]\n\n".into());
    });
    sse_response(Body::from_stream(ReceiverStream::new(receiver)))
}

async fn nodes_create(
    State(state): State<AppState>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    state.server.record_request();
    if !valid_management_auth(&state, &headers, true) {
        return auth_required();
    }
    if !peer.ip().is_loopback() && !state.lan {
        return error_json(
            StatusCode::FORBIDDEN,
            "forbidden",
            "node management connection policy mismatch",
        );
    }
    if !exact_json_content_type(&headers) {
        return error_json(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "invalid_request_error",
            "Content-Type must be exactly application/json",
        );
    }
    let reply = nodes_api::create(&state.server, &body);
    reply_response(reply)
}

async fn nodes_list(
    State(state): State<AppState>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
) -> Response {
    state
        .server
        .telemetry_requests
        .fetch_add(1, Ordering::Relaxed);
    if !valid_management_auth(&state, &headers, false) {
        return auth_required();
    }
    if !peer.ip().is_loopback() && !state.lan {
        return error_json(
            StatusCode::FORBIDDEN,
            "forbidden",
            "node management connection policy mismatch",
        );
    }
    reply_response(nodes_api::list(&state.server))
}

async fn nodes_progress(
    State(state): State<AppState>,
    Path(name): Path<String>,
    headers: HeaderMap,
) -> Response {
    state
        .server
        .telemetry_requests
        .fetch_add(1, Ordering::Relaxed);
    if !valid_management_auth(&state, &headers, false) {
        return auth_required();
    }
    let Some(source) = nodes_api::progress_source(&state.server, &name) else {
        return error_json(
            StatusCode::NOT_FOUND,
            "not_found",
            &format!("no onboarding progress for node {name}"),
        );
    };
    let (sender, receiver) = mpsc::channel::<Result<Bytes, Infallible>>(STREAM_CHANNEL_DEPTH);
    tokio::task::spawn_blocking(move || {
        let _ = nodes_api::produce_progress_frames(source, &name, |frame| {
            let started = Instant::now();
            let mut item = Ok(Bytes::from(frame));
            loop {
                match sender.try_send(item) {
                    Ok(()) => return Ok(()),
                    Err(mpsc::error::TrySendError::Full(returned))
                        if started.elapsed() < SLOW_CLIENT_GRACE =>
                    {
                        item = returned;
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(_) => {
                        return Err(io::Error::new(
                            io::ErrorKind::BrokenPipe,
                            "progress client disconnected or remained backpressured",
                        ))
                    }
                }
            }
        });
    });
    sse_response(Body::from_stream(ReceiverStream::new(receiver)))
}

async fn dashboard_login(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if !state.tls {
        return error_json(
            StatusCode::BAD_REQUEST,
            "tls_required",
            "dashboard sessions require HTTPS; use bearer authentication on loopback HTTP",
        );
    }
    if !valid_bearer(&state, &headers) {
        return auth_required();
    }
    let Some(expected_origin) = single_header(&headers, ORIGIN.as_str()) else {
        return error_json(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "Origin is required",
        );
    };
    let Some(request_origin) = request_origin(&state, &headers) else {
        return error_json(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "request Host must contain one valid authority",
        );
    };
    if expected_origin != request_origin {
        return error_json(
            StatusCode::FORBIDDEN,
            "origin_mismatch",
            "Origin must exactly match this server's HTTPS scheme and authority",
        );
    }
    let expected_origin = expected_origin.to_owned();
    let session = random_secret();
    let csrf = random_secret();
    state
        .dashboard_sessions
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .insert(
            session.clone(),
            DashboardSession {
                csrf: csrf.clone(),
                origin: expected_origin,
                expires: Instant::now() + Duration::from_secs(3600),
            },
        );
    let mut response =
        Json(serde_json::json!({"csrf_token": csrf, "expires_in": 3600})).into_response();
    let cookie =
        format!("muser_session={session}; Secure; HttpOnly; SameSite=Strict; Path=/; Max-Age=3600");
    response.headers_mut().insert(
        "set-cookie",
        HeaderValue::from_str(&cookie).expect("generated cookie is ASCII"),
    );
    response
}

async fn websocket_ticket(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if !valid_management_auth(&state, &headers, true) {
        return auth_required();
    }
    let ticket = random_secret();
    let expires = Instant::now() + Duration::from_secs(30);
    let mut tickets = state
        .websocket_tickets
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    tickets.retain(|_, expiry| *expiry > Instant::now());
    tickets.insert(ticket.clone(), expires);
    Json(serde_json::json!({"ticket": ticket, "expires_in": 30, "single_use": true}))
        .into_response()
}

#[derive(serde::Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct CreateSessionRequest {
    id: Option<String>,
}

async fn sessions_create(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if !valid_management_auth(&state, &headers, true) {
        return auth_required();
    }
    if !exact_json_content_type(&headers) {
        return error_json(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "invalid_request_error",
            "Content-Type must be exactly application/json",
        );
    }
    let request: CreateSessionRequest = match strict_json(&body) {
        Ok(request) => request,
        Err(response) => return response,
    };
    state.server.record_request();
    match state.server.logical_sessions.create(request.id.as_deref()) {
        Ok(session) => (StatusCode::CREATED, Json(session)).into_response(),
        Err(error) if error.contains("already exists") || error.contains("limit") => {
            error_json(StatusCode::CONFLICT, "conflict", &error)
        }
        Err(error) => error_json(StatusCode::BAD_REQUEST, "invalid_request_error", &error),
    }
}

async fn sessions_list(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if !valid_management_auth(&state, &headers, false) {
        return auth_required();
    }
    state.server.record_request();
    match state.server.logical_sessions.list() {
        Ok(sessions) => {
            Json(serde_json::json!({"object": "list", "data": sessions})).into_response()
        }
        Err(error) => error_json(StatusCode::SERVICE_UNAVAILABLE, "session_error", &error),
    }
}

async fn session_get(
    State(state): State<AppState>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Response {
    if !valid_management_auth(&state, &headers, false) {
        return auth_required();
    }
    state.server.record_request();
    match state.server.logical_sessions.get(&id) {
        Ok(Some(session)) => Json(session).into_response(),
        Ok(None) => error_json(StatusCode::NOT_FOUND, "not_found", "session does not exist"),
        Err(error) => error_json(StatusCode::BAD_REQUEST, "invalid_request_error", &error),
    }
}

async fn session_delete(
    State(state): State<AppState>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Response {
    if !valid_management_auth(&state, &headers, true) {
        return auth_required();
    }
    state.server.record_request();
    match state.server.logical_sessions.delete(&id) {
        Ok(true) => StatusCode::NO_CONTENT.into_response(),
        Ok(false) => error_json(StatusCode::NOT_FOUND, "not_found", "session does not exist"),
        Err(error) if error.contains("busy") => {
            error_json(StatusCode::CONFLICT, "conflict", &error)
        }
        Err(error) => error_json(StatusCode::BAD_REQUEST, "invalid_request_error", &error),
    }
}

async fn session_save(
    State(state): State<AppState>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Response {
    if !valid_management_auth(&state, &headers, true) {
        return auth_required();
    }
    state.server.record_request();
    match state.server.logical_sessions.save(&id) {
        Ok(path) => {
            Json(serde_json::json!({"id": id, "saved": true, "path": path})).into_response()
        }
        Err(error) if error.contains("does not exist") => {
            error_json(StatusCode::NOT_FOUND, "not_found", &error)
        }
        Err(error) if error.contains("busy") || error.contains("no committed") => {
            error_json(StatusCode::CONFLICT, "conflict", &error)
        }
        Err(error) => error_json(StatusCode::INTERNAL_SERVER_ERROR, "session_error", &error),
    }
}

async fn session_restore(
    State(state): State<AppState>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Response {
    if !valid_management_auth(&state, &headers, true) {
        return auth_required();
    }
    state.server.record_request();
    match state.server.logical_sessions.restore(&id) {
        Ok(session) => Json(session).into_response(),
        Err(error) if error.contains("No such file") => {
            error_json(StatusCode::NOT_FOUND, "not_found", &error)
        }
        Err(error) if error.contains("limit") => {
            error_json(StatusCode::CONFLICT, "conflict", &error)
        }
        Err(error) => error_json(StatusCode::BAD_REQUEST, "session_restore_error", &error),
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct MigrateSessionRequest {
    destination: String,
    mode: String,
    tier: Option<String>,
    transfer_id: Option<String>,
}

async fn session_migrate(
    State(state): State<AppState>,
    Path(id): Path<String>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if !valid_management_auth(&state, &headers, true) {
        return auth_required();
    }
    if !exact_json_content_type(&headers) {
        return error_json(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "invalid_request_error",
            "Content-Type must be exactly application/json",
        );
    }
    let request: MigrateSessionRequest = match strict_json(&body) {
        Ok(request) => request,
        Err(response) => return response,
    };
    if !matches!(request.mode.as_str(), "copy" | "move") {
        return error_json(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "mode must be 'copy' or 'move'",
        );
    }
    let tier = request.tier.as_deref().unwrap_or("decode");
    if tier == "storage" {
        let transfer_id = match request.transfer_id {
            Some(value) => value,
            None if request.destination == "local" => {
                return error_json(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    "restoring from storage requires transfer_id",
                )
            }
            None => new_transfer_id(),
        };
        if request.destination == "local" {
            let initial = match state.server.logical_sessions.transfer(&transfer_id) {
                Ok(view)
                    if view.session_id == id
                        && view.tier == "storage"
                        && view.mode == request.mode =>
                {
                    view
                }
                Ok(_) => {
                    return error_json(
                        StatusCode::CONFLICT,
                        "conflict",
                        "transfer ID is bound to a different storage session",
                    )
                }
                Err(error) => return error_json(StatusCode::NOT_FOUND, "not_found", &error),
            };
            let server = Arc::clone(&state.server);
            let mode = request.mode;
            tokio::task::spawn_blocking(move || {
                if let Err(error) = run_storage_restore(&server, &transfer_id, &id, &mode) {
                    record_transfer_failure(&server, &transfer_id, error);
                }
            });
            return (StatusCode::ACCEPTED, Json(initial)).into_response();
        }
        if let Err(error) = crate::node::ssh::validate_name(&request.destination) {
            return error_json(StatusCode::BAD_REQUEST, "invalid_request_error", &error);
        }
        let initial = if state.server.logical_sessions.transfer(&transfer_id).is_ok() {
            match state.server.logical_sessions.transfer(&transfer_id) {
                Ok(view)
                    if view.session_id == id
                        && view.destination == request.destination
                        && view.mode == request.mode
                        && view.tier == "storage" =>
                {
                    view
                }
                Ok(_) => {
                    return error_json(
                        StatusCode::CONFLICT,
                        "conflict",
                        "transfer ID is already bound to a different migration",
                    )
                }
                Err(error) => return error_json(StatusCode::CONFLICT, "conflict", &error),
            }
        } else {
            match state.server.logical_sessions.register_outgoing(
                &transfer_id,
                &id,
                &request.destination,
                &request.mode,
                "storage",
            ) {
                Ok(view) => view,
                Err(error) if error.contains("busy") || error.contains("no committed") => {
                    return error_json(StatusCode::CONFLICT, "conflict", &error)
                }
                Err(error) if error.contains("does not exist") => {
                    return error_json(StatusCode::NOT_FOUND, "not_found", &error)
                }
                Err(error) => {
                    return error_json(StatusCode::BAD_REQUEST, "invalid_request_error", &error)
                }
            }
        };
        let server = Arc::clone(&state.server);
        let destination = request.destination;
        let mode = request.mode;
        tokio::task::spawn_blocking(move || {
            if let Err(error) =
                run_storage_transfer(&server, &transfer_id, &id, &destination, &mode)
            {
                record_transfer_failure(&server, &transfer_id, error);
            }
        });
        return (StatusCode::ACCEPTED, Json(initial)).into_response();
    }
    if tier != "decode" {
        return error_json(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "tier must be 'decode' or 'storage'",
        );
    }
    let destination = match validate_decode_destination(&request.destination) {
        Ok(value) => value,
        Err(error) => return error_json(StatusCode::BAD_REQUEST, "invalid_request_error", &error),
    };
    let Some(api_key) = state.api_key.as_ref() else {
        return error_json(
            StatusCode::CONFLICT,
            "migration_not_configured",
            "decode-node migration requires this server to have an API-key file",
        );
    };
    let api_key = match std::str::from_utf8(api_key) {
        Ok(value) => value.to_owned(),
        Err(_) => {
            return error_json(
                StatusCode::CONFLICT,
                "migration_not_configured",
                "the configured API key is not valid UTF-8 and cannot be forwarded",
            )
        }
    };
    let transfer_id = request.transfer_id.unwrap_or_else(new_transfer_id);
    let initial = if state.server.logical_sessions.transfer(&transfer_id).is_ok() {
        match state.server.logical_sessions.transfer(&transfer_id) {
            Ok(view)
                if view.session_id == id
                    && view.destination == destination
                    && view.mode == request.mode =>
            {
                view
            }
            Ok(_) => {
                return error_json(
                    StatusCode::CONFLICT,
                    "conflict",
                    "transfer ID is already bound to a different migration",
                )
            }
            Err(error) => return error_json(StatusCode::CONFLICT, "conflict", &error),
        }
    } else {
        match state.server.logical_sessions.register_outgoing(
            &transfer_id,
            &id,
            &destination,
            &request.mode,
            tier,
        ) {
            Ok(view) => view,
            Err(error) if error.contains("busy") || error.contains("no committed") => {
                return error_json(StatusCode::CONFLICT, "conflict", &error)
            }
            Err(error) if error.contains("does not exist") => {
                return error_json(StatusCode::NOT_FOUND, "not_found", &error)
            }
            Err(error) => {
                return error_json(StatusCode::BAD_REQUEST, "invalid_request_error", &error)
            }
        }
    };
    let server = Arc::clone(&state.server);
    let mode = request.mode;
    tokio::task::spawn_blocking(move || {
        if let Err(error) =
            run_decode_transfer(&server, &transfer_id, &id, &destination, &mode, &api_key)
        {
            record_transfer_failure(&server, &transfer_id, error);
        }
    });
    (StatusCode::ACCEPTED, Json(initial)).into_response()
}

fn record_transfer_failure(server: &ServerState, transfer_id: &str, error: String) {
    let (status, source_deleted) = match server.logical_sessions.transfer(transfer_id) {
        Ok(view) if view.status == "completed" => (view.status, view.source_deleted),
        Ok(view)
            if view.status.starts_with("destination_committed")
                || view.status.starts_with("source_restored") =>
        {
            (view.status, view.source_deleted)
        }
        Ok(view) => ("ambiguous".into(), view.source_deleted),
        Err(_) => ("ambiguous".into(), false),
    };
    let _ =
        server
            .logical_sessions
            .update_transfer(transfer_id, &status, Some(error), source_deleted);
}

async fn session_transfer_get(
    State(state): State<AppState>,
    Path(transfer_id): Path<String>,
    headers: HeaderMap,
) -> Response {
    if !valid_management_auth(&state, &headers, false) {
        return auth_required();
    }
    match state.server.logical_sessions.transfer(&transfer_id) {
        Ok(view) => Json(view).into_response(),
        Err(error) if error.contains("No such file") => error_json(
            StatusCode::NOT_FOUND,
            "not_found",
            "session transfer does not exist",
        ),
        Err(error) => error_json(StatusCode::BAD_REQUEST, "transfer_error", &error),
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct InternalTransferPrepare {
    transfer_id: String,
    session_id: String,
    source: String,
    mode: String,
    tier: String,
    bytes: u64,
    sha256: String,
    transport_key: String,
    model_sha256: String,
    tokenizer_sha256: String,
    template_sha256: String,
    layout_abi: String,
    dflash_identity_sha256: Option<String>,
    vision_projector_sha256: Option<String>,
    vision_preprocessing_sha256: Option<String>,
}

async fn session_transfer_prepare(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if !valid_bearer(&state, &headers) {
        return auth_required();
    }
    if !state.tls {
        return error_json(
            StatusCode::FORBIDDEN,
            "tls_required",
            "session transfer requires TLS",
        );
    }
    if !exact_json_content_type(&headers) {
        return error_json(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "invalid_request_error",
            "Content-Type must be exactly application/json",
        );
    }
    let request: InternalTransferPrepare = match strict_json(&body) {
        Ok(value) => value,
        Err(response) => return response,
    };
    let transport_key = match decode_fixed::<32>(&request.transport_key, "transport_key") {
        Ok(value) => value,
        Err(error) => return error_json(StatusCode::BAD_REQUEST, "invalid_request_error", &error),
    };
    let tokenizer = match decode_hex32(&request.tokenizer_sha256) {
        Ok(value) => value,
        Err(error) => return error_json(StatusCode::BAD_REQUEST, "invalid_request_error", &error),
    };
    let template = match decode_hex32(&request.template_sha256) {
        Ok(value) => value,
        Err(error) => return error_json(StatusCode::BAD_REQUEST, "invalid_request_error", &error),
    };
    let identity = match session_identity(&state.server) {
        Ok(value) => value,
        Err(error) => return error_json(StatusCode::SERVICE_UNAVAILABLE, "unavailable", &error),
    };
    if request.model_sha256 != identity.model
        || tokenizer != identity.tokenizer
        || template != identity.template
        || request.layout_abi != identity.layout
        || request
            .dflash_identity_sha256
            .as_ref()
            .is_some_and(|value| Some(value) != identity.dflash.as_ref())
        || request
            .vision_projector_sha256
            .as_ref()
            .is_some_and(|value| Some(value) != identity.vision_projector.as_ref())
        || request
            .vision_preprocessing_sha256
            .as_ref()
            .is_some_and(|value| Some(value) != identity.vision_preprocessing.as_ref())
    {
        return error_json(
            StatusCode::CONFLICT,
            "identity_mismatch",
            "source and destination model/template/layout/assistant/vision identities differ",
        );
    }
    match state.server.logical_sessions.prepare_import(
        &request.transfer_id,
        &request.session_id,
        &request.source,
        &request.mode,
        &request.tier,
        request.bytes,
        &request.sha256,
        transport_key,
        &request.model_sha256,
        tokenizer,
        template,
        &request.layout_abi,
        request.dflash_identity_sha256.as_deref(),
        request.vision_projector_sha256.as_deref(),
        request.vision_preprocessing_sha256.as_deref(),
    ) {
        Ok(view) => Json(view).into_response(),
        Err(error) if error.contains("already bound") => {
            error_json(StatusCode::CONFLICT, "conflict", &error)
        }
        Err(error) => error_json(StatusCode::BAD_REQUEST, "transfer_prepare_error", &error),
    }
}

async fn session_transfer_payload(
    State(state): State<AppState>,
    Path(transfer_id): Path<String>,
    headers: HeaderMap,
    request: Request,
) -> Response {
    if !valid_bearer(&state, &headers) {
        return auth_required();
    }
    if !state.tls {
        return error_json(
            StatusCode::FORBIDDEN,
            "tls_required",
            "session transfer requires TLS",
        );
    }
    if headers
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        != Some("application/octet-stream")
    {
        return error_json(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "invalid_request_error",
            "Content-Type must be exactly application/octet-stream",
        );
    }
    let (staging, expected) = match state.server.logical_sessions.payload_path(&transfer_id) {
        Ok(value) => value,
        Err(error) => return error_json(StatusCode::CONFLICT, "conflict", &error),
    };
    let mut options = tokio::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        options.mode(0o600);
    }
    let mut output = match options.open(&staging).await {
        Ok(value) => value,
        Err(error) => {
            return error_json(
                StatusCode::CONFLICT,
                "transfer_payload_error",
                &format!("create staging payload: {error}"),
            )
        }
    };
    let mut received = 0u64;
    let mut stream = request.into_body().into_data_stream();
    while let Some(next) = stream.next().await {
        let chunk = match next {
            Ok(value) => value,
            Err(error) => {
                let _ = tokio::fs::remove_file(&staging).await;
                return error_json(
                    StatusCode::BAD_REQUEST,
                    "transfer_payload_error",
                    &format!("read transfer body: {error}"),
                );
            }
        };
        received = match received.checked_add(chunk.len() as u64) {
            Some(value) if value <= expected => value,
            _ => {
                let _ = tokio::fs::remove_file(&staging).await;
                return error_json(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "transfer_payload_error",
                    "transfer body exceeds its prepared byte size",
                );
            }
        };
        if let Err(error) = output.write_all(&chunk).await {
            let _ = tokio::fs::remove_file(&staging).await;
            return error_json(
                StatusCode::INTERNAL_SERVER_ERROR,
                "transfer_payload_error",
                &format!("write transfer body: {error}"),
            );
        }
    }
    if received != expected {
        let _ = tokio::fs::remove_file(&staging).await;
        return error_json(
            StatusCode::BAD_REQUEST,
            "transfer_payload_error",
            "transfer body ended before its prepared byte size",
        );
    }
    if let Err(error) = output.sync_all().await {
        let _ = tokio::fs::remove_file(&staging).await;
        return error_json(
            StatusCode::INTERNAL_SERVER_ERROR,
            "transfer_payload_error",
            &format!("sync transfer body: {error}"),
        );
    }
    drop(output);
    let server = Arc::clone(&state.server);
    match tokio::task::spawn_blocking(move || {
        server
            .logical_sessions
            .accept_payload(&transfer_id, &staging)
    })
    .await
    {
        Ok(Ok(view)) => Json(view).into_response(),
        Ok(Err(error)) => error_json(StatusCode::BAD_REQUEST, "transfer_payload_error", &error),
        Err(error) => error_json(
            StatusCode::INTERNAL_SERVER_ERROR,
            "transfer_payload_error",
            &error.to_string(),
        ),
    }
}

async fn session_transfer_commit(
    State(state): State<AppState>,
    Path(transfer_id): Path<String>,
    headers: HeaderMap,
) -> Response {
    if !valid_bearer(&state, &headers) {
        return auth_required();
    }
    if !state.tls {
        return error_json(
            StatusCode::FORBIDDEN,
            "tls_required",
            "session transfer requires TLS",
        );
    }
    let identity = match session_identity(&state.server) {
        Ok(value) => value,
        Err(error) => return error_json(StatusCode::SERVICE_UNAVAILABLE, "unavailable", &error),
    };
    let server = Arc::clone(&state.server);
    match tokio::task::spawn_blocking(move || {
        server.logical_sessions.commit_import(
            &transfer_id,
            &identity.model,
            identity.tokenizer,
            identity.template,
            &identity.layout,
            identity.dflash.as_deref(),
            identity.vision_projector.as_deref(),
            identity.vision_preprocessing.as_deref(),
        )
    })
    .await
    {
        Ok(Ok(view)) => Json(view).into_response(),
        Ok(Err(error)) if error.contains("identity") => {
            error_json(StatusCode::CONFLICT, "identity_mismatch", &error)
        }
        Ok(Err(error)) => error_json(StatusCode::BAD_REQUEST, "transfer_commit_error", &error),
        Err(error) => error_json(
            StatusCode::INTERNAL_SERVER_ERROR,
            "transfer_commit_error",
            &error.to_string(),
        ),
    }
}

fn new_transfer_id() -> String {
    use rand::RngCore as _;
    let mut bytes = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    format!(
        "xfer-{}",
        bytes
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    )
}

fn validate_decode_destination(value: &str) -> Result<String, String> {
    let uri: axum::http::Uri = value
        .parse()
        .map_err(|_| "destination must be an absolute HTTPS origin")?;
    if uri.scheme_str() != Some("https")
        || uri.authority().is_none()
        || (uri.path() != "/" && !uri.path().is_empty())
        || uri.query().is_some()
    {
        return Err("destination must be an HTTPS origin without path, query, or fragment".into());
    }
    Ok(value.trim_end_matches('/').to_owned())
}

fn decode_fixed<const N: usize>(value: &str, label: &str) -> Result<[u8; N], String> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(value)
        .map_err(|_| format!("{label} is not valid base64"))?;
    bytes
        .try_into()
        .map_err(|_| format!("{label} must decode to exactly {N} bytes"))
}

fn decode_hex32(value: &str) -> Result<[u8; 32], String> {
    if value.len() != 64 {
        return Err("identity digest must contain 64 lowercase hexadecimal characters".into());
    }
    let mut result = [0u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let text = std::str::from_utf8(pair).expect("ASCII-sized chunks");
        result[index] = u8::from_str_radix(text, 16)
            .map_err(|_| "identity digest is not lowercase hexadecimal".to_string())?;
        if pair.iter().any(|byte| byte.is_ascii_uppercase()) {
            return Err("identity digest is not lowercase hexadecimal".into());
        }
    }
    Ok(result)
}

fn encode_hex32(value: &[u8; 32]) -> String {
    value.iter().map(|byte| format!("{byte:02x}")).collect()
}

struct SessionIdentity {
    model: String,
    tokenizer: [u8; 32],
    template: [u8; 32],
    layout: String,
    dflash: Option<String>,
    vision_projector: Option<String>,
    vision_preprocessing: Option<String>,
}

fn session_identity(server: &ServerState) -> Result<SessionIdentity, String> {
    let runtime = server
        .inference
        .as_ref()
        .ok_or("inference is unavailable")?;
    let model = server
        .model_sha256
        .clone()
        .filter(|value| !value.is_empty())
        .ok_or("verified model identity is unavailable")?;
    Ok(SessionIdentity {
        model,
        tokenizer: runtime.model.tokenizer_metadata_sha256(),
        template: runtime.model.chat_template_sha256(),
        layout: "muse-kv-layout-v1".into(),
        dflash: runtime.dflash_identity_sha256.clone(),
        vision_projector: runtime
            .vision_identity
            .as_ref()
            .map(|identity| identity.projector_sha256.clone()),
        vision_preprocessing: runtime
            .vision_identity
            .as_ref()
            .map(|identity| identity.preprocessing_sha256.clone()),
    })
}

fn transfer_agent() -> Result<ureq::Agent, String> {
    use ureq::tls::{Certificate, RootCerts, TlsConfig};
    let root_certs = if let Some(path) = std::env::var_os("MUSER_DECODE_MIGRATION_CA") {
        let bytes = std::fs::read(&path).map_err(|error| {
            format!(
                "read MUSER_DECODE_MIGRATION_CA {}: {error}",
                std::path::Path::new(&path).display()
            )
        })?;
        let certificate = Certificate::from_pem(&bytes)
            .map_err(|error| format!("parse MUSER_DECODE_MIGRATION_CA: {error}"))?;
        RootCerts::new_with_certs(&[certificate])
    } else {
        RootCerts::PlatformVerifier
    };
    Ok(ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(30)))
        .timeout_send_request(Some(Duration::from_secs(30)))
        .timeout_send_body(Some(Duration::from_secs(60 * 60)))
        .timeout_recv_response(Some(Duration::from_secs(60)))
        .timeout_recv_body(Some(Duration::from_secs(60)))
        .tls_config(TlsConfig::builder().root_certs(root_certs).build())
        .build()
        .new_agent())
}

fn read_transfer_response(
    response: &mut ureq::http::Response<ureq::Body>,
) -> Result<crate::session_store::TransferView, String> {
    response
        .body_mut()
        .read_json()
        .map_err(|error| format!("decode destination transfer response: {error}"))
}

fn remote_transfer_status(
    agent: &ureq::Agent,
    destination: &str,
    transfer_id: &str,
    bearer: &str,
) -> Result<crate::session_store::TransferView, String> {
    let mut response = agent
        .get(format!("{destination}/v1/session-transfers/{transfer_id}"))
        .header("Authorization", bearer)
        .call()
        .map_err(|error| format!("reconcile destination transfer: {error}"))?;
    read_transfer_response(&mut response)
}

fn run_decode_transfer(
    server: &ServerState,
    transfer_id: &str,
    session_id: &str,
    destination: &str,
    mode: &str,
    api_key: &str,
) -> Result<(), String> {
    if server
        .logical_sessions
        .reconcile_outgoing_after_ack(transfer_id)?
    {
        return Ok(());
    }
    let current = server.logical_sessions.transfer(transfer_id)?;
    let export = match server.logical_sessions.resume_export(transfer_id) {
        Ok(value) => value,
        Err(_) if current.status == "starting" => server.logical_sessions.begin_export(
            transfer_id,
            session_id,
            destination,
            mode,
            "decode",
        )?,
        Err(error) => {
            return Err(format!(
                "durable transfer material cannot be resumed and will not be replaced: {error}"
            ))
        }
    };
    server
        .logical_sessions
        .update_transfer(transfer_id, "transferring", None, false)?;
    let bearer = format!("Bearer {api_key}");
    let prepare = InternalTransferPrepare {
        transfer_id: transfer_id.into(),
        session_id: session_id.into(),
        source: "muser-decode-source".into(),
        mode: mode.into(),
        tier: "decode".into(),
        bytes: export.view.bytes,
        sha256: export.view.sha256.clone(),
        transport_key: base64::engine::general_purpose::STANDARD.encode(export.transport_key),
        model_sha256: export.model_sha256,
        tokenizer_sha256: encode_hex32(&export.tokenizer_sha256),
        template_sha256: encode_hex32(&export.template_sha256),
        layout_abi: export.layout_abi,
        dflash_identity_sha256: export.dflash_identity_sha256,
        vision_projector_sha256: export.vision_projector_sha256,
        vision_preprocessing_sha256: export.vision_preprocessing_sha256,
    };
    let agent = transfer_agent()?;
    let mut prepare_response = agent
        .post(format!(
            "{destination}/__muser/v1/session-transfers/prepare"
        ))
        .header("Authorization", &bearer)
        .send_json(&prepare)
        .map_err(|error| format!("prepare destination transfer: {error}"))?;
    let prepared = read_transfer_response(&mut prepare_response)?;
    let committed = if prepared.status == "committed" {
        true
    } else {
        let upload_result = (|| {
            let file = std::fs::File::open(&export.payload)
                .map_err(|error| format!("open transfer payload: {error}"))?;
            agent
                .put(format!(
                    "{destination}/__muser/v1/session-transfers/{transfer_id}/payload"
                ))
                .header("Authorization", &bearer)
                .header("Content-Type", "application/octet-stream")
                .send(file)
                .map_err(|error| format!("upload destination transfer: {error}"))?;
            let mut response = agent
                .post(format!(
                    "{destination}/__muser/v1/session-transfers/{transfer_id}/commit"
                ))
                .header("Authorization", &bearer)
                .send_empty()
                .map_err(|error| format!("commit destination transfer: {error}"))?;
            let committed = read_transfer_response(&mut response)?;
            Ok::<bool, String>(committed.status == "committed")
        })();
        match upload_result {
            Ok(value) => value,
            Err(original) => {
                match remote_transfer_status(&agent, destination, transfer_id, &bearer) {
                    Ok(status) if status.status == "committed" => true,
                    Ok(status) => {
                        return Err(format!(
                            "{original}; destination reconciled as {}",
                            status.status
                        ))
                    }
                    Err(reconcile) => return Err(format!("{original}; {reconcile}")),
                }
            }
        }
    };
    if !committed {
        return Err("destination did not durably commit the transfer".into());
    }
    server
        .logical_sessions
        .update_transfer(transfer_id, "destination_committed", None, false)?;
    server
        .logical_sessions
        .reconcile_outgoing_after_ack(transfer_id)?;
    Ok(())
}

const STORAGE_PREPARE: &str = r#"set -eu
directory=$1
mkdir -p "$directory"
chmod 700 "$directory"
sync "$directory"
"#;

const STORAGE_COMMIT: &str = r#"set -eu
temporary=$1
final=$2
bytes=$3
digest=$4
directory=$5
if [ -f "$final" ]; then
    test "$(wc -c < "$final" | tr -d ' ')" = "$bytes"
    test "$(sha256sum "$final" | awk '{print $1}')" = "$digest"
    exit 0
fi
test -f "$temporary"
test "$(wc -c < "$temporary" | tr -d ' ')" = "$bytes"
test "$(sha256sum "$temporary" | awk '{print $1}')" = "$digest"
chmod 600 "$temporary"
mv "$temporary" "$final"
sync "$final"
sync "$directory"
"#;

const STORAGE_DELETE: &str = r#"set -eu
payload=$1
directory=$2
rm -f "$payload"
sync "$directory"
"#;

fn enrolled_storage_node(name: &str) -> Result<(crate::node::ssh::Ssh, String), String> {
    let home = crate::node::muser_home()?;
    let registry = crate::node::registry::Registry::load(&home)?;
    let entry = registry
        .get(name)
        .ok_or_else(|| format!("storage node {name:?} is not enrolled"))?;
    if entry.state != crate::node::registry::STATE_HEALTHY
        || entry.enrollment_version < 2
        || entry.hmac_epoch <= 0
    {
        return Err(format!(
            "storage node {name:?} is not healthy under enrollment v2"
        ));
    }
    let key = entry.key_path.as_deref().map(FsPath::new);
    let ssh = crate::node::ssh::Ssh::new(&entry.user, &entry.host, key)?;
    let directory = format!("{}/session-bundles", entry.lane_dir.trim_end_matches('/'));
    crate::node::ssh::validate_remote_path(&directory)?;
    Ok((ssh, directory))
}

fn storage_paths(directory: &str, transfer_id: &str) -> (String, String) {
    (
        format!("{directory}/.{transfer_id}.tmp"),
        format!("{directory}/{transfer_id}.bundle"),
    )
}

fn run_storage_transfer(
    server: &ServerState,
    transfer_id: &str,
    session_id: &str,
    destination: &str,
    mode: &str,
) -> Result<(), String> {
    if server
        .logical_sessions
        .reconcile_outgoing_after_ack(transfer_id)?
    {
        if server
            .logical_sessions
            .transfer(transfer_id)?
            .source_deleted
        {
            server
                .logical_sessions
                .remove_transfer_payload(transfer_id)?;
        }
        return Ok(());
    }
    let current = server.logical_sessions.transfer(transfer_id)?;
    let export = match server.logical_sessions.resume_export(transfer_id) {
        Ok(value) => value,
        Err(_) if current.status == "starting" => server.logical_sessions.begin_export(
            transfer_id,
            session_id,
            destination,
            mode,
            "storage",
        )?,
        Err(error) => {
            return Err(format!(
                "durable transfer material cannot be resumed and will not be replaced: {error}"
            ))
        }
    };
    server
        .logical_sessions
        .update_transfer(transfer_id, "transferring", None, false)?;
    let (ssh, directory) = enrolled_storage_node(destination)?;
    ssh.run(STORAGE_PREPARE, &[&directory])?;
    let (temporary, final_path) = storage_paths(&directory, transfer_id);
    ssh.scp(&export.payload, &temporary)?;
    let bytes = export.view.bytes.to_string();
    ssh.run(
        STORAGE_COMMIT,
        &[
            &temporary,
            &final_path,
            &bytes,
            &export.view.sha256,
            &directory,
        ],
    )?;
    server
        .logical_sessions
        .update_transfer(transfer_id, "destination_committed", None, false)?;
    server
        .logical_sessions
        .reconcile_outgoing_after_ack(transfer_id)?;
    if server
        .logical_sessions
        .transfer(transfer_id)?
        .source_deleted
    {
        server
            .logical_sessions
            .remove_transfer_payload(transfer_id)?;
    }
    Ok(())
}

fn run_storage_restore(
    server: &ServerState,
    transfer_id: &str,
    session_id: &str,
    mode: &str,
) -> Result<(), String> {
    let view = server.logical_sessions.transfer(transfer_id)?;
    if view.session_id != session_id || view.tier != "storage" || view.mode != mode {
        return Err("storage transfer does not match the requested session".into());
    }
    if view.status == "completed" {
        return Ok(());
    }
    let (ssh, directory) = enrolled_storage_node(&view.destination)?;
    let (_, remote_payload) = storage_paths(&directory, transfer_id);
    if matches!(
        view.status.as_str(),
        "source_restored" | "source_restored_remote_retained"
    ) {
        if mode == "move" {
            match ssh.run(STORAGE_DELETE, &[&remote_payload, &directory]) {
                Ok(_) => {
                    server.logical_sessions.update_transfer(
                        transfer_id,
                        "completed",
                        None,
                        false,
                    )?;
                }
                Err(error) => {
                    server.logical_sessions.update_transfer(
                        transfer_id,
                        "source_restored_remote_retained",
                        Some(error),
                        false,
                    )?;
                }
            }
        } else {
            server
                .logical_sessions
                .update_transfer(transfer_id, "completed", None, false)?;
        }
        return Ok(());
    }
    let final_payload = server.logical_sessions.transfer_payload_path(transfer_id)?;
    let staging = final_payload.with_extension(format!("restore-{}", new_transfer_id()));
    ssh.scp_from(&remote_payload, &staging)?;
    server
        .logical_sessions
        .adopt_export_payload(transfer_id, &staging)?;
    server.logical_sessions.restore_export(transfer_id)?;
    server
        .logical_sessions
        .update_transfer(transfer_id, "source_restored", None, false)?;
    if mode == "move" {
        match ssh.run(STORAGE_DELETE, &[&remote_payload, &directory]) {
            Ok(_) => {
                server
                    .logical_sessions
                    .update_transfer(transfer_id, "completed", None, false)?;
            }
            Err(error) => {
                server.logical_sessions.update_transfer(
                    transfer_id,
                    "source_restored_remote_retained",
                    Some(error),
                    false,
                )?;
            }
        }
    } else {
        server
            .logical_sessions
            .update_transfer(transfer_id, "completed", None, false)?;
    }
    Ok(())
}

#[derive(serde::Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct StreamQuery {
    ticket: Option<String>,
}

async fn websocket_stream(
    websocket: WebSocketUpgrade,
    State(state): State<AppState>,
    Query(query): Query<StreamQuery>,
    headers: HeaderMap,
) -> Response {
    // Browser WebSockets are not protected by fetch's same-origin response
    // rules. A dashboard cookie therefore requires an explicit Origin match;
    // long-lived bearer auth must first be exchanged for a single-use ticket.
    let cookie_authenticated = valid_dashboard_cookie(&state, &headers, true, false);
    let ticket_authenticated = query
        .ticket
        .as_deref()
        .is_some_and(|candidate| consume_ticket(&state, candidate));
    if !cookie_authenticated && !ticket_authenticated {
        return auth_required();
    }
    websocket.on_upgrade(move |socket| websocket_telemetry(socket, state.server))
}

fn consume_ticket(state: &AppState, candidate: &str) -> bool {
    let now = Instant::now();
    let mut tickets = state
        .websocket_tickets
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    tickets.retain(|_, expiry| *expiry > now);
    let Some((stored, _)) = tickets
        .iter()
        .find(|(ticket, _)| constant_time_equal(ticket.as_bytes(), candidate.as_bytes()))
        .map(|(ticket, expiry)| (ticket.clone(), *expiry))
    else {
        return false;
    };
    tickets.remove(&stored);
    true
}

async fn websocket_telemetry(mut socket: WebSocket, server: Arc<ServerState>) {
    server.telemetry_viewers.fetch_add(1, Ordering::Relaxed);
    struct Viewer(Arc<ServerState>);
    impl Drop for Viewer {
        fn drop(&mut self) {
            self.0.telemetry_viewers.fetch_sub(1, Ordering::Relaxed);
        }
    }
    let _viewer = Viewer(Arc::clone(&server));
    if ws_send_json(
        &mut socket,
        &serde_json::json!({
            "v": 2, "type": "hello", "schema": "muser.telemetry.v2",
            "snapshot_interval_s": 10, "ping_interval_s": 5
        }),
    )
    .await
    .is_err()
    {
        return;
    }

    let mut sequence = 0u64;
    let mut previous: Option<serde_json::Map<String, serde_json::Value>> = None;
    let mut tick = tokio::time::interval(Duration::from_secs(1));
    loop {
        tokio::select! {
            incoming = socket.recv() => match incoming {
                Some(Ok(WebSocketMessage::Close(_))) | None | Some(Err(_)) => break,
                Some(Ok(_)) => {}
            },
            _ = tick.tick() => {
                let snapshot = serde_json::to_value(metrics::build_snapshot(&server)).unwrap_or(serde_json::Value::Null);
                let object = snapshot.as_object().cloned().unwrap_or_default();
                let frame = if sequence.is_multiple_of(10) || previous.is_none() {
                    serde_json::json!({"v":2,"type":"snapshot","seq":sequence,"data":snapshot})
                } else {
                    let changed = object.iter().filter(|(key, value)| previous.as_ref().and_then(|old| old.get(*key)) != Some(*value)).map(|(key, value)| (key.clone(), value.clone())).collect::<serde_json::Map<_,_>>();
                    serde_json::json!({"v":2,"type":"section_delta","seq":sequence,"data":changed})
                };
                previous = Some(object);
                if ws_send_json(&mut socket, &frame).await.is_err() { break; }
                if sequence > 0 && sequence.is_multiple_of(5) && socket.send(WebSocketMessage::Ping(Bytes::from_static(b"muser"))).await.is_err() { break; }
                sequence += 1;
            }
        }
    }
}

async fn ws_send_json(
    socket: &mut WebSocket,
    value: &serde_json::Value,
) -> Result<(), axum::Error> {
    socket
        .send(WebSocketMessage::Text(value.to_string().into()))
        .await
}

async fn benchmark_shutdown(State(state): State<AppState>, body: Bytes) -> Response {
    let Some(control) = &state.benchmark else {
        return StatusCode::NOT_FOUND.into_response();
    };
    if body.as_ref() != control.shutdown_token {
        return error_json(
            StatusCode::FORBIDDEN,
            "forbidden",
            "invalid benchmark token",
        );
    }
    control.stop.store(true, Ordering::Release);
    Json(serde_json::json!({"ok": true})).into_response()
}

fn exact_json_content_type(headers: &HeaderMap) -> bool {
    headers
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        == Some("application/json")
}

fn valid_bearer(state: &AppState, headers: &HeaderMap) -> bool {
    let Some(expected) = state.api_key.as_deref() else {
        return false;
    };
    let Some(candidate) = headers
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
    else {
        return false;
    };
    constant_time_equal(candidate.as_bytes(), expected)
}

fn valid_management_auth(state: &AppState, headers: &HeaderMap, mutation: bool) -> bool {
    if valid_bearer(state, headers) {
        return true;
    }
    valid_dashboard_cookie(state, headers, mutation, mutation)
}

fn request_origin(state: &AppState, headers: &HeaderMap) -> Option<String> {
    let host = single_header(headers, HOST.as_str())?;
    let authority: axum::http::uri::Authority = host.parse().ok()?;
    if authority.as_str() != host {
        return None;
    }
    Some(format!(
        "{}://{}",
        if state.tls { "https" } else { "http" },
        authority
    ))
}

/// Authenticate the dashboard cookie against the server authority it was
/// minted for. Read-only management routes bind the cookie to Host; browser
/// WebSockets and all mutations additionally require an explicit Origin.
fn valid_dashboard_cookie(
    state: &AppState,
    headers: &HeaderMap,
    require_origin: bool,
    require_csrf: bool,
) -> bool {
    let Some(session_id) = cookie_value(headers, "muser_session") else {
        return false;
    };
    let Some(actual_origin) = request_origin(state, headers) else {
        return false;
    };
    let now = Instant::now();
    let mut sessions = state
        .dashboard_sessions
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    sessions.retain(|_, session| session.expires > now);
    let Some(session) = sessions.get(session_id) else {
        return false;
    };
    if session.origin != actual_origin {
        return false;
    }
    if require_origin && single_header(headers, ORIGIN.as_str()) != Some(session.origin.as_str()) {
        return false;
    }
    if !require_csrf {
        return true;
    }
    headers
        .get("x-csrf-token")
        .and_then(|value| value.to_str().ok())
        .is_some_and(|csrf| constant_time_equal(csrf.as_bytes(), session.csrf.as_bytes()))
}

fn single_header<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    let mut values = headers.get_all(name).iter();
    let value = values.next()?.to_str().ok()?;
    values.next().is_none().then_some(value)
}

fn cookie_value<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers
        .get("cookie")?
        .to_str()
        .ok()?
        .split(';')
        .find_map(|part| {
            let (key, value) = part.trim().split_once('=')?;
            (key == name).then_some(value)
        })
}

fn random_secret() -> String {
    use base64::Engine as _;
    use rand::RngCore as _;
    let mut bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

fn constant_time_equal(left: &[u8], right: &[u8]) -> bool {
    let mut difference = left.len() ^ right.len();
    for index in 0..left.len().max(right.len()) {
        difference |= usize::from(
            left.get(index).copied().unwrap_or(0) ^ right.get(index).copied().unwrap_or(0),
        );
    }
    difference == 0
}

fn auth_required() -> Response {
    let mut response = error_json(
        StatusCode::UNAUTHORIZED,
        "authentication_required",
        "a valid bearer API key is required",
    );
    response
        .headers_mut()
        .insert("www-authenticate", HeaderValue::from_static("Bearer"));
    response
}

fn sse_json(value: &impl serde::Serialize) -> String {
    format!(
        "data: {}\n\n",
        serde_json::to_string(value).unwrap_or_else(|error| format!("{{\"error\":\"{error}\"}}"))
    )
}

fn sse_response(body: Body) -> Response {
    let mut response = body.into_response();
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    response
        .headers_mut()
        .insert(CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    response
}

fn chat_error(error: openai::ChatError) -> Response {
    let (status, _, kind) = error.status();
    error_json(
        StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
        kind,
        &error.to_string(),
    )
}

fn error_json(status: StatusCode, kind: &str, message: &str) -> Response {
    (
        status,
        Json(serde_json::json!({"error": {"type": kind, "message": message}})),
    )
        .into_response()
}

fn reply_response(reply: nodes_api::Reply) -> Response {
    let status = StatusCode::from_u16(reply.code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let value = serde_json::from_str(&reply.body)
        .unwrap_or_else(|_| serde_json::json!({"error": reply.body}));
    (status, Json(value)).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn security_test_state(tls: bool) -> AppState {
        AppState {
            server: Arc::new(ServerState::new(None)),
            benchmark: None,
            api_key: None,
            lan: true,
            tls,
            dashboard_sessions: Arc::new(std::sync::Mutex::new(HashMap::new())),
            websocket_tickets: Arc::new(std::sync::Mutex::new(HashMap::new())),
            streams: StreamManager::default(),
            reasoning_controls: Arc::new(std::sync::Mutex::new(HashMap::new())),
        }
    }

    #[test]
    fn containment_rejects_nonloopback_without_tls_configuration() {
        assert!(validate_bind_host("127.0.0.1", 4949).is_ok());
        assert!(validate_bind_host("0.0.0.0", 4949).is_err());
    }

    #[test]
    fn content_type_is_exact_and_cors_is_absent_by_construction() {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        assert!(exact_json_content_type(&headers));
        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/json; charset=utf-8"),
        );
        assert!(!exact_json_content_type(&headers));
    }

    #[test]
    fn dashboard_chat_client_is_same_origin_and_textcontent_only() {
        // The chat pane is the one dashboard surface that drives an inference
        // route rather than a management route; pin it to the same-origin
        // streaming endpoint and to textContent token insertion.
        assert!(DASHBOARD_HTML.contains("ENDPOINT+\"/v1/chat/completions\""));
        assert!(DASHBOARD_HTML.contains("data===\"[DONE]\""));
        assert!(DASHBOARD_HTML.contains(".textContent+=d.content"));
        assert!(!DASHBOARD_HTML.contains("innerHTML+=d."));
    }

    #[test]
    fn dashboard_origin_is_exactly_the_tls_request_authority() {
        let state = security_test_state(true);
        let mut headers = HeaderMap::new();
        headers.insert(HOST, HeaderValue::from_static("muser.local:4949"));
        headers.insert(ORIGIN, HeaderValue::from_static("https://muser.local:4949"));
        assert_eq!(
            request_origin(&state, &headers).as_deref(),
            Some("https://muser.local:4949")
        );
        assert_eq!(
            headers.get(ORIGIN).and_then(|value| value.to_str().ok()),
            request_origin(&state, &headers).as_deref()
        );

        headers.insert(ORIGIN, HeaderValue::from_static("https://evil.local"));
        assert_ne!(
            headers.get(ORIGIN).and_then(|value| value.to_str().ok()),
            request_origin(&state, &headers).as_deref()
        );
        headers.append(HOST, HeaderValue::from_static("second.local"));
        assert!(request_origin(&state, &headers).is_none());
    }

    #[test]
    fn http2_authority_is_normalized_and_conflicts_fail_closed() {
        let mut request = Request::builder()
            .uri("https://muser.local:4949/snapshot")
            .body(Body::empty())
            .unwrap();
        normalize_authority(&mut request).unwrap();
        assert_eq!(
            request
                .headers()
                .get(HOST)
                .and_then(|value| value.to_str().ok()),
            Some("muser.local:4949")
        );

        let mut conflict = Request::builder()
            .uri("https://muser.local:4949/snapshot")
            .header(HOST, "other.local:4949")
            .body(Body::empty())
            .unwrap();
        assert_eq!(
            normalize_authority(&mut conflict),
            Err("request Host and HTTP/2 authority must match exactly")
        );
    }

    #[test]
    fn dashboard_cookie_reads_bind_host_and_websockets_bind_origin() {
        let state = security_test_state(true);
        state.dashboard_sessions.lock().unwrap().insert(
            "session".into(),
            DashboardSession {
                csrf: "csrf".into(),
                origin: "https://muser.local:4949".into(),
                expires: Instant::now() + Duration::from_secs(60),
            },
        );
        let mut headers = HeaderMap::new();
        headers.insert(HOST, HeaderValue::from_static("muser.local:4949"));
        headers.insert("cookie", HeaderValue::from_static("muser_session=session"));

        assert!(valid_management_auth(&state, &headers, false));
        assert!(!valid_dashboard_cookie(&state, &headers, true, false));
        headers.insert(ORIGIN, HeaderValue::from_static("https://muser.local:4949"));
        assert!(valid_dashboard_cookie(&state, &headers, true, false));
        assert!(!valid_management_auth(&state, &headers, true));
        headers.insert("x-csrf-token", HeaderValue::from_static("csrf"));
        assert!(valid_management_auth(&state, &headers, true));

        headers.insert(HOST, HeaderValue::from_static("other.local:4949"));
        assert!(!valid_management_auth(&state, &headers, false));
        assert!(!valid_dashboard_cookie(&state, &headers, true, false));
    }

    #[test]
    fn dropped_nonstream_handler_signals_its_blocking_worker() {
        let signal = Arc::new(AtomicBool::new(false));
        {
            let _guard = DisconnectCancellation::new(Arc::clone(&signal));
            assert!(!signal.load(Ordering::Acquire));
        }
        assert!(signal.load(Ordering::Acquire));

        let signal = Arc::new(AtomicBool::new(false));
        {
            let mut guard = DisconnectCancellation::new(Arc::clone(&signal));
            guard.disarm();
        }
        assert!(!signal.load(Ordering::Acquire));
    }

    #[test]
    fn final_connection_service_lifetime_signals_all_requests() {
        let first = Arc::new(AtomicBool::new(false));
        let second = Arc::new(AtomicBool::new(false));
        let requests = Arc::new(std::sync::Mutex::new(vec![
            Arc::downgrade(&first),
            Arc::downgrade(&second),
        ]));
        let lifetime = Arc::new(ConnectionLifetime(requests));
        let clone = Arc::clone(&lifetime);
        drop(clone);
        assert!(!first.load(Ordering::Acquire));
        assert!(!second.load(Ordering::Acquire));
        drop(lifetime);
        assert!(first.load(Ordering::Acquire));
        assert!(second.load(Ordering::Acquire));

        let first = Arc::new(AtomicBool::new(false));
        let second = Arc::new(AtomicBool::new(false));
        drop(DisconnectCancellation::new(Arc::clone(&first)));
        assert!(first.load(Ordering::Acquire));
        assert!(!second.load(Ordering::Acquire));
    }

    #[test]
    fn completion_dto_accepts_full_sampler_surface_and_rejects_nested_typos() {
        let request: CompletionRequest = serde_json::from_value(serde_json::json!({
            "model":"muse-glimmer-30b","prompt":[1,2,3],"n":4,
            "temperature":0.7,"top_p":0.9,"top_k":40,"typical_p":0.95,
            "min_p":0.05,"top_n_sigma":2.0,"repeat_penalty":1.1,
            "presence_penalty":0.2,"frequency_penalty":0.1,
            "dry_multiplier":0.8,"dry_base":1.75,"dry_allowed_length":2,
            "mirostat":2,"mirostat_tau":5.0,"mirostat_eta":0.1,
            "dynatemp_range":0.2,"dynatemp_exponent":1.0,
            "xtc_probability":0.1,"xtc_threshold":0.5,
            "logit_bias":[[42, 1.5], ["Muse", false]],
            "grammar":"root ::= \"ok\""
        }))
        .unwrap();
        assert_eq!(request.n, Some(4));
        assert_eq!(request.logit_bias.as_ref().unwrap()["42"], 1.5);
        assert_eq!(
            request.logit_bias.as_ref().unwrap()["Muse"],
            f32::NEG_INFINITY
        );
        assert!(
            serde_json::from_value::<OllamaGenerateRequest>(serde_json::json!({
                "model":"muse-glimmer-30b","prompt":"hi",
                "options":{"temperature":0.5,"typo":true}
            }))
            .is_err()
        );
    }

    #[test]
    fn ollama_unsupported_fields_are_rejected_even_when_json_null() {
        let clean: OllamaGenerateRequest = serde_json::from_value(serde_json::json!({
            "model":"muse-glimmer-30b", "prompt":"hi"
        }))
        .unwrap();
        assert_eq!(unsupported_ollama_field(&clean), None);

        for (field, value) in [
            ("suffix", serde_json::Value::Null),
            ("keep_alive", serde_json::json!(null)),
            ("template", serde_json::json!(null)),
        ] {
            let mut body = serde_json::json!({
                "model":"muse-glimmer-30b", "prompt":"hi"
            });
            body.as_object_mut().unwrap().insert(field.into(), value);
            let request: OllamaGenerateRequest = serde_json::from_value(body).unwrap();
            assert_eq!(unsupported_ollama_field(&request), Some(field));
        }
    }

    #[test]
    fn slot_file_envelope_rejects_unknown_top_level_fields() {
        let envelope = SlotCompatEnvelope {
            schema: "muser.slot-file.v1".into(),
            model_sha256: "00".repeat(32),
            tokenizer_sha256: [1; 32],
            template_sha256: [2; 32],
            layout_abi: "muse-kv-layout-v1".into(),
            slot: crate::state::SlotSnapshot {
                schema: "muser.slot-snapshot.v1".into(),
                target: muser_engine::cache::SessionCacheSnapshot {
                    position: 0,
                    tokens: Arc::from([]),
                    elements_per_token: 1,
                    layers: Arc::from([]),
                },
                logits: vec![0.0],
            },
        };
        let original = serde_json::to_value(envelope).unwrap();
        let mut value = original.clone();
        value
            .as_object_mut()
            .unwrap()
            .insert("future_layout".into(), serde_json::json!(true));
        assert!(serde_json::from_value::<SlotCompatEnvelope>(value).is_err());

        let mut value = original;
        value["slot"]
            .as_object_mut()
            .unwrap()
            .insert("future_slot_state".into(), serde_json::json!(true));
        assert!(serde_json::from_value::<SlotCompatEnvelope>(value).is_err());
    }

    #[test]
    fn native_response_fields_match_pinned_nested_selection() {
        let value = serde_json::json!({
            "content": "ok",
            "generation_settings": {"n_predict": 8, "seed": 7},
            "timings": {"predicted_n": 8}
        });
        let paths = vec![
            "content".to_string(),
            "generation_settings/n_predict".to_string(),
            "missing/value".to_string(),
        ];
        assert_eq!(
            select_response_fields(value, &paths),
            serde_json::json!({
                "content": "ok",
                "generation_settings/n_predict": 8
            })
        );
    }

    #[test]
    fn ollama_output_never_exposes_atem_transport_markers() {
        let (response, thinking) =
            ollama_visible_output(" to=self<|message|>plan<|eom|> to=user<|message|>answer")
                .unwrap();
        assert_eq!(thinking, "plan");
        assert_eq!(response, "answer");
    }

    #[test]
    fn decode_migration_destination_is_an_https_origin_only() {
        assert_eq!(
            validate_decode_destination("https://decode.example:4949/").unwrap(),
            "https://decode.example:4949"
        );
        assert!(validate_decode_destination("http://decode.example").is_err());
        assert!(validate_decode_destination("https://decode.example/path").is_err());
        assert!(validate_decode_destination("https://decode.example/?key=secret").is_err());
    }

    #[test]
    fn canonical_request_identity_ignores_json_whitespace_and_key_order() {
        let left = canonical_json_sha256(br#"{"messages":[],"model":"m"}"#).unwrap();
        let right = canonical_json_sha256(br#" { "model": "m", "messages": [] } "#).unwrap();
        assert_eq!(left, right);
        assert_ne!(
            left,
            canonical_json_sha256(br#"{"messages":[1],"model":"m"}"#).unwrap()
        );
    }

    #[test]
    fn resumable_stream_detaches_only_on_closed_socket_not_backpressure() {
        let (sender, receiver) = mpsc::channel::<Result<Bytes, Infallible>>(1);
        sender
            .try_send(Ok(Bytes::from_static(b"occupied")))
            .unwrap();
        let live = Cell::new(true);
        assert!(matches!(
            send_resumable_bounded(
                &sender,
                Bytes::from_static(b"blocked"),
                true,
                &live,
                Duration::ZERO,
            ),
            Err(openai::ChatError::Cancelled)
        ));
        assert!(live.get());

        drop(receiver);
        let live = Cell::new(true);
        assert!(send_resumable_bounded(
            &sender,
            Bytes::from_static(b"detached"),
            true,
            &live,
            Duration::ZERO,
        )
        .is_ok());
        assert!(!live.get());
    }
}
