//! Reusable live Handoff V2 receiver orchestration.
//!
//! Both the OpenAI server and the release qualifier use this exact path, so a
//! successful benchmark cannot accidentally bypass TLS, leaf pins, replay
//! admission, exact identity checks, or the detached atomic engine commit.
//!
//! One producer, one transfer at a time: the receive lease serializes requests
//! and the configuration names a single control endpoint and HMAC key id.
//! Connections that arrive for any other request are closed, never installed.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use kvpack_handoff::MultimodalIdentityV2;
use muser_engine::dflash::DFlashAssistant;
use muser_engine::Session;

use crate::config::ReceiverConfigV2;
use crate::control::{
    read_control, write_control, PrefillControlRequestV1, PrefillControlRequestV2,
    PrefillControlResponseV1, PrefillControlSegmentV2, PrefillControlStatusV1,
    ProducerPhaseReceiptV1, MUSER_PREFILL_CONTROL_ALPN,
};
use crate::identity::{
    transfer_id_matches, BeginExpectationsV2, CheckedSinkV2, REQUEST_ID_NAMESPACE,
};
use crate::muse_sink::MuseCacheShadow;
use crate::producer::{
    read_begin_v2, receive_begun_v2_with_replay_ack_phased, ReceiverError, ReceiverPolicy,
};
use crate::security::{
    accept_mtls, connect_mtls_with_alpn, load_mac_key, ClientTlsStream, ReplayLedger,
    ServerTlsStream, TlsFiles,
};
use crate::transport::{BeginAdmissionV2, FrameLimitsV2};

/// Either transport's server-side TLS stream. The `receive_v2`/`producer.rs`
/// family below the accept path is already generic over `impl Read + Write`
/// — this is the minimum needed so the accept path itself (which does own
/// the concrete TLS/transport setup) can select TCP or RDMA per receiver
/// process without duplicating any of that logic.
enum AnyServerTlsStream {
    Tcp(ServerTlsStream),
    #[cfg(feature = "melon-rdma")]
    Rdma(crate::security::rdma::ServerTlsStreamRdma),
}

impl Read for AnyServerTlsStream {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            AnyServerTlsStream::Tcp(s) => s.read(buf),
            #[cfg(feature = "melon-rdma")]
            AnyServerTlsStream::Rdma(s) => s.read(buf),
        }
    }
}

impl Write for AnyServerTlsStream {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            AnyServerTlsStream::Tcp(s) => s.write(buf),
            #[cfg(feature = "melon-rdma")]
            AnyServerTlsStream::Rdma(s) => s.write(buf),
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            AnyServerTlsStream::Tcp(s) => s.flush(),
            #[cfg(feature = "melon-rdma")]
            AnyServerTlsStream::Rdma(s) => s.flush(),
        }
    }
}

/// `MUSER_TRANSPORT=rdma` selects the RDMA accept path for every incoming
/// connection on this receiver process; unset (or any other value) is the
/// unchanged TCP default. A per-process switch, not a per-connection
/// negotiation — matching how `GGML_RPC_REQUIRE_RDMA` gates MelonDMA's
/// llama.cpp transport. First pass: an env var, not a `ReceiverConfigV2`
/// field, to avoid touching that struct's `deny_unknown_fields` schema and
/// `muser node add`'s config generator while this path is still being
/// proven — fold it into the generated config once proven on hardware.
fn rdma_transport_enabled() -> bool {
    std::env::var("MUSER_TRANSPORT")
        .map(|value| value == "rdma")
        .unwrap_or(false)
}

#[cfg(feature = "melon-rdma")]
fn rdma_dev() -> String {
    std::env::var("MUSER_RDMA_DEV").unwrap_or_else(|_| "mlx5_0".to_string())
}

#[cfg(feature = "melon-rdma")]
fn rdma_gid_index() -> i32 {
    // Re-verify with `ibv_devinfo -v` before trusting a previously-known
    // value — this exact GID table has drifted once already on the paired
    // Linux box (a bonded interface moved the expected RoCEv1 entry).
    std::env::var("MUSER_RDMA_GID")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}

/// Producer telemetry is read after the generation is already live, so it is
/// held to a deadline that decode can afford to lose.
const PRODUCER_RECEIPT_DEADLINE: Duration = Duration::from_millis(250);

/// Why one remote prefill did not become live. The server owns the fallback
/// counters; this is the taxonomy it counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoteReceiveCause {
    /// The receiver refused the request before it reached the wire.
    Receiver,
    /// The resident producer control channel could not be reached or refused.
    Control,
    /// No producer connection for this request arrived before the deadline.
    AcceptTimeout,
    /// TLS pins, ALPN, replay admission, exact identity, or the seal rejected
    /// the producer.
    Verify,
    /// The producer connection failed or aborted mid-transfer.
    Transfer,
    /// The engine refused the verified generation.
    Install,
}

impl RemoteReceiveCause {
    /// Stable phase label carried into every surfaced error message: a
    /// timeout that names its phase is diagnosable from the receiver alone,
    /// without correlating producer-side journals.
    pub fn phase(self) -> &'static str {
        match self {
            RemoteReceiveCause::Receiver => "receiver refusal",
            RemoteReceiveCause::Control => "producer control channel",
            RemoteReceiveCause::AcceptTimeout => "waiting for the producer connection",
            RemoteReceiveCause::Verify => "producer verification",
            RemoteReceiveCause::Transfer => "mid-transfer",
            RemoteReceiveCause::Install => "engine install",
        }
    }
}

#[derive(Debug, Clone, thiserror::Error)]
#[error("{} ({})", .message, .cause.phase())]
pub struct RemoteReceiveError {
    pub cause: RemoteReceiveCause,
    pub message: String,
}

impl RemoteReceiveError {
    fn new(cause: RemoteReceiveCause, message: impl Into<String>) -> Self {
        Self {
            cause,
            message: message.into(),
        }
    }

    /// Transport-level failures name the wire; everything else is either the
    /// engine refusing an install or a verification refusal.
    fn from_receive(error: ReceiverError, install_failed: bool) -> Self {
        let cause = match &error {
            _ if install_failed => RemoteReceiveCause::Install,
            ReceiverError::Transport(_)
            | ReceiverError::ProducerAbort(_)
            | ReceiverError::MissingBegin => RemoteReceiveCause::Transfer,
            ReceiverError::Handoff(_) | ReceiverError::Security(_) => RemoteReceiveCause::Verify,
        };
        Self::new(cause, error.to_string())
    }
}

#[derive(Debug, Clone)]
pub struct RemoteReceiveReceipt {
    pub transfer_id: String,
    pub generation: u64,
    pub installed_segments: u32,
    pub installed_bytes: u64,
    pub control_ns: u64,
    pub accept_ns: u64,
    pub transfer_commit_ns: u64,
    pub total_ns: u64,
    pub producer: Option<ProducerPhaseReceiptV1>,
    pub components: crate::muse_sink::ComponentInstallEvidence,
    /// N-series phase split: socket drain, verify, sink install, seal, commit.
    pub phases: crate::phase::HandoffPhaseNanos,
}

/// The commit path durably reserves every generation with
/// write+fsync+rename+directory-fsync before the ACK leaves. On a volume with
/// a slow directory-fsync tail this stalls the ACK and first decode by
/// hundreds of milliseconds at random (the 2026-08-18 p4 seal stall), so a
/// receiver whose replay ledger sits on such a volume is refused at bind
/// time. `scripts/gx10/durable_fsync_probe.py` is the standalone operator
/// check for the same pattern.
const LEDGER_RESERVE_PROBE_ITERATIONS: usize = 20;
const LEDGER_RESERVE_PROBE_MAX_TAIL: Duration = Duration::from_millis(100);

fn check_ledger_volume(ledger: &Path) -> Result<(), String> {
    check_ledger_volume_with(ledger, probe_ledger_reserve)
}

fn check_ledger_volume_with(
    ledger: &Path,
    probe: impl Fn(&Path) -> Result<Vec<Duration>, String>,
) -> Result<(), String> {
    let directory = ledger
        .parent()
        .ok_or_else(|| "replay ledger path has no parent directory".to_string())?;
    if !directory.is_dir() {
        std::fs::create_dir_all(directory)
            .map_err(|error| format!("create replay ledger directory: {error}"))?;
    }
    let samples = probe(directory)?;
    let tail = samples
        .iter()
        .max()
        .copied()
        .ok_or_else(|| "replay ledger probe produced no samples".to_string())?;
    if tail > LEDGER_RESERVE_PROBE_MAX_TAIL {
        return Err(format!(
            "replay ledger volume {} has a {tail:?} durable-reserve tail; \
             the handoff commit path would stall on it — point replay_ledger \
             at the internal disk (see scripts/gx10/durable_fsync_probe.py)",
            directory.display()
        ));
    }
    Ok(())
}

fn probe_ledger_reserve(directory: &Path) -> Result<Vec<Duration>, String> {
    use std::io::Write;
    let dir_handle = std::fs::File::open(directory)
        .map_err(|error| format!("open replay ledger directory: {error}"))?;
    let mut samples = Vec::with_capacity(LEDGER_RESERVE_PROBE_ITERATIONS);
    for index in 0..LEDGER_RESERVE_PROBE_ITERATIONS {
        let temporary = directory.join(format!(".ledger-probe-{}-{index}.tmp", std::process::id()));
        let final_path = directory.join(format!("ledger-probe-{}-{index}", std::process::id()));
        let started = Instant::now();
        let result = (|| -> Result<(), String> {
            let mut file = std::fs::File::create(&temporary)
                .map_err(|error| format!("create ledger probe file: {error}"))?;
            file.write_all(&[0u8; 4096])
                .and_then(|()| file.sync_all())
                .map_err(|error| format!("write+fsync ledger probe file: {error}"))?;
            drop(file);
            std::fs::rename(&temporary, &final_path)
                .map_err(|error| format!("rename ledger probe file: {error}"))?;
            dir_handle
                .sync_all()
                .map_err(|error| format!("fsync replay ledger directory: {error}"))?;
            Ok(())
        })();
        samples.push(started.elapsed());
        let _ = std::fs::remove_file(&final_path);
        result?;
    }
    Ok(samples)
}

pub struct RemoteReceiver {
    listener: TcpListener,
    receive_lease: Mutex<()>,
    replay: Mutex<ReplayLedger>,
    key: kvpack_handoff::MacKey,
    config: ReceiverConfigV2,
}

impl RemoteReceiver {
    pub fn bind(config: ReceiverConfigV2) -> Result<Self, String> {
        let key = load_mac_key(&config.hmac_key_file).map_err(|error| error.to_string())?;
        let replay =
            ReplayLedger::load(&config.replay_ledger).map_err(|error| error.to_string())?;
        check_ledger_volume(&config.replay_ledger)?;
        let listener = TcpListener::bind(config.listen)
            .map_err(|error| format!("bind {}: {error}", config.listen))?;
        listener
            .set_nonblocking(true)
            .map_err(|error| error.to_string())?;
        Ok(Self {
            listener,
            receive_lease: Mutex::new(()),
            replay: Mutex::new(replay),
            key,
            config,
        })
    }

    pub fn config(&self) -> &ReceiverConfigV2 {
        &self.config
    }

    /// Request and atomically install one exact remote prefix. If no resident
    /// control endpoint is configured, `wait_without_control` selects between
    /// fail-fast auto mode and the configured unsolicited-producer wait.
    pub fn receive(
        &self,
        session: &mut Session,
        dflash: Option<&mut DFlashAssistant>,
        prompt_witnesses: &[u32],
        multimodal: Option<(MultimodalIdentityV2, Vec<PrefillControlSegmentV2>)>,
        max_context: usize,
        wait_without_control: bool,
    ) -> Result<RemoteReceiveReceipt, String> {
        self.receive_classified(
            session,
            dflash,
            prompt_witnesses,
            multimodal,
            max_context,
            wait_without_control,
        )
        .map_err(|error| error.message)
    }

    /// `receive` with the fallback cause preserved for the server's counters.
    pub fn receive_classified(
        &self,
        session: &mut Session,
        dflash: Option<&mut DFlashAssistant>,
        prompt_witnesses: &[u32],
        multimodal: Option<(MultimodalIdentityV2, Vec<PrefillControlSegmentV2>)>,
        max_context: usize,
        wait_without_control: bool,
    ) -> Result<RemoteReceiveReceipt, RemoteReceiveError> {
        let total_started = Instant::now();
        let _lease = self.receive_lease.lock().map_err(|_| {
            RemoteReceiveError::new(
                RemoteReceiveCause::Receiver,
                "remote receiver lease was poisoned",
            )
        })?;
        if prompt_witnesses.len() < 2 {
            return Err(RemoteReceiveError::new(
                RemoteReceiveCause::Receiver,
                "remote prefill needs a held boundary token",
            ));
        }
        let now_unix_ms = unix_ms();
        let request_id = format!(
            "{REQUEST_ID_NAMESPACE}{}-{}",
            now_unix_ms,
            REMOTE_REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed)
        );
        let control_started = Instant::now();
        let mut control = self
            .start_control(
                &request_id,
                prompt_witnesses,
                multimodal.as_ref(),
                now_unix_ms,
            )
            .map_err(|message| RemoteReceiveError::new(RemoteReceiveCause::Control, message))?;
        let control_ns = nanos(control_started.elapsed());
        let wait = if control.is_some() || wait_without_control {
            self.config.producer_wait_for(prompt_witnesses.len())
        } else {
            Duration::ZERO
        };
        // Only a producer answering our own control request can be held to the
        // request id; an unsolicited producer has never seen one.
        let expected_prefix = control.is_some().then(|| request_id.clone());
        let accept_started = Instant::now();
        let (mut stream, admission) =
            self.accept_matching_begin(expected_prefix.as_deref(), wait)?;
        let accept_ns = nanos(accept_started.elapsed());
        let transfer_started = Instant::now();
        let expectations = BeginExpectationsV2 {
            identity: self.config.identity.clone(),
            prompt_token_ids: prompt_witnesses[..prompt_witnesses.len() - 1].to_vec(),
            target_cache_identity_sha256: self.config.target_cache_identity_sha256.clone(),
            dflash_identity_sha256: self.config.dflash_identity_sha256.clone(),
            multimodal: multimodal.as_ref().map(|(identity, _)| identity.clone()),
            expected_transfer_id_prefix: expected_prefix,
            max_context,
            prefix_cut: admission.prefix_cut,
            held_token_ids: session.token_history().to_vec(),
        };
        let policy = ReceiverPolicy {
            now_unix_ms,
            expected_key_id: &self.config.hmac_key_id,
            minimum_key_epoch: self.config.minimum_hmac_epoch,
            limits: Default::default(),
        };
        let mut replay = self.replay.lock().map_err(|_| {
            RemoteReceiveError::new(
                RemoteReceiveCause::Receiver,
                "remote replay ledger was poisoned",
            )
        })?;
        let component_evidence = std::sync::Arc::new(std::sync::Mutex::new(
            crate::muse_sink::ComponentInstallEvidence::default(),
        ));
        let phase_evidence = crate::phase::new_shared_phase();
        let (committed, install_failed) = if let Some(dflash) = dflash {
            let geometry = self.config.dflash_context_geometry.ok_or_else(|| {
                RemoteReceiveError::new(
                    RemoteReceiveCause::Receiver,
                    "combined receiver has no enrolled DFlash context geometry",
                )
            })?;
            let shadow =
                MuseCacheShadow::new_combined(session, dflash, geometry).map_err(|message| {
                    RemoteReceiveError::new(RemoteReceiveCause::Receiver, message)
                })?;
            let sink = CheckedSinkV2::new(
                expectations,
                shadow
                    .with_prefix_cut(admission.prefix_cut)
                    .with_evidence(std::sync::Arc::clone(&component_evidence))
                    .with_phase_timing(std::sync::Arc::clone(&phase_evidence)),
            );
            let install_failed = sink.install_failures();
            let committed = receive_begun_v2_with_replay_ack_phased(
                &mut stream,
                admission.manifest,
                self.key.clone(),
                policy,
                &mut replay,
                sink,
                Some(std::sync::Arc::clone(&phase_evidence)),
            );
            (committed, install_failed)
        } else {
            let sink = CheckedSinkV2::new(
                expectations,
                MuseCacheShadow::new(session)
                    .with_prefix_cut(admission.prefix_cut)
                    .with_evidence(std::sync::Arc::clone(&component_evidence))
                    .with_phase_timing(std::sync::Arc::clone(&phase_evidence)),
            );
            let install_failed = sink.install_failures();
            let committed = receive_begun_v2_with_replay_ack_phased(
                &mut stream,
                admission.manifest,
                self.key.clone(),
                policy,
                &mut replay,
                sink,
                Some(std::sync::Arc::clone(&phase_evidence)),
            );
            (committed, install_failed)
        };
        let committed = committed.map_err(|error| {
            let mut mapped = RemoteReceiveError::from_receive(error, install_failed.is_set());
            // Stamp where the time went: without this, a timeout here is
            // indistinguishable from one in control or accept, and the phase
            // has to be reconstructed from producer-side journals.
            mapped.message = format!(
                "{} [control {}ms, accept {}ms, {}ms into the transfer]",
                mapped.message,
                control_ns / 1_000_000,
                accept_ns / 1_000_000,
                transfer_started.elapsed().as_millis()
            );
            mapped
        })?;
        let transfer_commit_ns = nanos(transfer_started.elapsed());
        let mut producer = None;
        if let Some(stream) = control.as_mut() {
            // Handoff ACK is the TTFT commit boundary (`transfer_commit_ns`).
            // This later control-channel receipt is telemetry only, can never
            // roll the generation back, and must never hold decode behind the
            // producer: past its own short deadline it is simply absent.
            let _ = stream
                .sock
                .set_read_timeout(Some(PRODUCER_RECEIPT_DEADLINE));
            if let Ok(response) = read_control::<PrefillControlResponseV1>(stream) {
                if response.validate(&request_id).is_ok()
                    && response.status == PrefillControlStatusV1::Committed
                {
                    producer = response.receipt;
                }
            }
        }
        let components = component_evidence
            .lock()
            .map_err(|_| {
                RemoteReceiveError::new(
                    RemoteReceiveCause::Receiver,
                    "component install evidence lock was poisoned",
                )
            })?
            .clone();
        let phases = phase_evidence
            .lock()
            .map_err(|_| {
                RemoteReceiveError::new(
                    RemoteReceiveCause::Receiver,
                    "handoff phase evidence lock was poisoned",
                )
            })?
            .clone();
        Ok(RemoteReceiveReceipt {
            phases,
            transfer_id: committed.transfer_id,
            generation: committed.generation,
            installed_segments: committed.installed_segments,
            installed_bytes: committed.installed_bytes,
            control_ns,
            accept_ns,
            transfer_commit_ns,
            total_ns: nanos(total_started.elapsed()),
            producer,
            components,
        })
    }

    fn start_control(
        &self,
        request_id: &str,
        prompt_witnesses: &[u32],
        multimodal: Option<&(MultimodalIdentityV2, Vec<PrefillControlSegmentV2>)>,
        now_unix_ms: u64,
    ) -> Result<Option<ClientTlsStream>, String> {
        let Some(endpoint) = self.config.producer_control.as_ref() else {
            return Ok(None);
        };
        let host = self
            .config
            .advertised_receiver_host
            .as_ref()
            .ok_or_else(|| "advertised receiver host is absent".to_string())?;
        let mut stream = connect_mtls_with_alpn(
            endpoint.address,
            &endpoint.server_name,
            TlsFiles {
                certificate_chain: &self.config.certificate_chain,
                private_key: &self.config.private_key,
                peer_ca: &self.config.peer_ca,
                leaf_sha256_pins: &self.config.peer_leaf_sha256,
            },
            self.config.timeout(),
            MUSER_PREFILL_CONTROL_ALPN,
        )
        .map_err(|error| error.to_string())?;
        // The deadline the producer is told must match the patience this
        // receiver will actually extend to the transfer, which scales with
        // prompt depth; the flat timeout only suits shallow prompts.
        let deadline_unix_ms = now_unix_ms
            .checked_add(
                self.config
                    .producer_wait_for(prompt_witnesses.len())
                    .as_millis() as u64,
            )
            .ok_or_else(|| "control deadline overflow".to_string())?;
        if let Some((identity, segments)) = multimodal {
            let request = PrefillControlRequestV2 {
                schema_version: 2,
                request_id: request_id.into(),
                deadline_unix_ms,
                segments: segments.clone(),
                multimodal: identity.clone(),
                receiver_host: host.clone(),
                receiver_port: self.config.listen.port(),
            };
            request.validate(now_unix_ms, usize::MAX)?;
            write_control(&mut stream, &request)?;
        } else {
            let request = PrefillControlRequestV1 {
                schema_version: 1,
                request_id: request_id.into(),
                deadline_unix_ms,
                prompt_token_ids: prompt_witnesses.to_vec(),
                receiver_host: host.clone(),
                receiver_port: self.config.listen.port(),
            };
            request.validate(now_unix_ms, usize::MAX)?;
            write_control(&mut stream, &request)?;
        }
        let response: PrefillControlResponseV1 = read_control(&mut stream)?;
        response.validate(request_id)?;
        if response.status != PrefillControlStatusV1::Accepted {
            return Err(response
                .error
                .unwrap_or_else(|| "GX10 producer rejected the request".into()));
        }
        Ok(Some(stream))
    }

    /// Accept producer connections until one carries this request's Begin
    /// frame. A connection that fails the handshake, never sends Begin, or
    /// names another request is closed and the wait continues: a previous
    /// timed-out request's late producer must never fail the next request.
    fn accept_matching_begin(
        &self,
        expected_prefix: Option<&str>,
        wait: Duration,
    ) -> Result<(AnyServerTlsStream, BeginAdmissionV2), RemoteReceiveError> {
        let deadline = Instant::now() + wait;
        let mut dropped = 0usize;
        let mut last_drop = String::new();
        loop {
            let tcp = self.accept_until(deadline, dropped, &last_drop)?;
            match self.begin_from(tcp, deadline) {
                Ok((stream, admission))
                    if expected_prefix.is_none_or(|prefix| {
                        transfer_id_matches(prefix, &admission.manifest.transfer_id)
                    }) =>
                {
                    return Ok((stream, admission))
                }
                Ok((_, admission)) => {
                    dropped += 1;
                    last_drop = format!(
                        "transfer {} belongs to another request",
                        admission.manifest.transfer_id
                    );
                }
                Err(error) => {
                    dropped += 1;
                    last_drop = error;
                }
            }
        }
    }

    fn begin_from(
        &self,
        tcp: TcpStream,
        deadline: Instant,
    ) -> Result<(AnyServerTlsStream, BeginAdmissionV2), String> {
        let files = TlsFiles {
            certificate_chain: &self.config.certificate_chain,
            private_key: &self.config.private_key,
            peer_ca: &self.config.peer_ca,
            leaf_sha256_pins: &self.config.peer_leaf_sha256,
        };
        let mut stream = if rdma_transport_enabled() {
            #[cfg(feature = "melon-rdma")]
            {
                let rdma = crate::security::rdma::accept_mtls_over_rdma(
                    tcp,
                    files,
                    crate::security::MUSER_HANDOFF_ALPN,
                    &rdma_dev(),
                    rdma_gid_index(),
                )
                .map_err(|error| error.to_string())?;
                AnyServerTlsStream::Rdma(rdma)
            }
            #[cfg(not(feature = "melon-rdma"))]
            {
                return Err(
                    "MUSER_TRANSPORT=rdma requested but this build lacks the melon-rdma feature"
                        .to_string(),
                );
            }
        } else {
            let tcp_tls = accept_mtls(tcp, files, self.config.timeout())
                .map_err(|error| error.to_string())?;
            AnyServerTlsStream::Tcp(tcp_tls)
        };
        // A producer that handshakes and then says nothing must not hold the
        // whole socket timeout: admission is bounded by the accept deadline
        // this request is already waiting on.
        let admission = deadline
            .saturating_duration_since(Instant::now())
            .max(PRODUCER_RECEIPT_DEADLINE)
            .min(self.config.timeout());
        set_read_timeout(&stream, admission)?;
        let admission = read_begin_v2(&mut stream, FrameLimitsV2::default())
            .map_err(|error| error.to_string())?;
        // The producer connects and sends its begin within milliseconds, then
        // computes prefill before the first segment appears — for deep prompts
        // that compute was measured in minutes. The transfer reads must carry
        // the same depth-scaled patience as the accept wait, not the flat
        // config timeout, or a deep prefill dies here while the producer is
        // honestly working.
        let transfer_read = deadline
            .saturating_duration_since(Instant::now())
            .max(self.config.timeout());
        set_read_timeout(&stream, transfer_read)?;
        Ok((stream, admission))
    }

    fn accept_until(
        &self,
        deadline: Instant,
        dropped: usize,
        last_drop: &str,
    ) -> Result<TcpStream, RemoteReceiveError> {
        loop {
            match self.listener.accept() {
                Ok((tcp, _)) => {
                    tcp.set_nonblocking(false).map_err(|error| {
                        RemoteReceiveError::new(
                            RemoteReceiveCause::AcceptTimeout,
                            error.to_string(),
                        )
                    })?;
                    return Ok(tcp);
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    if Instant::now() >= deadline {
                        let detail = if dropped == 0 {
                            String::new()
                        } else {
                            format!(" (dropped {dropped} stale connections, last: {last_drop})")
                        };
                        return Err(RemoteReceiveError::new(
                            RemoteReceiveCause::AcceptTimeout,
                            format!("timed out waiting for the GX10 producer{detail}"),
                        ));
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(error) => {
                    return Err(RemoteReceiveError::new(
                        RemoteReceiveCause::AcceptTimeout,
                        format!("accept remote producer: {error}"),
                    ))
                }
            }
        }
    }
}

fn set_read_timeout(stream: &AnyServerTlsStream, timeout: Duration) -> Result<(), String> {
    match stream {
        AnyServerTlsStream::Tcp(s) => s
            .sock
            .set_read_timeout(Some(timeout))
            .map_err(|error| error.to_string()),
        // The RDMA byte-pipe's recv() has no socket-level read-timeout
        // equivalent yet — a known limitation of this first pass, not an
        // oversight. A stalled peer still surfaces (an RC QP's bounded
        // retry count eventually produces an error completion rather than
        // hanging the process forever), just not on this specific deadline.
        #[cfg(feature = "melon-rdma")]
        AnyServerTlsStream::Rdma(_) => Ok(()),
    }
}

static REMOTE_REQUEST_COUNTER: AtomicU64 = AtomicU64::new(0);

fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX)
}

fn nanos(value: Duration) -> u64 {
    value.as_nanos().min(u64::MAX as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ledger_volume_check_refuses_a_slow_reserve_tail() {
        let directory =
            std::env::temp_dir().join(format!("muser-ledger-probe-test-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let ledger = directory.join("replay.json");
        let refused = check_ledger_volume_with(&ledger, |_| {
            Ok(vec![Duration::from_micros(150), Duration::from_millis(700)])
        });
        assert!(refused.is_err());
        let accepted = check_ledger_volume_with(&ledger, |_| {
            Ok(vec![
                Duration::from_micros(150);
                LEDGER_RESERVE_PROBE_ITERATIONS
            ])
        });
        assert!(accepted.is_ok());
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn ledger_reserve_probe_produces_samples_and_cleans_up() {
        let directory =
            std::env::temp_dir().join(format!("muser-ledger-probe-live-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let samples = probe_ledger_reserve(&directory).unwrap();
        assert_eq!(samples.len(), LEDGER_RESERVE_PROBE_ITERATIONS);
        assert!(std::fs::read_dir(&directory).unwrap().next().is_none());
        let _ = std::fs::remove_dir_all(&directory);
    }
}
