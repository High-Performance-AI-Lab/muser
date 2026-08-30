//! TLS 1.3-only mutually authenticated transport with exact ALPN and leaf pins.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use rustls::client::ClientConnection;
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName};
use rustls::server::{ServerConfig, ServerConnection, WebPkiClientVerifier};
use rustls::{ClientConfig, RootCertStore, StreamOwned};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use kvpack_handoff::MacKey;

pub const MUSER_HANDOFF_ALPN: &[u8] = b"muser-kvpack-v2";
const HANDOFF_SOCKET_BUFFER_BYTES: usize = 8 * 1024 * 1024;
const MAX_PRIVATE_KEY_BYTES: u64 = 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum SecurityError {
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("TLS: {0}")]
    Tls(#[from] rustls::Error),
    #[error("TLS configuration: {0}")]
    Config(String),
    #[error("peer leaf certificate pin mismatch")]
    LeafPin,
    #[error("peer did not negotiate exact Muser handoff ALPN")]
    Alpn,
    #[error("replayed or stale generation")]
    Replay,
    #[error("handoff generation zero is never installable")]
    ZeroGeneration,
}

pub struct TlsFiles<'a> {
    pub certificate_chain: &'a Path,
    pub private_key: &'a Path,
    pub peer_ca: &'a Path,
    pub leaf_sha256_pins: &'a BTreeSet<String>,
}

pub type ClientTlsStream = StreamOwned<ClientConnection, TcpStream>;
pub type ServerTlsStream = StreamOwned<ServerConnection, TcpStream>;

/// Read a tenant HMAC key without accepting symlinks, loose Unix
/// permissions, short reads, or ambiguous encodings.
pub fn load_mac_key(path: &Path) -> Result<MacKey, SecurityError> {
    let before = std::fs::symlink_metadata(path)?;
    if before.file_type().is_symlink() || !before.is_file() {
        return Err(SecurityError::Config(
            "HMAC key must be a regular non-symlink file".into(),
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if before.mode() & 0o077 != 0 {
            return Err(SecurityError::Config(
                "HMAC key must not be accessible by group/other".into(),
            ));
        }
    }
    let file = File::open(path)?;
    let after = file.metadata()?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if before.dev() != after.dev() || before.ino() != after.ino() {
            return Err(SecurityError::Config(
                "HMAC key file changed while it was opened".into(),
            ));
        }
    }
    let mut bytes = Vec::new();
    file.take(4_096).read_to_end(&mut bytes)?;
    if after.len() > 4_096 {
        return Err(SecurityError::Config("HMAC key file is too large".into()));
    }
    if bytes.len() == 32 {
        return MacKey::from_slice(&bytes)
            .map_err(|error| SecurityError::Config(error.to_string()));
    }
    let text = std::str::from_utf8(&bytes)
        .map_err(|_| SecurityError::Config("HMAC key is not UTF-8 hex".into()))?
        .trim_end_matches(['\r', '\n']);
    MacKey::from_hex(text).map_err(|error| SecurityError::Config(error.to_string()))
}

pub fn connect_mtls(
    address: SocketAddr,
    server_name: &str,
    files: TlsFiles<'_>,
    timeout: Duration,
) -> Result<ClientTlsStream, SecurityError> {
    connect_mtls_with_alpn(address, server_name, files, timeout, MUSER_HANDOFF_ALPN)
}

pub fn connect_mtls_with_alpn(
    address: SocketAddr,
    server_name: &str,
    files: TlsFiles<'_>,
    timeout: Duration,
    alpn: &[u8],
) -> Result<ClientTlsStream, SecurityError> {
    validate_alpn(alpn)?;
    validate_pins(files.leaf_sha256_pins)?;
    let roots = load_roots(files.peer_ca)?;
    let certificates = load_certificates(files.certificate_chain)?;
    let key = load_private_key(files.private_key)?;
    // Select the provider on this connection instead of relying on rustls'
    // process-global feature inference. Workspace binaries may legitimately
    // contain both ring and aws-lc through unrelated HTTP dependencies.
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let mut config = ClientConfig::builder_with_provider(provider)
        .with_protocol_versions(&[&rustls::version::TLS13])
        .map_err(|error| SecurityError::Config(error.to_string()))?
        .with_root_certificates(roots)
        .with_client_auth_cert(certificates, key)
        .map_err(|error| SecurityError::Config(error.to_string()))?;
    config.alpn_protocols = vec![alpn.to_vec()];
    let name = ServerName::try_from(server_name.to_owned())
        .map_err(|error| SecurityError::Config(format!("invalid server name: {error}")))?;
    let connection = ClientConnection::new(Arc::new(config), name)?;
    let tcp = TcpStream::connect_timeout(&address, timeout)?;
    tune_handoff_socket(&tcp)?;
    tcp.set_read_timeout(Some(timeout))?;
    tcp.set_write_timeout(Some(timeout))?;
    let mut stream = StreamOwned::new(connection, tcp);
    while stream.conn.is_handshaking() {
        stream.conn.complete_io(&mut stream.sock)?;
    }
    verify_connection(
        stream.conn.alpn_protocol(),
        stream.conn.peer_certificates(),
        files.leaf_sha256_pins,
        alpn,
    )?;
    Ok(stream)
}

pub fn accept_mtls(
    tcp: TcpStream,
    files: TlsFiles<'_>,
    timeout: Duration,
) -> Result<ServerTlsStream, SecurityError> {
    accept_mtls_with_alpn(tcp, files, timeout, MUSER_HANDOFF_ALPN)
}

pub fn accept_mtls_with_alpn(
    tcp: TcpStream,
    files: TlsFiles<'_>,
    timeout: Duration,
    alpn: &[u8],
) -> Result<ServerTlsStream, SecurityError> {
    validate_alpn(alpn)?;
    validate_pins(files.leaf_sha256_pins)?;
    let roots = Arc::new(load_roots(files.peer_ca)?);
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let verifier = WebPkiClientVerifier::builder_with_provider(roots, Arc::clone(&provider))
        .build()
        .map_err(|error| SecurityError::Config(error.to_string()))?;
    let certificates = load_certificates(files.certificate_chain)?;
    let key = load_private_key(files.private_key)?;
    let mut config = ServerConfig::builder_with_provider(provider)
        .with_protocol_versions(&[&rustls::version::TLS13])
        .map_err(|error| SecurityError::Config(error.to_string()))?
        .with_client_cert_verifier(verifier)
        .with_single_cert(certificates, key)
        .map_err(|error| SecurityError::Config(error.to_string()))?;
    config.alpn_protocols = vec![alpn.to_vec()];
    tcp.set_read_timeout(Some(timeout))?;
    tcp.set_write_timeout(Some(timeout))?;
    tune_handoff_socket(&tcp)?;
    let connection = ServerConnection::new(Arc::new(config))?;
    let mut stream = StreamOwned::new(connection, tcp);
    while stream.conn.is_handshaking() {
        stream.conn.complete_io(&mut stream.sock)?;
    }
    verify_connection(
        stream.conn.alpn_protocol(),
        stream.conn.peer_certificates(),
        files.leaf_sha256_pins,
        alpn,
    )?;
    Ok(stream)
}

/// RDMA counterparts of [`connect_mtls_with_alpn`]/[`accept_mtls_with_alpn`].
/// Same TLS config, verification, and ALPN/leaf-pin logic — only the byte
/// transport underneath `StreamOwned` changes, from `TcpStream` to
/// [`crate::melon_rdma::MelonRdmaStream`]. `bootstrap` is an already
/// TCP-connected socket used only to exchange RDMA QP parameters before the
/// connection switches to RDMA SEND/RECV for everything after, including
/// the TLS handshake itself — the same bootstrap-then-upgrade pattern
/// MelonDMA's own `ggml-rpc/transport.cpp` uses.
#[cfg(feature = "melon-rdma")]
pub mod rdma {
    use super::*;
    use crate::melon_rdma::MelonRdmaStream;
    use std::net::TcpStream;
    use std::os::fd::IntoRawFd;

    pub type ClientTlsStreamRdma = StreamOwned<ClientConnection, MelonRdmaStream>;
    pub type ServerTlsStreamRdma = StreamOwned<ServerConnection, MelonRdmaStream>;

    #[allow(clippy::too_many_arguments)]
    pub fn connect_mtls_over_rdma(
        bootstrap: TcpStream,
        server_name: &str,
        files: TlsFiles<'_>,
        alpn: &[u8],
        rdma_dev: &str,
        rdma_gid_index: i32,
    ) -> Result<ClientTlsStreamRdma, SecurityError> {
        validate_alpn(alpn)?;
        validate_pins(files.leaf_sha256_pins)?;
        let roots = load_roots(files.peer_ca)?;
        let certificates = load_certificates(files.certificate_chain)?;
        let key = load_private_key(files.private_key)?;
        let provider = Arc::new(rustls::crypto::ring::default_provider());
        let mut config = ClientConfig::builder_with_provider(provider)
            .with_protocol_versions(&[&rustls::version::TLS13])
            .map_err(|error| SecurityError::Config(error.to_string()))?
            .with_root_certificates(roots)
            .with_client_auth_cert(certificates, key)
            .map_err(|error| SecurityError::Config(error.to_string()))?;
        config.alpn_protocols = vec![alpn.to_vec()];
        let name = ServerName::try_from(server_name.to_owned())
            .map_err(|error| SecurityError::Config(format!("invalid server name: {error}")))?;
        let connection = ClientConnection::new(Arc::new(config), name)?;

        let fd = bootstrap.into_raw_fd();
        let rdma = MelonRdmaStream::open(fd, rdma_dev, rdma_gid_index)?;

        let mut stream = StreamOwned::new(connection, rdma);
        while stream.conn.is_handshaking() {
            stream.conn.complete_io(&mut stream.sock)?;
        }
        verify_connection(
            stream.conn.alpn_protocol(),
            stream.conn.peer_certificates(),
            files.leaf_sha256_pins,
            alpn,
        )?;
        Ok(stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn accept_mtls_over_rdma(
        bootstrap: TcpStream,
        files: TlsFiles<'_>,
        alpn: &[u8],
        rdma_dev: &str,
        rdma_gid_index: i32,
    ) -> Result<ServerTlsStreamRdma, SecurityError> {
        validate_alpn(alpn)?;
        validate_pins(files.leaf_sha256_pins)?;
        let roots = Arc::new(load_roots(files.peer_ca)?);
        let provider = Arc::new(rustls::crypto::ring::default_provider());
        let verifier = WebPkiClientVerifier::builder_with_provider(roots, Arc::clone(&provider))
            .build()
            .map_err(|error| SecurityError::Config(error.to_string()))?;
        let certificates = load_certificates(files.certificate_chain)?;
        let key = load_private_key(files.private_key)?;
        let mut config = ServerConfig::builder_with_provider(provider)
            .with_protocol_versions(&[&rustls::version::TLS13])
            .map_err(|error| SecurityError::Config(error.to_string()))?
            .with_client_cert_verifier(verifier)
            .with_single_cert(certificates, key)
            .map_err(|error| SecurityError::Config(error.to_string()))?;
        config.alpn_protocols = vec![alpn.to_vec()];
        let connection = ServerConnection::new(Arc::new(config))?;

        let fd = bootstrap.into_raw_fd();
        let rdma = MelonRdmaStream::open(fd, rdma_dev, rdma_gid_index)?;

        let mut stream = StreamOwned::new(connection, rdma);
        while stream.conn.is_handshaking() {
            stream.conn.complete_io(&mut stream.sock)?;
        }
        verify_connection(
            stream.conn.alpn_protocol(),
            stream.conn.peer_certificates(),
            files.leaf_sha256_pins,
            alpn,
        )?;
        Ok(stream)
    }
}

fn verify_connection(
    alpn: Option<&[u8]>,
    peer: Option<&[CertificateDer<'_>]>,
    pins: &BTreeSet<String>,
    expected_alpn: &[u8],
) -> Result<(), SecurityError> {
    if alpn != Some(expected_alpn) {
        return Err(SecurityError::Alpn);
    }
    let leaf = peer
        .and_then(|chain| chain.first())
        .ok_or_else(|| SecurityError::Config("peer certificate chain is empty".into()))?;
    let digest = format!("{:x}", Sha256::digest(leaf.as_ref()));
    if !pins.contains(&digest) {
        return Err(SecurityError::LeafPin);
    }
    Ok(())
}

fn tune_handoff_socket(tcp: &TcpStream) -> Result<(), SecurityError> {
    tcp.set_nodelay(true)?;
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        let fd = tcp.as_raw_fd();
        let bytes = HANDOFF_SOCKET_BUFFER_BYTES as libc::c_int;
        unsafe {
            libc::setsockopt(
                fd,
                libc::SOL_SOCKET,
                libc::SO_RCVBUF,
                &bytes as *const libc::c_int as *const libc::c_void,
                std::mem::size_of_val(&bytes) as libc::socklen_t,
            );
            libc::setsockopt(
                fd,
                libc::SOL_SOCKET,
                libc::SO_SNDBUF,
                &bytes as *const libc::c_int as *const libc::c_void,
                std::mem::size_of_val(&bytes) as libc::socklen_t,
            );
        }
    }
    Ok(())
}

fn validate_alpn(alpn: &[u8]) -> Result<(), SecurityError> {
    if alpn.is_empty() || alpn.len() > 255 || !alpn.iter().all(u8::is_ascii) {
        return Err(SecurityError::Config("invalid ALPN protocol".into()));
    }
    Ok(())
}

fn validate_pins(pins: &BTreeSet<String>) -> Result<(), SecurityError> {
    if pins.is_empty()
        || pins.iter().any(|pin| {
            pin.len() != 64
                || !pin
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        })
    {
        return Err(SecurityError::Config(
            "leaf pins must be nonempty lowercase SHA-256 values".into(),
        ));
    }
    Ok(())
}

fn load_certificates(path: &Path) -> Result<Vec<CertificateDer<'static>>, SecurityError> {
    let certificates = CertificateDer::pem_file_iter(path)
        .map_err(|error| SecurityError::Config(format!("invalid certificate PEM: {error}")))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| SecurityError::Config(format!("invalid certificate PEM: {error}")))?;
    if certificates.is_empty() {
        return Err(SecurityError::Config(format!(
            "certificate chain is empty: {}",
            path.display()
        )));
    }
    Ok(certificates)
}

fn load_private_key(path: &Path) -> Result<PrivateKeyDer<'static>, SecurityError> {
    let bytes = read_private_key_file(path)?;
    PrivateKeyDer::from_pem_slice(&bytes)
        .map(|key| key.clone_key())
        .map_err(|error| SecurityError::Config(format!("invalid private-key PEM: {error}")))
}

/// TLS private material follows the same fail-closed file contract as the
/// handoff HMAC key. Parsing through the already-open descriptor closes the
/// validate-then-reopen race in `from_pem_file`.
fn read_private_key_file(path: &Path) -> Result<Vec<u8>, SecurityError> {
    let before = std::fs::symlink_metadata(path)?;
    if before.file_type().is_symlink() || !before.is_file() {
        return Err(SecurityError::Config(
            "TLS private key must be a regular non-symlink file".into(),
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;
        if before.mode() & 0o077 != 0 {
            return Err(SecurityError::Config(
                "TLS private key must not be accessible by group/other".into(),
            ));
        }
    }
    if before.len() == 0 || before.len() > MAX_PRIVATE_KEY_BYTES {
        return Err(SecurityError::Config(
            "TLS private key size is outside 1..=1048576 bytes".into(),
        ));
    }
    let file = File::open(path)?;
    let after = file.metadata()?;
    if before.len() != after.len() {
        return Err(SecurityError::Config(
            "TLS private key changed while it was opened".into(),
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt as _;
        if before.dev() != after.dev() || before.ino() != after.ino() || after.mode() & 0o077 != 0 {
            return Err(SecurityError::Config(
                "TLS private key changed while it was opened".into(),
            ));
        }
    }
    let mut bytes = Vec::with_capacity(before.len() as usize);
    file.take(MAX_PRIVATE_KEY_BYTES + 1)
        .read_to_end(&mut bytes)?;
    if bytes.is_empty() || bytes.len() as u64 > MAX_PRIVATE_KEY_BYTES {
        return Err(SecurityError::Config(
            "TLS private key size is outside 1..=1048576 bytes".into(),
        ));
    }
    Ok(bytes)
}

fn load_roots(path: &Path) -> Result<RootCertStore, SecurityError> {
    let certificates = load_certificates(path)?;
    let mut roots = RootCertStore::empty();
    let (added, ignored) = roots.add_parsable_certificates(certificates);
    if added == 0 || ignored != 0 {
        return Err(SecurityError::Config(format!(
            "invalid peer CA bundle: added {added}, ignored {ignored}"
        )));
    }
    Ok(roots)
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReplayState {
    highest_generation: BTreeMap<String, u64>,
}

/// Durable monotonic replay admission keyed by HMAC key ID and epoch.
pub struct ReplayLedger {
    path: PathBuf,
    state: ReplayState,
    latched_failure: Option<String>,
}

impl ReplayLedger {
    pub fn load(path: impl Into<PathBuf>) -> Result<Self, SecurityError> {
        let path = path.into();
        let state = match std::fs::read(&path) {
            Ok(bytes) => {
                validate_ledger_file(&path)?;
                serde_json::from_slice(&bytes)
                    .map_err(|error| SecurityError::Config(format!("replay ledger: {error}")))?
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let state = ReplayState::default();
                persist_replay_state(&path, &state)?;
                state
            }
            Err(error) => return Err(error.into()),
        };
        Ok(Self {
            path,
            state,
            latched_failure: None,
        })
    }

    /// Admission runs before any sink work: a generation the receiver can
    /// never install must be refused here, not after the engine commit.
    pub fn admit(&self, key_id: &str, epoch: u64, generation: u64) -> Result<(), SecurityError> {
        if let Some(error) = &self.latched_failure {
            return Err(SecurityError::Config(format!(
                "replay ledger is latched degraded until restart: {error}"
            )));
        }
        if generation == 0 {
            return Err(SecurityError::ZeroGeneration);
        }
        let key = format!("{key_id}:{epoch}");
        if self
            .state
            .highest_generation
            .get(&key)
            .is_some_and(|highest| generation <= *highest)
        {
            return Err(SecurityError::Replay);
        }
        Ok(())
    }

    /// Durably consume a generation after every component has prepared but
    /// before any live engine pointer is swapped. A failed reservation
    /// latches this ledger degraded; the receiver refuses all later traffic
    /// until the operator repairs storage and restarts the process.
    pub fn reserve(
        &mut self,
        key_id: &str,
        epoch: u64,
        generation: u64,
    ) -> Result<(), SecurityError> {
        self.admit(key_id, epoch, generation)?;
        let mut next = self.state.clone();
        next.highest_generation
            .insert(format!("{key_id}:{epoch}"), generation);
        if let Err(error) = persist_replay_state(&self.path, &next) {
            self.latched_failure = Some(error.to_string());
            return Err(error);
        }
        self.state = next;
        Ok(())
    }

    #[deprecated(note = "reserve before engine publication")]
    pub fn record(
        &mut self,
        key_id: &str,
        epoch: u64,
        generation: u64,
    ) -> Result<(), SecurityError> {
        self.reserve(key_id, epoch, generation)
    }
}

fn validate_ledger_file(path: &Path) -> Result<(), SecurityError> {
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(SecurityError::Config(
            "replay ledger must be a regular non-symlink file".into(),
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        if metadata.permissions().mode() & 0o077 != 0 {
            return Err(SecurityError::Config(
                "replay ledger must have mode 0600 or stricter".into(),
            ));
        }
    }
    Ok(())
}

fn persist_replay_state(path: &Path, state: &ReplayState) -> Result<(), SecurityError> {
    let bytes = serde_json::to_vec_pretty(state)
        .map_err(|error| SecurityError::Config(error.to_string()))?;
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;
    let temporary = parent.join(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("replay"),
        std::process::id()
    ));
    let result = (|| -> std::io::Result<()> {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            options.mode(0o600);
        }
        let mut output = options.open(&temporary)?;
        output.write_all(&bytes)?;
        output.write_all(b"\n")?;
        output.sync_all()?;
        std::fs::rename(&temporary, path)?;
        File::open(parent)?.sync_all()
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result.map_err(SecurityError::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_ledger_is_monotonic_and_durable() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("ledger.json");
        let mut ledger = ReplayLedger::load(&path).unwrap();
        ledger.admit("key", 4, 9).unwrap();
        ledger.reserve("key", 4, 9).unwrap();
        assert!(matches!(
            ledger.admit("key", 4, 9),
            Err(SecurityError::Replay)
        ));
        ReplayLedger::load(path)
            .unwrap()
            .admit("key", 4, 10)
            .unwrap();
    }

    #[test]
    fn generation_zero_never_reaches_a_sink() {
        let directory = tempfile::tempdir().unwrap();
        let ledger = ReplayLedger::load(directory.path().join("ledger.json")).unwrap();
        assert!(matches!(
            ledger.admit("key", 4, 0),
            Err(SecurityError::ZeroGeneration)
        ));
    }

    #[cfg(unix)]
    #[test]
    fn tls_private_key_reader_rejects_loose_files_and_symlinks() {
        use std::os::unix::fs::{symlink, PermissionsExt as _};

        let directory = tempfile::tempdir().unwrap();
        let key = directory.path().join("key.pem");
        std::fs::write(&key, b"private material").unwrap();
        std::fs::set_permissions(&key, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert!(matches!(
            read_private_key_file(&key),
            Err(SecurityError::Config(message)) if message.contains("group/other")
        ));

        std::fs::set_permissions(&key, std::fs::Permissions::from_mode(0o600)).unwrap();
        assert_eq!(read_private_key_file(&key).unwrap(), b"private material");

        let link = directory.path().join("key-link.pem");
        symlink(&key, &link).unwrap();
        assert!(matches!(
            read_private_key_file(&link),
            Err(SecurityError::Config(message)) if message.contains("non-symlink")
        ));
    }
}
