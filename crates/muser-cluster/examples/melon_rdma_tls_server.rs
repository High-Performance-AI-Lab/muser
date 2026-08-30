//! Validation ladder step 3: TLS-over-RDMA, server (Mac/receiver) side.
//! Confirms `rustls`'s handshake/verification logic is unaffected by
//! swapping `TcpStream` for `MelonRdmaStream` underneath `StreamOwned` —
//! same mTLS 1.3 + ALPN + leaf-pin logic as the real receiver, standalone
//! from muser's HMAC/replay-ledger/kvpack layers above it. The raw RDMA
//! byte-pipe itself was already proven separately (validation ladder step
//! 2, see `melon_rdma_loopback`); this only adds TLS on top of it.
//!
//! Usage: melon_rdma_tls_server <bootstrap_listen_addr> <cert_dir> <dev> <gid_index>
//! `cert_dir` must contain server.cert.pem, server.key.pem, ca.cert.pem, and
//! a `leaf_sha256_pins.txt` with one lowercase sha256 hex pin per line (the
//! peer/client leaf's pin).

use muser_cluster::security::rdma::accept_mtls_over_rdma;
use muser_cluster::security::TlsFiles;
use std::collections::BTreeSet;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;

const ALPN: &[u8] = b"melon-rdma-tls-selftest-v1";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!(
            "usage: {} <bootstrap_listen_addr> <cert_dir> <dev> <gid_index>",
            args[0]
        );
        std::process::exit(2);
    }
    let listen_addr = &args[1];
    let cert_dir = PathBuf::from(&args[2]);
    let dev = &args[3];
    let gid_index: i32 = args[4].parse().expect("gid_index must be an integer");

    let certificate_chain = cert_dir.join("server.cert.pem");
    let private_key = cert_dir.join("server.key.pem");
    let peer_ca = cert_dir.join("ca.cert.pem");
    let pins_text =
        std::fs::read_to_string(cert_dir.join("leaf_sha256_pins.txt")).expect("read pins file");
    let leaf_sha256_pins: BTreeSet<String> = pins_text
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();

    let listener = TcpListener::bind(listen_addr).expect("TCP bind failed");
    eprintln!("melon_rdma_tls_server: listening on {listen_addr} for the bootstrap connection");
    let (bootstrap, peer) = listener.accept().expect("TCP accept failed");
    eprintln!("melon_rdma_tls_server: bootstrap accepted from {peer}");

    let files = TlsFiles {
        certificate_chain: &certificate_chain,
        private_key: &private_key,
        peer_ca: &peer_ca,
        leaf_sha256_pins: &leaf_sha256_pins,
    };

    let mut stream = accept_mtls_over_rdma(bootstrap, files, ALPN, dev, gid_index)
        .expect("accept_mtls_over_rdma failed");
    eprintln!("melon_rdma_tls_server: RDMA-backed TLS handshake complete, ALPN/leaf-pin verified");

    let mut buf = [0u8; 64];
    let n = stream.read(&mut buf).expect("read failed");
    eprintln!(
        "melon_rdma_tls_server: received {n} bytes: {:?}",
        String::from_utf8_lossy(&buf[..n])
    );
    stream.write_all(b"ack").expect("write_all failed");
    println!("MELON_RDMA_TLS_SERVER PASS");
}
