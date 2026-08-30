//! Standalone loopback correctness test for `melon_rdma::MelonRdmaStream`,
//! independent of muser's TLS/HMAC/replay-ledger stack — validation ladder
//! step 2 (see the RDMA transport plan). Run against
//! `scripts/gx10/llamacpp/melon_rdma_stream.py --listen ...` on the peer.
//!
//! Phased, one-directional-at-a-time protocol (mirrors the Python side):
//! send an 8-byte length, the payload, then a SHA-256 digest; only after
//! all of that is fully sent does the server send back a 1-byte ack.
//! Overlapping sends in both directions at once starves the small,
//! fixed-depth RX ring on whichever side hasn't started draining it yet —
//! a test-protocol hazard this phasing avoids, not a byte-pipe question.
//!
//! Usage (client role — this binary always connects out):
//!   melon_rdma_loopback <host:port> <dev> <gid_index> [payload_bytes]

use muser_cluster::melon_rdma::MelonRdmaStream;
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::os::fd::IntoRawFd;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: {} <host:port> <dev> <gid_index> [payload_bytes]", args[0]);
        std::process::exit(2);
    }
    let addr = &args[1];
    let dev = &args[2];
    let gid_index: i32 = args[3].parse().expect("gid_index must be an integer");
    let payload_bytes: usize = args
        .get(4)
        .map(|s| s.parse().expect("payload_bytes must be an integer"))
        .unwrap_or(64 * 1024 * 1024);

    eprintln!("melon_rdma_loopback: connecting bootstrap TCP to {addr}");
    let bootstrap = TcpStream::connect(addr).expect("bootstrap TCP connect failed");

    eprintln!("melon_rdma_loopback: opening RDMA pipe (dev={dev} gid_index={gid_index})");
    let mut stream = MelonRdmaStream::open(bootstrap.into_raw_fd(), dev, gid_index)
        .expect("MelonRdmaStream::open failed");
    eprintln!("melon_rdma_loopback: RDMA pipe activated (client side)");

    let payload: Vec<u8> = (0..payload_bytes).map(|i| (i % 251) as u8).collect();
    let digest = Sha256::digest(&payload);

    let started = Instant::now();
    stream
        .write_all(&(payload_bytes as u64).to_be_bytes())
        .expect("length write_all failed");
    stream.write_all(&payload).expect("payload write_all failed");
    stream.write_all(&digest).expect("digest write_all failed");

    let mut ack = [0u8; 1];
    stream.read_exact(&mut ack).expect("ack read_exact failed");
    let elapsed = started.elapsed();

    let ok = ack[0] == 1;
    let gbit_s = (payload_bytes as f64 * 8.0 / elapsed.as_secs_f64()) / 1e9;
    eprintln!(
        "melon_rdma_loopback: sent {payload_bytes} bytes in {:.3}s ({gbit_s:.3} Gbit/s) server_ack_ok={ok}",
        elapsed.as_secs_f64()
    );
    if !ok {
        panic!("server reported a digest mismatch — byte-pipe correctness FAILED");
    }
    println!("MELON_RDMA_LOOPBACK PASS");
}
