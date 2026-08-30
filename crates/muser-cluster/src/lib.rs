//! `muser-cluster` — authenticated GX10 disaggregated prefill.
//!
//! Muser does **not** compile CUDA into its Mac binary. The release carries a
//! narrow llama.cpp GX10 adapter and resident muser-prefilld; this crate is
//! the shared control, transport, security, scheduling, and detached-cache
//! receiver implementation. Clean split: CUDA prefill -> authenticated
//! Handoff V2 tiles -> Metal scatter-on-arrival -> atomic commit -> Metal decode.
//!
//! Launch config: 1x M3 Ultra (decode) + 1x NVIDIA GX10/GB10 (prefill). The
//! receiver admits one producer at a time — one control endpoint, one HMAC key
//! id, and a replay ledger keyed per key id — so multiple concurrent prefill
//! nodes are roadmap, not this release.
//!
//! Transport: mTLS-TCP, with a release floor of 3.0 Gbps median effective
//! installed-payload throughput across three handoffs. RDMA/RoCE and the ATTO
//! TB5->100 GbE Mac upgrade are roadmap fast lanes with no code here; any
//! fabric figure for them is a vendor number, not a Muser measurement. Ships
//! the 13 NoPE tiles *during* prefill (position-free -> memcpy-relocatable),
//! and the matching SWA chunks with the last 2048 tokens of CUDA.
//! Gate: >=95% transfer hidden `[target]`.
//!
//! No TTFT multiplier is claimed until the frozen release matrix passes.

#![allow(dead_code)] // live paths are selected by server cluster configuration

/// Wire transport: mTLS-TCP framing and frame (de)serialization.
///
/// Source (PULL-AND-SIMPLIFY): the transport half of `kvpack-handoff`
/// (wire frames, TCP receiver, verify) plus Ferrite's
/// `main/spark_prefill*` GX10<->Mac disagg protocol.
pub mod transport;

/// Producer handshake and atomic authenticated receive.
///
/// Source (PULL-AND-SIMPLIFY): `main/spark_prefill*` (GX10<->Mac disagg
/// protocol) on the Ferrite side of the wire.
pub mod producer;

/// Shared live receiver used by serving and release qualification.
pub mod receiver;

/// Transfer amortization scheduling: stream the 13 NoPE tiles during
/// prefill, and ship each overlapping SWA chunk with those last tiles.
///
/// **No direct Ferrite source for the schedule itself** — Ferrite's
/// version is SPEC'D but NOT built per the catalogue (extends the 7B
/// streaming lane). muser-original, built to the spec in
/// `docs/muser-architecture.md` §3 "Transport".
pub mod schedule;

/// Muse-specific detached cache shadow used by the V2 receiver.
pub mod muse_sink;

/// TLS 1.3/mTLS/ALPN/leaf-pin establishment and durable replay admission.
pub mod security;

/// RDMA `Read + Write` byte-pipe (MelonDMA), used as a drop-in replacement
/// for `TcpStream` underneath the same mTLS/ALPN/leaf-pin/HMAC/replay-ledger
/// stack above. Only compiled with `--features melon-rdma`; a stock build
/// never touches it.
#[cfg(feature = "melon-rdma")]
pub mod melon_rdma;

/// Exact model/request/component admission wrapped around any V2 sink.
pub mod identity;

/// Strict on-disk receiver configuration for the remote-prefill lane.
pub mod config;

/// Authenticated request/receipt channel for the resident GX10 producer.
pub mod control;

/// Per-phase handoff timing evidence (socket drain / verify / install /
/// seal / commit) used by the link diagnostics.
pub mod phase;

/// Experimental authenticated round log for a remote speculative verifier.
/// Token transitions remain single-writer and ordered; immutable result
/// fragments may arrive in any order and become visible only at closure.
pub mod verifier;

/// Durable V2 remote-verifier transcript and renderer commit protocol.
/// Unlike the original bounded research skeleton, V2 carries reconstructible
/// token/RNG state, target-only signatures, terminal transitions, and a
/// filesystem WAL around staged renderer activation.
pub mod verifier_v2;

pub mod verifier_gateway;
/// Source-side authenticated result and Mirror-SD commit capabilities.
pub mod verifier_gateway_adapter;
/// Non-serving Rust gateway state machine over `verifier_v2`.
///
/// The codec and gateway core are intentionally separate from listener/mTLS
/// wiring so a loopback prototype cannot be mistaken for a deployed service.
pub mod verifier_gateway_codec;
