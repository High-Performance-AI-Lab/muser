//! `muser-kvpack` — thin local wrapper around kvpack.
//!
//! kvpack is the **one** shared external dependency in `muser`
//! (docs/muser-architecture.md §1), pinned to a path-pinned release source —
//! the `release/muser-alpha2` branch's `kvpack-core`/`kvpack`/`kvpack-handoff`
//! crates at `0.1.0-alpha.2` (`docs/release-provenance.md`), not a live git
//! dependency. It defines the sealed V2 wire format (HMAC over a canonical
//! manifest) that the CUDA producer on GX10 and the Metal consumer on Mac
//! must agree on: the producer reimplements that format itself
//! (`scripts/gx10/llamacpp/spark_kv_export.cpp` + `muser_v2_send.py`) rather
//! than linking this crate, so agreement isn't "one format authority linked
//! on both sides" but cross-verification — the authenticated Mac-side
//! receiver rejects anything the producer's reimplementation gets wrong,
//! which is why HMAC verification on restore is load-bearing, not a
//! formality. Everything else in `muser` duplicates Ferrite; kvpack does
//! not, because vendoring it would fork a format two independent
//! implementations must keep agreeing on.
//!
//! Already landed upstream (`github.com/High-Performance-AI-Lab/kvpack`):
//! Muse layout table K1 (NoPE
//! theta=0, fail-closed) + K3 (2-class 39-SWA/13-NoPE) — DONE, 5 tests.
//! Scalar-math identity K4 (qk_scale, output_mult, softcap, eps as f64
//! bits) — DONE, 17 tests, caught 2 real GGUF-vs-config regressions.
//! Session artifact K5 (13 NoPE planes as-is + 39 SWA windowed planes,
//! fail-closed resume) — DONE, 9 tests. Producer-side CUDA export — DONE,
//! byte-identical x2, proven on GX10.
//!
//! This crate re-exports the pinned-release API and adds three things that are
//! muser's own product surface, not upstream kvpack's job:
//! - the Muse K1/K3 layout table glue (`layout`)
//! - session save/restore + the relocation-as-memcpy helper (`session`)
//! - cache-economics accounting the dashboard renders (`economics`)

pub mod config;
pub mod economics;
pub mod layout;
pub mod remote;
pub mod resident;
pub mod reuse;
pub mod session;
