//! Batched GPU prefill driver — muse-fixed, no VM. macOS-only.
//!
//! **REIMPLEMENTED** (docs/muser-architecture.md §D), replacing Ferrite's
//! `forward_gpu/engine_prefill/*`. The implementation lives beside decode in
//! `decode.rs` so both routes share the exact layer graph. It is batched
//! over T query positions, exploiting the same weight-row-reuse
//! `weights.rs` documents ("prefill of T tokens ≈ one token's DRAM
//! traffic"). Release throughput remains gated on the paired campaign.
//!
//! Also the Mac-local fallback path when GX10 disaggregated prefill
//! (`muser-cluster`) isn't available or isn't worth the wire hop for a
//! short prompt.
//!
//! Chunks retain their activation/token arenas and encode the full 52-layer
//! graph into one serial command encoder. Cache placement remains explicit in
//! logical/physical ring metadata and never derives placement from absolute
//! positions.
