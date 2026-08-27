# `shaders/`

Metal shader sources (`.metal` files), embedded via `include_str!` and
runtime-compiled (`newLibraryWithSource`, PSO cached in
`metal::pso_cache`) — no Xcode step, no `.metallib` build artifact, pure
source checkout. See `docs/muser-architecture.md` §3 ("Build").

Empty until Phase 2 (`docs/muser-architecture.md`). Extraction manifest
(§C) for what lands here, from Ferrite's
`crates/ferrite-metal-shaders/shaders/`:

| shader | muse role | action |
|---|---|---|
| `rope.metal` (branch version) | NORM/GPT-J RoPE on SWA layers | PULL-CLEAN |
| `sigmoid_gate.metal` | attn-output sigmoid gate | PULL-CLEAN |
| `rms_norm_llamacpp.metal`, `fused_residual_rms_norm_llamacpp.metal`, `rms_norm_per_head.metal` | sandwich norms + per-head QK-norm | PULL-AND-SIMPLIFY |
| `flash_attn_ext_vec_llamacpp.metal` (+ `_f16`, `_reduce`) | flash decode, split-K | PULL-AND-SIMPLIFY (dk256 -> dk128) |
| `flash_attn_decode_vec_f16_gqa_lazyrope.metal`, `flash_attn_decode_vec_geop_dense_gqa.metal` | GQA head-packing decode | PULL-AND-SIMPLIFY |
| `flash_attn_prefill_q4_dk64.metal` | Mac-local prefill flash | PULL-AND-SIMPLIFY (-> dk128) |
| `matmul_q4k_*.metal`, `ffn_fused*.metal`, `embed_gather.metal`, `matvec_q6k_llama.metal` | proj/FFN/LM-head/embed | PULL-AND-SIMPLIFY |

See `docs/extraction-manifest.md` (or `docs/muser-architecture.md` §2) for
the full table with source paths.
