#include <metal_stdlib>
using namespace metal;

// ── rope_inplace ───────────────────────────────────────────────────────────
//
// Rotary Position Embedding applied in-place to Q and K tensors.
// Uses NEOX (half-split) convention: pairs element i with i + head_dim/2.
//
// For each (head h, pair index i):
//   θ  = pos / rope_base^(2i / head_dim)
//   q' = [ q[i]·cos(θ) − q[i+hd/2]·sin(θ),  q[i]·sin(θ) + q[i+hd/2]·cos(θ) ]
//
// Thread assignment: one thread per (head, pair).
//   gid ∈ [0, n_heads * half_hd)         → Q heads
//   gid ∈ [n_heads * half_hd, total)     → K heads (up to n_kv_heads)
//
// dispatch_threads( ((n_heads + n_kv_heads) * head_dim/2, 1, 1), (32, 1, 1) )
//   — or — dispatch_thread_groups with ceil division.
//
kernel void rope_inplace(
    device       float* q          [[ buffer(0) ]],   // [n_heads   × head_dim]
    device       float* k          [[ buffer(1) ]],   // [n_kv_heads × head_dim]
    constant     uint&  n_heads    [[ buffer(2) ]],
    constant     uint&  n_kv_heads [[ buffer(3) ]],
    constant     uint&  head_dim   [[ buffer(4) ]],
    constant     uint&  pos        [[ buffer(5) ]],
    constant     float& rope_base  [[ buffer(6) ]],
    uint gid [[ thread_position_in_grid ]])
{
    uint half_hd      = head_dim / 2;
    uint total_q_pairs = n_heads    * half_hd;
    uint total_pairs   = total_q_pairs + n_kv_heads * half_hd;

    if (gid >= total_pairs) return;

    bool is_q   = (gid < total_q_pairs);
    uint local  = is_q ? gid : (gid - total_q_pairs);
    uint head   = local / half_hd;
    uint pair_i = local % half_hd;

    float freq  = 1.0f / pow(rope_base, 2.0f * float(pair_i) / float(head_dim));
    float angle = float(pos) * freq;
    float cos_a = precise::cos(angle);
    float sin_a = precise::sin(angle);

    // NEOX convention: pair element i with element i + half_hd
    device float* base = is_q
        ? (q + head * head_dim)
        : (k + head * head_dim);

    float v0 = base[pair_i];
    float v1 = base[pair_i + half_hd];
    base[pair_i]           = v0 * cos_a - v1 * sin_a;
    base[pair_i + half_hd] = v0 * sin_a + v1 * cos_a;
}

// ── rope_batch ────────────────────────────────────────────────────────────
//
// Batched RoPE applied in-place to Q[B,n_heads,head_dim] and K[B,n_kv,hd].
// Each token b uses position (start_pos + b).
//
// dispatch: (ceil(total_pairs/32), B, 1) × (32, 1, 1)
//   — 32 threads per threadgroup, each handles one pair via stride loop
//
kernel void rope_batch(
    device       float* Q          [[ buffer(0) ]],  // [B × n_heads × head_dim]
    device       float* K          [[ buffer(1) ]],  // [B × n_kv_heads × head_dim]
    constant     uint&  n_heads    [[ buffer(2) ]],
    constant     uint&  n_kv_heads [[ buffer(3) ]],
    constant     uint&  head_dim   [[ buffer(4) ]],
    constant     uint&  start_pos  [[ buffer(5) ]],
    constant     float& rope_base  [[ buffer(6) ]],
    uint2 tgid [[ threadgroup_position_in_grid ]],
    uint  lid  [[ thread_index_in_simdgroup ]])
{
    const uint batch = tgid.y;
    const uint pair_id = tgid.x * 32u + lid;

    const uint half_hd       = head_dim / 2u;
    const uint total_q_pairs = n_heads    * half_hd;
    const uint total_pairs   = total_q_pairs + n_kv_heads * half_hd;
    if (pair_id >= total_pairs) return;

    const uint pos = start_pos + batch;

    const bool is_q  = (pair_id < total_q_pairs);
    const uint local = is_q ? pair_id : (pair_id - total_q_pairs);
    const uint head  = local / half_hd;
    const uint pi    = local % half_hd;

    const uint q_stride = n_heads    * head_dim;
    const uint k_stride = n_kv_heads * head_dim;

    // NEOX convention: pair element pi with element pi + half_hd
    device float* base = is_q
        ? (Q + batch * q_stride + head * head_dim)
        : (K + batch * k_stride + head * head_dim);

    float freq  = 1.0f / pow(rope_base, 2.0f * float(pi) / float(head_dim));
    float angle = float(pos) * freq;
    float cos_a = precise::cos(angle);
    float sin_a = precise::sin(angle);

    float v0 = base[pi];
    float v1 = base[pi + half_hd];
    base[pi]           = v0 * cos_a - v1 * sin_a;
    base[pi + half_hd] = v0 * sin_a + v1 * cos_a;
}

// ── fused_bias_rope_batch ─────────────────────────────────────────────────
//
// Fused bias add (Q/K/V) + RoPE (Q/K only) in ONE dispatch.
// Replaces 3 × broadcast_bias_add + rope_batch → saves 3 dispatches + 1 barrier.
//
// Mathematical insight: bias + rotation are both element-wise on the same
// activation buffers.  Fusing avoids 3 extra memory round-trips and
// the synchronisation barrier between bias-add and RoPE.
//
// Thread layout (per batch token):
//   gid [0, total_q_pairs)             → Q: add bias, then RoPE rotate
//   gid [total_q_pairs, total_pairs)   → K: add bias, then RoPE rotate
//   gid [total_pairs, total_pairs+kv_dim) → V: add bias only (no RoPE)
//
// The freq table (half_hd floats) is precomputed once at init:
//   freq[i] = 1.0 / pow(rope_base, 2.0 * i / head_dim)
// This eliminates per-thread pow() calls (1.57M per forward pass).
//
// dispatch: (ceil((total_pairs + kv_dim) / 32), B, 1) × (32, 1, 1)
//
kernel void fused_bias_rope_batch(
    device       float* Q          [[ buffer(0)  ]],  // [B × q_dim]   rw
    device       float* K          [[ buffer(1)  ]],  // [B × kv_dim]  rw
    device       float* V          [[ buffer(2)  ]],  // [B × kv_dim]  rw
    device const float* q_bias     [[ buffer(3)  ]],  // [q_dim]
    device const float* k_bias     [[ buffer(4)  ]],  // [kv_dim]
    device const float* v_bias     [[ buffer(5)  ]],  // [kv_dim]
    device const float* freq_table [[ buffer(6)  ]],  // [half_hd]  precomputed
    constant     uint&  n_heads    [[ buffer(7)  ]],
    constant     uint&  n_kv_heads [[ buffer(8)  ]],
    constant     uint&  head_dim   [[ buffer(9)  ]],
    constant     uint&  start_pos  [[ buffer(10) ]],
    uint2 tgid [[ threadgroup_position_in_grid ]],
    uint  lid  [[ thread_index_in_simdgroup ]])
{
    const uint batch = tgid.y;
    const uint gid   = tgid.x * 32u + lid;

    const uint half_hd       = head_dim / 2u;
    const uint q_dim         = n_heads    * head_dim;
    const uint kv_dim        = n_kv_heads * head_dim;
    const uint total_q_pairs = n_heads    * half_hd;
    const uint total_pairs   = total_q_pairs + n_kv_heads * half_hd;
    const uint pos           = start_pos + batch;

    if (gid < total_q_pairs) {
        // ── Q: add bias + RoPE (NEOX: pair i with i+half_hd) ──
        const uint head = gid / half_hd;
        const uint pi   = gid % half_hd;
        device float* base = Q + batch * q_dim + head * head_dim;
        const uint bias_off_lo = head * head_dim + pi;
        const uint bias_off_hi = head * head_dim + pi + half_hd;
        float v0 = base[pi]           + q_bias[bias_off_lo];
        float v1 = base[pi + half_hd] + q_bias[bias_off_hi];
        float angle = float(pos) * freq_table[pi];
        float cos_a = precise::cos(angle);
        float sin_a = precise::sin(angle);
        base[pi]           = v0 * cos_a - v1 * sin_a;
        base[pi + half_hd] = v0 * sin_a + v1 * cos_a;
    } else if (gid < total_pairs) {
        // ── K: add bias + RoPE (NEOX: pair i with i+half_hd) ──
        const uint local = gid - total_q_pairs;
        const uint head  = local / half_hd;
        const uint pi    = local % half_hd;
        device float* base = K + batch * kv_dim + head * head_dim;
        const uint bias_off_lo = head * head_dim + pi;
        const uint bias_off_hi = head * head_dim + pi + half_hd;
        float v0 = base[pi]           + k_bias[bias_off_lo];
        float v1 = base[pi + half_hd] + k_bias[bias_off_hi];
        float angle = float(pos) * freq_table[pi];
        float cos_a = precise::cos(angle);
        float sin_a = precise::sin(angle);
        base[pi]           = v0 * cos_a - v1 * sin_a;
        base[pi + half_hd] = v0 * sin_a + v1 * cos_a;
    } else if (gid < total_pairs + kv_dim) {
        // ── V: add bias only (no rotation) ──
        const uint vi = gid - total_pairs;
        V[batch * kv_dim + vi] += v_bias[vi];
    }
}

// ── fused_bias_q_rope_batch (A.3.g lazy-RoPE CF2 variant) ─────────────────
//
// Same as `fused_bias_rope_batch` except RoPE is applied ONLY to Q; K gets
// bias but NO rotation. V is bias-only (same as classical). No F16 cache
// store (unlike `fused_bias_q_rope_store_kv_batch`) — this variant is for
// the CF2 pure-Q4 prefill preamble where no F16 cache is allocated.
//
// Used by: engine_prefill/attention_routing.rs under CF2 mode when
// `config::use_lazy_rope()` is true. Leaves `batch.k` holding PRE-RoPE K
// so the downstream `flash_attn_prefill_q4_lazyrope` kernel can quantise
// pre-RoPE K into the Q4 cache and apply RoPE inline at read time.
kernel void fused_bias_q_rope_batch(
    device       float* Q          [[ buffer(0)  ]],
    device       float* K          [[ buffer(1)  ]],
    device       float* V          [[ buffer(2)  ]],
    device const float* q_bias     [[ buffer(3)  ]],
    device const float* k_bias     [[ buffer(4)  ]],
    device const float* v_bias     [[ buffer(5)  ]],
    device const float* freq_table [[ buffer(6)  ]],
    constant     uint&  n_heads    [[ buffer(7)  ]],
    constant     uint&  n_kv_heads [[ buffer(8)  ]],
    constant     uint&  head_dim   [[ buffer(9)  ]],
    constant     uint&  start_pos  [[ buffer(10) ]],
    uint2 tgid [[ threadgroup_position_in_grid ]],
    uint  lid  [[ thread_index_in_simdgroup ]])
{
    const uint batch = tgid.y;
    const uint gid   = tgid.x * 32u + lid;

    const uint half_hd       = head_dim / 2u;
    const uint q_dim         = n_heads    * head_dim;
    const uint kv_dim        = n_kv_heads * head_dim;
    const uint total_q_pairs = n_heads    * half_hd;
    const uint total_pairs   = total_q_pairs + n_kv_heads * half_hd;
    const uint pos           = start_pos + batch;

    if (gid < total_q_pairs) {
        // Q: bias + RoPE.
        const uint head = gid / half_hd;
        const uint pi   = gid % half_hd;
        device float* base = Q + batch * q_dim + head * head_dim;
        const uint bias_off_lo = head * head_dim + pi;
        const uint bias_off_hi = head * head_dim + pi + half_hd;
        float v0 = base[pi]           + q_bias[bias_off_lo];
        float v1 = base[pi + half_hd] + q_bias[bias_off_hi];
        float angle = float(pos) * freq_table[pi];
        float cos_a = precise::cos(angle);
        float sin_a = precise::sin(angle);
        base[pi]           = v0 * cos_a - v1 * sin_a;
        base[pi + half_hd] = v0 * sin_a + v1 * cos_a;
    } else if (gid < total_pairs) {
        // K: bias only — NO RoPE. batch.k holds pre-RoPE K.
        const uint local = gid - total_q_pairs;
        const uint head  = local / half_hd;
        const uint pi    = local % half_hd;
        device float* base = K + batch * kv_dim + head * head_dim;
        const uint bias_off_lo = head * head_dim + pi;
        const uint bias_off_hi = head * head_dim + pi + half_hd;
        base[pi]           = base[pi]           + k_bias[bias_off_lo];
        base[pi + half_hd] = base[pi + half_hd] + k_bias[bias_off_hi];
    } else if (gid < total_pairs + kv_dim) {
        // V: bias only.
        const uint vi = gid - total_pairs;
        V[batch * kv_dim + vi] += v_bias[vi];
    }
}

// ── fused_bias_rope_store_kv_batch ────────────────────────────────────────
//
// Same as fused_bias_rope_batch but ALSO writes K,V to the f16 KV cache.
// Eliminates the separate store_kv_batch_f16 dispatches + barrier before
// flash_attn_v2, saving 1 barrier + 4 dispatches per layer.
//
// dispatch: (ceil((total_pairs + kv_dim) / 32), B, 1) × (32, 1, 1)
//
kernel void fused_bias_rope_store_kv_batch(
    device       float* Q          [[ buffer(0)  ]],  // [B × q_dim]   rw
    device       float* K          [[ buffer(1)  ]],  // [B × kv_dim]  rw
    device       float* V          [[ buffer(2)  ]],  // [B × kv_dim]  rw
    device const float* q_bias     [[ buffer(3)  ]],  // [q_dim]
    device const float* k_bias     [[ buffer(4)  ]],  // [kv_dim]
    device const float* v_bias     [[ buffer(5)  ]],  // [kv_dim]
    device const float* freq_table [[ buffer(6)  ]],  // [half_hd]
    device       half*  K_cache    [[ buffer(7)  ]],  // [n_kv_heads × max_seq × DK]
    device       half*  V_cache    [[ buffer(8)  ]],  // [n_kv_heads × max_seq × DK]
    constant     uint&  n_heads    [[ buffer(9)  ]],
    constant     uint&  n_kv_heads [[ buffer(10) ]],
    constant     uint&  head_dim   [[ buffer(11) ]],
    constant     uint&  start_pos  [[ buffer(12) ]],
    constant     uint&  cache_stride [[ buffer(13) ]], // max_seq × head_dim (f16 elems)
    uint2 tgid [[ threadgroup_position_in_grid ]],
    uint  lid  [[ thread_index_in_simdgroup ]])
{
    const uint batch = tgid.y;
    const uint gid   = tgid.x * 32u + lid;

    const uint half_hd       = head_dim / 2u;
    const uint q_dim         = n_heads    * head_dim;
    const uint kv_dim        = n_kv_heads * head_dim;
    const uint total_q_pairs = n_heads    * half_hd;
    const uint total_pairs   = total_q_pairs + n_kv_heads * half_hd;
    const uint pos           = start_pos + batch;

    if (gid < total_q_pairs) {
        // ── Q: add bias + RoPE (NEOX: pair i with i+half_hd) ──
        const uint head = gid / half_hd;
        const uint pi   = gid % half_hd;
        device float* base = Q + batch * q_dim + head * head_dim;
        const uint bias_off_lo = head * head_dim + pi;
        const uint bias_off_hi = head * head_dim + pi + half_hd;
        float v0 = base[pi]           + q_bias[bias_off_lo];
        float v1 = base[pi + half_hd] + q_bias[bias_off_hi];
        float angle = float(pos) * freq_table[pi];
        float cos_a = precise::cos(angle);
        float sin_a = precise::sin(angle);
        base[pi]           = v0 * cos_a - v1 * sin_a;
        base[pi + half_hd] = v0 * sin_a + v1 * cos_a;
    } else if (gid < total_pairs) {
        // ── K: add bias + RoPE + store to f16 cache (NEOX) ──
        const uint local = gid - total_q_pairs;
        const uint head  = local / half_hd;
        const uint pi    = local % half_hd;
        device float* base = K + batch * kv_dim + head * head_dim;
        const uint bias_off_lo = head * head_dim + pi;
        const uint bias_off_hi = head * head_dim + pi + half_hd;
        float v0 = base[pi]           + k_bias[bias_off_lo];
        float v1 = base[pi + half_hd] + k_bias[bias_off_hi];
        float angle = float(pos) * freq_table[pi];
        float cos_a = precise::cos(angle);
        float sin_a = precise::sin(angle);
        float k0 = v0 * cos_a - v1 * sin_a;
        float k1 = v0 * sin_a + v1 * cos_a;
        base[pi]           = k0;
        base[pi + half_hd] = k1;
        // Write to f16 KV cache (NEOX layout: i and i+half_hd)
        const uint cache_base = head * cache_stride + pos * head_dim;
        K_cache[cache_base + pi]           = half(k0);
        K_cache[cache_base + pi + half_hd] = half(k1);
    } else if (gid < total_pairs + kv_dim) {
        // ── V: add bias + store to f16 cache ──
        const uint vi     = gid - total_pairs;
        const uint v_head = vi / head_dim;
        const uint v_off  = vi % head_dim;
        float val = V[batch * kv_dim + vi] + v_bias[vi];
        V[batch * kv_dim + vi] = val;
        const uint cache_off = v_head * cache_stride + pos * head_dim + v_off;
        V_cache[cache_off] = half(val);
    }
}

// ── rope_store_kv_batch_cached ───────────────────────────────────────────
//
// Same as rope_batch_cached but ALSO writes K,V to the f16 KV cache.
// For models without biases (LLaMA, Mistral).
// V is NOT rotated, just copied f32→f16 to cache.
//
// dispatch: (ceil((total_pairs + kv_dim) / 32), B, 1) × (32, 1, 1)
//   where total_pairs = (n_heads + n_kv_heads) * (head_dim / 2)
//
kernel void rope_store_kv_batch_cached(
    device       float* Q          [[ buffer(0)  ]],
    device       float* K          [[ buffer(1)  ]],
    device const float* V          [[ buffer(2)  ]],  // read-only: no bias
    device const float* freq_table [[ buffer(3)  ]],  // [half_hd]
    device       half*  K_cache    [[ buffer(4)  ]],  // [n_kv_heads × max_seq × DK]
    device       half*  V_cache    [[ buffer(5)  ]],  // [n_kv_heads × max_seq × DK]
    constant     uint&  n_heads    [[ buffer(6)  ]],
    constant     uint&  n_kv_heads [[ buffer(7)  ]],
    constant     uint&  head_dim   [[ buffer(8)  ]],
    constant     uint&  start_pos  [[ buffer(9)  ]],
    constant     uint&  cache_stride [[ buffer(10) ]], // max_seq × head_dim
    uint2 tgid [[ threadgroup_position_in_grid ]],
    uint  lid  [[ thread_index_in_simdgroup ]])
{
    const uint batch   = tgid.y;
    const uint pair_id = tgid.x * 32u + lid;

    const uint half_hd       = head_dim / 2u;
    const uint kv_dim        = n_kv_heads * head_dim;
    const uint total_q_pairs = n_heads    * half_hd;
    const uint total_pairs   = total_q_pairs + n_kv_heads * half_hd;
    const uint pos           = start_pos + batch;

    if (pair_id < total_q_pairs) {
        // ── Q: RoPE only (NEOX: pair i with i+half_hd) ──
        const uint head = pair_id / half_hd;
        const uint pi   = pair_id % half_hd;
        const uint q_stride = n_heads * head_dim;
        device float* base = Q + batch * q_stride + head * head_dim;
        float angle = float(pos) * freq_table[pi];
        float cos_a = precise::cos(angle);
        float sin_a = precise::sin(angle);
        float v0 = base[pi], v1 = base[pi + half_hd];
        base[pi]           = v0 * cos_a - v1 * sin_a;
        base[pi + half_hd] = v0 * sin_a + v1 * cos_a;
    } else if (pair_id < total_pairs) {
        // ── K: RoPE + store to f16 cache (NEOX) ──
        const uint local = pair_id - total_q_pairs;
        const uint head  = local / half_hd;
        const uint pi    = local % half_hd;
        device float* base = K + batch * kv_dim + head * head_dim;
        float angle = float(pos) * freq_table[pi];
        float cos_a = precise::cos(angle);
        float sin_a = precise::sin(angle);
        float v0 = base[pi], v1 = base[pi + half_hd];
        float k0 = v0 * cos_a - v1 * sin_a;
        float k1 = v0 * sin_a + v1 * cos_a;
        base[pi]           = k0;
        base[pi + half_hd] = k1;
        const uint cache_base = head * cache_stride + pos * head_dim;
        K_cache[cache_base + pi]           = half(k0);
        K_cache[cache_base + pi + half_hd] = half(k1);
    } else if (pair_id < total_pairs + kv_dim) {
        // ── V: f32→f16 copy to cache (no rotation) ──
        const uint vi     = pair_id - total_pairs;
        const uint v_head = vi / head_dim;
        const uint v_off  = vi % head_dim;
        float val = V[batch * kv_dim + vi];
        const uint cache_off = v_head * cache_stride + pos * head_dim + v_off;
        V_cache[cache_off] = half(val);
    }
}

// ── fused_bias_q_rope_store_kv_batch  (lazy-RoPE variant) ────────────────
//
// Same dispatch shape as fused_bias_rope_store_kv_batch but K stays
// PRE-RoPE in the cache. Q is rotated (so Q·K at attention time picks
// up the rotation), K and V are bias-added and stored raw.
//
// Why: under lazy-RoPE, the K cache holds pre-RoPE K values.
// Attention-time kernels then compute R(q_pos - k_pos) · K inline
// during the Q·K dot. The pre-RoPE K distribution is position-invariant,
// enabling stable per-layer subspace bases (Phase 4d L4a Grassmannian
// codec) — learned bases are meaningful only when the input distribution
// doesn't drift with position.
//
// Dispatch: IDENTICAL to fused_bias_rope_store_kv_batch. Only the
// K branch differs (skip the rotation math, store raw).
kernel void fused_bias_q_rope_store_kv_batch(
    device       float* Q          [[ buffer(0)  ]],
    device       float* K          [[ buffer(1)  ]],
    device       float* V          [[ buffer(2)  ]],
    device const float* q_bias     [[ buffer(3)  ]],
    device const float* k_bias     [[ buffer(4)  ]],
    device const float* v_bias     [[ buffer(5)  ]],
    device const float* freq_table [[ buffer(6)  ]],
    device       half*  K_cache    [[ buffer(7)  ]],
    device       half*  V_cache    [[ buffer(8)  ]],
    constant     uint&  n_heads    [[ buffer(9)  ]],
    constant     uint&  n_kv_heads [[ buffer(10) ]],
    constant     uint&  head_dim   [[ buffer(11) ]],
    constant     uint&  start_pos  [[ buffer(12) ]],
    constant     uint&  cache_stride [[ buffer(13) ]],
    uint2 tgid [[ threadgroup_position_in_grid ]],
    uint  lid  [[ thread_index_in_simdgroup ]])
{
    const uint batch = tgid.y;
    const uint gid   = tgid.x * 32u + lid;

    const uint half_hd       = head_dim / 2u;
    const uint q_dim         = n_heads    * head_dim;
    const uint kv_dim        = n_kv_heads * head_dim;
    const uint total_q_pairs = n_heads    * half_hd;
    const uint total_pairs   = total_q_pairs + n_kv_heads * half_hd;
    const uint pos           = start_pos + batch;

    if (gid < total_q_pairs) {
        // Q: bias + RoPE (same as classical)
        const uint head = gid / half_hd;
        const uint pi   = gid % half_hd;
        device float* base = Q + batch * q_dim + head * head_dim;
        const uint bias_off_lo = head * head_dim + pi;
        const uint bias_off_hi = head * head_dim + pi + half_hd;
        float v0 = base[pi]           + q_bias[bias_off_lo];
        float v1 = base[pi + half_hd] + q_bias[bias_off_hi];
        float angle = float(pos) * freq_table[pi];
        float cos_a = precise::cos(angle);
        float sin_a = precise::sin(angle);
        base[pi]           = v0 * cos_a - v1 * sin_a;
        base[pi + half_hd] = v0 * sin_a + v1 * cos_a;
    } else if (gid < total_pairs) {
        // K: bias only — NO RoPE. Store raw pre-RoPE K to cache.
        const uint local = gid - total_q_pairs;
        const uint head  = local / half_hd;
        const uint pi    = local % half_hd;
        device float* base = K + batch * kv_dim + head * head_dim;
        const uint bias_off_lo = head * head_dim + pi;
        const uint bias_off_hi = head * head_dim + pi + half_hd;
        float k0 = base[pi]           + k_bias[bias_off_lo];
        float k1 = base[pi + half_hd] + k_bias[bias_off_hi];
        base[pi]           = k0;
        base[pi + half_hd] = k1;
        const uint cache_base = head * cache_stride + pos * head_dim;
        K_cache[cache_base + pi]           = half(k0);
        K_cache[cache_base + pi + half_hd] = half(k1);
    } else if (gid < total_pairs + kv_dim) {
        // V: bias + store (same as classical)
        const uint vi     = gid - total_pairs;
        const uint v_head = vi / head_dim;
        const uint v_off  = vi % head_dim;
        float val = V[batch * kv_dim + vi] + v_bias[vi];
        V[batch * kv_dim + vi] = val;
        const uint cache_off = v_head * cache_stride + pos * head_dim + v_off;
        V_cache[cache_off] = half(val);
    }
}

// ── q_rope_store_kv_batch_cached  (lazy-RoPE, no-bias variant) ───────────
//
// Same as rope_store_kv_batch_cached but K stays PRE-RoPE in the cache.
// For bias-less models (LLaMA, Mistral) in lazy-RoPE mode.
kernel void q_rope_store_kv_batch_cached(
    device       float* Q          [[ buffer(0)  ]],
    device const float* K          [[ buffer(1)  ]],
    device const float* V          [[ buffer(2)  ]],
    device const float* freq_table [[ buffer(3)  ]],
    device       half*  K_cache    [[ buffer(4)  ]],
    device       half*  V_cache    [[ buffer(5)  ]],
    constant     uint&  n_heads    [[ buffer(6)  ]],
    constant     uint&  n_kv_heads [[ buffer(7)  ]],
    constant     uint&  head_dim   [[ buffer(8)  ]],
    constant     uint&  start_pos  [[ buffer(9)  ]],
    constant     uint&  cache_stride [[ buffer(10) ]],
    uint2 tgid [[ threadgroup_position_in_grid ]],
    uint  lid  [[ thread_index_in_simdgroup ]])
{
    const uint batch   = tgid.y;
    const uint pair_id = tgid.x * 32u + lid;

    const uint half_hd       = head_dim / 2u;
    const uint kv_dim        = n_kv_heads * head_dim;
    const uint total_q_pairs = n_heads    * half_hd;
    const uint total_pairs   = total_q_pairs + n_kv_heads * half_hd;
    const uint pos           = start_pos + batch;

    if (pair_id < total_q_pairs) {
        // Q: RoPE only (same as classical)
        const uint head = pair_id / half_hd;
        const uint pi   = pair_id % half_hd;
        const uint q_stride = n_heads * head_dim;
        device float* base = Q + batch * q_stride + head * head_dim;
        float angle = float(pos) * freq_table[pi];
        float cos_a = precise::cos(angle);
        float sin_a = precise::sin(angle);
        float v0 = base[pi], v1 = base[pi + half_hd];
        base[pi]           = v0 * cos_a - v1 * sin_a;
        base[pi + half_hd] = v0 * sin_a + v1 * cos_a;
    } else if (pair_id < total_pairs) {
        // K: f32→f16 copy, NO RoPE. Store raw pre-RoPE K to cache.
        const uint local = pair_id - total_q_pairs;
        const uint head  = local / half_hd;
        const uint pi    = local % half_hd;
        float k0 = K[batch * kv_dim + head * head_dim + pi];
        float k1 = K[batch * kv_dim + head * head_dim + pi + half_hd];
        const uint cache_base = head * cache_stride + pos * head_dim;
        K_cache[cache_base + pi]           = half(k0);
        K_cache[cache_base + pi + half_hd] = half(k1);
    } else if (pair_id < total_pairs + kv_dim) {
        // V: f32→f16 copy (same as classical)
        const uint vi     = pair_id - total_pairs;
        const uint v_head = vi / head_dim;
        const uint v_off  = vi % head_dim;
        float val = V[batch * kv_dim + vi];
        const uint cache_off = v_head * cache_stride + pos * head_dim + v_off;
        V_cache[cache_off] = half(val);
    }
}

// ── rope_batch_cached ────────────────────────────────────────────────────
//
// Same as rope_batch but uses precomputed freq_table instead of pow().
// For models without biases (LLaMA), this is the fast path.
//
// dispatch: (ceil(total_pairs/32), B, 1) × (32, 1, 1)
//
kernel void rope_batch_cached(
    device       float* Q          [[ buffer(0) ]],
    device       float* K          [[ buffer(1) ]],
    device const float* freq_table [[ buffer(2) ]],  // [half_hd]
    constant     uint&  n_heads    [[ buffer(3) ]],
    constant     uint&  n_kv_heads [[ buffer(4) ]],
    constant     uint&  head_dim   [[ buffer(5) ]],
    constant     uint&  start_pos  [[ buffer(6) ]],
    uint2 tgid [[ threadgroup_position_in_grid ]],
    uint  lid  [[ thread_index_in_simdgroup ]])
{
    const uint batch   = tgid.y;
    const uint pair_id = tgid.x * 32u + lid;

    const uint half_hd       = head_dim / 2u;
    const uint total_q_pairs = n_heads    * half_hd;
    const uint total_pairs   = total_q_pairs + n_kv_heads * half_hd;
    if (pair_id >= total_pairs) return;

    const uint pos = start_pos + batch;

    const bool is_q  = (pair_id < total_q_pairs);
    const uint local = is_q ? pair_id : (pair_id - total_q_pairs);
    const uint head  = local / half_hd;
    const uint pi    = local % half_hd;

    const uint q_stride = n_heads    * head_dim;
    const uint k_stride = n_kv_heads * head_dim;

    // NEOX convention: pair element i with element i + half_hd
    device float* base = is_q
        ? (Q + batch * q_stride + head * head_dim)
        : (K + batch * k_stride + head * head_dim);

    float angle = float(pos) * freq_table[pi];
    float cos_a = precise::cos(angle);
    float sin_a = precise::sin(angle);

    float v0 = base[pi];
    float v1 = base[pi + half_hd];
    base[pi]           = v0 * cos_a - v1 * sin_a;
    base[pi + half_hd] = v0 * sin_a + v1 * cos_a;
}

// ── NORM-convention RoPE (LLAMA_ROPE_TYPE_NORM) ─────────────────────────
//
// Every other kernel in this file uses the NEOX convention: rotate the pair
// (x[i], x[i + half_hd]). llama.cpp's `rope_norm` instead rotates the
// *interleaved* pair (x[2i], x[2i+1]). Both read the SAME frequency table —
// freq[j] = base^(-2j/head_dim) — so only the element pairing differs, and a
// model that needs one and gets the other still produces fluent text with
// silently wrong positions. That is why these are separate kernels rather
// than a runtime flag threaded through the NEOX ones.
//
// Muse Glimmer is the first NORM-rope architecture in the tree. Its GGUF
// converter un-permutes Q/K at conversion time precisely so the interleaved
// form is the correct one for the stored weights.

kernel void rope_norm_batch_cached(
    device       float* Q          [[ buffer(0) ]],
    device       float* K          [[ buffer(1) ]],
    device const float* freq_table [[ buffer(2) ]],  // [half_hd]
    constant     uint&  n_heads    [[ buffer(3) ]],
    constant     uint&  n_kv_heads [[ buffer(4) ]],
    constant     uint&  head_dim   [[ buffer(5) ]],
    constant     uint&  start_pos  [[ buffer(6) ]],
    uint2 tgid [[ threadgroup_position_in_grid ]],
    uint  lid  [[ thread_index_in_simdgroup ]])
{
    const uint batch   = tgid.y;
    const uint pair_id = tgid.x * 32u + lid;

    const uint half_hd       = head_dim / 2u;
    const uint total_q_pairs = n_heads    * half_hd;
    const uint total_pairs   = total_q_pairs + n_kv_heads * half_hd;
    if (pair_id >= total_pairs) return;

    const uint pos = start_pos + batch;

    const bool is_q  = (pair_id < total_q_pairs);
    const uint local = is_q ? pair_id : (pair_id - total_q_pairs);
    const uint head  = local / half_hd;
    const uint pi    = local % half_hd;

    const uint q_stride = n_heads    * head_dim;
    const uint k_stride = n_kv_heads * head_dim;

    device float* base = is_q
        ? (Q + batch * q_stride + head * head_dim)
        : (K + batch * k_stride + head * head_dim);

    float angle = float(pos) * freq_table[pi];
    float cos_a = precise::cos(angle);
    float sin_a = precise::sin(angle);

    // NORM convention: the pair is adjacent, at 2*pi and 2*pi + 1.
    const uint i0 = 2u * pi;
    float v0 = base[i0];
    float v1 = base[i0 + 1u];
    base[i0]      = v0 * cos_a - v1 * sin_a;
    base[i0 + 1u] = v0 * sin_a + v1 * cos_a;
}

kernel void rope_norm_store_kv_batch_cached(
    device       float* Q            [[ buffer(0)  ]],
    device       float* K            [[ buffer(1)  ]],
    device const float* V            [[ buffer(2)  ]],  // read-only: no bias
    device const float* freq_table   [[ buffer(3)  ]],  // [half_hd]
    device       half*  K_cache      [[ buffer(4)  ]],
    device       half*  V_cache      [[ buffer(5)  ]],
    constant     uint&  n_heads      [[ buffer(6)  ]],
    constant     uint&  n_kv_heads   [[ buffer(7)  ]],
    constant     uint&  head_dim     [[ buffer(8)  ]],
    constant     uint&  start_pos    [[ buffer(9)  ]],
    constant     uint&  cache_stride [[ buffer(10) ]],
    uint2 tgid [[ threadgroup_position_in_grid ]],
    uint  lid  [[ thread_index_in_simdgroup ]])
{
    const uint batch   = tgid.y;
    const uint pair_id = tgid.x * 32u + lid;

    const uint half_hd       = head_dim / 2u;
    const uint kv_dim        = n_kv_heads * head_dim;
    const uint total_q_pairs = n_heads    * half_hd;
    const uint total_pairs   = total_q_pairs + n_kv_heads * half_hd;
    const uint pos           = start_pos + batch;

    if (pair_id < total_q_pairs) {
        // ── Q: RoPE only (NORM: pair 2i with 2i+1) ──
        const uint head = pair_id / half_hd;
        const uint pi   = pair_id % half_hd;
        const uint q_stride = n_heads * head_dim;
        device float* base = Q + batch * q_stride + head * head_dim;
        float angle = float(pos) * freq_table[pi];
        float cos_a = precise::cos(angle);
        float sin_a = precise::sin(angle);
        const uint i0 = 2u * pi;
        float v0 = base[i0], v1 = base[i0 + 1u];
        base[i0]      = v0 * cos_a - v1 * sin_a;
        base[i0 + 1u] = v0 * sin_a + v1 * cos_a;
    } else if (pair_id < total_pairs) {
        // ── K: RoPE + store to f16 cache (NORM) ──
        const uint local = pair_id - total_q_pairs;
        const uint head  = local / half_hd;
        const uint pi    = local % half_hd;
        device float* base = K + batch * kv_dim + head * head_dim;
        float angle = float(pos) * freq_table[pi];
        float cos_a = precise::cos(angle);
        float sin_a = precise::sin(angle);
        const uint i0 = 2u * pi;
        float v0 = base[i0], v1 = base[i0 + 1u];
        float k0 = v0 * cos_a - v1 * sin_a;
        float k1 = v0 * sin_a + v1 * cos_a;
        base[i0]      = k0;
        base[i0 + 1u] = k1;
        const uint cache_base = head * cache_stride + pos * head_dim;
        K_cache[cache_base + i0]      = half(k0);
        K_cache[cache_base + i0 + 1u] = half(k1);
    } else if (pair_id < total_pairs + kv_dim) {
        // ── V: f32→f16 copy to cache (no rotation) ──
        const uint vi     = pair_id - total_pairs;
        const uint v_head = vi / head_dim;
        const uint v_off  = vi % head_dim;
        float val = V[batch * kv_dim + vi];
        const uint cache_off = v_head * cache_stride + pos * head_dim + v_off;
        V_cache[cache_off] = half(val);
    }
}
