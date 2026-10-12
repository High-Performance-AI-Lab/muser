//! Correctness-first, Muse-fixed Metal decode driver.
//!
//! This is a direct transcription of `reference::MuseModel::forward`.  It is
//! projections use the mapped GGUF bytes directly, one serial encoder owns a
//! complete token/chunk, and cache ring placement is carried as explicit
//! logical/physical metadata.

use crate::cache::{
    le_bytes_to_u16s, u16s_to_le_bytes, write_f16_tile, CachePlaneSnapshot, PlaneEncoding,
    SessionCacheSnapshot,
};
use crate::config::{MuseConfig, MuseConfigError, MuseLayerKind};
use crate::gguf::GgmlType;
use crate::metal::buffer::{GpuBuffer, GpuByteView, GpuBytes, GpuHalfBuffer};
use crate::metal::context::{MetalContext, MetalError};
use crate::metal::encode::MetalKernels;
use crate::weights::{MuseWeights, TensorLayout};
use objc::{sel, sel_impl};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::{Deref, Range};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

thread_local! {
    static STREAM_DECODE_DIAGNOSTICS: std::cell::RefCell<Option<crate::api::DecodeDiagnostics>> =
        const { std::cell::RefCell::new(None) };
}

pub(crate) fn take_stream_decode_diagnostics() -> Option<crate::api::DecodeDiagnostics> {
    STREAM_DECODE_DIAGNOSTICS.with(|slot| slot.borrow_mut().take())
}

fn stream_decode_profile_enabled() -> bool {
    std::env::var("MUSER_STREAM_DECODE_PROFILE").as_deref() == Ok("1")
}

fn install_stream_decode_diagnostics(diagnostics: crate::api::DecodeDiagnostics) {
    STREAM_DECODE_DIAGNOSTICS.with(|slot| *slot.borrow_mut() = Some(diagnostics));
}

// llama.cpp's Metal `flash_attn_ext_vec` always launches `nwg = 32` and only
// grows simdgroups (1→4) once `2 * nwg * nsg * 32 < visible`. Ferrite's
// occupancy-first cap of 96 oversubscribed the 13 full/NoPE planes and is
// the depth-rent we lose to llama as context grows. Keep short-context
// `nwg = min(blocks, 32)` so TG512 does not pay empty workgroups.
pub(crate) const MAX_DECODE_SPLIT_WORKGROUPS: usize = 32;
const MAX_DFLASH_BLOCK: usize = 16;
// Match the accepted Muse/Ferrite prefill route and the pinned llama.cpp
// comparator's physical ubatch.  The first standalone extraction used 128,
// which reread every quantized projection four times for a 512-token prompt
// and erased the prefill parity established before extirpation.  Larger
// contexts remain bounded by this arena size and stream through it.
const PREFILL_BATCH_TOKENS: usize = 512;
const MAX_TEACHER_FORCED_TOKENS: usize = 64;

fn llama_fa_prefill_route_available(
    token_count: usize,
    has_llama_flash_attention: bool,
    explicitly_disabled: bool,
    cross_vendor: bool,
) -> bool {
    token_count >= 20 && has_llama_flash_attention && !explicitly_disabled && !cross_vendor
}

fn llama_vec_prefill_route_available(
    token_count: usize,
    capacity: usize,
    has_llama_flash_attention: bool,
    cross_vendor: bool,
) -> bool {
    token_count < 20 && capacity >= 32 && has_llama_flash_attention && !cross_vendor
}

fn retained_prompt_subranges(
    chunk_start: usize,
    chunk_end: usize,
    prefix_rows: usize,
    sink_rows: usize,
    window_rows: usize,
) -> Vec<(usize, usize, usize)> {
    let mut retained = Vec::with_capacity(2);
    let capacity = sink_rows.saturating_add(window_rows);
    let intervals = if prefix_rows <= capacity {
        [(0, prefix_rows), (0, 0)]
    } else {
        [
            (0, sink_rows.min(prefix_rows)),
            (prefix_rows - window_rows.min(prefix_rows), prefix_rows),
        ]
    };
    for (start, end) in intervals {
        let retained_start = chunk_start.max(start);
        let retained_end = chunk_end.min(end);
        if retained_start < retained_end {
            retained.push((
                retained_start - chunk_start,
                retained_start,
                retained_end - retained_start,
            ));
        }
    }
    retained
}

#[derive(Debug, thiserror::Error)]
pub enum MetalModelError {
    #[error(transparent)]
    Config(#[from] MuseConfigError),
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error("tensor {name} uses unsupported Metal dtype {dtype:?}")]
    UnsupportedDtype { name: String, dtype: GgmlType },
    #[error(
        "tensor {name} uses {dtype:?}, which requires the pinned llama.cpp Metal library; set MUSER_GGML_METALLIB"
    )]
    MissingProjectionKernel { name: String, dtype: GgmlType },
    #[error("Metal KV cache for layer {layer} expected logical position {expected}, got {got}")]
    CacheDiscontinuity {
        layer: usize,
        expected: usize,
        got: usize,
    },
    #[error("invalid Metal KV snapshot: {0}")]
    InvalidSnapshot(String),
}

#[derive(Clone)]
struct Projection {
    name: String,
    layout: TensorLayout,
}

impl Projection {
    fn load(weights: &MuseWeights, name: String) -> Result<Self, MetalModelError> {
        let layout = weights.layout(&name)?;
        if !matches!(
            layout.dtype,
            GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K | GgmlType::NVFP4_E2M1 | GgmlType::F16
        ) {
            return Err(MetalModelError::UnsupportedDtype {
                name,
                dtype: layout.dtype,
            });
        }
        Ok(Self { name, layout })
    }

    fn view<'a>(&self, mapped: &'a GpuBytes) -> GpuByteView<'a> {
        mapped
            .view(self.layout.file_offset, self.layout.byte_len)
            .unwrap_or_else(|| panic!("validated GGUF tensor {} left mapped file", self.name))
    }

    fn nvfp4_scale_view<'a>(&self, mapped: &'a GpuBytes) -> Option<GpuByteView<'a>> {
        let offset = self.layout.nvfp4_scale_offset?;
        Some(
            mapped
                .view(offset, self.layout.nvfp4_scale_len)
                .unwrap_or_else(|| panic!("validated NVFP4 scale {} left mapped file", self.name)),
        )
    }
}

#[derive(Clone)]
struct LayerWeights {
    attn_norm: GpuBuffer,
    q: Projection,
    k: Projection,
    v: Projection,
    gate: Projection,
    q_norm: GpuBuffer,
    k_norm: GpuBuffer,
    output: Projection,
    post_attn_norm: GpuBuffer,
    ffn_norm: GpuBuffer,
    ffn_gate: Projection,
    ffn_up: Projection,
    ffn_down: Projection,
    post_ffn_norm: GpuBuffer,
}

struct MetalKvPlane {
    key: GpuHalfBuffer,
    value: GpuHalfBuffer,
    capacity: usize,
    len: usize,
    origin_logical: usize,
    origin_physical: usize,
    head_major: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MetalKvSnapshot {
    n_past: usize,
    layers: Vec<MetalPlaneSnapshot>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct MetalPlaneSnapshot {
    origin_logical: usize,
    len: usize,
    capacity: usize,
    key_logical: Vec<u16>,
    value_logical: Vec<u16>,
    head_major: bool,
}

/// Lightweight transactional checkpoint for one speculative verification
/// block. Growing NoPE planes only need their logical metadata rewound. SWA
/// planes may overwrite live ring rows, so the small set of destinations
/// touched by the candidate block is retained here instead of copying the
/// complete multi-gigabyte cache on every DFlash round.
pub(crate) struct MetalSpeculativeCheckpoint {
    start_position: usize,
    token_count: usize,
    planes: Vec<MetalSpeculativePlaneCheckpoint>,
}

struct MetalSpeculativePlaneCheckpoint {
    origin_logical: usize,
    origin_physical: usize,
    len: usize,
    overwritten_physical: Vec<usize>,
    overwritten_key: Vec<u16>,
    overwritten_value: Vec<u16>,
}

impl MetalKvPlane {
    fn new(
        context: &MetalContext,
        capacity: usize,
        kv_dim: usize,
        head_major: bool,
    ) -> Result<Self, MetalError> {
        Ok(Self {
            key: GpuHalfBuffer::zeros(context, capacity * kv_dim)?,
            value: GpuHalfBuffer::zeros(context, capacity * kv_dim)?,
            capacity,
            len: 0,
            origin_logical: 0,
            origin_physical: 0,
            head_major,
        })
    }

    fn uninitialized(
        context: &MetalContext,
        capacity: usize,
        kv_dim: usize,
        head_major: bool,
    ) -> Result<Self, MetalError> {
        Ok(Self {
            key: GpuHalfBuffer::uninitialized(context, capacity * kv_dim)?,
            value: GpuHalfBuffer::uninitialized(context, capacity * kv_dim)?,
            capacity,
            len: 0,
            origin_logical: 0,
            origin_physical: 0,
            head_major,
        })
    }

    /// Reserve the physical row for `position` and advance explicit ring
    /// metadata. No physical placement is derived from the absolute token ID.
    fn append(&mut self, layer: usize, position: usize) -> Result<usize, MetalModelError> {
        let expected = self.origin_logical + self.len;
        if position != expected {
            return Err(MetalModelError::CacheDiscontinuity {
                layer,
                expected,
                got: position,
            });
        }
        if self.len < self.capacity {
            let write = (self.origin_physical + self.len) % self.capacity;
            self.len += 1;
            Ok(write)
        } else {
            let write = self.origin_physical;
            self.origin_logical += 1;
            self.origin_physical = (self.origin_physical + 1) % self.capacity;
            Ok(write)
        }
    }

    fn append_batch(
        &mut self,
        layer: usize,
        start_position: usize,
        token_count: usize,
    ) -> Result<(usize, usize), MetalModelError> {
        let expected = self.origin_logical + self.len;
        if start_position != expected {
            return Err(MetalModelError::CacheDiscontinuity {
                layer,
                expected,
                got: start_position,
            });
        }
        let total = self
            .len
            .checked_add(token_count)
            .ok_or_else(|| MetalModelError::InvalidSnapshot("cache length overflow".into()))?;
        if total <= self.capacity {
            self.len = total;
        } else {
            let overflow = total - self.capacity;
            self.origin_logical += overflow;
            self.origin_physical = (self.origin_physical + overflow) % self.capacity;
            self.len = self.capacity;
        }
        let source_first = self.origin_logical.saturating_sub(start_position);
        Ok((source_first, token_count - source_first))
    }

    fn reset(&mut self) {
        self.len = 0;
        self.origin_logical = 0;
        self.origin_physical = 0;
    }

    fn snapshot(&self, kv_dim: usize, head_dim: usize) -> MetalPlaneSnapshot {
        let key = self.key.as_bits();
        let value = self.value.as_bits();
        let mut key_logical = Vec::with_capacity(self.len * kv_dim);
        let mut value_logical = Vec::with_capacity(self.len * kv_dim);
        for logical_offset in 0..self.len {
            let physical = (self.origin_physical + logical_offset) % self.capacity;
            if self.head_major {
                for kv_head in 0..kv_dim / head_dim {
                    let start = (kv_head * self.capacity + physical) * head_dim;
                    key_logical.extend_from_slice(&key[start..start + head_dim]);
                    value_logical.extend_from_slice(&value[start..start + head_dim]);
                }
            } else {
                let start = physical * kv_dim;
                key_logical.extend_from_slice(&key[start..start + kv_dim]);
                value_logical.extend_from_slice(&value[start..start + kv_dim]);
            }
        }
        MetalPlaneSnapshot {
            origin_logical: self.origin_logical,
            len: self.len,
            capacity: self.capacity,
            key_logical,
            value_logical,
            head_major: self.head_major,
        }
    }

    fn detached_from(
        context: &MetalContext,
        snapshot: &MetalPlaneSnapshot,
        kv_dim: usize,
        head_dim: usize,
    ) -> Result<Self, MetalModelError> {
        let expected = snapshot
            .len
            .checked_mul(kv_dim)
            .ok_or_else(|| MetalModelError::InvalidSnapshot("plane length overflow".into()))?;
        if snapshot.capacity == 0
            || snapshot.len > snapshot.capacity
            || snapshot.key_logical.len() != expected
            || snapshot.value_logical.len() != expected
        {
            return Err(MetalModelError::InvalidSnapshot(format!(
                "plane capacity={}, len={}, key={}, value={}, expected={expected}",
                snapshot.capacity,
                snapshot.len,
                snapshot.key_logical.len(),
                snapshot.value_logical.len()
            )));
        }
        let mut key = vec![0u16; snapshot.capacity * kv_dim];
        let mut value = vec![0u16; snapshot.capacity * kv_dim];
        // Install at the rotation a sequentially-built live ring holds at this
        // logical origin. Attention scans rows in physical order and float
        // accumulation is order-sensitive, so a restore packed at origin 0
        // can never replay a wrapped live session's logits bitwise (caught by
        // real_model_wrap_boundaries_and_detached_restore_replay_exactly).
        // NoPE planes never wrap (origin_logical is always 0), so their
        // rotation is 0 and this reduces to the previous layout.
        let rotation = snapshot.origin_logical % snapshot.capacity;
        if snapshot.head_major {
            for logical_offset in 0..snapshot.len {
                let physical = (rotation + logical_offset) % snapshot.capacity;
                for kv_head in 0..kv_dim / head_dim {
                    let source = (logical_offset * kv_dim) + kv_head * head_dim;
                    let destination = (kv_head * snapshot.capacity + physical) * head_dim;
                    key[destination..destination + head_dim]
                        .copy_from_slice(&snapshot.key_logical[source..source + head_dim]);
                    value[destination..destination + head_dim]
                        .copy_from_slice(&snapshot.value_logical[source..source + head_dim]);
                }
            }
        } else {
            let first = snapshot.len.min(snapshot.capacity - rotation);
            let head = first * kv_dim;
            let start = rotation * kv_dim;
            key[start..start + head].copy_from_slice(&snapshot.key_logical[..head]);
            value[start..start + head].copy_from_slice(&snapshot.value_logical[..head]);
            if first < snapshot.len {
                let tail = expected - head;
                key[..tail].copy_from_slice(&snapshot.key_logical[head..]);
                value[..tail].copy_from_slice(&snapshot.value_logical[head..]);
            }
        }
        Ok(Self {
            key: GpuHalfBuffer::from_bits(context, &key)?,
            value: GpuHalfBuffer::from_bits(context, &value)?,
            capacity: snapshot.capacity,
            len: snapshot.len,
            origin_logical: snapshot.origin_logical,
            origin_physical: rotation,
            head_major: snapshot.head_major,
        })
    }

    /// Retain the rows one speculative block can overwrite in this plane.
    /// A NoPE plane grows into unused storage, so only its logical metadata is
    /// kept; an SWA plane may overwrite at most `token_count` live ring rows.
    fn speculative_checkpoint(
        &self,
        kv_dim: usize,
        head_dim: usize,
        is_swa: bool,
        token_count: usize,
    ) -> Result<MetalSpeculativePlaneCheckpoint, MetalModelError> {
        let mut checkpoint = MetalSpeculativePlaneCheckpoint {
            origin_logical: self.origin_logical,
            origin_physical: self.origin_physical,
            len: self.len,
            overwritten_physical: Vec::new(),
            overwritten_key: Vec::new(),
            overwritten_value: Vec::new(),
        };
        if !is_swa {
            return Ok(checkpoint);
        }
        if token_count > self.capacity {
            return Err(MetalModelError::InvalidSnapshot(format!(
                "speculative block {token_count} exceeds SWA capacity {}",
                self.capacity
            )));
        }
        checkpoint.overwritten_physical.reserve(token_count);
        checkpoint.overwritten_key.reserve(token_count * kv_dim);
        checkpoint.overwritten_value.reserve(token_count * kv_dim);
        let key = self.key.as_bits();
        let value = self.value.as_bits();
        for offset in 0..token_count {
            let physical = (self.origin_physical + self.len + offset) % self.capacity;
            checkpoint.overwritten_physical.push(physical);
            if self.head_major {
                for kv_head in 0..kv_dim / head_dim {
                    let start = (kv_head * self.capacity + physical) * head_dim;
                    checkpoint
                        .overwritten_key
                        .extend_from_slice(&key[start..start + head_dim]);
                    checkpoint
                        .overwritten_value
                        .extend_from_slice(&value[start..start + head_dim]);
                }
            } else {
                let start = physical * kv_dim;
                checkpoint
                    .overwritten_key
                    .extend_from_slice(&key[start..start + kv_dim]);
                checkpoint
                    .overwritten_value
                    .extend_from_slice(&value[start..start + kv_dim]);
            }
        }
        Ok(checkpoint)
    }

    /// Restore the rejected tail of a speculative block and re-append the
    /// accepted prefix. Rows the target accepted stay exactly where the
    /// producer wrote them; every rejected destination is restored byte for
    /// byte before the next command buffer can read it.
    #[allow(clippy::too_many_arguments)]
    fn restore_speculative(
        &mut self,
        layer: usize,
        saved: &MetalSpeculativePlaneCheckpoint,
        kv_dim: usize,
        head_dim: usize,
        is_swa: bool,
        start_position: usize,
        token_count: usize,
        accepted: usize,
    ) -> Result<(), MetalModelError> {
        if is_swa {
            let key = self.key.as_mut_bits();
            let value = self.value.as_mut_bits();
            for offset in accepted..token_count {
                let physical = saved.overwritten_physical[offset];
                let source = offset * kv_dim;
                if self.head_major {
                    for kv_head in 0..kv_dim / head_dim {
                        let src = source + kv_head * head_dim;
                        let dst = (kv_head * self.capacity + physical) * head_dim;
                        key[dst..dst + head_dim]
                            .copy_from_slice(&saved.overwritten_key[src..src + head_dim]);
                        value[dst..dst + head_dim]
                            .copy_from_slice(&saved.overwritten_value[src..src + head_dim]);
                    }
                } else {
                    let destination = physical * kv_dim;
                    key[destination..destination + kv_dim]
                        .copy_from_slice(&saved.overwritten_key[source..source + kv_dim]);
                    value[destination..destination + kv_dim]
                        .copy_from_slice(&saved.overwritten_value[source..source + kv_dim]);
                }
            }
        }
        self.origin_logical = saved.origin_logical;
        self.origin_physical = saved.origin_physical;
        self.len = saved.len;
        if accepted != 0 {
            self.append_batch(layer, start_position, accepted)?;
        }
        Ok(())
    }

    fn write_logical_tile(
        &mut self,
        kv_dim: usize,
        head_dim: usize,
        logical_start: usize,
        count: usize,
        is_key: bool,
        bytes: &[u8],
    ) -> Result<(), MetalModelError> {
        if logical_start < self.origin_logical {
            return Err(MetalModelError::InvalidSnapshot(
                "remote KV tile starts before the plane origin".into(),
            ));
        }
        let logical_offset = logical_start - self.origin_logical;
        let physical = (self.origin_physical + logical_offset) % self.capacity;
        let dest = if is_key {
            self.key.as_mut_bits()
        } else {
            self.value.as_mut_bits()
        };
        let first = count.min(self.capacity - physical);
        let row_bytes = kv_dim
            .checked_mul(std::mem::size_of::<u16>())
            .ok_or_else(|| {
                MetalModelError::InvalidSnapshot("remote KV row size overflow".into())
            })?;
        let expected_bytes = count.checked_mul(row_bytes).ok_or_else(|| {
            MetalModelError::InvalidSnapshot("remote KV tile size overflow".into())
        })?;
        if count > self.capacity || bytes.len() != expected_bytes {
            return Err(MetalModelError::InvalidSnapshot(format!(
                "remote KV tile geometry mismatch: capacity={}, count={count}, bytes={}, expected={expected_bytes}",
                self.capacity,
                bytes.len()
            )));
        }
        let first_bytes = first.checked_mul(row_bytes).ok_or_else(|| {
            MetalModelError::InvalidSnapshot("remote KV tile size overflow".into())
        })?;
        write_f16_tile(
            dest,
            self.capacity,
            kv_dim,
            head_dim,
            self.head_major,
            physical,
            first,
            &bytes[..first_bytes],
        )
        .map_err(MetalModelError::InvalidSnapshot)?;
        if first < count {
            write_f16_tile(
                dest,
                self.capacity,
                kv_dim,
                head_dim,
                self.head_major,
                0,
                count - first,
                &bytes[first_bytes..],
            )
            .map_err(MetalModelError::InvalidSnapshot)?;
        }
        Ok(())
    }
}

/// Detached Metal KV generation filled as authenticated tiles arrive.
/// The live decode cache is not visible until `commit_remote_kv_install`.
pub(crate) struct MetalRemoteKvInstall {
    tokens: Arc<[u32]>,
    planes: Vec<MetalKvPlane>,
    next_key: Vec<usize>,
    next_value: Vec<usize>,
    expected_len: Vec<usize>,
    kv_dim: usize,
    head_dim: usize,
}

impl MetalRemoteKvInstall {
    fn plane_cursor(&mut self, layer: usize, is_key: bool) -> &mut usize {
        if is_key {
            &mut self.next_key[layer]
        } else {
            &mut self.next_value[layer]
        }
    }

    pub(crate) fn write_f16_tile(
        &mut self,
        layer: usize,
        is_key: bool,
        logical_start: u64,
        logical_count: u64,
        bytes: &[u8],
    ) -> Result<(), MetalModelError> {
        if layer >= self.planes.len() {
            return Err(MetalModelError::InvalidSnapshot(
                "remote KV tile names a missing layer".into(),
            ));
        }
        let start = usize::try_from(logical_start).map_err(|_| {
            MetalModelError::InvalidSnapshot("remote KV tile start overflow".into())
        })?;
        let count = usize::try_from(logical_count).map_err(|_| {
            MetalModelError::InvalidSnapshot("remote KV tile count overflow".into())
        })?;
        let expected = *self.plane_cursor(layer, is_key);
        if start != expected {
            return Err(MetalModelError::InvalidSnapshot(format!(
                "remote KV tile for layer {layer} arrived at {start}, expected {expected}"
            )));
        }
        self.planes[layer].write_logical_tile(
            self.kv_dim,
            self.head_dim,
            start,
            count,
            is_key,
            bytes,
        )?;
        *self.plane_cursor(layer, is_key) = start.checked_add(count).ok_or_else(|| {
            MetalModelError::InvalidSnapshot("remote KV tile range overflow".into())
        })?;
        Ok(())
    }

    pub(crate) fn validate_complete(&self) -> Result<(), MetalModelError> {
        for layer in 0..self.planes.len() {
            let origin = self.planes[layer].origin_logical;
            let end = origin + self.expected_len[layer];
            if self.next_key[layer] != end || self.next_value[layer] != end {
                return Err(MetalModelError::InvalidSnapshot(format!(
                    "remote KV layer {layer} is incomplete: K {} V {}, expected {end}",
                    self.next_key[layer], self.next_value[layer]
                )));
            }
        }
        Ok(())
    }
}

struct Activations {
    token_ids: GpuBytes,
    hidden: GpuBuffer,
    normed: GpuBuffer,
    projected: GpuBuffer,
    post_norm: GpuBuffer,
    q: GpuBuffer,
    k: GpuBuffer,
    v: GpuBuffer,
    gate: GpuBuffer,
    attention: GpuBuffer,
    attention_partials: GpuBuffer,
    attention_mask: GpuBytes,
    swa_llama_mask: GpuBytes,
    attention_kv_pad: GpuBytes,
    attention_kv_pad_masked: GpuBytes,
    ffn_gate: GpuBuffer,
    ffn_up: GpuBuffer,
    logits: GpuBuffer,
    dflash_hidden: GpuBuffer,
    dflash_logits: GpuBuffer,
    dflash_argmax_partial_values: GpuBuffer,
    dflash_argmax_partial_indices: GpuBuffer,
    dflash_argmax_results: GpuBuffer,
}

struct BatchActivations {
    hidden: GpuBuffer,
    normed: GpuBuffer,
    projected: GpuBuffer,
    post_norm: GpuBuffer,
    q: GpuBuffer,
    k: GpuBuffer,
    v: GpuBuffer,
    gate: GpuBuffer,
    attention: GpuBuffer,
    ffn_gate: GpuBuffer,
    ffn_up: GpuBuffer,
    nvfp4_quantized: GpuHalfBuffer,
    nvfp4_activation_scales: GpuBytes,
}

struct BatchWorkspace {
    activations: BatchActivations,
    token_ids: GpuBytes,
    swa_staged_key: GpuHalfBuffer,
    swa_staged_value: GpuHalfBuffer,
    /// Transient causal mask + block classifier for the pinned llama
    /// non-vec prefill attention route. `None` when the route is unavailable
    /// (no pinned metallib) or the chunk is below the vec boundary; sized
    /// for the full configured context so every chunk start reuses them.
    fa_prefill_mask: Option<GpuBytes>,
    fa_prefill_blk: Option<GpuBytes>,
    capture_buffers: Vec<GpuBuffer>,
    /// Verification logits must not alias the session-wide DFlash projection
    /// scratch: provisional lookahead projects while this workspace's target
    /// suffix is still pending.
    verify_logits: GpuBuffer,
}

struct DecodeBatchWorkspace {
    activations: BatchActivations,
    token_ids: GpuBytes,
    logits: GpuBuffer,
}

struct BatchTailCapture<'a> {
    layer: usize,
    attention: &'a GpuBuffer,
    residual: &'a GpuBuffer,
    debug: Option<DebugLayerCapture<'a>>,
}

struct DebugLayerCapture<'a> {
    embedding: &'a GpuBuffer,
    entry_norm: &'a GpuBuffer,
    attn_norm: &'a GpuBuffer,
    q: &'a GpuBuffer,
    k: &'a GpuBuffer,
    v: &'a GpuBuffer,
    gate: &'a GpuBuffer,
    q_norm: &'a GpuBuffer,
    k_norm: &'a GpuBuffer,
    q_rope: &'a GpuBuffer,
    k_rope: &'a GpuBuffer,
    attention_raw: &'a GpuBuffer,
    attn_o_proj: &'a GpuBuffer,
    ffn_inp: &'a GpuBuffer,
    ffn_norm: &'a GpuBuffer,
    ffn_gate: &'a GpuBuffer,
    ffn_up: &'a GpuBuffer,
    ffn_swiglu: &'a GpuBuffer,
    ffn_out: &'a GpuBuffer,
    result_norm: &'a GpuBuffer,
    result_projection: &'a GpuBuffer,
}

/// Submitted second half of a DFlash verification batch.  Layers through the
/// final DFlash capture point have completed before this is created; the
/// target suffix and LM head continue on Metal while public Core ML consumes
/// those exact captured rows on ANE.
pub(crate) struct PendingMetalDFlashVerify {
    command: metal::CommandBuffer,
    workspace: BatchWorkspace,
    batch_logits: GpuBuffer,
    token_count: usize,
    submitted_wall_ns: u64,
    capture_fc_ns: u64,
    _permit: AcceleratorPermit,
}

type MetalDFlashVerifyOverlap = (PendingMetalDFlashVerify, Vec<f32>, Option<Vec<f32>>);

/// Real Metal/ANE handoff boundary for one 16-token Muse SWA layer.
/// All rows are token-major, matching the public Core ML wrapper.
pub struct MuseSwaTailCapture {
    pub layer: usize,
    pub attention: Vec<f32>,
    pub residual: Vec<f32>,
    pub metal_hidden: Vec<f32>,
}

/// Isolated Metal execution of the same output/FFN/post-norm tail exported
/// to public Core ML. `wall_ns` includes command encoding, submission, and
/// completion, but excludes allocation and host input copies.
pub struct MuseSwaTailMetalResult {
    pub hidden: Vec<f32>,
    pub wall_ns: u64,
}

impl BatchWorkspace {
    fn new(
        context: &MetalContext,
        cfg: &MuseConfig,
        token_count: usize,
        fa_prefill_scratch: bool,
    ) -> Result<Self, MetalError> {
        let (fa_prefill_mask, fa_prefill_blk) = if fa_prefill_scratch {
            let mask_bytes = token_count
                .checked_mul(cfg.context_length)
                .and_then(|cells| cells.checked_mul(std::mem::size_of::<u16>()))
                .ok_or(MetalError::Allocation(usize::MAX))?;
            let blk_bytes = token_count
                .div_ceil(crate::metal::encode::LLAMA_FA_PREFILL_NQPTG as usize)
                .checked_mul(
                    cfg.context_length
                        .div_ceil(crate::metal::encode::LLAMA_FA_PREFILL_NCPSG as usize),
                )
                .ok_or(MetalError::Allocation(usize::MAX))?;
            (
                Some(GpuBytes::zeros(context, mask_bytes)?),
                Some(GpuBytes::zeros(context, blk_bytes)?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            activations: BatchActivations::new(context, cfg, token_count)?,
            token_ids: GpuBytes::zeros(context, token_count * std::mem::size_of::<u32>())?,
            swa_staged_key: GpuHalfBuffer::zeros(context, cfg.context_length * cfg.kv_dim())?,
            swa_staged_value: GpuHalfBuffer::zeros(context, cfg.context_length * cfg.kv_dim())?,
            fa_prefill_mask,
            fa_prefill_blk,
            capture_buffers: Vec::new(),
            verify_logits: GpuBuffer::zeros(context, token_count * cfg.vocab_size)?,
        })
    }

    fn ensure_capture_buffers(
        &mut self,
        context: &MetalContext,
        cfg: &MuseConfig,
        token_count: usize,
        count: usize,
    ) -> Result<(), MetalError> {
        while self.capture_buffers.len() < count {
            self.capture_buffers
                .push(GpuBuffer::zeros(context, token_count * cfg.hidden_dim)?);
        }
        Ok(())
    }

    fn fa_prefill_scratch(&self) -> Option<(&GpuBytes, &GpuBytes)> {
        self.fa_prefill_mask
            .as_ref()
            .zip(self.fa_prefill_blk.as_ref())
    }
}

impl DecodeBatchWorkspace {
    fn new(context: &MetalContext, cfg: &MuseConfig, rows: usize) -> Result<Self, MetalError> {
        Ok(Self {
            activations: BatchActivations::new(context, cfg, rows)?,
            token_ids: GpuBytes::zeros(context, rows * std::mem::size_of::<u32>())?,
            logits: GpuBuffer::zeros(context, rows * cfg.vocab_size)?,
        })
    }
}

impl BatchActivations {
    fn new(
        context: &MetalContext,
        cfg: &MuseConfig,
        token_count: usize,
    ) -> Result<Self, MetalError> {
        let sized = |width| GpuBuffer::zeros(context, token_count * width);
        let nvfp4_width = cfg.hidden_dim.max(cfg.attn_dim()).max(cfg.intermediate_dim);
        Ok(Self {
            hidden: sized(cfg.hidden_dim)?,
            normed: sized(cfg.hidden_dim)?,
            projected: sized(cfg.hidden_dim)?,
            post_norm: sized(cfg.hidden_dim)?,
            q: sized(cfg.attn_dim())?,
            k: sized(cfg.kv_dim())?,
            v: sized(cfg.kv_dim())?,
            gate: sized(cfg.attn_dim())?,
            attention: sized(cfg.attn_dim())?,
            ffn_gate: sized(cfg.intermediate_dim)?,
            ffn_up: sized(cfg.intermediate_dim)?,
            nvfp4_quantized: GpuHalfBuffer::zeros(context, token_count * nvfp4_width)?,
            nvfp4_activation_scales: GpuBytes::zeros(
                context,
                token_count * nvfp4_width / 16 * std::mem::size_of::<i32>(),
            )?,
        })
    }
}

impl Activations {
    fn new(context: &MetalContext, cfg: &MuseConfig) -> Result<Self, MetalError> {
        let masked = u16::to_le_bytes(0xfc00)
            .into_iter()
            .cycle()
            .take(cfg.context_length * std::mem::size_of::<u16>())
            .collect::<Vec<_>>();
        Ok(Self {
            token_ids: GpuBytes::zeros(
                context,
                MAX_TEACHER_FORCED_TOKENS * std::mem::size_of::<u32>(),
            )?,
            hidden: GpuBuffer::zeros(context, cfg.hidden_dim)?,
            normed: GpuBuffer::zeros(context, cfg.hidden_dim)?,
            projected: GpuBuffer::zeros(context, cfg.hidden_dim)?,
            post_norm: GpuBuffer::zeros(context, cfg.hidden_dim)?,
            q: GpuBuffer::zeros(context, cfg.attn_dim())?,
            k: GpuBuffer::zeros(context, cfg.kv_dim())?,
            v: GpuBuffer::zeros(context, cfg.kv_dim())?,
            gate: GpuBuffer::zeros(context, cfg.attn_dim())?,
            attention: GpuBuffer::zeros(context, cfg.attn_dim())?,
            attention_partials: GpuBuffer::zeros(
                context,
                cfg.n_heads * MAX_DECODE_SPLIT_WORKGROUPS * (2 + cfg.head_dim),
            )?,
            attention_mask: GpuBytes::zeros(
                context,
                cfg.context_length * std::mem::size_of::<u16>(),
            )?,
            swa_llama_mask: GpuBytes::from_bytes(context, &masked)?,
            attention_kv_pad: GpuBytes::zeros(
                context,
                32 * 2 * cfg.kv_dim() * std::mem::size_of::<u16>() * 2,
            )?,
            attention_kv_pad_masked: GpuBytes::zeros(
                context,
                32 * (2 * cfg.kv_dim() * std::mem::size_of::<u16>() * cfg.n_kv_heads
                    + std::mem::size_of::<u16>()),
            )?,
            ffn_gate: GpuBuffer::zeros(context, cfg.intermediate_dim)?,
            ffn_up: GpuBuffer::zeros(context, cfg.intermediate_dim)?,
            logits: GpuBuffer::zeros(context, cfg.vocab_size)?,
            dflash_hidden: GpuBuffer::zeros(context, MAX_DFLASH_BLOCK * cfg.hidden_dim)?,
            dflash_logits: GpuBuffer::zeros(context, MAX_DFLASH_BLOCK * cfg.vocab_size)?,
            dflash_argmax_partial_values: GpuBuffer::zeros(
                context,
                MAX_DFLASH_BLOCK * cfg.vocab_size.div_ceil(1024),
            )?,
            dflash_argmax_partial_indices: GpuBuffer::zeros(
                context,
                MAX_DFLASH_BLOCK * cfg.vocab_size.div_ceil(1024),
            )?,
            dflash_argmax_results: GpuBuffer::zeros(context, MAX_DFLASH_BLOCK)?,
        })
    }
}

/// Immutable Metal execution resources shared by every resident sequence.
/// Metal command submission is scheduler-serialized; retaining one context,
/// pipeline set, mapped weight arena, and GPU vector set avoids loading the
/// 16+ GiB target once per serving slot.
pub struct MetalShared {
    context: MetalContext,
    kernels: MetalKernels,
    _residency_set: Option<crate::metal::residency::ResidencySet>,
    mapped_weights: GpuBytes,
    embedding: Projection,
    output: Projection,
    output_norm: GpuBuffer,
    entry_norm_ones: GpuBuffer,
    rope_frequencies: GpuBuffer,
    rope_positions: GpuBytes,
    layers: Vec<LayerWeights>,
    scheduler: Arc<AcceleratorScheduler>,
    decode_batch_workspaces: Mutex<BTreeMap<usize, DecodeBatchWorkspace>>,
    // The accepted Ferrite batch residual/norm primitive, extended only for
    // Muse's distinct sandwich/pre-layer epsilons. Keep one rollback control
    // for an exact adjacent prefill A/B.
    fused_prefill_dual_norm: bool,
    // Prefill-only concurrent Q/K/V/gate and FFN gate+up. Decode already
    // groups those projections; serial prefill paid a launch tax on PP128.
    // `MUSER_SERIAL_PREFILL_DISPATCH` restores the previous encoder for A/B.
    concurrent_prefill_dispatch: bool,
    // Ferrite 897a6256b: measured 4-row/2-SIMD-group Q4_K SiLU gate+up.
    // The pinned baseline packet regressed with it on this model, so keep the
    // imported route explicitly opt-in for experiments.
    ferrite_ffn_gate_up: bool,
}

/// Sequence-local Metal state. Immutable execution resources are shared;
/// cache, activations, speculative workspaces, and logical position remain
/// isolated for this one resident sequence.
pub struct MetalMuseModel {
    pub cfg: MuseConfig,
    shared: Arc<MetalShared>,
    cache: Vec<MetalKvPlane>,
    activations: Activations,
    batch_workspaces: BTreeMap<usize, BatchWorkspace>,
    n_past: usize,
    sequence_id: usize,
    verify_route_banner_printed: bool,
}

pub(crate) struct MetalGreedyStreamResult {
    pub(crate) consumed_tokens: Vec<u32>,
    pub(crate) next_token: u32,
    pub(crate) final_logits: Vec<f32>,
    pub(crate) cancelled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AcceleratorWork {
    Decode,
    Prefill,
}

#[derive(Default)]
struct AcceleratorSchedulerState {
    active: bool,
    decode_waiting: BTreeSet<usize>,
    last_decode: Option<usize>,
}

/// One owner for the shared Metal queue. Decode work is selected first and
/// resident sequence IDs rotate in ascending cyclic order, preventing a hot
/// slot from repeatedly reacquiring the accelerator ahead of its peers.
struct AcceleratorScheduler {
    state: Mutex<AcceleratorSchedulerState>,
    ready: Condvar,
}

struct AcceleratorPermit {
    scheduler: Arc<AcceleratorScheduler>,
}

impl AcceleratorScheduler {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(AcceleratorSchedulerState::default()),
            ready: Condvar::new(),
        })
    }

    fn acquire(
        self: &Arc<Self>,
        sequence_id: usize,
        work: AcceleratorWork,
    ) -> Result<AcceleratorPermit, MetalModelError> {
        let mut state = self.state.lock().map_err(|_| {
            MetalModelError::InvalidSnapshot("accelerator scheduler is poisoned".into())
        })?;
        if work == AcceleratorWork::Decode && !state.decode_waiting.insert(sequence_id) {
            return Err(MetalModelError::InvalidSnapshot(format!(
                "sequence {sequence_id} queued duplicate decode work"
            )));
        }
        loop {
            let selected_decode = next_decode_sequence(&state);
            let eligible = !state.active
                && match work {
                    AcceleratorWork::Decode => selected_decode == Some(sequence_id),
                    AcceleratorWork::Prefill => selected_decode.is_none(),
                };
            if eligible {
                state.active = true;
                if work == AcceleratorWork::Decode {
                    state.decode_waiting.remove(&sequence_id);
                    state.last_decode = Some(sequence_id);
                }
                return Ok(AcceleratorPermit {
                    scheduler: Arc::clone(self),
                });
            }
            state = self.ready.wait(state).map_err(|_| {
                MetalModelError::InvalidSnapshot("accelerator scheduler is poisoned".into())
            })?;
        }
    }

    fn acquire_decode_group(
        self: &Arc<Self>,
        sequence_ids: &[usize],
    ) -> Result<AcceleratorPermit, MetalModelError> {
        if sequence_ids.is_empty() || sequence_ids.len() > 4 {
            return Err(MetalModelError::InvalidSnapshot(
                "decode group must contain 1..=4 sequences".into(),
            ));
        }
        let unique = sequence_ids.iter().copied().collect::<BTreeSet<_>>();
        if unique.len() != sequence_ids.len() {
            return Err(MetalModelError::InvalidSnapshot(
                "decode group contains a duplicate sequence".into(),
            ));
        }
        let mut state = self.state.lock().map_err(|_| {
            MetalModelError::InvalidSnapshot("accelerator scheduler is poisoned".into())
        })?;
        let mut inserted = Vec::with_capacity(sequence_ids.len());
        for &sequence_id in sequence_ids {
            if !state.decode_waiting.insert(sequence_id) {
                for inserted_id in inserted {
                    state.decode_waiting.remove(&inserted_id);
                }
                return Err(MetalModelError::InvalidSnapshot(format!(
                    "sequence {sequence_id} queued duplicate decode work"
                )));
            }
            inserted.push(sequence_id);
        }
        loop {
            let selected = next_decode_sequence(&state);
            let eligible = !state.active && selected.is_some_and(|id| unique.contains(&id));
            if eligible {
                state.active = true;
                let mut ordered = sequence_ids.to_vec();
                ordered.sort_by_key(|sequence_id| {
                    let last = state.last_decode.unwrap_or(usize::MAX);
                    if *sequence_id > last {
                        (0usize, *sequence_id)
                    } else {
                        (1usize, *sequence_id)
                    }
                });
                for sequence_id in ordered {
                    state.decode_waiting.remove(&sequence_id);
                    state.last_decode = Some(sequence_id);
                }
                return Ok(AcceleratorPermit {
                    scheduler: Arc::clone(self),
                });
            }
            state = self.ready.wait(state).map_err(|_| {
                MetalModelError::InvalidSnapshot("accelerator scheduler is poisoned".into())
            })?;
        }
    }

    fn has_waiting_decode(&self) -> bool {
        self.state
            .lock()
            .map(|state| !state.decode_waiting.is_empty())
            .unwrap_or(true)
    }
}

fn next_decode_sequence(state: &AcceleratorSchedulerState) -> Option<usize> {
    let Some(last) = state.last_decode else {
        return state.decode_waiting.first().copied();
    };
    state
        .decode_waiting
        .range((std::ops::Bound::Excluded(last), std::ops::Bound::Unbounded))
        .next()
        .copied()
        .or_else(|| state.decode_waiting.first().copied())
}

impl Drop for AcceleratorPermit {
    fn drop(&mut self) {
        if let Ok(mut state) = self.scheduler.state.lock() {
            state.active = false;
        }
        self.scheduler.ready.notify_all();
    }
}

impl Deref for MetalMuseModel {
    type Target = MetalShared;

    fn deref(&self) -> &Self::Target {
        &self.shared
    }
}

impl MetalMuseModel {
    pub fn new(
        cfg: MuseConfig,
        weights: MuseWeights,
        max_context: usize,
    ) -> Result<Self, MetalModelError> {
        let shared = Self::load_shared(&cfg, weights)?;
        Self::from_shared(cfg, shared, max_context, 0)
    }

    pub(crate) fn new_sequence_group(
        cfg: MuseConfig,
        weights: MuseWeights,
        max_context: usize,
        count: usize,
    ) -> Result<Vec<Self>, MetalModelError> {
        let shared = Self::load_shared(&cfg, weights)?;
        (0..count)
            .map(|sequence_id| {
                Self::from_shared(cfg.clone(), Arc::clone(&shared), max_context, sequence_id)
            })
            .collect()
    }

    fn load_shared(
        cfg: &MuseConfig,
        weights: MuseWeights,
    ) -> Result<Arc<MetalShared>, MetalModelError> {
        let context = MetalContext::new()?;
        let kernels = MetalKernels::new(&context)?;
        let mapped_weights = GpuBytes::from_mmap(&context, weights.mapped_file())?;
        let residency_set = crate::metal::residency::create_and_attach(
            &context.device,
            &context.queue,
            &[mapped_weights.metal()],
        );
        let embedding = Projection::load(&weights, "token_embd.weight".into())?;
        let output = Projection::load(&weights, "output.weight".into())?;
        if !matches!(embedding.layout.dtype, GgmlType::Q4_K | GgmlType::F16) {
            return Err(MetalModelError::UnsupportedDtype {
                name: embedding.name,
                dtype: embedding.layout.dtype,
            });
        }
        let output_norm = gpu_vector(&context, &weights, "output_norm.weight")?;
        let entry_norm_ones = GpuBuffer::from_f32(&context, &vec![1.0; cfg.hidden_dim])?;
        let rope_values = if let Some(path) = std::env::var_os("MUSER_CROSS_VENDOR_ROPE_CACHE") {
            let path = std::path::PathBuf::from(path);
            let metadata = std::fs::symlink_metadata(&path).map_err(|error| {
                MetalModelError::InvalidSnapshot(format!(
                    "cross-vendor RoPE cache {}: {error}",
                    path.display()
                ))
            })?;
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                return Err(MetalModelError::InvalidSnapshot(format!(
                    "cross-vendor RoPE cache is not a retained regular file: {}",
                    path.display()
                )));
            }
            let expected = cfg.context_length * cfg.head_dim * std::mem::size_of::<f32>();
            if metadata.len() != expected as u64 {
                return Err(MetalModelError::InvalidSnapshot(format!(
                    "cross-vendor RoPE cache {} has {} bytes, expected {expected}",
                    path.display(),
                    metadata.len()
                )));
            }
            std::fs::read(&path)
                .map_err(|error| {
                    MetalModelError::InvalidSnapshot(format!(
                        "read cross-vendor RoPE cache {}: {error}",
                        path.display()
                    ))
                })?
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte float")))
                .collect::<Vec<_>>()
        } else if std::env::var_os("MUSER_CROSS_VENDOR_QK").is_some() {
            crate::rope_nco::canonical_rope_table(
                cfg.context_length,
                cfg.head_dim,
                cfg.rope_base_swa,
            )
        } else {
            (0..cfg.head_dim / 2)
                .map(|index| {
                    1.0 / cfg
                        .rope_base_swa
                        .powf(2.0 * index as f32 / cfg.head_dim as f32)
                })
                .collect::<Vec<_>>()
        };
        let rope_frequencies = GpuBuffer::from_f32(&context, &rope_values)?;
        let rope_positions = GpuBytes::from_bytes(
            &context,
            &(0..cfg.context_length as u32)
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        )?;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for layer in 0..cfg.n_layers {
            let name = |suffix: &str| format!("blk.{layer}.{suffix}");
            let layer_weights = LayerWeights {
                attn_norm: gpu_vector(&context, &weights, &name("attn_norm.weight"))?,
                q: Projection::load(&weights, name("attn_q.weight"))?,
                k: Projection::load(&weights, name("attn_k.weight"))?,
                v: Projection::load(&weights, name("attn_v.weight"))?,
                gate: Projection::load(&weights, name("attn_gate.weight"))?,
                q_norm: gpu_vector(&context, &weights, &name("attn_q_norm.weight"))?,
                k_norm: gpu_vector(&context, &weights, &name("attn_k_norm.weight"))?,
                output: Projection::load(&weights, name("attn_output.weight"))?,
                post_attn_norm: gpu_vector(
                    &context,
                    &weights,
                    &name("post_attention_norm.weight"),
                )?,
                ffn_norm: gpu_vector(&context, &weights, &name("ffn_norm.weight"))?,
                ffn_gate: Projection::load(&weights, name("ffn_gate.weight"))?,
                ffn_up: Projection::load(&weights, name("ffn_up.weight"))?,
                ffn_down: Projection::load(&weights, name("ffn_down.weight"))?,
                post_ffn_norm: gpu_vector(&context, &weights, &name("post_ffw_norm.weight"))?,
            };
            for projection in [
                &layer_weights.q,
                &layer_weights.k,
                &layer_weights.v,
                &layer_weights.gate,
                &layer_weights.output,
                &layer_weights.ffn_gate,
                &layer_weights.ffn_up,
                &layer_weights.ffn_down,
            ] {
                if !kernels.supports_projection(projection.layout.dtype) {
                    return Err(MetalModelError::MissingProjectionKernel {
                        name: projection.name.clone(),
                        dtype: projection.layout.dtype,
                    });
                }
            }
            layers.push(layer_weights);
        }

        Ok(Arc::new(MetalShared {
            context,
            kernels,
            _residency_set: residency_set,
            mapped_weights,
            embedding,
            output,
            output_norm,
            entry_norm_ones,
            rope_frequencies,
            rope_positions,
            layers,
            scheduler: AcceleratorScheduler::new(),
            decode_batch_workspaces: Mutex::new(BTreeMap::new()),
            // The fused kernel reproduces the two pinned ggml f32x4 norm
            // reductions and their intervening f32 device-memory boundary.
            // Keep the split route only as an explicit diagnostic control.
            fused_prefill_dual_norm: std::env::var_os("MUSER_NO_FUSED_PREFILL_DUAL_NORM").is_none(),
            concurrent_prefill_dispatch: std::env::var_os("MUSER_SERIAL_PREFILL_DISPATCH")
                .is_none(),
            ferrite_ffn_gate_up: std::env::var_os("MUSER_FERRITE_FFN_GATE_UP").is_some(),
        }))
    }

    fn from_shared(
        cfg: MuseConfig,
        shared: Arc<MetalShared>,
        max_context: usize,
        sequence_id: usize,
    ) -> Result<Self, MetalModelError> {
        let mut cache = Vec::with_capacity(cfg.n_layers);
        for layer in 0..cfg.n_layers {
            let capacity = match cfg.layer_kinds[layer] {
                MuseLayerKind::SlidingRope => max_context.min(cfg.sliding_window).max(32),
                MuseLayerKind::FullNoPe => max_context.max(32),
            };
            // Zero-filled on purpose: wrapped SWA rows must never expose
            // uninitialized storage during a sequence boundary transition.
            cache.push(MetalKvPlane::new(
                &shared.context,
                capacity,
                cfg.kv_dim(),
                matches!(cfg.layer_kinds[layer], MuseLayerKind::FullNoPe),
            )?);
        }
        let activations = Activations::new(&shared.context, &cfg)?;
        Ok(Self {
            cfg,
            shared,
            cache,
            activations,
            batch_workspaces: BTreeMap::new(),
            n_past: 0,
            sequence_id,
            verify_route_banner_printed: false,
        })
    }

    pub fn position(&self) -> usize {
        self.n_past
    }

    pub fn reset(&mut self) {
        self.n_past = 0;
        for plane in &mut self.cache {
            plane.reset();
        }
    }

    /// Retain only the cache bytes that a short speculative block can
    /// overwrite. NoPE planes grow into unused storage, while each SWA layer
    /// touches at most `token_count` ring rows.
    pub(crate) fn speculative_checkpoint(
        &self,
        token_count: usize,
    ) -> Result<MetalSpeculativeCheckpoint, MetalModelError> {
        if token_count == 0 || token_count > MAX_DFLASH_BLOCK {
            return Err(MetalModelError::InvalidSnapshot(format!(
                "speculative checkpoint length {token_count} is outside 1..={MAX_DFLASH_BLOCK}"
            )));
        }
        let kv_dim = self.cfg.kv_dim();
        let head_dim = self.cfg.head_dim;
        let mut planes = Vec::with_capacity(self.cache.len());
        for (layer, plane) in self.cache.iter().enumerate() {
            planes.push(plane.speculative_checkpoint(
                kv_dim,
                head_dim,
                self.cfg.layer_kinds[layer].is_swa(),
                token_count,
            )?);
        }
        Ok(MetalSpeculativeCheckpoint {
            start_position: self.n_past,
            token_count,
            planes,
        })
    }

    /// Commit an accepted prefix from a speculative verification. The producer
    /// may have evaluated the whole block in one batch or stopped sequentially
    /// at the first rejected token. Rejected SWA destinations are restored
    /// before the next command buffer can read them; accepted K/V rows remain
    /// exactly where the producer wrote them. This is the ring-safe analogue
    /// of Ferrite's position rewind.
    pub(crate) fn commit_speculative_prefix(
        &mut self,
        checkpoint: MetalSpeculativeCheckpoint,
        accepted: usize,
    ) -> Result<(), MetalModelError> {
        let produced = self.n_past.checked_sub(checkpoint.start_position);
        if accepted > checkpoint.token_count
            || produced.is_none_or(|count| count < accepted || count > checkpoint.token_count)
            || checkpoint.planes.len() != self.cache.len()
        {
            return Err(MetalModelError::InvalidSnapshot(
                "speculative checkpoint does not match the completed batch".into(),
            ));
        }
        let kv_dim = self.cfg.kv_dim();
        let head_dim = self.cfg.head_dim;
        for (layer, (plane, saved)) in self
            .cache
            .iter_mut()
            .zip(checkpoint.planes.iter())
            .enumerate()
        {
            plane.restore_speculative(
                layer,
                saved,
                kv_dim,
                head_dim,
                self.cfg.layer_kinds[layer].is_swa(),
                checkpoint.start_position,
                checkpoint.token_count,
                accepted,
            )?;
        }
        self.n_past = checkpoint.start_position + accepted;
        Ok(())
    }

    fn take_batch_workspace(
        &mut self,
        token_count: usize,
    ) -> Result<BatchWorkspace, MetalModelError> {
        match self.batch_workspaces.remove(&token_count) {
            Some(workspace) => Ok(workspace),
            None => Ok(BatchWorkspace::new(
                &self.context,
                &self.cfg,
                token_count,
                self.llama_fa_prefill_available(token_count),
            )?),
        }
    }

    /// The pinned llama non-vec prefill route needs its mask/blk scratch only
    /// at or above the vec boundary (llama routes `batch < 20` to the vec
    /// kernel; Muser's pinned-vec path mirrors that). Cross-vendor sessions
    /// retain their declared serial attention route through
    /// `encode_flash_attention_v2`; allocating this scratch would otherwise
    /// silently route aligned NoPE chunks around it.
    fn llama_fa_prefill_available(&self, token_count: usize) -> bool {
        llama_fa_prefill_route_available(
            token_count,
            self.kernels.has_llama_flash_attn_vec(),
            std::env::var_os("MUSER_NO_LLAMA_FA_PREFILL").is_some(),
            std::env::var_os("MUSER_CROSS_VENDOR_QK").is_some(),
        )
    }

    fn return_batch_workspace(&mut self, token_count: usize, workspace: BatchWorkspace) {
        if token_count != PREFILL_BATCH_TOKENS {
            self.batch_workspaces
                .retain(|&tokens, _| tokens == PREFILL_BATCH_TOKENS);
        }
        self.batch_workspaces.insert(token_count, workspace);
    }

    /// Ferrite-derived production DFlash target LM-head projection. Scratch
    /// is retained with the target session so every draft round avoids both
    /// a full CPU quantized matmul and repeated Metal allocations.
    pub(crate) fn project_dflash_hidden(
        &mut self,
        hidden: &[f32],
    ) -> Result<Vec<f32>, MetalModelError> {
        if hidden.is_empty() || !hidden.len().is_multiple_of(self.cfg.hidden_dim) {
            return Err(MetalModelError::InvalidSnapshot(
                "DFlash LM-head hidden shape is invalid".into(),
            ));
        }
        let tokens = hidden.len() / self.cfg.hidden_dim;
        if tokens > MAX_DFLASH_BLOCK {
            return Err(MetalModelError::InvalidSnapshot(format!(
                "DFlash LM-head batch {tokens} exceeds {MAX_DFLASH_BLOCK}"
            )));
        }
        self.activations.dflash_hidden.as_mut_slice()[..hidden.len()].copy_from_slice(hidden);
        let input = self.activations.dflash_hidden.clone();
        self.project_dflash_buffer(&input, tokens)
    }

    /// Project DFlash output that already resides in shared Metal storage.
    /// The buffer may originate from the assistant's command queue; its
    /// completed command is the synchronization boundary before this call.
    pub(crate) fn project_dflash_buffer(
        &mut self,
        input: &GpuBuffer,
        tokens: usize,
    ) -> Result<Vec<f32>, MetalModelError> {
        if tokens == 0 || tokens > MAX_DFLASH_BLOCK || input.len() < tokens * self.cfg.hidden_dim {
            return Err(MetalModelError::InvalidSnapshot(
                "DFlash Metal LM-head buffer shape is invalid".into(),
            ));
        }
        let scheduler = Arc::clone(&self.shared.scheduler);
        let _permit = scheduler.acquire(self.sequence_id, AcceleratorWork::Decode)?;
        let command = self.context.queue.new_command_buffer();
        self.project_tokens(
            command,
            &self.output,
            input,
            &self.activations.dflash_logits,
            tokens,
        );
        dispatch(command, |encoder| {
            self.kernels.encode_scale_softcap_count(
                encoder,
                &self.activations.dflash_logits,
                tokens * self.cfg.vocab_size,
                self.cfg.logit_scale,
                self.cfg.final_logit_softcap,
            );
        });
        command.commit();
        self.context
            .wait_for_completion(command, Duration::from_secs(300))?;
        Ok(self.activations.dflash_logits.as_slice()[..tokens * self.cfg.vocab_size].to_vec())
    }

    pub(crate) fn project_dflash_hidden_argmax(
        &mut self,
        hidden: &[f32],
    ) -> Result<Vec<u32>, MetalModelError> {
        if hidden.is_empty() || !hidden.len().is_multiple_of(self.cfg.hidden_dim) {
            return Err(MetalModelError::InvalidSnapshot(
                "DFlash greedy LM-head hidden shape is invalid".into(),
            ));
        }
        let tokens = hidden.len() / self.cfg.hidden_dim;
        if tokens > MAX_DFLASH_BLOCK {
            return Err(MetalModelError::InvalidSnapshot(format!(
                "DFlash greedy LM-head batch {tokens} exceeds {MAX_DFLASH_BLOCK}"
            )));
        }
        self.activations.dflash_hidden.as_mut_slice()[..hidden.len()].copy_from_slice(hidden);
        let input = self.activations.dflash_hidden.clone();
        self.project_dflash_buffer_argmax(&input, tokens)
    }

    pub(crate) fn project_dflash_buffer_argmax(
        &mut self,
        input: &GpuBuffer,
        tokens: usize,
    ) -> Result<Vec<u32>, MetalModelError> {
        if tokens == 0 || tokens > MAX_DFLASH_BLOCK || input.len() < tokens * self.cfg.hidden_dim {
            return Err(MetalModelError::InvalidSnapshot(
                "DFlash greedy Metal LM-head buffer shape is invalid".into(),
            ));
        }
        let scheduler = Arc::clone(&self.shared.scheduler);
        let _permit = scheduler.acquire(self.sequence_id, AcceleratorWork::Decode)?;
        let command = self.context.queue.new_command_buffer();
        self.project_tokens(
            command,
            &self.output,
            input,
            &self.activations.dflash_logits,
            tokens,
        );
        dispatch(command, |encoder| {
            self.kernels.encode_scale_softcap_count(
                encoder,
                &self.activations.dflash_logits,
                tokens * self.cfg.vocab_size,
                self.cfg.logit_scale,
                self.cfg.final_logit_softcap,
            );
            self.kernels.encode_argmax_f32_rows(
                encoder,
                &self.activations.dflash_logits,
                &self.activations.dflash_argmax_partial_values,
                &self.activations.dflash_argmax_partial_indices,
                &self.activations.dflash_argmax_results,
                tokens,
                self.cfg.vocab_size,
            );
        });
        command.commit();
        self.context
            .wait_for_completion(command, Duration::from_secs(300))?;
        Ok(self.activations.dflash_argmax_results.as_slice()[..tokens]
            .iter()
            .map(|value| value.to_bits())
            .collect())
    }

    /// Keep a bounded dependent greedy block in flight. Command `i+1`
    /// reads command `i`'s four-byte argmax result directly as its embedding
    /// token, so host encoding and queue submission overlap GPU execution
    /// without changing any target-model dispatch or logit byte.
    pub(crate) fn forward_greedy_streaming(
        &mut self,
        first_token: u32,
        token_count: usize,
        excluded_tokens: &[u32],
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<MetalGreedyStreamResult, MetalModelError> {
        if token_count == 0
            || token_count > MAX_DFLASH_BLOCK
            || first_token as usize >= self.cfg.vocab_size
        {
            return Err(MetalModelError::InvalidSnapshot(
                "pipelined greedy decode requires a valid token and positive length".into(),
            ));
        }
        if self.n_past + token_count > self.cfg.context_length
            || excluded_tokens
                .iter()
                .any(|&token| token as usize >= self.cfg.vocab_size)
        {
            return Err(MetalModelError::InvalidSnapshot(
                "pipelined greedy decode exceeds model bounds".into(),
            ));
        }
        let scheduler = Arc::clone(&self.shared.scheduler);
        let _permit = scheduler.acquire(self.sequence_id, AcceleratorWork::Decode)?;
        let checkpoint = self.speculative_checkpoint(token_count)?;
        let mut workspace = self.take_batch_workspace(1)?;
        workspace.token_ids.as_mut_bytes()[..4].copy_from_slice(&first_token.to_le_bytes());
        let mut excluded_bytes = Vec::with_capacity(excluded_tokens.len().max(1) * 4);
        if excluded_tokens.is_empty() {
            excluded_bytes.extend_from_slice(&0u32.to_le_bytes());
        } else {
            for token in excluded_tokens {
                excluded_bytes.extend_from_slice(&token.to_le_bytes());
            }
        }
        let excluded = GpuBytes::from_bytes(&self.context, &excluded_bytes)?;
        let start_position = self.n_past;
        let mut pending = std::collections::VecDeque::with_capacity(token_count);
        let mut encoded = 0usize;
        while encoded < token_count {
            pending.push_back(self.encode_greedy_pipeline_command(
                &workspace,
                encoded,
                start_position + encoded,
                &excluded,
                excluded_tokens.len(),
            )?);
            encoded += 1;
        }

        let mut consumed_tokens = vec![first_token];
        let mut next_token = first_token;
        let mut completed = 0usize;
        let mut cancelled = false;
        while completed < token_count {
            let command = pending
                .pop_front()
                .expect("every encoded greedy command remains pending until completion");
            self.context
                .wait_for_completion(&command, Duration::from_secs(300))?;
            let produced = self.activations.dflash_argmax_results.as_slice()[completed].to_bits();
            if produced == u32::MAX || produced as usize >= self.cfg.vocab_size {
                return Err(MetalModelError::InvalidSnapshot(
                    "pipelined greedy argmax observed nonfinite logits or an invalid token".into(),
                ));
            }
            completed += 1;
            next_token = produced;
            if completed < token_count {
                if !on_token(produced) {
                    cancelled = true;
                } else {
                    consumed_tokens.push(produced);
                }
            }
            if cancelled {
                for command in pending.drain(..) {
                    self.context
                        .wait_for_completion(&command, Duration::from_secs(300))?;
                }
                self.n_past = start_position + encoded;
                self.commit_speculative_prefix(checkpoint, consumed_tokens.len())?;
                self.return_batch_workspace(1, workspace);
                return Ok(MetalGreedyStreamResult {
                    consumed_tokens,
                    next_token,
                    final_logits: Vec::new(),
                    cancelled: true,
                });
            }
        }
        self.n_past = start_position + token_count;
        let final_logits = self.activations.logits.as_slice().to_vec();
        self.return_batch_workspace(1, workspace);
        Ok(MetalGreedyStreamResult {
            consumed_tokens,
            next_token,
            final_logits,
            cancelled: false,
        })
    }

    fn encode_greedy_pipeline_command(
        &mut self,
        workspace: &BatchWorkspace,
        index: usize,
        position: usize,
        excluded: &GpuBytes,
        excluded_count: usize,
    ) -> Result<metal::CommandBuffer, MetalModelError> {
        let command = new_prefill_command_buffer(&self.context.queue).to_owned();
        let serial = new_prefill_graph_encoder(&command, self.concurrent_prefill_dispatch);
        dispatch(&serial, |encoder| {
            if index == 0 {
                let token = workspace
                    .token_ids
                    .view(0, std::mem::size_of::<u32>())
                    .expect("one checked greedy token");
                self.kernels.encode_embedding_q4k(
                    encoder,
                    self.embedding.view(&self.mapped_weights),
                    &token,
                    &workspace.activations.hidden,
                    self.cfg.hidden_dim,
                    self.cfg.vocab_size,
                    1,
                );
            } else {
                self.kernels.encode_embedding_q4k_from_u32_buffer(
                    encoder,
                    self.embedding.view(&self.mapped_weights),
                    &self.activations.dflash_argmax_results,
                    index - 1,
                    &workspace.activations.hidden,
                    self.cfg.hidden_dim,
                    self.cfg.vocab_size,
                );
            }
        });
        self.encode_batch_hidden(
            &workspace.activations,
            &workspace.swa_staged_key,
            &workspace.swa_staged_value,
            workspace.fa_prefill_scratch(),
            1,
            position,
            &serial,
            &[],
            &[],
            None,
            None,
        )?;
        dispatch(&serial, |encoder| {
            self.kernels.encode_greedy_argmax_f32(
                encoder,
                &self.activations.logits,
                &self.activations.dflash_argmax_partial_values,
                &self.activations.dflash_argmax_partial_indices,
                &self.activations.dflash_argmax_results,
                index,
                self.cfg.vocab_size,
                excluded,
                excluded_count,
            );
        });
        serial.encoder.end_encoding();
        command.commit();
        Ok(command)
    }

    /// Capture every cache row in logical order. The most recent command has
    /// completed before `forward` returns, so shared-buffer reads are stable.
    pub(crate) fn snapshot(&self) -> MetalKvSnapshot {
        MetalKvSnapshot {
            n_past: self.n_past,
            layers: self
                .cache
                .iter()
                .map(|plane| plane.snapshot(self.cfg.kv_dim(), self.cfg.head_dim))
                .collect(),
        }
    }

    /// Validate and materialize the entire snapshot into detached buffers,
    /// then atomically replace the live cache vector.
    pub(crate) fn restore(&mut self, snapshot: &MetalKvSnapshot) -> Result<(), MetalModelError> {
        if snapshot.n_past > self.cfg.context_length || snapshot.layers.len() != self.cfg.n_layers {
            return Err(MetalModelError::InvalidSnapshot(format!(
                "position {} or layer count {} disagrees with model",
                snapshot.n_past,
                snapshot.layers.len()
            )));
        }
        let mut detached = Vec::with_capacity(snapshot.layers.len());
        for (layer, plane) in snapshot.layers.iter().enumerate() {
            let required_capacity = self.cache[layer].capacity;
            let expected_len = if self.cfg.layer_kinds[layer].is_swa() {
                snapshot.n_past.min(required_capacity)
            } else {
                snapshot.n_past
            };
            let expected_origin = snapshot.n_past - expected_len;
            if plane.capacity != required_capacity
                || plane.len != expected_len
                || plane.origin_logical != expected_origin
                || plane.head_major != self.cache[layer].head_major
            {
                return Err(MetalModelError::InvalidSnapshot(format!(
                    "layer {layer}: capacity/len/origin {}/{}/{} expected {required_capacity}/{expected_len}/{expected_origin}",
                    plane.capacity, plane.len, plane.origin_logical
                )));
            }
            detached.push(MetalKvPlane::detached_from(
                &self.context,
                plane,
                self.cfg.kv_dim(),
                self.cfg.head_dim,
            )?);
        }
        self.cache = detached;
        self.n_past = snapshot.n_past;
        Ok(())
    }

    pub(crate) fn begin_remote_kv_install(
        &self,
        tokens: Arc<[u32]>,
    ) -> Result<MetalRemoteKvInstall, MetalModelError> {
        let position = tokens.len();
        if position == 0 {
            return Err(MetalModelError::InvalidSnapshot(
                "remote KV install needs a nonempty prefix".into(),
            ));
        }
        let mut planes = Vec::with_capacity(self.cfg.n_layers);
        let mut next_key = Vec::with_capacity(self.cfg.n_layers);
        let mut next_value = Vec::with_capacity(self.cfg.n_layers);
        let mut expected_len = Vec::with_capacity(self.cfg.n_layers);
        for layer in 0..self.cfg.n_layers {
            let live = &self.cache[layer];
            let len = if self.cfg.layer_kinds[layer].is_swa() {
                position.min(live.capacity)
            } else {
                position
            };
            let origin = position - len;
            let mut plane = MetalKvPlane::uninitialized(
                &self.context,
                live.capacity,
                self.cfg.kv_dim(),
                live.head_major,
            )?;
            plane.origin_logical = origin;
            // Match a sequentially-built ring exactly. Physical scan order is
            // numerically observable, so remote restore cannot repack the
            // retained logical tail at row zero.
            plane.origin_physical = origin % live.capacity;
            plane.len = len;
            planes.push(plane);
            next_key.push(origin);
            next_value.push(origin);
            expected_len.push(len);
        }
        Ok(MetalRemoteKvInstall {
            tokens,
            planes,
            next_key,
            next_value,
            expected_len,
            kv_dim: self.cfg.kv_dim(),
            head_dim: self.cfg.head_dim,
        })
    }

    /// Delta variant of `begin_remote_kv_install`: the detached generation is
    /// allocated for the full position, the held prefix `[0, cut)` is copied
    /// out of the live planes with the exact ring mapping, and the write
    /// cursors resume past the copied span so only suffix tiles are accepted.
    /// `commit`/`validate_complete` semantics are unchanged: the prefix
    /// copy-in counts as filled for `[0, cut)`.
    pub(crate) fn begin_remote_kv_install_delta(
        &self,
        tokens: Arc<[u32]>,
        prefix_cut: usize,
    ) -> Result<MetalRemoteKvInstall, MetalModelError> {
        if prefix_cut == 0 {
            return self.begin_remote_kv_install(tokens);
        }
        let position = tokens.len();
        if prefix_cut >= position || prefix_cut > self.n_past {
            return Err(MetalModelError::InvalidSnapshot(format!(
                "delta prefix cut {prefix_cut} must leave a suffix the live session holds (position {position}, live {})",
                self.n_past
            )));
        }
        let mut install = self.begin_remote_kv_install(tokens)?;
        let kv_dim = self.cfg.kv_dim();
        let head_dim = self.cfg.head_dim;
        for layer in 0..install.planes.len() {
            let live = &self.cache[layer];
            let span = crate::api::delta_prefix_span(
                install.planes[layer].origin_logical,
                position,
                prefix_cut,
                live.origin_logical,
                live.origin_logical + live.len,
            )
            .map_err(MetalModelError::InvalidSnapshot)?;
            if span.copy_start < span.copy_end {
                let plane = &mut install.planes[layer];
                let capacity = plane.capacity;
                let head_major = plane.head_major;
                let destination_origin = crate::api::RingOrigin {
                    logical: plane.origin_logical,
                    physical: plane.origin_physical,
                };
                let source_origin = crate::api::RingOrigin {
                    logical: live.origin_logical,
                    physical: live.origin_physical,
                };
                crate::api::copy_ring_prefix_rows(
                    live.key.as_bits(),
                    plane.key.as_mut_bits(),
                    capacity,
                    kv_dim,
                    head_dim,
                    head_major,
                    source_origin,
                    destination_origin,
                    span.copy_start,
                    span.copy_end,
                )
                .map_err(MetalModelError::InvalidSnapshot)?;
                crate::api::copy_ring_prefix_rows(
                    live.value.as_bits(),
                    plane.value.as_mut_bits(),
                    capacity,
                    kv_dim,
                    head_dim,
                    head_major,
                    source_origin,
                    destination_origin,
                    span.copy_start,
                    span.copy_end,
                )
                .map_err(MetalModelError::InvalidSnapshot)?;
            }
            install.next_key[layer] = span.resume;
            install.next_value[layer] = span.resume;
        }
        Ok(install)
    }

    pub(crate) fn commit_remote_kv_install(
        &mut self,
        install: MetalRemoteKvInstall,
    ) -> Result<(), MetalModelError> {
        install.validate_complete()?;
        self.commit_prepared_remote_kv_install(install);
        Ok(())
    }

    pub(crate) fn commit_prepared_remote_kv_install(&mut self, install: MetalRemoteKvInstall) {
        debug_assert!(install.validate_complete().is_ok());
        self.n_past = install.tokens.len();
        self.cache = install.planes;
    }

    pub(crate) fn export_cache_snapshot(
        &self,
        tokens: &[u32],
    ) -> Result<SessionCacheSnapshot, MetalModelError> {
        if self.n_past == 0 || tokens.len() != self.n_past {
            return Err(MetalModelError::InvalidSnapshot(
                "Metal cache export requires the exact nonempty token history".into(),
            ));
        }
        let raw = self.snapshot();
        let layers = raw
            .layers
            .iter()
            .enumerate()
            .map(|(layer, plane)| CachePlaneSnapshot {
                layer: layer as u32,
                logical_start: plane.origin_logical as u64,
                logical_count: plane.len as u64,
                encoding: PlaneEncoding::F16Le,
                key: u16s_to_le_bytes(&plane.key_logical),
                value: u16s_to_le_bytes(&plane.value_logical),
            })
            .collect::<Vec<_>>();
        let snapshot = SessionCacheSnapshot {
            position: self.n_past as u64,
            tokens: tokens.to_vec().into(),
            elements_per_token: self.cfg.kv_dim() as u32,
            layers: layers.into(),
        };
        snapshot
            .validate_for_window(self.cfg.sliding_window)
            .map_err(MetalModelError::InvalidSnapshot)?;
        Ok(snapshot)
    }

    pub(crate) fn install_cache_snapshot(
        &mut self,
        snapshot: &SessionCacheSnapshot,
    ) -> Result<(), MetalModelError> {
        snapshot
            .validate_for_window(self.cfg.sliding_window)
            .map_err(MetalModelError::InvalidSnapshot)?;
        if snapshot.elements_per_token as usize != self.cfg.kv_dim()
            || snapshot.encoding() != Some(PlaneEncoding::F16Le)
        {
            return Err(MetalModelError::InvalidSnapshot(
                "Metal cache snapshot geometry or encoding mismatch".into(),
            ));
        }
        let mut layers = Vec::with_capacity(snapshot.layers.len());
        for plane in snapshot.layers.iter() {
            let layer = plane.layer as usize;
            layers.push(MetalPlaneSnapshot {
                origin_logical: plane.logical_start as usize,
                len: plane.logical_count as usize,
                capacity: self.cache[layer].capacity,
                key_logical: le_bytes_to_u16s(&plane.key)
                    .map_err(MetalModelError::InvalidSnapshot)?,
                value_logical: le_bytes_to_u16s(&plane.value)
                    .map_err(MetalModelError::InvalidSnapshot)?,
                head_major: self.cache[layer].head_major,
            });
        }
        self.restore(&MetalKvSnapshot {
            n_past: snapshot.position as usize,
            layers,
        })
    }

    /// Run tokens sequentially through the real Metal graph. Sequential
    /// prefill is the correctness baseline; Stage 2 batching replaces it.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>, MetalModelError> {
        let mut logits = Vec::new();
        self.forward_into(tokens, &mut logits)?;
        Ok(logits)
    }

    /// `forward` into a caller-owned buffer. Single-token decode refills the
    /// buffer in place, so steady-state decode does not allocate one
    /// vocabulary-sized `Vec` per token. A failing forward leaves the buffer
    /// untouched, so the caller's previous distribution survives the error.
    pub fn forward_into(
        &mut self,
        tokens: &[u32],
        logits: &mut Vec<f32>,
    ) -> Result<(), MetalModelError> {
        if tokens.len() == 1 {
            let scheduler = Arc::clone(&self.shared.scheduler);
            let _permit = scheduler.acquire(self.sequence_id, AcceleratorWork::Decode)?;
            // The legacy one-token graph uses Ferrite fused residual/norm and
            // gate-up kernels whose rounding diverges from the source-pinned
            // llama Metal graph enough to breach public logprob tolerance.
            // The one-row batch graph dispatches the exact pinned kernels and
            // has the same KV transition, so it is the serving correctness
            // path until each fused kernel independently passes full-logit
            // parity.
            *logits = self.forward_batch(tokens)?;
            return Ok(());
        }
        let mut batched = None;
        let mut offset = 0;
        while offset < tokens.len() {
            // Long idle prefills retain the accepted 512-row physical batch.
            // Once a decoder is queued, the next prefill boundary shrinks to
            // 64 rows so decode can take ownership without another long
            // accelerator interval in front of it.
            let scheduler = Arc::clone(&self.shared.scheduler);
            let chunk_tokens = if scheduler.has_waiting_decode() {
                MAX_TEACHER_FORCED_TOKENS
            } else {
                PREFILL_BATCH_TOKENS
            };
            let end = (offset + chunk_tokens).min(tokens.len());
            let _permit = scheduler.acquire(self.sequence_id, AcceleratorWork::Prefill)?;
            let chunk = &tokens[offset..end];
            batched = Some(self.forward_batch(chunk)?);
            offset = end;
        }
        *logits = batched.unwrap_or_default();
        Ok(())
    }

    /// Benchmark-only teacher-forced decode sink. This preserves the exact
    /// one-token graph and KV transitions while submitting the known teacher
    /// sequence as one ordered Metal command buffer, matching the comparator's
    /// no-sampler and no-per-token-host-readback policy. The returned IDs are
    /// the teacher inputs (a workload witness), not sampled model outputs;
    /// greedy equality is qualified by the separate correctness lane.
    pub(crate) fn forward_teacher_forced(
        &mut self,
        tokens: &[u32],
    ) -> Result<Vec<u32>, MetalModelError> {
        if tokens.is_empty() || tokens.len() > MAX_TEACHER_FORCED_TOKENS {
            return Err(MetalModelError::InvalidSnapshot(format!(
                "teacher-forced block must contain 1..={MAX_TEACHER_FORCED_TOKENS} tokens"
            )));
        }
        if tokens.len() == 1 && std::env::var_os("MUSER_METAL_PHASE_PROFILE").is_some() {
            let scheduler = Arc::clone(&self.shared.scheduler);
            let _permit = scheduler.acquire(self.sequence_id, AcceleratorWork::Decode)?;
            self.forward_token(tokens[0])?;
            return Ok(tokens.to_vec());
        }
        let scheduler = Arc::clone(&self.shared.scheduler);
        let _permit = scheduler.acquire(self.sequence_id, AcceleratorWork::Decode)?;
        let mut token_staging = self.activations.token_ids.clone();
        for (destination, token) in token_staging
            .as_mut_bytes()
            .chunks_exact_mut(std::mem::size_of::<u32>())
            .zip(tokens)
        {
            destination.copy_from_slice(&token.to_le_bytes());
        }
        let queue = self.context.queue.clone();
        let start_position = self.n_past;
        // One retained concurrent encoder per teacher token, committed without
        // an intermediate wait. The queue serializes GPU work so token i+1
        // cannot race token i's residual/KV, while the host encodes i+1 during
        // i's GPU interval. A single mega-buffer idles the GPU for the whole
        // host encode; waiting after every token pays 64 round-trips.
        let mut last_command = None;
        for (index, _) in tokens.iter().enumerate() {
            let token_view = token_staging
                .view(
                    index * std::mem::size_of::<u32>(),
                    std::mem::size_of::<u32>(),
                )
                .expect("teacher token view");
            let command_buffer = queue.new_command_buffer();
            let serial = GraphEncoder::concurrent(
                command_buffer
                    .compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent),
            );
            self.encode_token(&serial, &token_view, start_position + index)?;
            serial.encoder.end_encoding();
            command_buffer.commit();
            last_command = Some(command_buffer);
        }
        self.context.wait_for_completion(
            last_command.expect("teacher-forced block encodes at least one command buffer"),
            Duration::from_secs(300),
        )?;
        self.n_past += tokens.len();
        // Deliberately do not touch `logits` from the CPU. The final projection
        // is the GPU-owned sink, matching the pinned llama fixture exactly.
        Ok(tokens.to_vec())
    }

    /// Target-layer capture for DFlash. Rows are returned in the same
    /// token-major `[token][selected-layer][hidden]` layout as the CPU oracle.
    pub fn forward_capturing_layers(
        &mut self,
        tokens: &[u32],
        layer_ids: &[usize],
    ) -> Result<(Vec<f32>, Vec<f32>), MetalModelError> {
        let mut logits = Vec::new();
        let mut captured = Vec::with_capacity(tokens.len() * layer_ids.len() * self.cfg.hidden_dim);
        let mut offset = 0;
        while offset < tokens.len() {
            let scheduler = Arc::clone(&self.shared.scheduler);
            let width = if scheduler.has_waiting_decode() {
                MAX_TEACHER_FORCED_TOKENS
            } else {
                PREFILL_BATCH_TOKENS
            };
            let end = (offset + width).min(tokens.len());
            let _permit = scheduler.acquire(self.sequence_id, AcceleratorWork::Prefill)?;
            let chunk = &tokens[offset..end];
            let (next_logits, rows) = self.forward_batch_capturing(chunk, layer_ids)?;
            logits = next_logits;
            captured.extend_from_slice(&rows);
            offset = end;
        }
        Ok((logits, captured))
    }

    /// Prefill a token-only DFlash prompt while keeping selected target rows
    /// in shared Metal storage. The assistant consumes chunk N on its own
    /// queue while this target executes chunk N+1. Only the newest target row
    /// crosses back to the CPU because it is the sole incremental row needed
    /// by the first speculative round.
    pub(crate) fn forward_dflash_prompt_pipelined(
        &mut self,
        tokens: &[u32],
        layer_ids: &[usize],
        assistant: &mut crate::metal::dflash::MetalDFlashForward,
        cache: &mut crate::dflash::DFlashContextKvCache,
    ) -> Result<
        (
            Vec<f32>,
            Vec<f32>,
            crate::metal::dflash::DFlashPromptPipelineStats,
        ),
        MetalModelError,
    > {
        if tokens.is_empty()
            || layer_ids.is_empty()
            || layer_ids.iter().any(|&layer| layer >= self.cfg.n_layers)
            || layer_ids.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(MetalModelError::InvalidSnapshot(
                "DFlash prompt pipeline requires tokens and ordered valid capture layers".into(),
            ));
        }
        let pipeline_error = |error: crate::dflash::DFlashError| {
            MetalModelError::InvalidSnapshot(format!("DFlash prompt pipeline: {error}"))
        };
        assistant
            .begin_prompt_pipeline(cache, PREFILL_BATCH_TOKENS)
            .map_err(pipeline_error)?;
        let run = (|| {
            let mut logits = Vec::new();
            let mut newest_hidden = Vec::with_capacity(layer_ids.len() * self.cfg.hidden_dim);
            let prefix_rows = tokens.len() - 1;
            let mut offset = 0usize;
            let mut chunk_index = 0usize;
            while offset < tokens.len() {
                let scheduler = Arc::clone(&self.shared.scheduler);
                let width = if scheduler.has_waiting_decode() {
                    MAX_TEACHER_FORCED_TOKENS
                } else {
                    PREFILL_BATCH_TOKENS
                };
                let end = (offset + width).min(tokens.len());
                let chunk = &tokens[offset..end];
                let retained = retained_prompt_subranges(
                    offset,
                    end.min(prefix_rows),
                    prefix_rows,
                    cache.sink_size,
                    cache.window_size,
                );
                let capture_required = !retained.is_empty() || end == tokens.len();
                let slot = chunk_index % 2;
                let capture = if capture_required {
                    Some(
                        assistant
                            .prompt_capture_slot(cache, slot, chunk.len())
                            .map_err(pipeline_error)?,
                    )
                } else {
                    None
                };
                let next_logits = {
                    let _permit = scheduler.acquire(self.sequence_id, AcceleratorWork::Prefill)?;
                    match capture.as_ref() {
                        Some(capture) => {
                            self.forward_batch_capturing_layer_major(chunk, layer_ids, capture)?
                        }
                        None => self.forward_batch(chunk)?,
                    }
                };
                logits = next_logits;

                if end == tokens.len() {
                    let hidden = self.cfg.hidden_dim;
                    newest_hidden.clear();
                    let capture = capture.as_ref().ok_or_else(|| {
                        MetalModelError::InvalidSnapshot(
                            "final DFlash prompt chunk was not captured".into(),
                        )
                    })?;
                    for layer in 0..layer_ids.len() {
                        let start = (layer * chunk.len() + chunk.len() - 1) * hidden;
                        newest_hidden.extend_from_slice(&capture.as_slice()[start..start + hidden]);
                    }
                }
                for (source_start, absolute_start, output_rows) in retained {
                    if cache.ctx_offset != absolute_start {
                        assistant
                            .advance_prompt_pipeline_to(cache, absolute_start)
                            .map_err(pipeline_error)?;
                    }
                    assistant
                        .enqueue_prompt_chunk(cache, slot, chunk.len(), source_start, output_rows)
                        .map_err(pipeline_error)?;
                }
                offset = end;
                chunk_index += 1;
            }
            if newest_hidden.len() != layer_ids.len() * self.cfg.hidden_dim {
                return Err(MetalModelError::InvalidSnapshot(
                    "DFlash prompt pipeline did not retain the newest target row".into(),
                ));
            }
            Ok((logits, newest_hidden))
        })();
        match run {
            Ok((logits, newest_hidden)) => assistant
                .finish_prompt_pipeline(cache)
                .map(|stats| (logits, newest_hidden, stats))
                .map_err(pipeline_error),
            Err(primary) => match assistant.abort_prompt_pipeline() {
                Ok(()) => Err(primary),
                Err(abort) => Err(MetalModelError::InvalidSnapshot(format!(
                    "{primary}; DFlash prompt abort also failed: {abort}"
                ))),
            },
        }
    }

    /// Return output-normalized decoder rows without projecting them back to
    /// vocabulary space. The production batch graph already owns these rows;
    /// this method performs one bounded readback for the embedding API.
    pub fn forward_final_hidden(
        &mut self,
        tokens: &[u32],
    ) -> Result<(Vec<f32>, Vec<f32>), MetalModelError> {
        let mut logits = Vec::new();
        let mut hidden = Vec::new();
        let mut offset = 0;
        while offset < tokens.len() {
            let scheduler = Arc::clone(&self.shared.scheduler);
            let width = if scheduler.has_waiting_decode() {
                MAX_TEACHER_FORCED_TOKENS
            } else {
                PREFILL_BATCH_TOKENS
            };
            let end = (offset + width).min(tokens.len());
            let _permit = scheduler.acquire(self.sequence_id, AcceleratorWork::Prefill)?;
            let chunk = &tokens[offset..end];
            let token_count = chunk.len();
            let start_position = self.n_past;
            let mut workspace = self.take_batch_workspace(token_count)?;
            for (destination, token) in workspace
                .token_ids
                .as_mut_bytes()
                .chunks_exact_mut(std::mem::size_of::<u32>())
                .zip(chunk)
            {
                destination.copy_from_slice(&token.to_le_bytes());
            }
            let token_view = workspace
                .token_ids
                .view(0, std::mem::size_of_val(chunk))
                .expect("complete token buffer");
            let queue = self.context.queue.clone();
            let command = new_prefill_command_buffer(&queue);
            let serial = new_prefill_graph_encoder(command, self.concurrent_prefill_dispatch);
            dispatch(&serial, |encoder| {
                self.kernels.encode_embedding_q4k(
                    encoder,
                    self.embedding.view(&self.mapped_weights),
                    &token_view,
                    &workspace.activations.hidden,
                    self.cfg.hidden_dim,
                    self.cfg.vocab_size,
                    token_count,
                );
            });
            let result = self.forward_batch_hidden(
                &workspace.activations,
                &workspace.swa_staged_key,
                &workspace.swa_staged_value,
                workspace.fa_prefill_scratch(),
                token_count,
                start_position,
                &serial,
                command,
                &[],
                &[],
                None,
                None,
            );
            match result {
                Ok((next_logits, _)) => {
                    logits = next_logits;
                    hidden.clear();
                    hidden.extend_from_slice(
                        &workspace.activations.hidden.as_slice()
                            [..token_count * self.cfg.hidden_dim],
                    );
                    self.return_batch_workspace(token_count, workspace);
                }
                Err(error) => {
                    self.return_batch_workspace(token_count, workspace);
                    return Err(error);
                }
            }
            offset = end;
        }
        Ok((logits, hidden))
    }

    /// Capture the exact production handoff tensors immediately after Muse's
    /// sigmoid attention gate, plus the Metal result after the same layer's
    /// output/FFN sandwich-norm tail. This is a qualification surface for the
    /// target-decoder ANE partition, not part of the stable engine API.
    pub fn forward_batch_capturing_swa_tail(
        &mut self,
        tokens: &[u32],
        layer: usize,
    ) -> Result<MuseSwaTailCapture, MetalModelError> {
        if tokens.len() != MAX_DFLASH_BLOCK
            || layer >= self.cfg.n_layers
            || !self.cfg.layer_kinds[layer].is_swa()
        {
            return Err(MetalModelError::InvalidSnapshot(
                "Muse target ANE capture requires 16 tokens and an SWA layer".into(),
            ));
        }
        let token_count = tokens.len();
        let start_position = self.n_past;
        let mut workspace = self.take_batch_workspace(token_count)?;
        for (destination, token) in workspace
            .token_ids
            .as_mut_bytes()
            .chunks_exact_mut(std::mem::size_of::<u32>())
            .zip(tokens)
        {
            destination.copy_from_slice(&token.to_le_bytes());
        }
        let token_view = workspace
            .token_ids
            .view(0, std::mem::size_of_val(tokens))
            .expect("complete target ANE capture token buffer");
        let attention = GpuBuffer::zeros(&self.context, token_count * self.cfg.attn_dim())?;
        let residual = GpuBuffer::zeros(&self.context, token_count * self.cfg.hidden_dim)?;
        let final_hidden = GpuBuffer::zeros(&self.context, token_count * self.cfg.hidden_dim)?;
        let queue = self.context.queue.clone();
        let command_buffer = queue.new_command_buffer();
        let serial = GraphEncoder::serial(command_buffer.new_compute_command_encoder());
        dispatch(&serial, |encoder| {
            self.kernels.encode_embedding_q4k(
                encoder,
                self.embedding.view(&self.mapped_weights),
                &token_view,
                &workspace.activations.hidden,
                self.cfg.hidden_dim,
                self.cfg.vocab_size,
                token_count,
            );
        });
        let result = self.forward_batch_hidden(
            &workspace.activations,
            &workspace.swa_staged_key,
            &workspace.swa_staged_value,
            workspace.fa_prefill_scratch(),
            token_count,
            start_position,
            &serial,
            command_buffer,
            &[layer],
            std::slice::from_ref(&final_hidden),
            None,
            Some(BatchTailCapture {
                layer,
                attention: &attention,
                residual: &residual,
                debug: None,
            }),
        );
        let output = result.map(|_| MuseSwaTailCapture {
            layer,
            attention: attention.as_slice().to_vec(),
            residual: residual.as_slice().to_vec(),
            metal_hidden: final_hidden.as_slice().to_vec(),
        });
        self.return_batch_workspace(token_count, workspace);
        output
    }

    /// Capture named f32 boundaries for one production batch graph layer.
    ///
    /// This is diagnostic-only plumbing for byte-for-byte comparator
    /// bisection. Serving does not call it.
    #[doc(hidden)]
    pub fn forward_capturing_debug_layer(
        &mut self,
        tokens: &[u32],
        layer: usize,
    ) -> Result<Vec<(&'static str, Vec<f32>)>, MetalModelError> {
        if tokens.is_empty() || layer >= self.cfg.n_layers {
            return Err(MetalModelError::InvalidSnapshot(
                "debug layer capture requires tokens and a valid layer".into(),
            ));
        }
        let token_count = tokens.len();
        let start_position = self.n_past;
        let mut workspace = self.take_batch_workspace(token_count)?;
        for (destination, token) in workspace
            .token_ids
            .as_mut_bytes()
            .chunks_exact_mut(std::mem::size_of::<u32>())
            .zip(tokens)
        {
            destination.copy_from_slice(&token.to_le_bytes());
        }
        let token_view = workspace
            .token_ids
            .view(0, std::mem::size_of_val(tokens))
            .expect("complete debug token buffer");
        let hidden = self.cfg.hidden_dim;
        let attn = self.cfg.attn_dim();
        let kv = self.cfg.kv_dim();
        let embedding = GpuBuffer::zeros(&self.context, token_count * hidden)?;
        let entry_norm = GpuBuffer::zeros(&self.context, token_count * hidden)?;
        let attn_norm = GpuBuffer::zeros(&self.context, token_count * hidden)?;
        let q = GpuBuffer::zeros(&self.context, token_count * attn)?;
        let k = GpuBuffer::zeros(&self.context, token_count * kv)?;
        let v = GpuBuffer::zeros(&self.context, token_count * kv)?;
        let gate = GpuBuffer::zeros(&self.context, token_count * attn)?;
        let q_norm = GpuBuffer::zeros(&self.context, token_count * attn)?;
        let k_norm = GpuBuffer::zeros(&self.context, token_count * kv)?;
        let q_rope = GpuBuffer::zeros(&self.context, token_count * attn)?;
        let k_rope = GpuBuffer::zeros(&self.context, token_count * kv)?;
        let attention_raw = GpuBuffer::zeros(&self.context, token_count * attn)?;
        let attention_gated = GpuBuffer::zeros(&self.context, token_count * attn)?;
        let attn_o_proj = GpuBuffer::zeros(&self.context, token_count * hidden)?;
        let ffn_inp = GpuBuffer::zeros(&self.context, token_count * hidden)?;
        let ffn_norm = GpuBuffer::zeros(&self.context, token_count * hidden)?;
        let ffn_gate = GpuBuffer::zeros(&self.context, token_count * self.cfg.intermediate_dim)?;
        let ffn_up = GpuBuffer::zeros(&self.context, token_count * self.cfg.intermediate_dim)?;
        let ffn_swiglu = GpuBuffer::zeros(&self.context, token_count * self.cfg.intermediate_dim)?;
        let ffn_out = GpuBuffer::zeros(&self.context, token_count * hidden)?;
        let result_norm = GpuBuffer::zeros(&self.context, token_count * hidden)?;
        let result_projection = GpuBuffer::zeros(&self.context, token_count * self.cfg.vocab_size)?;
        let residual = GpuBuffer::zeros(&self.context, token_count * hidden)?;
        let layer_out = GpuBuffer::zeros(&self.context, token_count * hidden)?;
        let queue = self.context.queue.clone();
        let command_buffer = queue.new_command_buffer();
        let serial = GraphEncoder::serial(command_buffer.new_compute_command_encoder());
        dispatch(&serial, |encoder| {
            self.kernels.encode_embedding_q4k(
                encoder,
                self.embedding.view(&self.mapped_weights),
                &token_view,
                &workspace.activations.hidden,
                hidden,
                self.cfg.vocab_size,
                token_count,
            );
        });
        let debug_capture = || {
            Some(BatchTailCapture {
                layer,
                attention: &attention_gated,
                residual: &residual,
                debug: Some(DebugLayerCapture {
                    embedding: &embedding,
                    entry_norm: &entry_norm,
                    attn_norm: &attn_norm,
                    q: &q,
                    k: &k,
                    v: &v,
                    gate: &gate,
                    q_norm: &q_norm,
                    k_norm: &k_norm,
                    q_rope: &q_rope,
                    k_rope: &k_rope,
                    attention_raw: &attention_raw,
                    attn_o_proj: &attn_o_proj,
                    ffn_inp: &ffn_inp,
                    ffn_norm: &ffn_norm,
                    ffn_gate: &ffn_gate,
                    ffn_up: &ffn_up,
                    ffn_swiglu: &ffn_swiglu,
                    ffn_out: &ffn_out,
                    result_norm: &result_norm,
                    result_projection: &result_projection,
                }),
            })
        };
        let result = if std::env::var_os("MUSER_DEBUG_STOP_AFTER_LAYER").is_some() {
            self.encode_batch_hidden_range(
                &workspace.activations,
                &workspace.swa_staged_key,
                &workspace.swa_staged_value,
                workspace.fa_prefill_scratch(),
                token_count,
                start_position,
                &serial,
                &[layer],
                std::slice::from_ref(&layer_out),
                None,
                None,
                debug_capture(),
                0..layer + 1,
                true,
                false,
            )
            .and_then(|()| {
                serial.encoder.end_encoding();
                command_buffer.commit();
                self.context
                    .wait_for_completion(command_buffer, Duration::from_secs(90))?;
                Ok((Vec::new(), Vec::new()))
            })
        } else {
            self.forward_batch_hidden(
                &workspace.activations,
                &workspace.swa_staged_key,
                &workspace.swa_staged_value,
                workspace.fa_prefill_scratch(),
                token_count,
                start_position,
                &serial,
                command_buffer,
                &[layer],
                std::slice::from_ref(&layer_out),
                None,
                debug_capture(),
            )
        };
        let output = result.map(|_| {
            [
                ("embedding", &embedding),
                ("entry_norm", &entry_norm),
                ("attn_norm-0", &attn_norm),
                ("Qcur-0", &q),
                ("Kcur-0", &k),
                ("Vcur-0", &v),
                ("attn_gate_proj-0", &gate),
                ("Qcur_normed-0", &q_norm),
                ("Kcur_normed-0", &k_norm),
                ("Qcur_rope-0", &q_rope),
                ("Kcur_rope-0", &k_rope),
                ("attn_out-0", &attention_raw),
                ("attn_gated-0", &attention_gated),
                ("attn_o_proj-0", &attn_o_proj),
                ("ffn_inp-0", &ffn_inp),
                ("ffn_norm-0", &ffn_norm),
                ("ffn_gate-0", &ffn_gate),
                ("ffn_up-0", &ffn_up),
                ("ffn_swiglu-0", &ffn_swiglu),
                ("ffn_out-0", &ffn_out),
                ("l_out-0", &layer_out),
                ("result_norm", &result_norm),
                ("result_projection", &result_projection),
            ]
            .into_iter()
            .map(|(name, buffer)| (name, buffer.as_slice().to_vec()))
            .collect()
        });
        self.return_batch_workspace(token_count, workspace);
        output
    }

    /// Execute only the Metal-owned reference tail corresponding to
    /// [`forward_batch_capturing_swa_tail`]. This makes the ANE partition's
    /// speed comparison independent of attention and the remaining layers.
    pub fn run_swa_tail_metal(
        &mut self,
        layer_index: usize,
        attention: &[f32],
        residual: &[f32],
    ) -> Result<MuseSwaTailMetalResult, MetalModelError> {
        let token_count = MAX_DFLASH_BLOCK;
        if layer_index >= self.cfg.n_layers
            || !self.cfg.layer_kinds[layer_index].is_swa()
            || attention.len() != token_count * self.cfg.attn_dim()
            || residual.len() != token_count * self.cfg.hidden_dim
        {
            return Err(MetalModelError::InvalidSnapshot(
                "Muse target ANE Metal-tail geometry differs".into(),
            ));
        }
        let mut workspace = self.take_batch_workspace(token_count)?;
        workspace
            .activations
            .attention
            .as_mut_slice()
            .copy_from_slice(attention);
        workspace
            .activations
            .normed
            .as_mut_slice()
            .copy_from_slice(residual);
        let layer = &self.layers[layer_index];
        let queue = self.context.queue.clone();
        let started = Instant::now();
        let command_buffer = queue.new_command_buffer();
        let serial = GraphEncoder::serial(command_buffer.new_compute_command_encoder());
        self.project_tokens(
            &serial,
            &layer.output,
            &workspace.activations.attention,
            &workspace.activations.projected,
            token_count,
        );
        dispatch(&serial, |encoder| {
            self.kernels.encode_fused_rms_norm_residual_add_batch(
                encoder,
                &workspace.activations.normed,
                &workspace.activations.projected,
                &layer.post_attn_norm,
                self.cfg.hidden_dim,
                self.cfg.post_norm_eps,
                token_count,
            );
        });
        dispatch(&serial, |encoder| {
            self.kernels.encode_rms_norm_mul(
                encoder,
                &workspace.activations.normed,
                &layer.ffn_norm,
                &workspace.activations.post_norm,
                self.cfg.hidden_dim,
                self.cfg.rms_eps,
                token_count,
            );
        });
        self.project_tokens(
            &serial,
            &layer.ffn_gate,
            &workspace.activations.post_norm,
            &workspace.activations.ffn_gate,
            token_count,
        );
        self.project_tokens(
            &serial,
            &layer.ffn_up,
            &workspace.activations.post_norm,
            &workspace.activations.ffn_up,
            token_count,
        );
        dispatch(&serial, |encoder| {
            self.kernels.encode_silu_mul(
                encoder,
                &workspace.activations.ffn_gate,
                &workspace.activations.ffn_up,
            );
        });
        self.project_tokens(
            &serial,
            &layer.ffn_down,
            &workspace.activations.ffn_gate,
            &workspace.activations.projected,
            token_count,
        );
        dispatch(&serial, |encoder| {
            self.kernels.encode_fused_rms_norm_residual_add_batch(
                encoder,
                &workspace.activations.normed,
                &workspace.activations.projected,
                &layer.post_ffn_norm,
                self.cfg.hidden_dim,
                self.cfg.post_norm_eps,
                token_count,
            );
        });
        serial.encoder.end_encoding();
        command_buffer.commit();
        let completed = self
            .context
            .wait_for_completion(command_buffer, Duration::from_secs(300))
            .map(|_| MuseSwaTailMetalResult {
                hidden: workspace.activations.normed.as_slice().to_vec(),
                wall_ns: started.elapsed().as_nanos() as u64,
            });
        self.return_batch_workspace(token_count, workspace);
        completed.map_err(Into::into)
    }

    /// Enter the decoder immediately after token embedding. Vision projector
    /// rows use the same entry RMSNorm and the same absolute positions as
    /// ordinary token embeddings.
    pub fn forward_embeddings(&mut self, embeddings: &[f32]) -> Result<Vec<f32>, MetalModelError> {
        debug_assert_eq!(embeddings.len() % self.cfg.hidden_dim, 0);
        let mut logits = Vec::new();
        for chunk in embeddings.chunks(PREFILL_BATCH_TOKENS * self.cfg.hidden_dim) {
            let scheduler = Arc::clone(&self.shared.scheduler);
            let _permit = scheduler.acquire(self.sequence_id, AcceleratorWork::Prefill)?;
            logits = self.forward_batch_embeddings(chunk)?;
        }
        Ok(logits)
    }

    /// Target-layer capture for projected image rows, preserving the exact
    /// token-major layout consumed by the DFlash assistant.
    pub fn forward_embeddings_capturing_layers(
        &mut self,
        embeddings: &[f32],
        layer_ids: &[usize],
    ) -> Result<(Vec<f32>, Vec<f32>), MetalModelError> {
        debug_assert_eq!(embeddings.len() % self.cfg.hidden_dim, 0);
        let mut logits = Vec::new();
        let mut captured =
            Vec::with_capacity(embeddings.len() / self.cfg.hidden_dim * layer_ids.len());
        let queue = self.context.queue.clone();
        for chunk in embeddings.chunks(PREFILL_BATCH_TOKENS * self.cfg.hidden_dim) {
            let scheduler = Arc::clone(&self.shared.scheduler);
            let _permit = scheduler.acquire(self.sequence_id, AcceleratorWork::Prefill)?;
            let token_count = chunk.len() / self.cfg.hidden_dim;
            let start_position = self.n_past;
            let mut workspace = self.take_batch_workspace(token_count)?;
            workspace.ensure_capture_buffers(
                &self.context,
                &self.cfg,
                token_count,
                layer_ids.len(),
            )?;
            workspace
                .activations
                .hidden
                .as_mut_slice()
                .copy_from_slice(chunk);
            let command = queue.new_command_buffer();
            let serial = GraphEncoder::serial(command.new_compute_command_encoder());
            let result = self.forward_batch_hidden(
                &workspace.activations,
                &workspace.swa_staged_key,
                &workspace.swa_staged_value,
                workspace.fa_prefill_scratch(),
                token_count,
                start_position,
                &serial,
                command,
                layer_ids,
                &workspace.capture_buffers[..layer_ids.len()],
                None,
                None,
            );
            self.return_batch_workspace(token_count, workspace);
            let (next_logits, rows) = result?;
            logits = next_logits;
            captured.extend_from_slice(&rows);
        }
        Ok((logits, captured))
    }

    fn forward_batch(&mut self, tokens: &[u32]) -> Result<Vec<f32>, MetalModelError> {
        debug_assert!(!tokens.is_empty());
        let profile_started =
            (tokens.len() == 1 && stream_decode_profile_enabled()).then(Instant::now);
        if profile_started.is_some() {
            STREAM_DECODE_DIAGNOSTICS.with(|slot| *slot.borrow_mut() = None);
        }
        let token_count = tokens.len();
        let start_position = self.n_past;
        let mut workspace = self.take_batch_workspace(token_count)?;
        let cfg = &self.cfg;
        for (destination, token) in workspace
            .token_ids
            .as_mut_bytes()
            .chunks_exact_mut(std::mem::size_of::<u32>())
            .zip(tokens)
        {
            destination.copy_from_slice(&token.to_le_bytes());
        }
        let token_view = workspace
            .token_ids
            .view(0, std::mem::size_of_val(tokens))
            .expect("complete token buffer");
        let queue = self.context.queue.clone();
        if std::env::var_os("MUSER_METAL_BATCH_PHASE_PROFILE").is_some() {
            let labels = self.batch_phase_labels(token_count, &[], false);
            let profiler = PhaseProfiler::new(queue);
            dispatch(&profiler, |encoder| {
                self.kernels.encode_embedding_q4k(
                    encoder,
                    self.embedding.view(&self.mapped_weights),
                    &token_view,
                    &workspace.activations.hidden,
                    cfg.hidden_dim,
                    cfg.vocab_size,
                    token_count,
                );
            });
            let encoded = self.encode_batch_hidden(
                &workspace.activations,
                &workspace.swa_staged_key,
                &workspace.swa_staged_value,
                workspace.fa_prefill_scratch(),
                token_count,
                start_position,
                &profiler,
                &[],
                &[],
                None,
                None,
            );
            profiler.print_batch_report(&labels);
            let result =
                encoded.and_then(|()| self.finish_batch_hidden(token_count, &[], &[], None));
            self.return_batch_workspace(token_count, workspace);
            return result.map(|(logits, _)| logits);
        }
        let command = new_prefill_command_buffer(&queue);

        let serial = new_prefill_graph_encoder(command, self.concurrent_prefill_dispatch);
        dispatch(&serial, |encoder| {
            self.kernels.encode_embedding_q4k(
                encoder,
                self.embedding.view(&self.mapped_weights),
                &token_view,
                &workspace.activations.hidden,
                cfg.hidden_dim,
                cfg.vocab_size,
                token_count,
            );
        });
        let result = if let Some(profile_started) = profile_started {
            self.forward_batch_hidden_profiled(
                &workspace.activations,
                &workspace.swa_staged_key,
                &workspace.swa_staged_value,
                workspace.fa_prefill_scratch(),
                token_count,
                start_position,
                &serial,
                command,
                profile_started,
            )
            .map(|(logits, diagnostics)| {
                install_stream_decode_diagnostics(diagnostics);
                (logits, Vec::new())
            })
        } else {
            self.forward_batch_hidden(
                &workspace.activations,
                &workspace.swa_staged_key,
                &workspace.swa_staged_value,
                workspace.fa_prefill_scratch(),
                token_count,
                start_position,
                &serial,
                command,
                &[],
                &[],
                None,
                None,
            )
        };
        self.return_batch_workspace(token_count, workspace);
        Ok(result?.0)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_batch_hidden_profiled(
        &mut self,
        batch: &BatchActivations,
        swa_staged_key: &GpuHalfBuffer,
        swa_staged_value: &GpuHalfBuffer,
        fa_prefill: Option<(&GpuBytes, &GpuBytes)>,
        token_count: usize,
        start_position: usize,
        command: &GraphEncoder<'_>,
        command_buffer: &metal::CommandBufferRef,
        profile_started: Instant,
    ) -> Result<(Vec<f32>, crate::api::DecodeDiagnostics), MetalModelError> {
        let prepared = Instant::now();
        self.encode_batch_hidden(
            batch,
            swa_staged_key,
            swa_staged_value,
            fa_prefill,
            token_count,
            start_position,
            command,
            &[],
            &[],
            None,
            None,
        )?;
        let encoded = Instant::now();
        command.encoder.end_encoding();
        let encoder_ended = Instant::now();
        command_buffer.commit();
        let committed = Instant::now();
        self.context
            .wait_for_completion(command_buffer, Duration::from_secs(300))?;
        let waited = Instant::now();
        let (logits, _) = self.finish_batch_hidden(token_count, &[], &[], None)?;
        let read_back = Instant::now();
        Ok((
            logits,
            crate::api::DecodeDiagnostics {
                model_prepare_ns: prepared.duration_since(profile_started).as_nanos() as u64,
                model_encode_ns: encoded.duration_since(prepared).as_nanos() as u64,
                encoder_end_ns: encoder_ended.duration_since(encoded).as_nanos() as u64,
                command_commit_ns: committed.duration_since(encoder_ended).as_nanos() as u64,
                gpu_wait_ns: waited.duration_since(committed).as_nanos() as u64,
                logits_readback_ns: read_back.duration_since(waited).as_nanos() as u64,
                ..crate::api::DecodeDiagnostics::default()
            },
        ))
    }

    fn forward_batch_capturing(
        &mut self,
        tokens: &[u32],
        layer_ids: &[usize],
    ) -> Result<(Vec<f32>, Vec<f32>), MetalModelError> {
        let token_count = tokens.len();
        let start_position = self.n_past;
        let mut workspace = self.take_batch_workspace(token_count)?;
        workspace.ensure_capture_buffers(&self.context, &self.cfg, token_count, layer_ids.len())?;
        let cfg = &self.cfg;
        for (destination, token) in workspace
            .token_ids
            .as_mut_bytes()
            .chunks_exact_mut(std::mem::size_of::<u32>())
            .zip(tokens)
        {
            destination.copy_from_slice(&token.to_le_bytes());
        }
        let token_view = workspace
            .token_ids
            .view(0, std::mem::size_of_val(tokens))
            .expect("complete token buffer");
        let queue = self.context.queue.clone();
        let command = queue.new_command_buffer();
        let serial = new_prefill_graph_encoder(command, self.concurrent_prefill_dispatch);
        dispatch(&serial, |encoder| {
            self.kernels.encode_embedding_q4k(
                encoder,
                self.embedding.view(&self.mapped_weights),
                &token_view,
                &workspace.activations.hidden,
                cfg.hidden_dim,
                cfg.vocab_size,
                token_count,
            );
        });
        let result = self.forward_batch_hidden(
            &workspace.activations,
            &workspace.swa_staged_key,
            &workspace.swa_staged_value,
            workspace.fa_prefill_scratch(),
            token_count,
            start_position,
            &serial,
            command,
            layer_ids,
            &workspace.capture_buffers[..layer_ids.len()],
            None,
            None,
        );
        self.return_batch_workspace(token_count, workspace);
        result
    }

    fn forward_batch_capturing_layer_major(
        &mut self,
        tokens: &[u32],
        layer_ids: &[usize],
        capture: &GpuBuffer,
    ) -> Result<Vec<f32>, MetalModelError> {
        let token_count = tokens.len();
        let needed = token_count
            .checked_mul(layer_ids.len())
            .and_then(|value| value.checked_mul(self.cfg.hidden_dim))
            .ok_or_else(|| {
                MetalModelError::InvalidSnapshot("DFlash capture size overflow".into())
            })?;
        if token_count == 0 || capture.len() < needed {
            return Err(MetalModelError::InvalidSnapshot(
                "DFlash layer-major capture buffer is too small".into(),
            ));
        }
        let start_position = self.n_past;
        let mut workspace = self.take_batch_workspace(token_count)?;
        for (destination, token) in workspace
            .token_ids
            .as_mut_bytes()
            .chunks_exact_mut(std::mem::size_of::<u32>())
            .zip(tokens)
        {
            destination.copy_from_slice(&token.to_le_bytes());
        }
        let token_view = workspace
            .token_ids
            .view(0, std::mem::size_of_val(tokens))
            .expect("complete DFlash prompt token buffer");
        let queue = self.context.queue.clone();
        let command = queue.new_command_buffer();
        let serial = new_prefill_graph_encoder(command, self.concurrent_prefill_dispatch);
        dispatch(&serial, |encoder| {
            self.kernels.encode_embedding_q4k(
                encoder,
                self.embedding.view(&self.mapped_weights),
                &token_view,
                &workspace.activations.hidden,
                self.cfg.hidden_dim,
                self.cfg.vocab_size,
                token_count,
            );
        });
        let result = self
            .encode_batch_hidden_range(
                &workspace.activations,
                &workspace.swa_staged_key,
                &workspace.swa_staged_value,
                workspace.fa_prefill_scratch(),
                token_count,
                start_position,
                &serial,
                layer_ids,
                &[],
                Some(capture),
                None,
                None,
                0..self.cfg.n_layers,
                true,
                true,
            )
            .and_then(|()| {
                serial.encoder.end_encoding();
                command.commit();
                self.context
                    .wait_for_completion(command, Duration::from_secs(300))?;
                self.finish_batch_hidden(token_count, &[], &[], None)
                    .map(|(logits, _)| logits)
            });
        self.return_batch_workspace(token_count, workspace);
        result
    }

    /// Ferrite-derived speculative verifier surface: run the complete target
    /// block in one batch graph, project every position, and capture the
    /// selected target layers in the same command buffer.
    pub(crate) fn forward_batch_all_logits_capturing(
        &mut self,
        tokens: &[u32],
        layer_ids: &[usize],
    ) -> Result<(Vec<f32>, Vec<f32>), MetalModelError> {
        if tokens.is_empty() || tokens.len() > MAX_DFLASH_BLOCK {
            return Err(MetalModelError::InvalidSnapshot(format!(
                "verification batch length {} is outside 1..={MAX_DFLASH_BLOCK}",
                tokens.len()
            )));
        }
        self.print_verify_route_banner(tokens.len());
        let scheduler = Arc::clone(&self.shared.scheduler);
        let _permit = scheduler.acquire(self.sequence_id, AcceleratorWork::Decode)?;
        let token_count = tokens.len();
        let start_position = self.n_past;
        let mut workspace = self.take_batch_workspace(token_count)?;
        workspace.ensure_capture_buffers(&self.context, &self.cfg, token_count, layer_ids.len())?;
        for (destination, token) in workspace
            .token_ids
            .as_mut_bytes()
            .chunks_exact_mut(std::mem::size_of::<u32>())
            .zip(tokens)
        {
            destination.copy_from_slice(&token.to_le_bytes());
        }
        let token_view = workspace
            .token_ids
            .view(0, std::mem::size_of_val(tokens))
            .expect("complete verification token buffer");
        let logits_len = token_count
            .checked_mul(self.cfg.vocab_size)
            .ok_or_else(|| MetalModelError::InvalidSnapshot("batch logits overflow".into()))?;
        let batch_logits = workspace.verify_logits.clone();
        debug_assert!(batch_logits.len() >= logits_len);
        let queue = self.context.queue.clone();
        if std::env::var_os("MUSER_METAL_BATCH_PHASE_PROFILE").is_some() {
            let labels = self.batch_phase_labels(token_count, layer_ids, true);
            let profiler = PhaseProfiler::new(queue);
            dispatch(&profiler, |encoder| {
                self.kernels.encode_embedding_q4k(
                    encoder,
                    self.embedding.view(&self.mapped_weights),
                    &token_view,
                    &workspace.activations.hidden,
                    self.cfg.hidden_dim,
                    self.cfg.vocab_size,
                    token_count,
                );
            });
            let encoded = self.encode_batch_hidden(
                &workspace.activations,
                &workspace.swa_staged_key,
                &workspace.swa_staged_value,
                workspace.fa_prefill_scratch(),
                token_count,
                start_position,
                &profiler,
                layer_ids,
                &workspace.capture_buffers[..layer_ids.len()],
                Some(&batch_logits),
                None,
            );
            profiler.print_batch_report(&labels);
            let result = encoded.and_then(|()| {
                self.finish_batch_hidden(
                    token_count,
                    layer_ids,
                    &workspace.capture_buffers[..layer_ids.len()],
                    Some(&batch_logits),
                )
            });
            self.return_batch_workspace(token_count, workspace);
            return result;
        }
        let command_buffer = queue.new_command_buffer();
        let serial = GraphEncoder::serial(command_buffer.new_compute_command_encoder());
        dispatch(&serial, |encoder| {
            self.kernels.encode_embedding_q4k(
                encoder,
                self.embedding.view(&self.mapped_weights),
                &token_view,
                &workspace.activations.hidden,
                self.cfg.hidden_dim,
                self.cfg.vocab_size,
                token_count,
            );
        });
        let result = self.forward_batch_hidden(
            &workspace.activations,
            &workspace.swa_staged_key,
            &workspace.swa_staged_value,
            workspace.fa_prefill_scratch(),
            token_count,
            start_position,
            &serial,
            command_buffer,
            layer_ids,
            &workspace.capture_buffers[..layer_ids.len()],
            Some(&batch_logits),
            None,
        );
        self.return_batch_workspace(token_count, workspace);
        result
    }

    fn print_verify_route_banner(&mut self, rows: usize) {
        if self.verify_route_banner_printed {
            return;
        }
        let layout = &self.layers[0].q.layout;
        let projection = match layout.dtype {
            GgmlType::NVFP4_E2M1 if layout.nvfp4_input_scale_inv.is_none() => {
                "nvfp4-weight-only-a16-q8"
            }
            GgmlType::NVFP4_E2M1
                if rows == 16
                    && layout.n_in.is_multiple_of(64)
                    && std::env::var_os("MUSER_NO_M16_N32").is_none() =>
            {
                "nvfp4-w4a4-prequant-m16-n32"
            }
            GgmlType::NVFP4_E2M1 => "nvfp4-w4a4-decomposed",
            GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K
                if rows == 16
                    && layout.n_in.is_multiple_of(256)
                    && layout.n_out.is_multiple_of(32)
                    && std::env::var_os("MUSER_NO_M16_N32").is_none() =>
            {
                "kquant-m16-n32"
            }
            GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K => "kquant-generic",
            GgmlType::F16 => "f16",
            _ => "unsupported",
        };
        eprintln!(
            "MUSER_VERIFY_ROUTE rows={rows} projection={projection} dtype={:?} activation_quant={} exact_override={} m16_disabled={}",
            layout.dtype,
            layout.nvfp4_input_scale_inv.is_some(),
            std::env::var_os("MUSER_CROSS_VENDOR_QK").is_some(),
            std::env::var_os("MUSER_NO_M16_N32").is_some(),
        );
        self.verify_route_banner_printed = true;
    }

    /// Execute through `capture_end` synchronously, then submit the remaining
    /// target layers and LM head without waiting. The returned hidden rows are
    /// exact target activations and are stable before ANE sees them. This is
    /// the narrow public-Metal half of Mirror-SD; no target result is accepted
    /// until [`Self::finish_dflash_verify_suffix`] succeeds.
    pub(crate) fn begin_dflash_verify_suffix(
        &mut self,
        tokens: &[u32],
        layer_ids: &[usize],
        capture_end: usize,
        capture_fc: Option<&dyn crate::dflash::DFlashProjectionBackend>,
    ) -> Result<MetalDFlashVerifyOverlap, MetalModelError> {
        if let Some(backend) = capture_fc {
            return self.begin_dflash_verify_suffix_capture_fc(tokens, layer_ids, backend);
        }
        let wall_started = Instant::now();
        if tokens.is_empty()
            || tokens.len() > MAX_DFLASH_BLOCK
            || capture_end == 0
            || capture_end >= self.cfg.n_layers
            || layer_ids.iter().any(|&layer| layer >= capture_end)
        {
            return Err(MetalModelError::InvalidSnapshot(
                "Mirror-SD verification split is outside the target graph".into(),
            ));
        }
        let scheduler = Arc::clone(&self.shared.scheduler);
        let permit = scheduler.acquire(self.sequence_id, AcceleratorWork::Decode)?;
        let token_count = tokens.len();
        let start_position = self.n_past;
        let mut workspace = self.take_batch_workspace(token_count)?;
        workspace.ensure_capture_buffers(&self.context, &self.cfg, token_count, layer_ids.len())?;
        for (destination, token) in workspace
            .token_ids
            .as_mut_bytes()
            .chunks_exact_mut(std::mem::size_of::<u32>())
            .zip(tokens)
        {
            destination.copy_from_slice(&token.to_le_bytes());
        }
        let token_view = workspace
            .token_ids
            .view(0, std::mem::size_of_val(tokens))
            .expect("complete Mirror-SD token buffer");
        let queue = self.context.queue.clone();
        let prefix = queue.new_command_buffer();
        let prefix_encoder = GraphEncoder::serial(prefix.new_compute_command_encoder());
        dispatch(&prefix_encoder, |encoder| {
            self.kernels.encode_embedding_q4k(
                encoder,
                self.embedding.view(&self.mapped_weights),
                &token_view,
                &workspace.activations.hidden,
                self.cfg.hidden_dim,
                self.cfg.vocab_size,
                token_count,
            );
        });
        self.encode_batch_hidden_range(
            &workspace.activations,
            &workspace.swa_staged_key,
            &workspace.swa_staged_value,
            workspace.fa_prefill_scratch(),
            token_count,
            start_position,
            &prefix_encoder,
            layer_ids,
            &workspace.capture_buffers[..layer_ids.len()],
            None,
            None,
            None,
            0..capture_end,
            true,
            false,
        )?;
        prefix_encoder.encoder.end_encoding();
        prefix.commit();
        self.context
            .wait_for_completion(prefix, Duration::from_secs(300))?;

        let mut captured = Vec::with_capacity(token_count * layer_ids.len() * self.cfg.hidden_dim);
        for token in 0..token_count {
            for buffer in &workspace.capture_buffers[..layer_ids.len()] {
                let start = token * self.cfg.hidden_dim;
                captured.extend_from_slice(&buffer.as_slice()[start..start + self.cfg.hidden_dim]);
            }
        }

        let logits_len = token_count
            .checked_mul(self.cfg.vocab_size)
            .ok_or_else(|| MetalModelError::InvalidSnapshot("batch logits overflow".into()))?;
        let batch_logits = workspace.verify_logits.clone();
        debug_assert!(batch_logits.len() >= logits_len);
        let suffix = queue.new_command_buffer();
        let suffix_encoder = GraphEncoder::serial(suffix.new_compute_command_encoder());
        // A command-buffer boundary is also a graph boundary. Do not rely on
        // the fused layer-49 tail's secondary normalized output surviving as
        // an implicit input to layer 50: materialize the suffix entry norm
        // from the authoritative residual again. This is mathematically the
        // same operation as the ordinary unfused graph and makes the split
        // verifier independent of cross-command temporary-buffer lifetime.
        if self.fused_prefill_dual_norm {
            let boundary = &self.layers[capture_end];
            dispatch(&suffix_encoder, |encoder| {
                self.kernels.encode_rms_norm_mul(
                    encoder,
                    &workspace.activations.normed,
                    &boundary.attn_norm,
                    &workspace.activations.post_norm,
                    self.cfg.hidden_dim,
                    self.cfg.rms_eps,
                    token_count,
                );
            });
        }
        self.encode_batch_hidden_range(
            &workspace.activations,
            &workspace.swa_staged_key,
            &workspace.swa_staged_value,
            workspace.fa_prefill_scratch(),
            token_count,
            start_position,
            &suffix_encoder,
            layer_ids,
            &workspace.capture_buffers[..layer_ids.len()],
            None,
            Some(&batch_logits),
            None,
            capture_end..self.cfg.n_layers,
            false,
            true,
        )?;
        suffix_encoder.encoder.end_encoding();
        suffix.commit();
        Ok((
            PendingMetalDFlashVerify {
                command: suffix.to_owned(),
                workspace,
                batch_logits,
                token_count,
                submitted_wall_ns: wall_started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                capture_fc_ns: 0,
                _permit: permit,
            },
            captured,
            None,
        ))
    }

    fn begin_dflash_verify_suffix_capture_fc(
        &mut self,
        tokens: &[u32],
        layer_ids: &[usize],
        backend: &dyn crate::dflash::DFlashProjectionBackend,
    ) -> Result<MetalDFlashVerifyOverlap, MetalModelError> {
        if tokens.is_empty()
            || tokens.len() > MAX_DFLASH_BLOCK
            || layer_ids.is_empty()
            || layer_ids.iter().any(|&layer| layer >= self.cfg.n_layers)
            || layer_ids.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(MetalModelError::InvalidSnapshot(
                "capture-FC pipeline requires ordered target layers".into(),
            ));
        }
        let scheduler = Arc::clone(&self.shared.scheduler);
        let permit = scheduler.acquire(self.sequence_id, AcceleratorWork::Decode)?;
        let token_count = tokens.len();
        let start_position = self.n_past;
        let mut workspace = self.take_batch_workspace(token_count)?;
        workspace.ensure_capture_buffers(&self.context, &self.cfg, token_count, layer_ids.len())?;
        for (destination, token) in workspace
            .token_ids
            .as_mut_bytes()
            .chunks_exact_mut(std::mem::size_of::<u32>())
            .zip(tokens)
        {
            destination.copy_from_slice(&token.to_le_bytes());
        }
        let token_view = workspace
            .token_ids
            .view(0, std::mem::size_of_val(tokens))
            .expect("complete capture-FC token buffer");
        let queue = self.context.queue.clone();

        let first_end = layer_ids[0] + 1;
        let first = queue.new_command_buffer();
        let first_encoder = GraphEncoder::serial(first.new_compute_command_encoder());
        dispatch(&first_encoder, |encoder| {
            self.kernels.encode_embedding_q4k(
                encoder,
                self.embedding.view(&self.mapped_weights),
                &token_view,
                &workspace.activations.hidden,
                self.cfg.hidden_dim,
                self.cfg.vocab_size,
                token_count,
            );
        });
        self.encode_batch_hidden_range(
            &workspace.activations,
            &workspace.swa_staged_key,
            &workspace.swa_staged_value,
            workspace.fa_prefill_scratch(),
            token_count,
            start_position,
            &first_encoder,
            layer_ids,
            &workspace.capture_buffers[..layer_ids.len()],
            None,
            None,
            None,
            0..first_end,
            true,
            false,
        )?;
        first_encoder.encoder.end_encoding();
        first.commit();
        let mut active = first.to_owned();
        let mut segment_start = first_end;
        let mut accumulated = vec![0.0f32; token_count * self.cfg.hidden_dim];
        let batch_logits = workspace.verify_logits.clone();
        let mut prefix_gpu_ns = 0u64;
        let mut capture_fc_ns = 0u64;

        for (capture, &layer) in layer_ids.iter().enumerate() {
            self.context
                .wait_for_completion(&active, Duration::from_secs(300))?;
            prefix_gpu_ns = prefix_gpu_ns.saturating_add(unsafe {
                let start: f64 = objc::msg_send![active, GPUStartTime];
                let end: f64 = objc::msg_send![active, GPUEndTime];
                ((end - start).max(0.0) * 1.0e9).min(u64::MAX as f64) as u64
            });
            let layer_rows = workspace.capture_buffers[capture].as_slice()
                [..token_count * self.cfg.hidden_dim]
                .to_vec();

            let next = queue.new_command_buffer();
            let next_encoder = GraphEncoder::serial(next.new_compute_command_encoder());
            if self.fused_prefill_dual_norm {
                let boundary = &self.layers[segment_start];
                dispatch(&next_encoder, |encoder| {
                    self.kernels.encode_rms_norm_mul(
                        encoder,
                        &workspace.activations.normed,
                        &boundary.attn_norm,
                        &workspace.activations.post_norm,
                        self.cfg.hidden_dim,
                        self.cfg.rms_eps,
                        token_count,
                    );
                });
            }
            let (end, output) = match layer_ids.get(capture + 1) {
                Some(&next_layer) => (next_layer + 1, false),
                None => (self.cfg.n_layers, true),
            };
            debug_assert_eq!(layer + 1, segment_start);
            self.encode_batch_hidden_range(
                &workspace.activations,
                &workspace.swa_staged_key,
                &workspace.swa_staged_value,
                workspace.fa_prefill_scratch(),
                token_count,
                start_position,
                &next_encoder,
                layer_ids,
                &workspace.capture_buffers[..layer_ids.len()],
                None,
                output.then_some(&batch_logits),
                None,
                segment_start..end,
                false,
                output,
            )?;
            next_encoder.encoder.end_encoding();
            next.commit();
            active = next.to_owned();
            segment_start = end;

            let capture_fc_started = Instant::now();
            let contribution = match backend.project_capture_fc_slice(capture, &layer_rows) {
                Ok(Some(values)) if values.len() == accumulated.len() => values,
                Ok(Some(values)) => {
                    let error = MetalModelError::InvalidSnapshot(format!(
                        "capture-FC slice {capture} returned {} elements, expected {}",
                        values.len(),
                        accumulated.len()
                    ));
                    self.context
                        .wait_for_completion(&active, Duration::from_secs(300))?;
                    self.return_batch_workspace(token_count, workspace);
                    return Err(error);
                }
                Ok(None) => {
                    self.context
                        .wait_for_completion(&active, Duration::from_secs(300))?;
                    self.return_batch_workspace(token_count, workspace);
                    return Err(MetalModelError::InvalidSnapshot(
                        "capture-FC backend declined a declared v8 slice".into(),
                    ));
                }
                Err(error) => {
                    self.context
                        .wait_for_completion(&active, Duration::from_secs(300))?;
                    self.return_batch_workspace(token_count, workspace);
                    return Err(MetalModelError::InvalidSnapshot(error));
                }
            };
            capture_fc_ns = capture_fc_ns.saturating_add(
                capture_fc_started
                    .elapsed()
                    .as_nanos()
                    .min(u64::MAX as u128) as u64,
            );
            for (sum, value) in accumulated.iter_mut().zip(contribution) {
                *sum += value;
            }
        }

        let mut captured = Vec::with_capacity(token_count * layer_ids.len() * self.cfg.hidden_dim);
        for token in 0..token_count {
            for buffer in &workspace.capture_buffers[..layer_ids.len()] {
                let start = token * self.cfg.hidden_dim;
                captured.extend_from_slice(&buffer.as_slice()[start..start + self.cfg.hidden_dim]);
            }
        }
        Ok((
            PendingMetalDFlashVerify {
                command: active,
                workspace,
                batch_logits,
                token_count,
                submitted_wall_ns: prefix_gpu_ns,
                capture_fc_ns,
                _permit: permit,
            },
            captured,
            Some(accumulated),
        ))
    }

    pub(crate) fn finish_dflash_verify_suffix(
        &mut self,
        pending: PendingMetalDFlashVerify,
    ) -> Result<(Vec<f32>, u64, u64), MetalModelError> {
        let PendingMetalDFlashVerify {
            command,
            workspace,
            batch_logits,
            token_count,
            submitted_wall_ns,
            capture_fc_ns,
            _permit,
        } = pending;
        let completed = self
            .context
            .wait_for_completion(&command, Duration::from_secs(300));
        let result = completed.map(|()| {
            self.n_past += token_count;
            let suffix_gpu_ns = unsafe {
                let start: f64 = objc::msg_send![command, GPUStartTime];
                let end: f64 = objc::msg_send![command, GPUEndTime];
                ((end - start).max(0.0) * 1.0e9).min(u64::MAX as f64) as u64
            };
            (
                batch_logits.as_slice()[..token_count * self.cfg.vocab_size].to_vec(),
                submitted_wall_ns.saturating_add(suffix_gpu_ns),
                capture_fc_ns,
            )
        });
        self.return_batch_workspace(token_count, workspace);
        Ok(result?)
    }

    fn batch_phase_labels(
        &self,
        token_count: usize,
        capture_layers: &[usize],
        all_logits: bool,
    ) -> Vec<String> {
        let mut labels = vec!["embedding".to_owned(), "entry_norm".to_owned()];
        let mut llama_fa_prefill_mask_labeled = false;
        for (layer, kind) in self.cfg.layer_kinds.iter().enumerate() {
            let prefix = format!("layer.{layer}");
            if layer == 0 || !self.fused_prefill_dual_norm {
                labels.push(format!("{prefix}.attn_norm"));
            }
            labels.push(format!("{prefix}.qkvg"));
            labels.push(format!("{prefix}.qk_norm"));
            if kind.uses_rope() {
                labels.push(format!("{prefix}.rope"));
            }
            let plane = &self.cache[layer];
            let contiguous = plane.origin_logical == 0
                && plane.origin_physical == 0
                && plane.len + token_count <= plane.capacity;
            if contiguous {
                let visible = plane.len + token_count;
                let llama_prefill = token_count >= 20
                    && !kind.is_swa()
                    && plane.head_major
                    && token_count
                        .is_multiple_of(crate::metal::encode::LLAMA_FA_PREFILL_NQPTG as usize)
                    && visible
                        .is_multiple_of(crate::metal::encode::LLAMA_FA_PREFILL_NCPSG as usize)
                    && self.llama_fa_prefill_available(token_count);
                labels.push(format!("{prefix}.kv_store"));
                if llama_prefill && !llama_fa_prefill_mask_labeled {
                    labels.push("fa_prefill_mask_blk".to_owned());
                    llama_fa_prefill_mask_labeled = true;
                }
                labels.push(format!("{prefix}.attention"));
            } else if kind.is_swa() {
                labels.push(format!("{prefix}.swa_stage"));
                labels.push(format!("{prefix}.attention"));
            } else {
                labels.push(format!("{prefix}.attention"));
                labels.push(format!("{prefix}.kv_store"));
            }
            for operation in ["sigmoid_gate", "output"] {
                labels.push(format!("{prefix}.{operation}"));
            }
            if self.fused_prefill_dual_norm {
                labels.push(format!("{prefix}.post_attn_ffn_norm"));
            } else {
                labels.push(format!("{prefix}.post_attn_norm"));
                labels.push(format!("{prefix}.ffn_norm"));
            }
            for operation in ["ffn_gate_up", "silu", "ffn_down"] {
                labels.push(format!("{prefix}.{operation}"));
            }
            labels.push(format!(
                "{prefix}.{}",
                if self.fused_prefill_dual_norm {
                    "post_ffn_next_norm"
                } else {
                    "post_ffn_norm"
                }
            ));
            if capture_layers.contains(&layer) {
                labels.push(format!("{prefix}.capture"));
            }
        }
        if !self.fused_prefill_dual_norm {
            labels.push("output_norm".to_owned());
        }
        if !all_logits && token_count > 1 {
            labels.push("copy_last_row".to_owned());
        }
        labels.push(if all_logits {
            "lm_head_batch".to_owned()
        } else {
            "lm_head".to_owned()
        });
        labels.push("softcap".to_owned());
        labels
    }

    fn forward_batch_embeddings(
        &mut self,
        embeddings: &[f32],
    ) -> Result<Vec<f32>, MetalModelError> {
        debug_assert!(!embeddings.is_empty());
        debug_assert_eq!(embeddings.len() % self.cfg.hidden_dim, 0);
        let token_count = embeddings.len() / self.cfg.hidden_dim;
        let start_position = self.n_past;
        let mut workspace = self.take_batch_workspace(token_count)?;
        workspace
            .activations
            .hidden
            .as_mut_slice()
            .copy_from_slice(embeddings);
        let queue = self.context.queue.clone();
        let command = queue.new_command_buffer();
        let serial = new_prefill_graph_encoder(command, self.concurrent_prefill_dispatch);
        let result = self.forward_batch_hidden(
            &workspace.activations,
            &workspace.swa_staged_key,
            &workspace.swa_staged_value,
            workspace.fa_prefill_scratch(),
            token_count,
            start_position,
            &serial,
            command,
            &[],
            &[],
            None,
            None,
        );
        self.return_batch_workspace(token_count, workspace);
        Ok(result?.0)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_batch_hidden(
        &mut self,
        batch: &BatchActivations,
        swa_staged_key: &GpuHalfBuffer,
        swa_staged_value: &GpuHalfBuffer,
        fa_prefill: Option<(&GpuBytes, &GpuBytes)>,
        token_count: usize,
        start_position: usize,
        command: &GraphEncoder<'_>,
        command_buffer: &metal::CommandBufferRef,
        capture_layers: &[usize],
        capture_buffers: &[GpuBuffer],
        batch_logits: Option<&GpuBuffer>,
        tail_capture: Option<BatchTailCapture<'_>>,
    ) -> Result<(Vec<f32>, Vec<f32>), MetalModelError> {
        self.encode_batch_hidden(
            batch,
            swa_staged_key,
            swa_staged_value,
            fa_prefill,
            token_count,
            start_position,
            command,
            capture_layers,
            capture_buffers,
            batch_logits,
            tail_capture,
        )?;
        command.encoder.end_encoding();
        command_buffer.commit();
        self.context
            .wait_for_completion(command_buffer, Duration::from_secs(300))?;
        self.finish_batch_hidden(token_count, capture_layers, capture_buffers, batch_logits)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_batch_hidden<T: EncodeTarget + ?Sized>(
        &mut self,
        batch: &BatchActivations,
        swa_staged_key: &GpuHalfBuffer,
        swa_staged_value: &GpuHalfBuffer,
        fa_prefill: Option<(&GpuBytes, &GpuBytes)>,
        token_count: usize,
        start_position: usize,
        command: &T,
        capture_layers: &[usize],
        capture_buffers: &[GpuBuffer],
        batch_logits: Option<&GpuBuffer>,
        tail_capture: Option<BatchTailCapture<'_>>,
    ) -> Result<(), MetalModelError> {
        self.encode_batch_hidden_range(
            batch,
            swa_staged_key,
            swa_staged_value,
            fa_prefill,
            token_count,
            start_position,
            command,
            capture_layers,
            capture_buffers,
            None,
            batch_logits,
            tail_capture,
            0..self.cfg.n_layers,
            true,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_batch_hidden_range<T: EncodeTarget + ?Sized>(
        &mut self,
        batch: &BatchActivations,
        swa_staged_key: &GpuHalfBuffer,
        swa_staged_value: &GpuHalfBuffer,
        fa_prefill: Option<(&GpuBytes, &GpuBytes)>,
        token_count: usize,
        start_position: usize,
        command: &T,
        capture_layers: &[usize],
        capture_buffers: &[GpuBuffer],
        layer_major_capture: Option<&GpuBuffer>,
        batch_logits: Option<&GpuBuffer>,
        tail_capture: Option<BatchTailCapture<'_>>,
        layers: Range<usize>,
        encode_entry: bool,
        encode_output: bool,
    ) -> Result<(), MetalModelError> {
        let cfg = &self.cfg;
        debug_assert!(
            layer_major_capture.is_some() || capture_buffers.len() == capture_layers.len()
        );
        debug_assert!(
            layer_major_capture.is_none() || capture_buffers.is_empty(),
            "capture has one GPU layout owner"
        );
        debug_assert!(layers.start <= layers.end && layers.end <= cfg.n_layers);
        if encode_entry {
            dispatch(command, |encoder| {
                self.kernels.encode_rms_norm_mul(
                    encoder,
                    &batch.hidden,
                    &self.entry_norm_ones,
                    &batch.normed,
                    cfg.hidden_dim,
                    cfg.rms_eps,
                    token_count,
                );
            });
            if let Some(debug) = tail_capture
                .as_ref()
                .and_then(|capture| capture.debug.as_ref())
            {
                dispatch(command, |encoder| {
                    self.kernels.encode_copy_f32(
                        encoder,
                        &batch.hidden,
                        debug.embedding,
                        token_count * cfg.hidden_dim,
                    );
                    self.kernels.encode_copy_f32(
                        encoder,
                        &batch.normed,
                        debug.entry_norm,
                        token_count * cfg.hidden_dim,
                    );
                });
            }
        }

        // The pinned llama non-vec prefill route shares one causal mask +
        // block classifier across every eligible full-attention layer in
        // this chunk; filled lazily at the first eligible layer.
        let mut llama_fa_prefill_mask_ready = false;
        for layer_index in layers {
            let layer = self.layers[layer_index].clone();
            // Layer N-1's fused tail already produced layer N's normalized
            // attention input. Layer zero has no predecessor.
            if layer_index == 0 || !self.fused_prefill_dual_norm {
                dispatch(command, |encoder| {
                    self.kernels.encode_rms_norm_mul(
                        encoder,
                        &batch.normed,
                        &layer.attn_norm,
                        &batch.post_norm,
                        cfg.hidden_dim,
                        cfg.rms_eps,
                        token_count,
                    );
                });
            }
            if let Some(debug) = tail_capture
                .as_ref()
                .filter(|capture| capture.layer == layer_index)
                .and_then(|capture| capture.debug.as_ref())
            {
                dispatch(command, |encoder| {
                    self.kernels.encode_copy_f32(
                        encoder,
                        &batch.post_norm,
                        debug.attn_norm,
                        token_count * cfg.hidden_dim,
                    );
                });
            }
            // Independent projections share a read-only normalized input and
            // write disjoint activations. Group them so a concurrent prefill
            // encoder can overlap the four GEMMs; a serial encoder still
            // submits them in order inside one closure.
            dispatch(command, |encoder| {
                self.encode_batch_projection(
                    encoder,
                    &layer.q,
                    &batch.post_norm,
                    &batch.q,
                    token_count,
                    batch,
                );
                self.encode_batch_projection(
                    encoder,
                    &layer.k,
                    &batch.post_norm,
                    &batch.k,
                    token_count,
                    batch,
                );
                self.encode_batch_projection(
                    encoder,
                    &layer.v,
                    &batch.post_norm,
                    &batch.v,
                    token_count,
                    batch,
                );
                self.encode_batch_projection(
                    encoder,
                    &layer.gate,
                    &batch.post_norm,
                    &batch.gate,
                    token_count,
                    batch,
                );
            });
            if let Some(debug) = tail_capture
                .as_ref()
                .filter(|capture| capture.layer == layer_index)
                .and_then(|capture| capture.debug.as_ref())
            {
                dispatch(command, |encoder| {
                    for (source, destination, count) in [
                        (&batch.q, debug.q, token_count * cfg.attn_dim()),
                        (&batch.k, debug.k, token_count * cfg.kv_dim()),
                        (&batch.v, debug.v, token_count * cfg.kv_dim()),
                        (&batch.gate, debug.gate, token_count * cfg.attn_dim()),
                    ] {
                        self.kernels
                            .encode_copy_f32(encoder, source, destination, count);
                    }
                });
            }
            dispatch(command, |encoder| {
                self.kernels.encode_qk_norm(
                    encoder,
                    &batch.q,
                    &layer.q_norm,
                    &batch.q,
                    cfg.head_dim,
                    cfg.rms_eps,
                    token_count * cfg.n_heads,
                );
                self.kernels.encode_qk_norm(
                    encoder,
                    &batch.k,
                    &layer.k_norm,
                    &batch.k,
                    cfg.head_dim,
                    cfg.rms_eps,
                    token_count * cfg.n_kv_heads,
                );
            });
            if let Some(debug) = tail_capture
                .as_ref()
                .filter(|capture| capture.layer == layer_index)
                .and_then(|capture| capture.debug.as_ref())
            {
                dispatch(command, |encoder| {
                    self.kernels.encode_copy_f32(
                        encoder,
                        &batch.q,
                        debug.q_norm,
                        token_count * cfg.attn_dim(),
                    );
                    self.kernels.encode_copy_f32(
                        encoder,
                        &batch.k,
                        debug.k_norm,
                        token_count * cfg.kv_dim(),
                    );
                });
            }
            if cfg.layer_kinds[layer_index].uses_rope() {
                dispatch(command, |encoder| {
                    self.kernels.encode_rope_norm_batch_cached(
                        encoder,
                        &batch.q,
                        &batch.k,
                        &self.rope_frequencies,
                        cfg.n_heads,
                        cfg.n_kv_heads,
                        cfg.head_dim,
                        start_position,
                        token_count,
                        self.rope_positions.view(
                            start_position * std::mem::size_of::<u32>(),
                            token_count * std::mem::size_of::<u32>(),
                        ),
                        cfg.rope_base_swa,
                        cfg.context_length,
                    );
                });
            }
            if let Some(debug) = tail_capture
                .as_ref()
                .filter(|capture| capture.layer == layer_index)
                .and_then(|capture| capture.debug.as_ref())
            {
                dispatch(command, |encoder| {
                    self.kernels.encode_copy_f32(
                        encoder,
                        &batch.q,
                        debug.q_rope,
                        token_count * cfg.attn_dim(),
                    );
                    self.kernels.encode_copy_f32(
                        encoder,
                        &batch.k,
                        debug.k_rope,
                        token_count * cfg.kv_dim(),
                    );
                });
            }

            let (old_origin_logical, old_origin_physical, old_len, capacity) = {
                let plane = &self.cache[layer_index];
                (
                    plane.origin_logical,
                    plane.origin_physical,
                    plane.len,
                    plane.capacity,
                )
            };
            let window = if cfg.layer_kinds[layer_index].is_swa() {
                cfg.sliding_window
            } else {
                0
            };
            let flash_contiguous = old_origin_logical == 0
                && old_origin_physical == 0
                && old_len + token_count <= capacity;
            if flash_contiguous {
                // Ferrite's production order is store-to-F16 then FA2. This
                // also makes prefill arithmetic match the live cache used by
                // subsequent decode rather than attending transient F32 K/V.
                let (source_first, source_count) = self.cache[layer_index].append_batch(
                    layer_index,
                    start_position,
                    token_count,
                )?;
                let plane = &self.cache[layer_index];
                // The pinned Metal backend selects its vec FA kernel for
                // batches below 20 queries.  Running one unmasked vec launch
                // per query with an exact visible-prefix length is equivalent
                // to its causal mask (NQPSG=1 for DK128), while reusing the
                // exact upstream PSO and reduction order.  This matters for
                // public embedding/logprob parity: the older local FA2 path
                // was mathematically close but diverged sharply after four
                // positions across 52 layers.  SWA cache rows are token-major
                // in Muser, so stage this short initial batch head-major before
                // calling the pinned kernel. Longer and resumed prefills keep
                // the production FA2 route below.
                let pinned_vec = llama_vec_prefill_route_available(
                    token_count,
                    plane.capacity,
                    self.kernels.has_llama_flash_attn_vec(),
                    std::env::var_os("MUSER_CROSS_VENDOR_QK").is_some(),
                );
                dispatch(command, |encoder| {
                    self.kernels.encode_kv_store_batch_f16(
                        encoder,
                        &batch.k,
                        &batch.v,
                        &plane.key,
                        &plane.value,
                        cfg.kv_dim(),
                        token_count,
                        source_first,
                        source_count,
                        start_position,
                        plane.capacity,
                        plane.origin_logical,
                        plane.origin_physical,
                        cfg.head_dim,
                        plane.head_major,
                    );
                });
                if pinned_vec {
                    dispatch(command, |encoder| {
                        for row in 0..token_count {
                            self.kernels.encode_llama_flash_attn_decode_vec_f16(
                                encoder,
                                &batch.q,
                                &plane.key,
                                &plane.value,
                                &self.activations.attention_mask,
                                &self.activations.attention_kv_pad,
                                &self.activations.attention_partials,
                                &batch.attention,
                                cfg.n_heads,
                                cfg.n_kv_heads,
                                cfg.head_dim,
                                start_position + row + 1,
                                plane.capacity,
                                0,
                                plane.origin_physical,
                                false,
                                cfg.attn_scale(),
                                plane.head_major,
                                row,
                                row,
                            );
                        }
                    });
                } else {
                    // Full NoPE layers at llama-shaped chunk bounds take the
                    // pinned non-vec `flash_attn_ext`: same kernel, tiling,
                    // and reduction order the comparator measures. SWA layers
                    // and unaligned shapes keep the local FA2 route.
                    let visible = start_position + token_count;
                    let llama_prefill = window == 0
                        && plane.head_major
                        && token_count
                            .is_multiple_of(crate::metal::encode::LLAMA_FA_PREFILL_NQPTG as usize)
                        && visible
                            .is_multiple_of(crate::metal::encode::LLAMA_FA_PREFILL_NCPSG as usize)
                        && fa_prefill.is_some();
                    if llama_prefill {
                        let (mask, blk) = fa_prefill.expect("route checked above");
                        if !llama_fa_prefill_mask_ready {
                            dispatch(command, |encoder| {
                                self.kernels.encode_llama_fa_prefill_mask_blk(
                                    encoder,
                                    mask,
                                    blk,
                                    start_position,
                                    token_count,
                                );
                            });
                            llama_fa_prefill_mask_ready = true;
                        }
                        dispatch(command, |encoder| {
                            self.kernels.encode_llama_flash_attn_prefill_f16(
                                encoder,
                                &batch.q,
                                &plane.key,
                                &plane.value,
                                mask,
                                blk,
                                &batch.attention,
                                token_count,
                                cfg.n_heads,
                                cfg.n_kv_heads,
                                cfg.head_dim,
                                start_position,
                                plane.capacity,
                                cfg.attn_scale(),
                            );
                        });
                    } else {
                        dispatch(command, |encoder| {
                            self.kernels.encode_flash_attention_v2(
                                encoder,
                                &batch.q,
                                &plane.key,
                                &plane.value,
                                &batch.attention,
                                token_count,
                                cfg.n_heads,
                                cfg.n_kv_heads,
                                cfg.head_dim,
                                start_position,
                                plane.capacity,
                                0,
                                window,
                                cfg.attn_scale(),
                                plane.head_major,
                            );
                        });
                    }
                }
            } else if cfg.layer_kinds[layer_index].is_swa() {
                // Preserve Ferrite FA2 after the explicit SWA ring wraps.
                // The staging arena is a detached logical tail: old ring rows
                // in logical order followed by this chunk, all F16 exactly as
                // the production cache stores them. The live ring is changed
                // only after attention has consumed the complete shadow.
                let staged_capacity = swa_staged_key.len() / cfg.kv_dim();
                debug_assert!(old_len + token_count <= staged_capacity);
                debug_assert!(!self.cache[layer_index].head_major);
                let llama_exact_single_row = token_count == 1
                    && self.kernels.has_llama_flash_attn_vec()
                    && capacity == cfg.sliding_window
                    && start_position < cfg.context_length
                    && std::env::var_os("MUSER_CROSS_VENDOR_QK").is_none();
                dispatch(command, |encoder| {
                    let plane = &self.cache[layer_index];
                    if llama_exact_single_row {
                        self.kernels.encode_stage_swa_llama_decode_f16(
                            encoder,
                            &batch.k,
                            &batch.v,
                            &plane.key,
                            &plane.value,
                            swa_staged_key,
                            swa_staged_value,
                            &self.activations.swa_llama_mask,
                            cfg.kv_dim(),
                            old_len,
                            old_origin_logical,
                            old_origin_physical,
                            capacity,
                            start_position,
                        );
                    } else {
                        self.kernels.encode_stage_swa_prefill_f16(
                            encoder,
                            &batch.k,
                            &batch.v,
                            &plane.key,
                            &plane.value,
                            swa_staged_key,
                            swa_staged_value,
                            cfg.kv_dim(),
                            old_len,
                            old_origin_physical,
                            capacity,
                            token_count,
                        );
                    }
                });
                dispatch(command, |encoder| {
                    if llama_exact_single_row {
                        let visible = (start_position + 1).div_ceil(256) * 256;
                        let staged: [&metal::ResourceRef; 3] = [
                            swa_staged_key.metal(),
                            swa_staged_value.metal(),
                            self.activations.swa_llama_mask.metal(),
                        ];
                        encoder.memory_barrier_with_resources(&staged);
                        self.kernels.encode_llama_flash_attn_decode_vec_f16(
                            encoder,
                            &batch.q,
                            swa_staged_key,
                            swa_staged_value,
                            &self.activations.swa_llama_mask,
                            &self.activations.attention_kv_pad_masked,
                            &self.activations.attention_partials,
                            &batch.attention,
                            cfg.n_heads,
                            cfg.n_kv_heads,
                            cfg.head_dim,
                            visible,
                            staged_capacity,
                            0,
                            old_origin_physical,
                            true,
                            cfg.attn_scale(),
                            false,
                            0,
                            0,
                        );
                    } else {
                        self.kernels.encode_flash_attention_v2(
                            encoder,
                            &batch.q,
                            swa_staged_key,
                            swa_staged_value,
                            &batch.attention,
                            token_count,
                            cfg.n_heads,
                            cfg.n_kv_heads,
                            cfg.head_dim,
                            start_position,
                            staged_capacity,
                            old_origin_logical,
                            window,
                            cfg.attn_scale(),
                            false,
                        );
                    }
                });
                let (_source_first, _source_count) = self.cache[layer_index].append_batch(
                    layer_index,
                    start_position,
                    token_count,
                )?;
            } else {
                // A growing NoPE plane cannot reach this route under valid
                // context bounds. Retain the explicit map as a fail-safe.
                // Attend before overwriting any still-visible old rows.
                dispatch(command, |encoder| {
                    let plane = &self.cache[layer_index];
                    self.kernels.encode_attention_prefill_f32(
                        encoder,
                        &batch.q,
                        &batch.k,
                        &batch.v,
                        &plane.key,
                        &plane.value,
                        &batch.attention,
                        token_count,
                        cfg.n_heads,
                        cfg.n_kv_heads,
                        cfg.head_dim,
                        start_position,
                        capacity,
                        old_origin_logical,
                        old_origin_physical,
                        old_len,
                        window,
                        cfg.attn_scale(),
                        plane.head_major,
                    );
                });
                let (source_first, source_count) = self.cache[layer_index].append_batch(
                    layer_index,
                    start_position,
                    token_count,
                )?;
                let plane = &self.cache[layer_index];
                dispatch(command, |encoder| {
                    self.kernels.encode_kv_store_batch_f16(
                        encoder,
                        &batch.k,
                        &batch.v,
                        &plane.key,
                        &plane.value,
                        cfg.kv_dim(),
                        token_count,
                        source_first,
                        source_count,
                        start_position,
                        plane.capacity,
                        plane.origin_logical,
                        plane.origin_physical,
                        cfg.head_dim,
                        plane.head_major,
                    );
                });
            }
            if let Some(debug) = tail_capture
                .as_ref()
                .filter(|capture| capture.layer == layer_index)
                .and_then(|capture| capture.debug.as_ref())
            {
                dispatch(command, |encoder| {
                    self.kernels.encode_copy_f32(
                        encoder,
                        &batch.attention,
                        debug.attention_raw,
                        token_count * cfg.attn_dim(),
                    );
                });
            }
            dispatch(command, |encoder| {
                self.kernels
                    .encode_sigmoid_gate(encoder, &batch.attention, &batch.gate);
            });
            if let Some(capture) = tail_capture
                .as_ref()
                .filter(|capture| capture.layer == layer_index)
            {
                dispatch(command, |encoder| {
                    self.kernels.encode_copy_f32(
                        encoder,
                        &batch.attention,
                        capture.attention,
                        token_count * cfg.attn_dim(),
                    );
                    self.kernels.encode_copy_f32(
                        encoder,
                        &batch.normed,
                        capture.residual,
                        token_count * cfg.hidden_dim,
                    );
                });
            }
            self.project_batch_tokens(
                command,
                &layer.output,
                &batch.attention,
                &batch.projected,
                token_count,
                batch,
            );
            if let Some(debug) = tail_capture
                .as_ref()
                .filter(|capture| capture.layer == layer_index)
                .and_then(|capture| capture.debug.as_ref())
            {
                dispatch(command, |encoder| {
                    self.kernels.encode_copy_f32(
                        encoder,
                        &batch.projected,
                        debug.attn_o_proj,
                        token_count * cfg.hidden_dim,
                    );
                });
            }
            // llama.cpp gathers the requested output row after the final
            // post-attention norm and before its residual add
            // (models/muse-glimmer.cpp). That get_rows boundary prevents the
            // Metal rms_norm+mul+add fusion used by earlier layers. Preserve
            // the same two rounding points for the one-row output graph.
            // DFlash/capture graphs with no output, multi-row decode, and the
            // cross-vendor route retain their existing contracts.
            let llama_final_row_boundary = token_count == 1
                && encode_output
                && start_position >= cfg.sliding_window
                && layer_index + 1 == cfg.n_layers
                && self.kernels.has_llama_flash_attn_vec()
                && std::env::var_os("MUSER_CROSS_VENDOR_QK").is_none();
            if llama_final_row_boundary {
                dispatch(command, |encoder| {
                    self.kernels.encode_rms_norm_mul(
                        encoder,
                        &batch.projected,
                        &layer.post_attn_norm,
                        &batch.post_norm,
                        cfg.hidden_dim,
                        cfg.post_norm_eps,
                        token_count,
                    );
                });
                dispatch(command, |encoder| {
                    self.kernels.encode_residual_add_batch(
                        encoder,
                        &batch.normed,
                        &batch.post_norm,
                        token_count * cfg.hidden_dim,
                    );
                });
                dispatch(command, |encoder| {
                    self.kernels.encode_rms_norm_mul(
                        encoder,
                        &batch.normed,
                        &layer.ffn_norm,
                        &batch.post_norm,
                        cfg.hidden_dim,
                        cfg.rms_eps,
                        token_count,
                    );
                });
            } else if self.fused_prefill_dual_norm {
                dispatch(command, |encoder| {
                    self.kernels
                        .encode_fused_norm_residual_rms_norm_batch_dual_eps(
                            encoder,
                            &batch.normed,
                            &batch.projected,
                            &batch.post_norm,
                            &layer.post_attn_norm,
                            &layer.ffn_norm,
                            cfg.hidden_dim,
                            cfg.post_norm_eps,
                            cfg.rms_eps,
                            token_count,
                        );
                });
            } else {
                dispatch(command, |encoder| {
                    self.kernels.encode_fused_rms_norm_residual_add_batch(
                        encoder,
                        &batch.normed,
                        &batch.projected,
                        &layer.post_attn_norm,
                        cfg.hidden_dim,
                        cfg.post_norm_eps,
                        token_count,
                    );
                });
                dispatch(command, |encoder| {
                    self.kernels.encode_rms_norm_mul(
                        encoder,
                        &batch.normed,
                        &layer.ffn_norm,
                        &batch.post_norm,
                        cfg.hidden_dim,
                        cfg.rms_eps,
                        token_count,
                    );
                });
            }
            if let Some(debug) = tail_capture
                .as_ref()
                .filter(|capture| capture.layer == layer_index)
                .and_then(|capture| capture.debug.as_ref())
            {
                dispatch(command, |encoder| {
                    self.kernels.encode_copy_f32(
                        encoder,
                        &batch.normed,
                        debug.ffn_inp,
                        token_count * cfg.hidden_dim,
                    );
                    self.kernels.encode_copy_f32(
                        encoder,
                        &batch.post_norm,
                        debug.ffn_norm,
                        token_count * cfg.hidden_dim,
                    );
                });
            }
            dispatch(command, |encoder| {
                self.encode_batch_projection(
                    encoder,
                    &layer.ffn_gate,
                    &batch.post_norm,
                    &batch.ffn_gate,
                    token_count,
                    batch,
                );
                self.encode_batch_projection(
                    encoder,
                    &layer.ffn_up,
                    &batch.post_norm,
                    &batch.ffn_up,
                    token_count,
                    batch,
                );
            });
            if let Some(debug) = tail_capture
                .as_ref()
                .filter(|capture| capture.layer == layer_index)
                .and_then(|capture| capture.debug.as_ref())
            {
                dispatch(command, |encoder| {
                    self.kernels.encode_copy_f32(
                        encoder,
                        &batch.ffn_gate,
                        debug.ffn_gate,
                        token_count * cfg.intermediate_dim,
                    );
                    self.kernels.encode_copy_f32(
                        encoder,
                        &batch.ffn_up,
                        debug.ffn_up,
                        token_count * cfg.intermediate_dim,
                    );
                });
            }
            dispatch(command, |encoder| {
                self.kernels
                    .encode_silu_mul(encoder, &batch.ffn_gate, &batch.ffn_up);
            });
            if let Some(debug) = tail_capture
                .as_ref()
                .filter(|capture| capture.layer == layer_index)
                .and_then(|capture| capture.debug.as_ref())
            {
                dispatch(command, |encoder| {
                    self.kernels.encode_copy_f32(
                        encoder,
                        &batch.ffn_gate,
                        debug.ffn_swiglu,
                        token_count * cfg.intermediate_dim,
                    );
                });
            }
            self.project_batch_tokens(
                command,
                &layer.ffn_down,
                &batch.ffn_gate,
                &batch.projected,
                token_count,
                batch,
            );
            if let Some(debug) = tail_capture
                .as_ref()
                .filter(|capture| capture.layer == layer_index)
                .and_then(|capture| capture.debug.as_ref())
            {
                dispatch(command, |encoder| {
                    self.kernels.encode_copy_f32(
                        encoder,
                        &batch.projected,
                        debug.ffn_out,
                        token_count * cfg.hidden_dim,
                    );
                });
            }
            if self.fused_prefill_dual_norm {
                let (next_norm, next_output) = if layer_index + 1 < cfg.n_layers {
                    (&self.layers[layer_index + 1].attn_norm, &batch.post_norm)
                } else {
                    (&self.output_norm, &batch.hidden)
                };
                dispatch(command, |encoder| {
                    self.kernels
                        .encode_fused_norm_residual_rms_norm_batch_dual_eps(
                            encoder,
                            &batch.normed,
                            &batch.projected,
                            next_output,
                            &layer.post_ffn_norm,
                            next_norm,
                            cfg.hidden_dim,
                            cfg.post_norm_eps,
                            cfg.rms_eps,
                            token_count,
                        );
                });
            } else {
                dispatch(command, |encoder| {
                    self.kernels.encode_fused_rms_norm_residual_add_batch(
                        encoder,
                        &batch.normed,
                        &batch.projected,
                        &layer.post_ffn_norm,
                        cfg.hidden_dim,
                        cfg.post_norm_eps,
                        token_count,
                    );
                });
            }
            if let Some(capture_index) = capture_layers
                .iter()
                .position(|&candidate| candidate == layer_index)
            {
                dispatch(command, |encoder| {
                    let count = token_count * cfg.hidden_dim;
                    if let Some(destination) = layer_major_capture {
                        self.kernels.encode_copy_f32_region(
                            encoder,
                            &batch.normed,
                            0,
                            destination,
                            capture_index * count,
                            count,
                        );
                    } else {
                        self.kernels.encode_copy_f32(
                            encoder,
                            &batch.normed,
                            &capture_buffers[capture_index],
                            count,
                        );
                    }
                });
            }
        }

        if !encode_output {
            return Ok(());
        }
        if !self.fused_prefill_dual_norm {
            dispatch(command, |encoder| {
                self.kernels.encode_rms_norm_mul(
                    encoder,
                    &batch.normed,
                    &self.output_norm,
                    &batch.hidden,
                    cfg.hidden_dim,
                    cfg.rms_eps,
                    token_count,
                );
            });
        }
        if let Some(debug) = tail_capture
            .as_ref()
            .filter(|capture| capture.layer + 1 == cfg.n_layers)
            .and_then(|capture| capture.debug.as_ref())
        {
            dispatch(command, |encoder| {
                self.kernels.encode_copy_f32(
                    encoder,
                    &batch.hidden,
                    debug.result_norm,
                    token_count * cfg.hidden_dim,
                );
            });
        }
        if let Some(logits) = batch_logits {
            debug_assert!(logits.len() >= token_count * cfg.vocab_size);
            self.project_batch_tokens(
                command,
                &self.output,
                &batch.hidden,
                logits,
                token_count,
                batch,
            );
            if let Some(debug) = tail_capture
                .as_ref()
                .filter(|capture| capture.layer + 1 == cfg.n_layers)
                .and_then(|capture| capture.debug.as_ref())
            {
                dispatch(command, |encoder| {
                    self.kernels.encode_copy_f32(
                        encoder,
                        logits,
                        debug.result_projection,
                        token_count * cfg.vocab_size,
                    );
                });
            }
            dispatch(command, |encoder| {
                if token_count == 1 && start_position < cfg.sliding_window {
                    self.kernels.encode_scale_softcap_legacy(
                        encoder,
                        logits,
                        cfg.logit_scale,
                        cfg.final_logit_softcap,
                    );
                } else {
                    self.kernels.encode_scale_softcap_count(
                        encoder,
                        logits,
                        token_count * cfg.vocab_size,
                        cfg.logit_scale,
                        cfg.final_logit_softcap,
                    );
                }
            });
        } else {
            let output_input = if token_count == 1 {
                // A one-row batch already is the last row. Project it
                // directly instead of copying 6,656 unchanged f32 values
                // solely to satisfy token-arena LM-head addressing.
                &batch.hidden
            } else {
                dispatch(command, |encoder| {
                    self.kernels.encode_copy_row(
                        encoder,
                        &batch.hidden,
                        &self.activations.hidden,
                        token_count - 1,
                    );
                });
                &self.activations.hidden
            };
            self.project(
                command,
                &self.output,
                output_input,
                &self.activations.logits,
            );
            if let Some(debug) = tail_capture
                .as_ref()
                .filter(|capture| capture.layer + 1 == cfg.n_layers)
                .and_then(|capture| capture.debug.as_ref())
            {
                dispatch(command, |encoder| {
                    self.kernels.encode_copy_f32(
                        encoder,
                        &self.activations.logits,
                        debug.result_projection,
                        cfg.vocab_size,
                    );
                });
            }
            dispatch(command, |encoder| {
                if token_count == 1 && start_position < cfg.sliding_window {
                    // Preserve the checked-in short-context greedy contract.
                    // The byte-exact pinned-llama unary chain is the standard
                    // wrapped-SWA (2K+) decode contract; short-context native
                    // prefill remains a separately flagged reconciliation.
                    self.kernels.encode_scale_softcap_legacy(
                        encoder,
                        &self.activations.logits,
                        cfg.logit_scale,
                        cfg.final_logit_softcap,
                    );
                } else {
                    self.kernels.encode_scale_softcap(
                        encoder,
                        &self.activations.logits,
                        cfg.logit_scale,
                        cfg.final_logit_softcap,
                    );
                }
            });
        }
        Ok(())
    }

    fn finish_batch_hidden(
        &mut self,
        token_count: usize,
        capture_layers: &[usize],
        capture_buffers: &[GpuBuffer],
        batch_logits: Option<&GpuBuffer>,
    ) -> Result<(Vec<f32>, Vec<f32>), MetalModelError> {
        let cfg = &self.cfg;
        self.n_past += token_count;
        let mut captured = Vec::with_capacity(token_count * capture_layers.len() * cfg.hidden_dim);
        for token in 0..token_count {
            for buffer in capture_buffers {
                let start = token * cfg.hidden_dim;
                captured.extend_from_slice(&buffer.as_slice()[start..start + cfg.hidden_dim]);
            }
        }
        let logits = match batch_logits {
            Some(rows) => rows.as_slice()[..token_count * cfg.vocab_size].to_vec(),
            None => self.activations.logits.as_slice().to_vec(),
        };
        Ok((logits, captured))
    }

    /// Pack one ready decode row from each resident sequence into a single
    /// weight pass. Sequence-local KV planes and positions remain disjoint;
    /// the row index is the physical slot ID for every packed activation.
    pub(crate) fn forward_decode_group(
        models: &mut [&mut Self],
        tokens: &[u32],
    ) -> Result<Vec<Vec<f32>>, MetalModelError> {
        let rows = models.len();
        if rows == 0 || rows > 4 || tokens.len() != rows {
            return Err(MetalModelError::InvalidSnapshot(
                "packed decode requires matching 1..=4 model and token rows".into(),
            ));
        }
        let shared = Arc::clone(&models[0].shared);
        if models
            .iter()
            .any(|model| !Arc::ptr_eq(&shared, &model.shared))
        {
            return Err(MetalModelError::InvalidSnapshot(
                "packed decode sequences do not share one Metal executor".into(),
            ));
        }
        let sequence_ids = models
            .iter()
            .map(|model| model.sequence_id)
            .collect::<Vec<_>>();
        let _permit = shared.scheduler.acquire_decode_group(&sequence_ids)?;
        let positions = models.iter().map(|model| model.n_past).collect::<Vec<_>>();

        let mut workspaces = shared.decode_batch_workspaces.lock().map_err(|_| {
            MetalModelError::InvalidSnapshot("decode batch workspace is poisoned".into())
        })?;
        if let std::collections::btree_map::Entry::Vacant(entry) = workspaces.entry(rows) {
            entry.insert(DecodeBatchWorkspace::new(
                &shared.context,
                &models[0].cfg,
                rows,
            )?);
        }
        let workspace = workspaces
            .get_mut(&rows)
            .expect("inserted decode workspace");
        for (destination, token) in workspace
            .token_ids
            .as_mut_bytes()
            .chunks_exact_mut(std::mem::size_of::<u32>())
            .zip(tokens)
        {
            destination.copy_from_slice(&token.to_le_bytes());
        }
        let token_view = workspace
            .token_ids
            .view(0, rows * std::mem::size_of::<u32>())
            .expect("packed token bytes");
        let command_buffer = shared.context.queue.new_command_buffer();
        let command = GraphEncoder::concurrent(
            command_buffer
                .compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent),
        );
        Self::encode_decode_group(
            models,
            &shared,
            workspace,
            &token_view,
            &positions,
            &command,
        )?;
        command.encoder.end_encoding();
        command_buffer.commit();
        shared
            .context
            .wait_for_completion(command_buffer, Duration::from_secs(300))?;

        let mut logits = Vec::with_capacity(rows);
        for (row, model) in models.iter_mut().enumerate() {
            model.n_past += 1;
            let start = row * model.cfg.vocab_size;
            let values = workspace.logits.as_slice()[start..start + model.cfg.vocab_size].to_vec();
            model
                .activations
                .logits
                .as_mut_slice()
                .copy_from_slice(&values);
            logits.push(values);
        }
        Ok(logits)
    }

    fn encode_decode_group<T: EncodeTarget + ?Sized>(
        models: &mut [&mut Self],
        shared: &MetalShared,
        workspace: &DecodeBatchWorkspace,
        token_view: &GpuByteView<'_>,
        positions: &[usize],
        command: &T,
    ) -> Result<(), MetalModelError> {
        let cfg = models[0].cfg.clone();
        let rows = models.len();
        let batch = &workspace.activations;
        dispatch(command, |encoder| {
            shared.kernels.encode_embedding_q4k(
                encoder,
                shared.embedding.view(&shared.mapped_weights),
                token_view,
                &batch.hidden,
                cfg.hidden_dim,
                cfg.vocab_size,
                rows,
            );
        });
        dispatch(command, |encoder| {
            shared.kernels.encode_rms_norm_mul(
                encoder,
                &batch.hidden,
                &shared.entry_norm_ones,
                &batch.normed,
                cfg.hidden_dim,
                cfg.rms_eps,
                rows,
            );
        });

        for layer_index in 0..cfg.n_layers {
            let layer = shared.layers[layer_index].clone();
            if layer_index == 0 || !shared.fused_prefill_dual_norm {
                dispatch(command, |encoder| {
                    shared.kernels.encode_rms_norm_mul(
                        encoder,
                        &batch.normed,
                        &layer.attn_norm,
                        &batch.post_norm,
                        cfg.hidden_dim,
                        cfg.rms_eps,
                        rows,
                    );
                });
            }
            dispatch(command, |encoder| {
                models[0].encode_decode_group_projection(
                    encoder,
                    &layer.q,
                    &batch.post_norm,
                    &batch.q,
                    rows,
                );
                models[0].encode_decode_group_projection(
                    encoder,
                    &layer.k,
                    &batch.post_norm,
                    &batch.k,
                    rows,
                );
                models[0].encode_decode_group_projection(
                    encoder,
                    &layer.v,
                    &batch.post_norm,
                    &batch.v,
                    rows,
                );
                models[0].encode_decode_group_projection(
                    encoder,
                    &layer.gate,
                    &batch.post_norm,
                    &batch.gate,
                    rows,
                );
            });
            dispatch(command, |encoder| {
                shared.kernels.encode_qk_norm(
                    encoder,
                    &batch.q,
                    &layer.q_norm,
                    &batch.q,
                    cfg.head_dim,
                    cfg.rms_eps,
                    rows * cfg.n_heads,
                );
                shared.kernels.encode_qk_norm(
                    encoder,
                    &batch.k,
                    &layer.k_norm,
                    &batch.k,
                    cfg.head_dim,
                    cfg.rms_eps,
                    rows * cfg.n_kv_heads,
                );
            });
            dispatch(command, |encoder| {
                for (row, model) in models.iter().enumerate() {
                    shared.kernels.encode_copy_f32_region(
                        encoder,
                        &batch.q,
                        row * cfg.attn_dim(),
                        &model.activations.q,
                        0,
                        cfg.attn_dim(),
                    );
                    shared.kernels.encode_copy_f32_region(
                        encoder,
                        &batch.k,
                        row * cfg.kv_dim(),
                        &model.activations.k,
                        0,
                        cfg.kv_dim(),
                    );
                    shared.kernels.encode_copy_f32_region(
                        encoder,
                        &batch.v,
                        row * cfg.kv_dim(),
                        &model.activations.v,
                        0,
                        cfg.kv_dim(),
                    );
                    shared.kernels.encode_copy_f32_region(
                        encoder,
                        &batch.gate,
                        row * cfg.attn_dim(),
                        &model.activations.gate,
                        0,
                        cfg.attn_dim(),
                    );
                }
            });
            for (row, model) in models.iter_mut().enumerate() {
                model.encode_group_sequence_attention(command, layer_index, positions[row])?;
            }
            dispatch(command, |encoder| {
                for (row, model) in models.iter().enumerate() {
                    shared.kernels.encode_copy_f32_region(
                        encoder,
                        &model.activations.attention,
                        0,
                        &batch.attention,
                        row * cfg.attn_dim(),
                        cfg.attn_dim(),
                    );
                    shared.kernels.encode_copy_f32_region(
                        encoder,
                        &model.activations.gate,
                        0,
                        &batch.gate,
                        row * cfg.attn_dim(),
                        cfg.attn_dim(),
                    );
                }
            });
            dispatch(command, |encoder| {
                shared
                    .kernels
                    .encode_sigmoid_gate(encoder, &batch.attention, &batch.gate);
            });
            models[0].project_decode_group_tokens(
                command,
                &layer.output,
                &batch.attention,
                &batch.projected,
                rows,
            );
            if shared.fused_prefill_dual_norm {
                dispatch(command, |encoder| {
                    shared
                        .kernels
                        .encode_fused_norm_residual_rms_norm_32sg_batch(
                            encoder,
                            &batch.normed,
                            &batch.projected,
                            &batch.post_norm,
                            &layer.post_attn_norm,
                            &layer.ffn_norm,
                            cfg.hidden_dim,
                            cfg.post_norm_eps,
                            cfg.rms_eps,
                            rows,
                        );
                });
            } else {
                dispatch(command, |encoder| {
                    shared.kernels.encode_fused_rms_norm_residual_add_batch(
                        encoder,
                        &batch.normed,
                        &batch.projected,
                        &layer.post_attn_norm,
                        cfg.hidden_dim,
                        cfg.post_norm_eps,
                        rows,
                    );
                });
                dispatch(command, |encoder| {
                    shared.kernels.encode_rms_norm_mul(
                        encoder,
                        &batch.normed,
                        &layer.ffn_norm,
                        &batch.post_norm,
                        cfg.hidden_dim,
                        cfg.rms_eps,
                        rows,
                    );
                });
            }
            dispatch(command, |encoder| {
                models[0].encode_decode_group_projection(
                    encoder,
                    &layer.ffn_gate,
                    &batch.post_norm,
                    &batch.ffn_gate,
                    rows,
                );
                models[0].encode_decode_group_projection(
                    encoder,
                    &layer.ffn_up,
                    &batch.post_norm,
                    &batch.ffn_up,
                    rows,
                );
            });
            dispatch(command, |encoder| {
                shared
                    .kernels
                    .encode_silu_mul(encoder, &batch.ffn_gate, &batch.ffn_up);
            });
            models[0].project_decode_group_tokens(
                command,
                &layer.ffn_down,
                &batch.ffn_gate,
                &batch.projected,
                rows,
            );
            if shared.fused_prefill_dual_norm {
                let (next_norm, next_output) = if layer_index + 1 < cfg.n_layers {
                    (&shared.layers[layer_index + 1].attn_norm, &batch.post_norm)
                } else {
                    (&shared.output_norm, &batch.hidden)
                };
                dispatch(command, |encoder| {
                    shared
                        .kernels
                        .encode_fused_norm_residual_rms_norm_32sg_batch(
                            encoder,
                            &batch.normed,
                            &batch.projected,
                            next_output,
                            &layer.post_ffn_norm,
                            next_norm,
                            cfg.hidden_dim,
                            cfg.post_norm_eps,
                            cfg.rms_eps,
                            rows,
                        );
                });
            } else {
                dispatch(command, |encoder| {
                    shared.kernels.encode_fused_rms_norm_residual_add_batch(
                        encoder,
                        &batch.normed,
                        &batch.projected,
                        &layer.post_ffn_norm,
                        cfg.hidden_dim,
                        cfg.post_norm_eps,
                        rows,
                    );
                });
            }
        }
        if !shared.fused_prefill_dual_norm {
            dispatch(command, |encoder| {
                shared.kernels.encode_rms_norm_mul(
                    encoder,
                    &batch.normed,
                    &shared.output_norm,
                    &batch.hidden,
                    cfg.hidden_dim,
                    cfg.rms_eps,
                    rows,
                );
            });
        }
        models[0].project_decode_group_tokens(
            command,
            &shared.output,
            &batch.hidden,
            &workspace.logits,
            rows,
        );
        dispatch(command, |encoder| {
            shared.kernels.encode_scale_softcap_count(
                encoder,
                &workspace.logits,
                rows * cfg.vocab_size,
                cfg.logit_scale,
                cfg.final_logit_softcap,
            );
        });
        Ok(())
    }

    fn encode_group_sequence_attention<T: EncodeTarget + ?Sized>(
        &mut self,
        command: &T,
        layer_index: usize,
        position: usize,
    ) -> Result<(), MetalModelError> {
        let cfg = &self.cfg;
        if cfg.layer_kinds[layer_index].uses_rope() {
            dispatch(command, |encoder| {
                self.kernels.encode_rope_norm_batch_cached(
                    encoder,
                    &self.activations.q,
                    &self.activations.k,
                    &self.rope_frequencies,
                    cfg.n_heads,
                    cfg.n_kv_heads,
                    cfg.head_dim,
                    position,
                    1,
                    self.rope_positions.view(
                        position * std::mem::size_of::<u32>(),
                        std::mem::size_of::<u32>(),
                    ),
                    cfg.rope_base_swa,
                    cfg.context_length,
                );
            });
        }
        let write_physical = self.cache[layer_index].append(layer_index, position)?;
        let plane = &self.cache[layer_index];
        let strict_attention = std::env::var_os("MUSER_CROSS_VENDOR_QK").is_some();
        let llama_vec_rows = (strict_attention || self.kernels.has_llama_flash_attn_vec())
            && plane.len > 0
            && plane.capacity >= 32
            && (plane.origin_physical == 0 || plane.len == plane.capacity);
        let llama_swa = llama_vec_rows && plane.len.is_multiple_of(32);
        if cfg.layer_kinds[layer_index].is_swa() {
            if llama_swa {
                dispatch(command, |encoder| {
                    self.kernels.encode_kv_store_f16(
                        encoder,
                        &self.activations.k,
                        &self.activations.v,
                        &plane.key,
                        &plane.value,
                        write_physical,
                    );
                    let kv: [&metal::ResourceRef; 2] = [plane.key.metal(), plane.value.metal()];
                    encoder.memory_barrier_with_resources(&kv);
                    self.kernels.encode_llama_flash_attn_decode_vec_f16(
                        encoder,
                        &self.activations.q,
                        &plane.key,
                        &plane.value,
                        &self.activations.attention_mask,
                        &self.activations.attention_kv_pad,
                        &self.activations.attention_partials,
                        &self.activations.attention,
                        cfg.n_heads,
                        cfg.n_kv_heads,
                        cfg.head_dim,
                        plane.len,
                        plane.capacity,
                        0,
                        plane.origin_physical,
                        false,
                        cfg.attn_scale(),
                        false,
                        0,
                        0,
                    );
                });
            } else {
                dispatch(command, |encoder| {
                    self.kernels.encode_kv_store_f16(
                        encoder,
                        &self.activations.k,
                        &self.activations.v,
                        &plane.key,
                        &plane.value,
                        write_physical,
                    );
                });
                dispatch(command, |encoder| {
                    self.kernels.encode_attention_decode_splitk_f16(
                        encoder,
                        &self.activations.q,
                        &plane.key,
                        &plane.value,
                        &self.activations.attention_partials,
                        &self.activations.attention,
                        cfg.n_heads,
                        cfg.n_kv_heads,
                        cfg.head_dim,
                        position,
                        plane.capacity,
                        plane.origin_logical,
                        plane.origin_physical,
                        cfg.sliding_window,
                        cfg.attn_scale(),
                    );
                });
            }
        } else if llama_vec_rows {
            dispatch(command, |encoder| {
                self.kernels.encode_kv_store_batch_f16(
                    encoder,
                    &self.activations.k,
                    &self.activations.v,
                    &plane.key,
                    &plane.value,
                    cfg.kv_dim(),
                    1,
                    0,
                    1,
                    position,
                    plane.capacity,
                    plane.origin_logical,
                    plane.origin_physical,
                    cfg.head_dim,
                    true,
                );
                let kv: [&metal::ResourceRef; 2] = [plane.key.metal(), plane.value.metal()];
                encoder.memory_barrier_with_resources(&kv);
                self.kernels.encode_llama_flash_attn_decode_vec_f16(
                    encoder,
                    &self.activations.q,
                    &plane.key,
                    &plane.value,
                    &self.activations.attention_mask,
                    &self.activations.attention_kv_pad,
                    &self.activations.attention_partials,
                    &self.activations.attention,
                    cfg.n_heads,
                    cfg.n_kv_heads,
                    cfg.head_dim,
                    plane.len,
                    plane.capacity,
                    0,
                    plane.origin_physical,
                    false,
                    cfg.attn_scale(),
                    true,
                    0,
                    0,
                );
            });
        } else {
            dispatch(command, |encoder| {
                self.kernels
                    .encode_ferrite_attention_decode_interleaved_f16(
                        encoder,
                        &self.activations.q,
                        &plane.key,
                        &plane.value,
                        &self.activations.attention_partials,
                        &self.activations.attention,
                        &self.activations.k,
                        &self.activations.v,
                        cfg.n_heads,
                        cfg.n_kv_heads,
                        cfg.head_dim,
                        position,
                        plane.capacity,
                        cfg.attn_scale(),
                    );
            });
        }
        Ok(())
    }

    fn forward_token(&mut self, token: u32) -> Result<(), MetalModelError> {
        let mut token_staging = self.activations.token_ids.clone();
        token_staging.as_mut_bytes()[..std::mem::size_of::<u32>()]
            .copy_from_slice(&token.to_le_bytes());
        let token_view = token_staging
            .view(0, std::mem::size_of::<u32>())
            .expect("four token bytes");
        let queue = self.context.queue.clone();
        if std::env::var_os("MUSER_METAL_PHASE_PROFILE").is_some() {
            let labels = self.token_phase_labels(self.n_past);
            let profiler = PhaseProfiler::new(queue.clone());
            self.encode_token(&profiler, &token_view, self.n_past)?;
            profiler.print_report(&labels);
            self.n_past += 1;
            return Ok(());
        }
        let command_buffer = queue.new_command_buffer();
        // One concurrent encoder owns the complete token. Graph dependencies
        // are explicit barriers; independent projection groups share a barrier
        // interval and may overlap, matching the accepted Ferrite/llama route.
        let serial = GraphEncoder::concurrent(
            command_buffer
                .compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent),
        );
        self.encode_token(&serial, &token_view, self.n_past)?;
        serial.encoder.end_encoding();
        command_buffer.commit();
        self.context
            .wait_for_completion(command_buffer, Duration::from_secs(300))?;
        self.n_past += 1;
        Ok(())
    }

    /// Exact closure schedule for the diagnostic token route. The cache route
    /// is selected after reserving the current row, so derive the post-append
    /// ring metadata instead of relying on a static per-layer label list.
    fn token_phase_labels(&self, position: usize) -> Vec<String> {
        let mut labels = vec!["embedding".to_owned(), "entry_norm".to_owned()];
        let strict_attention = std::env::var_os("MUSER_CROSS_VENDOR_QK").is_some();
        for (layer, kind) in self.cfg.layer_kinds.iter().copied().enumerate() {
            let prefix = format!("layer.{layer}");
            if layer == 0 {
                labels.push(format!("{prefix}.attn_norm"));
            }
            labels.push(format!("{prefix}.qkvg"));
            labels.push(format!("{prefix}.qk_norm"));
            if kind.uses_rope() {
                labels.push(format!("{prefix}.rope"));
            }

            let plane = &self.cache[layer];
            debug_assert_eq!(position, plane.origin_logical + plane.len);
            let (next_len, next_origin_physical) = if plane.len < plane.capacity {
                (plane.len + 1, plane.origin_physical)
            } else {
                (plane.capacity, (plane.origin_physical + 1) % plane.capacity)
            };
            let llama_vec_rows = (strict_attention || self.kernels.has_llama_flash_attn_vec())
                && next_len > 0
                && plane.capacity >= 32
                && (next_origin_physical == 0 || next_len == plane.capacity);
            if kind.is_swa() && !(llama_vec_rows && next_len.is_multiple_of(32)) {
                labels.push(format!("{prefix}.kv_store"));
                labels.push(format!("{prefix}.attention"));
            } else {
                labels.push(format!("{prefix}.kv_store_attention"));
            }
            for operation in [
                "sigmoid_gate",
                "o_proj",
                "post_attn_ffn_norm",
                "ffn_gate_up",
                "swiglu",
                "ffn_down",
                "post_ffn_next_norm",
            ] {
                labels.push(format!("{prefix}.{operation}"));
            }
        }
        labels.extend(["lm_head".to_owned(), "softcap".to_owned()]);
        labels
    }

    fn encode_token<T: EncodeTarget + ?Sized>(
        &mut self,
        command: &T,
        token_view: &GpuByteView<'_>,
        position: usize,
    ) -> Result<(), MetalModelError> {
        let cfg = &self.cfg;

        dispatch(command, |encoder| {
            self.kernels.encode_embedding_q4k(
                encoder,
                self.embedding.view(&self.mapped_weights),
                token_view,
                &self.activations.hidden,
                cfg.hidden_dim,
                cfg.vocab_size,
                1,
            );
        });
        dispatch(command, |encoder| {
            self.kernels.encode_rms_norm_mul(
                encoder,
                &self.activations.hidden,
                &self.entry_norm_ones,
                &self.activations.normed,
                cfg.hidden_dim,
                cfg.rms_eps,
                1,
            );
        });

        // `normed` is the current hidden buffer at layer entry. Each layer
        // writes the attention residual to `hidden`, then the FFN result back
        // to `normed`, so no full-width copy dispatch is needed.
        for layer_index in 0..cfg.n_layers {
            let layer = self.layers[layer_index].clone();
            // Every layer after layer zero receives its attention pre-norm
            // from the preceding layer's fused post-FFN tail.
            if layer_index == 0 {
                dispatch(command, |encoder| {
                    self.kernels.encode_rms_norm_mul(
                        encoder,
                        &self.activations.normed,
                        &layer.attn_norm,
                        &self.activations.post_norm,
                        cfg.hidden_dim,
                        cfg.rms_eps,
                        1,
                    );
                });
            }
            // llama.cpp and Ferrite issue the four independent attention
            // projections as one concurrent set. They share a read-only input
            // and mapped weight arena but write disjoint activations.
            dispatch(command, |encoder| {
                self.encode_projection(
                    encoder,
                    &layer.q,
                    &self.activations.post_norm,
                    &self.activations.q,
                    1,
                );
                self.encode_projection(
                    encoder,
                    &layer.k,
                    &self.activations.post_norm,
                    &self.activations.k,
                    1,
                );
                self.encode_projection(
                    encoder,
                    &layer.v,
                    &self.activations.post_norm,
                    &self.activations.v,
                    1,
                );
                self.encode_projection(
                    encoder,
                    &layer.gate,
                    &self.activations.post_norm,
                    &self.activations.gate,
                    1,
                );
            });
            dispatch(command, |encoder| {
                self.kernels.encode_qk_norm(
                    encoder,
                    &self.activations.q,
                    &layer.q_norm,
                    &self.activations.q,
                    cfg.head_dim,
                    cfg.rms_eps,
                    cfg.n_heads,
                );
                self.kernels.encode_qk_norm(
                    encoder,
                    &self.activations.k,
                    &layer.k_norm,
                    &self.activations.k,
                    cfg.head_dim,
                    cfg.rms_eps,
                    cfg.n_kv_heads,
                );
            });
            // Q and K normalization are independent and intentionally share
            // the concurrent set above.
            if cfg.layer_kinds[layer_index].uses_rope() {
                dispatch(command, |encoder| {
                    self.kernels.encode_rope_norm_batch_cached(
                        encoder,
                        &self.activations.q,
                        &self.activations.k,
                        &self.rope_frequencies,
                        cfg.n_heads,
                        cfg.n_kv_heads,
                        cfg.head_dim,
                        position,
                        1,
                        self.rope_positions.view(
                            position * std::mem::size_of::<u32>(),
                            std::mem::size_of::<u32>(),
                        ),
                        cfg.rope_base_swa,
                        cfg.context_length,
                    );
                });
            }

            let write_physical = self.cache[layer_index].append(layer_index, position)?;
            let plane = &self.cache[layer_index];
            let strict_attention = std::env::var_os("MUSER_CROSS_VENDOR_QK").is_some();
            let llama_vec_rows = (strict_attention || self.kernels.has_llama_flash_attn_vec())
                && plane.len > 0
                // The pinned vec kernel rounds KV reads to a 32-row block.
                // A deliberately tiny raw session can have a smaller backing
                // allocation, so taking the vec path would read past it and
                // poison the full distribution with NaNs.
                && plane.capacity >= 32
                && (plane.origin_physical == 0 || plane.len == plane.capacity);
            // Token-major SWA cannot use llama's pad kernel (nb11 is a full
            // token row, not one head). Only take that path when the window
            // is a multiple of 32 so the vec kernel never pads.
            let llama_swa = llama_vec_rows && plane.len.is_multiple_of(32);
            if cfg.layer_kinds[layer_index].is_swa() {
                if llama_swa {
                    dispatch(command, |encoder| {
                        self.kernels.encode_kv_store_f16(
                            encoder,
                            &self.activations.k,
                            &self.activations.v,
                            &plane.key,
                            &plane.value,
                            write_physical,
                        );
                        let kv: [&metal::ResourceRef; 2] = [plane.key.metal(), plane.value.metal()];
                        encoder.memory_barrier_with_resources(&kv);
                        self.kernels.encode_llama_flash_attn_decode_vec_f16(
                            encoder,
                            &self.activations.q,
                            &plane.key,
                            &plane.value,
                            &self.activations.attention_mask,
                            &self.activations.attention_kv_pad,
                            &self.activations.attention_partials,
                            &self.activations.attention,
                            cfg.n_heads,
                            cfg.n_kv_heads,
                            cfg.head_dim,
                            plane.len,
                            plane.capacity,
                            0,
                            plane.origin_physical,
                            false,
                            cfg.attn_scale(),
                            false,
                            0,
                            0,
                        );
                    });
                } else {
                    dispatch(command, |encoder| {
                        self.kernels.encode_kv_store_f16(
                            encoder,
                            &self.activations.k,
                            &self.activations.v,
                            &plane.key,
                            &plane.value,
                            write_physical,
                        );
                    });
                    dispatch(command, |encoder| {
                        self.kernels.encode_attention_decode_splitk_f16(
                            encoder,
                            &self.activations.q,
                            &plane.key,
                            &plane.value,
                            &self.activations.attention_partials,
                            &self.activations.attention,
                            cfg.n_heads,
                            cfg.n_kv_heads,
                            cfg.head_dim,
                            position,
                            plane.capacity,
                            plane.origin_logical,
                            plane.origin_physical,
                            cfg.sliding_window,
                            cfg.attn_scale(),
                        );
                    });
                }
            } else {
                debug_assert!(plane.head_major);
                if llama_vec_rows {
                    dispatch(command, |encoder| {
                        self.kernels.encode_kv_store_batch_f16(
                            encoder,
                            &self.activations.k,
                            &self.activations.v,
                            &plane.key,
                            &plane.value,
                            cfg.kv_dim(),
                            1,
                            0,
                            1,
                            position,
                            plane.capacity,
                            plane.origin_logical,
                            plane.origin_physical,
                            cfg.head_dim,
                            true,
                        );
                        let kv: [&metal::ResourceRef; 2] = [plane.key.metal(), plane.value.metal()];
                        encoder.memory_barrier_with_resources(&kv);
                        self.kernels.encode_llama_flash_attn_decode_vec_f16(
                            encoder,
                            &self.activations.q,
                            &plane.key,
                            &plane.value,
                            &self.activations.attention_mask,
                            &self.activations.attention_kv_pad,
                            &self.activations.attention_partials,
                            &self.activations.attention,
                            cfg.n_heads,
                            cfg.n_kv_heads,
                            cfg.head_dim,
                            plane.len,
                            plane.capacity,
                            0,
                            plane.origin_physical,
                            false,
                            cfg.attn_scale(),
                            true,
                            0,
                            0,
                        );
                    });
                } else {
                    dispatch(command, |encoder| {
                        self.kernels
                            .encode_ferrite_attention_decode_interleaved_f16(
                                encoder,
                                &self.activations.q,
                                &plane.key,
                                &plane.value,
                                &self.activations.attention_partials,
                                &self.activations.attention,
                                &self.activations.k,
                                &self.activations.v,
                                cfg.n_heads,
                                cfg.n_kv_heads,
                                cfg.head_dim,
                                position,
                                plane.capacity,
                                cfg.attn_scale(),
                            );
                    });
                }
            }
            dispatch(command, |encoder| {
                self.kernels.encode_sigmoid_gate(
                    encoder,
                    &self.activations.attention,
                    &self.activations.gate,
                );
            });
            self.project(
                command,
                &layer.output,
                &self.activations.attention,
                &self.activations.projected,
            );
            dispatch(command, |encoder| {
                self.kernels.encode_fused_norm_residual_rms_norm_32sg(
                    encoder,
                    &self.activations.normed,
                    &self.activations.projected,
                    &self.activations.post_norm,
                    &layer.post_attn_norm,
                    &layer.ffn_norm,
                    cfg.hidden_dim,
                    cfg.post_norm_eps,
                    cfg.rms_eps,
                );
            });
            if self.ferrite_ffn_gate_up
                && layer.ffn_gate.layout.dtype == GgmlType::Q4_K
                && layer.ffn_up.layout.dtype == GgmlType::Q4_K
            {
                // Port Ferrite 897a6256b wholesale: the four-row/two-SIMD-
                // group kernel reads the normalized input once for both Q4_K
                // projections and writes the final SiLU(gate) * up row.
                dispatch(command, |encoder| {
                    self.kernels.encode_ffn_q4k_gate_up_silu_4r2s(
                        encoder,
                        layer.ffn_gate.view(&self.mapped_weights),
                        layer.ffn_up.view(&self.mapped_weights),
                        &self.activations.post_norm,
                        &self.activations.ffn_gate,
                        cfg.intermediate_dim,
                        cfg.hidden_dim,
                    );
                });
            } else {
                // Exact upstream-matvec control and non-Q4_K fallback.
                dispatch(command, |encoder| {
                    self.encode_projection(
                        encoder,
                        &layer.ffn_gate,
                        &self.activations.post_norm,
                        &self.activations.ffn_gate,
                        1,
                    );
                    self.encode_projection(
                        encoder,
                        &layer.ffn_up,
                        &self.activations.post_norm,
                        &self.activations.ffn_up,
                        1,
                    );
                });
                dispatch(command, |encoder| {
                    self.kernels.encode_silu_mul(
                        encoder,
                        &self.activations.ffn_gate,
                        &self.activations.ffn_up,
                    );
                });
            }
            self.project(
                command,
                &layer.ffn_down,
                &self.activations.ffn_gate,
                &self.activations.projected,
            );
            let (next_norm, next_output) = if layer_index + 1 < cfg.n_layers {
                (
                    &self.layers[layer_index + 1].attn_norm,
                    &self.activations.post_norm,
                )
            } else {
                (&self.output_norm, &self.activations.hidden)
            };
            dispatch(command, |encoder| {
                self.kernels.encode_fused_norm_residual_rms_norm_32sg(
                    encoder,
                    &self.activations.normed,
                    &self.activations.projected,
                    next_output,
                    &layer.post_ffn_norm,
                    next_norm,
                    cfg.hidden_dim,
                    cfg.post_norm_eps,
                    cfg.rms_eps,
                );
            });
        }

        self.project(
            command,
            &self.output,
            &self.activations.hidden,
            &self.activations.logits,
        );
        dispatch(command, |encoder| {
            self.kernels.encode_scale_softcap(
                encoder,
                &self.activations.logits,
                cfg.logit_scale,
                cfg.final_logit_softcap,
            );
        });
        Ok(())
    }

    fn project<T: EncodeTarget + ?Sized>(
        &self,
        command: &T,
        projection: &Projection,
        input: &GpuBuffer,
        output: &GpuBuffer,
    ) {
        self.project_tokens(command, projection, input, output, 1);
    }

    fn project_tokens<T: EncodeTarget + ?Sized>(
        &self,
        command: &T,
        projection: &Projection,
        input: &GpuBuffer,
        output: &GpuBuffer,
        token_count: usize,
    ) {
        dispatch(command, |encoder| {
            self.encode_projection(encoder, projection, input, output, token_count);
        });
    }

    fn project_batch_tokens<T: EncodeTarget + ?Sized>(
        &self,
        command: &T,
        projection: &Projection,
        input: &GpuBuffer,
        output: &GpuBuffer,
        token_count: usize,
        batch: &BatchActivations,
    ) {
        dispatch(command, |encoder| {
            self.encode_batch_projection(encoder, projection, input, output, token_count, batch);
        });
    }

    fn encode_batch_projection(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        projection: &Projection,
        input: &GpuBuffer,
        output: &GpuBuffer,
        token_count: usize,
        batch: &BatchActivations,
    ) {
        if token_count == 16
            && projection.layout.dtype == GgmlType::NVFP4_E2M1
            && projection.layout.n_in.is_multiple_of(64)
            && std::env::var_os("MUSER_NO_M16_N32").is_none()
        {
            if let Some(input_scale_inv) = projection.layout.nvfp4_input_scale_inv {
                self.kernels.encode_nvfp4_w4a4_prequant_m16(
                    encoder,
                    projection.view(&self.mapped_weights),
                    projection
                        .nvfp4_scale_view(&self.mapped_weights)
                        .expect("validated NVFP4 projection has scales"),
                    projection.layout.nvfp4_scale2,
                    input_scale_inv,
                    input,
                    &batch.nvfp4_quantized,
                    &batch.nvfp4_activation_scales,
                    output,
                    projection.layout.n_in,
                    projection.layout.n_out,
                );
                return;
            }
        }
        self.encode_projection(encoder, projection, input, output, token_count);
    }

    fn project_decode_group_tokens<T: EncodeTarget + ?Sized>(
        &self,
        command: &T,
        projection: &Projection,
        input: &GpuBuffer,
        output: &GpuBuffer,
        rows: usize,
    ) {
        dispatch(command, |encoder| {
            self.encode_decode_group_projection(encoder, projection, input, output, rows);
        });
    }

    fn encode_decode_group_projection(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        projection: &Projection,
        input: &GpuBuffer,
        output: &GpuBuffer,
        rows: usize,
    ) {
        if projection.layout.dtype == GgmlType::F16 {
            self.kernels.encode_f16_matmul(
                encoder,
                projection.view(&self.mapped_weights),
                input,
                output,
                projection.layout.n_in,
                projection.layout.n_out,
                rows,
            );
            return;
        }
        if projection.layout.dtype == GgmlType::NVFP4_E2M1 {
            self.kernels.encode_nvfp4_matmul(
                encoder,
                projection.view(&self.mapped_weights),
                projection
                    .nvfp4_scale_view(&self.mapped_weights)
                    .expect("validated NVFP4 projection has scales"),
                projection.layout.nvfp4_scale2,
                projection.layout.nvfp4_input_scale_inv,
                input,
                output,
                projection.layout.n_in,
                projection.layout.n_out,
                rows,
            );
            return;
        }
        self.kernels.encode_quantized_decode_group(
            encoder,
            projection.view(&self.mapped_weights),
            input,
            output,
            projection.layout.dtype,
            projection.layout.n_in,
            projection.layout.n_out,
            rows,
        );
    }

    fn encode_projection(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        projection: &Projection,
        input: &GpuBuffer,
        output: &GpuBuffer,
        token_count: usize,
    ) {
        if projection.layout.dtype == GgmlType::F16 {
            self.kernels.encode_f16_matmul(
                encoder,
                projection.view(&self.mapped_weights),
                input,
                output,
                projection.layout.n_in,
                projection.layout.n_out,
                token_count,
            );
            return;
        }
        if projection.layout.dtype == GgmlType::NVFP4_E2M1 {
            self.kernels.encode_nvfp4_matmul(
                encoder,
                projection.view(&self.mapped_weights),
                projection
                    .nvfp4_scale_view(&self.mapped_weights)
                    .expect("validated NVFP4 projection has scales"),
                projection.layout.nvfp4_scale2,
                projection.layout.nvfp4_input_scale_inv,
                input,
                output,
                projection.layout.n_in,
                projection.layout.n_out,
                token_count,
            );
            return;
        }
        self.kernels.encode_quantized_matmul(
            encoder,
            projection.view(&self.mapped_weights),
            input,
            output,
            projection.layout.dtype,
            projection.layout.n_in,
            projection.layout.n_out,
            token_count,
        );
    }
}

fn gpu_vector(
    context: &MetalContext,
    weights: &MuseWeights,
    name: &str,
) -> Result<GpuBuffer, MetalModelError> {
    Ok(GpuBuffer::from_f32(context, &weights.f32_vec(name)?)?)
}

fn new_prefill_command_buffer(queue: &metal::CommandQueueRef) -> &metal::CommandBufferRef {
    // The mapped weights, model arenas, KV planes, and checked-out workspace
    // all outlive the synchronous final wait. Retaining every resource again
    // at each of the hundreds of dispatches adds Objective-C bookkeeping but
    // cannot extend any lifetime in this graph. This matches pinned
    // llama.cpp's commandBufferWithUnretainedReferences contract.
    queue.new_command_buffer_with_unretained_references()
}

fn new_prefill_graph_encoder(
    command: &metal::CommandBufferRef,
    concurrent: bool,
) -> GraphEncoder<'_> {
    if concurrent {
        GraphEncoder::concurrent(
            command.compute_command_encoder_with_dispatch_type(metal::MTLDispatchType::Concurrent),
        )
    } else {
        GraphEncoder::serial(command.new_compute_command_encoder())
    }
}

trait EncodeTarget {
    fn before_dispatch(&self) {}
    fn encode(&self, encode: impl FnOnce(&metal::ComputeCommandEncoderRef));
}

impl EncodeTarget for metal::CommandBufferRef {
    fn encode(&self, encode: impl FnOnce(&metal::ComputeCommandEncoderRef)) {
        let encoder = self.new_compute_command_encoder();
        encode(encoder);
        encoder.end_encoding();
    }
}

struct GraphEncoder<'a> {
    encoder: &'a metal::ComputeCommandEncoderRef,
    concurrent: bool,
    has_dispatch: std::cell::Cell<bool>,
}

/// One-shot, environment-gated phase profiler. Each graph group is submitted
/// as its own command buffer so Metal exposes an exact GPU interval. This is
/// diagnostic-only and never participates in normal serving or benchmarks.
struct PhaseProfiler {
    queue: metal::CommandQueue,
    gpu_ms: std::cell::RefCell<Vec<f64>>,
}

impl PhaseProfiler {
    fn new(queue: metal::CommandQueue) -> Self {
        Self {
            queue,
            gpu_ms: std::cell::RefCell::new(Vec::new()),
        }
    }

    fn print_report(&self, labels: &[String]) {
        let samples = self.gpu_ms.borrow();
        assert_eq!(
            labels.len(),
            samples.len(),
            "token phase labels must cover every dispatch group"
        );
        eprintln!(
            "[muser-metal-profile] dispatches={} labels={} gpu_ms_total={:.3}",
            samples.len(),
            labels.len(),
            samples.iter().sum::<f64>()
        );
        let mut buckets = BTreeMap::<&str, f64>::new();
        for (label, duration) in labels.iter().zip(samples.iter()) {
            *buckets.entry(label).or_default() += duration;
        }
        for (label, duration) in buckets {
            eprintln!("[muser-metal-profile] {label}={duration:.3}ms");
        }
    }

    fn print_batch_report(&self, labels: &[String]) {
        let samples = self.gpu_ms.borrow();
        // Dispatch groups drift as the batch DAG evolves (J1 added pad/reduce
        // boundaries); keep the profiler diagnostic rather than asserting an
        // exact label cover. Extra groups land in an `unlabeled` bucket.
        if labels.len() != samples.len() {
            eprintln!(
                "[muser-metal-batch-profile] label-drift labels={} dispatches={}",
                labels.len(),
                samples.len(),
            );
        }
        eprintln!(
            "[muser-metal-batch-profile] dispatches={} labels={} gpu_ms_total={:.3}",
            samples.len(),
            labels.len(),
            samples.iter().sum::<f64>()
        );
        let padded: Vec<&str> = labels
            .iter()
            .map(String::as_str)
            .chain(std::iter::repeat("unlabeled"))
            .take(samples.len())
            .collect();
        let mut operations = BTreeMap::<&str, f64>::new();
        for (label, duration) in padded.iter().zip(samples.iter()) {
            let operation = label.rsplit('.').next().unwrap_or(label);
            *operations.entry(operation).or_default() += duration;
        }
        for (operation, duration) in operations {
            eprintln!("[muser-metal-batch-profile] aggregate.{operation}={duration:.3}ms");
        }
        let mut top = padded
            .iter()
            .zip(samples.iter().copied())
            .collect::<Vec<_>>();
        top.sort_by(|left, right| right.1.total_cmp(&left.1));
        for (label, duration) in top.into_iter().take(32) {
            eprintln!("[muser-metal-batch-profile] top.{label}={duration:.3}ms");
        }
    }
}

impl EncodeTarget for PhaseProfiler {
    fn encode(&self, encode: impl FnOnce(&metal::ComputeCommandEncoderRef)) {
        let command = self.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encode(encoder);
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        let start: f64 = unsafe { objc::msg_send![command, GPUStartTime] };
        let end: f64 = unsafe { objc::msg_send![command, GPUEndTime] };
        self.gpu_ms.borrow_mut().push((end - start) * 1.0e3);
    }
}

impl<'a> GraphEncoder<'a> {
    fn serial(encoder: &'a metal::ComputeCommandEncoderRef) -> Self {
        Self {
            encoder,
            concurrent: false,
            has_dispatch: std::cell::Cell::new(false),
        }
    }

    fn concurrent(encoder: &'a metal::ComputeCommandEncoderRef) -> Self {
        Self {
            encoder,
            concurrent: true,
            has_dispatch: std::cell::Cell::new(false),
        }
    }
}

impl EncodeTarget for GraphEncoder<'_> {
    fn before_dispatch(&self) {
        if self.concurrent && self.has_dispatch.replace(true) {
            // Broad buffer scope exactly matches llama.cpp's dependency reset.
            // Independent kernels are deliberately grouped into one dispatch
            // closure, so every closure boundary is a real graph dependency.
            unsafe {
                let _: () = objc::msg_send![self.encoder, memoryBarrierWithScope: 1u64];
            }
        }
    }

    fn encode(&self, encode: impl FnOnce(&metal::ComputeCommandEncoderRef)) {
        encode(self.encoder);
    }
}

fn dispatch<T: EncodeTarget + ?Sized>(
    target: &T,
    encode: impl FnOnce(&metal::ComputeCommandEncoderRef),
) {
    target.before_dispatch();
    target.encode(encode);
}

#[cfg(test)]
mod tests {
    use super::{
        llama_fa_prefill_route_available, llama_vec_prefill_route_available,
        retained_prompt_subranges, AcceleratorScheduler, AcceleratorWork, GraphEncoder,
        MetalKvPlane, MetalMuseModel,
    };
    use crate::config::MuseLayerKind;
    use crate::metal::buffer::GpuBuffer;
    use crate::metal::context::MetalContext;
    use crate::reference::MuseModel;
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn llama_fa_prefill_never_overrides_the_cross_vendor_route() {
        assert!(llama_fa_prefill_route_available(512, true, false, false));
        assert!(!llama_fa_prefill_route_available(512, true, false, true));
        assert!(!llama_fa_prefill_route_available(512, true, true, false));
        assert!(!llama_fa_prefill_route_available(512, false, false, false));
        assert!(!llama_fa_prefill_route_available(19, true, false, false));
        assert!(llama_fa_prefill_route_available(20, true, false, false));
        assert!(llama_vec_prefill_route_available(19, 32, true, false));
        assert!(!llama_vec_prefill_route_available(19, 32, true, true));
        assert!(!llama_vec_prefill_route_available(20, 32, true, false));
    }

    #[test]
    fn prompt_pipeline_keeps_only_sink_and_trailing_window() {
        assert_eq!(
            retained_prompt_subranges(0, 512, 131_007, 64, 65_536),
            vec![(0, 0, 64)]
        );
        assert!(retained_prompt_subranges(512, 1_024, 131_007, 64, 65_536).is_empty());
        assert_eq!(
            retained_prompt_subranges(65_024, 65_536, 131_007, 64, 65_536),
            vec![(447, 65_471, 65)]
        );
        assert_eq!(
            retained_prompt_subranges(130_560, 131_007, 131_007, 64, 65_536),
            vec![(0, 130_560, 447)]
        );
    }

    #[test]
    fn short_prompt_pipeline_retains_every_prefix_row() {
        assert_eq!(
            retained_prompt_subranges(0, 511, 511, 64, 65_536),
            vec![(0, 0, 511)]
        );
    }

    fn install_test_kv_fixture(metal: &mut MetalMuseModel, tokens: &[u32], kv_dir: &str) {
        let mut snapshot = metal
            .export_cache_snapshot(tokens)
            .expect("export diagnostic prefix");
        let planes = snapshot
            .layers
            .iter()
            .map(|plane| {
                let layer = plane.layer as usize;
                let key = std::fs::read(
                    std::path::Path::new(kv_dir).join(format!("layer-{layer:02}.key.f16")),
                )
                .expect("read canonical llama key plane");
                let value = std::fs::read(
                    std::path::Path::new(kv_dir).join(format!("layer-{layer:02}.value.f16")),
                )
                .expect("read canonical llama value plane");
                crate::cache::CachePlaneSnapshot {
                    layer: plane.layer,
                    logical_start: plane.logical_start,
                    logical_count: plane.logical_count,
                    encoding: plane.encoding,
                    key: key.into(),
                    value: value.into(),
                }
            })
            .collect::<Vec<_>>();
        snapshot.layers = planes.into();
        metal
            .install_cache_snapshot(&snapshot)
            .expect("install canonical llama prefix");
    }

    #[test]
    fn accelerator_scheduler_prioritizes_decode_and_rotates_sequences() {
        let scheduler = AcceleratorScheduler::new();

        // Establish sequence 1 as the most recently served decoder.
        drop(
            scheduler
                .acquire(1, AcceleratorWork::Decode)
                .expect("initial decode permit"),
        );

        // Hold the owner while all three resident decoders queue. Cyclic
        // selection after sequence 1 must be 2, then wrap to 0, then 1.
        let owner = scheduler
            .acquire(9, AcceleratorWork::Prefill)
            .expect("prefill owner");
        let (sender, receiver) = std::sync::mpsc::channel();
        let mut workers = Vec::new();
        for sequence in [0usize, 1, 2] {
            let scheduler = Arc::clone(&scheduler);
            let sender = sender.clone();
            workers.push(std::thread::spawn(move || {
                let permit = scheduler
                    .acquire(sequence, AcceleratorWork::Decode)
                    .expect("decode permit");
                sender.send(sequence).expect("record decode order");
                drop(permit);
            }));
        }
        for _ in 0..100_000 {
            if scheduler
                .state
                .lock()
                .expect("scheduler state")
                .decode_waiting
                .len()
                == 3
            {
                break;
            }
            std::thread::yield_now();
        }
        assert_eq!(
            scheduler
                .state
                .lock()
                .expect("scheduler state")
                .decode_waiting
                .len(),
            3
        );
        drop(owner);
        assert_eq!(receiver.recv().unwrap(), 2);
        assert_eq!(receiver.recv().unwrap(), 0);
        assert_eq!(receiver.recv().unwrap(), 1);
        for worker in workers {
            worker.join().expect("decode worker");
        }

        // A queued prefill cannot pass a decoder that became ready while the
        // current accelerator interval was active.
        let owner = scheduler
            .acquire(9, AcceleratorWork::Prefill)
            .expect("second prefill owner");
        let (sender, receiver) = std::sync::mpsc::channel();
        let prefill_scheduler = Arc::clone(&scheduler);
        let prefill_sender = sender.clone();
        let prefill = std::thread::spawn(move || {
            let permit = prefill_scheduler
                .acquire(8, AcceleratorWork::Prefill)
                .expect("queued prefill");
            prefill_sender.send("prefill").unwrap();
            drop(permit);
        });
        let decode_scheduler = Arc::clone(&scheduler);
        let decode = std::thread::spawn(move || {
            let permit = decode_scheduler
                .acquire(3, AcceleratorWork::Decode)
                .expect("priority decode");
            sender.send("decode").unwrap();
            drop(permit);
        });
        for _ in 0..100_000 {
            if scheduler.has_waiting_decode() {
                break;
            }
            std::thread::yield_now();
        }
        assert!(scheduler.has_waiting_decode());
        drop(owner);
        assert_eq!(receiver.recv().unwrap(), "decode");
        assert_eq!(receiver.recv().unwrap(), "prefill");
        decode.join().unwrap();
        prefill.join().unwrap();
    }

    #[test]
    fn ring_metadata_advances_without_absolute_modulo_placement() {
        let context = MetalContext::new().expect("Metal context");
        let mut plane = MetalKvPlane::new(&context, 3, 1, false).expect("plane");
        assert_eq!(plane.append(0, 0).unwrap(), 0);
        assert_eq!(plane.append(0, 1).unwrap(), 1);
        assert_eq!(plane.append(0, 2).unwrap(), 2);
        assert_eq!(plane.append(0, 3).unwrap(), 0);
        assert_eq!((plane.origin_logical, plane.origin_physical), (1, 1));

        // Simulate installation into a detached layout whose physical origin
        // differs. Appends follow metadata, never `position % capacity`.
        plane.origin_physical = 2;
        assert_eq!(plane.append(0, 4).unwrap(), 2);
        assert_eq!((plane.origin_logical, plane.origin_physical), (2, 0));
    }

    #[test]
    fn remote_tiles_preserve_rotated_physical_origin_across_wrap() {
        let context = MetalContext::new().expect("Metal context");
        for head_major in [false, true] {
            let mut plane =
                MetalKvPlane::uninitialized(&context, 8, 2, head_major).expect("detached plane");
            plane.origin_logical = 5;
            plane.origin_physical = 5;
            plane.len = 4;
            let words = [50u16, 51, 60, 61, 70, 71, 80, 81];
            let bytes = words
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<_>>();
            plane
                .write_logical_tile(2, 1, 5, 4, true, &bytes)
                .expect("wrapped remote tile");
            plane
                .write_logical_tile(2, 1, 5, 4, false, &bytes)
                .expect("wrapped remote tile");
            let snapshot = plane.snapshot(2, 1);
            assert_eq!(snapshot.key_logical, words);
            assert_eq!(snapshot.value_logical, words);
            assert_eq!(plane.origin_physical, 5);
        }
    }

    #[test]
    fn snapshot_is_logical_and_restore_uses_a_detached_layout() {
        let context = MetalContext::new().expect("Metal context");
        let mut plane = MetalKvPlane::new(&context, 3, 2, false).expect("plane");
        for position in 0..5 {
            let physical = plane.append(0, position).unwrap();
            let start = physical * 2;
            plane.key.as_mut_bits()[start..start + 2]
                .copy_from_slice(&[(position * 4) as u16, (position * 4 + 1) as u16]);
            plane.value.as_mut_bits()[start..start + 2]
                .copy_from_slice(&[(position * 4 + 2) as u16, (position * 4 + 3) as u16]);
        }
        assert_eq!((plane.origin_logical, plane.origin_physical), (2, 2));
        let snapshot = plane.snapshot(2, 1);
        assert_eq!(snapshot.key_logical, vec![8, 9, 12, 13, 16, 17]);

        let restored = MetalKvPlane::detached_from(&context, &snapshot, 2, 1).unwrap();
        // Restores adopt the rotation a sequentially-built live ring holds at
        // this logical origin, so physical scan order (and therefore float
        // accumulation order) matches the session being replayed.
        assert_eq!(restored.origin_physical, plane.origin_physical);
        assert_eq!(restored.origin_physical, 2);
        assert_eq!(restored.snapshot(2, 1).key_logical, snapshot.key_logical);
        assert_eq!(
            restored.snapshot(2, 1).value_logical,
            snapshot.value_logical
        );
    }

    /// Write one recognizable K/V row for `position` at its physical slot.
    fn write_row(plane: &mut MetalKvPlane, kv_dim: usize, physical: usize, position: usize) {
        let start = physical * kv_dim;
        for element in 0..kv_dim {
            plane.key.as_mut_bits()[start + element] = (position * 10 + element) as u16;
            plane.value.as_mut_bits()[start + element] = (1_000 + position * 10 + element) as u16;
        }
    }

    /// Stand in for the producer: reserve each ring row and fill it.
    fn append_rows(plane: &mut MetalKvPlane, kv_dim: usize, start: usize, count: usize) {
        for position in start..start + count {
            let physical = plane.append(0, position).expect("contiguous append");
            write_row(plane, kv_dim, physical, position);
        }
    }

    /// The logical tail as ascending rows, independent of physical placement.
    fn expected_rows(kv_dim: usize, positions: std::ops::Range<usize>) -> (Vec<u16>, Vec<u16>) {
        let mut key = Vec::new();
        let mut value = Vec::new();
        for position in positions {
            for element in 0..kv_dim {
                key.push((position * 10 + element) as u16);
                value.push((1_000 + position * 10 + element) as u16);
            }
        }
        (key, value)
    }

    #[test]
    fn speculative_rejection_at_the_wrap_restores_evicted_rows() {
        let context = MetalContext::new().expect("Metal context");
        let mut plane = MetalKvPlane::new(&context, 4, 2, false).expect("plane");
        append_rows(&mut plane, 2, 0, 4);
        let checkpoint = plane
            .speculative_checkpoint(2, 2, true, 3)
            .expect("checkpoint");
        // The producer writes three candidates over the oldest ring rows.
        append_rows(&mut plane, 2, 4, 3);
        plane
            .restore_speculative(0, &checkpoint, 2, 2, true, 4, 3, 0)
            .expect("rejection");

        let restored = plane.snapshot(2, 2);
        let (key, value) = expected_rows(2, 0..4);
        assert_eq!(restored.key_logical, key);
        assert_eq!(restored.value_logical, value);
        assert_eq!(
            (plane.origin_logical, plane.origin_physical, plane.len),
            (0, 0, 4)
        );
    }

    #[test]
    fn speculative_partial_accept_at_the_wrap_commits_only_the_accepted_prefix() {
        let context = MetalContext::new().expect("Metal context");
        let mut plane = MetalKvPlane::new(&context, 4, 2, false).expect("plane");
        append_rows(&mut plane, 2, 0, 4);
        let checkpoint = plane
            .speculative_checkpoint(2, 2, true, 3)
            .expect("checkpoint");
        append_rows(&mut plane, 2, 4, 3);
        plane
            .restore_speculative(0, &checkpoint, 2, 2, true, 4, 3, 1)
            .expect("partial accept");

        // Position 4 evicts 0; the two rejected destinations hold 1 and 2
        // again, so the live window is exactly 1..5.
        let committed = plane.snapshot(2, 2);
        let (key, value) = expected_rows(2, 1..5);
        assert_eq!(committed.key_logical, key);
        assert_eq!(committed.value_logical, value);
        assert_eq!(
            (plane.origin_logical, plane.origin_physical, plane.len),
            (1, 1, 4)
        );
    }

    #[test]
    fn speculative_block_that_fills_then_wraps_commits_every_accepted_row() {
        let context = MetalContext::new().expect("Metal context");
        let mut plane = MetalKvPlane::new(&context, 4, 2, false).expect("plane");
        append_rows(&mut plane, 2, 0, 2);
        // Three candidates fill the remaining two rows and then wrap onto the
        // live row 0.
        let checkpoint = plane
            .speculative_checkpoint(2, 2, true, 3)
            .expect("checkpoint");
        assert_eq!(checkpoint.overwritten_physical, vec![2, 3, 0]);
        append_rows(&mut plane, 2, 2, 3);
        plane
            .restore_speculative(0, &checkpoint, 2, 2, true, 2, 3, 3)
            .expect("full accept");

        let committed = plane.snapshot(2, 2);
        let (key, value) = expected_rows(2, 1..5);
        assert_eq!(committed.key_logical, key);
        assert_eq!(committed.value_logical, value);
        assert_eq!(
            (plane.origin_logical, plane.origin_physical, plane.len),
            (1, 1, 4)
        );
    }

    #[test]
    fn speculative_block_that_fills_then_wraps_restores_the_evicted_row_on_rejection() {
        let context = MetalContext::new().expect("Metal context");
        let mut plane = MetalKvPlane::new(&context, 4, 2, false).expect("plane");
        append_rows(&mut plane, 2, 0, 2);
        let checkpoint = plane
            .speculative_checkpoint(2, 2, true, 3)
            .expect("checkpoint");
        append_rows(&mut plane, 2, 2, 3);
        plane
            .restore_speculative(0, &checkpoint, 2, 2, true, 2, 3, 0)
            .expect("rejection");

        let restored = plane.snapshot(2, 2);
        let (key, value) = expected_rows(2, 0..2);
        assert_eq!(restored.key_logical, key, "row 0 must survive the wrap");
        assert_eq!(restored.value_logical, value);
        assert_eq!(
            (plane.origin_logical, plane.origin_physical, plane.len),
            (0, 0, 2)
        );
    }

    #[test]
    fn speculative_nope_rejection_rewinds_growth_without_copying_rows() {
        let context = MetalContext::new().expect("Metal context");
        let mut plane = MetalKvPlane::new(&context, 8, 2, true).expect("plane");
        append_rows(&mut plane, 2, 0, 2);
        let checkpoint = plane
            .speculative_checkpoint(2, 2, false, 3)
            .expect("checkpoint");
        assert!(
            checkpoint.overwritten_key.is_empty(),
            "a growing NoPE plane never retains rows"
        );
        append_rows(&mut plane, 2, 2, 3);
        plane
            .restore_speculative(0, &checkpoint, 2, 2, false, 2, 3, 0)
            .expect("rejection");

        let restored = plane.snapshot(2, 2);
        let (key, value) = expected_rows(2, 0..2);
        assert_eq!(restored.key_logical, key);
        assert_eq!(restored.value_logical, value);
        assert_eq!((plane.origin_logical, plane.len), (0, 2));
    }

    #[test]
    fn real_model_detached_restore_replays_exact_suffix() {
        let Ok(path) = std::env::var("MUSER_MODEL") else {
            eprintln!("skipped: set MUSER_MODEL to the development Muse GGUF");
            return;
        };
        let loaded = crate::loader::load_components(std::path::Path::new(&path))
            .expect("Muse GGUF must load");
        let tokens = [200_000, 19_873, 24];
        let mut cpu = MuseModel::new(loaded.config.clone(), loaded.weights.clone(), tokens.len());
        let cpu_logits = cpu.forward(&tokens, None);
        let cpu_last = &cpu_logits[cpu_logits.len() - loaded.config.vocab_size..];

        let mut metal = MetalMuseModel::new(loaded.config.clone(), loaded.weights, tokens.len())
            .expect("Metal model");
        metal.forward(&tokens[..2]).expect("prefix");
        let snapshot = metal.snapshot();
        let first = metal.forward(&tokens[2..]).expect("first suffix");
        metal.restore(&snapshot).expect("detached restore");
        let repeated = metal.forward(&tokens[2..]).expect("replayed suffix");

        assert!(
            first.iter().all(|value| value.is_finite()),
            "live suffix produced non-finite logits"
        );
        assert!(
            repeated.iter().all(|value| value.is_finite()),
            "restored suffix produced non-finite logits"
        );
        assert_eq!(first, repeated, "restored suffix must replay bit-exactly");
        let max_error = cpu_last
            .iter()
            .zip(&first)
            .map(|(cpu, gpu)| (cpu - gpu).abs())
            .fold(0.0f32, f32::max);
        assert!(max_error <= 0.5, "CPU/Metal max logit error {max_error}");
        assert_eq!(argmax(cpu_last), argmax(&first));
    }

    /// Release-only regression for the authenticated remote-install path.
    ///
    /// Unlike `real_model_detached_restore_replays_exact_suffix`, this starts
    /// the consumer slot with fresh activation workspaces and fills only its
    /// detached KV planes. That distinction catches accidental dependencies
    /// on prefill workspace contents even when the exported logical KV bytes
    /// compare exactly. Set `MUSER_REMOTE_TOKEN_FIXTURE` to a whitespace-
    /// separated token fixture; its last token is held for local decode.
    #[test]
    fn real_model_remote_install_replays_exact_held_token() {
        let (Ok(model_path), Ok(token_path)) = (
            std::env::var("MUSER_MODEL"),
            std::env::var("MUSER_REMOTE_TOKEN_FIXTURE"),
        ) else {
            eprintln!(
                "not run: set MUSER_MODEL and MUSER_REMOTE_TOKEN_FIXTURE for remote-install parity"
            );
            return;
        };
        let loaded = crate::loader::load_components(std::path::Path::new(&model_path))
            .expect("Muse GGUF must load");
        let raw = std::fs::read_to_string(token_path).expect("token fixture must be readable");
        let tokens = raw
            .split_whitespace()
            .map(|value| value.parse::<u32>().expect("fixture token must be u32"))
            .collect::<Vec<_>>();
        assert!(tokens.len() >= 2, "fixture needs a prefix and held token");
        let cached = &tokens[..tokens.len() - 1];
        let held = tokens[tokens.len() - 1];
        let mut slots =
            MetalMuseModel::new_sequence_group(loaded.config, loaded.weights, tokens.len() + 1, 2)
                .expect("two Metal slots");
        slots[0].forward(cached).expect("local prefix");
        let snapshot = slots[0].snapshot();

        let mut install = slots[1]
            .begin_remote_kv_install(Arc::from(cached.to_vec()))
            .expect("detached remote generation");
        for (layer, plane) in snapshot.layers.iter().enumerate() {
            let key = plane
                .key_logical
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            let value = plane
                .value_logical
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            install
                .write_f16_tile(
                    layer,
                    true,
                    plane.origin_logical as u64,
                    plane.len as u64,
                    &key,
                )
                .expect("remote key plane");
            install
                .write_f16_tile(
                    layer,
                    false,
                    plane.origin_logical as u64,
                    plane.len as u64,
                    &value,
                )
                .expect("remote value plane");
        }
        slots[1]
            .commit_remote_kv_install(install)
            .expect("remote commit");

        let local = slots[0].forward(&[held]).expect("local held token");
        let remote = slots[1].forward(&[held]).expect("remote held token");
        assert_eq!(slots[0].snapshot().layers, slots[1].snapshot().layers);
        assert_eq!(local, remote, "remote install changed held-token logits");
    }

    /// Delta variant of the remote-install parity gate: the consumer slot
    /// already holds `[0, cut)`; the install copies that prefix out of the
    /// live planes and accepts only suffix tiles. The committed cache must be
    /// byte-identical to the sequentially-built full prefill.
    #[test]
    fn real_model_remote_delta_install_replays_exact_held_token() {
        let (Ok(model_path), Ok(token_path)) = (
            std::env::var("MUSER_MODEL"),
            std::env::var("MUSER_REMOTE_TOKEN_FIXTURE"),
        ) else {
            eprintln!(
                "not run: set MUSER_MODEL and MUSER_REMOTE_TOKEN_FIXTURE for delta-install parity"
            );
            return;
        };
        let loaded = crate::loader::load_components(std::path::Path::new(&model_path))
            .expect("Muse GGUF must load");
        let raw = std::fs::read_to_string(token_path).expect("token fixture must be readable");
        let tokens = raw
            .split_whitespace()
            .map(|value| value.parse::<u32>().expect("fixture token must be u32"))
            .collect::<Vec<_>>();
        assert!(
            tokens.len() >= 3,
            "delta fixture needs a prefix with cut room and a held token"
        );
        let cached = &tokens[..tokens.len() - 1];
        let held = tokens[tokens.len() - 1];
        let cut = cached.len() / 2;
        assert!(cut > 0, "delta fixture needs a nonempty held prefix");
        let mut slots =
            MetalMuseModel::new_sequence_group(loaded.config, loaded.weights, tokens.len() + 1, 2)
                .expect("two Metal slots");
        slots[0].forward(cached).expect("local prefix");
        let snapshot = slots[0].snapshot();

        slots[1].forward(&cached[..cut]).expect("held prefix");
        let mut install = slots[1]
            .begin_remote_kv_install_delta(Arc::from(cached.to_vec()), cut)
            .expect("detached delta generation");
        for (layer, plane) in snapshot.layers.iter().enumerate() {
            // The span schedule ships only what the cut does not already
            // hold: NoPE planes from the cut, SWA planes from the window.
            let origin = plane.origin_logical as u64;
            let start = (cut as u64).max(origin);
            let count = cached.len() as u64 - start;
            let kv_dim = plane.key_logical.len() / plane.len;
            let from = (start - origin) as usize * kv_dim;
            let key = plane.key_logical[from..]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            let value = plane.value_logical[from..]
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>();
            install
                .write_f16_tile(layer, true, start, count, &key)
                .expect("remote key span");
            install
                .write_f16_tile(layer, false, start, count, &value)
                .expect("remote value span");
        }
        slots[1]
            .commit_remote_kv_install(install)
            .expect("delta commit");

        let local = slots[0].forward(&[held]).expect("local held token");
        let remote = slots[1].forward(&[held]).expect("remote held token");
        assert_eq!(slots[0].snapshot().layers, slots[1].snapshot().layers);
        assert_eq!(local, remote, "delta install changed held-token logits");
    }

    #[test]
    fn real_model_shared_sequence_group_keeps_kv_and_logits_isolated() {
        let Ok(path) = std::env::var("MUSER_MODEL") else {
            eprintln!("not run: set MUSER_MODEL for the Apple release workflow");
            return;
        };
        let loaded = crate::loader::load_components(std::path::Path::new(&path))
            .expect("Muse GGUF must load");
        let mut group = MetalMuseModel::new_sequence_group(loaded.config, loaded.weights, 8, 2)
            .expect("shared sequence group");
        assert!(std::sync::Arc::ptr_eq(&group[0].shared, &group[1].shared));
        let prefix = [200_000, 19_873];
        let first = group[0].forward(&prefix).expect("slot zero prefill");
        assert_eq!(group[0].position(), 2);
        assert_eq!(group[1].position(), 0);
        let second = group[1].forward(&prefix).expect("slot one prefill");
        assert_eq!(first, second, "shared resources changed isolated logits");
        group[0].forward(&[24]).expect("slot zero decode");
        assert_eq!(group[0].position(), 3);
        assert_eq!(group[1].position(), 2);
    }

    #[test]
    fn real_model_packed_decode_matches_isolated_rows_and_kv() {
        let Ok(path) = std::env::var("MUSER_MODEL") else {
            eprintln!("not run: set MUSER_MODEL for the Apple release workflow");
            return;
        };
        let loaded = crate::loader::load_components(std::path::Path::new(&path))
            .expect("Muse GGUF must load");
        let vocab = loaded.config.vocab_size;
        let mut group = MetalMuseModel::new_sequence_group(loaded.config, loaded.weights, 16, 4)
            .expect("shared sequence group");
        let prompts: [&[u32]; 4] = [
            &[200_000, 19_873],
            &[200_000, 24, 10_676],
            &[200_000, 19_873, 24, 768],
            &[200_000],
        ];
        let tokens = [24, 768, 10_676, 19_873];
        for (row, prompt) in group.iter_mut().zip(prompts) {
            row.forward(prompt).expect("row prefill");
        }
        let baseline = group
            .iter()
            .map(MetalMuseModel::snapshot)
            .collect::<Vec<_>>();
        let isolated = group
            .iter_mut()
            .zip(tokens)
            .map(|(row, token)| row.forward(&[token]).expect("isolated decode"))
            .collect::<Vec<_>>();
        let isolated_kv = group
            .iter()
            .map(MetalMuseModel::snapshot)
            .collect::<Vec<_>>();
        for (row, snapshot) in group.iter_mut().zip(&baseline) {
            row.restore(snapshot).expect("restore row");
        }

        let mut rows = group.iter_mut().collect::<Vec<_>>();
        let packed = MetalMuseModel::forward_decode_group(&mut rows, &tokens)
            .expect("four-row packed decode");
        for row in 0..4 {
            assert_eq!(
                argmax(&isolated[row]),
                argmax(&packed[row]),
                "packed row {row} selected a different token"
            );
            let max_error = isolated[row]
                .iter()
                .zip(&packed[row])
                .map(|(expected, actual)| (expected - actual).abs())
                .fold(0.0f32, f32::max);
            assert_eq!(max_error, 0.0, "packed row {row} max logit error");
            assert_eq!(packed[row].len(), vocab);
            let actual = rows[row].snapshot();
            assert_eq!(actual.n_past, isolated_kv[row].n_past);
            for (layer, (actual, expected)) in actual
                .layers
                .iter()
                .zip(&isolated_kv[row].layers)
                .enumerate()
            {
                assert_eq!(
                    actual.key_logical, expected.key_logical,
                    "row {row} layer {layer} key drift"
                );
                assert_eq!(
                    actual.value_logical, expected.value_logical,
                    "row {row} layer {layer} value drift"
                );
            }
        }
    }

    #[test]
    fn real_model_llama_full_logits_match_bit_exactly() {
        let (Ok(model_path), Ok(logits_path)) = (
            std::env::var("MUSER_MODEL"),
            std::env::var("MUSER_LLAMA_LOGITS").or_else(|_| std::env::var("MUSER_GX10_LOGITS")),
        ) else {
            eprintln!("not run: set MUSER_MODEL and MUSER_LLAMA_LOGITS for llama parity");
            return;
        };
        let loaded = crate::loader::load_components(std::path::Path::new(&model_path))
            .expect("Muse GGUF must load");
        let context = std::env::var("MUSER_TEST_CONTEXT")
            .ok()
            .map(|raw| raw.parse::<usize>().expect("test context must be usize"))
            .unwrap_or(16);
        let mut metal = MetalMuseModel::new(loaded.config.clone(), loaded.weights, context)
            .expect("Metal model");
        let mut tokens = std::env::var("MUSER_LLAMA_TOKEN_FILE")
            .or_else(|_| std::env::var("MUSER_GX10_TOKEN_FILE"))
            .ok()
            .map(|path| std::fs::read_to_string(path).expect("llama logit token file"))
            .or_else(|| {
                std::env::var("MUSER_LLAMA_TOKENS")
                    .or_else(|_| std::env::var("MUSER_GX10_TOKENS"))
                    .ok()
            })
            .map(|raw| {
                raw.split(|character: char| character == ',' || character.is_ascii_whitespace())
                    .filter(|part| !part.is_empty())
                    .map(|part| part.parse::<u32>().expect("logit token must be u32"))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| vec![200_000, 198, 17_360]);
        if let Ok(held) = std::env::var("MUSER_LLAMA_HELD_TOKEN")
            .or_else(|_| std::env::var("MUSER_GX10_HELD_TOKEN"))
        {
            tokens.push(held.parse::<u32>().expect("held logit token must be u32"));
        }
        assert!(
            tokens.len() >= 2,
            "logit differential needs a prefix and held token"
        );
        let layer_dir = std::env::var("MUSER_GX10_LAYER_DIR").ok();
        let layers = (0..loaded.config.n_layers).collect::<Vec<_>>();
        metal
            .forward(&tokens[..tokens.len() - 1])
            .expect("local logit prefix");
        if let Ok(kv_dir) = std::env::var("MUSER_LLAMA_KV_DIR") {
            install_test_kv_fixture(&mut metal, &tokens[..tokens.len() - 1], &kv_dir);
        }
        let (local_rows, captured) = if layer_dir.is_some() {
            metal
                .forward_capturing_layers(&tokens[tokens.len() - 1..], &layers)
                .expect("local held-token decode with layer capture")
        } else {
            (
                metal
                    .forward(&tokens[tokens.len() - 1..])
                    .expect("local held-token decode"),
                Vec::new(),
            )
        };
        let local = local_rows
            .chunks_exact(loaded.config.vocab_size)
            .last()
            .expect("final local logit row");
        if let Some(layer_dir) = layer_dir {
            let hidden = loaded.config.hidden_dim;
            for layer in 0..loaded.config.n_layers {
                let path = std::path::Path::new(&layer_dir).join(format!("l_out-{layer}.f32"));
                let bytes = std::fs::read(&path)
                    .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
                assert_eq!(bytes.len(), hidden * 4, "{} byte length", path.display());
                let remote = bytes
                    .chunks_exact(4)
                    .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte float")))
                    .collect::<Vec<_>>();
                let start = layer * hidden;
                let local_layer = &captured[start..start + hidden];
                let exact = local_layer
                    .iter()
                    .zip(&remote)
                    .filter(|(left, right)| left.to_bits() == right.to_bits())
                    .count();
                let max_error = local_layer
                    .iter()
                    .zip(&remote)
                    .map(|(left, right)| (left - right).abs())
                    .fold(0.0f32, f32::max);
                eprintln!("llama layer {layer}: exact={exact}/{hidden} max_abs={max_error}");
                if exact != hidden {
                    break;
                }
            }
        }
        let bytes = std::fs::read(logits_path).expect("GX10 raw-f32 logits");
        assert_eq!(bytes.len(), loaded.config.vocab_size * 4);
        let remote = bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte float")))
            .collect::<Vec<_>>();
        let max_error = local
            .iter()
            .zip(&remote)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        let mean_error = local
            .iter()
            .zip(&remote)
            .map(|(left, right)| f64::from((left - right).abs()))
            .sum::<f64>()
            / local.len() as f64;
        let exact = local
            .iter()
            .zip(&remote)
            .filter(|(left, right)| left.to_bits() == right.to_bits())
            .count();
        assert_eq!(argmax(local), argmax(&remote), "llama selected token drift");
        assert_eq!(
            exact,
            local.len(),
            "llama full-logit drift: exact={exact}/{} max={max_error} mean={mean_error}",
            local.len()
        );
    }

    #[test]
    fn real_model_greedy_cancellation_restores_the_visible_prefix() {
        let (Ok(model_path), Ok(token_path), Ok(held)) = (
            std::env::var("MUSER_MODEL"),
            std::env::var("MUSER_LLAMA_TOKEN_FILE"),
            std::env::var("MUSER_LLAMA_HELD_TOKEN"),
        ) else {
            eprintln!(
                "not run: set MUSER_MODEL, MUSER_LLAMA_TOKEN_FILE, and MUSER_LLAMA_HELD_TOKEN"
            );
            return;
        };
        let prefix = std::fs::read_to_string(token_path)
            .expect("llama token file")
            .split_ascii_whitespace()
            .map(|token| token.parse::<u32>().expect("llama token must be u32"))
            .collect::<Vec<_>>();
        let held = held.parse::<u32>().expect("held token must be u32");
        let mut visible_tokens = prefix.clone();
        visible_tokens.push(held);
        let loaded = crate::loader::load_components(std::path::Path::new(&model_path))
            .expect("Muse GGUF must load");
        let mut metal =
            MetalMuseModel::new(loaded.config, loaded.weights, visible_tokens.len() + 3)
                .expect("Metal model");
        metal.forward(&prefix).expect("greedy cancellation prefix");
        let checkpoint = metal
            .export_cache_snapshot(&prefix)
            .expect("export greedy checkpoint");
        metal.forward(&[held]).expect("ordinary visible token");
        let expected = metal
            .export_cache_snapshot(&visible_tokens)
            .expect("export ordinary visible prefix");
        metal
            .install_cache_snapshot(&checkpoint)
            .expect("restore greedy checkpoint");

        let result = metal
            .forward_greedy_streaming(held, 4, &[], |_| false)
            .expect("cancelled greedy block");
        assert!(result.cancelled);
        assert_eq!(result.consumed_tokens, vec![held]);
        assert!(result.final_logits.is_empty());
        let actual = metal
            .export_cache_snapshot(&visible_tokens)
            .expect("export cancelled visible prefix");
        assert_eq!(
            actual, expected,
            "cancelled queued rows leaked into live KV"
        );
    }

    /// Local-only verifier feedback loop. This intentionally avoids the
    /// server, handoff, and Spark; the optional in-engine phase timer is the
    /// only breakdown mechanism used by the F-series remediation loop.
    #[test]
    fn real_model_local_m16_verify_diagnostic() {
        let (Ok(model_path), Ok(token_path)) = (
            std::env::var("MUSER_MODEL"),
            std::env::var("MUSER_LLAMA_TOKEN_FILE"),
        ) else {
            eprintln!("not run: set MUSER_MODEL and MUSER_LLAMA_TOKEN_FILE");
            return;
        };
        const PREFIX: usize = 32;
        const BLOCK: usize = 16;
        let tokens = std::fs::read_to_string(token_path)
            .expect("local verifier token file")
            .split_ascii_whitespace()
            .map(|token| token.parse::<u32>().expect("token must be u32"))
            .collect::<Vec<_>>();
        assert!(
            tokens.len() >= PREFIX + BLOCK,
            "local verifier fixture needs at least {} tokens",
            PREFIX + BLOCK
        );
        let loaded = crate::loader::load_components(std::path::Path::new(&model_path))
            .expect("local verifier GGUF must load");
        let mut metal = MetalMuseModel::new(loaded.config, loaded.weights, PREFIX + BLOCK + 1)
            .expect("local verifier Metal model");
        metal
            .forward(&tokens[..PREFIX])
            .expect("local verifier prefix");
        let started = std::time::Instant::now();
        let (logits, captures) = metal
            .forward_batch_all_logits_capturing(&tokens[PREFIX..PREFIX + BLOCK], &[])
            .expect("local 16-row verification");
        let elapsed = started.elapsed();
        assert_eq!(logits.len(), BLOCK * metal.cfg.vocab_size);
        assert!(captures.is_empty());
        println!(
            "MUSER_LOCAL_VERIFY rows={BLOCK} prefix={PREFIX} wall_ms={:.3} logits={} position={}",
            elapsed.as_secs_f64() * 1_000.0,
            logits.len(),
            metal.position(),
        );
    }

    /// The speculative-route analogue of the greedy cancellation fixture: a
    /// verification batch always writes all sixteen candidate rows, so a
    /// zero-length commit must restore the pre-batch KV byte-for-byte, and
    /// every partial commit must leave a state whose subsequent greedy
    /// continuation is the exact sequential chain. Batch-written accepted KV
    /// rows are not byte-compared against sequential decode: multi-row
    /// projections are a flagged-unreconciled boundary, and the production
    /// losslessness gate is token equality, not KV bytes.
    #[test]
    fn real_model_speculative_rollback_restores_the_visible_prefix() {
        let (Ok(model_path), Ok(token_path), Ok(held)) = (
            std::env::var("MUSER_MODEL"),
            std::env::var("MUSER_LLAMA_TOKEN_FILE"),
            std::env::var("MUSER_LLAMA_HELD_TOKEN"),
        ) else {
            eprintln!(
                "not run: set MUSER_MODEL, MUSER_LLAMA_TOKEN_FILE, and MUSER_LLAMA_HELD_TOKEN"
            );
            return;
        };
        const BLOCK: usize = 16;
        const CONTINUATION: usize = 8;
        let prefix = std::fs::read_to_string(token_path)
            .expect("llama token file")
            .split_ascii_whitespace()
            .map(|token| token.parse::<u32>().expect("llama token must be u32"))
            .collect::<Vec<_>>();
        // The held token pins the fixture identity like the greedy
        // cancellation test; the verified chain itself is whatever the
        // target's own greedy continuation is.
        let _held = held.parse::<u32>().expect("held token must be u32");
        let loaded = crate::loader::load_components(std::path::Path::new(&model_path))
            .expect("Muse GGUF must load");
        let vocab = loaded.config.vocab_size;
        let mut metal = MetalMuseModel::new(
            loaded.config,
            loaded.weights,
            prefix.len() + BLOCK + CONTINUATION + 1,
        )
        .expect("Metal model");
        let starting_logits = metal.forward(&prefix).expect("speculative prefix");
        let checkpoint = metal
            .export_cache_snapshot(&prefix)
            .expect("export speculative base checkpoint");

        // Discover the true greedy chain through the block plus the
        // continuation window used to prove post-commit state integrity.
        let mut chain = Vec::with_capacity(BLOCK + CONTINUATION);
        let mut logits = starting_logits.clone();
        for _ in 0..BLOCK + CONTINUATION {
            let next = argmax(&logits) as u32;
            chain.push(next);
            logits = metal.forward(&[next]).expect("ordinary chain token");
        }
        metal
            .install_cache_snapshot(&checkpoint)
            .expect("restore speculative base checkpoint");

        // Full rollback: every candidate rejected, nothing may stay live.
        {
            let wrong = (chain[0] + 1) % vocab as u32;
            let candidates = vec![wrong; BLOCK];
            let speculative = metal
                .speculative_checkpoint(candidates.len())
                .expect("speculative checkpoint");
            let (flat_logits, _hidden) = metal
                .forward_batch_all_logits_capturing(&candidates, &[])
                .expect("verification batch");
            assert_ne!(
                argmax(&starting_logits) as u32,
                candidates[0],
                "rejection case must reject the first candidate"
            );
            assert!(!flat_logits.is_empty());
            metal
                .commit_speculative_prefix(speculative, 0)
                .expect("roll back the whole speculative batch");
            let actual = metal
                .export_cache_snapshot(&prefix)
                .expect("export rolled-back prefix");
            assert_eq!(
                actual, checkpoint,
                "full speculative rollback changed the pre-batch KV"
            );
        }

        // Partial and full commits: the batched acceptance decision must
        // match sequential decode, and decoding on from the committed state
        // must reproduce the sequential greedy continuation exactly.
        for accepted in [1usize, 4, BLOCK] {
            let wrong = (chain.get(accepted).copied().unwrap_or(0) + 1) % vocab as u32;
            let mut candidates = chain[..accepted].to_vec();
            candidates.resize(BLOCK, wrong);
            let speculative = metal
                .speculative_checkpoint(candidates.len())
                .expect("speculative checkpoint");
            let (flat_logits, _hidden) = metal
                .forward_batch_all_logits_capturing(&candidates, &[])
                .expect("verification batch");
            let mut row_logits = starting_logits.as_slice();
            let mut decided = 0usize;
            for (offset, &candidate) in candidates.iter().enumerate() {
                if argmax(row_logits) as u32 != candidate {
                    break;
                }
                decided = offset + 1;
                if offset + 1 < BLOCK {
                    row_logits = &flat_logits[offset * vocab..(offset + 1) * vocab];
                }
            }
            assert_eq!(
                decided, accepted,
                "batched verification disagreed with sequential decode"
            );
            metal
                .commit_speculative_prefix(speculative, accepted)
                .expect("commit accepted speculative prefix");
            let mut continued = Vec::with_capacity(CONTINUATION);
            let mut logits = flat_logits[(accepted - 1) * vocab..accepted * vocab].to_vec();
            for _ in 0..CONTINUATION {
                let next = argmax(&logits) as u32;
                continued.push(next);
                logits = metal.forward(&[next]).expect("post-commit continuation");
            }
            assert_eq!(
                continued,
                chain[accepted..accepted + CONTINUATION].to_vec(),
                "post-commit state produced a different greedy chain (accepted={accepted})"
            );
            metal
                .install_cache_snapshot(&checkpoint)
                .expect("restore speculative base checkpoint");
        }
    }

    #[test]
    fn real_model_llama_layer0_stage_differential() {
        let (Ok(model_path), Ok(stage_dir)) = (
            std::env::var("MUSER_MODEL"),
            std::env::var("MUSER_LLAMA_STAGE_DIR"),
        ) else {
            eprintln!("not run: set MUSER_MODEL and MUSER_LLAMA_STAGE_DIR");
            return;
        };
        let loaded = crate::loader::load_components(std::path::Path::new(&model_path))
            .expect("Muse GGUF must load");
        let context = std::env::var("MUSER_TEST_CONTEXT")
            .ok()
            .map(|raw| raw.parse::<usize>().expect("test context must be usize"))
            .unwrap_or(16);
        let layer = std::env::var("MUSER_LLAMA_STAGE_LAYER")
            .ok()
            .map(|raw| raw.parse::<usize>().expect("stage layer must be usize"))
            .unwrap_or(0);
        let mut metal =
            MetalMuseModel::new(loaded.config, loaded.weights, context).expect("Metal model");
        let mut tokens = std::env::var("MUSER_LLAMA_STAGE_TOKEN_FILE")
            .ok()
            .map(|path| std::fs::read_to_string(path).expect("stage token file"))
            .or_else(|| std::env::var("MUSER_LLAMA_STAGE_TOKENS").ok())
            .map(|raw| {
                raw.split(',')
                    .flat_map(str::split_whitespace)
                    .map(|part| part.parse::<u32>().expect("stage token must be u32"))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| vec![200_000, 198, 17_360]);
        if let Ok(held) = std::env::var("MUSER_LLAMA_STAGE_HELD_TOKEN") {
            tokens.push(held.parse::<u32>().expect("held stage token must be u32"));
        }
        assert!(
            tokens.len() >= 2,
            "stage differential needs a prefix and held token"
        );
        let stages = if std::env::var_os("MUSER_LLAMA_STAGE_PREFILL").is_some() {
            metal
                .forward_capturing_debug_layer(&tokens[..tokens.len() - 1], layer)
                .expect("prefill stage capture")
        } else {
            metal
                .forward(&tokens[..tokens.len() - 1])
                .expect("local stage prefix");
            if let Ok(kv_dir) = std::env::var("MUSER_LLAMA_KV_DIR") {
                install_test_kv_fixture(&mut metal, &tokens[..tokens.len() - 1], &kv_dir);
            }
            metal
                .forward_capturing_debug_layer(&tokens[tokens.len() - 1..], layer)
                .expect("held-token stage capture")
        };
        let selected = std::env::var("MUSER_LLAMA_STAGE_NAMES").ok().map(|raw| {
            raw.split(',')
                .map(str::to_string)
                .collect::<std::collections::BTreeSet<_>>()
        });
        let output_dir = std::env::var("MUSER_LOCAL_STAGE_DIR")
            .ok()
            .map(std::path::PathBuf::from);
        if let Some(output_dir) = &output_dir {
            std::fs::create_dir_all(output_dir).unwrap_or_else(|error| {
                panic!("create stage directory {}: {error}", output_dir.display())
            });
        }
        let mut compared = 0usize;
        let mut all_exact = true;
        for (name, local) in stages {
            if selected
                .as_ref()
                .is_some_and(|selected| !selected.contains(name))
            {
                continue;
            }
            compared += 1;
            if let Some(output_dir) = &output_dir {
                let output = output_dir.join(format!("{name}.f32"));
                let bytes = local
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>();
                std::fs::write(&output, bytes)
                    .unwrap_or_else(|error| panic!("write {}: {error}", output.display()));
            }
            let reference_name = name
                .strip_suffix("-0")
                .map(|base| format!("{base}-{layer}"))
                .unwrap_or_else(|| (*name).to_owned());
            let path = std::path::Path::new(&stage_dir).join(format!("{reference_name}.f32"));
            if !path.exists() {
                continue;
            }
            let bytes = std::fs::read(&path)
                .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
            assert_eq!(
                bytes.len(),
                local.len() * 4,
                "{} byte length",
                path.display()
            );
            let reference = bytes
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte float")))
                .collect::<Vec<_>>();
            let exact = local
                .iter()
                .zip(&reference)
                .filter(|(left, right)| left.to_bits() == right.to_bits())
                .count();
            let max_error = local
                .iter()
                .zip(&reference)
                .map(|(left, right)| (left - right).abs())
                .fold(0.0f32, f32::max);
            eprintln!(
                "llama stage {name}: exact={exact}/{} max_abs={max_error}",
                local.len()
            );
            all_exact &= exact == local.len();
        }
        assert!(compared > 0, "stage differential selected no stages");
        assert!(all_exact, "one or more pinned llama layer stages drifted");
    }

    #[test]
    fn real_model_llama_lm_head_isolated_differential() {
        let (Ok(model_path), Ok(norm_path), Ok(projection_path)) = (
            std::env::var("MUSER_MODEL"),
            std::env::var("MUSER_LLAMA_RESULT_NORM"),
            std::env::var("MUSER_LLAMA_RESULT_PROJECTION"),
        ) else {
            eprintln!(
                "not run: set MUSER_MODEL, MUSER_LLAMA_RESULT_NORM, and MUSER_LLAMA_RESULT_PROJECTION"
            );
            return;
        };
        let loaded = crate::loader::load_components(std::path::Path::new(&model_path))
            .expect("Muse GGUF must load");
        let hidden_bytes = std::fs::read(norm_path).expect("read llama result norm");
        let hidden = hidden_bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte float")))
            .collect::<Vec<_>>();
        assert_eq!(hidden.len(), loaded.config.hidden_dim);
        let output_layout = loaded
            .weights
            .layout("output.weight")
            .expect("output weight layout");
        let mapped = loaded.weights.mapped_file();
        let metal =
            MetalMuseModel::new(loaded.config.clone(), loaded.weights, 1).expect("Metal model");
        let input = GpuBuffer::from_f32(&metal.context, &hidden).expect("LM-head input");
        let output =
            GpuBuffer::zeros(&metal.context, loaded.config.vocab_size).expect("LM-head output");
        let command = metal.context.queue.new_command_buffer();
        let serial = GraphEncoder::serial(command.new_compute_command_encoder());
        metal.project(&serial, &metal.output, &input, &output);
        serial.encoder.end_encoding();
        command.commit();
        metal
            .context
            .wait_for_completion(command, Duration::from_secs(300))
            .expect("isolated LM head");
        let expected_bytes = std::fs::read(projection_path).expect("read llama result projection");
        let expected = expected_bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte float")))
            .collect::<Vec<_>>();
        assert_eq!(expected.len(), output.len());
        let exact = output
            .as_slice()
            .iter()
            .zip(&expected)
            .filter(|(left, right)| left.to_bits() == right.to_bits())
            .count();
        let max_error = output
            .as_slice()
            .iter()
            .zip(&expected)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        if output_layout.dtype == crate::gguf::GgmlType::Q5_K {
            let row_bytes = output_layout.n_in / 256 * 176;
            let weights = &mapped[output_layout.file_offset..output_layout.file_offset + row_bytes];
            let cpu = crate::quant::dot_q5_k_f32(weights, &hidden, output_layout.n_in);
            eprintln!(
                "isolated LM head row0: cpu={cpu:?} metal={:?} llama={:?} layout={output_layout:?}",
                output.as_slice()[0],
                expected[0]
            );
        }
        assert_eq!(
            exact,
            expected.len(),
            "isolated llama LM-head drift: exact={exact}/{} max_abs={max_error}",
            expected.len()
        );
    }

    #[test]
    fn real_model_wrap_boundaries_and_detached_restore_replay_exactly() {
        let Ok(path) = std::env::var("MUSER_MODEL") else {
            eprintln!("skipped: set MUSER_MODEL to the development Muse GGUF");
            return;
        };
        let loaded = crate::loader::load_components(std::path::Path::new(&path))
            .expect("Muse GGUF must load");
        let cuts = [2_047usize, 2_048, 2_049, 2_559, 2_560];
        let tokens = (0..=2_560)
            .map(|index| {
                if index == 0 {
                    200_000
                } else {
                    [19_873, 24, 10_676, 768, 1_085, 13_634, 2_304, 1_509][(index - 1) % 8]
                }
            })
            .collect::<Vec<_>>();
        let mut metal = MetalMuseModel::new(loaded.config.clone(), loaded.weights, tokens.len())
            .expect("Metal model");
        let mut previous = 0;
        for cut in cuts {
            metal.forward(&tokens[previous..cut]).expect("prefix cut");
            assert_eq!(metal.position(), cut);
            let mut swa = 0;
            let mut nope = 0;
            for (kind, plane) in loaded.config.layer_kinds.iter().zip(&metal.cache) {
                match kind {
                    MuseLayerKind::SlidingRope => {
                        swa += 1;
                        assert_eq!(plane.len, cut.min(loaded.config.sliding_window));
                        assert_eq!(plane.origin_logical, cut.saturating_sub(plane.len));
                    }
                    MuseLayerKind::FullNoPe => {
                        nope += 1;
                        assert_eq!(plane.len, cut);
                        assert_eq!(plane.origin_logical, 0);
                    }
                }
                assert!(plane.origin_physical < plane.capacity);
            }
            assert_eq!((swa, nope), (39, 13));
            previous = cut;
        }

        let snapshot = metal.snapshot();
        let first = metal
            .forward(&tokens[2_560..2_561])
            .expect("wrapped suffix");
        assert_eq!(metal.position(), 2_561);
        metal
            .restore(&snapshot)
            .expect("detached restore at producer wrap");
        let repeated = metal
            .forward(&tokens[2_560..2_561])
            .expect("replayed wrapped suffix");
        assert_eq!(
            first, repeated,
            "2,560 restore must replay full logits exactly"
        );
    }

    #[test]
    fn real_model_operates_at_131008_when_long_gate_is_enabled() {
        if std::env::var_os("MUSER_RUN_131K").is_none() {
            eprintln!("skipped: set MUSER_RUN_131K=1 for the overnight long-context gate");
            return;
        }
        let path =
            std::env::var("MUSER_MODEL").expect("MUSER_MODEL is required with MUSER_RUN_131K");
        let loaded = crate::loader::load_components(std::path::Path::new(&path))
            .expect("Muse GGUF must load");
        let cut = 131_008usize;
        let tokens = (0..cut)
            .map(|index| {
                if index == 0 {
                    200_000
                } else {
                    [19_873, 24, 10_676, 768, 1_085, 13_634, 2_304, 1_509][(index - 1) % 8]
                }
            })
            .collect::<Vec<_>>();
        let mut metal = MetalMuseModel::new(loaded.config.clone(), loaded.weights, cut)
            .expect("131K Metal model");
        let logits = metal.forward(&tokens).expect("131008 prefill");
        assert_eq!(metal.position(), cut);
        assert!(logits.iter().all(|value| value.is_finite()));
        for (kind, plane) in loaded.config.layer_kinds.iter().zip(&metal.cache) {
            let expected = match kind {
                MuseLayerKind::SlidingRope => loaded.config.sliding_window,
                MuseLayerKind::FullNoPe => cut,
            };
            assert_eq!(plane.len, expected);
            assert_eq!(plane.origin_logical, cut - expected);
        }
    }

    fn argmax(values: &[f32]) -> usize {
        values
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index)
            .expect("nonempty logits")
    }
}
