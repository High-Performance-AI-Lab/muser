//! Ferrite-derived dense Metal projection backend for DFlash.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use objc::{sel, sel_impl};

use crate::dflash::{
    DFlashContextKvCache, DFlashDraftOutput, DFlashError, DFlashProjectionBackend, DFlashWeights,
};
use crate::gguf::GgmlType;
use crate::weights::TensorLayout;

use super::buffer::{GpuBuffer, GpuBytes};
use super::context::{MetalContext, MetalError};
use super::encode::MetalKernels;

struct Projection {
    weight: ProjectionWeight,
    input_width: usize,
    output_width: usize,
}

enum ProjectionWeight {
    Dense(GpuBuffer),
    Mapped(TensorLayout),
}

struct Scratch {
    input: GpuBuffer,
    output: GpuBuffer,
}

pub struct MetalDFlashProjection {
    context: MetalContext,
    kernels: MetalKernels,
    batch: usize,
    projections: BTreeMap<String, Projection>,
    scratch: Mutex<Scratch>,
}

struct MetalLayer {
    input_norm: GpuBuffer,
    post_attention_norm: GpuBuffer,
    q_norm: GpuBuffer,
    k_norm: GpuBuffer,
}

/// Persistent shared-memory arena for the complete five-layer assistant
/// forward. This is the accepted Ferrite layout: only the target-hidden rows
/// vary with the current verification batch; every other activation has fixed
/// block geometry and is retained between speculative rounds.
struct ForwardScratch {
    target_input: GpuBuffer,
    target_projected: GpuBuffer,
    hidden: GpuBuffer,
    normed: GpuBuffer,
    q: GpuBuffer,
    k_noise: GpuBuffer,
    v_noise: GpuBuffer,
    k_context: GpuBuffer,
    v_context: GpuBuffer,
    attention: GpuBuffer,
    attention_projected: GpuBuffer,
    gate: GpuBuffer,
    up: GpuBuffer,
    down: GpuBuffer,
    context_capacity: usize,
}

impl ForwardScratch {
    fn allocate(
        context: &MetalContext,
        config: &crate::dflash::DFlashConfig,
        context_capacity: usize,
    ) -> Result<Self, MetalError> {
        let batch = config.block_size;
        let hidden = config.hidden_size;
        let sampled = config.dflash_config.target_layer_ids.len();
        let q = config.num_attention_heads * config.head_dim;
        let kv = config.num_key_value_heads * config.head_dim;
        let intermediate = config.intermediate_size;
        Ok(Self {
            target_input: GpuBuffer::zeros(context, context_capacity * sampled * hidden)?,
            target_projected: GpuBuffer::zeros(context, context_capacity * hidden)?,
            hidden: GpuBuffer::zeros(context, batch * hidden)?,
            normed: GpuBuffer::zeros(context, batch * hidden)?,
            q: GpuBuffer::zeros(context, batch * q)?,
            k_noise: GpuBuffer::zeros(context, batch * kv)?,
            v_noise: GpuBuffer::zeros(context, batch * kv)?,
            k_context: GpuBuffer::zeros(context, context_capacity * kv)?,
            v_context: GpuBuffer::zeros(context, context_capacity * kv)?,
            attention: GpuBuffer::zeros(context, batch * q)?,
            attention_projected: GpuBuffer::zeros(context, batch * hidden)?,
            gate: GpuBuffer::zeros(context, batch * intermediate)?,
            up: GpuBuffer::zeros(context, batch * intermediate)?,
            down: GpuBuffer::zeros(context, batch * hidden)?,
            context_capacity,
        })
    }
}

/// The complete Ferrite DFlash GPU forward. All five layers are encoded into
/// one command buffer; the only host wait is the transactional round boundary.
pub struct MetalDFlashForward {
    context: MetalContext,
    kernels: MetalKernels,
    config: crate::dflash::DFlashConfig,
    mapped_weights: Option<GpuBytes>,
    projections: BTreeMap<String, Projection>,
    fc: Projection,
    hidden_norm: GpuBuffer,
    output_norm: GpuBuffer,
    layers: Vec<MetalLayer>,
    rope_frequencies: GpuBuffer,
    scratch: ForwardScratch,
    round_k: Vec<GpuBuffer>,
    round_v: Vec<GpuBuffer>,
    cached_k: Vec<GpuBuffer>,
    cached_v: Vec<GpuBuffer>,
    round_capacity: usize,
    cache_capacity: usize,
    synchronized_identity: Option<u64>,
    synchronized_revision: Option<u64>,
    prepared_generation: u64,
    active_prepared: Option<PreparedContextStamp>,
    prompt_capture_slots: Vec<GpuBuffer>,
    pending_prompt_chunk: Option<PendingPromptChunk>,
    prompt_pipeline_active: bool,
    prompt_pipeline_stats: DFlashPromptPipelineStats,
    max_context: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PreparedContextStamp {
    generation: u64,
    cache_identity: u64,
    cache_revision: u64,
    ctx_len: usize,
    ctx_offset: usize,
    sink_size: usize,
    window_size: usize,
    rows: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct DFlashPromptPipelineStats {
    pub(crate) chunks: usize,
    pub(crate) assistant_gpu_ns: u64,
    pub(crate) exposed_wait_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PromptCacheStamp {
    identity: u64,
    revision: u64,
    ctx_len: usize,
    ctx_offset: usize,
}

struct PendingPromptChunk {
    command: metal::CommandBuffer,
    cache: PromptCacheStamp,
    rows: usize,
    slot: usize,
}

/// Single-use capability for target-side DFlash context work completed before
/// the verifier publishes its exact committed prefix and next frontier.
///
/// The fields deliberately remain private. A handle is valid only for the
/// `MetalDFlashForward` and cache cut which created it; normal forwards and
/// newer preparations invalidate it.
#[derive(Debug)]
pub(crate) struct PreparedMetalDFlashContext {
    stamp: PreparedContextStamp,
}

/// Exact intermediate buffers from the target-feature side of DFlash.
///
/// This is a qualification surface, not a serving API.  Each field is read
/// after its own completed Metal command buffer so a cross-vendor comparator
/// can identify the first arithmetic boundary without running the assistant
/// layers or mutating a DFlash cache.
#[derive(Debug)]
pub struct DFlashContextBoundaryProbe {
    pub fc_out: Vec<f32>,
    pub enc_norm_out: Vec<f32>,
    pub k_projected_layer0: Vec<f32>,
    pub v_projected_layer0: Vec<f32>,
    pub k_normed_layer0: Vec<f32>,
    pub k_rope_layer0: Vec<f32>,
}

impl MetalDFlashForward {
    pub fn new(weights: &DFlashWeights, max_context: usize) -> Result<Self, MetalError> {
        let context = MetalContext::new()?;
        let kernels = MetalKernels::new(&context)?;
        let config = weights.config.clone();
        let hidden = config.hidden_size;
        let q = config.num_attention_heads * config.head_dim;
        let kv = config.num_key_value_heads * config.head_dim;
        let intermediate = config.intermediate_size;
        let sampled = config.dflash_config.target_layer_ids.len();
        let mapped_weights = weights
            .gguf_weights
            .as_ref()
            .map(|mapped| GpuBytes::from_mmap(&context, mapped.mapped_file()))
            .transpose()?;
        let mut projections = BTreeMap::new();
        let mut layers = Vec::with_capacity(weights.layers.len());
        for (layer, values) in weights.layers.iter().enumerate() {
            for (suffix, gguf_suffix, data, input_width, output_width) in [
                (
                    "q_proj",
                    "attn_q",
                    values.q_proj_weight.as_slice(),
                    hidden,
                    q,
                ),
                (
                    "k_proj",
                    "attn_k",
                    values.k_proj_weight.as_slice(),
                    hidden,
                    kv,
                ),
                (
                    "v_proj",
                    "attn_v",
                    values.v_proj_weight.as_slice(),
                    hidden,
                    kv,
                ),
                (
                    "o_proj",
                    "attn_output",
                    values.o_proj_weight.as_slice(),
                    q,
                    hidden,
                ),
                (
                    "gate_proj",
                    "ffn_gate",
                    values.gate_proj_weight.as_slice(),
                    hidden,
                    intermediate,
                ),
                (
                    "up_proj",
                    "ffn_up",
                    values.up_proj_weight.as_slice(),
                    hidden,
                    intermediate,
                ),
                (
                    "down_proj",
                    "ffn_down",
                    values.down_proj_weight.as_slice(),
                    intermediate,
                    hidden,
                ),
            ] {
                let gguf_name = format!("blk.{layer}.{gguf_suffix}.weight");
                projections.insert(
                    format!("layers.{layer}.{suffix}"),
                    Self::load_projection(
                        &context,
                        weights,
                        &gguf_name,
                        data,
                        input_width,
                        output_width,
                    )?,
                );
            }
            layers.push(MetalLayer {
                input_norm: GpuBuffer::from_f32(&context, &values.input_layernorm_weight)?,
                post_attention_norm: GpuBuffer::from_f32(
                    &context,
                    &values.post_attention_layernorm_weight,
                )?,
                q_norm: GpuBuffer::from_f32(&context, &values.q_norm_weight)?,
                k_norm: GpuBuffer::from_f32(&context, &values.k_norm_weight)?,
            });
        }
        let rope_values = if std::env::var_os("MUSER_CROSS_VENDOR_QK").is_some() {
            crate::rope_nco::canonical_rope_table(
                max_context,
                config.head_dim,
                config.rope_theta as f32,
            )
        } else {
            (0..config.head_dim / 2)
                .map(|index| {
                    1.0f32
                        / (config.rope_theta as f32)
                            .powf(2.0 * index as f32 / config.head_dim as f32)
                })
                .collect::<Vec<_>>()
        };
        let rope_frequencies = GpuBuffer::from_f32(&context, &rope_values)?;
        // Most rounds contain at most one verification block. The arena grows
        // geometrically on the first long-prompt round instead of reserving a
        // multi-gigabyte 131K activation slab at model load.
        let initial_context_capacity = config.block_size.min(max_context).max(1);
        let scratch = ForwardScratch::allocate(&context, &config, initial_context_capacity)?;
        let (round_k, round_v) = Self::allocate_layer_buffers(
            &context,
            config.num_hidden_layers,
            initial_context_capacity * kv,
        )?;
        let fc = Self::load_projection(
            &context,
            weights,
            "fc.weight",
            &weights.fc_weight,
            sampled * hidden,
            hidden,
        )?;
        Ok(Self {
            fc,
            hidden_norm: GpuBuffer::from_f32(&context, &weights.hidden_norm_weight)?,
            output_norm: GpuBuffer::from_f32(&context, &weights.norm_weight)?,
            context,
            kernels,
            config,
            mapped_weights,
            projections,
            layers,
            rope_frequencies,
            scratch,
            round_k,
            round_v,
            cached_k: Vec::new(),
            cached_v: Vec::new(),
            round_capacity: initial_context_capacity,
            cache_capacity: 0,
            synchronized_identity: None,
            synchronized_revision: None,
            prepared_generation: 0,
            active_prepared: None,
            prompt_capture_slots: Vec::new(),
            pending_prompt_chunk: None,
            prompt_pipeline_active: false,
            prompt_pipeline_stats: DFlashPromptPipelineStats::default(),
            max_context,
        })
    }

    fn load_projection(
        context: &MetalContext,
        weights: &DFlashWeights,
        gguf_name: &str,
        dense: &[f32],
        input_width: usize,
        output_width: usize,
    ) -> Result<Projection, MetalError> {
        let mapped = weights.gguf_weights.as_ref().and_then(|mapped| {
            let layout = mapped.layout(gguf_name).ok()?;
            matches!(
                layout.dtype,
                GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K
            )
            .then_some(layout)
        });
        let weight = match mapped {
            Some(layout) => {
                debug_assert_eq!((layout.n_in, layout.n_out), (input_width, output_width));
                ProjectionWeight::Mapped(layout)
            }
            None => ProjectionWeight::Dense(GpuBuffer::from_f32(context, dense)?),
        };
        Ok(Projection {
            weight,
            input_width,
            output_width,
        })
    }

    fn encode_projection(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        projection: &Projection,
        input: &GpuBuffer,
        output: &GpuBuffer,
        batch: usize,
    ) {
        match &projection.weight {
            ProjectionWeight::Dense(weight) => self.kernels.encode_dense_f32_batch(
                encoder,
                weight,
                input,
                output,
                projection.output_width,
                projection.input_width,
                batch,
            ),
            ProjectionWeight::Mapped(layout) => {
                let mapped = self
                    .mapped_weights
                    .as_ref()
                    .expect("mapped DFlash projection requires mapped GGUF storage");
                let view = mapped
                    .view(layout.file_offset, layout.byte_len)
                    .expect("validated DFlash projection lies inside mapped GGUF");
                self.kernels.encode_quantized_matmul(
                    encoder,
                    view,
                    input,
                    output,
                    layout.dtype,
                    projection.input_width,
                    projection.output_width,
                    batch,
                );
            }
        }
    }

    fn allocate_layer_buffers(
        context: &MetalContext,
        layers: usize,
        elements_per_layer: usize,
    ) -> Result<(Vec<GpuBuffer>, Vec<GpuBuffer>), MetalError> {
        let mut key = Vec::with_capacity(layers);
        let mut value = Vec::with_capacity(layers);
        for _ in 0..layers {
            key.push(GpuBuffer::zeros(context, elements_per_layer)?);
            value.push(GpuBuffer::zeros(context, elements_per_layer)?);
        }
        Ok((key, value))
    }

    fn ensure_round_capacity(&mut self, needed: usize) -> Result<(), MetalError> {
        if needed <= self.round_capacity {
            // After a prompt-sized round the arena keeps multi-GB slabs grown
            // for prefill; decode-sized rounds (<= block_size rows) would keep
            // them resident, and on unified memory they page in/out per round
            // inside the draft's blocking wait. Reallocate down to the
            // constructor's floor so the decode loop only touches small slabs.
            let floor = self.config.block_size.min(self.max_context).max(1);
            if needed <= floor && self.round_capacity > floor {
                self.scratch = ForwardScratch::allocate(&self.context, &self.config, floor)?;
                let kv = self.config.num_key_value_heads * self.config.head_dim;
                (self.round_k, self.round_v) = Self::allocate_layer_buffers(
                    &self.context,
                    self.config.num_hidden_layers,
                    floor * kv,
                )?;
                self.round_capacity = floor;
            }
            return Ok(());
        }
        let grown = needed.next_power_of_two().min(self.max_context).max(needed);
        self.scratch = ForwardScratch::allocate(&self.context, &self.config, grown)?;
        let kv = self.config.num_key_value_heads * self.config.head_dim;
        (self.round_k, self.round_v) =
            Self::allocate_layer_buffers(&self.context, self.config.num_hidden_layers, grown * kv)?;
        self.round_capacity = grown;
        Ok(())
    }

    fn ensure_cache_capacity(&mut self, needed: usize) -> Result<(), MetalError> {
        if needed <= self.cache_capacity {
            return Ok(());
        }
        let kv = self.config.num_key_value_heads * self.config.head_dim;
        (self.cached_k, self.cached_v) = Self::allocate_layer_buffers(
            &self.context,
            self.config.num_hidden_layers,
            needed * kv,
        )?;
        self.cache_capacity = needed;
        self.synchronized_identity = None;
        self.synchronized_revision = None;
        Ok(())
    }

    fn synchronize_cache(&mut self, cache: &DFlashContextKvCache) {
        let elements = cache.layout().elements(cache.ctx_len);
        for layer in 0..self.layers.len() {
            self.cached_k[layer].as_mut_slice()[..elements].copy_from_slice(cache.layer_k(layer));
            self.cached_v[layer].as_mut_slice()[..elements].copy_from_slice(cache.layer_v(layer));
        }
        self.synchronized_identity = Some(cache.identity());
        self.synchronized_revision = Some(cache.revision());
    }

    fn append_cached_rows(
        destination: &mut GpuBuffer,
        source: &GpuBuffer,
        width: usize,
        current: usize,
        sink: usize,
        window: usize,
        added: usize,
    ) {
        let total = current + added;
        let maximum = sink + window;
        let destination_values = destination.as_mut_slice();
        let source_values = source.as_slice();
        if total <= maximum {
            destination_values[current * width..total * width]
                .copy_from_slice(&source_values[..added * width]);
            return;
        }

        let sink_rows = sink.min(total);
        let sink_from_current = current.min(sink_rows);
        let sink_from_source = sink_rows - sink_from_current;
        let tail_rows = window.min(total - sink_rows);

        if added >= tail_rows {
            let source_start = added - tail_rows;
            destination_values[sink_rows * width..(sink_rows + tail_rows) * width]
                .copy_from_slice(&source_values[source_start * width..added * width]);
        } else {
            let prior_rows = tail_rows - added;
            let prior_start = current - prior_rows;
            destination_values.copy_within(prior_start * width..current * width, sink_rows * width);
            destination_values[(sink_rows + prior_rows) * width..(sink_rows + tail_rows) * width]
                .copy_from_slice(&source_values[..added * width]);
        }
        if sink_from_source > 0 {
            destination_values[sink_from_current * width..sink_rows * width]
                .copy_from_slice(&source_values[..sink_from_source * width]);
        }
    }

    fn commit_round(&mut self, cache: &mut DFlashContextKvCache, n_context: usize) {
        let width = cache.layout().width();
        let current = cache.ctx_len;
        for layer in 0..self.layers.len() {
            Self::append_cached_rows(
                &mut self.cached_k[layer],
                &self.round_k[layer],
                width,
                current,
                cache.sink_size,
                cache.window_size,
                n_context,
            );
            Self::append_cached_rows(
                &mut self.cached_v[layer],
                &self.round_v[layer],
                width,
                current,
                cache.sink_size,
                cache.window_size,
                n_context,
            );
            cache.append_layer(
                layer,
                self.round_k[layer].as_slice(),
                self.round_v[layer].as_slice(),
                n_context,
            );
        }
        cache.advance_round(n_context);
        self.synchronized_identity = Some(cache.identity());
        self.synchronized_revision = Some(cache.revision());
    }

    pub(crate) fn begin_prompt_pipeline(
        &mut self,
        cache: &DFlashContextKvCache,
        maximum_chunk_rows: usize,
    ) -> Result<(), DFlashError> {
        if maximum_chunk_rows == 0 || maximum_chunk_rows > self.max_context {
            return Err(DFlashError::Projection(format!(
                "DFlash prompt chunk {maximum_chunk_rows} outside 1..={}",
                self.max_context
            )));
        }
        if self.prompt_pipeline_active || self.pending_prompt_chunk.is_some() {
            return Err(DFlashError::Projection(
                "a DFlash prompt pipeline is already active".into(),
            ));
        }
        if cache.ctx_len != 0 || cache.ctx_offset != 0 {
            return Err(DFlashError::Projection(
                "DFlash prompt pipeline requires a fresh context cache".into(),
            ));
        }
        self.active_prepared = None;
        self.ensure_round_capacity(maximum_chunk_rows)
            .map_err(|error| DFlashError::Projection(error.to_string()))?;
        self.ensure_cache_capacity(cache.physical_capacity())
            .map_err(|error| DFlashError::Projection(error.to_string()))?;
        if !self.cache_is_synchronized(cache) {
            self.synchronize_cache(cache);
        }
        let elements = maximum_chunk_rows
            .checked_mul(self.config.dflash_config.target_layer_ids.len())
            .and_then(|value| value.checked_mul(self.config.hidden_size))
            .ok_or_else(|| DFlashError::Projection("DFlash prompt capture size overflow".into()))?;
        self.prompt_capture_slots.clear();
        for _ in 0..2 {
            self.prompt_capture_slots.push(
                GpuBuffer::zeros(&self.context, elements)
                    .map_err(|error| DFlashError::Projection(error.to_string()))?,
            );
        }
        self.prompt_pipeline_stats = DFlashPromptPipelineStats::default();
        self.prompt_pipeline_active = true;
        Ok(())
    }

    pub(crate) fn prompt_capture_slot(
        &mut self,
        cache: &mut DFlashContextKvCache,
        slot: usize,
        source_rows: usize,
    ) -> Result<GpuBuffer, DFlashError> {
        if !self.prompt_pipeline_active || source_rows == 0 {
            return Err(DFlashError::Projection(
                "DFlash prompt capture requested outside an active pipeline".into(),
            ));
        }
        if self
            .pending_prompt_chunk
            .as_ref()
            .is_some_and(|pending| pending.slot == slot)
        {
            self.finish_pending_prompt_chunk(cache)?;
        }
        let Some(buffer) = self.prompt_capture_slots.get(slot) else {
            return Err(DFlashError::Projection(format!(
                "DFlash prompt capture slot {slot} is invalid"
            )));
        };
        let needed = source_rows
            .checked_mul(self.config.dflash_config.target_layer_ids.len())
            .and_then(|value| value.checked_mul(self.config.hidden_size))
            .ok_or_else(|| DFlashError::Projection("DFlash prompt capture size overflow".into()))?;
        if needed > buffer.len() {
            return Err(DFlashError::Projection(format!(
                "DFlash prompt capture needs {needed} values, slot holds {}",
                buffer.len()
            )));
        }
        Ok(buffer.clone())
    }

    fn finish_pending_prompt_chunk(
        &mut self,
        cache: &mut DFlashContextKvCache,
    ) -> Result<(), DFlashError> {
        let Some(pending) = self.pending_prompt_chunk.take() else {
            return Ok(());
        };
        let wait_started = Instant::now();
        self.context
            .wait_for_completion(&pending.command, Duration::from_secs(300))
            .map_err(|error| DFlashError::Projection(error.to_string()))?;
        self.prompt_pipeline_stats.exposed_wait_ns = self
            .prompt_pipeline_stats
            .exposed_wait_ns
            .saturating_add(wait_started.elapsed().as_nanos().min(u64::MAX as u128) as u64);
        let expected = pending.cache;
        if cache.identity() != expected.identity
            || cache.revision() != expected.revision
            || cache.ctx_len != expected.ctx_len
            || cache.ctx_offset != expected.ctx_offset
        {
            self.synchronized_identity = None;
            self.synchronized_revision = None;
            return Err(DFlashError::Projection(
                "DFlash prompt cache changed while a chunk was in flight".into(),
            ));
        }
        let gpu_ns = unsafe {
            let start: f64 = objc::msg_send![pending.command, GPUStartTime];
            let end: f64 = objc::msg_send![pending.command, GPUEndTime];
            ((end - start).max(0.0) * 1.0e9).min(u64::MAX as f64) as u64
        };
        self.prompt_pipeline_stats.assistant_gpu_ns = self
            .prompt_pipeline_stats
            .assistant_gpu_ns
            .saturating_add(gpu_ns);
        self.prompt_pipeline_stats.chunks += 1;
        self.commit_round(cache, pending.rows);
        Ok(())
    }

    pub(crate) fn advance_prompt_pipeline_to(
        &mut self,
        cache: &mut DFlashContextKvCache,
        absolute_position: usize,
    ) -> Result<(), DFlashError> {
        if !self.prompt_pipeline_active || absolute_position < cache.ctx_offset {
            return Err(DFlashError::Projection(
                "DFlash prompt pipeline position regressed".into(),
            ));
        }
        self.finish_pending_prompt_chunk(cache)?;
        if absolute_position == cache.ctx_offset {
            return Ok(());
        }
        cache
            .advance_prompt_gap(absolute_position)
            .map_err(DFlashError::Projection)?;
        self.synchronized_identity = Some(cache.identity());
        self.synchronized_revision = Some(cache.revision());
        Ok(())
    }

    pub(crate) fn enqueue_prompt_chunk(
        &mut self,
        cache: &mut DFlashContextKvCache,
        slot: usize,
        source_rows: usize,
        source_start: usize,
        output_rows: usize,
    ) -> Result<(), DFlashError> {
        if !self.prompt_pipeline_active
            || output_rows == 0
            || source_start
                .checked_add(output_rows)
                .is_none_or(|end| end > source_rows)
            || slot >= self.prompt_capture_slots.len()
        {
            return Err(DFlashError::Projection(
                "DFlash prompt enqueue geometry is invalid".into(),
            ));
        }
        // One assistant chunk remains in flight while the target executes the
        // next chunk. Reap it only at this handoff, then reuse the compact K/V
        // work arena for the newly completed target capture.
        self.finish_pending_prompt_chunk(cache)?;
        if !self.cache_is_synchronized(cache) {
            return Err(DFlashError::Projection(
                "DFlash prompt accelerator mirror lost synchronization".into(),
            ));
        }
        let sampled = self.config.dflash_config.target_layer_ids.len();
        let hidden = self.config.hidden_size;
        let kv_width = self.config.num_key_value_heads * self.config.head_dim;
        let input = &self.prompt_capture_slots[slot];
        let command = self.context.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        self.kernels.encode_pack_dflash_layer_major(
            encoder,
            input,
            &self.scratch.target_input,
            crate::metal::encode::DFlashPackGeometry {
                source_tokens: source_rows,
                source_start,
                output_tokens: output_rows,
                layers: sampled,
                hidden,
            },
        );
        self.encode_projection(
            encoder,
            &self.fc,
            &self.scratch.target_input,
            &self.scratch.target_projected,
            output_rows,
        );
        self.kernels.encode_rms_norm_inplace(
            encoder,
            &self.scratch.target_projected,
            &self.hidden_norm,
            hidden,
            self.config.rms_norm_eps as f32,
            output_rows,
        );
        for layer in 0..self.layers.len() {
            let aux = &self.layers[layer];
            self.encode_projection(
                encoder,
                self.projection(layer, "k_proj"),
                &self.scratch.target_projected,
                &self.round_k[layer],
                output_rows,
            );
            self.encode_projection(
                encoder,
                self.projection(layer, "v_proj"),
                &self.scratch.target_projected,
                &self.round_v[layer],
                output_rows,
            );
            self.kernels.encode_rms_norm_per_head(
                encoder,
                &self.round_k[layer],
                &aux.k_norm,
                &self.round_k[layer],
                output_rows * self.config.num_key_value_heads,
                self.config.head_dim,
                self.config.rms_norm_eps as f32,
            );
            self.kernels.encode_rope_neox_batch_cached(
                encoder,
                &self.round_k[layer],
                &self.round_k[layer],
                &self.rope_frequencies,
                self.config.num_key_value_heads,
                0,
                self.config.head_dim,
                cache.ctx_offset,
                output_rows,
            );
            debug_assert!(self.round_k[layer].len() >= output_rows * kv_width);
            debug_assert!(self.round_v[layer].len() >= output_rows * kv_width);
        }
        encoder.end_encoding();
        command.commit();
        self.pending_prompt_chunk = Some(PendingPromptChunk {
            command: command.to_owned(),
            cache: PromptCacheStamp {
                identity: cache.identity(),
                revision: cache.revision(),
                ctx_len: cache.ctx_len,
                ctx_offset: cache.ctx_offset,
            },
            rows: output_rows,
            slot,
        });
        Ok(())
    }

    pub(crate) fn finish_prompt_pipeline(
        &mut self,
        cache: &mut DFlashContextKvCache,
    ) -> Result<DFlashPromptPipelineStats, DFlashError> {
        if !self.prompt_pipeline_active {
            return Err(DFlashError::Projection(
                "DFlash prompt pipeline is not active".into(),
            ));
        }
        let result = self.finish_pending_prompt_chunk(cache);
        self.prompt_pipeline_active = false;
        self.prompt_capture_slots.clear();
        if result.is_err() {
            self.synchronized_identity = None;
            self.synchronized_revision = None;
        }
        result.map(|()| self.prompt_pipeline_stats)
    }

    pub(crate) fn abort_prompt_pipeline(&mut self) -> Result<(), DFlashError> {
        let pending = self.pending_prompt_chunk.take();
        self.prompt_pipeline_active = false;
        self.prompt_capture_slots.clear();
        self.synchronized_identity = None;
        self.synchronized_revision = None;
        if let Some(pending) = pending {
            self.context
                .wait_for_completion(&pending.command, Duration::from_secs(300))
                .map_err(|error| DFlashError::Projection(error.to_string()))?;
        }
        Ok(())
    }

    fn projection(&self, layer: usize, suffix: &str) -> &Projection {
        &self.projections[&format!("layers.{layer}.{suffix}")]
    }

    fn validate_target_context_geometry(
        &self,
        target_hidden: &[f32],
        n_context: usize,
    ) -> Result<(), DFlashError> {
        if n_context == 0 || n_context > self.max_context {
            return Err(DFlashError::Projection(format!(
                "Metal DFlash context {n_context} outside 1..={}",
                self.max_context
            )));
        }
        let expected = n_context
            .checked_mul(self.config.dflash_config.target_layer_ids.len())
            .and_then(|elements| elements.checked_mul(self.config.hidden_size))
            .ok_or_else(|| DFlashError::Projection("Metal DFlash input size overflow".into()))?;
        if target_hidden.len() != expected {
            return Err(DFlashError::Projection(format!(
                "Metal DFlash target context has {} values, expected {expected}",
                target_hidden.len()
            )));
        }
        Ok(())
    }

    fn cache_is_synchronized(&self, cache: &DFlashContextKvCache) -> bool {
        self.synchronized_identity == Some(cache.identity())
            && self.synchronized_revision == Some(cache.revision())
    }

    fn cache_matches_prepared(cache: &DFlashContextKvCache, stamp: PreparedContextStamp) -> bool {
        cache.identity() == stamp.cache_identity
            && cache.revision() == stamp.cache_revision
            && cache.ctx_len == stamp.ctx_len
            && cache.ctx_offset == stamp.ctx_offset
            && cache.sink_size == stamp.sink_size
            && cache.window_size == stamp.window_size
    }

    /// Complete the seed-independent side of one DFlash round. The returned
    /// handle owns no buffers: it is a single-use capability tied to the
    /// retained per-forward arena and the exact cache cut used here.
    pub(crate) fn prepare_target_context(
        &mut self,
        target_hidden: &[f32],
        n_context: usize,
        cache: &DFlashContextKvCache,
    ) -> Result<PreparedMetalDFlashContext, DFlashError> {
        if self.prompt_pipeline_active || self.pending_prompt_chunk.is_some() {
            return Err(DFlashError::Projection(
                "prepared DFlash context overlaps an active prompt pipeline".into(),
            ));
        }
        self.validate_target_context_geometry(target_hidden, n_context)?;
        // A newer successful preparation supersedes any earlier capability.
        // Invalidate before touching the shared arena, but only after input
        // validation so a malformed speculative packet cannot destroy a valid
        // preparation which the caller still owns.
        self.active_prepared = None;
        self.ensure_round_capacity(n_context)
            .map_err(|error| DFlashError::Projection(error.to_string()))?;
        self.ensure_cache_capacity(cache.physical_capacity())
            .map_err(|error| DFlashError::Projection(error.to_string()))?;
        if !self.cache_is_synchronized(cache) {
            self.synchronize_cache(cache);
        }
        self.scratch.target_input.as_mut_slice()[..target_hidden.len()]
            .copy_from_slice(target_hidden);

        let hidden = self.config.hidden_size;
        let c = &self.config;
        let command = self.context.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        self.encode_projection(
            encoder,
            &self.fc,
            &self.scratch.target_input,
            &self.scratch.target_projected,
            n_context,
        );
        self.kernels.encode_rms_norm_inplace(
            encoder,
            &self.scratch.target_projected,
            &self.hidden_norm,
            hidden,
            c.rms_norm_eps as f32,
            n_context,
        );
        for layer in 0..self.layers.len() {
            let aux = &self.layers[layer];
            self.encode_projection(
                encoder,
                self.projection(layer, "k_proj"),
                &self.scratch.target_projected,
                &self.round_k[layer],
                n_context,
            );
            self.encode_projection(
                encoder,
                self.projection(layer, "v_proj"),
                &self.scratch.target_projected,
                &self.round_v[layer],
                n_context,
            );
            self.kernels.encode_rms_norm_per_head(
                encoder,
                &self.round_k[layer],
                &aux.k_norm,
                &self.round_k[layer],
                n_context * c.num_key_value_heads,
                c.head_dim,
                c.rms_norm_eps as f32,
            );
            self.kernels.encode_rope_neox_batch_cached(
                encoder,
                &self.round_k[layer],
                &self.round_k[layer],
                &self.rope_frequencies,
                c.num_key_value_heads,
                0,
                c.head_dim,
                cache.ctx_offset,
                n_context,
            );
        }
        encoder.end_encoding();
        command.commit();
        self.context
            .wait_for_completion(command, Duration::from_secs(120))
            .map_err(|error| DFlashError::Projection(error.to_string()))?;

        let generation = self.prepared_generation.checked_add(1).ok_or_else(|| {
            DFlashError::Projection("Metal DFlash prepared-handle generation exhausted".into())
        })?;
        self.prepared_generation = generation;
        let stamp = PreparedContextStamp {
            generation,
            cache_identity: cache.identity(),
            cache_revision: cache.revision(),
            ctx_len: cache.ctx_len,
            ctx_offset: cache.ctx_offset,
            sink_size: cache.sink_size,
            window_size: cache.window_size,
            rows: n_context,
        };
        self.active_prepared = Some(stamp);
        Ok(PreparedMetalDFlashContext { stamp })
    }

    /// Finish one prepared round after the verifier selects an exact prefix.
    /// Only that prefix participates in attention and is appended to the
    /// authoritative DFlash cache; prepared rejected rows remain unobservable.
    pub(crate) fn finish_prepared(
        &mut self,
        prepared: PreparedMetalDFlashContext,
        noise_embedding: &[f32],
        selected_context_rows: usize,
        cache: &mut DFlashContextKvCache,
    ) -> Result<DFlashDraftOutput, DFlashError> {
        let Some(active) = self.active_prepared else {
            return Err(DFlashError::Projection(
                "Metal DFlash prepared handle is no longer active".into(),
            ));
        };
        if active != prepared.stamp {
            return Err(DFlashError::Projection(
                "Metal DFlash prepared handle was superseded".into(),
            ));
        }
        let batch = self.config.block_size;
        let hidden = self.config.hidden_size;
        let expected_noise = batch.checked_mul(hidden).ok_or_else(|| {
            DFlashError::Projection("Metal DFlash noise input size overflow".into())
        })?;
        if noise_embedding.len() != expected_noise
            || selected_context_rows == 0
            || selected_context_rows > active.rows
            || !Self::cache_matches_prepared(cache, active)
            || !self.cache_is_synchronized(cache)
        {
            // The capability was consumed, so any validation failure must fail
            // closed rather than leaving an unreachable active preparation.
            self.active_prepared = None;
            return Err(DFlashError::Projection(format!(
                "Metal DFlash prepared finish mismatch (noise={}, selected={selected_context_rows}/{}, cache_identity={}, cache_revision={})",
                noise_embedding.len(),
                active.rows,
                cache.identity(),
                cache.revision(),
            )));
        }
        self.active_prepared = None;
        self.scratch.hidden.as_mut_slice()[..noise_embedding.len()]
            .copy_from_slice(noise_embedding);

        let c = &self.config;
        let intermediate = c.intermediate_size;
        let command = self.context.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        for layer in 0..self.layers.len() {
            let aux = &self.layers[layer];
            self.kernels.encode_rms_norm_mul(
                encoder,
                &self.scratch.hidden,
                &aux.input_norm,
                &self.scratch.normed,
                hidden,
                c.rms_norm_eps as f32,
                batch,
            );
            for (suffix, input, output) in [
                ("q_proj", &self.scratch.normed, &self.scratch.q),
                ("k_proj", &self.scratch.normed, &self.scratch.k_noise),
                ("v_proj", &self.scratch.normed, &self.scratch.v_noise),
            ] {
                self.encode_projection(
                    encoder,
                    self.projection(layer, suffix),
                    input,
                    output,
                    batch,
                );
            }
            self.kernels.encode_rms_norm_per_head(
                encoder,
                &self.scratch.q,
                &aux.q_norm,
                &self.scratch.q,
                batch * c.num_attention_heads,
                c.head_dim,
                c.rms_norm_eps as f32,
            );
            self.kernels.encode_rms_norm_per_head(
                encoder,
                &self.scratch.k_noise,
                &aux.k_norm,
                &self.scratch.k_noise,
                batch * c.num_key_value_heads,
                c.head_dim,
                c.rms_norm_eps as f32,
            );
            for (values, heads) in [
                (&self.scratch.k_noise, c.num_key_value_heads),
                (&self.scratch.q, c.num_attention_heads),
            ] {
                self.kernels.encode_rope_neox_batch_cached(
                    encoder,
                    values,
                    values,
                    &self.rope_frequencies,
                    heads,
                    0,
                    c.head_dim,
                    cache.ctx_offset + selected_context_rows,
                    batch,
                );
            }
            self.kernels.encode_dflash_dual_attention(
                encoder,
                &self.scratch.q,
                &self.cached_k[layer],
                &self.cached_v[layer],
                &self.round_k[layer],
                &self.round_v[layer],
                &self.scratch.k_noise,
                &self.scratch.v_noise,
                &self.scratch.attention,
                c.head_dim,
                cache.ctx_len,
                selected_context_rows,
                c.num_attention_heads,
                c.num_key_value_heads,
                batch,
            );
            self.encode_projection(
                encoder,
                self.projection(layer, "o_proj"),
                &self.scratch.attention,
                &self.scratch.attention_projected,
                batch,
            );
            self.kernels.encode_fused_residual_norm(
                encoder,
                &self.scratch.hidden,
                &self.scratch.attention_projected,
                &aux.post_attention_norm,
                &self.scratch.normed,
                hidden,
                c.rms_norm_eps as f32,
                batch,
            );
            for (suffix, output) in [
                ("gate_proj", &self.scratch.gate),
                ("up_proj", &self.scratch.up),
            ] {
                self.encode_projection(
                    encoder,
                    self.projection(layer, suffix),
                    &self.scratch.normed,
                    output,
                    batch,
                );
            }
            self.kernels.encode_silu_hadamard_batch(
                encoder,
                &self.scratch.gate,
                &self.scratch.up,
                batch * intermediate,
            );
            self.encode_projection(
                encoder,
                self.projection(layer, "down_proj"),
                &self.scratch.gate,
                &self.scratch.down,
                batch,
            );
            self.kernels.encode_residual_add_batch(
                encoder,
                &self.scratch.hidden,
                &self.scratch.down,
                batch * hidden,
            );
        }
        self.kernels.encode_rms_norm_inplace(
            encoder,
            &self.scratch.hidden,
            &self.output_norm,
            hidden,
            c.rms_norm_eps as f32,
            batch,
        );
        encoder.end_encoding();
        command.commit();
        self.context
            .wait_for_completion(command, Duration::from_secs(120))
            .map_err(|error| DFlashError::Projection(error.to_string()))?;
        self.commit_round(cache, selected_context_rows);
        let hidden_state = self.scratch.hidden.clone();
        Ok(DFlashDraftOutput {
            hidden_states: hidden_state.as_slice().to_vec(),
            n_draft_tokens: batch - 1,
            metal_hidden_states: Some(hidden_state),
        })
    }

    pub(crate) fn discard_prepared(
        &mut self,
        prepared: PreparedMetalDFlashContext,
    ) -> Result<(), DFlashError> {
        if self.active_prepared != Some(prepared.stamp) {
            return Err(DFlashError::Projection(
                "Metal DFlash prepared handle is no longer active".into(),
            ));
        }
        self.active_prepared = None;
        Ok(())
    }

    pub(crate) fn invalidate_prepared(&mut self) {
        self.active_prepared = None;
    }

    /// Run the minimal target-feature -> layer-0 K ladder used by the seam
    /// qualification harness.  The production `forward` remains one command
    /// buffer; this deliberately splits boundaries so host reads cannot alter
    /// the operation order being diagnosed.
    pub fn probe_context_boundaries(
        &mut self,
        target_hidden: &[f32],
        n_context: usize,
        start_position: usize,
    ) -> Result<DFlashContextBoundaryProbe, DFlashError> {
        let hidden = self.config.hidden_size;
        let sampled = self.config.dflash_config.target_layer_ids.len();
        let kv_width = self.config.num_key_value_heads * self.config.head_dim;
        if n_context == 0 || n_context > self.max_context {
            return Err(DFlashError::Projection(format!(
                "Metal DFlash probe context {n_context} outside 1..={}",
                self.max_context
            )));
        }
        if target_hidden.len() != n_context * sampled * hidden {
            return Err(DFlashError::Projection(format!(
                "Metal DFlash probe input has {} values, expected {}",
                target_hidden.len(),
                n_context * sampled * hidden
            )));
        }
        self.active_prepared = None;
        self.ensure_round_capacity(n_context)
            .map_err(|error| DFlashError::Projection(error.to_string()))?;
        self.scratch.target_input.as_mut_slice()[..target_hidden.len()]
            .copy_from_slice(target_hidden);

        let command = self.context.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        self.encode_projection(
            encoder,
            &self.fc,
            &self.scratch.target_input,
            &self.scratch.target_projected,
            n_context,
        );
        encoder.end_encoding();
        command.commit();
        self.context
            .wait_for_completion(command, Duration::from_secs(120))
            .map_err(|error| DFlashError::Projection(error.to_string()))?;
        let fc_out = self.scratch.target_projected.as_slice()[..n_context * hidden].to_vec();

        let command = self.context.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        self.kernels.encode_rms_norm_inplace(
            encoder,
            &self.scratch.target_projected,
            &self.hidden_norm,
            hidden,
            self.config.rms_norm_eps as f32,
            n_context,
        );
        encoder.end_encoding();
        command.commit();
        self.context
            .wait_for_completion(command, Duration::from_secs(120))
            .map_err(|error| DFlashError::Projection(error.to_string()))?;
        let enc_norm_out = self.scratch.target_projected.as_slice()[..n_context * hidden].to_vec();

        let command = self.context.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        self.encode_projection(
            encoder,
            self.projection(0, "k_proj"),
            &self.scratch.target_projected,
            &self.scratch.k_context,
            n_context,
        );
        self.encode_projection(
            encoder,
            self.projection(0, "v_proj"),
            &self.scratch.target_projected,
            &self.scratch.v_context,
            n_context,
        );
        encoder.end_encoding();
        command.commit();
        self.context
            .wait_for_completion(command, Duration::from_secs(120))
            .map_err(|error| DFlashError::Projection(error.to_string()))?;
        let k_projected_layer0 = self.scratch.k_context.as_slice()[..n_context * kv_width].to_vec();
        let v_projected_layer0 = self.scratch.v_context.as_slice()[..n_context * kv_width].to_vec();

        let command = self.context.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        self.kernels.encode_rms_norm_per_head(
            encoder,
            &self.scratch.k_context,
            &self.layers[0].k_norm,
            &self.scratch.k_context,
            n_context * self.config.num_key_value_heads,
            self.config.head_dim,
            self.config.rms_norm_eps as f32,
        );
        encoder.end_encoding();
        command.commit();
        self.context
            .wait_for_completion(command, Duration::from_secs(120))
            .map_err(|error| DFlashError::Projection(error.to_string()))?;
        let k_normed_layer0 = self.scratch.k_context.as_slice()[..n_context * kv_width].to_vec();

        let command = self.context.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        self.kernels.encode_rope_neox_batch_cached(
            encoder,
            &self.scratch.k_context,
            &self.scratch.k_context,
            &self.rope_frequencies,
            self.config.num_key_value_heads,
            0,
            self.config.head_dim,
            start_position,
            n_context,
        );
        encoder.end_encoding();
        command.commit();
        self.context
            .wait_for_completion(command, Duration::from_secs(120))
            .map_err(|error| DFlashError::Projection(error.to_string()))?;
        let k_rope_layer0 = self.scratch.k_context.as_slice()[..n_context * kv_width].to_vec();

        Ok(DFlashContextBoundaryProbe {
            fc_out,
            enc_norm_out,
            k_projected_layer0,
            v_projected_layer0,
            k_normed_layer0,
            k_rope_layer0,
        })
    }

    pub fn forward(
        &mut self,
        noise_embedding: &[f32],
        target_hidden: &[f32],
        n_context: usize,
        cache: &mut DFlashContextKvCache,
    ) -> Result<DFlashDraftOutput, DFlashError> {
        if self.prompt_pipeline_active || self.pending_prompt_chunk.is_some() {
            return Err(DFlashError::Projection(
                "DFlash forward overlaps an active prompt pipeline".into(),
            ));
        }
        let batch = self.config.block_size;
        let hidden = self.config.hidden_size;
        let sampled = self.config.dflash_config.target_layer_ids.len();
        let kv_width = self.config.num_key_value_heads * self.config.head_dim;
        let intermediate = self.config.intermediate_size;
        if n_context == 0 || n_context > self.max_context {
            return Err(DFlashError::Projection(format!(
                "Metal DFlash context {n_context} outside 1..={}",
                self.max_context
            )));
        }
        if noise_embedding.len() != batch * hidden
            || target_hidden.len() != n_context * sampled * hidden
        {
            return Err(DFlashError::Projection(
                "Metal DFlash input geometry mismatch".into(),
            ));
        }
        self.active_prepared = None;
        self.ensure_round_capacity(n_context)
            .map_err(|error| DFlashError::Projection(error.to_string()))?;
        debug_assert!(n_context <= self.scratch.context_capacity);
        self.ensure_cache_capacity(cache.sink_size + cache.window_size)
            .map_err(|error| DFlashError::Projection(error.to_string()))?;
        if !self.cache_is_synchronized(cache) {
            self.synchronize_cache(cache);
        }
        self.scratch.target_input.as_mut_slice()[..target_hidden.len()]
            .copy_from_slice(target_hidden);
        self.scratch.hidden.as_mut_slice()[..noise_embedding.len()]
            .copy_from_slice(noise_embedding);
        let c = &self.config;

        let command = self.context.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        self.encode_projection(
            encoder,
            &self.fc,
            &self.scratch.target_input,
            &self.scratch.target_projected,
            n_context,
        );
        self.kernels.encode_rms_norm_inplace(
            encoder,
            &self.scratch.target_projected,
            &self.hidden_norm,
            hidden,
            c.rms_norm_eps as f32,
            n_context,
        );
        for layer in 0..self.layers.len() {
            let aux = &self.layers[layer];
            self.kernels.encode_rms_norm_mul(
                encoder,
                &self.scratch.hidden,
                &aux.input_norm,
                &self.scratch.normed,
                hidden,
                c.rms_norm_eps as f32,
                batch,
            );
            for (suffix, input, output) in [
                ("q_proj", &self.scratch.normed, &self.scratch.q),
                ("k_proj", &self.scratch.normed, &self.scratch.k_noise),
                ("v_proj", &self.scratch.normed, &self.scratch.v_noise),
            ] {
                self.encode_projection(
                    encoder,
                    self.projection(layer, suffix),
                    input,
                    output,
                    batch,
                );
            }
            for (suffix, output) in [
                ("k_proj", &self.scratch.k_context),
                ("v_proj", &self.scratch.v_context),
            ] {
                self.encode_projection(
                    encoder,
                    self.projection(layer, suffix),
                    &self.scratch.target_projected,
                    output,
                    n_context,
                );
            }
            self.kernels.encode_rms_norm_per_head(
                encoder,
                &self.scratch.q,
                &aux.q_norm,
                &self.scratch.q,
                batch * c.num_attention_heads,
                c.head_dim,
                c.rms_norm_eps as f32,
            );
            for (values, heads) in [
                (&self.scratch.k_noise, batch * c.num_key_value_heads),
                (&self.scratch.k_context, n_context * c.num_key_value_heads),
            ] {
                self.kernels.encode_rms_norm_per_head(
                    encoder,
                    values,
                    &aux.k_norm,
                    values,
                    heads,
                    c.head_dim,
                    c.rms_norm_eps as f32,
                );
            }
            self.kernels.encode_rope_neox_batch_cached(
                encoder,
                &self.scratch.k_context,
                &self.scratch.k_context,
                &self.rope_frequencies,
                c.num_key_value_heads,
                0,
                c.head_dim,
                cache.ctx_offset,
                n_context,
            );
            self.kernels.encode_rope_neox_batch_cached(
                encoder,
                &self.scratch.k_noise,
                &self.scratch.k_noise,
                &self.rope_frequencies,
                c.num_key_value_heads,
                0,
                c.head_dim,
                cache.ctx_offset + n_context,
                batch,
            );
            self.kernels.encode_rope_neox_batch_cached(
                encoder,
                &self.scratch.q,
                &self.scratch.q,
                &self.rope_frequencies,
                c.num_attention_heads,
                0,
                c.head_dim,
                cache.ctx_offset + n_context,
                batch,
            );
            self.kernels.encode_dflash_dual_attention(
                encoder,
                &self.scratch.q,
                &self.cached_k[layer],
                &self.cached_v[layer],
                &self.scratch.k_context,
                &self.scratch.v_context,
                &self.scratch.k_noise,
                &self.scratch.v_noise,
                &self.scratch.attention,
                c.head_dim,
                cache.ctx_len,
                n_context,
                c.num_attention_heads,
                c.num_key_value_heads,
                batch,
            );
            self.kernels.encode_copy_f32(
                encoder,
                &self.scratch.k_context,
                &self.round_k[layer],
                n_context * kv_width,
            );
            self.kernels.encode_copy_f32(
                encoder,
                &self.scratch.v_context,
                &self.round_v[layer],
                n_context * kv_width,
            );
            self.encode_projection(
                encoder,
                self.projection(layer, "o_proj"),
                &self.scratch.attention,
                &self.scratch.attention_projected,
                batch,
            );
            self.kernels.encode_fused_residual_norm(
                encoder,
                &self.scratch.hidden,
                &self.scratch.attention_projected,
                &aux.post_attention_norm,
                &self.scratch.normed,
                hidden,
                c.rms_norm_eps as f32,
                batch,
            );
            for (suffix, output) in [
                ("gate_proj", &self.scratch.gate),
                ("up_proj", &self.scratch.up),
            ] {
                self.encode_projection(
                    encoder,
                    self.projection(layer, suffix),
                    &self.scratch.normed,
                    output,
                    batch,
                );
            }
            self.kernels.encode_silu_hadamard_batch(
                encoder,
                &self.scratch.gate,
                &self.scratch.up,
                batch * intermediate,
            );
            self.encode_projection(
                encoder,
                self.projection(layer, "down_proj"),
                &self.scratch.gate,
                &self.scratch.down,
                batch,
            );
            self.kernels.encode_residual_add_batch(
                encoder,
                &self.scratch.hidden,
                &self.scratch.down,
                batch * hidden,
            );
        }
        self.kernels.encode_rms_norm_inplace(
            encoder,
            &self.scratch.hidden,
            &self.output_norm,
            hidden,
            c.rms_norm_eps as f32,
            batch,
        );
        encoder.end_encoding();
        command.commit();
        self.context
            .wait_for_completion(command, Duration::from_secs(120))
            .map_err(|error| DFlashError::Projection(error.to_string()))?;
        self.commit_round(cache, n_context);
        let hidden_state = self.scratch.hidden.clone();
        Ok(DFlashDraftOutput {
            hidden_states: hidden_state.as_slice().to_vec(),
            n_draft_tokens: batch - 1,
            metal_hidden_states: Some(hidden_state),
        })
    }
}

impl MetalDFlashProjection {
    pub fn new(weights: &DFlashWeights) -> Result<Self, MetalError> {
        let context = MetalContext::new()?;
        let kernels = MetalKernels::new(&context)?;
        let config = &weights.config;
        let hidden = config.hidden_size;
        let q = config.num_attention_heads * config.head_dim;
        let kv = config.num_key_value_heads * config.head_dim;
        let intermediate = config.intermediate_size;
        let mut projections = BTreeMap::new();
        for (layer, values) in weights.layers.iter().enumerate() {
            for (suffix, data, input_width, output_width) in [
                ("q_proj", values.q_proj_weight.as_slice(), hidden, q),
                ("k_proj", values.k_proj_weight.as_slice(), hidden, kv),
                ("v_proj", values.v_proj_weight.as_slice(), hidden, kv),
                ("o_proj", values.o_proj_weight.as_slice(), q, hidden),
                (
                    "gate_proj",
                    values.gate_proj_weight.as_slice(),
                    hidden,
                    intermediate,
                ),
                (
                    "up_proj",
                    values.up_proj_weight.as_slice(),
                    hidden,
                    intermediate,
                ),
                (
                    "down_proj",
                    values.down_proj_weight.as_slice(),
                    intermediate,
                    hidden,
                ),
            ] {
                projections.insert(
                    format!("layers.{layer}.{suffix}"),
                    Projection {
                        weight: ProjectionWeight::Dense(GpuBuffer::from_f32(&context, data)?),
                        input_width,
                        output_width,
                    },
                );
            }
        }
        let batch = config.block_size;
        let max_input = projections
            .values()
            .map(|projection| batch * projection.input_width)
            .max()
            .unwrap_or(0);
        let max_output = projections
            .values()
            .map(|projection| batch * projection.output_width)
            .max()
            .unwrap_or(0);
        Ok(Self {
            scratch: Mutex::new(Scratch {
                input: GpuBuffer::zeros(&context, max_input)?,
                output: GpuBuffer::zeros(&context, max_output)?,
            }),
            context,
            kernels,
            batch,
            projections,
        })
    }
}

impl DFlashProjectionBackend for MetalDFlashProjection {
    fn project(&self, name: &str, input: &[f32]) -> Result<Vec<f32>, String> {
        let projection = self
            .projections
            .get(name)
            .ok_or_else(|| format!("Metal DFlash projection {name} is absent"))?;
        let expected = self.batch * projection.input_width;
        if input.len() != expected {
            return Err(format!(
                "Metal DFlash projection {name} expected {expected} inputs, got {}",
                input.len()
            ));
        }
        let mut scratch = self
            .scratch
            .lock()
            .map_err(|_| "Metal DFlash scratch lock poisoned")?;
        scratch.input.as_mut_slice()[..expected].copy_from_slice(input);
        let command = self.context.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        let ProjectionWeight::Dense(weight) = &projection.weight else {
            return Err("standalone Metal projection backend requires dense weights".into());
        };
        self.kernels.encode_dense_f32_batch(
            encoder,
            weight,
            &scratch.input,
            &scratch.output,
            projection.output_width,
            projection.input_width,
            self.batch,
        );
        encoder.end_encoding();
        command.commit();
        self.context
            .wait_for_completion(command, Duration::from_secs(30))
            .map_err(|error| error.to_string())?;
        Ok(scratch.output.as_slice()[..self.batch * projection.output_width].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::{MetalDFlashForward, PreparedContextStamp};
    use crate::dflash::{
        DFlashConfig, DFlashContextKvCache, DFlashLayerWeights, DFlashSpecificConfig, DFlashWeights,
    };

    fn matrix(rows: usize, columns: usize, phase: usize) -> Vec<f32> {
        (0..rows * columns)
            .map(|index| {
                let centered = ((index * 17 + phase * 13) % 29) as f32 - 14.0;
                centered / 97.0
            })
            .collect()
    }

    fn test_weights() -> DFlashWeights {
        let hidden = 8;
        let head_dim = 4;
        let heads = 2;
        let kv_heads = 1;
        let intermediate = 12;
        let sampled = 2;
        let layers = (0..5)
            .map(|layer| DFlashLayerWeights {
                input_layernorm_weight: vec![1.0 + layer as f32 * 0.01; hidden],
                post_attention_layernorm_weight: vec![0.9 + layer as f32 * 0.01; hidden],
                q_proj_weight: matrix(heads * head_dim, hidden, 10 + layer),
                k_proj_weight: matrix(kv_heads * head_dim, hidden, 20 + layer),
                v_proj_weight: matrix(kv_heads * head_dim, hidden, 30 + layer),
                o_proj_weight: matrix(hidden, heads * head_dim, 40 + layer),
                q_norm_weight: vec![1.0 + layer as f32 * 0.02; head_dim],
                k_norm_weight: vec![0.95 + layer as f32 * 0.02; head_dim],
                gate_proj_weight: matrix(intermediate, hidden, 50 + layer),
                up_proj_weight: matrix(intermediate, hidden, 60 + layer),
                down_proj_weight: matrix(hidden, intermediate, 70 + layer),
            })
            .collect();
        DFlashWeights {
            config: DFlashConfig {
                architectures: vec!["DFlashDraftModel".into()],
                block_size: 4,
                bos_token_id: 1,
                eos_token_id: 2,
                hidden_size: hidden,
                head_dim,
                intermediate_size: intermediate,
                num_attention_heads: heads,
                num_hidden_layers: 5,
                num_key_value_heads: kv_heads,
                num_target_layers: 4,
                vocab_size: 32,
                max_position_embeddings: 64,
                sliding_window: 64,
                rms_norm_eps: 1e-6,
                rope_theta: 10_000.0,
                dflash_config: DFlashSpecificConfig {
                    mask_token_id: 31,
                    target_layer_ids: (0..sampled).collect(),
                },
                dtype: Some("f32-test".into()),
            },
            fc_weight: matrix(hidden, sampled * hidden, 1),
            hidden_norm_weight: vec![1.0; hidden],
            norm_weight: vec![1.0; hidden],
            layers,
            gguf_weights: None,
        }
    }

    fn seeded_cache() -> DFlashContextKvCache {
        let mut cache = DFlashContextKvCache::new(5, 4, 2, 8);
        let rows = 9;
        for layer in 0..5 {
            let key = (0..rows * 4)
                .map(|index| (layer * 4 + index) as f32 / 31.0)
                .collect::<Vec<_>>();
            let value = (0..rows * 4)
                .map(|index| (layer * 4 + index + 3) as f32 / 37.0)
                .collect::<Vec<_>>();
            cache.append_layer(layer, &key, &value, rows);
        }
        cache.advance_round(rows);
        cache
    }

    fn gamma14_weights() -> DFlashWeights {
        let hidden = 128;
        let head_dim = 128;
        let intermediate = 12;
        let sampled = 2;
        let layers = (0..5)
            .map(|layer| DFlashLayerWeights {
                input_layernorm_weight: vec![1.0 + layer as f32 * 0.01; hidden],
                post_attention_layernorm_weight: vec![0.9 + layer as f32 * 0.01; hidden],
                q_proj_weight: matrix(hidden, hidden, 10 + layer),
                k_proj_weight: matrix(hidden, hidden, 20 + layer),
                v_proj_weight: matrix(hidden, hidden, 30 + layer),
                o_proj_weight: matrix(hidden, hidden, 40 + layer),
                q_norm_weight: vec![1.0 + layer as f32 * 0.02; head_dim],
                k_norm_weight: vec![0.95 + layer as f32 * 0.02; head_dim],
                gate_proj_weight: matrix(intermediate, hidden, 50 + layer),
                up_proj_weight: matrix(intermediate, hidden, 60 + layer),
                down_proj_weight: matrix(hidden, intermediate, 70 + layer),
            })
            .collect();
        DFlashWeights {
            config: DFlashConfig {
                architectures: vec!["DFlashDraftModel".into()],
                block_size: 16,
                bos_token_id: 1,
                eos_token_id: 2,
                hidden_size: hidden,
                head_dim,
                intermediate_size: intermediate,
                num_attention_heads: 1,
                num_hidden_layers: 5,
                num_key_value_heads: 1,
                num_target_layers: 4,
                vocab_size: 32,
                max_position_embeddings: 128,
                sliding_window: 128,
                rms_norm_eps: 1e-6,
                rope_theta: 500_000.0,
                dflash_config: DFlashSpecificConfig {
                    mask_token_id: 31,
                    target_layer_ids: (0..sampled).collect(),
                },
                dtype: Some("f32-gamma14-test".into()),
            },
            fc_weight: matrix(hidden, sampled * hidden, 1),
            hidden_norm_weight: vec![1.0; hidden],
            norm_weight: vec![1.0; hidden],
            layers,
            gguf_weights: None,
        }
    }

    fn gamma14_seeded_cache(weights: &DFlashWeights) -> DFlashContextKvCache {
        let width = weights.config.num_key_value_heads * weights.config.head_dim;
        let mut cache = DFlashContextKvCache::new(5, width, 2, 16);
        let rows = 18;
        for layer in 0..5 {
            let key = (0..rows * width)
                .map(|index| (layer * 101 + index) as f32 / 137.0)
                .collect::<Vec<_>>();
            let value = (0..rows * width)
                .map(|index| (layer * 103 + index + 7) as f32 / 149.0)
                .collect::<Vec<_>>();
            cache.append_layer(layer, &key, &value, rows);
        }
        cache.advance_round(rows);
        cache
    }

    fn gamma14_target_hidden(weights: &DFlashWeights) -> Vec<f32> {
        let rows = 15;
        let elements =
            rows * weights.config.dflash_config.target_layer_ids.len() * weights.config.hidden_size;
        (0..elements)
            .map(|index| ((index * 19 + 11) % 113) as f32 / 71.0 - 0.75)
            .collect()
    }

    fn gamma14_noise(weights: &DFlashWeights, phase: usize) -> Vec<f32> {
        let elements = weights.config.block_size * weights.config.hidden_size;
        (0..elements)
            .map(|index| ((index * 23 + phase * 17) % 127) as f32 / 79.0 - 0.8)
            .collect()
    }

    fn assert_cache_close(
        actual: &DFlashContextKvCache,
        expected: &DFlashContextKvCache,
        tolerance: f32,
    ) {
        assert_eq!(actual.ctx_len, expected.ctx_len);
        assert_eq!(actual.ctx_offset, expected.ctx_offset);
        assert_eq!(actual.revision(), expected.revision());
        for layer in 0..5 {
            assert_close(actual.layer_k(layer), expected.layer_k(layer), tolerance);
            assert_close(actual.layer_v(layer), expected.layer_v(layer), tolerance);
        }
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                actual.is_finite() && (actual - expected).abs() <= tolerance,
                "difference at {index}: {actual} != {expected} (tol={tolerance})"
            );
        }
    }

    #[test]
    fn split_preparation_matches_monolithic_for_full_and_rejected_prefix() {
        let weights = test_weights();
        let candidate_rows = 3;
        let sampled = weights.config.dflash_config.target_layer_ids.len();
        let hidden = weights.config.hidden_size;
        let target_hidden = (0..candidate_rows * sampled * hidden)
            .map(|index| ((index * 11) % 41) as f32 / 23.0 - 0.8)
            .collect::<Vec<_>>();
        let noise = (0..weights.config.block_size * hidden)
            .map(|index| ((index * 7) % 31) as f32 / 19.0 - 0.7)
            .collect::<Vec<_>>();

        for selected_rows in [candidate_rows, candidate_rows - 1] {
            let mut monolithic = MetalDFlashForward::new(&weights, 64).expect("monolithic Metal");
            let mut split = MetalDFlashForward::new(&weights, 64).expect("split Metal");
            let mut expected_cache = seeded_cache();
            let mut split_cache = seeded_cache();
            let selected_values = selected_rows * sampled * hidden;
            let expected = monolithic
                .forward(
                    &noise,
                    &target_hidden[..selected_values],
                    selected_rows,
                    &mut expected_cache,
                )
                .expect("monolithic forward");
            let prepared = split
                .prepare_target_context(&target_hidden, candidate_rows, &split_cache)
                .expect("prepare all candidate rows");
            let actual = split
                .finish_prepared(prepared, &noise, selected_rows, &mut split_cache)
                .expect("finish selected prefix");

            assert_close(&actual.hidden_states, &expected.hidden_states, 2e-5);
            assert_eq!(split_cache.ctx_len, expected_cache.ctx_len);
            assert_eq!(split_cache.ctx_offset, expected_cache.ctx_offset);
            for layer in 0..5 {
                assert_close(
                    split_cache.layer_k(layer),
                    expected_cache.layer_k(layer),
                    2e-5,
                );
                assert_close(
                    split_cache.layer_v(layer),
                    expected_cache.layer_v(layer),
                    2e-5,
                );
            }
        }
    }

    #[test]
    fn gamma14_retained_mirror_matches_clean_exact_finish_across_window_wrap() {
        let weights = gamma14_weights();
        let target_hidden = gamma14_target_hidden(&weights);
        let predicted_noise = gamma14_noise(&weights, 1);
        let mut clean_forward =
            MetalDFlashForward::new(&weights, 128).expect("clean gamma14 Metal");
        let mut mirror_forward =
            MetalDFlashForward::new(&weights, 128).expect("Mirror gamma14 Metal");
        let mut clean_cache = gamma14_seeded_cache(&weights);
        let mut mirror_cache = gamma14_seeded_cache(&weights);

        let clean_prepared = clean_forward
            .prepare_target_context(&target_hidden, 15, &clean_cache)
            .expect("clean gamma14 prepare");
        let clean = clean_forward
            .finish_prepared(clean_prepared, &predicted_noise, 15, &mut clean_cache)
            .expect("clean gamma14 finish");

        let retained_checkpoint = mirror_cache.checkpoint_append(15);
        let mirror_prepared = mirror_forward
            .prepare_target_context(&target_hidden, 15, &mirror_cache)
            .expect("Mirror gamma14 prepare");
        let mirror = mirror_forward
            .finish_prepared(mirror_prepared, &predicted_noise, 15, &mut mirror_cache)
            .expect("Mirror gamma14 provisional finish");
        // A matching authenticated frontier retains the append, so the
        // checkpoint is intentionally consumed without rollback.
        drop(retained_checkpoint);

        assert_close(&mirror.hidden_states, &clean.hidden_states, 2e-5);
        assert_cache_close(&mirror_cache, &clean_cache, 2e-5);
    }

    #[test]
    fn gamma14_rollback_and_exact_replay_match_clean_for_every_committed_prefix() {
        let weights = gamma14_weights();
        let target_hidden = gamma14_target_hidden(&weights);
        let predicted_noise = gamma14_noise(&weights, 1);
        let exact_noise = gamma14_noise(&weights, 2);
        assert_ne!(predicted_noise, exact_noise);
        let mut replay_forward =
            MetalDFlashForward::new(&weights, 128).expect("replay gamma14 Metal");
        let mut clean_forward =
            MetalDFlashForward::new(&weights, 128).expect("clean gamma14 Metal");

        for committed_rows in 1usize..=15 {
            let mut replay_cache = gamma14_seeded_cache(&weights);
            let mut clean_cache = gamma14_seeded_cache(&weights);
            let before = replay_cache.snapshot();
            let before_revision = replay_cache.revision();
            let checkpoint = replay_cache.checkpoint_append(15);
            let provisional_prepared = replay_forward
                .prepare_target_context(&target_hidden, 15, &replay_cache)
                .expect("provisional gamma14 prepare");
            replay_forward
                .finish_prepared(
                    provisional_prepared,
                    &predicted_noise,
                    15,
                    &mut replay_cache,
                )
                .expect("provisional gamma14 finish");
            replay_cache
                .rollback_append(checkpoint)
                .expect("provisional gamma14 rollback");
            assert_eq!(replay_cache.revision(), before_revision);
            assert_eq!(replay_cache.snapshot().position, before.position);
            assert_eq!(replay_cache.snapshot().layers, before.layers);

            // Preparing again forces the Metal mirror from its speculative
            // post-revision back onto the restored authoritative CPU shadow.
            let replay_prepared = replay_forward
                .prepare_target_context(&target_hidden, 15, &replay_cache)
                .expect("exact replay prepare");
            let replay = replay_forward
                .finish_prepared(
                    replay_prepared,
                    &exact_noise,
                    committed_rows,
                    &mut replay_cache,
                )
                .expect("exact replay finish");

            let clean_prepared = clean_forward
                .prepare_target_context(&target_hidden, 15, &clean_cache)
                .expect("clean exact prepare");
            let clean = clean_forward
                .finish_prepared(
                    clean_prepared,
                    &exact_noise,
                    committed_rows,
                    &mut clean_cache,
                )
                .expect("clean exact finish");

            assert_close(&replay.hidden_states, &clean.hidden_states, 2e-5);
            assert_cache_close(&replay_cache, &clean_cache, 2e-5);
        }
    }

    #[test]
    fn gamma14_failed_provisional_finish_rolls_back_an_unchanged_checkpoint() {
        let weights = gamma14_weights();
        let target_hidden = gamma14_target_hidden(&weights);
        let predicted_noise = gamma14_noise(&weights, 1);
        let mut forward = MetalDFlashForward::new(&weights, 128).expect("failed gamma14 Metal");
        let mut cache = gamma14_seeded_cache(&weights);
        let before = cache.snapshot();
        let before_revision = cache.revision();
        let checkpoint = cache.checkpoint_append(15);
        let prepared = forward
            .prepare_target_context(&target_hidden, 15, &cache)
            .expect("failed gamma14 prepare");
        assert!(forward
            .finish_prepared(prepared, &predicted_noise, 0, &mut cache)
            .is_err());
        cache
            .rollback_append(checkpoint)
            .expect("unchanged checkpoint rollback");
        assert_eq!(cache.revision(), before_revision);
        assert_eq!(cache.snapshot().position, before.position);
        assert_eq!(cache.snapshot().layers, before.layers);
    }

    #[test]
    fn prepared_stamp_binds_cache_incarnation_and_geometry() {
        let cache = seeded_cache();
        let stamp = PreparedContextStamp {
            generation: 7,
            cache_identity: cache.identity(),
            cache_revision: cache.revision(),
            ctx_len: cache.ctx_len,
            ctx_offset: cache.ctx_offset,
            sink_size: cache.sink_size,
            window_size: cache.window_size,
            rows: 3,
        };
        assert!(MetalDFlashForward::cache_matches_prepared(&cache, stamp));

        let same_cut_new_incarnation = seeded_cache();
        assert!(!MetalDFlashForward::cache_matches_prepared(
            &same_cut_new_incarnation,
            stamp,
        ));
        let mut changed_geometry = stamp;
        changed_geometry.window_size += 1;
        assert!(!MetalDFlashForward::cache_matches_prepared(
            &cache,
            changed_geometry,
        ));
    }

    #[test]
    fn prepared_handles_are_single_use_and_invalid_selection_commits_nothing() {
        let weights = test_weights();
        let rows = 3;
        let target_values =
            rows * weights.config.dflash_config.target_layer_ids.len() * weights.config.hidden_size;
        let target_hidden = vec![0.25; target_values];
        let noise = vec![0.5; weights.config.block_size * weights.config.hidden_size];
        let mut forward = MetalDFlashForward::new(&weights, 64).expect("Metal forward");
        let mut cache = seeded_cache();

        let superseded = forward
            .prepare_target_context(&target_hidden, rows, &cache)
            .expect("first preparation");
        let active = forward
            .prepare_target_context(&target_hidden, rows, &cache)
            .expect("replacement preparation");
        assert!(forward.discard_prepared(superseded).is_err());
        forward
            .discard_prepared(active)
            .expect("stale handle must not discard its replacement");

        let before_offset = cache.ctx_offset;
        let before_key = cache.layer_k(0).to_vec();
        let invalid = forward
            .prepare_target_context(&target_hidden, rows, &cache)
            .expect("preparation for invalid selection");
        assert!(forward
            .finish_prepared(invalid, &noise, 0, &mut cache)
            .is_err());
        assert_eq!(cache.ctx_offset, before_offset);
        assert_eq!(cache.layer_k(0), before_key);
    }

    /// Explicit, ignored real-artifact hook for measuring how much of a
    /// gamma-15 cycle moves before versus after the exact-frontier decision.
    #[cfg(feature = "release-real-model")]
    #[test]
    #[ignore = "requires MUSER_DFLASH and is diagnostic evidence, not CI"]
    fn diagnostic_real_gamma15_split_stage_timing() {
        use std::path::Path;
        use std::time::Instant;

        let path = std::env::var("MUSER_DFLASH")
            .expect("release-real-model split timing requires MUSER_DFLASH");
        let weights = DFlashWeights::load_metal(Path::new(&path)).expect("official DFlash GGUF");
        let config = weights.config.clone();
        assert_eq!(config.block_size, 16, "gamma-15 requires block_size=16");
        let sampled = config.dflash_config.target_layer_ids.len();
        let rows = config.block_size;
        let target_hidden = (0..rows * sampled * config.hidden_size)
            .map(|index| ((index * 17) % 101) as f32 / 101.0 - 0.5)
            .collect::<Vec<_>>();
        let noise = (0..config.block_size * config.hidden_size)
            .map(|index| ((index * 13) % 89) as f32 / 89.0 - 0.5)
            .collect::<Vec<_>>();
        let width = config.num_key_value_heads * config.head_dim;
        let mut forward = MetalDFlashForward::new(&weights, 131_072).expect("Metal DFlash");

        let mut monolithic_cache =
            DFlashContextKvCache::new(config.num_hidden_layers, width, 64, config.sliding_window);
        let monolithic = forward
            .forward(&noise, &target_hidden, rows, &mut monolithic_cache)
            .expect("real monolithic equivalence forward");
        let mut cache =
            DFlashContextKvCache::new(config.num_hidden_layers, width, 64, config.sliding_window);
        let prepared = forward
            .prepare_target_context(&target_hidden, rows, &cache)
            .expect("real split equivalence prepare");
        let split = forward
            .finish_prepared(prepared, &noise, rows, &mut cache)
            .expect("real split equivalence finish");
        assert_eq!(split.hidden_states, monolithic.hidden_states);
        for layer in 0..config.num_hidden_layers {
            assert_eq!(cache.layer_k(layer), monolithic_cache.layer_k(layer));
            assert_eq!(cache.layer_v(layer), monolithic_cache.layer_v(layer));
        }

        let warm = forward
            .prepare_target_context(&target_hidden, rows, &cache)
            .expect("warm prepare");
        forward
            .finish_prepared(warm, &noise, rows, &mut cache)
            .expect("warm finish");

        let mut prepare_ms = Vec::new();
        let mut finish_ms = Vec::new();
        for _ in 0..7 {
            let started = Instant::now();
            let prepared = forward
                .prepare_target_context(&target_hidden, rows, &cache)
                .expect("timed prepare");
            prepare_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
            let started = Instant::now();
            let output = forward
                .finish_prepared(prepared, &noise, rows, &mut cache)
                .expect("timed finish");
            finish_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
            std::hint::black_box(output.hidden_states[0]);
        }
        let mut monolithic_ms = Vec::new();
        for _ in 0..7 {
            let started = Instant::now();
            let output = forward
                .forward(&noise, &target_hidden, rows, &mut cache)
                .expect("timed monolithic forward");
            monolithic_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
            std::hint::black_box(output.hidden_states[0]);
        }
        prepare_ms.sort_by(f64::total_cmp);
        finish_ms.sort_by(f64::total_cmp);
        monolithic_ms.sort_by(f64::total_cmp);
        println!(
            "dflash-gamma15-split prepare_median_ms={:.3} finish_median_ms={:.3} monolithic_median_ms={:.3} prepare_samples_ms={prepare_ms:?} finish_samples_ms={finish_ms:?} monolithic_samples_ms={monolithic_ms:?}",
            prepare_ms[prepare_ms.len() / 2],
            finish_ms[finish_ms.len() / 2],
            monolithic_ms[monolithic_ms.len() / 2],
        );
    }

    #[test]
    fn round_arena_shrinks_back_to_block_floor_after_prompt_sized_growth() {
        let weights = test_weights();
        let floor = weights.config.block_size;
        let mut forward = MetalDFlashForward::new(&weights, 64).expect("forward init");
        assert_eq!(forward.round_capacity, floor);
        // A prompt-sized round grows the arena (prefill path).
        forward.ensure_round_capacity(32).expect("grow");
        assert!(forward.round_capacity >= 32);
        // The first decode-sized round after it must release the giant slabs;
        // otherwise unified memory pages them in/out inside every draft wait.
        forward.ensure_round_capacity(floor).expect("shrink");
        assert_eq!(forward.round_capacity, floor);
        // Growth still works afterwards (a later long prompt regrows cleanly).
        forward.ensure_round_capacity(48).expect("regrow");
        assert!(forward.round_capacity >= 48);
        forward.ensure_round_capacity(floor).expect("shrink again");
        assert_eq!(forward.round_capacity, floor);
    }
}
