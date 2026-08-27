use metal::{ComputeCommandEncoderRef, MTLSize};

use super::{dispatch_1d, set_value, MetalKernels, LLAMA_FA_NCPSG, LLAMA_FA_NWG};
use crate::metal::buffer::{GpuBuffer, GpuBytes, GpuHalfBuffer};

impl MetalKernels {
    /// Ferrite's accepted DK128 FlashAttention-2 prefill route. K/V must
    /// already be stored in a contiguous F16 cache; the Muser specialization
    /// selects token-major SWA or head-major NoPE strides explicitly. Wrapped
    /// SWA batches arrive through a compact logical-tail staging arena and set
    /// `cache_logical_base` to the ring's explicit logical origin.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_flash_attention_v2(
        &self,
        encoder: &ComputeCommandEncoderRef,
        query: &GpuBuffer,
        key_cache: &GpuHalfBuffer,
        value_cache: &GpuHalfBuffer,
        output: &GpuBuffer,
        token_count: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        start_position: usize,
        capacity: usize,
        cache_logical_base: usize,
        window: usize,
        scale: f32,
        head_major: bool,
    ) {
        debug_assert_eq!(head_dim, 128);
        debug_assert_eq!(query.len(), token_count * n_heads * head_dim);
        debug_assert_eq!(key_cache.len(), capacity * n_kv_heads * head_dim);
        debug_assert_eq!(value_cache.len(), key_cache.len());
        debug_assert_eq!(output.len(), query.len());
        if std::env::var_os("MUSER_CROSS_VENDOR_QK").is_some() {
            let old_visible = start_position
                .checked_sub(cache_logical_base)
                .expect("attention cache origin cannot follow the query position");
            debug_assert!(old_visible + token_count <= capacity);
            encoder.set_compute_pipeline_state(&self.cross_vendor_attention_prefill);
            encoder.set_buffer(0, Some(query.metal()), 0);
            encoder.set_buffer(1, Some(key_cache.metal()), 0);
            encoder.set_buffer(2, Some(value_cache.metal()), 0);
            encoder.set_buffer(3, Some(output.metal()), 0);
            set_value(encoder, 4, &(old_visible as u32));
            set_value(encoder, 5, &(capacity as u32));
            set_value(encoder, 6, &(n_heads as u32));
            set_value(encoder, 7, &(n_kv_heads as u32));
            set_value(encoder, 8, &(head_dim as u32));
            set_value(encoder, 9, &scale);
            set_value(encoder, 10, &(u32::from(head_major)));
            set_value(encoder, 11, &(window as u32));
            set_value(encoder, 12, &(token_count as u32));
            encoder.dispatch_thread_groups(
                MTLSize::new(n_heads as u64, token_count as u64, 1),
                MTLSize::new(32, 1, 1),
            );
            return;
        }
        if token_count == 1 && n_heads.is_multiple_of(8) && n_heads / n_kv_heads == 16 {
            self.bind(encoder, "muser_flash_attn_decode_gqa_fa2");
            encoder.set_buffer(0, Some(query.metal()), 0);
            encoder.set_buffer(1, Some(key_cache.metal()), 0);
            encoder.set_buffer(2, Some(value_cache.metal()), 0);
            encoder.set_buffer(3, Some(output.metal()), 0);
            set_value(encoder, 4, &(start_position as u32));
            set_value(encoder, 5, &(n_heads as u32));
            set_value(encoder, 6, &(n_kv_heads as u32));
            set_value(encoder, 7, &((capacity * head_dim) as u32));
            set_value(encoder, 8, &scale);
            set_value(encoder, 9, &(u32::from(head_major)));
            set_value(encoder, 10, &(cache_logical_base as u32));
            set_value(encoder, 11, &(window as u32));
            encoder.dispatch_thread_groups(
                MTLSize::new((n_heads / 8) as u64, 1, 1),
                MTLSize::new(128, 1, 1),
            );
            return;
        }
        self.bind(encoder, "flash_attn_v2");
        encoder.set_buffer(0, Some(query.metal()), 0);
        encoder.set_buffer(1, Some(key_cache.metal()), 0);
        encoder.set_buffer(2, Some(value_cache.metal()), 0);
        encoder.set_buffer(3, Some(output.metal()), 0);
        set_value(encoder, 4, &(token_count as u32));
        set_value(encoder, 5, &(start_position as u32));
        set_value(encoder, 6, &(n_heads as u32));
        set_value(encoder, 7, &(n_kv_heads as u32));
        set_value(encoder, 8, &((capacity * head_dim) as u32));
        set_value(encoder, 9, &scale);
        set_value(encoder, 10, &(window as u32));
        set_value(encoder, 11, &(u32::from(head_major)));
        set_value(encoder, 12, &(cache_logical_base as u32));
        set_value(encoder, 15, &0.0f32);
        encoder.dispatch_thread_groups(
            MTLSize::new(token_count.div_ceil(8) as u64, n_heads as u64, 1),
            MTLSize::new(128, 1, 1),
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_stage_swa_prefill_f16(
        &self,
        encoder: &ComputeCommandEncoderRef,
        current_key: &GpuBuffer,
        current_value: &GpuBuffer,
        ring_key: &GpuHalfBuffer,
        ring_value: &GpuHalfBuffer,
        staged_key: &GpuHalfBuffer,
        staged_value: &GpuHalfBuffer,
        kv_dim: usize,
        old_len: usize,
        old_origin_physical: usize,
        ring_capacity: usize,
        token_count: usize,
    ) {
        debug_assert_eq!(current_key.len(), token_count * kv_dim);
        debug_assert_eq!(current_value.len(), current_key.len());
        debug_assert_eq!(ring_key.len(), ring_capacity * kv_dim);
        debug_assert_eq!(ring_value.len(), ring_key.len());
        debug_assert!(staged_key.len() >= (old_len + token_count) * kv_dim);
        debug_assert_eq!(staged_value.len(), staged_key.len());
        debug_assert!(old_origin_physical < ring_capacity);
        self.bind(encoder, "muser_stage_swa_prefill_f16");
        encoder.set_buffer(0, Some(current_key.metal()), 0);
        encoder.set_buffer(1, Some(current_value.metal()), 0);
        encoder.set_buffer(2, Some(ring_key.metal()), 0);
        encoder.set_buffer(3, Some(ring_value.metal()), 0);
        encoder.set_buffer(4, Some(staged_key.metal()), 0);
        encoder.set_buffer(5, Some(staged_value.metal()), 0);
        set_value(encoder, 6, &(kv_dim as u32));
        set_value(encoder, 7, &(old_len as u32));
        set_value(encoder, 8, &(old_origin_physical as u32));
        set_value(encoder, 9, &(ring_capacity as u32));
        set_value(encoder, 10, &(token_count as u32));
        dispatch_1d(encoder, (old_len + token_count) * kv_dim);
    }

    /// Materialize Muser's compact SWA ring at llama.cpp's absolute,
    /// 256-row-padded KV indices for one-row decode. The staged mask retains
    /// llama's masked cells, so the pinned vec kernel sees the same reduction
    /// lanes rather than a mathematically equivalent compact permutation.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_stage_swa_llama_decode_f16(
        &self,
        encoder: &ComputeCommandEncoderRef,
        current_key: &GpuBuffer,
        current_value: &GpuBuffer,
        ring_key: &GpuHalfBuffer,
        ring_value: &GpuHalfBuffer,
        staged_key: &GpuHalfBuffer,
        staged_value: &GpuHalfBuffer,
        staged_mask: &GpuBytes,
        kv_dim: usize,
        old_len: usize,
        old_origin_logical: usize,
        old_origin_physical: usize,
        ring_capacity: usize,
        position: usize,
    ) {
        debug_assert_eq!(old_len, ring_capacity);
        debug_assert_eq!(current_key.len(), kv_dim);
        debug_assert_eq!(current_value.len(), kv_dim);
        debug_assert!(position < staged_key.len() / kv_dim);
        debug_assert_eq!(staged_value.len(), staged_key.len());
        debug_assert!(staged_mask.len() / std::mem::size_of::<u16>() > position);
        self.bind(encoder, "muser_stage_swa_llama_decode_f16");
        encoder.set_buffer(0, Some(current_key.metal()), 0);
        encoder.set_buffer(1, Some(current_value.metal()), 0);
        encoder.set_buffer(2, Some(ring_key.metal()), 0);
        encoder.set_buffer(3, Some(ring_value.metal()), 0);
        encoder.set_buffer(4, Some(staged_key.metal()), 0);
        encoder.set_buffer(5, Some(staged_value.metal()), 0);
        encoder.set_buffer(6, Some(staged_mask.metal()), 0);
        set_value(encoder, 7, &(kv_dim as u32));
        set_value(encoder, 8, &(old_len as u32));
        set_value(encoder, 9, &(old_origin_logical as u32));
        set_value(encoder, 10, &(old_origin_physical as u32));
        set_value(encoder, 11, &(ring_capacity as u32));
        set_value(encoder, 12, &(position as u32));
        dispatch_1d(encoder, (old_len + 1) * kv_dim);
    }

    /// Exact Ferrite a85048a90 cache-interleaved producer + LSE reducer for
    /// the growing NoPE planes. These planes are head-major and never wrap;
    /// SWA rings remain on Muser's explicit-origin kernel below.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_ferrite_attention_decode_interleaved_f16(
        &self,
        encoder: &ComputeCommandEncoderRef,
        query: &GpuBuffer,
        key_cache: &GpuHalfBuffer,
        value_cache: &GpuHalfBuffer,
        partials: &GpuBuffer,
        output: &GpuBuffer,
        current_key: &GpuBuffer,
        current_value: &GpuBuffer,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        position: usize,
        capacity: usize,
        scale: f32,
    ) {
        debug_assert_eq!(head_dim, 128);
        debug_assert_eq!(query.len(), n_heads * head_dim);
        debug_assert_eq!(current_key.len(), n_kv_heads * head_dim);
        debug_assert_eq!(current_value.len(), current_key.len());
        debug_assert_eq!(key_cache.len(), n_kv_heads * capacity * head_dim);
        debug_assert_eq!(value_cache.len(), key_cache.len());
        debug_assert!(position < capacity);
        let visible = position + 1;
        let (n_workgroups, n_simdgroups) = splitk_geometry(visible);
        let partial_stride = 2 + head_dim;
        debug_assert!(partials.len() >= n_heads * n_workgroups * partial_stride);

        encoder.set_compute_pipeline_state(self.ferrite_f16_interleaved(n_simdgroups));
        encoder.set_buffer(0, Some(query.metal()), 0);
        encoder.set_buffer(1, Some(key_cache.metal()), 0);
        encoder.set_buffer(2, Some(value_cache.metal()), 0);
        encoder.set_buffer(3, Some(partials.metal()), 0);
        encoder.set_buffer(4, Some(current_key.metal()), 0);
        encoder.set_buffer(5, Some(current_value.metal()), 0);
        set_value(encoder, 6, &(head_dim as u32));
        set_value(encoder, 7, &(position as u32));
        set_value(encoder, 8, &(capacity as u32));
        set_value(encoder, 9, &((n_heads / n_kv_heads) as u32));
        set_value(encoder, 10, &0u32);
        set_value(encoder, 11, &((capacity * head_dim) as u32));
        set_value(encoder, 12, &(n_workgroups as u32));
        set_value(encoder, 13, &(visible.div_ceil(n_workgroups) as u32));
        set_value(encoder, 14, &scale);
        set_value(encoder, 15, &0u32);
        encoder.set_buffer(16, None, 0);
        let shared_floats = head_dim + n_simdgroups * partial_stride;
        encoder.set_threadgroup_memory_length(
            0,
            ((shared_floats * std::mem::size_of::<f32>() + 15) & !15) as u64,
        );
        encoder.dispatch_thread_groups(
            MTLSize::new(n_heads as u64, n_workgroups as u64, 1),
            MTLSize::new(32, n_simdgroups as u64, 1),
        );

        let partial_barrier: [&metal::ResourceRef; 1] = [partials.metal()];
        encoder.memory_barrier_with_resources(&partial_barrier);
        encoder.set_compute_pipeline_state(self.ferrite_f16_reduce());
        encoder.set_buffer(0, Some(partials.metal()), 0);
        encoder.set_buffer(1, Some(output.metal()), 0);
        set_value(encoder, 2, &(head_dim as u32));
        set_value(encoder, 3, &(n_workgroups as u32));
        encoder.set_buffer(18, None, 0);
        encoder.dispatch_thread_groups(MTLSize::new(n_heads as u64, 1, 1), MTLSize::new(32, 1, 1));
        let output_barrier: [&metal::ResourceRef; 2] = [partials.metal(), output.metal()];
        encoder.memory_barrier_with_resources(&output_barrier);
    }

    /// Causal f16 mask fill plus the pinned `flash_attn_ext_blk` block
    /// classifier for the llama non-vec prefill route. Rows are chunk
    /// queries, columns the visible cache prefix; both outputs depend only
    /// on the chunk bounds, so one dispatch per prefill chunk is shared by
    /// every full-attention layer in that chunk. The classifier's
    /// skip/partial/dense bytes are what make the pinned kernel's causal
    /// prefill cheap on the fully-masked upper triangle.
    pub fn encode_llama_fa_prefill_mask_blk(
        &self,
        encoder: &ComputeCommandEncoderRef,
        mask: &GpuBytes,
        blk: &GpuBytes,
        start_position: usize,
        token_count: usize,
    ) {
        let visible = start_position + token_count;
        debug_assert!(token_count > 0);
        debug_assert_eq!(token_count % super::LLAMA_FA_PREFILL_NQPTG as usize, 0);
        debug_assert_eq!(visible % super::LLAMA_FA_PREFILL_NCPSG as usize, 0);
        let half = std::mem::size_of::<u16>() as u64;
        debug_assert!(mask.len() as u64 >= token_count as u64 * visible as u64 * half);
        let nblk0 = visible / super::LLAMA_FA_PREFILL_NCPSG as usize;
        let nblk1 = token_count / super::LLAMA_FA_PREFILL_NQPTG as usize;
        debug_assert!(blk.len() >= nblk0 * nblk1);
        self.bind(encoder, "muser_fa_causal_mask_f16");
        encoder.set_buffer(0, Some(mask.metal()), 0);
        set_value(encoder, 1, &(start_position as u32));
        set_value(encoder, 2, &(token_count as u32));
        set_value(encoder, 3, &(visible as u32));
        encoder.dispatch_threads(
            MTLSize::new(visible as u64, token_count as u64, 1),
            MTLSize::new(32, 8, 1),
        );
        let mask_barrier: [&metal::ResourceRef; 1] = [mask.metal()];
        encoder.memory_barrier_with_resources(&mask_barrier);
        let blk_args = GgmlMetalKargsFlashAttnExtBlk {
            ne01: token_count as i32,
            ne30: visible as i32,
            ne31: token_count as i32,
            ne32: 1,
            ne33: 1,
            nb31: visible as u64 * half,
            nb32: 0,
            nb33: 0,
        };
        let pipelines = self
            .llama_flash()
            .expect("llama flash-attn prefill requires MUSER_GGML_METALLIB");
        encoder.set_compute_pipeline_state(&pipelines.prefill_blk);
        set_value(encoder, 0, &blk_args);
        encoder.set_buffer(1, Some(mask.metal()), 0);
        encoder.set_buffer(2, Some(blk.metal()), 0);
        encoder.dispatch_thread_groups(
            MTLSize::new(nblk0 as u64, nblk1 as u64, 1),
            MTLSize::new(32, 1, 1),
        );
        let blk_barrier: [&metal::ResourceRef; 2] = [mask.metal(), blk.metal()];
        encoder.memory_barrier_with_resources(&blk_barrier);
    }

    /// llama.cpp `kernel_flash_attn_ext_f16_dk128_dv128` masked causal
    /// prefill for one full-attention (NoPE) layer. Caller guarantees the
    /// route contract: `token_count` 8-aligned,
    /// `start_position + token_count` 32-aligned, head-major f16 cache
    /// (ns10=128), mask/blk prepared by `encode_llama_fa_prefill_mask_blk`,
    /// and K/V for `[0, start_position + token_count)` already
    /// stored. Q and the output stay in Muser's token-major
    /// `[token, head, dim]` f32 layout via explicit strides.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_llama_flash_attn_prefill_f16(
        &self,
        encoder: &ComputeCommandEncoderRef,
        query: &GpuBuffer,
        key_cache: &GpuHalfBuffer,
        value_cache: &GpuHalfBuffer,
        mask: &GpuBytes,
        blk: &GpuBytes,
        output: &GpuBuffer,
        token_count: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        start_position: usize,
        capacity: usize,
        scale: f32,
    ) {
        debug_assert_eq!(head_dim, 128);
        debug_assert_eq!(query.len(), token_count * n_heads * head_dim);
        debug_assert_eq!(output.len(), query.len());
        debug_assert_eq!(key_cache.len(), n_kv_heads * capacity * head_dim);
        debug_assert_eq!(value_cache.len(), key_cache.len());
        let visible = start_position + token_count;
        debug_assert_eq!(token_count % super::LLAMA_FA_PREFILL_NQPTG as usize, 0);
        debug_assert_eq!(visible % super::LLAMA_FA_PREFILL_NCPSG as usize, 0);
        debug_assert!(visible <= capacity);
        let half = std::mem::size_of::<u16>() as u64;
        let float = std::mem::size_of::<f32>() as u64;
        debug_assert!(mask.len() as u64 >= token_count as u64 * visible as u64 * half);
        debug_assert!(
            blk.len()
                >= (visible / super::LLAMA_FA_PREFILL_NCPSG as usize)
                    * (token_count / super::LLAMA_FA_PREFILL_NQPTG as usize)
        );
        let pipelines = self
            .llama_flash()
            .expect("llama flash-attn prefill requires MUSER_GGML_METALLIB");

        // ggml strides against Muser layouts: Q rows are token-major with
        // head-interleaved 128-float rows (nb01 = full per-token stride,
        // nb02 = per-head stride); the head-major cache rows give ns10=128.
        let nb01 = (n_heads * head_dim) as u64 * float;
        let nb02 = head_dim as u64 * float;
        let nb11 = head_dim as u64 * half;
        let nb12 = capacity as u64 * nb11;
        let args = GgmlMetalKargsFlashAttnExtVec {
            ne01: token_count as i32,
            ne02: n_heads as i32,
            ne03: 1,
            nb01,
            nb02,
            nb03: token_count as u64 * nb01,
            ne11: visible as i32,
            ne_12_2: n_kv_heads as i32,
            ne_12_3: 1,
            ns10: head_dim as i32,
            nb11,
            nb12,
            nb13: n_kv_heads as u64 * nb12,
            ns20: head_dim as i32,
            nb21: nb11,
            nb22: nb12,
            nb23: n_kv_heads as u64 * nb12,
            ne31: token_count as i32,
            ne32: 1,
            ne33: 1,
            nb31: visible as u64 * half,
            nb32: 0,
            nb33: 0,
            ne1: n_heads as i32,
            ne2: token_count as i32,
            ne3: 1,
            scale,
            max_bias: 0.0,
            m0: 0.0,
            m1: 0.0,
            n_head_log2: 0,
            logit_softcap: 0.0,
        };
        encoder.set_compute_pipeline_state(&pipelines.prefill);
        set_value(encoder, 0, &args);
        encoder.set_buffer(1, Some(query.metal()), 0);
        encoder.set_buffer(2, Some(key_cache.metal()), 0);
        encoder.set_buffer(3, Some(value_cache.metal()), 0);
        encoder.set_buffer(4, Some(mask.metal()), 0);
        encoder.set_buffer(5, None, 0);
        encoder.set_buffer(6, Some(mask.metal()), 0);
        encoder.set_buffer(7, Some(blk.metal()), 0);
        encoder.set_buffer(8, Some(output.metal()), 0);
        // FATTN_SMEM(4): PAD(8 * (128 + 2*PAD(128,64) + 2*2*32) * 2, 16).
        encoder.set_threadgroup_memory_length(0, 8192);
        encoder.dispatch_thread_groups(
            MTLSize::new(
                (token_count / super::LLAMA_FA_PREFILL_NQPTG as usize) as u64,
                n_heads as u64,
                1,
            ),
            MTLSize::new(32, super::LLAMA_FA_PREFILL_NSG as u64, 1),
        );
        let output_barrier: [&metal::ResourceRef; 1] = [output.metal()];
        encoder.memory_barrier_with_resources(&output_barrier);
    }

    /// llama.cpp `kernel_flash_attn_ext_vec_f16_dk128_dv128`. Head-major
    /// NoPE planes use ns10=128; token-major SWA rings use ns10=256. `visible`
    /// is the number of cache rows to scan from physical 0. A wrapped SWA
    /// ring is valid when it is full: every slot is in-window and softmax is
    /// permutation-invariant.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_llama_flash_attn_decode_vec_f16(
        &self,
        encoder: &ComputeCommandEncoderRef,
        query: &GpuBuffer,
        key_cache: &GpuHalfBuffer,
        value_cache: &GpuHalfBuffer,
        mask: &GpuBytes,
        pad: &GpuBytes,
        tmp: &GpuBuffer,
        output: &GpuBuffer,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        visible: usize,
        capacity: usize,
        cache_row_offset: usize,
        cache_origin_physical: usize,
        use_mask: bool,
        scale: f32,
        head_major: bool,
        query_row: usize,
        output_row: usize,
    ) {
        debug_assert_eq!(head_dim, 128);
        debug_assert!(query.len() >= (query_row + 1) * n_heads * head_dim);
        debug_assert!(output.len() >= (output_row + 1) * n_heads * head_dim);
        debug_assert_eq!(key_cache.len(), n_kv_heads * capacity * head_dim);
        debug_assert_eq!(value_cache.len(), key_cache.len());
        debug_assert!(visible > 0);
        debug_assert!(cache_row_offset + visible <= capacity);
        debug_assert!(!head_major || cache_row_offset == 0);
        debug_assert!(cache_origin_physical < capacity);
        debug_assert!(!use_mask || mask.len() >= visible * std::mem::size_of::<u16>());
        if std::env::var_os("MUSER_CROSS_VENDOR_QK").is_some() {
            debug_assert_eq!(cache_row_offset, 0);
            encoder.set_compute_pipeline_state(&self.cross_vendor_attention_decode);
            encoder.set_buffer(0, Some(query.metal()), 0);
            encoder.set_buffer(1, Some(key_cache.metal()), 0);
            encoder.set_buffer(2, Some(value_cache.metal()), 0);
            encoder.set_buffer(3, Some(output.metal()), 0);
            set_value(encoder, 4, &(visible as u32));
            set_value(encoder, 5, &(capacity as u32));
            set_value(encoder, 6, &(n_heads as u32));
            set_value(encoder, 7, &(n_kv_heads as u32));
            set_value(encoder, 8, &(head_dim as u32));
            set_value(encoder, 9, &scale);
            set_value(encoder, 10, &(u32::from(head_major)));
            set_value(encoder, 11, &(query_row as u32));
            set_value(encoder, 12, &(output_row as u32));
            set_value(encoder, 13, &(cache_origin_physical as u32));
            encoder
                .dispatch_thread_groups(MTLSize::new(n_heads as u64, 1, 1), MTLSize::new(32, 1, 1));
            return;
        }
        let pipelines = self
            .llama_flash()
            .expect("llama flash-attn vec requires MUSER_GGML_METALLIB");
        let nwg = LLAMA_FA_NWG as usize;
        let ncpsg = LLAMA_FA_NCPSG as usize;
        let mut nsg = 1usize;
        while nsg < 4 && 2 * nwg * nsg * ncpsg < visible {
            nsg *= 2;
        }
        let has_kvpad = !visible.is_multiple_of(ncpsg);
        let half = std::mem::size_of::<u16>() as u64;
        let float = std::mem::size_of::<f32>() as u64;
        let (ns10, nb11, nb12, nb13) = if head_major {
            let nb11 = head_dim as u64 * half;
            let nb12 = capacity as u64 * nb11;
            (head_dim, nb11, nb12, n_kv_heads as u64 * nb12)
        } else {
            let nb11 = n_kv_heads as u64 * head_dim as u64 * half;
            let nb12 = head_dim as u64 * half;
            (n_kv_heads * head_dim, nb11, nb12, capacity as u64 * nb11)
        };
        let nb01 = head_dim as u64 * float;
        let nb02 = nb01;
        let nb03 = n_heads as u64 * nb02;
        debug_assert!(tmp.len() >= n_heads * nwg * (head_dim + 2));
        let mask_pad_bytes = if use_mask { 2 } else { 0 };
        assert!(
            pad.len() >= (ncpsg as u64 * (2 * nb11 * n_kv_heads as u64 + mask_pad_bytes)) as usize,
            "llama flash-attention pad scratch is undersized"
        );
        let cache_byte_offset = (cache_row_offset * n_kv_heads * head_dim) as u64 * half;

        let mask_ne = i32::from(use_mask);
        let mask_stride = if use_mask { (visible * 2) as u64 } else { 0 };
        if has_kvpad {
            let pad_args = GgmlMetalKargsFlashAttnExtPad {
                ne11: visible as i32,
                ne_12_2: n_kv_heads as i32,
                ne_12_3: 1,
                nb11,
                nb12,
                nb13,
                nb21: nb11,
                nb22: nb12,
                nb23: nb13,
                ne31: mask_ne,
                ne32: mask_ne,
                ne33: mask_ne,
                nb31: mask_stride,
                nb32: mask_stride,
                nb33: mask_stride,
            };
            encoder.set_compute_pipeline_state(if use_mask {
                &pipelines.pad
            } else {
                &pipelines.pad_unmasked
            });
            set_value(encoder, 0, &pad_args);
            encoder.set_buffer(1, Some(key_cache.metal()), cache_byte_offset);
            encoder.set_buffer(2, Some(value_cache.metal()), cache_byte_offset);
            encoder.set_buffer(3, use_mask.then(|| mask.metal()), 0);
            encoder.set_buffer(4, Some(pad.metal()), 0);
            encoder.dispatch_thread_groups(
                MTLSize::new(ncpsg as u64, n_kv_heads as u64, 1),
                MTLSize::new(32, 1, 1),
            );
            let pad_barrier: [&metal::ResourceRef; 1] = [pad.metal()];
            encoder.memory_barrier_with_resources(&pad_barrier);
        }

        let args = GgmlMetalKargsFlashAttnExtVec {
            ne01: 1,
            ne02: n_heads as i32,
            ne03: 1,
            nb01,
            nb02,
            nb03,
            ne11: visible as i32,
            ne_12_2: n_kv_heads as i32,
            ne_12_3: 1,
            ns10: ns10 as i32,
            nb11,
            nb12,
            nb13,
            ns20: ns10 as i32,
            nb21: nb11,
            nb22: nb12,
            nb23: nb13,
            ne31: mask_ne,
            ne32: mask_ne,
            ne33: mask_ne,
            nb31: mask_stride,
            nb32: mask_stride,
            nb33: mask_stride,
            ne1: 1,
            ne2: n_heads as i32,
            ne3: 1,
            scale,
            max_bias: 0.0,
            m0: 0.0,
            m1: 0.0,
            n_head_log2: 0,
            logit_softcap: 0.0,
        };
        encoder.set_compute_pipeline_state(pipelines.vec(ns10, nsg, has_kvpad, use_mask));
        set_value(encoder, 0, &args);
        encoder.set_buffer(
            1,
            Some(query.metal()),
            (query_row * n_heads * head_dim * std::mem::size_of::<f32>()) as u64,
        );
        encoder.set_buffer(2, Some(key_cache.metal()), cache_byte_offset);
        encoder.set_buffer(3, Some(value_cache.metal()), cache_byte_offset);
        encoder.set_buffer(4, use_mask.then(|| mask.metal()), 0);
        encoder.set_buffer(5, None, 0);
        encoder.set_buffer(6, Some(pad.metal()), 0);
        encoder.set_buffer(7, Some(tmp.metal()), 0);
        encoder.set_threadgroup_memory_length(0, (1024 * nsg as u64 + 15) & !15);
        encoder.dispatch_thread_groups(
            MTLSize::new(1, n_heads as u64, nwg as u64),
            MTLSize::new(32, nsg as u64, 1),
        );
        let tmp_barrier: [&metal::ResourceRef; 1] = [tmp.metal()];
        encoder.memory_barrier_with_resources(&tmp_barrier);

        let reduce = GgmlMetalKargsFlashAttnExtVecReduce {
            nrows: n_heads as i32,
        };
        encoder.set_compute_pipeline_state(&pipelines.reduce);
        set_value(encoder, 0, &reduce);
        encoder.set_buffer(1, Some(tmp.metal()), 0);
        encoder.set_buffer(
            2,
            Some(output.metal()),
            (output_row * n_heads * head_dim * std::mem::size_of::<f32>()) as u64,
        );
        encoder.dispatch_thread_groups(
            MTLSize::new(n_heads as u64, 1, 1),
            MTLSize::new(32 * nwg as u64, 1, 1),
        );
        let output_barrier: [&metal::ResourceRef; 2] = [tmp.metal(), output.metal()];
        encoder.memory_barrier_with_resources(&output_barrier);
    }

    pub fn encode_kv_store_f16(
        &self,
        encoder: &ComputeCommandEncoderRef,
        key: &GpuBuffer,
        value: &GpuBuffer,
        key_cache: &GpuHalfBuffer,
        value_cache: &GpuHalfBuffer,
        write_index: usize,
    ) {
        debug_assert_eq!(key.len(), value.len());
        debug_assert_eq!(key_cache.len(), value_cache.len());
        debug_assert!(key_cache.len() >= (write_index + 1) * key.len());
        self.bind(encoder, "muser_kv_store_f16");
        encoder.set_buffer(0, Some(key.metal()), 0);
        encoder.set_buffer(1, Some(value.metal()), 0);
        encoder.set_buffer(2, Some(key_cache.metal()), 0);
        encoder.set_buffer(3, Some(value_cache.metal()), 0);
        set_value(encoder, 4, &(key.len() as u32));
        set_value(encoder, 5, &(write_index as u32));
        dispatch_1d(encoder, key.len());
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_attention_decode_f32(
        &self,
        encoder: &ComputeCommandEncoderRef,
        query: &GpuBuffer,
        key_cache: &GpuHalfBuffer,
        value_cache: &GpuHalfBuffer,
        output: &GpuBuffer,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        position: usize,
        capacity: usize,
        origin_logical: usize,
        origin_physical: usize,
        window: usize,
        scale: f32,
    ) {
        debug_assert_eq!(head_dim, 128);
        debug_assert_eq!(query.len(), n_heads * head_dim);
        debug_assert_eq!(output.len(), n_heads * head_dim);
        debug_assert_eq!(key_cache.len(), capacity * n_kv_heads * head_dim);
        debug_assert_eq!(value_cache.len(), key_cache.len());
        debug_assert!(capacity > 0);
        debug_assert!(n_kv_heads > 0);
        debug_assert_eq!(n_heads % n_kv_heads, 0);
        debug_assert!(origin_physical < capacity);
        let visible = if window > 0 {
            (position + 1).min(window)
        } else {
            position + 1
        };
        debug_assert!(origin_logical <= position + 1 - visible);
        self.bind(encoder, "muser_attention_decode_f32");
        encoder.set_buffer(0, Some(query.metal()), 0);
        encoder.set_buffer(1, Some(key_cache.metal()), 0);
        encoder.set_buffer(2, Some(value_cache.metal()), 0);
        encoder.set_buffer(3, Some(output.metal()), 0);
        set_value(encoder, 4, &(n_heads as u32));
        set_value(encoder, 5, &(n_kv_heads as u32));
        set_value(encoder, 6, &(head_dim as u32));
        set_value(encoder, 7, &(position as u32));
        set_value(encoder, 8, &(capacity as u32));
        set_value(encoder, 9, &(origin_logical as u32));
        set_value(encoder, 10, &(origin_physical as u32));
        set_value(encoder, 11, &(window as u32));
        set_value(encoder, 12, &scale);
        dispatch_1d(encoder, n_heads);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_attention_decode_splitk_f16(
        &self,
        encoder: &ComputeCommandEncoderRef,
        query: &GpuBuffer,
        key_cache: &GpuHalfBuffer,
        value_cache: &GpuHalfBuffer,
        partials: &GpuBuffer,
        output: &GpuBuffer,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        position: usize,
        capacity: usize,
        origin_logical: usize,
        origin_physical: usize,
        window: usize,
        scale: f32,
    ) {
        debug_assert_eq!(head_dim, 128);
        debug_assert_eq!(query.len(), n_heads * head_dim);
        debug_assert_eq!(output.len(), query.len());
        debug_assert_eq!(key_cache.len(), capacity * n_kv_heads * head_dim);
        debug_assert_eq!(value_cache.len(), key_cache.len());
        debug_assert!(capacity > 0);
        debug_assert!(n_kv_heads > 0);
        debug_assert_eq!(n_heads % n_kv_heads, 0);
        debug_assert!(origin_physical < capacity);
        let visible = if window > 0 {
            (position + 1).min(window)
        } else {
            position + 1
        };
        debug_assert!(origin_logical <= position + 1 - visible);
        let (n_workgroups, n_simdgroups) = splitk_geometry(visible);
        let partial_stride = 2 + head_dim;
        debug_assert!(partials.len() >= n_heads * n_workgroups * partial_stride);

        let producer_inputs: [&metal::ResourceRef; 3] =
            [query.metal(), key_cache.metal(), value_cache.metal()];
        encoder.memory_barrier_with_resources(&producer_inputs);
        self.bind(encoder, "muser_attention_decode_splitk_f16");
        encoder.set_buffer(0, Some(query.metal()), 0);
        encoder.set_buffer(1, Some(key_cache.metal()), 0);
        encoder.set_buffer(2, Some(value_cache.metal()), 0);
        encoder.set_buffer(3, Some(partials.metal()), 0);
        set_value(encoder, 4, &(n_heads as u32));
        set_value(encoder, 5, &(n_kv_heads as u32));
        set_value(encoder, 6, &(position as u32));
        set_value(encoder, 7, &(capacity as u32));
        set_value(encoder, 8, &(origin_logical as u32));
        set_value(encoder, 9, &(origin_physical as u32));
        set_value(encoder, 10, &(window as u32));
        set_value(encoder, 11, &(n_workgroups as u32));
        set_value(encoder, 12, &(n_simdgroups as u32));
        set_value(encoder, 13, &scale);
        let shared_floats = head_dim + n_simdgroups * partial_stride;
        let shared_bytes = shared_floats * std::mem::size_of::<f32>();
        encoder.set_threadgroup_memory_length(0, ((shared_bytes + 15) & !15) as u64);
        encoder.dispatch_thread_groups(
            MTLSize::new(n_heads as u64, n_workgroups as u64, 1),
            MTLSize::new(32, n_simdgroups as u64, 1),
        );

        // The reducer consumes producer partials in this same serial encoder.
        // Scope the dependency to that allocation instead of stalling every
        // buffer used by the 52-layer command buffer.
        let barrier_resources: [&metal::ResourceRef; 1] = [partials.metal()];
        encoder.memory_barrier_with_resources(&barrier_resources);
        self.bind(encoder, "muser_attention_decode_splitk_reduce_f32");
        encoder.set_buffer(0, Some(partials.metal()), 0);
        encoder.set_buffer(1, Some(output.metal()), 0);
        set_value(encoder, 2, &(n_heads as u32));
        set_value(encoder, 3, &(n_workgroups as u32));
        encoder.dispatch_thread_groups(MTLSize::new(n_heads as u64, 1, 1), MTLSize::new(32, 1, 1));
        let reducer_outputs: [&metal::ResourceRef; 2] = [partials.metal(), output.metal()];
        encoder.memory_barrier_with_resources(&reducer_outputs);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_kv_store_batch_f16(
        &self,
        encoder: &ComputeCommandEncoderRef,
        key: &GpuBuffer,
        value: &GpuBuffer,
        key_cache: &GpuHalfBuffer,
        value_cache: &GpuHalfBuffer,
        kv_dim: usize,
        token_count: usize,
        source_first: usize,
        source_count: usize,
        start_position: usize,
        capacity: usize,
        origin_logical: usize,
        origin_physical: usize,
        head_dim: usize,
        head_major: bool,
    ) {
        debug_assert_eq!(key.len(), token_count * kv_dim);
        debug_assert_eq!(value.len(), key.len());
        debug_assert_eq!(key_cache.len(), capacity * kv_dim);
        debug_assert_eq!(value_cache.len(), key_cache.len());
        debug_assert!(source_first + source_count <= token_count);
        debug_assert!(start_position + source_first >= origin_logical);
        debug_assert_eq!(kv_dim % head_dim, 0);
        self.bind(encoder, "muser_kv_store_batch_f16");
        encoder.set_buffer(0, Some(key.metal()), 0);
        encoder.set_buffer(1, Some(value.metal()), 0);
        encoder.set_buffer(2, Some(key_cache.metal()), 0);
        encoder.set_buffer(3, Some(value_cache.metal()), 0);
        set_value(encoder, 4, &(kv_dim as u32));
        set_value(encoder, 5, &(source_first as u32));
        set_value(encoder, 6, &(source_count as u32));
        set_value(encoder, 7, &(start_position as u32));
        set_value(encoder, 8, &(capacity as u32));
        set_value(encoder, 9, &(origin_logical as u32));
        set_value(encoder, 10, &(origin_physical as u32));
        set_value(encoder, 11, &(head_dim as u32));
        set_value(encoder, 12, &(u32::from(head_major)));
        dispatch_1d(encoder, source_count * kv_dim);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_attention_prefill_f32(
        &self,
        encoder: &ComputeCommandEncoderRef,
        query: &GpuBuffer,
        current_key: &GpuBuffer,
        current_value: &GpuBuffer,
        key_cache: &GpuHalfBuffer,
        value_cache: &GpuHalfBuffer,
        output: &GpuBuffer,
        token_count: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        start_position: usize,
        capacity: usize,
        old_origin_logical: usize,
        old_origin_physical: usize,
        old_len: usize,
        window: usize,
        scale: f32,
        head_major: bool,
    ) {
        debug_assert_eq!(head_dim, 128);
        debug_assert_eq!(query.len(), token_count * n_heads * head_dim);
        debug_assert_eq!(current_key.len(), token_count * n_kv_heads * head_dim);
        debug_assert_eq!(current_value.len(), current_key.len());
        debug_assert_eq!(output.len(), query.len());
        debug_assert_eq!(key_cache.len(), capacity * n_kv_heads * head_dim);
        debug_assert_eq!(value_cache.len(), key_cache.len());
        debug_assert_eq!(old_origin_logical + old_len, start_position);
        debug_assert!(old_origin_physical < capacity);
        self.bind(encoder, "muser_attention_prefill_flash_f16");
        encoder.set_buffer(0, Some(query.metal()), 0);
        encoder.set_buffer(1, Some(current_key.metal()), 0);
        encoder.set_buffer(2, Some(current_value.metal()), 0);
        encoder.set_buffer(3, Some(key_cache.metal()), 0);
        encoder.set_buffer(4, Some(value_cache.metal()), 0);
        encoder.set_buffer(5, Some(output.metal()), 0);
        set_value(encoder, 6, &(token_count as u32));
        set_value(encoder, 7, &(n_heads as u32));
        set_value(encoder, 8, &(n_kv_heads as u32));
        set_value(encoder, 9, &(head_dim as u32));
        set_value(encoder, 10, &(start_position as u32));
        set_value(encoder, 11, &(capacity as u32));
        set_value(encoder, 12, &(old_origin_logical as u32));
        set_value(encoder, 13, &(old_origin_physical as u32));
        set_value(encoder, 14, &(old_len as u32));
        set_value(encoder, 15, &(window as u32));
        set_value(encoder, 16, &scale);
        set_value(encoder, 17, &(u32::from(head_major)));
        encoder.set_threadgroup_memory_length(0, (4 * std::mem::size_of::<f32>()) as u64);
        encoder.dispatch_thread_groups(
            MTLSize::new(n_heads as u64, token_count as u64, 1),
            MTLSize::new(head_dim as u64, 1, 1),
        );
    }
}

fn splitk_geometry(visible: usize) -> (usize, usize) {
    let blocks = visible.div_ceil(32).max(1);
    let n_workgroups = blocks.min(crate::decode::MAX_DECODE_SPLIT_WORKGROUPS);
    let mut n_simdgroups = 1;
    while n_simdgroups < 4 && 2 * n_workgroups * n_simdgroups * 32 < visible {
        n_simdgroups *= 2;
    }
    (n_workgroups, n_simdgroups)
}

#[repr(C)]
struct GgmlMetalKargsFlashAttnExtPad {
    ne11: i32,
    ne_12_2: i32,
    ne_12_3: i32,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    nb21: u64,
    nb22: u64,
    nb23: u64,
    ne31: i32,
    ne32: i32,
    ne33: i32,
    nb31: u64,
    nb32: u64,
    nb33: u64,
}

#[repr(C)]
struct GgmlMetalKargsFlashAttnExtVec {
    ne01: i32,
    ne02: i32,
    ne03: i32,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne11: i32,
    ne_12_2: i32,
    ne_12_3: i32,
    ns10: i32,
    nb11: u64,
    nb12: u64,
    nb13: u64,
    ns20: i32,
    nb21: u64,
    nb22: u64,
    nb23: u64,
    ne31: i32,
    ne32: i32,
    ne33: i32,
    nb31: u64,
    nb32: u64,
    nb33: u64,
    ne1: i32,
    ne2: i32,
    ne3: i32,
    scale: f32,
    max_bias: f32,
    m0: f32,
    m1: f32,
    n_head_log2: i32,
    logit_softcap: f32,
}

#[repr(C)]
struct GgmlMetalKargsFlashAttnExtVecReduce {
    nrows: i32,
}

/// `ggml_metal_kargs_flash_attn_ext_blk` from the pinned llama.cpp
/// `ggml-metal-impl.h` (commit 89e0aa6): mask-block classifier arguments.
#[repr(C)]
struct GgmlMetalKargsFlashAttnExtBlk {
    ne01: i32,
    ne30: i32,
    ne31: i32,
    ne32: i32,
    ne33: i32,
    nb31: u64,
    nb32: u64,
    nb33: u64,
}

#[cfg(test)]
mod tests {
    use super::splitk_geometry;

    #[test]
    fn splitk_geometry_keeps_swa_single_simdgroup_and_scales_nope() {
        assert_eq!(splitk_geometry(1), (1, 1));
        assert_eq!(splitk_geometry(2_048), (32, 1));
        assert_eq!(splitk_geometry(8_192), (32, 4));
        assert_eq!(splitk_geometry(32_768), (32, 4));
        assert_eq!(splitk_geometry(131_008), (32, 4));
    }

    #[test]
    fn llama_flash_attn_kargs_match_ggml_c_layout() {
        assert_eq!(
            std::mem::size_of::<super::GgmlMetalKargsFlashAttnExtVec>(),
            192
        );
        assert_eq!(
            std::mem::size_of::<super::GgmlMetalKargsFlashAttnExtVecReduce>(),
            4
        );
        assert_eq!(
            std::mem::size_of::<super::GgmlMetalKargsFlashAttnExtBlk>(),
            48
        );
    }

    #[test]
    fn splitk_schedule_owns_each_logical_block_exactly_once() {
        for visible in [1usize, 31, 32, 33, 2_048, 32_769, 131_008] {
            let blocks = visible.div_ceil(32);
            let (n_workgroups, n_simdgroups) = splitk_geometry(visible);
            let mut owned = Vec::new();
            for workgroup in 0..n_workgroups {
                for simdgroup in 0..n_simdgroups {
                    let mut block = workgroup * n_simdgroups + simdgroup;
                    while block < blocks {
                        owned.push(block);
                        block += n_workgroups * n_simdgroups;
                    }
                }
            }
            owned.sort_unstable();
            assert_eq!(owned, (0..blocks).collect::<Vec<_>>());
        }
    }
}
