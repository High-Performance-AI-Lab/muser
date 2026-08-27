use std::path::PathBuf;
use std::sync::Mutex;

use crate::coreml::{
    CoreMlStatefulModel, CoreMlTensorDataType, CoreMlTensorInput, CoreMlTensorSpec,
};
use crate::dflash::{DFlashConfig, DFlashStatefulAttentionInput};

use super::AneAttentionShardManifest;

const MASKED: f32 = -10_000.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MirrorIdentity {
    cache_identity: u64,
    cache_revision: u64,
    context_position: usize,
}

struct StatefulScratch {
    query: Vec<f32>,
    noise_key: Vec<f32>,
    noise_value: Vec<f32>,
    target_key: Vec<f32>,
    target_value: Vec<f32>,
    attention_mask: Vec<f32>,
    write_mask: Vec<f32>,
    output: Vec<f32>,
    valid_slots: Vec<u8>,
    mirror: Option<MirrorIdentity>,
}

pub(super) struct LoadedStatefulAttention {
    spec: AneAttentionShardManifest,
    model: CoreMlStatefulModel,
    scratch: Mutex<StatefulScratch>,
}

impl LoadedStatefulAttention {
    pub(super) fn load(
        path: PathBuf,
        spec: AneAttentionShardManifest,
        config: &DFlashConfig,
    ) -> Result<Self, String> {
        let (block, heads, kv_heads, head_dim) = (
            config.block_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
        );
        let query_shape = [1, heads, spec.query_size, head_dim];
        let kv_shape = [1, kv_heads, block, head_dim];
        let attention_mask_shape = [1, 1, spec.query_size, spec.max_context + block];
        let write_mask_shape = [1, 1, spec.max_context, block];
        let model = CoreMlStatefulModel::load(
            &path,
            &[
                CoreMlTensorSpec {
                    name: "query",
                    shape: &query_shape,
                    data_type: CoreMlTensorDataType::Float16,
                },
                CoreMlTensorSpec {
                    name: "noise_key",
                    shape: &kv_shape,
                    data_type: CoreMlTensorDataType::Float16,
                },
                CoreMlTensorSpec {
                    name: "noise_value",
                    shape: &kv_shape,
                    data_type: CoreMlTensorDataType::Float16,
                },
                CoreMlTensorSpec {
                    name: "target_key",
                    shape: &kv_shape,
                    data_type: CoreMlTensorDataType::Float16,
                },
                CoreMlTensorSpec {
                    name: "target_value",
                    shape: &kv_shape,
                    data_type: CoreMlTensorDataType::Float16,
                },
                CoreMlTensorSpec {
                    name: "attention_mask",
                    shape: &attention_mask_shape,
                    data_type: CoreMlTensorDataType::Float16,
                },
                CoreMlTensorSpec {
                    name: "kv_write_mask",
                    shape: &write_mask_shape,
                    data_type: CoreMlTensorDataType::Float16,
                },
            ],
            "attention",
            &query_shape,
            CoreMlTensorDataType::Float16,
        )?;
        let query_elements = heads * spec.query_size * head_dim;
        let kv_elements = kv_heads * block * head_dim;
        let max_context = spec.max_context;
        let attention_mask_elements = spec.query_size * attention_mask_shape[3];
        Ok(Self {
            spec,
            model,
            scratch: Mutex::new(StatefulScratch {
                query: vec![0.0; query_elements],
                noise_key: vec![0.0; kv_elements],
                noise_value: vec![0.0; kv_elements],
                target_key: vec![0.0; kv_elements],
                target_value: vec![0.0; kv_elements],
                attention_mask: vec![MASKED; attention_mask_elements],
                write_mask: vec![0.0; max_context * block],
                output: vec![0.0; query_elements],
                valid_slots: vec![0; max_context],
                mirror: None,
            }),
        })
    }

    pub(super) fn layer(&self) -> usize {
        self.spec.layer
    }

    pub(super) fn predict(
        &self,
        input: DFlashStatefulAttentionInput<'_>,
    ) -> Result<Vec<f32>, String> {
        let mut output = vec![0.0; input.block_size * input.attention_heads * input.head_dim];
        self.predict_into(input, &mut output)?;
        Ok(output)
    }

    pub(super) fn predict_into(
        &self,
        input: DFlashStatefulAttentionInput<'_>,
        output: &mut [f32],
    ) -> Result<(), String> {
        validate_input(&input, self.spec.max_context)?;
        let expected_output = input.block_size * input.attention_heads * input.head_dim;
        if output.len() != expected_output {
            return Err(format!(
                "ANE stateful attention output has {} elements, expected {expected_output}",
                output.len()
            ));
        }
        let mut scratch = self.scratch.lock().map_err(|_| {
            format!(
                "ANE stateful attention layer {} lock poisoned",
                self.spec.layer
            )
        })?;
        let expected = MirrorIdentity {
            cache_identity: input.cache_identity,
            cache_revision: input.cache_revision,
            context_position: input.context_position,
        };
        if scratch.mirror != Some(expected) {
            self.model.reset_state()?;
            scratch.mirror = None;
            self.replay_cache(&mut scratch, &input)?;
            scratch.mirror = Some(expected);
        }
        let combine_target_write = input.target_rows <= input.block_size;
        if !combine_target_write {
            self.install_target_rows(&mut scratch, &input)?;
        }
        for query_start in (0..input.block_size).step_by(self.spec.query_size) {
            let write_target = combine_target_write && query_start == 0;
            prepare_prediction(
                &mut scratch,
                &input,
                self.spec.max_context,
                self.spec.query_size,
                query_start,
                write_target,
            );
            predict_model(
                &self.model,
                &mut scratch,
                &input,
                self.spec.max_context,
                self.spec.query_size,
            )?;
            head_major_to_token_major_at(
                &scratch.output,
                output,
                query_start,
                self.spec.query_size,
                input.attention_heads,
                input.head_dim,
            );
        }
        // The CPU shadow advances only after every assistant layer succeeds.
        // Anticipating the next revision avoids replay on the normal path; a
        // later-layer failure leaves the old revision and therefore forces a
        // reset/replay on the next attempt.
        scratch.mirror = Some(MirrorIdentity {
            cache_identity: input.cache_identity,
            cache_revision: input.cache_revision.wrapping_add(1),
            context_position: input.context_position + input.target_rows,
        });
        Ok(())
    }

    fn install_target_rows(
        &self,
        scratch: &mut StatefulScratch,
        input: &DFlashStatefulAttentionInput<'_>,
    ) -> Result<(), String> {
        for start in (0..input.target_rows).step_by(input.block_size) {
            let rows = (input.target_rows - start).min(input.block_size);
            prepare_target_tile(scratch, input, start, rows);
            allow_noise(
                &mut scratch.attention_mask,
                self.spec.query_size,
                input.block_size,
                self.spec.max_context,
            );
            predict_model(
                &self.model,
                scratch,
                input,
                self.spec.max_context,
                self.spec.query_size,
            )?;
        }
        Ok(())
    }

    fn replay_cache(
        &self,
        scratch: &mut StatefulScratch,
        input: &DFlashStatefulAttentionInput<'_>,
    ) -> Result<(), String> {
        if input.context_rows == 0 {
            return Ok(());
        }
        let width = input.key_value_heads * input.head_dim;
        for start in (0..input.context_rows).step_by(input.block_size) {
            let rows = (input.context_rows - start).min(input.block_size);
            prepare_state_update(scratch, rows, input.block_size);
            pack_token_heads(
                &input.cached_key[start * width..(start + rows) * width],
                &mut scratch.target_key,
                rows,
                input.block_size,
                input.key_value_heads,
                input.head_dim,
            );
            pack_token_heads(
                &input.cached_value[start * width..(start + rows) * width],
                &mut scratch.target_value,
                rows,
                input.block_size,
                input.key_value_heads,
                input.head_dim,
            );
            for row in 0..rows {
                let logical = start + row;
                let absolute = retained_absolute_position(
                    logical,
                    input.context_rows,
                    input.context_position,
                    input.sink_size,
                );
                let slot = physical_slot(absolute, input.sink_size, input.window_size);
                scratch.write_mask[slot * input.block_size + row] = 1.0;
            }
            // Replay output is discarded, but leave the noise block visible so
            // fused SDPA never receives an entirely masked row.
            allow_noise(
                &mut scratch.attention_mask,
                self.spec.query_size,
                input.block_size,
                self.spec.max_context,
            );
            predict_model(
                &self.model,
                scratch,
                input,
                self.spec.max_context,
                self.spec.query_size,
            )?;
        }
        Ok(())
    }
}

fn prepare_target_tile(
    scratch: &mut StatefulScratch,
    input: &DFlashStatefulAttentionInput<'_>,
    start: usize,
    rows: usize,
) {
    let width = input.key_value_heads * input.head_dim;
    prepare_state_update(scratch, rows, input.block_size);
    pack_token_heads(
        &input.target_key[start * width..(start + rows) * width],
        &mut scratch.target_key,
        rows,
        input.block_size,
        input.key_value_heads,
        input.head_dim,
    );
    pack_token_heads(
        &input.target_value[start * width..(start + rows) * width],
        &mut scratch.target_value,
        rows,
        input.block_size,
        input.key_value_heads,
        input.head_dim,
    );
    for row in 0..rows {
        let absolute = input.context_position + start + row;
        let slot = physical_slot(absolute, input.sink_size, input.window_size);
        scratch.write_mask[slot * input.block_size + row] = 1.0;
    }
}

fn validate_input(
    input: &DFlashStatefulAttentionInput<'_>,
    max_context: usize,
) -> Result<(), String> {
    let q_width = input.attention_heads * input.head_dim;
    let kv_width = input.key_value_heads * input.head_dim;
    if input.block_size != 16
        || input.target_rows == 0
        || input.attention_heads != 32
        || input.key_value_heads != 8
        || input.head_dim != 128
        || input.sink_size + input.window_size != max_context
        || input.context_rows > max_context
        || input.query.len() != input.block_size * q_width
        || input.noise_key.len() != input.block_size * kv_width
        || input.noise_value.len() != input.block_size * kv_width
        || input.target_key.len() != input.target_rows * kv_width
        || input.target_value.len() != input.target_rows * kv_width
        || input.cached_key.len() != input.context_rows * kv_width
        || input.cached_value.len() != input.context_rows * kv_width
    {
        return Err("ANE stateful attention input geometry is invalid".into());
    }
    Ok(())
}

fn prepare_prediction(
    scratch: &mut StatefulScratch,
    input: &DFlashStatefulAttentionInput<'_>,
    max_context: usize,
    query_size: usize,
    query_start: usize,
    write_target: bool,
) {
    // Query and noise tensors are dense block-sized inputs and are overwritten
    // completely below.  Only masks and a short target tail can retain stale
    // values.  Avoiding the former whole-scratch clear removes roughly one
    // million redundant f32 stores from every five-layer draft round.
    clear_masks(scratch);
    pack_token_heads(
        &input.query[query_start * input.attention_heads * input.head_dim
            ..(query_start + query_size) * input.attention_heads * input.head_dim],
        &mut scratch.query,
        query_size,
        query_size,
        input.attention_heads,
        input.head_dim,
    );
    pack_token_heads(
        input.noise_key,
        &mut scratch.noise_key,
        input.block_size,
        input.block_size,
        input.key_value_heads,
        input.head_dim,
    );
    pack_token_heads(
        input.noise_value,
        &mut scratch.noise_value,
        input.block_size,
        input.block_size,
        input.key_value_heads,
        input.head_dim,
    );
    if write_target {
        if input.target_rows < input.block_size {
            scratch.target_key.fill(0.0);
            scratch.target_value.fill(0.0);
        }
        pack_token_heads(
            input.target_key,
            &mut scratch.target_key,
            input.target_rows,
            input.block_size,
            input.key_value_heads,
            input.head_dim,
        );
        pack_token_heads(
            input.target_value,
            &mut scratch.target_value,
            input.target_rows,
            input.block_size,
            input.key_value_heads,
            input.head_dim,
        );
    } else {
        // T=4 compatibility makes three query-only predictions after the
        // state-writing call. Keep their unused target input deterministic;
        // v7's T=16 route never enters this branch.
        scratch.target_key.fill(0.0);
        scratch.target_value.fill(0.0);
    }
    scratch.valid_slots.fill(0);
    for logical in 0..input.context_rows {
        let absolute = retained_absolute_position(
            logical,
            input.context_rows,
            input.context_position,
            input.sink_size,
        );
        scratch.valid_slots[physical_slot(absolute, input.sink_size, input.window_size)] = 1;
    }
    for row in 0..input.target_rows {
        let absolute = input.context_position + row;
        let slot = physical_slot(absolute, input.sink_size, input.window_size);
        if write_target {
            scratch.write_mask[slot * input.block_size + row] = 1.0;
        }
        scratch.valid_slots[slot] = 1;
    }
    let stride = max_context + input.block_size;
    for query in 0..query_size {
        let row = &mut scratch.attention_mask[query * stride..(query + 1) * stride];
        for (slot, &is_valid) in scratch.valid_slots.iter().enumerate() {
            if is_valid != 0 {
                row[slot] = 0.0;
            }
        }
    }
    allow_noise(
        &mut scratch.attention_mask,
        query_size,
        input.block_size,
        max_context,
    );
}

fn clear_masks(scratch: &mut StatefulScratch) {
    scratch.attention_mask.fill(MASKED);
    scratch.write_mask.fill(0.0);
}

fn prepare_state_update(scratch: &mut StatefulScratch, rows: usize, capacity: usize) {
    scratch.query.fill(0.0);
    scratch.noise_key.fill(0.0);
    scratch.noise_value.fill(0.0);
    if rows < capacity {
        scratch.target_key.fill(0.0);
        scratch.target_value.fill(0.0);
    }
    clear_masks(scratch);
}

fn allow_noise(mask: &mut [f32], queries: usize, block: usize, max_context: usize) {
    let stride = max_context + block;
    for query in 0..queries {
        mask[query * stride + max_context..(query + 1) * stride].fill(0.0);
    }
}

fn predict_model(
    model: &CoreMlStatefulModel,
    scratch: &mut StatefulScratch,
    input: &DFlashStatefulAttentionInput<'_>,
    max_context: usize,
    query_size: usize,
) -> Result<(), String> {
    let query_shape = [1, input.attention_heads, query_size, input.head_dim];
    let kv_shape = [1, input.key_value_heads, input.block_size, input.head_dim];
    let attention_mask_shape = [1, 1, query_size, max_context + input.block_size];
    let write_mask_shape = [1, 1, max_context, input.block_size];
    let StatefulScratch {
        query,
        noise_key,
        noise_value,
        target_key,
        target_value,
        attention_mask,
        write_mask,
        output,
        ..
    } = scratch;
    model.predict_into(
        &[
            CoreMlTensorInput {
                name: "query",
                shape: &query_shape,
                values: query,
            },
            CoreMlTensorInput {
                name: "noise_key",
                shape: &kv_shape,
                values: noise_key,
            },
            CoreMlTensorInput {
                name: "noise_value",
                shape: &kv_shape,
                values: noise_value,
            },
            CoreMlTensorInput {
                name: "target_key",
                shape: &kv_shape,
                values: target_key,
            },
            CoreMlTensorInput {
                name: "target_value",
                shape: &kv_shape,
                values: target_value,
            },
            CoreMlTensorInput {
                name: "attention_mask",
                shape: &attention_mask_shape,
                values: attention_mask,
            },
            CoreMlTensorInput {
                name: "kv_write_mask",
                shape: &write_mask_shape,
                values: write_mask,
            },
        ],
        output,
    )
}

fn pack_token_heads(
    source: &[f32],
    destination: &mut [f32],
    rows: usize,
    capacity: usize,
    heads: usize,
    head_dim: usize,
) {
    for token in 0..rows {
        for head in 0..heads {
            let source_offset = (token * heads + head) * head_dim;
            let destination_offset = (head * capacity + token) * head_dim;
            destination[destination_offset..destination_offset + head_dim]
                .copy_from_slice(&source[source_offset..source_offset + head_dim]);
        }
    }
}

fn head_major_to_token_major_at(
    source: &[f32],
    destination: &mut [f32],
    token_start: usize,
    rows: usize,
    heads: usize,
    head_dim: usize,
) {
    for token in 0..rows {
        for head in 0..heads {
            let source_offset = (head * rows + token) * head_dim;
            let destination_offset = ((token_start + token) * heads + head) * head_dim;
            destination[destination_offset..destination_offset + head_dim]
                .copy_from_slice(&source[source_offset..source_offset + head_dim]);
        }
    }
}

fn retained_absolute_position(
    logical_row: usize,
    context_rows: usize,
    context_position: usize,
    sink_size: usize,
) -> usize {
    let sink_rows = context_rows.min(sink_size);
    if logical_row < sink_rows {
        logical_row
    } else {
        let tail_rows = context_rows - sink_rows;
        context_position - tail_rows + logical_row - sink_rows
    }
}

fn physical_slot(absolute_position: usize, sink_size: usize, window_size: usize) -> usize {
    if absolute_position < sink_size {
        absolute_position
    } else {
        sink_size + (absolute_position - sink_size) % window_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_head_layout_round_trips() {
        let source = (0..24).map(|value| value as f32).collect::<Vec<_>>();
        let mut packed = vec![0.0; 24];
        pack_token_heads(&source, &mut packed, 3, 3, 2, 4);
        assert_eq!(
            packed,
            [
                0., 1., 2., 3., 8., 9., 10., 11., 16., 17., 18., 19., 4., 5., 6., 7., 12., 13.,
                14., 15., 20., 21., 22., 23.
            ]
        );
        let mut round_trip = vec![0.0; 24];
        head_major_to_token_major_at(&packed, &mut round_trip, 0, 3, 2, 4);
        assert_eq!(round_trip, source);
    }

    #[test]
    fn compact_tail_maps_to_state_ring_by_absolute_position() {
        let positions = (0..5)
            .map(|row| retained_absolute_position(row, 5, 9, 2))
            .collect::<Vec<_>>();
        assert_eq!(positions, [0, 1, 6, 7, 8]);
        let slots = positions
            .iter()
            .map(|&position| physical_slot(position, 2, 3))
            .collect::<Vec<_>>();
        assert_eq!(slots, [0, 1, 3, 4, 2]);
    }

    #[test]
    fn query_chunks_write_state_once_and_pack_disjoint_rows() {
        let block = 16;
        let heads = 32;
        let kv_heads = 8;
        let head_dim = 128;
        let query_size = 4;
        let max_context = 1088;
        let q_width = heads * head_dim;
        let kv_width = kv_heads * head_dim;
        let query = (0..block * q_width)
            .map(|index| index as f32)
            .collect::<Vec<_>>();
        let noise = vec![0.0; block * kv_width];
        let target = vec![1.0; block * kv_width];
        let cached = Vec::new();
        let input = DFlashStatefulAttentionInput {
            query: &query,
            noise_key: &noise,
            noise_value: &noise,
            target_key: &target,
            target_value: &target,
            cached_key: &cached,
            cached_value: &cached,
            block_size: block,
            target_rows: block,
            attention_heads: heads,
            key_value_heads: kv_heads,
            head_dim,
            context_position: 0,
            context_rows: 0,
            sink_size: 64,
            window_size: 1024,
            cache_identity: 7,
            cache_revision: 9,
        };
        let mut scratch = StatefulScratch {
            query: vec![0.0; query_size * q_width],
            noise_key: vec![0.0; block * kv_width],
            noise_value: vec![0.0; block * kv_width],
            target_key: vec![0.0; block * kv_width],
            target_value: vec![0.0; block * kv_width],
            attention_mask: vec![MASKED; query_size * (max_context + block)],
            write_mask: vec![0.0; max_context * block],
            output: vec![0.0; query_size * q_width],
            valid_slots: vec![0; max_context],
            mirror: None,
        };
        prepare_prediction(&mut scratch, &input, max_context, query_size, 0, true);
        assert_eq!(
            scratch
                .write_mask
                .iter()
                .filter(|&&value| value == 1.0)
                .count(),
            16
        );
        assert_eq!(scratch.query[0], query[0]);
        assert_eq!(scratch.query[head_dim], query[q_width]);
        assert_eq!(scratch.query[query_size * head_dim], query[head_dim]);

        prepare_prediction(&mut scratch, &input, max_context, query_size, 4, false);
        assert!(scratch.write_mask.iter().all(|&value| value == 0.0));
        assert!(scratch.target_key.iter().all(|&value| value == 0.0));
        assert_eq!(scratch.query[0], query[4 * q_width]);
        assert_eq!(scratch.query[head_dim], query[5 * q_width]);
        assert_eq!(
            scratch.query[query_size * head_dim],
            query[4 * q_width + head_dim]
        );
    }

    #[test]
    fn target_tiles_stream_beyond_model_block_capacity() {
        let block = 16;
        let heads = 2;
        let head_dim = 2;
        let width = heads * head_dim;
        let target_key = (0..33 * width)
            .map(|index| index as f32)
            .collect::<Vec<_>>();
        let target_value = target_key.clone();
        let empty = Vec::new();
        let input = DFlashStatefulAttentionInput {
            query: &empty,
            noise_key: &empty,
            noise_value: &empty,
            target_key: &target_key,
            target_value: &target_value,
            cached_key: &empty,
            cached_value: &empty,
            block_size: block,
            target_rows: 33,
            attention_heads: heads,
            key_value_heads: heads,
            head_dim,
            context_position: 100,
            context_rows: 0,
            sink_size: 64,
            window_size: 1024,
            cache_identity: 1,
            cache_revision: 1,
        };
        let mut scratch = StatefulScratch {
            query: vec![],
            noise_key: vec![],
            noise_value: vec![],
            target_key: vec![0.0; block * width],
            target_value: vec![0.0; block * width],
            attention_mask: vec![],
            write_mask: vec![0.0; 1088 * block],
            output: vec![],
            valid_slots: vec![0; 1088],
            mirror: None,
        };

        prepare_target_tile(&mut scratch, &input, 16, 16);
        assert_eq!(scratch.target_key[0], 64.0);
        assert_eq!(scratch.target_key[block * head_dim], 66.0);
        assert_eq!(scratch.write_mask[116 * block], 1.0);
        assert_eq!(scratch.write_mask[131 * block + 15], 1.0);

        prepare_target_tile(&mut scratch, &input, 32, 1);
        assert_eq!(scratch.target_key[0], 128.0);
        assert_eq!(scratch.target_key[block * head_dim], 130.0);
        assert_eq!(
            scratch
                .write_mask
                .iter()
                .filter(|&&value| value == 1.0)
                .count(),
            1
        );
        assert_eq!(scratch.write_mask[132 * block], 1.0);
    }
}
