use std::path::PathBuf;
use std::sync::Mutex;

use crate::coreml::{
    CoreMlStatefulModel, CoreMlTensorDataType, CoreMlTensorInput, CoreMlTensorSpec,
};
use crate::dflash::{DFlashConfig, DFlashFusedAttentionInput, DFlashFusedAttentionOutput};

use super::AneFusedAttentionShardManifest;

const MASKED: f32 = -10_000.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MirrorIdentity {
    cache_identity: u64,
    cache_revision: u64,
    context_position: usize,
}

struct Scratch {
    noise: Vec<f32>,
    query_selector: Vec<f32>,
    target: Vec<f32>,
    target_mask: Vec<f32>,
    target_cos: Vec<f32>,
    target_sin: Vec<f32>,
    noise_cos: Vec<f32>,
    noise_sin: Vec<f32>,
    replay_key: Vec<f32>,
    replay_value: Vec<f32>,
    replay_mode: Vec<f32>,
    attention_mask: Vec<f32>,
    write_mask: Vec<f32>,
    output: Vec<f32>,
    valid_slots: Vec<u8>,
    mirror: Option<MirrorIdentity>,
}

pub(super) struct LoadedFusedAttention {
    spec: AneFusedAttentionShardManifest,
    model: CoreMlStatefulModel,
    scratch: Mutex<Scratch>,
}

impl LoadedFusedAttention {
    pub(super) fn load(
        path: PathBuf,
        spec: AneFusedAttentionShardManifest,
        config: &DFlashConfig,
    ) -> Result<Self, String> {
        let (block, hidden, kv_heads, head_dim) = (
            config.block_size,
            config.hidden_size,
            config.num_key_value_heads,
            config.head_dim,
        );
        let kv_width = kv_heads * head_dim;
        let half = head_dim / 2;
        let hidden_shape = [1, hidden, block, 1];
        let selector_shape = [1, 1, block, block];
        let target_mask_shape = [1, 1, block, 1];
        let rope_shape = [block, half];
        let kv_shape = [1, kv_width, block, 1];
        let mode_shape = [1, 1, 1, 1];
        let attention_mask_shape = [1, 1, block, spec.max_context + block];
        let write_mask_shape = [1, 1, spec.max_context, block];
        let inputs = vec![
            tensor("noise_hidden", &hidden_shape),
            tensor("query_selector", &selector_shape),
            tensor("target_projected", &hidden_shape),
            tensor("target_mask", &target_mask_shape),
            tensor("target_rope_cos", &rope_shape),
            tensor("target_rope_sin", &rope_shape),
            tensor("noise_rope_cos", &rope_shape),
            tensor("noise_rope_sin", &rope_shape),
            tensor("replay_target_key", &kv_shape),
            tensor("replay_target_value", &kv_shape),
            tensor("replay_mode", &mode_shape),
            tensor("attention_mask", &attention_mask_shape),
            tensor("kv_write_mask", &write_mask_shape),
        ];
        let model = CoreMlStatefulModel::load(
            &path,
            &inputs,
            "attention_target_kv_bundle",
            &[1, spec.bundle_channels, block, 1],
            CoreMlTensorDataType::Float16,
        )?;
        let max_context = spec.max_context;
        let bundle_channels = spec.bundle_channels;
        let mut query_selector = vec![0.0; block * block];
        for row in 0..block {
            query_selector[row * block + row] = 1.0;
        }
        Ok(Self {
            spec,
            model,
            scratch: Mutex::new(Scratch {
                noise: vec![0.0; block * hidden],
                query_selector,
                target: vec![0.0; block * hidden],
                target_mask: vec![0.0; block],
                target_cos: vec![0.0; block * half],
                target_sin: vec![0.0; block * half],
                noise_cos: vec![0.0; block * half],
                noise_sin: vec![0.0; block * half],
                replay_key: vec![0.0; block * kv_width],
                replay_value: vec![0.0; block * kv_width],
                replay_mode: vec![0.0],
                attention_mask: vec![MASKED; block * (max_context + block)],
                write_mask: vec![0.0; max_context * block],
                output: vec![0.0; block * bundle_channels],
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
        input: DFlashFusedAttentionInput<'_>,
    ) -> Result<DFlashFusedAttentionOutput, String> {
        self.validate(&input)?;
        let mut scratch = self.scratch.lock().map_err(|_| {
            format!(
                "ANE fused attention layer {} lock poisoned",
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
        prepare_normal(&mut scratch, &input, self.spec.max_context);
        self.predict_scratch(&mut scratch, &input)?;
        let result = unpack_output(&scratch.output, &input);
        scratch.mirror = Some(MirrorIdentity {
            cache_identity: input.cache_identity,
            cache_revision: input.cache_revision.wrapping_add(1),
            context_position: input.context_position + input.target_rows,
        });
        Ok(result)
    }

    fn validate(&self, input: &DFlashFusedAttentionInput<'_>) -> Result<(), String> {
        let q_width = input.attention_heads * input.head_dim;
        let kv_width = input.key_value_heads * input.head_dim;
        if input.block_size != 16
            || input.target_rows == 0
            || input.target_rows > input.block_size
            || input.hidden_size != 6656
            || input.attention_heads != 32
            || input.key_value_heads != 8
            || input.head_dim != 128
            || input.sink_size + input.window_size != self.spec.max_context
            || input.context_rows > self.spec.max_context
            || input.noise_normed.len() != input.block_size * input.hidden_size
            || input.target_projected.len() != input.target_rows * input.hidden_size
            || input.cached_key.len() != input.context_rows * kv_width
            || input.cached_value.len() != input.context_rows * kv_width
            || self.spec.bundle_channels != q_width + 2 * kv_width
        {
            return Err("ANE fused attention input geometry is invalid".into());
        }
        Ok(())
    }

    fn replay_cache(
        &self,
        scratch: &mut Scratch,
        input: &DFlashFusedAttentionInput<'_>,
    ) -> Result<(), String> {
        let block = input.block_size;
        let kv_width = input.key_value_heads * input.head_dim;
        for start in (0..input.context_rows).step_by(block) {
            let rows = (input.context_rows - start).min(block);
            clear_prediction(scratch);
            scratch.replay_mode[0] = 1.0;
            pack_token_major(
                &input.cached_key[start * kv_width..(start + rows) * kv_width],
                &mut scratch.replay_key,
                rows,
                block,
                kv_width,
            );
            pack_token_major(
                &input.cached_value[start * kv_width..(start + rows) * kv_width],
                &mut scratch.replay_value,
                rows,
                block,
                kv_width,
            );
            for row in 0..rows {
                scratch.target_mask[row] = 1.0;
                let logical = start + row;
                let absolute = retained_absolute_position(
                    logical,
                    input.context_rows,
                    input.context_position,
                    input.sink_size,
                );
                let slot = physical_slot(absolute, input.sink_size, input.window_size);
                scratch.write_mask[slot * block + row] = 1.0;
                scratch.valid_slots[slot] = 1;
            }
            allow_valid_attention(scratch, block, self.spec.max_context);
            fill_rope(
                &mut scratch.target_cos,
                &mut scratch.target_sin,
                rows,
                block,
                input.head_dim,
                0,
                input.rope_theta,
            );
            fill_rope(
                &mut scratch.noise_cos,
                &mut scratch.noise_sin,
                block,
                block,
                input.head_dim,
                0,
                input.rope_theta,
            );
            self.predict_scratch(scratch, input)?;
        }
        Ok(())
    }

    fn predict_scratch(
        &self,
        scratch: &mut Scratch,
        input: &DFlashFusedAttentionInput<'_>,
    ) -> Result<(), String> {
        let block = input.block_size;
        let half = input.head_dim / 2;
        let hidden_shape = [1, input.hidden_size, block, 1];
        let selector_shape = [1, 1, block, block];
        let target_mask_shape = [1, 1, block, 1];
        let rope_shape = [block, half];
        let kv_shape = [1, input.key_value_heads * input.head_dim, block, 1];
        let mode_shape = [1, 1, 1, 1];
        let attention_mask_shape = [1, 1, block, self.spec.max_context + block];
        let write_mask_shape = [1, 1, self.spec.max_context, block];
        let values = vec![
            value("noise_hidden", &hidden_shape, &scratch.noise),
            value("query_selector", &selector_shape, &scratch.query_selector),
            value("target_projected", &hidden_shape, &scratch.target),
            value("target_mask", &target_mask_shape, &scratch.target_mask),
            value("target_rope_cos", &rope_shape, &scratch.target_cos),
            value("target_rope_sin", &rope_shape, &scratch.target_sin),
            value("noise_rope_cos", &rope_shape, &scratch.noise_cos),
            value("noise_rope_sin", &rope_shape, &scratch.noise_sin),
            value("replay_target_key", &kv_shape, &scratch.replay_key),
            value("replay_target_value", &kv_shape, &scratch.replay_value),
            value("replay_mode", &mode_shape, &scratch.replay_mode),
            value(
                "attention_mask",
                &attention_mask_shape,
                &scratch.attention_mask,
            ),
            value("kv_write_mask", &write_mask_shape, &scratch.write_mask),
        ];
        self.model.predict_into(&values, &mut scratch.output)
    }
}

fn tensor<'a>(name: &'a str, shape: &'a [usize]) -> CoreMlTensorSpec<'a> {
    CoreMlTensorSpec {
        name,
        shape,
        data_type: CoreMlTensorDataType::Float16,
    }
}

fn value<'a>(name: &'a str, shape: &'a [usize], values: &'a [f32]) -> CoreMlTensorInput<'a> {
    CoreMlTensorInput {
        name,
        shape,
        values,
    }
}

fn clear_prediction(scratch: &mut Scratch) {
    scratch.noise.fill(0.0);
    scratch.target.fill(0.0);
    scratch.target_mask.fill(0.0);
    scratch.replay_key.fill(0.0);
    scratch.replay_value.fill(0.0);
    scratch.replay_mode[0] = 0.0;
    scratch.attention_mask.fill(MASKED);
    scratch.write_mask.fill(0.0);
    scratch.valid_slots.fill(0);
}

fn prepare_normal(scratch: &mut Scratch, input: &DFlashFusedAttentionInput<'_>, max: usize) {
    clear_prediction(scratch);
    let block = input.block_size;
    pack_token_major(
        input.noise_normed,
        &mut scratch.noise,
        block,
        block,
        input.hidden_size,
    );
    pack_token_major(
        input.target_projected,
        &mut scratch.target,
        input.target_rows,
        block,
        input.hidden_size,
    );
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
        scratch.target_mask[row] = 1.0;
        let absolute = input.context_position + row;
        let slot = physical_slot(absolute, input.sink_size, input.window_size);
        scratch.valid_slots[slot] = 1;
        scratch.write_mask[slot * block + row] = 1.0;
    }
    allow_valid_attention(scratch, block, max);
    fill_rope(
        &mut scratch.target_cos,
        &mut scratch.target_sin,
        input.target_rows,
        block,
        input.head_dim,
        input.context_position,
        input.rope_theta,
    );
    fill_rope(
        &mut scratch.noise_cos,
        &mut scratch.noise_sin,
        block,
        block,
        input.head_dim,
        input.context_position + input.target_rows,
        input.rope_theta,
    );
}

fn allow_valid_attention(scratch: &mut Scratch, block: usize, max: usize) {
    let stride = max + block;
    for query in 0..block {
        let row = &mut scratch.attention_mask[query * stride..(query + 1) * stride];
        for (slot, valid) in scratch.valid_slots.iter().copied().enumerate() {
            if valid != 0 {
                row[slot] = 0.0;
            }
        }
        row[max..].fill(0.0);
    }
}

fn fill_rope(
    cosine: &mut [f32],
    sine: &mut [f32],
    rows: usize,
    capacity: usize,
    head_dim: usize,
    start: usize,
    theta: f64,
) {
    let half = head_dim / 2;
    cosine.fill(1.0);
    sine.fill(0.0);
    for row in 0..rows {
        for index in 0..half {
            let angle = (start + row) as f64 / theta.powf(2.0 * index as f64 / head_dim as f64);
            let (sin, cos) = angle.sin_cos();
            cosine[row * half + index] = cos as f32;
            sine[row * half + index] = sin as f32;
        }
    }
    debug_assert_eq!(cosine.len(), capacity * half);
}

fn pack_token_major(
    source: &[f32],
    destination: &mut [f32],
    rows: usize,
    capacity: usize,
    width: usize,
) {
    for row in 0..rows {
        for column in 0..width {
            destination[column * capacity + row] = source[row * width + column];
        }
    }
}

fn unpack_output(
    source: &[f32],
    input: &DFlashFusedAttentionInput<'_>,
) -> DFlashFusedAttentionOutput {
    let block = input.block_size;
    let q_width = input.attention_heads * input.head_dim;
    let kv_width = input.key_value_heads * input.head_dim;
    let unpack = |channel_start: usize, rows: usize, width: usize| {
        let mut result = vec![0.0; rows * width];
        for row in 0..rows {
            for column in 0..width {
                result[row * width + column] = source[(channel_start + column) * block + row];
            }
        }
        result
    };
    DFlashFusedAttentionOutput {
        attention: unpack(0, block, q_width),
        target_key: unpack(q_width, input.target_rows, kv_width),
        target_value: unpack(q_width + kv_width, input.target_rows, kv_width),
    }
}

fn retained_absolute_position(logical: usize, rows: usize, position: usize, sink: usize) -> usize {
    if logical < sink {
        logical
    } else {
        position - rows + logical
    }
}

fn physical_slot(position: usize, sink: usize, window: usize) -> usize {
    if position < sink {
        position
    } else {
        sink + (position - sink) % window
    }
}

#[cfg(test)]
mod tests {
    use super::{fill_rope, pack_token_major, physical_slot, retained_absolute_position};

    #[test]
    fn runtime_bundle_packing_is_channel_major() {
        let mut packed = vec![0.0; 12];
        pack_token_major(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &mut packed, 2, 4, 3);
        assert_eq!(
            packed,
            [1.0, 4.0, 0.0, 0.0, 2.0, 5.0, 0.0, 0.0, 3.0, 6.0, 0.0, 0.0]
        );
    }

    #[test]
    fn runtime_bundle_uses_the_existing_sink_window_ring() {
        let positions = (0..5)
            .map(|logical| retained_absolute_position(logical, 5, 9, 2))
            .collect::<Vec<_>>();
        assert_eq!(positions, [0, 1, 6, 7, 8]);
        assert_eq!(
            positions
                .into_iter()
                .map(|position| physical_slot(position, 2, 3))
                .collect::<Vec<_>>(),
            [0, 1, 3, 4, 2]
        );
    }

    #[test]
    fn runtime_bundle_rope_matches_the_cpu_schedule() {
        let mut cosine = vec![0.0; 8];
        let mut sine = vec![0.0; 8];
        fill_rope(&mut cosine, &mut sine, 2, 2, 8, 7, 10_000.0);
        for row in 0..2 {
            for index in 0..4 {
                let angle = (7 + row) as f64 / 10_000.0f64.powf(2.0 * index as f64 / 8.0);
                let (expected_sin, expected_cos) = angle.sin_cos();
                assert_eq!(cosine[row * 4 + index], expected_cos as f32);
                assert_eq!(sine[row * 4 + index], expected_sin as f32);
            }
        }
    }
}
