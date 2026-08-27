//! Muse Glimmer vision encoder, ported from the official llama.cpp mtmd graph
//! at `0b1bad14ff204627636aeb1de22ddcd5acb859d4`.
//!
//! The CPU path is deliberately straightforward and auditable: Lanczos image
//! resize, 50 transformer blocks, exact-erf GELU, 2-D RoPE, direct sparse
//! spatial windows, pixel shuffle, and the three projection matrices. It is
//! the oracle against which the Metal graph is qualified; it is not intended
//! to be the shipping performance path.

use std::path::Path;

use image::imageops::FilterType;

use crate::config::MuseConfigError;
use crate::gguf::GgufFile;
use crate::weights::{matmul, MuseWeights, TensorView};

const OFFICIAL_BLOCKS: usize = 50;
const CHANNELS: usize = 3;
const MERGE: usize = 2;
const SPARSE_FACTOR: usize = 4;
const MAX_IMAGE_TOKENS: usize = 4_096;
const ROPE_BASE: f32 = 10_000.0;

#[derive(Debug, thiserror::Error)]
pub enum VisionError {
    #[error(transparent)]
    Config(#[from] MuseConfigError),
    #[error("image decode failed: {0}")]
    Image(#[from] image::ImageError),
    #[error("invalid Muse vision artifact: {0}")]
    Invalid(String),
    #[error("Metal vision bridge: {0}")]
    Metal(String),
}

#[derive(Debug, Clone)]
pub struct VisionConfig {
    pub embedding_dim: usize,
    pub intermediate_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub patch_size: usize,
    pub norm_eps: f32,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub output_dim: usize,
    pub position_grid: usize,
}

impl VisionConfig {
    fn from_gguf(gguf: &GgufFile) -> Result<Self, VisionError> {
        let projector = gguf
            .meta_str("clip.projector_type")
            .or_else(|| gguf.meta_str("clip.vision.projector_type"));
        if projector != Some("muse-glimmer") {
            return Err(VisionError::Invalid(format!(
                "projector type is {projector:?}, expected muse-glimmer"
            )));
        }
        let required = |key: &str| {
            gguf.meta_u32(key)
                .map(|value| value as usize)
                .ok_or_else(|| VisionError::Invalid(format!("missing {key}")))
        };
        let embedding_dim = required("clip.vision.embedding_length")?;
        let intermediate_dim = required("clip.vision.feed_forward_length")?;
        let n_layers = required("clip.vision.block_count")?;
        let n_heads = required("clip.vision.attention.head_count")?;
        let patch_size = required("clip.vision.patch_size")?;
        let norm_eps = gguf
            .meta_f32("clip.vision.attention.layer_norm_epsilon")
            .ok_or_else(|| VisionError::Invalid("missing vision layer-norm epsilon".into()))?;
        let mean = gguf
            .meta_f32_array("clip.vision.image_mean")
            .ok_or_else(|| VisionError::Invalid("missing image mean".into()))?;
        let std = gguf
            .meta_f32_array("clip.vision.image_std")
            .ok_or_else(|| VisionError::Invalid("missing image std".into()))?;
        if n_layers != OFFICIAL_BLOCKS
            || embedding_dim % n_heads != 0
            || patch_size == 0
            || mean.len() < 3
            || std.len() < 3
            || std[..3].contains(&0.0)
        {
            return Err(VisionError::Invalid(format!(
                "geometry layers={n_layers} embd={embedding_dim} heads={n_heads} patch={patch_size}"
            )));
        }
        let position = gguf
            .tensor("v.position_embd.weight")
            .ok_or_else(|| VisionError::Invalid("missing learned position embedding".into()))?;
        let position_count = position.shape.get(1).copied().unwrap_or(0) as usize;
        let position_grid = (position_count as f64).sqrt() as usize;
        if position.shape.first().copied() != Some(embedding_dim as u64)
            || position_grid * position_grid != position_count
        {
            return Err(VisionError::Invalid(format!(
                "position embedding shape {:?} is not [{embedding_dim}, square]",
                position.shape
            )));
        }
        let output = gguf
            .tensor("mm.2.weight")
            .ok_or_else(|| VisionError::Invalid("missing mm.2.weight".into()))?;
        let output_dim = output.shape.get(1).copied().unwrap_or(0) as usize;
        Ok(Self {
            embedding_dim,
            intermediate_dim,
            n_layers,
            n_heads,
            patch_size,
            norm_eps,
            image_mean: [mean[0], mean[1], mean[2]],
            image_std: [std[0], std[1], std[2]],
            output_dim,
            position_grid,
        })
    }

    pub fn head_dim(&self) -> usize {
        self.embedding_dim / self.n_heads
    }
}

pub struct VisionModel {
    pub config: VisionConfig,
    weights: MuseWeights,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    metal: Option<crate::metal::vision::MetalVisionBridge>,
}

#[derive(Debug, Clone)]
pub struct PreprocessedImage {
    pub width: usize,
    pub height: usize,
    /// Channel-major normalized pixels `[channel][y][x]`.
    pub pixels: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpatialPlan {
    pub grid_w: usize,
    pub grid_h: usize,
    pub sparse_permutation: Vec<usize>,
    pub inverse_permutation: Vec<usize>,
    pub position_w: Vec<usize>,
    pub position_h: Vec<usize>,
    pub window_ranges: Vec<std::ops::Range<usize>>,
    pub downsample_permutation: Vec<usize>,
}

impl VisionModel {
    pub fn load(path: &Path) -> Result<Self, VisionError> {
        let gguf = GgufFile::parse_path(path)
            .map_err(|error| VisionError::Invalid(format!("parse mmproj: {error}")))?;
        let config = VisionConfig::from_gguf(&gguf)?;
        let weights = MuseWeights::open(path, &gguf)?;
        let model = Self {
            config,
            weights,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            metal: None,
        };
        model.validate_weights()?;
        Ok(model)
    }

    /// Load the auditable CPU oracle and attach the pinned upstream mtmd
    /// graph as the shipping Metal route. The same GGUF is parsed by both
    /// implementations so geometry and decoder width are checked before the
    /// first request reaches the accelerator.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn load_metal(path: &Path, bridge_path: &Path) -> Result<Self, VisionError> {
        let mut model = Self::load(path)?;
        model.metal = Some(
            crate::metal::vision::MetalVisionBridge::load(bridge_path, path)
                .map_err(VisionError::Metal)?,
        );
        Ok(model)
    }

    pub fn route_identity(&self) -> &'static str {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if self.metal.is_some() {
            return "mtmd-metal:muser-mtmd-muse-vision-v1";
        }
        "cpu-oracle"
    }

    pub fn preprocess_bytes(&self, encoded: &[u8]) -> Result<PreprocessedImage, VisionError> {
        let image = image::load_from_memory(encoded)?.to_rgb8();
        let (target_w, target_h) = target_size(
            image.width() as usize,
            image.height() as usize,
            self.config.patch_size * MERGE,
            MAX_IMAGE_TOKENS,
        );
        let resized = image::imageops::resize(
            &image,
            target_w as u32,
            target_h as u32,
            FilterType::Lanczos3,
        );
        let plane = target_w * target_h;
        let mut pixels = vec![0.0; CHANNELS * plane];
        for (index, pixel) in resized.pixels().enumerate() {
            for channel in 0..CHANNELS {
                pixels[channel * plane + index] = (pixel[channel] as f32 / 255.0
                    - self.config.image_mean[channel])
                    / self.config.image_std[channel];
            }
        }
        Ok(PreprocessedImage {
            width: target_w,
            height: target_h,
            pixels,
        })
    }

    /// Decoder positions occupied by this preprocessed image after the
    /// official 2x2 pixel shuffle. This is geometry-only: a remote-prefill
    /// caller can construct exact position witnesses without running the
    /// fifty-block projector locally.
    pub fn projected_token_count(&self, image: &PreprocessedImage) -> Result<usize, VisionError> {
        let patch = self.config.patch_size;
        if !image.width.is_multiple_of(patch * MERGE)
            || !image.height.is_multiple_of(patch * MERGE)
            || image.pixels.len() != CHANNELS * image.width * image.height
        {
            return Err(VisionError::Invalid(
                "preprocessed image geometry is invalid".into(),
            ));
        }
        Ok((image.width / patch) * (image.height / patch) / (MERGE * MERGE))
    }

    pub fn encode(&self, image: &PreprocessedImage) -> Result<Vec<Vec<f32>>, VisionError> {
        let patch = self.config.patch_size;
        if !image.width.is_multiple_of(patch * MERGE)
            || !image.height.is_multiple_of(patch * MERGE)
            || image.pixels.len() != CHANNELS * image.width * image.height
        {
            return Err(VisionError::Invalid(
                "preprocessed image geometry is invalid".into(),
            ));
        }
        let plan = spatial_plan(
            image.width / patch,
            image.height / patch,
            self.config.position_grid,
            MERGE,
        );
        let tokens = plan.grid_w * plan.grid_h;
        let mut patches = vec![0.0; tokens * CHANNELS * patch * patch];
        extract_patches(image, patch, &mut patches);
        let patch_weight = self.weights.view_2d(
            "v.patch_embd.weight",
            CHANNELS * patch * patch,
            self.config.embedding_dim,
        )?;
        let mut hidden = vec![0.0; tokens * self.config.embedding_dim];
        matmul(&patch_weight, &patches, tokens, &mut hidden);
        add_optional_bias(
            &self.weights,
            "v.patch_embd.bias",
            &mut hidden,
            self.config.embedding_dim,
        )?;
        add_position_embedding(
            &self.weights.view("v.position_embd.weight")?,
            self.config.position_grid,
            plan.grid_w,
            plan.grid_h,
            &mut hidden,
        );
        hidden = gather_rows(&hidden, self.config.embedding_dim, &plan.sparse_permutation);
        optional_layer_norm(
            &self.weights,
            "v.pre_ln.weight",
            "v.pre_ln.bias",
            &mut hidden,
            self.config.embedding_dim,
            self.config.norm_eps,
        )?;

        for layer in 0..self.config.n_layers {
            self.forward_block(layer, &plan, &mut hidden)?;
        }
        optional_layer_norm(
            &self.weights,
            "v.post_ln.weight",
            "v.post_ln.bias",
            &mut hidden,
            self.config.embedding_dim,
            self.config.norm_eps,
        )?;
        hidden = gather_rows(
            &hidden,
            self.config.embedding_dim,
            &plan.inverse_permutation,
        );
        hidden = pixel_shuffle(
            &hidden,
            self.config.embedding_dim,
            &plan.downsample_permutation,
        );
        hidden = project(
            &self.weights.view("mm.0.weight")?,
            &hidden,
            hidden.len() / (self.config.embedding_dim * MERGE * MERGE),
            true,
        );
        let output_tokens = hidden.len() / self.weights.view("mm.0.weight")?.n_out;
        hidden = project(
            &self.weights.view("mm.1.weight")?,
            &hidden,
            output_tokens,
            true,
        );
        hidden = project(
            &self.weights.view("mm.2.weight")?,
            &hidden,
            output_tokens,
            false,
        );
        Ok(hidden
            .chunks_exact(self.config.output_dim)
            .map(<[f32]>::to_vec)
            .collect())
    }

    /// Encode through the configured shipping route. The CPU model remains
    /// the qualification oracle; Metal consumes the original RGB image so
    /// preprocessing is performed by the exact pinned upstream mtmd graph.
    pub fn encode_accelerated(
        &self,
        encoded: &[u8],
        preprocessed: &PreprocessedImage,
    ) -> Result<Vec<Vec<f32>>, VisionError> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Some(metal) = self.metal.as_ref() {
            let original = image::load_from_memory(encoded)?.to_rgb8();
            let output_tokens = self.projected_token_count(preprocessed)?;
            return metal
                .encode_rgb(
                    original.as_raw(),
                    original.width() as usize,
                    original.height() as usize,
                    output_tokens,
                    self.config.output_dim,
                )
                .map_err(VisionError::Metal);
        }
        let _ = encoded;
        self.encode(preprocessed)
    }

    /// Return pinned-upstream normalized pixels in Muser's channel-major
    /// layout. This is qualification evidence for the ≤1/255 preprocessing
    /// gate and is not on the serving hot path.
    pub fn preprocess_upstream(
        &self,
        encoded: &[u8],
        expected: &PreprocessedImage,
    ) -> Result<PreprocessedImage, VisionError> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Some(metal) = self.metal.as_ref() {
            let original = image::load_from_memory(encoded)?.to_rgb8();
            let (width, height, interleaved) = metal
                .preprocess_rgb(
                    original.as_raw(),
                    original.width() as usize,
                    original.height() as usize,
                    expected.pixels.len(),
                )
                .map_err(VisionError::Metal)?;
            if width != expected.width || height != expected.height {
                return Err(VisionError::Metal(format!(
                    "upstream preprocessing emitted {width}x{height}, expected {}x{}",
                    expected.width, expected.height
                )));
            }
            let plane = width * height;
            let mut pixels = vec![0.0f32; interleaved.len()];
            for index in 0..plane {
                for channel in 0..CHANNELS {
                    pixels[channel * plane + index] = interleaved[index * CHANNELS + channel];
                }
            }
            return Ok(PreprocessedImage {
                width,
                height,
                pixels,
            });
        }
        let _ = encoded;
        Ok(expected.clone())
    }

    fn forward_block(
        &self,
        layer: usize,
        plan: &SpatialPlan,
        hidden: &mut Vec<f32>,
    ) -> Result<(), VisionError> {
        let dim = self.config.embedding_dim;
        let tokens = hidden.len() / dim;
        let residual = hidden.clone();
        let mut normed = hidden.clone();
        layer_norm_named(
            &self.weights,
            &format!("v.blk.{layer}.ln1.weight"),
            &format!("v.blk.{layer}.ln1.bias"),
            &mut normed,
            dim,
            self.config.norm_eps,
        )?;
        let mut q = matmul_named(
            &self.weights,
            &format!("v.blk.{layer}.attn_q.weight"),
            &normed,
            tokens,
        )?;
        let mut k = matmul_named(
            &self.weights,
            &format!("v.blk.{layer}.attn_k.weight"),
            &normed,
            tokens,
        )?;
        let v = matmul_named(
            &self.weights,
            &format!("v.blk.{layer}.attn_v.weight"),
            &normed,
            tokens,
        )?;
        add_optional_bias(
            &self.weights,
            &format!("v.blk.{layer}.attn_q.bias"),
            &mut q,
            dim,
        )?;
        add_optional_bias(
            &self.weights,
            &format!("v.blk.{layer}.attn_k.bias"),
            &mut k,
            dim,
        )?;
        let mut v = v;
        add_optional_bias(
            &self.weights,
            &format!("v.blk.{layer}.attn_v.bias"),
            &mut v,
            dim,
        )?;
        rope_2d(
            &mut q,
            &mut k,
            self.config.n_heads,
            self.config.head_dim(),
            &plan.position_w,
            &plan.position_h,
        );
        let global = layer + 1 == self.config.n_layers || (layer + 1).is_multiple_of(SPARSE_FACTOR);
        let ranges = if global {
            std::iter::once(0..tokens).collect()
        } else {
            plan.window_ranges.clone()
        };
        let attended = attention(
            &q,
            &k,
            &v,
            self.config.n_heads,
            self.config.head_dim(),
            &ranges,
        );
        let mut projected = matmul_named(
            &self.weights,
            &format!("v.blk.{layer}.attn_out.weight"),
            &attended,
            tokens,
        )?;
        add_optional_bias(
            &self.weights,
            &format!("v.blk.{layer}.attn_out.bias"),
            &mut projected,
            dim,
        )?;
        for (value, residual) in projected.iter_mut().zip(residual.iter()) {
            *value += *residual;
        }
        let ffn_residual = projected.clone();
        layer_norm_named(
            &self.weights,
            &format!("v.blk.{layer}.ln2.weight"),
            &format!("v.blk.{layer}.ln2.bias"),
            &mut projected,
            dim,
            self.config.norm_eps,
        )?;
        let mut ffn = matmul_named(
            &self.weights,
            &format!("v.blk.{layer}.ffn_up.weight"),
            &projected,
            tokens,
        )?;
        add_optional_bias(
            &self.weights,
            &format!("v.blk.{layer}.ffn_up.bias"),
            &mut ffn,
            self.config.intermediate_dim,
        )?;
        ffn.iter_mut().for_each(|value| *value = gelu_erf(*value));
        let mut output = matmul_named(
            &self.weights,
            &format!("v.blk.{layer}.ffn_down.weight"),
            &ffn,
            tokens,
        )?;
        add_optional_bias(
            &self.weights,
            &format!("v.blk.{layer}.ffn_down.bias"),
            &mut output,
            dim,
        )?;
        for (value, residual) in output.iter_mut().zip(ffn_residual.iter()) {
            *value += *residual;
        }
        *hidden = output;
        Ok(())
    }

    fn validate_weights(&self) -> Result<(), VisionError> {
        for name in [
            "v.patch_embd.weight",
            "v.position_embd.weight",
            "mm.0.weight",
            "mm.1.weight",
            "mm.2.weight",
        ] {
            self.weights.view(name)?;
        }
        for layer in 0..self.config.n_layers {
            for suffix in [
                "ln1.weight",
                "ln2.weight",
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_out.weight",
                "ffn_up.weight",
                "ffn_down.weight",
            ] {
                self.weights.view(&format!("v.blk.{layer}.{suffix}"))?;
            }
        }
        Ok(())
    }
}

pub fn target_size(
    image_w: usize,
    image_h: usize,
    patch_hw: usize,
    max_tokens: usize,
) -> (usize, usize) {
    let mut nph = image_h as f64 / patch_hw as f64;
    let mut npw = image_w as f64 / patch_hw as f64;
    let ratio = if nph > 0.0 { npw / nph } else { 1.0 };
    if nph * npw > max_tokens as f64 {
        nph = (max_tokens as f64 / ratio).sqrt();
        npw = nph * ratio;
    }
    let hs = [nph.floor() as usize, nph.ceil() as usize];
    let ws = [npw.floor() as usize, npw.ceil() as usize];
    let target_ar = image_h as f64 / image_w.max(1) as f64;
    let mut best: Option<(usize, usize, f64)> = None;
    for height in hs {
        for width in ws {
            if height == 0 || width == 0 || height * width > max_tokens {
                continue;
            }
            let distance = (height as f64 / width as f64 - target_ar).abs();
            if best.is_none_or(|(bh, bw, bd)| {
                distance < bd || (distance == bd && height * width > bh * bw)
            }) {
                best = Some((height, width, distance));
            }
        }
    }
    let (height, width, _) = best.unwrap_or((
        nph.round().max(1.0) as usize,
        npw.round().max(1.0) as usize,
        0.0,
    ));
    (width * patch_hw, height * patch_hw)
}

pub fn spatial_plan(grid_w: usize, grid_h: usize, window: usize, merge: usize) -> SpatialPlan {
    assert!(grid_w > 0 && grid_h > 0 && window > 0 && merge > 0);
    assert_eq!(grid_w % merge, 0);
    assert_eq!(grid_h % merge, 0);
    let tokens = grid_w * grid_h;
    let mut sparse_permutation = Vec::with_capacity(tokens);
    let mut window_ranges = Vec::new();
    for wy in 0..grid_h.div_ceil(window) {
        for wx in 0..grid_w.div_ceil(window) {
            let start = sparse_permutation.len();
            for dy in 0..window {
                for dx in 0..window {
                    let y = wy * window + dy;
                    let x = wx * window + dx;
                    if y < grid_h && x < grid_w {
                        sparse_permutation.push(y * grid_w + x);
                    }
                }
            }
            if sparse_permutation.len() > start {
                window_ranges.push(start..sparse_permutation.len());
            }
        }
    }
    let mut inverse_permutation = vec![0; tokens];
    let mut position_w = vec![0; tokens];
    let mut position_h = vec![0; tokens];
    for (permuted, &original) in sparse_permutation.iter().enumerate() {
        inverse_permutation[original] = permuted;
        position_w[permuted] = original % grid_w + 1;
        position_h[permuted] = original / grid_w + 1;
    }
    let mut downsample_permutation = Vec::with_capacity(tokens);
    for oy in 0..grid_h / merge {
        for ox in 0..grid_w / merge {
            for ry in 0..merge {
                for rx in 0..merge {
                    downsample_permutation.push((oy * merge + ry) * grid_w + ox * merge + rx);
                }
            }
        }
    }
    SpatialPlan {
        grid_w,
        grid_h,
        sparse_permutation,
        inverse_permutation,
        position_w,
        position_h,
        window_ranges,
        downsample_permutation,
    }
}

fn extract_patches(image: &PreprocessedImage, patch: usize, output: &mut [f32]) {
    let grid_w = image.width / patch;
    let grid_h = image.height / patch;
    let plane = image.width * image.height;
    let patch_dim = CHANNELS * patch * patch;
    for gy in 0..grid_h {
        for gx in 0..grid_w {
            let token = gy * grid_w + gx;
            let mut index = token * patch_dim;
            for channel in 0..CHANNELS {
                for py in 0..patch {
                    for px in 0..patch {
                        let source =
                            channel * plane + (gy * patch + py) * image.width + gx * patch + px;
                        output[index] = image.pixels[source];
                        index += 1;
                    }
                }
            }
        }
    }
}

fn add_position_embedding(
    position: &TensorView<'_>,
    base: usize,
    width: usize,
    height: usize,
    hidden: &mut [f32],
) {
    let dim = position.n_in;
    let source = position.to_f32();
    let scale_x = width as f32 / base as f32;
    let scale_y = height as f32 / base as f32;
    for y in 0..height {
        let fy = (y as f32 + 0.5) / scale_y - 0.5;
        let y0 = fy.floor().clamp(0.0, (base - 1) as f32) as usize;
        let y1 = (y0 + 1).min(base - 1);
        let dy = (fy - y0 as f32).clamp(0.0, 1.0);
        for x in 0..width {
            let fx = (x as f32 + 0.5) / scale_x - 0.5;
            let x0 = fx.floor().clamp(0.0, (base - 1) as f32) as usize;
            let x1 = (x0 + 1).min(base - 1);
            let dx = (fx - x0 as f32).clamp(0.0, 1.0);
            let destination = (y * width + x) * dim;
            for channel in 0..dim {
                let a = source[(y0 * base + x0) * dim + channel];
                let b = source[(y0 * base + x1) * dim + channel];
                let c = source[(y1 * base + x0) * dim + channel];
                let d = source[(y1 * base + x1) * dim + channel];
                hidden[destination + channel] += a * (1.0 - dx) * (1.0 - dy)
                    + b * dx * (1.0 - dy)
                    + c * (1.0 - dx) * dy
                    + d * dx * dy;
            }
        }
    }
}

fn gather_rows(input: &[f32], dim: usize, permutation: &[usize]) -> Vec<f32> {
    let mut output = vec![0.0; permutation.len() * dim];
    for (destination, &source) in permutation.iter().enumerate() {
        output[destination * dim..(destination + 1) * dim]
            .copy_from_slice(&input[source * dim..(source + 1) * dim]);
    }
    output
}

fn pixel_shuffle(input: &[f32], dim: usize, permutation: &[usize]) -> Vec<f32> {
    let stride = MERGE * MERGE;
    let outputs = permutation.len() / stride;
    let mut output = vec![0.0; outputs * dim * stride];
    for token in 0..outputs {
        for channel in 0..dim {
            for spatial in 0..stride {
                output[token * dim * stride + channel * stride + spatial] =
                    input[permutation[token * stride + spatial] * dim + channel];
            }
        }
    }
    output
}

fn rope_2d(
    q: &mut [f32],
    k: &mut [f32],
    heads: usize,
    head_dim: usize,
    position_w: &[usize],
    position_h: &[usize],
) {
    let tokens = position_w.len();
    let half = head_dim / 2;
    for token in 0..tokens {
        for head in 0..heads {
            let base = (token * heads + head) * head_dim;
            for (offset, position) in [(0, position_w[token]), (half, position_h[token])] {
                for pair in (0..half).step_by(2) {
                    let frequency = ROPE_BASE.powf(-(pair as f32) / half as f32);
                    let angle = position as f32 * frequency;
                    let (sin, cos) = angle.sin_cos();
                    for values in [&mut *q, &mut *k] {
                        let a = values[base + offset + pair];
                        let b = values[base + offset + pair + 1];
                        values[base + offset + pair] = a * cos - b * sin;
                        values[base + offset + pair + 1] = a * sin + b * cos;
                    }
                }
            }
        }
    }
}

fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    heads: usize,
    head_dim: usize,
    ranges: &[std::ops::Range<usize>],
) -> Vec<f32> {
    let mut output = vec![0.0; q.len()];
    let scale = 1.0 / (head_dim as f32).sqrt();
    for range in ranges {
        for query in range.clone() {
            for head in 0..heads {
                let qbase = (query * heads + head) * head_dim;
                let mut maximum = f32::NEG_INFINITY;
                for key in range.clone() {
                    let kbase = (key * heads + head) * head_dim;
                    let score = q[qbase..qbase + head_dim]
                        .iter()
                        .zip(&k[kbase..kbase + head_dim])
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        * scale;
                    maximum = maximum.max(score);
                }
                let mut denominator = 0.0;
                for key in range.clone() {
                    let kbase = (key * heads + head) * head_dim;
                    let score = q[qbase..qbase + head_dim]
                        .iter()
                        .zip(&k[kbase..kbase + head_dim])
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                        * scale;
                    let weight = (score - maximum).exp();
                    denominator += weight;
                    let vbase = (key * heads + head) * head_dim;
                    for dim in 0..head_dim {
                        output[qbase + dim] += weight * v[vbase + dim];
                    }
                }
                for dim in 0..head_dim {
                    output[qbase + dim] /= denominator;
                }
            }
        }
    }
    output
}

fn matmul_named(
    weights: &MuseWeights,
    name: &str,
    input: &[f32],
    tokens: usize,
) -> Result<Vec<f32>, VisionError> {
    let weight = weights.view(name)?;
    let mut output = vec![0.0; tokens * weight.n_out];
    matmul(&weight, input, tokens, &mut output);
    Ok(output)
}

fn project(weight: &TensorView<'_>, input: &[f32], tokens: usize, gelu: bool) -> Vec<f32> {
    let mut output = vec![0.0; tokens * weight.n_out];
    matmul(weight, input, tokens, &mut output);
    if gelu {
        output
            .iter_mut()
            .for_each(|value| *value = gelu_erf(*value));
    }
    output
}

fn gelu_erf(value: f32) -> f32 {
    0.5 * value * (1.0 + libm::erff(value * std::f32::consts::FRAC_1_SQRT_2))
}

fn layer_norm_named(
    weights: &MuseWeights,
    weight_name: &str,
    bias_name: &str,
    values: &mut [f32],
    dim: usize,
    epsilon: f32,
) -> Result<(), VisionError> {
    let weight = weights.f32_vec(weight_name)?;
    let bias = weights
        .contains(bias_name)
        .then(|| weights.f32_vec(bias_name))
        .transpose()?;
    layer_norm(values, dim, epsilon, &weight, bias.as_deref());
    Ok(())
}

fn optional_layer_norm(
    weights: &MuseWeights,
    weight_name: &str,
    bias_name: &str,
    values: &mut [f32],
    dim: usize,
    epsilon: f32,
) -> Result<(), VisionError> {
    if weights.contains(weight_name) {
        layer_norm_named(weights, weight_name, bias_name, values, dim, epsilon)?;
    }
    Ok(())
}

fn layer_norm(values: &mut [f32], dim: usize, epsilon: f32, weight: &[f32], bias: Option<&[f32]>) {
    for row in values.chunks_exact_mut(dim) {
        let mean = row.iter().sum::<f32>() / dim as f32;
        let variance = row.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / dim as f32;
        let inverse = 1.0 / (variance + epsilon).sqrt();
        for index in 0..dim {
            row[index] = (row[index] - mean) * inverse * weight[index]
                + bias.map_or(0.0, |bias| bias[index]);
        }
    }
}

fn add_optional_bias(
    weights: &MuseWeights,
    name: &str,
    values: &mut [f32],
    dim: usize,
) -> Result<(), VisionError> {
    if weights.contains(name) {
        let bias = weights.f32_vec(name)?;
        for row in values.chunks_exact_mut(dim) {
            for (value, bias) in row.iter_mut().zip(bias.iter()) {
                *value += *bias;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aspect_preserving_grid_matches_official_candidate_search() {
        assert_eq!(target_size(1920, 1080, 28, 4096), (1932, 1092));
        let (width, height) = target_size(640, 640, 28, 4096);
        assert_eq!(width, height);
        assert_eq!(width % 28, 0);
    }

    #[test]
    fn spatial_windows_and_inverse_are_exact_bijections() {
        let plan = spatial_plan(70, 38, 32, 2);
        assert_eq!(plan.sparse_permutation.len(), 70 * 38);
        assert_eq!(plan.window_ranges.len(), 6);
        for original in 0..70 * 38 {
            assert_eq!(
                plan.sparse_permutation[plan.inverse_permutation[original]],
                original
            );
        }
        assert_eq!(plan.downsample_permutation.len(), 70 * 38);
    }

    #[test]
    fn pixel_shuffle_uses_channel_outer_spatial_inner_order() {
        let input = (0..16).map(|value| value as f32).collect::<Vec<_>>();
        let plan = spatial_plan(2, 2, 2, 2);
        assert_eq!(
            pixel_shuffle(&input, 4, &plan.downsample_permutation),
            vec![
                0.0, 4.0, 8.0, 12.0, 1.0, 5.0, 9.0, 13.0, 2.0, 6.0, 10.0, 14.0, 3.0, 7.0, 11.0,
                15.0,
            ]
        );
    }

    #[test]
    fn exact_erf_gelu_pins_known_values() {
        assert!((gelu_erf(0.0) - 0.0).abs() < 1e-7);
        assert!((gelu_erf(1.0) - 0.841_344_7).abs() < 1e-6);
        assert!((gelu_erf(-1.0) + 0.158_655_26).abs() < 1e-6);
    }
}
