//! Memory-mapped Muse Glimmer weights and the batched matmul used by the
//! reference forward pass.
//!
//! Weights are never materialized as f32. Each output row is a contiguous run
//! of quantized blocks, so a row is read once from DRAM and dotted against all
//! `n_tokens` activation vectors while it is still resident in L1 — prefill of
//! T tokens therefore costs roughly the same DRAM traffic as a single token.

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;
use rayon::prelude::*;

use crate::gguf::{GgmlType, GgufFile};
use crate::quant::{
    dequant_block, dequant_nvfp4_row, dot_nvfp4_a16_q8_f32, dot_nvfp4_w4a4_f32, dot_q4_0_f32,
    dot_q4_k_f32, dot_q5_k_f32, dot_q6_k_f32, dot_q8_f32, f16_to_f32,
};

use super::config::MuseConfigError;

pub const NVFP4_SCALE_SUFFIX: &str = ".nvfp4_scale";
pub const NVFP4_SCALE2_SUFFIX: &str = ".nvfp4_scale2";
pub const NVFP4_INPUT_SCALE_INV_SUFFIX: &str = ".nvfp4_input_scale_inv";

#[derive(Debug, Clone, Copy)]
pub struct Nvfp4AuxView<'a> {
    pub block_scales: &'a [u8],
    pub scale2: f32,
    /// Compressed-tensors' stored input-global-scale divisor. When present,
    /// activations are dynamically quantized in groups of 16 before the dot.
    pub input_scale_inv: Option<f32>,
}

/// A single mapped tensor: raw bytes plus the geometry needed to index rows.
#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a> {
    pub raw: &'a [u8],
    pub dtype: GgmlType,
    /// Contiguous (inner) dimension — the input dim of a projection.
    pub n_in: usize,
    /// Row count — the output dim of a projection. 1 for a vector.
    pub n_out: usize,
    pub nvfp4: Option<Nvfp4AuxView<'a>>,
}

#[derive(Debug, Clone, Copy)]
pub struct TensorLayout {
    pub file_offset: usize,
    pub byte_len: usize,
    pub dtype: GgmlType,
    pub n_in: usize,
    pub n_out: usize,
    pub nvfp4_scale_offset: Option<usize>,
    pub nvfp4_scale_len: usize,
    pub nvfp4_scale2: f32,
    pub nvfp4_input_scale_inv: Option<f32>,
}

impl TensorView<'_> {
    /// Bytes occupied by one row of `n_in` elements.
    pub fn row_bytes(&self) -> usize {
        let be = self.dtype.block_elements();
        self.n_in.div_ceil(be) * self.dtype.block_size()
    }

    /// Dequantize the whole tensor to f32 (only used for norm vectors and
    /// single embedding rows — never for a projection matrix).
    pub fn to_f32(&self) -> Vec<f32> {
        let mut out = vec![0.0f32; self.n_in * self.n_out];
        for r in 0..self.n_out {
            dequant_row(self, r, &mut out[r * self.n_in..(r + 1) * self.n_in]);
        }
        out
    }
}

/// Dequantize row `r` of `t` into `out` (length `t.n_in`).
pub fn dequant_row(t: &TensorView<'_>, r: usize, out: &mut [f32]) {
    if t.dtype == GgmlType::NVFP4_E2M1 {
        let aux = t.nvfp4.expect("validated NVFP4 tensor has companions");
        let packed_row_bytes = t.n_in / 2;
        let scale_row_bytes = t.n_in / 16;
        dequant_nvfp4_row(
            &t.raw[r * packed_row_bytes..(r + 1) * packed_row_bytes],
            &aux.block_scales[r * scale_row_bytes..(r + 1) * scale_row_bytes],
            aux.scale2,
            out,
        );
        return;
    }
    let be = t.dtype.block_elements();
    let bs = t.dtype.block_size();
    let rb = t.row_bytes();
    let base = r * rb;
    let mut scratch = vec![0.0f32; be.max(1)];
    let n_blocks = t.n_in.div_ceil(be);
    for b in 0..n_blocks {
        let s = base + b * bs;
        dequant_block(t.dtype, &t.raw[s..s + bs], &mut scratch);
        let off = b * be;
        let n = be.min(t.n_in - off);
        out[off..off + n].copy_from_slice(&scratch[..n]);
    }
}

/// Dot one quantized row against an f32 vector of length `n`.
fn dot_row(t: &TensorView<'_>, r: usize, row: &[u8], x: &[f32], n: usize) -> f32 {
    match t.dtype {
        GgmlType::NVFP4_E2M1 => {
            let aux = t.nvfp4.expect("validated NVFP4 tensor has companions");
            let scale_row_bytes = n / 16;
            let scales = &aux.block_scales[r * scale_row_bytes..(r + 1) * scale_row_bytes];
            if let Some(input_scale_inv) = aux.input_scale_inv {
                dot_nvfp4_w4a4_f32(row, scales, aux.scale2, input_scale_inv, x)
            } else {
                dot_nvfp4_a16_q8_f32(row, scales, aux.scale2, x)
            }
        }
        GgmlType::Q4_K => dot_q4_k_f32(row, x, n),
        GgmlType::Q5_K => dot_q5_k_f32(row, x, n),
        GgmlType::Q6_K => dot_q6_k_f32(row, x, n),
        GgmlType::Q8_0 => dot_q8_f32(row, x, n),
        GgmlType::Q4_0 => dot_q4_0_f32(row, x, n),
        GgmlType::F32 => {
            let mut acc = 0.0f32;
            for i in 0..n {
                let b = &row[i * 4..i * 4 + 4];
                acc += f32::from_le_bytes([b[0], b[1], b[2], b[3]]) * x[i];
            }
            acc
        }
        GgmlType::F16 => {
            let mut acc = 0.0f32;
            for i in 0..n {
                let bits = u16::from_le_bytes([row[i * 2], row[i * 2 + 1]]);
                acc += f16_to_f32(bits) * x[i];
            }
            acc
        }
        other => {
            // Generic path: stream-dequant the row, then dot.
            let be = other.block_elements();
            let bs = other.block_size();
            let mut scratch = vec![0.0f32; be];
            let mut acc = 0.0f32;
            for b in 0..n.div_ceil(be) {
                dequant_block(other, &row[b * bs..(b + 1) * bs], &mut scratch);
                let off = b * be;
                let k = be.min(n - off);
                for i in 0..k {
                    acc += scratch[i] * x[off + i];
                }
            }
            acc
        }
    }
}

/// All mapped tensors of a Muse Glimmer checkpoint.
#[derive(Clone)]
pub struct MuseWeights {
    mmap: Arc<Mmap>,
    index: HashMap<String, (usize, usize, GgmlType, Vec<u64>)>,
}

impl MuseWeights {
    pub fn open(path: &Path, gguf: &GgufFile) -> Result<Self, MuseConfigError> {
        let file = File::open(path)
            .map_err(|e| MuseConfigError::Geometry(format!("open {}: {e}", path.display())))?;
        // SAFETY: the checkpoint is a read-only immutable input for the
        // lifetime of this process; we never write through the mapping.
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| MuseConfigError::Geometry(format!("mmap {}: {e}", path.display())))?;

        let mut index = HashMap::with_capacity(gguf.tensors.len());
        for t in &gguf.tensors {
            let start = (gguf.data_offset + t.offset) as usize;
            let n_elem: usize = t.shape.iter().product::<u64>() as usize;
            let be = t.dtype.block_elements();
            let len = n_elem.div_ceil(be) * t.dtype.block_size();
            if start + len > mmap.len() {
                return Err(MuseConfigError::Geometry(format!(
                    "tensor {} runs past end of file ({} + {} > {})",
                    t.name,
                    start,
                    len,
                    mmap.len()
                )));
            }
            index.insert(t.name.clone(), (start, len, t.dtype, t.shape.clone()));
        }
        let weights = Self {
            mmap: Arc::new(mmap),
            index,
        };
        let activation_precision = gguf.meta_str("muser.activation_precision").unwrap_or("f16");
        if !matches!(activation_precision, "f16" | "nvfp4") {
            return Err(MuseConfigError::Geometry(format!(
                "unsupported muser.activation_precision={activation_precision}"
            )));
        }
        for tensor in &gguf.tensors {
            if tensor.dtype == GgmlType::NVFP4_E2M1 {
                let (_, _, _, input_scale_inv) = weights.nvfp4_aux(&tensor.name, &tensor.shape)?;
                if (activation_precision == "nvfp4") != input_scale_inv.is_some() {
                    return Err(MuseConfigError::Geometry(format!(
                        "NVFP4 tensor {} does not match muser.activation_precision={activation_precision}",
                        tensor.name
                    )));
                }
            }
        }
        Ok(weights)
    }

    fn nvfp4_aux(
        &self,
        name: &str,
        shape: &[u64],
    ) -> Result<(usize, usize, f32, Option<f32>), MuseConfigError> {
        let n_in = shape.first().copied().unwrap_or(0) as usize;
        let n_out = shape.get(1).copied().unwrap_or(0) as usize;
        if shape.len() != 2 || n_in == 0 || n_out == 0 || !n_in.is_multiple_of(16) {
            return Err(MuseConfigError::Geometry(format!(
                "NVFP4 tensor {name} must be a nonempty 2-D matrix with n_in divisible by 16"
            )));
        }
        let scale_name = format!("{name}{NVFP4_SCALE_SUFFIX}");
        let (scale_start, scale_len, scale_dtype, scale_shape) = self
            .index
            .get(&scale_name)
            .ok_or_else(|| MuseConfigError::MissingTensor(scale_name.clone()))?;
        let expected_scale_shape = vec![(n_in / 16) as u64, n_out as u64];
        if *scale_dtype != GgmlType::F8_E4M3FN || *scale_shape != expected_scale_shape {
            return Err(MuseConfigError::Geometry(format!(
                "NVFP4 scale tensor {scale_name} must be F8_E4M3FN {expected_scale_shape:?}, got {scale_dtype:?} {scale_shape:?}"
            )));
        }
        let scales = &self.mmap[*scale_start..*scale_start + *scale_len];
        if scales.iter().any(|byte| matches!(byte, 0x7f | 0xff)) {
            return Err(MuseConfigError::Geometry(format!(
                "NVFP4 scale tensor {scale_name} contains E4M3FN NaN"
            )));
        }
        let scale2_name = format!("{name}{NVFP4_SCALE2_SUFFIX}");
        let (scale2_start, scale2_len, scale2_dtype, scale2_shape) =
            self.index
                .get(&scale2_name)
                .ok_or_else(|| MuseConfigError::MissingTensor(scale2_name.clone()))?;
        if *scale2_dtype != GgmlType::F32 || scale2_shape.as_slice() != [1] || *scale2_len != 4 {
            return Err(MuseConfigError::Geometry(format!(
                "NVFP4 tensor scale {scale2_name} must be scalar F32"
            )));
        }
        let bytes: [u8; 4] = self.mmap[*scale2_start..*scale2_start + 4]
            .try_into()
            .expect("validated scale2 length");
        let scale2 = f32::from_le_bytes(bytes);
        if !scale2.is_finite() || scale2 <= 0.0 {
            return Err(MuseConfigError::Geometry(format!(
                "NVFP4 tensor scale {scale2_name} must be finite and positive"
            )));
        }
        let input_scale_name = format!("{name}{NVFP4_INPUT_SCALE_INV_SUFFIX}");
        let input_scale_inv = match self.index.get(&input_scale_name) {
            Some((start, len, dtype, shape)) => {
                if *dtype != GgmlType::F32 || shape.as_slice() != [1] || *len != 4 {
                    return Err(MuseConfigError::Geometry(format!(
                        "NVFP4 activation scale {input_scale_name} must be scalar F32"
                    )));
                }
                let bytes: [u8; 4] = self.mmap[*start..*start + 4]
                    .try_into()
                    .expect("validated input scale length");
                let value = f32::from_le_bytes(bytes);
                if !value.is_finite() || value <= 0.0 {
                    return Err(MuseConfigError::Geometry(format!(
                        "NVFP4 activation scale {input_scale_name} must be finite and positive"
                    )));
                }
                Some(value)
            }
            None => None,
        };
        Ok((*scale_start, *scale_len, scale2, input_scale_inv))
    }

    pub fn view(&self, name: &str) -> Result<TensorView<'_>, MuseConfigError> {
        let (start, len, dtype, shape) = self
            .index
            .get(name)
            .ok_or_else(|| MuseConfigError::MissingTensor(name.to_string()))?;
        let n_in = shape.first().copied().unwrap_or(1) as usize;
        let n_out = shape.get(1).copied().unwrap_or(1) as usize;
        let nvfp4 = if *dtype == GgmlType::NVFP4_E2M1 {
            let (scale_start, scale_len, scale2, input_scale_inv) = self.nvfp4_aux(name, shape)?;
            Some(Nvfp4AuxView {
                block_scales: &self.mmap[scale_start..scale_start + scale_len],
                scale2,
                input_scale_inv,
            })
        } else {
            None
        };
        Ok(TensorView {
            raw: &self.mmap[*start..*start + *len],
            dtype: *dtype,
            n_in,
            n_out,
            nvfp4,
        })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// Reinterpret a tensor as a row-major 2-D matrix after validating its
    /// complete element count. Vision patch convolutions use GGML's native
    /// `[patch_w, patch_h, channels, output]` shape but are contiguous rows
    /// of flattened patches for the CPU oracle.
    pub fn view_2d(
        &self,
        name: &str,
        n_in: usize,
        n_out: usize,
    ) -> Result<TensorView<'_>, MuseConfigError> {
        let (_, _, _, shape) = self
            .index
            .get(name)
            .ok_or_else(|| MuseConfigError::MissingTensor(name.to_string()))?;
        let elements = shape.iter().try_fold(1usize, |product, dimension| {
            product.checked_mul(*dimension as usize)
        });
        if elements != Some(n_in.saturating_mul(n_out)) {
            return Err(MuseConfigError::Geometry(format!(
                "tensor {name} has shape {shape:?}, cannot reinterpret as [{n_in}, {n_out}]"
            )));
        }
        let mut view = self.view(name)?;
        view.n_in = n_in;
        view.n_out = n_out;
        Ok(view)
    }

    pub fn layout(&self, name: &str) -> Result<TensorLayout, MuseConfigError> {
        let (file_offset, byte_len, dtype, shape) = self
            .index
            .get(name)
            .ok_or_else(|| MuseConfigError::MissingTensor(name.to_string()))?;
        let (nvfp4_scale_offset, nvfp4_scale_len, nvfp4_scale2, nvfp4_input_scale_inv) =
            if *dtype == GgmlType::NVFP4_E2M1 {
                let (offset, len, scale2, input_scale_inv) = self.nvfp4_aux(name, shape)?;
                (Some(offset), len, scale2, input_scale_inv)
            } else {
                (None, 0, 1.0, None)
            };
        Ok(TensorLayout {
            file_offset: *file_offset,
            byte_len: *byte_len,
            dtype: *dtype,
            n_in: shape.first().copied().unwrap_or(1) as usize,
            n_out: shape.get(1).copied().unwrap_or(1) as usize,
            nvfp4_scale_offset,
            nvfp4_scale_len,
            nvfp4_scale2,
            nvfp4_input_scale_inv,
        })
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn mapped_file(&self) -> Arc<Mmap> {
        Arc::clone(&self.mmap)
    }

    pub fn f32_vec(&self, name: &str) -> Result<Vec<f32>, MuseConfigError> {
        Ok(self.view(name)?.to_f32())
    }

    /// One row of a 2-D tensor as f32 (used for token embedding lookup).
    pub fn row_f32(&self, name: &str, row: usize) -> Result<Vec<f32>, MuseConfigError> {
        let t = self.view(name)?;
        let mut out = vec![0.0f32; t.n_in];
        dequant_row(&t, row, &mut out);
        Ok(out)
    }
}

/// Batched matmul: `out[t][r] = dot(W_row_r, x[t])`.
///
/// `x` is token-major with stride `w.n_in`; `out` is token-major with stride
/// `w.n_out`. Parallel over row chunks so each weight row is fetched from DRAM
/// once and reused across all `n_tokens` dots.
pub fn matmul(w: &TensorView<'_>, x: &[f32], n_tokens: usize, out: &mut [f32]) {
    let n_in = w.n_in;
    let n_out = w.n_out;
    debug_assert_eq!(x.len(), n_tokens * n_in);
    debug_assert_eq!(out.len(), n_tokens * n_out);
    let rb = w.row_bytes();

    // Row-major [n_out][n_tokens] scratch, then transposed into `out`.
    let mut rt = vec![0.0f32; n_out * n_tokens];
    rt.par_chunks_mut(n_tokens)
        .enumerate()
        .for_each(|(r, dst)| {
            let row = &w.raw[r * rb..(r + 1) * rb];
            for (t, slot) in dst.iter_mut().enumerate() {
                *slot = dot_row(w, r, row, &x[t * n_in..(t + 1) * n_in], n_in);
            }
        });

    for r in 0..n_out {
        for t in 0..n_tokens {
            out[t * n_out + r] = rt[r * n_tokens + t];
        }
    }
}

/// Matmul restricted to a subset of output rows — used to score only the rows
/// of the LM head we care about. `out` is `[n_tokens][rows.len()]`.
pub fn matmul_rows(
    w: &TensorView<'_>,
    x: &[f32],
    n_tokens: usize,
    rows: &[usize],
    out: &mut [f32],
) {
    let n_in = w.n_in;
    let rb = w.row_bytes();
    let mut rt = vec![0.0f32; rows.len() * n_tokens];
    rt.par_chunks_mut(n_tokens)
        .enumerate()
        .for_each(|(i, dst)| {
            let r = rows[i];
            let row = &w.raw[r * rb..(r + 1) * rb];
            for (t, slot) in dst.iter_mut().enumerate() {
                *slot = dot_row(w, r, row, &x[t * n_in..(t + 1) * n_in], n_in);
            }
        });
    for (i, _) in rows.iter().enumerate() {
        for t in 0..n_tokens {
            out[t * rows.len() + i] = rt[i * n_tokens + t];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{GgufFile, MetadataValue, TensorInfo};

    fn nvfp4_fixture(
        include_scales: bool,
        input_scale_inv: Option<f32>,
    ) -> (tempfile::TempDir, GgufFile) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("fixture.gguf");
        let packed = [0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe];
        let mut bytes = packed.to_vec();
        bytes.push(0x38);
        bytes.extend_from_slice(&0.25f32.to_le_bytes());
        if let Some(value) = input_scale_inv {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        std::fs::write(&path, bytes).unwrap();
        let mut tensors = vec![TensorInfo {
            name: "matrix".into(),
            shape: vec![16, 1],
            dtype: GgmlType::NVFP4_E2M1,
            offset: 0,
        }];
        if include_scales {
            tensors.extend([
                TensorInfo {
                    name: format!("matrix{NVFP4_SCALE_SUFFIX}"),
                    shape: vec![1, 1],
                    dtype: GgmlType::F8_E4M3FN,
                    offset: 8,
                },
                TensorInfo {
                    name: format!("matrix{NVFP4_SCALE2_SUFFIX}"),
                    shape: vec![1],
                    dtype: GgmlType::F32,
                    offset: 9,
                },
            ]);
            if input_scale_inv.is_some() {
                tensors.push(TensorInfo {
                    name: format!("matrix{NVFP4_INPUT_SCALE_INV_SUFFIX}"),
                    shape: vec![1],
                    dtype: GgmlType::F32,
                    offset: 13,
                });
            }
        }
        let mut metadata = HashMap::new();
        if input_scale_inv.is_some() {
            metadata.insert(
                "muser.activation_precision".into(),
                MetadataValue::Str("nvfp4".into()),
            );
        }
        (
            directory,
            GgufFile {
                version: 3,
                metadata,
                tensors,
                data_offset: 0,
            },
        )
    }

    #[test]
    fn native_nvfp4_view_binds_companions_and_dequantizes_exactly() {
        let (directory, gguf) = nvfp4_fixture(true, None);
        let weights = MuseWeights::open(&directory.path().join("fixture.gguf"), &gguf).unwrap();
        let view = weights.view("matrix").unwrap();
        assert_eq!(view.row_bytes(), 8);
        assert_eq!(view.nvfp4.unwrap().block_scales, [0x38]);
        assert_eq!(view.nvfp4.unwrap().scale2.to_bits(), 0.25f32.to_bits());
        assert!(view.nvfp4.unwrap().input_scale_inv.is_none());
        let expected: [f32; 16] = [
            0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, -0.0, -0.125, -0.25, -0.375, -0.5, -0.75,
            -1.0, -1.5,
        ];
        let mut decoded = [0.0f32; 16];
        dequant_row(&view, 0, &mut decoded);
        for (actual, expected) in decoded.iter().zip(expected) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn native_nvfp4_loader_fails_closed_without_companions() {
        let (directory, gguf) = nvfp4_fixture(false, None);
        let error = MuseWeights::open(&directory.path().join("fixture.gguf"), &gguf)
            .err()
            .expect("missing scale must fail");
        assert!(
            matches!(error, MuseConfigError::MissingTensor(name) if name == "matrix.nvfp4_scale")
        );
    }

    #[test]
    fn native_w4a4_view_binds_input_scale_and_fails_closed_on_mixed_metadata() {
        let (directory, gguf) = nvfp4_fixture(true, Some(43.75));
        let weights = MuseWeights::open(&directory.path().join("fixture.gguf"), &gguf).unwrap();
        assert_eq!(
            weights
                .view("matrix")
                .unwrap()
                .nvfp4
                .unwrap()
                .input_scale_inv,
            Some(43.75)
        );

        let (directory, mut gguf) = nvfp4_fixture(true, None);
        gguf.metadata.insert(
            "muser.activation_precision".into(),
            MetadataValue::Str("nvfp4".into()),
        );
        let error = MuseWeights::open(&directory.path().join("fixture.gguf"), &gguf)
            .err()
            .expect("mixed W4A4 artifact must fail");
        assert!(
            matches!(error, MuseConfigError::Geometry(message) if message.contains("activation_precision"))
        );
    }
}
