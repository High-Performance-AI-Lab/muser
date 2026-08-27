use super::metadata::read_metadata_value;
use super::reader::{read_string, read_u32, read_u64};
use super::{GgmlType, GgufError, GgufFile, MetadataValue, TensorInfo, GGUF_MAGIC};
use std::collections::{HashMap, HashSet};
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

const MAX_METADATA_ITEMS: u64 = 100_000;
const MAX_TENSORS: u64 = 1_000_000;
const MAX_TENSOR_DIMS: u64 = 32;

impl GgufFile {
    /// Parse a GGUF file from disk.
    ///
    /// Tensor metadata and offsets are correctness-critical executable input.
    /// They are therefore read from the model itself, never from the legacy
    /// `<model>.ferrite-gguf-cache-v1` sidecar whose length+mtime stamp could
    /// not authenticate its contents.
    pub fn parse_path(path: &Path) -> Result<Self, GgufError> {
        let mut f = std::fs::File::open(path)?;
        Self::parse(&mut f)
    }

    /// Parse a GGUF file from a seekable reader.
    /// Only reads header + metadata + tensor descriptors (not tensor data).
    pub fn parse<R: Read + Seek>(r: &mut R) -> Result<Self, GgufError> {
        let start = r.stream_position()?;
        let file_end = r.seek(SeekFrom::End(0))?;
        r.seek(SeekFrom::Start(start))?;
        let magic = read_u32(r)?;
        if magic != GGUF_MAGIC {
            return Err(GgufError::BadMagic(magic));
        }

        let version = read_u32(r)?;
        if !(2..=3).contains(&version) {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let raw_tensors = read_u64(r)?;
        let raw_metadata = read_u64(r)?;
        let n_tensors = bounded_count(raw_tensors, MAX_TENSORS, "tensor entries")?;
        let n_kv = bounded_count(raw_metadata, MAX_METADATA_ITEMS, "metadata entries")?;
        ensure_minimum_bytes(r, file_end, raw_metadata, 12, "metadata entries")?;

        let mut metadata = HashMap::with_capacity(n_kv);
        for _ in 0..n_kv {
            let key = read_string(r)?;
            let type_tag = read_u32(r)?;
            let value = read_metadata_value(r, type_tag)?;
            if metadata.insert(key.clone(), value).is_some() {
                return Err(GgufError::DuplicateMetadataKey(key));
            }
        }

        ensure_minimum_bytes(r, file_end, raw_tensors, 24, "tensor entries")?;
        let mut tensors = Vec::with_capacity(n_tensors);
        let mut tensor_names = HashSet::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = read_string(r)?;
            if !tensor_names.insert(name.clone()) {
                return Err(GgufError::DuplicateTensorName(name));
            }
            let raw_dims = u64::from(read_u32(r)?);
            let n_dims = bounded_count(raw_dims, MAX_TENSOR_DIMS, "tensor dimensions")?;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(read_u64(r)?);
            }
            let dtype = GgmlType::from_u32(read_u32(r)?)?;
            let offset = read_u64(r)?;
            tensors.push(TensorInfo {
                name,
                shape,
                dtype,
                offset,
            });
        }

        let alignment = match metadata.get("general.alignment") {
            Some(MetadataValue::U32(a)) => *a as u64,
            _ => 32,
        };
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(GgufError::InvalidAlignment(alignment));
        }
        let pos = r.stream_position()?;
        let data_offset = pos
            .checked_add(alignment - 1)
            .map(|value| value / alignment * alignment)
            .ok_or(GgufError::LimitExceeded {
                what: "aligned header offset",
                value: pos,
                limit: u64::MAX - alignment + 1,
            })?;
        for tensor in &tensors {
            let byte_size = checked_tensor_size(tensor)?;
            let end = data_offset
                .checked_add(tensor.offset)
                .and_then(|offset| offset.checked_add(byte_size))
                .ok_or_else(|| GgufError::TensorSizeOverflow {
                    tensor: tensor.name.clone(),
                })?;
            if end > file_end {
                return Err(GgufError::TensorOutOfBounds {
                    tensor: tensor.name.clone(),
                });
            }
        }

        Ok(GgufFile {
            version,
            metadata,
            tensors,
            data_offset,
        })
    }
}

fn bounded_count(raw: u64, limit: u64, what: &'static str) -> Result<usize, GgufError> {
    if raw > limit {
        return Err(GgufError::LimitExceeded {
            what,
            value: raw,
            limit,
        });
    }
    usize::try_from(raw).map_err(|_| GgufError::LimitExceeded {
        what,
        value: raw,
        limit: usize::MAX as u64,
    })
}

fn ensure_minimum_bytes<R: Seek>(
    reader: &mut R,
    file_end: u64,
    count: u64,
    bytes_per_item: u64,
    what: &'static str,
) -> Result<(), GgufError> {
    let remaining = file_end.saturating_sub(reader.stream_position()?);
    if count > remaining / bytes_per_item {
        return Err(GgufError::LimitExceeded {
            what,
            value: count,
            limit: remaining / bytes_per_item,
        });
    }
    Ok(())
}

fn checked_tensor_size(tensor: &TensorInfo) -> Result<u64, GgufError> {
    let elements = tensor
        .shape
        .iter()
        .try_fold(1u64, |product, dimension| product.checked_mul(*dimension))
        .ok_or_else(|| GgufError::TensorSizeOverflow {
            tensor: tensor.name.clone(),
        })?
        .max(1);
    let block_elements = tensor.dtype.block_elements() as u64;
    let blocks = elements
        .checked_add(block_elements - 1)
        .map(|value| value / block_elements)
        .ok_or_else(|| GgufError::TensorSizeOverflow {
            tensor: tensor.name.clone(),
        })?;
    blocks
        .checked_mul(tensor.dtype.block_size() as u64)
        .ok_or_else(|| GgufError::TensorSizeOverflow {
            tensor: tensor.name.clone(),
        })
}
