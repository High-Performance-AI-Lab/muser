use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

use super::{SafeDtype, SafeTensorsError, SafeTensorsFile, TensorInfo};

impl SafeTensorsFile {
    pub fn parse<R: Read + Seek>(reader: &mut R) -> Result<Self, SafeTensorsError> {
        let start = reader.stream_position()?;
        let file_size = reader.seek(SeekFrom::End(0))?;
        reader.seek(SeekFrom::Start(start))?;
        let mut size = [0; 8];
        reader.read_exact(&mut size)?;
        let header_size = u64::from_le_bytes(size);
        if header_size > 100 * 1024 * 1024 {
            return Err(SafeTensorsError::HeaderTooLarge(header_size));
        }
        if header_size > file_size.saturating_sub(start).saturating_sub(8) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "SafeTensors header exceeds input",
            )
            .into());
        }
        let mut bytes = vec![0; header_size as usize];
        reader.read_exact(&mut bytes)?;
        let data_offset = start + 8 + header_size;
        let root: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|error| SafeTensorsError::InvalidJson(error.to_string()))?;
        let object = root
            .as_object()
            .ok_or_else(|| SafeTensorsError::InvalidJson("root is not an object".into()))?;
        let metadata = object
            .get("__metadata__")
            .and_then(|value| value.as_object())
            .map(|values| {
                values
                    .iter()
                    .filter_map(|(key, value)| Some((key.clone(), value.as_str()?.into())))
                    .collect()
            })
            .unwrap_or_default();
        let data_len = file_size.saturating_sub(data_offset);
        let mut tensors = HashMap::new();
        for (name, value) in object {
            if name == "__metadata__" {
                continue;
            }
            let item = value.as_object().ok_or_else(|| {
                SafeTensorsError::InvalidJson(format!("tensor '{name}' is not an object"))
            })?;
            let dtype =
                SafeDtype::from_str(item.get("dtype").and_then(|v| v.as_str()).ok_or_else(
                    || SafeTensorsError::MissingField {
                        name: name.clone(),
                        field: "dtype",
                    },
                )?)?;
            let shape_values = item
                .get("shape")
                .and_then(|v| v.as_array())
                .ok_or_else(|| SafeTensorsError::MissingField {
                    name: name.clone(),
                    field: "shape",
                })?;
            if shape_values.len() > 32 {
                return Err(SafeTensorsError::InvalidShape {
                    name: name.clone(),
                    detail: "rank exceeds 32",
                });
            }
            let shape = shape_values
                .iter()
                .map(|v| {
                    v.as_u64()
                        .and_then(|n| usize::try_from(n).ok())
                        .ok_or_else(|| SafeTensorsError::InvalidShape {
                            name: name.clone(),
                            detail: "dimensions must be platform-sized integers",
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let offsets = item
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .ok_or_else(|| SafeTensorsError::MissingField {
                    name: name.clone(),
                    field: "data_offsets",
                })?;
            if offsets.len() != 2 {
                return Err(SafeTensorsError::InvalidOffsets {
                    name: name.clone(),
                    start: 0,
                    end: 0,
                });
            }
            let data_start = offsets[0]
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| SafeTensorsError::InvalidOffsets {
                    name: name.clone(),
                    start: 0,
                    end: 0,
                })?;
            let data_end = offsets[1]
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| SafeTensorsError::InvalidOffsets {
                    name: name.clone(),
                    start: data_start,
                    end: 0,
                })?;
            if data_start > data_end {
                return Err(SafeTensorsError::InvalidOffsets {
                    name: name.clone(),
                    start: data_start,
                    end: data_end,
                });
            }
            if data_end as u64 > data_len {
                return Err(SafeTensorsError::OffsetOutOfBounds {
                    name: name.clone(),
                    start: data_start,
                    end: data_end,
                    data_len,
                });
            }
            let computed_bytes = shape
                .iter()
                .try_fold(1usize, |n, dim| n.checked_mul(*dim))
                .and_then(|n| n.checked_mul(dtype.byte_size()))
                .ok_or_else(|| SafeTensorsError::SizeOverflow { name: name.clone() })?;
            if data_end - data_start != computed_bytes {
                return Err(SafeTensorsError::SizeMismatch {
                    name: name.clone(),
                    header_bytes: data_end - data_start,
                    computed_bytes,
                });
            }
            tensors.insert(
                name.clone(),
                TensorInfo {
                    name: name.clone(),
                    dtype,
                    shape,
                    data_start,
                    data_end,
                },
            );
        }
        reader.seek(SeekFrom::Start(data_offset))?;
        Ok(Self {
            tensors,
            data_offset,
            metadata,
        })
    }
}
