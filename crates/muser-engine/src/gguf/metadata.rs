use super::reader::*;
use super::{GgufError, MetadataValue};
use std::io::Read;

const MAX_METADATA_ARRAY_ITEMS: u64 = 1_000_000;
const MAX_METADATA_NESTING: usize = 32;

pub(super) const GGUF_TYPE_UINT8: u32 = 0;
pub(super) const GGUF_TYPE_INT8: u32 = 1;
pub(super) const GGUF_TYPE_UINT16: u32 = 2;
pub(super) const GGUF_TYPE_INT16: u32 = 3;
pub(super) const GGUF_TYPE_UINT32: u32 = 4;
pub(super) const GGUF_TYPE_INT32: u32 = 5;
pub(super) const GGUF_TYPE_FLOAT32: u32 = 6;
pub(super) const GGUF_TYPE_BOOL: u32 = 7;
pub(super) const GGUF_TYPE_STRING: u32 = 8;
pub(super) const GGUF_TYPE_ARRAY: u32 = 9;
pub(super) const GGUF_TYPE_UINT64: u32 = 10;
pub(super) const GGUF_TYPE_INT64: u32 = 11;
pub(super) const GGUF_TYPE_FLOAT64: u32 = 12;

pub(super) fn read_metadata_value<R: Read>(
    r: &mut R,
    type_tag: u32,
) -> Result<MetadataValue, GgufError> {
    read_metadata_value_at_depth(r, type_tag, 0)
}

fn read_metadata_value_at_depth<R: Read>(
    r: &mut R,
    type_tag: u32,
    depth: usize,
) -> Result<MetadataValue, GgufError> {
    match type_tag {
        GGUF_TYPE_UINT8 => Ok(MetadataValue::U8(read_u8(r)?)),
        GGUF_TYPE_INT8 => Ok(MetadataValue::I8(read_i8(r)?)),
        GGUF_TYPE_UINT16 => Ok(MetadataValue::U16(read_u16(r)?)),
        GGUF_TYPE_INT16 => Ok(MetadataValue::I16(read_i16(r)?)),
        GGUF_TYPE_UINT32 => Ok(MetadataValue::U32(read_u32(r)?)),
        GGUF_TYPE_INT32 => Ok(MetadataValue::I32(read_i32(r)?)),
        GGUF_TYPE_FLOAT32 => Ok(MetadataValue::F32(read_f32(r)?)),
        GGUF_TYPE_BOOL => Ok(MetadataValue::Bool(read_bool(r)?)),
        GGUF_TYPE_STRING => Ok(MetadataValue::Str(read_string(r)?)),
        GGUF_TYPE_UINT64 => Ok(MetadataValue::U64(read_u64(r)?)),
        GGUF_TYPE_INT64 => Ok(MetadataValue::I64(read_i64(r)?)),
        GGUF_TYPE_FLOAT64 => Ok(MetadataValue::F64(read_f64(r)?)),
        GGUF_TYPE_ARRAY => {
            if depth >= MAX_METADATA_NESTING {
                return Err(GgufError::LimitExceeded {
                    what: "metadata nesting",
                    value: (depth + 1) as u64,
                    limit: MAX_METADATA_NESTING as u64,
                });
            }
            let elem_type = read_u32(r)?;
            let raw_len = read_u64(r)?;
            if raw_len > MAX_METADATA_ARRAY_ITEMS {
                return Err(GgufError::LimitExceeded {
                    what: "metadata array items",
                    value: raw_len,
                    limit: MAX_METADATA_ARRAY_ITEMS,
                });
            }
            let len = usize::try_from(raw_len).map_err(|_| GgufError::LimitExceeded {
                what: "metadata array items",
                value: raw_len,
                limit: usize::MAX as u64,
            })?;
            let mut elems = Vec::with_capacity(len.min(1024));
            for _ in 0..len {
                elems.push(read_metadata_value_at_depth(r, elem_type, depth + 1)?);
            }
            Ok(MetadataValue::Array(elems))
        }
        _ => Err(GgufError::UnsupportedMetadataType(type_tag)),
    }
}
