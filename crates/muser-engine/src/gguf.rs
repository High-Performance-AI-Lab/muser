//! GGUF file parser — reads header, metadata, and tensor descriptors.
//!
//! GGUF v3 binary format (little-endian):
//!   magic: u32 = 0x46475547 ("GGUF")
//!   version: u32
//!   n_tensors: u64
//!   n_kv: u64
//!   kv_pairs: [MetadataKV; n_kv]
//!   tensor_infos: [TensorInfo; n_tensors]
//!   `<padding to alignment>`
//!   tensor_data: [u8; ...]
//!
//! Reference: [GGUF specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

pub use types::{GgmlType, GgufError, GgufFile, MetadataValue, TensorInfo};

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" in LE

#[path = "gguf/accessors.rs"]
mod accessors;

#[path = "gguf/metadata.rs"]
mod metadata;

#[path = "gguf/parser.rs"]
mod parser;

#[path = "gguf/reader.rs"]
mod reader;

#[path = "gguf/types.rs"]
mod types;

#[cfg(test)]
#[path = "gguf/tests.rs"]
mod tests;
