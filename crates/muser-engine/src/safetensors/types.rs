use std::collections::HashMap;
use std::io;

use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafeDtype {
    Bool,
    U8,
    I8,
    I16,
    I32,
    I64,
    F16,
    BF16,
    F32,
    F64,
}

impl SafeDtype {
    pub fn byte_size(self) -> usize {
        match self {
            Self::Bool | Self::U8 | Self::I8 => 1,
            Self::I16 | Self::F16 | Self::BF16 => 2,
            Self::I32 | Self::F32 => 4,
            Self::I64 | Self::F64 => 8,
        }
    }

    pub(super) fn from_str(value: &str) -> Result<Self, SafeTensorsError> {
        match value {
            "BOOL" => Ok(Self::Bool),
            "U8" => Ok(Self::U8),
            "I8" => Ok(Self::I8),
            "I16" => Ok(Self::I16),
            "I32" => Ok(Self::I32),
            "I64" => Ok(Self::I64),
            "F16" => Ok(Self::F16),
            "BF16" => Ok(Self::BF16),
            "F32" => Ok(Self::F32),
            "F64" => Ok(Self::F64),
            other => Err(SafeTensorsError::UnknownDtype(other.into())),
        }
    }
}

impl std::fmt::Display for SafeDtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: SafeDtype,
    pub shape: Vec<usize>,
    pub data_start: usize,
    pub data_end: usize,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape
            .iter()
            .try_fold(1usize, |n, dim| n.checked_mul(*dim))
            .unwrap_or(usize::MAX)
    }
}

#[derive(Debug, Error)]
pub enum SafeTensorsError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("header too large: {0} bytes")]
    HeaderTooLarge(u64),
    #[error("invalid JSON header: {0}")]
    InvalidJson(String),
    #[error("unknown dtype: {0}")]
    UnknownDtype(String),
    #[error("missing field '{field}' in tensor '{name}'")]
    MissingField { name: String, field: &'static str },
    #[error("invalid data_offsets for tensor '{name}': start={start}, end={end}")]
    InvalidOffsets {
        name: String,
        start: usize,
        end: usize,
    },
    #[error("invalid shape for tensor '{name}': {detail}")]
    InvalidShape { name: String, detail: &'static str },
    #[error("tensor '{name}' shape or byte size overflows")]
    SizeOverflow { name: String },
    #[error("tensor '{name}' offsets [{start}, {end}) exceed data size {data_len}")]
    OffsetOutOfBounds {
        name: String,
        start: usize,
        end: usize,
        data_len: u64,
    },
    #[error("tensor '{name}' byte size mismatch: header {header_bytes}, shape {computed_bytes}")]
    SizeMismatch {
        name: String,
        header_bytes: usize,
        computed_bytes: usize,
    },
    #[error("tensor '{name}' not found")]
    TensorNotFound { name: String },
}

#[derive(Debug)]
pub struct SafeTensorsFile {
    pub tensors: HashMap<String, TensorInfo>,
    pub data_offset: u64,
    pub metadata: HashMap<String, String>,
}
