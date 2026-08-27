use std::collections::HashMap;
use std::io;
use thiserror::Error;

/// Quantization types we support (subset of GGML types).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
#[allow(non_camel_case_types)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    IQ1_M = 29,
    BF16 = 30,
    IQ3_K = 138,
    /// Muser-native NVFP4 E2M1 payload, two logical values per byte.
    /// Per-16 E4M3FN scales and the per-tensor f32 scale2 live in bound
    /// companion tensors; this keeps the serving representation at exactly
    /// 4.5 bits/weight without relying on llama.cpp's experimental format.
    NVFP4_E2M1 = 1000,
    /// Raw E4M3FN bytes used only by NVFP4 companion scale tensors.
    F8_E4M3FN = 1001,
}

impl GgmlType {
    pub(crate) fn from_u32(v: u32) -> Result<Self, GgufError> {
        match v {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            10 => Ok(Self::Q2_K),
            11 => Ok(Self::Q3_K),
            12 => Ok(Self::Q4_K),
            13 => Ok(Self::Q5_K),
            14 => Ok(Self::Q6_K),
            16 => Ok(Self::IQ2_XXS),
            17 => Ok(Self::IQ2_XS),
            18 => Ok(Self::IQ3_XXS),
            19 => Ok(Self::IQ1_S),
            20 => Ok(Self::IQ4_NL),
            21 => Ok(Self::IQ3_S),
            22 => Ok(Self::IQ2_S),
            23 => Ok(Self::IQ4_XS),
            29 => Ok(Self::IQ1_M),
            30 => Ok(Self::BF16),
            138 => Ok(Self::IQ3_K),
            1000 => Ok(Self::NVFP4_E2M1),
            1001 => Ok(Self::F8_E4M3FN),
            _ => Err(GgufError::UnsupportedType(v)),
        }
    }

    /// Bytes per block for this quantization type.
    pub fn block_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q4_0 => 2 + 16,
            Self::Q4_1 => 4 + 16,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 2 + 32,
            Self::Q2_K => 84,
            Self::Q3_K => 110,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
            Self::IQ2_XXS => 66,
            Self::IQ2_XS => 74,
            Self::IQ3_XXS => 98,
            Self::IQ3_S => 110,
            Self::IQ2_S => 82,
            Self::IQ4_NL => 18,
            Self::IQ4_XS => 136,
            Self::IQ1_S => 50,
            Self::IQ1_M => 56,
            Self::BF16 => 2,
            Self::IQ3_K => 110,
            Self::NVFP4_E2M1 => 1,
            Self::F8_E4M3FN => 1,
        }
    }

    /// Number of elements per block.
    pub fn block_elements(self) -> usize {
        match self {
            Self::F32 | Self::F16 | Self::BF16 => 1,
            Self::F8_E4M3FN => 1,
            Self::NVFP4_E2M1 => 2,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::IQ4_NL => 32,
            Self::Q2_K
            | Self::Q3_K
            | Self::Q4_K
            | Self::Q5_K
            | Self::Q6_K
            | Self::IQ2_XXS
            | Self::IQ2_XS
            | Self::IQ3_XXS
            | Self::IQ3_S
            | Self::IQ2_S
            | Self::IQ4_XS
            | Self::IQ1_S
            | Self::IQ1_M
            | Self::IQ3_K => 256,
        }
    }
}

/// A tensor descriptor from the GGUF file (metadata only, no data).
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: GgmlType,
    pub offset: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.shape
            .iter()
            .try_fold(1u64, |product, dimension| product.checked_mul(*dimension))
            .unwrap_or(u64::MAX)
            .max(1)
    }

    pub fn data_size(&self) -> usize {
        let n = usize::try_from(self.n_elements()).unwrap_or(usize::MAX);
        let bs = self.dtype.block_elements();
        let n_blocks = n.div_ceil(bs);
        n_blocks.saturating_mul(self.dtype.block_size())
    }
}

/// Parsed GGUF file header + tensor map.
#[derive(Debug)]
pub struct GgufFile {
    pub version: u32,
    pub metadata: HashMap<String, MetadataValue>,
    pub tensors: Vec<TensorInfo>,
    pub data_offset: u64,
}

#[derive(Debug, Clone)]
pub enum MetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Str(String),
    Array(Vec<MetadataValue>),
}

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("invalid GGUF magic: expected 0x47475546, got 0x{0:08X}")]
    BadMagic(u32),
    #[error("unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),
    #[error("unsupported GGML type: {0}")]
    UnsupportedType(u32),
    #[error("unsupported metadata type tag: {0}")]
    UnsupportedMetadataType(u32),
    #[error("GGUF {what} count {value} exceeds limit {limit}")]
    LimitExceeded {
        what: &'static str,
        value: u64,
        limit: u64,
    },
    #[error("invalid GGUF alignment {0}; expected a nonzero power of two")]
    InvalidAlignment(u64),
    #[error("GGUF tensor '{tensor}' shape or byte size overflows")]
    TensorSizeOverflow { tensor: String },
    #[error("GGUF tensor '{tensor}' byte range exceeds the file")]
    TensorOutOfBounds { tensor: String },
    #[error("duplicate GGUF metadata key '{0}'")]
    DuplicateMetadataKey(String),
    #[error("duplicate GGUF tensor name '{0}'")]
    DuplicateTensorName(String),
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
}
