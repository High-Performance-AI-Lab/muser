use super::{GgufFile, MetadataValue, TensorInfo};
use sha2::{Digest, Sha256};

impl GgufFile {
    /// Exact Jinja template stored in the GGUF. Serving code must derive its
    /// renderer and identity from these bytes, never from a parallel constant.
    pub fn chat_template(&self) -> Option<&str> {
        self.meta_str("tokenizer.chat_template")
    }

    /// Domain-separated canonical hash of all tokenizer metadata except the
    /// chat template, whose identity is carried separately. Keys are sorted;
    /// values retain explicit GGUF type tags, lengths, and little-endian bits.
    pub fn tokenizer_metadata_sha256(&self) -> [u8; 32] {
        let mut keys = self
            .metadata
            .keys()
            .filter(|key| {
                key.starts_with("tokenizer.") && key.as_str() != "tokenizer.chat_template"
            })
            .collect::<Vec<_>>();
        keys.sort_unstable();
        let mut hash = Sha256::new();
        hash.update(b"muser.gguf-tokenizer-metadata.v1\0");
        for key in keys {
            hash_len_bytes(&mut hash, key.as_bytes());
            hash_metadata_value(&mut hash, &self.metadata[key]);
        }
        hash.finalize().into()
    }

    /// Raw SHA-256 of the exact GGUF chat-template bytes. The metadata string
    /// has one canonical byte representation, so no normalization is applied.
    pub fn chat_template_sha256(&self) -> Option<[u8; 32]> {
        self.chat_template()
            .map(|template| Sha256::digest(template.as_bytes()).into())
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn tensor_file_offset(&self, info: &TensorInfo) -> u64 {
        self.data_offset + info.offset
    }

    pub fn meta_str(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key) {
            Some(MetadataValue::Str(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn meta_u32(&self, key: &str) -> Option<u32> {
        match self.metadata.get(key) {
            Some(MetadataValue::U32(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn meta_u64(&self, key: &str) -> Option<u64> {
        match self.metadata.get(key) {
            Some(MetadataValue::U64(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn meta_bool(&self, key: &str) -> Option<bool> {
        match self.metadata.get(key) {
            Some(MetadataValue::Bool(value)) => Some(*value),
            _ => None,
        }
    }

    pub fn meta_u32_array(&self, key: &str) -> Option<Vec<u32>> {
        match self.metadata.get(key) {
            Some(MetadataValue::Array(arr)) => {
                let vals: Vec<u32> = arr
                    .iter()
                    .filter_map(|v| match v {
                        MetadataValue::U32(n) => Some(*n),
                        MetadataValue::I32(n) => Some(*n as u32),
                        _ => None,
                    })
                    .collect();
                (!vals.is_empty()).then_some(vals)
            }
            _ => None,
        }
    }

    pub fn meta_array_len(&self, key: &str) -> Option<usize> {
        match self.metadata.get(key) {
            Some(MetadataValue::Array(values)) => Some(values.len()),
            _ => None,
        }
    }

    pub fn meta_bool_array(&self, key: &str) -> Option<Vec<bool>> {
        match self.metadata.get(key) {
            Some(MetadataValue::Array(arr)) => {
                let vals: Vec<bool> = arr
                    .iter()
                    .filter_map(|v| match v {
                        MetadataValue::Bool(b) => Some(*b),
                        MetadataValue::U8(n) => Some(*n != 0),
                        MetadataValue::U32(n) => Some(*n != 0),
                        MetadataValue::I32(n) => Some(*n != 0),
                        _ => None,
                    })
                    .collect();
                (!vals.is_empty()).then_some(vals)
            }
            _ => None,
        }
    }

    pub fn meta_f32(&self, key: &str) -> Option<f32> {
        match self.metadata.get(key) {
            Some(MetadataValue::F32(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn meta_f32_array(&self, key: &str) -> Option<Vec<f32>> {
        match self.metadata.get(key) {
            Some(MetadataValue::Array(values)) => {
                let result = values
                    .iter()
                    .filter_map(|value| match value {
                        MetadataValue::F32(value) => Some(*value),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                (!result.is_empty()).then_some(result)
            }
            _ => None,
        }
    }

    pub fn token_types(&self) -> Vec<i32> {
        match self.metadata.get("tokenizer.ggml.token_type") {
            Some(MetadataValue::Array(arr)) => arr
                .iter()
                .filter_map(|v| match v {
                    MetadataValue::I32(n) => Some(*n),
                    MetadataValue::U32(n) => Some(*n as i32),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    pub fn vocab(&self) -> Vec<String> {
        match self.metadata.get("tokenizer.ggml.tokens") {
            Some(MetadataValue::Array(arr)) => arr
                .iter()
                .filter_map(|v| {
                    if let MetadataValue::Str(s) = v {
                        Some(s.clone())
                    } else {
                        None
                    }
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    pub fn merges(&self) -> Vec<(String, String)> {
        match self.metadata.get("tokenizer.ggml.merges") {
            Some(MetadataValue::Array(arr)) => arr
                .iter()
                .filter_map(|v| {
                    if let MetadataValue::Str(s) = v {
                        let mut parts = s.splitn(2, ' ');
                        let a = parts.next()?.to_string();
                        let b = parts.next()?.to_string();
                        Some((a, b))
                    } else {
                        None
                    }
                })
                .collect(),
            _ => Vec::new(),
        }
    }
}

fn hash_len_bytes(hash: &mut Sha256, bytes: &[u8]) {
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}

fn hash_metadata_value(hash: &mut Sha256, value: &MetadataValue) {
    match value {
        MetadataValue::U8(value) => {
            hash.update([0]);
            hash.update(value.to_le_bytes());
        }
        MetadataValue::I8(value) => {
            hash.update([1]);
            hash.update(value.to_le_bytes());
        }
        MetadataValue::U16(value) => {
            hash.update([2]);
            hash.update(value.to_le_bytes());
        }
        MetadataValue::I16(value) => {
            hash.update([3]);
            hash.update(value.to_le_bytes());
        }
        MetadataValue::U32(value) => {
            hash.update([4]);
            hash.update(value.to_le_bytes());
        }
        MetadataValue::I32(value) => {
            hash.update([5]);
            hash.update(value.to_le_bytes());
        }
        MetadataValue::U64(value) => {
            hash.update([6]);
            hash.update(value.to_le_bytes());
        }
        MetadataValue::I64(value) => {
            hash.update([7]);
            hash.update(value.to_le_bytes());
        }
        MetadataValue::F32(value) => {
            hash.update([8]);
            hash.update(value.to_bits().to_le_bytes());
        }
        MetadataValue::F64(value) => {
            hash.update([9]);
            hash.update(value.to_bits().to_le_bytes());
        }
        MetadataValue::Bool(value) => {
            hash.update([10, u8::from(*value)]);
        }
        MetadataValue::Str(value) => {
            hash.update([11]);
            hash_len_bytes(hash, value.as_bytes());
        }
        MetadataValue::Array(values) => {
            hash.update([12]);
            hash.update((values.len() as u64).to_le_bytes());
            for value in values {
                hash_metadata_value(hash, value);
            }
        }
    }
}
