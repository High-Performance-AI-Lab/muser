use super::metadata::GGUF_TYPE_STRING;
use super::*;
use std::io::Cursor;

fn make_minimal_gguf() -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());
    buf.extend_from_slice(&1u64.to_le_bytes());

    let key = b"general.architecture";
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key);
    buf.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
    let val = b"qwen2";
    buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
    buf.extend_from_slice(val);

    buf
}

#[test]
fn parse_minimal_gguf() {
    let data = make_minimal_gguf();
    let mut cursor = Cursor::new(data);
    let gguf = GgufFile::parse(&mut cursor).unwrap();
    assert_eq!(gguf.version, 3);
    assert_eq!(gguf.meta_str("general.architecture"), Some("qwen2"));
    assert_eq!(gguf.tensors.len(), 0);
}

#[test]
fn parse_path_ignores_untrusted_metadata_sidecar() {
    let directory = tempfile::tempdir().unwrap();
    let model_path = directory.path().join("model.gguf");
    std::fs::write(&model_path, make_minimal_gguf()).unwrap();

    std::fs::write(model_path.with_extension("gguf.meta"), b"untrusted sidecar").unwrap();

    let parsed = GgufFile::parse_path(&model_path).unwrap();
    assert_eq!(parsed.meta_str("general.architecture"), Some("qwen2"));
}

#[test]
fn rejects_counts_that_cannot_fit_in_the_input() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&10_000u64.to_le_bytes());
    let error = GgufFile::parse(&mut Cursor::new(data)).unwrap_err();
    assert!(matches!(error, GgufError::LimitExceeded { .. }));
}

#[test]
fn rejects_zero_alignment_without_panicking() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&1u64.to_le_bytes());
    let key = b"general.alignment";
    data.extend_from_slice(&(key.len() as u64).to_le_bytes());
    data.extend_from_slice(key);
    data.extend_from_slice(&super::metadata::GGUF_TYPE_UINT32.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());
    let error = GgufFile::parse(&mut Cursor::new(data)).unwrap_err();
    assert!(matches!(error, GgufError::InvalidAlignment(0)));
}

#[test]
fn rejects_duplicate_metadata_keys_instead_of_overwriting_identity() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    data.extend_from_slice(&2u64.to_le_bytes());
    for value in [b"first".as_slice(), b"second".as_slice()] {
        let key = b"tokenizer.chat_template";
        data.extend_from_slice(&(key.len() as u64).to_le_bytes());
        data.extend_from_slice(key);
        data.extend_from_slice(&GGUF_TYPE_STRING.to_le_bytes());
        data.extend_from_slice(&(value.len() as u64).to_le_bytes());
        data.extend_from_slice(value);
    }
    let error = GgufFile::parse(&mut Cursor::new(data)).unwrap_err();
    assert!(
        matches!(error, GgufError::DuplicateMetadataKey(key) if key == "tokenizer.chat_template")
    );
}

#[test]
fn rejects_duplicate_tensor_names_instead_of_aliasing_weights() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&3u32.to_le_bytes());
    data.extend_from_slice(&2u64.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes());
    for offset in [0u64, 4] {
        let name = b"duplicate.weight";
        data.extend_from_slice(&(name.len() as u64).to_le_bytes());
        data.extend_from_slice(name);
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&1u64.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&offset.to_le_bytes());
    }
    data.resize(data.len().div_ceil(32) * 32 + 8, 0);
    let error = GgufFile::parse(&mut Cursor::new(data)).unwrap_err();
    assert!(matches!(error, GgufError::DuplicateTensorName(name) if name == "duplicate.weight"));
}

#[test]
fn tokenizer_identity_is_order_independent_typed_and_excludes_template() {
    let metadata = [
        (
            "tokenizer.ggml.model".to_string(),
            MetadataValue::Str("gpt2".into()),
        ),
        (
            "tokenizer.ggml.bos_token_id".to_string(),
            MetadataValue::U32(1),
        ),
        (
            "tokenizer.chat_template".to_string(),
            MetadataValue::Str("template-a".into()),
        ),
    ];
    let make = |values: &[(String, MetadataValue)]| GgufFile {
        version: 3,
        metadata: values.iter().cloned().collect(),
        tensors: Vec::new(),
        data_offset: 0,
    };
    let first = make(&metadata);
    let mut reversed = metadata.clone();
    reversed.reverse();
    let second = make(&reversed);
    assert_eq!(
        first.tokenizer_metadata_sha256(),
        second.tokenizer_metadata_sha256()
    );

    let mut changed_template = metadata.clone();
    changed_template[2].1 = MetadataValue::Str("template-b".into());
    let changed_template = make(&changed_template);
    assert_eq!(
        first.tokenizer_metadata_sha256(),
        changed_template.tokenizer_metadata_sha256()
    );
    assert_ne!(
        first.chat_template_sha256(),
        changed_template.chat_template_sha256()
    );

    let mut changed_type = metadata.clone();
    changed_type[1].1 = MetadataValue::I32(1);
    assert_ne!(
        first.tokenizer_metadata_sha256(),
        make(&changed_type).tokenizer_metadata_sha256()
    );
}
