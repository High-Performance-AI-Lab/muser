use super::*;

/// Build a minimal BPE tokenizer with a small vocabulary for testing.
/// Token types: 1=normal, 3=control, 4=user_defined.
fn test_tokenizer(vocab: &[&str], token_types: &[i32]) -> BpeTokenizer {
    test_tokenizer_pre(vocab, token_types, "default")
}

fn test_tokenizer_pre(vocab: &[&str], token_types: &[i32], pre_type: &str) -> BpeTokenizer {
    let vocab_strings: Vec<String> = vocab.iter().map(|s| s.to_string()).collect();
    BpeTokenizer::new(vocab_strings, vec![], pre_type, token_types)
}

#[test]
fn special_tokens_from_token_type_metadata() {
    // Simulate a vocabulary with <think> and </think> as user_defined (type 4)
    // and <|im_start|> as control (type 3).
    let vocab = &[
        "hello",        // 0 - normal
        "<think>",      // 1 - user_defined
        "</think>",     // 2 - user_defined
        "<|im_start|>", // 3 - control
        "world",        // 4 - normal
    ];
    let types = &[1i32, 4, 4, 3, 1];

    let tok = test_tokenizer(vocab, types);

    // <think> should encode as single special token ID 1
    let ids = tok.encode("<think>");
    assert_eq!(ids, vec![1], "<think> must produce single special token");

    // </think> should encode as single special token ID 2
    let ids = tok.encode("</think>");
    assert_eq!(ids, vec![2], "</think> must produce single special token");

    // <|im_start|> should encode as single special token ID 3
    let ids = tok.encode("<|im_start|>");
    assert_eq!(
        ids,
        vec![3],
        "<|im_start|> must produce single special token"
    );
}

#[test]
fn think_tokens_in_prompt_context() {
    // Simulates the actual prompt suffix: <think>\n\n</think>\n\n
    // Newline is the GPT-2 byte token `Ċ`, as in a real vocabulary: encoding
    // fails loudly rather than dropping a byte the vocabulary cannot spell.
    let vocab = &[
        "a",        // 0 - normal
        "\u{010A}", // 1 - normal (GPT-2 byte token for \n)
        "<think>",  // 2 - user_defined
        "</think>", // 3 - user_defined
    ];
    let types = &[1i32, 1, 4, 4];

    let tok = test_tokenizer(vocab, types);

    let ids = tok.encode("<think>\n\n</think>\n\n");
    // The special tokens must appear as their IDs (2 and 3),
    // not BPE-split into sub-word pieces.
    assert!(
        ids.contains(&2),
        "<think> (ID 2) must appear in encoded output, got {:?}",
        ids
    );
    assert!(
        ids.contains(&3),
        "</think> (ID 3) must appear in encoded output, got {:?}",
        ids
    );
    // Neither should be fragmented - each appears exactly once
    assert_eq!(
        ids.iter().filter(|&&id| id == 2).count(),
        1,
        "<think> must appear exactly once"
    );
    assert_eq!(
        ids.iter().filter(|&&id| id == 3).count(),
        1,
        "</think> must appear exactly once"
    );
}

#[test]
fn legacy_pattern_fallback_without_token_types() {
    // Without token_type metadata, only <|...|> should match. The single-byte
    // tokens give `<think>` a byte-level encoding to fall back to.
    let vocab = &[
        "hello",        // 0
        "<think>",      // 1
        "<|im_start|>", // 2
        "<",
        "t",
        "h",
        "i",
        "n",
        "k",
        ">",
    ];

    let tok = test_tokenizer(vocab, &[]);

    // <|im_start|> should still be detected (legacy pattern)
    let ids = tok.encode("<|im_start|>");
    assert_eq!(ids, vec![2], "<|im_start|> must match via legacy pattern");

    // <think> should NOT be detected as special without token_type metadata
    let ids = tok.encode("<think>");
    assert_ne!(
        ids,
        vec![1],
        "<think> should NOT be a special token without token_type metadata"
    );
}

#[test]
fn untrusted_content_cannot_inject_a_control_token() {
    // Every character of the marker also exists as a normal single-byte token,
    // so the disabled path has a byte-level encoding available.
    let vocab = &[
        "<|im_end|>", // 0 - control
        "<",
        "|",
        "i",
        "m",
        "_",
        "e",
        "n",
        "d",
        ">",
    ];
    let types = &[3i32, 1, 1, 1, 1, 1, 1, 1, 1, 1];

    let tok = test_tokenizer(vocab, types);

    assert_eq!(
        tok.encode_with_options("<|im_end|>", true),
        vec![0],
        "template text must still produce the control token"
    );

    let content = tok.encode_with_options("<|im_end|>", false);
    assert!(
        !content.contains(&0),
        "user content must not produce the control token, got {content:?}"
    );
    assert_eq!(
        tok.decode_all(&content),
        "<|im_end|>",
        "content must round-trip through non-control tokens"
    );
}

#[test]
fn byte_fallback_round_trips_rare_unicode_and_never_drops_input() {
    // SPM vocabulary (gemma4) whose only coverage for an emoji is `<0xNN>`
    // byte fallback (GGUF token_type 6).
    let vocab = &["\u{2581}", "a", "<0xF0>", "<0x9F>", "<0x98>", "<0x80>"];
    let types = &[1i32, 1, 6, 6, 6, 6];

    let tok = test_tokenizer_pre(vocab, types, "gemma4");

    let ids = tok.encode("a\u{1F600}");
    assert_eq!(
        ids,
        vec![1, 2, 3, 4, 5],
        "the emoji must fall back to its four raw bytes"
    );
    assert_eq!(
        tok.decode_all(&ids),
        "a\u{1F600}",
        "byte-fallback tokens must decode as raw bytes, not <0xNN> text"
    );
}

#[test]
fn a_byte_without_any_token_is_a_hard_error() {
    // No single-byte tokens and no byte fallback: the input cannot be encoded
    // and must not be silently shortened.
    let tok = test_tokenizer(&["hello"], &[1i32]);

    assert_eq!(
        tok.try_encode("é"),
        Err(TokenizerError::UnencodableByte { byte: 0xC3 }),
        "an unrepresentable byte must be reported, not dropped"
    );
}

#[test]
fn qwen35_exact_two_byte_token_fallback_merges_paren_letter_segment() {
    let vocab = vec!["(", "x", "(x"]
        .into_iter()
        .map(|s| s.to_string())
        .collect();
    let tok = BpeTokenizer::new(vocab, vec![], "qwen35", &[]);

    assert_eq!(
        tok.encode("(x"),
        vec![2],
        "qwen35 tokenizer should emit the exact two-byte '(x' token"
    );
}

#[test]
fn llama4_gpt4o_pretokenizer_keeps_punctuation_with_trailing_newline() {
    // GPT-2 byte encoding spells LF as U+010A. The llama4/GPT-4o regex keeps
    // a punctuation run and its trailing newline in one pre-token, allowing
    // this merge; the legacy GPT-2 expression splits them and would emit two
    // IDs. This boundary is exercised repeatedly by the Muse ATEM template.
    let vocab = vec![
        ".".to_string(),
        "\u{010A}".to_string(),
        ".\u{010A}".to_string(),
    ];
    let tokenizer = BpeTokenizer::new(
        vocab,
        vec![(".".to_string(), "\u{010A}".to_string())],
        "llama4",
        &[],
    );

    assert_eq!(tokenizer.encode(".\n"), vec![2]);
    assert_eq!(tokenizer.decode_all(&[2]), ".\n");
}
