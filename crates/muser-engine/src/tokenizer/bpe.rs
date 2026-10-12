use super::StreamingDetokenizer;
use fancy_regex::Regex;
use std::collections::HashMap;

// ── BPE Tokenizer (merge-order aware) ──────────────────────────────

/// BPE tokenizer that respects merge priority order from GGUF metadata.
///
/// Uses arena allocation: all token strings are packed into a single `Vec<u8>`
/// with `Vec<u32>` offsets, eliminating ~760K heap allocations and saving ~40 MB
/// for a 152K-token vocabulary. Merges use `(u32, u32)` token-ID pairs instead
/// of `(String, String)`.
pub struct BpeTokenizer {
    /// Packed token strings arena — all GGUF vocab strings concatenated
    tok_arena: Vec<u8>,
    /// tok_offsets[i]..tok_offsets[i+1] = byte range in tok_arena for token i
    tok_offsets: Vec<u32>,
    /// Packed decoded-bytes arena — GPT-2 reverse-mapped bytes for each token
    bytes_arena: Vec<u8>,
    /// bytes_offsets[i]..bytes_offsets[i+1] = byte range in bytes_arena for token i
    bytes_offsets: Vec<u32>,
    /// (left_token_id, right_token_id) → (merge_priority, merged_token_id)
    merge_table: HashMap<(u32, u32), (u32, u32)>,
    /// byte (0-255) → token ID for the GPT-2 single-byte token (u32::MAX if none)
    byte_to_token: [u32; 256],
    /// byte (0-255) → token ID for the `<0xNN>` byte-fallback token (u32::MAX
    /// if the vocabulary has none)
    byte_to_fallback: [u32; 256],
    /// Model-specific pre-tokenization regex.
    pre_regex: Regex,
    /// Special tokens sorted longest-first for greedy matching
    special_tokens: Vec<(String, u32)>,
    /// Qwen-specific exact two-byte token fallback keyed by decoded bytes.
    ///
    /// This is a narrow correctness path for segments like "(x" where the
    /// exact token exists in the vocab and there is only one possible BPE
    /// merge over the two-byte segment.
    qwen_two_byte_tokens: HashMap<[u8; 2], u32>,
    /// SPM-style BPE (Gemma 4): raw UTF-8, space→▁, no GPT-2 byte mapping
    pub(super) is_spm_bpe: bool,
    /// Raw UTF-8 char→token_id for SPM-style BPE (only populated when is_spm_bpe=true)
    char_to_token: HashMap<String, u32>,
}

/// Build the GPT-2 unicode-to-byte reverse mapping.
///
/// GPT-2 BPE maps each byte (0-255) to a unicode codepoint so that
/// all tokens are valid unicode strings. Printable ASCII and some Latin-1
/// characters map to themselves; the rest map to U+0100 onwards.
///
/// Returns a HashMap: unicode char → original byte value.
fn gpt2_unicode_to_byte() -> HashMap<char, u8> {
    let pairs = gpt2_byte_unicode_pairs();
    pairs.into_iter().map(|(b, c)| (c, b)).collect()
}

/// Build the GPT-2 byte→unicode mapping pairs.
/// Returns (byte_value, unicode_char) for all 256 bytes.
fn gpt2_byte_unicode_pairs() -> Vec<(u8, char)> {
    let mut pairs = Vec::with_capacity(256);
    let mut n: u32 = 0;

    // Ranges that map to themselves: '!' to '~', '¡' to '¬', '®' to 'ÿ'
    for b in 0u16..=255 {
        let keep = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        if keep {
            pairs.push((b as u8, char::from_u32(b as u32).unwrap()));
        }
    }
    // Remaining bytes get mapped to U+0100, U+0101, ...
    for b in 0u16..=255 {
        let keep = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        if !keep {
            pairs.push((b as u8, char::from_u32(256 + n).unwrap()));
            n += 1;
        }
    }
    pairs
}

enum Segment {
    Text(String),
    Special(u32),
}

/// Encode failures. Input is never dropped: a byte the vocabulary cannot
/// represent is reported instead of silently disappearing from the prompt.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum TokenizerError {
    #[error(
        "byte 0x{byte:02X} has no single-byte token and no <0xNN> byte fallback in this vocabulary"
    )]
    UnencodableByte { byte: u8 },
}

/// The raw byte a `<0xNN>` byte-fallback token stands for.
///
/// GGUF marks these with `token_type = 6`; converters that omit token types
/// are matched on the literal spelling. A token that declares another type is
/// an ordinary token even when it is spelled like a byte fallback.
fn byte_fallback_byte(token: &str, token_type: Option<i32>) -> Option<u8> {
    if matches!(token_type, Some(declared) if declared != 6) {
        return None;
    }
    let hex = token.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

impl BpeTokenizer {
    fn qwen_exact_two_byte_token_id(&self, word: &str) -> Option<u32> {
        if self.is_spm_bpe || self.qwen_two_byte_tokens.is_empty() {
            return None;
        }

        let bytes = word.as_bytes();
        if bytes.len() != 2 {
            return None;
        }
        self.qwen_two_byte_tokens
            .get(&[bytes[0], bytes[1]])
            .copied()
    }

    /// Build from GGUF vocab (token strings), merge rules, and optional token types.
    ///
    /// `pre_type` is the value of `tokenizer.ggml.pre` from GGUF metadata
    /// (e.g. "qwen2", "gpt2", "default"). It determines the pre-tokenization regex.
    ///
    /// `token_types` (from `tokenizer.ggml.token_type`) marks each token's role:
    /// 3=control, 4=user_defined — these are matched as special tokens before BPE.
    /// Pass an empty slice to fall back to pattern-based detection only.
    pub fn new(
        vocab: Vec<String>,
        merges: Vec<(String, String)>,
        pre_type: &str,
        token_types: &[i32],
    ) -> Self {
        // Build temporary token_to_id (dropped before Self is constructed)
        let mut token_to_id = HashMap::with_capacity(vocab.len());
        let total_bytes: usize = vocab.iter().map(|t| t.len()).sum();
        let mut tok_arena = Vec::with_capacity(total_bytes);
        let mut tok_offsets = Vec::with_capacity(vocab.len() + 1);

        for (id, token) in vocab.iter().enumerate() {
            token_to_id.insert(token.as_str(), id as u32);
            tok_offsets.push(tok_arena.len() as u32);
            tok_arena.extend_from_slice(token.as_bytes());
        }
        tok_offsets.push(tok_arena.len() as u32);

        // Build merge_table: (left_id, right_id) → (priority, merged_token_id)
        let mut merge_table = HashMap::with_capacity(merges.len());
        for (priority, (a, b)) in merges.iter().enumerate() {
            if let (Some(&aid), Some(&bid)) =
                (token_to_id.get(a.as_str()), token_to_id.get(b.as_str()))
            {
                let merged = format!("{}{}", a, b);
                if let Some(&mid) = token_to_id.get(merged.as_str()) {
                    merge_table.insert((aid, bid), (priority as u32, mid));
                }
            }
        }

        // Detect SPM-style BPE (Gemma 4): raw UTF-8, space→▁
        let spm_count = vocab.iter().filter(|t| t.starts_with('\u{2581}')).count();
        let gpt2_count = vocab.iter().filter(|t| t.starts_with('\u{0120}')).count();
        let is_spm_bpe = pre_type == "gemma4" || (spm_count > 100 && gpt2_count < 100);

        // Byte-fallback tokens stand for one raw byte on both encode and
        // decode; emitting their literal `<0xNN>` spelling would corrupt the
        // stream. Earliest ID wins so the table is deterministic.
        let mut byte_to_fallback = [u32::MAX; 256];
        let fallback_byte = vocab
            .iter()
            .enumerate()
            .map(|(id, token)| byte_fallback_byte(token, token_types.get(id).copied()))
            .collect::<Vec<_>>();
        for (id, byte) in fallback_byte.iter().enumerate() {
            if let Some(byte) = byte {
                if byte_to_fallback[*byte as usize] == u32::MAX {
                    byte_to_fallback[*byte as usize] = id as u32;
                }
            }
        }

        // Pack decoded bytes into arena using GPT-2 reverse mapping
        let u2b = gpt2_unicode_to_byte();
        let mut bytes_arena = Vec::with_capacity(total_bytes);
        let mut bytes_offsets = Vec::with_capacity(vocab.len() + 1);

        for (id, token) in vocab.iter().enumerate() {
            bytes_offsets.push(bytes_arena.len() as u32);
            if let Some(byte) = fallback_byte[id] {
                bytes_arena.push(byte);
            } else if is_spm_bpe {
                // SPM-style BPE (Gemma): tokens are raw UTF-8 strings.
                // Store exact UTF-8 bytes — do NOT use GPT-2 byte mapping.
                bytes_arena.extend_from_slice(token.as_bytes());
            } else {
                for c in token.chars() {
                    bytes_arena.push(u2b.get(&c).copied().unwrap_or(c as u8));
                }
            }
        }
        bytes_offsets.push(bytes_arena.len() as u32);

        // Build byte→unicode forward mapping and byte→token_id table
        let pairs = gpt2_byte_unicode_pairs();
        let mut byte_to_token = [u32::MAX; 256];
        for (b, c) in &pairs {
            let ch = c.to_string();
            if let Some(&id) = token_to_id.get(ch.as_str()) {
                byte_to_token[*b as usize] = id;
            }
        }

        // For SPM-style BPE: build char→token_id for raw UTF-8 characters
        let char_to_token = if is_spm_bpe {
            let mut m = HashMap::new();
            for (id, token) in vocab.iter().enumerate() {
                // Single-character tokens map directly (including ▁ and individual chars)
                if token.chars().count() == 1 {
                    m.insert(token.clone(), id as u32);
                }
            }
            // Also map single bytes that aren't already covered
            for b in 0u8..=255u8 {
                let s = String::from(b as char);
                if let std::collections::hash_map::Entry::Vacant(entry) = m.entry(s) {
                    if let Some(&id) = token_to_id.get(entry.key().as_str()) {
                        entry.insert(id);
                    }
                }
            }
            m
        } else {
            HashMap::new()
        };

        let qwen_two_byte_tokens = if matches!(pre_type, "qwen2" | "qwen35") && !is_spm_bpe {
            let mut m = HashMap::new();
            for id in 0..vocab.len() {
                let start = bytes_offsets[id] as usize;
                let end = bytes_offsets[id + 1] as usize;
                let bytes = &bytes_arena[start..end];
                if bytes.len() == 2 {
                    m.entry([bytes[0], bytes[1]]).or_insert(id as u32);
                }
            }
            m
        } else {
            HashMap::new()
        };

        // token_to_id is no longer needed — drop it before constructing Self
        drop(token_to_id);

        // Pre-tokenization regex based on model type
        let pattern = match pre_type {
            // llama.cpp maps the GGUF `llama4` pre-tokenizer identifier to
            // its GPT-4o regex. Keep this expression byte-for-byte aligned
            // with the source-pinned comparator: the optional punctuation
            // prefix and case-transition branches materially affect Muse's
            // chat-template token IDs.
            "llama4" | "gpt-4o" => concat!(
                r"[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))*((?=[\p{L}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?",
                r"|[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))+((?=[\p{L}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?",
                r"|\p{N}{1,3}",
                r"| ?[^\s\p{L}\p{N}]+[\r\n/]*",
                r"|\s*[\r\n]+",
                r"|\s+(?!\S)",
                r"|\s+",
            ),
            "qwen2" | "qwen35" => concat!(
                r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])",
                r"|[^\r\n\p{L}\p{N}]?\p{L}+",
                r"|\p{N}",
                r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
                r"|\s*[\r\n]+",
                r"|\s+(?!\S)",
                r"|\s+",
            ),
            "gemma4" => r"[^\n]+|[\n]+",
            _ if is_spm_bpe => r"[^\n]+|[\n]+",
            _ => concat!(
                r"'s|'t|'re|'ve|'m|'ll|'d",
                r"| ?\p{L}+",
                r"| ?\p{N}+",
                r"| ?[^\s\p{L}\p{N}]+",
                r"|\s+(?!\S)",
                r"|\s+",
            ),
        };
        let pre_regex = Regex::new(pattern).expect("invalid pre-tokenization regex");

        // Collect special tokens sorted longest-first for greedy matching.
        // Use GGUF token_type metadata when available (types 3=control, 4=user_defined),
        // otherwise fall back to <|...|> pattern matching.
        let mut special_tokens: Vec<(String, u32)> = vocab
            .iter()
            .enumerate()
            .filter(|(id, t)| {
                if !token_types.is_empty() && *id < token_types.len() {
                    let ty = token_types[*id];
                    // 3 = control (e.g. <|im_start|>), 4 = user_defined (e.g. <think>)
                    ty == 3 || ty == 4
                } else {
                    // Fallback: legacy pattern for models without token_type metadata
                    t.starts_with("<|") && t.ends_with("|>")
                }
            })
            .map(|(id, t)| (t.clone(), id as u32))
            .collect();
        special_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        if is_spm_bpe {
            eprintln!("[tokenizer] SPM-style BPE detected (gemma4): space→▁, raw UTF-8");
        }

        Self {
            tok_arena,
            tok_offsets,
            bytes_arena,
            bytes_offsets,
            merge_table,
            byte_to_token,
            byte_to_fallback,
            pre_regex,
            special_tokens,
            qwen_two_byte_tokens,
            is_spm_bpe,
            char_to_token,
        }
    }

    /// Get the raw GGUF token string for a token ID (GPT-2 byte-encoded).
    #[inline]
    fn token_str(&self, id: u32) -> &str {
        let start = self.tok_offsets[id as usize] as usize;
        let end = self.tok_offsets[id as usize + 1] as usize;
        // Safety: GGUF vocab strings are valid UTF-8
        std::str::from_utf8(&self.tok_arena[start..end]).unwrap_or("")
    }

    /// Get the decoded bytes for a token ID.
    #[inline]
    fn token_bytes(&self, id: u32) -> &[u8] {
        let start = self.bytes_offsets[id as usize] as usize;
        let end = self.bytes_offsets[id as usize + 1] as usize;
        &self.bytes_arena[start..end]
    }

    /// Encode text to token IDs using BPE merge order.
    ///
    /// 1. Pre-tokenize using the model-specific regex (splits into "words")
    /// 2. For each word, convert bytes → GPT-2 unicode characters
    /// 3. Apply BPE merges in priority order to each word independently
    /// 4. Look up token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.encode_with_options(text, true)
    }

    /// Encode text, choosing whether special/control tokens are matched.
    ///
    /// With `parse_special = false` the text is encoded byte-level and
    /// merge-level only, so a control marker spelled inside untrusted content
    /// (`<|im_end|>`) cannot inject the control token it names.
    ///
    /// Panics only when the vocabulary cannot represent a byte at all — a
    /// vocabulary defect, not a property of the input. Callers that must stay
    /// alive on such a vocabulary use `try_encode_with_options`.
    pub fn encode_with_options(&self, text: &str, parse_special: bool) -> Vec<u32> {
        self.try_encode_with_options(text, parse_special)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    /// Fallible `encode`.
    pub fn try_encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        self.try_encode_with_options(text, true)
    }

    /// Fallible `encode_with_options`.
    pub fn try_encode_with_options(
        &self,
        text: &str,
        parse_special: bool,
    ) -> Result<Vec<u32>, TokenizerError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        let mut result = Vec::new();
        if !parse_special {
            self.encode_segment(text, &mut result)?;
            return Ok(result);
        }

        // Split text on special tokens first, then BPE-encode the non-special segments
        let segments = self.split_special_tokens(text);
        for segment in segments {
            match segment {
                Segment::Special(id) => result.push(id),
                Segment::Text(s) => self.encode_segment(&s, &mut result)?,
            }
        }

        Ok(result)
    }

    /// Token ID for one raw byte: the single-byte token when the vocabulary
    /// has one, otherwise the `<0xNN>` byte-fallback token.
    fn byte_token(&self, byte: u8) -> Result<u32, TokenizerError> {
        for id in [
            self.byte_to_token[byte as usize],
            self.byte_to_fallback[byte as usize],
        ] {
            if id != u32::MAX {
                return Ok(id);
            }
        }
        Err(TokenizerError::UnencodableByte { byte })
    }

    /// Encode a text segment (no special tokens) using pre-tokenization + BPE.
    fn encode_segment(&self, text: &str, result: &mut Vec<u32>) -> Result<(), TokenizerError> {
        // SPM-style BPE (Gemma 4): replace spaces with ▁, use raw UTF-8 chars
        let text_for_bpe;
        let effective_text = if self.is_spm_bpe {
            text_for_bpe = text.replace(' ', "\u{2581}");
            text_for_bpe.as_str()
        } else {
            text
        };

        let mut pos = 0;
        while pos < effective_text.len() {
            let m = self.pre_regex.find_from_pos(effective_text, pos);
            match m {
                Ok(Some(m)) => {
                    let word = m.as_str();
                    if let Some(id) = self.qwen_exact_two_byte_token_id(word) {
                        result.push(id);
                        pos = m.end();
                        continue;
                    }
                    let mut ids: Vec<u32> = Vec::with_capacity(word.len());
                    if self.is_spm_bpe {
                        // SPM: map each UTF-8 character to its token ID, and a
                        // character the vocabulary lacks to its raw bytes.
                        let mut buffer = [0u8; 4];
                        for character in word.chars() {
                            let encoded = character.encode_utf8(&mut buffer);
                            match self.char_to_token.get(&*encoded) {
                                Some(&id) => ids.push(id),
                                None => {
                                    for &byte in encoded.as_bytes() {
                                        ids.push(self.byte_token(byte)?);
                                    }
                                }
                            }
                        }
                    } else {
                        // GPT-2: map each byte to its token ID
                        for byte in word.bytes() {
                            ids.push(self.byte_token(byte)?);
                        }
                    }

                    // BPE merge using token IDs
                    result.extend_from_slice(&self.bpe_merge_ids(ids));
                    pos = m.end();
                }
                _ => {
                    // No pre-token matched here. The byte is still encoded —
                    // dropping it would silently alter the prompt.
                    result.push(self.byte_token(effective_text.as_bytes()[pos])?);
                    pos += 1;
                }
            }
        }
        Ok(())
    }

    /// Split text into alternating Text/Special segments.
    fn split_special_tokens(&self, text: &str) -> Vec<Segment> {
        if self.special_tokens.is_empty() {
            return vec![Segment::Text(text.to_string())];
        }
        let mut segments = Vec::new();
        let mut remaining = text;
        while !remaining.is_empty() {
            // Find the earliest-occurring special token
            let mut best: Option<(usize, &str, u32)> = None;
            for (tok, id) in &self.special_tokens {
                if let Some(pos) = remaining.find(tok.as_str()) {
                    if best.is_none() || pos < best.unwrap().0 {
                        best = Some((pos, tok.as_str(), *id));
                    }
                }
            }
            match best {
                Some((pos, tok, id)) => {
                    if pos > 0 {
                        segments.push(Segment::Text(remaining[..pos].to_string()));
                    }
                    segments.push(Segment::Special(id));
                    remaining = &remaining[pos + tok.len()..];
                }
                None => {
                    segments.push(Segment::Text(remaining.to_string()));
                    break;
                }
            }
        }
        segments
    }

    /// Apply BPE merges to a sequence of token IDs.
    /// Adjacent pairs are merged using (left_id, right_id) → priority lookup.
    fn bpe_merge_ids(&self, mut ids: Vec<u32>) -> Vec<u32> {
        loop {
            if ids.len() < 2 {
                break;
            }

            let mut best_priority = u32::MAX;
            let mut best_idx = usize::MAX;

            for i in 0..ids.len() - 1 {
                if let Some(&(priority, _)) = self.merge_table.get(&(ids[i], ids[i + 1])) {
                    if priority < best_priority {
                        best_priority = priority;
                        best_idx = i;
                    }
                }
            }

            if best_idx == usize::MAX {
                break;
            }

            // Lookup merged token ID from precomputed table (no String concat needed)
            let (_, merged_id) = self.merge_table[&(ids[best_idx], ids[best_idx + 1])];
            ids[best_idx] = merged_id;
            ids.remove(best_idx + 1);
        }

        ids
    }

    /// Decode a token ID to its raw GGUF string (without GPT-2 byte mapping).
    pub fn decode_raw(&self, id: u32) -> &str {
        if (id as usize) < self.vocab_size() {
            self.token_str(id)
        } else {
            ""
        }
    }

    /// Decode a single token ID to a UTF-8 string (lossy).
    pub fn decode(&self, id: u32) -> String {
        if (id as usize) < self.vocab_size() {
            let s = String::from_utf8_lossy(self.token_bytes(id)).into_owned();
            if self.is_spm_bpe {
                s.replace('\u{2581}', " ")
            } else {
                s
            }
        } else {
            String::new()
        }
    }

    /// Zero-alloc decode: returns `Cow::Borrowed` for valid UTF-8 tokens (>99%),
    /// `Cow::Owned` only for rare invalid sequences.
    pub fn decode_ref(&self, id: u32) -> std::borrow::Cow<'_, str> {
        if (id as usize) < self.vocab_size() {
            let cow = String::from_utf8_lossy(self.token_bytes(id));
            if self.is_spm_bpe && cow.contains('\u{2581}') {
                std::borrow::Cow::Owned(cow.replace('\u{2581}', " "))
            } else {
                cow
            }
        } else {
            std::borrow::Cow::Borrowed("")
        }
    }

    /// Decode a sequence of token IDs to a UTF-8 string.
    pub fn decode_all(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if (id as usize) < self.vocab_size() {
                bytes.extend_from_slice(self.token_bytes(id));
            }
        }
        let s = String::from_utf8_lossy(&bytes).into_owned();
        if self.is_spm_bpe {
            s.replace('\u{2581}', " ")
        } else {
            s
        }
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.tok_offsets.len() - 1
    }

    /// Get decoded bytes for token `id`. Returns empty slice for invalid IDs.
    pub fn get_token_bytes(&self, id: u32) -> &[u8] {
        if (id as usize) < self.vocab_size() {
            self.token_bytes(id)
        } else {
            &[]
        }
    }

    /// Create a new stateful streaming detokenizer that buffers partial UTF-8
    /// sequences across token boundaries.
    pub fn streaming_detokenizer(&self) -> StreamingDetokenizer<'_> {
        StreamingDetokenizer::new(self)
    }
}
