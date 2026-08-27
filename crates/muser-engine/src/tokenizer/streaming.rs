use super::BpeTokenizer;

/// Stream-safe detokenizer: buffers incomplete multi-byte UTF-8 sequences across
/// token emissions so that a codepoint split across several BPE tokens (common
/// for emoji and CJK characters) decodes correctly instead of each fragment
/// becoming U+FFFD.
///
/// Algorithm: accumulate raw token bytes in `pending`. On each `push_token`,
/// extend `pending` and return the longest valid-UTF-8 prefix as an owned
/// `String`, retaining any trailing incomplete bytes for the next token.
/// Call `flush` at end-of-stream to emit any remaining bytes (lossy).
pub struct StreamingDetokenizer<'a> {
    tokenizer: &'a BpeTokenizer,
    pending: Vec<u8>,
}

impl<'a> StreamingDetokenizer<'a> {
    pub(super) fn new(tokenizer: &'a BpeTokenizer) -> Self {
        Self {
            tokenizer,
            pending: Vec::with_capacity(32),
        }
    }

    /// Push a single token id; return the complete-UTF-8 prefix text now
    /// emittable. Trailing partial bytes are retained for the next push.
    pub fn push_token(&mut self, id: u32) -> String {
        let bytes = self.tokenizer.get_token_bytes(id);
        self.pending.extend_from_slice(bytes);
        self.drain_complete()
    }

    /// End-of-stream: emit any remaining buffered bytes (lossy substitute for
    /// genuinely invalid sequences that never completed).
    pub fn flush(&mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        let out = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        self.apply_spm_fixup(out)
    }

    fn drain_complete(&mut self) -> String {
        // Find longest valid-UTF-8 prefix of self.pending.
        match std::str::from_utf8(&self.pending) {
            Ok(_) => match String::from_utf8(std::mem::take(&mut self.pending)) {
                Ok(out) => self.apply_spm_fixup(out),
                Err(err) => {
                    self.pending = err.into_bytes();
                    String::new()
                }
            },
            Err(e) => {
                let valid_up_to = e.valid_up_to();
                if valid_up_to == 0 {
                    // Nothing emittable yet; wait for more bytes. But guard against
                    // genuinely-invalid stuck buffers: if we've buffered more than 4
                    // bytes without any valid prefix, the sequence cannot complete
                    // (max UTF-8 codepoint is 4 bytes), so flush lossy.
                    if self.pending.len() > 4 {
                        let stuck: Vec<u8> = self.pending.drain(..1).collect();
                        let mut out = String::from_utf8_lossy(&stuck).into_owned();
                        out.push_str(&self.drain_complete());
                        return self.apply_spm_fixup(out);
                    }
                    return String::new();
                }
                let valid: Vec<u8> = self.pending.drain(..valid_up_to).collect();
                match String::from_utf8(valid) {
                    Ok(out) => self.apply_spm_fixup(out),
                    Err(err) => {
                        self.pending.splice(..0, err.into_bytes());
                        String::new()
                    }
                }
            }
        }
    }

    fn apply_spm_fixup(&self, s: String) -> String {
        if self.tokenizer.is_spm_bpe && s.contains('\u{2581}') {
            s.replace('\u{2581}', " ")
        } else {
            s
        }
    }
}
