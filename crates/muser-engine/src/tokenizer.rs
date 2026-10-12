//! Merge-order-aware GGUF BPE tokenizer and stream-safe detokenizer.

#[path = "tokenizer/bpe.rs"]
mod bpe;
#[path = "tokenizer/streaming.rs"]
mod streaming;

pub use bpe::{BpeTokenizer, TokenizerError};
pub use streaming::StreamingDetokenizer;

#[cfg(test)]
#[path = "tokenizer/tests.rs"]
mod tests;
