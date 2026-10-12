//! Small canonical control plane used to ask the resident GX10
//! muser-prefilld to prefill one exact token sequence. Cache bytes never ride
//! this channel; the daemon connects back over Handoff V2.

use std::io::{Read, Write};

use kvpack_handoff::{canonical_json, decode_canonical_json, MultimodalIdentityV2};
use serde::{Deserialize, Serialize};

pub const MUSER_PREFILL_CONTROL_ALPN: &[u8] = b"muser-prefill-control-v1";
const MAGIC: &[u8; 8] = b"MUPCTL1\0";
const MAX_CONTROL_JSON: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrefillControlRequestV1 {
    pub schema_version: u32,
    pub request_id: String,
    pub deadline_unix_ms: u64,
    pub prompt_token_ids: Vec<u32>,
    pub receiver_host: String,
    pub receiver_port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PrefillControlSegmentV2 {
    Tokens {
        token_ids: Vec<u32>,
    },
    Image {
        data_base64: String,
        sha256: String,
        projected_tokens: u32,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrefillControlRequestV2 {
    pub schema_version: u32,
    pub request_id: String,
    pub deadline_unix_ms: u64,
    pub segments: Vec<PrefillControlSegmentV2>,
    pub multimodal: MultimodalIdentityV2,
    pub receiver_host: String,
    pub receiver_port: u16,
}

impl PrefillControlRequestV2 {
    pub fn validate(&self, now_unix_ms: u64, vocab_size: usize) -> Result<(), String> {
        let mut positions = 0usize;
        let mut images = 0usize;
        for segment in &self.segments {
            match segment {
                PrefillControlSegmentV2::Tokens { token_ids } => {
                    if token_ids.is_empty()
                        || token_ids.iter().any(|token| *token as usize >= vocab_size)
                    {
                        return Err(
                            "multimodal token segment is empty or outside vocabulary".into()
                        );
                    }
                    positions = positions.saturating_add(token_ids.len());
                }
                PrefillControlSegmentV2::Image {
                    data_base64,
                    sha256,
                    projected_tokens,
                } => {
                    images += 1;
                    if data_base64.is_empty()
                        || data_base64.len() > 48 * 1024 * 1024
                        || !valid_digest(sha256)
                        || *projected_tokens == 0
                        || *projected_tokens > 4_096
                    {
                        return Err("multimodal image segment violates its closed bounds".into());
                    }
                    positions = positions.saturating_add(*projected_tokens as usize);
                }
            }
        }
        if self.schema_version != 2
            || !valid_id(&self.request_id)
            || self.deadline_unix_ms <= now_unix_ms
            || self.segments.is_empty()
            || images == 0
            || images > 8
            || !(2..=131_072).contains(&positions)
            || !valid_digest(&self.multimodal.projector_sha256)
            || !valid_digest(&self.multimodal.preprocessing_sha256)
            || !valid_digest(&self.multimodal.image_sequence_sha256)
            || self.receiver_host.is_empty()
            || self.receiver_host.len() > 253
            || !self
                .receiver_host
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b".:-_".contains(&byte))
            || self.receiver_port == 0
        {
            return Err("multimodal prefill control request violates its closed bounds".into());
        }
        if !matches!(self.segments.last(), Some(PrefillControlSegmentV2::Tokens { token_ids }) if !token_ids.is_empty())
        {
            return Err("multimodal request must hold a final text boundary token".into());
        }
        Ok(())
    }
}

impl PrefillControlRequestV1 {
    pub fn validate(&self, now_unix_ms: u64, vocab_size: usize) -> Result<(), String> {
        if self.schema_version != 1
            || !valid_id(&self.request_id)
            || self.deadline_unix_ms <= now_unix_ms
            || self.prompt_token_ids.len() < 2
            || self.prompt_token_ids.len() > 131_072
            || self
                .prompt_token_ids
                .iter()
                .any(|token| *token as usize >= vocab_size)
            || self.receiver_host.is_empty()
            || self.receiver_host.len() > 253
            || !self
                .receiver_host
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b".:-_".contains(&byte))
            || self.receiver_port == 0
        {
            return Err("prefill control request violates its closed bounds".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrefillControlResponseV1 {
    pub schema_version: u32,
    pub request_id: String,
    pub status: PrefillControlStatusV1,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub receipt: Option<ProducerPhaseReceiptV1>,
}

/// Producer-side phase times use Unix nanoseconds so the receiver can prove
/// whether transfer actually overlapped GX10 prefill. They are evidence, not
/// part of the atomic commit decision; Handoff V2 ACK remains that boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProducerPhaseReceiptV1 {
    pub prefill_start_unix_ns: u64,
    pub prefill_end_unix_ns: u64,
    pub state_saved_unix_ns: u64,
    pub transfer_start_unix_ns: u64,
    pub first_segment_sent_unix_ns: u64,
    pub transfer_acked_unix_ns: u64,
    pub prefill_tokens: u32,
    pub payload_bytes: u64,
    /// Sum of producer TLS `sendall` intervals for payload-bearing frames.
    /// This excludes prefill/export gaps and the receiver commit/ACK phase.
    pub payload_wire_ns: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PrefillControlStatusV1 {
    Accepted,
    Committed,
    Failed,
}

impl PrefillControlResponseV1 {
    pub fn validate(&self, request_id: &str) -> Result<(), String> {
        let error_shape = match self.status {
            PrefillControlStatusV1::Accepted | PrefillControlStatusV1::Committed => {
                self.error.is_none()
            }
            PrefillControlStatusV1::Failed => self
                .error
                .as_deref()
                .is_some_and(|message| !message.is_empty() && message.len() <= 1024),
        };
        if self.schema_version != 1 || self.request_id != request_id || !error_shape {
            return Err("prefill control response identity or status is invalid".into());
        }
        if self.status != PrefillControlStatusV1::Committed && self.receipt.is_some() {
            return Err("only a committed control response may carry a receipt".into());
        }
        if let Some(receipt) = &self.receipt {
            if receipt.prefill_tokens == 0
                || receipt.payload_bytes == 0
                || receipt.payload_wire_ns == 0
                || receipt.prefill_start_unix_ns > receipt.prefill_end_unix_ns
                || receipt.prefill_end_unix_ns > receipt.state_saved_unix_ns
                || receipt.transfer_start_unix_ns > receipt.first_segment_sent_unix_ns
                || receipt.first_segment_sent_unix_ns > receipt.transfer_acked_unix_ns
            {
                return Err("prefill producer receipt phase order is invalid".into());
            }
        }
        Ok(())
    }
}

pub fn write_control(writer: &mut impl Write, value: &impl Serialize) -> Result<(), String> {
    let bytes = canonical_json(value).map_err(|error| error.to_string())?;
    if bytes.is_empty() || bytes.len() > MAX_CONTROL_JSON {
        return Err("prefill control frame exceeds bounds".into());
    }
    writer
        .write_all(MAGIC)
        .and_then(|_| writer.write_all(&(bytes.len() as u32).to_le_bytes()))
        .and_then(|_| writer.write_all(&bytes))
        .and_then(|_| writer.flush())
        .map_err(|error| error.to_string())
}

pub fn read_control<T: serde::de::DeserializeOwned + Serialize>(
    reader: &mut impl Read,
) -> Result<T, String> {
    let mut magic = [0u8; 8];
    let mut length = [0u8; 4];
    reader
        .read_exact(&mut magic)
        .and_then(|_| reader.read_exact(&mut length))
        .map_err(|error| error.to_string())?;
    if &magic != MAGIC {
        return Err("prefill control frame magic mismatch".into());
    }
    let length = u32::from_le_bytes(length) as usize;
    if length == 0 || length > MAX_CONTROL_JSON {
        return Err("prefill control frame length is outside bounds".into());
    }
    let mut bytes = vec![0u8; length];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| error.to_string())?;
    decode_canonical_json(&bytes, MAX_CONTROL_JSON).map_err(|error| error.to_string())
}

fn valid_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"-_.".contains(&byte))
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_round_trip_is_canonical_and_bounded() {
        let request = PrefillControlRequestV1 {
            schema_version: 1,
            request_id: "r-1".into(),
            deadline_unix_ms: 99,
            prompt_token_ids: vec![1, 2],
            receiver_host: "127.0.0.1".into(),
            receiver_port: 29590,
        };
        let mut wire = Vec::new();
        write_control(&mut wire, &request).unwrap();
        let decoded: PrefillControlRequestV1 = read_control(&mut wire.as_slice()).unwrap();
        decoded.validate(1, 10).unwrap();
    }

    #[test]
    fn committed_receipt_requires_payload_only_wire_timing() {
        let mut response = PrefillControlResponseV1 {
            schema_version: 1,
            request_id: "r-1".into(),
            status: PrefillControlStatusV1::Committed,
            error: None,
            receipt: Some(ProducerPhaseReceiptV1 {
                prefill_start_unix_ns: 10,
                prefill_end_unix_ns: 20,
                state_saved_unix_ns: 30,
                transfer_start_unix_ns: 5,
                first_segment_sent_unix_ns: 15,
                transfer_acked_unix_ns: 40,
                prefill_tokens: 2_048,
                payload_bytes: 1_000_000,
                payload_wire_ns: 2_000_000,
            }),
        };
        response.validate("r-1").unwrap();
        response.receipt.as_mut().unwrap().payload_wire_ns = 0;
        assert!(response.validate("r-1").is_err());
    }
}
