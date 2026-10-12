//! mTLS-TCP wire transport. This is the only transport that exists here: the
//! frames below are written to a TLS 1.3 stream. Release qualification uses
//! measured installed-payload throughput with a 3.0 Gbps median floor.
//! RDMA/RoCE is roadmap, not code.
//!
//! Source: the audited in-tree `kvpack-handoff` snapshot plus Ferrite's
//! `main/spark_prefill*` wire framing.
//! PULL-AND-SIMPLIFY.

use std::io::{Read, Write};

use kvpack_handoff::{BeginManifestV2, SealManifestV2, SegmentDescriptorV2};
use serde::{Deserialize, Serialize};

const MAGIC: &[u8; 8] = b"KVPKV2\0\0";
const PREAMBLE_BYTES: usize = 20;

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum WireFrameV2 {
    Begin(BeginAdmissionV2),
    Segment {
        sequence: u32,
        payload: Vec<u8>,
    },
    DeferredSegment {
        descriptor: SegmentDescriptorV2,
        payload: Vec<u8>,
    },
    Seal(SealManifestV2),
    Ack {
        transfer_id: String,
        generation: u64,
    },
    Abort {
        reason: String,
    },
}

/// A Begin frame admitted off the wire: the typed manifest plus the delta
/// prefix cut. The vendored `BeginManifestV2` drops unknown fields at typed
/// parse, so the cut is lifted from the audited raw JSON at the frame
/// boundary and travels beside the manifest; 0 is today's full transfer.
#[derive(Debug, Clone)]
pub struct BeginAdmissionV2 {
    pub manifest: BeginManifestV2,
    pub prefix_cut: u64,
}

impl From<BeginManifestV2> for BeginAdmissionV2 {
    fn from(manifest: BeginManifestV2) -> Self {
        Self {
            manifest,
            prefix_cut: 0,
        }
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum Header {
    Begin {
        manifest: BeginManifestV2,
    },
    Segment {
        sequence: u32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        descriptor: Option<SegmentDescriptorV2>,
    },
    Seal {
        manifest: SealManifestV2,
    },
    Ack {
        transfer_id: String,
        generation: u64,
    },
    Abort {
        reason: String,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct FrameLimitsV2 {
    pub max_header_bytes: usize,
    pub max_payload_bytes: usize,
}
impl Default for FrameLimitsV2 {
    fn default() -> Self {
        Self {
            max_header_bytes: 8 * 1024 * 1024,
            max_payload_bytes: 512 * 1024 * 1024,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid V2 frame: {0}")]
    Invalid(String),
    #[error("JSON: {0}")]
    Json(#[from] serde_json::Error),
}

pub fn write_frame_v2(
    mut writer: impl Write,
    frame: &WireFrameV2,
    limits: FrameLimitsV2,
) -> Result<(), TransportError> {
    let (header, payload) = match frame {
        WireFrameV2::Begin(admission) => {
            // The vendored manifest has no prefix_cut field to serialize; a
            // Rust-side sender must never silently drop one.
            if admission.prefix_cut != 0 {
                return Err(TransportError::Invalid(
                    "the typed begin manifest cannot serialize a prefix cut".into(),
                ));
            }
            (
                Header::Begin {
                    manifest: admission.manifest.clone(),
                },
                &[][..],
            )
        }
        WireFrameV2::Segment { sequence, payload } => (
            Header::Segment {
                sequence: *sequence,
                descriptor: None,
            },
            payload.as_slice(),
        ),
        WireFrameV2::DeferredSegment {
            descriptor,
            payload,
        } => (
            Header::Segment {
                sequence: descriptor.sequence,
                descriptor: Some(descriptor.clone()),
            },
            payload.as_slice(),
        ),
        WireFrameV2::Seal(manifest) => (
            Header::Seal {
                manifest: manifest.clone(),
            },
            &[][..],
        ),
        WireFrameV2::Ack {
            transfer_id,
            generation,
        } => (
            Header::Ack {
                transfer_id: transfer_id.clone(),
                generation: *generation,
            },
            &[][..],
        ),
        WireFrameV2::Abort { reason } => (
            Header::Abort {
                reason: reason.clone(),
            },
            &[][..],
        ),
    };
    let json = serde_json::to_vec(&header)?;
    if json.len() > limits.max_header_bytes || payload.len() > limits.max_payload_bytes {
        return Err(TransportError::Invalid(
            "frame exceeds configured bounds".into(),
        ));
    }
    // One write for the preamble: three small writes are three TLS records.
    let mut preamble = [0u8; PREAMBLE_BYTES];
    preamble[..8].copy_from_slice(MAGIC);
    preamble[8..12].copy_from_slice(&(json.len() as u32).to_le_bytes());
    preamble[12..].copy_from_slice(&(payload.len() as u64).to_le_bytes());
    writer.write_all(&preamble)?;
    writer.write_all(&json)?;
    writer.write_all(payload)?;
    Ok(())
}

pub fn read_frame_v2(
    mut reader: impl Read,
    limits: FrameLimitsV2,
) -> Result<WireFrameV2, TransportError> {
    let mut preamble = [0; PREAMBLE_BYTES];
    reader.read_exact(&mut preamble)?;
    read_frame_v2_after_preamble(&mut reader, preamble, limits)
}

/// Read one frame and return active receive time, excluding producer pacing
/// before the frame's first byte becomes readable.
pub fn read_frame_v2_timed(
    mut reader: impl Read,
    limits: FrameLimitsV2,
) -> Result<(WireFrameV2, u64), TransportError> {
    let mut preamble = [0; PREAMBLE_BYTES];
    reader.read_exact(&mut preamble[..1])?;
    let active_started = std::time::Instant::now();
    reader.read_exact(&mut preamble[1..])?;
    let frame = read_frame_v2_after_preamble(&mut reader, preamble, limits)?;
    let active_ns = u64::try_from(active_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    Ok((frame, active_ns))
}

fn read_frame_v2_after_preamble(
    mut reader: impl Read,
    preamble: [u8; PREAMBLE_BYTES],
    limits: FrameLimitsV2,
) -> Result<WireFrameV2, TransportError> {
    if &preamble[..8] != MAGIC {
        return Err(TransportError::Invalid("bad frame magic".into()));
    }
    let hlen = u32::from_le_bytes(preamble[8..12].try_into().unwrap()) as usize;
    let plen_u64 = u64::from_le_bytes(preamble[12..].try_into().unwrap());
    let plen = usize::try_from(plen_u64)
        .map_err(|_| TransportError::Invalid("payload length exceeds platform".into()))?;
    if hlen == 0 || hlen > limits.max_header_bytes || plen > limits.max_payload_bytes {
        return Err(TransportError::Invalid(
            "frame lengths exceed bounds".into(),
        ));
    }
    let mut json = vec![0; hlen];
    reader.read_exact(&mut json)?;
    audit_frame_fields(&json)?;
    let header: Header = serde_json::from_slice(&json)?;
    match header {
        Header::Begin { manifest } if plen == 0 => Ok(WireFrameV2::Begin(BeginAdmissionV2 {
            manifest,
            prefix_cut: begin_prefix_cut(&json)?,
        })),
        Header::Seal { manifest } if plen == 0 => Ok(WireFrameV2::Seal(manifest)),
        Header::Ack {
            transfer_id,
            generation,
        } if plen == 0 => Ok(WireFrameV2::Ack {
            transfer_id,
            generation,
        }),
        Header::Abort { reason } if plen == 0 => Ok(WireFrameV2::Abort { reason }),
        Header::Segment {
            sequence,
            descriptor,
        } => {
            if plen == 0 {
                return Err(TransportError::Invalid("empty segment".into()));
            }
            // Exact capacity, no zero fill: the segment is overwritten whole.
            let mut payload = Vec::with_capacity(plen);
            reader.by_ref().take(plen_u64).read_to_end(&mut payload)?;
            if payload.len() != plen {
                return Err(TransportError::Invalid("truncated segment payload".into()));
            }
            match descriptor {
                Some(descriptor) if descriptor.sequence == sequence => {
                    Ok(WireFrameV2::DeferredSegment {
                        descriptor,
                        payload,
                    })
                }
                Some(_) => Err(TransportError::Invalid(
                    "segment sequence differs from its descriptor".into(),
                )),
                None => Ok(WireFrameV2::Segment { sequence, payload }),
            }
        }
        _ => Err(TransportError::Invalid(
            "control frame carries payload".into(),
        )),
    }
}

const BEGIN_FRAME_FIELDS: &[&str] = &["kind", "manifest"];
const BEGIN_MANIFEST_FIELDS: &[&str] = &[
    "protocol",
    "transfer_id",
    "generation",
    "created_unix_ms",
    "expires_unix_ms",
    "identity",
    "prompt_token_ids",
    "multimodal",
    "hmac",
    "components",
    "deferred_segments",
    "segments",
    // Delta handoffs name the held prefix they skip; the vendored typed
    // manifest cannot carry the field, so it is lifted from the audited raw
    // JSON into `BeginAdmissionV2` before the typed parse drops it.
    "prefix_cut",
];
const IDENTITY_FIELDS: &[&str] = &[
    "adapter_sha256",
    "chat_template_sha256",
    "context_policy_sha256",
    "model_revision",
    "model_sha256",
    "tokenizer_revision",
    "tokenizer_sha256",
];
const MULTIMODAL_FIELDS: &[&str] = &[
    "projector_sha256",
    "preprocessing_sha256",
    "image_sequence_sha256",
];
const HMAC_FIELDS: &[&str] = &["key_id", "epoch"];
const COMPONENT_FIELDS: &[&str] = &["id", "kind", "required", "identity_sha256"];
const DESCRIPTOR_FIELDS: &[&str] = &[
    "sequence",
    "component_id",
    "role",
    "layer",
    "logical_start",
    "logical_count",
    "element_type",
    "elements_per_token",
    "byte_len",
    "sha256",
];
const SEGMENT_FRAME_FIELDS: &[&str] = &["kind", "sequence", "descriptor"];
const SEAL_FRAME_FIELDS: &[&str] = &["kind", "manifest"];
const SEAL_MANIFEST_FIELDS: &[&str] = &["core", "hmac_sha256"];
const SEAL_CORE_FIELDS: &[&str] = &[
    "transfer_id",
    "generation",
    "begin_sha256",
    "descriptor_sha256",
    "payload_sha256",
    "segment_count",
    "total_bytes",
];
const ACK_FRAME_FIELDS: &[&str] = &["kind", "transfer_id", "generation"];
const ABORT_FRAME_FIELDS: &[&str] = &["kind", "reason"];

fn audit_frame_fields(json: &[u8]) -> Result<(), TransportError> {
    let frame: serde_json::Value = serde_json::from_slice(json)?;
    match frame.get("kind").and_then(serde_json::Value::as_str) {
        Some("begin") => audit_begin_value(&frame),
        Some("segment") => {
            audit_object(Some(&frame), SEGMENT_FRAME_FIELDS, "segment frame")?;
            if frame
                .get("descriptor")
                .is_some_and(|value| !value.is_null())
            {
                audit_object(
                    frame.get("descriptor"),
                    DESCRIPTOR_FIELDS,
                    "deferred segment descriptor",
                )?;
            }
            Ok(())
        }
        Some("seal") => {
            audit_object(Some(&frame), SEAL_FRAME_FIELDS, "seal frame")?;
            let manifest = frame.get("manifest");
            audit_object(manifest, SEAL_MANIFEST_FIELDS, "seal manifest")?;
            audit_object(
                manifest.and_then(|value| value.get("core")),
                SEAL_CORE_FIELDS,
                "seal core",
            )
        }
        Some("ack") => audit_object(Some(&frame), ACK_FRAME_FIELDS, "ACK frame"),
        Some("abort") => audit_object(Some(&frame), ABORT_FRAME_FIELDS, "abort frame"),
        _ => Ok(()),
    }
}

/// Version drift must fail at Begin, not after multi-GB of payload at seal
/// time. The V2 manifest types are declared upstream without
/// `deny_unknown_fields`, so the receiver audits their JSON against the exact
/// field names it understands. Shapes are re-checked by the typed parse.
fn audit_begin_value(frame: &serde_json::Value) -> Result<(), TransportError> {
    audit_object(Some(frame), BEGIN_FRAME_FIELDS, "begin frame")?;
    let manifest = frame.get("manifest");
    audit_object(manifest, BEGIN_MANIFEST_FIELDS, "begin manifest")?;
    let field = |name: &str| manifest.and_then(|manifest| manifest.get(name));
    audit_object(field("identity"), IDENTITY_FIELDS, "exact identity")?;
    audit_object(
        field("multimodal"),
        MULTIMODAL_FIELDS,
        "multimodal identity",
    )?;
    audit_object(field("hmac"), HMAC_FIELDS, "HMAC identity")?;
    for component in field("components")
        .and_then(|value| value.as_array())
        .into_iter()
        .flatten()
    {
        audit_object(Some(component), COMPONENT_FIELDS, "component")?;
    }
    for descriptor in field("segments")
        .and_then(|value| value.as_array())
        .into_iter()
        .flatten()
    {
        audit_object(Some(descriptor), DESCRIPTOR_FIELDS, "segment descriptor")?;
    }
    Ok(())
}

/// Lift the delta prefix cut out of an audited begin frame. The vendored
/// typed manifest drops fields it does not know, so the cut is captured from
/// the raw JSON here; a present-but-non-u64 cut fails closed just like any
/// other malformed begin field.
fn begin_prefix_cut(json: &[u8]) -> Result<u64, TransportError> {
    let frame: serde_json::Value = serde_json::from_slice(json)?;
    let Some(value) = frame
        .get("manifest")
        .and_then(|manifest| manifest.get("prefix_cut"))
    else {
        return Ok(0);
    };
    value.as_u64().ok_or_else(|| {
        TransportError::Invalid("begin prefix_cut is not an unsigned integer".into())
    })
}

fn audit_object(
    value: Option<&serde_json::Value>,
    known: &[&str],
    label: &str,
) -> Result<(), TransportError> {
    let Some(object) = value.and_then(|value| value.as_object()) else {
        return Ok(());
    };
    match object.keys().find(|key| !known.contains(&key.as_str())) {
        Some(unknown) => Err(TransportError::Invalid(format!(
            "{label} carries unknown field `{unknown}`: the producer is newer than this receiver"
        ))),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct PacedReader {
        bytes: Vec<u8>,
        position: usize,
        first_read: bool,
    }

    impl Read for PacedReader {
        fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
            if self.position == self.bytes.len() {
                return Ok(0);
            }
            if self.first_read {
                self.first_read = false;
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            let count = if self.position == 0 {
                1
            } else {
                output.len().min(self.bytes.len() - self.position)
            };
            output[..count].copy_from_slice(&self.bytes[self.position..self.position + count]);
            self.position += count;
            Ok(count)
        }
    }

    fn json_frame(value: serde_json::Value, payload: &[u8]) -> Vec<u8> {
        let json = serde_json::to_vec(&value).unwrap();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&(json.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&json);
        bytes.extend_from_slice(payload);
        bytes
    }

    #[test]
    fn oversize_rejected_before_payload_allocation() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        bytes.extend_from_slice(b"{}");
        assert!(matches!(
            read_frame_v2(bytes.as_slice(), FrameLimitsV2::default()),
            Err(TransportError::Invalid(_))
        ))
    }

    #[test]
    fn timed_frame_excludes_wait_for_first_byte() {
        let bytes = json_frame(
            serde_json::json!({"kind":"ack","transfer_id":"t","generation":1}),
            &[],
        );
        let reader = PacedReader {
            bytes,
            position: 0,
            first_read: true,
        };
        let total_started = std::time::Instant::now();
        let (frame, active_ns) = read_frame_v2_timed(reader, FrameLimitsV2::default()).unwrap();
        let total_ns = u64::try_from(total_started.elapsed().as_nanos()).unwrap();
        assert!(matches!(frame, WireFrameV2::Ack { .. }));
        assert!(total_ns.saturating_sub(active_ns) >= 15_000_000);
    }

    fn begin_frame(patch: impl FnOnce(&mut serde_json::Map<String, serde_json::Value>)) -> Vec<u8> {
        let manifest = serde_json::json!({
            "protocol": "kvpack-live-handoff-v2",
            "transfer_id": "t",
            "generation": 1,
            "created_unix_ms": 1,
            "expires_unix_ms": 2,
            "identity": {
                "adapter_sha256": "a".repeat(64),
                "chat_template_sha256": "b".repeat(64),
                "context_policy_sha256": "c".repeat(64),
                "model_revision": "m",
                "model_sha256": "d".repeat(64),
                "tokenizer_revision": "t",
                "tokenizer_sha256": "e".repeat(64),
            },
            "prompt_token_ids": [1],
            "multimodal": serde_json::Value::Null,
            "hmac": { "key_id": "k", "epoch": 1 },
            "components": [{
                "id": "target",
                "kind": "target_kv",
                "required": true,
                "identity_sha256": "f".repeat(64),
            }],
            "segments": [],
        });
        let mut manifest = manifest.as_object().unwrap().clone();
        patch(&mut manifest);
        let json = serde_json::to_vec(&serde_json::json!({
            "kind": "begin",
            "manifest": manifest,
        }))
        .unwrap();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&(json.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&json);
        bytes
    }

    #[test]
    fn today_s_producer_manifest_is_accepted() {
        assert!(matches!(
            read_frame_v2(begin_frame(|_| {}).as_slice(), FrameLimitsV2::default()),
            Ok(WireFrameV2::Begin(_))
        ))
    }

    #[test]
    fn newer_producer_field_fails_at_begin() {
        let frame = begin_frame(|manifest| {
            manifest.insert("compression".into(), serde_json::json!("zstd"));
        });
        assert!(matches!(
            read_frame_v2(frame.as_slice(), FrameLimitsV2::default()),
            Err(TransportError::Invalid(_))
        ));
        let nested = begin_frame(|manifest| {
            manifest["components"][0]
                .as_object_mut()
                .unwrap()
                .insert("codec".into(), serde_json::json!(2));
        });
        assert!(matches!(
            read_frame_v2(nested.as_slice(), FrameLimitsV2::default()),
            Err(TransportError::Invalid(_))
        ))
    }

    #[test]
    fn delta_begin_lifts_the_prefix_cut() {
        let frame = begin_frame(|manifest| {
            manifest.insert("prefix_cut".into(), serde_json::json!(512));
        });
        match read_frame_v2(frame.as_slice(), FrameLimitsV2::default()).unwrap() {
            WireFrameV2::Begin(admission) => assert_eq!(admission.prefix_cut, 512),
            _ => panic!("expected a begin frame"),
        }
    }

    #[test]
    fn delta_begin_rejects_a_non_integer_cut() {
        for value in [
            serde_json::json!("512"),
            serde_json::json!(512.5),
            serde_json::json!(-1),
            serde_json::json!(true),
        ] {
            let frame = begin_frame(|manifest| {
                manifest.insert("prefix_cut".into(), value);
            });
            assert!(matches!(
                read_frame_v2(frame.as_slice(), FrameLimitsV2::default()),
                Err(TransportError::Invalid(_))
            ));
        }
    }

    #[test]
    fn a_rust_sender_cannot_write_a_cut_the_typed_manifest_drops() {
        let manifest = match read_frame_v2(begin_frame(|_| {}).as_slice(), FrameLimitsV2::default())
            .unwrap()
        {
            WireFrameV2::Begin(admission) => admission.manifest,
            _ => panic!("expected a begin frame"),
        };
        let mut bytes = Vec::new();
        assert!(matches!(
            write_frame_v2(
                &mut bytes,
                &WireFrameV2::Begin(BeginAdmissionV2 {
                    manifest,
                    prefix_cut: 256,
                }),
                FrameLimitsV2::default()
            ),
            Err(TransportError::Invalid(_))
        ));
    }

    #[test]
    fn every_wire_variant_rejects_unknown_fields() {
        let ack = json_frame(
            serde_json::json!({
                "kind": "ack", "transfer_id": "t", "generation": 1,
                "accepted_without_durability": true
            }),
            &[],
        );
        assert!(read_frame_v2(ack.as_slice(), FrameLimitsV2::default()).is_err());

        let segment = json_frame(
            serde_json::json!({
                "kind": "segment", "sequence": 0,
                "descriptor": {
                    "sequence": 0, "component_id": "target", "role": "nope_key",
                    "layer": 0, "logical_start": 0, "logical_count": 1,
                    "element_type": "f16_le", "elements_per_token": 1,
                    "byte_len": 1, "sha256": "0".repeat(64), "compression": "none"
                }
            }),
            &[0],
        );
        assert!(read_frame_v2(segment.as_slice(), FrameLimitsV2::default()).is_err());

        let seal = json_frame(
            serde_json::json!({
                "kind": "seal",
                "manifest": {
                    "core": {
                        "transfer_id": "t", "generation": 1,
                        "begin_sha256": "0".repeat(64),
                        "descriptor_sha256": "0".repeat(64),
                        "payload_sha256": "0".repeat(64),
                        "segment_count": 0, "total_bytes": 0,
                        "commit_policy": "best_effort"
                    },
                    "hmac_sha256": "0".repeat(64)
                }
            }),
            &[],
        );
        assert!(read_frame_v2(seal.as_slice(), FrameLimitsV2::default()).is_err());
    }
}
