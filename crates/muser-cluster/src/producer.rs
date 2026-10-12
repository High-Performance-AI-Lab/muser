//! GX10 producer stream receive + atomic session negotiation.
//!
//! Source (PULL-AND-SIMPLIFY): Ferrite main/spark_prefill plus
//! kvpack-handoff. The pinned llama.cpp producer adapter and container tooling
//! live under this repository's `scripts/gx10/` tree.

use std::io::{Read, Write};

use crate::phase::{nanos, SegmentPhaseNanos, SharedHandoffPhase};
use kvpack_handoff::{
    AtomicReceiverV2, BeginManifestV2, CommittedGeneration, HandoffSinkV2, MacKey, ValidatedBeginV2,
};

use crate::security::{ReplayLedger, SecurityError};
use crate::transport::{
    read_frame_v2, read_frame_v2_timed, write_frame_v2, BeginAdmissionV2, FrameLimitsV2,
    TransportError, WireFrameV2,
};

#[derive(Debug, thiserror::Error)]
pub enum ReceiverError {
    #[error(transparent)]
    Transport(#[from] TransportError),
    #[error(transparent)]
    Handoff(#[from] kvpack_handoff::HandoffError),
    #[error("producer aborted: {0}")]
    ProducerAbort(String),
    #[error("expected begin frame")]
    MissingBegin,
    #[error(transparent)]
    Security(#[from] SecurityError),
}

pub struct ReceiverPolicy<'a> {
    pub now_unix_ms: u64,
    pub expected_key_id: &'a str,
    pub minimum_key_epoch: u64,
    pub limits: FrameLimitsV2,
}

/// Loopback/live stream core. TLS/mTLS and leaf-pin establishment happens
/// before this function receives the stream; every content byte is then
/// protected again by the V2 epoch-bound HMAC seal.
pub fn receive_v2(
    reader: &mut impl Read,
    key: MacKey,
    policy: ReceiverPolicy<'_>,
    sink: impl HandoffSinkV2,
) -> Result<CommittedGeneration, ReceiverError> {
    let admission = read_begin_v2(reader, policy.limits)?;
    receive_after_begin(reader, key, policy, sink, admission.manifest)
}

/// Read the Begin frame alone, so a caller holding several producer
/// connections can admit or drop one before any payload byte is transferred.
/// The returned admission carries the delta prefix cut beside the typed
/// manifest; only the request-aware receiver (`receiver.rs`) wires it into
/// the sink, so any other path that accepts a nonzero cut fails closed at
/// the first suffix tile.
pub fn read_begin_v2(
    reader: &mut impl Read,
    limits: FrameLimitsV2,
) -> Result<BeginAdmissionV2, ReceiverError> {
    match read_frame_v2(reader, limits)? {
        WireFrameV2::Begin(admission) => Ok(admission),
        _ => Err(ReceiverError::MissingBegin),
    }
}

fn receive_after_begin(
    reader: &mut impl Read,
    key: MacKey,
    policy: ReceiverPolicy<'_>,
    sink: impl HandoffSinkV2,
    manifest: BeginManifestV2,
) -> Result<CommittedGeneration, ReceiverError> {
    receive_after_begin_with_reservation(reader, key, policy, sink, manifest, || Ok(()))
}

fn validation_lock(context: &str) -> ReceiverError {
    ReceiverError::Handoff(kvpack_handoff::HandoffError::Validation(format!(
        "{context} lock was poisoned"
    )))
}

fn receive_after_begin_with_reservation(
    reader: &mut impl Read,
    key: MacKey,
    policy: ReceiverPolicy<'_>,
    sink: impl HandoffSinkV2,
    manifest: BeginManifestV2,
    reserve: impl FnOnce() -> Result<(), ReceiverError>,
) -> Result<CommittedGeneration, ReceiverError> {
    receive_after_begin_with_reservation_phased(reader, key, policy, sink, manifest, reserve, None)
}

/// The same receive loop with per-phase timing evidence (N series). Timing
/// is structural: read = frame read off the socket; process = the
/// vendored verify + sink install call; install = the sink's own subset;
/// seal/commit are timed at the transaction boundary.
fn receive_after_begin_with_reservation_phased(
    reader: &mut impl Read,
    key: MacKey,
    policy: ReceiverPolicy<'_>,
    sink: impl HandoffSinkV2,
    manifest: BeginManifestV2,
    reserve: impl FnOnce() -> Result<(), ReceiverError>,
    phases: Option<SharedHandoffPhase>,
) -> Result<CommittedGeneration, ReceiverError> {
    // The upstream lifetime check cannot say which side is wrong; a handoff
    // that is already expired on arrival is far more often producer/receiver
    // clock skew than a genuinely stale transfer.
    if policy.now_unix_ms > manifest.expires_unix_ms {
        return Err(ReceiverError::Handoff(
            kvpack_handoff::HandoffError::Validation(format!(
                "handoff expired at {} unix ms while the receiver clock reads {} unix ms: \
                 check producer/receiver clock skew",
                manifest.expires_unix_ms, policy.now_unix_ms
            )),
        ));
    }
    let begin = ValidatedBeginV2::validate(
        manifest,
        policy.now_unix_ms,
        policy.expected_key_id,
        policy.minimum_key_epoch,
    )?;
    let mut receiver = AtomicReceiverV2::begin(begin, key, sink)?;
    let mut reserve = Some(reserve);
    let frame_loop_started = std::time::Instant::now();
    loop {
        let frame_offset_ns = nanos(frame_loop_started.elapsed());
        let (frame, read_ns) = read_frame_v2_timed(&mut *reader, policy.limits)?;
        match frame {
            WireFrameV2::Segment { sequence, payload } => {
                let process_started = std::time::Instant::now();
                receiver.segment_ready(sequence, payload)?;
                let process_ns = nanos(process_started.elapsed());
                if let Some(phases) = &phases {
                    let mut phases = phases
                        .lock()
                        .map_err(|_| validation_lock("handoff phase evidence"))?;
                    let install_ns = std::mem::take(&mut phases.pending_install_ns);
                    phases.segment_read_ns = phases.segment_read_ns.saturating_add(read_ns);
                    phases.segment_process_ns =
                        phases.segment_process_ns.saturating_add(process_ns);
                    phases.segments.push(SegmentPhaseNanos {
                        sequence,
                        read_ns,
                        process_ns,
                        install_ns,
                        read_started_offset_ns: frame_offset_ns,
                    });
                }
            }
            WireFrameV2::DeferredSegment {
                descriptor,
                payload,
            } => {
                let sequence = descriptor.sequence;
                let process_started = std::time::Instant::now();
                receiver.segment_ready_deferred(descriptor, payload)?;
                let process_ns = nanos(process_started.elapsed());
                if let Some(phases) = &phases {
                    let mut phases = phases
                        .lock()
                        .map_err(|_| validation_lock("handoff phase evidence"))?;
                    let install_ns = std::mem::take(&mut phases.pending_install_ns);
                    phases.segment_read_ns = phases.segment_read_ns.saturating_add(read_ns);
                    phases.segment_process_ns =
                        phases.segment_process_ns.saturating_add(process_ns);
                    phases.segments.push(SegmentPhaseNanos {
                        sequence,
                        read_ns,
                        process_ns,
                        install_ns,
                        read_started_offset_ns: frame_offset_ns,
                    });
                }
            }
            WireFrameV2::Seal(seal) => {
                let seal_started = std::time::Instant::now();
                receiver.prepare_commit(seal)?;
                if let Some(phases) = &phases {
                    let mut phases = phases
                        .lock()
                        .map_err(|_| validation_lock("handoff phase evidence"))?;
                    phases.seal_ns = phases.seal_ns.saturating_add(nanos(seal_started.elapsed()));
                    phases.seal_read_offset_ns = frame_offset_ns;
                    phases.seal_read_unix_ns = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|span| span.as_nanos().min(u64::MAX as u128) as u64)
                        .unwrap_or(0);
                }
                reserve.take().expect("one seal per handoff")()?;
                return Ok(receiver.commit()?);
            }
            WireFrameV2::Abort { reason } => {
                receiver.abort();
                return Err(ReceiverError::ProducerAbort(reason));
            }
            WireFrameV2::Begin(_) => {
                receiver.abort();
                return Err(ReceiverError::Handoff(
                    kvpack_handoff::HandoffError::Validation("duplicate begin".into()),
                ));
            }
            WireFrameV2::Ack { .. } => {
                receiver.abort();
                return Err(ReceiverError::Handoff(
                    kvpack_handoff::HandoffError::Validation(
                        "producer sent an ACK to the receiver".into(),
                    ),
                ));
            }
        }
    }
}

/// Duplex receiver which emits ACK only after `commit()` has installed the
/// complete engine generation. Callers publish durable cache state after this
/// returns, never before the ACK.
pub fn receive_v2_with_ack(
    stream: &mut (impl Read + Write),
    key: MacKey,
    policy: ReceiverPolicy<'_>,
    sink: impl HandoffSinkV2,
) -> Result<CommittedGeneration, ReceiverError> {
    let limits = policy.limits;
    let receipt = receive_v2(&mut *stream, key, policy, sink)?;
    write_frame_v2(
        &mut *stream,
        &WireFrameV2::Ack {
            transfer_id: receipt.transfer_id.clone(),
            generation: receipt.generation,
        },
        limits,
    )?;
    stream.flush().map_err(TransportError::from)?;
    Ok(receipt)
}

/// Live receiver with durable replay admission. Replay state is recorded only
/// after the engine transaction commits, and the wire ACK follows that record.
pub fn receive_v2_with_replay_ack(
    stream: &mut (impl Read + Write),
    key: MacKey,
    policy: ReceiverPolicy<'_>,
    replay: &mut ReplayLedger,
    sink: impl HandoffSinkV2,
) -> Result<CommittedGeneration, ReceiverError> {
    let admission = read_begin_v2(&mut *stream, policy.limits)?;
    receive_begun_v2_with_replay_ack(stream, admission.manifest, key, policy, replay, sink)
}

/// `receive_v2_with_replay_ack` for a caller that already read the Begin frame
/// off the connection in order to admit it.
pub fn receive_begun_v2_with_replay_ack(
    stream: &mut (impl Read + Write),
    manifest: BeginManifestV2,
    key: MacKey,
    policy: ReceiverPolicy<'_>,
    replay: &mut ReplayLedger,
    sink: impl HandoffSinkV2,
) -> Result<CommittedGeneration, ReceiverError> {
    receive_begun_v2_with_replay_ack_phased(stream, manifest, key, policy, replay, sink, None)
}

/// `receive_begun_v2_with_replay_ack` with per-phase timing evidence.
pub fn receive_begun_v2_with_replay_ack_phased(
    stream: &mut (impl Read + Write),
    manifest: BeginManifestV2,
    key: MacKey,
    policy: ReceiverPolicy<'_>,
    replay: &mut ReplayLedger,
    sink: impl HandoffSinkV2,
    phases: Option<SharedHandoffPhase>,
) -> Result<CommittedGeneration, ReceiverError> {
    let key_id = manifest.hmac.key_id.clone();
    let epoch = manifest.hmac.epoch;
    let generation = manifest.generation;
    replay.admit(&key_id, epoch, generation)?;
    let limits = policy.limits;
    let receipt = receive_after_begin_with_reservation_phased(
        &mut *stream,
        key,
        policy,
        sink,
        manifest,
        || {
            replay
                .reserve(&key_id, epoch, generation)
                .map_err(ReceiverError::from)
        },
        phases,
    )?;
    // The authenticated generation was durably consumed before the engine
    // transaction became live. ACK loss is therefore safely reconciled by a
    // retry observing Replay rather than reinstalling the generation.
    if let Err(error) = write_frame_v2(
        &mut *stream,
        &WireFrameV2::Ack {
            transfer_id: receipt.transfer_id.clone(),
            generation: receipt.generation,
        },
        limits,
    )
    .and_then(|()| stream.flush().map_err(TransportError::from))
    {
        eprintln!("muser-cluster: handoff ACK failed after commit: {error}");
    }
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::{write_frame_v2, WireFrameV2};
    use kvpack_handoff::{
        sha256_hex, BeginManifestV2, ComponentKindV2, ComponentV2, ExactIdentityV1, HandoffSinkV2,
        HmacIdentityV2, SealManifestV2, SegmentDescriptorV2, SegmentRoleV2, VerifiedSealV2,
        VerifiedSegmentV2, LIVE_HANDOFF_PROTOCOL_V2,
    };
    use std::sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    };
    #[derive(Clone)]
    struct Sink {
        live: Arc<AtomicU64>,
        aborted: Arc<AtomicBool>,
        generation: u64,
        segments: u32,
        bytes: u64,
    }
    impl HandoffSinkV2 for Sink {
        fn begin(&mut self, _: &ValidatedBeginV2) -> kvpack_handoff::Result<()> {
            Ok(())
        }
        fn segment_ready(&mut self, s: VerifiedSegmentV2) -> kvpack_handoff::Result<()> {
            self.segments += 1;
            self.bytes += s.payload().len() as u64;
            Ok(())
        }
        fn prepare_commit(&mut self, _: &VerifiedSealV2) -> kvpack_handoff::Result<()> {
            Ok(())
        }
        fn commit(&mut self) -> kvpack_handoff::Result<CommittedGeneration> {
            self.live.store(self.generation, Ordering::SeqCst);
            Ok(CommittedGeneration {
                transfer_id: "loop".into(),
                generation: self.generation,
                installed_segments: self.segments,
                installed_bytes: self.bytes,
            })
        }
        fn abort(&mut self) {
            self.aborted.store(true, Ordering::SeqCst)
        }
    }
    fn material() -> (BeginManifestV2, ValidatedBeginV2, MacKey, Vec<Vec<u8>>) {
        let payloads = vec![vec![1, 2, 3, 4]];
        let descriptors = vec![SegmentDescriptorV2 {
            sequence: 0,
            component_id: "target".into(),
            role: SegmentRoleV2::NopeKey,
            layer: Some(3),
            logical_start: 0,
            logical_count: 1,
            element_type: "f32_le".into(),
            elements_per_token: 1,
            byte_len: 4,
            sha256: sha256_hex(&payloads[0]),
        }];
        let manifest = BeginManifestV2 {
            protocol: LIVE_HANDOFF_PROTOCOL_V2.into(),
            transfer_id: "loop".into(),
            generation: 9,
            created_unix_ms: 10,
            expires_unix_ms: 30,
            identity: ExactIdentityV1 {
                adapter_sha256: "a".repeat(64),
                chat_template_sha256: "b".repeat(64),
                context_policy_sha256: "c".repeat(64),
                model_revision: "m".into(),
                model_sha256: "d".repeat(64),
                tokenizer_revision: "t".into(),
                tokenizer_sha256: "e".repeat(64),
            },
            prompt_token_ids: vec![1],
            multimodal: None,
            hmac: HmacIdentityV2 {
                key_id: "loop-key".into(),
                epoch: 4,
            },
            components: vec![ComponentV2 {
                id: "target".into(),
                kind: ComponentKindV2::TargetKv,
                required: true,
                identity_sha256: "f".repeat(64),
            }],
            deferred_segments: false,
            segments: descriptors,
        };
        let validated = ValidatedBeginV2::validate(manifest.clone(), 20, "loop-key", 4).unwrap();
        (manifest, validated, MacKey::from_bytes([7; 32]), payloads)
    }
    #[test]
    fn phased_loopback_records_segment_and_commit_phases() {
        let (manifest, validated, key, payloads) = material();
        let seal = SealManifestV2::sign(&validated, &manifest.segments, &payloads, &key).unwrap();
        let mut wire = Vec::new();
        let limits = FrameLimitsV2::default();
        // The phased loop receives an already-read Begin manifest.
        write_frame_v2(
            &mut wire,
            &WireFrameV2::Segment {
                sequence: 0,
                payload: payloads[0].clone(),
            },
            limits,
        )
        .unwrap();
        write_frame_v2(&mut wire, &WireFrameV2::Seal(seal), limits).unwrap();
        let phases = crate::phase::new_shared_phase();
        let receipt = receive_after_begin_with_reservation_phased(
            &mut wire.as_slice(),
            key,
            ReceiverPolicy {
                now_unix_ms: 20,
                expected_key_id: "loop-key",
                minimum_key_epoch: 4,
                limits,
            },
            Sink {
                live: Arc::new(AtomicU64::new(0)),
                aborted: Arc::new(AtomicBool::new(false)),
                generation: 9,
                segments: 0,
                bytes: 0,
            },
            manifest,
            || Ok(()),
            Some(std::sync::Arc::clone(&phases)),
        )
        .unwrap();
        assert_eq!(receipt.generation, 9);
        let phases = phases.lock().unwrap();
        assert_eq!(phases.segments.len(), 1);
        assert_eq!(phases.segments[0].sequence, 0);
        // Structural invariants, not wall-clock expectations: every phase is
        // accounted, totals match the per-segment sum, and the derived verify
        // time never goes negative.
        assert_eq!(
            phases.segment_read_ns,
            phases.segments.iter().map(|s| s.read_ns).sum::<u64>()
        );
        assert_eq!(
            phases.segment_process_ns,
            phases.segments.iter().map(|s| s.process_ns).sum::<u64>()
        );
        assert!(phases.sink_install_ns <= phases.segment_process_ns);
        assert_eq!(
            phases.verify_ns(),
            phases.segment_process_ns - phases.sink_install_ns
        );
    }

    #[test]
    fn loopback_commits_only_after_seal() {
        let (manifest, validated, key, payloads) = material();
        let seal = SealManifestV2::sign(&validated, &manifest.segments, &payloads, &key).unwrap();
        let mut wire = Vec::new();
        let limits = FrameLimitsV2::default();
        write_frame_v2(&mut wire, &WireFrameV2::Begin(manifest.into()), limits).unwrap();
        write_frame_v2(
            &mut wire,
            &WireFrameV2::Segment {
                sequence: 0,
                payload: payloads[0].clone(),
            },
            limits,
        )
        .unwrap();
        write_frame_v2(&mut wire, &WireFrameV2::Seal(seal), limits).unwrap();
        let live = Arc::new(AtomicU64::new(0));
        let aborted = Arc::new(AtomicBool::new(false));
        let sink = Sink {
            live: live.clone(),
            aborted: aborted.clone(),
            generation: 9,
            segments: 0,
            bytes: 0,
        };
        let receipt = receive_v2(
            &mut wire.as_slice(),
            key,
            ReceiverPolicy {
                now_unix_ms: 20,
                expected_key_id: "loop-key",
                minimum_key_epoch: 4,
                limits,
            },
            sink,
        )
        .unwrap();
        assert_eq!(receipt.generation, 9);
        assert_eq!(live.load(Ordering::SeqCst), 9);
        assert!(!aborted.load(Ordering::SeqCst));
    }
    #[test]
    fn truncation_aborts_and_preserves_previous_generation() {
        let (manifest, _validated, key, _) = material();
        let mut wire = Vec::new();
        let limits = FrameLimitsV2::default();
        write_frame_v2(&mut wire, &WireFrameV2::Begin(manifest.into()), limits).unwrap();
        let live = Arc::new(AtomicU64::new(3));
        let aborted = Arc::new(AtomicBool::new(false));
        let sink = Sink {
            live: live.clone(),
            aborted: aborted.clone(),
            generation: 9,
            segments: 0,
            bytes: 0,
        };
        assert!(receive_v2(
            &mut wire.as_slice(),
            key,
            ReceiverPolicy {
                now_unix_ms: 20,
                expected_key_id: "loop-key",
                minimum_key_epoch: 4,
                limits
            },
            sink
        )
        .is_err());
        assert_eq!(live.load(Ordering::SeqCst), 3);
        assert!(aborted.load(Ordering::SeqCst));
    }

    struct Duplex {
        input: std::io::Cursor<Vec<u8>>,
        output: Vec<u8>,
    }
    impl Read for Duplex {
        fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
            self.input.read(buffer)
        }
    }
    impl Write for Duplex {
        fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
            self.output.write(buffer)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn reservation_failure_preserves_the_previous_live_generation_and_latches() {
        let (manifest, validated, key, payloads) = material();
        let seal = SealManifestV2::sign(&validated, &manifest.segments, &payloads, &key).unwrap();
        let limits = FrameLimitsV2::default();
        let mut wire = Vec::new();
        write_frame_v2(&mut wire, &WireFrameV2::Begin(manifest.into()), limits).unwrap();
        write_frame_v2(
            &mut wire,
            &WireFrameV2::Segment {
                sequence: 0,
                payload: payloads[0].clone(),
            },
            limits,
        )
        .unwrap();
        write_frame_v2(&mut wire, &WireFrameV2::Seal(seal), limits).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let parent = directory.path().join("ledger");
        std::fs::create_dir(&parent).unwrap();
        let mut replay = ReplayLedger::load(parent.join("ledger.json")).unwrap();
        // The ledger can no longer be written: its parent becomes a regular file.
        std::fs::remove_file(parent.join("ledger.json")).unwrap();
        std::fs::remove_dir(&parent).unwrap();
        std::fs::write(&parent, b"not a directory").unwrap();
        let live = Arc::new(AtomicU64::new(0));
        let result = receive_v2_with_replay_ack(
            &mut Duplex {
                input: std::io::Cursor::new(wire),
                output: Vec::new(),
            },
            key,
            ReceiverPolicy {
                now_unix_ms: 20,
                expected_key_id: "loop-key",
                minimum_key_epoch: 4,
                limits,
            },
            &mut replay,
            Sink {
                live: Arc::clone(&live),
                aborted: Arc::new(AtomicBool::new(false)),
                generation: 9,
                segments: 0,
                bytes: 0,
            },
        );
        assert!(result.is_err());
        assert_eq!(live.load(Ordering::SeqCst), 0);
        assert!(
            replay.admit("loop-key", 4, 10).is_err(),
            "durability failure must latch the route degraded"
        );
    }

    #[test]
    fn deferred_descriptor_wire_commits_atomically() {
        let (mut manifest, _, key, payloads) = material();
        let descriptors = manifest.segments.clone();
        manifest.deferred_segments = true;
        manifest.segments.clear();
        let validated = ValidatedBeginV2::validate(manifest.clone(), 20, "loop-key", 4).unwrap();
        let seal = SealManifestV2::sign(&validated, &descriptors, &payloads, &key).unwrap();
        let limits = FrameLimitsV2::default();
        let mut wire = Vec::new();
        write_frame_v2(&mut wire, &WireFrameV2::Begin(manifest.into()), limits).unwrap();
        write_frame_v2(
            &mut wire,
            &WireFrameV2::DeferredSegment {
                descriptor: descriptors[0].clone(),
                payload: payloads[0].clone(),
            },
            limits,
        )
        .unwrap();
        write_frame_v2(&mut wire, &WireFrameV2::Seal(seal), limits).unwrap();
        let live = Arc::new(AtomicU64::new(0));
        let receipt = receive_v2(
            &mut wire.as_slice(),
            key,
            ReceiverPolicy {
                now_unix_ms: 20,
                expected_key_id: "loop-key",
                minimum_key_epoch: 4,
                limits,
            },
            Sink {
                live: Arc::clone(&live),
                aborted: Arc::new(AtomicBool::new(false)),
                generation: 9,
                segments: 0,
                bytes: 0,
            },
        )
        .unwrap();
        assert_eq!(receipt.installed_segments, 1);
        assert_eq!(live.load(Ordering::SeqCst), 9);
    }
}
