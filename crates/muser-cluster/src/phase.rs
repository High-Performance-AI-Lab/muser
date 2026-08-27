//! Per-phase handoff timing evidence (N series).
//!
//! Splits the receiver's transfer span into the phases the M2 link analysis
//! needs: active socket drain after each frame's first byte is readable
//! (excluding producer compute/pacing between frames), per-segment
//! verify+install processing (with the sink install subset measured inside
//! the sink), seal preparation, and the atomic commit. Times are nanoseconds
//! from `std::time::Instant` deltas.
//! Collection is structural: no phase total may exceed its enclosing span,
//! and per-segment entries are recorded in wire order.

use std::sync::{Arc, Mutex};

/// One segment's receive-phase times, in wire order.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct SegmentPhaseNanos {
    pub sequence: u32,
    /// Active frame drain after the first byte, excluding producer pacing.
    pub read_ns: u64,
    /// Time in `segment_ready*`: HMAC/hash verification plus sink install.
    pub process_ns: u64,
    /// The sink-install subset of `process_ns`, measured inside the sink.
    pub install_ns: u64,
    /// Frame-loop offset at which this segment's read began.
    pub read_started_offset_ns: u64,
}

/// Whole-handoff phase totals plus the per-segment breakdown.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct HandoffPhaseNanos {
    pub segments: Vec<SegmentPhaseNanos>,
    /// Sum of per-segment socket-drain time.
    pub segment_read_ns: u64,
    /// Frame-loop offset at which the seal frame's read began.
    pub seal_read_offset_ns: u64,
    /// Absolute unix nanoseconds at which the seal frame's read began.
    pub seal_read_unix_ns: u64,
    /// Sum of per-segment verify+install time.
    pub segment_process_ns: u64,
    /// Sum of the sink-install subset.
    pub sink_install_ns: u64,
    /// Seal frame read + `prepare_commit` (seal HMAC verify + sink prepare).
    pub seal_ns: u64,
    /// The atomic engine commit.
    pub commit_ns: u64,
    /// Install time of the in-flight segment, written by the sink and
    /// consumed by the frame loop when it records the segment entry.
    pub pending_install_ns: u64,
}

impl HandoffPhaseNanos {
    /// HMAC/hash verification time, derived as process minus sink install.
    pub fn verify_ns(&self) -> u64 {
        self.segment_process_ns.saturating_sub(self.sink_install_ns)
    }
}

/// Shared collector owned by the receiver and observed through the receipt.
pub type SharedHandoffPhase = Arc<Mutex<HandoffPhaseNanos>>;

/// `Duration` to nanoseconds, saturating.
pub fn nanos(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

pub fn new_shared_phase() -> SharedHandoffPhase {
    Arc::new(Mutex::new(HandoffPhaseNanos::default()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_time_is_process_minus_install() {
        let mut phases = HandoffPhaseNanos::default();
        phases.segments.push(SegmentPhaseNanos {
            sequence: 0,
            read_ns: 10,
            process_ns: 7,
            install_ns: 3,
            read_started_offset_ns: 0,
        });
        phases.segment_process_ns = 7;
        phases.sink_install_ns = 3;
        assert_eq!(phases.verify_ns(), 4);
        phases.sink_install_ns = 9; // inconsistent: never negative
        assert_eq!(phases.verify_ns(), 0);
    }
}
