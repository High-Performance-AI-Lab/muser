//! Minimal UTC RFC 3339 timestamp formatting, `std`-only.
//!
//! `docs/metrics-schema.json` requires `generated_at`/`SessionEvent.ts` as
//! `format: date-time` (ISO-8601 / RFC 3339) strings. There is no date/time
//! crate in this workspace (deliberately — see the workspace `Cargo.toml`
//! comment on keeping `muser`'s dependency footprint to what the muse-only
//! path actually needs) so this is a small, self-contained civil-calendar
//! conversion instead of pulling in `chrono`/`time`.
//!
//! The date math is Howard Hinnant's well-known `civil_from_days` algorithm
//! (<http://howardhinnant.github.io/date_algorithms.html>), which is exact
//! (no drift, correct leap years including the 100/400 rule) for any day
//! count a `SystemTime` on this machine could plausibly produce.

use std::time::{SystemTime, UNIX_EPOCH};

/// `(year, month, day)` for the given number of days since the Unix epoch
/// (1970-01-01), civil calendar, proleptic Gregorian. `days` may be
/// negative (pre-1970).
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// Format a Unix timestamp (seconds since epoch, may be fractional-truncated
/// by the caller) as an RFC 3339 UTC timestamp: `YYYY-MM-DDTHH:MM:SSZ`.
pub fn unix_to_rfc3339(unix_secs: i64) -> String {
    let days = unix_secs.div_euclid(86_400);
    let secs_of_day = unix_secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    let hh = secs_of_day / 3600;
    let mm = (secs_of_day % 3600) / 60;
    let ss = secs_of_day % 60;
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

/// RFC 3339 UTC timestamp for "now" (wall clock). Falls back to the epoch
/// string if the system clock is somehow before 1970 (`SystemTime` on a
/// sane host never is, but this avoids a panic on a `.unwrap()`).
pub fn now_rfc3339() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    unix_to_rfc3339(secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_is_1970_01_01() {
        assert_eq!(unix_to_rfc3339(0), "1970-01-01T00:00:00Z");
    }

    #[test]
    fn known_timestamp_round_trips_by_hand_arithmetic() {
        // 2024-01-01T00:00:00Z, cross-checked against `date -u -r 1704067200`.
        assert_eq!(unix_to_rfc3339(1_704_067_200), "2024-01-01T00:00:00Z");
    }

    #[test]
    fn mid_day_time_of_day_is_correct() {
        // 2024-01-01T12:34:56Z = 1704067200 + 12*3600 + 34*60 + 56.
        let secs = 1_704_067_200 + 12 * 3600 + 34 * 60 + 56;
        assert_eq!(unix_to_rfc3339(secs), "2024-01-01T12:34:56Z");
    }

    #[test]
    fn now_is_well_formed_and_in_the_2020s() {
        let s = now_rfc3339();
        assert_eq!(s.len(), 20);
        assert!(s.starts_with("20"));
        assert!(s.ends_with('Z'));
    }
}
