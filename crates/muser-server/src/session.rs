//! In-process session state: token history, sampling state, KV handle,
//! origin (`created|resumed|migrated`) — the model the dashboard's
//! SESSIONS table (`docs/metrics-schema.md` §2) reads from.
//!
//! Complements `api::session` (the HTTP surface); this is the server-side
//! session struct itself. Source lineage: `ferrite-server/src/api/
//! session.rs` + `stages/session.rs`, PULL-AND-SIMPLIFY.
//!
//! [`SessionRegistry`] is populated for the lifetime of each OpenAI request.
//! It exposes the current prompt/decode token count and is removed on every
//! success, error, and cancellation path by an RAII guard. Durable prefix
//! generations remain private to `muser-kvpack`; they are not presented as
//! active clients.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::Instant;

use serde::Serialize;

use crate::timefmt;

/// Bounded length of the in-memory event log.
const EVENT_LOG_CAP: usize = 60;

/// How a session came to exist. Wire-shape match for
/// `docs/metrics-schema.json` `#/$defs/Session.origin`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Origin {
    #[serde(rename = "created")]
    Created,
    #[serde(rename = "resumed")]
    Resumed,
    #[serde(rename = "migrated")]
    Migrated,
}

#[derive(Debug)]
struct SessionRecord {
    id: String,
    tokens: u64,
    node: String,
    origin: Origin,
    started: Instant,
}

/// Wire-shape match for `docs/metrics-schema.json` `#/$defs/Session`.
#[derive(Debug, Clone, Serialize)]
pub struct SessionView {
    pub id: String,
    pub tokens: u64,
    pub node: String,
    pub age_s: f64,
    pub origin: Origin,
}

/// Live in-process session table. One instance lives in `ServerState` for
/// the process lifetime. Real: `create`/`remove`/`list` mutate/read actual
/// state behind a `Mutex`, ages are real `Instant::elapsed()` durations —
/// there is no simulated advance-by-tick here.
#[derive(Debug, Default)]
pub struct SessionRegistry(Mutex<Vec<SessionRecord>>);

impl SessionRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a newly created/resumed/migrated session.
    pub fn create(
        &self,
        id: impl Into<String>,
        tokens: u64,
        node: impl Into<String>,
        origin: Origin,
    ) {
        let mut sessions = self.0.lock().unwrap_or_else(|e| e.into_inner());
        sessions.push(SessionRecord {
            id: id.into(),
            tokens,
            node: node.into(),
            origin,
            started: Instant::now(),
        });
    }

    /// Remove a session (e.g. on completion/eviction). Returns `true` if a
    /// session with that id was present.
    pub fn remove(&self, id: &str) -> bool {
        let mut sessions = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let before = sessions.len();
        sessions.retain(|s| s.id != id);
        sessions.len() != before
    }

    /// Update the live token count for an in-flight session (e.g. as decode
    /// progresses). No-op if the id isn't present.
    pub fn set_tokens(&self, id: &str, tokens: u64) {
        let mut sessions = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(s) = sessions.iter_mut().find(|s| s.id == id) {
            s.tokens = tokens;
        }
    }

    /// Number of currently tracked sessions.
    pub fn len(&self) -> usize {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Snapshot the live table as the wire `Session[]` shape, ages computed
    /// from real elapsed wall time at the moment of the call.
    pub fn list(&self) -> Vec<SessionView> {
        let sessions = self.0.lock().unwrap_or_else(|e| e.into_inner());
        sessions
            .iter()
            .map(|s| SessionView {
                id: s.id.clone(),
                tokens: s.tokens,
                node: s.node.clone(),
                age_s: s.started.elapsed().as_secs_f64(),
                origin: s.origin,
            })
            .collect()
    }
}

/// Session lifecycle event kind. Wire-shape match for
/// `docs/metrics-schema.json` `#/$defs/SessionEvent.kind`. A superset of
/// [`Origin`] (adds `saved`/`evicted`, which aren't ways a session can
/// *begin*).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum EventKind {
    #[serde(rename = "created")]
    Created,
    #[serde(rename = "saved")]
    Saved,
    #[serde(rename = "resumed")]
    Resumed,
    #[serde(rename = "migrated")]
    Migrated,
    #[serde(rename = "evicted")]
    Evicted,
}

/// Which storage tier, for event detail lines. Wire-shape match for
/// `docs/metrics-schema.json` `#/$defs/SessionEvent.detail.tier`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DetailTier {
    #[serde(rename = "resident")]
    Resident,
    #[serde(rename = "ssd")]
    Ssd,
    #[serde(rename = "rdma_pool")]
    RdmaPool,
}

/// Wire-shape match for `docs/metrics-schema.json`
/// `#/$defs/SessionEvent.detail`. All fields optional — a caller fills in
/// whatever it genuinely knows for that event.
#[derive(Debug, Clone, Default, Serialize)]
pub struct EventDetail {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_node: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to_node: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tier: Option<DetailTier>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seconds_saved: Option<f64>,
}

/// Wire-shape match for `docs/metrics-schema.json` `#/$defs/SessionEvent`.
#[derive(Debug, Clone, Serialize)]
pub struct SessionEvent {
    pub ts: String,
    pub kind: EventKind,
    pub session: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<EventDetail>,
}

/// Live, bounded lifecycle-event log — the `_events` extension the
/// dashboard's event panel reads. Newest first, capped at [`EVENT_LOG_CAP`].
#[derive(Debug, Default)]
pub struct EventLog(Mutex<VecDeque<SessionEvent>>);

impl EventLog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&self, kind: EventKind, session: impl Into<String>, detail: Option<EventDetail>) {
        let mut log = self.0.lock().unwrap_or_else(|e| e.into_inner());
        log.push_front(SessionEvent {
            ts: timefmt::now_rfc3339(),
            kind,
            session: session.into(),
            detail,
        });
        while log.len() > EVENT_LOG_CAP {
            log.pop_back();
        }
    }

    pub fn list(&self) -> Vec<SessionEvent> {
        self.0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_starts_empty() {
        let r = SessionRegistry::new();
        assert!(r.is_empty());
        assert!(r.list().is_empty());
    }

    #[test]
    fn create_then_list_round_trips() {
        let r = SessionRegistry::new();
        r.create("sx-1", 4096, "m3ultra-0", Origin::Resumed);
        let views = r.list();
        assert_eq!(views.len(), 1);
        assert_eq!(views[0].id, "sx-1");
        assert_eq!(views[0].tokens, 4096);
        assert_eq!(views[0].node, "m3ultra-0");
        assert_eq!(views[0].origin, Origin::Resumed);
        assert!(views[0].age_s >= 0.0);
    }

    #[test]
    fn remove_reports_whether_it_was_present() {
        let r = SessionRegistry::new();
        r.create("sx-1", 100, "m3ultra-0", Origin::Created);
        assert!(r.remove("sx-1"));
        assert!(!r.remove("sx-1"));
        assert!(r.is_empty());
    }

    #[test]
    fn event_log_is_bounded_and_newest_first() {
        let log = EventLog::new();
        for i in 0..(EVENT_LOG_CAP + 10) {
            log.push(EventKind::Created, format!("sx-{i}"), None);
        }
        let events = log.list();
        assert_eq!(events.len(), EVENT_LOG_CAP);
        // newest push was sx-(CAP+9); it must be first.
        assert_eq!(events[0].session, format!("sx-{}", EVENT_LOG_CAP + 9));
    }
}
