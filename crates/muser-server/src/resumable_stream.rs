//! Bounded resumable SSE sessions for the pinned llama-server stream API.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tokio::sync::Notify;

const MAX_STREAM_BYTES: usize = 4 * 1024 * 1024;
const COMPLETED_TTL: Duration = Duration::from_secs(300);

#[derive(Clone, Default)]
pub(crate) struct StreamManager {
    sessions: Arc<Mutex<HashMap<String, Arc<StreamSession>>>>,
}

pub(crate) struct StreamSession {
    id: String,
    started_at: u64,
    inner: Mutex<Inner>,
    changed: Notify,
}

pub(crate) struct StreamFinishGuard(Arc<StreamSession>);

impl Drop for StreamFinishGuard {
    fn drop(&mut self) {
        self.0.finish();
    }
}

struct Inner {
    bytes: VecDeque<u8>,
    dropped: usize,
    total: usize,
    done: bool,
    cancelled: bool,
    completed_at: u64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct StreamView {
    pub conversation_id: String,
    pub is_done: bool,
    pub total_bytes: usize,
    pub started_at: u64,
    pub completed_at: u64,
}

pub(crate) enum ReadSnapshot {
    Lost,
    Data(Vec<u8>),
    Pending,
    Done,
}

impl StreamManager {
    pub(crate) fn create_or_replace(&self, id: String) -> Arc<StreamSession> {
        self.gc();
        let fresh = Arc::new(StreamSession {
            id: id.clone(),
            started_at: now(),
            inner: Mutex::new(Inner {
                bytes: VecDeque::with_capacity(64 * 1024),
                dropped: 0,
                total: 0,
                done: false,
                cancelled: false,
                completed_at: 0,
            }),
            changed: Notify::new(),
        });
        if let Some(previous) = lock(&self.sessions).insert(id, Arc::clone(&fresh)) {
            previous.cancel();
        }
        fresh
    }

    pub(crate) fn get(&self, id: &str) -> Option<Arc<StreamSession>> {
        self.gc();
        lock(&self.sessions).get(id).cloned()
    }

    pub(crate) fn lookup(&self, requested: &[String]) -> Vec<StreamView> {
        self.gc();
        let sessions = lock(&self.sessions);
        let mut views = Vec::new();
        for requested in requested {
            let prefix = format!("{requested}::");
            for session in sessions.values() {
                if session.id == *requested || session.id.starts_with(&prefix) {
                    let view = session.view();
                    if !views.iter().any(|existing: &StreamView| {
                        existing.conversation_id == view.conversation_id
                    }) {
                        views.push(view);
                    }
                }
            }
        }
        views.sort_by(|left, right| left.conversation_id.cmp(&right.conversation_id));
        views
    }

    pub(crate) fn evict_and_cancel(&self, id: &str) {
        if let Some(session) = lock(&self.sessions).remove(id) {
            session.cancel();
        }
    }

    fn gc(&self) {
        let cutoff = now().saturating_sub(COMPLETED_TTL.as_secs());
        lock(&self.sessions).retain(|_, session| {
            let completed = lock(&session.inner).completed_at;
            completed == 0 || completed > cutoff
        });
    }
}

impl StreamSession {
    pub(crate) fn finish_guard(self: &Arc<Self>) -> StreamFinishGuard {
        StreamFinishGuard(Arc::clone(self))
    }

    pub(crate) fn append(&self, bytes: &[u8]) -> bool {
        if bytes.is_empty() {
            return true;
        }
        let mut inner = lock(&self.inner);
        if inner.done || inner.cancelled {
            return false;
        }
        inner.total = inner.total.saturating_add(bytes.len());
        if bytes.len() >= MAX_STREAM_BYTES {
            inner.dropped = inner.total - MAX_STREAM_BYTES;
            inner.bytes.clear();
            inner
                .bytes
                .extend(bytes[bytes.len() - MAX_STREAM_BYTES..].iter().copied());
        } else {
            let excess = inner
                .bytes
                .len()
                .saturating_add(bytes.len())
                .saturating_sub(MAX_STREAM_BYTES);
            for _ in 0..excess {
                inner.bytes.pop_front();
            }
            inner.dropped = inner.dropped.saturating_add(excess);
            inner.bytes.extend(bytes.iter().copied());
        }
        drop(inner);
        self.changed.notify_waiters();
        true
    }

    pub(crate) fn finish(&self) {
        let mut inner = lock(&self.inner);
        if !inner.done {
            inner.done = true;
            inner.completed_at = now();
        }
        drop(inner);
        self.changed.notify_waiters();
    }

    pub(crate) fn cancel(&self) {
        let mut inner = lock(&self.inner);
        inner.cancelled = true;
        inner.done = true;
        inner.completed_at = now();
        drop(inner);
        self.changed.notify_waiters();
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        lock(&self.inner).cancelled
    }

    pub(crate) fn snapshot(&self, offset: usize) -> ReadSnapshot {
        let inner = lock(&self.inner);
        if offset < inner.dropped {
            return ReadSnapshot::Lost;
        }
        let end = inner.dropped + inner.bytes.len();
        if offset < end {
            return ReadSnapshot::Data(
                inner
                    .bytes
                    .iter()
                    .skip(offset - inner.dropped)
                    .copied()
                    .collect(),
            );
        }
        if inner.done {
            ReadSnapshot::Done
        } else {
            ReadSnapshot::Pending
        }
    }

    pub(crate) async fn changed(&self) {
        self.changed.notified().await;
    }

    fn view(&self) -> StreamView {
        let inner = lock(&self.inner);
        StreamView {
            conversation_id: self.id.clone(),
            is_done: inner.done,
            total_bytes: inner.total,
            started_at: self.started_at,
            completed_at: inner.completed_at,
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replacement_cancels_old_and_lookup_never_enumerates_unknown_ids() {
        let manager = StreamManager::default();
        let old = manager.create_or_replace("known".into());
        let fresh = manager.create_or_replace("known".into());
        assert!(old.is_cancelled());
        assert!(!fresh.is_cancelled());
        assert!(manager.lookup(&["unknown".into()]).is_empty());
        assert_eq!(manager.lookup(&["known".into()]).len(), 1);
    }

    #[test]
    fn ring_reports_offsets_that_fell_out_of_the_bound() {
        let manager = StreamManager::default();
        let session = manager.create_or_replace("x".into());
        assert!(session.append(&vec![b'a'; MAX_STREAM_BYTES + 7]));
        assert!(matches!(session.snapshot(0), ReadSnapshot::Lost));
        assert!(
            matches!(session.snapshot(7), ReadSnapshot::Data(bytes) if bytes.len() == MAX_STREAM_BYTES)
        );
    }
}
