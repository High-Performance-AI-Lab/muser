//! Status view — what the registry says, plus what the network says now.
//!
//! The registry records the last thing that happened; the probe records
//! whether the daemon is answering this second. Both are reported, and they
//! are never conflated: a `healthy` node whose port is shut is exactly the
//! state an operator needs to see.

use std::io::Write;
use std::time::Duration;

use console::style;

use super::registry::{NodeEntry, Registry, DAEMON_PORT};
use super::ssh::Ssh;
use super::Result;

/// Contract: a one-second TCP connect, no longer. `muser node status` is
/// polled by the dashboard and must answer promptly with many nodes down.
const PROBE_TIMEOUT: Duration = Duration::from_secs(1);

pub fn run(home: &std::path::Path, json: bool) -> Result<()> {
    let registry = Registry::load(home)?;
    let probed = registry
        .nodes
        .iter()
        .map(|entry| (entry, probe(entry)))
        .collect::<Vec<_>>();

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    if json {
        let value = probed
            .iter()
            .map(|(entry, live)| {
                let mut object = serde_json::to_value(entry).unwrap_or(serde_json::Value::Null);
                if let Some(map) = object.as_object_mut() {
                    // Internal cache receipt: useful in the private registry,
                    // not part of the status or dashboard API.
                    map.remove("consumer_validation");
                    map.insert("daemon_port".into(), serde_json::json!(DAEMON_PORT));
                    map.insert("daemon_alive".into(), serde_json::json!(live.is_some()));
                    map.insert(
                        "daemon_connect_ms".into(),
                        match live {
                            Some(elapsed) => serde_json::json!(super::daemon::millis(*elapsed)),
                            None => serde_json::Value::Null,
                        },
                    );
                }
                object
            })
            .collect::<Vec<_>>();
        let text = serde_json::to_string_pretty(&value)
            .map_err(|error| format!("encode status: {error}"))?;
        let _ = writeln!(out, "{text}");
        return Ok(());
    }

    if probed.is_empty() {
        let _ = writeln!(
            out,
            "no nodes yet — add one with {}",
            style("muser node add user@host").bold()
        );
        return Ok(());
    }
    let _ = writeln!(
        out,
        "{:<16} {:<24} {:<13} {:<8} LINK",
        "NAME", "TARGET", "STATE", "DAEMON"
    );
    for (entry, live) in &probed {
        let daemon = match live {
            Some(_) => style("up").green(),
            None => style("down").red(),
        };
        let link = match (entry.netqual_gbps, entry.netqual_rtt_ms) {
            (Some(gbps), Some(rtt)) => format!("{gbps:.2} Gbps / {rtt:.2} ms"),
            _ => "unmeasured".to_string(),
        };
        let _ = writeln!(
            out,
            "{:<16} {:<24} {:<13} {:<8} {}",
            entry.name,
            format!("{}@{}", entry.user, entry.host),
            entry.state,
            daemon.to_string(),
            link
        );
        if let Some(error) = &entry.last_error {
            let _ = writeln!(out, "{:<16} {}", "", style(error).red());
        }
    }
    Ok(())
}

fn probe(entry: &NodeEntry) -> Option<Duration> {
    Ssh::new(&entry.user, &entry.host, None)
        .ok()?
        .tcp_probe(DAEMON_PORT, PROBE_TIMEOUT)
        .ok()
}
