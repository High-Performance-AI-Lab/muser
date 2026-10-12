//! The `muser.node-progress.v2` protocol: one JSON object per line on
//! stdout, relayed verbatim by the server as SSE `data:` events.
//!
//! Every line is flushed as it is written. The server tails this stream to
//! drive the dashboard's "Add node" button, so a buffered line is a stalled
//! progress bar; correctness here is "the reader sees it now", not
//! throughput.

use std::io::Write;

use console::style;
use serde::Serialize;

pub const PROGRESS_SCHEMA: &str = "muser.node-progress.v2";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Step {
    Preflight,
    Deploy,
    Model,
    Enroll,
    Daemon,
    Netqual,
    Smoke,
}

impl Step {
    pub fn label(self) -> &'static str {
        match self {
            Step::Preflight => "preflight",
            Step::Deploy => "deploy",
            Step::Model => "model",
            Step::Enroll => "enroll",
            Step::Daemon => "daemon",
            Step::Netqual => "netqual",
            Step::Smoke => "smoke",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    Start,
    Ok,
    Fail,
    Info,
    Planned,
}

#[derive(Serialize)]
struct Line<'a> {
    schema: &'static str,
    step: Step,
    status: Status,
    detail: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<&'a serde_json::Value>,
}

/// Emits the protocol either as JSON lines (`--json`, what the server
/// spawns) or as human lines (what an operator running the CLI sees).
#[derive(Debug, Clone, Copy)]
pub struct Progress {
    json: bool,
}

impl Progress {
    pub fn new(json: bool) -> Self {
        Self { json }
    }

    pub fn emit(&self, step: Step, status: Status, detail: &str) {
        self.write(step, status, detail, None);
    }

    pub fn emit_data(&self, step: Step, status: Status, detail: &str, data: serde_json::Value) {
        self.write(step, status, detail, Some(&data));
    }

    /// A dry-run plan line: what this step *would* do, never what it did.
    pub fn plan(&self, step: Step, detail: &str) {
        self.write(step, Status::Planned, &format!("would {detail}"), None);
    }

    /// A dry-run plan line carrying the exact argv the step would execute.
    pub fn plan_command(&self, step: Step, detail: &str, argv: &[String]) {
        let data = serde_json::json!({ "command": argv, "dry_run": true });
        self.write(
            step,
            Status::Planned,
            &format!("would {detail}"),
            Some(&data),
        );
    }

    fn write(&self, step: Step, status: Status, detail: &str, data: Option<&serde_json::Value>) {
        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        if self.json {
            let line = Line {
                schema: PROGRESS_SCHEMA,
                step,
                status,
                detail,
                data,
            };
            // A line that cannot be serialized would silently truncate the
            // stream the dashboard is reading; say so on stderr instead.
            match serde_json::to_string(&line) {
                Ok(text) => {
                    let _ = writeln!(out, "{text}");
                }
                Err(error) => eprintln!("muser node: progress line is unserializable: {error}"),
            }
        } else {
            let mark = match status {
                Status::Start => style("\u{2192}").cyan(),
                Status::Ok => style("\u{2713}").green(),
                Status::Fail => style("\u{2717}").red(),
                Status::Info => style("\u{00b7}").dim(),
                Status::Planned => style("\u{25cb}").dim(),
            };
            let _ = writeln!(
                out,
                "{mark} {:<9} {detail}",
                style(step.label()).bold().dim()
            );
            if let Some(value) = data.filter(|_| status != Status::Ok) {
                if let Some(command) = value.get("command").and_then(|v| v.as_array()) {
                    let rendered = command
                        .iter()
                        .filter_map(|v| v.as_str())
                        .collect::<Vec<_>>()
                        .join(" ");
                    let _ = writeln!(out, "  {:<9}   {}", "", style(rendered).dim());
                }
            }
        }
        let _ = out.flush();
    }
}
