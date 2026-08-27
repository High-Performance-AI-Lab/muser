//! SSH/SCP shell-outs for the node pipeline.
//!
//! `BatchMode=yes` is unconditional: this process never prompts for, reads,
//! or stores a password, and a host that would ask for one fails fast
//! instead of hanging the dashboard's progress stream. Authentication is the
//! agent/keychain, plus an optional `-i <key>`.
//!
//! Remote scripts are fed on stdin (`bash -s -- <args>`) so no caller-
//! supplied value is ever pasted into a shell command line; the values that
//! *do* reach an argv (user, host, paths) are checked against closed
//! grammars first.

use std::io::Write;
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use super::Result;

/// Applied to every SSH/SCP invocation. `ConnectTimeout` keeps an
/// unreachable node from stalling a step for the TCP default.
const SSH_OPTIONS: [&str; 6] = [
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "StrictHostKeyChecking=yes",
];

#[derive(Debug, Clone)]
pub struct Ssh {
    pub user: String,
    pub host: String,
    pub key_path: Option<PathBuf>,
}

/// What a remote script did.
#[derive(Debug, Clone)]
pub struct Outcome {
    pub code: i32,
    pub stdout: String,
    pub stderr: String,
}

impl Outcome {
    pub fn failure(&self, target: &str) -> String {
        let detail = if self.stderr.is_empty() {
            self.stdout.trim().to_string()
        } else {
            self.stderr.clone()
        };
        format!("ssh {target} exited {}: {detail}", self.code)
    }
}

impl Ssh {
    pub fn new(user: &str, host: &str, key_path: Option<&Path>) -> Result<Self> {
        validate_user(user)?;
        validate_host(host)?;
        Ok(Self {
            user: user.to_string(),
            host: host.to_string(),
            key_path: key_path.map(Path::to_path_buf),
        })
    }

    pub fn target(&self) -> String {
        format!("{}@{}", self.user, self.host)
    }

    fn identity(&self) -> Vec<String> {
        match &self.key_path {
            Some(path) => vec!["-i".to_string(), path.display().to_string()],
            None => Vec::new(),
        }
    }

    /// The exact argv a `run` would execute, for `--dry-run` plan lines. The
    /// script itself travels on stdin and is therefore not part of argv;
    /// callers pass a one-line summary to the progress emitter instead.
    pub fn argv(&self, args: &[&str]) -> Vec<String> {
        let mut argv = vec!["ssh".to_string()];
        argv.extend(SSH_OPTIONS.iter().map(|value| value.to_string()));
        argv.extend(self.identity());
        argv.push(self.target());
        argv.push("bash".into());
        argv.push("-s".into());
        argv.push("--".into());
        argv.extend(args.iter().map(|value| value.to_string()));
        argv
    }

    pub fn scp_argv(&self, local: &Path, remote_path: &str) -> Vec<String> {
        let mut argv = vec!["scp".to_string(), "-q".to_string()];
        argv.extend(SSH_OPTIONS.iter().map(|value| value.to_string()));
        argv.extend(self.identity());
        argv.push(local.display().to_string());
        argv.push(format!("{}:{remote_path}", self.target()));
        argv
    }

    /// The remote script's outcome: its exit code, its stdout, and the tail
    /// of its stderr. A non-zero exit is data here, not an error — several
    /// steps ask questions whose answer is "no".
    pub fn exec(
        &self,
        script: &str,
        args: &[&str],
        relay: Option<&dyn Fn(&str)>,
    ) -> Result<Outcome> {
        let argv = self.argv(args);
        let mut child = Command::new(&argv[0])
            .args(&argv[1..])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| format!("spawn ssh {}: {error}", self.target()))?;
        child
            .stdin
            .as_mut()
            .ok_or("ssh stdin is unavailable")?
            .write_all(script.as_bytes())
            .map_err(|error| format!("write remote script: {error}"))?;
        drop(child.stdin.take());

        // stderr is drained on its own thread: a chatty remote script must
        // not deadlock against a full pipe while we are reading stdout.
        let mut errors = child.stderr.take().ok_or("ssh stderr is unavailable")?;
        let draining = std::thread::spawn(move || {
            let mut text = String::new();
            let _ = std::io::Read::read_to_string(&mut errors, &mut text);
            text
        });
        let mut stdout = String::new();
        {
            let pipe = child.stdout.take().ok_or("ssh stdout is unavailable")?;
            let reader = std::io::BufReader::new(pipe);
            for line in std::io::BufRead::lines(reader) {
                let line = line.map_err(|error| format!("read remote output: {error}"))?;
                if let Some(relay) = relay {
                    relay(&line);
                }
                stdout.push_str(&line);
                stdout.push('\n');
            }
        }
        let status = child
            .wait()
            .map_err(|error| format!("wait for ssh {}: {error}", self.target()))?;
        let stderr = draining.join().unwrap_or_default();
        Ok(Outcome {
            code: status.code().unwrap_or(-1),
            stdout,
            stderr: tail(&stderr),
        })
    }

    /// Run `script` on the node with `args` in `$1..$n`, failing on any
    /// non-zero exit. Returns stdout.
    pub fn run(&self, script: &str, args: &[&str]) -> Result<String> {
        let outcome = self.exec(script, args, None)?;
        if outcome.code == 0 {
            return Ok(outcome.stdout);
        }
        Err(outcome.failure(&self.target()))
    }

    /// Like `run`, but relays each stdout line as it arrives — the long
    /// steps (a model download, a daemon start) print their own progress.
    pub fn run_relayed(&self, script: &str, args: &[&str], relay: &dyn Fn(&str)) -> Result<String> {
        let outcome = self.exec(script, args, Some(relay))?;
        if outcome.code == 0 {
            return Ok(outcome.stdout);
        }
        Err(outcome.failure(&self.target()))
    }

    pub fn scp(&self, local: &Path, remote_path: &str) -> Result<()> {
        validate_remote_path(remote_path)?;
        let argv = self.scp_argv(local, remote_path);
        let output = Command::new(&argv[0])
            .args(&argv[1..])
            .output()
            .map_err(|error| format!("spawn scp: {error}"))?;
        if output.status.success() {
            return Ok(());
        }
        Err(format!(
            "scp {} -> {remote_path} failed: {}",
            local.display(),
            tail(&String::from_utf8_lossy(&output.stderr))
        ))
    }

    pub fn scp_from(&self, remote_path: &str, local: &Path) -> Result<()> {
        validate_remote_path(remote_path)?;
        let mut argv = vec!["scp".to_string(), "-q".to_string()];
        argv.extend(SSH_OPTIONS.iter().map(|value| value.to_string()));
        argv.extend(self.identity());
        argv.push(format!("{}:{remote_path}", self.target()));
        argv.push(local.display().to_string());
        let output = Command::new(&argv[0])
            .args(&argv[1..])
            .output()
            .map_err(|error| format!("spawn scp: {error}"))?;
        if output.status.success() {
            return Ok(());
        }
        Err(format!(
            "scp {remote_path} -> {} failed: {}",
            local.display(),
            tail(&String::from_utf8_lossy(&output.stderr))
        ))
    }

    /// The hostname ssh would actually dial, from `ssh -G`. Resolves
    /// ~/.ssh/config aliases that plain getaddrinfo cannot; falls back to the
    /// literal host when ssh gives nothing usable.
    pub fn effective_host(&self) -> String {
        let mut command = Command::new("ssh");
        command.arg("-G");
        if let Some(key) = &self.key_path {
            command.arg("-i").arg(key);
        }
        command.arg(format!("{}@{}", self.user, self.host));
        if let Ok(output) = command.output() {
            if output.status.success() {
                for line in String::from_utf8_lossy(&output.stdout).lines() {
                    if let Some(rest) = line.strip_prefix("hostname ") {
                        let candidate = rest.trim();
                        if !candidate.is_empty() {
                            return candidate.to_string();
                        }
                    }
                }
            }
        }
        self.host.clone()
    }

    /// One TCP connect, timed. Used for the daemon port wait, the live
    /// status probe, and the RTT estimate.
    pub fn tcp_probe(&self, port: u16, timeout: Duration) -> Result<Duration> {
        let address = match (self.host.as_str(), port).to_socket_addrs() {
            Ok(mut addresses) => addresses.next(),
            Err(_) => None,
        };
        let address = match address {
            Some(address) => address,
            // The host may be an ssh alias getaddrinfo cannot see.
            None => {
                let effective = self.effective_host();
                (effective.as_str(), port)
                    .to_socket_addrs()
                    .map_err(|error| {
                        format!("resolve {}:{port} (via {effective}): {error}", self.host)
                    })?
                    .next()
                    .ok_or_else(|| {
                        format!("{}:{port} ({effective}) resolved to no address", self.host)
                    })?
            }
        };
        let started = Instant::now();
        TcpStream::connect_timeout(&address, timeout)
            .map_err(|error| format!("connect {address}: {error}"))?;
        Ok(started.elapsed())
    }
}

/// `user@host`, both halves inside their closed grammars.
pub fn parse_target(target: &str) -> Result<(String, String)> {
    let (user, host) = target
        .split_once('@')
        .ok_or("a node target must be user@host")?;
    validate_user(user)?;
    validate_host(host)?;
    Ok((user.to_string(), host.to_string()))
}

pub fn validate_user(user: &str) -> Result<()> {
    if user.is_empty()
        || user.len() > 32
        || !user
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
    {
        return Err(format!("ssh user {user:?} is outside its closed grammar"));
    }
    Ok(())
}

pub fn validate_host(host: &str) -> Result<()> {
    if host.is_empty()
        || host.len() > 253
        || !host
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b".-_".contains(&byte))
    {
        return Err(format!("ssh host {host:?} is outside its closed grammar"));
    }
    Ok(())
}

/// Registry names become local directory names and remote path components.
pub fn validate_name(name: &str) -> Result<()> {
    if name.is_empty()
        || name.len() > 64
        || name.starts_with('.')
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b".-_".contains(&byte))
    {
        return Err(format!("node name {name:?} is outside its closed grammar"));
    }
    Ok(())
}

/// Remote paths are absolute, traversal-free and free of shell metacharacters.
pub fn validate_remote_path(path: &str) -> Result<()> {
    if !path.starts_with('/')
        || path.len() > 4096
        || path.contains("..")
        || !path
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"/._-".contains(&byte))
    {
        return Err(format!(
            "remote path {path:?} is outside its closed grammar"
        ));
    }
    Ok(())
}

fn tail(text: &str) -> String {
    let trimmed = text.trim();
    match trimmed.char_indices().nth_back(600) {
        Some((index, _)) => format!("...{}", &trimmed[index..]),
        None => trimmed.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_mode_is_never_optional() {
        let ssh = Ssh::new("muser", "gx10.local", None).unwrap();
        let argv = ssh.argv(&["one"]);
        assert!(argv.windows(2).any(|pair| pair == ["-o", "BatchMode=yes"]));
        assert_eq!(argv.last().unwrap(), "one");
        assert!(argv.contains(&"muser@gx10.local".to_string()));
    }

    #[test]
    fn identity_file_is_passed_through_when_present() {
        let ssh = Ssh::new("muser", "gx10.local", Some(Path::new("/k/id_ed25519"))).unwrap();
        let argv = ssh.argv(&[]);
        assert!(argv.windows(2).any(|pair| pair == ["-i", "/k/id_ed25519"]));
    }

    #[test]
    fn closed_grammars_refuse_injection() {
        assert!(parse_target("root@gx10; rm -rf /").is_err());
        assert!(parse_target("gx10.local").is_err());
        assert!(validate_name("../escape").is_err());
        assert!(validate_remote_path("relative/path").is_err());
        assert!(validate_remote_path("/lane/../etc").is_err());
        assert!(validate_remote_path("/home/muser/.muser/lane/gx10").is_ok());
    }
}
