use std::collections::BTreeSet;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::Duration;

use kvpack_handoff::ExactIdentityV1;
use muser_engine::dflash::DFlashContextGeometry;
use serde::Deserialize;

/// Numeric contract selected by the Spark NVFP4 producer. Legacy receiver
/// configurations predate the split and therefore deserialize as `None`;
/// newly generated F-series configurations must name the mode explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Nvfp4ProducerMode {
    Exact,
    Native,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReceiverConfigV2 {
    pub schema_version: u32,
    pub listen: SocketAddr,
    pub certificate_chain: PathBuf,
    pub private_key: PathBuf,
    pub peer_ca: PathBuf,
    pub peer_leaf_sha256: BTreeSet<String>,
    pub hmac_key_file: PathBuf,
    pub hmac_key_id: String,
    pub minimum_hmac_epoch: u64,
    pub replay_ledger: PathBuf,
    pub timeout_ms: u64,
    pub wait_for_producer_ms: u64,
    #[serde(default)]
    pub advertised_receiver_host: Option<String>,
    #[serde(default)]
    pub producer_control: Option<ProducerControlV1>,
    #[serde(default)]
    pub producer_mode: Option<Nvfp4ProducerMode>,
    pub identity: ExactIdentityV1,
    pub target_cache_identity_sha256: String,
    #[serde(default)]
    pub dflash_identity_sha256: Option<String>,
    /// Context shape stamped during enrollment from the digest-verified
    /// DFlash sidecar. It is paired with the component digest so a receiver
    /// can never silently substitute its own window.
    #[serde(default)]
    pub dflash_context_geometry: Option<DFlashContextGeometry>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProducerControlV1 {
    pub address: SocketAddr,
    pub server_name: String,
}

impl ReceiverConfigV2 {
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|error| format!("read cluster config {}: {error}", path.display()))?;
        let mut config: Self = serde_json::from_slice(&bytes)
            .map_err(|error| format!("parse cluster config {}: {error}", path.display()))?;
        let base = path.parent().unwrap_or_else(|| Path::new("."));
        for candidate in [
            &mut config.certificate_chain,
            &mut config.private_key,
            &mut config.peer_ca,
            &mut config.hmac_key_file,
            &mut config.replay_ledger,
        ] {
            if candidate.is_relative() {
                *candidate = base.join(&*candidate);
            }
        }
        config.validate()?;
        Ok(config)
    }

    pub fn timeout(&self) -> Duration {
        Duration::from_millis(self.timeout_ms)
    }

    pub fn producer_wait(&self) -> Duration {
        Duration::from_millis(self.wait_for_producer_ms)
    }

    /// Producer wait scaled to the requested depth. The configured wait is
    /// the floor; deep prompts earn more patience, because the kquant-lane
    /// producer was measured prefilling ~12.8k tokens in about six minutes
    /// (2026-08-28) — past any flat budget that suits shallow prompts. The
    /// allowance is 35 ms per prompt token (a ~28 tok/s worst-case
    /// producer), capped at the 900 s config ceiling.
    pub fn producer_wait_for(&self, prompt_tokens: usize) -> Duration {
        let scaled = Duration::from_millis((prompt_tokens as u64).saturating_mul(35));
        self.producer_wait()
            .max(scaled)
            .min(Duration::from_millis(900_000))
    }

    fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1 {
            return Err("cluster config schema_version must be 1".into());
        }
        if self.timeout_ms == 0
            || self.timeout_ms > 900_000
            || self.wait_for_producer_ms == 0
            || self.wait_for_producer_ms > self.timeout_ms
            || self.minimum_hmac_epoch == 0
            || self.hmac_key_id.is_empty()
            || self.hmac_key_id.len() > 128
        {
            return Err("cluster config timing or HMAC bounds are invalid".into());
        }
        for (label, digest) in [
            ("model", self.identity.model_sha256.as_str()),
            ("tokenizer", self.identity.tokenizer_sha256.as_str()),
            ("chat template", self.identity.chat_template_sha256.as_str()),
            (
                "context policy",
                self.identity.context_policy_sha256.as_str(),
            ),
            ("adapter", self.identity.adapter_sha256.as_str()),
            ("target cache", self.target_cache_identity_sha256.as_str()),
        ] {
            validate_digest(label, digest)?;
        }
        match (&self.dflash_identity_sha256, self.dflash_context_geometry) {
            (Some(digest), Some(geometry)) => {
                validate_digest("DFlash", digest)?;
                geometry.validate()?;
            }
            (None, None) => {}
            _ => {
                return Err(
                    "DFlash identity and dflash_context_geometry must be declared \
                     together; if this cluster config predates the current schema, \
                     re-run `muser node add <user@host>` (or the dashboard's Add \
                     node) to regenerate it"
                        .into(),
                )
            }
        }
        if self.producer_mode == Some(Nvfp4ProducerMode::Native)
            && self.dflash_identity_sha256.is_some()
        {
            return Err("native producer mode cannot enroll DFlash context geometry".into());
        }
        if self.peer_leaf_sha256.is_empty() {
            return Err("cluster config must pin at least one producer TLS leaf".into());
        }
        if let Some(control) = &self.producer_control {
            let advertised = self
                .advertised_receiver_host
                .as_deref()
                .ok_or("producer_control requires advertised_receiver_host")?;
            if control.server_name.is_empty()
                || control.server_name.len() > 253
                || advertised.is_empty()
                || advertised.len() > 253
                || !control
                    .server_name
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b".-_".contains(&byte))
                || !advertised
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b".:-_".contains(&byte))
            {
                return Err("producer control server/receiver host is invalid".into());
            }
        }
        for pin in &self.peer_leaf_sha256 {
            validate_digest("producer TLS leaf", pin)?;
        }
        for (label, path) in [
            ("certificate chain", &self.certificate_chain),
            ("private key", &self.private_key),
            ("peer CA", &self.peer_ca),
            ("HMAC key", &self.hmac_key_file),
        ] {
            if !path.is_file() {
                return Err(format!("{label} is not a file: {}", path.display()));
            }
        }
        Ok(())
    }
}

fn validate_digest(label: &str, value: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{label} identity must be lowercase SHA-256"));
    }
    Ok(())
}
