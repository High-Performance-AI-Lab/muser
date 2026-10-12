//! Fail-closed durable cache configuration for the server.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use kvpack::{load_store_key, LocalStore, StoreConfig};
use muser_engine::config::MuseConfig;
use serde::Deserialize;

use crate::layout::MuseIdentity;
use crate::session::DurableCache;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DurableConfigV1 {
    pub schema_version: u32,
    pub root: PathBuf,
    pub key_file: PathBuf,
    pub operator_tenant_id: String,
    pub key_epoch: u64,
    pub minimum_readable_key_epoch: u64,
    pub catalog_epoch: u64,
    pub quota_bytes: u64,
    pub staging_quota_bytes: u64,
    pub endurance_bytes_per_five_minutes: u64,
    pub identity: IdentityV1,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IdentityV1 {
    pub model_sha256: String,
    pub adapter_sha256: String,
    pub tokenizer_sha256: String,
    pub chat_template_sha256: String,
    pub context_policy_sha256: String,
    pub model_revision: String,
    pub tokenizer_revision: String,
    pub weight_precision: String,
}

#[derive(Debug, thiserror::Error)]
pub enum DurableConfigError {
    #[error("read durable config {0}: {1}")]
    Read(PathBuf, std::io::Error),
    #[error("parse durable config: {0}")]
    Parse(serde_json::Error),
    #[error("durable config is invalid: {0}")]
    Invalid(String),
    #[error(transparent)]
    Store(#[from] kvpack::StoreError),
}

impl DurableConfigV1 {
    pub fn load(path: &Path) -> Result<Self, DurableConfigError> {
        let bytes = std::fs::read(path)
            .map_err(|error| DurableConfigError::Read(path.to_path_buf(), error))?;
        let mut value: Self = serde_json::from_slice(&bytes).map_err(DurableConfigError::Parse)?;
        let base = path.parent().unwrap_or_else(|| Path::new("."));
        if value.root.is_relative() {
            value.root = base.join(&value.root);
        }
        if value.key_file.is_relative() {
            value.key_file = base.join(&value.key_file);
        }
        value.validate()?;
        Ok(value)
    }

    pub fn model_sha256(&self) -> Result<[u8; 32], DurableConfigError> {
        digest("model_sha256", &self.identity.model_sha256)
    }

    pub fn open(self, config: MuseConfig) -> Result<DurableCache, DurableConfigError> {
        let identity = MuseIdentity {
            model_sha256: digest("model_sha256", &self.identity.model_sha256)?,
            adapter_sha256: digest("adapter_sha256", &self.identity.adapter_sha256)?,
            tokenizer_sha256: digest("tokenizer_sha256", &self.identity.tokenizer_sha256)?,
            chat_template_sha256: digest(
                "chat_template_sha256",
                &self.identity.chat_template_sha256,
            )?,
            context_policy_sha256: digest(
                "context_policy_sha256",
                &self.identity.context_policy_sha256,
            )?,
            model_revision: self.identity.model_revision,
            tokenizer_revision: self.identity.tokenizer_revision,
            weight_precision: self.identity.weight_precision,
        };
        let key = load_store_key(&self.key_file, &self.root)?;
        let store = Arc::new(LocalStore::open(
            StoreConfig {
                object_root: self.root.join("objects"),
                catalog_path: self.root.join("catalog/catalog.sqlite"),
                operator_tenant_id: self.operator_tenant_id.into_bytes(),
                key_epoch: self.key_epoch,
                minimum_readable_key_epoch: self.minimum_readable_key_epoch,
                catalog_epoch: self.catalog_epoch,
                quota_bytes: self.quota_bytes,
                staging_quota_bytes: self.staging_quota_bytes,
                endurance_bytes_per_five_minutes: self.endurance_bytes_per_five_minutes,
            },
            key,
        )?);
        Ok(DurableCache::new(store, config, identity))
    }

    fn validate(&self) -> Result<(), DurableConfigError> {
        if self.schema_version != 1
            || self.operator_tenant_id.is_empty()
            || self.operator_tenant_id.len() > 256
            || self.key_epoch == 0
            || self.minimum_readable_key_epoch == 0
            || self.minimum_readable_key_epoch > self.key_epoch
            || self.catalog_epoch == 0
            || self.quota_bytes == 0
            || self.staging_quota_bytes == 0
            || self.endurance_bytes_per_five_minutes == 0
            || self.identity.model_revision.is_empty()
            || self.identity.tokenizer_revision.is_empty()
            || !matches!(self.identity.weight_precision.as_str(), "q4_k_xl" | "nvfp4")
        {
            return Err(DurableConfigError::Invalid(
                "schema, epochs, quotas, labels, or weight precision".into(),
            ));
        }
        for (name, value) in [
            ("model_sha256", &self.identity.model_sha256),
            ("adapter_sha256", &self.identity.adapter_sha256),
            ("tokenizer_sha256", &self.identity.tokenizer_sha256),
            ("chat_template_sha256", &self.identity.chat_template_sha256),
            (
                "context_policy_sha256",
                &self.identity.context_policy_sha256,
            ),
        ] {
            digest(name, value)?;
        }
        if !self.key_file.is_file() || self.root.as_os_str().is_empty() {
            return Err(DurableConfigError::Invalid(
                "key file is absent or root is empty".into(),
            ));
        }
        Ok(())
    }
}

fn digest(name: &str, value: &str) -> Result<[u8; 32], DurableConfigError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(DurableConfigError::Invalid(format!(
            "{name} must be lowercase SHA-256"
        )));
    }
    let mut output = [0u8; 32];
    for (index, chunk) in value.as_bytes().chunks_exact(2).enumerate() {
        let text = std::str::from_utf8(chunk).expect("validated ASCII");
        output[index] = u8::from_str_radix(text, 16).expect("validated hexadecimal");
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kvpack::create_store_key_random;

    #[test]
    fn relative_config_resolves_inside_its_directory() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("store");
        std::fs::create_dir_all(root.join("keys")).unwrap();
        create_store_key_random(&root.join("keys/root.key"), &root).unwrap();
        let d = "11".repeat(32);
        let body = serde_json::json!({
            "schema_version": 1,
            "root": "store",
            "key_file": "store/keys/root.key",
            "operator_tenant_id": "muser-test",
            "key_epoch": 1,
            "minimum_readable_key_epoch": 1,
            "catalog_epoch": 1,
            "quota_bytes": 1048576,
            "staging_quota_bytes": 1048576,
            "endurance_bytes_per_five_minutes": 1048576,
            "identity": {
                "model_sha256": d,
                "adapter_sha256": "22".repeat(32),
                "tokenizer_sha256": "33".repeat(32),
                "chat_template_sha256": "44".repeat(32),
                "context_policy_sha256": "55".repeat(32),
                "model_revision": "development",
                "tokenizer_revision": "embedded",
                "weight_precision": "q4_k_xl"
            }
        });
        let path = temp.path().join("durable.json");
        std::fs::write(&path, serde_json::to_vec(&body).unwrap()).unwrap();
        let loaded = DurableConfigV1::load(&path).unwrap();
        assert_eq!(loaded.root, root);
        assert_eq!(loaded.model_sha256().unwrap(), [0x11; 32]);
    }

    #[test]
    fn nvfp4_is_a_distinct_admitted_cache_identity() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("store");
        std::fs::create_dir_all(root.join("keys")).unwrap();
        create_store_key_random(&root.join("keys/root.key"), &root).unwrap();
        let d = "11".repeat(32);
        let body = serde_json::json!({
            "schema_version": 1,
            "root": "store",
            "key_file": "store/keys/root.key",
            "operator_tenant_id": "muser-test",
            "key_epoch": 1,
            "minimum_readable_key_epoch": 1,
            "catalog_epoch": 1,
            "quota_bytes": 1048576,
            "staging_quota_bytes": 1048576,
            "endurance_bytes_per_five_minutes": 1048576,
            "identity": {
                "model_sha256": d,
                "adapter_sha256": "22".repeat(32),
                "tokenizer_sha256": "33".repeat(32),
                "chat_template_sha256": "44".repeat(32),
                "context_policy_sha256": "55".repeat(32),
                "model_revision": "development",
                "tokenizer_revision": "embedded",
                "weight_precision": "nvfp4"
            }
        });
        let path = temp.path().join("durable.json");
        std::fs::write(&path, serde_json::to_vec(&body).unwrap()).unwrap();
        let loaded = DurableConfigV1::load(&path).unwrap();
        assert_eq!(loaded.identity.weight_precision, "nvfp4");
    }
}
