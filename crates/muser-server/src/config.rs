//! Optional `muser.toml` launch config for `muser up`.
//!
//! Every field is optional. The config may select the pinned Hugging Face
//! repository by identifier and change model location, bind host, port, and
//! TLS/auth file locations; it may not change the pinned model identity or
//! download URL.
//!
//! This is `muser up`'s own launch configuration — model *architecture*
//! config (`MuseConfig` in `muser-engine`) is a completely different thing.

use std::path::{Path, PathBuf};

use serde::Deserialize;

#[derive(Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct FileConfig {
    pub node: Option<String>,
    pub gguf_path: Option<PathBuf>,
    pub hf_repo: Option<String>,
    pub host: Option<String>,
    pub port: Option<u16>,
    pub tls_cert: Option<PathBuf>,
    pub tls_key: Option<PathBuf>,
    pub api_key_file: Option<PathBuf>,
}

impl FileConfig {
    /// Load from an explicit `--config` path, or fall back to `./muser.toml`
    /// if present. A missing *implicit* default is not an error (most users
    /// never create the file); a missing or unparsable *explicit* path is.
    pub fn load(explicit: Option<&Path>) -> Result<Self, ConfigError> {
        match explicit {
            Some(path) => Self::read(path),
            None => {
                let default_path = Path::new("muser.toml");
                if default_path.exists() {
                    Self::read(default_path)
                } else {
                    Ok(Self::default())
                }
            }
        }
    }

    fn read(path: &Path) -> Result<Self, ConfigError> {
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError {
            path: path.to_path_buf(),
            kind: ConfigErrorKind::Io(source),
        })?;
        toml::from_str(&text).map_err(|source| ConfigError {
            path: path.to_path_buf(),
            kind: ConfigErrorKind::Parse(source),
        })
    }
}

#[derive(Debug, thiserror::Error)]
#[error("config file {path}: {kind}")]
pub struct ConfigError {
    path: PathBuf,
    #[source]
    kind: ConfigErrorKind,
}

#[derive(Debug, thiserror::Error)]
enum ConfigErrorKind {
    #[error("{0}")]
    Io(std::io::Error),
    #[error("{0}")]
    Parse(toml::de::Error),
}
