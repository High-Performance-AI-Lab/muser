//! Fail-closed GGUF resolution for the one model identity in the v0.1
//! contract. Location and the requested Hugging Face repository identifier
//! are configurable; revision, immutable URL, byte size, and SHA-256 come
//! only from `docs/release-artifacts.json`.

use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::time::Duration;

use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use sha2::{Digest, Sha256};

mod download;
mod manifest;
mod registry;

pub use manifest::PinnedArtifact;

pub const TARGET_ARTIFACT: &str = "target";
pub const LLAMA_CPP_COMMIT: &str = "89e0aa6fd362617d9073e0dafc18e41241521572";
pub const GGML_METALLIB_BYTES: u64 = 7_314_848;
pub const GGML_METALLIB_SHA256: &str =
    "c018ab9c01c18bf79a83272d25973a981138bdeec9ac677e63afef07b21ac033";
pub const GGML_METALLIB_RECEIPT_BYTES: u64 = 1_041;
pub const GGML_METALLIB_RECEIPT_SHA256: &str =
    "29ae842f2f778260fc63b6c51769312264195811d71e8d50c2e93c3d4003deb0";
const RUNTIME_ASSET_BASE: &str =
    "https://github.com/High-Performance-AI-Lab/muser/releases/download/nvfp4-consumer-d5109a1-v1";

pub struct ResolvedModel {
    pub path: PathBuf,
    pub size_bytes: u64,
    pub source: ModelSource,
}

pub enum ModelSource {
    /// Already on disk — nothing downloaded this run.
    Local,
    /// Freshly downloaded this run, from this URL.
    Downloaded { url: String },
}

pub struct ResolveRequest<'a> {
    pub repository: &'a str,
    pub target_path: &'a Path,
}

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("embedded release artifact manifest is invalid: {0}")]
    Manifest(String),

    #[error("requesting {url}: {source}")]
    Http {
        url: String,
        #[source]
        source: reqwest::Error,
    },

    #[error("{url} responded with HTTP {status}")]
    Gated { url: String, status: u16 },

    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error(
        "downloaded {actual} bytes from {url}, server advertised {expected} — file is truncated or the connection dropped"
    )]
    SizeMismatch {
        url: String,
        expected: u64,
        actual: u64,
    },

    #[error("sha256 mismatch for {path}: expected {expected}, got {actual}")]
    ChecksumMismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },

    #[error("size mismatch for {path}: expected {expected} bytes, got {actual}")]
    ArtifactSizeMismatch {
        path: PathBuf,
        expected: u64,
        actual: u64,
    },

    #[error("Hugging Face repository {requested:?} is not pinned; expected {expected:?}")]
    UnknownRepository { requested: String, expected: String },

    #[error("unsupported model transport URL: {0}")]
    TransportUrl(String),
}

/// The user-facing, "print this and it's obvious what to do next" version
/// of a `ModelError`. Kept separate from `Display` (used for logs / the
/// short one-liner) because the two failure modes that matter most here —
/// no URL configured, and a gated/unavailable download — deserve more than
/// one line.
pub fn friendly_error_message(err: &ModelError) -> String {
    match err {
        ModelError::Gated { url, status } => format!(
            "{url}\n\x20 responded with HTTP {status}.\n\n\
             \x20 That usually means the model is gated (needs a license/token) or the\n\
             \x20 URL is stale — muser will not guess a replacement for you.\n\n\
             \x20 Fix: obtain the GGUF through whatever channel you're licensed to use,\n\
             \x20 then either place it at the default path or point --gguf at it\n\
             \x20 directly.",
        ),
        other => other.to_string(),
    }
}

pub fn pinned_artifact(name: &str) -> Result<PinnedArtifact, ModelError> {
    let repository = pinned_repository_id()?;
    pinned_artifact_for_repository(&repository, name)
}

pub fn pinned_repository_id() -> Result<String, ModelError> {
    registry::pinned_repository_id()
}

pub fn pinned_artifact_for_repository(
    repository: &str,
    name: &str,
) -> Result<PinnedArtifact, ModelError> {
    registry::resolve(repository, name)
}

/// Download independently pinned chunks and publish their concatenation only
/// after both the per-chunk and whole-file identities verify. Completed chunks
/// remain beside the destination after an interrupted run, so retrying a
/// multi-gigabyte release does not start from byte zero.
pub fn download_pinned_parts(
    parts: &[PinnedArtifact],
    target: &Path,
    expected_bytes: u64,
    expected_sha256: &str,
    progress: impl Fn(u64, u64),
) -> Result<u64, ModelError> {
    if parts.is_empty() || !lower_hex(expected_sha256, 64) {
        return Err(ModelError::Manifest(
            "multipart artifact has no parts or an invalid final SHA-256".into(),
        ));
    }
    let total = parts.iter().try_fold(0_u64, |sum, part| {
        sum.checked_add(part.bytes)
            .ok_or_else(|| ModelError::Manifest("multipart byte count overflow".into()))
    })?;
    if total != expected_bytes {
        return Err(ModelError::Manifest(format!(
            "multipart artifact totals {total} bytes, expected {expected_bytes}"
        )));
    }
    download::download_parts(parts, target, expected_bytes, expected_sha256, progress)
}

pub fn default_model_path() -> Result<PathBuf, ModelError> {
    default_model_path_from(std::env::var_os("MUSER_HOME"), std::env::var_os("HOME"))
}

pub fn default_metallib_path() -> Result<PathBuf, ModelError> {
    Ok(
        muser_root_from(std::env::var_os("MUSER_HOME"), std::env::var_os("HOME"))?
            .join("runtime")
            .join("llama-metal-89e0aa6fd362")
            .join("llama.metallib"),
    )
}

/// Resolve the source-pinned llama.cpp Metal library used by the few GGML
/// kernels in the Mac decoder. An explicit environment path remains valid;
/// otherwise a clean checkout downloads the 7 MB release artifact and its
/// source receipt, with both identities pinned in the binary.
pub fn ensure_metallib(explicit: Option<&Path>) -> Result<PathBuf, ModelError> {
    if let Some(path) = explicit {
        let path = path.canonicalize().map_err(|source| io_err(path, source))?;
        let receipt = path.with_file_name("source-receipt.json");
        validate_metallib(&path, &receipt)?;
        return Ok(path);
    }

    let path = default_metallib_path()?;
    let receipt = path.with_file_name("source-receipt.json");
    let release_assets = [
        (
            &path,
            PinnedArtifact {
                filename: "llama-metal-89e0aa6fd362.metallib".into(),
                revision: LLAMA_CPP_COMMIT.into(),
                url: format!("{RUNTIME_ASSET_BASE}/llama-metal-89e0aa6fd362.metallib"),
                bytes: GGML_METALLIB_BYTES,
                sha256: GGML_METALLIB_SHA256.into(),
            },
        ),
        (
            &receipt,
            PinnedArtifact {
                filename: "llama-metal-89e0aa6fd362-source-receipt.json".into(),
                revision: LLAMA_CPP_COMMIT.into(),
                url: format!("{RUNTIME_ASSET_BASE}/llama-metal-89e0aa6fd362-source-receipt.json"),
                bytes: GGML_METALLIB_RECEIPT_BYTES,
                sha256: GGML_METALLIB_RECEIPT_SHA256.into(),
            },
        ),
    ];
    for (target, artifact) in release_assets {
        if !target.exists() {
            download_pinned_parts(
                std::slice::from_ref(&artifact),
                target,
                artifact.bytes,
                &artifact.sha256,
                |_, _| {},
            )?;
        }
    }
    validate_metallib(&path, &receipt)?;
    Ok(path)
}

/// Bind the verified metallib into the process before any Metal device is
/// created. This is a no-op for CPU-only/non-macOS builds.
pub fn activate_metallib() -> Result<Option<PathBuf>, ModelError> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        let path = match std::env::var_os("MUSER_GGML_METALLIB") {
            Some(value) => {
                let path = PathBuf::from(value)
                    .canonicalize()
                    .map_err(|source| io_err(Path::new("MUSER_GGML_METALLIB"), source))?;
                let receipt = std::env::var_os("MUSER_GGML_METALLIB_RECEIPT")
                    .map(PathBuf::from)
                    .unwrap_or_else(|| path.with_file_name("source-receipt.json"));
                validate_metallib(&path, &receipt)?;
                path
            }
            None => ensure_metallib(None)?,
        };
        std::env::set_var("MUSER_GGML_METALLIB", &path);
        Ok(Some(path))
    }
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        Ok(None)
    }
}

pub fn validate_metallib(path: &Path, receipt_path: &Path) -> Result<(), ModelError> {
    for candidate in [path, receipt_path] {
        let metadata =
            std::fs::symlink_metadata(candidate).map_err(|source| io_err(candidate, source))?;
        if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
            return Err(ModelError::Manifest(format!(
                "Metal runtime artifact is not a regular file: {}",
                candidate.display()
            )));
        }
    }
    let receipt_bytes =
        std::fs::read(receipt_path).map_err(|source| io_err(receipt_path, source))?;
    let receipt_digest = hex(&Sha256::digest(&receipt_bytes));
    if receipt_bytes.len() as u64 != GGML_METALLIB_RECEIPT_BYTES
        || receipt_digest != GGML_METALLIB_RECEIPT_SHA256
    {
        return Err(ModelError::ChecksumMismatch {
            path: receipt_path.to_path_buf(),
            expected: GGML_METALLIB_RECEIPT_SHA256.into(),
            actual: receipt_digest,
        });
    }
    let receipt: serde_json::Value = serde_json::from_slice(&receipt_bytes)
        .map_err(|error| ModelError::Manifest(format!("Metal source receipt: {error}")))?;
    let metadata = std::fs::metadata(path).map_err(|source| io_err(path, source))?;
    let digest = sha256_file(path)?;
    if metadata.len() != GGML_METALLIB_BYTES
        || digest != GGML_METALLIB_SHA256
        || receipt.get("schema").and_then(serde_json::Value::as_str)
            != Some("muser.llama_metallib.source_receipt.v1")
        || receipt
            .get("source_commit")
            .and_then(serde_json::Value::as_str)
            != Some(LLAMA_CPP_COMMIT)
        || receipt
            .get("binary_size_bytes")
            .and_then(serde_json::Value::as_u64)
            != Some(GGML_METALLIB_BYTES)
        || receipt
            .get("binary_sha256")
            .and_then(serde_json::Value::as_str)
            != Some(GGML_METALLIB_SHA256)
    {
        return Err(ModelError::ChecksumMismatch {
            path: path.to_path_buf(),
            expected: GGML_METALLIB_SHA256.into(),
            actual: digest,
        });
    }
    Ok(())
}

fn default_model_path_from(
    muser_home: Option<std::ffi::OsString>,
    home: Option<std::ffi::OsString>,
) -> Result<PathBuf, ModelError> {
    Ok(muser_root_from(muser_home, home)?
        .join("models")
        .join(pinned_artifact(TARGET_ARTIFACT)?.filename))
}

fn muser_root_from(
    muser_home: Option<std::ffi::OsString>,
    home: Option<std::ffi::OsString>,
) -> Result<PathBuf, ModelError> {
    if let Some(value) = muser_home {
        if value.is_empty() {
            return Err(ModelError::Manifest("MUSER_HOME is empty".into()));
        }
        Ok(PathBuf::from(value))
    } else {
        let home = home
            .filter(|value| !value.is_empty())
            .ok_or_else(|| ModelError::Manifest("neither MUSER_HOME nor HOME is set".into()))?;
        Ok(PathBuf::from(home).join(".muser"))
    }
}

pub fn validate_pinned_artifact(path: &Path, name: &str) -> Result<PinnedArtifact, ModelError> {
    let artifact = pinned_artifact(name)?;
    let actual = std::fs::metadata(path)
        .map_err(|source| io_err(path, source))?
        .len();
    if actual != artifact.bytes {
        return Err(ModelError::ArtifactSizeMismatch {
            path: path.to_path_buf(),
            expected: artifact.bytes,
            actual,
        });
    }
    verify_checksum(path, &artifact.sha256)?;
    Ok(artifact)
}

/// Validate a lane artifact against an already-authenticated configuration
/// digest. This is intentionally separate from the public release manifest:
/// enrolled remote-prefill lanes may qualify a new producer artifact without
/// changing the release candidate or weakening ordinary local serving.
pub fn validate_configured_artifact(path: &Path, expected: &str) -> Result<String, ModelError> {
    if !lower_hex(expected, 64) {
        return Err(ModelError::Manifest(
            "configured model SHA-256 is not lowercase hex".into(),
        ));
    }
    let bytes = std::fs::metadata(path)
        .map_err(|source| io_err(path, source))?
        .len();
    if bytes == 0 {
        return Err(ModelError::ArtifactSizeMismatch {
            path: path.to_path_buf(),
            expected: 1,
            actual: 0,
        });
    }
    verify_checksum(path, expected)?;
    Ok(expected.into())
}

pub fn resolve(req: ResolveRequest<'_>) -> Result<ResolvedModel, ModelError> {
    let artifact = pinned_artifact_for_repository(req.repository, TARGET_ARTIFACT)?;
    if req.target_path.is_file() {
        validate_pinned_artifact(req.target_path, TARGET_ARTIFACT)?;
        let size_bytes = artifact.bytes;
        println!(
            "  {} local GGUF found — {} ({:.1} GB)",
            style("\u{2713}").green().bold(),
            req.target_path.display(),
            size_bytes as f64 / 1e9
        );
        return Ok(ResolvedModel {
            path: req.target_path.to_path_buf(),
            size_bytes,
            source: ModelSource::Local,
        });
    }

    println!(
        "  no local GGUF at {} — downloading from {}",
        req.target_path.display(),
        artifact.url
    );
    let pb = new_download_bar(Some(artifact.bytes));
    let result = download::download_file(&artifact, req.target_path, |downloaded, total| {
        pb.set_length(total);
        pb.set_position(downloaded);
    });
    pb.finish_and_clear();
    let size_bytes = result?;
    println!(
        "  {} size verified — {} bytes",
        style("\u{2713}").green().bold(),
        size_bytes
    );
    println!(
        "  {} sha256 verified against pinned manifest",
        style("\u{2713}").green().bold()
    );
    println!(
        "  {} saved to {}",
        style("\u{2713}").green().bold(),
        req.target_path.display()
    );
    Ok(ResolvedModel {
        path: req.target_path.to_path_buf(),
        size_bytes,
        source: ModelSource::Downloaded { url: artifact.url },
    })
}

fn verify_checksum(path: &Path, expected: &str) -> Result<(), ModelError> {
    println!("  verifying sha256 against the configured hash...");
    let actual = sha256_file(path)?;
    if !actual.eq_ignore_ascii_case(expected) {
        return Err(ModelError::ChecksumMismatch {
            path: path.to_path_buf(),
            expected: expected.to_string(),
            actual,
        });
    }
    println!("  {} sha256 verified", style("\u{2713}").green().bold());
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, ModelError> {
    let mut file = File::open(path).map_err(|source| io_err(path, source))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf).map_err(|source| io_err(path, source))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

fn new_download_bar(total: Option<u64>) -> ProgressBar {
    let pb = match total {
        Some(total) => {
            let pb = ProgressBar::new(total);
            pb.set_style(
                ProgressStyle::with_template(
                    "  {spinner:.cyan} downloading [{bar:32.cyan/blue}] {bytes}/{total_bytes} · {bytes_per_sec} · ETA {eta}",
                )
                .unwrap()
                .progress_chars("=> "),
            );
            pb
        }
        None => {
            let pb = ProgressBar::new_spinner();
            pb.set_style(
                ProgressStyle::with_template(
                    "  {spinner:.cyan} downloading — {bytes} received · {bytes_per_sec}",
                )
                .unwrap(),
            );
            pb
        }
    };
    pb.enable_steady_tick(Duration::from_millis(100));
    pb
}

pub(super) fn io_err(path: &Path, source: io::Error) -> ModelError {
    ModelError::Io {
        path: path.to_path_buf(),
        source,
    }
}

pub(super) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub(super) fn lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub(super) fn safe_filename(value: &str) -> bool {
    !value.is_empty()
        && std::path::Path::new(value)
            .components()
            .all(|component| matches!(component, std::path::Component::Normal(_)))
        && std::path::Path::new(value).components().count() == 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_manifest_is_closed_and_pins_all_three_artifacts() {
        assert_eq!(
            pinned_repository_id().unwrap(),
            "meta-models/Muse-Glimmer-30B-GGUF"
        );
        let target = pinned_artifact("target").expect("target pin");
        let vision = pinned_artifact("vision").expect("vision pin");
        let dflash = pinned_artifact("dflash").expect("DFlash pin");
        assert_eq!(target.bytes, 16_756_681_056);
        assert_eq!(vision.bytes, 1_400_328_928);
        assert_eq!(dflash.bytes, 1_631_205_312);
        assert_eq!(
            target.sha256,
            "7e9b74b7c8875e9e265695df9613bf6290f2392e479ce740495a129019c488d8"
        );
        assert!(matches!(
            pinned_artifact("unknown"),
            Err(ModelError::Manifest(_))
        ));
    }

    #[test]
    fn repository_selector_cannot_change_the_trust_root() {
        assert!(matches!(
            pinned_artifact_for_repository("someone/compatible-model", TARGET_ARTIFACT),
            Err(ModelError::UnknownRepository { .. })
        ));
    }

    #[test]
    fn default_path_uses_only_muser_home_or_dot_muser() {
        let direct = default_model_path_from(Some("/operator/muser".into()), None).unwrap();
        assert_eq!(
            direct,
            Path::new("/operator/muser/models/muse-glimmer-30B-kquant-17gb.gguf")
        );
        let fallback = default_model_path_from(None, Some("/operator".into())).unwrap();
        assert_eq!(
            fallback,
            Path::new("/operator/.muser/models/muse-glimmer-30B-kquant-17gb.gguf")
        );
        assert!(default_model_path_from(Some("".into()), Some("/operator".into())).is_err());
        assert!(default_model_path_from(None, None).is_err());
    }

    #[test]
    fn artifact_names_and_digests_are_closed_values() {
        for unsafe_name in ["../model.gguf", "models/model.gguf", "/model.gguf", ""] {
            assert!(!safe_filename(unsafe_name), "{unsafe_name}");
        }
        assert!(safe_filename("model.gguf"));
        assert!(lower_hex(&"a".repeat(64), 64));
        assert!(!lower_hex(&"A".repeat(64), 64));
        assert!(!lower_hex(&"g".repeat(64), 64));
    }

    #[test]
    fn configured_location_cannot_change_artifact_identity() {
        let path = std::env::temp_dir().join(format!(
            "muser-wrong-sized-model-{}-{}.gguf",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::write(&path, b"not the pinned model").expect("write test artifact");

        let result = validate_pinned_artifact(&path, TARGET_ARTIFACT);
        let _ = std::fs::remove_file(&path);
        assert!(matches!(
            result,
            Err(ModelError::ArtifactSizeMismatch {
                expected: 16_756_681_056,
                actual: 20,
                ..
            })
        ));
    }

    #[test]
    fn configured_lane_digest_admits_only_the_exact_nonempty_artifact() {
        let path = std::env::temp_dir().join(format!(
            "muser-configured-model-{}-{}.gguf",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::write(&path, b"native-nvfp4-fixture").expect("write test artifact");
        let digest = sha256_file(&path).unwrap();
        assert_eq!(
            validate_configured_artifact(&path, &digest).unwrap(),
            digest
        );
        assert!(matches!(
            validate_configured_artifact(&path, &"0".repeat(64)),
            Err(ModelError::ChecksumMismatch { .. })
        ));
        assert!(matches!(
            validate_configured_artifact(&path, &"A".repeat(64)),
            Err(ModelError::Manifest(_))
        ));
        std::fs::remove_file(path).unwrap();
    }
}
