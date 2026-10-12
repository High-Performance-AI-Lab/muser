use std::collections::BTreeMap;

use serde::Deserialize;

use super::{lower_hex, safe_filename, ModelError};

const ARTIFACT_MANIFEST: &str = include_str!("../../../../docs/release-artifacts.json");
const RELEASE_SCHEMA: &str = "muser.release-artifacts.v2";
const RELEASE_REPOSITORY: &str = "meta-models/Muse-Glimmer-30B-GGUF";

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PinnedArtifact {
    pub filename: String,
    pub revision: String,
    pub url: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ArtifactManifest {
    schema: String,
    pub(super) revision: String,
    pub(super) repository: String,
    pub(super) artifacts: BTreeMap<String, PinnedArtifact>,
}

pub(super) fn embedded() -> Result<ArtifactManifest, ModelError> {
    parse(ARTIFACT_MANIFEST)
}

fn parse(input: &str) -> Result<ArtifactManifest, ModelError> {
    let manifest: ArtifactManifest =
        serde_json::from_str(input).map_err(|error| ModelError::Manifest(error.to_string()))?;
    if manifest.schema != RELEASE_SCHEMA
        || !lower_hex(&manifest.revision, 40)
        || manifest.repository != RELEASE_REPOSITORY
        || manifest
            .artifacts
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>()
            != ["dflash", "target", "vision"]
    {
        return Err(ModelError::Manifest(
            "schema or root revision is outside the v0.1 contract".into(),
        ));
    }
    for (name, artifact) in &manifest.artifacts {
        if artifact.revision != manifest.revision
            || artifact.bytes == 0
            || !lower_hex(&artifact.sha256, 64)
            || !safe_filename(&artifact.filename)
            || artifact.url
                != format!(
                    "https://huggingface.co/{}/resolve/{}/{}?download=true",
                    manifest.repository, artifact.revision, artifact.filename
                )
        {
            return Err(ModelError::Manifest(format!(
                "artifact {name:?} has an invalid immutable identity"
            )));
        }
    }
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_parser_rejects_transport_and_malformed_digest_substitution() {
        let source = ARTIFACT_MANIFEST;
        for changed in [
            source.replacen("https://huggingface.co/", "https://mirror.invalid/", 1),
            source.replacen(
                "7e9b74b7c8875e9e265695df9613bf6290f2392e479ce740495a129019c488d8",
                &"g".repeat(64),
                1,
            ),
        ] {
            assert!(parse(&changed).is_err());
        }
    }
}
