use super::{manifest, ModelError, PinnedArtifact};

pub(super) fn pinned_repository_id() -> Result<String, ModelError> {
    Ok(manifest::embedded()?.repository)
}

pub(super) fn resolve(repository: &str, artifact_name: &str) -> Result<PinnedArtifact, ModelError> {
    let mut manifest = manifest::embedded()?;
    if repository != manifest.repository {
        return Err(ModelError::UnknownRepository {
            requested: repository.to_owned(),
            expected: manifest.repository,
        });
    }
    manifest
        .artifacts
        .remove(artifact_name)
        .ok_or_else(|| ModelError::Manifest(format!("artifact {artifact_name:?} is missing")))
}
