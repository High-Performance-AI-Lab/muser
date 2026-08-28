use std::path::{Path, PathBuf};
use std::time::Duration;

use futures_util::StreamExt;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use super::{hex, io_err, ModelError, PinnedArtifact};

pub(super) fn download_file(
    artifact: &PinnedArtifact,
    target: &Path,
    progress: impl Fn(u64, u64),
) -> Result<u64, ModelError> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|source| io_err(target, source))?;
    runtime.block_on(download_file_async(artifact, target, progress))
}

pub(super) fn download_parts(
    parts: &[PinnedArtifact],
    target: &Path,
    expected_bytes: u64,
    expected_sha256: &str,
    progress: impl Fn(u64, u64),
) -> Result<u64, ModelError> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|source| io_err(target, source))?;
    runtime.block_on(download_parts_async(
        parts,
        target,
        expected_bytes,
        expected_sha256,
        progress,
    ))
}

async fn download_parts_async(
    parts: &[PinnedArtifact],
    target: &Path,
    expected_bytes: u64,
    expected_sha256: &str,
    progress: impl Fn(u64, u64),
) -> Result<u64, ModelError> {
    prepare_parent(target).await?;
    let mut completed = 0_u64;
    progress(0, expected_bytes);
    for (index, part) in parts.iter().enumerate() {
        let cached = cached_part_path(target, index);
        if !cached_part_matches(&cached, part).await? {
            remove_regular_generated_file(&cached).await?;
            download_file_async(part, &cached, |done, _| {
                progress(completed.saturating_add(done), expected_bytes);
            })
            .await?;
        }
        completed = completed.saturating_add(part.bytes);
        progress(completed, expected_bytes);
    }

    let partial = partial_path(target);
    remove_regular_generated_file(&partial).await?;
    let mut output = tokio::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&partial)
        .await
        .map_err(|source| io_err(&partial, source))?;
    let mut hasher = Sha256::new();
    let mut assembled = 0_u64;
    let result = async {
        let mut buffer = vec![0_u8; 1024 * 1024];
        for index in 0..parts.len() {
            let cached = cached_part_path(target, index);
            let mut input = tokio::fs::File::open(&cached)
                .await
                .map_err(|source| io_err(&cached, source))?;
            loop {
                let count = input
                    .read(&mut buffer)
                    .await
                    .map_err(|source| io_err(&cached, source))?;
                if count == 0 {
                    break;
                }
                output
                    .write_all(&buffer[..count])
                    .await
                    .map_err(|source| io_err(&partial, source))?;
                hasher.update(&buffer[..count]);
                assembled = assembled.saturating_add(count as u64);
            }
        }
        let combined = PinnedArtifact {
            filename: target
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
            revision: "multipart".into(),
            url: parts[0].url.clone(),
            bytes: expected_bytes,
            sha256: expected_sha256.into(),
        };
        finish_download(&combined, target, &partial, output, hasher, assembled).await
    }
    .await;
    if result.is_err() {
        let _ = tokio::fs::remove_file(&partial).await;
        return result;
    }
    for index in 0..parts.len() {
        let _ = tokio::fs::remove_file(cached_part_path(target, index)).await;
    }
    result
}

async fn cached_part_matches(path: &Path, artifact: &PinnedArtifact) -> Result<bool, ModelError> {
    let metadata = match tokio::fs::symlink_metadata(path).await {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(source) => return Err(io_err(path, source)),
    };
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Err(ModelError::Manifest(format!(
            "multipart cache path is not a regular file: {}",
            path.display()
        )));
    }
    if metadata.len() != artifact.bytes {
        return Ok(false);
    }
    let mut input = tokio::fs::File::open(path)
        .await
        .map_err(|source| io_err(path, source))?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = input
            .read(&mut buffer)
            .await
            .map_err(|source| io_err(path, source))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(hex(&digest.finalize()) == artifact.sha256)
}

async fn remove_regular_generated_file(path: &Path) -> Result<(), ModelError> {
    match tokio::fs::symlink_metadata(path).await {
        Ok(metadata) if metadata.file_type().is_file() && !metadata.file_type().is_symlink() => {
            tokio::fs::remove_file(path)
                .await
                .map_err(|source| io_err(path, source))
        }
        Ok(_) => Err(ModelError::Manifest(format!(
            "refusing to replace non-regular generated path: {}",
            path.display()
        ))),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(io_err(path, source)),
    }
}

fn cached_part_path(target: &Path, index: usize) -> PathBuf {
    let mut name = target.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".muser-part-{index:02}"));
    target.with_file_name(name)
}

async fn download_file_async(
    artifact: &PinnedArtifact,
    target: &Path,
    progress: impl Fn(u64, u64),
) -> Result<u64, ModelError> {
    // The validated release registry emits only immutable HTTPS URLs. The
    // file transport is private to this module so local fixtures exercise
    // the same size/hash/atomic-cache path without contacting Hugging Face.
    if let Some(path) = file_url_path(&artifact.url)? {
        return download_local_fixture(artifact, target, &path, progress).await;
    }
    if !artifact.url.starts_with("https://") && !artifact.url.starts_with("http://") {
        return Err(ModelError::TransportUrl(artifact.url.clone()));
    }

    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(15))
        .read_timeout(Duration::from_secs(30))
        .build()
        .map_err(|source| ModelError::Http {
            url: artifact.url.clone(),
            source,
        })?;
    let response = client
        .get(&artifact.url)
        .send()
        .await
        .map_err(|source| ModelError::Http {
            url: artifact.url.clone(),
            source,
        })?;
    if !response.status().is_success() {
        return Err(ModelError::Gated {
            url: artifact.url.clone(),
            status: response.status().as_u16(),
        });
    }
    if let Some(actual) = response.content_length() {
        if actual != artifact.bytes {
            return Err(ModelError::ArtifactSizeMismatch {
                path: target.to_path_buf(),
                expected: artifact.bytes,
                actual,
            });
        }
    }

    prepare_parent(target).await?;
    let partial = partial_path(target);
    let result = stream_chunks(
        artifact,
        target,
        &partial,
        response.bytes_stream(),
        progress,
    )
    .await;
    if result.is_err() {
        let _ = tokio::fs::remove_file(&partial).await;
    }
    result
}

async fn stream_chunks<S, B>(
    artifact: &PinnedArtifact,
    target: &Path,
    partial: &Path,
    mut stream: S,
    progress: impl Fn(u64, u64),
) -> Result<u64, ModelError>
where
    S: futures_util::Stream<Item = Result<B, reqwest::Error>> + Unpin,
    B: AsRef<[u8]>,
{
    remove_partial(partial).await?;
    let mut file = tokio::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(partial)
        .await
        .map_err(|source| io_err(partial, source))?;
    let mut hasher = Sha256::new();
    let mut downloaded = 0_u64;
    progress(0, artifact.bytes);
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|source| ModelError::Http {
            url: artifact.url.clone(),
            source,
        })?;
        let chunk = chunk.as_ref();
        downloaded = downloaded.saturating_add(chunk.len() as u64);
        if downloaded > artifact.bytes {
            return Err(ModelError::ArtifactSizeMismatch {
                path: target.to_path_buf(),
                expected: artifact.bytes,
                actual: downloaded,
            });
        }
        file.write_all(chunk)
            .await
            .map_err(|source| io_err(partial, source))?;
        hasher.update(chunk);
        progress(downloaded, artifact.bytes);
    }
    finish_download(artifact, target, partial, file, hasher, downloaded).await
}

async fn download_local_fixture(
    artifact: &PinnedArtifact,
    target: &Path,
    source: &Path,
    progress: impl Fn(u64, u64),
) -> Result<u64, ModelError> {
    let actual = tokio::fs::metadata(source)
        .await
        .map_err(|source_error| io_err(source, source_error))?
        .len();
    if actual != artifact.bytes {
        return Err(ModelError::ArtifactSizeMismatch {
            path: target.to_path_buf(),
            expected: artifact.bytes,
            actual,
        });
    }
    prepare_parent(target).await?;
    let partial = partial_path(target);
    remove_partial(&partial).await?;
    let mut input = tokio::fs::File::open(source)
        .await
        .map_err(|source_error| io_err(source, source_error))?;
    let mut output = tokio::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&partial)
        .await
        .map_err(|source_error| io_err(&partial, source_error))?;
    let mut hasher = Sha256::new();
    let mut downloaded = 0_u64;
    let mut buffer = vec![0_u8; 1024 * 1024];
    progress(0, artifact.bytes);
    let result = async {
        loop {
            let count = input
                .read(&mut buffer)
                .await
                .map_err(|source_error| io_err(source, source_error))?;
            if count == 0 {
                break;
            }
            downloaded += count as u64;
            output
                .write_all(&buffer[..count])
                .await
                .map_err(|source_error| io_err(&partial, source_error))?;
            hasher.update(&buffer[..count]);
            progress(downloaded, artifact.bytes);
        }
        finish_download(artifact, target, &partial, output, hasher, downloaded).await
    }
    .await;
    if result.is_err() {
        let _ = tokio::fs::remove_file(&partial).await;
    }
    result
}

async fn finish_download(
    artifact: &PinnedArtifact,
    target: &Path,
    partial: &Path,
    mut file: tokio::fs::File,
    hasher: Sha256,
    downloaded: u64,
) -> Result<u64, ModelError> {
    file.flush()
        .await
        .map_err(|source| io_err(partial, source))?;
    file.sync_all()
        .await
        .map_err(|source| io_err(partial, source))?;
    drop(file);
    if downloaded != artifact.bytes {
        return Err(ModelError::ArtifactSizeMismatch {
            path: target.to_path_buf(),
            expected: artifact.bytes,
            actual: downloaded,
        });
    }
    let actual = hex(&hasher.finalize());
    if actual != artifact.sha256 {
        return Err(ModelError::ChecksumMismatch {
            path: target.to_path_buf(),
            expected: artifact.sha256.clone(),
            actual,
        });
    }
    tokio::fs::rename(partial, target)
        .await
        .map_err(|source| io_err(target, source))?;
    Ok(downloaded)
}

async fn prepare_parent(target: &Path) -> Result<(), ModelError> {
    if let Some(parent) = target.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|source| io_err(parent, source))?;
    }
    Ok(())
}

async fn remove_partial(path: &Path) -> Result<(), ModelError> {
    match tokio::fs::remove_file(path).await {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(io_err(path, source)),
    }
}

fn partial_path(target: &Path) -> PathBuf {
    let mut name = target.file_name().unwrap_or_default().to_os_string();
    name.push(".part");
    target.with_file_name(name)
}

fn file_url_path(url: &str) -> Result<Option<PathBuf>, ModelError> {
    let Some(value) = url.strip_prefix("file://") else {
        return Ok(None);
    };
    if !value.starts_with('/') || value.contains(['?', '#', '%']) {
        return Err(ModelError::TransportUrl(url.to_owned()));
    }
    Ok(Some(PathBuf::from(value)))
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::Mutex;

    use super::*;

    fn fixture_dir(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "muser-model-download-{label}-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ))
    }

    fn digest(bytes: &[u8]) -> String {
        hex(&Sha256::digest(bytes))
    }

    #[test]
    fn file_fixture_streams_progress_and_enters_cache_only_after_digest_match() {
        let root = fixture_dir("pass");
        std::fs::create_dir_all(&root).unwrap();
        let source = root.join("fixture.gguf");
        let target = root.join("models").join("cached.gguf");
        let bytes = b"pinned local fixture\n".repeat(131_072);
        std::fs::write(&source, &bytes).unwrap();
        let artifact = PinnedArtifact {
            filename: "cached.gguf".into(),
            revision: "a".repeat(40),
            url: format!("file://{}", source.display()),
            bytes: bytes.len() as u64,
            sha256: digest(&bytes),
        };
        let seen = Mutex::new(Vec::new());
        let written = download_file(&artifact, &target, |done, total| {
            seen.lock().unwrap().push((done, total));
        })
        .unwrap();
        assert_eq!(written, bytes.len() as u64);
        assert_eq!(std::fs::read(&target).unwrap(), bytes);
        assert_eq!(seen.lock().unwrap().last(), Some(&(written, written)));
        assert!(!partial_path(&target).exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn file_fixture_digest_mismatch_refuses_and_removes_partial_cache_entry() {
        let root = fixture_dir("mismatch");
        std::fs::create_dir_all(&root).unwrap();
        let source = root.join("fixture.gguf");
        let target = root.join("models").join("cached.gguf");
        let bytes = b"wrong identity";
        std::fs::write(&source, bytes).unwrap();
        let artifact = PinnedArtifact {
            filename: "cached.gguf".into(),
            revision: "a".repeat(40),
            url: format!("file://{}", source.display()),
            bytes: bytes.len() as u64,
            sha256: "0".repeat(64),
        };
        assert!(matches!(
            download_file(&artifact, &target, |_, _| {}),
            Err(ModelError::ChecksumMismatch { .. })
        ));
        assert!(!target.exists());
        assert!(!partial_path(&target).exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn reqwest_streams_from_a_loopback_fixture_server_without_hf_access() {
        let root = fixture_dir("http");
        let target = root.join("models").join("cached.gguf");
        let bytes = b"loopback reqwest fixture\n".repeat(65_536);
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let served = bytes.clone();
        let server = std::thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            let mut request = [0_u8; 4096];
            let count = socket.read(&mut request).unwrap();
            assert!(String::from_utf8_lossy(&request[..count]).starts_with("GET /fixture.gguf "));
            write!(
                socket,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                served.len()
            )
            .unwrap();
            for chunk in served.chunks(16 * 1024) {
                socket.write_all(chunk).unwrap();
            }
        });
        let artifact = PinnedArtifact {
            filename: "cached.gguf".into(),
            revision: "a".repeat(40),
            url: format!("http://{address}/fixture.gguf"),
            bytes: bytes.len() as u64,
            sha256: digest(&bytes),
        };
        let seen = Mutex::new(Vec::new());
        assert_eq!(
            download_file(&artifact, &target, |done, _| {
                seen.lock().unwrap().push(done);
            })
            .unwrap(),
            bytes.len() as u64
        );
        server.join().unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), bytes);
        assert_eq!(seen.lock().unwrap().last(), Some(&(bytes.len() as u64)));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn multipart_download_reuses_verified_chunks_and_publishes_atomically() {
        let root = fixture_dir("multipart");
        let sources = root.join("sources");
        let target = root.join("models").join("native.gguf");
        std::fs::create_dir_all(&sources).unwrap();
        let payloads = [
            b"first pinned chunk\n".repeat(4096),
            b"second pinned chunk\n".repeat(2048),
            b"tail\n".repeat(1024),
        ];
        let mut parts = Vec::new();
        let mut combined = Vec::new();
        for (index, payload) in payloads.iter().enumerate() {
            let source = sources.join(format!("part-{index:02}"));
            std::fs::write(&source, payload).unwrap();
            combined.extend_from_slice(payload);
            parts.push(PinnedArtifact {
                filename: format!("native.gguf.part-{index:02}"),
                revision: "a".repeat(40),
                url: format!("file://{}", source.display()),
                bytes: payload.len() as u64,
                sha256: digest(payload),
            });
        }

        // Simulate an interrupted earlier run: chunk zero is already in the
        // generated cache and its original source is no longer available.
        std::fs::create_dir_all(target.parent().unwrap()).unwrap();
        std::fs::write(cached_part_path(&target, 0), &payloads[0]).unwrap();
        std::fs::remove_file(sources.join("part-00")).unwrap();

        let seen = Mutex::new(Vec::new());
        assert_eq!(
            download_parts(
                &parts,
                &target,
                combined.len() as u64,
                &digest(&combined),
                |done, total| seen.lock().unwrap().push((done, total)),
            )
            .unwrap(),
            combined.len() as u64
        );
        assert_eq!(std::fs::read(&target).unwrap(), combined);
        assert!(!partial_path(&target).exists());
        for index in 0..parts.len() {
            assert!(!cached_part_path(&target, index).exists());
        }
        assert_eq!(
            seen.lock().unwrap().last(),
            Some(&(combined.len() as u64, combined.len() as u64))
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}
