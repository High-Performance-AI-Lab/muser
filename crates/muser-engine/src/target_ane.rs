//! Public-CoreML target-decode tail for Muse SWA layers.
//!
//! This is deliberately separate from the five-layer DFlash backend.  Metal
//! owns attention and KV state; two ANE-resident programs own the weight-heavy
//! output/FFN projections for one measured layer partition.

use std::collections::BTreeSet;
use std::fs::File;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::sync::Mutex;

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::coreml::CoreMlModel;

const HIDDEN: usize = 6656;
const ATTENTION: usize = 4096;
const MAX_PACKAGE_BYTES: u64 = 250 * 1024 * 1024;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    version: u32,
    schema: String,
    backend: String,
    compute_units: String,
    weight_dtype: String,
    projection_operator: String,
    model_sha256: String,
    layer: usize,
    kind: String,
    batch: usize,
    split: Vec<usize>,
    metal_ops: Vec<String>,
    ane_ops: Vec<String>,
    toolchain: serde_json::Value,
    packages: Vec<Package>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Package {
    order: usize,
    path: PathBuf,
    bytes: u64,
    sha256: String,
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

struct LoadedPackage {
    spec: Package,
    model: CoreMlModel,
    input: Mutex<Vec<f32>>,
    output: Mutex<Vec<f32>>,
}

pub struct MuseTargetAneTail {
    manifest: Manifest,
    head: LoadedPackage,
    continuation: LoadedPackage,
    inference_lock: Mutex<()>,
}

pub struct MuseTargetTailResult {
    pub ffn_input: Vec<f32>,
    pub ffn_normed: Vec<f32>,
    pub down_projection: Vec<f32>,
}

impl MuseTargetAneTail {
    pub fn load(manifest_path: &Path, expected_model_sha256: &str) -> Result<Self, String> {
        let manifest: Manifest = serde_json::from_slice(
            &std::fs::read(manifest_path)
                .map_err(|error| format!("read {}: {error}", manifest_path.display()))?,
        )
        .map_err(|error| format!("parse {}: {error}", manifest_path.display()))?;
        if manifest.version != 1
            || manifest.schema != "muser.muse-target-coreml-export-plan.v1"
            || manifest.backend != "public_coreml"
            || manifest.compute_units != "CPU_AND_NE"
            || manifest.weight_dtype != "int8"
            || manifest.projection_operator != "conv1x1"
            || manifest.model_sha256 != expected_model_sha256
            || manifest.kind != "swa_rope_2048"
            || manifest.layer >= 52
            || manifest.layer % 4 == 3
            || manifest.batch != 16
            || manifest.split.iter().sum::<usize>() != 19968
            || manifest.packages.len() != 2
        {
            return Err("Muse target ANE manifest contract differs".into());
        }
        let required_metal = BTreeSet::from([
            "qkvg",
            "qk_norm",
            "rope",
            "kv",
            "attention",
            "sigmoid_gate",
            "post_ffn_norm_residual",
        ]);
        if manifest
            .metal_ops
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
            != required_metal
        {
            return Err("Muse target ANE manifest Metal boundary differs".into());
        }
        let root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
        let mut packages = manifest.packages.clone();
        packages.sort_by_key(|package| package.order);
        if packages[0].order != 0
            || packages[0].input_shape != [1, ATTENTION + HIDDEN, 16, 1]
            || packages[0].output_shape != [1, 3 * HIDDEN, 16, 1]
            || packages[1].order != 1
            || packages[1].input_shape != [1, HIDDEN, 16, 1]
            || packages[1].output_shape != [1, HIDDEN, 16, 1]
        {
            return Err("Muse target ANE package geometry differs".into());
        }
        let mut loaded = Vec::new();
        for spec in packages {
            let path = resolve_beneath(root, &spec.path)?;
            if spec.bytes == 0
                || spec.bytes > MAX_PACKAGE_BYTES
                || tree_size(&path)? != spec.bytes
                || tree_sha256(&path)? != spec.sha256
            {
                return Err(format!(
                    "Muse target ANE package {} identity differs",
                    spec.order
                ));
            }
            let input_elements = spec.input_shape.iter().product();
            let output_elements = spec.output_shape.iter().product();
            let model = CoreMlModel::load(
                &path,
                "input",
                "output",
                &spec.input_shape,
                &spec.output_shape,
            )?;
            loaded.push(LoadedPackage {
                spec,
                model,
                input: Mutex::new(vec![0.0; input_elements]),
                output: Mutex::new(vec![0.0; output_elements]),
            });
        }
        let continuation = loaded.pop().expect("validated continuation");
        let head = loaded.pop().expect("validated head");
        Ok(Self {
            manifest,
            head,
            continuation,
            inference_lock: Mutex::new(()),
        })
    }

    pub fn layer(&self) -> usize {
        self.manifest.layer
    }

    pub fn run(&self, attention: &[f32], residual: &[f32]) -> Result<MuseTargetTailResult, String> {
        let _guard = self
            .inference_lock
            .lock()
            .map_err(|_| "Muse target ANE inference lock poisoned")?;
        let batch = self.manifest.batch;
        if attention.len() != batch * ATTENTION || residual.len() != batch * HIDDEN {
            return Err("Muse target ANE input geometry differs".into());
        }
        let head_channels = ATTENTION + HIDDEN;
        let mut head_input = self
            .head
            .input
            .lock()
            .map_err(|_| "head input lock poisoned")?;
        for token in 0..batch {
            for channel in 0..ATTENTION {
                head_input[channel * batch + token] = attention[token * ATTENTION + channel];
            }
            for channel in 0..HIDDEN {
                head_input[(ATTENTION + channel) * batch + token] =
                    residual[token * HIDDEN + channel];
            }
        }
        debug_assert_eq!(head_input.len(), batch * head_channels);
        let mut head_output = self
            .head
            .output
            .lock()
            .map_err(|_| "head output lock poisoned")?;
        self.head
            .model
            .predict_into(&head_input, &self.head.spec.input_shape, &mut head_output)?;
        let mut ffn_input = vec![0.0; batch * HIDDEN];
        let mut ffn_normed = vec![0.0; batch * HIDDEN];
        let mut down_projection = vec![0.0; batch * HIDDEN];
        for token in 0..batch {
            for channel in 0..HIDDEN {
                ffn_input[token * HIDDEN + channel] = head_output[channel * batch + token];
                ffn_normed[token * HIDDEN + channel] =
                    head_output[(HIDDEN + channel) * batch + token];
                down_projection[token * HIDDEN + channel] =
                    head_output[(2 * HIDDEN + channel) * batch + token];
            }
        }
        drop(head_output);
        drop(head_input);
        let mut continuation_input = self
            .continuation
            .input
            .lock()
            .map_err(|_| "continuation input lock poisoned")?;
        for token in 0..batch {
            for channel in 0..HIDDEN {
                continuation_input[channel * batch + token] = ffn_normed[token * HIDDEN + channel];
            }
        }
        let mut continuation_output = self
            .continuation
            .output
            .lock()
            .map_err(|_| "continuation output lock poisoned")?;
        self.continuation.model.predict_into(
            &continuation_input,
            &self.continuation.spec.input_shape,
            &mut continuation_output,
        )?;
        for token in 0..batch {
            for channel in 0..HIDDEN {
                down_projection[token * HIDDEN + channel] +=
                    continuation_output[channel * batch + token];
            }
        }
        Ok(MuseTargetTailResult {
            ffn_input,
            ffn_normed,
            down_projection,
        })
    }
}

fn resolve_beneath(root: &Path, relative: &Path) -> Result<PathBuf, String> {
    if relative.is_absolute()
        || relative.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err("Muse target ANE package path is unsafe".into());
    }
    let root = root
        .canonicalize()
        .map_err(|error| format!("canonicalize {}: {error}", root.display()))?;
    let path = root.join(relative);
    if !path
        .canonicalize()
        .map_err(|error| error.to_string())?
        .starts_with(&root)
    {
        return Err("Muse target ANE package escapes artifact root".into());
    }
    Ok(path)
}

fn tree_files(path: &Path) -> Result<Vec<(PathBuf, PathBuf)>, String> {
    fn walk(root: &Path, path: &Path, output: &mut Vec<(PathBuf, PathBuf)>) -> Result<(), String> {
        let mut entries = std::fs::read_dir(path)
            .map_err(|error| error.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string())?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let child = entry.path();
            let metadata = child
                .symlink_metadata()
                .map_err(|error| error.to_string())?;
            if metadata.file_type().is_symlink() {
                return Err("Muse target ANE package contains a symlink".into());
            }
            if metadata.is_dir() {
                walk(root, &child, output)?;
            } else if metadata.is_file() {
                output.push((child.strip_prefix(root).unwrap().to_path_buf(), child));
            } else {
                return Err("Muse target ANE package contains a special entry".into());
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    walk(path, path, &mut files)?;
    Ok(files)
}

fn tree_size(path: &Path) -> Result<u64, String> {
    tree_files(path)?
        .into_iter()
        .try_fold(0u64, |total, (_, file)| {
            total
                .checked_add(file.metadata().map_err(|error| error.to_string())?.len())
                .ok_or_else(|| "Muse target ANE package size overflow".into())
        })
}

fn tree_sha256(path: &Path) -> Result<String, String> {
    let mut digest = Sha256::new();
    for (relative, file) in tree_files(path)? {
        let name = relative.to_string_lossy();
        let bytes = name.as_bytes();
        let size = file.metadata().map_err(|error| error.to_string())?.len();
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(bytes);
        digest.update(size.to_le_bytes());
        let mut source = File::open(file).map_err(|error| error.to_string())?;
        let mut buffer = vec![0u8; 1024 * 1024];
        loop {
            let count = source
                .read(&mut buffer)
                .map_err(|error| error.to_string())?;
            if count == 0 {
                break;
            }
            digest.update(&buffer[..count]);
        }
    }
    Ok(format!("{:x}", digest.finalize()))
}
