//! CPU-only, one-projection-at-a-time bridge from official DFlash GGUF to CoreML.

use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use muser_engine::dflash::{write_gguf_projection_f32, DFlashConfig};
use muser_engine::gguf::GgufFile;
use muser_engine::weights::MuseWeights;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("muser-dflash-extract: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut values = std::env::args().skip(1);
    let mut artifact: Option<PathBuf> = None;
    let mut tensor: Option<String> = None;
    let mut output: Option<PathBuf> = None;
    let mut describe = false;
    let mut tensor_layouts = false;
    let mut raw_tensor = false;
    while let Some(flag) = values.next() {
        match flag.as_str() {
            "--artifact" => artifact = Some(PathBuf::from(next(&mut values, &flag)?)),
            "--tensor" => tensor = Some(next(&mut values, &flag)?),
            "--output" => output = Some(PathBuf::from(next(&mut values, &flag)?)),
            "--describe" => describe = true,
            "--tensor-layouts" => tensor_layouts = true,
            "--raw-tensor" => raw_tensor = true,
            _ => return Err(format!("unknown argument {flag}")),
        }
    }
    let artifact = artifact.ok_or("--artifact is required")?;
    if tensor_layouts {
        if describe || tensor.is_some() || output.is_some() {
            return Err("--tensor-layouts cannot be combined with other operations".into());
        }
        let gguf = GgufFile::parse_path(&artifact).map_err(|error| error.to_string())?;
        let rows = gguf
            .tensors
            .iter()
            .map(|tensor| {
                serde_json::json!({
                    "name": tensor.name,
                    "dtype": format!("{:?}", tensor.dtype),
                    "shape": tensor.shape,
                })
            })
            .collect::<Vec<_>>();
        println!(
            "{}",
            serde_json::to_string(&rows).map_err(|error| error.to_string())?
        );
        return Ok(());
    }
    if describe {
        if tensor.is_some() || output.is_some() {
            return Err("--describe cannot be combined with --tensor/--output".into());
        }
        let gguf = GgufFile::parse_path(&artifact).map_err(|error| error.to_string())?;
        let config = DFlashConfig::from_gguf(&gguf).map_err(|error| error.to_string())?;
        println!(
            "{}",
            serde_json::to_string(&config).map_err(|error| error.to_string())?
        );
        return Ok(());
    }
    let tensor = tensor.ok_or("--tensor is required unless --describe is used")?;
    let output = output.ok_or("--output is required unless --describe is used")?;
    ensure_new_regular_parent(&output)?;
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)
        .map_err(|error| format!("create {}: {error}", output.display()))?;
    let mut guard = IncompleteOutput::new(&output);
    let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file);
    let (rows, columns) = if raw_tensor {
        write_raw_tensor_f32(&artifact, &tensor, &mut writer)?
    } else {
        write_gguf_projection_f32(&artifact, &tensor, &mut writer)
            .map_err(|error| error.to_string())?
    };
    writer.flush().map_err(|error| error.to_string())?;
    writer
        .get_ref()
        .sync_all()
        .map_err(|error| error.to_string())?;
    guard.complete();
    println!(
        "{}",
        serde_json::json!({
            "schema": "muser.dflash-projection-f32.v1",
            "tensor": tensor,
            "rows": rows,
            "columns": columns,
            "bytes": rows * columns * size_of::<f32>(),
            "output": output,
        })
    );
    Ok(())
}

fn write_raw_tensor_f32(
    artifact: &Path,
    tensor_name: &str,
    mut output: impl Write,
) -> Result<(usize, usize), String> {
    let gguf = GgufFile::parse_path(artifact).map_err(|error| error.to_string())?;
    let weights = MuseWeights::open(artifact, &gguf).map_err(|error| error.to_string())?;
    let tensor = weights
        .view(tensor_name)
        .map_err(|error| error.to_string())?;
    let mut row = vec![0.0f32; tensor.n_in];
    let mut encoded = vec![0u8; tensor.n_in * size_of::<f32>()];
    for row_index in 0..tensor.n_out {
        muser_engine::weights::dequant_row(&tensor, row_index, &mut row);
        for (bytes, value) in encoded.chunks_exact_mut(4).zip(&row) {
            bytes.copy_from_slice(&value.to_le_bytes());
        }
        output
            .write_all(&encoded)
            .map_err(|error| error.to_string())?;
    }
    Ok((tensor.n_out, tensor.n_in))
}

fn next(values: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    values
        .next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn ensure_new_regular_parent(path: &Path) -> Result<(), String> {
    if path.exists() || path.is_symlink() {
        return Err(format!("refusing to replace output: {}", path.display()));
    }
    let parent = path.parent().ok_or("output has no parent")?;
    let metadata = parent
        .symlink_metadata()
        .map_err(|error| format!("stat {}: {error}", parent.display()))?;
    if !metadata.file_type().is_dir() || metadata.file_type().is_symlink() {
        return Err(format!(
            "output parent is not a regular directory: {}",
            parent.display()
        ));
    }
    Ok(())
}

struct IncompleteOutput<'a> {
    path: &'a Path,
    complete: bool,
}

impl<'a> IncompleteOutput<'a> {
    fn new(path: &'a Path) -> Self {
        Self {
            path,
            complete: false,
        }
    }

    fn complete(&mut self) {
        self.complete = true;
    }
}

impl Drop for IncompleteOutput<'_> {
    fn drop(&mut self) {
        if !self.complete {
            let _ = std::fs::remove_file(self.path);
        }
    }
}
