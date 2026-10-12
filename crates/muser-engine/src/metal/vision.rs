//! In-process bridge to the pinned upstream mtmd Muse Metal graph.

use std::ffi::{c_char, c_uchar, c_void, CStr, CString};
use std::path::Path;
use std::sync::Mutex;

use libloading::Library;

type Load = unsafe extern "C" fn(*const c_char) -> *mut c_void;
type Free = unsafe extern "C" fn(*mut c_void);
type Abi = unsafe extern "C" fn() -> *const c_char;
type LastError = unsafe extern "C" fn() -> *const c_char;
type Preprocess = unsafe extern "C" fn(
    *mut c_void,
    *const c_uchar,
    usize,
    u32,
    u32,
    *mut f32,
    usize,
    *mut u32,
    *mut u32,
    *mut usize,
) -> i32;
type Encode = unsafe extern "C" fn(
    *mut c_void,
    *const c_uchar,
    usize,
    u32,
    u32,
    usize,
    *mut f32,
    usize,
    *mut usize,
) -> i32;

pub struct MetalVisionBridge {
    _library: Library,
    handle: *mut c_void,
    free: Free,
    preprocess: Preprocess,
    encode: Encode,
    last_error: LastError,
    inference_lock: Mutex<()>,
}

unsafe impl Send for MetalVisionBridge {}
unsafe impl Sync for MetalVisionBridge {}

impl MetalVisionBridge {
    pub fn load(library_path: &Path, mmproj_path: &Path) -> Result<Self, String> {
        // SAFETY: symbols are copied as function pointers and `_library`
        // remains owned until after the native handle is freed.
        unsafe {
            let library = Library::new(library_path)
                .map_err(|error| format!("load {}: {error}", library_path.display()))?;
            let load: Load = *library
                .get(b"muser_mtmd_load\0")
                .map_err(|error| format!("missing muser_mtmd_load: {error}"))?;
            let free: Free = *library
                .get(b"muser_mtmd_free\0")
                .map_err(|error| format!("missing muser_mtmd_free: {error}"))?;
            let abi: Abi = *library
                .get(b"muser_mtmd_abi\0")
                .map_err(|error| format!("missing muser_mtmd_abi: {error}"))?;
            let encode: Encode = *library
                .get(b"muser_mtmd_encode_rgb\0")
                .map_err(|error| format!("missing muser_mtmd_encode_rgb: {error}"))?;
            let preprocess: Preprocess = *library
                .get(b"muser_mtmd_preprocess_rgb\0")
                .map_err(|error| format!("missing muser_mtmd_preprocess_rgb: {error}"))?;
            let last_error: LastError = *library
                .get(b"muser_mtmd_last_error\0")
                .map_err(|error| format!("missing muser_mtmd_last_error: {error}"))?;
            let reported = c_string(abi());
            if reported != "muser-mtmd-muse-vision-v1" {
                return Err(format!("mtmd bridge ABI is {reported:?}"));
            }
            let path = CString::new(
                mmproj_path
                    .to_str()
                    .ok_or_else(|| "mmproj path is not UTF-8".to_string())?,
            )
            .map_err(|_| "mmproj path contains NUL".to_string())?;
            let handle = load(path.as_ptr());
            if handle.is_null() {
                return Err(c_string(last_error()));
            }
            Ok(Self {
                _library: library,
                handle,
                free,
                preprocess,
                encode,
                last_error,
                inference_lock: Mutex::new(()),
            })
        }
    }

    pub fn preprocess_rgb(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
        expected_elements: usize,
    ) -> Result<(usize, usize, Vec<f32>), String> {
        let _guard = self
            .inference_lock
            .lock()
            .map_err(|_| "mtmd vision lock poisoned")?;
        let width = u32::try_from(width).map_err(|_| "image width exceeds u32")?;
        let height = u32::try_from(height).map_err(|_| "image height exceeds u32")?;
        let mut output = vec![0.0f32; expected_elements];
        let mut output_width = 0u32;
        let mut output_height = 0u32;
        let mut output_elements = 0usize;
        // SAFETY: the bridge copies no more than `output.len()` elements and
        // the per-context lock serializes preprocessing with graph execution.
        let status = unsafe {
            (self.preprocess)(
                self.handle,
                rgb.as_ptr(),
                rgb.len(),
                width,
                height,
                output.as_mut_ptr(),
                output.len(),
                &mut output_width,
                &mut output_height,
                &mut output_elements,
            )
        };
        if status != 0 {
            return Err(unsafe { c_string((self.last_error)()) });
        }
        if output_elements != output.len() || output.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "mtmd preprocessing emitted {output_elements} values, expected {}, or nonfinite values",
                output.len()
            ));
        }
        Ok((output_width as usize, output_height as usize, output))
    }

    pub fn encode_rgb(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
        output_tokens: usize,
        embedding_dim: usize,
    ) -> Result<Vec<Vec<f32>>, String> {
        let _guard = self
            .inference_lock
            .lock()
            .map_err(|_| "mtmd vision lock poisoned")?;
        let width = u32::try_from(width).map_err(|_| "image width exceeds u32")?;
        let height = u32::try_from(height).map_err(|_| "image height exceeds u32")?;
        let elements = output_tokens
            .checked_mul(embedding_dim)
            .ok_or("vision output size overflow")?;
        let mut output = vec![0.0f32; elements];
        let mut actual_tokens = 0usize;
        // SAFETY: the native bridge copies exactly within the supplied input
        // and output capacities and the call is serialized for its context.
        let status = unsafe {
            (self.encode)(
                self.handle,
                rgb.as_ptr(),
                rgb.len(),
                width,
                height,
                embedding_dim,
                output.as_mut_ptr(),
                output.len(),
                &mut actual_tokens,
            )
        };
        if status != 0 {
            return Err(unsafe { c_string((self.last_error)()) });
        }
        if actual_tokens != output_tokens || output.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "mtmd emitted {actual_tokens} rows, expected {output_tokens}, or nonfinite values"
            ));
        }
        Ok(output
            .chunks_exact(embedding_dim)
            .map(<[f32]>::to_vec)
            .collect())
    }
}

impl Drop for MetalVisionBridge {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: this handle was returned by the paired load symbol and
            // is freed once before the library field is dropped.
            unsafe { (self.free)(self.handle) };
            self.handle = std::ptr::null_mut();
        }
    }
}

unsafe fn c_string(value: *const c_char) -> String {
    if value.is_null() {
        "mtmd bridge returned a null error string".into()
    } else {
        CStr::from_ptr(value).to_string_lossy().into_owned()
    }
}
