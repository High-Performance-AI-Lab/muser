//! Minimal standalone Metal device, queue, and runtime-compiled library.

use metal::{CommandQueue, CompileOptions, Device, Library, MTLLanguageVersion};
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

#[derive(Debug, thiserror::Error)]
pub enum MetalError {
    #[error("no Metal device is available")]
    NoDevice,
    #[error("Metal shader compilation failed: {0}")]
    ShaderCompile(String),
    #[error("llama.cpp Metal library {path} failed to load: {message}")]
    GgmlLibrary { path: PathBuf, message: String },
    #[error("Metal pipeline creation failed for {name}: {message}")]
    Pipeline { name: String, message: String },
    #[error("Metal command buffer failed: {0}")]
    Command(String),
    #[error(
        "Metal command exceeded its cooperative {seconds}s deadline (label={label:?}, status={status:?})"
    )]
    Deadline {
        seconds: u64,
        status: metal::MTLCommandBufferStatus,
        label: String,
    },
    #[error("Metal allocation of {0} bytes failed")]
    Allocation(usize),
}

pub struct MetalContext {
    pub device: Device,
    pub queue: CommandQueue,
    pub library: Library,
    /// Strict-f32 copy of the standalone Muse kernels.  The cross-vendor
    /// Q8 projection and integer NVFP4 routes must match CUDA's explicit
    /// scalar boundaries, while the ordinary serving kernels retain fast math.
    pub cross_vendor_library: Library,
    pub ggml_library: Option<Library>,
    pub ggml_library_path: Option<PathBuf>,
}

impl MetalContext {
    pub fn new() -> Result<Self, MetalError> {
        let device = Device::system_default().ok_or(MetalError::NoDevice)?;
        let queue = device.new_command_queue();
        let options = CompileOptions::new();
        // Ferrite's accepted production kernels and llama.cpp both compile
        // with fast math enabled. Exact Muse parity is guarded at the token
        // boundary; disabling this here materially slows attention, FFN, and
        // the norm/tiny-op stack without changing the imported GGML PSOs.
        options.set_fast_math_enabled(true);
        options.set_language_version(MTLLanguageVersion::V3_1);
        // The fixed Muse driver is local, while the operation kernels below
        // are clean extractions from Ferrite at a85048a90. Keeping the exact
        // source files separate makes their provenance and future diffing
        // auditable without bringing over Ferrite's runtime or route VM.
        let source = concat!(
            include_str!("../shaders/muse_reference.metal"),
            "\n",
            include_str!("../shaders/nvfp4.metal"),
            "\n",
            include_str!("../shaders/ferrite/sigmoid_gate.metal"),
            "\n",
            include_str!("../shaders/ferrite/rmsnorm_batch_tail.metal"),
            "\n",
            include_str!("../shaders/ferrite/rope.metal"),
            "\n",
            include_str!("../shaders/ferrite/matmul.metal"),
            "\n",
            include_str!("../shaders/ferrite/_q4k_helpers.metal"),
            "\n",
            include_str!("../shaders/ferrite/batch_f32_support.metal"),
            "\n",
            include_str!("../shaders/ferrite/batch_sgm_q4_aligned.metal"),
            "\n",
            include_str!("../shaders/ferrite/batch_m16_n32.metal"),
            "\n",
            include_str!("../shaders/ferrite/batch_ffn_activation_tail.metal"),
            "\n",
            include_str!("../shaders/ferrite/rms_norm_per_head.metal"),
            "\n",
            include_str!("../shaders/ferrite/ffn_fused.metal"),
            "\n",
            include_str!("../shaders/ferrite/ffn_fused_normed_quant.metal"),
            "\n",
            include_str!("../shaders/ferrite/ffn_fused_q4k_hidden.metal"),
            "\n",
            include_str!("../shaders/ferrite/ffn_fused_tail.metal"),
            "\n",
            include_str!("../shaders/ferrite/flash_attn_decode_prelude.metal"),
            "\n",
            include_str!("../shaders/ferrite/attention_dflash_dual.metal"),
            "\n",
            include_str!("../shaders/ferrite/flash_attn_v2.metal"),
            "\n",
            include_str!("../shaders/ferrite/flash_attn_decode_gqa_fa2.metal"),
            "\n",
            include_str!("../shaders/ferrite/copy_f32_buffer.metal"),
            "\n",
            include_str!("../shaders/ferrite/argmax_f32.metal"),
            "\n",
            include_str!("../shaders/ferrite/flash_attn_decode_vec_contiguous_f16.metal"),
            "\n",
            include_str!("../shaders/ferrite/flash_attn_decode_reduce_v2.metal"),
        );
        let library = device
            .new_library_with_source(source, &options)
            .map_err(MetalError::ShaderCompile)?;
        let cross_vendor_options = CompileOptions::new();
        cross_vendor_options.set_fast_math_enabled(false);
        cross_vendor_options.set_language_version(MTLLanguageVersion::V3_1);
        let cross_vendor_source = concat!(
            include_str!("../shaders/muse_reference.metal"),
            "\n",
            include_str!("../shaders/nvfp4.metal"),
        );
        let cross_vendor_library = device
            .new_library_with_source(cross_vendor_source, &cross_vendor_options)
            .map_err(MetalError::ShaderCompile)?;
        let ggml_library_path = std::env::var_os("MUSER_GGML_METALLIB").map(PathBuf::from);
        let ggml_library = match ggml_library_path.as_ref() {
            Some(path) => Some(device.new_library_with_file(path).map_err(|message| {
                MetalError::GgmlLibrary {
                    path: path.clone(),
                    message,
                }
            })?),
            None => None,
        };
        Ok(Self {
            device,
            queue,
            library,
            cross_vendor_library,
            ggml_library,
            ggml_library_path,
        })
    }

    pub fn ensure_completed(&self, command: &metal::CommandBufferRef) -> Result<(), MetalError> {
        match command.status() {
            metal::MTLCommandBufferStatus::Completed => Ok(()),
            status => Err(MetalError::Command(format!("status={status:?}"))),
        }
    }

    /// Wait without signals or external process control. Callers choose a
    /// cell-sized deadline; expiry returns the `Deadline` error (carrying the
    /// command buffer's observed status and label) and lets the guarded
    /// accelerator process preserve its evidence instead of freezing.
    pub fn wait_for_completion(
        &self,
        command: &metal::CommandBufferRef,
        deadline: Duration,
    ) -> Result<(), MetalError> {
        // Fast path: the buffer is very often already done (or already
        // errored) by the time a caller gets here, so check first and skip
        // the coordination machinery below entirely -- allocation-free.
        match command.status() {
            metal::MTLCommandBufferStatus::Completed => return Ok(()),
            metal::MTLCommandBufferStatus::Error => return self.ensure_completed(command),
            _ => {}
        }
        // `command.wait_until_completed()` blocks this thread unboundedly --
        // a wedged GPU freezes whatever serving thread called us, making the
        // deadline decorative. Metal's own `addCompletedHandler` callback
        // would let us wait without a dedicated thread at all, but building
        // one needs the `block` crate, currently only a transitive
        // dependency (pulled in by `metal`); adding it as a direct
        // dependency means touching Cargo.toml, which is out of scope here
        // (see api_notes). Instead, park the blocking wait on a detached
        // watcher thread and bound *this* thread's wait with a condvar so a
        // hang becomes a logged `Deadline` error, not a frozen box.
        let owned = command.to_owned();
        let signal = Arc::new((Mutex::new(false), Condvar::new()));
        let watcher_signal = Arc::clone(&signal);
        let watcher = std::thread::spawn(move || {
            owned.wait_until_completed();
            let (done, cvar) = &*watcher_signal;
            *done.lock().unwrap() = true;
            cvar.notify_one();
        });

        let (done, cvar) = &*signal;
        let guard = done.lock().unwrap();
        let (_guard, wait_result) = cvar
            .wait_timeout_while(guard, deadline, |&mut done| !done)
            .unwrap();

        if wait_result.timed_out() {
            // The watcher stays running (not joined) so a wedged GPU never
            // ties up this thread; it exits on its own once the command
            // buffer completes, if ever.
            return Err(MetalError::Deadline {
                seconds: deadline.as_secs(),
                status: command.status(),
                label: command.label().to_string(),
            });
        }

        let _ = watcher.join();
        match command.status() {
            metal::MTLCommandBufferStatus::Completed => Ok(()),
            metal::MTLCommandBufferStatus::Error => self.ensure_completed(command),
            _ => Err(MetalError::Command(
                "command buffer did not complete after wait".into(),
            )),
        }
    }
}
