//! Five-layer DFlash assistant extracted from Ferrite's accepted CPU oracle.
//!
//! The target hook is Muse-specific, but the assistant math and artifact
//! format supports both the development SafeTensors export and the official
//! llama.cpp-compatible k-quant GGUF sidecar.

mod attention;
mod cache;
mod config;
mod forward;
mod hidden;
mod ops;
mod projection;
mod spec;
mod weights;

pub use cache::{DFlashContextKvCache, DFlashContextSnapshot, DFlashKvCache};
pub use config::{
    DFlashConfig, DFlashContextGeometry, DFlashSpecificConfig, DFLASH_CONTEXT_SINK_SIZE,
};
pub use forward::{DFlashDraftOutput, DFlashForward};
pub use hidden::DFlashHiddenCache;
pub use projection::{
    DFlashAttentionProjections, DFlashFusedAttentionInput, DFlashFusedAttentionOutput,
    DFlashProjectionBackend, DFlashStatefulAttentionInput,
};
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) use spec::cycle_trace_enabled;
#[cfg(all(target_os = "macos", feature = "metal"))]
#[doc(hidden)]
pub use spec::PreparedDFlashTargetContext;
#[cfg(all(target_os = "macos", feature = "metal"))]
#[doc(hidden)]
pub use spec::{
    AuthenticatedDFlashTargetDecision, ProvisionalDFlashResolution, ProvisionalDFlashTargetContext,
};
pub use spec::{
    DFlashAssistant, DFlashCycleTrace, DFlashPreparedGreedy, DFlashPreparedSampled, DFlashRunError,
    DFlashSpecStats, PreparedDFlashContext,
};
pub use weights::{write_gguf_projection_f32, DFlashLayerWeights, DFlashWeights};

#[derive(Debug, thiserror::Error)]
pub enum DFlashError {
    #[error("IO error reading {0}: {1}")]
    Io(std::path::PathBuf, std::io::Error),
    #[error("config error: {0}")]
    Config(String),
    #[error("missing file: {0}")]
    MissingFile(std::path::PathBuf),
    #[error("SafeTensors error: {0}")]
    SafeTensors(String),
    #[error("GGUF assistant error: {0}")]
    Gguf(String),
    #[error("missing tensor: {0}")]
    MissingTensor(String),
    #[error("DFlash projection error: {0}")]
    Projection(String),
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl From<crate::metal::context::MetalError> for DFlashError {
    fn from(error: crate::metal::context::MetalError) -> Self {
        Self::Projection(error.to_string())
    }
}
