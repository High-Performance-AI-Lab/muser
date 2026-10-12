//! In-process pipeline cache for the fixed Muse shader set.

use std::collections::HashMap;

use metal::{ComputePipelineState, ComputePipelineStateRef, FunctionConstantValues};

use super::context::{MetalContext, MetalError};

pub struct PsoCache {
    states: HashMap<&'static str, ComputePipelineState>,
}

impl PsoCache {
    pub fn new(
        context: &MetalContext,
        names: impl IntoIterator<Item = &'static str>,
    ) -> Result<Self, MetalError> {
        let mut states = HashMap::new();
        for name in names {
            // Metal requires functions that declare function constants to be
            // obtained through the constant-values API even when every value
            // intentionally remains undefined and the shader uses its default
            // branch. This mirrors Ferrite's `make_fc_default` constructor.
            let constants =
                matches!(name, "ffn_q4k_gate_up_silu_4r2s").then(FunctionConstantValues::new);
            let function = context
                .library
                .get_function(name, constants)
                .map_err(|message| MetalError::Pipeline {
                    name: name.to_string(),
                    message,
                })?;
            let state = context
                .device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|message| MetalError::Pipeline {
                    name: name.to_string(),
                    message,
                })?;
            states.insert(name, state);
        }
        Ok(Self { states })
    }

    pub fn get(&self, name: &'static str) -> &ComputePipelineStateRef {
        self.states
            .get(name)
            .unwrap_or_else(|| panic!("unregistered Muse Metal pipeline {name}"))
    }
}
