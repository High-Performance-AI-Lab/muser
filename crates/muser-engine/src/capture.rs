//! Activation capture in llama.cpp's eval-callback shape.
//!
//! `llama-eval-callback` prints, per graph node, a whole-tensor f32 sum plus a
//! corner sample (first 3 / last 3 along every axis). Recording exactly the
//! same fingerprints here — same traversal order, same corner-selection rule —
//! makes the two runs directly comparable without patching llama.cpp or
//! shipping gigabytes of raw activations around.
//!
//! Reference: `common/debug.cpp`, `common_debug_print_tensor`.

use serde::{Deserialize, Serialize};

/// Number of leading/trailing entries printed per axis (llama.cpp's `n`).
const CORNER_N: usize = 3;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CornerSample {
    /// `[i0, i1, i2, i3]` index into the tensor.
    pub i: [usize; 4],
    pub v: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureEntry {
    pub name: String,
    /// ggml `ne`, padded to 4 dims with 1s. `ne[0]` is the contiguous axis.
    pub ne: [usize; 4],
    /// Whole-tensor sum, f32-accumulated in ggml traversal order.
    pub sum: f32,
    pub corners: Vec<CornerSample>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Capture {
    pub nodes: Vec<CaptureEntry>,
}

/// Indices llama.cpp actually prints along one axis.
fn corner_indices(ne: usize) -> Vec<usize> {
    if ne > 2 * CORNER_N {
        vec![0, 1, 2, ne - 3, ne - 2, ne - 1]
    } else {
        (0..ne).collect()
    }
}

impl Capture {
    /// Record a tensor. `ne` is given innermost-first and padded to 4 dims;
    /// `data` must be laid out with `ne[0]` contiguous.
    pub fn record(&mut self, name: &str, data: &[f32], ne: &[usize]) {
        let mut n = [1usize; 4];
        for (slot, v) in n.iter_mut().zip(ne.iter()) {
            *slot = *v;
        }
        debug_assert_eq!(
            data.len(),
            n[0] * n[1] * n[2] * n[3],
            "capture {name}: ne mismatch"
        );

        // f32 accumulation in ggml traversal order (i0 fastest), matching
        // common_debug_print_tensor so the two sums are comparable.
        let mut sum = 0.0f32;
        for i3 in 0..n[3] {
            for i2 in 0..n[2] {
                for i1 in 0..n[1] {
                    let base = ((i3 * n[2] + i2) * n[1] + i1) * n[0];
                    for i0 in 0..n[0] {
                        sum += data[base + i0];
                    }
                }
            }
        }

        let mut corners = Vec::new();
        for &i3 in &corner_indices(n[3]) {
            for &i2 in &corner_indices(n[2]) {
                for &i1 in &corner_indices(n[1]) {
                    for &i0 in &corner_indices(n[0]) {
                        let idx = ((i3 * n[2] + i2) * n[1] + i1) * n[0] + i0;
                        corners.push(CornerSample {
                            i: [i0, i1, i2, i3],
                            v: data[idx],
                        });
                    }
                }
            }
        }

        self.nodes.push(CaptureEntry {
            name: name.to_string(),
            ne: n,
            sum,
            corners,
        });
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("capture serializes")
    }
}
