#include <metal_stdlib>
using namespace metal;

// Element-wise sigmoid gating: attn_out[i] *= sigmoid(gate[i])
// dispatch: (ceil(n/1024), 1, 1) × (1024, 1, 1)

kernel void sigmoid_gate_inplace(
    device       float* attn_out [[ buffer(0) ]],
    device const float* gate     [[ buffer(1) ]],
    constant     uint&  n        [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]])
{
    if (gid < n) {
        attn_out[gid] *= 1.0f / (1.0f + exp(-gate[gid]));
    }
}
