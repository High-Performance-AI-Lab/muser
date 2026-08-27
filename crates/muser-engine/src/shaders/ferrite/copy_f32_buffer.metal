// Exact `copy_f32_buffer` kernel extracted from Ferrite's committed
// drift_measurement.metal without importing diagnostic/capture kernels.
kernel void copy_f32_buffer(
    device const float* src [[ buffer(0) ]],
    device       float* dst [[ buffer(1) ]],
    constant     uint& count [[ buffer(2) ]],
    uint tid [[ thread_position_in_grid ]])
{
    if (tid < count) {
        dst[tid] = src[tid];
    }
}

// Target prefill captures one contiguous buffer per selected layer. DFlash's
// FC consumes token-major rows, so repack the retained GPU capture without a
// multi-gigabyte CPU readback. `source_tokens` can exceed `output_tokens` on
// the final chunk because the newest prompt row stays out of the context K/V.
kernel void pack_dflash_layer_major_f32(
    device const float* src [[ buffer(0) ]],
    device       float* dst [[ buffer(1) ]],
    constant     uint& source_tokens [[ buffer(2) ]],
    constant     uint& source_start [[ buffer(3) ]],
    constant     uint& output_tokens [[ buffer(4) ]],
    constant     uint& layers [[ buffer(5) ]],
    constant     uint& hidden [[ buffer(6) ]],
    uint tid [[ thread_position_in_grid ]])
{
    const uint token_width = layers * hidden;
    const uint count = output_tokens * token_width;
    if (tid < count) {
        const uint token = tid / token_width;
        const uint within = tid - token * token_width;
        const uint layer = within / hidden;
        const uint column = within - layer * hidden;
        const uint source =
            (layer * source_tokens + source_start + token) * hidden + column;
        dst[tid] = src[source];
    }
}
