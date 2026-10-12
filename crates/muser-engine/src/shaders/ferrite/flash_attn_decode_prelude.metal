#include <metal_stdlib>
using namespace metal;

// Function-constant ABI copied from Ferrite's flash_attn_decode_vec.metal at
// a85048a90. Muser keeps only the declarations consumed by the exact F16
// interleaved producer and reducer selected below.
constant uint FC_DK [[function_constant(40)]];
constant bool HAS_FC_DK = is_function_constant_defined(FC_DK);
constant uint FC_F16_INTERLEAVED_NSG [[function_constant(98)]];

struct DecodeParams {
    uint pos;
    uint seq_len;
    uint nwg;
    uint chunk_size;
};
constant bool USE_DECODE_PARAMS_BUF [[function_constant(92)]];
constant bool HAS_USE_DECODE_PARAMS_BUF = is_function_constant_defined(USE_DECODE_PARAMS_BUF);
constant bool BIND_DECODE_PARAMS = HAS_USE_DECODE_PARAMS_BUF && USE_DECODE_PARAMS_BUF;
