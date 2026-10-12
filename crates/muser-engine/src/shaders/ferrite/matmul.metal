#include <metal_stdlib>

#ifdef FERRITE_HAS_TENSOR_OPS
#include <metal_tensor>
// MetalPerformancePrimitives provides mpp::tensor_ops::matmul2d.
// This header is in the macOS SDK; runtime compilation finds it via
// the default include search paths of [MTLDevice newLibraryWithSource:].
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

// If the MPP header isn't found at runtime, the fallback simdgroup path
// is used (FERRITE_HAS_TENSOR_OPS only defined when Metal 4 is detected).
using namespace metal;
#endif

using namespace metal;

#ifdef FERRITE_HAS_TENSOR_OPS
// MPP cooperative destinations are uninitialized storage. Every tensor-op
// kernel uses multiply_accumulate mode, so leaving even one valid lane
// untouched makes the first K tile accumulate onto indeterminate data.
// Keep the required mask/capacity walk in one fully-unrolled helper.
template <typename CooperativeTensor>
static inline void ferrite_zero_cooperative_tensor(
    thread CooperativeTensor& destination
) {
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < destination.get_capacity(); ++i) {
        // Xcode 27 exposes the cooperative validity mask through
        // is_valid_element(); the MPP header's example calls the same
        // operation get_mask().
        if (destination.is_valid_element(i)) {
            destination[i] = 0.0f;
        }
    }
}
#endif

constant uint FC_Q4K_V4_ROWS [[function_constant(10)]];
constant uint FC_Q4K_V4_COLS [[function_constant(11)]];
constant bool HAS_FC_Q4K_V4_ROWS = is_function_constant_defined(FC_Q4K_V4_ROWS);
constant bool HAS_FC_Q4K_V4_COLS = is_function_constant_defined(FC_Q4K_V4_COLS);

// Thin-GEMM batch size — specialized at PSO creation time.
// When defined, the compiler eliminates unused batch slots and their registers.
constant uint FC_THIN_GEMM_BATCH [[function_constant(12)]];
constant bool HAS_FC_THIN_GEMM_BATCH = is_function_constant_defined(FC_THIN_GEMM_BATCH);

//
// Q4_K block layout (GGUF, 144 bytes, 256 elements):
//   bytes  0- 1: d    (fp16 super-scale for quantized values)
//   bytes  2- 3: dmin (fp16 super-scale for quantized mins)
//   bytes  4-15: scales[12] — 6-bit packed: 8 sub-block scales + 8 sub-block mins
//   bytes 16-143: qs[128] — nibble-packed 4-bit values (2 per byte)
//
// 8 sub-blocks of 32 elements each.  4 outer groups of 64 elements, each
// group using 32 qs bytes.  qs byte g*32+l:
//   low  nibble → element g*64 + l      (sub-block 2g)
//   high nibble → element g*64 + l + 32 (sub-block 2g+1)
//
// get_scale_min_k4(j) for sub-block j:
//   j < 4: sc = scales[j] & 63,  m = scales[j+4] & 63
//   j >= 4: sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
//           m  = (scales[j+4] >> 4 ) | ((scales[j  ] >> 6) << 4)
//
// The q4x4 dequant helper and llama-style down-proj kernel below are adapted
// from ggml-org/llama.cpp's Metal Q4_K path (`ggml-metal.metal`). Credit to the
// llama.cpp developers for the original q4x4 tiling/dequant structure.
//
struct block_q4_K_llama {
    half  d;
    half  dmin;
    uchar scales[12];
    uchar qs[128];
};

static inline void get_scale_min_k4(uint j,
                                     device const uchar* scales,
                                     thread uint& sc,
                                     thread uint& m) {
    if (j < 4u) {
        sc = uint(scales[j      ]) & 0x3Fu;
        m  = uint(scales[j + 4u]) & 0x3Fu;
    } else {
        sc = (uint(scales[j + 4u]) & 0x0Fu) | ((uint(scales[j - 4u]) >> 6u) << 4u);
        m  = (uint(scales[j + 4u]) >> 4u)   | ((uint(scales[j      ]) >> 6u) << 4u);
    }
}

// Vectorized scale/min extractor: takes 3 pre-loaded uint words covering the
// 12-byte Q4_K scale array (bytes 4-15 of a block).
//   sd0 = *uint(blk+4)  → scales[0..3]
//   sd1 = *uint(blk+8)  → scales[4..7]
//   sd2 = *uint(blk+12) → scales[8..11]
// For j < 4:  sc = scales[j]&63,    m = scales[j+4]&63
// For j >= 4: sc = (scales[j+4]&0xF)|((scales[j-4]>>6)<<4)
//             m  = (scales[j+4]>>4) |((scales[j  ]>>6)<<4)
static inline void get_scale_min_k4_fast(uint j,
                                          uint sd0, uint sd1, uint sd2,
                                          thread uint& sc, thread uint& m) {
    if (j < 4u) {
        uint shift = j * 8u;
        sc = (sd0 >> shift) & 0x3Fu;
        m  = (sd1 >> shift) & 0x3Fu;
    } else {
        uint shift = (j - 4u) * 8u;
        uint bz = (sd2 >> shift) & 0xFFu;  // scales[j+4]
        uint bx = (sd0 >> shift) & 0xFFu;  // scales[j-4]
        uint by = (sd1 >> shift) & 0xFFu;  // scales[j]
        sc = (bz & 0x0Fu) | ((bx >> 6u) << 4u);
        m  = (bz >> 4u)   | ((by >> 6u) << 4u);
    }
}

// Precompute all 8 sub-block scale factors for a Q4_K block.
// Hoists d*sc and -(dmin*m) out of the per-element inner loop,
// reducing the dependency chain from 4 ops to 2 (FMA + MUL).
// Also replaces 8 byte-based scale loads with 3 vectorized uint loads.
static inline void decode_all_q4k_scales(
    float d, float dmin,
    uint sd0, uint sd1, uint sd2,
    thread float d_sc[8],
    thread float neg_dm[8])
{
    // Sub-blocks 0..3 (j < 4): sc = scales[j] & 63, m = scales[j+4] & 63
    for (uint j = 0u; j < 4u; j++) {
        uint shift = j * 8u;
        d_sc[j]   = d * float((sd0 >> shift) & 0x3Fu);
        neg_dm[j] = -(dmin * float((sd1 >> shift) & 0x3Fu));
    }
    // Sub-blocks 4..7 (j >= 4): cross-byte extraction
    for (uint j = 0u; j < 4u; j++) {
        uint shift = j * 8u;
        uint bz = (sd2 >> shift) & 0xFFu;
        uint bx = (sd0 >> shift) & 0xFFu;
        uint by = (sd1 >> shift) & 0xFFu;
        d_sc[j + 4u]   = d * float((bz & 0x0Fu) | ((bx >> 6u) << 4u));
        neg_dm[j + 4u] = -(dmin * float((bz >> 4u) | ((by >> 6u) << 4u)));
    }
}

// Half-precision variant: dequant scales and mins into half registers.
// Q4_K quants are in [0,15] and scales are natively fp16, so half is exact.
// Use with per-block half accumulation, promote to float between blocks.
static inline void decode_all_q4k_scales_h(
    half d, half dmin,
    uint sd0, uint sd1, uint sd2,
    thread half d_sc[8],
    thread half neg_dm[8])
{
    for (uint j = 0u; j < 4u; j++) {
        uint shift = j * 8u;
        d_sc[j]   = d * half((sd0 >> shift) & 0x3Fu);
        neg_dm[j] = -(dmin * half((sd1 >> shift) & 0x3Fu));
    }
    for (uint j = 0u; j < 4u; j++) {
        uint shift = j * 8u;
        uint bz = (sd2 >> shift) & 0xFFu;
        uint bx = (sd0 >> shift) & 0xFFu;
        uint by = (sd1 >> shift) & 0xFFu;
        d_sc[j + 4u]   = d * half((bz & 0x0Fu) | ((bx >> 6u) << 4u));
        neg_dm[j + 4u] = -(dmin * half((bz >> 4u) | ((by >> 6u) << 4u)));
    }
}

static inline uchar2 get_scale_min_k4_just2_llama(short j,
                                                   short k,
                                                   device const uchar * q) {
    return j < 4
        ? uchar2{uchar(q[j + 0 + k] & 63), uchar(q[j + 4 + k] & 63)}
        : uchar2{uchar((q[j + 4 + k] & 0xFu) | ((q[j - 4 + k] & 0xC0u) >> 2)),
                 uchar((q[j + 4 + k] >> 4) | ((q[j + 0 + k] & 0xC0u) >> 2))};
}

static inline void dequantize_q4_K_llama(device const block_q4_K_llama * xb,
                                         short il,
                                         thread float4x4 & reg) {
    device const uchar * q = xb->qs;

    short is = (il / 4) * 2;
    q = q + (il / 4) * 32 + 16 * (il & 1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2_llama(is, il / 2, xb->scales);
    const float d   = il < 2 ? float(xb->d) : float(xb->d) / 16.0f;
    const float min = float(xb->dmin);
    const float dl  = d * float(sc[0]);
    const float ml  = min * float(sc[1]);

    const ushort mask = il < 2 ? 0x0Fu : 0xF0u;
    for (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = dl * float(q[i] & mask) - ml;
    }
}
