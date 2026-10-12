// Narrow in-process bridge to the pinned llama.cpp Muse vision graph.
// It exposes only official preprocessing and raw RGB -> projected decoder
// embeddings. Text inference, KV state, sampling, and serving remain Muser's.

#include "clip.h"
#include "clip-impl.h"
#include "mtmd-image.h"

#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace {
thread_local std::string last_error;

struct bridge_context {
    clip_ctx * clip = nullptr;
    std::unique_ptr<mtmd_image_preprocessor_muse_glimmer> preprocessor;

    ~bridge_context() {
        preprocessor.reset();
        if (clip) clip_free(clip);
    }
};

int fail(const std::string & message) {
    last_error = message;
    return 1;
}

bool checked_rgb_size(uint32_t width, uint32_t height, size_t rgb_bytes) {
    if (width == 0 || height == 0 || width > SIZE_MAX / height) return false;
    const size_t pixels = (size_t) width * height;
    return pixels <= SIZE_MAX / 3 && rgb_bytes == pixels * 3;
}

clip_image_f32 preprocess(
    bridge_context * handle,
    const unsigned char * rgb,
    size_t rgb_bytes,
    uint32_t width,
    uint32_t height) {
    if (!handle || !handle->clip || !handle->preprocessor || !rgb ||
        !checked_rgb_size(width, height, rgb_bytes)) {
        throw std::runtime_error("invalid RGB input");
    }
    clip_image_u8 image;
    image.set_size({(int) width, (int) height}, false);
    image.cpy_buf(std::vector<uint8_t>(rgb, rgb + rgb_bytes));
    auto output = handle->preprocessor->preprocess(image);
    if (output.entries.size() != 1 || output.entries.front().is_placeholder()) {
        throw std::runtime_error("Muse preprocessing did not emit exactly one image");
    }
    return std::move(output.entries.front());
}
} // namespace

extern "C" {

const char * muser_mtmd_last_error() {
    return last_error.c_str();
}

const char * muser_mtmd_abi() {
    return "muser-mtmd-muse-vision-v1";
}

void * muser_mtmd_load(const char * mmproj_path) {
    last_error.clear();
    if (!mmproj_path || !*mmproj_path) {
        fail("mmproj path is empty");
        return nullptr;
    }
    try {
        clip_context_params params {
            /* use_gpu           */ true,
            /* flash_attn_type   */ CLIP_FLASH_ATTN_TYPE_ENABLED,
            /* image_min_tokens  */ 0,
            /* image_max_tokens  */ 4096,
            /* warmup            */ true,
            /* cb_eval           */ nullptr,
            /* cb_eval_user_data */ nullptr,
            /* no_alloc          */ false,
            /* progress_callback */ nullptr,
            /* progress_callback_user_data */ nullptr,
        };
        auto loaded = clip_init(mmproj_path, params);
        if (loaded.ctx_a) clip_free(loaded.ctx_a);
        if (loaded.ctx_gen_a) clip_free(loaded.ctx_gen_a);
        if (!loaded.ctx_v) {
            fail("clip failed to load a vision projector");
            return nullptr;
        }
        if (clip_get_projector_type(loaded.ctx_v) != PROJECTOR_TYPE_MUSE_GLIMMER) {
            clip_free(loaded.ctx_v);
            fail("projector is not Muse Glimmer");
            return nullptr;
        }
        auto handle = std::make_unique<bridge_context>();
        handle->clip = loaded.ctx_v;
        handle->preprocessor =
            std::make_unique<mtmd_image_preprocessor_muse_glimmer>(handle->clip);
        return handle.release();
    } catch (const std::exception & error) {
        fail(error.what());
        return nullptr;
    }
}

void muser_mtmd_free(void * opaque) {
    auto * handle = static_cast<bridge_context *>(opaque);
    delete handle;
}

int muser_mtmd_preprocess_rgb(
    void * opaque,
    const unsigned char * rgb,
    size_t rgb_bytes,
    uint32_t width,
    uint32_t height,
    float * output,
    size_t output_capacity,
    uint32_t * output_width,
    uint32_t * output_height,
    size_t * output_elements) {
    last_error.clear();
    if (!output || !output_width || !output_height || !output_elements) {
        return fail("invalid preprocessing output");
    }
    try {
        const auto image = preprocess(
            static_cast<bridge_context *>(opaque), rgb, rgb_bytes, width, height);
        const auto & pixels = image.get_ro_buf();
        *output_width = (uint32_t) image.nx();
        *output_height = (uint32_t) image.ny();
        *output_elements = pixels.size();
        if (pixels.size() > output_capacity) {
            return fail("preprocessing output capacity is too small");
        }
        std::memcpy(output, pixels.data(), pixels.size() * sizeof(float));
        return 0;
    } catch (const std::exception & error) {
        return fail(error.what());
    }
}

int muser_mtmd_encode_rgb(
    void * opaque,
    const unsigned char * rgb,
    size_t rgb_bytes,
    uint32_t width,
    uint32_t height,
    size_t expected_embedding_dim,
    float * output,
    size_t output_capacity,
    size_t * output_tokens) {
    last_error.clear();
    auto * handle = static_cast<bridge_context *>(opaque);
    if (!handle || !handle->clip || !output || !output_tokens ||
        expected_embedding_dim == 0) {
        return fail("invalid encoding input");
    }
    try {
        const auto image = preprocess(handle, rgb, rgb_bytes, width, height);
        const size_t actual_dim = (size_t) clip_n_mmproj_embd(handle->clip);
        if (actual_dim != expected_embedding_dim) {
            return fail("projector embedding dimension differs from Muser GGUF parsing");
        }
        const int count = clip_n_output_tokens(handle->clip, &image);
        if (count <= 0 || (size_t) count > SIZE_MAX / actual_dim) {
            return fail("projected-token count is invalid");
        }
        const size_t elements = (size_t) count * actual_dim;
        *output_tokens = (size_t) count;
        if (elements > output_capacity) {
            return fail("output capacity is smaller than projected embeddings");
        }
        // This pinned mtmd API treats an empty vector as an explicit request
        // to execute the graph without copying its final embedding tensor.
        // Bind the exact destination shape before evaluation.
        std::vector<float> encoded(elements);
        if (!clip_image_encode(handle->clip, 1, &image, encoded) ||
            encoded.size() != elements) {
            return fail("Muse vision graph failed or emitted an invalid shape");
        }
        std::memcpy(output, encoded.data(), elements * sizeof(float));
        return 0;
    } catch (const std::exception & error) {
        return fail(error.what());
    }
}

} // extern "C"
