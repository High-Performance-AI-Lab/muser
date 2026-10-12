// Spark disaggregated-prefill: llama.cpp KV producer export leg (POC).
//
// Loads a GGUF model on the GX10 GPU and prefills the exact prompt token IDs
// from a fixture file (one u32 per line) into a single sequence. In live mode,
// NoPE tiles are copied from CUDA as they become complete; FIFO writes run on
// a bounded host queue so the next llama_decode overlaps TLS backpressure. The
// exact logical SWA tail is gathered from llama.cpp's cell metadata and
// streamed in the same 512-token tiles as NoPE once those positions are
// inside the final window, so the ~78 MiB SWA burst overlaps the last
// 2048 tokens of CUDA instead of draining after prefill. No
// target session file is materialized. The consumer evaluates the final prompt
// token itself, so only N-1 positions are cached and shipped.
//
// Layout facts (llama.cpp master, verified against llama-kv-cache.cpp
// state_write_data): with flash attention ENABLED, v_trans = false, so both
// K and V serialize per layer as rows of n_embd_k_gqa/n_embd_v_gqa fp16
// values per cell, cell-major — i.e. exactly the canonical
// canonical-kv-f16-le-v1 plane layout [token][kv_head][head_dim], little
// endian. scripts/gx10/llamacpp/llamacpp_session_send.py parses and ships
// the planes without any transpose.
//
// Prints epoch-ms marker lines compatible with the live harness:
//   prefill_compute_start_epoch_ms <ms>   (regexed by the product driver)
//
// MUSER_LIVE_BATCH_FULL=1 (default off, A/B only): live mode keeps the job's
// full n_batch instead of clamping the prefill step to one 512-token tile. The
// emitted frame schedule is unchanged — tiles stay 512-aligned, in ascending
// order, with the same ranges — only the number of tiles completed per
// llama_decode changes (up to four at n_batch=2048). The export drain is one
// llama_synchronize per decode batch rather than one per tile, which is the
// documented fallback for the missing per-ubatch fence: the pinned llama.cpp
// exposes no cheap event that is guaranteed to be recorded after a ubatch's
// SET_ROWS, and reaching cudaMemcpyAsync on a side stream would require CUDA
// headers and a cudart link that the pinned build command below does not
// carry. Residual cost with the flag on: the per-tile device-to-host copy
// (6.8 MiB NoPE, up to 20.4 MiB SWA) is still synchronous, but the pipeline
// drain that precedes it is paid once per 2048 tokens instead of once per 512.
//
// Build (on the GX10, in the CUDA devel container, against the existing
// llama.cpp build tree):
//   g++ -O2 -std=c++17 -pthread spark_kv_export.cpp -I/src/include -I/src/ggml/include \
//       -L/src/build/bin -lllama -lggml -lggml-base -lggml-cpu -lggml-cuda \
//       -Wl,-rpath,/src/build/bin -o /src/build/bin/spark_kv_export

#include "llama.h"
#include "llama-ext.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "ggml-backend.h"
#include "llama-kv-cache-iswa.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <fstream>
#include <iterator>
#include <mutex>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#ifndef F_SETPIPE_SZ
#define F_SETPIPE_SZ 1031
#endif

struct MultimodalSegment {
    bool image;
    std::string path;
    int projected_tokens;
    std::string sha256;
};

struct TensorDump {
    std::string name;
    std::string path;
    bool written = false;
};

struct DumpState {
    std::vector<TensorDump> dumps;
    bool decode_only = false;
    bool in_decode = false;
};

static bool dump_tensor_callback(ggml_tensor * tensor, bool ask, void * opaque) {
    auto * state = static_cast<DumpState *>(opaque);
    if (state->decode_only && !state->in_decode) {
        return false;
    }
    auto * dumps = &state->dumps;
    auto it = std::find_if(dumps->begin(), dumps->end(), [&](const TensorDump & dump) {
        return !dump.written && dump.name == tensor->name;
    });
    if (ask) {
        return it != dumps->end();
    }
    if (it == dumps->end()) {
        return true;
    }
    if (tensor->type != GGML_TYPE_F32) {
        std::fprintf(stderr, "tensor dump %s has type %s, expected f32\n",
            tensor->name, ggml_type_name(tensor->type));
        std::exit(1);
    }
    const size_t bytes = ggml_nbytes(tensor);
    // Prefill batches request no logits and therefore expose result_output as
    // a legitimate zero-row tensor. Keep waiting for the held boundary-token
    // decode rather than sealing an empty diagnostic file.
    if (bytes == 0) {
        return true;
    }
    std::vector<uint8_t> contents(bytes);
    ggml_backend_tensor_get(tensor, contents.data(), 0, bytes);
    std::ofstream output(it->path, std::ios::binary | std::ios::trunc);
    if (!output) {
        std::fprintf(stderr, "cannot create tensor dump %s\n", it->path.c_str());
        std::exit(1);
    }
    output.write(reinterpret_cast<const char *>(contents.data()), contents.size());
    if (!output) {
        std::fprintf(stderr, "cannot write tensor dump %s\n", it->path.c_str());
        std::exit(1);
    }
    it->written = true;
    std::fprintf(stderr,
        "[spark-kv-export] dumped tensor %s type=f32 shape=%lld,%lld,%lld,%lld bytes=%zu path=%s\n",
        tensor->name, (long long) tensor->ne[0], (long long) tensor->ne[1],
        (long long) tensor->ne[2], (long long) tensor->ne[3], bytes, it->path.c_str());
    return true;
}

#pragma pack(push, 1)
struct StreamPlaneHeader {
    char magic[8];
    uint32_t layer;
    uint8_t role;
    uint64_t logical_start;
    uint32_t logical_count;
    uint32_t elements_per_token;
    uint64_t byte_len;
};
#pragma pack(pop)

// Bounded host queue so llama_decode of tile i+1 overlaps fwrite/TLS of tile i.
// The FIFO byte stream is still concatenated StreamPlaneHeader + payload frames.
class KvStream {
public:
    KvStream() = default;
    KvStream(const KvStream &) = delete;
    KvStream & operator=(const KvStream &) = delete;
    ~KvStream() { (void) close(); }

    // queue_depth bounds the host queue in blobs; each blob is one packed tile
    // (26 NoPE planes or a 13-layer SWA group) of at most 512 * 512 bytes per
    // plane, i.e. ~6.8 MiB.
    int open(const char * path, size_t queue_depth) {
        max_queued = queue_depth;
        file = std::fopen(path, "wb");
        if (!file) {
            std::fprintf(stderr, "cannot open live NoPE FIFO %s\n", path);
            return 1;
        }
        const int fd = fileno(file);
        if (fd >= 0 && fcntl(fd, F_SETPIPE_SZ, 8 << 20) < 0) {
            (void) fcntl(fd, F_SETPIPE_SZ, 1 << 20);
        }
        std::setvbuf(file, nullptr, _IOFBF, 1 << 20);
        worker = std::thread(&KvStream::run, this);
        return 0;
    }

    int enqueue(std::vector<unsigned char> && blob) {
        if (blob.empty()) {
            return 0;
        }
        std::unique_lock<std::mutex> lock(mu);
        cv.wait(lock, [&] { return queue.size() < max_queued || failed; });
        if (failed) {
            std::fprintf(stderr, "live KV FIFO writer failed\n");
            return 1;
        }
        queue.push(std::move(blob));
        cv.notify_all();
        return 0;
    }

    int close() {
        if (closed) {
            return 0;
        }
        closed = true;
        {
            std::lock_guard<std::mutex> lock(mu);
            stop = true;
        }
        cv.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
        int rc = 0;
        if (file != nullptr) {
            if (std::fclose(file) != 0) {
                std::fprintf(stderr, "close live NoPE FIFO failed\n");
                rc = 1;
            }
            file = nullptr;
        }
        if (failed) {
            std::fprintf(stderr, "live KV FIFO writer failed\n");
            rc = 1;
        }
        return rc;
    }

private:
    size_t max_queued = 8;
    FILE * file = nullptr;
    std::thread worker;
    std::mutex mu;
    std::condition_variable cv;
    std::queue<std::vector<unsigned char>> queue;
    bool stop = false;
    bool failed = false;
    bool closed = false;

    void run() {
        while (true) {
            std::vector<unsigned char> blob;
            {
                std::unique_lock<std::mutex> lock(mu);
                cv.wait(lock, [&] { return !queue.empty() || stop || failed; });
                if (queue.empty()) {
                    break;
                }
                blob = std::move(queue.front());
                queue.pop();
            }
            cv.notify_all();
            const unsigned char * cursor = blob.data();
            size_t remaining = blob.size();
            while (remaining > 0) {
                const size_t written = std::fwrite(cursor, 1, remaining, file);
                if (written == 0) {
                    std::lock_guard<std::mutex> lock(mu);
                    failed = true;
                    cv.notify_all();
                    return;
                }
                cursor += written;
                remaining -= written;
            }
            if (std::fflush(file) != 0) {
                std::lock_guard<std::mutex> lock(mu);
                failed = true;
                cv.notify_all();
                return;
            }
        }
    }
};

struct DFlashInjector {
    llama_context * target_ctx;
    llama_context * draft_ctx;
    llama_batch * draft_inject;
    const int32_t * target_layer_ids;
    uint32_t target_layer_ids_n;
    int32_t target_hidden;
    int32_t draft_hidden;
    int32_t n_ubatch;
    std::vector<float> * features;
    const std::vector<float> * external_features;
    KvStream * kv_stream;
    uint64_t * next_nope_start;
    uint64_t n_prefill;
};

// Default off. Read once so an A/B arm cannot change mid-run inside the warm
// engine's job loop.
static bool live_batch_full() {
    static const bool enabled = [] {
        const char * raw = std::getenv("MUSER_LIVE_BATCH_FULL");
        return raw != nullptr && std::strcmp(raw, "1") == 0;
    }();
    return enabled;
}

static bool is_nope_layer(uint32_t layer) {
    return layer >= 3 && layer < 52 && layer % 4 == 3;
}

static void append_plane_header(
        std::vector<unsigned char> & blob,
        uint32_t layer,
        uint8_t role,
        uint64_t start,
        uint32_t count,
        size_t bytes) {
    const StreamPlaneHeader header = {
        {'M', 'U', 'S', 'E', 'N', 'P', '1', '\0'},
        layer,
        role,
        start,
        count,
        256,
        (uint64_t) bytes,
    };
    const size_t off = blob.size();
    blob.resize(off + sizeof(header));
    std::memcpy(blob.data() + off, &header, sizeof(header));
}

static int stream_nope_tile(
        llama_context * ctx,
        KvStream * stream,
        uint64_t start,
        uint32_t count,
        bool already_synced) {
    auto * iswa = dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(ctx));
    if (!iswa) {
        std::fprintf(stderr, "Muse streaming export requires llama_kv_cache_iswa\n");
        return 1;
    }
    llama_kv_cache * base = iswa->get_base();
    if (base->values_are_transposed()) {
        std::fprintf(stderr, "Muse streaming export requires flash-attention V layout\n");
        return 1;
    }
    // The caller must have drained the decode that wrote [start, start+count);
    // no KV row may be read before its graph's SET_ROWS has completed.
    if (!already_synced) {
        llama_synchronize(ctx);
    }
    const auto layer_ids = base->get_layer_ids();
    std::vector<unsigned char> blob;
    blob.reserve((size_t) 26 * (sizeof(StreamPlaneHeader) + (size_t) count * 512));
    for (uint32_t layer : layer_ids) {
        if (!is_nope_layer(layer)) continue;
        for (uint8_t role = 0; role < 2; ++role) {
            ggml_tensor * tensor = role == 0
                ? base->get_k_storage((int32_t) layer)
                : base->get_v_storage((int32_t) layer);
            if (!tensor || tensor->type != GGML_TYPE_F16 || tensor->nb[0] != 2 ||
                tensor->ne[0] != 256 || tensor->nb[1] != 512 ||
                start + count > (uint64_t) tensor->ne[1]) {
                std::fprintf(stderr,
                    "Muse streaming plane geometry mismatch at layer %u role %u\n",
                    layer, role);
                return 1;
            }
            const size_t bytes = (size_t) count * tensor->nb[1];
            append_plane_header(blob, layer, role, start, count, bytes);
            const size_t off = blob.size();
            blob.resize(off + bytes);
            ggml_backend_tensor_get(tensor, blob.data() + off, (size_t) start * tensor->nb[1], bytes);
        }
    }
    return stream->enqueue(std::move(blob));
}

static uint64_t swa_window_start(uint64_t position) {
    return position > 2048 ? position - 2048 : 0;
}

static int stream_swa_range(
        llama_context * ctx,
        KvStream * stream,
        uint64_t start,
        uint32_t count,
        bool already_synced) {
    if (count == 0) {
        return 0;
    }
    auto * iswa = dynamic_cast<llama_kv_cache_iswa *>(llama_get_memory(ctx));
    if (!iswa) {
        std::fprintf(stderr, "Muse SWA export requires llama_kv_cache_iswa\n");
        return 1;
    }
    llama_kv_cache * swa = iswa->get_swa();
    if (swa->values_are_transposed()) {
        std::fprintf(stderr, "Muse SWA export requires flash-attention V layout\n");
        return 1;
    }
    const auto cells = swa->get_cells_for_positions(0, (llama_pos) start, count);
    if (cells.size() != count) {
        std::fprintf(stderr,
            "Muse SWA metadata lacks exact logical range [%llu, %llu)\n",
            (unsigned long long) start, (unsigned long long) (start + count));
        return 1;
    }
    if (!already_synced) {
        llama_synchronize(ctx);
    }
    std::vector<uint32_t> layers;
    for (uint32_t layer : swa->get_layer_ids()) {
        if (is_nope_layer(layer)) {
            std::fprintf(stderr, "NoPE layer %u appeared in the SWA cache\n", layer);
            return 1;
        }
        layers.push_back(layer);
    }
    constexpr size_t kGroup = 13;
    for (size_t group = 0; group < layers.size(); group += kGroup) {
        const size_t n_layers = std::min(kGroup, layers.size() - group);
        std::vector<unsigned char> blob;
        blob.reserve(n_layers * 2 * (sizeof(StreamPlaneHeader) + (size_t) count * 512));
        for (size_t i = 0; i < n_layers; ++i) {
            const uint32_t layer = layers[group + i];
            for (uint8_t plane = 0; plane < 2; ++plane) {
                ggml_tensor * tensor = plane == 0
                    ? swa->get_k_storage((int32_t) layer)
                    : swa->get_v_storage((int32_t) layer);
                if (!tensor || tensor->type != GGML_TYPE_F16 || tensor->nb[0] != 2 ||
                    tensor->ne[0] != 256 || tensor->nb[1] != 512) {
                    std::fprintf(stderr,
                        "Muse SWA plane geometry mismatch at layer %u plane %u\n",
                        layer, plane);
                    return 1;
                }
                const size_t bytes = (size_t) count * tensor->nb[1];
                append_plane_header(blob, layer, (uint8_t) (plane + 2), start, count, bytes);
                const size_t off = blob.size();
                blob.resize(off + bytes);
                unsigned char * dest = blob.data() + off;
                for (uint32_t logical = 0; logical < count;) {
                    const uint32_t physical = cells[logical];
                    if ((uint64_t) physical >= (uint64_t) tensor->ne[1]) {
                        std::fprintf(stderr, "Muse SWA cell lies outside tensor storage\n");
                        return 1;
                    }
                    uint32_t run = 1;
                    while (logical + run < count &&
                           cells[logical + run] == physical + run) {
                        ++run;
                    }
                    ggml_backend_tensor_get(
                        tensor,
                        dest + (size_t) logical * tensor->nb[1],
                        (size_t) physical * tensor->nb[1],
                        (size_t) run * tensor->nb[1]);
                    logical += run;
                }
            }
        }
        if (stream->enqueue(std::move(blob)) != 0) {
            return 1;
        }
    }
    return 0;
}

static int stream_swa_for_completed_tile(
        llama_context * ctx,
        KvStream * stream,
        uint64_t n_prefill,
        uint64_t tile_start,
        uint32_t tile_count) {
    const uint64_t window_start = swa_window_start(n_prefill);
    const uint64_t tile_end = tile_start + tile_count;
    if (tile_end <= window_start) {
        return 0;
    }
    const uint64_t chunk_start = std::max(tile_start, window_start);
    return stream_swa_range(ctx, stream, chunk_start, (uint32_t) (tile_end - chunk_start), true);
}

static int stream_complete_nope(DFlashInjector * state, const llama_batch & batch) {
    if (!state || !state->kv_stream || !state->next_nope_start) return 0;
    uint64_t complete = *state->next_nope_start;
    for (int row = 0; row < batch.n_tokens; ++row) {
        complete = std::max(complete, (uint64_t) batch.pos[row] + 1);
    }
    // One drain per decode batch, not per tile: once llama_synchronize returns,
    // every ubatch of the batch that produced these positions has finished,
    // including its KV SET_ROWS. A full-width live batch therefore pays one
    // drain for up to four tiles instead of four drains.
    bool synced = false;
    while (*state->next_nope_start < complete) {
        const uint64_t remaining = complete - *state->next_nope_start;
        const uint32_t count = (uint32_t) std::min<uint64_t>(512, remaining);
        if (count < 512 && complete < state->n_prefill) {
            break;
        }
        if (!synced) {
            llama_synchronize(state->target_ctx);
            synced = true;
        }
        const uint64_t start = *state->next_nope_start;
        if (stream_nope_tile(state->target_ctx, state->kv_stream, start, count, true) != 0) {
            return 1;
        }
        if (stream_swa_for_completed_tile(
                state->target_ctx, state->kv_stream, state->n_prefill, start, count) != 0) {
            return 1;
        }
        *state->next_nope_start += count;
    }
    return 0;
}

// Consume the target-layer rows produced by exactly one target decode batch
// and advance the DFlash cache at those same decoder positions. This callback
// is shared by text batches and mtmd's internally-batched projected image rows.
static int inject_dflash_batch(llama_batch target_batch, void * opaque) {
    auto * state = static_cast<DFlashInjector *>(opaque);
    if (!state) return 0;
    if (!state->draft_ctx) return stream_complete_nope(state, target_batch);
    const int32_t encoded_width =
        (int32_t) state->target_layer_ids_n * state->target_hidden;
    for (int offset = 0; offset < target_batch.n_tokens; offset += state->n_ubatch) {
        const int chunk = std::min(state->n_ubatch, target_batch.n_tokens - offset);
        state->features->resize((size_t) chunk * encoded_width);
        if (state->external_features) {
            const int first_pos = target_batch.pos[offset];
            if (first_pos < 0 || first_pos + chunk > (int) state->n_prefill) {
                std::fprintf(stderr, "external DFlash feature positions are out of range\n");
                return 1;
            }
            for (int row = 0; row < chunk; ++row) {
                if (target_batch.pos[offset + row] != first_pos + row) {
                    std::fprintf(stderr, "external DFlash feature positions are not contiguous\n");
                    return 1;
                }
            }
            const float * src = state->external_features->data() +
                (size_t) first_pos * encoded_width;
            std::memcpy(state->features->data(), src,
                (size_t) chunk * encoded_width * sizeof(float));
        } else {
            for (uint32_t k = 0; k < state->target_layer_ids_n; ++k) {
                const float * layer = llama_get_embeddings_layer_inp(
                    state->target_ctx, (uint32_t) state->target_layer_ids[k]);
                if (!layer) {
                    std::fprintf(stderr, "target layer %d was not captured for DFlash\n",
                        state->target_layer_ids[k]);
                    return 1;
                }
                for (int row = 0; row < chunk; ++row) {
                    float * dst = state->features->data() +
                        (size_t) row * encoded_width + (size_t) k * state->target_hidden;
                    const float * src = layer +
                        (size_t) (offset + row) * state->target_hidden;
                    std::memcpy(dst, src, (size_t) state->target_hidden * sizeof(float));
                }
            }
        }
        llama_batch encode_batch = {
            /*.n_tokens =*/ chunk,
            /*.token    =*/ nullptr,
            /*.embd     =*/ state->features->data(),
            /*.pos      =*/ nullptr,
            /*.n_seq_id =*/ nullptr,
            /*.seq_id   =*/ nullptr,
            /*.logits   =*/ nullptr,
        };
        const int encode_rc = llama_encode(state->draft_ctx, encode_batch);
        if (encode_rc != 0) {
            std::fprintf(stderr, "DFlash encoder failed rc=%d\n", encode_rc);
            return encode_rc;
        }
        const float * fused = llama_get_embeddings_nextn(state->draft_ctx);
        if (!fused) {
            std::fprintf(stderr, "DFlash encoder produced no fused embeddings\n");
            return 1;
        }
        state->draft_inject->n_tokens = chunk;
        std::memcpy(state->draft_inject->embd, fused,
            (size_t) chunk * state->draft_hidden * sizeof(float));
        for (int row = 0; row < chunk; ++row) {
            const int source = offset + row;
            state->draft_inject->pos[row] = target_batch.pos[source];
            state->draft_inject->n_seq_id[row] = 1;
            state->draft_inject->seq_id[row][0] = 0;
            state->draft_inject->logits[row] = 0;
        }
        // The draft's bidirectional attention is a distinct f32 ABI: Muser's
        // Metal kernel keeps Q/K/V in f32 and reduces one 128-wide head across
        // a 32-lane SIMD group. Select the matching CUDA compatibility kernel
        // only while the draft graph is submitted; target attention retains
        // its independently qualified arithmetic.
        if (setenv("MUSER_DFLASH_ATTENTION_F32", "1", 1) != 0) {
            std::fprintf(stderr, "cannot select DFlash f32 attention parity mode\n");
            return 1;
        }
        // The draft's NEOX rope must consume the same canonical NCO (cos,sin)
        // bytes as the Mac's strict route. Select the integer-phase kernel
        // only while the draft context injection is submitted; the target's
        // qualified rope path is unreachable through it.
        if (setenv("MUSER_DFLASH_ROPE_NCO", "1", 1) != 0) {
            std::fprintf(stderr, "cannot select DFlash NCO rope parity mode\n");
            return 1;
        }
        const int inject_rc = llama_decode(state->draft_ctx, *state->draft_inject);
        if (unsetenv("MUSER_DFLASH_ATTENTION_F32") != 0) {
            std::fprintf(stderr, "cannot clear DFlash f32 attention parity mode\n");
            return 1;
        }
        if (unsetenv("MUSER_DFLASH_ROPE_NCO") != 0) {
            std::fprintf(stderr, "cannot clear DFlash NCO rope parity mode\n");
            return 1;
        }
        if (inject_rc != 0) {
            std::fprintf(stderr, "DFlash cache injection failed rc=%d\n", inject_rc);
            return inject_rc;
        }
    }
    return stream_complete_nope(state, target_batch);
}

static std::vector<llama_token> read_tokens(const std::string & path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open token fixture " + path);
    }
    std::vector<llama_token> output;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty()) {
            output.push_back((llama_token) std::strtol(line.c_str(), nullptr, 10));
        }
    }
    return output;
}

static std::vector<unsigned char> read_bytes(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open image " + path);
    }
    return std::vector<unsigned char>(std::istreambuf_iterator<char>(in), {});
}

static std::vector<float> read_dflash_features(
        const std::string & path, size_t expected_values) {
    const uint32_t endian_probe = 1;
    if (*reinterpret_cast<const unsigned char *>(&endian_probe) != 1) {
        throw std::runtime_error("external DFlash f32-le requires a little-endian host");
    }
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("cannot open external DFlash features " + path);
    }
    const std::streamoff size = in.tellg();
    const size_t expected_bytes = expected_values * sizeof(float);
    if (size < 0 || (uint64_t) size != (uint64_t) expected_bytes) {
        throw std::runtime_error(
            "external DFlash feature byte count differs from prompt geometry");
    }
    std::vector<float> output(expected_values);
    in.seekg(0);
    if (!in.read(reinterpret_cast<char *>(output.data()), (std::streamsize) expected_bytes)) {
        throw std::runtime_error("cannot read complete external DFlash feature file");
    }
    return output;
}

static std::vector<MultimodalSegment> read_multimodal_plan(const std::string & path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open multimodal plan " + path);
    }
    std::vector<MultimodalSegment> output;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<std::string> fields;
        std::stringstream stream(line);
        std::string field;
        while (std::getline(stream, field, '\t')) fields.push_back(field);
        if (fields.size() == 2 && fields[0] == "tokens") {
            output.push_back({false, fields[1], 0, {}});
        } else if (fields.size() == 4 && fields[0] == "image") {
            const int projected = std::stoi(fields[2]);
            if (projected < 1 || fields[3].size() != 64) {
                throw std::runtime_error("multimodal image plan fields are invalid");
            }
            output.push_back({true, fields[1], projected, fields[3]});
        } else {
            throw std::runtime_error("multimodal plan line is invalid");
        }
    }
    if (output.empty() || output.back().image) {
        throw std::runtime_error("multimodal plan must end with a token segment");
    }
    return output;
}

static int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

struct JobRequest {
    std::string tokens_path;
    std::string out_path;
    std::string load_path;
    std::string nope_fifo_path;
    std::string draft_out_path;
    std::string dflash_features_path;
    std::string multimodal_plan_path;
    std::string stdout_path;
    std::string status_path;
    int n_ctx = 32768;
    int n_batch = 2048;
    int n_ubatch = 512;
    int flash_attn = 1;
    int threads = 4;
    int skip_tail = 1;
};

struct Engine {
    llama_model * model = nullptr;
    llama_model * draft_model = nullptr;
    mtmd_context * vision_ctx = nullptr;
    const int32_t * target_layer_ids = nullptr;
    uint32_t target_layer_ids_n = 0;
    int32_t target_hidden = 0;
    int32_t draft_hidden = 0;
    DumpState * dump_state = nullptr;
    bool cpu_math_oracle_layer0 = false;
    llama_context * cached_ctx = nullptr;
    int cached_n_ctx = 0;
    int cached_n_batch = 0;
    int cached_n_ubatch = 0;
    int cached_flash_attn = -1;
};

struct JobSession {
    llama_context * ctx = nullptr;
    llama_context * draft_ctx = nullptr;
    llama_batch batch = {};
    llama_batch draft_inject = {};
    bool batch_ok = false;
    bool draft_inject_ok = false;
    FILE * markers = stdout;
    bool close_markers = false;
    bool keep_ctx = false;

    ~JobSession() {
        if (batch_ok) llama_batch_free(batch);
        if (draft_inject_ok) llama_batch_free(draft_inject);
        if (draft_ctx) llama_free(draft_ctx);
        if (ctx && !keep_ctx) llama_free(ctx);
        if (close_markers && markers && markers != stdout) {
            std::fclose(markers);
        }
    }
};

static bool valid_abs_path(const std::string & path) {
    return !path.empty() && path[0] == '/' && path.find('\0') == std::string::npos;
}

static int parse_int_field(const std::string & raw, int min, int max, int * out) {
    char * end = nullptr;
    const long value = std::strtol(raw.c_str(), &end, 10);
    if (!end || *end || value < min || value > max) {
        return 1;
    }
    *out = (int) value;
    return 0;
}

static int parse_job_file(const char * path, JobRequest * job) {
    if (!valid_abs_path(path)) {
        std::fprintf(stderr, "job path must be absolute\n");
        return 1;
    }
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "cannot open job file %s\n", path);
        return 1;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const size_t split = line.find(' ');
        if (split == std::string::npos || split == 0 || split + 1 == line.size()) {
            std::fprintf(stderr, "invalid job line in %s\n", path);
            return 1;
        }
        const std::string key = line.substr(0, split);
        const std::string value = line.substr(split + 1);
        auto take_path = [&](std::string * dest) {
            if (!valid_abs_path(value)) {
                std::fprintf(stderr, "job field %s is not an absolute path\n", key.c_str());
                return 1;
            }
            *dest = value;
            return 0;
        };
        if (key == "tokens") {
            if (take_path(&job->tokens_path) != 0) return 1;
        } else if (key == "out") {
            if (take_path(&job->out_path) != 0) return 1;
        } else if (key == "load_session") {
            if (take_path(&job->load_path) != 0) return 1;
        } else if (key == "nope_fifo") {
            if (take_path(&job->nope_fifo_path) != 0) return 1;
        } else if (key == "draft_out") {
            if (take_path(&job->draft_out_path) != 0) return 1;
        } else if (key == "dflash_features") {
            if (take_path(&job->dflash_features_path) != 0) return 1;
        } else if (key == "multimodal_plan") {
            if (take_path(&job->multimodal_plan_path) != 0) return 1;
        } else if (key == "stdout") {
            if (take_path(&job->stdout_path) != 0) return 1;
        } else if (key == "status") {
            if (take_path(&job->status_path) != 0) return 1;
        } else if (key == "n_ctx") {
            if (parse_int_field(value, 2, 131072, &job->n_ctx) != 0) return 1;
        } else if (key == "n_batch") {
            if (parse_int_field(value, 1, 2048, &job->n_batch) != 0) return 1;
        } else if (key == "n_ubatch") {
            if (parse_int_field(value, 1, 512, &job->n_ubatch) != 0) return 1;
        } else if (key == "flash_attn") {
            if (parse_int_field(value, 0, 1, &job->flash_attn) != 0) return 1;
        } else if (key == "threads") {
            if (parse_int_field(value, 1, 64, &job->threads) != 0) return 1;
        } else if (key == "skip_tail") {
            if (parse_int_field(value, 0, 1, &job->skip_tail) != 0) return 1;
        } else {
            std::fprintf(stderr, "unknown job field %s\n", key.c_str());
            return 1;
        }
    }
    if (job->tokens_path.empty() || job->status_path.empty() ||
        (job->out_path.empty() && job->load_path.empty() && job->nope_fifo_path.empty() &&
         job->draft_out_path.empty())) {
        std::fprintf(stderr, "job file %s is missing required fields\n", path);
        return 1;
    }
    if (!job->nope_fifo_path.empty() && !job->load_path.empty()) {
        std::fprintf(stderr, "job cannot combine nope_fifo with load_session\n");
        return 1;
    }
    if (!job->dflash_features_path.empty() &&
        (job->draft_out_path.empty() || !job->out_path.empty() || !job->load_path.empty() ||
         !job->nope_fifo_path.empty() || !job->multimodal_plan_path.empty())) {
        std::fprintf(stderr,
            "external DFlash features require draft_out and cannot combine with target outputs\n");
        return 1;
    }
    return 0;
}

static void write_status(const std::string & path, int rc) {
    if (path.empty()) {
        return;
    }
    const std::string temporary = path + ".tmp";
    FILE * out = std::fopen(temporary.c_str(), "w");
    if (!out) {
        std::fprintf(stderr, "cannot write job status %s\n", path.c_str());
        return;
    }
    std::fprintf(out, "%s\n", rc == 0 ? "ok" : "fail");
    std::fflush(out);
    fsync(fileno(out));
    std::fclose(out);
    if (std::rename(temporary.c_str(), path.c_str()) != 0) {
        std::fprintf(stderr, "cannot publish job status %s\n", path.c_str());
    }
}

static int run_job(Engine * engine, const JobRequest & job);

static int warmup_cuda(Engine * engine) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    cparams.n_batch = 512;
    cparams.n_ubatch = 512;
    cparams.n_seq_max = 1;
    cparams.n_threads = 4;
    cparams.n_threads_batch = 4;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cparams.type_k = GGML_TYPE_F16;
    cparams.type_v = GGML_TYPE_F16;
    llama_context * ctx = llama_init_from_model(engine->model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "CUDA warmup context init failed\n");
        return 1;
    }
    llama_batch batch = llama_batch_init(1, 0, 1);
    batch.n_tokens = 1;
    batch.token[0] = 1;
    batch.pos[0] = 0;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = 0;
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    llama_free(ctx);
    if (rc != 0) {
        std::fprintf(stderr, "CUDA warmup decode failed rc=%d\n", rc);
        return 1;
    }
    std::fprintf(stderr, "[spark-kv-export] CUDA warmup complete\n");
    std::fflush(stderr);
    return 0;
}

static bool dump_requested(const Engine * engine) {
    return engine->dump_state != nullptr && !engine->dump_state->dumps.empty();
}

static bool can_reuse_context(const Engine * engine, const JobRequest & job) {
    return engine->cached_ctx != nullptr
        && engine->cached_n_ctx >= job.n_ctx
        && engine->cached_n_batch == job.n_batch
        && engine->cached_n_ubatch == job.n_ubatch
        && engine->cached_flash_attn == job.flash_attn
        && !dump_requested(engine)
        && !engine->cpu_math_oracle_layer0
        && job.draft_out_path.empty()
        && job.dflash_features_path.empty();
}

static void discard_cached_context(Engine * engine) {
    if (engine->cached_ctx) {
        llama_free(engine->cached_ctx);
        engine->cached_ctx = nullptr;
    }
    engine->cached_n_ctx = 0;
    engine->cached_n_batch = 0;
    engine->cached_n_ubatch = 0;
    engine->cached_flash_attn = -1;
}

static int serve_jobs(Engine * engine, const std::string & fifo_path) {
    const int fd = open(fifo_path.c_str(), O_RDWR);
    if (fd < 0) {
        std::fprintf(stderr, "cannot open job FIFO %s: %s\n", fifo_path.c_str(), strerror(errno));
        return 1;
    }
    FILE * jobs = fdopen(fd, "r");
    if (!jobs) {
        std::fprintf(stderr, "cannot fdopen job FIFO %s\n", fifo_path.c_str());
        close(fd);
        return 1;
    }
    std::fprintf(stderr, "[spark-kv-export] engine ready, waiting for jobs\n");
    std::fflush(stderr);
    char line[4096];
    while (fgets(line, sizeof(line), jobs)) {
        size_t n = std::strlen(line);
        while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) {
            line[--n] = 0;
        }
        if (n == 0) {
            continue;
        }
        JobRequest job;
        if (parse_job_file(line, &job) != 0) {
            write_status(job.status_path, 1);
            continue;
        }
        write_status(job.status_path, run_job(engine, job));
    }
    std::fclose(jobs);
    discard_cached_context(engine);
    return 1;
}

static int run_job(Engine * engine, const JobRequest & job) {
    JobSession session;
    if (!job.stdout_path.empty()) {
        session.markers = std::fopen(job.stdout_path.c_str(), "w");
        if (!session.markers) {
            std::fprintf(stderr, "cannot open job stdout %s\n", job.stdout_path.c_str());
            return 1;
        }
        session.close_markers = true;
    }
    std::vector<llama_token> tokens;
    std::vector<MultimodalSegment> multimodal_plan;
    try {
        tokens = read_tokens(job.tokens_path);
        if (!job.multimodal_plan_path.empty()) {
            multimodal_plan = read_multimodal_plan(job.multimodal_plan_path);
        }
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 2;
    }
    const int n_prefill = (int) tokens.size() - job.skip_tail;
    if (n_prefill < 1 || n_prefill > job.n_ctx) {
        std::fprintf(stderr, "n_prefill %d out of range (tokens=%zu, n_ctx=%d)\n",
            n_prefill, tokens.size(), job.n_ctx);
        return 2;
    }
    if (!job.multimodal_plan_path.empty() && !engine->vision_ctx) {
        std::fprintf(stderr, "job requested a multimodal plan without a loaded projector\n");
        return 2;
    }
    if (!job.draft_out_path.empty() && !engine->draft_model) {
        std::fprintf(stderr, "job requested DFlash output without a loaded draft model\n");
        return 2;
    }
    std::vector<float> external_dflash_features;
    if (!job.dflash_features_path.empty()) {
        if (job.draft_out_path.empty() || !job.out_path.empty() || !job.load_path.empty() ||
            !job.nope_fifo_path.empty() || !job.multimodal_plan_path.empty()) {
            std::fprintf(stderr,
                "external DFlash features require an isolated DFlash-output job\n");
            return 2;
        }
        try {
            external_dflash_features = read_dflash_features(
                job.dflash_features_path,
                (size_t) n_prefill * engine->target_layer_ids_n * engine->target_hidden);
        } catch (const std::exception & error) {
            std::fprintf(stderr, "%s\n", error.what());
            return 2;
        }
    }

    std::fprintf(stderr, "[spark-kv-export] tokens=%zu prefill=%d n_ctx=%d fa=%d\n",
        tokens.size(), n_prefill, job.n_ctx, job.flash_attn);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = (uint32_t) job.n_ctx;
    cparams.n_batch         = (uint32_t) job.n_batch;
    cparams.n_ubatch        = (uint32_t) job.n_ubatch;
    cparams.n_seq_max       = 1;
    cparams.n_threads       = job.threads;
    cparams.n_threads_batch = job.threads;
    cparams.flash_attn_type = job.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                             : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.type_k = GGML_TYPE_F16;
    cparams.type_v = GGML_TYPE_F16;
    if (engine->dump_state && !engine->dump_state->dumps.empty()) {
        cparams.cb_eval = dump_tensor_callback;
        cparams.cb_eval_user_data = engine->dump_state;
    }
    if (engine->cpu_math_oracle_layer0) {
        cparams.offload_kqv = false;
    }

    int ctx_n_ctx = job.n_ctx;
    if (can_reuse_context(engine, job)) {
        session.ctx = engine->cached_ctx;
        ctx_n_ctx = engine->cached_n_ctx;
        engine->cached_ctx = nullptr;
        llama_memory_clear(llama_get_memory(session.ctx), true);
        std::fprintf(stderr, "[spark-kv-export] reused context n_ctx=%d for job n_ctx=%d\n",
            ctx_n_ctx, job.n_ctx);
    } else {
        discard_cached_context(engine);
        session.ctx = llama_init_from_model(engine->model, cparams);
        if (!session.ctx) {
            std::fprintf(stderr, "context init failed\n");
            return 1;
        }
    }

    std::vector<float> draft_features;
    if (engine->draft_model) {
        if (job.dflash_features_path.empty()) {
            for (uint32_t k = 0; k < engine->target_layer_ids_n; ++k) {
                llama_set_embeddings_layer_inp(
                    session.ctx, (uint32_t) engine->target_layer_ids[k], true);
            }
        }
        llama_context_params dparams = cparams;
        // Muser's DFlash context cache is f32. Inheriting the target's f16 KV
        // types rounded every GX-produced draft K/V row before the handoff;
        // widening those bytes in the sender cannot recover proposal parity.
        // Keep target KV f16, but retain the five-layer assistant state in the
        // exact format consumed by Metal.
        dparams.type_k = GGML_TYPE_F32;
        dparams.type_v = GGML_TYPE_F32;
        dparams.ctx_other = session.ctx;
        session.draft_ctx = llama_init_from_model(engine->draft_model, dparams);
        if (!session.draft_ctx) {
            std::fprintf(stderr, "DFlash context init failed\n");
            return 1;
        }
        llama_set_embeddings_nextn(session.draft_ctx, true, true);
        llama_set_causal_attn(session.draft_ctx, false);
        session.draft_inject = llama_batch_init(job.n_ubatch, engine->draft_hidden, 1);
        session.draft_inject_ok = true;
    }

    std::fprintf(stderr, "[spark-kv-export] engine ready, starting prefill\n");
    session.batch = llama_batch_init(job.n_batch, 0, 1);
    session.batch_ok = true;
    DFlashInjector dflash_injector = {
        session.ctx, session.draft_ctx, &session.draft_inject, engine->target_layer_ids,
        engine->target_layer_ids_n, engine->target_hidden, engine->draft_hidden,
        job.n_ubatch, &draft_features,
        external_dflash_features.empty() ? nullptr : &external_dflash_features,
        nullptr, nullptr, (uint64_t) n_prefill,
    };
    KvStream kv_stream;
    KvStream * live = nullptr;
    const bool live_full_batch = live_batch_full();
    if (!job.nope_fifo_path.empty()) {
        // A full-width live batch completes up to four tiles at once, and each
        // tile enqueues one NoPE blob plus up to three 13-layer SWA blobs: 16
        // blobs. The bounded host queue must absorb a whole batch or FIFO
        // backpressure lands back on the decode loop; 24 blobs of ~6.8 MiB
        // bounds the queue at ~164 MiB.
        if (kv_stream.open(job.nope_fifo_path.c_str(), live_full_batch ? 24 : 8) != 0) {
            return 1;
        }
        live = &kv_stream;
        if (live_full_batch) {
            std::fprintf(stderr,
                "[spark-kv-export] MUSER_LIVE_BATCH_FULL=1: live prefill keeps "
                "n_batch=%d (default-off experiment)\n", job.n_batch);
        }
    }
    // Live mode without the flag clamps the prefill step to one tile. Clamp to
    // n_batch too: session.batch only holds job.n_batch rows.
    const int prefill_step = (live && !live_full_batch)
        ? std::min(512, job.n_batch)
        : job.n_batch;
    uint64_t next_nope_start = 0;
    dflash_injector.kv_stream = live;
    dflash_injector.next_nope_start = &next_nope_start;

    const int64_t prefill_start = now_ms();
    std::fprintf(session.markers, "prefill_compute_start_epoch_ms %lld\n", (long long) prefill_start);
    std::fflush(session.markers);

    if (!job.dflash_features_path.empty()) {
        for (int i = 0; i < n_prefill;) {
            const int n = std::min(prefill_step, n_prefill - i);
            session.batch.n_tokens = n;
            for (int row = 0; row < n; ++row) {
                session.batch.token[row] = tokens[i + row];
                session.batch.pos[row] = i + row;
                session.batch.n_seq_id[row] = 1;
                session.batch.seq_id[row][0] = 0;
                session.batch.logits[row] = 0;
            }
            if (inject_dflash_batch(session.batch, &dflash_injector) != 0) {
                return 1;
            }
            i += n;
        }
    } else if (!job.load_path.empty()) {
        std::vector<llama_token> loaded(tokens.size());
        size_t n_loaded = 0;
        if (!llama_state_load_file(session.ctx, job.load_path.c_str(), loaded.data(),
                                   loaded.size(), &n_loaded)) {
            std::fprintf(stderr, "llama_state_load_file failed\n");
            return 1;
        }
        if ((int) n_loaded != n_prefill ||
            !std::equal(loaded.begin(), loaded.begin() + n_loaded, tokens.begin())) {
            std::fprintf(stderr, "loaded session prefix mismatch (%zu vs %d)\n",
                n_loaded, n_prefill);
            return 1;
        }
        std::fprintf(session.markers, "session_loaded_tokens %zu\n", n_loaded);
        std::fflush(session.markers);
    } else if (multimodal_plan.empty()) {
        for (int i = 0; i < n_prefill; ) {
            const int n = std::min(prefill_step, n_prefill - i);
            session.batch.n_tokens = n;
            for (int j = 0; j < n; ++j) {
                session.batch.token[j]     = tokens[i + j];
                session.batch.pos[j]       = i + j;
                session.batch.n_seq_id[j]  = 1;
                session.batch.seq_id[j][0] = 0;
                session.batch.logits[j]    = 0;
            }
            const int rc = llama_decode(session.ctx, session.batch);
            if (rc != 0) {
                std::fprintf(stderr, "llama_decode failed rc=%d at %d\n", rc, i);
                return 1;
            }
            if (inject_dflash_batch(session.batch, &dflash_injector) != 0) {
                return 1;
            }
            i += n;
        }
    } else {
        int n_past = 0;
        for (size_t segment_index = 0; segment_index < multimodal_plan.size(); ++segment_index) {
            const auto & segment = multimodal_plan[segment_index];
            if (!segment.image) {
                std::vector<llama_token> text;
                try {
                    text = read_tokens(segment.path);
                } catch (const std::exception & error) {
                    std::fprintf(stderr, "%s\n", error.what());
                    return 2;
                }
                if (segment_index + 1 == multimodal_plan.size()) {
                    if ((int) text.size() <= job.skip_tail) {
                        std::fprintf(stderr, "final text segment cannot hold the requested tail\n");
                        return 2;
                    }
                    text.resize(text.size() - job.skip_tail);
                }
                for (int offset = 0; offset < (int) text.size();) {
                    const int n = std::min(prefill_step, (int) text.size() - offset);
                    session.batch.n_tokens = n;
                    for (int row = 0; row < n; ++row) {
                        session.batch.token[row] = text[offset + row];
                        session.batch.pos[row] = n_past + row;
                        session.batch.n_seq_id[row] = 1;
                        session.batch.seq_id[row][0] = 0;
                        session.batch.logits[row] = 0;
                    }
                    const int rc = llama_decode(session.ctx, session.batch);
                    if (rc != 0) {
                        std::fprintf(stderr, "multimodal text decode failed rc=%d at %d\n", rc, n_past);
                        return 1;
                    }
                    if (inject_dflash_batch(session.batch, &dflash_injector) != 0) {
                        return 1;
                    }
                    n_past += n;
                    offset += n;
                }
                continue;
            }

            std::vector<unsigned char> bytes;
            try {
                bytes = read_bytes(segment.path);
            } catch (const std::exception & error) {
                std::fprintf(stderr, "%s\n", error.what());
                return 2;
            }
            mtmd_helper_bitmap_wrapper wrapper = mtmd_helper_bitmap_init_from_buf(
                engine->vision_ctx, bytes.data(), bytes.size(), false);
            if (!wrapper.bitmap || wrapper.video_ctx) {
                std::fprintf(stderr, "multimodal input is not a supported still image\n");
                return 1;
            }
            const char * marker = mtmd_get_marker(engine->vision_ctx);
            mtmd_input_text text = {marker, std::strlen(marker), false, true};
            mtmd_input_chunks * chunks = mtmd_input_chunks_init();
            const mtmd_bitmap * bitmaps[] = {wrapper.bitmap};
            if (!chunks || mtmd_tokenize(engine->vision_ctx, chunks, &text, bitmaps, 1) != 0) {
                std::fprintf(stderr, "multimodal image tokenization failed\n");
                return 1;
            }
            const mtmd_input_chunk * image_chunk = nullptr;
            for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
                const mtmd_input_chunk * candidate = mtmd_input_chunks_get(chunks, i);
                if (mtmd_input_chunk_get_type(candidate) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
                    if (image_chunk) {
                        std::fprintf(stderr, "one still image expanded to multiple image chunks\n");
                        return 1;
                    }
                    image_chunk = candidate;
                }
            }
            if (!image_chunk ||
                (int) mtmd_input_chunk_get_n_tokens(image_chunk) != segment.projected_tokens) {
                std::fprintf(stderr,
                    "projected image row count differs from the authenticated request\n");
                return 1;
            }
            if (mtmd_encode_chunk(engine->vision_ctx, image_chunk) != 0) {
                std::fprintf(stderr, "multimodal image encoder failed\n");
                return 1;
            }
            float * encoded = mtmd_get_output_embd(engine->vision_ctx);
            llama_pos new_n_past = n_past;
            const int rc = encoded ? mtmd_helper_decode_image_chunk(
                engine->vision_ctx, session.ctx, image_chunk, encoded, n_past, 0,
                prefill_step,
                &new_n_past,
                (session.draft_ctx || live) ? inject_dflash_batch : nullptr,
                (session.draft_ctx || live) ? &dflash_injector : nullptr) : 1;
            if (rc != 0 || new_n_past != n_past + segment.projected_tokens) {
                std::fprintf(stderr, "multimodal image decode failed rc=%d at %d\n", rc, n_past);
                return 1;
            }
            n_past = new_n_past;
            mtmd_input_chunks_free(chunks);
            mtmd_bitmap_free(wrapper.bitmap);
        }
        if (n_past != n_prefill) {
            std::fprintf(stderr, "multimodal plan produced %d positions, expected %d\n",
                n_past, n_prefill);
            return 1;
        }
    }

    const int64_t prefill_end = now_ms();
    std::fprintf(session.markers, "prefill_compute_end_epoch_ms %lld\n", (long long) prefill_end);
    std::fflush(session.markers);
    if (live) {
        // Prefill is over, so one drain covers every tile still owed.
        if (next_nope_start < (uint64_t) n_prefill) {
            llama_synchronize(session.ctx);
        }
        while (next_nope_start < (uint64_t) n_prefill) {
            const uint32_t count = (uint32_t) std::min<uint64_t>(
                512, (uint64_t) n_prefill - next_nope_start);
            if (stream_nope_tile(session.ctx, live, next_nope_start, count, true) != 0) {
                return 1;
            }
            if (stream_swa_for_completed_tile(
                    session.ctx, live, (uint64_t) n_prefill, next_nope_start, count) != 0) {
                return 1;
            }
            next_nope_start += count;
        }
    }

    if (!job.out_path.empty()) {
        if (!llama_state_save_file(
                session.ctx, job.out_path.c_str(), tokens.data(), (size_t) n_prefill)) {
            std::fprintf(stderr, "llama_state_save_file failed\n");
            return 1;
        }
        const int64_t saved = now_ms();
        std::fprintf(session.markers, "state_saved_epoch_ms %lld\n", (long long) saved);
        std::fprintf(session.markers,
            "[spark-kv-export] prefill_tokens %d prefill_seconds %.3f state_save_seconds %.3f\n",
            n_prefill, (prefill_end - prefill_start) / 1000.0, (saved - prefill_end) / 1000.0);
        std::fflush(session.markers);
    }
    if (session.draft_ctx && !llama_state_save_file(
            session.draft_ctx, job.draft_out_path.c_str(), tokens.data(), (size_t) n_prefill)) {
        std::fprintf(stderr, "DFlash llama_state_save_file failed\n");
        return 1;
    }
    if (live && kv_stream.close() != 0) {
        return 1;
    }
    const int64_t export_complete = now_ms();
    std::fprintf(session.markers, "export_complete_epoch_ms %lld\n", (long long) export_complete);
    std::fflush(session.markers);

    if (job.dflash_features_path.empty() && n_prefill < (int) tokens.size() &&
        (live == nullptr || dump_requested(engine))) {
        if (engine->dump_state) {
            engine->dump_state->in_decode = true;
            if (engine->dump_state->decode_only) {
                std::fprintf(stderr, "[spark-kv-export] tensor dumps armed for held-token decode\n");
            }
        }
        session.batch.n_tokens = 1;
        session.batch.token[0]    = tokens[n_prefill];
        session.batch.pos[0]      = n_prefill;
        session.batch.n_seq_id[0] = 1;
        session.batch.seq_id[0][0] = 0;
        session.batch.logits[0]   = 1;
        if (llama_decode(session.ctx, session.batch) != 0) {
            std::fprintf(stderr, "probe decode failed\n");
        } else {
            const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(engine->model));
            const float * logits = llama_get_logits_ith(session.ctx, 0);
            std::vector<int> idx(n_vocab);
            for (int v = 0; v < n_vocab; ++v) idx[v] = v;
            const int top = n_vocab < 8 ? n_vocab : 8;
            std::partial_sort(idx.begin(), idx.begin() + top, idx.end(),
                [&](int a, int b) { return logits[a] > logits[b]; });
            std::fprintf(session.markers, "probe_top8");
            for (int t = 0; t < top; ++t) {
                std::fprintf(session.markers, " %d:%.4f", idx[t], logits[idx[t]]);
            }
            std::fprintf(session.markers, "\n");
            std::fflush(session.markers);
        }
    }

    if (engine->dump_state) {
        for (const TensorDump & dump : engine->dump_state->dumps) {
            if (!dump.written) {
                std::fprintf(stderr, "requested tensor dump was not produced: %s\n",
                    dump.name.c_str());
                return 1;
            }
        }
    }
    if (!dump_requested(engine) && !engine->cpu_math_oracle_layer0 && job.draft_out_path.empty()) {
        discard_cached_context(engine);
        engine->cached_ctx = session.ctx;
        engine->cached_n_ctx = ctx_n_ctx;
        engine->cached_n_batch = job.n_batch;
        engine->cached_n_ubatch = job.n_ubatch;
        engine->cached_flash_attn = job.flash_attn;
        session.keep_ctx = true;
    }
    return 0;
}

static void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model M.gguf --tokens F.tokens [--out S.bin]\n"
        "          [--n-ctx 32768] [--n-batch 2048] [--n-ubatch 512]\n"
        "          [--flash-attn 1] [--threads 4] [--skip-tail 1]\n"
        "          [--serve-jobs FIFO]\n"
        "          [--tensor-cpu REGEX]...\n"
        "          [--dump-tensor-f32 NAME=PATH]... [--dump-decode-only]\n"
        "          [--cuda-cpu-order-q4k-tensor NAME]\n"
        "          [--cuda-cpu-order-q6k-tensor NAME]\n"
        "          [--cuda-cpu-order-q5k-tensor NAME]\n"
        "          [--cuda-cpu-order-qk-prefix PREFIX]\n"
        "          [--cuda-cpu-order-rope]\n"
        "          [--cuda-cpu-order-attn]\n"
        "          [--cuda-cpu-order-nonlearned]\n"
        "          [--cuda-cpu-order-rms]\n"
        "          [--cpu-math-oracle-layer0]\n"
        "          [--cuda-metal-compat]\n"
        "          [--cuda-metal-compatible-full]\n"
        "          [--draft-model D.gguf --draft-out D.session]\n"
        "          [--dflash-features TARGET.f32]\n"
        "          [--mmproj P.gguf --multimodal-plan P.tsv]\n"
        "          [--nope-fifo PATH]\n",
        argv0);
}

int main(int argc, char ** argv) {
    // A dropped Mac connection closes the NoPE FIFO read end out from under
    // KvStream::run's fwrite loop. Default SIGPIPE disposition would kill the
    // whole process (and the warm engine with it); take the EPIPE as a normal
    // write error instead and let write_status() record a per-job "fail".
    std::signal(SIGPIPE, SIG_IGN);
    std::string model_path, tokens_path, out_path;
    std::string draft_model_path, draft_out_path;
    std::string dflash_features_path;
    std::string mmproj_path, multimodal_plan_path;
    std::string load_path;
    std::string nope_fifo_path;
    std::string serve_jobs_path;
    std::vector<std::string> tensor_cpu_patterns;
    DumpState dump_state;
    std::string cuda_cpu_order_q4k_tensor;
    std::string cuda_cpu_order_q6k_tensor;
    std::string cuda_cpu_order_q5k_tensor;
    std::string cuda_cpu_order_qk_prefix;
    bool cpu_math_oracle_layer0 = false;
    bool cuda_metal_compat = false;
    bool cuda_cpu_order_rope = false;
    bool cuda_cpu_order_attn = false;
    bool cuda_cpu_order_nonlearned = false;
    bool cuda_cpu_order_rms = false;
    bool cuda_metal_compatible_full = false;
    int n_ctx = 32768;
    int n_batch = 2048;
    int n_ubatch = 512;
    int flash_attn = 1;
    int threads = 4;
    int skip_tail = 1;  // cache tokens [0, N-skip_tail): the consumer holds the last one

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char * name) -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", name); std::exit(2); }
            return argv[++i];
        };
        if (a == "--model")            model_path  = next("--model");
        else if (a == "--tokens")      tokens_path = next("--tokens");
        else if (a == "--out")         out_path    = next("--out");
        else if (a == "--draft-model") draft_model_path = next("--draft-model");
        else if (a == "--draft-out")   draft_out_path = next("--draft-out");
        else if (a == "--dflash-features") {
            dflash_features_path = next("--dflash-features");
        }
        else if (a == "--mmproj")      mmproj_path = next("--mmproj");
        else if (a == "--multimodal-plan") multimodal_plan_path = next("--multimodal-plan");
        else if (a == "--load-session") load_path  = next("--load-session");
        else if (a == "--nope-fifo") nope_fifo_path = next("--nope-fifo");
        else if (a == "--serve-jobs") serve_jobs_path = next("--serve-jobs");
        else if (a == "--tensor-cpu") tensor_cpu_patterns.push_back(next("--tensor-cpu"));
        else if (a == "--dump-tensor-f32") {
            std::string value = next("--dump-tensor-f32");
            const size_t split = value.find('=');
            if (split == std::string::npos || split == 0 || split + 1 == value.size()) {
                std::fprintf(stderr, "--dump-tensor-f32 expects NAME=PATH\n");
                return 2;
            }
            dump_state.dumps.push_back({value.substr(0, split), value.substr(split + 1), false});
        }
        else if (a == "--dump-decode-only") dump_state.decode_only = true;
        else if (a == "--cuda-cpu-order-q4k-tensor") {
            cuda_cpu_order_q4k_tensor = next("--cuda-cpu-order-q4k-tensor");
        }
        else if (a == "--cuda-cpu-order-q6k-tensor") {
            cuda_cpu_order_q6k_tensor = next("--cuda-cpu-order-q6k-tensor");
        }
        else if (a == "--cuda-cpu-order-q5k-tensor") {
            cuda_cpu_order_q5k_tensor = next("--cuda-cpu-order-q5k-tensor");
        }
        else if (a == "--cuda-cpu-order-qk-prefix") {
            cuda_cpu_order_qk_prefix = next("--cuda-cpu-order-qk-prefix");
        }
        else if (a == "--cpu-math-oracle-layer0") cpu_math_oracle_layer0 = true;
        else if (a == "--cuda-metal-compat") cuda_metal_compat = true;
        else if (a == "--cuda-metal-compatible-full") {
            cuda_metal_compatible_full = true;
            cuda_metal_compat = true;
            cuda_cpu_order_rope = true;
            cuda_cpu_order_attn = true;
            cuda_cpu_order_nonlearned = true;
            cuda_cpu_order_rms = true;
        }
        else if (a == "--cuda-cpu-order-rope") cuda_cpu_order_rope = true;
        else if (a == "--cuda-cpu-order-attn") cuda_cpu_order_attn = true;
        else if (a == "--cuda-cpu-order-nonlearned") cuda_cpu_order_nonlearned = true;
        else if (a == "--cuda-cpu-order-rms") cuda_cpu_order_rms = true;
        else if (a == "--n-ctx")       n_ctx       = std::stoi(next("--n-ctx"));
        else if (a == "--n-batch")     n_batch     = std::stoi(next("--n-batch"));
        else if (a == "--n-ubatch")    n_ubatch    = std::stoi(next("--n-ubatch"));
        else if (a == "--flash-attn")  flash_attn  = std::stoi(next("--flash-attn"));
        else if (a == "--threads")     threads     = std::stoi(next("--threads"));
        else if (a == "--skip-tail")   skip_tail   = std::stoi(next("--skip-tail"));
        else { usage(argv[0]); return 2; }
    }
    const bool serving = !serve_jobs_path.empty();
    if (model_path.empty() ||
        (serving
             ? (!tokens_path.empty() || !out_path.empty() || !load_path.empty() ||
                !nope_fifo_path.empty() || !draft_out_path.empty() ||
                !dflash_features_path.empty() || !multimodal_plan_path.empty() ||
                !dump_state.dumps.empty())
             : (tokens_path.empty() ||
                (out_path.empty() && load_path.empty() && nope_fifo_path.empty() &&
                 draft_out_path.empty())))) {
        usage(argv[0]);
        return 2;
    }
    if (!serving && !dflash_features_path.empty() &&
        (draft_model_path.empty() || !out_path.empty() || !load_path.empty() ||
         !nope_fifo_path.empty() || !multimodal_plan_path.empty())) {
        std::fprintf(stderr,
            "--dflash-features requires the draft pair and cannot combine with target outputs\n");
        return 2;
    }
    if (!serving && (draft_model_path.empty() != draft_out_path.empty() ||
        (!draft_model_path.empty() && !load_path.empty()))) {
        std::fprintf(stderr,
            "--draft-model and --draft-out are a pair and cannot be used with --load-session\n");
        return 2;
    }
    if (!serving && (mmproj_path.empty() != multimodal_plan_path.empty() ||
        (!mmproj_path.empty() && !load_path.empty()))) {
        std::fprintf(stderr,
            "--mmproj and --multimodal-plan are a pair and cannot combine with load mode\n");
        return 2;
    }
    if (!nope_fifo_path.empty() && !load_path.empty()) {
        std::fprintf(stderr, "--nope-fifo cannot combine with --load-session\n");
        return 2;
    }
    if (std::any_of(tensor_cpu_patterns.begin(), tensor_cpu_patterns.end(),
                    [](const std::string & pattern) { return pattern.empty(); })) {
        std::fprintf(stderr, "--tensor-cpu patterns cannot be empty\n");
        return 2;
    }
    if (cpu_math_oracle_layer0 && cuda_metal_compat) {
        std::fprintf(stderr,
            "--cpu-math-oracle-layer0 cannot combine with --cuda-metal-compat\n");
        return 2;
    }
    if (cpu_math_oracle_layer0 && cuda_cpu_order_attn) {
        std::fprintf(stderr,
            "--cpu-math-oracle-layer0 cannot combine with --cuda-cpu-order-attn\n");
        return 2;
    }
    if (cpu_math_oracle_layer0 && cuda_cpu_order_nonlearned) {
        std::fprintf(stderr,
            "--cpu-math-oracle-layer0 cannot combine with --cuda-cpu-order-nonlearned\n");
        return 2;
    }
    if (cpu_math_oracle_layer0 && cuda_cpu_order_rms) {
        std::fprintf(stderr,
            "--cpu-math-oracle-layer0 cannot combine with --cuda-cpu-order-rms\n");
        return 2;
    }
    if (cpu_math_oracle_layer0) {
        // Keep the embedding lookup and every learned layer-0 operation on the
        // same host backend.  Combined with offload_kqv=false below and strict
        // host-op routing, this is a correctness oracle for the first complete
        // layer boundary.  It is deliberately not a production/performance
        // route; the CUDA compatibility kernels are qualified against it.
        tensor_cpu_patterns.push_back("^token_embd\\.weight$");
        tensor_cpu_patterns.push_back("^blk\\.0\\..*$");
    }

    std::vector<llama_token> tokens;
    std::vector<MultimodalSegment> multimodal_plan;
    if (!serving) {
        try {
            tokens = read_tokens(tokens_path);
            if (!multimodal_plan_path.empty()) {
                multimodal_plan = read_multimodal_plan(multimodal_plan_path);
            }
        } catch (const std::exception & error) {
            std::fprintf(stderr, "%s\n", error.what());
            return 2;
        }
        const int n_prefill = (int) tokens.size() - skip_tail;
        if (n_prefill < 1 || n_prefill > n_ctx) {
            std::fprintf(stderr, "n_prefill %d out of range (tokens=%zu, n_ctx=%d)\n",
                n_prefill, tokens.size(), n_ctx);
            return 2;
        }
        std::fprintf(stderr, "[spark-kv-export] model=%s tokens=%zu prefill=%d n_ctx=%d fa=%d\n",
            model_path.c_str(), tokens.size(), n_prefill, n_ctx, flash_attn);
        (void) multimodal_plan;
    } else {
        std::fprintf(stderr, "[spark-kv-export] model=%s serving jobs from %s\n",
            model_path.c_str(), serve_jobs_path.c_str());
    }

    // On an integrated CUDA device llama.cpp may choose CUDA_Host as the
    // concrete buffer for a CPU tensor override.  The scheduler normally
    // re-offloads operations backed by that host buffer when the batch has at
    // least GGML_OP_OFFLOAD_MIN_BATCH rows (32 by default).  That makes a
    // seemingly CPU-pinned projection execute on CUDA for realistic prefills,
    // even though the same two-token debug probe executes on CPU.  Keep this
    // diagnostic override strict so its result does not depend on batch size.
    if (cuda_metal_compat) {
        setenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE", "metal", 1);
        setenv("MUSER_CUDA_METAL_COMPAT", "1", 1);
        std::fprintf(stderr,
            "[spark-kv-export] Metal-compatible CUDA matmul: "
            "f16 operands with f32 accumulation/output\n");
    }
    if (cuda_metal_compatible_full) {
        // Empty prefix deliberately selects every Q4_K/Q5_K/Q6_K learned
        // projection. Together with the compatible nonlearned kernels this
        // is the production full-logit route, not a one-tensor diagnostic.
        setenv("MUSER_CUDA_CPU_ORDER_QK_PREFIX", "", 1);
        const std::string rope_nctx = std::to_string(n_ctx);
        setenv("MUSER_CUDA_CPU_ORDER_ROPE", rope_nctx.c_str(), 1);
        setenv("MUSER_CUDA_CPU_ORDER_ATTN", "1", 1);
        setenv("MUSER_CUDA_CPU_ORDER_NONLEARNED", "1", 1);
        setenv("MUSER_CUDA_CPU_ORDER_RMS", "1", 1);
        // Cross-vendor qualification uses the integer-Q8 projection kernels
        // on both Metal and CUDA.  Leaving this variable unset selects the
        // existing CPU-order CUDA kernels, whose reductions are integer-exact.
        if (!getenv("MUSER_CROSS_VENDOR_QK")) {
            setenv("MUSER_CUDA_METAL_ORDER", "1", 1);
        }
        setenv("GGML_CUDA_DISABLE_FUSION", "1", 1);
        setenv("MUSER_CUDA_METAL_COMPAT_STRICT", "1", 1);
        std::fprintf(stderr,
            "[spark-kv-export] strict full Metal-compatible CUDA math enabled for all layers\n");
    }
    if (!cuda_cpu_order_q4k_tensor.empty()) {
        setenv("MUSER_CUDA_CPU_ORDER_Q4K_TENSOR", cuda_cpu_order_q4k_tensor.c_str(), 1);
        std::fprintf(stderr, "[spark-kv-export] CUDA CPU-order Q4_K tensor %s\n",
            cuda_cpu_order_q4k_tensor.c_str());
    }
    if (!cuda_cpu_order_q6k_tensor.empty()) {
        setenv("MUSER_CUDA_CPU_ORDER_Q6K_TENSOR", cuda_cpu_order_q6k_tensor.c_str(), 1);
        std::fprintf(stderr, "[spark-kv-export] CUDA CPU-order Q6_K tensor %s\n",
            cuda_cpu_order_q6k_tensor.c_str());
    }
    if (!cuda_cpu_order_q5k_tensor.empty()) {
        setenv("MUSER_CUDA_CPU_ORDER_Q5K_TENSOR", cuda_cpu_order_q5k_tensor.c_str(), 1);
        std::fprintf(stderr, "[spark-kv-export] CUDA CPU-order Q5_K tensor %s\n",
            cuda_cpu_order_q5k_tensor.c_str());
    }
    if (!cuda_cpu_order_qk_prefix.empty()) {
        setenv("MUSER_CUDA_CPU_ORDER_QK_PREFIX", cuda_cpu_order_qk_prefix.c_str(), 1);
        std::fprintf(stderr, "[spark-kv-export] CUDA CPU-order QK prefix %s\n",
            cuda_cpu_order_qk_prefix.c_str());
    }
    if (cuda_cpu_order_rope) {
        const std::string rope_nctx = std::to_string(n_ctx);
        setenv("MUSER_CUDA_CPU_ORDER_ROPE", rope_nctx.c_str(), 1);
        std::fprintf(stderr, "[spark-kv-export] CUDA CPU-order RoPE enabled\n");
    }
    if (cuda_cpu_order_attn) {
        setenv("MUSER_CUDA_CPU_ORDER_ATTN", "1", 1);
        setenv("GGML_CUDA_DISABLE_FUSION", "1", 1);
        std::fprintf(stderr, "[spark-kv-export] CUDA CPU-order attention enabled\n");
    }
    if (cuda_cpu_order_nonlearned) {
        setenv("MUSER_CUDA_CPU_ORDER_NONLEARNED", "1", 1);
        setenv("GGML_CUDA_DISABLE_FUSION", "1", 1);
        std::fprintf(stderr, "[spark-kv-export] CUDA CPU-order non-learned ops enabled\n");
    }
    if (cuda_cpu_order_rms) {
        setenv("MUSER_CUDA_CPU_ORDER_RMS", "1", 1);
        setenv("GGML_CUDA_DISABLE_FUSION", "1", 1);
        std::fprintf(stderr, "[spark-kv-export] CUDA CPU-order 1e-8 RMS enabled\n");
    }
    if (!tensor_cpu_patterns.empty()) {
        setenv("GGML_OP_OFFLOAD_MIN_BATCH", "2147483647", 1);
        std::fprintf(stderr,
            "[spark-kv-export] strict CPU tensor routing: "
            "GGML_OP_OFFLOAD_MIN_BATCH=2147483647\n");
    }
    if (cpu_math_oracle_layer0) {
        std::fprintf(stderr,
            "[spark-kv-export] layer-0 CPU math oracle enabled: "
            "embedding, learned layer-0 ops, attention, and KV stay on host\n");
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;
    if (!tensor_cpu_patterns.empty()) {
        // llama.cpp treats a CPU override as "pick from the CPU buft list",
        // which prefers CUDA_Host on UMA.  That keeps Q6_K matmuls on the
        // default CUDA kernel.  Disable host/extra buffers and GPU layers so
        // --tensor-cpu is a real ARM vec_dot oracle.
        mparams.n_gpu_layers = 0;
        mparams.no_host = true;
        mparams.use_extra_bufts = false;
        std::fprintf(stderr,
            "[spark-kv-export] true CPU oracle: n_gpu_layers=0 no_host=1 "
            "use_extra_bufts=0\n");
    }
    std::vector<llama_model_tensor_buft_override> tensor_cpu_overrides;
    tensor_cpu_overrides.reserve(tensor_cpu_patterns.size() + 1);
    for (const std::string & pattern : tensor_cpu_patterns) {
        tensor_cpu_overrides.push_back({pattern.c_str(), ggml_backend_cpu_buffer_type()});
        std::fprintf(stderr, "[spark-kv-export] target tensor override %s=CPU\n",
            pattern.c_str());
    }
    if (!tensor_cpu_overrides.empty()) {
        tensor_cpu_overrides.push_back({nullptr, nullptr});
        mparams.tensor_buft_overrides = tensor_cpu_overrides.data();
    }
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { std::fprintf(stderr, "model load failed\n"); return 1; }
    // Overrides qualify target-producer arithmetic only. The optional DFlash
    // artifact has a separate model identity and must retain its pinned CUDA
    // route unless it is explicitly given its own compatibility campaign.
    mparams.tensor_buft_overrides = nullptr;

    Engine engine;
    engine.model = model;
    engine.dump_state = &dump_state;
    engine.cpu_math_oracle_layer0 = cpu_math_oracle_layer0;

    if (!mmproj_path.empty()) {
        mtmd_context_params vparams = mtmd_context_params_default();
        engine.vision_ctx = mtmd_init_from_file(mmproj_path.c_str(), model, vparams);
        if (!engine.vision_ctx || !mtmd_support_vision(engine.vision_ctx)) {
            std::fprintf(stderr, "multimodal projector load failed or lacks vision support\n");
            return 1;
        }
    }

    if (!draft_model_path.empty()) {
        engine.draft_model = llama_model_load_from_file(draft_model_path.c_str(), mparams);
        if (!engine.draft_model) {
            std::fprintf(stderr, "DFlash model load failed\n");
            return 1;
        }
        engine.target_layer_ids = llama_model_target_layer_ids(engine.draft_model);
        engine.target_layer_ids_n = llama_model_target_layer_ids_n(engine.draft_model);
        if (!engine.target_layer_ids || engine.target_layer_ids_n != 5) {
            std::fprintf(stderr, "DFlash model must identify exactly five target layers\n");
            return 1;
        }
        engine.target_hidden = llama_model_n_embd(model);
        engine.draft_hidden = llama_model_n_embd(engine.draft_model);
        if (engine.target_hidden <= 0 || engine.draft_hidden != engine.target_hidden) {
            std::fprintf(stderr,
                "DFlash/target hidden dimensions differ (%d vs %d)\n",
                engine.draft_hidden, engine.target_hidden);
            return 1;
        }
        for (uint32_t k = 0; k < engine.target_layer_ids_n; ++k) {
            if (engine.target_layer_ids[k] < 0) {
                std::fprintf(stderr, "DFlash target layer id is negative\n");
                return 1;
            }
        }
        std::fprintf(stderr,
            "[spark-kv-export] combined DFlash model loaded (%u target layers)\n",
            engine.target_layer_ids_n);
    }

    int rc = 0;
    if (serving) {
        rc = warmup_cuda(&engine);
        if (rc == 0) {
            rc = serve_jobs(&engine, serve_jobs_path);
        }
    } else {
        JobRequest job;
        job.tokens_path = tokens_path;
        job.out_path = out_path;
        job.load_path = load_path;
        job.nope_fifo_path = nope_fifo_path;
        job.draft_out_path = draft_out_path;
        job.dflash_features_path = dflash_features_path;
        job.multimodal_plan_path = multimodal_plan_path;
        job.n_ctx = n_ctx;
        job.n_batch = n_batch;
        job.n_ubatch = n_ubatch;
        job.flash_attn = flash_attn;
        job.threads = threads;
        job.skip_tail = skip_tail;
        rc = run_job(&engine, job);
    }

    if (engine.cached_ctx) llama_free(engine.cached_ctx);
    if (engine.vision_ctx) mtmd_free(engine.vision_ctx);
    if (engine.draft_model) llama_model_free(engine.draft_model);
    llama_model_free(model);
    llama_backend_free();
    return rc;
}
