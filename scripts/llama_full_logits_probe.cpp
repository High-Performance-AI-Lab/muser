#include "ggml-backend.h"
#include "llama.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <sys/stat.h>

namespace {

struct args {
    std::string model;
    std::string tokens;
    std::string output;
    std::string capture_dir;
    std::string capture_name;
    int32_t teacher_token = -1;
    int32_t capture_layer = -1;
    int32_t prompt_positions = 2048;
};

bool parse_i32(const char * text, int32_t & value) {
    char * end = nullptr;
    errno = 0;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        parsed < std::numeric_limits<int32_t>::min() ||
        parsed > std::numeric_limits<int32_t>::max()) {
        return false;
    }
    value = static_cast<int32_t>(parsed);
    return true;
}

bool parse_args(int argc, char ** argv, args & result) {
    for (int index = 1; index < argc; ++index) {
        const std::string name = argv[index];
        if (index + 1 >= argc) {
            std::fprintf(stderr, "%s requires a value\n", name.c_str());
            return false;
        }
        const char * value = argv[++index];
        if (name == "--model") {
            result.model = value;
        } else if (name == "--tokens") {
            result.tokens = value;
        } else if (name == "--teacher-token") {
            if (!parse_i32(value, result.teacher_token)) {
                std::fprintf(stderr, "--teacher-token must be int32\n");
                return false;
            }
        } else if (name == "--output") {
            result.output = value;
        } else if (name == "--capture-dir") {
            result.capture_dir = value;
        } else if (name == "--capture-layer") {
            if (!parse_i32(value, result.capture_layer) || result.capture_layer < 0) {
                std::fprintf(stderr, "--capture-layer must be non-negative int32\n");
                return false;
            }
        } else if (name == "--prompt-positions") {
            if (!parse_i32(value, result.prompt_positions) || result.prompt_positions < 1) {
                std::fprintf(stderr, "--prompt-positions must be a positive int32\n");
                return false;
            }
        } else if (name == "--capture-name") {
            result.capture_name = value;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", name.c_str());
            return false;
        }
    }
    if (result.model.empty() || result.tokens.empty() || result.output.empty() ||
        result.teacher_token < 0) {
        std::fprintf(stderr,
            "required: --model PATH --tokens PATH --teacher-token ID --output PATH\n");
        return false;
    }
    if (result.capture_dir.empty() != (result.capture_layer < 0)) {
        std::fprintf(stderr, "--capture-dir and --capture-layer must be supplied together\n");
        return false;
    }
    return true;
}

struct capture_state {
    std::string directory;
    std::string layer_suffix;
    std::string only;
    bool enabled = false;
    bool teacher = false;
    bool failed = false;
    size_t files = 0;
    std::map<std::string, size_t> prompt_counts;
};

bool capture_wanted(const char * name, const capture_state & state) {
    const std::string value = name;
    if (!state.only.empty()) {
        if (state.only == "kv-*") {
            return !state.teacher &&
                (value.compare(0, 12, "Kcur_normed-") == 0 ||
                 value.compare(0, 10, "Kcur_rope-") == 0 ||
                 value.compare(0, 5, "Vcur-") == 0);
        }
        return state.teacher && (value == state.only ||
            (state.only == "l_out-*" && value.compare(0, 6, "l_out-") == 0));
    }
    if (!state.teacher) {
        return value == "Kcur" + state.layer_suffix ||
            value == "Kcur_normed" + state.layer_suffix ||
            value == "Kcur_rope" + state.layer_suffix ||
            value == "Vcur" + state.layer_suffix;
    }
    if (value == "result_norm" || value == "result_output") {
        return true;
    }
    static const char * labels[] = {
        "attn_norm", "Qcur", "Kcur", "Vcur", "attn_gate_proj",
        "Qcur_normed", "Kcur_normed", "Qcur_rope", "Kcur_rope",
        "attn_out", "attn_gate_sig", "attn_gated", "attn_o_proj",
        "attn_post_norm", "ffn_inp", "ffn_norm", "ffn_gate", "ffn_up",
        "ffn_swiglu", "ffn_out", "ffn_post_norm", "l_out",
    };
    for (const char * label : labels) {
        if (value == std::string(label) + state.layer_suffix) {
            return true;
        }
    }
    return false;
}

bool capture_eval(struct ggml_tensor * tensor, bool ask, void * user_data) {
    auto & state = *static_cast<capture_state *>(user_data);
    if (!state.enabled || !capture_wanted(tensor->name, state)) {
        return false;
    }
    if (ask) {
        return true;
    }
    if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor)) {
        std::fprintf(stderr, "capture tensor %s is not contiguous f32\n", tensor->name);
        state.failed = true;
        return true;
    }
    const size_t bytes = ggml_nbytes(tensor);
    std::vector<uint8_t> data(bytes);
    ggml_backend_tensor_get(tensor, data.data(), 0, bytes);
    std::string filename = std::string(tensor->name) + ".f32";
    if (!state.teacher) {
        const size_t ordinal = state.prompt_counts[tensor->name]++;
        filename = "prompt." + std::to_string(ordinal) + "." + filename;
    }
    const std::string path = state.directory + "/" + filename;
    if (std::FILE * existing = std::fopen(path.c_str(), "rb")) {
        std::fclose(existing);
        if (!state.teacher) {
            std::fprintf(stderr, "refusing duplicate prompt capture %s\n", tensor->name);
            state.failed = true;
        }
        return true;
    }
    std::FILE * output = std::fopen(path.c_str(), "wbx");
    bool written = false;
    if (output != nullptr) {
        const bool complete = std::fwrite(data.data(), 1, data.size(), output) == data.size();
        const bool flushed = std::fflush(output) == 0;
        const bool closed = std::fclose(output) == 0;
        written = complete && flushed && closed;
        output = nullptr;
    }
    if (!written) {
        std::fprintf(stderr, "failed to write capture tensor %s\n", tensor->name);
        state.failed = true;
    } else {
        ++state.files;
    }
    return true;
}

bool load_tokens(const std::string & path, std::vector<llama_token> & tokens) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        std::fprintf(stderr, "cannot open token fixture: %s\n", path.c_str());
        return false;
    }
    std::string text((std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>());
    if (text.empty() || text.size() > 2 * 1024 * 1024) {
        std::fprintf(stderr, "token fixture has an invalid text length\n");
        return false;
    }
    for (char & value : text) {
        if (value == ',') {
            value = ' ';
        }
    }
    std::istringstream stream(text);
    uint64_t value = 0;
    while (stream >> value) {
        if (value > static_cast<uint64_t>(std::numeric_limits<llama_token>::max())) {
            std::fprintf(stderr, "token fixture contains an out-of-range ID\n");
            return false;
        }
        tokens.push_back(static_cast<llama_token>(value));
    }
    if (!stream.eof()) {
        std::fprintf(stderr, "token fixture contains a non-numeric field\n");
        return false;
    }
    return !tokens.empty();
}

bool write_exclusive(const std::string & path, const float * logits, size_t count) {
    if (std::FILE * existing = std::fopen(path.c_str(), "rb")) {
        std::fclose(existing);
        std::fprintf(stderr, "refusing to replace output: %s\n", path.c_str());
        return false;
    }
    std::FILE * output = std::fopen(path.c_str(), "wbx");
    if (output == nullptr) {
        std::fprintf(stderr, "cannot create output %s: %s\n",
            path.c_str(), std::strerror(errno));
        return false;
    }
    const bool ok = std::fwrite(logits, sizeof(float), count, output) == count &&
        std::fflush(output) == 0 && std::fclose(output) == 0;
    if (!ok) {
        std::fprintf(stderr, "failed to publish complete logits output\n");
    }
    return ok;
}

} // namespace

int main(int argc, char ** argv) {
    args options;
    if (!parse_args(argc, argv, options)) {
        return 2;
    }
    std::vector<llama_token> tokens;
    if (!load_tokens(options.tokens, tokens) ||
        tokens.size() != static_cast<size_t>(options.prompt_positions)) {
        std::fprintf(stderr, "token fixture does not match --prompt-positions\n");
        return 2;
    }

    ggml_backend_load_all();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(options.model.c_str(), model_params);
    if (model == nullptr) {
        std::fprintf(stderr, "failed to load model\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t vocab_size = llama_vocab_n_tokens(vocab);
    for (size_t index = 0; index < tokens.size(); ++index) {
        if (tokens[index] < 0 || tokens[index] >= vocab_size) {
            std::fprintf(stderr, "prompt token %zu is outside the vocabulary\n", index);
            llama_model_free(model);
            return 2;
        }
    }
    if (options.teacher_token >= vocab_size) {
        std::fprintf(stderr, "teacher token is outside the vocabulary\n");
        llama_model_free(model);
        return 2;
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = options.prompt_positions + 1;
    context_params.n_batch = 2048;
    context_params.n_ubatch = 512;
    context_params.n_seq_max = 1;
    context_params.n_threads = 20;
    context_params.n_threads_batch = 20;
    context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    context_params.type_k = GGML_TYPE_F16;
    context_params.type_v = GGML_TYPE_F16;
    context_params.no_perf = false;
    capture_state capture;
    if (!options.capture_dir.empty()) {
        struct stat directory_stat;
        if (lstat(options.capture_dir.c_str(), &directory_stat) != 0 ||
            !S_ISDIR(directory_stat.st_mode)) {
            std::fprintf(stderr, "capture directory must be a real directory\n");
            llama_model_free(model);
            return 2;
        }
        capture.directory = options.capture_dir;
        capture.layer_suffix = "-" + std::to_string(options.capture_layer);
        capture.only = options.capture_name;
        capture.enabled = true;
        context_params.cb_eval = capture_eval;
        context_params.cb_eval_user_data = &capture;
    }
    llama_context * context = llama_init_from_model(model, context_params);
    if (context == nullptr) {
        std::fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    if (llama_decode(context, llama_batch_get_one(tokens.data(), tokens.size())) != 0) {
        std::fprintf(stderr, "prompt decode failed\n");
        llama_free(context);
        llama_model_free(model);
        return 1;
    }
    llama_token teacher = options.teacher_token;
    capture.teacher = true;
    if (llama_decode(context, llama_batch_get_one(&teacher, 1)) != 0) {
        std::fprintf(stderr, "teacher-token decode failed\n");
        llama_free(context);
        llama_model_free(model);
        return 1;
    }
    if (capture.failed || (!options.capture_dir.empty() && capture.files == 0)) {
        std::fprintf(stderr, "teacher-token boundary capture failed\n");
        llama_free(context);
        llama_model_free(model);
        return 1;
    }
    const float * logits = llama_get_logits(context);
    if (logits == nullptr || !write_exclusive(options.output, logits, vocab_size)) {
        llama_free(context);
        llama_model_free(model);
        return 1;
    }

    std::printf(
        "{\"schema\":\"muser.llama-full-logits-probe.v1\","
        "\"prompt_tokens\":%d,\"teacher_token\":%d,\"vocab_size\":%d,"
        "\"context\":%d,\"batch\":2048,\"ubatch\":512,"
        "\"gpu_layers\":99,\"flash_attention\":true,\"kv\":\"f16\","
        "\"capture_layer\":%d,\"capture_files\":%zu,"
        "\"output\":\"%s\"}\n",
        options.prompt_positions, options.teacher_token, vocab_size,
        options.prompt_positions + 1, options.capture_layer, capture.files,
        options.output.c_str());
    llama_perf_context_print(context);
    llama_free(context);
    llama_model_free(model);
    return 0;
}
