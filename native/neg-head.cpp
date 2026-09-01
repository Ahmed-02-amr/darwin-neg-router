#include "neg-head.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <queue>
#include <utility>

namespace {

template <typename T>
bool read_value(std::ifstream & input, T & value) {
    return static_cast<bool>(input.read(reinterpret_cast<char *>(&value), sizeof(T)));
}

bool read_floats(std::ifstream & input, std::vector<float> & values, size_t count) {
    values.resize(count);
    return static_cast<bool>(input.read(
        reinterpret_cast<char *>(values.data()),
        static_cast<std::streamsize>(count * sizeof(float))));
}

float gelu_exact(float value) {
    constexpr float inv_sqrt_two = 0.70710678118654752440f;
    return 0.5f * value * (1.0f + std::erf(value * inv_sqrt_two));
}

float softplus(float value) {
    if (value > 20.0f) {
        return value;
    }
    if (value < -20.0f) {
        return std::exp(value);
    }
    return std::log1p(std::exp(value));
}

} // namespace

bool darwin_neg_head::load(
        const std::string & path,
        int32_t expected_hidden,
        std::string & error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "cannot open NEG head: " + path;
        return false;
    }

    std::array<char, 8> magic{};
    uint32_t version = 0;
    uint32_t file_hidden = 0;
    uint32_t file_reduced = 0;
    uint32_t file_top_k = 0;
    float file_threshold = 0.0f;
    float file_temperature_scale = 0.0f;

    if (!input.read(magic.data(), magic.size()) ||
            !read_value(input, version) ||
            !read_value(input, file_hidden) ||
            !read_value(input, file_reduced) ||
            !read_value(input, file_top_k) ||
            !read_value(input, file_threshold) ||
            !read_value(input, file_temperature_scale)) {
        error = "NEG head header is truncated";
        return false;
    }

    constexpr std::array<char, 8> expected_magic{'D', 'N', 'E', 'G', 'v', '0', '0', '1'};
    if (magic != expected_magic || version != 1) {
        error = "NEG head has an unsupported format";
        return false;
    }
    if (file_hidden != static_cast<uint32_t>(expected_hidden) ||
            file_hidden == 0 || file_reduced == 0 ||
            file_reduced > file_hidden || file_top_k == 0) {
        error = "NEG head dimensions do not match the loaded language model";
        return false;
    }
    if (!std::isfinite(file_threshold) || !std::isfinite(file_temperature_scale) ||
            file_temperature_scale <= 0.0f) {
        error = "NEG head gate parameters are invalid";
        return false;
    }

    hidden = static_cast<int32_t>(file_hidden);
    reduced = static_cast<int32_t>(file_reduced);
    top_k = static_cast<int32_t>(file_top_k);
    threshold = file_threshold;
    temperature_scale = file_temperature_scale;

    if (!read_floats(input, proj_down_weight, static_cast<size_t>(hidden) * reduced) ||
            !read_floats(input, proj_down_bias, reduced) ||
            !read_floats(input, proj_out_weight, reduced) ||
            !read_value(input, proj_out_bias)) {
        error = "NEG head weights are truncated";
        return false;
    }

    char trailing = 0;
    if (input.read(&trailing, 1)) {
        error = "NEG head contains unexpected trailing data";
        return false;
    }

    scratch.resize(reduced);
    return true;
}

darwin_neg_observation darwin_neg_head::evaluate(const float * hidden_state) {
    for (int32_t row = 0; row < reduced; ++row) {
        const float * weights = proj_down_weight.data() + static_cast<size_t>(row) * hidden;
        float sum = proj_down_bias[row];
#if defined(_OPENMP)
#pragma omp simd reduction(+:sum)
#endif
        for (int32_t column = 0; column < hidden; ++column) {
            sum += weights[column] * hidden_state[column];
        }
        scratch[row] = gelu_exact(sum);
    }

    float output = proj_out_bias;
#if defined(_OPENMP)
#pragma omp simd reduction(+:output)
#endif
    for (int32_t index = 0; index < reduced; ++index) {
        output += proj_out_weight[index] * scratch[index];
    }
    const float predicted_entropy = softplus(output);
    return {predicted_entropy, predicted_entropy > threshold};
}

void darwin_neg_head::guide_logits(float * logits, int32_t n_vocab) const {
    using scored_token = std::pair<float, int32_t>;
    std::priority_queue<
        scored_token,
        std::vector<scored_token>,
        std::greater<scored_token>> best;

    const int32_t keep = std::min(top_k, n_vocab);
    for (int32_t token = 0; token < n_vocab; ++token) {
        const float scaled = logits[token] / temperature_scale;
        if (static_cast<int32_t>(best.size()) < keep) {
            best.emplace(scaled, token);
        } else if (scaled > best.top().first) {
            best.pop();
            best.emplace(scaled, token);
        }
    }

    std::vector<scored_token> retained;
    retained.reserve(best.size());
    while (!best.empty()) {
        retained.push_back(best.top());
        best.pop();
    }

    std::fill(logits, logits + n_vocab, -std::numeric_limits<float>::infinity());
    for (const auto & [score, token] : retained) {
        logits[token] = score;
    }
}
