#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct darwin_neg_observation {
    float predicted_entropy = 0.0f;
    bool active = false;
};

class darwin_neg_head {
public:
    bool load(const std::string & path, int32_t expected_hidden, std::string & error);

    darwin_neg_observation evaluate(const float * hidden_state);

    // Applies the released top-k gate and the released temperature scale. The
    // transformation is meaningful for sampled decoding; it intentionally
    // leaves the ordering of logits unchanged.
    void guide_logits(float * logits, int32_t n_vocab) const;

    bool loaded() const { return !proj_down_weight.empty(); }
    int32_t hidden_size() const { return hidden; }
    int32_t reduced_size() const { return reduced; }
    int32_t top_k_value() const { return top_k; }
    float threshold_value() const { return threshold; }
    float temperature_scale_value() const { return temperature_scale; }

private:
    int32_t hidden = 0;
    int32_t reduced = 0;
    int32_t top_k = 20;
    float threshold = 1.175187349319458f;
    float temperature_scale = 0.5983633399009705f;

    std::vector<float> proj_down_weight;
    std::vector<float> proj_down_bias;
    std::vector<float> proj_out_weight;
    float proj_out_bias = 0.0f;
    std::vector<float> scratch;
};
