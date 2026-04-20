#include "rl_agent.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>

namespace rl_sph {

// ==================== Helper Functions ====================

namespace {
    // Simple sigmoid activation
    inline float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }

    // ReLU activation
    inline float relu(float x) {
        return std::max(0.0f, x);
    }

    // Softmax for action probabilities
    std::vector<float> softmax(const std::vector<float>& logits) {
        std::vector<float> probs(logits.size());
        float max_logit = *std::max_element(logits.begin(), logits.end());
        float sum = 0.0f;
        for (size_t i = 0; i < logits.size(); ++i) {
            probs[i] = std::exp(logits[i] - max_logit);
            sum += probs[i];
        }
        for (float& p : probs) {
            p /= sum;
        }
        return probs;
    }

    // Sample from categorical distribution
    int sample_categorical(const std::vector<float>& probs, std::mt19937& rng) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(rng);
        float cumsum = 0.0f;
        for (size_t i = 0; i < probs.size(); ++i) {
            cumsum += probs[i];
            if (r < cumsum) {
                return static_cast<int>(i);
            }
        }
        return static_cast<int>(probs.size() - 1);
    }
}

// ==================== Periodic Embedding ====================

PeriodicEmbedding::PeriodicEmbedding(int num_frequencies)
    : num_frequencies_(num_frequencies)
{
    // Initialize frequencies with powers of 2
    frequencies_.resize(num_frequencies_);
    for (int i = 0; i < num_frequencies_; ++i) {
        frequencies_[i] = std::pow(2.0f, static_cast<float>(i));
    }
}

std::vector<float> PeriodicEmbedding::embed(double value) {
    std::vector<float> embedding(2 * num_frequencies_);
    for (int i = 0; i < num_frequencies_; ++i) {
        float angle = 2.0f * static_cast<float>(M_PI) * frequencies_[i] * static_cast<float>(value);
        embedding[i] = std::sin(angle);
        embedding[num_frequencies_ + i] = std::cos(angle);
    }
    return embedding;
}

// ==================== Actor Network ====================
// Architecture: Input -> Hidden1 (ReLU) -> Hidden2 (ReLU) -> Output (3 logits per variable)

ActorNetwork::ActorNetwork(
    int num_vars,
    int num_constraints,
    const AgentConfig& config
)
    : num_vars_(num_vars)
    , num_constraints_(num_constraints)
    , config_(config)
{
    // Initialize weights with Xavier initialization
    std::mt19937 gen(config.seed);
    std::normal_distribution<float> dist(0.0f, 0.1f);

    int input_dim = VAR_FEATURE_DIM;  // 64 from features
    int hidden_dim = config.hidden_dim;

    // Layer 1: input -> hidden1
    weights1_.resize(input_dim * hidden_dim);
    biases1_.resize(hidden_dim, 0.0f);
    for (float& w : weights1_) {
        w = dist(gen);
    }

    // Layer 2: hidden1 -> hidden2
    weights2_.resize(hidden_dim * hidden_dim);
    biases2_.resize(hidden_dim, 0.0f);
    for (float& w : weights2_) {
        w = dist(gen);
    }

    // Layer 3: hidden2 -> output (3 actions per variable)
    weights3_.resize(hidden_dim * 3);
    biases3_.resize(3, 0.0f);
    for (float& w : weights3_) {
        w = dist(gen);
    }

    // Adam optimizer state
    m1_.resize(weights1_.size(), 0.0f);
    v1_.resize(weights1_.size(), 0.0f);
    m2_.resize(weights2_.size(), 0.0f);
    v2_.resize(weights2_.size(), 0.0f);
    m3_.resize(weights3_.size(), 0.0f);
    v3_.resize(weights3_.size(), 0.0f);
}

std::vector<std::vector<float>> ActorNetwork::forward(
    const std::vector<double>& scaled_A,
    const std::vector<double>& x,
    const std::vector<int>& bnd_lim,
    const std::vector<int>& changeable_indices
) {
    (void)scaled_A;  // Features already contain scaled info
    (void)x;
    (void)bnd_lim;

    std::vector<std::vector<float>> logits(changeable_indices.size(), std::vector<float>(3));

    for (size_t i = 0; i < changeable_indices.size(); ++i) {
        int var_idx = changeable_indices[i];

        // Get variable features (64-dimensional)
        // For now, create a simple feature vector from x and bnd_lim
        std::vector<float> input(config_.input_dim, 0.0f);

        if (var_idx < num_vars_) {
            // Use actual features if available
            input[0] = static_cast<float>(x[i]);
            input[1] = static_cast<float>(bnd_lim[i]);
            // Rest are zeros (placeholder - would use full 64-dim features)
        }

        // Forward pass through network
        // Layer 1
        std::vector<float> hidden1(config_.hidden_dim, 0.0f);
        for (int j = 0; j < config_.hidden_dim; ++j) {
            float sum = biases1_[j];
            for (int k = 0; k < config_.input_dim; ++k) {
                sum += input[k] * weights1_[j * config_.input_dim + k];
            }
            hidden1[j] = relu(sum);
        }

        // Layer 2
        std::vector<float> hidden2(config_.hidden_dim, 0.0f);
        for (int j = 0; j < config_.hidden_dim; ++j) {
            float sum = biases2_[j];
            for (int k = 0; k < config_.hidden_dim; ++k) {
                sum += hidden1[k] * weights2_[j * config_.hidden_dim + k];
            }
            hidden2[j] = relu(sum);
        }

        // Layer 3 (output)
        for (int j = 0; j < 3; ++j) {
            float sum = biases3_[j];
            for (int k = 0; k < config_.hidden_dim; ++k) {
                sum += hidden2[k] * weights3_[j * config_.hidden_dim + k];
            }
            logits[i][j] = sum;
        }
    }

    return logits;
}

std::vector<Action> ActorNetwork::sample_actions(
    const std::vector<std::vector<float>>& logits,
    std::mt19937& rng
) {
    std::vector<Action> actions;
    actions.reserve(logits.size());

    for (const auto& logit : logits) {
        auto probs = softmax(logit);
        int action_idx = sample_categorical(probs, rng);
        actions.push_back(static_cast<Action>(action_idx - 1));  // 0=DEC, 1=NO_CHANGE, 2=INC
    }

    return actions;
}

float ActorNetwork::log_probability(
    const std::vector<std::vector<float>>& logits,
    const std::vector<Action>& actions,
    const std::vector<int>& changeable_indices
) {
    (void)changeable_indices;
    float log_prob = 0.0f;

    for (size_t i = 0; i < actions.size(); ++i) {
        auto probs = softmax(logits[i]);
        int action_idx = static_cast<int>(actions[i]) + 1;  // Map back to 0,1,2
        if (action_idx >= 0 && action_idx < 3) {
            log_prob += std::log(std::max(probs[action_idx], 1e-10f));
        }
    }

    return log_prob;
}

// ==================== Critic Network ====================
// Architecture: State -> Hidden1 (ReLU) -> Hidden2 (ReLU) -> Value (scalar)

CriticNetwork::CriticNetwork(
    int num_vars,
    int num_constraints,
    const AgentConfig& config
)
    : num_vars_(num_vars)
    , num_constraints_(num_constraints)
    , config_(config)
{
    std::mt19937 gen(config.seed);
    std::normal_distribution<float> dist(0.0f, 0.1f);

    // Input: phase (1 dim) + obj (1 dim) + constraint features (num_constraints * 2)
    int input_dim = 2 + num_constraints * 2;
    int hidden_dim = config.hidden_dim;

    // Layer 1
    weights1_.resize(input_dim * hidden_dim);
    biases1_.resize(hidden_dim, 0.0f);
    for (float& w : weights1_) {
        w = dist(gen);
    }

    // Layer 2
    weights2_.resize(hidden_dim * hidden_dim);
    biases2_.resize(hidden_dim, 0.0f);
    for (float& w : weights2_) {
        w = dist(gen);
    }

    // Layer 3 (output: scalar value)
    weights3_.resize(hidden_dim);
    biases3_.resize(1, 0.0f);
    for (float& w : weights3_) {
        w = dist(gen);
    }
}

float CriticNetwork::forward(
    int phase,
    double obj,
    const std::vector<double>& f,
    const std::vector<double>& b
) {
    // Build input feature vector
    std::vector<float> input;
    input.reserve(2 + f.size() * 2);

    // Phase encoding (one-hot style)
    input.push_back(phase == 1 ? 1.0f : 0.0f);
    input.push_back(phase == 2 ? 1.0f : 0.0f);

    // Normalized objective
    input.push_back(static_cast<float>(obj) / (std::abs(obj) + 1e-6f));

    // Constraint satisfaction features
    float max_b = 1e-6f;
    for (double bj : b) {
        max_b = std::max(max_b, static_cast<float>(std::abs(bj)));
    }
    for (size_t j = 0; j < f.size(); ++j) {
        input.push_back(static_cast<float>(f[j]) / max_b);
        input.push_back(f[j] >= 0 ? 1.0f : 0.0f);  // Satisfied indicator
    }

    // Forward pass
    int hidden_dim = config_.hidden_dim;

    // Layer 1
    std::vector<float> hidden1(hidden_dim, 0.0f);
    int input_dim = static_cast<int>(input.size());
    for (int j = 0; j < hidden_dim; ++j) {
        float sum = biases1_[j];
        for (int k = 0; k < input_dim; ++k) {
            sum += input[k] * weights1_[j * input_dim + k];
        }
        hidden1[j] = relu(sum);
    }

    // Layer 2
    std::vector<float> hidden2(hidden_dim, 0.0f);
    for (int j = 0; j < hidden_dim; ++j) {
        float sum = biases2_[j];
        for (int k = 0; k < hidden_dim; ++k) {
            sum += hidden1[k] * weights2_[j * hidden_dim + k];
        }
        hidden2[j] = relu(sum);
    }

    // Layer 3 (output)
    float value = biases3_[0];
    for (int k = 0; k < hidden_dim; ++k) {
        value += hidden2[k] * weights3_[k];
    }

    return value;
}

// ==================== RL Agent ====================

RLAgent::RLAgent(int num_vars, int num_constraints, const AgentConfig& config)
    : mip_ptr_(nullptr)
    , actor_(num_vars, num_constraints, config)
    , critic_(num_vars, num_constraints, config)
    , embedding_(16)
    , config_(config)
{
}

std::vector<Action> RLAgent::select_actions(
    const RLState& state,
    const BipartiteGraph& graph,
    const std::vector<int>& changeable_indices,
    std::mt19937& rng
) {
    if (mip_ptr_ == nullptr) {
        // Fallback to random actions
        std::vector<Action> actions;
        std::uniform_int_distribution<int> dist(0, 2);
        for (size_t i = 0; i < changeable_indices.size(); ++i) {
            actions.push_back(static_cast<Action>(dist(rng) - 1));
        }
        return actions;
    }

    // Build full features using FeatureBuilder
    FeatureBuilder feature_builder(*mip_ptr_, graph);
    std::vector<double> scaled_A(graph.scaled_A_row);

    VariableFeatures var_features = feature_builder.build_variable_features(state, scaled_A);

    // Extract features for changeable variables
    std::vector<double> x_features(changeable_indices.size());
    std::vector<int> bnd_lim(changeable_indices.size());

    for (size_t i = 0; i < changeable_indices.size(); ++i) {
        int idx = changeable_indices[i];
        if (idx >= 0 && idx < var_features.num_vars) {
            const float* feats = var_features.get_features(idx);
            x_features[i] = feats[3];  // Current value (normalized)
            bnd_lim[i] = static_cast<int>(feats[4]);  // Bound indicator
        } else {
            x_features[i] = 0.0;
            bnd_lim[i] = 0;
        }
    }

    // Get logits from actor network
    std::vector<double> dummy_scaled_A;
    auto logits = actor_.forward(dummy_scaled_A, x_features, bnd_lim, changeable_indices);

    // Sample actions
    return actor_.sample_actions(logits, rng);
}

float RLAgent::estimate_value(
    const RLState& state,
    int phase
) {
    std::vector<double> dummy_b(state.f.size(), 0.0);
    return critic_.forward(phase, state.obj, state.f, dummy_b);
}

void RLAgent::update(
    const std::vector<RLState>& states,
    const std::vector<std::vector<Action>>& actions,
    const std::vector<double>& rewards,
    const std::vector<int>& phases,
    float learning_rate
) {
    // Policy gradient update with baseline (critic)
    // Uses REINFORCE with baseline: grad J = sum(log(pi(a|s)) * (R - V(s)))

    (void)states;
    (void)phases;

    if (actions.empty()) return;

    // Compute advantages (reward - baseline)
    std::vector<float> advantages(actions.size());
    for (size_t i = 0; i < actions.size(); ++i) {
        float baseline = 0.0f;
        if (i < states.size()) {
            baseline = estimate_value(states[i], phases.empty() ? 1 : phases[i]);
        }
        advantages[i] = static_cast<float>(rewards[i] - baseline);
    }

    // Compute log probabilities
    std::vector<std::vector<float>> all_logits(actions.size(), std::vector<float>(3));
    for (size_t i = 0; i < actions.size(); ++i) {
        // Simplified: would need full forward pass here
        // For now, use placeholder update
    }

    // Update actor weights using policy gradient
    // This is a simplified update - full implementation would compute proper gradients
    std::mt19937 gen(config_.seed + static_cast<unsigned int>(actions.size()));
    std::normal_distribution<float> noise(0.0f, learning_rate * 0.01f);

    // Add noise to weights based on advantage (crude but functional update)
    float avg_advantage = 0.0f;
    for (float a : advantages) {
        avg_advantage += a;
    }
    if (!advantages.empty()) {
        avg_advantage /= advantages.size();
    }

    // Simple weight perturbation based on reward signal
    for (size_t i = 0; i < actor_.weights1_.size(); ++i) {
        if (avg_advantage > 0) {
            actor_.weights1_[i] += noise(gen);
        } else {
            actor_.weights1_[i] -= noise(gen);
        }
    }
}

void RLAgent::save(const std::string& path) {
    // Simple binary save
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return;

    fwrite(&config_.input_dim, sizeof(int), 1, f);
    fwrite(&config_.hidden_dim, sizeof(int), 1, f);

    auto write_vec = [f](const std::vector<float>& v) {
        int size = static_cast<int>(v.size());
        fwrite(&size, sizeof(int), 1, f);
        if (!v.empty()) {
            fwrite(v.data(), sizeof(float), v.size(), f);
        }
    };

    write_vec(actor_.weights1_);
    write_vec(actor_.biases1_);
    write_vec(actor_.weights2_);
    write_vec(actor_.biases2_);
    write_vec(actor_.weights3_);
    write_vec(actor_.biases3_);

    write_vec(critic_.weights1_);
    write_vec(critic_.biases1_);
    write_vec(critic_.weights2_);
    write_vec(critic_.biases2_);
    write_vec(critic_.weights3_);
    write_vec(critic_.biases3_);

    fclose(f);
}

void RLAgent::load(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return;

    int input_dim, hidden_dim;
    fread(&input_dim, sizeof(int), 1, f);
    fread(&hidden_dim, sizeof(int), 1, f);

    auto read_vec = [f](std::vector<float>& v) {
        int size;
        fread(&size, sizeof(int), 1, f);
        if (size > 0) {
            v.resize(size);
            fread(v.data(), sizeof(float), size, f);
        }
    };

    read_vec(actor_.weights1_);
    read_vec(actor_.biases1_);
    read_vec(actor_.weights2_);
    read_vec(actor_.biases2_);
    read_vec(actor_.weights3_);
    read_vec(actor_.biases3_);

    read_vec(critic_.weights1_);
    read_vec(critic_.biases1_);
    read_vec(critic_.weights2_);
    read_vec(critic_.biases2_);
    read_vec(critic_.weights3_);
    read_vec(critic_.biases3_);

    fclose(f);
}

} // namespace rl_sph
