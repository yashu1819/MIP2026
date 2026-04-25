#include "rl_agent.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>
#include <fstream>
#include <cstring>
#include <iostream>

namespace rl_sph {

// ==================== Helper Functions ====================
namespace {
    inline float relu(float x) { return std::max(0.0f, x); }

    std::vector<float> softmax(const std::vector<float>& logits) {
        std::vector<float> probs(logits.size());
        float max_l = *std::max_element(logits.begin(), logits.end());
        float sum = 0.0f;
        for (size_t i = 0; i < logits.size(); ++i) {
            probs[i] = std::exp(logits[i] - max_l);
            sum += probs[i];
        }
        for (float& p : probs) p /= sum;
        return probs;
    }

    int sample_categorical(const std::vector<float>& probs, std::mt19937& rng) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(rng), cumsum = 0.0f;
        for (size_t i = 0; i < probs.size(); ++i) {
            cumsum += probs[i];
            if (r < cumsum) return static_cast<int>(i);
        }
        return static_cast<int>(probs.size() - 1);
    }
} // anon namespace

// ==================== Periodic Embedding ====================
PeriodicEmbedding::PeriodicEmbedding(int num_frequencies) : num_frequencies_(num_frequencies) {
    frequencies_.resize(num_frequencies_);
    for (int i = 0; i < num_frequencies_; ++i)
        frequencies_[i] = std::pow(2.0f, static_cast<float>(i));
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

#ifdef USE_LIBTORCH
torch::Tensor PeriodicEmbedding::embed_batch(const std::vector<double>& values) {
    int n = static_cast<int>(values.size());
    auto result = torch::zeros({n, 2 * num_frequencies_}, torch::kFloat32);
    for (int i = 0; i < num_frequencies_; ++i) {
        float freq = frequencies_[i];
        for (int j = 0; j < n; ++j) {
            float theta = 2.0f * static_cast<float>(M_PI) * freq * static_cast<float>(values[j]);
            result[j][i] = std::sin(theta);
            result[j][i + num_frequencies_] = std::cos(theta);
        }
    }
    return result;
}
#endif

// ==================== CPU Actor Network ====================
ActorNetworkCPU::ActorNetworkCPU(int num_vars, int num_constraints, const AgentConfig& config)
    : num_vars_(num_vars), num_constraints_(num_constraints), config_(config)
{
    std::mt19937 gen(config.seed);
    std::normal_distribution<float> dist(0.0f, 0.1f);
    int input_dim = VAR_FEATURE_DIM, hidden_dim = config.hidden_dim;

    weights1_.resize(input_dim * hidden_dim);
    biases1_.resize(hidden_dim, 0.0f);
    for (float& w : weights1_) w = dist(gen);

    weights2_.resize(hidden_dim * hidden_dim);
    biases2_.resize(hidden_dim, 0.0f);
    for (float& w : weights2_) w = dist(gen);

    weights3_.resize(hidden_dim * 3);
    biases3_.resize(3, 0.0f);
    for (float& w : weights3_) w = dist(gen);
}

ActorNetworkCPU::~ActorNetworkCPU() {}

std::vector<std::vector<float> > ActorNetworkCPU::forward(
    const std::vector<double>& var_features,
    const std::vector<int>& changeable_indices
) {
    (void)changeable_indices;
    int num_ch = static_cast<int>(var_features.size()) / VAR_FEATURE_DIM;
    int hd = config_.hidden_dim;
    std::vector<std::vector<float> > logits(num_ch, std::vector<float>(3));

    for (int i = 0; i < num_ch; ++i) {
        std::vector<float> input(VAR_FEATURE_DIM);
        for (int k = 0; k < VAR_FEATURE_DIM; ++k)
            input[k] = static_cast<float>(var_features[i * VAR_FEATURE_DIM + k]);

        std::vector<float> h1(hd, 0.0f);
        for (int j = 0; j < hd; ++j) {
            float s = biases1_[j];
            for (int k = 0; k < VAR_FEATURE_DIM; ++k)
                s += input[k] * weights1_[j * VAR_FEATURE_DIM + k];
            h1[j] = relu(s);
        }

        std::vector<float> h2(hd, 0.0f);
        for (int j = 0; j < hd; ++j) {
            float s = biases2_[j];
            for (int k = 0; k < hd; ++k) s += h1[k] * weights2_[j * hd + k];
            h2[j] = relu(s);
        }

        for (int j = 0; j < 3; ++j) {
            float s = biases3_[j];
            for (int k = 0; k < hd; ++k) s += h2[k] * weights3_[j * hd + k];
            logits[i][j] = s;
        }
    }
    return logits;
}

std::vector<Action> ActorNetworkCPU::sample_actions(
    const std::vector<std::vector<float> >& logits, std::mt19937& rng
) {
    std::vector<Action> actions;
    actions.reserve(logits.size());
    for (const auto& logit : logits) {
        auto probs = softmax(logit);
        int idx = sample_categorical(probs, rng);
        actions.push_back(static_cast<Action>(idx - 1));
    }
    return actions;
}

float ActorNetworkCPU::log_probability(
    const std::vector<std::vector<float> >& logits,
    const std::vector<Action>& actions
) {
    float lp = 0.0f;
    for (size_t i = 0; i < actions.size(); ++i) {
        auto probs = softmax(logits[i]);
        int idx = static_cast<int>(actions[i]) + 1;
        if (idx >= 0 && idx < 3) lp += std::log(std::max(probs[idx], 1e-10f));
    }
    return lp;
}

// ==================== CPU Critic Network ====================
CriticNetworkCPU::CriticNetworkCPU(int num_vars, int num_constraints, const AgentConfig& config)
    : num_vars_(num_vars), num_constraints_(num_constraints), config_(config)
{
    std::mt19937 gen(config.seed);
    std::normal_distribution<float> dist(0.0f, 0.1f);
    int input_dim = 2 + 1 + num_constraints * 2, hd = config.hidden_dim;

    weights1_.resize(input_dim * hd); biases1_.resize(hd, 0.0f);
    for (float& w : weights1_) w = dist(gen);
    weights2_.resize(hd * hd); biases2_.resize(hd, 0.0f);
    for (float& w : weights2_) w = dist(gen);
    weights3_.resize(hd); biases3_.resize(1, 0.0f);
    for (float& w : weights3_) w = dist(gen);
}

CriticNetworkCPU::~CriticNetworkCPU() {}

float CriticNetworkCPU::forward(int phase, double obj,
    const std::vector<double>& f, const std::vector<double>& b)
{
    int hd = config_.hidden_dim;
    std::vector<float> input;
    input.reserve(3 + f.size() * 2);
    input.push_back(phase == 1 ? 1.0f : 0.0f);
    input.push_back(phase == 2 ? 1.0f : 0.0f);
    input.push_back(static_cast<float>(obj) / (std::abs(static_cast<float>(obj)) + 1e-6f));

    float max_b = 1e-6f;
    for (double bj : b) max_b = std::max(max_b, static_cast<float>(std::abs(bj)));
    for (size_t j = 0; j < f.size(); ++j) {
        input.push_back(static_cast<float>(f[j]) / max_b);
        input.push_back((f[j] >= 0) ? 1.0f : 0.0f);
    }

    int id = static_cast<int>(input.size());
    std::vector<float> h1(hd, 0.0f);
    for (int j = 0; j < hd; ++j) {
        float s = biases1_[j];
        for (int k = 0; k < id; ++k) s += input[k] * weights1_[j * id + k];
        h1[j] = relu(s);
    }
    std::vector<float> h2(hd, 0.0f);
    for (int j = 0; j < hd; ++j) {
        float s = biases2_[j];
        for (int k = 0; k < hd; ++k) s += h1[k] * weights2_[j * hd + k];
        h2[j] = relu(s);
    }
    float val = biases3_[0];
    for (int k = 0; k < hd; ++k) val += h2[k] * weights3_[k];
    return val;
}

// ==================== RL Agent ====================
RLAgent::RLAgent(int num_vars, int num_constraints, const AgentConfig& config)
    : mip_ptr_(nullptr)
    , embedding_(new PeriodicEmbedding(16))
    , config_(config)
    , num_vars_(num_vars)
    , num_constraints_(num_constraints)
    , actor_cpu_(new ActorNetworkCPU(num_vars, num_constraints, config))
    , critic_cpu_(new CriticNetworkCPU(num_vars, num_constraints, config))
#ifdef USE_LIBTORCH
    , actor_torch_(VAR_FEATURE_DIM, CONST_FEATURE_DIM, config.hidden_dim, config.num_heads, config.num_layers)
    , critic_torch_(VAR_FEATURE_DIM, CONST_FEATURE_DIM, config.hidden_dim)
    , optimizer_(nullptr)
#endif
{
#ifdef USE_LIBTORCH
    // Collect all parameters from both actor and critic
    std::vector<torch::Tensor> all_params;
    for (auto& p : actor_torch_->parameters()) all_params.push_back(p);
    for (auto& p : critic_torch_->parameters()) all_params.push_back(p);

    auto rmsprop_opts = torch::optim::RMSpropOptions(config.learning_rate)
        .alpha(config.rmsprop_alpha)
        .eps(config.rmsprop_epsilon)
        .weight_decay(config.weight_decay);
    optimizer_ = std::make_shared<torch::optim::RMSprop>(all_params, rmsprop_opts);
#endif
}

RLAgent::~RLAgent() {
    delete actor_cpu_;
    delete critic_cpu_;
    delete embedding_;
}

#ifdef USE_LIBTORCH
// Build variable feature tensor for GNN actor
torch::Tensor RLAgent::build_var_feature_tensor(
    const RLState& state,
    const BipartiteGraph& graph
) {
    if (mip_ptr_ == nullptr)
        return torch::zeros({num_vars_, VAR_FEATURE_DIM}, torch::kFloat32);

    FeatureBuilder fb(*mip_ptr_, graph);
    std::vector<double> scaled_A(graph.scaled_A_row);
    VariableFeatures vf = fb.build_variable_features(state, scaled_A);

    auto tensor = torch::zeros({num_vars_, VAR_FEATURE_DIM}, torch::kFloat32);
    auto acc = tensor.accessor<float, 2>();

    for (int i = 0; i < num_vars_; ++i) {
        const float* feats = vf.get_features(i);
        for (int j = 0; j < VAR_FEATURE_DIM; ++j) {
            acc[i][j] = feats[j];
        }
    }
    return tensor;
}

// Build constraint feature tensor for GNN
torch::Tensor RLAgent::build_constr_feature_tensor(
    const RLState& state,
    const BipartiteGraph& graph
) {
    if (mip_ptr_ == nullptr)
        return torch::zeros({num_constraints_, CONST_FEATURE_DIM}, torch::kFloat32);

    FeatureBuilder fb(*mip_ptr_, graph);
    std::vector<double> scaled_A(graph.scaled_A_row);
    ConstraintFeatures cf = fb.build_constraint_features(state, scaled_A);

    auto tensor = torch::zeros({num_constraints_, CONST_FEATURE_DIM}, torch::kFloat32);
    auto acc = tensor.accessor<float, 2>();

    for (int j = 0; j < num_constraints_; ++j) {
        const float* feats = cf.get_features(j);
        for (int k = 0; k < CONST_FEATURE_DIM; ++k) {
            acc[j][k] = feats[k];
        }
    }
    return tensor;
}

// Build edge index tensor for graph message passing
// Returns (2, num_edges) tensor where row 0 = source indices, row 1 = target indices
torch::Tensor RLAgent::build_edge_index_tensor(
    const BipartiteGraph& graph
) {
    int num_edges = static_cast<int>(graph.edges.size());
    auto edge_index = torch::zeros({2, num_edges}, torch::kLong);
    auto acc = edge_index.accessor<long, 2>();

    for (int i = 0; i < num_edges; ++i) {
        acc[0][i] = graph.edges[i].first;   // var index
        acc[1][i] = graph.edges[i].second;  // constraint index
    }
    return edge_index;
}

// Build edge weight tensor from constraint matrix coefficients
// Returns (num_edges,) tensor with |A[var, constraint]| for each edge
torch::Tensor RLAgent::build_edge_weight_tensor(
    const BipartiteGraph& graph
) {
    int num_edges = static_cast<int>(graph.edges.size());
    auto edge_weights = torch::zeros({num_edges}, torch::kFloat32);
    auto acc = edge_weights.accessor<float, 1>();

    // Use scaled_A_row to get coefficient for each edge
    // scaled_A_row is flattened CSR format
    for (int i = 0; i < num_edges; ++i) {
        int var_idx = graph.edges[i].first;
        int constr_idx = graph.edges[i].second;

        // Find the coefficient in the sparse matrix
        // A_row_ptr[constr_idx] to A_row_ptr[constr_idx+1] gives range in A_col_idx
        int start = graph.A_row_ptr[constr_idx];
        int end = graph.A_row_ptr[constr_idx + 1];

        float coef = 1.0f;  // default weight
        for (int k = start; k < end; ++k) {
            if (graph.A_col_idx[k] == var_idx) {
                coef = static_cast<float>(std::abs(graph.scaled_A_row[k]));
                break;
            }
        }
        acc[i] = coef;
    }
    return edge_weights;
}

// Build changeable mask for actor forward pass
torch::Tensor RLAgent::build_changeable_mask(
    int num_vars,
    const std::vector<int>& changeable_indices
) {
    auto mask = torch::zeros({num_vars}, torch::kBool);
    auto acc = mask.accessor<bool, 1>();
    for (int idx : changeable_indices) {
        if (idx >= 0 && idx < num_vars) {
            acc[idx] = true;
        }
    }
    return mask;
}
#endif

std::vector<Action> RLAgent::select_actions(
    const RLState& state, const BipartiteGraph& graph,
    const std::vector<int>& changeable_indices, std::mt19937& rng)
{
    // Guard: if no variables to change, return empty
    if (changeable_indices.empty()) return std::vector<Action>();

#ifdef USE_LIBTORCH
    torch::NoGradGuard no_grad;

    // Build feature tensors
    auto var_features = build_var_feature_tensor(state, graph);
    auto constr_features = build_constr_feature_tensor(state, graph);
    auto edge_index = build_edge_index_tensor(graph);
    auto edge_weights = build_edge_weight_tensor(graph);
    auto changeable_mask = build_changeable_mask(num_vars_, changeable_indices);

    // Actor forward pass with GNN (inference mode, with edge weights)
    auto logits_t = actor_torch_->forward(var_features, constr_features, edge_index, changeable_mask, /*is_training=*/false, edge_weights);

    int nc = static_cast<int>(changeable_indices.size());
    std::vector<Action> actions;
    actions.reserve(nc);

    auto acc = logits_t.accessor<float, 2>();
    for (int i = 0; i < nc; ++i) {
        std::vector<float> lg = {acc[i][0], acc[i][1], acc[i][2]};
        auto probs = softmax(lg);
        int idx = sample_categorical(probs, rng);
        actions.push_back(static_cast<Action>(idx - 1));
    }
    return actions;
#else
    if (mip_ptr_ == nullptr) {
        std::vector<Action> actions;
        std::uniform_int_distribution<int> dist(0, 2);
        for (size_t i = 0; i < changeable_indices.size(); ++i)
            actions.push_back(static_cast<Action>(dist(rng) - 1));
        return actions;
    }
    FeatureBuilder fb(*mip_ptr_, graph);
    std::vector<double> scaled_A(graph.scaled_A_row);
    VariableFeatures vf = fb.build_variable_features(state, scaled_A);

    std::vector<double> features;
    features.reserve(changeable_indices.size() * VAR_FEATURE_DIM);
    for (int idx : changeable_indices) {
        if (idx >= 0 && idx < vf.num_vars) {
            const float* f = vf.get_features(idx);
            for (int j = 0; j < VAR_FEATURE_DIM; ++j) features.push_back(f[j]);
        } else {
            for (int j = 0; j < VAR_FEATURE_DIM; ++j) features.push_back(0.0);
        }
    }
    auto logits = actor_cpu_->forward(features, changeable_indices);
    return actor_cpu_->sample_actions(logits, rng);
#endif
}

TrainingForwardResult RLAgent::select_actions_training(
    const RLState& state, const BipartiteGraph& graph,
    const std::vector<int>& changeable_indices, int phase, std::mt19937& rng)
{
    (void)rng;
    TrainingForwardResult result;
    int nc = static_cast<int>(changeable_indices.size());

    // Guard: if no variables to change, return defaults
    if (nc == 0) {
        result.log_prob_sum = 0.0f;
        result.state_value = estimate_value(state, phase);
        return result;
    }

#ifdef USE_LIBTORCH
    // Build feature tensors
    auto var_features = build_var_feature_tensor(state, graph);
    auto constr_features = build_constr_feature_tensor(state, graph);
    auto edge_index = build_edge_index_tensor(graph);
    auto edge_weights = build_edge_weight_tensor(graph);
    auto changeable_mask = build_changeable_mask(num_vars_, changeable_indices);

    // Actor forward pass (with gradients, training mode, with edge weights)
    auto logits_t = actor_torch_->forward(var_features, constr_features, edge_index, changeable_mask, /*is_training=*/true, edge_weights);

    // Sample actions and compute log probs
    auto probs_t = torch::softmax(logits_t, /*dim=*/1);  // (nc, 3)
    
    // discourage action = 0 (index 1)
    probs_t.select(1, 1) *= 0.1;
    probs_t = probs_t / probs_t.sum(1, true);

    // std::cout << "probs: " << probs_t << std::endl;
    std::cout << "min: " << probs_t.min().item<float>() << " max: " << probs_t.max().item<float>() << " sum: " << probs_t.sum().item<float>() << std::endl;
    // auto dist = torch::multinomial(probs_t, 1);  // (nc, 1)

    float eps = 0.3;  // start HIGH
    auto rand_mask = torch::rand({probs_t.size(0), 1}, probs_t.options());
    auto random_actions = torch::randint(0, 3, {probs_t.size(0), 1}, torch::kLong);
    auto sampled = torch::multinomial(probs_t, 1);
    // override some actions with random
    auto use_random = (rand_mask < eps);
    auto dist = torch::where(use_random, random_actions, sampled);

    auto log_probs = torch::log_softmax(logits_t, 1);

    result.log_prob_sum = 0.0f;
    auto dist_acc = dist.accessor<long, 2>();
    auto logits_acc = logits_t.accessor<float, 2>();

    result.actions.reserve(nc);
    result.logits.resize(nc, std::vector<float>(3));
    for (int i = 0; i < nc; ++i) {
        int ai = static_cast<int>(dist_acc[i][0]);
        result.actions.push_back(static_cast<Action>(ai - 1));
        for (int j = 0; j < 3; ++j) result.logits[i][j] = logits_acc[i][j];
        result.log_prob_sum += log_probs[i][ai].item<float>();
    }

    // Critic forward pass (with gradients)
    float obj_norm = static_cast<float>(state.obj) / (std::abs(static_cast<float>(state.obj)) + 1e-6f);
    auto obj_tensor = torch::tensor(obj_norm, torch::kFloat32);

    // Build reverse edge index for critic
    auto edge_index_var2constr = build_edge_index_tensor(graph);  // [var; constr]
    auto edge_index_constr2var = torch::stack({
        edge_index_var2constr[1],  // constr indices become source
        edge_index_var2constr[0]   // var indices become target
    });

    auto value_t = critic_torch_->forward(
        var_features,
        constr_features,
        edge_index_var2constr,
        edge_index_constr2var,
        (phase == 1) ? 0 : 1,
        obj_tensor
    );
    result.state_value = value_t.item<float>();
#else
    // CPU fallback
    if (mip_ptr_ != nullptr) {
        FeatureBuilder fb(*mip_ptr_, graph);
        std::vector<double> scaled_A(graph.scaled_A_row);
        VariableFeatures vf = fb.build_variable_features(state, scaled_A);
        std::vector<double> feats;
        feats.reserve(nc * VAR_FEATURE_DIM);
        for (int idx : changeable_indices) {
            if (idx >= 0 && idx < vf.num_vars) {
                const float* f = vf.get_features(idx);
                for (int j = 0; j < VAR_FEATURE_DIM; ++j) feats.push_back(f[j]);
            } else {
                for (int j = 0; j < VAR_FEATURE_DIM; ++j) feats.push_back(0.0);
            }
        }
        result.logits = actor_cpu_->forward(feats, changeable_indices);
        result.actions = actor_cpu_->sample_actions(result.logits, rng);
        result.log_prob_sum = actor_cpu_->log_probability(result.logits, result.actions);
    } else {
        std::uniform_int_distribution<int> d(0, 2);
        for (int i = 0; i < nc; ++i)
            result.actions.push_back(static_cast<Action>(d(rng) - 1));
        result.log_prob_sum = 0.0f;
    }
    result.state_value = estimate_value(state, phase);
#endif
    return result;
}

float RLAgent::estimate_value(const RLState& state, int phase) {
#ifdef USE_LIBTORCH
    torch::NoGradGuard no_grad;

    if (mip_ptr_ == nullptr) {
        // Fallback for CPU
        std::vector<double> dummy_b(state.f.size(), 0.0);
        return critic_cpu_->forward(phase, state.obj, state.f, dummy_b);
    }

    // Build graph and feature tensors
    BipartiteGraph graph = build_graph(*mip_ptr_);
    auto var_features = build_var_feature_tensor(state, graph);
    auto constr_features = build_constr_feature_tensor(state, graph);

    float obj_norm = static_cast<float>(state.obj) / (std::abs(static_cast<float>(state.obj)) + 1e-6f);
    auto obj_tensor = torch::tensor(obj_norm, torch::kFloat32);

    // Build edge indices
    auto edge_index_var2constr = build_edge_index_tensor(graph);
    auto edge_index_constr2var = torch::stack({
        edge_index_var2constr[1],
        edge_index_var2constr[0]
    });

    auto value_t = critic_torch_->forward(
        var_features,
        constr_features,
        edge_index_var2constr,
        edge_index_constr2var,
        (phase == 1) ? 0 : 1,
        obj_tensor
    );
    return value_t.item<float>();
#else
    std::vector<double> dummy_b(state.f.size(), 0.0);
    return critic_cpu_->forward(phase, state.obj, state.f, dummy_b);
#endif
}

void RLAgent::update(
    const std::vector<RLState>& states,
    const std::vector<std::vector<Action> >& actions,
    const std::vector<double>& rewards,
    const std::vector<int>& phases,
    const std::vector<std::vector<int> >& changeable_indices_list,
    const BipartiteGraph& graph,
    float gamma)
{
    if (actions.empty()) return;
    size_t T = actions.size();

#ifdef USE_LIBTORCH
    // Compute values for all states
    std::vector<float> values(T + 1, 0.0f);
    for (size_t i = 0; i < T; ++i)
        values[i] = estimate_value(states[i], phases[i]);
    if (T < states.size())
        values[T] = estimate_value(states[T], phases[T - 1]);

    // Compute TD errors (advantages)
    std::vector<float> deltas(T);
    for (size_t i = 0; i < T; ++i)
        deltas[i] = static_cast<float>(rewards[i]) + gamma * values[i + 1] - values[i];

    // Build edge index and edge weights once
    auto edge_index = build_edge_index_tensor(graph);
    auto edge_weights = build_edge_weight_tensor(graph);
    auto edge_index_constr2var = torch::stack({
        edge_index[1],
        edge_index[0]
    });

    // Accumulate loss over trajectory
    torch::Tensor total_loss = torch::zeros({}, torch::kFloat32);

    for (size_t i = 0; i < T; ++i) {
        // Build feature tensors
        auto var_features = build_var_feature_tensor(states[i], graph);
        auto constr_features = build_constr_feature_tensor(states[i], graph);
        auto changeable_mask = build_changeable_mask(num_vars_, changeable_indices_list[i]);

        // Actor forward pass (training mode, with edge weights)
        auto logits_t = actor_torch_->forward(var_features, constr_features, edge_index, changeable_mask, /*is_training=*/true, edge_weights);
        auto log_probs = torch::log_softmax(logits_t, 1);

        // Gather log probs for taken actions
        int nc = static_cast<int>(actions[i].size());
        auto action_indices = torch::zeros({nc, 1}, torch::kLong);
        for (int j = 0; j < nc; ++j)
            action_indices[j][0] = static_cast<int>(actions[i][j]) + 1;

        auto selected_log_probs = log_probs.gather(1, action_indices).squeeze(1);
        auto log_prob_sum = selected_log_probs.sum();

        // Actor loss: -log pi(a|s) * delta
        float delta = deltas[i];
        auto actor_loss = -log_prob_sum * delta;

        // Critic forward pass
        float obj_norm = static_cast<float>(states[i].obj) /
                         (std::abs(static_cast<float>(states[i].obj)) + 1e-6f);
        auto obj_t = torch::tensor(obj_norm, torch::kFloat32);

        auto v = critic_torch_->forward(
            var_features,
            constr_features,
            edge_index,
            edge_index_constr2var,
            (phases[i] == 1) ? 0 : 1,
            obj_t
        );

        auto td_target = torch::tensor(static_cast<float>(rewards[i]) + gamma * values[i + 1]);
        auto critic_loss = torch::pow(td_target - v, 2);

        total_loss = total_loss + actor_loss + critic_loss;
    }

    total_loss = total_loss / static_cast<float>(T);
    optimizer_->zero_grad();
    total_loss.backward();
    optimizer_->step();
#else
    // CPU fallback: simplified weight perturbation
    std::vector<float> advantages(T);
    for (size_t i = 0; i < T; ++i) {
        float baseline = estimate_value(states[i], phases.empty() ? 1 : phases[i]);
        advantages[i] = static_cast<float>(rewards[i]) - baseline;
    }
    float avg_adv = 0.0f;
    for (float a : advantages) avg_adv += a;
    if (!advantages.empty()) avg_adv /= static_cast<float>(advantages.size());

    std::mt19937 gen(config_.seed + static_cast<unsigned>(T));
    std::normal_distribution<float> noise(0.0f, config_.learning_rate * 0.01f);
    if (avg_adv > 0) {
        for (size_t i = 0; i < actor_cpu_->weights1_.size(); ++i)
            actor_cpu_->weights1_[i] += noise(gen);
    } else if (avg_adv < 0) {
        for (size_t i = 0; i < actor_cpu_->weights1_.size(); ++i)
            actor_cpu_->weights1_[i] -= noise(gen);
    }
#endif
}

void RLAgent::save(const std::string& path) {
#ifdef USE_LIBTORCH
    torch::save(actor_torch_, path + ".actor");
    torch::save(critic_torch_, path + ".critic");
#else
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return;
    fwrite(&config_.input_dim, sizeof(int), 1, f);
    fwrite(&config_.hidden_dim, sizeof(int), 1, f);
    auto wv = [](FILE* fp, const std::vector<float>& v) {
        int sz = static_cast<int>(v.size());
        fwrite(&sz, sizeof(int), 1, fp);
        if (!v.empty()) fwrite(v.data(), sizeof(float), v.size(), fp);
    };
    wv(f, actor_cpu_->weights1_); wv(f, actor_cpu_->biases1_);
    wv(f, actor_cpu_->weights2_); wv(f, actor_cpu_->biases2_);
    wv(f, actor_cpu_->weights3_); wv(f, actor_cpu_->biases3_);
    wv(f, critic_cpu_->weights1_); wv(f, critic_cpu_->biases1_);
    wv(f, critic_cpu_->weights2_); wv(f, critic_cpu_->biases2_);
    wv(f, critic_cpu_->weights3_); wv(f, critic_cpu_->biases3_);
    fclose(f);
#endif
}

void RLAgent::load(const std::string& path) {
#ifdef USE_LIBTORCH
    torch::load(actor_torch_, path + ".actor");
    torch::load(critic_torch_, path + ".critic");
#else
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return;
    int id, hd;
    if (fread(&id, sizeof(int), 1, f) != 1) { fclose(f); return; }
    if (fread(&hd, sizeof(int), 1, f) != 1) { fclose(f); return; }
    auto rv = [](FILE* fp, std::vector<float>& v) {
        int sz;
        if (fread(&sz, sizeof(int), 1, fp) != 1) return;
        if (sz > 0) { v.resize(sz); fread(v.data(), sizeof(float), sz, fp); }
    };
    rv(f, actor_cpu_->weights1_); rv(f, actor_cpu_->biases1_);
    rv(f, actor_cpu_->weights2_); rv(f, actor_cpu_->biases2_);
    rv(f, actor_cpu_->weights3_); rv(f, actor_cpu_->biases3_);
    rv(f, critic_cpu_->weights1_); rv(f, critic_cpu_->biases1_);
    rv(f, critic_cpu_->weights2_); rv(f, critic_cpu_->biases2_);
    rv(f, critic_cpu_->weights3_); rv(f, critic_cpu_->biases3_);
    fclose(f);
#endif
}

} // namespace rl_sph
