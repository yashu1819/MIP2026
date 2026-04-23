#ifndef RL_AGENT_H
#define RL_AGENT_H

#include "rl_state.h"
#include "rl_graph.h"
#include "rl_features.h"
#include <vector>
#include <random>
#include <memory>

#ifdef USE_LIBTORCH
#include <torch/torch.h>
#include <torch/script.h>
#endif

namespace rl_sph {

// Neural network agent for RL-SPH
// Uses Transformer-based GNN architecture (Section 3.2 of paper)

struct AgentConfig {
    int input_dim = 64;       // Input feature dimension (from VariableFeatures)
    int hidden_dim = 128;     // Hidden layer dimension
    int num_heads = 4;        // Number of attention heads
    int num_layers = 4;       // Number of transformer layers
    float dropout = 0.1f;     // Dropout rate
    float learning_rate = 1e-4f;
    // RMSprop parameters (from paper)
    float rmsprop_alpha = 0.99f;
    float rmsprop_epsilon = 1e-5f;
    float weight_decay = 1e-3f;
    int seed = 42;
};

// Periodic embedding for numerical features (from paper)
// PE(z) = concat(sin(z_tilde), cos(z_tilde))
// where z_tilde = [2*pi*w1*z, ..., 2*pi*wk*z]
class PeriodicEmbedding {
public:
    PeriodicEmbedding(int num_frequencies = 16);

    std::vector<float> embed(double value);

#ifdef USE_LIBTORCH
    torch::Tensor embed_batch(const std::vector<double>& values);
#endif

private:
    int num_frequencies_;
    std::vector<float> frequencies_;  // w_i = 2^i
};

// Result of a forward pass during training
// Contains both the chosen actions and the data needed for gradient computation
struct TrainingForwardResult {
    std::vector<Action> actions;           // Sampled actions
    std::vector<std::vector<float> > logits; // Raw logits per variable (num_changeable x 3)
    float log_prob_sum;                    // Sum of log probs of selected actions
    float state_value;                     // Critic value V(s)
};

#ifdef USE_LIBTORCH

// ==================== LibTorch Implementations ====================

// Actor network: Transformer-based GNN
// Input: Variable features (64-dim) + positional encoding
// Output: Action logits for each variable (3 actions: decrease, no_change, increase)
struct ActorNetworkTorchImpl : torch::nn::Module {
    torch::nn::Linear var_embedding{nullptr};
    torch::nn::TransformerEncoderLayer transformer_layer{nullptr};
    torch::nn::TransformerEncoder transformer{nullptr};
    torch::nn::Linear action_head{nullptr};

    ActorNetworkTorchImpl(int input_dim, int hidden_dim, int num_heads, int num_layers, float dropout) {
        // Variable feature embedding
        var_embedding = register_module("var_embedding",
            torch::nn::Linear(input_dim, hidden_dim));

        // Transformer encoder layer
        auto layer_opts = torch::nn::TransformerEncoderLayerOptions(hidden_dim, num_heads)
            .dim_feedforward(hidden_dim * 4)
            .dropout(dropout)
            .activation(torch::kReLU);

        transformer_layer = register_module("transformer_layer",
            torch::nn::TransformerEncoderLayer(layer_opts));

        // Multi-layer transformer encoder
        auto encoder_opts = torch::nn::TransformerEncoderOptions(transformer_layer, num_layers);
        transformer = register_module("transformer",
            torch::nn::TransformerEncoder(encoder_opts));

        // Action head: outputs 3 logits per variable
        action_head = register_module("action_head",
            torch::nn::Linear(hidden_dim, 3));

        // Initialize weights
        reset_parameters();
    }

    void reset_parameters() {
        torch::nn::init::xavier_uniform_(var_embedding->weight);
        torch::nn::init::constant_(var_embedding->bias, 0);
        torch::nn::init::xavier_uniform_(action_head->weight);
        torch::nn::init::constant_(action_head->bias, 0);
    }

    // Forward pass
    // x: (num_changeable, input_dim)
    // Returns: (num_changeable, 3) action logits
    torch::Tensor forward(torch::Tensor x) {
        // Embed variable features: (num_changeable, hidden_dim)
        x = torch::relu(var_embedding->forward(x));

        // Add batch dimension for transformer: (num_changeable, 1, hidden_dim)
        x = x.unsqueeze(1);

        // Transformer encoding: (num_changeable, 1, hidden_dim)
        x = transformer->forward(x);

        // Remove batch dimension: (num_changeable, hidden_dim)
        x = x.squeeze(1);

        // Action logits: (num_changeable, 3)
        return action_head->forward(x);
    }
};
TORCH_MODULE(ActorNetworkTorch);

// Critic network: State value estimation
// Input: Phase encoding + objective + constraint features
// Output: Scalar value estimate V(s)
struct CriticNetworkTorchImpl : torch::nn::Module {
    torch::nn::Embedding phase_embedding{nullptr};
    torch::nn::Linear obj_encoder{nullptr};
    torch::nn::Linear constraint_encoder{nullptr};
    torch::nn::Linear value_fc1{nullptr};
    torch::nn::Linear value_fc2{nullptr};

    CriticNetworkTorchImpl(int num_constraints, int hidden_dim) {
        // Phase embedding (2 phases)
        phase_embedding = register_module("phase_embedding",
            torch::nn::Embedding(2, hidden_dim));

        // Objective encoder
        obj_encoder = register_module("obj_encoder",
            torch::nn::Linear(1, hidden_dim));

        // Constraint encoder: takes flattened constraint features
        // [f_j, slack_indicator, violation_magnitude] per constraint = 3 features
        int constraint_input_dim = num_constraints * 3;
        constraint_encoder = register_module("constraint_encoder",
            torch::nn::Linear(constraint_input_dim, hidden_dim));

        // Value head: concatenation of 3 embeddings -> hidden -> scalar
        value_fc1 = register_module("value_fc1",
            torch::nn::Linear(hidden_dim * 3, hidden_dim));
        value_fc2 = register_module("value_fc2",
            torch::nn::Linear(hidden_dim, 1));

        reset_parameters();
    }

    void reset_parameters() {
        torch::nn::init::xavier_uniform_(obj_encoder->weight);
        torch::nn::init::constant_(obj_encoder->bias, 0);
        torch::nn::init::xavier_uniform_(constraint_encoder->weight);
        torch::nn::init::constant_(constraint_encoder->bias, 0);
        torch::nn::init::xavier_uniform_(value_fc1->weight);
        torch::nn::init::constant_(value_fc1->bias, 0);
        torch::nn::init::xavier_uniform_(value_fc2->weight);
        torch::nn::init::constant_(value_fc2->bias, 0);
    }

    // Forward pass
    // phase: 0 or 1 (phase-1 -> index 0, phase-2 -> index 1)
    // obj: scalar normalized objective
    // constr_features: flattened (num_constraints * 3,)
    torch::Tensor forward(int phase, torch::Tensor obj, torch::Tensor constr_features) {
        // Phase encoding: (1, hidden_dim)
        auto phase_idx = torch::tensor({phase}, torch::kLong);
        auto phase_emb = phase_embedding->forward(phase_idx);  // (1, hidden_dim)

        // Objective encoding: (1, hidden_dim)
        auto obj_emb = torch::relu(obj_encoder->forward(obj.unsqueeze(0)));  // (1, hidden_dim)

        // Constraint encoding: (1, hidden_dim)
        auto constr_emb = torch::relu(
            constraint_encoder->forward(constr_features.unsqueeze(0)));  // (1, hidden_dim)

        // Combine: (1, hidden_dim * 3)
        auto combined = torch::cat({phase_emb, obj_emb, constr_emb}, /*dim=*/1);

        // Value estimation: scalar
        auto x = torch::relu(value_fc1->forward(combined));
        return value_fc2->forward(x).squeeze();  // scalar
    }
};
TORCH_MODULE(CriticNetworkTorch);

#endif  // USE_LIBTORCH

// ==================== CPU-Only Fallback Implementation ====================
// Used when LibTorch is not available (simplified 3-layer MLP)

// CPU Actor: 3-layer MLP fallback
class ActorNetworkCPU {
public:
    ActorNetworkCPU(int num_vars, int num_constraints, const AgentConfig& config);
    ~ActorNetworkCPU();

    // Forward pass: returns action logits for each changeable variable
    std::vector<std::vector<float> > forward(
        const std::vector<double>& var_features,  // Flattened: [num_changeable * VAR_FEATURE_DIM]
        const std::vector<int>& changeable_indices
    );

    // Sample actions from policy
    std::vector<Action> sample_actions(
        const std::vector<std::vector<float> >& logits,
        std::mt19937& rng
    );

    // Get log probability of taken actions
    float log_probability(
        const std::vector<std::vector<float> >& logits,
        const std::vector<Action>& actions
    );

    // MLP weights (public for save/load and training perturbation)
    std::vector<float> weights1_;
    std::vector<float> biases1_;
    std::vector<float> weights2_;
    std::vector<float> biases2_;
    std::vector<float> weights3_;
    std::vector<float> biases3_;

private:
    int num_vars_;
    int num_constraints_;
    AgentConfig config_;
};

// CPU Critic: 3-layer MLP fallback for state value estimation
class CriticNetworkCPU {
public:
    CriticNetworkCPU(int num_vars, int num_constraints, const AgentConfig& config);
    ~CriticNetworkCPU();

    // Forward pass: returns value estimate
    float forward(
        int phase,
        double obj,
        const std::vector<double>& f,
        const std::vector<double>& b
    );

    // MLP weights (public for save/load and training perturbation)
    std::vector<float> weights1_;
    std::vector<float> biases1_;
    std::vector<float> weights2_;
    std::vector<float> biases2_;
    std::vector<float> weights3_;
    std::vector<float> biases3_;

private:
    int num_vars_;
    int num_constraints_;
    AgentConfig config_;
};

// ==================== Full RL Agent ====================
// Combines actor + critic, handles both LibTorch and CPU paths

class RLAgent {
public:
    RLAgent(int num_vars, int num_constraints, const AgentConfig& config = AgentConfig());
    ~RLAgent();

    // Set MIP problem reference (required before using select_actions)
    void set_mip_problem(const MIPProblem& mip) { mip_ptr_ = &mip; }

    // Select actions for given state (inference mode — no gradient tracking)
    std::vector<Action> select_actions(
        const RLState& state,
        const BipartiteGraph& graph,
        const std::vector<int>& changeable_indices,
        std::mt19937& rng
    );

    // Select actions and return training data (training mode)
    TrainingForwardResult select_actions_training(
        const RLState& state,
        const BipartiteGraph& graph,
        const std::vector<int>& changeable_indices,
        int phase,
        std::mt19937& rng
    );

    // Estimate value of state
    float estimate_value(
        const RLState& state,
        int phase
    );

    // Update networks using collected experiences (policy gradient + TD)
    // states[i], actions[i], rewards[i] form a trajectory
    void update(
        const std::vector<RLState>& states,
        const std::vector<std::vector<Action> >& actions,
        const std::vector<double>& rewards,
        const std::vector<int>& phases,
        const std::vector<std::vector<int> >& changeable_indices_list,
        const BipartiteGraph& graph,
        float gamma
    );

    // Save/load model weights
    void save(const std::string& path);
    void load(const std::string& path);

    // Accessors
    const AgentConfig& config() const { return config_; }

private:
    const MIPProblem* mip_ptr_;
    PeriodicEmbedding* embedding_;
    AgentConfig config_;
    int num_vars_;
    int num_constraints_;

    // CPU fallback networks (always available)
    ActorNetworkCPU* actor_cpu_;
    CriticNetworkCPU* critic_cpu_;

#ifdef USE_LIBTORCH
    // LibTorch networks
    ActorNetworkTorch actor_torch_;
    CriticNetworkTorch critic_torch_;
    std::shared_ptr<torch::optim::RMSprop> optimizer_;

    // Build feature tensor for changeable variables
    torch::Tensor build_feature_tensor(
        const RLState& state,
        const BipartiteGraph& graph,
        const std::vector<int>& changeable_indices
    );

    // Build constraint feature tensor for critic
    torch::Tensor build_constraint_tensor(
        const RLState& state
    );
#endif
};

} // namespace rl_sph

#endif // RL_AGENT_H
