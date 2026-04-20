#ifndef RL_AGENT_H
#define RL_AGENT_H

#include "rl_state.h"
#include "rl_graph.h"
#include "rl_features.h"
#include <vector>
#include <random>

namespace rl_sph {

// Neural network agent for RL-SPH
// Uses Transformer-based GNN architecture (Section 3.2 of paper)
//
// NOTE: This is a placeholder for future LibTorch implementation.
// The current implementation uses random actions. To enable full
// neural network functionality:
// 1. Install LibTorch: https://pytorch.org/get-started/locally/
// 2. Uncomment LibTorch includes in rl_agent.cpp
// 3. Implement forward pass in ActorNetwork and CriticNetwork

struct AgentConfig {
    int input_dim = 64;       // Input feature dimension
    int hidden_dim = 128;     // Hidden layer dimension
    int num_heads = 4;        // Number of attention heads
    int num_layers = 4;       // Number of transformer layers
    float dropout = 0.1f;     // Dropout rate
    float learning_rate = 1e-4f;
    int seed = 42;
};

// Actor network: outputs action probabilities
// Input: (scaled_A, x_features, bnd_lim)
// Output: probability distribution over {decrease, no_change, increase}
class ActorNetwork {
public:
    ActorNetwork(int num_vars, int num_constraints, const AgentConfig& config);

    // Forward pass: returns action logits for each variable
    // logits[i] = [P(decrease), P(no_change), P(increase)] for var i
    std::vector<std::vector<float> > forward(
        const std::vector<double>& scaled_A,
        const std::vector<double>& x,
        const std::vector<int>& bnd_lim,
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
        const std::vector<Action>& actions,
        const std::vector<int>& changeable_indices
    );

private:
    int num_vars_;
    int num_constraints_;
    AgentConfig config_;

public:
    // Network weights (3-layer MLP) - public for training access
    std::vector<float> weights1_;  // input -> hidden1
    std::vector<float> biases1_;
    std::vector<float> weights2_;  // hidden1 -> hidden2
    std::vector<float> biases2_;
    std::vector<float> weights3_;  // hidden2 -> output (3 actions)
    std::vector<float> biases3_;

    // Adam optimizer state
    std::vector<float> m1_, v1_, m2_, v2_, m3_, v3_;
};

// Critic network: estimates state value V(s)
// Input: (phase_feature, obj_encoded, f_scaled)
// Output: scalar value estimate
class CriticNetwork {
public:
    CriticNetwork(int num_vars, int num_constraints, const AgentConfig& config);

    // Forward pass: returns value estimate
    float forward(
        int phase,
        double obj,
        const std::vector<double>& f,
        const std::vector<double>& b
    );

private:
    int num_vars_;
    int num_constraints_;
    AgentConfig config_;

public:
    // Network weights (3-layer MLP) - public for training access
    std::vector<float> weights1_;  // input -> hidden1
    std::vector<float> biases1_;
    std::vector<float> weights2_;  // hidden1 -> hidden2
    std::vector<float> biases2_;
    std::vector<float> weights3_;  // hidden2 -> output (scalar value)
    std::vector<float> biases3_;
};

// Periodic embedding for numerical features (from paper)
// PE(z) = concat(sin(z_tilde), cos(z_tilde))
// where z_tilde = [2*pi*w1*z, ..., 2*pi*wk*z]
class PeriodicEmbedding {
public:
    PeriodicEmbedding(int num_frequencies = 16);

    std::vector<float> embed(double value);

private:
    int num_frequencies_;
    std::vector<float> frequencies_;  // trainable w_i
};

// Full RL agent combining actor and critic
class RLAgent {
public:
    RLAgent(int num_vars, int num_constraints, const AgentConfig& config = AgentConfig());

    // Set MIP problem reference (required before using select_actions)
    void set_mip_problem(const MIPProblem& mip) { mip_ptr_ = &mip; }

    // Select actions for given state
    std::vector<Action> select_actions(
        const RLState& state,
        const BipartiteGraph& graph,
        const std::vector<int>& changeable_indices,
        std::mt19937& rng
    );

    // Estimate value of state
    float estimate_value(
        const RLState& state,
        int phase
    );

    // Update networks (training step)
    void update(
        const std::vector<RLState>& states,
        const std::vector<std::vector<Action> >& actions,
        const std::vector<double>& rewards,
        const std::vector<int>& phases,
        float learning_rate
    );

    // Save/load model weights
    void save(const std::string& path);
    void load(const std::string& path);

private:
    const MIPProblem* mip_ptr_;  // Pointer to allow copy/move
    ActorNetwork actor_;
    CriticNetwork critic_;
    PeriodicEmbedding embedding_;
    AgentConfig config_;
};

} // namespace rl_sph

#endif // RL_AGENT_H
