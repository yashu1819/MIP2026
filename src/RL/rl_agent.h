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
// Graph Neural Network with bipartite message passing (RL-SPH paper Section 3.2)

// ------------------- Graph Attention Layer for Bipartite Graphs -------------------
// Message passing: variables attend to neighboring constraints via graph edges
// Uses multi-head attention mechanism
// FIX: Now returns both updated variable AND constraint features for bidirectional message passing

struct BipartiteGraphAttentionLayerImpl : torch::nn::Module {
    torch::nn::Linear var_linear{nullptr};       // Transform variable features
    torch::nn::Linear constr_linear{nullptr};    // Transform constraint features
    torch::nn::Linear attention_query{nullptr};  // Query projection for attention
    torch::nn::Linear attention_key{nullptr};    // Key projection for attention
    torch::nn::Linear attention_value{nullptr};  // Value projection for output
    // Reverse attention: constraints attend to variables
    torch::nn::Linear reverse_query{nullptr};
    torch::nn::Linear reverse_key{nullptr};
    torch::nn::Linear reverse_value{nullptr};

    int hidden_dim;
    int num_heads;
    int head_dim;

    BipartiteGraphAttentionLayerImpl(int var_dim, int constr_dim, int hidden_dim_, int num_heads_)
        : hidden_dim(hidden_dim_), num_heads(num_heads_), head_dim(hidden_dim_ / num_heads_)
    {
        // Feature transformation layers
        var_linear = register_module("var_linear", torch::nn::Linear(var_dim, hidden_dim));
        constr_linear = register_module("constr_linear", torch::nn::Linear(constr_dim, hidden_dim));

        // Multi-head attention projections (variables <- constraints)
        attention_query = register_module("attention_query", torch::nn::Linear(hidden_dim, hidden_dim));
        attention_key = register_module("attention_key", torch::nn::Linear(hidden_dim, hidden_dim));
        attention_value = register_module("attention_value", torch::nn::Linear(hidden_dim, hidden_dim));

        // Reverse attention (constraints <- variables)
        reverse_query = register_module("reverse_query", torch::nn::Linear(hidden_dim, hidden_dim));
        reverse_key = register_module("reverse_key", torch::nn::Linear(hidden_dim, hidden_dim));
        reverse_value = register_module("reverse_value", torch::nn::Linear(hidden_dim, hidden_dim));

        reset_parameters();
    }

    void reset_parameters() {
        torch::nn::init::xavier_uniform_(var_linear->weight);
        torch::nn::init::constant_(var_linear->bias, 0);
        torch::nn::init::xavier_uniform_(constr_linear->weight);
        torch::nn::init::constant_(constr_linear->bias, 0);
        torch::nn::init::xavier_uniform_(attention_query->weight);
        torch::nn::init::constant_(attention_query->bias, 0);
        torch::nn::init::xavier_uniform_(attention_key->weight);
        torch::nn::init::constant_(attention_key->bias, 0);
        torch::nn::init::xavier_uniform_(attention_value->weight);
        torch::nn::init::constant_(attention_value->bias, 0);
        torch::nn::init::xavier_uniform_(reverse_query->weight);
        torch::nn::init::constant_(reverse_query->bias, 0);
        torch::nn::init::xavier_uniform_(reverse_key->weight);
        torch::nn::init::constant_(reverse_key->bias, 0);
        torch::nn::init::xavier_uniform_(reverse_value->weight);
        torch::nn::init::constant_(reverse_value->bias, 0);
    }

    // Forward pass with BIDIRECTIONAL bipartite graph message passing
    // var_features: (num_vars, var_dim)
    // constr_features: (num_constraints, constr_dim)
    // edge_index: (2, num_edges) - [var_idx; constr_idx] pairs
    // edge_weights: (num_edges,) - optional weights from constraint matrix
    // Returns: tuple of (updated_var_features, updated_constr_features)
    std::tuple<torch::Tensor, torch::Tensor> forward_bidirectional(
        torch::Tensor var_features,
        torch::Tensor constr_features,
        torch::Tensor edge_index,  // [2, num_edges]: row 0 = var indices, row 1 = constr indices
        const torch::Tensor& edge_weights = torch::Tensor()  // optional (num_edges,) weights
    ) {
        int num_vars = var_features.size(0);
        int num_constraints = constr_features.size(0);
        int num_edges = edge_index.size(1);
        (void)num_edges;

        // Transform features to hidden dimension
        auto var_h = torch::relu(var_linear->forward(var_features));    // (num_vars, hidden_dim)
        auto constr_h = torch::relu(constr_linear->forward(constr_features));  // (num_constraints, hidden_dim)

        // Get edge indices
        auto row = edge_index[0];  // (num_edges,) - variable indices
        auto col = edge_index[1];  // (num_edges,) - constraint indices

        // ========== FORWARD: Variables <- Constraints ==========
        auto var_updated = compute_attention_message(
            var_h, constr_h, row, col, num_vars,
            attention_query, attention_key, attention_value,
            edge_weights
        );

        // ========== REVERSE: Constraints <- Variables ==========
        auto constr_updated = compute_attention_message(
            constr_h, var_h, col, row, num_constraints,
            reverse_query, reverse_key, reverse_value,
            edge_weights  // same weights apply (undirected graph)
        );

        return std::make_tuple(var_updated, constr_updated);
    }

    // Helper: compute attention-based message passing with edge weights
    // target_h: (num_targets, hidden_dim)
    // source_h: (num_sources, hidden_dim)
    // target_idx, source_idx: (num_edges,) - edge connectivity
    // edge_weights: (num_edges,) - optional edge weights from A matrix
    // num_targets: total number of target nodes
    torch::Tensor compute_attention_message(
        const torch::Tensor& target_h,
        const torch::Tensor& source_h,
        const torch::Tensor& target_idx,
        const torch::Tensor& source_idx,
        int num_targets,
        torch::nn::Linear& query_proj,
        torch::nn::Linear& key_proj,
        torch::nn::Linear& value_proj,
        const torch::Tensor& edge_weights = torch::Tensor()  // optional (num_edges,) weights
    ) {
        int num_edges = target_idx.numel();

        if (num_edges == 0 || num_targets == 0) {
            return target_h;
        }

        // Compute query (from targets) and key/value (from sources)
        auto query = query_proj->forward(target_h);      // (num_targets, hidden_dim)
        auto key = key_proj->forward(source_h);          // (num_sources, hidden_dim)
        auto value = value_proj->forward(source_h);      // (num_sources, hidden_dim)

        // Gather features for edges
        auto query_per_edge = query.index_select(0, target_idx);   // (num_edges, hidden_dim)
        auto key_per_edge = key.index_select(0, source_idx);       // (num_edges, hidden_dim)
        auto value_per_edge = value.index_select(0, source_idx);   // (num_edges, hidden_dim)

        // Compute attention score per edge: dot product of query and key
        auto edge_scores = (query_per_edge * key_per_edge).sum(1, /*keepdim=*/true);  // (num_edges, 1)

        // FIX: Incorporate edge weights into attention scores
        // Edge weights bias the attention - higher weight = more attention
        if (edge_weights.defined() && edge_weights.numel() == num_edges) {
            edge_scores = edge_scores + edge_weights.view({-1, 1});
        }

        // // Apply softmax per target node (over its neighboring sources)
        // auto attention_weights = torch::zeros_like(edge_scores);  // (num_edges, 1)

        // for (int t = 0; t < num_targets; ++t) {
        //     auto target_mask = (target_idx == t);  // (num_edges,) boolean mask
        //     auto target_edge_indices = torch::nonzero(target_mask).squeeze(-1);

        //     if (target_edge_indices.numel() > 0) {
        //         auto target_edge_scores = edge_scores.index_select(0, target_edge_indices);
        //         auto target_edge_weights = torch::softmax(target_edge_scores, 0);
        //         attention_weights = attention_weights.scatter(0, target_edge_indices, target_edge_weights);
        //     }
        // }

        // Compute exp scores
        auto exp_scores = torch::exp(edge_scores);  // (num_edges, 1)
        // Sum per target node
        auto denom = torch::zeros(
            {num_targets, 1},
            torch::TensorOptions().dtype(edge_scores.dtype()).device(edge_scores.device())
        );
        // Ensure index is long and correct shape
        auto idx = target_idx.to(torch::kLong).reshape({-1, 1});
        // Aggregate exp_scores per target
        denom.scatter_add_(0, idx, exp_scores);
        // Gather denominator per edge
        auto denom_per_edge = denom.index_select(0, target_idx.to(torch::kLong));
        // Final attention weights
        auto attention_weights = exp_scores / (denom_per_edge + 1e-8);


        // Weighted sum of values
        auto weighted_values = value_per_edge * attention_weights;  // (num_edges, hidden_dim)

        // Aggregate messages by summing for each target
        auto output = torch::zeros({num_targets, hidden_dim},torch::TensorOptions().dtype(target_h.dtype()).device(target_h.device()));
        output.scatter_add_(0,
            target_idx.view({-1, 1}).expand({-1, hidden_dim}),
            weighted_values
        );

        return output;
    }

    // Legacy forward for backward compatibility (returns only variable features)
    torch::Tensor forward(
        torch::Tensor var_features,
        torch::Tensor constr_features,
        torch::Tensor edge_index
    ) {
        auto [var_out, constr_out] = forward_bidirectional(var_features, constr_features, edge_index);
        return var_out;
    }
};
TORCH_MODULE(BipartiteGraphAttentionLayer);

// ------------------- Actor Network: GAT-based -------------------
// Input: Variable features (64-dim) + Constraint features (32-dim) + Graph edges
// Output: Action logits for each variable (3 actions)
// FIX: Now uses bidirectional message passing with evolving constraint embeddings
// FIX: Multi-layer propagation loop for deeper reasoning

struct ActorNetworkTorchImpl : torch::nn::Module {
    torch::nn::Linear var_embedding{nullptr};       // Initial variable feature embedding
    torch::nn::Linear constr_embedding{nullptr};    // Initial constraint feature embedding

    // GAT layer for multi-layer propagation (shared across layers)
    BipartiteGraphAttentionLayer gat_layer{nullptr};

    torch::nn::Linear var_transform{nullptr};       // Post-GAT variable transformation
    torch::nn::Linear action_head{nullptr};         // Output action logits
    torch::nn::Linear edge_weight_encoder{nullptr}; // Encodes A matrix coefficients as edge weights

    int hidden_dim;
    int num_layers;
    float dropout_rate;

    ActorNetworkTorchImpl(int var_input_dim, int constr_input_dim, int hidden_dim_, int num_heads, int num_layers_)
        : hidden_dim(hidden_dim_), num_layers(num_layers_), dropout_rate(0.1f)
    {
        // Initial embeddings
        var_embedding = register_module("var_embedding", torch::nn::Linear(var_input_dim, hidden_dim));
        constr_embedding = register_module("constr_embedding", torch::nn::Linear(constr_input_dim, hidden_dim));

        // Edge weight encoder: maps scalar coefficient to hidden_dim bias for attention
        edge_weight_encoder = register_module("edge_weight_encoder", torch::nn::Linear(1, hidden_dim));

        // Single GAT layer used in multi-layer propagation loop
        gat_layer = register_module("gat_layer", BipartiteGraphAttentionLayer(hidden_dim, hidden_dim, hidden_dim, num_heads));

        // Post-GAT transformation
        var_transform = register_module("var_transform", torch::nn::Linear(hidden_dim, hidden_dim));

        // Action head: outputs 3 logits per variable
        action_head = register_module("action_head", torch::nn::Linear(hidden_dim, 3));

        reset_parameters();
    }

    void reset_parameters() {
        torch::nn::init::xavier_uniform_(var_embedding->weight);
        torch::nn::init::constant_(var_embedding->bias, 0);
        torch::nn::init::xavier_uniform_(constr_embedding->weight);
        torch::nn::init::constant_(constr_embedding->bias, 0);
        torch::nn::init::xavier_uniform_(edge_weight_encoder->weight);
        torch::nn::init::constant_(edge_weight_encoder->bias, 0);
        torch::nn::init::xavier_uniform_(var_transform->weight);
        torch::nn::init::constant_(var_transform->bias, 0);
        torch::nn::init::xavier_uniform_(action_head->weight);
        torch::nn::init::constant_(action_head->bias, 0);
    }

    // Forward pass with BIDIRECTIONAL graph message passing
    // FIX: Both variable AND constraint features evolve across layers
    // FIX: Dropout now controlled by training flag
    // FIX: Edge weights from A matrix incorporated into attention
    // FIX: Multi-layer propagation loop for deeper reasoning (configurable num_layers)
    // var_features: (num_vars, var_input_dim)
    // constr_features: (num_constraints, constr_input_dim)
    // edge_index: (2, num_edges) - bipartite graph edges [var_idx; constr_idx]
    // edge_weights: (num_edges,) - optional coefficients from constraint matrix
    // changeable_mask: (num_vars,) - boolean mask for changeable variables
    // is_training: whether to apply dropout
    // Returns: (num_changeable, 3) action logits
    torch::Tensor forward(
        torch::Tensor var_features,
        torch::Tensor constr_features,
        torch::Tensor edge_index,
        torch::Tensor changeable_mask,
        bool is_training = false,
        const torch::Tensor& edge_weights = torch::Tensor()  // optional (num_edges,)
    ) {
        // Initial embeddings
        auto var_h = torch::relu(var_embedding->forward(var_features));  // (num_vars, hidden_dim)
        auto constr_h = torch::relu(constr_embedding->forward(constr_features));  // (num_constraints, hidden_dim)

        // Encode edge weights if provided
        torch::Tensor encoded_weights;
        if (edge_weights.defined() && edge_weights.numel() > 0) {
            // Encode scalar weights to hidden_dim for attention bias
            encoded_weights = edge_weight_encoder->forward(edge_weights.view({-1, 1}));
            // Sum across hidden dim to get scalar bias per edge
            encoded_weights = encoded_weights.sum(1);  // (num_edges,)
        }

        // FIX: Multi-layer propagation loop - iterative V<->C updates
        // Each iteration performs bidirectional message passing
        for (int layer = 0; layer < num_layers; ++layer) {
            std::tie(var_h, constr_h) = gat_layer->forward_bidirectional(var_h, constr_h, edge_index, encoded_weights);
            var_h = torch::dropout(var_h, dropout_rate, is_training);
        }

        // Post-GAT transformation
        var_h = torch::relu(var_transform->forward(var_h));  // (num_vars, hidden_dim)

        // Action logits for all variables
        auto all_logits = action_head->forward(var_h);  // (num_vars, 3)

        // Extract logits for changeable variables only
        auto changeable_indices = torch::nonzero(changeable_mask).squeeze(-1);  // (num_changeable,)
        if (changeable_indices.numel() == 0) {
            return torch::zeros({0, 3},torch::TensorOptions().dtype(var_features.dtype()).device(var_features.device()));
        }
        auto logits = all_logits.index_select(0, changeable_indices);  // (num_changeable, 3)

        return logits;
    }
};
TORCH_MODULE(ActorNetworkTorch);

// ------------------- Critic Network: GNN-based -------------------
// Input: Variable features + Constraint features + Graph structure + Phase + Objective
// Output: Scalar value estimate V(s)

struct CriticNetworkTorchImpl : torch::nn::Module {
    torch::nn::Linear var_embedding{nullptr};       // Variable feature embedding
    torch::nn::Linear constr_embedding{nullptr};    // Constraint feature embedding

    // Graph convolution layers for critic
    torch::nn::Linear var_gcn1{nullptr};
    torch::nn::Linear var_gcn2{nullptr};
    torch::nn::Linear constr_gcn1{nullptr};
    torch::nn::Linear constr_gcn2{nullptr};

    torch::nn::Embedding phase_embedding{nullptr};  // Phase encoding (2 phases)
    torch::nn::Linear obj_encoder{nullptr};         // Objective encoding

    // Value head
    torch::nn::Linear value_fc1{nullptr};
    torch::nn::Linear value_fc2{nullptr};

    int hidden_dim;

    CriticNetworkTorchImpl(int var_input_dim, int constr_input_dim, int hidden_dim_)
        : hidden_dim(hidden_dim_)
    {
        // Initial embeddings
        var_embedding = register_module("var_embedding", torch::nn::Linear(var_input_dim, hidden_dim));
        constr_embedding = register_module("constr_embedding", torch::nn::Linear(constr_input_dim, hidden_dim));

        // Graph convolution layers (simplified GCN-style)
        var_gcn1 = register_module("var_gcn1", torch::nn::Linear(hidden_dim, hidden_dim));
        var_gcn2 = register_module("var_gcn2", torch::nn::Linear(hidden_dim, hidden_dim));
        constr_gcn1 = register_module("constr_gcn1", torch::nn::Linear(hidden_dim, hidden_dim));
        constr_gcn2 = register_module("constr_gcn2", torch::nn::Linear(hidden_dim, hidden_dim));

        // Phase and objective encodings
        phase_embedding = register_module("phase_embedding", torch::nn::Embedding(2, hidden_dim));
        obj_encoder = register_module("obj_encoder", torch::nn::Linear(1, hidden_dim));

        // Value head (concatenates 4 vectors: var_pooled, constr_pooled, phase_emb, obj_emb)
        value_fc1 = register_module("value_fc1", torch::nn::Linear(hidden_dim * 4, hidden_dim));
        value_fc2 = register_module("value_fc2", torch::nn::Linear(hidden_dim, 1));

        reset_parameters();
    }

    void reset_parameters() {
        torch::nn::init::xavier_uniform_(var_embedding->weight);
        torch::nn::init::constant_(var_embedding->bias, 0);
        torch::nn::init::xavier_uniform_(constr_embedding->weight);
        torch::nn::init::constant_(constr_embedding->bias, 0);
        torch::nn::init::xavier_uniform_(var_gcn1->weight);
        torch::nn::init::constant_(var_gcn1->bias, 0);
        torch::nn::init::xavier_uniform_(var_gcn2->weight);
        torch::nn::init::constant_(var_gcn2->bias, 0);
        torch::nn::init::xavier_uniform_(constr_gcn1->weight);
        torch::nn::init::constant_(constr_gcn1->bias, 0);
        torch::nn::init::xavier_uniform_(constr_gcn2->weight);
        torch::nn::init::constant_(constr_gcn2->bias, 0);
        torch::nn::init::xavier_uniform_(obj_encoder->weight);
        torch::nn::init::constant_(obj_encoder->bias, 0);
        torch::nn::init::xavier_uniform_(value_fc1->weight);
        torch::nn::init::constant_(value_fc1->bias, 0);
        torch::nn::init::xavier_uniform_(value_fc2->weight);
        torch::nn::init::constant_(value_fc2->bias, 0);
    }

    // Graph convolution helper: aggregate neighbor messages
    // source_features: (num_sources, hidden_dim)
    // source_indices: (num_edges,) - indices of source nodes for each edge
    // target_indices: (num_edges,) - indices of target nodes for each edge
    // num_targets: total number of target nodes
    // transform: linear layer to apply before aggregation
    torch::Tensor graph_conv(
        torch::Tensor source_features,
        torch::Tensor source_indices,
        torch::Tensor target_indices,
        int num_targets,
        torch::nn::Linear& transform
    ) {
        if (num_targets == 0 || source_indices.numel() == 0) {
            return torch::zeros({num_targets, hidden_dim},torch::TensorOptions().dtype(source_features.dtype()).device(source_features.device()));
        }

        // Transform source features
        auto h = torch::relu(transform->forward(source_features));  // (num_sources, hidden_dim)

        // Gather transformed features for each edge
        auto source_h = h.index_select(0, source_indices);  // (num_edges, hidden_dim)

        // Aggregate by summing messages for each target node
        auto output = torch::zeros({num_targets, h.size(1)},torch::TensorOptions().dtype(source_features.dtype()).device(source_features.device()));
        output.scatter_add_(0,
            target_indices.view({-1, 1}).expand({-1, h.size(1)}),
            source_h
        );  // (num_targets, hidden_dim)

        return output;
    }

    // Forward pass with graph convolutions
    // var_features: (num_vars, var_input_dim)
    // constr_features: (num_constraints, constr_input_dim)
    // edge_index_var2constr: (2, num_edges) - [var_idx; constr_idx]
    // edge_index_constr2var: (2, num_edges) - [constr_idx; var_idx] (reverse edges)
    // phase: 0 or 1
    // obj: normalized objective value
    // Returns: scalar value estimate
    torch::Tensor forward(
        torch::Tensor var_features,
        torch::Tensor constr_features,
        torch::Tensor edge_index_var2constr,  // [var_idx; constr_idx]
        torch::Tensor edge_index_constr2var,  // [constr_idx; var_idx]
        int phase,
        torch::Tensor obj
    ) {
        int nv = var_features.size(0);
        int nc = constr_features.size(0);

        // Initial embeddings
        auto var_h = torch::relu(var_embedding->forward(var_features));  // (num_vars, hidden_dim)
        auto constr_h = torch::relu(constr_embedding->forward(constr_features));  // (num_constraints, hidden_dim)

        // Variable graph convolutions (aggregate messages FROM constraints TO variables)
        // edge_index_var2constr[0] = var indices (targets), edge_index_var2constr[1] = constr indices (sources)
        if (edge_index_var2constr.numel() > 0) {
            auto var_msg = graph_conv(constr_h, edge_index_var2constr[1], edge_index_var2constr[0], nv, var_gcn1);
            var_h = var_h + var_msg;  // Residual connection
        }
        var_h = torch::relu(var_gcn2->forward(var_h));

        // Constraint graph convolutions (aggregate messages FROM variables TO constraints)
        // edge_index_constr2var[0] = constr indices (sources), edge_index_constr2var[1] = var indices (targets)
        if (edge_index_constr2var.numel() > 0) {
            auto constr_msg = graph_conv(var_h, edge_index_constr2var[1], edge_index_constr2var[0], nc, constr_gcn1);
            constr_h = constr_h + constr_msg;  // Residual connection
        }
        constr_h = torch::relu(constr_gcn2->forward(constr_h));

        // Global pooling: mean over all variables and constraints
        auto var_pooled = var_h.mean(0);  // (hidden_dim,)
        auto constr_pooled = (nc > 0) ? constr_h.mean(0) : torch::zeros({hidden_dim},torch::TensorOptions().dtype(var_h.dtype()).device(var_h.device()));

        // Phase encoding
        auto phase_idx = torch::tensor({phase}, torch::TensorOptions().dtype(torch::kLong).device(var_features.device()));
        std::cout << "phase_idx device: " << phase_idx.device() << " emb device: " << phase_embedding->weight.device() << std::endl;
        auto phase_emb = phase_embedding->forward(phase_idx).squeeze(0);  // (hidden_dim,)

        // Objective encoding
        auto obj_2d = obj.reshape({1, 1});
        auto obj_emb = torch::relu(obj_encoder->forward(obj_2d)).squeeze(0);  // (hidden_dim,)

        // Concatenate all embeddings: (hidden_dim * 4,)
        auto combined = torch::cat({var_pooled, constr_pooled, phase_emb, obj_emb});

        // Value head
        auto x = torch::relu(value_fc1->forward(combined.unsqueeze(0)));  // (1, hidden_dim)
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
    RLAgent(const AgentConfig& config = AgentConfig());
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

    // CPU fallback networks (always available)
    ActorNetworkCPU* actor_cpu_;
    CriticNetworkCPU* critic_cpu_;

#ifdef USE_LIBTORCH
    // LibTorch networks
    ActorNetworkTorch actor_torch_;
    CriticNetworkTorch critic_torch_;
    std::shared_ptr<torch::optim::RMSprop> optimizer_;
    torch::Device device_;

    // Build feature tensors for GNN
    torch::Tensor build_var_feature_tensor(
        const RLState& state,
        const BipartiteGraph& graph
    );

    torch::Tensor build_constr_feature_tensor(
        const RLState& state,
        const BipartiteGraph& graph
    );

    // Build edge index tensor for graph message passing
    torch::Tensor build_edge_index_tensor(
        const BipartiteGraph& graph
    );

    // Build edge weight tensor from constraint matrix coefficients
    torch::Tensor build_edge_weight_tensor(
        const BipartiteGraph& graph
    );

    // Build changeable mask for actor
    torch::Tensor build_changeable_mask(
        int num_vars,
        const std::vector<int>& changeable_indices
    );
#endif
};

} // namespace rl_sph

#endif // RL_AGENT_H
