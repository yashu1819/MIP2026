#ifndef RL_FEATURES_H
#define RL_FEATURES_H

#include "rl_state.h"
#include "rl_graph.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace rl_sph {

// Feature dimensions (from paper Appendix C)
enum {
    VAR_FEATURE_DIM = 64,      // Variable feature dimension
    CONST_FEATURE_DIM = 32,    // Constraint feature dimension
    PERIODIC_EMBED_DIM = 32    // Periodic embedding dimension
};

// Variable features for actor network
// Features: [c_i, lb_i, ub_i, x_i, bnd_lim, PE(x_i), degree, centrality]
struct VariableFeatures {
    std::vector<float> features;  // Flattened: [n_vars * VAR_FEATURE_DIM]
    int num_vars;
    int feature_dim;

    VariableFeatures() : num_vars(0), feature_dim(VAR_FEATURE_DIM) {}

    // Get feature vector for variable i
    float* get_features(int var_idx) {
        return &features[var_idx * feature_dim];
    }

    // Get feature vector for variable i (const version)
    const float* get_features(int var_idx) const {
        return &features[var_idx * feature_dim];
    }
};

// Constraint features for critic network
// Features: [b_j, f_j, slack, degree, scaled_A_row]
struct ConstraintFeatures {
    std::vector<float> features;  // Flattened: [n_constraints * CONST_FEATURE_DIM]
    int num_constraints;
    int feature_dim;

    ConstraintFeatures() : num_constraints(0), feature_dim(CONST_FEATURE_DIM) {}

    // Get feature vector for constraint j
    float* get_features(int constr_idx) {
        return &features[constr_idx * feature_dim];
    }

    // Get feature vector for constraint j (const version)
    const float* get_features(int constr_idx) const {
        return &features[constr_idx * feature_dim];
    }
};

// Feature builder for RL-SPH neural network
class FeatureBuilder {
public:
    FeatureBuilder(const MIPProblem& mip, const BipartiteGraph& graph);

    // Build variable features
    VariableFeatures build_variable_features(
        const RLState& state,
        const std::vector<double>& scaled_A
    );

    // Build constraint features
    ConstraintFeatures build_constraint_features(
        const RLState& state,
        const std::vector<double>& scaled_A
    );

    // Build periodic embedding for a scalar value
    std::vector<float> periodic_embed(double value);

    // Scale objective coefficients to [-1, 1]
    std::vector<float> scale_objective_coeffs() const;

    // Compute variable degrees (number of constraints each var appears in)
    std::vector<int> compute_var_degrees() const;

    // Compute constraint degrees (number of variables in each constraint)
    std::vector<int> compute_constraint_degrees() const;

    // Compute variable centrality (inverse of average constraint degree)
    std::vector<float> compute_var_centrality() const;

private:
    const MIPProblem& mip_;
    const BipartiteGraph& graph_;
    int num_vars_;
    int num_constraints_;

    // Precomputed scaling factors
    float max_c_;      // max |c_i|
    float max_b_;      // max |b_j|
    float max_degree_; // max variable degree
};

// Implementation

inline FeatureBuilder::FeatureBuilder(
    const MIPProblem& mip,
    const BipartiteGraph& graph
)
    : mip_(mip)
    , graph_(graph)
    , num_vars_(mip.num_cols)
    , num_constraints_(mip.num_rows)
{
    // Precompute scaling factors
    max_c_ = 1e-10f;
    for (double c : mip_.c) {
        max_c_ = std::max(max_c_, static_cast<float>(std::abs(c)));
    }

    max_b_ = 1e-10f;
    for (double b : mip_.b) {
        max_b_ = std::max(max_b_, static_cast<float>(std::abs(b)));
    }

    max_degree_ = 1.0f;
    auto degrees = compute_var_degrees();
    for (int d : degrees) {
        max_degree_ = std::max(max_degree_, static_cast<float>(d));
    }
}

// inline std::vector<float> FeatureBuilder::periodic_embed(double value) {
//     // Periodic embedding: PE(z) = concat(sin(2*pi*w_i*z), cos(2*pi*w_i*z))
//     // Using frequencies w_i = 2^i for i = 0, ..., k-1
//     int k = PERIODIC_EMBED_DIM / 2;
//     std::vector<float> embedding(PERIODIC_EMBED_DIM);

//     for (int i = 0; i < k; ++i) {
//         float freq = std::pow(2.0f, static_cast<float>(i));
//         float angle = 2.0f * static_cast<float>(M_PI) * freq * static_cast<float>(value);
//         embedding[i] = std::sin(angle);
//         embedding[k + i] = std::cos(angle);
//     }

//     return embedding;
// }

inline std::vector<float> FeatureBuilder::periodic_embed(double value) {
    // Clamp value to reasonable range to prevent overflow
    double clamped_value = value;
    if (std::isnan(clamped_value) || std::isinf(clamped_value)) {
        clamped_value = 0.0;
    }
    // Clamp to [-1e6, 1e6] to prevent extreme frequencies from causing issues
    clamped_value = std::max(-1e6, std::min(1e6, clamped_value));
    
    int k = PERIODIC_EMBED_DIM / 2;
    std::vector<float> embedding(PERIODIC_EMBED_DIM, 0.0f);

    for (int i = 0; i < k; ++i) {
        float freq = std::pow(2.0f, static_cast<float>(i));
        // Cap frequency to prevent overflow
        if (freq > 1e4f) freq = 1e4f;
        
        double angle = 2.0 * M_PI * freq * clamped_value;
        // Use double for intermediate calculation, then cast to float
        double sin_val = std::sin(angle);
        double cos_val = std::cos(angle);
        
        embedding[i] = static_cast<float>(sin_val);
        embedding[k + i] = static_cast<float>(cos_val);
        
        // Safety check
        if (std::isnan(embedding[i]) || std::isinf(embedding[i])) {
            embedding[i] = 0.0f;
        }
        if (std::isnan(embedding[k + i]) || std::isinf(embedding[k + i])) {
            embedding[k + i] = 0.0f;
        }
    }

    return embedding;
}

// inline std::vector<float> FeatureBuilder::scale_objective_coeffs() const {
//     // Scale c to [-1, 1]: c_scaled = c / max(|c|)
//     std::vector<float> scaled_c(num_vars_);

//     for (int i = 0; i < num_vars_; ++i) {
//         scaled_c[i] = static_cast<float>(mip_.c[i]) / max_c_;
//     }
//     return scaled_c;
// }
inline std::vector<float> FeatureBuilder::scale_objective_coeffs() const {
    std::vector<float> scaled_c(num_vars_, 0.0f);
    float safe_max_c = (max_c_ > 1e-10f) ? max_c_ : 1.0f;
    
    for (int i = 0; i < num_vars_; ++i) {
        float val = static_cast<float>(mip_.c[i]) / safe_max_c;
        // Clamp to reasonable range
        if (std::isnan(val) || std::isinf(val)) {
            val = 0.0f;
        }
        val = std::max(-10.0f, std::min(10.0f, val));
        scaled_c[i] = val;
    }
    return scaled_c;
}

inline std::vector<int> FeatureBuilder::compute_var_degrees() const {
    std::vector<int> degrees(num_vars_, 0);
    for (int i = 0; i < num_vars_; ++i) {
        degrees[i] = static_cast<int>(graph_.var_to_constraints[i].size());
    }
    return degrees;
}

inline std::vector<int> FeatureBuilder::compute_constraint_degrees() const {
    std::vector<int> degrees(num_constraints_, 0);
    for (int j = 0; j < num_constraints_; ++j) {
        degrees[j] = static_cast<int>(graph_.constraint_to_vars[j].size());
    }
    return degrees;
}

inline std::vector<float> FeatureBuilder::compute_var_centrality() const {
    // Centrality: inverse of average degree of neighboring constraints
    std::vector<float> centrality(num_vars_, 0.0f);
    auto constr_degrees = compute_constraint_degrees();

    for (int i = 0; i < num_vars_; ++i) {
        const auto& constraints = graph_.var_to_constraints[i];
        if (constraints.empty()) {
            centrality[i] = 1.0f;  // Isolated variable
            continue;
        }

        double avg_constr_degree = 0.0;
        for (int constr_idx : constraints) {
            avg_constr_degree += constr_degrees[constr_idx];
        }
        avg_constr_degree /= constraints.size();

        // Higher centrality = appears in fewer/lower-degree constraints
        centrality[i] = (avg_constr_degree > 0) ?
            static_cast<float>(1.0 / avg_constr_degree) : 1.0f;
    }

    // Normalize to [0, 1]
    float max_cent = 1e-10f;
    for (float c : centrality) {
        max_cent = std::max(max_cent, c);
    }
    for (int i = 0; i < num_vars_; ++i) {
        centrality[i] /= max_cent;
    }

    return centrality;
}

// inline VariableFeatures FeatureBuilder::build_variable_features(
//     const RLState& state,
//     const std::vector<double>& scaled_A
// ) {
//     (void)scaled_A;
//     VariableFeatures vf;
//     vf.num_vars = num_vars_;
//     vf.features.resize(num_vars_ * VAR_FEATURE_DIM, 0.0f);

//     auto scaled_c = scale_objective_coeffs();
//     auto var_degrees = compute_var_degrees();
//     auto var_centrality = compute_var_centrality();

//     for (int i = 0; i < num_vars_; ++i) {
//         float* f = vf.get_features(i);
//         int idx = 0;

//         // Feature 0: Scaled objective coefficient c_i / max(|c|)
//         f[idx++] = scaled_c[i];

//         // Feature 1: Normalized lower bound
//         f[idx++] = static_cast<float>(mip_.lb[i]) / (max_b_ + 1e-10f);

//         // Feature 2: Normalized upper bound
//         f[idx++] = static_cast<float>(mip_.ub[i]) / (max_b_ + 1e-10f);

//         // Feature 3: Current value (normalized)
//         f[idx++] = static_cast<float>(state.x[i]) / (max_b_ + 1e-10f);

//         // Feature 4: Bound indicator (1 if at or beyond bound)
//         f[idx++] = static_cast<float>(
//             (state.x[i] <= mip_.lb[i] + 1e-6) ||
//             (state.x[i] >= mip_.ub[i] - 1e-6)
//         );

//         // Feature 5-36: Periodic embedding of x_i (32 dimensions)
//         auto pe = periodic_embed(state.x[i]);
//         for (int j = 0; j < PERIODIC_EMBED_DIM; ++j) {
//             f[idx++] = pe[j];
//         }

//         // Feature 37: Normalized degree
//         f[idx++] = static_cast<float>(var_degrees[i]) / max_degree_;

//         // Feature 38: Centrality
//         f[idx++] = var_centrality[i];

//         // Feature 39: Variable type indicator (1 if integer/binary)
//         f[idx++] = (mip_.vartype[i] != VarType::CONTINUOUS) ? 1.0f : 0.0f;

//         // Feature 40: Binary indicator (1 if binary)
//         f[idx++] = (mip_.vartype[i] == VarType::BINARY) ? 1.0f : 0.0f;

//         // Features 41-63: Reserved for future use (zero-padded)
//         // Can be used for additional features like:
//         // - Reduced cost (if available from LP solver)
//         // - Pseudocost estimates
//         // - Conflict graph information
//         while (idx < VAR_FEATURE_DIM) {
//             f[idx++] = 0.0f;
//         }
//     }

//     return vf;
// }

inline VariableFeatures FeatureBuilder::build_variable_features(
    const RLState& state,
    const std::vector<double>& scaled_A
) {
    (void)scaled_A;
    VariableFeatures vf;
    vf.num_vars = num_vars_;
    vf.features.resize(num_vars_ * VAR_FEATURE_DIM, 0.0f);

    auto scaled_c = scale_objective_coeffs();
    auto var_degrees = compute_var_degrees();
    auto var_centrality = compute_var_centrality();

    // Safety: Pre-compute global bounds for clamping
    const float MAX_FEATURE_VAL = 10.0f;
    const float MIN_FEATURE_VAL = -10.0f;
    const float EPS = 1e-8f;

    for (int i = 0; i < num_vars_; ++i) {
        float* f = vf.get_features(i);
        int idx = 0;

        // Helper lambda to safely set values
        auto set_safe = [&](float val) {
            if (std::isnan(val) || std::isinf(val)) {
                f[idx++] = 0.0f;
            } else {
                f[idx++] = std::max(MIN_FEATURE_VAL, std::min(MAX_FEATURE_VAL, val));
            }
        };

        // Feature 0: Scaled objective coefficient c_i / max(|c|)
        float c_val = scaled_c[i];
        set_safe(c_val);

        // Feature 1: Normalized lower bound
        float lb_val = static_cast<float>(mip_.lb[i]) / (max_b_ + EPS);
        set_safe(lb_val);

        // Feature 2: Normalized upper bound
        float ub_val = static_cast<float>(mip_.ub[i]);
        // Handle infinite bounds (common in ILP)
        if (std::isinf(ub_val) || ub_val > 1e30f) {
            ub_val = 1e6f;  // Large but finite
        }
        ub_val = ub_val / (max_b_ + EPS);
        set_safe(ub_val);

        // Feature 3: Current value (normalized)
        float x_val = static_cast<float>(state.x[i]);
        if (std::isinf(x_val) || std::isnan(x_val)) {
            x_val = 0.0f;
        }
        x_val = x_val / (max_b_ + EPS);
        set_safe(x_val);

        // Feature 4: Bound indicator (1 if at or beyond bound)
        bool at_lower = (state.x[i] <= mip_.lb[i] + 1e-6);
        bool at_upper = false;
        if (mip_.ub[i] < 1e30f) {  // Only check if upper bound is finite
            at_upper = (state.x[i] >= mip_.ub[i] - 1e-6);
        }
        float bound_flag = (at_lower || at_upper) ? 1.0f : 0.0f;
        set_safe(bound_flag);

        // Feature 5-36: Periodic embedding of x_i (32 dimensions)
        auto pe = periodic_embed(state.x[i]);
        for (int j = 0; j < PERIODIC_EMBED_DIM; ++j) {
            float pe_val = pe[j];
            if (std::isnan(pe_val) || std::isinf(pe_val)) {
                pe_val = 0.0f;
            }
            pe_val = std::max(-1.0f, std::min(1.0f, pe_val));  // PE should be in [-1,1]
            f[idx++] = pe_val;
        }

        // Feature 37: Normalized degree
        float degree_val = static_cast<float>(var_degrees[i]) / (max_degree_ + EPS);
        set_safe(degree_val);

        // Feature 38: Centrality
        float cent_val = var_centrality[i];
        if (std::isnan(cent_val) || std::isinf(cent_val)) {
            cent_val = 0.0f;
        }
        cent_val = std::max(0.0f, std::min(1.0f, cent_val));
        set_safe(cent_val);

        // Feature 39: Variable type indicator (1 if integer/binary)
        float type_val = (mip_.vartype[i] != VarType::CONTINUOUS) ? 1.0f : 0.0f;
        set_safe(type_val);

        // Feature 40: Binary indicator (1 if binary)
        float binary_val = (mip_.vartype[i] == VarType::BINARY) ? 1.0f : 0.0f;
        set_safe(binary_val);

        // Features 41-63: Reserved (zero-padded)
        while (idx < VAR_FEATURE_DIM) {
            f[idx++] = 0.0f;
        }
    }

    // Optional: Global feature normalization for the entire tensor
    float global_max = 0.0f;
    for (size_t i = 0; i < vf.features.size(); ++i) {
        global_max = std::max(global_max, std::abs(vf.features[i]));
    }
    if (global_max > 10.0f) {
        float scale = 10.0f / global_max;
        for (size_t i = 0; i < vf.features.size(); ++i) {
            vf.features[i] *= scale;
        }
    }

    return vf;
}

inline ConstraintFeatures FeatureBuilder::build_constraint_features(
    const RLState& state,
    const std::vector<double>& /*scaled_A*/
) {
    ConstraintFeatures cf;
    cf.num_constraints = num_constraints_;
    cf.features.resize(num_constraints_ * CONST_FEATURE_DIM, 0.0f);

    auto constr_degrees = compute_constraint_degrees();
    float max_degree = 1.0f;
    for (int d : constr_degrees) {
        max_degree = std::max(max_degree, static_cast<float>(d));
    }

    for (int j = 0; j < num_constraints_; ++j) {
        float* f = cf.get_features(j);
        int idx = 0;

        // Feature 0: Normalized RHS b_j
        f[idx++] = static_cast<float>(mip_.b[j]) / (max_b_ + 1e-10f);

        // Feature 1: Constraint satisfaction f_j = b_j - Ax_j (normalized)
        f[idx++] = static_cast<float>(state.f[j]) / (max_b_ + 1e-10f);

        // Feature 2: Slack/surplus (positive if satisfied)
        f[idx++] = (state.f[j] >= 0) ? 1.0f : 0.0f;

        // Feature 3: Violation magnitude (if violated)
        f[idx++] = (state.f[j] < 0) ?
            static_cast<float>(-state.f[j]) / (max_b_ + 1e-10f) : 0.0f;

        // Feature 4: Normalized degree
        f[idx++] = static_cast<float>(constr_degrees[j]) / max_degree;

        // Feature 5: Constraint type indicator
        // (could be extended for >=, =, range constraints)
        f[idx++] = 0.0f;  // All converted to <= in current implementation

        // Features 6-31: Scaled A row (sparse encoding)
        // For each variable in constraint, encode (var_idx, scaled_coeff)
        // This is a simplified dense encoding; sparse would be more efficient
        const auto& vars = graph_.constraint_to_vars[j];
        int max_vars_to_encode = std::min(static_cast<int>(vars.size()),
                                          CONST_FEATURE_DIM - 6);
        for (int k = 0; k < max_vars_to_encode; ++k) {
            int var_idx = vars[k];
            (void)var_idx;
            // Find coefficient in scaled_A (simplified - would need proper lookup)
            float coeff = 1.0f;  // Placeholder
            f[idx++] = coeff;
        }

        // Zero-pad remaining features
        while (idx < CONST_FEATURE_DIM) {
            f[idx++] = 0.0f;
        }
    }

    return cf;
}

// Convenience function to build all features
struct AllFeatures {
    VariableFeatures var_features;
    ConstraintFeatures constr_features;
};

inline AllFeatures build_all_features(
    const MIPProblem& mip,
    const BipartiteGraph& graph,
    const RLState& state
) {
    FeatureBuilder builder(mip, graph);

    // Placeholder for scaled_A - would be computed from equilibration scaling
    std::vector<double> scaled_A(graph.scaled_A_row.size());
    for (size_t i = 0; i < scaled_A.size(); ++i) {
        scaled_A[i] = graph.scaled_A_row[i];  // Already scaled in build_graph
    }

    AllFeatures features;
    features.var_features = builder.build_variable_features(state, scaled_A);
    features.constr_features = builder.build_constraint_features(state, scaled_A);

    return features;
}

} // namespace rl_sph

#endif // RL_FEATURES_H
