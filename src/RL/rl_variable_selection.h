#ifndef RL_VARIABLE_SELECTION_H
#define RL_VARIABLE_SELECTION_H

#include "rl_graph.h"
#include "rl_state.h"
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include "../mip_problem.h"

namespace rl_sph {

// Variable selection strategy (Algorithm 3)
class VariableSelector {
public:
    VariableSelector(
        const MIPProblem& mip,
        const BipartiteGraph& graph
    ) : mip_(mip), graph_(graph), rng_(std::random_device{}()) {}

    // Select variables to change at timestep t
    // Returns indices of changeable variables
    std::vector<int> select_variables(
        const RLState& state,
        int phase
    );

private:
    const MIPProblem& mip_;
    const BipartiteGraph& graph_;
    std::mt19937 rng_;

    // Sample k indices with given probabilities
    std::vector<int> sample_weighted(
        const std::vector<double>& scores,
        int k
    );

    // Get top k indices by score
    std::vector<int> top_k(
        const std::vector<double>& scores,
        int k
    );
};

// Select variables for phase 1 or phase 2 (MILP-aware: skips continuous variables)
inline std::vector<int> VariableSelector::select_variables(
    const RLState& state,
    int phase
) {
    int n = state.n;
    int m = state.m;

    // MILP: Build list of integer/binary variable indices only
    std::vector<int> integer_vars;
    integer_vars.reserve(n);
    for (int i = 0; i < n; ++i) {
        if (mip_.vartype[i] == VarType::INTEGER ||
            mip_.vartype[i] == VarType::BINARY) {
            integer_vars.push_back(i);
        }
    }

    // If no integer variables, return empty (nothing for RL to modify)
    if (integer_vars.empty()) return {};

    int n_int = static_cast<int>(integer_vars.size());

    // Number of seed and neighbor variables: p = q = log2(n_int)
    int p = static_cast<int>(std::log2(static_cast<double>(n_int)));
    int q = p;
    if (p < 1) p = 1;
    if (q < 1) q = 1;

    // Score only integer/binary variables
    // We use a map from integer_var index to score
    std::vector<double> score_seed(n, 0.0);

    if (phase == 1) {
        // Phase 1: Select variables in violated constraints
        for (int i : integer_vars) {
            double score = 0.0;
            for (int constr_idx : graph_.var_to_constraints[i]) {
                if (state.f[constr_idx] < 0) {
                    score += 1.0;
                }
            }
            score_seed[i] = score;
        }

        // Weight: prefer variables with lower objective coefficient
        double max_c = 0.0;
        for (double c : mip_.c) {
            max_c = std::max(max_c, std::abs(c));
        }
        if (max_c < 1e-10) max_c = 1.0;

        for (int i : integer_vars) {
            double weight = (max_c - std::abs(mip_.c[i]) + 1.0) / max_c;
            score_seed[i] *= weight;
        }

    } else {
        // Phase 2: Select variables in well-satisfied constraints
        for (int i : integer_vars) {
            double score = 0.0;
            for (int constr_idx : graph_.var_to_constraints[i]) {
                if (state.f[constr_idx] > 0) {
                    score += 1.0;
                }
            }
            score_seed[i] = score;
        }

        // Weight: prefer variables with higher objective coefficient
        double max_c = 0.0;
        for (double c : mip_.c) {
            max_c = std::max(max_c, std::abs(c));
        }
        if (max_c < 1e-10) max_c = 1.0;

        for (int i : integer_vars) {
            double weight = std::abs(mip_.c[i]) / max_c;
            score_seed[i] *= weight;
        }

        // Invert scores for phase 2
        double max_score = 0.0;
        for (int i : integer_vars) {
            max_score = std::max(max_score, score_seed[i]);
        }
        for (int i : integer_vars) {
            score_seed[i] = max_score - score_seed[i] + 1.0;
        }
    }

    // Zero out scores for continuous variables (safety measure)
    for (int i = 0; i < n; ++i) {
        if (mip_.vartype[i] == VarType::CONTINUOUS) {
            score_seed[i] = 0.0;
        }
    }

    // Sample p seed variables according to scores
    std::vector<int> seed_indices = sample_weighted(score_seed, p);

    // Neighbor selection: variables sharing constraints with seeds
    std::vector<int> constraint_counts(m, 0);
    for (int seed_idx : seed_indices) {
        for (int constr_idx : graph_.var_to_constraints[seed_idx]) {
            constraint_counts[constr_idx]++;
        }
    }

    // Score neighbors — only integer/binary variables
    std::vector<double> score_neighbor(n, 0.0);
    for (int i : integer_vars) {
        double score = 0.0;
        for (int constr_idx : graph_.var_to_constraints[i]) {
            score += constraint_counts[constr_idx];
        }
        score_neighbor[i] = score;
    }

    // Prevent selecting seed variables as neighbors
    for (int seed_idx : seed_indices) {
        score_neighbor[seed_idx] = -1.0;
    }

    // Zero out continuous variables from neighbor selection
    for (int i = 0; i < n; ++i) {
        if (mip_.vartype[i] == VarType::CONTINUOUS) {
            score_neighbor[i] = -1.0;
        }
    }

    // Select top q neighbors
    std::vector<int> neighbor_indices = top_k(score_neighbor, q);

    // Combine seeds and neighbors
    std::vector<int> changeable = seed_indices;
    changeable.insert(changeable.end(), neighbor_indices.begin(), neighbor_indices.end());

    return changeable;
}

// Sample k indices with probability proportional to scores
inline std::vector<int> VariableSelector::sample_weighted(
    const std::vector<double>& scores,
    int k
) {
    std::vector<int> result;
    int n = scores.size();

    // Compute total score
    double total = 0.0;
    for (double s : scores) {
        if (s > 0) total += s;
    }

    if (total < 1e-10) {
        // Uniform random if all scores are zero
        std::vector<int> indices(n);
        for (int i = 0; i < n; ++i) indices[i] = i;
        std::shuffle(indices.begin(), indices.end(), rng_);
        for (int i = 0; i < std::min(k, n); ++i) {
            result.push_back(indices[i]);
        }
        return result;
    }

    // Roulette wheel selection
    std::uniform_real_distribution<double> dist(0.0, total);
    std::vector<bool> selected(n, false);

    while (result.size() < static_cast<size_t>(k) && result.size() < static_cast<size_t>(n)) {
        double r = dist(rng_);
        double cumsum = 0.0;
        for (int i = 0; i < n; ++i) {
            if (selected[i] || scores[i] <= 0) continue;
            cumsum += scores[i];
            if (cumsum >= r) {
                selected[i] = true;
                result.push_back(i);
                break;
            }
        }
    }

    return result;
}

// Get top k indices by score
inline std::vector<int> VariableSelector::top_k(
    const std::vector<double>& scores,
    int k
) {
    std::vector<std::pair<double, int> > scored_indices;
    int n = scores.size();

    for (int i = 0; i < n; ++i) {
        if (scores[i] >= 0) {  // Skip marked-as-invalid
            std::pair<double, int> p;
            p.first = scores[i];
            p.second = i;
            scored_indices.push_back(p);
        }
    }

    // Sort by score descending
    std::sort(scored_indices.begin(), scored_indices.end());
    std::reverse(scored_indices.begin(), scored_indices.end());

    std::vector<int> result;
    for (size_t i = 0; i < std::min(static_cast<size_t>(k), scored_indices.size()); ++i) {
        result.push_back(scored_indices[i].second);
    }

    return result;
}

} // namespace rl_sph

#endif // RL_VARIABLE_SELECTION_H
