#ifndef RL_VARIABLE_SELECTION_H
#define RL_VARIABLE_SELECTION_H

#include "rl_graph.h"
#include "rl_state.h"
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include "../Definition/mip_problem.h"

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

// Select variables for phase 1 or phase 2
inline std::vector<int> VariableSelector::select_variables(
    const RLState& state,
    int phase
) {
    int n = state.n;
    int m = state.m;

    // Number of seed and neighbor variables: p = q = log2(n)
    int p = static_cast<int>(std::log2(n));
    int q = p;
    if (p < 1) p = 1;
    if (q < 1) q = 1;

    // Build binary incidence matrix indicator: A_j,i != 0
    // For each variable, compute score based on constraints it appears in

    std::vector<double> score_seed(n, 0.0);

    if (phase == 1) {
        // Phase 1: Select variables in violated constraints
        // Score: sum over violated constraints where var appears

        // Indicator: which constraints are violated
        std::vector<int> violated_constraints;
        for (int j = 0; j < m; ++j) {
            if (state.f[j] < 0) {
                violated_constraints.push_back(j);
            }
        }

        // Score for each variable
        for (int i = 0; i < n; ++i) {
            double score = 0.0;
            for (int constr_idx : graph_.var_to_constraints[i]) {
                // Check if constraint is violated
                if (state.f[constr_idx] < 0) {
                    score += 1.0;
                }
            }
            score_seed[i] = score;
        }

        // Weight: prefer variables with lower objective coefficient
        // weight = (max(|c|) - |c| + 1) / max(|c|)
        double max_c = 0.0;
        for (double c : mip_.c) {
            max_c = std::max(max_c, std::abs(c));
        }
        if (max_c < 1e-10) max_c = 1.0;

        for (int i = 0; i < n; ++i) {
            double weight = (max_c - std::abs(mip_.c[i]) + 1.0) / max_c;
            score_seed[i] *= weight;
        }

    } else {
        // Phase 2: Select variables in low-risk (well-satisfied) constraints
        // Score: sum over satisfied constraints where var appears

        for (int i = 0; i < n; ++i) {
            double score = 0.0;
            for (int constr_idx : graph_.var_to_constraints[i]) {
                // Check if constraint is well-satisfied (f > 0)
                if (state.f[constr_idx] > 0) {
                    score += 1.0;
                }
            }
            score_seed[i] = score;
        }

        // Weight: prefer variables with higher objective coefficient
        // weight = |c| / max(|c|)
        double max_c = 0.0;
        for (double c : mip_.c) {
            max_c = std::max(max_c, std::abs(c));
        }
        if (max_c < 1e-10) max_c = 1.0;

        for (int i = 0; i < n; ++i) {
            double weight = std::abs(mip_.c[i]) / max_c;
            score_seed[i] *= weight;
        }

        // Invert scores: higher score = lower risk = more likely to select
        // score = max(score) - score + 1
        double max_score = 0.0;
        for (double s : score_seed) {
            max_score = std::max(max_score, s);
        }
        for (int i = 0; i < n; ++i) {
            score_seed[i] = max_score - score_seed[i] + 1.0;
        }
    }

    // Sample p seed variables according to scores
    std::vector<int> seed_indices = sample_weighted(score_seed, p);

    // Neighbor selection: variables sharing constraints with seeds
    // g = rowwise sum of A[:, seeds]
    std::vector<int> constraint_counts(m, 0);
    for (int seed_idx : seed_indices) {
        for (int constr_idx : graph_.var_to_constraints[seed_idx]) {
            constraint_counts[constr_idx]++;
        }
    }

    // score_neighbor = g^T * A = for each var, count shared constraints with seeds
    std::vector<double> score_neighbor(n, 0.0);
    for (int i = 0; i < n; ++i) {
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

    while (result.size() < static_cast<size_t>(k) && result.size() < n) {
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
