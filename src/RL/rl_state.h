#ifndef RL_STATE_H
#define RL_STATE_H

#include <vector>
#include <cmath>
#include "../mip_problem.h"

namespace rl_sph {

// Observation state S_t = (x_t, f_t, obj_t)
struct RLState {
    // Current solution (integer values)
    std::vector<double> x;

    // Constraint satisfaction: f = b - Ax
    // f_j >= 0 means constraint j is satisfied
    std::vector<double> f;

    // Current objective value
    double obj;

    // Left-hand side values: Ax
    std::vector<double> lhs;

    // Number of variables
    int n;

    // Number of constraints
    int m;
};

// Compute LHS = Ax given solution x
inline std::vector<double> compute_lhs(
    const MIPProblem& mip,
    const std::vector<double>& x
) {
    std::vector<double> lhs(mip.num_rows, 0.0);

    // CSR format: row i has values csr_val[row_ptr[i]..row_ptr[i+1]-1]
    for (int i = 0; i < mip.num_rows; ++i) {
        double sum = 0.0;
        for (int j = mip.csr_row_ptr[i]; j < mip.csr_row_ptr[i + 1]; ++j) {
            int col = mip.csr_col_idx[j];
            sum += mip.csr_val[j] * x[col];
        }
        lhs[i] = sum;
    }
    return lhs;
}

// Compute constraint satisfaction f = b - Ax
inline std::vector<double> compute_feasibility_state(
    const MIPProblem& mip,
    const std::vector<double>& lhs
) {
    std::vector<double> f(mip.num_rows);
    for (int i = 0; i < mip.num_rows; ++i) {
        f[i] = mip.b[i] - lhs[i];
    }
    return f;
}

// Compute objective c^T x
inline double compute_objective_value(
    const MIPProblem& mip,
    const std::vector<double>& x
) {
    double obj = mip.obj_offset;
    for (int i = 0; i < mip.num_cols; ++i) {
        obj += mip.c[i] * x[i];
    }
    return obj;
}

// Create initial state from solution
inline RLState create_state(
    const MIPProblem& mip,
    const std::vector<double>& x
) {
    RLState state;
    state.n = mip.num_cols;
    state.m = mip.num_rows;
    state.x = x;
    state.lhs = compute_lhs(mip, x);
    state.f = compute_feasibility_state(mip, state.lhs);
    state.obj = compute_objective_value(mip, x);
    return state;
}

// Check if a variable is integer/binary (i.e., not continuous)
inline bool is_integer_variable(const MIPProblem& mip, int var_idx) {
    return mip.vartype[var_idx] == VarType::INTEGER ||
           mip.vartype[var_idx] == VarType::BINARY;
}

// Check if solution is feasible (MILP-aware: checks integrality)
inline bool is_feasible(
    const MIPProblem& mip,
    const RLState& state,
    double constr_tol = 1e-6,
    double bound_tol = 1e-6,
    double int_tol = 1e-5
) {
    // Check constraint satisfaction
    for (int i = 0; i < state.m; ++i) {
        if (state.f[i] < -constr_tol) {
            return false;
        }
    }

    // Check bounds and integrality
    for (int i = 0; i < state.n; ++i) {
        if (state.x[i] < mip.lb[i] - bound_tol ||
            state.x[i] > mip.ub[i] + bound_tol) {
            return false;
        }
        // MILP: integer/binary variables must have integer values
        if (is_integer_variable(mip, i)) {
            double frac = std::abs(state.x[i] - std::round(state.x[i]));
            if (frac > int_tol) {
                return false;
            }
        }
    }

    return true;
}

// Check if bounds are satisfied
inline bool bounds_satisfied(
    const MIPProblem& mip,
    const RLState& state,
    double tol = 1e-6
) {
    for (int i = 0; i < state.n; ++i) {
        if (state.x[i] < mip.lb[i] - tol ||
            state.x[i] > mip.ub[i] + tol) {
            return false;
        }
    }
    return true;
}

// Action: +1 (increase), 0 (no change), -1 (decrease)
enum class Action : int {
    DECREASE = -1,
    NO_CHANGE = 0,
    INCREASE = 1
};

// Apply actions to get new solution (MILP-aware)
// - Skips continuous variables (RL only modifies integer/binary)
// - Clamps binary variables to {0, 1}
// - Clamps integer variables to [lb, ub] and rounds
inline std::vector<double> apply_actions(
    const std::vector<double>& x,
    const std::vector<Action>& actions,
    const std::vector<int>& changeable_indices
) {
    std::vector<double> x_new = x;
    for (size_t i = 0; i < changeable_indices.size(); ++i) {
        int var_idx = changeable_indices[i];
        x_new[var_idx] += static_cast<int>(actions[i]);
    }
    return x_new;
}

// // MILP-aware version: skips continuous, clamps binary/integer
// inline std::vector<double> apply_actions_milp(
//     const MIPProblem& mip,
//     const std::vector<double>& x,
//     const std::vector<Action>& actions,
//     const std::vector<int>& changeable_indices
// ) {
//     std::vector<double> x_new = x;
//     for (size_t i = 0; i < changeable_indices.size(); ++i) {
//         int var_idx = changeable_indices[i];

//         // Skip continuous variables — they should not be modified by ±1
//         if (mip.vartype[var_idx] == VarType::CONTINUOUS) {
//             continue;
//         }

//         // Apply action
//         x_new[var_idx] += static_cast<int>(actions[i]);

//         // Clamp to bounds
//         if (mip.vartype[var_idx] == VarType::BINARY) {
//             // Binary: clamp to {0, 1}
//             if (x_new[var_idx] < 0.5) x_new[var_idx] = 0.0;
//             else x_new[var_idx] = 1.0;
//         } else {
//             // Integer: clamp to [lb, ub] and round
//             x_new[var_idx] = std::round(x_new[var_idx]);
//             if (x_new[var_idx] < mip.lb[var_idx]) x_new[var_idx] = mip.lb[var_idx];
//             if (x_new[var_idx] > mip.ub[var_idx]) x_new[var_idx] = mip.ub[var_idx];
//         }
//     }
//     return x_new;
// }



inline std::vector<double> apply_actions_milp(
    const MIPProblem& mip,
    const std::vector<double>& x,
    const std::vector<Action>& actions,
    const std::vector<int>& changeable_indices
) {
    std::vector<double> x_new = x;

    for (size_t i = 0; i < changeable_indices.size(); ++i) {
        int var_idx = changeable_indices[i];

        // Skip continuous variables
        if (mip.vartype[var_idx] == VarType::CONTINUOUS) {
            continue;
        }

        int action = static_cast<int>(actions[i]);  // {-1, 0, +1}

        // FIX 1: Binary variables → DIRECT ASSIGNMENT
        if (mip.vartype[var_idx] == VarType::BINARY) {
            if (action == 1) {
                x_new[var_idx] = 1.0;
            } else if (action == -1) {
                x_new[var_idx] = 0.0;
            }
            // action == 0 → keep unchanged
        }

        // FIX 2: Integer variables → controlled step
        else {
            double new_val = x_new[var_idx] + action;

            // Round to integer
            new_val = std::round(new_val);

            // Clamp to bounds
            if (new_val < mip.lb[var_idx]) new_val = mip.lb[var_idx];
            if (new_val > mip.ub[var_idx]) new_val = mip.ub[var_idx];

            x_new[var_idx] = new_val;
        }
    }

    // FIX 3: Ensure at least one change (critical for escaping stagnation)
    bool changed = false;
    for (size_t i = 0; i < x.size(); ++i) {
        if (std::abs(x[i] - x_new[i]) > 1e-8) {
            changed = true;
            break;
        }
    }

    if (!changed && !changeable_indices.empty()) {
        int idx = changeable_indices[rand() % changeable_indices.size()];

        if (mip.vartype[idx] == VarType::BINARY) {
            x_new[idx] = 1.0 - x_new[idx];  // flip
        } else {
            // small random perturbation for integer
            int delta = (rand() % 2 == 0) ? -1 : 1;
            double new_val = std::round(x_new[idx] + delta);

            if (new_val < mip.lb[idx]) new_val = mip.lb[idx];
            if (new_val > mip.ub[idx]) new_val = mip.ub[idx];

            x_new[idx] = new_val;
        }
    }

    return x_new;
}

} // namespace rl_sph

#endif // RL_STATE_H
