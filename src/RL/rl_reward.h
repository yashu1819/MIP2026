#ifndef RL_REWARD_H
#define RL_REWARD_H

#include "rl_state.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include "../Definition/mip_problem.h"

namespace rl_sph {

// Reward computation for RL-SPH
class RewardComputer {
public:
    RewardComputer(const MIPProblem& mip) : mip_(mip), alpha_(2.0) {}

    // Compute total reward for transition
    double compute_reward(
        const RLState& state_prev,
        const RLState& state_curr,
        double obj_incumbent,
        int phase
    );

    // Phase 1 reward (feasibility focus)
    double phase1_reward(
        const RLState& state_prev,
        const RLState& state_curr,
        double obj_incumbent
    );

    // Phase 2 reward (optimization focus)
    double phase2_reward(
        const RLState& state_prev,
        const RLState& state_curr,
        double obj_incumbent
    );

    // Feasibility reward component
    double feasibility_reward(
        const RLState& state_prev,
        const RLState& state_curr,
        int num_changeable
    );

    // Bound reward: penalty for bound violations
    double bound_reward(const RLState& state);

    // Constraint reward: improvement/deterioration in constraint violations
    double constraint_reward(
        const RLState& state_prev,
        const RLState& state_curr
    );

    // Exploration penalty: -100 if no move made
    double exploration_reward(
        const RLState& state_prev,
        const RLState& state_curr
    );

private:
    const MIPProblem& mip_;
    double alpha_;  // Toward-optimal bias

    // Normalize objective delta
    double normalize_obj_delta(double obj_prev, double obj_curr) const {
        double max_c = 0.0;
        for (double c : mip_.c) {
            max_c = std::max(max_c, std::abs(c));
        }
        if (max_c < 1e-10) max_c = 1.0;
        return std::abs(obj_curr - obj_prev) / max_c;
    }
};

// Compute bound reward: -1 for each violated bound
inline double RewardComputer::bound_reward(const RLState& state) {
    double reward = 0.0;
    for (int i = 0; i < state.n; ++i) {
        if (state.x[i] < mip_.lb[i] || state.x[i] > mip_.ub[i]) {
            reward -= 1.0;
        }
    }
    return reward;
}

// Compute constraint reward
inline double RewardComputer::constraint_reward(
    const RLState& state_prev,
    const RLState& state_curr
) {
    double reward = 0.0;
    for (int j = 0; j < state_curr.m; ++j) {
        double prev_violation = std::min(state_prev.f[j], 0.0);
        double curr_violation = std::min(state_curr.f[j], 0.0);
        reward += (curr_violation - prev_violation);
    }
    return reward;
}

// Compute feasibility reward
inline double RewardComputer::feasibility_reward(
    const RLState& state_prev,
    const RLState& state_curr,
    int num_changeable
) {
    double r_bound = bound_reward(state_curr);
    double r_const = constraint_reward(state_prev, state_curr);

    if (num_changeable > 0) {
        return r_bound + r_const / std::sqrt(static_cast<double>(num_changeable));
    }
    return r_bound;
}

// Phase 1 reward
inline double RewardComputer::phase1_reward(
    const RLState& state_prev,
    const RLState& state_curr,
    double obj_incumbent
) {
    (void)obj_incumbent;
    double r_bound = bound_reward(state_curr);
    double r_const = constraint_reward(state_prev, state_curr);
    double delta_obj = normalize_obj_delta(state_prev.obj, state_curr.obj);
    bool bounds_ok = bounds_satisfied(mip_, state_curr);
    bool obj_better = state_curr.obj < state_prev.obj;

    // Case 1 & 2: bounds violated, constraint improvement
    if (r_const >= 0 && !bounds_ok) {
        if (obj_better) {
            return r_bound;  // Case 1
        } else {
            return r_bound - delta_obj;  // Case 2
        }
    }

    // Case 3: bounds satisfied, constraint improvement, obj better
    if (r_const >= 0 && bounds_ok && obj_better) {
        return feasibility_reward(state_prev, state_curr, state_curr.n) + delta_obj;
    }

    // Case 4: constraint violation or obj worse
    if (r_const < 0 || (!obj_better && bounds_ok)) {
        return feasibility_reward(state_prev, state_curr, state_curr.n) - delta_obj;
    }

    // Case 5: otherwise
    return feasibility_reward(state_prev, state_curr, state_curr.n);
}

// Phase 2 reward
inline double RewardComputer::phase2_reward(
    const RLState& state_prev,
    const RLState& state_curr,
    double obj_incumbent
) {
    bool feasible = is_feasible(mip_, state_curr);
    bool better = state_curr.obj < obj_incumbent;
    double delta_obj = normalize_obj_delta(state_prev.obj, state_curr.obj);
    double r_feas = feasibility_reward(state_prev, state_curr, state_curr.n);

    if (feasible) {
        if (better) {
            return delta_obj;  // Case 1: better feasible
        } else {
            return -delta_obj * alpha_;  // Case 2: worse feasible
        }
    } else {
        if (better) {
            return r_feas;  // Case 3: infeasible but obj better
        } else {
            return r_feas * alpha_;  // Case 4: infeasible and obj worse
        }
    }
}

// Total reward
inline double RewardComputer::compute_reward(
    const RLState& state_prev,
    const RLState& state_curr,
    double obj_incumbent,
    int phase
) {
    double r_opt = (phase == 1) ?
        phase1_reward(state_prev, state_curr, obj_incumbent) :
        phase2_reward(state_prev, state_curr, obj_incumbent);

    // Exploration penalty
    double r_explore = 0.0;
    bool no_move = true;
    for (int i = 0; i < state_curr.n; ++i) {
        if (std::abs(state_curr.x[i] - state_prev.x[i]) > 1e-10) {
            no_move = false;
            break;
        }
    }
    if (no_move) {
        r_explore = -100.0;
    }

    return r_opt + r_explore;
}

} // namespace rl_sph

#endif // RL_REWARD_H
