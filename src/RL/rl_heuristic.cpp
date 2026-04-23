#include "rl_heuristic.h"
#include "rl_lp_subproblem.h"
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace rl_sph {

static double get_time_seconds() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now.time_since_epoch()).count();
}

RLHeuristic::RLHeuristic(const MIPProblem& mip, const RLConfig& config)
    : mip_(mip)
    , config_(config)
    , graph_(build_graph(mip))
    , reward_computer_(mip)
    , var_selector_(mip, graph_)
    , agent_(mip.num_cols, mip.num_rows, AgentConfig())
    , incumbent_obj_(std::numeric_limits<double>::infinity())
    , found_feasible_(false)
    , phase_(1)
    , rng_(config.seed)
{
    // Give the agent a reference to the MIP for feature building
    agent_.set_mip_problem(mip_);

    // Load pretrained model if provided
    if (!config_.model_path.empty()) {
        agent_.load(config_.model_path);
    }
}

std::vector<double> RLHeuristic::initialize_solution() {
    std::vector<double> x(mip_.num_cols);

    if (config_.init_method == "random") {
        // Random init: only randomize integer/binary variables
        std::vector<int> int_indices;
        for (int i = 0; i < mip_.num_cols; ++i) {
            if (mip_.vartype[i] == VarType::CONTINUOUS) {
                // Continuous: set to lower bound
                x[i] = mip_.lb[i];
            } else {
                int_indices.push_back(i);
                x[i] = 0.0;
            }
        }
        int num_ones = std::max(1, static_cast<int>(int_indices.size()) / 100);
        std::shuffle(int_indices.begin(), int_indices.end(), rng_);
        for (int i = 0; i < num_ones && i < static_cast<int>(int_indices.size()); ++i) {
            x[int_indices[i]] = 1.0;
        }
    } else {
        // LP-relaxation based: start at lower bounds
        for (int i = 0; i < mip_.num_cols; ++i) {
            if (mip_.vartype[i] == VarType::CONTINUOUS) {
                x[i] = mip_.lb[i];
            } else if (mip_.vartype[i] == VarType::BINARY) {
                x[i] = (mip_.lb[i] >= 0.5) ? 1.0 : 0.0;
            } else {
                // Integer: round to nearest integer within bounds
                x[i] = std::floor(mip_.lb[i] + 0.5);
            }
            if (x[i] < mip_.lb[i]) x[i] = mip_.lb[i];
            if (x[i] > mip_.ub[i]) x[i] = mip_.ub[i];
        }
    }
    return x;
}

void RLHeuristic::update_state(const std::vector<double>& x_new) {
    current_state_ = create_state(mip_, x_new);
}

// One step of solution search (Algorithm 1) — now uses the RL agent
double RLHeuristic::search_step() {
    RLState prev_state = current_state_;

    // Step 1: Select changeable variables (Algorithm 3)
    std::vector<int> changeable = var_selector_.select_variables(current_state_, phase_);

    // Step 2: Get actions from the RL agent (NOT random anymore)
    std::vector<Action> actions = agent_.select_actions(
        current_state_, graph_, changeable, rng_);

    // Step 3: Apply actions (MILP-aware: skips continuous, clamps binary/integer)
    std::vector<double> x_new = apply_actions_milp(mip_, current_state_.x, actions, changeable);

    // Step 3b: Solve LP sub-problem for continuous variables
    solve_lp_subproblem(mip_, x_new);

    // Step 4: Update state
    update_state(x_new);

    // Step 5: Compute reward
    double reward = reward_computer_.compute_reward(
        prev_state, current_state_, incumbent_obj_, phase_);

    // Step 6: Update incumbent if better feasible
    bool feasible = is_feasible(mip_, current_state_);
    if (feasible && current_state_.obj < incumbent_obj_) {
        incumbent_ = current_state_.x;
        incumbent_obj_ = current_state_.obj;
        found_feasible_ = true;
    }

    // Step 7: Rollback if needed
    bool should_rollback = false;
    if (phase_ == 1) {
        if (!bounds_satisfied(mip_, current_state_)) should_rollback = true;
    } else {
        if (!feasible || current_state_.obj >= incumbent_obj_) should_rollback = true;
    }
    if (should_rollback) current_state_ = prev_state;

    return reward;
}

Solution RLHeuristic::run() {
    double start_time = get_time_seconds();

    std::vector<double> x_init = initialize_solution();
    current_state_ = create_state(mip_, x_init);

    if (is_feasible(mip_, current_state_)) {
        incumbent_ = current_state_.x;
        incumbent_obj_ = current_state_.obj;
        found_feasible_ = true;
    }

    int step = 0;
    int phase1_count = 0;

    while (step < config_.max_steps) {
        double elapsed = get_time_seconds() - start_time;
        if (elapsed > config_.time_limit) break;

        search_step();

        if (phase_ == 1) {
            phase1_count++;
            if (phase1_count >= config_.phase1_steps) phase_ = 2;
        }
        step++;
    }

    Solution result;
    if (found_feasible_) {
        result.x = incumbent_;
        result.obj_value = incumbent_obj_;
        result.feasible = true;
    } else {
        result.x = current_state_.x;
        result.obj_value = current_state_.obj;
        result.feasible = false;
    }
    return result;
}

Solution run_rl_heuristic(const MIPProblem& mip, const RLConfig& config) {
    RLHeuristic heuristic(mip, config);
    return heuristic.run();
}

} // namespace rl_sph
