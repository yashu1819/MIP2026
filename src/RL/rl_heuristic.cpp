#include "rl_heuristic.h"
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace rl_sph {

// Helper: get current time in seconds
static double get_time_seconds() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now.time_since_epoch()).count();
}

// RLHeuristic constructor
RLHeuristic::RLHeuristic(const MIPProblem& mip, const RLConfig& config)
    : mip_(mip)
    , config_(config)
    , graph_(build_graph(mip))
    , reward_computer_(mip)
    , var_selector_(mip, graph_)
    , incumbent_obj_(std::numeric_limits<double>::infinity())
    , found_feasible_(false)
    , phase_(1)
    , rng_(config.seed)
{
    // Seed the variable selector's RNG as well
    // Note: var_selector_ has its own rng_, we can't reseed it easily
    // For now, we accept the separate seeding
}

// Initialize solution using LP-relaxation or random assignment
std::vector<double> RLHeuristic::initialize_solution() {
    std::vector<double> x(mip_.num_cols);

    if (config_.init_method == "random") {
        // Random initialization: set ~1% of variables to 1, rest to 0
        // Or use half the number of nonzeros from previous instance
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        int num_ones = std::max(1, mip_.num_cols / 100);
        std::vector<int> indices(mip_.num_cols);
        for (int i = 0; i < mip_.num_cols; ++i) {
            indices[i] = i;
        }
        std::shuffle(indices.begin(), indices.end(), rng_);

        for (int i = 0; i < mip_.num_cols; ++i) {
            x[i] = 0.0;
        }
        for (int i = 0; i < num_ones; ++i) {
            x[indices[i]] = 1.0;
        }
    } else {
        // LP-relaxation based initialization
        // For now, use a simple heuristic: round LP solution or use bounds
        // In practice, this would call an LP solver

        // Simple heuristic: start at lower bounds, rounded to integer
        for (int i = 0; i < mip_.num_cols; ++i) {
            if (mip_.vartype[i] == VarType::CONTINUOUS) {
                x[i] = mip_.lb[i];
            } else if (mip_.vartype[i] == VarType::BINARY) {
                x[i] = (mip_.lb[i] >= 0.5) ? 1.0 : 0.0;
            } else {
                // Integer: round lower bound
                x[i] = std::floor(mip_.lb[i] + 0.5);
            }

            // Ensure within bounds
            if (x[i] < mip_.lb[i]) x[i] = mip_.lb[i];
            if (x[i] > mip_.ub[i]) x[i] = mip_.ub[i];
        }
    }

    return x;
}

// Update state after applying actions
void RLHeuristic::update_state(const std::vector<double>& x_new) {
    current_state_ = create_state(mip_, x_new);
}

// Rollback: restore previous state (used when move is rejected)
void RLHeuristic::rollback_state() {
    // For now, this is a no-op since we handle rollback in search_step
    // In a full implementation, we'd keep a backup of the previous state
}

// One step of solution search (Algorithm 1)
double RLHeuristic::search_step() {
    // Save previous state for rollback
    RLState prev_state = current_state_;
    std::vector<double> prev_x = current_state_.x;

    // Step 1: Select changeable variables (Algorithm 3)
    std::vector<int> changeable = var_selector_.select_variables(current_state_, phase_);

    // Step 2: Get actions from policy (for now, random actions)
    // TODO: Replace with trained neural network
    std::vector<Action> actions;
    std::uniform_int_distribution<int> action_dist(0, 2);

    for (size_t i = 0; i < changeable.size(); ++i) {
        int action_idx = action_dist(rng_);
        Action action = static_cast<Action>(action_idx - 1);  // -1, 0, or 1
        actions.push_back(action);
    }

    // Step 3: Apply actions to get new solution
    std::vector<double> x_new = apply_actions(current_state_.x, actions, changeable);

    // Step 4: Update state
    update_state(x_new);

    // Step 5: Compute reward
    double reward = reward_computer_.compute_reward(
        prev_state,
        current_state_,
        incumbent_obj_,
        phase_
    );

    // Step 6-8: Update incumbent if better feasible solution
    bool feasible = is_feasible(mip_, current_state_);
    if (feasible && current_state_.obj < incumbent_obj_) {
        incumbent_ = current_state_.x;
        incumbent_obj_ = current_state_.obj;
        found_feasible_ = true;
    }

    // Step 9-11: Rollback if needed
    bool should_rollback = false;
    if (phase_ == 1) {
        // In phase 1, rollback if bounds are violated
        if (!bounds_satisfied(mip_, current_state_)) {
            should_rollback = true;
        }
    } else {
        // In phase 2, rollback unless better feasible solution found
        if (!feasible || current_state_.obj >= incumbent_obj_) {
            should_rollback = true;
        }
    }

    if (should_rollback) {
        current_state_ = prev_state;
    }

    return reward;
}

// Run the heuristic
Solution RLHeuristic::run() {
    double start_time = get_time_seconds();

    // Initialize solution
    std::vector<double> x_init = initialize_solution();
    current_state_ = create_state(mip_, x_init);

    // Track best solution found
    if (is_feasible(mip_, current_state_)) {
        incumbent_ = current_state_.x;
        incumbent_obj_ = current_state_.obj;
        found_feasible_ = true;
    }

    // Main search loop
    int step = 0;
    int phase1_count = 0;

    while (step < config_.max_steps) {
        // Check time limit
        double elapsed = get_time_seconds() - start_time;
        if (elapsed > config_.time_limit) {
            break;
        }

        // Perform one search step
        double reward = search_step();

        // Phase transition logic
        if (phase_ == 1) {
            phase1_count++;
            if (phase1_count >= config_.phase1_steps) {
                phase_ = 2;
            }
        }

        step++;
    }

    // Create and return Solution object
    Solution result;
    if (found_feasible_) {
        result.x = incumbent_;
        result.obj_value = incumbent_obj_;
        result.feasible = true;
    } else {
        // Return last solution even if infeasible
        result.x = current_state_.x;
        result.obj_value = current_state_.obj;
        result.feasible = false;
    }

    return result;
}

// Convenience function to run RL heuristic
Solution run_rl_heuristic(const MIPProblem& mip, const RLConfig& config) {
    RLHeuristic heuristic(mip, config);
    return heuristic.run();
}

} // namespace rl_sph
