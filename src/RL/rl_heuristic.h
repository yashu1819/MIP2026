#ifndef RL_HEURISTIC_H
#define RL_HEURISTIC_H

#include "../Definition/mip_problem.h"
#include "../Definition/solution.h"
#include "rl_graph.h"
#include "rl_state.h"
#include "rl_reward.h"
#include "rl_variable_selection.h"
#include <vector>
#include <string>

namespace rl_sph {

// Configuration for RL-SPH heuristic
struct RLConfig {
    // Time limit in seconds
    double time_limit = 60.0;

    // Maximum number of timesteps
    int max_steps = 2000;

    // Steps to stay in phase 1 (feasibility)
    int phase1_steps = 500;

    // Initial solution method: "lp" or "random"
    std::string init_method = "lp";

    // Whether to use trained model (inference) or train
    bool train_mode = false;

    // Path to load/save model weights
    std::string model_path = "";

    // Random seed
    int seed = 42;
};

// RL-SPH Heuristic for MILP
// Implements Algorithm 1: Solution search
class RLHeuristic {
public:
    RLHeuristic(const MIPProblem& mip, const RLConfig& config = RLConfig());

    // Run heuristic to find feasible solution
    // Returns best solution found
    Solution run();

    // Get incumbent solution
    const std::vector<double>& get_incumbent() const { return incumbent_; }

    // Get incumbent objective value
    double get_incumbent_obj() const { return incumbent_obj_; }

    // Check if feasible solution was found
    bool found_feasible() const { return found_feasible_; }

private:
    const MIPProblem& mip_;
    RLConfig config_;

    BipartiteGraph graph_;
    RewardComputer reward_computer_;
    VariableSelector var_selector_;

    // Current state
    RLState current_state_;

    // Incumbent (best feasible solution)
    std::vector<double> incumbent_;
    double incumbent_obj_;
    bool found_feasible_;

    // Current phase (1 = feasibility, 2 = optimization)
    int phase_;

    // Random number generator
    std::mt19937 rng_;

    // Initialize solution (LP-relaxation or random)
    std::vector<double> initialize_solution();

    // One step of solution search (Algorithm 1)
    double search_step();

    // Update state after actions
    void update_state(const std::vector<double>& x_new);

    // Rollback state change
    void rollback_state();
};

// Run RL-SPH heuristic on MIP problem
Solution run_rl_heuristic(
    const MIPProblem& mip,
    const RLConfig& config = RLConfig()
);

} // namespace rl_sph

#endif // RL_HEURISTIC_H
