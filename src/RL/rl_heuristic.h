#ifndef RL_HEURISTIC_H
#define RL_HEURISTIC_H

#include "../Definition/mip_problem.h"
#include "../Definition/solution.h"
#include "rl_graph.h"
#include "rl_state.h"
#include "rl_reward.h"
#include "rl_variable_selection.h"
#include "rl_agent.h"
#include <vector>
#include <string>

namespace rl_sph {

// Configuration for RL-SPH heuristic
struct RLConfig {
    double time_limit = 60.0;
    int max_steps = 2000;
    int phase1_steps = 500;
    std::string init_method = "lp";
    bool train_mode = false;
    std::string model_path = "";
    int seed = 42;
};

// RL-SPH Heuristic for MILP — Implements Algorithm 1: Solution search
class RLHeuristic {
public:
    RLHeuristic(const MIPProblem& mip, const RLConfig& config = RLConfig());

    Solution run();

    const std::vector<double>& get_incumbent() const { return incumbent_; }
    double get_incumbent_obj() const { return incumbent_obj_; }
    bool found_feasible() const { return found_feasible_; }

private:
    const MIPProblem& mip_;
    RLConfig config_;

    BipartiteGraph graph_;
    RewardComputer reward_computer_;
    VariableSelector var_selector_;
    RLAgent agent_;  // Neural network agent (replaces random actions)

    RLState current_state_;
    std::vector<double> incumbent_;
    double incumbent_obj_;
    bool found_feasible_;
    int phase_;
    std::mt19937 rng_;

    std::vector<double> initialize_solution();
    double search_step();
    void update_state(const std::vector<double>& x_new);
};

Solution run_rl_heuristic(
    const MIPProblem& mip,
    const RLConfig& config = RLConfig()
);

} // namespace rl_sph

#endif // RL_HEURISTIC_H
