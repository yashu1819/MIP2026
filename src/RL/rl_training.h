#ifndef RL_TRAINING_H
#define RL_TRAINING_H

#include "rl_agent.h"
#include "rl_heuristic.h"
#include "rl_logger.h" 
#include "../mip_problem.h"
#include <vector>
#include <string>
#include <random>

namespace rl_sph {

// Training configuration (from Appendix C of paper)
struct TrainingConfig {
    // Number of parameter updates
    int num_updates = 5000;

    // Maximum steps per instance
    int max_steps = 2000;

    // Steps to stay in phase 1 before switching to phase 2
    int phase1_steps = 500;

    // Number of instances to train on in parallel
    int batch_size = 64;

    // Discount factor for TD learning
    float gamma = 0.99f;

    // Learning rate
    float learning_rate = 1e-4f;

    // RMSprop parameters
    float rmsprop_alpha = 0.99f;
    float rmsprop_epsilon = 1e-5f;
    float weight_decay = 1e-3f;

    // Random seed
    int seed = 42;

    // Path to training instances (MPS files)
    std::vector<std::string> training_files;

    // Save path for trained model
    std::string save_path = "rl_sph_model.pt";

    // Logging
    bool verbose = true;
    int log_interval = 100;  // Log every N updates
};

// Training statistics
struct TrainingStats {
    double avg_reward = 0.0;
    double avg_feasibility_rate = 0.0;
    double avg_objective = 0.0;
    double avg_time_per_update = 0.0;
    int total_updates = 0;
    int feasible_solutions_found = 0;
};

// RL-SPH Trainer
// Implements Algorithm 2: Learning a policy for RL-SPH
class RLTrainer {
public:
    RLTrainer(const TrainingConfig& config);
    ~RLTrainer();

    // Train on provided instances
    TrainingStats train();

    // Train on a single instance
    TrainingStats train_on_instance(
        const MIPProblem& mip,
        int instance_idx
    );

    // Get trained agent
    const RLAgent* get_agent() const { return agent_ptr_; }
    RLAgent* get_agent() { return agent_ptr_; }

    // Save/load model
    void save_model(const std::string& path);
    void load_model(const std::string& path);

    // Get training history
    const std::vector<TrainingStats>& get_history() const { return history_; }

private:
    TrainingConfig config_;
    RLAgent* agent_ptr_;
    std::mt19937 rng_;
    std::vector<TrainingStats> history_;

    EpisodeStats last_episode_stats_;

    // One training update (Algorithm 2, lines 8-21)
    double training_step(
        const MIPProblem& mip,
        std::mt19937& rng
    );

    EpisodeStats training_episode(
        const MIPProblem& mip,
        std::mt19937& rng
    );

    // Initialize solution (Appendix C)
    std::vector<double> initialize_solution(
        const MIPProblem& mip,
        std::mt19937& rng,
        int& prev_num_ones
    );

    // Compute TD error
    float compute_td_error(
        double reward,
        float value_next,
        float value_current,
        float gamma
    );

    // Log training progress
    void log_progress(
        int update_num,
        const TrainingStats& stats
    );
};

// Train RL-SPH heuristic on dataset
TrainingStats train_rl_heuristic(
    const std::vector<std::string>& mps_files,
    const TrainingConfig& config = TrainingConfig()
);

} // namespace rl_sph

#endif // RL_TRAINING_H
