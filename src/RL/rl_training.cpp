#include "rl_training.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <fstream>

namespace rl_sph {

// Helper: get current time in seconds
static double get_time_seconds() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now.time_since_epoch()).count();
}

RLTrainer::RLTrainer(const TrainingConfig& config)
    : config_(config)
    , agent_ptr_(nullptr)
    , rng_(config.seed)
{
}

RLTrainer::~RLTrainer() {
    delete agent_ptr_;
}

// Initialize solution (Appendix C)
std::vector<double> RLTrainer::initialize_solution(
    const MIPProblem& mip,
    std::mt19937& rng,
    int& prev_num_ones
) {
    std::vector<double> x(mip.num_cols, 0.0);

    // For IS, CA, SC, MVC (binary problems): random initialization
    // Set r variables to 1, where r is half of previous instance's ones
    // or 1% for first instance

    int num_ones;
    if (prev_num_ones == 0) {
        // First instance: 1% of variables
        num_ones = std::max(1, mip.num_cols / 100);
    } else {
        // Subsequent: half of previous
        num_ones = prev_num_ones / 2;
    }

    // Randomly select variables to set to 1
    std::vector<int> indices(mip.num_cols);
    for (int i = 0; i < mip.num_cols; ++i) {
        indices[i] = i;
    }
    std::shuffle(indices.begin(), indices.end(), rng);

    for (int i = 0; i < num_ones; ++i) {
        x[indices[i]] = 1.0;
    }

    // Store for next instance
    prev_num_ones = num_ones;

    return x;
}

float RLTrainer::compute_td_error(
    double reward,
    float value_next,
    float value_current,
    float gamma
) {
    return static_cast<float>(reward) + gamma * value_next - value_current;
}

void RLTrainer::log_progress(
    int update_num,
    const TrainingStats& stats
) {
    if (config_.verbose && update_num % config_.log_interval == 0) {
        std::cout << "Update " << update_num
                  << ": avg_reward=" << stats.avg_reward
                  << ", avg_feas=" << stats.avg_feasibility_rate
                  << ", avg_obj=" << stats.avg_objective
                  << ", time=" << stats.avg_time_per_update << "s"
                  << std::endl;
    }
}

// One training step (Algorithm 2, lines 8-21)
double RLTrainer::training_step(
    const MIPProblem& mip,
    std::mt19937& rng
) {
    // Line 2: Get initial solution
    static int prev_num_ones = 0;
    std::vector<double> x = initialize_solution(mip, rng, prev_num_ones);

    // Line 3: Initial observation
    RLState state = create_state(mip, x);

    // Lines 4-6: Initialize incumbent and phase
    std::vector<double> incumbent;
    double obj_incumbent = std::numeric_limits<double>::infinity();
    int phase = 1;
    bool stay_in_phase1 = true;

    VariableSelector var_selector(mip, build_graph(mip));
    RewardComputer reward_computer(mip);

    double total_reward = 0.0;
    int step = 0;
    int phase1_count = 0;

    // Lines 8-21: Main loop
    while (step < config_.max_steps) {
        // Line 9: Search step (Algorithm 1)
        RLState prev_state = state;

        // Select variables (Algorithm 3)
        std::vector<int> changeable = var_selector.select_variables(state, phase);

        // Get actions from agent
        BipartiteGraph graph = build_graph(mip);
        std::vector<Action> actions = agent_ptr_->select_actions(
            state, graph, changeable, rng
        );

        // Apply actions
        std::vector<double> x_new = apply_actions(state.x, actions, changeable);
        state = create_state(mip, x_new);

        // Compute reward (Line 9)
        double reward = reward_computer.compute_reward(
            prev_state, state, obj_incumbent, phase
        );
        total_reward += reward;

        // Lines 6-8: Update incumbent
        bool feasible = is_feasible(mip, state);
        if (feasible && state.obj < obj_incumbent) {
            incumbent = state.x;
            obj_incumbent = state.obj;
        }

        // Lines 9-14: Phase transition
        if (stay_in_phase1 && (feasible || step >= config_.phase1_steps)) {
            if (step >= config_.phase1_steps) {
                stay_in_phase1 = false;
                phase = 2;
            }
        } else if (!stay_in_phase1 && feasible) {
            phase = 2;
        }

        // Line 18-20: Training update (simplified - full version would use LibTorch)
        // For now, we just collect the experience
        // Full implementation would:
        // - Compute TD error: delta = r + gamma * V(s') - V(s)
        // - Update actor: theta += lr * grad(log(pi(a|s)) * delta)
        // - Update critic: minimize delta^2

        step++;
    }

    return total_reward;
}

TrainingStats RLTrainer::train_on_instance(
    const MIPProblem& mip,
    int instance_idx
) {
    TrainingStats stats;
    double start_time = get_time_seconds();

    // Run training step
    double total_reward = training_step(mip, rng_);

    double elapsed = get_time_seconds() - start_time;

    stats.avg_reward = total_reward;
    stats.avg_time_per_update = elapsed;
    stats.total_updates = 1;

    (void)instance_idx;  // For logging

    return stats;
}

TrainingStats RLTrainer::train() {
    TrainingStats overall_stats;
    std::vector<TrainingStats> instance_stats;

    double total_start = get_time_seconds();

    // Load training instances
    std::vector<MIPProblem> instances;
    for (const auto& file : config_.training_files) {
        try {
            MIPProblem mip;
            mip.load_from_mps(file);
            mip.finalize();
            instances.push_back(mip);
        } catch (const std::exception& e) {
            if (config_.verbose) {
                std::cerr << "Warning: Failed to load " << file
                          << ": " << e.what() << std::endl;
            }
        }
    }

    if (instances.empty()) {
        if (config_.verbose) {
            std::cerr << "No training instances loaded. Training aborted." << std::endl;
        }
        return overall_stats;
    }

    // Initialize agent with correct dimensions
    int num_vars = instances[0].num_cols;
    int num_constraints = instances[0].num_rows;
    if (agent_ptr_ == nullptr) {
        agent_ptr_ = new RLAgent(num_vars, num_constraints, AgentConfig());
    }

    // Training loop (Algorithm 2, lines 1-22)
    for (int update = 0; update < config_.num_updates; ++update) {
        // Sample batch of instances
        std::uniform_int_distribution<size_t> dist(0, instances.size() - 1);

        double batch_reward = 0.0;
        int batch_feasible = 0;

        for (int b = 0; b < config_.batch_size; ++b) {
            size_t idx = dist(rng_);
            TrainingStats inst_stats = train_on_instance(instances[idx], idx);
            batch_reward += inst_stats.avg_reward;
        }

        // Update statistics
        overall_stats.avg_reward = (overall_stats.avg_reward * update + batch_reward)
                                    / (update + 1);
        overall_stats.total_updates = update + 1;

        // Log progress
        if (config_.verbose && update % config_.log_interval == 0) {
            double elapsed = get_time_seconds() - total_start;
            std::cout << "Update " << update
                      << ": avg_reward=" << overall_stats.avg_reward
                      << ", time=" << elapsed << "s"
                      << std::endl;
        }

        // Store history
        TrainingStats step_stats;
        step_stats.avg_reward = batch_reward / config_.batch_size;
        step_stats.total_updates = update + 1;
        history_.push_back(step_stats);
    }

    // Save model
    if (!config_.save_path.empty()) {
        save_model(config_.save_path);
    }

    return overall_stats;
}

void RLTrainer::save_model(const std::string& path) {
    if (agent_ptr_ != nullptr) {
        agent_ptr_->save(path);
    }
}

void RLTrainer::load_model(const std::string& path) {
    if (agent_ptr_ != nullptr) {
        agent_ptr_->load(path);
    }
}

// Convenience function
TrainingStats train_rl_heuristic(
    const std::vector<std::string>& mps_files,
    const TrainingConfig& config
) {
    TrainingConfig cfg = config;
    cfg.training_files = mps_files;

    RLTrainer trainer(cfg);
    return trainer.train();
}

} // namespace rl_sph
