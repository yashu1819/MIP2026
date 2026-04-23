#include "rl_training.h"
#include "rl_lp_subproblem.h"
#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <fstream>

namespace rl_sph {

static double get_time_seconds() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now.time_since_epoch()).count();
}

RLTrainer::RLTrainer(const TrainingConfig& config)
    : config_(config), agent_ptr_(nullptr), rng_(config.seed) {}

RLTrainer::~RLTrainer() { delete agent_ptr_; }

std::vector<double> RLTrainer::initialize_solution(
    const MIPProblem& mip, std::mt19937& rng, int& prev_num_ones)
{
    std::vector<double> x(mip.num_cols, 0.0);
    int num_ones = (prev_num_ones == 0) ? std::max(1, mip.num_cols / 100) : prev_num_ones / 2;
    std::vector<int> indices(mip.num_cols);
    for (int i = 0; i < mip.num_cols; ++i) indices[i] = i;
    std::shuffle(indices.begin(), indices.end(), rng);
    for (int i = 0; i < num_ones && i < mip.num_cols; ++i) x[indices[i]] = 1.0;
    prev_num_ones = num_ones;
    return x;
}

float RLTrainer::compute_td_error(double reward, float value_next, float value_current, float gamma) {
    return static_cast<float>(reward) + gamma * value_next - value_current;
}

void RLTrainer::log_progress(int update_num, const TrainingStats& stats) {
    if (config_.verbose && update_num % config_.log_interval == 0) {
        std::cout << "Update " << update_num
                  << ": avg_reward=" << stats.avg_reward
                  << ", feas_rate=" << stats.avg_feasibility_rate
                  << ", time=" << stats.avg_time_per_update << "s" << std::endl;
    }
}

// One training episode — collects trajectory and calls agent.update()
double RLTrainer::training_step(const MIPProblem& mip, std::mt19937& rng) {
    static int prev_num_ones = 0;
    std::vector<double> x = initialize_solution(mip, rng, prev_num_ones);
    RLState state = create_state(mip, x);

    std::vector<double> incumbent;
    double obj_incumbent = std::numeric_limits<double>::infinity();
    int phase = 1;
    bool stay_in_phase1 = true;

    BipartiteGraph graph = build_graph(mip);
    VariableSelector var_selector(mip, graph);
    RewardComputer reward_computer(mip);

    // Trajectory storage for gradient update
    std::vector<RLState> traj_states;
    std::vector<std::vector<Action> > traj_actions;
    std::vector<double> traj_rewards;
    std::vector<int> traj_phases;
    std::vector<std::vector<int> > traj_changeable;

    double total_reward = 0.0;
    int step = 0;

    while (step < config_.max_steps) {
        RLState prev_state = state;

        // Select variables (Algorithm 3)
        std::vector<int> changeable = var_selector.select_variables(state, phase);

        // Get actions from agent (training mode with log_probs)
        TrainingForwardResult fwd = agent_ptr_->select_actions_training(
            state, graph, changeable, phase, rng);

        // Apply actions (MILP-aware)
        std::vector<double> x_new = apply_actions_milp(mip, state.x, fwd.actions, changeable);

        // Solve LP sub-problem for continuous variables
        solve_lp_subproblem(mip, x_new);

        state = create_state(mip, x_new);

        // Compute reward
        double reward = reward_computer.compute_reward(prev_state, state, obj_incumbent, phase);
        total_reward += reward;

        // Update incumbent
        bool feasible = is_feasible(mip, state);
        if (feasible && state.obj < obj_incumbent) {
            incumbent = state.x;
            obj_incumbent = state.obj;
        }

        // Store trajectory (no rollback during training — paper Algorithm 2)
        traj_states.push_back(prev_state);
        traj_actions.push_back(fwd.actions);
        traj_rewards.push_back(reward);
        traj_phases.push_back(phase);
        traj_changeable.push_back(changeable);

        // Phase transition
        if (stay_in_phase1 && (feasible || step >= config_.phase1_steps)) {
            if (step >= config_.phase1_steps) {
                stay_in_phase1 = false;
                phase = 2;
            }
        } else if (!stay_in_phase1 && feasible) {
            phase = 2;
        }

        step++;
    }

    // Add final state for value bootstrapping
    traj_states.push_back(state);

    // Gradient update using collected trajectory
    agent_ptr_->update(
        traj_states, traj_actions, traj_rewards,
        traj_phases, traj_changeable, graph, config_.gamma);

    return total_reward;
}

TrainingStats RLTrainer::train_on_instance(const MIPProblem& mip, int instance_idx) {
    TrainingStats stats;
    double start_time = get_time_seconds();
    double total_reward = training_step(mip, rng_);
    stats.avg_reward = total_reward;
    stats.avg_time_per_update = get_time_seconds() - start_time;
    stats.total_updates = 1;
    (void)instance_idx;
    return stats;
}

TrainingStats RLTrainer::train() {
    TrainingStats overall_stats;
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
            if (config_.verbose)
                std::cerr << "Warning: Failed to load " << file << ": " << e.what() << std::endl;
        }
    }

    if (instances.empty()) {
        if (config_.verbose) std::cerr << "No training instances loaded." << std::endl;
        return overall_stats;
    }

    // Initialize agent
    int nv = instances[0].num_cols, nc = instances[0].num_rows;
    if (agent_ptr_ == nullptr) {
        AgentConfig ac;
        ac.learning_rate = config_.learning_rate;
        ac.rmsprop_alpha = config_.rmsprop_alpha;
        ac.rmsprop_epsilon = config_.rmsprop_epsilon;
        ac.weight_decay = config_.weight_decay;
        agent_ptr_ = new RLAgent(nv, nc, ac);
    }

    // Training loop (Algorithm 2)
    for (int update = 0; update < config_.num_updates; ++update) {
        std::uniform_int_distribution<size_t> dist(0, instances.size() - 1);
        double batch_reward = 0.0;

        for (int b = 0; b < config_.batch_size; ++b) {
            size_t idx = dist(rng_);
            // Set MIP reference for the current instance
            agent_ptr_->set_mip_problem(instances[idx]);
            TrainingStats ist = train_on_instance(instances[idx], static_cast<int>(idx));
            batch_reward += ist.avg_reward;
        }

        overall_stats.avg_reward = (overall_stats.avg_reward * update + batch_reward) / (update + 1);
        overall_stats.total_updates = update + 1;

        if (config_.verbose && update % config_.log_interval == 0) {
            double elapsed = get_time_seconds() - total_start;
            std::cout << "Update " << update << ": avg_reward=" << overall_stats.avg_reward
                      << ", time=" << elapsed << "s" << std::endl;
        }

        TrainingStats step_stats;
        step_stats.avg_reward = batch_reward / config_.batch_size;
        step_stats.total_updates = update + 1;
        history_.push_back(step_stats);
    }

    if (!config_.save_path.empty()) save_model(config_.save_path);
    return overall_stats;
}

void RLTrainer::save_model(const std::string& path) {
    if (agent_ptr_) agent_ptr_->save(path);
}

void RLTrainer::load_model(const std::string& path) {
    if (agent_ptr_) agent_ptr_->load(path);
}

TrainingStats train_rl_heuristic(const std::vector<std::string>& mps_files, const TrainingConfig& config) {
    TrainingConfig cfg = config;
    cfg.training_files = mps_files;
    RLTrainer trainer(cfg);
    return trainer.train();
}

} // namespace rl_sph
