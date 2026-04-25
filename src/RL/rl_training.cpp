#include "rl_training.h"
#include "rl_lp_subproblem.h"
#include "rl_logger.h"
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

    // For first instance, use 1% of variables as ones; for subsequent, use half of previous
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
// Now returns EpisodeStats for detailed diagnostics
EpisodeStats RLTrainer::training_episode(const MIPProblem& mip, std::mt19937& rng) {
    EpisodeStats es;
    double ep_start = get_time_seconds();

    static int prev_num_ones = 0;
    std::vector<double> x = initialize_solution(mip, rng, prev_num_ones);
    RLState state = create_state(mip, x);

    // Record initial violations
    es.constraint_violations_start = count_violated_constraints(mip, state);
    es.bound_violations_start      = count_violated_bounds(mip, state);

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

    // Detailed step-level logging interval (every 10% of max_steps)
    int step_log_interval = std::max(1, config_.max_steps / 10);

    while (step < config_.max_steps) {
        RLState prev_state = state;

        // Select variables (Algorithm 3)
        std::vector<int> changeable = var_selector.select_variables(state, phase);

        // Skip if no changeable variables
        if (changeable.empty()) {
            step++;
            continue;
        } 

        // Get actions from agent (training mode with log_probs)
        TrainingForwardResult fwd = agent_ptr_->select_actions_training(
            state, graph, changeable, phase, rng);

        // Apply actions (MILP-aware)
        std::vector<double> x_new = apply_actions_milp(mip, state.x, fwd.actions, changeable);
    
        double diff = 0;
        for (size_t i = 0; i < state.x.size(); i++) {
            diff += std::abs(state.x[i] - x_new[i]);
        }

        if (diff < 1e-6 && !changeable.empty()) {
            std::cout << "No state change → forcing flip\n";
            int idx = changeable[rand() % changeable.size()];

            // flip binary variable
            if (mip.vartype[idx] == VarType::BINARY) {
                x_new[idx] = 1.0 - x_new[idx];
            }
        }

        // Solve LP sub-problem for continuous variables
        solve_lp_subproblem(mip, x_new);

        state = create_state(mip, x_new);

        // Compute reward
        double reward = reward_computer.compute_reward(prev_state, state, obj_incumbent, phase);
        reward /= std::sqrt((double)mip.num_cols); // Reward Normalization
        total_reward += reward;

        // Track reward stats
        es.min_reward = std::min(es.min_reward, reward);
        es.max_reward = std::max(es.max_reward, reward);
        es.sum_reward += reward;

        // Update incumbent
        bool feasible = is_feasible(mip, state);
        if (feasible) es.feasible_count++;

        if (feasible && state.obj < obj_incumbent) {
            incumbent = state.x;
            obj_incumbent = state.obj;
            es.incumbent_updates++;
            es.best_obj = std::min(es.best_obj, state.obj);

            // Record incumbent improvement
            IncumbentRecord rec;
            rec.obj_value = state.obj;
            rec.timestamp = get_time_seconds() - ep_start;
            rec.step      = step;
            rec.phase     = phase;
            rec.feasible  = true;
            es.incumbent_trace.push_back(rec);
        }

        // Phase tracking
        if (phase == 1) es.phase1_steps++;
        else            es.phase2_steps++;

        // Step-level logging during training (verbose level 2)
        if (config_.verbose && config_.log_interval <= 5 && step % step_log_interval == 0) {
            print_training_step(step, phase, reward, state.obj,
                                feasible, obj_incumbent,
                                static_cast<int>(changeable.size()));
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

    // Record final violations
    es.constraint_violations_end = count_violated_constraints(mip, state);
    es.bound_violations_end      = count_violated_bounds(mip, state);

    // Gradient update using collected trajectory
    agent_ptr_->update(
        traj_states, traj_actions, traj_rewards,
        traj_phases, traj_changeable, graph, config_.gamma);

    es.total_reward    = total_reward;
    es.total_steps     = step;
    es.elapsed_seconds = get_time_seconds() - ep_start;

    return es;
}

// Legacy wrapper — keeps old interface working
double RLTrainer::training_step(const MIPProblem& mip, std::mt19937& rng) {
    EpisodeStats es = training_episode(mip, rng);
    return es.total_reward;
}

TrainingStats RLTrainer::train_on_instance(const MIPProblem& mip, int instance_idx) {
    TrainingStats stats;
    double start_time = get_time_seconds();

    // CRITICAL: Update agent's MIP reference for THIS instance
    agent_ptr_->set_mip_problem(mip);

    EpisodeStats es = training_episode(mip, rng_);

    stats.avg_reward = es.total_reward;
    stats.avg_time_per_update = get_time_seconds() - start_time;
    stats.total_updates = 1;
    stats.feasible_solutions_found = es.incumbent_updates;
    stats.avg_feasibility_rate = (es.total_steps > 0)
        ? static_cast<double>(es.feasible_count) / es.total_steps : 0.0;
    stats.avg_objective = es.best_obj;

    // Store episode stats for later retrieval
    last_episode_stats_ = es;

    (void)instance_idx;
    return stats;
}

TrainingStats RLTrainer::train() {
    TrainingStats overall_stats;
    double total_start = get_time_seconds();

    // Load training instances
    std::vector<MIPProblem> instances;
    if (config_.verbose) {
        print_training_banner();
        std::cout << "  Loading training instances..." << std::endl;
    }

    for (const auto& file : config_.training_files) {
        try {
            MIPProblem mip;
            mip.load_from_mps(file);
            mip.finalize();
            instances.push_back(mip);
            if (config_.verbose) {
                std::cout << "    ✓ Loaded: " << file
                          << " (" << mip.num_cols << " vars, "
                          << mip.num_rows << " constrs)" << std::endl;
            }
        } catch (const std::exception& e) {
            if (config_.verbose)
                std::cerr << "    ✗ Failed: " << file << ": " << e.what() << std::endl;
        }
    }

    if (instances.empty()) {
        if (config_.verbose) std::cerr << "No training instances loaded." << std::endl;
        return overall_stats;
    }

    if (config_.verbose) {
        std::cout << "  Instances loaded: " << instances.size() << std::endl;
        print_thin_separator();
    }

    // ========== FIXED: Create agent WITHOUT dimensions ==========
    if (agent_ptr_ == nullptr) {
        AgentConfig ac;
        ac.learning_rate = config_.learning_rate;
        ac.rmsprop_alpha = config_.rmsprop_alpha;
        ac.rmsprop_epsilon = config_.rmsprop_epsilon;
        ac.weight_decay = config_.weight_decay;
        ac.hidden_dim = 128;
        ac.num_heads = 4;
        ac.num_layers = 4;
        ac.dropout = 0.1f;
        
        // CRITICAL FIX: Create agent without dimensions!
        agent_ptr_ = new RLAgent(ac);  // No nv, nc parameters!
        
        if (config_.verbose) {
            std::cout << "  Agent initialized " << std::endl;
        }
    }

    // Print table header
    if (config_.verbose) {
        std::cout << std::endl;
        print_training_update_header();
    }

    // Reward history for trend analysis
    std::vector<double> reward_history;
    int total_feasible_found = 0;

    // Accumulate episode stats across all updates for periodic summary
    EpisodeStats accumulated_es;

    // Training loop (Algorithm 2)
    for (int update = 0; update < config_.num_updates; ++update) {
        std::uniform_int_distribution<size_t> dist(0, instances.size() - 1);
        double batch_reward = 0.0;
        int batch_feasible = 0;
        int batch_rollbacks = 0;
        int batch_p1 = 0, batch_p2 = 0;
        double batch_best_obj = std::numeric_limits<double>::infinity();

        for (int b = 0; b < config_.batch_size; ++b) {
            size_t idx = dist(rng_);

            // Set MIP reference for the current instance
            agent_ptr_->set_mip_problem(instances[idx]);
            
            TrainingStats ist = train_on_instance(instances[idx], static_cast<int>(idx));
            batch_reward += ist.avg_reward;

            // Aggregate episode-level stats
            batch_feasible += last_episode_stats_.feasible_count;
            batch_rollbacks += last_episode_stats_.rollback_count;
            batch_p1 += last_episode_stats_.phase1_steps;
            batch_p2 += last_episode_stats_.phase2_steps;
            if (last_episode_stats_.best_obj < batch_best_obj)
                batch_best_obj = last_episode_stats_.best_obj;
            total_feasible_found += last_episode_stats_.incumbent_updates;

            // Print detailed episode summary periodically
            if (config_.verbose && config_.log_interval <= 5) {
                print_episode_summary(last_episode_stats_,
                                      update * config_.batch_size + b);
            }
        }

        double avg_batch_reward = batch_reward / config_.batch_size;
        overall_stats.avg_reward = (overall_stats.avg_reward * update + batch_reward) / (update + 1);
        overall_stats.total_updates = update + 1;
        overall_stats.feasible_solutions_found = total_feasible_found;

        reward_history.push_back(avg_batch_reward);

        // Print compact update row
        if (config_.verbose && update % config_.log_interval == 0) {
            double elapsed = get_time_seconds() - total_start;
            print_training_update_row(
                update, avg_batch_reward, batch_best_obj,
                batch_feasible, batch_rollbacks,
                batch_p1, batch_p2, elapsed);
        }

        TrainingStats step_stats;
        step_stats.avg_reward = avg_batch_reward;
        step_stats.total_updates = update + 1;
        step_stats.feasible_solutions_found = batch_feasible;
        history_.push_back(step_stats);
    }

    // Final summary
    double total_time = get_time_seconds() - total_start;
    if (config_.verbose) {
        print_training_final_summary(
            config_.num_updates, total_time,
            overall_stats.avg_reward,
            reward_history,
            total_feasible_found);
    }

    if (!config_.save_path.empty()) {
        save_model(config_.save_path);
        if (config_.verbose) {
            std::cout << "  Model saved to: " << config_.save_path << std::endl;
        }
    }

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
