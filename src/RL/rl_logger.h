#ifndef RL_LOGGER_H
#define RL_LOGGER_H

#include "rl_state.h"
#include "../Definition/mip_problem.h"
#include <vector>
#include <string>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <sstream>
#include <algorithm>
#include <numeric>

namespace rl_sph {

// ============================================================
//  Incumbent record — stores every improvement event
// ============================================================
struct IncumbentRecord {
    double obj_value;
    double timestamp;   // seconds since start
    int    step;
    int    phase;
    bool   feasible;
};

// ============================================================
//  Episode statistics — filled during one training episode
// ============================================================
struct EpisodeStats {
    double total_reward        = 0.0;
    int    total_steps         = 0;
    int    feasible_count      = 0;   // steps that were feasible
    int    rollback_count      = 0;
    int    incumbent_updates   = 0;
    int    phase1_steps        = 0;
    int    phase2_steps        = 0;
    double best_obj            = std::numeric_limits<double>::infinity();
    double worst_obj           = -std::numeric_limits<double>::infinity();
    double sum_reward          = 0.0;
    double min_reward          = std::numeric_limits<double>::infinity();
    double max_reward          = -std::numeric_limits<double>::infinity();
    double elapsed_seconds     = 0.0;
    int    constraint_violations_start = 0;
    int    constraint_violations_end   = 0;
    int    bound_violations_start      = 0;
    int    bound_violations_end        = 0;
    std::vector<IncumbentRecord> incumbent_trace;
};

// ============================================================
//  Primal Integral computation
//  PI = integral from 0 to T of p(t) dt
//  where p(t) = (best_known - incumbent(t)) / |best_known|
//  Using incumbent_trace for piecewise-constant interpolation.
// ============================================================
inline double compute_primal_integral(
    const std::vector<IncumbentRecord>& trace,
    double total_time,
    double reference_obj = std::numeric_limits<double>::infinity())
{
    if (trace.empty() || total_time <= 0.0) return 0.0;

    // Use best found objective as reference if none given
    double best_obj = reference_obj;
    if (std::isinf(best_obj)) {
        for (const auto& r : trace)
            if (r.feasible) best_obj = std::min(best_obj, r.obj_value);
    }
    if (std::isinf(best_obj)) return 0.0;  // no feasible solution ever

    double denom = std::max(std::abs(best_obj), 1e-10);
    double integral = 0.0;
    double prev_time = 0.0;
    double prev_gap  = 1.0;   // gap before first incumbent

    for (const auto& r : trace) {
        if (!r.feasible) continue;
        double t = r.timestamp;
        if (t > total_time) break;
        // accumulate gap from prev_time to t
        integral += prev_gap * (t - prev_time);
        prev_time = t;
        prev_gap  = std::max(0.0, (r.obj_value - best_obj) / denom);
    }
    // remaining time after last update
    integral += prev_gap * (total_time - prev_time);
    return integral;
}

// ============================================================
//  Constraint / bound violation helpers
// ============================================================
inline int count_violated_constraints(const MIPProblem& mip, const RLState& state, double tol = 1e-6) {
    int cnt = 0;
    for (int j = 0; j < state.m; ++j)
        if (state.f[j] < -tol) ++cnt;
    return cnt;
}

inline int count_violated_bounds(const MIPProblem& mip, const RLState& state, double tol = 1e-6) {
    int cnt = 0;
    for (int i = 0; i < state.n; ++i)
        if (state.x[i] < mip.lb[i] - tol || state.x[i] > mip.ub[i] + tol) ++cnt;
    return cnt;
}

inline double max_constraint_violation(const RLState& state) {
    double mv = 0.0;
    for (int j = 0; j < state.m; ++j)
        mv = std::max(mv, -state.f[j]);
    return mv;
}

// ============================================================
//  Formatted printing utilities
// ============================================================

// Simple horizontal line
inline void print_separator(int width = 72) {
    std::cout << std::string(width, '=') << std::endl;
}

inline void print_thin_separator(int width = 72) {
    std::cout << std::string(width, '-') << std::endl;
}

// Training banner
inline void print_training_banner() {
    print_separator();
    std::cout << "  RL-SPH TRAINING — Detailed Diagnostics Enabled" << std::endl;
    print_separator();
}

// Inference banner
inline void print_inference_banner() {
    print_separator();
    std::cout << "  RL-SPH INFERENCE — Detailed Diagnostics Enabled" << std::endl;
    print_separator();
}

// ============================================================
//  Training: per-update log line (compact table row)
// ============================================================
inline void print_training_update_header() {
    std::cout << std::left
              << std::setw(8)  << "Update"
              << std::setw(12) << "AvgReward"
              << std::setw(10) << "BestObj"
              << std::setw(10) << "Feasible"
              << std::setw(10) << "Rollbacks"
              << std::setw(10) << "P1Steps"
              << std::setw(10) << "P2Steps"
              << std::setw(10) << "Time(s)"
              << std::endl;
    print_thin_separator();
}

inline void print_training_update_row(
    int update, double avg_reward, double best_obj,
    int feasible_count, int rollback_count,
    int p1_steps, int p2_steps, double elapsed)
{
    std::cout << std::left << std::fixed << std::setprecision(4)
              << std::setw(8)  << update
              << std::setw(12) << avg_reward
              << std::setw(10) << (std::isinf(best_obj) ? "inf" : (std::ostringstream() << std::fixed << std::setprecision(2) << best_obj).str())
              << std::setw(10) << feasible_count
              << std::setw(10) << rollback_count
              << std::setw(10) << p1_steps
              << std::setw(10) << p2_steps
              << std::setw(10) << std::setprecision(2) << elapsed
              << std::endl;
}

// ============================================================
//  Training: detailed episode summary (every N episodes)
// ============================================================
inline void print_episode_summary(const EpisodeStats& es, int episode_idx) {
    print_thin_separator();
    std::cout << "  Episode " << episode_idx << " Summary" << std::endl;
    print_thin_separator();
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Steps:              " << es.total_steps
              << "  (Phase1=" << es.phase1_steps << ", Phase2=" << es.phase2_steps << ")" << std::endl;
    std::cout << "  Total reward:       " << es.total_reward << std::endl;
    std::cout << "  Reward range:       [" << es.min_reward << ", " << es.max_reward << "]" << std::endl;
    std::cout << "  Feasible steps:     " << es.feasible_count << " / " << es.total_steps << std::endl;
    std::cout << "  Rollbacks:          " << es.rollback_count << std::endl;
    std::cout << "  Incumbent updates:  " << es.incumbent_updates << std::endl;
    std::cout << "  Best objective:     ";
    if (std::isinf(es.best_obj)) std::cout << "(none found)";
    else std::cout << es.best_obj;
    std::cout << std::endl;
    std::cout << "  Constr violations:  " << es.constraint_violations_start
              << " -> " << es.constraint_violations_end << std::endl;
    std::cout << "  Bound violations:   " << es.bound_violations_start
              << " -> " << es.bound_violations_end << std::endl;
    std::cout << "  Episode time:       " << std::setprecision(3) << es.elapsed_seconds << "s" << std::endl;

    // Primal integral
    if (!es.incumbent_trace.empty()) {
        double pi = compute_primal_integral(es.incumbent_trace, es.elapsed_seconds);
        std::cout << "  Primal integral:    " << std::setprecision(6) << pi << std::endl;
    }
    print_thin_separator();
}

// ============================================================
//  Training: final training summary
// ============================================================
inline void print_training_final_summary(
    int total_updates, double total_time,
    double final_avg_reward,
    const std::vector<double>& reward_history,
    int total_feasible_found)
{
    print_separator();
    std::cout << "  TRAINING COMPLETE" << std::endl;
    print_separator();
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Total updates:         " << total_updates << std::endl;
    std::cout << "  Total time:            " << std::setprecision(2) << total_time << "s" << std::endl;
    std::cout << "  Avg time/update:       " << std::setprecision(4)
              << (total_updates > 0 ? total_time / total_updates : 0.0) << "s" << std::endl;
    std::cout << "  Final avg reward:      " << final_avg_reward << std::endl;

    if (!reward_history.empty()) {
        double first_10 = 0.0, last_10 = 0.0;
        int n = static_cast<int>(reward_history.size());
        int window = std::min(n, 10);
        for (int i = 0; i < window; ++i) first_10 += reward_history[i];
        for (int i = n - window; i < n; ++i) last_10 += reward_history[i];
        first_10 /= window;
        last_10  /= window;
        std::cout << "  Reward (first " << window << "):     " << first_10 << std::endl;
        std::cout << "  Reward (last  " << window << "):     " << last_10 << std::endl;
        std::cout << "  Reward trend:          "
                  << (last_10 > first_10 ? "IMPROVING ↑" : (last_10 < first_10 ? "DECLINING ↓" : "STABLE →"))
                  << std::endl;
    }
    std::cout << "  Feasible solutions:    " << total_feasible_found << std::endl;
    print_separator();
}

// ============================================================
//  Inference: step-level logging (every N steps)
// ============================================================
inline void print_inference_step(
    int step, int phase, double obj, double incumbent_obj,
    bool feasible, int violated_constraints, int violated_bounds,
    double reward, double elapsed)
{
    std::cout << std::fixed << std::setprecision(4)
              << "  [Step " << std::setw(5) << step << "]"
              << "  Phase=" << phase
              << "  Obj=" << std::setw(12) << obj
              << "  Inc=";
    if (std::isinf(incumbent_obj)) std::cout << std::setw(12) << "inf";
    else std::cout << std::setw(12) << incumbent_obj;
    std::cout << "  Feas=" << (feasible ? "Y" : "N")
              << "  ViolC=" << std::setw(4) << violated_constraints
              << "  ViolB=" << std::setw(4) << violated_bounds
              << "  R=" << std::setw(10) << reward
              << "  t=" << std::setprecision(2) << elapsed << "s"
              << std::endl;
}

// ============================================================
//  Inference: final detailed report
// ============================================================
inline void print_inference_report(
    const MIPProblem& mip,
    const std::vector<double>& solution,
    double obj_value, bool feasible,
    double elapsed, int total_steps,
    const std::vector<IncumbentRecord>& trace,
    int phase1_steps, int phase2_steps,
    int rollback_count)
{
    print_separator();
    std::cout << "  RL-SPH INFERENCE REPORT" << std::endl;
    print_separator();

    // Problem info
    int nb = 0, ni = 0, nc = 0;
    for (int i = 0; i < mip.num_cols; ++i) {
        if (mip.vartype[i] == VarType::BINARY) nb++;
        else if (mip.vartype[i] == VarType::INTEGER) ni++;
        else nc++;
    }
    std::cout << "  Problem:         " << mip.num_cols << " vars ("
              << nb << " bin, " << ni << " int, " << nc << " cont), "
              << mip.num_rows << " constraints" << std::endl;

    // Solution info
    std::cout << std::fixed;
    std::cout << "  Status:          " << (feasible ? "FEASIBLE" : "INFEASIBLE") << std::endl;
    std::cout << "  Objective:       " << std::setprecision(6) << obj_value << std::endl;
    std::cout << "  Total steps:     " << total_steps
              << " (Phase1=" << phase1_steps << ", Phase2=" << phase2_steps << ")" << std::endl;
    std::cout << "  Rollbacks:       " << rollback_count << std::endl;
    std::cout << "  Elapsed time:    " << std::setprecision(3) << elapsed << "s" << std::endl;

    // Non-zero count
    int nonzero = 0;
    for (double v : solution)
        if (std::abs(v) > 1e-6) nonzero++;
    std::cout << "  Non-zero vars:   " << nonzero << " / " << mip.num_cols << std::endl;

    // Constraint analysis
    if (feasible) {
        RLState final_state = create_state(mip, solution);
        double min_slack = std::numeric_limits<double>::infinity();
        int tight_count = 0;
        for (int j = 0; j < final_state.m; ++j) {
            min_slack = std::min(min_slack, final_state.f[j]);
            if (std::abs(final_state.f[j]) < 1e-6) tight_count++;
        }
        std::cout << "  Min slack:       " << std::setprecision(6) << min_slack << std::endl;
        std::cout << "  Tight constrs:   " << tight_count << " / " << mip.num_rows << std::endl;
    }

    // Incumbent trace
    if (!trace.empty()) {
        print_thin_separator();
        std::cout << "  Incumbent Improvement Trace:" << std::endl;
        std::cout << "  " << std::left
                  << std::setw(8)  << "#"
                  << std::setw(14) << "Objective"
                  << std::setw(10) << "Step"
                  << std::setw(8)  << "Phase"
                  << std::setw(12) << "Time(s)"
                  << std::endl;
        for (size_t i = 0; i < trace.size(); ++i) {
            std::cout << "  " << std::left << std::fixed
                      << std::setw(8)  << i + 1
                      << std::setw(14) << std::setprecision(4) << trace[i].obj_value
                      << std::setw(10) << trace[i].step
                      << std::setw(8)  << trace[i].phase
                      << std::setw(12) << std::setprecision(3) << trace[i].timestamp
                      << std::endl;
        }

        // Primal integral
        double pi = compute_primal_integral(trace, elapsed);
        std::cout << std::endl;
        std::cout << "  Primal Integral: " << std::setprecision(6) << pi << std::endl;
    }

    print_separator();
}

// ============================================================
//  Inference: progress bar (simple text-based)
// ============================================================
inline void print_progress_bar(int current, int total, double elapsed, int bar_width = 40) {
    float progress = static_cast<float>(current) / std::max(total, 1);
    int filled = static_cast<int>(progress * bar_width);

    std::cout << "\r  [";
    for (int i = 0; i < bar_width; ++i) {
        if (i < filled) std::cout << "█";
        else std::cout << "░";
    }
    std::cout << "] " << std::fixed << std::setprecision(1)
              << (progress * 100.0f) << "% "
              << current << "/" << total
              << "  " << std::setprecision(1) << elapsed << "s" << std::flush;
}

// ============================================================
//  Training: step-level log for within an episode
// ============================================================
inline void print_training_step(
    int step, int phase, double reward, double obj,
    bool feasible, double incumbent_obj, int num_changeable)
{
    std::cout << std::fixed << std::setprecision(4)
              << "    [T-Step " << std::setw(4) << step << "]"
              << " Ph=" << phase
              << " R=" << std::setw(10) << reward
              << " Obj=" << std::setw(12) << obj
              << " Feas=" << (feasible ? "Y" : "N")
              << " Inc=";
    if (std::isinf(incumbent_obj)) std::cout << "inf";
    else std::cout << std::setw(12) << incumbent_obj;
    std::cout << " Vars=" << num_changeable
              << std::endl;
}

} // namespace rl_sph

#endif // RL_LOGGER_H
