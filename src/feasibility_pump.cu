#include "feasibility_pump.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <random>
#include <unordered_set>

/* ================= CLASS IMPLEMENTATION ================= */

FeasibilityPump::FeasibilityPump(const MIPProblem& p)
    : prob(p), aux_prob(p)
{
    x.resize(prob.num_cols);
    x_lp.resize(prob.num_cols);
    residuals.resize(prob.num_rows);
    original_c = prob.c;
}

FeasibilityPump::~FeasibilityPump() {
}

/* ================= CORE METHODS ================= */

void FeasibilityPump::solve_lp() {
    LPRelaxation lp;
    lp.build_from_mip(aux_prob);
    lp.solve();
    x_lp = lp.x;
    x = x_lp;
}

void FeasibilityPump::round_x() {
    for (int j = 0; j < prob.num_cols; ++j) {
        if (prob.vartype[j] == VarType::BINARY)
            x[j] = (x[j] < 0.5) ? 0.0 : 1.0;
        else if (prob.vartype[j] == VarType::INTEGER)
            x[j] = std::round(x[j]);
        x[j] = std::max(prob.lb[j], std::min(prob.ub[j], x[j]));
    }
}

void FeasibilityPump::compute_residuals() {
    for (int i = 0; i < prob.num_rows; ++i) {
        double ax = 0.0;
        int start = prob.csr_row_ptr[i];
        int end = prob.csr_row_ptr[i + 1];
        for (int k = start; k < end; ++k)
            ax += prob.csr_val[k] * x[prob.csr_col_idx[k]];
        residuals[i] = ax - prob.b[i];
    }
}

bool FeasibilityPump::check_feasibility(double constr_tol) {
    compute_residuals();
    for (int i = 0; i < prob.num_rows; ++i)
        if (std::abs(residuals[i]) > constr_tol)
            return false;
    return prob.check_feasible(x);
}

uint64_t FeasibilityPump::hash_rounded() const {
    uint64_t h = 0;
    for (double v : x) {
        uint64_t iv = static_cast<uint64_t>(std::round(v));
        h ^= iv + 0x9e3779b9ULL + (h << 6) + (h >> 2);
    }
    return h;
}

void FeasibilityPump::perturb() {
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    const double flip_prob = 0.15;

    for (int j = 0; j < prob.num_cols; ++j) {
        if (prob.vartype[j] == VarType::CONTINUOUS) continue;
        double frac = x_lp[j] - std::floor(x_lp[j]);
        if (dis(gen) < flip_prob && std::abs(frac - 0.5) < 0.35) {
            if (prob.vartype[j] == VarType::BINARY) {
                x[j] = 1.0 - x[j];
            } else if (prob.vartype[j] == VarType::INTEGER) {
                // Simple perturbation for general integer: shift by +/-1 with bounds check
                double shift = (dis(gen) < 0.5) ? -1.0 : 1.0;
                x[j] = std::max(prob.lb[j], std::min(prob.ub[j], x[j] + shift));
            }
        }
    }
}

void FeasibilityPump::perturb_objective() {
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<> dis(-0.1, 0.1);
    aux_prob.c = original_c;
    for (int j = 0; j < prob.num_cols; ++j) {
        if (prob.vartype[j] != VarType::CONTINUOUS) {
            aux_prob.c[j] += dis(gen);
        }
    }
}

/* ================= MAIN LOOP ================= */

Solution FeasibilityPump::run(const FeasibilityPumpParams& params) {
    for (int r = 0; r < params.max_restarts; ++r) {
        if (r > 0) {
            perturb_objective();
        }
        solve_lp();  // sets x_lp and x = x_lp
        seen_rounded.clear();

        for (int it = 0; it < params.max_iters; ++it) {
            round_x();

            if (check_feasibility(params.constr_tol)) {
                return {true, x};
            }

            uint64_t h = hash_rounded();
            if (seen_rounded.count(h)) {
                perturb();
                seen_rounded.clear();
            }
            seen_rounded.insert(h);

            // Set objective for projection LP: min sum |x_j - xbar_j| for integer vars
            std::vector<double> saved_c = aux_prob.c;
            for (int j = 0; j < prob.num_cols; ++j) {
                if (prob.vartype[j] == VarType::CONTINUOUS || prob.vartype[j] == VarType::INTEGER) {
                    aux_prob.c[j] = 0.0;  // Ignore general integer for now (requires aux vars)
                } else {  // BINARY
                    double bar = x[j];
                    aux_prob.c[j] = (bar < 0.5) ? 1.0 : -1.0;
                }
            }

            // Solve projection LP
            LPRelaxation lp;
            lp.build_from_mip(aux_prob);
            lp.solve();
            x = lp.x;
            x_lp = x;  // Update for perturbation reference

            // Restore original (possibly perturbed) c
            aux_prob.c = saved_c;
        }
    }

    Solution sol;
    sol.feasible = false;
    return sol;
}
