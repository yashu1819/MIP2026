#include "gurobi_lp.h"
#include <cmath>
#include <limits>
#include <iostream>

using namespace milp;

// NOTE: This file contains two possibilities:
// - If you compile with -DGUROBI_AVAILABLE and link Gurobi, you can replace the fallback
//   implementation with actual Gurobi API usage.
// - Otherwise this fallback does a tiny brute-force search for small problems (demo only).

GurobiLPSolver::GurobiLPSolver()
: time_limit_seconds_(10.0)
{}

GurobiLPSolver::~GurobiLPSolver() {}

void GurobiLPSolver::load_problem(const MILPProblem& p) {
    problem_copy_ = p; // shallow copy fine for our types
}

void GurobiLPSolver::set_time_limit(double seconds) {
    time_limit_seconds_ = seconds;
}

// Simple brute-force LP relaxation solver for tiny problems (<=4 vars).
// It ignores integrality constraints (LP relaxation).
Solution GurobiLPSolver::solve() {
    const MILPProblem& p = problem_copy_;
    Solution sol;
    int n = p.num_vars();
    sol.x.assign(n, 0.0);
    sol.feasible = false;
    sol.obj_value = std::numeric_limits<double>::infinity();

    if (n == 0) {
        sol.feasible = true;
        sol.obj_value = 0.0;
        return sol;
    }

    if (n > 6) {
        // Fallback: set continuous solution as midpoint of bounds (simple heuristic)
        for (int i = 0; i < n; ++i) {
            double lb = p.vars[i].lb;
            double ub = p.vars[i].ub;
            if (!std::isfinite(lb) && !std::isfinite(ub)) sol.x[i] = 0.0;
            else if (!std::isfinite(lb)) sol.x[i] = ub;
            else if (!std::isfinite(ub)) sol.x[i] = lb;
            else sol.x[i] = 0.5*(lb + ub);
        }
        // compute objective and feasibility
        bool ok = true;
        for (const auto& c : p.cons) {
            double s = 0.0;
            for (size_t j = 0; j < c.var_idx.size(); ++j) {
                s += c.coeff[j] * sol.x[c.var_idx[j]];
            }
            if (s + 1e-9 < c.lb || s - 1e-9 > c.ub) { ok = false; break; }
        }
        sol.feasible = ok;
        double obj = 0.0;
        for (int i = 0; i < n; ++i) obj += p.obj[i] * sol.x[i];
        sol.obj_value = (p.minimize ? obj : -obj);
        return sol;
    }

    // Grid search
    const int STEPS = 50; // coarse; increase for accuracy
    std::vector<double> cur(n, 0.0);
    std::vector<double> step(n, 1.0);
    std::vector<double> lb(n), ub(n);
    for (int i = 0; i < n; ++i) {
        lb[i] = p.vars[i].lb;
        ub[i] = p.vars[i].ub;
        if (!std::isfinite(lb[i])) lb[i] = -10.0;
        if (!std::isfinite(ub[i])) ub[i] = 10.0;
        step[i] = (ub[i] - lb[i]) / static_cast<double>(STEPS);
        if (step[i] <= 0) step[i] = 1.0;
    }

    // recursive enumeration
    std::function<void(int)> dfs = [&](int idx) {
        if (idx == n) {
            // check constraints
            for (const auto& c : p.cons) {
                double s = 0.0;
                for (size_t j = 0; j < c.var_idx.size(); ++j) {
                    s += c.coeff[j] * cur[c.var_idx[j]];
                }
                if (s + 1e-9 < c.lb || s - 1e-9 > c.ub) return;
            }
            double obj = 0.0;
            for (int i = 0; i < n; ++i) obj += p.obj[i] * cur[i];
            double val = p.minimize ? obj : -obj;
            if (val < sol.obj_value) {
                sol.obj_value = val;
                sol.x = cur;
                sol.feasible = true;
            }
            return;
        }
        // iterate
        for (int s = 0; s <= STEPS; ++s) {
            double v = lb[idx] + s * step[idx];
            if (v < p.vars[idx].lb - 1e-9) continue;
            if (v > p.vars[idx].ub + 1e-9) break;
            cur[idx] = v;
            dfs(idx+1);
        }
    };

    dfs(0);
    return sol;
}
