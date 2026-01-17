#include "feasibility_pump.h"

#include <cmath>
#include <chrono>

/* ================= Constructor ================= */

FeasibilityPump::FeasibilityPump(const MIPProblem& mip)
    : mip_(mip), lp_(mip), last_runtime_sec_(0.0) {}

/* ================= Helpers ================= */

bool FeasibilityPump::is_integral(
    const std::vector<double>& x,
    double tol
) const {
    for (int j = 0; j < mip_.num_cols; ++j) {
        if (mip_.vartype[j] != VarType::CONTINUOUS) {
            double r = std::round(x[j]);
            if (std::fabs(x[j] - r) > tol) {
                return false;
            }
        }
    }
    return true;
}

std::vector<double> FeasibilityPump::round_solution(
    const std::vector<double>& x
) const {
    std::vector<double> xr = x;

    for (int j = 0; j < mip_.num_cols; ++j) {
        if (mip_.vartype[j] == VarType::BINARY) {
            xr[j] = (x[j] >= 0.5) ? 1.0 : 0.0;
        } else if (mip_.vartype[j] == VarType::INTEGER) {
            xr[j] = std::round(x[j]);
        }
        // enforce bounds
        if (xr[j] < mip_.lb[j]) xr[j] = mip_.lb[j];
        if (xr[j] > mip_.ub[j]) xr[j] = mip_.ub[j];
    }

    return xr;
}

/* ================= Main FP Loop ================= */

bool FeasibilityPump::run(
    Solution& sol,
    int max_iters,
    double integrality_tol
) {
    auto t0 = std::chrono::steady_clock::now();
    sol.clear();

    // Step 1: LP relaxation
    if (!lp_.solve()) {
        return false;
    }

    std::vector<double> x_lp = lp_.x;

    for (int iter = 0; iter < max_iters; ++iter) {

        // Step 2: rounding
        std::vector<double> x_int = round_solution(x_lp);

        // Step 3: integrality check
        if ((is_integral(x_int, integrality_tol))  && mip_.check_feasible(x_int, 1e-6, integrality_tol)  ) {

            double obj = mip_.obj_offset;
            for (int j = 0; j < mip_.num_cols; ++j) {
                obj += mip_.c[j] * x_int[j];
            }

            sol.feasible = true;
            sol.x = std::move(x_int);
            sol.obj_value = obj;

            auto t1 = std::chrono::steady_clock::now();
            last_runtime_sec_ =
                std::chrono::duration<double>(t1 - t0).count();
            return true;
        }

        // Step 4: distance objective (L1 pull)
        for (int j = 0; j < mip_.num_cols; ++j) {
            if (mip_.vartype[j] != VarType::CONTINUOUS) {
                double target = std::round(x_lp[j]);
                lp_.c[j] = (x_lp[j] >= target) ? 1.0 : -1.0;
            } else {
                lp_.c[j] = mip_.c[j];
            }
        }

        // Step 5: re-solve LP
        if (!lp_.solve()) {
            break;
        }

        x_lp = lp_.x;
    }

    auto t1 = std::chrono::steady_clock::now();
    last_runtime_sec_ =
        std::chrono::duration<double>(t1 - t0).count();

    return false;
}

