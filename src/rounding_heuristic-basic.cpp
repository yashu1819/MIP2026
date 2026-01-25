#include "rounding_heuristic.h"


#include <cmath>
#include <random>
#include <omp.h>

RoundingHeuristic::RoundingHeuristic(const MIPProblem& mip_)
    : mip(&mip_) {
    best_solution.clear();
}


void RoundingHeuristic::set_fractional_solution(
    const std::vector<double>& x)
{
    x_frac = &x;
}


bool RoundingHeuristic::run(int num_attempts)
{
    if (!mip || !x_frac) return false;

    const int n = mip->num_cols;
    const int m = mip->num_rows;

    Solution global_best;
    global_best.clear();

    #pragma omp parallel
    {
        Solution local_best;
        local_best.clear();

        std::vector<double> x_int(n);

        // thread-local RNG
        std::mt19937 rng(1234 + omp_get_thread_num());
        std::uniform_real_distribution<double> uni(0.0, 1.0);

        #pragma omp for schedule(static)
        for (int k = 0; k < num_attempts; ++k) {

            // 1. Rounding step
            double base_threshold = 0.5;
            double noise = 0.1 * (uni(rng) - 0.5);
            double threshold = base_threshold + noise;

            for (int j = 0; j < n; ++j) {
                double v = (*x_frac)[j];

                if (mip->vartype[j] == VarType::BINARY) {
                    x_int[j] = (v >= threshold) ? 1.0 : 0.0;
                }
                else if (mip->vartype[j] == VarType::INTEGER) {
                    x_int[j] = std::round(v);
                }
                else { // continuous
                    x_int[j] = v;
                }

                // bounds
                if (x_int[j] < mip->lb[j]) x_int[j] = mip->lb[j];
                if (x_int[j] > mip->ub[j]) x_int[j] = mip->ub[j];
            }


            // 2. Feasibility check
            bool feasible = true;

            for (int i = 0; i < m; ++i) {
                double activity = 0.0;

                for (int p = mip->csr_row_ptr[i];
                     p < mip->csr_row_ptr[i + 1]; ++p) {
                    activity += mip->csr_val[p] *
                                x_int[mip->csr_col_idx[p]];
                }

                if (activity > mip->b[i] + 1e-6) {
                    feasible = false;
                    break;
                }
            }

            if (!feasible) continue;

            // integrality check
            for (int j = 0; j < n; ++j) {
                if (mip->vartype[j] != VarType::CONTINUOUS) {
                    if (std::fabs(x_int[j] - std::round(x_int[j])) > 1e-5) {
                        feasible = false;
                        break;
                    }
                }
            }

            if (!feasible) continue;

            double obj = mip->obj_offset;       // 3. Objective
            for (int j = 0; j < n; ++j)
                obj += mip->c[j] * x_int[j];


            if (!local_best.feasible || obj < local_best.obj_value) { // 4. Thread-local best
                local_best.feasible = true;
                local_best.obj_value = obj;
                local_best.x = x_int;
            }
        }

        // 5. Global reduction
        #pragma omp critical
        {
            if (local_best.feasible &&
                (!global_best.feasible ||
                 local_best.obj_value < global_best.obj_value)) {
                global_best = local_best;
            }
        }
    }

    best_solution = global_best;
    return best_solution.feasible;
}
