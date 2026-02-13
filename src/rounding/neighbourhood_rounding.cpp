#include "neighbourhood_rounding.h"
#include "mip_problem.h"
#include "solution.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <vector>


void compute_locks(
    const MIPProblem& mip,
    std::vector<int>& locks_up,
    std::vector<int>& locks_down
)
{
    int n = mip.num_cols;
    int m = mip.num_rows;

    locks_up.assign(n, 0);
    locks_down.assign(n, 0);

    for (int row = 0; row < m; row++) {
        for (int idx = mip.csr_row_ptr[row];
             idx < mip.csr_row_ptr[row+1];
             idx++)
        {
            int col = mip.csr_col_idx[idx];
            double val = mip.csr_val[idx];

            if (val > 0)
                locks_up[col]++;
            else if (val < 0)
                locks_down[col]++;
        }
    }
}


bool repair_solution(
    const MIPProblem& mip,
    std::vector<double>& x,
    const std::vector<bool>& is_free,
    const std::vector<int>& locks_up,
    const std::vector<int>& locks_down,
    int max_iter = 5000
)
{
    int n = mip.num_cols;
    int m = mip.num_rows;

    std::vector<double> activity(m, 0.0);

    // compute initial activities
    for (int row = 0; row < m; row++) {
        for (int idx = mip.csr_row_ptr[row];
             idx < mip.csr_row_ptr[row+1];
             idx++)
        {
            int col = mip.csr_col_idx[idx];
            activity[row] += mip.csr_val[idx] * x[col];
        }
    }

    for (int iter = 0; iter < max_iter; iter++) {

        // find most violated constraint
        int worst_row = -1;
        double worst_viol = 0.0;

        for (int row = 0; row < m; row++) {
            double viol = activity[row] - mip.b[row];
            if (viol > worst_viol) {
                worst_viol = viol;
                worst_row = row;
            }
        }

        if (worst_row == -1)
            return true;  // feasible

        int best_var = -1;
        int best_score = 1e9;
        int best_dir = 0;

        // search free variables in violated row
        for (int idx = mip.csr_row_ptr[worst_row];
             idx < mip.csr_row_ptr[worst_row+1];
             idx++)
        {
            int col = mip.csr_col_idx[idx];
            double coeff = mip.csr_val[idx];

            if (!is_free[col])
                continue;

            int direction = 0;
            int lock_val = 0;

            if (coeff > 0) {
                direction = -1;
                lock_val = locks_down[col];
            }
            else if (coeff < 0) {
                direction = +1;
                lock_val = locks_up[col];
            }

            if (lock_val < best_score) {
                best_score = lock_val;
                best_var = col;
                best_dir = direction;
            }
        }

        if (best_var == -1)
            return false;  // no repair possible

        // apply move
        x[best_var] += best_dir;
        double delta = best_dir;

        // incremental update via CSC
        for (int idx = mip.csc_col_ptr[best_var];
             idx < mip.csc_col_ptr[best_var+1];
             idx++)
        {
            int row = mip.csc_row_idx[idx];
            activity[row] += mip.csc_val[idx] * delta;
        }
    }

    return false;
}