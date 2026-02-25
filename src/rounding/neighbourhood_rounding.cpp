// #include "neighbourhood_rounding.h"
// #include "mip_problem.h"
// #include "solution.h"
// #include <cuda_runtime.h>
// #include <algorithm>
// #include <iostream>
// #include <vector>


// void compute_locks(
//     const MIPProblem& mip,
//     std::vector<int>& locks_up,
//     std::vector<int>& locks_down
// )
// {
//     int n = mip.num_cols;
//     int m = mip.num_rows;

//     locks_up.assign(n, 0);
//     locks_down.assign(n, 0);

//     for (int row = 0; row < m; row++) {
//         for (int idx = mip.csr_row_ptr[row];
//              idx < mip.csr_row_ptr[row+1];
//              idx++)
//         {
//             int col = mip.csr_col_idx[idx];
//             double val = mip.csr_val[idx];

//             if (val > 0)
//                 locks_up[col]++;
//             else if (val < 0)
//                 locks_down[col]++;
//         }
//     }
// }


// bool repair_solution(
//     const MIPProblem& mip,
//     std::vector<double>& x,
//     const std::vector<bool>& is_free,
//     const std::vector<int>& locks_up,
//     const std::vector<int>& locks_down,
//     int max_iter = 10000
// )
// {
//     int n = mip.num_cols;
//     int m = mip.num_rows;

//     std::vector<double> activity(m, 0.0);

//     // compute initial activities
//     for (int row = 0; row < m; row++) {
//         for (int idx = mip.csr_row_ptr[row];
//              idx < mip.csr_row_ptr[row+1];
//              idx++)
//         {
//             int col = mip.csr_col_idx[idx];
//             activity[row] += mip.csr_val[idx] * x[col];
//         }
//     }

//     for (int iter = 0; iter < max_iter; iter++) {

//         // find most violated constraint
//         int worst_row = -1;
//         double worst_viol = 0.0;

//         for (int row = 0; row < m; row++) {
//             double viol = activity[row] - mip.b[row];
//             if (viol > worst_viol) {
//                 worst_viol = viol;
//                 worst_row = row;
//             }
//         }

//         if (worst_row == -1)
//             return true;  // feasible

//         int best_var = -1;
//         int best_score = 1e9;
//         int best_dir = 0;

//         // search free variables in violated row
//         for (int idx = mip.csr_row_ptr[worst_row];
//              idx < mip.csr_row_ptr[worst_row+1];
//              idx++)
//         {
//             int col = mip.csr_col_idx[idx];
//             double coeff = mip.csr_val[idx];

//             if (!is_free[col])
//                 continue;

//             int direction = 0;
//             int lock_val = 0;

//             if (coeff > 0) {
//                 direction = -1;
//                 lock_val = locks_down[col];
//             }
//             else if (coeff < 0) {
//                 direction = +1;
//                 lock_val = locks_up[col];
//             }

//             if (lock_val < best_score) {
//                 best_score = lock_val;
//                 best_var = col;
//                 best_dir = direction;
//             }
//         }

//         if (best_var == -1)
//             return false;  // no repair possible

//         // apply move
//         x[best_var] += best_dir;
//         double delta = best_dir;

//         // incremental update via CSC
//         for (int idx = mip.csc_col_ptr[best_var];
//              idx < mip.csc_col_ptr[best_var+1];
//              idx++)
//         {
//             int row = mip.csc_row_idx[idx];
//             activity[row] += mip.csc_val[idx] * delta;
//         }
//     }

//     return false;
// }



#include<cmath>
#include "neighbourhood_rounding.h"
#include "mip_problem.h"
#include "solution.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <queue>

using namespace std;

// CORRECTED: Lock computation
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

            // FIXED: Correct lock semantics
            if (val > 0) {
                // Positive coeff: decreasing x_j helps, increasing hurts
                // So this constrains rounding UP
                locks_up[col]++;
            }
            else if (val < 0) {
                // Negative coeff: increasing x_j helps, decreasing hurts  
                // So this constrains rounding DOWN
                locks_down[col]++;
            }
        }
    }
}

// ENHANCED: Lock-aware initial rounding
std::vector<double> get_lock_based_rounding(
    const std::vector<double>& x_lp,
    const std::vector<int>& locks_up,
    const std::vector<int>& locks_down,
    const std::vector<VarType>& vartype,
    double eps
) {
    int n = x_lp.size();
    std::vector<double> x(n);
    
    for (int i = 0; i < n; i++) {
        if (vartype[i] == VarType::CONTINUOUS) {
            x[i] = x_lp[i];
        } else {
            double floor_val = std::floor(static_cast<double>(x_lp[i]));
            double ceil_val = std::ceil(static_cast<double>(x_lp[i]));
            double nearest = std::round(static_cast<double>(x_lp[i]));
            
            // If within eps of integer, fix it
            if (std::abs(x_lp[i] - nearest) <= eps) {
                x[i] = nearest;
            }
            // Otherwise use lock information
            else if (locks_up[i] < locks_down[i]) {
                // Fewer constraints block rounding up → round up
                x[i] = ceil_val;
            }
            else if (locks_down[i] < locks_up[i]) {
                // Fewer constraints block rounding down → round down
                x[i] = floor_val;
            }
            else {
                // Tie: round to nearest
                x[i] = nearest;
            }
        }
    }
    return x;
}

// ENHANCED: Better repair with priority queue for constraint selection
bool repair_solution(
    const MIPProblem& mip,
    std::vector<double>& x,
    const std::vector<bool>& is_free,
    const std::vector<int>& locks_up,
    const std::vector<int>& locks_down,
    int max_iter = 10000
)
{
    int n = mip.num_cols;
    int m = mip.num_rows;

    std::vector<double> activity(m, 0.0);
    std::vector<bool> variable_is_free = is_free;  // mutable copy

    // Compute initial activities
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
        // Find most violated constraint
        int worst_row = -1;
        double worst_viol = 1e-9;  // small tolerance

        for (int row = 0; row < m; row++) {
            double viol = activity[row] - mip.b[row];
            if (viol > worst_viol) {
                worst_viol = viol;
                worst_row = row;
            }
        }

        if (worst_row == -1)
            return true;  // feasible

        // ENHANCED: Better variable selection with scoring
        int best_var = -1;
        double best_score = std::numeric_limits<double>::infinity();
        int best_dir = 0;
        double best_improvement = 0;

        for (int idx = mip.csr_row_ptr[worst_row];
             idx < mip.csr_row_ptr[worst_row+1];
             idx++)
        {
            int col = mip.csr_col_idx[idx];
            double coeff = mip.csr_val[idx];

            if (!variable_is_free[col])
                continue;

            // Determine direction and score
            int direction = 0;
            int lock_val = 0;
            double improvement = 0;

            if (coeff > 0 && x[col] > mip.lb[col]) {
                // Can decrease x[col] to reduce LHS
                direction = -1;
                lock_val = locks_down[col];
                improvement = coeff;  // How much we reduce activity
            }
            else if (coeff < 0 && x[col] < mip.ub[col]) {
                // Can increase x[col] to reduce LHS (since coeff < 0)
                direction = +1;
                lock_val = locks_up[col];
                improvement = -coeff;
            }

            if (direction != 0) {
                // Score: prefer fewer locks, more improvement
                double score = lock_val - 0.1 * improvement;
                
                if (score < best_score) {
                    best_score = score;
                    best_var = col;
                    best_dir = direction;
                    best_improvement = improvement;
                }
            }
        }

        if (best_var == -1) {
            // No repair possible for this constraint
            // ENHANCED: Try to find any variable that can help, even if not optimal
            return false;
        }

        // Apply move
        double old_val = x[best_var];
        x[best_var] += best_dir;
        
        // ENHANCED: Check bounds after move
        x[best_var] = std::max(mip.lb[best_var], 
                               std::min(mip.ub[best_var], x[best_var]));
        
        double actual_delta = x[best_var] - old_val;
        
        if (actual_delta == 0) continue;  // Move was blocked by bounds

        // Incremental update via CSC
        for (int idx = mip.csc_col_ptr[best_var];
             idx < mip.csc_col_ptr[best_var+1];
             idx++)
        {
            int row = mip.csc_row_idx[idx];
            activity[row] += mip.csc_val[idx] * actual_delta;
        }
        
        // ENHANCED: If variable hits bound, mark as fixed
        if (x[best_var] <= mip.lb[best_var] || x[best_var] >= mip.ub[best_var]) {
            variable_is_free[best_var] = false;
        }
    }

    return false;  // max iterations reached
}