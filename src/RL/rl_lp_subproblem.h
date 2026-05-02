#ifndef RL_LP_SUBPROBLEM_H
#define RL_LP_SUBPROBLEM_H

#include "../mip_problem.h"
#include <vector>
#include <cmath>
#include <iostream>

// CLP headers for LP sub-problem
#include <coin/ClpSimplex.hpp>
#include <coin/CoinPackedMatrix.hpp>

namespace rl_sph {

// Solve LP sub-problem for continuous variables after fixing integer variables.
// Given a MILP and a solution x where integer/binary variables are fixed,
// this solves:  min c^T x  s.t. Ax <= b, lb <= x <= ub
// with integer/binary variables fixed to their current values.
//
// Returns true if LP is feasible; updates x in-place for continuous variables.
inline bool solve_lp_subproblem(
    const MIPProblem& mip,
    std::vector<double>& x
) {
    int n = mip.num_cols;
    int m = mip.num_rows;

    // Count continuous variables
    int num_continuous = 0;
    for (int i = 0; i < n; ++i) {
        if (mip.vartype[i] == VarType::CONTINUOUS) num_continuous++;
    }

    // If no continuous variables, nothing to solve
    if (num_continuous == 0) return true;

    // Build LP: we use CLP directly
    ClpSimplex lp;
    lp.setLogLevel(0);  // Suppress output

    // Set up the full problem but fix integer variables
    lp.resize(0, n);  // 0 rows, n columns initially

    // Set column bounds and objective
    for (int i = 0; i < n; ++i) {
        if (mip.vartype[i] == VarType::CONTINUOUS) {
            // Continuous: free to optimize within bounds
            lp.setColumnLower(i, mip.lb[i]);
            lp.setColumnUpper(i, mip.ub[i]);
        } else {
            // Integer/Binary: fix to current value
            lp.setColumnLower(i, x[i]);
            lp.setColumnUpper(i, x[i]);
        }
        lp.setObjectiveCoefficient(i, mip.c[i]);
    }

    // Add constraints from CSR representation (Ax <= b)
    for (int i = 0; i < m; ++i) {
        int nnz = mip.csr_row_ptr[i + 1] - mip.csr_row_ptr[i];
        if (nnz == 0) continue;

        std::vector<int> indices(nnz);
        std::vector<double> values(nnz);
        for (int j = 0; j < nnz; ++j) {
            indices[j] = mip.csr_col_idx[mip.csr_row_ptr[i] + j];
            values[j] = mip.csr_val[mip.csr_row_ptr[i] + j];
        }

        // Ax <= b  =>  -inf <= Ax <= b[i]
        lp.addRow(nnz, indices.data(), values.data(),
                  -COIN_DBL_MAX, mip.b[i]);
    }

    // Solve LP
    lp.primal();

    // Check if optimal
    if (lp.isProvenOptimal()) {
        const double* sol = lp.primalColumnSolution();
        // Update only continuous variables
        for (int i = 0; i < n; ++i) {
            if (mip.vartype[i] == VarType::CONTINUOUS) {
                x[i] = sol[i];
            }
        }
        return true;
    }

    // LP infeasible — continuous variables stay at current values
    return false;
}

} // namespace rl_sph

#endif // RL_LP_SUBPROBLEM_H
