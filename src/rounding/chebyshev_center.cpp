#include "chebyshev_center.h"
#include "lp_relaxation.h"
#include <cmath>
#include <iostream>


std::pair<ChebyshevResult, MIPProblem> compute_chebyshev_center(const MIPProblem& mip)
{
    ChebyshevResult result;

    int n = mip.num_cols;
    int m = mip.num_rows;

    MIPProblem cheb;

    // Only original rows
    cheb.num_rows = m;
    cheb.num_cols = n + 1; // extra r

    // Objective: maximize r  (minimize -r)
    cheb.c.assign(n + 1, 0.0);
    cheb.c[n] = -1.0;

    cheb.obj_offset = 0.0;

    // Keep original bounds
    cheb.lb = mip.lb;
    cheb.ub = mip.ub;

    // Add r bounds
    cheb.lb.push_back(0.0);      // r >= 0
    cheb.ub.push_back(1e20);

    cheb.vartype.assign(n + 1, VarType::CONTINUOUS);

    // Copy RHS
    cheb.b = mip.b;

    // Clear COO
    cheb.coo_row.clear();
    cheb.coo_col.clear();
    cheb.coo_val.clear();

    // Build matrix
    for (int i = 0; i < m; ++i)
    {
        double norm_ai = 0.0;

        int start = mip.csr_row_ptr[i];
        int end   = mip.csr_row_ptr[i + 1];

        for (int p = start; p < end; ++p)
        {
            double val = mip.csr_val[p];
            norm_ai += val * val;

            cheb.coo_row.push_back(i);
            cheb.coo_col.push_back(mip.csr_col_idx[p]);
            cheb.coo_val.push_back(val);
        }

        norm_ai = std::sqrt(norm_ai);

        if (norm_ai > 0.0)
        {
            cheb.coo_row.push_back(i);
            cheb.coo_col.push_back(n); // r
            cheb.coo_val.push_back(norm_ai);
        }
    }

    cheb.finalize();

    LPRelaxation lp(cheb);

    if (!lp.solve())
    {
        std::cerr << "Chebyshev LP failed.\n";
        return std::make_pair(result, cheb);
    }

    result.x_center.assign(lp.x.begin(), lp.x.begin() + n);
    result.radius = lp.x[n];
    result.feasible = true;

    return std::make_pair(result, cheb);
}
