#include "chebyshev_center.h"
#include "lp_relaxation.h"
#include <cmath>
#include <iostream>

ChebyshevResult compute_chebyshev_center(const MIPProblem& mip)
{
    ChebyshevResult result;

    int n = mip.num_cols;
    int m = mip.num_rows;

    // --------------------------------------------
    // 1️⃣ Build new MIPProblem for Chebyshev LP
    // --------------------------------------------
    MIPProblem cheb;

    cheb.num_rows = m;
    cheb.num_cols = n + 1; // add r variable

    // Objective: minimize -r
    cheb.c.assign(n + 1, 0.0);
    cheb.c[n] = -1.0;  // minimize -r

    cheb.obj_offset = 0.0;

    // Bounds
    cheb.lb = mip.lb;
    cheb.ub = mip.ub;

    cheb.lb.push_back(0.0);                 // r >= 0
    cheb.ub.push_back(1e20);                // r unbounded above

    // Variable types (all continuous)
    cheb.vartype.assign(n + 1, VarType::CONTINUOUS); 

    // RHS
    cheb.b = mip.b;

    // --------------------------------------------
    // 2️⃣ Build new matrix with extra r column
    // --------------------------------------------

    for (int i = 0; i < m; ++i)
    {
        double norm_ai = 0.0;

        int start = mip.csr_row_ptr[i];
        int end   = mip.csr_row_ptr[i + 1];

        for (int p = start; p < end; ++p) {
            double val = mip.csr_val[p];
            norm_ai += val * val;
        }

        norm_ai = std::sqrt(norm_ai);

        // copy original row
        for (int p = start; p < end; ++p) {
            cheb.coo_row.push_back(i);
            cheb.coo_col.push_back(mip.csr_col_idx[p]);
            cheb.coo_val.push_back(mip.csr_val[p]);
        }

        // add coefficient for r
        if (norm_ai > 0.0) {
            cheb.coo_row.push_back(i);
            cheb.coo_col.push_back(n);   // r index
            cheb.coo_val.push_back(norm_ai);
        }
    }

    // Build CSR/CSC
    cheb.finalize();

    // --------------------------------------------
    // 3️⃣ Solve LP
    // --------------------------------------------
    LPRelaxation lp(cheb);

    bool ok = lp.solve();
    if (!ok) {
        std::cerr << "Chebyshev LP failed.\n";
        return result;
    }

    result.x_center.assign(lp.x.begin(), lp.x.begin() + n);
    result.radius = lp.x[n];
    result.feasible = true;

    return result;
}
