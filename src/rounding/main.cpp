#include <iostream>
#include <string>
#include <chrono>
#include <cmath>

#include "mip_problem.h"
#include "lp_relaxation.h"
#include "chebyshev_center.h"

double compute_min_slack(
    const MIPProblem& mip,
    const std::vector<double>& x)
{
    double min_slack = 1e100;

    for (int i = 0; i < mip.num_rows; ++i)
    {
        double lhs = 0.0;

        int start = mip.csr_row_ptr[i];
        int end   = mip.csr_row_ptr[i+1];

        for (int p = start; p < end; ++p)
        {
            int j = mip.csr_col_idx[p];
            lhs += mip.csr_val[p] * x[j];
        }

        double slack = mip.b[i] - lhs;
        min_slack = std::min(min_slack, slack);
    }

    return min_slack;
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cout << "Usage: ./solver instance.mps\n";
        return 1;
    }

    std::string filename = argv[1];

    MIPProblem mip;
    mip.load_from_mps(filename);
    mip.finalize();

    std::cout << "Rows: " << mip.num_rows
              << " Cols: " << mip.num_cols << "\n";

    // // LP solve
    // LPRelaxation lp(mip);

    // auto t0 = std::chrono::high_resolution_clock::now();
    // bool ok = lp.solve();
    // auto t1 = std::chrono::high_resolution_clock::now();

    // if (!ok) {
    //     std::cout << "LP infeasible.\n";
    //     return 1;
    // }

    // double lp_time =
    //     std::chrono::duration<double>(t1 - t0).count();

    // std::cout << "LP obj = " << lp.obj_value
    //           << " time = " << lp_time << "\n";

    // double lp_slack = compute_min_slack(mip, lp.x);

    // std::cout << "LP min slack = " << lp_slack << "\n";

    // bool okk = mip.check_feasible(lp.x);
    // std::cout << "MIP feasible(Need not be)? " << okk << "\n";



    // Chebyshev center
    auto t2 = std::chrono::high_resolution_clock::now();
    auto [cheb, cheb_mip] = compute_chebyshev_center(mip);
    auto t3 = std::chrono::high_resolution_clock::now();

    if (!cheb.feasible) {
        std::cout << "Chebyshev failed.\n";
        return 1;
    }

    double cheb_time =
        std::chrono::duration<double>(t3 - t2).count();

    // int v1 = cheb_mip.c.size();
    // std::cout << "Objective coefficient of r = " << cheb_mip.c[v1-1] << "\n";
    std::cout << "Chebyshev radius = "
              << cheb.radius
              << " time = " << cheb_time << "\n";

    double cheb_slack =
        compute_min_slack(mip, cheb.x_center);

    std::cout << "Chebyshev min slack = "
              << cheb_slack << "\n";

    int active_count = 0;

    for (int i = 0; i < mip.num_rows; ++i)
    {
        double lhs = 0.0;

        for (int p = mip.csr_row_ptr[i];
            p < mip.csr_row_ptr[i+1]; ++p)
            lhs += mip.csr_val[p] * cheb.x_center[mip.csr_col_idx[p]];

        double slack = mip.b[i] - lhs;

        if (fabs(slack) < 1e-6)
            active_count++;
    }
    std::cout << "Active constraints at Chebyshev center: "
            << active_count << "\n";


    // --------------------------------------------------
    // DEBUG: Check Chebyshev feasibility in original LP
    // --------------------------------------------------

    std::cout << "\n--- Debug Chebyshev Feasibility ---\n";

    if (!cheb.feasible) {
        std::cout << "Chebyshev solve failed.\n";
    }
    else
    {
        const std::vector<double>& x = cheb.x_center;

        // 1️⃣ Print first few coordinates
        std::cout << "First 10 Non-Zero Chebyshev variables:\n";
        int cnt = 0;
        for (int j = 0; j < mip.num_cols; ++j){
            if(cnt == 10) break;
            if(x[j] == 0) continue;

            cnt++;
            std::cout << "x[" << j << "] = " << x[j] << "\n";
        }

        // 2️⃣ Check Ax <= b
        double max_violation = 0.0;
        int worst_row = -1;

        for (int i = 0; i < mip.num_rows; ++i)
        {
            double lhs = 0.0;

            for (int p = mip.csr_row_ptr[i];
                p < mip.csr_row_ptr[i+1]; ++p)
            {
                int j = mip.csr_col_idx[p];
                lhs += mip.csr_val[p] * x[j];
            }

            double violation = lhs - mip.b[i];

            if (violation > max_violation) {
                max_violation = violation;
                worst_row = i;
            }
        }

        std::cout << "Max constraint violation (Ax <= b): "
                << max_violation << "\n";

        if (worst_row >= 0)
            std::cout << "Worst violated row = "
                    << worst_row << "\n";

        // 3️⃣ Check bounds
        double max_lb_violation = 0.0;
        double max_ub_violation = 0.0;

        for (int j = 0; j < mip.num_cols; ++j)
        {
            if (x[j] < mip.lb[j])
                max_lb_violation =
                    std::max(max_lb_violation,
                            mip.lb[j] - x[j]);

            if (x[j] > mip.ub[j])
                max_ub_violation =
                    std::max(max_ub_violation,
                            x[j] - mip.ub[j]);
        }

        std::cout << "Max lower bound violation: "
                << max_lb_violation << "\n";

        std::cout << "Max upper bound violation: "
                << max_ub_violation << "\n";

        // 4️⃣ Use built-in feasibility checker
        bool feas = mip.check_feasible(x);

        std::cout << "check_feasible() says: "
                << feas << "\n";
    }



    return 0;
}
