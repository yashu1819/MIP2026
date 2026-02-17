// // #include <iostream>
// // #include <string>
// // #include <chrono>

// // #include "mip_problem.h"
// // #include "lp_relaxation.h"
// // #include "solution.h"
// // #include "neighbourhood_rounding.h"


// // // // forward declaration from neighbourhood_rounding.cpp
// // // Solution run_gpu_neighborhood_rounding(
// // //     const MIPProblem& mip,
// // //     const std::vector<double>& x_lp,
// // //     int K = 1024
// // // );

// // int main(int argc, char* argv[])
// // {
// //     if (argc < 2) {
// //         std::cout << "Usage: ./solver instance.mps\n";
// //         return 1;
// //     }

// //     std::string filename = argv[1];

// //     std::cout << "=====================================\n";
// //     std::cout << "Loading MIP: " << filename << "\n";
// //     std::cout << "=====================================\n";

// //     MIPProblem mip;
// //     mip.load_from_mps(filename);
// //     mip.finalize();

// //     std::cout << "Rows: " << mip.num_rows
// //               << "  Cols: " << mip.num_cols << "\n";

// //     // -----------------------------
// //     // 1. Solve LP relaxation (cuOpt)
// //     // -----------------------------
// //     std::cout << "\nSolving LP relaxation using cuOpt...\n";

// //     LPRelaxation lp(mip);

// //     auto t0 = std::chrono::high_resolution_clock::now();
// //     bool lp_ok = lp.solve();
// //     auto t1 = std::chrono::high_resolution_clock::now();

// //     if (!lp_ok) {
// //         std::cout << "LP relaxation infeasible.\n";
// //         return 1;
// //     }

// //     double lp_time =
// //         std::chrono::duration<double>(t1 - t0).count();

// //     std::cout << "LP solved.\n";
// //     std::cout << "LP objective = " << lp.obj_value << "\n";
// //     std::cout << "LP time      = " << lp_time << " sec\n";

// //     // -----------------------------
// //     // 2. GPU Neighborhood Rounding
// //     // -----------------------------
// //     std::cout << "\nRunning GPU neighborhood rounding...\n";

// //     auto t2 = std::chrono::high_resolution_clock::now();

// //     Solution sol =
// //         run_gpu_neighbourhood_rounding(mip, lp.x, 1024);

// //     auto t3 = std::chrono::high_resolution_clock::now();

// //     double heur_time =
// //         std::chrono::duration<double>(t3 - t2).count();

// //     std::cout << "Heuristic time = " << heur_time << " sec\n";

// //     // -----------------------------
// //     // 3. Print result
// //     // -----------------------------
// //     if (sol.feasible) {
// //         std::cout << "\nFeasible integer solution found!\n";
// //         std::cout << "Objective = " << sol.obj_value << "\n";
// //     } else {
// //         std::cout << "\nNo feasible solution found.\n";
// //     }

// //     std::cout << "=====================================\n";
// //     return 0;
// // }



// #include <iostream>
// #include <string>
// #include <chrono>
// #include <cmath>

// #include "mip_problem.h"
// #include "lp_relaxation.h"
// #include "solution.h"
// #include "chebyshev_center.h"

// double compute_min_slack(
//     const MIPProblem& mip,
//     const std::vector<double>& x)
// {
//     double min_slack = 1e100;

//     for (int i = 0; i < mip.num_rows; ++i) {
//         double lhs = 0.0;

//         for (int j = 0; j < mip.num_cols; ++j) {
//             lhs += mip.A(i, j) * x[j];
//         }

//         double slack = mip.b(i) - lhs;
//         min_slack = std::min(min_slack, slack);
//     }

//     return min_slack;
// }

// int main(int argc, char* argv[])
// {
//     if (argc < 2) {
//         std::cout << "Usage: ./solver instance.mps\n";
//         return 1;
//     }

//     std::string filename = argv[1];

//     std::cout << "=====================================\n";
//     std::cout << "Loading MIP: " << filename << "\n";
//     std::cout << "=====================================\n";

//     MIPProblem mip;
//     mip.load_from_mps(filename);
//     mip.finalize();

//     std::cout << "Rows: " << mip.num_rows
//               << "  Cols: " << mip.num_cols << "\n";

//     // ---------------------------------------------------
//     // 1️⃣ Solve LP Relaxation
//     // ---------------------------------------------------
//     std::cout << "\nSolving LP relaxation...\n";

//     LPRelaxation lp(mip);

//     auto t0 = std::chrono::high_resolution_clock::now();
//     bool lp_ok = lp.solve();
//     auto t1 = std::chrono::high_resolution_clock::now();

//     if (!lp_ok) {
//         std::cout << "LP relaxation infeasible.\n";
//         return 1;
//     }

//     double lp_time =
//         std::chrono::duration<double>(t1 - t0).count();

//     std::cout << "LP solved.\n";
//     std::cout << "LP objective = " << lp.obj_value << "\n";
//     std::cout << "LP time      = " << lp_time << " sec\n";

//     double lp_min_slack =
//         compute_min_slack(mip, lp.x);

//     std::cout << "LP min slack (distance to boundary) = "
//               << lp_min_slack << "\n";

//     // ---------------------------------------------------
//     // 2️⃣ Compute Chebyshev Center
//     // ---------------------------------------------------
//     std::cout << "\nComputing Chebyshev center...\n";

//     auto t2 = std::chrono::high_resolution_clock::now();
//     ChebyshevResult cheb =
//         compute_chebyshev_center(mip);
//     auto t3 = std::chrono::high_resolution_clock::now();

//     double cheb_time =
//         std::chrono::duration<double>(t3 - t2).count();

//     if (!cheb.feasible) {
//         std::cout << "Chebyshev LP failed.\n";
//         return 1;
//     }

//     std::cout << "Chebyshev radius = "
//               << cheb.radius << "\n";

//     std::cout << "Chebyshev time   = "
//               << cheb_time << " sec\n";

//     double cheb_min_slack =
//         compute_min_slack(mip, cheb.x_center);

//     std::cout << "Chebyshev min slack = "
//               << cheb_min_slack << "\n";

//     // ---------------------------------------------------
//     // 3️⃣ Compare Interior Quality
//     // ---------------------------------------------------
//     std::cout << "\nInterior comparison:\n";

//     if (cheb_min_slack > lp_min_slack)
//         std::cout << "Chebyshev is deeper inside polyhedron.\n";
//     else
//         std::cout << "LP solution closer to boundary.\n";

//     std::cout << "=====================================\n";

//     return 0;
// }



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

    // LP solve
    LPRelaxation lp(mip);

    auto t0 = std::chrono::high_resolution_clock::now();
    bool ok = lp.solve();
    auto t1 = std::chrono::high_resolution_clock::now();

    if (!ok) {
        std::cout << "LP infeasible.\n";
        return 1;
    }

    double lp_time =
        std::chrono::duration<double>(t1 - t0).count();

    std::cout << "LP obj = " << lp.obj_value
              << " time = " << lp_time << "\n";

    double lp_slack =
        compute_min_slack(mip, lp.x);

    std::cout << "LP min slack = " << lp_slack << "\n";

    bool okk = mip.check_feasible(lp.x);
    std::cout << "LP feasible? " << okk << "\n";


    std::cout << "\n--- Checking LP vs MIP structural consistency ---\n";

bool same = true;

// 1️⃣ Objective
if (lp.c.size() != mip.c.size()) {
    std::cout << "Objective size mismatch\n";
    same = false;
} else {
    for (size_t i = 0; i < lp.c.size(); ++i) {
        if (fabs(lp.c[i] - mip.c[i]) > 1e-12) {
            std::cout << "Objective mismatch at " << i << "\n";
            same = false;
            break;
        }
    }
}

// 2️⃣ RHS
if (lp.b.size() != mip.b.size()) {
    std::cout << "RHS size mismatch\n";
    same = false;
} else {
    for (size_t i = 0; i < lp.b.size(); ++i) {
        if (fabs(lp.b[i] - mip.b[i]) > 1e-12) {
            std::cout << "RHS mismatch at row " << i << "\n";
            same = false;
            break;
        }
    }
}

// 3️⃣ Bounds
for (int j = 0; j < mip.num_cols; ++j) {
    if (fabs(lp.lb[j] - mip.lb[j]) > 1e-12) {
        std::cout << "LB mismatch at col " << j << "\n";
        same = false;
        break;
    }
    if (fabs(lp.ub[j] - mip.ub[j]) > 1e-12) {
        std::cout << "UB mismatch at col " << j << "\n";
        same = false;
        break;
    }
}

// 4️⃣ CSR row ptr
if (lp.csr_row_ptr != mip.csr_row_ptr) {
    std::cout << "CSR row_ptr mismatch\n";
    same = false;
}

// 5️⃣ CSR col idx
if (lp.csr_col_idx != mip.csr_col_idx) {
    std::cout << "CSR col_idx mismatch\n";
    same = false;
}

// 6️⃣ CSR values
if (lp.csr_val.size() != mip.csr_val.size()) {
    std::cout << "CSR value size mismatch\n";
    same = false;
} else {
    for (size_t i = 0; i < lp.csr_val.size(); ++i) {
        if (fabs(lp.csr_val[i] - mip.csr_val[i]) > 1e-12) {
            std::cout << "CSR value mismatch at index " << i << "\n";
            same = false;
            break;
        }
    }
}

if (same)
    std::cout << "LP and MIP matrices are IDENTICAL\n";
else
    std::cout << "LP and MIP matrices DIFFER\n";



    // Chebyshev center
    auto t2 = std::chrono::high_resolution_clock::now();
    ChebyshevResult cheb = compute_chebyshev_center(mip);
    auto t3 = std::chrono::high_resolution_clock::now();

    if (!cheb.feasible) {
        std::cout << "Chebyshev failed.\n";
        return 1;
    }

    double cheb_time =
        std::chrono::duration<double>(t3 - t2).count();

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
        std::cout << "First 10 Chebyshev variables:\n";
        for (int j = 0; j < std::min(10, mip.num_cols); ++j)
            std::cout << "x[" << j << "] = " << x[j] << "\n";

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
        bool feas =
            mip.check_feasible(x);

        std::cout << "check_feasible() says: "
                << feas << "\n";
    }



    return 0;
}
