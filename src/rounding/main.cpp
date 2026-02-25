// #include <iostream>
// #include <string>
// #include <chrono>
// #include <cmath>

// #include "mip_problem.h"
// #include "lp_relaxation.h"
// #include "chebyshev_center.h"

// double compute_min_slack(
//     const MIPProblem& mip,
//     const std::vector<double>& x)
// {
//     double min_slack = 1e100;

//     for (int i = 0; i < mip.num_rows; ++i)
//     {
//         double lhs = 0.0;

//         int start = mip.csr_row_ptr[i];
//         int end   = mip.csr_row_ptr[i+1];

//         for (int p = start; p < end; ++p)
//         {
//             int j = mip.csr_col_idx[p];
//             lhs += mip.csr_val[p] * x[j];
//         }

//         double slack = mip.b[i] - lhs;
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

//     MIPProblem mip;
//     mip.load_from_mps(filename);
//     mip.finalize();

//     std::cout << "Rows: " << mip.num_rows
//               << " Cols: " << mip.num_cols << "\n";

//     // // LP solve
//     // LPRelaxation lp(mip);

//     // auto t0 = std::chrono::high_resolution_clock::now();
//     // bool ok = lp.solve();
//     // auto t1 = std::chrono::high_resolution_clock::now();

//     // if (!ok) {
//     //     std::cout << "LP infeasible.\n";
//     //     return 1;
//     // }

//     // double lp_time =
//     //     std::chrono::duration<double>(t1 - t0).count();

//     // std::cout << "LP obj = " << lp.obj_value
//     //           << " time = " << lp_time << "\n";

//     // double lp_slack = compute_min_slack(mip, lp.x);

//     // std::cout << "LP min slack = " << lp_slack << "\n";

//     // bool okk = mip.check_feasible(lp.x);
//     // std::cout << "MIP feasible(Need not be)? " << okk << "\n";



//     // Chebyshev center
//     auto t2 = std::chrono::high_resolution_clock::now();
//     auto [cheb, cheb_mip] = compute_chebyshev_center(mip);
//     auto t3 = std::chrono::high_resolution_clock::now();

//     if (!cheb.feasible) {
//         std::cout << "Chebyshev failed.\n";
//         return 1;
//     }

//     double cheb_time =
//         std::chrono::duration<double>(t3 - t2).count();

//     // int v1 = cheb_mip.c.size();
//     // std::cout << "Objective coefficient of r = " << cheb_mip.c[v1-1] << "\n";
//     std::cout << "Chebyshev radius = "
//               << cheb.radius
//               << " time = " << cheb_time << "\n";

//     double cheb_slack =
//         compute_min_slack(mip, cheb.x_center);

//     std::cout << "Chebyshev min slack = "
//               << cheb_slack << "\n";

//     int active_count = 0;

//     for (int i = 0; i < mip.num_rows; ++i)
//     {
//         double lhs = 0.0;

//         for (int p = mip.csr_row_ptr[i];
//             p < mip.csr_row_ptr[i+1]; ++p)
//             lhs += mip.csr_val[p] * cheb.x_center[mip.csr_col_idx[p]];

//         double slack = mip.b[i] - lhs;

//         if (fabs(slack) < 1e-6)
//             active_count++;
//     }
//     std::cout << "Active constraints at Chebyshev center: "
//             << active_count << "\n";


//     // --------------------------------------------------
//     // DEBUG: Check Chebyshev feasibility in original LP
//     // --------------------------------------------------

//     std::cout << "\n--- Debug Chebyshev Feasibility ---\n";

//     if (!cheb.feasible) {
//         std::cout << "Chebyshev solve failed.\n";
//     }
//     else
//     {
//         const std::vector<double>& x = cheb.x_center;

//         // 1️⃣ Print first few coordinates
//         std::cout << "First 10 Non-Zero Chebyshev variables:\n";
//         int cnt = 0;
//         for (int j = 0; j < mip.num_cols; ++j){
//             if(cnt == 10) break;
//             if(x[j] == 0) continue;

//             cnt++;
//             std::cout << "x[" << j << "] = " << x[j] << "\n";
//         }

//         // 2️⃣ Check Ax <= b
//         double max_violation = 0.0;
//         int worst_row = -1;

//         for (int i = 0; i < mip.num_rows; ++i)
//         {
//             double lhs = 0.0;

//             for (int p = mip.csr_row_ptr[i];
//                 p < mip.csr_row_ptr[i+1]; ++p)
//             {
//                 int j = mip.csr_col_idx[p];
//                 lhs += mip.csr_val[p] * x[j];
//             }

//             double violation = lhs - mip.b[i];

//             if (violation > max_violation) {
//                 max_violation = violation;
//                 worst_row = i;
//             }
//         }

//         std::cout << "Max constraint violation (Ax <= b): "
//                 << max_violation << "\n";

//         if (worst_row >= 0)
//             std::cout << "Worst violated row = "
//                     << worst_row << "\n";

//         // 3️⃣ Check bounds
//         double max_lb_violation = 0.0;
//         double max_ub_violation = 0.0;

//         for (int j = 0; j < mip.num_cols; ++j)
//         {
//             if (x[j] < mip.lb[j])
//                 max_lb_violation =
//                     std::max(max_lb_violation,
//                             mip.lb[j] - x[j]);

//             if (x[j] > mip.ub[j])
//                 max_ub_violation =
//                     std::max(max_ub_violation,
//                             x[j] - mip.ub[j]);
//         }

//         std::cout << "Max lower bound violation: "
//                 << max_lb_violation << "\n";

//         std::cout << "Max upper bound violation: "
//                 << max_ub_violation << "\n";

//         // 4️⃣ Use built-in feasibility checker
//         bool feas = mip.check_feasible(x);

//         std::cout << "check_feasible() says: "
//                 << feas << "\n";
//     }



//     return 0;
// }



#include <iostream>
#include <string>
#include <chrono>
#include <cmath>
#include <iomanip>

#include "mip_problem.h"
#include "lp_relaxation.h"
#include "chebyshev_center.h"
#include "neighbourhood_rounding.h"

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

void print_solution_summary(const std::string& name,
                            const MIPProblem& mip,
                            const std::vector<double>& x,
                            bool check_feas = true)
{
    std::cout << "\n--- " << name << " ---\n";
    
    // Objective value
    double obj = mip.obj_offset;
    for (int i = 0; i < mip.num_cols; ++i)
        obj += mip.c[i] * x[i];
    std::cout << "Objective: " << std::fixed << std::setprecision(6) << obj << "\n";
    
    // Min slack
    double min_slack = compute_min_slack(mip, x);
    std::cout << "Min slack: " << min_slack << "\n";
    
    // Feasibility
    if (check_feas) {
        bool feas = mip.check_feasible(x);
        std::cout << "Feasible: " << (feas ? "YES" : "NO") << "\n";
    }
    
    // First few non-zero values
    std::cout << "Non-zero variables (first 10):\n";
    int cnt = 0;
    for (int j = 0; j < mip.num_cols && cnt < 10; ++j) {
        if (std::abs(x[j]) > 1e-9) {
            std::cout << "  x[" << j << "] = " << x[j] << "\n";
            cnt++;
        }
    }
    if (cnt == 0) std::cout << "  (all zero)\n";
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cout << "Usage: ./solver instance.mps [options]\n";
        std::cout << "Options:\n";
        std::cout << "  --lp-only       : Only solve LP relaxation\n";
        std::cout << "  --chebyshev     : Compute Chebyshev center\n";
        std::cout << "  --rounding      : Run GPU rounding heuristic (default)\n";
        std::cout << "  --all           : Run all methods\n";
        std::cout << "  -K <num>        : Number of rounding candidates (default: 1024)\n";
        return 1;
    }

    std::string filename = argv[1];
    std::string mode = "--rounding";  // default
    int K = 1024;  // default number of candidates

    // Parse command line arguments
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--lp-only" || arg == "--chebyshev" || 
            arg == "--rounding" || arg == "--all") {
            mode = arg;
        }
        else if (arg == "-K" && i + 1 < argc) {
            K = std::stoi(argv[++i]);
        }
    }

    // Load problem
    std::cout << "Loading: " << filename << "\n";
    MIPProblem mip;
    mip.load_from_mps(filename);
    mip.finalize();

    std::cout << "Problem stats:\n";
    std::cout << "  Rows: " << mip.num_rows << "\n";
    std::cout << "  Cols: " << mip.num_cols << "\n";
    std::cout << "  Nonzeros: " << mip.csr_val.size() << "\n";

    // ============================================================
    // 1. LP RELAXATION
    // ============================================================
    if (mode == "--lp-only" || mode == "--all") {
        std::cout << "\n========================================\n";
        std::cout << "SOLVING LP RELAXATION\n";
        std::cout << "========================================\n";

        LPRelaxation lp(mip);

        auto t0 = std::chrono::high_resolution_clock::now();
        bool ok = lp.solve();
        auto t1 = std::chrono::high_resolution_clock::now();

        double lp_time = std::chrono::duration<double>(t1 - t0).count();

        if (!ok) {
            std::cout << "LP infeasible.\n";
            return 1;
        }

        std::cout << "LP solve time: " << lp_time << " s\n";
        print_solution_summary("LP RELAXATION", mip, lp.x, false);
        
        double lp_slack = compute_min_slack(mip, lp.x);
        std::cout << "LP solution is " 
                  << (mip.check_feasible(lp.x) ? "feasible" : "infeasible")
                  << " for original MIP\n";
    }

    // ============================================================
    // 2. CHEBYSHEV CENTER
    // ============================================================
    if (mode == "--chebyshev" || mode == "--all") {
        std::cout << "\n========================================\n";
        std::cout << "COMPUTING CHEBYSHEV CENTER\n";
        std::cout << "========================================\n";

        auto t2 = std::chrono::high_resolution_clock::now();
        auto [cheb, cheb_mip] = compute_chebyshev_center(mip);
        auto t3 = std::chrono::high_resolution_clock::now();

        double cheb_time = std::chrono::duration<double>(t3 - t2).count();

        if (!cheb.feasible) {
            std::cout << "Chebyshev center computation failed.\n";
        } else {
            std::cout << "Chebyshev time: " << cheb_time << " s\n";
            std::cout << "Chebyshev radius: " << cheb.radius << "\n";
            print_solution_summary("CHEBYSHEV CENTER", mip, cheb.x_center, true);
            
            // Active constraints
            int active_count = 0;
            for (int i = 0; i < mip.num_rows; ++i) {
                double lhs = 0.0;
                for (int p = mip.csr_row_ptr[i]; p < mip.csr_row_ptr[i+1]; ++p)
                    lhs += mip.csr_val[p] * cheb.x_center[mip.csr_col_idx[p]];
                if (std::fabs(mip.b[i] - lhs) < 1e-6)
                    active_count++;
            }
            std::cout << "Active constraints: " << active_count << "\n";
        }
    }

    // ============================================================
    // 3. GPU ROUNDING HEURISTIC (DEFAULT)
    // ============================================================
    if (mode == "--rounding" || mode == "--all") {
        std::cout << "\n========================================\n";
        std::cout << "GPU ROUNDING HEURISTIC (K=" << K << ")\n";
        std::cout << "========================================\n";

        // First need LP solution
        std::cout << "Step 1: Solving LP relaxation...\n";
        LPRelaxation lp(mip);
        auto t_lp0 = std::chrono::high_resolution_clock::now();
        bool lp_ok = lp.solve();
        auto t_lp1 = std::chrono::high_resolution_clock::now();
        
        if (!lp_ok) {
            std::cout << "LP infeasible, cannot apply rounding.\n";
            return 1;
        }
        
        double lp_time = std::chrono::duration<double>(t_lp1 - t_lp0).count();
        std::cout << "LP solve time: " << lp_time << " s\n";

        // Run GPU rounding
        std::cout << "\nStep 2: Running GPU rounding with K=" << K << " candidates...\n";
        auto t_round0 = std::chrono::high_resolution_clock::now();
        Solution round_sol = run_gpu_neighbourhood_rounding(mip, lp.x, K);
        auto t_round1 = std::chrono::high_resolution_clock::now();
        
        double round_time = std::chrono::duration<double>(t_round1 - t_round0).count();
        std::cout << "Rounding time: " << round_time << " s\n";

        if (round_sol.feasible) {
            std::cout << "\n*** FEASIBLE SOLUTION FOUND ***\n";
            print_solution_summary("ROUNDING HEURISTIC", mip, round_sol.x, true);
        } else {
            std::cout << "\n*** NO FEASIBLE SOLUTION FOUND ***\n";
            std::cout << "Best candidate was infeasible after repair.\n";
        }

        // Summary comparison
        std::cout << "\n========================================\n";
        std::cout << "SUMMARY\n";
        std::cout << "========================================\n";
        std::cout << "LP solve time:        " << std::setw(10) << lp_time << " s\n";
        std::cout << "Rounding time:        " << std::setw(10) << round_time << " s\n";
        std::cout << "Total time:           " << std::setw(10) << (lp_time + round_time) << " s\n";
        std::cout << "Solution found:       " << (round_sol.feasible ? "YES" : "NO") << "\n";
        if (round_sol.feasible) {
            std::cout << "Objective value:      " << round_sol.obj_value << "\n";
        }
    }

    // ============================================================
    // DEBUG: Detailed feasibility check if requested
    // ============================================================
    if (argc > 2 && std::string(argv[2]) == "--debug") {
        std::cout << "\n========================================\n";
        std::cout << "DETAILED DEBUG OUTPUT\n";
        std::cout << "========================================\n";
        
        // Re-run everything with detailed checks
        LPRelaxation lp(mip);
        lp.solve();
        
        auto [cheb, cheb_mip] = compute_chebyshev_center(mip);
        Solution round_sol = run_gpu_neighbourhood_rounding(mip, lp.x, K);

        std::cout << "\n--- LP Solution Debug ---\n";
        double max_viol = 0;
        for (int i = 0; i < mip.num_rows; ++i) {
            double lhs = 0;
            for (int p = mip.csr_row_ptr[i]; p < mip.csr_row_ptr[i+1]; ++p)
                lhs += mip.csr_val[p] * lp.x[mip.csr_col_idx[p]];
            max_viol = std::max(max_viol, lhs - mip.b[i]);
        }
        std::cout << "LP max violation: " << max_viol << "\n";

        if (cheb.feasible) {
            std::cout << "\n--- Chebyshev Center Debug ---\n";
            max_viol = 0;
            for (int i = 0; i < mip.num_rows; ++i) {
                double lhs = 0;
                for (int p = mip.csr_row_ptr[i]; p < mip.csr_row_ptr[i+1]; ++p)
                    lhs += mip.csr_val[p] * cheb.x_center[mip.csr_col_idx[p]];
                double viol = lhs - mip.b[i];
                if (viol > max_viol) {
                    max_viol = viol;
                    std::cout << "Row " << i << ": lhs=" << lhs << ", b=" << mip.b[i] 
                              << ", viol=" << viol << "\n";
                }
            }
        }

        if (round_sol.feasible) {
            std::cout << "\n--- Rounding Solution Debug ---\n";
            max_viol = 0;
            for (int i = 0; i < mip.num_rows; ++i) {
                double lhs = 0;
                for (int p = mip.csr_row_ptr[i]; p < mip.csr_row_ptr[i+1]; ++p)
                    lhs += mip.csr_val[p] * round_sol.x[mip.csr_col_idx[p]];
                double viol = lhs - mip.b[i];
                max_viol = std::max(max_viol, viol);
                if (viol > 0) {
                    std::cout << "Violated row " << i << ": lhs=" << lhs 
                              << ", b=" << mip.b[i] << "\n";
                }
            }
            std::cout << "Rounding max violation: " << max_viol << "\n";
        }
    }

    return 0;
}