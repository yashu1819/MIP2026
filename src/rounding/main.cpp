#include <iostream>
#include <string>
#include <chrono>
#include <cmath>
#include <iomanip>

#include "mip_problem.h"
#include "../relaxation/lp_relaxation.h"
#include "chebyshev_center.h"
#include "chebyshev_center_relaxed.h"
#include "analytic_center.h"
#include "interior_point.h"
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
        std::cout << "  --lp-only        : Only solve LP relaxation\n";
        std::cout << "  --chebyshev      : Compute standard Chebyshev center\n";
        std::cout << "  --analytic       : Compute analytic center (works for lower-dim polyhedra)\n";
        std::cout << "  --relaxed-cheb   : Compute relaxed Chebyshev center\n";
        std::cout << "  --interior       : Compute robust interior point (auto-detects lower-dim)\n";
        std::cout << "  --rounding       : Run GPU rounding heuristic (default)\n";
        std::cout << "  --rounding-ip    : Run GPU rounding using interior point as starting point\n";
        std::cout << "  --all            : Run all methods\n";
        std::cout << "  -K <num>         : Number of rounding candidates (default: 1024)\n";
        return 1;
    }

    std::string filename = argv[1];
    std::string mode = "--rounding";  // default
    int K = 1024;  // default number of candidates
    bool use_interior_point = false;  // use interior point for rounding

    // Parse command line arguments
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--lp-only" || arg == "--chebyshev" || arg == "--analytic" ||
            arg == "--relaxed-cheb" || arg == "--interior" ||
            arg == "--rounding" || arg == "--all") {
            mode = arg;
        }
        else if (arg == "--rounding-ip") {
            mode = "--rounding";
            use_interior_point = true;
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
    // 2. CHEBYSHEV CENTER (Standard)
    // ============================================================
    if (mode == "--chebyshev" || mode == "--all") {
        std::cout << "\n========================================\n";
        std::cout << "COMPUTING CHEBYSHEV CENTER\n";
        std::cout << "========================================\n";
        std::cout << "Note: Standard Chebyshev may give radius=0 for\n";
        std::cout << "      lower-dimensional polyhedra.\n";
        std::cout << "      Try --analytic or --relaxed-cheb instead.\n";

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
            if (cheb.radius < 1e-8) {
                std::cout << "*** WARNING: Radius is zero - polyhedron may be lower-dimensional ***\n";
                std::cout << "*** Try --analytic for analytic center ***\n";
            }
        }
    }

    // ============================================================
    // 3. ANALYTIC CENTER (For lower-dimensional polyhedra)
    // ============================================================
    if (mode == "--analytic" || mode == "--all") {
        std::cout << "\n========================================\n";
        std::cout << "COMPUTING ANALYTIC CENTER\n";
        std::cout << "========================================\n";
        std::cout << "Note: Analytic center works even when\n";
        std::cout << "      polyhedron is lower-dimensional.\n";

        auto t2 = std::chrono::high_resolution_clock::now();
        AnalyticCenterResult ac = compute_analytic_center(mip);
        auto t3 = std::chrono::high_resolution_clock::now();

        double ac_time = std::chrono::duration<double>(t3 - t2).count();

        if (!ac.feasible) {
            std::cout << "Analytic center computation failed.\n";
        } else {
            std::cout << "Analytic center time: " << ac_time << " s\n";
            std::cout << "Newton iterations: " << ac.iterations << "\n";
            std::cout << "Min slack: " << ac.min_slack << "\n";
            print_solution_summary("ANALYTIC CENTER", mip, ac.x_center, true);
        }
    }

    // ============================================================
    // 4. RELAXED CHEBYSHEV CENTER
    // ============================================================
    if (mode == "--relaxed-cheb" || mode == "--all") {
        std::cout << "\n========================================\n";
        std::cout << "COMPUTING RELAXED CHEBYSHEV CENTER\n";
        std::cout << "========================================\n";

        auto t2 = std::chrono::high_resolution_clock::now();
        RelaxedChebyshevResult rcheb = compute_relaxed_chebyshev_center(mip);
        auto t3 = std::chrono::high_resolution_clock::now();

        double rcheb_time = std::chrono::duration<double>(t3 - t2).count();

        if (!rcheb.feasible) {
            std::cout << "Relaxed Chebyshev computation failed.\n";
        } else {
            std::cout << "Relaxed Chebyshev time: " << rcheb_time << " s\n";
            std::cout << "Radius: " << rcheb.radius << "\n";
            std::cout << "Relaxed radius: " << rcheb.relaxed_radius << "\n";
            std::cout << "Is lower-dimensional: " << (rcheb.is_lower_dimensional ? "YES" : "NO") << "\n";
            std::cout << "Active constraints: " << rcheb.num_active << "\n";
            print_solution_summary("RELAXED CHEBYSHEV", mip, rcheb.x_center, true);
        }
    }

    // ============================================================
    // 5. INTERIOR POINT (Robust for lower-dimensional polyhedra)
    // ============================================================
    if (mode == "--interior" || mode == "--all") {
        std::cout << "\n========================================\n";
        std::cout << "COMPUTING ROBUST INTERIOR POINT\n";
        std::cout << "========================================\n";
        std::cout << "Note: Automatically handles lower-dimensional\n";
        std::cout << "      polyhedra by using analytic center.\n";

        auto t2 = std::chrono::high_resolution_clock::now();
        InteriorPointResult ip = compute_interior_point(mip);
        auto t3 = std::chrono::high_resolution_clock::now();

        double ip_time = std::chrono::duration<double>(t3 - t2).count();

        if (!ip.feasible) {
            std::cout << "Interior point computation failed.\n";
        } else {
            std::cout << "Interior point time: " << ip_time << " s\n";
            std::cout << "Method used: ";
            if (ip.method_used == InteriorPointMethod::CHEBYSHEV) {
                std::cout << "Chebyshev (true interior)\n";
            } else if (ip.method_used == InteriorPointMethod::ANALYTIC_CENTER) {
                std::cout << "Analytic Center (lower-dim)\n";
            } else if (ip.method_used == InteriorPointMethod::RELATIVE_INTERIOR_LP) {
                std::cout << "Relative Interior LP\n";
            } else if (ip.method_used == InteriorPointMethod::PERTURBED_LP) {
                std::cout << "Perturbed LP\n";
            }
            std::cout << "Is interior: " << (ip.is_interior ? "YES" : "NO") << "\n";
            std::cout << "Min slack: " << ip.min_slack << "\n";
            std::cout << "Radius proxy: " << ip.radius_proxy << "\n";
            std::cout << "Active constraints: " << ip.num_active_constraints << "\n";
            print_solution_summary("INTERIOR POINT", mip, ip.x_center, true);
        }
    }

    // ============================================================
    // 6. GPU ROUNDING HEURISTIC (DEFAULT)
    // ============================================================
    if (mode == "--rounding" || mode == "--all") {
        std::cout << "\n========================================\n";
        std::cout << "GPU ROUNDING HEURISTIC (K=" << K << ")\n";
        std::cout << "========================================\n";

        std::vector<double> rounding_start_point;
        std::string start_method = "LP Relaxation";

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

        // Decide starting point for rounding
        if (use_interior_point) {
            std::cout << "\nStep 1b: Computing interior point for rounding...\n";
            auto t_ip0 = std::chrono::high_resolution_clock::now();
            InteriorPointResult ip = compute_interior_point(mip);
            auto t_ip1 = std::chrono::high_resolution_clock::now();
            double ip_time = std::chrono::duration<double>(t_ip1 - t_ip0).count();

            if (ip.feasible) {
                rounding_start_point = ip.x_center;
                start_method = "Interior Point";
                std::cout << "Interior point time: " << ip_time << " s\n";
                std::cout << "Using interior point (method: ";
                if (ip.method_used == InteriorPointMethod::CHEBYSHEV) {
                    std::cout << "Chebyshev";
                } else if (ip.method_used == InteriorPointMethod::ANALYTIC_CENTER) {
                    std::cout << "Analytic Center";
                } else if (ip.method_used == InteriorPointMethod::RELATIVE_INTERIOR_LP) {
                    std::cout << "Relative Interior LP";
                } else if (ip.method_used == InteriorPointMethod::PERTURBED_LP) {
                    std::cout << "Perturbed LP";
                }
                std::cout << ")\n";
            } else {
                std::cout << "Interior point failed, falling back to LP solution\n";
                rounding_start_point = lp.x;
            }
        } else {
            rounding_start_point = lp.x;
        }

        // Run GPU rounding
        std::cout << "\nStep 2: Running GPU rounding with K=" << K << " candidates...\n";
        auto t_round0 = std::chrono::high_resolution_clock::now();
        Solution round_sol = run_gpu_neighbourhood_rounding(mip, rounding_start_point, K);
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
        std::cout << "Starting point:       " << start_method << "\n";
        std::cout << "LP solve time:        " << std::setw(10) << lp_time << " s\n";
        std::cout << "Rounding time:        " << std::setw(10) << round_time << " s\n";
        std::cout << "Total time:           " << std::setw(10) << (lp_time + (use_interior_point ? 0 : 0) + round_time) << " s\n";
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