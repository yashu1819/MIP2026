#include <iostream>
#include <string>
#include <chrono>

#include "mip_problem.h"
#include "lp_relaxation.h"
#include "solution.h"
#include "neighbourhood_rounding.h"


// // forward declaration from neighbourhood_rounding.cpp
// Solution run_gpu_neighborhood_rounding(
//     const MIPProblem& mip,
//     const std::vector<double>& x_lp,
//     int K = 1024
// );

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cout << "Usage: ./solver instance.mps\n";
        return 1;
    }

    std::string filename = argv[1];

    std::cout << "=====================================\n";
    std::cout << "Loading MIP: " << filename << "\n";
    std::cout << "=====================================\n";

    MIPProblem mip;
    mip.load_from_mps(filename);
    mip.finalize();

    std::cout << "Rows: " << mip.num_rows
              << "  Cols: " << mip.num_cols << "\n";

    // -----------------------------
    // 1. Solve LP relaxation (cuOpt)
    // -----------------------------
    std::cout << "\nSolving LP relaxation using cuOpt...\n";

    LPRelaxation lp(mip);

    auto t0 = std::chrono::high_resolution_clock::now();
    bool lp_ok = lp.solve();
    auto t1 = std::chrono::high_resolution_clock::now();

    if (!lp_ok) {
        std::cout << "LP relaxation infeasible.\n";
        return 1;
    }

    double lp_time =
        std::chrono::duration<double>(t1 - t0).count();

    std::cout << "LP solved.\n";
    std::cout << "LP objective = " << lp.obj_value << "\n";
    std::cout << "LP time      = " << lp_time << " sec\n";

    // -----------------------------
    // 2. GPU Neighborhood Rounding
    // -----------------------------
    std::cout << "\nRunning GPU neighborhood rounding...\n";

    auto t2 = std::chrono::high_resolution_clock::now();

    Solution sol =
        run_gpu_neighbourhood_rounding(mip, lp.x, 1024);

    auto t3 = std::chrono::high_resolution_clock::now();

    double heur_time =
        std::chrono::duration<double>(t3 - t2).count();

    std::cout << "Heuristic time = " << heur_time << " sec\n";

    // -----------------------------
    // 3. Print result
    // -----------------------------
    if (sol.feasible) {
        std::cout << "\nFeasible integer solution found!\n";
        std::cout << "Objective = " << sol.obj_value << "\n";
    } else {
        std::cout << "\nNo feasible solution found.\n";
    }

    std::cout << "=====================================\n";
    return 0;
}
