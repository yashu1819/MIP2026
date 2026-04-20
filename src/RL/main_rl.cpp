#include "rl_heuristic.h"
#include "../Definition/mip_problem.h"
#include <iostream>
#include <string>
#include <chrono>

// Simple test driver for RL-SPH heuristic
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <mps_file> [time_limit] [max_steps]" << std::endl;
        return 1;
    }

    std::string mps_file = argv[1];
    double time_limit = (argc >= 3) ? std::stod(argv[2]) : 60.0;
    int max_steps = (argc >= 4) ? std::stoi(argv[3]) : 2000;

    std::cout << "RL-SPH Heuristic Test" << std::endl;
    std::cout << "=====================" << std::endl;
    std::cout << "MPS file: " << mps_file << std::endl;
    std::cout << "Time limit: " << time_limit << "s" << std::endl;
    std::cout << "Max steps: " << max_steps << std::endl;
    std::cout << std::endl;

    // Load MIP problem
    MIPProblem mip;
    try {
        mip.load_from_mps(mps_file);
        mip.finalize();
        std::cout << "Loaded MIP problem:" << std::endl;
        std::cout << "  Variables: " << mip.num_cols << std::endl;
        std::cout << "  Constraints: " << mip.num_rows << std::endl;
        std::cout << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading MPS file: " << e.what() << std::endl;
        return 1;
    }

    // Configure RL heuristic
    rl_sph::RLConfig config;
    config.time_limit = time_limit;
    config.max_steps = max_steps;
    config.phase1_steps = 500;
    config.init_method = "lp";  // or "random"
    config.seed = 42;

    // Run RL heuristic
    std::cout << "Running RL-SPH heuristic..." << std::endl;
    auto start = std::chrono::steady_clock::now();

    Solution solution = rl_sph::run_rl_heuristic(mip, config);

    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    // Report results
    std::cout << std::endl;
    std::cout << "Results:" << std::endl;
    std::cout << "--------" << std::endl;
    std::cout << "Time elapsed: " << elapsed << "s" << std::endl;
    std::cout << "Feasible: " << (solution.feasible ? "Yes" : "No") << std::endl;
    std::cout << "Objective value: " << solution.obj_value << std::endl;

    // Count non-zero variables
    int nonzero = 0;
    for (double v : solution.x) {
        if (std::abs(v) > 1e-6) nonzero++;
    }
    std::cout << "Non-zero variables: " << nonzero << " / " << mip.num_cols << std::endl;

    // Verify feasibility
    if (solution.feasible) {
        bool verified = mip.check_feasible(solution.x);
        std::cout << "Feasibility verified: " << (verified ? "Yes" : "No") << std::endl;
    }

    std::cout << std::endl;

    return solution.feasible ? 0 : 1;
}
