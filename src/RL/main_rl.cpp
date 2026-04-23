#include "rl_heuristic.h"
#include "../Definition/mip_problem.h"
#include <iostream>
#include <string>
#include <chrono>
#include <cmath>

// RL-SPH Inference Driver
// Usage: ./rl_sph_test <mps_file> [time_limit] [max_steps] [model_path]
//
// Arguments:
//   mps_file    : Path to MPS format MILP file (required)
//   time_limit  : Time limit in seconds (default: 60)
//   max_steps   : Maximum search steps (default: 2000)
//   model_path  : Path to trained model weights (optional, uses random if not given)
//
// Example:
//   ./rl_sph_test problem.mps 60 2000
//   ./rl_sph_test problem.mps 60 2000 trained_model.pt

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <mps_file> [time_limit] [max_steps] [model_path]" << std::endl;
        std::cerr << std::endl;
        std::cerr << "Arguments:" << std::endl;
        std::cerr << "  mps_file    : Path to MPS format MILP file" << std::endl;
        std::cerr << "  time_limit  : Time limit in seconds (default: 60)" << std::endl;
        std::cerr << "  max_steps   : Maximum search steps (default: 2000)" << std::endl;
        std::cerr << "  model_path  : Path to trained model (optional)" << std::endl;
        return 1;
    }

    std::string mps_file = argv[1];
    double time_limit = (argc >= 3) ? std::stod(argv[2]) : 60.0;
    int max_steps = (argc >= 4) ? std::stoi(argv[3]) : 2000;
    std::string model_path = (argc >= 5) ? argv[4] : "";

    std::cout << "RL-SPH Heuristic (MILP)" << std::endl;
    std::cout << "========================" << std::endl;
    std::cout << "MPS file:    " << mps_file << std::endl;
    std::cout << "Time limit:  " << time_limit << "s" << std::endl;
    std::cout << "Max steps:   " << max_steps << std::endl;
    if (!model_path.empty()) {
        std::cout << "Model:       " << model_path << std::endl;
    } else {
        std::cout << "Model:       (untrained — random weights)" << std::endl;
    }
    std::cout << std::endl;

    // Load MIP problem
    MIPProblem mip;
    try {
        mip.load_from_mps(mps_file);
        mip.finalize();
    } catch (const std::exception& e) {
        std::cerr << "Error loading MPS file: " << e.what() << std::endl;
        return 1;
    }

    // Count variable types
    int num_binary = 0, num_integer = 0, num_continuous = 0;
    for (int i = 0; i < mip.num_cols; ++i) {
        if (mip.vartype[i] == VarType::BINARY) num_binary++;
        else if (mip.vartype[i] == VarType::INTEGER) num_integer++;
        else num_continuous++;
    }

    std::cout << "Problem loaded:" << std::endl;
    std::cout << "  Variables:    " << mip.num_cols
              << " (binary=" << num_binary
              << ", integer=" << num_integer
              << ", continuous=" << num_continuous << ")" << std::endl;
    std::cout << "  Constraints:  " << mip.num_rows << std::endl;
    std::cout << std::endl;

    // Configure RL heuristic
    rl_sph::RLConfig config;
    config.time_limit = time_limit;
    config.max_steps = max_steps;
    config.phase1_steps = std::min(500, max_steps / 4);
    config.init_method = "lp";
    config.model_path = model_path;
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
    std::cout << "Time elapsed:  " << elapsed << "s" << std::endl;
    std::cout << "Feasible:      " << (solution.feasible ? "Yes" : "No") << std::endl;
    std::cout << "Objective:     " << solution.obj_value << std::endl;

    // Count non-zero variables
    int nonzero = 0;
    for (double v : solution.x) {
        if (std::abs(v) > 1e-6) nonzero++;
    }
    std::cout << "Non-zero vars: " << nonzero << " / " << mip.num_cols << std::endl;

    // Verify feasibility
    if (solution.feasible) {
        bool verified = mip.check_feasible(solution.x);
        std::cout << "Verified:      " << (verified ? "Yes" : "No") << std::endl;
    }

    std::cout << std::endl;
    return solution.feasible ? 0 : 1;
}
