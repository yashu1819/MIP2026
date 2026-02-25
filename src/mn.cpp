// mn.cpp
#include "mip_problem.h"
#include "fp.h"
#include <iostream>
#include <string>
#include <cstdlib>
#include <ctime>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <instance.mps>\n";
        return 1;
    }

    double start_time = getTime();

    std::string filename = argv[1];
    std::cout << "Loading instance: " << filename << "\n";

    MIPProblem mip;
    try {
        mip.load_from_mps(filename);
        mip.finalize();
        std::cout << "Problem loaded: " << mip.num_rows << " rows, " 
                  << mip.num_cols << " cols\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    // Run Feasibility Pump
    FPResult result = run_feasibility_pump_cuda(
        mip, 
        180.0,  // 3 min time limit
        1e-5, 1e-6, 1e-5
    );

    if (result.found_feasible) {
        std::cout << "\nSUCCESS!\n";
        std::cout << "Objective : " << result.objective_value << "\n";
        std::cout << "Iterations: " << result.iterations_used << "\n";
        std::cout << "x = [";
        for (double v : result.solution) std::cout << " " << v;
        std::cout << " ]\n";
    } else {
        std::cout << "\n" << result.status_msg << "\n";
        std::cout << "Iterations performed: " << result.iterations_used << "\n";
    }

    double time_taken = getTime() - start_time;
    std::cout << "Time taken: " << time_taken << " seconds\n";

    return 0;
}
