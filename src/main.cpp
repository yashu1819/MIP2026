#include "mip_problem.h"
#include <limits>
#include "feasibility_jump.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem>
#include <iomanip>
namespace fs = std::filesystem;

void writeToFile(std::string file, char* argv, double time, double objVal, bool feas){
    fs::path p(argv);
    std::string instanceName = p.stem().string(); 

    // 2. Open the file in append mode
    // std::ios::app ensures we don't overwrite existing results
    std::ofstream csvFile;
    csvFile.open(file, std::ios::app);

    if (csvFile.is_open()) {
        // 3. Write the row: Name, Objective Value, Time
        // Fixed precision for the double values ensures clean CSV formatting
        csvFile << instanceName << "," 
		<<std::fixed << std::setprecision(1) << feas << ","
                << std::fixed << std::setprecision(6) << objVal << "," 
                << std::fixed << std::setprecision(4) << time << "\n";
        
        csvFile.close();
    } else {
        std::cerr << "Error: Could not open CSV file: " << file << std::endl;
    }    
}
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./fj_solver <instance_name> <output_dir>\n";
        return 1;
    }

    std::string instance_file = argv[1];
    std::string output_dir = argv[2];

    std::filesystem::create_directories(output_dir);

    // ---------------- INPUT TIME ----------------
    double t_start = getTime();

    MIPProblem prob;
    prob.load_from_mps(instance_file);
    prob.finalize();

    double t_after_input = getTime();
    double input_time = t_after_input - t_start;

    // ---------------- SOLVE ----------------
    double solve_start = getTime();

    FeasibilityJump fj(prob);
    FeasibilityJumpParams params;
    params.max_iters = 1000000;

    Solution sol = fj.run(params);

    double solve_end = getTime();
    double elapsed = solve_end - solve_start;

    // ---------------- OBJECTIVE (RECOMPUTE) ----------------
    double objVal = 0.0;
    for (int i = 0; i < prob.num_cols; i++)
        objVal += prob.c[i] * sol.x[i];
    objVal -= prob.obj_offset;

    // ---------------- WRITE SOLUTION FILE ----------------
    std::string sol_filename = output_dir + "/solution_1.sol";
    std::ofstream sol_file(sol_filename);

    sol_file << std::setprecision(17);
    sol_file << "=obj= " << objVal << "\n";

    for (int i = 0; i < prob.num_cols; i++) {
        std::string name = prob.var_names[i].empty()
            ? "x" + std::to_string(i)
            : prob.var_names[i];

        sol_file << name << " " << sol.x[i] << "\n";
    }

    sol_file.close();

    // ---------------- WRITE TIMING FILE ----------------
    std::ofstream timing_file(output_dir + "/timing.log");
    timing_file << std::fixed << std::setprecision(3);

    // input time
    timing_file << "input\t" << input_time << "\n";

    // solution time (must be <= 301)
    if (elapsed <= 301.0) {
        timing_file << "solution_1.sol\t" << elapsed << "\n";
    }

    timing_file.close();

    return 0;
}
