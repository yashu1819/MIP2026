#include "mip_problem.h"
#include "feasibility_pump.h"
#include <iostream>
#include <fstream>
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
    if (argc < 2) {
        std::cerr << "Usage: ./fj_solver instance.mps\n";
        return 1;
    }
    double t1 = getTime();
    MIPProblem prob;
    prob.load_from_mps(argv[1]);
    prob.finalize();

    FeasibilityPump fj(prob);

    FeasibilityPumpParams params;
   // params.max_iters = 1000000;

    Solution sol = fj.run(params);
//    std::cout<<"Solution :\n";
  //  for (auto var: sol.x)std::cout<<var<<" ";
//    std::cout<<"\n";
    double objVal=0;
    for (int i=0; i<prob.num_cols; i++)objVal+=prob.c[i]*sol.x[i];
    objVal-=prob.obj_offset;
    std::cout<<"Objective value = "<<objVal<<"\n";
    if (prob.check_feasible(sol.x)) {
        std::cout << "Feasible solution found\n";
    } else {
        std::cout << "No feasible solution found\n";
    }

    double t2= getTime();
    std::cout<<"Time taken = "<<t2-t1<<"\n";
    writeToFile("../Benchmark/FJresults.csv", argv[1], t2-t1, objVal, prob.check_feasible(sol.x));
    return 0;
}

