#include "mip_problem.h"
#include "feasibility_pump.h"
#include "solution.h"
#include <iostream>
#include <fstream>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./fp_solver instance.mps\n";
        return 1;
    }

    MIPProblem mip;
    mip.load_from_mps(argv[1]);
    mip.finalize();

    Solution sol;
    FeasibilityPump fp(mip);
if (fp.run(sol,500)) {
    std::cout << "Feasible solution found by FP\n";
    std::cout << "Time (sec): " << fp.last_runtime_sec() << "\n";
    std::cout << "Objective : " << sol.obj_value << "\n";

    bool ok = mip.check_feasible(sol.x, 1e-6, 1e-5);

    if (ok) {
        std::cout << "Solution PASSES feasibility check\n";
    } else {
        std::cout << "Solution FAILS feasibility check\n";
    }

}
else {
        std::cout << "No feasible solution found\n";
        std::cout << "Time (sec): " << fp.last_runtime_sec() << "\n";
    }

    return 0;
}

