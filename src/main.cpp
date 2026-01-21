#include "mip_problem.h"
#include "feasibility_jump.h"
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./fj_solver instance.mps\n";
        return 1;
    }

    MIPProblem prob;
    prob.load_from_mps(argv[1]);
    prob.finalize();

    FeasibilityJump fj(prob);

    FeasibilityJumpParams params;
    params.max_iters = 5000;

    Solution sol = fj.run(params);
    std::cout<<"Solution :\n";
  //  for (auto var: sol.x)std::cout<<var<<" ";
    std::cout<<"\n";
    if (sol.feasible) {
        std::cout << "Feasible solution found\n";
    } else {
        std::cout << "No feasible solution found\n";
    }

    return 0;
}

