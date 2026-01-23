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
    params.max_iters = 1000000;

    Solution sol = fj.run(params);
    std::cout<<"Solution :\n";
  //  for (auto var: sol.x)std::cout<<var<<" ";
    std::cout<<"\n";
    std::cout<<prob.check_feasible(sol.x)<<"\n";
    double objVal=0;
    for (int i=0; i<prob.num_cols; i++)objVal+=prob.c[i]*sol.x[i];
    std::vector<double> init_weights(prob.num_rows, 1.0);
    objVal-=prob.obj_offset;
    std::cout<<"Objective value = "<<objVal<<"\n";
    if (sol.feasible) {
        std::cout << "Feasible solution found\n";
    } else {
        std::cout << "No feasible solution found\n";
    }

    return 0;
}

