#include "mip_problem.h"
#include "lp_relaxation.h"
#include <iostream>

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <instance.mps>\n";
        return 1;
    }

    MIPProblem mip;
    mip.load_from_mps(argv[1]);
    mip.finalize();

    LPRelaxation lp(mip);

    bool ok = lp.solve();

    if (!ok) {
        std::cout << "LP relaxation not solved to feasibility" << std::endl;
        return 2;
    }

    std::cout << "LP relaxation solved" << std::endl;
    std::cout << "Objective value: " << lp.obj_value << std::endl;
  

    return 0;
}

