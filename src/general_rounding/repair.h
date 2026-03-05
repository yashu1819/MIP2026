#ifndef REPAIR_H
#define REPAIR_H

#include "../Definition/mip_problem.h"
#include "../Definition/solution.h"

bool repair_solution(
    const MIPProblem& mip,
    std::vector<double>& x,
    int max_iter
);

#endif