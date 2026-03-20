#ifndef ACTIVITY_H
#define ACTIVITY_H

#include "../mip_problem.h"
#include <vector>

void compute_constraint_activity_gpu(
    const MIPProblem& mip,
    const std::vector<double>& x,
    std::vector<double>& activity
);

#endif