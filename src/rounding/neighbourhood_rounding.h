#ifndef NEIGHBOURHOOD_ROUNDING_H
#define NEIGHBOURHOOD_ROUNDING_H

#include "mip_problem.h"
#include "solution.h"

Solution run_gpu_neighborhood_rounding(
    const MIPProblem& mip,
    const std::vector<double>& x_lp,
    int K = 1024
);

#endif
