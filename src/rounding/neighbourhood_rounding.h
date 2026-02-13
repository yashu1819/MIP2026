#ifndef NEIGHBOURHOOD_ROUNDING_H
#define NEIGHBOURHOOD_ROUNDING_H

#include "mip_problem.h"
#include "solution.h"
#include <vector>

Solution run_gpu_neighbourhood_rounding(
    const MIPProblem& mip,
    const std::vector<double>& x_lp,
    int K
);

void compute_locks(
    const MIPProblem& mip,
    std::vector<int>& locks_up,
    std::vector<int>& locks_down
);

bool repair_solution(
    const MIPProblem& mip,
    std::vector<double>& x,
    const std::vector<bool>& is_free,
    const std::vector<int>& locks_up,
    const std::vector<int>& locks_down,
    int max_iter
);

#endif
