#ifndef LOCKS_ROUNDING_H
#define LOCKS_ROUNDING_H

#include "../Definition/mip_problem.h"
#include "../Definition/solution.h"
#include "locks.h"

#include <vector>

/*
    Lock-Guided Rounding Heuristic

    Input:
        - MIPProblem
        - fractional LP solution
        - variable locks

    Output:
        - rounded integer candidate solution
*/

Solution lock_guided_rounding_gpu(
    const MIPProblem& mip,
    const std::vector<double>& fractional_solution,
    const VariableLocks& locks
);

#endif