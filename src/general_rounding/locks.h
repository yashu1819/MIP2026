#ifndef GENERAL_ROUNDING_LOCKS_H
#define GENERAL_ROUNDING_LOCKS_H

#include "../Definition/mip_problem.h"
#include <vector>

struct VariableLocks {
    std::vector<int> up_locks;
    std::vector<int> down_locks;
};

// GPU computation of locks
VariableLocks compute_variable_locks_gpu(const MIPProblem& mip);

#endif