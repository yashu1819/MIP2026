#ifndef REPAIR_H
#define REPAIR_H

#include "mip_problem.h"
#include "solution.h"

double compute_objective(
    const MIPProblem& mip,
    const std::vector<double>& x
);

void perturb_solution(
    const MIPProblem& mip,
    std::vector<double>& x,
    const std::vector<bool>& is_fixed,
    int num_changes = 20
);

bool repair_solution(
    const MIPProblem& mip,
    std::vector<double>& x,
    const std::vector<bool>& is_fixed,
    int max_iter
);

bool repair_solution_improved(
    const MIPProblem& mip,
    std::vector<double>& x,
    const std::vector<bool>& is_fixed,
    int top_k,
    int max_iter,
    double tol = 1e-6
);

Solution optimize_with_repair(
    const MIPProblem& mip,
    const std::vector<double>& x_start,
    const std::vector<bool>& is_fixed,
    int top_k,
    double time_limit,
    double tol = 1e-6
);

#endif
