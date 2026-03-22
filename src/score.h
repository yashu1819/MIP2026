#ifndef SCORE_H
#define SCORE_H

#include "mip_problem.h"
#include <vector>

double compute_violation_score(
    const MIPProblem& mip,
    const std::vector<double>& activity
);

#endif
