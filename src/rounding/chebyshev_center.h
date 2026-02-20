#ifndef CHEBYSHEV_CENTER_H
#define CHEBYSHEV_CENTER_H

#include "mip_problem.h"
#include<utility>
#include <vector>

struct ChebyshevResult {
    std::vector<double> x_center;
    double radius = 0.0;
    bool feasible = false;
};

std::pair<ChebyshevResult, MIPProblem> compute_chebyshev_center(const MIPProblem& mip);

#endif
