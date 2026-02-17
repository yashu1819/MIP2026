#ifndef CHEBYSHEV_CENTER_H
#define CHEBYSHEV_CENTER_H

#include "mip_problem.h"
#include <vector>

struct ChebyshevResult {
    std::vector<double> x_center;
    double radius = 0.0;
    bool feasible = false;
};

ChebyshevResult compute_chebyshev_center(const MIPProblem& mip);

#endif
