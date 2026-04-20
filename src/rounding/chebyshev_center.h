#ifndef CHEBYSHEV_CENTER_H
#define CHEBYSHEV_CENTER_H

#include "mip_problem.h"
#include <utility>
#include <vector>

/*
 * ChebyshevResult:
 * - x_center: the computed Chebyshev center
 * - radius: the maximum inscribed ball radius
 * - feasible: true if LP was solved successfully
 *
 * Note: For lower-dimensional polyhedra, radius will be 0.
 * In such cases, use compute_relaxed_chebyshev_center() or
 * compute_analytic_center() instead.
 */
struct ChebyshevResult {
    std::vector<double> x_center;
    double radius = 0.0;
    bool feasible = false;
};

/*
 * Compute the standard Chebyshev center of the LP relaxation.
 * Returns both the result and the augmented MIP formulation used.
 */
std::pair<ChebyshevResult, MIPProblem> compute_chebyshev_center(const MIPProblem& mip);

#endif
