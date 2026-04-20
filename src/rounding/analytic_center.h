#ifndef ANALYTIC_CENTER_H
#define ANALYTIC_CENTER_H

#include "mip_problem.h"
#include <vector>

/*
 * AnalyticCenterResult:
 * - x_center: the analytic center of the polyhedron
 * - slack: minimum slack across all constraints (should be > 0 for interior)
 * - feasible: true if LP was solved successfully
 * - iterations: number of Newton iterations used
 *
 * The analytic center is defined as the minimizer of the logarithmic barrier:
 *   min  -sum(log(s_i))
 *   s.t. Ax + s = b, s > 0
 *
 * Unlike Chebyshev center, this works for lower-dimensional polyhedra because
 * it converges to the "center" of the feasible face rather than trying to
 * fit a ball of positive radius.
 */

struct AnalyticCenterResult {
    std::vector<double> x_center;
    double min_slack = 0.0;
    bool feasible = false;
    int iterations = 0;
};

/*
 * Compute the analytic center of the LP relaxation feasible region.
 *
 * Uses a damped Newton method on the logarithmic barrier function.
 * The algorithm:
 * 1. Start from a strictly feasible point (via LP solve with slack)
 * 2. Apply Newton steps to minimize the barrier function
 * 3. Use line search to ensure we stay in the feasible region
 *
 * Parameters:
 * - mip: the MIP problem (only constraints matter, not integrality)
 * - max_iter: maximum Newton iterations (default 50)
 * - tol: convergence tolerance (default 1e-8)
 * - backtrack_alpha: line search parameter (default 0.01)
 * - backtrack_beta: line search parameter (default 0.5)
 *
 * Returns:
 * - AnalyticCenterResult with the center point and diagnostics
 */
AnalyticCenterResult compute_analytic_center(
    const MIPProblem& mip,
    int max_iter = 50,
    double tol = 1e-8,
    double backtrack_alpha = 0.01,
    double backtrack_beta = 0.5
);

/*
 * Compute analytic center starting from a given point.
 * Useful when you already have a feasible LP solution.
 */
AnalyticCenterResult compute_analytic_center_from_point(
    const MIPProblem& mip,
    const std::vector<double>& x0,
    int max_iter = 50,
    double tol = 1e-8
);

#endif
