#ifndef CHEBYSHEV_CENTER_RELAXED_H
#define CHEBYSHEV_CENTER_RELAXED_H

#include "mip_problem.h"
#include <utility>
#include <vector>

/*
 * RelaxedChebyshevResult:
 * - x_center: the computed center point
 * - radius: the achieved radius (may be 0 for lower-dimensional polyhedra)
 * - relaxed_radius: radius in the relaxed formulation (always >= 0)
 * - feasible: true if optimization succeeded
 * - num_active: number of active constraints at solution
 * - is_lower_dimensional: true if radius ≈ 0 (polyhedron has no interior)
 *
 * This formulation handles lower-dimensional polyhedra by:
 * 1. Adding soft constraints with penalty for violation
 * 2. Using a modified objective that works even when max radius = 0
 * 3. Finding the "central point" on the feasible face
 */

struct RelaxedChebyshevResult {
    std::vector<double> x_center;
    double radius = 0.0;
    double relaxed_radius = 0.0;
    bool feasible = false;
    int num_active = 0;
    bool is_lower_dimensional = false;
};

/*
 * Compute Chebyshev center with relaxation for lower-dimensional polyhedra.
 *
 * Standard Chebyshev center solves:
 *   max  r
 *   s.t. a_i^T x + ||a_i|| * r <= b_i  for all i
 *
 * This fails when the polyhedron is lower-dimensional (r = 0).
 *
 * The relaxed formulation:
 *   max  r - M * sum(s_i)
 *   s.t. a_i^T x + ||a_i|| * r <= b_i + s_i,  s_i >= 0
 *
 * Where M is a large penalty. This allows slight violations to find
 * a "central" point even on a lower-dimensional face.
 *
 * Alternative: Use the analytic center when radius = 0 is detected.
 */
RelaxedChebyshevResult compute_relaxed_chebyshev_center(
    const MIPProblem& mip,
    double penalty = 1000.0,  // Penalty for constraint violation
    bool fallback_to_analytic = true  // Use analytic center if radius = 0
);

/*
 * Compute the Chebyshev center of the projection onto a subset of variables.
 *
 * For lower-dimensional polyhedra, sometimes the issue is that some variables
 * are fixed by equality constraints. This function attempts to identify
 * the "free" subspace and compute the center there.
 */
RelaxedChebyshevResult compute_projected_chebyshev_center(
    const MIPProblem& mip,
    const std::vector<int>& free_vars,  // Indices of potentially free variables
    double eps = 1e-6
);

/*
 * Find a point in the relative interior of the polyhedron.
 *
 * Uses the following strategy:
 * 1. Try standard Chebyshev - if radius > 0, done
 * 2. If radius = 0, identify active constraints
 * 3. Solve a secondary LP to maximize distance from active constraints
 *
 * This gives a point that is "central" within the feasible face.
 */
RelaxedChebyshevResult compute_relative_interior_point(
    const MIPProblem& mip,
    double active_tol = 1e-6
);

#endif
