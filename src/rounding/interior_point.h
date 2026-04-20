#ifndef INTERIOR_POINT_H
#define INTERIOR_POINT_H

#include "mip_problem.h"
#include <vector>
#include <utility>

/*
 * InteriorPointResult:
 * - x_center: the computed interior point
 * - method_used: which method was used to find the point
 * - is_interior: true if point is in relative interior (not on boundary)
 * - min_slack: minimum slack across all constraints
 * - radius_proxy: proxy for "how interior" the point is
 *
 * Methods (in order of preference):
 * 1. Chebyshev (if radius > 0) - best, gives true interior point
 * 2. Analytic Center - works for lower-dimensional polyhedra
 * 3. Relative Interior LP - secondary optimization on the face
 * 4. Perturbed LP - small perturbation to get off the boundary
 */
enum class InteriorPointMethod {
    CHEBYSHEV,
    ANALYTIC_CENTER,
    RELATIVE_INTERIOR_LP,
    PERTURBED_LP
};

struct InteriorPointResult {
    std::vector<double> x_center;
    InteriorPointMethod method_used = InteriorPointMethod::ANALYTIC_CENTER;
    bool is_interior = false;
    double min_slack = 0.0;
    double radius_proxy = 0.0;
    bool feasible = false;

    // Diagnostics
    int num_active_constraints = 0;
    std::vector<bool> is_constraint_active;
};

/*
 * Compute a point in the relative interior of the polyhedron.
 *
 * This is the main entry point - it automatically detects when the
 * polyhedron is lower-dimensional and uses the appropriate method.
 *
 * Strategy:
 * 1. Try standard Chebyshev - if radius > eps, we have a true interior point
 * 2. If radius ~ 0, detect lower-dimensional structure
 * 3. Use analytic center (works for lower-dimensional polyhedra)
 * 4. Fallback: solve a secondary LP to maximize distance from active constraints
 *
 * Parameters:
 * - mip: the MIP problem
 * - eps: tolerance for detecting zero radius (default 1e-8)
 * - min_slack_target: target minimum slack for analytic center (default 1e-4)
 *
 * Returns:
 * - InteriorPointResult with the point and diagnostics
 */
InteriorPointResult compute_interior_point(
    const MIPProblem& mip,
    double eps = 1e-8,
    double min_slack_target = 1e-4
);

/*
 * Detect the structure of the polyhedron.
 *
 * Identifies:
 * - Equality constraints (explicit or implicit)
 * - Fixed variables
 * - Redundant constraints
 *
 * Returns:
 * - Pair of (num_equalities, num_fixed_vars)
 */
std::pair<int, int> detect_polyhedron_structure(
    const MIPProblem& mip,
    double tol = 1e-8
);

/*
 * Find a point that maximizes distance from a set of active constraints.
 *
 * Given a set of constraints that are known to be active (tight),
 * find a point that is as far as possible from these constraints
 * while still satisfying all constraints.
 *
 * This is useful when the polyhedron is lower-dimensional - we want
 * to be "central" within the feasible face.
 */
InteriorPointResult compute_relative_interior_lp(
    const MIPProblem& mip,
    const std::vector<bool>& active_constraints,
    double eps = 1e-6
);

/*
 * Perturb a boundary point to move it into the relative interior.
 *
 * Given a point on the boundary of the polyhedron, apply small
 * perturbations to move it into the interior (if possible).
 */
InteriorPointResult perturb_to_interior(
    const MIPProblem& mip,
    const std::vector<double>& x_boundary,
    double perturbation_size = 1e-4
);

#endif
