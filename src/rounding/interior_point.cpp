#include "interior_point.h"
#include "chebyshev_center.h"
#include "chebyshev_center_relaxed.h"
#include "analytic_center.h"
#include "../relaxation/lp_relaxation.h"
#include <cmath>
#include <algorithm>
#include <iostream>

/*
 * Interior Point Computation for Lower-Dimensional Polyhedra
 *
 * The key challenge: standard Chebyshev center maximizes the radius of an
 * inscribed ball. For lower-dimensional polyhedra (e.g., a square in 3D space),
 * the maximum radius is 0, and the solution is not unique.
 *
 * Our approach:
 * 1. Try Chebyshev first - if radius > 0, we have a true interior point
 * 2. If radius = 0, use analytic center which converges to the "center" of the face
 * 3. As fallback, solve a secondary LP to maximize distance from active constraints
 */

// Compute minimum slack across all constraints
static double compute_min_slack(const MIPProblem& mip, const std::vector<double>& x) {
    double min_s = 1e100;
    for (int i = 0; i < mip.num_rows; i++) {
        double ax = 0.0;
        for (int p = mip.csr_row_ptr[i]; p < mip.csr_row_ptr[i + 1]; p++) {
            ax += mip.csr_val[p] * x[mip.csr_col_idx[p]];
        }
        min_s = std::min(min_s, mip.b[i] - ax);
    }
    return min_s;
}

// Count active constraints
static int count_active_constraints(
    const MIPProblem& mip,
    const std::vector<double>& x,
    double tol = 1e-6
) {
    int count = 0;
    for (int i = 0; i < mip.num_rows; i++) {
        double ax = 0.0;
        for (int p = mip.csr_row_ptr[i]; p < mip.csr_row_ptr[i + 1]; p++) {
            ax += mip.csr_val[p] * x[mip.csr_col_idx[p]];
        }
        if (std::abs(mip.b[i] - ax) < tol) {
            count++;
        }
    }
    return count;
}

std::pair<int, int> detect_polyhedron_structure(
    const MIPProblem& mip,
    double tol
) {
    int num_equalities = 0;
    int num_fixed_vars = 0;

    // Check for fixed variables
    for (int j = 0; j < mip.num_cols; j++) {
        if (std::abs(mip.ub[j] - mip.lb[j]) < tol) {
            num_fixed_vars++;
        }
    }

    // Check for explicit equality constraints
    // (In our formulation, all constraints are <=, so equalities would be
    //  represented as two constraints: a^T x <= b and -a^T x <= -b)
    // For now, we just count constraints that are likely equalities
    // by checking if they're always tight at any feasible point

    // A more sophisticated check would solve two LPs per constraint:
    // max/min a^T x to see if the constraint is always tight
    // For efficiency, we skip this and rely on the solver to detect

    return std::make_pair(num_equalities, num_fixed_vars);
}

InteriorPointResult compute_relative_interior_lp(
    const MIPProblem& mip,
    const std::vector<bool>& active_constraints,
    double eps
) {
    InteriorPointResult result;
    int n = mip.num_cols;
    int m = mip.num_rows;

    // Count active constraints
    int num_active = 0;
    for (bool active : active_constraints) {
        if (active) num_active++;
    }

    if (num_active == 0) {
        // No active constraints - any interior point works
        // Solve a simple LP to get an interior point
        LPRelaxation lp(mip);
        if (!lp.solve()) {
            result.feasible = false;
            return result;
        }
        result.x_center = lp.x;
        result.min_slack = compute_min_slack(mip, lp.x);
        result.feasible = true;
        result.method_used = InteriorPointMethod::PERTURBED_LP;
        return result;
    }

    // Build a new LP that maximizes distance from active constraints
    // Objective: maximize sum of slacks for active constraints
    // This is: max sum_i (b_i - a_i^T x) for active i
    //        = sum_i b_i - (sum_i a_i)^T x
    // So we minimize (sum_i a_i)^T x

    MIPProblem interior_mip;
    interior_mip.num_rows = m;
    interior_mip.num_cols = n;

    // New objective: minimize sum of coefficients for active constraints
    interior_mip.c.assign(n, 0.0);
    for (int i = 0; i < m; i++) {
        if (active_constraints[i]) {
            for (int p = mip.csr_row_ptr[i]; p < mip.csr_row_ptr[i + 1]; p++) {
                interior_mip.c[mip.csr_col_idx[p]] += mip.csr_val[p];
            }
        }
    }
    interior_mip.obj_offset = 0.0;

    // Keep original bounds and constraints
    interior_mip.lb = mip.lb;
    interior_mip.ub = mip.ub;
    interior_mip.vartype.assign(n, VarType::CONTINUOUS);
    interior_mip.b = mip.b;
    interior_mip.coo_row = mip.coo_row;
    interior_mip.coo_col = mip.coo_col;
    interior_mip.coo_val = mip.coo_val;
    interior_mip.finalize();

    // Solve - we want to MINIMIZE the objective
    // So we negate it and maximize
    for (int j = 0; j < n; j++) {
        interior_mip.c[j] = -interior_mip.c[j];
    }

    LPRelaxation lp(interior_mip);
    if (!lp.solve()) {
        result.feasible = false;
        return result;
    }

    result.x_center = lp.x;
    result.min_slack = compute_min_slack(mip, lp.x);
    result.feasible = true;
    result.method_used = InteriorPointMethod::RELATIVE_INTERIOR_LP;

    // Count how many constraints are still active
    result.num_active_constraints = count_active_constraints(mip, lp.x, eps);
    result.is_constraint_active.assign(m, false);
    for (int i = 0; i < m; i++) {
        double ax = 0.0;
        for (int p = mip.csr_row_ptr[i]; p < mip.csr_row_ptr[i + 1]; p++) {
            ax += mip.csr_val[p] * lp.x[mip.csr_col_idx[p]];
        }
        result.is_constraint_active[i] = (std::abs(mip.b[i] - ax) < eps);
    }

    return result;
}

InteriorPointResult perturb_to_interior(
    const MIPProblem& mip,
    const std::vector<double>& x_boundary,
    double perturbation_size
) {
    InteriorPointResult result;
    int n = mip.num_cols;
    int m = mip.num_rows;

    // Identify which constraints are active at the boundary point
    std::vector<bool> active(m, false);
    for (int i = 0; i < m; i++) {
        double ax = 0.0;
        for (int p = mip.csr_row_ptr[i]; p < mip.csr_row_ptr[i + 1]; p++) {
            ax += mip.csr_val[p] * x_boundary[mip.csr_col_idx[p]];
        }
        if (std::abs(mip.b[i] - ax) < 1e-6) {
            active[i] = true;
        }
    }

    // Compute a perturbation direction
    // We want to move in a direction that increases slack for active constraints
    // Direction d should satisfy: a_i^T d > 0 for active constraints with a_i^T x = b_i

    // Simple approach: use gradient of active constraints
    std::vector<double> direction(n, 0.0);
    for (int i = 0; i < m; i++) {
        if (active[i]) {
            for (int p = mip.csr_row_ptr[i]; p < mip.csr_row_ptr[i + 1]; p++) {
                direction[mip.csr_col_idx[p]] -= mip.csr_val[p];  // Move opposite to gradient
            }
        }
    }

    // Normalize direction
    double norm = 0.0;
    for (int j = 0; j < n; j++) {
        norm += direction[j] * direction[j];
    }
    norm = std::sqrt(norm);
    if (norm > 0) {
        for (int j = 0; j < n; j++) {
            direction[j] /= norm;
        }
    }

    // Apply perturbation
    std::vector<double> x_perturbed = x_boundary;
    for (int j = 0; j < n; j++) {
        x_perturbed[j] += perturbation_size * direction[j];
        // Respect bounds
        x_perturbed[j] = std::max(mip.lb[j], std::min(mip.ub[j], x_perturbed[j]));
    }

    result.x_center = x_perturbed;
    result.min_slack = compute_min_slack(mip, x_perturbed);
    result.feasible = true;
    result.method_used = InteriorPointMethod::PERTURBED_LP;
    result.radius_proxy = result.min_slack;

    return result;
}

InteriorPointResult compute_interior_point(
    const MIPProblem& mip,
    double eps,
    double min_slack_target
) {
    InteriorPointResult result;
    int n = mip.num_cols;
    int m = mip.num_rows;

    // ========================================================
    // Step 1: Try standard Chebyshev center
    // ========================================================
    auto [cheb, cheb_mip] = compute_chebyshev_center(mip);

    if (cheb.feasible && cheb.radius > eps) {
        // Success! We have a true interior point with positive radius
        result.x_center = cheb.x_center;
        result.method_used = InteriorPointMethod::CHEBYSHEV;
        result.is_interior = true;
        result.min_slack = compute_min_slack(mip, cheb.x_center);
        result.radius_proxy = cheb.radius;
        result.feasible = true;
        result.num_active_constraints = count_active_constraints(mip, cheb.x_center, eps);
        return result;
    }

    // ========================================================
    // Step 2: Chebyshev gave radius ~ 0 - polyhedron is lower-dimensional
    // Use analytic center which works for lower-dimensional polyhedra
    // ========================================================
    AnalyticCenterResult ac = compute_analytic_center(mip);

    if (ac.feasible) {
        result.x_center = ac.x_center;
        result.method_used = InteriorPointMethod::ANALYTIC_CENTER;
        result.min_slack = ac.min_slack;
        result.radius_proxy = ac.min_slack;  // Use min slack as proxy for radius
        result.feasible = true;
        result.num_active_constraints = count_active_constraints(mip, ac.x_center, eps);

        // Check if we're in the relative interior
        // (min_slack > 0 means we're not on the boundary of the feasible face)
        result.is_interior = (ac.min_slack > min_slack_target);

        // Store which constraints are active
        result.is_constraint_active.assign(m, false);
        for (int i = 0; i < m; i++) {
            double ax = 0.0;
            for (int p = mip.csr_row_ptr[i]; p < mip.csr_row_ptr[i + 1]; p++) {
                ax += mip.csr_val[p] * ac.x_center[mip.csr_col_idx[p]];
            }
            result.is_constraint_active[i] = (std::abs(mip.b[i] - ax) < eps);
        }

        return result;
    }

    // ========================================================
    // Step 3: Analytic center failed - try relative interior LP
    // ========================================================
    // If we have a Chebyshev point (even with radius 0), use it to identify
    // active constraints and solve the relative interior LP
    if (cheb.feasible) {
        std::vector<bool> active(m, false);
        for (int i = 0; i < m; i++) {
            double ax = 0.0;
            for (int p = mip.csr_row_ptr[i]; p < mip.csr_row_ptr[i + 1]; p++) {
                ax += mip.csr_val[p] * cheb.x_center[mip.csr_col_idx[p]];
            }
            active[i] = (std::abs(mip.b[i] - ax) < 1e-5);
        }

        InteriorPointResult rilp = compute_relative_interior_lp(mip, active, eps);
        if (rilp.feasible) {
            rilp.method_used = InteriorPointMethod::RELATIVE_INTERIOR_LP;
            return rilp;
        }
    }

    // ========================================================
    // Step 4: Last resort - perturb the LP solution
    // ========================================================
    LPRelaxation lp(mip);
    if (lp.solve()) {
        InteriorPointResult perturbed = perturb_to_interior(mip, lp.x, 1e-4);
        perturbed.method_used = InteriorPointMethod::PERTURBED_LP;
        return perturbed;
    }

    // Everything failed
    result.feasible = false;
    return result;
}
