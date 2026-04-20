#include "chebyshev_center_relaxed.h"
#include "chebyshev_center.h"
#include "lp_relaxation.h"
#include "analytic_center.h"
#include <cmath>
#include <algorithm>
#include <iostream>

/*
 * Relaxed Chebyshev Center Implementation
 *
 * Key insight: When the polyhedron is lower-dimensional, the standard
 * Chebyshev center gives radius = 0. We handle this by:
 *
 * 1. First trying the standard formulation
 * 2. If radius ≈ 0, detecting that we're on a lower-dimensional face
 * 3. Finding the "central" point on that face using a secondary optimization
 */

RelaxedChebyshevResult compute_relaxed_chebyshev_center(
    const MIPProblem& mip,
    double penalty,
    bool fallback_to_analytic
) {
    RelaxedChebyshevResult result;
    int n = mip.num_cols;
    int m = mip.num_rows;

    // First, try standard Chebyshev formulation
    MIPProblem cheb_mip;

    cheb_mip.num_rows = m;
    cheb_mip.num_cols = n + 1;  // Extra variable for radius r

    // Objective: maximize r (minimize -r)
    cheb_mip.c.assign(n + 1, 0.0);
    cheb_mip.c[n] = -1.0;
    cheb_mip.obj_offset = 0.0;

    // Bounds: original variables keep their bounds, r >= 0
    cheb_mip.lb = mip.lb;
    cheb_mip.ub = mip.ub;
    cheb_mip.lb.push_back(0.0);  // r >= 0
    cheb_mip.ub.push_back(1e20);

    cheb_mip.vartype.assign(n + 1, VarType::CONTINUOUS);
    cheb_mip.b = mip.b;
    cheb_mip.coo_row.clear();
    cheb_mip.coo_col.clear();
    cheb_mip.coo_val.clear();

    // Build constraints: a_i^T x + ||a_i|| * r <= b_i
    for (int i = 0; i < m; i++) {
        double norm_ai = 0.0;
        for (int p = mip.csr_row_ptr[i]; p < mip.csr_row_ptr[i + 1]; p++) {
            double val = mip.csr_val[p];
            norm_ai += val * val;

            cheb_mip.coo_row.push_back(i);
            cheb_mip.coo_col.push_back(mip.csr_col_idx[p]);
            cheb_mip.coo_val.push_back(val);
        }
        norm_ai = std::sqrt(norm_ai);

        if (norm_ai > 0.0) {
            cheb_mip.coo_row.push_back(i);
            cheb_mip.coo_col.push_back(n);  // r column
            cheb_mip.coo_val.push_back(norm_ai);
        }
    }

    cheb_mip.finalize();

    // Solve the Chebyshev LP
    LPRelaxation lp(cheb_mip);
    if (!lp.solve()) {
        result.feasible = false;
        return result;
    }

    // Extract solution
    result.x_center.assign(n);
    for (int j = 0; j < n; j++) {
        result.x_center[j] = lp.x[j];
    }
    result.radius = lp.x[n];
    result.relaxed_radius = result.radius;
    result.feasible = true;

    // Count active constraints
    result.num_active = 0;
    for (int i = 0; i < m; i++) {
        double lhs = 0.0;
        for (int p = mip.csr_row_ptr[i]; p < mip.csr_row_ptr[i + 1]; p++) {
            lhs += mip.csr_val[p] * result.x_center[mip.csr_col_idx[p]];
        }
        if (std::abs(mip.b[i] - lhs) < 1e-6) {
            result.num_active++;
        }
    }

    // Check if polyhedron is lower-dimensional
    result.is_lower_dimensional = (result.radius < 1e-8);

    // If lower-dimensional and fallback requested, use analytic center
    if (result.is_lower_dimensional && fallback_to_analytic) {
        AnalyticCenterResult ac = compute_analytic_center(mip);
        if (ac.feasible) {
            result.x_center = ac.x_center;
            result.radius = 0.0;
            result.relaxed_radius = ac.min_slack;  // Use min slack as "radius" proxy
        }
    }

    return result;
}

RelaxedChebyshevResult compute_projected_chebyshev_center(
    const MIPProblem& mip,
    const std::vector<int>& free_vars,
    double eps
) {
    RelaxedChebyshevResult result;
    int n = mip.num_cols;
    int m = mip.num_rows;

    if (free_vars.empty()) {
        // Fall back to standard Chebyshev
        return compute_relaxed_chebyshev_center(mip);
    }

    int n_free = free_vars.size();

    // Build a reduced problem with only free variables
    MIPProblem reduced;
    reduced.num_cols = n_free + 1;  // +1 for radius
    reduced.num_rows = m;

    // Map from reduced index to original index
    std::vector<int> var_map(n_free);
    for (int i = 0; i < n_free; i++) {
        var_map[i] = free_vars[i];
    }

    // Objective: maximize r
    reduced.c.assign(n_free + 1, 0.0);
    reduced.c[n_free] = -1.0;
    reduced.obj_offset = 0.0;

    // Bounds for free variables
    reduced.lb.resize(n_free);
    reduced.ub.resize(n_free);
    for (int i = 0; i < n_free; i++) {
        reduced.lb[i] = mip.lb[free_vars[i]];
        reduced.ub[i] = mip.ub[free_vars[i]];
    }
    reduced.lb.push_back(0.0);  // r >= 0
    reduced.ub.push_back(1e20);

    reduced.vartype.assign(n_free + 1, VarType::CONTINUOUS);
    reduced.b = mip.b;

    // Build constraints in reduced space
    reduced.coo_row.clear();
    reduced.coo_col.clear();
    reduced.coo_val.clear();

    for (int i = 0; i < m; i++) {
        double norm_ai = 0.0;
        for (int p = mip.csr_row_ptr[i]; p < mip.csr_row_ptr[i + 1]; p++) {
            int col = mip.csr_col_idx[p];
            double val = mip.csr_val[p];

            // Find if this variable is in free_vars
            auto it = std::find(free_vars.begin(), free_vars.end(), col);
            if (it != free_vars.end()) {
                int reduced_idx = std::distance(free_vars.begin(), it);
                norm_ai += val * val;

                reduced.coo_row.push_back(i);
                reduced.coo_col.push_back(reduced_idx);
                reduced.coo_val.push_back(val);
            }
        }
        norm_ai = std::sqrt(norm_ai);

        if (norm_ai > 0.0) {
            reduced.coo_row.push_back(i);
            reduced.coo_col.push_back(n_free);  // r column
            reduced.coo_val.push_back(norm_ai);
        }
    }

    reduced.finalize();

    // Solve reduced problem
    LPRelaxation lp(reduced);
    if (!lp.solve()) {
        result.feasible = false;
        return result;
    }

    // Map solution back to full space
    result.x_center.assign(n, 0.0);
    for (int i = 0; i < n_free; i++) {
        result.x_center[free_vars[i]] = lp.x[i];
    }
    result.radius = lp.x[n_free];
    result.relaxed_radius = result.radius;
    result.feasible = true;

    // Count active constraints
    result.num_active = 0;
    for (int i = 0; i < m; i++) {
        double lhs = 0.0;
        for (int p = mip.csr_row_ptr[i]; p < mip.csr_row_ptr[i + 1]; p++) {
            lhs += mip.csr_val[p] * result.x_center[mip.csr_col_idx[p]];
        }
        if (std::abs(mip.b[i] - lhs) < 1e-6) {
            result.num_active++;
        }
    }

    result.is_lower_dimensional = (result.radius < eps);

    return result;
}

RelaxedChebyshevResult compute_relative_interior_point(
    const MIPProblem& mip,
    double active_tol
) {
    RelaxedChebyshevResult result;
    int n = mip.num_cols;
    int m = mip.num_rows;

    // Step 1: Try standard Chebyshev
    auto [cheb, cheb_mip] = compute_chebyshev_center(mip);

    if (cheb.feasible && cheb.radius > active_tol) {
        // Found a proper interior point
        result.x_center = cheb.x_center;
        result.radius = cheb.radius;
        result.relaxed_radius = cheb.radius;
        result.feasible = true;
        return result;
    }

    // Step 2: Radius is zero - we're on a lower-dimensional face
    // Find which constraints are active at the Chebyshev center
    std::vector<bool> is_active(m, false);
    std::vector<double> slack(m);

    for (int i = 0; i < m; i++) {
        double lhs = 0.0;
        for (int p = mip.csr_row_ptr[i]; p < mip.csr_row_ptr[i + 1]; p++) {
            lhs += mip.csr_val[p] * cheb.x_center[mip.csr_col_idx[p]];
        }
        slack[i] = mip.b[i] - lhs;
        is_active[i] = (slack[i] < active_tol);
    }

    // Step 3: Solve secondary LP to maximize distance from active constraints
    // min  sum of slacks for active constraints (to push away from them)
    // s.t. Ax <= b (all original constraints)

    // Create a new objective that minimizes the sum of active constraint slacks
    // This is equivalent to: min sum_i (b_i - a_i^T x) for active i
    //                      = sum_i b_i - sum_i a_i^T x
    //                      = const - (sum_i a_i)^T x
    // So we minimize -(sum_i a_i)^T x, i.e., maximize (sum_i a_i)^T x

    MIPProblem interior_mip;
    interior_mip.num_rows = m;
    interior_mip.num_cols = n;

    // New objective: maximize sum of coefficients for active constraints
    interior_mip.c.assign(n, 0.0);
    for (int i = 0; i < m; i++) {
        if (is_active[i]) {
            for (int p = mip.csr_row_ptr[i]; p < mip.csr_row_ptr[i + 1]; p++) {
                interior_mip.c[mip.csr_col_idx[p]] -= mip.csr_val[p];
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

    // Solve
    LPRelaxation lp(interior_mip);
    if (!lp.solve()) {
        result.feasible = false;
        return result;
    }

    result.x_center = lp.x;
    result.radius = 0.0;  // Still on the face

    // Compute new slacks
    result.num_active = 0;
    double min_slack = 1e100;
    for (int i = 0; i < m; i++) {
        double lhs = 0.0;
        for (int p = mip.csr_row_ptr[i]; p < mip.csr_row_ptr[i + 1]; p++) {
            lhs += mip.csr_val[p] * result.x_center[mip.csr_col_idx[p]];
        }
        slack[i] = mip.b[i] - lhs;
        if (slack[i] < active_tol) {
            result.num_active++;
        }
        min_slack = std::min(min_slack, slack[i]);
    }

    result.relaxed_radius = min_slack;
    result.feasible = true;
    result.is_lower_dimensional = true;

    return result;
}
