#include "analytic_center.h"
#include "lp_relaxation.h"
#include <cmath>
#include <algorithm>
#include <iostream>

/*
 * Analytic Center Computation via Damped Newton Method
 *
 * The analytic center minimizes the logarithmic barrier:
 *   f(x) = -sum_{i=1}^{m} log(b_i - a_i^T x)
 *
 * Gradient: ∇f(x) = A^T * (1./s)  where s = b - Ax (slacks)
 * Hessian:  ∇²f(x) = A^T * diag(1./s²) * A
 *
 * Newton step: Δx = -∇²f(x)^{-1} * ∇f(x)
 *
 * For lower-dimensional polyhedra, the Hessian becomes ill-conditioned.
 * We add regularization to handle this case.
 */

// Compute Newton direction with regularization for rank-deficient case
static bool compute_newton_direction(
    const MIPProblem& mip,
    const std::vector<double>& x,
    std::vector<double>& delta,
    double regularization = 1e-8
) {
    int n = mip.num_cols;
    int m = mip.num_rows;

    // Compute slacks s = b - Ax
    std::vector<double> slacks(m);
    for (int i = 0; i < m; i++) {
        double ax = 0.0;
        for (int p = mip.csr_row_ptr[i]; p < mip.csr_row_ptr[i + 1]; p++) {
            ax += mip.csr_val[p] * x[mip.csr_col_idx[p]];
        }
        slacks[i] = mip.b[i] - ax;
        if (slacks[i] <= 0) return false;  // Not strictly feasible
    }

    // Compute gradient: g = A^T * (1./s)
    // Using CSC for efficient A^T * v operation
    std::vector<double> grad(n, 0.0);
    for (int j = 0; j < n; j++) {
        for (int p = mip.csc_col_ptr[j]; p < mip.csc_col_ptr[j + 1]; p++) {
            int row = mip.csc_row_idx[p];
            double val = mip.csc_val[p];
            grad[j] += val / slacks[row];
        }
    }

    // Compute Hessian: H = A^T * diag(1./s²) * A
    // We'll solve (H + reg*I) * delta = -grad using conjugate gradient
    // For efficiency, we compute H * v on the fly

    // Initialize delta = -grad
    delta = grad;
    for (int j = 0; j < n; j++) {
        delta[j] = -delta[j];
    }

    // Solve using conjugate gradient with regularization
    // (H + λI) * d = -g
    const int cg_max_iter = 2 * n;
    const double cg_tol = 1e-10;

    std::vector<double> r(n);  // residual
    std::vector<double> p(n);  // search direction
    std::vector<double> Hdp(n);  // H * d_p

    // r = -g - (H + λI) * delta = -g - H*delta - λ*delta
    // But delta = -g initially, so r = -g - H*(-g) - λ*(-g) = -g + H*g + λ*g

    // First compute H * grad
    // H * v = A^T * diag(1/s²) * A * v
    std::vector<double> Av(m, 0.0);
    for (int i = 0; i < m; i++) {
        for (int p = mip.csr_row_ptr[i]; p < mip.csr_row_ptr[i + 1]; p++) {
            Av[i] += mip.csr_val[p] * grad[mip.csr_col_idx[p]];
        }
    }

    std::vector<double> AT_D_Av(n, 0.0);
    for (int j = 0; j < n; j++) {
        for (int p = mip.csc_col_ptr[j]; p < mip.csc_col_ptr[j + 1]; p++) {
            int row = mip.csc_row_idx[p];
            double val = mip.csc_val[p];
            AT_D_Av[j] += val * Av[row] / (slacks[row] * slacks[row]);
        }
    }

    // r = -grad + AT_D_Av + regularization * grad
    for (int j = 0; j < n; j++) {
        r[j] = -grad[j] + AT_D_Av[j] + regularization * grad[j];
    }

    // Initial search direction
    double rs_old = 0.0;
    for (int j = 0; j < n; j++) {
        p[j] = r[j];
        rs_old += r[j] * r[j];
    }

    // CG iterations
    for (int cg_iter = 0; cg_iter < cg_max_iter; cg_iter++) {
        if (rs_old < cg_tol * cg_tol) break;

        // Compute H * p
        for (int i = 0; i < m; i++) {
            Av[i] = 0.0;
            for (int p = mip.csr_row_ptr[i]; p < mip.csr_row_ptr[i + 1]; p++) {
                Av[i] += mip.csr_val[p] * p[mip.csr_col_idx[p]];
            }
        }

        for (int j = 0; j < n; j++) {
            Hdp[j] = 0.0;
            for (int p = mip.csc_col_ptr[j]; p < mip.csc_col_ptr[j + 1]; p++) {
                int row = mip.csc_row_idx[p];
                double val = mip.csc_val[p];
                Hdp[j] += val * Av[row] / (slacks[row] * slacks[row]);
            }
            Hdp[j] += regularization * p[j];  // Regularization
        }

        // alpha = rs_old / (p^T * H * p)
        double pTHp = 0.0;
        for (int j = 0; j < n; j++) {
            pTHp += p[j] * Hdp[j];
        }

        if (std::abs(pTHp) < 1e-15) break;  // Hessian is singular

        double alpha = rs_old / pTHp;

        // Update delta and residual
        for (int j = 0; j < n; j++) {
            delta[j] += alpha * p[j];
            r[j] -= alpha * Hdp[j];
        }

        double rs_new = 0.0;
        for (int j = 0; j < n; j++) {
            rs_new += r[j] * r[j];
        }

        if (rs_new < cg_tol * cg_tol) break;

        // Update search direction
        double beta = rs_new / rs_old;
        for (int j = 0; j < n; j++) {
            p[j] = r[j] + beta * p[j];
        }
        rs_old = rs_new;
    }

    return true;
}

// Check if x is strictly feasible (all slacks > 0)
static bool is_strictly_feasible(
    const MIPProblem& mip,
    const std::vector<double>& x,
    double min_slack = 1e-6
) {
    for (int i = 0; i < mip.num_rows; i++) {
        double ax = 0.0;
        for (int p = mip.csr_row_ptr[i]; p < mip.csr_row_ptr[i + 1]; p++) {
            ax += mip.csr_val[p] * x[mip.csr_col_idx[p]];
        }
        if (mip.b[i] - ax < min_slack) {
            return false;
        }
    }

    // Also check bounds
    for (int j = 0; j < mip.num_cols; j++) {
        if (x[j] < mip.lb[j] + min_slack || x[j] > mip.ub[j] - min_slack) {
            return false;
        }
    }

    return true;
}

// Compute minimum slack
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

AnalyticCenterResult compute_analytic_center(
    const MIPProblem& mip,
    int max_iter,
    double tol,
    double backtrack_alpha,
    double backtrack_beta
) {
    AnalyticCenterResult result;
    int n = mip.num_cols;
    int m = mip.num_rows;

    // Step 1: Find a strictly feasible starting point
    // Solve LP with artificial slack to get interior point
    LPRelaxation lp(mip);
    if (!lp.solve()) {
        // LP infeasible
        result.feasible = false;
        return result;
    }

    // Check if LP solution is strictly feasible
    // If not, we need to find an interior point
    std::vector<double> x = lp.x;

    // If the solution is on the boundary, move slightly inside
    double min_slack = compute_min_slack(mip, x);
    if (min_slack < 1e-6) {
        // Try to find a better starting point by solving a different LP
        // that maximizes the minimum slack (Chebyshev approach)
        // For now, just use a small perturbation
        std::cerr << "LP solution on boundary, attempting perturbation...\n";

        // Perturb towards center of bounds
        for (int j = 0; j < n; j++) {
            double mid = 0.5 * (mip.lb[j] + mip.ub[j]);
            if (std::isfinite(mip.ub[j])) {
                x[j] = 0.99 * x[j] + 0.01 * mid;
            }
        }

        // Check if perturbation helped
        min_slack = compute_min_slack(mip, x);
        if (min_slack < 1e-8) {
            // Still on boundary - use relaxed approach
            // Add small positive slack to all variables
            for (int j = 0; j < n; j++) {
                x[j] += 1e-4;
            }
        }
    }

    // Step 2: Run damped Newton method
    double lambda = 1e-8;  // Regularization parameter
    double lambda_decay = 0.5;  // How fast to reduce regularization

    for (int iter = 0; iter < max_iter; iter++) {
        result.iterations = iter + 1;

        // Compute Newton direction
        std::vector<double> delta;
        if (!compute_newton_direction(mip, x, delta, lambda)) {
            // Newton direction computation failed
            // Reduce regularization and retry
            lambda *= 10.0;
            if (lambda > 1e-4) {
                // Too much regularization - give up
                break;
            }
            continue;
        }

        // Compute Newton decrement to check convergence
        // lambda_newton = sqrt(delta^T * H * delta)
        // For simplicity, use ||delta|| as proxy
        double delta_norm = 0.0;
        for (int j = 0; j < n; j++) {
            delta_norm += delta[j] * delta[j];
        }
        delta_norm = std::sqrt(delta_norm);

        if (delta_norm < tol) {
            // Converged
            break;
        }

        // Line search: find step size t such that x + t*delta is feasible
        double t = 1.0;
        double max_t = 1.0;

        // Find maximum step size that keeps us feasible
        for (int i = 0; i < m; i++) {
            double aTd = 0.0;
            for (int p = mip.csr_row_ptr[i]; p < mip.csr_row_ptr[i + 1]; p++) {
                aTd += mip.csr_val[p] * delta[mip.csr_col_idx[p]];
            }

            if (aTd > 0) {
                double slack = mip.b[i];
                for (int p = mip.csr_row_ptr[i]; p < mip.csr_row_ptr[i + 1]; p++) {
                    slack -= mip.csr_val[p] * x[mip.csr_col_idx[p]];
                }
                double t_i = 0.99 * slack / aTd;  // 99% of max to stay strictly inside
                max_t = std::min(max_t, t_i);
            }
        }

        // Also respect variable bounds
        for (int j = 0; j < n; j++) {
            if (delta[j] > 0 && std::isfinite(mip.ub[j])) {
                double t_j = 0.99 * (mip.ub[j] - x[j]) / delta[j];
                max_t = std::min(max_t, t_j);
            }
            if (delta[j] < 0 && std::isfinite(mip.lb[j])) {
                double t_j = 0.99 * (x[j] - mip.lb[j]) / (-delta[j]);
                max_t = std::min(max_t, t_j);
            }
        }

        t = std::min(1.0, max_t);

        // Backtracking line search for sufficient decrease
        // (We use a simple version - could be more sophisticated)
        for (int ls_iter = 0; ls_iter < 20; ls_iter++) {
            std::vector<double> x_new(n);
            for (int j = 0; j < n; j++) {
                x_new[j] = x[j] + t * delta[j];
            }

            if (is_strictly_feasible(mip, x_new, 1e-10)) {
                x = x_new;
                break;
            }

            t *= backtrack_beta;
            if (t < 1e-12) {
                // Step size too small - reduce regularization instead
                lambda *= 2.0;
                break;
            }
        }

        // Reduce regularization as we converge
        if (iter % 5 == 0 && lambda > 1e-12) {
            lambda *= lambda_decay;
        }
    }

    // Store result
    result.x_center = x;
    result.min_slack = compute_min_slack(mip, x);
    result.feasible = true;

    return result;
}

AnalyticCenterResult compute_analytic_center_from_point(
    const MIPProblem& mip,
    const std::vector<double>& x0,
    int max_iter,
    double tol
) {
    // Validate starting point
    if ((int)x0.size() != mip.num_cols) {
        AnalyticCenterResult result;
        result.feasible = false;
        return result;
    }

    // Check if x0 is strictly feasible
    if (!is_strictly_feasible(mip, x0, 1e-8)) {
        // Try to find a feasible point first
        return compute_analytic_center(mip, max_iter, tol);
    }

    // Run Newton method from x0
    int n = mip.num_cols;
    int m = mip.num_rows;

    std::vector<double> x = x0;
    double lambda = 1e-8;

    AnalyticCenterResult result;

    for (int iter = 0; iter < max_iter; iter++) {
        result.iterations = iter + 1;

        std::vector<double> delta;
        if (!compute_newton_direction(mip, x, delta, lambda)) {
            lambda *= 10.0;
            if (lambda > 1e-4) break;
            continue;
        }

        double delta_norm = 0.0;
        for (int j = 0; j < n; j++) {
            delta_norm += delta[j] * delta[j];
        }
        delta_norm = std::sqrt(delta_norm);

        if (delta_norm < tol) break;

        // Simple step
        double t = 0.5;  // Conservative step
        for (int j = 0; j < n; j++) {
            x[j] += t * delta[j];
        }

        if (iter % 5 == 0 && lambda > 1e-12) {
            lambda *= 0.5;
        }
    }

    result.x_center = x;
    result.min_slack = compute_min_slack(mip, x);
    result.feasible = true;

    return result;
}
