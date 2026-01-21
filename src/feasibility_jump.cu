#include "feasibility_jump.h"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

/* ================= CUDA KERNELS ================= */

__global__ void compute_residuals_kernel(
    int num_rows,
    const int* csr_row_ptr,
    const int* csr_col_idx,
    const double* csr_val,
    const double* x,
    const double* b,
    double* residuals
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rows) return;

    double ax = 0.0;
    for (int k = csr_row_ptr[i]; k < csr_row_ptr[i + 1]; ++k)
        ax += csr_val[k] * x[csr_col_idx[k]];

    residuals[i] = b[i] - ax;
}



/* ================= CLASS ================= */

FeasibilityJump::FeasibilityJump(const MIPProblem& p)
    : prob(p), rng(1234)
{
    x.resize(prob.num_cols);
    residuals.resize(prob.num_rows);
    weights.assign(prob.num_rows, 1.0);

    cudaMalloc(&d_x, prob.num_cols * sizeof(double));
    cudaMalloc(&d_residuals, prob.num_rows * sizeof(double));
    cudaMalloc(&d_weights, prob.num_rows * sizeof(double));
    cudaMalloc(&d_b, prob.num_rows * sizeof(double));
    cudaMalloc(&d_csr_row_ptr, (prob.num_rows + 1) * sizeof(int));
    cudaMalloc(&d_csr_col_idx, prob.csr_col_idx.size() * sizeof(int));
    cudaMalloc(&d_csr_val, prob.csr_val.size() * sizeof(double));

    cudaMemcpy(d_b, prob.b.data(),
               prob.num_rows * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_row_ptr, prob.csr_row_ptr.data(),
               (prob.num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_col_idx, prob.csr_col_idx.data(),
               prob.csr_col_idx.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_val, prob.csr_val.data(),
               prob.csr_val.size() * sizeof(double), cudaMemcpyHostToDevice);
}

FeasibilityJump::~FeasibilityJump() {
    cudaFree(d_x);
    cudaFree(d_residuals);
    cudaFree(d_weights);
    cudaFree(d_b);
    cudaFree(d_csr_row_ptr);
    cudaFree(d_csr_col_idx);
    cudaFree(d_csr_val);
}

void FeasibilityJump::initialize() {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    for (int j = 0; j < prob.num_cols; ++j) {
        if (prob.vartype[j] == VarType::BINARY)
            x[j] = (U(rng) < 0.5) ? 0.0 : 1.0;
        else if (prob.vartype[j] == VarType::INTEGER)
            x[j] = std::round(prob.lb[j] + U(rng) * (prob.ub[j] - prob.lb[j]));
        else
            x[j] = std::max(prob.lb[j], std::min(0.0, prob.ub[j]));
    }
    std::fill(weights.begin(), weights.end(), 1.0);
}

double FeasibilityJump::compute_weighted_violation_gpu() {

    // copy x
    cudaMemcpy(d_x, x.data(),
               prob.num_cols * sizeof(double),
               cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (prob.num_rows + block - 1) / block;

    // compute residuals = b - Ax
    compute_residuals_kernel<<<grid, block>>>(
        prob.num_rows,
        d_csr_row_ptr,
        d_csr_col_idx,
        d_csr_val,
        d_x,
        d_b,
        d_residuals
    );

    // copy residuals back
    cudaMemcpy(residuals.data(),
               d_residuals,
               prob.num_rows * sizeof(double),
               cudaMemcpyDeviceToHost);

    // CPU weighted violation
    double total = 0.0;
    for (int i = 0; i < prob.num_rows; ++i) {
        if (residuals[i] < 0.0)
            total += weights[i] * (-residuals[i]);
    }

    return total;
}

void FeasibilityJump::update_weights(double tol) {
    for (int i = 0; i < prob.num_rows; ++i)
        if (residuals[i] < -tol)
            weights[i] += 1.0;
}

bool FeasibilityJump::try_jump(int j, double& best_val) {
    double old = x[j];
    double best_x = old;

    auto test = [&](double v) {
        x[j] = std::max(prob.lb[j], std::min(v, prob.ub[j]));
        double val = compute_weighted_violation_gpu();
        if (val < best_val) {
            best_val = val;
            best_x = x[j];
        }
    };

    if (prob.vartype[j] == VarType::BINARY) {
        test(1.0 - old);
    } else {
        test(old + 1.0);
        test(old - 1.0);
    }

    x[j] = best_x;
    return best_x != old;
}

/* ================= MAIN LOOP ================= */

Solution FeasibilityJump::run(const FeasibilityJumpParams& params) {

    for (int r = 0; r < params.max_restarts; ++r) {

        initialize();
        double best = compute_weighted_violation_gpu();

        for (int it = 0; it < params.max_iters; ++it) {

            if (best <= params.constr_tol && prob.check_feasible(x)) {
                Solution sol;
                sol.feasible = true;
                sol.x = x;
                return sol;
            }

            int j = rng() % prob.num_cols;
            bool moved = try_jump(j, best);

            if (!moved)
                update_weights(params.constr_tol);
        }
    }

    Solution sol;
    sol.feasible = false;
    return sol;
}

