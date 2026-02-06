#include "feasibility_pump.h"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <iostream>

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

    residuals[i] = ax - b[i];
}

__global__ void projection_step_kernel(
    int num_rows,
    const int* csr_row_ptr,
    const int* csr_col_idx,
    const double* csr_val,
    const double* residuals,
    double step,
    double* x
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rows) return;

    if (residuals[i] <= 0.0) return;

    double r = residuals[i];
    for (int k = csr_row_ptr[i]; k < csr_row_ptr[i + 1]; ++k) {
        int j = csr_col_idx[k];
        atomicAdd(&x[j], -step * csr_val[k] * r);
    }
}

/* ================= CLASS IMPLEMENTATION ================= */

FeasibilityPump::FeasibilityPump(const MIPProblem& p)
    : prob(p)
{
    x.resize(prob.num_cols);
    x_lp.resize(prob.num_cols);
    residuals.resize(prob.num_rows);

    cudaMalloc(&d_x, prob.num_cols * sizeof(double));
    cudaMalloc(&d_residuals, prob.num_rows * sizeof(double));
    cudaMalloc(&d_b, prob.num_rows * sizeof(double));

    cudaMalloc(&d_vartype, prob.num_cols * sizeof(uint8_t));

    cudaMalloc(&d_csr_row_ptr, (prob.num_rows + 1) * sizeof(int));
    cudaMalloc(&d_csr_col_idx, prob.csr_col_idx.size() * sizeof(int));
    cudaMalloc(&d_csr_val, prob.csr_val.size() * sizeof(double));

    cudaMemcpy(d_b, prob.b.data(),
               prob.num_rows * sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_vartype, prob.vartype.data(),
               prob.num_cols * sizeof(uint8_t), cudaMemcpyHostToDevice);

    cudaMemcpy(d_csr_row_ptr, prob.csr_row_ptr.data(),
               (prob.num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_csr_col_idx, prob.csr_col_idx.data(),
               prob.csr_col_idx.size() * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_csr_val, prob.csr_val.data(),
               prob.csr_val.size() * sizeof(double), cudaMemcpyHostToDevice);
}

FeasibilityPump::~FeasibilityPump() {
    cudaFree(d_x);
    cudaFree(d_residuals);
    cudaFree(d_b);
    cudaFree(d_vartype);
    cudaFree(d_csr_row_ptr);
    cudaFree(d_csr_col_idx);
    cudaFree(d_csr_val);
}

/* ================= CORE METHODS ================= */

void FeasibilityPump::solve_lp() {
    LPRelaxation lp;
    lp.build_from_mip(prob);
    lp.solve();
    x_lp = lp.x;
}

void FeasibilityPump::round_x() {
    for (int j = 0; j < prob.num_cols; ++j) {
        if (prob.vartype[j] == VarType::BINARY)
            x[j] = (x_lp[j] < 0.5) ? 0.0 : 1.0;
        else if (prob.vartype[j] == VarType::INTEGER)
            x[j] = std::round(x_lp[j]);
        else
            x[j] = x_lp[j];
    }
}

bool FeasibilityPump::check_feasibility() {
    for (int i = 0; i < prob.num_rows; ++i)
        if (residuals[i] > 0.0)
            return false;
    return prob.check_feasible(x);
}

/* ================= MAIN LOOP ================= */

FPSolution FeasibilityPump::run(const FeasibilityPumpParams& params) {
    int block = 256;
    int grid = (prob.num_rows + block - 1) / block;

    for (int r = 0; r < params.max_restarts; ++r) {
        solve_lp();
        round_x();

        for (int it = 0; it < params.max_iters; ++it) {
            cudaMemcpy(d_x, x.data(),
                       prob.num_cols * sizeof(double), cudaMemcpyHostToDevice);

            compute_residuals_kernel<<<grid, block>>>(
                prob.num_rows, d_csr_row_ptr, d_csr_col_idx,
                d_csr_val, d_x, d_b, d_residuals
            );

            projection_step_kernel<<<grid, block>>>(
                prob.num_rows, d_csr_row_ptr, d_csr_col_idx,
                d_csr_val, d_residuals, params.step_size, d_x
            );

            cudaMemcpy(x.data(), d_x,
                       prob.num_cols * sizeof(double), cudaMemcpyDeviceToHost);

            round_x();

            cudaMemcpy(d_x, x.data(),
                       prob.num_cols * sizeof(double), cudaMemcpyHostToDevice);

            compute_residuals_kernel<<<grid, block>>>(
                prob.num_rows, d_csr_row_ptr, d_csr_col_idx,
                d_csr_val, d_x, d_b, d_residuals
            );

            cudaMemcpy(residuals.data(), d_residuals,
                       prob.num_rows * sizeof(double), cudaMemcpyDeviceToHost);

            if (check_feasibility()) {
                FPSolution sol;
                sol.feasible = true;
                sol.x = x;
                return sol;
            }
        }
    }

    FPSolution sol;
    sol.feasible = false;
    return sol;
}

