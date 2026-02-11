#include <cuda_runtime.h>
#include <cmath>
#include <limits>

#include "mip_problem.h"
#include "solution.h"
#include <vector>
#include <algorithm>


__global__ void neighborhood_rounding_kernel(
    int num_rows,
    int num_cols,

    const double* x_lp,          // LP solution
    const int* vartype,          // VarType enum as int
    const int* csr_row_ptr,
    const int* csr_col_idx,
    const double* csr_val,
    const double* b,

    double* candidates,          // [K][num_cols]
    double* violation_scores,    // [K]
    double eps_base
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= gridDim.x * blockDim.x) return;

    double eps = eps_base * (1.0 + 0.1 * k);  // deterministic variation

    // --- Step 1: rounding + fixing ---
    for (int i = 0; i < num_cols; i++) {
        double xi = x_lp[i];
        double ri = round(xi);

        if (vartype[i] == 0) { // continuous
            candidates[k * num_cols + i] = xi;
        } else {
            if (fabs(xi - ri) <= eps) {
                // FIX variable
                candidates[k * num_cols + i] = ri;
            } else {
                // FREE variable → nearest integer
                candidates[k * num_cols + i] = ri;
            }
        }
    }

    // --- Step 2: constraint violation evaluation ---
    double total_violation = 0.0;

    for (int row = 0; row < num_rows; row++) {
        double sum = 0.0;
        for (int idx = csr_row_ptr[row]; idx < csr_row_ptr[row + 1]; idx++) {
            int col = csr_col_idx[idx];
            sum += csr_val[idx] * candidates[k * num_cols + col];
        }
        double viol = sum - b[row];
        if (viol > 0.0)
            total_violation += viol;
    }

    violation_scores[k] = total_violation;
}


Solution run_gpu_neighborhood_rounding(
    const MIPProblem& mip,
    const std::vector<double>& x_lp,
    int K = 1024
) {
    Solution best;

    int n = mip.num_cols;
    int m = mip.num_rows;

    // --- Allocate device memory ---
    double *d_xlp, *d_b, *d_val, *d_candidates, *d_scores;
    int *d_vartype, *d_rowptr, *d_colidx;

    cudaMalloc(&d_xlp, n * sizeof(double));
    cudaMalloc(&d_vartype, n * sizeof(int));
    cudaMalloc(&d_rowptr, (m + 1) * sizeof(int));
    cudaMalloc(&d_colidx, mip.csr_col_idx.size() * sizeof(int));
    cudaMalloc(&d_val, mip.csr_val.size() * sizeof(double));
    cudaMalloc(&d_b, m * sizeof(double));

    cudaMalloc(&d_candidates, K * n * sizeof(double));
    cudaMalloc(&d_scores, K * sizeof(double));

    // --- Copy data ---
    cudaMemcpy(d_xlp, x_lp.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, mip.b.data(), m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowptr, mip.csr_row_ptr.data(), (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colidx, mip.csr_col_idx.data(),
               mip.csr_col_idx.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, mip.csr_val.data(),
               mip.csr_val.size() * sizeof(double), cudaMemcpyHostToDevice);

    std::vector<int> vt(n);
    for (int i = 0; i < n; i++) vt[i] = static_cast<int>(mip.vartype[i]);
    cudaMemcpy(d_vartype, vt.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    // --- Launch kernel ---
    int threads = 256;
    int blocks = (K + threads - 1) / threads;

    neighborhood_rounding_kernel<<<blocks, threads>>>(
        m, n,
        d_xlp,
        d_vartype,
        d_rowptr,
        d_colidx,
        d_val,
        d_b,
        d_candidates,
        d_scores,
        1e-3
    );

    // --- Copy results back ---
    std::vector<double> scores(K);
    std::vector<double> candidates(K * n);

    cudaMemcpy(scores.data(), d_scores, K * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(candidates.data(), d_candidates, K * n * sizeof(double), cudaMemcpyDeviceToHost);

    // --- Select best candidate ---
    int best_k = std::min_element(scores.begin(), scores.end()) - scores.begin();

    best.x.assign(
        candidates.begin() + best_k * n,
        candidates.begin() + (best_k + 1) * n
    );

    best.feasible = mip.check_feasible(best.x);
    if (best.feasible) {
        best.obj_value = mip.obj_offset;
        for (int i = 0; i < n; i++)
            best.obj_value += mip.c[i] * best.x[i];
    }

    // --- Cleanup ---
    cudaFree(d_xlp);
    cudaFree(d_vartype);
    cudaFree(d_rowptr);
    cudaFree(d_colidx);
    cudaFree(d_val);
    cudaFree(d_b);
    cudaFree(d_candidates);
    cudaFree(d_scores);

    return best;
}

