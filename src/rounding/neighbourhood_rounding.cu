#include <cuda_runtime.h>
#include <cmath>
#include <limits>

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

