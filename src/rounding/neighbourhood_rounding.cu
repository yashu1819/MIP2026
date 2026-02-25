// #include <cuda_runtime.h>
// #include <cmath>
// #include <limits>

// #include "mip_problem.h"
// #include "neighbourhood_rounding.h"
// #include "solution.h"
// #include <vector>
// #include <algorithm>


// __global__ void neighbourhood_rounding_kernel(
//     int num_rows,
//     int num_cols,

//     const double* x_lp,          // LP solution
//     const int* vartype,          // VarType enum as int
//     const int* csr_row_ptr,
//     const int* csr_col_idx,
//     const double* csr_val,
//     const double* b,

//     double* candidates,          // [K][num_cols]
//     int* free_mask,
//     double* violation_scores,    // [K]
//     double eps_base
// ) {
//     int k = blockIdx.x * blockDim.x + threadIdx.x;
//     if (k >= gridDim.x * blockDim.x) return;

//     double eps = eps_base * (1.0 + 0.1 * k);  // deterministic variation

//     // --- Step 1: rounding + fixing ---
//     for (int i = 0; i < num_cols; i++) {
//         double xi = x_lp[i];
//         double ri = round(xi);

//         if (vartype[i] == 0) { // continuous
//             candidates[k * num_cols + i] = xi;
//         } else {
//             // if (fabs(xi - ri) <= eps) {
//             //     // FIX variable
//             //     candidates[k * num_cols + i] = ri;
//             // } else {
//             //     // FREE variable → nearest integer
//             //     candidates[k * num_cols + i] = ri;
//             // }
//             bool is_free = (fabs(xi - ri) > eps);

//             candidates[k * num_cols + i] = ri;
//             free_mask[k * num_cols + i] = is_free ? 1 : 0;

//         }
//     }

//     // --- Step 2: constraint violation evaluation ---
//     double total_violation = 0.0;

//     for (int row = 0; row < num_rows; row++) {
//         double sum = 0.0;
//         for (int idx = csr_row_ptr[row]; idx < csr_row_ptr[row + 1]; idx++) {
//             int col = csr_col_idx[idx];
//             sum += csr_val[idx] * candidates[k * num_cols + col];
//         }
//         double viol = sum - b[row];
//         if (viol > 0.0)
//             total_violation += viol;
//     }

//     violation_scores[k] = total_violation;
// }


// Solution run_gpu_neighbourhood_rounding(
//     const MIPProblem& mip,
//     const std::vector<double>& x_lp,
//     int K = 1024
// ) {
//     Solution best;

//     int n = mip.num_cols;
//     int m = mip.num_rows;

//     // --- Allocate device memory ---
//     double *d_xlp, *d_b, *d_val, *d_candidates, *d_scores;
//     int *d_vartype, *d_rowptr, *d_colidx, *d_free_mask;

//     cudaMalloc(&d_xlp, n * sizeof(double));
//     cudaMalloc(&d_vartype, n * sizeof(int));
//     cudaMalloc(&d_rowptr, (m + 1) * sizeof(int));
//     cudaMalloc(&d_colidx, mip.csr_col_idx.size() * sizeof(int));
//     cudaMalloc(&d_val, mip.csr_val.size() * sizeof(double));
//     cudaMalloc(&d_b, m * sizeof(double));

//     cudaMalloc(&d_candidates, K * n * sizeof(double));
//     cudaMalloc(&d_scores, K * sizeof(double));
//     cudaMalloc(&d_free_mask, K * n * sizeof(int));

//     // --- Copy data ---
//     cudaMemcpy(d_xlp, x_lp.data(), n * sizeof(double), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_b, mip.b.data(), m * sizeof(double), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_rowptr, mip.csr_row_ptr.data(), (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_colidx, mip.csr_col_idx.data(),
//                mip.csr_col_idx.size() * sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_val, mip.csr_val.data(),
//                mip.csr_val.size() * sizeof(double), cudaMemcpyHostToDevice);

//     std::vector<int> vt(n);
//     for (int i = 0; i < n; i++) vt[i] = static_cast<int>(mip.vartype[i]);
//     cudaMemcpy(d_vartype, vt.data(), n * sizeof(int), cudaMemcpyHostToDevice);

//     // --- Launch kernel ---
//     int threads = 256;
//     int blocks = (K + threads - 1) / threads;

//     neighbourhood_rounding_kernel<<<blocks, threads>>>(
//         m, n,
//         d_xlp,
//         d_vartype,
//         d_rowptr,
//         d_colidx,
//         d_val,
//         d_b,
//         d_candidates,
//         d_free_mask,
//         d_scores,
//         1e-3
//     );

//     // --- Copy results back ---
//     std::vector<double> scores(K);
//     std::vector<int> free_mask(K * n);
//     std::vector<double> candidates(K * n);

//     cudaMemcpy(scores.data(), d_scores, K * sizeof(double), cudaMemcpyDeviceToHost);
//     cudaMemcpy(candidates.data(), d_candidates, K * n * sizeof(double), cudaMemcpyDeviceToHost);
//     cudaMemcpy(free_mask.data(), d_free_mask, K * n * sizeof(int), cudaMemcpyDeviceToHost);

//     std::vector<int> locks_up, locks_down;
//     compute_locks(mip, locks_up, locks_down);

//     // // --- Select best candidate ---
//     // int best_k = std::min_element(scores.begin(), scores.end()) - scores.begin();

//     // best.x.assign(
//     //     candidates.begin() + best_k * n,
//     //     candidates.begin() + (best_k + 1) * n
//     // );

//     // best.feasible = mip.check_feasible(best.x);
//     // if (best.feasible) {
//     //     best.obj_value = mip.obj_offset;
//     //     for (int i = 0; i < n; i++)
//     //         best.obj_value += mip.c[i] * best.x[i];
//     // }

//     // Repair Loop 
//     // Try repairing best few candidates
//     for (int trial = 0; trial < std::min(K, 5); trial++) {

//         int idx = trial; // try top few if you want to sort later

//         std::vector<double> x(
//             candidates.begin() + idx * n,
//             candidates.begin() + (idx + 1) * n
//         );

//         std::vector<bool> is_free(n);
//         for (int i = 0; i < n; i++)
//             is_free[i] = (free_mask[idx * n + i] == 1);

//         bool feasible = repair_solution(
//             mip, x,
//             is_free,
//             locks_up,
//             locks_down,
//             5000
//         );

//         if (feasible) {
//             best.x = x;
//             best.feasible = true;

//             best.obj_value = mip.obj_offset;
//             for (int i = 0; i < n; i++)
//                 best.obj_value += mip.c[i] * x[i];

//             break;
//         }
//     }


//     // --- Cleanup ---
//     cudaFree(d_xlp);
//     cudaFree(d_vartype);
//     cudaFree(d_rowptr);
//     cudaFree(d_colidx);
//     cudaFree(d_val);
//     cudaFree(d_b);
//     cudaFree(d_candidates);
//     cudaFree(d_scores);
//     cudaFree(d_free_mask);

//     return best;
// }





#include <cuda_runtime.h>
#include <cmath>
#include <limits>

#include "mip_problem.h"
#include "neighbourhood_rounding.h"
#include "solution.h"
#include <vector>
#include <algorithm>

// ENHANCED: Lock-aware GPU kernel with better memory access
__global__ void neighbourhood_rounding_kernel(
    int num_rows,
    int num_cols,
    int K,

    const double* x_lp,
    const int* vartype,
    const int* csr_row_ptr,
    const int* csr_col_idx,
    const double* csr_val,
    const double* b,
    
    // NEW: Add lock information for intelligent rounding
    const int* locks_up,
    const int* locks_down,

    double* candidates,
    int* free_mask,
    double* violation_scores,
    double* best_rounded,    // NEW: Output best lock-based rounding
    double eps_base
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    double eps = eps_base * (1.0 + 0.1 * (k % 10));  // Vary epsilon
    
    double local_violation = 0.0;
    
    // Process each variable
    for (int i = 0; i < num_cols; i++) {
        double xi = x_lp[i];
        int var_type = vartype[i];
        
        if (var_type == 0) { // CONTINUOUS
            candidates[k * num_cols + i] = xi;
            free_mask[k * num_cols + i] = 0;
        } else {
            double floor_val = floor(xi);
            double ceil_val = ceil(xi);
            double nearest = round(xi);
            double rounded_val;
            bool is_free = false;
            
            // Determine rounding based on locks (if k == 0, use best; else explore)
            if (k == 0) {
                // Best lock-based rounding
                if (fabs(xi - nearest) <= eps) {
                    rounded_val = nearest;
                } else if (locks_up[i] < locks_down[i]) {
                    rounded_val = ceil_val;   // Round up (safer)
                } else if (locks_down[i] < locks_up[i]) {
                    rounded_val = floor_val;  // Round down (safer)
                } else {
                    rounded_val = nearest;
                }
            } else {
                // Exploration: vary rounding for near-half-integer values
                double frac = xi - floor_val;
                if (fabs(frac - 0.5) < 0.1) {
                    // Near tie: alternate based on k
                    rounded_val = (k % 2 == 0) ? floor_val : ceil_val;
                    is_free = true;
                } else if (fabs(xi - nearest) <= eps) {
                    rounded_val = nearest;
                } else {
                    rounded_val = nearest;
                    is_free = (fabs(xi - nearest) > eps);
                }
            }
            
            candidates[k * num_cols + i] = rounded_val;
            free_mask[k * num_cols + i] = is_free ? 1 : 0;
        }
    }

    __syncthreads();  // Ensure all threads done with rounding

    // Evaluate constraints with coalesced memory access
    for (int row = 0; row < num_rows; row++) {
        double sum = 0.0;
        for (int idx = csr_row_ptr[row]; idx < csr_row_ptr[row + 1]; idx++) {
            int col = csr_col_idx[idx];
            sum += csr_val[idx] * candidates[k * num_cols + col];
        }
        double viol = sum - b[row];
        if (viol > 0.0)
            local_violation += viol;
    }

    violation_scores[k] = local_violation;
    
    // Store best candidate (k=0) separately
    if (k == 0) {
        for (int i = 0; i < num_cols; i++) {
            best_rounded[i] = candidates[i];
        }
    }
}

// ENHANCED: Main function with better candidate selection
Solution run_gpu_neighbourhood_rounding(
    const MIPProblem& mip,
    const std::vector<double>& x_lp,
    int K = 1024
) {
    Solution best;
    best.feasible = false;
    best.obj_value = std::numeric_limits<double>::infinity();

    int n = mip.num_cols;
    int m = mip.num_rows;

    // Compute locks on CPU (could be GPU-accelerated too)
    std::vector<int> locks_up, locks_down;
    compute_locks(mip, locks_up, locks_down);

    // Device allocation
    double *d_xlp, *d_b, *d_val, *d_candidates, *d_scores, *d_best_rounded;
    int *d_vartype, *d_rowptr, *d_colidx, *d_free_mask;
    int *d_locks_up, *d_locks_down;

    cudaMalloc(&d_xlp, n * sizeof(double));
    cudaMalloc(&d_vartype, n * sizeof(int));
    cudaMalloc(&d_rowptr, (m + 1) * sizeof(int));
    cudaMalloc(&d_colidx, mip.csr_col_idx.size() * sizeof(int));
    cudaMalloc(&d_val, mip.csr_val.size() * sizeof(double));
    cudaMalloc(&d_b, m * sizeof(double));
    cudaMalloc(&d_locks_up, n * sizeof(int));
    cudaMalloc(&d_locks_down, n * sizeof(int));

    cudaMalloc(&d_candidates, K * n * sizeof(double));
    cudaMalloc(&d_scores, K * sizeof(double));
    cudaMalloc(&d_free_mask, K * n * sizeof(int));
    cudaMalloc(&d_best_rounded, n * sizeof(double));

    // Copy data
    cudaMemcpy(d_xlp, x_lp.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, mip.b.data(), m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowptr, mip.csr_row_ptr.data(), (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colidx, mip.csr_col_idx.data(),
               mip.csr_col_idx.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, mip.csr_val.data(),
               mip.csr_val.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_locks_up, locks_up.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_locks_down, locks_down.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    std::vector<int> vt(n);
    for (int i = 0; i < n; i++) vt[i] = static_cast<int>(mip.vartype[i]);
    cudaMemcpy(d_vartype, vt.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int threads = 256;
    int blocks = (K + threads - 1) / threads;

    neighbourhood_rounding_kernel<<<blocks, threads>>>(
        m, n, K,
        d_xlp,
        d_vartype,
        d_rowptr,
        d_colidx,
        d_val,
        d_b,
        d_locks_up,
        d_locks_down,
        d_candidates,
        d_free_mask,
        d_scores,
        d_best_rounded,
        1e-3
    );

    // Get results
    std::vector<double> scores(K);
    std::vector<int> free_mask(K * n);
    std::vector<double> candidates(K * n);

    cudaMemcpy(scores.data(), d_scores, K * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(candidates.data(), d_candidates, K * n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(free_mask.data(), d_free_mask, K * n * sizeof(int), cudaMemcpyDeviceToHost);

    // ENHANCED: Sort candidates by violation score
    std::vector<int> indices(K);
    for (int i = 0; i < K; i++) indices[i] = i;
    std::sort(indices.begin(), indices.end(),
              [&](int a, int b) { return scores[a] < scores[b]; });

    // Try repairing top candidates
    int max_repair_attempts = std::min(K, 20);  // Try more candidates
    for (int attempt = 0; attempt < max_repair_attempts; attempt++) {
        int idx = indices[attempt];
        
        std::vector<double> x(
            candidates.begin() + idx * n,
            candidates.begin() + (idx + 1) * n
        );

        std::vector<bool> is_free(n);
        for (int i = 0; i < n; i++)
            is_free[i] = (free_mask[idx * n + i] == 1);

        // ENHANCED: If low violation, try without repair first
        if (scores[idx] < 1e-6) {
            if (mip.check_feasible(x)) {
                best.x = x;
                best.feasible = true;
                // Compute objective...
                break;
            }
        }

        bool feasible = repair_solution(
            mip, x, is_free, locks_up, locks_down, 10000
        );

        if (feasible) {
            best.x = x;
            best.feasible = true;
            best.obj_value = mip.obj_offset;
            for (int i = 0; i < n; i++)
                best.obj_value += mip.c[i] * x[i];
            break;
        }
    }

    // Cleanup
    cudaFree(d_xlp); cudaFree(d_vartype); cudaFree(d_rowptr);
    cudaFree(d_colidx); cudaFree(d_val); cudaFree(d_b);
    cudaFree(d_locks_up); cudaFree(d_locks_down);
    cudaFree(d_candidates); cudaFree(d_scores); 
    cudaFree(d_free_mask); cudaFree(d_best_rounded);

    return best;
}