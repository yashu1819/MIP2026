#include "feasibility_jump.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>

/* ================= CUDA KERNELS ================= */

// Standard parallel reduction within a block
__device__ double blockReduceSum(double val) {
    static __shared__ double shared[32]; 
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0;
    if (wid == 0) {
        for (int offset = 16; offset > 0; offset /= 2)
            val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

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
    for (int k = csr_row_ptr[i]; k < csr_row_ptr[i + 1]; ++k) {
        ax += csr_val[k] * x[csr_col_idx[k]];
    }
    residuals[i] = b[i] - ax;
}

__global__ void evaluate_scores_kernel(
    int num_cols,
    const int* csc_col_ptr,
    const int* csc_row_idx,
    const double* csc_val,
    const double* x,
    const double* residuals,
    const double* weights,
    const double* lb,
    const double* ub,
    const uint8_t* vartype,
    double* out_scores,
    double* out_deltas
) {
    int j = blockIdx.x; // Each SM (block) handles 1 variable
    if (j >= num_cols) return;

    int tid = threadIdx.x;
    double current_x = x[j];
    
    // 1. Determine candidate delta (Simplified: flipping for Binary, +/-1 for Integer)
    double delta = 0.0;
    if (vartype[j] == 2) { // BINARY
        delta = (current_x > 0.5) ? -1.0 : 1.0;
    } else { // INTEGER or CONTINUOUS (Testing +/- 1)
        // Here we just test +1 for simplicity; a robust version tests multiple deltas
        delta = (current_x + 1.0 <= ub[j]) ? 1.0 : -1.0;
    }

    double local_improvement = 0.0;

    // 2. Parallelize over constraints variable j belongs to (using CSC)
    for (int k = csc_col_ptr[j] + tid; k < csc_col_ptr[j + 1]; k += blockDim.x) {
        int i = csc_row_idx[k];
        double a_ij = csc_val[k];
        double w = weights[i];
        
        double r_old = residuals[i];
        double r_new = r_old - a_ij * delta;

        // Violation = max(0, -residual) for Ax <= b
        double v_old = (r_old < 0.0) ? -r_old : 0.0;
        double v_new = (r_new < 0.0) ? -r_new : 0.0;

        local_improvement += w * (v_old - v_new);
    }

    // 3. Reduction within the block (SM) to get total improvement for variable j
    double total_improvement = blockReduceSum(local_improvement);

    if (tid == 0) {
        out_scores[j] = total_improvement;
        out_deltas[j] = delta;
    }
}

/* ================= CLASS IMPLEMENTATION ================= */

FeasibilityJump::FeasibilityJump(const MIPProblem& p)
    : prob(p), rng(1234)
{
    // Allocate Host memory
    x.resize(prob.num_cols);
    residuals.resize(prob.num_rows);
    h_scores.resize(prob.num_cols);
    h_deltas.resize(prob.num_cols);

    // Allocate Device memory
    cudaMalloc(&d_x, prob.num_cols * sizeof(double));
    cudaMalloc(&d_residuals, prob.num_rows * sizeof(double));
    cudaMalloc(&d_weights, prob.num_rows * sizeof(double));
    cudaMalloc(&d_b, prob.num_rows * sizeof(double));
    cudaMalloc(&d_lb, prob.num_cols * sizeof(double));
    cudaMalloc(&d_ub, prob.num_cols * sizeof(double));
    cudaMalloc(&d_vartype, prob.num_cols * sizeof(uint8_t));

    // Matrix memory (CSR for residuals, CSC for scoring)
    cudaMalloc(&d_csr_row_ptr, (prob.num_rows + 1) * sizeof(int));
    cudaMalloc(&d_csr_col_idx, prob.csr_col_idx.size() * sizeof(int));
    cudaMalloc(&d_csr_val, prob.csr_val.size() * sizeof(double));
    
    cudaMalloc(&d_csc_col_ptr, (prob.num_cols + 1) * sizeof(int));
    cudaMalloc(&d_csc_row_idx, prob.csc_row_idx.size() * sizeof(int));
    cudaMalloc(&d_csc_val, prob.csc_val.size() * sizeof(double));

    // Scoring result buffers
    cudaMalloc(&d_scores, prob.num_cols * sizeof(double));
    cudaMalloc(&d_deltas, prob.num_cols * sizeof(double));
    weights.resize(prob.num_rows);
    // Initial copies
    cudaMemcpy(d_b, prob.b.data(), prob.num_rows * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lb, prob.lb.data(), prob.num_cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ub, prob.ub.data(), prob.num_cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vartype, prob.vartype.data(), prob.num_cols * sizeof(uint8_t), cudaMemcpyHostToDevice);

    cudaMemcpy(d_csr_row_ptr, prob.csr_row_ptr.data(), (prob.num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_col_idx, prob.csr_col_idx.data(), prob.csr_col_idx.size()* sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr_val, prob.csr_val.data(), prob.csr_val.size() * sizeof(double), cudaMemcpyHostToDevice);
    // Corrected cudaMemcpy calls
    cudaMemcpy(d_csc_col_ptr, prob.csc_col_ptr.data(), (prob.num_cols + 1) * sizeof(int), cudaMemcpyHostToDevice);
   // Fixed: Added prob.csc_row_idx.data() as the source pointer
   cudaMemcpy(d_csc_row_idx, prob.csc_row_idx.data(),prob.csc_row_idx.size() * sizeof(int), cudaMemcpyHostToDevice);
   cudaMemcpy(d_csc_val, prob.csc_val.data(),  prob.csc_val.size() * sizeof(double), cudaMemcpyHostToDevice); 
}

FeasibilityJump::~FeasibilityJump() {
    cudaFree(d_x); cudaFree(d_residuals); cudaFree(d_weights);
    cudaFree(d_b); cudaFree(d_lb); cudaFree(d_ub); cudaFree(d_vartype);
    cudaFree(d_csr_row_ptr); cudaFree(d_csr_col_idx); cudaFree(d_csr_val);
    cudaFree(d_csc_col_ptr); cudaFree(d_csc_row_idx); cudaFree(d_csc_val);
    cudaFree(d_scores); cudaFree(d_deltas);
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
    LPRelaxation lp;
    lp.build_from_mip(prob);
    lp.solve();
    for (int j=0; j<prob.num_cols; j++){
     
        if (prob.vartype[j] == VarType::BINARY)
            x[j] = (lp.x[j] < 0.5) ? 0.0 : 1.0;
        else if (prob.vartype[j] == VarType::INTEGER)
            x[j] = std::round(lp.x[j]);
        else
            x[j] = lp.x[j];
    }

std::fill(weights.begin(), weights.end(), 1.0);
    cudaMemcpy(d_weights, weights.data(), 
               prob.num_rows * sizeof(double), 
               cudaMemcpyHostToDevice);}

Solution FeasibilityJump::run(const FeasibilityJumpParams& params) {
    int block_size = 256;
    int grid_rows = (prob.num_rows + block_size - 1) / block_size;
    int t1= getTime();
    for (int r = 0; r < params.max_restarts; ++r) {
        initialize();

        for (int it = 0; it < params.max_iters; ++it) {
           //0. Check for time limit
	   if (getTime()-t1>300){
	      Solution sol; sol.feasible = false; sol.x = x;
                return sol;

	   }	



            // 1. Sync x to GPU and compute current residuals
            cudaMemcpy(d_x, x.data(), prob.num_cols * sizeof(double), cudaMemcpyHostToDevice);
            compute_residuals_kernel<<<grid_rows, block_size>>>(
                prob.num_rows, d_csr_row_ptr, d_csr_col_idx, d_csr_val, d_x, d_b, d_residuals
            );

            // 2. Evaluate all variables in parallel (Each block handles 1 column)
            evaluate_scores_kernel<<<prob.num_cols, 256>>>(
                prob.num_cols, d_csc_col_ptr, d_csc_row_idx, d_csc_val,
                d_x, d_residuals, d_weights, d_lb, d_ub, (uint8_t*)d_vartype,
                d_scores, d_deltas
            );

            // 3. Find the best improvement (Greedy step)
            cudaMemcpy(h_scores.data(), d_scores, prob.num_cols * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_deltas.data(), d_deltas, prob.num_cols * sizeof(double), cudaMemcpyDeviceToHost);

            int best_j = -1;
            double max_improvement = 1e-9; // Only jump if improvement is positive

            for (int j = 0; j < prob.num_cols; ++j) {
                if (h_scores[j] > max_improvement) {
                    max_improvement = h_scores[j];
                    best_j = j;
                }
            }

            // 4. Update or Increment Weights
            if (best_j != -1) {
                x[best_j] += h_deltas[best_j];
            } else {
                // No improving move found: pull residuals to update weights on CPU (or do on GPU)
                cudaMemcpy(residuals.data(), d_residuals, prob.num_rows * sizeof(double), cudaMemcpyDeviceToHost);
                //bool all_feasible = true;
                for (int i = 0; i < prob.num_rows; ++i) {
                    if (residuals[i] < -params.constr_tol) {
                        weights[i] += 1.0;
              //          all_feasible = false;
                    }
                }
                cudaMemcpy(d_weights, weights.data(), prob.num_rows * sizeof(double), cudaMemcpyHostToDevice);
	    }  
            if ( prob.check_feasible(x)) {
                Solution sol; sol.feasible = true; sol.x = x;
                return sol;
            }
            
        }
    }

    Solution sol; sol.feasible = false;
    return sol;
}
