#include "parallel_kopt.h"
#include "repair.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>
#include <algorithm>
#include <random>

double getTimeKopt() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double>(duration).count();
}

__device__ double atomicMinDouble(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(fmin(__longlong_as_double(assumed), val)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void evaluate_kopt_kernel(
    int num_cols,
    int num_rows,
    const int* csc_col_ptr,
    const int* csc_row_idx,
    const double* csc_val,
    const double* c,
    const double* x,
    const double* lb,
    const double* ub,
    const uint8_t* vartype,
    const double* act_best,
    const double* b,
    double* out_best_delta_obj,
    int* out_best_i,
    double* out_best_di,
    int* out_best_j,
    double* out_best_dj
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_cols) return;

    double x_i = x[i];
    
    // We only consider Integer/Binary variables for the primary move to keep it discrete
    if (vartype[i] == 0) return; // 0 == CONTINUOUS

    double candidates_di[2];
    int num_candidates = 0;
    
    if (vartype[i] == 2) { // BINARY
        if (x_i > 0.5) candidates_di[num_candidates++] = -1.0;
        else candidates_di[num_candidates++] = 1.0;
    } else { // INTEGER
        if (x_i - 1.0 >= lb[i]) candidates_di[num_candidates++] = -1.0;
        if (x_i + 1.0 <= ub[i]) candidates_di[num_candidates++] = 1.0;
    }

    for (int idx = 0; idx < num_candidates; ++idx) {
        double di = candidates_di[idx];
        double delta_obj_i = c[i] * di;
        
        // Check 1-opt
        bool feasible_1opt = true;
        for (int k = csc_col_ptr[i]; k < csc_col_ptr[i+1]; ++k) {
            int row = csc_row_idx[k];
            double a_ij = csc_val[k];
            if (act_best[row] + a_ij * di > b[row] + 1e-6) {
                feasible_1opt = false;
                break;
            }
        }
        
        if (feasible_1opt && delta_obj_i < -1e-6) {
            double old_best = atomicMinDouble(out_best_delta_obj, delta_obj_i);
            if (delta_obj_i < old_best) {
                *out_best_i = i;
                *out_best_di = di;
                *out_best_j = -1;
                *out_best_dj = 0.0;
            }
        }

        // Try 2-opt (combine with j)
        // Optimization: only search a few j's or just rely on CPU repair for K-opt.
        // For GPU, we'll try a bounded subset of j's to keep kernel fast.
        for (int j = max(0, i - 100); j < min(num_cols, i + 100); ++j) {
            if (i == j || vartype[j] == 0) continue;
            
            double x_j = x[j];
            double dj = (vartype[j] == 2) ? ((x_j > 0.5) ? -1.0 : 1.0) : 1.0; // simplistic dj
            if (x_j + dj > ub[j] || x_j + dj < lb[j]) continue;
            
            double delta_obj_ij = delta_obj_i + c[j] * dj;
            if (delta_obj_ij >= -1e-6) continue;
            
            bool feasible_2opt = true;
            for (int k = csc_col_ptr[i]; k < csc_col_ptr[i+1]; ++k) {
                int row = csc_row_idx[k];
                double a_ij = csc_val[k];
                double total_act = act_best[row] + a_ij * di;
                // Add j's contribution
                for(int pk = csc_col_ptr[j]; pk < csc_col_ptr[j+1]; ++pk) {
                    if (csc_row_idx[pk] == row) {
                        total_act += csc_val[pk] * dj;
                        break;
                    }
                }
                if (total_act > b[row] + 1e-6) { feasible_2opt = false; break; }
            }
            if (!feasible_2opt) continue;
            
            // Check remaining j constraints
            for (int pk = csc_col_ptr[j]; pk < csc_col_ptr[j+1]; ++pk) {
                int row = csc_row_idx[pk];
                // Check if row already checked via i
                bool checked = false;
                for (int k = csc_col_ptr[i]; k < csc_col_ptr[i+1]; ++k) {
                    if (csc_row_idx[k] == row) { checked = true; break; }
                }
                if (!checked) {
                    if (act_best[row] + csc_val[pk] * dj > b[row] + 1e-6) {
                        feasible_2opt = false; break;
                    }
                }
            }
            
            if (feasible_2opt) {
                double old_best = atomicMinDouble(out_best_delta_obj, delta_obj_ij);
                if (delta_obj_ij < old_best) {
                    *out_best_i = i;
                    *out_best_di = di;
                    *out_best_j = j;
                    *out_best_dj = dj;
                }
            }
        }
    }
}

ParallelKOpt::ParallelKOpt(const MIPProblem& p) : prob(p) {
    cudaMalloc(&d_x_best, prob.num_cols * sizeof(double));
    cudaMalloc(&d_act_best, prob.num_rows * sizeof(double));
    cudaMalloc(&d_lb, prob.num_cols * sizeof(double));
    cudaMalloc(&d_ub, prob.num_cols * sizeof(double));
    cudaMalloc(&d_c, prob.num_cols * sizeof(double));
    cudaMalloc(&d_b, prob.num_rows * sizeof(double));
    cudaMalloc(&d_vartype, prob.num_cols * sizeof(uint8_t));

    cudaMalloc(&d_csc_col_ptr, (prob.num_cols + 1) * sizeof(int));
    cudaMalloc(&d_csc_row_idx, prob.csc_row_idx.size() * sizeof(int));
    cudaMalloc(&d_csc_val, prob.csc_val.size() * sizeof(double));

    cudaMemcpy(d_lb, prob.lb.data(), prob.num_cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ub, prob.ub.data(), prob.num_cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, prob.c.data(), prob.num_cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, prob.b.data(), prob.num_rows * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vartype, prob.vartype.data(), prob.num_cols * sizeof(uint8_t), cudaMemcpyHostToDevice);

    cudaMemcpy(d_csc_col_ptr, prob.csc_col_ptr.data(), (prob.num_cols + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csc_row_idx, prob.csc_row_idx.data(), prob.csc_row_idx.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csc_val, prob.csc_val.data(), prob.csc_val.size() * sizeof(double), cudaMemcpyHostToDevice);
}

ParallelKOpt::~ParallelKOpt() {
    cudaFree(d_x_best);
    cudaFree(d_act_best);
    cudaFree(d_lb);
    cudaFree(d_ub);
    cudaFree(d_c);
    cudaFree(d_b);
    cudaFree(d_vartype);
    cudaFree(d_csc_col_ptr);
    cudaFree(d_csc_row_idx);
    cudaFree(d_csc_val);
}

void ParallelKOpt::compute_activity(const std::vector<double>& x, std::vector<double>& activity) {
    activity.assign(prob.num_rows, 0.0);
    for (int i = 0; i < prob.num_rows; ++i) {
        for (int p = prob.csr_row_ptr[i]; p < prob.csr_row_ptr[i+1]; ++p) {
            activity[i] += prob.csr_val[p] * x[prob.csr_col_idx[p]];
        }
    }
}

double ParallelKOpt::compute_objective(const std::vector<double>& x) {
    double obj = prob.obj_offset;
    for (int i = 0; i < prob.num_cols; ++i) obj += prob.c[i] * x[i];
    return obj;
}

void ParallelKOpt::save_solution(const std::string& output_dir, int sol_idx, const Solution& sol, double elapsed_time, double input_time) {
    std::string sol_filename = output_dir + "/solution_" + std::to_string(sol_idx) + ".sol";
    std::ofstream sol_file(sol_filename);
    sol_file << std::setprecision(17);
    sol_file << "=obj= " << sol.obj_value << "\n";
    for (int i = 0; i < prob.num_cols; i++) {
        std::string name = prob.var_names[i].empty() ? "x" + std::to_string(i) : prob.var_names[i];
        sol_file << name << " " << sol.x[i] << "\n";
    }
    sol_file.close();

    std::ofstream timing_file(output_dir + "/timing.log", std::ios_base::app);
    timing_file << std::fixed << std::setprecision(3);
    timing_file << "solution_" << sol_idx << ".sol\t" << (input_time + elapsed_time) << "\n";
    timing_file.close();
}

Solution ParallelKOpt::run(const Solution& initial_sol, 
                           const KOptParams& params, 
                           const std::string& output_dir, 
                           double t_start, 
                           double input_time,
                           int start_sol_idx) {
    
    Solution best_sol = initial_sol;
    if (best_sol.obj_value == std::numeric_limits<double>::infinity()) {
        best_sol.obj_value = compute_objective(best_sol.x);
    }
    
    std::vector<double> act_best;
    compute_activity(best_sol.x, act_best);
    
    int sol_idx = start_sol_idx;
    int no_improve = 0;
    
    double* d_best_delta_obj;
    int* d_best_i;
    double* d_best_di;
    int* d_best_j;
    double* d_best_dj;
    cudaMalloc(&d_best_delta_obj, sizeof(double));
    cudaMalloc(&d_best_i, sizeof(int));
    cudaMalloc(&d_best_di, sizeof(double));
    cudaMalloc(&d_best_j, sizeof(int));
    cudaMalloc(&d_best_dj, sizeof(double));

    std::mt19937 rng(12345);
    
    while (no_improve < params.MaxIter && (getTimeKopt() - t_start) < params.time_limit) {
        
        // 1. Try GPU exact 1-opt and 2-opt fully feasible moves
        cudaMemcpy(d_x_best, best_sol.x.data(), prob.num_cols * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_act_best, act_best.data(), prob.num_rows * sizeof(double), cudaMemcpyHostToDevice);
        
        double init_delta = 0.0;
        int init_idx = -1;
        cudaMemcpy(d_best_delta_obj, &init_delta, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_best_i, &init_idx, sizeof(int), cudaMemcpyHostToDevice);
        
        int blockSize = 256;
        int numBlocks = (prob.num_cols + blockSize - 1) / blockSize;
        evaluate_kopt_kernel<<<numBlocks, blockSize>>>(
            prob.num_cols, prob.num_rows, d_csc_col_ptr, d_csc_row_idx, d_csc_val,
            d_c, d_x_best, d_lb, d_ub, d_vartype, d_act_best, d_b,
            d_best_delta_obj, d_best_i, d_best_di, d_best_j, d_best_dj
        );
        cudaDeviceSynchronize();
        
        double h_best_delta;
        int h_best_i, h_best_j;
        double h_best_di, h_best_dj;
        cudaMemcpy(&h_best_delta, d_best_delta_obj, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_best_i, d_best_i, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_best_di, d_best_di, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_best_j, d_best_j, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_best_dj, d_best_dj, sizeof(double), cudaMemcpyDeviceToHost);
        
        bool found_improving = false;

        if (h_best_delta < -1e-6 && h_best_i != -1) {
            // Apply move
            best_sol.x[h_best_i] += h_best_di;
            if (h_best_j != -1) best_sol.x[h_best_j] += h_best_dj;
            
            compute_activity(best_sol.x, act_best); // update
            best_sol.obj_value += h_best_delta;
            found_improving = true;
            
            save_solution(output_dir, sol_idx++, best_sol, getTimeKopt() - t_start, input_time);
            no_improve = 0;
            continue;
        }

        // 2. If GPU fully feasible moves failed, use Sub-MIP repair (CPU) with random seed selection
        if (!found_improving) {
            std::vector<double> trial_x = best_sol.x;
            std::vector<bool> is_fixed(prob.num_cols, false);
            
            // Randomly select a variable with non-zero objective coefficient
            std::vector<int> candidates;
            for(int i=0; i<prob.num_cols; ++i) {
                if(abs(prob.c[i]) > 1e-6 && prob.vartype[i] != VarType::CONTINUOUS) {
                    candidates.push_back(i);
                }
            }
            
            if(candidates.empty()) break;
            std::shuffle(candidates.begin(), candidates.end(), rng);
            
            int chosen_j = -1;
            double chosen_di = 0.0;
            
            for(int j : candidates) {
                double di = (prob.c[j] > 0) ? -1.0 : 1.0;
                if(trial_x[j] + di >= prob.lb[j] && trial_x[j] + di <= prob.ub[j]) {
                    chosen_j = j;
                    chosen_di = di;
                    break;
                }
            }
            
            if(chosen_j != -1) {
                trial_x[chosen_j] += chosen_di;
                is_fixed[chosen_j] = true; // Fix this variable in the restricted sub-mip
                
                // Try to repair the rest
                bool repaired = repair_solution_improved(prob, trial_x, is_fixed, 10, params.R, 1e-6);
                
                if(repaired) {
                    double new_obj = compute_objective(trial_x);
                    if(new_obj < best_sol.obj_value - 1e-6) {
                        best_sol.x = trial_x;
                        best_sol.obj_value = new_obj;
                        compute_activity(best_sol.x, act_best);
                        save_solution(output_dir, sol_idx++, best_sol, getTimeKopt() - t_start, input_time);
                        found_improving = true;
                        no_improve = 0;
                    }
                }
            }
        }
        
        if (!found_improving) {
            no_improve++;
        }
    }
    
    cudaFree(d_best_delta_obj);
    cudaFree(d_best_i);
    cudaFree(d_best_di);
    cudaFree(d_best_j);
    cudaFree(d_best_dj);
    
    return best_sol;
}
