#ifndef PARALLEL_KOPT_H
#define PARALLEL_KOPT_H

#include "mip_problem.h"
#include "solution.h"
#include <string>

struct KOptParams {
    int Kmax = 3;
    int W = 1024;      // Parallel workers (CUDA blocks/threads)
    int B = 10;        // Beam width per worker
    int R = 50;        // Repair max iterations
    int MaxIter = 100; // Outer iterations without improvement
    double time_limit = 300.0;
};

class ParallelKOpt {
public:
    ParallelKOpt(const MIPProblem& p);
    ~ParallelKOpt();

    // Runs the Parallel Adaptive K-Opt heuristic.
    // Writes better solutions continuously to output_dir
    Solution run(const Solution& initial_sol, 
                 const KOptParams& params, 
                 const std::string& output_dir, 
                 double t_start, 
                 double input_time,
                 int start_sol_idx);

private:
    const MIPProblem& prob;
    
    // Device pointers
    double* d_x_best;
    double* d_act_best;
    
    // Matrix memory (CSR & CSC)
    int* d_csr_row_ptr;
    int* d_csr_col_idx;
    double* d_csr_val;

    int* d_csc_col_ptr;
    int* d_csc_row_idx;
    double* d_csc_val;

    // Bounds and costs
    double* d_lb;
    double* d_ub;
    double* d_c;
    double* d_b;
    uint8_t* d_vartype;

    // Output buffers for local bests from GPU
    struct LocalBest {
        int i, j, k;          // up to 3 variables changed (for Kmax=3)
        double di, dj, dk;    // changes applied
        double delta_obj;
        int feasible;
    };

    LocalBest* d_local_best;
    
    // Helper to evaluate constraints on CPU and compute activity
    void compute_activity(const std::vector<double>& x, std::vector<double>& activity);
    double compute_objective(const std::vector<double>& x);
    
    // Save solution helper
    void save_solution(const std::string& output_dir, int sol_idx, const Solution& sol, double elapsed_time, double input_time);
};

#endif
