#ifndef FEASIBILITY_JUMP_H
#define FEASIBILITY_JUMP_H

#include "lp_relaxation.h"
#include "mip_problem.h"
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include "solution.h"

struct FeasibilityJumpParams {
    int max_restarts = 1;
    int max_iters = 1000;
    double constr_tol = 1e-6;
};


class FeasibilityJump {
public:
    FeasibilityJump(const MIPProblem& p);
    ~FeasibilityJump();

    void initialize();
    Solution run(const FeasibilityJumpParams& params);

private:
    const MIPProblem& prob;
    std::mt19937 rng;
    
    std::vector<double> x;
    std::vector<double> residuals;
    std::vector<double> weights;

    // --- Added for Parallel Scoring ---
    std::vector<double> h_scores;
    std::vector<double> h_deltas;

    // Device Pointers
    double *d_x, *d_residuals, *d_weights, *d_b;
    double *d_lb, *d_ub;
    uint8_t *d_vartype;

    // CSR Matrix (for residuals)
    int *d_csr_row_ptr, *d_csr_col_idx;
    double *d_csr_val;

    // CSC Matrix (for scoring)
    int *d_csc_col_ptr, *d_csc_row_idx;
    double *d_csc_val;

    // Scoring buffers
    double *d_scores, *d_deltas;
};

#endif
