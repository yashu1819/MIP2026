#ifndef FEASIBILITY_JUMP_H
#define FEASIBILITY_JUMP_H

#include "mip_problem.h"
#include "solution.h"
#include <vector>
#include <random>

struct FeasibilityJumpParams {
    int max_iters = 500000;
    int max_restarts = 100;
    double constr_tol = 1e-6;
    double weight_inc = 1.0;
};

class FeasibilityJump {
public:
    FeasibilityJump(const MIPProblem& prob);
    ~FeasibilityJump();

    Solution run(const FeasibilityJumpParams& params);

private:
    const MIPProblem& prob;

    /* host state */
    std::vector<double> x;
    std::vector<double> residuals;   // r = b - Ax
    std::vector<double> weights;     // constraint weights

    /* device (persistent) */
    double *d_x = nullptr;
    double *d_residuals = nullptr;
    double *d_weights = nullptr;
    double *d_b = nullptr;
    int *d_csr_row_ptr = nullptr;
    int *d_csr_col_idx = nullptr;
    double *d_csr_val = nullptr;

    std::mt19937 rng;

    void initialize();
    double compute_weighted_violation_gpu();
    void update_weights(double tol);
    bool try_jump(int j, double& best_val);
};

#endif

