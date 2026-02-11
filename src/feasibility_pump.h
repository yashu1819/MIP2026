#ifndef FEASIBILITY_PUMP_H
#define FEASIBILITY_PUMP_H

#include "lp_relaxation.h"
#include "mip_problem.h"
#include "solution.h"
#include <vector>
#include <cuda_runtime.h>

struct FeasibilityPumpParams {
    int max_restarts = 10;
    int max_iters = 200;
    double constr_tol = 1e-6;
    double step_size = 0.01;
};

class FeasibilityPump {
public:
    FeasibilityPump(const MIPProblem& p);
    ~FeasibilityPump();

    Solution run(const FeasibilityPumpParams& params);

private:
    void solve_lp();
    void round_x();
    bool check_feasibility(double constr_tol);

private:
    const MIPProblem& prob;

    std::vector<double> x;
    std::vector<double> x_lp;
    std::vector<double> residuals;

    // Device memory
    double *d_x = nullptr;
    double *d_residuals = nullptr;
    double *d_b = nullptr;

    int *d_csr_row_ptr = nullptr;
    int *d_csr_col_idx = nullptr;
    double *d_csr_val = nullptr;
};

#endif

