#ifndef FEASIBILITY_PUMP_H
#define FEASIBILITY_PUMP_H

#include "lp_relaxation.h"
#include "mip_problem.h"
#include <vector>
#include <random>
#include <cuda_runtime.h>

struct FeasibilityPumpParams {
    int max_restarts = 10;
    int max_iters = 200;
    double constr_tol = 1e-6;
    double step_size = 0.01;
};

struct FPSolution {
    bool feasible;
    std::vector<double> x;
};

class FeasibilityPump {
public:
    FeasibilityPump(const MIPProblem& p);
    ~FeasibilityPump();

    FPSolution run(const FeasibilityPumpParams& params);

private:
    void solve_lp();
    void round_x();
    bool check_feasibility();

private:
    const MIPProblem& prob;

    std::vector<double> x;        // current integer point
    std::vector<double> x_lp;     // LP solution
    std::vector<double> residuals;

    // --- Device memory ---
    double *d_x, *d_residuals, *d_b;
    int *d_csr_row_ptr, *d_csr_col_idx;
    double *d_csr_val;

    uint8_t *d_vartype;
};

#endif

