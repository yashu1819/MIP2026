#ifndef FEASIBILITY_PUMP_H
#define FEASIBILITY_PUMP_H

#include "lp_relaxation.h"
#include "mip_problem.h"
#include "solution.h"
#include <vector>
#include <unordered_set>
#include <random>
#include <cstdint>

struct FeasibilityPumpParams {
    int max_restarts = 10;
    int max_iters = 200;
    double constr_tol = 1e-6;
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
    void compute_residuals();
    void perturb();
    void perturb_objective();
    uint64_t hash_rounded() const;

private:
    const MIPProblem& prob;
    MIPProblem aux_prob;
    std::vector<double> original_c;
    std::unordered_set<uint64_t> seen_rounded;

    std::vector<double> x;
    std::vector<double> x_lp;
    std::vector<double> residuals;
};

#endif
