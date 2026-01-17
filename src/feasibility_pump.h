#ifndef FEASIBILITY_PUMP_H
#define FEASIBILITY_PUMP_H

#include "mip_problem.h"
#include "lp_relaxation.h"
#include "solution.h"
#include <chrono>

class FeasibilityPump {
public:
    explicit FeasibilityPump(const MIPProblem& mip);

    // returns true if solution found
    bool run(
        Solution& sol,
        int max_iters = 50,
        double integrality_tol = 1e-6
    );

    double last_runtime_sec() const { return last_runtime_sec_; }

private:
    const MIPProblem& mip_;
    LPRelaxation lp_;
    double last_runtime_sec_ = 0.0;

    bool is_integral(const std::vector<double>& x, double tol) const;
    std::vector<double> round_solution(const std::vector<double>& x) const;
};

#endif

