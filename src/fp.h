// fp.h
#ifndef FP_H
#define FP_H

#include "mip_problem.h"
#include <vector>
#include <string>

struct FPResult {
    bool found_feasible = false;
    double objective_value = 0.0;
    std::vector<double> solution;
    std::string status_msg;
    int iterations_used = 0;
    double time_used_sec = 0.0;
};

FPResult run_feasibility_pump_cuda(
    const MIPProblem& mip,
    double time_limit_sec = 180.0,  // default 3 min
    double frac_tolerance = 1e-5,
    double constr_tol = 1e-6,
    double int_tol = 1e-5
);

#endif
