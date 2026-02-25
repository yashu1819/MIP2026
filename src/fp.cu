// fp.cu
#include "fp.h"
#include "lp_relaxation.h"
#include "mip_problem.h"

#include <vector>
#include <cmath>
#include <cstdio>
#include <string>
#include <algorithm>
#include <cstdlib>   // rand()
#include <ctime>     // time()

static double round_variable(double val, VarType type) {
    double r = std::round(val);
    if (type == VarType::BINARY) {
        return (r <= 0.5) ? 0.0 : 1.0;
    }
    return r;
}

static std::vector<double> get_rounded_solution(
    const std::vector<double>& x_frac,
    const std::vector<VarType>& vartype,
    double& total_frac_distance
) {
    total_frac_distance = 0.0;
    std::vector<double> x_int = x_frac;
    for (size_t j = 0; j < x_frac.size(); ++j) {
        if (vartype[j] == VarType::CONTINUOUS) continue;
        double old = x_int[j];
        x_int[j] = round_variable(old, vartype[j]);
        total_frac_distance += std::fabs(x_int[j] - old);
    }
    return x_int;
}

FPResult run_feasibility_pump_cuda(
    const MIPProblem& mip,
    double time_limit_sec,
    double frac_tolerance,
    double constr_tol,
    double int_tol
) {
    FPResult res;
    res.iterations_used = 0;

    if (mip.num_cols == 0) {
        res.status_msg = "Empty problem";
        return res;
    }

    // Seed random number generator once
    static bool seeded = false;
    if (!seeded) {
        std::srand(static_cast<unsigned>(std::time(nullptr)));
        seeded = true;
    }

    double start_time = getTime();  // from mip_problem.h

    // History for cycling detection (last 8 rounded solutions)
    std::vector<std::vector<double>> history;

    // Initial LP relaxation
    LPRelaxation lp(mip);
    if (!lp.solve()) {
        res.status_msg = "Initial LP relaxation failed";
        res.time_used_sec = getTime() - start_time;
        return res;
    }

    std::vector<double> x_current = lp.x;
    printf("Initial LP objective: %.6f\n", lp.obj_value);

    int iter = 0;
    const int SAFETY_MAX_ITER = 1000;  // prevent infinite loop

    while ((getTime() - start_time < time_limit_sec) && iter < SAFETY_MAX_ITER) {
        res.iterations_used = ++iter;
        res.time_used_sec = getTime() - start_time;

        double frac_dist = 0.0;
        std::vector<double> x_rounded = get_rounded_solution(x_current, mip.vartype, frac_dist);

        // Full feasibility check
        if (mip.check_feasible(x_rounded, constr_tol, int_tol)) {
            res.found_feasible = true;
            res.solution = x_rounded;
            res.objective_value = 0.0;
            for (size_t j = 0; j < mip.c.size(); ++j) {
                res.objective_value += mip.c[j] * x_rounded[j];
            }
	    res.objective_value-=mip.obj_offset;
            res.status_msg = "Feasible integer solution found";
            printf("→ SUCCESS at iteration %d  (frac dist = %.2e)\n", iter, frac_dist);
            res.time_used_sec = getTime() - start_time;
            break;
        }

        // Very small fractional distance → try forced rounding
        if (frac_dist > 0.0 && frac_dist < 1e-4) {
            printf("Very small frac dist (%.2e) — trying forced rounding (iter %d)\n", frac_dist, iter);

            std::vector<double> forced = x_current;
            for (size_t j = 0; j < forced.size(); ++j) {
                if (mip.vartype[j] != VarType::CONTINUOUS) {
                    forced[j] = std::round(forced[j]);
                    if (mip.vartype[j] == VarType::BINARY) {
                        forced[j] = (forced[j] < 0.5) ? 0.0 : 1.0;
                    }
                }
            }

            if (mip.check_feasible(forced, constr_tol * 10.0, int_tol)) {
                res.found_feasible = true;
                res.solution = forced;
                res.objective_value = 0.0;
                for (size_t j = 0; j < mip.c.size(); ++j) {
                    res.objective_value += mip.c[j] * forced[j];
                }
		res.objective_value-=mip.obj_offset;
                res.status_msg = "Feasible solution found via forced rounding";
                printf("→ Forced rounding SUCCESS at iter %d\n", iter);
                res.time_used_sec = getTime() - start_time;
                break;
            }
        }

        // Cycling detection
        bool cycling = false;
        for (const auto& prev : history) {
            if (std::equal(x_rounded.begin(), x_rounded.end(), prev.begin())) {
                cycling = true;
                break;
            }
        }

        history.push_back(x_rounded);
        if (history.size() > 8) history.erase(history.begin());

        // Perturbation
        bool perturb = cycling ||
                       (frac_dist > 1e-6 && frac_dist < 0.05 && iter % 12 == 0 && iter >= 12);

        if (perturb) {
            printf("→ Perturbation applied (cycling/small frac) at iter %d\n", iter);

            for (size_t j = 0; j < x_rounded.size(); ++j) {
                if (mip.vartype[j] != VarType::CONTINUOUS) {
                    double f = std::abs(x_current[j] - std::round(x_current[j]));
                    if (f > 1e-7 && (std::rand() % 100 < 35)) {  // ~35% chance for fractional vars
                        x_rounded[j] = 1.0 - x_rounded[j];  // flip
                    }
                }
            }
        }

        // Distance-weighted pump objective
        std::vector<double> pump_objective = mip.c;

        for (size_t j = 0; j < mip.num_cols; ++j) {
            if (mip.vartype[j] == VarType::CONTINUOUS) continue;

            double frac_j = std::abs(x_current[j] - std::round(x_current[j]));
            if (frac_j < 1e-8) continue;  // already integer

            double weight = 1.0 / std::max(frac_j, 1e-7);  // inverse fractional distance
            weight = std::min(weight, 1e7);  // cap to avoid numerical blow-up

            double sign = (x_current[j] >= std::round(x_current[j])) ? -1.0 : 1.0;

            pump_objective[j] = sign * weight;
        }

        // Solve next LP
        LPRelaxation pump_lp(mip);
        pump_lp.c = pump_objective;
        pump_lp.obj_offset = mip.obj_offset;

        if (!pump_lp.solve()) {
            res.status_msg = "Pump LP solve failed at iteration " + std::to_string(iter);
            res.time_used_sec = getTime() - start_time;
            break;
        }

        x_current = pump_lp.x;

        printf("Iter %3d   frac=%.2e   obj=%.6f\n", iter, frac_dist, pump_lp.obj_value);
    }

    if (!res.found_feasible) {
        res.status_msg = "Time limit reached without feasible integer solution";
        res.time_used_sec = getTime() - start_time;
    }

    return res;
}
