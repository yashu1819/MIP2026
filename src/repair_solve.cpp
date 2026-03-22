#include "epsilon_search.h"
#include "activity.h"
#include "score.h"
#include "repair.h"

#include "mip_problem.h"
#include "lp_relaxation.h"
#include "solution.h"
#include <iostream>
#include <limits>


Solution solve_with_epsilon_repair( // CORE SOLVER FUNCTION
    MIPProblem& mip,
    double time_limit
)
{
    //--------------------------------------------------
    // Solve LP relaxation
    //--------------------------------------------------

    LPRelaxation lp(mip);

    if(!lp.solve())
    {
        std::cout << "LP solve failed\n";
        return Solution();
    }

    std::vector<double> x_frac = lp.x;

    //--------------------------------------------------
    // Epsilon Neighborhood Search
    //--------------------------------------------------

    auto candidates = epsilon_neighborhood_search(mip, x_frac, 32);

    int K = std::min(5, (int)candidates.size());

    //--------------------------------------------------
    // Repair candidates → find best feasible start
    //--------------------------------------------------

    std::vector<double> best_x;
    bool best_feasible = false;
    double best_obj = std::numeric_limits<double>::infinity();

    for(int i=0;i<K;i++)
    {
        std::vector<double> x = candidates[i].x;
        std::vector<bool> is_fixed = candidates[i].is_fixed;

        bool ok = repair_solution_improved(mip, x, is_fixed, 5, 500);
        if(ok)
        {
            double obj = compute_objective(mip, x);
            if(!best_feasible || obj < best_obj)
            {
                best_feasible = true;
                best_obj = obj;
                best_x = x;
            }
        }
    }

    //--------------------------------------------------
    // If no feasible solution → return empty
    //--------------------------------------------------

    if(!best_feasible)
    {
        std::cout << "\nNo feasible solution found.\n";
        return Solution();
    }

    //--------------------------------------------------
    // Optimization Phase
    //--------------------------------------------------

    std::vector<bool> is_fixed(mip.num_cols, false);

    Solution best_sol = optimize_with_repair(
        mip,
        best_x,
        is_fixed,
        5,
        time_limit
    );

    return best_sol;
}

