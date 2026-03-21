// #include "epsilon_search.h"
// #include "activity.h"
// #include "score.h"
// #include "repair.h"

// #include "../mip_problem.h"
// #include "../lp_relaxation.h"

// #include <iostream>
// #include <limits>

// int main(int argc, char** argv)
// {
//     if(argc < 2)
//     {
//         std::cout << "Usage: ./solver instance.mps\n";
//         return 0;
//     }

//     std::string filename = argv[1];

//     //--------------------------------------------------
//     // Load MIP
//     //--------------------------------------------------

//     MIPProblem mip;
//     mip.load_from_mps(filename);
//     mip.finalize();

//     std::cout << "Model loaded\n";
//     std::cout << "Rows : " << mip.num_rows << "\n";
//     std::cout << "Cols : " << mip.num_cols << "\n";

//     //--------------------------------------------------
//     // Solve LP relaxation
//     //--------------------------------------------------

//     std::cout << "\nSolving LP relaxation...\n";

//     LPRelaxation lp(mip);

//     if(!lp.solve())
//     {
//         std::cout << "LP solve failed\n";
//         return 0;
//     }

//     std::vector<double> x_frac = lp.x;

//     std::cout << "LP solved\n";
//     std::cout << "LP objective = " << lp.obj_value << "\n";

//     //--------------------------------------------------
//     // Epsilon Neighborhood Search
//     //--------------------------------------------------

//     std::cout << "\nRunning epsilon neighborhood search...\n";

//     auto candidates = epsilon_neighborhood_search(mip, x_frac, 32);
//     int K = std::min(5, (int)candidates.size());
//     std::cout << "Top epsilon candidates:\n";
//     for(int i=0;i<K;i++) std::cout << "eps=" << candidates[i].epsilon << " score=" << candidates[i].score << "\n";

//     //--------------------------------------------------
//     // Repair candidates → find best feasible start
//     //--------------------------------------------------

//     std::vector<double> best_x;
//     bool best_feasible = false;
//     double best_obj = std::numeric_limits<double>::infinity();

//     for(int i=0;i<K;i++)
//     {
//         std::vector<double> x = candidates[i].x;
//         std::vector<bool> is_fixed = candidates[i].is_fixed;

//         std::cout << "\nRepairing candidate " << i << "\n";

//         bool ok = repair_solution_improved(mip, x, is_fixed, 5, 500);

//         // bool feasible = mip.check_feasible(x);

//         if(ok)
//         {
//             double obj = compute_objective(mip, x);

//             std::cout << "Feasible with objective = " << obj << "\n";

//             if(!best_feasible || obj < best_obj)
//             {
//                 best_feasible = true;
//                 best_obj = obj;
//                 best_x = x;
//             }
//         }
//         else
//         {
//             std::cout << "Not feasible\n";
//         }
//     }

//     //--------------------------------------------------
//     // If no feasible solution → exit
//     //--------------------------------------------------

//     if(!best_feasible)
//     {
//         std::cout << "\nNo feasible solution found.\n";
//         return 0;
//     }

//     //--------------------------------------------------
//     // Optimization Phase (THIS WAS MISSING)
//     //--------------------------------------------------

//     std::cout << "\nStarting optimization phase...\n";

//     std::vector<bool> is_fixed(mip.num_cols, false);

//     Solution best_sol = optimize_with_repair(
//         mip,
//         best_x,
//         is_fixed,
//         5,      // top_k
//         300.0     // seconds (tune this)
//     );

//     //--------------------------------------------------
//     // Final Output
//     //--------------------------------------------------

//     std::cout << "\n=====================================\n";
//     std::cout << "FINAL RESULT\n";
    
//     std::cout << "Feasible: YES\n";
//     std::cout << "Objective: " << best_sol.obj_value << "\n";

//     std::cout << "First 10 variables:\n";
//     for(int i=0;i<10 && i<mip.num_cols;i++)
//         std::cout << "x["<<i<<"] = " << best_sol.x[i] << "\n";

//     std::cout << "=====================================\n";

//     return 0;
// }


#include "epsilon_search.h"
#include "activity.h"
#include "score.h"
#include "repair.h"

#include "../mip_problem.h"
#include "../lp_relaxation.h"

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

    std::cout << "\nSolving LP relaxation...\n";

    LPRelaxation lp(mip);

    if(!lp.solve())
    {
        std::cout << "LP solve failed\n";
        return Solution();
    }

    std::vector<double> x_frac = lp.x;

    std::cout << "LP solved\n";
    std::cout << "LP objective = " << lp.obj_value << "\n";

    //--------------------------------------------------
    // Epsilon Neighborhood Search
    //--------------------------------------------------

    std::cout << "\nRunning epsilon neighborhood search...\n";

    auto candidates = epsilon_neighborhood_search(mip, x_frac, 32);

    int K = std::min(5, (int)candidates.size());

    std::cout << "Top epsilon candidates:\n";
    for(int i=0;i<K;i++)
    {
        std::cout << "eps=" << candidates[i].epsilon
                  << " score=" << candidates[i].score << "\n";
    }

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

        std::cout << "\nRepairing candidate " << i << "\n";

        bool ok = repair_solution_improved(mip, x, is_fixed, 5, 500);

        if(ok)
        {
            double obj = compute_objective(mip, x);

            std::cout << "Feasible with objective = " << obj << "\n";

            if(!best_feasible || obj < best_obj)
            {
                best_feasible = true;
                best_obj = obj;
                best_x = x;
            }
        }
        else
        {
            std::cout << "Not feasible\n";
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

    std::cout << "\nStarting optimization phase...\n";

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

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        std::cout << "Usage: ./solver instance.mps\n";
        return 0;
    }

    std::string filename = argv[1];

    //--------------------------------------------------
    // Load MIP
    //--------------------------------------------------

    MIPProblem mip;
    mip.load_from_mps(filename);
    mip.finalize();

    double time_limit = 30.0;  // change as needed

    Solution sol = solve_with_epsilon_repair(mip, time_limit);

    std::cout << "Final Objective: " << sol.obj_value << "\n";

    return 0;
}