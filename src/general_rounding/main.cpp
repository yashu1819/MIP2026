#include "locks.h"
#include "locks_rounding.h"
#include "epsilon_search.h"
#include "activity.h"
#include "score.h"
#include "repair.h"

#include "../Definition/mip_problem.h"
#include "../relaxation/lp_relaxation.h"

#include <iostream>

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

    std::cout << "Model loaded\n";
    std::cout << "Rows : " << mip.num_rows << "\n";
    std::cout << "Cols : " << mip.num_cols << "\n";

    //--------------------------------------------------
    // Solve LP relaxation
    //--------------------------------------------------

    std::cout << "\nSolving LP relaxation...\n";

    LPRelaxation lp(mip);

    bool ok = lp.solve();

    if(!ok)
    {
        std::cout << "LP solve failed\n";
        return 0;
    }

    std::vector<double> x_frac = lp.x;

    std::cout << "LP solved\n";
    std::cout << "LP objective = " << lp.obj_value << "\n";

    //--------------------------------------------------
    // Compute Locks
    //--------------------------------------------------

    std::cout << "\nComputing variable locks (GPU)...\n";

    VariableLocks locks = compute_variable_locks_gpu(mip);

    std::cout << "Locks computed\n";

    for(int i=0;i<10 && i<mip.num_cols;i++)
    {
        std::cout
        << "Var " << i
        << " up=" << locks.up_locks[i]
        << " down=" << locks.down_locks[i]
        << "\n";
    }

    //--------------------------------------------------
    // Lock Guided Rounding
    //--------------------------------------------------

    std::cout << "\nRunning lock-guided rounding...\n";

    Solution sol = lock_guided_rounding_gpu(mip, x_frac, locks);

    std::cout << "Rounded solution (first 10 variables)\n";

    for(int i=0;i<10 && i<mip.num_cols;i++)
    {
        std::cout
        << "x[" << i << "] = "
        << sol.x[i]
        << "\n";
    }

    //--------------------------------------------------
    // Feasibility check
    //--------------------------------------------------

    bool feasible = mip.check_feasible(sol.x);

    std::cout << "\nFeasible after rounding: "
              << (feasible ? "YES" : "NO")
              << "\n";


    if(!mip.check_feasible(sol.x))
    {
        std::cout<<"Running epsilon neighborhood search\n";

        auto candidates = epsilon_neighborhood_search(
            mip,
            x_frac,
            32
        );

        int K = 5;

        std::cout<<"Top epsilon candidates:\n";

        for(int i=0;i<K;i++)
        {
            std::cout
            <<"eps="<<candidates[i].epsilon
            <<" score="<<candidates[i].score
            <<"\n";
        }

        //-------------------------------------------------
        // Try repair
        //-------------------------------------------------

        for(int i=0;i<K;i++)
        {
            std::vector<double> x = candidates[i].x;

            std::cout<<"\nRepairing candidate "<<i<<"\n";

            //-------------------------------------------------
            // Initial score
            //-------------------------------------------------

            std::vector<double> activity;

            compute_constraint_activity_gpu(mip,x,activity);

            double before_score = compute_violation_score(mip,activity);

            std::cout<<"Initial violation score = "
                    <<before_score<<"\n";

            //-------------------------------------------------
            // Run repair
            //-------------------------------------------------

            bool ok = repair_solution(mip,x,500);

            //-------------------------------------------------
            // Score after repair
            //-------------------------------------------------

            compute_constraint_activity_gpu(mip,x,activity);

            double after_score = compute_violation_score(mip,activity);

            std::cout<<"After repair score = "
                    <<after_score<<"\n";

            //-------------------------------------------------
            // Feasibility
            //-------------------------------------------------

            bool feasible = mip.check_feasible(x);

            std::cout<<"Feasible = "
                    <<(feasible ? "YES" : "NO")<<"\n";

            //-------------------------------------------------
            // Print first variables
            //-------------------------------------------------

            std::cout<<"Repaired solution (first 10 vars):\n";

            for(int j=0;j<10 && j<mip.num_cols;j++)
            {
                std::cout<<"x["<<j<<"] = "<<x[j]<<"\n";
            }

            //-------------------------------------------------
            // Accept solution if feasible
            //-------------------------------------------------

            if(ok && feasible)
            {
                std::cout<<"\nFeasible solution found after repair!\n";

                sol.x = x;
                sol.feasible = true;

                break;
            }
        }
    }

    return 0;
}