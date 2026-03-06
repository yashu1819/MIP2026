#include "locks.h"
#include "locks_rounding.h"
#include "epsilon_search.h"
#include "activity.h"
#include "score.h"
#include "repair.h"

#include "../Definition/mip_problem.h"
#include "../relaxation/lp_relaxation.h"

#include <iostream>


std::string detect_problem_type(const MIPProblem& mip)
{
    int int_count = 0;
    int bin_count = 0;
    int cont_count = 0;

    for(int j=0;j<mip.num_cols;j++)
    {
        if(mip.vartype[j] == VarType::BINARY)
            bin_count++;
        else if(mip.vartype[j] == VarType::INTEGER)
            int_count++;
        else
            cont_count++;
    }

    if(cont_count == 0 && int_count == 0)
        return "PB";

    if(cont_count == 0 && bin_count < mip.num_cols)
        return "PI";

    if(cont_count > 0 && int_count == 0)
        return "MB";

    return "MI";
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

    double start_time = getTime();

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

        double best_score_overall = 1e100;
        double max_score_improvement = 0.0;
        int best_candidate_index = -1;
        std::vector<double> best_x = sol.x;

        int limit = std::min(K, (int)candidates.size());
        for(int i=0;i<limit;i++)
        {
            std::vector<double> x = candidates[i].x;
            std::vector<bool> is_fixed = candidates[i].is_fixed;

            std::cout<<"\nRepairing candidate "<<i<<"\n";

            int fixed_count = 0;
            for(bool f : is_fixed) if(f) fixed_count++;

            std::cout<<"Fixed variables = "
                    <<fixed_count<<" / "<<mip.num_cols<<"\n";

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

            // bool ok = repair_solution(mip, x, is_fixed, 2000);
            bool ok = repair_solution_improved(mip, x, is_fixed, 2000);

            //-------------------------------------------------
            // Score after repair
            //-------------------------------------------------

            compute_constraint_activity_gpu(mip,x,activity);

            double after_score = compute_violation_score(mip,activity);

            std::cout<<"After repair score = "
                    <<after_score<<"\n";

            //-------------------------------------------------
            // Improvement statistics
            //-------------------------------------------------

            double improvement = before_score - after_score;

            if(improvement > max_score_improvement)
                max_score_improvement = improvement;

            if(after_score < best_score_overall)
            {
                best_score_overall = after_score;
                best_x = x;
                best_candidate_index = i;
            }

            //-------------------------------------------------
            // Feasibility check
            //-------------------------------------------------

            bool feasible = mip.check_feasible(x);

            std::cout<<"Feasible = "
                    <<(feasible ? "YES" : "NO")<<"\n";

            //-------------------------------------------------
            // Early success threshold
            //-------------------------------------------------

            //-------------------------------------------------
            // Debug output
            //-------------------------------------------------

            std::cout<<"Repaired solution (first 10 vars):\n";

            for(int j=0;j<10 && j<mip.num_cols;j++)
            {
                std::cout<<"x["<<j<<"] = "<<x[j]<<"\n";
            }
        }

        double end_time = getTime();
        double elapsed = end_time - start_time;
        int result = (best_score_overall < 1e-5) ? 1 : 0;

        std::cout<<"\n-------------------------------------\n";
        std::cout<<"Repair statistics\n";
        std::cout<<"Best score achieved = "
                <<best_score_overall<<"\n";

        std::cout<<"Best candidate index = "
                <<best_candidate_index<<"\n";
        std::cout<<"Max score improvement = "
                <<max_score_improvement<<"\n";

        bool near_feasible = (best_score_overall < 1e-5);

        std::cout<<"Solution found = "
                <<(near_feasible ? "YES" : "NO")<<"\n";
        std::cout<<"-------------------------------------\n";

        sol.x = best_x;

        std::string problem_type = detect_problem_type(mip);

        std::cout << "CSV_RESULT,"
                << filename << ","
                << mip.num_cols << ","
                << problem_type << ","
                << best_candidate_index << ","
                << best_score_overall*0.1 << ","
                << result << ","
                << elapsed << ","
                << max_score_improvement
                << std::endl;
    }

    return 0;
}