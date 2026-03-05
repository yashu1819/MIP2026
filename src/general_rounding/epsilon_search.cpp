// #include "epsilon_search.h"
// #include "epsilon_generator.h"
// #include "activity.h"
// #include "score.h"

// #include <algorithm>
// #include <cmath>

// std::vector<EpsilonCandidate> epsilon_neighborhood_search(
//     const MIPProblem& mip,
//     const std::vector<double>& x_frac,
//     int num_eps
// )
// {
//     auto epsilons = generate_epsilons_random(num_eps,0.5);

//     std::vector<EpsilonCandidate> results;

//     for(double eps:epsilons)
//     {
//         std::vector<double> x = x_frac;

//         for(int j=0;j<mip.num_cols;j++)
//         {
//             double frac = std::fabs(x[j] - std::round(x[j]));

//             if(frac < eps)
//                 x[j] = std::round(x[j]);
//         }

//         std::vector<double> activity;

//         compute_constraint_activity_gpu(mip,x,activity);

//         double score = compute_violation_score(mip,activity);

//         EpsilonCandidate cand;
//         cand.epsilon = eps;
//         cand.score = score;
//         cand.x = x;

//         results.push_back(cand);
//     }

//     std::sort(results.begin(),results.end(),
//         [](const EpsilonCandidate& a,const EpsilonCandidate& b)
//         {
//             return a.score < b.score;
//         });

//     return results;
// }


#include "epsilon_search.h"
#include "epsilon_generator.h"
#include "activity.h"
#include "score.h"

#include <algorithm>
#include <cmath>

std::vector<EpsilonCandidate> epsilon_neighborhood_search(
    const MIPProblem& mip,
    const std::vector<double>& x_frac,
    int num_eps
)
{
    auto epsilons = generate_epsilons_random(num_eps,0.5);

    std::vector<EpsilonCandidate> results;

    for(double eps : epsilons)
    {
        std::vector<double> x = x_frac;

        std::vector<bool> is_fixed(mip.num_cols,false);

        //------------------------------------------------
        // determine fixed / free variables
        //------------------------------------------------

        for(int j=0;j<mip.num_cols;j++)
        {
            if(mip.vartype[j] == VarType::CONTINUOUS)
            {
                // continuous variables always free
                is_fixed[j] = false;
                continue;
            }

            double frac = std::fabs(x[j] - std::round(x[j]));

            if(frac < eps)
            {
                x[j] = std::round(x[j]);
                is_fixed[j] = true;
            }
            else
            {
                is_fixed[j] = false;
            }
        }

        //------------------------------------------------
        // compute violation score
        //------------------------------------------------

        std::vector<double> activity;

        compute_constraint_activity_gpu(mip,x,activity);

        double score = compute_violation_score(mip,activity);

        //------------------------------------------------
        // store candidate
        //------------------------------------------------

        EpsilonCandidate cand;

        cand.epsilon = eps;
        cand.score = score;
        cand.x = x;
        cand.is_fixed = is_fixed;

        results.push_back(cand);
    }

    //------------------------------------------------
    // sort candidates
    //------------------------------------------------

    std::sort(results.begin(),results.end(),
        [](const EpsilonCandidate& a,const EpsilonCandidate& b)
        {
            return a.score < b.score;
        });

    return results;
}