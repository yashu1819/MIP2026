#include "epsilon_search.h"
#include "activity.h"
#include "score.h"

#include <algorithm>
#include <cmath>
#include <random>

std::vector<double> generate_epsilons_random(
    int num_eps,
    double max_eps
)
{
    std::vector<double> eps(num_eps);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0,max_eps);

    for(int i=0;i<num_eps;i++)
        eps[i]=dist(rng);

    return eps;
}

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