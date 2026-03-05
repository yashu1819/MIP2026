#ifndef EPSILON_SEARCH_H
#define EPSILON_SEARCH_H

#include "../Definition/mip_problem.h"
#include "../Definition/solution.h"

#include <vector>

// struct EpsilonCandidate
// {
//     double epsilon;
//     double score;
// };

struct EpsilonCandidate
{
    double epsilon;
    double score;
    std::vector<double> x;

    std::vector<bool> is_fixed; // variable status
};

std::vector<EpsilonCandidate> epsilon_neighborhood_search(
    const MIPProblem& mip,
    const std::vector<double>& x_frac,
    int num_eps
);

#endif