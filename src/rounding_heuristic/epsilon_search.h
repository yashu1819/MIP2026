#ifndef EPSILON_SEARCH_H
#define EPSILON_SEARCH_H

#include "../mip_problem.h"
#include "../solution.h"

#include <vector>

struct EpsilonCandidate
{
    double epsilon;
    double score;
    std::vector<double> x;

    std::vector<bool> is_fixed; // variable status
};

std::vector<double> generate_epsilons_random(
    int num_eps,
    double max_eps
);

std::vector<EpsilonCandidate> epsilon_neighborhood_search(
    const MIPProblem& mip,
    const std::vector<double>& x_frac,
    int num_eps
);

#endif