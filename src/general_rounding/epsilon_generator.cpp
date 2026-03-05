#include "epsilon_generator.h"
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