#include "score.h"
#include <cmath>

double compute_violation_score(
    const MIPProblem& mip,
    const std::vector<double>& activity
)
{
    double score = 0.0;

    for(int i=0;i<mip.num_rows;i++)
    {
        double slack = activity[i] - mip.b[i];

        if(slack > 0)
            score += slack*slack;
            // score += slack;
    }

    return score;
}