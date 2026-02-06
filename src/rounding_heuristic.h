#ifndef ROUNDING_HEURISTIC_H
#define ROUNDING_HEURISTIC_H

#include "mip_problem.h"
#include "solution.h"
#include <vector>

using namespace std;

struct RoundingHeuristic{
	const MIPProblem* mip = nullptr; // MIP problem
	const vector<double>* x_frac = nullptr;  // input fractional solution

	Solution best_solution; // output

	explicit RoundingHeuristic(const MIPProblem& mip_);

	void set_fractional_solution(const vector<double>& x);

	bool run(int num_attempts);
};

#endif
