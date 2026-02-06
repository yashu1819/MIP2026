#ifndef HEURISTIC_H
#define HEURISTIC_H

#include mip_problem.h
#include solution.h

class Heuristic {
 public:
	 Heuristic(){};
	 virtual ~Heuristic(){};
	 virtual bool solve(MIPProblem p, Solution s1);
	 

}


#endif
