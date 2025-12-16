#ifndef LP_SOLVER_H
#define LP_SOLVER_H

#include "../problem/milp_problem.h"
#include "../solution/solution.h"
#include <memory>

namespace milp {

class LPSolver {
public:
    virtual ~LPSolver() = default;
    virtual void load_problem(const MILPProblem& p) = 0;
    virtual Solution solve() = 0;
    virtual void set_time_limit(double seconds) { (void)seconds; }
};

using LPSolverPtr = std::unique_ptr<LPSolver>;

} // namespace milp

#endif // LP_SOLVER_H
