#include <iostream>
#include "problem/mps_reader.h"
#include "solver/gurobi_lp.h"
#include "solution/solution.h"
#include "utils/timer.h"

using namespace milp;

int main(int argc, char** argv) {
    std::string infile = (argc > 1) ? argv[1] : "example";

    MILPProblem problem;
    MpsReader reader;
    if (!reader.loadFromFile(infile, problem)) {
        std::cerr << "Failed to load problem. Use 'example' or a supported path.\n";
        return 1;
    }

    problem.print_summary();

    GurobiLPSolver solver;
    solver.set_time_limit(5.0);
    solver.load_problem(problem);

    Timer t;
    Solution sol = solver.solve();
    double elapsed = t.elapsed_seconds();

    std::cout << "LP solve finished in " << elapsed << " seconds\n";
    sol.print(std::cout);

    return 0;
}
