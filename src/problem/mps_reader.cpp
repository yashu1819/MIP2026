#include "mps_reader.h"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace milp;

bool MpsReader::loadFromFile(const std::string& filename, MILPProblem& out_problem) {
    if (filename.empty() || filename == "example") {
        // Build a simple demo problem:
        // Minimize:   1*x0 + 2*x1
        // Subject to: 1*x0 + 2*x1 >= 3
        //             x0, x1 >= 0
        out_problem.name = "demo_example";
        int x0 = out_problem.add_variable(0.0, 10.0, false, "x0");
        int x1 = out_problem.add_variable(0.0, 10.0, false, "x1");
        out_problem.set_obj_coef(x0, 1.0);
        out_problem.set_obj_coef(x1, 2.0);
        std::vector<int> idx = {x0, x1};
        std::vector<double> coeff = {1.0, 2.0};
        out_problem.add_constraint(idx, coeff, 3.0, std::numeric_limits<double>::infinity(), "c0");
        return true;
    }

    // Try to open file and detect minimal MPS (not implemented)
    std::ifstream ifs(filename);
    if (!ifs) {
        std::cerr << "[MpsReader] Cannot open file: " << filename << "\n";
        return false;
    }
    std::cerr << "[MpsReader] MPS parsing not implemented. Provide 'example' to load demo.\n";
    return false;
}
