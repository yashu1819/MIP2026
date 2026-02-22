#include <fstream>
#include <sstream>
#include <iomanip>
#include "mip_problem.h"
#include <iostream>
int main() {

    std::ofstream out("partitions.csv");
    if (!out.is_open()) {
        std::cerr << "Could not create partitions.csv\n";
        return 1;
    }

    for (int i = 1; i <= 50; ++i) {


        std::stringstream filename;
        filename << "../test_set/instances/instance_"
                 << std::setw(2) << std::setfill('0') << i
                 << ".mps";

        MIPProblem prob;
        prob.load_from_mps(filename.str().c_str());

        std::string type =prob.classify();

        out << std::setw(2) << std::setfill('0') << i
            << "," << type;

        if (i != 50)
            out << "\n";
    }

    out.close();
    return 0;
}
