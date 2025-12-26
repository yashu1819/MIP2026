#include <iostream>
#include <sstream>
#include <string>
#include <filesystem>
#include "coin/CbcModel.hpp"
#include "coin/OsiClpSolverInterface.hpp"

namespace fs = std::filesystem;

int main() {
    std::string inputDir  = "test_set/instances";
    std::string outputDir = "test_set/relaxedInstances";

    for (int i = 1; i <= 50; i++) {
        std::stringstream in, out;

        if (i < 10) {
            in  << inputDir  << "/instance_0" << i << ".mps";
            out << outputDir << "/relaxed_0"  << i << ".mps";
        } else {
            in  << inputDir  << "/instance_"  << i << ".mps";
            out << outputDir << "/relaxed_"   << i << ".mps";
        }

        std::string inFile  = in.str();
        std::string outFile = out.str();

        OsiClpSolverInterface solver;

        // load MPS
        solver.readMps(inFile.c_str());

        // relax integrality (CBC stores integer attributes on solver layer)
        int numCols = solver.getNumCols();
        for (int col = 0; col < numCols; col++) {
            solver.setInteger(col, false); // this is the relaxation
        }

        // write relaxed model
        solver.writeMps(outFile.c_str());

        std::cout << "written: " << outFile << std::endl;
    }

    return 0;
}
