#include <iostream>
#include <sstream>
#include <string>
#include <filesystem>
#include <iomanip> // For std::setw and std::setfill
#include "coin/CbcModel.hpp"
#include "coin/OsiClpSolverInterface.hpp"

namespace fs = std::filesystem;

int main() {
    std::string inputDir  = "test_set/instances";
    std::string outputDir = "test_set/relaxedInstances";

    // Create directory if it doesn't exist
    if (!fs::exists(outputDir)) {
        fs::create_directories(outputDir);
    }

    for (int i = 1; i <= 50; i++) {
        // Using stringstreams or format to build paths
        std::stringstream inPath, outPath;
        
        // Formats number to 2 digits (e.g., 01, 02...)
        inPath << inputDir << "/instance_" << std::setw(2) << std::setfill('0') << i << ".mps";
        
        // FIX: We provide the name WITHOUT the extension because writeMps adds it
        outPath << outputDir << "/relaxed_" << std::setw(2) << std::setfill('0') << i;

        std::string inFile  = inPath.str();
        std::string outBase = outPath.str();

        if (!fs::exists(inFile)) {
            std::cerr << "Skipping: " << inFile << " (not found)" << std::endl;
            continue;
        }

        OsiClpSolverInterface solver;

        // Load MPS
        if (solver.readMps(inFile.c_str()) != 0) {
            std::cerr << "Error reading: " << inFile << std::endl;
            continue;
        }

        // RELAXATION FIX: Use setContinuous instead of setInteger
        int numCols = solver.getNumCols();
        for (int col = 0; col < numCols; col++) {
            if (solver.isInteger(col)) {
                solver.setContinuous(col); 
            }
        }

        // Write model (adds .mps and possibly .gz automatically)
        solver.writeMps(outBase.c_str());

        // Check if it created .mps.gz and rename it to just .mps if you want it uncompressed
        std::string finalExpected = outBase + ".mps";
        std::string autoGzipped  = outBase + ".mps.gz";

        if (fs::exists(autoGzipped)) {
            std::cout << "Uncompressing auto-generated .gz file..." << std::endl;
            std::string cmd = "gunzip -f " + autoGzipped;
            system(cmd.c_str());
        }

        std::cout << "Successfully written: " << finalExpected << std::endl;
    }

    return 0;
}
