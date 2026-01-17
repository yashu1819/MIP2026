#include "mip_problem.h"
#include "lp_relaxation.h"

#include <iostream>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include <chrono>

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <instance.mps>\n";
        return 1;
    }

    // ---------- extract instance name ----------
    // .../instance_08.mps -> instance_08
    std::string instance =
        std::filesystem::path(argv[1]).stem().string();

    // ---------- load problem ----------
    MIPProblem mip;
    mip.load_from_mps(argv[1]);
    mip.finalize();

    LPRelaxation lp(mip);

    // ---------- time measurement ----------
    auto t0 = std::chrono::high_resolution_clock::now();
    bool ok = lp.solve();
    auto t1 = std::chrono::high_resolution_clock::now();

    double time_sec =
        std::chrono::duration<double>(t1 - t0).count();

    if (!ok) {
        std::cerr << "LP relaxation not solved\n";
        return 2;
    }

    double obj = lp.obj_value;

    // ---------- append to CSV ----------
    std::ofstream csv("../Benchmark/h100PDLPresults.csv", std::ios::app);
    if (!csv) {
        std::cerr << "Failed to open results.csv\n";
        return 3;
    }

    csv << instance << ","
        << std::setprecision(15) << time_sec << ","
        << std::setprecision(15) << obj << "\n";

    return 0;
}

