#include <iostream>
#include <string>
#include <chrono>
#include <cmath>

#include "mip_problem.h"
#include "lp_relaxation.h"
#include "chebyshev_center.h"

using namespace std;

double compute_min_slack(const MIPProblem& mip, const std::vector<double>& x){
    double min_slack = 1e100;

    for (int i = 0; i < mip.num_rows; ++i){
        double lhs = 0.0;

        int start = mip.csr_row_ptr[i];
        int end   = mip.csr_row_ptr[i+1];

        for (int p = start; p < end; ++p){
            int j = mip.csr_col_idx[p];
            lhs += mip.csr_val[p] * x[j];
        }

        double slack = mip.b[i] - lhs;
        min_slack = std::min(min_slack, slack);
    }
    return min_slack;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: ./solver instance.mps\n";
        return 1;
    }

    std::string filename = argv[1];

    MIPProblem mip;
    mip.load_from_mps(filename);
    mip.finalize();

    std::cout << "Rows: " << mip.num_rows << " Cols: " << mip.num_cols << "\n";

    // LP solve
    LPRelaxation lp(mip);

    auto t0 = std::chrono::high_resolution_clock::now();
    bool ok = lp.solve();
    auto t1 = std::chrono::high_resolution_clock::now();

    if (!ok) {
        std::cout << "LP infeasible.\n";
        return 1;
    }

    double lp_time = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "LP obj = " << lp.obj_value << " time = " << lp_time << "\n";

    bool okk = lp.check_feasible_fast(lp.x);
    if(okk) std::cout << "LP feasible? : True " << "\n";
    else std::cout << "LP feasible? : False" << "\n";

    double lp_slack = compute_min_slack(mip, lp.x);
    std::cout << "LP min slack = " << lp_slack << "\n";

}