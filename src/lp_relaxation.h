#ifndef LP_RELAXATION_H
#define LP_RELAXATION_H

#include "mip_problem.h"
#include <vector>
#include <limits>

/*
  LPRelaxation:
  - Continuous relaxation of a MIPProblem
  - Same Ax <= b
  - Same bounds
  - Same objective
  - All variables continuous
*/

struct LPRelaxation {
    int num_rows = 0;
    int num_cols = 0;

    // objective: min c^T x
    std::vector<double> c;
    double obj_offset = 0.0;
    // bounds
    std::vector<double> lb;
    std::vector<double> ub;

    // constraints Ax <= b
    std::vector<double> b;

    // CSR representation
    std::vector<int> csr_row_ptr;
    std::vector<int> csr_col_idx;
    std::vector<double> csr_val;

    // CSC representation
    std::vector<int> csc_col_ptr;
    std::vector<int> csc_row_idx;
    std::vector<double> csc_val;

    // ---- Solution ----
    std::vector<double> x;                 // best primal solution
    double obj_value = std::numeric_limits<double>::infinity();

    LPRelaxation() = default;

    // build from MIP
    explicit LPRelaxation(const MIPProblem& mip);

    void build_from_mip(const MIPProblem& mip);

    // Solve LP relaxation
    // Returns true if a feasible solution is found
    bool solve();
};

#endif

