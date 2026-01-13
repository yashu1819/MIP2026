#ifndef MIP_PROBLEM_H
#define MIP_PROBLEM_H

#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>

enum class VarType : uint8_t { CONTINUOUS = 0, INTEGER = 1, BINARY = 2 };

struct MIPProblem {
    int num_rows = 0;   // constraints after Ax<=b conversion
    int num_cols = 0;   // variables

    // objective
    std::vector<double> c;

    // bounds
    std::vector<double> lb;
    std::vector<double> ub;

    // variable types
    std::vector<VarType> vartype;

    // Ax <= b
    std::vector<double> b;

    // COO (build-time)
    std::vector<int> coo_row;
    std::vector<int> coo_col;
    std::vector<double> coo_val;

    // CSR
    std::vector<int> csr_row_ptr;
    std::vector<int> csr_col_idx;
    std::vector<double> csr_val;

    // CSC
    std::vector<int> csc_col_ptr;
    std::vector<int> csc_row_idx;
    std::vector<double> csc_val;

    // ---- API ----
    MIPProblem();

    // load from MPS file (core, no SOS / quadratic)
    void load_from_mps(const std::string& filename);

    // build CSR + CSC
    void finalize();

private:
    // helpers
    void ensure_col(int col);
    void add_row_sparse(const std::vector<std::pair<int,double>>& entries, char sense, double rhs);
};

#endif

