#include "mip_problem.h"
#include <cmath>
#include <iostream>

#include <coin/OsiClpSolverInterface.hpp>
#include <coin/CoinPackedMatrix.hpp>
#include <coin/ClpSimplex.hpp>
#include <stdexcept>
#include <limits>

MIPProblem::MIPProblem()
    : num_rows(0), num_cols(0) {}

/* ---------------- helpers ---------------- */

void MIPProblem::ensure_col(int col)
{
    if (col >= num_cols) {
        num_cols = col + 1;
        c.resize(num_cols, 0.0);
        lb.resize(num_cols, 0.0);
        ub.resize(num_cols, std::numeric_limits<double>::infinity());
        vartype.resize(num_cols, VarType::CONTINUOUS);
    }
}

void MIPProblem::add_row_sparse(
    const std::vector<std::pair<int,double>>& entries,
    char sense,
    double rhs)
{
    auto add_le = [&](const std::vector<std::pair<int,double>>& e, double r) {
        int row = num_rows;
        for (auto& p : e) {
            coo_row.push_back(row);
            coo_col.push_back(p.first);
            coo_val.push_back(p.second);
        }
        b.push_back(r);
        num_rows++;
    };

    if (sense == 'L') {
        add_le(entries, rhs);
    }
    else if (sense == 'G') {
        std::vector<std::pair<int,double>> neg;
        neg.reserve(entries.size());
        for (auto& p : entries)
            neg.emplace_back(p.first, -p.second);
        add_le(neg, -rhs);
    }
    else if (sense == 'E') {
        add_le(entries, rhs);
        std::vector<std::pair<int,double>> neg;
        neg.reserve(entries.size());
        for (auto& p : entries)
            neg.emplace_back(p.first, -p.second);
        add_le(neg, -rhs);
    }
}

/* ---------------- MPS loader (Coin-OR) ---------------- */

void MIPProblem::load_from_mps(const std::string& filename)
{
    OsiClpSolverInterface solver;

    if (solver.readMps(filename.c_str()) != 0) {
        throw std::runtime_error("Coin-OR failed to read MPS file");
    }

    const int n = solver.getNumCols();
    const int m = solver.getNumRows();

    num_cols = n;
    num_rows = 0;
    
   const auto& colNames = solver.getColNames();
    var_names = colNames;  // store in your class (std::vector<std::string>)



    /* ---- objective ---- */
    c.assign(n, 0.0);
    const double* obj = solver.getObjCoefficients();
    for (int j = 0; j < n; ++j)
        c[j] = obj[j];
   ClpSimplex* model = solver.getModelPtr();
    obj_offset = model->objectiveOffset();
    /* ---- bounds ---- */
    lb.assign(n, 0.0);
    ub.assign(n, std::numeric_limits<double>::infinity());
    vartype.assign(n, VarType::CONTINUOUS);

    const double* col_lb = solver.getColLower();
    const double* col_ub = solver.getColUpper();

    for (int j = 0; j < n; ++j) {
        lb[j] = col_lb[j];
        ub[j] = col_ub[j];
    }

    /* ---- variable types ---- */
    for (int j = 0; j < n; ++j) {
        if (solver.isBinary(j))
            vartype[j] = VarType::BINARY;
        else if (solver.isInteger(j))
            vartype[j] = VarType::INTEGER;
    }

    /* ---- constraints ---- */
    const CoinPackedMatrix* A = solver.getMatrixByRow();
    const double* row_lb = solver.getRowLower();
    const double* row_ub = solver.getRowUpper();

    const double INF = solver.getInfinity();

    for (int i = 0; i < m; ++i) {
        CoinShallowPackedVector row = A->getVector(i);

        std::vector<std::pair<int,double>> entries;
        entries.reserve(row.getNumElements());

        for (int k = 0; k < row.getNumElements(); ++k) {
            entries.emplace_back(
                row.getIndices()[k],
                row.getElements()[k]
            );
        }

        // upper bound: Ax <= ub
        if (row_ub[i] < INF) {
            add_row_sparse(entries, 'L', row_ub[i]);
        }

        // lower bound: Ax >= lb  ->  -Ax <= -lb
        if (row_lb[i] > -INF) {
            std::vector<std::pair<int,double>> neg;
            neg.reserve(entries.size());
            for (auto& p : entries)
                neg.emplace_back(p.first, -p.second);
            add_row_sparse(neg, 'L', -row_lb[i]);
        }
    }
}

/* ---------------- finalize CSR / CSC ---------------- */

void MIPProblem::finalize()
{
    const int nnz = (int)coo_val.size();

    /* ---- CSR ---- */
    csr_row_ptr.assign(num_rows + 1, 0);
    csr_col_idx.assign(nnz, 0);
    csr_val.assign(nnz, 0.0);

    for (int r : coo_row)
        csr_row_ptr[r + 1]++;

    for (int i = 1; i <= num_rows; ++i)
        csr_row_ptr[i] += csr_row_ptr[i - 1];

    std::vector<int> pos = csr_row_ptr;
    for (int k = 0; k < nnz; ++k) {
        int r = coo_row[k];
        int d = pos[r]++;
        csr_col_idx[d] = coo_col[k];
        csr_val[d] = coo_val[k];
    }

    /* ---- CSC ---- */
    csc_col_ptr.assign(num_cols + 1, 0);
    csc_row_idx.assign(nnz, 0);
    csc_val.assign(nnz, 0.0);

    for (int c0 : coo_col)
        csc_col_ptr[c0 + 1]++;

    for (int j = 1; j <= num_cols; ++j)
        csc_col_ptr[j] += csc_col_ptr[j - 1];

    pos = csc_col_ptr;
    for (int k = 0; k < nnz; ++k) {
        int c0 = coo_col[k];
        int d = pos[c0]++;
        csc_row_idx[d] = coo_row[k];
        csc_val[d] = coo_val[k];
    }
}
bool MIPProblem::check_feasible(
    const std::vector<double>& x,
    double constr_tol,
    double int_tol
) const {
    // Dimension check
    if ((int)x.size() != num_cols) {
    //    std::cerr << "Feasibility check failed: wrong dimension\n";
        return false;
    }

    /* ================= Bounds ================= */

    for (int j = 0; j < num_cols; ++j) {
        if (x[j] < lb[j] - constr_tol || x[j] > ub[j] + constr_tol) {
      //      std::cerr << "Bound violation at var " << j << "\n";
          return false;
        }
    }

    /* ================= Integrality ================= */

    for (int j = 0; j < num_cols; ++j) {
        if (vartype[j] == VarType::INTEGER ||
            vartype[j] == VarType::BINARY) {

            double r = std::round(x[j]);
            if (std::fabs(x[j] - r) > int_tol) {
       //         std::cerr << "Integrality violation at var " << j << "\n";
                return false;
            }

            if (vartype[j] == VarType::BINARY) {
                if (!(r == 0.0 || r == 1.0)) {
         //           std::cerr << "Binary violation at var " << j << "\n";
                    return false;
                }
            }
        }
    }

    /* ================= Constraints: Ax <= b ================= */

    // Using CSR: row-wise accumulation
    for (int i = 0; i < num_rows; ++i) {
        double activity = 0.0;
        for (int p = csr_row_ptr[i]; p < csr_row_ptr[i + 1]; ++p) {
            activity += csr_val[p] * x[csr_col_idx[p]];
        }

        if (activity > b[i] + constr_tol) {
           // std::cerr << "Constraint violation at row " << i
             //         << " : activity = " << activity
               //       << " , rhs = " << b[i] << "\n";
            return false;
        }
    }

    return true;
}
double getTime() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    // Convert to seconds as a double
    return std::chrono::duration<double>(duration).count();
}

std::string MIPProblem::classify() {

    bool hasBinary = false;
    bool hasInteger = false;
    bool hasContinuous = false;

    for (VarType vt : vartype) {
        if (vt == VarType::BINARY)
            hasBinary = true;
        else if (vt == VarType::INTEGER)
            hasInteger = true;
        else if (vt == VarType::CONTINUOUS)
            hasContinuous = true;
    }

    // Rules:
    // Pure BP  -> only binary
    // Pure IP  -> at least 1 integer, no continuous (binary allowed)
    // MBP      -> binary + continuous, no integer
    // MILP     -> integer + continuous

    if (hasInteger) {
        if (hasContinuous)
            return "MILP";
        else
            return "Pure IP";
    }

    if (hasBinary && hasContinuous)
        return "MBP";

    if (hasBinary && !hasContinuous)
        return "Pure BP";

    return "LP";
}
