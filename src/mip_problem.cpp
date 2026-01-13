#include "mip_problem.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <limits>

MIPProblem::MIPProblem() {}

void MIPProblem::ensure_col(int col)
{
    if (col >= num_cols) {
        num_cols = col + 1;
        c.resize(num_cols, 0.0);
        lb.resize(num_cols, -std::numeric_limits<double>::infinity());
        ub.resize(num_cols,  std::numeric_limits<double>::infinity());
        vartype.resize(num_cols, VarType::CONTINUOUS);
    }
}

void MIPProblem::add_row_sparse(
    const std::vector<std::pair<int,double>>& entries,
    char sense,
    double rhs)
{
    auto add_le = [&](const std::vector<std::pair<int,double>>& e, double r) {
        int row = (int)b.size();
        for (auto &p : e) {
            ensure_col(p.first);
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
        for (auto &p : entries)
            neg.push_back({p.first, -p.second});
        add_le(neg, -rhs);
    }
    else if (sense == 'E') {
        add_le(entries, rhs);
        std::vector<std::pair<int,double>> neg;
        neg.reserve(entries.size());
        for (auto &p : entries)
            neg.push_back({p.first, -p.second});
        add_le(neg, -rhs);
    }
    else {
        throw std::runtime_error("Unknown constraint sense");
    }
}

void MIPProblem::load_from_mps(const std::string& filename)
{
    std::ifstream in(filename);
    if (!in)
        throw std::runtime_error("Cannot open MPS file");

    enum Section { NONE, ROWS, COLUMNS, RHS, BOUNDS };
    Section sec = NONE;

    std::unordered_map<std::string,int> row_id;
    std::unordered_map<std::string,int> col_id;

    std::string obj_row;

    struct RowData {
        std::vector<std::pair<int,double>> entries;
        char sense;
        double rhs = 0.0;
    };

    std::vector<RowData> rows;

    bool in_integer_block = false;

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '*')
            continue;

        std::stringstream ss(line);
        std::string tok;
        ss >> tok;

        if (tok == "ROWS")    { sec = ROWS;    continue; }
        if (tok == "COLUMNS") { sec = COLUMNS; continue; }
        if (tok == "RHS")     { sec = RHS;     continue; }
        if (tok == "BOUNDS")  { sec = BOUNDS;  continue; }
        if (tok == "ENDATA")  break;

        /* ---------- ROWS ---------- */
        if (sec == ROWS) {
            char s = tok[0];
            std::string rname;
            ss >> rname;

            if (s == 'N') {
                obj_row = rname;
            } else {
                int id = (int)rows.size();
                row_id[rname] = id;
                rows.push_back(RowData());
                rows.back().sense = s;
            }
        }

        /* ---------- COLUMNS ---------- */
        else if (sec == COLUMNS) {

            // MARKER handling
            if (tok == "MARKER") {
                std::string marker;
                ss >> marker;
                if (marker.find("INTORG") != std::string::npos)
                    in_integer_block = true;
                else if (marker.find("INTEND") != std::string::npos)
                    in_integer_block = false;
                continue;
            }

            std::string cname = tok;

            if (!col_id.count(cname)) {
                int id = num_cols;
                col_id[cname] = id;
                ensure_col(id);
            }
            int cid = col_id[cname];

            // MARKER integer block
            if (in_integer_block && vartype[cid] == VarType::CONTINUOUS)
                vartype[cid] = VarType::INTEGER;

            std::string r1;
            double v1;

            // first pair
            if (ss >> r1 >> v1) {
                if (r1 == obj_row) {
                    c[cid] = v1;
                } else {
                    auto it = row_id.find(r1);
                    if (it != row_id.end())
                        rows[it->second].entries.push_back({cid, v1});
                }
            }

            // optional second pair
            if (ss >> r1 >> v1) {
                if (r1 == obj_row) {
                    c[cid] = v1;
                } else {
                    auto it = row_id.find(r1);
                    if (it != row_id.end())
                        rows[it->second].entries.push_back({cid, v1});
                }
            }
        }

        /* ---------- RHS ---------- */
        else if (sec == RHS) {
            std::string rhsname = tok;
            (void)rhsname;

            std::string r1;
            double v1;

            if (ss >> r1 >> v1) {
                auto it = row_id.find(r1);
                if (it != row_id.end())
                    rows[it->second].rhs = v1;
            }

            if (ss >> r1 >> v1) {
                auto it = row_id.find(r1);
                if (it != row_id.end())
                    rows[it->second].rhs = v1;
            }
        }

        /* ---------- BOUNDS ---------- */
        else if (sec == BOUNDS) {
            std::string btype = tok;
            std::string bname, cname;
            double val = 0.0;

            ss >> bname >> cname;
            if (ss >> val) {}

            if (!col_id.count(cname)) {
                int id = num_cols;
                col_id[cname] = id;
                ensure_col(id);
            }
            int cid = col_id[cname];

            if (btype == "LO") {
                lb[cid] = val;
            }
            else if (btype == "UP") {
                ub[cid] = val;
            }
            else if (btype == "FX") {
                lb[cid] = val;
                ub[cid] = val;
            }
            else if (btype == "BV") {
                lb[cid] = 0.0;
                ub[cid] = 1.0;
                vartype[cid] = VarType::BINARY;
            }
            else if (btype == "LI") {
                lb[cid] = val;
                vartype[cid] = VarType::INTEGER;
            }
            else if (btype == "UI") {
                ub[cid] = val;
                vartype[cid] = VarType::INTEGER;
            }
        }
    }

    /* ---------- Convert all rows to Ax <= b ---------- */
    for (auto &r : rows)
        add_row_sparse(r.entries, r.sense, r.rhs);
}

void MIPProblem::finalize()
{
    int nnz = (int)coo_val.size();

    // CSR
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

    // CSC
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

