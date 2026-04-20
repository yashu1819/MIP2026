#include "rl_graph.h"
#include <cmath>
#include <algorithm>

#include "../Definition/mip_problem.h"

namespace rl_sph {

// Build bipartite graph from MIP problem
BipartiteGraph build_graph(const MIPProblem& mip) {
    BipartiteGraph graph;
    graph.num_vars = mip.num_cols;
    graph.num_constraints = mip.num_rows;

    graph.constraint_to_vars.resize(mip.num_rows);
    graph.var_to_constraints.resize(mip.num_cols);

    // Build adjacency from CSR representation
    for (int i = 0; i < mip.num_rows; ++i) {
        for (int j = mip.csr_row_ptr[i]; j < mip.csr_row_ptr[i + 1]; ++j) {
            int col = mip.csr_col_idx[j];
            graph.constraint_to_vars[i].push_back(col);
            graph.var_to_constraints[col].push_back(i);
            graph.edges.push_back({col, i});  // (var, constraint)
        }
    }

    // Store scaled constraint matrix
    graph.scaled_A_row = mip.csr_val;
    graph.A_row_ptr = mip.csr_row_ptr;
    graph.A_col_idx = mip.csr_col_idx;

    return graph;
}

// Scale constraint matrix to [-1, 1] using equilibration scaling
std::vector<double> scale_constraint_matrix(
    const MIPProblem& mip,
    std::vector<double>& A_scaled
) {
    A_scaled.resize(mip.csr_val.size());

    // For each constraint, find max absolute coefficient
    std::vector<double> scale_factor(mip.num_rows, 1.0);
    for (int i = 0; i < mip.num_rows; ++i) {
        double max_val = 0.0;
        for (int j = mip.csr_row_ptr[i]; j < mip.csr_row_ptr[i + 1]; ++j) {
            max_val = std::max(max_val, std::abs(mip.csr_val[j]));
        }
        if (max_val > 1e-10) {
            scale_factor[i] = 1.0 / max_val;
        }
    }

    // Apply scaling
    for (int i = 0; i < mip.num_rows; ++i) {
        for (int j = mip.csr_row_ptr[i]; j < mip.csr_row_ptr[i + 1]; ++j) {
            A_scaled[j] = mip.csr_val[j] * scale_factor[i];
        }
    }

    return scale_factor;
}

} // namespace rl_sph
