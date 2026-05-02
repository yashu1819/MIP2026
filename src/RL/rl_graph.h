#ifndef RL_GRAPH_H
#define RL_GRAPH_H

#include "../mip_problem.h"
#include <vector>
#include <unordered_map>

namespace rl_sph {

// Bipartite graph representation of ILP
// Variables and constraints as two node sets
struct BipartiteGraph {
    int num_vars;
    int num_constraints;

    // Adjacency: constraint -> variables (which vars appear in each constraint)
    std::vector<std::vector<int> > constraint_to_vars;

    // Adjacency: variable -> constraints (which constraints each var appears in)
    std::vector<std::vector<int> > var_to_constraints;

    // Constraint matrix info (scaled)
    std::vector<double> scaled_A_row;  // flattened CSR values
    std::vector<int> A_row_ptr;        // CSR row pointers
    std::vector<int> A_col_idx;        // CSR column indices

    // Variable-constraint incidence (sparse)
    std::vector<std::pair<int, int> > edges;  // (var_idx, constraint_idx)
};

// Build bipartite graph from MIP problem
BipartiteGraph build_graph(const MIPProblem& mip);

// Get variable indices that appear in a constraint
inline const std::vector<int>& get_vars_in_constraint(
    const BipartiteGraph& graph,
    int constraint_idx
) {
    return graph.constraint_to_vars[constraint_idx];
}

// Get constraint indices that a variable appears in
inline const std::vector<int>& get_constraints_for_var(
    const BipartiteGraph& graph,
    int var_idx
) {
    return graph.var_to_constraints[var_idx];
}

// Scale the constraint matrix to [-1, 1] (equilibration scaling)
std::vector<double> scale_constraint_matrix(
    const MIPProblem& mip,
    std::vector<double>& A_scaled  // output
);

} // namespace rl_sph

#endif // RL_GRAPH_H
