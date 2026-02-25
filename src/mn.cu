#include "fp.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    // ── Toy binary MIP example ───────────────────────────────────────
    // min  3x₁ + 4x₂ + 5x₃
    // s.t.  x₁ + x₂ + x₃  =  2
    //       2x₁ +     x₃  =  3
    //       0 ≤ x ≤ 1    integer

    cuopt_int_t n_vars = 3;
    cuopt_int_t n_cons = 2;

    double c[] = {3.0, 4.0, 5.0};

    // CSR: row0: 1 1 1
    //      row1: 2 0 1
    double      A_values[]    = {1,1,1,  2,1};
    cuopt_int_t A_col_indices[] = {0,1,2,  0,2};
    cuopt_int_t A_row_ptr[]   = {0, 3, 5};

    double b[] = {2.0, 3.0};

    double lb[] = {0,0,0};
    double ub[] = {1,1,1};

    cuopt_int_t var_types[] = {1,1,1};   // all integer (binary here)

    FPResult r = run_feasibility_pump_cuda(
        n_vars, n_cons, c,
        A_values, A_col_indices, A_row_ptr, b,
        lb, ub, var_types,
        100,          // max pump iterations
        5.0           // sec per LP solve
    );

    if (r.found_feasible) {
        printf("\nSUCCESS — Feasible integer solution found!\n");
        printf("Objective: %.2f\nx = [ ", r.objective_value);
        for (size_t i = 0; i < r.n_vars; i++) {
            printf("%.0f ", r.solution[i]);
        }
        printf("]\n");
        free(r.solution);
    } else {
        printf("\n%s\n", r.status_msg);
    }

    return 0;
}
