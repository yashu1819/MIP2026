#include "lp_relaxation.h"

#include <cuopt/linear_programming/cuopt_c.h>
#include <stdexcept>
#include <limits>

LPRelaxation::LPRelaxation(const MIPProblem& mip) {
    build_from_mip(mip);
}

void LPRelaxation::build_from_mip(const MIPProblem& mip)
{
    num_rows = mip.num_rows;
    num_cols = mip.num_cols;

    c  = mip.c;
    lb = mip.lb;
    ub = mip.ub;
    b  = mip.b;
    obj_offset=-1* mip.obj_offset;
    csr_row_ptr = mip.csr_row_ptr;
    
    csr_col_idx = mip.csr_col_idx;
  
    csr_val     = mip.csr_val;

    csc_col_ptr = mip.csc_col_ptr;
    csc_row_idx = mip.csc_row_idx;
    csc_val     = mip.csc_val;

    x.assign(num_cols, 0.0);
    obj_value = std::numeric_limits<double>::infinity();
}

bool LPRelaxation::solve()
{ std::vector<cuopt_int_t> csr_row_ptr_cast(csr_row_ptr.begin(), csr_row_ptr.end());std::vector<cuopt_int_t> csr_col_idx_cast(csr_col_idx.begin(), csr_col_idx.end());
    cuOptOptimizationProblem problem = nullptr;
    cuOptSolverSettings settings = nullptr;
    cuOptSolution solution = nullptr;

    // cuOpt expects ranged constraints: lb <= A x <= ub
    // Our model is Ax <= b  ==>  -inf <= Ax <= b
    std::vector<cuopt_float_t> constr_lb(num_rows, -CUOPT_INFINITY);
    std::vector<cuopt_float_t> constr_ub(num_rows);
    for (int i = 0; i < num_rows; ++i)
        constr_ub[i] = static_cast<cuopt_float_t>(b[i]);

    // variable types: all continuous (LP relaxation)
    std::vector<char> var_types(num_cols, CUOPT_CONTINUOUS);

    cuopt_int_t status;

    status = cuOptCreateRangedProblem(
        num_rows,
        num_cols,
        CUOPT_MINIMIZE,
        0.0, // objective offset
        reinterpret_cast<const cuopt_float_t*>(c.data()),
        csr_row_ptr_cast.data(),
        csr_col_idx_cast.data(),
        reinterpret_cast<const cuopt_float_t*>(csr_val.data()),
        constr_lb.data(),
        constr_ub.data(),
        reinterpret_cast<const cuopt_float_t*>(lb.data()),
        reinterpret_cast<const cuopt_float_t*>(ub.data()),
        var_types.data(),
        &problem);

    
    status = cuOptCreateSolverSettings(&settings);
   
    // Disable cuOpt output
    cuOptSetIntegerParameter(settings, CUOPT_LOG_TO_CONSOLE, 0);

    // PDLP solver
    cuOptSetIntegerParameter(settings, CUOPT_METHOD,CUOPT_METHOD_PDLP);
    cuOptSetFloatParameter(settings, CUOPT_ABSOLUTE_PRIMAL_TOLERANCE, 1e-6);
    // setting time limit if needed
    // status = cuOptSetFloatParameter(settings, CUOPT_TIME_LIMIT, 50.0f);
   
    //solve problem 
    status = cuOptSolve(problem, settings, &solution);
    

    cuopt_float_t obj;
    status = cuOptGetObjectiveValue(solution, &obj);
    
    obj_value = obj+ obj_offset  ;

    std::vector<cuopt_float_t> sol(num_cols);
    status = cuOptGetPrimalSolution(solution, sol.data());
    
    for (int j = 0; j < num_cols; ++j)
        x[j] = sol[j];

    cuOptDestroyProblem(&problem);
    cuOptDestroySolverSettings(&settings);
    cuOptDestroySolution(&solution);

    return true;
}

