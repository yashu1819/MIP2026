/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuopt/linear_programming/cuopt_c.h>
#include <stdio.h>
#include <stdlib.h>

const char* termination_status_to_string(cuopt_int_t termination_status)
{
    switch (termination_status) {
        case CUOPT_TERIMINATION_STATUS_OPTIMAL:
            return "Optimal";
        case CUOPT_TERIMINATION_STATUS_INFEASIBLE:
            return "Infeasible";
        case CUOPT_TERIMINATION_STATUS_UNBOUNDED:
            return "Unbounded";
        case CUOPT_TERIMINATION_STATUS_ITERATION_LIMIT:
            return "Iteration limit";
        case CUOPT_TERIMINATION_STATUS_TIME_LIMIT:
            return "Time limit";
        case CUOPT_TERIMINATION_STATUS_NUMERICAL_ERROR:
            return "Numerical error";
        case CUOPT_TERIMINATION_STATUS_PRIMAL_FEASIBLE:
            return "Primal feasible";
        case CUOPT_TERIMINATION_STATUS_FEASIBLE_FOUND:
            return "Feasible found"; // This is likely what we will hit
        default:
            return "Unknown";
    }
}

cuopt_int_t solve_mps_file(const char* filename)
{
    cuOptOptimizationProblem problem = NULL;
    cuOptSolverSettings settings = NULL;
    cuOptSolution solution = NULL;
    cuopt_int_t status;
    cuopt_float_t time;
    cuopt_int_t termination_status;
    cuopt_float_t objective_value;
    cuopt_int_t num_variables;
    cuopt_float_t* solution_values = NULL;

    printf("Reading and solving MPS file: %s\n", filename);

    // 1. Create the problem from MPS file
    status = cuOptReadProblem(filename, &problem);
    if (status != CUOPT_SUCCESS) {
        printf("Error creating problem from MPS file: %d\n", status);
        goto DONE;
    }

    status = cuOptGetNumVariables(problem, &num_variables);
    if (status != CUOPT_SUCCESS) {
        printf("Error getting number of variables: %d\n", status);
        goto DONE;
    }

    // 2. Create solver settings
    status = cuOptCreateSolverSettings(&settings);
    if (status != CUOPT_SUCCESS) {
        printf("Error creating solver settings: %d\n", status);
        goto DONE;
    }

    // --- KEY PARAMETERS FOR FEASIBILITY ONLY ---
    
    // Set solve mode to Heuristic: this tells cuOpt to focus on finding a solution 
    // rather than proving optimality.
    cuOptSetIntParameter(settings, CUOPT_SOLVE_MODE, CUOPT_SOLVE_MODE_HEURISTIC);
    
    // Set a very loose MIP gap. If a solution is found within 99% of the best bound, stop.
    cuOptSetFloatParameter(settings, CUOPT_RELATIVE_MIP_GAP, 0.99);

    // Optional: Set a time limit in case the problem is extremely hard to even satisfy
    // cuOptSetFloatParameter(settings, CUOPT_TIME_LIMIT, 60.0); 

    // -------------------------------------------

    // 3. Solve the problem
    status = cuOptSolve(problem, settings, &solution);
    if (status != CUOPT_SUCCESS) {
        printf("Error solving problem: %d\n", status);
        goto DONE;
    }

    // 4. Get solution information
    status = cuOptGetSolveTime(solution, &time);
    status = cuOptGetTerminationStatus(solution, &termination_status);
    status = cuOptGetObjectiveValue(solution, &objective_value);

    // Print results
    printf("\nResults:\n");
    printf("--------\n");
    printf("Number of variables: %d\n", num_variables);
    printf("Termination status: %s (%d)\n", termination_status_to_string(termination_status), termination_status);
    printf("Solve time: %f seconds\n", time);
    
    // Only print values if a feasible solution exists
    if (termination_status == CUOPT_TERIMINATION_STATUS_FEASIBLE_FOUND || 
        termination_status == CUOPT_TERIMINATION_STATUS_PRIMAL_FEASIBLE ||
        termination_status == CUOPT_TERIMINATION_STATUS_OPTIMAL) {
        
        printf("Objective value: %f\n", objective_value);

        solution_values = (cuopt_float_t*)malloc(num_variables * sizeof(cuopt_float_t));
        status = cuOptGetPrimalSolution(solution, solution_values);
        
        printf("\nSolution found! First few variables:\n");
        int print_limit = (num_variables > 10) ? 10 : num_variables;
        for (cuopt_int_t i = 0; i < print_limit; i++) {
            printf("x%d = %f\n", i + 1, solution_values[i]);
        }
    } else {
        printf("No feasible solution was found in heuristic mode.\n");
    }

DONE:
    if (solution_values) free(solution_values);
    if (problem) cuOptDestroyProblem(&problem);
    if (settings) cuOptDestroySolverSettings(&settings);
    if (solution) cuOptDestroySolution(&solution);

    return status;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <mps_file_path>\n", argv[0]);
        return 1;
    }

    cuopt_int_t status = solve_mps_file(argv[1]);
    return (status == CUOPT_SUCCESS) ? 0 : 1;
}
