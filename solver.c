 #include <stdio.h>

#include <stdlib.h>

#include <cuopt/linear_programming/cuopt_c.h>


int main(int argc, char* argv[]) {

    if (argc < 2) {

        printf("Usage: %s <input_mps_file> [output_file]\n", argv[0]);

        return 1;

    }

    const char* mps_file = argv[1];

    const char* output_file = (argc >= 3 ? argv[2] : "solution.txt");


    cuOptOptimizationProblem problem = NULL;

    cuOptSolverSettings settings = NULL;

    cuOptSolution solution = NULL;

    cuopt_int_t status;


    // Create the problem from MPS file

    status = cuOptReadProblem(mps_file, &problem);

    if (status != CUOPT_SUCCESS) {

        fprintf(stderr, "Error reading MPS file: %d\n", status);

        return 1;

    }


    // Create solver settings

    status = cuOptCreateSolverSettings(&settings);

    if (status != CUOPT_SUCCESS) {

        fprintf(stderr, "Error creating solver settings: %d\n", status);

        cuOptDestroyProblem(&problem);

        return 1;

    }


    // Use PDLP method (GPU-accelerated first-order LP solver)

    status = cuOptSetIntegerParameter(settings, CUOPT_METHOD, CUOPT_METHOD_PDLP);

    if (status != CUOPT_SUCCESS) {

        fprintf(stderr, "Error setting solver method: %d\n", status);

        goto CLEANUP;

    }
    status = cuOptSetFloatParameter(settings, CUOPT_ABSOLUTE_PRIMAL_TOLERANCE, 0.000001);

  if (status != CUOPT_SUCCESS) {

    printf("Error setting optimality tolerance: %d\n", status);

    goto CLEANUP;

  }

    // Set a 180-second time limit

    status = cuOptSetFloatParameter(settings, CUOPT_TIME_LIMIT, 180.0f);

    if (status != CUOPT_SUCCESS) {

        fprintf(stderr, "Error setting time limit: %d\n", status);

        goto CLEANUP;

    }


    // Ensure PDLP saves the best primal solution found so far

    status = cuOptSetIntegerParameter(settings, CUOPT_SAVE_BEST_PRIMAL_SO_FAR, 1);

    if (status != CUOPT_SUCCESS) {

        fprintf(stderr, "Error setting save-best-solution flag: %d\n", status);

        goto CLEANUP;

    }


    // Solve the problem

    status = cuOptSolve(problem, settings, &solution);

    if (status != CUOPT_SUCCESS) {

        fprintf(stderr, "Error solving problem: %d\n", status);

        goto CLEANUP;

    }


    // (Optional) get termination status for logging

    cuopt_int_t term_status = 0;

    cuOptGetTerminationStatus(solution, &term_status);


    // Get objective value of the (best) solution

    cuopt_float_t objective_value = 0.0f;

    status = cuOptGetObjectiveValue(solution, &objective_value);

    if (status != CUOPT_SUCCESS) {

        fprintf(stderr, "Error getting objective value: %d\n", status);

        goto CLEANUP;

    }


    // Retrieve number of variables to size the solution array

    cuopt_int_t num_vars = 0;

    status = cuOptGetNumVariables(problem, &num_vars);

    if (status != CUOPT_SUCCESS) {

        fprintf(stderr, "Error getting number of variables: %d\n", status);

        goto CLEANUP;

    }


    // Allocate array and get primal solution (variable values)

    cuopt_float_t* x = (cuopt_float_t*) malloc(num_vars * sizeof(cuopt_float_t));

    if (!x) {

        fprintf(stderr, "Memory allocation failed\n");

        goto CLEANUP;

    }

    status = cuOptGetPrimalSolution(solution, x);

    if (status != CUOPT_SUCCESS) {

        fprintf(stderr, "Error getting solution values: %d\n", status);

        free(x);

        goto CLEANUP;

    }


    // Write objective and solution values to the output file

    FILE* fout = fopen(output_file, "w");

    if (!fout) {

        fprintf(stderr, "Error opening output file\n");

        free(x);

        goto CLEANUP;

    }

    fprintf(fout, "Objective = %f\n", objective_value);

    for (cuopt_int_t i = 0; i < num_vars; i++) {

        fprintf(fout, "x%d = %f\n", i+1, x[i]);

    }

    fclose(fout);

    free(x);


    printf("Solve completed (status %d). Objective = %f. Solution written to %s\n",

           term_status, objective_value, output_file);


CLEANUP:

    cuOptDestroySolution(&solution);

    cuOptDestroySolverSettings(&settings);

    cuOptDestroyProblem(&problem);

    return (status == CUOPT_SUCCESS ? 0 : 1);

} 

