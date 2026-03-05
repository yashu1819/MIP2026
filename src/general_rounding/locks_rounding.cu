#include "locks_rounding.h"
#include "locks_rounding_kernel.cuh"

#include <cuda_runtime.h>
#include <iostream>

Solution lock_guided_rounding_gpu(
    const MIPProblem& mip,
    const std::vector<double>& fractional_solution,
    const VariableLocks& locks
)
{
    int n = mip.num_cols;

    Solution sol;
    sol.x.resize(n);

    //--------------------------------------------------
    // Build integer mask from vartype
    //--------------------------------------------------

    std::vector<int> is_integer(n);

    for(int j = 0; j < n; j++)
    {
        if(mip.vartype[j] == VarType::CONTINUOUS)
            is_integer[j] = 0;
        else
            is_integer[j] = 1;
    }

    //--------------------------------------------------
    // Device memory
    //--------------------------------------------------

    double *d_xfrac, *d_lb, *d_ub, *d_xout;
    int *d_up, *d_down, *d_int;

    cudaMalloc(&d_xfrac, n*sizeof(double));
    cudaMalloc(&d_lb, n*sizeof(double));
    cudaMalloc(&d_ub, n*sizeof(double));
    cudaMalloc(&d_xout, n*sizeof(double));

    cudaMalloc(&d_up, n*sizeof(int));
    cudaMalloc(&d_down, n*sizeof(int));
    cudaMalloc(&d_int, n*sizeof(int));

    //--------------------------------------------------
    // Copy data to GPU
    //--------------------------------------------------

    cudaMemcpy(d_xfrac, fractional_solution.data(), n*sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_lb, mip.lb.data(), n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ub, mip.ub.data(), n*sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(d_up, locks.up_locks.data(), n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_down, locks.down_locks.data(), n*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_int, is_integer.data(), n*sizeof(int), cudaMemcpyHostToDevice);

    //--------------------------------------------------
    // Launch kernel
    //--------------------------------------------------

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    lock_round_kernel<<<grid_size, block_size>>>(
        n,
        d_xfrac,
        d_up,
        d_down,
        d_lb,
        d_ub,
        d_int,
        d_xout
    );

    cudaDeviceSynchronize();

    //--------------------------------------------------
    // Copy result back
    //--------------------------------------------------

    cudaMemcpy(sol.x.data(), d_xout, n*sizeof(double), cudaMemcpyDeviceToHost);

    //--------------------------------------------------
    // Cleanup
    //--------------------------------------------------

    cudaFree(d_xfrac);
    cudaFree(d_lb);
    cudaFree(d_ub);
    cudaFree(d_xout);
    cudaFree(d_up);
    cudaFree(d_down);
    cudaFree(d_int);

    sol.feasible = false;

    return sol;
}