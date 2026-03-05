#include "locks.h"
#include "locks_kernel.cuh"

#include <cuda_runtime.h>
#include <iostream>

VariableLocks compute_variable_locks_gpu(const MIPProblem& mip)
{
    VariableLocks locks;

    int n = mip.num_cols;
    int nnz = mip.csc_val.size();

    locks.up_locks.resize(n);
    locks.down_locks.resize(n);

    // Device pointers
    int *d_col_ptr, *d_row_idx;
    double *d_val;

    int *d_up_locks, *d_down_locks;

    cudaMalloc(&d_col_ptr, (n+1)*sizeof(int));
    cudaMalloc(&d_row_idx, nnz*sizeof(int));
    cudaMalloc(&d_val, nnz*sizeof(double));

    cudaMalloc(&d_up_locks, n*sizeof(int));
    cudaMalloc(&d_down_locks, n*sizeof(int));

    // copy data
    cudaMemcpy(d_col_ptr, mip.csc_col_ptr.data(), (n+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_idx, mip.csc_row_idx.data(), nnz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, mip.csc_val.data(), nnz*sizeof(double), cudaMemcpyHostToDevice);

    // launch kernel
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    compute_locks_kernel<<<grid_size, block_size>>>(
        n,
        d_col_ptr,
        d_row_idx,
        d_val,
        d_up_locks,
        d_down_locks
    );

    cudaDeviceSynchronize();

    // copy back
    cudaMemcpy(locks.up_locks.data(), d_up_locks, n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(locks.down_locks.data(), d_down_locks, n*sizeof(int), cudaMemcpyDeviceToHost);

    // cleanup
    cudaFree(d_col_ptr);
    cudaFree(d_row_idx);
    cudaFree(d_val);
    cudaFree(d_up_locks);
    cudaFree(d_down_locks);

    return locks;
}