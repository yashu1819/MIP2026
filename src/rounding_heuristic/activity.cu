#include "activity.h"
#include "activity_kernel.cuh"

#include <cuda_runtime.h>

void compute_constraint_activity_gpu(
    const MIPProblem& mip,
    const std::vector<double>& x,
    std::vector<double>& activity
)
{
    int m = mip.num_rows;
    int n = mip.num_cols;
    int nnz = mip.csr_val.size();

    activity.resize(m);

    int *d_row_ptr, *d_col_idx;
    double *d_val, *d_x, *d_activity;

    cudaMalloc(&d_row_ptr,(m+1)*sizeof(int));
    cudaMalloc(&d_col_idx,nnz*sizeof(int));
    cudaMalloc(&d_val,nnz*sizeof(double));
    cudaMalloc(&d_x,n*sizeof(double));
    cudaMalloc(&d_activity,m*sizeof(double));

    cudaMemcpy(d_row_ptr,mip.csr_row_ptr.data(),(m+1)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx,mip.csr_col_idx.data(),nnz*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_val,mip.csr_val.data(),nnz*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_x,x.data(),n*sizeof(double),cudaMemcpyHostToDevice);

    int block=256;
    int grid=(m+block-1)/block;

    csr_activity_kernel<<<grid,block>>>(
        m,d_row_ptr,d_col_idx,d_val,d_x,d_activity
    );

    cudaMemcpy(activity.data(),d_activity,m*sizeof(double),cudaMemcpyDeviceToHost);

    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_val);
    cudaFree(d_x);
    cudaFree(d_activity);
}