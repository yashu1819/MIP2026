#ifndef ACTIVITY_KERNEL_CUH
#define ACTIVITY_KERNEL_CUH

#include <cuda_runtime.h>

__global__
void csr_activity_kernel(
    int m,
    const int* row_ptr,
    const int* col_idx,
    const double* val,
    const double* x,
    double* activity
)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if(row >= m) return;

    double sum = 0.0;

    for(int k = row_ptr[row]; k < row_ptr[row+1]; k++)
    {
        int col = col_idx[k];
        sum += val[k] * x[col];
    }

    activity[row] = sum;
}

#endif