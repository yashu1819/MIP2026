#ifndef LOCKS_KERNEL_CUH
#define LOCKS_KERNEL_CUH

#include <cuda_runtime.h>

__global__
void compute_locks_kernel(
    int num_cols,
    const int* col_ptr,
    const int* row_idx,
    const double* val,
    int* up_locks,
    int* down_locks
){
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j >= num_cols) return;

    int start = col_ptr[j];
    int end   = col_ptr[j+1];

    int up = 0;
    int down = 0;

    for(int k = start; k < end; k++)
    {
        double a = val[k];

        if(a > 0)
            down++;
        else if(a < 0)
            up++;
    }

    up_locks[j] = up;
    down_locks[j] = down;
}

#endif