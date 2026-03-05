#ifndef LOCKS_ROUNDING_KERNEL_CUH
#define LOCKS_ROUNDING_KERNEL_CUH

#include <cuda_runtime.h>
#include <math.h>

/*
    Each thread processes ONE variable.

    If variable is integer/binary:
        round according to lock counts

    If continuous:
        keep fractional value
*/

__global__
void lock_round_kernel(
    int n,
    const double* x_frac,
    const int* up_locks,
    const int* down_locks,
    const double* lb,
    const double* ub,
    const int* is_integer,
    double* x_out
)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j >= n)
        return;

    double x = x_frac[j];

    // Continuous variable → keep value
    if (!is_integer[j])
    {
        x_out[j] = x;
        return;
    }

    double floor_val = floor(x);
    double ceil_val  = ceil(x);

    int up   = up_locks[j];
    int down = down_locks[j];

    double rounded;

    // Lock-guided decision
    if (down < up)
        rounded = floor_val;
    else
        rounded = ceil_val;

    // Respect bounds
    if (rounded < lb[j]) rounded = lb[j];
    if (rounded > ub[j]) rounded = ub[j];

    x_out[j] = rounded;
}

#endif