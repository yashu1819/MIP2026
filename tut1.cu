#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cfloat>
#include <cuda/atomic> // Required for CCCL atomic_ref
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// Error checking macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
     fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
     if (abort) exit(code);
  }
}

struct MoveResult {
   int mutex;       // 0 = unlocked, 1 = locked
   float delta_obj;
   int i;           
   int j;           
   int di;          
   int dj;          
};

/**
* CUDA Kernel: Hybrid Sparse/Dense 2-Opt Check
* - One Block per pair (i, j)
* - Uses Sparse CSC to iterate active constraints.
* - Uses Dense Matrix for O(1) coefficient lookups and intersection checks.
* - Uses CCCL atomic_ref for efficient global synchronization (wait/notify).
*/
__global__ void find_2opt_move_kernel_hybrid(
   const int* __restrict__ d_A_col_ptr,
   const int* __restrict__ d_A_row_ind,
   const float* __restrict__ d_A_val,
   const float* __restrict__ d_A_dense, // PRECOMPUTED DENSE LOOKUP
   const float* __restrict__ d_b,
   const float* __restrict__ d_c,
   const float* __restrict__ d_x,
   const float* __restrict__ d_lb,
   const float* __restrict__ d_ub,
   const float* __restrict__ d_activity,
   int num_vars,
   MoveResult* d_result
) {
   int i = blockIdx.x;
   int j = blockIdx.y;

   if (i >= num_vars || j >= num_vars || i >= j) return;

   // Get Column Ranges from Global Memory
   int start_i = d_A_col_ptr[i];
   int end_i   = d_A_col_ptr[i+1];

   int start_j = d_A_col_ptr[j];
   int end_j   = d_A_col_ptr[j+1];

   __shared__ int s_feasible;

   float c_i = d_c[i];
   float c_j = d_c[j];
   float x_i = d_x[i];
   float x_j = d_x[j];

   // Loop through perturbations
   for (int di = -1; di <= 1; ++di) {
       float new_xi = x_i + (float)di;
       if (new_xi < d_lb[i] || new_xi > d_ub[i]) continue;

       for (int dj = -1; dj <= 1; ++dj) {
           if (di == 0 && dj == 0) continue;
           float new_xj = x_j + (float)dj;
           if (new_xj < d_lb[j] || new_xj > d_ub[j]) continue;

           float potential_delta = c_i * (float)di + c_j * (float)dj;
           
           // Optimization: Read strictly to avoid overhead, but careful with stale data
           if (potential_delta >= d_result->delta_obj) continue;

           // --- Feasibility Check ---
           if (threadIdx.x == 0) s_feasible = 1;
           __syncthreads();

           // 1. Iterate constraints in Column I
           if (di != 0) {
               int count_i = end_i - start_i;
               for (int k = threadIdx.x; k < count_i; k += blockDim.x) {
                   int idx = start_i + k;
                   // !!Coalesced reads!!
                   //  Each neighboring thread in the block is accessing neighboring elements.
                   int row = d_A_row_ind[idx];
                   float val_i = d_A_val[idx];
                   
                   float val_j = 0.0f;
                   if (dj != 0) {
                       val_j = d_A_dense[row * num_vars + j];
                   }

                   float change = val_i * (float)di + val_j * (float)dj;
                   
                   if (d_activity[row] + change > d_b[row] + 1e-5f) {
                       s_feasible = 0;
                   }
               }
           }
           __syncthreads();
           if(s_feasible == 0) {
             continue;
           }
           // 2. Iterate constraints in Column J
           if (dj != 0) {
               int count_j = end_j - start_j;
               for (int k = threadIdx.x; k < count_j; k += blockDim.x) {
                   if (s_feasible == 0) break;

                   int idx = start_j + k;
                   // coalesced read!
                   int row = d_A_row_ind[idx];

                   if (di != 0) {
                       if (d_A_dense[row * num_vars + i] != 0.0f) {
                           continue;
                       }
                   }
                   // coalesced read
                   float val_j = d_A_val[idx];
                   float change = val_j * (float)dj;

                   if (d_activity[row] + change > d_b[row] + 1e-5f) {
                       s_feasible = 0;
                   }
               }
           }
           
           __syncthreads();

           if (threadIdx.x == 0 && s_feasible) {
               // --- CRITICAL SECTION WITH CCCL ATOMIC_REF ---
               // 1. Check if we are potentially better (Optimization: avoids unnecessary locking)
               if (potential_delta < d_result->delta_obj) {
                   // Create an atomic reference to the mutex in global memory
                   // Scope: Device (visible to all threads on the GPU)
                   cuda::atomic_ref lock(d_result->mutex);

                   // 2. Acquire Lock
                   // Try to exchange 0 -> 1.
                   // If it returns 1, the lock is held, so we WAIT.
                   // 'wait(1)' puts the thread to sleep efficiently if the value is 1.
                   while (lock.exchange(1, cuda::std::memory_order_acquire) != 0) {
                       lock.wait(1, cuda::std::memory_order_relaxed);
                   }

                   // 3. Double-Check inside lock (Required because another thread might have updated while we waited)
                   if (potential_delta < d_result->delta_obj) {
                       d_result->delta_obj = potential_delta;
                       d_result->i = i;
                       d_result->j = j;
                       d_result->di = di;
                       d_result->dj = dj;
                   }

                   // 4. Release Lock
                   lock.store(0, cuda::std::memory_order_release);
                   
                   // Wake up any threads sleeping in 'wait()'
                   lock.notify_all();
               }
           }
           // wait so that next iteration waits until the current iteration results are written
           __syncthreads();
       }
   }
}

int main() {
   const int N = 3; // Vars
   const int M = 2; // Constraints

   std::vector h_c = {-2.0f, -3.0f, -4.0f};
   std::vector h_b = {4.0f, 3.0f};
   
   std::vector h_A_dense = {
       3.0f, 2.0f, 1.0f,
       1.0f, 1.0f, 2.0f 
   };

   // --- Convert to CSC ---
   std::vector<int> h_col_ptr = {0};
   std::vector<int> h_row_ind;
   std::vector<float> h_val;

   for (int col = 0; col < N; ++col) {
       for (int row = 0; row < M; ++row) {
           float val = h_A_dense[row * N + col];
           if (abs(val) > 1e-6) {
               h_row_ind.push_back(row);
               h_val.push_back(val);
           }
       }
       h_col_ptr.push_back(h_row_ind.size());
   }

   std::vector<float> h_lb(N, 0.0f);
   std::vector<float> h_ub(N, 1.0f);
   std::vector<float> h_x = {0.0f, 0.0f, 0.0f};
   
   std::vector<float> h_activity(M, 0.0f);
  
   for(int k = 0; k < M; ++k){
    for(int j = 0; j < N; ++j){
        h_activity[k] += h_A_dense[k * N + j] * h_x[j];
    }
}
thrust::device_vector<int>   d_A_col_ptr = h_col_ptr;
thrust::device_vector<int>   d_A_row_ind = h_row_ind;
thrust::device_vector<float> d_A_val     = h_val;
thrust::device_vector<float> d_A_dense   = h_A_dense;

thrust::device_vector<float> d_b  = h_b;
thrust::device_vector<float> d_c  = h_c;
thrust::device_vector<float> d_x  = h_x;
thrust::device_vector<float> d_lb = h_lb;
thrust::device_vector<float> d_ub = h_ub;
thrust::device_vector<float> d_activity = h_activity;

   // Initialize result on device
   MoveResult initial_res = {0, 0.0f, -1, -1, 0, 0};
   thrust::device_vector<MoveResult> d_result(1, initial_res);


   // --- Launch Kernel ---
   dim3 grid(N, N);
   int threadsPerBlock = 128;

   std::cout << "Launching Hybrid Kernel using Thrust..." << std::endl;
   
   // Use raw_pointer_cast to extract pointers for the kernel
   find_2opt_move_kernel_hybrid<<<grid, threadsPerBlock>>>(
       thrust::raw_pointer_cast(d_A_col_ptr.data()),
       thrust::raw_pointer_cast(d_A_row_ind.data()),
       thrust::raw_pointer_cast(d_A_val.data()),
       thrust::raw_pointer_cast(d_A_dense.data()),
       thrust::raw_pointer_cast(d_b.data()),
       thrust::raw_pointer_cast(d_c.data()),
       thrust::raw_pointer_cast(d_x.data()),
       thrust::raw_pointer_cast(d_lb.data()),
       thrust::raw_pointer_cast(d_ub.data()),
       thrust::raw_pointer_cast(d_activity.data()),
       N,
       thrust::raw_pointer_cast(d_result.data())
   );
   cudaCheckError(cudaDeviceSynchronize());

   // --- Results ---
   MoveResult best_move;
   cudaMemcpy(&best_move, thrust::raw_pointer_cast(d_result.data()), sizeof(MoveResult), cudaMemcpyDeviceToHost);

   std::cout << "Best Move Found:\n";
   std::cout << "  Obj Delta: " << best_move.delta_obj << "\n";
   
   if (best_move.delta_obj < 0) {
       std::cout << "  Apply: x[" << best_move.i << "] += " << best_move.di << "\n";
       std::cout << "  Apply: x[" << best_move.j << "] += " << best_move.dj << "\n";
       h_x[best_move.i] += best_move.di;
       h_x[best_move.j] += best_move.dj;
       std::cout << "  New Solution Vector: [ ";
       for(float val : h_x) std::cout << val << " ";
       std::cout << "]" << std::endl;
   }

   return 0;
}
