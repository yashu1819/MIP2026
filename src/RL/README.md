# RL-SPH: Reinforcement Learning Start Primal Heuristic for MILP

This directory contains an implementation of RL-SPH, a reinforcement learning-based primal heuristic for solving Mixed Integer Linear Programs (MILPs).

## Overview

RL-SPH is based on the paper:
> Lee, T.-H., Kim, M.-S. "RL-SPH: Learning to Achieve Feasible Solutions for Integer Linear Programs", arXiv:2411.19517v6, 2025.

The heuristic uses a Transformer-based GNN with Actor-Critic reinforcement learning to:
1. **Phase 1**: Find the first feasible solution from an infeasible starting point
2. **Phase 2**: Improve the incumbent solution

## Files

| File | Description | Status |
| :--- | :--- | :--- |
| `rl_graph.h/cpp` | Bipartite graph representation of ILP | Complete |
| `rl_state.h` | Observation state S_t = (x_t, f_t, obj_t) | Complete |
| `rl_reward.h` | Two-phase reward computation | Complete |
| `rl_variable_selection.h` | Variable selection (Algorithm 3) | Complete |
| `rl_heuristic.h/cpp` | Main heuristic (Algorithm 1) | Complete |
| `rl_agent.h/cpp` | Actor-Critic GNN agent | Placeholder (random actions) |
| `rl_training.h/cpp` | Training loop (Algorithm 2) | Complete |
| `rl_features.h` | Neural network input features (64-dim var, 32-dim constraint) | Complete |
| `main_rl.cpp` | Test driver | Complete |
| `CMakeLists.txt` | Build configuration | Complete |
| `REPORT.md` | Detailed paper summary and implementation notes | Complete |
| `README.md` | This file | Complete |

## Building

### Prerequisites

- CMake 3.14+
- C++11 compiler
- COIN-OR libraries (Clp, OsiClp, CoinUtils)

### Installation (macOS)

```bash
# Install COIN-OR via Homebrew
brew install coin-or-tools
```

### Installation (Ubuntu/Debian)

```bash
sudo apt-get install -y coinor-libclp-dev coinor-libosi-dev coinor-libcoinutils-dev
```

### Build

```bash
cd RL/
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Optional: LibTorch for Full Neural Network

To enable the full transformer-based GNN (currently uses random actions):

```bash
# Download LibTorch from https://pytorch.org/get-started/locally/
# Extract to /path/to/libtorch

cd build
cmake -DUSE_LIBTORCH=ON -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
make -j$(nproc)
```

## Usage

### Running the Heuristic (Inference)

```bash
./rl_sph_test <mps_file> [time_limit] [max_steps]
```

**Arguments:**
- `mps_file`: Path to MPS format MILP file
- `time_limit`: Time limit in seconds (default: 60)
- `max_steps`: Maximum search steps (default: 2000)

**Example:**
```bash
./rl_sph_test ../test_instances/knapsack.mps 30 1000
```

### Training (Future with LibTorch)

```cpp
#include "rl_training.h"

std::vector<std::string> training_files = {
    "instances/instance1.mps",
    "instances/instance2.mps",
    // ...
};

rl_sph::TrainingConfig config;
config.num_updates = 5000;
config.max_steps = 2000;
config.phase1_steps = 500;
config.batch_size = 64;
config.save_path = "trained_model.pt";

auto stats = rl_sph::train_rl_heuristic(training_files, config);
```

## Configuration

### RLConfig (Inference)

```cpp
rl_sph::RLConfig config;
config.time_limit = 60.0;      // Time limit in seconds
config.max_steps = 2000;       // Maximum steps
config.phase1_steps = 500;     // Steps in feasibility phase
config.init_method = "lp";     // "lp" or "random"
config.seed = 42;              // Random seed
```

### TrainingConfig (Training)

```cpp
rl_sph::TrainingConfig config;
config.num_updates = 5000;     // Parameter updates
config.max_steps = 2000;       // Steps per instance
config.phase1_steps = 500;     // Phase 1 duration
config.batch_size = 64;        // Parallel instances
config.learning_rate = 1e-4f;  // Learning rate
config.gamma = 0.99f;          // Discount factor
```

## Algorithm Details

### Key Components

1. **Observation**: `S_t = (x_t, f_t, obj_t)`
   - `x_t`: Current integer solution
   - `f_t = b - Ax_t`: Constraint satisfaction
   - `obj_t = c^T x_t`: Objective value

2. **Action Space**: For each variable
   - Increase (+1)
   - No change (0)
   - Decrease (-1)

3. **Variable Selection**: At each step, select `2 * log2(n)` variables
   - Phase 1: Variables in violated constraints
   - Phase 2: Variables in well-satisfied constraints

4. **Reward Function**: Two-phase design
   - Phase 1: Feasibility reward + objective improvement bonus
   - Phase 2: Objective improvement + feasibility penalty

### Hyperparameters (from paper)

| Parameter | Value |
|-----------|-------|
| Updates (N) | 5000 |
| Max steps (T_max) | 2000 |
| Phase 1 steps (T_stay) | 500 |
| Batch size | 64 |
| Learning rate | 1e-4 |
| Discount (gamma) | 0.99 |
| Toward-optimal bias (alpha) | 2.0 |

## Current Status

### Completed Components

| Component | File | Status |
|-----------|------|--------|
| Bipartite graph representation | `rl_graph.h/cpp` | Complete |
| State representation | `rl_state.h` | Complete |
| Two-phase reward system | `rl_reward.h` | Complete |
| Variable selection (Alg. 3) | `rl_variable_selection.h` | Complete |
| Solution search (Alg. 1) | `rl_heuristic.h/cpp` | Complete |
| Training loop (Alg. 2) | `rl_training.h/cpp` | Complete |
| Feature engineering | `rl_features.h` | Complete |
| Build system | `CMakeLists.txt` | Complete |

### Current Limitations

1. **Neural Network Placeholder**: The actor-critic agent (`rl_agent.h/cpp`) uses random actions. Full implementation requires LibTorch.
2. **CPU Only**: No GPU acceleration without LibTorch.
3. **Untested**: The heuristic logic is complete but not yet validated on benchmark instances.

## Next Steps

### Priority 1: LibTorch Integration (Required for Full Functionality)

1. **Install LibTorch**:
   ```bash
   # Option A: pip (CPU only)
   pip3 install torch

   # Option B: Download from https://pytorch.org/get-started/locally/
   # Option C: NVIDIA container (GPU support)
   docker pull nvcr.io/nvidia/pytorch:24.04-py3
   ```

2. **Update `rl_agent.h`** - Add LibTorch includes:
   ```cpp
   #ifdef USE_LIBTORCH
   #include <torch/torch.h>
   #include <torch/script.h>
   #endif
   ```

3. **Implement `ActorNetworkImpl`** as `torch::nn::Module`:
   ```cpp
   struct ActorNetworkImpl : torch::nn::Module {
       torch::nn::Linear var_embedding{nullptr};
       torch::nn::TransformerEncoder transformer{nullptr};
       torch::nn::Linear action_head{nullptr};

       ActorNetworkImpl(int input_dim, int hidden_dim, int num_heads, int num_layers) {
           var_embedding = register_module("var_embedding",
               torch::nn::Linear(input_dim, hidden_dim));
           // ... add transformer layers
           action_head = register_module("action_head",
               torch::nn::Linear(hidden_dim, 3));  // 3 actions
       }

       torch::Tensor forward(torch::Tensor x) {
           x = torch::relu(var_embedding->forward(x));
           x = transformer->forward(x);
           return action_head->forward(x);
       }
   };
   ```

4. **Add RMSprop optimizer** in `RLAgent`:
   ```cpp
   #ifdef USE_LIBTORCH
   torch::optim::RMSprop optimizer_;
   #endif
   ```

### Priority 2: Testing & Validation

5. **Test on benchmark instances**:
   ```bash
   ./rl_sph_test ../test_instances/knapsack.mps 60 2000
   ```

6. **Add metrics logging**:
   - Feasibility rate
   - Primal gap
   - Convergence curves

### Priority 3: Training Pipeline

7. **Implement experience replay buffer** in `rl_training.h`
8. **Add batched training** for parallel instance simulation
9. **Model checkpointing** - save/load trained weights

### Priority 4: Performance Optimization

10. **GPU acceleration** via LibTorch CUDA tensors
11. **Sparse tensor operations** for large-scale MILPs
12. **Multi-threading** for variable selection

## Troubleshooting

### Build Errors

**Error**: `unknown type name 'MIPProblem'`
- Ensure `../Definition/mip_problem.h` exists and is included correctly

**Error**: `no member named 'values' in 'Solution'`
- Check that Solution struct uses `x`, `obj_value`, `feasible` members

### Runtime Errors

**Error**: `Failed to load MPS file`
- Verify MPS file format (standard MPS format required)
- Check file path is correct

## License

This implementation is for educational/research purposes. Please cite the original paper if used in research.

## References

[1] Lee, T.-H., Kim, M.-S. "RL-SPH: Learning to Achieve Feasible Solutions for Integer Linear Programs", arXiv:2411.19517v6, 2025.

[2] Bengio, Y., Lodi, A., Prouvost, A. "Machine learning for combinatorial optimization: a methodological tour d'horizon", EJOR 2021.

[3] Gasse, M. et al. "Exact combinatorial optimization with graph convolutional neural networks", NeurIPS 2019.
