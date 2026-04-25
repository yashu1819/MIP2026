# RL-SPH: Reinforcement Learning Start Primal Heuristic for MILP

This directory contains an implementation of RL-SPH, a reinforcement learning-based primal heuristic for solving Mixed Integer Linear Programs (MILPs), adapted from the ILP-focused paper.

## Overview

RL-SPH is based on the paper:
> Lee, T.-H., Kim, M.-S. "RL-SPH: Learning to Achieve Feasible Solutions for Integer Linear Programs", arXiv:2411.19517v6, 2025.

The heuristic uses a Transformer-based GNN with Actor-Critic reinforcement learning to:
1. **Phase 1**: Find the first feasible solution from an infeasible starting point
2. **Phase 2**: Improve the incumbent solution

**MILP Adaptation**: The original paper targets pure ILPs. We adapt the algorithm for MILPs by:
- Filtering continuous variables from variable selection (only integer/binary variables are modified by RL)
- Solving an LP sub-problem for continuous variables after fixing integers (via CLP)
- Adding integrality checks for feasibility verification

## Files

| File | Description | Status |
| :--- | :--- | :--- |
| `rl_graph.h/cpp` | Bipartite graph representation | ✅ Complete |
| `rl_state.h` | Observation state S_t = (x_t, f_t, obj_t) + MILP-aware actions | ✅ Complete |
| `rl_reward.h` | Two-phase reward computation | ✅ Complete |
| `rl_variable_selection.h` | Variable selection (Algorithm 3, MILP-aware) | ✅ Complete |
| `rl_heuristic.h/cpp` | Main heuristic (Algorithm 1) + LP sub-problem | ✅ Complete |
| `rl_agent.h/cpp` | Actor-Critic GNN agent (LibTorch + CPU fallback) | ✅ Complete |
| `rl_training.h/cpp` | Training loop (Algorithm 2) | ✅ Complete |
| `rl_features.h` | Neural network input features (64-dim var, 32-dim constraint) | ✅ Complete |
| `rl_lp_subproblem.h` | LP sub-problem solver for continuous variables (CLP) | ✅ Complete |
| `main_rl.cpp` | Inference driver | ✅ Complete |
| `main_train.cpp` | Training driver | ✅ Complete |
| `CMakeLists.txt` | Build configuration (C++17, both executables) | ✅ Complete |

## Building

### Prerequisites

- CMake 3.14+
- C++17 compiler (GCC 7+ or Clang 5+)
- COIN-OR libraries (Clp, OsiClp, CoinUtils)

### Installation (Ubuntu/Debian)

```bash
sudo apt-get install -y coinor-libclp-dev coinor-libosi-dev coinor-libcoinutils-dev
```

### Installation (macOS)

```bash
brew install coin-or-tools
```

### Build (without LibTorch — CPU fallback)

```bash
cd RL/
rm -rf build && mkdir build && cd build
cmake ..
make -j$(nproc)
```

This produces two executables:
- `rl_sph_test` — Inference (run heuristic on a single MPS file)
- `rl_sph_train` — Training (learn policy from multiple MPS files)

### Build (with LibTorch — Full Transformer GNN)

```bash
# Download LibTorch from https://pytorch.org/get-started/locally/
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.2.0%2Bcpu.zip
unzip libtorch-*.zip

# sudo apt install gdb # For installing gdb debugger

cd RL/
rm -rf build && mkdir build && cd build
cmake -DUSE_LIBTORCH=ON -DCMAKE_PREFIX_PATH=/home/ubuntu/libtorch ..
# cmake -DUSE_LIBTORCH=ON -DCMAKE_PREFIX_PATH=/home/ubuntu/libtorch -DCMAKE_BUILD_TYPE=Debug ..
make -j$(nproc)
```

## Usage

### Training (Step 1 — Learn a Policy)

Before running inference, you must train the model on training MPS files:

```bash
# Train on a directory of MPS files
./rl_sph_train ./training_instances/ --updates 500 --batch 4 --save model.pt
# eg. ./rl_sph_train ../../../test_set/instances/ --updates 1

# Train on specific files
./rl_sph_train a.mps b.mps c.mps --updates 1000 --save model.pt
#eg. ./rl_sph_train ../../../test_set/instances/instance_01.mps.gz ../../../test_set/instances/instance_25.mps.gz ../../../test_set/instances/instance_26.mps.gz --updates 1
```

**Training flags:**
| Flag | Default | Description |
|------|---------|-------------|
| `--updates N` | 5000 | Number of training updates |
| `--steps N` | 2000 | Max steps per episode |
| `--batch N` | 8 | Batch size (instances per update) |
| `--lr F` | 1e-4 | Learning rate |
| `--save PATH` | `rl_sph_model.pt` | Path to save trained model |
| `--load PATH` | — | Path to load pretrained model |

**Input format**: Standard MPS format files (`.mps`). These are the standard format for MILP/ILP problems used by CPLEX, Gurobi, SCIP, etc.

### Inference (Step 2 — Run Heuristic)

```bash
# With trained model
./rl_sph_test problem.mps 60 2000 model.pt

# Without model (random weights — for testing only)
./rl_sph_test problem.mps 60 2000
```

**Arguments:**
| Position | Name | Default | Description |
|----------|------|---------|-------------|
| 1 | `mps_file` | — | Path to MPS file (required) |
| 2 | `time_limit` | 60 | Time limit in seconds |
| 3 | `max_steps` | 2000 | Maximum search steps |
| 4 | `model_path` | — | Path to trained model (optional) |

## Algorithm Details

### MILP Pipeline (Per Step)

```
1. Select variables    → Only integer/binary variables (Algorithm 3)
2. RL agent actions    → Actor network outputs ±1 or 0 for each selected variable
3. Apply actions       → Skip continuous, clamp binary to {0,1}, integer to [lb,ub]
4. LP sub-problem      → Solve LP for optimal continuous variable values (CLP)
5. Compute reward      → Two-phase reward function
6. Update state        → S_{t+1} = (x_{t+1}, f_{t+1}, obj_{t+1})
7. Rollback if needed  → Revert if bounds violated (Phase 1) or worse (Phase 2)
```

### Key Components

1. **Observation**: `S_t = (x_t, f_t, obj_t)`
2. **Action Space**: `{-1, 0, +1}` per variable (integer/binary only)
3. **Variable Selection**: `p = q = log2(n_int)` seed + neighbor variables
4. **Reward**: Two-phase design (feasibility-focused → optimization-focused)
5. **Training**: Actor-Critic with TD error, RMSprop optimizer

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

## MILP Adaptation (ILP → MILP)

The RL-SPH paper assumes all variables are integer. We adapted for MILPs:

| Change | Where |
|--------|-------|
| Variable selection filters continuous variables | `rl_variable_selection.h` |
| `apply_actions_milp()` skips continuous, clamps binary/integer | `rl_state.h` |
| `is_feasible()` checks integrality for integer/binary vars | `rl_state.h` |
| LP sub-problem solves for continuous vars after fixing integers | `rl_lp_subproblem.h` |
| Initialization sets continuous vars to lower bound | `rl_heuristic.cpp` |
| Feature engineering encodes variable type (features 39-40) | `rl_features.h` |

## Troubleshooting

### CMake cache error from macOS
```bash
rm -rf build && mkdir build && cd build
cmake ..
```

### Missing COIN-OR
```bash
sudo apt-get install -y coinor-libclp-dev coinor-libosi-dev coinor-libcoinutils-dev
```

### std::filesystem linker error (GCC < 9)
The CMakeLists.txt automatically links `-lstdc++fs` for GCC < 9.

## References

[1] Lee, T.-H., Kim, M.-S. "RL-SPH: Learning to Achieve Feasible Solutions for Integer Linear Programs", arXiv:2411.19517v6, 2025.

[2] Gasse, M. et al. "Exact combinatorial optimization with graph convolutional neural networks", NeurIPS 2019.

[3] Han, Q. et al. "A GNN-guided predict-and-search framework for MILP", ICLR 2023.
