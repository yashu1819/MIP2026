# RL-SPH: Reinforcement Learning Start Primal Heuristic

## Paper Summary

**Title:** RL-SPH: Learning to Achieve Feasible Solutions for Integer Linear Programs
**Authors:** Tae-Hoon Lee, Min-Soo Kim (KAIST)
**Source:** arXiv:2411.19517v6 (2025)

---

## 1. Key Contributions

1. **First end-to-end primal heuristic** that independently generates feasible solutions for ILPs (including non-binary integer variables)
2. **Reinforcement learning framework** tailored for ILP feasibility learning
3. **Transformer-based GNN** to capture long-range variable-constraint dependencies
4. **Two-phase reward system**: feasibility first, then optimization

---

## 2. Problem Formulation

Standard ILP form (minimization):
```
minimize    cᵀx
subject to  Ax ≤ b
            lᵢ ≤ xᵢ ≤ uᵢ
            xᵢ ∈ ℤ (integer variables)
```

---

## 3. RL Framework Design

### 3.1 Observation Space
`Sₜ = (xₜ, fₜ, objₜ)`
- `xₜ`: current integer solution
- `fₜ = b - Axₜ`: constraint satisfaction vector (fₜ,ⱼ ≥ 0 means satisfied)
- `objₜ = cᵀxₜ`: objective value

### 3.2 Action Space
For each variable, 3 actions:
- **Increase** (+1)
- **No change** (0)
- **Decrease** (-1)

### 3.3 Reward Functions

**Total Reward:** `Rₜ,total = Rₜ,opt + Rₜ,explore`

**Exploration penalty:**
```
Rₜ,explore = -100 if xₜ₊₁ = xₜ (no move)
             0 otherwise
```

**Feasibility Reward:**
```
Rₜ,F = Rₜ,bound + (1/ñ)Rₜ,const
```
where:
- `Rₜ,bound = -Σᵢ I(xₜ₊₁,ᵢ ∉ [lᵢ, uᵢ])` (bound violations)
- `Rₜ,const = Σⱼ min(fₜ₊₁,ⱼ, 0) - min(fₜ,ⱼ, 0)` (constraint improvement)

**Phase 1 Reward** (until first feasible solution):
```
Rₜ,p1 = Rₜ,bound                              if Rₜ,const ≥ 0 ∧ obj better ∧ bounds violated
      = Rₜ,bound - Δobjₜ                      if Rₜ,const ≥ 0 ∧ obj worse ∧ bounds violated
      = Rₜ,F + Δobjₜ                          if Rₜ,const ≥ 0 ∧ obj better ∧ bounds satisfied
      = Rₜ,F - Δobjₜ                          if Rₜ,const < 0 ∧ obj worse
      = Rₜ,F                                  otherwise
```

**Phase 2 Reward** (optimization):
```
Rₜ,p2 = Δobjₜ                                 if feasible ∧ better than incumbent
      = -Δobjₜ · α                            if feasible ∧ worse than incumbent
      = Rₜ,F                                  if infeasible ∧ obj better
      = Rₜ,F · α                              if infeasible ∧ obj worse
```
where α = 2 (toward-optimal bias)

---

## 4. GNN Architecture

### 4.1 Transformer Encoder
- **Input features:**
  - Scaled objective coefficients: `scale(cᵀ|A)` to [-1, 1]
  - Binary bound feature: `bnd_lim = 1` if variable at/exceeds bound
  - Periodic embedding of variable values: `PE(z) = ⊕(sin(z̃), cos(z̃))`
    where `z̃ = [2πw₁z, ..., 2πwₖz]`

### 4.2 Phase-Separated Layers
- **Actor layer:** Outputs action probabilities πθ(A|S, phase)
- **Critic layer:** Estimates value Vθ(S, phase)
- Separate parameters for each phase (different goals)

---

## 5. Variable Selection Strategy (Algorithm 3)

At each timestep, select `ñ = p + q` changeable variables:

### Phase 1 (Feasibility):
1. **Seed selection:** Variables in violated constraints
   - Score: `score_seed = I(fₜ < 0)ᵀ · I(A ≠ 0)`
   - Weight by: `(max(|c|) - |c| + 1) / max(|c|)` (prefer low-cost variables)
2. **Neighbor selection:** Variables sharing constraints with seeds
   - `g = rowwise_sum(A[:, seed_indices])`
   - `score_neighbor = gᵀ · A`

### Phase 2 (Optimization):
1. **Seed selection:** Variables in low-risk constraints
   - Score: `score_seed = I(fₜ > 0)ᵀ · I(A ≠ 0)`
   - Weight by: `|c| / max(|c|)` (prefer high-impact variables)

Both: `p = q = log₂(n)` variables

---

## 6. Training Algorithm (Algorithm 2)

```
1. Initialize: x₀ (LP-relaxation or random), xᵦ = ∅, objᵦ = ∞, phase = 1
2. For each training instance:
   a. Select variables (Alg. 3)
   b. Agent predicts actions Aₜ ~ πθ(·|Sₜ, phaseₜ)
   c. Update solution: xₜ₊₁ = move(xₜ, Aₜ)
   d. Compute reward Rₜ,total
   e. Update incumbent if xₜ₊₁ feasible and better
   f. TD-error: δₜ = Rₜ,total + γ·Vθ(Sₜ₊₁) - Vθ(Sₜ)
   g. Loss: Lθ = -log πθ(Aₜ|Sₜ)·δₜ + δₜ²
   h. Update θ with RMSprop
3. Stay in phase 1 for Tₛₜₐᵧ steps, then switch to phase 2
```

**Hyperparameters:**
- N = 5000 updates
- Tₘₐₓ = 2000 steps per instance
- Tₛₜₐᵧ = 500 steps in phase 1
- Training: 64 instances in parallel

---

## 7. Solution Search (Algorithm 1)

```
1. Select variables S̃ₜ (Alg. 3)
2. Aₜ ← πθ(S̃ₜ, phaseₜ)
3. xₜ₊₁ ← move(xₜ, Aₜ)
4. Sₜ₊₁ ← observe(M, xₜ₊₁)
5. Rₜ,total ← reward(M, Sₜ₊₁, Sₜ, objᵦ, phaseₜ)
6. If xₜ₊₁ ∈ F and objₜ₊₁ < objᵦ:
     objᵦ ← objₜ₊₁
     xᵦ ← xₜ₊₁
7. Else if (phase 1 and bounds violated) or phase 2:
     Sₜ₊₁ ← Sₜ (rollback)
8. Return Rₜ,total, Sₜ₊₁, xᵦ, objᵦ
```

---

## 8. Experimental Results

| Dataset | FR(%) | PG(%) RL-SPH | PG(%) Best Baseline | PI RL-SPH |
|---------|-------|--------------|---------------------|-----------|
| IS      | 100   | 0.14         | 18.09 (Rounding)    | 2.5       |
| CA      | 100   | 3.82         | 12.02 (Rounding)    | 21.9      |
| SC      | 100   | 9.67         | 0.00 (Diving)       | 168.0     |
| MVC     | 100   | 0.81         | 6.76 (FP)           | 5.0       |
| NBI     | 100   | 0.00         | -                   | 10.8      |

**Key findings:**
- 44× lower primal gap, 2.3× lower primal integral vs baselines
- Random initialization performs comparably to LP initialization
- Transformer encoder crucial for feasibility (100% FR vs 0% without)

---

## 9. Implementation Status

### Files Created:
```
RL/
├── REPORT.md                    # This file
├── README.md                    # Usage documentation
├── rl_heuristic.h/cpp           # Main RL heuristic (Algorithm 1) ✓
├── rl_agent.h/cpp               # Actor-Critic GNN agent (placeholder) ✓
├── rl_reward.h                  # Reward computation ✓
├── rl_variable_selection.h      # Variable selection (Algorithm 3) ✓
├── rl_training.h/cpp            # Training loop (Algorithm 2) ✓
├── rl_graph.h/cpp               # Bipartite graph representation ✓
├── rl_state.h                   # State representation S_t = (x_t, f_t, obj_t) ✓
├── rl_features.h                # Feature engineering (64-dim var, 32-dim const) ✓
├── main_rl.cpp                  # Entry point / test driver ✓
└── CMakeLists.txt               # Build configuration ✓
```

### Dependencies:
- COIN-OR (Clp, OsiClp, CoinUtils) for LP relaxation
- PyTorch C++ (LibTorch) for neural network (optional, currently uses random actions)

---

## 10. Key Design Decisions

1. **Improvement heuristic approach:** Start from LP-relaxation or rounded solution, use RL to find feasible + improve
2. **Phase separation:** Critical for learning feasibility before optimization
3. **Variable selection:** Reduces action space from 3ⁿ to 3^(2log₂n) = n²
4. **Periodic embedding:** Handles unbounded variable values
5. **Toward-optimal bias α=2:** Guides exploration toward better solutions
6. **Feature dimensions:** 64-dim variable features, 32-dim constraint features (from paper Appendix C)
7. **Modular design:** Core algorithms (Alg 1-3) independent of neural network implementation

---

## 11. Progress Log

### Session 1 (2026-04-19)
- [x] Read and analyzed RL-SPH paper
- [x] Created REPORT.md with comprehensive summary
- [x] Identified existing code structures (MIPProblem, repair functions)

### Session 2 (2026-04-20)
- [x] Created `rl_graph.h/cpp` - Bipartite graph representation of ILP
- [x] Created `rl_state.h` - Observation state S_t = (x_t, f_t, obj_t)
- [x] Created `rl_reward.h` - Two-phase reward computation (feasibility + optimization)
- [x] Created `rl_variable_selection.h` - Algorithm 3 implementation
- [x] Created `rl_heuristic.h/cpp` - Main RL heuristic (Algorithm 1)
- [x] Created `rl_agent.h/cpp` - Actor-Critic GNN agent (placeholder for LibTorch)
- [x] Created `rl_training.h/cpp` - Training loop (Algorithm 2)
- [x] Created `main_rl.cpp` - Test driver executable
- [x] Created `CMakeLists.txt` - Build configuration
- [x] Created `README.md` - Usage documentation

### Session 3 (2026-04-21) - Feature Engineering
- [x] Created `rl_features.h` - Neural network input feature builder
  - VariableFeatures: 64-dim features (objective coeffs, bounds, periodic embedding, degree, centrality)
  - ConstraintFeatures: 32-dim features (RHS, slack, violation magnitude)
  - PeriodicEmbedding: sin/cos encoding for unbounded values
  - FeatureBuilder: scales and normalizes all features to [0,1]
- [x] Updated `rl_agent.h/cpp` - Integrated feature builder into agent
- [x] Updated `rl_training.h/cpp` - Fixed agent initialization with pointer semantics

**Implementation Status:**
| Component | Status | File |
|-----------|--------|------|
| Bipartite graph | Complete | `rl_graph.h/cpp` |
| State representation | Complete | `rl_state.h` |
| Reward computation | Complete | `rl_reward.h` |
| Variable selection (Alg. 3) | Complete | `rl_variable_selection.h` |
| Solution search (Alg. 1) | Complete | `rl_heuristic.h/cpp` |
| Training loop (Alg. 2) | Complete | `rl_training.h/cpp` |
| Feature engineering | Complete | `rl_features.h` |
| Neural network | Placeholder | `rl_agent.h/cpp` |
| Build system | Complete | `CMakeLists.txt` |

**Current Limitations:**
1. Neural network uses random actions (placeholder in `rl_agent.cpp`)
2. Full neural network requires LibTorch installation
3. Not yet tested on benchmark instances

---

## 12. Next Steps for Full Implementation

### Priority 1: LibTorch Integration (Required)

The current implementation has all algorithmic components complete but uses random actions. To enable the full transformer-based GNN:

1. **Install LibTorch**:
   ```bash
   # Option A: pip (CPU only)
   pip3 install torch

   # Option B: Download from https://pytorch.org/get-started/locally/

   # Option C: NVIDIA container (GPU support)
   docker pull nvcr.io/nvidia/pytorch:24.04-py3
   ```

2. **Update `CMakeLists.txt`**:
   ```bash
   cd build
   cmake -DUSE_LIBTORCH=ON -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
   make -j$(nproc)
   ```

3. **Update `rl_agent.h`** - Add LibTorch includes:
   ```cpp
   #ifdef USE_LIBTORCH
   #include <torch/torch.h>
   #include <torch/script.h>
   #endif
   ```

4. **Implement `ActorNetworkImpl`** as `torch::nn::Module`:
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

5. **Add RMSprop optimizer** in `RLAgent`:
   ```cpp
   #ifdef USE_LIBTORCH
   torch::optim::RMSprop optimizer_;
   #endif
   ```

### Priority 2: Testing & Validation

6. **Test on benchmark instances**:
   ```bash
   ./rl_sph_test ../test_instances/knapsack.mps 60 2000
   ```

7. **Add metrics logging**:
   - Feasibility rate (% instances solved)
   - Primal gap to optimality
   - Convergence curves

8. **Validate against paper results**:
   | Dataset | Expected FR | Expected PG |
   |---------|-------------|-------------|
   | IS      | 100%        | 0.14%       |
   | CA      | 100%        | 3.82%       |
   | SC      | 100%        | 9.67%       |
   | MVC     | 100%        | 0.81%       |
   | NBI     | 100%        | 0.00%       |

### Priority 3: Training Pipeline

9. **Implement experience replay buffer** in `rl_training.h`:
   ```cpp
   struct Experience {
       RLState state;
       std::vector<Action> actions;
       double reward;
       RLState next_state;
       int phase;
   };
   std::deque<Experience> replay_buffer;
   ```

10. **Add batched training** for parallel instance simulation

11. **Model checkpointing** - save/load trained weights

### Priority 4: Performance Optimization

12. **GPU acceleration** via LibTorch CUDA tensors

13. **Sparse tensor operations** for large-scale MILPs

14. **Multi-threading** for variable selection

---

## 13. Build Instructions

### macOS

```bash
# Install COIN-OR dependencies
brew install coin-or-tools

# Build
cd RL
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Ubuntu/Debian

```bash
# Install COIN-OR dependencies
sudo apt-get install -y coinor-libclp-dev coinor-libosi-dev coinor-libcoinutils-dev

# Build
cd RL
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### With LibTorch (GPU Support via NVIDIA Container)

```bash
# Pull PyTorch container
docker pull nvcr.io/nvidia/pytorch:24.04-py3

# Run container
docker run --gpus all -it --rm \
    -v /path/to/MIP2026:/workspace/MIP2026 \
    nvcr.io/nvidia/pytorch:24.04-py3

# Inside container:
cd /workspace/MIP2026/src/RL
mkdir build && cd build

# Enable LibTorch build
cmake -DUSE_LIBTORCH=ON ..
make -j$(nproc)
```

---

## References

[1] Lee, T.-H., Kim, M.-S. "RL-SPH: Learning to Achieve Feasible Solutions for Integer Linear Programs", arXiv:2411.19517v6, 2025.

[2] Gasse, M. et al. "Exact combinatorial optimization with graph convolutional neural networks", NeurIPS 2019.

[3] Han, Q. et al. "A GNN-guided predict-and-search framework for MILP", ICLR 2023 (PAS).

---

## References

[1] Lee, T.-H., Kim, M.-S. "RL-SPH: Learning to Achieve Feasible Solutions for Integer Linear Programs", arXiv:2411.19517v6, 2025.

[2] Gasse, M. et al. "Exact combinatorial optimization with graph convolutional neural networks", NeurIPS 2019.

[3] Han, Q. et al. "A GNN-guided predict-and-search framework for MILP", ICLR 2023 (PAS).
