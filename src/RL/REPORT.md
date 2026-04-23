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
| Neural network (LibTorch) | Complete | `rl_agent.h/cpp` |
| Neural network (CPU fallback) | Complete | `rl_agent.h/cpp` |
| Build system | Complete | `CMakeLists.txt` |

**Current Limitations:**
1. Not yet tested on benchmark instances
2. CPU fallback uses simplified weight perturbation (not true policy gradient)
3. GPU acceleration not yet validated

### Session 4 (2026-04-23) - LibTorch Integration (Priority 1)
- [x] Restructured `rl_agent.h` — renamed CPU classes to `ActorNetworkCPU`/`CriticNetworkCPU`
- [x] Added LibTorch modules `ActorNetworkTorchImpl` / `CriticNetworkTorchImpl` with `TORCH_MODULE` macros
- [x] Implemented Transformer-based GNN actor: embedding → TransformerEncoder → action_head
- [x] Implemented critic: phase_embedding + obj_encoder + constraint_encoder → value_head
- [x] Changed optimizer from Adam to RMSprop (matching paper specification)
- [x] Implemented `build_feature_tensor()` and `build_constraint_tensor()` for LibTorch path
- [x] Implemented `select_actions_training()` with gradient-tracked forward passes
- [x] Implemented full `update()` with policy gradient loss + TD critic loss + backprop
- [x] Wired up agent in `rl_heuristic.cpp` — replaced random actions with `agent_.select_actions()`
- [x] Updated `rl_training.cpp` — collects full trajectory and calls `agent->update()`
- [x] Fixed `CMakeLists.txt` — removed duplicates, C++17 for LibTorch, propagate USE_LIBTORCH
- [x] Added `TrainingForwardResult` struct for training-mode forward passes
- [x] Updated `README.md`, `REPORT.md`, `PSEUDOCODE.md`

---

## 12. Next Steps for Full Implementation

### ~~Priority 0: LibTorch Integration~~ ✅ COMPLETED (Session 4)

The full Transformer-based Actor-Critic GNN is implemented:
- Actor: `features(64) → Linear → ReLU → TransformerEncoder(4L, 4H) → Linear → logits(3)`
- Critic: `phase_emb + obj_emb + constraint_emb → concat → MLP → V(s)`
- Loss: `L = -log π(a|s)·δ + δ²`, Optimizer: RMSprop (α=0.99, ε=1e-5, lr=1e-4)

---

## 13. ILP → MILP Adaptation (Critical Gap)

The RL-SPH paper (Lee & Kim, 2025) is designed for **pure ILPs** where all variables are integer. Our use-case is **MILPs** (Mixed Integer Linear Programs) which contain both integer and continuous variables. This creates several mismatches:

### What's Already Handled
- Feature engineering (`rl_features.h`) encodes variable type as features (39-40: `is_integer`, `is_binary`)
- `MIPProblem` stores `VarType` enum (CONTINUOUS, INTEGER, BINARY) for each variable

### What's NOT Yet Handled (Future Work)

| Component | Issue | Fix Required |
|-----------|-------|-------------|
| **Variable selection** (`rl_variable_selection.h`) | Continuous variables can be selected as seeds/neighbors | Filter: only select vars where `vartype != CONTINUOUS` |
| **Action application** (`rl_state.h::apply_actions`) | ±1 applied to continuous variables is meaningless | Skip continuous vars; only modify integer/binary |
| **Post-action clamping** | No clamping after action | Binary: clamp to {0,1}; Integer: clamp to [lb, ub] and round |
| **Continuous variable fixing** | Continuous vars are never properly set | After fixing integers, solve LP sub-problem for optimal continuous values |
| **State evaluation** | Feasibility checked with integers at non-integer values | Add integrality check only for integer/binary variables |

### Proposed Approach for MILP
1. **Restrict RL agent to integer variables only** — variable selection filters out continuous
2. **After each RL step**, solve an LP with integer variables fixed to find optimal continuous values
3. **Use LP reduced costs** as additional features for the RL agent
4. This is similar to the "fix-and-optimize" approach used in MILP heuristics

---

## 14. Future Priority List

### Priority 1: MILP Adaptation
- [ ] Filter continuous variables from selection in `rl_variable_selection.h`
- [ ] Clamp actions to valid bounds in `apply_actions()` — binary stays {0,1}
- [ ] Solve LP sub-problem for continuous variables after fixing integers
- [ ] Add LP reduced costs as features for RL agent

### Priority 2: Testing & Validation
- [ ] Test on benchmark instances (KNAPSACK, MVC, IS, CA, SC, NBI)
- [ ] Add metrics logging — feasibility rate, primal gap, convergence curves
- [ ] Validate against paper results on ILP benchmarks

### Priority 3: Training Pipeline
- [ ] Experience replay buffer
- [ ] Batched training for parallel instance simulation
- [ ] Model checkpointing

### Priority 4: Performance Optimization
- [ ] GPU acceleration via LibTorch CUDA tensors
- [ ] Sparse tensor operations for large-scale MILPs
- [ ] Multi-threading for variable selection

---

## 15. Build Instructions

### macOS

```bash
brew install coin-or-tools
cd RL && mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Ubuntu/Debian

```bash
sudo apt-get install -y coinor-libclp-dev coinor-libosi-dev coinor-libcoinutils-dev
cd RL && mkdir build && cd build
cmake ..
make -j$(nproc)
```

### With LibTorch

```bash
# Download LibTorch
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.2.0%2Bcpu.zip
unzip libtorch-*.zip

cd RL && rm -rf build && mkdir build && cd build
cmake -DUSE_LIBTORCH=ON -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
make -j$(nproc)
```

---

## References

[1] Lee, T.-H., Kim, M.-S. "RL-SPH: Learning to Achieve Feasible Solutions for Integer Linear Programs", arXiv:2411.19517v6, 2025.

[2] Gasse, M. et al. "Exact combinatorial optimization with graph convolutional neural networks", NeurIPS 2019.

[3] Han, Q. et al. "A GNN-guided predict-and-search framework for MILP", ICLR 2023 (PAS).

