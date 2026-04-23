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
├── PSEUDOCODE.md                # Complete pseudocode reference
├── rl_heuristic.h/cpp           # Main RL heuristic (Algorithm 1) ✅
├── rl_agent.h/cpp               # Actor-Critic GNN (LibTorch + CPU fallback) ✅
├── rl_reward.h                  # Two-phase reward computation ✅
├── rl_variable_selection.h      # Variable selection (Algorithm 3, MILP-aware) ✅
├── rl_training.h/cpp            # Training loop (Algorithm 2) ✅
├── rl_graph.h/cpp               # Bipartite graph representation ✅
├── rl_state.h                   # State + MILP-aware actions ✅
├── rl_features.h                # Feature engineering (64-dim var, 32-dim constraint) ✅
├── rl_lp_subproblem.h           # LP sub-problem for continuous variables (CLP) ✅
├── main_rl.cpp                  # Inference driver ✅
├── main_train.cpp               # Training driver ✅
└── CMakeLists.txt               # Build (C++17, both executables) ✅
```

### Dependencies:
- COIN-OR (Clp, OsiClp, CoinUtils) — required for LP sub-problem
- PyTorch C++ (LibTorch) — optional, for full Transformer GNN

### Implementation Status Table:

| Component | Status | File |
|-----------|--------|------|
| Bipartite graph | ✅ Complete | `rl_graph.h/cpp` |
| State representation | ✅ Complete | `rl_state.h` |
| Reward computation | ✅ Complete | `rl_reward.h` |
| Variable selection (Alg. 3) | ✅ Complete (MILP-aware) | `rl_variable_selection.h` |
| Solution search (Alg. 1) | ✅ Complete | `rl_heuristic.h/cpp` |
| Training loop (Alg. 2) | ✅ Complete | `rl_training.h/cpp` |
| Feature engineering | ✅ Complete | `rl_features.h` |
| Neural network (LibTorch) | ✅ Complete | `rl_agent.h/cpp` |
| Neural network (CPU fallback) | ✅ Complete | `rl_agent.h/cpp` |
| MILP variable filtering | ✅ Complete | `rl_variable_selection.h` |
| MILP action clamping | ✅ Complete | `rl_state.h` |
| LP sub-problem (continuous vars) | ✅ Complete | `rl_lp_subproblem.h` |
| Integrality check | ✅ Complete | `rl_state.h` |
| Training driver | ✅ Complete | `main_train.cpp` |
| Inference driver | ✅ Complete | `main_rl.cpp` |
| Build system | ✅ Complete | `CMakeLists.txt` |

---

## 10. Key Design Decisions

1. **Improvement heuristic approach:** Start from LP-relaxation or rounded solution, use RL to find feasible + improve
2. **Phase separation:** Critical for learning feasibility before optimization
3. **Variable selection:** Reduces action space from 3ⁿ to 3^(2log₂n) = n²
4. **Periodic embedding:** Handles unbounded variable values
5. **Toward-optimal bias α=2:** Guides exploration toward better solutions
6. **Feature dimensions:** 64-dim variable features, 32-dim constraint features (from paper Appendix C)
7. **Modular design:** Core algorithms (Alg 1-3) independent of neural network implementation
8. **MILP adaptation:** RL agent restricted to integer/binary variables; LP sub-problem for continuous

---

## 11. Progress Log

### Session 1 (2026-04-19)
- [x] Read and analyzed RL-SPH paper
- [x] Created REPORT.md with comprehensive summary
- [x] Identified existing code structures (MIPProblem, repair functions)

### Session 2 (2026-04-20)
- [x] Created core RL files: state, graph, reward, variable selection, heuristic, agent, training
- [x] Created test driver and CMakeLists.txt

### Session 3 (2026-04-21) — Feature Engineering
- [x] Created `rl_features.h` — 64-dim variable + 32-dim constraint features
- [x] Integrated feature builder into agent

### Session 4 (2026-04-23) — LibTorch Integration
- [x] Implemented Transformer-based GNN actor + critic in LibTorch
- [x] Changed optimizer to RMSprop, implemented full gradient-based update()
- [x] Wired agent into heuristic and training loops

### Session 5 (2026-04-23) — MILP Adaptation + Pipeline Completion
- [x] **MILP variable selection**: Filtered continuous variables from `select_variables()` — only integer/binary vars selected as seeds/neighbors. `p = log2(n_int)` instead of `log2(n)`
- [x] **MILP action clamping**: Created `apply_actions_milp()` — skips continuous vars, clamps binary to {0,1}, clamps integer to [lb,ub] and rounds
- [x] **Integrality check**: Updated `is_feasible()` to verify integer/binary vars have integer values
- [x] **LP sub-problem**: Created `rl_lp_subproblem.h` using CLP — after fixing integers, solves LP for optimal continuous variable values
- [x] **Integrated LP sub-problem** into both `rl_heuristic.cpp` and `rl_training.cpp`
- [x] **Training driver**: Created `main_train.cpp` with CLI flags (--updates, --steps, --batch, --lr, --save, --load), directory scanning for .mps files
- [x] **Inference driver**: Updated `main_rl.cpp` with model_path argument and variable type reporting
- [x] **Build system**: Updated `CMakeLists.txt` — builds both `rl_sph_test` and `rl_sph_train`, added stdc++fs for GCC<9, fixed compile definitions syntax
- [x] **Initialization**: Updated `initialize_solution()` to handle continuous vars (set to lb, only randomize integer/binary)
- [x] **Syntax review**: Fixed `rl_training.h` signature mismatch, removed dead code in main_train.cpp
- [x] Updated all documentation (README.md, REPORT.md, PSEUDOCODE.md)

---

## 12. ILP → MILP Adaptation (Completed)

The RL-SPH paper targets pure ILPs. We adapted for MILPs:

| Change | Status | File |
|--------|--------|------|
| Variable selection filters continuous variables | ✅ Done | `rl_variable_selection.h` |
| `apply_actions_milp()` skips continuous, clamps binary/integer | ✅ Done | `rl_state.h` |
| `is_feasible()` checks integrality | ✅ Done | `rl_state.h` |
| LP sub-problem for continuous vars (CLP) | ✅ Done | `rl_lp_subproblem.h` |
| Initialization handles continuous vars | ✅ Done | `rl_heuristic.cpp` |
| Feature engineering encodes variable type | ✅ Done | `rl_features.h` |

---

## 13. Future Priority List

### Priority 1: Testing & Validation
- [ ] Train on benchmark instances (MPS files)
- [ ] Add metrics logging — feasibility rate, primal gap, convergence curves
- [ ] Validate against paper results on ILP benchmarks

### Priority 2: Training Pipeline Improvements
- [ ] Experience replay buffer
- [ ] Batched training for parallel instance simulation
- [ ] Add LP reduced costs as additional features for MILP

### Priority 3: Performance Optimization
- [ ] GPU acceleration via LibTorch CUDA tensors
- [ ] Sparse tensor operations for large-scale MILPs
- [ ] Multi-threading for variable selection

---

## 14. Build & Run Instructions

### Build (Ubuntu/Debian)
```bash
sudo apt-get install -y coinor-libclp-dev coinor-libosi-dev coinor-libcoinutils-dev
cd RL && rm -rf build && mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Train
```bash
./rl_sph_train ./training_instances/ --updates 500 --batch 4 --save model.pt
```

### Inference
```bash
./rl_sph_test problem.mps 60 2000 model.pt
```

---

## References

[1] Lee, T.-H., Kim, M.-S. "RL-SPH: Learning to Achieve Feasible Solutions for Integer Linear Programs", arXiv:2411.19517v6, 2025.

[2] Gasse, M. et al. "Exact combinatorial optimization with graph convolutional neural networks", NeurIPS 2019.

[3] Han, Q. et al. "A GNN-guided predict-and-search framework for MILP", ICLR 2023.
