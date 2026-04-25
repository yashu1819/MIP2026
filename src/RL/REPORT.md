# RL-SPH: Reinforcement Learning-Based Start Primal Heuristic for MILP

## Endsem Report

---

## 1. Introduction

### 1.1 Background

Mixed-Integer Linear Programming (MILP) problems are NP-hard combinatorial optimization problems that arise in numerous real-world applications including logistics, scheduling, resource allocation, and path planning. A standard MILP is formulated as:

$$
\begin{aligned}
\text{minimize} \quad & \mathbf{c}^T \mathbf{x} \\
\text{subject to} \quad & \mathbf{A}\mathbf{x} \leq \mathbf{b} \\
& l_i \leq x_i \leq u_i \\
& x_i \in \mathbb{Z} \quad \forall i \in \mathcal{I} \text{ (integer variables)}
\end{aligned}
$$

where $\mathbf{x} \in \mathbb{R}^n$ is the decision vector, $\mathbf{c} \in \mathbb{R}^n$ contains objective coefficients, $\mathbf{A} \in \mathbb{R}^{m \times n}$ is the constraint matrix, $\mathbf{b} \in \mathbb{R}^m$ contains right-hand side values, and $\mathcal{I}$ denotes the set of integer/binary variables.

### 1.2 Primal Heuristics

Primal heuristics play a crucial role in quickly finding feasible solutions for NP-hard MILP problems. Unlike methods that aim for optimality guarantees, primal heuristics focus on finding high-quality feasible solutions without theoretical guarantees. Traditional primal heuristics include:

- **Diving methods**: Fix fractional variables iteratively and resolve
- **Feasibility Pump (FP)**: Alternate between LP-feasible and ILP-feasible solutions
- **Rounding**: Round fractional LP solutions up or down
- **Relaxation Enforced Neighborhood Search (RENS)**: Construct and solve sub-problems

### 1.3 Motivation for RL-SPH

Existing End-to-End Learning-based Primal Heuristics (E2EPH) face several limitations:

1. **Binary variable focus**: Most methods only handle binary variables
2. **Dependence on ILP solvers**: Cannot independently generate feasible solutions
3. **Local dependency modeling**: GCN-based architectures struggle with long-range variable-constraint relationships
4. **Inaccurate predictions**: ML predictions can lead to constraint violations

**RL-SPH** (Reinforcement Learning-based Start Primal Heuristic) addresses these limitations by:
- Handling general integer variables (not just binary)
- Operating as a standalone heuristic without requiring an ILP solver
- Using a Transformer-based GNN for capturing long-range dependencies
- Learning feasibility through a carefully designed two-phase reward system

---

## 2. Methodology

### 2.1 Overview of RL-SPH

RL-SPH is a novel reinforcement learning-based start primal heuristic that learns to generate high-quality feasible solutions for MILP problems in an end-to-end manner. The method operates as follows:

1. **Initialization**: Start from an initial solution (LP-relaxation or random assignment)
2. **Iterative improvement**: At each timestep, select a subset of variables and decide actions to modify them
3. **Two-phase learning**: Phase 1 focuses on achieving feasibility; Phase 2 focuses on optimization
4. **Incumbent tracking**: Maintain the best feasible solution found throughout the search

### 2.2 Bipartite Graph Representation

The MILP instance is represented as a bipartite graph $\mathcal{G} = (\mathcal{V} \cup \mathcal{C}, \mathcal{E})$ where:

- **Variable nodes** $\mathcal{V}$: One node for each decision variable $x_i$
- **Constraint nodes** $\mathcal{C}$: One node for each constraint $a_j^T \mathbf{x} \leq b_j$
- **Edges** $\mathcal{E}$: An edge exists between variable node $i$ and constraint node $j$ if and only if variable $i$ appears in constraint $j$ (i.e., $A_{ji} \neq 0$)

This representation naturally captures the sparse structure of MILP instances and enables efficient message passing between variables and constraints.

### 2.3 Reinforcement Learning Framework

#### 2.3.1 Observation Space

At timestep $t$, the agent observes the state $\mathcal{S}_t = (\mathbf{x}_t, \mathbf{f}_t, \text{obj}_t)$ where:

- $\mathbf{x}_t \in \mathbb{Z}^{|\mathcal{I}|}$: Current integer solution vector
- $\mathbf{f}_t = \mathbf{b} - \mathbf{A}\mathbf{x}_t \in \mathbb{R}^m$: Constraint satisfaction vector
  - $f_{t,j} \geq 0$ indicates constraint $j$ is satisfied
  - $f_{t,j} < 0$ indicates constraint $j$ is violated
- $\text{obj}_t = \mathbf{c}^T \mathbf{x}_t$: Current objective value

#### 2.3.2 Action Space

For each selected variable, the agent can take one of three discrete actions:

| Action | Symbol | Effect |
|--------|--------|--------|
| Increase | $+1$ | $x_i \leftarrow x_i + 1$ |
| No change | $0$ | $x_i \leftarrow x_i$ |
| Decrease | $-1$ | $x_i \leftarrow x_i - 1$ |

For MILP adaptation:
- **Binary variables**: Actions map to $\{0, 1\}$ directly
- **Integer variables**: Actions are clamped to $[l_i, u_i]$ and rounded
- **Continuous variables**: Not modified by RL; optimized via LP sub-problem

#### 2.3.3 Variable Selection Strategy

To reduce the action space from $3^n$ to a manageable size, RL-SPH selects $\tilde{n} = p + q$ variables at each timestep, where $p = q = \lceil \log_2 |\mathcal{I}| \rceil$.

**Phase 1 (Feasibility Focus)**:
1. **Seed selection**: Variables appearing in violated constraints
   - Score: $\text{score}_i = \sum_{j: A_{ji} \neq 0} \mathbb{I}(f_j < 0)$
   - Weighted by: $(\max|\mathbf{c}| - |c_i| + 1) / \max|\mathbf{c}|$ (prefer low-cost variables)
2. **Neighbor selection**: Variables sharing constraints with seeds
   - Compute: $g = \mathbf{A}^T \cdot \mathbb{I}(\text{seed constraints})$
   - Select top-$q$ by score

**Phase 2 (Optimization Focus)**:
1. **Seed selection**: Variables in well-satisfied constraints (low violation risk)
   - Score: $\text{score}_i = \sum_{j: A_{ji} \neq 0} \mathbb{I}(f_j > 0)$
   - Weighted by: $|c_i| / \max|\mathbf{c}|$ (prefer high-impact variables)

**MILP Adaptation**: Only integer/binary variables are eligible for selection; continuous variables are filtered out.

### 2.4 Reward Functions

The reward system is the core innovation of RL-SPH, designed to guide the agent through two distinct phases.

#### 2.4.1 Exploration Penalty

To prevent premature termination and encourage exploration:

$$
\mathcal{R}_{t,\text{explore}} = \begin{cases}
-100 & \text{if } \mathbf{x}_{t+1} = \mathbf{x}_t \text{ (no move)} \\
0 & \text{otherwise}
\end{cases}
$$

#### 2.4.2 Feasibility Reward

The feasibility reward measures progress toward constraint satisfaction:

$$
\mathcal{R}_{t,\text{F}} = \mathcal{R}_{t,\text{bound}} + \frac{1}{\sqrt{\tilde{n}}}\mathcal{R}_{t,\text{const}}
$$

where:
- **Bound reward**: $\mathcal{R}_{t,\text{bound}} = -\sum_{i} \mathbb{I}(x_i \notin [l_i, u_i])$
- **Constraint reward**: $\mathcal{R}_{t,\text{const}} = \sum_{j} \min(f_{t+1,j}, 0) - \min(f_{t,j}, 0)$

#### 2.4.3 Phase 1 Reward (Feasibility Phase)

Phase 1 continues until the first feasible solution is found. The reward prioritizes constraint improvement over objective quality:

$$
\mathcal{R}_{t,p1} = \begin{cases}
\mathcal{R}_{t,\text{bound}} & \text{if } \mathcal{R}_{t,\text{const}} \geq 0 \land \text{obj better} \land \text{bounds violated} \\
\mathcal{R}_{t,\text{bound}} - \Delta\text{obj}_t & \text{if } \mathcal{R}_{t,\text{const}} \geq 0 \land \text{obj worse} \land \text{bounds violated} \\
\mathcal{R}_{t,\text{F}} + \Delta\text{obj}_t & \text{if } \mathcal{R}_{t,\text{const}} \geq 0 \land \text{obj better} \land \text{bounds satisfied} \\
\mathcal{R}_{t,\text{F}} - \Delta\text{obj}_t & \text{if } \mathcal{R}_{t,\text{const}} < 0 \land \text{obj worse} \\
\mathcal{R}_{t,\text{F}} & \text{otherwise}
\end{cases}
$$

where $\Delta\text{obj}_t = |\text{obj}_{t+1} - \text{obj}_t| / \max(|\mathbf{c}|)$ is the normalized objective change.

#### 2.4.4 Phase 2 Reward (Optimization Phase)

Once a feasible solution is found, Phase 2 focuses on improving the objective:

$$
\mathcal{R}_{t,p2} = \begin{cases}
\Delta\text{obj}_t & \text{if feasible} \land \text{better than incumbent} \\
-\Delta\text{obj}_t \cdot \alpha & \text{if feasible} \land \text{worse than incumbent} \\
\mathcal{R}_{t,\text{F}} & \text{if infeasible} \land \text{obj better} \\
\mathcal{R}_{t,\text{F}} \cdot \alpha & \text{if infeasible} \land \text{obj worse}
\end{cases}
$$

where $\alpha = 2$ is the **toward-optimal bias** that penalizes moves away from the incumbent more heavily.

#### 2.4.5 Total Reward

$$
\mathcal{R}_{t,\text{total}} = \mathcal{R}_{t,\text{opt}} + \mathcal{R}_{t,\text{explore}}
$$

where $\mathcal{R}_{t,\text{opt}} = \mathcal{R}_{t,p1}$ in Phase 1 and $\mathcal{R}_{t,\text{opt}} = \mathcal{R}_{t,p2}$ in Phase 2.

---

## 3. Neural Network Architecture

### 3.1 Overview

RL-SPH employs a Transformer-based Graph Neural Network (GNN) architecture consisting of two components:

1. **Actor Network**: Outputs a probability distribution over actions $\pi_\theta(\mathcal{A}_t | \mathcal{S}_t, \text{phase}_t)$
2. **Critic Network**: Estimates the value function $V_\theta(\mathcal{S}_t, \text{phase}_t)$

Both networks use **phase-separated layers** to handle the different goals of Phase 1 (feasibility) and Phase 2 (optimization).

### 3.2 Feature Engineering

#### 3.2.1 Variable Features (64-dimensional)

For each variable $x_i$, the following features are encoded:

| Feature | Dimension | Description |
|---------|-----------|-------------|
| Scaled objective coefficient | 1 | $c_i / \max(|\mathbf{c}|)$ |
| Normalized lower bound | 1 | $l_i / \max(|\mathbf{b}|)$ |
| Normalized upper bound | 1 | $u_i / \max(|\mathbf{b}|)$ |
| Current value | 1 | $x_i / \max(|\mathbf{b}|)$ |
| Bound indicator | 1 | Binary: 1 if at or near bound |
| Periodic embedding of $x_i$ | 32 | $\text{PE}(z) = \bigoplus_{k=1}^{16} (\sin(2\pi w_k z), \cos(2\pi w_k z))$ |
| Variable degree | 1 | Number of constraints containing $x_i$ |
| Variable centrality | 1 | Inverse of average constraint degree |
| Integer indicator | 1 | Binary: 1 if integer/binary variable |
| Binary indicator | 1 | Binary: 1 if binary variable |
| Reserved | 23 | Zero-padded for future features |

**Periodic Embedding**: To handle unbounded variable values, RL-SPH uses periodic embedding:

$$
\text{PE}(z) = \bigoplus_{k=1}^{K} \left(\sin(2\pi w_k z), \cos(2\pi w_k z)\right)
$$

where $w_k = 2^{k-1}$ are frequencies and $\oplus$ denotes concatenation.

#### 3.2.2 Constraint Features (32-dimensional)

For each constraint $j$, the following features are encoded:

| Feature | Dimension | Description |
|---------|-----------|-------------|
| Normalized RHS | 1 | $b_j / \max(|\mathbf{b}|)$ |
| Constraint satisfaction | 1 | $f_j / \max(|\mathbf{b}|)$ |
| Slack indicator | 1 | Binary: 1 if $f_j \geq 0$ |
| Violation magnitude | 1 | $|f_j| / \max(|\mathbf{b}|)$ if violated |
| Constraint degree | 1 | Number of variables in constraint |
| Constraint type | 1 | Reserved for $\leq, \geq, =$ encoding |
| Scaled coefficients | 26 | Encoded $A_{ji}$ values |

### 3.3 Actor Network: Transformer-based GNN

The actor network processes the bipartite graph through multiple layers of bidirectional message passing:

#### 3.3.1 Initial Embedding

Variable and constraint features are first projected to a common hidden dimension $d_{\text{hidden}} = 128$:

$$
\begin{aligned}
\mathbf{h}_\mathcal{V}^{(0)} &= \text{ReLU}(\mathbf{W}_\mathcal{V} \mathbf{f}_\mathcal{V} + \mathbf{b}_\mathcal{V}) \\
\mathbf{h}_\mathcal{C}^{(0)} &= \text{ReLU}(\mathbf{W}_\mathcal{C} \mathbf{f}_\mathcal{C} + \mathbf{b}_\mathcal{C})
\end{aligned}
$$

#### 3.3.2 Bipartite Graph Attention Layer

The core of the architecture is the **Bipartite Graph Attention Layer** which performs bidirectional message passing:

**Variables ← Constraints (Forward)**:
$$
\begin{aligned}
\mathbf{q}_i &= \mathbf{W}_Q \mathbf{h}_{\mathcal{V}, i} \\
\mathbf{k}_j &= \mathbf{W}_K \mathbf{h}_{\mathcal{C}, j} \\
\mathbf{v}_j &= \mathbf{W}_V \mathbf{h}_{\mathcal{C}, j} \\
\alpha_{ij} &= \frac{\exp(\mathbf{q}_i^T \mathbf{k}_j \cdot w_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(\mathbf{q}_i^T \mathbf{k}_k \cdot w_{ik})} \\
\mathbf{h}_{\mathcal{V}, i}^{\text{new}} &= \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{v}_j
\end{aligned}
$$

**Constraints ← Variables (Reverse)**:
$$
\begin{aligned}
\mathbf{q}'_j &= \mathbf{W}'_Q \mathbf{h}_{\mathcal{C}, j} \\
\mathbf{k}'_i &= \mathbf{W}'_K \mathbf{h}_{\mathcal{V}, i} \\
\mathbf{v}'_i &= \mathbf{W}'_V \mathbf{h}_{\mathcal{V}, i} \\
\beta_{ji} &= \frac{\exp(\mathbf{q}'_j^T \mathbf{k}'_i \cdot w_{ij})}{\sum_{k \in \mathcal{N}(j)} \exp(\mathbf{q}'_j^T \mathbf{k}'_k \cdot w_{kj})} \\
\mathbf{h}_{\mathcal{C}, j}^{\text{new}} &= \sum_{i \in \mathcal{N}(j)} \beta_{ji} \mathbf{v}'_i
\end{aligned}
$$

where $w_{ij}$ are edge weights derived from the constraint matrix coefficients $|A_{ji}|$.

#### 3.3.3 Multi-Layer Propagation

The attention layer is applied $L = 4$ times with residual connections:

$$
\mathbf{h}_\mathcal{V}^{(l+1)} = \text{Dropout}(\text{GAT}(\mathbf{h}_\mathcal{V}^{(l)}, \mathbf{h}_\mathcal{C}^{(l)})) + \mathbf{h}_\mathcal{V}^{(l)}
$$

#### 3.3.4 Action Head

After the final GNN layer, variable features are transformed and mapped to action logits:

$$
\begin{aligned}
\mathbf{h}_\mathcal{V}^{\text{final}} &= \text{ReLU}(\mathbf{W}_{\text{transform}} \mathbf{h}_\mathcal{V}^{(L)}) \\
\text{logits}_i &= \mathbf{W}_{\text{action}} \mathbf{h}_{\mathcal{V}, i}^{\text{final}} \in \mathbb{R}^3
\end{aligned}
$$

The probability of each action is computed via softmax:

$$
\pi_\theta(a_i | \mathcal{S}) = \frac{\exp(\text{logits}_{i, a_i})}{\sum_{a \in \{-1, 0, 1\}} \exp(\text{logits}_{i, a})}
$$

### 3.4 Critic Network: Value Estimation

The critic network estimates the expected cumulative reward from the current state:

#### 3.4.1 Input Encoding

The critic receives a richer input than the actor:

1. **Variable features**: Processed through GCN layers
2. **Constraint features**: Processed through GCN layers
3. **Phase encoding**: One-hot embedding of current phase (2 dimensions)
4. **Objective encoding**: Normalized objective value $\text{obj} / (|\text{obj}| + \epsilon)$

#### 3.4.2 Graph Convolution Layers

The critic uses simplified GCN-style message passing:

$$
\mathbf{h}_\mathcal{V}^{(l+1)} = \text{ReLU}(\mathbf{W}_{\text{GCN}} \cdot \text{AGGREGATE}(\{\mathbf{h}_\mathcal{C}^{(l)} : j \in \mathcal{N}(i)\})) + \mathbf{h}_\mathcal{V}^{(l)}
$$

#### 3.4.3 Value Head

After graph convolutions, global pooling and concatenation:

$$
\begin{aligned}
\mathbf{h}_{\text{var}}^{\text{pooled}} &= \frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} \mathbf{h}_{\mathcal{V}, i} \\
\mathbf{h}_{\text{constr}}^{\text{pooled}} &= \frac{1}{|\mathcal{C}|} \sum_{j \in \mathcal{C}} \mathbf{h}_{\mathcal{C}, j} \\
\mathbf{h}_{\text{combined}} &= [\mathbf{h}_{\text{var}}^{\text{pooled}}; \mathbf{h}_{\text{constr}}^{\text{pooled}}; \text{PhaseEmb}(\text{phase}); \text{ObjEmb}(\text{obj})] \\
V_\theta(\mathcal{S}) &= \mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \mathbf{h}_{\text{combined}} + \mathbf{b}_1) + b_2
\end{aligned}
$$

### 3.5 Why Transformer-based GNN?

Traditional GCN-based E2EPH methods struggle with **long-range dependencies** in the bipartite graph. Consider variables $x_2$ and $x_3$ that are 4 hops apart:

```
x₂ -- a₁ -- x₁ -- a₂ -- x₃
```

A change in $x_2$ can affect $x_1$ (via constraint $a_1$), which in turn affects $x_3$ (via constraint $a_2$). Modeling such interactions requires:

- **GCN**: At least 4 layers (prone to oversmoothing)
- **Transformer**: Single attention layer captures all pairwise interactions

The attention mechanism allows each node to attend to all other nodes, enabling effective learning of long-range variable-constraint relationships critical for MILP feasibility.

---

## 4. Training Algorithm

### 4.1 Actor-Critic Framework

RL-SPH uses the **Actor-Critic (AC)** algorithm, which combines:

- **Policy-based actor**: Learns a policy $\pi_\theta(\mathcal{A} | \mathcal{S})$ mapping states to action distributions
- **Value-based critic**: Learns a value function $V_\theta(\mathcal{S})$ estimating expected cumulative reward

### 4.2 Training Procedure (Algorithm 2)

```
Algorithm: RL-SPH Training
Input: Training instances {M₁, M₂, ...}, hyperparameters
Output: Trained actor πθ and critic Vθ

1: Initialize: θ (network parameters), γ = 0.99 (discount)
2: for update = 1 to N_updates do
3:     Sample batch of instances from training set
4:     for each instance M in batch do
5:         // Initialize solution
6:         x₀ ← LP-relaxation or random assignment
7:         S₀ ← (x₀, f₀, obj₀), incumbent x_b = ∅, obj_b = ∞
8:         phase ← 1, stay_in_phase1 ← true
9:         
10:        // Collect trajectory
11:        for t = 0 to T_max do
12:            Select variables: Ĩ_t ← SELECT_VARIABLES(S_t, phase)
13:            Sample actions: A_t ~ πθ(· | S_t, phase_t)
14:            Update solution: x_{t+1} ← MOVE(x_t, A_t)
15:            Observe: S_{t+1} ← (x_{t+1}, f_{t+1}, obj_{t+1})
16:            Compute reward: R_t ← REWARD(S_{t+1}, S_t, obj_b, phase)
17:            
18:            // Update incumbent
19:            if S_{t+1} feasible AND obj_{t+1} < obj_b then
20:                x_b ← x_{t+1}, obj_b ← obj_{t+1}
21:            end if
22:            
23:            // Phase transition
24:            if phase == 1 AND (feasible OR t ≥ T_stay) then
25:                phase ← 2
26:            end if
27:        end for
28:        
29:        // Compute TD errors
30:        for t = 0 to T_max do
31:            δ_t ← R_t + γ · Vθ(S_{t+1}) - Vθ(S_t)
32:        end for
33:        
34:        // Accumulate losses
35:        L_actor ← -Σ_t log πθ(A_t | S_t) · δ_t
36:        L_critic ← Σ_t δ_t²
37:        L_total ← L_actor + L_critic
38:        
39:        // Update parameters
40:        θ ← θ - η · ∇_θ L_total  (using RMSprop)
41:    end for
42: end for
```

### 4.3 Loss Function

The total loss combines actor and critic losses:

$$
\mathcal{L}(\theta) = -\sum_{t=0}^{T} \log \pi_\theta(\mathcal{A}_t | \mathcal{S}_t, \text{phase}_t) \cdot \delta_t + \sum_{t=0}^{T} \delta_t^2
$$

where the **TD error** $\delta_t$ is:

$$
\delta_t = \mathcal{R}_t + \gamma \cdot V_\theta(\mathcal{S}_{t+1}, \text{phase}_{t+1}) - V_\theta(\mathcal{S}_t, \text{phase}_t)
$$

### 4.4 Optimization

RL-SPH uses **RMSprop** optimizer with the following hyperparameters:

| Hyperparameter | Value |
|----------------|-------|
| Learning rate $\eta$ | $10^{-4}$ |
| RMSprop $\alpha$ | $0.99$ |
| RMSprop $\epsilon$ | $10^{-5}$ |
| Weight decay | $10^{-3}$ |
| Discount factor $\gamma$ | $0.99$ |

### 4.5 Training Hyperparameters

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| $N_{\text{updates}}$ | 5000 | Number of parameter updates |
| $T_{\text{max}}$ | 2000 | Maximum steps per instance |
| $T_{\text{stay}}$ | 500 | Minimum steps in Phase 1 |
| Batch size | 64 | Instances trained in parallel |
| $p, q$ | $\lceil \log_2 |\mathcal{I}| \rceil$ | Variables selected per step |

---

## 5. Inference Pipeline

### 5.1 Solution Search (Algorithm 1)

During inference, the trained agent searches for feasible solutions:

```
Algorithm: RL-SPH Inference (Solution Search)
Input: MILP instance M, trained actor πθ, time limit T_limit
Output: Best feasible solution x_b

1: Initialize: x₀ ← LP-relaxation or random, x_b = ∅, obj_b = ∞
2: S₀ ← (x₀, f₀, obj₀), phase ← 1, step ← 0
3: 
4: while step < T_max AND time < T_limit do
5:     // Variable selection
6:     Ĩ_t ← SELECT_VARIABLES(S_t, phase)  // Algorithm 3
7:     
8:     // Action selection (greedy or sampling)
9:     A_t ← πθ(· | S_t, phase)  // Sample or argmax
10:    
11:    // Apply actions
12:    x_{t+1} ← APPLY_ACTIONS_MILP(M, x_t, A_t, Ĩ_t)
13:    
14:    // LP sub-problem for continuous variables
15:    SOLVE_LP_SUBPROBLEM(M, x_{t+1})
16:    
17:    // Update state
18:    S_{t+1} ← (x_{t+1}, f_{t+1}, obj_{t+1})
19:    
20:    // Update incumbent
21:    if S_{t+1} feasible AND obj_{t+1} < obj_b then
22:        x_b ← x_{t+1}, obj_b ← obj_{t+1}
23:    end if
24:    
25:    // Rollback (optional during inference)
26:    if (phase == 1 AND bounds violated) OR (phase == 2 AND not better) then
27:        S_{t+1} ← S_t  // Rollback to previous state
28:    end if
29:    
30:    // Phase transition
31:    if phase == 1 AND step ≥ T_stay then
32:        phase ← 2
33:    end if
34:    
35:    step ← step + 1
36: end while
37: 
38: return x_b, obj_b
```

### 5.2 MILP Adaptation

The original RL-SPH paper targets pure ILPs. For MILPs with continuous variables:

1. **Variable Selection Filtering**: Only integer/binary variables are selected as seeds or neighbors
2. **Action Application**: 
   - Binary variables: Actions map directly to $\{0, 1\}$
   - Integer variables: Actions are clamped to $[l_i, u_i]$ and rounded
   - Continuous variables: Not modified by RL actions
3. **LP Sub-problem**: After fixing integer variables, solve an LP to optimize continuous variables:

$$
\begin{aligned}
\text{minimize} \quad & \mathbf{c}^T \mathbf{x} \\
\text{subject to} \quad & \mathbf{A}\mathbf{x} \leq \mathbf{b} \\
& \mathbf{x}_\mathcal{I} = \bar{\mathbf{x}}_\mathcal{I} \text{ (fixed integer values)} \\
& l_j \leq x_j \leq u_j \quad \forall j \in \mathcal{C} \text{ (continuous)}
\end{aligned}
$$

The LP sub-problem is solved using **COIN-OR CLP** (Constraint Linear Programming) solver.

### 5.3 Initialization Strategies

RL-SPH supports two initialization methods:

1. **LP-based**: Start from LP-relaxation solution (rounded for integer variables)
2. **Random**: Random assignment within bounds

Experimental results show that random initialization performs comparably to LP initialization, suggesting RL-SPH can find feasible solutions even from poor starting points.

---

## 6. Experimental Results

### 6.1 Benchmark Datasets

RL-SPH was evaluated on five NP-hard MILP benchmarks:

| Dataset | Description | Variables | Constraints |
|---------|-------------|-----------|-------------|
| IS | Independent Set | 1500 nodes | ~3000 |
| CA | Combinatorial Auction | 4000 items | 2715 |
| SC | Set Covering | 3000 | 2000 |
| MVC | Minimum Vertex Cover | 3000 nodes | 11931 |
| NBI | Non-Binary Integers | 2000 integer | 2000 |

### 6.2 Evaluation Metrics

1. **Feasibility Rate (FR)**: Percentage of instances where at least one feasible solution is found
2. **Primal Gap (PG)**: Quality of solution relative to best-known solution (BKS)
   $$
   \text{PG}(\text{obj}_b) = \frac{|\text{obj}_b - \text{BKS}|}{\max(|\text{obj}_b|, |\text{BKS}|, \epsilon)} \times 100
   $$
3. **Primal Integral (PI)**: Speed of convergence
   $$
   \text{PI}(T) = \sum_{t=1}^{T} \text{PG}(\text{obj}_{b,t})
   $$

### 6.3 Comparison with Baselines

RL-SPH was compared against four traditional primal heuristics:

| Method | Description |
|--------|-------------|
| Diving | SCIP's 15 diving heuristics |
| FP (Feasibility Pump) | Alternates between LP/ILP solutions |
| Rounding | SCIP's 6 rounding heuristics |
| RENS | Relaxation Enforced Neighborhood Search |

**Key Results**:

| Dataset | FR (%) RL-SPH | PG (%) RL-SPH | Best Baseline PG |
|---------|---------------|---------------|------------------|
| IS | 100 | 0.14 | 18.09 (Rounding) |
| CA | 100 | 3.82 | 12.02 (Rounding) |
| SC | 100 | 9.67 | 0.00 (Diving) |
| MVC | 100 | 0.23 | 6.76 (FP) |
| NBI | 100 | 0.14 | N/A |

**Summary**:
- **44× lower primal gap** on average compared to baselines
- **2.3× lower primal integral** (faster convergence)
- **100% feasibility rate** on all benchmarks

### 6.4 Ablation Study

An ablation study on GNN architecture components:

| Components | FR (%) | PG (%) |
|------------|--------|--------|
| No components (∅) | 0 | N/A |
| RC (Reward Context) + PSL (Phase-Separated Layers) | 100 | 33.98 |
| TE (Transformer Encoder) + RC | 100 | 12.93 |
| Full (TE + RC + PSL) | 100 | **0.62** |

The Transformer encoder is crucial for feasibility (0% FR without it), and all components together achieve the best results.

---

## 7. Key Data Structures

### 7.1 RLState

```cpp
struct RLState {
    std::vector<double> x;    // Current solution
    std::vector<double> f;    // Constraint satisfaction: f = b - Ax
    double obj;               // Current objective value
    std::vector<double> lhs;  // Left-hand side: Ax
    int n;                    // Number of variables
    int m;                    // Number of constraints
};
```

### 7.2 BipartiteGraph

```cpp
struct BipartiteGraph {
    int num_vars;
    int num_constraints;
    std::vector<std::vector<int>> constraint_to_vars;    // Adjacency
    std::vector<std::vector<int>> var_to_constraints;    // Adjacency
    std::vector<double> scaled_A_row;                    // Scaled coefficients
    std::vector<int> A_row_ptr, A_col_idx;               // CSR format
    std::vector<std::pair<int, int>> edges;              // Edge list
};
```

### 7.3 Action Enum

```cpp
enum class Action : int {
    DECREASE = -1,    // x_i ← x_i - 1
    NO_CHANGE = 0,    // x_i ← x_i
    INCREASE = 1      // x_i ← x_i + 1
};
```

---

## 8. Conclusion

RL-SPH represents a significant advancement in learning-based primal heuristics for MILP:

1. **First end-to-end start primal heuristic** capable of independently generating feasible solutions for MILPs with general integer variables
2. **Transformer-based GNN architecture** effectively captures long-range variable-constraint dependencies
3. **Two-phase reward system** guides learning: feasibility first, then optimization
4. **MILP adaptation** handles continuous variables via LP sub-problems
5. **Superior empirical performance**: 44× lower primal gap, 2.3× lower primal integral compared to traditional heuristics

### 8.1 Future Directions

1. **Experience replay**: Store and reuse trajectories for more efficient training
2. **GPU acceleration**: Leverage LibTorch CUDA for faster training
3. **Reduced cost features**: Incorporate LP dual information for better guidance
4. **Transfer learning**: Pre-train on synthetic instances, fine-tune on specific problem classes
5. **Integration with branch-and-bound**: Use RL-SPH as a node heuristic within a MIP solver

---

## References

1. Lee, T.-H., Kim, M.-S. "RL-SPH: Learning to Achieve Feasible Solutions for Integer Linear Programs", arXiv:2411.19517v6, 2025.

2. Gasse, M. et al. "Exact combinatorial optimization with graph convolutional neural networks", NeurIPS 2019.

3. Han, Q. et al. "A GNN-guided predict-and-search framework for MILP", ICLR 2023.

4. Bengio, Y., Lodi, A., Prouvost, A. "Machine learning for combinatorial optimization: a methodological tour d'horizon", European Journal of Operational Research, 2021.

5. Berthold, T. "Primal heuristics for mixed integer programs", PhD thesis, Zuse Institute Berlin, 2006.
