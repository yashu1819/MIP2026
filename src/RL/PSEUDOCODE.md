# RL-SPH: Complete Pseudocode Reference

This document contains the complete pseudocode for the RL-SPH (Reinforcement Learning Start Primal Heuristic) implementation.

---

## Table of Contents

1. [Data Structures](#1-data-structures)
2. [Algorithm 1: Solution Search](#2-algorithm-1-solution-search)
3. [Algorithm 2: Training Loop](#3-algorithm-2-training-loop)
4. [Algorithm 3: Variable Selection](#4-algorithm-3-variable-selection)
5. [Reward Computation](#5-reward-computation)
6. [Feature Engineering](#6-feature-engineering)
7. [Neural Network Architecture](#7-neural-network-architecture)

---

## 1. Data Structures

### 1.1 RL State (S_t)

```
struct RLState:
    x: vector<double>      // Current integer solution (n variables)
    f: vector<double>      // Constraint satisfaction: f = b - Ax (m constraints)
    lhs: vector<double>    // Left-hand side: Ax
    obj: double            // Objective value: c^T x
    n: int                 // Number of variables
    m: int                 // Number of constraints
```

### 1.2 Action Space

```
enum Action:
    DECREASE = -1    // Decrease variable by 1
    NO_CHANGE = 0    // Keep variable unchanged
    INCREASE = 1     // Increase variable by 1
```

### 1.3 Bipartite Graph

```
struct BipartiteGraph:
    num_vars: int
    num_constraints: int
    constraint_to_vars: vector<vector<int>>      // Adjacency: constraint -> variables
    var_to_constraints: vector<vector<int>>      // Adjacency: variable -> constraints
    scaled_A_row: vector<double>                 // Scaled constraint matrix (CSR values)
    A_row_ptr: vector<int>                       // CSR row pointers
    A_col_idx: vector<int>                       // CSR column indices
    edges: vector<pair<int, int>>                // (var_idx, constraint_idx) pairs
```

### 1.4 Configuration

```
struct RLConfig:
    time_limit: double = 60.0      // Time limit in seconds
    max_steps: int = 2000          // Maximum timesteps
    phase1_steps: int = 500        // Steps in feasibility phase
    init_method: string = "lp"     // "lp" or "random"
    train_mode: bool = false       // Training vs inference
    model_path: string = ""        // Path to model weights
    seed: int = 42                 // Random seed
```

---

## 2. Algorithm 1: Solution Search

**Purpose:** Run RL-SPH heuristic to find feasible/improved solution

```
ALGORITHM 1: Solution Search (run_rl_heuristic)

INPUT:  MIP problem M, configuration config
OUTPUT: Best solution x_b found

1:  // Initialize
2:  x_0 <- INITIALIZE_SOLUTION(M, config.init_method)
3:  S_0 <- CREATE_STATE(M, x_0)
4:  x_b <- empty, obj_b <- infinity, found_feasible <- false
5:  phase <- 1, step <- 0
6:
7:  // Check if initial solution is feasible
8:  if IS_FEASIBLE(M, S_0) then
9:      x_b <- S_0.x, obj_b <- S_0.obj, found_feasible <- true
10:
11: // Main search loop
12: while step < config.max_steps AND elapsed_time < config.time_limit do
13:     // === Algorithm 1 Step 1: Select Variables ===
14:     S_tilde_t <- SELECT_VARIABLES(S_t, phase)  // See Algorithm 3
15:
16:     // === Algorithm 1 Step 2: Get Actions ===
17:     A_t <- AGENT.select_actions(S_t, graph, S_tilde_t)
18:
19:     // === Algorithm 1 Step 3: Apply Actions ===
20:     x_{t+1} <- APPLY_ACTIONS(S_t.x, A_t, S_tilde_t)
21:
22:     // === Algorithm 1 Step 4: Observe New State ===
23:     S_{t+1} <- CREATE_STATE(M, x_{t+1})
24:
25:     // === Algorithm 1 Step 5: Compute Reward ===
26:     R_t <- COMPUTE_REWARD(S_t, S_{t+1}, obj_b, phase)  // See Section 5
27:
28:     // === Algorithm 1 Step 6: Update Incumbent ===
29:     if IS_FEASIBLE(M, S_{t+1}) AND S_{t+1}.obj < obj_b then
30:         obj_b <- S_{t+1}.obj
31:         x_b <- S_{t+1}.x
32:         found_feasible <- true
33:
34:     // === Algorithm 1 Step 7: Rollback Decision ===
35:     should_rollback <- false
36:
37:     if phase == 1 then
38:         // Phase 1: Rollback if bounds violated
39:         if NOT BOUNDS_SATISFIED(M, S_{t+1}) then
40:             should_rollback <- true
41:     else
42:         // Phase 2: Rollback unless better feasible
43:         if NOT IS_FEASIBLE(M, S_{t+1}) OR S_{t+1}.obj >= obj_b then
44:             should_rollback <- true
45:
46:     if should_rollback then
47:         S_{t+1} <- S_t  // Restore previous state
48:
49:     // === Phase Transition ===
50:     if phase == 1 AND step >= config.phase1_steps then
51:         phase <- 2
52:
53:     step <- step + 1
54:
55: return CREATE_SOLUTION(x_b, obj_b, found_feasible)
```

### Helper: Initialize Solution

```
FUNCTION INITIALIZE_SOLUTION(M, method):
    x <- vector of size M.num_cols

    if method == "random" then
        // Set ~1% of variables to 1
        num_ones <- max(1, M.num_cols / 100)
        indices <- [0, 1, ..., M.num_cols - 1]
        SHUFFLE(indices)
        x <- zeros
        for i in 0 to num_ones - 1 do
            x[indices[i]] <- 1.0
    else
        // LP-relaxation based (simplified)
        for i in 0 to M.num_cols - 1 do
            if M.vartype[i] == CONTINUOUS then
                x[i] <- M.lb[i]
            else if M.vartype[i] == BINARY then
                x[i] <- (M.lb[i] >= 0.5) ? 1.0 : 0.0
            else  // Integer
                x[i] <- floor(M.lb[i] + 0.5)
            x[i] <- clamp(x[i], M.lb[i], M.ub[i])

    return x
```

### Helper: Apply Actions

```
FUNCTION APPLY_ACTIONS(x, actions, changeable_indices):
    x_new <- copy of x
    for i in 0 to length(actions) - 1 do
        var_idx <- changeable_indices[i]
        x_new[var_idx] <- x_new[var_idx] + int(actions[i])
    return x_new
```

### Helper: Create State

```
FUNCTION CREATE_STATE(M, x):
    state.n <- M.num_cols
    state.m <- M.num_rows
    state.x <- x
    state.lhs <- COMPUTE_LHS(M, x)
    state.f <- M.b - state.lhs
    state.obj <- M.obj_offset + dot(M.c, x)
    return state

FUNCTION COMPUTE_LHS(M, x):
    lhs <- vector of size M.num_rows
    for i in 0 to M.num_rows - 1 do
        sum <- 0.0
        for j in M.csr_row_ptr[i] to M.csr_row_ptr[i+1] - 1 do
            col <- M.csr_col_idx[j]
            sum <- sum + M.csr_val[j] * x[col]
        lhs[i] <- sum
    return lhs
```

---

## 3. Algorithm 2: Training Loop

**Purpose:** Train the RL agent using policy gradient with baseline

```
ALGORITHM 2: Training Loop (train_rl_heuristic)

INPUT:  Training instances {M_1, ..., M_N}, configuration config
OUTPUT: Trained agent parameters theta

1:  // Initialize agent
2:  theta <- RANDOM_INITIALIZATION()
3:  optimizer <- RMSprop(lr = config.learning_rate)
4:  replay_buffer <- empty
5:
6:  for update = 1 to config.num_updates do
7:      // Sample mini-batch of instances
8:      batch_rewards <- []
9:
10:     for b = 1 to config.batch_size do
11:         // Sample random instance
12:         k <- UNIFORM_RANDOM(1, N)
13:         M <- M_k
14:
15:         // Run episode (Algorithm 1 without rollback)
16:         episode_data <- RUN_TRAINING_EPISODE(M, theta)
17:         batch_rewards <- batch_rewards + episode_data.total_reward
18:
19:         // Store experience
20:         replay_buffer.push(episode_data.transitions)
21:
22:     // Compute advantage estimates
23:     for each transition (S_t, A_t, R_t, S_{t+1}) in replay_buffer do
24:         // TD error: delta = R + gamma * V(S') - V(S)
25:         V_t <- CRITIC_FORWARD(S_t, theta_c)
26:         V_{t+1} <- CRITIC_FORWARD(S_{t+1}, theta_c)
27:         delta_t <- R_t + config.gamma * V_{t+1} - V_t
28:
29:         // Store advantage
30:         advantage_t <- delta_t
31:
32:     // Policy gradient update
33:     for each transition in replay_buffer do
34:         log_prob <- log(PI(A_t | S_t, theta_a))
35:         loss <- -log_prob * advantage_t
36:
37:         // Actor update
38:         gradient <- BACKPROP(loss, theta_a)
39:         optimizer.step(theta_a, gradient)
40:
41:         // Critic update (TD loss)
42:         critic_loss <- delta_t^2
43:         gradient <- BACKPROP(critic_loss, theta_c)
44:         optimizer.step(theta_c, gradient)
45:
46:     // Clear replay buffer
47:     replay_buffer.clear()
48:
49:     // Log progress
50:     if update % config.log_interval == 0 then
51:         PRINT("Update", update, ": avg_reward =", mean(batch_rewards))
52:
53: return theta
```

### Training Episode

```
FUNCTION RUN_TRAINING_EPISODE(M, theta):
    // Initialize
    prev_num_ones <- 0
    x <- INITIALIZE_SOLUTION_TRAINING(M, prev_num_ones)
    S <- CREATE_STATE(M, x)
    incumbent <- empty, obj_incumbent <- infinity
    phase <- 1, stay_phase1 <- true
    total_reward <- 0
    transitions <- []
    step <- 0
    phase1_count <- 0

    while step < config.max_steps do
        prev_S <- S

        // Select variables
        changeable <- SELECT_VARIABLES(S, phase)

        // Get actions from policy
        A <- AGENT.select_actions(S, changeable, theta)

        // Apply actions
        x_new <- APPLY_ACTIONS(S.x, A, changeable)
        S <- CREATE_STATE(M, x_new)

        // Compute reward
        R <- COMPUTE_REWARD(prev_S, S, obj_incumbent, phase)
        total_reward <- total_reward + R

        // Update incumbent
        if IS_FEASIBLE(M, S) AND S.obj < obj_incumbent then
            incumbent <- S.x
            obj_incumbent <- S.obj

        // Phase transition
        if stay_phase1 AND (IS_FEASIBLE(M, S) OR step >= config.phase1_steps) then
            if step >= config.phase1_steps then
                stay_phase1 <- false
                phase <- 2
        else if NOT stay_phase1 AND IS_FEASIBLE(M, S) then
            phase <- 2

        // Store transition
        transitions.append((prev_S, A, R, S, phase))

        step <- step + 1

    return {total_reward, transitions}
```

### Initialize Solution for Training

```
FUNCTION INITIALIZE_SOLUTION_TRAINING(M, prev_num_ones):
    x <- zeros(M.num_cols)

    if prev_num_ones == 0 then
        // First instance: 1% of variables
        num_ones <- max(1, M.num_cols / 100)
    else
        // Subsequent: half of previous
        num_ones <- prev_num_ones / 2

    indices <- [0, 1, ..., M.num_cols - 1]
    SHUFFLE(indices)

    for i in 0 to num_ones - 1 do
        x[indices[i]] <- 1.0

    prev_num_ones <- num_ones
    return x
```

---

## 4. Algorithm 3: Variable Selection

**Purpose:** Select which variables to modify at each timestep

```
ALGORITHM 3: Variable Selection (select_variables)

INPUT:  State S_t, current phase
OUTPUT: Indices of changeable variables (size = p + q)

1:  n <- S_t.n, m <- S_t.m
2:  p <- floor(log2(n)), q <- p  // Number of seed and neighbor variables
3:  p <- max(1, p), q <- max(1, q)
4:
5:  // === Seed Selection ===
6:  score_seed <- vector of size n, initialized to 0
7:
8:  if phase == 1 then  // Feasibility phase
9:      // Select variables in VIOLATED constraints
10:     for i in 0 to n - 1 do
11:         for each constraint j in var_to_constraints[i] do
12:             if S_t.f[j] < 0 then  // Violated
13:                 score_seed[i] <- score_seed[i] + 1
14:
15:     // Weight: prefer low-cost variables
16:     max_c <- max(|c_i| for all i)
17:     for i in 0 to n - 1 do
18:         weight <- (max_c - |c_i| + 1) / max_c
19:         score_seed[i] <- score_seed[i] * weight
20:
21: else  // phase == 2, Optimization phase
22:     // Select variables in LOW-RISK (well-satisfied) constraints
23:     for i in 0 to n - 1 do
24:         for each constraint j in var_to_constraints[i] do
25:             if S_t.f[j] > 0 then  // Well-satisfied
26:                 score_seed[i] <- score_seed[i] + 1
27:
28:     // Weight: prefer high-impact variables
29:     max_c <- max(|c_i| for all i)
30:     for i in 0 to n - 1 do
31:         weight <- |c_i| / max_c
32:         score_seed[i] <- score_seed[i] * weight
33:
34:     // Invert: higher score = lower risk = more likely to select
35:     max_score <- max(score_seed)
36:     for i in 0 to n - 1 do
37:         score_seed[i] <- max_score - score_seed[i] + 1
38:
39: // Sample p seed variables (weighted sampling)
40: seed_indices <- SAMPLE_WEIGHTED(score_seed, k = p)
41:
42: // === Neighbor Selection ===
43: // Count how many seeds share each constraint
44: constraint_counts <- vector of size m, zeros
45: for each seed_idx in seed_indices do
46:     for each constraint j in var_to_constraints[seed_idx] do
47:         constraint_counts[j] <- constraint_counts[j] + 1
48:
49: // Score neighbors: variables sharing constraints with seeds
50: score_neighbor <- vector of size n, zeros
51: for i in 0 to n - 1 do
52:     for each constraint j in var_to_constraints[i] do
53:         score_neighbor[i] <- score_neighbor[i] + constraint_counts[j]
54:
55: // Prevent selecting seed variables as neighbors
56: for each seed_idx in seed_indices do
57:     score_neighbor[seed_idx] <- -1  // Mark as invalid
58:
59: // Select top q neighbors
60: neighbor_indices <- TOP_K(score_neighbor, k = q)
61:
62: // Combine seeds and neighbors
63: changeable <- seed_indices + neighbor_indices
64:
65: return changeable
```

### Weighted Sampling

```
FUNCTION SAMPLE_WEIGHTED(scores, k):
    result <- []
    total <- sum(scores[i] for i where scores[i] > 0)

    if total < epsilon then
        // Uniform random if all scores are zero
        indices <- [0, 1, ..., n-1]
        SHUFFLE(indices)
        return first k elements of indices

    // Roulette wheel selection
    selected <- vector of booleans, size n, all false

    while length(result) < k AND length(result) < n do
        r <- UNIFORM_RANDOM(0, total)
        cumsum <- 0

        for i in 0 to n - 1 do
            if selected[i] OR scores[i] <= 0 then continue
            cumsum <- cumsum + scores[i]
            if cumsum >= r then
                selected[i] <- true
                result.append(i)
                break

    return result
```

### Top-K Selection

```
FUNCTION TOP_K(scores, k):
    scored_pairs <- []
    for i in 0 to n - 1 do
        if scores[i] >= 0 then  // Not marked invalid
            scored_pairs.append((scores[i], i))

    SORT scored_pairs by score DESCENDING
    return first k indices from scored_pairs
```

---

## 5. Reward Computation

**Purpose:** Compute reward signal for policy learning

### Total Reward

```
FUNCTION COMPUTE_REWARD(S_prev, S_curr, obj_incumbent, phase):
    // Exploration penalty
    r_explore <- 0
    if S_curr.x == S_prev.x (no move made) then
        r_explore <- -100

    // Phase-specific reward
    if phase == 1 then
        r_opt <- PHASE1_REWARD(S_prev, S_curr, obj_incumbent)
    else
        r_opt <- PHASE2_REWARD(S_prev, S_curr, obj_incumbent)

    return r_opt + r_explore
```

### Feasibility Reward Component

```
FUNCTION FEASIBILITY_REWARD(S_prev, S_curr):
    num_changeable <- number of variables being modified

    // Bound reward: -1 for each violated bound
    r_bound <- 0
    for i in 0 to S_curr.n - 1 do
        if S_curr.x[i] < lb[i] OR S_curr.x[i] > ub[i] then
            r_bound <- r_bound - 1

    // Constraint reward: improvement in violations
    r_const <- 0
    for j in 0 to S_curr.m - 1 do
        prev_violation <- min(S_prev.f[j], 0)
        curr_violation <- min(S_curr.f[j], 0)
        r_const <- r_const + (curr_violation - prev_violation)

    // Normalized combination
    if num_changeable > 0 then
        return r_bound + r_const / sqrt(num_changeable)
    else
        return r_bound
```

### Phase 1 Reward (Feasibility Focus)

```
FUNCTION PHASE1_REWARD(S_prev, S_curr, obj_incumbent):
    r_bound <- BOUND_REWARD(S_curr)
    r_const <- CONSTRAINT_REWARD(S_prev, S_curr)
    delta_obj <- |S_curr.obj - S_prev.obj| / max(|c|)

    bounds_ok <- all bounds satisfied in S_curr
    obj_better <- S_curr.obj < S_prev.obj

    // Case 1: Constraint improvement, obj better, bounds violated
    if r_const >= 0 AND obj_better AND NOT bounds_ok then
        return r_bound

    // Case 2: Constraint improvement, obj worse, bounds violated
    if r_const >= 0 AND NOT obj_better AND NOT bounds_ok then
        return r_bound - delta_obj

    // Case 3: Constraint improvement, obj better, bounds satisfied
    if r_const >= 0 AND obj_better AND bounds_ok then
        return FEASIBILITY_REWARD(S_prev, S_curr) + delta_obj

    // Case 4: Constraint deterioration or obj worse
    if r_const < 0 OR (NOT obj_better AND bounds_ok) then
        return FEASIBILITY_REWARD(S_prev, S_curr) - delta_obj

    // Case 5: Otherwise
    return FEASIBILITY_REWARD(S_prev, S_curr)
```

### Phase 2 Reward (Optimization Focus)

```
FUNCTION PHASE2_REWARD(S_prev, S_curr, obj_incumbent):
    alpha <- 2.0  // Toward-optimal bias
    feasible <- IS_FEASIBLE(M, S_curr)
    better <- S_curr.obj < obj_incumbent
    delta_obj <- |S_curr.obj - S_prev.obj| / max(|c|)
    r_feas <- FEASIBILITY_REWARD(S_prev, S_curr)

    if feasible then
        if better then
            // Case 1: Better than incumbent
            return delta_obj
        else
            // Case 2: Worse than incumbent
            return -delta_obj * alpha
    else
        if better then
            // Case 3: Infeasible but objective improved
            return r_feas
        else
            // Case 4: Infeasible and objective worse
            return r_feas * alpha
```

---

## 6. Feature Engineering

### Variable Features (64-dimensional)

```
FUNCTION BUILD_VARIABLE_FEATURES(M, graph, S_t):
    features <- matrix of size (n_vars x 64)

    // Precompute scaling
    max_c <- max(|c_i|)
    max_b <- max(|b_j|)
    max_degree <- max variable degree

    for each variable i do
        f <- features[i]  // 64-dim vector
        idx <- 0

        // Feature 0: Scaled objective coefficient
        f[idx++] <- c_i / max_c

        // Feature 1-2: Normalized bounds
        f[idx++] <- lb_i / max_b
        f[idx++] <- ub_i / max_b

        // Feature 3: Current value (normalized)
        f[idx++] <- x_i / max_b

        // Feature 4: Bound indicator
        f[idx++] <- 1.0 if (x_i <= lb_i OR x_i >= ub_i) else 0.0

        // Features 5-36: Periodic embedding of x_i (32 dimensions)
        pe <- PERIODIC_EMBED(x_i, k=16)
        for j in 0 to 31 do
            f[idx++] <- pe[j]

        // Feature 37: Normalized degree
        degree <- number of constraints containing variable i
        f[idx++] <- degree / max_degree

        // Feature 38: Centrality
        centrality <- 1.0 / avg_degree_of_neighboring_constraints
        f[idx++] <- normalize(centrality)

        // Feature 39: Integer indicator
        f[idx++] <- 1.0 if vartype[i] is integer else 0.0

        // Feature 40: Binary indicator
        f[idx++] <- 1.0 if vartype[i] is binary else 0.0

        // Features 41-63: Reserved (zero-padded)
        while idx < 64 do
            f[idx++] <- 0.0

    return features
```

### Periodic Embedding

```
FUNCTION PERIODIC_EMBED(value, k=16):
    // PE(z) = concat(sin(2*pi*w_i*z), cos(2*pi*w_i*z))
    // Frequencies: w_i = 2^i for i = 0, ..., k-1

    embedding <- vector of size 2k

    for i in 0 to k-1 do
        freq <- 2^i
        angle <- 2 * pi * freq * value
        embedding[i] <- sin(angle)
        embedding[k + i] <- cos(angle)

    return embedding
```

### Constraint Features (32-dimensional)

```
FUNCTION BUILD_CONSTRAINT_FEATURES(M, graph, S_t):
    features <- matrix of size (n_constraints x 32)

    max_b <- max(|b_j|)

    for each constraint j do
        f <- features[j]  // 32-dim vector
        idx <- 0

        // Feature 0: Normalized RHS
        f[idx++] <- b_j / max_b

        // Feature 1: Constraint satisfaction (normalized)
        f[idx++] <- f_j / max_b  // where f_j = b_j - Ax

        // Feature 2: Slack indicator
        f[idx++] <- 1.0 if f_j >= 0 else 0.0

        // Feature 3: Violation magnitude
        f[idx++] <- max(0, -f_j / max_b)

        // Feature 4: Normalized degree
        degree <- number of variables in constraint
        f[idx++] <- degree / max_degree

        // Feature 5: Constraint type (all <= in current implementation)
        f[idx++] <- 0.0

        // Features 6-31: Encoded A row (simplified)
        vars <- variables in constraint j
        for k in 0 to min(|vars|, 26) - 1 do
            f[idx++] <- 1.0  // Placeholder for coefficient encoding

        // Zero-pad remaining
        while idx < 32 do
            f[idx++] <- 0.0

    return features
```

---

## 7. Neural Network Architecture

### Actor Network

```
STRUCT ActorNetwork:
    // Transformer-based GNN
    var_embedding: Linear(64, hidden_dim)      // Variable feature embedding
    transformer: TransformerEncoder(
        d_model = hidden_dim
        nhead = num_heads
        num_layers = num_layers
        dim_feedforward = hidden_dim * 4
        dropout = 0.1
    )
    action_head: Linear(hidden_dim, 3)         // 3 actions per variable

FUNCTION ACTOR_FORWARD(variable_features, changeable_indices):
    // variable_features: (n_vars, 64)
    // changeable_indices: indices of variables being modified

    // Embed variable features
    x <- RELU(var_embedding(variable_features))
    // x: (n_vars, hidden_dim)

    // Extract features for changeable variables only
    x_changeable <- x[changeable_indices]
    // x_changeable: (num_changeable, hidden_dim)

    // Add positional encoding for sequence
    x_changeable <- x_changeable + positional_encoding

    // Transformer encoding
    x_encoded <- transformer(x_changeable)

    // Action logits
    logits <- action_head(x_encoded)
    // logits: (num_changeable, 3) - one logit per action per variable

    return logits  // Action probabilities via softmax
```

### Critic Network

```
STRUCT CriticNetwork:
    // State value estimation
    phase_encoding: Embedding(2, hidden_dim)   // Phase 1 or 2
    obj_encoder: Linear(1, hidden_dim)         // Objective encoding
    constraint_encoder: Linear(num_constraints * 3, hidden_dim)
    value_head: Sequential(
        Linear(hidden_dim, hidden_dim),
        ReLU(),
        Linear(hidden_dim, 1)
    )

FUNCTION CRITIC_FORWARD(phase, obj, constraint_features):
    // phase: 1 or 2
    // obj: current objective value
    // constraint_features: (num_constraints, 3) - [f_j, slack, violation]

    // Encode phase
    phase_emb <- phase_encoding(phase)

    // Encode objective (normalized)
    obj_norm <- obj / (|obj| + epsilon)
    obj_emb <- RELU(obj_encoder([obj_norm]))

    // Encode constraint features
    constr_flat <- flatten(constraint_features)
    constr_emb <- RELU(constraint_encoder(constr_flat))

    // Combine embeddings
    combined <- phase_emb + obj_emb + constr_emb

    // Value estimation
    value <- value_head(combined)

    return value[0]  // Scalar value estimate
```

### Action Sampling

```
FUNCTION SAMPLE_ACTIONS(logits, rng):
    actions <- []

    for each variable's logits in logits do
        // Softmax to get probabilities
        probs <- SOFTMAX(logits_i)
        // probs: [P(decrease), P(no_change), P(increase)]

        // Sample from categorical distribution
        r <- UNIFORM_RANDOM(0, 1)
        cumsum <- 0

        for action in [DECREASE, NO_CHANGE, INCREASE] do
            cumsum <- cumsum + probs[action]
            if r < cumsum then
                actions.append(action)
                break

    return actions
```

### Policy Gradient Update

```
FUNCTION UPDATE_POLICY(states, actions, rewards, next_states, phases):
    // Compute TD errors (advantages)
    advantages <- []
    for i in 0 to length(states) - 1 do
        V_s <- CRITIC_FORWARD(states[i], phases[i])
        V_s_next <- CRITIC_FORWARD(next_states[i], phases[i])
        delta <- rewards[i] + gamma * V_s_next - V_s
        advantages.append(delta)

    // Actor loss: -log(pi(a|s)) * advantage
    actor_loss <- 0
    for i in 0 to length(states) - 1 do
        logits <- ACTOR_FORWARD(states[i])
        log_prob <- log(PROBABILITY(logits, actions[i]))
        actor_loss <- actor_loss - log_prob * advantages[i]

    // Critic loss: TD error squared
    critic_loss <- 0
    for i in 0 to length(states) - 1 do
        critic_loss <- critic_loss + advantages[i]^2

    // Backprop and optimize
    BACKPROP(actor_loss, actor_params)
    BACKPROP(critic_loss, critic_params)
    OPTIMIZER.step()

    return actor_loss, critic_loss
```

---

## 8. ILP → MILP Adaptation (Future Work)

**Problem:** The RL-SPH paper assumes all variables are integer. For MILPs, continuous variables must be handled differently.

### 8.1 Modified Variable Selection (✅ DONE — `rl_variable_selection.h`)

```
MODIFIED ALGORITHM 3: Variable Selection for MILP

// Filter: only consider integer/binary variables for RL actions
integer_vars <- []
for i in 0 to n-1 do
    if vartype[i] == INTEGER or vartype[i] == BINARY then
        integer_vars.append(i)

n_int <- length(integer_vars)
p <- floor(log2(n_int))
q <- p

// Seed scoring operates only on integer_vars
// ... (same as Algorithm 3, but restricted to integer_vars)

return changeable    // Only integer/binary variable indices
```

### 8.2 Modified Action Application (✅ DONE — `rl_state.h::apply_actions_milp()`)

```
FUNCTION apply_actions_milp(x, actions, changeable, mip):
    x_new <- copy(x)

    // Apply ±1 only to integer/binary variables
    for i in 0 to length(changeable)-1 do
        idx <- changeable[i]
        if mip.vartype[idx] == CONTINUOUS then
            continue    // Skip continuous variables
        x_new[idx] <- x_new[idx] + actions[i]

        // Post-action clamping
        if mip.vartype[idx] == BINARY then
            x_new[idx] <- clamp(x_new[idx], 0, 1)
        else
            x_new[idx] <- clamp(x_new[idx], mip.lb[idx], mip.ub[idx])
            x_new[idx] <- round(x_new[idx])

    // Solve LP sub-problem for continuous variables
    // Fix all integer variables, optimize over continuous
    x_continuous <- solve_lp_subproblem(mip, x_new, integer_fixed=true)
    for i in 0 to n-1 do
        if mip.vartype[i] == CONTINUOUS then
            x_new[i] <- x_continuous[i]

    return x_new
```

### 8.3 Additional Features for MILP (❌ TODO — future enhancement)

```
// Extra features per variable for MILP:
feature[k]   = lp_reduced_cost[i]      // From LP relaxation
feature[k+1] = is_at_lp_bound[i]       // 1 if var is at LP bound
feature[k+2] = lp_relaxation_value[i]  // LP relaxation solution value
feature[k+3] = fractionality[i]        // |x_LP - round(x_LP)| for integer vars
```

---

## Summary of Key Equations

### State Representation
- `S_t = (x_t, f_t, obj_t)` where:
  - `x_t`: current solution
  - `f_t = b - Ax_t`: constraint satisfaction
  - `obj_t = c^T x_t`: objective value

### Variable Selection
- `p = q = floor(log2(n_int))` integer/binary variables only (MILP-adapted)
- Phase 1 seed score: `sum over violated constraints`
- Phase 2 seed score: `sum over satisfied constraints`

### Reward Functions
- **Exploration penalty**: `R_explore = -100` if no move
- **Feasibility reward**: `R_F = R_bound + R_const / sqrt(num_changeable)`
- **Phase 1**: Feasibility-focused with objective bonus
- **Phase 2**: `R = delta_obj` (better) or `-alpha * delta_obj` (worse), `alpha = 2`

### Training
- **TD error**: `delta = R + gamma * V(s') - V(s)`
- **Actor loss**: `L_actor = -log(pi(a|s)) * delta`
- **Critic loss**: `L_critic = delta^2`

---

## Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Algorithm 1 (Solution Search) | ✅ Complete | `rl_heuristic.cpp` |
| Algorithm 2 (Training Loop) | ✅ Complete | `rl_training.cpp` |
| Algorithm 3 (Variable Selection) | ✅ Complete (MILP-aware) | `rl_variable_selection.h` |
| Reward Computation | ✅ Complete | `rl_reward.h` |
| Actor-Critic (LibTorch) | ✅ Complete | `rl_agent.h/cpp` |
| Actor-Critic (CPU fallback) | ✅ Complete | `rl_agent.h/cpp` |
| Feature Engineering | ✅ Complete | `rl_features.h` |
| MILP Variable Filtering | ✅ Complete | `rl_variable_selection.h` |
| MILP Action Clamping | ✅ Complete | `rl_state.h::apply_actions_milp()` |
| LP Sub-problem for Continuous | ✅ Complete | `rl_lp_subproblem.h` |
| Integrality Check | ✅ Complete | `rl_state.h::is_feasible()` |
| Training Driver | ✅ Complete | `main_train.cpp` |
| Inference Driver | ✅ Complete | `main_rl.cpp` |
| MILP Extra Features | ❌ TODO | Section 8.3 (LP reduced costs) |

---

## File Reference

| Algorithm | File | Key Functions |
|-----------|------|---------------|
| Solution Search (Alg 1) | `rl_heuristic.cpp` | `run()`, `search_step()` |
| Training (Alg 2) | `rl_training.cpp` | `train()`, `training_step()` |
| Variable Selection (Alg 3) | `rl_variable_selection.h` | `select_variables()` |
| Reward Computation | `rl_reward.h` | `compute_reward()`, `phase1_reward()`, `phase2_reward()` |
| State Management | `rl_state.h` | `create_state()`, `is_feasible()`, `apply_actions_milp()` |
| LP Sub-problem | `rl_lp_subproblem.h` | `solve_lp_subproblem()` |
| Graph Building | `rl_graph.h/cpp` | `build_graph()` |
| Features | `rl_features.h` | `build_variable_features()`, `build_constraint_features()` |
| Agent (LibTorch) | `rl_agent.h/cpp` | `select_actions()`, `select_actions_training()`, `update()` |
| Agent (CPU fallback) | `rl_agent.h/cpp` | `ActorNetworkCPU::forward()`, `CriticNetworkCPU::forward()` |
| Training Driver | `main_train.cpp` | `main()`, CLI argument parsing |
| Inference Driver | `main_rl.cpp` | `main()`, model loading |
