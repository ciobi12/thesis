# Row-Based Reinforcement Learning Architecture

## Conceptual Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INPUT IMAGE (H x W)                                 │
│                   (Noisy CT-like / Vessel / Root scan)                       │
└────────────────────────────────────────┬────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ROW-BY-ROW SEQUENTIAL PROCESSING                         │
│                    (Bottom → Top or Top → Bottom)                            │
└─────────────────────────────────────────────────────────────────────────────┘

                    ╔═════════════════════════════════════╗
                    ║      ROW t (CURRENT DECISION)       ║
                    ╚═════════════════════════════════════╝
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
            │  CURRENT ROW │ │  HISTORY     │ │  FUTURE      │
            │   PIXELS     │ │  (K prev     │ │  (K next     │
            │   (W x C)    │ │   rows)      │ │   rows)      │
            └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
                   │                │                │
                   └────────────┬───┴────────────────┘
                                ▼
                    ┌─────────────────────────┐
                    │   CNN ENCODER NETWORK   │
                    │   ┌──────────────────┐  │
                    │   │ Row Encoder (64) │  │
                    │   ├──────────────────┤  │
                    │   │ Hist Encoder(32) │  │
                    │   ├──────────────────┤  │
                    │   │Future Encoder(32)│  │
                    │   └──────────────────┘  │
                    │   ┌──────────────────┐  │
                    │   │ Attention Layer  │  │
                    │   └──────────────────┘  │
                    │   ┌──────────────────┐  │
                    │   │ Decision Layers  │  │
                    │   └──────────────────┘  │
                    └────────────┬────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │  Q-VALUES: (W x 2)      │
                    │  [Background | Path]    │
                    │  for each pixel         │
                    └────────────┬────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │  ACTION: (W,)           │
                    │  Binary decision        │
                    │  per pixel [0 or 1]     │
                    └────────────┬────────────┘
                                 ▼
            ╔═══════════════════════════════════════╗
            ║           REWARD CALCULATION          ║
            ╚═══════════════════════════════════════╝
                   │              │              │
        ┌──────────┴───┐  ┌───────┴────────┐  ┌─┴───────────┐
        │ BASE REWARD  │  │ CONTINUITY REW │  │ GRADIENT REW│
        │ (Accuracy)   │  │ (Smoothness)   │  │ (Intensity) │
        │              │  │                │  │             │
        │ -|a - gt|    │  │  Decay-based   │  │ Transition  │
        │              │  │  linking to    │  │ smoothness  │
        │              │  │  prev rows     │  │             │
        └──────────┬───┘  └───────┬────────┘  └─┬───────────┘
                   └──────────┬───┴──────────────┘
                              ▼
                    ┌───────────────────┐
                    │ TOTAL REWARD (R)  │
                    │ = Σ pixel_rewards │
                    └─────────┬─────────┘
                              ▼
                    ┌───────────────────────────┐
                    │   UPDATE BUFFERS          │
                    │   • prev_preds_buffer     │
                    │   • prev_rows_buffer      │
                    └─────────┬─────────────────┘
                              ▼
                    ┌───────────────────────────┐
                    │   STORE TRANSITION        │
                    │   (s, a, r, s', done)     │
                    │   → REPLAY BUFFER         │
                    └─────────┬─────────────────┘
                              │
            ╔═════════════════╧═════════════════╗
            ║         DQN TRAINING LOOP         ║
            ╠═══════════════════════════════════╣
            ║  1. Sample batch from replay      ║
            ║  2. Compute Q(s,a) - policy net   ║
            ║  3. Compute target = r + γ·max   ║
            ║     Q(s',a') - target net         ║
            ║  4. Loss = MSE(Q(s,a), target)    ║
            ║  5. Backprop & update policy net  ║
            ║  6. Periodically: target ← policy ║
            ╚═══════════════════════════════════╝
                              │
                              ▼
                    Move to NEXT ROW (t+1)
                              │
                              ▼
               ┌──────────────────────────────┐
               │  REPEAT UNTIL ALL ROWS       │
               │  PROCESSED (terminated=True) │
               └──────────────────────────────┘
                              │
                              ▼
            ╔═════════════════════════════════════╗
            ║    FINAL RECONSTRUCTION RESULT      ║
            ║    (H x W binary prediction)        ║
            ║    ┌─────────────────────────────┐  ║
            ║    │ ░░░░███░░░░░░███░░░░░░░     │  ║
            ║    │ ░░███░███░░███░███░░░░      │  ║
            ║    │ ░███░░░███████░░███░░       │  ║
            ║    │ ███░░░░░███░░░░░███░        │  ║
            ║    └─────────────────────────────┘  ║
            ╚═════════════════════════════════════╝
```

## Key Components Breakdown

### 1. **Sequential Processing Strategy**
- **Direction**: Bottom → Top (for roots) or Top → Bottom
- **Unit of decision**: One complete row at a time
- **State**: Current row pixels + K previous rows (history) + K future rows (lookahead)
- **Episode**: One complete image traversal

### 2. **Observation Space** (Dict)
```python
{
    "row_pixels": (W, C),           # Current row image data
    "prev_preds": (history_len, W), # K previous predictions (NOT used by network)
    "prev_rows": (history_len, W, C),  # K previous image rows
    "future_rows": (future_len, W, C), # K upcoming image rows (lookahead)
    "row_index": (1,)               # Normalized position [0,1]
}
```

### 3. **Action Space**
- **Type**: `MultiBinary(W)` - per-pixel binary decision
- **Meaning**: For each of W pixels: 0 = background, 1 = path/vessel
- **Selection**: Epsilon-greedy over Q-values (W × 2 matrix)

### 4. **Neural Network Architecture**
```
PerPixelCNNWithHistory:
├─ Row Encoder: Conv1d(C → 64 → 64)
├─ History Encoder: Conv1d(history_len*C → 32 → 32)
├─ Future Encoder: Conv1d(future_len*C → 32 → 32)
├─ Attention Layer: 128 → 64 (sigmoid gating)
└─ Decision Layer: 128 → 128 → 64 → 2
```
- **Input**: (B, W, C) row + (B, history_len, W, C) + (B, future_len, W, C)
- **Output**: (B, W, 2) Q-values for [background, path] per pixel

### 5. **Reward Components**

#### Base Reward (Accuracy)
```python
base_reward = -base_coef * |action - ground_truth|
```
Per-pixel L1 distance to ground truth mask.

#### Continuity Reward (Vertical Connectivity)
```python
# For each active pixel in current row:
#   Check 5-pixel neighborhood in previous row
#   Reward = 0 if connected to previous structure
#   Penalty = -1.0 if isolated (not connected)
continuity_reward = continuity_coef * linking_term
```
Ensures vertical path coherence across rows.

#### Gradient Reward (Intensity Smoothness)
```python
# For each active pixel:
#   Compare intensity to nearest active pixel group in prev row
#   Reward smooth transitions, penalize intensity jumps
gradient_reward = -gradient_coef * intensity_difference
```
Encourages segmentation along smooth intensity regions.

### 6. **Training Workflow**

**Per Episode (one image):**
```python
for each row t in [0, H):
    obs = get_observation(row_t, history, future)
    q_values = policy_net(obs)              # (W, 2)
    action = epsilon_greedy(q_values)       # (W,)
    reward = compute_reward(action, gt)     # scalar
    next_obs = get_observation(row_t+1, ...)
    
    replay_buffer.push(obs, action, reward, next_obs, done)
    
    if len(replay_buffer) >= batch_size:
        batch = replay_buffer.sample(batch_size)
        loss = MSE(Q(s,a), r + γ·max Q(s',a'))
        optimizer.step()
    
    update_history_buffers(action)
```

**Hyperparameters:**
- `history_len`: 3-5 rows (memory depth)
- `future_len`: 3 rows (lookahead)
- `continuity_coef`: 0.1-0.2 (smoothness weight)
- `gradient_coef`: 0.1-1.0 (intensity consistency)
- `epsilon`: 1.0 → 0.01 (exploration decay)
- `gamma`: 0.99 (discount factor)
- `batch_size`: 32-64
- `target_update_every`: 500 steps

### 7. **Advantages of Row-Based Approach**
✓ **Natural sequential structure** for elongated branching patterns  
✓ **Enforces connectivity** through continuity rewards  
✓ **Handles occlusions** via historical context  
✓ **Per-pixel decisions** capture fine-grained detail  
✓ **Efficient inference** (linear in H)

### 8. **Key Design Choices**

| Choice | Rationale |
|--------|-----------|
| Row-by-row traversal | Aligns with growth direction (roots, vessels) |
| Binary action per pixel | Simplifies Q-learning, reduces action space |
| History + Future context | Bridges gaps, provides directional cues |
| Decay-based continuity | Balances flexibility vs smoothness |
| Replay buffer | Breaks temporal correlation, stable learning |
| Target network | Reduces Q-value overestimation |

### 9. **Metrics Tracked**
- **Training**: IoU, F1, Accuracy, Coverage, Precision, Recall
- **Per-row**: True Positives, False Positives, False Negatives
- **Reward decomposition**: Base, Continuity, Gradient contributions

### 10. **Output**
- **Per-episode**: (H × W) binary reconstruction
- **Visualization**: Side-by-side {Original | Ground Truth | Prediction}
- **Storage**: Results saved to `dqn_row_based/results/{dataset}/`
