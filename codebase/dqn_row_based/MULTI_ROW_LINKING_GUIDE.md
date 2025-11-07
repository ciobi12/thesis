# Multi-Row Action Linking Strategies

## Overview
The goal is to link the **current row's actions** to **previous rows' actions** in the reward function to encourage vertical path continuity.

---

## Strategy 1: Exponential Decay (Currently Implemented)

### Concept
Weight each previous row by how recent it is. Recent rows have more influence.

### Implementation
```python
continuity_rewards = np.zeros((self.W,), dtype=np.float32)
decay_factor = 0.7  # tune this: 0.5-0.9

for i, prev_row in enumerate(reversed(list(self.prev_preds_buffer))):
    weight = (decay_factor ** i)  # 1.0, 0.7, 0.49, 0.34, ...
    continuity_rewards += -weight * np.abs(action - prev_row)

continuity_rewards *= self.continuity_coef
```

### When to use
- **Default choice** - works well for most cases
- Good when path changes gradually (smooth curves)
- Recent history is more reliable than distant history

### Tuning
- `decay_factor = 0.9`: Long memory, slow decay (for very smooth paths)
- `decay_factor = 0.7`: **Balanced (recommended)**
- `decay_factor = 0.5`: Short memory, fast decay (for wiggly paths)

---

## Strategy 2: Path Centroid (Average Position)

### Concept
Reward actions that align with the **average position** of predictions across all history rows.

### Implementation
```python
# Compute centroid of previous predictions
prev_stack = np.array(list(self.prev_preds_buffer))  # (history_len, W)
prev_centroid = prev_stack.mean(axis=0)  # (W,)

# Reward alignment with centroid
centroid_rewards = -self.continuity_coef * np.abs(action - prev_centroid)
pixel_rewards = base_rewards + centroid_rewards
```

### When to use
- Path has **consistent direction** (e.g., straight vertical lines)
- Want to penalize sudden deviations from overall trend
- Helpful for **filtering noise** (outlier rows get averaged out)

### Pros/Cons
- ✅ Robust to noise in individual rows
- ✅ Encourages straight, consistent paths
- ❌ Can be too rigid for curving paths
- ❌ Treats all history equally (no recency bias)

---

## Strategy 3: Vertical Alignment Score

### Concept
For each column `j`, check if action[j]=1 **and** any previous row had prediction=1 nearby (within tolerance).

### Implementation
```python
alignment_rewards = np.zeros((self.W,), dtype=np.float32)
tolerance = 2  # pixels

for j in range(self.W):
    if action[j] > 0.5:  # predicted "path"
        # Check if any previous row had path near column j
        aligned = False
        for prev_row in self.prev_preds_buffer:
            j_start = max(0, j - tolerance)
            j_end = min(self.W, j + tolerance + 1)
            if np.any(prev_row[j_start:j_end] > 0.5):
                aligned = True
                break
        
        alignment_rewards[j] = 0.5 if aligned else -1.0

pixel_rewards = base_rewards + self.continuity_coef * alignment_rewards
```

### When to use
- **Bridging occlusions**: Path disappears for a few rows, need to maintain continuity
- Want to **penalize isolated pixels** not connected to history
- Paths with **gaps** or **low contrast regions**

### Tuning
- `tolerance = 1`: Strict alignment (for thin paths)
- `tolerance = 2-3`: **Moderate (recommended for most cases)**
- `tolerance = 5`: Loose alignment (for wide/fuzzy paths)

---

## Strategy 4: Hybrid (Decay + Alignment)

### Concept
Combine exponential decay smoothness with alignment-based connectivity.

### Implementation
```python
# 1. Smoothness with decay
smooth_rewards = np.zeros((self.W,), dtype=np.float32)
decay_factor = 0.7
for i, prev_row in enumerate(reversed(list(self.prev_preds_buffer))):
    weight = (decay_factor ** i)
    smooth_rewards += -weight * np.abs(action - prev_row)

# 2. Connectivity/alignment
alignment_rewards = np.zeros((self.W,), dtype=np.float32)
tolerance = 2
for j in range(self.W):
    if action[j] > 0.5:
        aligned = any(
            np.any(prev_row[max(0, j-tolerance):min(self.W, j+tolerance+1)] > 0.5)
            for prev_row in self.prev_preds_buffer
        )
        alignment_rewards[j] = 0.5 if aligned else -1.0

# Combine both
continuity_rewards = smooth_rewards + 0.5 * alignment_rewards
pixel_rewards = base_rewards + self.continuity_coef * continuity_rewards
```

### When to use
- **Best of both worlds**: smoothness + connectivity
- Challenging images with **both occlusions AND noise**
- Want to bridge gaps while maintaining smooth trajectories

---

## Strategy 5: Weighted Voting

### Concept
Each previous row "votes" on where the path should be. Current action rewarded if it matches majority vote.

### Implementation
```python
# Build vote map: each prev row votes for columns where it predicted path
vote_map = np.zeros((self.W,), dtype=np.float32)
for prev_row in self.prev_preds_buffer:
    vote_map += (prev_row > 0.5).astype(np.float32)

# Normalize votes to [0,1]
vote_map /= len(self.prev_preds_buffer)

# Reward alignment with vote
voting_rewards = -np.abs(action - vote_map)
pixel_rewards = base_rewards + self.continuity_coef * voting_rewards
```

### When to use
- Path has **consistent width** across rows
- Want democratic consensus from history
- Robust to individual row errors

---

## Comparison Table

| Strategy | Best For | Smoothness | Occlusion Handling | Complexity |
|----------|----------|------------|-------------------|------------|
| **Exponential Decay** | Gradual curves | ⭐⭐⭐⭐ | ⭐⭐ | Low |
| **Centroid** | Straight lines | ⭐⭐⭐ | ⭐⭐⭐ | Low |
| **Alignment** | Gaps/occlusions | ⭐⭐ | ⭐⭐⭐⭐⭐ | Medium |
| **Hybrid** | Complex images | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Medium |
| **Voting** | Noisy paths | ⭐⭐⭐ | ⭐⭐⭐ | Low |

---

## Recommended Starting Points

### For your L-system tree images:
```python
# Use Hybrid (already implemented Decay, add Alignment):
continuity_coef = 0.1
decay_factor = 0.7
history_len = 3
tolerance = 2
```

### For very noisy images with occlusions:
```python
# Use Alignment (Strategy 3)
continuity_coef = 0.2
history_len = 5
tolerance = 3
```

### For smooth, gradual paths:
```python
# Use Exponential Decay (Strategy 1, current)
continuity_coef = 0.15
decay_factor = 0.8
history_len = 3
```

---

## How to Switch Strategies

Simply replace the continuity_rewards calculation in `env.py` line ~66-76 with your chosen strategy's code block.

### Quick Test
To compare strategies:
1. Train for 5 epochs with Strategy A
2. Save model + reconstruction
3. Train for 5 epochs with Strategy B
4. Compare reconstructions visually and via IoU/F1 metrics

---

## Debug Tips

Add this in your training loop to monitor continuity:
```python
if episode % 10 == 0 and not done:
    cont_mean = info['continuity_rewards'].mean()
    cont_std = info['continuity_rewards'].std()
    print(f"Row {info['row_index']}: Continuity μ={cont_mean:.3f}, σ={cont_std:.3f}")
```

**Good signs**:
- Continuity rewards stabilize over epochs (mean approaches 0)
- Low variance (model is consistent)

**Bad signs**:
- Very negative continuity rewards → actions don't match history (increase `continuity_coef`)
- High variance → model is unstable (reduce learning rate or increase history_len)
