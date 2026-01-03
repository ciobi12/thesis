# Improving Global Coherence in 3D Root Segmentation

## Problem Analysis

Based on your results, we have identified three key issues:

### 1. Severe Overfitting
- **Train DICE**: ~0.97 (excellent)
- **Val DICE**: ~0.45 (poor)
- **Gap**: 52 percentage points

This indicates the model is memorizing training subvolumes rather than learning generalizable features.

### 2. Fast Epsilon Decay
- Current: Reaches 0.01 by epoch ~5
- Desired: Reach 0.01 by epoch 35-40

The agent stops exploring too early, settling into suboptimal policies.

### 3. Poor Global Coherence
The model segments each subvolume independently with no understanding of:
- How subvolumes connect at boundaries
- Overall structure connectivity
- Cross-subvolume path continuation

This is especially problematic for **transversal roots** that cross subvolume boundaries.

---

## Implemented Fixes

### Fix 1: Slower Epsilon Decay
```python
# OLD (too fast):
decay_rate = 3.0 / epsilon_decay_steps

# NEW (reaches 0.01 around epoch 37):
decay_rate = 0.5 / epsilon_decay_steps
```

### Fix 2: Increased Data Augmentation
```python
# Increased probabilities for regularization
AUGMENT_PROB = 0.5   # was 0.1
INTENSITY_PROB = 0.4  # was 0.3
NOISE_PROB = 0.3      # was 0.2
```

### Fix 3: Dropout Regularization
Added `Dropout2d(0.2)` after each ReLU in the encoder layers of `PerPixelCNNWithHistory`.

### Fix 4: New Environment with Global Coherence (`env_global_coherence.py`)
Added two new reward components computed at episode end:

1. **Connectivity Reward**: Penalizes fragmentation
   - Counts connected components in 3D
   - Rewards having one large connected structure
   - Formula: `connectivity_coef * (1 - fragmentation_penalty)`

2. **Boundary Consistency Reward**: Encourages smooth transitions
   - Compares predictions at subvolume edges with neighbors
   - Formula: `boundary_coef * consistency_score`

---

## Architectural Recommendations

### Short-term (Easy to Implement)

#### A. Multi-Axis Processing
Process each subvolume along multiple axes and ensemble predictions:
```python
def multi_axis_predict(vol, mask, policy_net):
    pred_d = predict_along_axis(vol, policy_net, axis=0)  # D slices
    pred_h = predict_along_axis(vol, policy_net, axis=1)  # H slices  
    pred_w = predict_along_axis(vol, policy_net, axis=2)  # W slices
    
    # Ensemble by voting
    return ((pred_d + pred_h + pred_w) >= 2).astype(np.uint8)
```

This helps with transversal roots that appear vertical in one axis but horizontal in others.

#### B. Two-Pass Training per Epoch
1. **Pass 1**: Process all subvolumes, collect boundary predictions
2. **Pass 2**: Re-process with neighbor context from Pass 1

```python
# Pass 1: Get initial predictions
boundary_cache = {}
for pos, vol in subvolumes:
    pred = predict_subvolume(vol, policy_net)
    boundary_cache[pos] = extract_boundaries(pred)

# Pass 2: Re-train with neighbor context
for pos, vol, mask in subvolumes:
    neighbor_context = get_neighbor_boundaries(pos, boundary_cache)
    env = PathReconstructionEnvGlobalCoherence(
        vol, mask, 
        neighbor_context=neighbor_context
    )
    train_on_episode(env, policy_net)
```

#### C. Early Stopping with Patience
Stop training when validation DICE doesn't improve for N epochs:
```python
patience = 10
best_val_dice = 0
epochs_without_improvement = 0

for epoch in range(num_epochs):
    val_dice = evaluate()
    if val_dice > best_val_dice:
        best_val_dice = val_dice
        epochs_without_improvement = 0
        save_best_model()
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print("Early stopping!")
            break
```

### Medium-term (Moderate Effort)

#### D. Post-Processing with CRF
Apply Conditional Random Field after DQN predictions to enforce global consistency:
```python
import pydensecrf.densecrf as dcrf

def apply_crf(volume, prediction, theta_alpha=80, theta_beta=13):
    """Refine predictions using CRF."""
    d = dcrf.DenseCRF(volume.size, 2)
    # Add unary potentials from predictions
    # Add pairwise potentials for smoothness
    refined = d.inference(5)
    return refined
```

#### E. Graph-Based Connectivity Refinement
After segmentation, build a graph and remove disconnected small components:
```python
from scipy import ndimage

def refine_connectivity(pred, min_component_size=100):
    labeled, n_components = ndimage.label(pred)
    sizes = ndimage.sum(pred, labeled, range(1, n_components + 1))
    
    # Keep only components larger than threshold
    mask = np.zeros_like(pred)
    for i, size in enumerate(sizes, 1):
        if size >= min_component_size:
            mask[labeled == i] = 1
    return mask
```

#### F. Curriculum Learning
Train on easier subvolumes first, then harder ones:
```python
def get_difficulty_score(vol, mask):
    """Higher = harder (more occlusion, more transversal roots)"""
    fg_ratio = mask.mean()
    edge_ratio = sobel_edge_strength(mask)
    return edge_ratio / (fg_ratio + 1e-6)

# Sort by difficulty and train progressively
train_order = sorted(subvolumes, key=get_difficulty_score)
```

### Long-term (Significant Effort)

#### G. Hierarchical Multi-Agent Architecture
Instead of treating each subvolume independently:

```
Level 3: Global Agent (full volume overview)
            ↓ provides context
Level 2: Regional Agents (2x2x2 groups of subvolumes)
            ↓ provides context
Level 1: Local Agents (individual subvolumes)
            ↑ reports predictions
```

The Global Agent sees downsampled full volume and guides Regional Agents, which in turn guide Local Agents.

#### H. 3D U-Net Encoder for Context
Replace 2D slice processing with 3D patches that see local 3D context:
```python
class ContextEncoder3D(nn.Module):
    def __init__(self):
        self.conv3d = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=1)
        # ... 3D encoder-decoder architecture
        
    def forward(self, subvolume):
        # Process full 3D subvolume
        features = self.conv3d(subvolume)
        # Generate per-voxel Q-values
        return q_values  # (D, H, W, 2)
```

This natively handles transversal structures.

#### I. Cross-Subvolume Message Passing (GNN)
Model subvolumes as nodes in a graph where edges connect neighbors:
```python
import torch_geometric

# Build graph where nodes = subvolumes, edges = neighbors
edge_index = build_neighbor_graph(positions)

# Message passing: share boundary features
for layer in range(n_layers):
    node_features = gnn_layer(node_features, edge_index)
    
# Each node now has context from neighbors
```

---

## Recommended Action Plan

### Phase 1 (Immediate - 1-2 days)
1. ✅ Fix epsilon decay (done)
2. ✅ Increase augmentation (done)
3. ✅ Add dropout (done)
4. Run training with new settings
5. Add early stopping

### Phase 2 (Short-term - 1 week)
1. Implement multi-axis processing
2. Implement two-pass training with neighbor context
3. Add post-processing (connected component filtering)

### Phase 3 (Medium-term - 2-3 weeks)
1. Implement 3D context encoder (replace 2D slice processing)
2. Add CRF post-processing
3. Experiment with curriculum learning

### Phase 4 (Long-term - 1+ month)
1. Hierarchical multi-agent architecture
2. Cross-subvolume GNN message passing

---

## Hyperparameter Recommendations

Based on your results, try these settings:

```python
# Training
num_epochs = 100           # More epochs with early stopping
global_eval_interval = 3   # More frequent evaluation
lr = 5e-4                  # Lower learning rate
batch_size = 32            # Smaller batches for regularization

# Environment
continuity_coef = 0.2      # Stronger continuity
continuity_decay_factor = 0.7
dice_coef = 0.5            # Reduce DICE influence
boundary_coef = 0.5        # New: boundary consistency
connectivity_coef = 0.3    # New: connectivity reward

# Network
dropout_rate = 0.3         # More dropout

# Exploration
epsilon_decay_steps = 15000  # Even slower decay
epsilon_start = 0.2         # Start with less exploration (use pretrained knowledge)
```

---

## Key Insight: Why Transversal Roots Fail

The current slice-by-slice processing has an inherent bias:
- Roots parallel to D-axis appear in many consecutive slices → strong continuity signal
- Roots perpendicular to D-axis appear briefly → weak continuity signal

**Solutions**:
1. Multi-axis processing (ensemble D, H, W predictions)
2. 3D convolutions that see local 3D neighborhoods
3. Explicit orientation-invariant features

The multi-axis approach is the easiest to implement and should significantly improve transversal root detection.
