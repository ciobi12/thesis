# DQN Slice-Based Segmentation

This folder contains a slice-wise DQN approach for 3D volume segmentation.

## Overview

Unlike the row-based approach that processes 2D images row by row, this approach processes 3D volumes slice by slice:

- **Input**: 3D volumes of shape `(D, H, W)` where D is depth, H is height, W is width
- **Processing**: The agent traverses the volume from bottom to top (or top to bottom), making decisions for each entire slice at a time.

## Versions

### V1 (Original) - For Synthetic Data
- `env.py`: Basic environment for slice-wise volume traversal
- `dqn.py`: Simple CNN architectures
- `main.py`: Basic training loop

### V2 (Enhanced) - For Real Large Volumes
Designed for training on limited real CT data (e.g., 1 training volume + 1 validation volume).

### V3 (Continuity-Focused) ⭐ **RECOMMENDED FOR EXTREME IMBALANCE**

Designed specifically for CT scans with **extreme class imbalance** (0.02-0.1% foreground).

**Key innovations:**

1. **Root-Focused Rewards**: Only compute rewards in regions containing roots (ROI), ignoring the vast background.

2. **Overlap-Based Continuity**: Uses IoU between adjacent slice predictions to ensure 3D structure coherence:
   ```
   continuity = weighted_avg(IoU(pred_t, pred_{t-i}) for i in 1..history_len)
   ```

3. **Cross-Slice Connectivity**: Penalizes predictions that are disconnected from the previous slice:
   ```
   if (pred_t ∩ dilate(pred_{t-1})) > 0: connected = 1.0
   else: connected = 0.0  # PENALTY
   ```

4. **Patch-Based Training**: For extreme imbalance, extracts patches (e.g., 128x128) centered on root regions instead of processing full 512x512 slices.

5. **Recall-Focused**: Extra reward for finding roots (reduces false negatives).

**Usage (V3):**

```bash
# Patch-based training (recommended for <0.5% foreground)
python -m dqn_slice_based.main_v3 \
    --train_dir data/rapids-p/train \
    --val_dir data/rapids-p/val \
    --use_patches \
    --patch_size 128 \
    --epochs 100 \
    --episodes_per_epoch 20 \
    --continuity_coef 0.5 \
    --connectivity_coef 0.3 \
    --dice_coef 1.0 \
    --recall_coef 0.5

# Full-slice training (for moderate imbalance)
python -m dqn_slice_based.main_v3 \
    --train_dir data/rapids-p/train \
    --val_dir data/rapids-p/val \
    --roi_dilation 20 \
    --epochs 50
```

**Key Parameters (V3):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use_patches` | False | Enable patch-based training |
| `--patch_size` | 128 | Patch size (64-256) |
| `--continuity_coef` | 0.5 | Overlap continuity weight |
| `--connectivity_coef` | 0.3 | Cross-slice connectivity weight |
| `--dice_coef` | 1.0 | Dice accuracy weight |
| `--recall_coef` | 0.5 | Recall bonus weight |
| `--roi_dilation` | 10 | ROI dilation radius |

---

### V4 (Intensity-Aware Multi-Directional) ⭐ **LATEST**

Designed for **roots growing in ANY direction** (vertical + transversal) with **characteristic intensity signatures**.

**Key innovations:**

1. **Intensity-Based Reward**: Root voxels have a characteristic narrow intensity band. The model learns this profile from ground truth and rewards predictions matching it:
   ```python
   # Root intensity: mean=55/255, std=6/255 (narrow band)
   # Background: mean=57/255, std=33/255 (broad distribution)
   likelihood = exp(-0.5 * ((intensity - root_mean) / root_std)^2)
   intensity_reward = avg(likelihood at predicted locations)
   ```

2. **Multi-Directional Continuity**:
   - **Vertical (Z-axis)**: Overlap between adjacent slice predictions with dilation for lateral movement
   - **Transversal (XY-plane)**: Penalizes fragmented predictions within each slice
   - **Combined**: Handles roots growing in any direction

3. **Intensity Hint Input**: Additional input channel showing per-pixel likelihood of being root based on intensity. Helps guide predictions.

4. **Same Plant, Different Weeks**: Train on week2 scan, validate on week3 scan of the same pot. The root structure differs but intensity profile is similar.

**Usage (V4):**

```bash
# Recommended settings for roots growing in all directions
python -m dqn_slice_based.main_v4 \
    --train_dir data/rapids-p/train \
    --val_dir data/rapids-p/val \
    --use_patches \
    --patch_size 128 \
    --epochs 100 \
    --episodes_per_epoch 20 \
    --dice_coef 1.0 \
    --intensity_coef 0.5 \
    --vertical_cont_coef 0.4 \
    --transversal_cont_coef 0.3 \
    --recall_coef 0.5 \
    --intensity_margin 1.5
```

**Key Parameters (V4):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--intensity_coef` | 0.5 | Weight for intensity-based reward |
| `--vertical_cont_coef` | 0.4 | Vertical (Z) continuity weight |
| `--transversal_cont_coef` | 0.3 | Transversal (XY) continuity weight |
| `--intensity_margin` | 1.5 | Std devs for root intensity range |
| `--use_patches` | False | Enable patch-based training |
| `--patch_size` | 128 | Patch size for training |
| `--dice_coef` | 1.0 | Dice accuracy weight |
| `--recall_coef` | 0.5 | Recall bonus weight |

**Reward Formula (V4):**
```
reward = dice_coef * Dice(pred, gt) +
         recall_coef * Recall(pred, gt) +
         intensity_coef * IntensityLikelihood(pred, volume) +
         vertical_cont_coef * VerticalContinuity(pred, prev_pred) +
         transversal_cont_coef * TransversalContinuity(pred)
```

---

**V2 improvements (inherited):**

1. **Data Augmentation**
   - Random slice sampling (not always bottom-to-top)
   - Intensity augmentation
   - Volume flips along different axes
   - Multiple episodes per volume per epoch

2. **Memory-Efficient Training**
   - Mixed precision (FP16) training
   - Gradient accumulation for larger effective batch sizes
   - Slice sampling (train on 64 slices per episode, not all 512)

3. **Improved Reward System**
   - Class-weighted rewards (higher weight for foreground/vessel pixels)
   - Boundary-aware rewards (extra penalty for missing boundaries)
   - Continuity rewards with exponential decay

4. **Enhanced Network Architectures**
   - `PerPixelDQNV2`: Multi-scale encoder with residual blocks and attention
   - `UNetDQN`: U-Net style architecture for better multi-scale context
   - Separate encoders for current slice and history

5. **Better Training Strategy**
   - Prioritized Experience Replay
   - Warmup epochs before epsilon decay
   - Cosine annealing learning rate schedule
   - Longer exploration phase

## Files

### V1 (Basic)
- `env.py`: Basic environment for slice-wise volume traversal
- `dqn.py`: Simple CNN architectures
- `main.py`: Basic training loop

### V2 (For Real Data)
- `env_v2.py`: Enhanced environment with:
  - `PathReconstructionEnvV2`: Full volume processing with augmentation
  - `SliceSamplerEnv`: Random contiguous slice sampling for diversity
- `dqn_v2.py`: Enhanced architectures:
  - `PerPixelDQNV2`: Multi-scale CNN with attention
  - `UNetDQN`: U-Net style architecture
  - `PrioritizedReplayBuffer`: Better sample efficiency
- `main_v2.py`: Training script for real volumes

## Usage

### For Real Data (V2)

```bash
python -m dqn_slice_based.main_v2 \
    --train_dir data/ct_like/rapids-p/train \
    --val_dir data/ct_like/rapids-p/val \
    --epochs 50 \
    --episodes_per_epoch 10 \
    --slices_per_episode 64 \
    --batch_size 16 \
    --model_type perpixel \
    --foreground_weight 5.0
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--episodes_per_epoch` | 10 | Episodes per epoch (more = more diversity) |
| `--slices_per_episode` | 64 | Slices sampled per episode |
| `--batch_size` | 16 | Mini-batch size |
| `--model_type` | perpixel | 'perpixel' or 'unet' |
| `--foreground_weight` | 5.0 | Class weight for foreground pixels |
| `--base_channels` | 32 | Base channel count for network |

### Handling Different Volume Sizes

The V2 architecture handles arbitrary volume sizes:
- Training: 512×512×512
- Validation: 800×466×471

The network uses fully convolutional layers so it can process any spatial dimension.

## Training Tips for Limited Data

1. **Increase `episodes_per_epoch`**: More episodes = more random slice combinations from the same volume

2. **Use data augmentation**: The `augment_volume()` function creates multiple variants:
   - Original
   - Horizontal flip
   - Vertical flip  
   - Intensity scaled (0.9x, 1.1x)

3. **Higher `foreground_weight`**: If vessels are sparse, increase to 10-20

4. **Smaller `slices_per_episode`**: More episodes with fewer slices increases diversity

5. **Consider U-Net architecture**: For 512×512 slices, U-Net may capture better multi-scale features

6. **Longer training**: With limited data, more epochs help generalization
