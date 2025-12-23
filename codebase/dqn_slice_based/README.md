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

**Key improvements:**

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
