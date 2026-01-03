# 3D U-Net for Volumetric Segmentation

Classic 3D U-Net implementation for segmenting branching structures (e.g., root systems) in CT-like volumetric data.

## Features

- **3D U-Net Architecture**: Full encoder-decoder with skip connections
- **Class Imbalance Handling**: Multiple loss functions optimized for sparse foreground
  - Dice Loss
  - Focal Loss
  - Tversky Loss
  - Combined Loss (Dice + Focal)
- **Data Augmentation**: Random flips, rotations, intensity shifts, Gaussian noise
- **Foreground Filtering**: Option to train only on samples with sufficient foreground

## Usage

### Training

```bash
# Basic training with combined loss (recommended for class imbalance)
python unet3d_baseline.py \
    --data_dir /path/to/subvolumes \
    --save_dir models/unet3d \
    --epochs 100 \
    --batch_size 2 \
    --lr 1e-3 \
    --loss combined \
    --augment

# Train only on samples with foreground (faster, focused learning)
python unet3d_baseline.py \
    --data_dir /path/to/subvolumes \
    --save_dir models/unet3d \
    --min_fg_ratio 0.0001 \
    --augment

# Use Tversky loss (penalizes false negatives more)
python unet3d_baseline.py \
    --loss tversky \
    --augment
```

### Evaluation

```bash
# Evaluate on validation set
python evaluate_unet3d.py \
    --checkpoint models/unet3d/best_model.pth \
    --data_dir /path/to/subvolumes \
    --split val

# Generate visualizations
python evaluate_unet3d.py \
    --checkpoint models/unet3d/best_model.pth \
    --visualize \
    --output_dir results/unet3d
```

## Model Architecture

```
Input: (B, 1, 100, 64, 64)
         ↓
    [Conv Block] → 32 channels
         ↓ (MaxPool)
    [Conv Block] → 64 channels
         ↓ (MaxPool)
    [Conv Block] → 128 channels
         ↓ (MaxPool)
    [Conv Block] → 256 channels
         ↓ (MaxPool)
    [Bottleneck] → 256 channels
         ↓ (Upsample + Skip)
    [Conv Block] → 128 channels
         ↓ (Upsample + Skip)
    [Conv Block] → 64 channels
         ↓ (Upsample + Skip)
    [Conv Block] → 32 channels
         ↓ (Upsample + Skip)
    [1x1 Conv + Sigmoid] → 1 channel
         ↓
Output: (B, 1, 100, 64, 64)
```

Total parameters: ~12.9M

## Loss Functions

### Combined Loss (Default)
Best for high class imbalance:
```
L = 0.5 * DiceLoss + 0.5 * FocalLoss
```

### Dice Loss
Optimizes overlap directly:
```
Dice = 2 * |P ∩ T| / (|P| + |T|)
L = 1 - Dice
```

### Tversky Loss
Generalized Dice with control over FP/FN penalty:
```
Tversky = TP / (TP + α*FP + β*FN)
L = 1 - Tversky
```
With α=0.3, β=0.7 to penalize missed roots more than false positives.

### Focal Loss
Down-weights easy examples:
```
FL = -α * (1-p)^γ * log(p)
```
With α=0.75 for positive class focus, γ=2.0 for hard example mining.

## Data Format

Expected directory structure:
```
subvolumes/
├── train/
│   ├── volumes/
│   │   ├── week2_d00_h00_w00.npy
│   │   └── ...
│   └── masks/
│       ├── week2_d00_h00_w00.npy
│       └── ...
└── val/
    ├── volumes/
    └── masks/
```

- Volumes: uint8 numpy arrays, shape (D, H, W) = (100, 64, 64)
- Masks: Binary uint8 numpy arrays, same shape

## Training Tips

1. **Use foreground filtering** (`--min_fg_ratio 0.0001`): Most subvolumes are pure background, filtering speeds up training and focuses learning on relevant samples.

2. **Enable augmentation** (`--augment`): Essential for small datasets to prevent overfitting.

3. **Start with combined loss**: It balances global overlap (Dice) with per-pixel classification (Focal).

4. **Adjust batch size**: With (100, 64, 64) volumes, batch size 2-4 typically fits in 8GB GPU memory.

5. **Monitor recall**: With sparse foreground, high precision but low recall indicates the model is being too conservative.

## Outputs

- `models/unet3d/best_model.pth`: Best checkpoint by validation Dice
- `models/unet3d/latest_model.pth`: Most recent checkpoint
- `models/unet3d/history.json`: Training/validation metrics per epoch
