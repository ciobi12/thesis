"""
Classic 3D U-Net for volumetric segmentation of branching structures.

Designed for the rapids-p subvolumes dataset with high class imbalance.
Uses Dice loss + Focal loss to handle the imbalanced foreground/background ratio.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import os
import argparse
from tqdm import tqdm
import json
import csv
from datetime import datetime
import matplotlib.pyplot as plt


# =============================================================================
# Dataset
# =============================================================================

class SubvolumeDataset(Dataset):
    """Dataset for 3D subvolumes from the rapids-p dataset."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        augment: bool = False,
        min_foreground_ratio: float = 0.0,  # 0.0 = include all
        normalize: bool = True
    ):
        """
        Args:
            data_dir: Path to the subvolumes directory (contains train/val subdirs)
            split: 'train' or 'val'
            augment: Whether to apply data augmentation
            min_foreground_ratio: Minimum foreground ratio to include a sample
            normalize: Whether to normalize input to [0, 1]
        """
        self.data_dir = Path(data_dir) / split
        self.volumes_dir = self.data_dir / "volumes"
        self.masks_dir = self.data_dir / "masks"
        self.augment = augment and (split == "train")
        self.normalize = normalize
        
        # Get all volume files
        all_files = sorted(list(self.volumes_dir.glob("*.npy")))
        
        # Filter by foreground ratio if needed
        if min_foreground_ratio > 0:
            self.files = []
            for f in all_files:
                mask_path = self.masks_dir / f.name
                mask = np.load(mask_path)
                if mask.sum() / mask.size >= min_foreground_ratio:
                    self.files.append(f)
            print(f"[{split}] Filtered to {len(self.files)}/{len(all_files)} samples with fg_ratio >= {min_foreground_ratio}")
        else:
            self.files = all_files
            print(f"[{split}] Loaded {len(self.files)} samples")
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        vol_path = self.files[idx]
        mask_path = self.masks_dir / vol_path.name
        
        # Load data
        volume = np.load(vol_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.float32)
        
        # Normalize volume to [0, 1]
        if self.normalize:
            volume = volume / 255.0
        
        # Data augmentation
        if self.augment:
            volume, mask = self._augment(volume, mask)
        
        # Add channel dimension: (D, H, W) -> (1, D, H, W)
        volume = volume[np.newaxis, ...]
        mask = mask[np.newaxis, ...]
        
        return torch.from_numpy(volume), torch.from_numpy(mask)
    
    def _augment(
        self,
        volume: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random augmentations."""
        # Random flips along each axis
        if np.random.rand() > 0.5:
            volume = np.flip(volume, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        if np.random.rand() > 0.5:
            volume = np.flip(volume, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        if np.random.rand() > 0.5:
            volume = np.flip(volume, axis=2).copy()
            mask = np.flip(mask, axis=2).copy()
        
        # Random 90-degree rotations in XY plane
        k = np.random.randint(0, 4)
        if k > 0:
            volume = np.rot90(volume, k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k, axes=(1, 2)).copy()
        
        # Random intensity shift and scale
        if np.random.rand() > 0.5:
            shift = np.random.uniform(-0.1, 0.1)
            scale = np.random.uniform(0.9, 1.1)
            volume = np.clip(volume * scale + shift, 0, 1)
        
        # Random Gaussian noise
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.02, volume.shape)
            volume = np.clip(volume + noise, 0, 1).astype(np.float32)
        
        return volume, mask


# =============================================================================
# 3D U-Net Architecture
# =============================================================================

class ConvBlock3D(nn.Module):
    """Double 3D convolution block with batch normalization."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None
    ):
        super().__init__()
        mid_channels = mid_channels or out_channels
        
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownBlock3D(nn.Module):
    """Downsampling block: maxpool followed by double conv."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            ConvBlock3D(in_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class UpBlock3D(nn.Module):
    """Upsampling block: upsample, concat with skip connection, double conv."""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = ConvBlock3D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ConvBlock3D(in_channels, out_channels)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        
        # Handle size mismatch due to odd dimensions
        diff_d = skip.size(2) - x.size(2)
        diff_h = skip.size(3) - x.size(3)
        diff_w = skip.size(4) - x.size(4)
        
        x = F.pad(x, [
            diff_w // 2, diff_w - diff_w // 2,
            diff_h // 2, diff_h - diff_h // 2,
            diff_d // 2, diff_d - diff_d // 2
        ])
        
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """
    Classic 3D U-Net architecture for volumetric segmentation.
    
    Input: (B, 1, D, H, W)
    Output: (B, 1, D, H, W) with sigmoid activation
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_features: int = 32,
        bilinear: bool = True
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (1 for binary segmentation)
            base_features: Number of features in first layer (doubles each level)
            bilinear: Use bilinear upsampling (True) or transposed conv (False)
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        # Feature channel sizes
        f = base_features  # 32
        factor = 2 if bilinear else 1
        
        # Encoder (downsampling path)
        self.inc = ConvBlock3D(in_channels, f)           # 1 -> 32
        self.down1 = DownBlock3D(f, f * 2)               # 32 -> 64
        self.down2 = DownBlock3D(f * 2, f * 4)           # 64 -> 128
        self.down3 = DownBlock3D(f * 4, f * 8)           # 128 -> 256
        self.down4 = DownBlock3D(f * 8, f * 16 // factor)  # 256 -> 256 (bilinear) or 512
        
        # Decoder (upsampling path)
        self.up1 = UpBlock3D(f * 16, f * 8 // factor, bilinear)   # 512 -> 128
        self.up2 = UpBlock3D(f * 8, f * 4 // factor, bilinear)    # 256 -> 64
        self.up3 = UpBlock3D(f * 4, f * 2 // factor, bilinear)    # 128 -> 32
        self.up4 = UpBlock3D(f * 2, f, bilinear)                  # 64 -> 32
        
        # Output layer
        self.outc = nn.Conv3d(f, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)      # (B, 32, D, H, W)
        x2 = self.down1(x1)   # (B, 64, D/2, H/2, W/2)
        x3 = self.down2(x2)   # (B, 128, D/4, H/4, W/4)
        x4 = self.down3(x3)   # (B, 256, D/8, H/8, W/8)
        x5 = self.down4(x4)   # (B, 256, D/16, H/16, W/16)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)  # (B, 128, D/8, H/8, W/8)
        x = self.up2(x, x3)   # (B, 64, D/4, H/4, W/4)
        x = self.up3(x, x2)   # (B, 32, D/2, H/2, W/2)
        x = self.up4(x, x1)   # (B, 32, D, H, W)
        
        # Output
        logits = self.outc(x)
        return torch.sigmoid(logits)


# =============================================================================
# Loss Functions for Class Imbalance
# =============================================================================

class DiceLoss(nn.Module):
    """Dice loss for handling class imbalance."""
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted probabilities (B, 1, D, H, W)
            target: Ground truth binary mask (B, 1, D, H, W)
        """
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha: Weighting factor for the positive class
            gamma: Focusing parameter (higher = more focus on hard examples)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted probabilities (B, 1, D, H, W)
            target: Ground truth binary mask (B, 1, D, H, W)
        """
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Clamp for numerical stability
        pred_flat = torch.clamp(pred_flat, min=1e-7, max=1 - 1e-7)
        
        # BCE loss per pixel
        bce = -target_flat * torch.log(pred_flat) - (1 - target_flat) * torch.log(1 - pred_flat)
        
        # Focal term
        p_t = pred_flat * target_flat + (1 - pred_flat) * (1 - target_flat)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        
        focal_loss = focal_weight * bce
        return focal_loss.mean()


class TverskyLoss(nn.Module):
    """
    Tversky loss - generalization of Dice loss.
    
    With alpha=beta=0.5, equivalent to Dice loss.
    Higher alpha penalizes FP more, higher beta penalizes FN more.
    """
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6):
        """
        Args:
            alpha: Weight for false positives
            beta: Weight for false negatives (higher = penalize missed detections more)
            smooth: Smoothing factor for numerical stability
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky


class CombinedLoss(nn.Module):
    """
    Combined loss: Dice + Focal for robust training with class imbalance.
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        focal_alpha: float = 0.75,  # High alpha for positive class
        focal_gamma: float = 2.0
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        return self.dice_weight * dice + self.focal_weight * focal


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5
) -> dict:
    """
    Compute segmentation metrics.
    
    Returns:
        dict with 'dice', 'iou', 'precision', 'recall', 'accuracy'
    """
    pred_binary = (pred > threshold).float()
    
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    
    tp = (pred_flat * target_flat).sum().item()
    fp = (pred_flat * (1 - target_flat)).sum().item()
    fn = ((1 - pred_flat) * target_flat).sum().item()
    tn = ((1 - pred_flat) * (1 - target_flat)).sum().item()
    
    eps = 1e-7
    
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy
    }


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    metrics_sum = {'dice': 0, 'iou': 0, 'precision': 0, 'recall': 0, 'accuracy': 0}
    
    pbar = tqdm(dataloader, desc="Training")
    for volumes, masks in pbar:
        volumes = volumes.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(volumes)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Compute metrics
        with torch.no_grad():
            batch_metrics = compute_metrics(outputs, masks)
            for k in metrics_sum:
                metrics_sum[k] += batch_metrics[k]
        
        pbar.set_postfix({'loss': loss.item(), 'dice': batch_metrics['dice']})
    
    if scheduler is not None:
        scheduler.step()
    
    n = len(dataloader)
    return {
        'loss': total_loss / n,
        **{k: v / n for k, v in metrics_sum.items()}
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> dict:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    metrics_sum = {'dice': 0, 'iou': 0, 'precision': 0, 'recall': 0, 'accuracy': 0}
    
    for volumes, masks in tqdm(dataloader, desc="Validation"):
        volumes = volumes.to(device)
        masks = masks.to(device)
        
        outputs = model(volumes)
        loss = criterion(outputs, masks)
        
        total_loss += loss.item()
        
        batch_metrics = compute_metrics(outputs, masks)
        for k in metrics_sum:
            metrics_sum[k] += batch_metrics[k]
    
    n = len(dataloader)
    return {
        'loss': total_loss / n,
        **{k: v / n for k, v in metrics_sum.items()}
    }


def plot_training_curves(history: dict, save_dir: Path):
    """Plot and save training curves for loss, Dice, and IoU."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Dice
    axes[1].plot(epochs, history['train_dice'], 'b-', label='Train')
    axes[1].plot(epochs, history['val_dice'], 'r-', label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice')
    axes[1].set_title('Dice Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # IoU
    axes[2].plot(epochs, history['train_iou'], 'b-', label='Train')
    axes[2].plot(epochs, history['val_iou'], 'r-', label='Val')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('IoU')
    axes[2].set_title('IoU Score')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_dir / 'training_curves.png'}")


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int = 100,
    save_dir: str = "models/unet3d",
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    early_stopping_patience: int = 15
) -> dict:
    """
    Full training loop with validation and checkpointing.
    
    Returns:
        Training history dict
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_dice': [], 'val_dice': [],
        'train_iou': [], 'val_iou': []
    }
    
    best_val_dice = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}, "
              f"IoU: {train_metrics['iou']:.4f}, Recall: {train_metrics['recall']:.4f}")
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, "
              f"IoU: {val_metrics['iou']:.4f}, Recall: {val_metrics['recall']:.4f}")
        
        # Log history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_dice'].append(train_metrics['dice'])
        history['val_dice'].append(val_metrics['dice'])
        history['train_iou'].append(train_metrics['iou'])
        history['val_iou'].append(val_metrics['iou'])
        
        # Save best model
        if val_metrics['dice'] > best_val_dice:
            best_val_dice = val_metrics['dice']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_metrics['dice'],
                'val_iou': val_metrics['iou']
            }, save_dir / "best_model.pth")
            print(f"[*] New best model saved with val_dice={best_val_dice:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Save latest model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history
        }, save_dir / "latest_model.pth")
    
    # Save training history as JSON
    with open(save_dir / "history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save training history as CSV for easy viewing
    csv_path = save_dir / "training_metrics.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_dice', 'val_dice', 'train_iou', 'val_iou'])
        for i in range(len(history['train_loss'])):
            writer.writerow([
                i + 1,
                f"{history['train_loss'][i]:.6f}",
                f"{history['val_loss'][i]:.6f}",
                f"{history['train_dice'][i]:.6f}",
                f"{history['val_dice'][i]:.6f}",
                f"{history['train_iou'][i]:.6f}",
                f"{history['val_iou'][i]:.6f}"
            ])
    print(f"Training metrics saved to {csv_path}")
    
    # Plot and save training curves
    plot_training_curves(history, save_dir)
    
    return history


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train 3D U-Net for volumetric segmentation")
    parser.add_argument("--data_dir", type=str, 
                        default="/home/sysadmin/thesis/data/rapids-p/subvolumes",
                        help="Path to subvolumes directory")
    parser.add_argument("--save_dir", type=str,
                        default="models/unet3d",
                        help="Directory to save models")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--base_features", type=int, default=32, help="Base feature channels")
    parser.add_argument("--loss", type=str, default="combined", 
                        choices=["dice", "focal", "tversky", "combined"],
                        help="Loss function to use")
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    parser.add_argument("--min_fg_ratio", type=float, default=0.0,
                        help="Minimum foreground ratio to include samples")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--early_stopping", type=int, default=15, help="Early stopping patience")
    
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data
    print("\n[1] Loading datasets...")
    train_dataset = SubvolumeDataset(
        args.data_dir, split="train", 
        augment=args.augment, 
        min_foreground_ratio=args.min_fg_ratio
    )
    val_dataset = SubvolumeDataset(
        args.data_dir, split="val",
        augment=False,
        min_foreground_ratio=args.min_fg_ratio
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Model
    print("\n[2] Creating model...")
    model = UNet3D(
        in_channels=1,
        out_channels=1,
        base_features=args.base_features,
        bilinear=True
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function
    print(f"\n[3] Setting up {args.loss} loss...")
    if args.loss == "dice":
        criterion = DiceLoss()
    elif args.loss == "focal":
        criterion = FocalLoss(alpha=0.75, gamma=2.0)
    elif args.loss == "tversky":
        criterion = TverskyLoss(alpha=0.3, beta=0.7)  # Penalize FN more
    else:  # combined
        criterion = CombinedLoss(dice_weight=0.5, focal_weight=0.5)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    # Train
    print(f"\n[4] Starting training...")
    print(f"    Epochs: {args.epochs}")
    print(f"    Batch size: {args.batch_size}")
    print(f"    Learning rate: {args.lr}")
    print(f"    Augmentation: {args.augment}")
    
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        save_dir=args.save_dir,
        scheduler=scheduler,
        early_stopping_patience=args.early_stopping
    )
    
    print("\n[5] Training complete!")
    print(f"Best validation Dice: {max(history['val_dice']):.4f}")
    
    # Print final metrics summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"{'Metric':<20} {'Train (final)':<15} {'Val (final)':<15} {'Val (best)':<15}")
    print("-"*60)
    print(f"{'Loss':<20} {history['train_loss'][-1]:<15.4f} {history['val_loss'][-1]:<15.4f} {min(history['val_loss']):<15.4f}")
    print(f"{'Dice':<20} {history['train_dice'][-1]:<15.4f} {history['val_dice'][-1]:<15.4f} {max(history['val_dice']):<15.4f}")
    print(f"{'IoU':<20} {history['train_iou'][-1]:<15.4f} {history['val_iou'][-1]:<15.4f} {max(history['val_iou']):<15.4f}")
    print("="*60)
    
    # Load best model for visualization
    best_checkpoint = torch.load(Path(args.save_dir) / "best_model.pth")
    model.load_state_dict(best_checkpoint['model_state_dict'])

if __name__ == "__main__":
    main()
