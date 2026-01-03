"""
Inference and evaluation script for 3D U-Net segmentation.

Loads a trained model and evaluates on validation/test data.
Can also run inference on full volumes (stitching subvolumes together).
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import nibabel as nib

from unet3d_baseline import (
    UNet3D, 
    SubvolumeDataset, 
    compute_metrics,
    DiceLoss
)


def load_model(
    checkpoint_path: str,
    device: torch.device,
    base_features: int = 32
) -> UNet3D:
    """Load a trained model from checkpoint."""
    model = UNet3D(
        in_channels=1,
        out_channels=1,
        base_features=base_features
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    if 'val_dice' in checkpoint:
        print(f"  Validation Dice: {checkpoint['val_dice']:.4f}")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    
    return model


@torch.no_grad()
def evaluate_dataset(
    model: UNet3D,
    data_dir: str,
    split: str,
    device: torch.device,
    batch_size: int = 2,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate model on a dataset split.
    
    Returns:
        Dictionary with aggregated metrics
    """
    from torch.utils.data import DataLoader
    
    dataset = SubvolumeDataset(data_dir, split=split, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    all_metrics = {'dice': [], 'iou': [], 'precision': [], 'recall': [], 'accuracy': []}
    
    for volumes, masks in tqdm(loader, desc=f"Evaluating {split}"):
        volumes = volumes.to(device)
        masks = masks.to(device)
        
        outputs = model(volumes)
        metrics = compute_metrics(outputs, masks, threshold=threshold)
        
        for k, v in metrics.items():
            all_metrics[k].append(v)
    
    # Aggregate
    agg_metrics = {}
    for k, v in all_metrics.items():
        agg_metrics[f'{k}_mean'] = np.mean(v)
        agg_metrics[f'{k}_std'] = np.std(v)
    
    return agg_metrics


@torch.no_grad()
def predict_subvolume(
    model: UNet3D,
    volume: np.ndarray,
    device: torch.device,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Predict segmentation for a single subvolume.
    
    Args:
        model: Trained UNet3D model
        volume: Input volume (D, H, W), uint8 or float32
        device: torch device
        threshold: Threshold for binary prediction
        
    Returns:
        Binary segmentation mask (D, H, W)
    """
    # Normalize
    if volume.dtype == np.uint8:
        volume = volume.astype(np.float32) / 255.0
    
    # Add batch and channel dims: (D, H, W) -> (1, 1, D, H, W)
    x = torch.from_numpy(volume[np.newaxis, np.newaxis, ...]).to(device)
    
    # Predict
    pred = model(x)
    
    # Threshold and convert back
    pred_binary = (pred > threshold).float()
    return pred_binary[0, 0].cpu().numpy()

@torch.no_grad()
def reconstruct_full_volume(
    model: nn.Module,
    volume_path: str,
    mask_path: str,
    device: torch.device,
    save_dir: Path,
    threshold: float = 0.5,
    subvol_shape: Tuple[int, int, int] = (100, 64, 64)
):
    """
    Run inference on the original full volume using a sliding window approach.
    
    Args:
        model: Trained UNet3D model
        volume_path: Path to original volume (.npy or .nii/.nii.gz)
        mask_path: Path to ground truth mask (.npy or .nii/.nii.gz)
        device: torch device
        save_dir: Directory to save outputs
        threshold: Threshold for binary prediction
        subvol_shape: Size of sliding window (must match model's expected input)
    
    Saves:
    - predicted_mask.nii.gz: Full predicted segmentation mask
    - orthogonal_views.png: Visualization of axial/coronal/sagittal slices
    - overlay_views.png: TP/FP/FN overlay visualization
    - metrics.txt: Full volume metrics
    """
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load original volume
    print(f"Loading volume from {volume_path}...")
    if volume_path.endswith('.npy'):
        volume = np.load(volume_path).astype(np.float32)
    else:
        volume = nib.load(volume_path).get_fdata().astype(np.float32)
    
    # Normalize to [0, 1]
    if volume.max() > 1.0:
        volume = volume / 255.0
    
    # Load ground truth mask
    print(f"Loading mask from {mask_path}...")
    if mask_path.endswith('.npy'):
        gt_mask = np.load(mask_path).astype(np.uint8)
    else:
        gt_mask = nib.load(mask_path).get_fdata().astype(np.uint8)
    
    D, H, W = volume.shape
    sub_d, sub_h, sub_w = subvol_shape
    
    print(f"Volume shape: {volume.shape}")
    print(f"Subvolume shape: {subvol_shape}")
    
    # Compute how many subvolumes fit (with padding for edges)
    n_d = int(np.ceil(D / sub_d))
    n_h = int(np.ceil(H / sub_h))
    n_w = int(np.ceil(W / sub_w))
    
    # Pad volume to fit exact grid
    pad_d = n_d * sub_d - D
    pad_h = n_h * sub_h - H
    pad_w = n_w * sub_w - W
    
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        volume_padded = np.pad(volume, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        print(f"Padded volume shape: {volume_padded.shape}")
    else:
        volume_padded = volume
    
    print(f"Grid: {n_d} x {n_h} x {n_w} = {n_d * n_h * n_w} subvolumes")
    
    # Initialize prediction array (padded size)
    full_pred_padded = np.zeros_like(volume_padded, dtype=np.uint8)
    
    # Process each subvolume with sliding window
    total_subvols = n_d * n_h * n_w
    with tqdm(total=total_subvols, desc="Processing subvolumes") as pbar:
        for d_idx in range(n_d):
            for h_idx in range(n_h):
                for w_idx in range(n_w):
                    d_start = d_idx * sub_d
                    h_start = h_idx * sub_h
                    w_start = w_idx * sub_w
                    
                    # Extract subvolume
                    subvol = volume_padded[
                        d_start:d_start + sub_d,
                        h_start:h_start + sub_h,
                        w_start:w_start + sub_w
                    ]
                    
                    # Predict
                    x = torch.from_numpy(subvol[np.newaxis, np.newaxis, ...]).to(device)
                    output = model(x)
                    pred = (output > threshold).float()[0, 0].cpu().numpy().astype(np.uint8)
                    
                    # Place in full prediction
                    full_pred_padded[
                        d_start:d_start + sub_d,
                        h_start:h_start + sub_h,
                        w_start:w_start + sub_w
                    ] = pred
                    
                    pbar.update(1)
    
    # Remove padding to get original size
    full_pred = full_pred_padded[:D, :H, :W]
    
    # Save predicted mask as NIfTI
    vol_name = Path(volume_path).stem.replace('-contrast-adjusted', '').replace('-joint', '')
    nib.save(
        nib.Nifti1Image(full_pred.astype(np.uint8), affine=np.eye(4)),
        save_dir / f"{vol_name}_predicted_mask.nii.gz"
    )
    print(f"Saved predicted mask to {save_dir / f'{vol_name}_predicted_mask.nii.gz'}")
    
    # Compute full volume metrics
    tp = ((gt_mask > 0) & (full_pred > 0)).sum()
    fp = ((gt_mask == 0) & (full_pred > 0)).sum()
    fn = ((gt_mask > 0) & (full_pred == 0)).sum()
    tn = ((gt_mask == 0) & (full_pred == 0)).sum()
    
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # Save metrics
    with open(save_dir / f"{vol_name}_metrics.txt", 'w') as f:
        f.write(f"Full Volume Metrics for {vol_name}\n")
        f.write(f"{'='*40}\n")
        f.write(f"Volume shape: {full_pred.shape}\n")
        f.write(f"Dice:      {dice:.4f}\n")
        f.write(f"IoU:       {iou:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"\nVoxel counts:\n")
        f.write(f"  GT foreground:   {int(gt_mask.sum()):,}\n")
        f.write(f"  Pred foreground: {int(full_pred.sum()):,}\n")
        f.write(f"  TP: {int(tp):,}, FP: {int(fp):,}, FN: {int(fn):,}\n")
    
    print(f"Dice: {dice:.4f}, IoU: {iou:.4f}, Recall: {recall:.4f}")
    
    # Create orthogonal view visualization
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # Find slices with most foreground
    gt_d_slice = np.argmax(gt_mask.sum(axis=(1, 2)))
    gt_h_slice = np.argmax(gt_mask.sum(axis=(0, 2)))
    gt_w_slice = np.argmax(gt_mask.sum(axis=(0, 1)))
    
    # Row 0: Input
    axes[0, 0].imshow(volume[gt_d_slice], cmap='gray')
    axes[0, 0].set_title(f'Input - Axial (z={gt_d_slice})')
    axes[0, 1].imshow(volume[:, gt_h_slice, :], cmap='gray')
    axes[0, 1].set_title(f'Input - Coronal (y={gt_h_slice})')
    axes[0, 2].imshow(volume[:, :, gt_w_slice], cmap='gray')
    axes[0, 2].set_title(f'Input - Sagittal (x={gt_w_slice})')
    
    # Row 1: Ground truth
    axes[1, 0].imshow(gt_mask[gt_d_slice], cmap='gray')
    axes[1, 0].set_title(f'GT - Axial (z={gt_d_slice})')
    axes[1, 1].imshow(gt_mask[:, gt_h_slice, :], cmap='gray')
    axes[1, 1].set_title(f'GT - Coronal (y={gt_h_slice})')
    axes[1, 2].imshow(gt_mask[:, :, gt_w_slice], cmap='gray')
    axes[1, 2].set_title(f'GT - Sagittal (x={gt_w_slice})')
    
    # Row 2: Prediction
    axes[2, 0].imshow(full_pred[gt_d_slice], cmap='gray')
    axes[2, 0].set_title(f'Pred - Axial (z={gt_d_slice})')
    axes[2, 1].imshow(full_pred[:, gt_h_slice, :], cmap='gray')
    axes[2, 1].set_title(f'Pred - Coronal (y={gt_h_slice})')
    axes[2, 2].imshow(full_pred[:, :, gt_w_slice], cmap='gray')
    axes[2, 2].set_title(f'Pred - Sagittal (x={gt_w_slice})')
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.suptitle(f'{vol_name} - Dice: {dice:.4f}, IoU: {iou:.4f}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / f"{vol_name}_orthogonal_views.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create overlay visualization (TP=green, FP=blue, FN=red)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for ax_idx, (slice_idx, title, axis) in enumerate([
        (gt_d_slice, f'Axial (z={gt_d_slice})', 0),
        (gt_h_slice, f'Coronal (y={gt_h_slice})', 1),
        (gt_w_slice, f'Sagittal (x={gt_w_slice})', 2)
    ]):
        if axis == 0:
            img = volume[slice_idx]
            gt_slice = gt_mask[slice_idx]
            pred_slice = full_pred[slice_idx]
        elif axis == 1:
            img = volume[:, slice_idx, :]
            gt_slice = gt_mask[:, slice_idx, :]
            pred_slice = full_pred[:, slice_idx, :]
        else:
            img = volume[:, :, slice_idx]
            gt_slice = gt_mask[:, :, slice_idx]
            pred_slice = full_pred[:, :, slice_idx]
        
        # Create RGB overlay
        overlay = np.stack([img, img, img], axis=-1)
        tp_mask = (gt_slice > 0) & (pred_slice > 0)
        fn_mask = (gt_slice > 0) & (pred_slice == 0)
        fp_mask = (gt_slice == 0) & (pred_slice > 0)
        
        overlay[tp_mask] = [0, 1, 0]  # Green = TP
        overlay[fn_mask] = [1, 0, 0]  # Red = FN
        overlay[fp_mask] = [0, 0, 1]  # Blue = FP
        
        axes[ax_idx].imshow(overlay)
        axes[ax_idx].set_title(title)
        axes[ax_idx].axis('off')
    
    plt.suptitle(f'{vol_name} Overlay (Green=TP, Red=FN, Blue=FP)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / f"{vol_name}_overlay_views.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualizations to {save_dir}")
    print(f"\nInference complete!")

def visualize_3d_comparison(
    mask_gt: np.ndarray,
    mask_pred: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Create a 3D visualization comparing ground truth and prediction.
    Shows orthogonal slices through the center.
    """
    D, H, W = mask_gt.shape
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Ground truth
    axes[0, 0].imshow(mask_gt[D//2], cmap='gray')
    axes[0, 0].set_title(f'GT - Axial (z={D//2})')
    axes[0, 1].imshow(mask_gt[:, H//2, :], cmap='gray')
    axes[0, 1].set_title(f'GT - Coronal (y={H//2})')
    axes[0, 2].imshow(mask_gt[:, :, W//2], cmap='gray')
    axes[0, 2].set_title(f'GT - Sagittal (x={W//2})')
    
    # Prediction
    axes[1, 0].imshow(mask_pred[D//2], cmap='gray')
    axes[1, 0].set_title(f'Pred - Axial (z={D//2})')
    axes[1, 1].imshow(mask_pred[:, H//2, :], cmap='gray')
    axes[1, 1].set_title(f'Pred - Coronal (y={H//2})')
    axes[1, 2].imshow(mask_pred[:, :, W//2], cmap='gray')
    axes[1, 2].set_title(f'Pred - Sagittal (x={W//2})')
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved 3D comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate 3D U-Net segmentation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--volume", type=str,
                        default="/home/sysadmin/thesis/data/rapids-p/week3-joint-contrast-adjusted.npy",
                        help="Path to input volume (.npy or .nii/.nii.gz)")
    parser.add_argument("--mask", type=str,
                        default="/home/sysadmin/thesis/data/rapids-p/week3-joint-root-class.nii.gz",
                        help="Path to ground truth mask (.npy or .nii/.nii.gz)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for binary prediction")
    parser.add_argument("--base_features", type=int, default=32,
                        help="Base feature channels (must match training)")
    parser.add_argument("--output_dir", type=str, default="results/unet3d",
                        help="Directory for output visualizations")
    
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, device, args.base_features)
    
    # Run inference on full volume
    print(f"\nRunning inference on {args.volume}...")
    reconstruct_full_volume(
        model=model,
        volume_path=args.volume,
        mask_path=args.mask,
        device=device,
        save_dir=Path(args.output_dir),
        threshold=args.threshold
    )

if __name__ == "__main__":
    main()
