"""
Preprocess large CT volumes into smaller subvolumes for training.

Given a volume of shape (D, H, W), split into subvolumes
of shape (sub_d, sub_h, sub_w).

- Discard marginal subvolumes (those that don't fit full 64x64 slices)
- Discard corner subvolumes that are pure background (no foreground in mask)
- Save remaining subvolumes as individual .npy files
"""

import os
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Tuple, List, Optional


def load_volume(path: str) -> np.ndarray:
    """Load volume from .npy or .nii/.nii.gz file."""
    if path.endswith('.npy'):
        return np.load(path)
    else:
        return nib.load(path).get_fdata()


def compute_subvolume_grid(
    volume_shape: Tuple[int, int, int],
    subvol_shape: Tuple[int, int, int]
) -> Tuple[int, int, int]:
    """
    Compute how many full subvolumes fit along each axis.
    
    Returns:
        (n_depth, n_height, n_width): Number of subvolumes per axis
    """
    D, H, W = volume_shape
    sub_d, sub_h, sub_w = subvol_shape
    
    n_depth = D // sub_d
    n_height = H // sub_h
    n_width = W // sub_w
    
    return n_depth, n_height, n_width


def extract_subvolume(
    volume: np.ndarray,
    d_idx: int, h_idx: int, w_idx: int,
    subvol_shape: Tuple[int, int, int]
) -> np.ndarray:
    """Extract a subvolume at the given grid indices."""
    sub_d, sub_h, sub_w = subvol_shape
    
    d_start = d_idx * sub_d
    h_start = h_idx * sub_h
    w_start = w_idx * sub_w
    
    return volume[
        d_start:d_start + sub_d,
        h_start:h_start + sub_h,
        w_start:w_start + sub_w
    ].copy()


def is_corner_background(
    mask_subvol: np.ndarray,
    min_foreground_ratio: float = 0.0001
) -> bool:
    """
    Check if a subvolume is mostly background (corner region).
    
    Args:
        mask_subvol: Binary mask subvolume
        min_foreground_ratio: Minimum ratio of foreground voxels to keep
        
    Returns:
        True if subvolume should be discarded (is background)
    """
    fg_ratio = mask_subvol.sum() / mask_subvol.size
    return fg_ratio < min_foreground_ratio


def split_volume_into_subvolumes(
    volume: np.ndarray,
    mask: np.ndarray,
    subvol_shape: Tuple[int, int, int] = (100, 64, 64),
    min_foreground_ratio: float = 0.0001,
    verbose: bool = True
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[int, int, int]]]:
    """
    Split volume and mask into subvolumes, discarding margins and empty corners.
    
    Args:
        volume: Full CT volume (D, H, W)
        mask: Full segmentation mask (D, H, W)
        subvol_shape: Target subvolume shape (sub_d, sub_h, sub_w)
        min_foreground_ratio: Minimum foreground ratio to keep subvolume
        verbose: Print progress info
        
    Returns:
        volumes: List of volume subvolumes
        masks: List of mask subvolumes
        indices: List of (d_idx, h_idx, w_idx) grid positions
    """
    D, H, W = volume.shape
    sub_d, sub_h, sub_w = subvol_shape
    
    # Compute grid
    n_depth, n_height, n_width = compute_subvolume_grid(volume.shape, subvol_shape)
    
    if verbose:
        print(f"Volume shape: {volume.shape}")
        print(f"Subvolume shape: {subvol_shape}")
        print(f"Grid: {n_depth} x {n_height} x {n_width} = {n_depth * n_height * n_width} potential subvolumes")
        print(f"Margins discarded: D={D - n_depth*sub_d}, H={H - n_height*sub_h}, W={W - n_width*sub_w}")
    
    volumes = []
    masks = []
    indices = []
    
    n_discarded_empty = 0
    
    for d_idx in range(n_depth):
        for h_idx in range(n_height):
            for w_idx in range(n_width):
                # Extract subvolumes
                vol_sub = extract_subvolume(volume, d_idx, h_idx, w_idx, subvol_shape)
                mask_sub = extract_subvolume(mask, d_idx, h_idx, w_idx, subvol_shape)
                
                # Check if it's a background-only corner
                if is_corner_background(mask_sub, min_foreground_ratio):
                    n_discarded_empty += 1
                    continue
                
                volumes.append(vol_sub)
                masks.append(mask_sub)
                indices.append((d_idx, h_idx, w_idx))
    
    if verbose:
        print(f"Discarded {n_discarded_empty} empty/corner subvolumes")
        print(f"Kept {len(volumes)} subvolumes with foreground")
        
        # Statistics on kept subvolumes
        fg_ratios = [m.sum() / m.size * 100 for m in masks]
        print(f"Foreground ratios: min={min(fg_ratios):.4f}%, max={max(fg_ratios):.4f}%, mean={np.mean(fg_ratios):.4f}%")
    
    return volumes, masks, indices


def save_subvolumes(
    volumes: List[np.ndarray],
    masks: List[np.ndarray],
    indices: List[Tuple[int, int, int]],
    output_dir: str,
    prefix: str = "subvol"
):
    """
    Save subvolumes to disk in separate volumes/ and masks/ folders.
    
    Args:
        volumes: List of volume subvolumes
        masks: List of mask subvolumes
        indices: Grid indices for naming
        output_dir: Base output directory
        prefix: Filename prefix
    """
    volumes_dir = Path(output_dir) / "volumes"
    masks_dir = Path(output_dir) / "masks"
    
    volumes_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving subvolumes...")
    print(f"Volumes dir: {volumes_dir}")
    print(f"Masks dir: {masks_dir}")
    
    for i, (vol, mask, (d, h, w)) in enumerate(zip(volumes, masks, indices)):
        name = f"{prefix}_d{d:02d}_h{h:02d}_w{w:02d}"
        
        # Save volume and mask to separate folders
        np.save(volumes_dir / f"{name}.npy", vol.astype(np.uint8))
        np.save(masks_dir / f"{name}.npy", (mask > 0).astype(np.uint8))
    
    print(f"Saved {len(volumes)} subvolumes")


def visualize_grid(
    volume_shape: Tuple[int, int, int],
    subvol_shape: Tuple[int, int, int],
    kept_indices: List[Tuple[int, int, int]],
    output_path: Optional[str] = None
):
    """Visualize which grid positions were kept vs discarded."""
    import matplotlib.pyplot as plt
    
    n_d, n_h, n_w = compute_subvolume_grid(volume_shape, subvol_shape)
    
    # Create grid visualization (sum over depth for 2D view)
    grid = np.zeros((n_h, n_w))
    for d, h, w in kept_indices:
        grid[h, w] += 1
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(grid, cmap='YlOrRd', origin='lower')
    ax.set_xlabel('Width index')
    ax.set_ylabel('Height index')
    ax.set_title(f'Subvolume grid: kept positions (summed over {n_d} depth slabs)\n'
                 f'Total: {len(kept_indices)} subvolumes kept')
    plt.colorbar(im, ax=ax, label='Count across depth')
    
    # Add grid lines
    for x in range(n_w + 1):
        ax.axvline(x - 0.5, color='gray', linewidth=0.5, alpha=0.5)
    for y in range(n_h + 1):
        ax.axhline(y - 0.5, color='gray', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Grid visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Split large CT volume into subvolumes')
    
    parser.add_argument('--volume', type=str, required=True,
                        help='Path to volume (.npy or .nii)')
    parser.add_argument('--mask', type=str, required=True,
                        help='Path to mask (.npy or .nii.gz)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for subvolumes')
    parser.add_argument('--prefix', type=str, default='subvol',
                        help='Filename prefix')
    
    # Subvolume dimensions
    parser.add_argument('--sub_d', type=int, default=100,
                        help='Subvolume depth')
    parser.add_argument('--sub_h', type=int, default=64,
                        help='Subvolume height')
    parser.add_argument('--sub_w', type=int, default=64,
                        help='Subvolume width')
    
    # Filtering
    parser.add_argument('--min_fg_ratio', type=float, default=0.0001,
                        help='Minimum foreground ratio to keep subvolume')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true',
                        help='Save grid visualization')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading volume: {args.volume}")
    volume = load_volume(args.volume)
    
    print(f"Loading mask: {args.mask}")
    mask = load_volume(args.mask)
    
    assert volume.shape == mask.shape, f"Shape mismatch: {volume.shape} vs {mask.shape}"
    
    # Split into subvolumes
    subvol_shape = (args.sub_d, args.sub_h, args.sub_w)
    
    volumes, masks, indices = split_volume_into_subvolumes(
        volume, mask,
        subvol_shape=subvol_shape,
        min_foreground_ratio=args.min_fg_ratio,
        verbose=True
    )
    
    # Save
    save_subvolumes(
        volumes, masks, indices,
        output_dir=args.output_dir,
        prefix=args.prefix
    )
    
    # Visualize
    if args.visualize:
        vis_path = Path(args.output_dir) / "grid_visualization.png"
        visualize_grid(volume.shape, subvol_shape, indices, str(vis_path))
    
    print("\nDone!")


if __name__ == "__main__":
    main()
