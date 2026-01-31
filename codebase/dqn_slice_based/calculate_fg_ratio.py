"""
Calculate foreground ratio for volumes in the CT-Like dataset.
Reports the percentage of voxels labeled as 1 in each mask.
"""

import os
import numpy as np
from pathlib import Path


def calculate_fg_ratios(data_dir: str):
    """
    Calculate foreground ratio for all masks in the dataset.

    Args:
        data_dir: Path to dataset directory containing train/val subdirectories
    """
    results = {"train": [], "val": []}

    for split in ["train", "val"]:
        masks_dir = os.path.join(data_dir, split, "masks")

        if not os.path.exists(masks_dir):
            print(f"Warning: {masks_dir} does not exist")
            continue

        mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith(".npy")])

        print(f"\n{'='*60}")
        print(f"{split.upper()} SET ({len(mask_files)} volumes)")
        print(f"{'='*60}")
        print(f"{'File':<30} {'Shape':<20} {'FG Ratio':<12} {'FG Voxels':<15}")
        print("-" * 80)

        for mask_file in mask_files:
            mask_path = os.path.join(masks_dir, mask_file)
            mask = np.load(mask_path)

            total_voxels = mask.size
            fg_voxels = (mask > 0).sum()
            fg_ratio = fg_voxels / total_voxels

            results[split].append({
                "file": mask_file,
                "shape": mask.shape,
                "fg_ratio": fg_ratio,
                "fg_voxels": fg_voxels,
                "total_voxels": total_voxels
            })

            print(f"{mask_file:<30} {str(mask.shape):<20} {fg_ratio*100:>8.4f}%   {fg_voxels:>12,}")

    # Print summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")

    for split in ["train", "val"]:
        if not results[split]:
            continue

        ratios = [r["fg_ratio"] for r in results[split]]
        fg_voxels = [r["fg_voxels"] for r in results[split]]

        print(f"\n{split.upper()}:")
        print(f"  Number of volumes: {len(results[split])}")
        print(f"  Foreground ratio:")
        print(f"    Mean:   {np.mean(ratios)*100:.4f}%")
        print(f"    Std:    {np.std(ratios)*100:.4f}%")
        print(f"    Min:    {np.min(ratios)*100:.4f}%")
        print(f"    Max:    {np.max(ratios)*100:.4f}%")
        print(f"    Median: {np.median(ratios)*100:.4f}%")
        print(f"  Foreground voxels per volume:")
        print(f"    Mean:   {np.mean(fg_voxels):,.0f}")
        print(f"    Min:    {np.min(fg_voxels):,}")
        print(f"    Max:    {np.max(fg_voxels):,}")

    # Combined statistics
    all_ratios = [r["fg_ratio"] for split in results.values() for r in split]
    if all_ratios:
        print(f"\nOVERALL ({len(all_ratios)} volumes):")
        print(f"  Mean FG ratio: {np.mean(all_ratios)*100:.4f}%")
        print(f"  Std FG ratio:  {np.std(all_ratios)*100:.4f}%")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate foreground ratio for CT-Like dataset")
    parser.add_argument("--data_dir", type=str, default="../../data/ct_like/3d",
                        help="Path to dataset directory")
    args = parser.parse_args()

    calculate_fg_ratios(args.data_dir)
