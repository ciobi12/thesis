"""
Script to split all L-system volumes into subvolumes for training.
"""

import os
import subprocess
from pathlib import Path

# Configuration
DATA_DIR = "/home/sysadmin/thesis/data/ct_like/3d"
OUTPUT_DIR = "/home/sysadmin/thesis/data/ct_like/3d_subvolumes"
SUB_D = 32
SUB_H = 32
SUB_W = 32
MIN_FG_RATIO = 0.0001

def process_split(split_name: str):
    """Process all volumes in a split (train or val)."""
    print(f"\n{'='*60}")
    print(f"Processing {split_name} split")
    print(f"{'='*60}")

    volumes_dir = os.path.join(DATA_DIR, split_name, "volumes")
    masks_dir = os.path.join(DATA_DIR, split_name, "masks")

    # Get list of volume files
    volume_files = sorted([f for f in os.listdir(volumes_dir) if f.endswith('.npy')])

    print(f"Found {len(volume_files)} volumes to process")

    for i, vol_file in enumerate(volume_files):
        vol_path = os.path.join(volumes_dir, vol_file)

        # Get corresponding mask file
        mask_file = vol_file.replace("volume", "mask")
        mask_path = os.path.join(masks_dir, mask_file)

        if not os.path.exists(mask_path):
            print(f"Skipping {vol_file} - no matching mask found")
            continue

        # Extract volume ID (e.g., "000" from "root_volume_000.npy")
        vol_id = vol_file.replace("root_volume_", "").replace(".npy", "")

        # Output directory for this volume's subvolumes
        output_path = os.path.join(OUTPUT_DIR, split_name, f"vol_{vol_id}")

        print(f"\n[{i+1}/{len(volume_files)}] Processing {vol_file} -> {output_path}")

        # Run preprocess_subvolumes.py
        cmd = [
            "python3", "preprocess_subvolumes.py",
            "--volume", vol_path,
            "--mask", mask_path,
            "--output_dir", output_path,
            "--prefix", "subvol",
            "--sub_d", str(SUB_D),
            "--sub_h", str(SUB_H),
            "--sub_w", str(SUB_W),
            "--min_fg_ratio", str(MIN_FG_RATIO),
            "--visualize"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"ERROR processing {vol_file}:")
            print(result.stderr)
        else:
            # Parse output to show summary
            for line in result.stdout.split('\n'):
                if 'Kept' in line or 'Discarded' in line or 'Foreground ratios' in line:
                    print(f"  {line}")

def main():
    # Process train and val splits
    for split_name in ["train", "val"]:
        process_split(split_name)

    print(f"\n{'='*60}")
    print("All volumes processed!")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
