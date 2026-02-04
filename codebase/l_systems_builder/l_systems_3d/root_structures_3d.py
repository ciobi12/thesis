import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import random
from scipy.ndimage import gaussian_filter
import os

from preset import LSYSTEM_PRESETS

# -------------------------------
# L-System generation
# -------------------------------
def generate_lsystem(axiom, rules, iterations):
    """Generate an L-system string after a given number of iterations."""
    current = axiom
    for _ in range(iterations):
        next_seq = []
        for ch in current:
            next_seq.append(rules.get(ch, ch))  # Apply rule if exists
        current = "".join(next_seq)
    return current

# -------------------------------
# 3D Turtle Interpreter
# -------------------------------
def draw_lsystem_3d(instructions, angle, step):
    """Interpret L-system instructions in 3D space."""
    # Turtle state: position, heading, left, up vectors
    pos = np.array([0.0, 0.0, 0.0])
    heading = np.array([0.0, 0.0, 1.0])  # Forward direction along +Z (will grow downward)
    left = np.array([-1.0, 0.0, 0.0])
    up = np.array([0.0, 1.0, 0.0])

    stack = []
    lines = []

    def rotate(vec, axis, theta):
        """Rotate vector vec around axis by theta degrees."""
        axis = axis / np.linalg.norm(axis)
        theta_rad = math.radians(theta)
        return (vec * math.cos(theta_rad) +
                np.cross(axis, vec) * math.sin(theta_rad) +
                axis * np.dot(axis, vec) * (1 - math.cos(theta_rad)))

    for cmd in instructions:
        if cmd == "F":  # Move forward and draw
            new_pos = pos + heading * step
            lines.append((pos.copy(), new_pos.copy()))
            pos = new_pos
        elif cmd == "f":  # Move forward without drawing
            pos += heading * step
        elif cmd == "+":  # Turn left around up vector
            heading = rotate(heading, up, angle)
            left = rotate(left, up, angle)
        elif cmd == "-":  # Turn right around up vector
            heading = rotate(heading, up, -angle)
            left = rotate(left, up, -angle)
        elif cmd == "&":  # Pitch down
            heading = rotate(heading, left, angle)
            up = rotate(up, left, angle)
        elif cmd == "^":  # Pitch up
            heading = rotate(heading, left, -angle)
            up = rotate(up, left, -angle)
        elif cmd == "\\":  # Roll left
            left = rotate(left, heading, angle)
            up = rotate(up, heading, angle)
        elif cmd == "/":  # Roll right
            left = rotate(left, heading, -angle)
            up = rotate(up, heading, -angle)
        elif cmd == "|":  # Turn 180 degrees
            heading = -heading
            left = -left
        elif cmd == "[":  # Save state
            stack.append((pos.copy(), heading.copy(), left.copy(), up.copy()))
        elif cmd == "]":  # Restore state
            pos, heading, left, up = stack.pop()

    return lines

# -------------------------------
# Convert line segments to voxel volume
# -------------------------------
def draw_line_3d_volume(volume, p1, p2, thickness=1):
    """Draw a 3D line in the volume using DDA algorithm."""
    D, H, W = volume.shape
    
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    
    steps = int(max(abs(dx), abs(dy), abs(dz))) + 1
    if steps == 0:
        return
    
    x_inc = dx / steps
    y_inc = dy / steps
    z_inc = dz / steps
    
    x, y, z = p1[0], p1[1], p1[2]
    
    for _ in range(steps + 1):
        ix, iy, iz = int(round(x)), int(round(y)), int(round(z))
        
        for dz_off in range(-thickness + 1, thickness):
            for dy_off in range(-thickness + 1, thickness):
                for dx_off in range(-thickness + 1, thickness):
                    if dx_off**2 + dy_off**2 + dz_off**2 <= thickness**2:
                        vx = ix + dx_off
                        vy = iy + dy_off
                        vz = iz + dz_off
                        if 0 <= vx < W and 0 <= vy < H and 0 <= vz < D:
                            volume[vz, vy, vx] = 1
        
        x += x_inc
        y += y_inc
        z += z_inc

def segments_to_volume(segments, volume_shape=(64, 64, 64), thickness=1):
    """Convert line segments to a voxel volume.
    
    Structure grows along Z-axis with starting point at z=0 (upper-most slice).
    Volume shape is (D, H, W) = (depth, height, width).
    """
    D, H, W = volume_shape
    volume = np.zeros(volume_shape, dtype=np.uint8)
    
    if len(segments) == 0:
        return volume
    
    # Collect all points from segments
    all_points = []
    for start, end in segments:
        all_points.append(start)
        all_points.append(end)
    all_points = np.array(all_points)
    
    # Get bounds
    min_vals = all_points.min(axis=0)
    max_vals = all_points.max(axis=0)
    
    # Add margin
    margin = 2
    
    # Scale and translate points to fit in volume
    # X -> W (width), Y -> H (height), Z -> D (depth)
    scaled_segments = []
    for start, end in segments:
        scaled_start = np.zeros(3)
        scaled_end = np.zeros(3)
        
        for i, (s, e) in enumerate(zip(start, end)):
            range_val = max_vals[i] - min_vals[i]
            if range_val < 1e-8:
                range_val = 1.0
            
            if i == 0:  # X -> W
                scaled_start[i] = (s - min_vals[i]) / range_val * (W - 1 - 2*margin) + margin
                scaled_end[i] = (e - min_vals[i]) / range_val * (W - 1 - 2*margin) + margin
            elif i == 1:  # Y -> H
                scaled_start[i] = (s - min_vals[i]) / range_val * (H - 1 - 2*margin) + margin
                scaled_end[i] = (e - min_vals[i]) / range_val * (H - 1 - 2*margin) + margin
            else:  # Z -> D (depth grows along Z, starting at z=0)
                scaled_start[i] = (s - min_vals[i]) / range_val * (D - 1 - 2*margin) + margin
                scaled_end[i] = (e - min_vals[i]) / range_val * (D - 1 - 2*margin) + margin
        
        scaled_segments.append((scaled_start, scaled_end))
    
    # Draw lines
    for start, end in scaled_segments:
        draw_line_3d_volume(volume, start, end, thickness)
    
    return volume

def add_gaussian_noise(volume, noise_level=0.1):
    """Add Gaussian noise to volume."""
    noisy = volume.astype(np.float32)
    noise = np.random.normal(0, noise_level, volume.shape)
    noisy = noisy + noise
    return np.clip(noisy, 0, 1)

def create_ct_volume(mask, background_intensity=0.7, root_intensity_range=(0.2, 0.4)):
    """Create CT-like volume where roots are DARKER than background.
    
    Args:
        mask: Binary mask (1=root, 0=background)
        background_intensity: Base intensity for background (soil/medium)
        root_intensity_range: (min, max) intensity range for root structures
    
    Returns:
        Float volume with CT-like contrast
    """
    volume = np.zeros_like(mask, dtype=np.float32)
    
    # Background gets high intensity (lighter)
    volume[mask == 0] = background_intensity
    
    # Roots get low intensity (darker) with slight variation
    root_locations = mask == 1
    num_root_voxels = root_locations.sum()
    if num_root_voxels > 0:
        root_intensities = np.random.uniform(
            root_intensity_range[0], 
            root_intensity_range[1], 
            num_root_voxels
        )
        volume[root_locations] = root_intensities
    
    return volume

def add_ct_artifacts(volume, noise_sigma=0.15, streak_prob=0.15, blur_sigma=0.7):
    """Add CT-like artifacts: Gaussian noise, streaks, and reconstruction blur.
    
    Args:
        volume: Float volume with CT-like contrast
        noise_sigma: Standard deviation of Gaussian noise
        streak_prob: Probability of streak artifacts per slice
        blur_sigma: Gaussian blur to simulate CT reconstruction
    
    Returns:
        Noisy volume with artifacts
    """
    noisy = volume.copy()
    
    # Add stronger Gaussian noise
    noise = np.random.normal(0, noise_sigma, volume.shape)
    noisy = noisy + noise
    
    # Add random streak artifacts (simulate beam hardening, ring artifacts)
    for z in range(volume.shape[0]):
        if np.random.rand() < streak_prob:
            # Vertical or horizontal streaks
            if np.random.rand() < 0.5:
                # Horizontal streak with stronger intensity
                streak_intensity = np.random.uniform(-0.2, 0.2)
                streak_y = np.random.randint(0, volume.shape[1])
                noisy[z, streak_y, :] += streak_intensity
            else:
                # Vertical streak with stronger intensity
                streak_intensity = np.random.uniform(-0.2, 0.2)
                streak_x = np.random.randint(0, volume.shape[2])
                noisy[z, :, streak_x] += streak_intensity
    
    # Add occasional ring artifacts (concentric patterns in CT)
    if np.random.rand() < 0.2:  # 20% chance per volume (increased)
        z_slice = np.random.randint(0, volume.shape[0])
        center_y, center_x = volume.shape[1]//2, volume.shape[2]//2
        y, x = np.ogrid[:volume.shape[1], :volume.shape[2]]
        radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        ring_pattern = np.sin(radius * 0.5) * 0.08  # Stronger rings
        noisy[z_slice] += ring_pattern
    
    # Add root-like dark artifacts (blobs with similar intensity to real roots)
    num_artifacts = np.random.randint(3, 8)  # 3-7 artifacts per volume
    for _ in range(num_artifacts):
        # Random position
        z = np.random.randint(0, volume.shape[0])
        y = np.random.randint(0, volume.shape[1])
        x = np.random.randint(0, volume.shape[2])
        
        # Random size (small blobs)
        size = np.random.randint(2, 5)
        
        # Root-like dark intensity (similar to actual roots: 0.2-0.4)
        artifact_intensity = np.random.uniform(0.15, 0.45)
        
        # Create blob artifact
        for dz in range(-size, size+1):
            for dy in range(-size, size+1):
                for dx in range(-size, size+1):
                    nz, ny, nx = z + dz, y + dy, x + dx
                    if (0 <= nz < volume.shape[0] and 
                        0 <= ny < volume.shape[1] and 
                        0 <= nx < volume.shape[2]):
                        # Distance-based falloff for more realistic blobs
                        dist = np.sqrt(dz**2 + dy**2 + dx**2)
                        if dist < size:
                            weight = 1.0 - (dist / size)
                            # Blend artifact with existing intensity
                            noisy[nz, ny, nx] = noisy[nz, ny, nx] * (1 - weight) + artifact_intensity * weight
    
    # Add elongated artifacts (simulate root-like linear structures)
    num_linear = np.random.randint(2, 5)  # 2-4 linear artifacts
    for _ in range(num_linear):
        # Random start position
        z = np.random.randint(0, volume.shape[0])
        y = np.random.randint(0, volume.shape[1])
        x = np.random.randint(0, volume.shape[2])
        
        # Random direction
        direction = np.random.randn(3)
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        
        # Random length
        length = np.random.randint(5, 15)
        
        # Root-like intensity
        artifact_intensity = np.random.uniform(0.2, 0.4)
        
        # Draw line
        for step in range(length):
            nz = int(z + step * direction[0])
            ny = int(y + step * direction[1])
            nx = int(x + step * direction[2])
            if (0 <= nz < volume.shape[0] and 
                0 <= ny < volume.shape[1] and 
                0 <= nx < volume.shape[2]):
                noisy[nz, ny, nx] = artifact_intensity
    
    # Smooth slightly to simulate CT reconstruction
    noisy = gaussian_filter(noisy, sigma=blur_sigma)
    
    return np.clip(noisy, 0, 1)

def generate_dataset(num_samples=5, 
                     presets = {"plant_basic": {"axiom": "F",
                                                "rules": {"F": "F/[+F]F&[-F]F"},
                                                "angle": 25,
                                                "description": "Basic 3D plant with balanced branching"}
                                                },
                     size = (128, 128, 128),
                     noise_sigma=0.15,
                     streak_prob=0.15, 
                     output_dir="generated_roots"):
    """Generate multiple root volumes with variations."""
    os.makedirs(output_dir, exist_ok=True)
    
    i = 0

    os.makedirs(os.path.join(output_dir, "train", "volumes"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "train", "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val", "volumes"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val", "masks"), exist_ok=True)
    
    while i < num_samples:
        print(f"Generating sample {i+1}/{num_samples} ...")
        
        # Vary parameters for each sample - optimized for THIN, CONNECTED root structures
        iterations = np.random.randint(3, 5)  # Moderate iterations
        step = np.random.uniform(1.0, 2.0)  # Step size for forward movement
        thickness = 1  # Thin roots (single voxel width)
        
        rand_preset = random.choice(list(presets.values()))
        print(f"Using preset: {rand_preset['description']}")
        # Generate root structure with full 3D turtle graphics
        lsys_str = generate_lsystem(
            axiom = rand_preset["axiom"],
            rules =  rand_preset["rules"],
            iterations=iterations)

        segments = draw_lsystem_3d(lsys_str, 25, step)
        
        # Convert to volume with connected lines (this is the clean mask)
        mask = segments_to_volume(segments, volume_shape = size, thickness=thickness)
        print("Sparsity:")
        print(f"\nVolume shape: {mask.shape} (D, H, W)")
        print(f"Non-zero voxels: {mask.sum()}")
        print(f"Voxel occupancy: {mask.sum() / mask.size * 100:.2f}%")
        
        # Create CT-like volume: roots are DARKER than background
        background_intensity = np.random.uniform(0.65, 0.75)  # Vary background per sample
        root_intensity_range = (0.2, 0.4)  # Roots are darker
        volume_clean = create_ct_volume(mask, background_intensity, root_intensity_range)
        
        # Add CT artifacts and noise (with stronger parameters for more challenging dataset)
        volume_noisy = add_ct_artifacts(volume_clean, 
                                        noise_sigma=noise_sigma,
                                        streak_prob=streak_prob, 
                                        blur_sigma=0.7)
        
        # Convert to uint8 for storage
        volume_noisy_uint8 = (volume_noisy * 255).astype(np.uint8)
        
        # Save files
        if np.random.rand() < 0.8:
            np.save(os.path.join(output_dir, "train", "volumes", f"root_volume_{i:03d}.npy"), volume_noisy_uint8)
            np.save(os.path.join(output_dir, "train", "masks", f"root_mask_{i:03d}.npy"), mask)
        else:
            np.save(os.path.join(output_dir, "val", "volumes", f"root_volume_{i:03d}.npy"), volume_noisy_uint8)
            np.save(os.path.join(output_dir, "val", "masks", f"root_mask_{i:03d}.npy"), mask)   
        
        print(f"  ✓ Saved: root_volume_{i:03d}.npy and root_mask_{i:03d}.npy")
        i += 1  # Only increment on successful generation
    
# Generate dataset
if __name__ == "__main__":    
    print("Generating root structure dataset...")
    ds_dir = "../../../data/ct_like/3d"
    generate_dataset(num_samples=20,
                     presets=LSYSTEM_PRESETS,
                     size=(128, 128, 128), 
                     noise_sigma=0.1,
                     streak_prob=0.1,
                     output_dir=ds_dir)
    print("\nDone! Volumes saved to 'data/ct_like/3d' directory")

    val_vols_path = os.path.join(ds_dir, "val", "volumes")
    val_masks_path = os.path.join(ds_dir, "val", "masks")
    
    # Find the validation volume with the highest number of non-zero voxels
    best_vol_path = None
    best_nonzero_count = 0
    for vol_file in os.listdir(val_vols_path):
        mask_file = vol_file.replace("volume", "mask")
        mask_tmp = np.load(os.path.join(val_masks_path, mask_file))
        nonzero_count = np.count_nonzero(mask_tmp)
        if nonzero_count > best_nonzero_count:
            best_nonzero_count = nonzero_count
            best_vol_path = vol_file
    
    print(f"Selected volume with highest voxel count: {best_vol_path} ({best_nonzero_count} non-zero voxels)")
    mask_path = best_vol_path.replace("volume", "mask")

    volume = np.load(os.path.join(val_vols_path, best_vol_path)).astype(np.float32) / 255.0
    mask = np.load(os.path.join(val_masks_path, mask_path)).astype(np.float32)  # Don't divide by 255, mask is already 0/1
    print(np.unique(mask))
    val_masks = []
    for file in os.listdir(val_masks_path):
        val_masks.append(np.load(os.path.join(val_masks_path, file)).astype(np.float32))  # Don't divide by 255 
    
    # Select top 3 slices with highest number of non-zero voxels
    D = mask.shape[0]
    slice_nonzero_counts = [(z, np.count_nonzero(mask[z])) for z in range(D)]
    slice_nonzero_counts.sort(key=lambda x: x[1], reverse=True)
    slice_indices = sorted([s[0] for s in slice_nonzero_counts[:3]])
    print(f"Selected slices with highest voxel counts: {[(s[0], s[1]) for s in slice_nonzero_counts[:3]]}")
    
    print("\nVisualizing sample with slices...")
    # fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    # fig.suptitle('Volume Slices: Noisy (1st row) | Mask (2nd row)', fontsize=14)
    
    # for i, slice_idx in enumerate(slice_indices):
    #     # Noisy volume slice
    #     axes[0, i].imshow(volume[slice_idx], cmap='gray', vmin=0, vmax=1)
    #     axes[0, i].set_title(f'Noisy - Slice {slice_idx}/{D}')
    #     axes[0, i].axis('off')
        
    #     # Mask slice
    #     axes[1, i].imshow(mask[slice_idx], cmap='gray', vmin=0, vmax=1)
    #     axes[1, i].set_title(f'Mask - Slice {slice_idx}/{D}')
    #     axes[1, i].axis('off')
        
    #     # # Overlay: show mask in red on top of noisy volume
    #     # overlay = np.stack([
    #     #     volume[slice_idx] + 0.5 * mask[slice_idx],  # Red channel
    #     #     volume[slice_idx],                          # Green channel
    #     #     volume[slice_idx]                           # Blue channel
    #     # ], axis=-1)
    #     # overlay = np.clip(overlay, 0, 1)
    #     # axes[i, 2].imshow(overlay)
    #     # axes[i, 2].set_title(f'Overlay - Slice {slice_idx}/{D}')
    #     # axes[i, 2].axis('off')
    
    # plt.tight_layout()
    # plt.savefig('slice_visualization.png', dpi=150, bbox_inches='tight')
    # print("Saved slice visualization to 'slice_visualization.png'")
    # plt.show()
    
    # # 3D visualization with PyVista
    # print("\nShowing 3D visualization...")
    # pv.global_theme.allow_empty_mesh = True
    # plotter = pv.Plotter(shape=(1, 2), window_size=(1800, 600))
    
    # # Add main title
    # plotter.add_text("Generated 3D L-systems", position='upper_edge', font_size=16, color='black')
    
    # for i, mask in enumerate(val_masks[:2]):
    #     # Noisy volume - multiple contours
    #     plotter.subplot(0, i)
    #     grid = pv.ImageData()
    #     grid.dimensions = np.array(mask.shape)
    #     grid.origin = (0, 0, 0)
    #     grid.spacing = (1, 1, 1)
    #     grid.point_data["values"] = mask.flatten(order="F")

    #     # Use adaptive threshold based on volume statistics
    #     threshold = min(0.3, mask.max() * 0.5)
    #     contour = grid.contour([0.5])
    #     if contour.n_points > 0:
    #         plotter.add_mesh(contour, color="tan", opacity=0.7)
    #     else:
    #         plotter.add_text("Empty mask", font_size=10)
        
    #     # Set camera to view structure straight (front view looking at XY plane)
    #     # plotter.camera_position = 'yz'
    #     # plotter.camera.elevation = 0
    #     # plotter.camera.azimuth = 30
    
    # try:
    #     # Save snapshot (off_screen mode allows screenshot without display)
    #     # plotter.screenshot('3d_lsystems_visualization.png')
    #     # print("Saved 3D visualization to '3d_lsystems_visualization.png'")
    #     plotter.show()
    #     # plotter.close()
    #     # plt.imshow(plt.imread('3d_lsystems_visualization.png'))
    # except Exception as e:
    #     print(f"Could not save 3D plot: {e}")
    #     print("Visualization completed successfully (slice images saved).")