import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_filter, binary_dilation
import os

# Different L-system rule sets for variety
RULE_SETS = [
    {"F": "F[+F]F[-F]F"},           # Original: balanced branching
    {"F": "F[++F][--F]F"},          # Wider angles
    {"F": "FF[+F][-F]F"},           # Longer segments
    {"F": "F[+F]F[-F][F]"},         # More complex
    {"F": "F[+F][F][-F]F"},         # Asymmetric
]

def apply_rules(axiom, iterations, rules):
    seq = axiom
    for _ in range(iterations):
        seq = "".join(rules.get(ch, ch) for ch in seq)
    return seq

def generate_root_structure(iterations=3, angle_base=np.pi/6, step=0.1, 
                            angle_variation=0.1, rules_idx=0, seed=None):
    """Generate a root structure using L-systems with variations."""
    if seed is not None:
        np.random.seed(seed)
    
    rules = RULE_SETS[rules_idx % len(RULE_SETS)]
    axiom = "F"
    sequence = apply_rules(axiom, iterations, rules)
    
    # Parameters with variation
    points = [[0, 0, 0]]
    lines = []
    stack = []
    direction = np.array([0, 0, -1])  # downward root growth
    
    for ch in sequence:
        if ch == "F":
            # Add slight randomness to step length
            current_step = step * (1 + np.random.uniform(-0.2, 0.2))
            new_point = points[-1] + current_step * direction
            points.append(new_point.tolist())
            lines.append([2, len(points)-2, len(points)-1])
        elif ch == "+":
            # Vary angle slightly
            angle = angle_base + np.random.uniform(-angle_variation, angle_variation)
            # Rotate around x-axis
            rot = np.array([[1, 0, 0],
                           [0, np.cos(angle), -np.sin(angle)],
                           [0, np.sin(angle), np.cos(angle)]])
            direction = rot @ direction
        elif ch == "-":
            # Vary angle slightly
            angle = angle_base + np.random.uniform(-angle_variation, angle_variation)
            # Rotate around y-axis
            rot = np.array([[np.cos(angle), 0, np.sin(angle)],
                           [0, 1, 0],
                           [-np.sin(angle), 0, np.cos(angle)]])
            direction = rot @ direction
        elif ch == "[":
            stack.append((points[-1], direction.copy()))
        elif ch == "]":
            if stack:
                pos, direction = stack.pop()
                points.append(pos)
    
    return np.array(points), lines

def points_to_volume(points, res_d=64, res_h=16, res_w=16, thickness=1):
    """Convert point cloud to voxel volume."""
    volume = np.zeros((res_d, res_h, res_w), dtype=np.uint8)
    
    # Normalize each axis separately
    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    
    scaled_points = np.zeros_like(points)
    scaled_points[:, 0] = (points[:, 0] - min_vals[0]) / (max_vals[0] - min_vals[0] + 1e-8) * (res_w - 1)
    scaled_points[:, 1] = (points[:, 1] - min_vals[1]) / (max_vals[1] - min_vals[1] + 1e-8) * (res_h - 1)
    scaled_points[:, 2] = (points[:, 2] - min_vals[2]) / (max_vals[2] - min_vals[2] + 1e-8) * (res_d - 1)
    
    scaled_points = scaled_points.astype(int)
    
    # Fill voxels with optional thickness
    for p in scaled_points:
        x, y, z = p
        if 0 <= z < res_d and 0 <= y < res_h and 0 <= x < res_w:
            volume[z, y, x] = 1
    
    # Thicken the structure slightly
    if thickness > 1:
        from scipy.ndimage import binary_dilation
        volume = binary_dilation(volume, iterations=thickness-1).astype(np.uint8)
    
    return volume

def add_gaussian_noise(volume, noise_level=0.1):
    """Add Gaussian noise to volume."""
    noisy = volume.astype(np.float32)
    noise = np.random.normal(0, noise_level, volume.shape)
    noisy = noisy + noise
    return np.clip(noisy, 0, 1)

def add_ct_artifacts(volume, noise_sigma=0.05, streak_prob=0.02):
    """Add CT-like artifacts: Gaussian noise and occasional streaks."""
    # Start with float conversion
    noisy = volume.astype(np.float32)
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_sigma, volume.shape)
    noisy = noisy + noise
    
    # Add random streak artifacts (simulate beam hardening)
    for z in range(volume.shape[0]):
        if np.random.rand() < streak_prob:
            streak_intensity = np.random.uniform(0.1, 0.3)
            streak_y = np.random.randint(0, volume.shape[1])
            noisy[z, streak_y, :] += streak_intensity
    
    # Smooth slightly to simulate CT reconstruction
    noisy = gaussian_filter(noisy, sigma=0.5)
    
    return np.clip(noisy, 0, 1)

def generate_dataset(num_samples=5, size = (64, 16, 16), output_dir="generated_roots"):
    """Generate multiple root volumes with variations."""
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_samples):
        print(f"Generating sample {i+1}/{num_samples}...")
        
        # Vary parameters for each sample
        iterations = np.random.randint(2, 5)
        angle_base = np.random.uniform(np.pi/8, np.pi/4)
        step = np.random.uniform(0.08, 0.12)
        rules_idx = i % len(RULE_SETS)
        thickness = np.random.randint(1, 3)
        
        # Generate root structure
        points, lines = generate_root_structure(
            iterations=iterations,
            angle_base=angle_base,
            step=step,
            angle_variation=0.1,
            rules_idx=rules_idx,
            seed=42 + i
        )
        
        # Convert to volume (this is the clean mask)
        mask = points_to_volume(points, res_d=size[0], res_h=size[1], res_w=size[2], thickness=thickness)
        
        # Create noisy volume (simulating CT scan)
        volume_clean = mask.copy()
        volume_noisy = add_ct_artifacts(volume_clean, noise_sigma=0.08, streak_prob=0.03)
        
        # Convert to uint8 for storage
        volume_noisy_uint8 = (volume_noisy * 255).astype(np.uint8)
        
        # Save files
        np.save(os.path.join(output_dir, f"root_volume_{i:03d}.npy"), volume_noisy_uint8)
        np.save(os.path.join(output_dir, f"root_mask_{i:03d}.npy"), mask)
        
        print(f"  Saved: root_volume_{i:03d}.npy and root_mask_{i:03d}.npy")
        print(f"  Mask voxels: {mask.sum()}, Shape: {mask.shape}")

# Generate dataset
if __name__ == "__main__":
    print("Generating root structure dataset...")
    generate_dataset(num_samples=10, size=(128, 128, 128), output_dir="data/ct_like/3d")
    print("\nDone! Volumes saved to 'data/ct_like/3d/' directory")
    
    # Visualize one example
    print("\nVisualizing sample with slices...")
    volume = np.load("data/ct_like/3d/val/root_volume_003.npy").astype(np.float32) / 255.0
    print(volume.shape)
    mask = np.load("data/ct_like/3d/val/root_mask_003.npy")
    
    # Select slices to visualize (beginning, middle, end)
    D = volume.shape[0]
    slice_indices = [D//4, D//2, 3*D//4]  # 25%, 50%, 75% through volume
    
    import matplotlib.pyplot as plt
    
    # Create matplotlib figure for slice comparison
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle('Volume Slices: Noisy (left) | Mask (middle) | Overlay (right)', fontsize=14)
    
    for i, slice_idx in enumerate(slice_indices):
        # Noisy volume slice
        axes[i, 0].imshow(volume[slice_idx], cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(f'Noisy - Slice {slice_idx}/{D}')
        axes[i, 0].axis('off')
        
        # Mask slice
        axes[i, 1].imshow(mask[slice_idx], cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Mask - Slice {slice_idx}/{D}')
        axes[i, 1].axis('off')
        
        # Overlay: show mask in red on top of noisy volume
        overlay = np.stack([
            volume[slice_idx] + 0.5 * mask[slice_idx],  # Red channel
            volume[slice_idx],                          # Green channel
            volume[slice_idx]                           # Blue channel
        ], axis=-1)
        overlay = np.clip(overlay, 0, 1)
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f'Overlay - Slice {slice_idx}/{D}')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('slice_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved slice visualization to 'slice_visualization.png'")
    plt.show()
    
    # 3D visualization with PyVista
    print("\nShowing 3D visualization...")
    plotter = pv.Plotter(shape=(1, 2))
    
    # Noisy volume - multiple contours
    plotter.subplot(0, 0)
    grid1 = pv.ImageData()
    grid1.dimensions = np.array(volume.shape)
    grid1.origin = (0, 0, 0)
    grid1.spacing = (1, 1, 1)
    grid1.point_data["values"] = volume.flatten(order="F")
    contour1 = grid1.contour([0.3])
    plotter.add_mesh(contour1, color="tan", opacity=0.7)
    plotter.add_text("Noisy Volume", font_size=10)
    
    # Clean mask
    plotter.subplot(0, 1)
    grid2 = pv.ImageData()
    grid2.dimensions = np.array(mask.shape)
    grid2.origin = (0, 0, 0)
    grid2.spacing = (1, 1, 1)
    grid2.point_data["values"] = mask.flatten(order="F")
    contour2 = grid2.contour([0.5])
    plotter.add_mesh(contour2, color="brown")
    plotter.add_text("Ground Truth Mask", font_size=10)
    
    plotter.show()