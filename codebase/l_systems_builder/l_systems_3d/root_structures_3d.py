import numpy as np
import pyvista as pv
from scipy.ndimage import gaussian_filter, binary_dilation
import os

# Different axioms for structural variety
AXIOMS = [
    "F",                    # Simple trunk
    "X",                    # X-based system (needs X rules)
    "FF",                   # Longer initial trunk
    "F[+F][-F]",           # Pre-branched
    "FX",                   # Mixed system
    "F[X]F",               # Bracketed mixed
    "FFF",                  # Very long trunk
    "F[+X][-X]F",          # Complex start
]

# Expanded rule sets with multiple symbol transformations
RULE_SETS = [
    # Classic balanced branching
    {"F": "F[+F]F[-F]F"},
    
    # X-based systems (good for complex structures)
    {"X": "F[+X][-X]FX", "F": "FF"},
    {"X": "F[-X][X]F[+FX]", "F": "FF"},
    {"X": "F[+X]F[-X]+X", "F": "F"},
    
    # Wider angles and asymmetry
    {"F": "F[++F][--F]F"},
    {"F": "FF[+F][-F]F"},
    {"F": "F[+F]F[-F][F]"},
    {"F": "F[+F][F][-F]F"},
    
    # Dense branching patterns
    {"F": "FF[+F][+F][-F][-F]F"},
    {"F": "F[++F][+F][-F][--F]F"},
    
    # Sparse, long segments
    {"F": "FFF[+F][-F]"},
    {"F": "FF[+FF][-FF]F"},
    
    # Complex mixed systems
    {"X": "F[+X]F[-X]", "F": "FF"},
    {"X": "F[-X][+X]FX", "F": "F[+F]"},
    {"X": "F[++X][--X]F", "F": "FF[+F]"},
    
    # Hierarchical branching
    {"F": "F[+F[-F]]F[--F[+F]]"},
    {"F": "FF[+++F][---F]F"},
    
    # Alternating patterns
    {"X": "F[+F][-X]", "F": "FX"},
    {"X": "F[-F][+X]F", "F": "F"},
    
    # Very dense root mats
    {"F": "F[+F][-F][++F][--F]F"},
    {"X": "F[+X][X][-X][F]", "F": "FF"},
]

def apply_rules(axiom, iterations, rules):
    seq = axiom
    for _ in range(iterations):
        seq = "".join(rules.get(ch, ch) for ch in seq)
    return seq

def generate_root_structure(iterations=3, angle_base=np.pi/6, step=0.1, 
                            angle_variation=0.1, rules_idx=0, axiom_idx=0, seed=None):
    """Generate a root structure using L-systems with variations.
    
    Args:
        iterations: Number of L-system iterations
        angle_base: Base branching angle
        step: Step size for each F move
        angle_variation: Random variation in angles
        rules_idx: Index into RULE_SETS
        axiom_idx: Index into AXIOMS
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
    
    rules = RULE_SETS[rules_idx % len(RULE_SETS)]
    axiom = AXIOMS[axiom_idx % len(AXIOMS)]
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

def generate_dataset(num_samples=5, size = (64, 16, 16), output_dir="generated_roots", min_voxels=100):
    """Generate multiple root volumes with variations."""
    os.makedirs(output_dir, exist_ok=True)
    
    i = 0
    attempts = 0
    max_attempts = num_samples * 5  # Avoid infinite loops
    
    while i < num_samples and attempts < max_attempts:
        attempts += 1
        print(f"Generating sample {i+1}/{num_samples} (attempt {attempts})...")
        
        # Vary parameters for each sample
        iterations = np.random.randint(4, 6)
        angle_base = np.random.uniform(np.pi/8, np.pi/4)
        step = np.random.uniform(0.08, 0.12)
        rules_idx = np.random.randint(0, len(RULE_SETS))  # Random rule set
        axiom_idx = np.random.randint(0, len(AXIOMS))     # Random axiom
        thickness = np.random.randint(2, 4)  # Increased max thickness
        
        print(f"  Using Axiom: {AXIOMS[axiom_idx]}, Rules: {list(RULE_SETS[rules_idx].keys())}")
        
        # Generate root structure
        points, lines = generate_root_structure(
            iterations=iterations,
            angle_base=angle_base,
            step=step,
            angle_variation=0.1,
            rules_idx=rules_idx,
            axiom_idx=axiom_idx,
            seed=42 + attempts
        )
        
        # Convert to volume (this is the clean mask)
        mask = points_to_volume(points, res_d=size[0], res_h=size[1], res_w=size[2], thickness=thickness)
        
        # Check if structure is too sparse
        voxel_count = mask.sum()
        if voxel_count < min_voxels:
            print(f"  ⚠ Skipping: only {voxel_count} voxels (< {min_voxels} threshold). Regenerating...")
            continue
        
        # Check if structure is too sparse
        voxel_count = mask.sum()
        if voxel_count < min_voxels:
            print(f"  ⚠ Skipping: only {voxel_count} voxels (< {min_voxels} threshold). Regenerating...")
            continue
        
        # Create CT-like volume: roots are DARKER than background
        background_intensity = np.random.uniform(0.65, 0.75)  # Vary background per sample
        root_intensity_range = (0.2, 0.4)  # Roots are darker
        volume_clean = create_ct_volume(mask, background_intensity, root_intensity_range)
        
        # Add CT artifacts and noise (with stronger parameters for more challenging dataset)
        volume_noisy = add_ct_artifacts(volume_clean, noise_sigma=0.15, streak_prob=0.15, blur_sigma=0.7)
        
        # Convert to uint8 for storage
        volume_noisy_uint8 = (volume_noisy * 255).astype(np.uint8)
        
        # Save files
        np.save(os.path.join(output_dir, f"root_volume_{i:03d}.npy"), volume_noisy_uint8)
        np.save(os.path.join(output_dir, f"root_mask_{i:03d}.npy"), mask)
        
        print(f"  ✓ Saved: root_volume_{i:03d}.npy and root_mask_{i:03d}.npy")
        print(f"  Mask voxels: {voxel_count}, Shape: {mask.shape}")
        i += 1  # Only increment on successful generation
    
    if attempts >= max_attempts:
        print(f"\n⚠ Warning: Reached max attempts ({max_attempts}). Generated {i}/{num_samples} samples.")

# Generate dataset
if __name__ == "__main__":
    print("Generating root structure dataset...")
    generate_dataset(num_samples=20, size=(128, 128, 128), output_dir="data/ct_like/3d_new", min_voxels=500)
    print("\nDone! Volumes saved to 'data/ct_like/3d_new/' directory")
    
    # Visualize one example
    print("\nVisualizing sample with slices...")
    volume = np.load("data/ct_like/3d_new/root_volume_003.npy").astype(np.float32) / 255.0
    print(volume.shape)
    mask = np.load("data/ct_like/3d_new/root_mask_003.npy")
    
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
    pv.global_theme.allow_empty_mesh = True
    plotter = pv.Plotter(shape=(1, 2))
    
    # Noisy volume - multiple contours
    plotter.subplot(0, 0)
    grid1 = pv.ImageData()
    grid1.dimensions = np.array(volume.shape)
    grid1.origin = (0, 0, 0)
    grid1.spacing = (1, 1, 1)
    grid1.point_data["values"] = volume.flatten(order="F")
    
    # Use adaptive threshold based on volume statistics
    threshold = min(0.3, volume.max() * 0.5)
    contour1 = grid1.contour([threshold])
    if contour1.n_points > 0:
        plotter.add_mesh(contour1, color="tan", opacity=0.7)
        plotter.add_text(f"Noisy Volume (threshold={threshold:.2f})", font_size=10)
    else:
        plotter.add_text("Noisy Volume (empty contour)", font_size=10)
    
    # Clean mask
    plotter.subplot(0, 1)
    grid2 = pv.ImageData()
    grid2.dimensions = np.array(mask.shape)
    grid2.origin = (0, 0, 0)
    grid2.spacing = (1, 1, 1)
    grid2.point_data["values"] = mask.flatten(order="F")
    contour2 = grid2.contour([0.5])
    if contour2.n_points > 0:
        plotter.add_mesh(contour2, color="brown")
        plotter.add_text("Ground Truth Mask", font_size=10)
    else:
        plotter.add_text("Ground Truth Mask (empty)", font_size=10)
    
    try:
        plotter.show()
    except Exception as e:
        print(f"Could not display 3D plot (no display available): {e}")
        print("Visualization completed successfully (slice images saved).")