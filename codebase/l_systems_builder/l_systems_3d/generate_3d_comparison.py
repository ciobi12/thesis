"""
Generate 3D L-system root structures with mask visualizations
Similar to 2D comparison dataset but for 3D volumes
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import os

# -------------------------------
# L-System generation
# -------------------------------
def generate_lsystem(axiom, rules, iterations):
    """Generate an L-system string after a given number of iterations."""
    current = axiom
    for _ in range(iterations):
        next_seq = []
        for ch in current:
            next_seq.append(rules.get(ch, ch))
        current = "".join(next_seq)
    return current

# -------------------------------
# Predefined 3D L-System configurations
# -------------------------------
LSYSTEM_3D_PRESETS = {
    "taproot_simple": {
        "axiom": "F",
        "rules": {"F": "FF[+F][-F]"},
        "angle": 30,
        "description": "Simple taproot with lateral branches"
    },
    "taproot_deep": {
        "axiom": "F",
        "rules": {"F": "FFF[+F][-F][&F]"},
        "angle": 25,
        "description": "Deep taproot with 3D branching"
    },
    "fibrous_dense": {
        "axiom": "F",
        "rules": {"F": "F[+F][&F][-F][^F]F"},
        "angle": 30,
        "description": "Dense fibrous root network"
    },
    "dichotomous_3d": {
        "axiom": "F",
        "rules": {"F": "F[+&F][-^F]"},
        "angle": 25,
        "description": "3D dichotomous branching"
    },
    "tree_bushy": {
        "axiom": "F",
        "rules": {"F": "F[+F][&F]F[-F][^F]F"},
        "angle": 22,
        "description": "Bushy tree with many branches"
    },
    "coral_3d": {
        "axiom": "F",
        "rules": {"F": "F[+F][&F]F[-F][^F]"},
        "angle": 30,
        "description": "3D coral structure"
    },
    "complex_3d_a": {
        "axiom": "F",
        "rules": {"F": "F[+F&F]F[-F^F]F"},
        "angle": 25,
        "description": "Complex 3D pattern A"
    },
    "vine_3d": {
        "axiom": "F",
        "rules": {"F": "F/[+F]F\\[-F]F"},
        "angle": 22,
        "description": "3D vine structure"
    },
    "herringbone_3d": {
        "axiom": "F",
        "rules": {"F": "F[-F]F[&F]F[+F]F"},
        "angle": 30,
        "description": "3D herringbone pattern"
    },
    "adventitious": {
        "axiom": "F[+F][-F]",
        "rules": {"F": "FF[+F][-F][/F]"},
        "angle": 30,
        "description": "Adventitious root system"
    },
}

# -------------------------------
# 3D Turtle Interpreter
# -------------------------------
def draw_lsystem_3d(instructions, angle, step):
    """Interpret L-system instructions in 3D space."""
    pos = np.array([0.0, 0.0, 0.0])
    heading = np.array([0.0, 0.0, 1.0])  # Forward direction along +Z
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
        if cmd == "F":
            new_pos = pos + heading * step
            lines.append((pos.copy(), new_pos.copy()))
            pos = new_pos
        elif cmd == "f":
            pos += heading * step
        elif cmd == "+":
            heading = rotate(heading, up, angle)
            left = rotate(left, up, angle)
        elif cmd == "-":
            heading = rotate(heading, up, -angle)
            left = rotate(left, up, -angle)
        elif cmd == "&":
            heading = rotate(heading, left, angle)
            up = rotate(up, left, angle)
        elif cmd == "^":
            heading = rotate(heading, left, -angle)
            up = rotate(up, left, -angle)
        elif cmd == "\\":
            left = rotate(left, heading, angle)
            up = rotate(up, heading, angle)
        elif cmd == "/":
            left = rotate(left, heading, -angle)
            up = rotate(up, heading, -angle)
        elif cmd == "|":
            heading = -heading
            left = -left
        elif cmd == "[":
            stack.append((pos.copy(), heading.copy(), left.copy(), up.copy()))
        elif cmd == "]":
            pos, heading, left, up = stack.pop()

    return lines

# -------------------------------
# Convert segments to voxel volume
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
    """Convert line segments to a voxel volume."""
    D, H, W = volume_shape
    volume = np.zeros(volume_shape, dtype=np.uint8)
    
    if len(segments) == 0:
        return volume
    
    all_points = []
    for start, end in segments:
        all_points.append(start)
        all_points.append(end)
    all_points = np.array(all_points)
    
    min_vals = all_points.min(axis=0)
    max_vals = all_points.max(axis=0)
    
    margin = 2
    
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
            else:  # Z -> D
                scaled_start[i] = (s - min_vals[i]) / range_val * (D - 1 - 2*margin) + margin
                scaled_end[i] = (e - min_vals[i]) / range_val * (D - 1 - 2*margin) + margin
        
        scaled_segments.append((scaled_start, scaled_end))
    
    for start, end in scaled_segments:
        draw_line_3d_volume(volume, start, end, thickness)
    
    return volume

# -------------------------------
# Generate comparison dataset
# -------------------------------
def generate_3d_comparison_dataset(output_dir="l_systems_3d/examples/comparison_dataset"):
    """Generate 5 different 3D L-system structures with visualizations."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Select 5 diverse patterns
    selected_patterns = [
        'taproot_deep',
        'fibrous_dense',
        'dichotomous_3d',
        'coral_3d',
        'vine_3d'
    ]
    
    iterations = 4
    step = 1.0
    volume_shape = (128, 128, 128)
    
    print(f"Generating 3D comparison dataset for {len(selected_patterns)} patterns...")
    print("=" * 70)
    
    results = []
    
    for idx, pattern_name in enumerate(selected_patterns, 1):
        preset = LSYSTEM_3D_PRESETS[pattern_name]
        
        print(f"\n{idx}. Processing: {pattern_name}")
        print(f"   Description: {preset['description']}")
        
        # Generate L-system string
        lsystem_str = generate_lsystem(preset["axiom"], preset["rules"], iterations)
        
        # Interpret and get line segments
        segments = draw_lsystem_3d(lsystem_str, preset["angle"], step)
        print(f"   Generated {len(segments)} segments")
        
        # Convert to voxel volume
        volume = segments_to_volume(segments, volume_shape=volume_shape, thickness=1)
        
        print(f"   Non-zero voxels: {volume.sum()} ({volume.sum() / volume.size * 100:.2f}% occupancy)")
        
        # Save volume as numpy file
        volume_path = os.path.join(output_dir, f"{idx:02d}_{pattern_name}_volume.npy")
        np.save(volume_path, volume)
        
        results.append({
            "name": pattern_name,
            "preset": preset,
            "segments": segments,
            "volume": volume,
            "idx": idx
        })
        
        # === CREATE INDIVIDUAL VISUALIZATION ===
        fig = plt.figure(figsize=(8, 8))
        
        # 3D structure plot only
        ax = fig.add_subplot(111, projection='3d')
        for start, end in segments:
            xs, ys, zs = zip(start, end)
            ax.plot(xs, ys, zs, color="green", linewidth=0.5)
        ax.set_title(preset['description'], fontsize=14, weight='bold')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        
        plt.tight_layout()
        
        comparison_path = os.path.join(output_dir, f"{idx:02d}_{pattern_name}_comparison.png")
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   ✓ Saved comparison to: {comparison_path}")
    
    print("\n" + "=" * 70)
    print(f"✓ Dataset generation complete!")
    print(f"✓ Location: {output_dir}/")
    print(f"✓ Total files: {len(selected_patterns) * 2} files")
    print(f"   - {len(selected_patterns)} comparison figures")
    print(f"   - {len(selected_patterns)} volume files (.npy)")
    
    return results

# -------------------------------
# Generate combined overview
# -------------------------------
def generate_combined_overview(results, output_dir="l_systems_3d/examples/comparison_dataset"):
    """Generate a single overview figure showing all 5 patterns as 3D structures."""
    
    print("\nGenerating combined overview figure...")
    
    # Create grid: 1 row x 5 columns for 3D structures
    fig = plt.figure(figsize=(25, 5))
    
    for col, res in enumerate(results):
        ax = fig.add_subplot(1, 5, col + 1, projection='3d')
        
        # Plot 3D structure
        for start, end in res["segments"]:
            xs, ys, zs = zip(start, end)
            ax.plot(xs, ys, zs, color="green", linewidth=0.5)
        
        ax.set_title(f"{res['idx']}. {res['preset']['description']}", 
                     fontsize=11, weight='bold')
        ax.set_xlabel("X", fontsize=8)
        ax.set_ylabel("Y", fontsize=8)
        ax.set_zlabel("Z", fontsize=8)
    
    plt.suptitle('3D L-System Root Patterns - 3D Structures', 
                 fontsize=16, weight='bold', y=0.98)
    plt.tight_layout()
    
    overview_path = os.path.join(output_dir, "00_overview_all_patterns.png")
    plt.savefig(overview_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved overview to: {overview_path}")

# -------------------------------
# Main execution
# -------------------------------
if __name__ == "__main__":
    output_dir = "l_systems_3d/examples/comparison_dataset"
    
    # Generate individual comparison images
    results = generate_3d_comparison_dataset(output_dir)
    
    # Generate combined overview
    generate_combined_overview(results, output_dir)
    
    print("\n" + "=" * 70)
    print("✓ All 3D visualizations complete!")
    print("=" * 70)
