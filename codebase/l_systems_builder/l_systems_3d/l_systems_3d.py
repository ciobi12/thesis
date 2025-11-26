import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Dict, Optional
import os


class LSystem3DGenerator:
    """
    Generates 3D L-system structures for slice-based reconstruction.
    Optimized for elongated, root-like structures with high depth-to-width ratio.
    
    Similar to LSystem2DGenerator but extends to 3D with gravity bias for
    vertical root growth suitable for slice-wise processing.
    """
    
    def __init__(
        self,
        axiom: str = "X",
        rules: Dict[str, str] = None,
        angle: float = 25.0,
        forward_length: float = 1.0,
    ):
        """
        Initialize 3D L-system generator.
        
        Args:
            axiom: Starting symbol
            rules: Production rules (e.g., {"X": "F[-X][+X][/X][\\X]FX", "F": "FF"})
            angle: Branching angle in degrees
            forward_length: Length of forward step
        """
        self.axiom = axiom
        self.rules = rules or {
            "X": "F[-X][+X][/X][\\X]FX",
            "F": "FF"
        }
        self.angle = np.radians(angle)
        self.forward_length = forward_length
        
    def generate(self, iterations: int = 3) -> str:
        """Apply L-system production rules iteratively."""
        current = self.axiom
        for _ in range(iterations):
            next_str = ""
            for char in current:
                next_str += self.rules.get(char, char)
            current = next_str
        return current
    
    def _rotate_axis(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """Rotation matrix for arbitrary axis using Rodrigues' formula."""
        axis = axis / np.linalg.norm(axis)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    
    def interpret_to_3d(
        self,
        lstring: str,
        start_pos: np.ndarray = None,
        start_dir: np.ndarray = None,
        gravity_bias: float = 0.6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert L-system string to 3D coordinates.
        
        Commands:
            F: Move forward
            +/-: Yaw left/right (rotate around up vector)
            /\\: Pitch down/up (rotate around right vector)
            &^: Roll left/right (rotate around direction)
            []: Save/restore state
        
        Args:
            lstring: Generated L-system string
            start_pos: Starting position [x, y, z] (default: origin)
            start_dir: Starting direction vector (default: downward -Z)
            gravity_bias: Weight towards downward growth (0-1, higher = more vertical)
            
        Returns:
            points: (N, 3) array of coordinates
            segments: (M, 2) array of segment endpoint indices
        """
        if start_pos is None:
            start_pos = np.array([0.0, 0.0, 0.0])
        if start_dir is None:
            start_dir = np.array([0.0, 0.0, -1.0])  # Grow downward (negative Z)
        
        # Normalize direction
        start_dir = start_dir / np.linalg.norm(start_dir)
        
        # State: position, direction, up-vector
        pos = start_pos.copy()
        direction = start_dir.copy()
        up = np.array([0.0, 1.0, 0.0])  # Initial up vector
        
        # Stack for bracketed segments
        stack = []
        
        points = [pos.copy()]
        segments = []
        point_idx = 0
        
        for char in lstring:
            if char == 'F':
                # Move forward with gravity bias
                gravity_vec = np.array([0.0, 0.0, -1.0])
                biased_dir = (1 - gravity_bias) * direction + gravity_bias * gravity_vec
                biased_dir = biased_dir / (np.linalg.norm(biased_dir) + 1e-8)
                
                new_pos = pos + biased_dir * self.forward_length
                point_idx += 1
                points.append(new_pos.copy())
                segments.append([point_idx - 1, point_idx])
                pos = new_pos
                direction = biased_dir
                
            elif char == '+':
                # Yaw right (rotate around up vector)
                rot = self._rotate_axis(up, self.angle)
                direction = rot @ direction
                
            elif char == '-':
                # Yaw left (rotate around up vector)
                rot = self._rotate_axis(up, -self.angle)
                direction = rot @ direction
                
            elif char == '/':
                # Pitch down (rotate around right vector)
                right = np.cross(direction, up)
                right = right / (np.linalg.norm(right) + 1e-8)
                rot = self._rotate_axis(right, self.angle)
                direction = rot @ direction
                up = rot @ up
                
            elif char == '\\':
                # Pitch up (rotate around right vector)
                right = np.cross(direction, up)
                right = right / (np.linalg.norm(right) + 1e-8)
                rot = self._rotate_axis(right, -self.angle)
                direction = rot @ direction
                up = rot @ up
                
            elif char == '&':
                # Roll left (rotate around direction)
                rot = self._rotate_axis(direction, self.angle)
                up = rot @ up
                
            elif char == '^':
                # Roll right (rotate around direction)
                rot = self._rotate_axis(direction, -self.angle)
                up = rot @ up
                
            elif char == '[':
                # Save state
                stack.append((pos.copy(), direction.copy(), up.copy(), point_idx))
                
            elif char == ']':
                # Restore state
                if stack:
                    pos, direction, up, point_idx = stack.pop()
        
        return np.array(points), np.array(segments)
    
    def _draw_line_3d(
        self,
        volume: np.ndarray,
        p1: np.ndarray,
        p2: np.ndarray,
        thickness: int = 1
    ):
        """Draw thick 3D line in volume using DDA algorithm."""
        D, H, W = volume.shape
        
        # DDA (Digital Differential Analyzer) algorithm
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
        
        for _ in range(steps):
            # Draw sphere at current point for thickness
            ix, iy, iz = int(round(x)), int(round(y)), int(round(z))
            
            for dz_off in range(-thickness, thickness + 1):
                for dy_off in range(-thickness, thickness + 1):
                    for dx_off in range(-thickness, thickness + 1):
                        if dx_off**2 + dy_off**2 + dz_off**2 <= thickness**2:
                            vx = ix + dx_off
                            vy = iy + dy_off
                            vz = iz + dz_off
                            
                            if 0 <= vx < W and 0 <= vy < H and 0 <= vz < D:
                                volume[vz, vy, vx] = 1.0
            
            x += x_inc
            y += y_inc
            z += z_inc
    
    def rasterize_to_volume(
        self,
        points: np.ndarray,
        segments: np.ndarray,
        volume_shape: Tuple[int, int, int] = (16, 16, 64),
        line_thickness: int = 1,
        align_top: bool = True
    ) -> np.ndarray:
        """
        Rasterize 3D line segments to voxel volume.
        
        Args:
            points: (N, 3) coordinate array
            segments: (M, 2) segment indices
            volume_shape: (width, height, depth) in voxels
            line_thickness: Voxel radius for lines
            align_top: Center in XY, align top to Z=0 (roots grow downward)
            
        Returns:
            volume: (depth, height, width) binary voxel array (NumPy convention)
        """
        W, H, D = volume_shape
        volume = np.zeros((D, H, W), dtype=np.float32)
        
        if len(points) == 0:
            return volume
        
        # Normalize coordinates to volume
        points_normalized = points.copy()
        
        if align_top:
            # Center in XY plane
            min_coords = points_normalized.min(axis=0)
            max_coords = points_normalized.max(axis=0)
            
            x_center = (min_coords[0] + max_coords[0]) / 2
            y_center = (min_coords[1] + max_coords[1]) / 2
            
            x_offset = (W - 1) / 2 - x_center
            y_offset = (H - 1) / 2 - y_center
            
            # Align to start at top slice (z=0) and grow downward
            z_offset = -min_coords[2]
            
            points_normalized[:, 0] += x_offset
            points_normalized[:, 1] += y_offset
            points_normalized[:, 2] += z_offset
        
        # Draw segments
        for seg in segments:
            if seg[0] < len(points_normalized) and seg[1] < len(points_normalized):
                p1 = points_normalized[seg[0]]
                p2 = points_normalized[seg[1]]
                self._draw_line_3d(volume, p1, p2, line_thickness)
        
        return volume
    
    def draw_lsystem_ct_style(
        self,
        iterations: int = 4,
        volume_shape: Tuple[int, int, int] = (16, 16, 64),
        skip_segments: bool = True,
        skip_probability: float = 0.15,
        add_noise: bool = True,
        noise_std: float = 0.1,
        ct_background_intensity: int = 80,
        root_intensity_range: Tuple[int, int] = (180, 220),
        line_thickness: int = 1,
        gravity_bias: float = 0.6,
        align_top: bool = True,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate CT-like 3D volume with realistic artifacts.
        
        Mimics `draw_lsystem_ct_style()` from 2D version but for 3D volumes.
        Designed for slice-based reconstruction where agent processes depth-wise.
        
        Args:
            iterations: L-system iterations (3-5 for deep structures)
            volume_shape: (width, height, depth) voxels
            skip_segments: Enable random segment gaps (occlusions)
            skip_probability: Probability of segment occlusion (0.1-0.3 typical)
            add_noise: Add Gaussian noise
            noise_std: Noise standard deviation (0.08-0.15 typical)
            ct_background_intensity: Background gray value (0-255)
            root_intensity_range: (min, max) intensity for root structure
            line_thickness: Voxel thickness of branches (1-2 for thin roots)
            gravity_bias: Downward growth bias (0.5-0.8 for vertical roots)
            align_top: Center structure in XY, grow downward from Z=0
            seed: Random seed for reproducibility
            
        Returns:
            ct_volume: (D, H, W) uint8 volume (CT-like appearance)
            mask_volume: (D, H, W) float32 binary ground truth
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate L-system string
        lstring = self.generate(iterations)
        
        # Convert to 3D coordinates
        points, segments = self.interpret_to_3d(
            lstring,
            gravity_bias=gravity_bias
        )
        
        # Create ground truth mask
        mask_volume = self.rasterize_to_volume(
            points, segments, volume_shape, line_thickness, align_top
        )
        
        # Create CT-like volume (start with background)
        ct_volume = np.ones_like(mask_volume, dtype=np.float32) * ct_background_intensity
        
        # Add root structure with intensity variation
        root_mask = mask_volume > 0
        if root_mask.sum() > 0:
            root_intensities = np.random.randint(
                root_intensity_range[0],
                root_intensity_range[1],
                size=root_mask.sum()
            )
            ct_volume[root_mask] = root_intensities
        
        # Apply segment skipping (occlusions) - similar to 2D version
        if skip_segments and len(segments) > 0:
            # Normalize points for segment masking
            points_normalized = points.copy()
            W, H, D = volume_shape
            
            if align_top:
                min_coords = points.min(axis=0)
                max_coords = points.max(axis=0)
                x_offset = (W - 1) / 2 - (min_coords[0] + max_coords[0]) / 2
                y_offset = (H - 1) / 2 - (min_coords[1] + max_coords[1]) / 2
                z_offset = -min_coords[2]
                
                points_normalized[:, 0] += x_offset
                points_normalized[:, 1] += y_offset
                points_normalized[:, 2] += z_offset
            
            for seg in segments:
                if np.random.rand() < skip_probability:
                    if seg[0] < len(points_normalized) and seg[1] < len(points_normalized):
                        p1 = points_normalized[seg[0]]
                        p2 = points_normalized[seg[1]]
                        
                        # Blank out segment in CT volume
                        temp_vol = np.zeros_like(ct_volume)
                        self._draw_line_3d(temp_vol, p1, p2, line_thickness)
                        ct_volume[temp_vol > 0] = ct_background_intensity
        
        # Add Gaussian noise
        if add_noise:
            noise = np.random.normal(0, noise_std * 255, ct_volume.shape)
            ct_volume = np.clip(ct_volume + noise, 0, 255)
        
        return ct_volume.astype(np.uint8), mask_volume.astype(np.float32)


# Predefined L-system rules (analogous to 2D versions)
PREDEFINED_3D_LSYSTEMS = {
    "taproot": {
        "axiom": "A",
        "rules": {"A": "F[+A][-A]FA", "F": "FF"},
        "angle": 20.0,
        "description": "Main vertical root with small lateral branches"
    },
    "fibrous": {
        "axiom": "X",
        "rules": {"X": "F[-X][+X][/X][\\X]FX", "F": "FF"},
        "angle": 30.0,
        "description": "Dense branching network"
    },
    "adventitious": {
        "axiom": "F",
        "rules": {"F": "FF[+F][-F][/F][\\F]"},
        "angle": 25.0,
        "description": "Moderate spreading branches"
    },
    "deep_root": {
        "axiom": "X",
        "rules": {"X": "F[-X][+X]FX", "F": "FFF"},
        "angle": 15.0,
        "description": "Very elongated structure with sparse branching"
    }
}


def visualize_3d_volume(
    ct_volume: np.ndarray,
    mask_volume: np.ndarray,
    slice_indices: list = None,
    save_path: str = None
):
    """
    Visualize 3D volume as 2D slices (similar to 2D visualize_result).
    
    Args:
        ct_volume: (D, H, W) CT-like volume
        mask_volume: (D, H, W) ground truth mask
        slice_indices: List of Z-indices to display (default: evenly spaced)
        save_path: Optional path to save figure
    """
    D, H, W = ct_volume.shape
    
    if slice_indices is None:
        # Show 3 evenly spaced slices
        slice_indices = [D // 4, D // 2, 3 * D // 4]
    
    n_slices = len(slice_indices)
    fig, axes = plt.subplots(2, n_slices, figsize=(4 * n_slices, 8))
    
    if n_slices == 1:
        axes = axes.reshape(2, 1)
    
    for i, z in enumerate(slice_indices):
        axes[0, i].imshow(ct_volume[z], cmap='gray', vmin=0, vmax=255)
        axes[0, i].set_title(f'CT Slice Z={z}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(mask_volume[z], cmap='gray', vmin=0, vmax=1)
        axes[1, i].set_title(f'Mask Slice Z={z}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_3d_volume_scatter(
    ct_volume: np.ndarray = None,
    mask_volume: np.ndarray = None,
    use_mask: bool = True,
    threshold: int = 128,
    alpha: float = 0.3,
    point_size: int = 20,
    save_path: str = None,
    show_axes_labels: bool = True,
    elev: float = 20,
    azim: float = 45
):
    """
    Visualize 3D volume as scatter plot with X, Y, Z axes.
    
    Args:
        ct_volume: (D, H, W) CT-like volume (optional)
        mask_volume: (D, H, W) ground truth mask (optional)
        use_mask: If True, visualize mask_volume; else ct_volume
        threshold: Intensity threshold for CT volume display
        alpha: Point transparency (0-1)
        point_size: Size of scatter points
        save_path: Optional path to save figure
        show_axes_labels: Show axis labels
        elev: Elevation angle for 3D view
        azim: Azimuth angle for 3D view
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if use_mask and mask_volume is not None:
        volume = mask_volume
        voxel_indices = np.where(volume > 0)
        colors = 'red'
        title = '3D Root Structure (Ground Truth)'
    elif ct_volume is not None:
        volume = ct_volume
        voxel_indices = np.where(volume > threshold)
        colors = ct_volume[voxel_indices] / 255.0  # Normalize for grayscale
        title = f'3D Root Structure (CT Volume, threshold={threshold})'
    else:
        raise ValueError("Must provide either ct_volume or mask_volume")
    
    # Extract Z, Y, X coordinates (NumPy convention)
    z_coords, y_coords, x_coords = voxel_indices
    
    # Plot scatter
    scatter = ax.scatter(
        x_coords, y_coords, z_coords,
        c=colors,
        cmap='gray' if not use_mask else None,
        alpha=alpha,
        s=point_size,
        edgecolors='none'
    )
    
    # Set labels
    if show_axes_labels:
        ax.set_xlabel('X (Width)', fontsize=12)
        ax.set_ylabel('Y (Height)', fontsize=12)
        ax.set_zlabel('Z (Depth)', fontsize=12)
    
    ax.set_title(title, fontsize=14, pad=20)
    
    # Set view angle
    ax.view_init(elev=elev, azim=azim)
    
    # Equal aspect ratio
    D, H, W = volume.shape
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_zlim(0, D)
    
    # Invert Z-axis so depth increases downward
    ax.invert_zaxis()
    
    # Add colorbar for CT volume
    if not use_mask and ct_volume is not None:
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label('CT Intensity', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"3D visualization saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Test generation with improved parameters
    print("=== Testing 3D L-System Generator ===\n")
    
    # Use a more branching rule
    lsys = LSystem3DGenerator(
        axiom="X",
        rules={
            "X": "F[-X][+X][/X][\\X]FX",  # More balanced 4-way branching
            "F": "FF"
        },
        angle=30.0,  # Wider branching angles
        forward_length=1.5  # Longer segments for better visibility
    )
    
    ct_vol, mask_vol = lsys.draw_lsystem_ct_style(
        iterations=3,  # Reduce iterations to avoid overcrowding
        volume_shape=(32, 32, 64),  # Increase XY resolution
        skip_probability=0.1,  # Lower occlusion for visualization
        gravity_bias=0.3,  # Much lower - allow more lateral growth
        line_thickness=2,  # Thicker branches
        seed=123
    )
    
    print(f"Generated volume shape: {ct_vol.shape}")
    print(f"Root voxel occupancy: {mask_vol.sum() / mask_vol.size * 100:.2f}%")
    print(f"Non-zero slices: {(mask_vol.sum(axis=(1,2)) > 0).sum()} / {ct_vol.shape[0]}\n")
    
    # Visualize with better settings
    visualize_3d_volume(
        ct_vol, mask_vol,
        slice_indices=[16, 32, 48],
        save_path='l_systems_builder/l_systems_3d/3d_root_sample.png'
    )
    
    # 3D scatter with adjusted view
    visualize_3d_volume_scatter(
        mask_volume=mask_vol,
        use_mask=True,
        alpha=0.6,
        point_size=50,  # Larger points
        elev=30,  # Better viewing angle
        azim=60,
        save_path='l_systems_builder/l_systems_3d/3d_root_scatter.png'
    )
    
    # Test different archetypes
    print("\n=== Testing Taproot (more vertical) ===")
    lsys_taproot = LSystem3DGenerator(
        axiom="A",
        rules={"A": "F[+A][-A]FA", "F": "FF"},
        angle=20.0,
        forward_length=2.0
    )
    
    ct_tap, mask_tap = lsys_taproot.draw_lsystem_ct_style(
        iterations=4,
        volume_shape=(32, 32, 64),
        gravity_bias=0.5,  # Moderate for taproot
        line_thickness=2,
        seed=456
    )
    
    visualize_3d_volume_scatter(
        mask_volume=mask_tap,
        use_mask=True,
        alpha=0.5,
        point_size=40,
        elev=25,
        azim=45,
        save_path='l_systems_builder/l_systems_3d/3d_taproot_scatter.png'
    )
    
    print(f"Taproot occupancy: {mask_tap.sum() / mask_tap.size * 100:.2f}%")