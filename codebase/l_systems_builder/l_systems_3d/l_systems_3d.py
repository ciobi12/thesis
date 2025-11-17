import os
import math
import numpy as np
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



class LSystem3DGenerator:
    def __init__(self, axiom: str, rules: Dict[str, str]):
        self.axiom = axiom
        self.rules = rules
        self.segments = []
        self.iterations = None

    def build_l_sys(self,
                    iterations: int,
                    step: float = 10.0,
                    angle_deg: float = 25.0,
                    start_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> List[Tuple[float, float, float, float, float, float]]:
        
        """Interpret commands into a list of line segments in 3D space [(x0,y0,z0,x1,y1,z1), ...]."""

        self.iterations = iterations

        # Generate string
        s = self.axiom
        for _ in range(self.iterations):
            s = "".join(self.rules.get(ch, ch) for ch in s)

        # Initialize position and orientation vectors
        position = np.array(start_pos, dtype=float)
        
        # Heading (H), Left (L), Up (U) vectors - standard turtle orientation
        H = np.array([1.0, 0.0, 0.0])  # Forward direction
        L = np.array([0.0, 1.0, 0.0])  # Left direction
        U = np.array([0.0, 0.0, 1.0])  # Up direction
        
        stack = []
        angle_rad = math.radians(angle_deg)
     
        for ch in s:
            if ch == "F":  # Draw forward
                new_position = position + step * H
                self.segments.append((*position, *new_position))
                position = new_position
                
            elif ch == "f":  # Move forward without drawing
                position = position + step * H
                
            elif ch == "+":  # Turn right around U axis
                # Rotate H and L around U
                H_new = H * math.cos(angle_rad) + L * math.sin(angle_rad)
                L_new = -H * math.sin(angle_rad) + L * math.cos(angle_rad)
                H, L = H_new, L_new
                
            elif ch == "-":  # Turn left around U axis
                # Rotate H and L around U (opposite direction)
                H_new = H * math.cos(-angle_rad) + L * math.sin(-angle_rad)
                L_new = -H * math.sin(-angle_rad) + L * math.cos(-angle_rad)
                H, L = H_new, L_new
                
            elif ch == "&":  # Pitch down around L axis
                # Rotate H and U around L
                H_new = H * math.cos(angle_rad) + U * math.sin(angle_rad)
                U_new = -H * math.sin(angle_rad) + U * math.cos(angle_rad)
                H, U = H_new, U_new
                
            elif ch == "^":  # Pitch up around L axis
                # Rotate H and U around L (opposite direction)
                H_new = H * math.cos(-angle_rad) + U * math.sin(-angle_rad)
                U_new = -H * math.sin(-angle_rad) + U * math.cos(-angle_rad)
                H, U = H_new, U_new
                
            elif ch == "\\":  # Roll left around H axis
                # Rotate L and U around H
                L_new = L * math.cos(angle_rad) + U * math.sin(angle_rad)
                U_new = -L * math.sin(angle_rad) + U * math.cos(angle_rad)
                L, U = L_new, U_new
                
            elif ch == "/":  # Roll right around H axis
                # Rotate L and U around H (opposite direction)
                L_new = L * math.cos(-angle_rad) + U * math.sin(-angle_rad)
                U_new = -L * math.sin(-angle_rad) + U * math.cos(-angle_rad)
                L, U = L_new, U_new
                
            elif ch == "|":  # Turn around 180 degrees
                H = -H
                L = -L
                
            elif ch == "[":
                stack.append((position.copy(), H.copy(), L.copy(), U.copy()))
                
            elif ch == "]":
                position, H, L, U = stack.pop()
                
        return self.segments

    def render(self, elev=30, azim=45, show_axes=True, show_fig = False, save_fig=False, filename = None):
        """Render the 3D L-system in an interactive matplotlib window."""
        if not self.segments:
            print("No segments to render. Run build_l_sys first.")
            return
            
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        for segment in self.segments:
            x0, y0, z0, x1, y1, z1 = segment
            ax.plot([x0, x1], [y0, y1], [z0, z1], color='green', linewidth=1)
        
        # Set equal aspect ratio for all axes
        all_points = np.array([(x0, y0, z0) for x0, y0, z0, _, _, _ in self.segments] + 
                              [(x1, y1, z1) for _, _, _, x1, y1, z1 in self.segments])
        
        if len(all_points) > 0:
            max_range = np.array([all_points[:, 0].max() - all_points[:, 0].min(),
                                 all_points[:, 1].max() - all_points[:, 1].min(),
                                 all_points[:, 2].max() - all_points[:, 2].min()]).max() / 2.0
            
            mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
            mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
            mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_title('3D L-System (Interactive - Click and drag to rotate)')
        ax.view_init(elev=elev, azim=azim)

        if not show_axes:
            ax.set_axis_off()
        
        # Enable interactive mode
        if show_fig:
            plt.ion()
            plt.show()

        if save_fig:
            plt.savefig(os.path.join("examples", "images", filename + '.png'), dpi=300)
        
        return fig, ax
     
    def to_voxel_grid(self, grid_size: Tuple[int, int, int] = (64, 64, 64), 
                      align_bottom: bool = True, save_voxel_grid = False, filepath = None) -> np.ndarray:
        """
        Convert segments to a 3D voxel grid for 3D CNN processing.
        Lines are exactly 1 voxel thick.
        
        Args:
            grid_size: (width, height, depth) of the voxel grid
            align_bottom: If True, align the lowest point to the bottom slice (z=0)
            save_voxel_grid: If True, save the voxel grid to a .npy file
            filepath: Path to save the .npy file if save_voxel_grid is True
            
        Returns:
            Binary 3D numpy array of shape (D, W, H) where 1=structure, 0=empty
            D is the depth dimension (slices), where D=0 is the bottom
        """
        if not self.segments:
            return np.zeros(grid_size, dtype=np.uint8)
            
        # Get bounding box
        all_points = []
        for x0, y0, z0, x1, y1, z1 in self.segments:
            all_points.extend([(x0, y0, z0), (x1, y1, z1)])
        all_points = np.array(all_points)
            
        min_coords = all_points.min(axis=0)
        max_coords = all_points.max(axis=0)
        ranges = max_coords - min_coords
        ranges = np.where(ranges == 0, 1, ranges)  # Avoid division by zero
            
        # Initialize voxel grid
        voxel_grid = np.zeros(grid_size, dtype=np.uint8)
        W, H, D = grid_size
            
        # Add padding (but not on bottom if align_bottom=True)
        if align_bottom:
            # No padding at bottom, small padding at top for Z
            padding_xy = 0.1
            padding_z_bottom = 0.0  # No padding at bottom
            padding_z_top = 0.05    # Small padding at top
            
            scale_xy = np.array([W, H]) * (1 - 2 * padding_xy) / ranges[:2]
            scale_z = D * (1 - padding_z_bottom - padding_z_top) / ranges[2]
            scale = np.array([scale_xy[0], scale_xy[1], scale_z])
            
            # Shift Z so minimum is at the bottom (z=0 after padding)
            offset = np.array([
                padding_xy * W,
                padding_xy * H,
                padding_z_bottom * D  # Start from bottom
            ])
        else:
            # Centered padding
            padding = 0.1
            scale = np.array([W, H, D]) * (1 - 2 * padding) / ranges
            offset = np.array([padding * W, padding * H, padding * D])
            
        def to_voxel(x, y, z):
            """Convert world coordinates to voxel indices"""
            vx = int((x - min_coords[0]) * scale[0] + offset[0])
            vy = int((y - min_coords[1]) * scale[1] + offset[1])
            vz = int((z - min_coords[2]) * scale[2] + offset[2])
            return np.clip(vx, 0, W-1), np.clip(vy, 0, H-1), np.clip(vz, 0, D-1)
            
        # Rasterize line segments using Bresenham 3D
        for x0, y0, z0, x1, y1, z1 in self.segments:
            vx0, vy0, vz0 = to_voxel(x0, y0, z0)
            vx1, vy1, vz1 = to_voxel(x1, y1, z1)
                
            # Bresenham line drawing in 3D (single voxel thickness)
            points = self._bresenham_3d(vx0, vy0, vz0, vx1, vy1, vz1)
            for px, py, pz in points:
                voxel_grid[px, py, pz] = 1
        
        z, x, y = voxel_grid.nonzero()
        voxel_grid = voxel_grid[list(np.unique(z)), :, :]
        if save_voxel_grid:
            np.save(filepath, voxel_grid)
            print(f"Saved voxel grid with shape {voxel_grid.shape}")
            print(f"Occupied voxels: {voxel_grid.sum()} / {voxel_grid.size}")
            print(f"Bottom slice (z=0) occupied voxels: {voxel_grid[:, :, 0].sum()}")
        
        return voxel_grid
        
    def _bresenham_3d(self, x0, y0, z0, x1, y1, z1):
        """3D Bresenham line algorithm"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        dz = abs(z1 - z0)
            
        xs = 1 if x1 > x0 else -1
        ys = 1 if y1 > y0 else -1
        zs = 1 if z1 > z0 else -1
            
        # Driving axis is X
        if dx >= dy and dx >= dz:
            p1 = 2 * dy - dx
            p2 = 2 * dz - dx
            while x0 != x1:
                points.append((x0, y0, z0))
                x0 += xs
                if p1 >= 0:
                    y0 += ys
                    p1 -= 2 * dx
                if p2 >= 0:
                    z0 += zs
                    p2 -= 2 * dx
                p1 += 2 * dy
                p2 += 2 * dz

        # Driving axis is Y
        elif dy >= dx and dy >= dz:
            p1 = 2 * dx - dy
            p2 = 2 * dz - dy
            while y0 != y1:
                points.append((x0, y0, z0))
                y0 += ys
                if p1 >= 0:
                    x0 += xs
                    p1 -= 2 * dy
                if p2 >= 0:
                    z0 += zs
                    p2 -= 2 * dy
                p1 += 2 * dx
                p2 += 2 * dz

        # Driving axis is Z
        else:
            p1 = 2 * dy - dz
            p2 = 2 * dx - dz
            while z0 != z1:
                points.append((x0, y0, z0))
                z0 += zs
                if p1 >= 0:
                    y0 += ys
                    p1 -= 2 * dz
                if p2 >= 0:
                    x0 += xs
                    p2 -= 2 * dz
                p1 += 2 * dy
                p2 += 2 * dx
            
        points.append((x1, y1, z1))
        return points
    

