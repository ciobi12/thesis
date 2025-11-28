import numpy as np
import pyvista as pv

# Define L-system rules
rules = {
    "F": "F[+F]F[-F]F"   # Forward + branch left/right
}

def apply_rules(axiom, iterations):
    seq = axiom
    for _ in range(iterations):
        seq = "".join(rules.get(ch, ch) for ch in seq)
    return seq

# Generate sequence
axiom = "F"
sequence = apply_rules(axiom, 4)

# Parameters
step = 0.1
angle = np.pi / 6  # 30 degrees

# Stack for branching
points = [[0,0,0]]
lines = []
stack = []
direction = np.array([0,0,-1])  # downward root growth

for ch in sequence:
    if ch == "F":
        new_point = points[-1] + step * direction
        points.append(new_point.tolist())
        lines.append([2, len(points)-2, len(points)-1])
    elif ch == "+":
        # rotate randomly around x-axis
        rot = np.array([[1,0,0],
                        [0,np.cos(angle),-np.sin(angle)],
                        [0,np.sin(angle), np.cos(angle)]])
        direction = rot @ direction
    elif ch == "-":
        # rotate randomly around y-axis
        rot = np.array([[np.cos(angle),0,np.sin(angle)],
                        [0,1,0],
                        [-np.sin(angle),0,np.cos(angle)]])
        direction = rot @ direction
    elif ch == "[":
        stack.append((points[-1], direction.copy()))
    elif ch == "]":
        pos, direction = stack.pop()
        points.append(pos)
        
# Convert to PyVista PolyData
points = np.array(points)
root_mesh = pv.PolyData(points)
root_mesh.lines = np.hstack(lines)

plotter = pv.Plotter()
plotter.add_mesh(root_mesh, color="brown", line_width=2)
# plotter.show()

# Assume points is your Nx3 array of root coordinates
points = np.array(points)

# Choose resolution with more depth along z
res_x, res_y, res_z = 16, 16, 64   # elongated z-axis

volume = np.zeros((res_x, res_y, res_z), dtype=np.uint8)

# Normalize each axis separately
min_vals = points.min(axis=0)
max_vals = points.max(axis=0)

scaled_points = np.zeros_like(points)
scaled_points[:,0] = (points[:,0] - min_vals[0]) / (max_vals[0]-min_vals[0]+1e-8) * (res_x-1)
scaled_points[:,1] = (points[:,1] - min_vals[1]) / (max_vals[1]-min_vals[1]+1e-8) * (res_y-1)
scaled_points[:,2] = (points[:,2] - min_vals[2]) / (max_vals[2]-min_vals[2]+1e-8) * (res_z-1)

scaled_points = scaled_points.astype(int)

# Fill voxels
for p in scaled_points:
    x,y,z = p
    volume[x,y,z] = 1

# Save elongated volume
np.save("root_volume_longZ.npy", volume)

# Load your voxel volume
volume = np.load("root_volume_longZ.npy")

# Create ImageData grid
grid = pv.ImageData()
grid.dimensions = np.array(volume.shape)  
grid.origin = (0, 0, 0)
grid.spacing = (1, 1, 1)

# Attach voxel values as point data (not cell data!)
grid.point_data["values"] = volume.flatten(order="F")

# Extract isosurface at threshold 0.5
contour = grid.contour([0.5])

# Plot
plotter = pv.Plotter()
plotter.add_mesh(contour, color="brown")
plotter.show()

