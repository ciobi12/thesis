from l_systems_builder.l_systems_3d.l_systems_3d import LSystem3DGenerator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

l_system = LSystem3DGenerator(axiom="A",
                       rules={"A": "F[&+B]F[&-B]FA",
                              "B": "F[//+C][//-C]",
                              "C": "F"})
# Build the L-system with specified iterations and parameters
segments = l_system.build_l_sys(3, step = 5, angle_deg = 45)
# Render the generated 3D L-system
l_system.render(elev = 30, azim = 35, save_fig = False, show_fig=True)
volume = l_system.to_voxel_grid(grid_size=(64,64,64), line_thickness=0)

z,x,y = volume.nonzero()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, zdir='z', c= 'red')
plt.ion()
fig.show()