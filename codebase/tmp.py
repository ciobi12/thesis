from l_systems_builder.l_systems_3d.root_structures_3d import RULE_SETS, generate_root_structure, points_to_volume
import pyvista as pv 
import numpy as np

points, _ = generate_root_structure()
volume = points_to_volume(points, res_d=128, res_h=128, res_w=128, thickness=2)

plotter = pv.Plotter()
plotter.subplot(0, 0)
grid1 = pv.ImageData()
grid1.dimensions = np.array(volume.shape)
grid1.origin = (0, 0, 0)
grid1.spacing = (1, 1, 1)
grid1.point_data["values"] = volume.flatten(order="F")
contour1 = grid1.contour([0.3])
plotter.add_mesh(contour1, color="tan", opacity=0.7)
plotter.add_text("Noisy Volume", font_size=10)